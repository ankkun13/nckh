"""
Data Preprocessor for Multi-Omics Cancer Data
==============================================
Handles:
  - NA removal / imputation
  - Z-score normalization (or MinMax)
  - Gene filtering by mutation frequency
  - Cross-omics gene/sample alignment
  - Gene label integration (CGC driver vs non-driver)
  - Train/test data leakage prevention
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class DataPreprocessor:
    """
    Preprocesses raw multi-omics data for the GNN pipeline.
    Replicates and enhances the filtering logic from the original
    MODCAN utils.filter_data() function.
    """
    
    def __init__(self, config: dict):
        self.normalization = config['preprocessing']['normalization']
        self.na_strategy = config['preprocessing']['na_strategy']
        self.mut_freq_min = config['preprocessing']['mutation_freq_min']
        self.mut_freq_max_ratio = config['preprocessing']['mutation_freq_max_ratio']
    
    def process(self, data: dict) -> dict:
        """
        Full preprocessing pipeline.
        
        Args:
            data: Raw data dictionary from MultiOmicsDataLoader.
        
        Returns:
            Processed data dictionary with aligned and normalized matrices.
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Find common genes across PPI and omics data
        data = self._align_genes_with_ppi(data)
        
        # Step 2: Handle missing values
        data = self._handle_na(data)
        
        # Step 3: Filter samples by mutation frequency
        data = self._filter_by_mutation_frequency(data)
        
        # Step 4: Normalize omics data
        data = self._normalize(data)
        
        # Step 5: Create gene labels (driver/non-driver)
        data = self._create_gene_labels(data)
        
        # Step 6: Build gene feature matrix
        data = self._build_gene_features(data)
        
        logger.info("Preprocessing completed.")
        self._log_processed_summary(data)
        
        return data
    
    def _align_genes_with_ppi(self, data: dict) -> dict:
        """
        Filter genes to keep only those present in the PPI network.
        Aligns all omics matrices to the same gene set.
        Replicates logic from MODCAN utils.filter_data().
        """
        logger.info("Aligning genes with PPI network...")
        
        ppi = data['ppi']
        ppi_genes = set(np.unique(np.concatenate([ppi[:, 0], ppi[:, 1]])))
        
        # Use expression genes as the reference gene set
        exp_genes = set(data['exp_genes'])
        common_genes = sorted(exp_genes.intersection(ppi_genes))
        
        logger.info("PPI genes: %d, Expression genes: %d, Common genes: %d",
                     len(ppi_genes), len(exp_genes), len(common_genes))
        
        # Build index mapping for expression genes
        exp_gene_list = list(data['exp_genes'])
        common_idx = [exp_gene_list.index(g) for g in common_genes if g in exp_gene_list]
        
        # Filter expression data
        data['exp_data'] = data['exp_data'][common_idx, :]
        data['exp_genes'] = np.array(common_genes)
        
        # Filter methylation data (same gene order)
        if len(data['met_data']) > 0:
            met_gene_list = list(data['met_genes'])
            met_idx = [met_gene_list.index(g) for g in common_genes if g in met_gene_list]
            data['met_data'] = data['met_data'][met_idx, :]
            data['met_genes'] = np.array([data['met_genes'][i] for i in met_idx])
        
        # Filter normal data
        if len(data['exp_normal']) > 0:
            normal_gene_list = list(data['normal_exp_genes'])
            normal_idx = [normal_gene_list.index(g) for g in common_genes
                          if g in normal_gene_list]
            data['exp_normal'] = data['exp_normal'][normal_idx, :]
            data['normal_exp_genes'] = np.array([data['normal_exp_genes'][i] for i in normal_idx])
        
        if len(data['met_normal']) > 0:
            normal_met_list = list(data['normal_met_genes'])
            normal_met_idx = [normal_met_list.index(g) for g in common_genes
                              if g in normal_met_list]
            data['met_normal'] = data['met_normal'][normal_met_idx, :]
            data['normal_met_genes'] = np.array([data['normal_met_genes'][i] for i in normal_met_idx])
        
        # Filter mutation data
        if len(data['mut_data']) > 0:
            mut_gene_list = list(data['mut_genes'])
            mut_idx = [mut_gene_list.index(g) for g in common_genes if g in mut_gene_list]
            data['mut_data'] = data['mut_data'][mut_idx, :]
            data['mut_genes'] = np.array([data['mut_genes'][i] for i in mut_idx])
        
        # Store common gene list for reference
        data['gene_names'] = np.array(common_genes)
        
        log_tensor_info(logger, "Aligned exp_data", data['exp_data'])
        return data
    
    def _handle_na(self, data: dict) -> dict:
        """Handle missing values based on configured strategy."""
        logger.info("Handling missing values (strategy: %s)...", self.na_strategy)
        
        omics_keys = ['exp_data', 'met_data', 'exp_normal', 'met_normal']
        
        for key in omics_keys:
            if key in data and isinstance(data[key], np.ndarray) and data[key].size > 0:
                na_count = np.isnan(data[key]).sum()
                if na_count > 0:
                    logger.info("  %s: %d NaN values found", key, na_count)
                    
                    if self.na_strategy == 'drop':
                        # Replace NaN with 0 (gene-level, safe for omics)
                        data[key] = np.nan_to_num(data[key], nan=0.0)
                    elif self.na_strategy == 'impute_median':
                        col_medians = np.nanmedian(data[key], axis=0)
                        nan_mask = np.isnan(data[key])
                        for col_idx in range(data[key].shape[1]):
                            data[key][nan_mask[:, col_idx], col_idx] = col_medians[col_idx]
                    elif self.na_strategy == 'impute_zero':
                        data[key] = np.nan_to_num(data[key], nan=0.0)
                else:
                    logger.info("  %s: no NaN values", key)
        
        return data
    
    def _filter_by_mutation_frequency(self, data: dict) -> dict:
        """
        Filter samples by mutation frequency.
        Replicates MODCAN's filter logic:
          - Keep samples where mutation count is between
            [mut_freq_min, gene_count * mut_freq_max_ratio]
        """
        logger.info("Filtering by mutation frequency...")
        
        if data['mut_data'].size == 0:
            logger.warning("No mutation data available, skipping mutation filter")
            return data
        
        mut_matrix = data['mut_data']
        n_genes = mut_matrix.shape[0]
        
        # Column (sample) sums
        col_sum = np.sum(mut_matrix, axis=0)
        col_mask = (col_sum >= self.mut_freq_min) & \
                   (col_sum <= n_genes * self.mut_freq_max_ratio)
        
        n_kept = col_mask.sum()
        logger.info("Mutation filter: %d/%d samples kept (min=%d, max_ratio=%.2f)",
                     n_kept, len(col_mask), self.mut_freq_min, self.mut_freq_max_ratio)
        
        # Apply filter to all sample-aligned matrices
        if n_kept > 0 and n_kept < len(col_mask):
            data['mut_data'] = data['mut_data'][:, col_mask]
            data['mut_samples'] = data['mut_samples'][col_mask]
            
            # Also filter exp and met data to matching samples
            # Find common samples between mutation-filtered and exp
            mut_samples_set = set(data['mut_samples'])
            exp_sample_mask = np.array([s in mut_samples_set for s in data['exp_samples']])
            
            if exp_sample_mask.sum() > 0:
                data['exp_data'] = data['exp_data'][:, exp_sample_mask]
                data['exp_samples'] = data['exp_samples'][exp_sample_mask]
            
            met_sample_mask = np.array([s in mut_samples_set for s in data['met_samples']])
            if met_sample_mask.sum() > 0:
                data['met_data'] = data['met_data'][:, met_sample_mask]
                data['met_samples'] = data['met_samples'][met_sample_mask]
        
        # Build co-mutation matrix for edge info (from MODCAN select_edge_info)
        data['co_mut_matrix'] = self._select_edge_info(
            (data['mut_data'] > 0).astype(np.float32)
        )
        log_tensor_info(logger, "co_mut_matrix", data['co_mut_matrix'])
        
        return data
    
    def _select_edge_info(self, mut_binary: np.ndarray) -> np.ndarray:
        """
        Select samples with sufficient co-mutation overlap.
        Replicates MODCAN utils.select_edge_info().
        """
        co_mut = np.dot(mut_binary.T, mut_binary)
        sum_co_mut = np.sum(co_mut, axis=1)
        mut_diag = np.diagonal(co_mut)
        
        # Avoid division by zero
        mut_diag_safe = np.where(mut_diag == 0, 1, mut_diag)
        overlap_time = sum_co_mut / mut_diag_safe
        
        sample_idx = np.where(overlap_time >= 0.05 * mut_binary.shape[1])[0]
        return mut_binary[:, sample_idx]
    
    def _normalize(self, data: dict) -> dict:
        """Normalize omics data using Z-score or MinMax."""
        logger.info("Normalizing data (method: %s)...", self.normalization)
        
        omics_keys = ['exp_data', 'met_data']
        
        for key in omics_keys:
            if key in data and data[key].size > 0:
                if self.normalization == 'zscore':
                    scaler = StandardScaler()
                    # Normalize per gene (axis=0: each gene across samples)
                    data[key] = scaler.fit_transform(data[key].T).T.astype(np.float32)
                elif self.normalization == 'minmax':
                    scaler = MinMaxScaler()
                    data[key] = scaler.fit_transform(data[key].T).T.astype(np.float32)
                    data[key] = np.clip(data[key], 1e-4, 1 - 1e-4)
                
                log_tensor_info(logger, f"Normalized {key}", data[key])
        
        return data
    
    def _create_gene_labels(self, data: dict) -> dict:
        """
        Create binary gene labels based on CGC driver genes.
        Replicates MODCAN MgcnPre.get_gene_label().
        
        Output shape: (n_genes, 2) where:
            - Column 0: 1 if driver gene, 0 otherwise
            - Column 1: 1 if gene is labeled (driver OR non-driver), 0 if unlabeled
        """
        logger.info("Creating gene labels...")
        
        gene_names = data['gene_names']
        driver_genes = set(data['driver_genes'])
        non_driver_genes = set(data['non_driver_genes'])
        
        driver_label = np.array([1 if g in driver_genes else 0 for g in gene_names])
        non_driver_label = np.array([1 if g in non_driver_genes else 0 for g in gene_names])
        
        # A gene is "labeled" if it's either a known driver or known non-driver
        all_labeled = driver_label + non_driver_label
        
        gene_label = np.column_stack([driver_label, all_labeled])
        
        n_driver = driver_label.sum()
        n_non_driver = non_driver_label.sum()
        n_unlabeled = len(gene_names) - (n_driver + n_non_driver)
        
        logger.info("Gene labels: %d drivers, %d non-drivers, %d unlabeled (total: %d)",
                     n_driver, n_non_driver, n_unlabeled, len(gene_names))
        
        data['gene_labels'] = gene_label
        return data
    
    def _build_gene_features(self, data: dict) -> dict:
        """
        Build initial gene feature matrix.
        Combines multi-omics features per gene (placeholder for clustering-based features).
        This will be enhanced in Phase 3 with cluster-specific features.
        """
        logger.info("Building initial gene feature matrix...")
        
        n_genes = len(data['gene_names'])
        features = []
        
        # SNV ratio (global)
        if data['snv_data'].size > 0:
            snv_genes_set = {g: i for i, g in enumerate(data['snv_genes'])}
            snv_ratio = np.zeros(n_genes, dtype=np.float32)
            n_samples = data['snv_data'].shape[1] if data['snv_data'].ndim > 1 else 1
            for i, gene in enumerate(data['gene_names']):
                if gene in snv_genes_set:
                    idx = snv_genes_set[gene]
                    snv_ratio[i] = data['snv_data'][idx].sum() / max(n_samples, 1)
            features.append(snv_ratio)
        
        # CNV ratio (global)
        if data['cnv_data'].size > 0:
            cnv_genes_set = {g: i for i, g in enumerate(data['cnv_genes'])}
            cnv_ratio = np.zeros(n_genes, dtype=np.float32)
            n_samples = data['cnv_data'].shape[1] if data['cnv_data'].ndim > 1 else 1
            for i, gene in enumerate(data['gene_names']):
                if gene in cnv_genes_set:
                    idx = cnv_genes_set[gene]
                    cnv_ratio[i] = np.abs(data['cnv_data'][idx]).sum() / max(n_samples, 1)
            features.append(cnv_ratio)
        
        # Expression diff (tumor - normal mean)
        if data['exp_data'].size > 0 and data['exp_normal'].size > 0:
            exp_mean_tumor = np.mean(data['exp_data'], axis=1)
            exp_mean_normal = np.mean(data['exp_normal'], axis=1)
            # Align lengths
            min_len = min(len(exp_mean_tumor), len(exp_mean_normal))
            exp_diff = exp_mean_tumor[:min_len] - exp_mean_normal[:min_len]
            if len(exp_diff) < n_genes:
                exp_diff = np.pad(exp_diff, (0, n_genes - len(exp_diff)))
            features.append(exp_diff[:n_genes].astype(np.float32))
        
        # Methylation diff (tumor - normal mean)
        if data['met_data'].size > 0 and data['met_normal'].size > 0:
            met_mean_tumor = np.mean(data['met_data'], axis=1)
            met_mean_normal = np.mean(data['met_normal'], axis=1)
            min_len = min(len(met_mean_tumor), len(met_mean_normal))
            met_diff = met_mean_tumor[:min_len] - met_mean_normal[:min_len]
            if len(met_diff) < n_genes:
                met_diff = np.pad(met_diff, (0, n_genes - len(met_diff)))
            features.append(met_diff[:n_genes].astype(np.float32))
        
        if features:
            gene_feature_matrix = np.column_stack(features)
        else:
            gene_feature_matrix = np.zeros((n_genes, 1), dtype=np.float32)
        
        # Scale features (MinMax per feature)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        gene_feature_matrix = scaler.fit_transform(gene_feature_matrix)
        gene_feature_matrix = np.clip(gene_feature_matrix, 1e-4, 1 - 1e-4).astype(np.float32)
        
        data['gene_features'] = gene_feature_matrix
        log_tensor_info(logger, "gene_features", gene_feature_matrix)
        
        return data
    
    def _log_processed_summary(self, data: dict):
        """Log preprocessing results summary."""
        logger.info("")
        logger.info("─" * 50)
        logger.info("PREPROCESSING SUMMARY for %s", data['cancer_type'])
        logger.info("─" * 50)
        logger.info("  Aligned genes: %d", len(data['gene_names']))
        logger.info("  Gene features shape: %s", data['gene_features'].shape)
        logger.info("  Gene labels shape: %s", data['gene_labels'].shape)
        logger.info("  Driver genes found: %d", data['gene_labels'][:, 0].sum())
        if data['exp_data'].size > 0:
            logger.info("  Expression matrix: %s", data['exp_data'].shape)
        if data['met_data'].size > 0:
            logger.info("  Methylation matrix: %s", data['met_data'].shape)
        logger.info("─" * 50)
