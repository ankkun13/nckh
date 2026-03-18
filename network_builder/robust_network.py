"""
Robust Co-Association Network Builder — CORE IMPROVEMENT
========================================================
Replaces MODCAN's Pearson correlation (utils.calculate_co_association)
with Random Forest Feature Importance or Mutual Information.

CRITICAL: Only uses training set data to avoid data leakage into
the graph structure. All edge weight computations exclude test data.

Key changes from MODCAN:
  - np.corrcoef() → RandomForest / MutualInformation
  - Static threshold → Dynamic percentile-based threshold
  - Adjacency matrix saved as .pt (PyTorch) for direct GNN ingestion
"""

import logging
import gc
import time
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed

import torch



logger = logging.getLogger("pipeline")


class RobustCoAssociationNetwork:
    """
    Builds differential co-association network using robust methods.
    
    Pipeline:
    1. For each patient subgroup (cluster):
       a. Compute gene-gene importance matrix (RF or MI) using TRAIN data only
       b. Apply dynamic threshold to create sparse edge set
    2. Compute differential: |importance_tumor - importance_normal|
    3. Multiply element-wise with PPI adjacency for biological constraint
    4. Build Chebyshev polynomial expansion for GNN input
    """
    
    def __init__(self, config: dict):
        self.method = config['network']['method']
        self.threshold_percentile = config['network']['threshold_percentile']
        self.net_split = config['network']['net_split']
        self.ppi_weight_type = config['network']['ppi_weight_type']
        self.n_estimators = config['network']['n_estimators']
        self.n_jobs = config['network']['n_jobs']
        self.top_k_genes = config['network'].get('top_k_genes', 2000)
        self.max_features_mi = config['network'].get('max_features_mi', 500)
    
    def build(self, data: dict) -> dict:
        """
        Build the complete graph data for GNN input.
        
        Args:
            data: Preprocessed data dictionary with cluster_labels.
        
        Returns:
            Dictionary with graph adjacency data ready for GNN.
        """
        logger.info("Building robust co-association network (method: %s)...",
                     self.method)
        
        gene_names = data['gene_names']
        cluster_labels = data['cluster_labels']
        
        # Step 1: Compute differential co-association for EXP
        logger.info("Computing differential co-association for Expression data...")
        diff_co_exp = self._compute_differential_coassociation(
            tumor_data=data['exp_data'],
            normal_data=data['exp_normal'],
            gene_names=gene_names,
            cluster_labels=cluster_labels
        )
        
        # Step 2: Compute differential co-association for MET
        logger.info("Computing differential co-association for Methylation data...")
        diff_co_met = self._compute_differential_coassociation(
            tumor_data=data['met_data'],
            normal_data=data['met_normal'],
            gene_names=gene_names,
            cluster_labels=cluster_labels
        )
        
        # Step 3: Combine EXP and MET (weighted)
        diff_co_combined = []
        for exp_diff, met_diff in zip(diff_co_exp, diff_co_met):
            combined = 0.7 * exp_diff + 0.3 * met_diff
            diff_co_combined.append(combined)
        
        # Step 4: Build PPI adjacency matrix (sparse)
        logger.info("Building PPI adjacency matrix...")
        ppi_network = self._build_ppi_adjacency(
            data['ppi'], gene_names, self.ppi_weight_type
        )
        
        # Step 5: Multiply with PPI constraint (sparse)
        logger.info("Applying PPI biological constraint...")
        co_data_net = self._apply_ppi_constraint(diff_co_combined, ppi_network)
        
        # Log edge counts per slice
        for s, adj_s in enumerate(co_data_net):
            logger.info("  Slice %d: %d edges", s, adj_s.nnz)
        
        # Cleanup
        del diff_co_exp, diff_co_met, diff_co_combined, ppi_network
        gc.collect()
        
        graph_data = {
            'co_data_net': co_data_net,
            'n_slices': len(co_data_net)
        }
        
        return graph_data
    
    def _compute_differential_coassociation(
        self, tumor_data: np.ndarray, normal_data: np.ndarray,
        gene_names: np.ndarray, cluster_labels: np.ndarray
    ) -> list:
        """
        Compute differential co-association per cluster.
        REPLACES MODCAN's np.corrcoef with RF/MI.
        
        Args:
            tumor_data: Gene x Sample matrix (tumor)
            normal_data: Gene x Sample matrix (normal)
            gene_names: Array of gene names
            cluster_labels: Cluster assignment per sample (1-indexed)
        
        Returns:
            List of differential co-association matrices (one per cluster)
        """
        n_genes = tumor_data.shape[0]
        unique_clusters = np.unique(cluster_labels)
        
        # Compute importance matrix for normal data (sparse)
        logger.info("  Computing %s importance for normal tissue...", self.method)
        importance_normal = self._compute_importance_matrix(normal_data)
        
        diff_co_data = []
        for cluster_id in unique_clusters:
            logger.info("  Computing %s importance for cluster %d...",
                         self.method, cluster_id)
            
            # Get tumor samples in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            
            # Ensure mask length matches data columns
            effective_mask = cluster_mask[:tumor_data.shape[1]]
            if effective_mask.sum() == 0:
                diff_co_data.append(
                    sp.csr_matrix((n_genes, n_genes), dtype=np.float32)
                )
                continue
            
            tumor_cluster = tumor_data[:, effective_mask]
            
            # Compute importance matrix for this cluster (sparse)
            importance_cluster = self._compute_importance_matrix(tumor_cluster)
            
            # Compute differential (sparse)
            if self.net_split <= 0:
                # Binary differential (sparse)
                cluster_binary = (importance_cluster.copy().astype(bool)).astype(np.float32)
                normal_binary = (importance_normal.copy().astype(bool)).astype(np.float32)
                # XOR via (A + B) - 2*A*B
                xor_result = cluster_binary + normal_binary - 2 * cluster_binary.multiply(normal_binary)
                xor_result.eliminate_zeros()
                diff_cor = xor_result.tocsr()
            else:
                # Weighted differential with threshold (sparse)
                diff = abs(importance_cluster - importance_normal)
                diff.eliminate_zeros()
                # Apply threshold: keep only values > net_split
                diff = diff.tocsr()
                diff.data[diff.data <= self.net_split] = 0
                diff.eliminate_zeros()
                diff_cor = diff
            
            diff_co_data.append(diff_cor)
            
            logger.info("    Cluster %d: %d differential edges",
                        cluster_id, diff_cor.nnz)
        
        return diff_co_data
    
    def _compute_importance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute gene-gene importance matrix using RF or MI.
        OPTIMIZED: Use sparse matrix to save memory + speed.
        """

        if data.size == 0:
            logger.warning("Empty data provided to importance computation")
            n_genes = data.shape[0] if data.ndim > 0 else 0
            return sp.csr_matrix((n_genes, n_genes), dtype=np.float32)

        n_genes, n_samples = data.shape

        if n_samples < 5:
            logger.warning("Too few samples (%d) for importance computation, "
                           "falling back to correlation", n_samples)
            corr = np.abs(np.corrcoef(
                data + 1e-4 * np.random.normal(0, 1, data.shape)
            )).astype(np.float32)
            np.fill_diagonal(corr, 0)
            return sp.csr_matrix(corr)

        # Transpose to (samples x genes) for sklearn
        X = data.T  # (n_samples, n_genes)

        # ✅ OPTIMIZATION 1: Filter to top genes by variance
        logger.info("  Filtering to top genes by variance...")
        gene_variances = np.var(X, axis=0)
        top_k = min(self.top_k_genes, n_genes)
        top_gene_indices = np.argsort(-gene_variances)[:top_k]
        X_filtered = X[:, top_gene_indices]

        logger.info("  Computing importance for top %d genes (of %d by variance), "
                     "max_features=%d, n_jobs=%s",
                     top_k, n_genes, self.max_features_mi, self.n_jobs)

        # ✅ OPTIMIZATION 2: Use SPARSE matrix instead of dense!
        # Store (row, col, value) tuples
        row_indices = []
        col_indices = []
        data_values = []

        if self.method == 'random_forest':
            self._compute_rf_importance_sparse(
                X_filtered, top_k, 
                row_indices, col_indices, data_values
            )
        elif self.method == 'mutual_information':
            self._compute_mi_importance_sparse(
                X_filtered, top_k,
                row_indices, col_indices, data_values
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # ✅ Build sparse matrix
        importance_sparse = sp.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(top_k, top_k),
            dtype=np.float32
        )
        importance_sparse = importance_sparse.tocsr()

        # ✅ Symmetrize sparse matrix
        importance_sparse = importance_sparse + importance_sparse.T
        importance_sparse.data /= 2

        # ✅ Map indices from top-k local to global (stay sparse)
        coo = importance_sparse.tocoo()
        global_rows = top_gene_indices[coo.row]
        global_cols = top_gene_indices[coo.col]
        importance_global = sp.coo_matrix(
            (coo.data, (global_rows, global_cols)),
            shape=(n_genes, n_genes),
            dtype=np.float32
        ).tocsr()

        # Remove diagonal
        importance_global.setdiag(0)
        importance_global.eliminate_zeros()

        # Apply dynamic threshold (on sparse data directly)
        if self.threshold_percentile > 0:
            pos_values = importance_global.data[importance_global.data > 0]
            if len(pos_values) > 0:
                threshold = np.percentile(pos_values, self.threshold_percentile)
                importance_global.data[importance_global.data < threshold] = 0
                importance_global.eliminate_zeros()

        logger.info("    Importance matrix: %d non-zero entries (sparse)",
                    importance_global.nnz)

        return importance_global
        
    
    def _compute_rf_importance_sparse(self, X: np.ndarray, n_genes: int,
                                       row_indices: list, col_indices: list,
                                       data_values: list) -> None:
        """
        Compute RF importance and store as sparse matrix.
        Memory-efficient: only store non-zero values.
        Parallelized via joblib for speed.
        """
        logger.info("  Computing RF importance for %d genes (sparse, n_jobs=%s)...",
                    n_genes, self.n_jobs)

        n_estimators = min(self.n_estimators, 50)
        max_depth = 3
        max_feat = self.max_features_mi
        start_time = time.time()

        def _rf_single_gene(i):
            """Compute RF importance for a single gene."""
            y = X[:, i]
            if np.std(y) < 1e-10:
                return [], [], []

            feature_mask = np.ones(n_genes, dtype=bool)
            feature_mask[i] = False
            X_features = X[:, feature_mask]
            feature_indices = np.arange(n_genes)[feature_mask]

            # ✅ Feature subsampling: keep top-variance features
            if X_features.shape[1] > max_feat:
                feat_var = np.var(X_features, axis=0)
                top_feat_idx = np.argsort(-feat_var)[:max_feat]
                X_features = X_features[:, top_feat_idx]
                feature_indices = feature_indices[top_feat_idx]

            if X_features.shape[1] < 2:
                return [], [], []

            try:
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=max(2, X.shape[0] // 20),
                    max_features='sqrt',
                    n_jobs=1,
                    random_state=42
                )
                rf.fit(X_features, y)

                importances = rf.feature_importances_
                threshold = (
                    np.percentile(importances[importances > 0], 90)
                    if np.any(importances > 0) else 0
                )

                local_rows, local_cols, local_vals = [], [], []
                for j_local, j_global in enumerate(feature_indices):
                    if importances[j_local] > threshold:
                        local_rows.append(i)
                        local_cols.append(j_global)
                        local_vals.append(importances[j_local])
                return local_rows, local_cols, local_vals

            except Exception as e:
                logger.warning("RF failed for gene %d: %s", i, str(e))
                return [], [], []

        # ✅ Parallel execution
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(_rf_single_gene)(i) for i in range(n_genes)
        )

        for local_rows, local_cols, local_vals in results:
            row_indices.extend(local_rows)
            col_indices.extend(local_cols)
            data_values.extend(local_vals)

        elapsed = time.time() - start_time
        logger.info("  RF importance completed in %.1f seconds (%d edges)",
                    elapsed, len(data_values))


    def _compute_mi_importance_sparse(self, X: np.ndarray, n_genes: int,
                                       row_indices: list, col_indices: list,
                                       data_values: list) -> None:
        """
        Compute MI importance and store as sparse matrix.
        Memory-efficient: only store non-zero values.
        Parallelized via joblib for speed.
        """
        logger.info("  Computing MI importance for %d genes (sparse, n_jobs=%s)...",
                    n_genes, self.n_jobs)

        max_feat = self.max_features_mi
        n_neighbors = min(3, X.shape[0] - 1)
        start_time = time.time()

        def _mi_single_gene(i):
            """Compute MI importance for a single gene."""
            y = X[:, i]
            if np.std(y) < 1e-10:
                return [], [], []

            feature_mask = np.ones(n_genes, dtype=bool)
            feature_mask[i] = False
            X_features = X[:, feature_mask]
            feature_indices = np.arange(n_genes)[feature_mask]

            # ✅ Feature subsampling: keep top-variance features
            if X_features.shape[1] > max_feat:
                feat_var = np.var(X_features, axis=0)
                top_feat_idx = np.argsort(-feat_var)[:max_feat]
                X_features = X_features[:, top_feat_idx]
                feature_indices = feature_indices[top_feat_idx]

            if X_features.shape[1] < 2:
                return [], [], []

            try:
                mi_scores = mutual_info_regression(
                    X_features, y,
                    n_neighbors=n_neighbors,
                    random_state=42
                )

                threshold = (
                    np.percentile(mi_scores[mi_scores > 0], 90)
                    if np.any(mi_scores > 0) else 0
                )

                local_rows, local_cols, local_vals = [], [], []
                for j_local, j_global in enumerate(feature_indices):
                    if mi_scores[j_local] > threshold:
                        local_rows.append(i)
                        local_cols.append(j_global)
                        local_vals.append(mi_scores[j_local])
                return local_rows, local_cols, local_vals

            except Exception as e:
                logger.warning("MI failed for gene %d: %s", i, str(e))
                return [], [], []

        # ✅ Parallel execution
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(_mi_single_gene)(i) for i in range(n_genes)
        )

        for local_rows, local_cols, local_vals in results:
            row_indices.extend(local_rows)
            col_indices.extend(local_cols)
            data_values.extend(local_vals)

        elapsed = time.time() - start_time
        logger.info("  MI importance completed in %.1f seconds (%d edges)",
                    elapsed, len(data_values))
    
    def _build_ppi_adjacency(self, ppi: np.ndarray, gene_names: np.ndarray,
                              weight_type: int) -> sp.csr_matrix:
        """
        Build PPI adjacency matrix (weighted, sparse).
        Replicates MODCAN utils.ppi_limitation().
        """
        n_genes = len(gene_names)
        
        if weight_type == 0:
            return sp.csr_matrix((n_genes, n_genes), dtype=np.float32)
        
        gene_list = list(gene_names)
        gene_set = set(gene_names)
        
        # Filter PPI to genes in our set
        ppi_mask = np.array([
            str(row[0]) in gene_set and str(row[1]) in gene_set
            for row in ppi
        ])
        ppi_filtered = ppi[ppi_mask]
        
        idx_0 = [gene_list.index(str(row[0])) for row in ppi_filtered]
        idx_1 = [gene_list.index(str(row[1])) for row in ppi_filtered]
        edges = np.array([float(row[2]) for row in ppi_filtered])
        
        # Apply weight type
        if weight_type == 1:
            edges = (edges > 0).astype(np.float32)
        elif weight_type == 2:
            edges = self._sigmoid(edges / np.max(edges)).astype(np.float32)
        elif weight_type == 3:
            edges = (edges / np.max(edges)).astype(np.float32)
        
        # Build sparse matrix (stay sparse!)
        ppi_network = sp.coo_matrix(
            (edges, (idx_0, idx_1)),
            shape=(n_genes, n_genes),
            dtype=np.float32
        ).tocsr()
        
        # Symmetrize: max(A, A^T) in sparse
        ppi_t = ppi_network.T.tocsr()
        # Element-wise max via: max(A,B) = (A+B + |A-B|) / 2
        ppi_sum = ppi_network + ppi_t
        ppi_diff = abs(ppi_network - ppi_t)
        ppi_network = (ppi_sum + ppi_diff) / 2
        ppi_network = ppi_network.tocsr()
        ppi_network.eliminate_zeros()
        
        logger.info("PPI adjacency: %d nodes, %d edges (sparse)",
                     n_genes, ppi_network.nnz)
        
        return ppi_network
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def _apply_ppi_constraint(self, diff_co_data: list,
                               ppi_network: sp.spmatrix) -> list:
        """
        Element-wise multiply differential co-association with PPI (sparse).
        Replicates MODCAN utils.construct_association_network().
        """
        co_data_net = []
        for i, diff_cor in enumerate(diff_co_data):
            # Element-wise multiply two sparse matrices
            constrained = diff_cor.multiply(ppi_network).tocsr()
            constrained.eliminate_zeros()
            logger.info("  Cluster %d: %d edges after PPI constraint",
                        i, constrained.nnz)
            co_data_net.append(constrained)
        
        return co_data_net
    
    # NOTE: _build_hypergraph, _get_support_matrices, _chebyshev_polynomials,
    # and _normalize_adj have been removed.
    # PyG's GCNConv performs D^{-1/2} A D^{-1/2} normalization internally,
    # making manual Chebyshev expansion unnecessary and memory-wasteful
    # (it caused OOM by densifying sparse matrices from ~21K to 273M edges).
