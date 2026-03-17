"""
Gene Feature Extraction Module
===============================
Python port of R Gene_Feature.R.
Computes per-gene features combining tumor vs normal differences
across patient subgroups and PPI network topology metrics.
"""

import logging
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from joblib import parallel_backend
from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class GeneFeatureExtractor:
    """
    Extracts gene features for GNN input.
    Combines per-cluster multi-omics features with PPI network centrality.
    Replaces R Gene_Feature.R logic.
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def extract(self, data: dict) -> np.ndarray:
        """
        Extract comprehensive gene features combining:
          - Per-cluster: SNV ratio, CNV ratio, EXP diff, MET diff
          - Global: aggregated across all samples
          - PPI topology: degree, clustering coeff, closeness, betweenness

        Args:
            data: Preprocessed data dictionary with cluster_labels.
        
        Returns:
            Gene feature matrix (n_genes x n_features).
        """
        logger.info("Extracting gene features...")
        
        gene_names = data['gene_names']
        n_genes = len(gene_names)
        cluster_labels = data['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        all_features = []
        feature_names = []
        
        # --- Per-cluster features ---
        for k in unique_clusters:
            cluster_mask = (cluster_labels == k)
            cluster_features, cluster_feat_names = self._compute_cluster_features(
                data, gene_names, cluster_mask, k
            )
            all_features.append(cluster_features)
            feature_names.extend(cluster_feat_names)
        
        # --- Global features (all samples) ---
        global_mask = np.ones(len(cluster_labels), dtype=bool)
        global_features, global_feat_names = self._compute_cluster_features(
            data, gene_names, global_mask, 'global'
        )
        all_features.append(global_features)
        feature_names.extend(global_feat_names)
        
        # --- PPI topology features ---
        ppi_features, ppi_feat_names = self._compute_ppi_features(
            data['ppi'], gene_names
        )
        all_features.append(ppi_features)
        feature_names.extend(ppi_feat_names)
        
        # Concatenate all features
        gene_feature_matrix = np.column_stack(all_features)
        
        # Handle NaN values
        gene_feature_matrix = np.nan_to_num(gene_feature_matrix, nan=0.0)
        
        # Scale features (MinMax)
        scaler = MinMaxScaler()
        gene_feature_matrix = scaler.fit_transform(gene_feature_matrix)
        gene_feature_matrix = np.clip(gene_feature_matrix, 1e-4, 1 - 1e-4).astype(np.float32)
        
        logger.info("Gene features: %s (%d features: %s)",
                     gene_feature_matrix.shape, len(feature_names),
                     ', '.join(feature_names[:8]) + ('...' if len(feature_names) > 8 else ''))
        log_tensor_info(logger, "gene_feature_matrix", gene_feature_matrix)
        
        data['gene_features'] = gene_feature_matrix
        data['feature_names'] = feature_names
        
        return gene_feature_matrix
    
    def _compute_cluster_features(self, data: dict, gene_names: np.ndarray,
                                   sample_mask: np.ndarray,
                                   cluster_id) -> tuple:
        """
        Compute per-cluster features: SNV ratio, CNV ratio, EXP diff, MET diff.
        """
        n_genes = len(gene_names)
        features = np.zeros((n_genes, 4), dtype=np.float32)
        suffix = f"_{cluster_id}"
        
        n_cluster_samples = sample_mask.sum()
        if n_cluster_samples == 0:
            return features, [f"snv_ratio{suffix}", f"cnv_ratio{suffix}",
                              f"exp_diff{suffix}", f"met_diff{suffix}"]
        
        # SNV ratio
        if data['snv_data'].size > 0:
            snv_genes_map = {g: i for i, g in enumerate(data['snv_genes'])}
            # Find matching samples
            snv_sample_mask = sample_mask[:min(len(sample_mask), data['snv_data'].shape[1])]
            for i, gene in enumerate(gene_names):
                if gene in snv_genes_map:
                    idx = snv_genes_map[gene]
                    if idx < data['snv_data'].shape[0]:
                        vals = data['snv_data'][idx, :len(snv_sample_mask)]
                        features[i, 0] = vals[snv_sample_mask[:len(vals)]].sum() / n_cluster_samples
        
        # CNV ratio
        if data['cnv_data'].size > 0:
            cnv_genes_map = {g: i for i, g in enumerate(data['cnv_genes'])}
            cnv_sample_mask = sample_mask[:min(len(sample_mask), data['cnv_data'].shape[1])]
            for i, gene in enumerate(gene_names):
                if gene in cnv_genes_map:
                    idx = cnv_genes_map[gene]
                    if idx < data['cnv_data'].shape[0]:
                        vals = np.abs(data['cnv_data'][idx, :len(cnv_sample_mask)])
                        features[i, 1] = vals[cnv_sample_mask[:len(vals)]].sum() / n_cluster_samples
        
        # EXP diff (cluster mean - normal mean)
        if data['exp_data'].size > 0 and data['exp_normal'].size > 0:
            exp_sample_mask = sample_mask[:min(len(sample_mask), data['exp_data'].shape[1])]
            if exp_sample_mask.sum() > 0:
                exp_cluster_mean = np.mean(data['exp_data'][:, exp_sample_mask[:data['exp_data'].shape[1]]], axis=1)
                exp_normal_mean = np.mean(data['exp_normal'], axis=1)
                min_len = min(len(exp_cluster_mean), len(exp_normal_mean), n_genes)
                features[:min_len, 2] = (exp_cluster_mean[:min_len] - exp_normal_mean[:min_len])
        
        # MET diff (cluster mean - normal mean)
        if data['met_data'].size > 0 and data['met_normal'].size > 0:
            met_sample_mask = sample_mask[:min(len(sample_mask), data['met_data'].shape[1])]
            if met_sample_mask.sum() > 0:
                met_cluster_mean = np.mean(data['met_data'][:, met_sample_mask[:data['met_data'].shape[1]]], axis=1)
                met_normal_mean = np.mean(data['met_normal'], axis=1)
                min_len = min(len(met_cluster_mean), len(met_normal_mean), n_genes)
                features[:min_len, 3] = (met_cluster_mean[:min_len] - met_normal_mean[:min_len])
        
        feat_names = [f"snv_ratio{suffix}", f"cnv_ratio{suffix}",
                      f"exp_diff{suffix}", f"met_diff{suffix}"]
        
        return features, feat_names
    
    def _compute_ppi_features(self, ppi: np.ndarray,
                               gene_names: np.ndarray) -> tuple:
        """
        Compute PPI network topology features:
          - Degree centrality
          - Clustering coefficient
          - Closeness centrality
          - Betweenness centrality
        """
        logger.info("Computing PPI topology features...")
        
        n_genes = len(gene_names)
        features = np.zeros((n_genes, 4), dtype=np.float32)
        
        # Build NetworkX graph from PPI
        G = nx.Graph()
        for row in ppi:
            gene1, gene2 = str(row[0]), str(row[1])
            score = float(row[2]) if len(row) > 2 else 1.0
            G.add_edge(gene1, gene2, weight=score)
        
        G = G.to_undirected()
        
        gene_set = set(gene_names)
        ppi_genes = set(G.nodes()) & gene_set
        logger.info("PPI network: %d nodes, %d edges, %d overlap with gene set",
                     G.number_of_nodes(), G.number_of_edges(), len(ppi_genes))
        
        # Compute centrality metrics
        degree_cent = nx.degree_centrality(G)
        clustering_coeff = nx.clustering(G)
        
        # Closeness and betweenness can be expensive, compute only for relevant genes
        
        # closeness_cent = nx.closeness_centrality(G)
        # Lấy largest connected component
        # largest_cc = max(nx.connected_components(G), key=len)
        # G_largest = G.subgraph(largest_cc).copy()

        # # Tính closeness chỉ cho component này
        # closeness_cent_partial = nx.closeness_centrality(G_largest)

        # # Gen ngoài component có closeness = 0
        # closeness_cent = {node: closeness_cent_partial.get(node, 0) 
        #             for node in G.nodes()}

        logger.info("Computing betweenness centrality (approximate with k=1000)...")
        # Chạy betweenness trên multi-core
        with parallel_backend('threading', n_jobs=self.config.get('n_jobs', -1)):
            betweenness_cent = nx.betweenness_centrality(
                G,
                normalized=True,
                k=min(1000, G.number_of_nodes()),  # Approximate for large graphs
                seed=42
            )

        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        for gene in ppi_genes:
            if gene in gene_name_to_idx:
                idx = gene_name_to_idx[gene]
                features[idx, 0] = degree_cent.get(gene, 0)
                features[idx, 1] = clustering_coeff.get(gene, 0)
                features[idx, 2] = betweenness_cent.get(gene, 0)
        
        feat_names = ["deg_cent", "clu_coef", "bet_cent"]
        
        return features, feat_names
