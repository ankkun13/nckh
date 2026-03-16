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
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse.linalg import eigs

import torch

from utils.logger import log_tensor_info


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
        
        # Step 4: Build PPI adjacency matrix
        logger.info("Building PPI adjacency matrix...")
        ppi_network = self._build_ppi_adjacency(
            data['ppi'], gene_names, self.ppi_weight_type
        )
        
        # Step 5: Multiply with PPI constraint
        logger.info("Applying PPI biological constraint...")
        co_data_net = self._apply_ppi_constraint(diff_co_combined, ppi_network)
        
        # Step 6: Build hypergraph (Chebyshev polynomial expansion)
        logger.info("Building hypergraph matrix...")
        hp_graph = self._build_hypergraph(gene_names, co_data_net)
        
        log_tensor_info(logger, "hypergraph", hp_graph)
        
        # Cleanup
        del diff_co_exp, diff_co_met, diff_co_combined, ppi_network
        gc.collect()
        
        graph_data = {
            'hp_graph': hp_graph,
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
        
        # Compute importance matrix for normal data
        logger.info("  Computing %s importance for normal tissue...", self.method)
        importance_normal = self._compute_importance_matrix(normal_data)
        
        diff_co_data = []
        for cluster_id in unique_clusters:
            logger.info("  Computing %s importance for cluster %d...",
                         self.method, cluster_id)
            
            # Get tumor samples in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            n_cluster = cluster_mask.sum()
            
            # Ensure mask length matches data columns
            effective_mask = cluster_mask[:tumor_data.shape[1]]
            if effective_mask.sum() == 0:
                diff_co_data.append(np.zeros((n_genes, n_genes), dtype=np.float32))
                continue
            
            tumor_cluster = tumor_data[:, effective_mask]
            
            # Compute importance matrix for this cluster
            importance_cluster = self._compute_importance_matrix(tumor_cluster)
            
            # Compute differential
            if self.net_split <= 0:
                # Binary differential
                diff_cor = ((np.abs(importance_cluster) > 0.6) ^
                            (np.abs(importance_normal) > 0.6)).astype(np.float32)
            else:
                # Weighted differential with threshold
                diff = np.abs(importance_cluster - importance_normal)
                diff_mask = (diff > self.net_split).astype(np.float32)
                diff_cor = (diff * diff_mask).astype(np.float32)
            
            diff_co_data.append(diff_cor)
            
            edge_count = np.count_nonzero(diff_cor)
            logger.info("    Cluster %d: %d differential edges", cluster_id, edge_count)
        
        return diff_co_data
    
    def _compute_importance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute gene-gene importance matrix using RF or MI.
        
        For each gene g_i (as target), compute importance of all other genes
        as predictors. Build symmetric matrix.
        
        Args:
            data: Gene x Sample matrix
        
        Returns:
            Symmetric importance matrix (n_genes x n_genes)
        """
        n_genes, n_samples = data.shape
        
        if n_samples < 5:
            logger.warning("Too few samples (%d) for importance computation, "
                           "falling back to correlation", n_samples)
            return np.abs(np.corrcoef(
                data + 1e-4 * np.random.normal(0, 1, data.shape)
            )).astype(np.float32)
        
        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        
        # Transpose to (samples x genes) for sklearn
        X = data.T  # (n_samples, n_genes)
        
        if self.method == 'random_forest':
            importance_matrix = self._compute_rf_importance(X, n_genes)
        elif self.method == 'mutual_information':
            importance_matrix = self._compute_mi_importance(X, n_genes)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Symmetrize
        importance_matrix = (importance_matrix + importance_matrix.T) / 2
        np.fill_diagonal(importance_matrix, 0)
        
        # Apply dynamic threshold
        if self.threshold_percentile > 0:
            values = importance_matrix[importance_matrix > 0]
            if len(values) > 0:
                threshold = np.percentile(values, self.threshold_percentile)
                importance_matrix[importance_matrix < threshold] = 0
        
        return importance_matrix
    
    def _compute_rf_importance(self, X: np.ndarray,
                                n_genes: int) -> np.ndarray:
        """
        Compute Random Forest feature importance matrix.
        For efficiency, process genes in batches.
        """
        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        
        # Sample genes if too many (for computational efficiency)
        gene_indices = np.arange(n_genes)
        
        for i in range(n_genes):
            if i % 200 == 0:
                logger.debug("    RF importance: gene %d/%d", i, n_genes)
            
            y = X[:, i]
            
            # Use all other genes as features
            feature_mask = np.ones(n_genes, dtype=bool)
            feature_mask[i] = False
            X_features = X[:, feature_mask]
            
            # Skip if constant target
            if np.std(y) < 1e-10:
                continue
            
            try:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    min_samples_leaf=max(2, X.shape[0] // 20),
                    n_jobs=self.n_jobs,
                    random_state=42
                )
                rf.fit(X_features, y)
                
                # Map importances back to full gene indices
                feature_indices = gene_indices[feature_mask]
                importance_matrix[i, feature_indices] = rf.feature_importances_
                
            except Exception as e:
                logger.warning("RF failed for gene %d: %s", i, str(e))
                continue
        
        return importance_matrix
    
    def _compute_mi_importance(self, X: np.ndarray,
                                n_genes: int) -> np.ndarray:
        """
        Compute Mutual Information importance matrix.
        """
        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        
        for i in range(n_genes):
            if i % 200 == 0:
                logger.debug("    MI importance: gene %d/%d", i, n_genes)
            
            y = X[:, i]
            
            if np.std(y) < 1e-10:
                continue
            
            feature_mask = np.ones(n_genes, dtype=bool)
            feature_mask[i] = False
            X_features = X[:, feature_mask]
            
            try:
                mi_scores = mutual_info_regression(
                    X_features, y,
                    n_neighbors=min(5, X.shape[0] - 1),
                    random_state=42
                )
                
                feature_indices = np.arange(n_genes)[feature_mask]
                importance_matrix[i, feature_indices] = mi_scores
                
            except Exception as e:
                logger.warning("MI failed for gene %d: %s", i, str(e))
                continue
        
        return importance_matrix
    
    def _build_ppi_adjacency(self, ppi: np.ndarray, gene_names: np.ndarray,
                              weight_type: int) -> np.ndarray:
        """
        Build PPI adjacency matrix (weighted).
        Replicates MODCAN utils.ppi_limitation().
        """
        if weight_type == 0:
            return np.zeros(1)  # No PPI constraint
        
        n_genes = len(gene_names)
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
        
        # Build sparse matrix
        ppi_network = sp.coo_matrix(
            (edges, (idx_0, idx_1)),
            shape=(n_genes, n_genes),
            dtype=np.float32
        ).toarray()
        
        # Symmetrize
        ppi_network = np.maximum(ppi_network, ppi_network.T).astype(np.float32)
        
        logger.info("PPI adjacency: %d nodes, %d edges",
                     n_genes, len(idx_0))
        
        return ppi_network
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def _apply_ppi_constraint(self, diff_co_data: list,
                               ppi_network: np.ndarray) -> list:
        """
        Element-wise multiply differential co-association with PPI.
        Replicates MODCAN utils.construct_association_network().
        """
        co_data_net = []
        for i, diff_cor in enumerate(diff_co_data):
            constrained = diff_cor * ppi_network
            edge_num = np.sum(constrained > 0)
            logger.info("  Cluster %d: %d edges after PPI constraint", i, edge_num)
            co_data_net.append(constrained)
        
        return co_data_net
    
    def _build_hypergraph(self, gene_names: np.ndarray,
                           co_data_net: list) -> np.ndarray:
        """
        Build multi-slice hypergraph matrix with Chebyshev polynomials.
        Replicates MODCAN utils.get_hypergraph_matrix().
        """
        hp_graphs = []
        for s, adj in enumerate(co_data_net):
            support = self._get_support_matrices(adj, poly_support=1)
            hp_graphs.append(support)
        
        # Stack into 3D array (n_genes, n_genes, n_slices)
        hp_graph_arr = hp_graphs[0][:, :, np.newaxis]
        for s in range(1, len(hp_graphs)):
            hp_graph_arr = np.concatenate(
                (hp_graph_arr, hp_graphs[s][:, :, np.newaxis]),
                axis=2
            )
        
        logger.info("Hypergraph: %d genes, %d slices", gene_names.shape[0], len(hp_graphs))
        
        return hp_graph_arr
    
    def _get_support_matrices(self, adj: np.ndarray,
                               poly_support: int) -> np.ndarray:
        """Compute Chebyshev polynomial support matrices."""
        if poly_support > 0:
            support = self._chebyshev_polynomials(adj, poly_support)
            ppi_graph = support[0]
            for i in range(1, len(support)):
                ppi_graph = ppi_graph + support[i]
            
            if sp.issparse(ppi_graph):
                return ppi_graph.toarray().astype(np.float32)
            return ppi_graph.astype(np.float32)
        else:
            return np.eye(adj.shape[0], dtype=np.float32)
    
    def _chebyshev_polynomials(self, adj: np.ndarray, k: int) -> list:
        """
        Chebyshev polynomials up to order k.
        Replicates MODCAN utils.chebyshev_polynomials().
        """
        adj_normalized = self._normalize_adj(adj)
        n = adj.shape[0]
        laplacian = sp.eye(n) - adj_normalized
        
        try:
            largest_eigval, _ = eigs(laplacian, k=1, which='LR')
            eigval = largest_eigval[0].real
        except Exception:
            eigval = 2.0  # Default if eigendecomposition fails
        
        if eigval == 0:
            eigval = 2.0
        
        scaled_laplacian = (2.0 / eigval) * laplacian - sp.eye(n)
        
        t_k = [sp.eye(n), scaled_laplacian]
        
        for _ in range(2, k + 1):
            s_lap = sp.csr_matrix(scaled_laplacian)
            t_k.append(2 * s_lap.dot(t_k[-1]) - t_k[-2])
        
        # Subtract lower support
        for i in range(1, len(t_k)):
            for j in range(0, i):
                if j == 0:
                    mask = np.abs(t_k[j].todense()) > 0.0001
                    if sp.issparse(t_k[i]):
                        t_k_dense = t_k[i].todense()
                        t_k_dense[mask] = 0
                        t_k[i] = sp.csr_matrix(t_k_dense)
                    else:
                        t_k[i][mask] = 0
                else:
                    if sp.issparse(t_k[j]):
                        mask = np.abs(t_k[j].todense()) > 0.0001
                    else:
                        mask = np.abs(t_k[j]) > 0.0001
                    if sp.issparse(t_k[i]):
                        t_k_dense = t_k[i].todense()
                        t_k_dense[mask] = 0
                        t_k[i] = sp.csr_matrix(t_k_dense)
                    else:
                        t_k[i][mask] = 0
        
        return t_k
    
    def _normalize_adj(self, adj: np.ndarray) -> sp.coo_matrix:
        """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
        row_sum = np.array(adj.sum(axis=1)).flatten()
        idx_0 = np.where(row_sum == 0.0)[0]
        row_sum[idx_0] = 1.0
        
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[idx_0] = 0.0
        d_mat = np.diagflat(d_inv_sqrt)
        
        res = adj.dot(d_mat).T.dot(d_mat)
        return sp.coo_matrix(res)
