"""
Spectral Clustering Pipeline for Patient Subgroup Identification
================================================================
Python port of R Cluster.R (SNFtool-based).
Uses Similarity Network Fusion (SNF) to integrate:
  - Community Cohesion Scores (CCS) / gene features
  - Gene Expression data
  - Methylation data
Then applies Spectral Clustering on the fused similarity matrix.
"""

import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class SpectralClusteringPipeline:
    """
    Multi-omics spectral clustering via Similarity Network Fusion.
    Replaces R SNFtool pipeline with pure Python/scikit-learn implementation.
    """
    
    def __init__(self, config: dict):
        self.n_clusters = config['clustering']['n_clusters']
        self.k_neighbors = config['clustering']['snf_k_neighbors']
        self.alpha = config['clustering']['snf_alpha']
        self.n_iterations = config['clustering']['snf_iterations']
        self.top_genes = config['clustering']['top_variance_genes']
    
    def fit_predict(self, data: dict) -> np.ndarray:
        """
        Perform multi-omics spectral clustering.
        
        Args:
            data: Preprocessed data dictionary with exp_data, met_data, etc.
        
        Returns:
            Cluster labels array (1-indexed, shape: n_samples)
        """
        logger.info("Starting spectral clustering pipeline...")
        
        # Select top-variance genes for clustering
        exp_data = self._select_top_variance_genes(data['exp_data'], self.top_genes)
        met_data = self._select_top_variance_genes(data['met_data'], self.top_genes)
        
        # Normalize data
        scaler = StandardScaler()
        exp_normalized = scaler.fit_transform(exp_data.T)  # samples x genes
        met_normalized = scaler.fit_transform(met_data.T)
        
        # Compute distance matrices
        dist_exp = squareform(pdist(exp_normalized, metric='euclidean'))
        dist_met = squareform(pdist(met_normalized, metric='euclidean'))
        
        logger.info("Distance matrices: EXP=%s, MET=%s", dist_exp.shape, dist_met.shape)
        
        # Compute affinity matrices
        W_exp = self._affinity_matrix(dist_exp, self.k_neighbors, self.alpha)
        W_met = self._affinity_matrix(dist_met, self.k_neighbors, self.alpha)
        
        # Run Similarity Network Fusion
        W_fused = self._snf([W_exp, W_met], self.k_neighbors, self.n_iterations)
        
        log_tensor_info(logger, "Fused similarity matrix", W_fused)
        
        # Determine optimal number of clusters if auto
        if self.n_clusters <= 0:
            self.n_clusters = self._find_optimal_k(W_fused, max_k=10)
        
        # Spectral clustering
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42,
            n_init=20
        )
        labels = sc.fit_predict(W_fused)
        
        # Convert to 1-indexed (consistent with MODCAN)
        labels = labels + 1
        
        # Evaluate clustering quality
        sil_score = silhouette_score(W_fused, labels, metric='precomputed')
        logger.info("Spectral clustering: %d clusters, silhouette score = %.4f",
                     self.n_clusters, sil_score)
        
        # Log cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info("  Cluster %d: %d samples (%.1f%%)",
                         cluster_id, count, 100 * count / len(labels))
        
        return labels
    
    def _select_top_variance_genes(self, matrix: np.ndarray,
                                    top_n: int) -> np.ndarray:
        """Select top-N genes by variance across samples."""
        if matrix.size == 0 or top_n >= matrix.shape[0]:
            return matrix
        
        variances = np.var(matrix, axis=1)
        top_idx = np.argsort(variances)[-top_n:]
        return matrix[top_idx, :]
    
    def _affinity_matrix(self, dist: np.ndarray, k: int,
                          alpha: float) -> np.ndarray:
        """
        Compute affinity matrix using scaled exponential similarity kernel.
        Equivalent to SNFtool::affinityMatrix in R.
        
        W_ij = exp(-d_ij^2 / (alpha * mean(knn_i) * mean(knn_j)))
        """
        n = dist.shape[0]
        
        # For each sample, find mean distance to K nearest neighbors
        knn_means = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(dist[i, :])
            # sorted_dists[0] is distance to self (0), take next K
            knn_means[i] = np.mean(sorted_dists[1:k + 1]) + 1e-10
        
        # Compute affinity
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                scale = alpha * knn_means[i] * knn_means[j]
                if scale > 0:
                    W[i, j] = np.exp(-dist[i, j] ** 2 / scale)
                    W[j, i] = W[i, j]
        
        return W
    
    def _snf(self, affinity_matrices: list, k: int,
             n_iterations: int) -> np.ndarray:
        """
        Similarity Network Fusion (SNF) algorithm.
        Iteratively fuses multiple affinity networks using two-step diffusion.
        
        Reference: Wang et al., 2014, Nature Methods
        """
        n_views = len(affinity_matrices)
        n = affinity_matrices[0].shape[0]
        
        # Normalize affinity matrices to create transition matrices
        P_list = []
        S_list = []
        
        for W in affinity_matrices:
            P = self._normalize_network(W)
            S = self._knn_network(W, k)
            P_list.append(P)
            S_list.append(S)
        
        # Iterative diffusion
        for iteration in range(n_iterations):
            P_new_list = []
            for v in range(n_views):
                # Average transition matrices from all other views
                P_avg = np.zeros((n, n))
                for u in range(n_views):
                    if u != v:
                        P_avg += P_list[u]
                P_avg /= (n_views - 1)
                
                # Diffuse: P_new = S_v @ P_avg @ S_v^T
                P_new = S_list[v] @ P_avg @ S_list[v].T
                P_new = self._normalize_network(P_new)
                P_new_list.append(P_new)
            
            P_list = P_new_list
        
        # Final fused matrix: average all views
        W_fused = np.zeros((n, n))
        for P in P_list:
            W_fused += P
        W_fused /= n_views
        
        # Symmetrize
        W_fused = (W_fused + W_fused.T) / 2
        np.fill_diagonal(W_fused, 0)
        
        return W_fused
    
    def _normalize_network(self, W: np.ndarray) -> np.ndarray:
        """Normalize affinity matrix: P = D^{-1} @ W."""
        row_sum = W.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        D_inv = np.diag(1.0 / row_sum)
        return D_inv @ W
    
    def _knn_network(self, W: np.ndarray, k: int) -> np.ndarray:
        """Create K-nearest neighbor network from affinity matrix."""
        n = W.shape[0]
        S = np.zeros_like(W)
        
        for i in range(n):
            # Find K largest similarities (excluding self)
            row = W[i, :].copy()
            row[i] = -np.inf
            top_k_idx = np.argsort(row)[-k:]
            S[i, top_k_idx] = W[i, top_k_idx]
        
        # Symmetrize
        S = (S + S.T) / 2
        
        # Normalize
        row_sum = S.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        S = np.diag(1.0 / row_sum) @ S
        
        return S
    
    def _find_optimal_k(self, W: np.ndarray, max_k: int = 10) -> int:
        """Determine optimal number of clusters using silhouette score."""
        logger.info("Auto-determining optimal number of clusters...")
        
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            sc = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            labels = sc.fit_predict(W)
            score = silhouette_score(W, labels, metric='precomputed')
            logger.info("  k=%d: silhouette=%.4f", k, score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info("Optimal k=%d (silhouette=%.4f)", best_k, best_score)
        return best_k
