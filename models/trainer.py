"""
GNN Trainer with K-Fold Cross-Validation
=========================================
Full training pipeline with:
  - Dynamic CUDA device selection
  - Focal Loss for class imbalance
  - Early stopping on validation AUPR
  - K-fold cross-validation
  - OOM prevention (torch.cuda.empty_cache)
  - Comprehensive execution logging
"""

import os
import gc
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from models.gnn_model import DriverGeneGNN, MGCNLegacy
from models.focal_loss import FocalLoss, MaskedCrossEntropyLoss
from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class GNNTrainer:
    """
    Trains the GNN model with cross-validation.
    Handles device management, data splitting, and metric tracking.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.early_patience = config['training']['early_stopping_patience']
        self.min_epoch = config['training']['early_stopping_min_epoch']
        self.cv_folds = config['evaluation'].get('cv_folds', 10)
        self.dropout = config['model']['dropout']
        self.hidden_channels = config['model']['hidden_channels']
        self.conv_type = config['model']['conv_type']
        self.use_residual = config['model']['use_residual']
        
        # Focal loss config
        self.focal_alpha = config['focal_loss']['alpha']
        self.focal_gamma = config['focal_loss']['gamma']
        
        # Device setup
        device_cfg = config['training']['device']
        if device_cfg == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_cfg)
        
        logger.info("Training device: %s", self.device)
        if self.device.type == 'cuda':
            logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
            logger.info("CUDA memory: %.2f GB",
                         torch.cuda.get_device_properties(0).total_memory / 1e9)
    
    def train_with_cv(self, data: dict) -> dict:
        """
        Train with K-fold cross-validation.
        
        Args:
            data: Processed data with gene_features, gene_labels, graph_data.
        
        Returns:
            Dictionary with per-fold results:
              - performance_measures: list of (test_metrics, all_metrics, loss)
              - best_model_state: state dict of best model
              - predictions: per-fold predictions
        """
        gene_features = data['gene_features']
        gene_labels = data['gene_labels']  # (n_genes, 2): [is_driver, is_labeled]
        graph_data = data['graph_data']
        co_data_net = graph_data['co_data_net']
        
        n_genes = gene_features.shape[0]
        n_features = gene_features.shape[1]
        n_slices = graph_data['n_slices']
        
        logger.info("Training configuration:")
        logger.info("  Genes: %d, Features: %d, Graph slices: %d", n_genes, n_features, n_slices)
        logger.info("  Epochs: %d, LR: %.4f, CV folds: %d", self.epochs, self.lr, self.cv_folds)
        logger.info("  Model: %s, Hidden: %s, Residual: %s",
                     self.conv_type, self.hidden_channels, self.use_residual)
        
        # Prepare edge indices from sparse adjacency matrices (Stay on CPU for now)
        edge_indices, edge_weights = self._adjacency_to_edge_index(co_data_net)
        
        # Prepare labels and masks
        y_label = gene_labels[:, 0]  # Driver labels
        y_mask = gene_labels[:, 1]   # Labeled mask
        labeled_idx = np.where(y_mask == 1)[0]
        
        logger.info("  Labeled genes: %d, Drivers: %d, Non-drivers: %d",
                     len(labeled_idx), y_label[labeled_idx].sum(),
                     len(labeled_idx) - y_label[labeled_idx].sum())
        
        # ✅ OPTIMIZATION: Move static data to device ONCE
        logger.info("  Moving static data to %s...", self.device)
        x_device = torch.FloatTensor(gene_features).to(self.device)
        edge_idx_device = [ei.to(self.device) for ei in edge_indices]
        edge_w_device = [ew.to(self.device) for ew in edge_weights] if edge_weights else None
        y_label_full_t = torch.FloatTensor(y_label).to(self.device)
        
        # Split: 75% train+val, 25% test (fixed)
        kf_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        outer_splits = list(kf_outer.split(labeled_idx, y_label[labeled_idx]))
        train_val_idx, test_idx = labeled_idx[outer_splits[0][0]], labeled_idx[outer_splits[0][1]]
        
        test_mask_t = torch.BoolTensor(self._make_mask(n_genes, test_idx)).to(self.device)
        y_test_t = torch.FloatTensor(self._make_y(y_label, test_idx, n_genes)).to(self.device)
        
        # K-fold CV on train+val
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        performance_measures = []
        all_predictions = []
        best_overall_aupr = 0
        best_model_state = None
        
        for cv_run, (train_split, val_split) in enumerate(
            kf.split(train_val_idx, y_label[train_val_idx])
        ):
            logger.info("")
            logger.info("=" * 60)
            logger.info("Cross-validation fold %d/%d", cv_run + 1, self.cv_folds)
            logger.info("=" * 60)
            
            train_idx = train_val_idx[train_split]
            val_idx = train_val_idx[val_split]
            
            train_mask_t = torch.BoolTensor(self._make_mask(n_genes, train_idx)).to(self.device)
            val_mask_t = torch.BoolTensor(self._make_mask(n_genes, val_idx)).to(self.device)
            y_train_t = torch.FloatTensor(self._make_y(y_label, train_idx, n_genes)).to(self.device)
            y_val_t = torch.FloatTensor(self._make_y(y_label, val_idx, n_genes)).to(self.device)
            
            logger.info("  Train: %d, Val: %d, Test: %d",
                         len(train_idx), len(val_idx), len(test_idx))
            
            # Build and train model
            model, test_perf, all_perf, preds = self._single_cv_run(
                x_device, edge_idx_device, edge_w_device,
                y_train_t, train_mask_t, y_val_t, val_mask_t,
                y_test_t, test_mask_t, y_label_full_t,
                n_slices, cv_run
            )
            
            performance_measures.append({
                'cv': cv_run,
                'test_metrics': test_perf,
                'all_metrics': all_perf,
            })
            all_predictions.append(preds)
            
            if test_perf['aupr'] > best_overall_aupr:
                best_overall_aupr = test_perf['aupr']
                best_model_state = copy.deepcopy(model.state_dict())
            
            # Cleanup Fold
            del model, train_mask_t, val_mask_t, y_train_t, y_val_t
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # Full Cleanup
        del x_device, edge_idx_device, edge_w_device, y_label_full_t, test_mask_t, y_test_t
        gc.collect()
        
        # Summary
        avg_auc = np.mean([pm['test_metrics']['auc'] for pm in performance_measures])
        avg_aupr = np.mean([pm['test_metrics']['aupr'] for pm in performance_measures])
        logger.info("")
        logger.info("=" * 60)
        logger.info("CV RESULTS SUMMARY")
        logger.info("  Average Test AUC:  %.4f", avg_auc)
        logger.info("  Average Test AUPR: %.4f", avg_aupr)
        logger.info("  Best Test AUPR:    %.4f", best_overall_aupr)
        logger.info("=" * 60)
        
        results = {
            'performance_measures': performance_measures,
            'predictions': all_predictions,
            'best_model_state': best_model_state,
            'gene_names': data['gene_names'],
            'gene_labels': gene_labels,
        }
        
        return results
    
    def _single_cv_run(self, x, edge_indices, edge_weights,
                        y_train_t, train_mask_t, y_val_t, val_mask_t,
                        y_test_t, test_mask_t, y_label_full_t,
                        n_slices, cv_run):
        """Train and evaluate a single CV fold."""
        
        n_features = x.shape[1]
        n_genes = x.shape[0]
        
        # Build model
        model = DriverGeneGNN(
            in_channels=n_features,
            hidden_channels=self.hidden_channels,
            num_classes=2,
            n_graphs=n_slices,
            conv_type=self.conv_type,
            dropout=self.dropout,
            use_residual=self.use_residual
        ).to(self.device)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Training loop
        best_val_aupr = 0
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            
            logits, probs = model(x, edge_indices, edge_weights)
            
            loss = criterion(
                logits[train_mask_t],
                y_train_t[train_mask_t].long(),
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0 or epoch == 1:
                model.eval()
                with torch.inference_mode():
                    _, val_probs = model(x, edge_indices, edge_weights)
                
                val_metrics = self._compute_metrics(
                    val_probs, y_val_t, val_mask_t
                )
                
                if val_metrics['aupr'] > best_val_aupr and epoch >= self.min_epoch:
                    best_val_aupr = val_metrics['aupr']
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    
                    logger.info(
                        "  ★ Epoch %d: loss=%.4f | val_auc=%.4f | val_aupr=%.4f",
                        epoch, loss.item(), val_metrics['auc'], val_metrics['aupr']
                    )
                else:
                    patience_counter += 1
                
                if epoch % 100 == 0:
                    logger.info(
                        "    Epoch %d: loss=%.4f | val_auc=%.4f | val_aupr=%.4f",
                        epoch, loss.item(), val_metrics['auc'], val_metrics['aupr']
                    )
                
                # Early stopping
                if patience_counter * 10 >= self.early_patience:
                    logger.info("  Early stopping at epoch %d (No improvement for %d epochs)", 
                                epoch, self.early_patience)
                    break
            
            # CUDA memory management
            if self.device.type == 'cuda' and epoch % 100 == 0:
                torch.cuda.empty_cache()
        
        # Load best weights for evaluation
        model.load_state_dict(best_model_state)
        model.eval()
        
        with torch.inference_mode():
            _, test_probs = model(x, edge_indices, edge_weights)
        
        test_metrics = self._compute_metrics(test_probs, y_test_t, test_mask_t)
        logger.info("  Test: auc=%.4f, aupr=%.4f", test_metrics['auc'], test_metrics['aupr'])
        
        # All-gene predictions
        all_mask = torch.ones(n_genes, dtype=torch.bool, device=self.device)
        all_metrics = self._compute_metrics(test_probs, y_label_full_t, all_mask)
        
        predictions = test_probs.cpu().detach().numpy()
        
        # Cleanup
        del best_model_state, x, edge_indices, edge_weights
        
        return model, test_metrics, all_metrics, predictions
    
    def _compute_metrics(self, probs: torch.Tensor, labels: torch.Tensor,
                          mask: torch.Tensor) -> dict:
        """Compute AUC and AUPR for masked predictions."""
        probs_np = probs[mask].cpu().detach().numpy()
        labels_np = labels[mask].cpu().detach().numpy().astype(int)
        
        # Need positive and negative samples to compute AUC/AUPR
        if labels_np.size == 0 or labels_np.sum() == 0 or labels_np.sum() == len(labels_np):
            return {'auc': 0.0, 'aupr': 0.0, 'acc': 0.0}
        
        try:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
            aupr = average_precision_score(labels_np, probs_np[:, 1])
            preds = (probs_np[:, 1] > 0.5).astype(int)
            acc = (preds == labels_np).mean()
        except Exception:
            auc, aupr, acc = 0.0, 0.0, 0.0
        
        return {'auc': auc, 'aupr': aupr, 'acc': acc}
    
    def _adjacency_to_edge_index(self, adjacency_list: list) -> tuple:
        """
        Convert list of sparse adjacency matrices (scipy.sparse)
        to PyG edge_index format.
        
        Returns:
            edge_indices: List of (2 x n_edges) tensors
            edge_weights: List of (n_edges,) tensors
        """
        n_slices = len(adjacency_list)
        edge_indices = []
        edge_weights = []
        
        for s in range(n_slices):
            adj = adjacency_list[s].tocoo()
            rows = adj.row
            cols = adj.col
            values = adj.data
            
            edge_index = torch.LongTensor(np.vstack([rows, cols]))
            edge_weight = torch.FloatTensor(values)
            
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
            
            logger.info("  Graph slice %d: %d edges", s, len(rows))
        
        return edge_indices, edge_weights
    
    @staticmethod
    def _make_mask(n_genes: int, indices: np.ndarray) -> np.ndarray:
        """Create boolean mask array."""
        mask = np.zeros(n_genes, dtype=bool)
        mask[indices] = True
        return mask
    
    @staticmethod
    def _make_y(y_label: np.ndarray, indices: np.ndarray,
                n_genes: int) -> np.ndarray:
        """Create label array with only specified indices set."""
        y = np.zeros(n_genes, dtype=np.float32)
        y[indices] = y_label[indices]
        return y
