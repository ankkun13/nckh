"""
Metrics Calculator for GNN Evaluation
======================================
Computes AUROC, AUPR, and other performance metrics.
Saves results to TSV files for reproducibility.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    accuracy_score, f1_score, classification_report
)

from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class MetricsCalculator:
    """Computes and aggregates performance metrics across CV folds."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def compute(self, results: dict) -> dict:
        """
        Compute aggregated metrics from training results.
        
        Args:
            results: Dictionary from GNNTrainer.train_with_cv().
        
        Returns:
            Dictionary of computed metrics.
        """
        logger.info("Computing evaluation metrics...")
        
        performance = results['performance_measures']
        gene_names = results['gene_names']
        gene_labels = results['gene_labels']
        predictions = results['predictions']
        
        # Per-fold metrics
        fold_metrics = []
        for pm in performance:
            fold_metrics.append({
                'cv': pm['cv'],
                'test_auc': pm['test_metrics']['auc'],
                'test_aupr': pm['test_metrics']['aupr'],
                'test_acc': pm['test_metrics']['acc'],
                'all_auc': pm['all_metrics']['auc'],
                'all_aupr': pm['all_metrics']['aupr'],
            })
        
        df_metrics = pd.DataFrame(fold_metrics)
        
        # Aggregate
        metrics = {
            'per_fold': df_metrics,
            'avg_test_auc': df_metrics['test_auc'].mean(),
            'std_test_auc': df_metrics['test_auc'].std(),
            'avg_test_aupr': df_metrics['test_aupr'].mean(),
            'std_test_aupr': df_metrics['test_aupr'].std(),
            'avg_test_acc': df_metrics['test_acc'].mean(),
            'best_fold': df_metrics['test_aupr'].idxmax(),
        }
        
        # Average predictions across folds
        avg_predictions = np.mean(predictions, axis=0)
        metrics['avg_predictions'] = avg_predictions
        
        # Gene ranking by driver probability
        driver_probs = avg_predictions[:, 1] if avg_predictions.ndim > 1 else avg_predictions
        gene_ranking = pd.DataFrame({
            'gene': gene_names,
            'driver_prob': driver_probs,
            'is_driver': gene_labels[:, 0],
            'is_labeled': gene_labels[:, 1]
        }).sort_values('driver_prob', ascending=False)
        
        metrics['gene_ranking'] = gene_ranking
        
        # Top predicted drivers
        top_k = 50
        top_genes = gene_ranking.head(top_k)
        n_true_drivers = top_genes['is_driver'].sum()
        logger.info("Top %d predicted drivers: %d are true CGC drivers (precision=%.2f)",
                     top_k, n_true_drivers, n_true_drivers / top_k)
        
        logger.info("Average Test AUC:  %.4f ± %.4f", metrics['avg_test_auc'], metrics['std_test_auc'])
        logger.info("Average Test AUPR: %.4f ± %.4f", metrics['avg_test_aupr'], metrics['std_test_aupr'])
        
        return metrics
    
    def save(self, metrics: dict, results_dir: str):
        """Save metrics to files."""
        os.makedirs(results_dir, exist_ok=True)
        
        # Save per-fold metrics
        metrics['per_fold'].to_csv(
            os.path.join(results_dir, 'performance_measures.tsv'),
            sep='\t', index=False
        )
        
        # Save gene ranking
        metrics['gene_ranking'].to_csv(
            os.path.join(results_dir, 'gene_ranking.tsv'),
            sep='\t', index=False
        )
        
        # Save summary
        with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
            f.write("GNN Cancer Driver Gene Identification — Results Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Average Test AUC:  {metrics['avg_test_auc']:.4f} ± {metrics['std_test_auc']:.4f}\n")
            f.write(f"Average Test AUPR: {metrics['avg_test_aupr']:.4f} ± {metrics['std_test_aupr']:.4f}\n")
            f.write(f"Average Test ACC:  {metrics['avg_test_acc']:.4f}\n")
            f.write(f"Best CV fold: {metrics['best_fold']}\n")
            f.write("\nTop 20 predicted driver genes:\n")
            f.write(metrics['gene_ranking'].head(20).to_string(index=False))
            f.write("\n")
        
        logger.info("Metrics saved to %s", results_dir)
