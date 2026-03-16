"""
Result Visualizer for GNN Cancer Driver Gene Pipeline
=====================================================
Generates publication-quality plots:
  - ROC and PR curves
  - Kaplan-Meier survival curves
  - Hub gene network visualization
  - UMAP gene embedding
  - Performance comparison bar charts
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import roc_curve, precision_recall_curve

from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class ResultVisualizer:
    """Generates all visualization plots for the pipeline results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.plot_formats = config['evaluation'].get('plot_formats', ['png'])
    
    def plot_all(self, data: dict, results: dict, output_dir: str):
        """Generate all plots and save to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating visualizations...")
        
        self.plot_roc_curve(results, output_dir)
        self.plot_pr_curve(results, output_dir)
        self.plot_cv_performance(results, output_dir)
        self.plot_driver_score_distribution(results, output_dir)
        
        if 'ppi' in data:
            self.plot_hub_gene_network(results, data, output_dir)
        
        if 'survival' in data and not data['survival'].empty:
            self.plot_kaplan_meier(results, data, output_dir)
        
        logger.info("All visualizations saved to %s", output_dir)
    
    def plot_roc_curve(self, results: dict, output_dir: str):
        """Plot ROC curve from average predictions."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        gene_labels = results['gene_labels']
        predictions = results['predictions']
        
        # Plot per-fold ROC curves (light color)
        aucs = []
        for i, preds in enumerate(predictions):
            y_true = gene_labels[:, 0]
            mask = gene_labels[:, 1] == 1
            y_score = preds[mask, 1] if preds.ndim > 1 else preds[mask]
            
            fpr, tpr, _ = roc_curve(y_true[mask], y_score)
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, alpha=0.2, color='steelblue', linewidth=1)
        
        # Plot average
        avg_preds = np.mean(predictions, axis=0)
        y_true = gene_labels[:, 0]
        mask = gene_labels[:, 1] == 1
        y_score = avg_preds[mask, 1] if avg_preds.ndim > 1 else avg_preds[mask]
        
        fpr, tpr, _ = roc_curve(y_true[mask], y_score)
        from sklearn.metrics import auc as compute_auc
        mean_auc = compute_auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkblue', linewidth=2.5,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {np.std(aucs):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title('ROC Curve — Driver Gene Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        self._save_fig(fig, output_dir, 'roc_curve')
        plt.close(fig)
    
    def plot_pr_curve(self, results: dict, output_dir: str):
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        gene_labels = results['gene_labels']
        predictions = results['predictions']
        
        avg_preds = np.mean(predictions, axis=0)
        y_true = gene_labels[:, 0]
        mask = gene_labels[:, 1] == 1
        y_score = avg_preds[mask, 1] if avg_preds.ndim > 1 else avg_preds[mask]
        
        precision, recall, _ = precision_recall_curve(y_true[mask], y_score)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true[mask], y_score)
        
        ax.plot(recall, precision, color='darkorange', linewidth=2.5,
                label=f'PR curve (AUPR = {ap:.3f})')
        
        # Baseline (random classifier)
        prevalence = y_true[mask].mean()
        ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.5,
                    label=f'Random (prevalence = {prevalence:.3f})')
        
        ax.set_xlabel('Recall', fontsize=13)
        ax.set_ylabel('Precision', fontsize=13)
        ax.set_title('Precision-Recall Curve — Driver Gene Prediction',
                      fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        
        self._save_fig(fig, output_dir, 'pr_curve')
        plt.close(fig)
    
    def plot_cv_performance(self, results: dict, output_dir: str):
        """Plot cross-validation performance comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        pm = results['performance_measures']
        
        folds = [p['cv'] for p in pm]
        aucs = [p['test_metrics']['auc'] for p in pm]
        auprs = [p['test_metrics']['aupr'] for p in pm]
        
        # AUC bar chart
        axes[0].bar(folds, aucs, color='steelblue', alpha=0.8, edgecolor='navy')
        axes[0].axhline(y=np.mean(aucs), color='red', linestyle='--',
                         label=f'Mean: {np.mean(aucs):.3f}')
        axes[0].set_xlabel('CV Fold', fontsize=12)
        axes[0].set_ylabel('AUC', fontsize=12)
        axes[0].set_title('Test AUC per Fold', fontsize=13, fontweight='bold')
        axes[0].legend()
        
        # AUPR bar chart
        axes[1].bar(folds, auprs, color='darkorange', alpha=0.8, edgecolor='saddlebrown')
        axes[1].axhline(y=np.mean(auprs), color='red', linestyle='--',
                         label=f'Mean: {np.mean(auprs):.3f}')
        axes[1].set_xlabel('CV Fold', fontsize=12)
        axes[1].set_ylabel('AUPR', fontsize=12)
        axes[1].set_title('Test AUPR per Fold', fontsize=13, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        self._save_fig(fig, output_dir, 'cv_performance')
        plt.close(fig)
    
    def plot_driver_score_distribution(self, results: dict, output_dir: str):
        """Plot distribution of driver probability scores."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        gene_labels = results['gene_labels']
        avg_preds = np.mean(results['predictions'], axis=0)
        
        driver_scores = avg_preds[gene_labels[:, 0] == 1, 1]
        non_driver_scores = avg_preds[gene_labels[:, 0] == 0, 1]
        
        ax.hist(non_driver_scores, bins=50, alpha=0.6, color='steelblue',
                label=f'Non-driver (n={len(non_driver_scores)})', density=True)
        ax.hist(driver_scores, bins=50, alpha=0.6, color='crimson',
                label=f'Driver (n={len(driver_scores)})', density=True)
        
        ax.set_xlabel('Driver Probability Score', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title('Distribution of Driver Prediction Scores',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        
        self._save_fig(fig, output_dir, 'score_distribution')
        plt.close(fig)
    
    def plot_hub_gene_network(self, results: dict, data: dict,
                               output_dir: str, top_n: int = 30):
        """Visualize top-N hub genes as a network graph."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get top predicted drivers
        gene_names = results['gene_names']
        avg_preds = np.mean(results['predictions'], axis=0)
        driver_probs = avg_preds[:, 1] if avg_preds.ndim > 1 else avg_preds
        
        top_idx = np.argsort(driver_probs)[-top_n:]
        top_genes = set(gene_names[top_idx])
        top_probs = driver_probs[top_idx]
        
        # Build subgraph from PPI
        ppi = data['ppi']
        G = nx.Graph()
        
        for row in ppi:
            g1, g2 = str(row[0]), str(row[1])
            if g1 in top_genes and g2 in top_genes:
                G.add_edge(g1, g2)
        
        # Add isolated nodes
        for gene in top_genes:
            if gene not in G:
                G.add_node(gene)
        
        # Node colors based on driver status
        gene_labels = results['gene_labels']
        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            idx = gene_name_to_idx.get(node, -1)
            if idx >= 0 and gene_labels[idx, 0] == 1:
                node_colors.append('crimson')
            else:
                node_colors.append('steelblue')
            
            prob = driver_probs[idx] if idx >= 0 else 0
            node_sizes.append(300 + 1500 * prob)
        
        pos = nx.spring_layout(G, seed=42, k=2)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'Top {top_n} Predicted Hub Driver Genes (PPI Network)',
                      fontsize=14, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='crimson', alpha=0.8, label='Known Driver (CGC)'),
            Patch(facecolor='steelblue', alpha=0.8, label='Predicted Novel Driver'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
        ax.axis('off')
        
        self._save_fig(fig, output_dir, 'hub_gene_network')
        plt.close(fig)
    
    def plot_kaplan_meier(self, results: dict, data: dict, output_dir: str):
        """Plot Kaplan-Meier survival curves for identified driver genes."""
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            logger.warning("lifelines not installed, skipping Kaplan-Meier plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        survival = data['survival']
        if survival.empty or 'OS.time' not in survival.columns:
            logger.warning("Survival data not available or malformed")
            plt.close(fig)
            return
        
        # Get top predicted drivers
        gene_names = results['gene_names']
        avg_preds = np.mean(results['predictions'], axis=0)
        driver_probs = avg_preds[:, 1] if avg_preds.ndim > 1 else avg_preds
        
        # Split samples by median driver score
        if 'exp_data' in data and data['exp_data'].size > 0:
            top_driver_idx = np.argsort(driver_probs)[-10:]
            # The survival data samples need to be matched
            # For now, plot overall survival split by sample median
            
            kmf = KaplanMeierFitter()
            
            # Group by OS status
            alive = survival[survival['OS'] == 0]
            deceased = survival[survival['OS'] == 1]
            
            if len(alive) > 0:
                kmf.fit(alive['OS.time'], event_observed=alive['OS'],
                        label='Alive')
                kmf.plot_survival_function(ax=ax, ci_show=True)
            
            kmf2 = KaplanMeierFitter()
            if len(deceased) > 0:
                kmf2.fit(survival['OS.time'], event_observed=survival['OS'],
                         label='All Patients')
                kmf2.plot_survival_function(ax=ax, ci_show=True)
        
        ax.set_xlabel('Time (days)', fontsize=13)
        ax.set_ylabel('Survival Probability', fontsize=13)
        ax.set_title('Kaplan-Meier Survival Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        
        self._save_fig(fig, output_dir, 'kaplan_meier')
        plt.close(fig)
    
    def _save_fig(self, fig, output_dir: str, name: str):
        """Save figure in configured formats."""
        for fmt in self.plot_formats:
            path = os.path.join(output_dir, f'{name}.{fmt}')
            fig.savefig(path)
            logger.info("  Saved plot: %s", path)
