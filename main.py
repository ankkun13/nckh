"""
GNN Cancer Driver Gene Identification Pipeline — Main Orchestrator
==================================================================
Coordinates execution of all pipeline stages:
  1. Preprocess  — Load and normalize multi-omics data
  2. Cluster     — Patient subgroup clustering via spectral clustering
  3. Network     — Build robust co-association network (RF/MI)
  4. Train       — Train GNN model (GCN/GAT via PyG)
  5. Evaluate    — Compute metrics and generate visualizations

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --stage train --cancer TCGA-LUAD
    python main.py --config configs/config.yaml --epochs 500 --cv_folds 5
"""

import argparse
import os
import sys
import yaml
import time
from pathlib import Path

from utils.logger import setup_logger, log_stage


def load_config(config_path: str) -> dict:
    """Load and return YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with command-line arguments."""
    if args.cancer:
        config['data']['cancer_type'] = args.cancer
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.cv_folds:
        config['evaluation']['cv_folds'] = args.cv_folds
    if args.method:
        config['network']['method'] = args.method
    if args.device:
        config['training']['device'] = args.device
    return config


def run_preprocess(config: dict, logger):
    """Stage 1: Data loading and preprocessing."""
    log_stage(logger, "DATA LOADING & PREPROCESSING")
    
    from data_processing.data_loader import MultiOmicsDataLoader
    from data_processing.preprocessor import DataPreprocessor
    
    loader = MultiOmicsDataLoader(config)
    cancer_type = config['data']['cancer_type']
    raw_data = loader.load(cancer_type)
    
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.process(raw_data)
    
    log_stage(logger, "DATA LOADING & PREPROCESSING", "END")
    return processed_data


def run_cluster(config: dict, processed_data: dict, logger):
    """Stage 2: Patient subgroup clustering."""
    log_stage(logger, "PATIENT SUBGROUP CLUSTERING")
    
    from clustering.spectral_clustering import SpectralClusteringPipeline
    from clustering.feature_engineering import GeneFeatureExtractor
    
    clustering_pipeline = SpectralClusteringPipeline(config)
    cluster_labels = clustering_pipeline.fit_predict(processed_data)
    processed_data['cluster_labels'] = cluster_labels
    
    feature_extractor = GeneFeatureExtractor(config)
    gene_features = feature_extractor.extract(processed_data)
    processed_data['gene_features'] = gene_features
    
    log_stage(logger, "PATIENT SUBGROUP CLUSTERING", "END")
    return processed_data


def run_network(config: dict, processed_data: dict, logger):
    """Stage 3: Build robust co-association network (CORE IMPROVEMENT)."""
    log_stage(logger, "ROBUST CO-ASSOCIATION NETWORK (RF/MI)")
    
    from network_builder.robust_network import RobustCoAssociationNetwork
    
    network_builder = RobustCoAssociationNetwork(config)
    graph_data = network_builder.build(processed_data)
    processed_data['graph_data'] = graph_data
    
    log_stage(logger, "ROBUST CO-ASSOCIATION NETWORK (RF/MI)", "END")
    return processed_data


def run_train(config: dict, processed_data: dict, logger):
    """Stage 4: GNN training with cross-validation."""
    log_stage(logger, "GNN MODEL TRAINING")
    
    from models.trainer import GNNTrainer
    
    trainer = GNNTrainer(config)
    results = trainer.train_with_cv(processed_data)
    
    log_stage(logger, "GNN MODEL TRAINING", "END")
    return results


def run_evaluate(config: dict, processed_data: dict, train_results: dict, logger):
    """Stage 5: Evaluation and visualization."""
    log_stage(logger, "EVALUATION & VISUALIZATION")
    
    from evaluation.metrics import MetricsCalculator
    from evaluation.visualizer import ResultVisualizer
    
    results_dir = config['evaluation']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_calc = MetricsCalculator(config)
    metrics = metrics_calc.compute(train_results)
    metrics_calc.save(metrics, results_dir)
    
    visualizer = ResultVisualizer(config)
    visualizer.plot_all(processed_data, train_results, results_dir)
    
    log_stage(logger, "EVALUATION & VISUALIZATION", "END")
    return metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='GNN Cancer Driver Gene Identification Pipeline'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['preprocess', 'cluster', 'network', 'train', 'evaluate', 'all'],
                        help='Pipeline stage to run (default: all)')
    parser.add_argument('--cancer', type=str, default=None,
                        help='Cancer type override (e.g., TCGA-LUAD)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs override')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--cv_folds', type=int, default=None,
                        help='Number of cross-validation folds override')
    parser.add_argument('--method', type=str, default=None,
                        choices=['random_forest', 'mutual_information'],
                        help='Network construction method override')
    parser.add_argument('--device', type=str, default=None,
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device override')
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Setup logging
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        log_level=config['logging']['log_level']
    )
    
    cancer_type = config['data']['cancer_type']
    logger.info("Pipeline configuration loaded from: %s", args.config)
    logger.info("Cancer type: %s", cancer_type)
    logger.info("Stage: %s", args.stage)
    
    start_time = time.time()
    
    # Track data across stages
    processed_data = None
    train_results = None
    
    try:
        # Stage 1: Preprocess
        if args.stage in ['preprocess', 'all']:
            processed_data = run_preprocess(config, logger)
        
        # Stage 2: Cluster
        if args.stage in ['cluster', 'all']:
            if processed_data is None:
                processed_data = run_preprocess(config, logger)
            processed_data = run_cluster(config, processed_data, logger)
        
        # Stage 3: Network
        if args.stage in ['network', 'all']:
            if processed_data is None:
                processed_data = run_preprocess(config, logger)
                processed_data = run_cluster(config, processed_data, logger)
            processed_data = run_network(config, processed_data, logger)
        
        # Stage 4: Train
        if args.stage in ['train', 'all']:
            if processed_data is None:
                processed_data = run_preprocess(config, logger)
                processed_data = run_cluster(config, processed_data, logger)
                processed_data = run_network(config, processed_data, logger)
            train_results = run_train(config, processed_data, logger)
        
        # Stage 5: Evaluate
        if args.stage in ['evaluate', 'all']:
            if processed_data is None or train_results is None:
                processed_data = run_preprocess(config, logger)
                processed_data = run_cluster(config, processed_data, logger)
                processed_data = run_network(config, processed_data, logger)
                train_results = run_train(config, processed_data, logger)
            run_evaluate(config, processed_data, train_results, logger)
        
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 72)
        logger.info("Pipeline completed successfully in %.2f seconds", elapsed)
        logger.info("=" * 72)
        
    except Exception as e:
        logger.error("Pipeline failed with error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
