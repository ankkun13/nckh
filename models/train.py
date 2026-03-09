"""
train.py — Training Loop cho MODCANGNNModel

Features:
    - Full training loop với Focal Loss
    - Early stopping theo val_AUPR
    - Learning rate scheduler (CosineAnnealingLR)
    - Checkpoint lưu model tốt nhất
    - Logging chi tiết mỗi epoch
    - Hỗ trợ multi-subtype training

Sử dụng:
    python models/train.py --config configs/config.yaml --cancer LUAD --subtype 0
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml

from models.gnn import build_model_from_config
from models.focal_loss import FocalLoss
from models.utils import build_graph_data


def setup_logger(log_dir: str, cancer: str, subtype: int) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_{cancer}_sub{subtype}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("train")


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data, mask, device):
    """Tính AUROC và AUPR trên tập được chỉ định bởi mask."""
    model.eval()
    topo = getattr(data, "x_topo", None)
    logits = model(
        data.x,
        data.edge_index,
        topo_features=topo,
        edge_weight=data.edge_weight,
    )
    probs = torch.sigmoid(logits)[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, 0.0  # Không đủ điều kiện tính metrics

    auroc = roc_auc_score(labels, probs)
    aupr  = average_precision_score(labels, probs)
    return float(auroc), float(aupr)


# ── Training Loop ──────────────────────────────────────────────────────────────
def train_one_subtype(
    data,
    model,
    criterion,
    cfg: dict,
    checkpoint_dir: Path,
    logger: logging.Logger,
    subtype: int,
) -> dict:
    """Training loop cho một subtype. Trả về lịch sử metrics."""
    train_cfg = cfg["training"]
    device = next(model.parameters()).device

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler_type = train_cfg["scheduler"]["type"]
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["scheduler"]["T_max"],
            eta_min=1e-5,
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        scheduler = None

    # Early stopping setup
    es_cfg = train_cfg["early_stopping"]
    best_val_aupr   = -1.0
    best_epoch      = 0
    patience_counter = 0
    history = []

    logger.info(f"{'─'*60}")
    logger.info(f"[TRAIN] Bắt đầu training — Subtype {subtype}")
    logger.info(f"  Epochs: {train_cfg['epochs']}, LR: {train_cfg['learning_rate']}")
    logger.info(f"  Early stopping patience: {es_cfg['patience']}")
    logger.info(f"{'─'*60}")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        topo = getattr(data, "x_topo", None)
        logits = model(
            data.x,
            data.edge_index,
            topo_features=topo,
            edge_weight=data.edge_weight,
        )

        # Tính loss chỉ trên train nodes
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # Gradient clipping (ổn định training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Evaluate
        if epoch % 5 == 0 or epoch == 1:
            train_auroc, train_aupr = evaluate(model, data, data.train_mask, device)
            val_auroc,   val_aupr   = evaluate(model, data, data.val_mask,   device)

            lr_current = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:4d}/{train_cfg['epochs']} | "
                f"Loss={loss.item():.4f} | "
                f"Train AUROC={train_auroc:.3f} AUPR={train_aupr:.3f} | "
                f"Val AUROC={val_auroc:.3f} AUPR={val_aupr:.3f} | "
                f"LR={lr_current:.6f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_auroc": train_auroc,
                "train_aupr": train_aupr,
                "val_auroc": val_auroc,
                "val_aupr": val_aupr,
            })

            # Lưu best model
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_epoch = epoch
                patience_counter = 0
                checkpoint_path = checkpoint_dir / f"best_model_sub{subtype}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "val_aupr": val_aupr,
                    "model_config": model.get_config(),
                }, checkpoint_path)
            else:
                patience_counter += 5

            # Early stopping
            if es_cfg["enabled"] and patience_counter >= es_cfg["patience"]:
                logger.info(
                    f"[EARLY STOP] Epoch {epoch} | Best epoch: {best_epoch} | "
                    f"Best val_AUPR={best_val_aupr:.4f}"
                )
                break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_aupr": best_val_aupr,
    }


# ── Test Evaluation ────────────────────────────────────────────────────────────
def test_final_model(
    model,
    data,
    checkpoint_dir: Path,
    subtype: int,
    results_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Load best checkpoint và đánh giá trên test set."""
    checkpoint_path = checkpoint_dir / f"best_model_sub{subtype}.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"[TEST] Loaded best model từ epoch {ckpt['epoch']}")

    test_auroc, test_aupr = evaluate(model, data, data.test_mask, next(model.parameters()).device)
    logger.info(f"[TEST RESULT] AUROC={test_auroc:.4f} | AUPR={test_aupr:.4f}")

    # Lưu predictions
    model.eval()
    with torch.no_grad():
        topo = getattr(data, "x_topo", None)
        logits = model(data.x, data.edge_index, topo_features=topo, edge_weight=data.edge_weight)
        probs = torch.sigmoid(logits).cpu().numpy()

    gene_names = data.gene_names
    pred_df = pd.DataFrame({
        "gene": gene_names,
        "driver_probability": probs,
        "true_label": data.y.cpu().numpy(),
        "in_test": data.test_mask.cpu().numpy(),
    })
    pred_df = pred_df.sort_values("driver_probability", ascending=False)

    pred_file = results_dir / f"predictions_subtype{subtype}.csv"
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"[SAVE] Predictions: {pred_file}")
    logger.info(f"       Top 20 predicted drivers:")
    top20 = pred_df[pred_df["in_test"]].head(20)
    for _, row in top20.iterrows():
        logger.info(f"    {row['gene']:20s}  P(driver)={row['driver_probability']:.4f}  True={int(row['true_label'])}")

    return {"test_auroc": test_auroc, "test_aupr": test_aupr}


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train GNN — Driver Gene Identification")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--cancer",  default="LUAD", choices=["LUAD", "LUSC"])
    parser.add_argument("--subtype", default=None, type=int,
                        help="Subtype cụ thể cần train (None = train tất cả)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    labels_dir    = Path(cfg["paths"]["raw"]["labels"])
    fused_dir     = processed_dir / "fused_networks"
    log_dir       = cfg["paths"]["logs"]
    results_dir   = Path(cfg["paths"].get("results", "/workspace/results")) / cancer
    checkpoint_dir = Path(cfg["paths"].get("models", "/workspace/models/checkpoints")) / cancer

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device_str = cfg["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(processed_dir / "patient_splits.json") as f:
        splits = json.load(f)

    subtype_labels = pd.read_csv(processed_dir / "subtype_labels.csv")
    subtypes_to_train = (
        [args.subtype] if args.subtype is not None
        else sorted(subtype_labels["subtype"].unique())
    )

    all_results = {}

    for subtype in subtypes_to_train:
        logger = setup_logger(log_dir, cancer, subtype)
        logger.info("=" * 65)
        logger.info(f"  MODCAN-GNN Training — {cancer} | Subtype {subtype}")
        logger.info(f"  Device: {device}")
        logger.info("=" * 65)

        # Tải fused network
        fused_adj_file  = fused_dir / f"fused_adjacency_subtype{subtype}.npz"
        fused_gene_file = fused_dir / f"fused_genes_subtype{subtype}.txt"
        if not fused_adj_file.exists():
            logger.warning(f"[SKIP] Không tìm thấy fused network cho subtype {subtype}")
            continue

        adj = sp.load_npz(str(fused_adj_file))
        with open(fused_gene_file) as f:
            gene_list = [line.strip() for line in f if line.strip()]

        # Xây PyG Data object
        data = build_graph_data(
            fused_adj=adj,
            gene_list=gene_list,
            processed_dir=processed_dir,
            labels_dir=labels_dir,
            train_samples=splits["train"],
            val_samples=splits["val"],
            test_samples=splits["test"],
            model_cfg=cfg["model"],
            device=device,
            logger=logger,
        )

        # Khởi tạo model
        model = build_model_from_config(
            cfg,
            omics_dim=data.omics_dim,
            topo_dim=data.topo_dim,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Model params: {n_params:,}")

        # Focal Loss
        fl_cfg = cfg["model"]["focal_loss"]
        criterion = FocalLoss(alpha=fl_cfg["alpha"], gamma=fl_cfg["gamma"])

        # Train
        train_result = train_one_subtype(
            data, model, criterion, cfg, checkpoint_dir, logger, subtype
        )

        # Test
        test_result = test_final_model(
            model, data, checkpoint_dir, subtype, results_dir, logger
        )

        all_results[str(subtype)] = {**train_result, **test_result}

        # Lưu history
        history_df = pd.DataFrame(train_result["history"])
        history_df.to_csv(results_dir / f"training_history_sub{subtype}.csv", index=False)

    # Tóm tắt cuối
    with open(results_dir / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 65)
    logger.info("  HOÀN THÀNH TRAINING")
    for subtype, res in all_results.items():
        logger.info(
            f"  Subtype {subtype}: Test AUROC={res['test_auroc']:.4f}, "
            f"Test AUPR={res['test_aupr']:.4f}"
        )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
