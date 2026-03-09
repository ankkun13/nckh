"""
utils.py (v2) — Data loading và graph utilities cho GNN training

Cập nhật để tích hợp:
    1. gene_labels.csv (từ preprocess.py v2) — nhãn thực từ CGC + non-driver
    2. CNV features (binary) như omics modality thứ 4
    3. Methylation gene summary (từ preprocess v2)
    4. Masks dựa trên labeled genes × patient split (semi-supervised)

Sử dụng:
    from models.utils import build_graph_data
    data = build_graph_data(fused_adj, gene_list, processed_dir, ...)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import Node2Vec


# ── Logger ─────────────────────────────────────────────────────────────────────
def _get_logger(name: str = "utils") -> logging.Logger:
    return logging.getLogger(name)


# =============================================================================
# MODULE 1: Load omics features cho từng gen
# =============================================================================
def load_omics_features(
    processed_dir: Path,
    gene_list: list[str],
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Tải và ghép EXP + Methylation + SNV (+ CNV nếu có) thành feature vector.

    Mỗi gen được đại diện bởi vector:
        [rna_mean, rna_std, rna_median, rna_max, rna_min,   (5 dims)
         meth_mean, meth_std,                               (2 dims)
         mutation_rate, mutation_count,                     (2 dims)
         cnv_amp_rate, cnv_del_rate]                        (2 dims, nếu có)

    Returns:
        Tensor shape (n_genes, n_features)
    """
    logger.info("[FEATURES] Xây dựng omics feature matrix...")
    feature_parts = []

    # ── RNA-seq ──────────────────────────────────────────────────────────────
    rnaseq_file = processed_dir / "rnaseq_normalized.csv.gz"
    if rnaseq_file.exists():
        rnaseq = pd.read_csv(rnaseq_file, index_col=0, compression="gzip")
        rnaseq = rnaseq.reindex(gene_list).fillna(0)
        rna_feat = pd.DataFrame({
            "rna_mean":   rnaseq.mean(axis=1),
            "rna_std":    rnaseq.std(axis=1).fillna(0),
            "rna_median": rnaseq.median(axis=1),
            "rna_max":    rnaseq.max(axis=1),
            "rna_min":    rnaseq.min(axis=1),
        }, index=gene_list)
        feature_parts.append(rna_feat)
        logger.info(f"  -> RNA-seq: {rna_feat.shape[1]} features")
    else:
        logger.warning("  [WARN] rnaseq_normalized.csv.gz không tìm thấy")

    # ── Methylation gene summary ──────────────────────────────────────────────
    meth_summary_file = processed_dir / "methylation_gene_summary.csv"
    if meth_summary_file.exists():
        meth_summary = pd.read_csv(meth_summary_file, index_col="gene_symbol")
        meth_summary = meth_summary.reindex(gene_list).fillna(0)
        feature_parts.append(meth_summary[["meth_mean", "meth_std"]])
        logger.info(f"  -> Methylation: 2 features")
    else:
        logger.warning("  [WARN] methylation_gene_summary.csv không tìm thấy")

    # ── SNV binary matrix ─────────────────────────────────────────────────────
    snv_file = processed_dir / "mutation_binary.csv"
    if snv_file.exists():
        snv = pd.read_csv(snv_file, index_col=0)
        snv = snv.reindex(gene_list).fillna(0)
        snv_feat = pd.DataFrame({
            "mutation_rate":  snv.mean(axis=1),
            "mutation_count": snv.sum(axis=1),
        }, index=gene_list)
        feature_parts.append(snv_feat)
        logger.info(f"  -> SNV: 2 features")
    else:
        logger.warning("  [WARN] mutation_binary.csv không tìm thấy")

    # ── CNV binary matrix (nếu có) ────────────────────────────────────────────
    cnv_file = processed_dir / "cnv_binary.csv"
    if cnv_file.exists():
        cnv = pd.read_csv(cnv_file, index_col=0)
        cnv = cnv.reindex(gene_list).fillna(0)
        # Phân biệt amplification (>0) và deletion (<0) nếu data cho phép
        # Với binary matrix (0/1), dùng rate đơn giản
        cnv_feat = pd.DataFrame({
            "cnv_rate": cnv.mean(axis=1),
        }, index=gene_list)
        feature_parts.append(cnv_feat)
        logger.info(f"  -> CNV: 1 feature")

    if not feature_parts:
        logger.warning("  [WARN] Không có omics features! Dùng random (test mode)")
        return torch.randn(len(gene_list), 8)

    X_df = pd.concat(feature_parts, axis=1).fillna(0)
    X = torch.tensor(X_df.values, dtype=torch.float32)
    logger.info(f"  -> Total omics features: {X.shape[1]} dims")
    return X


# =============================================================================
# MODULE 2: Load labels từ gene_labels.csv
# =============================================================================
def load_gene_labels(
    processed_dir: Path,
    gene_list: list[str],
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """
    Tải nhãn driver/non-driver từ gene_labels.csv.

    Returns:
        y:             Tensor (n_genes,) — 1.0=driver, 0.0=non-driver, -1=unknown
        labeled_mask:  BoolTensor (n_genes,) — True nếu gen có nhãn
    """
    label_file = processed_dir / "gene_labels.csv"

    y = torch.full((len(gene_list),), fill_value=-1.0, dtype=torch.float32)
    labeled_mask = torch.zeros(len(gene_list), dtype=torch.bool)

    if not label_file.exists():
        logger.warning(f"  [WARN] gene_labels.csv không tìm thấy: {label_file}")
        # Fallback: thử dùng driver_gene_set.txt cũ
        driver_file = processed_dir.parent.parent / "raw" / "labels" / "driver_gene_set.txt"
        if driver_file.exists():
            with open(driver_file) as f:
                driver_set = {line.strip() for line in f if line.strip()}
            gene_idx = {g: i for i, g in enumerate(gene_list)}
            for gene, idx in gene_idx.items():
                if gene in driver_set:
                    y[idx] = 1.0
                    labeled_mask[idx] = True
        return y, labeled_mask

    label_df = pd.read_csv(label_file)
    gene_idx = {g: i for i, g in enumerate(gene_list)}

    n_driver = 0
    n_non_driver = 0
    for _, row in label_df.iterrows():
        gene = row["gene"]
        if gene in gene_idx:
            idx = gene_idx[gene]
            y[idx] = float(row["label"])
            labeled_mask[idx] = True
            if row["label"] == 1:
                n_driver += 1
            else:
                n_non_driver += 1

    total_labeled = labeled_mask.sum().item()
    logger.info(
        f"  -> Labels: {total_labeled} labeled genes | "
        f"driver={n_driver} | non-driver={n_non_driver} | "
        f"unlabeled={len(gene_list) - total_labeled}"
    )
    return y, labeled_mask


# =============================================================================
# MODULE 3: Node2Vec topology embeddings
# =============================================================================
def learn_node2vec_embeddings(
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int = 64,
    walk_length: int = 30,
    context_size: int = 10,
    walks_per_node: int = 20,
    num_negative_samples: int = 1,
    device: torch.device = torch.device("cpu"),
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """Học Node2Vec embeddings trên đồ thị."""
    if logger:
        logger.info(f"[NODE2VEC] dim={embedding_dim}, walks={walks_per_node}, len={walk_length}")

    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=1.0, q=1.0, sparse=True,
    ).to(device)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    loader    = model.loader(batch_size=128, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(1, 51):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if logger and epoch % 10 == 0:
            logger.info(f"  Node2Vec Epoch {epoch:3d}: loss={total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        emb = model()
    if logger:
        logger.info(f"  -> Completed: {emb.shape}")
    return emb.cpu()


# =============================================================================
# MODULE 4: Build PyG Data
# =============================================================================
def build_graph_data(
    fused_adj: sp.csr_matrix,
    gene_list: list[str],
    processed_dir: Path,
    train_samples: list[str],
    val_samples: list[str],
    test_samples: list[str],
    model_cfg: dict,
    device: torch.device,
    logger: logging.Logger,
) -> Data:
    """
    Xây dựng đầy đủ PyG Data object từ fused adjacency matrix.

    Masks: Gen có đột biến trong tập train/val/test samples → thuộc mask đó.
    Label: Dùng gene_labels.csv (driver=1, non-driver=0, unknown=-1).
    """
    n_genes = len(gene_list)
    logger.info(f"\n[GRAPH] Xây dựng PyG Data ({n_genes:,} genes)...")

    # Edge index và weight
    edge_index, edge_weight = from_scipy_sparse_matrix(fused_adj)
    logger.info(f"  -> Edges: {edge_index.shape[1]:,}")

    # Omics features
    x_omics = load_omics_features(processed_dir, gene_list, logger)
    omics_dim = x_omics.shape[1]

    # Labels
    y, labeled_mask = load_gene_labels(processed_dir, gene_list, logger)

    # Node2Vec topology (nếu bật representation_separation)
    topo_dim = model_cfg.get("node2vec", {}).get("embedding_dim", 64)
    if model_cfg.get("representation_separation", True):
        x_topo = learn_node2vec_embeddings(
            edge_index=edge_index,
            num_nodes=n_genes,
            embedding_dim=topo_dim,
            walk_length=model_cfg["node2vec"].get("walk_length", 30),
            context_size=model_cfg["node2vec"].get("context_size", 10),
            walks_per_node=model_cfg["node2vec"].get("num_walks", 20),
            device=device,
            logger=logger,
        )
    else:
        x_topo = None
        topo_dim = 0

    # Masks — dựa trên SNV mutation presence × patient split
    snv_file = processed_dir / "mutation_binary.csv"
    gene_idx = {g: i for i, g in enumerate(gene_list)}

    def get_mutated_gene_mask(samples: list[str]) -> torch.BoolTensor:
        """Gen nào bị đột biến ở ít nhất 1 sample trong tập này."""
        if not snv_file.exists():
            return torch.zeros(n_genes, dtype=torch.bool)
        snv = pd.read_csv(snv_file, index_col=0)
        avail = [s for s in samples if s in snv.columns]
        if not avail:
            return torch.zeros(n_genes, dtype=torch.bool)
        sub = snv.reindex(gene_list).reindex(columns=avail).fillna(0)
        mutated = (sub.sum(axis=1) > 0).values
        return torch.tensor(mutated, dtype=torch.bool)

    train_mut_mask = get_mutated_gene_mask(train_samples)
    val_mut_mask   = get_mutated_gene_mask(val_samples)
    test_mut_mask  = get_mutated_gene_mask(test_samples)

    # Kết hợp với labeled_mask: train/val/test mask = labeled & mutation mask
    train_mask = labeled_mask & train_mut_mask
    val_mask   = labeled_mask & val_mut_mask
    test_mask  = labeled_mask & test_mut_mask

    # Fallback: nếu masks quá ít (< 5 gen), mở rộng sang tất cả labeled
    if train_mask.sum() < 5:
        logger.warning("[WARN] Train mask quá nhỏ → dùng tất cả labeled genes")
        n = n_genes
        perm = torch.where(labeled_mask)[0]
        perm = perm[torch.randperm(len(perm))]
        n_tr = int(0.7 * len(perm))
        n_va = int(0.15 * len(perm))
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[perm[:n_tr]] = True
        val_mask[perm[n_tr:n_tr + n_va]] = True
        test_mask[perm[n_tr + n_va:]] = True

    logger.info(
        f"  -> Masks (labeled only): "
        f"train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}"
    )

    # Tạo Data
    data = Data(
        x=x_omics.to(device),
        edge_index=edge_index.to(device),
        edge_weight=edge_weight.to(device).float(),
        y=y.to(device),
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        labeled_mask=labeled_mask.to(device),
        gene_names=gene_list,
        num_nodes=n_genes,
    )
    if x_topo is not None:
        data.x_topo = x_topo.to(device)
    data.omics_dim = omics_dim
    data.topo_dim  = topo_dim

    return data
