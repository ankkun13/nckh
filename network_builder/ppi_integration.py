"""
ppi_integration.py — Kết hợp (Fuse) Co-Association Network với PPI STRING

Theo MODCAN paper: mạng đồng liên kết vi phân cần được kết hợp với
mạng tương tác vật lý (PPI) từ STRING database để tạo ra đồ thị phong phú hơn.

Phương pháp fuse:
    - Union: A_fused = w1 * A_coassoc + w2 * A_ppi (mặc định)
    - Intersection: chỉ giữ cạnh tồn tại trong CẢ HAI mạng

Đầu ra: Ma trận kề tổng hợp đã align các gene nodes.

Sử dụng:
    python network_builder/ppi_integration.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def setup_logger(log_dir: str, name: str = "ppi_integration") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(name)


def load_ppi(ppi_dir: Path, logger: logging.Logger) -> tuple[sp.csr_matrix, list[str]]:
    """Tải PPI adjacency matrix và gene list từ download_ppi.py output."""
    adj_file  = ppi_dir / "ppi_adjacency.npz"
    gene_file = ppi_dir / "ppi_genes.txt"

    adj = sp.load_npz(str(adj_file))
    with open(gene_file) as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info(f"[PPI] Tải xong: {adj.shape[0]:,} × {adj.shape[1]:,}, {adj.nnz:,} edges")
    return adj, genes


def load_coassociation(
    coassoc_dir: Path, subtype: int, logger: logging.Logger
) -> tuple[sp.csr_matrix, list[str]]:
    """Tải co-association adjacency và gene list cho một subtype."""
    adj_file  = coassoc_dir / f"coassoc_adjacency_subtype{subtype}.npz"
    gene_file = coassoc_dir / f"coassoc_genes_subtype{subtype}.txt"

    if not adj_file.exists():
        raise FileNotFoundError(f"Không tìm thấy: {adj_file}")

    adj = sp.load_npz(str(adj_file))
    with open(gene_file) as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info(f"[COASSOC] Subtype {subtype}: {adj.shape}, {adj.nnz:,} edges")
    return adj, genes


def align_to_common_genes(
    adj1: sp.csr_matrix,
    genes1: list[str],
    adj2: sp.csr_matrix,
    genes2: list[str],
    logger: logging.Logger,
) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
    """
    Align hai adjacency matrices về cùng tập gene (intersection).
    Buộc cả hai matrix có cùng kích thước và thứ tự gene.
    """
    common_genes = sorted(set(genes1) & set(genes2))
    logger.info(
        f"  [ALIGN] Genes1={len(genes1):,}, Genes2={len(genes2):,}, "
        f"Chung={len(common_genes):,}"
    )

    def reindex_matrix(adj: sp.csr_matrix, genes: list[str], target: list[str]) -> sp.csr_matrix:
        gene_idx = {g: i for i, g in enumerate(genes)}
        target_idx = {g: i for i, g in enumerate(target)}
        n = len(target)

        cx = adj.tocoo()
        new_rows, new_cols, new_data = [], [], []
        for r, c, d in zip(cx.row, cx.col, cx.data):
            gr, gc = genes[r], genes[c]
            if gr in target_idx and gc in target_idx:
                new_rows.append(target_idx[gr])
                new_cols.append(target_idx[gc])
                new_data.append(d)

        return sp.csr_matrix(
            (new_data, (new_rows, new_cols)),
            shape=(n, n),
            dtype=np.float32,
        )

    adj1_aligned = reindex_matrix(adj1, genes1, common_genes)
    adj2_aligned = reindex_matrix(adj2, genes2, common_genes)
    return adj1_aligned, adj2_aligned, common_genes


def fuse_networks(
    adj_coassoc: sp.csr_matrix,
    adj_ppi: sp.csr_matrix,
    method: str,
    w_coassoc: float,
    w_ppi: float,
    logger: logging.Logger,
) -> sp.csr_matrix:
    """
    Kết hợp hai mạng thành một adjacency matrix tổng hợp.

    Args:
        method: "union" hoặc "intersection"
        w_coassoc: Trọng số của co-association network
        w_ppi: Trọng số của PPI network
    """
    # Chuẩn hoá từng mạng về [0, 1]
    def normalize_sparse(adj: sp.csr_matrix) -> sp.csr_matrix:
        max_val = adj.max()
        if max_val > 0:
            return adj / max_val
        return adj

    adj_coassoc_norm = normalize_sparse(adj_coassoc)
    adj_ppi_norm     = normalize_sparse(adj_ppi)

    if method == "union":
        # Cộng có trọng số — giữ cạnh từ BẤT KỲ mạng nào
        fused = w_coassoc * adj_coassoc_norm + w_ppi * adj_ppi_norm
        logger.info(f"  [FUSE] Union: {adj_coassoc_norm.nnz} + {adj_ppi_norm.nnz} → {fused.nnz} edges")
    elif method == "intersection":
        # Chỉ giữ cạnh tồn tại trong CẢ HAI mạng
        mask_coassoc = adj_coassoc_norm.astype(bool)
        mask_ppi     = adj_ppi_norm.astype(bool)
        intersection_mask = mask_coassoc.multiply(mask_ppi)  # element-wise AND
        fused = intersection_mask.multiply(
            w_coassoc * adj_coassoc_norm + w_ppi * adj_ppi_norm
        )
        logger.info(f"  [FUSE] Intersection: {intersection_mask.nnz} edges")
    else:
        raise ValueError(f"Phương pháp fuse không hợp lệ: {method}")

    # Đảm bảo đồ thị vô hướng và không có self-loop
    fused = fused.maximum(fused.T)
    fused.setdiag(0)
    fused.eliminate_zeros()

    return fused.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="PPI Integration — Fuse co-association with PPI")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD", choices=["LUAD", "LUSC"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    ppi_dir       = Path(cfg["paths"]["raw"]["ppi"])
    coassoc_dir   = processed_dir / "coassociation_networks"
    log_dir       = cfg["paths"]["logs"]
    fused_dir     = processed_dir / "fused_networks"
    fused_dir.mkdir(parents=True, exist_ok=True)

    fusion_cfg = cfg["network_builder"]["ppi_fusion"]
    fuse_method = fusion_cfg["method"]
    w_coassoc   = fusion_cfg["coassoc_weight"]
    w_ppi       = fusion_cfg["ppi_weight"]

    logger = setup_logger(log_dir, f"ppi_integration_{cancer}")
    logger.info("=" * 65)
    logger.info(f"  PPI Fusion — {cancer} | Method: {fuse_method}")
    logger.info("=" * 65)

    # Tải PPI network
    adj_ppi, ppi_genes = load_ppi(ppi_dir, logger)

    # Xử lý từng subtype
    subtype_labels = pd.read_csv(processed_dir / "subtype_labels.csv")
    subtypes = sorted(subtype_labels["subtype"].unique())
    summary = {}

    for subtype in subtypes:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[SUBTYPE {subtype}]")

        try:
            adj_coassoc, coassoc_genes = load_coassociation(coassoc_dir, subtype, logger)
        except FileNotFoundError as e:
            logger.warning(f"  [SKIP] {e}")
            continue

        # Align gene sets
        adj_coassoc_a, adj_ppi_a, common_genes = align_to_common_genes(
            adj_coassoc, coassoc_genes,
            adj_ppi, ppi_genes,
            logger,
        )

        # Fuse
        adj_fused = fuse_networks(
            adj_coassoc_a, adj_ppi_a,
            method=fuse_method,
            w_coassoc=w_coassoc,
            w_ppi=w_ppi,
            logger=logger,
        )

        # Lưu kết quả
        fused_adj_file  = fused_dir / f"fused_adjacency_subtype{subtype}.npz"
        fused_gene_file = fused_dir / f"fused_genes_subtype{subtype}.txt"

        sp.save_npz(str(fused_adj_file), adj_fused)
        with open(fused_gene_file, "w") as f:
            f.write("\n".join(common_genes))

        n_edges = adj_fused.nnz // 2
        density = n_edges / (len(common_genes) * (len(common_genes) - 1) / 2 + 1)
        summary[str(subtype)] = {
            "n_genes": len(common_genes),
            "n_edges": n_edges,
            "density": float(density),
        }
        logger.info(
            f"  -> Fused: {len(common_genes):,} genes, {n_edges:,} edges, "
            f"density={density:.5f}"
        )
        logger.info(f"  -> Đã lưu: {fused_adj_file}")

    with open(fused_dir / "fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"  HOÀN THÀNH: PPI Fusion cho {len(subtypes)} subtypes")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
