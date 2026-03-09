"""
build_ppi_from_local.py — Xây dựng PPI adjacency matrix từ file local STRING

Thay thế download_ppi.py khi đã có file local:
    /home/ankkun/Downloads/data/network/string_full_v12_0.7.txt

Format input: protein_1 \t protein_2 \t score (gene symbol, không cần mapping)

Sử dụng:
    python network_builder/build_ppi_from_local.py --config configs/config.yaml
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


def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"build_ppi_local_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("ppi_local")


def build_ppi_adjacency(
    ppi_df: pd.DataFrame,
    top_percentile: float,
    logger: logging.Logger,
) -> tuple[sp.csr_matrix, list[str]]:
    """
    Xây adjacency sparse matrix từ edge list PPI.

    Args:
        ppi_df:         DataFrame với cột (gene_a, gene_b, score)
        top_percentile: Chỉ giữ top X% cạnh (theo MODCAN: top 70% → lọc 30% thấp nhất)
    """
    # Lọc top percentile nếu cần
    if top_percentile < 100:
        threshold = ppi_df["score"].quantile(1 - top_percentile / 100)
        n_before = len(ppi_df)
        ppi_df = ppi_df[ppi_df["score"] >= threshold].copy()
        logger.info(f"  [FILTER] Top {top_percentile}%: {n_before:,} → {len(ppi_df):,} edges (score >= {threshold:.0f})")

    # Loại self-loops và duplicate
    ppi_df = ppi_df[ppi_df["gene_a"] != ppi_df["gene_b"]].copy()
    edge_key = ppi_df.apply(lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1)
    ppi_df   = ppi_df[~edge_key.duplicated()].copy()
    logger.info(f"  [DEDUP] Sau loại self-loop & duplicate: {len(ppi_df):,} edges")

    # Xây gene list và index
    genes    = sorted(set(ppi_df["gene_a"]) | set(ppi_df["gene_b"]))
    gene_idx = {g: i for i, g in enumerate(genes)}
    n        = len(genes)
    logger.info(f"  [GENES] {n:,} unique genes")

    rows = ppi_df["gene_a"].map(gene_idx).values
    cols = ppi_df["gene_b"].map(gene_idx).values
    # Normalize score → [0, 1]
    weights = (ppi_df["score"].values / 1000.0).astype(np.float32)

    # Vô hướng (thêm cả chiều ngược)
    rows_full    = np.concatenate([rows, cols])
    cols_full    = np.concatenate([cols, rows])
    weights_full = np.concatenate([weights, weights])

    adj = sp.csr_matrix(
        (weights_full, (rows_full, cols_full)),
        shape=(n, n),
        dtype=np.float32,
    )
    logger.info(f"  [ADJ] {n}×{n}, {adj.nnz:,} non-zeros")
    return adj, genes


def main():
    parser = argparse.ArgumentParser(description="Build PPI Adjacency from local STRING file")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ppi_file     = Path(cfg["paths"]["raw"]["local_ppi_filtered"])
    out_dir      = Path(cfg["paths"]["raw"]["ppi"])
    log_dir      = cfg["paths"]["logs"]
    top_pct      = cfg["data_acquisition"]["ppi"].get("top_percentile", 70)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("  PPI Build (Local) — STRING v12 (score ≥ 700)")
    logger.info("=" * 60)
    logger.info(f"  Input : {ppi_file}")
    logger.info(f"  Output: {out_dir}")

    # Đọc PPI
    logger.info("[LOAD] Đang đọc file PPI...")
    ppi_df = pd.read_csv(ppi_file, sep="\t")
    ppi_df.columns = ["gene_a", "gene_b", "score"]
    logger.info(f"  -> {len(ppi_df):,} edges, score range: [{ppi_df['score'].min()}, {ppi_df['score'].max()}]")

    # Build adjacency
    adj, genes = build_ppi_adjacency(ppi_df, top_pct, logger)

    # Lưu kết quả
    adj_file  = out_dir / "ppi_adjacency.npz"
    gene_file = out_dir / "ppi_genes.txt"
    edge_file = out_dir / "ppi_string_v12.csv"

    sp.save_npz(str(adj_file), adj)
    logger.info(f"[SAVE] {adj_file}")

    with open(gene_file, "w") as f:
        f.write("\n".join(genes))
    logger.info(f"[SAVE] {gene_file} ({len(genes):,} genes)")

    # Lưu top-percentile edge list
    ppi_df.to_csv(edge_file, index=False)
    logger.info(f"[SAVE] {edge_file}")

    summary = {
        "n_genes": len(genes),
        "n_edges": len(ppi_df),
        "min_score": int(ppi_df["score"].min()),
        "max_score": int(ppi_df["score"].max()),
        "top_percentile": top_pct,
    }
    with open(out_dir / "ppi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"  HOÀN THÀNH: {len(genes):,} genes, {len(ppi_df):,} edges")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
