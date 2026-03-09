"""
download_ppi.py — Tải mạng lưới PPI từ STRING database v12.0

Tải về:
    - File tương tác protein (9606.protein.links.v12.0.txt.gz)
    - File ánh xạ protein ID → gene symbol (9606.protein.info.v12.0.txt.gz)

Đầu ra:
    - data/raw/ppi/ppi_string_v12.csv  — Bảng cạnh (gene_a, gene_b, combined_score)
    - data/raw/ppi/ppi_adjacency.npz   — Ma trận kề thưa (sparse)

Sử dụng:
    python data_acquisition/download_ppi.py --config configs/config.yaml
"""

import os
import gzip
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from tqdm import tqdm
import yaml

# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logger(log_dir: str, name: str = "download_ppi") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


# ── STRING download URLs ───────────────────────────────────────────────────────
STRING_BASE = "https://stringdb-downloads.org/download"

STRING_LINKS_URL = (
    f"{STRING_BASE}/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    f"{STRING_BASE}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)


def download_file(url: str, dest_path: Path, logger: logging.Logger) -> Path:
    """Tải file từ URL, hiển thị progress bar. Bỏ qua nếu đã tồn tại."""
    if dest_path.exists():
        logger.info(f"[SKIP] File đã tồn tại: {dest_path}")
        return dest_path

    logger.info(f"[DOWNLOAD] {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info(f"  -> Đã lưu: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return dest_path


def load_protein_info(info_gz_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Đọc file ánh xạ protein_id → gene_symbol."""
    logger.info(f"[PARSE] Đọc protein info: {info_gz_path}")
    with gzip.open(info_gz_path, "rt") as f:
        df = pd.read_csv(f, sep="\t")

    # Cột cần: '#string_protein_id', 'preferred_name'
    rename_map = {
        "#string_protein_id": "protein_id",
        "preferred_name": "gene_symbol",
    }
    df = df.rename(columns=rename_map)[["protein_id", "gene_symbol"]]
    logger.info(f"  -> {len(df):,} proteins/genes được ánh xạ")
    return df


def load_string_links(
    links_gz_path: Path,
    prot_to_gene: dict,
    min_score: int,
    top_percentile: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Đọc file tương tác STRING và lọc theo ngưỡng.

    Args:
        min_score:      Ngưỡng confidence tối thiểu (0–1000). default=400
        top_percentile: Chỉ giữ top X% cạnh theo combined_score. default=70
    """
    logger.info(f"[PARSE] Đọc STRING links: {links_gz_path}")
    with gzip.open(links_gz_path, "rt") as f:
        df = pd.read_csv(f, sep=" ")

    logger.info(f"  -> Tổng cạnh trước lọc: {len(df):,}")

    # 1. Lọc theo min_score
    df = df[df["combined_score"] >= min_score].copy()
    logger.info(f"  -> Sau lọc min_score>={min_score}: {len(df):,} cạnh")

    # 2. Ánh xạ protein_id → gene_symbol
    df["gene_a"] = df["protein1"].map(prot_to_gene)
    df["gene_b"] = df["protein2"].map(prot_to_gene)
    df = df.dropna(subset=["gene_a", "gene_b"])

    # 3. Chỉ giữ top percentile (theo MODCAN paper: top 70%)
    threshold = df["combined_score"].quantile(1 - top_percentile / 100)
    df = df[df["combined_score"] >= threshold].copy()
    logger.info(
        f"  -> Sau lọc top {top_percentile}%: {len(df):,} cạnh "
        f"(score >= {threshold:.0f})"
    )

    # 4. Loại bỏ self-loops và duplicate edges
    df = df[df["gene_a"] != df["gene_b"]].copy()
    df["edge_key"] = df.apply(
        lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1
    )
    df = df.drop_duplicates(subset=["edge_key"])
    df = df[["gene_a", "gene_b", "combined_score"]].reset_index(drop=True)

    logger.info(f"  -> Sau loại self-loop & duplicate: {len(df):,} cạnh duy nhất")
    logger.info(f"  -> Số gen unique: {len(set(df['gene_a']) | set(df['gene_b'])):,}")
    return df


def build_adjacency_matrix(
    ppi_df: pd.DataFrame, logger: logging.Logger
) -> tuple[sp.csr_matrix, list]:
    """Xây dựng ma trận kề thưa từ bảng cạnh PPI."""
    genes = sorted(set(ppi_df["gene_a"].tolist() + ppi_df["gene_b"].tolist()))
    gene_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    rows = ppi_df["gene_a"].map(gene_idx).values
    cols = ppi_df["gene_b"].map(gene_idx).values
    # Chuẩn hoá trọng số về [0, 1]
    weights = (ppi_df["combined_score"].values / 1000.0).astype(np.float32)

    # Đồ thị vô hướng: thêm cả chiều ngược
    rows_full = np.concatenate([rows, cols])
    cols_full = np.concatenate([cols, rows])
    weights_full = np.concatenate([weights, weights])

    adj = sp.csr_matrix(
        (weights_full, (rows_full, cols_full)),
        shape=(n, n),
        dtype=np.float32,
    )
    logger.info(f"  -> Ma trận kề: {n}x{n}, {adj.nnz:,} phần tử non-zero")
    return adj, genes


def main():
    parser = argparse.ArgumentParser(description="Tải PPI từ STRING v12")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Đường dẫn config.yaml"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ppi_cfg = cfg["data_acquisition"]["ppi"]
    out_dir = Path(cfg["paths"]["raw"]["ppi"])
    log_dir = cfg["paths"]["logs"]
    min_score = ppi_cfg.get("min_score", 400)
    top_pct = ppi_cfg.get("top_percentile", 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, "download_ppi")

    logger.info("=" * 60)
    logger.info("  STRING PPI Download — MODCAN-GNN Pipeline")
    logger.info("=" * 60)
    logger.info(f"Min score        : {min_score}")
    logger.info(f"Top percentile   : {top_pct}%")
    logger.info(f"Output directory : {out_dir}")

    # ── Bước 1: Tải files ────────────────────────────────────────────────────
    links_gz = download_file(STRING_LINKS_URL, out_dir / "9606.protein.links.v12.gz", logger)
    info_gz  = download_file(STRING_INFO_URL,  out_dir / "9606.protein.info.v12.gz",  logger)

    # ── Bước 2: Ánh xạ protein → gene ───────────────────────────────────────
    prot_info = load_protein_info(info_gz, logger)
    prot_to_gene = dict(zip(prot_info["protein_id"], prot_info["gene_symbol"]))

    # ── Bước 3: Đọc và lọc cạnh ─────────────────────────────────────────────
    ppi_df = load_string_links(links_gz, prot_to_gene, min_score, top_pct, logger)

    # ── Bước 4: Lưu edge list ────────────────────────────────────────────────
    edge_file = out_dir / "ppi_string_v12.csv"
    ppi_df.to_csv(edge_file, index=False)
    logger.info(f"[SAVE] Edge list: {edge_file}")

    # ── Bước 5: Xây dựng và lưu adjacency matrix ────────────────────────────
    adj, genes = build_adjacency_matrix(ppi_df, logger)
    adj_file = out_dir / "ppi_adjacency.npz"
    sp.save_npz(str(adj_file), adj)
    logger.info(f"[SAVE] Adjacency matrix (sparse): {adj_file}")

    # Lưu danh sách gene theo thứ tự index
    gene_file = out_dir / "ppi_genes.txt"
    with open(gene_file, "w") as f:
        f.write("\n".join(genes))
    logger.info(f"[SAVE] Gene list: {gene_file} ({len(genes):,} genes)")

    logger.info("=" * 60)
    logger.info("  HOÀN THÀNH: STRING PPI download & preprocessing")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
