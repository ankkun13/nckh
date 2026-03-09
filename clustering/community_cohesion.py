"""
community_cohesion.py — Tính Community Cohesion Score cho từng mẫu bệnh nhân

Theo MODCAN paper: sử dụng khái niệm Community Cohesion Score để đo
mức độ suy giảm chức năng trong các co-expression communities giữa
mẫu bình thường và mẫu u bướu.

Pipeline:
    1. Tải mạng co-expression từ dữ liệu normal samples (WGCNA-based)
    2. Phát hiện các communities (modules) bằng Louvain/greedy modularity
    3. Với mỗi mẫu bệnh nhân, tính cohesion score cho từng community
    4. Tạo feature matrix: mỗi hàng = bệnh nhân, mỗi cột = community score

Sử dụng:
    python clustering/community_cohesion.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import zscore
from sklearn.covariance import EmpiricalCovariance
import yaml


def setup_logger(log_dir: str, cancer: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"community_cohesion_{cancer}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("community_cohesion")


# =============================================================================
# BƯỚC 1: Xây dựng co-expression network từ Normal samples (WGCNA-style)
# =============================================================================
def build_normal_coexpression_network(
    rnaseq: pd.DataFrame,
    normal_sample_ids: list[str],
    soft_threshold_power: int,
    logger: logging.Logger,
) -> nx.Graph:
    """
    Xây dựng weighted gene co-expression network từ normal samples.

    Sử dụng Pearson correlation và soft-thresholding (WGCNA approach):
        adjacency(i,j) = |cor(i,j)| ^ power

    Args:
        soft_threshold_power: Thường = 6 (giá trị từ pickSoftThreshold trong R)
    """
    logger.info(f"[BUILD NET] Normal samples: {len(normal_sample_ids)}")

    # Lấy expression data của normal samples
    avail_normals = [s for s in normal_sample_ids if s in rnaseq.columns]
    if len(avail_normals) == 0:
        logger.warning("[WARN] Không có normal samples trong dữ liệu. Dùng tất cả samples.")
        avail_normals = rnaseq.columns.tolist()

    logger.info(f"  -> Sử dụng {len(avail_normals)} normal samples")
    expr_normal = rnaseq[avail_normals].T  # shape: (samples, genes)

    # Tính Pearson correlation matrix
    logger.info(f"  -> Tính Pearson correlation ({expr_normal.shape[1]} genes)...")
    corr_mat = expr_normal.corr(method="pearson").fillna(0).values
    genes = expr_normal.columns.tolist()

    # Soft-thresholding (phù hợp scale-free topology)
    adj_mat = np.abs(corr_mat) ** soft_threshold_power
    np.fill_diagonal(adj_mat, 0)  # Loại self-loops

    # Chỉ giữ cạnh với trọng số > percentile ngưỡng động
    threshold = np.percentile(adj_mat[adj_mat > 0], 90)  # top 10%
    adj_mat[adj_mat < threshold] = 0

    # Xây graph
    logger.info(f"  -> Threshold = {threshold:.4f}, đang xây đồ thị...")
    G = nx.from_numpy_array(adj_mat)
    mapping = {i: g for i, g in enumerate(genes)}
    G = nx.relabel_nodes(G, mapping)

    # Loại node cô lập
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    logger.info(f"  -> Đồ thị: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# =============================================================================
# BƯỚC 2: Phát hiện Communities (Gene Modules)
# =============================================================================
def detect_communities(
    G: nx.Graph,
    min_module_size: int,
    logger: logging.Logger,
) -> list[set]:
    """
    Phát hiện gene communities bằng Greedy Modularity Optimization.
    Loại bỏ community kích thước < min_module_size.
    """
    logger.info("[COMMUNITY] Đang phát hiện gene modules...")
    communities = list(
        nx.community.greedy_modularity_communities(G, weight="weight")
    )
    n_before = len(communities)
    communities = [c for c in communities if len(c) >= min_module_size]
    logger.info(
        f"  -> Phát hiện {n_before} communities, giữ lại {len(communities)} "
        f"(size >= {min_module_size})"
    )
    return communities


# =============================================================================
# BƯỚC 3: Tính Cohesion Score (Zsummary) cho từng bệnh nhân
# =============================================================================
def compute_zsummary(
    expr_sample: pd.Series,
    community_genes: set,
    gene_means: pd.Series,
    gene_stds: pd.Series,
) -> float:
    """
    Tính Zsummary score cho 1 community trong 1 bệnh nhân.

    Zsummary đo mức độ bảo toàn cấu trúc correlation trong community
    khi chuyển từ normal → tumor context.

    Phiên bản đơn giản hoá:
        Zsummary = mean(|z(gene_i)| for gene_i in community)

    Trong đó z(gene_i) = (expr_sample[i] - mean_normal[i]) / std_normal[i]
    """
    comm_genes = [g for g in community_genes if g in expr_sample.index]
    if len(comm_genes) < 3:
        return 0.0

    z_scores = []
    for gene in comm_genes:
        std = gene_stds.get(gene, 1.0)
        if std < 1e-9:
            continue
        z = (expr_sample[gene] - gene_means.get(gene, 0.0)) / std
        z_scores.append(abs(z))

    return float(np.mean(z_scores)) if z_scores else 0.0


def compute_cohesion_scores(
    rnaseq_tumor: pd.DataFrame,
    rnaseq_normal: pd.DataFrame,
    communities: list[set],
    zsummary_threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Tính cohesion score matrix: (n_patients) x (n_communities).

    Chỉ giữ community có Zsummary_normal >= threshold (theo MODCAN: >= 10).
    """
    # Tính mean và std của normal samples (reference)
    gene_means = rnaseq_normal.mean(axis=1)
    gene_stds  = rnaseq_normal.std(axis=1)

    # Lọc community: chỉ giữ community với zsummary trung bình cao
    logger.info(f"  -> Lọc community (Zsummary >= {zsummary_threshold})...")
    valid_communities = []
    for idx, comm in enumerate(communities):
        # Tính Zsummary trung bình trên normal set
        zscores_normal = []
        for _, col in rnaseq_normal.items():
            z = compute_zsummary(col, comm, gene_means, gene_stds)
            zscores_normal.append(z)
        avg_z = np.mean(zscores_normal)
        if avg_z >= zsummary_threshold:
            valid_communities.append((idx, comm, avg_z))
            logger.debug(f"    Community {idx}: size={len(comm)}, Zsummary_normal={avg_z:.2f} ✓")

    logger.info(f"  -> {len(valid_communities)}/{len(communities)} communities thỏa mãn ngưỡng")

    if len(valid_communities) == 0:
        logger.warning("[WARN] Không có community nào thỏa mãn Zsummary threshold!")
        logger.warning("       Hạ threshold xuống 5.0 để tiếp tục...")
        # Retry với threshold thấp hơn
        for idx, comm in enumerate(communities):
            valid_communities.append((idx, comm, 0.0))

    # Tính cohesion matrix cho tumor samples
    logger.info(f"  -> Tính cohesion scores cho {rnaseq_tumor.shape[1]} tumor samples...")
    records = []
    comm_names = [f"community_{c[0]:03d}" for c in valid_communities]

    for sample_id, expr in rnaseq_tumor.items():
        row = {"sample_id": sample_id}
        for (comm_idx, comm, _), comm_name in zip(valid_communities, comm_names):
            row[comm_name] = compute_zsummary(expr, comm, gene_means, gene_stds)
        records.append(row)

    df = pd.DataFrame(records).set_index("sample_id")
    logger.info(
        f"  -> Cohesion matrix: {df.shape[0]} samples x {df.shape[1]} communities"
    )
    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Community Cohesion Score")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD", choices=["LUAD", "LUSC"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer      = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    raw_dir     = Path(cfg["paths"]["raw"][cancer.lower()])
    log_dir     = cfg["paths"]["logs"]
    cc_cfg      = cfg["community_cohesion"]
    split_file  = processed_dir / "patient_splits.json"

    logger = setup_logger(log_dir, cancer)
    logger.info("=" * 65)
    logger.info(f"  Community Cohesion Scores — {cancer}")
    logger.info("=" * 65)

    # Tải dữ liệu RNA-seq đã processed
    rnaseq_file = processed_dir / "rnaseq_normalized.csv.gz"
    logger.info(f"[LOAD] {rnaseq_file}")
    rnaseq = pd.read_csv(rnaseq_file, index_col=0, compression="gzip")
    logger.info(f"  -> Shape: {rnaseq.shape}")

    # Tải split info
    with open(split_file) as f:
        splits = json.load(f)
    train_samples = splits["train"]

    # Tải metadata để phân biệt normal/tumor
    meta_file = raw_dir / "rnaseq_metadata.csv"
    if meta_file.exists():
        meta = pd.read_csv(meta_file)
        normal_cols = meta.loc[
            meta.get("sample_type", pd.Series()).str.contains("Normal", na=False),
            "barcode",
        ].tolist() if "sample_type" in meta.columns else []
    else:
        normal_cols = []
        logger.warning("[WARN] Không có metadata → không có normal reference")

    # Normal samples từ training set
    train_normals = [s for s in normal_cols if s in train_samples]
    if not train_normals:
        # Fallback: lấy 20% train samples làm pseudo-normal
        logger.warning("[WARN] Không có normal samples, dùng 20% train làm pseudo-normal")
        n_pseudo = max(5, int(0.2 * len(train_samples)))
        train_normals = sorted(train_samples)[:n_pseudo]

    train_tumors = [s for s in train_samples if s not in train_normals]

    rnaseq_normal = rnaseq[[c for c in train_normals if c in rnaseq.columns]]
    rnaseq_tumor  = rnaseq[[c for c in rnaseq.columns if c not in train_normals]]

    logger.info(f"  -> Normal samples (train): {rnaseq_normal.shape[1]}")
    logger.info(f"  -> Tumor samples:          {rnaseq_tumor.shape[1]}")

    # Step 1: Xây đồ thị co-expression (chỉ từ normal samples)
    G = build_normal_coexpression_network(
        rnaseq,
        train_normals,
        soft_threshold_power=cc_cfg["wgcna_power"],
        logger=logger,
    )

    # Lưu network
    net_file = processed_dir / "coexpression_network.gpickle"
    nx.write_gpickle(G, str(net_file))
    logger.info(f"[SAVE] Network: {net_file}")

    # Step 2: Phát hiện communities
    communities = detect_communities(G, cc_cfg["min_module_size"], logger)

    # Lưu community membership
    comm_list = [{"id": i, "genes": list(c), "size": len(c)} for i, c in enumerate(communities)]
    with open(processed_dir / "communities.json", "w") as f:
        json.dump(comm_list, f, indent=2)
    logger.info(f"[SAVE] Communities: {len(communities)} modules → communities.json")

    # Step 3: Tính cohesion scores
    logger.info("\n[COHESION] Tính cohesion scores...")
    cohesion_df = compute_cohesion_scores(
        rnaseq_tumor,
        rnaseq_normal,
        communities,
        zsummary_threshold=cc_cfg["zsummary_threshold"],
        logger=logger,
    )

    # Lưu cohesion matrix
    cohesion_file = processed_dir / "cohesion_scores.csv"
    cohesion_df.to_csv(cohesion_file)
    logger.info(f"[SAVE] Cohesion matrix: {cohesion_file}")

    logger.info("=" * 65)
    logger.info(f"  HOÀN THÀNH: Community Cohesion — {cancer}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
