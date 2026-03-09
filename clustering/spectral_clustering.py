"""
spectral_clustering.py — Phân nhóm phụ phân tử bệnh nhân ung thư phổi

Thuật toán: Spectral Clustering (theo MODCAN paper)
Input: Cohesion Score Matrix từ community_cohesion.py
Output:
    - Nhãn phân nhóm cho từng bệnh nhân (subtype labels)
    - Báo cáo Silhouette Score để chọn K tối ưu
    - Kaplan-Meier plot xác nhận ý nghĩa lâm sàng

Sử dụng:
    python clustering/spectral_clustering.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import yaml


def setup_logger(log_dir: str, cancer: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"spectral_clustering_{cancer}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("spectral_clustering")


# =============================================================================
# BƯỚC 1: Tìm K tối ưu với Silhouette Score
# =============================================================================
def find_optimal_k(
    X: np.ndarray,
    k_range: range,
    n_neighbors: int,
    affinity: str,
    random_seed: int,
    logger: logging.Logger,
) -> tuple[int, dict]:
    """
    Thử SpectralClustering với K từ k_range, báo cáo Silhouette và CH score.

    Returns:
        (optimal_k, scores_dict)
    """
    logger.info(f"[K-SEARCH] Tìm K tối ưu trong range {list(k_range)}...")
    logger.info(f"  -> Affinity: {affinity}, n_neighbors: {n_neighbors}")

    results = {}
    for k in k_range:
        sc = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            n_neighbors=n_neighbors,
            random_state=random_seed,
            n_jobs=-1,
        )
        labels = sc.fit_predict(X)

        # Cần ít nhất 2 clusters với mỗi cluster có ≥ 2 samples để tính Silhouette
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or min(np.bincount(labels)) < 2:
            logger.warning(f"  K={k}: Không đủ điều kiện tính Silhouette (labels={unique_labels})")
            continue

        sil = silhouette_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        results[k] = {
            "labels": labels.tolist(),
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "cluster_sizes": np.bincount(labels).tolist(),
        }
        logger.info(
            f"  K={k:2d} | Silhouette={sil:.4f} | CH={ch:.1f} "
            f"| Sizes={np.bincount(labels).tolist()}"
        )

    if not results:
        raise ValueError("Không có K nào cho kết quả hợp lệ!")

    # Chọn K tối ưu theo Silhouette cao nhất
    optimal_k = max(results, key=lambda k: results[k]["silhouette"])
    logger.info(f"\n  ✓ K tối ưu = {optimal_k} (Silhouette = {results[optimal_k]['silhouette']:.4f})")
    return optimal_k, results


# =============================================================================
# BƯỚC 2: Vẽ báo cáo Silhouette
# =============================================================================
def plot_silhouette_report(
    scores: dict,
    optimal_k: int,
    out_path: Path,
    logger: logging.Logger,
):
    """Vẽ biểu đồ Silhouette Score vs K."""
    k_vals     = sorted(scores.keys())
    sil_vals   = [scores[k]["silhouette"] for k in k_vals]
    ch_vals    = [scores[k]["calinski_harabasz"] for k in k_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Chọn Số Phân Nhóm Tối Ưu (K)", fontsize=14, fontweight="bold")

    # Silhouette
    ax = axes[0]
    bars = ax.bar(
        k_vals, sil_vals,
        color=["#e74c3c" if k == optimal_k else "#3498db" for k in k_vals],
        alpha=0.85, edgecolor="white"
    )
    ax.set_xlabel("Số phân nhóm K", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score (cao hơn = tốt hơn)", fontsize=11)
    ax.axvline(optimal_k, color="#e74c3c", linestyle="--", alpha=0.6, label=f"K*={optimal_k}")
    ax.legend()
    ax.set_xticks(k_vals)

    # Calinski-Harabasz
    ax = axes[1]
    ax.plot(k_vals, ch_vals, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax.axvline(optimal_k, color="#e74c3c", linestyle="--", alpha=0.6, label=f"K*={optimal_k}")
    ax.set_xlabel("Số phân nhóm K", fontsize=12)
    ax.set_ylabel("Calinski-Harabasz Score", fontsize=12)
    ax.set_title("Calinski-Harabasz Score (cao hơn = tốt hơn)", fontsize=11)
    ax.legend()
    ax.set_xticks(k_vals)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Silhouette report: {out_path}")


# =============================================================================
# BƯỚC 3: Kaplan-Meier — Xác nhận ý nghĩa lâm sàng
# =============================================================================
def kaplan_meier_subtypes(
    labels: np.ndarray,
    sample_ids: list[str],
    clinical_df: pd.DataFrame,
    out_path: Path,
    logger: logging.Logger,
):
    """
    Vẽ Kaplan-Meier cho từng phân nhóm và chạy log-rank test.
    """
    if clinical_df.empty:
        logger.warning("[KM] Không có dữ liệu lâm sàng, bỏ qua Kaplan-Meier")
        return

    # Tìm cột OS (Overall Survival)
    duration_col = next(
        (c for c in clinical_df.columns if "days_to" in c.lower() and "death" in c.lower()),
        None
    )
    event_col = next(
        (c for c in clinical_df.columns if "vital_status" in c.lower()),
        None
    )
    id_col = next(
        (c for c in clinical_df.columns if "submitter_id" in c.lower() or "barcode" in c.lower()),
        None
    )

    if not all([duration_col, event_col, id_col]):
        logger.warning(f"[KM] Thiếu cột lâm sàng ({duration_col}, {event_col}, {id_col})")
        return

    # Ghép nhãn subtype với clinical
    subtype_df = pd.DataFrame({
        id_col: sample_ids,
        "subtype": labels,
    })
    subtype_df[id_col] = subtype_df[id_col].apply(
        lambda x: "-".join(str(x).split("-")[:3])
    )

    merged = clinical_df.merge(subtype_df, on=id_col, how="inner")
    if merged.empty:
        logger.warning("[KM] Không join được clinical data với subtype labels")
        return

    merged[duration_col] = pd.to_numeric(merged[duration_col], errors="coerce")
    merged[event_col] = (merged[event_col].str.lower() == "dead").astype(int)
    merged = merged.dropna(subset=[duration_col])

    # Vẽ Kaplan-Meier
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(np.unique(labels))))
    kmf = KaplanMeierFitter()

    p_values = []
    unique_subtypes = sorted(merged["subtype"].unique())

    for i, subtype in enumerate(unique_subtypes):
        mask = merged["subtype"] == subtype
        T = merged.loc[mask, duration_col]
        E = merged.loc[mask, event_col]
        n = mask.sum()
        kmf.fit(T, E, label=f"Subtype {subtype} (n={n})")
        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2.5)

    # Log-rank test (pairwise giữa subtype 0 vs others)
    if len(unique_subtypes) >= 2:
        ref = unique_subtypes[0]
        for other in unique_subtypes[1:]:
            mask_ref = merged["subtype"] == ref
            mask_oth = merged["subtype"] == other
            result = logrank_test(
                merged.loc[mask_ref, duration_col], merged.loc[mask_oth, duration_col],
                merged.loc[mask_ref, event_col],   merged.loc[mask_oth, event_col],
            )
            p_values.append(result.p_value)

    p_text = " | ".join([f"p={p:.3e}" for p in p_values])
    ax.set_title(f"Kaplan-Meier — Phân Tích Sinh Tồn theo Phân Nhóm\n{p_text}", fontsize=12)
    ax.set_xlabel("Số ngày", fontsize=11)
    ax.set_ylabel("Xác suất sống sót", fontsize=11)
    ax.legend(framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[PLOT] Kaplan-Meier: {out_path} | p-values: {p_text}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Spectral Clustering — Patient Subtyping")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD", choices=["LUAD", "LUSC"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    raw_dir       = Path(cfg["paths"]["raw"][cancer.lower()])
    log_dir       = cfg["paths"]["logs"]
    cl_cfg        = cfg["clustering"]

    # Tạo thư mục kết quả
    results_dir = Path(cfg["paths"].get("results", "/workspace/results")) / cancer
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir, cancer)
    logger.info("=" * 65)
    logger.info(f"  Spectral Clustering — {cancer}")
    logger.info("=" * 65)

    # Tải Cohesion Score Matrix
    cohesion_file = processed_dir / "cohesion_scores.csv"
    logger.info(f"[LOAD] {cohesion_file}")
    cohesion_df = pd.read_csv(cohesion_file, index_col="sample_id")
    logger.info(f"  -> Shape: {cohesion_df.shape}")

    # Chuẩn hoá lại cohesion scores trước clustering
    scaler = StandardScaler()
    X = scaler.fit_transform(cohesion_df.fillna(0))
    sample_ids = cohesion_df.index.tolist()

    # Tìm K tối ưu
    k_range = range(cl_cfg["k_range"][0], cl_cfg["k_range"][1] + 1)
    optimal_k, scores = find_optimal_k(
        X,
        k_range,
        n_neighbors=cl_cfg["n_neighbors"],
        affinity=cl_cfg["affinity"],
        random_seed=cl_cfg["random_seed"],
        logger=logger,
    )

    # Vẽ báo cáo Silhouette
    plot_silhouette_report(
        scores,
        optimal_k,
        results_dir / "silhouette_report.png",
        logger,
    )

    # Phân nhóm với K tối ưu
    best_labels = np.array(scores[optimal_k]["labels"])
    logger.info(f"\n[CLUSTER] Kết quả với K={optimal_k}:")
    for k, size in enumerate(scores[optimal_k]["cluster_sizes"]):
        logger.info(f"  Subtype {k}: {size} bệnh nhân")

    # Lưu nhãn phân nhóm
    subtype_df = pd.DataFrame({
        "sample_id": sample_ids,
        "subtype": best_labels,
    })
    subtype_file = processed_dir / "subtype_labels.csv"
    subtype_df.to_csv(subtype_file, index=False)
    logger.info(f"[SAVE] Subtype labels: {subtype_file}")

    # Lưu toàn bộ kết quả scoring
    with open(results_dir / "clustering_scores.json", "w") as f:
        scores_serializable = {
            str(k): {kk: vv if not isinstance(vv, list) else vv
                     for kk, vv in v.items() if kk != "labels"}
            for k, v in scores.items()
        }
        json.dump(
            {"optimal_k": optimal_k, "scores": scores_serializable},
            f, indent=2
        )

    # Kaplan-Meier (nếu có clinical data)
    clinical_file = raw_dir / "clinical.csv"
    if clinical_file.exists():
        clinical_df = pd.read_csv(clinical_file)
        kaplan_meier_subtypes(
            best_labels,
            sample_ids,
            clinical_df,
            results_dir / "kaplan_meier_subtypes.png",
            logger,
        )
    else:
        logger.warning(f"[WARN] Không tìm thấy: {clinical_file}")

    logger.info("=" * 65)
    logger.info(f"  HOÀN THÀNH: Spectral Clustering — K={optimal_k} subtypes")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
