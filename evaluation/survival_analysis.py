"""
survival_analysis.py — Phân tích sinh tồn Kaplan-Meier cho các driver genes

Tính toán:
    - Chia bệnh nhân thành 2 nhóm: mang/không mang gen driver được tìm ra
    - Vẽ Kaplan-Meier curves với confidence intervals
    - Log-rank test kiểm định ý nghĩa thống kê

Sử dụng:
    python evaluation/survival_analysis.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import yaml


def survival_analysis_for_gene(
    gene: str,
    predictions_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
    threshold: float = 0.5,
):
    """Vẽ KM curve cho 1 gen cụ thể."""
    # Xác định bệnh nhân có đột biến ở gen này
    # (Predictions là gene-level → cần liên kết với mutation matrix)
    pass  # Tích hợp đầy đủ trong bước tích hợp pipeline


def kaplan_meier_by_subtype(
    subtype_labels: pd.DataFrame,  # sample_id, subtype
    clinical_df: pd.DataFrame,
    cancer: str,
    results_dir: Path,
    logger: logging.Logger,
):
    """
    Vẽ Kaplan-Meier survival curves cho từng phân nhóm phụ phân tử.
    Dùng Overall Survival (OS) làm endpoint.
    """
    # Tìm cột survival
    day_col = next(
        (c for c in clinical_df.columns if "days" in c.lower() and "death" in c.lower()),
        None,
    )
    status_col = next(
        (c for c in clinical_df.columns if "vital_status" in c.lower()),
        None,
    )
    id_col = next(
        (c for c in clinical_df.columns if "submitter_id" in c.lower()),
        None,
    )

    if not all([day_col, status_col, id_col]):
        logger.warning(f"[KM] Thiếu cột survival: {day_col}, {status_col}, {id_col}")
        return

    # Chuẩn hóa patient ID
    subtype_labels = subtype_labels.copy()
    subtype_labels["patient_id"] = subtype_labels["sample_id"].apply(
        lambda x: "-".join(str(x).split("-")[:3])
    )
    clinical_df = clinical_df.copy()
    clinical_df["patient_id"] = clinical_df[id_col]

    merged = clinical_df.merge(subtype_labels[["patient_id", "subtype"]], on="patient_id", how="inner")
    merged[day_col] = pd.to_numeric(merged[day_col], errors="coerce")
    merged["event"] = (merged[status_col].str.lower() == "dead").astype(int)
    merged = merged.dropna(subset=[day_col])

    if merged.empty:
        logger.warning("[KM] Không có dữ liệu sau khi merge!")
        return

    # Vẽ
    colors = plt.cm.Set1(np.linspace(0, 0.8, merged["subtype"].nunique()))
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()

    subtypes = sorted(merged["subtype"].unique())
    p_values = []

    for i, st in enumerate(subtypes):
        mask = merged["subtype"] == st
        T = merged.loc[mask, day_col]
        E = merged.loc[mask, "event"]
        n = mask.sum()
        kmf.fit(T, E, label=f"Subtype {st} (n={n})")
        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2.5, ci_show=True)

        if i > 0:  # Log-rank test vs subtype 0
            mask_ref = merged["subtype"] == subtypes[0]
            result = logrank_test(
                merged.loc[mask_ref, day_col], T,
                merged.loc[mask_ref, "event"],  E,
            )
            p_values.append(f"Sub0 vs Sub{st}: p={result.p_value:.3e}")

    p_str = " | ".join(p_values) if p_values else ""
    ax.set_title(
        f"Kaplan-Meier Overall Survival — {cancer}\n{p_str}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Ngày", fontsize=11)
    ax.set_ylabel("Xác suất sống sót", fontsize=11)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    out_path = results_dir / f"kaplan_meier_{cancer}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[SAVE] Kaplan-Meier: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    raw_dir       = Path(cfg["paths"]["raw"][cancer.lower()])
    results_dir   = Path(cfg["paths"].get("results", "/workspace/results")) / cancer
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger("survival")

    clinical_file = raw_dir / "clinical.csv"
    subtype_file  = processed_dir / "subtype_labels.csv"

    if not clinical_file.exists():
        logger.warning(f"Clinical file không tồn tại: {clinical_file}")
        return
    if not subtype_file.exists():
        logger.warning(f"Subtype labels không tồn tại: {subtype_file}")
        return

    clinical_df   = pd.read_csv(clinical_file)
    subtype_df    = pd.read_csv(subtype_file)

    kaplan_meier_by_subtype(subtype_df, clinical_df, cancer, results_dir, logger)


if __name__ == "__main__":
    main()
