"""
preprocess.py (v2) — Tiền xử lý dữ liệu đa thể học từ LOCAL FILES

Thay thế phiên bản cũ dựa trên TCGA API.
Dùng TCGALocalLoader để đọc trực tiếp từ /home/ankkun/Downloads/data/

Pipeline:
    1. Load EXP, MET, SNV (+ CNV tùy chọn) từ local files
    2. Tìm giao of gen và samples chung
    3. Z-score normalize EXP (đã log2 sẵn) và MET theo gene
    4. Lọc gen mutation rate thấp trong SNV
    5. Chia Train/Val/Test theo PATIENT ID (tránh data leakage)
    6. Export tất cả ra data/processed/LUAD/ hoặc LUSC/
    7. Ghi đầy đủ preprocessing log (JSON)

Sử dụng:
    python data_acquisition/preprocess.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

from data_acquisition.data_loader import TCGALocalLoader


# ── Logger & StepLogger ────────────────────────────────────────────────────────
def setup_logger(log_dir: str, cancer: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"preprocess_v2_{cancer}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("preprocess_v2")


class StepLogger:
    def __init__(self, logger):
        self.logger = logger
        self.steps = []
        self._n = 0

    def log(self, step: str, msg: str, meta: dict = None):
        self._n += 1
        entry = {"step": self._n, "name": step,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "message": msg}
        if meta:
            entry.update(meta)
        self.steps.append(entry)
        self.logger.info(f"[Step {self._n:02d}] {step}: {msg}")

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2, ensure_ascii=False)


# =============================================================================
# NORMALIZE
# =============================================================================
def zscore_by_gene(df: pd.DataFrame, step_log: StepLogger, name: str):
    """Z-score theo chiều gene (axis=1) — mỗi gen chuẩn hóa độc lập."""
    scaler = StandardScaler()
    # StandardScaler hoạt động theo columns → transpose
    scaled_T = scaler.fit_transform(df.T.values)
    df_scaled = pd.DataFrame(scaled_T.T, index=df.index, columns=df.columns)
    step_log.log(
        f"zscore_{name}",
        f"Z-score hoàn tất | mean={df_scaled.values.mean():.4f} std~1",
        {"shape": list(df_scaled.shape)},
    )
    return df_scaled, scaler


# =============================================================================
# PATIENT-LEVEL SPLIT
# =============================================================================
def patient_level_split(
    all_samples: list[str],
    ratios: list[float],
    seed: int,
    step_log: StepLogger,
) -> dict[str, list[str]]:
    """
    Chia samples theo PATIENT ID để tránh data leakage.
    TCGA barcode: TCGA-XX-XXXX-... → patient = TCGA-XX-XXXX
    """
    # Trích patient ID
    patient_ids = sorted({"-".join(s.split("-")[:3]) for s in all_samples})
    train_r, val_r, test_r = ratios

    train_pts, temp_pts = train_test_split(
        patient_ids, test_size=(val_r + test_r), random_state=seed
    )
    val_pts, test_pts = train_test_split(
        temp_pts, test_size=(test_r / (val_r + test_r)), random_state=seed
    )

    def pts_to_samples(pts):
        pt_set = set(pts)
        return [s for s in all_samples if "-".join(s.split("-")[:3]) in pt_set]

    splits = {
        "train": pts_to_samples(train_pts),
        "val":   pts_to_samples(val_pts),
        "test":  pts_to_samples(test_pts),
    }
    step_log.log(
        "patient_split",
        f"Train={len(splits['train'])} Val={len(splits['val'])} Test={len(splits['test'])} (patients: {len(train_pts)}/{len(val_pts)}/{len(test_pts)})",
        {k: len(v) for k, v in splits.items()},
    )
    return splits


# =============================================================================
# FILTER
# =============================================================================
def filter_snv_by_mutation_rate(
    snv: pd.DataFrame,
    min_rate: float,
    step_log: StepLogger,
) -> pd.DataFrame:
    """Loại gen bị đột biến ở < min_rate bệnh nhân."""
    n_before = snv.shape[0]
    rates = snv.mean(axis=1)
    snv_filtered = snv[rates >= min_rate]
    step_log.log(
        "filter_snv_rate",
        f"Loại {n_before - snv_filtered.shape[0]} genes (rate < {min_rate})",
        {"genes_before": n_before, "genes_after": snv_filtered.shape[0]},
    )
    return snv_filtered


# =============================================================================
# BUILD METHYLATION GENE SUMMARY
# =============================================================================
def build_methylation_gene_summary(
    met: pd.DataFrame,
    step_log: StepLogger,
) -> pd.DataFrame:
    """
    Tổng hợp methylation theo gene: mean và std trên toàn bộ samples.
    Output: DataFrame với cột (gene_symbol, meth_mean, meth_std)
    """
    gene_summary = pd.DataFrame({
        "meth_mean": met.mean(axis=1),
        "meth_std":  met.std(axis=1).fillna(0),
    }, index=met.index)
    step_log.log(
        "meth_gene_summary",
        f"Tổng hợp methylation theo gene: {gene_summary.shape[0]} genes",
    )
    return gene_summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Preprocess v2 — Local TCGA Data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD", choices=["LUAD", "LUSC"])
    parser.add_argument("--no-cnv", action="store_true",
                        help="Bỏ qua CNV nếu gặp vấn đề đọc file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer    = args.cancer.upper()
    data_root = cfg["paths"]["raw_local_root"]     # /home/ankkun/Downloads/data
    out_dir   = Path(cfg["paths"]["processed"][cancer.lower()])
    log_dir   = cfg["paths"]["logs"]

    out_dir.mkdir(parents=True, exist_ok=True)
    logger   = setup_logger(log_dir, cancer)
    step_log = StepLogger(logger)

    pp_cfg = cfg["preprocessing"]

    logger.info("=" * 65)
    logger.info(f"  Preprocessing v2 (Local Data) — {cancer}")
    logger.info("=" * 65)

    # ── BƯỚC 1: Khởi tạo loader ───────────────────────────────────────────────
    loader = TCGALocalLoader(data_root=data_root, cancer=cancer)
    step_log.log("init_loader", f"TCGALocalLoader: {data_root}")

    # ── BƯỚC 2: Load tất cả omics data ───────────────────────────────────────
    logger.info("\n[LOAD] Đang tải dữ liệu omics...")
    exp = loader.load_expression()
    step_log.log("load_exp", f"EXP loaded: {exp.shape}")

    met = loader.load_methylation()
    step_log.log("load_met", f"MET loaded: {met.shape}")

    snv = loader.load_snv()
    step_log.log("load_snv", f"SNV loaded: {snv.shape}")

    cnv = None
    if not args.no_cnv:
        try:
            cnv = loader.load_cnv()
            step_log.log("load_cnv", f"CNV loaded: {cnv.shape}")
        except FileNotFoundError as e:
            logger.warning(f"  [WARN] CNV không tải được: {e}")

    # Normal samples (cho community cohesion)
    normal_exp = loader.load_normal_expression()
    step_log.log("load_normal_exp", f"Normal EXP loaded: {normal_exp.shape}")

    # ── BƯỚC 3: Tìm genes và samples chung ───────────────────────────────────
    logger.info("\n[ALIGN] Tìm giao của genes và samples...")

    # Genes: giao giữa EXP ∩ MET ∩ SNV (∩ CNV nếu có)
    dfs_for_genes = [exp, met, snv]
    if cnv is not None:
        dfs_for_genes.append(cnv)
    common_genes = loader.get_common_genes(exp, met, snv, cnv)
    step_log.log("common_genes", f"{len(common_genes)} genes chung")

    # Samples: giao giữa EXP ∩ MET (SNV có thể khác nhau nhẹ)
    common_samples = loader.get_common_samples(
        exp, met,
        names=["EXP", "MET"]
    )
    # Xử lý SNV theo common_samples
    snv_common_samples = [s for s in common_samples if s in snv.columns]
    step_log.log("common_samples",
                 f"{len(common_samples)} samples (EXP∩MET), SNV có {len(snv_common_samples)}")

    # Cắt về chung
    exp = exp.loc[common_genes, common_samples]
    met = met.loc[common_genes, common_samples]
    snv_aligned = snv.loc[[g for g in common_genes if g in snv.index], snv_common_samples]
    # Reindex SNV về đúng common_genes và common_samples
    snv_aligned = snv_aligned.reindex(index=common_genes, columns=common_samples).fillna(0)
    if cnv is not None:
        cnv_common_samples = [s for s in common_samples if s in cnv.columns]
        cnv_aligned = cnv.loc[
            [g for g in common_genes if g in cnv.index], cnv_common_samples
        ].reindex(index=common_genes, columns=common_samples).fillna(0)
    else:
        cnv_aligned = None

    # Normal (align genes)
    normal_genes_common = [g for g in common_genes if g in normal_exp.index]
    normal_exp_aligned = normal_exp.loc[normal_genes_common]
    step_log.log("align_all", f"Align hoàn tất: {len(common_genes)} genes × {len(common_samples)} samples")

    # ── BƯỚC 4: Z-score normalize ─────────────────────────────────────────────
    logger.info("\n[NORMALIZE] Z-score normalization...")
    exp_norm, exp_scaler = zscore_by_gene(exp, step_log, "exp")
    met_norm, met_scaler = zscore_by_gene(met, step_log, "met")
    normal_exp_norm, _ = zscore_by_gene(normal_exp_aligned, step_log, "normal_exp")

    # ── BƯỚC 5: Tổng hợp Methylation gene summary ────────────────────────────
    met_gene_summary = build_methylation_gene_summary(met_norm, step_log)

    # ── BƯỚC 6: Lọc SNV theo mutation rate ───────────────────────────────────
    logger.info("\n[FILTER] Lọc SNV...")
    snv_aligned_int = snv_aligned.astype(int)
    snv_filtered = filter_snv_by_mutation_rate(
        snv_aligned_int,
        min_rate=pp_cfg["mutation"]["min_mutation_rate"],
        step_log=step_log,
    )

    # ── BƯỚC 7: Patient-level split ───────────────────────────────────────────
    logger.info("\n[SPLIT] Patient-level train/val/test split...")
    splits = patient_level_split(
        all_samples=common_samples,
        ratios=pp_cfg["train_val_test_split"],
        seed=pp_cfg["random_seed"],
        step_log=step_log,
    )

    # ── BƯỚC 8: Load và save labels ───────────────────────────────────────────
    logger.info("\n[LABELS] Load driver / non-driver gene labels...")
    driver_genes, non_driver_genes = loader.load_labels()

    # Lọc chỉ giữ những gen có trong common_genes
    drivers_in_data    = driver_genes    & set(common_genes)
    non_drivers_in_data = non_driver_genes & set(common_genes)
    step_log.log(
        "filter_labels_to_data",
        f"Drivers trong data: {len(drivers_in_data)}/{len(driver_genes)} | "
        f"Non-drivers: {len(non_drivers_in_data)}/{len(non_driver_genes)}",
    )

    # Tạo label vector (chỉ cho gen có nhãn rõ ràng)
    label_records = []
    for gene in common_genes:
        if gene in drivers_in_data:
            label_records.append({"gene": gene, "label": 1, "source": "CGC"})
        elif gene in non_drivers_in_data:
            label_records.append({"gene": gene, "label": 0, "source": "non_driver"})
        # Gen không có nhãn → KHÔNG đưa vào (semi-supervised)
    label_df = pd.DataFrame(label_records)
    step_log.log(
        "build_labels",
        f"Label DataFrame: {len(label_df)} labeled genes | "
        f"driver={label_df['label'].sum()}, non-driver={(label_df['label']==0).sum()}",
    )

    # ── BƯỚC 9: Save tất cả outputs ──────────────────────────────────────────
    logger.info("\n[SAVE] Đang lưu kết quả...")

    exp_norm.to_csv(out_dir / "rnaseq_normalized.csv.gz", compression="gzip")
    step_log.log("save_exp", f"rnaseq_normalized.csv.gz: {exp_norm.shape}")

    met_norm.to_csv(out_dir / "methylation_normalized.csv.gz", compression="gzip")
    step_log.log("save_met", f"methylation_normalized.csv.gz: {met_norm.shape}")

    met_gene_summary.to_csv(out_dir / "methylation_gene_summary.csv", index_label="gene_symbol")
    step_log.log("save_met_summary", f"methylation_gene_summary.csv: {met_gene_summary.shape}")

    snv_filtered.to_csv(out_dir / "mutation_binary.csv")
    step_log.log("save_snv", f"mutation_binary.csv: {snv_filtered.shape}")

    if cnv_aligned is not None:
        cnv_aligned.to_csv(out_dir / "cnv_binary.csv")
        step_log.log("save_cnv", f"cnv_binary.csv: {cnv_aligned.shape}")

    normal_exp_norm.to_csv(out_dir / "rnaseq_normal_normalized.csv.gz", compression="gzip")
    step_log.log("save_normal", f"rnaseq_normal_normalized.csv.gz: {normal_exp_norm.shape}")

    # Lưu gene list
    pd.Series(common_genes, name="gene").to_csv(
        out_dir / "common_genes.txt", index=False, header=False
    )

    # Lưu patient splits
    with open(out_dir / "patient_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    step_log.log("save_splits", "patient_splits.json")

    # Lưu label file
    label_df.to_csv(out_dir / "gene_labels.csv", index=False)
    step_log.log("save_labels", f"gene_labels.csv: {len(label_df)} labeled genes")

    # Lưu survival data
    try:
        surv = loader.load_survival()
        surv.to_csv(out_dir / "clinical.csv", index=False)
        step_log.log("save_survival", f"clinical.csv: {len(surv)} records")
    except FileNotFoundError:
        logger.warning("[WARN] Không lưu được survival data")

    # Lưu preprocessing log
    step_log.save(out_dir / "preprocessing_log.json")

    logger.info("=" * 65)
    logger.info(f"  HOÀN THÀNH: Preprocess v2 — {cancer}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"  Genes: {len(common_genes)}, Samples: {len(common_samples)}")
    logger.info(f"  Labeled: {len(label_df)} ({label_df['label'].sum()} driver, "
                f"{(label_df['label']==0).sum()} non-driver)")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
