"""
download_labels.py — Tải nhãn gen ung thư (Driver Genes) từ NCG và COSMIC

Nguồn:
    - NCG (Network of Cancer Genes) v7.1: gen ung thư đã được xác nhận
    - COSMIC Cancer Gene Census (CGC) Tier 1 & 2

Đầu ra:
    - data/raw/labels/driver_genes.csv  — Bảng gen (gene, source, tier, is_driver)
    - data/raw/labels/driver_gene_set.txt — Danh sách tên gen (1 gen/dòng)

Sử dụng:
    python data_acquisition/download_labels.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
import yaml

# ── URLs ──────────────────────────────────────────────────────────────────────
# NCG7 — danh sách gen ung thư đã xác nhận (cancer genes, cancer type, etc.)
NCG_DOWNLOAD_URL = (
    "http://ncg.kcl.ac.uk/files/NCG7_cancergenes.tsv"
)

# COSMIC CGC (phiên bản public, không cần đăng nhập)
# Tier 1 & 2: downloaded from COSMIC Open Data
COSMIC_CGC_URL = (
    "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v99/"
    "cancer_gene_census.csv"
)

# Dự phòng: NCG API endpoint (nếu file trực tiếp không accessible)
NCG_API_URL = "http://ncg.kcl.ac.uk/query.php"


def setup_logger(log_dir: str, name: str = "download_labels") -> logging.Logger:
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


def download_ncg(out_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Tải danh sách gen ung thư từ NCG v7.

    Trả về DataFrame: [gene_symbol, cancer_type, ncg_type (canonical/candidate)]
    """
    ncg_file = out_dir / "ncg7_raw.tsv"

    if ncg_file.exists():
        logger.info(f"[SKIP] NCG file đã tồn tại: {ncg_file}")
        return pd.read_csv(ncg_file, sep="\t")

    logger.info(f"[DOWNLOAD] NCG7 từ: {NCG_DOWNLOAD_URL}")

    try:
        resp = requests.get(NCG_DOWNLOAD_URL, timeout=60)
        resp.raise_for_status()

        with open(ncg_file, "w") as f:
            f.write(resp.text)
        df = pd.read_csv(ncg_file, sep="\t")
        logger.info(f"  -> NCG7: {len(df):,} dòng, {df['symbol'].nunique():,} genes unique")
        return df

    except requests.RequestException as e:
        logger.warning(f"  -> Không tải được NCG trực tiếp: {e}")
        logger.info("  -> Thử API NCG...")

        # Fallback: dùng API
        params = {"term": "cancer_gene", "db": "NCG7", "format": "tsv"}
        resp = requests.get(NCG_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        with open(ncg_file, "w") as f:
            f.write(resp.text)
        df = pd.read_csv(ncg_file, sep="\t")
        logger.info(f"  -> NCG7 (via API): {len(df):,} dòng")
        return df


def parse_ncg(df_raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Chuẩn hoá bảng NCG thành định dạng chung."""
    # NCG fields: symbol, type (canonical/candidate), cancer_type, ...
    if "symbol" not in df_raw.columns:
        # Thử tên cột khác
        col_map = {c: c.lower().strip() for c in df_raw.columns}
        df_raw = df_raw.rename(columns=col_map)

    gene_col = next((c for c in df_raw.columns if "symbol" in c or "gene" in c), None)
    type_col = next((c for c in df_raw.columns if "type" in c), None)

    if gene_col is None:
        logger.error("❌ Không tìm thấy cột gene trong NCG file!")
        return pd.DataFrame(columns=["gene_symbol", "source", "tier", "is_driver"])

    records = []
    for _, row in df_raw.iterrows():
        gene = str(row[gene_col]).strip()
        ncg_type = str(row[type_col]).strip().lower() if type_col else "unknown"
        tier = 1 if "canonical" in ncg_type else 2
        records.append({
            "gene_symbol": gene,
            "source": "NCG7",
            "tier": tier,
            "ncg_type": ncg_type,
            "is_driver": 1,
        })

    df = pd.DataFrame(records).drop_duplicates(subset=["gene_symbol"])
    logger.info(f"  -> NCG7 parsed: {len(df):,} driver genes")
    return df


def download_cosmic_cgc(
    out_dir: Path, tiers: list[int], logger: logging.Logger
) -> pd.DataFrame:
    """
    Tải COSMIC Cancer Gene Census (CGC).

    COSMIC yêu cầu đăng ký. Nếu không có token, dùng phiên bản public
    đã download sẵn hoặc tải từ GitHub COSMIC mirror.
    """
    cosmic_file = out_dir / "cosmic_cgc_raw.csv"

    # Nếu file chưa có, cung cấp danh sách hardcoded các gen phổ biến
    # để pipeline vẫn chạy được khi không có credential COSMIC
    if not cosmic_file.exists():
        logger.info(f"[DOWNLOAD] COSMIC CGC từ Public Data...")
        # Sử dụng COSMIC public endpoint (không cần login)
        cosmic_public_url = (
            "https://raw.githubusercontent.com/SuLab/WikidataIntegrator/main/"
            "wikidataintegrator/wdi_helpers/COSMIC_genes.tsv"
        )
        try:
            resp = requests.get(cosmic_public_url, timeout=60)
            resp.raise_for_status()
            with open(cosmic_file.with_suffix(".tsv"), "w") as f:
                f.write(resp.text)
            df_raw = pd.read_csv(cosmic_file.with_suffix(".tsv"), sep="\t")
            df_raw.to_csv(cosmic_file, index=False)
            logger.info(f"  -> COSMIC public: {len(df_raw):,} genes")
        except Exception as e:
            logger.warning(f"  -> Download COSMIC thất bại: {e}")
            logger.info(
                "  -> [NOTE] Để tải COSMIC CGC đầy đủ, hãy đăng nhập tại "
                "https://cancer.sanger.ac.uk và đặt file vào: "
                f"{cosmic_file}"
            )
            # Trả về LUSC/LUAD-specific drivers được biết đến từ literature
            known_lung_drivers = [
                "KRAS", "EGFR", "TP53", "STK11", "KEAP1", "NF1", "RB1",
                "CDKN2A", "MET", "RET", "ALK", "ROS1", "BRAF", "ERBB2",
                "NRAS", "PIK3CA", "PTEN", "SMARCA4", "NFE2L2", "CUL3",
                "ARID1A", "ATM", "FGFR1", "SOX2", "CCND1"
            ]
            fallback_df = pd.DataFrame({
                "gene_symbol": known_lung_drivers,
                "source": "literature_fallback",
                "tier": 1,
                "is_driver": 1,
            })
            logger.warning(f"  -> Dùng {len(fallback_df)} gen từ literature làm fallback COSMIC")
            return fallback_df

    df_raw = pd.read_csv(cosmic_file)
    logger.info(f"  -> COSMIC raw: {len(df_raw):,} dòng")
    return df_raw


def parse_cosmic(
    df_raw: pd.DataFrame, tiers: list[int], logger: logging.Logger
) -> pd.DataFrame:
    """Chuẩn hoá bảng COSMIC CGC."""
    records = []
    gene_col = next(
        (c for c in df_raw.columns if "gene" in c.lower() or "symbol" in c.lower()), None
    )
    tier_col = next((c for c in df_raw.columns if "tier" in c.lower()), None)

    for _, row in df_raw.iterrows():
        gene = str(row[gene_col]).strip() if gene_col else ""
        tier = int(row[tier_col]) if tier_col and pd.notna(row[tier_col]) else 1

        if gene and tier in tiers:
            records.append({
                "gene_symbol": gene,
                "source": "COSMIC_CGC",
                "tier": tier,
                "is_driver": 1,
            })

    df = pd.DataFrame(records).drop_duplicates(subset=["gene_symbol"])
    logger.info(f"  -> COSMIC CGC parsed (tier {tiers}): {len(df):,} drivers")
    return df


def merge_labels(
    ncg_df: pd.DataFrame,
    cosmic_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Hợp nhất NCG + COSMIC, ưu tiên COSMIC tier 1 nếu xung đột."""
    combined = pd.concat([ncg_df, cosmic_df], ignore_index=True)
    # Gen xuất hiện trong cả hai nguồn → giữ bản tier thấp nhất (tín nhiệm cao hơn)
    combined = (
        combined
        .sort_values("tier")
        .drop_duplicates(subset=["gene_symbol"], keep="first")
        .reset_index(drop=True)
    )

    n_total = len(combined)
    n_ncg = combined["source"].str.contains("NCG").sum()
    n_cosmic = combined["source"].str.contains("COSMIC").sum()
    n_overlap = n_total - n_ncg - n_cosmic + (
        len(ncg_df) + len(cosmic_df) - n_total
    )

    logger.info("=" * 50)
    logger.info("  TỔNG HỢP NHÃN DRIVER GENE")
    logger.info(f"  NCG7   : {len(ncg_df):,} genes")
    logger.info(f"  COSMIC : {len(cosmic_df):,} genes")
    logger.info(f"  Hợp nhất: {n_total:,} genes (sau dedup)")
    logger.info("=" * 50)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Tải nhãn Driver Gene từ NCG & COSMIC")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["raw"]["labels"])
    log_dir = cfg["paths"]["logs"]
    tiers   = cfg["data_acquisition"]["labels"].get("cosmic_tier", [1, 2])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir, "download_labels")
    logger.info("=" * 60)
    logger.info("  Driver Gene Label Download — MODCAN-GNN")
    logger.info("=" * 60)

    # Tải NCG
    logger.info("[MODULE 1/2] NCG v7 ─────────────────────")
    ncg_raw  = download_ncg(out_dir, logger)
    ncg_df   = parse_ncg(ncg_raw, logger)

    # Tải COSMIC
    logger.info("[MODULE 2/2] COSMIC CGC ──────────────────")
    cosmic_raw = download_cosmic_cgc(out_dir, tiers, logger)
    cosmic_df  = parse_cosmic(cosmic_raw, tiers, logger)

    # Merge
    final_df = merge_labels(ncg_df, cosmic_df, logger)

    # Lưu kết quả
    out_csv = out_dir / "driver_genes.csv"
    final_df.to_csv(out_csv, index=False)
    logger.info(f"[SAVE] {out_csv}")

    out_txt = out_dir / "driver_gene_set.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(final_df["gene_symbol"].tolist()))
    logger.info(f"[SAVE] {out_txt}")

    logger.info("HOÀN THÀNH: Driver gene labels đã sẵn sàng.")


if __name__ == "__main__":
    main()
