"""
data_loader.py — Đọc dữ liệu TCGA đã có sẵn từ thư mục local

Format dữ liệu (tất cả đều dùng pattern nhất quán):
    *_data.txt / *_matrix.txt — Ma trận số không header (genes x samples)
    *_gene.txt               — Danh sách gen (1 gene/dòng, tương ứng rows)
    *_sample.txt             — Danh sách mẫu (1 barcode/dòng, tương ứng cols)

Các nguồn dữ liệu:
    TCGA_UCSC_EXP/   → RNA-seq đã log2 transform
    TCGA_UCSC_MET/   → Methylation beta values
    TCGA_hg38_SNV/   → Binary SNV mutation matrix
    TCGA_UCSC_CNV/   → Binary CNV matrix
    TCGA_UCSC_normal/→ Normal tissue (exp + met)
    survival/        → OS survival data
    network/         → STRING PPI (gene symbol, đã filter score ≥ 700)
    reference/       → 579_CGC.txt (driver) + 2179_non_driver_genes.txt

Sử dụng:
    from data_acquisition.data_loader import TCGALocalLoader
    loader = TCGALocalLoader(data_root="/home/ankkun/Downloads/data", cancer="LUAD")
    exp = loader.load_expression()      # DataFrame (genes x samples)
    met = loader.load_methylation()     # DataFrame (genes x samples)
    snv = loader.load_snv()             # DataFrame (genes x samples) binary
    cnv = loader.load_cnv()             # DataFrame (genes x samples) binary
    normal_exp = loader.load_normal_expression()
    normal_met = loader.load_normal_methylation()
    survival = loader.load_survival()   # DataFrame với cột OS, OS.time
    ppi = loader.load_ppi()             # DataFrame (gene_a, gene_b, score)
    drivers, non_drivers = loader.load_labels()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── Logger ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class TCGALocalLoader:
    """
    Loader cho dữ liệu TCGA đã xử lý sẵn từ thư mục local.

    Args:
        data_root: Đường dẫn gốc đến thư mục data/ (chứa TCGA_UCSC_EXP, v.v.)
        cancer:    Mã ung thư: "LUAD" hoặc "LUSC"
    """

    CANCER_TYPES = ["LUAD", "LUSC", "BLCA", "BRCA", "COAD", "HNSC",
                    "KIRC", "KIRP", "PRAD", "UCEC"]

    def __init__(self, data_root: str, cancer: str = "LUAD"):
        self.root = Path(data_root)
        self.cancer = cancer.upper()
        self.project = f"TCGA-{self.cancer}"

        assert self.cancer in self.CANCER_TYPES, \
            f"cancer phải là một trong {self.CANCER_TYPES}"

        logger.info(f"[LOADER] Khởi tạo TCGALocalLoader: {self.project}")
        logger.info(f"  data_root = {self.root}")
        self._verify_paths()

    def _verify_paths(self):
        """Kiểm tra các thư mục bắt buộc tồn tại."""
        required = [
            self.root / "TCGA_UCSC_EXP"  / self.project,
            self.root / "TCGA_UCSC_MET"  / self.project,
            self.root / "TCGA_hg38_SNV"  / "snv_matrix",
            self.root / "TCGA_UCSC_normal" / self.project,
            self.root / "network",
            self.root / "reference",
            self.root / "survival",
        ]
        for p in required:
            if not p.exists():
                logger.warning(f"  [WARN] Không tìm thấy: {p}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_matrix_txt(
        self,
        matrix_file: Path,
        gene_file: Path,
        sample_file: Path,
        name: str = "",
        dtype=np.float32,
    ) -> pd.DataFrame:
        """
        Đọc bộ 3 file (matrix + gene + sample) và trả về DataFrame.

        Cấu trúc:
            matrix_file: tab-separated, không header, shape (n_genes, n_samples)
            gene_file:   1 gene/dòng
            sample_file: 1 sample barcode/dòng
        """
        if not matrix_file.exists():
            raise FileNotFoundError(f"Không tìm thấy: {matrix_file}")

        genes   = self._read_list(gene_file)
        samples = self._read_list(sample_file)

        logger.info(f"  [READ] {name}: {len(genes)} genes × {len(samples)} samples")

        # Đọc ma trận — tab separated, không header
        mat = np.loadtxt(matrix_file, dtype=dtype)

        # Kiểm tra kích thước
        expected = (len(genes), len(samples))
        if mat.shape != expected:
            # Thử transpose nếu shape bị ngược
            if mat.shape == (expected[1], expected[0]):
                logger.info(f"    -> Transpose ma trận từ {mat.shape} → {expected}")
                mat = mat.T
            else:
                raise ValueError(
                    f"{name}: Kích thước ma trận {mat.shape} không khớp với "
                    f"genes({len(genes)}) × samples({len(samples)})"
                )

        df = pd.DataFrame(mat, index=genes, columns=samples)
        logger.info(f"    -> DataFrame: {df.shape}")
        return df

    def _read_list(self, path: Path) -> list[str]:
        """Đọc file danh sách (1 item/dòng)."""
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    # ── Public loaders ─────────────────────────────────────────────────────────

    def load_expression(self) -> pd.DataFrame:
        """
        Tải RNA-seq gene expression (đã log2 transform).
        Shape: (n_genes, n_tumor_samples)
        """
        base = self.root / "TCGA_UCSC_EXP" / self.project
        return self._load_matrix_txt(
            matrix_file = base / f"{self.project}_exp_data.txt",
            gene_file   = base / f"{self.project}_gene.txt",
            sample_file = base / f"{self.project}_sample.txt",
            name        = "RNA-seq EXP",
        )

    def load_methylation(self) -> pd.DataFrame:
        """
        Tải DNA Methylation (beta values 0–1).
        Shape: (n_genes, n_tumor_samples)
        """
        base = self.root / "TCGA_UCSC_MET" / self.project
        return self._load_matrix_txt(
            matrix_file = base / f"{self.project}_met_data.txt",
            gene_file   = base / f"{self.project}_gene.txt",
            sample_file = base / f"{self.project}_sample.txt",
            name        = "Methylation",
        )

    def load_snv(self) -> pd.DataFrame:
        """
        Tải binary SNV mutation matrix (hg38).
        Shape: (n_genes, n_tumor_samples) — 1 = có đột biến
        """
        base = self.root / "TCGA_hg38_SNV" / "snv_matrix"
        return self._load_matrix_txt(
            matrix_file = base / f"{self.project}_snv_matrix.txt",
            gene_file   = base / f"{self.project}_snv_gene.txt",
            sample_file = base / f"{self.project}_snv_sample.txt",
            name        = "SNV Binary",
            dtype       = np.int8,
        )

    def load_cnv(self) -> pd.DataFrame:
        """
        Tải binary CNV (Copy Number Variation) matrix.
        Shape: (n_genes, n_tumor_samples)
        """
        base = self.root / "TCGA_UCSC_CNV" / "cnv_matrix"
        return self._load_matrix_txt(
            matrix_file = base / f"{self.project}_cnv_matrix.txt",
            gene_file   = base / f"{self.project}_cnv_gene.txt",
            sample_file = base / f"{self.project}_cnv_sample.txt",
            name        = "CNV Binary",
            dtype       = np.int8,
        )

    def load_normal_expression(self) -> pd.DataFrame:
        """
        Tải RNA-seq của mô bình thường (Normal tissue).
        Dùng làm reference cho WGCNA community cohesion.
        """
        base = self.root / "TCGA_UCSC_normal" / self.project
        gene_file   = base / f"{self.project}_gene.txt"
        sample_file = base / f"{self.project}_exp_sample.txt"
        matrix_file = base / f"{self.project}_exp_normal.txt"
        return self._load_matrix_txt(matrix_file, gene_file, sample_file, "Normal EXP")

    def load_normal_methylation(self) -> pd.DataFrame:
        """
        Tải Methylation của mô bình thường.
        """
        base = self.root / "TCGA_UCSC_normal" / self.project
        gene_file   = base / f"{self.project}_gene.txt"
        sample_file = base / f"{self.project}_met_sample.txt"
        matrix_file = base / f"{self.project}_met_normal.txt"
        return self._load_matrix_txt(matrix_file, gene_file, sample_file, "Normal MET")

    def load_survival(self) -> pd.DataFrame:
        """
        Tải clinical survival data (OS, OS.time).
        Columns: sample, OS (0/1), _PATIENT, OS.time (days)
        """
        path = self.root / "survival" / f"{self.project}.survival.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy survival file: {path}")
        df = pd.read_csv(path, sep="\t")
        # Chuẩn hóa tên cột
        df = df.rename(columns={
            "sample": "sample_id",
            "_PATIENT": "patient_id",
            "OS": "event",
            "OS.time": "duration_days",
        })
        logger.info(f"  [READ] Survival: {len(df)} records, event_rate={df['event'].mean():.2%}")
        return df

    def load_ppi(self, min_score: int = 400, use_filtered: bool = True) -> pd.DataFrame:
        """
        Tải STRING v12 PPI network.

        Args:
            min_score:    Ngưỡng confidence tối thiểu
            use_filtered: Nếu True, dùng file string_full_v12_0.7.txt
                          (đã filter score ≥ 700, 4.1M edges — khuyến nghị)
                          Nếu False, dùng file đầy đủ (13.7M edges — rất lớn)
        """
        if use_filtered:
            path = self.root / "network" / "string_full_v12_0.7.txt"
        else:
            path = self.root / "network" / "string_full_v12.txt"

        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy PPI file: {path}")

        logger.info(f"  [READ] PPI: {path.name}...")
        df = pd.read_csv(path, sep="\t")
        df.columns = ["gene_a", "gene_b", "score"]

        if min_score > 0 and not use_filtered:
            n_before = len(df)
            df = df[df["score"] >= min_score]
            logger.info(f"    -> Lọc score >= {min_score}: {n_before:,} → {len(df):,} edges")

        # Loại bỏ self-loops
        df = df[df["gene_a"] != df["gene_b"]].reset_index(drop=True)
        logger.info(f"    -> PPI: {len(df):,} edges, {len(set(df['gene_a'])|set(df['gene_b'])):,} genes")
        return df

    def load_labels(self) -> tuple[set[str], set[str]]:
        """
        Tải nhãn gen driver và non-driver từ reference files.

        Returns:
            (driver_genes, non_driver_genes) — hai set gene symbols
        """
        cgc_file = self.root / "reference" / "579_CGC.txt"
        neg_file = self.root / "reference" / "2179_non_driver_genes.txt"

        if not cgc_file.exists():
            raise FileNotFoundError(f"Driver gene file không tìm thấy: {cgc_file}")

        driver_genes = set(self._read_list(cgc_file))
        logger.info(f"  [READ] Driver genes (CGC): {len(driver_genes)}")

        non_driver_genes = set()
        if neg_file.exists():
            non_driver_genes = set(self._read_list(neg_file))
            logger.info(f"  [READ] Non-driver genes: {len(non_driver_genes)}")
        else:
            logger.warning("  [WARN] Không tìm thấy non-driver gene file")

        # Loại bỏ overlap
        overlap = driver_genes & non_driver_genes
        if overlap:
            logger.warning(f"  [WARN] {len(overlap)} gen xuất hiện trong cả hai list → bỏ khỏi non-driver")
            non_driver_genes -= overlap

        return driver_genes, non_driver_genes

    def get_common_genes(
        self,
        exp: pd.DataFrame,
        met: pd.DataFrame,
        snv: pd.DataFrame,
        cnv: Optional[pd.DataFrame] = None,
    ) -> list[str]:
        """Tìm tập gen chung giữa tất cả các omics data."""
        gene_sets = [set(exp.index), set(met.index), set(snv.index)]
        if cnv is not None:
            gene_sets.append(set(cnv.index))
        common = sorted(set.intersection(*gene_sets))
        logger.info(f"  [INTERSECT] Gen chung (EXP ∩ MET ∩ SNV{' ∩ CNV' if cnv is not None else ''}): {len(common):,}")
        return common

    def get_common_samples(
        self,
        *dfs: pd.DataFrame,
        names: list[str] = None,
    ) -> list[str]:
        """Tìm tập sample chung giữa nhiều data frames."""
        col_sets = [set(df.columns) for df in dfs]
        common = sorted(set.intersection(*col_sets))
        if names:
            for name, df in zip(names, dfs):
                logger.info(f"  {name}: {df.shape[1]} samples")
        logger.info(f"  [INTERSECT] Samples chung: {len(common)}")
        return common

    def summary(self) -> dict:
        """In tóm tắt tổng quan về dữ liệu tải được."""
        info = {"cancer": self.cancer, "project": self.project, "data_root": str(self.root)}
        try:
            exp = self.load_expression()
            info["exp_genes"] = exp.shape[0]
            info["exp_samples"] = exp.shape[1]
        except Exception as e:
            info["exp_error"] = str(e)
        try:
            snv = self.load_snv()
            info["snv_genes"] = snv.shape[0]
            info["snv_samples"] = snv.shape[1]
        except Exception as e:
            info["snv_error"] = str(e)
        try:
            surv = self.load_survival()
            info["survival_samples"] = len(surv)
        except Exception as e:
            info["survival_error"] = str(e)
        try:
            drivers, non_drivers = self.load_labels()
            info["n_drivers"] = len(drivers)
            info["n_non_drivers"] = len(non_drivers)
        except Exception as e:
            info["label_error"] = str(e)
        return info
