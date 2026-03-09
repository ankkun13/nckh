"""
robust_coassociation.py — CẢI TIẾN CỐT LÕI: Xây dựng Đồ thị Đồng liên kết Vi phân

Thay thế Pearson Correlation (dễ nhiễu) bằng hai phương pháp vững chắc hơn:
    1. Random Forest Feature Importance (mặc định, theo config)
    2. Mutual Information (MI) - phi tham số, bắt quan hệ phi tuyến

Nguyên tắc CHỐNG DATA LEAKAGE:
    ⚠️  TUYỆT ĐỐI: Chỉ fit Random Forest / tính MI trên TRAIN SET!
    ⚠️  Test set KHÔNG được dùng để tính trọng số cạnh.

Đầu ra:
    Một class RobustCoAssociationNetwork có API:
        .fit(X_train, genes)           — fit trên train data
        .build_adjacency(X_any, ...) — áp dụng lên bất kỳ tập nào
        .get_edge_list()               — [(gene_a, gene_b, weight)]

Sử dụng:
    python network_builder/robust_coassociation.py --config configs/config.yaml --cancer LUAD
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelBinarizer
from joblib import Parallel, delayed
import yaml


def setup_logger(log_dir: str, name: str = "robust_coassociation") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(name)


# =============================================================================
# CLASS TRUNG TÂM: RobustCoAssociationNetwork
# =============================================================================
class RobustCoAssociationNetwork:
    """
    Xây dựng mạng lưới đồng liên kết gen dựa trên Random Forest
    hoặc Mutual Information — thay thế Pearson Correlation.

    Đây là cải tiến cốt lõi so với MODCAN gốc.

    Args:
        method: "random_forest", "mutual_information", "gradient_boosting"
        edge_threshold_type: "dynamic" (percentile) hoặc "fixed"
        edge_threshold_percentile: Chỉ tạo cạnh nếu score >= percentile này
        edge_threshold_fixed: Dùng nếu type="fixed"
        rf_params: Dict tham số cho RandomForestClassifier
        n_jobs: Số thread song song
    """

    def __init__(
        self,
        method: Literal["random_forest", "mutual_information", "gradient_boosting"] = "random_forest",
        edge_threshold_type: str = "dynamic",
        edge_threshold_percentile: float = 90.0,
        edge_threshold_fixed: float = 0.01,
        rf_params: Optional[dict] = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.method = method
        self.edge_threshold_type = edge_threshold_type
        self.edge_threshold_percentile = edge_threshold_percentile
        self.edge_threshold_fixed = edge_threshold_fixed
        self.rf_params = rf_params or {
            "n_estimators": 200,
            "max_features": "sqrt",
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Trạng thái sau fit()
        self.genes_: list[str] = []
        self.importance_matrix_: Optional[np.ndarray] = None  # (n_genes, n_genes)
        self.adjacency_: Optional[sp.csr_matrix] = None       # Ma trận kề cuối
        self.edge_threshold_value_: Optional[float] = None
        self._is_fitted: bool = False

    # ── BƯỚC 1: Fit (chỉ trên TRAIN SET) ────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
    ) -> "RobustCoAssociationNetwork":
        """
        Học trọng số đồng liên kết từ train data.

        Args:
            X_train: DataFrame shape (n_samples, n_genes) — CHỈ TRAIN SET
        """
        if logger:
            logger.info(f"[FIT] Method: {self.method}")
            logger.info(f"      X_train: {X_train.shape} (samples x genes)")

        self.genes_ = list(X_train.columns)
        n_genes = len(self.genes_)

        # Ma trận lưu importance: importance_matrix[i,j] = importance của gene j khi predict gene i
        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)

        if self.method == "random_forest":
            importance_matrix = self._fit_random_forest(
                X_train.values, n_genes, logger
            )
        elif self.method == "mutual_information":
            importance_matrix = self._fit_mutual_information(
                X_train.values, n_genes, logger
            )
        elif self.method == "gradient_boosting":
            importance_matrix = self._fit_gradient_boosting(
                X_train.values, n_genes, logger
            )
        else:
            raise ValueError(f"Phương pháp không hợp lệ: {self.method}")

        # Đối xứng hoá: lấy max(i,j, j,i) để đảm bảo vô hướng
        importance_matrix = np.maximum(importance_matrix, importance_matrix.T)
        np.fill_diagonal(importance_matrix, 0)
        self.importance_matrix_ = importance_matrix

        # Tính ngưỡng
        nonzero_scores = importance_matrix[importance_matrix > 0]
        if len(nonzero_scores) == 0:
            self.edge_threshold_value_ = 0.0
        elif self.edge_threshold_type == "dynamic":
            self.edge_threshold_value_ = float(
                np.percentile(nonzero_scores, self.edge_threshold_percentile)
            )
        else:
            self.edge_threshold_value_ = self.edge_threshold_fixed

        self._is_fitted = True

        if logger:
            n_edges = np.sum(importance_matrix >= self.edge_threshold_value_) // 2
            logger.info(f"  -> Ngưỡng cạnh: {self.edge_threshold_value_:.4f}")
            logger.info(f"  -> Số cạnh vượt ngưỡng: {n_edges:,}")

        return self

    def _fit_random_forest(
        self,
        X: np.ndarray,
        n_genes: int,
        logger: Optional[logging.Logger],
    ) -> np.ndarray:
        """
        Với mỗi gene i, fit RF để predict gene i từ tất cả genes khác.
        Feature importance của gene j = ảnh hưởng của gene j đến gene i.
        """
        if logger:
            logger.info(f"  [RF] Fit {n_genes} Random Forest models...")

        def _fit_one_gene(i: int) -> tuple[int, np.ndarray]:
            y = X[:, i]
            # Binarize target thành nhị phân (cao/thấp) để dùng Classifier
            median_val = np.median(y)
            y_bin = (y > median_val).astype(int)

            # Exclude gene i khỏi features
            X_feat = np.delete(X, i, axis=1)

            rf = RandomForestClassifier(**self.rf_params)
            rf.fit(X_feat, y_bin)
            importances = rf.feature_importances_  # shape (n_genes - 1,)

            # Chèn 0 vào vị trí i để khôi phục shape (n_genes,)
            full_imp = np.insert(importances, i, 0.0)
            return i, full_imp

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_fit_one_gene)(i) for i in range(n_genes)
        )

        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        for i, imp in results:
            importance_matrix[i] = imp

        if logger:
            logger.info(f"  [RF] Hoàn thành. Mean importance: {importance_matrix.mean():.5f}")
        return importance_matrix

    def _fit_mutual_information(
        self,
        X: np.ndarray,
        n_genes: int,
        logger: Optional[logging.Logger],
    ) -> np.ndarray:
        """
        Với mỗi gene i, tính MI giữa gene i và tất cả genes còn lại.
        MI không yêu cầu giả định tuyến tính → bắt được quan hệ phi tuyến.
        """
        if logger:
            logger.info(f"  [MI] Tính mutual information cho {n_genes} genes...")

        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)

        for i in range(n_genes):
            y = X[:, i]
            # Binarize target
            median_val = np.median(y)
            y_bin = (y > median_val).astype(int)

            mi_scores = mutual_info_classif(
                X, y_bin,
                discrete_features=False,
                n_neighbors=3,
                random_state=self.random_state,
            )
            mi_scores[i] = 0.0  # Loại self-loop
            importance_matrix[i] = mi_scores

        if logger:
            logger.info(f"  [MI] Hoàn thành. Mean MI: {importance_matrix.mean():.5f}")
        return importance_matrix

    def _fit_gradient_boosting(
        self,
        X: np.ndarray,
        n_genes: int,
        logger: Optional[logging.Logger],
    ) -> np.ndarray:
        """
        Tương tự RF nhưng dùng Gradient Boosting (chậm hơn, đôi khi chính xác hơn).
        """
        if logger:
            logger.info(f"  [GB] Fit {n_genes} Gradient Boosting models...")

        def _fit_one_gene(i: int) -> tuple[int, np.ndarray]:
            y = X[:, i]
            median_val = np.median(y)
            y_bin = (y > median_val).astype(int)
            X_feat = np.delete(X, i, axis=1)

            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state,
            )
            gb.fit(X_feat, y_bin)
            importances = gb.feature_importances_
            full_imp = np.insert(importances, i, 0.0)
            return i, full_imp

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_fit_one_gene)(i) for i in range(n_genes)
        )

        importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        for i, imp in results:
            importance_matrix[i] = imp

        return importance_matrix

    # ── BƯỚC 2: Xây dựng Adjacency Matrix ─────────────────────────────────
    def build_adjacency(self) -> sp.csr_matrix:
        """Áp dụng ngưỡng để tạo adjacency matrix từ importance_matrix."""
        assert self._is_fitted, "Phải gọi fit() trước!"

        adj = self.importance_matrix_.copy()
        adj[adj < self.edge_threshold_value_] = 0.0
        adj_sparse = sp.csr_matrix(adj, dtype=np.float32)
        self.adjacency_ = adj_sparse
        return adj_sparse

    # ── BƯỚC 3: Lấy danh sách cạnh ─────────────────────────────────────────
    def get_edge_list(self) -> list[tuple[str, str, float]]:
        """Trả về [(gene_a, gene_b, weight)] cho các cạnh vượt ngưỡng."""
        assert self.adjacency_ is not None, "Phải gọi build_adjacency() trước!"
        cx = self.adjacency_.tocoo()
        edges = []
        for i, j, w in zip(cx.row, cx.col, cx.data):
            if i < j:  # Tránh duplicate (vô hướng)
                edges.append((self.genes_[i], self.genes_[j], float(w)))
        return sorted(edges, key=lambda x: x[2], reverse=True)

    def summary(self) -> dict:
        """Thống kê ngắn gọn về mạng đã xây dựng."""
        if not self._is_fitted:
            return {"status": "not_fitted"}
        n_genes = len(self.genes_)
        n_edges = len(self.get_edge_list()) if self.adjacency_ is not None else 0
        return {
            "method": self.method,
            "n_genes": n_genes,
            "n_edges": n_edges,
            "edge_threshold": self.edge_threshold_value_,
            "density": n_edges / (n_genes * (n_genes - 1) / 2) if n_genes > 1 else 0,
        }


# =============================================================================
# Hàm chạy độc lập: Xây mạng cho từng subtype
# =============================================================================
def build_coassociation_per_subtype(
    rnaseq: pd.DataFrame,               # genes x samples (ĐÃ normalized)
    subtype_labels: pd.DataFrame,       # sample_id, subtype
    train_samples: list[str],
    config: dict,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[int, RobustCoAssociationNetwork]:
    """
    Với mỗi subtype, fit RobustCoAssociationNetwork trên TRAIN SAMPLES của subtype đó.

    ⚠️  CRITICAL: Không dùng Test/Val samples để fit mạng!
    """
    nb_cfg = config["network_builder"]
    method = nb_cfg["method"]
    rf_params = {
        "n_estimators": nb_cfg["rf"]["n_estimators"],
        "max_features": nb_cfg["rf"]["max_features"],
        "n_jobs": nb_cfg["rf"]["n_jobs"],
        "random_state": nb_cfg["rf"]["random_state"],
    }
    et = nb_cfg["edge_threshold"]

    subtypes = sorted(subtype_labels["subtype"].unique())
    networks: dict[int, RobustCoAssociationNetwork] = {}

    for subtype in subtypes:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[SUBTYPE {subtype}] Fitting Co-Association Network...")

        # Lấy TRAIN samples của subtype này
        subtype_train = subtype_labels[
            (subtype_labels["subtype"] == subtype) &
            (subtype_labels["sample_id"].isin(train_samples))
        ]["sample_id"].tolist()

        logger.info(f"  -> Train samples của subtype {subtype}: {len(subtype_train)}")

        if len(subtype_train) < 10:
            logger.warning(f"  -> [SKIP] Quá ít mẫu ({len(subtype_train)} < 10) để fit RF/MI!")
            continue

        # Transpose: samples x genes
        avail_samples = [s for s in subtype_train if s in rnaseq.columns]
        X_train = rnaseq[avail_samples].T  # (n_samples, n_genes)

        # Giảm số gen cho RF (dùng top variant genes để tăng tốc)
        gene_var = X_train.var(axis=0)
        top_genes = gene_var.nlargest(min(3000, X_train.shape[1])).index.tolist()
        X_train = X_train[top_genes]
        logger.info(f"  -> Sử dụng {len(top_genes)} high-variance genes")

        # Khởi tạo và fit
        network = RobustCoAssociationNetwork(
            method=method,
            edge_threshold_type=et["type"],
            edge_threshold_percentile=et["percentile"],
            edge_threshold_fixed=et["fixed_value"],
            rf_params=rf_params,
        )
        network.fit(X_train, logger=logger)
        network.build_adjacency()

        # Lưu adjacency sparse matrix
        adj_file = out_dir / f"coassoc_adjacency_subtype{subtype}.npz"
        sp.save_npz(str(adj_file), network.adjacency_)
        logger.info(f"  -> Đã lưu adjacency: {adj_file}")

        # Lưu gene list
        gene_file = out_dir / f"coassoc_genes_subtype{subtype}.txt"
        with open(gene_file, "w") as f:
            f.write("\n".join(network.genes_))

        # Lưu edge list
        edges = network.get_edge_list()
        edge_df = pd.DataFrame(edges, columns=["gene_a", "gene_b", "weight"])
        edge_df.to_csv(out_dir / f"coassoc_edges_subtype{subtype}.csv", index=False)

        summary = network.summary()
        logger.info(f"  -> {summary}")
        networks[subtype] = network

    return networks


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Robust Co-Association Network Builder")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cancer", default="LUAD", choices=["LUAD", "LUSC"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    log_dir       = cfg["paths"]["logs"]
    out_dir       = processed_dir / "coassociation_networks"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir, f"robust_coassociation_{cancer}")
    logger.info("=" * 65)
    logger.info(f"  Robust Co-Association Network — {cancer}")
    logger.info(f"  Method: {cfg['network_builder']['method']}")
    logger.info("=" * 65)

    # Tải dữ liệu
    rnaseq = pd.read_csv(processed_dir / "rnaseq_normalized.csv.gz", index_col=0, compression="gzip")
    subtype_df = pd.read_csv(processed_dir / "subtype_labels.csv")

    with open(processed_dir / "patient_splits.json") as f:
        splits = json.load(f)
    train_samples = splits["train"]

    logger.info(f"RNA-seq shape: {rnaseq.shape}")
    logger.info(f"Subtypes: {sorted(subtype_df['subtype'].unique())}")

    # Xây mạng cho từng subtype
    networks = build_coassociation_per_subtype(
        rnaseq, subtype_df, train_samples, cfg, out_dir, logger
    )

    # Lưu tóm tắt
    summary = {str(k): v.summary() for k, v in networks.items()}
    with open(out_dir / "networks_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"  HOÀN THÀNH: {len(networks)} Co-Association Networks")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
