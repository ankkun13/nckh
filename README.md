# MODCAN-GNN: Cải tiến mạng lưới đồng liên kết vi phân cho định vị gen gây ung thư phổi

> **Dự án NCKH** — *Bioinformatics & Graph Neural Networks*
>
> Cải tiến phương pháp xây dựng mạng lưới đồng liên kết vi phân trong kiến trúc MODCAN để định vị gen gây ung thư phổi theo nhóm phụ phân tử (LUAD/LUSC).

---

## 📌 Đổi mới cốt lõi

| Thành phần | MODCAN gốc | Dự án này |
|---|---|---|
| **Mạng đồng liên kết** | Pearson Correlation | **Random Forest / Mutual Information** |
| **Xử lý Heterophily** | ✗ | **Representation Separation (SGCD)** |
| **Chống Over-smoothing** | ✗ | **Initial Residual Connection (MLGCN-Driver)** |
| **Mất cân bằng nhãn** | Cross Entropy | **Focal Loss (α=0.75, γ=2)** |

---

## 🏗️ Cấu trúc dự án

```
nckh/
├── docker/                     # Dockerfile + docker-compose (CUDA 12.1)
├── configs/config.yaml         # Toàn bộ tham số pipeline
├── data_acquisition/
│   ├── download_tcga.R         # TCGAbiolinks: RNA-seq, Methylation, SNV, Clinical
│   ├── download_ppi.py         # STRING v12 PPI network
│   ├── download_labels.py      # NCG7 + COSMIC CGC driver gene labels
│   └── preprocess.py           # NA removal, log2, Z-score, patient split
├── clustering/
│   ├── community_cohesion.py   # WGCNA-style Zsummary scores
│   └── spectral_clustering.py  # Spectral Clustering + Silhouette + KM plot
├── network_builder/            # 🔑 CẢI TIẾN CỐT LÕI
│   ├── robust_coassociation.py # RobustCoAssociationNetwork (RF/MI)
│   └── ppi_integration.py      # Fuse co-assoc + STRING PPI
├── models/
│   ├── gnn.py                  # MODCANGNNModel (GAT + Residual + Sep.)
│   ├── focal_loss.py           # Focal Loss
│   ├── utils.py                # Graph data loader + Node2Vec
│   └── train.py                # Training loop + Early stopping
├── evaluation/
│   ├── metrics.py              # AUROC, AUPR, F1, ROC/PR curves
│   ├── survival_analysis.py    # Kaplan-Meier (lifelines)
│   └── network_visualization.py # Hub gene network (networkx)
└── requirements.txt
```

---

## 🚀 Chạy nhanh với Docker

### Yêu cầu
- Docker + `nvidia-container-toolkit` (NVIDIA GPU)
- CUDA driver >= 12.1

### Khởi động

```bash
cd nckh/docker

# Build image (lần đầu ~15-20 phút)
docker compose build

# Khởi động Jupyter Lab tại http://localhost:8888
docker compose up jupyter
```

### Chạy toàn bộ pipeline

```bash
# Bước 1: Thu thập dữ liệu
docker compose run --rm data_acquisition

# Bước 2-3: Preprocessing + Clustering
docker compose run --rm preprocess
docker compose run --rm clustering

# Bước 4: Training GNN (GPU)
docker compose run --rm train
```

### Hoặc chạy từng bước thủ công

```bash
docker compose run --rm pipeline bash

# Bên trong container:
Rscript data_acquisition/download_tcga.R --cancer LUAD
python data_acquisition/preprocess.py --cancer LUAD
python clustering/community_cohesion.py --cancer LUAD
python clustering/spectral_clustering.py --cancer LUAD
python network_builder/robust_coassociation.py --cancer LUAD
python network_builder/ppi_integration.py --cancer LUAD
python models/train.py --cancer LUAD
python evaluation/survival_analysis.py --cancer LUAD
python evaluation/network_visualization.py --cancer LUAD --subtype 0
```

---

## ⚙️ Cấu hình (`configs/config.yaml`)

Tất cả tham số được quản lý tập trung. Các tham số quan trọng:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `network_builder.method` | `random_forest` | Thuật toán xây mạng: `random_forest`, `mutual_information`, `gradient_boosting` |
| `network_builder.edge_threshold.percentile` | `90` | Ngưỡng động tạo cạnh (top 10%) |
| `clustering.k_range` | `[2, 8]` | Khoảng K tìm kiếm |
| `model.num_layers` | `2` | Số lớp GNN (tối đa 3) |
| `model.architecture` | `gat` | `gat` hoặc `gcn` |
| `model.focal_loss.alpha` | `0.75` | Trọng số lớp positive (driver) |
| `training.epochs` | `500` | Số epochs |
| `training.early_stopping.patience` | `50` | Patience cho early stopping |

---

## 🛡️ Tránh Data Leakage

> **Nguyên tắc bắt buộc:** Random Forest / Mutual Information để tính trọng số cạnh trong mạng co-association **CHỈ được fit trên train set**.

Chi tiết trong `network_builder/robust_coassociation.py`:

```python
# ✅ ĐÚNG — chỉ dùng train_samples
network.fit(X_train, logger=logger)

# ❌ SAI — không được dùng test/val để fit
# network.fit(X_all, logger=logger)
```

---

## 📊 Đầu ra

| File | Mô tả |
|---|---|
| `results/LUAD/predictions_subtypeN.csv` | Xác suất driver cho từng gen |
| `results/LUAD/training_history_subN.csv` | AUROC/AUPR theo epoch |
| `results/LUAD/silhouette_report.png` | Biểu đồ chọn K tối ưu |
| `results/LUAD/kaplan_meier_LUAD.png` | KM survival curves |
| `results/LUAD/hub_gene_network_subtype0.png` | Hub gene network |
| `logs/*.log` | Pipeline logs đầy đủ |

---

## 📚 Tài liệu tham khảo

1. **MODCAN**: Li et al., *BMC Bioinformatics* 2026 — `ai-agent/s12859-025-06331-w.pdf`
2. **SGCD**: Li et al., *Briefings in Bioinformatics* 2025 — `ai-agent/bbae691.pdf`
3. **MLGCN-Driver**: Wei et al., *BMC Bioinformatics* 2025 — `ai-agent/s12859-025-06260-8.pdf`

---

## 📝 Ghi chú lâm sàng

Dữ liệu y tế cần tuân thủ quy định TCGA Data Access. Mọi kết quả dự đoán **chỉ mang tính nghiên cứu**, không dùng cho chẩn đoán lâm sàng.
