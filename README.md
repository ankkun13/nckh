# GNN Cancer Driver Gene Identification Pipeline

Hệ thống xác định gen điều khiển ung thư dựa trên Mạng nơ-ron đồ thị (GNN) và dữ liệu đa omics (Multi-omics). Dự án này cải tiến pipeline MODCAN bằng cách thay thế độ đo tương quan Pearson bằng các phương pháp mạnh mẽ hơn (Random Forest/Mutual Information) và chuyển đổi toàn bộ sang Python/PyTorch Geometric.

## 🌟 Tính năng nổi bật

- **Robust Co-association**: Sử dụng Random Forest Feature Importance hoặc Mutual Information để xây dựng mạng lưới gen, chính xác hơn Pearson truyền thống.
- **PyTorch Geometric (PyG)**: Tích hợp GNN hiện đại với hỗ trợ CUDA, giúp tăng tốc huấn luyện trên GPU.
- **Multi-omics Integration**: Kết hợp dữ liệu Biểu hiện gen (EXP), Methylation (MET), CNV, SNV và tương tác protein (PPI).
- **Patient Subgrouping**: Tự động phân nhóm bệnh nhân bằng Similarity Network Fusion (SNF) và Spectral Clustering (thuần Python).
- **Focal Loss**: Giải quyết vấn đề mất cân bằng dữ liệu cực hạn (gen điều khiển chỉ chiếm ~2-5%).
- **Automation**: Tự động hóa toàn bộ từ tiền xử lý đến đánh giá chỉ với một lệnh chạy.

## 📂 Cấu trúc thư mục

```
nckh/
├── configs/config.yaml              # Cấu hình tham số tập trung
├── main.py                          # Script điều phối chính (Orchestrator)
├── Makefile                         # Lệnh chạy nhanh cho từng giai đoạn
├── run_pipeline.sh                  # Script chạy pipeline tự động
├── requirements.txt                 # Danh sách thư viện cần thiết
├── data_processing/                 # Nạp và tiền xử lý dữ liệu
│   ├── data_loader.py               # Tìm và nạp dữ liệu từ thư mục data
│   └── preprocessor.py              # Chuẩn hóa Z-score, loại bỏ NA, lọc gen
├── clustering/                      # Phân nhóm bệnh nhân & Trích xuất đặc trưng
│   ├── spectral_clustering.py       # Thuật toán SNF + Spectral Clustering
│   └── feature_engineering.py       # Tính toán đặc trưng gen & PPI Topology
├── network_builder/                 # Xây dựng mạng lưới co-association
│   └── robust_network.py            # Cốt lõi: Sử dụng RF/MI thay thế Pearson
├── models/                          # Kiến trúc GNN và Huấn luyện
│   ├── gnn_model.py                 # GCN/GAT với skip connections (PyG)
│   ├── focal_loss.py                # Focal Loss cho dữ liệu mất cân bằng
│   └── trainer.py                   # Vòng lặp huấn luyện CUDA & Cross-validation
├── evaluation/                      # Đánh giá và Trực quan hóa
│   ├── metrics.py                   # Tính AUROC, AUPR, Ranking gen
│   └── visualizer.py                # Vẽ biểu đồ ROC, PR, Kaplan-Meier, Hub genes
└── utils/
    └── logger.py                    # Ghi log chi tiết (audit trail)
```

## 🚀 Hướng dẫn cài đặt

Yêu cầu: Python 3.10+, CUDA (nếu dùng GPU).

```bash
# Clone dự án và vào thư mục
cd nckh/

# Cài đặt các thư viện cơ bản
pip install -r requirements.txt

# Lưu ý: Nếu dùng GPU, hãy cài đặt PyTorch phù hợp với phiên bản CUDA của bạn
# Ví dụ cho CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🛠️ Hướng dẫn sử dụng

### 1. Chạy toàn bộ pipeline (End-to-End)
Mặc định sẽ chạy cho loại ung thư `TCGA-LUAD`.

```bash
# Cách 1: Sử dụng Makefile (khuyên dùng)
make all

# Cách 2: Sử dụng Shell script
./run_pipeline.sh

# Cách 3: Chạy trực tiếp main.py
python main.py --config configs/config.yaml
```

### 2. Chạy từng giai đoạn
Bạn có thể chạy riêng lẻ để debug hoặc kiểm tra kết quả trung gian.

```bash
make preprocess   # Chỉ tiền xử lý dữ liệu
make cluster      # Chỉ phân nhóm bệnh nhân
make network      # Xây dựng mạng lưới (giai đoạn tốn thời gian nhất)
make train        # Chỉ huấn luyện GNN
make evaluate     # Chỉ tính toán metrics và vẽ biểu đồ
```

### 3. Tùy chỉnh tham số
Bạn có thể thay đổi loại ung thư hoặc số epoch trực tiếp từ dòng lệnh:

```bash
# Chạy cho ung thư phổi (LUSC) với 500 epoch
make all CANCER=TCGA-LUSC EXTRA_ARGS="--epochs 500"

# Thay đổi phương pháp xây dựng mạng lưới sang Mutual Information
python main.py --method mutual_information
```

## 📊 Kết quả đầu ra

Sau khi chạy, kết quả sẽ được lưu tại thư mục `./results/`:
- `performance_measures.tsv`: AUC/AUPR chi tiết từng fold.
- `gene_ranking.tsv`: Danh sách gen được xếp hạng theo xác suất là gen điều khiển.
- `summary.txt`: Tóm tắt kết quả cuối cùng.
- Các biểu đồ: `roc_curve.png`, `pr_curve.png`, `hub_gene_network.png`,...

---
**NCKH Project - 2026**
