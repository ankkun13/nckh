# [System Role]
Bạn là một Data Engineer có nhiệm vụ viết script Python để tải bộ dữ liệu đa thể học TCGA phục vụ cho dự án nghiên cứu xác định gen gây ung thư phổi. Dữ liệu gốc đã được đóng gói thành **một file nén duy nhất** và đẩy lên **một nền tảng storage** (Google Drive, MinIO, hoặc URL trực tiếp). Script cần tự động tải về, giải nén, **sàng lọc chỉ giữ dữ liệu LUAD và LUSC**, rồi tổ chức vào thư mục `data/` của dự án.

# [Bối cảnh dự án]
- Dự án **CHỈ** nghiên cứu trên 2 loại ung thư phổi: **LUAD** (Lung Adenocarcinoma) và **LUSC** (Lung Squamous Cell Carcinoma).
- Bộ dữ liệu gốc trên storage chứa dữ liệu của **10 loại ung thư** (BLCA, BRCA, COAD, HNSC, KIRC, KIRP, LUAD, LUSC, PRAD, UCEC) — tổng cộng 214 files.
- **Yêu cầu bắt buộc**: Sau khi script chạy xong, thư mục `data/` chỉ được chứa **đúng 48 files** liên quan đến LUAD, LUSC, và tài nguyên chung (network + reference + survival). **Không được có dữ liệu của bất kỳ loại ung thư nào khác** (BLCA, BRCA, COAD, HNSC, KIRC, KIRP, PRAD, UCEC).

# [Cấu trúc dữ liệu gốc trên storage]

File nén (`.tar.gz` hoặc `.zip`) khi giải nén sẽ có cấu trúc sau:

```
data/
├── TCGA_UCSC_EXP/           # RNA-seq gene expression (log2-transformed)
│   ├── TCGA-LUAD/           # ← GIỮ
│   │   ├── TCGA-LUAD_exp_data.txt    # Ma trận genes × samples (tab-separated, không header)
│   │   ├── TCGA-LUAD_gene.txt        # Danh sách gen (1 gene/dòng, tương ứng rows)
│   │   └── TCGA-LUAD_sample.txt      # Danh sách sample barcode (1 barcode/dòng, tương ứng cols)
│   ├── TCGA-LUSC/           # ← GIỮ
│   ├── TCGA-BLCA/           # ← XÓA
│   ├── TCGA-BRCA/           # ← XÓA
│   └── ... (8 loại khác)   # ← XÓA
│
├── TCGA_UCSC_MET/           # DNA Methylation beta values (0–1)
│   ├── TCGA-LUAD/           # ← GIỮ (3 files: _met_data.txt, _gene.txt, _sample.txt)
│   ├── TCGA-LUSC/           # ← GIỮ
│   └── ... (8 loại khác)   # ← XÓA
│
├── TCGA_hg38_SNV/           # Binary somatic mutation matrix (hg38)
│   └── snv_matrix/
│       ├── TCGA-LUAD_snv_matrix.txt  # ← GIỮ (binary 0/1, genes × samples)
│       ├── TCGA-LUAD_snv_gene.txt    # ← GIỮ
│       ├── TCGA-LUAD_snv_sample.txt  # ← GIỮ
│       ├── TCGA-LUSC_snv_*           # ← GIỮ
│       ├── TCGA-BLCA_snv_*           # ← XÓA
│       └── ...                       # ← XÓA
│
├── TCGA_UCSC_CNV/           # Copy Number Variation (binary)
│   └── cnv_matrix/
│       ├── TCGA-LUAD_cnv_*           # ← GIỮ
│       ├── TCGA-LUSC_cnv_*           # ← GIỮ
│       └── ...                       # ← XÓA
│
├── TCGA_Mutation/           # Mutation format (có overlap với SNV)
│   ├── TCGA-LUAD/           # ← GIỮ (3 files: _mutation.txt, _gene.txt, _sample.txt)
│   ├── TCGA-LUSC/           # ← GIỮ
│   └── ... (8 loại khác)   # ← XÓA
│
├── TCGA_UCSC_normal/        # Normal tissue expression + methylation
│   ├── TCGA-LUAD/           # ← GIỮ (5 files: _exp_normal, _exp_sample, _gene, _met_normal, _met_sample)
│   ├── TCGA-LUSC/           # ← GIỮ
│   └── ... (8 loại khác)   # ← XÓA
│
├── survival/                # Clinical survival data (OS, OS.time)
│   ├── TCGA-LUAD.survival.tsv  # ← GIỮ
│   ├── TCGA-LUSC.survival.tsv  # ← GIỮ
│   ├── TCGA-BLCA.survival.tsv  # ← XÓA
│   └── ...                     # ← XÓA
│
├── network/                 # STRING PPI (tài nguyên chung — KHÔNG phân biệt cancer type)
│   ├── string_full_v12.txt      # ← GIỮ (13.7M edges, đầy đủ)
│   └── string_full_v12_0.7.txt  # ← GIỮ (4.1M edges, đã lọc score ≥ 700)
│
└── reference/               # Driver/Non-driver gene labels (tài nguyên chung)
    ├── 579_CGC.txt              # ← GIỮ (579 COSMIC Cancer Gene Census)
    └── 2179_non_driver_genes.txt # ← GIỮ (2179 non-driver genes)
```

# [Cấu trúc đích sau khi script chạy xong]

Thư mục `data/` cần đạt đúng cấu trúc này — **sạch sẽ, không thừa**:

```
data/
├── TCGA_UCSC_EXP/
│   ├── TCGA-LUAD/  (3 files)
│   └── TCGA-LUSC/  (3 files)
├── TCGA_UCSC_MET/
│   ├── TCGA-LUAD/  (3 files)
│   └── TCGA-LUSC/  (3 files)
├── TCGA_hg38_SNV/
│   └── snv_matrix/ (6 files: LUAD + LUSC)
├── TCGA_UCSC_CNV/
│   └── cnv_matrix/ (6 files: LUAD + LUSC)
├── TCGA_Mutation/
│   ├── TCGA-LUAD/  (3 files)
│   └── TCGA-LUSC/  (3 files)
├── TCGA_UCSC_normal/
│   ├── TCGA-LUAD/  (5 files)
│   └── TCGA-LUSC/  (5 files)
├── survival/       (2 files: LUAD + LUSC)
├── network/        (2 files: STRING PPI)
└── reference/      (2 files: CGC + non-driver)
```

**Tổng cộng: 48 files. KHÔNG CÓ thư mục hoặc file nào khác.**

# [Yêu cầu kỹ thuật cho script]

## 1. Tải dữ liệu từ storage
- Hỗ trợ nhiều nguồn: **Google Drive** (dùng `gdown`), **MinIO/S3** (dùng `boto3`), hoặc **URL trực tiếp** (`wget`/`requests`).
- URL nguồn và phương thức tải cần được cấu hình qua `configs/config.yaml` hoặc biến môi trường, **không được hardcode**.
- Hiển thị **progress bar** khi tải (dùng `tqdm`).
- Nếu file đã tải sẵn → **skip tải lại** (kiểm tra hash MD5/SHA256 nếu có).

## 2. Giải nén
- Hỗ trợ: `.tar.gz`, `.tar.bz2`, `.zip`.
- Giải nén vào một **thư mục tạm** (tempdir), **không** giải nén thẳng vào `data/`.

## 3. Sàng lọc — **CRITICAL**
- Danh sách cancer types hợp lệ được định nghĩa rõ ràng: `ALLOWED_CANCERS = ["LUAD", "LUSC"]`.
- Quá trình sàng lọc spải kiểm tra **từng file và thư mục**:
  - **Thư mục con** trong `TCGA_UCSC_EXP/`, `TCGA_UCSC_MET/`, `TCGA_Mutation/`, `TCGA_UCSC_normal/`: Chỉ copy `TCGA-LUAD/` và `TCGA-LUSC/`. Bỏ qua mọi thư mục `TCGA-BLCA/`, `TCGA-BRCA/`, v.v.
  - **Files trong flat directories** (`snv_matrix/`, `cnv_matrix/`): Chỉ copy file có pattern `TCGA-LUAD_*` hoặc `TCGA-LUSC_*`.
  - **Survival**: Chỉ copy `TCGA-LUAD.survival.tsv` và `TCGA-LUSC.survival.tsv`.
  - **Network & Reference**: Copy toàn bộ (dữ liệu chung, không phân biệt cancer type).
- **Sau khi sàng lọc xong**: Chạy kiểm tra tự động (assert) để đảm bảo:
  - Không tồn tại bất kỳ file nào chứa tên BLCA, BRCA, COAD, HNSC, KIRC, KIRP, PRAD, UCEC trong đường dẫn.
  - Tổng số file khớp với số lượng kỳ vọng (48 files).
  - Xóa thư mục tạm giải nén.

## 4. Validation & Integrity Check
- Sau khi copy xong, kiểm tra **tính toàn vẹn dữ liệu**:
  - Mỗi bộ 3 file (matrix + gene + sample) phải có kích thước nhất quán:
    `rows(matrix) == lines(gene_file)` và `cols(matrix) == lines(sample_file)`.
  - Không có file rỗng (0 bytes).
- In ra bảng tóm tắt:
  ```
  ┌────────────────────┬──────────┬─────────┬──────────┐
  │ Data Type          │ Cancer   │ Genes   │ Samples  │
  ├────────────────────┼──────────┼─────────┼──────────┤
  │ TCGA_UCSC_EXP      │ LUAD     │ 18,057  │ 463      │
  │ TCGA_UCSC_EXP      │ LUSC     │ 18,057  │ 364      │
  │ ...                │ ...      │ ...     │ ...      │
  └────────────────────┴──────────┴─────────┴──────────┘
  ```

## 5. Logging & Reproducibility
- Ghi log chi tiết tất cả các bước (download, decompress, filter, validate) vào `logs/data_loader_YYYYMMDD_HHMMSS.log`.
- Lưu metadata vào `data/DATA_MANIFEST.json`:
  ```json
  {
    "download_url": "...",
    "download_date": "2026-03-09T22:00:00",
    "archive_hash": "sha256:abc...",
    "allowed_cancers": ["LUAD", "LUSC"],
    "total_files": 48,
    "files": [
      {"path": "TCGA_UCSC_EXP/TCGA-LUAD/TCGA-LUAD_exp_data.txt", "size_bytes": 123456},
      ...
    ]
  }
  ```

## 6. Cấu hình qua config.yaml
Thêm section `data_source` trong `configs/config.yaml`:
```yaml
data_source:
  method: "gdrive"            # "gdrive" | "s3" | "url"
  url: "https://drive.google.com/..."   # URL/ID tùy method
  archive_name: "tcga_multi_omics.tar.gz"
  archive_hash: null           # SHA256 hash (nếu có, dùng để verify)
  allowed_cancers: ["LUAD", "LUSC"]
  data_dir: "data"             # Thư mục đích trong project
  temp_dir: "/tmp/tcga_download"   # Thư mục tạm giải nén
  skip_if_exists: true         # Bỏ qua nếu data/ đã đầy đủ
```

## 7. CLI Interface
```bash
# Tải + giải nén + sàng lọc
python data_acquisition/data_loader.py --config configs/config.yaml

# Chỉ validate (không tải lại)
python data_acquisition/data_loader.py --config configs/config.yaml --validate-only

# Force re-download
python data_acquisition/data_loader.py --config configs/config.yaml --force
```

# [Format định dạng dữ liệu — Reference]

Tất cả các file dữ liệu ma trận đều dùng cùng pattern **bộ 3 file**:

| File | Mô tả | Format |
|---|---|---|
| `*_data.txt` / `*_matrix.txt` / `*_mutation.txt` / `*_normal.txt` | Ma trận dữ liệu | tab-separated, **không header**, shape `(n_genes × n_samples)` |
| `*_gene.txt` | Danh sách gen | 1 gene symbol / dòng, tương ứng **rows** của ma trận |
| `*_sample.txt` | Danh sách mẫu | 1 TCGA barcode / dòng, tương ứng **columns** của ma trận |

File survival: `TCGA-LUAD.survival.tsv` — tab-separated **có header**: `sample`, `OS`, `_PATIENT`, `OS.time`.

File PPI: `string_full_v12_0.7.txt` — tab-separated **có header**: `protein_1`, `protein_2`, `score`.

File reference: 1 gene symbol / dòng, **không header**.

# [Output mong đợi]

Khi chạy xong script, console phải in ra tóm tắt:

```
✅ Data Loader — Hoàn tất
   Cancer types: LUAD, LUSC
   Total files:  48 / 48 (expected)
   REJECTED:     168 files (8 cancer types loại bỏ)
   Validation:   PASSED (tất cả matrix shapes nhất quán)
   Manifest:     data/DATA_MANIFEST.json
```
