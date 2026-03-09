#!/usr/bin/env Rscript
# =============================================================================
# download_tcga.R — Tải dữ liệu đa thể học từ TCGA bằng TCGAbiolinks
#
# Sử dụng:
#   Rscript download_tcga.R --cancer LUAD --outdir /workspace/data/raw/tcga_luad
#   Rscript download_tcga.R --cancer LUSC --outdir /workspace/data/raw/tcga_lusc
#
# Dữ liệu tải về:
#   1. RNA-Seq gene expression (HTSeq - Counts, log2 transformed)
#   2. DNA Methylation (450k BeadChip, beta values)
#   3. Somatic Mutation (SNV/Indel - MAF format)
#   4. Clinical data (bệnh nhân, survival)
# =============================================================================

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(dplyr)
  library(readr)
  library(jsonlite)
  library(optparse)
  library(logger)
})

# ── Argument Parsing ──────────────────────────────────────────────────────────
option_list <- list(
  make_option(c("--cancer"),
    type    = "character",
    default = "LUAD",
    help    = "Loại ung thư: LUAD hoặc LUSC [default: %default]"
  ),
  make_option(c("--outdir"),
    type    = "character",
    default = "/workspace/data/raw/tcga_luad",
    help    = "Thư mục lưu dữ liệu tải về [default: %default]"
  ),
  make_option(c("--log_dir"),
    type    = "character",
    default = "/workspace/logs",
    help    = "Thư mục lưu log [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

# ── Setup Logging ─────────────────────────────────────────────────────────────
dir.create(opt$log_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(opt$outdir,  showWarnings = FALSE, recursive = TRUE)

log_file <- file.path(
  opt$log_dir,
  paste0("download_tcga_", opt$cancer, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")
)

log_appender(appender_file(log_file))
log_threshold(INFO)

log_info("================================================================")
log_info("  TCGA Data Download Script — Dự án MODCAN-GNN")
log_info("================================================================")
log_info("Cancer type : {opt$cancer}")
log_info("Output dir  : {opt$outdir}")
log_info("Log file    : {log_file}")
log_info("TCGAbiolinks version: {packageVersion('TCGAbiolinks')}")

# ── Xác định project TCGA ─────────────────────────────────────────────────────
TCGA_PROJECT <- paste0("TCGA-", opt$cancer)
OUTDIR       <- opt$outdir

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: Tải và lưu một loại dữ liệu omics
# ─────────────────────────────────────────────────────────────────────────────
download_and_save <- function(query_fn, save_name, transform_fn = NULL) {
  out_path <- file.path(OUTDIR, paste0(save_name, ".rds"))

  if (file.exists(out_path)) {
    log_info("[SKIP] {save_name}.rds đã tồn tại, bỏ qua tải lại.")
    return(readRDS(out_path))
  }

  log_info("[START] Đang tải: {save_name} ...")
  tryCatch({
    query <- query_fn()
    log_info("  -> Số mẫu tìm thấy: {nrow(getResults(query))}")

    GDCdownload(query,
      method    = "api",
      files.per.chunk = 10,
      directory = file.path(OUTDIR, "GDCdata")
    )
    log_info("  -> Tải xong, đang chuẩn bị dữ liệu...")

    data <- GDCprepare(query, directory = file.path(OUTDIR, "GDCdata"))

    if (!is.null(transform_fn)) {
      data <- transform_fn(data)
    }

    saveRDS(data, out_path)
    log_info("  -> Đã lưu: {out_path}")
    return(data)

  }, error = function(e) {
    log_error("  -> Lỗi khi tải {save_name}: {conditionMessage(e)}")
    stop(e)
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. RNA-Seq — Gene Expression
# ─────────────────────────────────────────────────────────────────────────────
log_info("────────────────────────────────────────")
log_info("[MODULE 1/4] RNA-Seq Gene Expression")
log_info("────────────────────────────────────────")

rnaseq_data <- download_and_save(
  query_fn = function() {
    GDCquery(
      project                 = TCGA_PROJECT,
      data.category           = "Transcriptome Profiling",
      data.type               = "Gene Expression Quantification",
      workflow.type           = "STAR - Counts",
      sample.type             = c("Primary Tumor", "Solid Tissue Normal")
    )
  },
  save_name = "rnaseq_raw",
  transform_fn = function(se) {
    # Trích xuất ma trận counts từ SummarizedExperiment
    log_info("  -> Trích xuất matrix counts từ SummarizedExperiment...")
    counts_mat <- assay(se, "unstranded")
    col_data   <- as.data.frame(colData(se))
    list(
      counts    = as.data.frame(counts_mat),
      col_data  = col_data,
      row_data  = as.data.frame(rowData(se))
    )
  }
)

# Lưu thêm dạng CSV cho Python pipeline
if (!is.null(rnaseq_data$counts)) {
  write_csv(
    tibble::rownames_to_column(rnaseq_data$counts, "gene_id"),
    file.path(OUTDIR, "rnaseq_counts.csv")
  )
  write_csv(rnaseq_data$col_data, file.path(OUTDIR, "rnaseq_metadata.csv"))
  log_info("  -> Đã export CSV: rnaseq_counts.csv, rnaseq_metadata.csv")
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DNA Methylation — 450k BeadChip
# ─────────────────────────────────────────────────────────────────────────────
log_info("────────────────────────────────────────")
log_info("[MODULE 2/4] DNA Methylation (450k)")
log_info("────────────────────────────────────────")

meth_data <- download_and_save(
  query_fn = function() {
    GDCquery(
      project         = TCGA_PROJECT,
      data.category   = "DNA Methylation",
      data.type       = "Methylation Beta Value",
      platform        = "Illumina Human Methylation 450",
      sample.type     = c("Primary Tumor", "Solid Tissue Normal")
    )
  },
  save_name = "methylation_raw",
  transform_fn = function(se) {
    log_info("  -> Trích xuất beta values từ SummarizedExperiment...")
    beta_mat  <- assay(se)
    col_data  <- as.data.frame(colData(se))
    row_data  <- as.data.frame(rowData(se))
    list(
      beta_values = as.data.frame(beta_mat),
      col_data    = col_data,
      row_data    = row_data
    )
  }
)

if (!is.null(meth_data$beta_values)) {
  # File methylation rất lớn (~480k CpG sites), lưu dạng compressed CSV
  data.table::fwrite(
    tibble::rownames_to_column(meth_data$beta_values, "cpg_id"),
    file.path(OUTDIR, "methylation_beta.csv.gz"),
    compress = "gzip"
  )
  write_csv(meth_data$col_data, file.path(OUTDIR, "methylation_metadata.csv"))
  log_info("  -> Đã export (compressed): methylation_beta.csv.gz")
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Somatic Mutation — MAF file (SNV/Indel)
# ─────────────────────────────────────────────────────────────────────────────
log_info("────────────────────────────────────────")
log_info("[MODULE 3/4] Somatic Mutation (MAF)")
log_info("────────────────────────────────────────")

mutation_data <- download_and_save(
  query_fn = function() {
    GDCquery(
      project             = TCGA_PROJECT,
      data.category       = "Simple Nucleotide Variation",
      data.type           = "Masked Somatic Mutation",
      workflow.type       = "Aliquot Ensemble Somatic Variant Merging and Masking",
      sample.type         = "Primary Tumor"
    )
  },
  save_name = "mutation_raw"
  # GDCprepare trả về data.frame cho MAF, không cần transform thêm
)

if (!is.null(mutation_data)) {
  # Trích ma trận nhị phân: gen x bệnh nhân (1 = đột biến, 0 = không)
  log_info("  -> Tạo binary mutation matrix (gene x sample)...")

  if (is.data.frame(mutation_data)) {
    # Chọn cột quan trọng từ MAF
    key_cols <- c(
      "Hugo_Symbol", "Tumor_Sample_Barcode",
      "Variant_Classification", "Variant_Type",
      "HGVSp_Short"
    )
    key_cols_present <- intersect(key_cols, colnames(mutation_data))

    maf_slim <- mutation_data[, key_cols_present]
    write_csv(maf_slim, file.path(OUTDIR, "mutations_maf.csv"))
    log_info("  -> Đã export: mutations_maf.csv (slim MAF)")
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clinical Data
# ─────────────────────────────────────────────────────────────────────────────
log_info("────────────────────────────────────────")
log_info("[MODULE 4/4] Clinical Data")
log_info("────────────────────────────────────────")

clinical_data <- download_and_save(
  query_fn = function() {
    GDCquery_clinic(
      project = TCGA_PROJECT,
      type    = "clinical"
    )
    # Trả về data.frame trực tiếp từ API
  },
  save_name = "clinical_raw"
)

# Lưu lâm sàng dạng CSV để Python dùng cho Kaplan-Meier
if (is.data.frame(clinical_data)) {
  # Chọn cột liên quan đến survival analysis
  survival_cols <- c(
    "submitter_id",
    "gender", "age_at_index",
    "vital_status",
    "days_to_death", "days_to_last_follow_up",
    "ajcc_pathologic_stage",
    "primary_diagnosis"
  )
  survival_cols_present <- intersect(survival_cols, colnames(clinical_data))
  clinical_slim <- clinical_data[, survival_cols_present]
  write_csv(clinical_slim, file.path(OUTDIR, "clinical.csv"))
  log_info("  -> Đã export: clinical.csv ({nrow(clinical_slim)} bệnh nhân, {ncol(clinical_slim)} cột)")
}

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log_info("================================================================")
log_info("  TÓM TẮT KẾT QUẢ TẢI DỮ LIỆU — {TCGA_PROJECT}")
log_info("================================================================")

summary_info <- list(
  project         = TCGA_PROJECT,
  download_time   = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  output_dir      = OUTDIR,
  files_created   = list.files(OUTDIR, pattern = "\\.csv|\\.rds|\\.gz", recursive = FALSE)
)

write_json(summary_info, file.path(OUTDIR, "download_summary.json"), pretty = TRUE)
log_info("  -> Tóm tắt: {file.path(OUTDIR, 'download_summary.json')}")
log_info("================================================================")
log_info("  HOÀN THÀNH! Data acquisition cho {TCGA_PROJECT}")
log_info("================================================================")
