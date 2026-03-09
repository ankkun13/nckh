# =============================================================================
# Dockerfile for MODCAN-GNN Lung Cancer Driver Gene Identification Pipeline
# Base: NVIDIA CUDA 12.1 + Ubuntu 22.04
# Includes: Python 3.10, PyTorch (CUDA), PyTorch Geometric, R + TCGAbiolinks
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="NCKH Research Team"
LABEL description="MODCAN-GNN Pipeline: Multi-omics GNN for lung cancer driver gene identification"
LABEL version="1.0"

# ── Prevent interactive prompts during apt-get ────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    unzip \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    # R and dependencies
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgit2-dev \
    # PDF tools (for later reporting)
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Set Python 3.10 as default ────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Upgrade pip ───────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── Install R packages: BiocManager + TCGAbiolinks ────────────────────────────
# Run as root, install to system library for container-wide access
RUN Rscript -e " \
    options(repos = c(CRAN = 'https://cloud.r-project.org')); \
    install.packages('BiocManager', ask = FALSE, update = FALSE); \
    BiocManager::install(version = '3.18', ask = FALSE, update = FALSE); \
    BiocManager::install(c( \
        'TCGAbiolinks', \
        'SummarizedExperiment', \
        'GenomicRanges', \
        'BiocParallel' \
    ), ask = FALSE, update = FALSE); \
    install.packages(c('dplyr', 'readr', 'jsonlite', 'optparse', 'logger'), \
        ask = FALSE, update = FALSE); \
    "

# ── Python: Core scientific stack ─────────────────────────────────────────────
# Install CPU-independent packages first for Docker layer caching
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scipy==1.13.0 \
    scikit-learn==1.4.2 \
    networkx==3.3 \
    joblib==1.4.0

# ── Python: PyTorch with CUDA 12.1 ───────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Python: PyTorch Geometric and dependencies ────────────────────────────────
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

RUN pip install --no-cache-dir torch-geometric==2.5.3

# ── Python: Bioinformatics & Visualization ───────────────────────────────────
RUN pip install --no-cache-dir \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    lifelines==0.29.0 \
    umap-learn==0.5.6 \
    plotly==5.22.0 \
    # Bioinformatics
    biopython==1.83 \
    pyranges==0.0.129 \
    # Data IO
    h5py==3.11.0 \
    pyarrow==16.0.0 \
    openpyxl==3.1.3 \
    # Utilities
    tqdm==4.66.4 \
    loguru==0.7.2 \
    pyyaml==6.0.1 \
    omegaconf==2.3.0 \
    click==8.1.7 \
    requests==2.32.2 \
    # Jupyter (for notebooks/)
    jupyterlab==4.2.0 \
    ipywidgets==8.1.3

# ── Python: Random Forest / MI based network construction ────────────────────
RUN pip install --no-cache-dir \
    imbalanced-learn==0.12.2 \
    shap==0.45.0 \
    optuna==3.6.1

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /workspace

# ── Create logs directory ─────────────────────────────────────────────────────
RUN mkdir -p /workspace/logs /workspace/data

# ── Environment variables for Python & CUDA ──────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# ── Default command (override via docker-compose) ─────────────────────────────
CMD ["bash"]
