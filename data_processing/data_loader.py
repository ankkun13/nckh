"""
Multi-Omics Data Loader for TCGA Cancer Datasets
=================================================
Auto-discovers and loads cancer type data from the data directory:
  - Gene Expression (EXP)
  - Promoter Methylation (MET)
  - Copy Number Variation (CNV)
  - Single Nucleotide Variation (SNV) / Mutation
  - Normal tissue data (EXP, MET)
  - PPI network (STRINGv12)
  - Reference driver genes (CGC, non-driver)
  - Clinical survival data
"""

import os
import logging
import tarfile
import tempfile
import requests
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from utils.logger import log_tensor_info


logger = logging.getLogger("pipeline")


class MultiOmicsDataLoader:
    """
    Loads and structures multi-omics data for a specified cancer type.
    Data is expected in the MODCAN directory layout under `data.base_path`.
    Automatically downloads and extracts the data archive if missing.
    """
    
    def __init__(self, config: dict):
        self.base_path = Path(config['data']['base_path'])
        self.ppi_file = config['data']['ppi_network']
        self.driver_file = config['data']['driver_genes']
        self.non_driver_file = config['data']['non_driver_genes']
        self.storage_url = config['data'].get('github_storage_url')
        
        # Download and extract data first
        self._download_and_extract()
    
    def _download_and_extract(self):
        """
        Download the data archive from the storage URL and extract it.
        The archive is expected to contain a 'data/' directory.
        """
        if self.base_path.exists() and any(self.base_path.iterdir()):
            logger.info("Data already exists at %s, skipping download.", self.base_path)
            return

        if not self.storage_url:
            logger.warning("Data missing and no storage_url provided in config.")
            return

        # Convert GitHub blob URL to raw if necessary
        url = self.storage_url
        if "github.com" in url and "/blob/" in url:
            # url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            url = url

        logger.info("Downloading data from %s", url)
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                archive_file = tmp_path / "data.tar.gz"
                
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(archive_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info("Extracting data...")
                with tarfile.open(archive_file, "r:gz") as tar:
                    tar.extractall(path=tmp_path)
                
                # The archive might contain a 'data' folder or files directly
                # We expect a 'data' folder based on user description
                extracted_data = tmp_path / "data"
                if not extracted_data.exists():
                    # Fallback search if it's nested or named differently
                    for item in tmp_path.iterdir():
                        if item.is_dir() and (item / "network").exists():
                            extracted_data = item
                            break
                    else:
                        extracted_data = tmp_path

                # Move contents to base_path
                self.base_path.mkdir(parents=True, exist_ok=True)
                for item in extracted_data.iterdir():
                    if item.name == "data.tar.gz": continue # skip the archive itself if extracted in same dir
                    dest = self.base_path / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(dest))
                
                logger.info("Data successfully loaded to %s", self.base_path)
        except Exception as e:
            logger.error("Failed to download or extract data: %s", str(e))
            raise

    def load(self, cancer_type: str) -> dict:
        """
        Load all data for a given cancer type.
        """
        logger.info("Loading data for cancer type: %s", cancer_type)
        
        data = {}
        
        # --- Expression data (tumor) ---
        data['exp_data'], data['exp_genes'], data['exp_samples'] = \
            self._load_omics_matrix('TCGA_UCSC_EXP', cancer_type, 'exp')
        
        # --- Methylation data (tumor) ---
        data['met_data'], data['met_genes'], data['met_samples'] = \
            self._load_omics_matrix('TCGA_UCSC_MET', cancer_type, 'met')
        
        # --- CNV data ---
        data['cnv_data'], data['cnv_genes'], data['cnv_samples'] = \
            self._load_cnv_matrix(cancer_type)
        
        # --- SNV data ---
        data['snv_data'], data['snv_genes'], data['snv_samples'] = \
            self._load_snv_matrix(cancer_type)
        
        # --- Mutation data ---
        data['mut_data'], data['mut_genes'], data['mut_samples'] = \
            self._load_mutation_matrix(cancer_type)
        
        # --- Normal tissue data ---
        data['exp_normal'], data['normal_exp_genes'], data['normal_exp_samples'] = \
            self._load_normal_data(cancer_type, 'exp')
        data['met_normal'], data['normal_met_genes'], data['normal_met_samples'] = \
            self._load_normal_data(cancer_type, 'met')
        
        # --- PPI network ---
        data['ppi'] = self._load_ppi_network()
        
        # --- Reference driver genes ---
        data['driver_genes'] = self._load_driver_genes()
        data['non_driver_genes'] = self._load_non_driver_genes()
        
        # --- Survival data ---
        data['survival'] = self._load_survival_data(cancer_type)
        
        data['cancer_type'] = cancer_type
        
        self._log_data_summary(data)
        return data
    
    def _load_omics_matrix(self, data_dir: str, cancer_type: str,
                           data_type: str) -> tuple:
        """Load omics matrix (EXP or MET) from TCGA_UCSC format."""
        dir_path = self.base_path / data_dir / cancer_type
        
        data_file = dir_path / f"{cancer_type}_{data_type}_data.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Data file not found: %s", data_file)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        try:
            matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
            # Ensure matrix is 2D even if 1 sample
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
        except Exception as e:
            logger.error("Error loading %s: %s", data_file, e)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        
        logger.info("Loaded %s %s: matrix=%s", cancer_type, data_type, matrix.shape)
        return matrix, genes, samples
    
    def _load_cnv_matrix(self, cancer_type: str) -> tuple:
        """Load CNV data from cnv_matrix directory."""
        dir_path = self.base_path / 'TCGA_UCSC_CNV' / 'cnv_matrix'
        data_file = dir_path / f"{cancer_type}_cnv_matrix.txt"
        gene_file = dir_path / f"{cancer_type}_cnv_gene.txt"
        sample_file = dir_path / f"{cancer_type}_cnv_sample.txt"
        
        if not data_file.exists():
            logger.warning("CNV data file not found: %s", data_file)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        try:
            matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
            if matrix.ndim == 1: matrix = matrix.reshape(-1, 1)
        except Exception as e:
            logger.error("Error loading CNV %s: %s", data_file, e)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])

        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        return matrix, genes, samples
    
    def _load_snv_matrix(self, cancer_type: str) -> tuple:
        """Load SNV data from snv_matrix directory."""
        dir_path = self.base_path / 'TCGA_hg38_SNV' / 'snv_matrix'
        data_file = dir_path / f"{cancer_type}_snv_matrix.txt"
        gene_file = dir_path / f"{cancer_type}_snv_gene.txt"
        sample_file = dir_path / f"{cancer_type}_snv_sample.txt"
        
        if not data_file.exists():
            logger.warning("SNV data file not found: %s", data_file)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        try:
            matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
            if matrix.ndim == 1: matrix = matrix.reshape(-1, 1)
        except Exception as e:
            logger.error("Error loading SNV %s: %s", data_file, e)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])

        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        return matrix, genes, samples
    
    def _load_mutation_matrix(self, cancer_type: str) -> tuple:
        """Load mutation data from TCGA_Mutation directory."""
        dir_path = self.base_path / 'TCGA_Mutation' / cancer_type
        data_file = dir_path / f"{cancer_type}_mutation.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Mutation data file not found: %s", data_file)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        try:
            matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
            if matrix.ndim == 1: matrix = matrix.reshape(-1, 1)
        except Exception as e:
            logger.error("Error loading Mutation %s: %s", data_file, e)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])

        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        return matrix, genes, samples
    
    def _load_normal_data(self, cancer_type: str, data_type: str) -> tuple:
        """Load normal tissue data (EXP or MET)."""
        dir_path = self.base_path / 'TCGA_UCSC_normal' / cancer_type
        data_file = dir_path / f"{cancer_type}_{data_type}_normal.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Normal %s data not found: %s", data_type, data_file)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])
        
        try:
            matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
            if matrix.ndim == 1: matrix = matrix.reshape(-1, 1)
        except Exception as e:
            logger.error("Error loading normal %s %s: %s", data_type, data_file, e)
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([])

        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        
        if sample_file.exists():
            samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        else:
            samples = np.array([f"normal_{i}" for i in range(matrix.shape[1])])
        return matrix, genes, samples
    
    def _load_ppi_network(self) -> np.ndarray:
        """Load PPI network."""
        ppi_path = self.base_path / 'network' / self.ppi_file
        if not ppi_path.exists():
            logger.error("PPI network file not found: %s", ppi_path)
            raise FileNotFoundError(f"PPI network not found: {ppi_path}")
        
        ppi_df = pd.read_csv(ppi_path, sep='\t', header=0)
        return ppi_df.values
    
    def _load_driver_genes(self) -> np.ndarray:
        """Load CGC reference driver genes."""
        path = self.base_path / 'reference' / self.driver_file
        if not path.exists():
            logger.error("Driver genes file not found: %s", path)
            raise FileNotFoundError(f"Driver genes not found: {path}")
        
        return pd.read_csv(path, sep='\t', header=None).iloc[:, 0].values
    
    def _load_non_driver_genes(self) -> np.ndarray:
        """Load non-driver genes."""
        path = self.base_path / 'reference' / self.non_driver_file
        if not path.exists():
            logger.error("Non-driver genes file not found: %s", path)
            raise FileNotFoundError(f"Non-driver genes not found: {path}")
        
        return pd.read_csv(path, sep='\t', header=None).iloc[:, 0].values
    
    def _load_survival_data(self, cancer_type: str) -> pd.DataFrame:
        """Load clinical survival data."""
        path = self.base_path / 'survival' / f"{cancer_type}.survival.tsv"
        if not path.exists():
            logger.warning("Survival data not found: %s", path)
            return pd.DataFrame()
        
        return pd.read_csv(path, sep='\t')
    
    def _log_data_summary(self, data: dict):
        """Log a summary of all loaded data."""
        logger.info("")
        logger.info("─" * 50)
        logger.info("DATA LOADING SUMMARY for %s", data['cancer_type'])
        logger.info("─" * 50)
        
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                log_tensor_info(logger, key, value)
            elif isinstance(value, np.ndarray):
                logger.info("  %s: %d items", key, len(value))
            elif isinstance(value, pd.DataFrame):
                logger.info("  %s: DataFrame shape %s", key, value.shape)
            elif isinstance(value, str):
                logger.info("  %s: %s", key, value)
        
        logger.info("─" * 50)
