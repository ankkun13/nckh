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
import requests
import zipfile
import gzip
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
    Supports automatic downloading from GitHub storage if files are missing.
    """
    
    def __init__(self, config: dict):
        self.base_path = Path(config['data']['base_path'])
        self.ppi_file = config['data']['ppi_network']
        self.driver_file = config['data']['driver_genes']
        self.non_driver_file = config['data']['non_driver_genes']
        self.storage_url = config['data'].get('github_storage_url')
    
    def _download_if_not_exists(self, relative_path: str):
        """
        Download a file from GitHub storage if it's not present locally.
        Supports automatic extraction of .gz and .zip files.
        
        Args:
            relative_path: Path relative to base_path (e.g., 'network/string.txt').
        """
        local_path = self.base_path / relative_path
        if local_path.exists():
            return
            
        if not self.storage_url:
            logger.warning("File missing and no github_storage_url provided: %s", relative_path)
            return

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        base_url = self.storage_url.rstrip('/')
        
        # Try different extensions (plain, .gz, .zip)
        possible_extensions = ['', '.gz', '.zip']
        
        download_success = False
        for ext in possible_extensions:
            url = f"{base_url}/{relative_path}{ext}"
            temp_download_path = local_path.with_suffix(local_path.suffix + ext)
            
            try:
                logger.info("Attempting to download from: %s", url)
                response = requests.get(url, timeout=30, stream=True)
                
                if response.status_code == 200:
                    with open(temp_download_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Handle extraction
                    if ext == '.gz':
                        logger.info("Decompressing .gz file...")
                        with gzip.open(temp_download_path, 'rb') as f_in:
                            with open(local_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        temp_download_path.unlink() # Remove .gz after extraction
                    elif ext == '.zip':
                        logger.info("Extracting .zip file...")
                        with zipfile.ZipFile(temp_download_path, 'r') as zip_ref:
                            zip_ref.extractall(local_path.parent)
                        temp_download_path.unlink() # Remove .zip
                    
                    logger.info("Successfully obtained: %s", relative_path)
                    download_success = True
                    break
            except Exception as e:
                logger.debug("Failed attempt for %s: %s", url, str(e))
                continue
                
        if not download_success:
            logger.error("Failed to download %s or its compressed versions from GitHub storage.", relative_path)

    def load(self, cancer_type: str) -> dict:
        """
        Load all data for a given cancer type.
        
        Args:
            cancer_type: TCGA cancer type string (e.g., 'TCGA-LUAD')
        
        Returns:
            Dictionary containing all omics data, network, and labels.
        """
        logger.info("Loading data for cancer type: %s", cancer_type)
        logger.info("Data base path: %s", self.base_path.resolve())
        
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
        """
        Load omics matrix (EXP or MET) from TCGA_UCSC format.
        
        Directory structure:
            data_dir/cancer_type/cancer_type_{data_type}_data.txt
            data_dir/cancer_type/cancer_type_gene.txt
            data_dir/cancer_type/cancer_type_sample.txt
        """
        rel_data = f"TCGA_UCSC_EXP/{cancer_type}/{cancer_type}_{data_type}_data.txt" if data_dir == 'TCGA_UCSC_EXP' else f"TCGA_UCSC_MET/{cancer_type}/{cancer_type}_{data_type}_data.txt"
        rel_gene = f"TCGA_UCSC_EXP/{cancer_type}/{cancer_type_gene}.txt" if data_dir == 'TCGA_UCSC_EXP' else f"TCGA_UCSC_MET/{cancer_type}/{cancer_type_gene}.txt"
        
        # Simplified for now, the relative path logic needs to be precise
        self._download_if_not_exists(f"{data_dir}/{cancer_type}/{cancer_type}_{data_type}_data.txt")
        self._download_if_not_exists(f"{data_dir}/{cancer_type}/{cancer_type}_gene.txt")
        self._download_if_not_exists(f"{data_dir}/{cancer_type}/{cancer_type}_sample.txt")
        
        dir_path = self.base_path / data_dir / cancer_type
        
        data_file = dir_path / f"{cancer_type}_{data_type}_data.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Data file not found: %s", data_file)
            return np.array([]), np.array([]), np.array([])
        
        matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        
        logger.info("Loaded %s %s: matrix=%s, genes=%d, samples=%d",
                     cancer_type, data_type, matrix.shape, len(genes), len(samples))
        return matrix, genes, samples
    
    def _load_cnv_matrix(self, cancer_type: str) -> tuple:
        """Load CNV data from cnv_matrix directory."""
        self._download_if_not_exists(f"TCGA_UCSC_CNV/cnv_matrix/{cancer_type}_cnv_matrix.txt")
        self._download_if_not_exists(f"TCGA_UCSC_CNV/cnv_matrix/{cancer_type}_cnv_gene.txt")
        self._download_if_not_exists(f"TCGA_UCSC_CNV/cnv_matrix/{cancer_type}_cnv_sample.txt")
        
        dir_path = self.base_path / 'TCGA_UCSC_CNV' / 'cnv_matrix'
        
        data_file = dir_path / f"{cancer_type}_cnv_matrix.txt"
        gene_file = dir_path / f"{cancer_type}_cnv_gene.txt"
        sample_file = dir_path / f"{cancer_type}_cnv_sample.txt"
        
        if not data_file.exists():
            logger.warning("CNV data file not found: %s", data_file)
            return np.array([]), np.array([]), np.array([])
        
        matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        
        logger.info("Loaded %s CNV: matrix=%s, genes=%d, samples=%d",
                     cancer_type, matrix.shape, len(genes), len(samples))
        return matrix, genes, samples
    
    def _load_snv_matrix(self, cancer_type: str) -> tuple:
        """Load SNV data from snv_matrix directory."""
        self._download_if_not_exists(f"TCGA_hg38_SNV/snv_matrix/{cancer_type}_snv_matrix.txt")
        self._download_if_not_exists(f"TCGA_hg38_SNV/snv_matrix/{cancer_type}_snv_gene.txt")
        self._download_if_not_exists(f"TCGA_hg38_SNV/snv_matrix/{cancer_type}_snv_sample.txt")
        
        dir_path = self.base_path / 'TCGA_hg38_SNV' / 'snv_matrix'
        
        data_file = dir_path / f"{cancer_type}_snv_matrix.txt"
        gene_file = dir_path / f"{cancer_type}_snv_gene.txt"
        sample_file = dir_path / f"{cancer_type}_snv_sample.txt"
        
        if not data_file.exists():
            logger.warning("SNV data file not found: %s", data_file)
            return np.array([]), np.array([]), np.array([])
        
        matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        
        logger.info("Loaded %s SNV: matrix=%s, genes=%d, samples=%d",
                     cancer_type, matrix.shape, len(genes), len(samples))
        return matrix, genes, samples
    
    def _load_mutation_matrix(self, cancer_type: str) -> tuple:
        """Load mutation data from TCGA_Mutation directory."""
        self._download_if_not_exists(f"TCGA_Mutation/{cancer_type}/{cancer_type}_mutation.txt")
        self._download_if_not_exists(f"TCGA_Mutation/{cancer_type}/{cancer_type}_gene.txt")
        self._download_if_not_exists(f"TCGA_Mutation/{cancer_type}/{cancer_type}_sample.txt")
        
        dir_path = self.base_path / 'TCGA_Mutation' / cancer_type
        
        data_file = dir_path / f"{cancer_type}_mutation.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Mutation data file not found: %s", data_file)
            return np.array([]), np.array([]), np.array([])
        
        matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        
        logger.info("Loaded %s Mutation: matrix=%s, genes=%d, samples=%d",
                     cancer_type, matrix.shape, len(genes), len(samples))
        return matrix, genes, samples
    
    def _load_normal_data(self, cancer_type: str, data_type: str) -> tuple:
        """Load normal tissue data (EXP or MET)."""
        self._download_if_not_exists(f"TCGA_UCSC_normal/{cancer_type}/{cancer_type}_{data_type}_normal.txt")
        self._download_if_not_exists(f"TCGA_UCSC_normal/{cancer_type}/{cancer_type}_gene.txt")
        self._download_if_not_exists(f"TCGA_UCSC_normal/{cancer_type}/{cancer_type}_sample.txt")
        
        dir_path = self.base_path / 'TCGA_UCSC_normal' / cancer_type
        
        data_file = dir_path / f"{cancer_type}_{data_type}_normal.txt"
        gene_file = dir_path / f"{cancer_type}_gene.txt"
        sample_file = dir_path / f"{cancer_type}_sample.txt"
        
        if not data_file.exists():
            logger.warning("Normal %s data not found: %s", data_type, data_file)
            return np.array([]), np.array([]), np.array([])
        
        matrix = pd.read_csv(data_file, sep='\t', header=None).values.astype(np.float32)
        genes = pd.read_csv(gene_file, sep='\t', header=None).iloc[:, 0].values
        
        if sample_file.exists():
            samples = pd.read_csv(sample_file, sep='\t', header=None).iloc[:, 0].values
        else:
            samples = np.array([f"normal_{i}" for i in range(matrix.shape[1])])
        
        logger.info("Loaded %s Normal %s: matrix=%s, genes=%d, samples=%d",
                     cancer_type, data_type, matrix.shape, len(genes), len(samples))
        return matrix, genes, samples
    
    def _load_ppi_network(self) -> np.ndarray:
        """Load PPI network as structured array with columns: protein_1, protein_2, score."""
        self._download_if_not_exists(f"network/{self.ppi_file}")
        ppi_path = self.base_path / 'network' / self.ppi_file
        
        if not ppi_path.exists():
            logger.error("PPI network file not found: %s", ppi_path)
            raise FileNotFoundError(f"PPI network not found: {ppi_path}")
        
        ppi_df = pd.read_csv(ppi_path, sep='\t', header=0)
        ppi = ppi_df.values
        
        logger.info("Loaded PPI network: %d interactions", len(ppi))
        return ppi
    
    def _load_driver_genes(self) -> np.ndarray:
        """Load CGC reference driver genes."""
        self._download_if_not_exists(f"reference/{self.driver_file}")
        path = self.base_path / 'reference' / self.driver_file
        
        if not path.exists():
            logger.error("Driver genes file not found: %s", path)
            raise FileNotFoundError(f"Driver genes not found: {path}")
        
        genes = pd.read_csv(path, sep='\t', header=None).iloc[:, 0].values
        logger.info("Loaded %d reference driver genes (CGC)", len(genes))
        return genes
    
    def _load_non_driver_genes(self) -> np.ndarray:
        """Load non-driver genes."""
        self._download_if_not_exists(f"reference/{self.non_driver_file}")
        path = self.base_path / 'reference' / self.non_driver_file
        
        if not path.exists():
            logger.error("Non-driver genes file not found: %s", path)
            raise FileNotFoundError(f"Non-driver genes not found: {path}")
        
        genes = pd.read_csv(path, sep='\t', header=None).iloc[:, 0].values
        logger.info("Loaded %d non-driver genes", len(genes))
        return genes
    
    def _load_survival_data(self, cancer_type: str) -> pd.DataFrame:
        """Load clinical survival data."""
        self._download_if_not_exists(f"survival/{cancer_type}.survival.tsv")
        path = self.base_path / 'survival' / f"{cancer_type}.survival.tsv"
        
        if not path.exists():
            logger.warning("Survival data not found: %s", path)
            return pd.DataFrame()
        
        survival = pd.read_csv(path, sep='\t')
        logger.info("Loaded survival data: %d patients", len(survival))
        return survival
    
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
