"""
Logger utility for the GNN Cancer Driver Gene Identification Pipeline.
Provides dual-output logging (console + execution.log) with medical-grade
traceability for tensor sizes, preprocessing steps, and pipeline execution.
"""

import logging
import sys
import os
import numpy as np
from datetime import datetime


def setup_logger(log_file: str = "./execution.log", log_level: str = "INFO") -> logging.Logger:
    """
    Configure dual-output logger (console + file).
    
    Args:
        log_file: Path to the log file.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("pipeline")
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File always captures everything
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log session start
    logger.info("=" * 72)
    logger.info("Pipeline session started at %s", datetime.now().isoformat())
    logger.info("=" * 72)
    
    return logger


def log_tensor_info(logger: logging.Logger, name: str, data, level: str = "info"):
    """
    Log shape and dtype information for numpy arrays and torch tensors.
    Satisfies medical-grade traceability requirements.
    
    Args:
        logger: Logger instance.
        name: Descriptive name for the tensor/array.
        data: numpy array, torch tensor, or pandas DataFrame.
        level: Log level string.
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    
    if isinstance(data, np.ndarray):
        # Check if data is numeric type
        if np.issubdtype(data.dtype, np.number):
            try:
                log_fn(
                    "[TENSOR] %s | type=ndarray | shape=%s | dtype=%s | "
                    "min=%.4f | max=%.4f | nan_count=%d",
                    name, data.shape, data.dtype,
                    np.nanmin(data), np.nanmax(data), np.isnan(data).sum()
                )
            except (TypeError, ValueError):
                # Fallback if still fails
                log_fn(
                    "[TENSOR] %s | type=ndarray | shape=%s | dtype=%s",
                    name, data.shape, data.dtype
                )
        else:
            # Non-numeric data (strings, objects, etc.)
            try:
                unique_count = len(np.unique(data))
            except TypeError:
                unique_count = "N/A"
            log_fn(
                "[TENSOR] %s | type=ndarray | shape=%s | dtype=%s | unique_count=%s",
                name, data.shape, data.dtype, unique_count
            )
    elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
        # torch.Tensor
        import torch
        if isinstance(data, torch.Tensor):
            log_fn(
                "[TENSOR] %s | type=torch.Tensor | shape=%s | dtype=%s | "
                "device=%s | min=%.4f | max=%.4f",
                name, tuple(data.shape), data.dtype, data.device,
                data.min().item(), data.max().item()
            )
    elif hasattr(data, 'shape'):
        # pandas DataFrame
        log_fn(
            "[TENSOR] %s | type=%s | shape=%s",
            name, type(data).__name__, data.shape
        )
    else:
        log_fn("[DATA] %s | type=%s | len=%s", name, type(data).__name__,
               len(data) if hasattr(data, '__len__') else 'N/A')


def log_stage(logger: logging.Logger, stage_name: str, status: str = "START"):
    """Log pipeline stage transitions with visual separators."""
    if status == "START":
        logger.info("")
        logger.info("━" * 72)
        logger.info("▶ STAGE: %s", stage_name)
        logger.info("━" * 72)
    elif status == "END":
        logger.info("✓ STAGE COMPLETED: %s", stage_name)
        logger.info("─" * 72)
