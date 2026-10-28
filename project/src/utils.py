"""Utility functions: seeding, hardware detection, logging, and configuration loading."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, torch, and torch.cuda for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect available accelerator (CUDA, MPS, or CPU) with advisory warning on CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Hardware detected: CUDA GPU (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Hardware detected: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logging.warning(
            "Hardware detected: CPU only! Training google/muril-base-cased on CPU will be very slow. "
            "For full dataset training, Google Colab (T4/A100) or Kaggle GPU is strongly recommended. "
            "Use the --debug flag for smoke-testing."
        )
    return device


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Configure structured logging to console and optional log file.

    Args:
        log_file: Path to destination log file.

    Returns:
        Root logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-entry
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def find_project_root() -> Path:
    """Determine the project root directory whether invoked from root or project subdir."""
    current = Path.cwd()
    if (current / "config.yaml").exists() and (current / "src").exists():
        return current
    if (current / "project" / "config.yaml").exists():
        return current / "project"
    # Fallback to directory containing this file's parent
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path relative to project root or workspace, handling ../ gracefully."""
    p = Path(path_str)
    if p.is_absolute():
        return p

    base = base_dir or find_project_root()
    direct = (base / p).resolve()
    if direct.exists():
        return direct

    # Try parent directory if '../' was already relative to root
    alt = (base.parent / p).resolve()
    if alt.exists():
        return alt

    # If it starts with ../ and base is project, try from base
    clean_p = Path(str(p).lstrip("../").lstrip("..\\"))
    workspace_alt = (base.parent / clean_p).resolve()
    if workspace_alt.exists():
        return workspace_alt

    return direct


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration YAML file.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Config dictionary.
    """
    if config_path is None:
        proj_root = find_project_root()
        config_path = proj_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg
