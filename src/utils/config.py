"""
config.py
Responsible for: Loading and validating YAML config.
"""

import yaml
import os
from typing import Any


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML config file and return as dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get(config: dict, *keys: str, default: Any = None) -> Any:
    """
    Safe nested key access.
    Example: get(cfg, 'training', 'batch_size', default=32)
    """
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val