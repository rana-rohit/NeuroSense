"""
logger.py
Responsible for: Consistent logging across all modules.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/logs",
               level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to console + file.

    Args:
        name   : module name (e.g. 'trainer', 'preprocessor')
        log_dir: directory to save log files
        level  : logging level

    Returns:
        logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger    = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger   # avoid duplicate handlers on re-import

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh  = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger