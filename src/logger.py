"""
logger.py
---------
Centralized logging configuration for the Water Quality LSTM project.
Import `get_logger(__name__)` in any module to get a consistent logger.

All log files are stored in the `logs/` directory at the project root.
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_filename = os.path.join(LOG_DIR, f"wq_lstm_{datetime.now().strftime('%Y%m%d')}.log")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a daily log file.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (INFO and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --- File handler (DEBUG and above) ---
    file_handler = logging.FileHandler(_log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
