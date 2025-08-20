# sequence_encoder/logging_config.py

import logging
import sys
import os
from bioclust.core import config

def setup_logger(name: str = "sequence_encoder", level: int = logging.INFO, enable: bool = True) -> logging.Logger:
    """
    Set up and return a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    enable : bool
        If False, disables all logging output.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)

    if not enable:
        logger.disabled = True
        return logger

    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cfg = config.get_config()
    base_log_dir = cfg.cache_paths.logs()
    os.makedirs(base_log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{base_log_dir}/sequence_encoder.log")
    
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger