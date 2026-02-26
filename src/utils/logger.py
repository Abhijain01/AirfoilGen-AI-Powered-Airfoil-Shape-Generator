"""
Logging setup — consistent logging across the project
"""
import os
import sys
import logging
from datetime import datetime


def setup_logger(name, log_dir="logs", level=logging.INFO):
    """
    Set up a logger that writes to both console and file

    Usage:
        logger = setup_logger("training")
        logger.info("Training started")
        logger.warning("Loss is high")
        logger.error("Something went wrong")
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger '{name}' initialized. Log file: {log_file}")

    return logger