"""
Logging utilities for training pipelines.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    output_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with optional file output.
    
    Args:
        name: Logger name
        output_dir: Optional directory to save logs
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s: %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(output_dir / 'training.log')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
