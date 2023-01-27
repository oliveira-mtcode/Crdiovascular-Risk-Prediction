"""
Logging utilities for cardiovascular risk prediction pipeline.

This module provides comprehensive logging setup and management
for the cardiovascular risk prediction pipeline.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys


def setup_logging(level: str = 'INFO', 
                 verbose: bool = False,
                 log_file: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        verbose: Enable verbose logging
        log_file: Path to log file
        config: Configuration dictionary containing logging settings
    """
    # Get logging configuration
    if config:
        log_config = config.get('logging', {})
        level = log_config.get('level', level)
        log_file = log_config.get('file', log_file)
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        max_file_size = log_config.get('max_file_size', '10MB')
        backup_count = log_config.get('backup_count', 5)
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        max_file_size = '10MB'
        backup_count = 5
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        if max_file_size.endswith('MB'):
            max_bytes = int(max_file_size[:-2]) * 1024 * 1024
        elif max_file_size.endswith('KB'):
            max_bytes = int(max_file_size[:-2]) * 1024
        else:
            max_bytes = int(max_file_size)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for verbose mode
    if verbose:
        logging.getLogger('cv_risk_pipeline').setLevel(logging.DEBUG)
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('plotly').setLevel(logging.WARNING)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed. Level: {level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        
        return result
    
    return wrapper
