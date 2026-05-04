"""
Centralized logging system for the Vision App.

Provides a singleton logger instance accessible throughout the application
to maintain a consistent "Source of Truth" for runtime behavior tracking.

Usage:
    from vision_app.core.logger import log
    log.info("ClassName", "Event description")
    log.exception("ClassName", "Error occurred: {error_msg}")
"""

import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """
    Centralized singleton logger with support for multiple log levels and routing.
    
    Log levels:
    - DEBUG: Temporary debugging entries (file only, removed after issue resolved)
    - VERBOSE: Permanent granular details (file only, gated by config flags)
    - INFO: Core application events (file + stdout)
    - WARNING: User attention needed, non-fatal (file + stdout)
    - ERROR: Fatal logic failures leading to termination (file + stdout)
    - EXCEPTION: Unexpected runtime failures with full stack trace (file + stdout)
    """
    
    _instance: Optional["Logger"] = None
    
    def __init__(self, log_dir: Optional[Path] = None, verbose_mode: bool = False):
        """
        Initialize the logger with optional log directory.
        
        Args:
            log_dir: Directory to store log files. If None, logs are not written to disk.
            verbose_mode: If True, VERBOSE level logs are written to stdout.
        """
        self.log_dir = log_dir
        self.verbose_mode = verbose_mode
        self.file_handler: Optional[logging.FileHandler] = None
        self.session_start = datetime.now()
        
        # Create file handler if log_dir is provided
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._setup_file_handler()
    
    def _setup_file_handler(self) -> None:
        """Set up file handler with session-based filename."""
        if not self.log_dir:
            return
        
        # Create directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Format: app_YYYY-MM-DD_HH-mm-ss.log
        timestamp = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = self.log_dir / f"app_{timestamp}.log"
        
        # Create file handler with append mode
        self.file_handler = logging.FileHandler(str(log_file), mode="a")
        self.file_handler.setLevel(logging.DEBUG)
        
        # Set formatter: timestamp, source, level, message
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_handler.setFormatter(formatter)
    
    @classmethod
    def get_instance(cls, log_dir: Optional[Path] = None, verbose_mode: bool = False) -> "Logger":
        """
        Get or create the singleton logger instance.
        
        Args:
            log_dir: Directory to store log files (used on first instantiation or to update).
            verbose_mode: If True, VERBOSE logs are written to stdout.
            
        Returns:
            The singleton Logger instance.
        """
        if cls._instance is None:
            cls._instance = cls(log_dir=log_dir, verbose_mode=verbose_mode)
        else:
            # Update log_dir if it was None and now provided
            if log_dir is not None and cls._instance.log_dir is None:
                cls._instance.log_dir = log_dir
                cls._instance._setup_file_handler()
            # Update verbose_mode
            cls._instance.verbose_mode = verbose_mode
        return cls._instance
    
    def _log_to_file(self, level: str, source: str, message: str) -> None:
        """Write log entry to file."""
        if not self.file_handler:
            return
        
        logger = logging.getLogger(source)
        logger.addHandler(self.file_handler)
        
        level_map = {
            "DEBUG": logging.DEBUG,
            "VERBOSE": logging.DEBUG,  # VERBOSE uses DEBUG level in file
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "EXCEPTION": logging.ERROR,
        }
        
        logger.setLevel(level_map.get(level, logging.INFO))
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
    
    def debug(self, source: str, message: str) -> None:
        """Log a DEBUG entry (file only). Temporary debugging—remove after issue resolved."""
        self._log_to_file("DEBUG", source, message)
    
    def verbose(self, source: str, message: str) -> None:
        """Log a VERBOSE entry (file only, gated by verbose_mode for stdout)."""
        self._log_to_file("VERBOSE", source, message)
        if self.verbose_mode:
            print(f"[VERBOSE] {source}: {message}")
    
    def info(self, source: str, message: str) -> None:
        """Log an INFO entry (file + stdout). Core application events."""
        self._log_to_file("INFO", source, message)
        print(f"[INFO] {source}: {message}")
    
    def warning(self, source: str, message: str) -> None:
        """Log a WARNING entry (file + stdout). User attention needed, non-fatal."""
        self._log_to_file("WARNING", source, message)
        print(f"[WARNING] {source}: {message}")
    
    def error(self, source: str, message: str) -> None:
        """Log an ERROR entry (file + stdout). Fatal logic failures."""
        self._log_to_file("ERROR", source, message)
        print(f"[ERROR] {source}: {message}")
    
    def exception(self, source: str, message: str) -> None:
        """
        Log an EXCEPTION entry (file + stdout). Unexpected runtime failures with full stack trace.
        
        Args:
            source: The class/function name where exception occurred.
            message: Description of the exception.
        """
        tb_str = traceback.format_exc()
        full_message = f"{message}\n{tb_str}"
        self._log_to_file("EXCEPTION", source, full_message)
        print(f"[EXCEPTION] {source}: {message}")
        print(tb_str)


def initialize_logger(log_dir: Optional[Path] = None, verbose_mode: bool = False):
    """Initialize the logger with log directory and verbose mode."""
    Logger.get_instance(log_dir=log_dir, verbose_mode=verbose_mode)


# Global singleton instance
log: Logger = Logger.get_instance()
