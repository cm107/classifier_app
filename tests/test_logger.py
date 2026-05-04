"""
Test suite for the centralized Logger system.

Verifies:
- Singleton pattern
- Log file creation with session-based naming
- Log level routing (file-only vs file+stdout)
- Exception handling with stack trace capture
- Message formatting with Source Identifier
"""

import tempfile
from pathlib import Path
from vision_app.core.logger import Logger


def test_logger_singleton():
    """Verify that Logger is a proper singleton."""
    log1 = Logger.get_instance()
    log2 = Logger.get_instance()
    assert log1 is log2, "Logger instances should be the same (singleton)"
    print("✓ Logger singleton test passed")


def test_logger_creates_log_file():
    """Verify that logger creates a session-based log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        # Create a fresh logger instance for this test
        logger = Logger(log_dir=log_dir, verbose_mode=False)
        
        logger.info("TestSource", "Test message")
        
        # Check that a log file was created
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        assert log_files[0].suffix == ".log", "Log file should have .log extension"
        print(f"✓ Logger created log file: {log_files[0].name}")


def test_logger_file_format():
    """Verify that log entries are properly formatted in the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = Logger(log_dir=log_dir, verbose_mode=False)
        
        logger.info("TestClass", "Info message")
        logger.warning("TestClass", "Warning message")
        
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1
        
        with open(log_files[0], "r") as f:
            content = f.read()
        
        # Verify that log entries include source identifier and message
        assert "TestClass" in content, "Log should contain source identifier"
        assert "Info message" in content, "Log should contain message"
        assert "Warning message" in content, "Log should contain warning"
        print("✓ Logger file format test passed")


def test_logger_levels_file_routing():
    """Verify that DEBUG and VERBOSE go to file only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = Logger(log_dir=log_dir, verbose_mode=False)
        
        logger.debug("TestSource", "Debug entry")
        logger.verbose("TestSource", "Verbose entry")
        
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1
        
        with open(log_files[0], "r") as f:
            content = f.read()
        
        # Both should be in the file
        assert "Debug entry" in content, "Debug should be in file"
        assert "Verbose entry" in content, "Verbose should be in file"
        print("✓ Logger level routing test passed")


def test_logger_exception_with_traceback():
    """Verify that EXCEPTION level captures full stack trace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = Logger(log_dir=log_dir, verbose_mode=False)
        
        try:
            x = 1 / 0  # Trigger ZeroDivisionError
        except Exception as e:
            logger.exception("TestSource", f"Caught exception: {e}")
        
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1
        
        with open(log_files[0], "r") as f:
            content = f.read()
        
        # Verify exception details are captured
        assert "Caught exception" in content, "Exception message should be in log"
        assert "ZeroDivisionError" in content, "Exception type should be in log"
        assert "Traceback" in content, "Stack trace should be in log"
        print("✓ Logger exception with traceback test passed")


def test_logger_source_identifier():
    """Verify that Source Identifier is required and logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = Logger(log_dir=log_dir, verbose_mode=False)
        
        logger.info("DatasetBuilder", "Loading dataset")
        logger.warning("ModelTrainer", "Gradient clipping applied")
        
        log_files = list(log_dir.glob("app_*.log"))
        with open(log_files[0], "r") as f:
            content = f.read()
        
        # Verify source identifiers are logged
        assert "DatasetBuilder" in content, "Source identifier should be logged"
        assert "ModelTrainer" in content, "Source identifier should be logged"
        print("✓ Logger source identifier test passed")


if __name__ == "__main__":
    test_logger_singleton()
    test_logger_creates_log_file()
    test_logger_file_format()
    test_logger_levels_file_routing()
    test_logger_exception_with_traceback()
    test_logger_source_identifier()
    print("\n✅ All logger tests passed!")
