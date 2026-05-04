"""
main.py — Application entry point.

Responsibilities:
    - Resolve the storage/ directory relative to this file (portable).
    - Enable High-DPI scaling.
    - Create storage subdirectories on first run.
    - Instantiate QApplication and MainWindow, then enter the event loop.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vision_app.core.logger import initialize_logger
from vision_app.core.utils import ConfigLoader
from vision_app.ui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Storage root — always resolved relative to main.py so the app is portable.
# ---------------------------------------------------------------------------
STORAGE_ROOT = Path(__file__).parent / "storage"


def _ensure_storage_dirs():
    for subdir in ("datasets", "models", "logs", "trash"):
        (STORAGE_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def main():
    _ensure_storage_dirs()

    # Initialize logger with storage/logs directory
    log_dir = STORAGE_ROOT / "logs"
    config = ConfigLoader()
    verbose_mode = config.get("verbose_logging", False)
    initialize_logger(log_dir=log_dir, verbose_mode=verbose_mode)
    from vision_app.core.logger import log
    log.info("main", "Vision App started")

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Vision App")
    app.setOrganizationName("VisionApp")

    window = MainWindow(storage_root=STORAGE_ROOT, app=app)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
