"""
main_window.py — Application shell.

Architecture (submodule pattern):
    MainWindow
        ├── NavigationController  — sidebar QListWidget + QStackedWidget
        ├── StatusBarManager      — GPU label + temporary notification area
        └── ThemeEngine           — loads and applies style.qss
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# NavigationController
# ---------------------------------------------------------------------------
class NavigationController:
    """
    Manages the sidebar QListWidget and the central QStackedWidget.

    Pages are registered with add_page() and selected by clicking the sidebar.
    """

    _NAV_ITEMS = [
        ("🗂  Dataset",   "Manage and prepare your image datasets"),
        ("⚙  Train",      "Configure hyperparameters and start training"),
        ("📦  Models",    "Browse, load, and export trained models"),
        ("🎥  Inference", "Live webcam / video / image inference"),
    ]

    def __init__(self, parent: "MainWindow"):
        self._parent = parent

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(180)
        self.sidebar.setSpacing(2)

        for label, tooltip in self._NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setToolTip(tooltip)
            item.setSizeHint(item.sizeHint().__class__(180, 44))
            self.sidebar.addItem(item)

        # Central stacked widget
        self.stack = QStackedWidget()

        # Connect selection → page switch
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        # Default to first page
        self.sidebar.setCurrentRow(0)

    def add_page(self, widget: QWidget):
        """Append a page to the stack. Call in the same order as _NAV_ITEMS."""
        self.stack.addWidget(widget)

    def go_to(self, index: int):
        self.sidebar.setCurrentRow(index)


# ---------------------------------------------------------------------------
# StatusBarManager
# ---------------------------------------------------------------------------
class StatusBarManager:
    """
    Manages the main window's QStatusBar.

    Left  : temporary notification messages (auto-clear after 4 s)
    Right : permanent GPU / CPU label
    """

    _CLEAR_MS = 4_000

    def __init__(self, parent: "MainWindow"):
        self._parent = parent
        status_bar: QStatusBar = parent.statusBar()

        # Permanent GPU info label (right side)
        self._gpu_label = QLabel(self._gpu_info())
        self._gpu_label.setObjectName("gpu_label")
        status_bar.addPermanentWidget(self._gpu_label)

        # Auto-clear timer
        self._clear_timer = QTimer(parent)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(status_bar.clearMessage)

    def show_message(self, text: str, duration_ms: int = _CLEAR_MS):
        """Display a temporary status message."""
        self._parent.statusBar().showMessage(text, duration_ms)

    def update_gpu_label(self):
        self._gpu_label.setText(self._gpu_info())

    @staticmethod
    def _gpu_info() -> str:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU: {name}"
        return "GPU: CPU only"


# ---------------------------------------------------------------------------
# ThemeEngine
# ---------------------------------------------------------------------------
class ThemeEngine:
    """Loads style.qss and applies it to the QApplication."""

    _QSS_PATH = Path(__file__).parent / "resources" / "style.qss"

    def __init__(self, parent: "MainWindow"):
        self._parent = parent

    def apply(self, app: "QApplication"):
        """Read style.qss and apply to the running QApplication."""
        if self._QSS_PATH.exists():
            app.setStyleSheet(self._QSS_PATH.read_text(encoding="utf-8"))
        else:
            self._parent.status_bar_manager.show_message(
                f"Warning: style.qss not found at {self._QSS_PATH}"
            )


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    Application shell. Owns navigation, status bar, and theme.

    Pages (in sidebar order):
        0 — DatasetManagerWidget   (Milestone 4)
        1 — TrainingMonitorWidget  (Milestone 5, stub for now)
        2 — ModelManagerWidget     (Milestone 6, stub for now)
        3 — InferenceViewWidget    (Milestone 6, stub for now)

    Args:
        storage_root : Path to the top-level storage/ directory.
        app          : The running QApplication (needed for theme application).
    """

    def __init__(self, storage_root: Path, app: "QApplication"):
        super().__init__()
        self.storage_root = Path(storage_root)

        self.setWindowTitle("Vision App")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # --- Submodules ---
        self.nav = NavigationController(self)
        self.status_bar_manager = StatusBarManager(self)
        self.theme_engine = ThemeEngine(self)
        self.theme_engine.apply(app)

        # --- Build pages ---
        self._build_pages()

        # --- Central layout ---
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.nav.sidebar)
        layout.addWidget(self.nav.stack)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Page construction
    # ------------------------------------------------------------------

    def _build_pages(self):
        from vision_app.ui.widgets.dataset_tab import DatasetManagerWidget

        # Page 0 — Dataset
        self.dataset_page = DatasetManagerWidget(self.storage_root, self)
        self.nav.add_page(self.dataset_page)

        # Pages 1-3 — stubs until Milestones 5-6
        for label in ("Train", "Models", "Inference"):
            self.nav.add_page(self._stub_page(label))

    @staticmethod
    def _stub_page(label: str) -> QWidget:
        page = QWidget()
        lyt = QHBoxLayout(page)
        lbl = QLabel(f"{label} — coming soon")
        lbl.setObjectName("section_title")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lyt.addWidget(lbl)
        return page

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent):
        """Graceful shutdown — abort any running workers (Milestone 6)."""
        event.accept()
