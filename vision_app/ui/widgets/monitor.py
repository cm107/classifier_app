"""
monitor.py — Training Monitor UI.

Classes:
    LiveGraph              — pyqtgraph-based real-time loss / accuracy plot.
    TrainingMonitorWidget  — Progress bars, metric cards, Start/Abort controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vision_app.core.trainer import TrainingMetrics
from vision_app.worker.train_worker import TrainWorker
from vision_app.core.logger import log


# ---------------------------------------------------------------------------
# Pyqtgraph global style
# ---------------------------------------------------------------------------
pg.setConfigOptions(antialias=True, background="#181825", foreground="#cdd6f4")


# ---------------------------------------------------------------------------
# LiveGraph
# ---------------------------------------------------------------------------
class LiveGraph(QWidget):
    """
    Real-time dual-axis plot: training loss (left axis) and validation
    accuracy (right axis) over epochs.

    Performance note: pyqtgraph uses OpenGL-accelerated rendering and can
    handle thousands of points per second without UI lag.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._epochs: list[int] = []
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._val_accs: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget(title="Training Progress")
        self._plot_widget.setLabel("left", "Loss")
        self._plot_widget.setLabel("bottom", "Epoch")
        self._plot_widget.addLegend(offset=(10, 10))
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self._plot_widget)

        # Curves
        self._train_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#cba6f7", width=2), name="Train Loss"
        )
        self._val_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#89b4fa", width=2), name="Val Loss"
        )

        # Second y-axis for accuracy
        self._acc_view = pg.ViewBox()
        self._plot_widget.scene().addItem(self._acc_view)
        self._plot_widget.getAxis("right").linkToView(self._acc_view)
        self._plot_widget.getAxis("right").setLabel("Accuracy (%)")
        self._plot_widget.showAxis("right")
        self._acc_view.setXLink(self._plot_widget)

        self._acc_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color="#a6e3a1", width=2), name="Val Acc"
        )
        self._acc_view.addItem(self._acc_curve)

        # Sync the right axis when the main view changes
        self._plot_widget.getViewBox().sigResized.connect(self._sync_axes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plot(self, metrics: TrainingMetrics):
        """Append one epoch's data and refresh all curves."""
        self._epochs.append(metrics.epoch)
        self._train_losses.append(metrics.train_loss)
        self._val_losses.append(metrics.val_loss)
        self._val_accs.append(metrics.val_accuracy)

        self._train_curve.setData(self._epochs, self._train_losses)
        self._val_curve.setData(self._epochs, self._val_losses)
        self._acc_curve.setData(self._epochs, self._val_accs)
        self._acc_view.autoRange()

    def reset(self):
        """Clear all curves for a new training run."""
        self._epochs.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])
        self._acc_curve.setData([], [])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sync_axes(self):
        self._acc_view.setGeometry(self._plot_widget.getViewBox().sceneBoundingRect())


# ---------------------------------------------------------------------------
# _MetricCard — small display widget for a single scalar value
# ---------------------------------------------------------------------------
class _MetricCard(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._lbl = QLabel(label)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")

        self._val = QLabel("—")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val.setStyleSheet("font-size: 20px; font-weight: bold; color: #cba6f7;")

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)
        self.setStyleSheet("background: #313244; border-radius: 8px;")

    def set_value(self, text: str):
        self._val.setText(text)


# ---------------------------------------------------------------------------
# TrainingMonitorWidget
# ---------------------------------------------------------------------------
class TrainingMonitorWidget(QWidget):
    """
    The "Command Center" for training.

    Layout:
        ┌─ hyperparameter panel (left) ─┬─ graph (right) ─┐
        │  HyperParameterWidget         │   LiveGraph      │
        ├───────────────────────────────┴──────────────────┤
        │  metric cards: Loss | Val Loss | Accuracy | LR   │
        ├──────────────────────────────────────────────────┤
        │  epoch progress bar                               │
        │  [Start Training]  [Abort]   status label        │
        └──────────────────────────────────────────────────┘

    The hyperparameter panel is injected via set_hyperparameter_widget()
    so that MainWindow can position it however it likes.
    """

    training_started = Signal()
    training_finished = Signal(bool)   # True = success, False = aborted/error

    def __init__(self, storage_root: Path, parent=None):
        super().__init__(parent)
        self._storage_root = Path(storage_root)
        self._worker: Optional[TrainWorker] = None
        self._thread: Optional[QThread] = None
        self._hyper_widget = None   # set by set_hyperparameter_widget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Training Monitor")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # ── Top: graph ──
        self._graph = LiveGraph()
        self._graph.setMinimumHeight(260)
        layout.addWidget(self._graph)

        # ── Metric cards ──
        cards_row = QHBoxLayout()
        self._card_train_loss = _MetricCard("Train Loss")
        self._card_val_loss   = _MetricCard("Val Loss")
        self._card_accuracy   = _MetricCard("Accuracy")
        self._card_lr         = _MetricCard("Learning Rate")
        for card in (self._card_train_loss, self._card_val_loss,
                     self._card_accuracy, self._card_lr):
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # ── Epoch progress bar ──
        prog_group = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog_group)

        self._epoch_label = QLabel("Epoch — / —")
        self._epoch_bar = QProgressBar()
        self._epoch_bar.setRange(0, 1)
        self._epoch_bar.setValue(0)
        prog_layout.addWidget(self._epoch_label)
        prog_layout.addWidget(self._epoch_bar)
        layout.addWidget(prog_group)

        # ── Controls ──
        ctrl_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setObjectName("primary_button")
        self._start_btn.setFixedHeight(36)

        self._abort_btn = QPushButton("■  Abort")
        self._abort_btn.setObjectName("danger_button")
        self._abort_btn.setFixedHeight(36)
        self._abort_btn.setEnabled(False)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet("color: #a6adc8;")

        ctrl_row.addWidget(self._start_btn)
        ctrl_row.addWidget(self._abort_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._status_label)
        layout.addLayout(ctrl_row)

        # ── Wire buttons ──
        self._start_btn.clicked.connect(self._on_start)
        self._abort_btn.clicked.connect(self._on_abort)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_hyperparameter_widget(self, widget):
        """Register the HyperParameterWidget so we can read its config."""
        self._hyper_widget = widget

    # ------------------------------------------------------------------
    # Slot: progress_updated
    # ------------------------------------------------------------------

    def _on_progress(self, metrics: TrainingMetrics):
        self._graph.update_plot(metrics)
        self._card_train_loss.set_value(f"{metrics.train_loss:.4f}")
        self._card_val_loss.set_value(f"{metrics.val_loss:.4f}")
        self._card_accuracy.set_value(f"{metrics.val_accuracy:.1f}%")
        self._card_lr.set_value(f"{metrics.learning_rate:.2e}")

        self._epoch_bar.setValue(metrics.epoch)
        self._epoch_label.setText(
            f"Epoch {metrics.epoch} / {metrics.total_epochs}"
        )

    # ------------------------------------------------------------------
    # Start / abort
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._hyper_widget is None:
            self._status_label.setText("Error: no hyperparameter widget connected.")
            return

        cfg = self._hyper_widget.get_train_config()
        dataset_path = cfg.get("dataset_path")
        if not dataset_path or not Path(dataset_path).exists():
            self._status_label.setText("Error: select a valid dataset first.")
            return

        self._graph.reset()
        self._epoch_bar.setRange(0, cfg["epochs"])
        self._epoch_bar.setValue(0)
        self._status_label.setText("Initialising…")
        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)

        # Build worker + thread
        self._worker = TrainWorker(cfg)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._worker.signals.progress_updated.connect(self._on_progress)
        self._worker.signals.status_changed.connect(self._status_label.setText)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.error_occurred.connect(
            lambda msg: self._status_label.setText(f"Error: {msg}")
        )
        self._thread.started.connect(self._worker.run)
        # Clean up thread after worker finishes
        self._worker.signals.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()
        log.info("TrainingMonitorWidget", f"Training started: dataset={cfg.get('dataset_path', 'unknown')}, epochs={cfg['epochs']}, phase={cfg.get('phase', 'supervised')}")
        self.training_started.emit()

    def _on_abort(self):
        if self._worker:
            self._worker.lifecycle.request_abort()
        log.info("TrainingMonitorWidget", "Training abort requested by user")
        self._status_label.setText("Aborting…")
        self._abort_btn.setEnabled(False)

    def _on_finished(self, success: bool):
        self._start_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        msg = "Training complete." if success else "Training aborted."
        log.info("TrainingMonitorWidget", msg)
        self._status_label.setText(msg)
        self._worker = None
        self.training_finished.emit(success)
