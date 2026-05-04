"""
live_view.py — Live inference UI.

Submodule pattern (all owned by InferenceViewWidget):
    SourceController  — Input source selector + Play/Stop toggle.
    CameraViewport    — Anti-flicker QLabel frame display with scaling.
    OverlayPainter    — Transparent top overlay: top-k predictions + bars.
    RecordingManager  — Optional frame recording to a video file.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vision_app.worker.stream_worker import StreamWorker
from vision_app.core.logger import log


# ---------------------------------------------------------------------------
# SourceController
# ---------------------------------------------------------------------------
class SourceController:
    """
    Manages the input source selection (webcam index, video file, image dir)
    and provides a Play/Stop toggle button.

    The actual worker start/stop is delegated to InferenceViewWidget.
    """

    _SOURCES = ["Webcam (0)", "Webcam (1)", "Video File…", "Image Directory…"]

    def __init__(self, parent: "InferenceViewWidget"):
        self._parent = parent

        self.group = QGroupBox("Input Source")
        layout = QVBoxLayout(self.group)
        layout.setSpacing(8)

        self._source_combo = QComboBox()
        self._source_combo.addItems(self._SOURCES)
        layout.addWidget(self._source_combo)

        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setEnabled(False)
        layout.addWidget(self._browse_btn)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setObjectName("primary_button")
        layout.addWidget(self._play_btn)

        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._browse_btn.clicked.connect(self._on_browse)
        self._play_btn.clicked.connect(self._parent._on_play_stop)

        self._custom_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_playing(self, playing: bool):
        self._play_btn.setText("■  Stop" if playing else "▶  Play")
        self._source_combo.setEnabled(not playing)
        self._browse_btn.setEnabled(
            not playing and self._source_combo.currentIndex() >= 2
        )

    def get_source(self):
        """Return the OpenCV-compatible source (int or str)."""
        idx = self._source_combo.currentIndex()
        if idx == 0:
            return 0
        if idx == 1:
            return 1
        # "Video File…" or "Image Directory…" — use browsed path
        return self._custom_path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_source_changed(self, index: int):
        self._browse_btn.setEnabled(index >= 2)
        self._custom_path = None

    def _on_browse(self):
        idx = self._source_combo.currentIndex()
        if idx == 2:  # Video File
            path, _ = QFileDialog.getOpenFileName(
                self._parent, "Select Video File", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
        else:  # Image Directory
            path = QFileDialog.getExistingDirectory(
                self._parent, "Select Image Directory"
            )
        if path:
            self._custom_path = path


# ---------------------------------------------------------------------------
# CameraViewport
# ---------------------------------------------------------------------------
class CameraViewport(QLabel):
    """
    Displays live frames from StreamWorker.

    The label scales the pixmap to fit its current size (keepAspectRatio)
    using Qt.KeepAspectRatio without upscaling artefacts.  An optional
    OverlayPainter child widget is kept in sync during resize.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #11111b;")

        self._current_image: Optional[QImage] = None
        self.overlay: Optional["OverlayPainter"] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_frame(self, q_image: QImage):
        """Scale the incoming QImage to fit and display it."""
        self._current_image = q_image
        self._refresh_pixmap()

    def clear_frame(self):
        self._current_image = None
        self.clear()

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()
        if self.overlay:
            self.overlay.setGeometry(self.rect())
            self.overlay.raise_()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _refresh_pixmap(self):
        if self._current_image is None:
            return
        pixmap = QPixmap.fromImage(self._current_image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


# ---------------------------------------------------------------------------
# OverlayPainter
# ---------------------------------------------------------------------------
class OverlayPainter(QWidget):
    """
    Transparent widget that floats over CameraViewport.

    Draws a semi-transparent box at the top of the frame showing the top-k
    predictions with probability bars.

    Call update_results(list) to refresh the display.
    """

    _BAR_MAX_W  = 160   # pixels for 100 % probability
    _ROW_H      = 26
    _PAD        = 10
    _BOX_ALPHA  = 180   # 0–255

    def __init__(self, parent: "InferenceViewWidget"):
        super().__init__(parent._viewport)
        self._parent  = parent
        self._results: list[tuple[str, float]] = []

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_results(self, results: list):
        """
        Args:
            results : list of (class_name: str, probability: float) tuples.
        """
        self._results = results
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        if not self._results:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        n = len(self._results)
        box_h = n * self._ROW_H + self._PAD * 2
        box_w = 280

        # Semi-transparent background
        painter.setBrush(QColor(24, 24, 37, self._BOX_ALPHA))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self._PAD, self._PAD, box_w, box_h, 8, 8)

        for i, (name, prob) in enumerate(self._results):
            y = self._PAD * 2 + i * self._ROW_H

            # Probability bar background
            painter.setBrush(QColor(49, 50, 68))
            painter.drawRoundedRect(
                self._PAD * 2, y + 4,
                self._BAR_MAX_W, self._ROW_H - 8, 3, 3
            )

            # Filled portion
            bar_w = max(4, int(prob * self._BAR_MAX_W))
            color = QColor("#a6e3a1") if prob > 0.7 else QColor("#f9e2af") if prob > 0.4 else QColor("#f38ba8")
            painter.setBrush(color)
            painter.drawRoundedRect(
                self._PAD * 2, y + 4,
                bar_w, self._ROW_H - 8, 3, 3
            )

            # Class name + percentage text
            painter.setPen(QPen(QColor("#cdd6f4")))
            painter.drawText(
                self._PAD * 2 + self._BAR_MAX_W + 6, y,
                120, self._ROW_H,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"{name[:16]}  {prob * 100:.1f}%",
            )

        painter.end()


# ---------------------------------------------------------------------------
# RecordingManager
# ---------------------------------------------------------------------------
class RecordingManager:
    """
    Optional frame-to-video recording.

    Frames are appended as BGR via add_frame(QImage).  Output is saved to
    storage/logs/recording_<timestamp>.avi.
    """

    _FPS    = 25
    _FOURCC = cv2.VideoWriter_fourcc(*"XVID")

    def __init__(self, parent: "InferenceViewWidget"):
        self._parent    = parent
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._out_path: Optional[Path] = None

        self.group = QGroupBox("Recording")
        layout = QHBoxLayout(self.group)
        self._rec_btn = QPushButton("● Record")
        self._rec_btn.setObjectName("danger_button")
        self._rec_btn.clicked.connect(self._toggle)
        layout.addWidget(self._rec_btn)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._status_lbl)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_frame(self, q_image: QImage):
        if not self._recording or self._writer is None:
            return
        try:
            img = q_image.convertToFormat(QImage.Format.Format_RGB888)
            ptr = img.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                img.height(), img.width(), 3
            )
            self._writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        except Exception:
            pass  # frame drop; non-critical

    def stop(self):
        if self._recording:
            self._stop_recording()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _toggle(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        logs_dir = self._parent._storage_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_path = logs_dir / f"recording_{ts}.avi"

        # Derive frame size from the viewport's current size
        vp = self._parent._viewport
        w = max(vp.width(), 640)
        h = max(vp.height(), 480)

        self._writer = cv2.VideoWriter(
            str(self._out_path), self._FOURCC, self._FPS, (w, h)
        )
        self._recording = True
        self._rec_btn.setText("■  Stop")
        log.info("RecordingManager", f"Recording started: {self._out_path.name}")
        self._status_lbl.setText(self._out_path.name)

    def _stop_recording(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        self._recording = False
        self._rec_btn.setText("● Record")
        log.info("RecordingManager", f"Recording stopped: {self._out_path.name if self._out_path else 'none'}")
        self._status_lbl.setText(
            f"Saved: {self._out_path.name}" if self._out_path else ""
        )


# ---------------------------------------------------------------------------
# InferenceViewWidget
# ---------------------------------------------------------------------------
class InferenceViewWidget(QWidget):
    """
    "Inference" tab — the real-time application of the trained model.

    Layout:
        ┌─ Left panel (240px) ─┬─ Right panel (stretch) ─┐
        │  Model selector      │   CameraViewport         │
        │  SourceController    │     └ OverlayPainter      │
        │  RecordingManager    │                          │
        └──────────────────────┴──────────────────────────┘

    Submodules:
        self.source_ctrl   — SourceController
        self.overlay       — OverlayPainter
        self.recorder      — RecordingManager
        self._viewport     — CameraViewport (internal, exposed for overlay)
    """

    def __init__(self, storage_root: Path, parent=None):
        super().__init__(parent)
        self._storage_root = Path(storage_root)
        self._stream_worker: Optional[StreamWorker] = None
        self._stream_thread: Optional[QThread]      = None

        # ── Build submodules ──
        self.source_ctrl = SourceController(self)
        self._viewport   = CameraViewport(self)
        self.overlay     = OverlayPainter(self)
        self.recorder    = RecordingManager(self)

        # Link overlay to viewport
        self._viewport.overlay = self.overlay
        self.overlay.setGeometry(self._viewport.rect())
        self.overlay.raise_()

        # ── Layout ──
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left panel
        left = QWidget()
        left.setFixedWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(12)

        left_layout.addWidget(self._make_title("Inference"))

        # Model selector
        self._model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(self._model_group)
        self._model_combo = QComboBox()
        self._refresh_models_btn = QPushButton("Refresh")
        self._refresh_models_btn.clicked.connect(self._refresh_models)
        mg_layout.addWidget(self._model_combo)
        mg_layout.addWidget(self._refresh_models_btn)
        self._refresh_models()
        left_layout.addWidget(self._model_group)

        left_layout.addWidget(self.source_ctrl.group)
        left_layout.addWidget(self.recorder.group)
        left_layout.addStretch()

        # Status
        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        left_layout.addWidget(self._status_label)

        root_layout.addWidget(left)
        root_layout.addWidget(self._viewport, stretch=1)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def stop_stream(self):
        """Gracefully stop the stream. Safe to call when not streaming."""
        if self._stream_worker:
            self._stream_worker.stop()
        if self._stream_thread and self._stream_thread.isRunning():
            self._stream_thread.wait(3_000)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_title")
        return lbl

    def _refresh_models(self):
        self._model_combo.blockSignals(True)
        current = self._model_combo.currentText()
        self._model_combo.clear()
        models_root = self._storage_root / "models"
        if models_root.exists():
            for f in sorted(models_root.iterdir()):
                if f.suffix == ".pth":
                    self._model_combo.addItem(f.name, userData=str(f))
        idx = self._model_combo.findText(current)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    def _selected_model_path(self) -> Optional[Path]:
        data = self._model_combo.currentData()
        return Path(data) if data else None

    # ------------------------------------------------------------------
    # Play / Stop
    # ------------------------------------------------------------------

    def _on_play_stop(self):
        if self._stream_thread and self._stream_thread.isRunning():
            self._do_stop()
        else:
            self._do_play()

    def _do_play(self):
        source = self.source_ctrl.get_source()
        if source is None:
            self._status_label.setText("Browse to a video file / directory first.")
            return

        model_path = self._selected_model_path()
        if model_path is None or not model_path.exists():
            self._status_label.setText("Select a .pth model first.")
            return

        # Build model + label_map via InferenceEngine
        try:
            from vision_app.core.inference import InferenceEngine
            engine = InferenceEngine(model_path)
        except Exception as exc:
            self._status_label.setText(f"Failed to load model: {exc}")
            return

        self._stream_worker = StreamWorker(
            source=source,
            model=engine.model,
            label_map=engine.label_map,
        )
        self._stream_thread = QThread(self)
        self._stream_worker.moveToThread(self._stream_thread)

        self._stream_worker.frame_ready.connect(self._viewport.update_frame)
        self._stream_worker.frame_ready.connect(self.recorder.add_frame)
        self._stream_worker.results_ready.connect(self.overlay.update_results)
        self._stream_worker.status_changed.connect(self._status_label.setText)
        self._stream_worker.finished.connect(self._on_stream_finished)
        self._stream_thread.started.connect(self._stream_worker.run)
        self._stream_worker.finished.connect(self._stream_thread.quit)
        self._stream_thread.finished.connect(self._stream_thread.deleteLater)

        self._stream_thread.start()
        self.source_ctrl.set_playing(True)
        log.info("InferenceViewWidget", f"Live inference started: model={model_path.name if model_path else 'none'}")
        self._status_label.setText("Stream starting…")

    def _do_stop(self):
        if self._stream_worker:
            self._stream_worker.stop()
        self.recorder.stop()
        self.source_ctrl.set_playing(False)
        log.info("InferenceViewWidget", "Live inference stopped by user")
        self._status_label.setText("Stopping…")

    def _on_stream_finished(self):
        self._stream_worker = None
        self._viewport.clear_frame()
        self.overlay.update_results([])
        self.source_ctrl.set_playing(False)
        self._status_label.setText("Stream stopped.")
