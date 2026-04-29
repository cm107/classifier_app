"""
stream_worker.py — Background QThread wrapper for live webcam / video inference.

The worker captures frames via OpenCV, runs them through the InferenceEngine,
and emits two signals per frame:
    frame_ready   — QImage ready to paint on a QLabel viewport
    results_ready — list of (class_name, probability) tuples for the overlay

FPS is capped at TARGET_FPS (default 30) using QThread.msleep() to prevent
overwhelming the GUI thread.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QMutex, QMutexLocker, QObject, QThread, Signal
from PySide6.QtGui import QImage
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_FPS: int = 30
_MS_PER_FRAME: int = 1000 // TARGET_FPS  # ~33 ms


# ---------------------------------------------------------------------------
# StreamWorker
# ---------------------------------------------------------------------------
class StreamWorker(QObject):
    """
    Captures frames from a webcam, video file, or image directory and
    emits them together with inference results.

    Signals:
        frame_ready   (QImage)       — BGR→RGB converted frame as QImage
        results_ready (list)         — [(class_name: str, prob: float), …]
                                       sorted by probability descending
        status_changed(str)          — human-readable status messages
        finished      ()             — emitted when the loop exits cleanly

    Args:
        source        : OpenCV-compatible source — int (webcam index), str
                        (file path), or None to defer.
        model         : A ScratchResNet (or compatible nn.Module) in eval mode.
        label_map     : {int: str} mapping from class index to class name.
        image_size    : Model input size (square crop).
        mean / std    : Normalisation stats matching those used during training.
        top_k         : Number of top predictions to include in results_ready.
    """

    frame_ready = Signal(QImage)
    results_ready = Signal(list)   # list[tuple[str, float]]
    status_changed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        source,
        model: torch.nn.Module,
        label_map: dict,
        image_size: int = 224,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        top_k: int = 3,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._source = source
        self._model = model
        self._label_map = label_map
        self._top_k = top_k
        self._abort = False
        self._mutex = QMutex()

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # ------------------------------------------------------------------
    # Public control (called from GUI thread)
    # ------------------------------------------------------------------

    def stop(self):
        with QMutexLocker(self._mutex):
            self._abort = True

    def set_source(self, source):
        """Hot-swap the video source before the next frame capture."""
        with QMutexLocker(self._mutex):
            self._source = source

    # ------------------------------------------------------------------
    # Main loop — runs on the worker QThread
    # ------------------------------------------------------------------

    def run(self):
        with QMutexLocker(self._mutex):
            source = self._source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status_changed.emit(f"Failed to open source: {source}")
            self.finished.emit()
            return

        self.status_changed.emit("Stream started.")
        self._model.eval()

        while True:
            with QMutexLocker(self._mutex):
                if self._abort:
                    break

            loop_start_ms = QThread.currentThread().property("_loop_start")

            ret, frame_bgr = cap.read()
            if not ret:
                self.status_changed.emit("End of stream.")
                break

            # --- BGR → RGB ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- Emit display frame ---
            h, w, ch = frame_rgb.shape
            q_image = QImage(
                frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
            ).copy()  # .copy() detaches from the numpy buffer
            self.frame_ready.emit(q_image)

            # --- Run inference ---
            results = self._infer(frame_rgb)
            self.results_ready.emit(results)

            # --- FPS cap ---
            QThread.msleep(_MS_PER_FRAME)

        cap.release()
        self.status_changed.emit("Stream stopped.")
        self.finished.emit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer(self, frame_rgb: np.ndarray) -> list:
        """
        Run the model on a single frame and return top-k results.

        Returns:
            [(class_name, probability), …] sorted by probability descending.
        """
        try:
            tensor = self._preprocess(frame_rgb).unsqueeze(0)  # (1, C, H, W)
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)

            top_probs, top_indices = probs.topk(
                min(self._top_k, probs.size(0))
            )

            return [
                (self._label_map.get(idx.item(), str(idx.item())), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        except Exception:  # noqa: BLE001
            return []
