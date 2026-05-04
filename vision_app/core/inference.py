"""
inference.py — InferenceEngine: batch & single-frame prediction.

Classes:
    InferenceEngine — Loads a ScratchResNet checkpoint and runs predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from vision_app.core.logger import log
from vision_app.core.model import ScratchResNet


class InferenceEngine:
    """
    Loads a trained ScratchResNet checkpoint and provides a simple
    predict_batch() / predict_single() interface.

    This class is pure Python — zero Qt dependencies.  It is safe to
    instantiate and call from any QThread (as StreamWorker does), provided
    torch.no_grad() is active for memory efficiency.

    Args:
        checkpoint_path : Path to a .pth checkpoint produced by StateManager.
        device          : "cuda", "cpu", or None for auto-detect.
        top_k           : Number of top predictions returned per image.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
        top_k: int = 3,
    ):
        self._top_k = top_k
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model, self._label_map = self._load(Path(checkpoint_path))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def label_map(self) -> dict:
        """int → class_name mapping extracted from the checkpoint."""
        return self._label_map

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_batch(self, tensor: torch.Tensor) -> list:
        """
        Run inference on a pre-processed batch.

        Args:
            tensor : Float tensor of shape (B, C, H, W), already normalised.

        Returns:
            List of B lists; each inner list contains
            (class_name: str, probability: float) tuples sorted descending
            by probability, length = min(top_k, num_classes).
        """
        tensor = tensor.to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)

        k = min(self._top_k, probs.size(1))
        top_probs, top_indices = probs.topk(k, dim=1)

        results = []
        for prob_row, idx_row in zip(top_probs, top_indices):
            results.append([
                (self._label_map.get(idx.item(), str(idx.item())), p.item())
                for idx, p in zip(idx_row, prob_row)
            ])
        return results

    def predict_single(self, tensor: torch.Tensor) -> list:
        """
        Convenience wrapper for a single image tensor (C, H, W) or (1, C, H, W).

        Returns:
            list of (class_name: str, probability: float) tuples.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return self.predict_batch(tensor)[0]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, checkpoint_path: Path) -> tuple:
        """Reconstruct model + label_map from a checkpoint file."""
        # weights_only=False is required because checkpoints contain non-tensor
        # metadata (label_map dict, epoch int, etc.).  These files are produced
        # by our own training pipeline and stored in a trusted local directory
        # (storage/models/), so the security risk is acceptable.
        ckpt = torch.load(
            str(checkpoint_path),
            map_location=self._device,
            weights_only=False,
        )

        meta: dict = ckpt.get("meta", {})
        label_map: dict = meta.get("label_map", {})
        # JSON round-trip may convert int keys to str; normalise to int.
        label_map = {int(k): v for k, v in label_map.items()}

        num_classes = ckpt.get("num_classes", len(label_map) or 2)
        model = ScratchResNet(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(self._device)
        model.eval()

        log.info("InferenceEngine", f"Model loaded: {checkpoint_path.name}, classes={num_classes}, device={self._device}")
        return model, label_map
