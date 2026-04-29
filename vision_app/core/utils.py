"""
utils.py — Shared utilities for the core engine.

Classes:
    StatsCalculator  — Per-channel mean/std computation over an ImageFolder split.
    ModelExporter    — TorchScript tracing and optional ONNX export (Milestone 6).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# StatsCalculator
# ---------------------------------------------------------------------------
class StatsCalculator:
    """
    Compute per-channel mean and standard deviation over a dataset split.

    Usage:
        calc = StatsCalculator()
        mean, std = calc.compute(Path("storage/datasets/MyData/train"))
        # Pass mean/std to StandardTransformations or ContrastiveTransformations.

    Args:
        image_size  : Images are resized to this square before computing stats.
        batch_size  : DataLoader batch size (larger = faster, more RAM).
        num_workers : DataLoader worker threads.
    """

    def __init__(
        self,
        image_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute(self, dataset_path: Path) -> tuple:
        """
        Compute channel-wise mean and std for all images under dataset_path.
        dataset_path must follow the ImageFolder layout (class subdirs).

        Returns:
            (mean, std) — each a list of 3 floats in [R, G, B] order.
        """
        dataset_path = Path(dataset_path)

        _transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),  # scales to [0, 1]
        ])

        dataset = datasets.ImageFolder(str(dataset_path), transform=_transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        # Welford-style online accumulation for numerical stability
        mean = torch.zeros(3)
        m2 = torch.zeros(3)
        n_pixels = 0

        for images, _ in loader:
            # images: (B, C, H, W)
            b, c, h, w = images.shape
            pixels = images.permute(1, 0, 2, 3).reshape(c, -1)  # (C, B*H*W)
            batch_n = pixels.shape[1]

            batch_mean = pixels.mean(dim=1)
            batch_var = pixels.var(dim=1, unbiased=False)

            # Parallel Welford merge
            delta = batch_mean - mean
            mean += delta * batch_n / (n_pixels + batch_n)
            m2 += batch_var * batch_n + delta ** 2 * n_pixels * batch_n / (n_pixels + batch_n)
            n_pixels += batch_n

        std = (m2 / n_pixels).sqrt()
        return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# ModelExporter — stub, implemented in Milestone 6
# ---------------------------------------------------------------------------
class ModelExporter:
    """TorchScript tracing and optional ONNX export."""

    def export_torchscript(self, model, output_path: Path, example_input_size=(1, 3, 224, 224)):
        raise NotImplementedError("ModelExporter is implemented in Milestone 6.")

    def export_onnx(self, model, output_path: Path, example_input_size=(1, 3, 224, 224)):
        raise NotImplementedError("ModelExporter is implemented in Milestone 6.")
