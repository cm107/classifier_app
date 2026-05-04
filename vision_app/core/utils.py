"""
utils.py — Shared utilities for the core engine.

Classes:
    StatsCalculator  — Per-channel mean/std computation over an ImageFolder split.
    ConfigLoader     — Read/write config.yaml using PyYAML.
    ModelExporter    — TorchScript tracing and optional ONNX export (Milestone 6).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_app.core.logger import log


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
# ModelExporter
# ---------------------------------------------------------------------------
class ModelExporter:
    """
    Export a trained ScratchResNet to portable inference formats.

    Methods:
        export_torchscript — Trace and save as a .pt TorchScript file.
        export_onnx        — Export to ONNX (requires 'onnx' package).
    """

    def export_torchscript(
        self,
        model: torch.nn.Module,
        output_path: Path,
        example_input_size: tuple = (1, 3, 224, 224),
    ) -> Path:
        """
        Trace the model with torch.jit.trace and save as a .pt file.

        Args:
            model             : nn.Module in eval mode.
            output_path       : Destination .pt file path.
            example_input_size: Dummy input shape for tracing (B, C, H, W).

        Returns:
            Resolved path to the saved .pt file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy = torch.zeros(*example_input_size, device=device)

        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)

        traced.save(str(output_path))
        log.info("ModelExporter", f"Exported TorchScript: {output_path.name}")
        return output_path

    def export_onnx(
        self,
        model: torch.nn.Module,
        output_path: Path,
        example_input_size: tuple = (1, 3, 224, 224),
        opset_version: int = 17,
    ) -> Path:
        """
        Export the model to ONNX format.

        Requires the 'onnx' package (pip install onnx).

        Returns:
            Resolved path to the saved .onnx file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy = torch.zeros(*example_input_size, device=device)

        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        log.info("ModelExporter", f"Exported ONNX: {output_path.name}")
        return output_path


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------
class ConfigLoader:
    """
    Read and write config.yaml using PyYAML.

    The config file is resolved relative to the project root (parent of the
    directory containing utils.py → vision_app/core/../.. = project root).

    Usage:
        loader = ConfigLoader()               # uses default config.yaml location
        cfg = loader.load()
        cfg["defaults"]["batch_size"] = 64
        loader.save(cfg)

    Args:
        config_path : Explicit path to config.yaml. If None, resolves to
                      <project_root>/config.yaml automatically.
    """

    def __init__(self, config_path: Path = None):
        if config_path is None:
            # vision_app/core/utils.py → ../../ = project root
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        self.config_path = Path(config_path)

    def load(self) -> dict:
        """Load and return the config as a plain dict."""
        import yaml

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {self.config_path}. "
                "Ensure config.yaml exists in the project root."
            )
        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        log.info("ConfigLoader", f"Config loaded: {self.config_path.name}")
        return config

    def save(self, config: dict):
        """Persist a modified config dict back to config.yaml."""
        import yaml

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        log.info("ConfigLoader", f"Config saved: {self.config_path.name}")

    def get(self, *keys, default=None):
        """
        Convenience accessor for nested keys.
        e.g. loader.get("defaults", "batch_size", default=32)
        """
        cfg = self.load()
        for key in keys:
            if not isinstance(cfg, dict):
                return default
            cfg = cfg.get(key, default)
        return cfg
