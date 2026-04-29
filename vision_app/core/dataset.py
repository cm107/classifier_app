"""
dataset.py — PyTorch Dataset classes and image transformation pipelines.

Classes:
    ClassificationDataset   — ImageFolder-compatible Dataset with label smoothing
    ContrastiveTransformations — Dual-view augmentation for SSL (SimCLR/NT-Xent)
    StandardTransformations    — Augmented training & clean validation transforms
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image extensions recognised as valid samples
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# ClassificationDataset
# ---------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    """
    Reads images from an ImageFolder-style directory:
        root_path/
            class_a/
                img1.jpg
            class_b/
                img2.jpg

    Args:
        root_path      : Path to a single split directory (e.g. datasets/X/train).
        transform      : A callable applied to the PIL Image before returning.
        label_smoothing: If > 0, returns a soft one-hot vector instead of an int
                         label. Smoothing is distributed uniformly across all
                         non-target classes.
    """

    def __init__(
        self,
        root_path: Path,
        transform=None,
        label_smoothing: float = 0.0,
    ):
        self.root_path = Path(root_path)
        self.transform = transform
        self.label_smoothing = label_smoothing

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._load_samples()

    def _load_samples(self):
        classes = sorted(
            d.name for d in self.root_path.iterdir() if d.is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = self.root_path / cls
            for img_path in sorted(cls_path.iterdir()):
                if (
                    img_path.is_file()
                    and img_path.suffix.lower() in _IMAGE_EXTENSIONS
                ):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.label_smoothing > 0.0:
            n_classes = len(self.class_to_idx)
            smooth = self.label_smoothing / max(n_classes - 1, 1)
            soft_label = torch.full((n_classes,), smooth)
            soft_label[label] = 1.0 - self.label_smoothing
            return image, soft_label

        return image, label


# ---------------------------------------------------------------------------
# ContrastiveTransformations
# ---------------------------------------------------------------------------
class ContrastiveTransformations:
    """
    Returns two independently-augmented views of the same image for
    self-supervised learning (SimCLR / NT-Xent).

    Augmentation pipeline (fixed per the SSL standard):
        RandomResizedCrop → RandomHorizontalFlip → ColorJitter (p=0.8)
        → RandomGrayscale (p=0.2) → GaussianBlur (p=0.5)
        → ToTensor → Normalize

    Args:
        image_size : Target crop size (default 224).
        mean / std : Normalisation statistics; defaults to ImageNet values
                     until custom stats are computed.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: list = None,
        std: list = None,
    ):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        # Kernel size must be odd; derive from image_size
        blur_kernel = image_size // 10 * 2 + 1

        self._transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                )],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=blur_kernel)], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image) -> tuple:
        """Return (view_1, view_2) — two independent augmentations."""
        return self._transform(image), self._transform(image)


# ---------------------------------------------------------------------------
# StandardTransformations
# ---------------------------------------------------------------------------
class StandardTransformations:
    """
    Standard transforms for supervised training (augmented) or validation
    (centre-crop only).

    Args:
        image_size : Target size for the model input (default 224).
        mean / std : Per-channel normalisation stats.
                     Pass the output of StatsCalculator for best accuracy.
        augment    : True for the training split, False for val/test.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: list = None,
        std: list = None,
        augment: bool = True,
    ):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            # Resize slightly larger then centre-crop — standard eval protocol
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)
