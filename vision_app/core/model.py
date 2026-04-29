"""
model.py — Neural network architecture for scratch training.

Classes:
    ResidualBlock   — Two 3x3 convolutions with a shortcut path and He init.
    ScratchResNet   — Full ResNet-style backbone with GAP and a dual head switch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions and an optional
    1x1 shortcut convolution for dimension matching.

    He (Kaiming) initialization is applied to all Conv2d weights in __init__.

    Args:
        in_channels  : Number of input feature maps.
        out_channels : Number of output feature maps.
        stride       : Stride applied to the first convolution and the shortcut.
                       stride=2 halves the spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # --- Main path ---
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Shortcut path ---
        # A 1x1 conv is needed whenever spatial size or channel count changes.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out = out + self.shortcut(identity)
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# ScratchResNet
# ---------------------------------------------------------------------------
class ScratchResNet(nn.Module):
    """
    ResNet-style backbone built entirely from ResidualBlocks.

    Architecture:
        Prep layer (3 → 64, stride 1)
        Stage 1   (64 → 128, stride 2)  × blocks_per_stage[0]
        Stage 2   (128 → 256, stride 2) × blocks_per_stage[1]
        Stage 3   (256 → 512, stride 2) × blocks_per_stage[2]
        GAP → flatten → head

    Dual Head Switch (controlled by self.use_projection_head):
        True  → Projection Head (2-layer MLP, for SSL / NT-Xent)
        False → Linear Classifier (single fc, for supervised training)

    Args:
        num_classes         : Number of output classes for supervised mode.
        projection_dim      : Output dimension of the SSL projection head.
        blocks_per_stage    : Number of ResidualBlocks in each of the 3 stages.
        image_size          : Used only for documentation; GAP handles any size.
    """

    def __init__(
        self,
        num_classes: int,
        projection_dim: int = 128,
        blocks_per_stage: tuple = (2, 2, 2),
    ):
        super().__init__()

        # --- Initial prep layer ---
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.prep[0].weight, mode="fan_out", nonlinearity="relu")

        # --- Residual stages ---
        self.layer1 = self._make_stage(64,  128, stride=2, n_blocks=blocks_per_stage[0])
        self.layer2 = self._make_stage(128, 256, stride=2, n_blocks=blocks_per_stage[1])
        self.layer3 = self._make_stage(256, 512, stride=2, n_blocks=blocks_per_stage[2])

        # --- Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 512, H, W) → (B, 512, 1, 1)

        # --- Projection Head (SSL) ---
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

        # --- Linear Classifier (Supervised) ---
        self.fc = nn.Linear(512, num_classes)

        # Default to supervised mode
        self.use_projection_head: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)  # (B, 512)

        if self.use_projection_head:
            return self.projection_head(x)
        return self.fc(x)

    @property
    def backbone_parameters(self):
        """All parameters excluding the two heads."""
        head_ids = {id(p) for p in self.projection_head.parameters()} | \
                   {id(p) for p in self.fc.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, stride: int, n_blocks: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
