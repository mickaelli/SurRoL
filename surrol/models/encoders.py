"""Encoders for SurRoL observations.

This provides a thin interface so image encoders can be swapped (CNN, DINO, CLIP-like).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """Base visual encoder.

    The default implementation is a light CNN; replace `forward` in subclasses to plug in
    pretrained DINO/CLIP-style backbones.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        return F.adaptive_avg_pool2d(x, 1).view(image.size(0), -1)


class StateVisualFusion(nn.Module):
    """Fuse visual features with state vectors when obs_mode is rgb+state/both."""

    def __init__(self, visual_encoder: VisualEncoder, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.visual = visual_encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.visual.out_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.visual(image)
        fused = torch.cat([v, state], dim=-1)
        return self.mlp(fused)


def build_visual_encoder(kind: str = "cnn", in_channels: int = 3, out_dim: int = 256) -> VisualEncoder:
    if kind == "cnn":
        return VisualEncoder(in_channels=in_channels, out_dim=out_dim)
    raise ValueError(f"Unknown visual encoder kind: {kind}")
