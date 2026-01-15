# mm/projectors.py
from __future__ import annotations

import torch
from torch import nn


class MLP2Projector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, text_dim, bias=True),
            nn.GELU(),
            nn.Linear(text_dim, text_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, vision_dim)
        return self.net(x)


def build_projector(kind: str, vision_dim: int, text_dim: int) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp2":
        return MLP2Projector(vision_dim=vision_dim, text_dim=text_dim)
    if kind == "perceiver":
        raise NotImplementedError("Perceiver resampler not yet implemented.")
    raise ValueError(f"Unknown projector kind: {kind}")
