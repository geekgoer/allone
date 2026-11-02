"""Residual control MLP with the same FC count and widths as a plain MLP."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_width: int):
        super().__init__()
        if min(input_dim, output_dim, hidden_layers, hidden_width) < 1:
            raise ValueError("all model dimensions must be positive")
        self.stem = nn.Linear(input_dim, hidden_width)
        self.blocks = nn.ModuleList(nn.Linear(hidden_width, hidden_width) for _ in range(hidden_layers - 1))
        self.head = nn.Linear(hidden_width, output_dim)
        self.residual_scale = 1.0 / math.sqrt(max(hidden_layers - 1, 1))
        for layer in (self.stem, *self.blocks):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for block in self.blocks:
            x = torch.relu(x + self.residual_scale * block(x))
        return self.head(x)
