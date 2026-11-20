"""Shared Transformer model layers."""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sine/cosine encodings (eq. 1-2 of the notebook), followed by dropout."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        pos = torch.arange(max_len).unsqueeze(1)                              # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)         # (1, L, D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
