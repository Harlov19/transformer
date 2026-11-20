"""Scaled dot-product attention and multi-head attention."""
import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    q: (..., T, d_k)   k: (..., S, d_k)   v: (..., S, d_v)
    mask: bool, broadcastable to (..., T, S); True = attend.
    Returns (output (..., T, d_v), attention weights (..., T, S)).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    out = (dropout(weights) if dropout is not None else weights) @ v
    return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.keep_attn = False   # set True to store weights for visualisation
        self.attn = None

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)  # (B, H, L, d_k)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.drop)
        if self.keep_attn:
            self.attn = weights.detach()  # (B, H, T, S)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d_k)
        return self.w_o(out)
