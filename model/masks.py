"""Padding and causal masks."""
"""Boolean masks. Convention everywhere: True = may attend, False = blocked."""
import torch


def padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, L) token ids -> (B, 1, 1, L) mask over the key positions."""
    return (seq != pad_id)[:, None, None, :]


def causal_mask(length: int, device=None) -> torch.Tensor:
    """(1, 1, L, L) lower-triangular mask: position i sees positions <= i."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))[None, None]


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Padding mask AND look-ahead mask for decoder self-attention -> (B, 1, T, T)."""
    return padding_mask(tgt, pad_id) & causal_mask(tgt.size(1), tgt.device)
