"""Configuration for the Transformer model."""
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 64        # longest sequence (source or target) the model can handle
    pad_id: int = 0
    tie_weights: bool = True  # share embedding and output projection (needs shared vocab)
