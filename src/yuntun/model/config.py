from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class Qwen3Config:
    """Configuration class for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960  # Default for Qwen3-0.6B
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False

    # Qwen specific
    qk_norm: bool = True
    head_dim: Optional[int] = (
        None  # If set, overrides hidden_size // num_attention_heads
    )

    # Tensor Parallelism
    tp_size: int = 1

    def __post_init__(self):
        self.head_dim = (
            self.head_dim
            if self.head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be <= num_attention_heads={self.num_attention_heads}"
            )

    @property
    def device(self) -> torch.device:
        # Helper to get default device if needed, though usually passed explicitly in code
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
