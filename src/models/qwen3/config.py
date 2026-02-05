from dataclasses import dataclass
import torch
from typing import Optional

@dataclass(slots=True)
class Qwen3ModelConfig:
    vocab_size: int = 151_936
    max_position_embeddings: int = 40_960
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    qk_norm: bool = True
    rope_theta: float = 1_000_000.0
    use_bf16: bool = True
    
    # Runtime settings merged
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.dtype is None:
            if self.use_bf16 and self.device == "cuda":
                bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                if not bf16_ok:
                    try:
                        torch.tensor([1.0], dtype=torch.bfloat16, device=self.device)
                        bf16_ok = True
                    except Exception:
                        bf16_ok = False
                self.dtype = torch.bfloat16 if bf16_ok else torch.float32
            else:
                self.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        return self.hidden_size // self.num_attention_heads
