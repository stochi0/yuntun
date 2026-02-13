import torch
from typing import Tuple, Optional


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10000.0,
    context_length: int = 4096,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos / sin tables (float32 recommended) and return (cos, sin)
    shapes: (context_length, head_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    half = head_dim // 2
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half))
    )
    positions = torch.arange(0, context_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def apply_rope_with_offset(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_offset: int = 0
) -> torch.Tensor:
    """
    Apply RoPE to x.
    x: (batch, heads, seq, head_dim)
    cos/sin: (context_len, head_dim)
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]

    # Ensure cos/sin are large enough
    if position_offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"RoPE index out of range: request {position_offset + seq_len}, max {cos.shape[0]}"
        )

    positions = torch.arange(
        position_offset, position_offset + seq_len, device=cos.device
    )

    # Slice and reshape for broadcasting: (1, 1, seq, head_dim)
    cos_slice = cos[positions].view(1, 1, seq_len, head_dim)
    sin_slice = sin[positions].view(1, 1, seq_len, head_dim)

    x_half1 = x[..., : head_dim // 2]
    x_half2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x_half2, x_half1), dim=-1)

    out = x * cos_slice.to(x.dtype) + rotated * sin_slice.to(x.dtype)
    return out
