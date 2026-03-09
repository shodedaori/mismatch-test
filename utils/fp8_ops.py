"""FP8 simulation utilities.

Supports two FP8 attention backends selected via :class:`Fp8Backend`:

* ``SIMULATE`` (default) – Simulates FP8 E4M3 precision loss by
  down-casting and up-casting tensors using per-tensor max-abs scaling.
  Works on **any** GPU (no FP8 Tensor Core required).

* ``FLASHINFER`` – Uses FlashInfer's real FP8 attention kernel
  (``flashinfer.single_prefill_with_kv_cache`` /
  ``flashinfer.single_decode_with_kv_cache``).
  Requires GPU Compute Capability ≥ 8.9 (Ada / Hopper) and the
  ``flashinfer`` package to be installed.
"""

from __future__ import annotations

import enum
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# FP8 E4M3 constants
# ---------------------------------------------------------------------------

# FP8 E4M3 max representable value
_FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class Fp8Backend(enum.Enum):
    """Select how FP8 precision is applied in the attention kernel.

    Attributes:
        SIMULATE:    Software simulation via cast-down / cast-up (any GPU).
        FLASHINFER:  Real FP8 kernel via the ``flashinfer`` library
                     (requires GPU CC ≥ 8.9 and flashinfer installed).
    """
    SIMULATE = "simulate"
    FLASHINFER = "flashinfer"


# ---------------------------------------------------------------------------
# Backend: SIMULATE
# ---------------------------------------------------------------------------

def simulate_fp8(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Simulate FP8 precision loss via cast-down / cast-up.

    Uses per-tensor max-abs scaling so values fit in the FP8 E4M3 range.
    Falls back to manual mantissa rounding when float8 types are unavailable.

    Args:
        tensor: Input tensor (any floating dtype).
        dtype: Target FP8 dtype (default: float8_e4m3fn).

    Returns:
        Tensor of the **same dtype** as input, with FP8 precision loss applied.
    """
    if tensor.numel() == 0:
        return tensor

    original_dtype = tensor.dtype

    # Per-tensor scaling
    amax = tensor.detach().abs().amax().clamp(min=1e-12)
    scale = amax / _FP8_E4M3_MAX

    scaled = tensor / scale

    try:
        quantized = scaled.to(dtype).to(original_dtype)
    except (RuntimeError, TypeError):
        quantized = _manual_fp8_round(scaled).to(original_dtype)

    return quantized * scale


def _manual_fp8_round(tensor: torch.Tensor) -> torch.Tensor:
    """Fallback: simulate FP8 E4M3 by rounding to 3 mantissa bits."""
    signs = tensor.sign()
    abs_vals = tensor.abs().clamp(min=1e-38)

    exponents = torch.floor(torch.log2(abs_vals))
    step = torch.pow(2.0, exponents - 3)  # 3 mantissa bits
    quantized = (abs_vals / step).round() * step

    return signs * quantized


# ---------------------------------------------------------------------------
# Backend: FLASHINFER
# ---------------------------------------------------------------------------

def _check_flashinfer() -> None:
    """Raise ImportError if flashinfer is not installed."""
    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "flashinfer is not installed. "
            "Install it with: pip install flashinfer\n"
            "Note: flashinfer requires Linux and GPU CC ≥ 8.9 (Ada/Hopper)."
        ) from exc


def fp8_preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
    """Quantize Q/K/V to float8_e4m3fn and compute per-tensor scales.

    This matches FlashInfer's expected input contract:
    caller provides pre-quantized FP8 tensors + float scalar scales.

    Args:
        q: Query tensor, shape ``[num_heads, seq_len, head_dim]``, BF16.
        k: Key tensor, same layout.
        v: Value tensor, same layout.

    Returns:
        Tuple of ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``
        where the scales are Python floats (``amax / 448.0``).
    """
    def _quantize(t: torch.Tensor):
        amax = t.detach().abs().amax().clamp(min=1e-12)
        scale = (amax / _FP8_E4M3_MAX).item()
        t_fp8 = (t / scale).to(torch.float8_e4m3fn)
        return t_fp8, scale

    q_fp8, q_scale = _quantize(q)
    k_fp8, k_scale = _quantize(k)
    v_fp8, v_scale = _quantize(v)
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


def flashinfer_fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Run FP8 attention via the FlashInfer kernel.

    Quantizes Q/K/V to ``float8_e4m3fn``, then calls
    ``flashinfer.single_prefill_with_kv_cache`` with per-tensor scales.
    The output is returned in **BF16**.

    Args:
        q: Query, shape ``[num_heads, seq_len, head_dim]``, BF16.
        k: Key, same layout.
        v: Value, same layout.
        sm_scale: Softmax scale (typically ``1 / sqrt(head_dim)``).
        causal: Whether to apply causal masking.

    Returns:
        Attention output in BF16, shape ``[num_heads, seq_len, head_dim]``.
    """
    _check_flashinfer()
    import flashinfer

    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = fp8_preprocess_qkv(q, k, v)

    # FlashInfer expects [seq_len, num_heads, head_dim]
    q_fi = q_fp8.transpose(0, 1).contiguous()   # [S, H, D]
    k_fi = k_fp8.transpose(0, 1).contiguous()
    v_fi = v_fp8.transpose(0, 1).contiguous()

    out = flashinfer.single_prefill_with_kv_cache(
        q_fi, k_fi, v_fi,
        causal=causal,
        sm_scale=sm_scale,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
    )
    # out: [S, H, D] -> [H, S, D]
    return out.transpose(0, 1).to(torch.bfloat16)
