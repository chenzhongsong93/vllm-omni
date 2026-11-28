# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Q/K RMSNorm + RoPE for the MiniMax H3 video VAE on Ascend NPU.

The remote checkpoint normalizes Q/K in FP32 (``MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM``),
casts back to the activation dtype, then applies rotate-half RoPE. Eager execution
turns that into ~15 kernels per block (cast, square, mean, rsqrt, mul, chunk, neg,
cat, cat, mul, add). The CUDA operator set fuses it with Triton; this module does
the same with two ``torch_npu`` kernels:

* ``npu_rms_norm`` -- affine-free RMSNorm (a cached all-ones gamma keeps the
  multiply exact, as in the reference where the weight is ``None``);
* ``npu_rotary_mul(..., rotary_mode="half")`` -- ``x * cos + rotate_half(x) * sin``.

Every unsupported case returns ``None`` so the caller falls back to the remote
implementation instead of raising.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["npu_rms_norm_available", "try_qk_norm_rope_npu"]

_TORCH_NPU: Any | None = None
_PROBED = False
_ONES_GAMMA: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def _npu_ops() -> Any | None:
    """Return ``torch_npu`` when both kernels this module needs exist."""

    global _TORCH_NPU, _PROBED
    if not _PROBED:
        _PROBED = True
        try:
            import torch_npu

            _TORCH_NPU = torch_npu
        except Exception:  # noqa: BLE001 - any import failure means "no NPU"
            _TORCH_NPU = None
    if _TORCH_NPU is None:
        return None
    if not hasattr(_TORCH_NPU, "npu_rms_norm") or not hasattr(_TORCH_NPU, "npu_rotary_mul"):
        return None
    return _TORCH_NPU


def npu_rms_norm_available() -> bool:
    """Whether this process can run the fused NPU path at all."""

    return _npu_ops() is not None


def _ones_gamma(ops: Any, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (dim, device.type, device.index, dtype)
    gamma = _ONES_GAMMA.get(key)
    if gamma is None:
        gamma = torch.ones(dim, dtype=dtype, device=device)
        _ONES_GAMMA[key] = gamma
    return gamma


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Reference helper, used only by the pure-torch checks in tests."""

    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _fuse_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    ops: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse norm+rope with an injected ``ops`` object (``torch_npu`` or a stub).

    Semantics must match the remote checkpoint exactly:
    ``norm(x.float()) -> x.dtype -> x * cos + rotate_half(x) * sin``.
    """

    head_dim = int(query.shape[-1])

    def one(tensor: torch.Tensor) -> torch.Tensor:
        gamma = _ones_gamma(ops, head_dim, tensor.device, torch.float32)
        normalized, _ = ops.npu_rms_norm(tensor.float(), gamma, float(eps))
        normalized = normalized.to(tensor.dtype)
        return ops.npu_rotary_mul(
            normalized,
            cos.to(tensor.dtype),
            sin.to(tensor.dtype),
            "half",
        )

    return one(query), one(key)


def _rotation_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> bool:
    """Whether the pair can take the fused kernel without changing semantics."""

    if cos is None or sin is None or not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        return False
    if query.ndim != 4 or key.ndim != 4 or query.shape != key.shape:
        return False
    if cos.ndim != 4 or sin.ndim != 4:
        return False
    # Full rotary only: partial rotary would need a slice+cat around the kernel.
    if cos.shape[-1] != query.shape[-1] or sin.shape[-1] != query.shape[-1]:
        return False
    if cos.shape[:-2] != query.shape[:-2] or sin.shape[:-2] != query.shape[:-2]:
        return False
    if cos.shape[-2] != 1 or sin.shape[-2] != 1:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return bool(query.is_contiguous() and key.is_contiguous())


def try_qk_norm_rope_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused Q/K normalization + RoPE, or ``None`` to use the remote path."""

    ops = _npu_ops()
    if ops is None or query.device.type != "npu" or key.device.type != "npu":
        return None
    if rotary_pos_emb is None or not isinstance(rotary_pos_emb, tuple) or len(rotary_pos_emb) != 2:
        return None
    cos, sin = rotary_pos_emb
    if not _rotation_supported(query, key, cos, sin):
        return None
    if not float(eps) > 0.0:
        return None
    return _fuse_qk(query, key, cos, sin, eps, ops)
