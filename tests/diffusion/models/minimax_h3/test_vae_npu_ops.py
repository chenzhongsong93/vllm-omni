# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the NPU operator set of the MiniMax H3 video VAE decoder.

The fused Q/K norm+RoPE path must reproduce the remote checkpoint's eager
semantics exactly; the NPU entry must fall back (return ``None``) wherever it
cannot guarantee that.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import npu_qk_norm_rope as npu_ops


class _StubNpuOps:
    """Reference implementations of the two kernels ``torch_npu`` would run."""

    def __init__(self, rotary_mode_seen: list[str] | None = None) -> None:
        self.rotary_mode_seen = rotary_mode_seen if rotary_mode_seen is not None else []

    @staticmethod
    def npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), gamma, float(eps))
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(eps))
        return normalized, rstd

    def npu_rotary_mul(self, x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, mode: str) -> torch.Tensor:
        self.rotary_mode_seen.append(mode)
        assert mode == "half"
        return x * r1 + npu_ops._rotate_half(x) * r2


def _reference_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The remote checkpoint's eager sequence: FP32 norm, cast back, rotate-half."""

    def one(tensor: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.rms_norm(tensor.float(), (tensor.shape[-1],), None, float(eps)).to(
            tensor.dtype
        )
        c = cos.to(tensor.dtype)
        s = sin.to(tensor.dtype)
        return normalized * c + npu_ops._rotate_half(normalized) * s

    return one(query), one(key)


def test_fused_qk_matches_reference_semantics() -> None:
    torch.manual_seed(0)
    batch, seq, heads, head_dim = 2, 5, 3, 8
    query = torch.randn(batch, seq, heads, head_dim, dtype=torch.float16)
    key = torch.randn(batch, seq, heads, head_dim, dtype=torch.float16) * 0.5
    cos = torch.randn(batch, seq, 1, head_dim, dtype=torch.float16).abs() + 0.1
    sin = torch.randn(batch, seq, 1, head_dim, dtype=torch.float16) * 0.3

    expected_q, expected_k = _reference_qk(query, key, cos, sin, 1e-6)
    actual_q, actual_k = npu_ops._fuse_qk(query, key, cos, sin, 1e-6, _StubNpuOps())

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_fused_qk_uses_half_rotary_mode() -> None:
    ops = _StubNpuOps()
    query = torch.randn(1, 2, 2, 4, dtype=torch.float16)
    cos = torch.ones(1, 2, 1, 4, dtype=torch.float16)
    sin = torch.zeros(1, 2, 1, 4, dtype=torch.float16)

    npu_ops._fuse_qk(query, query, cos, sin, 1e-6, ops)

    assert ops.rotary_mode_seen == ["half", "half"]


def test_ones_gamma_is_reused_per_shape() -> None:
    ops = _StubNpuOps()
    query = torch.randn(1, 2, 2, 4, dtype=torch.float16)
    cos = torch.ones(1, 2, 1, 4, dtype=torch.float16)
    sin = torch.zeros(1, 2, 1, 4, dtype=torch.float16)
    npu_ops._ONES_GAMMA.clear()

    npu_ops._fuse_qk(query, query, cos, sin, 1e-6, ops)
    first = dict(npu_ops._ONES_GAMMA)
    npu_ops._fuse_qk(query, query, cos, sin, 1e-6, ops)

    assert len(npu_ops._ONES_GAMMA) == len(first)


def test_try_qk_norm_rope_returns_none_for_cpu_tensors() -> None:
    query = torch.randn(1, 2, 2, 4, dtype=torch.float16)
    cos = torch.ones(1, 2, 1, 4, dtype=torch.float16)
    sin = torch.zeros(1, 2, 1, 4, dtype=torch.float16)

    assert npu_ops.try_qk_norm_rope_npu(query, query, (cos, sin), 1e-6) is None


def test_partial_rotary_is_rejected() -> None:
    query = torch.randn(1, 2, 2, 8, dtype=torch.float16)
    cos = torch.ones(1, 2, 1, 4, dtype=torch.float16)  # half of head_dim
    sin = torch.zeros(1, 2, 1, 4, dtype=torch.float16)

    assert npu_ops._rotation_supported(query, query, cos, sin) is False


def test_full_rotary_is_accepted() -> None:
    query = torch.randn(1, 2, 2, 8, dtype=torch.float16)
    cos = torch.ones(1, 2, 1, 8, dtype=torch.float16)
    sin = torch.zeros(1, 2, 1, 8, dtype=torch.float16)

    assert npu_ops._rotation_supported(query, query, cos, sin) is True
    assert npu_ops._rotation_supported(query, torch.randn_like(query), None, sin) is False


def test_operator_table_has_an_npu_entry() -> None:
    assert any(entry.supports is dispatch._supports_npu for entry in dispatch.H3_VAE_OPERATOR_TABLE)


def test_supports_npu_is_false_on_cpu() -> None:
    assert dispatch._supports_npu(torch.device("cpu")) is False


def test_scaled_residual_falls_back_on_npu() -> None:
    hidden = torch.randn(2, 3)
    update = torch.randn(2, 3)
    scale = torch.randn(3)
    assert dispatch._no_scaled_residual(hidden, update, scale) is None


if __name__ == "__main__":  # pragma: no cover - convenience for direct runs
    raise SystemExit(pytest.main([__file__, "-q"]))
