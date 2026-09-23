"""Flash-Next's wide prefill projections as dequantize + dense GEMM (2026-09-23).

On by default for prefill forwards of 2,048 rows or more; decode and verify
widths keep the quantized matmul, and ``MTPLX_QWEN4_PREFILL_DQ_GEMM=0`` turns
the lane off.  The lane only ships because, at these widths, MLX's quantized
matmul and a dense GEMM over the dequantized weight are bit-identical: the
tests below pin that on two of the model's own projection shapes, so an MLX
upgrade that changes either kernel's accumulation order fails here first.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mtplx.attention_context import attention_phase
from mtplx.models import qwen4_exp


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    mx.eval(a, b)
    return a.dtype == b.dtype and np.array_equal(
        np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))
    )


def _quantized(n, k, bits, group, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=group, bits=bits)


def test_the_lane_is_on_by_default_and_the_switch_turns_it_off(monkeypatch):
    x = mx.zeros((1, 4096, 64), dtype=mx.bfloat16)
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    with attention_phase("prefill"):
        assert qwen4_exp._prefill_dq_gemm_applies(x)
        for off in ("0", "false", "no", "off", " OFF "):
            monkeypatch.setenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", off)
            assert not qwen4_exp._prefill_dq_gemm_applies(x)


@pytest.mark.parametrize(
    "rows,phase,expected",
    [(4096, "prefill", True), (2048, "prefill", True), (2047, "prefill", False),
     (4096, None, False), (4096, "verify", False), (1, "prefill", False)],
)
def test_the_lane_needs_a_wide_prefill_forward(monkeypatch, rows, phase, expected):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    x = mx.zeros((1, rows, 64), dtype=mx.bfloat16)
    with attention_phase(phase):
        assert qwen4_exp._prefill_dq_gemm_applies(x) is expected


@pytest.mark.parametrize(
    "n,k,bits,group",
    [(1280, 2560, 4, 32),   # routed/shared gate+up geometry, 4-bit g32
     (2560, 640, 8, 64)],   # shared-expert down projection, 8-bit g64
)
def test_the_dense_gemm_is_bit_identical_to_the_quantized_matmul(monkeypatch, n, k, bits, group):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    wq, sc, bi = _quantized(n, k, bits, group)
    x = (mx.random.normal((1, 2048, k)) * 0.5).astype(mx.bfloat16)
    with attention_phase("prefill"):
        got = qwen4_exp._projection(x, wq, sc, bi, group_size=group, bits=bits, mode="affine")
    want = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=group, bits=bits)
    assert _bits_equal(got, want)


def test_narrow_forwards_keep_the_quantized_matmul(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    wq, sc, bi = _quantized(64, 256, 4, 32)
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)

    def refuse(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a narrow forward dequantized its weight")

    monkeypatch.setattr(qwen4_exp.mx, "dequantize", refuse)
    with attention_phase("prefill"):
        got = qwen4_exp._projection(x, wq, sc, bi, group_size=32, bits=4, mode="affine")
    monkeypatch.undo()
    want = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=32, bits=4)
    assert _bits_equal(got, want)


def test_linear_routes_only_a_bias_free_quantized_linear(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    layer = nn.QuantizedLinear(256, 64, bias=False, group_size=32, bits=4)
    layer.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 2048, 256)).astype(mx.bfloat16)
    dense = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=32, bits=4)
    with attention_phase("prefill"):
        assert _bits_equal(qwen4_exp._linear(layer, x), mx.matmul(x, dense.T))
    # A layer with a bias, a plain Linear, and a verify-width call are the module call.
    biased = nn.QuantizedLinear(256, 64, bias=True, group_size=32, bits=4)
    biased.set_dtype(mx.bfloat16)
    plain = nn.Linear(256, 64, bias=False)
    plain.set_dtype(mx.bfloat16)
    with attention_phase("prefill"):
        assert _bits_equal(qwen4_exp._linear(biased, x), biased(x))
        assert _bits_equal(qwen4_exp._linear(plain, x), plain(x))
    with attention_phase("verify"):
        assert _bits_equal(qwen4_exp._linear(layer, x[:, :4]), layer(x[:, :4]))
