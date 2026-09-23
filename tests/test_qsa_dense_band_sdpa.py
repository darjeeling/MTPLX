"""Dense-band QSA attention with the mask folded into the softmax (2026-09-23).

``mtplx.kernels.qsa_dense_band_sdpa`` must be bit-identical to
``mx.fast.scaled_dot_product_attention`` wherever it is eligible (head dim 256,
where MLX runs its unfused fallback): on both of MLX's softmax kernels (one
pass up to 4,096 columns, looped above), across the 4,096 boundary, for sparse
and dense masks and for a fully masked row.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels.qsa_dense_band_sdpa import (
    dense_band_eligible,
    dense_band_sdpa,
    masked_softmax,
)


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _case(S, T, density, seed, hq=24, hkv=2, d=256):
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, S, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, T, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, T, d)).astype(mx.bfloat16)
    qpos = (T - S) + mx.arange(S)
    tpos = mx.arange(T)
    causal = tpos[None, :] <= qpos[:, None]
    picked = mx.random.uniform(shape=(S, T)) < density
    tail = tpos[None, :] >= qpos[:, None] - 63
    mask = (causal & (picked | tail))[None, None]
    return q, k, v, mask


@pytest.mark.parametrize(
    "S,T,density",
    [(64, 4096, 0.25),     # one-pass softmax, exactly the limit
     (37, 4082, 0.25),     # one-pass, ragged last thread
     (16, 1000, 0.5),      # one-pass, a few simdgroups
     (40, 4097, 0.25),     # looped, one column past the limit
     (64, 8192, 0.25),     # looped (the 8K dense chunk)
     (48, 12288, 0.17),
     (32, 16384, 0.125)],  # looped (the 16K dense chunk)
)
def test_bit_identical_to_the_stock_sdpa(S, T, density):
    q, k, v, mask = _case(S, T, density, seed=S * 7 + T)
    scale = 256 ** -0.5
    assert dense_band_eligible(q, k, mask)
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    got = dense_band_sdpa(q, k, v, scale=scale, mask=mask)
    assert got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize("T", [3000, 9000])
def test_a_fully_masked_row_attends_uniformly_like_the_fallback(T):
    q, k, v, mask = _case(12, T, 0.3, seed=T)
    mask = mx.array(np.array(mask).copy())
    mask[..., 0, :] = False
    scale = 256 ** -0.5
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    got = dense_band_sdpa(q, k, v, scale=scale, mask=mask)
    assert not np.isnan(np.array(got.astype(mx.float32))).any()
    assert np.array_equal(_bits(got), _bits(want))


def test_masked_softmax_is_where_then_softmax():
    mx.random.seed(3)
    scores = (mx.random.normal((2, 3, 20, 5000)) * 4).astype(mx.bfloat16)
    mask = mx.random.uniform(shape=(20, 5000)) < 0.4
    want = mx.softmax(mx.where(mask, scores, mx.array(-3.3895313892515355e38, mx.bfloat16)), axis=-1, precise=True)
    assert np.array_equal(_bits(masked_softmax(scores, mask)), _bits(want))


def test_eligibility_is_the_fallback_regime_only():
    q, k, v, mask = _case(16, 512, 0.5, seed=1)
    assert dense_band_eligible(q, k, mask)
    assert not dense_band_eligible(q[:, :, :8], k, mask[:, :, :8])          # vector-kernel band
    assert not dense_band_eligible(q[..., :128], k[..., :128], mask)        # MLX has a fused kernel
    assert not dense_band_eligible(q.astype(mx.float16), k.astype(mx.float16), mask)
    assert not dense_band_eligible(q, k, None)
    assert not dense_band_eligible(q, k, mask.astype(mx.bfloat16))          # additive masks stay stock
    assert not dense_band_eligible(q, k, mx.broadcast_to(mask, (1, 24, 16, 512)))


def test_the_model_switch_and_the_prefill_gate(monkeypatch):
    from mtplx.attention_context import attention_phase
    from mtplx.models import qwen4_exp

    q, k, v, mask = _case(64, 512, 0.5, seed=2)
    monkeypatch.delenv("MTPLX_QSA_DENSE_BAND_SDPA", raising=False)
    with attention_phase("prefill"):
        assert qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
        assert not qwen4_exp._qsa_dense_band_sdpa_applies(q[:, :, :31], k, mask[:, :, :31])
        for off in ("0", "false", "no", "off"):
            monkeypatch.setenv("MTPLX_QSA_DENSE_BAND_SDPA", off)
            assert not qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
    monkeypatch.delenv("MTPLX_QSA_DENSE_BAND_SDPA", raising=False)
    for phase in (None, "verify", "decode"):
        with attention_phase(phase):
            assert not qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
