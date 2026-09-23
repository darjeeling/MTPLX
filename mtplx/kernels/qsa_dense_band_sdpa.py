"""Flash-Next dense-band attention without the materialized ``where``.

Below the block-sparse crossover a wide QSA prefill forward attends through a
boolean mask (the top-512 blocks, the visible tail and the causal edge).  At
head dim 256 MLX has no fused kernel for that call, so
``mx.fast.scaled_dot_product_attention`` runs its fallback:

    scores = (bf16(scale) * q) @ k^T              # [B, Hkv, rep, S, T] bf16
    scores = where(mask, scores, finfo(bf16).min)  # a second full copy
    probs  = softmax(scores, precise=True)         # a third
    out    = probs @ v

At 4,096 rows and 16K of history each of those planes is 3.2 GB, so the
``where`` alone moves 6.4 GB per layer.  This module runs the same fallback
with the mask folded into the softmax: one Metal kernel reads the scores and
the mask and writes the probabilities.  The kernel is MLX's own
``softmax_single_row`` / ``softmax_looped`` (kernels/softmax.h: float
accumulation, ``fast::exp``, four reads per thread, the looped variant above
4,096 columns at 1,024 threads) with the masked value substituted at the load,
and the two matmuls are the same MLX matmuls the fallback issues, so the output
is bit-identical to the stock call (tests/test_qsa_dense_band_sdpa.py).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

#: MLX's softmax switches from one pass per row to the looped kernel above this
#: many columns (SOFTMAX_LOOPED_LIMIT), and reads four values per thread.
_LOOPED_LIMIT = 4096
_N_READS = 4
_LOOPED_THREADS = 1024
#: The fallback's masked value: ``finfo(bfloat16).min`` (a fully masked row
#: attends uniformly, it does not produce NaN).
_MASK_VALUE = "-3.38953139e38f"

_COMMON = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int axis_size = T_LEN;
    const device T* in = scores + size_t(row) * size_t(axis_size);
    const device bool* m = mask + size_t(row % S_LEN) * size_t(axis_size);
    device T* o = out + size_t(row) * size_t(axis_size);
    const float MASKED = float(T(MASK_VALUE));
    const float NEG_INF = -metal::numeric_limits<float>::infinity();
    const float FINITE_MIN = -metal::numeric_limits<float>::max();
    threadgroup float local_max[32];
    threadgroup float local_normalizer[32];
"""

# kernels/softmax.h softmax_single_row, with where(mask, x, MASKED) at the load.
_SINGLE_ROW = _COMMON + """
    float ld[N_READS];
    const int base = int(lid) * N_READS;
    if (base + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) {
            ld[i] = m[base + i] ? float(in[base + i]) : MASKED;
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            ld[i] = (base + i < axis_size) ? (m[base + i] ? float(in[base + i]) : MASKED) : NEG_INF;
        }
    }
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = NEG_INF;
        local_normalizer[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = FINITE_MIN;
    for (int i = 0; i < N_READS; i++) {
        maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    if (simd_lane_id == 0) {
        local_max[simd_group_id] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        maxval = simd_max(local_max[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_max[0] = maxval;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
        float exp_x = metal::fast::exp(ld[i] - maxval);
        ld[i] = exp_x;
        normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (simd_lane_id == 0) {
        local_normalizer[simd_group_id] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        normalizer = simd_sum(local_normalizer[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_normalizer[0] = normalizer;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_normalizer[0];
    if (base + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) {
            o[base + i] = T(ld[i] * normalizer);
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            if (base + i < axis_size) {
                o[base + i] = T(ld[i] * normalizer);
            }
        }
    }
"""

# kernels/softmax.h softmax_looped, with where(mask, x, MASKED) at both loads.
_LOOPED = _COMMON + """
    const int lsize = int(threads_per_threadgroup.x);
    const int rounds = (axis_size + N_READS * lsize - 1) / (N_READS * lsize);
    float prevmax;
    float maxval = FINITE_MIN;
    float normalizer = 0;
    for (int r = 0; r < rounds; r++) {
        int offset = r * lsize * N_READS + int(lid) * N_READS;
        float vals[N_READS];
        if (offset + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
                vals[i] = m[offset + i] ? float(in[offset + i]) : MASKED;
            }
        } else {
            for (int i = 0; i < N_READS; i++) {
                vals[i] = (offset + i < axis_size)
                    ? (m[offset + i] ? float(in[offset + i]) : MASKED) : NEG_INF;
            }
        }
        prevmax = maxval;
        for (int i = 0; i < N_READS; i++) {
            maxval = (maxval < vals[i]) ? vals[i] : maxval;
        }
        normalizer *= metal::fast::exp(prevmax - maxval);
        for (int i = 0; i < N_READS; i++) {
            normalizer += metal::fast::exp(vals[i] - maxval);
        }
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= metal::fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    prevmax = maxval;
    if (simd_lane_id == 0) {
        local_max[simd_group_id] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = simd_max(local_max[simd_lane_id]);
    normalizer *= metal::fast::exp(prevmax - maxval);
    if (simd_lane_id == 0) {
        local_normalizer[simd_group_id] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    normalizer = 1 / normalizer;
    for (int r = 0; r < rounds; r++) {
        int offset = r * lsize * N_READS + int(lid) * N_READS;
        for (int i = 0; i < N_READS; i++) {
            if (offset + i < axis_size) {
                float v = m[offset + i] ? float(in[offset + i]) : MASKED;
                o[offset + i] = T(metal::fast::exp(v - maxval) * normalizer);
            }
        }
    }
"""


@lru_cache(maxsize=2)
def _kernel(looped: bool):
    return mx.fast.metal_kernel(
        name="mtplx_qsa_dense_masked_softmax_" + ("looped" if looped else "row"),
        input_names=["scores", "mask"],
        output_names=["out"],
        header=f"#define MASK_VALUE ({_MASK_VALUE})\n#define N_READS {_N_READS}\n",
        source=_LOOPED if looped else _SINGLE_ROW,
    )


def masked_softmax(scores: mx.array, mask: mx.array) -> mx.array:
    """``softmax(where(mask, scores, finfo.min), axis=-1, precise=True)`` in one pass.

    ``scores`` is bf16 ``[..., S, T]``; ``mask`` is a bool ``[S, T]`` plane
    shared by every leading index (the dense band's per-row mask).
    """

    S, T = int(scores.shape[-2]), int(scores.shape[-1])
    rows = scores.size // T
    looped = T > _LOOPED_LIMIT
    if looped:
        threads = _LOOPED_THREADS
    else:
        threads = 32 * (((T + _N_READS - 1) // _N_READS + 31) // 32)
    (probs,) = _kernel(looped)(
        inputs=[scores, mask],
        template=[("T", scores.dtype), ("S_LEN", S), ("T_LEN", T)],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[scores.shape],
        output_dtypes=[scores.dtype],
    )
    return probs


def dense_band_eligible(q: mx.array, k: mx.array, mask) -> bool:
    """The calls whose stock SDPA is MLX's unfused fallback, and nothing else.

    Head dim 256 (no fused MLX kernel), more than 8 query rows (the vector
    kernel's band), bf16, GQA with equal q/v head dims, and a boolean
    ``[1, 1, S, T]`` mask.  Metadata only.
    """

    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_:
        return False
    if q.ndim != 4 or k.ndim != 4 or q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16:
        return False
    B, Hq, S, D = (int(d) for d in q.shape)
    Hkv, T = int(k.shape[1]), int(k.shape[2])
    if D != 256 or int(k.shape[3]) != D or S <= 8 or B != 1:
        return False
    if Hkv <= 0 or Hq % Hkv != 0:
        return False
    if tuple(int(d) for d in mask.shape) != (1, 1, S, T):
        return False
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def dense_band_sdpa(q, k, v, *, scale: float, mask: mx.array) -> mx.array:
    """MLX's SDPA fallback with the mask folded into the softmax.

    Same ops in the same order as ``mx.fast.scaled_dot_product_attention``'s
    fallback for these shapes; only the ``where`` + ``softmax`` pair runs as
    one kernel.  Callers check :func:`dense_band_eligible` first.
    """

    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    q = mx.array(scale, dtype=q.dtype) * q
    if rep > 1:
        q = q.reshape(B, Hkv, rep, S, D)
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2))
    probs = masked_softmax(scores, mask.reshape(S, -1))
    out = mx.matmul(probs, v)
    if rep > 1:
        out = out.reshape(B, Hq, S, -1)
    return out
