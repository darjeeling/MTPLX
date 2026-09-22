"""Small-M 2-bit ternary GEMV for the Bonsai 2 trunk, bit-exact with stock MLX.

WHAT THE MEASUREMENT SAID (2026-09-21, M5 Max, MLX 0.32.2, real pack
``Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed``, fans at maximum, laptop in
use so every figure is a ratio taken in the same process, never a claim)
-------------------------------------------------------------------------
One decode step of the 27B ternary trunk at M = 1 is 22.5 ms in that window:
the 402 packed projections are about 21 ms of it (stock ``qmv_fast`` streams
the K = 5120 shapes at 350-420 GB/s and the N = 5120-6144 shapes at 250-300
GB/s against a 614 GB/s bus; the lm_head alone reaches 596 GB/s), the 256
Hadamard rotations are 2.4 ms (10.7 percent, four dispatches each), and
everything else is under 1 ms.  At M = 2 the step costs 1.24 plain steps, at
M = 4 1.93.  Receipts: ``outputs/bonsai-kernel/`` in the worktree.

On this GPU class (``applegpu_g15`` and later) stock ``mx.quantized_matmul``
dispatches two different kernels by row count: ``qmv_fast`` at M = 1 (32
lanes x 16 contiguous values per 512-block, byte sums, ``scale*accum +
sum*bias`` per block, ``simd_sum``) and ``qmv_wide`` at M = 2..12 (eight lanes
per output row striding the 128-value groups, eight-value sub-chunk sums of
``scale*code + bias`` values, a shuffle ladder).  Both form only exact
products (fp16 x fp16 fits fp32; ``scale*code + bias`` is one of -s, 0, +s),
so the order of the additions is the whole exactness contract and both orders
are portable.  A consequence worth knowing: the two stock orders round
differently, so a verify row at M >= 2 is NOT bit-identical to the same row
decoded at M = 1 today (measured: ``stock_equals_rowwise_decode`` false in
every cell).

WHAT WAS TRIED, ALL BIT-EXACT UNLESS NOTED, ALL AGAINST STOCK IN THE SAME
PROCESS (``outputs/bonsai-kernel/receipts``)
-------------------------------------------------------------------------
* M = 1 (``qmv_fast`` order): rows per simdgroup 1..16, simdgroups per
  threadgroup 1..4, two- and four-block unrolls, explicit register prefetch
  of 2..10 weight words, a byte-to-float4 code table in threadgroup memory,
  sibling projections in one dispatch: every variant 0.62..1.03x stock.
  Stock ``qmv_fast`` is at the optimum of its own arithmetic order here.
* M >= 2 (``qmv_wide`` order): fp32 activations, weights loaded up front,
  fewer active lane groups: all slower.  Two rows per eight-lane group with
  four simdgroups per threadgroup (this kernel) is the one layout that wins:
  1.0..1.15x per projection at M = 4 and 1.05..1.15x at M = 8, 0.85..0.99x at
  M = 2 and mixed at M = 3, so it is served for M = 4..8 only.  Full model step
  on the real pack: 1.009x at M = 4, 1.089x at M = 8, noise floor about 2
  percent (``outputs/bonsai-kernel/model-ab-wide-*.json``).
* A crossrow kernel in the M = 1 order for every M (verify rows then equal
  decode rows bit for bit, a stronger MTP property) is exact against the
  row-wise decode kernel at every M but 0.4..0.9x stock ``qmv_wide``.
* A one-dispatch rotate (fp16 in, signs, the Walsh-Hadamard butterflies of
  ``hadamard_n`` in registers with ``simd_shuffle_xor``, fp16 out) is exact
  and 2.5 us faster per rotation in a dependent chain, within noise at the
  model level; it is kept as a prototype under ``outputs/bonsai-kernel``.

WHY THE SWITCH IS OFF BY DEFAULT
--------------------------------
The pack ships at draft depth 1 and allows depth 3, so the verify rows the
product sees are M = 2 and at most M = 4, where this kernel is a no-op and a
one percent reading under load.  It is measured only on an M5 Max; a tune
that is a wash here must not risk a smaller Mac unmeasured.
``MTPLX_BONSAI_TERNARY_KERNEL=1`` arms it for the founder's quiet-window
A/B; the install probe then proves every served row count bit-exact against
stock on every packed shape of the loaded pack before a single call is
served, and any miss disables the kernel for the process with the reason in
the demotion ledger (``/health`` -> ``degradation.demotions``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import cache
from typing import Any

import mlx.core as mx

from mtplx import demotions

ENV = "MTPLX_BONSAI_TERNARY_KERNEL"
#: Row counts the kernel is written for.  The lower bound is a measurement,
#: not a limit of the kernel (it is exact from M = 2): see the module doc.
MIN_ROWS = 4
MAX_ROWS = 8
#: Stock ``qmv_wide`` tiles at most five vectors per threadgroup; the
#: per-vector chain never sees the tiling, so the same tiling keeps register
#: pressure equal to stock's at M = 6..8.
_MAX_VECTORS_PER_TILE = 5
_GROUP_SIZE = 128
_BITS = 2
_LANES_PER_ROW = 8
_ROWS_PER_LANE_GROUP = 2
_LANE_GROUPS = 4
_SIMDGROUPS = 4
ROWS_PER_THREADGROUP = _ROWS_PER_LANE_GROUP * _LANE_GROUPS * _SIMDGROUPS
#: Kind under which a disabled or refused kernel reports in the ledger.
DEMOTION_KIND = "bonsai_ternary_kernel_stock"

_COUNTS: dict[str, int] = {
    "served_calls": 0,
    "fallback_calls": 0,
    "probe_cells": 0,
    "probe_misses": 0,
}
_STATE: dict[str, Any] = {
    "installed": False,
    "served_rows": (),
    "disabled_reason": None,
    "probe": {},
}
#: (width, outputs) -> the row counts proven bit-exact on that shape.  A
#: shape is probed once per process; a pack is served the intersection over
#: its own shapes, so a second pack loaded later never rides another pack's
#: verdict.
_PROBED: dict[tuple[int, int], frozenset] = {}


def enabled() -> bool:
    """``MTPLX_BONSAI_TERNARY_KERNEL`` is off unless set to a true value."""

    return (os.environ.get(ENV, "0").strip().lower()) in {"1", "true", "yes", "on"}


def counters() -> dict[str, int]:
    return dict(_COUNTS)


def engagement() -> dict[str, Any]:
    """Counters plus the install verdict: which path ran, and why."""

    report = dict(_COUNTS)
    report["enabled"] = enabled()
    report["installed"] = bool(_STATE["installed"])
    report["served_rows"] = list(_STATE["served_rows"])
    report["disabled_reason"] = _STATE["disabled_reason"]
    report["probe"] = dict(_STATE["probe"])
    return report


def reset_for_tests() -> None:
    for key in _COUNTS:
        _COUNTS[key] = 0
    _STATE.update(installed=False, served_rows=(), disabled_reason=None, probe={})
    _PROBED.clear()


# ---------------------------------------------------------------------------
# The kernel: the stock qmv_wide order, two rows per eight-lane group
# ---------------------------------------------------------------------------
_HEADER = """
    using namespace metal;

    // Stock dequantize<float, 8, 2>: sc[j] * (byte & mask_j) + b, which is
    // exactly -s, 0 or +s for a ternary code (the products are exact).
    inline void dequantize2(
        const device uint8_t* w, float scale, float bias, thread float* w_local) {
      const float s = float(scale);
      const float b = float(bias);
      float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
      for (int i = 0; i < 2; i++) {
        w_local[4 * i] = static_cast<float>(sc[0] * (w[i] & 0x03) + b);
        w_local[4 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0x0c) + b);
        w_local[4 * i + 2] = static_cast<float>(sc[2] * (w[i] & 0x30) + b);
        w_local[4 * i + 3] = static_cast<float>(sc[3] * (w[i] & 0xc0) + b);
      }
    }
"""

# Per (row, vector) the arithmetic is stock qmv_wide<T, 128, 2, V, 8>: lane
# k_lane sums the groups g = k_lane, k_lane + 8, ... in order, each group as
# sixteen eight-value sub-chunk sums added to the running result, then the
# eight lanes of a row combine with shuffle_down 4, 2, 1.  What differs from
# stock is only the layout: a lane group carries R = 2 rows, so the eight
# activation values it converts per sub-chunk serve two rows instead of one.
_SOURCE = """
    const int K = int(K_size);
    const int N = int(N_size);
    const int M = int(M_size);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const short k_lane = simd_lid % 8;
    const short lane_group = simd_lid / 8;
    constexpr int R = 2;
    constexpr int S = 4;
    constexpr int ROWS_PER_SG = 4 * R;
    const int out_row0 = int(threadgroup_position_in_grid.y) * (ROWS_PER_SG * S)
        + int(simd_gid) * ROWS_PER_SG + int(lane_group) * R;
    const int vec0 = int(threadgroup_position_in_grid.x) * V;
    const int in_vec_size_w = K / 4;
    const int in_vec_size_g = K / GS;
    const device uint8_t* w8 = (const device uint8_t*)w;
    const device uint8_t* wrow[R];
    const device T* srow[R];
    const device T* brow[R];
    for (int r = 0; r < R; ++r) {
      const int row = min(out_row0 + r, N - 1);
      wrow[r] = w8 + row * in_vec_size_w;
      srow[r] = scales + row * in_vec_size_g;
      brow[r] = biases + row * in_vec_size_g;
    }
    const device T* xv[V];
    for (int v = 0; v < V; ++v) xv[v] = x + min(vec0 + v, M - 1) * K;
    float result[R][V];
    for (int r = 0; r < R; ++r) for (int v = 0; v < V; ++v) result[r][v] = 0.0f;

    for (int g = k_lane; g < in_vec_size_g; g += 8) {
      float scale[R];
      float bias[R];
      for (int r = 0; r < R; ++r) {
        scale[r] = srow[r][g];
        bias[r] = brow[r][g];
      }
      for (int sc = 0; sc < GS / 8; ++sc) {
        const int k0 = g * GS + sc * 8;
        float xf[V][8];
        for (int v = 0; v < V; ++v) {
          const device T* xc = xv[v] + k0;
          for (int i = 0; i < 8; ++i) xf[v][i] = static_cast<float>(xc[i]);
        }
        for (int r = 0; r < R; ++r) {
          float w_dq[8];
          dequantize2(wrow[r] + k0 / 4, scale[r], bias[r], w_dq);
          for (int v = 0; v < V; ++v) {
            float acc = 0;
            for (int i = 0; i < 8; ++i) acc += xf[v][i] * w_dq[i];
            result[r][v] += acc;
          }
        }
      }
    }
    for (int r = 0; r < R; ++r) {
      for (int v = 0; v < V; ++v) {
        result[r][v] += simd_shuffle_down(result[r][v], 4);
        result[r][v] += simd_shuffle_down(result[r][v], 2);
        result[r][v] += simd_shuffle_down(result[r][v], 1);
      }
    }
    if (k_lane == 0) {
      for (int r = 0; r < R; ++r) {
        if (out_row0 + r < N) {
          for (int v = 0; v < V; ++v) {
            if (vec0 + v < M) y[(vec0 + v) * N + out_row0 + r] = static_cast<T>(result[r][v]);
          }
        }
      }
    }
"""


@cache
def _kernel(vectors_per_tile: int):
    return mx.fast.metal_kernel(
        name=f"mtplx_bonsai_ternary_wide_v{vectors_per_tile}",
        input_names=["x", "w", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=_SOURCE,
        header=_HEADER,
    )


def _tiling(rows: int) -> tuple[int, int]:
    """(tiles, vectors per tile), the stock qmv_wide tiling."""

    tiles = (rows + _MAX_VECTORS_PER_TILE - 1) // _MAX_VECTORS_PER_TILE
    return tiles, (rows + tiles - 1) // tiles


def stock_matmul(x: mx.array, module) -> mx.array:
    """The path this kernel must match: ``mx.quantized_matmul`` on the packed words."""

    return mx.quantized_matmul(
        x,
        module["weight"],
        scales=module["scales"],
        biases=module["biases"],
        transpose=True,
        group_size=int(module.group_size),
        bits=int(module.bits),
    )


def _rows(x: mx.array) -> int:
    return int(x.shape[-2]) if x.ndim >= 2 else 1


def shape_eligible(x: mx.array, module) -> bool:
    """Contract checks that do not depend on the install verdict."""

    if x.dtype != mx.float16 or x.ndim < 2:
        return False
    if int(getattr(module, "bits", 0) or 0) != _BITS:
        return False
    if int(getattr(module, "group_size", 0) or 0) != _GROUP_SIZE:
        return False
    if getattr(module, "mode", "affine") not in (None, "affine"):
        return False
    w = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    if w is None or scales is None or biases is None:
        return False
    k = int(x.shape[-1])
    if k % _GROUP_SIZE or k <= 0:
        return False
    if w.dtype != mx.uint32 or int(w.shape[-1]) * 16 != k:
        return False
    if scales.dtype != mx.float16 or biases.dtype != mx.float16:
        return False
    # One leading batch entry at most: the kernel reads [M, K] only.
    return int(x.size) == _rows(x) * k


def serves(x: mx.array, module) -> bool:
    """The hot-path check: this module was armed for this row count."""

    served = getattr(module, "_ternary_rows", None)
    return served is not None and _rows(x) in served


def ternary_matmul(x: mx.array, module) -> mx.array:
    """``x @ W.T`` for the module's served row counts; the stock path otherwise.

    Counted either way, so a request record can say which path ran.
    """

    if not serves(x, module) or not shape_eligible(x, module):
        _COUNTS["fallback_calls"] += 1
        demotions.note(
            DEMOTION_KIND, "an armed projection met a call the kernel does not take"
        )
        return stock_matmul(x, module)
    _COUNTS["served_calls"] += 1
    return _launch(x, module)


def _launch(x: mx.array, module) -> mx.array:
    lead = x.shape[:-2]
    rows = _rows(x)
    k = int(x.shape[-1])
    n = int(module["weight"].shape[0])
    tiles, per_tile = _tiling(rows)
    x2 = x.reshape(rows, k)
    (y,) = _kernel(per_tile)(
        inputs=[x2, module["weight"], module["scales"], module["biases"], rows, k, n],
        template=[("T", x.dtype), ("GS", _GROUP_SIZE), ("V", per_tile)],
        grid=(
            32 * tiles,
            _SIMDGROUPS * ((n + ROWS_PER_THREADGROUP - 1) // ROWS_PER_THREADGROUP),
            1,
        ),
        threadgroup=(32, _SIMDGROUPS, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*lead, rows, n)


# ---------------------------------------------------------------------------
# The install probe
# ---------------------------------------------------------------------------
def _probe_input(rows: int, width: int, seed: int) -> mx.array:
    """Deterministic activations at the real width, sampled like a residual
    stream after the Hadamard transform (unit scale, a few large values)."""

    key = mx.random.key(seed)
    base = mx.random.normal((rows, width), key=key) * 2.0
    spikes = mx.random.uniform(shape=(rows, width), key=mx.random.split(key)[1]) < 0.002
    return mx.where(spikes, mx.array(96.0), base).astype(mx.float16)


def _probe_shape(
    width: int, outputs: int, module, seeds: int
) -> tuple[frozenset, list[str]]:
    """The row counts bit-exact against stock on one shape, and the misses."""

    exact: set[int] = set()
    misses: list[str] = []
    for rows in range(MIN_ROWS, MAX_ROWS + 1):
        ok = True
        for seed in range(seeds):
            x = _probe_input(rows, width, 1000 * rows + seed)
            if not shape_eligible(x, module):
                ok = False
                misses.append(f"M={rows} {width}x{outputs}: shape refused")
                break
            _COUNTS["probe_cells"] += 1
            want = stock_matmul(x, module)
            got = _launch(x, module)
            same = mx.array_equal(want, got)
            mx.eval(same)
            if not bool(same.item()):
                _COUNTS["probe_misses"] += 1
                ok = False
                misses.append(f"M={rows} {width}x{outputs} seed {seed}")
                break
        if ok:
            exact.add(rows)
    return frozenset(exact), misses


def install(modules: Iterable[Any], *, logger=None, seeds: int = 2) -> dict[str, Any]:
    """Arm ``modules`` for exactly the row counts proven bit-exact against
    stock on every one of their shapes.

    ``modules`` are one pack's ``HadamardQuantizedLinear`` layers.  Each
    distinct (width, outputs) shape is probed once per process at every
    candidate row count on deterministic inputs; a row count with a miss on
    any shape of the pack is not served for that pack, and a pack with no
    row count left is not armed at all, with the reason in the demotion
    ledger.  Returns the report (also available from ``engagement()``).
    """

    linears = [
        m for m in modules if hasattr(m, "input_dims") and hasattr(m, "output_dims")
    ]
    for module in linears:
        module._ternary = None
        module._ternary_rows = None
    if not mx.metal.is_available():
        return _disable("Metal is not available; the kernel has no portable spelling")
    if not linears:
        return _disable("no packed projection to install on")
    shapes: dict[tuple[int, int], Any] = {}
    for module in linears:
        shapes.setdefault((int(module.input_dims), int(module.output_dims)), module)
    misses: list[str] = []
    served = frozenset(range(MIN_ROWS, MAX_ROWS + 1))
    for shape, module in shapes.items():
        if shape not in _PROBED:
            exact, shape_misses = _probe_shape(shape[0], shape[1], module, seeds)
            _PROBED[shape] = exact
            misses.extend(shape_misses)
        served &= _PROBED[shape]
    probe = {
        "shapes": [f"{w}x{n}" for (w, n) in shapes],
        "candidate_rows": list(range(MIN_ROWS, MAX_ROWS + 1)),
        "served_rows": sorted(served),
        "misses": misses,
        "cells": int(_COUNTS["probe_cells"]),
    }
    if not served:
        _STATE["probe"] = probe
        return _disable(
            "the install probe found no row count bit-exact against stock on every "
            "packed shape (" + "; ".join(misses[:4]) + ")"
        )
    for module in linears:
        module._ternary_rows = served
    _STATE.update(
        installed=True,
        served_rows=tuple(sorted(served)),
        disabled_reason=None,
        probe=probe,
    )
    if logger is not None:
        logger.info(
            "bonsai ternary kernel: on for verify rows %s after %d exact probe cells on %d shapes",
            sorted(served),
            probe["cells"],
            len(shapes),
        )
    return probe


def _disable(reason: str) -> dict[str, Any]:
    _STATE.update(installed=False, served_rows=(), disabled_reason=reason)
    _STATE["probe"] = dict(_STATE.get("probe") or {}, disabled_reason=reason)
    demotions.note(DEMOTION_KIND, reason)
    return dict(_STATE["probe"])


def disabled_reason() -> str | None:
    return _STATE["disabled_reason"]


def installed() -> bool:
    return bool(_STATE["installed"])


__all__ = [
    "DEMOTION_KIND",
    "ENV",
    "MAX_ROWS",
    "MIN_ROWS",
    "ROWS_PER_THREADGROUP",
    "counters",
    "disabled_reason",
    "enabled",
    "engagement",
    "install",
    "installed",
    "reset_for_tests",
    "serves",
    "shape_eligible",
    "stock_matmul",
    "ternary_matmul",
]
