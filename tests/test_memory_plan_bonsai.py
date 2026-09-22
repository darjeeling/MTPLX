"""Planner verdicts for the 8.2 GiB Bonsai pack, before and after the measured rule.

Before the 2026-09-21 measurement the 16 GiB class REFUSED this pack
(``model_fits`` False, the 4096 field a fallback, not an admission): the
12 GiB engine budget could not fund 8.2 GiB of weights + the 3 GiB runtime
transient + the 1 GiB bank floor + one KV block. The measurement
(``scripts/bonsai_memory_table.py``, outputs/release-2114/speed/
bonsai-memory-1810/memory.json) showed the peak WITHOUT a resident bank is
weights + dense KV + 3.056 to 3.075 GiB in every completed row, so on the
16 GiB class 4K peaks at 11.55 GiB and 8K at 11.78 to 11.80 GiB under the
12 GiB budget while 16K peaks at 12.11 GiB over it; the q8 KV setting did
not lower any of those peaks (the 16K peaks were byte-identical). The
tight-machine rule therefore admits 8192 tokens on 16 GiB with the bank
floor at zero and the KV counted at its dense width. Every other class is
unchanged.

Actual pack bytes (including the MTP sidecar) are read by the measurement
script; these deliberately pin the requested round-number 8.2 GiB baseline.
"""

import pytest

from mtplx.memory_plan import (
    BANK_FLOOR_BYTES,
    GIB,
    RUNTIME_TRANSIENTS_BYTES,
    TIGHT_MACHINE_MARGIN_BYTES,
    bank_dynamic_ceiling,
    describe_plan,
    plan_memory,
)

BONSAI_WEIGHTS = 8_804_682_956  # floor(8.2 * GiB), including all weights
BONSAI_KV_PER_TOKEN = 65_536  # 16 full-attention layers, K+V, 4 heads, dim 256, fp16


def _plan(ram: int, quant: str = "off", **kw):
    return plan_memory(
        total_ram_bytes=ram * GIB,
        model_weights_bytes=BONSAI_WEIGHTS,
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        kv_quantization=quant,
        model_max_context=262_144,
        **kw,
    )


@pytest.mark.parametrize("ram,budget,admitted,off_context,q8_context", [
    (16, 12, True, 8192, 8192),  # was (16, 12, False, 4096, 4096) before the rule
    (18, 13.5, True, 20480, 36864),
    (24, 18, True, 94208, 172032),
    (32, 24, True, 192512, 262144),
])
@pytest.mark.parametrize("quant", ["off", "q8"])
def test_bonsai_planner_verdict(ram, budget, admitted, off_context, q8_context, quant):
    plan = _plan(ram, quant)
    assert plan.available
    assert plan.usable_bytes == int(budget * GIB)
    assert plan.model_fits is admitted
    assert plan.context_window_resolved == (off_context if quant == "off" else q8_context)
    if ram == 16:
        # Tight machine: the KV rate is dense whatever the paged setting says.
        assert plan.kv_bytes_per_token_effective == 65_536
    else:
        assert plan.kv_bytes_per_token_effective == (65_536 if quant == "off" else 36_044)
        assert plan.tight_machine is False
        assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
        assert plan.runtime_transients_bytes == RUNTIME_TRANSIENTS_BYTES


@pytest.mark.parametrize("quant", ["off", "q8"])
def test_16_gib_admits_bonsai_with_the_bank_floor_at_zero(quant):
    plan = _plan(16, quant, dense_decode_ceiling=32_768)
    usable = 12 * GIB
    transients = RUNTIME_TRANSIENTS_BYTES + TIGHT_MACHINE_MARGIN_BYTES
    assert plan.model_fits and plan.tight_machine
    assert plan.bank_floor_bytes == 0
    assert plan.runtime_transients_bytes == transients
    # 12 GiB - 8.2 GiB - 3.25 GiB = 0.55 GiB of KV at 64 KiB per token:
    # 8.5K tokens, block-aligned down to 8192 (the measured 8K row peaked
    # 0.2 GiB under the budget; 12K would not).
    assert plan.context_window_fit == 8192
    assert plan.bank_idle_max_bytes == usable - BONSAI_WEIGHTS - transients
    assert plan.bank_steady_bytes == (
        usable - BONSAI_WEIGHTS - transients - 8192 * BONSAI_KV_PER_TOKEN
    )
    assert plan.headroom_bytes == 0
    # Steady state fits the envelope with the margin, not the bank floor.
    assert (
        BONSAI_WEIGHTS + plan.kv_reserve_bytes + plan.bank_steady_bytes + transients
        <= usable
    )
    assert any(note.startswith("tight machine") for note in plan.notes)
    assert not any("model does not fit" in note for note in plan.notes)
    assert "tight machine" in describe_plan(plan)
    assert "MODEL DOES NOT FIT" not in describe_plan(plan)
    data = plan.to_dict()
    assert data["tight_machine"] is True
    assert data["bank_floor_bytes"] == 0
    assert data["runtime_transients_bytes"] == transients


def test_tight_machine_dynamic_ceiling_can_reach_zero():
    plan = _plan(16, dense_decode_ceiling=32_768)
    assert bank_dynamic_ceiling(plan, 0) == plan.bank_idle_max_bytes
    # An 8K live KV (0.5 GiB) leaves 22 MiB; a 1 GiB working set leaves nothing.
    assert bank_dynamic_ceiling(plan, 8192 * BONSAI_KV_PER_TOKEN) == plan.bank_steady_bytes
    assert bank_dynamic_ceiling(plan, 1 * GIB) == 0
    # An observed spike below the plan's own transient does not loosen it.
    assert bank_dynamic_ceiling(plan, 0, transient_bytes=RUNTIME_TRANSIENTS_BYTES) == (
        plan.bank_idle_max_bytes
    )


def test_tight_machine_still_refuses_what_the_margin_cannot_fund():
    # 9.5 GiB of weights: 12 - 9.5 - 3.25 < 0 even without the bank floor.
    plan = plan_memory(
        total_ram_bytes=16 * GIB,
        model_weights_bytes=int(9.5 * GIB),
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        model_max_context=262_144,
    )
    assert plan.model_fits is False
    assert plan.tight_machine is False
    assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
    assert plan.context_window_resolved == 4096  # fallback, not an admission
    assert any("model does not fit" in note for note in plan.notes)


def test_tight_machine_counts_q8_kv_at_the_dense_width():
    # Measured: the 16K peaks on the 16 GiB class were byte-identical for
    # KV off and q8, so q8 must not buy phantom context on a tight machine.
    assert _plan(16, "q8").context_window_fit == _plan(16, "off").context_window_fit


def test_a_machine_that_funds_the_bank_floor_is_not_tight():
    plan = _plan(18, dense_decode_ceiling=32_768)
    assert plan.model_fits and not plan.tight_machine
    assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
    assert plan.bank_steady_bytes >= BANK_FLOOR_BYTES
