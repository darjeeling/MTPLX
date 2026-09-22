"""Before-measurement pins for the 8.2 GiB Bonsai pack; no policy changes.

The 4096-token fallback on a refused plan is NOT an admitted context.
Actual pack bytes (including the MTP sidecar) are read by the measurement
script; these deliberately pin the requested round-number 8.2 GiB baseline.
"""

import pytest

from mtplx.memory_plan import GIB, plan_memory


@pytest.mark.parametrize("ram,budget,admitted,off_context,q8_context", [
    (16, 12, False, 4096, 4096),
    (18, 13.5, True, 20480, 36864),
    (24, 18, True, 94208, 172032),
    (32, 24, True, 192512, 262144),
])
@pytest.mark.parametrize("quant", ["off", "q8"])
def test_bonsai_current_planner_verdict(ram, budget, admitted, off_context, q8_context, quant):
    plan = plan_memory(
        total_ram_bytes=ram * GIB,
        model_weights_bytes=8_804_682_956,  # floor(8.2 * GiB), including all weights
        kv_bytes_per_token=65_536,  # 16 full-attention layers, K+V, 4 heads, dim 256, fp16
        kv_quantization=quant,
        model_max_context=262_144,
    )
    assert plan.available
    assert plan.usable_bytes == int(budget * GIB)
    assert plan.model_fits is admitted
    assert plan.context_window_resolved == (off_context if quant == "off" else q8_context)
    assert plan.kv_bytes_per_token_effective == (65_536 if quant == "off" else 36_044)
    if not admitted:
        assert any("model does not fit" in note for note in plan.notes)
