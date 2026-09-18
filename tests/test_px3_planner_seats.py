"""Planner replay on simulated seats (PX.3, 2026-09-18).

The fixture is the read-only planner replay of the 2026-09-18 policy audit
(`04_memory_plan_table.py`) turned into a test: pack constants measured from
disk that night, the server's Metal-cap arithmetic, and the same two calls
into ``plan_memory`` the server makes. No MLX model, no GPU.

Pack constants (bytes on disk, 2026-09-18):

* Flash-Next Optimized Speed: 83,037,597,915 B of weight files (77.33 GiB,
  the n-gram table excluded), 32,000,154,008 B streamed table, KV 24,576
  B/token, QSA aux 7,872 B/token, dense-prefill transient 104,448 B/token.
* Qwen3.8-27B Optimized Speed: 20,683,241,105 B (19.26 GiB), KV 65,536.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.memory_plan import GIB, plan_memory
from mtplx.server import openai

MODEL_MAX = 262_144

FLASH_NEXT = {
    "weights": 83_037_597_915,
    "table": 32_000_154_008,
    "kv": 24_576,
    "aux": 7_872,
    "dense_transient": 104_448,
    "resident_floor_family": True,
}
DENSE_27B = {
    "weights": 20_683_241_105,
    "table": 0,
    "kv": 65_536,
    "aux": 0,
    "dense_transient": 0,
    "resident_floor_family": False,
}


def _fake_mx():
    metal = SimpleNamespace(is_available=lambda: True)
    return SimpleNamespace(
        metal=metal,
        set_memory_limit=lambda value: None,
        set_wired_limit=lambda value: None,
    )


def _seat(pack: dict, ram_gb: int, *, sparse_prefill: bool = True):
    """(caps, fit plan, served default window) for one simulated seat."""

    ram = ram_gb * GIB
    floor = None
    if pack["resident_floor_family"]:
        floor = pack["weights"] + openai._resident_floor_margin_bytes(ram)
    caps = openai._apply_metal_memory_caps(
        mx_module=_fake_mx(), total_ram_bytes=ram, minimum_resident_bytes=floor
    )
    if not caps.get("applied"):
        return caps, None, None
    fit = plan_memory(
        total_ram_bytes=ram,
        model_weights_bytes=pack["weights"],
        ngram_table_streamed_bytes=pack["table"],
        kv_bytes_per_token=pack["kv"],
        model_max_context=MODEL_MAX,
        usable_bytes_override=caps["memory_limit_bytes"],
        usable_bytes_explicit=caps.get("memory_limit_source") == "env",
        aux_bytes_per_token=pack["aux"],
        prefill_transient_bytes_per_token=(
            0 if sparse_prefill else pack["dense_transient"]
        ),
    )
    from mtplx.backends.descriptors import NATIVE_CONTRACT_DESCRIPTOR

    window = openai._select_backend_context_window(
        NATIVE_CONTRACT_DESCRIPTOR,
        model_max=MODEL_MAX,
        requested=None,
        machine_fit=openai._machine_fit_for_default_window(fit),
    )
    return caps, fit, window


@pytest.fixture(autouse=True)
def _no_operator_caps(monkeypatch):
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)


# --- PX.3(a): "does not fit" serves the floor window, never the model max ---


@pytest.mark.parametrize("ram_gb", [16, 24])
def test_27b_that_does_not_fit_serves_the_floor_window(ram_gb):
    _caps, fit, window = _seat(DENSE_27B, ram_gb)
    assert fit.available and not fit.model_fits
    assert window == 4_096
    assert window != MODEL_MAX


def test_machine_fit_is_zero_only_when_the_plan_is_unavailable():
    unavailable = plan_memory(total_ram_bytes=None, model_weights_bytes=1)
    assert openai._machine_fit_for_default_window(unavailable) == 0
    assert openai._machine_fit_for_default_window(None) == 0


def test_explicit_window_and_allow_swap_still_win_over_the_floor():
    from mtplx.backends.descriptors import NATIVE_CONTRACT_DESCRIPTOR

    _caps, fit, _window = _seat(DENSE_27B, 24)
    floor = openai._machine_fit_for_default_window(fit)
    assert floor == 4_096
    # Explicit --context-window always wins (the plan warns, never refuses).
    assert (
        openai._select_backend_context_window(
            NATIVE_CONTRACT_DESCRIPTOR,
            model_max=MODEL_MAX,
            requested=32_768,
            machine_fit=floor,
        )
        == 32_768
    )
    # --allow-swap passes machine_fit=0 and keeps the model maximum.
    assert (
        openai._select_backend_context_window(
            NATIVE_CONTRACT_DESCRIPTOR,
            model_max=MODEL_MAX,
            requested=None,
            machine_fit=0,
        )
        == MODEL_MAX
    )


@pytest.mark.parametrize(
    ("ram_gb", "expected"),
    [(36, 57_344), (48, 204_800), (64, MODEL_MAX), (96, MODEL_MAX), (128, MODEL_MAX)],
)
def test_27b_seats_that_fit_are_unchanged(ram_gb, expected):
    _caps, fit, window = _seat(DENSE_27B, ram_gb)
    assert fit.model_fits
    assert window == expected
