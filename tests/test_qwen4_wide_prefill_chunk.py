"""Flash-Next's own prefill width (2026-09-18).

The speed lane stamps a 4,096-row prefill chunk and a 16,384-token sparse
attention crossover on tensor-unit GPUs only; the wide chunk is granted per
request against live memory and a refusal is the 2,048-row plan that ships
today.  An operator's explicit chunk or crossover always wins.
"""

from __future__ import annotations

import argparse
import json

import pytest

from mtplx import generation
from mtplx.models import qwen4_exp
from mtplx.server import openai as oa
from tests.test_env_flag_parsing import (
    _FLASH_NEXT_LANE_KEYS,
    _flash_next_fixed_m4_config,
    _flash_next_quantization,
)

PREFILL_KEYS = (
    "MTPLX_QWEN4_PREFILL_WIDE_CHUNK",
    "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",
)
GIB = 2**30


def _lane_args(tmp_path, monkeypatch):
    for key in (
        *_FLASH_NEXT_LANE_KEYS,
        *PREFILL_KEYS,
        "MTPLX_QSA_GATHER",
        "MTPLX_FUSED_GATE_UP",
    ):
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / "flash-next"
    model.mkdir()
    config = _flash_next_fixed_m4_config()
    config["quantization"] = _flash_next_quantization(lm_head_bits=8, stage3=True)
    (model / "config.json").write_text(json.dumps(config))
    return argparse.Namespace(
        model=str(model),
        verify_strategy="batched",
        generation_mode="mtp",
        scheduler_mode="serial",
    )


def test_prefill_width_is_stamped_on_tensor_unit_gpus(tmp_path, monkeypatch):
    from mtplx.profiles import normalize_runtime_env_overrides

    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: True)
    overrides = oa._server_runtime_env_overrides(args, {})
    assert overrides["MTPLX_QWEN4_PREFILL_WIDE_CHUNK"] == "4096"
    assert overrides["MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT"] == "16384"
    # The general crossovers are not touched: a warm turn's short suffix
    # keeps the 32,768 its own A/B chose.
    assert "MTPLX_QSA_PREFILL_MIN_CONTEXT" not in overrides
    assert "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT" not in overrides
    assert normalize_runtime_env_overrides(overrides) == overrides


def test_other_gpus_keep_the_portable_prefill_values(tmp_path, monkeypatch):
    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: False)
    overrides = oa._server_runtime_env_overrides(args, {})
    for key in PREFILL_KEYS:
        assert key not in overrides, key


@pytest.mark.parametrize("key", PREFILL_KEYS)
def test_an_operator_export_wins_over_the_stamp(tmp_path, monkeypatch, key):
    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: True)
    monkeypatch.setenv(key, "0")
    overrides = oa._server_runtime_env_overrides(args, {})
    assert key not in overrides


def test_gpu_detector_follows_the_family_fallback_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert oa._qwen4_tensor_unit_gpu() is False


def _memory(monkeypatch, *, limit, live, per_token=28_416, released=0):
    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    state = {"live": live}

    def release():
        state["live"] -= released
        return released

    monkeypatch.setattr(generation, "_metal_memory_limit_bytes", lambda rt: limit)
    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: state["live"])
    monkeypatch.setattr(
        generation, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: per_token
    )
    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", release)


def test_the_wide_chunk_is_off_until_the_lane_names_a_width(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", raising=False)
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_the_wide_chunk_is_granted_while_memory_allows(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(
            None, prompt_tokens=65536, receipt=receipt
        )
        == 4096
    )
    assert receipt["granted"] is True
    assert receipt["need_bytes"] == 65536 * 28_416 + int(3.0 * GIB)
    assert receipt["threshold_bytes"] == int(110 * GIB * 0.90)


def test_a_tight_machine_keeps_the_2048_row_plan(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=96 * GIB, live=85 * GIB)
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(
            None, prompt_tokens=131072, receipt=receipt
        )
        is None
    )
    assert receipt["granted"] is False


def test_the_allocator_cache_is_released_before_a_refusal(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=97 * GIB, released=10 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=16384) == 4096


def test_short_prompts_and_pinned_chunks_are_left_alone(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=2048) is None
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_the_profile_stamped_chunk_knobs_are_not_an_operator_pin(monkeypatch):
    """Every profile exports auto / 2048 / 2048; only a moved knob is a pin."""

    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) == 4096
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "1024")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_an_unknown_memory_limit_does_not_refuse(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=0, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) == 4096


def test_the_compiled_selector_accepts_the_canonical_and_the_wide_width(monkeypatch):
    monkeypatch.delenv("MTPLX_QSA_PREFILL_COMPILE_ROWS", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", raising=False)
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048,)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048, 4096)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "junk")
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048,)


def test_the_lowered_crossover_applies_to_wide_forwards_only(monkeypatch):
    for key in (
        "MTPLX_QSA_PREFILL_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",
    ):
        monkeypatch.delenv(key, raising=False)
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 32768
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "16384")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2048, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2047, 32768) == 32768
    assert qwen4_exp._qsa_prefill_crossover(64, 32768) == 32768
    # It can only lower the crossover, never raise an operator's lower one.
    assert qwen4_exp._qsa_prefill_crossover(4096, 8192) == 8192
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "0")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 32768
