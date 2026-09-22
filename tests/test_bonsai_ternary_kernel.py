"""The small-M ternary GEMV for the Bonsai 2 trunk (mtplx.kernels.bonsai_ternary).

Exactness is the law: every served row count must be bit-exact against the
stock ``mx.quantized_matmul`` path on the real Bonsai shapes and on the
synthetic two-layer pack, the install probe must serve only what it proved,
the switch must be off by default, an ineligible call must take the stock
path byte for byte, and a tiny-model generation with the kernel on and off
must produce the same tokens (fixed seed, the pack's native sampler, draft
depth 3 so every verify round has four rows). GPU tests skip without Metal.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels import bonsai_ternary as bt
from mtplx.models import prism_hadamard_qwen35 as ph
from tests import prism_hadamard_synth as synth

# The packed shapes of the real 27B pack (width, outputs): GatedDeltaNet
# in_proj_qkv / in_proj_z / out_proj, attention q / k / o, MLP gate / down,
# and the lm_head.
REAL_SHAPES = [
    (5120, 10240),
    (5120, 6144),
    (6144, 5120),
    (5120, 12288),
    (5120, 1024),
    (5120, 17408),
    (17408, 5120),
    (5120, 248320),
]

_HAS_METAL = mx.metal.is_available()
gpu = pytest.mark.skipif(not _HAS_METAL, reason="the ternary kernel is a Metal kernel")


def _packed_module(width: int, outputs: int, seed: int) -> ph.HadamardQuantizedLinear:
    """A HadamardQuantizedLinear with random ternary codes at a real shape."""

    mx.random.seed(seed)
    module = ph.HadamardQuantizedLinear(width, outputs, block=1024)
    words = mx.random.randint(0, 2**31 - 1, module.weight.shape).astype(mx.uint32)
    three = (words & 0x55555555) & ((words >> 1) & 0x55555555)
    module.weight = words ^ three  # code 3 never appears in a ternary pack
    module.scales = mx.random.uniform(0.005, 0.03, module.scales.shape).astype(
        mx.float16
    )
    module.biases = -module.scales
    mx.eval(module.parameters())
    return module


def _activation(rows: int, width: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    x = mx.random.normal((1, rows, width)) * 2.0
    spikes = mx.random.uniform(shape=(1, rows, width)) < 0.002
    return mx.where(spikes, mx.array(96.0), x).astype(mx.float16)


@pytest.fixture(autouse=True)
def _fresh_kernel_state(monkeypatch):
    monkeypatch.delenv(bt.ENV, raising=False)
    bt.reset_for_tests()
    yield
    bt.reset_for_tests()


# -- the switch and the contract (no GPU needed) ------------------------------


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(bt.ENV, raising=False)
    assert not bt.enabled()
    monkeypatch.setenv(bt.ENV, "1")
    assert bt.enabled()
    monkeypatch.setenv(bt.ENV, "0")
    assert not bt.enabled()


def test_shape_contract_refuses_what_the_kernel_does_not_take():
    module = _packed_module(1024, 128, 1)
    ok = _activation(4, 1024, 2)
    assert bt.shape_eligible(ok, module)
    assert not bt.shape_eligible(ok.astype(mx.float32), module)  # fp16 activations only
    assert not bt.shape_eligible(
        _activation(4, 1000, 3)[..., :1000], module
    )  # width % 128
    assert not bt.shape_eligible(
        mx.zeros((2, 4, 1024), dtype=mx.float16), module
    )  # one batch entry
    module.bits = 4
    assert not bt.shape_eligible(ok, module)  # 2-bit words only
    module.bits = 2
    module.group_size = 64
    assert not bt.shape_eligible(ok, module)  # group 128 only


def test_unarmed_module_is_never_served():
    module = _packed_module(1024, 128, 4)
    x = _activation(4, 1024, 5)
    assert not bt.serves(x, module)
    assert module._ternary is None and module._ternary_rows is None


# -- parity against stock at every served row count ----------------------------


@gpu
@pytest.mark.parametrize("width,outputs", REAL_SHAPES)
def test_real_shapes_are_bit_exact_at_every_candidate_row_count(width, outputs):
    module = _packed_module(width, outputs, 11)
    for rows in range(bt.MIN_ROWS, bt.MAX_ROWS + 1):
        for seed in (1, 2):
            x = _activation(rows, width, 100 * rows + seed)
            want = bt.stock_matmul(x, module)
            got = bt._launch(x, module)
            assert got.shape == want.shape and got.dtype == want.dtype
            assert mx.array_equal(want, got).item(), (
                f"{width}x{outputs} M={rows} seed={seed} is not bit-exact with stock"
            )


@gpu
def test_install_probe_serves_only_proven_row_counts_and_arms_the_modules():
    modules = [
        _packed_module(1024, 2048, 21),
        _packed_module(2048, 1024, 22),
        _packed_module(1024, 384, 23),
    ]
    probe = bt.install(modules)
    assert bt.installed(), probe
    assert probe["served_rows"] == list(range(bt.MIN_ROWS, bt.MAX_ROWS + 1)), probe
    assert probe["misses"] == []
    assert probe["cells"] == len(modules) * (bt.MAX_ROWS - bt.MIN_ROWS + 1) * 2
    for module in modules:
        assert module._ternary_rows == frozenset(probe["served_rows"])
    report = bt.engagement()
    assert report["installed"] and report["probe_misses"] == 0
    assert report["served_rows"] == probe["served_rows"]


@gpu
def test_a_probe_miss_removes_the_row_count_and_a_full_miss_disables(monkeypatch):
    from mtplx import demotions

    module = _packed_module(1024, 512, 31)
    real_launch = bt._launch

    def poisoned(x, mod):  # a kernel that is wrong for six rows only
        y = real_launch(x, mod)
        return y + mx.array(1.0, dtype=y.dtype) if int(x.shape[-2]) == 6 else y

    monkeypatch.setattr(bt, "_launch", poisoned)
    probe = bt.install([module])
    assert bt.installed()
    assert 6 not in probe["served_rows"] and 4 in probe["served_rows"]
    assert any("M=6" in miss for miss in probe["misses"])

    bt.reset_for_tests()
    demotions.reset()
    monkeypatch.setattr(
        bt, "_launch", lambda x, mod: real_launch(x, mod) + mx.array(1.0, dtype=x.dtype)
    )
    probe = bt.install([module])
    assert not bt.installed()
    assert probe["served_rows"] == [] and "no row count" in (bt.disabled_reason() or "")
    assert module._ternary_rows is None
    assert demotions.counts().get(bt.DEMOTION_KIND, 0) == 1


@gpu
def test_armed_module_falls_back_to_stock_outside_its_row_counts():
    module = _packed_module(1024, 512, 41)
    bt.install([module])
    module._ternary = bt
    before = bt.counters()
    for rows in (1, 2, 3, 9):
        x = _activation(rows, 1024, 50 + rows)
        assert not bt.serves(x, module)
        assert mx.array_equal(
            module(x), bt.stock_matmul(module.rotate(x), module)
        ).item()
    after = bt.counters()
    assert after["served_calls"] == before["served_calls"]
    x = _activation(4, 1024, 60)
    assert bt.serves(x, module)
    assert mx.array_equal(module(x), bt.stock_matmul(module.rotate(x), module)).item()
    assert bt.counters()["served_calls"] == before["served_calls"] + 1


# -- through the loader on the synthetic pack -----------------------------------


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> synth.SyntheticPack:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return synth.build_synthetic_pack(tmp_path_factory.mktemp("bonsai") / "pack")
    finally:
        mx.set_default_device(previous)


def _load(path: Path):
    from mlx_lm.utils import load_model

    model, _config = load_model(
        Path(path), get_model_classes=lambda config: (ph.Model, ph.ModelArgs)
    )
    return model, model.post_weight_load(path)


@gpu
def test_loader_leaves_every_projection_stock_with_the_switch_off(pack):
    model, report = _load(pack.path)
    assert report["ternary_kernel"] == {
        "enabled": False,
        "installed": False,
        "served_rows": [],
    }
    for _record, module in model._packed_modules():
        if isinstance(module, ph.HadamardQuantizedLinear):
            assert module._ternary is None


@gpu
def test_loader_arms_the_kernel_and_the_logits_match_stock_bit_for_bit(
    pack, monkeypatch
):
    monkeypatch.setenv(bt.ENV, "1")
    model, report = _load(pack.path)
    verdict = report["ternary_kernel"]
    assert verdict["enabled"] and verdict["installed"], verdict
    assert verdict["served_rows"] == list(range(bt.MIN_ROWS, bt.MAX_ROWS + 1)), verdict
    linears = [
        m
        for _r, m in model._packed_modules()
        if isinstance(m, ph.HadamardQuantizedLinear)
    ]
    assert linears and all(m._ternary is bt for m in linears)

    ids = mx.array([[1, 5, 9, 200]])  # four rows: a depth-3 verify window
    bt.reset_for_tests()
    with_kernel = model(ids)
    mx.eval(with_kernel)
    served = bt.counters()["served_calls"]
    assert served == len(linears), (served, len(linears))
    for module in linears:
        module._ternary = None
    stock = model(ids)
    mx.eval(stock)
    assert np.array_equal(
        np.asarray(with_kernel.astype(mx.float32)), np.asarray(stock.astype(mx.float32))
    )


@gpu
def test_generation_tokens_are_identical_with_the_kernel_on_and_off(
    pack, tmp_path, monkeypatch
):
    from mtplx import runtime
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    path = tmp_path / "pack"
    shutil.copytree(pack.path, path)
    synth.write_synthetic_mtp_sidecar(synth.SyntheticPack(path, {}, {}, {}, []))
    sampler = SamplerConfig(
        temperature=1.0, top_p=0.95, top_k=20
    )  # the pack's native sampler
    prompt = [1, 5, 9, 200, 17, 33]

    def run(enabled: bool) -> tuple[list[int], int]:
        monkeypatch.setenv(bt.ENV, "1" if enabled else "0")
        bt.reset_for_tests()
        rt = runtime.load(path, mtp=True)
        assert rt.mtp_enabled
        verdict = rt.model._prism_post_load_report["ternary_kernel"]
        assert verdict["installed"] is enabled, verdict
        out = generate_mtpk(
            rt,
            prompt,
            max_tokens=24,
            sampler=sampler,
            speculative_depth=3,
            seed=7,
            stop_token_ids=set(),
        )
        return list(out.tokens), bt.counters()["served_calls"]

    off_tokens, off_served = run(False)
    on_tokens, on_served = run(True)
    assert off_served == 0
    assert on_served > 0, "the kernel never engaged on a depth-3 verify"
    assert on_tokens == off_tokens, (on_tokens, off_tokens)
    assert len(on_tokens) == 24
