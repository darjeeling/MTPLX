"""The Flash-Next Forge lane writes files the runtime's own loaders read back.

PR #508's unit tests cover the arithmetic of the lane. These cover the two
places where a converter and a runtime can disagree without anyone noticing:

* the n-gram table. The lane streams it shard by shard into a hand-written
  safetensors file; the runtime gathers rows from that file with its own
  reader. A layout mismatch reads garbage rows, not an error.
* the draft head. A raw checkpoint stores its norms zero-centred ((1 + w)
  convention) and its experts packed; the runtime expects absolute norms and
  split experts, and shifts two of the norms itself. A head that is shifted
  twice, or not at all, still loads, still runs, and proposes nonsense:
  acceptance near zero and decode slower than plain decoding, with no
  warning (the defect PR #511 fixed for another family).

Both are checked end to end on a tiny synthetic source, on the CPU device.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mtplx.commands import forge_qwen4_exp as lane
from mtplx.models import qwen4_exp as runtime
from mtplx.models.qwen4_exp import Model, Qwen4ExpMTP, TextArgs


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


# --------------------------------------------------------------------- n-gram
def _ngram_source(root: Path, shards: list[mx.array]) -> Path:
    source = root / "source"
    source.mkdir()
    weight_map = {}
    for index, table in enumerate(shards):
        key = lane.NGRAM_KEY.format(layer=1, index=index)
        name = f"model-{index:05d}.safetensors"
        # A second tensor in front of the table moves its data offset off zero.
        mx.save_safetensors(
            str(source / name),
            {"a.pad": mx.zeros((3,), dtype=mx.float16), key: table},
        )
        weight_map[key] = name
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"ple_layer_ids": [2]}})
    )
    return source


def test_the_streamed_ngram_table_reads_back_through_the_runtime_gather(tmp_path):
    mx.random.seed(3)
    shards = [mx.random.normal((rows, 160)).astype(mx.bfloat16) for rows in (7, 5)]
    source = _ngram_source(tmp_path, shards)

    report = lane.write_ngram_sidecar(source, tmp_path / "pack", bits=4, group=32)

    assert report["rows"] == 12 and report["dim"] == 160
    written = tmp_path / "pack" / lane.NGRAM_FILE
    assert not written.with_suffix(".partial").exists()

    # Shard by shard equals the whole table at once: quantization is per row.
    q, s, b = mx.quantize(mx.concatenate(shards, axis=0), group_size=32, bits=4)
    loaded = mx.load(str(written))
    assert bool(mx.all(loaded["ngram.weight"] == q))
    assert bool(mx.all(loaded["ngram.scales"].view(mx.uint16) == s.view(mx.uint16)))
    assert bool(mx.all(loaded["ngram.biases"].view(mx.uint16) == b.view(mx.uint16)))

    # The runtime's reader, built the way NGramTable.attach_sidecar builds it.
    header, data_start = runtime._read_safetensors_header(written)
    meta = header["__metadata__"]
    entries = {
        name: (header[f"ngram.{name}"], data_start)
        for name in ("weight", "scales", "biases")
    }
    gather = runtime._SidecarGather(
        written,
        entries,
        bits=int(meta["ngram_bits"]),
        group_size=int(meta["ngram_group_size"]),
    )
    ids = np.array([0, 6, 7, 11, 3], dtype=np.int64)  # both shards, both edges
    want = mx.dequantize(q, s, b, group_size=32, bits=4)[mx.array(ids)]
    got = gather.gather_np(ids)
    assert got.shape == (5, 160)
    assert bool(mx.all(got.astype(mx.float32) == want.astype(mx.float32)))


def test_the_production_layout_is_the_one_the_runtime_defaults_to():
    # attach_sidecar falls back to 4-bit / 32 when the metadata is absent.
    assert lane.NGRAM_PRODUCTION_LAYOUT == (4, 32)


# ----------------------------------------------------------------- draft head
def _tiny_args() -> TextArgs:
    # 2 * moe_intermediate != hidden and moe_intermediate != hidden, so the
    # packed-expert layout is decidable, as it is on the real model
    # (hidden 2560, expert width 640).
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
    )


_NORM_SUFFIXES = (
    *Model._HF_NORM_SHIFT_SUFFIXES,
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def _raw_head(args: TextArgs, *, layout: str) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """A raw checkpoint's ``mtp.`` tensors and the head the runtime must end
    up with. Norms are stored zero-centred in the raw form."""

    mx.random.seed(11)
    expected: dict[str, mx.array] = {}
    raw: dict[str, mx.array] = {}
    experts: dict[str, mx.array] = {}
    for name, value in tree_flatten(Qwen4ExpMTP(args).parameters()):
        tensor = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
        if value.ndim == 1 and any(name.endswith(s) for s in _NORM_SUFFIXES):
            raw["mtp." + name] = tensor
            expected[name] = (tensor.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            continue
        expected[name] = tensor
        if ".mlp.switch_mlp." in name:
            experts[name] = tensor
            continue
        raw["mtp." + name] = tensor
    prefix = "layers.0.mlp"
    gate = experts[f"{prefix}.switch_mlp.gate_proj.weight"]  # [E, inter, hidden]
    up = experts[f"{prefix}.switch_mlp.up_proj.weight"]
    down = experts[f"{prefix}.switch_mlp.down_proj.weight"]  # [E, hidden, inter]
    if layout == "hub":  # Linear [out, in] halves
        raw[f"mtp.{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=1)
        raw[f"mtp.{prefix}.experts.down_proj"] = down
    else:  # transformers save_pretrained: the bmm orientation
        raw[f"mtp.{prefix}.experts.gate_up_proj"] = mx.concatenate(
            [gate.swapaxes(1, 2), up.swapaxes(1, 2)], axis=-1
        )
        raw[f"mtp.{prefix}.experts.down_proj"] = down.swapaxes(1, 2)
    return raw, expected


def _head_source(root: Path, raw: dict[str, mx.array], args: TextArgs) -> Path:
    source = root / "source"
    source.mkdir()
    mx.save_safetensors(str(source / "model-00001.safetensors"), raw)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model-00001.safetensors" for key in raw}})
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"hidden_size": args.hidden_size}})
    )
    return source


def _attach(pack: Path, args: TextArgs) -> Qwen4ExpMTP:
    holder = SimpleNamespace(language_model=SimpleNamespace(args=args))
    assert Model.attach_mtp(holder, pack) is True  # strict load: every key, no extras
    return holder.language_model.mtp


@pytest.mark.parametrize("layout", ["hub", "transformers"])
def test_a_bf16_head_loads_with_absolute_norms_and_split_experts(tmp_path, layout):
    args = _tiny_args()
    raw, expected = _raw_head(args, layout=layout)
    source = _head_source(tmp_path, raw, args)

    report = lane.write_mtp_sidecar(source, tmp_path / "pack", mtp_bits=0, mtp_group=32)

    assert report["written"] is True
    head = _attach(tmp_path / "pack", args)
    loaded = dict(tree_flatten(head.parameters()))
    # "bf16" is the experts and the attention projections. The router gate,
    # the shared expert and the indexer projection are 8-bit in every recipe,
    # as they are in the trunk.
    eight_bit = {name[: -len(".scales")] for name in loaded if name.endswith(".scales")}
    assert eight_bit == {
        "layers.0.mlp.gate",
        "layers.0.mlp.shared_expert_gate",
        "layers.0.mlp.shared_expert.gate_proj",
        "layers.0.mlp.shared_expert.up_proj",
        "layers.0.mlp.shared_expert.down_proj",
        "layers.0.self_attn.indexer.index_qk_proj",
    }
    for name, want in expected.items():
        module = name[: -len(".weight")]
        if module in eight_bit:
            decoded = mx.dequantize(
                loaded[name],
                loaded[f"{module}.scales"],
                loaded[f"{module}.biases"],
                group_size=64,
                bits=8,
            )
            error = mx.max(mx.abs(decoded.astype(mx.float32) - want.astype(mx.float32)))
            # 8-bit steps over a +-0.15 range, with the scale and bias held
            # in bf16 (8 mantissa bits): about 0.0025 at worst.
            assert float(error) < 0.005, name
            continue
        got = loaded[name]
        assert got.shape == want.shape, name
        assert bool(mx.all(got.astype(mx.float32) == want.astype(mx.float32))), name
    assert {n for n in loaded if n.endswith(".weight")} == set(expected)
    # The two the runtime shifts itself were left alone by the lane, the rest
    # were shifted exactly once: every norm sits near 1, none near 0 or 2.
    for name, got in loaded.items():
        if got.ndim == 1 and any(name.endswith(s) for s in _NORM_SUFFIXES):
            assert 0.5 < float(mx.mean(got.astype(mx.float32))) < 1.5, name


def test_a_quantized_head_loads_with_the_trunk_recipe(tmp_path):
    args = _tiny_args()
    raw, expected = _raw_head(args, layout="hub")
    source = _head_source(tmp_path, raw, args)

    lane.write_mtp_sidecar(source, tmp_path / "pack", mtp_bits=4, mtp_group=32, qsa_8bit=True)

    head = _attach(tmp_path / "pack", args)
    layer = head.layers[0]
    assert (layer.mlp.switch_mlp.gate_proj.bits, layer.mlp.switch_mlp.gate_proj.group_size) == (4, 32)
    assert (layer.mlp.switch_mlp.down_proj.bits, layer.mlp.switch_mlp.down_proj.group_size) == (4, 32)
    assert (layer.mlp.gate.bits, layer.mlp.gate.group_size) == (8, 64)
    assert (layer.mlp.shared_expert.down_proj.bits, layer.mlp.shared_expert.down_proj.group_size) == (8, 64)
    assert (layer.self_attn.q_proj.bits, layer.self_attn.q_proj.group_size) == (8, 64)
    assert not hasattr(head.fc_hidden, "bits")  # structural pieces stay bf16
    # Quantized experts decode back to the source within 4-bit error.
    gate = layer.mlp.switch_mlp.gate_proj
    decoded = mx.dequantize(gate.weight, gate.scales, gate.biases, group_size=32, bits=4)
    want = expected["layers.0.mlp.switch_mlp.gate_proj.weight"].astype(mx.float32)
    assert float(mx.max(mx.abs(decoded.astype(mx.float32) - want))) < 0.02


def test_the_lanes_norm_list_is_the_models_own():
    """A copy that drifts from the model's list would write a head whose norms
    are wrong by exactly 1.0, which loads and runs."""
    assert tuple(lane.MTP_NORM_SHIFT_SUFFIXES) == tuple(Model._HF_NORM_SHIFT_SUFFIXES)


def test_the_lanes_eight_bit_set_is_inside_the_models_recipe():
    predicate = Model.quant_predicate.fget(None)
    module = SimpleNamespace(to_quantized=lambda **_: None)
    for suffix in lane.MTP_EIGHT_BIT_SUFFIXES:
        assert predicate(f"language_model.model.layers.3.{suffix}", module) == {
            "bits": 8,
            "group_size": 64,
        }, suffix
