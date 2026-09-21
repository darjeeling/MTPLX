#!/usr/bin/env python3
"""Assemble Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed from Prism ML's pack.

The MTPLX pack is Prism ML's Ternary Bonsai 2 27B (Apache-2.0) plus a
Qwen3.8-27B MTP draft head:

* ``model.safetensors`` is Prism's file, hard linked (or copied) byte for byte.
  It holds the rotated ternary language model AND the 333 vision tower tensors;
  its sha256 is checked against the value Prism publishes in ``files.json`` and
  recorded. Nothing is re-quantized, re-saved or split.
* ``model.safetensors.index.json`` is generated from the file's header so every
  tool that reads an index (the vision resolver, inspect) sees all tensors.
* ``mtp.safetensors`` is the float16 draft head of a Qwen3.8-27B MTPLX pack.
  The trunk runs in float16 on every chip, so a bfloat16 head is cast once
  here (a head whose floats are already float16 is linked byte for byte).
* ``config.json`` is Prism's config with the MTP layer declared and MTPLX's
  head contract added. The rotation metadata (``modules``, ``hadamard.json``)
  and the vision keys are carried unchanged.
* ``mtplx_runtime.json``, ``README.md``, ``LICENSE``, ``NOTICE.txt`` and
  ``MTPLX_PACK_MANIFEST.json`` (sha256 of every file, provenance) are written.

Vision is mandatory. A source without its vision tensors, ``vision_config`` or
``preprocessor_config.json`` is refused; the built pack is re-checked.

Nothing is ever deleted: an existing output folder is refused unless
``--move-existing-aside`` is given, which renames it into ``_aside/``.
This script only reads safetensors headers and copies files. The one tensor
operation (casting a bfloat16 head to float16) runs on the CPU device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

PACK_NAME = "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
PUBLIC_MODEL_ID = "mtplx-bonsai-2-27b-optimized-speed"
HF_REPO = "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
MODEL_FAMILY = "qwen3_8"
MIN_ENGINE_VERSION = "2.11.4"
SOURCE_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
BASE_TRUNK = "Qwen/Qwen3.8-27B"
MODEL_TYPE = "prism_hadamard_qwen35"
VISION_PREFIX = "vision_tower."
MODELS_DIR = Path.home() / ".mtplx" / "models"
DEFAULT_SOURCE = MODELS_DIR / "prism-ml--Ternary-Bonsai-2-27B-mlx-2bit"
DEFAULT_MTP_SOURCES = (
    MODELS_DIR / "Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    MODELS_DIR / "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    MODELS_DIR / "Qwen3.8-27B-MTPLX-Optimized-Speed",
    MODELS_DIR / "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
)
# Carried from Prism's pack unchanged.
CARRIED_FILES = (
    "LICENSE",
    "NOTICE.txt",
    "hadamard.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
REQUIRED_SOURCE_FILES = (
    "config.json",
    "model.safetensors",
    "hadamard.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "LICENSE",
    "NOTICE.txt",
)
# Official Qwen3.8 processor sidecars, carried from the MTP source pack when
# it has them (clients that build a processor from the folder read these).
OPTIONAL_PROCESSOR_FILES = ("processor_config.json", "video_preprocessor_config.json")
SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


class PackBuildError(RuntimeError):
    pass


def sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise PackBuildError(f"{path} is not a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > 256 * 1024 * 1024:
            raise PackBuildError(f"{path} has an implausible header length {length}")
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def check_source(source: Path) -> dict[str, Any]:
    """Validate Prism's pack, including everything the vision tower needs."""

    missing = [name for name in REQUIRED_SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise PackBuildError(f"{source} is missing {', '.join(missing)}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != MODEL_TYPE:
        raise PackBuildError(
            f"{source} declares model_type {config.get('model_type')!r}, not {MODEL_TYPE!r}"
        )
    if not isinstance(config.get("modules"), list) or not config["modules"]:
        raise PackBuildError(f"{source}/config.json carries no rotated module list")
    components = config.get("components") if isinstance(config.get("components"), dict) else {}
    if not components.get("vision") or not isinstance(config.get("vision_config"), dict):
        raise PackBuildError(
            f"{source} does not declare its vision tower. Bonsai ships as a "
            "vision-language model; a text-only pack is never built."
        )
    header = safetensors_header(source / "model.safetensors")
    vision = [key for key in header if key.startswith(VISION_PREFIX)]
    if not vision:
        raise PackBuildError(
            f"{source}/model.safetensors carries no {VISION_PREFIX}* tensors. "
            "Bonsai ships as a vision-language model; a text-only pack is never built."
        )
    return {"config": config, "header": header, "vision_tensors": len(vision)}


def resolve_mtp_source(explicit: str | None) -> Path:
    candidates = [Path(explicit).expanduser()] if explicit else list(DEFAULT_MTP_SOURCES)
    for candidate in candidates:
        path = candidate / "mtp.safetensors" if candidate.is_dir() else candidate
        if path.is_file():
            return path
    raise PackBuildError(
        "no Qwen3.8-27B MTP head found; pass --mtp-source <pack folder or mtp.safetensors>"
    )


def head_float_dtypes(path: Path) -> set[str]:
    return {
        str(entry.get("dtype"))
        for entry in safetensors_header(path).values()
        if str(entry.get("dtype")) in {"F16", "BF16", "F32", "F64"}
    }


def place_file(source: Path, target: Path, *, mode: str) -> str:
    """Hard link when asked and possible, else copy. Returns what happened."""

    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def write_float16_head(source: Path, target: Path, *, mode: str) -> dict[str, Any]:
    dtypes = head_float_dtypes(source)
    if dtypes <= {"F16"}:
        return {"action": place_file(source, target, mode=mode), "cast_from": None}
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        tensors = dict(mx.load(str(source)))
        lost = 0
        for name, value in list(tensors.items()):
            if not mx.issubdtype(value.dtype, mx.floating) or value.dtype == mx.float16:
                continue
            cast = value.astype(mx.float16)
            finite = mx.all(mx.isfinite(cast)).item()
            if not finite:
                raise PackBuildError(f"{name} overflows float16; this head cannot be used")
            lost += int((cast.astype(mx.float32) != value.astype(mx.float32)).sum().item())
            tensors[name] = cast
        mx.eval(tensors)
        mx.save_safetensors(str(target), tensors, metadata={"format": "mlx"})
    finally:
        mx.set_default_device(previous)
    return {
        "action": "cast_to_float16",
        "cast_from": sorted(dtypes),
        "elements_changed_by_cast": lost,
    }


def build_config(source_config: dict[str, Any], mtp_pack_config: dict[str, Any] | None) -> dict[str, Any]:
    config = json.loads(json.dumps(source_config))
    text = config.setdefault("text_config", {})
    text["mtp_num_hidden_layers"] = 1
    text["mtp_use_dedicated_embeddings"] = False
    config.setdefault("components", {})["mtp"] = True
    config["mlx_lm_extra_tensors"] = {"mtp_file": "mtp.safetensors"}
    contract = (mtp_pack_config or {}).get("mtplx_mtp_contract") or {
        "base_hidden_variant": "post_norm",
        "concat_order": "embedding_hidden",
        "hidden_variant": "post_norm",
        "mtp_position_mode": "local",
        "mtp_quant_group_size": 64,
        "mtp_quant_mode": "affine",
    }
    config["mtplx_mtp_contract"] = contract
    return config


def mtp_quantization_from_header(path: Path) -> dict[str, Any] | None:
    """The head's own packing, read from its tensors, never from the trunk.

    The trunk is 2-bit group 128. A head contract that inherited those numbers
    would unpack a 4-bit head as 2-bit without any error, so the contract
    written into the pack always states the head's bits and group size.
    """

    header = safetensors_header(path)
    weight = header.get("mtp.fc.weight")
    scales = header.get("mtp.fc.scales")
    if not weight or not scales or weight.get("dtype") != "U32":
        return None
    out_dims, packed = weight["shape"]
    _rows, groups = scales["shape"]
    fc_inputs = None
    for key, entry in header.items():
        if key.endswith("pre_fc_norm_hidden.weight"):
            fc_inputs = 2 * int(entry["shape"][0])
    if not fc_inputs:
        return None
    bits = packed * 32 // fc_inputs
    group_size = fc_inputs // groups
    del out_dims
    return {
        "bits": int(bits),
        "group_size": int(group_size),
        "mode": "affine",
        "policy": "all",
        "prequantized": True,
    }


def build_runtime_contract(
    *, mtplx_version: str, provenance: dict[str, Any], head: dict[str, Any] | None
) -> dict[str, Any]:
    mtp_contract: dict[str, Any] = {
        "base_hidden_variant": "post_norm",
        "concat_order": "embedding_hidden",
        "hidden_variant": "post_norm",
        "mtp_position_mode": "local",
    }
    if head:
        mtp_contract.update(
            mtp_quant_bits=head["bits"],
            mtp_quant_group_size=head["group_size"],
            mtp_quant_mode=head["mode"],
            mtp_quant_policy=head["policy"],
            mtp_prequantized=True,
        )
    return {
        "arch_id": "qwen3-next-mtp",
        "artifact_role": "mtplx-pack",
        "base_trunk": BASE_TRUNK,
        "source_repo": SOURCE_REPO,
        "public_model_id": PUBLIC_MODEL_ID,
        "model_family": MODEL_FAMILY,
        "min_engine_version": MIN_ENGINE_VERSION,
        "mtplx_version": mtplx_version,
        "precision_variant": "fp16",
        "precision_policy": {
            "variant": "fp16",
            "note": (
                "The pack is float16 scaled. Every chip runs it in float16; "
                "there is no bfloat16 tensor in it, so M1 and M2 need no sibling."
            ),
        },
        "capabilities": {"vision": True, "mtp": True},
        "sampler": dict(SAMPLER),
        "recommended_draft_sampler": dict(SAMPLER),
        "recommended_profile": "turbo",
        # Placeholder. The measured acceptance by depth decides the default;
        # the main GPU session fills mtp_depth_default and speed_evidence in.
        "mtp_depth_default": 2,
        "mtp_depth_default_status": "placeholder_until_measured",
        "mtp_depth_max": 3,
        "mtp_sidecar": (
            f"int{head['bits']}-g{head['group_size']}-prequantized" if head else "float16"
        ),
        "mtp_contract": mtp_contract,
        "exactness_baseline": {
            "reference": "Prism ML bundled runtime (runtime/vision_artifact.py, mlx-vlm 0.6.3)",
            "method": "teacher-forced logits, 3 prompts x 256 positions, plus one image prompt",
            "status": "pending_measurement",
        },
        "verified_on": {},
        "forge_provenance": provenance,
    }


def build_index(header: dict[str, Any], total_size: int) -> dict[str, Any]:
    return {
        "metadata": {"total_size": int(total_size), "format": "mlx"},
        "weight_map": {key: "model.safetensors" for key in sorted(header)},
    }


def render_card(*, source_sha: str, head_note: str) -> str:
    return CARD_TEMPLATE.format(
        source_sha=source_sha, head_note=head_note, hf_repo=HF_REPO,
        min_engine_version=MIN_ENGINE_VERSION,
    )


CARD_TEMPLATE = """---
license: apache-2.0
library_name: mtplx
pipeline_tag: image-text-to-text
base_model:
- prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
- Qwen/Qwen3.8-27B
tags:
- mtplx
- mlx
- apple-silicon
- macos
- vision
- ternary
- 2-bit
- speculative-decoding
- multi-token-prediction
- qwen3.8
- bonsai
- prismml
- coding
---

# Bonsai 2 27B MTPLX Optimized Speed

A compact 27B-class vision-language model for Macs, with a draft head for speculative decoding. This pack is for [MTPLX](https://mtplx.com).

**Full credit for the model goes to [Prism ML](https://prismml.com).** The
language model and the vision tower in this pack are Prism ML's
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit),
carried byte for byte. Created using Bonsai by Prism ML. Bonsai 2 27B is built
from [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Alibaba Cloud.
Both are licensed under Apache 2.0. Read Prism ML's model card and whitepaper
for the method, the benchmarks and the limitations of the model itself.

## What is in this pack

| File | What it is |
| :--- | :--- |
| `model.safetensors` | Prism ML's file, unchanged (sha256 `{source_sha}`). It holds the ternary language model and the float16 vision tower. |
| `mtp.safetensors` | The multi-token prediction draft head of Qwen3.8-27B, from the MTPLX Qwen 3.8 27B pack. {head_note} |
| `hadamard.json`, `config.json` | Prism ML's rotation metadata, unchanged, plus the MTPLX head contract in `config.json`. |
| `mtplx_runtime.json` | Sampler defaults and the MTPLX runtime contract. |
| `MTPLX_PACK_MANIFEST.json` | The sha256 of every file and where each one came from. |

This is a vision-language model. Image input works in MTPLX the same way it
does for the other Qwen 3.8 27B packs.

## What MTPLX adds

Prism ML stores every projection of the language model as ternary weights in a
rotated basis. A runtime has to apply a matching Hadamard transform to the
activations, or the output is wrong. MTPLX loads this model type natively,
keeps the weights packed at 2 bits, and refuses to load a pack whose rotation
metadata or vision tower is missing.

The draft head lets MTPLX propose several tokens per step and verify them with
the full model. Verification uses exact speculative sampling, so the output
distribution is the same as decoding one token at a time. The head was trained
on the original Qwen3.8-27B, not on the ternary model, so it agrees with Bonsai
less often than with the model it was trained on. If the head does not make
decoding faster on your Mac, turn speculative decoding off in MTPLX. The output
distribution is the same either way.

## Speed

<!-- MTPLX maintainers: fill this table from the measured runs before publishing. -->

| Mac | Context | Decode without the head | Decode with the head | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: |
| to be measured | | | | |

## How to run it

```bash
pip install mtplx
mtplx start
```

Pick Bonsai 2 27B in the model list, or pass
`--model {hf_repo}` to `mtplx serve`.
The macOS app lists it in the model picker. This pack requires MTPLX
{min_engine_version} or newer.

## Recommended settings

These are Prism ML's and Qwen's recommendations, and MTPLX applies them by
default.

| Mode | temperature | top_p | top_k | presence penalty |
| :--- | ---: | ---: | ---: | ---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

The pack uses the `qwen3_8` behavior family. MTPLX supplies the reasoning
defaults from that runtime policy; explicit user settings take precedence.

## Memory

The pack manifest records the actual file sizes. Catalog memory figures are
provisional until the measured memory table is available. The context cache
grows with the conversation; MTPLX plans context from the artifact and the
memory available on your Mac.

## License and attribution

Apache 2.0. `LICENSE` and `NOTICE.txt` are carried from Prism ML's pack.
This pack contains Prism ML's Ternary Bonsai 2 27B (Copyright 2026 Prism ML,
Inc.), which is built from Qwen3.8-27B (Copyright 2026 Alibaba Cloud), and the
Qwen3.8-27B draft head as packed for MTPLX.
"""


def _mtplx_version() -> str:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from mtplx.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def build_pack(
    source: Path,
    output: Path,
    *,
    mtp_source: Path,
    link_mode: str = "hardlink",
    move_existing_aside: bool = False,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser()
    checked = check_source(source)

    source_sha = None
    published = None
    files_json = source / "files.json"
    if files_json.is_file():
        published = (
            json.loads(files_json.read_text(encoding="utf-8"))
            .get("model.safetensors", {})
            .get("sha256")
        )
    if verify_source_hash:
        source_sha = sha256_file(source / "model.safetensors")
        if published and source_sha != published:
            raise PackBuildError(
                "model.safetensors does not match the sha256 Prism ML publishes in "
                f"files.json ({source_sha} != {published}); download it again"
            )

    if output.exists():
        if not move_existing_aside:
            raise PackBuildError(
                f"{output} already exists. Nothing is deleted; pass "
                "--move-existing-aside to rename it into _aside/."
            )
        aside = output.parent / "_aside"
        aside.mkdir(parents=True, exist_ok=True)
        output.rename(aside / f"{output.name}-{time.strftime('%Y%m%dT%H%M%S')}")
    output.mkdir(parents=True)

    actions: dict[str, str] = {}
    actions["model.safetensors"] = place_file(
        source / "model.safetensors", output / "model.safetensors", mode=link_mode
    )
    for name in CARRIED_FILES:
        if (source / name).is_file():
            actions[name] = place_file(source / name, output / name, mode="copy")
    mtp_pack_dir = mtp_source.parent
    for name in OPTIONAL_PROCESSOR_FILES:
        if (mtp_pack_dir / name).is_file():
            actions[name] = place_file(mtp_pack_dir / name, output / name, mode="copy")

    head_report = write_float16_head(mtp_source, output / "mtp.safetensors", mode=link_mode)
    if head_float_dtypes(output / "mtp.safetensors") - {"F16"}:
        raise PackBuildError("the draft head still carries non-float16 tensors")
    head_quant = mtp_quantization_from_header(output / "mtp.safetensors")

    mtp_pack_config = None
    if (mtp_pack_dir / "config.json").is_file():
        mtp_pack_config = json.loads((mtp_pack_dir / "config.json").read_text(encoding="utf-8"))
    config = build_config(checked["config"], mtp_pack_config)
    if head_quant:
        config["mtplx_mtp_quantization"] = dict(head_quant)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    header = checked["header"]
    total_size = (source / "model.safetensors").stat().st_size
    (output / "model.safetensors.index.json").write_text(
        json.dumps(build_index(header, total_size), indent=2) + "\n", encoding="utf-8"
    )

    provenance = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "scripts/build_bonsai_mtplx_pack.py",
        "source_repo": SOURCE_REPO,
        "source_model_sha256": source_sha,
        "source_model_sha256_published_by_prism": published,
        "source_model_bytes": total_size,
        "trunk_tensors": len(header) - checked["vision_tensors"],
        "vision_tensors": checked["vision_tensors"],
        "mtp_head_from": mtp_pack_dir.name,
        "mtp_head": head_report,
        "requantized": False,
    }
    runtime_contract = build_runtime_contract(
        mtplx_version=_mtplx_version(), provenance=provenance, head=head_quant
    )
    (output / "mtplx_runtime.json").write_text(
        json.dumps(runtime_contract, indent=2) + "\n", encoding="utf-8"
    )
    head_note = (
        "Stored in float16 to match the model."
        if head_report["action"] != "cast_to_float16"
        else "Cast once from bfloat16 to float16 to match the model."
    )
    (output / "README.md").write_text(
        render_card(source_sha=source_sha or published or "see the manifest", head_note=head_note),
        encoding="utf-8",
    )

    verify_built_pack(output)

    manifest_files = {}
    for path in sorted(p for p in output.iterdir() if p.is_file()):
        if path.name == "model.safetensors" and source_sha:
            digest = source_sha  # same bytes (hard link or verified copy below)
            if actions["model.safetensors"] == "copy":
                digest = sha256_file(path)
                if digest != source_sha:
                    raise PackBuildError("the copied model.safetensors differs from the source")
        else:
            digest = sha256_file(path)
        manifest_files[path.name] = {
            "sha256": digest,
            "size": path.stat().st_size,
            "placed_by": actions.get(path.name, "written"),
        }
    manifest = {
        "pack": PACK_NAME, "public_model_id": PUBLIC_MODEL_ID,
        "model_family": MODEL_FAMILY, "min_engine_version": MIN_ENGINE_VERSION,
        "provenance": provenance, "files": manifest_files,
    }
    (output / "MTPLX_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_built_pack(output: Path) -> None:
    """The built pack must still be a complete vision-language pack."""

    check_source_like = ("config.json", "model.safetensors", "hadamard.json",
                         "preprocessor_config.json", "tokenizer.json", "LICENSE",
                         "NOTICE.txt", "mtp.safetensors", "mtplx_runtime.json",
                         "model.safetensors.index.json")
    missing = [name for name in check_source_like if not (output / name).is_file()]
    if missing:
        raise PackBuildError(f"built pack is missing {', '.join(missing)}")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if not config.get("components", {}).get("vision") or not isinstance(
        config.get("vision_config"), dict
    ):
        raise PackBuildError("built pack lost its vision declaration")
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if not any(key.startswith(VISION_PREFIX) for key in index.get("weight_map", {})):
        raise PackBuildError("built pack index carries no vision tower tensors")


def stamp_pack(
    pack: Path,
    *,
    exactness: dict[str, Any] | None = None,
    exactness_status: str | None = None,
    mtp_depth_default: int | None = None,
    speed_evidence: dict[str, Any] | None = None,
    verified_on: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record measured results in mtplx_runtime.json after the GPU runs.

    The builder writes ``exactness_baseline.status = pending_measurement``,
    which keeps the pack on the unverified tier on purpose. The session that
    measures parity and the draft head stamps the numbers here. The previous
    contract is moved into ``../_aside/``; nothing is deleted.
    """

    pack = pack.expanduser()
    contract_path = pack / "mtplx_runtime.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if exactness is not None or exactness_status is not None:
        baseline = dict(contract.get("exactness_baseline") or {})
        if exactness is not None:
            baseline["measured"] = exactness
        if exactness_status is not None:
            baseline["status"] = exactness_status
        contract["exactness_baseline"] = baseline
    if mtp_depth_default is not None:
        if not 1 <= int(mtp_depth_default) <= int(contract.get("mtp_depth_max", 3)):
            raise PackBuildError("mtp_depth_default is outside 1..mtp_depth_max")
        contract["mtp_depth_default"] = int(mtp_depth_default)
        contract.pop("mtp_depth_default_status", None)
    if speed_evidence is not None:
        contract["speed_evidence"] = speed_evidence
    if verified_on is not None:
        contract["verified_on"] = verified_on
    aside = pack.parent / "_aside"
    aside.mkdir(parents=True, exist_ok=True)
    contract_path.rename(
        aside / f"{pack.name}-mtplx_runtime-{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    manifest_path = pack / "MTPLX_PACK_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("files", {})["mtplx_runtime.json"] = {
            "sha256": sha256_file(contract_path),
            "size": contract_path.stat().st_size,
            "placed_by": "stamped",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return contract


def _read_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    text = Path(value).expanduser().read_text(encoding="utf-8").strip()
    # Cell scripts print one JSON line last; accept a whole log file.
    return json.loads(text.splitlines()[-1]) if "\n" in text else json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stamp",
        default=None,
        metavar="PACK",
        help="record measured results in an already built pack instead of building",
    )
    parser.add_argument("--exactness-json", default=None)
    parser.add_argument("--exactness-status", default=None, help="for example: passed")
    parser.add_argument("--mtp-depth-default", type=int, default=None)
    parser.add_argument("--speed-evidence-json", default=None)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(MODELS_DIR / PACK_NAME))
    parser.add_argument("--mtp-source", default=None)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--move-existing-aside", action="store_true")
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="do not hash the 8.6 GB source file (the manifest then has no trunk sha256)",
    )
    args = parser.parse_args(argv)
    if args.stamp:
        try:
            contract = stamp_pack(
                Path(args.stamp),
                exactness=_read_json_arg(args.exactness_json),
                exactness_status=args.exactness_status,
                mtp_depth_default=args.mtp_depth_default,
                speed_evidence=_read_json_arg(args.speed_evidence_json),
                verified_on={
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "hardware": __import__("platform").platform(),
                },
            )
        except (PackBuildError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"stamped": args.stamp, "contract": contract}, sort_keys=True))
        return 0
    try:
        manifest = build_pack(
            Path(args.source),
            Path(args.output),
            mtp_source=resolve_mtp_source(args.mtp_source),
            link_mode=args.link_mode,
            move_existing_aside=args.move_existing_aside,
            verify_source_hash=not args.skip_source_hash,
        )
    except PackBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "pack": str(Path(args.output).expanduser()),
                "files": {k: v["placed_by"] for k, v in manifest["files"].items()},
                "source_model_sha256": manifest["provenance"]["source_model_sha256"],
                "mtp_head": manifest["provenance"]["mtp_head"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
