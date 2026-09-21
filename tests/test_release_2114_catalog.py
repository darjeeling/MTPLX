"""Release identities and the shared app/CLI recommendation contract (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.artifacts import _hf_repo_id_from_ref
from mtplx.backends.descriptors import model_family_from_inspection
from mtplx.commands.public import _model_ref_from_public_model_id
from mtplx.default_models import public_model_id_for_ref
from mtplx.model_catalog import catalog_model_matching, catalog_model_with_id

NEW_MODELS = (
    ("flash-next-optimized-quality", "qwen4_exp", 169_900_000_000, 166.2),
    ("bonsai-2-27b-optimized-speed", "qwen3_8", 8_200_000_000, 10.32),
)


@pytest.mark.parametrize("catalog_id,family,size,peak", NEW_MODELS)
def test_new_pack_identity_round_trips(catalog_id, family, size, peak, tmp_path):
    model = catalog_model_with_id(catalog_id)
    assert (model.size_bytes, model.peak_memory_gib) == (size, peak)
    public_id = f"mtplx-{catalog_id}"
    refs = [catalog_id, model.hf_model_id, *[a for a in model.aliases if " " not in a]]
    for ref in refs:
        assert catalog_model_matching(ref) == model
        assert public_model_id_for_ref(ref) == public_id
        assert _model_ref_from_public_model_id(ref) == model.hf_model_id
        assert _hf_repo_id_from_ref(ref) == model.hf_model_id
        assert model_family_from_inspection(model_ref=ref) == family
    for name in {Path(model.hf_model_id).name, *[a for a in model.aliases if "MTPLX" in a]}:
        local = tmp_path / name
        local.mkdir()
        assert catalog_model_matching(local) == model
        assert public_model_id_for_ref(local) == public_id
        assert model_family_from_inspection(model_ref=str(local)) == family
    # A derivative does not acquire a first-party served identity.
    derivative = Path(model.hf_model_id).name + "-third-party"
    assert public_model_id_for_ref(derivative) != public_id
    assert catalog_model_matching(derivative) is None


def test_legacy_bonsai_runtime_identity_and_family(tmp_path):
    pack = tmp_path / "renamed-pack"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text(json.dumps({
        "public_model_id": "mtplx-bonsai-38-27b-optimized-speed",
        "arch_id": "qwen3-next-mtp",
    }))
    assert public_model_id_for_ref(pack) == "mtplx-bonsai-2-27b-optimized-speed"
    assert model_family_from_inspection(model_ref=str(pack)) == "qwen3_8"
