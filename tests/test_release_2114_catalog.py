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

MATRIX = json.loads((Path(__file__).parent / "fixtures/release_2114_recommendations.json").read_text())


@pytest.mark.parametrize("row", MATRIX, ids=lambda r: f"{r['tier']}-{r['ram_gib']}")
def test_app_cli_ram_matrix(row, monkeypatch):
    from mtplx import default_models as defaults
    from mtplx.model_catalog import recommended_catalog_ids, recommended_models
    from mtplx.ui import onboarding

    monkeypatch.setenv(defaults.QWEN38_OPTIMIZED_SPEED_MODEL_ENV, "off")
    monkeypatch.setenv(defaults.SPEED_MODEL_ENV, "off")
    monkeypatch.delenv(defaults.DEFAULT_MODEL_VARIANT_ENV, raising=False)
    monkeypatch.setattr(defaults, "_QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES", ())
    args = dict(memory_gib=row["ram_gib"], chip_tier=row["tier"])
    assert recommended_catalog_ids(**args) == row["raw"]
    assert [m.id for m in recommended_models(**args)] == row["visible"]
    hardware = dict(apple_silicon_generation="m2" if row["tier"] == "legacy" else "m5", memory_gib=row["ram_gib"])
    if row["default"] is None:
        with pytest.raises(defaults.DefaultModelUnavailable):
            defaults.select_default_model(hardware=hardware)
        return
    selection = defaults.select_default_model(hardware=hardware)
    assert catalog_model_matching(selection.hf_model).id == row["default"]
    monkeypatch.setattr(onboarding, "_verified_default_selection", lambda: selection)
    panels = []
    monkeypatch.setattr(onboarding, "_step_panel", lambda **kw: panels.extend(kw["options"]))
    monkeypatch.setattr(onboarding, "_prompt_choice", lambda *args, **kw: kw["default"])
    assert onboarding.screen_model(installed=[]) == selection.model
    offered = [title.split("  ·")[0] for _, title, _ in panels[:-2]]
    expected = [catalog_model_with_id(i).display_name for i in row["visible"]]
    # Older CLI labels omit a space in Qwen3.5; the actual catalog identities agree.
    assert [x.replace("Qwen3.5", "Qwen 3.5") for x in offered] == expected


def test_bonsai_bound_matches_swift_and_can_move_to_24(monkeypatch):
    import re
    from mtplx import model_catalog as catalog
    from mtplx.default_models import select_default_model
    swift = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift").read_text()
    bound = float(re.search(r"bonsaiRecommendationMinGiB: Double = ([0-9.]+)", swift)[1])
    assert bound == catalog.BONSAI_RECOMMENDATION_MIN_GIB == 16
    monkeypatch.setattr(catalog, "BONSAI_RECOMMENDATION_MIN_GIB", 24)
    for ram, winner in [(16, "qwen35-9b-optimized-speed"), (18, "qwen35-9b-optimized-speed"), (24, "bonsai-2-27b-optimized-speed")]:
        offered = catalog.recommended_catalog_ids(memory_gib=ram, chip_tier="modern")
        assert offered[0] == winner
        assert "bonsai-2-27b-optimized-speed" in offered
        assert catalog_model_matching(select_default_model(hardware={"apple_silicon_generation": "m5", "memory_gib": ram}).hf_model).id == winner


@pytest.mark.parametrize("ram,peak,verdict", [(256, 170.7, "tight_fit"), (256, 170.6, "recommended"), (16, 10.7, "tight_fit"), (16, 10.6, "recommended")])
def test_badge_safety_boundaries(ram, peak, verdict):
    from dataclasses import replace
    from mtplx.model_catalog import evaluate_feasibility
    model = replace(catalog_model_with_id("bonsai-2-27b-optimized-speed"), peak_memory_gib=peak)
    assert evaluate_feasibility(model, chip_tier="modern", ram_gib=ram, disk_free_gib=1000).verdict == verdict
