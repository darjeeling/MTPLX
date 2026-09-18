"""``mtplx doctor --explain``: what the engine picked, why, and what it demoted.

Pure assembly and rendering. The inputs are handed in (a ``/health`` payload
when a server answers, the tensor-unit probe, and the sections other phases
register), so every line is testable without a server, a model or a GPU.
"""

from __future__ import annotations

from typing import Any, Callable

from mtplx import demotions

# Where a resolved choice came from. One vocabulary for every section.
SOURCE_DEFAULT = "default"
SOURCE_FAMILY = "family"
SOURCE_TUNED = "tuned"
SOURCE_ENV = "forced by env"
SOURCE_FLAG = "launch flag"
SOURCE_DEMOTED = "demoted"


def tensor_unit_section(probe: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """The tensor-unit (NAX) gate, from the one detector."""

    if probe is not None:
        return dict(probe())
    try:
        from mtplx import nax_detect

        hardware = bool(nax_detect.nax_hardware_available())
        forced = bool(nax_detect.gpu_family_fallback_forced())
        return {
            "architecture": nax_detect.gpu_architecture(),
            "hardware": hardware,
            "route": hardware and not forced,
            "fallback_forced": forced,
        }
    except Exception as exc:  # a doctor line must never break doctor
        return {"error": repr(exc)}


def build_explain_report(
    *,
    health: dict[str, Any] | None,
    server_url: str | None,
    tensor_units: dict[str, Any] | None = None,
    lane: dict[str, Any] | None = None,
    family_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the explain block of the doctor report."""

    report: dict[str, Any] = {
        "server_url": server_url,
        "server_reachable": False,
        "tensor_units": tensor_units if tensor_units is not None else tensor_unit_section(),
    }
    if lane is not None:
        report["lane"] = lane
    if family_settings is not None:
        report["family_settings"] = family_settings
    payload = health if isinstance(health, dict) else None
    if payload is not None and payload.get("ok") is False and "error" in payload:
        payload = None
    if payload is None:
        report["demotions"] = None
        report["demotions_note"] = (
            "No MTPLX server answered, so there are no live counters to read. "
            "Start a server (or pass --port) and run this again."
        )
        return report
    report["server_reachable"] = True
    degradation = payload.get("degradation")
    ledger = degradation.get("demotions") if isinstance(degradation, dict) else None
    if isinstance(ledger, dict):
        report["demotions"] = ledger
    else:
        report["demotions"] = None
        report["demotions_note"] = (
            "This server does not report demotion counters (it predates them)."
        )
    settings = payload.get("settings")
    if isinstance(settings, dict):
        report["served"] = {
            key: settings.get(key)
            for key in (
                "model",
                "depth",
                "adaptive_policy",
                "scheduler_mode",
                "prefill_chunk_tokens",
                "reasoning_effort",
                "ssd_session_cache",
                "ssd_session_cache_max_size",
            )
            if key in settings
        }
    served_lane = payload.get("resolved_lane")
    if isinstance(served_lane, dict):
        report["served_lane"] = served_lane
    return report


def _lane_lines(title: str, lane: dict[str, Any]) -> list[str]:
    lines = [title]
    choices = lane.get("choices") if isinstance(lane.get("choices"), dict) else {}
    for key in sorted(choices):
        item = choices[key]
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or SOURCE_DEFAULT
            reason = item.get("reason")
            line = f"  {key}: {value} ({source})"
            if reason:
                line += f". {reason}"
            lines.append(line)
        else:
            lines.append(f"  {key}: {item}")
    for note in lane.get("notes") or ():
        lines.append(f"  note: {note}")
    return lines


def render_explain_lines(report: dict[str, Any]) -> list[str]:
    """Plain lines for the terminal."""

    lines: list[str] = ["Explain"]
    units = report.get("tensor_units") or {}
    if "error" in units:
        lines.append(f"tensor units: unknown ({units['error']})")
    else:
        arch = units.get("architecture") or "unknown GPU"
        if units.get("fallback_forced"):
            lines.append(
                f"tensor units: off for this process ({arch}; "
                "MTPLX_FORCE_GPU_FAMILY_FALLBACK is set, so every lane takes "
                "the path an M1 to M4 takes)"
            )
        elif units.get("hardware"):
            lines.append(f"tensor units: available ({arch})")
        else:
            lines.append(
                f"tensor units: not available ({arch}); needs GPU generation 17 "
                "and macOS 26.2, so the portable kernels serve this Mac"
            )
    family = report.get("family_settings")
    if isinstance(family, dict) and family.get("rows"):
        lines.append(
            f"model-tuned settings for {family.get('model') or 'the default model'} "
            f"(family {family.get('family')}):"
        )
        for row in family["rows"]:
            lines.append(f"  {row['key']}: {row['value']} ({row['source']})")
            lines.append(f"      {row['receipt']}")
    lane = report.get("lane")
    if isinstance(lane, dict):
        lines.extend(_lane_lines("lane this Mac resolves for a new launch:", lane))
    served_lane = report.get("served_lane")
    if isinstance(served_lane, dict):
        lines.extend(_lane_lines("lane the running server resolved:", served_lane))
    served = report.get("served")
    if isinstance(served, dict) and served:
        lines.append("running server settings:")
        for key in sorted(served):
            lines.append(f"  {key}: {served[key]}")
    ledger = report.get("demotions")
    if isinstance(ledger, dict):
        lines.append(f"demotions since the server started ({report.get('server_url')}):")
        lines.extend("  " + line for line in demotions.explain_lines(ledger))
    else:
        lines.append(f"demotions: {report.get('demotions_note')}")
    return lines
