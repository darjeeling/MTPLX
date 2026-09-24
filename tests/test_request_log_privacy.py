"""The default-on request JSONL must keep its "no prompt content" promise (#326).

Literal user text stays on the in-RAM surfaces (dashboard ring, trace labels);
the durable line carries a non-reversible digest under the same key unless
MTPLX_REQUEST_LOG_CONTENT=1 explicitly opts back in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtplx.server.openai import (
    _record_request_metrics,
    _redact_request_log_record,
    parse_args,
)

SECRET = "the launch codes are in the blue folder"


def _state_with_log(tmp_path: Path) -> SimpleNamespace:
    log_path = tmp_path / "request-log.jsonl"
    args = parse_args(["--warmup-tokens", "0", "--request-log-jsonl", str(log_path)])
    return SimpleNamespace(args=args, last_metrics=[], log_path=log_path)


def _logged_record(state: SimpleNamespace) -> dict:
    lines = state.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_durable_log_redacts_user_preview(tmp_path, monkeypatch):
    monkeypatch.delenv("MTPLX_REQUEST_LOG_CONTENT", raising=False)
    state = _state_with_log(tmp_path)
    _record_request_metrics(
        state,
        {
            "request_id": "r1",
            "request_last_user_preview": SECRET,
            "completion_tokens": 3,
        },
    )
    record = _logged_record(state)
    assert SECRET not in state.log_path.read_text(encoding="utf-8")
    assert record["request_last_user_preview"].startswith("sha256:")
    # The digest is stable so forensics can still correlate repeated turns.
    assert (
        record["request_last_user_preview"]
        == _redact_request_log_record({"request_last_user_preview": SECRET})[
            "request_last_user_preview"
        ]
    )
    # The in-RAM ring (dashboard "recent") keeps the literal preview.
    assert state.last_metrics[-1]["request_last_user_preview"] == SECRET


def test_opt_in_keeps_literal_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_REQUEST_LOG_CONTENT", "1")
    state = _state_with_log(tmp_path)
    _record_request_metrics(
        state,
        {"request_id": "r2", "request_last_user_preview": SECRET},
    )
    assert _logged_record(state)["request_last_user_preview"] == SECRET


def test_redaction_leaves_other_fields_alone(monkeypatch):
    monkeypatch.delenv("MTPLX_REQUEST_LOG_CONTENT", raising=False)
    record = {"completion_tokens": 7, "request_last_user_preview": None}
    assert _redact_request_log_record(record) == record


def test_metadata_only_allows_only_numeric_usage(tmp_path, monkeypatch):
    from mtplx import log_privacy

    monkeypatch.setenv("MTPLX_REQUEST_LOG_CONTENT", "1")
    state = _state_with_log(tmp_path)
    state.args.log_privacy = "metadata-only"
    record = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "tok_s": 9.5,
        "request_last_user_preview": SECRET,
        "error": SECRET,
        "session_id": SECRET,
        "stop_sequence_matched": SECRET,
        "token_ids": [1, 2, 3],
        "arbitrary_numeric": 123,
        "ttft_s": SECRET,
        "elapsed_s": float("nan"),
    }
    _record_request_metrics(state, record)
    saved = _logged_record(state)
    assert saved["prompt_tokens"] == 12
    assert saved["completion_tokens"] == 3
    assert saved["tok_s"] == 9.5
    assert set(saved) <= log_privacy.NUMERIC_FIELDS | {"logged_at_s"}
    assert SECRET not in state.log_path.read_text()
    assert "ttft_s" not in saved and "elapsed_s" not in saved
    # The policy does not rewrite the response or in-memory metrics/cache.
    assert state.last_metrics[-1]["request_last_user_preview"] == SECRET


def test_metadata_only_overrides_capture_trace_and_debug_env(tmp_path, monkeypatch):
    from mtplx import log_privacy
    from mtplx.generation import _DecodeTrace
    from mtplx.request_capture import capture_request
    from mtplx.sampling import SamplerConfig
    from mtplx.server import openai
    from mtplx.server.flight_recorder import resolve_flight_recorder

    monkeypatch.setattr(log_privacy, "_enabled", True)
    for name in (
        "MTPLX_REQUEST_CAPTURE_DIR",
        "MTPLX_DEBUG_POSTCOMMIT_MISMATCH_DIR",
        "MTPLX_DECODE_TRACE_JSONL",
        "MTPLX_FLIGHT_RECORDER",
    ):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("MTPLX_FLIGHT_TEXT", "always")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TEXT", "1")
    recorder = resolve_flight_recorder(SimpleNamespace())
    assert not recorder.enabled and recorder.text_mode == "off"
    capture_request("test", {"prompt": SECRET})
    state = SimpleNamespace(args=SimpleNamespace())
    openai._dump_postcommit_mismatch(
        state, prompt_ids=[1], final_token_ids=[1], history_ids=[2], divergence=0
    )
    monkeypatch.setattr(openai, "_STREAM_CENSUS_DIR", str(tmp_path))
    openai._stream_census_record("secret", 'data: {"choices": []}')
    trace = _DecodeTrace(
        prompt_tokens=1,
        max_tokens=1,
        speculative_depth=1,
        sampler=SamplerConfig(),
        verify_strategy="staged",
        verify_core="stock",
        mtp_history_policy="x",
        mtp_cache_policy="x",
        trace_label=SECRET,
        trace_metadata={"prompt": SECRET},
    )
    assert not trace.enabled and trace.path is None
    assert list(tmp_path.iterdir()) == []


def test_private_console_suppresses_raw_output_and_tracebacks(monkeypatch, capsys):
    import logging

    import pytest

    from mtplx.log_privacy import private_console

    original_factory = logging.getLogRecordFactory()
    with private_console(SimpleNamespace(log_privacy="metadata-only")):
        print(SECRET)
        print(SECRET, file=sys.stderr)
        try:
            raise ValueError(SECRET)
        except ValueError:
            logging.getLogger("privacy-test").exception(SECRET)
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert "details omitted" in output.err
    assert logging.getLogRecordFactory() is original_factory
    with (
        pytest.raises(RuntimeError, match="details omitted"),
        private_console(SimpleNamespace(log_privacy="metadata-only")),
    ):
        raise ValueError(SECRET)


def test_metadata_only_real_request_paths(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient
    from test_server_openai import (
        _fake_generation,
        _fake_state,
        _fake_streaming_generation,
    )

    from mtplx import log_privacy
    from mtplx.server import openai

    monkeypatch.setattr(log_privacy, "_enabled", True)
    monkeypatch.setenv("MTPLX_REQUEST_LOG_CONTENT", "1")
    state = _fake_state()
    state.args.log_privacy = "metadata-only"
    state.args.request_log_jsonl = str(tmp_path / "usage.jsonl")
    monkeypatch.setattr(openai, "_encode_messages", lambda *args, **kwargs: [1, 2, 3])
    client = TestClient(openai.create_app(state))
    for stream in (False, True):
        generation = (
            _fake_streaming_generation(SECRET)
            if stream
            else lambda *a, **k: _fake_generation(SECRET)
        )
        monkeypatch.setattr(openai, "_run_generation", generation)
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={
                "model": "test",
                "messages": [{"role": "user", "content": SECRET}],
                "max_tokens": 8,
                "stream": stream,
            },
        )
        assert response.status_code == 200
    openai._record_stream_cancellation_metric(
        state,
        response_id=SECRET,
        session_id=SECRET,
        prompt_tokens=12,
        streamed_completion_tokens=2,
        stream_started_s=time.perf_counter() - 1,
        reason=SECRET,
        client_disconnected=True,
        request_observability={},
    )

    def fail(*args, **kwargs):
        raise ValueError(SECRET)

    monkeypatch.setattr(openai, "_run_generation", fail)
    # Runtime failures may be represented as HTTP or an ASGI exception. Both
    # must pass through the console guard when served by the standalone main.
    with log_privacy.private_console(state.args):
        try:
            client.post(
                "/v1/chat/completions",
                headers={"x-mtplx-cache-mode": "bypass"},
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": SECRET}],
                    "max_tokens": 8,
                },
            )
        except ValueError:
            pass
    text = (tmp_path / "usage.jsonl").read_text()
    assert SECRET not in text
    rows = [json.loads(line) for line in text.splitlines()]
    # Fake generation bypasses its normal metrics sink; the real cancellation
    # path above writes one receipt. Normal receipt filtering is tested separately.
    assert len(rows) == 1
    assert any(row.get("completion_tokens", 0) > 0 for row in rows)
    assert all(set(row) <= log_privacy.NUMERIC_FIELDS | {"logged_at_s"} for row in rows)
