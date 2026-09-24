"""Opt-in process policy for the standalone server, independent of cache storage."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
from typing import Any

_enabled = False


def metadata_only(args: Any = None) -> bool:
    return _enabled or getattr(args, "log_privacy", None) == "metadata-only"


def activate(args: Any) -> None:
    global _enabled
    _enabled = getattr(args, "log_privacy", None) == "metadata-only"


# Exact names, scalar numeric values only. Never accept arbitrary user labels,
# nested dictionaries, stringified exceptions or token arrays at the disk sink.
NUMERIC_FIELDS = frozenset(
    [
        "completed_at_s",
        "prompt_tokens",
        "completion_tokens",
        "generated_tokens",
        "total_tokens",
        "elapsed_s",
        "tok_s",
        "end_to_end_tok_s",
        "prompt_eval_time_s",
        "cache_restore_time_s",
        "prompt_target_prefill_time_s",
        "prompt_mtp_history_time_s",
        "prompt_target_prefill_tok_s",
        "prompt_mtp_history_tok_s",
        "prompt_tps",
        "prefill_tok_s",
        "prefill_compute_tok_s",
        "prefill_wall_tok_s",
        "ttft_s",
        "decode_elapsed_s",
        "request_elapsed_s",
        "request_tok_s",
        "decode_tok_s",
        "accepted_drafts",
        "rejected_drafts",
        "warmup",
        "cache_hit",
        "cached_tokens",
        "cached_prompt_tokens",
        "acceptance_rate",
        "accept_rate",
        "peak_memory_gb",
        "streamed_completion_tokens",
        "request_cancelled",
        "request_failed",
        "client_disconnected",
        "partial_decode_tok_s",
        "cancellation_elapsed_s",
        "server_elapsed_s",
    ]
)


def usage_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key in NUMERIC_FIELDS
        and type(value) in (int, float, bool)
        and (type(value) is not float or math.isfinite(value))
    }


class _DiscardText(io.TextIOBase):
    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        return len(value)


@contextlib.contextmanager
def private_console(args: Any):
    """Serving only: suppress free-form prints and sanitize logging at creation.

    Startup instructions are printed before entering this context. Usage remains
    available in JSONL/API. This also protects stdout/stderr redirected by a
    service manager; raw third-party exceptions can contain request bodies.
    """
    if not metadata_only(args):
        yield
        return
    factory = logging.getLogRecordFactory()
    diagnostic_stream = sys.stderr

    def safe_factory(*values, **kwargs):
        record = factory(*values, **kwargs)
        record.msg = "Server diagnostic; details omitted by metadata-only policy"
        record.args = ()
        record.exc_info = record.exc_text = record.stack_info = None
        if record.levelno >= logging.WARNING:
            diagnostic_stream.write(
                "Server warning/error; details omitted by metadata-only policy\n"
            )
            diagnostic_stream.flush()
        return record

    # Uvicorn uses non-propagating handlers, so the factory emits a fixed
    # operational notice before their output is discarded.
    logging.setLogRecordFactory(safe_factory)
    try:
        with (
            contextlib.redirect_stdout(_DiscardText()),
            contextlib.redirect_stderr(_DiscardText()),
        ):
            try:
                yield
            except Exception:  # noqa: BLE001 - never expose arbitrary exception bodies
                raise RuntimeError(
                    "Server failed; details omitted by metadata-only policy"
                ) from None
    finally:
        logging.setLogRecordFactory(factory)
