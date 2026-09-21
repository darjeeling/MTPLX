"""Explicit exceptions for staging the two 2.11.4 packs before upload."""
from __future__ import annotations

import argparse

from mtplx.profiles import (
    BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_QUALITY_HF_MODEL_ID,
)

NOT_YET_PUBLISHED_REPOS = frozenset({
    BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
    FLASH_NEXT_OPTIMIZED_QUALITY_HF_MODEL_ID,
})


def add_not_yet_published_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--not-yet-published", action="append", default=[],
        choices=sorted(NOT_YET_PUBLISHED_REPOS), metavar="REPO",
        help="Explicitly mark a new 2.11.4 repo as not yet published; repeat per repo. Remove after upload.",
    )
