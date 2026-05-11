"""Canonical path constants for NPUPy XDNA results and artifacts."""

from __future__ import annotations

from pathlib import Path

RESULTS_ROOT = Path("/home/lutet/ece511/npupy_xdna/results")
TIMINGS_DIR = RESULTS_ROOT / "timings"
PLOTS_DIR = RESULTS_ROOT / "plots"
XCLBIN_CACHE_DIR = RESULTS_ROOT / "xclbin_cache"
EVIDENCE_DIR = RESULTS_ROOT / "evidence"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"


def ensure_dirs() -> None:
    """Create all canonical result directories if they do not exist."""
    for d in (TIMINGS_DIR, PLOTS_DIR, XCLBIN_CACHE_DIR, EVIDENCE_DIR, CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
