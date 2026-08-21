"""Where this tool keeps things, per the XDG Base Directory spec.

Writing into the working directory means a session run from a repo drops transcripts and
audio into that repo, and the correctness of `.gitignore` becomes load-bearing. These are
user data, not build output, so they belong under the user's data directory.

macOS's own convention is `~/Library/Application Support`, but XDG is what's asked for
here and the env vars are honoured either way, so `XDG_DATA_HOME=~/Library/...` gets that
behaviour without a code change.

    transcripts  $XDG_DATA_HOME/localtranscription/out          (~/.local/share/...)
    recordings   $XDG_DATA_HOME/localtranscription/recordings
    checkpoints  $XDG_CACHE_HOME/localtranscription/models      (~/.cache/...)

Quantised checkpoints are cache, not data: they're GBs, and `lt quantize` rebuilds any of
them from the upstream weights. Losing that directory costs time, not work.
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "localtranscription"


def _base(env: str, default: str) -> Path:
    """An XDG base directory. Empty or relative values are ignored, as the spec requires."""
    raw = os.environ.get(env, "")
    root = Path(raw) if raw and Path(raw).is_absolute() else Path.home() / default
    return root / APP


def data_dir() -> Path:
    return _base("XDG_DATA_HOME", ".local/share")


def cache_dir() -> Path:
    return _base("XDG_CACHE_HOME", ".cache")


def out_dir() -> Path:
    """Where transcripts go."""
    return data_dir() / "out"


def record_dir() -> Path:
    """Where per-utterance audio and manifests go."""
    return data_dir() / "recordings"


def models_dir() -> Path:
    """Where `lt quantize` writes, and where `lt models` looks."""
    return cache_dir() / "models"
