"""Where this tool keeps things, per the XDG Base Directory spec.

Writing into the working directory means a session run from a repo drops transcripts and
audio into that repo, and the correctness of `.gitignore` becomes load-bearing. These are
user data, not build output, so they belong under the user's data directory.

macOS's own convention is `~/Library/Application Support`, but XDG is what's asked for
here and the env vars are honoured either way, so `XDG_DATA_HOME=~/Library/...` gets that
behaviour without a code change.

    config       $XDG_CONFIG_HOME/mouth/config.toml   (~/.config/...)
    transcripts  $XDG_DATA_HOME/mouth/out             (~/.local/share/...)
    recordings   $XDG_DATA_HOME/mouth/recordings
    checkpoints  $XDG_CACHE_HOME/mouth/models         (~/.cache/...)

Quantised checkpoints are cache, not data: they're GBs, and `lt quantize` rebuilds any of
them from the upstream weights. Losing that directory costs time, not work.
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "mouth"


def _base(env: str, default: str) -> Path:
    """An XDG base directory. Empty or relative values are ignored, as the spec requires."""
    raw = os.environ.get(env, "")
    root = Path(raw) if raw and Path(raw).is_absolute() else Path.home() / default
    return root / APP


def data_dir() -> Path:
    return _base("XDG_DATA_HOME", ".local/share")


def cache_dir() -> Path:
    return _base("XDG_CACHE_HOME", ".cache")


def config_dir() -> Path:
    return _base("XDG_CONFIG_HOME", ".config")


def config_file() -> Path:
    """Defaults for the flags, written by hand. See config.py.

    Config, not data: it is the only file here a user edits, and the only one whose loss
    changes what a command *does* rather than what it has produced.
    """
    return config_dir() / "config.toml"


def out_dir() -> Path:
    """Where transcripts go."""
    return data_dir() / "out"


def record_dir() -> Path:
    """Where per-utterance audio and manifests go."""
    return data_dir() / "recordings"


def models_dir() -> Path:
    """Where `lt quantize` writes, and where `lt models` looks."""
    return cache_dir() / "models"


def calibration_file() -> Path:
    """Remembered VAD thresholds, keyed by input device.

    Cache, not data: it is a measurement of a room that `lt dictate --recalibrate`
    retakes in a second. It lives here because a per-launch calibration costs more than
    loading the model does, which is the whole reason dictation can start on a keypress.
    """
    return cache_dir() / "calibration.json"
