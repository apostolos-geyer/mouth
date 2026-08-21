"""CoreML model loading, and the compute unit each model is actually fastest on.

Apple's dispatch hint is per-model, not per-process, and the right answer is not uniform:
measured on an M3 Max, segmentation runs the same on ANE or GPU (8.8ms per 10s window),
while the embedding model is 2.7x faster letting CoreML pick (11.6ms) than pinned to the
Neural Engine (31.5ms). So each model carries its own default and `--compute-units`
overrides all of them.

Weights come from FluidInference's conversion of the pyannote community-1 family, which
is ungated unlike the pyannote originals.
"""

from __future__ import annotations

import functools
from pathlib import Path

DIARIZATION_REPO = "FluidInference/speaker-diarization-coreml"

# "ALL" lets CoreML choose across ANE/GPU/CPU; the others pin it.
COMPUTE_UNITS = ("ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY")


class DiarizationUnavailable(RuntimeError):
    """Raised with an actionable message when the diarization deps aren't installed."""


def _require():
    try:
        import coremltools as ct
    except ImportError as e:
        raise DiarizationUnavailable(
            "diarization needs coremltools; install it with `uv sync --extra diarize`"
        ) from e
    return ct


def fetch(repo: str, patterns: list[str]) -> Path:
    """Download just the model directories we need, and return the snapshot root."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=patterns))


@functools.lru_cache(maxsize=8)
def load(path: str, compute_units: str = "ALL"):
    """Load a compiled .mlmodelc.

    Compiled rather than .mlpackage: CoreML would otherwise compile on every process
    start, which on the segmentation model is most of its load time.
    """
    ct = _require()
    if compute_units not in COMPUTE_UNITS:
        raise DiarizationUnavailable(
            f"unknown compute units {compute_units!r}; choose from {', '.join(COMPUTE_UNITS)}"
        )
    if not Path(path).exists():
        raise DiarizationUnavailable(f"no CoreML model at {path}")
    return ct.models.CompiledMLModel(
        str(path), compute_units=getattr(ct.ComputeUnit, compute_units)
    )


def resolve(repo: str, name: str, compute_units: str = "ALL", override: str | None = None):
    """Fetch `name`.mlmodelc out of `repo` (or use a local override) and load it."""
    if override:
        return load(str(Path(override)), compute_units)
    root = fetch(repo, [f"{name}.mlmodelc/**"])
    return load(str(root / f"{name}.mlmodelc"), compute_units)
