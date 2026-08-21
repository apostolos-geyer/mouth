"""Reading audio files that aren't already 16kHz mono wav.

soundfile handles wav/flac but not m4a/mp3, and the recordings people actually want to
diarize are usually whatever their meeting tool produced. PyAV is already in the tree
(via the ASR stack) and decodes and resamples in one pass, so there's no ffmpeg
subprocess to depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .vad import SAMPLE_RATE


def load(path: Path | str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any supported file to mono float32 at `sample_rate`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif", ".ogg"}:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr == sample_rate:
            return data
        # Fall through to PyAV rather than hand-rolling a resampler.

    import av

    container = av.open(str(path))
    if not container.streams.audio:
        raise ValueError(f"{path} has no audio stream")
    resampler = av.audio.resampler.AudioResampler(
        format="flt", layout="mono", rate=sample_rate
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):  # flush
        chunks.append(out.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    # The resampler is configured format="flt", so the chunks are already float32 --
    # concatenating with a dtype avoids a second full-length copy of the decoded file.
    return np.concatenate(chunks, dtype=np.float32)
