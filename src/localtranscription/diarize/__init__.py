"""Speaker diarization: who spoke when.

Two modes, deliberately different shapes:

- **offline** (`offline.py`) runs the pyannote community-1 family -- powerset segmentation
  over a sliding window, masked speaker embeddings, then clustering across the whole
  recording. It sees the entire session at once, so it can assign globally consistent
  speaker ids and split overlapped speech.
- **streaming** (`streaming.py`) runs NVIDIA's Sortformer, which is end-to-end and
  causal: it emits per-frame speaker activity as audio arrives, holding its own speaker
  cache so ids stay stable across a session.

Both run their heavy model through CoreML rather than torch, which is what makes them
fast enough to sit alongside ASR on the same machine -- see `coreml.py` for why the
compute unit is chosen per model rather than globally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Turn:
    """One speaker's continuous activity, in seconds from the start of the audio."""

    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, start: float, end: float) -> float:
        """Seconds this turn shares with [start, end). Used to label ASR words."""
        return max(0.0, min(self.end, end) - max(self.start, start))


@runtime_checkable
class Diarizer(Protocol):
    """What the engine requires of a diarizer. One method wide, like Backend."""

    name: str
    detail: str

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[Turn]: ...


def label_words(words: list[dict], turns: list[Turn]) -> list[dict]:
    """Attach a speaker to each timed word by whichever turn it overlaps most.

    Word timings come from the forced aligner and turns from the diarizer; neither knows
    about the other, so the join is done here on time alone. A word overlapping nothing
    keeps speaker=None rather than being forced into the nearest turn -- silence between
    speakers is a real answer.
    """
    out = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = t.overlap(w["start"], w["end"])
            if ov > best_ov:
                best, best_ov = t.speaker, ov
        out.append({**w, "speaker": best})
    return out


def merge_turns(turns: list[Turn], max_gap: float = 0.5,
                min_duration: float = 0.25) -> list[Turn]:
    """Join same-speaker turns separated by less than `max_gap`, drop slivers.

    Frame-level output is choppy -- a speaker's own pauses between words read as turn
    boundaries. Merging first, then dropping what's still too short, keeps a brief "mm"
    while discarding single-frame flicker.
    """
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: (t.start, t.speaker))
    merged = [ordered[0]]
    for t in ordered[1:]:
        last = merged[-1]
        if t.speaker == last.speaker and t.start - last.end <= max_gap:
            merged[-1] = Turn(last.start, max(last.end, t.end), last.speaker)
        else:
            merged.append(t)
    return [t for t in merged if t.duration >= min_duration]
