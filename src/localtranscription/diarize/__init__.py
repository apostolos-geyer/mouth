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

    Both sides grow with recording length, so the obvious nested loop is quadratic in it:
    a 28-minute interview is ~2M overlap tests, two hours ~40M -- as much time as the
    diarization it post-processes. Turns are sorted and disjoint per speaker, so a
    binary search bounds the candidates to the few that can actually overlap.
    """
    if not turns or not words:
        return [{**w, "speaker": None} for w in words]

    ordered = sorted(turns, key=lambda t: t.start)
    starts = np.fromiter((t.start for t in ordered), dtype=float, count=len(ordered))
    ends = np.fromiter((t.end for t in ordered), dtype=float, count=len(ordered))
    speakers = [t.speaker for t in ordered]
    # A turn can only reach a word if it starts before the word ends; scanning back from
    # there stops as soon as turns end before the word starts.
    longest = float((ends - starts).max())

    out = []
    for w in words:
        w_start, w_end = w["start"], w["end"]
        hi = int(np.searchsorted(starts, w_end, side="right"))
        lo = int(np.searchsorted(starts, w_start - longest, side="left"))
        best, best_ov = None, 0.0
        for i in range(lo, hi):
            ov = min(ends[i], w_end) - max(starts[i], w_start)
            if ov > best_ov:
                best, best_ov = speakers[i], ov
        out.append({**w, "speaker": best})
    return out


def merge_turns(
    turns: list[Turn], max_gap: float = 0.5, min_duration: float = 0.25
) -> list[Turn]:
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


def speaker_changes(turns) -> tuple[float, ...]:
    """The times a different voice takes over, as boundaries for the segmenter.

    Only where the speaker actually differs from the previous turn: a diarizer emits
    several consecutive turns for one person talking through their own pauses, and cutting
    on those would chop a sentence for no gain. The first turn is not a boundary either --
    nothing precedes it.
    """
    out, previous = [], None
    for turn in sorted(turns, key=lambda t: t.start):
        if previous is not None and turn.speaker != previous:
            out.append(float(turn.start))
        previous = turn.speaker
    return tuple(out)


def speaker_blocks(segments, turns, duration: float):
    """Group utterances into runs of one voice: [(speaker, start, end, text)].

    A block is what a reader thinks of as one person's turn at talking, and -- because
    utterances are already cut at speaker changes -- it is also a contiguous span of audio
    containing exactly one voice. That is what makes it alignable as a single piece.

    Attribution is by which turn each utterance overlaps most. With the cuts in place that
    is not really a vote any more: an utterance cannot straddle a change.
    """
    if not segments:
        return []
    bounds = [start for start, _ in segments] + [duration]

    owned = []
    for i, (start, text) in enumerate(segments):
        end = max(bounds[i + 1], start)
        held: dict = {}
        for t in turns:
            over = t.overlap(start, end)
            if over > 0:
                held[t.speaker] = held.get(t.speaker, 0.0) + over
        owned.append((max(held, key=lambda k: held[k]) if held else None, start, end, text))

    blocks = []
    for speaker, start, end, text in owned:
        if blocks and blocks[-1][0] == speaker:
            blocks[-1] = (speaker, blocks[-1][1], end, f"{blocks[-1][3]} {text}")
        else:
            blocks.append((speaker, start, end, text))
    return blocks
