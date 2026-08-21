"""Persist per-utterance audio and the partial trajectory that produced its text.

This is the substrate for two things: the v4 caching benchmark (replay a real session's
exact partial schedule and compare compute), and the correction loop (audio paired with
what the model thought it heard).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vad import SAMPLE_RATE


@dataclass
class _Pending:
    interims: list = field(default_factory=list)


class SessionRecorder:
    """Writes utt-NNNN.flac plus a manifest.jsonl line per utterance.

    Partials arrive before the final for the same utterance, so they're buffered by
    utterance start and flushed when the final lands.
    """

    def __init__(self, root: Path, session_id: str, sample_rate: int = SAMPLE_RATE):
        self.dir = Path(root) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.jsonl"
        self.sample_rate = sample_rate
        self.n = 0
        self._pending: dict[float, _Pending] = {}
        self._lock = threading.Lock()

    # Belt and braces: _pending is drained by add()/discard(), but an unforeseen path that
    # strands an utterance must not grow the dict without bound over a long session.
    MAX_PENDING = 32

    def note_interim(self, start: float, audio_sec: float, text: str, took: float):
        with self._lock:
            self._pending.setdefault(start, _Pending()).interims.append(
                {"at": round(audio_sec, 3), "text": text, "took": round(took, 3)}
            )
            while len(self._pending) > self.MAX_PENDING:
                self._pending.pop(next(iter(self._pending)))  # oldest first

    def discard(self, start: float):
        """Drop buffered interims for an utterance whose final never produced text.

        Without this, an utterance that transcribes to nothing -- or raises -- leaves its
        partials parked in _pending for the life of the session.
        """
        with self._lock:
            self._pending.pop(start, None)

    def add(self, audio: np.ndarray, start: float, text: str, words, language: str,
            took: float) -> Path:
        import soundfile as sf

        with self._lock:
            self.n += 1
            uid = f"utt-{self.n:04d}"
            pending = self._pending.pop(start, _Pending())

        path = self.dir / f"{uid}.flac"
        try:
            # FLAC needs an integer subtype; PCM_16 is lossless for our purposes and
            # roughly halves the size versus raw wav.
            sf.write(str(path), audio, self.sample_rate, format="FLAC", subtype="PCM_16")
        except (RuntimeError, OSError, ValueError):
            # soundfile raises RuntimeError for an unsupported format/subtype pairing and
            # OSError for anything libsndfile refuses to open. Falling back to wav keeps
            # the utterance; losing it to an encoding preference would not be a trade.
            path = self.dir / f"{uid}.wav"
            sf.write(str(path), audio, self.sample_rate)

        entry = {
            "id": uid,
            "audio": path.name,
            "start": round(start, 3),
            "duration": round(len(audio) / self.sample_rate, 3),
            "language": language,
            "hypothesis": text,
            "took": round(took, 3),
            # The schedule the partials actually fired on -- replaying this is how the
            # caching work gets measured against a like-for-like baseline.
            "interims": pending.interims,
            "words": words,
        }
        with self._lock, self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return path


def load_manifest(session_dir: Path) -> list[dict]:
    """Read a recorded session back. Audio paths are resolved to absolute."""
    session_dir = Path(session_dir)
    entries = []
    manifest = session_dir / "manifest.jsonl"
    if not manifest.exists():
        return entries
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            # A hard exit can truncate the final append mid-line; skip it rather than
            # failing to load an otherwise good session.
            continue
        e["audio_path"] = str(session_dir / e["audio"])
        entries.append(e)
    return entries
