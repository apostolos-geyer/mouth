"""Frame sources: the live microphone, or a wav replayed as if it were one."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .audio import load as load_audio
from .vad import FRAME_LEN, FRAME_MS, SAMPLE_RATE


def ambient_threshold(levels) -> float:
    """Turn measured frame RMS into a VAD threshold.

    Shared so the mic and a replayed file agree: these constants are tuned, and having
    them written twice meant tuning the live path silently left replay -- what tests and
    demos actually run -- behaving differently.
    """
    if len(levels) == 0:
        return 0.01
    return max(3.0 * float(np.median(levels)), 0.005)


def _mic_key(mic: int | None) -> str:
    return "default" if mic is None else str(mic)


def cached_threshold(mic: int | None, path: Path | None = None) -> float | None:
    """A threshold measured on a previous run, or None.

    Keyed by device because thresholds describe a microphone in a room, not a machine --
    a laptop mic and a desk condenser do not share one. Any unreadable or malformed file
    reads as "no cached value": a stale cache must cost a calibration, never a crash.
    """
    path = path or paths.calibration_file()
    try:
        entry = json.loads(path.read_text())[_mic_key(mic)]
        value = float(entry["threshold"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if value > 0 else None


def remember_threshold(mic: int | None, value: float, path: Path | None = None) -> None:
    """Record a calibration for next time. Best-effort: never fails a session."""
    path = path or paths.calibration_file()
    try:
        try:
            table = json.loads(path.read_text())
            if not isinstance(table, dict):
                table = {}
        except (OSError, ValueError):
            table = {}
        table[_mic_key(mic)] = {"threshold": round(value, 6), "at": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=2, sort_keys=True))
    except OSError:
        pass


@dataclass
class MicSource:
    """Live microphone frames. Also exposes ambient calibration."""

    mic: int | None = None
    _q: queue.Queue = field(default_factory=queue.Queue)
    _stream: object | None = None

    def open(self):
        import sounddevice as sd

        def callback(indata, _frames, _time, _status):
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_LEN,
            device=self.mic,
            callback=callback,
        )
        self._stream.start()
        return self

    def close(self):
        if self._stream is not None:
            self._stream.stop()  # ty: ignore[unresolved-attribute]
            self._stream.close()  # ty: ignore[unresolved-attribute]

    def calibrate(self, seconds: float = 1.0, stop=None) -> float:
        """Measure ambient noise so the threshold suits the room, not a guess."""
        levels = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                break
            try:
                levels.append(float(np.sqrt(np.mean(self._q.get(timeout=0.5) ** 2))))
            except queue.Empty:
                break
        return ambient_threshold(levels)

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        while not stop.is_set():
            try:
                yield self._q.get(timeout=0.2)
            except queue.Empty:
                continue


@dataclass
class WavSource:
    """Replay a wav as if it were the mic -- for testing and demos without a device."""

    path: Path
    realtime: bool = False

    def open(self):
        # audio.load is the same decoder `lt diarize` uses: it downmixes and resamples
        # anything PyAV can read. Rolling a second, stricter loader here meant --wav
        # refused files the tool could already open one module away.
        self._audio = load_audio(self.path)
        return self

    def close(self):
        pass

    @property
    def seconds(self) -> float:
        """How much audio there is. Known here and nowhere else without decoding twice."""
        return len(self._audio) / SAMPLE_RATE

    def calibrate(self, seconds: float = 1.0, stop=None) -> float:
        head = self._audio[: int(seconds * SAMPLE_RATE)]
        if head.size == 0:
            return 0.01
        frames = head[: head.size - head.size % FRAME_LEN].reshape(-1, FRAME_LEN)
        return ambient_threshold(np.sqrt((frames**2).mean(axis=1)))

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        for i in range(0, len(self._audio) - FRAME_LEN, FRAME_LEN):
            if stop.is_set():
                return
            if self.realtime:
                time.sleep(FRAME_MS / 1000)
            yield self._audio[i : i + FRAME_LEN]


def make_source(mic: int | None, wav: Path | None, realtime: bool = False):
    return WavSource(wav, realtime=realtime) if wav else MicSource(mic)
