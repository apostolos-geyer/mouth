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


def ambient_threshold(levels, quiet=50.0) -> float:
    """Turn measured frame RMS into a VAD threshold: three times the noise floor.

    `quiet` is which percentile of the given frames is taken to *be* the noise floor. The
    median is right when you are listening to a room and most frames are silence, which is
    the mic's situation: it has one second, taken before anyone speaks.

    It is wrong for audio that has been edited. A recording with the silence cut out has
    speech at its median, so 3x median lands above most of the speech -- measured on a
    28-minute interview, a threshold of 0.146 where the floor was 0.004, which dropped 36%
    of the words. A file can look at all of itself before deciding, so it uses a low
    percentile and finds the floor that is actually there.

    Both paths keep the same 3x and the same 0.005 minimum, so mic audio lands in the same
    place either way -- on a recorded utterance here, p10 clamps to exactly the 0.005 the
    median path already produced.
    """
    if len(levels) == 0:
        return 0.01
    return max(3.0 * float(np.percentile(levels, quiet)), 0.005)


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
    #: Already-decoded audio, for a caller that had to look at the file before the session
    #: started -- `m transcribe --speakers` diarizes first, to know where the speaker
    #: changes are. Without this the file would be decoded twice.
    preloaded: np.ndarray | None = None

    def open(self):
        # audio.load is the same decoder `m diarize` uses: it downmixes and resamples
        # anything PyAV can read. Rolling a second, stricter loader here meant --wav
        # refused files the tool could already open one module away.
        self._audio = (
            self.preloaded if self.preloaded is not None else load_audio(self.path)
        )
        return self

    def close(self):
        pass

    @property
    def seconds(self) -> float:
        """How much audio there is. Known here and nowhere else without decoding twice."""
        return len(self._audio) / SAMPLE_RATE

    @property
    def audio(self) -> np.ndarray:
        """The decoded file. Exposed so a caller that also wants to diarize it does not
        decode a second time -- an hour of audio is ~4s of PyAV and 460 MB of float32."""
        return self._audio

    def calibrate(self, seconds: float = 1.0, stop=None) -> float:
        """The noise floor of the whole file, not of its first second.

        `seconds` is ignored, and that is the point: it exists because a microphone has to
        commit to a threshold before it has heard anything. A file is already here. Judging
        it by its opening second assumes the recording starts with silence, and the one
        that does not -- an interview with the pauses edited out -- opens on speech and
        calibrates 30x too high.
        """
        a = self._audio
        if a.size < FRAME_LEN:
            return 0.01
        frames = a[: a.size - a.size % FRAME_LEN].reshape(-1, FRAME_LEN)
        return ambient_threshold(np.sqrt((frames**2).mean(axis=1)), quiet=10.0)

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        for i in range(0, len(self._audio) - FRAME_LEN, FRAME_LEN):
            if stop.is_set():
                return
            if self.realtime:
                time.sleep(FRAME_MS / 1000)
            yield self._audio[i : i + FRAME_LEN]


def make_source(
    mic: int | None,
    wav: Path | None,
    realtime: bool = False,
    audio: np.ndarray | None = None,
):
    return WavSource(wav, realtime=realtime, preloaded=audio) if wav else MicSource(mic)
