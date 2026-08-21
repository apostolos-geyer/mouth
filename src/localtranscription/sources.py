"""Frame sources: the live microphone, or a wav replayed as if it were one."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .vad import FRAME_LEN, FRAME_MS, SAMPLE_RATE


@dataclass
class MicSource:
    """Live microphone frames. Also exposes ambient calibration."""

    mic: Optional[int] = None
    _q: queue.Queue = field(default_factory=queue.Queue)
    _stream: object = None

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
            self._stream.stop()
            self._stream.close()

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
        if not levels:
            return 0.01
        return max(3.0 * float(np.median(levels)), 0.005)

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
        import soundfile as sf

        audio, sr = sf.read(str(self.path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{self.path} is {sr}Hz; expected {SAMPLE_RATE}Hz")
        self._audio = audio
        return self

    def close(self):
        pass

    def calibrate(self, seconds: float = 1.0, stop=None) -> float:
        head = self._audio[: int(seconds * SAMPLE_RATE)]
        if head.size == 0:
            return 0.01
        frames = head[: head.size - head.size % FRAME_LEN].reshape(-1, FRAME_LEN)
        levels = np.sqrt((frames**2).mean(axis=1))
        return max(3.0 * float(np.median(levels)), 0.005)

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        for i in range(0, len(self._audio) - FRAME_LEN, FRAME_LEN):
            if stop.is_set():
                return
            if self.realtime:
                time.sleep(FRAME_MS / 1000)
            yield self._audio[i : i + FRAME_LEN]


def make_source(mic: Optional[int], wav: Optional[Path], realtime: bool = False):
    return WavSource(wav, realtime=realtime) if wav else MicSource(mic)
