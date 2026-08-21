#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "qwen-asr>=0.0.6",
#     "torch>=2.13.0",
#     "sounddevice>=0.5.1",
#     "numpy>=1.26",
# ]
# ///
"""Live mic transcription with Qwen3-ASR + Qwen3-ForcedAligner.

Adapted from the offline qwen-transcriber pipeline. The model has a streaming API
(streaming_transcribe) but it is vLLM-only and drops timestamps, so instead we cut the
mic stream into utterances with an energy VAD and push each one through the same
transcribe() call the offline tool used. That keeps the forced aligner, so we still get
word-level timestamps -- just offset onto a session-wide clock.

Ctrl-C to stop; the session transcript is written on exit.
"""

import argparse
import json
import queue
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from qwen_asr import Qwen3ASRModel

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples

# VAD tuning, in frames
SPEECH_FRAMES_TO_START = 3  # 90ms over threshold opens an utterance
SILENCE_FRAMES_TO_END = 25  # 750ms under threshold closes it
PREROLL_FRAMES = 10  # 300ms kept before onset so word starts aren't clipped
TAIL_FRAMES = 7  # 210ms of the closing silence kept; the rest is trimmed
# Gate on *voiced* frames, not clip length: every clip carries pre-roll plus trailing
# silence, so a length check would pass a 120ms cough as a ~1.1s utterance.
MIN_SPEECH_SEC = 0.3
MAX_UTTERANCE_SEC = 30.0  # forced aligner tops out at 180s; flush well before


# ---------------------------------------------------------------- output formats
# Same shapes the offline tool emitted, but over plain dicts since we re-base
# every timestamp onto the session clock.


def fmt_srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(t: float) -> str:
    mm, ss = divmod(int(t), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"


def words_to_srt(items, max_words_per_cue=12, max_gap=0.8):
    cues, cur = [], []
    for it in items:
        if cur and (it["start"] - cur[-1]["end"] > max_gap or len(cur) >= max_words_per_cue):
            cues.append(cur)
            cur = []
        cur.append(it)
    if cur:
        cues.append(cur)

    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_time(cue[0]['start'])} --> {fmt_srt_time(cue[-1]['end'])}")
        lines.append(" ".join(w["text"] for w in cue))
        lines.append("")
    return "\n".join(lines)


def words_to_timestamped_md(items, group_seconds=20):
    lines = []
    bucket_start = None
    bucket_words = []

    def flush():
        if bucket_words:
            lines.append(f"**[{fmt_clock(bucket_start)}]** {' '.join(bucket_words)}")
            lines.append("")

    for it in items:
        if bucket_start is None:
            bucket_start = it["start"]
        if it["start"] - bucket_start > group_seconds:
            flush()
            bucket_start = it["start"]
            bucket_words = []
        bucket_words.append(it["text"])
    flush()
    return "\n".join(lines)


# ---------------------------------------------------------------- transcription worker


_STOP = object()


class Transcriber:
    """Drains utterances off a queue so the mic never blocks on inference.

    Holds a Thread rather than subclassing one. CPython keeps private attributes on
    Thread and they move between versions -- _stop was a method in 3.12, and 3.13 added
    a _handle attribute -- so any underscore name here risks silently shadowing one.
    Composition sidesteps that whole class of bug.
    """

    def __init__(self, model, language):
        self.model = model
        self.language = language
        self.work = queue.Queue()
        self.words = []  # session-clock word dicts
        self.segments = []  # (start, text) per utterance
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, audio: np.ndarray, offset: float):
        self.work.put((audio, offset))

    def close(self):
        """Signal end of input and block until the backlog is transcribed."""
        self.work.put(_STOP)
        self.thread.join()

    def _run(self):
        while True:
            item = self.work.get()
            if item is _STOP:
                return
            audio, offset = item
            try:
                self._transcribe_one(audio, offset)
            except Exception as e:  # a bad chunk shouldn't kill the session
                print(f"  [transcribe failed at {fmt_clock(offset)}: {e}]", file=sys.stderr)

    def _transcribe_one(self, audio: np.ndarray, offset: float):
        t0 = time.monotonic()
        results = self.model.transcribe(
            audio=(audio, SAMPLE_RATE),
            language=self.language,
            return_time_stamps=True,
        )
        result = results[0]
        text = (result.text or "").strip()
        if not text:
            return

        items = list(result.time_stamps) if result.time_stamps is not None else []
        for w in items:
            self.words.append(
                {
                    "text": w.text,
                    "start": offset + w.start_time,
                    "end": offset + w.end_time,
                }
            )
        self.segments.append((offset, text))

        audio_sec = len(audio) / SAMPLE_RATE
        took = time.monotonic() - t0
        backlog = self.work.qsize()
        lag = f" ({took:.1f}s for {audio_sec:.1f}s audio"
        lag += f", {backlog} queued)" if backlog else ")"
        print(f"[{fmt_clock(offset)}] {text}")
        print(f"  \033[2m{lag.strip()}\033[0m", file=sys.stderr)


def write_outputs(out_dir: Path, segments, words, stem=None) -> Path | None:
    """Write the same four artifacts the offline tool produced."""
    segments = sorted(segments, key=lambda s: s[0])
    words = sorted(words, key=lambda w: w["start"])
    if not segments:
        print("Nothing transcribed.", file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"session-{time.strftime('%Y%m%d-%H%M%S')}"

    (out_dir / f"{stem}.txt").write_text(
        "\n".join(text for _, text in segments) + "\n", encoding="utf-8"
    )
    (out_dir / f"{stem}.words.json").write_text(json.dumps(words, indent=2), encoding="utf-8")
    if words:
        (out_dir / f"{stem}.srt").write_text(words_to_srt(words), encoding="utf-8")
        (out_dir / f"{stem}.timestamped.md").write_text(
            words_to_timestamped_md(words), encoding="utf-8"
        )

    print(
        f"{len(segments)} utterances, {len(words)} timed words -> {out_dir}/{stem}.*",
        file=sys.stderr,
    )
    return out_dir / stem


# ---------------------------------------------------------------- mic + VAD


def segment_utterances(frame_iter, threshold: float):
    """Cut a stream of fixed-size frames into utterances.

    Yields (audio, session_offset_seconds). The offset is derived from frames consumed,
    so it stays on the same clock the word timestamps get added to. Kept independent of
    the mic so it can be exercised with synthetic frames.
    """
    preroll = deque(maxlen=PREROLL_FRAMES)
    utterance: list[np.ndarray] = []
    speech_run = 0
    silence_run = 0
    voiced = 0  # loud frames in the current utterance
    in_speech = False
    utt_start = 0.0
    consumed = 0

    def build(utterance, silence_run, voiced):
        if voiced * FRAME_LEN / SAMPLE_RATE < MIN_SPEECH_SEC:
            return None  # a blip, not speech
        trim = max(0, silence_run - TAIL_FRAMES)
        kept = utterance[: len(utterance) - trim] if trim else utterance
        return np.concatenate(kept)

    for frame in frame_iter:
        consumed += 1
        now = consumed * FRAME_LEN / SAMPLE_RATE
        loud = float(np.sqrt(np.mean(frame**2))) > threshold

        if not in_speech:
            preroll.append(frame)
            speech_run = speech_run + 1 if loud else 0
            if speech_run >= SPEECH_FRAMES_TO_START:
                in_speech = True
                silence_run = 0
                voiced = speech_run
                utterance = list(preroll)
                # clock back to the start of the pre-roll we kept
                utt_start = max(0.0, now - len(utterance) * FRAME_LEN / SAMPLE_RATE)
                preroll.clear()
            continue

        utterance.append(frame)
        if loud:
            silence_run = 0
            voiced += 1
        else:
            silence_run += 1
        too_long = len(utterance) * FRAME_LEN / SAMPLE_RATE >= MAX_UTTERANCE_SEC

        if silence_run >= SILENCE_FRAMES_TO_END or too_long:
            audio = build(utterance, silence_run, voiced)
            if audio is not None:
                yield audio, utt_start
            in_speech = False
            speech_run = 0
            voiced = 0
            utterance = []
            preroll.clear()

    if in_speech and utterance:  # stream ended mid-utterance
        audio = build(utterance, silence_run, voiced)
        if audio is not None:
            yield audio, utt_start


def calibrate(frames: queue.Queue, seconds: float) -> float:
    """Measure ambient noise so the threshold suits the room, not a guess."""
    levels = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            frame = frames.get(timeout=0.5)
        except queue.Empty:
            break
        levels.append(float(np.sqrt(np.mean(frame**2))))
    if not levels:
        return 0.01
    ambient = float(np.median(levels))
    return max(3.0 * ambient, 0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("out"))
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="mps", help="torch device")
    ap.add_argument("--mic", type=int, default=None, help="input device index")
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="RMS VAD threshold; default auto")
    args = ap.parse_args()

    if args.list_mics:
        print(sd.query_devices())
        return

    print(f"Loading {ASR_MODEL} + {ALIGNER_MODEL} on {args.device}...", file=sys.stderr)
    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        forced_aligner=ALIGNER_MODEL,
        dtype=torch.bfloat16,
        device_map=args.device,
        # 180s chunks of dense speech overflow the 512-token default and get silently
        # truncated. Live utterances are short, but keep parity with the offline tool.
        max_new_tokens=2048,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": args.device},
    )

    # First inference on MPS pays graph-compilation cost; eat it before the mic is live.
    print("Warming up...", file=sys.stderr)
    model.transcribe(
        audio=(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE),
        language=args.language,
        return_time_stamps=True,
    )

    frames: queue.Queue = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  [audio: {status}]", file=sys.stderr)
        frames.put(indata[:, 0].copy())

    worker = Transcriber(model, args.language)
    worker.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_LEN,
        device=args.mic,
        callback=callback,
    )

    with stream:
        if args.threshold is None:
            print("Calibrating ambient noise, stay quiet for 1s...", file=sys.stderr)
            threshold = calibrate(frames, 1.0)
        else:
            threshold = args.threshold
        print(f"VAD threshold: {threshold:.5f}", file=sys.stderr)
        print("\033[1mListening. Ctrl-C to stop.\033[0m\n", file=sys.stderr)

        # Flip a flag rather than raising, so the segmenter exits its loop normally and
        # still flushes an utterance that was in progress when you hit Ctrl-C.
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        def mic_frames():
            while not stop.is_set():
                try:
                    yield frames.get(timeout=0.2)
                except queue.Empty:
                    continue

        for audio, offset in segment_utterances(mic_frames(), threshold):
            worker.submit(audio, offset)

        print("\nStopping...", file=sys.stderr)

    worker.close()
    write_outputs(args.out_dir, worker.segments, worker.words)


if __name__ == "__main__":
    main()
