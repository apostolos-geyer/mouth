"""Model loading, the inference worker, and the session driver both front ends share."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .backends import (
    DEFAULT_ALIGNER,
    DEFAULT_ASR,
    Backend,
    load_backend,
    open_partials,
)
from .recorder import SessionRecorder
from .sources import make_source
from .vad import MIN_SPEECH_SEC, SAMPLE_RATE, Cadence, Chunk, segment_utterances

LANGUAGES = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]

_STOP = object()

# Longest we'll wait on an in-flight inference when quitting. Covers a typical final so
# it isn't lost; past that the daemon thread is abandoned rather than blocking exit.
SHUTDOWN_TIMEOUT = 2.0


@dataclass
class Segment:
    start: float
    text: str
    audio_sec: float
    took: float
    # For provisional segments only: the prefix the decoder has committed to. A front end
    # can render this settled and the remainder as still-moving. Empty when the backend
    # re-transcribes the whole prefix each time, because then nothing is settled.
    stable: str = ""


@dataclass
class Config:
    # Defaults resolve under XDG rather than the working directory: a session run inside
    # a repo shouldn't write transcripts and audio into it. See paths.py.
    out_dir: Path = field(default_factory=paths.out_dir)
    language: str = "English"
    device: str = "mps"
    backend: str = "torch"
    # Weights are configuration, not a constant: any of these may be a local directory,
    # notably a quantised one from `lt quantize`.
    model: str = DEFAULT_ASR
    aligner: str = DEFAULT_ALIGNER
    # "auto" resolves per backend -- bf16 for torch/MPS, fp16 for MLX. Ignored for the
    # weights themselves when the checkpoint is already quantised.
    dtype: str = "auto"
    mic: int | None = None
    wav: Path | None = None
    threshold: float | None = None
    # None means no partials at all, which is what `--interim 0` asks for.
    cadence: Cadence | None = field(default_factory=Cadence)
    # Voiced audio an utterance needs before it is one. See vad.MIN_SPEECH_SEC.
    min_speech: float = MIN_SPEECH_SEC
    record: bool = True
    record_dir: Path = field(default_factory=paths.record_dir)
    # How provisional passes are computed.
    #   "reencode" -- re-transcribe the whole prefix each time. Best text, and it heals:
    #                 a word already on screen can be revised. Cost is quadratic in
    #                 utterance length, which the geometric cadence exists to contain.
    #   "x-draft"  -- re-transcribe the prefix, but decode it against the previous pass
    #                 as a speculative draft: same text, far fewer forward passes.
    #                 Experimental, and mlx only. See backends._MlxDraftDecoder.
    #   "stream"   -- feed only new audio to a decoder that keeps its KV cache. Linear
    #                 cost (measured 3.4x -> 1.2x one full pass on a 20s utterance), but
    #                 text appends rather than healing and first text waits for
    #                 `stream_chunk_sec`. Needs a backend with open_stream().
    partials: str = "reencode"
    stream_chunk_sec: float = 2.0
    session_id: str = ""
    # Whether finals run the forced aligner. Off is for callers that want a string and
    # not a transcript -- `lt dictate` -- and pairs with load_backend(align=False), which
    # is what actually saves the load. Leaving this on with an unaligned backend raises.
    timestamps: bool = True
    # How long a deliberate stop waits on an in-flight final. The default suits a session
    # front end, where quitting means quitting and one lost utterance sits among many.
    # A single-utterance caller raises it: abandoning that final loses the whole result.
    shutdown_timeout: float = SHUTDOWN_TIMEOUT

    def stamped(self) -> str:
        return self.session_id or f"session-{time.strftime('%Y%m%d-%H%M%S')}"


class Transcriber:
    """Drains chunks off a queue so capture never blocks on inference.

    Holds a Thread rather than subclassing one: CPython keeps private attributes on
    Thread and they move between versions (_stop was a method in 3.12, 3.13 added a
    _handle attribute), so any underscore name here risks silently shadowing one.
    """

    def __init__(
        self,
        backend: Backend,
        language,
        on_segment=None,
        on_error=None,
        on_interim=None,
        recorder: SessionRecorder | None = None,
        partials=None,
        timestamps: bool = True,
    ):
        self.timestamps = timestamps
        # How provisional text gets made. Never None: "no special decoder" is
        # _ReencodePartials, so this is one object rather than two optional ones and a
        # branch. Finals never touch it -- they run the aligner and are what gets saved.
        self.partials = partials or open_partials(
            backend, mode="reencode", language=language
        )
        self._partial_at: float | None = None
        self.backend = backend
        self.language = language
        self.on_segment = on_segment
        self.on_error = on_error
        self.on_interim = on_interim
        self.recorder = recorder
        self.work = queue.Queue()
        self.words = []
        self.segments = []
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def backlog(self) -> int:
        return self.work.qsize()

    def start(self):
        self.thread.start()

    def submit(self, chunk: Chunk):
        self.work.put(chunk)

    def close(self, drain: bool = True, timeout: float | None = None) -> bool:
        """Stop the worker. Returns False if it was still busy and got abandoned.

        drain=False is the quit path: partials are disposable by definition, so they're
        dropped rather than waited out, but queued finals are kept because those are the
        product. That's the difference between instant exit and sitting through a backlog.
        """
        if not drain:
            kept = []
            try:
                while True:
                    item = self.work.get_nowait()
                    if getattr(item, "final", False):
                        kept.append(item)
                    else:
                        self.dropped += 1
            except queue.Empty:
                pass
            for item in kept:
                self.work.put(item)

        self.work.put(_STOP)
        if self.thread.ident is None:
            return (
                True  # never started; joining it would raise, and this is a teardown path
            )
        self.thread.join(timeout)
        # Daemon thread: if inference is wedged, it dies with the process rather than
        # holding up exit.
        return not self.thread.is_alive()

    def _run(self):
        while True:
            item = self.work.get()
            if item is _STOP:
                return
            # An interim is only worth running if nothing newer is already waiting --
            # otherwise partials pile up and starve the finals that actually get kept.
            # An *incremental* one can't be dropped: its audio is a delta the decoder has
            # not seen, so skipping it would put a hole in the stream.
            if not item.final and not item.incremental and not self.work.empty():
                self.dropped += 1
                continue
            try:
                self._transcribe_one(item)
            except Exception as e:  # a bad chunk shouldn't kill the session
                if item.final and self.recorder:
                    self.recorder.discard(item.start)
                if self.on_error:
                    self.on_error(item.start, str(e))

    def _emit_interim(self, seg: Segment) -> None:
        """Publish one provisional segment.

        However it was produced -- streamed, drafted, re-encoded -- a partial ends here.
        These four lines used to be written out once per production path, and the copies
        had already started to differ: only the streamed one set `stable`.
        """
        if self.recorder:
            self.recorder.note_interim(seg.start, seg.audio_sec, seg.text, seg.took)
        if self.on_interim:
            self.on_interim(seg)

    def _partial(self, chunk: Chunk) -> Segment | None:
        """Provisional text for an utterance still in progress.

        The decoder owns *how*; this owns when its state is stale. Utterance boundaries
        were tracked in two places before -- here for the stream, inside the decoder for
        the draft -- which is two answers to one question.
        """
        if self._partial_at != chunk.start:
            # Only if it was actually following one: the first utterance of a session has
            # no previous state, and resetting a decoder that never ran is a no-op worth
            # not performing.
            if self._partial_at is not None:
                self.partials.reset()
            self._partial_at = chunk.start
        t0 = time.monotonic()
        text = self.partials.text(chunk.audio).strip()
        if not text:
            return None
        return Segment(
            chunk.start,
            text,
            len(chunk.audio) / SAMPLE_RATE,
            time.monotonic() - t0,
            stable=self.partials.stable,
        )

    def _transcribe_one(self, chunk: Chunk):
        if not chunk.final:
            seg = self._partial(chunk)
            if seg is not None:
                self._emit_interim(seg)
            return

        # Whatever happens below -- empty text, an exception -- this utterance is over,
        # and the next one must not inherit its decoder state.
        if self._partial_at is not None:
            self.partials.reset()
            self._partial_at = None

        t0 = time.monotonic()
        out = self.backend.transcribe(
            chunk.audio,
            SAMPLE_RATE,
            language=self.language,
            # Only finals reach here now; partials never run the aligner.
            timestamps=self.timestamps,
        )
        text = out.text.strip()
        if not text:
            if self.recorder:
                self.recorder.discard(chunk.start)
            return
        took = time.monotonic() - t0
        audio_sec = len(chunk.audio) / SAMPLE_RATE

        words = [
            {"text": w.text, "start": chunk.start + w.start, "end": chunk.start + w.end}
            for w in out.words
        ]
        self.words.extend(words)
        self.segments.append((chunk.start, text))
        # Publish before writing: recorder.add() encodes FLAC synchronously on this
        # thread, and the final is already correct -- making the screen wait on the disk
        # also delays the next utterance's first partial behind it.
        if self.on_segment:
            self.on_segment(Segment(chunk.start, text, audio_sec, took))
        if self.recorder:
            self.recorder.add(chunk.audio, chunk.start, text, words, self.language, took)


def run_session(cfg: Config, hooks, backend=None, stop: threading.Event | None = None):
    """Drive one capture session. Returns (segments, words, recorder).

    hooks needs: status(str), ready(threshold), bind_stop(Event), level(rms, in_speech),
    segment(Segment), error(offset, msg); optionally interim(Segment).

    Pass a preloaded backend to skip loading here -- the TUI must, because loading spawns
    a subprocess and Textual's replacement stdout has no real fileno for it to inherit
    ("bad value(s) in fds_to_keep").
    """
    backend = backend or load_backend(
        cfg.backend,
        model=cfg.model,
        aligner=cfg.aligner,
        device=cfg.device,
        dtype=cfg.dtype,
        on_status=hooks.status,
        align=cfg.timestamps,
    )
    # The caller may own the event so quitting works before we ever get here -- otherwise
    # there's a window during load and calibration where nothing can stop the session.
    stop = stop if stop is not None else threading.Event()
    session_id = cfg.stamped()
    recorder = SessionRecorder(cfg.record_dir, session_id) if cfg.record else None

    # Falling back is a cost, not a failure: a backend that cannot do the asked-for mode
    # still transcribes, just more expensively, so say so rather than end the session over
    # a provisional pass.
    partials = open_partials(
        backend, mode=cfg.partials, language=cfg.language, chunk_sec=cfg.stream_chunk_sec
    )
    if partials.mode != cfg.partials:
        hooks.status(f"{backend.name} cannot do {cfg.partials} partials; re-encoding them")

    source = make_source(cfg.mic, cfg.wav, realtime=True).open()
    worker = Transcriber(
        backend,
        cfg.language,
        on_segment=hooks.segment,
        on_error=hooks.error,
        on_interim=getattr(hooks, "interim", None),
        recorder=recorder,
        partials=partials,
        timestamps=cfg.timestamps,
    )
    worker.start()
    # Publish immediately: results accumulate on the worker, so a front end that quits
    # mid-session can still save what's already transcribed instead of waiting on us.
    attach = getattr(hooks, "attach", None)
    if attach:
        attach(worker, recorder)
    try:
        hooks.bind_stop(stop)
        threshold = cfg.threshold
        if threshold is None:
            hooks.status("calibrating ambient noise, stay quiet")
            threshold = source.calibrate(1.0, stop)
        if not stop.is_set():
            hooks.ready(threshold)
            for chunk in segment_utterances(
                source.frames(stop),
                threshold,
                on_level=hooks.level,
                cadence=cfg.cadence,
                incremental=partials.incremental,
                min_speech=cfg.min_speech,
            ):
                worker.submit(chunk)
    finally:
        source.close()
        # Deliberate stop -> drop pending partials. Natural end (wav ran out) -> finish.
        worker.close(drain=not stop.is_set(), timeout=cfg.shutdown_timeout)
    return worker.segments, worker.words, recorder
