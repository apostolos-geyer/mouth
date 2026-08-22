#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "qwen-asr>=0.0.6",
#     "torch>=2.13.0",
#     "sounddevice>=0.5.1",
#     "numpy>=1.26",
#     "soundfile>=0.13",
#     "typer>=0.15",
#     "textual>=1.0",
# ]
# ///
"""mouth v2 -- live mic transcription, CLI + TUI.

Same engine as live.py (Qwen3-ASR + Qwen3-ForcedAligner, energy VAD, worker thread);
v2 adds a typer CLI and a Textual TUI. Self-contained on purpose so v1 stays untouched.

  ./live2.py tui              full-screen live view
  ./live2.py cli              plain streaming output
  ./live2.py devices          list microphones
"""

from __future__ import annotations

import json
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples

SPEECH_FRAMES_TO_START = 3  # 90ms over threshold opens an utterance
SILENCE_FRAMES_TO_END = 25  # 750ms under threshold closes it
PREROLL_FRAMES = 10  # 300ms kept before onset so word starts aren't clipped
TAIL_FRAMES = 7  # 210ms of the closing silence kept; the rest is trimmed
# Gate on *voiced* frames, not clip length: every clip carries pre-roll plus trailing
# silence, so a length check would pass a 120ms cough as a ~1.1s utterance.
MIN_SPEECH_SEC = 0.3
MAX_UTTERANCE_SEC = 30.0  # forced aligner tops out at 180s; flush well before

LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese",
    "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish",
    "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian",
    "Macedonian",
]


# ---------------------------------------------------------------- output formats


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


def write_outputs(out_dir: Path, segments, words, stem=None) -> Optional[Path]:
    """Write the same four artifacts the offline tool produced."""
    segments = sorted(segments, key=lambda s: s[0])
    words = sorted(words, key=lambda w: w["start"])
    if not segments:
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
    return out_dir / stem


# ---------------------------------------------------------------- VAD


@dataclass
class Chunk:
    """A slice of audio to transcribe.

    final=False is a provisional look at an utterance still in progress; it gets
    superseded by the final pass over the same (longer) audio.
    """

    audio: np.ndarray
    start: float
    final: bool


def segment_utterances(frame_iter, threshold: float, on_level=None, interim_every: float = 0.0):
    """Cut a stream of fixed-size frames into utterances.

    Yields Chunks. on_level(rms, in_speech) is called per frame so a UI can draw a meter
    without re-reading the audio.

    With interim_every > 0, an in-progress utterance also emits a provisional Chunk that
    often, carrying everything heard so far -- so text appears while you're still talking
    instead of only after you stop. Re-sending the whole prefix each time is what qwen's
    own streaming mode does; cutting at a fixed boundary instead would slice mid-word.
    """
    preroll = deque(maxlen=PREROLL_FRAMES)
    utterance: list[np.ndarray] = []
    speech_run = 0
    silence_run = 0
    voiced = 0
    in_speech = False
    utt_start = 0.0
    consumed = 0
    last_interim = 0

    interim_frames = int(interim_every * SAMPLE_RATE / FRAME_LEN) if interim_every else 0
    min_voiced = MIN_SPEECH_SEC * SAMPLE_RATE / FRAME_LEN

    def build(utterance, silence_run, voiced):
        if voiced < min_voiced:
            return None  # a blip, not speech
        trim = max(0, silence_run - TAIL_FRAMES)
        kept = utterance[: len(utterance) - trim] if trim else utterance
        return np.concatenate(kept)

    for frame in frame_iter:
        consumed += 1
        now = consumed * FRAME_LEN / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(frame**2)))
        loud = rms > threshold
        if on_level is not None:
            on_level(rms, in_speech)

        if not in_speech:
            preroll.append(frame)
            speech_run = speech_run + 1 if loud else 0
            if speech_run >= SPEECH_FRAMES_TO_START:
                in_speech = True
                silence_run = 0
                voiced = speech_run
                utterance = list(preroll)
                utt_start = max(0.0, now - len(utterance) * FRAME_LEN / SAMPLE_RATE)
                last_interim = 0
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
                yield Chunk(audio, utt_start, final=True)
            in_speech = False
            speech_run = 0
            voiced = 0
            utterance = []
            preroll.clear()
        elif (
            interim_frames
            and voiced >= min_voiced
            and len(utterance) - last_interim >= interim_frames
        ):
            last_interim = len(utterance)
            yield Chunk(np.concatenate(utterance), utt_start, final=False)

    if in_speech and utterance:  # stream ended mid-utterance
        audio = build(utterance, silence_run, voiced)
        if audio is not None:
            yield Chunk(audio, utt_start, final=True)


# ---------------------------------------------------------------- engine

_STOP = object()


@dataclass
class Segment:
    start: float
    text: str
    audio_sec: float
    took: float


class Transcriber:
    """Drains utterances off a queue so capture never blocks on inference.

    Holds a Thread rather than subclassing one: CPython keeps private attributes on
    Thread and they move between versions (_stop was a method in 3.12, 3.13 added a
    _handle attribute), so any underscore name here risks silently shadowing one.
    """

    def __init__(self, model, language, on_segment=None, on_error=None, on_interim=None):
        self.model = model
        self.language = language
        self.on_segment = on_segment
        self.on_error = on_error
        self.on_interim = on_interim
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

    def close(self):
        self.work.put(_STOP)
        self.thread.join()

    def _run(self):
        while True:
            item = self.work.get()
            if item is _STOP:
                return
            # An interim is only worth running if nothing newer is already waiting --
            # otherwise partials pile up and starve the finals that actually get kept.
            if not item.final and not self.work.empty():
                self.dropped += 1
                continue
            try:
                self._transcribe_one(item)
            except Exception as e:  # a bad chunk shouldn't kill the session
                if self.on_error:
                    self.on_error(item.start, str(e))

    def _transcribe_one(self, chunk: Chunk):
        t0 = time.monotonic()
        results = self.model.transcribe(
            audio=(chunk.audio, SAMPLE_RATE),
            language=self.language,
            # Interim timestamps get discarded when the final lands, so skip the aligner.
            return_time_stamps=chunk.final,
        )
        result = results[0]
        text = (result.text or "").strip()
        if not text:
            return

        if not chunk.final:
            if self.on_interim:
                self.on_interim(Segment(chunk.start, text, len(chunk.audio) / SAMPLE_RATE,
                                        time.monotonic() - t0))
            return

        items = list(result.time_stamps) if result.time_stamps is not None else []
        for w in items:
            self.words.append(
                {"text": w.text, "start": chunk.start + w.start_time, "end": chunk.start + w.end_time}
            )
        seg = Segment(chunk.start, text, len(chunk.audio) / SAMPLE_RATE, time.monotonic() - t0)
        self.segments.append((chunk.start, text))
        if self.on_segment:
            self.on_segment(seg)


def load_model(device: str, on_status=None):
    import torch
    from qwen_asr import Qwen3ASRModel

    say = on_status or (lambda m: None)
    say(f"loading {ASR_MODEL} on {device}")
    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        forced_aligner=ALIGNER_MODEL,
        dtype=torch.bfloat16,
        device_map=device,
        # 512 default silently truncated long chunks in the offline tool; keep parity.
        max_new_tokens=2048,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": device},
    )
    say("warming up")
    # First inference on MPS pays graph-compilation cost; eat it before capture starts.
    model.transcribe(
        audio=(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE),
        language="English",
        return_time_stamps=True,
    )
    return model


# ---------------------------------------------------------------- frame sources


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

    def calibrate(self, seconds: float = 1.0) -> float:
        """Measure ambient noise so the threshold suits the room, not a guess."""
        levels = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
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

    def calibrate(self, seconds: float = 1.0) -> float:
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


@dataclass
class Config:
    out_dir: Path
    language: str
    device: str
    mic: Optional[int] = None
    wav: Optional[Path] = None
    threshold: Optional[float] = None
    interim: float = 1.2  # seconds between provisional passes; 0 disables


def run_session(cfg: Config, hooks, model=None) -> tuple[list, list]:
    """Shared driver for both front ends.

    hooks needs: status(str), level(rms, in_speech), segment(Segment), error(off, msg),
    ready(threshold), bind_stop(Event).

    Pass a preloaded model to skip loading here -- the TUI must, because loading spawns
    a subprocess and Textual's replacement stdout has no real fileno for it to inherit
    ("bad value(s) in fds_to_keep").
    """
    model = model or load_model(cfg.device, hooks.status)
    source = make_source(cfg.mic, cfg.wav, realtime=True).open()
    worker = Transcriber(
        model,
        cfg.language,
        on_segment=hooks.segment,
        on_error=hooks.error,
        on_interim=getattr(hooks, "interim", None),
    )
    worker.start()
    try:
        threshold = cfg.threshold
        if threshold is None:
            hooks.status("calibrating ambient noise, stay quiet")
            threshold = source.calibrate(1.0)
        hooks.ready(threshold)

        stop = threading.Event()
        hooks.bind_stop(stop)
        for chunk in segment_utterances(
            source.frames(stop), threshold, on_level=hooks.level, interim_every=cfg.interim
        ):
            worker.submit(chunk)
    finally:
        source.close()
        worker.close()
    return worker.segments, worker.words


# ---------------------------------------------------------------- TUI


def build_tui(cfg: Config, model):
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Footer, Header, RichLog, Static

    class Meter(Static):
        """Input level bar; colour tracks VAD state."""

        WIDTH = 28

        def __init__(self):
            super().__init__("")
            self.rms = 0.0
            self.in_speech = False
            self.threshold = 0.01

        def render_bar(self) -> str:
            # log-ish scale so quiet speech is still visible
            frac = min(1.0, (self.rms / max(self.threshold, 1e-6)) / 4.0)
            filled = int(frac * self.WIDTH)
            colour = "green" if self.in_speech else "grey42"
            bar = "█" * filled + "░" * (self.WIDTH - filled)
            tag = "SPEECH" if self.in_speech else "  --  "
            return f"[{colour}]{bar}[/] [b]{tag}[/b]"

    class TranscriptApp(App):
        CSS = """
        Screen { layout: vertical; }
        #bar { height: 3; padding: 0 1; border: round $primary; }
        #meter { width: 40; content-align: left middle; }
        #stats { content-align: right middle; }
        RichLog { border: round $primary; padding: 0 1; }
        #live { height: auto; min-height: 1; padding: 0 2; color: $text-muted; }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("p", "pause", "Pause"),
            ("c", "clear", "Clear"),
        ]

        def __init__(self):
            super().__init__()
            self.meter = Meter()
            self.stats = Static("", id="stats")
            self.log_view = RichLog(wrap=True, markup=True, auto_scroll=True)
            self.live = Static("", id="live")
            self.paused = False
            self.stop_event: Optional[threading.Event] = None
            self.started = time.monotonic()
            self.n_words = 0
            self.backlog = 0
            self.status_text = "starting"
            self.result = ([], [])

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="bar"):
                yield self.meter
                yield self.stats
            yield self.log_view
            yield self.live
            yield Footer()

        def on_mount(self):
            self.title = "mouth"
            self.sub_title = f"{cfg.language} · {cfg.device}"
            self.set_interval(1 / 15, self.refresh_bar)
            self.run_worker(self.pipeline, thread=True)

        def refresh_bar(self):
            self.meter.update(self.meter.render_bar())
            elapsed = fmt_clock(time.monotonic() - self.started)
            paused = " [yellow]PAUSED[/]" if self.paused else ""
            q = f" · queue {self.backlog}" if self.backlog else ""
            self.stats.update(
                f"[dim]{self.status_text}[/] · {elapsed} · {self.n_words} words{q}{paused}"
            )

        # --- hooks called from the worker thread ---
        def status(self, msg):
            self.status_text = msg

        def ready(self, threshold):
            self.meter.threshold = threshold
            self.status_text = "listening"
            self.call_from_thread(
                self.log_view.write, f"[dim]VAD threshold {threshold:.5f} — speak.[/]"
            )

        def bind_stop(self, stop):
            self.stop_event = stop

        def level(self, rms, in_speech):
            if self.paused:
                self.meter.rms, self.meter.in_speech = 0.0, False
                return
            self.meter.rms, self.meter.in_speech = rms, in_speech

        def interim(self, seg: Segment):
            # provisional -- will be rewritten by the final pass over the same utterance
            self.call_from_thread(self.live.update, f"[dim italic]… {seg.text}[/]")

        def segment(self, seg: Segment):
            self.n_words = len(seg.text.split()) + self.n_words
            self.backlog = 0
            self.call_from_thread(self.live.update, "")
            self.call_from_thread(
                self.log_view.write,
                f"[cyan]{fmt_clock(seg.start)}[/]  {seg.text}"
                f"  [dim]({seg.took:.1f}s/{seg.audio_sec:.1f}s)[/]",
            )

        def error(self, offset, msg):
            self.call_from_thread(
                self.log_view.write, f"[red]{fmt_clock(offset)}  transcribe failed: {msg}[/]"
            )

        def pipeline(self):
            try:
                self.result = run_session(cfg, self, model=model)
                # A wav source ends on its own; the mic only stops when you quit.
                self.status_text = "done · press q to save"
            except Exception as e:
                self.call_from_thread(self.log_view.write, f"[red]fatal: {e}[/]")
                self.status_text = "failed"

        # --- actions ---
        def action_pause(self):
            self.paused = not self.paused

        def action_clear(self):
            self.log_view.clear()
            self.live.update("")

        def action_quit(self):
            if self.stop_event:
                self.stop_event.set()
            self.exit()

    return TranscriptApp()


# ---------------------------------------------------------------- CLI


def _cli():
    import typer
    from rich.console import Console

    console = Console()
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
        help="Live local transcription with [b]Qwen3-ASR[/b] + forced alignment.",
    )

    def common(
        out: Path = typer.Option(Path("out"), "--out", "-o", help="Where transcripts go."),
        language: str = typer.Option("English", "--language", "-l", help="ASR language hint."),
        device: str = typer.Option("mps", "--device", help="torch device."),
        mic: Optional[int] = typer.Option(None, "--mic", "-m", help="Input device index."),
        wav: Optional[Path] = typer.Option(
            None, "--wav", help="Replay a 16kHz wav instead of the mic."
        ),
        threshold: Optional[float] = typer.Option(
            None, "--threshold", "-t", help="RMS VAD threshold [dim](default: auto)[/]."
        ),
        interim: float = 1.2,
    ) -> Config:
        if language not in LANGUAGES:
            raise typer.BadParameter(f"{language!r} not supported. See --help for the list.")
        return Config(out, language, device, mic, wav, threshold, interim)

    @app.command()
    def devices():
        """List available microphones."""
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                console.print(f"  [cyan]{i:>2}[/]  {d['name']}  [dim]({d['default_samplerate']:.0f}Hz)[/]")

    @app.command()
    def languages():
        """List supported ASR languages."""
        console.print(", ".join(LANGUAGES))

    @app.command()
    def tui(
        out: Path = typer.Option(Path("out"), "--out", "-o"),
        language: str = typer.Option("English", "--language", "-l"),
        device: str = typer.Option("mps", "--device"),
        mic: Optional[int] = typer.Option(None, "--mic", "-m"),
        wav: Optional[Path] = typer.Option(None, "--wav"),
        threshold: Optional[float] = typer.Option(None, "--threshold", "-t"),
        interim: float = typer.Option(
            1.2, "--interim", help="Seconds between provisional passes [dim](0 disables)[/]."
        ),
    ):
        """Full-screen live view [dim](q quit · p pause · c clear)[/]."""
        cfg = common(out, language, device, mic, wav, threshold, interim)
        # Load before entering full-screen: subprocess spawning breaks under Textual's
        # stdout, and it's a nicer wait with a visible status line.
        with console.status("[dim]loading model…[/]") as st:
            model = load_model(cfg.device, lambda m: st.update(f"[dim]{m}…[/]"))
        ui = build_tui(cfg, model)
        ui.run()
        segments, words = ui.result
        _report(console, cfg, segments, words)

    @app.command()
    def cli(
        out: Path = typer.Option(Path("out"), "--out", "-o"),
        language: str = typer.Option("English", "--language", "-l"),
        device: str = typer.Option("mps", "--device"),
        mic: Optional[int] = typer.Option(None, "--mic", "-m"),
        wav: Optional[Path] = typer.Option(None, "--wav"),
        threshold: Optional[float] = typer.Option(None, "--threshold", "-t"),
        interim: float = typer.Option(
            1.2, "--interim", help="Seconds between provisional passes [dim](0 disables)[/]."
        ),
    ):
        """Stream transcriptions to stdout [dim](Ctrl-C to stop)[/]."""
        import contextlib

        from rich.live import Live
        from rich.text import Text

        cfg = common(out, language, device, mic, wav, threshold, interim)

        # Provisional text rewrites itself in place, which needs a terminal that can take
        # the line back. Piped to a file, finals-only keeps the output clean.
        show_interim = sys.stdout.isatty() and cfg.interim > 0
        # transient: the provisional line is scratch space, so leave nothing behind.
        live = Live(Text(""), console=console, refresh_per_second=12, transient=True)

        class Hooks:
            def status(self, msg):
                console.print(f"[dim]{msg}…[/]", highlight=False)

            def ready(self, threshold):
                console.print(f"[dim]VAD threshold {threshold:.5f}[/]")
                console.print("[b]Listening.[/] Ctrl-C to stop.\n")

            def bind_stop(self, stop):
                # Flip a flag rather than raising, so an in-progress utterance still flushes.
                signal.signal(signal.SIGINT, lambda *_: stop.set())

            def level(self, rms, in_speech):
                pass

            def interim(self, seg):
                if show_interim:
                    live.update(Text(f"… {seg.text}", style="dim italic"))

            def segment(self, seg):
                # Live keeps its region at the bottom, so console.print lands above it.
                live.update(Text(""))
                console.print(
                    f"[cyan]{fmt_clock(seg.start)}[/]  {seg.text}"
                    f"  [dim]({seg.took:.1f}s/{seg.audio_sec:.1f}s)[/]",
                    highlight=False,
                )

            def error(self, offset, msg):
                live.update(Text(""))
                console.print(f"[red]{fmt_clock(offset)}  transcribe failed: {msg}[/]")

        # Load before the Live region starts: loading writes its own progress bars, and
        # two things driving the cursor at once garbles both.
        with console.status("[dim]loading model…[/]") as st:
            model = load_model(cfg.device, lambda m: st.update(f"[dim]{m}…[/]"))

        with (live if show_interim else contextlib.nullcontext()):
            segments, words = run_session(cfg, Hooks(), model=model)
        _report(console, cfg, segments, words)

    return app


def _report(console, cfg: Config, segments, words):
    if not segments:
        console.print("[yellow]Nothing transcribed.[/]")
        return
    base = write_outputs(cfg.out_dir, segments, words)
    console.print(
        f"\n[green]{len(segments)}[/] utterances, [green]{len(words)}[/] timed words"
        f" → [b]{base}[/b].{{txt,words.json,srt,timestamped.md}}"
    )


if __name__ == "__main__":
    _cli()()
