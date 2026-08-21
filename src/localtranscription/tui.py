"""Textual front end.

Partials grow in place in the transcript itself -- the in-progress utterance is a live
widget that keeps rewriting until the final pass replaces it, so healing is visible where
the text will actually land rather than on a separate status line.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import ClassVar

from rich.markup import escape
from rich.text import Text

from .backends import Biasable
from .engine import Segment, run_session
from .formats import fmt_clock

# Tokyo Night-ish; reads well on dark terminals and keeps the meter legible.
BORDER = "#3b4261"
DIM = "#565f89"
TEXT = "#c0caf5"
BLUE = "#7aa2f7"
CYAN = "#2ac3de"
GREEN = "#9ece6a"
YELLOW = "#e0af68"
RED = "#f7768e"
MAGENTA = "#bb9af7"

# No leading space: silence should read as a baseline, not a hole in the graph.
BLOCKS = "▁▂▃▄▅▆▇█"
DB_FLOOR = -60.0


def to_db(rms: float) -> float:
    return max(DB_FLOOR, 20 * math.log10(max(rms, 1e-9)))


def db_frac(rms: float) -> float:
    """dB mapped to 0..1 -- far better visual dynamics than raw RMS."""
    return min(1.0, max(0.0, (to_db(rms) - DB_FLOOR) / -DB_FLOOR))


def gradient_bar(frac: float, width: int, marker: int | None = None) -> str:
    """Filled bar, green→yellow→red, with an optional tick showing the VAD threshold."""
    filled = int(frac * width)
    g_end, y_end = int(width * 0.60), int(width * 0.85)
    g = min(filled, g_end)
    y = max(0, min(filled, y_end) - g_end)
    r = max(0, filled - y_end)
    empty = width - filled

    out = f"[{GREEN}]{'█' * g}[/]"
    if y:
        out += f"[{YELLOW}]{'█' * y}[/]"
    if r:
        out += f"[{RED}]{'█' * r}[/]"
    if empty:
        tail = "░" * empty
        if marker is not None and filled <= marker < width:
            i = marker - filled
            tail = tail[:i] + "│" + tail[i + 1 :]
        out += f"[{BORDER}]{tail}[/]"
    return out


def sparkline(values, threshold: float, width: int) -> str:
    """Scrolling level history, btop-style. Coloured by whether the VAD saw speech."""
    vals = list(values)[-width:]
    pad = width - len(vals)
    out, buf, last = [], "", None
    for v in vals:
        lvl = int(db_frac(v) * (len(BLOCKS) - 1))
        colour = GREEN if v > threshold else DIM
        ch = BLOCKS[max(0, min(len(BLOCKS) - 1, lvl))]
        if colour == last:
            buf += ch
        else:
            if buf:
                out.append(f"[{last}]{buf}[/]")
            buf, last = ch, colour
    if buf:
        out.append(f"[{last}]{buf}[/]")
    return f"[{BORDER}]{'·' * pad}[/]" + "".join(out)


def build_tui(cfg, backend):
    from textual.app import App, ComposeResult
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import Footer, Input, Static

    class Line(Static):
        """One utterance. Lives as a partial, then hardens into the final."""

        def __init__(self, start: float):
            super().__init__("")
            self.start = start
            self.add_class("line")

        def show_partial(self, text: str):
            self.update(
                Text.assemble(
                    (f"{fmt_clock(self.start)}  ", DIM),
                    (text, f"italic {DIM}"),
                    ("▌", MAGENTA),
                )
            )

        def show_final(self, seg: Segment):
            self.update(
                Text.assemble(
                    (f"{fmt_clock(seg.start)}  ", CYAN),
                    (seg.text, TEXT),
                    (f"   {seg.took:.1f}s / {seg.audio_sec:.1f}s", DIM),
                )
            )

    class Transcript(VerticalScroll):
        """Append-only log, except the tail line which rewrites while speech is live."""

        # One widget per utterance, so an all-day dictation session would otherwise grow
        # the widget tree without bound. Only the *view* is capped -- the transcript that
        # gets written comes from the worker, not from here.
        MAX_LINES = 400

        def __init__(self):
            super().__init__(id="transcript")
            self.active: Line | None = None

        def _trim(self):
            excess = len(self.children) - self.MAX_LINES
            if excess <= 0:
                return
            # One batched removal rather than N scheduled ones -- each remove() queues a
            # message, and trimming per-line would flood the pump on a long session.
            doomed = [c for c in list(self.children)[:excess] if c is not self.active]
            if doomed:
                self.remove_children(doomed)

        def partial(self, start: float, text: str):
            if self.active is None or self.active.start != start:
                self.active = Line(start)
                self.mount(self.active)
            self.active.show_partial(text)
            self._trim()
            self.scroll_end(animate=False)

        def final(self, seg: Segment):
            line = self.active if self.active and self.active.start == seg.start else None
            if line is None:
                line = Line(seg.start)
                self.mount(line)
            line.show_final(seg)
            self.active = None
            self._trim()
            self.scroll_end(animate=False)

        def note(self, msg: str, colour: str = DIM):
            self.mount(Static(Text(msg, style=colour), classes="line"))
            self._trim()
            self.scroll_end(animate=False)

        def wipe(self):
            for child in list(self.children):
                child.remove()
            self.active = None

    class TranscriptApp(App):
        CSS = f"""
        Screen {{ background: $surface; }}
        #hud {{
            height: 6; margin: 0 1; padding: 0 1;
            border: round {BORDER}; border-title-color: {BLUE};
            border-title-style: bold;
        }}
        #transcript {{
            margin: 0 1; padding: 0 1;
            border: round {BORDER}; border-title-color: {BLUE};
            border-title-style: bold;
        }}
        .line {{ height: auto; padding: 0 0 1 0; }}
        Footer {{ background: $surface; }}
        #context {{
            margin: 0 1; border: round {BLUE}; display: none;
        }}
        #context.open {{ display: block; }}
        """
        BINDINGS: ClassVar = [
            ("q", "quit", "quit"),
            ("p", "pause", "pause"),
            ("c", "clear", "clear"),
            ("k", "context", "context"),
        ]

        METER_W = 34
        SPARK_W = 34

        def __init__(self):
            super().__init__()
            self.meter = Static("", id="meter")
            self.stats = Static("", id="stats")
            self.transcript = Transcript()
            self.hud = Vertical(self.meter, self.stats, id="hud")
            # Words the model should expect. Hidden until `k`, because it is empty far
            # more often than not and an always-visible empty field is just less screen.
            # No `value=` here: Input's value is a Textual reactive, and setting one
            # fires a watcher that needs a running app. build_tui runs before the app
            # exists (loading spawns subprocesses, which Textual's stdout breaks), so
            # `lt tui --context "..."` crashed on construction. Seeded in on_mount.
            self.context = Input(
                placeholder="names, jargon, spellings the model should expect",
                id="context",
            )

            self.rms = 0.0
            self.in_speech = False
            self.threshold = 0.01
            self.history = deque(maxlen=self.SPARK_W)
            self.paused = False
            # Owned here, not created inside run_session -- otherwise quitting during
            # model load or calibration has nothing to signal and the session runs on.
            self.stop = threading.Event()
            self.worker = None
            self.recorder = None
            self.started = time.monotonic()
            self.n_words = 0
            self.n_utts = 0
            self.status_text = "starting"
            self.result = ([], [], None)

        def compose(self) -> ComposeResult:
            yield self.hud
            yield self.context
            yield self.transcript
            yield Footer()

        def action_context(self):
            """Show the context field and focus it, or put it away again."""
            open_now = not self.context.has_class("open")
            self.context.set_class(open_now, "open")
            if open_now:
                self.context.focus()
            else:
                self.transcript.focus()

        def on_input_submitted(self, event: Input.Submitted):
            """Enter applies it. Live, because that is the point of editing it here.

            The backend reads `context` when it decodes, so this lands on the next
            utterance -- not the one in flight, which is already through the encoder.
            """
            if not isinstance(backend, Biasable):
                self.transcript.note("this backend cannot take context", YELLOW)
                return
            backend.context = event.value.strip()
            self.context.remove_class("open")
            self.transcript.focus()
            self.transcript.note(
                f"context: {backend.context}" if backend.context else "context cleared"
            )

        def on_mount(self):
            if isinstance(backend, Biasable) and backend.context:
                self.context.value = backend.context
            self.hud.border_title = "input"
            self.transcript.border_title = "transcript"
            self.title = "localtranscription"
            self.set_interval(1 / 15, self.tick)
            self.run_worker(self.pipeline, thread=True)

        def tick(self):
            self.history.append(0.0 if self.paused else self.rms)
            marker = int(db_frac(self.threshold) * self.METER_W)
            tag = (
                f"[{YELLOW}]PAUSED[/]"
                if self.paused
                else f"[b {GREEN}]SPEECH[/]"
                if self.in_speech
                else f"[{DIM}]idle  [/]"
            )
            self.meter.update(
                f"{sparkline(self.history, self.threshold, self.SPARK_W)}\n"
                f"{gradient_bar(0 if self.paused else db_frac(self.rms), self.METER_W, marker)}"
                f"  {tag}  [{DIM}]{to_db(self.rms):>5.1f} dB[/]"
            )

            def cell(label, value, colour=TEXT):
                return f"[{DIM}]{label}[/] [{colour}]{value}[/]"

            q = self.worker.backlog if self.worker else 0
            self.stats.update(
                "  ".join(
                    [
                        cell("elapsed", fmt_clock(time.monotonic() - self.started)),
                        cell("utts", self.n_utts),
                        cell("words", self.n_words),
                        cell("queue", q, YELLOW if q else TEXT),
                        cell("thr", f"{self.threshold:.4f}"),
                        cell("lang", cfg.language, BLUE),
                        cell("via", getattr(backend, "name", "?"), MAGENTA),
                        (f"[{RED}]●[/] [{DIM}]rec[/]" if cfg.record else ""),
                    ]
                )
                + f"\n[{DIM}]{self.status_text}[/]"
            )

        # --- hooks, called from the worker thread ---
        def status(self, msg):
            self.status_text = msg

        def ready(self, threshold):
            self.threshold = threshold
            self.status_text = "listening"
            self.call_from_thread(
                self.transcript.note, f"ready · VAD threshold {threshold:.5f} · speak"
            )

        def bind_stop(self, stop):
            pass  # the app owns the event and passes it in

        def attach(self, worker, recorder):
            self.worker, self.recorder = worker, recorder

        def collected(self):
            """Whatever is transcribed so far -- valid even mid-session, so quitting
            saves the work instead of racing run_session's return value."""
            if self.worker is not None:
                return self.worker.segments, self.worker.words, self.recorder
            return self.result

        def level(self, rms, in_speech):
            self.rms = 0.0 if self.paused else rms
            self.in_speech = in_speech and not self.paused

        def interim(self, seg: Segment):
            self.call_from_thread(self.transcript.partial, seg.start, escape(seg.text))

        def segment(self, seg: Segment):
            self.n_words += len(seg.text.split())
            self.n_utts += 1
            self.call_from_thread(self.transcript.final, seg)

        def error(self, offset, msg):
            self.call_from_thread(
                self.transcript.note, f"{fmt_clock(offset)}  transcribe failed: {msg}", RED
            )

        def pipeline(self):
            try:
                self.result = run_session(cfg, self, backend=backend, stop=self.stop)
                # A wav source ends on its own; the mic only stops when you quit.
                self.status_text = "done · press q to save"
            except Exception as e:
                self.call_from_thread(self.transcript.note, f"fatal: {e}", RED)
                self.status_text = "failed"

        # --- actions ---
        def action_pause(self):
            self.paused = not self.paused

        def action_clear(self):
            self.transcript.wipe()

        def action_quit(self):
            self.stop.set()
            self.exit()

    return TranscriptApp()
