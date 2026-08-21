"""Typer entrypoint: `localtranscription` / `lt`."""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .backends import (
    BACKENDS,
    DEFAULT_ALIGNER,
    DEFAULT_ASR,
    DEFAULT_DTYPE,
    DTYPES,
    BackendUnavailable,
    available,
    describe_checkpoint,
    load_backend,
)
from . import quantize as qz
from .engine import LANGUAGES, Config, run_session
from .formats import fmt_clock, write_outputs
from .vad import Cadence

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Live local transcription with [b]Qwen3-ASR[/b] + forced alignment.",
)

# Shared across the run commands; typer accepts the same OptionInfo in several signatures.
OUT = typer.Option(Path("out"), "--out", "-o", help="Where transcripts go.")
LANG = typer.Option("English", "--language", "-l", help="ASR language hint.")
DEVICE = typer.Option("mps", "--device", help="torch device [dim](torch backend only)[/].")
BACKEND = typer.Option("torch", "--backend", "-b",
                       help="Inference backend: [b]torch[/b] or [b]mlx[/b].")
MODEL = typer.Option(DEFAULT_ASR, "--model", "-M",
                     help="ASR weights: a HF repo id or a local directory "
                          "[dim](e.g. one built by `lt quantize`)[/].")
ALIGNER = typer.Option(DEFAULT_ALIGNER, "--aligner",
                       help="Forced-aligner weights: HF repo id or local directory.")
DTYPE = typer.Option("auto", "--dtype",
                     help="Compute precision: [b]auto[/b], bf16, fp16 or fp32. "
                          "[dim]auto = bf16 on torch, fp16 on mlx.[/]")
MIC = typer.Option(None, "--mic", "-m", help="Input device index.")
WAV = typer.Option(None, "--wav", help="Replay a 16kHz wav instead of the mic.")
THRESH = typer.Option(None, "--threshold", "-t", help="RMS VAD threshold [dim](auto)[/].")
FIRST = typer.Option(0.4, "--interim", help="When the first partial fires, and the floor between partials.")
GROWTH = typer.Option(1.6, "--growth", help="Partial spacing growth [dim](1.0 = fixed spacing)[/].")
MAXGAP = typer.Option(3.0, "--max-gap", help="Longest a partial may lag on a long utterance.")
REC = typer.Option(True, "--record/--no-record", help="Save per-utterance audio + manifest.")
RECDIR = typer.Option(Path("recordings"), "--record-dir", help="Where recordings go.")


def _config(out, language, device, mic, wav, threshold, first, growth, max_gap, record,
            record_dir, backend="torch", model=DEFAULT_ASR, aligner=DEFAULT_ALIGNER,
            dtype="auto") -> Config:
    if language not in LANGUAGES:
        raise typer.BadParameter(f"{language!r} not supported. Try `lt languages`.")
    if backend not in BACKENDS:
        raise typer.BadParameter(f"{backend!r} unknown. Try `lt backends`.")
    if dtype not in ("auto", *DTYPES):
        raise typer.BadParameter(f"{dtype!r} unknown. Choose auto, {', '.join(DTYPES)}.")
    # A local path that doesn't exist is a typo, not a repo id -- catching it here beats a
    # hub 404 after the spinner has been up for a while.
    for label, ref in (("--model", model), ("--aligner", aligner)):
        if ("/" in ref or ref.startswith(".")) and Path(ref).parent.exists() \
                and not Path(ref).exists() and Path(ref).parts[0] not in ("Qwen",):
            raise typer.BadParameter(f"{label} {ref!r} looks like a path but doesn't exist.")
    cadence = Cadence(first=first, growth=growth, max_gap=max_gap) if first > 0 else None
    return Config(
        out_dir=out, language=language, device=device, backend=backend, model=model,
        aligner=aligner, dtype=dtype, mic=mic, wav=wav, threshold=threshold,
        cadence=cadence, record=record, record_dir=record_dir,
    )


def _load(cfg: Config):
    """Load the chosen backend behind a status spinner, failing with a usable message."""
    try:
        with console.status("[dim]loading model…[/]") as st:
            return load_backend(
                cfg.backend, model=cfg.model, aligner=cfg.aligner,
                device=cfg.device, dtype=cfg.dtype,
                on_status=lambda m: st.update(f"[dim]{m}…[/]"),
            )
    except BackendUnavailable as e:
        raise typer.BadParameter(str(e)) from e


def _report(cfg: Config, segments, words, recorder):
    if not segments:
        console.print("[yellow]Nothing transcribed.[/]")
        return
    base = write_outputs(cfg.out_dir, segments, words)
    console.print(
        f"\n[green]{len(segments)}[/] utterances, [green]{len(words)}[/] timed words"
        f" → [b]{base}[/b].{{txt,words.json,srt,timestamped.md}}"
    )
    if recorder:
        console.print(f"[dim]recorded {recorder.n} utterances → {recorder.dir}[/]")


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
def backends():
    """List inference backends and whether their dependencies are installed."""
    for name in BACKENDS:
        ok = available(name)
        mark = "[green]✓ installed[/]" if ok else "[yellow]not installed[/]"
        console.print(f"  [cyan]{name:<6}[/] {mark}  [dim]default dtype {DEFAULT_DTYPE[name]}[/]")


@app.command()
def models(model_dir: Path = typer.Option(Path("models"), "--dir",
                                          help="Where local checkpoints live.")):
    """List local checkpoints available to [b]--model[/b]."""
    console.print(f"  [cyan]{DEFAULT_ASR}[/]  [dim]upstream · unquantised[/]")
    console.print(f"  [cyan]{DEFAULT_ALIGNER}[/]  [dim]upstream · aligner[/]")
    found = sorted(p for p in model_dir.glob("*") if (p / "config.json").exists()) \
        if model_dir.exists() else []
    if not found:
        console.print(f"\n[dim]No local checkpoints in {model_dir}/. "
                      f"Build one with `lt quantize`.[/]")
        return
    console.print()
    for p in found:
        tag = describe_checkpoint(str(p)) or "unquantised"
        console.print(f"  [cyan]{p}[/]  [dim]{tag} · {qz.size_gb(p):.2f} GB[/]")


@app.command()
def quantize(
    model: str = typer.Argument(DEFAULT_ASR, help="Source weights: HF repo id or directory."),
    bits: int = typer.Option(8, "--bits", help="Affine width: 2, 3, 4, 5, 6 or 8."),
    group_size: int = typer.Option(64, "--group-size", help="Affine group size: 32, 64 or 128."),
    mode: str = typer.Option("affine", "--mode",
                             help="affine, mxfp4, mxfp8 or nvfp4 [dim](float modes fix bits and group size)[/]."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Destination [dim](default models/<name>-<tag>)[/]."),
):
    """Build a quantised MLX checkpoint, then run it with [b]--backend mlx -M <dir>[/b].

    The largest lever on this machine. Measured on an M3 Max against the 1.7B, per
    transcribe() call: an 8-bit checkpoint is [b]2.2x faster than the bf16/MPS default[/b]
    and 4.8x faster than the same weights in MLX fp16, for +0.04pp WER upstream. 4-bit
    buys another ~1.7x on long clips at +0.43pp.
    """
    if not available("mlx"):
        raise typer.BadParameter(
            "quantisation needs mlx-qwen3-asr; install it with `uv sync --extra mlx`"
        )
    if mode not in qz.MODES:
        raise typer.BadParameter(f"{mode!r} unknown. Choose from {', '.join(qz.MODES)}.")
    try:
        with console.status("[dim]quantising…[/]") as st:
            dest = qz.quantize(model, bits=bits, group_size=group_size, mode=mode, out=out,
                               on_status=lambda m: st.update(f"[dim]{m}…[/]"))
    except (ValueError, OSError) as e:
        raise typer.BadParameter(str(e)) from e
    # The aligner is the same architecture with a classification head, so it quantises
    # through the same path -- but it is passed with a different flag.
    flag = "--aligner" if "aligner" in model.lower() else "-M"
    console.print(
        f"[green]{dest}[/]  [dim]{describe_checkpoint(str(dest))} · "
        f"{qz.size_gb(dest):.2f} GB[/]\n"
        f"[dim]run it:[/] lt tui --backend mlx {flag} {dest}"
    )


@app.command()
def diarize(
    audio_file: Path = typer.Argument(..., help="Audio to diarize (wav, flac, m4a, mp3, mp4)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write RTTM here [dim](default: stdout only)[/]."),
    num_speakers: Optional[int] = typer.Option(None, "--speakers", "-n", help="Exact speaker count, if known."),
    min_speakers: int = typer.Option(1, "--min-speakers"),
    max_speakers: int = typer.Option(8, "--max-speakers"),
    threshold: float = typer.Option(0.95, "--threshold", help="Cosine distance at which two voices are one person."),
    hop: float = typer.Option(1.0, "--hop", help="Seconds between analysis windows [dim](lower = finer, slower)[/]."),
    compute_units: str = typer.Option("ALL", "--compute-units", help="ALL, CPU_AND_NE, CPU_AND_GPU or CPU_ONLY."),
    words: Optional[Path] = typer.Option(None, "--words", help="A words.json to label with speakers."),
):
    """Diarize a recording: who spoke when [dim](offline, whole file at once)[/].

    Runs the pyannote community-1 family through CoreML. Measured on an M3 Max: a
    28-minute 3-speaker interview in 25s, [b]67x realtime[/b].
    """
    import time

    from .audio import load as load_audio
    from .diarize import label_words
    from .diarize.coreml import COMPUTE_UNITS, DiarizationUnavailable
    from .diarize.offline import OfflineConfig, OfflineDiarizer

    if compute_units not in COMPUTE_UNITS:
        raise typer.BadParameter(f"{compute_units!r} unknown. Choose from {', '.join(COMPUTE_UNITS)}.")
    cfg = OfflineConfig(hop_sec=hop, threshold=threshold, num_speakers=num_speakers,
                        min_speakers=min_speakers, max_speakers=max_speakers,
                        compute_units=compute_units)
    try:
        with console.status("[dim]loading diarization models…[/]") as st:
            engine = OfflineDiarizer(cfg, on_status=lambda m: st.update(f"[dim]{m}…[/]"))
        signal = load_audio(audio_file)
        dur = len(signal) / 16000
        with console.status(f"[dim]diarizing {dur/60:.1f} min…[/]"):
            t0 = time.monotonic()
            turns = engine.diarize(signal, 16000)
            took = time.monotonic() - t0
    except (DiarizationUnavailable, FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e

    speakers = sorted({t.speaker for t in turns})
    console.print(
        f"[green]{len(speakers)}[/] speakers, [green]{len(turns)}[/] turns "
        f"[dim]({dur/60:.1f} min in {took:.1f}s · {dur/took:.0f}x realtime)[/]"
    )
    for spk in speakers:
        held = sum(t.duration for t in turns if t.speaker == spk)
        console.print(f"  [cyan]speaker {spk}[/]  {held/60:5.2f} min  [dim]{held/dur*100:4.1f}%[/]")
    for t in turns:
        console.print(f"  [dim]{fmt_clock(t.start)}-{fmt_clock(t.end)}[/]  speaker {t.speaker}",
                      highlight=False)

    if out:
        from .formats import rttm

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rttm(turns, audio_file.stem))
        console.print(f"\n[dim]→ {out}[/]")

    if words:
        import json

        labelled = label_words(json.loads(words.read_text()), turns)
        dest = words.with_suffix(".speakers.json")
        dest.write_text(json.dumps(labelled, indent=2))
        console.print(f"[dim]→ {dest}  ({sum(w['speaker'] is not None for w in labelled)}"
                      f"/{len(labelled)} words labelled)[/]")


@app.command()
def cadence(
    length: float = typer.Argument(20.0, help="Utterance length to simulate, in seconds."),
    first: float = FIRST,
    growth: float = GROWTH,
    max_gap: float = MAXGAP,
):
    """Show when partials fire, and what the schedule costs.

    Cost is the audio re-encoded across all passes: every partial re-processes its whole
    prefix, so the schedule -- not the model -- decides whether that stays linear.
    """
    c = Cadence(first=first, growth=growth, max_gap=max_gap)
    points = c.schedule(length)
    total = sum(points) + length  # partials plus the final pass
    console.print(f"partials at: [cyan]{', '.join(f'{p:g}s' for p in points)}[/]")
    console.print(f"count: [b]{len(points)}[/]   first text after [b]{points[0] if points else length:g}s[/]")
    console.print(
        f"audio processed: [b]{total:.1f}s[/] for a {length:g}s utterance"
        f"  [dim](x{total / length:.1f} realtime)[/]"
    )


@app.command()
def tui(
    out: Path = OUT, language: str = LANG, device: str = DEVICE, mic: Optional[int] = MIC,
    wav: Optional[Path] = WAV, threshold: Optional[float] = THRESH, first: float = FIRST,
    growth: float = GROWTH, max_gap: float = MAXGAP, record: bool = REC,
    record_dir: Path = RECDIR, backend: str = BACKEND, model: str = MODEL,
    aligner: str = ALIGNER, dtype: str = DTYPE,
):
    """Full-screen live view [dim](q quit · p pause · c clear)[/]."""
    from .tui import build_tui

    cfg = _config(out, language, device, mic, wav, threshold, first, growth, max_gap,
                  record, record_dir, backend, model, aligner, dtype)
    # Load before entering full-screen: subprocess spawning breaks under Textual's stdout.
    ui = build_tui(cfg, _load(cfg))
    ui.run()
    # collected(), not result: run_session may not have returned yet when the UI closes,
    # and everything already transcribed is sitting on the worker.
    _report(cfg, *ui.collected())
    # main() hard-exits from here, which also covers a wedged inference still sitting on
    # Textual's pool thread.


@app.command()
def cli(
    out: Path = OUT, language: str = LANG, device: str = DEVICE, mic: Optional[int] = MIC,
    wav: Optional[Path] = WAV, threshold: Optional[float] = THRESH, first: float = FIRST,
    growth: float = GROWTH, max_gap: float = MAXGAP, record: bool = REC,
    record_dir: Path = RECDIR, backend: str = BACKEND, model: str = MODEL,
    aligner: str = ALIGNER, dtype: str = DTYPE,
):
    """Stream transcriptions to stdout [dim](Ctrl-C to stop)[/]."""
    from rich.live import Live
    from rich.text import Text

    cfg = _config(out, language, device, mic, wav, threshold, first, growth, max_gap,
                  record, record_dir, backend, model, aligner, dtype)

    # Provisional text rewrites itself in place, which needs a terminal that can take the
    # line back. Piped to a file, finals-only keeps the output clean.
    show_interim = sys.stdout.isatty() and cfg.cadence is not None
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

    # Load before the Live region starts: loading writes its own progress bars, and two
    # things driving the cursor at once garbles both.
    backend_obj = _load(cfg)

    with (live if show_interim else contextlib.nullcontext()):
        result = run_session(cfg, Hooks(), backend=backend_obj)
    _report(cfg, *result)


def main():
    """Entrypoint that always leaves immediately.

    Model loading spins up ThreadPoolExecutors inside huggingface/transformers, and
    concurrent.futures registers an atexit hook that joins those workers *non-daemonically*.
    Ctrl-C during load therefore hangs on shutdown -- the interpreter is stuck in
    `_python_exit -> t.join()` with all our own work already finished. Nothing useful
    happens after a command returns, so exit hard rather than let atexit block.
    """
    code = 0
    try:
        app()
    except SystemExit as e:  # click's normal exit path
        code = e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        code = 130
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
