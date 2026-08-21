"""Typer entrypoint: `localtranscription` / `lt`."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple, Optional

import typer
from rich.console import Console
from rich.markup import escape

from .backends import (
    BACKENDS,
    DEFAULT_ALIGNER,
    DEFAULT_ASR,
    DTYPES,
    BackendUnavailable,
    available,
    describe_checkpoint,
    load_backend,
    local_checkpoints,
)
from . import config as cfgfile
from . import paths
from . import quantize as qz
from .sources import cached_threshold, remember_threshold
from .diarize.offline import OfflineConfig

# Tuned clustering policy lives in OfflineConfig; the CLI mirrors its defaults rather
# than restating them. It had already drifted -- --threshold said 0.95 against the
# config's 0.65, and because the flag always wins, the documented value was dead
# everywhere except the tests.
_DIA = OfflineConfig()
from .engine import LANGUAGES, Config, run_session
from .formats import fmt_clock, write_outputs
from .vad import MAX_UTTERANCE_SEC, MIN_SPEECH_SEC, Cadence

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Live local transcription with [b]Qwen3-ASR[/b] + forced alignment.",
)

CONFIG = typer.Option(None, "--config", envvar="LT_CONFIG", metavar="PATH",
                      help="Config file to read "
                           "[dim](default ~/.config/localtranscription/config.toml)[/].")


class _Root(NamedTuple):
    """What the root callback resolved, for `lt config` to report."""
    explicit: Optional[Path]
    path: Optional[Path]
    defaults: dict


def _params(group) -> dict[str, dict[str, str]]:
    """Every command's options, as {written form: parameter name}.

    Read off the built CLI rather than listed here, so the config file is validated
    against the real options and can't drift away from them. Both spellings resolve,
    because both are things a person sees: `--record-dir` is what --help prints,
    `record_dir` is what it's called underneath, and for `lt models` the two differ
    (`--dir` sets `model_dir`). Long options only -- `--no-record` as a key would read
    as "no_record = true means record", which is backwards.
    """
    out: dict[str, dict[str, str]] = {}
    for name, cmd in group.commands.items():
        alias = {}
        for prm in cmd.params:
            if not prm.name:
                continue
            alias[prm.name] = prm.name
            for opt in prm.opts:
                if opt.startswith("--"):
                    alias[opt[2:].replace("-", "_")] = prm.name
        out[name] = alias
    return out


@app.callback()
def _root(ctx: typer.Context, config: Optional[Path] = CONFIG):
    """Live local transcription with [b]Qwen3-ASR[/b] + forced alignment."""
    ctx.obj = _Root(explicit=config, path=None, defaults={})
    # `lt config` is how you inspect a file that may not parse, so it does its own
    # loading and reports the failure instead of being taken down by it.
    if ctx.invoked_subcommand == "config":
        return
    try:
        path = cfgfile.locate(config)
        defaults = cfgfile.default_map(cfgfile.read(path), _params(ctx.command)) \
            if path else {}
    except cfgfile.ConfigError as e:
        raise typer.BadParameter(str(e), param_hint="--config") from e
    # This is the whole mechanism: click consults default_map per parameter during
    # parsing, so a flag that was actually typed still wins, with no plumbing per flag.
    ctx.default_map = defaults
    ctx.obj = _Root(explicit=config, path=path, defaults=defaults)


# Shared across the run commands; typer accepts the same OptionInfo in several signatures.
OUT = typer.Option(paths.out_dir(), "--out", "-o", help="Where transcripts go.")
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
MINSPEECH = typer.Option(MIN_SPEECH_SEC, "--min-speech",
                         help="Voiced audio an utterance needs to count at all. "
                              "[dim]Lower for single words; a spoken \"Claude\" only "
                              "just clears the default.[/]")
FIRST = typer.Option(0.4, "--interim", help="When the first partial fires, and the floor between partials.")
GROWTH = typer.Option(1.6, "--growth", help="Partial spacing growth [dim](1.0 = fixed spacing)[/].")
MAXGAP = typer.Option(3.0, "--max-gap", help="Longest a partial may lag on a long utterance.")
PARTIALS = typer.Option("reencode", "--partials",
                        help="How provisional text is computed: [b]reencode[/b] "
                             "(re-transcribe the prefix; heals, quadratic) or "
                             "[b]stream[/b] (feed only new audio to a cached decoder; "
                             "linear, appends). [dim]stream needs --backend mlx.[/]")
CHUNKSEC = typer.Option(2.0, "--stream-chunk",
                        help="Seconds of audio per streaming decode [dim](--partials stream)[/].")
XDRAFT = typer.Option(False, "--x-partial-draft",
                      help="[b]Experimental.[/] Decode each partial against the previous "
                           "one as a speculative draft: same text, far fewer forward "
                           "passes. [dim]mlx + --partials reencode only.[/]")
REC = typer.Option(True, "--record/--no-record", help="Save per-utterance audio + manifest.")
RECDIR = typer.Option(paths.record_dir(), "--record-dir", help="Where recordings go.")


def _config(*, out, language, device, mic, wav, threshold, first, growth, max_gap,
            record, record_dir, backend, model, aligner, dtype, partials,
            stream_chunk, min_speech=MIN_SPEECH_SEC, x_partial_draft=False) -> Config:
    """Validate CLI values and build a Config.

    Keyword-only: seventeen positional arguments in the same order at two call sites is a
    silent mis-assignment waiting to happen, and the string-typed ones (model, aligner,
    dtype, partials) would swap without a TypeError.
    """
    if language not in LANGUAGES:
        raise typer.BadParameter(f"{language!r} not supported. Try `lt languages`.")
    if backend not in BACKENDS:
        raise typer.BadParameter(f"{backend!r} unknown. Try `lt backends`.")
    if dtype not in ("auto", *DTYPES):
        raise typer.BadParameter(f"{dtype!r} unknown. Choose auto, {', '.join(DTYPES)}.")
    if partials not in ("reencode", "stream"):
        raise typer.BadParameter(f"{partials!r} unknown. Choose reencode or stream.")
    if stream_chunk <= 0:
        raise typer.BadParameter("--stream-chunk must be positive.")
    # 0 is meaningful -- every opened utterance counts -- but negative is a typo, and
    # an utterance can't outlast the 30s cap.
    if not 0 <= min_speech < MAX_UTTERANCE_SEC:
        raise typer.BadParameter(
            f"--min-speech must be between 0 and {MAX_UTTERANCE_SEC:g} seconds."
        )
    # Checkpoint refs are resolved in backends.resolve_checkpoint, which knows about the
    # checkpoint directory. An earlier guard here only fired when the ref's *parent*
    # existed, so a stale `models/foo` sailed past it and died as a Hub 401.
    cadence = Cadence(first=first, growth=growth, max_gap=max_gap) if first > 0 else None
    return Config(
        out_dir=out, language=language, device=device, backend=backend, model=model,
        aligner=aligner, dtype=dtype, mic=mic, wav=wav, threshold=threshold,
        cadence=cadence, record=record, record_dir=record_dir,
        partials=partials, stream_chunk_sec=stream_chunk, min_speech=min_speech,
        x_partial_draft=x_partial_draft,
    )


def _load(cfg: Config, out: Console = console, on_status=None):
    """Load the chosen backend behind a status spinner, failing with a usable message.

    on_status replaces the spinner rather than adding to it: a caller whose stderr carries
    a machine-readable stream can't also have a spinner redrawing over it.
    """
    def go(status):
        return load_backend(
            cfg.backend, model=cfg.model, aligner=cfg.aligner,
            device=cfg.device, dtype=cfg.dtype, on_status=status,
            # No aligner load at all when nothing will ask for word timings.
            align=cfg.timestamps,
        )

    try:
        if on_status is not None:
            return go(on_status)
        with out.status("[dim]loading model…[/]") as st:
            return go(lambda m: st.update(f"[dim]{m}…[/]"))
    except BackendUnavailable as e:
        raise typer.BadParameter(str(e)) from e


def _report(cfg: Config, segments, words, recorder):
    if not segments:
        console.print("[yellow]Nothing transcribed.[/]")
        return
    # The session owns its name. Letting write_outputs invent one stamped it at a
    # different moment from the recorder's directory, so the two artifacts for one
    # session could not be matched up by name.
    base = write_outputs(cfg.out_dir, segments, words, stem=cfg.stamped())
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
        console.print(f"  [cyan]{name:<6}[/] {mark}  "
                      f"[dim]default dtype {BACKENDS[name].default_dtype}[/]")


@app.command()
def models(model_dir: Path = typer.Option(paths.models_dir(), "--dir",
                                          help="Where local checkpoints live.")):
    """List local checkpoints available to [b]--model[/b]."""
    console.print("[dim]upstream[/]")
    console.print(f"  [cyan]{DEFAULT_ASR}[/]  [dim]unquantised[/]")
    console.print(f"  [cyan]{DEFAULT_ALIGNER}[/]  [dim]aligner[/]")
    found = local_checkpoints() if model_dir == paths.models_dir() else \
        (sorted(p for p in model_dir.glob("*") if (p / "config.json").exists())
         if model_dir.exists() else [])
    if not found:
        console.print(f"\n[dim]No local checkpoints in {model_dir}. "
                      f"Build one with `lt quantize`.[/]")
        return
    # Name the directory once rather than on every row: these paths are long, and a
    # wrapped list is harder to read than the thing it's listing.
    console.print(f"\n[dim]{model_dir}[/]")
    for p in found:
        tag = describe_checkpoint(str(p)) or "unquantised"
        console.print(f"  [cyan]{p.name}[/]  [dim]{tag} · {qz.size_gb(p):.2f} GB[/]")


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
    min_speakers: int = typer.Option(_DIA.min_speakers, "--min-speakers"),
    max_speakers: int = typer.Option(_DIA.max_speakers, "--max-speakers"),
    threshold: float = typer.Option(_DIA.threshold, "--threshold", help="Cosine distance at which two voices are one person."),
    hop: float = typer.Option(_DIA.hop_sec, "--hop", help="Seconds between analysis windows [dim](lower = finer, slower)[/]."),
    compute_units: str = typer.Option(_DIA.compute_units, "--compute-units", help="ALL, CPU_AND_NE, CPU_AND_GPU or CPU_ONLY."),
    words: Optional[Path] = typer.Option(None, "--words", help="A words.json to label with speakers."),
):
    """Diarize a recording: who spoke when [dim](offline, whole file at once)[/].

    Runs the pyannote community-1 family through CoreML. Measured on an M3 Max: a
    28-minute 3-speaker interview in 25s, [b]67x realtime[/b].
    """
    import time

    from .audio import SAMPLE_RATE
    from .audio import load as load_audio
    from .diarize import label_words
    from .diarize.coreml import COMPUTE_UNITS, DiarizationUnavailable
    from .diarize.offline import OfflineDiarizer

    if compute_units not in COMPUTE_UNITS:
        raise typer.BadParameter(f"{compute_units!r} unknown. Choose from {', '.join(COMPUTE_UNITS)}.")
    cfg = OfflineConfig(hop_sec=hop, threshold=threshold, num_speakers=num_speakers,
                        min_speakers=min_speakers, max_speakers=max_speakers,
                        compute_units=compute_units)
    try:
        with console.status("[dim]loading diarization models…[/]") as st:
            engine = OfflineDiarizer(cfg, on_status=lambda m: st.update(f"[dim]{m}…[/]"))
        signal = load_audio(audio_file)
        dur = len(signal) / SAMPLE_RATE
        with console.status(f"[dim]diarizing {dur/60:.1f} min…[/]"):
            t0 = time.monotonic()
            turns = engine.diarize(signal, SAMPLE_RATE)
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


@app.command("paths")
def paths_():
    """Show where config, transcripts, recordings and checkpoints are kept."""
    rows = [
        ("config", paths.config_file(), "--config"),
        ("transcripts", paths.out_dir(), "--out"),
        ("recordings", paths.record_dir(), "--record-dir"),
        ("checkpoints", paths.models_dir(), "--model"),
    ]
    for label, path, flag in rows:
        mark = "[green]✓[/]" if path.exists() else "[dim]· (not yet created)[/]"
        console.print(f"  [cyan]{label}[/] [dim]({flag})[/]  {mark}")
        console.print(f"    {path}", highlight=False)
    console.print(
        "\n[dim]XDG_CONFIG_HOME / XDG_DATA_HOME / XDG_CACHE_HOME move these. Checkpoints\n"
        "live under the cache because `lt quantize` rebuilds them; transcripts and\n"
        "recordings do not.[/]"
    )


@app.command("config")
def config_(
    ctx: typer.Context,
    init: bool = typer.Option(False, "--init", help="Write a starter config file."),
    edit: bool = typer.Option(False, "--edit",
                              help="Open it in [b]$EDITOR[/b], creating it if needed."),
):
    """Show the config file and what it sets [dim](--init to start one)[/].

    The file only changes what a flag [b]defaults[/b] to; a flag you type still wins.
    Bare keys are for `tui`, `cli` and `dictate`; other commands take a table:

        backend = "mlx"
        language = "Greek"

        [dictate]
        hold = true
    """
    root: _Root = ctx.obj
    path = root.explicit.expanduser() if root.explicit else paths.config_file()

    if init or (edit and not path.exists()):
        try:
            cfgfile.write_template(path)
        except cfgfile.ConfigError as e:
            raise typer.BadParameter(str(e)) from e
        console.print(f"[green]wrote[/] {path}")
    if edit:
        import click

        click.edit(filename=str(path))

    if not path.is_file():
        console.print(f"[dim]{path}[/]\n[yellow]no config file[/] "
                      f"[dim]— `lt config --init` starts one[/]")
        raise typer.Exit(0 if init else 1)

    console.print(f"[green]✓[/] {path}", highlight=False)
    try:
        defaults = cfgfile.default_map(cfgfile.read(path), _params(ctx.parent.command))
    except cfgfile.ConfigError as e:
        # Not BadParameter: `lt config` on a broken file should read as a report about
        # that file, not as a misuse of `lt config`. escape() because these messages
        # quote section names, and [dictate] is also rich markup.
        console.print(f"\n[red]{escape(str(e))}[/]")
        raise typer.Exit(1) from e
    if not defaults:
        console.print("[dim]sets nothing — every line is commented out.[/]")
        return
    group = ctx.parent.command
    for cmd, values in sorted(defaults.items()):
        console.print(f"\n[b]lt {cmd}[/]")
        by_name = {prm.name: prm for prm in group.commands[cmd].params}
        for name, value in sorted(values.items()):
            console.print(f"  [cyan]{_shown(by_name[name], value)}[/]", highlight=False)


def _shown(prm, value) -> str:
    """A config entry as the command line it stands in for.

    Worth the few lines: `--hold True` is not a thing you could type, and the whole
    claim this file makes is that it does what typing the flag would.
    """
    opt = max((o for o in prm.opts if o.startswith("--")), key=len, default=prm.name)
    if not isinstance(value, bool):
        return f"{opt} {value}"
    if value:
        return opt
    # An on/off pair has a real name for off; a lone flag can only be described.
    return next((o for o in prm.secondary_opts if o.startswith("--")), f"{opt} off")


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
    aligner: str = ALIGNER, dtype: str = DTYPE, partials: str = PARTIALS,
    stream_chunk: float = CHUNKSEC, min_speech: float = MINSPEECH,
    x_partial_draft: bool = XDRAFT,
):
    """Full-screen live view [dim](q quit · p pause · c clear)[/]."""
    from .tui import build_tui

    cfg = _config(
        out=out, language=language, device=device, mic=mic, wav=wav, threshold=threshold,
        first=first, growth=growth, max_gap=max_gap, record=record, record_dir=record_dir,
        backend=backend, model=model, aligner=aligner, dtype=dtype, partials=partials,
        stream_chunk=stream_chunk, min_speech=min_speech,
        x_partial_draft=x_partial_draft,
    )
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
    aligner: str = ALIGNER, dtype: str = DTYPE, partials: str = PARTIALS,
    stream_chunk: float = CHUNKSEC, min_speech: float = MINSPEECH,
    x_partial_draft: bool = XDRAFT,
):
    """Stream transcriptions to stdout [dim](Ctrl-C to stop)[/]."""
    from rich.live import Live
    from rich.text import Text

    cfg = _config(
        out=out, language=language, device=device, mic=mic, wav=wav, threshold=threshold,
        first=first, growth=growth, max_gap=max_gap, record=record, record_dir=record_dir,
        backend=backend, model=model, aligner=aligner, dtype=dtype, partials=partials,
        stream_chunk=stream_chunk, min_speech=min_speech,
        x_partial_draft=x_partial_draft,
    )

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


# ------------------------------------------------------------------ dictate

WAIT = typer.Option(8.0, "--wait",
                    help="Give up if speech hasn't started within this many seconds "
                         "[dim](0 = wait forever)[/].")
EVENTS = typer.Option(False, "--events",
                      help="Emit JSON lines on stderr: levels, state, text.")
RECAL = typer.Option(False, "--recalibrate",
                     help="Re-measure the room instead of reusing the cached threshold.")
DICT_REC = typer.Option(False, "--record/--no-record",
                        help="Save the utterance's audio + manifest.")
DICT_FIRST = typer.Option(0.0, "--interim",
                          help="Emit provisional text this many seconds in "
                               "[dim](0 = off; only useful with --events)[/].")
HOLD = typer.Option(False, "--hold",
                    help="Keep listening through pauses until signalled, instead of "
                         "stopping at the first one. [dim]For hold-to-talk.[/]")


class _Events:
    """JSON lines on stderr.

    The split is the whole interface: **stdout is the transcript and nothing else**, so
    `lt dictate | pbcopy` works with no flags, while anything that wants a level meter or
    a state machine subscribes to stderr without disturbing that.
    """

    def __init__(self, on: bool):
        self.on = on

    def __call__(self, event: str, **fields):
        if not self.on:
            return
        sys.stderr.write(json.dumps({"event": event, **fields}) + "\n")
        sys.stderr.flush()


@app.command()
def dictate(
    language: str = LANG, mic: Optional[int] = MIC, wav: Optional[Path] = WAV,
    threshold: Optional[float] = THRESH, recalibrate: bool = RECAL, wait: float = WAIT,
    events: bool = EVENTS, interim: float = DICT_FIRST, hold: bool = HOLD,
    record: bool = DICT_REC, record_dir: Path = RECDIR, backend: str = BACKEND,
    model: str = MODEL, dtype: str = DTYPE, device: str = DEVICE,
    min_speech: float = MINSPEECH, x_partial_draft: bool = XDRAFT,
):
    """Speech to stdout, then exit. [dim]A surface to compose on.[/]

    Talk; stop talking; the text is on stdout. Nothing else ever is — status goes to
    stderr — so it pipes:

        lt dictate | pbcopy
        lt dictate | tee -a ~/notes.md

    [b]SIGINT means "I stopped talking"[/], not "abort": the utterance in progress is
    still transcribed and printed. That is what makes hold-to-talk work from any hotkey
    manager with no daemon and no protocol — start it on key down, `kill -INT` it on key
    up.

    Two ways to decide when you're done, and a key-driven one wants the second:

    [b]default[/] — the pause ends it. The VAD closes an utterance after 750ms of silence
    and that is the whole result. Right for a bare `lt dictate | pbcopy` with nothing
    driving it.

    [b]--hold[/] — the signal ends it. Pauses no longer stop anything, so you can think
    mid-sentence while still holding the key; every utterance is transcribed as it
    closes and they're joined on release. Costs nothing in latency: only the last one
    is still outstanding when the signal lands.

    Exits 1 with nothing on stdout if nothing was heard, so `||` works.
    """
    err = Console(stderr=True)
    emit = _Events(events)

    cfg = _config(
        out=paths.out_dir(), language=language, device=device, mic=mic, wav=wav,
        threshold=threshold, first=interim, growth=1.6, max_gap=3.0,
        record=record, record_dir=record_dir, backend=backend, model=model,
        aligner=DEFAULT_ALIGNER, dtype=dtype, partials="reencode", stream_chunk=2.0,
        min_speech=min_speech, x_partial_draft=x_partial_draft,
    )
    # Dictation wants a string, not a transcript. This is what skips loading the 0.6B
    # aligner as well as running it -- see load_backend(align=...).
    cfg.timestamps = False
    # One utterance is the entire result here, so it's worth waiting out. The session
    # default assumes a lost final is one among many.
    cfg.shutdown_timeout = 15.0

    # A threshold describes a room and a microphone, and measuring one costs more than
    # loading the model does -- which would make it the reason dictation felt slow.
    # Never reuse one across --wav, whose "room" is a file.
    if cfg.threshold is None and not recalibrate and wav is None:
        cfg.threshold = cached_threshold(mic)
    measuring = cfg.threshold is None

    t0 = time.monotonic()
    emit("loading", model=cfg.model, backend=cfg.backend)
    backend_obj = _load(cfg, err, on_status=(lambda m: emit("loading", detail=m))
                        if events else None)
    load_took = time.monotonic() - t0

    class Hooks:
        def __init__(self):
            self.stop = None
            self.parts = []
            self.heard = False
            self.deadline = None

        def status(self, msg):
            emit("status", detail=msg)

        def ready(self, threshold):
            if measuring and wav is None:
                remember_threshold(mic, threshold)
            self.deadline = time.monotonic() + wait if wait > 0 else None
            emit("ready", threshold=round(threshold, 6), calibrated=measuring,
                 load=round(load_took, 3))
            if not events:
                err.print(f"[b]Speak.[/] [dim](threshold {threshold:.5f}, "
                          f"loaded in {load_took:.1f}s)[/]")

        def bind_stop(self, stop):
            self.stop = stop
            # Flip the flag rather than raising: segment_utterances flushes whatever it
            # was holding when the frames run out, so the utterance still lands.
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop.set())

        def level(self, rms, in_speech):
            if in_speech and not self.heard:
                self.heard = True
                emit("speech")
            emit("level", rms=round(rms, 5), speech=in_speech)
            if not self.heard and self.deadline and time.monotonic() > self.deadline:
                self.stop.set()

        def interim(self, seg):
            emit("partial", text=seg.text)

        def segment(self, seg):
            self.parts.append(seg.text)
            emit("final", text=seg.text, took=round(seg.took, 3),
                 audio=round(seg.audio_sec, 3))
            if not events:
                err.print(f"[dim]{seg.took:.2f}s for {seg.audio_sec:.1f}s of audio[/]")
            # Under --hold the pause is just a pause: whoever is holding the key decides
            # when this ends, and utterances closed along the way are pieces of one
            # dictation. Otherwise the first pause is the whole job.
            if not hold:
                self.stop.set()

        def error(self, offset, msg):
            emit("error", message=msg)
            if not events:
                err.print(f"[red]transcribe failed: {msg}[/]")

    hooks = Hooks()
    run_session(cfg, hooks, backend=backend_obj)

    # Joined with a space, not a newline: under --hold these are pauses inside one
    # dictation, not separate lines, and the destination is a text field.
    text = " ".join(p.strip() for p in hooks.parts if p.strip())
    if not text:
        emit("empty")
        if not events:
            err.print("[yellow]nothing heard[/]")
        raise typer.Exit(1)

    # No trailing newline when piped: the caller is pasting this into a text field, and a
    # stray newline sends the message. A terminal still gets one so the prompt lands right.
    sys.stdout.write(f"{text}\n" if sys.stdout.isatty() else text)
    sys.stdout.flush()


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
