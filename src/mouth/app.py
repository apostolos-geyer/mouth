"""Typer entrypoint: `m`."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from . import config as cfgfile
from . import paths
from . import quantize as qz
from .backends import (
    BACKENDS,
    DEFAULT_ALIGNER,
    DEFAULT_ASR,
    DTYPES,
    PARTIAL_MODES,
    Aligning,
    BackendUnavailable,
    available,
    describe_checkpoint,
    load_backend,
    local_checkpoints,
    open_partials,
    resolve_checkpoint,
)
from .diarize.offline import OfflineConfig
from .sources import cached_threshold, remember_threshold

# Tuned clustering policy lives in OfflineConfig; the CLI mirrors its defaults rather
# than restating them. It had already drifted -- --threshold said 0.95 against the
# config's 0.65, and because the flag always wins, the documented value was dead
# everywhere except the tests.
_DIA = OfflineConfig()
from .diarize import speaker_changes
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

CONFIG = typer.Option(
    None,
    "--config",
    envvar="LT_CONFIG",
    metavar="PATH",
    help="Config file to read [dim](default ~/.config/mouth/config.toml)[/].",
)


def _config_path(ctx: typer.Context) -> Path:
    """The config file this invocation is about: --config if given, else the default.

    Shared because `m tune --write` grew its own copy and then ignored the flag: it read
    defaults out of the file it was pointed at and wrote its result somewhere else.
    """
    explicit = ctx.obj if isinstance(ctx.obj, Path) else None
    return explicit.expanduser() if explicit else paths.config_file()


def _params(group) -> dict[str, dict[str, str]]:
    """Every command's options, as {written form: parameter name}.

    Read off the built CLI rather than listed here, so the config file is validated
    against the real options and can't drift away from them. Both spellings resolve,
    because both are things a person sees: `--record-dir` is what --help prints,
    `record_dir` is what it's called underneath, and for `m models` the two differ
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
                    # cfgfile._norm, not a second copy: these keys and the ones read out
                    # of the file have to normalise identically or a real option silently
                    # fails to resolve.
                    alias[cfgfile._norm(opt[2:])] = prm.name
        out[name] = alias
    return out


@app.callback()
def _root(ctx: typer.Context, config: Path | None = CONFIG):
    """Live local transcription with [b]Qwen3-ASR[/b] + forced alignment."""
    # The --config path, for the commands that need to know which file was meant. Only
    # that: `m config` re-reads and re-parses on purpose, because it has to survive a
    # file too broken for this callback to have parsed at all.
    ctx.obj = config
    # `m config` is how you inspect a file that may not parse, so it does its own
    # loading and reports the failure instead of being taken down by it.
    if ctx.invoked_subcommand == "config":
        return
    try:
        path = cfgfile.locate(config)
        defaults = (
            cfgfile.default_map(cfgfile.read(path), _params(ctx.command)) if path else {}
        )
    except cfgfile.ConfigError as e:
        raise typer.BadParameter(str(e), param_hint="--config") from e
    # This is the whole mechanism: click consults default_map per parameter during
    # parsing, so a flag that was actually typed still wins, with no plumbing per flag.
    ctx.default_map = defaults


# Shared across the run commands; typer accepts the same OptionInfo in several signatures.
OUT = typer.Option(paths.out_dir(), "--out", "-o", help="Where transcripts go.")
LANG = typer.Option("English", "--language", "-l", help="ASR language hint.")
DEVICE = typer.Option("mps", "--device", help="torch device [dim](torch backend only)[/].")
BACKEND = typer.Option(
    "torch", "--backend", "-b", help="Inference backend: [b]torch[/b] or [b]mlx[/b]."
)
MODEL = typer.Option(
    DEFAULT_ASR,
    "--model",
    "-M",
    help="ASR weights: a HF repo id or a local directory "
    "[dim](e.g. one built by `m quantize`)[/].",
)
ALIGNER = typer.Option(
    DEFAULT_ALIGNER,
    "--aligner",
    help="Forced-aligner weights: HF repo id or local directory.",
)
DTYPE = typer.Option(
    "auto",
    "--dtype",
    help="Compute precision: [b]auto[/b], bf16, fp16 or fp32. "
    "[dim]auto = bf16 on torch, fp16 on mlx.[/]",
)
MIC = typer.Option(None, "--mic", "-m", help="Input device index.")
WAV = typer.Option(None, "--wav", help="Replay a 16kHz wav instead of the mic.")
THRESH = typer.Option(None, "--threshold", "-t", help="RMS VAD threshold [dim](auto)[/].")
CONTEXT = typer.Option(
    "",
    "--context",
    metavar="TEXT",
    help="Words to expect: names, jargon, spellings. "
    "[dim]The model biases decoding toward them.[/]",
)
MINSPEECH = typer.Option(
    MIN_SPEECH_SEC,
    "--min-speech",
    help="Voiced audio an utterance needs to count at all. "
    '[dim]Lower for single words; a spoken "Claude" only '
    "just clears the default.[/]",
)
FIRST = typer.Option(
    0.4, "--interim", help="When the first partial fires, and the floor between partials."
)
GROWTH = typer.Option(
    1.6, "--growth", help="Partial spacing growth [dim](1.0 = fixed spacing)[/]."
)
MAXGAP = typer.Option(
    3.0, "--max-gap", help="Longest a partial may lag on a long utterance."
)
PARTIALS = typer.Option(
    "reencode",
    "--partials",
    help="How provisional text is computed: [b]reencode[/b] (re-transcribe the prefix; "
    "heals, quadratic), [b]x-draft[/b] (reencode, decoded against the previous pass as "
    "a draft -- same text, far fewer passes; [i]experimental[/i]) or [b]stream[/b] "
    "(feed only new audio to a cached decoder; linear, appends). "
    "[dim]x-draft and stream need --backend mlx.[/]",
)
CHUNKSEC = typer.Option(
    2.0,
    "--stream-chunk",
    help="Seconds of audio per streaming decode [dim](--partials stream)[/].",
)
REC = typer.Option(
    True, "--record/--no-record", help="Save per-utterance audio + manifest."
)
RECDIR = typer.Option(paths.record_dir(), "--record-dir", help="Where recordings go.")


def _config(
    *,
    out,
    language,
    device,
    mic,
    wav,
    threshold,
    first,
    growth,
    max_gap,
    record,
    record_dir,
    backend,
    model,
    aligner,
    dtype,
    partials,
    stream_chunk,
    min_speech=MIN_SPEECH_SEC,
    context="",
    session_id="",
) -> Config:
    """Validate CLI values and build a Config.

    Keyword-only: seventeen positional arguments in the same order at two call sites is a
    silent mis-assignment waiting to happen, and the string-typed ones (model, aligner,
    dtype, partials) would swap without a TypeError.
    """
    if language not in LANGUAGES:
        raise typer.BadParameter(f"{language!r} not supported. Try `m languages`.")
    if backend not in BACKENDS:
        raise typer.BadParameter(f"{backend!r} unknown. Try `m backends`.")
    if dtype not in ("auto", *DTYPES):
        raise typer.BadParameter(f"{dtype!r} unknown. Choose auto, {', '.join(DTYPES)}.")
    if partials not in PARTIAL_MODES:
        raise typer.BadParameter(
            f"{partials!r} unknown. Choose {', '.join(PARTIAL_MODES)}."
        )
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
        out_dir=out,
        language=language,
        device=device,
        backend=backend,
        model=model,
        aligner=aligner,
        dtype=dtype,
        mic=mic,
        wav=wav,
        threshold=threshold,
        cadence=cadence,
        record=record,
        record_dir=record_dir,
        partials=partials,
        stream_chunk_sec=stream_chunk,
        min_speech=min_speech,
        context=context,
        session_id=session_id,
    )


def _load(cfg: Config, out: Console = console, on_status=None, align: bool | None = None):
    """Load the chosen backend behind a status spinner, failing with a usable message.

    on_status replaces the spinner rather than adding to it: a caller whose stderr carries
    a machine-readable stream can't also have a spinner redrawing over it.
    """

    def go(status):
        return load_backend(
            cfg.backend,
            model=cfg.model,
            aligner=cfg.aligner,
            device=cfg.device,
            dtype=cfg.dtype,
            on_status=status,
            # No aligner load at all when nothing will ask for word timings. The override
            # is for a caller that wants the aligner without running it per utterance --
            # `m transcribe --speakers` times whole speaker blocks afterwards instead.
            align=cfg.timestamps if align is None else align,
        )

    try:
        if on_status is not None:
            return go(on_status)
        with out.status("[dim]loading model…[/]") as st:
            return go(lambda m: st.update(f"[dim]{m}…[/]"))
    except BackendUnavailable as e:
        raise typer.BadParameter(str(e)) from e


def _report(cfg: Config, segments, words, recorder, turns=None) -> Path | None:
    """Write the artifacts and announce them. Returns the output stem path, or None
    when there was nothing to write -- the file front end turns that into a nonzero
    exit, so a script can tell an empty VAD result from a successful run."""
    if not segments:
        console.print("[yellow]Nothing transcribed.[/]")
        return None
    # The session owns its name. Letting write_outputs invent one stamped it at a
    # different moment from the recorder's directory, so the two artifacts for one
    # session could not be matched up by name.
    base = write_outputs(cfg.out_dir, segments, words, stem=cfg.stamped(), turns=turns)
    kinds = "txt,words.json,srt,timestamped.md"
    if turns:
        kinds += ",rttm,speakers.json,speakers.md"
    console.print(
        f"\n[green]{len(segments)}[/] utterances, [green]{len(words)}[/] timed words"
        f" → [b]{base}[/b].{{{kinds}}}"
    )
    if recorder:
        console.print(f"[dim]recorded {recorder.n} utterances → {recorder.dir}[/]")
    return base


@app.command()
def devices():
    """List available microphones."""
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            console.print(
                f"  [cyan]{i:>2}[/]  {d['name']}  [dim]({d['default_samplerate']:.0f}Hz)[/]"
            )


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
        console.print(
            f"  [cyan]{name:<6}[/] {mark}  "
            f"[dim]default dtype {BACKENDS[name].default_dtype}[/]"
        )


@app.command()
def models(
    model_dir: Path = typer.Option(
        paths.models_dir(), "--dir", help="Where local checkpoints live."
    ),
):
    """List local checkpoints available to [b]--model[/b]."""
    console.print("[dim]upstream[/]")
    console.print(f"  [cyan]{DEFAULT_ASR}[/]  [dim]unquantised[/]")
    console.print(f"  [cyan]{DEFAULT_ALIGNER}[/]  [dim]aligner[/]")
    found = (
        local_checkpoints()
        if model_dir == paths.models_dir()
        else (
            sorted(p for p in model_dir.glob("*") if (p / "config.json").exists())
            if model_dir.exists()
            else []
        )
    )
    if not found:
        console.print(
            f"\n[dim]No local checkpoints in {model_dir}. Build one with `m quantize`.[/]"
        )
        return
    # Name the directory once rather than on every row: these paths are long, and a
    # wrapped list is harder to read than the thing it's listing.
    console.print(f"\n[dim]{model_dir}[/]")
    for p in found:
        tag = describe_checkpoint(str(p)) or "unquantised"
        console.print(f"  [cyan]{p.name}[/]  [dim]{tag} · {qz.size_gb(p):.2f} GB[/]")


@app.command()
def quantize(
    model: str = typer.Argument(
        DEFAULT_ASR, help="Source weights: HF repo id or directory."
    ),
    bits: int = typer.Option(8, "--bits", help="Affine width: 2, 3, 4, 5, 6 or 8."),
    group_size: int = typer.Option(
        64, "--group-size", help="Affine group size: 32, 64 or 128."
    ),
    mode: str = typer.Option(
        "affine",
        "--mode",
        help="affine, mxfp4, mxfp8 or nvfp4 [dim](float modes fix bits and group size)[/].",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Destination [dim](default models/<name>-<tag>)[/]."
    ),
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
            dest = qz.quantize(
                model,
                bits=bits,
                group_size=group_size,
                mode=mode,
                out=out,
                on_status=lambda m: st.update(f"[dim]{m}…[/]"),
            )
    except (ValueError, OSError) as e:
        raise typer.BadParameter(str(e)) from e
    # The aligner is the same architecture with a classification head, so it quantises
    # through the same path -- but it is passed with a different flag.
    flag = "--aligner" if "aligner" in model.lower() else "-M"
    console.print(
        f"[green]{dest}[/]  [dim]{describe_checkpoint(str(dest))} · "
        f"{qz.size_gb(dest):.2f} GB[/]\n"
        f"[dim]run it:[/] m tui --backend mlx {flag} {dest}"
    )


@app.command()
def diarize(
    audio_file: Path = typer.Argument(
        ..., help="Audio to diarize (wav, flac, m4a, mp3, mp4)."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write RTTM here [dim](default: stdout only)[/]."
    ),
    num_speakers: int | None = typer.Option(
        None, "--speakers", "-n", help="Exact speaker count, if known."
    ),
    min_speakers: int = typer.Option(_DIA.min_speakers, "--min-speakers"),
    max_speakers: int = typer.Option(_DIA.max_speakers, "--max-speakers"),
    threshold: float = typer.Option(
        _DIA.threshold,
        "--threshold",
        help="Cosine distance at which two voices are one person.",
    ),
    hop: float = typer.Option(
        _DIA.hop_sec,
        "--hop",
        help="Seconds between analysis windows [dim](lower = finer, slower)[/].",
    ),
    compute_units: str = typer.Option(
        _DIA.compute_units,
        "--compute-units",
        help="ALL, CPU_AND_NE, CPU_AND_GPU or CPU_ONLY.",
    ),
    words: Path | None = typer.Option(
        None, "--words", help="A words.json to label with speakers."
    ),
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
        raise typer.BadParameter(
            f"{compute_units!r} unknown. Choose from {', '.join(COMPUTE_UNITS)}."
        )
    cfg = OfflineConfig(
        hop_sec=hop,
        threshold=threshold,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        compute_units=compute_units,
    )
    try:
        with console.status("[dim]loading diarization models…[/]") as st:
            engine = OfflineDiarizer(cfg, on_status=lambda m: st.update(f"[dim]{m}…[/]"))
        signal = load_audio(audio_file)
        dur = len(signal) / SAMPLE_RATE
        with console.status(f"[dim]diarizing {dur / 60:.1f} min…[/]"):
            t0 = time.monotonic()
            turns = engine.diarize(signal, SAMPLE_RATE)
            took = time.monotonic() - t0
    except (DiarizationUnavailable, FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e

    speakers = sorted({t.speaker for t in turns})
    console.print(
        f"[green]{len(speakers)}[/] speakers, [green]{len(turns)}[/] turns "
        f"[dim]({dur / 60:.1f} min in {took:.1f}s · {dur / took:.0f}x realtime)[/]"
    )
    for spk in speakers:
        held = sum(t.duration for t in turns if t.speaker == spk)
        console.print(
            f"  [cyan]speaker {spk}[/]  {held / 60:5.2f} min  [dim]{held / dur * 100:4.1f}%[/]"
        )
    for t in turns:
        console.print(
            f"  [dim]{fmt_clock(t.start)}-{fmt_clock(t.end)}[/]  speaker {t.speaker}",
            highlight=False,
        )

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
        console.print(
            f"[dim]→ {dest}  ({sum(w['speaker'] is not None for w in labelled)}"
            f"/{len(labelled)} words labelled)[/]"
        )


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
        "live under the cache because `m quantize` rebuilds them; transcripts and\n"
        "recordings do not.[/]"
    )


@app.command("config")
def config_(
    ctx: typer.Context,
    init: bool = typer.Option(False, "--init", help="Write a starter config file."),
    edit: bool = typer.Option(
        False, "--edit", help="Open it in [b]$EDITOR[/b], creating it if needed."
    ),
):
    """Show the config file and what it sets [dim](--init to start one)[/].

    The file only changes what a flag [b]defaults[/b] to; a flag you type still wins.
    Bare keys reach the commands that listen and the two that describe them; anything
    else takes a table named after it:

        backend = "mlx"
        language = "Greek"

        [dictate]
        hold = true
    """
    # This typer vendors its own click core, so TyperGroup is not a click.Group and an
    # isinstance narrowing here is simply false. The attribute is what matters.
    group = ctx.parent.command if ctx.parent else ctx.command
    commands = getattr(group, "commands", {})
    path = _config_path(ctx)

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
        console.print(
            f"[dim]{path}[/]\n[yellow]no config file[/] "
            f"[dim]— `m config --init` starts one[/]"
        )
        raise typer.Exit(0 if init else 1)

    console.print(f"[green]✓[/] {path}", highlight=False)
    try:
        defaults = cfgfile.default_map(cfgfile.read(path), _params(group))
    except cfgfile.ConfigError as e:
        # Not BadParameter: `m config` on a broken file should read as a report about
        # that file, not as a misuse of `m config`. escape() because these messages
        # quote section names, and [dictate] is also rich markup.
        console.print(f"\n[red]{escape(str(e))}[/]")
        raise typer.Exit(1) from e
    if not defaults:
        console.print("[dim]sets nothing — every line is commented out.[/]")
        return
    for cmd, values in sorted(defaults.items()):
        console.print(f"\n[b]m {cmd}[/]")
        by_name = {prm.name: prm for prm in commands[cmd].params}
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
def tune(
    ctx: typer.Context,
    wav: Path | None = typer.Option(
        None, "--wav", help="Measure against a recording instead of the microphone."
    ),
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Save the result without asking [dim](keeps a .bak of the old config)[/].",
    ),
    phrases: int = typer.Option(3, "--phrases", help="How many things to say."),
    mic: int | None = MIC,
    backend: str = BACKEND,
    model: str = MODEL,
    aligner: str = ALIGNER,
    dtype: str = DTYPE,
    language: str = LANG,
):
    """Set this up for your machine and your voice.

    Asks you to say a few things, times how fast your machine transcribes them, and
    offers you a choice about how quickly text should appear while you talk. Everything
    it suggests is measured here, not copied from a table.
    """
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    from . import tune as tn

    cfg = _config(
        out=paths.out_dir(),
        language=language,
        device=DEVICE.default,
        mic=mic,
        wav=None,
        threshold=None,
        first=0.0,
        growth=1.6,
        max_gap=3.0,
        record=False,
        record_dir=paths.record_dir(),
        backend=backend,
        model=model,
        aligner=aligner,
        dtype=dtype,
        partials="reencode",
        stream_chunk=2.0,
    )
    cfg.timestamps = False

    console.print()
    console.print(
        Panel.fit("[b]Setting up for your machine and your voice[/]", border_style="cyan")
    )

    # ------------------------------------------------------------------ 1. machine
    box = tn.machine()
    kit = Table.grid(padding=(0, 2))
    kit.add_column(style="dim", justify="right")
    kit.add_column()
    kit.add_row("computer", f"{box.chip}  ·  {box.memory_gb:g} GB")
    # Resolve first: a bare name like "qwen3-asr-1.7b-q8g64" is not a path, and
    # describe_checkpoint reads the config.json inside the directory.
    quant = describe_checkpoint(resolve_checkpoint(cfg.model))
    kit.add_row(
        "speech model",
        f"{Path(cfg.model).name}" + (f"  [dim]({quant})[/]" if quant else ""),
    )
    console.print("\n[b]1 · What you're running on[/]\n")
    console.print(kit)
    if box.apple_silicon and backend != "mlx":
        console.print(
            "\n  [yellow]This Mac can run about twice as fast on a converted "
            "model.[/]\n  [dim]See `m quantize`.[/]"
        )

    # ------------------------------------------------------------------ 2. listen
    console.print("\n[b]2 · Your voice[/]\n")
    samples = _tune_listen(wav, mic, phrases)
    if not samples:
        raise typer.BadParameter("Nothing was recorded. Try again and speak up.")

    shortest = min(samples, key=lambda s: s.voiced_sec)
    min_speech = tn.suggest_min_speech(samples)
    missed = shortest.voiced_sec < MIN_SPEECH_SEC
    console.print(
        f"\n  The shortest thing you said lasted [b]{shortest.voiced_sec:.2f} seconds[/]."
    )
    console.print(
        "  [green]It will now pick up words that short.[/]"
        + (
            f"\n  [dim]Out of the box it needs {MIN_SPEECH_SEC:g}s and would have missed "
            f"that one.[/]"
            if missed
            else "\n  [dim]That already clears the standard setting.[/]"
        )
    )

    # ------------------------------------------------------------------ 3. measure
    console.print("\n[b]3 · How fast this machine transcribes[/]\n")
    t0 = time.monotonic()
    backend_obj = _load(cfg)
    console.print(f"  [dim]model ready in {time.monotonic() - t0:.1f}s[/]")
    drafted, full, drafting = _tune_measure(cfg, backend_obj, samples)

    # ------------------------------------------------------------------ 4. choose
    REF = 20.0
    verdicts = [tn.evaluate(p, drafted, full, REF) for p in tn.PROFILES]
    best = tn.recommend(verdicts)

    table = Table(header_style="dim", box=None, padding=(0, 3), pad_edge=False)
    table.add_column(" ", no_wrap=True)
    table.add_column("option", style="cyan", no_wrap=True)
    table.add_column("text updates", no_wrap=True)
    table.add_column("lags you by", no_wrap=True)
    table.add_column("effort", no_wrap=True)
    for v in verdicts:
        pick = "[green]★[/]" if v is best else " "
        strain = {
            "easy": "[green]easy[/]",
            "works for it": "some",
            "strained": "[yellow]hard[/]",
            "too much": "[red]can't keep up[/]",
        }[v.headroom]
        table.add_row(
            pick,
            v.profile.name,
            f"every {v.profile.cadence.max_gap:g}s",
            f"up to {v.stale_max:.1f}s",
            strain,
        )
    console.print("\n[b]4 · How quickly should text appear while you talk?[/]\n")
    console.print(table)
    console.print(f"\n  [green]★ {best.profile.name}[/] — {best.profile.blurb}.")
    if drafting and not best.safe_without_drafting:
        console.print(
            "  [dim]Leans on a speed-up that noisy rooms can lose; if it ever "
            "falls behind, pick the option above it.[/]"
        )

    choice = best.profile
    if not write:
        names = [p.name for p in tn.PROFILES]
        picked = Prompt.ask("\n  keep", choices=[*names, "quit"], default=best.profile.name)
        if picked == "quit":
            console.print("  [dim]nothing written[/]")
            return
        choice = next(p for p in tn.PROFILES if p.name == picked)

    text = tn.render(
        backend=cfg.backend,
        model=model,
        aligner=aligner,
        min_speech=min_speech,
        profile=choice,
        drafting=drafting,
        note=f"{box.chip} · measured {time.strftime('%Y-%m-%d')}",
    )
    dest = _config_path(ctx)
    backup = cfgfile.save(dest, text)
    if backup is not None:
        console.print(f"  [dim]previous settings kept at {backup.name}[/]")
    console.print(f"\n  [green]Saved.[/] [dim]{dest}[/]")
    console.print("  [dim]Run `m tui` to use it, or `m config` to see it.[/]\n")


def _tune_listen(wav: Path | None, mic: int | None, phrases: int):
    """Collect a few utterances, from the microphone or a recording.

    Split out so the command above reads as the five steps a person goes through, and so
    the microphone half is one function rather than a branch inside a long body.
    """
    from . import tune as tn
    from .sources import make_source

    if wav is not None:
        src = make_source(None, wav).open()
        try:
            got = list(tn.samples_from(src, src.calibrate(1.0)))
        finally:
            src.close()
        console.print(f"  [dim]{wav.name} · {len(got)} phrases[/]")
        return got

    ASKS = [
        'a single short word — your name, or "okay"',
        "a whole sentence, the way you'd normally talk",
        "one more sentence, a longer one",
    ]
    src = make_source(mic, None).open()
    out = []
    try:
        with console.status("[dim]listening to the room, stay quiet for a second…[/]"):
            threshold = src.calibrate(1.0)
        for i in range(phrases):
            ask = ASKS[i] if i < len(ASKS) else "anything else"
            console.print(f"  [cyan]{i + 1}.[/] Say {ask}. [dim]listening…[/]", end="\r")
            heard = tn.capture(src, threshold)
            if heard is None:
                console.print(f"  [yellow]{i + 1}. didn't catch that[/]{' ' * 40}")
                continue
            out.append(heard)
            console.print(
                f"  [green]{i + 1}. got it[/] [dim]({heard.seconds:.1f}s)[/]{' ' * 40}"
            )
    finally:
        src.close()
    return out


def _tune_measure(cfg, backend_obj, samples):
    """Draw a progress bar over tune.measure(). The loop itself lives in tune.py."""
    from rich.progress import BarColumn, Progress, TextColumn

    from . import tune as tn

    longest = max(samples, key=lambda s: s.seconds)
    steps = len(tn.bench_lengths(longest.seconds))
    drafted_decoder = open_partials(backend_obj, mode="x-draft", language=cfg.language)
    drafter = drafted_decoder if drafted_decoder.mode == "x-draft" else None
    with Progress(
        TextColumn("  [dim]{task.description}[/]"),
        BarColumn(bar_width=28),
        TextColumn("[dim]{task.completed}/{task.total}[/]"),
        console=console,
        transient=True,
    ) as bar:
        job = bar.add_task("timing", total=steps)
        drafted, full = tn.measure(
            backend_obj,
            drafter,
            longest,
            cfg.language,
            on_step=lambda: bar.advance(job),
        )

    one_sec = full.at(1.0)
    console.print(
        f"  A second of speech takes [b]{one_sec:.2f}s[/] to turn into text"
        f"  [dim](about {1 / one_sec:.0f}x faster than real time)[/]"
    )
    slow, fast = full.at(10.0), drafted.at(10.0)
    if drafter is not None and slow > fast * 1.05:
        console.print(
            f"  [dim]An optional speed-up makes long sentences {slow / fast:.1f}x "
            f"cheaper here.[/]"
        )
    return drafted, full, drafter is not None


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(
        ..., help="Audio to transcribe (wav, flac, m4a, mp3, mp4)."
    ),
    out: Path = OUT,
    language: str = LANG,
    context: str = CONTEXT,
    device: str = DEVICE,
    backend: str = BACKEND,
    model: str = MODEL,
    aligner: str = ALIGNER,
    dtype: str = DTYPE,
    threshold: float | None = THRESH,
    min_speech: float = MINSPEECH,
    speakers: bool = typer.Option(
        False, "--speakers", help="Also work out who spoke when, and label the transcript."
    ),
    num_speakers: int | None = typer.Option(
        None,
        "--num-speakers",
        "-n",
        help="Exact speaker count, if you know it [dim](implies --speakers)[/].",
    ),
    record: bool = typer.Option(
        False, "--record/--no-record", help="Save per-utterance audio + manifest."
    ),
    record_dir: Path = RECDIR,
    stem: str | None = typer.Option(
        None,
        "--stem",
        help="Name for the output files [dim](default: session-timestamp)[/].",
    ),
):
    """Transcribe a file, as fast as the machine can [dim](not in real time)[/].

    The same VAD, the same model, the same outputs as a live session — but the audio is
    already on disk, so nothing waits on a clock. Measured on an M3 Max with the 8-bit
    checkpoint, finals run at about [b]17x realtime[/b].

    Partials are off: there is nobody watching text land, and provisional passes are the
    expensive half of a live session.

    With [b]--speakers[/b] it also diarizes and labels the transcript, which is the whole
    job for a recording of more than one person. Same decode, same pass over the file:
    running `m diarize` afterwards would re-read and re-analyse it.
    """
    cfg = _config(
        out=out,
        language=language,
        device=device,
        mic=None,
        wav=audio_file,
        threshold=threshold,
        first=0.0,
        growth=1.6,
        max_gap=3.0,
        record=record,
        record_dir=record_dir,
        backend=backend,
        model=model,
        aligner=aligner,
        dtype=dtype,
        partials="reencode",
        stream_chunk=2.0,
        min_speech=min_speech,
        context=context,
        session_id=stem or "",
    )
    cfg.realtime = False
    # A file arrives faster than the model consumes it, so the queue builds a backlog of
    # finished utterances. The 2s default is sized for quitting a live session, where one
    # abandoned final sits among many; here it silently truncated the transcript -- a
    # 28-minute interview came back as 826 words because the drain gave up. Nothing is
    # waiting on this, and the work is bounded by the file.
    cfg.shutdown_timeout = 3600.0
    if not audio_file.exists():
        raise typer.BadParameter(f"{audio_file}: no such file")

    # Diarize first when asked, so speaker changes can cut utterances. Silence is the
    # only boundary the VAD finds on its own, and a recording with the pauses edited out
    # has almost none: 58 utterances for 28 minutes, each holding several people. The
    # decoded audio is handed to the session so the file is read once.
    turns, source = None, None
    if speakers or num_speakers is not None:
        from .audio import load as load_audio
        from .sources import make_source

        audio = load_audio(audio_file)
        turns = _diarize_audio(audio, num_speakers)
        cfg.cuts = speaker_changes(turns)
        console.print(f"  [dim]{len(cfg.cuts)} speaker changes to cut on[/]")
        source = make_source(None, audio_file, realtime=False, audio=audio).open()

    # Align whole speaker blocks after the fact rather than each utterance as it lands:
    # one contiguous span of one voice, timed in one pass, so word timings run continuously
    # across the block and every word carries its speaker by construction. Needs the
    # aligner loaded without the session running it per utterance.
    by_block = turns is not None
    backend_obj = _load(cfg, align=True) if by_block else _load(cfg)
    if by_block and not isinstance(backend_obj, Aligning):
        by_block = False  # falls back to per-utterance timings, which still work
        console.print(f"  [dim]{backend_obj.name} times utterances, not blocks[/]")
    cfg.timestamps = not by_block

    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    bar = Progress(
        TextColumn("[dim]{task.description}[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    class Hooks:
        def __init__(self):
            self.job = None
            self.seconds = 0.0
            self.t0 = 0.0
            self.audio = None

        def source(self, src):
            self.seconds = getattr(src, "seconds", 0.0)
            # Kept so --speakers can diarize what was already decoded.
            self.audio = getattr(src, "audio", None)

        def status(self, msg):
            console.print(f"[dim]{msg}…[/]", highlight=False)

        def ready(self, threshold):
            console.print(
                f"[dim]{self.seconds / 60:.1f} min · VAD threshold {threshold:.5f}[/]"
            )
            self.t0 = time.monotonic()
            self.job = bar.add_task("transcribing", total=max(self.seconds, 0.001))
            bar.start()

        def bind_stop(self, stop):
            signal.signal(signal.SIGINT, lambda *_: stop.set())

        def level(self, rms, in_speech):
            pass

        def interim(self, seg):
            pass

        def segment(self, seg):
            if self.job is not None:
                # Progress is where the audio has been consumed to, which is the end of
                # the utterance just finished -- not a count of utterances, whose total
                # nobody knows until the file runs out.
                bar.update(self.job, completed=min(seg.start + seg.audio_sec, self.seconds))

        def error(self, offset, msg):
            console.print(f"[red]{fmt_clock(offset)}  transcribe failed: {msg}[/]")

    hooks = Hooks()
    try:
        result = run_session(cfg, hooks, backend=backend_obj, source=source)
    finally:
        bar.stop()

    took = time.monotonic() - hooks.t0 if hooks.t0 else 0.0
    if took > 0 and hooks.seconds:
        console.print(
            f"[dim]{hooks.seconds / 60:.1f} min in {took:.1f}s · "
            f"{hooks.seconds / took:.0f}x realtime[/]"
        )

    segments, words, recorder = result
    if by_block:
        words = _align_blocks(backend_obj, audio, segments, turns, cfg.language)
    base = _report(cfg, segments, words, recorder, turns=turns)
    if base is None:
        raise typer.Exit(1)
    # One plain line on stderr: the path a script parses. No rich markup, no ANSI --
    # the same contract as `m dictate`'s stdout, which carries the transcript and
    # nothing else so "a state machine subscribes to stderr without disturbing that".
    sys.stderr.write(f"{base}\n")
    sys.stderr.flush()


def _align_blocks(backend, audio, segments, turns, language: str):
    """Time each speaker block against its own span, and label its words as it goes."""
    from .audio import SAMPLE_RATE
    from .diarize import speaker_blocks

    blocks = speaker_blocks(segments, turns, len(audio) / SAMPLE_RATE)
    out = []
    with console.status("[dim]timing words…[/]") as st:
        for i, (speaker, start, end, text) in enumerate(blocks, 1):
            st.update(f"[dim]timing words, block {i}/{len(blocks)}…[/]")
            pcm = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            if not text.strip() or len(pcm) < SAMPLE_RATE // 10:
                continue
            try:
                timed = backend.align(pcm, text, language=language)
            except Exception as e:  # one bad block must not lose the transcript
                console.print(f"[yellow]  block at {fmt_clock(start)} not timed: {e}[/]")
                continue
            out += [
                {
                    "text": w.text,
                    "start": start + w.start,
                    "end": start + w.end,
                    "speaker": speaker,
                }
                for w in timed
            ]
    console.print(f"  [dim]{len(out)} words timed across {len(blocks)} speaker blocks[/]")
    return out


def _diarize_audio(audio, num_speakers: int | None):
    """Who spoke when, over audio that is already decoded and in memory."""
    from .audio import SAMPLE_RATE
    from .diarize.coreml import DiarizationUnavailable
    from .diarize.offline import OfflineConfig, OfflineDiarizer

    if audio is None:
        raise typer.BadParameter("--speakers needs a file, not a live source.")
    try:
        cfg = OfflineConfig(num_speakers=num_speakers)
        with console.status("[dim]loading diarization models…[/]") as st:
            engine = OfflineDiarizer(cfg, on_status=lambda m: st.update(f"[dim]{m}…[/]"))
        dur = len(audio) / SAMPLE_RATE
        with console.status(f"[dim]working out who spoke, {dur / 60:.1f} min…[/]"):
            t0 = time.monotonic()
            turns = engine.diarize(audio, SAMPLE_RATE)
            took = time.monotonic() - t0
    except (DiarizationUnavailable, ValueError) as e:
        raise typer.BadParameter(str(e)) from e

    found = sorted({t.speaker for t in turns})
    console.print(
        f"[green]{len(found)}[/] speakers, [green]{len(turns)}[/] turns "
        f"[dim]({dur / took:.0f}x realtime)[/]"
    )
    for spk in found:
        held = sum(t.duration for t in turns if t.speaker == spk)
        console.print(
            f"  [cyan]speaker {spk}[/]  {held / 60:5.2f} min  [dim]{held / dur * 100:4.1f}%[/]"
        )
    return turns


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
    console.print(
        f"count: [b]{len(points)}[/]   first text after [b]{points[0] if points else length:g}s[/]"
    )
    console.print(
        f"audio processed: [b]{total:.1f}s[/] for a {length:g}s utterance"
        f"  [dim](x{total / length:.1f} realtime)[/]"
    )


@app.command()
def tui(
    out: Path = OUT,
    language: str = LANG,
    device: str = DEVICE,
    mic: int | None = MIC,
    wav: Path | None = WAV,
    threshold: float | None = THRESH,
    first: float = FIRST,
    growth: float = GROWTH,
    max_gap: float = MAXGAP,
    record: bool = REC,
    record_dir: Path = RECDIR,
    backend: str = BACKEND,
    model: str = MODEL,
    aligner: str = ALIGNER,
    dtype: str = DTYPE,
    partials: str = PARTIALS,
    stream_chunk: float = CHUNKSEC,
    min_speech: float = MINSPEECH,
    context: str = CONTEXT,
):
    """Full-screen live view [dim](q quit · p pause · c clear)[/]."""
    from .tui import build_tui

    cfg = _config(
        out=out,
        language=language,
        device=device,
        mic=mic,
        wav=wav,
        threshold=threshold,
        first=first,
        growth=growth,
        max_gap=max_gap,
        record=record,
        record_dir=record_dir,
        backend=backend,
        model=model,
        aligner=aligner,
        dtype=dtype,
        partials=partials,
        stream_chunk=stream_chunk,
        min_speech=min_speech,
        context=context,
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
    out: Path = OUT,
    language: str = LANG,
    device: str = DEVICE,
    mic: int | None = MIC,
    wav: Path | None = WAV,
    threshold: float | None = THRESH,
    first: float = FIRST,
    growth: float = GROWTH,
    max_gap: float = MAXGAP,
    record: bool = REC,
    record_dir: Path = RECDIR,
    backend: str = BACKEND,
    model: str = MODEL,
    aligner: str = ALIGNER,
    dtype: str = DTYPE,
    partials: str = PARTIALS,
    stream_chunk: float = CHUNKSEC,
    min_speech: float = MINSPEECH,
    context: str = CONTEXT,
):
    """Stream transcriptions to stdout [dim](Ctrl-C to stop)[/]."""
    from rich.live import Live
    from rich.text import Text

    cfg = _config(
        out=out,
        language=language,
        device=device,
        mic=mic,
        wav=wav,
        threshold=threshold,
        first=first,
        growth=growth,
        max_gap=max_gap,
        record=record,
        record_dir=record_dir,
        backend=backend,
        model=model,
        aligner=aligner,
        dtype=dtype,
        partials=partials,
        stream_chunk=stream_chunk,
        min_speech=min_speech,
        context=context,
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

    with live if show_interim else contextlib.nullcontext():
        result = run_session(cfg, Hooks(), backend=backend_obj)
    _report(cfg, *result)


# ------------------------------------------------------------------ dictate

WAIT = typer.Option(
    8.0,
    "--wait",
    help="Give up if speech hasn't started within this many seconds "
    "[dim](0 = wait forever)[/].",
)
EVENTS = typer.Option(
    False, "--events", help="Emit JSON lines on stderr: levels, state, text."
)
RECAL = typer.Option(
    False,
    "--recalibrate",
    help="Re-measure the room instead of reusing the cached threshold.",
)
DICT_REC = typer.Option(
    False, "--record/--no-record", help="Save the utterance's audio + manifest."
)
DICT_FIRST = typer.Option(
    0.0,
    "--interim",
    help="Emit provisional text this many seconds in "
    "[dim](0 = off; only useful with --events)[/].",
)
HOLD = typer.Option(
    False,
    "--hold",
    help="Keep listening through pauses until signalled, instead of "
    "stopping at the first one. [dim]For hold-to-talk.[/]",
)


class _Events:
    """JSON lines on stderr.

    The split is the whole interface: **stdout is the transcript and nothing else**, so
    `m dictate | pbcopy` works with no flags, while anything that wants a level meter or
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
    language: str = LANG,
    mic: int | None = MIC,
    wav: Path | None = WAV,
    threshold: float | None = THRESH,
    recalibrate: bool = RECAL,
    wait: float = WAIT,
    events: bool = EVENTS,
    interim: float = DICT_FIRST,
    hold: bool = HOLD,
    record: bool = DICT_REC,
    record_dir: Path = RECDIR,
    backend: str = BACKEND,
    model: str = MODEL,
    dtype: str = DTYPE,
    device: str = DEVICE,
    min_speech: float = MINSPEECH,
    context: str = CONTEXT,
):
    """Speech to stdout, then exit. [dim]A surface to compose on.[/]

    Talk; stop talking; the text is on stdout. Nothing else ever is — status goes to
    stderr — so it pipes:

        m dictate | pbcopy
        m dictate | tee -a ~/notes.md

    [b]SIGINT means "I stopped talking"[/], not "abort": the utterance in progress is
    still transcribed and printed. That is what makes hold-to-talk work from any hotkey
    manager with no daemon and no protocol — start it on key down, `kill -INT` it on key
    up.

    Two ways to decide when you're done, and a key-driven one wants the second:

    [b]default[/] — the pause ends it. The VAD closes an utterance after 750ms of silence
    and that is the whole result. Right for a bare `m dictate | pbcopy` with nothing
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
        out=paths.out_dir(),
        language=language,
        device=device,
        mic=mic,
        wav=wav,
        threshold=threshold,
        first=interim,
        growth=1.6,
        max_gap=3.0,
        record=record,
        record_dir=record_dir,
        backend=backend,
        model=model,
        aligner=DEFAULT_ALIGNER,
        dtype=dtype,
        partials="reencode",
        stream_chunk=2.0,
        min_speech=min_speech,
        context=context,
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
    backend_obj = _load(
        cfg, err, on_status=(lambda m: emit("loading", detail=m)) if events else None
    )
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
            emit(
                "ready",
                threshold=round(threshold, 6),
                calibrated=measuring,
                load=round(load_took, 3),
            )
            if not events:
                err.print(
                    f"[b]Speak.[/] [dim](threshold {threshold:.5f}, "
                    f"loaded in {load_took:.1f}s)[/]"
                )

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
                assert self.stop is not None  # bind_stop runs before any frame arrives
                self.stop.set()

        def interim(self, seg):
            emit("partial", text=seg.text)

        def segment(self, seg):
            self.parts.append(seg.text)
            emit(
                "final",
                text=seg.text,
                took=round(seg.took, 3),
                audio=round(seg.audio_sec, 3),
            )
            if not events:
                err.print(f"[dim]{seg.took:.2f}s for {seg.audio_sec:.1f}s of audio[/]")
            # Under --hold the pause is just a pause: whoever is holding the key decides
            # when this ends, and utterances closed along the way are pieces of one
            # dictation. Otherwise the first pause is the whole job.
            if not hold:
                assert self.stop is not None  # bind_stop runs before any chunk closes
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
