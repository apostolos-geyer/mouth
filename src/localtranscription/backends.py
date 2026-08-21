"""Pluggable ASR backends.

The engine only ever needs one thing from a model: hand it audio, get back text and
optionally word timings. Everything else -- torch vs MLX, which checkpoint, device
placement, dtype, quantisation, library-specific result objects -- lives behind this
boundary.

Normalising to our own Word/Transcription types (rather than passing a library's objects
through) is what keeps that true: `torch` calls it `time_stamps` with `.start_time`
attributes, `mlx` calls it `segments` with `["start"]` keys, and the engine should know
about neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Upstream weights. Any of these can be replaced with a local directory -- notably a
# quantised one built by `lt quantize` -- so nothing here is a hard-coded destiny.
DEFAULT_ASR = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

DTYPES = ("bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Word:
    """One aligned word, timed relative to the start of the audio handed in."""

    text: str
    start: float
    end: float


@dataclass
class Transcription:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = ""


@runtime_checkable
class Backend(Protocol):
    """What the engine requires. Deliberately one method wide."""

    name: str
    detail: str

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: str,
        timestamps: bool,
    ) -> Transcription: ...


class PartialStream(Protocol):
    """An in-progress utterance that accepts audio incrementally.

    Reached via an optional `streaming = True` class attribute plus `open_stream()`.
    Deliberately *not* part of the Backend protocol: making it a required attribute there
    breaks every existing implementer, including the fakes in the test suite, for a
    capability most backends won't have. The engine probes with getattr instead.

    Only ever used for provisional passes. The final pass is a plain transcribe() over the
    whole utterance, because that is what carries the forced aligner and what gets saved --
    so nothing here can affect the transcript that lands on disk.
    """

    stable: str
    """The prefix the decoder has committed to and won't revise.

    Monotonic: the largest surviving prefix across turns. A front end renders this
    settled and the tail after it as still-moving.
    """

    def feed(self, pcm: np.ndarray) -> str:
        """Add audio, return the best text so far (cumulative, not a delta)."""

    def close(self) -> None:
        """Drop the session's state. The next utterance starts clean."""


class BackendUnavailable(RuntimeError):
    """Raised with an actionable message when a backend's deps aren't installed."""


def _dtype(table: dict, name: str, backend: str):
    """Map a friendly dtype name onto a library's own type, or say what is accepted."""
    resolved = table.get(name)
    if resolved is None:
        raise BackendUnavailable(
            f"unknown dtype {name!r} for the {backend} backend; "
            f"choose from {', '.join(DTYPES)}"
        )
    return resolved


def local_checkpoints() -> list[Path]:
    """Every quantised checkpoint on disk, newest naming first.

    One definition of what counts -- a directory with a config.json -- shared by
    `lt models` and by resolve_checkpoint's "Available:" message, so the two can't
    disagree about what exists.
    """
    from . import paths

    models = paths.models_dir()
    if not models.exists():
        return []
    return sorted(p for p in models.glob("*") if (p / "config.json").exists())


def resolve_checkpoint(ref: str) -> str:
    """Turn a --model/--aligner value into something loadable, or say why it isn't.

    Accepts, in order: an existing path; a name or path relative to the checkpoint
    directory; a Hugging Face repo id. The middle case is the one that matters -- `lt
    quantize` writes under XDG_CACHE_HOME, so `-M qwen3-asr-1.7b-q8g64` and the older
    `-M models/qwen3-asr-1.7b-q8g64` both have to find it without the caller typing an
    absolute path.

    Anything that looks local but isn't there fails here, with the list of what is. The
    alternative is what this function was written to stop: the name falls through to the
    Hub, which reports `401 Unauthorized` for a repo that was never a repo.
    """
    from . import paths

    candidate = Path(ref).expanduser()
    if candidate.exists():
        return str(candidate)

    models = paths.models_dir()
    for guess in (models / ref, models / candidate.name):
        if guess.exists():
            return str(guess)

    # A repo id is exactly `owner/name` -- but so is `models/whatever`, and that one was
    # plainly meant to be a directory. Treat a leading component that names the checkpoint
    # directory, or an existing directory here, as proof the caller meant a path.
    parts = [p for p in ref.split("/") if p]
    local_intent = (
        ref.startswith((".", "/", "~"))
        or parts[0] == models.name
        or Path(parts[0]).is_dir()
    )
    if len(parts) == 2 and not local_intent:
        return ref

    known = [p.name for p in local_checkpoints()]
    listing = ("\n  " + "\n  ".join(known)) if known else " (none yet -- run `lt quantize`)"
    raise BackendUnavailable(
        f"no checkpoint {ref!r}: not a path, and not in {models}."
        f"\nAvailable:{listing}"
    )


def describe_checkpoint(model: str) -> str:
    """A short tag for a checkpoint: its quantisation if local, else 'bf16/fp16'.

    Quantisation is a property of the weights on disk, not a runtime flag, so the only
    honest place to read it from is the checkpoint itself.
    """
    cfg = Path(model) / "quantization_config.json"
    if not cfg.exists():
        return ""
    try:
        q = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return ""
    mode = q.get("mode", "affine")
    bits, group = q.get("bits", "?"), q.get("group_size", "?")
    return f"q{bits}/g{group}" if mode == "affine" else f"{mode}/g{group}"


# ---------------------------------------------------------------- torch / MPS


class TorchBackend:
    """PyTorch + transformers on MPS. The reference implementation."""

    name = "torch"
    # What this backend is fastest in when --dtype is "auto": torch/MPS is strongest in
    # bf16, while MLX's kernels and every published MLX quantisation are fp16.
    default_dtype = "bf16"
    requires = ("torch", "qwen_asr")
    takes_device = True
    # qwen_asr's own streaming path is vLLM-only, and vLLM has no Metal support, so there
    # is nothing to hook here. Partials re-transcribe the prefix on this backend.
    streaming = False

    def __init__(self, model: str = DEFAULT_ASR,
                 aligner: Optional[str] = DEFAULT_ALIGNER,
                 device: str = "mps", dtype: str = "bf16", on_status=None):
        import torch
        from qwen_asr import Qwen3ASRModel
        from transformers.utils import logging as hf_logging

        # "generation flags are not valid: ['temperature']" and "Setting pad_token_id" are
        # benign and fire on every load and every generate. Errors still surface.
        hf_logging.set_verbosity_error()
        # Our own status spinner covers loading; transformers' shard bar just fights it for
        # the cursor. Hub *download* bars are separate and still show on first run.
        hf_logging.disable_progress_bar()

        resolved = _dtype({"bf16": torch.bfloat16, "fp16": torch.float16,
                           "fp32": torch.float32}, dtype, "torch")

        say = on_status or (lambda m: None)
        say(f"loading {model} on {device}")
        self.model_id, self.aligner_id = model, aligner
        self.detail = f"{device} · {dtype}"
        self._model = Qwen3ASRModel.from_pretrained(
            model,
            forced_aligner=aligner,
            dtype=resolved,
            device_map=device,
            # 512 default silently truncated long chunks in the offline tool; keep parity.
            max_new_tokens=2048,
            forced_aligner_kwargs={"dtype": resolved, "device_map": device},
        )

    def transcribe(self, audio, sample_rate, *, language, timestamps) -> Transcription:
        results = self._model.transcribe(
            audio=(audio, sample_rate),
            language=language,
            return_time_stamps=timestamps,
        )
        r = results[0]
        items = r.time_stamps if r.time_stamps is not None else []
        return Transcription(
            text=r.text or "",
            words=[Word(w.text, w.start_time, w.end_time) for w in items],
            language=getattr(r, "language", "") or "",
        )


# ---------------------------------------------------------------- MLX


def _load_mlx_model(path_or_repo: str, dtype):
    """Load an MLX checkpoint, honouring its quantisation *mode*.

    mlx_qwen3_asr's own loader reads bits/group_size out of quantization_config.json but
    always calls nn.quantize(mode="affine"). The mxfp4/mxfp8/nvfp4 modes save `.scales`
    with no `.biases`, so an affine re-quantisation builds a parameter tree the checkpoint
    can't fill and load_weights raises. Mirroring the load path here is what makes those
    formats selectable at all.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_qwen3_asr.config import Qwen3ASRConfig
    from mlx_qwen3_asr.convert import remap_weights
    from mlx_qwen3_asr.load_models import (
        _cast_tree_dtype,
        _load_safetensors,
        _materialize_tied_lm_head_weights,
        _quantized_module_paths,
        _resolve_path,
    )
    from mlx_qwen3_asr.model import Qwen3ASRModel

    model_path = _resolve_path(path_or_repo)
    config = Qwen3ASRConfig.from_dict(json.loads((model_path / "config.json").read_text()))
    weights = _materialize_tied_lm_head_weights(
        remap_weights(_load_safetensors(model_path)), config
    )
    model = Qwen3ASRModel(config)

    if any(k.endswith(".scales") for k in weights):
        qcfg_path = model_path / "quantization_config.json"
        qcfg = json.loads(qcfg_path.read_text()) if qcfg_path.exists() else {}
        quantized_paths = _quantized_module_paths(weights)
        nn.quantize(
            model,
            bits=int(qcfg.get("bits", 4)),
            group_size=int(qcfg.get("group_size", 64)),
            mode=str(qcfg.get("mode", "affine")),
            class_predicate=lambda p, _m: p in quantized_paths,
        )
        model.load_weights(list(weights.items()))
    else:
        model.load_weights(list(weights.items()))
        if dtype != mx.float32:
            model.load_weights(
                list(tree_flatten(_cast_tree_dtype(model.parameters(), dtype)))
            )

    mx.eval(model.parameters())
    model.eval()
    # Session() needs one of these to find the tokenizer next to the weights.
    model._source_model_id = path_or_repo
    model._resolved_model_path = str(model_path)
    return model


class _MlxStream:
    """One in-progress utterance on mlx-qwen3-asr's incremental decoder.

    `feed_audio` encodes only the audio it hasn't seen and keeps the decoder KV cache
    across turns, so the cost of watching an utterance grow is linear in its length rather
    than quadratic. It buffers internally and decodes once it holds `chunk_sec`, which is
    also how often the text can change.
    """

    def __init__(self, session, language: str, chunk_sec: float, max_context_sec: float):
        self._session = session
        self._language = language
        self._chunk_sec = chunk_sec
        self._max_context_sec = max_context_sec
        self._state = None

    @property
    def stable(self) -> str:
        """mlx-qwen3-asr keeps `stable_text` monotonic by design; see PartialStream."""
        return (getattr(self._state, "stable_text", "") or "").strip()

    def feed(self, pcm: np.ndarray) -> str:
        if self._state is None:
            self._state = self._session.init_streaming(
                language=self._language,
                chunk_size_sec=self._chunk_sec,
                max_context_sec=self._max_context_sec,
            )
        self._state = self._session.feed_audio(np.asarray(pcm, dtype=np.float32), self._state)
        return (self._state.text or "").strip()

    def close(self) -> None:
        # Dropping the state drops the KV cache with it; the next utterance must not
        # inherit this one's decoder context or its text.
        self._state = None


class MlxBackend:
    """MLX port (mlx-qwen3-asr).

    Fastest option here *once the checkpoint is quantised*. Measured on an M3 Max against
    the same 1.7B weights, per transcribe() call:

        clip     torch bf16   mlx fp16   mlx q8/g64   mlx q4/g64
        2s          0.284s     0.613s      0.129s       0.110s
        8s          1.080s     2.362s      0.470s       0.379s
        30s         2.658s     7.216s      1.961s       1.135s

    So the port is 2.1-2.7x *slower* than torch unquantized -- which is what an earlier
    revision of this file recorded -- and 2.2x *faster* than torch at 8-bit. The model is
    memory-bandwidth-bound; shrinking the weights is the whole optimisation, and fp16 vs
    bf16 is noise beside it. 8-bit is the default recommendation because upstream measures
    it at +0.04pp WER; 4-bit costs +0.43pp for another ~1.7x on long clips.

    Build the checkpoints with `lt quantize`. Nothing here downloads a quantised model:
    quantisation is a property of weights on disk, and `--model` points at them.

    Differences this adapter absorbs: `return_timestamps` vs `return_time_stamps`,
    `.segments` dicts vs `.time_stamps` objects, a single result rather than a list, and no
    device argument at all (MLX uses unified memory, so --device is meaningless here).

    Note their README is misleading in two places, both of which bite:
      - `dtype` is annotated `mx.Dtype`, not a string. Passing "float16" reaches
        `x.astype("float16")` and raises TypeError during load. We keep a friendly string
        at this boundary and resolve it to an mx.Dtype here.
      - `forced_aligner` is `Optional[str | ForcedAligner]`, not a bool. `True` sails
        through `_resolve_aligner` and is returned *as* the aligner, so every timestamped
        call dies on an attribute error.
    """

    name = "mlx"
    default_dtype = "fp16"
    requires = ("mlx_qwen3_asr",)
    takes_device = False  # unified memory; there is no device to place anything on
    streaming = True

    def __init__(self, model: str = DEFAULT_ASR,
                 aligner: Optional[str] = DEFAULT_ALIGNER,
                 dtype: str = "fp16", on_status=None):
        try:
            import mlx.core as mx
            from mlx_qwen3_asr import ForcedAligner, Session
        except ImportError as e:
            raise BackendUnavailable(
                "the mlx backend needs mlx-qwen3-asr; install it with "
                "`uv sync --extra mlx`, or use --backend torch"
            ) from e

        resolved = _dtype({"fp16": mx.float16, "bf16": mx.bfloat16,
                           "fp32": mx.float32}, dtype, "mlx")

        say = on_status or (lambda m: None)
        say(f"loading {model} via mlx")
        self.model_id, self.aligner_id = model, aligner
        quant = describe_checkpoint(model)
        self.detail = f"mlx · {quant or dtype}"

        # Session rather than the module-level transcribe(): it owns its state explicitly
        # instead of relying on a process-global model cache. Hand it an already-loaded
        # model so quantisation mode is honoured -- see _load_mlx_model.
        self._session = Session(model=_load_mlx_model(model, resolved), dtype=resolved)

        # Build the aligner ONCE and hand over the instance. Given None or a string,
        # _resolve_aligner() constructs a fresh ForcedAligner on every call -- reloading
        # 0.6B of weights per final. Passing an instance is the only branch that reuses it,
        # which is what their own CLI does.
        self._aligner = None
        if aligner is not None:
            say(f"loading {aligner} via mlx")
            self._aligner = ForcedAligner(aligner, dtype=resolved)

    def transcribe(self, audio, sample_rate, *, language, timestamps) -> Transcription:
        if timestamps and self._aligner is None:
            # Not a crash upstream, which is why it's worth catching here: mlx would take
            # forced_aligner=None as "make one", and rebuild 0.6B of weights on every
            # call. A silent 10x slowdown is worse than either a crash or a refusal.
            raise BackendUnavailable(
                "this backend was loaded with align=False; it cannot produce timestamps"
            )
        r = self._session.transcribe(
            (audio, sample_rate),
            language=language,
            return_timestamps=timestamps,
            # Instance, never True/str -- see __init__.
            forced_aligner=self._aligner if timestamps else None,
            # Parity with the torch backend, where the 512 default silently truncated.
            max_new_tokens=2048,
        )
        segments = getattr(r, "segments", None) or []
        return Transcription(
            text=r.text or "",
            words=[Word(s["text"], s["start"], s["end"]) for s in segments],
            language=getattr(r, "language", "") or "",
        )

    def open_stream(self, *, language: str, chunk_sec: float = 2.0,
                    max_context_sec: float = 30.0) -> _MlxStream:
        return _MlxStream(self._session, language, chunk_sec, max_context_sec)


# ---------------------------------------------------------------- registry

BACKENDS = {"torch": TorchBackend, "mlx": MlxBackend}


def available(name: str) -> bool:
    """Whether a backend's imports resolve, without loading any weights."""
    import importlib.util

    cls = BACKENDS.get(name)
    if cls is None:  # an unknown backend is not "available"
        return False
    return all(importlib.util.find_spec(m) is not None for m in cls.requires)


def resolve_dtype(backend: str, dtype: Optional[str]) -> str:
    """`--dtype auto` means whichever precision that backend is actually fast in."""
    if dtype in (None, "", "auto"):
        cls = BACKENDS.get(backend)
        return cls.default_dtype if cls else "fp16"
    if dtype not in DTYPES:
        raise BackendUnavailable(f"unknown dtype {dtype!r}; choose from {', '.join(DTYPES)}")
    return dtype


def open_partial_stream(backend: Backend, *, language: str,
                        chunk_sec: float) -> Optional[PartialStream]:
    """A stream for provisional passes, or None if this backend has no incremental decode.

    The capability is probed rather than required of the Backend protocol: making it a
    required attribute breaks every existing implementer, including the fakes in the test
    suite, for something most backends won't have. Probing it *here* rather than in the
    engine is what keeps the attribute name and the constructor's keywords -- which are
    backend facts -- inside this module.
    """
    if not getattr(backend, "streaming", False):
        return None
    return backend.open_stream(language=language, chunk_sec=chunk_sec)


def load_backend(name: str, *, model: Optional[str] = None,
                 aligner: Optional[str] = None, device: str = "mps",
                 dtype: Optional[str] = None, on_status=None,
                 warmup: bool = True, align: bool = True) -> Backend:
    """Load a backend. align=False skips the forced aligner entirely.

    Not the same as passing timestamps=False per call: the aligner is a second set of
    weights, and a caller that will never ask for word timings should not pay to load
    them. `lt dictate` is that caller -- it wants a string, not a transcript.
    """
    if name not in BACKENDS:
        raise BackendUnavailable(
            f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}"
        )
    cls = BACKENDS[name]
    model = resolve_checkpoint(model or DEFAULT_ASR)
    aligner = resolve_checkpoint(aligner or DEFAULT_ALIGNER) if align else None
    dtype = resolve_dtype(name, dtype)

    kwargs = {"model": model, "aligner": aligner, "dtype": dtype, "on_status": on_status}
    if cls.takes_device:
        kwargs["device"] = device
    backend = cls(**kwargs)

    if warmup:
        say = on_status or (lambda m: None)
        say("warming up")
        # First inference pays graph-compilation cost; eat it before capture starts. Warm
        # the path that will actually run: asking for timestamps here without an aligner
        # raises, and warming the aligned path when nothing will use it is wasted load.
        backend.transcribe(
            np.zeros(16000, dtype=np.float32), 16000, language="English",
            timestamps=align,
        )
    return backend
