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

# Historical names, still imported elsewhere.
ASR_MODEL = DEFAULT_ASR
ALIGNER_MODEL = DEFAULT_ALIGNER

# What each backend runs at when --dtype isn't given. These differ because the fast path
# differs: torch/MPS is strongest in bf16, MLX's kernels and every published MLX
# quantisation are fp16.
DEFAULT_DTYPE = {"torch": "bf16", "mlx": "fp16"}

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


class BackendUnavailable(RuntimeError):
    """Raised with an actionable message when a backend's deps aren't installed."""


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

    def __init__(self, model: str = DEFAULT_ASR, aligner: str = DEFAULT_ALIGNER,
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

        resolved = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(dtype)
        if resolved is None:
            raise BackendUnavailable(
                f"unknown dtype {dtype!r} for the torch backend; choose from "
                f"{', '.join(DTYPES)}"
            )

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

    def __init__(self, model: str = DEFAULT_ASR, aligner: str = DEFAULT_ALIGNER,
                 dtype: str = "fp16", on_status=None):
        try:
            import mlx.core as mx
            from mlx_qwen3_asr import ForcedAligner, Session
        except ImportError as e:
            raise BackendUnavailable(
                "the mlx backend needs mlx-qwen3-asr; install it with "
                "`uv sync --extra mlx`, or use --backend torch"
            ) from e

        resolved = {"fp16": mx.float16, "bf16": mx.bfloat16,
                    "fp32": mx.float32}.get(dtype)
        if resolved is None:
            raise BackendUnavailable(
                f"unknown dtype {dtype!r} for the mlx backend; choose from "
                f"{', '.join(DTYPES)}"
            )

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
        say(f"loading {aligner} via mlx")
        self._aligner = ForcedAligner(aligner, dtype=resolved)

    def transcribe(self, audio, sample_rate, *, language, timestamps) -> Transcription:
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


# ---------------------------------------------------------------- registry

BACKENDS = {"torch": TorchBackend, "mlx": MlxBackend}


def available(name: str) -> bool:
    """Whether a backend's imports resolve, without loading any weights."""
    import importlib.util

    needed = {"torch": ("torch", "qwen_asr"), "mlx": ("mlx_qwen3_asr",)}.get(name, ())
    return all(importlib.util.find_spec(m) is not None for m in needed)


def resolve_dtype(backend: str, dtype: Optional[str]) -> str:
    """`--dtype auto` means whichever precision that backend is actually fast in."""
    if dtype in (None, "", "auto"):
        return DEFAULT_DTYPE.get(backend, "fp16")
    if dtype not in DTYPES:
        raise BackendUnavailable(f"unknown dtype {dtype!r}; choose from {', '.join(DTYPES)}")
    return dtype


def load_backend(name: str, *, model: Optional[str] = None,
                 aligner: Optional[str] = None, device: str = "mps",
                 dtype: Optional[str] = None, on_status=None,
                 warmup: bool = True) -> Backend:
    if name not in BACKENDS:
        raise BackendUnavailable(
            f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}"
        )
    cls = BACKENDS[name]
    model = model or DEFAULT_ASR
    aligner = aligner or DEFAULT_ALIGNER
    dtype = resolve_dtype(name, dtype)

    # device is torch-only; MLX has unified memory and no device argument.
    kwargs = {"model": model, "aligner": aligner, "dtype": dtype, "on_status": on_status}
    if name == "torch":
        kwargs["device"] = device
    backend = cls(**kwargs)

    if warmup:
        say = on_status or (lambda m: None)
        say("warming up")
        # First inference pays graph-compilation cost; eat it before capture starts.
        backend.transcribe(
            np.zeros(16000, dtype=np.float32), 16000, language="English", timestamps=True
        )
    return backend
