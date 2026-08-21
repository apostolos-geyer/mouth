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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# The one sample rate this tool works in; imported rather than restated so a partial
# decoder cannot disagree with the VAD that cut its audio.
from .vad import SAMPLE_RATE

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


@runtime_checkable
class PartialDecoder(Protocol):
    """How provisional text gets made for an utterance still in progress.

    Three of these exist and they differ in what they cost, not in what they are for, so
    the engine holds exactly one and never asks which. They were three branches in
    Transcriber before, which meant three copies of the emit tail, two spellings of "drop
    the per-utterance state", and utterance-change detection at two different altitudes.

    Only ever used for provisional passes. The final pass is a plain transcribe() over the
    whole utterance, because that is what carries the forced aligner and what gets saved --
    so nothing behind this protocol can affect the transcript that lands on disk.
    """

    mode: str
    """Which `--partials` value this actually is.

    Not necessarily the one that was asked for: a backend that cannot stream falls back to
    re-encoding, and the session says so rather than pretending.
    """

    incremental: bool
    """Whether `text()` wants only the audio since the last call, rather than the whole
    utterance so far. True only for a decoder carrying its own state across calls, and it
    decides how the VAD cuts chunks -- so it is a property of the decoder, not a flag the
    caller passes."""

    stable: str
    """The prefix the decoder has committed to and won't revise, or "" when nothing is.

    Monotonic where it is non-empty. A front end renders this settled and the tail after
    it as still-moving. Empty for anything that re-transcribes the prefix each pass,
    because then nothing is settled and every word can still change.
    """

    def text(self, pcm: np.ndarray) -> str:
        """The best text so far. Cumulative, never a delta, whatever `incremental` says."""

    def reset(self) -> None:
        """Drop per-utterance state. The next utterance starts clean."""


class _ReencodePartials:
    """The default: re-transcribe the whole prefix each pass.

    No state, so `reset()` has nothing to do -- which is the point. It exists so that
    "no special partial decoder" is a decoder like the others rather than a branch, and
    the engine can hold one unconditionally.
    """

    mode = "reencode"
    incremental = False
    stable = ""

    def __init__(self, backend: Backend, language: str):
        self._backend = backend
        self._language = language

    def text(self, pcm: np.ndarray) -> str:
        return self._backend.transcribe(
            pcm, SAMPLE_RATE, language=self._language, timestamps=False
        ).text

    def reset(self) -> None:
        pass


@runtime_checkable
class Streaming(Protocol):
    """A backend that can decode an utterance incrementally."""

    def open_stream(
        self, *, language: str, chunk_sec: float = 2.0, max_context_sec: float = 30.0
    ) -> PartialDecoder: ...


@runtime_checkable
class Biasable(Protocol):
    """A backend whose decoding can be biased toward expected words."""

    context: str
    """Names, jargon, spellings. Qwen3-ASR biases decoding toward them; "" means no bias.

    An attribute rather than a transcribe() argument, because it describes the session and
    not one utterance -- and because that is what lets a front end change it while the
    session runs, which is the whole point of editing it in the TUI.

    Off the Backend protocol for the same reason as the two above: Backend is one method
    wide, and every fake in the test suite is a backend without being this.
    """


@runtime_checkable
class Drafting(Protocol):
    """A backend that can decode a partial against the previous one as a draft."""

    def open_draft(self, *, language: str) -> _MlxDraftDecoder: ...


# Optional capabilities, kept off Backend so it stays one method wide, and expressed as
# protocols rather than a `streaming = True` flag beside the method. The flag was a second
# thing to keep true: a backend could carry it without the method, or grow the method and
# forget it, and the getattr probe that read it had to be wrapped in a type suppression
# broad enough to hide an unprobed call as well. isinstance against these narrows instead,
# so the checker still rejects reaching for open_draft() without asking.
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
        f"no checkpoint {ref!r}: not a path, and not in {models}.\nAvailable:{listing}"
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
    # is nothing to hook here: no open_stream, so `isinstance(backend, Streaming)` is
    # False and partials re-transcribe the prefix on this backend.

    def __init__(
        self,
        model: str = DEFAULT_ASR,
        aligner: str | None = DEFAULT_ALIGNER,
        device: str = "mps",
        dtype: str = "bf16",
        on_status=None,
        context: str = "",
    ):
        self.context = context
        import torch
        from qwen_asr import Qwen3ASRModel
        from transformers.utils import logging as hf_logging

        # "generation flags are not valid: ['temperature']" and "Setting pad_token_id" are
        # benign and fire on every load and every generate. Errors still surface.
        hf_logging.set_verbosity_error()
        # Our own status spinner covers loading; transformers' shard bar just fights it for
        # the cursor. Hub *download* bars are separate and still show on first run.
        hf_logging.disable_progress_bar()

        resolved = _dtype(
            {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32},
            dtype,
            "torch",
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
            context=self.context,
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
    import mlx.core as mx  # ty: ignore[unresolved-import]
    from mlx import nn
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
                # mlx ships no stubs, so tree_flatten's return type is guesswork here.
                list(tree_flatten(_cast_tree_dtype(model.parameters(), dtype)))  # ty: ignore[invalid-argument-type]
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

    mode = "stream"
    incremental = True

    def __init__(self, backend, language: str, chunk_sec: float, max_context_sec: float):
        # The backend, not its session: `context` is a mutable session setting and a front
        # end can change it mid-run, so it has to be read when a stream opens rather than
        # captured when this was built.
        self._backend = backend
        self._language = language
        self._chunk_sec = chunk_sec
        self._max_context_sec = max_context_sec
        self._state = None

    @property
    def stable(self) -> str:
        """mlx-qwen3-asr keeps `stable_text` monotonic by design; see PartialStream."""
        return (getattr(self._state, "stable_text", "") or "").strip()

    def text(self, pcm: np.ndarray) -> str:
        if self._state is None:
            self._state = self._backend._session.init_streaming(
                language=self._language,
                context=self._backend.context,
                chunk_size_sec=self._chunk_sec,
                max_context_sec=self._max_context_sec,
            )
        self._state = self._backend._session.feed_audio(
            np.asarray(pcm, dtype=np.float32), self._state
        )
        return (self._state.text or "").strip()

    def reset(self) -> None:
        # Dropping the state drops the KV cache with it; the next utterance must not
        # inherit this one's decoder context or its text.
        self._state = None


class _MlxDraftDecoder:
    """Partial passes for one utterance, decoded against the previous partial as a draft.

    The measurement that motivates this: on the quantised MLX path a partial spends 95%+
    of its time in `generate` and 3-5% in the audio encoder, so re-encoding the prefix --
    the cost everyone reaches for first -- is not the problem. Re-*decoding* it is. Every
    partial regenerates the whole transcript one token at a time, and decode is
    memory-bandwidth-bound: 8.94ms per token here, whatever the token is.

    But a partial's answer is almost exactly the previous partial's answer plus a few
    words. That makes the previous answer a free draft, and `step_many` verifies a whole
    draft in one pass over the weights:

        k      step_many    k sequential
        16       21.7ms         143.1ms     6.6x
        64       30.9ms         572.5ms    18.5x

    **Lossless by construction, though not bit-identical.** A draft token is accepted
    only where it equals the model's own argmax at that position, and transcribe()
    decodes greedily, so the accepted path is the path plain decoding would have taken --
    and healing survives, because a word the model now wants to revise simply fails to
    match and decode resumes there.

    The caveat is float, not logic: `step_many` batches k positions into one matmul and
    `step` does them one at a time, and the two do not agree in the last bits. On a near
    tie the argmax can flip. Measured over 60 partials on seven clips, output matched the
    library on six; the seventh -- music bleeding into speech, where the model was
    already unstable -- dropped a duplicated word ("real realistic" -> "realistic"). With
    WINDOW=0, which runs this same loop with no drafting, that clip matches exactly, so
    the loop is right and the batched kernel is the difference. This is provisional text
    that a final pass overwrites; it is not the transcript.

    Mirrors the accept/trim loop in mlx_qwen3_asr.generate.generate_speculative, whose
    draft comes from a second model. Ours comes from the last pass and costs nothing.
    """

    #: Draft this many tokens per verification. Past ~64 the win flattens (the pass stops
    #: being bandwidth-bound) while a rejection wastes more, and utterance-length drafts
    #: would make a single mismatch expensive.
    WINDOW = 64

    mode = "x-draft"
    incremental = False
    stable = ""

    #: How many recent tokens identify where we are in the previous answer. Too short and
    #: a common phrase matches in the wrong place; too long and nothing matches after a
    #: revision. 8 is roughly a clause.
    KEY = 8

    #: And the shortest key allowed to place us. Without a floor the search walks down to
    #: a single token, which is the state every partial *ends* in: once decode passes the
    #: end of the previous answer, every key of 2+ tokens contains a new one and misses,
    #: while a 1-token key still hits somewhere unrelated. Measured over the new tail,
    #: that issued a draft 58% of the time, averaging 37 tokens, of which 0.34 were
    #: accepted -- a ~26ms verification to produce a token an 8.94ms step would have.
    #: 60-120ms per partial, which is 20-30% of what drafting was saving.
    MIN_KEY = 3

    #: A verification costs ~3.5 sequential steps (30.9ms against 8.94ms), and replaces
    #: accepted+1 of them. Below this it is losing money, so stop drafting the pass.
    MIN_ACCEPTED = 3

    #: Judged over the last few verifications rather than all of them. A cumulative mean
    #: cannot fall: a pass that accepts ~380 tokens over six verifications sits at ~63,
    #: and the handful of bad verifications at the tail can never drag that under 3. The
    #: guard was therefore only able to catch a pass that was bad from its first look.
    RECENT = 6

    def __init__(self, backend, language: str):
        # See _MlxStream: the backend, so a context change lands on the next pass.
        self._backend = backend
        self._language = language
        self._prev: list[int] = []
        self._ngram: dict = {}
        self._recent: deque[int] = deque(maxlen=self.RECENT)
        self._paying = True

    def reset(self) -> None:
        """Drop the draft. The next utterance's text is not this one's.

        Only the draft: the accept history is per pass and text() clears it.
        """
        self._prev = []

    def _index(self) -> None:
        """Index the previous answer by n-gram, once per pass.

        The lookup below runs inside the decode loop, so it must not be a scan: the naive
        version walked ~400 tokens x KEY slices per verification, in Python, next to a
        30ms forward pass. Building {n-gram: position} once per pass is O(len) and makes
        every lookup a dict hit. Later positions overwrite earlier ones, which is the
        "most recent occurrence wins" rule a repeated phrase needs.
        """
        self._ngram = {}
        prev = self._prev
        for n in range(1, self.KEY + 1):
            for i in range(len(prev) - n + 1):
                self._ngram[tuple(prev[i : i + n])] = i + n

    def _draft(self, out: list[int]) -> list[int]:
        """What the previous pass said next, from wherever we are in it now.

        Located by matching the tail of what we have generated against the previous
        answer, rather than by index. Index alignment only survives substitutions: one
        word inserted near the start and every later token is off by one, the draft stops
        matching, and acceptance collapses for the rest of the utterance -- which is
        exactly what a partial does when a revision lands. Matching on the text itself
        re-finds the place. Longest key first, and never shorter than MIN_KEY: a common
        short phrase can match anywhere, a clause usually can't.
        """
        if not self._prev or not self._paying:
            return []
        for n in range(min(self.KEY, len(out)), self.MIN_KEY - 1, -1):
            at = self._ngram.get(tuple(out[-n:]))
            if at is not None:
                return self._prev[at : at + self.WINDOW]
        return []

    def text(self, pcm: np.ndarray) -> str:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from mlx_qwen3_asr.audio import compute_features
        from mlx_qwen3_asr.generate import (
            GenerationConfig,
            _detect_repetition,
            resolve_max_new_tokens,
        )
        from mlx_qwen3_asr.tokenizer import parse_asr_output

        # Per pass, not per utterance: the first partials carry almost no text to draft
        # from, so they accept little through no fault of the policy. Latching on that
        # switched drafting off for exactly the long later passes it pays best on.
        self._recent.clear()
        self._paying = True
        self._index()

        session = self._backend._session
        model, tok = session.model, session.tokenizer
        dtype = session.dtype
        cfg = GenerationConfig(
            max_new_tokens=resolve_max_new_tokens(
                2048, audio_duration_sec=len(pcm) / SAMPLE_RATE
            )
        )

        mel, lens = compute_features(pcm)
        feats, _ = model.audio_tower(mel.astype(dtype), lens)
        ids = mx.array(
            [
                tok.build_prompt_tokens(
                    n_audio_tokens=feats.shape[1],
                    language=self._language,
                    context=self._backend.context,
                )
            ]
        )
        seq = ids.shape[1]
        pos = mx.arange(seq)[None, :]

        cache = model.create_cache(max_seq_len=seq + cfg.max_new_tokens)
        logits = model.prefill(
            input_ids=ids,
            audio_features=feats,
            position_ids=mx.stack([pos, pos, pos], axis=1),
            cache=cache,
        )
        token = int(mx.argmax(logits[0, -1]).item())
        out = [token]
        decode_pos = mx.arange(seq, seq + cfg.max_new_tokens + 1)[None, :]
        decode_pos = mx.stack([decode_pos, decode_pos, decode_pos], axis=1)

        step = 1
        while step < cfg.max_new_tokens:
            if token in cfg.eos_token_ids or _detect_repetition(out):
                break
            # Whatever the last pass said from here on. Empty once we pass its end --
            # which is the tail this partial exists to add.
            draft = self._draft(out)[: max(0, cfg.max_new_tokens - step - 1)]
            if not draft:
                logits = model.step(
                    input_ids=mx.array([[token]]),
                    position_ids=decode_pos[:, :, step - 1 : step],
                    cache=cache,
                )
                token = int(mx.argmax(logits[0, -1]).item())
                out.append(token)
                step += 1
                continue

            verify = model.step_many(
                input_ids=mx.array([[token, *draft]]),
                position_ids=decode_pos[:, :, step - 1 : step + len(draft)],
                cache=cache,
            )
            pred = mx.argmax(verify, axis=-1)[0].tolist()  # already python ints
            n = 0
            while n < len(draft) and pred[n] == draft[n]:
                n += 1
            self._recent.append(n)
            # Stop paying for drafts once they stop paying for themselves. Costs a few
            # probes per pass rather than a whole pass run at a loss, which is what
            # unstable audio -- music, crosstalk -- does to inter-partial agreement.
            if (
                len(self._recent) == self.RECENT
                and sum(self._recent) / self.RECENT < self.MIN_ACCEPTED
            ):
                self._paying = False
            # Rewind the KV the rejected tail wrote, or the cache no longer describes the
            # path we are on. Same trim as upstream's speculative loop.
            cache.trim(len(draft) - n)

            stop = False
            for tk in draft[:n]:
                token = tk
                out.append(tk)
                step += 1
                if (
                    step >= cfg.max_new_tokens
                    or tk in cfg.eos_token_ids
                    or _detect_repetition(out)
                ):
                    stop = True
                    break
            if stop:
                break
            token = pred[n]
            out.append(token)
            step += 1

        while out and out[-1] in cfg.eos_token_ids:
            out.pop()
        self._prev = list(out)
        _, text = parse_asr_output(tok.decode(out), user_language=self._language)
        return text


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

    def __init__(
        self,
        model: str = DEFAULT_ASR,
        aligner: str | None = DEFAULT_ALIGNER,
        dtype: str = "fp16",
        on_status=None,
        context: str = "",
    ):
        self.context = context
        try:
            import mlx.core as mx  # ty: ignore[unresolved-import]
            from mlx_qwen3_asr import ForcedAligner, Session
        except ImportError as e:
            raise BackendUnavailable(
                "the mlx backend needs mlx-qwen3-asr; install it with "
                "`uv sync --extra mlx`, or use --backend torch"
            ) from e

        resolved = _dtype(
            {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}, dtype, "mlx"
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
            context=self.context,
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

    def open_stream(
        self, *, language: str, chunk_sec: float = 2.0, max_context_sec: float = 30.0
    ) -> _MlxStream:
        return _MlxStream(self, language, chunk_sec, max_context_sec)

    def open_draft(self, *, language: str) -> _MlxDraftDecoder:
        return _MlxDraftDecoder(self, language)


# ---------------------------------------------------------------- registry

BACKENDS = {"torch": TorchBackend, "mlx": MlxBackend}


def available(name: str) -> bool:
    """Whether a backend's imports resolve, without loading any weights."""
    import importlib.util

    cls = BACKENDS.get(name)
    if cls is None:  # an unknown backend is not "available"
        return False
    return all(importlib.util.find_spec(m) is not None for m in cls.requires)


def resolve_dtype(backend: str, dtype: str | None) -> str:
    """`--dtype auto` means whichever precision that backend is actually fast in."""
    if dtype in (None, "", "auto"):
        cls = BACKENDS.get(backend)
        return cls.default_dtype if cls else "fp16"
    if dtype not in DTYPES:
        raise BackendUnavailable(
            f"unknown dtype {dtype!r}; choose from {', '.join(DTYPES)}"
        )
    return dtype


PARTIAL_MODES = ("reencode", "stream", "x-draft")


def open_partials(
    backend: Backend, *, mode: str, language: str, chunk_sec: float = 2.0
) -> PartialDecoder:
    """The partial decoder for a mode, falling back to re-encoding if it isn't available.

    Always returns one, so the engine holds a decoder rather than an optional decoder plus
    two branches. Check `.mode` against what you asked for to find out whether the backend
    could do it -- falling back is a cost, not a failure, and a session should say so
    rather than fail over a provisional pass.

    Both capabilities are probed here rather than in the engine, which is what keeps the
    method names and their keywords -- backend facts -- inside this module.
    """
    if mode == "stream" and isinstance(backend, Streaming):
        return backend.open_stream(language=language, chunk_sec=chunk_sec)
    if mode == "x-draft" and isinstance(backend, Drafting):
        return backend.open_draft(language=language)
    return _ReencodePartials(backend, language)


def load_backend(
    name: str,
    *,
    model: str | None = None,
    aligner: str | None = None,
    device: str = "mps",
    dtype: str | None = None,
    on_status=None,
    warmup: bool = True,
    align: bool = True,
    context: str = "",
) -> Backend:
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

    kwargs = {
        "model": model,
        "aligner": aligner,
        "dtype": dtype,
        "on_status": on_status,
        "context": context,
    }
    if cls.takes_device:
        kwargs["device"] = device
    # takes_device is a runtime discriminator over two constructors that genuinely differ:
    # MLX uses unified memory and has no device to place anything on, so it does not take
    # the argument at all. A checker can only see the union of the two signatures.
    backend = cls(**kwargs)  # ty: ignore[invalid-argument-type]

    if warmup:
        say = on_status or (lambda m: None)
        say("warming up")
        # First inference pays graph-compilation cost; eat it before capture starts. Warm
        # the path that will actually run: asking for timestamps here without an aligner
        # raises, and warming the aligned path when nothing will use it is wasted load.
        backend.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            language="English",
            timestamps=align,
        )
    return backend
