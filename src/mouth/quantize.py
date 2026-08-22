"""Build a quantised MLX checkpoint from an upstream one.

Quantisation is the single largest lever on this machine -- 8-bit is 2.2x faster than the
bf16/MPS default and 4.8x faster than the same weights in MLX fp16 -- and it is a property
of weights on disk, not a runtime flag. So it gets its own step, and `--model` points at
the result.

mlx-qwen3-asr ships convert.quantize_model but not the repo's scripts/convert.py, so the
save side is reproduced here: remapped weights plus a quantization_config.json that its
loader (and ours, which additionally honours `mode`) reads back.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import paths

# Copied alongside the weights so the checkpoint is self-contained -- Session() resolves
# the tokenizer from the model directory, and a bare safetensors file has no tokenizer.
SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)

# Group size each mode actually supports. affine is free-ish; the float modes are fixed by
# the format (mx.quantize raises rather than rounding), so we pick for the caller.
MODE_GROUP_SIZE = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}
MODES = ("affine", "mxfp4", "mxfp8", "nvfp4")


def default_out(model: str, bits: int, group_size: int, mode: str) -> Path:
    """<cache>/models/<name>-q8g64 -- readable at a glance in `m models`.

    Cache, not data: these are GBs and this command rebuilds any of them from the
    upstream weights, so losing the directory costs time rather than work.
    """
    stem = Path(model).name.lower()
    tag = f"q{bits}g{group_size}" if mode == "affine" else f"{mode}g{group_size}"
    return paths.models_dir() / f"{stem}-{tag}"


def quantize(
    model: str,
    *,
    bits: int = 8,
    group_size: int | None = None,
    mode: str = "affine",
    out: Path | None = None,
    on_status=None,
) -> Path:
    """Quantise `model` and return the directory it was written to."""
    import mlx.core as mx  # ty: ignore[unresolved-import]
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_qwen3_asr.config import Qwen3ASRConfig
    from mlx_qwen3_asr.convert import remap_weights
    from mlx_qwen3_asr.load_models import (
        _load_safetensors,
        _materialize_tied_lm_head_weights,
        _resolve_path,
    )
    from mlx_qwen3_asr.model import Qwen3ASRModel

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    group_size = MODE_GROUP_SIZE.get(mode, group_size if group_size is not None else 64)
    if mode != "affine":
        bits = 8 if mode == "mxfp8" else 4  # the format fixes the width

    say = on_status or (lambda m: None)
    out = Path(out) if out else default_out(model, bits, group_size, mode)

    say(f"reading {model}")
    src = _resolve_path(model)
    config = Qwen3ASRConfig.from_dict(json.loads((src / "config.json").read_text()))
    weights = _materialize_tied_lm_head_weights(
        remap_weights(_load_safetensors(src)), config
    )
    net = Qwen3ASRModel(config)
    net.load_weights(list(weights.items()))
    mx.eval(net.parameters())

    say(f"quantising {mode} {bits}-bit, group {group_size}")
    nn.quantize(net, bits=bits, group_size=group_size, mode=mode)
    mx.eval(net.parameters())

    say(f"writing {out}")
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out / "model.safetensors"),
        dict(tree_flatten(net.parameters())),
        metadata={"format": "mlx"},
    )
    for name in SIDECARS:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    (out / "quantization_config.json").write_text(
        json.dumps({"bits": bits, "group_size": group_size, "mode": mode}, indent=2)
    )
    return out


def size_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.glob("*.safetensors")) / 1e9
