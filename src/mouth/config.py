"""A config file, so the flags you always pass stop being flags you always pass.

    ~/.config/mouth/config.toml     ($XDG_CONFIG_HOME honoured)

Nothing here is a new setting: every key is an existing flag, and all the file does is
change what that flag *defaults* to. An explicit flag always wins -- not by convention
but by construction, because this feeds Click's `default_map` and the layering happens
inside the parser. There is no per-flag plumbing to forget and no "was this passed?"
sentinel to get wrong, which is the failure mode of every hand-rolled version of this.

A bare key reaches every command that has that option, which is almost all of them: a
setting for "how this machine transcribes" is wrong for `m tui` and right for `m cli`
only by accident. The exceptions are named in EXCLUDED below and there are three, each
one a command where a flag name means something else -- `--threshold` is an RMS gate to a
session and a cosine distance to `diarize`, and a bare key reaching both would collapse
every speaker into one. Those take a table named after the command: `--threshold` is an RMS gate to a session and a cosine distance to `diarize`,
and a bare key that reached both would quietly ruin one of them.

    backend = "mlx"
    model = "qwen3-asr-1.7b-q8g64"
    language = "Greek"

    [dictate]
    hold = true

    [diarize]
    threshold = 0.7

Unknown keys are an error rather than a shrug. A config file is write-once and read
never; a typo that silently does nothing is a setting you believe is on for months.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import paths

#: Commands a bare key must NOT reach, and why -- the only place a flag name means
#: something different from what it means everywhere else.
#:
#: This used to be the other way round: a hand-written list of commands bare keys *did*
#: reach. Three commands were added after it and all three were wrong for it -- `tune`
#: benchmarked stock torch while the config pointed everything else at a quantised MLX
#: checkpoint, `cadence` printed the shipped schedule rather than the configured one, and
#: `transcribe` gated speech at 0.3s against a config asking for 0.15s. Each was a silent
#: wrong answer, not a failure, and each needed a fifth copy of the list to be updated.
#:
#: Inverted, a new command inherits the settings by default and only an actual name
#: collision needs writing down -- which is a fact about the flag, visible where the flag
#: is declared, rather than a fact about the roster.
EXCLUDED = {
    "diarize": "--threshold is a cosine distance between voices, not an RMS gate, and "
    "--out is one RTTM file rather than a directory",
    "quantize": "--model is the checkpoint to convert and --out where to write it; both "
    "are the opposite of what they mean to a session",
    "models": "--dir is where checkpoints are looked for, not where anything is written",
}


def reaches(command: str) -> bool:
    """Whether a bare key is allowed to reach this command."""
    return command not in EXCLUDED


_TEMPLATE = """\
# mouth -- defaults for the flags you'd otherwise type every time.
# A flag on the command line still beats anything in here.
#
# Bare keys below reach every command that has the option. A few commands read a
# flag name differently ({excluded}) and take a table instead.
# TOML rule worth knowing: bare keys must come before the first [table] or they
# land inside it.

# backend = "mlx"                   # torch | mlx            (`m backends`)
# model = "qwen3-asr-1.7b-q8g64"    # HF repo id, or a local checkpoint from `m quantize`
# language = "English"              # `m languages`
# mic = 2                           # `m devices`
# record = true                     # keep per-utterance audio + manifest
# out = "~/Documents/transcripts"   # where transcripts land

# [dictate]
# hold = true                       # a pause is not the end; the signal is
# record = false
# wait = 8.0                        # give up if speech hasn't started

# [tui]
# partials = "stream"               # needs backend = "mlx"

# [diarize]
# threshold = 0.65                  # cosine distance -- NOT the VAD threshold above
# max_speakers = 8
"""


#: Keys that used to exist, and what replaced them. A config file is written once and
#: read never, so "unknown option --x-partial-draft" would be a true statement that
#: teaches nothing -- the option did exist, and the setting still does.
RENAMED = {"x_partial_draft": 'partials = "x-draft"'}


class ConfigError(Exception):
    """A config file that exists but can't be honoured. Always names the file."""


def locate(explicit: Path | None = None) -> Path | None:
    """The config file to read, or None if there is none.

    A `--config` that doesn't exist is an error, not a fallback: it was named. A missing
    default file just means no config.
    """
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_file():
            raise ConfigError(f"{path}: no such config file")
        return path
    default = paths.config_file()
    return default if default.is_file() else None


def read(path: Path) -> dict[str, Any]:
    """Parse a config file. Both failure modes name the file, since --config may have."""
    import tomllib

    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: {e.strerror or e}") from e


def _norm(key: str) -> str:
    """`--max-gap` in the shell, `max-gap` or `max_gap` in the file. One option."""
    return key.replace("-", "_")


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _suggest(key: str, known, fmt=_flag) -> str:
    import difflib  # error path only, and app.py imports this module on every command

    near = difflib.get_close_matches(_norm(key), sorted(known), n=1)
    return f" Did you mean {fmt(near[0])}?" if near else ""


def _value(value: Any) -> Any:
    """`~/notes` means the same thing here as it does in the shell.

    Click casts these to their parameter types for us -- str to Path, int to float -- but
    it does not expand `~`, and a config file is exactly where someone writes one.
    """
    if isinstance(value, str) and value.startswith("~"):
        return str(Path(value).expanduser())
    return value


def default_map(
    data: Mapping[str, Any], params: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    """Layer a parsed config into Click's per-command default map.

    `params` is {command: {written form: parameter name}}, read off the built CLI -- so
    the file is validated against the real options, and a flag renamed upstream turns
    into an error here rather than a silently dead setting.
    """
    bare = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    out: dict[str, dict[str, Any]] = {}

    shared = {alias for cmd, al in params.items() if reaches(cmd) for alias in al}
    for key, value in bare.items():
        name = _norm(key)
        if name not in shared:
            # Real option, wrong scope: say so, because "unknown option" would be a lie
            # and the fix is a table, not a spelling.
            elsewhere = sorted(c for c, al in params.items() if name in al)
            if elsewhere:
                raise ConfigError(
                    f"{_flag(name)} belongs to `m {elsewhere[0]}`, where it means "
                    f"something else ({EXCLUDED[elsewhere[0]]}). Put it under "
                    f"[{elsewhere[0]}] if that is what you meant."
                )
            if name in RENAMED:
                raise ConfigError(
                    f"{_flag(name)} is now {RENAMED[name]}. It is a value of --partials, "
                    f"not a flag beside it."
                )
            raise ConfigError(f"unknown option {_flag(name)}.{_suggest(key, shared)}")
        for cmd, alias in params.items():
            if reaches(cmd) and name in alias:
                out.setdefault(cmd, {})[alias[name]] = _value(value)

    for cmd, table in tables.items():
        alias = params.get(cmd)
        if alias is None:
            raise ConfigError(f"[{cmd}] is not a command.{_suggest(cmd, params, fmt=str)}")
        for key, value in table.items():
            name = _norm(key)
            if name not in alias:
                raise ConfigError(f"[{cmd}] has no {_flag(name)}.{_suggest(key, alias)}")
            out.setdefault(cmd, {})[alias[name]] = _value(value)
    return out


def template() -> str:
    """The starter config, with the command list filled in from SESSION."""
    return _TEMPLATE.format(excluded=", ".join(f"m {c}" for c in sorted(EXCLUDED)))


def write_template(path: Path) -> None:
    """Drop a starter config. Never clobbers: that file is hand-written by definition."""
    if path.exists():
        raise ConfigError(f"{path}: already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template())


def save(path: Path, text: str) -> Path | None:
    """Write a config, keeping the previous one alongside it.

    Returns where the previous one went, or None if there wasn't one.

    Here rather than in the caller because this module owns the file: `m tune --write`
    was reaching past it to hardcode the location, the .bak rule and the mkdir, which is
    also how it came to ignore the --config path it had just been given.
    """
    path = path.expanduser()
    backup = None
    if path.exists():
        backup = path.with_suffix(".toml.bak")
        backup.write_text(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return backup
