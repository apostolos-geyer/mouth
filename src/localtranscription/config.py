"""A config file, so the flags you always pass stop being flags you always pass.

    ~/.config/localtranscription/config.toml     ($XDG_CONFIG_HOME honoured)

Nothing here is a new setting: every key is an existing flag, and all the file does is
change what that flag *defaults* to. An explicit flag always wins -- not by convention
but by construction, because this feeds Click's `default_map` and the layering happens
inside the parser. There is no per-flag plumbing to forget and no "was this passed?"
sentinel to get wrong, which is the failure mode of every hand-rolled version of this.

Bare keys apply to the commands that listen, plus the two that describe them -- `tune`
measures the stack the others run, and `cadence` prints what a schedule costs, so both
are wrong unless they see the same settings. Anything else takes a table named after the
command, because the same flag name does not mean the same thing everywhere: `--threshold` is an RMS gate to a session and a cosine distance to `diarize`,
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

#: Which commands a bare key applies to: the ones that open a session and share an option
#: vocabulary, plus the two that exist to describe them: `tune`, which measures the stack
#: the others run, and `cadence`, which prints what a partial schedule costs. Both were
#: omitted at first and both were wrong for it -- tune benchmarked stock torch while the
#: config pointed every real command at a quantised MLX checkpoint, and `lt cadence`
#: simulated the shipped schedule rather than the one configured.
#: See the module docstring for why this still isn't "all of them".
SESSION = ("tui", "cli", "dictate", "tune", "cadence")


def session_list() -> str:
    """The SESSION tuple as prose, so no doc string has to restate it.

    It was written out by hand in five places and had already drifted in one of them.
    """
    names = [f"`lt {c}`" for c in SESSION]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


_TEMPLATE = """\
# localtranscription -- defaults for the flags you'd otherwise type every time.
# A flag on the command line still beats anything in here.
#
# Bare keys below apply to:
#   {session}
# Every other command takes a table. TOML rule worth knowing: bare keys must come
# before the first [table] or they land inside it.

# backend = "mlx"                   # torch | mlx            (`lt backends`)
# model = "qwen3-asr-1.7b-q8g64"    # HF repo id, or a local checkpoint from `lt quantize`
# language = "English"              # `lt languages`
# mic = 2                           # `lt devices`
# record = true                     # keep per-utterance audio + manifest
# out = "~/Documents/transcripts"   # where transcripts land

# [dictate]
# hold = true                       # a pause is not the end; the signal is
# record = false
# wait = 8.0                        # give up if speech hasn't started

# [tui]
# partials = "stream"               # needs backend = "mlx"
# x-partial-draft = true            # experimental: ~1.6x cheaper reencode partials

# [diarize]
# threshold = 0.65                  # cosine distance -- NOT the VAD threshold above
# max_speakers = 8
"""


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

    shared = {alias for cmd in SESSION for alias in params.get(cmd, {})}
    for key, value in bare.items():
        name = _norm(key)
        if name not in shared:
            # Real option, wrong scope: say so, because "unknown option" would be a lie
            # and the fix is a table, not a spelling.
            elsewhere = sorted(c for c, al in params.items() if name in al)
            if elsewhere:
                raise ConfigError(
                    f"{_flag(name)} belongs to `lt {elsewhere[0]}`, which needs its own "
                    f"table: put it under [{elsewhere[0]}]."
                )
            raise ConfigError(f"unknown option {_flag(name)}.{_suggest(key, shared)}")
        for cmd in SESSION:
            alias = params.get(cmd, {})
            if name in alias:
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
    return _TEMPLATE.format(session="  ".join(f"lt {c}" for c in SESSION))


def write_template(path: Path) -> None:
    """Drop a starter config. Never clobbers: that file is hand-written by definition."""
    if path.exists():
        raise ConfigError(f"{path}: already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template())


def save(path: Path, text: str) -> Path | None:
    """Write a config, keeping the previous one alongside it.

    Returns where the previous one went, or None if there wasn't one.

    Here rather than in the caller because this module owns the file: `lt tune --write`
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
