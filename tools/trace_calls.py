#!/usr/bin/env python3
"""Runtime call trace restricted to src/mouth, grouped by the proposed split.

    uv run python tools/trace_calls.py -- transcribe FILE --speakers --out /tmp/x
    uv run python tools/trace_calls.py --pytest -- tests/ -q

Catches what tools/callgraph.py cannot: dispatch through a variable. `hooks.segment()`,
`backend.transcribe()`, `source.frames()` and `partials.text()` are the whole extension
surface of this program and are invisible to any AST pass.

Threads are traced too -- the Transcriber and the mic source each run on their own, and
the busiest edge in the program (`vad.segment_utterances -> hooks.level`, once per 30ms
frame) is reached from inside the generator that `run_session` drives.

Two things to know before believing a number:
  * the profiler costs ~10x wall clock, and Textual's async timeouts fail under it.
  * the CLI is entered through `app.app()` rather than `app.main()`, because main() ends
    in os._exit(), which skips the write below. See app.py:1531.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

ASSIGN = {
    "paths": "core",
    "vad": "core",
    "audio": "core",
    "formats": "core",
    "recorder": "core",
    "sources": "core",
    "backends": "core",
    "engine": "core",
    "quantize": "core",
    "diarize": "diarize",
    "app": "cli",
    "tui": "cli",
    "tune": "cli",
    "config": "cli",
}

SRC = Path("src/mouth").resolve()
edges: Counter[tuple[str, str]] = Counter()
stacks: dict[int, list[str | None]] = {}


def qual(frame) -> str | None:
    try:
        rel = Path(frame.f_code.co_filename).resolve().relative_to(SRC)
    except (ValueError, OSError):
        return None
    mod = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
    return f"{mod}:{frame.f_code.co_qualname}"


def profiler(frame, event, arg):
    stack = stacks.setdefault(threading.get_ident(), [])
    if event == "call":
        q = qual(frame)
        if q is None:
            stack.append(None)
            return
        caller = next((x for x in reversed(stack) if x), None)
        if caller and caller != q:
            edges[(caller, q)] += 1
        stack.append(q)
    elif event == "return" and stack:
        stack.pop()


def top(mod: str) -> str:
    p = mod.split(".")
    return ".".join(p[:2]) if p[0] == "diarize" and len(p) > 1 else p[0]


def dist(qualname: str) -> str:
    return ASSIGN.get(qualname.split(":", maxsplit=1)[0].split(".", maxsplit=1)[0], "?")


def report():
    per_edge: dict[tuple[str, str], int] = defaultdict(int)
    per_pair: dict[tuple[str, str], int] = defaultdict(int)
    for (a, b), n in edges.items():
        ma, mb = a.split(":")[0], b.split(":")[0]
        if top(ma) != top(mb):
            per_edge[(top(ma), top(mb))] += 1
            per_pair[(top(ma), top(mb))] += n
    print(f"\ndistinct call edges: {len(edges)}")
    print("\nmodule call graph (edges / calls):")
    for k in sorted(per_pair, key=lambda k: -per_pair[k]):
        seam = "   <== seam" if ASSIGN.get(k[0]) != ASSIGN.get(k[1]) else ""
        print(f"  {k[0]:16} -> {k[1]:18} {per_edge[k]:3} /{per_pair[k]:9,}{seam}")

    print("\nevery cross-package call edge, by volume:")
    rows = [(n, a, b) for (a, b), n in edges.items() if dist(a) != dist(b)]
    by_dir: dict[tuple[str, str], list[int]] = defaultdict(list)
    for n, a, b in sorted(rows, reverse=True):
        by_dir[(dist(a), dist(b))].append(n)
        print(f"  {n:8,}  [{dist(a):7} -> {dist(b):7}]  {a:48} -> {b}")

    total = sum(n for n, _, _ in rows) or 1
    print("\nby direction:")
    for k, ns in sorted(by_dir.items(), key=lambda kv: -sum(kv[1])):
        print(
            f"  {k[0]:7} -> {k[1]:7}  {len(ns):3} edges  {sum(ns):8,} calls  "
            f"{sum(ns) / total:5.1%}"
        )


def main() -> int:
    args = sys.argv[1:]
    use_pytest = "--pytest" in args
    if "--" in args:
        args = args[args.index("--") + 1 :]
    code = 0
    if use_pytest:
        import pytest

        sys.setprofile(profiler)
        threading.setprofile(profiler)
        try:
            code = int(pytest.main(args))
        finally:
            sys.setprofile(None)
            threading.setprofile(None)
    else:
        # app() and not main(): main() ends in os._exit and would skip the report.
        from mouth.app import app

        sys.argv = ["m", *args]
        sys.setprofile(profiler)
        threading.setprofile(profiler)
        try:
            app()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
        finally:
            sys.setprofile(None)
            threading.setprofile(None)
    report()
    out = next((a for a in sys.argv if a.endswith(".json")), None)
    if out:
        Path(out).write_text(
            json.dumps(
                {
                    "edges": [
                        {"caller": a, "callee": b, "n": n}
                        for (a, b), n in edges.most_common()
                    ]
                },
                indent=1,
            )
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
