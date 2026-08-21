#!/usr/bin/env python3
"""Static call graph over the package, grouped by the proposed workspace split.

    uv run python tools/callgraph.py src/localtranscription

Exists because an import graph does not answer the question this repo's split turns on.
`formats.py` imports nothing internal at module scope and calls `diarize.label_words`
from inside two function bodies; only a call graph shows that.

Resolution is deliberately conservative -- a call it cannot resolve is counted, never
guessed, and the count is printed so the edge totals read as a floor rather than a truth:

    f()          a name bound in this scope or an enclosing one (import, def, class)
    mod.f()      `mod` bound to a package module by `from . import mod`
    Cls()        a bound class  =>  edge to Cls.__init__
    self.m()     a method on the enclosing class or an in-package base
    obj.m()      UNRESOLVED -- which is how hooks, backend, source and partials are
                 reached, so run tools/trace_calls.py for those.

Function-scope imports are honoured; this codebase relies on them heavily and disables
ruff's PLC0415 to say so.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

# Which distribution each top-level module is headed for. Keep in step with
# docs/plans/2026-08-21-uv-workspace.md §2.2 -- this is what makes the seam report mean
# anything, and a module missing here shows up as "?" rather than being silently core.
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


class Graph:
    def __init__(self, root: Path):
        self.root = root
        self.pkg = root.name
        self.modules = {self._modname(p): p for p in sorted(root.rglob("*.py"))}
        self.trees = {m: ast.parse(p.read_text(), str(p)) for m, p in self.modules.items()}
        self.defs: dict[str, str] = {}
        self.methods: dict[str, dict[str, str]] = defaultdict(dict)
        self.bases: dict[str, list[str]] = {}
        self.edges: set[tuple[str, str]] = set()
        self.unresolved: dict[str, int] = defaultdict(int)
        for mod, tree in self.trees.items():
            self._collect(tree, mod, mod, None)
        for mod in self.trees:
            self._walk(mod)

    def _modname(self, path: Path) -> str:
        parts = [self.pkg, *path.relative_to(self.root).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    # ------------------------------------------------------------------ symbols

    def _collect(self, node, mod: str, prefix: str, cls: str | None):
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                q = f"{prefix}.{child.name}"
                self.defs[q] = "class"
                self.bases[q] = [ast.unparse(b) for b in child.bases]
                self._collect(child, mod, q, q)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                q = f"{prefix}.{child.name}"
                self.defs[q] = "method" if cls else "func"
                if cls:
                    self.methods[cls][child.name] = q
                self._collect(child, mod, q, None)

    def _bindings(self, stmt, mod: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                if a.name.split(".")[0] == self.pkg:
                    out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level:
                base = mod.split(".")
                is_pkg = self.modules.get(mod) and self.modules[mod].name == "__init__.py"
                path = base if is_pkg else base[:-1]
                up = stmt.level - 1
                root = ".".join(path[: len(path) - up] if up else path)
                target = f"{root}.{stmt.module}" if stmt.module else root
            elif stmt.module and stmt.module.split(".")[0] == self.pkg:
                target = stmt.module
            else:
                return out
            for a in stmt.names:
                out[a.asname or a.name] = f"{target}.{a.name}"
        return out

    # ------------------------------------------------------------------ calls

    def _add(self, caller: str, callee: str):
        if self.defs.get(callee) == "class":
            callee = self.methods.get(callee, {}).get("__init__", callee)
        if caller != callee:
            self.edges.add((caller, callee))

    def _call(self, func, names: dict[str, str], caller: str, cls: str | None):
        if isinstance(func, ast.Name):
            bound = names.get(func.id)
            if bound and bound in self.defs:
                self._add(caller, bound)
            elif bound not in self.modules:
                self.unresolved[func.id] += 1
            return
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base, attr = func.value.id, func.attr
            if base == "self" and cls:
                hit = self.methods.get(cls, {}).get(attr)
                if not hit:
                    for b in self.bases.get(cls, []):
                        hit = self.methods.get(names.get(b, ""), {}).get(attr)
                        if hit:
                            break
                if hit:
                    self._add(caller, hit)
                    return
            bound = names.get(base)
            if bound in self.modules and f"{bound}.{attr}" in self.defs:
                self._add(caller, f"{bound}.{attr}")
                return
            if bound in self.defs and self.defs[bound] == "class":
                hit = self.methods.get(bound, {}).get(attr)
                if hit:
                    self._add(caller, hit)
                    return
        self.unresolved[f".{getattr(func, 'attr', '?')}"] += 1

    def _walk(self, mod: str):
        root: dict[str, str] = {}
        for node in self.trees[mod].body:
            root.update(self._bindings(node, mod))
        for q in self.defs:
            if q.rsplit(".", 1)[0] == mod:
                root[q.rsplit(".", 1)[1]] = q

        def descend(node, names: dict[str, str], owner: str, cls: str | None):
            local = dict(names)
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Import | ast.ImportFrom):
                        local.update(self._bindings(sub, mod))
            for stmt in node.body:
                if isinstance(stmt, ast.ClassDef):
                    q = f"{owner}.{stmt.name}"
                    local[stmt.name] = q
                    descend(stmt, local, q, q if q in self.defs else cls)
                elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    q = f"{owner}.{stmt.name}"
                    local[stmt.name] = q
                    # cls is carried into the body, not reset: a method body -- and any
                    # function nested inside one -- still resolves self.foo() against the
                    # class it was defined in. Resetting it here silently drops every
                    # intra-class edge, which is 12 of them in this tree.
                    descend(stmt, local, q, cls)
                else:
                    for n in ast.walk(stmt):
                        if isinstance(
                            n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                        ):
                            continue
                        if isinstance(n, ast.Call):
                            self._call(n.func, local, owner, cls)

        descend(self.trees[mod], root, mod, None)

    # ------------------------------------------------------------------ report

    def top(self, q: str) -> str:
        p = q.split(".")[1:]
        return ".".join(p[:2]) if p[0] == "diarize" and len(p) > 1 else p[0]

    def dist(self, q: str) -> str:
        return ASSIGN.get(q.split(".")[1], "?")

    def report(self):
        print(
            f"defs={len(self.defs)}  resolved edges={len(self.edges)}  "
            f"unresolved call sites={sum(self.unresolved.values())}"
        )
        cross, intra = defaultdict(list), defaultdict(int)
        for a, b in sorted(self.edges):
            da, db = self.dist(a), self.dist(b)
            (
                cross[(da, db)].append((a, b))
                if da != db
                else intra.__setitem__(da, intra[da] + 1)
            )
        print("\nintra-package edges:", dict(sorted(intra.items())))
        print("\nCROSS-PACKAGE edges (the seam):")
        for (da, db), items in sorted(cross.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {da} -> {db}   ({len(items)})")
            for a, b in items:
                print(
                    f"      {a.removeprefix(self.pkg + '.'):<50} -> "
                    f"{b.removeprefix(self.pkg + '.')}"
                )
        weights: dict[tuple[str, str], int] = defaultdict(int)
        for a, b in self.edges:
            if self.top(a) != self.top(b):
                weights[(self.top(a), self.top(b))] += 1
        print("\nmodule-level call graph:")
        for (a, b), n in sorted(weights.items(), key=lambda kv: -kv[1]):
            seam = "  <== seam" if ASSIGN.get(a) != ASSIGN.get(b) else ""
            print(f"  {a:16} -> {b:22} {n:3}{seam}")


if __name__ == "__main__":
    g = Graph(Path(sys.argv[1] if len(sys.argv) > 1 else "src/localtranscription"))
    g.report()
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(
            json.dumps({"edges": sorted(g.edges), "assign": ASSIGN}, indent=1)
        )
