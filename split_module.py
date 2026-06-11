#!/usr/bin/env python3
"""Byte-preserving splitter for engine_src monolith modules.

The engine is rebuilt by concatenating engine_src/*.py in filename order, so a
split is safe ONLY if it is byte-identical after rebuild. This tool cuts a module
at TOP-LEVEL statement boundaries (def/class/assignment/if...) into ordered
sub-files `<stem>__NN<ext>` whose concatenation equals the original exactly --
zero behavior change, a navigable tree. Files with a single top-level node
(e.g. a lone big class) are skipped: those need a method-level refactor, not a
byte split.

    python split_module.py engine_src/07_skills.py --target 380          # dry run
    python split_module.py engine_src/07_skills.py --target 380 --apply  # do it
"""
import ast
import pathlib
import sys


def _start_line(node) -> int:
    if getattr(node, "decorator_list", None):
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


def split_file(path: pathlib.Path, target_lines: int = 380, apply: bool = False):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    tops = list(ast.parse(text).body)
    if len(tops) < 2:
        print(f"SKIP {path.name}: {len(tops)} top-level node(s) -- not byte-splittable (needs a method refactor)")
        return []
    starts = [_start_line(n) for n in tops]              # 1-based start of each top node
    cuts, last = [0], 0
    for i in range(1, len(tops)):
        s = starts[i] - 1                                # 0-based line index of node i
        if s - last >= target_lines:
            cuts.append(s)
            last = s
    cuts.append(len(lines))
    if len(cuts) <= 2:
        print(f"SKIP {path.name}: {len(lines)} lines fit in one chunk under target={target_lines}")
        return []
    parts = ["".join(lines[cuts[i]:cuts[i + 1]]) for i in range(len(cuts) - 1)]
    assert "".join(parts) == text, f"INTERNAL: split of {path.name} is NOT byte-identical"
    names = [f"{path.stem}__{i:02d}{path.suffix}" for i in range(len(parts))]
    sizes = [len(p.splitlines()) for p in parts]
    print(f"SPLIT {path.name} ({len(lines)} lines) -> {len(parts)} parts {sizes}: {names}")
    if apply:
        for nm, ch in zip(names, parts):
            (path.parent / nm).write_text(ch)
        path.unlink()
        print(f"  applied: wrote {len(parts)} parts, removed {path.name}")
    return names


if __name__ == "__main__":
    args = sys.argv[1:]
    apply = "--apply" in args
    target = 380
    if "--target" in args:
        target = int(args[args.index("--target") + 1])
    files = [a for a in args if not a.startswith("--") and a.isascii() and a.endswith(".py")]
    for f in files:
        split_file(pathlib.Path(f), target_lines=target, apply=apply)
