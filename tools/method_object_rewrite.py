#!/usr/bin/env python3
"""Pass A of the ExplorerHarness.run decomposition (Replace Method with
Method Object): mechanically rewrite every run()-level local variable to an
attribute on a `rs` state object, so Pass B can cut run() into phase methods
as pure text moves.

Scope-exact via `symtable`, with a traversal that mirrors CPython's visit
order (a comprehension's FIRST iterable, lambda defaults and def
decorators/defaults/annotations evaluate in the ENCLOSING scope; everything
else in the child scope) -- so names are resolved exactly as the compiler
resolves them. Hazards handled explicitly:
  - `except ... as e` binders cannot be attributes -> left as plain locals
  - inner `def f` names: the def statement keeps its plain name; a
    `rs.f = f` export is inserted right after the def so cross-phase
    references (rewritten to rs.f) resolve
  - `nonlocal x` where every x got rewritten -> statement removed
Safety: every edit position is substring-verified before applying; after
rewriting, a resolver pass asserts no rewritten name still resolves outside
an inner scope (nothing missed); the output must compile.

Usage: python tools/method_object_rewrite.py <file.py> <ClassName.method> <state_var>
Rewrites the file in place (git is the undo).
"""
import ast
import sys
import symtable
from pathlib import Path

SCOPE_AST = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
COMP_AST = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def find_method(tree: ast.Module, cls_name: str, meth_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == meth_name:
                    return sub
    raise SystemExit(f"{cls_name}.{meth_name} not found")


def find_table(table: symtable.SymbolTable, cls_name: str, meth_name: str) -> symtable.SymbolTable:
    for ch in table.get_children():
        if ch.get_type() == "class" and ch.get_name() == cls_name:
            for f in ch.get_children():
                if f.get_name() == meth_name:
                    return f
    raise SystemExit(f"symtable for {cls_name}.{meth_name} not found")


class Walker:
    """Scope-chain traversal in CPython symtable visit order. Calls
    on_name(name_node, chain) for every ast.Name and on_except(handler) for
    every named except handler owned by the ROOT scope; records direct inner
    defs of the root scope."""

    def __init__(self, root_ast, root_table, on_name=None):
        self.on_name = on_name or (lambda n, c: None)
        self.except_names: set[str] = set()
        self.inner_defs: list[ast.FunctionDef] = []
        self.root_table = root_table
        self.iters = {id(root_table): iter(root_table.get_children())}
        self.body(root_ast, [root_table])

    def child_table(self, scope_node, chain):
        it = self.iters[id(chain[-1])]
        tbl = next(it)
        if tbl.get_lineno() != scope_node.lineno:
            raise SystemExit(f"scope pairing drift: ast@{scope_node.lineno} vs "
                             f"symtable {tbl.get_name()}@{tbl.get_lineno()}")
        self.iters[id(tbl)] = iter(tbl.get_children())
        return tbl

    def body(self, func_node, chain):
        for stmt in func_node.body:
            self.visit(stmt, chain)

    def visit(self, node, chain):
        if isinstance(node, ast.Name):
            self.on_name(node, chain)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                self.visit(dec, chain)
            a = node.args
            for d in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                self.visit(d, chain)
            for arg in (a.posonlyargs + a.args + a.kwonlyargs
                        + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else [])):
                if arg.annotation is not None:
                    self.visit(arg.annotation, chain)
            if node.returns is not None:
                self.visit(node.returns, chain)
            if chain[-1] is self.root_table:
                self.inner_defs.append(node)
            tbl = self.child_table(node, chain)
            self.body(node, chain + [tbl])
            return
        if isinstance(node, ast.Lambda):
            a = node.args
            for d in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                self.visit(d, chain)
            tbl = self.child_table(node, chain)
            self.visit(node.body, chain + [tbl])
            return
        if isinstance(node, COMP_AST):
            self.visit(node.generators[0].iter, chain)        # enclosing scope!
            tbl = self.child_table(node, chain)
            inner = chain + [tbl]
            for gi, gen in enumerate(node.generators):
                self.visit(gen.target, inner)
                if gi > 0:
                    self.visit(gen.iter, inner)
                for cond in gen.ifs:
                    self.visit(cond, inner)
            if isinstance(node, ast.DictComp):
                self.visit(node.key, inner)
                self.visit(node.value, inner)
            else:
                self.visit(node.elt, inner)
            return
        if isinstance(node, ast.ExceptHandler) and node.name and chain[-1] is self.root_table:
            self.except_names.add(node.name)
        for child in ast.iter_child_nodes(node):
            self.visit(child, chain)


def resolves_to(name: str, chain: list):
    """The symtable scope this name binds in (innermost-out), or None."""
    try:
        sym = chain[-1].lookup(name)
        if sym.is_global():
            return None
    except KeyError:
        pass
    for scope in reversed(chain):
        try:
            if scope.lookup(name).is_local():
                return scope
        except KeyError:
            continue
    return None


def main() -> None:
    path = Path(sys.argv[1])
    cls_name, meth_name = sys.argv[2].split(".")
    rs = sys.argv[3]
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    run_ast = find_method(tree, cls_name, meth_name)
    run_st = find_table(symtable.symtable(src, str(path), "exec"), cls_name, meth_name)

    params = {s.get_name() for s in run_st.get_symbols() if s.is_parameter()}
    locals_all = {s.get_name() for s in run_st.get_symbols()
                  if s.is_local() and not s.is_parameter()}

    probe = Walker(run_ast, run_st)          # discover except binders + inner defs
    rewrite = locals_all - probe.except_names - params
    if rs in rewrite or rs in params:
        raise SystemExit(f"state var {rs!r} collides with an existing name")

    edits = []                                # (lineno, col, name)

    def on_name(node, chain):
        if node.id in rewrite and resolves_to(node.id, chain) is run_st:
            edits.append((node.lineno, node.col_offset, node.id))

    Walker(run_ast, run_st, on_name=on_name)

    # nonlocal statements whose every name is rewritten -> drop the line
    drop_lines = set()
    for child in ast.walk(run_ast):
        if isinstance(child, ast.Nonlocal):
            if all(n in rewrite for n in child.names):
                if child.lineno != child.end_lineno:
                    raise SystemExit("multiline nonlocal unsupported")
                drop_lines.add(child.lineno)
            elif any(n in rewrite for n in child.names):
                raise SystemExit(f"mixed nonlocal at line {child.lineno}")

    # apply edits right-to-left per line, verifying substrings
    by_line: dict[int, list] = {}
    for ln, col, name in edits:
        by_line.setdefault(ln, []).append((col, name))
    for ln, lst in by_line.items():
        text = lines[ln - 1]
        for col, name in sorted(set(lst), reverse=True):
            if text[col:col + len(name)] != name:
                raise SystemExit(f"verify failed at {ln}:{col} expected {name!r} "
                                 f"got {text[col:col + len(name)]!r}")
            text = text[:col] + f"{rs}.{name}" + text[col + len(name):]
        lines[ln - 1] = text

    # insert `rs.f = f` after each inner def owned by run
    inserts = []                              # (after_lineno, text)
    for d in probe.inner_defs:
        indent = " " * d.col_offset
        inserts.append((d.end_lineno, f"{indent}{rs}.{d.name} = {d.name}\n"))
    # insert `rs = _RunState()` before the first body statement
    first = run_ast.body[0]
    inserts.append((first.lineno - 1, f"{' ' * first.col_offset}{rs} = _RunState()\n"))

    for ln in drop_lines:
        lines[ln - 1] = ""
    for after_ln, text in sorted(inserts, reverse=True):
        lines.insert(after_ln, text)

    header = ('class _RunState:\n'
              '    """Method-object state for ExplorerHarness.run: every cross-phase\n'
              '    local of the old monolithic run() lives here as an attribute\n'
              '    (rs.<name>), so the phase methods share dataflow without a\n'
              '    1500-line scope."""\n'
              '\n\n')
    out = header + "".join(lines)
    compile(out, str(path), "exec")

    # resolver post-check: nothing in `rewrite` may still resolve to run/global
    tree2 = ast.parse(out)
    run2 = find_method(tree2, cls_name, meth_name)
    st2 = find_table(symtable.symtable(out, str(path), "exec"), cls_name, meth_name)
    missed = []

    def check(node, chain):
        if node.id in rewrite:
            tgt = resolves_to(node.id, chain)
            if tgt is None or tgt is st2:
                missed.append((node.lineno, node.id))

    Walker(run2, st2, on_name=check)
    # inner-def names + `rs = _RunState()` legitimately remain locals of run
    allowed = {d.name for d in probe.inner_defs} | {rs}
    missed = [(ln, n) for ln, n in missed if n not in allowed]
    if missed:
        raise SystemExit(f"missed rewrites: {missed[:20]}")

    path.write_text(out)
    print(f"rewrote {len(edits)} references across {len(by_line)} lines; "
          f"{len(probe.inner_defs)} inner defs exported; {len(drop_lines)} nonlocal dropped; "
          f"{len(rewrite)} locals -> {rs}.*; except kept local: {sorted(probe.except_names)}")


if __name__ == "__main__":
    main()
