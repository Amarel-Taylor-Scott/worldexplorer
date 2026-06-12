#!/usr/bin/env python3
"""Pass B of the ExplorerHarness.run decomposition: cut the rs.*-rewritten
run() into phase methods at the section-comment boundaries. Pure text moves
-- bodies keep their exact lines (same indentation depth under the new defs);
run() becomes a slim orchestrator.

Safety: every marker must match exactly one line, in order; the concatenated
slices must reproduce the original body byte-for-byte; any phase that ASSIGNS
a module global must carry the right `global` declaration (injected where the
original single declaration at the top of run() no longer covers it); the
output must compile.

Usage: python tools/split_run_phases.py engine_src/16_harness.py
"""
import ast
import sys
from pathlib import Path

# (method name, unique marker substring of its first line)
PHASES = [
    ("_setup", None),                       # from first body stmt after rs creation
    ("_load_data", "# ---- data ---"),
    ("_quarantine_probe", "# ---- SEALED HOLDOUT quarantine"),
    ("_build_atlas", "# ---- TERRAIN ATLAS"),
    ("_beacons", "# ---- BEACON FIELD (v15)"),
    ("_pre_scans", "# ---- SYMMETRY FIELD"),
    ("_phase1_explore", "# ---- PHASE 1: developmental explorers"),
    ("_raid1", "# ---- RAID 1 (v10)"),
    ("_phase2_evolve", "# ---- PHASE 2: metaheuristic evolution"),
    ("_raid2", "# ---- RAID 2: the predator attacks"),
    ("_ablation_dive", "# ---- CHAMPION ABLATION (v10)"),
    ("_trail_reports", "# ---- TOPOGRAPHY: texture every trail"),
    ("_select_members", "# ---- members: regime + uniqueness"),
    ("_ensemble", "rs.result = nested_ensemble("),
    ("_forward_holdout", "# ---- forward-drift check + forward gate"),
    ("_governor", "# ---- v27 RUNTIME COMPLEXITY-GENERALIZATION GOVERNOR"),
    ("_forensics", "# ---- v21 FORENSIC REGIME-SCIENCE"),
    ("_forward_gate", "# v18 FORWARD-GATE ERROR BARS"),
    ("_shipping_court", "# ---- v27 ANTI-OVERFIT SHIPPING COURT"),
    ("_shrink_chorus_shape", "# ---- v16 SHRUNK BLEND"),
    ("_health_alarms", "rs.monitor = HealthMonitor("),
    ("_sealed_audit", "# ---- SEALED HOLDOUT: evaluated ONCE"),
    ("_final_refit_submit", "# ---- final refit on FULL train"),
    ("_cairn_ledger", "# ---- CAIRN (v10)"),
    ("_chronicle", "rs.evo_summary = {}"),
    ("_summarize", "rs.summary = {"),
]
# phases that ASSIGN module globals but lost run()'s top declaration
INJECT_GLOBALS = {"_build_atlas": "        global ATLAS, GAUGE\n"}
KNOWN_GLOBALS = {"ATLAS", "GAUGE", "META", "PROFILE", "BEACONS", "FCLUST", "HABITAT"}


def main() -> None:
    path = Path(sys.argv[1])
    lines = path.read_text().splitlines(keepends=True)
    run_i = next(i for i, l in enumerate(lines) if l.startswith("    def run(self)"))
    rs_i = run_i + 1
    assert lines[rs_i].strip() == "rs = _RunState()", lines[rs_i]
    ret_i = next(i for i, l in enumerate(lines) if l.startswith("        return rs.summary"))
    body = lines[rs_i + 1: ret_i + 1]        # everything after rs creation incl. return

    # locate boundaries
    bounds = [0]
    for name, marker in PHASES[1:]:
        hits = [i for i, l in enumerate(body) if marker in l]
        if len(hits) != 1:
            raise SystemExit(f"marker for {name} matched {len(hits)} lines: {marker!r}")
        bounds.append(hits[0])
    if bounds != sorted(bounds):
        raise SystemExit("phase markers out of order")
    bounds.append(len(body))

    slices = [body[bounds[i]: bounds[i + 1]] for i in range(len(PHASES))]
    if "".join("".join(s) for s in slices) != "".join(body):
        raise SystemExit("reassembly mismatch -- slices do not reproduce the body")

    new = lines[:run_i]
    new.append("    def run(self) -> dict[str, Any]:\n")
    new.append("        rs = _RunState()\n")
    for name, _ in PHASES[:-1]:
        new.append(f"        self.{name}(rs)\n")
    new.append(f"        return self.{PHASES[-1][0]}(rs)\n")
    for (name, _), sl in zip(PHASES, slices):
        ret_t = "dict[str, Any]" if name == "_summarize" else "None"
        new.append("\n")
        new.append(f'    def {name}(self, rs: "_RunState") -> {ret_t}:\n')
        if name in INJECT_GLOBALS:
            new.append(INJECT_GLOBALS[name])
        new.extend(sl)
    new.extend(lines[ret_i + 1:])

    out = "".join(new)
    compile(out, str(path), "exec")

    # global-store check: any phase assigning a known module global must
    # declare it (the assignment would otherwise silently create a local)
    tree = ast.parse(out)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExplorerHarness")
    problems = []
    for meth in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        declared = {g for node in ast.walk(meth) for g in
                    (node.names if isinstance(node, ast.Global) else [])}
        stored = {node.id for node in ast.walk(meth)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                  and node.id in KNOWN_GLOBALS}
        if stored - declared:
            problems.append(f"{meth.name}: stores {sorted(stored - declared)} without global decl")
    if problems:
        raise SystemExit("global-store check failed:\n  " + "\n  ".join(problems))

    path.write_text(out)
    print(f"split run() into {len(PHASES)} phase methods "
          f"({ret_i - rs_i} body lines redistributed); run() is now the orchestrator")


if __name__ == "__main__":
    main()
