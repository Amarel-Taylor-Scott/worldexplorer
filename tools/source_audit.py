#!/usr/bin/env python3
"""Check that WorldExplorer source stays in GitHub, not in Kaggle kernels."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def require(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        fail(f"missing {path}")
    ok(f"found {path}")
    return p


def main() -> int:
    engine_files = sorted((ROOT / "engine_src").glob("*.py"))
    if len(engine_files) < 10:
        fail(f"engine_src has too few source modules: {len(engine_files)}")
    ok(f"engine_src modules: {len(engine_files)}")

    require("worldexplorer/__init__.py")
    require("worldexplorer/kaggle.py")
    require("worldexplorer/_engine.py")
    require("sync_engine.py")
    require("tools/fleet.py")
    require("tools/memory_matrices.py")
    require("tools/telemetry_guidance.py")
    require("tools/route_carve.py")
    require("kaggle/bootstrap_kernel.py")

    bootstrap = (ROOT / "kaggle/bootstrap_kernel.py").read_text(encoding="utf-8")
    bootstrap_lines = bootstrap.count("\n") + 1
    if bootstrap_lines > 320:
        fail(f"kaggle/bootstrap_kernel.py is too large for a slim bootstrap: {bootstrap_lines} lines")
    ok(f"bootstrap remains slim: {bootstrap_lines} lines")

    required_bootstrap_tokens = [
        '"source_policy": "github_first"',
        "def _install_github()",
        "wx.kaggle.run(CONFIG)",
    ]
    for token in required_bootstrap_tokens:
        if token not in bootstrap:
            fail(f"bootstrap missing {token!r}")
    ok("bootstrap is GitHub-first and delegates to wx.kaggle.run")

    forbidden_bootstrap_tokens = [
        "class HarnessConfig",
        "class TerrainAtlas",
        "def _fit_greedyols",
        "def run_lesson",
        "def robust_oos_select",
    ]
    for token in forbidden_bootstrap_tokens:
        if token in bootstrap:
            fail(f"bootstrap appears to contain engine logic: {token}")
    ok("bootstrap does not contain known engine internals")

    fleet = (ROOT / "tools/fleet.py").read_text(encoding="utf-8")
    required_fleet_tokens = [
        'default="git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git"',
        'br.add_argument("--repo-ref", default="master")',
        'bs.set_defaults(internet=True)',
        'br.set_defaults(internet=True)',
        '"source_mode": source_policy',
    ]
    for token in required_fleet_tokens:
        if token not in fleet:
            fail(f"fleet generator missing {token!r}")
    ok("fleet generator defaults to GitHub-first slim kernels")

    print("source audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
