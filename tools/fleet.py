#!/usr/bin/env python3
"""Launch a FLEET of parallel Kaggle runs (mix of T4-GPU and CPU) for faster,
more diverse learning.

Each fleet member is a self-contained copy of the engine kernel with a different
config injected -- seed, time budget, accelerator, wide/narrow lean, governor
strength -- so the runs explore different regions of the search space AT ONCE.
Every run writes learning_ledger.json + explorer_findings_graph.json; attach one
run's output to the next (or merge the ledgers) and the colony learns across the
whole fleet, not one run at a time. GPU members exercise the capacity-heavy
skills; CPU members run the ridge-family search for free (no GPU quota) -- the
shipped blends are ridge-family anyway, so CPU members are first-class.

Subcommands (build is local + safe; push/collect/submit hit Kaggle):
  python tools/fleet.py build   [--kernel K] [--out DIR]   # write one kernel dir per member
  python tools/fleet.py push    [--out DIR]                # `kaggle kernels push` every member
  python tools/fleet.py status                             # poll every member's run
  python tools/fleet.py collect [--out DIR] [--submit]     # download outputs (+ submit each CSV)

The push/collect-submit steps export code/predictions to Kaggle; the agent is
blocked from that, so run them yourself or hand AUTOMATION.md's prompt to a tool
with shell access. Credentials come from ~/.config/worldexplorer/.env.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KERNEL = ROOT.parent / "kaggle" / "drw_world_explorer_v36" / "kernel.py"
DEFAULT_OUT = ROOT.parent / "kaggle" / "fleet"
ENV_PATH = Path.home() / ".config" / "worldexplorer" / ".env"
COMP = "drw-crypto-market-prediction"

# A diverse sweep -- 3 GPU + 3 CPU -- varying the wide<->narrow lean (the open
# question after v33 measured width_decay_corr=+0.26), governor strength, seed,
# and budget. All in the proven light-to-mid search zone (the sealed cliff).
FLEET = [
    {"name": "gpu-wide",     "gpu": True,  "ov": {"SEED": 7,   "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.8}},
    {"name": "gpu-sharp",    "gpu": True,  "ov": {"SEED": 123, "TIME_BUDGET_MIN": 90, "WIDTH_BIAS_START": 0.3}},
    {"name": "gpu-governor", "gpu": True,  "ov": {"SEED": 5,   "TIME_BUDGET_MIN": 90, "GOV_LAMBDA_SCALE": 1.0, "GOV_LAMBDA_MAX": 0.08}},
    {"name": "cpu-balanced", "gpu": False, "ov": {"SEED": 42,  "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.5}},
    {"name": "cpu-sharp",    "gpu": False, "ov": {"SEED": 99,  "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.3}},
    {"name": "cpu-light",    "gpu": False, "ov": {"SEED": 777, "TIME_BUDGET_MIN": 45}},
]


def slug(name: str) -> str:
    return f"drw-wx-{name}"


def load_env() -> dict:
    env = dict(os.environ)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def kaggle_user() -> str:
    user = load_env().get("KAGGLE_USERNAME")
    if not user:
        sys.exit("missing KAGGLE_USERNAME (set it in ~/.config/worldexplorer/.env or env)")
    return user


def selected_members(a) -> list[dict]:
    wanted = set(a.members or [])
    if not wanted:
        return list(FLEET)
    names = {m["name"] for m in FLEET}
    unknown = sorted(wanted - names)
    if unknown:
        sys.exit(f"unknown fleet member(s): {', '.join(unknown)}")
    return [m for m in FLEET if m["name"] in wanted]


def _inject(kernel_text: str, ov: dict, *, gpu: bool = False) -> str:
    """Append `CFG.<k> = <v>` overrides as the LAST thing before the run guard,
    so they win over the kernel's own built-in override block."""
    lines = "\n".join(f"CFG.{k} = {v!r}" for k, v in ov.items())
    guard = ""
    if gpu:
        guard = """
# Kaggle sometimes assigns older GPUs (for example P100/K80) to GPU-enabled
# batch kernels. The installed torch build may not contain kernels for those
# architectures; degrade those runs to the CPU-safe schedule instead of crashing.
try:
    _fleet_gpu_names = "|".join(_gpu_names()).lower()
    if any(name in _fleet_gpu_names for name in ("p100", "k80")):
        HAVE_TORCH = False
        N_GPUS = 0
        log("fleet_gpu_degraded", reason="unsupported_torch_gpu", gpu_names=_fleet_gpu_names)
except Exception as _fleet_exc:
    HAVE_TORCH = False
    N_GPUS = 0
    log("fleet_gpu_degraded", reason=f"gpu_probe_failed:{_fleet_exc}")
"""
    block = f"\n# ---- FLEET OVERRIDES (parallel-learning member) ----\n{lines}\n{guard}\n"
    anchor = 'if __name__ == "__main__":'
    i = kernel_text.rindex(anchor)
    return kernel_text[:i] + block + kernel_text[i:]


def cmd_build(a) -> None:
    base = Path(a.kernel).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = selected_members(a)
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(_inject(base, m["ov"], gpu=bool(m["gpu"])))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug(m['name'])}", "title": f"DRW WX {m['name']}",
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(m["gpu"]), "enable_internet": False,
            "competition_sources": [COMP]}, indent=2))
        print(f"built {m['name']:14s} {'GPU' if m['gpu'] else 'CPU'}  {m['ov']}")
    print(f"\n{len(members)} members -> {out}\n"
          f"next: python tools/fleet.py push --out {out}")


def cmd_push(a) -> None:
    out = Path(a.out)
    failures = 0
    for m in selected_members(a):
        d = out / m["name"]
        if not (d / "kernel.py").exists():
            print(f"skip {m['name']} (not built)"); continue
        print(f"+ kaggle kernels push -p {d}")
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=False)
        failures += int(r.returncode != 0)
    if failures:
        print(f"\npush finished with {failures} failed member(s); retry with --members when slots free")
    print("poll with: python tools/fleet.py status")


def cmd_status(a) -> None:
    user = kaggle_user()
    for m in selected_members(a):
        r = subprocess.run(["kaggle", "kernels", "status", f"{user}/{slug(m['name'])}"],
                           capture_output=True, text=True)
        print(f"{m['name']:14s} {(r.stdout + r.stderr).strip()[:90]}")


def cmd_collect(a) -> None:
    out = Path(a.out)
    user = kaggle_user()
    for m in selected_members(a):
        dest = out / m["name"] / "output"
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "kernels", "output", f"{user}/{slug(m['name'])}",
                           "-p", str(dest)], check=False)
        if r.returncode != 0:
            print(f"{m['name']:14s} output unavailable")
            continue
        sub = dest / "submission.csv"
        if sub.exists():
            print(f"{m['name']:14s} submission.csv ready ({sub})")
            if a.submit:
                subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                                "-f", str(sub), "-m", f"fleet {m['name']}"], check=False)
        else:
            print(f"{m['name']:14s} no submission.csv yet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="parallel Kaggle fleet launcher")
    sub = ap.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build"); b.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--members", nargs="*", default=None); b.set_defaults(fn=cmd_build)
    p = sub.add_parser("push"); p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--members", nargs="*", default=None); p.set_defaults(fn=cmd_push)
    s = sub.add_parser("status"); s.add_argument("--members", nargs="*", default=None); s.set_defaults(fn=cmd_status)
    c = sub.add_parser("collect"); c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument("--members", nargs="*", default=None)
    c.add_argument("--submit", action="store_true"); c.set_defaults(fn=cmd_collect)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
