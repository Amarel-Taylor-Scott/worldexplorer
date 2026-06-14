#!/usr/bin/env python3
"""worldexplorer publish/submit mechanism -- ONE script for every external action.

The Claude Code safety classifier refuses to let the AGENT bulk-export code to
external destinations (GitHub push, Kaggle dataset upload). This script is the
sanctioned path: YOU run it (or you grant the agent the matching Bash allow-rule
-- see AUTOMATION.md), and it reads your credentials from the local vault so no
token ever appears on a command line.

Credentials (first found wins): ~/.config/worldexplorer/.env  or  env vars
  GITHUB_OWNER, GITHUB_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN

Subcommands:
  python tools/publish.py github          [--repo worldexplorer] [--private]
  python tools/publish.py kaggle-dataset  [--slug worldexplorer-engine]
  python tools/publish.py kernel-push <kernel.py> [--slug drw-world-explorer]
                                          [--gpu] [--comp drw-crypto-market-prediction]
                                          [--dataset OWNER/SLUG ...]
  python tools/publish.py submit <submission.csv>
                                          [--comp drw-crypto-market-prediction] [-m MSG]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ENV_PATH = Path.home() / ".config" / "worldexplorer" / ".env"
ROOT = Path(__file__).resolve().parent.parent          # the repo root (worldexplorer/)


def load_env() -> dict:
    env = dict(os.environ)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def sh(cmd: list, *, display=None, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in (display or cmd)))
    return subprocess.run(cmd, check=True, **kw)


def cmd_github(a, env) -> None:
    owner = env.get("GITHUB_OWNER") or env.get("GITHUB_USERNAME")
    tok = env.get("GITHUB_TOKEN")
    if not (owner and tok):
        sys.exit("missing GITHUB_OWNER / GITHUB_TOKEN (set them in ~/.config/worldexplorer/.env)")
    body = json.dumps({"name": a.repo, "private": bool(a.private),
                       "description": "world-explorer: zero-config self-improving tabular ML engine"}).encode()
    req = urllib.request.Request("https://api.github.com/user/repos", data=body,
                                 headers={"Authorization": f"token {tok}",
                                          "Accept": "application/vnd.github+json"})
    try:
        urllib.request.urlopen(req)
        print(f"created repo {owner}/{a.repo}")
    except Exception as e:
        print(f"repo create skipped ({e}); assuming it already exists")
    push_url = f"https://github.com/{owner}/{a.repo}.git"
    with tempfile.TemporaryDirectory(prefix="wx_git_askpass_") as td:
        askpass = Path(td) / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "prompt = ' '.join(sys.argv[1:]).lower()\n"
            "if 'username' in prompt:\n"
            "    print(os.environ.get('GITHUB_OWNER', 'x-access-token'))\n"
            "else:\n"
            "    print(os.environ['GITHUB_TOKEN'])\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        git_env = dict(os.environ)
        git_env.update({
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "GITHUB_OWNER": owner,
            "GITHUB_TOKEN": tok,
        })
        sh(["git", "-C", str(ROOT), "push", push_url, "HEAD:refs/heads/master"], env=git_env)
    subprocess.run(["git", "-C", str(ROOT), "remote", "remove", "origin"],
                   stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin",
                    f"https://github.com/{owner}/{a.repo}.git"])     # clean, token-less
    print(f"PUSHED -> https://github.com/{owner}/{a.repo}")


def cmd_kaggle_dataset(a, env) -> None:
    owner = env.get("KAGGLE_USERNAME")
    if not owner:
        sys.exit("missing KAGGLE_USERNAME")
    payload = Path(tempfile.mkdtemp(prefix="wx_ds_")) / "d"
    (payload / "worldexplorer").mkdir(parents=True)
    (payload / "wheelhouse").mkdir(parents=True)
    try:
        sh([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w",
            str(payload / "wheelhouse"), str(ROOT)])
        wheels = sorted((payload / "wheelhouse").glob("worldexplorer-*.whl"))
        if wheels:
            (payload / "LATEST_WHEEL.txt").write_text(wheels[-1].name + "\n", encoding="utf-8")
    except Exception as e:
        print(f"wheel build skipped ({e}); source-package fallback will remain available")
    for p in (ROOT / "worldexplorer").glob("*.py"):       # the importable package only
        shutil.copy(p, payload / "worldexplorer" / p.name)
    (payload / "dataset-metadata.json").write_text(json.dumps(
        {"title": a.slug, "id": f"{owner}/{a.slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    status = subprocess.run(["kaggle", "datasets", "status", f"{owner}/{a.slug}"],
                            capture_output=True, text=True)
    exists = status.returncode == 0 and "404" not in (status.stdout + status.stderr)
    if exists:
        sh(["kaggle", "datasets", "version", "-p", str(payload), "-r", "zip", "-m", "update engine"])
    else:
        sh(["kaggle", "datasets", "create", "-p", str(payload), "-r", "zip"])
    print(f"DATASET -> {owner}/{a.slug}  (attach this to the thin kernel for Internet-OFF runs)")


def cmd_kernel_push(a, env) -> None:
    owner = env.get("KAGGLE_USERNAME")
    if not owner:
        sys.exit("missing KAGGLE_USERNAME")
    d = Path(tempfile.mkdtemp(prefix="wx_k_"))
    shutil.copy(a.kernel, d / "kernel.py")
    meta = {"id": f"{owner}/{a.slug}", "title": a.slug.replace("-", " ").title(),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(a.gpu), "enable_internet": bool(a.internet),
            "competition_sources": [a.comp] if a.comp else [],
            "dataset_sources": list(a.dataset or [])}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    sh(["kaggle", "kernels", "push", "-p", str(d)])
    print(f"KERNEL pushed -> {owner}/{a.slug}; poll with: kaggle kernels status {owner}/{a.slug}")


def cmd_submit(a, env) -> None:
    if not os.path.exists(a.file):
        sys.exit(f"no such file: {a.file}")
    sh(["kaggle", "competitions", "submit", "-c", a.comp, "-f", a.file, "-m", a.message])
    print(f"SUBMITTED {a.file} -> {a.comp}; check: kaggle competitions submissions {a.comp}")


def main(argv=None) -> int:
    env = load_env()
    ap = argparse.ArgumentParser(description="worldexplorer publish/submit mechanism")
    sub = ap.add_subparsers(dest="action", required=True)

    g = sub.add_parser("github"); g.add_argument("--repo", default="worldexplorer")
    g.add_argument("--private", action="store_true"); g.set_defaults(fn=cmd_github)

    k = sub.add_parser("kaggle-dataset"); k.add_argument("--slug", default="worldexplorer-engine")
    k.set_defaults(fn=cmd_kaggle_dataset)

    p = sub.add_parser("kernel-push"); p.add_argument("kernel")
    p.add_argument("--slug", default="drw-world-explorer"); p.add_argument("--gpu", action="store_true")
    p.add_argument("--internet", dest="internet", action="store_true",
                   help="allow the kernel to pull code from GitHub at runtime")
    p.add_argument("--offline", dest="internet", action="store_false",
                   help="disable internet; attach a wheel/source dataset instead")
    p.add_argument("--comp", default="drw-crypto-market-prediction")
    p.add_argument("--dataset", nargs="*", default=None)
    p.set_defaults(internet=True)
    p.set_defaults(fn=cmd_kernel_push)

    s = sub.add_parser("submit"); s.add_argument("file")
    s.add_argument("--comp", default="drw-crypto-market-prediction")
    s.add_argument("-m", "--message", default="worldexplorer auto-submit"); s.set_defaults(fn=cmd_submit)

    a = ap.parse_args(argv)
    a.fn(a, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
