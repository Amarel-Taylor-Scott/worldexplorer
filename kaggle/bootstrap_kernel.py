# ============================================================================
# worldexplorer -- THIN BOOTSTRAP KERNEL  (all the logic lives in a GitHub repo)
# ----------------------------------------------------------------------------
# Paste this WHOLE cell into one Kaggle code cell. It carries ONLY the config
# (where the data is, the target, the budget) + how to fetch the engine. The
# ~10k-line engine is no longer pasted -- it is downloaded from the public repo
# (Internet ON) or read from an attached dataset (Internet OFF).
#
# HOW TO RUN
#   A) INTERNET ON  (Settings -> Internet -> On): nothing else needed; the cell
#      pip-installs worldexplorer from GitHub.
#   B) INTERNET OFF (e.g. DRW, code competitions): upload the worldexplorer repo
#      as a Kaggle Dataset once, then Add Input -> that dataset, and set
#      CONFIG["engine_dataset"] to its path (or just leave it None -- the cell
#      auto-finds any attached folder containing worldexplorer/__init__.py).
#   SELF-IMPROVEMENT: also Add Input -> a PREVIOUS run's OUTPUT so the engine
#      reads world_cairn.json / learning_ledger.json (and, for the v36 advisor
#      loop, advisor_instructions.json). No code change -- it finds them itself.
# ============================================================================

CONFIG = {
    # ---- where the logic comes from -------------------------------------------
    "repo": "git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git",  # add @<tag> to pin a version
    "engine_dataset": None,        # OFFLINE: path to an attached dataset that holds the worldexplorer/ pkg
                                   #          (None = auto-find any attached folder with worldexplorer/__init__.py)
    # ---- where the data is (None = auto-detect the competition input dir) ------
    "data_root": None,             # e.g. "/kaggle/input/drw-crypto-market-prediction"
    "target": "label",             # the column to predict (None = auto-detect)
    "train": None, "test": None, "sample_submission": None,   # file names; auto if None
    "submission_target_col": None,  # default = the LAST column of sample_submission
    # ---- how to run -----------------------------------------------------------
    "metric": "auto",              # "auto" | "pearson" | "gini" | "spearman" | "rmse"
    "geometry": "auto",            # "auto" | "temporal" | "random"
    "time_budget_min": 120,        # minutes of search (or "auto"); ~120 = the v34+ default
    "out": "/kaggle/working",
    "overrides": {},               # any HarnessConfig field, e.g. {"WIDTH_BIAS_START": 0.8, "SEED": 7}
}

# ---- acquire worldexplorer: attached wheel -> pip(GitHub) -> attached source --
import glob
import importlib
import os
import re
import subprocess
import sys


def _have() -> bool:
    try:
        import worldexplorer  # noqa: F401
        return True
    except Exception:
        return False


def _candidate_roots() -> list[str]:
    roots = []
    if CONFIG.get("engine_dataset"):
        roots.append(CONFIG["engine_dataset"])
    roots += [os.path.dirname(os.path.dirname(p))
              for p in glob.glob("/kaggle/input/*/worldexplorer/__init__.py")
              + glob.glob("/kaggle/input/*/*/worldexplorer/__init__.py")]
    roots += [os.path.dirname(p)
              for p in glob.glob("/kaggle/input/*/wheelhouse")
              + glob.glob("/kaggle/input/*/*/wheelhouse")]
    out = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


def _wheel_key(path: str):
    name = os.path.basename(path)
    m = re.search(r"worldexplorer-([^-]+)-", name)
    if not m:
        return ((), name)
    parts = []
    for p in re.split(r"[._+]", m.group(1)):
        try:
            parts.append(int(p))
        except Exception:
            parts.append(p)
    return (parts, name)


def _install_attached_wheel() -> bool:
    wheels = []
    for root in _candidate_roots():
        wheels += glob.glob(os.path.join(root, "wheelhouse", "worldexplorer-*.whl"))
        wheels += glob.glob(os.path.join(root, "worldexplorer-*.whl"))
    wheels = sorted(set(wheels), key=_wheel_key)
    if not wheels:
        return False
    wheel = wheels[-1]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--force-reinstall", "--no-deps", wheel],
                       check=True, timeout=300)
        print("[bootstrap] installed attached wheel", wheel)
        importlib.invalidate_caches()
        return _have()
    except Exception as e:
        print("[bootstrap] attached wheel install failed:", e)
        return False


if not _install_attached_wheel() and not _have():  # 1) Internet ON: pip straight from GitHub
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                        "--no-cache-dir", CONFIG["repo"]],
                       check=True, timeout=900)
    except Exception as e:
        print("[bootstrap] pip-from-GitHub failed (Internet OFF?):", e)
    importlib.invalidate_caches()

if not _have():                                  # 2) Internet OFF: the repo attached as a dataset
    for r in _candidate_roots():
        if os.path.exists(os.path.join(r, "worldexplorer", "__init__.py")):
            sys.path.insert(0, r)
            print("[bootstrap] using attached worldexplorer at", r)
            break
    importlib.invalidate_caches()

if not _have():
    raise SystemExit(
        "could not acquire worldexplorer. Either turn Internet ON (Settings -> Internet), "
        "or upload the worldexplorer repo as a Kaggle Dataset, Add Input -> it, and set "
        "CONFIG['engine_dataset'] to its path.")

import worldexplorer as wx

print("worldexplorer", wx.__version__, "ready")
result = wx.kaggle.run(CONFIG)
print(result)
