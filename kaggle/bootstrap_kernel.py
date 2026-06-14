# ============================================================================
# worldexplorer -- THIN BOOTSTRAP KERNEL  (all the logic lives in a GitHub repo)
# ----------------------------------------------------------------------------
# Paste this WHOLE cell into one Kaggle code cell. It carries ONLY the config
# (where the data is, the target, the budget) + how to fetch the engine. The
# ~10k-line engine is no longer pasted -- it is downloaded from the public repo
# (Internet ON) or read from an attached dataset/wheel mirror (Internet OFF).
#
# HOW TO RUN
#   A) INTERNET ON  (Settings -> Internet -> On): default. The cell force-
#      reinstalls worldexplorer from GitHub, then runs.
#   B) INTERNET OFF (e.g. code competitions): attach the worldexplorer wheel
#      mirror/source dataset and set CONFIG["source_policy"] = "wheel_first".
#      CONFIG["engine_dataset"] can point at that dataset path, or stay None
#      if the bootstrap can auto-find it under /kaggle/input.
#   SELF-IMPROVEMENT: also Add Input -> a PREVIOUS run's OUTPUT so the engine
#      reads world_cairn.json / learning_ledger.json (and, for the v36 advisor
#      loop, advisor_instructions.json). No code change -- it finds them itself.
# ============================================================================

CONFIG = {
    # ---- where the logic comes from -------------------------------------------
    "repo": "git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git",  # add @<tag> to pin a version
    "source_policy": "github_first",  # "github_first" | "wheel_first" | "github_only" | "wheel_only"
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

# ---- acquire worldexplorer: GitHub-first online; wheel/source fallback offline -
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


def _install_github() -> bool:
    repo = CONFIG.get("repo")
    if not repo:
        return False
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--upgrade", "--force-reinstall", "--no-deps",
                        "--no-cache-dir", repo],
                       check=True, timeout=900)
        print("[bootstrap] installed worldexplorer from GitHub", repo)
        importlib.invalidate_caches()
        return _have()
    except Exception as e:
        print("[bootstrap] pip-from-GitHub failed (Internet OFF?):", e)
        return False


def _use_attached_source() -> bool:
    for r in _candidate_roots():
        if os.path.exists(os.path.join(r, "worldexplorer", "__init__.py")):
            sys.path.insert(0, r)
            print("[bootstrap] using attached worldexplorer source at", r)
            importlib.invalidate_caches()
            return _have()
    return False


policy = str(CONFIG.get("source_policy", "github_first")).lower()

if policy == "github_first":
    _install_github() or _install_attached_wheel() or _use_attached_source()
elif policy == "wheel_first":
    _install_attached_wheel() or _use_attached_source() or _install_github()
elif policy == "github_only":
    _install_github()
elif policy == "wheel_only":
    _install_attached_wheel() or _use_attached_source()
else:
    print("[bootstrap] unknown source_policy, using github_first:", policy)
    _install_github() or _install_attached_wheel() or _use_attached_source()

importlib.invalidate_caches()

if not _have():
    raise SystemExit(
        "could not acquire worldexplorer. Either turn Internet ON (Settings -> Internet), "
        "or attach the worldexplorer wheel/source dataset, set CONFIG['source_policy'] "
        "to 'wheel_first', and set CONFIG['engine_dataset'] if auto-detection misses it.")

import worldexplorer as wx

print("worldexplorer", wx.__version__, "ready")
result = wx.kaggle.run(CONFIG)
print(result)
