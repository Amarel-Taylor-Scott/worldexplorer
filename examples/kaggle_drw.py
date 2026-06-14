# ============================================================================
# THIN KAGGLE FRONT END (DRW) -- the entire notebook is this cell.
# The engine is downloaded from GitHub (Internet ON) or read from an attached
# dataset (Internet OFF); no 10k-line paste. See kaggle/bootstrap_kernel.py for
# the fully general, robust version with the Internet-OFF fallback baked in.
# ============================================================================
import subprocess, sys

# 1. acquire worldexplorer (Internet ON). For Internet-OFF competitions, attach
#    the repo as a Kaggle Dataset and use kaggle/bootstrap_kernel.py instead.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git"], check=True)

import worldexplorer as wx

# 2. config-driven run: it auto-detects metric=pearson, time geometry, ids,
#    budget, etc., runs the whole v1-v36 engine, and writes submission.csv mapped
#    onto the competition's sample_submission.
result = wx.kaggle.run({
    "data_root": "/kaggle/input/drw-crypto-market-prediction",
    "target": "label",
    "time_budget_min": 120,          # ~2 h search; or "auto", or 690 for the full 11.5 h
    "out": "/kaggle/working",
})
print(result)

# SELF-IMPROVEMENT + ADVISOR LOOP: attach THIS notebook's OUTPUT as an input to
# the next run -- worldexplorer finds world_cairn.json / learning_ledger.json
# (governor warm-start, survivors, the self-tuning width's measured evidence)
# and advisor_instructions.json. Each run also writes explorer_findings_graph.json;
# feed it to an LLM out-of-band (tools/advisor_stub.py) for the next run's advice.
