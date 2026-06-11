# ============================================================================
# THIN KAGGLE FRONT END -- the entire notebook is this cell.
# It installs the engine from GitHub and runs it; no 8,500-line paste.
# ============================================================================
import subprocess, sys

# 1. install worldexplorer straight from GitHub (Kaggle already has torch/lgbm/xgb)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/taylorsamarel/worldexplorer.git"], check=True)

import pandas as pd
import worldexplorer as wx

D = "/kaggle/input/drw-crypto-market-prediction"

# 2. zero-config run: it auto-detects metric=pearson, time geometry, ids, budget, etc.
result = wx.explore(
    f"{D}/train.parquet",
    target="label",
    test=f"{D}/test.parquet",
    out="/kaggle/working",
    time_budget=690,            # 11.5 h; or "auto"
)

# 3. map predictions onto the competition's sample_submission and save
sub = pd.read_csv(f"{D}/sample_submission.csv")
sub[sub.columns[-1]] = result.predictions["prediction"].to_numpy()
sub.to_csv("/kaggle/working/submission.csv", index=False)

print(result)
print("reports written to /kaggle/working:",
      "complexity_governor.json, learning_ledger.json, shipping_court_report.csv, ...")

# SELF-IMPROVEMENT: attach THIS notebook's output as an input dataset to the next
# run; worldexplorer will find world_cairn.json / learning_ledger.json and stand
# on it (governor beta warm-start, survivors, anti-priors). Run N+1 learns from run N.
