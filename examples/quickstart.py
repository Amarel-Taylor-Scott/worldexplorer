"""Generic quickstart -- any tabular dataframe + a target column name.
    pip install git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git
"""
import worldexplorer as wx

# A) dataframe in memory -> honest holdout score (no test set needed)
import pandas as pd
df = pd.read_parquet("my_data.parquet")          # must contain the target column
result = wx.explore(df, target="y")              # auto-detects everything else
print("holdout score:", result.score)
print("detected:", result.profile.summary())
result.predictions.to_csv("predictions.csv", index=False)

# B) train + separate test (test has no target) -> predictions for the test rows
result = wx.explore("train.csv", target="y", test="test.csv", out="./out")
result.predictions.to_csv("test_predictions.csv", index=False)

# C) just inspect what it would do, no run:
print(wx.profile(df, target="y").summary())

# D) sklearn-ish:
ae = wx.AutoExplorer(time_budget=10).fit(df, target="y")
preds = ae.predict()
