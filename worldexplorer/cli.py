"""worldexplorer CLI:  worldexplorer train.parquet --target label [--test test.parquet] [--out ./out]"""
import argparse, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worldexplorer",
                                 description="Zero-config tabular ML. Point it at data + target; it figures out the rest.")
    ap.add_argument("data", help="training file (parquet/csv) containing the target column")
    ap.add_argument("--target", default=None, help="target column (auto-detected if obvious)")
    ap.add_argument("--test", default=None, help="optional test file (no target)")
    ap.add_argument("--out", default=None, help="output dir for predictions + reports")
    ap.add_argument("--time-budget", default="auto", help="minutes (or 'auto')")
    ap.add_argument("--profile-only", action="store_true", help="just print the auto-detected profile")
    a = ap.parse_args(argv)
    from . import explore
    r = explore(a.data, target=a.target, test=a.test, out=a.out, time_budget=a.time_budget,
                profile_only=a.profile_only)
    if a.profile_only:
        print(r.summary()); return 0
    print(r)
    if a.out:
        r.predictions.to_csv(f"{a.out}/predictions.csv", index=False)
        print(f"predictions -> {a.out}/predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
