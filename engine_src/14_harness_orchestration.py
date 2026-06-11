# ----------------------------------------------------------------------------
# 11. Harness orchestration
# ----------------------------------------------------------------------------

def write_submission(pred: np.ndarray, root: Path | None) -> None:
    if root is not None and (root / "sample_submission.csv").exists():
        sample = pd.read_csv(root / "sample_submission.csv")
        tgt = "prediction" if "prediction" in sample.columns else sample.columns[-1]
        if len(pred) != len(sample):
            raise ValueError(f"prediction rows {len(pred)} != sample rows {len(sample)}")
        sample[tgt] = pred.astype(np.float32)
        if not np.isfinite(sample[tgt].to_numpy(np.float64)).all():
            raise ValueError("non-finite predictions in submission")
        sample.to_csv(OUT / "submission.csv", index=False)
        write_json({"rows": int(len(sample)), "target_col": tgt,
                    "pred_mean": float(np.mean(pred)), "pred_std": float(np.std(pred))},
                   "submission_contract.json")
        log("submission_written", rows=len(sample), pred_std=round(float(np.std(pred)), 5))
    else:
        pd.DataFrame({"ID": np.arange(1, len(pred) + 1), "prediction": pred.astype(np.float32)}) \
            .to_csv(OUT / "explorer_predictions.csv", index=False)
        log("predictions_written", rows=len(pred), note="no sample_submission; wrote explorer_predictions.csv")


