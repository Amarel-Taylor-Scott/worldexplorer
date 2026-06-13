#!/usr/bin/env python3
"""v36 ADVISOR LOOP -- the OUT-OF-BAND bridge between a run's findings graph and
the next run's exploration priors. Run this on YOUR machine (it needs internet /
an LLM; Kaggle runs Internet-OFF, which is why this is out-of-band):

    1. Download a finished run's output -> it contains explorer_findings_graph.json
    2. python tools/advisor_stub.py <run_output_dir> [--out advisor_instructions.json]
    3. Attach the produced advisor_instructions.json as an INPUT to the next
       Kaggle run. The engine reads it (ADVISOR_INGEST) and turns it into
       additive bandit priors + warm genomes, all re-measured through the
       honest doors.

THE SACRED RULE (enforced by the engine, restated here so any advisor honors it):
the advisor shapes WHERE/HOW to explore ONLY -- families, transforms, skills,
feature communities, warm-genome hypotheses. It must NOT use labels, the
leaderboard, or hidden targets; its output keys are exactly the ones below and
the engine ignores anything else. The advisor never ships a model.

This file ships a DETERMINISTIC heuristic advisor (no LLM, no internet) so the
loop is testable end-to-end today. To use a real LLM, replace `heuristic_advice`
with a call that sends findings["measured_laws"], findings["feature_topology"],
findings["interesting_regions"], findings["research_nodes"] and
findings["open_questions"] to your model with findings["advice_schema"] as the
required response format, then validate the response with `sanitize`.
"""
import argparse
import json
from pathlib import Path

ALLOWED = ("family_priors", "transform_priors", "skill_priors", "warm_genomes", "notes")


def heuristic_advice(findings: dict) -> dict:
    """A transparent, label-free baseline advisor: lean exploration toward the
    feature communities with high directional CONSENSUS and LOW train->test
    shift (the meaning most likely to survive the measured out-of-support test),
    toward agreement/stability families, and away from the high-shift, lone-
    correlate regions. Replace with an LLM call for richer guidance."""
    fam = {"consensus": 1.0, "sign_stability": 0.6, "invariant": 0.6, "stabsel": 0.4,
           "pls_weight": 0.4, "testlike_stable": 0.8, "tail": 0.3,
           "shadow": -0.4, "periphery": -0.3, "echo": -0.2}
    laws = findings.get("measured_laws", {})
    wdc = laws.get("width_decay_corr")
    skill = {"linear_ols": 0.4, "greedy_ols": 0.4, "pls": 0.4, "bayes_ridge": 0.3,
             "majority_vote": 0.3}
    if wdc is not None and wdc > 0:        # wide paths decayed more -> nudge toward sharp/simple
        skill["single_factor"] = 0.4
        skill["theil_sen"] = 0.3
    notes = ("Lean toward high-consensus / low-shift feature communities (signal that should survive "
             "the feature-disjoint test); keep the wide/narrow mix but, given width_decay_corr"
             f"={wdc}, do not over-reward in-sample-robust width. Label-free; hypotheses only.")
    return {"family_priors": fam, "skill_priors": skill,
            "warm_genomes": ["majority_vote|consensus24_identity", "linear_ols|consensus50_identity",
                             "pls|consensus64_identity"],
            "notes": notes}


def sanitize(advice: dict) -> dict:
    """Keep only the allowed (sacred-rule-safe) keys and coerce types. Any
    label/LB-flavored field an LLM might invent is dropped here AND by the
    engine (defense in depth)."""
    out = {}
    for k in ("family_priors", "transform_priors", "skill_priors"):
        v = advice.get(k)
        if isinstance(v, dict):
            out[k] = {str(kk): float(max(-1.0, min(1.0, float(vv)))) for kk, vv in v.items()}
    wg = advice.get("warm_genomes")
    if isinstance(wg, list):
        out["warm_genomes"] = [str(g) for g in wg][:16]
    if advice.get("notes"):
        out["notes"] = str(advice["notes"])[:2000]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a finished run's output dir (holds explorer_findings_graph.json)")
    ap.add_argument("--out", default="advisor_instructions.json")
    args = ap.parse_args()
    fp = Path(args.run_dir) / "explorer_findings_graph.json"
    if not fp.exists():
        raise SystemExit(f"no explorer_findings_graph.json under {args.run_dir} "
                         "(run a v36+ kernel first, with EXPORT_FINDINGS on)")
    findings = json.loads(fp.read_text())
    advice = sanitize(heuristic_advice(findings))   # <-- swap heuristic_advice for your LLM call
    Path(args.out).write_text(json.dumps(advice, indent=2) + "\n")
    print(f"wrote {args.out}: "
          + ", ".join(f"{k}={len(v) if isinstance(v, (list, dict)) else 1}" for k, v in advice.items()))
    print("attach this file as an input to the next Kaggle run (the engine ingests it as priors).")


if __name__ == "__main__":
    main()
