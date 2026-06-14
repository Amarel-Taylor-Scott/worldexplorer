from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import telemetry_guidance as tg  # noqa: E402


def test_external_reference_scores_do_not_become_champion_or_weight_source(tmp_path):
    scores_csv = tmp_path / "scores.csv"
    scores_csv.write_text(
        "\n".join(
            [
                "filename,private,public,description",
                "external.csv,0.11164,0.10872,external reference not WorldExplorer",
                "we_best.csv,0.08619,0.06682,atlas route carve grok weird rank residual add 0.06",
                "we_neighbor.csv,0.08577,0.06599,route carve forager pow15 conflict sub 0.12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    out_dir = tmp_path / "out"

    tg.main(
        [
            "--scores-csv",
            str(scores_csv),
            "--fleet-dir",
            str(fleet_dir),
            "--out",
            str(out_dir),
            "--champion-json",
            str(tmp_path / "missing_champion.json"),
        ]
    )

    guidance = json.loads((out_dir / "next_iteration_weights.json").read_text())
    assert guidance["champion"]["private"] == 0.08619
    assert guidance["champion"]["public"] == 0.06682
    assert guidance["candidate_weights"]["external.csv"]["action"] == "external_reference_only"
    assert guidance["candidate_weights"]["external.csv"]["generalization"] == 0.0
    assert guidance["candidate_weights"]["external.csv"]["private_replay"] == 0.0
    assert guidance["candidate_weights"]["we_best.csv"]["generalization"] > 0.0
