from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fleet  # noqa: E402


def test_revival_members_retest_old_material_and_quarantine_mlp_branches():
    members = fleet.revival_members(
        "revive-test",
        count=6,
        seed=20260615,
        gpu_frac=0.5,
        time_budget=210.0,
    )

    assert len(members) == 6
    neural = [m for m in members if str(m["revival_mode"]).startswith("mlp")]
    old = [m for m in members if not str(m["revival_mode"]).startswith("mlp")]

    assert neural
    assert old
    assert any("mlp_assoc|" in "|".join(m["ov"]["WARM_GENOMES"]) for m in neural)
    assert all(m["ov"]["GROK_INCUBATION"] is True for m in neural)
    assert all(m["ov"]["GROK_SHIP_ELIGIBLE"] is False for m in neural)
    assert all("branch_only" in m["trust_policy"] for m in neural)
    assert all(m["ov"]["REVIVAL_REQUIRE_CURRENT_EVIDENCE"] is True for m in members)
    assert all(m["ov"]["ROBUST_OOS_SELECT"] is True for m in members)
    assert all(m["ov"]["SHIPPING_COURT"] is True for m in members)


def test_grok_members_seed_mlp_warm_genomes_without_shipping_eligibility():
    members = fleet.grok_members("grok-test", count=3, seed=7, gpu_frac=0.0)

    assert members
    for member in members:
        ov = member["ov"]
        assert ov["GROK_INCUBATION"] is True
        assert ov["GROK_SHIP_ELIGIBLE"] is False
        assert any(genome.startswith("mlp_assoc|") for genome in ov["WARM_GENOMES"])


def test_bootstrap_config_preserves_grokking_overrides_for_slim_kernel():
    template = fleet.DEFAULT_BOOTSTRAP.read_text(encoding="utf-8")
    text = fleet._inject_bootstrap_config(
        template,
        {"repo": "git+https://github.com/example/worldexplorer.git", "source_policy": "github_first"},
        {"GROK_INCUBATION": True, "GROK_SHIP_ELIGIBLE": False, "GROK_BRANCH_MODE": "mlp-dropout-grok"},
    )

    assert '"GROK_INCUBATION": true' in text
    assert '"GROK_SHIP_ELIGIBLE": false' in text
    assert "grokking_incubation_report.json" in text
