from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import memory_matrices as mm  # noqa: E402


def test_evidence_gate_rejects_high_stress_false_agreement_branch():
    impact_rows = [
        {
            "impact_id": "impact_bad",
            "surface_id": "surface_bad",
            "source_kind": "operation:route_carve",
            "source_id": "op_bad",
            "coordinate": "weird_pocket",
            "operator_move": "carve_signal_then_retest_global_surface",
            "local_effect": 0.91,
            "global_effect": 0.18,
            "false_agreement_risk": 0.72,
            "false_disagreement_risk": 0.10,
            "overfit_risk": 0.67,
            "foundation_stress_delta": 0.66,
            "branch_priority": 0.80,
        }
    ]
    operation_rows = [
        {
            "operation_id": "op_bad",
            "operation_type": "route_carve",
            "rehab_stage": "residual_add",
            "lens": "rank",
            "external_private": 0.086,
            "external_public": 0.066,
        }
    ]

    gates = mm.evidence_gate_matrix(impact_rows, operation_rows, [], [])

    assert gates[0]["promotion_allowed"] == 0
    assert gates[0]["branch_allowed"] == 0
    assert gates[0]["decision"] == "reject_but_remember_as_hazard"


def test_evidence_gate_allows_supported_candidate_only_as_retest_candidate():
    impact_rows = [
        {
            "impact_id": "impact_good",
            "surface_id": "surface_good",
            "source_kind": "operation:route_carve",
            "source_id": "op_good",
            "coordinate": "stable_pocket",
            "operator_move": "residual_add",
            "local_effect": 0.86,
            "global_effect": 0.78,
            "false_agreement_risk": 0.04,
            "false_disagreement_risk": 0.12,
            "overfit_risk": 0.05,
            "foundation_stress_delta": 0.06,
            "branch_priority": 0.72,
        }
    ]
    operation_rows = [
        {
            "operation_id": "op_good",
            "operation_type": "route_carve",
            "rehab_stage": "residual_add",
            "lens": "rank",
            "external_private": 0.086,
            "external_public": 0.067,
        }
    ]
    route_strength_rows = [
        {
            "route_strength_id": "rs_good",
            "operator_type": "residual_add:rank",
            "branch_value": 0.80,
        }
    ]

    gates = mm.evidence_gate_matrix(impact_rows, operation_rows, route_strength_rows, [])

    assert gates[0]["evidence_grade"] == "A_supported_candidate"
    assert gates[0]["promotion_allowed"] == 1
    assert gates[0]["decision"] == "candidate_retest_for_main_world"
    assert "shipping court" in gates[0]["required_next_evidence"]


def test_validation_budget_keeps_external_reference_out_of_selection_pressure():
    score_rows = [
        {
            "filename": "external.csv",
            "private": 0.11164,
            "public": 0.10872,
            "description": "external reference not WorldExplorer",
        },
        {
            "filename": "we.csv",
            "private": 0.08619,
            "public": 0.06682,
            "description": "atlas route carve grok weird rank residual add 0.06",
        },
    ]

    rows = mm.validation_budget_ledger([], [], [], score_rows)
    by_world = {row["validation_world"]: row for row in rows}

    assert by_world["external_reference_only"]["selection_use_count"] == 0
    assert by_world["external_reference_only"]["policy"] == "reference_only_never_weight"
    assert by_world["external_private_score_context"]["candidate_count_seen"] == 1


def test_indirect_route_support_cannot_promote_without_direct_observation():
    impact_rows = [
        {
            "impact_id": "impact_indirect",
            "surface_id": "surface_indirect",
            "source_kind": "operation:route_carve",
            "source_id": "op_indirect",
            "coordinate": "unsubmitted_variant",
            "operator_move": "residual_add",
            "local_effect": 0.90,
            "global_effect": 0.82,
            "false_agreement_risk": 0.03,
            "false_disagreement_risk": 0.08,
            "overfit_risk": 0.04,
            "foundation_stress_delta": 0.05,
            "branch_priority": 0.80,
        }
    ]
    operation_rows = [
        {
            "operation_id": "op_indirect",
            "operation_type": "route_carve",
            "rehab_stage": "residual_add",
            "lens": "rank",
        }
    ]
    route_strength_rows = [
        {"route_strength_id": f"rs_{i}", "operator_type": "residual_add:rank", "branch_value": 0.8}
        for i in range(5)
    ]

    gates = mm.evidence_gate_matrix(impact_rows, operation_rows, route_strength_rows, [])

    assert gates[0]["evidence_grade"] == "B_branch_with_retest"
    assert gates[0]["promotion_allowed"] == 0
    assert gates[0]["branch_allowed"] == 1
