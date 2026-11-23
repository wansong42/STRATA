"""TO-W6-02 FIX-01 · M5B reading invariants for future reads.

This module is additive: it never imports from or mutates ``obs_equiv_m5b``
and is not consulted by the already-read M5B artifacts.  It replaces the
single original predicate (``sum(leaf probability) > 0`` for every reading)
with four invariants and one role-free feasibility rule.

INV-1  leaf probability mass and the capped flag are mutually exclusive
INV-2  a capped unit must not emit a point estimate
INV-3  probabilities are in [0, 1], the expected leaf mass is preserved,
       resolution/unresolved mass sums to 1, and NaN/Inf is rejected
INV-4  structural validation is recomputable from the cell record alone,
       with no sha pin and no digest parameter

OD-03 unification: ``is_feasible_point_estimate`` reads only the reading row.
Group role is deliberately not an input, so a CONTROL_GROUP and a
CHANGE_GROUP row with identical fields receive the same decision.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Mapping, Sequence

CELL_SCHEMA = "obs_equiv.m5b_impl.cell.v1"
SPLIT_CONFIRMATORY = "confirmatory"
ROLE_CHANGE = "CHANGE_GROUP"
ROLE_CONTROL = "CONTROL_GROUP"
ROLE_NO_INFORMATIVE = "NO_INFORMATIVE_ACTION"
CONTROL_IDS = (
    "RANDOM",
    "GEOMETRIC_FROZEN",
    "SPACE_FILL",
    "QUERY_AGNOSTIC_IG",
    "QUERY_BALD_LIKE",
    "EC2_LIKE",
    "GREEDY_RISK_ONE_STEP",
    "FULL_TREE_OPT",
)
BUDGET_GRID = (1, 2, 3, 4)
B_CAP = 4
STOCHASTIC_REPLICATES = 32
REQUIRED_READING_FIELDS = (
    "strategy",
    "stochastic",
    "replicates",
    "seeds",
    "resolution_mass",
    "unresolved_mass",
    "censored_expected_cost",
    "resolved_only_mean_cost",
    "risk_at_k",
    "bayes_at_k",
    "undecided_mass_at_k",
    "root_risk",
    "root_bayes",
    "leaves",
)
POINT_ESTIMATE_FIELDS = (
    "cost_gap",
    "cost_gap_mean",
    "opt_gap",
    "risk_gap",
    "risk_gap_mean_at_B",
    "difference",
    "gain",
)
INDEPENDENT_RISK_SOURCES = ("OPT_FIXED_BUDGET_INDEPENDENT",)


def _parse_fraction(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty string")
        return Fraction(text)
    raise ValueError(f"unsupported numeric type: {type(value).__name__}")


def _capped(row: Mapping) -> bool:
    return bool(row.get("capped_cost_not_computed", False))


def _leaves(row: Mapping):
    leaves = row.get("leaves")
    return leaves if isinstance(leaves, list) else None


def _dedupe(violations: Sequence[str]) -> list:
    return list(dict.fromkeys(violations))


def is_feasible_point_estimate(row: Mapping) -> bool:
    """Role-free feasibility rule; group role is not read here by design."""
    leaves = _leaves(row)
    return bool(leaves) and not _capped(row)


def leaf_probability_mass(row: Mapping) -> Fraction:
    leaves = _leaves(row)
    if leaves is None:
        raise ValueError("missing leaves")
    return sum((_parse_fraction(leaf["probability"]) for leaf in leaves), Fraction(0))


def expected_leaf_mass(row: Mapping) -> Fraction:
    if row.get("stochastic"):
        return Fraction(row.get("replicates", STOCHASTIC_REPLICATES))
    return Fraction(1)


def check_inv1_mass_capped_exclusive(row: Mapping) -> list:
    violations = []
    leaves = _leaves(row)
    if leaves is None:
        return ["inv1.missing_leaves"]
    capped = _capped(row)
    try:
        mass = leaf_probability_mass(row)
    except (KeyError, TypeError, ValueError):
        violations.append("inv1.unparsable_leaf_probability")
        mass = None
    if capped:
        if leaves:
            violations.append("inv1.capped_with_nonempty_leaves")
        if mass not in (None, Fraction(0)):
            violations.append("inv1.capped_with_positive_leaf_mass")
    else:
        if not leaves:
            violations.append("inv1.empty_leaves_without_capped_flag")
        if mass is not None and leaves and mass <= 0:
            violations.append("inv1.nonpositive_leaf_mass_without_capped_flag")
    declared = row.get("leaf_probability_mass")
    if declared is not None:
        try:
            declared_mass = _parse_fraction(declared)
        except (TypeError, ValueError):
            violations.append("inv1.unparsable_declared_leaf_mass")
        else:
            if not leaves and declared_mass != 0:
                violations.append("inv1.declared_mass_without_leaves")
            if leaves and mass is not None and declared_mass != mass:
                violations.append("inv1.declared_mass_mismatch")
    return _dedupe(violations)


def _has_non_null(value) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(item is not None for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(item is not None for item in value)
    return True


def check_inv2_capped_no_point_estimate(
    row: Mapping, comparison: Mapping | None = None
) -> list:
    if not _capped(row):
        return []
    violations = []
    for field in POINT_ESTIMATE_FIELDS:
        if field in row and _has_non_null(row[field]):
            violations.append(f"inv2.point_estimate_field:{field}")
    censored = row.get("censored_expected_cost")
    if censored is not None:
        try:
            value = _parse_fraction(censored)
        except (TypeError, ValueError):
            violations.append("inv2.unparsable_censored_cost")
        else:
            if value not in (Fraction(0), Fraction(B_CAP)):
                violations.append("inv2.censored_cost_point_estimate")
    resolved_mean = row.get("resolved_only_mean_cost")
    if resolved_mean is not None:
        try:
            value = _parse_fraction(resolved_mean)
        except (TypeError, ValueError):
            violations.append("inv2.unparsable_resolved_mean")
        else:
            if value != 0:
                violations.append("inv2.resolved_mean_point_estimate")
    independent_risk = row.get("risk_source") in INDEPENDENT_RISK_SOURCES
    if not independent_risk:
        for field in ("risk_at_k", "bayes_at_k", "undecided_mass_at_k"):
            sequence = row.get(field)
            if sequence is None:
                continue
            for index, item in enumerate(sequence):
                if item is None:
                    continue
                try:
                    value = _parse_fraction(item)
                except (TypeError, ValueError):
                    violations.append(f"inv2.unparsable_numeric:{field}:{index}")
                else:
                    if value != 0:
                        violations.append(f"inv2.nonzero_point_estimate:{field}")
    if comparison is not None:
        for field in POINT_ESTIMATE_FIELDS:
            if field in comparison and _has_non_null(comparison[field]):
                violations.append(f"inv2.comparison_point_estimate:{field}")
    return _dedupe(violations)


def check_inv3_probability_domain(row: Mapping) -> list:
    violations = []
    leaves = _leaves(row)
    if leaves is None:
        return ["inv3.missing_leaves"]
    if leaves:
        probabilities = []
        for index, leaf in enumerate(leaves):
            if not isinstance(leaf, Mapping) or "probability" not in leaf:
                violations.append(f"inv3.missing_leaf_probability:{index}")
                continue
            try:
                probability = _parse_fraction(leaf["probability"])
            except (TypeError, ValueError):
                violations.append(f"inv3.unparsable_leaf_probability:{index}")
                continue
            if not (Fraction(0) <= probability <= Fraction(1)):
                violations.append(f"inv3.leaf_probability_out_of_range:{index}")
            probabilities.append(probability)
        if len(probabilities) == len(leaves):
            total = sum(probabilities, Fraction(0))
            expected = expected_leaf_mass(row)
            if total != expected:
                violations.append(f"inv3.leaf_mass_sum:{total}!={expected}")
            if row.get("stochastic"):
                for index, probability in enumerate(probabilities):
                    if probability != 1:
                        violations.append(
                            f"inv3.stochastic_leaf_probability_not_one:{index}"
                        )
    masses = {}
    for field in ("resolution_mass", "unresolved_mass"):
        value = row.get(field)
        if value is None:
            violations.append(f"inv3.missing_{field}")
            continue
        try:
            parsed = _parse_fraction(value)
        except (TypeError, ValueError):
            violations.append(f"inv3.unparsable_{field}")
            continue
        if not (Fraction(0) <= parsed <= Fraction(1)):
            violations.append(f"inv3.mass_out_of_range:{field}")
        masses[field] = parsed
    if (
        "resolution_mass" in masses
        and "unresolved_mass" in masses
        and masses["resolution_mass"] + masses["unresolved_mass"] != 1
    ):
        violations.append("inv3.resolution_plus_unresolved_not_one")
    for field in ("censored_expected_cost", "root_risk", "root_bayes"):
        value = row.get(field)
        if value is None:
            violations.append(f"inv3.missing_{field}")
            continue
        try:
            parsed = _parse_fraction(value)
        except (TypeError, ValueError):
            violations.append(f"inv3.unparsable_or_nonfinite:{field}")
            continue
        if field in ("root_risk", "root_bayes") and not (
            Fraction(0) <= parsed <= Fraction(1)
        ):
            violations.append(f"inv3.out_of_unit_interval:{field}")
    resolved_mean = row.get("resolved_only_mean_cost")
    if resolved_mean is not None:
        try:
            _parse_fraction(resolved_mean)
        except (TypeError, ValueError):
            violations.append("inv3.unparsable_or_nonfinite:resolved_only_mean_cost")
    for field in ("risk_at_k", "bayes_at_k", "undecided_mass_at_k"):
        sequence = row.get(field)
        if sequence is None:
            violations.append(f"inv3.missing_{field}")
            continue
        for index, item in enumerate(sequence):
            if item is None:
                continue
            try:
                parsed = _parse_fraction(item)
            except (TypeError, ValueError):
                violations.append(f"inv3.unparsable_or_nonfinite:{field}:{index}")
                continue
            if not (Fraction(0) <= parsed <= Fraction(1)):
                violations.append(f"inv3.out_of_unit_interval:{field}:{index}")
    return _dedupe(violations)


def validate_reading_row(row: Mapping, comparison: Mapping | None = None) -> list:
    return _dedupe(
        check_inv1_mass_capped_exclusive(row)
        + check_inv2_capped_no_point_estimate(row, comparison)
        + check_inv3_probability_domain(row)
    )


def validate_cell_record_structural(cell_record: Mapping, assignment_cell: Mapping) -> dict:
    """INV-4: full structural recomputation from the record alone.

    No digest parameter is accepted and no sha constant is consulted.
    """
    issues = []
    missing = []
    probability_issues = []
    if cell_record.get("schema") != CELL_SCHEMA:
        issues.append("schema")
    if cell_record.get("split") != SPLIT_CONFIRMATORY:
        issues.append("split")
    if cell_record.get("cell_id") != assignment_cell.get("cell_id"):
        issues.append("cell_id")
    if cell_record.get("mechanism") != assignment_cell.get("mechanism"):
        issues.append("mechanism")
    if cell_record.get("world_family") != assignment_cell.get("world_family"):
        issues.append("world_family")
    if cell_record.get("eligible") is not True:
        issues.append("eligible")
    groups = cell_record.get("groups", [])
    if len(groups) != 16:
        issues.append("group_count")
    roles = {}
    for group in groups:
        roles[group.get("role")] = roles.get(group.get("role"), 0) + 1
    if roles.get(ROLE_CHANGE, 0) + roles.get(ROLE_NO_INFORMATIVE, 0) != 8:
        issues.append("change_group_count")
    if roles.get(ROLE_CONTROL, 0) != 8:
        issues.append("control_group_count")
    if sorted(group.get("pair_id") for group in groups) != sorted(
        assignment_cell.get("group_ids", [])
    ):
        issues.append("pair_ids_match_manifest")
    for group in groups:
        readings = group.get("readings", {})
        if set(readings) != set(CONTROL_IDS):
            for absent in sorted(set(CONTROL_IDS) - set(readings)):
                missing.append(
                    {
                        "pair_id": group.get("pair_id"),
                        "strategy": absent,
                        "missing_fields": ["<reading_absent>"],
                    }
                )
            continue
        for strategy_id, reading in readings.items():
            absent_fields = sorted(set(REQUIRED_READING_FIELDS) - set(reading))
            if absent_fields:
                missing.append(
                    {
                        "pair_id": group.get("pair_id"),
                        "strategy": strategy_id,
                        "missing_fields": absent_fields,
                    }
                )
                continue
            if reading["strategy"] != strategy_id:
                issues.append(f"strategy_label:{group.get('pair_id')}:{strategy_id}")
            if len(reading["risk_at_k"]) != B_CAP + 1:
                issues.append(f"risk_len:{group.get('pair_id')}:{strategy_id}")
            if len(reading["bayes_at_k"]) != B_CAP + 1:
                issues.append(f"bayes_len:{group.get('pair_id')}:{strategy_id}")
            if len(reading["undecided_mass_at_k"]) != B_CAP:
                issues.append(f"undecided_len:{group.get('pair_id')}:{strategy_id}")
            if reading["stochastic"]:
                if strategy_id != "RANDOM":
                    issues.append(f"stochastic_flag:{group.get('pair_id')}:{strategy_id}")
                if reading.get("replicates") != STOCHASTIC_REPLICATES:
                    issues.append(f"replicates:{group.get('pair_id')}:{strategy_id}")
                if (
                    len(reading["seeds"]) != STOCHASTIC_REPLICATES
                    or len(reading["leaves"]) != STOCHASTIC_REPLICATES
                ):
                    issues.append(f"random_replicates:{group.get('pair_id')}")
            else:
                if reading["seeds"]:
                    issues.append(
                        f"deterministic_seeds:{group.get('pair_id')}:{strategy_id}"
                    )
                if reading.get("replicates") != 1:
                    issues.append(f"replicates:{group.get('pair_id')}:{strategy_id}")
            for violation in validate_reading_row(reading):
                probability_issues.append(
                    {
                        "pair_id": group.get("pair_id"),
                        "strategy": strategy_id,
                        "violation": violation,
                    }
                )
    return {
        "cell_id": cell_record.get("cell_id"),
        "issues": issues,
        "roles": roles,
        "missing_readings": missing,
        "probability_issues": probability_issues,
        "all_pass": not issues and not missing and not probability_issues,
    }
