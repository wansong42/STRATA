from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fractureflow.obs_equiv_benchmark import (
    ALLOWED_CLAIM,
    COLLAPSE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    LOG_EPSILON,
    METHOD_CLAIMS,
    dataset_fingerprint,
    file_sha256,
    load_dataset,
    query_distance_prior_parameters,
    query_distance_scaled_prior_prediction,
    read_json,
    read_jsonl,
    zero_training_geometry_prediction,
)
from fractureflow.obs_equiv_manifest import (
    ContractError,
    canonical_json,
    finite_number,
    normalized_weights,
    sha256_json,
)
from fractureflow.observation_equivalence import CONTROL_MECHANISM

SCHEMA_REPORT = "obs_equiv.holdout.report.v1"
SCHEMA_GROUP = "obs_equiv.holdout.group.v1"
SCHEMA_FAMILY = "obs_equiv.holdout.family.v1"
SCHEMA_PARTITION = "obs_equiv.holdout.partition.v1"
SCHEMA_ARTIFACTS = "obs_equiv.holdout.artifacts.v1"
SCHEMA_VERDICT = "obs_equiv.holdout.verdict.v1"
SCHEMA_GROUPS = "obs_equiv.holdout.groups.v1"
SCHEMA_FAMILIES = "obs_equiv.holdout.families.v1"
SCHEMA_LEAVE_ONE_OUT = "obs_equiv.holdout.leave_one_out.v1"
SCHEMA_SUMMARY = "obs_equiv.holdout.summary.v1"
SCHEMA_RUN_META = "obs_equiv.holdout.run_meta.v1"
SCHEMA_RECEIPT = "obs_equiv.holdout.receipt.v1"

SCOPES = ("query_change", "query_invariant_control")
SCOPE_ORDER = {scope: index for index, scope in enumerate(SCOPES)}
METHOD_ORDER = tuple(METHOD_CLAIMS)
BASELINE_METHODS = tuple(name for name in METHOD_ORDER if METHOD_CLAIMS[name] == "BASELINE")
DIAGNOSTIC_METHODS = tuple(name for name in METHOD_ORDER if METHOD_CLAIMS[name] == "DIAGNOSTIC_ONLY")
PARTITIONS = ("mechanism", "world_family")
ABSOLUTE_TOLERANCE = 1e-12
SELECTION_MODE = "NONE_FIXED_BASELINES"
VERDICT_CONSISTENT = "HOLDOUT_DIAGNOSTIC_CONSISTENT"
VERDICT_INCONSISTENT = "HOLDOUT_DIAGNOSTIC_INCONSISTENT"
VERDICT_BLOCKED = "BLOCKED_BY_DESIGN"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scope(evaluator: Mapping[str, Any]) -> str:
    return "query_change" if bool(evaluator["expected_query_change"]) else "query_invariant_control"


def _close(first: float, second: float) -> bool:
    return abs(float(first) - float(second)) <= ABSOLUTE_TOLERANCE


def _query_separation(public: Mapping[str, Any]) -> float:
    params = public["observation"]["query"]["params"]
    point_a = params["A"]
    point_b = params["B"]
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))


def _visible_object_count(public: Mapping[str, Any]) -> int:
    return len(public["observation"]["observed_objects"])


def _case_signature(evaluator: Mapping[str, Any]) -> str:
    """同 case 的 change 行与配对 control 行共享 scenes 与 visible 对象。

    签名只用于折内配对完整性检查，不进入任何预测输入。
    """
    return sha256_json({
        "scenes": evaluator["scenes"],
        "visible_object_ids": list(evaluator["visible_object_ids"]),
    })


def _case_mechanism(
    evaluator: Mapping[str, Any],
    family_mechanisms: Mapping[str, Sequence[str]],
) -> str | None:
    mechanism = evaluator.get("mechanism")
    if isinstance(mechanism, str) and mechanism and mechanism != CONTROL_MECHANISM:
        return mechanism
    mechanisms = family_mechanisms.get(evaluator.get("world_family"), [])
    if len(mechanisms) == 1:
        return mechanisms[0]
    return None


def _case_pair_index(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for public, evaluator in records:
        index[_case_signature(evaluator)].append((public["pair_id"], _scope(evaluator)))
    return dict(index)


def mechanism_family_lookup(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for _, evaluator in records:
        if evaluator["expected_query_change"]:
            mapping[evaluator["mechanism"]].add(evaluator["world_family"])
    if not mapping:
        raise SystemExit("empty sample set: no query-change mechanism")
    return {name: sorted(families) for name, families in sorted(mapping.items())}


def family_mechanism_lookup(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, list[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for _, evaluator in records:
        if evaluator["expected_query_change"]:
            reverse[evaluator["world_family"]].add(evaluator["mechanism"])
    if not reverse:
        raise SystemExit("empty sample set: no query-change world family")
    return {family: sorted(mechanisms) for family, mechanisms in sorted(reverse.items())}


def group_metrics(evaluator: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    method = prediction.get("method")
    if method not in METHOD_CLAIMS:
        raise ContractError(f"unknown method: {method}")
    probability = finite_number(prediction.get("p_connected"), "p_connected")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("p_connected must be in [0, 1]")
    reference = finite_number(evaluator.get("reference_posterior"), "reference_posterior")
    if not 0.0 <= reference <= 1.0:
        raise ContractError("reference_posterior must be in [0, 1]")
    labels = [bool(value) for value in evaluator["query_labels"]]
    weights = normalized_weights(evaluator["prior_weights"], len(labels))
    brier = 0.0
    log_loss = 0.0
    high_confidence_error = 0.0
    for weight, label in zip(weights, labels):
        target = 1.0 if label else 0.0
        brier += weight * (probability - target) ** 2
        log_loss -= weight * (
            target * math.log(max(probability, LOG_EPSILON))
            + (1.0 - target) * math.log(max(1.0 - probability, LOG_EPSILON))
        )
        confident = max(probability, 1.0 - probability) >= HIGH_CONFIDENCE_THRESHOLD
        if confident and (probability >= 0.5) != label:
            high_confidence_error += weight
    reference_interior = LOG_EPSILON < reference < 1.0 - LOG_EPSILON
    collapsed = (
        probability <= COLLAPSE_THRESHOLD or probability >= 1.0 - COLLAPSE_THRESHOLD
        if reference_interior
        else None
    )
    weight_sum = sum(weights)
    return {
        "method": method,
        "method_claim": METHOD_CLAIMS[method],
        "p_connected": probability,
        "reference_posterior": reference,
        "prior_weights": [float(weight) for weight in weights],
        "query_labels": labels,
        "n_labels": len(labels),
        "expected_brier": brier,
        "expected_log_loss": log_loss,
        "posterior_l1": abs(probability - reference),
        "posterior_rmse": abs(probability - reference),
        "high_confidence_error_weighted": high_confidence_error,
        "reference_interior": reference_interior,
        "collapsed": collapsed,
        "collapse_status": (
            "ESTIMATED" if reference_interior else "NOT_APPLICABLE_NO_INTERIOR_REFERENCE"
        ),
        "denominators": {
            "brier": {"value": weight_sum, "unit": "PRIOR_WEIGHT_MASS"},
            "log_loss": {"value": weight_sum, "unit": "PRIOR_WEIGHT_MASS"},
            "high_confidence_error": {"value": weight_sum, "unit": "PRIOR_WEIGHT_MASS"},
            "collapse": {
                "value": 1 if reference_interior else 0,
                "unit": "GROUPS_WITH_INTERIOR_REFERENCE",
            },
        },
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "EMPTY_NOT_APPLICABLE",
        "n_groups": 0,
        "n_scenes": 0,
        "expected_query_brier": None,
        "expected_query_log_loss": None,
        "posterior_l1": None,
        "posterior_rmse": None,
        "n_reference_interior": 0,
        "collapsed_count": 0,
        "posterior_collapse_rate": None,
        "posterior_collapse_status": "NOT_APPLICABLE_NO_INTERIOR_REFERENCE",
        "expected_high_confidence_error_rate": None,
        "denominators": {
            "brier": {"value": 0, "unit": "GROUPS"},
            "log_loss": {"value": 0, "unit": "GROUPS"},
            "collapse_rate": {"value": 0, "unit": "GROUPS_WITH_INTERIOR_REFERENCE"},
            "high_confidence_error_rate": {"value": 0, "unit": "GROUPS"},
        },
    }


def scope_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _empty_summary()
    n_groups = len(rows)
    n_interior = sum(1 for row in rows if row["reference_interior"])
    collapsed = sum(1 for row in rows if row["reference_interior"] and row["collapsed"])
    return {
        "status": "ESTIMATED",
        "n_groups": n_groups,
        "n_scenes": sum(int(row["n_labels"]) for row in rows),
        "expected_query_brier": sum(float(row["expected_brier"]) for row in rows) / n_groups,
        "expected_query_log_loss": sum(float(row["expected_log_loss"]) for row in rows) / n_groups,
        "posterior_l1": sum(float(row["posterior_l1"]) for row in rows) / n_groups,
        "posterior_rmse": math.sqrt(
            sum(float(row["posterior_rmse"]) ** 2 for row in rows) / n_groups
        ),
        "n_reference_interior": n_interior,
        "collapsed_count": collapsed,
        "posterior_collapse_rate": collapsed / n_interior if n_interior else None,
        "posterior_collapse_status": (
            "ESTIMATED" if n_interior else "NOT_APPLICABLE_NO_INTERIOR_REFERENCE"
        ),
        "expected_high_confidence_error_rate": (
            sum(float(row["high_confidence_error_weighted"]) for row in rows) / n_groups
        ),
        "denominators": {
            "brier": {"value": n_groups, "unit": "GROUPS"},
            "log_loss": {"value": n_groups, "unit": "GROUPS"},
            "posterior_l1": {"value": n_groups, "unit": "GROUPS"},
            "posterior_rmse": {"value": n_groups, "unit": "GROUPS"},
            "collapse_rate": {
                "value": n_interior,
                "unit": "GROUPS_WITH_INTERIOR_REFERENCE",
            },
            "high_confidence_error_rate": {"value": n_groups, "unit": "GROUPS"},
        },
    }


def build_group_listing(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    family_mechanisms = family_mechanism_lookup(records)
    rows = []
    for public, evaluator in records:
        pair_id = evaluator["pair_id"]
        family = evaluator["world_family"]
        if family not in family_mechanisms:
            raise ContractError(f"world family lacks a query-change mechanism: {family}")
        if evaluator["expected_query_change"] and evaluator["mechanism"] not in family_mechanisms[family]:
            raise ContractError("query-change mechanism differs from its world family mapping")
        for method in METHOD_ORDER:
            prediction = predictions.get((pair_id, method))
            if prediction is None:
                raise ContractError(f"missing prediction: {(pair_id, method)}")
            metrics = group_metrics(evaluator, prediction)
            rows.append({
                "schema": SCHEMA_GROUP,
                "scope": _scope(evaluator),
                "pair_id": pair_id,
                "world_family": family,
                "mechanism": evaluator["mechanism"],
                "family_mechanisms": list(family_mechanisms[family]),
                "query_separation": _query_separation(public),
                "visible_object_count": _visible_object_count(public),
                **metrics,
            })
    rows.sort(key=lambda row: (
        SCOPE_ORDER[row["scope"]],
        row["world_family"],
        row["mechanism"],
        row["pair_id"],
        METHOD_ORDER.index(row["method"]),
    ))
    return rows


def build_family_summaries(group_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(row["scope"], row["world_family"], row["method"])].append(row)
    summaries = []
    for (scope, family, method) in sorted(
        grouped,
        key=lambda key: (SCOPE_ORDER[key[0]], key[1], METHOD_ORDER.index(key[2])),
    ):
        rows = grouped[(scope, family, method)]
        summary = scope_summary(rows)
        family_mechanism_names = sorted({
            mechanism for row in rows for mechanism in row["family_mechanisms"]
        })
        summaries.append({
            "schema": SCHEMA_FAMILY,
            "scope": scope,
            "world_family": family,
            "family_mechanisms": family_mechanism_names,
            "method": method,
            "method_claim": METHOD_CLAIMS[method],
            **summary,
        })
    return summaries


def _pooled_summaries(group_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(row["method"], row["scope"])].append(row)
    return {
        key: scope_summary(rows)
        for key, rows in grouped.items()
    }


def _fold_pair_ids(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    family_mechanisms: Mapping[str, Sequence[str]],
    *,
    held_mechanism: str | None = None,
    held_families: set[str] | None = None,
) -> dict[str, list[str]]:
    if (held_mechanism is None) == (held_families is None):
        raise ContractError("a fold must hold out exactly one mechanism or one family")
    ids: dict[str, list[str]] = {scope: [] for scope in SCOPES}
    for public, evaluator in records:
        if held_mechanism is not None:
            case_mechanism = _case_mechanism(evaluator, family_mechanisms)
            if case_mechanism is None:
                raise ContractError(
                    "control row cannot be attributed to a mechanism; "
                    "paired controls would leak across folds"
                )
            selected = case_mechanism == held_mechanism
        else:
            selected = evaluator["world_family"] in held_families
        if selected:
            ids[_scope(evaluator)].append(public["pair_id"])
    return {scope: sorted(values) for scope, values in ids.items()}


def _method_block(
    group_rows: Sequence[Mapping[str, Any]],
    holdout_ids: Mapping[str, Sequence[str]],
    complement_ids: Mapping[str, Sequence[str]],
    method: str,
) -> dict[str, Any]:
    holdout = {}
    complement = {}
    for scope in SCOPES:
        holdout_rows = [
            row for row in group_rows
            if row["method"] == method
            and row["scope"] == scope
            and row["pair_id"] in holdout_ids[scope]
        ]
        complement_rows = [
            row for row in group_rows
            if row["method"] == method
            and row["scope"] == scope
            and row["pair_id"] in complement_ids[scope]
        ]
        holdout[scope] = scope_summary(holdout_rows)
        complement[scope] = scope_summary(complement_rows)
    holdout_brier = holdout["query_change"]["expected_query_brier"]
    complement_brier = complement["query_change"]["expected_query_brier"]
    delta_brier = (
        float(holdout_brier) - float(complement_brier)
        if holdout_brier is not None and complement_brier is not None
        else None
    )
    holdout_log_loss = holdout["query_change"]["expected_query_log_loss"]
    complement_log_loss = complement["query_change"]["expected_query_log_loss"]
    delta_log_loss = (
        float(holdout_log_loss) - float(complement_log_loss)
        if holdout_log_loss is not None and complement_log_loss is not None
        else None
    )
    if delta_brier is None:
        direction = "NOT_ESTIMABLE"
    elif _close(delta_brier, 0.0):
        direction = "EQUAL"
    elif delta_brier > 0.0:
        direction = "HOLDOUT_WORSE"
    else:
        direction = "HOLDOUT_BETTER"
    holdout_collapse = holdout["query_change"]["posterior_collapse_rate"]
    complement_collapse = complement["query_change"]["posterior_collapse_rate"]
    collapse_delta = (
        float(holdout_collapse) - float(complement_collapse)
        if holdout_collapse is not None and complement_collapse is not None
        else None
    )
    return {
        "method": method,
        "method_claim": METHOD_CLAIMS[method],
        "holdout": holdout,
        "complement": complement,
        "delta_brier_change": delta_brier,
        "delta_log_loss_change": delta_log_loss,
        "direction": direction,
        "collapse_rate_delta_change": collapse_delta,
        "excluded_from_rank": METHOD_CLAIMS[method] != "BASELINE",
        "excluded_from_direction_verdict": METHOD_CLAIMS[method] != "BASELINE",
    }


def _rank_baselines(method_blocks: Mapping[str, Mapping[str, Any]], side: str) -> list[str]:
    for method in BASELINE_METHODS:
        if method_blocks[method][side]["query_change"]["expected_query_brier"] is None:
            raise ContractError(f"cannot rank an empty {side} change summary: {method}")
    return sorted(
        BASELINE_METHODS,
        key=lambda method: (
            float(method_blocks[method][side]["query_change"]["expected_query_brier"]),
            METHOD_ORDER.index(method),
        ),
    )


def build_partition(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    group_rows: Sequence[Mapping[str, Any]],
    mechanism_families: Mapping[str, Sequence[str]],
    family_mechanisms: Mapping[str, Sequence[str]],
    partition: str,
) -> dict[str, Any]:
    if partition == "mechanism":
        fold_keys = sorted(mechanism_families)
    elif partition == "world_family":
        fold_keys = sorted(family_mechanisms)
    else:
        raise ContractError(f"unknown partition: {partition}")
    if len(fold_keys) < 2:
        raise ContractError(f"holdout requires at least two {partition} folds")
    expected_scope_ids = {
        scope: sorted(
            public["pair_id"] for public, evaluator in records
            if _scope(evaluator) == scope
        )
        for scope in SCOPES
    }
    case_index = _case_pair_index(records)
    for signature, members in case_index.items():
        changes = [pair_id for pair_id, scope in members if scope == "query_change"]
        controls = [pair_id for pair_id, scope in members if scope == "query_invariant_control"]
        if len(changes) != 1 or len(controls) != 1 or len(members) != 2:
            raise ContractError(
                f"case pairing is not exactly one change + one control: {signature}"
            )
    folds = []
    for key in fold_keys:
        held_families = (
            set(mechanism_families[key]) if partition == "mechanism" else {key}
        )
        if not held_families:
            raise ContractError(f"empty held-out family set: {key}")
        all_families = set(family_mechanisms)
        if partition == "world_family" and not (all_families - held_families):
            raise ContractError(f"holdout fold leaves no complement: {key}")
        if partition == "mechanism":
            holdout_ids = _fold_pair_ids(
                records, family_mechanisms, held_mechanism=key
            )
        else:
            holdout_ids = _fold_pair_ids(
                records, family_mechanisms, held_families={key}
            )
        complement_ids = {
            scope: sorted(set(expected_scope_ids[scope]) - set(holdout_ids[scope]))
            for scope in SCOPES
        }
        for scope in SCOPES:
            if not complement_ids[scope]:
                raise ContractError(f"holdout fold leaves an empty complement: {key}")
            if set(holdout_ids[scope]) & set(complement_ids[scope]):
                raise ContractError("holdout and complement overlap")
            if sorted(holdout_ids[scope] + complement_ids[scope]) != expected_scope_ids[scope]:
                raise ContractError("holdout partition does not cover its scope")
        holdout_all = {
            pair_id for scope in SCOPES for pair_id in holdout_ids[scope]
        }
        pairing_violations = []
        for signature, members in case_index.items():
            sides = {pair_id in holdout_all for pair_id, _ in members}
            if len(sides) != 1:
                pairing_violations.append(signature)
        if pairing_violations:
            raise ContractError(
                "paired change/control rows were split across fold boundary: "
                f"{pairing_violations[:3]}"
            )
        method_blocks = {
            method: _method_block(group_rows, holdout_ids, complement_ids, method)
            for method in METHOD_ORDER
        }
        holdout_order = _rank_baselines(method_blocks, "holdout")
        complement_order = _rank_baselines(method_blocks, "complement")
        rank_preserved = holdout_order == complement_order
        directions = [method_blocks[method]["direction"] for method in BASELINE_METHODS]
        direction_agreement = len(set(directions)) == 1
        opposite_direction = (
            "HOLDOUT_WORSE" in directions and "HOLDOUT_BETTER" in directions
        )
        direction_vacuous = all(direction == "EQUAL" for direction in directions)
        inconsistency_reasons = []
        if not rank_preserved:
            inconsistency_reasons.append("baseline_brier_rank_not_preserved")
        if opposite_direction:
            inconsistency_reasons.append("baseline_directions_opposite")
        fold = {
            "schema": SCHEMA_PARTITION,
            "partition": partition,
            "held_out_key": key,
            "held_out_families": sorted(held_families),
            "held_out_mechanism": (
                key if partition == "mechanism"
                else (family_mechanisms[key][0] if len(family_mechanisms[key]) == 1 else None)
            ),
            "selection": {
                "mode": SELECTION_MODE,
                "events": [],
                "heldout_pairs_used_in_selection": 0,
                "basis": "FIXED_DECLARED_BASELINES_ONLY",
            },
            "holdout_pair_ids": holdout_ids,
            "complement_pair_ids": complement_ids,
            "n": {
                "holdout_pairs": sum(len(values) for values in holdout_ids.values()),
                "complement_pairs": sum(len(values) for values in complement_ids.values()),
                "holdout_change_groups": len(holdout_ids["query_change"]),
                "complement_change_groups": len(complement_ids["query_change"]),
                "holdout_control_groups": len(holdout_ids["query_invariant_control"]),
                "complement_control_groups": len(complement_ids["query_invariant_control"]),
            },
            "partition_checks": {
                "disjoint": True,
                "complement_non_empty": True,
                "pair_ids_are_exhaustive": True,
                "paired_change_control_same_fold": True,
                "paired_cases_checked": len(case_index),
            },
            "methods": method_blocks,
            "baseline_rank": {
                "holdout": holdout_order,
                "complement": complement_order,
                "rank_scope": "BASELINE_METHODS_ONLY_BRIER_ASCENDING",
                "tiebreak": "METHOD_DECLARATION_ORDER",
                "rank_preserved_all_baselines": rank_preserved,
            },
            "direction_checks": {
                "directions": directions,
                "direction_agreement": direction_agreement,
                "direction_vacuous_all_equal": direction_vacuous,
                "opposite_direction": opposite_direction,
            },
            "consistent_with_complement": not inconsistency_reasons,
            "inconsistency_reasons": inconsistency_reasons,
        }
        folds.append(fold)
    return {
        "schema": SCHEMA_PARTITION,
        "partition": partition,
        "folds": folds,
        "n_folds": len(folds),
        "all_folds_consistent": all(fold["consistent_with_complement"] for fold in folds),
        "all_folds_rank_preserved": all(
            fold["baseline_rank"]["rank_preserved_all_baselines"] for fold in folds
        ),
        "all_folds_direction_vacuous": all(
            fold["direction_checks"]["direction_vacuous_all_equal"] for fold in folds
        ),
    }


def _unique_signatures(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    scope: str,
) -> list[dict[str, Any]]:
    signatures = []
    seen = set()
    for _, evaluator in records:
        if _scope(evaluator) != scope:
            continue
        signature = {
            "prior_weights": [float(weight) for weight in evaluator["prior_weights"]],
            "query_labels": [bool(label) for label in evaluator["query_labels"]],
            "reference_posterior": float(evaluator["reference_posterior"]),
        }
        key = canonical_json(signature)
        if key not in seen:
            seen.add(key)
            signatures.append(signature)
    return signatures


def build_artifact_checks(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    group_rows: Sequence[Mapping[str, Any]],
    family_summaries: Sequence[Mapping[str, Any]],
    mechanism_families: Mapping[str, Sequence[str]],
    family_mechanisms: Mapping[str, Sequence[str]],
    leave_one_out: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    one_to_one = (
        all(len(families) == 1 for families in mechanism_families.values())
        and all(len(mechanisms) == 1 for mechanisms in family_mechanisms.values())
        and len(mechanism_families) == len(family_mechanisms)
    )
    family_sets = {tuple(sorted(families)) for families in mechanism_families.values()}
    shared_family_labels = (
        len(family_sets) == 1
        and len(next(iter(family_sets))) >= 2
        and len(family_mechanisms) >= 2
        and all(
            len(mechanisms) == len(mechanism_families)
            for mechanisms in family_mechanisms.values()
        )
    )
    lomo_folds = {fold["held_out_key"]: fold for fold in leave_one_out["mechanism"]["folds"]}
    lofo_folds = {fold["held_out_key"]: fold for fold in leave_one_out["world_family"]["folds"]}
    identity_rows = []
    cross_rows = []
    if one_to_one:
        for mechanism, families in mechanism_families.items():
            family = families[0]
            lomo = lomo_folds[mechanism]
            lofo = lofo_folds[family]
            same_holdout = lomo["holdout_pair_ids"] == lofo["holdout_pair_ids"]
            same_complement = lomo["complement_pair_ids"] == lofo["complement_pair_ids"]
            identity_rows.append({
                "mechanism": mechanism,
                "world_family": family,
                "holdout_pair_ids_equal": same_holdout,
                "complement_pair_ids_equal": same_complement,
                "equal": same_holdout and same_complement,
            })
        identity_all_equal = all(row["equal"] for row in identity_rows)
        identity_reason = "ONE_TO_ONE_MAPPING_MAKES_PARTITIONS_IDENTICAL"
        identical_pairs = sum(1 for row in identity_rows if row["equal"])
        compared_pairs = len(identity_rows)
        max_overlap = 1.0 if identity_all_equal else 0.0
    else:
        max_overlap = 0.0
        for mechanism in sorted(lomo_folds):
            lomo = lomo_folds[mechanism]
            lomo_ids = {
                pair_id for values in lomo["holdout_pair_ids"].values() for pair_id in values
            }
            for family in sorted(lofo_folds):
                lofo = lofo_folds[family]
                lofo_ids = {
                    pair_id for values in lofo["holdout_pair_ids"].values() for pair_id in values
                }
                same_holdout = lomo["holdout_pair_ids"] == lofo["holdout_pair_ids"]
                same_complement = lomo["complement_pair_ids"] == lofo["complement_pair_ids"]
                overlap = len(lomo_ids & lofo_ids) / max(len(lomo_ids), len(lofo_ids), 1)
                max_overlap = max(max_overlap, overlap)
                cross_rows.append({
                    "mechanism": mechanism,
                    "world_family": family,
                    "holdout_pair_ids_equal": same_holdout,
                    "complement_pair_ids_equal": same_complement,
                    "holdout_overlap_fraction": overlap,
                    "equal": same_holdout and same_complement,
                })
        compared_pairs = len(cross_rows)
        identical_pairs = sum(1 for row in cross_rows if row["equal"])
        identity_all_equal = bool(compared_pairs) and identical_pairs == compared_pairs
        identity_reason = "MAPPING_NOT_ONE_TO_ONE"
    prediction_invariance: dict[str, dict[str, Any]] = {}
    families = sorted(family_mechanisms)
    for scope in SCOPES:
        prediction_invariance[scope] = {}
        for method in METHOD_ORDER:
            rows = [
                row for row in group_rows
                if row["scope"] == scope and row["method"] == method
            ]
            values = sorted({float(row["p_connected"]) for row in rows})
            family_means = {}
            for family in families:
                family_values = [
                    float(row["p_connected"]) for row in rows if row["world_family"] == family
                ]
                family_means[family] = sum(family_values) / len(family_values)
            prediction_invariance[scope][method] = {
                "n_groups": len(rows),
                "distinct_probability_values": values,
                "constant_across_all_groups": len(values) == 1,
                "family_mean_probabilities": family_means,
                "constant_across_family_means": len(set(family_means.values())) == 1,
            }
    prior_label_signature = {}
    for scope in SCOPES:
        signatures = _unique_signatures(records, scope)
        prior_label_signature[scope] = {
            "distinct_count": len(signatures),
            "distinct_signatures": signatures,
            "identical_across_groups": len(signatures) == 1,
            "n_groups": sum(1 for _, evaluator in records if _scope(evaluator) == scope),
        }
    between_family_metric_range: dict[str, dict[str, Any]] = {}
    for scope in SCOPES:
        between_family_metric_range[scope] = {}
        for method in METHOD_ORDER:
            brier_values = {}
            log_loss_values = {}
            for family in families:
                rows = [
                    row for row in family_summaries
                    if row["scope"] == scope
                    and row["world_family"] == family
                    and row["method"] == method
                ]
                if len(rows) != 1:
                    raise ContractError("family summary is not unique")
                brier_values[family] = float(rows[0]["expected_query_brier"])
                log_loss_values[family] = float(rows[0]["expected_query_log_loss"])
            brier_span = max(brier_values.values()) - min(brier_values.values())
            log_loss_span = max(log_loss_values.values()) - min(log_loss_values.values())
            between_family_metric_range[scope][method] = {
                "family_expected_query_brier": brier_values,
                "family_expected_query_log_loss": log_loss_values,
                "brier_max_minus_min": brier_span,
                "log_loss_max_minus_min": log_loss_span,
                "zero_variation": _close(brier_span, 0.0) and _close(log_loss_span, 0.0),
            }
    family_sensitive_baseline = {
        "method": "QUERY_DISTANCE_SCALED_PRIOR",
        "method_present": "QUERY_DISTANCE_SCALED_PRIOR" in METHOD_ORDER,
        "per_mechanism_distinct_probabilities": {},
        "per_mechanism_distinct_query_separations": {},
        "per_mechanism_distinct_counts": {},
        "min_distinct_probabilities_across_mechanisms": 0,
        "cross_family_variation_requirement_met": False,
        "selection_events": 0,
    }
    if family_sensitive_baseline["method_present"]:
        for mechanism in sorted(mechanism_families):
            rows = [
                row for row in group_rows
                if row["scope"] == "query_change"
                and row["method"] == "QUERY_DISTANCE_SCALED_PRIOR"
                and row["mechanism"] == mechanism
            ]
            probabilities = sorted({float(row["p_connected"]) for row in rows})
            distances = sorted({float(row["query_separation"]) for row in rows})
            family_sensitive_baseline["per_mechanism_distinct_probabilities"][mechanism] = probabilities
            family_sensitive_baseline["per_mechanism_distinct_query_separations"][mechanism] = distances
            family_sensitive_baseline["per_mechanism_distinct_counts"][mechanism] = len(probabilities)
        counts = list(family_sensitive_baseline["per_mechanism_distinct_counts"].values())
        family_sensitive_baseline["min_distinct_probabilities_across_mechanisms"] = (
            min(counts) if counts else 0
        )
        family_sensitive_baseline["cross_family_variation_requirement_met"] = (
            bool(counts) and all(count >= 2 for count in counts)
        )
    query_geometry_variation = {"families": [], "scope": "query_change"}
    for family in families:
        changes = [
            (public, evaluator) for public, evaluator in records
            if evaluator["world_family"] == family and evaluator["expected_query_change"]
        ]
        query_geometry_variation["families"].append({
            "world_family": family,
            "family_mechanisms": list(family_mechanisms[family]),
            "n_change_groups": len(changes),
            "change_query_separations": sorted({_query_separation(public) for public, _ in changes}),
            "visible_object_counts": sorted({_visible_object_count(public) for public, _ in changes}),
            "r_attach_values": sorted({
                float(public["observation"]["query"]["params"]["r_attach"])
                for public, _ in changes
            }),
        })
    separations = sorted({
        value
        for item in query_geometry_variation["families"]
        for value in item["change_query_separations"]
    })
    visible_counts = sorted({
        value
        for item in query_geometry_variation["families"]
        for value in item["visible_object_counts"]
    })
    query_geometry_variation["distinct_change_query_separations"] = separations
    query_geometry_variation["distinct_visible_object_counts"] = visible_counts
    query_geometry_variation["geometry_varies_across_families"] = (
        len(separations) > 1 or len(visible_counts) > 1
    )
    all_baseline_constant_change = all(
        prediction_invariance["query_change"][method]["constant_across_all_groups"]
        for method in BASELINE_METHODS
    )
    all_methods_constant_change = all(
        prediction_invariance["query_change"][method]["constant_across_all_groups"]
        for method in METHOD_ORDER
    )
    query_geometry_variation["predictions_invariant_despite_geometry"] = (
        query_geometry_variation["geometry_varies_across_families"]
        and all_baseline_constant_change
    )
    flags = []
    if one_to_one:
        flags.append("MECHANISM_FAMILY_ONE_TO_ONE")
    if identity_all_equal:
        flags.append("LOMO_LOFO_PARTITION_IDENTICAL")
    if not one_to_one and not family_sensitive_baseline["cross_family_variation_requirement_met"]:
        flags.append("FAMILY_SENSITIVE_BASELINE_DOES_NOT_VARY_ACROSS_FAMILIES")
    for scope in SCOPES:
        for method in METHOD_ORDER:
            if prediction_invariance[scope][method]["constant_across_all_groups"]:
                flags.append(f"PREDICTIONS_CONSTANT_ACROSS_ALL_GROUPS:{method}:{scope}")
        if prior_label_signature[scope]["identical_across_groups"]:
            flags.append(f"PRIOR_LABEL_SIGNATURE_IDENTICAL_ACROSS_GROUPS:{scope}")
        for method in METHOD_ORDER:
            if between_family_metric_range[scope][method]["zero_variation"]:
                flags.append(f"ZERO_BETWEEN_FAMILY_VARIATION:{method}:{scope}")
    if query_geometry_variation["predictions_invariant_despite_geometry"]:
        flags.append("GEOMETRY_VARIES_BUT_PREDICTIONS_CONSTANT:query_change")
    control_rows = [
        evaluator for _, evaluator in records if not evaluator["expected_query_change"]
    ]
    control_rows_with_legacy_label = sum(
        1 for evaluator in control_rows if evaluator.get("mechanism") == CONTROL_MECHANISM
    )
    if control_rows_with_legacy_label == 0:
        control_mechanism_mode = "CASE_MECHANISM"
    elif control_rows_with_legacy_label == len(control_rows):
        control_mechanism_mode = "LEGACY_CONTROL_LABEL"
    else:
        control_mechanism_mode = "MIXED"
    if one_to_one:
        interpretation = (
            "The pilot exposes template-internal artifacts. Mechanism and world family are "
            "one-to-one, leave-one-mechanism-out and leave-one-world-family-out hold out the "
            "same rows, and every method emits one probability per scope despite query "
            "separation and visible-object count varying by family. Cross-family "
            "consistency is therefore construction-invariant and cannot separate template "
            "memorization from transfer."
        )
    else:
        interpretation = (
            "Mechanism and world family are decoupled by the cross product: every family "
            "label appears under every mechanism, so leave-one-mechanism-out and "
            "leave-one-world-family-out hold out different row sets. The added "
            "QUERY_DISTANCE_SCALED_PRIOR baseline reads only the public query endpoint "
            "distance and is family-sensitive by construction; it is a discriminative probe, "
            "not a model or a superiority claim. Remaining constant methods are still "
            "construction-invariant and are flagged as such."
        )
    return {
        "schema": SCHEMA_ARTIFACTS,
        "mechanism_family_mapping": {
            "mechanism_to_world_families": {
                name: list(families) for name, families in mechanism_families.items()
            },
            "world_family_to_mechanisms": {
                name: list(mechanisms) for name, mechanisms in family_mechanisms.items()
            },
            "one_to_one": one_to_one,
            "shared_family_labels_across_mechanisms": shared_family_labels,
            "declared_mapping": {
                name: list(families) for name, families in mechanism_families.items()
            },
            "declared_mapping_source": "OBSERVED_DATA",
            "control_mechanism_label": CONTROL_MECHANISM,
            "control_rows_mechanism_mode": control_mechanism_mode,
        },
        "lomo_lofo_partition_identity": {
            "rows": identity_rows,
            "cross_partition_rows": cross_rows,
            "all_equal": identity_all_equal,
            "reason": identity_reason,
            "pairwise_comparisons": {
                "n_pairs_compared": compared_pairs,
                "identical_pairs": identical_pairs,
                "max_holdout_overlap_fraction": max_overlap,
            },
        },
        "prediction_invariance": prediction_invariance,
        "prior_label_signature": prior_label_signature,
        "between_family_metric_range": between_family_metric_range,
        "family_sensitive_baseline": family_sensitive_baseline,
        "query_geometry_variation": query_geometry_variation,
        "all_baseline_methods_constant_across_all_change_groups": all_baseline_constant_change,
        "all_methods_constant_across_all_change_groups": all_methods_constant_change,
        "flags": flags,
        "template_artifact_present": bool(flags),
        "interpretation": interpretation,
    }


def build_sample_counts(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    group_rows: Sequence[Mapping[str, Any]],
    family_summaries: Sequence[Mapping[str, Any]],
    leave_one_out: Mapping[str, Mapping[str, Any]],
    family_mechanisms: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    change_by_family: dict[str, int] = defaultdict(int)
    control_by_family: dict[str, int] = defaultdict(int)
    change_by_mechanism: dict[str, int] = defaultdict(int)
    change_by_cell: dict[tuple[str, str], int] = defaultdict(int)
    control_by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for _, evaluator in records:
        mechanism = _case_mechanism(evaluator, family_mechanisms)
        family = evaluator["world_family"]
        if evaluator["expected_query_change"]:
            change_by_family[family] += 1
            change_by_mechanism[evaluator["mechanism"]] += 1
            if mechanism is not None:
                change_by_cell[(mechanism, family)] += 1
        else:
            control_by_family[family] += 1
            if mechanism is not None:
                control_by_cell[(mechanism, family)] += 1
    change_counts = [change_by_family[family] for family in sorted(change_by_family)]
    cell_changes = [change_by_cell[cell] for cell in sorted(change_by_cell)]
    return {
        "pairs_total": len(records),
        "predictions_checked": len(records) * len(METHOD_ORDER),
        "group_metric_rows": len(group_rows),
        "family_summary_rows": len(family_summaries),
        "change_groups": sum(change_by_family.values()),
        "control_groups": sum(control_by_family.values()),
        "world_families": sorted(change_by_family),
        "mechanisms": sorted(change_by_mechanism),
        "change_groups_per_family": dict(sorted(change_by_family.items())),
        "control_groups_per_family": dict(sorted(control_by_family.items())),
        "change_groups_per_mechanism": dict(sorted(change_by_mechanism.items())),
        "mechanism_family_cells": sorted(f"{mechanism}|{family}" for mechanism, family in change_by_cell),
        "change_groups_per_cell": {
            f"{mechanism}|{family}": value
            for (mechanism, family), value in sorted(change_by_cell.items())
        },
        "control_groups_per_cell": {
            f"{mechanism}|{family}": value
            for (mechanism, family), value in sorted(control_by_cell.items())
        },
        "min_change_groups_per_cell": min(cell_changes) if cell_changes else 0,
        "max_change_groups_per_cell": max(cell_changes) if cell_changes else 0,
        "min_change_groups_per_family": min(change_counts) if change_counts else 0,
        "max_change_groups_per_family": max(change_counts) if change_counts else 0,
        "scenes_per_change_group": sorted({
            len(evaluator["query_labels"]) for _, evaluator in records
            if evaluator["expected_query_change"]
        }),
        "scenes_per_control_group": sorted({
            len(evaluator["query_labels"]) for _, evaluator in records
            if not evaluator["expected_query_change"]
        }),
        "leave_one_out_folds": {
            partition: leave_one_out[partition]["n_folds"] for partition in PARTITIONS
        },
        "independent_family_units": len(change_by_family),
        "independent_mechanism_units": len(change_by_mechanism),
        "mechanism_family_confounded": None,
    }


def _not_estimable_items(counts: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "item": "model_generalization",
            "reason": "NO_LEARNED_MODEL_NO_TRAINING",
        },
        {
            "item": "selection_effect",
            "reason": "FIXED_DECLARED_BASELINES_NO_SELECTION_EVENTS",
        },
        {
            "item": "mechanism_vs_world_family_separation",
            "reason": (
                "MECHANISM_FAMILY_ONE_TO_ONE_IN_THIS_PILOT"
                if counts.get("mechanism_family_confounded")
                else "CROSS_PRODUCT_DESIGN_DESCRIPTIVE_ONLY_"
                f"{counts['min_change_groups_per_cell']}_GROUPS_PER_CELL"
            ),
        },
        {
            "item": "between_family_variance_uncertainty",
            "reason": (
                f"{counts['independent_family_units']} families with "
                f"{counts['min_change_groups_per_family']} change groups each; "
                "observed span is descriptive only"
            ),
        },
        {
            "item": "standard_error_and_confidence_interval",
            "reason": "TOO_FEW_INDEPENDENT_UNITS",
        },
        {
            "item": "bootstrap_or_patch_interval",
            "reason": "FORBIDDEN_DO_NOT_MANUFACTURE_INDEPENDENCE",
        },
        {
            "item": "heldout_family_absolute_error_level",
            "reason": "SYNTHETIC_TEMPLATE_INTERNAL_ONLY",
        },
    ]


def cross_check_score(
    score_path: str | Path | None,
    dataset_sha256: str,
    pooled: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if score_path is None:
        return {"status": "NOT_PROVIDED", "fields_checked": 0}
    path = Path(score_path)
    if not path.is_file():
        return {"status": "NOT_PROVIDED", "path": str(path), "fields_checked": 0}
    score = read_json(path)
    if score.get("schema") != "obs_equiv.score.v1":
        raise ContractError("score schema is not obs_equiv.score.v1")
    if score.get("dataset_sha256") != dataset_sha256:
        raise ContractError("score dataset fingerprint differs")
    protocol = score.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("score protocol is missing")
    methods_section = score.get("methods")
    if not isinstance(methods_section, Mapping):
        raise ContractError("score methods are missing")
    if protocol.get("scopes_not_combined") is not True:
        raise ContractError("score protocol mixes scopes")
    if protocol.get("method_claims") != METHOD_CLAIMS:
        raise ContractError("score method claims differ")
    for key, expected in (
        ("log_epsilon", LOG_EPSILON),
        ("collapse_threshold", COLLAPSE_THRESHOLD),
        ("high_confidence_threshold", HIGH_CONFIDENCE_THRESHOLD),
    ):
        if protocol.get(key) is None:
            raise ContractError(f"score protocol is missing {key}")
        if not _close(float(protocol[key]), expected):
            raise ContractError(f"score {key} differs")
    fields = (
        "n_groups",
        "n_scenes",
        "n_reference_interior",
        "expected_query_brier",
        "expected_query_log_loss",
        "posterior_l1",
        "posterior_rmse",
        "expected_high_confidence_error_rate",
        "posterior_collapse_rate",
        "posterior_collapse_status",
    )
    max_abs_diff = 0.0
    fields_checked = 0
    for method in METHOD_ORDER:
        for scope in SCOPES:
            ours = pooled.get((method, scope))
            method_section = methods_section.get(method)
            if not isinstance(method_section, Mapping):
                raise ContractError(f"score is missing method: {method}")
            theirs = method_section.get(scope)
            if ours is None or theirs is None:
                raise ContractError(f"score is missing {method}/{scope}")
            for field in fields:
                fields_checked += 1
                our_value = ours.get(field)
                their_value = theirs.get(field)
                if our_value is None or their_value is None:
                    if our_value is not their_value:
                        raise ContractError(f"score field differs: {method}/{scope}/{field}")
                    continue
                if isinstance(our_value, str) or isinstance(their_value, str):
                    if our_value != their_value:
                        raise ContractError(f"score field differs: {method}/{scope}/{field}")
                    continue
                if isinstance(our_value, int) and not isinstance(our_value, bool):
                    if int(our_value) != int(their_value):
                        raise ContractError(f"score field differs: {method}/{scope}/{field}")
                    continue
                difference = abs(float(our_value) - float(their_value))
                max_abs_diff = max(max_abs_diff, difference)
                if difference > ABSOLUTE_TOLERANCE:
                    raise ContractError(f"score field differs: {method}/{scope}/{field}")
    return {
        "status": "MATCH",
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "fields_checked": fields_checked,
        "max_abs_diff": max_abs_diff,
        "scope": "ALL_METHODS_BOTH_DECLARED_SCOPES",
    }


def load_holdout_inputs(
    data_dir: str | Path,
    predictions_dir: str | Path,
) -> dict[str, Any]:
    data_root = Path(data_dir)
    prediction_root = Path(predictions_dir)
    records = load_dataset(data_root)
    data_run = read_json(data_root / "run.json")
    fingerprint = dataset_fingerprint(records)
    if data_run.get("dataset_sha256") != fingerprint:
        raise ContractError("dataset fingerprint differs from data run")
    scope = data_run.get("scope")
    if not isinstance(scope, Mapping) or scope.get("allowed_claim") != ALLOWED_CLAIM:
        raise ContractError("dataset does not declare the frozen synthetic claim")
    if scope.get("truth_levels") != ["EXACT_SYNTHETIC"]:
        raise ContractError("holdout diagnostics require EXACT_SYNTHETIC only")
    if scope.get("REAL_GENERALIZATION_CLAIM") != "NOT_SUPPORTED":
        raise ContractError("dataset must not claim real generalization")
    if scope.get("FULL_3D_REAL_TRUTH") != "NOT_SUPPORTED":
        raise ContractError("dataset must not claim full 3D real truth")
    files = data_run.get("files")
    if not isinstance(files, Mapping):
        raise ContractError("data run files must be an object")
    for name in ("public.jsonl", "evaluator.jsonl", "validation.json"):
        entry = files.get(name)
        path = data_root / name
        if not isinstance(entry, Mapping) or not path.is_file():
            raise ContractError(f"data run is missing {name}")
        if entry.get("sha256") != file_sha256(path) or entry.get("bytes") != path.stat().st_size:
            raise ContractError(f"data file hash differs: {name}")
    validation = read_json(data_root / "validation.json")
    if validation.get("pass") is not True or validation.get("dataset_sha256") != fingerprint:
        raise ContractError("validation artifact is not bound to this dataset")
    if validation.get("allowed_claim") != ALLOWED_CLAIM:
        raise ContractError("validation allowed_claim differs")
    change_count = sum(1 for _, evaluator in records if evaluator["expected_query_change"])
    control_count = len(records) - change_count
    if data_run.get("pair_count") != len(records):
        raise ContractError("dataset pair count differs from data run")
    if validation.get("pair_count") != len(records):
        raise ContractError("validation pair count differs from dataset")
    if validation.get("query_change_count") != change_count:
        raise ContractError("validation query-change count differs from dataset")
    if validation.get("query_invariant_control_count") != control_count:
        raise ContractError("validation control count differs from dataset")
    prediction_run = read_json(prediction_root / "run.json")
    if prediction_run.get("dataset_sha256") != fingerprint:
        raise ContractError("predictions are not bound to this dataset version")
    if canonical_json(prediction_run.get("scope")) != canonical_json(scope):
        raise ContractError("prediction scope differs from dataset scope")
    if prediction_run.get("methods") != list(METHOD_CLAIMS):
        raise ContractError("prediction method list differs from the frozen method set")
    if prediction_run.get("method_claims") != METHOD_CLAIMS:
        raise ContractError("prediction method claims differ")
    if prediction_run.get("prior_source") != "USER_DECLARED_NOT_ESTIMATED_FROM_TEST":
        raise ContractError("constant prior must be declared, not fitted")
    prediction_path = prediction_root / "predictions.jsonl"
    if not prediction_path.is_file():
        raise ContractError("predictions.jsonl is missing")
    if prediction_run.get("predictions_sha256") != file_sha256(prediction_path):
        raise ContractError("prediction file hash differs from prediction run")
    prediction_rows = read_jsonl(prediction_path)
    if prediction_run.get("prediction_count") != len(prediction_rows):
        raise ContractError("prediction count differs from prediction run")
    constant_prior = finite_number(prediction_run.get("constant_prior"), "constant_prior")
    if not 0.0 <= constant_prior <= 1.0:
        raise ContractError("constant prior must be in [0, 1]")
    declared_prior = finite_number(
        data_run["config"].get("constant_baseline_prior"), "constant_baseline_prior"
    )
    if not _close(constant_prior, declared_prior):
        raise ContractError("constant prior differs from declared config")
    d0, tau = query_distance_prior_parameters(data_run["config"])
    if canonical_json(prediction_run.get("query_distance_scaled_prior")) != canonical_json(
        {"d0": d0, "tau": tau}
    ):
        raise ContractError("query distance prior constants differ from declared config")
    public_by_id = {public["pair_id"]: public for public, _ in records}
    evaluator_by_id = {evaluator["pair_id"]: evaluator for _, evaluator in records}
    predictions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for prediction in prediction_rows:
        pair_id = prediction.get("pair_id")
        method = prediction.get("method")
        if pair_id not in evaluator_by_id:
            raise ContractError(f"prediction references unknown pair: {pair_id}")
        if method not in METHOD_CLAIMS:
            raise ContractError(f"unknown prediction method: {method}")
        key = (pair_id, method)
        if key in predictions:
            raise ContractError(f"duplicate prediction: {key}")
        if prediction.get("allowed_claim") != METHOD_CLAIMS[method]:
            raise ContractError(f"prediction claim differs from frozen mapping: {key}")
        if prediction.get("status") != "PREDICTED":
            raise ContractError(f"non-predicted row cannot enter metrics: {key}")
        if not _close(finite_number(prediction.get("coverage"), "coverage"), 1.0):
            raise ContractError(f"metrics require full coverage: {key}")
        probability = finite_number(prediction.get("p_connected"), "p_connected")
        if not 0.0 <= probability <= 1.0:
            raise ContractError(f"p_connected must be in [0, 1]: {key}")
        if method == "CONSTANT_PRIOR":
            expected = constant_prior
        elif method == "ZERO_TRAINING_GEOMETRY":
            expected = zero_training_geometry_prediction(public_by_id[pair_id])
        elif method == "QUERY_DISTANCE_SCALED_PRIOR":
            expected = query_distance_scaled_prior_prediction(
                public_by_id[pair_id], d0, tau
            )
        else:
            expected = float(evaluator_by_id[pair_id]["reference_posterior"])
        if not _close(probability, expected):
            raise ContractError(f"prediction value differs from recomputation: {key}")
        predictions[key] = prediction
    expected_keys = {
        (pair_id, method) for pair_id in evaluator_by_id for method in METHOD_ORDER
    }
    if set(predictions) != expected_keys:
        raise ContractError("predictions do not cover every pair and method")
    return {
        "records": records,
        "predictions": predictions,
        "data_run": data_run,
        "prediction_run": prediction_run,
        "dataset_sha256": fingerprint,
        "constant_prior": constant_prior,
        "data_dir": str(data_root.resolve()),
        "predictions_dir": str(prediction_root.resolve()),
        "predictions_sha256": file_sha256(prediction_path),
    }


def build_holdout_report(
    inputs: Mapping[str, Any],
    score_path: str | Path | None = None,
) -> dict[str, Any]:
    records = inputs["records"]
    predictions = inputs["predictions"]
    mechanism_families = mechanism_family_lookup(records)
    family_mechanisms = family_mechanism_lookup(records)
    families = sorted(family_mechanisms)
    if len(families) < 2:
        raise ContractError("holdout requires at least two world families")
    if len(mechanism_families) < 2:
        raise ContractError("holdout requires at least two mechanisms")
    group_rows = build_group_listing(records, predictions)
    family_summaries = build_family_summaries(group_rows)
    pooled = _pooled_summaries(group_rows)
    leave_one_out = {
        partition: build_partition(
            records, group_rows, mechanism_families, family_mechanisms, partition
        )
        for partition in PARTITIONS
    }
    artifacts = build_artifact_checks(
        records, group_rows, family_summaries, mechanism_families, family_mechanisms,
        leave_one_out,
    )
    counts = build_sample_counts(
        records, group_rows, family_summaries, leave_one_out, family_mechanisms
    )
    counts["mechanism_family_confounded"] = artifacts["mechanism_family_mapping"]["one_to_one"]
    counts["score_fields_checked"] = 0
    score_check = cross_check_score(score_path, inputs["dataset_sha256"], pooled)
    counts["score_fields_checked"] = score_check["fields_checked"]
    all_fold_deltas = []
    for partition in PARTITIONS:
        for fold in leave_one_out[partition]["folds"]:
            for method in BASELINE_METHODS:
                all_fold_deltas.append(fold["methods"][method]["delta_brier_change"])
    all_deltas_zero = all(
        delta is not None and _close(delta, 0.0) for delta in all_fold_deltas
    )
    any_inconsistency = any(
        not fold["consistent_with_complement"]
        for partition in PARTITIONS
        for fold in leave_one_out[partition]["folds"]
    )
    if any_inconsistency:
        direction_status = "INCONSISTENT"
    elif all_deltas_zero:
        direction_status = "CONSISTENT_EQUAL_LEVEL_TRIVIAL"
    else:
        direction_status = "CONSISTENT_NONZERO"
    direction_diagnostic = {
        "status": direction_status,
        "all_fold_rank_preserved": all(
            leave_one_out[partition]["all_folds_rank_preserved"]
            for partition in PARTITIONS
        ),
        "all_baseline_directions_agreement": all(
            fold["direction_checks"]["direction_agreement"]
            for partition in PARTITIONS
            for fold in leave_one_out[partition]["folds"]
        ),
        "all_fold_deltas_zero": all_deltas_zero,
        "reference_posterior_excluded_from_rank": True,
        "note": (
            "No model, training, or selection exists. This is a descriptive comparison "
            "of fixed baselines between a held-out family and its complement."
        ),
    }
    report = {
        "schema": SCHEMA_REPORT,
        "status": "OK",
        "created_at_utc": utc_now(),
        "allowed_claim": ALLOWED_CLAIM,
        "REAL_GENERALIZATION_CLAIM": "NOT_SUPPORTED",
        "FULL_3D_REAL_TRUTH": "NOT_SUPPORTED",
        "dataset_sha256": inputs["dataset_sha256"],
        "predictions_sha256": inputs["predictions_sha256"],
        "data_dir": inputs["data_dir"],
        "predictions_dir": inputs["predictions_dir"],
        "scope": {
            "site_ids": inputs["data_run"]["scope"]["site_ids"],
            "split_groups": inputs["data_run"]["scope"]["split_groups"],
            "world_families": inputs["data_run"]["scope"]["world_families"],
            "truth_levels": inputs["data_run"]["scope"]["truth_levels"],
        },
        "protocol": {
            "primary_scope": "query_change",
            "control_scope": "query_invariant_control",
            "scopes_not_combined": True,
            "higher_brier_is_worse": True,
            "log_epsilon": LOG_EPSILON,
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "method_claims": dict(METHOD_CLAIMS),
            "baseline_methods": list(BASELINE_METHODS),
            "diagnostic_methods": list(DIAGNOSTIC_METHODS),
            "selection": {
                "mode": SELECTION_MODE,
                "events": [],
                "heldout_groups_used_in_selection": 0,
                "constant_prior_source": inputs["prediction_run"]["prior_source"],
                "geometry_rule": "FIXED_ZERO_TRAINING_ALGORITHM",
                "reference_posterior_claim": "DIAGNOSTIC_ONLY",
            },
            "no_model": True,
            "no_training": True,
            "no_selection": True,
            "no_framework_choice": True,
            "no_bootstrap": True,
            "no_patch_augmentation": True,
        },
        "n": counts,
        "not_estimable": _not_estimable_items(counts),
        "score_check": score_check,
        "group_listing": group_rows,
        "family_summaries": family_summaries,
        "pooled_summaries": {
            f"{method}/{scope}": summary
            for (method, scope), summary in sorted(
                pooled.items(), key=lambda item: (
                    METHOD_ORDER.index(item[0][0]), SCOPE_ORDER[item[0][1]]
                )
            )
        },
        "leave_one_out": leave_one_out,
        "artifact_checks": artifacts,
        "direction_diagnostic": direction_diagnostic,
    }
    verdict, reasons = classify_verdict(report)
    if verdict != VERDICT_INCONSISTENT:
        reasons.append(
            "No learned model, training, or selection exists; this verdict concerns "
            "fixed baseline behavior only and cannot support a model-generalization claim."
        )
    report["verdict"] = verdict
    report["verdict_reasons"] = reasons
    return report


def classify_verdict(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    if report.get("status") != "OK":
        return VERDICT_BLOCKED, ["report status is not OK"]
    inconsistent = []
    for partition in PARTITIONS:
        for fold in report["leave_one_out"][partition]["folds"]:
            if not fold["baseline_rank"]["rank_preserved_all_baselines"]:
                inconsistent.append(
                    f"{partition}:{fold['held_out_key']}:baseline_brier_rank_not_preserved"
                )
            if fold["direction_checks"]["opposite_direction"]:
                inconsistent.append(
                    f"{partition}:{fold['held_out_key']}:baseline_directions_opposite"
                )
    if inconsistent:
        return VERDICT_INCONSISTENT, inconsistent
    artifacts = report["artifact_checks"]
    blocking = []
    if artifacts["mechanism_family_mapping"]["one_to_one"]:
        blocking.append(
            "mechanism and world_family are one-to-one; leave-one-mechanism-out and "
            "leave-one-world-family-out are not independent partitions"
        )
    if artifacts["lomo_lofo_partition_identity"]["all_equal"]:
        blocking.append(
            "leave-one-mechanism-out and leave-one-world-family-out hold out identical rows"
        )
    if artifacts["all_baseline_methods_constant_across_all_change_groups"]:
        blocking.append(
            "every baseline emits one probability across all query_change groups; "
            "cross-family consistency is construction-invariant"
        )
    if report["n"]["min_change_groups_per_family"] < 2:
        blocking.append(
            "fewer than two query_change groups per family; within-family replication is absent"
        )
    if blocking:
        return VERDICT_BLOCKED, blocking
    return VERDICT_CONSISTENT, [
        "no baseline rank flip or opposite-sign nonzero direction was observed across folds; "
        "the design permits a non-vacuous descriptive direction check"
    ]


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_collapse(row: Mapping[str, Any]) -> str:
    if not row["reference_interior"]:
        return "不适用"
    return "是" if row["collapsed"] else "否"


def _short_family(name: str) -> str:
    return name.removeprefix("PLANAR_DISK_").removesuffix("_V1")


def _verification_section(verification: Mapping[str, Any] | None) -> list[str]:
    if not verification:
        return [
            "本次运行未传入外部日志；pytest 与 smoke 的命令、rc、枚举数见 "
            "`implementation_receipt.json` 与同目录 `pytest.log` / `smoke.log`。",
        ]
    pytest_info = verification.get("pytest") or {}
    smoke_info = verification.get("smoke") or {}
    if pytest_info.get("status") != "PROVIDED" and smoke_info.get("status") != "PROVIDED":
        return [
            "本次运行未传入外部日志；pytest 与 smoke 的命令、rc、枚举数见 "
            "`implementation_receipt.json` 与同目录 `pytest.log` / `smoke.log`。",
        ]
    lines = ["| 项 | 命令 | rc | 枚举数 | 日志 |", "|---|---|---:|---:|---|"]
    if pytest_info.get("status") == "PROVIDED":
        lines.append(
            f"| pytest | `{pytest_info.get('command', 'N/A')}` | "
            f"{pytest_info.get('rc', 'N/A')} | "
            f"passed={pytest_info.get('passed', 'N/A')}, failed={pytest_info.get('failed', 'N/A')} | "
            f"`{pytest_info.get('log', 'N/A')}` |"
        )
    if smoke_info.get("status") == "PROVIDED":
        summary = smoke_info.get("summary") or {}
        lines.append(
            f"| smoke | `{smoke_info.get('command', 'N/A')}` | "
            f"{smoke_info.get('rc', 'N/A')} | "
            f"pairs={summary.get('n_pairs', 'N/A')}, "
            f"change={summary.get('n_change_groups', 'N/A')}, "
            f"control={summary.get('n_control_groups', 'N/A')}, "
            f"predictions={summary.get('n_predictions_checked', 'N/A')} | "
            f"`{smoke_info.get('log', 'N/A')}` |"
        )
    return lines


def render_report_md(
    report: Mapping[str, Any],
    verification: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> str:
    heading = title or "ObsEquiv 机制/世界族留出诊断报告（2026-10-24）"
    lines = [
        f"# {heading}",
        "",
        f"- **判级**：`{report['verdict']}`",
        "- **范围**：`SYNTHETIC_DECLARED_DOMAIN_ONLY` / `EXACT_SYNTHETIC`",
        f"- **dataset_sha256**：`{report['dataset_sha256']}`",
        f"- **predictions_sha256**：`{report['predictions_sha256']}`",
        f"- **score.json 交叉核验**：`{report['score_check']['status']}`"
        + (
            f"（{report['score_check'].get('fields_checked')} 字段，"
            f"max_abs_diff={report['score_check'].get('max_abs_diff')}）"
            if report["score_check"]["status"] == "MATCH" else ""
        ),
        "",
        "> 本报告是评估侧诊断：不训练模型、不选择框架、不产生模型泛化声明。",
        "> `REFERENCE_POSTERIOR_DIAGNOSTIC` 只作答案卷诊断，已排除在基线排序与判级之外。",
        "",
        "## 1. 结论摘要",
        "",
    ]
    artifacts = report["artifact_checks"]
    direction = report["direction_diagnostic"]
    if artifacts["mechanism_family_mapping"]["one_to_one"]:
        reading_note = (
            "**读法（必须保留）**：本 pilot 的机制与 `world_family` 一一对应，"
            "留一机制与留一族留出的是同一批行；基线在每个 scope 内各自只输出一个概率。"
            "因此“held-out family 与其余 family 同向”是构造恒真，"
            "不能当作经验证据，更不能写成模型泛化。"
        )
        section2_note = (
            "主组的 `mechanism` 与 `world_family` 一一对应；表内 `mechanism` 列即该主组声明的机制。"
            "控制组的原始 `mechanism` 字段为 `QUERY_INVARIANT_CONTROL`，其族机制见 JSON 的 "
            "`family_mechanisms`，下表用“族机制”列展示。"
        )
    else:
        family_sensitive = artifacts["family_sensitive_baseline"]
        counts = family_sensitive["per_mechanism_distinct_counts"]
        reading_note = (
            "**读法（必须保留）**：机制与 `world_family` 已解耦为交叉乘积；"
            "留一机制与留一族留出的是不同行集。族敏感基线 `QUERY_DISTANCE_SCALED_PRIOR` "
            "只读 public 查询端点距离，同一机制跨族不同概率值数 = "
            f"`{counts}`（最小值 `{family_sensitive['min_distinct_probabilities_across_mechanisms']}`）。"
            "它只是判别力探针：优于或劣于常数先验都不构成方法优越性、模型成绩或泛化证据。"
        )
        section2_note = (
            "主组的 `mechanism` 与 `world_family` 是交叉乘积：每个族在全部三机制下各出现一次。"
            "控制行携带其配对 case 的机制标签，因此 LOMO 折可以把配对控制与主行同折进出；"
            "控制组的族机制见 JSON 的 `family_mechanisms`。"
        )
    lines.extend([
        f"1. 主组共 `{report['n']['change_groups']}` 个，控制组共 "
        f"`{report['n']['control_groups']}` 个；每个族主组/控制组各 "
        f"`{report['n']['min_change_groups_per_family']}`/`"
        f"{min(report['n']['control_groups_per_family'].values())}` 个。",
        f"2. 留出方向诊断：`{direction['status']}`；所有折的基线 Brier 排序保持="
        f"`{direction['all_fold_rank_preserved']}`，所有基线 delta 为 0="
        f"`{direction['all_fold_deltas_zero']}`。",
        f"3. 模板内假象：`{artifacts['template_artifact_present']}`；共 "
        f"`{len(artifacts['flags'])}` 条标记。",
        f"4. 判级理由：{'；'.join(report['verdict_reasons'])}",
        "",
        reading_note,
        "",
        "## 2. 主组逐组分解（`query_change`）",
        "",
        section2_note,
        "",
        "分母说明：组级 Brier/log loss 的分母是先验权重质量和（=1）；"
        "collapse 的分母是“参考后验位于 (1e-6, 1-1e-6) 的组数”，控制组该分母为 0。"
        "`collapse_status` 取值只有 `ESTIMATED` 或 `NOT_APPLICABLE_NO_INTERIOR_REFERENCE`。",
        "",
        "| family | mechanism | pair_id | method | p | Brier | log loss | collapse | collapse_status | collapse 分母 | 参考内点 |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|---|",
    ])
    for row in report["group_listing"]:
        if row["scope"] != "query_change":
            continue
        lines.append(
            f"| {_short_family(row['world_family'])} | {row['mechanism']} | `{row['pair_id']}` | "
            f"`{row['method']}` | {row['p_connected']:.4f} | "
            f"{_format_number(row['expected_brier'])} | "
            f"{_format_number(row['expected_log_loss'])} | "
            f"{_format_collapse(row)} | {row['collapse_status']} | "
            f"{row['denominators']['collapse']['value']} | "
            f"{'是' if row['reference_interior'] else '否'} |"
        )
    lines.extend([
        "",
        "## 3. 控制组逐组分解（`query_invariant_control`，单独列，不混算）",
        "",
        "| family | 族机制 | pair_id | method | p | Brier | log loss | collapse | collapse_status | collapse 分母 | 参考内点 |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|---|",
    ])
    for row in report["group_listing"]:
        if row["scope"] != "query_invariant_control":
            continue
        lines.append(
            f"| {_short_family(row['world_family'])} | "
            f"{', '.join(row['family_mechanisms'])} | `{row['pair_id']}` | "
            f"`{row['method']}` | {row['p_connected']:.4f} | "
            f"{_format_number(row['expected_brier'])} | "
            f"{_format_number(row['expected_log_loss'])} | "
            f"{_format_collapse(row)} | {row['collapse_status']} | "
            f"{row['denominators']['collapse']['value']} | "
            f"{'是' if row['reference_interior'] else '否'} |"
        )
    lines.extend([
        "",
        "## 4. 族级汇总（主组与控制组分栏）",
        "",
        "| scope | family | method | n | Brier | log loss | collapse | collapse_status | collapse 分母 | 高置信错误率 |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|",
    ])
    for row in report["family_summaries"]:
        lines.append(
            f"| `{row['scope']}` | {_short_family(row['world_family'])} | "
            f"`{row['method']}` | {row['n_groups']} | "
            f"{_format_number(row['expected_query_brier'])} | "
            f"{_format_number(row['expected_query_log_loss'])} | "
            f"{'N/A' if row['posterior_collapse_rate'] is None else _format_number(row['posterior_collapse_rate'])} | "
            f"{row['posterior_collapse_status']} | "
            f"{row['denominators']['collapse_rate']['value']} | "
            f"{_format_number(row['expected_high_confidence_error_rate'])} |"
        )
    lines.extend([
        "",
        "## 5. 留出清单（留一机制 / 留一族，主组口径）",
        "",
        "`held-out` 组不参与任何选择；由于没有学习模型，也没有可拟合的超参数，"
        "选择事件数为 0。`REFERENCE_POSTERIOR_DIAGNOSTIC` 不参与排序。",
        "",
        "| partition | held-out | method | holdout Brier | complement Brier | delta | direction | rank 保持 |",
        "|---|---|---|---:|---:|---:|---|---|",
    ])
    for partition in PARTITIONS:
        for fold in report["leave_one_out"][partition]["folds"]:
            for method in METHOD_ORDER:
                block = fold["methods"][method]
                lines.append(
                    f"| `{partition}` | `{fold['held_out_key']}` | `{method}` | "
                    f"{_format_number(block['holdout']['query_change']['expected_query_brier'])} | "
                    f"{_format_number(block['complement']['query_change']['expected_query_brier'])} | "
                    f"{_format_number(block['delta_brier_change'])} | "
                    f"`{block['direction']}` | "
                    f"{'排除(诊断)' if block['excluded_from_rank'] else ('是' if fold['baseline_rank']['rank_preserved_all_baselines'] else '否')} |"
                )
    lines.extend([
        "",
        "## 6. 模板假象检查",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 机制 ↔ 世界族一一对应 | `{artifacts['mechanism_family_mapping']['one_to_one']}` |",
        f"| 留一机制与留一族留出同一批行 | `{artifacts['lomo_lofo_partition_identity']['all_equal']}` |",
        f"| 所有基线在 {report['n']['change_groups']} 个主组上只输出一个概率 | `{artifacts['all_baseline_methods_constant_across_all_change_groups']}` |",
        f"| 族敏感基线跨族出现 ≥2 个不同值 | `{artifacts['family_sensitive_baseline']['cross_family_variation_requirement_met']}` |",
        f"| 主组先验/标签/参考后验签名唯一 | `{artifacts['prior_label_signature']['query_change']['identical_across_groups']}` |",
        f"| 族间 Brier/log loss 跨度为零 | `{all(item['zero_variation'] for item in artifacts['between_family_metric_range']['query_change'].values())}` |",
        f"| 查询几何随族变化但预测不变 | `{artifacts['query_geometry_variation']['predictions_invariant_despite_geometry']}` |",
        "",
        "主组查询几何随族变化：",
        "",
        "| family | 查询端点间距 | 可见对象数 | r_attach |",
        "|---|---:|---:|---:|",
    ])
    for item in artifacts["query_geometry_variation"]["families"]:
        lines.append(
            f"| {_short_family(item['world_family'])} | "
            f"{', '.join(_format_number(value, 3) for value in item['change_query_separations'])} | "
            f"{', '.join(str(value) for value in item['visible_object_counts'])} | "
            f"{', '.join(_format_number(value, 3) for value in item['r_attach_values'])} |"
        )
    lines.extend([
        "",
        "标记清单：",
        "",
        "```text",
        *artifacts["flags"],
        "```",
        "",
        artifacts["interpretation"],
        "",
        "## 7. 样本量与不可估计项",
        "",
        f"- 总对数：`{report['n']['pairs_total']}`；主组：`{report['n']['change_groups']}`；"
        f"控制组：`{report['n']['control_groups']}`。",
        f"- 族数：`{report['n']['independent_family_units']}`；机制数："
        f"`{report['n']['independent_mechanism_units']}`；"
        f"每族主组数：`{report['n']['min_change_groups_per_family']}`。",
        f"- 逐组指标行：`{report['n']['group_metric_rows']}`；族级汇总行："
        f"`{report['n']['family_summary_rows']}`；预测检查："
        f"`{report['n']['predictions_checked']}`。",
        "",
        "| 不可估计项 | 原因 |",
        "|---|---|",
    ])
    for item in report["not_estimable"]:
        lines.append(f"| `{item['item']}` | {item['reason']} |")
    min_cell = report["n"]["min_change_groups_per_cell"]
    max_cell = report["n"]["max_change_groups_per_cell"]
    cell_phrase = f"{min_cell}" if min_cell == max_cell else f"{min_cell}–{max_cell}"
    lines.extend([
        "",
        "未使用 bootstrap、patch、增强样本或重复轨迹预算点制造独立性；"
        f"也不把同一格（机制×族）的 {report['n']['min_change_groups_per_cell']} 个主组"
        "当成独立场地。",
        "",
        "## 8. 判级规则与理由",
        "",
        "判级规则（预注册，代码实现）：",
        "",
        "1. 若任一族/机制折出现基线 Brier 排序翻转，或基线方向同时出现 "
        "`HOLDOUT_WORSE` 与 `HOLDOUT_BETTER`，判 `HOLDOUT_DIAGNOSTIC_INCONSISTENT`。",
        "2. 否则若机制与族一一对应、留一机制与留一族为同一划分、"
        "或所有基线在全部主组上只输出一个概率，判 `BLOCKED_BY_DESIGN`。",
        "3. 否则判 `HOLDOUT_DIAGNOSTIC_CONSISTENT`。",
        "",
        f"本次判级：`{report['verdict']}`。",
        "",
        "理由：",
        "",
        *[f"- {reason}" for reason in report["verdict_reasons"]],
        "",
        "## 9. 验证与复现",
        "",
        *_verification_section(verification),
        "",
        "本报告只运行定向测试与一次 smoke；四道门禁读数另件登记（见 pilot 目录的 gates/ 与 r303/），不以本报告代替。",
        "",
        "## 10. 不能说的话",
        "",
        "- 不能说模型泛化、机制外推、真实域可靠性或完整三维真值。",
        "- 不能把 `REFERENCE_POSTERIOR_DIAGNOSTIC` 当对外成绩。",
        "- 不能把零训练几何基线的 collapse 当方法缺陷之外的模型结论；它只是负结果基线。",
        (
            "- 不能把留一机制与留一族当两项独立证据；本 pilot 中它们是同一划分。"
            if artifacts["mechanism_family_mapping"]["one_to_one"]
            else f"- 不能把两个不同划分说成两项独立证据：每格只有 {cell_phrase} 个主组，且仍零训练零模型。"
        ),
        "- 不能把族敏感基线的跨族变化说成方法优越性；它只读查询端点距离，是判别力探针。",
        f"- 不能用 bootstrap/patch 给 {report['n']['independent_family_units']} 个族、"
        f"每格 {report['n']['min_change_groups_per_cell']} 个主组制造置信区间。",
        "",
        "## 11. 文件",
        "",
        "- 机器清单：`groups.json`、`families.json`、`leave_one_out.json`、"
        "`artifact_checks.json`、`summary.json`、`run_meta.json`。",
        "- 判级：`verdict.json`。",
        "- 回执：`implementation_receipt.json`。",
        "- 日志：`pytest.log`、`smoke.log`（若本次运行提供）。",
    ])
    return "\n".join(lines) + "\n"
