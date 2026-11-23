from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fractureflow.obs_equiv_geometry import two_point_connected
from fractureflow.obs_equiv_manifest import (
    SCHEMA_RUN,
    SCHEMA_SCORE,
    ContractError,
    Disk,
    Scene,
    TwoPointQuery,
    assert_public_safe,
    canonical_json,
    finite_number,
    sha256_json,
)
from fractureflow.observation_equivalence import (
    LEGACY_V2_MECHANISM_WORLD_FAMILIES,
    family_parameters,
    validate_pair,
)

LOG_EPSILON = 1e-6
COLLAPSE_THRESHOLD = 0.05
HIGH_CONFIDENCE_THRESHOLD = 0.9
ALLOWED_CLAIM = "SYNTHETIC_DECLARED_DOMAIN_ONLY"
METHOD_CLAIMS = {
    "CONSTANT_PRIOR": "BASELINE",
    "ZERO_TRAINING_GEOMETRY": "BASELINE",
    "QUERY_DISTANCE_SCALED_PRIOR": "BASELINE",
    "REFERENCE_POSTERIOR_DIAGNOSTIC": "DIAGNOSTIC_ONLY",
}

# R-08（v6）：prediction 行角色旗标（角色驱动主表过滤；废方法名黑名单）。
V6_METHOD_ROLES = {
    "CONSTANT_PRIOR": {
        "metric_role": "DEPLOYABLE_PREDICTOR",
        "deployable_predictor": True,
        "uses_held_out_target_for_selection": False,
    },
    "ZERO_TRAINING_GEOMETRY": {
        "metric_role": "DEPLOYABLE_PREDICTOR",
        "deployable_predictor": True,
        "uses_held_out_target_for_selection": False,
    },
    "QUERY_DISTANCE_SCALED_PRIOR": {
        "metric_role": "DEPLOYABLE_PREDICTOR",
        "deployable_predictor": True,
        "uses_held_out_target_for_selection": False,
    },
    "REFERENCE_POSTERIOR_DIAGNOSTIC": {
        "metric_role": "ORACLE_DIAGNOSTIC",
        "deployable_predictor": False,
        "uses_held_out_target_for_selection": True,
    },
}


def _is_v6_config(config: Mapping[str, Any]) -> bool:
    return isinstance(config, Mapping) and config.get("config_schema") == "obs_equiv.config.v6.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prepare_output_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"output path is not a directory: {target}")
    if target.is_dir() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, document: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(document) + "\n")
    return target


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    rows = list(records)
    if not rows:
        raise SystemExit("refusing to write an empty JSONL dataset")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for record in rows:
            handle.write(canonical_json(record) + "\n")
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise SystemExit(f"empty sample set: {path}")
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> str:
    ordered = sorted(records, key=lambda pair: pair[0]["pair_id"])
    return sha256_json({
        "public": [public for public, _ in ordered],
        "evaluator": [evaluator for _, evaluator in ordered],
    })


def dataset_scope(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], allowed_claim: str) -> dict[str, Any]:
    if allowed_claim != ALLOWED_CLAIM:
        raise ContractError("only SYNTHETIC_DECLARED_DOMAIN_ONLY is supported")
    return {
        "truth_levels": sorted({evaluator["truth_level"] for _, evaluator in records}),
        "world_families": sorted({evaluator["world_family"] for _, evaluator in records}),
        "site_ids": sorted({evaluator["site_id"] for _, evaluator in records}),
        "split_groups": sorted({evaluator["split_group"] for _, evaluator in records}),
        "allowed_claim": allowed_claim,
        "REAL_GENERALIZATION_CLAIM": "NOT_SUPPORTED",
        "FULL_3D_REAL_TRUTH": "NOT_SUPPORTED",
    }


def save_dataset(records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], output_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not records:
        raise SystemExit("empty sample set")
    pair_ids = [public.get("pair_id") for public, _ in records]
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in pair_ids):
        raise ContractError("every pair_id must be a non-empty string")
    if len(pair_ids) != len(set(pair_ids)):
        raise ContractError("pair_id values must be globally unique")
    allowed_claim = config.get("allowed_claim")
    if allowed_claim != ALLOWED_CLAIM:
        raise ContractError("the current pilot requires SYNTHETIC_DECLARED_DOMAIN_ONLY")
    fingerprint = dataset_fingerprint(records)
    scope = dataset_scope(records, allowed_claim)
    validations = [validate_pair(public, evaluator) for public, evaluator in records]
    if any(not validation["pass"] for validation in validations):
        raise ContractError("dataset contains invalid pairs")
    change_count = sum(bool(evaluator["expected_query_change"]) for _, evaluator in records)
    control_count = len(records) - change_count
    if control_count < 1:
        raise ContractError("dataset requires at least one query-invariant control")
    ordered = sorted(records, key=lambda pair: pair[0]["pair_id"])
    target = prepare_output_dir(output_dir)
    public_path = write_jsonl(target / "public.jsonl", [public for public, _ in ordered])
    evaluator_path = write_jsonl(target / "evaluator.jsonl", [evaluator for _, evaluator in ordered])
    validation_document = {
        "schema": "obs_equiv.validation.v1",
        "pass": True,
        "dataset_sha256": fingerprint,
        "allowed_claim": allowed_claim,
        "pair_count": len(records),
        "query_change_count": change_count,
        "query_invariant_control_count": len(records) - change_count,
        "enumerated_checks": sum(validation["enumerated_checks"] for validation in validations),
        "public_row_order": "PAIR_ID_ASCENDING",
        "evaluator_row_order": "PAIR_ID_ASCENDING",
        "readings": {
            "scope": "ALL_GENERATED_PAIRS",
            "enumeration": "EXHAUSTIVE_PAIR_VALIDATION",
            "measured_at": utc_now(),
        },
    }
    validation_path = write_json(target / "validation.json", validation_document)
    files = {}
    for path in (public_path, evaluator_path, validation_path):
        files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    run = {
        "schema": SCHEMA_RUN,
        "status": "BUILT_AND_VALIDATED",
        "dataset_sha256": fingerprint,
        "scope": scope,
        "created_at_utc": utc_now(),
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "pair_count": len(records),
        "public_row_order": "PAIR_ID_ASCENDING",
        "evaluator_row_order": "PAIR_ID_ASCENDING",
        "files": files,
    }
    write_json(target / "run.json", run)
    return run


def load_dataset(data_dir: str | Path, validate: bool = True) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    root = Path(data_dir)
    if root.is_file():
        # R-21：把 config/草案误当数据集喂入三个数据入口时必须显式拒绝。
        raise ContractError("data_dir must be a dataset directory; a file is not a dataset")
    public_rows = read_jsonl(root / "public.jsonl")
    evaluator_rows = read_jsonl(root / "evaluator.jsonl")
    if len(public_rows) != len(evaluator_rows):
        raise ContractError("public and evaluator row counts differ")
    public_by_id = {}
    evaluator_by_id = {}
    for row in public_rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ContractError("public pair_id must be a non-empty string")
        if pair_id in public_by_id:
            raise ContractError(f"duplicate public pair_id: {pair_id}")
        public_by_id[pair_id] = row
    for row in evaluator_rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ContractError("evaluator pair_id must be a non-empty string")
        if pair_id in evaluator_by_id:
            raise ContractError(f"duplicate evaluator pair_id: {pair_id}")
        evaluator_by_id[pair_id] = row
    if set(public_by_id) != set(evaluator_by_id):
        raise ContractError("public and evaluator pair ids differ")
    records = [(public_by_id[pair_id], evaluator_by_id[pair_id]) for pair_id in sorted(public_by_id)]
    if validate:
        failures = [public["pair_id"] for public, evaluator in records if not validate_pair(public, evaluator)["pass"]]
        if failures:
            raise ContractError(f"invalid dataset pairs: {failures}")
    return records


def _expected_composition(config: Mapping[str, Any]) -> tuple[int, list[str], list[str], int, bool]:
    pair_count = config.get("pair_count")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count <= 0:
        raise ContractError("config pair_count must be a positive integer")
    mechanisms = config.get("mechanisms", ["HIDDEN_BRIDGE"])
    if not isinstance(mechanisms, list) or not mechanisms or any(
        not isinstance(name, str) for name in mechanisms
    ):
        raise ContractError("config mechanisms must be a non-empty list")
    if len(set(mechanisms)) != len(mechanisms):
        raise ContractError("config mechanisms must be unique")
    families = family_parameters(config)
    if families is None:
        family_names = []
        for mechanism in mechanisms:
            if mechanism not in LEGACY_V2_MECHANISM_WORLD_FAMILIES:
                raise ContractError(f"unknown configured mechanism: {mechanism}")
            family_names.append(LEGACY_V2_MECHANISM_WORLD_FAMILIES[mechanism])
        return pair_count, mechanisms, family_names, pair_count * len(mechanisms) * 2, False
    family_names = [item["name"] for item in families]
    return (
        pair_count,
        mechanisms,
        family_names,
        pair_count * len(mechanisms) * len(family_names) * 2,
        True,
    )


def _verify_v6_composition(config: Mapping[str, Any], records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
    """v6 组合校验（TO-W4-01）：按冻结 unit_manifest 核对单元×槽结构。"""
    from fractureflow.v6_obs_equiv import REPEAT_COUNT, frozen_unit_manifest

    units = frozen_unit_manifest()
    expected_rows = {
        unit["unit_id"]: (2 if unit["identifiability"] == "IDENTIFIABLE_FROM_PUBLIC_OBSERVATION" else 8)
        for unit in units
    }
    if len(records) != sum(expected_rows.values()):
        raise ContractError("v6 dataset size does not match the frozen unit manifest")
    split_by_unit = {unit["unit_id"]: unit["split_pool"] for unit in units}
    counts: dict[str, int] = {}
    slots: dict[str, set[int]] = {}
    for _, evaluator in records:
        uid = evaluator.get("unit_id")
        if uid not in expected_rows:
            raise ContractError(f"v6 row references unknown unit: {uid}")
        counts[uid] = counts.get(uid, 0) + 1
        slots.setdefault(uid, set()).add(evaluator.get("slot"))
        if evaluator.get("split_pool") != split_by_unit[uid]:
            raise ContractError(f"v6 split_pool differs from the frozen manifest: {uid}")
        if evaluator.get("repeat_index") not in range(REPEAT_COUNT):
            raise ContractError(f"v6 repeat index out of range: {uid}")
        if evaluator.get("config_schema") != "obs_equiv.config.v6.v1":
            raise ContractError("v6 evaluator row must carry the v6 config schema marker")
    for uid, expected in expected_rows.items():
        if counts.get(uid, 0) != expected:
            raise ContractError(f"v6 unit row count differs from the frozen manifest: {uid}")
        if expected == 8 and len(slots[uid]) != 8:
            raise ContractError(f"v6 non-identifiable unit must occupy 8 distinct slots: {uid}")
        if expected == 2 and len(slots[uid]) != 2:
            raise ContractError(f"v6 identifiable unit must occupy 2 distinct slots: {uid}")


def _verify_data_run(
    data_dir: str | Path,
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[dict[str, Any], str, Mapping[str, Any]]:
    root = Path(data_dir)
    run = read_json(root / "run.json")
    fingerprint = dataset_fingerprint(records)
    if run.get("dataset_sha256") != fingerprint:
        raise ContractError("dataset fingerprint differs from data run")
    if run.get("pair_count") != len(records):
        raise ContractError("dataset pair count differs from data run")
    config = run.get("config")
    if not isinstance(config, Mapping) or config.get("allowed_claim") != ALLOWED_CLAIM:
        raise ContractError("data run config does not declare the allowed claim")
    config_sha = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if run.get("config_sha256") != config_sha:
        raise ContractError("data run config hash differs")
    site_id = config.get("site_id")
    split_group = config.get("split_group")
    if not isinstance(site_id, str) or not site_id or not isinstance(split_group, str) or not split_group:
        raise ContractError("config site_id/split_group missing")
    if config.get("config_schema") == "obs_equiv.config.v6.v1":
        # v6（TO-W4-01）：组合校验按冻结 unit_manifest，不走 v5 的 pair_count 公式。
        for _, evaluator in records:
            if evaluator.get("site_id") != site_id or evaluator.get("split_group") != split_group:
                raise ContractError("dataset site/split differs from config")
        _verify_v6_composition(config, records)
    else:
        pair_count, mechanisms, families, expected_size, decoupled = _expected_composition(config)
        if len(records) != expected_size:
            raise ContractError("dataset size does not match configured composition")
        if decoupled:
            cells: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
            for _, evaluator in records:
                if evaluator.get("site_id") != site_id or evaluator.get("split_group") != split_group:
                    raise ContractError("dataset site/split differs from config")
                cells.setdefault(
                    (evaluator.get("mechanism"), evaluator.get("world_family")), []
                ).append(evaluator)
            expected_cells = {(mechanism, family) for mechanism in mechanisms for family in families}
            if set(cells) != expected_cells:
                raise ContractError("dataset mechanism x family cells differ from config")
            for cell, cell_rows in cells.items():
                cell_changes = sum(bool(row.get("expected_query_change")) for row in cell_rows)
                cell_controls = len(cell_rows) - cell_changes
                if cell_changes != pair_count or cell_controls != pair_count:
                    raise ContractError(f"mechanism x family composition differs from config: {cell}")
        else:
            by_family = {}
            for _, evaluator in records:
                if evaluator.get("site_id") != site_id or evaluator.get("split_group") != split_group:
                    raise ContractError("dataset site/split differs from config")
                by_family.setdefault(evaluator.get("world_family"), []).append(evaluator)
            if set(by_family) != set(families):
                raise ContractError("dataset world families differ from configured mechanisms")
            for family in families:
                family_rows = by_family[family]
                family_changes = sum(bool(row.get("expected_query_change")) for row in family_rows)
                family_controls = len(family_rows) - family_changes
                if family_changes != pair_count or family_controls != pair_count:
                    raise ContractError(f"family composition differs from config: {family}")
    scope = run.get("scope")
    expected_scope = dataset_scope(records, ALLOWED_CLAIM)
    if not isinstance(scope, Mapping) or canonical_json(scope) != canonical_json(expected_scope):
        raise ContractError("dataset scope differs from data run")
    files = run.get("files")
    if not isinstance(files, Mapping):
        raise ContractError("data run files must be an object")
    for name in ("public.jsonl", "evaluator.jsonl", "validation.json"):
        entry = files.get(name)
        path = root / name
        if not isinstance(entry, Mapping) or not path.is_file():
            raise ContractError(f"data run is missing {name}")
        if entry.get("sha256") != file_sha256(path) or entry.get("bytes") != path.stat().st_size:
            raise ContractError(f"data file hash differs: {name}")
    validation = read_json(root / "validation.json")
    if validation.get("pass") is not True or validation.get("dataset_sha256") != fingerprint:
        raise ContractError("validation artifact is not bound to this dataset")
    if validation.get("pair_count") != len(records):
        raise ContractError("validation pair count differs from dataset")
    change_count = sum(bool(evaluator["expected_query_change"]) for _, evaluator in records)
    if validation.get("query_change_count") != change_count:
        raise ContractError("validation query-change count differs from dataset")
    if validation.get("query_invariant_control_count") != len(records) - change_count:
        raise ContractError("validation control count differs from dataset")
    if validation.get("allowed_claim") != ALLOWED_CLAIM:
        raise ContractError("validation allowed_claim differs from the frozen claim")
    expected_checks = sum(
        validate_pair(public, evaluator)["enumerated_checks"]
        for public, evaluator in records
    )
    if validation.get("enumerated_checks") != expected_checks:
        raise ContractError("validation check count differs from dataset")
    return run, fingerprint, scope


def validate_dataset(data_dir: str | Path) -> dict[str, Any]:
    try:
        records = load_dataset(data_dir, validate=False)
        run, fingerprint, scope = _verify_data_run(data_dir, records)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        return {
            "schema": "obs_equiv.revalidation.v1",
            "pass": False,
            "pair_count": 0,
            "failed_pair_ids": [],
            "enumerated_checks": 0,
            "errors": [str(exc)],
            "run_binding": {"pass": False},
            "measured_at": utc_now(),
        }
    rows = []
    for public, evaluator in records:
        result = validate_pair(public, evaluator)
        rows.append({"pair_id": public["pair_id"], **result})
    failed = [row["pair_id"] for row in rows if not row["pass"]]
    return {
        "schema": "obs_equiv.revalidation.v1",
        "pass": not failed,
        "dataset_sha256": fingerprint,
        "pair_count": len(rows),
        "failed_pair_ids": failed,
        "enumerated_checks": sum(row["enumerated_checks"] for row in rows),
        "rows": rows,
        "run_binding": {
            "pass": True,
            "config_sha256": run["config_sha256"],
            "scope": scope,
        },
        "measured_at": utc_now(),
    }


def public_visible_geometry(public: Mapping[str, Any]) -> tuple[Scene, TwoPointQuery]:
    assert_public_safe(public)
    observation = public["observation"] if "observation" in public else public
    disks = tuple(
        Disk(item["public_id"], item["center"], item["normal"], item["radius"])
        for item in observation["observed_objects"]
    )
    return Scene("public_visible_only", disks), TwoPointQuery.from_dict(observation["query"])


def query_distance_prior_parameters(config: Mapping[str, Any]) -> tuple[float, float]:
    if not isinstance(config, Mapping):
        raise ContractError("config must be an object")
    params = config.get("query_distance_scaled_prior")
    if not isinstance(params, Mapping):
        raise ContractError("config must declare query_distance_scaled_prior")
    if set(params) != {"d0", "tau"}:
        raise ContractError("query_distance_scaled_prior must contain exactly d0 and tau")
    d0 = finite_number(params.get("d0"), "d0")
    tau = finite_number(params.get("tau"), "tau")
    if tau <= 0.0:
        raise ContractError("tau must be positive")
    return d0, tau


def query_endpoint_distance(public: Mapping[str, Any]) -> float:
    observation = public["observation"] if "observation" in public else public
    params = observation["query"]["params"]
    point_a = params["A"]
    point_b = params["B"]
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))


def query_distance_scaled_prior_prediction(
    public: Mapping[str, Any],
    d0: float,
    tau: float,
) -> float:
    """族敏感零训练基线：只读 public 查询端点距离 d。

    形式 p = 1 / (1 + exp((d - d0) / tau))。d0 与 tau 是 config 预注册常数，
    本函数不读 evaluator、不读 world_family、不读 mechanism。
    """
    assert_public_safe(public)
    distance = query_endpoint_distance(public)
    return 1.0 / (1.0 + math.exp((distance - d0) / tau))


def zero_training_geometry_prediction(public: Mapping[str, Any]) -> float:
    scene, query = public_visible_geometry(public)
    return 1.0 if two_point_connected(scene, query) else 0.0


def run_baselines(data_dir: str | Path, output_dir: str | Path, constant_prior: float) -> dict[str, Any]:
    prior = finite_number(constant_prior, "constant_prior")
    if not 0.0 <= prior <= 1.0:
        raise ContractError("constant_prior must be in [0, 1]")
    records = load_dataset(data_dir)
    data_run, fingerprint, scope = _verify_data_run(data_dir, records)
    declared_prior = finite_number(
        data_run["config"].get("constant_baseline_prior"), "constant_baseline_prior"
    )
    if abs(prior - declared_prior) > 1e-12:
        raise ContractError("constant prior differs from declared config")
    d0, tau = query_distance_prior_parameters(data_run["config"])
    target = prepare_output_dir(output_dir)
    v6 = _is_v6_config(data_run["config"])
    predictions = []
    for public, evaluator in records:
        geometry_probability = zero_training_geometry_prediction(public)
        distance_probability = query_distance_scaled_prior_prediction(public, d0, tau)
        for method, probability in (
            ("CONSTANT_PRIOR", prior),
            ("ZERO_TRAINING_GEOMETRY", geometry_probability),
            ("QUERY_DISTANCE_SCALED_PRIOR", distance_probability),
            ("REFERENCE_POSTERIOR_DIAGNOSTIC", float(evaluator["reference_posterior"])),
        ):
            row = {
                "pair_id": public["pair_id"],
                "method": method,
                "p_connected": probability,
                "coverage": 1.0,
                "status": "PREDICTED",
                "allowed_claim": METHOD_CLAIMS[method],
            }
            if v6:
                row.update(V6_METHOD_ROLES[method])
            predictions.append(row)
    prediction_path = write_jsonl(target / "predictions.jsonl", predictions)
    run = {
        "schema": "obs_equiv.baseline_run.v1",
        "dataset_sha256": fingerprint,
        "scope": scope,
        "created_at_utc": utc_now(),
        "data_dir": str(Path(data_dir).resolve()),
        "methods": list(METHOD_CLAIMS),
        "method_claims": dict(METHOD_CLAIMS),
        "baseline_scopes": {
            "CONSTANT_PRIOR": "DECLARED_PRIOR_ONLY",
            "ZERO_TRAINING_GEOMETRY": "PUBLIC_OBSERVED_OBJECTS_ONLY",
            "QUERY_DISTANCE_SCALED_PRIOR": "PUBLIC_QUERY_ENDPOINT_DISTANCE_ONLY",
            "REFERENCE_POSTERIOR_DIAGNOSTIC": "EVALUATOR_DIAGNOSTIC_ONLY",
        },
        "constant_prior": prior,
        "prior_source": "USER_DECLARED_NOT_ESTIMATED_FROM_TEST",
        "query_distance_scaled_prior": {"d0": d0, "tau": tau},
        "selection_events": 0,
        "prediction_count": len(predictions),
        "predictions_sha256": file_sha256(prediction_path),
    }
    if v6:
        # R-08：角色映射随 run.json 冻结；反例方法只作门用派生，不新增预测行。
        run["method_roles"] = dict(V6_METHOD_ROLES)
        run["method_roles"]["COUNTEREXAMPLE_GEOMETRY_FLIP"] = {
            "metric_role": "GATE_COUNTEREXAMPLE_DIAGNOSTIC",
            "deployable_predictor": False,
            "uses_held_out_target_for_selection": True,
            "prediction_rows_effect": "GATE_ONLY_DERIVED_NO_NEW_PREDICTION_ROWS",
        }
    write_json(target / "run.json", run)
    return run


def _probability(value: Any) -> float:
    probability = finite_number(value, "p_connected")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("predicted probability must be in [0, 1]")
    return probability


def _metrics(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    if not rows:
        return {"status": "EMPTY_NOT_APPLICABLE", "n_groups": 0}
    brier = 0.0
    log_loss = 0.0
    posterior_l1 = 0.0
    posterior_squared = 0.0
    collapsed = 0
    reference_interior = 0
    high_confidence_error = 0.0
    scene_count = 0
    for _, evaluator, prediction in rows:
        probability = _probability(prediction["p_connected"])
        reference = float(evaluator["reference_posterior"])
        weights = evaluator["prior_weights"]
        labels = evaluator["query_labels"]
        scene_count += len(labels)
        for weight, label in zip(weights, labels):
            target = 1.0 if label else 0.0
            brier += float(weight) * (probability - target) ** 2
            log_loss -= float(weight) * (
                target * math.log(max(probability, LOG_EPSILON))
                + (1.0 - target) * math.log(max(1.0 - probability, LOG_EPSILON))
            )
            predicted_label = probability >= 0.5
            confident = max(probability, 1.0 - probability) >= HIGH_CONFIDENCE_THRESHOLD
            if confident and predicted_label != bool(label):
                high_confidence_error += float(weight)
        posterior_l1 += abs(probability - reference)
        posterior_squared += (probability - reference) ** 2
        if LOG_EPSILON < reference < 1.0 - LOG_EPSILON:
            reference_interior += 1
            if probability <= COLLAPSE_THRESHOLD or probability >= 1.0 - COLLAPSE_THRESHOLD:
                collapsed += 1
    count = len(rows)
    return {
        "status": "ESTIMATED",
        "n_groups": count,
        "n_scenes": scene_count,
        "expected_query_brier": brier / count,
        "expected_query_log_loss": log_loss / count,
        "posterior_l1": posterior_l1 / count,
        "posterior_rmse": math.sqrt(posterior_squared / count),
        "n_reference_interior": reference_interior,
        "posterior_collapse_rate": collapsed / reference_interior if reference_interior else None,
        "posterior_collapse_status": "ESTIMATED" if reference_interior else "NOT_APPLICABLE_NO_INTERIOR_REFERENCE",
        "expected_high_confidence_error_rate": high_confidence_error / count,
    }


def evaluate_predictions(data_dir: str | Path, predictions_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    records = load_dataset(data_dir)
    data_run, fingerprint, scope_document = _verify_data_run(data_dir, records)
    prediction_path = Path(predictions_dir) / "predictions.jsonl"
    prediction_run = read_json(Path(predictions_dir) / "run.json")
    if prediction_run.get("dataset_sha256") != fingerprint:
        raise ContractError("predictions are not bound to this dataset version")
    if canonical_json(prediction_run.get("scope")) != canonical_json(scope_document):
        raise ContractError("prediction scope differs from dataset scope")
    if prediction_run.get("methods") != list(METHOD_CLAIMS):
        raise ContractError("prediction method list differs from the frozen method set")
    if prediction_run.get("method_claims") != METHOD_CLAIMS:
        raise ContractError("prediction method claims differ from the frozen mapping")
    prediction_rows = read_jsonl(prediction_path)
    if prediction_run.get("prediction_count") != len(prediction_rows):
        raise ContractError("prediction count differs from prediction run")
    if prediction_run.get("predictions_sha256") != file_sha256(prediction_path):
        raise ContractError("prediction file hash differs from prediction run")
    constant_prior = finite_number(prediction_run.get("constant_prior"), "constant_prior")
    if not 0.0 <= constant_prior <= 1.0:
        raise ContractError("constant prior must be in [0, 1]")
    declared_prior = finite_number(
        data_run["config"].get("constant_baseline_prior"), "constant_baseline_prior"
    )
    if abs(constant_prior - declared_prior) > 1e-12:
        raise ContractError("constant prior differs from declared config")
    d0, tau = query_distance_prior_parameters(data_run["config"])
    declared_distance = prediction_run.get("query_distance_scaled_prior")
    if canonical_json(declared_distance) != canonical_json({"d0": d0, "tau": tau}):
        raise ContractError("query distance prior constants differ from declared config")
    public_by_id = {public["pair_id"]: public for public, _ in records}
    evaluator_by_id = {evaluator["pair_id"]: evaluator for _, evaluator in records}
    grouped = {}
    seen = set()
    for prediction in prediction_rows:
        pair_id = prediction.get("pair_id")
        method = prediction.get("method")
        if pair_id not in evaluator_by_id:
            raise ContractError(f"prediction references unknown pair: {pair_id}")
        if method not in METHOD_CLAIMS:
            raise ContractError(f"unknown prediction method: {method}")
        key = (pair_id, method)
        if key in seen:
            raise ContractError(f"duplicate prediction: {key}")
        if prediction.get("allowed_claim") != METHOD_CLAIMS[method]:
            raise ContractError(f"prediction claim differs from frozen mapping: {key}")
        if prediction.get("status") != "PREDICTED":
            raise ContractError(f"non-predicted row cannot enter current metrics: {key}")
        if finite_number(prediction.get("coverage"), "coverage") != 1.0:
            raise ContractError(f"current metrics require full coverage: {key}")
        seen.add(key)
        grouped.setdefault(method, []).append((pair_id, evaluator_by_id[pair_id], prediction))
    if not grouped:
        raise SystemExit("empty prediction set")
    if set(grouped) != set(METHOD_CLAIMS):
        raise ContractError("prediction methods do not cover the frozen method set")
    for method, rows in grouped.items():
        for pair_id, evaluator, prediction in rows:
            probability = _probability(prediction["p_connected"])
            if method == "CONSTANT_PRIOR":
                expected = constant_prior
            elif method == "ZERO_TRAINING_GEOMETRY":
                expected = zero_training_geometry_prediction(public_by_id[pair_id])
            elif method == "QUERY_DISTANCE_SCALED_PRIOR":
                expected = query_distance_scaled_prior_prediction(
                    public_by_id[pair_id], d0, tau
                )
            else:
                expected = float(evaluator["reference_posterior"])
            if abs(probability - expected) > 1e-12:
                raise ContractError(
                    f"prediction value differs from recomputation: {(pair_id, method)}"
                )
    methods = {}
    expected_pair_ids = set(evaluator_by_id)
    for method, rows in sorted(grouped.items()):
        actual_pair_ids = {pair_id for pair_id, _, _ in rows}
        if actual_pair_ids != expected_pair_ids:
            raise ContractError(f"method {method} does not cover every dataset pair")
        scopes = {"query_change": [], "query_invariant_control": []}
        for pair_id, evaluator, prediction in rows:
            scope = "query_change" if evaluator["expected_query_change"] else "query_invariant_control"
            scopes[scope].append((pair_id, evaluator, prediction))
        if not scopes["query_change"]:
            raise SystemExit(f"primary query_change scope is empty for method {method}")
        methods[method] = {scope: _metrics(scope_rows) for scope, scope_rows in scopes.items()}
    score = {
        "schema": SCHEMA_SCORE,
        "dataset_sha256": fingerprint,
        "scope": scope_document,
        "allowed_claim": scope_document["allowed_claim"],
        "REAL_GENERALIZATION_CLAIM": "NOT_SUPPORTED",
        "FULL_3D_REAL_TRUTH": "NOT_SUPPORTED",
        "created_at_utc": utc_now(),
        "protocol": {
            "primary_scope": "query_change",
            "control_scope": "query_invariant_control",
            "scopes_not_combined": True,
            "log_epsilon": LOG_EPSILON,
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "reference_posterior_allowed_claim": "DIAGNOSTIC_ONLY",
            "method_claims": dict(METHOD_CLAIMS),
            "diagnostic_methods": [
                name for name, claim in METHOD_CLAIMS.items() if claim == "DIAGNOSTIC_ONLY"
            ],
        },
        "readings": {
            "scope": "ALL_PREDICTIONS_BY_METHOD_AND_DECLARED_SCOPE",
            "enumeration": "EXHAUSTIVE_GROUP_EVALUATION",
            "measured_at": utc_now(),
        },
        "methods": methods,
    }
    write_json(output_path, score)
    return score
