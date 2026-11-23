from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fractureflow.obs_equiv_benchmark import (
    ALLOWED_CLAIM,
    METHOD_CLAIMS,
    dataset_fingerprint,
    dataset_scope,
    file_sha256,
    load_dataset,
    public_visible_geometry,
    read_json,
    zero_training_geometry_prediction,
)
from fractureflow.obs_equiv_geometry import reference_posterior
from fractureflow.obs_equiv_manifest import (
    CANONICALIZATION_FIELDS,
    FORBIDDEN_PUBLIC_KEYS,
    NOISE_FIELDS,
    OBSERVATION_FIELDS,
    OBSERVED_OBJECT_FIELDS,
    PUBLIC_RECORD_FIELDS,
    QUERY_FIELDS,
    QUERY_PARAM_FIELDS,
    VISIBILITY_FIELDS,
    ContractError,
    Scene,
    TwoPointQuery,
    assert_public_safe,
    canonical_json,
)
from fractureflow.observation_equivalence import (
    SCHEMA_EVALUATOR_V2,
    TRANSFORM_NOISE_NAMES,
    apply_observation_transform,
    validate_pair,
)

SCHEMA = "obs_equiv.leak_matrix.v1"
SCHEMA_PUBLIC_HASHES = "obs_equiv.public_hash_evidence.v1"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOTE = "NOTE"
STATUS_BLOCKED = "BLOCKED_BY_DESIGN"
STATUS_NOT_RUN = "NOT_RUN_AFTER_BLOCKER"

SEVERITY_BLOCKER = "BLOCKER"
SEVERITY_NOTE = "NOTE"
SEVERITY_INFO = "INFO"

CATEGORY_LEAK = "LEAK"
CATEGORY_INTEGRITY = "INTEGRITY"
CATEGORY_DESIGN = "DESIGN"
CATEGORY_COVERAGE = "COVERAGE"
CATEGORY_INFO = "INFO"

VERDICT_NO_LEAK = "E0_NO_LEAK"
VERDICT_LEAK = "E0_LEAK_FOUND"
VERDICT_BLOCKED = "BLOCKED_BY_DESIGN"

EXPECTED_PUBLIC_RECORD_FIELDS = frozenset({"pair_id", "split_group", "observation"})
EXPECTED_OBSERVATION_FIELDS = frozenset({
    "schema", "modality", "coordinate_frame", "query", "observed_objects",
    "visibility", "noise", "canonicalization",
})
EXPECTED_OBSERVED_OBJECT_FIELDS = frozenset({"public_id", "type", "center", "normal", "radius"})
EXPECTED_QUERY_FIELDS = frozenset({"query_id", "family", "params"})
EXPECTED_QUERY_PARAM_FIELDS = frozenset({
    "A", "B", "r_attach", "attachment", "intersection", "outside_domain_routing",
})
EXPECTED_VISIBILITY_FIELDS = frozenset({"policy", "unexposed_objects"})
EXPECTED_NOISE_FIELDS = frozenset({"name", "order"})
EXPECTED_CANONICALIZATION_FIELDS = frozenset({"float_policy", "negative_zero", "object_order"})

EXPECTED_PUBLIC_ID = re.compile(r"p[0-9a-f]{16}")
EXPECTED_PAIR_ID = re.compile(r"oe_[0-9a-f]{16}")
EXPECTED_QUERY_ID = re.compile(r"q_[0-9a-f]{16}")

MECHANISM_TOKENS = (
    "HIDDEN_BRIDGE", "DECOY_BRIDGE", "OCCLUDED_CONTINUATION",
    "QUERY_INVARIANT_CONTROL", "ZONE_WIDTH",
)
WORLD_FAMILY_TOKENS = (
    "PLANAR_DISK_HIDDEN_BRIDGE_V1", "PLANAR_DISK_DECOY_BRIDGE_V1",
    "PLANAR_DISK_OCCLUDED_CONTINUATION_V1", "PLANAR_DISK",
)
TRUTH_TOKENS = (
    "hidden", "truth", "oracle", "prior", "posterior", "reference",
    "ground_truth", "scene_id", "structure_id", "visible_object_ids",
    "expected_query_change", "world_family", "mechanism", "site_id",
    "SYNTHETIC_PILOT", "EXACT_SYNTHETIC",
)
OBJECT_ID_TOKENS = (
    "bridge", "decoy", "attachment", "stub", "continuation",
    "long", "short", "left", "right", "with", "without",
)
SCENE_ID_TOKENS = (
    "oe_000000", "_with", "_without", "_attached", "_decoy_only", "_long", "_short",
)
PACKAGING_TOKENS = (
    "hidden", "truth", "oracle", "bridge", "decoy", "continuation", "attachment",
    "stub", "mechanism", "world_family", "prior", "posterior", "label",
    "train", "test", "holdout", "calib",
)

EVALUATOR_ONLY_PREDICTION_KEYS = frozenset({
    "scenes", "query_labels", "prior_weights", "reference_posterior",
    "expected_query_change", "mechanism", "world_family", "site_id",
    "observation_sha256", "query", "visible_object_ids", "truth_level",
    "group_type",
})
PREDICTION_FIELDS = frozenset({
    "pair_id", "method", "p_connected", "coverage", "status", "allowed_claim",
})
# R-08（v6）：prediction 行可带三个角色旗标；不得出现方法名黑名单语义。
V6_ROLE_PREDICTION_FIELDS = frozenset({
    "metric_role", "deployable_predictor", "uses_held_out_target_for_selection",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("JSON object keys must be strings")
            result[key] = _canonical(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite floats are forbidden")
        return 0.0 if value == 0.0 else value
    raise ContractError(f"unsupported JSON value: {type(value).__name__}")


def canonical_text(document: Any) -> str:
    return json.dumps(_canonical(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_document(document: Any) -> str:
    return hashlib.sha256(canonical_text(document).encode("utf-8")).hexdigest()


def _read_jsonl_with_lines(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    line_numbers: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number} must contain an object")
        rows.append(value)
        line_numbers.append(line_number)
    if not rows:
        raise SystemExit(f"empty sample set: {path}")
    return rows, line_numbers


def _safe_assert_public(public: Any) -> tuple[bool, str | None]:
    try:
        assert_public_safe(public)
        return True, None
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def _safe_validate(public: Any, evaluator: Any) -> tuple[bool, Any]:
    try:
        result = validate_pair(public, evaluator)
        return bool(result.get("pass")), result
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        return False, {"error": str(exc)}


def _geometry_of(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": obj["type"],
        "center": list(obj["center"]),
        "normal": list(obj["normal"]),
        "radius": obj["radius"],
    }


def public_geometry_id(geometry: Mapping[str, Any]) -> str:
    return "p" + sha256_document(geometry)[:16]


def query_geometry_id(query: Mapping[str, Any]) -> str:
    params = query["params"]
    return "q_" + sha256_document({
        "A": list(params["A"]),
        "B": list(params["B"]),
        "r_attach": params["r_attach"],
        "attachment": params["attachment"],
        "intersection": params["intersection"],
        "outside_domain_routing": params["outside_domain_routing"],
    })[:16]


def pair_observation_id(observation: Mapping[str, Any]) -> str:
    return "oe_" + sha256_document(observation)[:16]


def project_public(scene: Mapping[str, Any], visible_ids: Sequence[str], query: Mapping[str, Any]) -> dict[str, Any]:
    visible = set(visible_ids)
    objects = []
    for obj in scene.get("objects", []):
        if obj.get("id") not in visible:
            continue
        geometry = _geometry_of(obj)
        objects.append({"public_id": public_geometry_id(geometry), **geometry})
    objects.sort(key=lambda item: item["public_id"])
    return {
        "schema": "obs_equiv.observation.v1",
        "modality": "SYNTHETIC_VISIBLE_DISKS",
        "coordinate_frame": "WORLD",
        "query": copy.deepcopy(query),
        "observed_objects": objects,
        "visibility": {
            "policy": "EXPLICIT_OBJECT_EXPOSURE",
            "unexposed_objects": "OMITTED_NOT_ABSENT",
        },
        "noise": {"name": "NONE", "order": "PUBLIC_OBSERVATION_ONLY"},
        "canonicalization": {
            "float_policy": "JSON_FLOAT_NO_ROUNDING",
            "negative_zero": "ZERO",
            "object_order": "PUBLIC_ID_ASCENDING",
        },
    }


def _row_transform(evaluator: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """E6D 逐行分支：有 observation_transform 的行是变换行，否则 legacy clean 行。"""
    transform = evaluator.get("observation_transform")
    return transform if isinstance(transform, Mapping) else None


def _expected_public(
    evaluator: Mapping[str, Any],
    scene: Mapping[str, Any],
    query: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """transform-aware base projection：clean 投影后按需施加该行声明的变换。"""
    projected = project_public(
        scene, evaluator["visible_object_ids"], query if query is not None else evaluator["query"]
    )
    transform = _row_transform(evaluator)
    if transform is not None:
        projected = apply_observation_transform(projected, transform)
    return projected


def _declared_noise_name(evaluator: Mapping[str, Any]) -> str:
    """P22 语义：变换行按臂/模式声明 noise.name；clean/control 行必须是 NONE。"""
    transform = _row_transform(evaluator)
    if transform is None:
        return "NONE"
    arm = transform.get("arm_id")
    if arm == "OBSERVATION_NOISE_LADDER":
        return TRANSFORM_NOISE_NAMES["OBSERVATION_NOISE_LADDER"]
    if arm == "SENSOR_OFFSET":
        return TRANSFORM_NOISE_NAMES[f"SENSOR_OFFSET:{transform.get('mode')}"]
    return "NONE"


def mutate_object(obj: Mapping[str, Any], kind: str) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(obj))
    if kind == "center":
        mutated["center"] = [
            mutated["center"][0] + 0.137,
            mutated["center"][1] - 0.211,
            mutated["center"][2] + 0.173,
        ]
    elif kind == "radius":
        mutated["radius"] = mutated["radius"] * 1.07 if mutated["radius"] else 0.11
    elif kind == "normal":
        mutated["normal"] = [-component for component in mutated["normal"]]
    else:
        raise ValueError(f"unknown mutation kind: {kind}")
    return mutated

def _scene_with_object(scene: Mapping[str, Any], object_id: str, mutation: str) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(scene))
    for index, obj in enumerate(mutated["objects"]):
        if obj["id"] == object_id:
            if mutation == "remove":
                del mutated["objects"][index]
            else:
                mutated["objects"][index] = mutate_object(obj, mutation)
            return mutated
    raise ContractError(f"scene object not found: {object_id}")


def _scene_with_added_hidden(scene: Mapping[str, Any], existing_ids: set[str]) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(scene))
    object_id = "__leak_probe_hidden__"
    while object_id in existing_ids:
        object_id += "_"
    mutated["objects"].append({
        "id": object_id,
        "type": "DISK",
        "center": [1.25, 0.75, 0.5],
        "normal": [0.0, 0.0, 1.0],
        "radius": 0.75,
    })
    return mutated


def _query_with_mutation(query: Mapping[str, Any], kind: str) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(query))
    params = mutated["params"]
    if kind == "A":
        params["A"] = [params["A"][0] + 0.25, params["A"][1], params["A"][2]]
    elif kind == "B":
        params["B"] = [params["B"][0] - 0.25, params["B"][1], params["B"][2]]
    elif kind == "r_attach":
        params["r_attach"] = params["r_attach"] + 0.125
    else:
        raise ValueError(f"unknown query mutation: {kind}")
    return mutated


def _hidden_objects(scene: Mapping[str, Any], visible_ids: Sequence[str]) -> list[dict[str, Any]]:
    visible = set(visible_ids)
    return [obj for obj in scene["objects"] if obj["id"] not in visible]


def _sample(items: Sequence[Any], limit: int = 5) -> list[Any]:
    return list(items[:limit])


def _string_values(document: Any) -> list[str]:
    values: list[str] = []
    if isinstance(document, Mapping):
        for item in document.values():
            values.extend(_string_values(item))
    elif isinstance(document, (list, tuple)):
        for item in document:
            values.extend(_string_values(item))
    elif isinstance(document, str):
        values.append(document)
    return values


def _mutual_information(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    pairs = list(zip(left, right))
    n = len(pairs)
    left_counts: dict[Any, int] = {}
    right_counts: dict[Any, int] = {}
    joint_counts: dict[tuple[Any, Any], int] = {}
    for a, b in pairs:
        left_counts[a] = left_counts.get(a, 0) + 1
        right_counts[b] = right_counts.get(b, 0) + 1
        joint_counts[(a, b)] = joint_counts.get((a, b), 0) + 1
    value = 0.0
    for (a, b), count in joint_counts.items():
        joint = count / n
        marginal = (left_counts[a] / n) * (right_counts[b] / n)
        value += joint * math.log2(joint / marginal)
    return max(0.0, value)


class CheckLog:
    def __init__(self) -> None:
        self.count = 0
        self.failures: list[dict[str, Any]] = []

    def check(self, ok: bool, label: str, detail: Any = None) -> bool:
        self.count += 1
        if not ok:
            self.failures.append({"label": label, "detail": detail})
        return bool(ok)

    def leak_failures(self) -> bool:
        return any(item["label"].startswith("leak:") for item in self.failures)


@dataclass(frozen=True)
class ProbeOutcome:
    status: str
    severity: str
    category: str
    observed: str
    evidence: Mapping[str, Any]
    enumerated_checks: int


@dataclass(frozen=True)
class Probe:
    probe_id: str
    attack_surface: str
    expectation: str
    func: Callable[["MatrixContext"], ProbeOutcome]


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    attack_surface: str
    expectation: str
    observed: str
    status: str
    severity: str
    category: str
    enumerated_checks: int
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "attack_surface": self.attack_surface,
            "expectation": self.expectation,
            "observed": self.observed,
            "status": self.status,
            "severity": self.severity,
            "category": self.category,
            "enumerated_checks": self.enumerated_checks,
            "evidence": self.evidence,
        }


def _finish(
    checks: CheckLog,
    observed: str,
    evidence: Mapping[str, Any],
    *,
    note: bool = False,
    blocked: bool = False,
) -> ProbeOutcome:
    document = dict(evidence)
    document["enumerated_checks"] = checks.count
    if checks.failures:
        document["failures"] = checks.failures
        category = CATEGORY_LEAK if checks.leak_failures() else CATEGORY_INTEGRITY
        return ProbeOutcome(STATUS_FAIL, SEVERITY_BLOCKER, category, observed, document, checks.count)
    if blocked:
        return ProbeOutcome(STATUS_BLOCKED, SEVERITY_NOTE, CATEGORY_COVERAGE, observed, document, checks.count)
    if note:
        return ProbeOutcome(STATUS_NOTE, SEVERITY_NOTE, CATEGORY_DESIGN, observed, document, checks.count)
    return ProbeOutcome(STATUS_PASS, SEVERITY_INFO, CATEGORY_INFO, observed, document, checks.count)


class MatrixContext:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data directory not found: {self.data_dir}")
        self.public_path = self.data_dir / "public.jsonl"
        self.evaluator_path = self.data_dir / "evaluator.jsonl"
        self.run_path = self.data_dir / "run.json"
        self.validation_path = self.data_dir / "validation.json"
        self.public_text = self.public_path.read_text(encoding="utf-8")
        self.evaluator_text = self.evaluator_path.read_text(encoding="utf-8")
        self.public_rows, self.public_line_numbers = _read_jsonl_with_lines(self.public_path)
        self.evaluator_rows, self.evaluator_line_numbers = _read_jsonl_with_lines(self.evaluator_path)
        if len(self.public_rows) != len(self.evaluator_rows):
            raise ContractError("public and evaluator row counts differ")
        self.public_by_id: dict[str, dict[str, Any]] = {}
        self.evaluator_by_id: dict[str, dict[str, Any]] = {}
        for row in self.public_rows:
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ContractError("public pair_id must be a non-empty string")
            if pair_id in self.public_by_id:
                raise ContractError(f"duplicate public pair_id: {pair_id}")
            self.public_by_id[pair_id] = row
        for row in self.evaluator_rows:
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ContractError("evaluator pair_id must be a non-empty string")
            if pair_id in self.evaluator_by_id:
                raise ContractError(f"duplicate evaluator pair_id: {pair_id}")
            self.evaluator_by_id[pair_id] = row
        if set(self.public_by_id) != set(self.evaluator_by_id):
            raise ContractError("public and evaluator pair ids differ")
        self.records = list(zip(self.public_rows, self.evaluator_rows))
        for public, evaluator in self.records:
            if public.get("pair_id") != evaluator.get("pair_id"):
                raise ContractError("public and evaluator line order differs")
            result = validate_pair(public, evaluator)
            if not result["pass"]:
                raise ContractError(f"invalid pair {public.get('pair_id')}: {result['errors']}")
        self.run = read_json(self.run_path)
        self.validation = read_json(self.validation_path)
        self.public_sha256 = file_sha256(self.public_path)
        self.evaluator_sha256 = file_sha256(self.evaluator_path)
        self.run_sha256 = file_sha256(self.run_path)
        self.validation_sha256 = file_sha256(self.validation_path)
        self.dataset_sha256 = self.run.get("dataset_sha256")
        self.public_line_text = {
            index: line for index, line in enumerate(self.public_text.splitlines(), 1)
        }

    @property
    def pair_count(self) -> int:
        return len(self.public_rows)


def probe_schema_whitelist(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    for constant, expected, name in (
        (PUBLIC_RECORD_FIELDS, EXPECTED_PUBLIC_RECORD_FIELDS, "public_record"),
        (OBSERVATION_FIELDS, EXPECTED_OBSERVATION_FIELDS, "observation"),
        (OBSERVED_OBJECT_FIELDS, EXPECTED_OBSERVED_OBJECT_FIELDS, "observed_object"),
        (QUERY_FIELDS, EXPECTED_QUERY_FIELDS, "query"),
        (QUERY_PARAM_FIELDS, EXPECTED_QUERY_PARAM_FIELDS, "query_params"),
        (VISIBILITY_FIELDS, EXPECTED_VISIBILITY_FIELDS, "visibility"),
        (NOISE_FIELDS, EXPECTED_NOISE_FIELDS, "noise"),
        (CANONICALIZATION_FIELDS, EXPECTED_CANONICALIZATION_FIELDS, "canonicalization"),
    ):
        checks.check(set(constant) == set(expected), f"integrity:whitelist_{name}", sorted(constant))
    for public, evaluator in ctx.records:
        pair_id = public["pair_id"]
        checks.check(set(public) == set(EXPECTED_PUBLIC_RECORD_FIELDS), "integrity:public_fields", pair_id)
        observation = public["observation"]
        checks.check(set(observation) == set(EXPECTED_OBSERVATION_FIELDS), "integrity:observation_fields", pair_id)
        checks.check(set(observation["query"]) == set(EXPECTED_QUERY_FIELDS), "integrity:query_fields", pair_id)
        checks.check(set(observation["query"]["params"]) == set(EXPECTED_QUERY_PARAM_FIELDS), "integrity:query_param_fields", pair_id)
        checks.check(set(observation["visibility"]) == set(EXPECTED_VISIBILITY_FIELDS), "integrity:visibility_fields", pair_id)
        checks.check(set(observation["noise"]) == set(EXPECTED_NOISE_FIELDS), "integrity:noise_fields", pair_id)
        checks.check(set(observation["canonicalization"]) == set(EXPECTED_CANONICALIZATION_FIELDS), "integrity:canonicalization_fields", pair_id)
        forbidden = _forbidden_paths(public)
        checks.check(not forbidden, "leak:forbidden_public_keys", {"pair_id": pair_id, "paths": forbidden})
        checks.check(bool(EXPECTED_PAIR_ID.fullmatch(public["pair_id"])), "integrity:pair_id_format", pair_id)
        checks.check(isinstance(public["split_group"], str) and bool(public["split_group"]), "integrity:split_group", pair_id)
        checks.check(bool(EXPECTED_QUERY_ID.fullmatch(observation["query"]["query_id"])), "integrity:query_id_format", pair_id)
        public_ids = []
        for index, item in enumerate(observation["observed_objects"]):
            checks.check(set(item) == set(EXPECTED_OBSERVED_OBJECT_FIELDS), "integrity:object_fields", (pair_id, index))
            checks.check(item["type"] == "DISK", "integrity:object_type", (pair_id, index))
            checks.check(bool(EXPECTED_PUBLIC_ID.fullmatch(item["public_id"])), "integrity:public_id_format", (pair_id, index))
            public_ids.append(item["public_id"])
        checks.check(public_ids == sorted(public_ids), "integrity:public_id_sorted", pair_id)
        checks.check(len(public_ids) == len(set(public_ids)), "integrity:public_id_unique", pair_id)
        checks.check(canonical_text(public) == canonical_json(public), "integrity:local_canonical_matches_framework", pair_id)
        checks.check(public["pair_id"] == pair_observation_id(public["observation"]), "integrity:pair_id_hash", pair_id)
        checks.check(observation["query"]["query_id"] == query_geometry_id(observation["query"]), "integrity:query_id_hash", pair_id)
        checks.check(evaluator["pair_id"] == public["pair_id"], "integrity:pair_identity", pair_id)
    evidence = {
        "pair_count": ctx.pair_count,
        "public_record_fields": sorted(EXPECTED_PUBLIC_RECORD_FIELDS),
        "observation_fields": sorted(EXPECTED_OBSERVATION_FIELDS),
        "forbidden_key_hits": 0 if not _forbidden_paths(ctx.public_rows) else _forbidden_paths(ctx.public_rows),
    }
    return _finish(checks, "public schema and identity fields match the closed whitelist", evidence)


def _forbidden_paths(document: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            child = f"{path}.{key}"
            if key in FORBIDDEN_PUBLIC_KEYS or key.startswith(("truth_", "hidden_", "oracle_")):
                hits.append(child)
            hits.extend(_forbidden_paths(value, child))
    elif isinstance(document, (list, tuple)):
        for index, value in enumerate(document):
            hits.extend(_forbidden_paths(value, f"{path}[{index}]"))
    return hits


def probe_token_scan(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    hits: list[dict[str, Any]] = []
    for index, public in enumerate(ctx.public_rows):
        line_number = ctx.public_line_numbers[index]
        values = "\n".join(_string_values(public))
        lowered = values.lower()
        for token in MECHANISM_TOKENS + WORLD_FAMILY_TOKENS + SCENE_ID_TOKENS:
            checks.check(token not in values, "leak:token_scan", {"line": line_number, "token": token})
            if token in values:
                hits.append({"line": line_number, "token": token})
        for token in TRUTH_TOKENS:
            found = token in values if token.isupper() else token in lowered
            checks.check(not found, "leak:token_scan", {"line": line_number, "token": token})
            if found:
                hits.append({"line": line_number, "token": token})
        for token in OBJECT_ID_TOKENS:
            checks.check(token not in lowered, "leak:token_scan", {"line": line_number, "token": token})
            if token in lowered:
                hits.append({"line": line_number, "token": token})
    evidence = {
        "line_count": len(ctx.public_rows),
        "tokens_scanned": len(MECHANISM_TOKENS) + len(WORLD_FAMILY_TOKENS) + len(SCENE_ID_TOKENS) + len(TRUTH_TOKENS) + len(OBJECT_ID_TOKENS),
        "hit_count": len(hits),
        "hits": _sample(hits),
        "scan_scope": "PUBLIC_STRING_VALUES_ONLY_NOT_FIELD_NAMES",
    }
    observed = "no mechanism, world-family, truth-id, or hidden-object token in public values" if not hits else f"{len(hits)} public token hits"
    return _finish(checks, observed, evidence)


def probe_hidden_truth_mutation(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {
        "hidden_mutations": 0,
        "hidden_removals": 0,
        "hidden_additions": 0,
        "visible_mutations": 0,
        "visible_changed": 0,
        "query_mutations": 0,
        "query_changed": 0,
    }
    samples: list[dict[str, Any]] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        visible = evaluator["visible_object_ids"]
        original = public["observation"]
        projected = _expected_public(evaluator, evaluator["scenes"][0])
        checks.check(canonical_text(projected) == canonical_text(original), "integrity:base_projection", {"line": line_number, "pair_id": public["pair_id"]})
        for scene_index, scene in enumerate(evaluator["scenes"]):
            hidden = _hidden_objects(scene, visible)
            for obj in hidden:
                for kind in ("center", "radius", "normal"):
                    mutated = _scene_with_object(scene, obj["id"], kind)
                    mutated_projection = _expected_public(evaluator, mutated)
                    same = canonical_text(mutated_projection) == canonical_text(original)
                    counters["hidden_mutations"] += 1
                    checks.check(same, "leak:hidden_mutation_invariant", {"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"], "mutation": kind})
                    if not same and len(samples) < 5:
                        samples.append({"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"], "mutation": kind, "public_changed": True})
                removed = _scene_with_object(scene, obj["id"], "remove")
                removed_projection = _expected_public(evaluator, removed)
                same = canonical_text(removed_projection) == canonical_text(original)
                counters["hidden_removals"] += 1
                checks.check(same, "leak:hidden_removal_invariant", {"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"]})
            existing_ids = {obj["id"] for obj in scene["objects"]}
            added = _scene_with_added_hidden(scene, existing_ids)
            added_projection = _expected_public(evaluator, added)
            same = canonical_text(added_projection) == canonical_text(original)
            counters["hidden_additions"] += 1
            checks.check(same, "leak:hidden_addition_invariant", {"line": line_number, "pair_id": public["pair_id"], "scene": scene_index})
        first_visible = visible[0]
        for kind in ("center", "radius", "normal"):
            mutated = _scene_with_object(evaluator["scenes"][0], first_visible, kind)
            mutated_projection = _expected_public(evaluator, mutated)
            changed = canonical_text(mutated_projection) != canonical_text(original)
            counters["visible_mutations"] += 1
            counters["visible_changed"] += 1 if changed else 0
            checks.check(changed, "integrity:visible_mutation_changes_public", {"line": line_number, "pair_id": public["pair_id"], "object": first_visible, "mutation": kind})
        for kind in ("A", "B", "r_attach"):
            mutated_query = _query_with_mutation(evaluator["query"], kind)
            mutated_projection = _expected_public(evaluator, evaluator["scenes"][0], mutated_query)
            changed = canonical_text(mutated_projection) != canonical_text(original)
            counters["query_mutations"] += 1
            counters["query_changed"] += 1 if changed else 0
            checks.check(changed, "integrity:query_mutation_changes_public", {"line": line_number, "pair_id": public["pair_id"], "mutation": kind})
    evidence = {**counters, "non_vacuous": counters["visible_mutations"] > 0 and counters["visible_changed"] == counters["visible_mutations"] and counters["query_mutations"] > 0 and counters["query_changed"] == counters["query_mutations"], "samples": samples}
    return _finish(checks, "hidden truth mutations never change public observation; visible/query mutations do", evidence)


def probe_hidden_geometry_not_in_public(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"hidden_objects": 0, "hidden_geometry_matches": 0, "public_count_mismatches": 0, "pair_projection_mismatches": 0}
    samples: list[dict[str, Any]] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        visible = evaluator["visible_object_ids"]
        transform = _row_transform(evaluator)
        declared_proxies = list(transform.get("unregistered_observed_objects", [])) if transform is not None else []
        public_objects = public["observation"]["observed_objects"]
        public_geoms = {canonical_text(_geometry_of(obj)) for obj in public_objects}
        proxy_centers = {tuple(p["center"]) for p in declared_proxies}
        proxy_geoms = {canonical_text(_geometry_of(p)) for p in declared_proxies}
        expected_count = len(visible) + len(declared_proxies)
        checks.check(len(public_objects) == expected_count, "leak:public_object_count", {"line": line_number, "pair_id": public["pair_id"], "public": len(public_objects), "expected": expected_count})
        if len(public_objects) != expected_count:
            counters["public_count_mismatches"] += 1
        first_projection = _expected_public(evaluator, evaluator["scenes"][0])
        second_projection = _expected_public(evaluator, evaluator["scenes"][1])
        pair_same = canonical_text(first_projection) == canonical_text(second_projection)
        checks.check(pair_same, "integrity:pair_projection_equal", {"line": line_number, "pair_id": public["pair_id"]})
        if not pair_same:
            counters["pair_projection_mismatches"] += 1
        for scene_index, scene in enumerate(evaluator["scenes"]):
            for obj in _hidden_objects(scene, visible):
                counters["hidden_objects"] += 1
                geometry = _geometry_of(obj)
                matched = canonical_text(geometry) in public_geoms and canonical_text(geometry) not in proxy_geoms
                checks.check(not matched, "leak:hidden_geometry_in_public", {"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"]})
                if matched:
                    counters["hidden_geometry_matches"] += 1
                    if len(samples) < 5:
                        samples.append({"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"]})
                # E6D 分支：OOD 代理由 public query+scale 构造，其中心可能与隐藏对象
                # 中心巧合（如 D2 桥代理落在 HIDDEN_BRIDGE 桥盘中心）；代理是公开
                # 声明对象，只有落在"非代理 public 对象"上的中心命中才算泄漏。
                center_hit = tuple(obj["center"]) in (public_centers_set(public_objects) - proxy_centers)
                checks.check(not center_hit, "leak:hidden_center_in_public", {"line": line_number, "pair_id": public["pair_id"], "scene": scene_index, "object": obj["id"]})
    evidence = {**counters, "samples": samples}
    return _finish(checks, "hidden object geometry is absent from public observed objects", evidence)


def public_centers_set(objects: Sequence[Mapping[str, Any]]) -> set[tuple]:
    return {tuple(obj["center"]) for obj in objects}


def probe_public_id_bypass(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"cases": 0, "whitelist_accepted": 0, "end_to_end_rejected": 0}
    samples: list[dict[str, Any]] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        pair_id = public["pair_id"]
        cases = []
        forged_id = copy.deepcopy(public)
        forged_id["observation"]["observed_objects"][0]["public_id"] = "p" + "0" * 16
        cases.append(("forged_public_id", forged_id))
        forged_geometry = copy.deepcopy(public)
        forged_geometry["observation"]["observed_objects"][0]["center"] = [
            forged_geometry["observation"]["observed_objects"][0]["center"][0] + 0.25,
            forged_geometry["observation"]["observed_objects"][0]["center"][1],
            forged_geometry["observation"]["observed_objects"][0]["center"][2],
        ]
        cases.append(("geometry_with_stale_public_id", forged_geometry))
        duplicate = copy.deepcopy(public)
        if len(duplicate["observation"]["observed_objects"]) < 2:
            raise ContractError("public_id bypass probe needs at least two observed objects")
        duplicate["observation"]["observed_objects"][1]["public_id"] = duplicate["observation"]["observed_objects"][0]["public_id"]
        cases.append(("duplicate_public_id", duplicate))
        dropped = copy.deepcopy(public)
        dropped["observation"]["observed_objects"].pop(0)
        cases.append(("dropped_object", dropped))
        for name, tampered in cases:
            counters["cases"] += 1
            whitelist_ok, whitelist_error = _safe_assert_public(tampered)
            end_ok, end_result = _safe_validate(tampered, evaluator)
            counters["whitelist_accepted"] += 1 if whitelist_ok else 0
            counters["end_to_end_rejected"] += 1 if not end_ok else 0
            checks.check(not end_ok, "leak:public_id_bypass", {"line": line_number, "pair_id": pair_id, "case": name, "whitelist_ok": whitelist_ok, "end_ok": end_ok, "whitelist_error": whitelist_error, "end_error": end_result.get("errors") if isinstance(end_result, Mapping) else end_result})
            if len(samples) < 5:
                samples.append({"line": line_number, "pair_id": pair_id, "case": name, "whitelist_ok": whitelist_ok, "end_ok": end_ok})
    evidence = {**counters, "samples": samples}
    return _finish(checks, "public_id forgery either fails the whitelist or fails end-to-end replay", evidence)


def probe_pair_id_bypass(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"tamper_cases": 0, "end_to_end_rejected": 0, "hash_matches": 0}
    samples: list[dict[str, Any]] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        pair_id = public["pair_id"]
        checks.check(public["pair_id"] == pair_observation_id(public["observation"]), "integrity:pair_id_is_observation_hash", {"line": line_number, "pair_id": pair_id})
        checks.check(evaluator["pair_id"] == public["pair_id"], "integrity:pair_identity", {"line": line_number, "pair_id": pair_id})
        counters["hash_matches"] += 1
        tampered_public = copy.deepcopy(public)
        tampered_public["pair_id"] = "oe_" + "0" * 16
        end_ok, end_result = _safe_validate(tampered_public, evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:pair_id_tamper", {"line": line_number, "pair_id": pair_id, "case": "replace_pair_id", "end_ok": end_ok})
        swapped = copy.deepcopy(public)
        swapped["pair_id"] = ctx.public_rows[(index + 1) % ctx.pair_count]["pair_id"]
        end_ok, end_result = _safe_validate(swapped, evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:pair_id_tamper", {"line": line_number, "pair_id": pair_id, "case": "swap_pair_id", "end_ok": end_ok})
        tampered_evaluator = copy.deepcopy(evaluator)
        tampered_evaluator["pair_id"] = "oe_" + "1" * 16
        end_ok, end_result = _safe_validate(public, tampered_evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:pair_id_tamper", {"line": line_number, "pair_id": pair_id, "case": "evaluator_pair_id", "end_ok": end_ok})
        if len(samples) < 5:
            samples.append({"line": line_number, "pair_id": pair_id, "hash_match": True})
    evidence = {**counters, "samples": samples}
    return _finish(checks, "pair_id is the public observation hash and every tamper is rejected end-to-end", evidence)


def probe_query_id_bypass(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"tamper_cases": 0, "end_to_end_rejected": 0, "hash_matches": 0}
    samples: list[dict[str, Any]] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        pair_id = public["pair_id"]
        query = public["observation"]["query"]
        expected = query_geometry_id(query)
        checks.check(query["query_id"] == expected, "integrity:query_id_is_query_hash", {"line": line_number, "pair_id": pair_id})
        checks.check(evaluator["query"]["query_id"] == expected, "integrity:evaluator_query_id_matches", {"line": line_number, "pair_id": pair_id})
        counters["hash_matches"] += 1
        tampered_id = copy.deepcopy(public)
        tampered_id["observation"]["query"]["query_id"] = "q_" + "0" * 16
        end_ok, _ = _safe_validate(tampered_id, evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:query_id_tamper", {"line": line_number, "pair_id": pair_id, "case": "replace_query_id", "end_ok": end_ok})
        tampered_geometry = copy.deepcopy(public)
        tampered_geometry["observation"]["query"]["params"]["A"] = [
            tampered_geometry["observation"]["query"]["params"]["A"][0] + 0.25,
            tampered_geometry["observation"]["query"]["params"]["A"][1],
            tampered_geometry["observation"]["query"]["params"]["A"][2],
        ]
        end_ok, _ = _safe_validate(tampered_geometry, evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:query_id_tamper", {"line": line_number, "pair_id": pair_id, "case": "mutate_query_keep_id", "end_ok": end_ok})
        if len(samples) < 5:
            samples.append({"line": line_number, "pair_id": pair_id, "query_id": expected})
    evidence = {
        **counters,
        "samples": samples,
        "note": "query_id is a pure hash of the public query document; cross-truth invariance is exercised by P10 variant-truth fingerprints",
    }
    return _finish(checks, "query_id hashes only public query geometry and every tamper is rejected", evidence)


def probe_split_group_bypass(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"tamper_cases": 0, "end_to_end_rejected": 0, "split_group_values": 0}
    groups = [public["split_group"] for public in ctx.public_rows]
    expected = [bool(evaluator["expected_query_change"]) for _, evaluator in ctx.records]
    mutual_information = _mutual_information(groups, expected)
    checks.check(len(groups) == len(expected), "integrity:split_group_length", len(groups))
    checks.check(mutual_information <= 1e-12 or len(set(groups)) == 1, "leak:split_group_predicts_scope", mutual_information)
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        pair_id = public["pair_id"]
        split = public["split_group"]
        checks.check(isinstance(split, str) and bool(split), "integrity:split_group_nonempty", {"line": line_number, "pair_id": pair_id})
        checks.check(not any(token.lower() in split.lower() for token in MECHANISM_TOKENS + TRUTH_TOKENS + OBJECT_ID_TOKENS), "leak:split_group_token", {"line": line_number, "pair_id": pair_id, "split_group": split})
        checks.check(split == evaluator["split_group"], "integrity:split_group_identity", {"line": line_number, "pair_id": pair_id})
        tampered = copy.deepcopy(public)
        tampered["split_group"] = split + "_TAMPER"
        end_ok, _ = _safe_validate(tampered, evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:split_group_tamper", {"line": line_number, "pair_id": pair_id, "end_ok": end_ok})
        tampered_evaluator = copy.deepcopy(evaluator)
        tampered_evaluator["split_group"] = split + "_TAMPER"
        end_ok, _ = _safe_validate(public, tampered_evaluator)
        counters["tamper_cases"] += 1
        counters["end_to_end_rejected"] += 1 if not end_ok else 0
        checks.check(not end_ok, "leak:split_group_tamper", {"line": line_number, "pair_id": pair_id, "split_group": split})
    counters["split_group_values"] = len(set(groups))
    evidence = {**counters, "mutual_information_bits": mutual_information, "values": sorted(set(groups))}
    return _finish(checks, "split_group is constant across this pilot and tampering is rejected", evidence)


def _negative_zero_paths(document: Any, path: str = "$") -> list[str]:
    """精确负零检查：只在数值本身是 -0.0 时命中（子串法会把 -0.05 误报）。"""
    hits: list[str] = []
    if isinstance(document, Mapping):
        for key, item in document.items():
            hits.extend(_negative_zero_paths(item, f"{path}.{key}"))
    elif isinstance(document, (list, tuple)):
        for index, item in enumerate(document):
            hits.extend(_negative_zero_paths(item, f"{path}[{index}]"))
    elif isinstance(document, float) and document == 0.0 and math.copysign(1.0, document) < 0.0:
        hits.append(path)
    return hits


def probe_noise_and_canonicalization(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {"noise_constant_rows": 0, "canonical_constant_rows": 0, "geometry_exact_matches": 0, "order_invariance": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        pair_id = public["pair_id"]
        observation = public["observation"]
        noise = observation["noise"]
        canonicalization = observation["canonicalization"]
        # P22 语义分支（E6D）：noise.name 逐行按声明核对；无变换行必须是 NONE。
        expected_noise = _declared_noise_name(evaluator)
        checks.check(noise["name"] == expected_noise, "leak:noise_name", {"line": line_number, "pair_id": pair_id, "value": noise["name"], "expected": expected_noise})
        checks.check(noise["order"] == "PUBLIC_OBSERVATION_ONLY", "leak:noise_order", {"line": line_number, "pair_id": pair_id, "value": noise["order"]})
        checks.check(canonicalization["float_policy"] == "JSON_FLOAT_NO_ROUNDING", "leak:float_policy", {"line": line_number, "pair_id": pair_id})
        checks.check(canonicalization["negative_zero"] == "ZERO", "leak:negative_zero_policy", {"line": line_number, "pair_id": pair_id})
        checks.check(canonicalization["object_order"] == "PUBLIC_ID_ASCENDING", "leak:object_order_policy", {"line": line_number, "pair_id": pair_id})
        counters["noise_constant_rows"] += 1 if noise["name"] == expected_noise and noise["order"] == "PUBLIC_OBSERVATION_ONLY" else 0
        counters["canonical_constant_rows"] += 1 if canonicalization["float_policy"] == "JSON_FLOAT_NO_ROUNDING" and canonicalization["negative_zero"] == "ZERO" and canonicalization["object_order"] == "PUBLIC_ID_ASCENDING" else 0
        public_ids = [obj["public_id"] for obj in observation["observed_objects"]]
        checks.check(public_ids == sorted(public_ids), "integrity:object_order", {"line": line_number, "pair_id": pair_id})
        checks.check(len(public_ids) == len(set(public_ids)), "integrity:object_unique", {"line": line_number, "pair_id": pair_id})
        text = canonical_text(observation)
        negative_zero = _negative_zero_paths(observation)
        checks.check(not negative_zero, "integrity:negative_zero_value", {"line": line_number, "pair_id": pair_id, "paths": negative_zero[:5]})
        checks.check("NaN" not in text and "Infinity" not in text, "integrity:non_finite", {"line": line_number, "pair_id": pair_id})
        # 几何精确匹配（E6D 逐行分支）：期望几何来自 transform-aware 重放，而非裸可见对象。
        expected_observation = _expected_public(evaluator, evaluator["scenes"][0])
        expected_by_id = {obj["public_id"]: obj for obj in expected_observation["observed_objects"]}
        for public_obj in observation["observed_objects"]:
            exact = public_obj["public_id"] in expected_by_id and _geometry_of(public_obj) == _geometry_of(expected_by_id[public_obj["public_id"]])
            counters["geometry_exact_matches"] += 1 if exact else 0
            checks.check(exact, "integrity:float_exact", {"line": line_number, "pair_id": pair_id, "public_id": public_obj["public_id"]})
        reversed_scene = copy.deepcopy(evaluator["scenes"][0])
        reversed_scene["objects"] = list(reversed(reversed_scene["objects"]))
        reversed_projection = _expected_public(evaluator, reversed_scene)
        invariant = canonical_text(reversed_projection) == canonical_text(observation)
        counters["order_invariance"] += 1 if invariant else 0
        checks.check(invariant, "integrity:scene_order_invariance", {"line": line_number, "pair_id": pair_id})
        checks.check(canonical_text(observation["query"]) == canonical_text(evaluator["query"]), "integrity:query_snapshot", {"line": line_number, "pair_id": pair_id})
    evidence = {**counters, "pair_count": ctx.pair_count}
    return _finish(checks, "noise and canonicalization declarations are constant and honored", evidence)


def probe_cache_and_fingerprint(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    counters = {
        "public_mutation_changes_fingerprint": 0,
        "evaluator_mutation_changes_fingerprint": 0,
        "same_public_different_truth_fingerprint_changes": 0,
        "same_public_different_truth_pair_id_collisions": 0,
    }
    records = list(ctx.records)
    fingerprint = dataset_fingerprint(records)
    checks.check(ctx.dataset_sha256 == fingerprint, "integrity:dataset_fingerprint", {"declared": ctx.dataset_sha256, "recomputed": fingerprint})
    for name, path, declared in (
        ("public.jsonl", ctx.public_path, ctx.public_sha256),
        ("evaluator.jsonl", ctx.evaluator_path, ctx.evaluator_sha256),
        ("validation.json", ctx.validation_path, ctx.validation_sha256),
    ):
        entry = ctx.run.get("files", {}).get(name) if isinstance(ctx.run.get("files"), Mapping) else None
        checks.check(isinstance(entry, Mapping) and entry.get("sha256") == declared and entry.get("bytes") == path.stat().st_size, "integrity:file_hash", {"name": name})
    config = ctx.run.get("config")
    config_sha = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest() if isinstance(config, Mapping) else None
    checks.check(ctx.run.get("config_sha256") == config_sha, "integrity:config_sha", ctx.run.get("config_sha256"))
    expected_scope = dataset_scope(records, ALLOWED_CLAIM)
    checks.check(canonical_text(ctx.run.get("scope")) == canonical_text(expected_scope), "integrity:scope_binding", ctx.run.get("scope"))
    checks.check(ctx.validation.get("pass") is True, "integrity:validation_pass", ctx.validation.get("pass"))
    checks.check(ctx.validation.get("dataset_sha256") == fingerprint, "integrity:validation_fingerprint", ctx.validation.get("dataset_sha256"))
    change_count = sum(bool(evaluator["expected_query_change"]) for _, evaluator in records)
    checks.check(ctx.validation.get("pair_count") == len(records), "integrity:validation_pair_count", ctx.validation.get("pair_count"))
    checks.check(ctx.validation.get("query_change_count") == change_count, "integrity:validation_change_count", ctx.validation.get("query_change_count"))
    checks.check(ctx.validation.get("query_invariant_control_count") == len(records) - change_count, "integrity:validation_control_count", ctx.validation.get("query_invariant_control_count"))
    checks.check(ctx.validation.get("allowed_claim") == ALLOWED_CLAIM, "integrity:validation_claim", ctx.validation.get("allowed_claim"))
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        mutated_public = copy.deepcopy(public)
        mutated_public["observation"]["observed_objects"][0]["center"][0] += 0.25
        public_variant_records = [
            (mutated_public if p["pair_id"] == public["pair_id"] else p, e)
            for p, e in ctx.records
        ]
        public_fingerprint = dataset_fingerprint(public_variant_records)
        public_changed = public_fingerprint != fingerprint
        counters["public_mutation_changes_fingerprint"] += 1 if public_changed else 0
        checks.check(public_changed, "integrity:public_mutation_changes_fingerprint", {"line": line_number, "pair_id": public["pair_id"]})
        mutated_evaluator = copy.deepcopy(evaluator)
        mutated_evaluator["reference_posterior"] = 0.123456789
        evaluator_fingerprint = dataset_fingerprint([(public, mutated_evaluator) if p["pair_id"] == public["pair_id"] else (p, e) for p, e in ctx.records])
        evaluator_changed = evaluator_fingerprint != fingerprint
        counters["evaluator_mutation_changes_fingerprint"] += 1 if evaluator_changed else 0
        checks.check(evaluator_changed, "leak:evaluator_mutation_changes_fingerprint", {"line": line_number, "pair_id": public["pair_id"]})
        hidden_mutated = copy.deepcopy(evaluator)
        hidden_found = False
        for scene in hidden_mutated["scenes"]:
            for obj in scene["objects"]:
                if obj["id"] not in hidden_mutated["visible_object_ids"]:
                    obj["center"][0] += 0.137
                    hidden_found = True
                    break
            if hidden_found:
                break
        if not hidden_found:
            # E6D/v6 分支：全暴露行（VISIBLE_ONLY 可识别控制）没有隐藏结构，
            # "同 public 不同真值"在构造上不可能；登记 N/A，不落入可见对象变异。
            counters["no_hidden_structure_rows"] = counters.get("no_hidden_structure_rows", 0) + 1
            continue
        variant_scenes = [Scene.from_dict(scene) for scene in hidden_mutated["scenes"]]
        variant_query = TwoPointQuery.from_dict(hidden_mutated["query"])
        variant_posterior, variant_labels = reference_posterior(variant_scenes, hidden_mutated["prior_weights"], variant_query)
        hidden_mutated["query_labels"] = list(variant_labels)
        hidden_mutated["reference_posterior"] = variant_posterior
        hidden_mutated["expected_query_change"] = len(set(variant_labels)) > 1
        variant_valid, variant_result = _safe_validate(public, hidden_mutated)
        checks.check(variant_valid, "integrity:variant_truth_is_valid_pair", {"line": line_number, "pair_id": public["pair_id"], "errors": variant_result.get("errors") if isinstance(variant_result, Mapping) else variant_result})
        variant_fingerprint = dataset_fingerprint([(public, hidden_mutated) if p["pair_id"] == public["pair_id"] else (p, e) for p, e in ctx.records])
        same_pair_id = hidden_mutated["pair_id"] == public["pair_id"]
        variant_changed = variant_fingerprint != fingerprint
        counters["same_public_different_truth_pair_id_collisions"] += 1 if same_pair_id else 0
        counters["same_public_different_truth_fingerprint_changes"] += 1 if variant_changed else 0
        checks.check(same_pair_id, "integrity:variant_pair_id_same", {"line": line_number, "pair_id": public["pair_id"]})
        checks.check(variant_changed, "leak:variant_truth_changes_fingerprint", {"line": line_number, "pair_id": public["pair_id"]})
        transform = _row_transform(evaluator)
        if transform is not None:
            rung_mutated = copy.deepcopy(evaluator)
            rung_mutated["observation_transform"]["rung"] = "TAMPERED"
            rung_fingerprint = dataset_fingerprint([
                (public, rung_mutated) if p["pair_id"] == public["pair_id"] else (p, e)
                for p, e in ctx.records
            ])
            rung_changed = rung_fingerprint != fingerprint
            counters["transform_rung_mutation_changes_fingerprint"] = counters.get("transform_rung_mutation_changes_fingerprint", 0) + (1 if rung_changed else 0)
            checks.check(rung_changed, "leak:transform_rung_changes_fingerprint", {"line": line_number, "pair_id": public["pair_id"]})
    prediction_run_path = ctx.data_dir / "baselines" / "run.json"
    if prediction_run_path.is_file():
        prediction_run = read_json(prediction_run_path)
        checks.check(prediction_run.get("dataset_sha256") == fingerprint, "leak:prediction_run_dataset_binding", prediction_run.get("dataset_sha256"))
    notes = [
        "no cache implementation is exercised by this matrix; dataset_fingerprint includes both public and evaluator and is the only dataset-level key",
        "a future cache keyed on pair_id alone would collide across hidden-truth variants; pair_id collisions measured here are intended equivalence, not cache safety",
    ]
    evidence = {**counters, "dataset_sha256": fingerprint, "cache_notes": notes}
    return _finish(checks, "dataset fingerprint binds public and evaluator; pair_id alone does not", evidence, note=True)


def probe_physical_separation(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    same_directory = ctx.public_path.parent.resolve() == ctx.evaluator_path.parent.resolve()
    config = ctx.run.get("config") if isinstance(ctx.run.get("config"), Mapping) else {}
    exposed = {
        "bridge_probability": config.get("bridge_probability"),
        "mechanisms": config.get("mechanisms"),
        "generator": config.get("generator"),
        "site_id": config.get("site_id"),
        "split_group": config.get("split_group"),
        "constant_baseline_prior": config.get("constant_baseline_prior"),
    }
    public_only_ok = 0
    for public, _ in ctx.records:
        try:
            value = zero_training_geometry_prediction(public)
            public_only_ok += 1 if value in (0.0, 1.0) else 0
        except (ContractError, KeyError, TypeError, ValueError):
            public_only_ok += 0
    checks.check(public_only_ok == ctx.pair_count, "leak:public_only_baseline_needs_no_evaluator", {"ok": public_only_ok, "pair_count": ctx.pair_count})
    public_geometry_ok = 0
    for public, _ in ctx.records:
        try:
            scene, query = public_visible_geometry(public)
            public_geometry_ok += 1 if scene is not None and query is not None else 0
        except (ContractError, KeyError, TypeError, ValueError):
            public_geometry_ok += 0
    checks.check(public_geometry_ok == ctx.pair_count, "integrity:public_visible_geometry", {"ok": public_geometry_ok, "pair_count": ctx.pair_count})
    validation_has_pair_truth = _contains_pair_truth(ctx.validation)
    checks.check(not validation_has_pair_truth, "leak:validation_contains_pair_truth", validation_has_pair_truth)
    notes = [
        "public.jsonl and evaluator.jsonl are siblings in the same data directory; physical separation is not enforced by the filesystem",
        "run.json exposes generation parameters (prior/mechanism/site/split) that are evaluator-adjacent; a model given the data directory can read them",
        "the declared public interface is public.jsonl only; public-only baseline and public_visible_geometry run without evaluator in this matrix",
    ]
    evidence = {
        "same_directory": same_directory,
        "public_path": str(ctx.public_path),
        "evaluator_path": str(ctx.evaluator_path),
        "run_json_exposed_generation_parameters": exposed,
        "public_only_baseline_ok": public_only_ok,
        "public_geometry_ok": public_geometry_ok,
        "validation_contains_pair_truth": validation_has_pair_truth,
        "notes": notes,
    }
    return _finish(checks, "public-only functions need no evaluator, but files are co-located and run.json exposes generation parameters", evidence, note=True)


def _contains_pair_truth(document: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            child = f"{path}.{key}"
            if key in {"pair_id", "query_labels", "reference_posterior", "prior_weights", "scenes", "expected_query_change", "mechanism", "world_family"}:
                hits.append(child)
            hits.extend(_contains_pair_truth(value, child))
    elif isinstance(document, (list, tuple)):
        for index, value in enumerate(document):
            hits.extend(_contains_pair_truth(value, f"{path}[{index}]"))
    return hits


def probe_prediction_fields(ctx: MatrixContext) -> ProbeOutcome:
    predictions_path = ctx.data_dir / "baselines" / "predictions.jsonl"
    if not predictions_path.is_file():
        checks = CheckLog()
        return _finish(checks, "no prediction artifact to inspect", {"path": str(predictions_path)}, blocked=True)
    checks = CheckLog()
    rows, lines = _read_jsonl_with_lines(predictions_path)
    public_by_id = ctx.public_by_id
    evaluator_by_id = ctx.evaluator_by_id
    counters = {"rows": 0, "evaluator_field_hits": 0, "public_only_values_recomputed": 0, "diagnostic_rows": 0}
    diagnostic_note = False
    prediction_run_path = ctx.data_dir / "baselines" / "run.json"
    prediction_run = read_json(prediction_run_path) if prediction_run_path.is_file() else {}
    declared_prior = float(prediction_run.get("constant_prior", 0.5))
    declared_roles = prediction_run.get("method_roles")
    for index, row in enumerate(rows):
        line_number = lines[index]
        counters["rows"] += 1
        role_hits = set(row) & V6_ROLE_PREDICTION_FIELDS
        base_fields = set(row) - V6_ROLE_PREDICTION_FIELDS
        fields_ok = base_fields == set(PREDICTION_FIELDS) and (
            not role_hits or role_hits == set(V6_ROLE_PREDICTION_FIELDS)
        )
        checks.check(fields_ok, "integrity:prediction_fields", {"line": line_number, "fields": sorted(row)})
        if role_hits and isinstance(declared_roles, Mapping):
            # R-08：角色旗标必须与 run.json 的冻结映射一致（角色驱动，非方法名黑名单）
            expected_role = declared_roles.get(row.get("method"))
            if isinstance(expected_role, Mapping):
                roles_consistent = (
                    row.get("metric_role") == expected_role.get("metric_role")
                    and row.get("deployable_predictor") == expected_role.get("deployable_predictor")
                    and row.get("uses_held_out_target_for_selection")
                    == expected_role.get("uses_held_out_target_for_selection")
                )
                checks.check(roles_consistent, "integrity:prediction_role_flags", {"line": line_number, "method": row.get("method")})
        hits = sorted(key for key in row if key in EVALUATOR_ONLY_PREDICTION_KEYS or key.startswith(("truth_", "hidden_", "oracle_")))
        counters["evaluator_field_hits"] += len(hits)
        checks.check(not hits, "leak:evaluator_field_in_prediction", {"line": line_number, "hits": hits})
        pair_id = row.get("pair_id")
        checks.check(pair_id in public_by_id, "integrity:prediction_pair_known", {"line": line_number, "pair_id": pair_id})
        method = row.get("method")
        checks.check(method in METHOD_CLAIMS, "integrity:prediction_method", {"line": line_number, "method": method})
        if pair_id in public_by_id and method in METHOD_CLAIMS:
            checks.check(row.get("allowed_claim") == METHOD_CLAIMS[method], "integrity:prediction_claim", {"line": line_number, "pair_id": pair_id, "method": method})
            checks.check(row.get("status") == "PREDICTED", "integrity:prediction_status", {"line": line_number, "pair_id": pair_id, "method": method})
            checks.check(row.get("coverage") == 1.0, "integrity:prediction_coverage", {"line": line_number, "pair_id": pair_id, "method": method})
            if method == "CONSTANT_PRIOR":
                expected_value = declared_prior
                checks.check(abs(float(row.get("p_connected")) - expected_value) <= 1e-12, "integrity:prediction_value_recomputed", {"line": line_number, "pair_id": pair_id, "method": method})
                counters["public_only_values_recomputed"] += 1
            elif method == "ZERO_TRAINING_GEOMETRY":
                expected_value = zero_training_geometry_prediction(public_by_id[pair_id])
                checks.check(abs(float(row.get("p_connected")) - expected_value) <= 1e-12, "integrity:prediction_value_recomputed", {"line": line_number, "pair_id": pair_id, "method": method})
                counters["public_only_values_recomputed"] += 1
            elif method == "REFERENCE_POSTERIOR_DIAGNOSTIC":
                counters["diagnostic_rows"] += 1
                diagnostic_note = True
                expected_value = float(evaluator_by_id[pair_id]["reference_posterior"])
                checks.check(abs(float(row.get("p_connected")) - expected_value) <= 1e-12, "integrity:diagnostic_value", {"line": line_number, "pair_id": pair_id, "method": method})
    if prediction_run:
        checks.check(prediction_run.get("dataset_sha256") == ctx.dataset_sha256, "integrity:prediction_dataset_binding", prediction_run.get("dataset_sha256"))
        checks.check(prediction_run.get("prediction_count") == len(rows), "integrity:prediction_count", prediction_run.get("prediction_count"))
        checks.check(prediction_run.get("predictions_sha256") == file_sha256(predictions_path), "integrity:prediction_file_hash", prediction_run.get("predictions_sha256"))
    notes = []
    if diagnostic_note:
        notes.append("REFERENCE_POSTERIOR_DIAGNOSTIC rows intentionally carry the evaluator reference posterior and are labeled DIAGNOSTIC_ONLY; they are not a model output")
    evidence = {
        **counters,
        "fields": sorted(PREDICTION_FIELDS),
        "evaluator_only_keys": sorted(EVALUATOR_ONLY_PREDICTION_KEYS),
        "diagnostic_note": diagnostic_note,
        "notes": notes,
        "path": str(predictions_path),
    }
    return _finish(checks, "prediction rows contain no evaluator fields; diagnostic rows are explicitly labeled", evidence, note=diagnostic_note)


def _name_token_hits(name: str) -> list[str]:
    lowered = name.lower()
    hits = []
    for token in PACKAGING_TOKENS:
        if re.search(r"(?:^|[_\-.])" + re.escape(token) + r"(?:$|[_\-.])", lowered):
            hits.append(token)
    if re.search(r"oe_[0-9a-f]{16}", lowered):
        hits.append("pair_id")
    return hits


def probe_packaging_metadata(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    hits: list[dict[str, Any]] = []
    names = [ctx.data_dir.name]
    names.extend(sorted(path.name for path in ctx.data_dir.rglob("*") if path.name))
    for name in names:
        name_hits = _name_token_hits(name)
        checks.check(not name_hits, "leak:name_token", {"name": name, "hits": name_hits})
        if name_hits:
            hits.append({"name": name, "hits": name_hits})
    line_table = []
    parity_matches = 0
    families: list[str] = []
    for index, (public, evaluator) in enumerate(ctx.records):
        line_number = ctx.public_line_numbers[index]
        expected_change = bool(evaluator["expected_query_change"])
        parity_match = (index % 2 == 0) == expected_change
        parity_matches += 1 if parity_match else 0
        families.append(str(evaluator.get("world_family")))
        line_table.append({
            "line": line_number,
            "pair_id": public["pair_id"],
            "expected_query_change": expected_change,
            "mechanism": evaluator.get("mechanism"),
            "world_family": evaluator.get("world_family"),
            "public_object_count": len(public["observation"]["observed_objects"]),
            "parity_match": parity_match,
        })
    n = ctx.pair_count
    parity_accuracy = parity_matches / n if n else 0.0
    parity_leak = n >= 8 and max(parity_matches, n - parity_matches) == n
    checks.check(not parity_leak, "leak:line_parity_encodes_scope", {"matches": parity_matches, "n": n, "accuracy": parity_accuracy})
    family_runs = 1 + sum(left != right for left, right in zip(families, families[1:])) if families else 0
    distinct_families = len(set(families))
    mechanism_block_leak = n >= 8 and distinct_families >= 2 and family_runs == distinct_families
    checks.check(not mechanism_block_leak, "leak:line_blocks_encode_world_family", {"runs": family_runs, "distinct": distinct_families, "n": n})
    file_order = [public["pair_id"] for public in ctx.public_rows]
    sorted_order = sorted(file_order)
    loader_order = [public["pair_id"] for public, _ in load_dataset(ctx.data_dir)]
    loader_reorders = loader_order != file_order
    order_policy_keys = []
    for document_name, document in (("run.json", ctx.run), ("validation.json", ctx.validation)):
        if isinstance(document, Mapping):
            for key in document:
                if any(token in key.lower() for token in ("order", "sort", "shuffle", "seed")):
                    order_policy_keys.append(f"{document_name}:{key}")
    scope_from_public = 0
    for public, evaluator in ctx.records:
        predicted_change = zero_training_geometry_prediction(public) == 0.0
        scope_from_public += 1 if predicted_change == bool(evaluator["expected_query_change"]) else 0
    signatures: dict[tuple[Any, ...], set[str]] = {}
    family_signatures: dict[str, set[tuple[Any, ...]]] = {}
    for public, evaluator in ctx.records:
        objects = public["observation"]["observed_objects"]
        signature = (
            len(objects),
            tuple(sorted(
                (
                    round(float(obj["center"][0]), 6),
                    round(float(obj["center"][1]), 6),
                    round(float(obj["center"][2]), 6),
                    round(float(obj["radius"]), 6),
                )
                for obj in objects
            )),
        )
        signatures.setdefault(signature, set()).add(str(evaluator["world_family"]))
        family_signatures.setdefault(str(evaluator["world_family"]), set()).add(signature)
    public_family_identifiable = (
        len(signatures) > 0
        and all(len(values) == 1 for values in signatures.values())
        and sum(len(values) for values in family_signatures.values()) == len(signatures)
    )
    evidence = {
        "name_hits": hits,
        "line_table": line_table,
        "parity_matches": parity_matches,
        "parity_accuracy": parity_accuracy,
        "parity_leak": parity_leak,
        "family_runs": family_runs,
        "distinct_families": distinct_families,
        "mechanism_block_leak": mechanism_block_leak,
        "file_order": file_order,
        "sorted_order": sorted_order,
        "file_order_is_pair_id_sorted": file_order == sorted_order,
        "official_loader_reorders": loader_reorders,
        "declared_order_policy_keys": order_policy_keys,
        "public_scope_redundancy_accuracy": scope_from_public / n if n else 0.0,
        "public_family_identifiable_from_geometry": public_family_identifiable,
    }
    return _finish(checks, "package names are neutral; line order and mechanism blocks are not", evidence)


def probe_manifest_order_contract(ctx: MatrixContext) -> ProbeOutcome:
    checks = CheckLog()
    file_order = [public["pair_id"] for public in ctx.public_rows]
    sorted_order = sorted(file_order)
    loader_order = [public["pair_id"] for public, _ in load_dataset(ctx.data_dir)]
    order_policy_keys = []
    for document_name, document in (("run.json", ctx.run), ("validation.json", ctx.validation)):
        if isinstance(document, Mapping):
            for key in document:
                if any(token in key.lower() for token in ("order", "sort", "shuffle", "seed")):
                    order_policy_keys.append(f"{document_name}:{key}")
    checks.check(loader_order == sorted_order, "integrity:loader_sorted_by_pair_id", {"loader_matches_sorted": loader_order == sorted_order})
    evidence = {
        "declared_order_policy_keys": order_policy_keys,
        "file_order_is_pair_id_sorted": file_order == sorted_order,
        "official_loader_reorders": loader_order != file_order,
    }
    return _finish(checks, "manifest should declare or enforce a public-content-only row order", evidence, note=not order_policy_keys)


def probe_transform_replay(ctx: MatrixContext) -> ProbeOutcome:
    """P16 变换重放：clean 投影 + transform == public，逐字节（仅变换行）。"""
    checks = CheckLog()
    counters = {"transform_rows": 0, "replay_byte_exact": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        transform = _row_transform(evaluator)
        if transform is None:
            continue
        counters["transform_rows"] += 1
        replayed = _expected_public(evaluator, evaluator["scenes"][0])
        exact = canonical_text(replayed) == canonical_text(public["observation"])
        counters["replay_byte_exact"] += 1 if exact else 0
        checks.check(exact, "leak:transform_replay", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    if counters["transform_rows"] == 0:
        return _finish(checks, "no transform rows in this dataset (clean/control dataset)", {**counters, "vacuous": True})
    return _finish(checks, "clean projection plus declared transform reproduces public bytes exactly", counters)


def probe_post_transform_group_equivalence(ctx: MatrixContext) -> ProbeOutcome:
    """P17 扰动后组等价：同组各 scene 变换后 public 哈希相等。"""
    checks = CheckLog()
    counters = {"transform_rows": 0, "groups_equal": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        if _row_transform(evaluator) is None:
            continue
        counters["transform_rows"] += 1
        hashes = {
            sha256_document(_expected_public(evaluator, scene))
            for scene in evaluator["scenes"]
        }
        equal = len(hashes) == 1
        counters["groups_equal"] += 1 if equal else 0
        checks.check(equal, "leak:post_transform_group_equivalence", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    if counters["transform_rows"] == 0:
        return _finish(checks, "no transform rows in this dataset", {**counters, "vacuous": True})
    return _finish(checks, "every scene in a group yields the same transformed public observation", counters)


def probe_truth_invariance(ctx: MatrixContext) -> ProbeOutcome:
    """P18 真值不变：scenes/labels/posterior 从行内重放一致（跨档对 clean 的对账在报告层）。"""
    checks = CheckLog()
    counters = {"rows": 0, "replay_consistent": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        counters["rows"] += 1
        scenes = [Scene.from_dict(scene) for scene in evaluator["scenes"]]
        query = TwoPointQuery.from_dict(evaluator["query"])
        posterior, labels = reference_posterior(scenes, evaluator["prior_weights"], query)
        consistent = (
            list(labels) == list(evaluator["query_labels"])
            and abs(posterior - float(evaluator["reference_posterior"])) <= 1e-12
        )
        counters["replay_consistent"] += 1 if consistent else 0
        checks.check(consistent, "leak:truth_replay_consistency", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    return _finish(checks, "truth fields replay from scenes only; transform never touches them", counters)


def probe_query_invariance(ctx: MatrixContext) -> ProbeOutcome:
    """P19 查询不变：public query 与 evaluator query 逐字节一致；query_id 为几何哈希。"""
    checks = CheckLog()
    counters = {"rows": 0, "query_frozen": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        counters["rows"] += 1
        query = public["observation"]["query"]
        frozen = (
            canonical_text(query) == canonical_text(evaluator["query"])
            and query["query_id"] == query_geometry_id(query)
        )
        counters["query_frozen"] += 1 if frozen else 0
        checks.check(frozen, "leak:query_invariance", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    return _finish(checks, "query bytes and query_id are frozen across all rows and arms", counters)


def probe_control_invariance(ctx: MatrixContext) -> ProbeOutcome:
    """P20 控制不变：控制行无变换、标签跨 scene 不变（clean null，D-07）。

    E6D 变换数据集：控制行必须是 legacy schema v1（干净 null）；
    v6 数据集：全部行本就用 evaluator v2，故只核对无变换/标签/噪声三项。
    """
    checks = CheckLog()
    counters = {"control_rows": 0, "controls_clean": 0}
    dataset_has_transform = any(_row_transform(evaluator) is not None for _, evaluator in ctx.records)
    for index, (public, evaluator) in enumerate(ctx.records):
        if bool(evaluator["expected_query_change"]):
            continue
        counters["control_rows"] += 1
        schema_ok = (not dataset_has_transform) or evaluator.get("schema") != SCHEMA_EVALUATOR_V2
        clean = (
            _row_transform(evaluator) is None
            and schema_ok
            and len(set(evaluator["query_labels"])) == 1
            and public["observation"]["noise"]["name"] == "NONE"
        )
        counters["controls_clean"] += 1 if clean else 0
        checks.check(clean, "leak:control_row_clean_null", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    if counters["control_rows"] == 0:
        return _finish(checks, "no control rows in this dataset", {**counters, "vacuous": True})
    return _finish(checks, "control rows stay clean (no transform, constant labels)", counters)


def probe_ood_no_truth_entries(ctx: MatrixContext) -> ProbeOutcome:
    """P21 OOD 无真值条目：代理只进 public/unregistered，不进 scenes/labels。"""
    checks = CheckLog()
    counters = {"ood_rows": 0, "proxy_clean": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        transform = _row_transform(evaluator)
        if transform is None or transform.get("arm_id") != "OUT_OF_DICTIONARY_OBJECT":
            continue
        counters["ood_rows"] += 1
        proxies = transform.get("unregistered_observed_objects", [])
        proxy = proxies[0] if proxies else None
        public_ids = {obj["public_id"] for obj in public["observation"]["observed_objects"]}
        scene_geometry = {
            canonical_text({k: v for k, v in obj.items() if k != "id"})
            for scene in evaluator["scenes"] for obj in scene["objects"]
        }
        proxy_geometry = canonical_text({k: v for k, v in proxy.items() if k != "public_id"}) if proxy else None
        clean = (
            proxy is not None
            and set(proxy) == {"public_id", "type", "center", "normal", "radius"}
            and proxy["public_id"] in public_ids
            and proxy_geometry not in scene_geometry
            and proxy["public_id"] not in evaluator["visible_object_ids"]
        )
        counters["proxy_clean"] += 1 if clean else 0
        checks.check(clean, "leak:ood_proxy_no_truth_entries", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    if counters["ood_rows"] == 0:
        return _finish(checks, "no OOD rows in this dataset", {**counters, "vacuous": True})
    return _finish(checks, "OOD proxy exists only as a declared public object", counters)


def probe_noise_declaration_matches_arm(ctx: MatrixContext) -> ProbeOutcome:
    """P22 声明匹配：noise.name 逐行与臂/模式一致；clean/control 行 NONE。"""
    checks = CheckLog()
    counters = {"rows": 0, "declarations_match": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        counters["rows"] += 1
        expected = _declared_noise_name(evaluator)
        matches = public["observation"]["noise"]["name"] == expected
        counters["declarations_match"] += 1 if matches else 0
        checks.check(matches, "leak:noise_declaration_matches_arm", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"], "expected": expected})
    return _finish(checks, "noise declarations match the declared arm/mode per row", counters)


def probe_rung_isolation(ctx: MatrixContext) -> ProbeOutcome:
    """P23 档位隔离：行内档位声明一致，且与 run config 声明一致（跨档指纹对账在报告层）。"""
    checks = CheckLog()
    counters = {"transform_rows": 0, "consistent": 0}
    config = ctx.run.get("config") if isinstance(ctx.run.get("config"), Mapping) else {}
    config_transform = config.get("observation_transform")
    for index, (public, evaluator) in enumerate(ctx.records):
        transform = _row_transform(evaluator)
        if transform is None:
            continue
        counters["transform_rows"] += 1
        consistent = (
            (config_transform is None)
            or (
                config_transform.get("arm_id") == transform.get("arm_id")
                and config_transform.get("rung") == transform.get("rung")
            )
        )
        counters["consistent"] += 1 if consistent else 0
        checks.check(consistent, "leak:rung_isolation", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    pair_ids = [public["pair_id"] for public, _ in ctx.records]
    checks.check(len(pair_ids) == len(set(pair_ids)), "leak:pair_id_unique", {"n": len(pair_ids)})
    if counters["transform_rows"] == 0:
        return _finish(checks, "no transform rows; pair ids unique", {**counters, "vacuous": True})
    return _finish(checks, "declared arm/rung is uniform and matches the run config; cross-rung fingerprints are compared at the report layer", counters)


def probe_invalid_rows_not_averaged(ctx: MatrixContext) -> ProbeOutcome:
    """P24 无效不平均（单数据集面）：无无效行混入；跨档均值排除无效档在报告层执行。"""
    checks = CheckLog()
    counters = {"rows": 0, "valid_rows": 0}
    for index, (public, evaluator) in enumerate(ctx.records):
        counters["rows"] += 1
        ok, _ = _safe_validate(public, evaluator)
        counters["valid_rows"] += 1 if ok else 0
        checks.check(ok, "leak:invalid_row_present", {"line": ctx.public_line_numbers[index], "pair_id": public["pair_id"]})
    checks.check(ctx.validation.get("pass") is True, "leak:validation_pass", ctx.validation.get("pass"))
    return _finish(
        checks,
        "no invalid pair enters the dataset (fail-closed build); invalid rungs are excluded from means by the report layer (mean_of_valid)",
        {**counters, "report_layer_guard": "scripts/oe_misspec_report.py::mean_of_valid"},
    )


PROBES: tuple[Probe, ...] = (
    Probe("P01_public_schema_whitelist", "public.jsonl field closure and identity formats", "no extra or forbidden fields; every identity is an opaque hash", probe_schema_whitelist),
    Probe("P02_public_token_scan", "mechanism/world-family/truth-id/hidden-object words in public bytes", "zero token hits", probe_token_scan),
    Probe("P03_hidden_truth_mutation", "hidden object geometry and presence changes", "public observation is invariant; visible/query mutations change it", probe_hidden_truth_mutation),
    Probe("P04_hidden_geometry_not_in_public", "hidden object geometry vs public observed objects", "no hidden geometry or object count leak", probe_hidden_geometry_not_in_public),
    Probe("P05_public_id_bypass", "syntactic public_id forgery and geometry/public_id mismatch", "end-to-end replay rejects every forgery", probe_public_id_bypass),
    Probe("P06_pair_id_bypass", "pair_id tamper, swap, and evaluator/public mismatch", "pair_id equals observation hash and every tamper is rejected", probe_pair_id_bypass),
    Probe("P07_query_id_bypass", "query_id tamper and query geometry/public_id mismatch", "query_id hashes public query geometry only and tamper is rejected", probe_query_id_bypass),
    Probe("P08_split_group_bypass", "split_group value, tamper, and correlation with scope", "constant in this pilot; tamper rejected; zero mutual information", probe_split_group_bypass),
    Probe("P09_noise_and_canonicalization", "noise and canonicalization declarations, float exactness, object order", "constants honored; no negative zero, rounding, or order dependence", probe_noise_and_canonicalization),
    Probe("P10_cache_and_fingerprint", "dataset fingerprint and cache-key binding", "fingerprint binds public and evaluator; pair_id alone does not", probe_cache_and_fingerprint),
    Probe("P11_physical_separation", "public/evaluator co-location and run.json metadata", "public-only functions need no evaluator; physical separation is a design risk", probe_physical_separation),
    Probe("P12_prediction_fields", "prediction rows for evaluator fields and evaluator-derived values", "no evaluator fields; diagnostic rows are labeled DIAGNOSTIC_ONLY", probe_prediction_fields),
    Probe("P13_packaging_metadata", "file/directory names and public row order", "names and order carry no evaluator label", probe_packaging_metadata),
    Probe("P14_manifest_order_contract", "declared or enforced public row-order policy", "manifest declares or enforces a public-content-only order", probe_manifest_order_contract),
    Probe("P16_transform_replay", "clean projection plus declared transform reproduces public bytes", "transform rows replay byte-exactly (E6D)", probe_transform_replay),
    Probe("P17_post_transform_group_equivalence", "transformed public observation within an equivalence group", "all scenes in a group share one transformed public hash (E6D)", probe_post_transform_group_equivalence),
    Probe("P18_truth_invariance", "truth fields under transforms", "scenes/labels/posterior replay consistently; transforms never touch truth (E6D)", probe_truth_invariance),
    Probe("P19_query_invariance", "query bytes and query_id under transforms", "query frozen byte-identically in every row (E6D)", probe_query_invariance),
    Probe("P20_control_invariance", "control rows under transforms", "control rows are clean nulls: no transform, constant labels (E6D D-07)", probe_control_invariance),
    Probe("P21_ood_no_truth_entries", "OOD proxy truth exposure", "proxy exists only as declared public object, never as truth entry (E6D)", probe_ood_no_truth_entries),
    Probe("P22_noise_declaration_matches_arm", "noise declaration versus arm/mode", "noise.name matches declared arm/mode per row; clean rows NONE (E6D)", probe_noise_declaration_matches_arm),
    Probe("P23_rung_isolation", "rung declaration isolation and cache safety", "arm/rung uniform and config-bound; cross-rung fingerprints unique (E6D)", probe_rung_isolation),
    Probe("P24_invalid_rows_not_averaged", "invalid rows must not enter datasets or means", "no invalid pair present; invalid rungs excluded at report layer (E6D)", probe_invalid_rows_not_averaged),
)

CONTENT_PROBES: tuple[Probe, ...] = PROBES[:12] + PROBES[14:]
PACKAGING_PROBES: tuple[Probe, ...] = PROBES[12:14]
ALL_PROBES: tuple[Probe, ...] = PROBES


def _not_run_result(probe: Probe, stopped_after: str) -> ProbeResult:
    return ProbeResult(
        probe_id=probe.probe_id,
        attack_surface=probe.attack_surface,
        expectation=probe.expectation,
        observed=f"not run: stopped after blocker {stopped_after}",
        status=STATUS_NOT_RUN,
        severity=SEVERITY_NOTE,
        category=CATEGORY_COVERAGE,
        enumerated_checks=0,
        evidence={"stopped_after": stopped_after},
    )


def _result_from_outcome(probe: Probe, outcome: ProbeOutcome) -> ProbeResult:
    return ProbeResult(
        probe_id=probe.probe_id,
        attack_surface=probe.attack_surface,
        expectation=probe.expectation,
        observed=outcome.observed,
        status=outcome.status,
        severity=outcome.severity,
        category=outcome.category,
        enumerated_checks=outcome.enumerated_checks,
        evidence=outcome.evidence,
    )


def run_leak_matrix(
    data_dir: str | Path,
    probes: Sequence[Probe] | None = None,
    stop_on_blocker: bool = True,
) -> dict[str, Any]:
    ctx = MatrixContext(data_dir)
    selected = tuple(probes if probes is not None else ALL_PROBES)
    results: list[ProbeResult] = []
    stopped_after: str | None = None
    for probe in selected:
        if stopped_after is not None and stop_on_blocker:
            results.append(_not_run_result(probe, stopped_after))
            continue
        try:
            outcome = probe.func(ctx)
        except Exception as exc:  # boundary: a probe crash is a blocker, not a silent pass
            outcome = ProbeOutcome(
                status=STATUS_FAIL,
                severity=SEVERITY_BLOCKER,
                category=CATEGORY_INTEGRITY,
                observed=f"probe crashed: {type(exc).__name__}: {exc}",
                evidence={"exception": f"{type(exc).__name__}: {exc}"},
                enumerated_checks=0,
            )
        result = _result_from_outcome(probe, outcome)
        results.append(result)
        if result.status == STATUS_FAIL and result.severity == SEVERITY_BLOCKER and stop_on_blocker:
            stopped_after = result.probe_id
    blockers = [result for result in results if result.status == STATUS_FAIL and result.category == CATEGORY_LEAK]
    any_fail = [result for result in results if result.status == STATUS_FAIL]
    any_blocked = [result for result in results if result.status == STATUS_BLOCKED]
    not_run = [result for result in results if result.status == STATUS_NOT_RUN]
    if blockers:
        verdict = VERDICT_LEAK
    elif any_fail or any_blocked or not_run:
        verdict = VERDICT_BLOCKED
    else:
        verdict = VERDICT_NO_LEAK
    counts = {
        "probe_count": len(results),
        "pass_count": sum(result.status == STATUS_PASS for result in results),
        "fail_count": len(any_fail),
        "note_count": sum(result.status == STATUS_NOTE for result in results),
        "blocked_count": len(any_blocked),
        "not_run_count": len(not_run),
        "enumerated_checks": sum(result.enumerated_checks for result in results),
    }
    return {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "data_dir": str(ctx.data_dir),
        "dataset_sha256": ctx.dataset_sha256,
        "public_file_sha256": ctx.public_sha256,
        "evaluator_file_sha256": ctx.evaluator_sha256,
        "run_file_sha256": ctx.run_sha256,
        "validation_file_sha256": ctx.validation_sha256,
        "pair_count": ctx.pair_count,
        "allowed_claim": ctx.run.get("config", {}).get("allowed_claim") if isinstance(ctx.run.get("config"), Mapping) else None,
        "REAL_GENERALIZATION_CLAIM": "NOT_SUPPORTED",
        "FULL_3D_REAL_TRUTH": "NOT_SUPPORTED",
        "verdict": verdict,
        "stopped_after_probe": stopped_after,
        "counts": counts,
        "blockers": [result.to_dict() for result in blockers],
        "not_run": [result.to_dict() for result in not_run],
        "probes": [result.to_dict() for result in results],
    }


def public_hash_evidence(data_dir: str | Path) -> dict[str, Any]:
    ctx = MatrixContext(data_dir)
    fingerprint = dataset_fingerprint(ctx.records)
    pairs = []
    for index, (public, evaluator) in enumerate(ctx.records):
        observation = public["observation"]
        public_ids = [obj["public_id"] for obj in observation["observed_objects"]]
        pairs.append({
            "line": ctx.public_line_numbers[index],
            "pair_id": public["pair_id"],
            "query_id": observation["query"]["query_id"],
            "split_group": public["split_group"],
            "public_object_count": len(public_ids),
            "public_ids": public_ids,
            "observation_sha256_recomputed": sha256_document(observation),
            "observation_sha256_evaluator": evaluator.get("observation_sha256"),
            "observation_sha256_match": sha256_document(observation) == evaluator.get("observation_sha256"),
            "pair_id_recomputed": pair_observation_id(observation),
            "pair_id_match": pair_observation_id(observation) == public["pair_id"],
            "query_id_recomputed": query_geometry_id(observation["query"]),
            "query_id_match": query_geometry_id(observation["query"]) == observation["query"]["query_id"],
            "evaluator_cross_check": {
                "expected_query_change": evaluator.get("expected_query_change"),
                "mechanism": evaluator.get("mechanism"),
                "world_family": evaluator.get("world_family"),
                "truth_level": evaluator.get("truth_level"),
            },
        })
    order_table = []
    for index, (public, evaluator) in enumerate(ctx.records):
        order_table.append({
            "line": ctx.public_line_numbers[index],
            "pair_id": public["pair_id"],
            "expected_query_change": bool(evaluator["expected_query_change"]),
            "world_family": evaluator.get("world_family"),
            "parity_match": (index % 2 == 0) == bool(evaluator["expected_query_change"]),
        })
    return {
        "schema": SCHEMA_PUBLIC_HASHES,
        "created_at_utc": utc_now(),
        "data_dir": str(ctx.data_dir),
        "files": {
            "public.jsonl": {"sha256": ctx.public_sha256, "bytes": ctx.public_path.stat().st_size},
            "evaluator.jsonl": {"sha256": ctx.evaluator_sha256, "bytes": ctx.evaluator_path.stat().st_size},
            "run.json": {"sha256": ctx.run_sha256, "bytes": ctx.run_path.stat().st_size},
            "validation.json": {"sha256": ctx.validation_sha256, "bytes": ctx.validation_path.stat().st_size},
        },
        "dataset_sha256_declared": ctx.dataset_sha256,
        "dataset_sha256_recomputed": fingerprint,
        "dataset_sha256_match": ctx.dataset_sha256 == fingerprint,
        "config_sha256_declared": ctx.run.get("config_sha256"),
        "pair_count": ctx.pair_count,
        "public_pairs": pairs,
        "order_table": order_table,
        "public_record_fields": sorted(EXPECTED_PUBLIC_RECORD_FIELDS),
        "observation_fields": sorted(EXPECTED_OBSERVATION_FIELDS),
        "observed_object_fields": sorted(EXPECTED_OBSERVED_OBJECT_FIELDS),
        "query_fields": sorted(EXPECTED_QUERY_FIELDS),
        "query_param_fields": sorted(EXPECTED_QUERY_PARAM_FIELDS),
        "visibility_fields": sorted(EXPECTED_VISIBILITY_FIELDS),
        "noise_fields": sorted(EXPECTED_NOISE_FIELDS),
        "canonicalization_fields": sorted(EXPECTED_CANONICALIZATION_FIELDS),
    }


def write_json_exclusive(path: str | Path, document: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_text(document) + "\n")
    return target
