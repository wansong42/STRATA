from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA_SCENE = "obs_equiv.scene.v1"
SCHEMA_OBSERVATION = "obs_equiv.observation.v1"
SCHEMA_EVALUATOR = "obs_equiv.evaluator.v1"
SCHEMA_EVALUATOR_V2 = "obs_equiv.evaluator.v2"
SCHEMA_RUN = "obs_equiv.run.v1"
SCHEMA_SCORE = "obs_equiv.score.v1"
GROUP_EXACT = "EXACT_OBSERVATION_EQUIVALENT"
GROUP_EPSILON = "EPSILON_OBSERVATION_EQUIVALENT"
GROUP_REAL_CENSORING = "REAL_CENSORING_HOLDOUT"
TRUTH_EXACT_SYNTHETIC = "EXACT_SYNTHETIC"
QUERY_FAMILY = "TWO_POINT_CONNECTIVITY"
UNIT_NORMAL_TOL = 1e-8

# E6D（D-02）：公共 observation schema 字符串不动，只扩 noise.name 枚举；
# legacy 无变换行的 noise.name 仍必须是 NONE（由变换行专属检查约束，见
# observation_equivalence.transform 分支），此处枚举仅放行四臂声明的取值。
ALLOWED_NOISE_NAMES = frozenset({
    "NONE",
    "HASH_UNIFORM_DISK_GEOMETRY_V1",
    "SENSOR_OFFSET_GLOBAL_V1",
    "SENSOR_OFFSET_DIFFERENTIAL_V1",
})

FORBIDDEN_PUBLIC_KEYS = frozenset({
    "scene_id", "structure_id", "truth_id", "truth_sets", "true_center", "true_normal",
    "hidden_truth", "hidden_objects", "full_network", "ground_truth_network",
    "ground_truth", "oracle_label", "dfn_seed", "frac_id", "frac_ids", "query_labels",
    "reference_posterior", "prior_weights", "visible_object_ids", "expected_query_change",
    # E6D（D-02）：变换元数据只许出现在 evaluator 行，禁入 public 面。
    "observation_transform", "arm_id", "rung", "unregistered_observed_objects",
    "clean_public_sha256", "corrupted_public_sha256",
})
PUBLIC_RECORD_FIELDS = frozenset({"pair_id", "split_group", "observation"})
OBSERVATION_FIELDS = frozenset({
    "schema", "modality", "coordinate_frame", "query", "observed_objects",
    "visibility", "noise", "canonicalization",
})
OBSERVED_OBJECT_FIELDS = frozenset({"public_id", "type", "center", "normal", "radius"})
QUERY_FIELDS = frozenset({"query_id", "family", "params"})
QUERY_PARAM_FIELDS = frozenset({
    "A", "B", "r_attach", "attachment", "intersection", "outside_domain_routing",
})
VISIBILITY_FIELDS = frozenset({"policy", "unexposed_objects"})
NOISE_FIELDS = frozenset({"name", "order"})
CANONICALIZATION_FIELDS = frozenset({"float_policy", "negative_zero", "object_order"})


class ContractError(ValueError):
    pass


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def vec3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError(f"{name} must contain three numbers")
    return (
        finite_number(value[0], f"{name}[0]"),
        finite_number(value[1], f"{name}[1]"),
        finite_number(value[2], f"{name}[2]"),
    )


def _normal(value: Any, name: str) -> tuple[float, float, float]:
    normal = vec3(value, name)
    norm = math.sqrt(sum(component * component for component in normal))
    if abs(norm - 1.0) > UNIT_NORMAL_TOL:
        raise ContractError(f"{name} must be a unit vector")
    return normal


@dataclass(frozen=True)
class Disk:
    object_id: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float

    def __init__(self, object_id: str, center: Sequence[float], normal: Sequence[float], radius: float):
        if not isinstance(object_id, str) or not object_id.strip():
            raise ContractError("object_id must be a non-empty string")
        parsed_radius = finite_number(radius, "radius")
        if parsed_radius <= 0.0:
            raise ContractError("radius must be positive")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "center", vec3(center, "center"))
        object.__setattr__(self, "normal", _normal(normal, "normal"))
        object.__setattr__(self, "radius", parsed_radius)

    def geometry_dict(self) -> dict[str, Any]:
        return {
            "type": "DISK",
            "center": list(self.center),
            "normal": list(self.normal),
            "radius": self.radius,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.object_id, **self.geometry_dict()}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Disk":
        if not isinstance(document, Mapping):
            raise ContractError("disk must be an object")
        if document.get("type") != "DISK":
            raise ContractError("only DISK objects are supported")
        return cls(document.get("id"), document.get("center"), document.get("normal"), document.get("radius"))


@dataclass(frozen=True)
class Scene:
    scene_id: str
    disks: tuple[Disk, ...]

    def __init__(self, scene_id: str, disks: Sequence[Disk]):
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ContractError("scene_id must be a non-empty string")
        parsed = tuple(disks)
        if any(not isinstance(disk, Disk) for disk in parsed):
            raise ContractError("disks must contain Disk instances")
        ids = [disk.object_id for disk in parsed]
        if len(ids) != len(set(ids)):
            raise ContractError("disk ids must be unique")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "disks", parsed)

    def disk(self, object_id: str) -> Disk:
        for disk in self.disks:
            if disk.object_id == object_id:
                return disk
        raise ContractError(f"unknown disk id: {object_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_SCENE,
            "structure_id": self.scene_id,
            "objects": [disk.to_dict() for disk in self.disks],
            "relations": [],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Scene":
        if not isinstance(document, Mapping):
            raise ContractError("scene must be an object")
        if document.get("schema") != SCHEMA_SCENE:
            raise ContractError("unsupported scene schema")
        objects = document.get("objects")
        if not isinstance(objects, list):
            raise ContractError("scene.objects must be a list")
        if "relations" in document and not isinstance(document["relations"], list):
            raise ContractError("scene.relations must be a list")
        return cls(document.get("structure_id"), [Disk.from_dict(item) for item in objects])


@dataclass(frozen=True)
class TwoPointQuery:
    query_id: str
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    attach_radius: float

    def __init__(self, query_id: str, point_a: Sequence[float], point_b: Sequence[float], attach_radius: float):
        if not isinstance(query_id, str) or not query_id.strip():
            raise ContractError("query_id must be a non-empty string")
        radius = finite_number(attach_radius, "attach_radius")
        if radius < 0.0:
            raise ContractError("attach_radius must be non-negative")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "point_a", vec3(point_a, "point_a"))
        object.__setattr__(self, "point_b", vec3(point_b, "point_b"))
        object.__setattr__(self, "attach_radius", radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "family": QUERY_FAMILY,
            "params": {
                "A": list(self.point_a),
                "B": list(self.point_b),
                "r_attach": self.attach_radius,
                "attachment": "POINT_TO_FINITE_DISK_DISTANCE_LE",
                "intersection": "STRICT_NON_TANGENT",
                "outside_domain_routing": "DISABLED",
            },
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "TwoPointQuery":
        if not isinstance(document, Mapping) or document.get("family") != QUERY_FAMILY:
            raise ContractError("unsupported query")
        params = document.get("params")
        if not isinstance(params, Mapping):
            raise ContractError("query.params must be an object")
        if params.get("attachment") != "POINT_TO_FINITE_DISK_DISTANCE_LE":
            raise ContractError("unsupported attachment convention")
        if params.get("intersection") != "STRICT_NON_TANGENT":
            raise ContractError("unsupported intersection convention")
        if params.get("outside_domain_routing") != "DISABLED":
            raise ContractError("unsupported domain convention")
        return cls(document.get("query_id"), params.get("A"), params.get("B"), params.get("r_attach"))


def normalized_weights(weights: Sequence[float], expected_count: int) -> tuple[float, ...]:
    if not isinstance(weights, (list, tuple)) or len(weights) != expected_count or expected_count < 1:
        raise ContractError("prior weights must match the scene count")
    parsed = tuple(finite_number(weight, "prior_weight") for weight in weights)
    if any(weight < 0.0 for weight in parsed):
        raise ContractError("prior weights must be non-negative")
    if not math.isclose(sum(parsed), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("prior weights must sum to one")
    return parsed


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


def canonical_json(document: Any) -> str:
    return json.dumps(_canonical(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(document: Any) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _forbidden_hits(document: Any, path: str = "$") -> list[str]:
    hits = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            child = f"{path}.{key}"
            if key in FORBIDDEN_PUBLIC_KEYS or key.startswith(("truth_", "hidden_", "oracle_")):
                hits.append(child)
            hits.extend(_forbidden_hits(value, child))
    elif isinstance(document, (list, tuple)):
        for index, value in enumerate(document):
            hits.extend(_forbidden_hits(value, f"{path}[{index}]"))
    return hits


def _exact_fields(document: Any, expected: frozenset[str], path: str) -> None:
    if not isinstance(document, Mapping) or set(document) != expected:
        raise ContractError(f"{path} fields must be exactly {sorted(expected)}")


def _assert_observation_contract(observation: Any) -> None:
    _exact_fields(observation, OBSERVATION_FIELDS, "observation")
    if observation["schema"] != SCHEMA_OBSERVATION:
        raise ContractError("unsupported observation schema")
    if observation["modality"] != "SYNTHETIC_VISIBLE_DISKS":
        raise ContractError("unsupported observation modality")
    if observation["coordinate_frame"] != "WORLD":
        raise ContractError("unsupported coordinate frame")
    _exact_fields(observation["query"], QUERY_FIELDS, "observation.query")
    _exact_fields(observation["query"]["params"], QUERY_PARAM_FIELDS, "observation.query.params")
    _exact_fields(observation["visibility"], VISIBILITY_FIELDS, "observation.visibility")
    _exact_fields(observation["noise"], NOISE_FIELDS, "observation.noise")
    _exact_fields(observation["canonicalization"], CANONICALIZATION_FIELDS, "observation.canonicalization")
    TwoPointQuery.from_dict(observation["query"])
    if not isinstance(observation["query"]["query_id"], str) or not re.fullmatch(
        r"q_[0-9a-f]{16}", observation["query"]["query_id"]
    ):
        raise ContractError("query_id must be an opaque query-geometry hash")
    if observation["visibility"]["policy"] != "EXPLICIT_OBJECT_EXPOSURE":
        raise ContractError("unsupported visibility policy")
    if observation["visibility"]["unexposed_objects"] != "OMITTED_NOT_ABSENT":
        raise ContractError("unsupported unexposed-object semantics")
    if observation["noise"]["name"] not in ALLOWED_NOISE_NAMES or observation["noise"]["order"] != "PUBLIC_OBSERVATION_ONLY":
        raise ContractError("unsupported noise contract")
    canonicalization = observation["canonicalization"]
    if (
        canonicalization["float_policy"] != "JSON_FLOAT_NO_ROUNDING"
        or canonicalization["negative_zero"] != "ZERO"
        or canonicalization["object_order"] != "PUBLIC_ID_ASCENDING"
    ):
        raise ContractError("unsupported canonicalization contract")
    objects = observation["observed_objects"]
    if not isinstance(objects, list):
        raise ContractError("observation.observed_objects must be a list")
    for index, item in enumerate(objects):
        _exact_fields(item, OBSERVED_OBJECT_FIELDS, f"observation.observed_objects[{index}]")
        if item["type"] != "DISK":
            raise ContractError("only DISK observations are supported")
        public_id = item["public_id"]
        if not isinstance(public_id, str) or not re.fullmatch(r"p[0-9a-f]{16}", public_id):
            raise ContractError("public_id must be an opaque geometry hash")
        Disk(public_id, item["center"], item["normal"], item["radius"])
    public_ids = [item["public_id"] for item in objects]
    if public_ids != sorted(public_ids):
        raise ContractError("observed objects must be sorted by public_id")
    if len(public_ids) != len(set(public_ids)):
        raise ContractError("public geometry ids must be unique")


def assert_public_safe(document: Any) -> None:
    hits = _forbidden_hits(document)
    if hits:
        raise ContractError(f"hidden fields in public payload: {hits}")
    if isinstance(document, Mapping) and "observation" in document:
        _exact_fields(document, PUBLIC_RECORD_FIELDS, "public")
        pair_id = document["pair_id"]
        if not isinstance(pair_id, str) or not re.fullmatch(r"oe_[0-9a-f]{16}", pair_id):
            raise ContractError("pair_id must be an opaque public-observation hash")
        if not isinstance(document["split_group"], str) or not document["split_group"]:
            raise ContractError("split_group must be a non-empty string")
        _assert_observation_contract(document["observation"])
    else:
        _assert_observation_contract(document)
    canonical_json(document)
