from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fractureflow.obs_equiv_geometry import (
    disks_intersect,
    point_disk_distance,
    reference_posterior,
    scene_length_scale,
)
from fractureflow.obs_equiv_manifest import (
    ALLOWED_NOISE_NAMES,
    GROUP_EXACT,
    SCHEMA_EVALUATOR,
    SCHEMA_EVALUATOR_V2,
    SCHEMA_OBSERVATION,
    TRUTH_EXACT_SYNTHETIC,
    ContractError,
    Disk,
    Scene,
    TwoPointQuery,
    assert_public_safe,
    canonical_json,
    finite_number,
    normalized_weights,
    sha256_json,
)


CONTROL_MECHANISM = "QUERY_INVARIANT_CONTROL"
SUPPORTED_MECHANISMS = ("HIDDEN_BRIDGE", "DECOY_BRIDGE", "OCCLUDED_CONTINUATION")
MIN_DECOUPLED_FAMILIES = 3

# 旧 v2 配置（没有 families 键）的冻结回退：机制与族仍是 1:1。
# 只用于重放 engineer_pilot_v3 及更早数据；带 families 的新配置不得走此表。
LEGACY_V2_MECHANISM_WORLD_FAMILIES = {
    "HIDDEN_BRIDGE": "PLANAR_DISK_HIDDEN_BRIDGE_V1",
    "DECOY_BRIDGE": "PLANAR_DISK_DECOY_BRIDGE_V1",
    "OCCLUDED_CONTINUATION": "PLANAR_DISK_OCCLUDED_CONTINUATION_V1",
}


def family_parameters(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...] | None:
    """解析 config 的 families 段。

    返回 None 表示旧 v2 配置（机制↔族 1:1 的冻结回退），仅用于重放 v3 及更早数据。
    新配置必须声明 >= 3 个族；族名不得编码机制名，族参数只允许 name 与 scale。
    E6D D-01 唯一例外：显式 allow_single_family_screen=true 且 role 以 MISSPEC_
    开头时，允许恰好 1 个族（错设臂筛查专用；EC-1：对其它 role 一律拒绝）。
    """
    if not isinstance(config, Mapping):
        raise ContractError("config must be an object")
    families = config.get("families")
    if families is None:
        return None
    if not isinstance(families, list):
        raise ContractError("families must be a list of at least three tiers")
    screen = config.get("allow_single_family_screen", False)
    if not isinstance(screen, bool):
        raise ContractError("allow_single_family_screen must be a boolean")
    if screen:
        role = config.get("role")
        if not isinstance(role, str) or not role.startswith(MISSPEC_ROLE_PREFIX):
            raise ContractError(
                "allow_single_family_screen is reserved for MISSPEC_ prefixed roles"
            )
        if len(families) != 1:
            raise ContractError(
                "allow_single_family_screen admits exactly one family for screening"
            )
    elif len(families) < MIN_DECOUPLED_FAMILIES:
        raise ContractError("families must be a list of at least three tiers")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in families:
        if not isinstance(item, Mapping) or set(item) != {"name", "scale"}:
            raise ContractError("each family must contain exactly name and scale")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ContractError("family name must be a non-empty string")
        if name in seen:
            raise ContractError("family names must be unique")
        for mechanism in SUPPORTED_MECHANISMS:
            if mechanism in name:
                raise ContractError("family names must not encode mechanism names")
        scale = finite_number(item.get("scale"), f"family scale {name}")
        if scale <= 0.0:
            raise ContractError("family scale must be positive")
        seen.add(name)
        parsed.append({"name": name, "scale": scale})
    return tuple(parsed)


def configured_mechanisms(config: Mapping[str, Any]) -> list[str]:
    mechanisms = config.get("mechanisms", ["HIDDEN_BRIDGE"])
    if not isinstance(mechanisms, list) or any(not isinstance(name, str) for name in mechanisms):
        raise ContractError("mechanisms must be a list of strings")
    if not mechanisms:
        raise SystemExit("empty sample set: no mechanisms selected")
    if len(set(mechanisms)) != len(mechanisms):
        raise ContractError("mechanisms must be unique")
    unknown = [name for name in mechanisms if name not in SUPPORTED_MECHANISMS]
    if unknown:
        raise ContractError(f"unsupported mechanisms: {unknown}")
    return list(mechanisms)


def public_geometry_id(disk: Disk) -> str:
    return "p" + sha256_json(disk.geometry_dict())[:16]


def opaque_query_id(query: TwoPointQuery) -> str:
    return "q_" + sha256_json({
        "A": list(query.point_a),
        "B": list(query.point_b),
        "r_attach": query.attach_radius,
        "attachment": "POINT_TO_FINITE_DISK_DISTANCE_LE",
        "intersection": "STRICT_NON_TANGENT",
        "outside_domain_routing": "DISABLED",
    })[:16]


def opaque_pair_id(observation: Mapping[str, Any]) -> str:
    return "oe_" + sha256_json(observation)[:16]


def observe_visible_disks(scene: Scene, query: TwoPointQuery, visible_object_ids: Sequence[str]) -> dict[str, Any]:
    visible = tuple(visible_object_ids)
    if len(visible) != len(set(visible)):
        raise ContractError("visible object ids must be unique")
    objects = []
    for object_id in visible:
        disk = scene.disk(object_id)
        objects.append({"public_id": public_geometry_id(disk), **disk.geometry_dict()})
    objects.sort(key=lambda item: item["public_id"])
    ids = [item["public_id"] for item in objects]
    if len(ids) != len(set(ids)):
        raise ContractError("public geometry ids are not unique")
    payload = {
        "schema": SCHEMA_OBSERVATION,
        "modality": "SYNTHETIC_VISIBLE_DISKS",
        "coordinate_frame": "WORLD",
        "query": query.to_dict(),
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
    assert_public_safe(payload)
    return payload


def public_record(pair_id: str, split_group: str, observation: Mapping[str, Any]) -> dict[str, Any]:
    record = {"pair_id": pair_id, "split_group": split_group, "observation": observation}
    assert_public_safe(record)
    return record


def build_pair(
    pair_id: str | None,
    scenes: Sequence[Scene],
    prior_weights: Sequence[float],
    query: TwoPointQuery,
    visible_object_ids: Sequence[str],
    expected_query_change: bool,
    mechanism: str,
    site_id: str,
    split_group: str,
    world_family: str,
    *,
    observation_transform: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pair_id is not None and (not isinstance(pair_id, str) or not pair_id.strip()):
        raise ContractError("pair_id must be a non-empty string when supplied")
    if len(scenes) < 2:
        raise ContractError("an equivalence group requires at least two scenes")
    weights = normalized_weights(prior_weights, len(scenes))
    observations = [observe_visible_disks(scene, query, visible_object_ids) for scene in scenes]
    hashes = [sha256_json(observation) for observation in observations]
    if len(set(hashes)) != 1:
        raise ContractError("public observations are not exactly equivalent")
    transform_payload: dict[str, Any] | None = None
    if observation_transform is not None:
        # E6D（§3.2）：变换在 hashes/pair_id/evaluator 构造之前完成；
        # 无变换时本分支不执行，legacy 路径逐字节不变。
        transform_payload = _build_transform_payload(observations[0], observation_transform)
        observations = [apply_observation_transform(o, transform_payload) for o in observations]
        hashes = [sha256_json(observation) for observation in observations]
        if len(set(hashes)) != 1:
            raise ContractError("transformed public observations are not exactly equivalent")
    computed_pair_id = opaque_pair_id(observations[0])
    if pair_id is None:
        pair_id = computed_pair_id
    elif pair_id != computed_pair_id:
        raise ContractError("pair_id must equal the public observation hash")
    posterior, labels = reference_posterior(scenes, weights, query)
    changed = len(set(labels)) > 1
    if changed != bool(expected_query_change):
        raise ContractError("query labels do not match the declared control type")
    public = public_record(pair_id, split_group, observations[0])
    evaluator = {
        "schema": SCHEMA_EVALUATOR_V2 if transform_payload is not None else SCHEMA_EVALUATOR,
        "pair_id": pair_id,
        "group_type": GROUP_EXACT,
        "truth_level": TRUTH_EXACT_SYNTHETIC,
        "world_family": world_family,
        "mechanism": mechanism,
        "site_id": site_id,
        "split_group": split_group,
        "expected_query_change": bool(expected_query_change),
        "observation_sha256": hashes[0],
        "query": query.to_dict(),
        "scenes": [scene.to_dict() for scene in scenes],
        "visible_object_ids": list(visible_object_ids),
        "prior_weights": list(weights),
        "query_labels": list(labels),
        "reference_posterior": posterior,
    }
    if transform_payload is not None:
        evaluator["observation_transform"] = transform_payload
    validation = validate_pair(public, evaluator)
    if not validation["pass"]:
        raise ContractError(f"pair validation failed: {validation['errors']}")
    return public, evaluator


def validate_pair(public: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    checks = {}
    errors = []
    try:
        pair_id = evaluator.get("pair_id")
        checks["pair_id_valid"] = isinstance(pair_id, str) and bool(pair_id.strip())
        if evaluator.get("group_type") != GROUP_EXACT:
            raise ContractError("this implementation only validates exact observation groups")
        if evaluator.get("truth_level") != TRUTH_EXACT_SYNTHETIC:
            raise ContractError("this implementation only validates exact synthetic truth")
        scenes = [Scene.from_dict(item) for item in evaluator["scenes"]]
        query = TwoPointQuery.from_dict(evaluator["query"])
        visible = evaluator["visible_object_ids"]
        weights = normalized_weights(evaluator["prior_weights"], len(scenes))
        posterior, labels = reference_posterior(scenes, weights, query)
        clean_observations = [observe_visible_disks(scene, query, visible) for scene in scenes]
        transform = evaluator.get("observation_transform")
        if transform is not None:
            # E6D（§3.3）：重放只读 evaluator.observation_transform + clean scenes + query；
            # public_equal / replay / hash 用 transformed，真值重放仍用 clean scenes。
            observations = [apply_observation_transform(o, transform) for o in clean_observations]
        else:
            observations = clean_observations
        hashes = [sha256_json(observation) for observation in observations]
        checks["public_equal"] = len(set(hashes)) == 1
        checks["pair_id_is_public_hash"] = pair_id == opaque_pair_id(public["observation"])
        checks["query_id_is_public_hash"] = query.query_id == opaque_query_id(query)
        checks["public_pair_identity"] = public.get("pair_id") == evaluator.get("pair_id")
        checks["public_split_identity"] = public.get("split_group") == evaluator.get("split_group")
        checks["public_observation_replay"] = canonical_json(public["observation"]) == canonical_json(observations[0])
        checks["observation_hash"] = bool(public["observation"]) and evaluator.get("observation_sha256") == hashes[0]
        checks["query_snapshot"] = canonical_json(public["observation"]["query"]) == canonical_json(evaluator["query"])
        checks["query_labels_replay"] = list(labels) == evaluator["query_labels"]
        checks["reference_posterior_replay"] = abs(posterior - float(evaluator["reference_posterior"])) <= 1e-12
        changed = len(set(labels)) > 1
        checks["expected_query_change"] = changed == evaluator["expected_query_change"]
        if transform is not None:
            checks.update(_transform_checks(
                public, evaluator, transform, clean_observations[0], observations[0]
            ))
        assert_public_safe(public)
        checks["public_safe"] = True
    except (KeyError, TypeError, ValueError) as exc:
        checks["public_safe"] = False
        errors.append(str(exc))
    failed = [name for name, passed in checks.items() if not passed]
    errors.extend(f"failed check: {name}" for name in failed)
    return {
        "pass": not errors and bool(checks),
        "checks": checks,
        "errors": errors,
        "enumerated_checks": len(checks),
    }


@dataclass(frozen=True)
class _Case:
    scenes: tuple[Scene, ...]
    weights: tuple[float, ...]
    change_query: TwoPointQuery
    control_query: TwoPointQuery
    visible_object_ids: tuple[str, ...]
    mechanism: str
    world_family: str


def _pilot_parameters(config: Mapping[str, Any]) -> tuple[int, float, float, str, str]:
    if not isinstance(config, Mapping):
        raise ContractError("config must be an object")
    count = config.get("pair_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ContractError("pair_count must be a positive integer")
    probability = finite_number(config.get("bridge_probability"), "bridge_probability")
    if not 0.0 < probability < 1.0:
        raise ContractError("bridge_probability must be strictly between zero and one")
    spacing = finite_number(config.get("spacing"), "spacing")
    if spacing <= 0.0:
        raise ContractError("spacing must be positive")
    site_id = config.get("site_id")
    split_group = config.get("split_group")
    if not isinstance(site_id, str) or not site_id or not isinstance(split_group, str) or not split_group:
        raise ContractError("site_id and split_group are required")
    return count, probability, spacing, site_id, split_group


def _materialize_cases(
    cases: Sequence[_Case],
    config: Mapping[str, Any],
    *,
    decoupled: bool,
    observation_transform: Mapping[str, Any] | None = None,
    family_scale: float | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    _, _, _, site_id, split_group = _pilot_parameters(config)
    records = []
    for case in cases:
        change_query = TwoPointQuery(
            opaque_query_id(case.change_query),
            case.change_query.point_a,
            case.change_query.point_b,
            case.change_query.attach_radius,
        )
        control_query = TwoPointQuery(
            opaque_query_id(case.control_query),
            case.control_query.point_a,
            case.control_query.point_b,
            case.control_query.attach_radius,
        )
        change_transform: dict[str, Any] | None = None
        if observation_transform is not None:
            if family_scale is None:
                raise ContractError("observation_transform requires a family scale")
            change_transform = {**observation_transform, "scale": finite_number(family_scale, "family scale")}
        records.append(build_pair(
            None, case.scenes, case.weights, change_query,
            case.visible_object_ids, True, case.mechanism, site_id, split_group,
            case.world_family, observation_transform=change_transform,
        ))
        # 交叉乘积下控制行必须携带其配对 case 的机制，否则 LOMO 折无法把配对控制
        # 与主行同折进出（M4 回执登记过的控制组泄漏风险）。旧 v2 配置保持原标签。
        control_mechanism = case.mechanism if decoupled else CONTROL_MECHANISM
        # D-07（CLEAN_CONTROL_NO_TRANSFORM）：所有臂的控制行一律 clean 构建，
        # 不携带 observation_transform，走 legacy 13 项检查，public 与 clean 逐字节相同。
        records.append(build_pair(
            None, case.scenes, case.weights, control_query,
            case.visible_object_ids, False, control_mechanism, site_id,
            split_group, case.world_family,
        ))
    return records


def hidden_bridge_cases(config: Mapping[str, Any], family: Mapping[str, Any]) -> list[_Case]:
    count, probability, spacing, _, _ = _pilot_parameters(config)
    scale = finite_number(family.get("scale"), "family scale")
    name = family.get("name")
    if not isinstance(name, str) or not name:
        raise ContractError("family name is required")
    cases = []
    for index in range(count):
        offset = index * spacing * scale
        left = Disk("left", (0.0, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        right = Disk("right", (2.0 * scale, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        bridge = Disk("bridge", (1.0 * scale, offset, 0.0), (0.0, 0.0, 1.0), 1.5 * scale)
        prefix = f"oe_{index:06d}"
        with_bridge = Scene(prefix + "_with", [left, right, bridge])
        without_bridge = Scene(prefix + "_without", [left, right])
        cases.append(_Case(
            scenes=(with_bridge, without_bridge),
            weights=(probability, 1.0 - probability),
            change_query=TwoPointQuery("change", left.center, right.center, 0.0),
            control_query=TwoPointQuery(
                "control", left.center, (0.0, offset + 0.5 * scale, 0.0), 0.0
            ),
            visible_object_ids=("left", "right"),
            mechanism="HIDDEN_BRIDGE",
            world_family=name,
        ))
    return cases


def decoy_bridge_cases(config: Mapping[str, Any], family: Mapping[str, Any]) -> list[_Case]:
    count, probability, spacing, _, _ = _pilot_parameters(config)
    scale = finite_number(family.get("scale"), "family scale")
    name = family.get("name")
    if not isinstance(name, str) or not name:
        raise ContractError("family name is required")
    cases = []
    for index in range(count):
        offset = index * spacing * scale
        left = Disk("left", (0.0, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        right = Disk("right", (2.5 * scale, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        decoy = Disk("decoy", (0.8 * scale, offset, 0.0), (0.0, 0.0, 1.0), 1.0 * scale)
        attachment = Disk("attachment", (1.9 * scale, offset, 0.0), (0.0, 0.0, 1.0), 0.7 * scale)
        prefix = f"decoy_{index:06d}"
        attached = Scene(prefix + "_attached", [left, right, decoy, attachment])
        decoy_only = Scene(prefix + "_decoy_only", [left, right, decoy])
        cases.append(_Case(
            scenes=(attached, decoy_only),
            weights=(probability, 1.0 - probability),
            change_query=TwoPointQuery("change", left.center, right.center, 0.0),
            control_query=TwoPointQuery("control", left.center, decoy.center, 0.0),
            visible_object_ids=("left", "right", "decoy"),
            mechanism="DECOY_BRIDGE",
            world_family=name,
        ))
    return cases


def occluded_continuation_cases(config: Mapping[str, Any], family: Mapping[str, Any]) -> list[_Case]:
    count, probability, spacing, _, _ = _pilot_parameters(config)
    scale = finite_number(family.get("scale"), "family scale")
    name = family.get("name")
    if not isinstance(name, str) or not name:
        raise ContractError("family name is required")
    cases = []
    for index in range(count):
        offset = index * spacing * scale
        left = Disk("left", (0.0, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        right = Disk("right", (3.0 * scale, offset, 0.0), (1.0, 0.0, 0.0), 1.0 * scale)
        stub = Disk("stub", (0.9 * scale, offset, 0.0), (0.0, 0.0, 1.0), 1.0 * scale)
        long = Disk("continuation", (1.7 * scale, offset, 0.0), (0.0, 0.0, 1.0), 1.5 * scale)
        short = Disk("continuation", (1.8 * scale, offset, 0.0), (0.0, 0.0, 1.0), 1.0 * scale)
        prefix = f"continuation_{index:06d}"
        long_scene = Scene(prefix + "_long", [left, right, stub, long])
        short_scene = Scene(prefix + "_short", [left, right, stub, short])
        cases.append(_Case(
            scenes=(long_scene, short_scene),
            weights=(probability, 1.0 - probability),
            change_query=TwoPointQuery("change", left.center, right.center, 0.0),
            control_query=TwoPointQuery("control", left.center, stub.center, 0.0),
            visible_object_ids=("left", "right", "stub"),
            mechanism="OCCLUDED_CONTINUATION",
            world_family=name,
        ))
    return cases


CASE_BUILDERS = {
    "HIDDEN_BRIDGE": hidden_bridge_cases,
    "DECOY_BRIDGE": decoy_bridge_cases,
    "OCCLUDED_CONTINUATION": occluded_continuation_cases,
}


def _families_for(mechanism: str, config: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    families = family_parameters(config)
    if families is None:
        if mechanism not in LEGACY_V2_MECHANISM_WORLD_FAMILIES:
            raise ContractError(f"unknown legacy mechanism: {mechanism}")
        return (
            ({"name": LEGACY_V2_MECHANISM_WORLD_FAMILIES[mechanism], "scale": 1.0},),
            False,
        )
    return families, True


def build_pilot(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if isinstance(config, Mapping) and config.get("config_schema") == "obs_equiv.config.v6.v1":
        # v6 预注册构建（TO-W4-01）：独立生成器，v5 路径完全不经过这里。
        from fractureflow.v6_obs_equiv import build_v6_records
        return build_v6_records(config)
    _pilot_parameters(config)
    mechanisms = configured_mechanisms(config)
    # E6D（S4）：空记录检查前先解析 transform，非法 config 尽早失败。
    observation_transform = parse_observation_transform(config)
    records = []
    for name in mechanisms:
        families, decoupled = _families_for(name, config)
        for family in families:
            records.extend(_materialize_cases(
                CASE_BUILDERS[name](config, family), config, decoupled=decoupled,
                observation_transform=observation_transform,
                family_scale=family["scale"],
            ))
    if not records:
        raise SystemExit("empty sample set")
    return records


def hidden_bridge_pilot(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    families, decoupled = _families_for("HIDDEN_BRIDGE", config)
    records = []
    for family in families:
        records.extend(_materialize_cases(
            hidden_bridge_cases(config, family), config, decoupled=decoupled
        ))
    return records


# =====================================================================
# E6D observation transforms · 裁定 e6d_architect_ruling_20261025（D-01…D-07，
# 全 ADOPT A + EC-1…EC-5）。计划：e6d_implementation_plan_20261024/plan.md。
# 关键约束：
#   * 唯一分支条件是 evaluator 行内是否出现 observation_transform（§4.1）；
#     无该键的 legacy 路径逐字节不变（13 项检查、evaluator 16 键、schema v1）。
#   * 变换行：evaluator 17 键、schema v2、25 项检查；control 行一律 clean（D-07）。
#   * D-06 盘面保持：噪声中心位移投影到每盘自己的平面、法向不扰动；
#     传感器轴固定 (0,1,0)（x 是 left/right 盘的法向，会把端点推出盘面）。
# =====================================================================

SCHEMA_OBSERVATION_TRANSFORM = "obs_equiv.observation_transform.v1"
TRANSFORM_APPLIES_TO = "PUBLIC_OBSERVATION_ONLY"
MISSPEC_ROLE_PREFIX = "MISSPEC_"

TRANSFORM_ARMS = (
    "OBSERVATION_NOISE_LADDER",
    "OCCLUSION_FRACTION",
    "OUT_OF_DICTIONARY_OBJECT",
    "SENSOR_OFFSET",
)
SENSOR_MODES = ("GLOBAL_RIGID", "DIFFERENTIAL")
_NOISE_RUNGS = {"S005": 0.05, "S010": 0.10, "S020": 0.20, "S040": 0.40}
_OCCLUSION_RUNGS = {"F025": 0.25, "F050": 0.50, "F075": 0.75}
_SENSOR_RUNGS = {"D010": 0.10, "D025": 0.25, "D050": 0.50}
SENSOR_AXIS = (0.0, 1.0, 0.0)

TRANSFORM_NOISE_NAMES = {
    "OBSERVATION_NOISE_LADDER": "HASH_UNIFORM_DISK_GEOMETRY_V1",
    "SENSOR_OFFSET:GLOBAL_RIGID": "SENSOR_OFFSET_GLOBAL_V1",
    "SENSOR_OFFSET:DIFFERENTIAL": "SENSOR_OFFSET_DIFFERENTIAL_V1",
}

_TRANSFORM_PAYLOAD_FIELDS = frozenset({
    "transform_schema", "arm_id", "rung", "mode", "seed", "parameters",
    "applies_to", "clean_public_sha256", "corrupted_public_sha256",
    "unregistered_observed_objects",
})
_TRANSFORM_PARAM_FIELDS = {
    "OBSERVATION_NOISE_LADDER": frozenset({
        "s_nominal", "scale", "s_effective", "center_projection", "normal_perturbation",
    }),
    "OCCLUSION_FRACTION": frozenset({"f", "scale", "radius_source"}),
    "OUT_OF_DICTIONARY_OBJECT": frozenset({
        "proxy_kind", "scale", "anchor", "normal", "radius_formula",
        "isolated_offset", "isolated_radius_factor",
    }),
    "SENSOR_OFFSET": frozenset({"delta_nominal", "scale", "delta_effective", "axis", "ordering"}),
}
_TRANSFORM_FORBIDDEN_KEYS = frozenset({
    "scenes", "query_labels", "prior_weights", "reference_posterior",
    "visible_object_ids", "mechanism", "world_family", "site_id", "split_group",
})
_TRANSFORM_FORBIDDEN_PREFIXES = ("truth_", "hidden_", "oracle_")


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _assert_finite_tree(value: Any, path: str = "$transform") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite float in transform payload: {path}")


def _forbidden_transform_hits(value: Any, path: str = "$transform") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in _TRANSFORM_FORBIDDEN_KEYS or key.startswith(_TRANSFORM_FORBIDDEN_PREFIXES):
                hits.append(child)
            hits.extend(_forbidden_transform_hits(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(_forbidden_transform_hits(item, f"{path}[{index}]"))
    return hits


def parse_observation_transform(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """解析 config 顶层 observation_transform 段（E6D §3.2）。

    无该键返回 None（legacy 路径，逐字节不变）。有该键则严格白名单校验；
    未知臂/档/模式/键一律 ContractError，不得回退默认值。
    """
    if not isinstance(config, Mapping):
        raise ContractError("config must be an object")
    if "observation_transform" not in config:
        return None
    spec = config["observation_transform"]
    if not isinstance(spec, Mapping):
        raise ContractError("observation_transform must be an object")
    arm = spec.get("arm_id")
    if arm not in TRANSFORM_ARMS:
        raise ContractError(f"unknown observation_transform arm: {arm}")
    rung = spec.get("rung")
    if arm == "OBSERVATION_NOISE_LADDER":
        valid_rungs = set(_NOISE_RUNGS)
    elif arm == "OCCLUSION_FRACTION":
        valid_rungs = set(_OCCLUSION_RUNGS)
    elif arm == "OUT_OF_DICTIONARY_OBJECT":
        valid_rungs = {"D1", "D2"}
    else:
        valid_rungs = set(_SENSOR_RUNGS)
    if rung not in valid_rungs:
        raise ContractError(f"unknown rung {rung} for arm {arm}")
    mode: str | None = None
    if arm == "SENSOR_OFFSET":
        if set(spec) != {"arm_id", "rung", "mode", "seed"}:
            raise ContractError("sensor transform must contain exactly arm_id, rung, mode, seed")
        mode = spec.get("mode")
        if mode not in SENSOR_MODES:
            raise ContractError(f"unknown sensor mode: {mode}")
    else:
        if set(spec) != {"arm_id", "rung", "seed"}:
            raise ContractError("observation_transform must contain exactly arm_id, rung, seed")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("observation_transform seed must be an integer")
    return {"arm_id": arm, "rung": rung, "mode": mode, "seed": seed}


def _hash_signed_unit(seed: int, arm_id: str, rung: str, clean_public_id: str, component: str) -> float:
    """噪声确定性向量（§5.2）：sha256 键导出 [-1,1)，禁用 random。"""
    key = f"{seed}|{arm_id}|{rung}|{clean_public_id}|{component}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    unit = int(digest[:16], 16) / float(1 << 64)
    return 2.0 * unit - 1.0


def _unit_in_plane(direction: Sequence[float], normal: Sequence[float]) -> tuple[float, float, float]:
    """把方向投影到该盘自己的平面并单位化（D-06 盘面保持）。"""
    dot = sum(d * n for d, n in zip(direction, normal))
    projected = tuple(d - dot * n for d, n in zip(direction, normal))
    norm = math.sqrt(sum(c * c for c in projected))
    if norm > 1e-12:
        return tuple(c / norm for c in projected)  # type: ignore[return-value]
    axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    axis = min(axes, key=lambda a: abs(sum(x * y for x, y in zip(a, normal))))
    dot_axis = sum(x * y for x, y in zip(axis, normal))
    projected_axis = tuple(x - dot_axis * y for x, y in zip(axis, normal))
    norm_axis = math.sqrt(sum(c * c for c in projected_axis))
    if norm_axis <= 1e-12:
        raise ContractError("cannot construct an in-plane direction for this normal")
    return tuple(c / norm_axis for c in projected_axis)  # type: ignore[return-value]


def _ood_proxy(clean_observation: Mapping[str, Any], rung: str, scale: float) -> dict[str, Any]:
    """OOD 代理的唯一数学入口（§5.4）：只读 public query 与 scale，不读隐藏真值。

    附带构建期预检（fail-closed，ContractError）：D2 要求端点严格盘内且代理与
    至少一个可见对象判交；D1 要求代理不判交任何可见对象且不附着任一端点。
    """
    params = clean_observation["query"]["params"]
    point_a = tuple(finite_number(v, "query A") for v in params["A"])
    point_b = tuple(finite_number(v, "query B") for v in params["B"])
    r_attach = finite_number(params["r_attach"], "r_attach")
    midpoint = tuple((a + b) / 2.0 for a, b in zip(point_a, point_b))
    objects = [
        Disk(item["public_id"], item["center"], item["normal"], item["radius"])
        for item in clean_observation["observed_objects"]
    ]
    length_scale = scene_length_scale(Scene("proxy_precheck", tuple(objects)))
    if rung == "D2":
        span = math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))
        radius = 0.75 * span + 0.1 * scale
        center = midpoint
    else:
        radius = 0.5 * scale
        center = tuple(c + 3.0 * scale * axis for c, axis in zip(midpoint, SENSOR_AXIS))
    proxy = Disk("proxy", center, (0.0, 0.0, 1.0), radius)
    if rung == "D2":
        for point in (point_a, point_b):
            inside = False
            for disk in objects:
                delta = tuple(p - c for p, c in zip(point, disk.center))
                axial = sum(d * n for d, n in zip(delta, disk.normal))
                planar = tuple(d - axial * n for d, n in zip(delta, disk.normal))
                planar_norm = math.sqrt(sum(c * c for c in planar))
                if planar_norm < disk.radius:
                    inside = True
                    break
            if not inside:
                raise ContractError("ARM_INVALID_OOD_PROXY_ENDPOINT_NOT_STRICTLY_INSIDE")
        if not any(disks_intersect(proxy, disk, length_scale) for disk in objects):
            raise ContractError("ARM_INVALID_OOD_PROXY_NOT_INTERSECTING")
    else:
        if any(disks_intersect(proxy, disk, length_scale) for disk in objects):
            raise ContractError("ARM_INVALID_OOD_ISOLATED_PROXY_INTERSECTS")
        for point in (point_a, point_b):
            if point_disk_distance(proxy, point) <= r_attach:
                raise ContractError("ARM_INVALID_OOD_ISOLATED_PROXY_ATTACHED_TO_ENDPOINT")
    return {"public_id": public_geometry_id(proxy), **proxy.geometry_dict()}


def _validate_transform_payload(transform: Mapping[str, Any]) -> None:
    """payload 严格校验（§3.1 约束全集；违反即 ContractError）。"""
    if not isinstance(transform, Mapping) or set(transform) != _TRANSFORM_PAYLOAD_FIELDS:
        raise ContractError("observation_transform fields must be exactly the ten contract keys")
    if transform["transform_schema"] != SCHEMA_OBSERVATION_TRANSFORM:
        raise ContractError("unsupported transform schema")
    if transform["applies_to"] != TRANSFORM_APPLIES_TO:
        raise ContractError("transform applies_to must be PUBLIC_OBSERVATION_ONLY")
    arm = transform["arm_id"]
    if arm not in TRANSFORM_ARMS:
        raise ContractError(f"unknown transform arm: {arm}")
    rung = transform["rung"]
    mode = transform["mode"]
    if arm == "OBSERVATION_NOISE_LADDER":
        if mode != "IN_PLANE_CENTER_RADIUS" or rung not in _NOISE_RUNGS:
            raise ContractError("noise transform mode/rung mismatch")
    elif arm == "OCCLUSION_FRACTION":
        if mode != "NONE" or rung not in _OCCLUSION_RUNGS:
            raise ContractError("occlusion transform mode/rung mismatch")
    elif arm == "OUT_OF_DICTIONARY_OBJECT":
        if mode != "NONE" or rung not in ("D1", "D2"):
            raise ContractError("ood transform mode/rung mismatch")
    else:
        if mode not in SENSOR_MODES or rung not in _SENSOR_RUNGS:
            raise ContractError("sensor transform mode/rung mismatch")
    seed = transform["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("transform seed must be an integer")
    if not _hex64(transform["clean_public_sha256"]) or not _hex64(transform["corrupted_public_sha256"]):
        raise ContractError("transform hashes must be 64 lowercase hex digits")
    parameters = transform["parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != _TRANSFORM_PARAM_FIELDS[arm]:
        raise ContractError(f"transform parameters must be exactly {sorted(_TRANSFORM_PARAM_FIELDS[arm])}")
    _assert_finite_tree(transform)
    hits = _forbidden_transform_hits(transform)
    if hits:
        raise ContractError(f"forbidden truth/hidden fields in transform payload: {hits}")
    if arm == "OBSERVATION_NOISE_LADDER":
        if parameters["center_projection"] != "PER_DISK_PLANE":
            raise ContractError("noise center_projection must be PER_DISK_PLANE (D-06)")
        if parameters["normal_perturbation"] != "DISABLED_BY_ATTACHMENT_CONTRACT":
            raise ContractError("noise normal_perturbation must be DISABLED_BY_ATTACHMENT_CONTRACT (D-06)")
        if not math.isclose(parameters["s_effective"], parameters["s_nominal"] * parameters["scale"], rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("noise s_effective must equal s_nominal * scale")
    elif arm == "OCCLUSION_FRACTION":
        if not 0.0 < parameters["f"] < 1.0:
            raise ContractError("occlusion fraction must be strictly between zero and one")
        if parameters["radius_source"] != "CLEAN_PUBLIC_OBSERVATION_RADIUS":
            raise ContractError("occlusion radius_source must be the clean public radius (F8)")
    elif arm == "SENSOR_OFFSET":
        if list(parameters["axis"]) != list(SENSOR_AXIS):
            raise ContractError("sensor axis must be (0,1,0) (D-06; x is the left/right normal)")
        if not math.isclose(parameters["delta_effective"], parameters["delta_nominal"] * parameters["scale"], rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("sensor delta_effective must equal delta_nominal * scale")
        expected_ordering = "GLOBAL_RIGID_ALL_OBJECTS" if mode == "GLOBAL_RIGID" else "CLEAN_PUBLIC_ID_ASCENDING_SPREAD"
        if parameters["ordering"] != expected_ordering:
            raise ContractError("sensor ordering must match the declared mode")
    unregistered = transform["unregistered_observed_objects"]
    if not isinstance(unregistered, list):
        raise ContractError("unregistered_observed_objects must be a list")
    if arm == "OUT_OF_DICTIONARY_OBJECT":
        if len(unregistered) != 1 or set(unregistered[0]) != {"public_id", "type", "center", "normal", "radius"}:
            raise ContractError("ood transform must carry exactly one full proxy object dict")
    elif unregistered:
        raise ContractError("only ood transforms may carry unregistered observed objects")


def apply_observation_transform(
    observation: Mapping[str, Any], transform: Mapping[str, Any]
) -> dict[str, Any]:
    """唯一变换数学入口（P-03）：clean public observation + payload → corrupted。

    只读 observation（clean public）与 payload；噪声按 clean public_id 键控；
    传感器 DIFFERENTIAL 按输入顺序（即 clean public_id 升序）展开；OOD 代理取
    payload.unregistered_observed_objects（构建期由 _ood_proxy 计算并预检）。
    """
    arm = transform.get("arm_id")
    if arm not in TRANSFORM_ARMS:
        raise ContractError(f"unknown transform arm: {arm}")
    parameters = transform.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ContractError("transform parameters must be an object")
    objects = [dict(item) for item in observation["observed_objects"]]
    unregistered = transform.get("unregistered_observed_objects", [])
    noise_name: str | None = None
    if arm == "OBSERVATION_NOISE_LADDER":
        s_nominal = finite_number(parameters["s_nominal"], "s_nominal")
        s_effective = finite_number(parameters["s_effective"], "s_effective")
        for item in objects:
            clean_id = item["public_id"]
            components = tuple(
                _hash_signed_unit(transform["seed"], arm, transform["rung"], clean_id, c)
                for c in ("center_x", "center_y", "center_z")
            )
            in_plane = _unit_in_plane(components, item["normal"])
            item["center"] = [c + s_effective * d for c, d in zip(item["center"], in_plane)]
            u_r = _hash_signed_unit(transform["seed"], arm, transform["rung"], clean_id, "radius")
            item["radius"] = item["radius"] * (1.0 + 0.5 * s_nominal * u_r)
        noise_name = TRANSFORM_NOISE_NAMES[arm]
    elif arm == "OCCLUSION_FRACTION":
        fraction = finite_number(parameters["f"], "f")
        for item in objects:
            item["radius"] = item["radius"] * (1.0 - fraction)
    elif arm == "OUT_OF_DICTIONARY_OBJECT":
        if len(unregistered) != 1:
            raise ContractError("ood transform requires exactly one proxy object")
        objects.append(dict(unregistered[0]))
    elif arm == "SENSOR_OFFSET":
        delta_effective = finite_number(parameters["delta_effective"], "delta_effective")
        axis = list(parameters["axis"])
        if transform.get("mode") == "GLOBAL_RIGID":
            for item in objects:
                item["center"] = [c + delta_effective * a for c, a in zip(item["center"], axis)]
        elif transform.get("mode") == "DIFFERENTIAL":
            count = len(objects)
            if count < 2:
                raise ContractError("ARM_INVALID_SENSOR_DIFFERENTIAL_NEEDS_TWO_OBJECTS")
            for index, item in enumerate(objects):
                spread = 2.0 * index / (count - 1) - 1.0
                item["center"] = [c + delta_effective * spread * a for c, a in zip(item["center"], axis)]
        else:
            raise ContractError(f"unknown sensor mode: {transform.get('mode')}")
        noise_name = TRANSFORM_NOISE_NAMES[f"{arm}:{transform.get('mode')}"]
    for item in objects:
        disk = Disk(item["public_id"], item["center"], item["normal"], item["radius"])
        item["public_id"] = public_geometry_id(disk)
    objects.sort(key=lambda item: item["public_id"])
    ids = [item["public_id"] for item in objects]
    if len(ids) != len(set(ids)):
        raise ContractError("public geometry ids are not unique")
    corrupted = dict(observation)
    corrupted["observed_objects"] = objects
    if noise_name is not None:
        if noise_name not in ALLOWED_NOISE_NAMES:
            raise ContractError(f"undeclared noise name: {noise_name}")
        corrupted["noise"] = {"name": noise_name, "order": "PUBLIC_OBSERVATION_ONLY"}
    assert_public_safe(corrupted)
    return corrupted


def _build_transform_payload(clean_observation: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """从 config 级 spec（含 scale）构造完整 payload（§3.1 十键）。"""
    arm = spec["arm_id"]
    rung = spec["rung"]
    mode = spec["mode"]
    seed = spec["seed"]
    scale = finite_number(spec["scale"], "transform scale")
    if scale <= 0.0:
        raise ContractError("transform scale must be positive")
    if arm == "OBSERVATION_NOISE_LADDER":
        s_nominal = _NOISE_RUNGS[rung]
        parameters = {
            "s_nominal": s_nominal,
            "scale": scale,
            "s_effective": s_nominal * scale,
            "center_projection": "PER_DISK_PLANE",
            "normal_perturbation": "DISABLED_BY_ATTACHMENT_CONTRACT",
        }
        declared_mode = "IN_PLANE_CENTER_RADIUS"
    elif arm == "OCCLUSION_FRACTION":
        parameters = {
            "f": _OCCLUSION_RUNGS[rung],
            "scale": scale,
            "radius_source": "CLEAN_PUBLIC_OBSERVATION_RADIUS",
        }
        declared_mode = "NONE"
    elif arm == "OUT_OF_DICTIONARY_OBJECT":
        if rung == "D2":
            parameters = {
                "proxy_kind": "BRIDGE_PROXY",
                "scale": scale,
                "anchor": "PUBLIC_QUERY_MIDPOINT",
                "normal": [0.0, 0.0, 1.0],
                "radius_formula": "0.75*|B-A|+0.1*scale",
                "isolated_offset": None,
                "isolated_radius_factor": None,
            }
        else:
            parameters = {
                "proxy_kind": "ISOLATED_PROXY",
                "scale": scale,
                "anchor": "PUBLIC_QUERY_MIDPOINT",
                "normal": [0.0, 0.0, 1.0],
                "radius_formula": "isolated_radius_factor*scale",
                "isolated_offset": [0.0, 3.0, 0.0],
                "isolated_radius_factor": 0.5,
            }
        declared_mode = "NONE"
    else:
        delta_nominal = _SENSOR_RUNGS[rung]
        parameters = {
            "delta_nominal": delta_nominal,
            "scale": scale,
            "delta_effective": delta_nominal * scale,
            "axis": list(SENSOR_AXIS),
            "ordering": "GLOBAL_RIGID_ALL_OBJECTS" if mode == "GLOBAL_RIGID" else "CLEAN_PUBLIC_ID_ASCENDING_SPREAD",
        }
        declared_mode = mode
    unregistered: list[dict[str, Any]] = []
    if arm == "OUT_OF_DICTIONARY_OBJECT":
        unregistered.append(_ood_proxy(clean_observation, rung, scale))
    payload: dict[str, Any] = {
        "transform_schema": SCHEMA_OBSERVATION_TRANSFORM,
        "arm_id": arm,
        "rung": rung,
        "mode": declared_mode,
        "seed": seed,
        "parameters": parameters,
        "applies_to": TRANSFORM_APPLIES_TO,
        "clean_public_sha256": sha256_json(clean_observation),
        "corrupted_public_sha256": "",
        "unregistered_observed_objects": unregistered,
    }
    corrupted = apply_observation_transform(clean_observation, payload)
    payload["corrupted_public_sha256"] = sha256_json(corrupted)
    _validate_transform_payload(payload)
    return payload


def _transform_checks(
    public: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    transform: Mapping[str, Any],
    clean_observation: Mapping[str, Any],
    corrupted_observation: Mapping[str, Any],
) -> dict[str, bool]:
    """transform 路径追加的 12 项检查（§4.2 固定名称与顺序）。"""
    checks: dict[str, bool] = {}

    def record(name: str, fn) -> None:
        try:
            checks[name] = bool(fn())
        except Exception:  # noqa: BLE001 — 失败只记 False，不抛未捕获异常
            checks[name] = False

    record("transform_schema_supported", lambda: (
        evaluator.get("schema") == SCHEMA_EVALUATOR_V2
        and transform.get("transform_schema") == SCHEMA_OBSERVATION_TRANSFORM
    ))
    record("transform_applies_public_only", lambda: transform.get("applies_to") == TRANSFORM_APPLIES_TO)
    record("transform_arm_known", lambda: transform.get("arm_id") in TRANSFORM_ARMS)

    def _mode_known() -> bool:
        arm = transform.get("arm_id")
        mode = transform.get("mode")
        if arm == "OBSERVATION_NOISE_LADDER":
            return mode == "IN_PLANE_CENTER_RADIUS"
        if arm in ("OCCLUSION_FRACTION", "OUT_OF_DICTIONARY_OBJECT"):
            return mode == "NONE"
        return mode in SENSOR_MODES

    record("transform_mode_known", _mode_known)
    record("transform_parameters_valid", lambda: (
        _validate_transform_payload(transform) is None and True
    ))

    def _payload_fields_exact() -> bool:
        if not isinstance(transform, Mapping) or set(transform) != _TRANSFORM_PAYLOAD_FIELDS:
            return False
        if _forbidden_transform_hits(transform):
            return False
        seed = transform["seed"]
        return (
            (not isinstance(seed, bool)) and isinstance(seed, int)
            and _hex64(transform["clean_public_sha256"])
            and _hex64(transform["corrupted_public_sha256"])
        )

    record("transform_payload_fields_exact", _payload_fields_exact)
    record("transform_clean_hash_matches", lambda: (
        transform.get("clean_public_sha256") == sha256_json(clean_observation)
    ))
    record("transform_corrupted_hash_matches", lambda: (
        transform.get("corrupted_public_sha256") == sha256_json(corrupted_observation)
    ))
    record("transform_query_untouched", lambda: (
        canonical_json(corrupted_observation["query"]) == canonical_json(clean_observation["query"])
    ))

    def _public_id_recomputed() -> bool:
        objects = corrupted_observation["observed_objects"]
        recomputed = [
            public_geometry_id(Disk(item["public_id"], item["center"], item["normal"], item["radius"]))
            for item in objects
        ]
        ids = [item["public_id"] for item in objects]
        return (
            recomputed == ids
            and ids == sorted(ids)
            and len(ids) == len(set(ids))
            and evaluator.get("pair_id") == opaque_pair_id(public["observation"])
        )

    record("transform_public_id_recomputed", _public_id_recomputed)

    def _unregistered_consistent() -> bool:
        arm = transform.get("arm_id")
        unregistered = transform.get("unregistered_observed_objects")
        if not isinstance(unregistered, list):
            return False
        clean_ids = {item["public_id"] for item in clean_observation["observed_objects"]}
        corrupted_ids = {item["public_id"] for item in corrupted_observation["observed_objects"]}
        if arm == "OUT_OF_DICTIONARY_OBJECT":
            if len(unregistered) != 1 or set(unregistered[0]) != {"public_id", "type", "center", "normal", "radius"}:
                return False
            proxy = unregistered[0]
            recomputed = _ood_proxy(clean_observation, transform.get("rung"), transform["parameters"]["scale"])
            if canonical_json(proxy) != canonical_json(recomputed):
                return False
            return proxy["public_id"] in corrupted_ids and proxy["public_id"] not in clean_ids
        if unregistered:
            return False
        # 非代理臂不得增删对象（几何可变 → public_id 可变，但对象数不变）
        return len(corrupted_ids) == len(clean_ids)

    record("transform_unregistered_objects_consistent", _unregistered_consistent)

    def _endpoints_attached() -> bool:
        disks = tuple(
            Disk(item["public_id"], item["center"], item["normal"], item["radius"])
            for item in corrupted_observation["observed_objects"]
        )
        scene = Scene("public_corrupted", disks)
        query = TwoPointQuery.from_dict(corrupted_observation["query"])
        return _min_pdd(scene, query)

    record("transform_query_endpoints_attached", _endpoints_attached)
    return checks


def _min_pdd(scene: Scene, query: TwoPointQuery) -> bool:
    for point in (query.point_a, query.point_b):
        best = min(point_disk_distance(disk, point) for disk in scene.disks)
        if best > query.attach_radius:
            return False
    return True
