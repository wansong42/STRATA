# -*- coding: utf-8 -*-
"""v6 预注册实现的几何、单元清单与数据集构建（TO-W4-01）。

依据（只读冻结面）：
- results/obs_equiv_20261024/v6_prereg_design_r2_20261025/prereg_r2.json
  （sha256 647672b34d17aec3e9cbe2d717fc783f8cddd3d541d17d3fc467581c7844dd4d）
- config_draft_r2.json（sha256 1b2842caddd6cd3541e2475b0bc8c320ed1f39aea9e7265e2e650f2aa811549a）
- split_manifest_sha256 f96806512c892a7088784fa1ffb7d88e20897a176e6432a5618688cde20839dd

设计要点（r2 冻结口径）：
- 36 单元 = 6 机制 × 2 结构变体 × 3 族；VISIBLE_ONLY 为可识别控制（退化组）。
- 每非可识别单元 8 槽 = 4 标签制度 × 2 平移复跑；slot_condition_map 冻结
  （repeat 0: T2F/F2T/IT/IF = 槽 0-3；repeat 1: 槽 4-7）。
- offset = (unit_block_base + slot) * spacing * scale；unit_block_base 在本模块
  的 geometry spec 中冻结为 8 × unit_index。
- 先验权重附着在角色上（present=p, absent=1-p）；场景顺序无语义。
- 几何按 config_draft_r2 的符号参数实例化，并逐机制断言"实现的关系"（R-18）。

本模块不做 IO 写入；构建由四个固定命令经 build_pilot 的 v6 分支调用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fractureflow.obs_equiv_geometry import disks_intersect, reference_posterior, scene_length_scale, two_point_connected
from fractureflow.obs_equiv_manifest import (
    ContractError,
    Disk,
    Scene,
    TwoPointQuery,
    canonical_json,
    sha256_json,
)
from fractureflow.observation_equivalence import opaque_query_id, observe_visible_disks

ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = ROOT / "results/obs_equiv_20261024/v6_prereg_design_r2_20261025/prereg_r2.json"
PREREG_SHA256 = "647672b34d17aec3e9cbe2d717fc783f8cddd3d541d17d3fc467581c7844dd4d"
CONFIG_SCHEMA_V6 = "obs_equiv.config.v6.v1"
EVALUATOR_SCHEMA_V2 = "obs_equiv.evaluator.v2"
GROUP_EXACT = "EXACT_OBSERVATION_EQUIVALENT"
TRUTH_EXACT_SYNTHETIC = "EXACT_SYNTHETIC"

FAMILIES = (
    ("PLANAR_DISK_COMPACT_V1", 0.8),
    ("PLANAR_DISK_STANDARD_V1", 1.0),
    ("PLANAR_DISK_WIDE_V1", 1.2),
)

# 机制顺序 = prereg mechanism_axes 的顺序；变体顺序 = unit_manifest 的声明顺序。
MECHANISMS = (
    {
        "name": "HIDDEN_BRIDGE",
        "kind": "LEGACY_MECHANISM",
        "role": "LEGACY_CHANGE_MECHANISM",
        "identifiability": "NOT_IDENTIFIABLE_IN_PRINCIPLE",
        "variants": (
            ("BRIDGE_THICK_R1P5", {"bridge_radius": 1.5}),
            ("BRIDGE_THIN_R1P1", {"bridge_radius": 1.1}),
        ),
        "public_marker": 1.0,
    },
    {
        "name": "DECOY_BRIDGE",
        "kind": "LEGACY_MECHANISM",
        "role": "LEGACY_CHANGE_MECHANISM",
        "identifiability": "NOT_IDENTIFIABLE_IN_PRINCIPLE",
        "variants": (
            ("ATTACHMENT_R0P7", {"attachment_radius": 0.7}),
            ("ATTACHMENT_R0P9", {"attachment_radius": 0.9}),
        ),
        "public_marker": 1.02,
    },
    {
        "name": "OCCLUDED_CONTINUATION",
        "kind": "LEGACY_MECHANISM",
        "role": "LEGACY_CHANGE_MECHANISM",
        "identifiability": "NOT_IDENTIFIABLE_IN_PRINCIPLE",
        "variants": (
            ("SHORT_GAP_0P20", {"short_gap": 0.20}),
            ("SHORT_GAP_0P10", {"short_gap": 0.10}),
        ),
        "public_marker": 1.04,
    },
    {
        "name": "CHAIN_VS_DIRECT",
        "kind": "NEW_STRUCTURAL_MECHANISM",
        "role": "NEW_STRUCTURAL_MECHANISM",
        "identifiability": "NOT_IDENTIFIABLE_IN_PRINCIPLE",
        "variants": (
            ("CHAIN_GAP_0P05", {"chain_gap": 0.05}),
            ("CHAIN_GAP_0P15", {"chain_gap": 0.15}),
        ),
        "public_marker": 1.06,
    },
    {
        "name": "CROSSING_VS_PASSING",
        "kind": "NEW_STRUCTURAL_MECHANISM",
        "role": "NEW_STRUCTURAL_MECHANISM",
        "identifiability": "NOT_IDENTIFIABLE_IN_PRINCIPLE",
        "variants": (
            ("PASS_OFFSET_1P5", {"passing_offset": 1.6}),
            ("PASS_OFFSET_2P0", {"passing_offset": 2.1}),
        ),
        "public_marker": 1.08,
    },
    {
        "name": "VISIBLE_ONLY",
        "kind": "IDENTIFIABILITY_CONTROL",
        "role": "IDENTIFIABILITY_CONTROL",
        "identifiability": "IDENTIFIABLE_FROM_PUBLIC_OBSERVATION",
        "variants": (
            ("VISIBLE_TRUE_CHAIN", {"chain_layout": "TRUE"}),
            ("VISIBLE_FALSE_GAP", {"chain_layout": "FALSE"}),
        ),
        "public_marker": 1.1,
    },
)

LABEL_REGIMES = ("CHANGE_TRUE_TO_FALSE", "CHANGE_FALSE_TO_TRUE", "INVARIANT_TRUE", "INVARIANT_FALSE")
PRIMARY_REGIMES = ("CHANGE_TRUE_TO_FALSE", "CHANGE_FALSE_TO_TRUE")
# slot_condition_map（r2 冻结）：(repeat, regime) -> slot
SLOT_CONDITION_MAP = {
    (0, "CHANGE_TRUE_TO_FALSE"): 0,
    (0, "CHANGE_FALSE_TO_TRUE"): 1,
    (0, "INVARIANT_TRUE"): 2,
    (0, "INVARIANT_FALSE"): 3,
    (1, "CHANGE_TRUE_TO_FALSE"): 4,
    (1, "CHANGE_FALSE_TO_TRUE"): 5,
    (1, "INVARIANT_TRUE"): 6,
    (1, "INVARIANT_FALSE"): 7,
}
OFFSET_BLOCK_SIZE = 8
REPEAT_COUNT = 2
SPACING = 4.0

# CROSSING 的 passing 变体：几何接触阈值 = sqrt(r_hidden^2 - offset_x^2) + sqrt(r_visible^2)
# = sqrt(1.1^2-1.0^2)+1.08 = 1.5377；符号表 1.5 不构成"擦肩"，实例化取 1.6/2.1
# （VARIANT_NAME_RETAINED；realized_relation_assertion 逐机制断言）。
PASSING_CONTACT_THRESHOLD_NOTE = "1.5*scale does not realize passing; instantiated 1.6/2.1"


def unit_id(world_family: str, mechanism: str, structural_variant: str) -> str:
    return "u_" + sha256_json({
        "world_family": world_family,
        "mechanism": mechanism,
        "structural_variant": structural_variant,
    })[:16]


def frozen_prereg() -> dict[str, Any]:
    digest = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    if digest != PREREG_SHA256:
        raise ContractError(f"prereg r2 sha256 drift: {digest}")
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def frozen_unit_manifest() -> list[dict[str, Any]]:
    return list(frozen_prereg()["independent_unit"]["unit_manifest"])


def split_manifest_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([
        {"unit_id": unit["unit_id"], "split_pool": unit["split_pool"]}
        for unit in sorted(units, key=lambda item: item["unit_id"])
    ])


def _scene(prefix: str, objects: Sequence[Disk]) -> Scene:
    return Scene(prefix, list(objects))


def case_geometry(
    mechanism: str,
    variant_params: Mapping[str, Any],
    scale: float,
    offset: float,
) -> dict[str, Any]:
    """单槽几何：present/absent 两景 + 四条查询（change / invariant_true / invariant_false）。

    visible 对象在两景完全相同；hidden 对象只在 present 景。所有对象共享 y=offset。
    """
    marker = next(m["public_marker"] for m in MECHANISMS if m["name"] == mechanism)
    s = scale
    y = offset
    vis = max(1.0 * s * marker, 0.0)
    far = Disk("far", (-4.0 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
    left = Disk("left", (0.0, y, 0.0), (1.0, 0.0, 0.0), vis)
    inv_true = TwoPointQuery(
        "inv_true", left.center, (left.center[0], left.center[1] + 0.5 * s, left.center[2]), 0.0
    )
    inv_false = TwoPointQuery("inv_false", left.center, far.center, 0.0)
    visible_ids: tuple[str, ...]
    present: list[Disk]
    absent: list[Disk]
    change_query: TwoPointQuery
    if mechanism == "HIDDEN_BRIDGE":
        right = Disk("right", (2.0 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        bridge = Disk("bridge", (1.0 * s, y, 0.0), (0.0, 0.0, 1.0),
                      variant_params["bridge_radius"] * s)
        visible_ids = ("left", "right", "far")
        present = [left, right, far, bridge]
        absent = [left, right, far]
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    elif mechanism == "DECOY_BRIDGE":
        right = Disk("right", (2.5 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        decoy = Disk("decoy", (0.8 * s, y, 0.0), (0.0, 0.0, 1.0), 1.0 * s)
        attachment = Disk("attachment", (1.9 * s, y, 0.0), (0.0, 0.0, 1.0),
                         variant_params["attachment_radius"] * s)
        visible_ids = ("left", "right", "decoy", "far")
        present = [left, right, decoy, far, attachment]
        absent = [left, right, decoy, far]
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    elif mechanism == "OCCLUDED_CONTINUATION":
        right = Disk("right", (3.0 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        stub = Disk("stub", (0.9 * s, y, 0.0), (0.0, 0.0, 1.0), 1.0 * s)
        long_disk = Disk("continuation", (1.7 * s, y, 0.0), (0.0, 0.0, 1.0), 1.5 * s)
        # 短续：半径由"与 right 的法向距离 - 目标 gap"派生（R-18 realized relation）
        short_radius = (3.0 - 1.8) * s - variant_params["short_gap"] * s
        short_disk = Disk("continuation", (1.8 * s, y, 0.0), (0.0, 0.0, 1.0), short_radius)
        visible_ids = ("left", "right", "stub", "far")
        present = [left, right, stub, far, long_disk]
        absent = [left, right, stub, far, short_disk]
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    elif mechanism == "CHAIN_VS_DIRECT":
        chain_gap = variant_params["chain_gap"]
        # B 端固定在 2.70*scale（geometry_spec_contract 的 d 目标）；链右盘按 gap
        # 落在 2.65/2.75，两者都与 right 盘相接但彼此留 gap。
        chain_left = Disk("chain_left", (0.6 * s, y, 0.0), (0.0, 0.0, 1.0), 1.0 * s)
        chain_right_offset = 0.6 + 2.0 + chain_gap
        chain_right = Disk("chain_right", (chain_right_offset * s, y, 0.0), (0.0, 0.0, 1.0), 1.0 * s)
        right = Disk("right", (2.70 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        direct_bridge = Disk("direct_bridge", (1.35 * s, y, 0.0), (0.0, 0.0, 1.0), 1.5 * s)
        visible_ids = ("left", "right", "far")
        # 角色语义（prior ROLE_ATTACHED_PRESENT_VS_ABSENT）：present = 连通的一景
        present = [left, right, far, direct_bridge]
        absent = [left, right, far, chain_left, chain_right]
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    elif mechanism == "CROSSING_VS_PASSING":
        right = Disk("right", (2.0 * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        crossing = Disk("crossing", (1.0 * s, y + 0.4 * s, 0.0), (0.0, 0.0, 1.0), 1.1 * s)
        passing = Disk("passing", (1.0 * s, y + variant_params["passing_offset"] * s, 0.0),
                       (0.0, 0.0, 1.0), 1.1 * s)
        visible_ids = ("left", "right", "far")
        present = [left, right, far, crossing]
        absent = [left, right, far, passing]
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    elif mechanism == "VISIBLE_ONLY":
        if variant_params["chain_layout"] == "TRUE":
            mid_offset, mid_radius, right_offset = 1.1, 1.2, 2.2
        else:
            mid_offset, mid_radius, right_offset = 1.25, 0.6, 2.5
        mid = Disk("visible_mid", (mid_offset * s, y, 0.0), (0.0, 0.0, 1.0), mid_radius * s)
        right = Disk("right", (right_offset * s, y, 0.0), (1.0, 0.0, 0.0), vis)
        visible_ids = ("left", "right", "far", "visible_mid")
        present = [left, right, far, mid]
        absent = [left, right, far, mid]  # 退化组：两景全暴露且相同
        change_query = TwoPointQuery("change", left.center, right.center, 0.0)
    else:
        raise ContractError(f"unknown v6 mechanism: {mechanism}")
    query_by_regime = {
        "CHANGE_TRUE_TO_FALSE": change_query,
        "CHANGE_FALSE_TO_TRUE": change_query,
        "INVARIANT_TRUE": inv_true,
        "INVARIANT_FALSE": inv_false,
    }
    return {
        "present_objects": present,
        "absent_objects": absent,
        "visible_ids": visible_ids,
        "query_by_regime": query_by_regime,
        "marker": marker,
        "scale": s,
    }


def _labels(scene_with: Scene, scene_without: Scene, query: TwoPointQuery) -> tuple[bool, bool]:
    return (
        two_point_connected(scene_with, query),
        two_point_connected(scene_without, query),
    )


def realized_relations(mechanism: str, variant_params: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """R-18 realized_relation_assertion：按实几何算出的关系（不是符号假设）。"""
    geom = case_geometry(mechanism, variant_params, scale, 0.0)
    with_scene = _scene("with", geom["present_objects"])
    without_scene = _scene("without", geom["absent_objects"])
    change = geom["query_by_regime"]["CHANGE_TRUE_TO_FALSE"]
    labels_change = _labels(with_scene, without_scene, change)
    labels_inv_true = _labels(with_scene, without_scene, geom["query_by_regime"]["INVARIANT_TRUE"])
    labels_inv_false = _labels(with_scene, without_scene, geom["query_by_regime"]["INVARIANT_FALSE"])
    detail: dict[str, Any] = {
        "change_labels": list(labels_change),
        "invariant_true_labels": list(labels_inv_true),
        "invariant_false_labels": list(labels_inv_false),
        "query_distance": float(
            ((change.point_a[0] - change.point_b[0]) ** 2
             + (change.point_a[1] - change.point_b[1]) ** 2
             + (change.point_a[2] - change.point_b[2]) ** 2) ** 0.5
        ),
    }
    if mechanism == "VISIBLE_ONLY":
        layout = variant_params["chain_layout"]
        expected = ([True, True] if layout == "TRUE" else [False, False])
        detail["identifiable_expected"] = expected
        detail["identifiable_realized"] = list(labels_change)
    if mechanism == "CHAIN_VS_DIRECT":
        chain_left = next(d for d in geom["absent_objects"] if d.object_id == "chain_left")
        chain_right = next(d for d in geom["absent_objects"] if d.object_id == "chain_right")
        length_scale = scene_length_scale(without_scene)
        detail["chain_disks_intersect"] = disks_intersect(chain_left, chain_right, length_scale)
        detail["chain_gap_target"] = variant_params["chain_gap"]
        detail["chain_right_offset"] = chain_right.center[0]
    if mechanism == "OCCLUDED_CONTINUATION":
        short = next(d for d in geom["absent_objects"] if d.object_id == "continuation")
        long_disk = next(d for d in geom["present_objects"] if d.object_id == "continuation")
        right = next(d for d in geom["present_objects"] if d.object_id == "right")
        length_scale = scene_length_scale(with_scene)
        detail["short_disk_radius"] = short.radius
        detail["long_reaches_right"] = disks_intersect(long_disk, right, length_scale)
        detail["short_reaches_right"] = disks_intersect(short, right, length_scale)
        detail["short_gap_target"] = variant_params["short_gap"]
    if mechanism == "CROSSING_VS_PASSING":
        hidden = next(d for d in geom["present_objects"] if d.object_id == "crossing")
        passing = next(d for d in geom["absent_objects"] if d.object_id == "passing")
        segment_point = (1.0 * scale, 0.0, 0.0)
        segment_distance_crossing = abs(hidden.center[1] - segment_point[1])
        segment_distance_passing = abs(passing.center[1] - segment_point[1])
        left = next(d for d in geom["present_objects"] if d.object_id == "left")
        right = next(d for d in geom["present_objects"] if d.object_id == "right")
        length_scale = scene_length_scale(with_scene)
        detail["crossing_segment_distance"] = segment_distance_crossing
        detail["passing_segment_distance"] = segment_distance_passing
        detail["crossing_touches_left"] = disks_intersect(hidden, left, length_scale)
        detail["crossing_touches_right"] = disks_intersect(hidden, right, length_scale)
        detail["passing_touches_left"] = disks_intersect(passing, left, length_scale)
        detail["passing_touches_right"] = disks_intersect(passing, right, length_scale)
        detail["passing_offset_instantiated"] = variant_params["passing_offset"]
    return detail


def normalized_layout_signature(objects: Sequence[Mapping[str, Any]], y_offset: float, scale: float) -> str:
    """R-07 public 布局签名：减共同 y 并按族尺度归一，哈希中心与半径（机制名不参与）。"""
    normalized = []
    for obj in objects:
        center = obj["center"] if isinstance(obj, Mapping) else list(obj.center)
        radius = obj["radius"] if isinstance(obj, Mapping) else obj.radius
        normalized.append([
            round(float(center[0]) / scale, 9),
            round((float(center[1]) - y_offset) / scale, 9),
            round(float(center[2]) / scale, 9),
            round(float(radius) / scale, 9),
        ])
    return sha256_json(sorted(normalized))


def normalized_hidden_signature(objects: Sequence[Disk], y_offset: float, scale: float) -> str:
    normalized = []
    for disk in objects:
        normalized.append([
            round(disk.center[0] / scale, 9),
            round((disk.center[1] - y_offset) / scale, 9),
            round(disk.center[2] / scale, 9),
            round(disk.normal[0], 9),
            round(disk.normal[1], 9),
            round(disk.normal[2], 9),
            round(disk.radius / scale, 9),
        ])
    return sha256_json(sorted(normalized))


def ordered_units() -> list[dict[str, Any]]:
    """36 单元按 (机制序, 变体序, 族序) 排列，带 unit_block_base = 8×index。"""
    units = []
    index = 0
    for mechanism in MECHANISMS:
        for variant_name, variant_params in mechanism["variants"]:
            for family_name, scale in FAMILIES:
                units.append({
                    "unit_index": index,
                    "unit_block_base": OFFSET_BLOCK_SIZE * index,
                    "mechanism": mechanism["name"],
                    "mechanism_kind": mechanism["kind"],
                    "mechanism_role": mechanism["role"],
                    "identifiability": mechanism["identifiability"],
                    "structural_variant": variant_name,
                    "variant_params": dict(variant_params),
                    "world_family": family_name,
                    "family_scale": scale,
                    "public_marker": mechanism["public_marker"],
                    "unit_id": unit_id(family_name, mechanism["name"], variant_name),
                })
                index += 1
    return units


def _direction_bit(uid: str, repeat_index: int) -> int:
    digest = hashlib.sha256(f"{uid}|{repeat_index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 1


def build_geometry_spec() -> dict[str, Any]:
    """geometry spec（R-18/R-07 冻结面）：精确坐标、端点、d、关系断言、签名、gate 值。"""
    units = ordered_units()
    per_mechanism = {}
    layout_signatures: dict[str, set[str]] = {}
    hidden_signatures: dict[str, set[str]] = {}
    layout_by_variant: dict[str, dict[str, str]] = {}
    for mechanism in MECHANISMS:
        name = mechanism["name"]
        entries = []
        for variant_name, variant_params in mechanism["variants"]:
            for family_name, scale in FAMILIES:
                relation = realized_relations(name, variant_params, scale)
                geom = case_geometry(name, variant_params, scale, 0.0)
                if name == "VISIBLE_ONLY":
                    layout_source = geom["present_objects"]
                else:
                    layout_source = [d for d in geom["present_objects"] if d.object_id in geom["visible_ids"]]
                public_objects = [
                    {"public_id": "", "type": "DISK", "center": list(d.center),
                     "normal": list(d.normal), "radius": d.radius}
                    for d in layout_source
                ]
                layout_sig = normalized_layout_signature(public_objects, 0.0, scale)
                hidden_objects = (
                    [d for d in geom["present_objects"] if d.object_id not in geom["visible_ids"]]
                    + [d for d in geom["absent_objects"] if d.object_id not in geom["visible_ids"]]
                )
                hidden_sig = normalized_hidden_signature(hidden_objects, 0.0, scale)
                layout_signatures.setdefault(name, set()).add(layout_sig)
                hidden_signatures.setdefault(name, set()).add(hidden_sig)
                layout_by_variant.setdefault(name, {})[f"{variant_name}|{family_name}"] = layout_sig
                change = geom["query_by_regime"]["CHANGE_TRUE_TO_FALSE"]
                entries.append({
                    "world_family": family_name,
                    "family_scale": scale,
                    "structural_variant": variant_name,
                    "variant_parameters": dict(variant_params),
                    "unit_id": unit_id(family_name, name, variant_name),
                    "query_endpoint_A": list(change.point_a),
                    "query_endpoint_B": list(change.point_b),
                    "query_distance_d": relation["query_distance"],
                    "query_r_attach": change.attach_radius,
                    "invariant_true_endpoints": [
                        list(geom["query_by_regime"]["INVARIANT_TRUE"].point_a),
                        list(geom["query_by_regime"]["INVARIANT_TRUE"].point_b),
                    ],
                    "invariant_false_endpoints": [
                        list(geom["query_by_regime"]["INVARIANT_FALSE"].point_a),
                        list(geom["query_by_regime"]["INVARIANT_FALSE"].point_b),
                    ],
                    "visible_objects": [
                        {"id": d.object_id, "center": list(d.center), "normal": list(d.normal), "radius": d.radius}
                        for d in geom["present_objects"] if d.object_id in geom["visible_ids"]
                    ],
                    "hidden_objects_present": [
                        {"id": d.object_id, "center": list(d.center), "normal": list(d.normal), "radius": d.radius}
                        for d in geom["present_objects"] if d.object_id not in geom["visible_ids"]
                    ],
                    "hidden_objects_absent": [
                        {"id": d.object_id, "center": list(d.center), "normal": list(d.normal), "radius": d.radius}
                        for d in geom["absent_objects"] if d.object_id not in geom["visible_ids"]
                    ],
                    "realized": relation,
                    "normalized_public_layout_signature": layout_sig,
                    "normalized_hidden_signature": hidden_sig,
                })
        per_mechanism[name] = {
            "public_marker": mechanism["public_marker"],
            "variant_kind": "IDENTIFIABILITY_VARIANT" if name == "VISIBLE_ONLY" else "MARGIN_VARIANT",
            "entries": entries,
        }

    mechanism_names = [m["name"] for m in MECHANISMS]
    set_list = [layout_signatures[name] for name in mechanism_names]
    pairwise_disjoint = all(
        not (set_list[i] & set_list[j])
        for i in range(len(set_list)) for j in range(i + 1, len(set_list))
    )
    per_mechanism_counts = {name: len(sigs) for name, sigs in layout_signatures.items()}
    non_identifiable = [m["name"] for m in MECHANISMS if m["name"] != "VISIBLE_ONLY"]
    layout_distinct = {
        "mechanism_count": len(mechanism_names),
        "per_mechanism_counts": per_mechanism_counts,
        "sets_pairwise_disjoint": pairwise_disjoint,
        "normalization": "subtract common y offset and divide by family scale",
        "non_identifiable_mechanisms_single_layout": all(
            per_mechanism_counts[name] == 1 for name in non_identifiable
        ),
        "visible_only_variant_layouts": per_mechanism_counts["VISIBLE_ONLY"],
        "visible_only_note": "可识别控制的两个变体按设计具有不同 public 布局（恒等性来自公开几何）",
        "assertion_passed": pairwise_disjoint and all(
            per_mechanism_counts[name] == 1 for name in non_identifiable
        ),
    }
    hidden_distinct = {
        "per_mechanism_counts": {name: len(sigs) for name, sigs in hidden_signatures.items()},
        "non_identifiable_at_least_two_variants": all(
            len(hidden_signatures[name]) >= 2 for name in non_identifiable
        ),
        "visible_only_no_hidden_structure": len(hidden_signatures["VISIBLE_ONLY"]) == 1,
        "assertion_passed": all(len(hidden_signatures[name]) >= 2 for name in non_identifiable),
    }
    gate_values = _gate_values(units)
    spec = {
        "schema": "obs_equiv.v6_geometry_spec.v1",
        "unit": "v6_impl_20261025",
        "prereg_sha256": PREREG_SHA256,
        "split_manifest_sha256": split_manifest_sha256(frozen_unit_manifest()),
        "spacing": SPACING,
        "offset_block_size": OFFSET_BLOCK_SIZE,
        "offset_formula": "offset = (unit_block_base + slot) * spacing * scale",
        "repeat_count": REPEAT_COUNT,
        "slot_condition_map": {f"{repeat}|{regime}": slot for (repeat, regime), slot in SLOT_CONDITION_MAP.items()},
        "unit_block_base_table": [
            {"unit_id": u["unit_id"], "unit_index": u["unit_index"], "unit_block_base": u["unit_block_base"]}
            for u in units
        ],
        "units_total": len(units),
        "identifiable_units": sum(1 for u in units if u["identifiability"] == "IDENTIFIABLE_FROM_PUBLIC_OBSERVATION"),
        "passing_contact_note": PASSING_CONTACT_THRESHOLD_NOTE,
        "mechanisms": per_mechanism,
        "public_layout_distinctness": layout_distinct,
        "hidden_variant_distinctness": hidden_distinct,
        "identifiable_sanity_gate_values": gate_values,
        "public_layout_parameter_role": "marker_on_visible_radii_only; label-regime assertion uses realized geometry",
    }
    spec["sha256_body"] = sha256_json({k: v for k, v in spec.items() if k != "sha256_body"})
    return spec


def _gate_values(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """R-03 判定值：从冻结几何复算（首读前），与 prereg 的 P7 设计算术对账。"""
    from fractureflow.obs_equiv_benchmark import query_distance_scaled_prior_prediction

    values = []
    for unit in units:
        if unit["identifiability"] != "IDENTIFIABLE_FROM_PUBLIC_OBSERVATION":
            continue
        geom = case_geometry(unit["mechanism"], unit["variant_params"], unit["family_scale"], 0.0)
        raw_query = geom["query_by_regime"]["CHANGE_TRUE_TO_FALSE"]
        query = TwoPointQuery(
            opaque_query_id(raw_query), raw_query.point_a, raw_query.point_b, raw_query.attach_radius
        )
        with_scene = _scene("with", geom["present_objects"])
        label = two_point_connected(with_scene, query)
        reference = 1.0 if label else 0.0
        observation = observe_visible_disks(with_scene, query, geom["visible_ids"])
        public = {"pair_id": "oe_" + "0" * 16, "split_group": "V6", "observation": observation}
        q_qdsp = query_distance_scaled_prior_prediction(public, 2.0, 1.0)
        q_ztg = 1.0 if label else 0.0
        d = ((query.point_a[0] - query.point_b[0]) ** 2
             + (query.point_a[1] - query.point_b[1]) ** 2
             + (query.point_a[2] - query.point_b[2]) ** 2) ** 0.5
        values.append({
            "unit_id": unit["unit_id"],
            "world_family": unit["world_family"],
            "structural_variant": unit["structural_variant"],
            "reference_posterior": reference,
            "query_distance_d": d,
            "expected_excess_zero_training_geometry": (q_ztg - reference) ** 2,
            "expected_excess_query_distance_scaled_prior": (q_qdsp - reference) ** 2,
            "expected_excess_counterexample_flipped": (1.0 - q_ztg - reference) ** 2,
        })
    return {"units": values, "source": "RECOMPUTED_FROM_FROZEN_GEOMETRY_SPEC"}


def build_v6_records(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """v6 数据集构建（build_pilot 的 v6 分支调用）。"""
    from fractureflow.observation_equivalence import build_pair

    if config.get("config_schema") != CONFIG_SCHEMA_V6:
        raise ContractError("build_v6_records requires config_schema obs_equiv.config.v6.v1")
    for key in ("pair_count", "bridge_probability"):
        if key in config:
            raise ContractError(f"v6 config must not carry the v5 key {key}")
    site_id = config.get("site_id")
    split_group = config.get("split_group")
    if not isinstance(site_id, str) or not site_id or not isinstance(split_group, str) or not split_group:
        raise ContractError("v6 config requires site_id and split_group")
    prior_arm = config.get("prior_arm_id")
    present_weight = config.get("present_weight")
    absent_weight = config.get("absent_weight")
    if not isinstance(present_weight, (int, float)) or not isinstance(absent_weight, (int, float)):
        raise ContractError("v6 config requires numeric prior weights")
    if abs(float(present_weight) + float(absent_weight) - 1.0) > 1e-9:
        raise ContractError("v6 prior weights must sum to one")
    if config.get("repeat_count") != REPEAT_COUNT or config.get("offset_block_size") != OFFSET_BLOCK_SIZE:
        raise ContractError("v6 repeat/offset constants differ from the frozen design")
    declared_families = {item["name"]: float(item["scale"]) for item in config.get("families", [])}
    if declared_families != {name: scale for name, scale in FAMILIES}:
        raise ContractError("v6 families must be the three frozen tiers")
    units = ordered_units()
    frozen_units = {u["unit_id"]: u for u in frozen_unit_manifest()}
    if set(frozen_units) != {u["unit_id"] for u in units}:
        raise ContractError("generated unit ids differ from the frozen unit manifest")
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for unit in units:
        frozen = frozen_units[unit["unit_id"]]
        pool = frozen["split_pool"]
        identifiable = unit["identifiability"] == "IDENTIFIABLE_FROM_PUBLIC_OBSERVATION"
        regimes = LABEL_REGIMES if not identifiable else ("IDENTIFIABLE",)
        for repeat in range(REPEAT_COUNT):
            direction = _direction_bit(unit["unit_id"], repeat)
            for regime in regimes:
                if identifiable:
                    slot = 0 if unit["structural_variant"] == "VISIBLE_TRUE_CHAIN" else 1
                    slot = slot + repeat  # 可识别单元 2 槽（repeat 维度）
                    label_regime = ("IDENTIFIABLE_TRUE" if unit["structural_variant"] == "VISIBLE_TRUE_CHAIN"
                                    else "IDENTIFIABLE_FALSE")
                else:
                    slot = SLOT_CONDITION_MAP[(repeat, regime)]
                    label_regime = regime
                offset = (unit["unit_block_base"] + slot) * SPACING * unit["family_scale"]
                geom = case_geometry(unit["mechanism"], unit["variant_params"], unit["family_scale"], offset)
                regime_key = "CHANGE_TRUE_TO_FALSE" if identifiable else regime
                raw_query = geom["query_by_regime"][regime_key]
                query = TwoPointQuery(
                    opaque_query_id(raw_query), raw_query.point_a, raw_query.point_b, raw_query.attach_radius
                )
                if identifiable:
                    scene_present = _scene(f"v6_{unit['unit_id']}_{slot}_a", geom["present_objects"])
                    scene_absent = _scene(f"v6_{unit['unit_id']}_{slot}_b", geom["absent_objects"])
                else:
                    scene_present = _scene(f"v6_{unit['unit_id']}_{slot}_present", geom["present_objects"])
                    scene_absent = _scene(f"v6_{unit['unit_id']}_{slot}_absent", geom["absent_objects"])
                # 方向：change 槽的顺序由声明制度决定（按实几何标签选择，机制间语义可反）；
                # 不变/可识别槽无顺序语义，用键控方向位覆盖两种顺序（LABEL_ORDER_COVERAGE）。
                label_present = two_point_connected(scene_present, query)
                label_absent = two_point_connected(scene_absent, query)
                if label_regime in PRIMARY_REGIMES:
                    want = (True, False) if label_regime == "CHANGE_TRUE_TO_FALSE" else (False, True)
                    if (label_present, label_absent) == want:
                        present_first = True
                    elif (label_absent, label_present) == want:
                        present_first = False
                    else:
                        raise ContractError(
                            f"mechanism geometry cannot realize regime {label_regime} "
                            f"for unit {unit['unit_id']} (labels {label_present}/{label_absent})"
                        )
                else:
                    present_first = bool(direction)
                if present_first:
                    scenes = (scene_present, scene_absent)
                    weights = (float(present_weight), float(absent_weight))
                    scene_roles = ("PRESENT", "ABSENT")
                else:
                    scenes = (scene_absent, scene_present)
                    weights = (float(absent_weight), float(present_weight))
                    scene_roles = ("ABSENT", "PRESENT")
                expected_change = label_regime in PRIMARY_REGIMES
                pair, evaluator = build_pair(
                    None, scenes, weights, query, geom["visible_ids"], expected_change,
                    unit["mechanism"], site_id, split_group, unit["world_family"],
                )
                # 标签制度断言（R-09）：实几何的标签必须与该槽声明的制度一致
                realized_labels = list(evaluator["query_labels"])
                declared_pattern = {
                    "CHANGE_TRUE_TO_FALSE": [True, False],
                    "CHANGE_FALSE_TO_TRUE": [False, True],
                    "INVARIANT_TRUE": [True, True],
                    "INVARIANT_FALSE": [False, False],
                    "IDENTIFIABLE_TRUE": [True, True],
                    "IDENTIFIABLE_FALSE": [False, False],
                }[label_regime]
                if realized_labels != declared_pattern:
                    raise ContractError(
                        f"label regime not realized: {unit['unit_id']} slot {slot} "
                        f"declared {label_regime} pattern {declared_pattern} realized {realized_labels}"
                    )
                evaluator.update({
                    "schema": EVALUATOR_SCHEMA_V2,
                    "unit_id": unit["unit_id"],
                    "unit_index": unit["unit_index"],
                    "structural_variant": unit["structural_variant"],
                    "variant_kind": "IDENTIFIABILITY_VARIANT" if identifiable else "MARGIN_VARIANT",
                    "mechanism_kind": unit["mechanism_kind"],
                    "mechanism_role": unit["mechanism_role"],
                    "identifiability": unit["identifiability"],
                    "split_pool": pool,
                    "primary_eligible": bool(frozen["primary_eligible"]),
                    "label_regime": label_regime,
                    "repeat_index": repeat,
                    "slot": slot,
                    "offset": offset,
                    "prior_arm_id": prior_arm,
                    "prior_semantics": "ROLE_ATTACHED_PRESENT_VS_ABSENT",
                    "scene_roles": list(scene_roles),
                    "direction_bit": direction,
                    "public_marker": unit["public_marker"],
                    "config_schema": CONFIG_SCHEMA_V6,
                })
                records.append((pair, evaluator))
    if not records:
        raise SystemExit("empty sample set")
    return records
