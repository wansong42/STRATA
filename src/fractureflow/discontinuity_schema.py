# -*- coding: utf-8 -*-
"""G59 多类型结构契约 S/Z/V/R（planar_interface / zone / cavity / relations）.

水印：CONTRACT_INFRA / ZERO_RESEARCH_READING（零研究读数，未迁移任何旧数据）

依据：DEEP_RESEARCH_complex_rock_discontinuity.md §1.2（四层对象）、§4.1（研究对象字段）、
§6.1（discontinuity_schema.py）、§6.3 P1、§7 第 2 条；任务书
results/architect_main_20261012_dispatch/p18_20261021/task_order_G59_discontinuity_schema_20261021.md。

本模块只做契约（schema/校验/序列化/旧 v1 只读解析），不训练、不读研究读数、不改旧模块。
纯标准库实现（不 import numpy/torch），保证导入轻量；旧 fracture_instance.py 的字段名与取值
范围由本单开工时逐条读源码核实（见 RESULTS/legacy_compat.json 的 field_mapping 段），
不凭记忆写字段。

冻结设计（与 prereg.json 一致，读数后禁改）：
  * 对象字段：id, object_type, geometry, support_scale, orientation, extent, quality,
    visibility, provenance, uncertainty, label_status（+ extensions 透传旧 v1 的 bingham/schema_ver）。
  * object_type ∈ {planar_interface, zone, cavity}；cavity 仅留结构位，状态 NOT_IMPLEMENTED（validate 拒绝）。
  * visibility ∈ {observed, observed_negative, not_observed, occluded, device_failed}。
  * label_status ∈ {EXACT, EXPERT_INTERPRETED, DERIVED_GEOMETRIC, WEAK, UNLABELED}。
  * 平面 geometry：normal 必须为单位向量（模长偏差 > 1e-6 一律拒绝，不静默归一化）、offset 有限、
    extent_box（bbox_min/bbox_max）+ 可选 polygon_xy / trace_segments。
  * 区域 geometry：center, axis（单位）, width（严格 > 0）, extent_along_axis（严格 > 0）,
    可选 thickness_subzones{core, damage}（各自 > 0 且 core+damage <= width）。
  * 关系边：src_id, dst_id, relation_type ∈ 六型, confidence ∈ [0,1], provenance, label_status；
    端点必须存在于 S∪Z∪V；自环仅 intersects 允许；(src,dst,type) 重复拒绝。
  * 隐藏真值键硬编码 8 个（任务书 §4.3）：to_public_dict 递归扫描 0 命中，from_public_dict 遇键必抛。
  * 空旧记录流 ⇒ SystemExit（禁空对空记 PASS）。

使用（不写盘，只返回 JSON-ready 结构）：
    sys = DiscontinuitySystem.from_public_dict(payload)
    payload2 = sys.to_public_dict()          # 隐藏字段与 NaN 双守卫
    viol = sys.validate()                    # list[Violation]
    obj = from_fracture_instance_v1(record)  # 旧 fracture_instance_v1 记录 → planar_interface
    rec = to_fracture_instance_v1(obj)       # 反向，仅平面型
"""
from __future__ import annotations

import copy
import json
import math

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "SCHEMA_VER",
    "LEGACY_SCHEMA_VER",
    "WATERMARK",
    "OBJECT_TYPES",
    "VISIBILITY_VALUES",
    "LABEL_STATUS_VALUES",
    "RELATION_TYPES",
    "SEGMENT_MARKS",
    "GATE_STATUSES",
    "HIDDEN_FIELD_KEYS",
    "SchemaIncompatibleError",
    "HiddenFieldError",
    "Violation",
    "PlanarGeometry",
    "ZoneGeometry",
    "RelationEdge",
    "DiscontinuityObject",
    "DiscontinuitySystem",
    "INVARIANTS",
    "from_fracture_instance_v1",
    "to_fracture_instance_v1",
    "parse_legacy_stream",
    "scan_hidden_fields",
    "assert_no_hidden_fields",
    "audit_system",
    "derive_orientation_from_normal",
    "derive_orientation_from_axis",
]

SCHEMA_VER = "g59.discontinuity_schema.v1"
LEGACY_SCHEMA_VER = "fracture_instance_v1"
WATERMARK = "CONTRACT_INFRA / ZERO_RESEARCH_READING"

OBJECT_TYPES = ("planar_interface", "zone", "cavity")
VISIBILITY_VALUES = ("observed", "observed_negative", "not_observed", "occluded", "device_failed")
LABEL_STATUS_VALUES = ("EXACT", "EXPERT_INTERPRETED", "DERIVED_GEOMETRIC", "WEAK", "UNLABELED")
RELATION_TYPES = ("intersects", "continues", "offsets", "contains", "adjacent", "barrier_candidate")
# 逐字镜像旧 fracture_instance.py 的 SEGMENT_MARKS / GATE_STATUSES（开工时读源码核实，只读不改）
SEGMENT_MARKS = ("clean", "broken", "eroded", "occluded", "merged", "grazing_excluded")
GATE_STATUSES = ("pass", "warn", "fail", "unrated")

# 任务书 §4.3 硬编码隐藏真值键清单（顺序即判据顺序，读数后禁改）
HIDDEN_FIELD_KEYS = (
    "truth_id",
    "hidden_truth",
    "true_normal",
    "true_center",
    "full_network",
    "hidden_objects",
    "oracle_label",
    "ground_truth_network",
)

UNIT_TOL = 1e-6          # 法向/轴向单位性容差（prereg 冻结）
BBOX_TOL = 1e-6          # 迹线点落在 extent bbox 内的容差（镜像旧 v1 validate_set）


class SchemaIncompatibleError(ValueError):
    """结构不兼容（旧 v1 缺字段/非法取值、公共 dict 结构损坏、校验不通过）。"""


class HiddenFieldError(SchemaIncompatibleError):
    """命中隐藏真值字段（真理隔离守卫）。"""


# ---------------------------------------------------------------------------
# 隐藏字段递归扫描（truth isolation）
# ---------------------------------------------------------------------------

def scan_hidden_fields(obj: Any, path: str = "$") -> List[dict]:
    """递归扫描 Mapping 键名，返回全部隐藏字段命中 [{path, key}]（0 命中 = 空列表）。"""
    hits: List[dict] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = "%s.%s" % (path, k)
            if isinstance(k, str) and k in HIDDEN_FIELD_KEYS:
                hits.append({"path": p, "key": k})
            hits.extend(scan_hidden_fields(v, p))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(scan_hidden_fields(v, "%s[%d]" % (path, i)))
    return hits


def assert_no_hidden_fields(obj: Any, where: str = "payload") -> None:
    """命中任何隐藏字段即抛 HiddenFieldError（列出全部命中，不静默丢）。"""
    hits = scan_hidden_fields(obj)
    if hits:
        raise HiddenFieldError(
            "%s 含隐藏真值字段 %d 处，拒绝序列化/解析（不静默丢）：%s"
            % (where, len(hits), ", ".join(h["path"] for h in hits[:8]))
        )


# ---------------------------------------------------------------------------
# 取值原语（严格结构解析；语义合法性交给 validate）
# ---------------------------------------------------------------------------

def _finite_num(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise SchemaIncompatibleError("%s 必须是数值，收到 %r" % (what, type(x).__name__))
    v = float(x)
    if not math.isfinite(v):
        raise SchemaIncompatibleError("%s 必须有限，收到 %r" % (what, v))
    return v


def _num_any(x: Any, what: str) -> float:
    """数值转换（允许 NaN/inf 进入对象，交由 validate 判红；禁止 bool/非数值）。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise SchemaIncompatibleError("%s 必须是数值，收到 %r" % (what, type(x).__name__))
    return float(x)


def _vec3(x: Any, what: str) -> Tuple[float, float, float]:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise SchemaIncompatibleError("%s 必须是长度 3 的数组，收到 %r" % (what, x))
    return (_num_any(x[0], what + "[0]"), _num_any(x[1], what + "[1]"), _num_any(x[2], what + "[2]"))


def _mapping(x: Any, what: str) -> dict:
    if not isinstance(x, Mapping):
        raise SchemaIncompatibleError("%s 必须是 Mapping，收到 %r" % (what, type(x).__name__))
    return copy.deepcopy(dict(x))


def _bbox(x: Any, what: str) -> dict:
    if not isinstance(x, Mapping):
        raise SchemaIncompatibleError("%s 必须是 Mapping" % what)
    if "bbox_min" not in x or "bbox_max" not in x:
        raise SchemaIncompatibleError("%s 缺 bbox_min/bbox_max" % what)
    return {
        "bbox_min": list(_vec3(x["bbox_min"], what + ".bbox_min")),
        "bbox_max": list(_vec3(x["bbox_max"], what + ".bbox_max")),
    }


def _norm3(v) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _isfinite_vec3(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c)) for c in v
    )


def derive_orientation_from_normal(normal) -> dict:
    """平面产状（只读派生）：dip 由无向法向定义；方位角因 v1 无向约定在 180° 内不可判，登记 None。"""
    nz = min(1.0, max(0.0, abs(float(normal[2]))))
    return {
        "kind": "derived_from_normal",
        "normal": [float(c) for c in normal],
        "dip_deg": math.degrees(math.acos(nz)),
        "dip_direction_deg": None,
        "azimuth_ambiguity_deg": 180.0,
        "convention": "undirected normal (fracture_instance_v1 sign-indefinite); dip from horizontal",
    }


def derive_orientation_from_axis(axis) -> dict:
    """区域轴向产状（只读派生）：plunge = asin(axis_z)；trend 在水平投影退化时登记 None。"""
    ax = [float(c) for c in axis]
    plunge = math.degrees(math.asin(min(1.0, max(-1.0, ax[2]))))
    hx, hy = ax[0], ax[1]
    h = math.hypot(hx, hy)
    trend = None
    if h > 1e-12:
        trend = math.degrees(math.atan2(hx, hy)) % 360.0
    return {
        "kind": "derived_from_axis",
        "axis": ax,
        "trend_deg": trend,
        "plunge_deg": plunge,
        "convention": "unit axis; trend from north clockwise; trend=None when axis is vertical",
    }


# ---------------------------------------------------------------------------
# 几何对象
# ---------------------------------------------------------------------------

@dataclass
class PlanarGeometry:
    """平面型有限面几何：n·x = offset，n 单位法向；extent_box 为轴对齐包围盒。"""

    normal: Tuple[float, float, float]
    offset: float
    extent_box: dict = field(default_factory=lambda: {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.0, 0.0, 0.0]})
    polygon_xy: Optional[list] = None
    trace_segments: list = field(default_factory=list)

    kind = "planar"

    def to_dict(self) -> dict:
        return {
            "kind": "planar",
            "normal": [float(c) for c in self.normal],
            "offset": float(self.offset),
            "extent_box": copy.deepcopy(self.extent_box),
            "polygon_xy": copy.deepcopy(self.polygon_xy),
            "trace_segments": copy.deepcopy(self.trace_segments),
        }


@dataclass
class ZoneGeometry:
    """区域型有限厚度体：中心 + 单位轴 + 宽度（严格 > 0）+ 沿轴延展长度（严格 > 0）。"""

    center: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    width: float
    extent_along_axis: float
    thickness_subzones: Optional[dict] = None

    kind = "zone"

    def to_dict(self) -> dict:
        return {
            "kind": "zone",
            "center": [float(c) for c in self.center],
            "axis": [float(c) for c in self.axis],
            "width": float(self.width),
            "extent_along_axis": float(self.extent_along_axis),
            "thickness_subzones": copy.deepcopy(self.thickness_subzones),
        }


# ---------------------------------------------------------------------------
# 关系边
# ---------------------------------------------------------------------------

@dataclass
class RelationEdge:
    src_id: str
    dst_id: str
    relation_type: str
    confidence: float
    provenance: dict = field(default_factory=dict)
    label_status: str = "UNLABELED"

    def to_dict(self) -> dict:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation_type": self.relation_type,
            "confidence": float(self.confidence),
            "provenance": copy.deepcopy(self.provenance),
            "label_status": self.label_status,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "RelationEdge":
        if not isinstance(d, Mapping):
            raise SchemaIncompatibleError("relation edge 必须是 Mapping")
        for k in ("src_id", "dst_id", "relation_type", "confidence", "provenance", "label_status"):
            if k not in d:
                raise SchemaIncompatibleError("relation edge 缺字段 %r" % k)
        if not isinstance(d["src_id"], str) or not isinstance(d["dst_id"], str):
            raise SchemaIncompatibleError("relation edge src_id/dst_id 必须是字符串")
        return cls(
            src_id=d["src_id"],
            dst_id=d["dst_id"],
            relation_type=str(d["relation_type"]),
            confidence=_num_any(d["confidence"], "relation.confidence"),
            provenance=_mapping(d["provenance"], "relation.provenance"),
            label_status=str(d["label_status"]),
        )


# ---------------------------------------------------------------------------
# 结构对象
# ---------------------------------------------------------------------------

_REQUIRED_OBJECT_KEYS = (
    "id", "object_type", "geometry", "support_scale", "orientation", "extent",
    "quality", "visibility", "provenance", "uncertainty", "label_status",
)


@dataclass
class DiscontinuityObject:
    """S/Z/V 通用对象；geometry 必须是 PlanarGeometry / ZoneGeometry / cavity 结构位 dict。"""

    id: str
    object_type: str
    geometry: Any
    support_scale: float
    orientation: dict
    extent: dict
    quality: dict
    visibility: str
    provenance: dict
    uncertainty: dict
    label_status: str
    extensions: dict = field(default_factory=dict)

    # -- 序列化 ------------------------------------------------------------
    def _payload(self) -> dict:
        if isinstance(self.geometry, (PlanarGeometry, ZoneGeometry)):
            geom = self.geometry.to_dict()
        elif isinstance(self.geometry, Mapping):
            geom = copy.deepcopy(dict(self.geometry))
        else:
            raise SchemaIncompatibleError("geometry 类型非法：%r" % (type(self.geometry).__name__,))
        return {
            "id": self.id,
            "object_type": self.object_type,
            "geometry": geom,
            "support_scale": float(self.support_scale),
            "orientation": copy.deepcopy(self.orientation),
            "extent": copy.deepcopy(self.extent),
            "quality": copy.deepcopy(self.quality),
            "visibility": self.visibility,
            "provenance": copy.deepcopy(self.provenance),
            "uncertainty": copy.deepcopy(self.uncertainty),
            "label_status": self.label_status,
            "extensions": copy.deepcopy(self.extensions),
        }

    def to_public_dict(self) -> dict:
        """公开视图：只含白名单字段；隐藏真值命中即抛、非有限 JSON 即抛（不静默）。"""
        payload = self._payload()
        assert_no_hidden_fields(payload, where="object[%r]" % (self.id,))
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise SchemaIncompatibleError("object[%r] 公开载荷非 JSON-finite：%s" % (self.id, exc))
        return payload

    to_dict = to_public_dict

    # -- 反序列化 ----------------------------------------------------------
    @classmethod
    def from_public_dict(cls, d: Mapping) -> "DiscontinuityObject":
        assert_no_hidden_fields(d, where="object payload")
        obj = _parse_object(d)
        viol = audit_system(_system_of([obj]))["violations"]
        if viol:
            raise SchemaIncompatibleError(
                "object[%r] 校验不过：%s" % (getattr(obj, "id", "?"),
                                          "; ".join("%s@%s: %s" % (v["code"], v["path"], v["message"])
                                                    for v in viol[:6]))
            )
        return obj


def _parse_object(d: Any) -> DiscontinuityObject:
    if not isinstance(d, Mapping):
        raise SchemaIncompatibleError("对象必须是 Mapping")
    for k in _REQUIRED_OBJECT_KEYS:
        if k not in d:
            raise SchemaIncompatibleError("对象缺必填字段 %r" % k)
    g = d["geometry"]
    if not isinstance(g, Mapping):
        raise SchemaIncompatibleError("geometry 必须是 Mapping")
    kind = g.get("kind")
    if kind == "planar":
        if "normal" not in g or "offset" not in g:
            raise SchemaIncompatibleError("planar geometry 缺 normal/offset")
        geom: Any = PlanarGeometry(
            normal=_vec3(g["normal"], "geometry.normal"),
            offset=_num_any(g["offset"], "geometry.offset"),
            extent_box=_bbox(g.get("extent_box", {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.0, 0.0, 0.0]}),
                             "geometry.extent_box"),
            polygon_xy=copy.deepcopy(g.get("polygon_xy")),
            trace_segments=copy.deepcopy(g.get("trace_segments", [])),
        )
    elif kind == "zone":
        for k in ("center", "axis", "width", "extent_along_axis"):
            if k not in g:
                raise SchemaIncompatibleError("zone geometry 缺字段 %r" % k)
        sub = g.get("thickness_subzones")
        geom = ZoneGeometry(
            center=_vec3(g["center"], "geometry.center"),
            axis=_vec3(g["axis"], "geometry.axis"),
            width=_num_any(g["width"], "geometry.width"),
            extent_along_axis=_num_any(g["extent_along_axis"], "geometry.extent_along_axis"),
            thickness_subzones=None if sub is None else copy.deepcopy(dict(sub)),
        )
    elif kind == "cavity":
        geom = copy.deepcopy(dict(g))
    else:
        raise SchemaIncompatibleError("未知 geometry.kind：%r" % (kind,))
    return DiscontinuityObject(
        id=d["id"],
        object_type=d["object_type"],
        geometry=geom,
        support_scale=_num_any(d["support_scale"], "support_scale"),
        orientation=copy.deepcopy(d["orientation"]),
        extent=_mapping(d["extent"], "extent"),
        quality=_mapping(d["quality"], "quality"),
        visibility=d["visibility"],
        provenance=_mapping(d["provenance"], "provenance"),
        uncertainty=_mapping(d["uncertainty"], "uncertainty"),
        label_status=d["label_status"],
        extensions=_mapping(d.get("extensions", {}), "extensions"),
    )


def _system_of(objs: Iterable[DiscontinuityObject]) -> "DiscontinuitySystem":
    s = DiscontinuitySystem()
    for o in objs:
        if o.object_type == "zone":
            s.zones.append(o)
        elif o.object_type == "cavity":
            s.cavities.append(o)
        else:
            s.surfaces.append(o)
    return s


# ---------------------------------------------------------------------------
# 系统（S ∪ Z ∪ V + R）
# ---------------------------------------------------------------------------

@dataclass
class DiscontinuitySystem:
    surfaces: List[DiscontinuityObject] = field(default_factory=list)
    zones: List[DiscontinuityObject] = field(default_factory=list)
    cavities: List[DiscontinuityObject] = field(default_factory=list)
    relations: List[RelationEdge] = field(default_factory=list)
    scene_id: Optional[str] = None
    m_per_px: Optional[float] = None
    frame: dict = field(default_factory=dict)

    # -- 序列化 ------------------------------------------------------------
    def _payload(self) -> dict:
        return {
            "schema": SCHEMA_VER,
            "scene_id": self.scene_id,
            "m_per_px": None if self.m_per_px is None else float(self.m_per_px),
            "frame": copy.deepcopy(self.frame),
            "surfaces": [o._payload() for o in self.surfaces],
            "zones": [o._payload() for o in self.zones],
            "cavities": [o._payload() for o in self.cavities],
            "relations": [e.to_dict() for e in self.relations],
        }

    def to_public_dict(self) -> dict:
        payload = self._payload()
        assert_no_hidden_fields(payload, where="system public payload")
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise SchemaIncompatibleError("system 公开载荷非 JSON-finite：%s" % exc)
        return payload

    to_dict = to_public_dict

    # -- 反序列化 ----------------------------------------------------------
    @classmethod
    def parse_public_dict(cls, d: Mapping, *, scan_hidden: bool = True) -> "DiscontinuitySystem":
        """结构解析（不做语义 validate）；scan_hidden=True 时先做真理隔离守卫。"""
        if not isinstance(d, Mapping):
            raise SchemaIncompatibleError("system payload 必须是 Mapping")
        if scan_hidden:
            assert_no_hidden_fields(d, where="system payload")
        sys_ = cls(
            surfaces=[_parse_object(o) for o in (d.get("surfaces") or [])],
            zones=[_parse_object(o) for o in (d.get("zones") or [])],
            cavities=[_parse_object(o) for o in (d.get("cavities") or [])],
            relations=[RelationEdge.from_dict(e) for e in (d.get("relations") or [])],
            scene_id=d.get("scene_id"),
            m_per_px=None if d.get("m_per_px") is None else _num_any(d.get("m_per_px"), "m_per_px"),
            frame=_mapping(d.get("frame", {}), "frame"),
        )
        return sys_

    @classmethod
    def from_public_dict(cls, d: Mapping) -> "DiscontinuitySystem":
        """公共 dict → 系统；隐藏键必抛，语义违例必抛（fail-closed）。"""
        sys_ = cls.parse_public_dict(d, scan_hidden=True)
        viol = audit_system(sys_)["violations"]
        if viol:
            raise SchemaIncompatibleError(
                "system 校验不过（%d 项）：%s"
                % (len(viol), "; ".join("%s@%s: %s" % (v["code"], v["path"], v["message"])
                                        for v in viol[:6]))
            )
        return sys_

    # -- 校验 --------------------------------------------------------------
    def validate(self) -> List["Violation"]:
        """返回全部不变量违例（空列表 = 通过）。"""
        return audit_system(self)["violations"]

    # -- 便捷构造 ----------------------------------------------------------
    def add_surface(self, obj: DiscontinuityObject) -> "DiscontinuitySystem":
        self.surfaces.append(obj)
        return self

    def add_zone(self, obj: DiscontinuityObject) -> "DiscontinuitySystem":
        self.zones.append(obj)
        return self

    def add_relation(self, edge: RelationEdge) -> "DiscontinuitySystem":
        self.relations.append(edge)
        return self

    def all_objects(self) -> List[DiscontinuityObject]:
        return list(self.surfaces) + list(self.zones) + list(self.cavities)


# ---------------------------------------------------------------------------
# 旧 fracture_instance_v1 只读桥（字段名/取值逐条读源码核实，不凭记忆）
# ---------------------------------------------------------------------------

def _legacy_plane_offset(n, d) -> float:
    return float(d)


def from_fracture_instance_v1(record: Mapping) -> DiscontinuityObject:
    """旧 v1 记录 → planar_interface 对象（只读；不改旧模块、不迁移数据）。

    逐字段对映（源码核实）：
      id → id；plane.n → geometry.normal（必须单位，缺失/非单位/零向量一律抛，不默认零向量）；
      plane.d → geometry.offset；extent.bbox_min/max → extent + geometry.extent_box；
      extent.segments → geometry.trace_segments；quality → quality；provenance → provenance；
      bingham + schema_ver → extensions（反向原样回填，保证往返 0 丢字段）。
    v1 无 visibility/label_status/uncertainty/support_scale/orientation → 显式默认登记，见 mapping 表。
    """
    if not isinstance(record, Mapping):
        raise SchemaIncompatibleError("旧 v1 记录必须是 Mapping")
    assert_no_hidden_fields(record, where="legacy record")
    schema_ver = record.get("schema_ver")
    if schema_ver != LEGACY_SCHEMA_VER:
        raise SchemaIncompatibleError(
            "旧 v1 记录 schema_ver 必须是 %r，收到 %r" % (LEGACY_SCHEMA_VER, schema_ver)
        )
    for k in ("id", "plane", "extent", "quality", "provenance"):
        if k not in record:
            raise SchemaIncompatibleError("旧 v1 记录缺字段 %r" % k)

    plane = record["plane"]
    if not isinstance(plane, Mapping):
        raise SchemaIncompatibleError("旧 v1 plane 必须是 Mapping")
    if "n" not in plane:
        # 任务书 §7-12 的 nrm 即"法向字段"；现役 v1 真实字段名是 plane.n（源码核实）
        raise SchemaIncompatibleError("旧 v1 记录缺法向字段 plane.n（禁默认零向量）")
    n = _vec3(plane["n"], "plane.n")
    n_norm = _norm3(n)
    if not all(math.isfinite(c) for c in n):
        raise SchemaIncompatibleError("旧 v1 plane.n 含非有限值")
    if abs(n_norm - 1.0) > UNIT_TOL:
        raise SchemaIncompatibleError(
            "旧 v1 plane.n 非单位向量（模长 %.12g，容差 %g）；拒绝静默归一化" % (n_norm, UNIT_TOL)
        )
    if "d" not in plane:
        raise SchemaIncompatibleError("旧 v1 记录缺 plane.d")
    d = _finite_num(plane["d"], "plane.d")

    extent = record["extent"]
    if not isinstance(extent, Mapping):
        raise SchemaIncompatibleError("旧 v1 extent 必须是 Mapping")
    bbox = _bbox(extent, "extent")
    segs_raw = extent.get("segments", [])
    if not isinstance(segs_raw, list):
        raise SchemaIncompatibleError("旧 v1 extent.segments 必须是 list")
    segments = copy.deepcopy(segs_raw)
    _validate_legacy_segments(segments, bbox)

    quality = _legacy_quality(record["quality"])
    provenance = _mapping(record["provenance"], "provenance")
    _validate_legacy_provenance(provenance)

    bingham = record.get("bingham")
    extensions = {
        "fracture_instance_v1.schema_ver": LEGACY_SCHEMA_VER,
        "fracture_instance_v1.bingham": copy.deepcopy(bingham) if bingham is not None else {
            "M": None, "Z": None, "status": "placeholder_r185_backfill"},
    }

    diag = math.sqrt(sum((float(bbox["bbox_max"][i]) - float(bbox["bbox_min"][i])) ** 2 for i in range(3)))
    if not math.isfinite(diag) or diag <= 0.0:
        raise SchemaIncompatibleError("旧 v1 extent 包围盒退化（对角线必须 > 0）")

    obj = DiscontinuityObject(
        id=record["id"],
        object_type="planar_interface",
        geometry=PlanarGeometry(
            normal=n,
            offset=d,
            extent_box={"bbox_min": list(bbox["bbox_min"]), "bbox_max": list(bbox["bbox_max"])},
            polygon_xy=None,
            trace_segments=segments,
        ),
        support_scale=diag,
        orientation=derive_orientation_from_normal(n),
        extent={"bbox_min": list(bbox["bbox_min"]), "bbox_max": list(bbox["bbox_max"]),
                "segments": segments},
        quality=quality,
        visibility="observed",
        provenance=provenance,
        uncertainty={
            "kind": "not_provided",
            "level": "UNKNOWN",
            "note": "fracture_instance_v1 无 uncertainty 字段；按 UNKNOWN 登记（不是零不确定性声明）",
        },
        label_status="UNLABELED",
        extensions=extensions,
    )
    return obj


def _validate_legacy_segments(segments: list, bbox: dict) -> None:
    seen = set()
    for i, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            raise SchemaIncompatibleError("extent.segments[%d] 必须是 Mapping" % i)
        if "seg_id" not in seg or not isinstance(seg["seg_id"], str) or not seg["seg_id"]:
            raise SchemaIncompatibleError("extent.segments[%d] 缺非空 seg_id" % i)
        if seg["seg_id"] in seen:
            raise SchemaIncompatibleError("extent.segments 重复 seg_id %r" % seg["seg_id"])
        seen.add(seg["seg_id"])
        if "mark" not in seg or seg["mark"] not in SEGMENT_MARKS:
            raise SchemaIncompatibleError(
                "extent.segments[%d] mark 必须 ∈ %r，收到 %r" % (i, SEGMENT_MARKS, seg.get("mark"))
            )
        if "width_px" in seg:
            _finite_num(seg["width_px"], "segment.width_px")
        q = seg.get("quality")
        if q is not None:
            _legacy_quality(q)
        has_m = isinstance(seg.get("pts_m"), list) and len(seg.get("pts_m") or []) > 0
        has_px = isinstance(seg.get("pts_px"), list) and len(seg.get("pts_px") or []) > 0
        if not (has_m or has_px):
            raise SchemaIncompatibleError("extent.segments[%d] 既无 pts_m 也无 pts_px" % i)
        if has_m:
            pts = seg["pts_m"]
            if len(pts) < 2:
                raise SchemaIncompatibleError("extent.segments[%d].pts_m 少于 2 点" % i)
            for p in pts:
                if not _isfinite_vec3(p):
                    raise SchemaIncompatibleError("extent.segments[%d].pts_m 含非有限 3D 点" % i)
                for k in range(3):
                    if float(p[k]) < float(bbox["bbox_min"][k]) - BBOX_TOL or float(p[k]) > float(bbox["bbox_max"][k]) + BBOX_TOL:
                        raise SchemaIncompatibleError(
                            "extent.segments[%d].pts_m 超出 extent bbox（旧 v1 validate_set 规则）" % i)
        if has_px:
            for p in seg["pts_px"]:
                if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(
                        isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c)) for c in p):
                    raise SchemaIncompatibleError("extent.segments[%d].pts_px 含非有限 2D 点" % i)


def _legacy_quality(q: Any) -> dict:
    if not isinstance(q, Mapping):
        raise SchemaIncompatibleError("旧 v1 quality 必须是 Mapping")
    for k in ("inlier_rate", "residual_m"):
        if k not in q:
            raise SchemaIncompatibleError("旧 v1 quality 缺字段 %r" % k)
    inlier = _finite_num(q["inlier_rate"], "quality.inlier_rate")
    residual = _finite_num(q["residual_m"], "quality.residual_m")
    if not (0.0 <= inlier <= 1.0):
        raise SchemaIncompatibleError("旧 v1 quality.inlier_rate 必须 ∈ [0,1]，收到 %r" % inlier)
    if residual < 0.0:
        raise SchemaIncompatibleError("旧 v1 quality.residual_m 必须 >= 0，收到 %r" % residual)
    gate = q.get("gate", "unrated")
    if gate not in GATE_STATUSES:
        raise SchemaIncompatibleError("旧 v1 quality.gate 必须 ∈ %r，收到 %r" % (GATE_STATUSES, gate))
    return copy.deepcopy(dict(q))


def _validate_legacy_provenance(p: dict) -> None:
    for k in ("source", "method", "method_version", "created_at"):
        if k not in p:
            raise SchemaIncompatibleError("旧 v1 provenance 缺字段 %r" % k)
        if not isinstance(p[k], str):
            raise SchemaIncompatibleError("旧 v1 provenance.%s 必须是字符串" % k)


def to_fracture_instance_v1(obj: DiscontinuityObject) -> dict:
    """planar_interface 对象 → 旧 v1 记录（仅平面型；缺字段抛 SchemaIncompatibleError）。"""
    if obj.object_type != "planar_interface":
        raise SchemaIncompatibleError(
            "to_fracture_instance_v1 只支持 planar_interface，收到 %r" % (obj.object_type,)
        )
    if not isinstance(obj.geometry, PlanarGeometry):
        raise SchemaIncompatibleError("planar_interface 的 geometry 必须是 PlanarGeometry")
    geom = obj.geometry
    if not _isfinite_vec3(list(geom.normal)):
        raise SchemaIncompatibleError("geometry.normal 必须是有限 3 向量")
    if abs(_norm3(geom.normal) - 1.0) > UNIT_TOL:
        raise SchemaIncompatibleError("geometry.normal 必须是单位向量（拒绝导出非单位法向）")
    if not math.isfinite(float(geom.offset)):
        raise SchemaIncompatibleError("geometry.offset 必须有限")
    if not isinstance(obj.extent, Mapping):
        raise SchemaIncompatibleError("object.extent 必须是 Mapping")
    bbox = _bbox(obj.extent, "object.extent")
    segments = obj.extent.get("segments", [])
    if not isinstance(segments, list):
        raise SchemaIncompatibleError("object.extent.segments 必须是 list")
    q = _legacy_quality(obj.quality)
    prov = _mapping(obj.provenance, "provenance")
    _validate_legacy_provenance(prov)
    bingham = obj.extensions.get("fracture_instance_v1.bingham")
    if bingham is None:
        bingham = {"M": None, "Z": None, "status": "placeholder_r185_backfill"}
    rec = {
        "schema_ver": LEGACY_SCHEMA_VER,
        "id": obj.id,
        "plane": {"n": [float(c) for c in geom.normal], "d": float(geom.offset)},
        "bingham": copy.deepcopy(bingham),
        "extent": {
            "bbox_min": list(bbox["bbox_min"]),
            "bbox_max": list(bbox["bbox_max"]),
            "segments": copy.deepcopy(segments),
        },
        "quality": q,
        "provenance": prov,
    }
    assert_no_hidden_fields(rec, where="to_fracture_instance_v1 output")
    return rec


def parse_legacy_stream(records: Iterable[Mapping]) -> List[DiscontinuityObject]:
    """旧记录流 → 对象列表；空流一律 SystemExit（禁空对空记 PASS）。"""
    recs = list(records)
    if not recs:
        raise SystemExit("parse_legacy_stream: 空记录流，拒绝记 PASS（禁空对空）")
    return [from_fracture_instance_v1(r) for r in recs]


# ---------------------------------------------------------------------------
# 不变量校验（27 条；每条统计 n_checked / n_violations / first_violation）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message}


INVARIANTS = (
    ("INV-01", "id 非空且在 S∪Z∪V 内唯一"),
    ("INV-02", "object_type ∈ 三型枚举"),
    ("INV-03", "平面法向为有限单位向量（模长偏差 <= 1e-6，拒绝零向量/非单位）"),
    ("INV-04", "平面 offset 有限"),
    ("INV-05", "extent bbox（及平面 geometry.extent_box）为有限 3 维且 min <= max"),
    ("INV-06", "迹线段完整性（seg_id/ mark / width_px / pts 形状与 bbox 包含）"),
    ("INV-07", "区域轴为有限单位向量"),
    ("INV-08", "区域宽度严格 > 0（宽度 <= 0 必须拒绝）"),
    ("INV-09", "区域沿轴延展为有限且严格 > 0"),
    ("INV-10", "区域中心为有限 3 维"),
    ("INV-11", "thickness_subzones 仅 {core,damage}、各自 > 0 且 core+damage <= width"),
    ("INV-12", "visibility ∈ 五态枚举"),
    ("INV-13", "label_status ∈ 五态枚举"),
    ("INV-14", "quality 为 Mapping；inlier_rate∈[0,1]、residual_m>=0、gate∈枚举（在场时）"),
    ("INV-15", "support_scale 有限且 > 0"),
    ("INV-16", "orientation 为 Mapping 且派生法向/轴向与 geometry 一致"),
    ("INV-17", "uncertainty 为 Mapping 且内部数值全部有限"),
    ("INV-18", "object_type 与 geometry 类型一致（zone↔ZoneGeometry 等）"),
    ("INV-19", "cavity 状态 NOT_IMPLEMENTED（系统内空腔列表必须为空）"),
    ("INV-20", "relation_type ∈ 六型枚举"),
    ("INV-21", "关系边端点存在于 S∪Z∪V"),
    ("INV-22", "关系 confidence 有限且 ∈ [0,1]"),
    ("INV-23", "自环仅 intersects 允许（其余自环拒绝）"),
    ("INV-24", "(src_id,dst_id,relation_type) 不重复"),
    ("INV-25", "provenance 为 Mapping（对象与关系边）"),
    ("INV-26", "对象/关系边递归扫描隐藏真值字段 0 命中"),
    ("INV-27", "系统公开载荷可 JSON 序列化且无 NaN/Inf（allow_nan=False 通过）"),
)

_INV_DESC = dict(INVARIANTS)
_INV_ORDER = [iid for iid, _ in INVARIANTS]


class _Auditor:
    def __init__(self) -> None:
        self.stats: Dict[str, dict] = {
            iid: {"invariant": iid, "description": _INV_DESC[iid], "n_checked": 0,
                  "n_violations": 0, "first_violation": None}
            for iid in _INV_ORDER
        }
        self.violations: List[Violation] = []

    def check(self, iid: str, ok: bool, path: str, message: str) -> bool:
        st = self.stats[iid]
        st["n_checked"] += 1
        if not ok:
            st["n_violations"] += 1
            v = Violation(iid, path, message)
            self.violations.append(v)
            if st["first_violation"] is None:
                st["first_violation"] = v.to_dict()
        return ok


def _obj_path(scope: str, idx: int, obj) -> str:
    oid = getattr(obj, "id", "?")
    return "%s[%d]<%s>" % (scope, idx, oid)


def _all_numbers_finite(x: Any) -> bool:
    stack = [x]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Mapping):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, float):
            if not math.isfinite(cur):
                return False
        elif isinstance(cur, int) and not isinstance(cur, bool):
            continue
    return True


def _raw_object_fields(obj) -> dict:
    if isinstance(obj.geometry, (PlanarGeometry, ZoneGeometry)):
        geom = obj.geometry.to_dict()
    elif isinstance(obj.geometry, Mapping):
        geom = dict(obj.geometry)
    else:
        geom = {"geometry_repr": repr(obj.geometry)}
    return {
        "geometry": geom,
        "orientation": obj.orientation,
        "extent": obj.extent,
        "quality": obj.quality,
        "provenance": obj.provenance,
        "uncertainty": obj.uncertainty,
        "extensions": obj.extensions,
    }


def audit_system(system: DiscontinuitySystem) -> dict:
    """跑全部 27 条不变量；返回 {per_invariant, violations, n_*_total, n_invariants}。"""
    au = _Auditor()
    objs = system.all_objects()
    ids = []
    for i, o in enumerate(objs):
        p = _obj_path("objects", i, o)
        # INV-01 id 非空
        au.check("INV-01", isinstance(o.id, str) and bool(o.id.strip()), p + ".id",
                 "id 必须为非空字符串（收到 %r）" % (o.id,))
        if isinstance(o.id, str):
            ids.append((o.id, p))
        # INV-02 object_type 枚举
        au.check("INV-02", o.object_type in OBJECT_TYPES, p + ".object_type",
                 "object_type 必须 ∈ %r，收到 %r" % (OBJECT_TYPES, o.object_type))
        # geometry 相关
        is_planar = isinstance(o.geometry, PlanarGeometry)
        is_zone = isinstance(o.geometry, ZoneGeometry)
        if is_planar:
            n = o.geometry.normal
            okv = _isfinite_vec3(list(n) if isinstance(n, (list, tuple)) else n)
            au.check("INV-03", okv, p + ".geometry.normal", "平面法向必须是有限 3 向量")
            if okv:
                nrm = _norm3(n)
                au.check("INV-03", abs(nrm - 1.0) <= UNIT_TOL, p + ".geometry.normal",
                         "平面法向必须单位（模长 %.12g，容差 %g；零向量同样拒绝）" % (nrm, UNIT_TOL))
            au.check("INV-04", isinstance(o.geometry.offset, (int, float))
                     and not isinstance(o.geometry.offset, bool)
                     and math.isfinite(float(o.geometry.offset)), p + ".geometry.offset",
                     "平面 offset 必须有限")
            _check_bbox(au, "INV-05", p + ".geometry.extent_box", o.geometry.extent_box)
            _check_segments(au, "INV-06", p + ".geometry.trace_segments",
                            o.geometry.trace_segments, o.extent)
        elif is_zone:
            ax = o.geometry.axis
            okv = _isfinite_vec3(list(ax) if isinstance(ax, (list, tuple)) else ax)
            au.check("INV-07", okv, p + ".geometry.axis", "区域轴必须是有限 3 向量")
            if okv:
                nrm = _norm3(ax)
                au.check("INV-07", abs(nrm - 1.0) <= UNIT_TOL, p + ".geometry.axis",
                         "区域轴必须单位（模长 %.12g，容差 %g）" % (nrm, UNIT_TOL))
            w = o.geometry.width
            okw = isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(float(w))
            au.check("INV-08", okw and float(w) > 0.0, p + ".geometry.width",
                     "区域宽度必须有限且严格 > 0（收到 %r；宽度 <= 0 拒绝）" % (w,))
            e = o.geometry.extent_along_axis
            oke = isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(float(e))
            au.check("INV-09", oke and float(e) > 0.0, p + ".geometry.extent_along_axis",
                     "沿轴延展必须有限且严格 > 0（收到 %r）" % (e,))
            c = o.geometry.center
            au.check("INV-10", _isfinite_vec3(list(c) if isinstance(c, (list, tuple)) else c),
                     p + ".geometry.center", "区域中心必须是有限 3 向量")
            sub = o.geometry.thickness_subzones
            if sub is None:
                au.check("INV-11", True, p + ".geometry.thickness_subzones", "")
            else:
                ok = isinstance(sub, Mapping) and set(sub.keys()) <= {"core", "damage"}
                au.check("INV-11", ok, p + ".geometry.thickness_subzones",
                         "thickness_subzones 必须是 Mapping 且键 ⊆ {core,damage}，收到 %r" % (sub,))
                if ok:
                    vals = list(sub.values())
                    okn = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                              and math.isfinite(float(v)) and float(v) > 0.0 for v in vals)
                    au.check("INV-11", okn, p + ".geometry.thickness_subzones",
                             "子带厚度必须有限且 > 0，收到 %r" % (sub,))
                    if okn and okw:
                        au.check("INV-11", sum(float(v) for v in vals) <= float(w) + 0.0,
                                 p + ".geometry.thickness_subzones",
                                 "core+damage 之和必须 <= width（%r vs %r）"
                                 % (sum(float(v) for v in vals), w))
        # INV-12 / 13
        au.check("INV-12", o.visibility in VISIBILITY_VALUES, p + ".visibility",
                 "visibility 必须 ∈ %r，收到 %r" % (VISIBILITY_VALUES, o.visibility))
        au.check("INV-13", o.label_status in LABEL_STATUS_VALUES, p + ".label_status",
                 "label_status 必须 ∈ %r，收到 %r" % (LABEL_STATUS_VALUES, o.label_status))
        # INV-14 quality
        q = o.quality
        okq = isinstance(q, Mapping)
        au.check("INV-14", okq, p + ".quality", "quality 必须是 Mapping，收到 %r" % (type(q).__name__,))
        if okq:
            if "inlier_rate" in q:
                v = q["inlier_rate"]
                au.check("INV-14", isinstance(v, (int, float)) and not isinstance(v, bool)
                         and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0, p + ".quality.inlier_rate",
                         "inlier_rate 必须有限且 ∈ [0,1]，收到 %r" % (v,))
            if "residual_m" in q:
                v = q["residual_m"]
                au.check("INV-14", isinstance(v, (int, float)) and not isinstance(v, bool)
                         and math.isfinite(float(v)) and float(v) >= 0.0, p + ".quality.residual_m",
                         "residual_m 必须有限且 >= 0，收到 %r" % (v,))
            if "gate" in q:
                au.check("INV-14", q["gate"] in GATE_STATUSES, p + ".quality.gate",
                         "gate 必须 ∈ %r，收到 %r" % (GATE_STATUSES, q["gate"]))
        # INV-15 support_scale
        ss = o.support_scale
        au.check("INV-15", isinstance(ss, (int, float)) and not isinstance(ss, bool)
                 and math.isfinite(float(ss)) and float(ss) > 0.0, p + ".support_scale",
                 "support_scale 必须有限且 > 0，收到 %r" % (ss,))
        # INV-16 orientation
        ori = o.orientation
        oko = isinstance(ori, Mapping)
        au.check("INV-16", oko, p + ".orientation", "orientation 必须是 Mapping")
        if oko:
            if is_planar:
                au.check("INV-16", _isfinite_vec3(list(ori.get("normal", []))) and
                         all(abs(float(ori["normal"][k]) - float(o.geometry.normal[k])) <= 1e-12 for k in range(3)),
                         p + ".orientation.normal", "orientation.normal 必须与 geometry.normal 一致")
            elif is_zone:
                au.check("INV-16", _isfinite_vec3(list(ori.get("axis", []))) and
                         all(abs(float(ori["axis"][k]) - float(o.geometry.axis[k])) <= 1e-12 for k in range(3)),
                         p + ".orientation.axis", "orientation.axis 必须与 geometry.axis 一致")
        # INV-17 uncertainty
        unc = o.uncertainty
        au.check("INV-17", isinstance(unc, Mapping) and _all_numbers_finite(unc), p + ".uncertainty",
                 "uncertainty 必须是 Mapping 且数值全部有限")
        # INV-18 类型-几何一致
        if o.object_type == "planar_interface":
            au.check("INV-18", is_planar, p + ".geometry", "planar_interface 必须配 PlanarGeometry")
        elif o.object_type == "zone":
            au.check("INV-18", is_zone, p + ".geometry", "zone 必须配 ZoneGeometry")
        elif o.object_type == "cavity":
            au.check("INV-18", isinstance(o.geometry, Mapping) and o.geometry.get("kind") == "cavity",
                     p + ".geometry", "cavity 必须配 kind=cavity 结构位 dict")
        else:
            au.check("INV-18", False, p + ".geometry", "未知 object_type 无法匹配 geometry")
        # INV-19 cavity 拒绝（逐对象也计一次）
        au.check("INV-19", o.object_type != "cavity", p,
                 "cavity 状态 NOT_IMPLEMENTED：首期不实现空腔几何，系统内禁止 cavity 对象")
        # INV-25 provenance
        au.check("INV-25", isinstance(o.provenance, Mapping), p + ".provenance",
                 "provenance 必须是 Mapping")
        # INV-26 隐藏字段
        hits = scan_hidden_fields(_raw_object_fields(o), path=p)
        au.check("INV-26", not hits, p, "对象递归扫描命中隐藏字段 %d 处：%s"
                 % (len(hits), ", ".join(h["path"] for h in hits[:6])))
        # INV-05 对象级 extent
        _check_bbox(au, "INV-05", p + ".extent", o.extent)

    # id 唯一（系统级）
    seen: Dict[str, str] = {}
    for oid, p in ids:
        if oid in seen:
            au.check("INV-01", False, p + ".id", "重复 id %r（首次出现于 %s）" % (oid, seen[oid]))
        else:
            seen[oid] = p
            au.check("INV-01", True, p + ".id", "")

    id_set = set(seen.keys())
    rseen: Dict[tuple, int] = {}
    for i, e in enumerate(system.relations):
        p = "relations[%d]" % i
        au.check("INV-20", e.relation_type in RELATION_TYPES, p + ".relation_type",
                 "relation_type 必须 ∈ %r，收到 %r" % (RELATION_TYPES, getattr(e, "relation_type", None)))
        au.check("INV-21", e.src_id in id_set, p + ".src_id",
                 "关系边端点必须在 S∪Z∪V 内：src_id=%r" % (e.src_id,))
        au.check("INV-21", e.dst_id in id_set, p + ".dst_id",
                 "关系边端点必须在 S∪Z∪V 内：dst_id=%r" % (e.dst_id,))
        conf = e.confidence
        au.check("INV-22", isinstance(conf, (int, float)) and not isinstance(conf, bool)
                 and math.isfinite(float(conf)) and 0.0 <= float(conf) <= 1.0, p + ".confidence",
                 "confidence 必须有限且 ∈ [0,1]，收到 %r" % (conf,))
        if e.src_id == e.dst_id:
            au.check("INV-23", e.relation_type == "intersects", p + ".self_loop",
                     "自环仅 intersects 允许，收到 %r" % (e.relation_type,))
        else:
            au.check("INV-23", True, p + ".self_loop", "")
        key = (e.src_id, e.dst_id, e.relation_type)
        if key in rseen:
            au.check("INV-24", False, p, "重复关系边 %r（首次出现于 relations[%d]）" % (key, rseen[key]))
        else:
            rseen[key] = i
            au.check("INV-24", True, p, "")
        au.check("INV-25", isinstance(e.provenance, Mapping), p + ".provenance",
                 "关系边 provenance 必须是 Mapping")
        hits = scan_hidden_fields({"provenance": e.provenance, "label_status": e.label_status},
                                  path=p)
        au.check("INV-26", not hits, p, "关系边递归扫描命中隐藏字段 %d 处：%s"
                 % (len(hits), ", ".join(h["path"] for h in hits[:6])))

    # INV-19 系统级：空腔列表为空（一次系统级检查）
    au.check("INV-19", len(system.cavities) == 0, "$.cavities",
             "cavity 状态 NOT_IMPLEMENTED：系统 cavities 必须为空，收到 %d 个" % len(system.cavities))

    # INV-26 系统级 frame/scene 元数据
    hits = scan_hidden_fields({"scene_id": system.scene_id, "frame": system.frame}, path="$")
    au.check("INV-26", not hits, "$", "系统元数据扫描命中隐藏字段 %d 处" % len(hits))

    # INV-27 JSON-finite（allow_nan=False 通过）
    try:
        json.dumps(system._payload(), ensure_ascii=False, allow_nan=False)
        au.check("INV-27", True, "$", "")
    except (ValueError, TypeError) as exc:
        au.check("INV-27", False, "$", "公开载荷无法以 allow_nan=False 序列化：%s" % exc)

    n_checked = sum(s["n_checked"] for s in au.stats.values())
    return {
        "schema": "g59.invariants_audit.v1",
        "n_invariants": len(_INV_ORDER),
        "per_invariant": {iid: au.stats[iid] for iid in _INV_ORDER},
        "violations": [v.to_dict() for v in au.violations],
        "n_checked_total": n_checked,
        "n_violations_total": len(au.violations),
    }


def _check_bbox(au: "_Auditor", iid: str, path: str, box: Any) -> None:
    ok = isinstance(box, Mapping) and "bbox_min" in box and "bbox_max" in box
    au.check(iid, ok, path, "bbox 必须是含 bbox_min/bbox_max 的 Mapping，收到 %r" % (box,))
    if not ok:
        return
    lo, hi = box["bbox_min"], box["bbox_max"]
    okv = _isfinite_vec3(lo) and _isfinite_vec3(hi)
    au.check(iid, okv, path, "bbox_min/bbox_max 必须是有限 3 向量")
    if okv:
        okm = all(float(lo[k]) <= float(hi[k]) + 0.0 for k in range(3))
        au.check(iid, okm, path, "bbox 必须逐轴 min <= max（收到 %r / %r）" % (lo, hi))


def _check_segments(au: "_Auditor", iid: str, path: str, segments: Any, extent: Any) -> None:
    if not isinstance(segments, list):
        au.check(iid, False, path, "trace_segments 必须是 list，收到 %r" % (type(segments).__name__,))
        return
    au.check(iid, True, path, "")
    seen = set()
    bbox = extent if isinstance(extent, Mapping) else {}
    for j, seg in enumerate(segments):
        sp = "%s[%d]" % (path, j)
        if not isinstance(seg, Mapping):
            au.check(iid, False, sp, "迹线段必须是 Mapping")
            continue
        sid = seg.get("seg_id")
        au.check(iid, isinstance(sid, str) and bool(sid), sp + ".seg_id", "seg_id 必须为非空字符串")
        if isinstance(sid, str):
            au.check(iid, sid not in seen, sp + ".seg_id", "对象内 seg_id 必须唯一，重复 %r" % (sid,))
            seen.add(sid)
        au.check(iid, seg.get("mark") in SEGMENT_MARKS, sp + ".mark",
                 "mark 必须 ∈ %r，收到 %r" % (SEGMENT_MARKS, seg.get("mark")))
        if "width_px" in seg:
            wp = seg["width_px"]
            au.check(iid, isinstance(wp, (int, float)) and not isinstance(wp, bool)
                     and math.isfinite(float(wp)) and float(wp) > 0.0, sp + ".width_px",
                     "width_px 必须有限且 > 0，收到 %r" % (wp,))
        q = seg.get("quality")
        if q is not None:
            au.check(iid, isinstance(q, Mapping), sp + ".quality", "segment.quality 必须是 Mapping")
            if isinstance(q, Mapping):
                if "inlier_rate" in q:
                    v = q["inlier_rate"]
                    au.check(iid, isinstance(v, (int, float)) and not isinstance(v, bool)
                             and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0,
                             sp + ".quality.inlier_rate", "inlier_rate 必须 ∈ [0,1]")
                if "residual_m" in q:
                    v = q["residual_m"]
                    au.check(iid, isinstance(v, (int, float)) and not isinstance(v, bool)
                             and math.isfinite(float(v)) and float(v) >= 0.0,
                             sp + ".quality.residual_m", "residual_m 必须 >= 0")
                if "gate" in q:
                    au.check(iid, q["gate"] in GATE_STATUSES, sp + ".quality.gate",
                             "gate 必须 ∈ %r" % (GATE_STATUSES,))
        has_m = isinstance(seg.get("pts_m"), list) and len(seg.get("pts_m") or []) > 0
        has_px = isinstance(seg.get("pts_px"), list) and len(seg.get("pts_px") or []) > 0
        au.check(iid, has_m or has_px, sp, "迹线段必须至少含 pts_m 或 pts_px")
        if has_m:
            pts = seg["pts_m"]
            au.check(iid, len(pts) >= 2, sp + ".pts_m", "pts_m 至少 2 点")
            for p in pts:
                au.check(iid, _isfinite_vec3(p), sp + ".pts_m", "pts_m 必须是有限 3D 点")
            if bbox and all(k in bbox for k in ("bbox_min", "bbox_max")) and _isfinite_vec3(
                    bbox["bbox_min"]) and _isfinite_vec3(bbox["bbox_max"]):
                inside = all(
                    float(bbox["bbox_min"][k]) - BBOX_TOL <= float(p[k]) <= float(bbox["bbox_max"][k]) + BBOX_TOL
                    for p in pts for k in range(3))
                au.check(iid, inside, sp + ".pts_m", "pts_m 必须落在 extent bbox 内（含 1e-6 容差）")
        if has_px:
            for p in seg["pts_px"]:
                au.check(iid, isinstance(p, (list, tuple)) and len(p) == 2 and all(
                    isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c))
                    for c in p), sp + ".pts_px", "pts_px 必须是有限 2D 点")
