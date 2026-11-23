# -*- coding: utf-8 -*-
"""G26 · CERT-0 目标契约 + CERT-1 有限枚举仪器 (新主线第一窗口)。

本模块实现《技术方案 v1.0》§2/§4.1 的七契约与 CERT-1 的完整枚举/保守投影/
证书验证工具面。纪律 (与 results/g26_cert0_cert1/prereg.json 同册):

* ``r_attach`` 无默认值: QuerySpec 缺该字段即拒运行 (TypeError 构造路 /
  ContractError from_mapping 路)。
* 端点必须以固定世界坐标给出; 整数盘索引替代被接口拒收 (§14.3-11)。
* 凡只为计算方便而设的限制一律标 ``RestrictionKind.COMPUTATIONAL_RESTRICTION``,
  禁冒充地质事实。
* 真值隔离: 识别/成本/投影函数只消费观测签名 + 模型表 (允许状态空间/标签函数),
  不读"真实状态"字段; 隐藏真值访问由 poison 对象在消费路径上触发异常。
* 五状态输出; 空可行集合 = MODEL_CONFLICT (不产生单值), 不可区分对以
  status + null 表示 (§11.1), 禁写非有限数字。
* 本模块零 json 写盘 (写出件一律在脚本侧 allow_nan=False)。
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_IDENTIFICATION = "connectivity-identification/v1"


class ContractError(ValueError):
    """契约违规 (缺字段/非法形/真值隔离违规等): 拒绝运行, 不降级。"""


class TruthAccessViolation(RuntimeError):
    """隐藏真值字段被消费路径访问: 真值隔离违规。"""


# ---------------------------------------------------------------------------
# 限制标注: 禁把计算方便冒充地质事实
# ---------------------------------------------------------------------------

class RestrictionKind(str, Enum):
    GEOLOGICAL_ASSUMPTION = "GEOLOGICAL_ASSUMPTION"
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"
    COMPUTATIONAL_RESTRICTION = "COMPUTATIONAL_RESTRICTION"
    INSTRUMENT_ASSUMPTION = "INSTRUMENT_ASSUMPTION"


@dataclass(frozen=True)
class Restriction:
    kind: RestrictionKind
    statement: str
    provenance: str

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "statement": self.statement,
                "provenance": self.provenance}


def computational_restrictions(restrictions: Sequence[Restriction]) -> List[Restriction]:
    return [r for r in restrictions if r.kind == RestrictionKind.COMPUTATIONAL_RESTRICTION]


# ---------------------------------------------------------------------------
# 契约 1: ThetaSpec —— 允许状态空间
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaSpec:
    """完整允许状态范围 (域/数量与尺寸范围/未观测规则/边界规则/几何容差)。

    任何"只为计算方便"的限制必须标 COMPUTATIONAL_RESTRICTION。
    """

    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    n_uncertain_range: Tuple[int, int]
    size_range: Tuple[float, float]
    unobserved_rule: str
    boundary_rule: str
    geometric_tolerance: float
    restrictions: Tuple[Restriction, ...]

    def validate(self) -> None:
        dom = np.asarray(self.domain, dtype=np.float64)
        if dom.shape != (3, 2) or not np.all(np.isfinite(dom)):
            raise ContractError(f"ThetaSpec.domain 形状/有限性非法: {self.domain!r}")
        if np.any(dom[:, 1] < dom[:, 0]):
            raise ContractError("ThetaSpec.domain hi < lo")
        lo, hi = self.n_uncertain_range
        if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo <= hi):
            raise ContractError(f"ThetaSpec.n_uncertain_range 非法: {self.n_uncertain_range!r}")
        slo, shi = self.size_range
        if not (np.isfinite(slo) and np.isfinite(shi) and 0.0 <= slo <= shi):
            raise ContractError(f"ThetaSpec.size_range 非法: {self.size_range!r}")
        if not (np.isfinite(self.geometric_tolerance) and self.geometric_tolerance >= 0.0):
            raise ContractError("ThetaSpec.geometric_tolerance 非法")
        for r in self.restrictions:
            if not isinstance(r, Restriction):
                raise ContractError("ThetaSpec.restrictions 含非 Restriction 项")

    def to_dict(self) -> dict:
        return {
            "domain": [list(x) for x in self.domain],
            "n_uncertain_range": list(self.n_uncertain_range),
            "size_range": list(self.size_range),
            "unobserved_rule": self.unobserved_rule,
            "boundary_rule": self.boundary_rule,
            "geometric_tolerance": float(self.geometric_tolerance),
            "restrictions": [r.to_dict() for r in self.restrictions],
        }


# ---------------------------------------------------------------------------
# 契约 2/3: ObservationDesign / ObservationRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationDesign:
    """观测设计: 类型/世界坐标/井轴轨迹/曝光范围/单位/观测核标识。"""

    design_id: str
    kind: str
    world_points: Tuple[Tuple[float, float, float], ...]
    directions: Tuple[Tuple[float, float, float], ...]
    exposure: Tuple[float, float]
    units: str
    kernel_id: str

    def validate(self) -> None:
        if not self.design_id or not self.kind or not self.units or not self.kernel_id:
            raise ContractError("ObservationDesign 标识/单位/核字段不得为空")
        pts = np.asarray(self.world_points, dtype=np.float64)
        dirs = np.asarray(self.directions, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or dirs.shape != pts.shape:
            raise ContractError(f"ObservationDesign 坐标/方向形状非法: {pts.shape}/{dirs.shape}")
        if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(dirs))):
            raise ContractError("ObservationDesign 含 NaN/inf")
        nrm = np.linalg.norm(dirs, axis=1)
        if np.any(np.abs(nrm - 1.0) > 1e-8):
            raise ContractError("ObservationDesign 方向未归一 (偏离 1 超 1e-8)")
        ex = np.asarray(self.exposure, dtype=np.float64)
        if ex.shape != (2,) or not np.all(np.isfinite(ex)) or ex[1] < ex[0]:
            raise ContractError(f"ObservationDesign.exposure 非法: {self.exposure!r}")

    def to_dict(self) -> dict:
        return {"design_id": self.design_id, "kind": self.kind,
                "world_points": [list(p) for p in self.world_points],
                "directions": [list(d) for d in self.directions],
                "exposure": list(self.exposure), "units": self.units,
                "kernel_id": self.kernel_id}


@dataclass(frozen=True)
class ObservationRecord:
    """可见记录条目; 缺测与负观测显式编码: 未曝光 != 曝光后未检出。

    * ``detected=True``  ⇒ 记录在场 (坐标/产状/深度有效)。
    * ``detected=False`` 且 ``exposure_depth`` 在场且 ``missing_reason=None``
      ⇒ 曝光后未检出 (显式负观测)。
    * ``missing_reason`` 在场 (如 NOT_EXPOSED) ⇒ 未曝光/无记录, 不得当作负观测。
    """

    design_id: str
    well_id: int
    depth: float
    position: Tuple[float, float, float]
    dip_deg: float
    dipdir_deg: float
    detected: bool
    quantization: Tuple[float, float]
    exposure_depth: Optional[Tuple[float, float]] = None
    missing_reason: Optional[str] = None

    def validate(self) -> None:
        if not isinstance(self.detected, bool):
            raise ContractError("ObservationRecord.detected 必须为 bool (禁 int 混用)")
        q = np.asarray(self.quantization, dtype=np.float64)
        if q.shape != (2,) or not np.all(np.isfinite(q)) or np.any(q <= 0.0):
            raise ContractError(f"quantization 非法: {self.quantization!r} (应为 (depth_m, angle_deg) 正格距)")
        if self.detected:
            vals = np.asarray([self.depth, *self.position, self.dip_deg, self.dipdir_deg], dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise ContractError("detected=True 的记录含 NaN/inf")
        else:
            if self.exposure_depth is None and self.missing_reason is None:
                raise ContractError("负记录必须显式标注: exposure_depth (曝光后未检出) 或 missing_reason")
            if self.exposure_depth is not None:
                ex = np.asarray(self.exposure_depth, dtype=np.float64)
                if ex.shape != (2,) or ex[1] < ex[0]:
                    raise ContractError(f"exposure_depth 非法: {self.exposure_depth!r}")

    def to_dict(self) -> dict:
        return {"design_id": self.design_id, "well_id": int(self.well_id),
                "depth": float(self.depth), "position": list(self.position),
                "dip_deg": float(self.dip_deg), "dipdir_deg": float(self.dipdir_deg),
                "detected": bool(self.detected),
                "quantization": list(self.quantization),
                "exposure_depth": list(self.exposure_depth) if self.exposure_depth else None,
                "missing_reason": self.missing_reason}


# ---------------------------------------------------------------------------
# 契约 4: NoiseSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseSpec:
    """误差分布或确定性界/参数来源/已知未知状态/适用域。"""

    kind: str
    params: Dict[str, float]
    source: str
    known: bool
    validity: str

    def validate(self) -> None:
        if not self.kind or not self.source or not self.validity:
            raise ContractError("NoiseSpec kind/source/validity 不得为空")
        for k, v in self.params.items():
            if not (isinstance(k, str) and np.isfinite(v)):
                raise ContractError(f"NoiseSpec.params 非法项: {k!r}={v!r}")
        if not isinstance(self.known, bool):
            raise ContractError("NoiseSpec.known 必须为 bool")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "source": self.source,
                "known": bool(self.known), "validity": self.validity}


# ---------------------------------------------------------------------------
# 契约 5: QuerySpec —— r_attach 无默认值; 端点固定世界坐标 (整数盘索引拒收)
# ---------------------------------------------------------------------------

WORLD_COORDINATE = "WORLD_COORDINATE"
_REJECTED_ENDPOINT_FORMS = ("DISK_INDEX", "INDEX", "INTEGER_INDEX")
_REJECTED_ENDPOINT_KEYS = ("a_index", "b_index", "A_index", "B_index", "a_disk", "b_disk")


def _check_world_point(name: str, P) -> Tuple[float, float, float]:
    if isinstance(P, bool):
        raise ContractError(f"端点 {name} 为 bool: 非法 (拒收)")
    if isinstance(P, (int, float)):
        raise ContractError(
            f"端点 {name} 为标量 {P!r}: 疑似整数盘索引替代固定世界坐标 —— 接口拒收 "
            f"(§14.3-11); 端点必须为 3 维世界坐标")
    try:
        arr = np.asarray(P, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ContractError(f"端点 {name} 不可解析为坐标: {e}") from e
    if arr.shape != (3,):
        raise ContractError(f"端点 {name} 形状非法: {arr.shape} (应为 (3,))")
    if not np.all(np.isfinite(arr)):
        raise ContractError(f"端点 {name} 含 NaN/inf")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class QuerySpec:
    """贯通目标查询规格: 固定世界坐标 A/B、目标域、r_attach、判交与附着约定。

    ``r_attach`` **无默认值**: 直接构造缺字段抛 TypeError; ``from_mapping``
    缺键抛 ContractError —— 两条路都拒运行。
    """

    query_id: str
    A: Tuple[float, float, float]
    B: Tuple[float, float, float]
    target_domain: str
    r_attach: float
    intersection_convention: str
    attachment_convention: str
    endpoint_form: str = WORLD_COORDINATE

    def validate(self) -> None:
        if self.endpoint_form in _REJECTED_ENDPOINT_FORMS:
            raise ContractError(
                f"endpoint_form={self.endpoint_form!r}: 整数盘索引替代固定世界坐标 —— 拒收 "
                f"(§14.3-11); 端点必须为固定世界坐标")
        if self.endpoint_form != WORLD_COORDINATE:
            raise ContractError(f"endpoint_form 未知: {self.endpoint_form!r}")
        if not self.query_id or not self.target_domain:
            raise ContractError("query_id/target_domain 不得为空")
        if not (np.isscalar(self.r_attach) and np.isfinite(self.r_attach) and float(self.r_attach) >= 0.0):
            raise ContractError(f"r_attach 必须非负有限标量且显式传入, 收到 {self.r_attach!r}")
        _check_world_point("A", self.A)
        _check_world_point("B", self.B)
        if not self.intersection_convention or not self.attachment_convention:
            raise ContractError("判交/附着约定必须显式写出")

    @staticmethod
    def from_mapping(m: Mapping) -> "QuerySpec":
        required = ("query_id", "A", "B", "target_domain", "r_attach",
                    "intersection_convention", "attachment_convention", "endpoint_form")
        for k in required:
            if k not in m:
                raise ContractError(
                    f"QuerySpec 缺字段 {k!r}: 拒运行 (r_attach 无默认值, 端点形必须显式声明)")
        for bad in _REJECTED_ENDPOINT_KEYS:
            if bad in m:
                raise ContractError(
                    f"检测到端点索引键 {bad!r}: 整数盘索引替代固定世界坐标 —— 拒收 (§14.3-11)")
        if m["endpoint_form"] in _REJECTED_ENDPOINT_FORMS:
            raise ContractError(
                f"endpoint_form={m['endpoint_form']!r}: 整数盘索引替代固定世界坐标 —— 拒收 (§14.3-11)")
        spec = QuerySpec(
            query_id=m["query_id"], A=m["A"], B=m["B"],
            target_domain=m["target_domain"], r_attach=m["r_attach"],
            intersection_convention=m["intersection_convention"],
            attachment_convention=m["attachment_convention"],
            endpoint_form=m["endpoint_form"],
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        return {"query_id": self.query_id, "A": list(self.A), "B": list(self.B),
                "target_domain": self.target_domain, "r_attach": float(self.r_attach),
                "intersection_convention": self.intersection_convention,
                "attachment_convention": self.attachment_convention,
                "endpoint_form": self.endpoint_form}


# ---------------------------------------------------------------------------
# 契约 5b: 首期唯一可证查询语义 (P2-S / G65) —— 准入闸
#
# 修复前 target_domain / intersection_convention / attachment_convention 只做
# 非空校验, 不进入计算 (VALIDATED_ONLY)。本节把三者升级为可机械判定的语义声明:
# 只有 PLANAR_UNION_CONNECTIVITY 的已登记声明放行, 其余一律
# UnsupportedQuerySemantics(status="UNSUPPORTED_QUERY_SEMANTICS", field=<字段名>)。
# 支持路径传给 two_point_connected 的实参保持不变 (A/B/centers/normals/radii/
# r_attach), 不触碰几何内核。
# ---------------------------------------------------------------------------

SUPPORTED_QUERY_KIND = "PLANAR_UNION_CONNECTIVITY"
UNSUPPORTED_QUERY_SEMANTICS = "UNSUPPORTED_QUERY_SEMANTICS"


class UnsupportedQuerySemantics(ContractError):
    """查询语义不在首期唯一可证语义内: 拒绝运行, 不静默接受自由文本。

    ``status`` 是机读判级码 ``UNSUPPORTED_QUERY_SEMANTICS``; ``field`` 指出
    触发拒绝的字段名 (target_domain / intersection_convention /
    attachment_convention), ``value`` 保留原始声明, 便于审计。
    """

    status = UNSUPPORTED_QUERY_SEMANTICS

    def __init__(self, field, value=None, reason=""):
        self.field = str(field)
        self.value = value
        self.reason = str(reason)
        super().__init__("%s field=%r value=%r: %s"
                         % (self.status, self.field, self.value, self.reason))


#: ``WORLD_COORDINATE box [x0,x1]^3`` 的机械解析式 (闭立方域)
_WORLD_COORDINATE_BOX_RX = re.compile(
    r"^WORLD_COORDINATE box \[([-+0-9.eE]+),([-+0-9.eE]+)\]\^3$")

#: 判交约定白名单: 逐字匹配内核实际语义 (邻盘真相交, 严格不等式)
_SUPPORTED_INTERSECTION_CONVENTIONS = {
    "connectivity_qoi two_point_connected: adjacent discs true-intersect "
    "(intersect_disk.disk_pair_intersect, strict inequality)": {
        "status": "SUPPORTED_EXACT",
        "equivalence": (
            "逐字点名内核入口 connectivity_qoi.two_point_connected 与 "
            "intersect_disk.disk_pair_intersect, 并声明 strict inequality; "
            "intersect_disk 的实际比较为严格 '<', 声明与实现一致。"),
    },
    "connectivity_qoi two_point_connected：相邻盘真相交"
    "（intersect_disk.disk_pair_intersect，严格不等式）": {
        "status": "SUPPORTED_EQUIVALENT",
        "equivalence": (
            "中文同义写法: 点名同一内核函数与同一 '严格不等式' 比较; "
            "仅语言/标点不同, 无参数、阈值或几何分支差异。"),
    },
}

#: 附着约定白名单: 逐字匹配 attach_points 的 <= r_attach (含等号)
_SUPPORTED_ATTACHMENT_CONVENTIONS = {
    "endpoint captured by disc iff point-to-disc distance <= r_attach (inclusive)": {
        "status": "SUPPORTED_EXACT",
        "equivalence": (
            "断言点到圆盘距离 <= r_attach 且含等号; connectivity_qoi.attach_points "
            "实际用 d <= r_attach, d 由 disk_point_distance 闭式给出, 一致。"),
    },
    "点盘距 <= r_attach（含等号）被盘捕获；点盘距 = "
    "sqrt(平面距^2 + max(0, 面内径向超出)^2)": {
        "status": "SUPPORTED_EQUIVALENT",
        "equivalence": (
            "给出同一闭式距离 sqrt(vn^2 + max(0, d_planar - r)^2) 与含等号捕获; "
            "与 disk_point_distance / attach_points 逐项一致, 仅语言不同。"),
    },
}


def _normalize_semantics_text(value) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def parse_target_domain(value) -> dict:
    """把 ``target_domain`` 解析为世界坐标闭立方域; 不可机械解析即拒绝。"""
    text = _normalize_semantics_text(value)
    if not text:
        raise UnsupportedQuerySemantics(
            "target_domain", value, "域声明缺失/为空: 无法机械定位 A、B 的声明域")
    m = _WORLD_COORDINATE_BOX_RX.match(text)
    if not m:
        raise UnsupportedQuerySemantics(
            "target_domain", value,
            "不是可机械解析的 'WORLD_COORDINATE box [x0,x1]^3' 世界坐标立方域声明")
    try:
        lo = float(m.group(1))
        hi = float(m.group(2))
    except (TypeError, ValueError):
        raise UnsupportedQuerySemantics(
            "target_domain", value, "域端点不可解析为浮点") from None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise UnsupportedQuerySemantics("target_domain", value, "域端点非有限")
    if hi < lo:
        raise UnsupportedQuerySemantics("target_domain", value, "域上界小于下界")
    return {"form": "WORLD_COORDINATE_BOX", "lo": [lo, lo, lo],
            "hi": [hi, hi, hi], "closed": True, "raw": value}


def _declares_strict_attachment(text: str) -> bool:
    t = text.replace("≤", "<=").replace("≦", "<=")
    return bool(re.search(r"<(?!=)", t)) or "严格小于" in t or "strictly less" in t.lower()


def _require_whitelisted(field: str, value, table: dict) -> dict:
    text = _normalize_semantics_text(value)
    if not text:
        raise UnsupportedQuerySemantics(field, value, "约定声明缺失/为空")
    hit = table.get(text)
    if hit is None:
        hint = ""
        low = text.lower()
        if field == "attachment_convention" and _declares_strict_attachment(text):
            hint = "; 声明为严格 '<' 而内核捕获是 '<=' (含等号): 边界点归属不同"
        elif field == "intersection_convention" and (
                "sphere" in low or "bounding" in low or "球" in text):
            hint = "; 声明为球近似/外包判交, 与内核真相交不同"
        raise UnsupportedQuerySemantics(
            field, value,
            "不在首期白名单内 (自由散文/未登记约定不得默认放行)" + hint)
    return {"raw": value, "normalized": text, "status": hit["status"],
            "equivalence": hit["equivalence"]}


def query_semantics_whitelist() -> dict:
    """返回首期支持语义白名单的只读副本 (审计/文档用; 计算路径不依赖它)。"""
    return {
        "query_kind": SUPPORTED_QUERY_KIND,
        "target_domain_form": "WORLD_COORDINATE box [x0,x1]^3",
        "target_domain_policy": "闭立方域; A、B 逐轴含边界",
        "intersection_convention": {
            k: dict(v) for k, v in _SUPPORTED_INTERSECTION_CONVENTIONS.items()},
        "attachment_convention": {
            k: dict(v) for k, v in _SUPPORTED_ATTACHMENT_CONVENTIONS.items()},
    }


def require_supported_query_semantics(spec: QuerySpec) -> dict:
    """首期唯一可证查询 ``PLANAR_UNION_CONNECTIVITY`` 的准入闸。

    * ``target_domain``: 解析为 ``WORLD_COORDINATE box [x0,x1]^3`` 闭立方域,
      A、B 必须落在声明域内 (越界 -> field=target_domain);
    * ``intersection_convention``: 必须声明内核实际使用的邻盘真相交/严格不等式;
    * ``attachment_convention``: 必须声明点到圆盘距离 ``<= r_attach`` (含等号)。

    任一项不满足即抛 ``UnsupportedQuerySemantics`` (status=
    ``UNSUPPORTED_QUERY_SEMANTICS``, 带字段名), 不吞异常、不回退默认值。
    通过时返回归一化语义记录; 该记录不进入 ``two_point_connected`` 实参。
    """
    if not isinstance(spec, QuerySpec):
        raise ContractError("require_supported_query_semantics 需要 QuerySpec")
    spec.validate()
    domain = parse_target_domain(spec.target_domain)
    lo = np.asarray(domain["lo"], dtype=np.float64)
    hi = np.asarray(domain["hi"], dtype=np.float64)
    for name in ("A", "B"):
        P = np.asarray(getattr(spec, name), dtype=np.float64)
        if not (np.all(P >= lo) and np.all(P <= hi)):
            raise UnsupportedQuerySemantics(
                "target_domain", spec.target_domain,
                "端点 %s=%s 越出声明域 lo=%s hi=%s (闭域, 逐轴含边界)"
                % (name, [float(x) for x in P], domain["lo"], domain["hi"]))
    inter = _require_whitelisted(
        "intersection_convention", spec.intersection_convention,
        _SUPPORTED_INTERSECTION_CONVENTIONS)
    att = _require_whitelisted(
        "attachment_convention", spec.attachment_convention,
        _SUPPORTED_ATTACHMENT_CONVENTIONS)
    return {"query_kind": SUPPORTED_QUERY_KIND, "target_domain": domain,
            "intersection_convention": inter, "attachment_convention": att,
            "endpoint_form": spec.endpoint_form}


def two_point_target(spec: QuerySpec, centers, normals, radii) -> bool:
    """固定世界坐标端点全链查询 (复用 connectivity_qoi.two_point_connected)。

    这是本单唯一合法的目标计算入口; 不接受任何盘索引端点形。
    支持路径: 先过首期语义准入闸, 再以原实参调用几何内核 (实参不变)。
    """
    if not isinstance(spec, QuerySpec):
        raise ContractError("two_point_target 需要 QuerySpec (禁止裸端点参数)")
    spec.validate()
    require_supported_query_semantics(spec)
    from fractureflow.connectivity_qoi import two_point_connected
    A = np.asarray(spec.A, dtype=np.float64)
    B = np.asarray(spec.B, dtype=np.float64)
    return bool(two_point_connected(centers, normals, radii, A, B, r_attach=float(spec.r_attach)))


# ---------------------------------------------------------------------------
# 契约 6: ActionSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    """动作规格: 位置/方向/曝光/成本单位/可执行条件/失败模型。"""

    action_id: str
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    exposure: Tuple[float, float]
    cost: float
    cost_unit: str
    executable_condition: str
    failure_model: str

    def validate(self) -> None:
        if not self.action_id or not self.cost_unit:
            raise ContractError("ActionSpec 标识/成本单位不得为空")
        if not (np.isscalar(self.cost) and np.isfinite(self.cost) and float(self.cost) >= 0.0):
            raise ContractError(f"ActionSpec.cost 非法: {self.cost!r}")
        p = np.asarray(self.position, dtype=np.float64)
        d = np.asarray(self.direction, dtype=np.float64)
        if p.shape != (3,) or d.shape != (3,) or not (np.all(np.isfinite(p)) and np.all(np.isfinite(d))):
            raise ContractError("ActionSpec 位置/方向形状或有限性非法")
        if abs(float(np.linalg.norm(d)) - 1.0) > 1e-8:
            raise ContractError("ActionSpec 方向未归一")
        ex = np.asarray(self.exposure, dtype=np.float64)
        if ex.shape != (2,) or ex[1] < ex[0]:
            raise ContractError(f"ActionSpec.exposure 非法: {self.exposure!r}")
        if not self.executable_condition or not self.failure_model:
            raise ContractError("ActionSpec 可执行条件/失败模型必须显式写出 (禁默认免费成功)")

    def to_dict(self) -> dict:
        return {"action_id": self.action_id, "position": list(self.position),
                "direction": list(self.direction), "exposure": list(self.exposure),
                "cost": float(self.cost), "cost_unit": self.cost_unit,
                "executable_condition": self.executable_condition,
                "failure_model": self.failure_model}


# ---------------------------------------------------------------------------
# 契约 7: IdentificationResult —— 五状态 + 分报字段 (schema §16.4)
# ---------------------------------------------------------------------------

class IdentificationStatus(str, Enum):
    CERTIFIED_CONNECTED = "CERTIFIED_CONNECTED"
    CERTIFIED_DISCONNECTED = "CERTIFIED_DISCONNECTED"
    WITNESSED_AMBIGUOUS = "WITNESSED_AMBIGUOUS"
    SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"
    MODEL_CONFLICT = "MODEL_CONFLICT"


class ProofKind(str, Enum):
    NONE_YET = "NONE_YET"
    COMPLETE_ENUMERATION_EXCLUSION = "COMPLETE_ENUMERATION_EXCLUSION"


class GuaranteeKind(str, Enum):
    EXACT_FINITE_COMPLETE_ENUMERATION = "EXACT_FINITE_COMPLETE_ENUMERATION"
    UNIFORM_MODEL_CONDITIONAL = "UNIFORM_MODEL_CONDITIONAL"
    PRIOR_PREDICTIVE_MARGINAL = "PRIOR_PREDICTIVE_MARGINAL"
    NONE = "NONE"


ATTACHED = "ALL_ATTACHED"
UNATTACHED = "ALL_UNATTACHED"
MIXED = "MIXED"


def attachment_summary(flags: Sequence[bool]) -> str:
    """一组状态的端点附着汇总; 空集拒 (空容器不得当读数)。"""
    if len(flags) == 0:
        raise ContractError("空集合的 attachment 汇总无意义: 拒发 (军规⑭ 同类)")
    s = {bool(f) for f in flags}
    if s == {True}:
        return ATTACHED
    if s == {False}:
        return UNATTACHED
    return MIXED


@dataclass
class IdentificationResult:
    model_id: str
    query_id: str
    observation_id: str
    guarantee_kind: str
    alpha_total: Optional[float]
    status: str
    target_set: Tuple[int, ...]
    attachment_status: str
    proof_kind: str
    witnesses: List[dict]
    unresolved_region_count: Optional[int]
    search_complete: bool
    numeric_semantics: str
    source_fingerprints: Dict[str, str]
    reason: str
    attachment_counts: Dict[str, int] = field(default_factory=dict)
    schema: str = SCHEMA_IDENTIFICATION

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "query_id": self.query_id,
            "observation_id": self.observation_id,
            "guarantee_kind": self.guarantee_kind,
            "alpha_total": self.alpha_total,
            "status": self.status,
            "target_set": list(self.target_set),
            "attachment_status": self.attachment_status,
            "attachment_counts": dict(self.attachment_counts),
            "proof_kind": self.proof_kind,
            "witnesses": [dict(w) for w in self.witnesses],
            "unresolved_region_count": self.unresolved_region_count,
            "search_complete": bool(self.search_complete),
            "numeric_semantics": self.numeric_semantics,
            "source_fingerprints": dict(self.source_fingerprints),
            "reason": self.reason,
        }


def validate_identification_result(doc: Mapping) -> None:
    """§16.4 条件必填检查: null/空容器不是任何科学断言的 PASS 证据。"""
    if doc.get("schema") != SCHEMA_IDENTIFICATION:
        raise ContractError(f"schema 非法: {doc.get('schema')!r}")
    for k in ("model_id", "query_id", "observation_id", "reason"):
        if not isinstance(doc.get(k), str) or not doc.get(k):
            raise ContractError(f"必填字段 {k!r} 缺失/为空")
    st = doc.get("status")
    valid = {s.value for s in IdentificationStatus}
    if st not in valid:
        raise ContractError(f"status 非法: {st!r}")
    ts = doc.get("target_set")
    if not isinstance(ts, list) or any((not isinstance(t, int)) or isinstance(t, bool) or t not in (0, 1) for t in ts):
        raise ContractError(f"target_set 非法: {ts!r}")
    if sorted(set(ts)) != ts:
        raise ContractError(f"target_set 必须去重升序: {ts!r}")
    att = doc.get("attachment_status")
    if not (isinstance(att, str) and att.startswith("A=") and ";B=" in att):
        raise ContractError(f"attachment_status 形非法 (应 'A=<..>;B=<..>'): {att!r}")
    if not isinstance(doc.get("source_fingerprints"), dict) or not doc["source_fingerprints"]:
        raise ContractError("source_fingerprints 缺失/为空: 禁空对空")
    wit = doc.get("witnesses")
    if not isinstance(wit, list):
        raise ContractError("witnesses 必须为 list")
    labels = sorted({w.get("label") for w in wit})
    if st == IdentificationStatus.CERTIFIED_CONNECTED.value:
        if ts != [1]:
            raise ContractError("CERTIFIED_CONNECTED 必须 target_set=[1]")
        if doc.get("proof_kind") != ProofKind.COMPLETE_ENUMERATION_EXCLUSION.value:
            raise ContractError("CERTIFIED 必须带可重验排除证据 (proof_kind)")
        if not doc.get("search_complete"):
            raise ContractError("CERTIFIED 必须 search_complete=true")
        if labels != [1]:
            raise ContractError("CERTIFIED_CONNECTED 需要 label=1 的可见见证 (空容器非证据)")
    elif st == IdentificationStatus.CERTIFIED_DISCONNECTED.value:
        if ts != [0]:
            raise ContractError("CERTIFIED_DISCONNECTED 必须 target_set=[0]")
        if doc.get("proof_kind") != ProofKind.COMPLETE_ENUMERATION_EXCLUSION.value:
            raise ContractError("CERTIFIED 必须带可重验排除证据 (proof_kind)")
        if not doc.get("search_complete"):
            raise ContractError("CERTIFIED 必须 search_complete=true")
        if labels != [0]:
            raise ContractError("CERTIFIED_DISCONNECTED 需要 label=0 的可见见证 (空容器非证据)")
    elif st == IdentificationStatus.WITNESSED_AMBIGUOUS.value:
        if ts != [0, 1]:
            raise ContractError("WITNESSED_AMBIGUOUS 必须 target_set=[0,1]")
        if labels != [0, 1]:
            raise ContractError("WITNESSED_AMBIGUOUS 必须两种合法见证齐")
    elif st == IdentificationStatus.SEARCH_UNRESOLVED.value:
        if ts != [0, 1]:
            raise ContractError("SEARCH_UNRESOLVED 保守保留 [0,1]")
        if doc.get("search_complete"):
            raise ContractError("SEARCH_UNRESOLVED 必须 search_complete=false")
    elif st == IdentificationStatus.MODEL_CONFLICT.value:
        if ts != []:
            raise ContractError("MODEL_CONFLICT 的 target_set 必须为空 (计入冲突, 不算单值)")
    alpha = doc.get("alpha_total")
    if alpha is not None and not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
                                  and np.isfinite(float(alpha))):
        raise ContractError(f"alpha_total 非法: {alpha!r}")


def build_identification_result(*, model_id: str, query_id: str, observation_id: str,
                                status: str, target_set: Sequence[int],
                                attach_a: str, attach_b: str,
                                attach_counts: Optional[Dict[str, int]] = None,
                                guarantee_kind: str = GuaranteeKind.EXACT_FINITE_COMPLETE_ENUMERATION.value,
                                alpha_total: Optional[float] = None,
                                proof_kind: str = ProofKind.NONE_YET.value,
                                witnesses: Optional[List[dict]] = None,
                                search_complete: bool = True,
                                numeric_semantics: str = "EXACT_FINITE_COMPLETE_ENUMERATION",
                                source_fingerprints: Optional[Dict[str, str]] = None,
                                reason: str = "") -> IdentificationResult:
    res = IdentificationResult(
        model_id=model_id, query_id=query_id, observation_id=observation_id,
        guarantee_kind=guarantee_kind, alpha_total=alpha_total, status=status,
        target_set=tuple(sorted(set(int(t) for t in target_set))),
        attachment_status=f"A={attach_a};B={attach_b}",
        proof_kind=proof_kind, witnesses=witnesses or [],
        unresolved_region_count=None, search_complete=search_complete,
        numeric_semantics=numeric_semantics,
        source_fingerprints=source_fingerprints or {}, reason=reason,
        attachment_counts=attach_counts or {})
    validate_identification_result(res.to_dict())
    return res


# ---------------------------------------------------------------------------
# 有限枚举机器: 状态表 / 等价类 / 保守投影
# ---------------------------------------------------------------------------

def enumerate_states(n_bits: int) -> List[int]:
    if not (isinstance(n_bits, int) and n_bits > 0):
        raise ContractError(f"n_bits 非法: {n_bits!r}")
    return list(range(1 << n_bits))


def signature_of(state: int, read_bits: Sequence[int]) -> int:
    """状态在基线观测设计下的签名 (读位集合的压缩位掩码)。"""
    sig = 0
    for i, b in enumerate(sorted(read_bits)):
        if state & (1 << b):
            sig |= 1 << i
    return sig


def partition_by_signature(n_bits: int, read_bits: Sequence[int]) -> Dict[int, List[int]]:
    """观测等价类: 签名 → 状态列表 (完整枚举, 不抽样)。"""
    read = sorted(set(int(b) for b in read_bits))
    for b in read:
        if not (0 <= b < n_bits):
            raise ContractError(f"read_bit 越界: {b}")
    classes: Dict[int, List[int]] = {}
    for s in range(1 << n_bits):
        classes.setdefault(signature_of(s, read), []).append(s)
    return classes


def project_target_set(members: Sequence[int], labels: Sequence[int]) -> List[int]:
    """逐等价类求完整目标集合 (只许完整枚举/解析构造, 禁采样替代)。"""
    if len(members) == 0:
        raise ContractError("空成员集的目标投影无意义: 拒发")
    out = {int(labels[m]) for m in members}
    return sorted(out)


def attach_counts_of(members: Sequence[int], attach_a: Sequence[bool],
                     attach_b: Sequence[bool]) -> Dict[str, int]:
    return {
        "n": len(members),
        "a_attached_true": sum(1 for m in members if attach_a[m]),
        "b_attached_true": sum(1 for m in members if attach_b[m]),
    }


def classify_status(target_labels: Sequence[int], *, feasible: bool = True) -> str:
    """按保守目标集合给五状态 (搜索完成假设由调用方给 search_complete)。"""
    if (not feasible) or (len(target_labels) == 0):
        return IdentificationStatus.MODEL_CONFLICT.value
    ts = sorted(set(int(t) for t in target_labels))
    if ts == [1]:
        return IdentificationStatus.CERTIFIED_CONNECTED.value
    if ts == [0]:
        return IdentificationStatus.CERTIFIED_DISCONNECTED.value
    if ts == [0, 1]:
        return IdentificationStatus.WITNESSED_AMBIGUOUS.value
    raise ContractError(f"目标集合非法: {ts!r}")


@dataclass
class FiniteBinaryTable:
    """A 层式二值状态空间的模型表 (识别函数只消费本表 + 观测签名)。"""

    n_bits: int
    labels: Tuple[int, ...]                     # C(s) 每状态
    attach_a: Tuple[bool, ...]
    attach_b: Tuple[bool, ...]
    read_bits: Tuple[int, ...]                  # 基线观测设计读位
    model_id: str = "FINITE-BINARY"

    def validate(self) -> None:
        n = 1 << self.n_bits
        for name, arr in (("labels", self.labels), ("attach_a", self.attach_a), ("attach_b", self.attach_b)):
            if len(arr) != n:
                raise ContractError(f"表 {name} 长度 {len(arr)} != 2^{self.n_bits}")
        if any(int(v) not in (0, 1) for v in self.labels):
            raise ContractError("labels 只允许 0/1")

    def members_of_signature(self, sig: int) -> List[int]:
        return [s for s in range(1 << self.n_bits) if signature_of(s, self.read_bits) == sig]


def identify_from_signature(table: FiniteBinaryTable, signature: int, *,
                            query_id: str, observation_id: str,
                            source_fingerprints: Optional[Dict[str, str]] = None,
                            reason: str = "") -> IdentificationResult:
    """识别消费路径 (真值隔离): 只读签名 + 模型表; 不读任何"真实状态"字段。

    这是被 poison 控制检验的"实际消费路径": 观测对象里的隐藏真值字段
    不会出现在本函数任何读路径上。
    """
    table.validate()
    members = table.members_of_signature(signature)
    if len(members) == 0:
        res = build_identification_result(
            model_id=table.model_id, query_id=query_id, observation_id=observation_id,
            status=IdentificationStatus.MODEL_CONFLICT.value, target_set=[],
            attach_a=UNATTACHED, attach_b=UNATTACHED,
            search_complete=True,
            source_fingerprints=source_fingerprints or {"table": table.model_id},
            reason=reason or "观测签名不在允许状态空间像内 (空可行集合)")
        return res
    ts = project_target_set(members, table.labels)
    status = classify_status(ts)
    a_sum = attachment_summary([table.attach_a[m] for m in members])
    b_sum = attachment_summary([table.attach_b[m] for m in members])
    witnesses: List[dict] = []
    if status == IdentificationStatus.WITNESSED_AMBIGUOUS.value:
        w0 = next(m for m in members if table.labels[m] == 0)
        w1 = next(m for m in members if table.labels[m] == 1)
        witnesses = [{"label": 0, "state_id": w0}, {"label": 1, "state_id": w1}]
    elif status == IdentificationStatus.CERTIFIED_CONNECTED.value:
        witnesses = [{"label": 1, "state_id": members[0]}]
    elif status == IdentificationStatus.CERTIFIED_DISCONNECTED.value:
        witnesses = [{"label": 0, "state_id": members[0]}]
    return build_identification_result(
        model_id=table.model_id, query_id=query_id, observation_id=observation_id,
        status=status, target_set=ts, attach_a=a_sum, attach_b=b_sum,
        attach_counts=attach_counts_of(members, table.attach_a, table.attach_b),
        proof_kind=(ProofKind.COMPLETE_ENUMERATION_EXCLUSION.value
                    if status.startswith("CERTIFIED") else ProofKind.NONE_YET.value),
        witnesses=witnesses, search_complete=True,
        source_fingerprints=source_fingerprints or {"table": table.model_id},
        reason=reason or f"完整枚举 {len(members)} 个一致状态的目标投影")


# ---------------------------------------------------------------------------
# 成本参考: B*_batch 覆盖问题 + V(H) 最优最坏决策树 (§11.1-11.2)
# 代价一律以预量化整数 (cost_int) 进入比较, 禁用浮点直接比较。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostAction:
    action_id: str
    cost_int: int

    def __post_init__(self):
        if not self.action_id or not isinstance(self.cost_int, int) or self.cost_int <= 0:
            raise ContractError(f"CostAction 非法: {self.action_id!r}/{self.cost_int!r}")


def outcome_profiles(members: Sequence[int], actions: Sequence[CostAction],
                     outcome_fn: Callable[[int, str], object]) -> Tuple[tuple, ...]:
    return tuple(tuple(outcome_fn(m, a.action_id) for a in actions) for m in members)


def collapse_by_profiles(members: Sequence[int], labels_of: Callable[[int], int],
                         actions: Sequence[CostAction],
                         outcome_fn: Callable[[int, str], object]) -> List[dict]:
    """成员按全池结果向量折叠; 折叠组内混标签 ⇒ 任何动作都不可区分。"""
    groups: Dict[tuple, dict] = {}
    for i, m in enumerate(members):
        prof = tuple(outcome_fn(m, a.action_id) for a in actions)
        g = groups.setdefault(prof, {"profile": prof, "members": [], "labels": set()})
        g["members"].append(int(m))
        g["labels"].add(int(labels_of(m)))
    out = []
    for g in groups.values():
        g["labels"] = tuple(sorted(g["labels"]))
        g["members"] = tuple(g["members"])
        out.append(g)
    return out


def _unseparable_pairs(members: Sequence[int], labels_of: Callable[[int], int],
                       groups: Sequence[dict]) -> Tuple[int, List[Tuple[int, int]]]:
    """不可区分对计数 + 前 32 对示例 (组内混标签的跨答案对)。"""
    count = 0
    examples: List[Tuple[int, int]] = []
    for g in groups:
        if len(g["labels"]) <= 1:
            continue
        w0 = [m for m in g["members"] if labels_of(m) == 0]
        w1 = [m for m in g["members"] if labels_of(m) == 1]
        count += len(w0) * len(w1)
        for x in w0:
            for y in w1:
                if len(examples) < 32:
                    examples.append((int(x), int(y)))
    return count, examples


def _pair_masks(members: Sequence[int], labels_of: Callable[[int], int],
                actions: Sequence[CostAction],
                profiles: Sequence[tuple]) -> dict:
    """跨答案对的分离掩码 (Python 大整数位图)。"""
    idx = 0
    action_masks = {a.action_id: 0 for a in actions}
    full = 0
    unsep_count = 0
    unsep_examples: List[Tuple[int, int]] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if labels_of(members[i]) == labels_of(members[j]):
                continue
            full |= 1 << idx
            seps = [a for k, a in enumerate(actions) if profiles[i][k] != profiles[j][k]]
            if not seps:
                unsep_count += 1
                if len(unsep_examples) < 32:
                    unsep_examples.append((int(members[i]), int(members[j])))
            for a in seps:
                action_masks[a.action_id] |= 1 << idx
            idx += 1
    return {"n_pairs": idx, "full_mask": full, "action_masks": action_masks,
            "unsep_count": unsep_count, "unsep_examples": unsep_examples}


def minimum_batch_cost(members: Sequence[int], labels_of: Callable[[int], int],
                       actions: Sequence[CostAction],
                       outcome_fn: Callable[[int, str], object]) -> dict:
    """B*_batch: 覆盖全部跨答案对的最小动作子集代价 (§11.1)。

    小动作集 (≤20) 全枚举求最优; 不可区分对 ⇒ status + cost null (禁非有限数)。
    """
    if len(actions) > 20:
        raise ContractError("minimum_batch_cost 全枚举上限 20 动作; 大集请用 greedy 路")
    profiles = outcome_profiles(members, actions, outcome_fn)
    pm = _pair_masks(members, labels_of, actions, profiles)
    if pm["n_pairs"] == 0:
        return {"status": "NO_CROSS_PAIRS", "cost_int": 0, "chosen": [],
                "uncovered_count": 0, "uncovered_examples": [],
                "subsets_checked": 0, "method": "exhaustive"}
    if pm["unsep_count"] > 0:
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "chosen": [],
                "uncovered_count": pm["unsep_count"], "uncovered_examples": pm["unsep_examples"],
                "subsets_checked": 0, "method": "exhaustive"}
    n = len(actions)
    subs = []
    for mask in range(1, 1 << n):
        ids = tuple(sorted(actions[k].action_id for k in range(n) if mask & (1 << k)))
        cost = sum(actions[k].cost_int for k in range(n) if mask & (1 << k))
        subs.append((cost, ids, mask))
    subs.sort(key=lambda t: (t[0], t[1]))
    checked = 0
    for cost, ids, mask in subs:
        checked += 1
        m = 0
        for k in range(n):
            if mask & (1 << k):
                m |= pm["action_masks"][actions[k].action_id]
        if m == pm["full_mask"]:
            return {"status": "OPTIMAL", "cost_int": cost, "chosen": list(ids),
                    "uncovered_count": 0, "uncovered_examples": [],
                    "subsets_checked": checked, "method": "exhaustive",
                    "optimality": "ascending_cost_full_enumeration"}
    return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "chosen": [],
            "uncovered_count": pm["unsep_count"], "uncovered_examples": pm["unsep_examples"],
            "subsets_checked": checked, "method": "exhaustive"}


def greedy_batch_cost(members: Sequence[int], labels_of: Callable[[int], int],
                      actions: Sequence[CostAction],
                      outcome_fn: Callable[[int, str], object]) -> dict:
    """加权覆盖贪心 (大动作集路; 本单规模下作为交叉核对上报, 不得替代全枚举)。"""
    profiles = outcome_profiles(members, actions, outcome_fn)
    pm = _pair_masks(members, labels_of, actions, profiles)
    if pm["n_pairs"] == 0:
        return {"status": "NO_CROSS_PAIRS", "cost_int": 0, "chosen": [], "method": "greedy"}
    if pm["unsep_count"] > 0:
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "chosen": [],
                "method": "greedy"}
    uncovered = pm["full_mask"]
    cost = 0
    chosen: List[str] = []
    remaining = dict(pm["action_masks"])
    while uncovered:
        best = None
        for aid, m in sorted(remaining.items()):
            gain = bin(m & uncovered).count("1")
            if gain == 0:
                continue
            a = next(x for x in actions if x.action_id == aid)
            ratio = gain / a.cost_int
            if best is None or ratio > best[0] or (ratio == best[0] and aid < best[1]):
                best = (ratio, aid, a.cost_int, gain)
        if best is None:
            return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None,
                    "chosen": chosen, "method": "greedy"}
        _, aid, ac, _ = best
        chosen.append(aid)
        cost += ac
        uncovered &= ~remaining[aid]
        remaining.pop(aid)
    return {"status": "GREEDY_FEASIBLE", "cost_int": cost, "chosen": chosen, "method": "greedy"}


def _group_branch_partition(group_ids: Sequence[int], groups: Sequence[dict],
                            action_index: int) -> Dict[object, List[int]]:
    branch: Dict[object, List[int]] = {}
    for g in group_ids:
        out = groups[g]["profile"][action_index]
        branch.setdefault(out, []).append(g)
    return branch


def optimal_worst_case_cost(members: Sequence[int], labels_of: Callable[[int], int],
                            actions: Sequence[CostAction],
                            outcome_fn: Callable[[int, str], object],
                            include_tree: bool = False) -> dict:
    """V(H): 最优最坏成本决策树 (§11.2); 只枚举非空可行分支, 空结局不当免费成功。"""
    actions = sorted(actions, key=lambda a: a.action_id)
    groups = collapse_by_profiles(members, labels_of, actions, outcome_fn)
    mixed = [g for g in groups if len(g["labels"]) > 1]
    if mixed:
        count, examples = _unseparable_pairs(members, labels_of, groups)
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None,
                "tree": None, "uncovered_count": count, "uncovered_examples": examples,
                "method": "dp_over_collapsed_groups"}
    memo: Dict[frozenset, Tuple[int, Optional[dict]]] = {}

    def solve(gids: frozenset) -> Tuple[int, Optional[dict]]:
        labs = {l for g in gids for l in groups[g]["labels"]}
        if len(labs) <= 1:
            lab = next(iter(labs)) if labs else 0
            return 0, ({"leaf": lab} if include_tree else None)
        if gids in memo:
            return memo[gids]
        best = None
        for ai, a in enumerate(actions):
            branch = _group_branch_partition(sorted(gids), groups, ai)
            if len(branch) < 2:
                continue  # 单结局动作无信息: 跳过 (保递归良基)
            worst = 0
            subt: Dict[str, object] = {}
            ok = True
            for out, gs in sorted(branch.items(), key=lambda kv: repr(kv[0])):
                c, sub = solve(frozenset(gs))
                if c is None:
                    ok = False
                    break
                worst = max(worst, c)
                if include_tree:
                    subt[repr(out)] = sub
            if not ok:
                continue
            cand = (a.cost_int + worst, a.action_id, subt)
            if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand
        if best is None:
            memo[gids] = (None, None)  # type: ignore[assignment]
            return None, None  # type: ignore[return-value]
        tree = {"action": best[1], "branches": best[2]} if include_tree else None
        memo[gids] = (best[0], tree)
        return best[0], tree

    cost, tree = solve(frozenset(range(len(groups))))
    if cost is None:
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "tree": None,
                "method": "dp_over_collapsed_groups"}
    return {"status": "OPTIMAL", "cost_int": cost, "tree": tree,
            "n_groups": len(groups), "method": "dp_over_collapsed_groups",
            "optimality": "min_over_actions_max_over_nonempty_branches_memoized"}


def iterative_deepening_tree_cost(members: Sequence[int], labels_of: Callable[[int], int],
                                  actions: Sequence[CostAction],
                                  outcome_fn: Callable[[int, str], object],
                                  budget_cap: Optional[int] = None) -> Optional[int]:
    """V(H) 的独立复算路 (预算存在性判定 + 升序扫描), 用于与 DP 对账。"""
    actions = sorted(actions, key=lambda a: a.action_id)
    groups = collapse_by_profiles(members, labels_of, actions, outcome_fn)
    if any(len(g["labels"]) > 1 for g in groups):
        return None
    max_cost = sum(a.cost_int for a in actions)
    cap = budget_cap if budget_cap is not None else max_cost
    memo: Dict[Tuple[frozenset, int], bool] = {}

    def exists(gids: frozenset, budget: int) -> bool:
        labs = {l for g in gids for l in groups[g]["labels"]}
        if len(labs) <= 1:
            return True
        key = (gids, budget)
        if key in memo:
            return memo[key]
        ok_any = False
        for ai, a in enumerate(actions):
            if a.cost_int > budget:
                continue
            branch = _group_branch_partition(sorted(gids), groups, ai)
            if len(branch) < 2:
                continue
            if all(exists(frozenset(gs), budget - a.cost_int) for gs in branch.values()):
                ok_any = True
                break
        memo[key] = ok_any
        return ok_any

    all_groups = frozenset(range(len(groups)))
    for b in range(0, cap + 1):
        if exists(all_groups, b):
            return b
    return None


# ---------------------------------------------------------------------------
# 三维图连通 (A 层式有限图)
# ---------------------------------------------------------------------------

def graph_connected(n_nodes: int, edges: Sequence[Tuple[int, int]],
                    present: Sequence[bool], a: int, b: int) -> bool:
    """在场边子图上判 a-b 同分量 (小图 union-find, 完整枚举用)。"""
    if len(edges) != len(present):
        raise ContractError("edges 与 present 长度不一致")
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), p in zip(edges, present):
        if p:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return find(a) == find(b)


# ---------------------------------------------------------------------------
# 真值隔离 (poison) + 观测释放边界 (不释放 fracture_id)
# ---------------------------------------------------------------------------

FORBIDDEN_RELEASE_KEYS = ("frac_id", "fracture_id", "truth_sets", "obs_mask", "set_id")


class _PoisonTruth:
    """隐藏真值字段的 poison: 任何属性访问都抛 TruthAccessViolation。"""

    def __getattribute__(self, name):
        raise TruthAccessViolation(f"隐藏真值字段被访问 (属性 {name!r}): 真值隔离违规")

    def __repr__(self):
        return "<POISONED_HIDDEN_TRUTH>"

    def __bool__(self):
        raise TruthAccessViolation("隐藏真值字段被布尔化: 真值隔离违规")


def make_poisoned_truth() -> object:
    return _PoisonTruth()


def release_records(raw_records: Sequence[Mapping], positions, *, well_id: int,
                    design_id: str, whitelist: Sequence[str],
                    exposure_depth: Tuple[float, float],
                    quantization: Tuple[float, float]) -> dict:
    """生产者边界: 把 obs_forward 原始记录释放为公开记录 (不释放 fracture_id)。

    释放 = 重建新字典; ``frac_id`` 仅用于内部取位置, 不进入释放品;
    剥离字段列表随件返回作为证据 (加牙: 原始记录必含 frac_id 且释放品必无)。
    """
    pos = np.asarray(positions, dtype=np.float64)
    out_records: List[dict] = []
    stripped = set()
    for r in raw_records:
        if "frac_id" not in r:
            raise ContractError("原始记录缺 frac_id: 释放边界不成立 (来源不是 obs_forward?)")
        fid = int(r["frac_id"])
        stripped |= {k for k in r.keys() if k not in ("depth", "dip", "dipdir")}
        rec_full = {
            "design_id": design_id,
            "well_id": int(well_id),
            "depth_m": float(r["depth"]),
            "position_world": [float(round(float(x), 8)) for x in pos[fid]],
            "dip_deg": float(r["dip"]),
            "dipdir_deg": float(r["dipdir"]),
            "detected": True,
            "quantization": list(quantization),
            "exposure_depth_m": list(exposure_depth),
        }
        rec = {k: v for k, v in rec_full.items() if k in whitelist}
        for k in ("well_id", "depth_m", "detected"):
            if k not in rec:
                raise ContractError(f"白名单缺必需记录键 {k!r}: 拒发")
        out_records.append(rec)
    payload = {
        "design_id": design_id,
        "well_id": int(well_id),
        "exposure_depth_m": list(exposure_depth),
        "quantization": list(quantization),
        "detected": bool(out_records),
        "records": sorted(out_records, key=lambda r: r["depth_m"]),
    }
    assert_released_payload(payload)
    return {"payload": payload, "stripped_fields": sorted(stripped),
            "raw_had_frac_id": any("frac_id" in r for r in raw_records)}


def assert_released_payload(payload: Mapping) -> None:
    """释放载荷不得含任何隐藏状态键 (递归检查)。"""
    stack = [payload]
    while stack:
        x = stack.pop()
        if isinstance(x, Mapping):
            for k, v in x.items():
                if str(k) in FORBIDDEN_RELEASE_KEYS:
                    raise ContractError(f"释放载荷含隐藏键 {k!r}: 拒发")
                stack.append(v)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)


def record_key(rec: Mapping) -> Tuple[int, float, float, float]:
    return (int(rec["well_id"]), float(rec["depth_m"]),
            float(rec["dip_deg"]), float(rec["dipdir_deg"]))


class DictionaryDFN:
    """obs_forward 的哑型适配 (需 .centers/.normals/.radii/.sets, 可选 domain/_seed)。"""

    def __init__(self, centers, normals, radii, sets=None, domain=None, seed=None):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        if sets is None:
            sets = np.zeros(self.centers.shape[0], dtype=np.int64)
        self.sets = np.asarray(sets, dtype=np.int64)
        self.domain = domain
        self._seed = seed


def borehole_signature(dfn: DictionaryDFN,
                       well_entries: Sequence[Tuple[Sequence[float], Sequence[float]]]
                       ) -> Tuple[Tuple[Tuple[int, float, float, float], ...], ...]:
    """状态预测签名: 每井记录键多集 (well_id, depth, dip, dipdir) 升序。

    与 release_records 产出的释放记录走同一 obs_forward 路径 ⇒ 键可逐位比较。
    """
    from fractureflow.obs_forward import virtual_borehole
    sigs = []
    for w, (x0, d) in enumerate(well_entries):
        obs = virtual_borehole(np.asarray(x0, dtype=np.float64),
                               np.asarray(d, dtype=np.float64), dfn,
                               roughness_deg=0.0, seed=0)
        keys = tuple(sorted(record_key({"well_id": w, "depth_m": r["depth"],
                                        "dip_deg": r["dip"], "dipdir_deg": r["dipdir"]})
                            for r in obs.records))
        sigs.append(keys)
    return tuple(sigs)


__all__ = [
    "SCHEMA_IDENTIFICATION", "ContractError", "TruthAccessViolation",
    "RestrictionKind", "Restriction", "computational_restrictions",
    "ThetaSpec", "ObservationDesign", "ObservationRecord", "NoiseSpec",
    "QuerySpec", "ActionSpec", "two_point_target",
    "SUPPORTED_QUERY_KIND", "UNSUPPORTED_QUERY_SEMANTICS",
    "UnsupportedQuerySemantics", "parse_target_domain",
    "require_supported_query_semantics", "query_semantics_whitelist",
    "IdentificationStatus", "ProofKind", "GuaranteeKind",
    "ATTACHED", "UNATTACHED", "MIXED", "attachment_summary",
    "IdentificationResult", "validate_identification_result", "build_identification_result",
    "enumerate_states", "signature_of", "partition_by_signature",
    "project_target_set", "attach_counts_of", "classify_status",
    "FiniteBinaryTable", "identify_from_signature",
    "CostAction", "outcome_profiles", "collapse_by_profiles",
    "minimum_batch_cost", "greedy_batch_cost", "optimal_worst_case_cost",
    "iterative_deepening_tree_cost",
    "graph_connected", "FORBIDDEN_RELEASE_KEYS", "make_poisoned_truth",
    "release_records", "assert_released_payload", "record_key", "DictionaryDFN",
    "borehole_signature",
]

# ===========================================================================
# CERT-2 (G28) · 统计构造层: 检验反演 / 误差域反演 / 目标层复合检验 /
# 两类校准 / 顺序更新 / 不可能性基线 / 失败模式分类 / 四轨出件
#
# 纪律 (与 results/g28_cert2_statistical_prototype/prereg.json 同册):
# * 本层全部函数 truth-free: 只消费模型表 (允许状态空间 + 标签函数 + 预测映射)、
#   观测向量、在册噪声核与在册阈值; 零 "真实状态" 形参。
# * 覆盖审计与评估侧读数一律在线目录的独立实现件里完成, 不在本模块 (禁交叉 import);
#   本模块因此只发"假设", 不发"假设被检验过"的结论。
# * 拒绝权只属于 EXACT_FINITE_COMPLETE_ENUMERATION 的上确界。LOWER_BOUND_ONLY
#   一律无拒绝权; 试图用它排除答案即 ContractError (§6.3)。
# * 阈值来源只许 EXACT_OUTCOME_ENUMERATION / MC_JOINT_DISTRIBUTION; 学习临界值拒 (§6.1)。
# * 本模块零 json 写盘; 非有限数一律拒。
# ===========================================================================

SCHEMA_CERT2 = "connectivity-cert2-statistical/v1"

THRESHOLD_SOURCES = ("EXACT_OUTCOME_ENUMERATION", "MC_JOINT_DISTRIBUTION")
BINARY_KERNEL_FAMILIES = ("SYMMETRIC_BIT_FLIP", "ASYMMETRIC_BIT_FLIP")
CONTINUOUS_KERNEL_FAMILIES = ("GAUSSIAN_ADDITIVE_IID",)
ELLIPSOID_APPLICABILITY = (
    "误差律为高斯且各维独立", "维数固定且与预测映射同维", "协方差已知 (known=True)",
    "无数量变化 (候选槽位数 == 预测映射维数)", "无漏检/未曝光编码", "临界值非学习所得")


class SupSemantics(str, Enum):
    EXACT_FINITE_COMPLETE_ENUMERATION = "EXACT_FINITE_COMPLETE_ENUMERATION"
    LOWER_BOUND_ONLY = "LOWER_BOUND_ONLY"


class NarrowBoundCause(str, Enum):
    NARROW_BOUND_OBTAINED = "NARROW_BOUND_OBTAINED"
    STRUCTURAL_AMBIGUITY = "STRUCTURAL_AMBIGUITY"
    STATISTICAL_INVALID = "STATISTICAL_INVALID"
    COMPUTATION_INCOMPLETE = "COMPUTATION_INCOMPLETE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TrackKind(str, Enum):
    VALIDITY = "VALIDITY"
    INFORMATION = "INFORMATION"
    COMPUTATION = "COMPUTATION"
    OBSERVATION = "OBSERVATION"


@dataclass(frozen=True)
class NoiseKernel:
    """已知合成噪声核 (可完整模拟或可完整穷举; 第一版禁学习临界值)。

    * ``SYMMETRIC_BIT_FLIP``: 每坐标以 eta 独立翻转预测位。
    * ``ASYMMETRIC_BIT_FLIP``: 预测 0→观测 1 以 p01, 预测 1→观测 0 以 p10。
    * ``GAUSSIAN_ADDITIVE_IID``: Z = F_O(theta) + N(0, sigma^2 I) (误差域反演路专用)。
    """

    name: str
    family: str
    eta: Optional[float] = None
    p01: Optional[float] = None
    p10: Optional[float] = None
    sigma: Optional[float] = None
    known: bool = True
    validity: str = ""
    provenance: str = ""

    def validate(self) -> None:
        if not self.name or not self.family:
            raise ContractError("NoiseKernel.name/family 不得为空")
        if self.family not in BINARY_KERNEL_FAMILIES + CONTINUOUS_KERNEL_FAMILIES:
            raise ContractError(f"NoiseKernel.family 未在册: {self.family!r}")
        if not isinstance(self.known, bool):
            raise ContractError("NoiseKernel.known 必须为 bool")
        if not self.validity or not self.provenance:
            raise ContractError("NoiseKernel 必须显式写出适用域 validity 与出处 provenance")
        for v in (self.eta, self.p01, self.p10, self.sigma):
            if v is not None and not (np.isscalar(v) and np.isfinite(float(v))):
                raise ContractError(f"NoiseKernel 参数非有限: {v!r}")
        if self.family == "SYMMETRIC_BIT_FLIP":
            if self.eta is None or not (0.0 <= float(self.eta) <= 1.0):
                raise ContractError(f"SYMMETRIC_BIT_FLIP 需 eta∈[0,1]: {self.eta!r}")
        elif self.family == "ASYMMETRIC_BIT_FLIP":
            if self.p01 is None or self.p10 is None:
                raise ContractError("ASYMMETRIC_BIT_FLIP 需同时给出 p01 与 p10")
            if not (0.0 <= float(self.p01) <= 1.0 and 0.0 <= float(self.p10) <= 1.0):
                raise ContractError("p01/p10 必须落在 [0,1]")
        else:
            if self.sigma is None or float(self.sigma) <= 0.0:
                raise ContractError(f"GAUSSIAN_ADDITIVE_IID 需 sigma>0: {self.sigma!r}")

    def is_binary(self) -> bool:
        return self.family in BINARY_KERNEL_FAMILIES

    def mismatch_probs(self, pred_bits: np.ndarray) -> np.ndarray:
        """每个坐标『观测位 ≠ 预测位』的概率; pred_bits: (..., d) 0/1 数组。"""
        self.validate()
        pb = np.asarray(pred_bits)
        if pb.ndim == 0:
            pb = pb.reshape(1)
        if self.family == "SYMMETRIC_BIT_FLIP":
            return np.full(pb.shape, float(self.eta), dtype=np.float64)
        if self.family == "ASYMMETRIC_BIT_FLIP":
            return np.where(pb.astype(bool), float(self.p10), float(self.p01)).astype(np.float64)
        raise ContractError(f"{self.family}: 无二值翻转语义 (实值核走误差域反演路)")


def _bits_of(value: int, d: int) -> np.ndarray:
    out = np.zeros(d, dtype=np.uint8)
    for i in range(d):
        if value & (1 << i):
            out[i] = 1
    return out


def bits_matrix(d: int) -> np.ndarray:
    """全部 2^d 个 d 位向量的解码表 (行序 = 整数编码)。"""
    if not (isinstance(d, int) and 0 < d <= 20):
        raise ContractError(f"bits_matrix 维数非法: {d!r} (上限 20)")
    return np.asarray([_bits_of(v, d) for v in range(1 << d)], dtype=np.uint8)


def popcount_table(n_values: int) -> np.ndarray:
    if not (isinstance(n_values, int) and n_values > 0):
        raise ContractError(f"popcount_table 非法: {n_values!r}")
    t = np.zeros(n_values, dtype=np.int64)
    i = 1
    while i < n_values:
        t[i] = t[i >> 1] + (i & 1)
        i += 1
    return t


def hamming_statistic(pred_bits, y_bits) -> np.ndarray:
    """T_theta(y) = #{坐标: 预测位 ≠ 观测位} (大 = 不相容)。

    真实观测与每次 MC 抽样走同一个本函数 —— 统计量选择对数据零特判 (§6.1)。
    """
    p = np.asarray(pred_bits, dtype=np.uint8)
    y = np.asarray(y_bits, dtype=np.uint8)
    if p.shape[-1] != y.shape[-1] or p.shape[-1] == 0:
        raise ContractError(f"hamming_statistic 维数不符: {p.shape} vs {y.shape}")
    return np.count_nonzero(y != p, axis=-1)


def outcome_prob_table(kernel: NoiseKernel, pred_int: int, d: int) -> np.ndarray:
    """完整穷举 2^d 结局的概率 (逐坐标乘积; 非近似非采样)。"""
    kernel.validate()
    if not kernel.is_binary():
        raise ContractError(f"{kernel.family}: 二值结局穷举不适用")
    preds = _bits_of(int(pred_int), d)
    q = kernel.mismatch_probs(preds)
    outs = bits_matrix(d)
    mism = outs != preds[None, :]
    prob = np.prod(np.where(mism, q[None, :], 1.0 - q[None, :]), axis=1)
    tot = float(prob.sum())
    if not np.isclose(tot, 1.0, rtol=0.0, atol=1e-12):
        raise ContractError(f"结局概率穷举未闭合: Σ = {tot!r}")
    return prob


def statistic_pmf(kernel: NoiseKernel, pred_int: int, d: int) -> np.ndarray:
    """T 的零分布 (完整穷举按 T 分箱) + 卷积独立复算对账。"""
    prob = outcome_prob_table(kernel, pred_int, d)
    pop = popcount_table(1 << d)
    xor = np.bitwise_xor(np.arange(1 << d, dtype=np.int64), int(pred_int))
    tvals = pop[xor]
    pmf = np.bincount(tvals, weights=prob, minlength=d + 1).astype(np.float64)
    ref = _poisson_binomial_convolution(kernel.mismatch_probs(_bits_of(int(pred_int), d)))
    if not np.allclose(pmf, ref, rtol=0.0, atol=1e-12):
        raise ContractError("穷举 pmf 与卷积复算不符: 仪器自对账失败")
    return pmf


def _poisson_binomial_convolution(qs: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0])
    for q in np.asarray(qs, dtype=np.float64).ravel():
        pmf = np.convolve(pmf, np.array([1.0 - q, q]))
    return pmf


def exact_p_value(pmf: np.ndarray, t_obs: int) -> float:
    """p = P(T >= T_obs) (含并列; 完整穷举 ⇒ 精确, 无 MC 分辨率限制)。"""
    pmf = np.asarray(pmf, dtype=np.float64)
    t_obs = int(t_obs)
    if t_obs < 0 or t_obs >= pmf.shape[0]:
        raise ContractError(f"t_obs 越界: {t_obs} (pmf 长度 {pmf.shape[0]})")
    return float(pmf[t_obs:].sum())


def mc_statistic_draws(kernel: NoiseKernel, pred_int: int, d: int, B_mc: int,
                       seed: int) -> np.ndarray:
    """抽 B_mc 次独立观测 Y_b ~ P_{theta,kernel}, 再走同一 hamming_statistic 得 T_b。"""
    kernel.validate()
    if not (isinstance(B_mc, int) and B_mc >= 1):
        raise ContractError(f"B_mc 非法: {B_mc!r}")
    preds = _bits_of(int(pred_int), d)
    q = kernel.mismatch_probs(preds)
    rng = np.random.default_rng(int(seed))
    u = rng.random((B_mc, d))
    yb = (u < q[None, :]).astype(np.uint8) ^ preds[None, :]
    return hamming_statistic(preds[None, :], yb).astype(np.int64)


def mc_p_value_from_table(sorted_t: np.ndarray, t_obs: int, B_mc: int) -> float:
    """p_theta(y) = (1 + Σ_b 1[T_b >= T_obs]) / (B_mc + 1)。"""
    st = np.asarray(sorted_t, dtype=np.int64)
    if st.ndim != 1 or st.size != int(B_mc):
        raise ContractError(f"MC 表形状/长度不符: {st.shape} vs B_mc={B_mc}")
    if B_mc >= 2 and np.any(st[1:] < st[:-1]):
        raise ContractError("MC 表未升序排序")
    ge = int(B_mc) - int(np.searchsorted(st, int(t_obs), side="left"))
    return (1.0 + ge) / (float(B_mc) + 1.0)


def resolution_guard(alpha, B_mc: int) -> dict:
    """B_mc 分辨率纪律: alpha < 1/(B_mc+1) ⇒ 无可达分辨率, 出结果前如实报。"""
    a = float(alpha)
    B = int(B_mc)
    if not (np.isfinite(a) and 0.0 < a < 1.0):
        raise ContractError(f"alpha 非法: {alpha!r}")
    if B < 1:
        raise ContractError(f"B_mc 非法: {B_mc!r}")
    min_p = 1.0 / (B + 1.0)
    n_reject_levels = int(np.floor(a * (B + 1.0)))
    has_resolution = n_reject_levels >= 1
    return {"alpha": a, "B_mc": B, "min_resolvable_p": min_p,
            "reachable_rejection_levels": n_reject_levels,
            "has_resolution": bool(has_resolution),
            "verdict": "RESOLUTION_OK" if has_resolution else "NO_RESOLUTION",
            "rule": ("p <= alpha 需 Σ_b 1[T_b>=T_obs] <= alpha*(B_mc+1) - 1; "
                     "满足该式的非负整数个数 = 0 ⇒ 该检验无所需分辨率, 禁产出全绿")}


def require_threshold_source(src: str) -> None:
    if src not in THRESHOLD_SOURCES:
        raise ContractError(
            f"阈值来源 {src!r} 非法: CERT-2 第一版只许解析穷举或 MC 联合分布, 禁学习临界值 (§6.1)")


def require_resolution(alpha, B_mc: int) -> dict:
    g = resolution_guard(alpha, B_mc)
    if not g["has_resolution"]:
        raise ContractError(
            "NO_RESOLUTION: alpha={:.6g} < 1/(B_mc+1)={:.6g} ⇒ 无可达分辨率, "
            "按纪律出结果前如实报, 禁产生全绿 (任务单 §1①)".format(alpha, g["min_resolvable_p"]))
    return g


def confidence_mask(p_by_theta, alpha) -> np.ndarray:
    """候选状态在 p_theta > alpha 时保留。"""
    p = np.asarray(p_by_theta, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ContractError(f"confidence_mask 需非空一维 p 向量: shape={p.shape}")
    if not np.all(np.isfinite(p)):
        raise ContractError("p 向量含非有限项")
    if np.any((p <= 0.0) | (p > 1.0)):
        raise ContractError("p 值越出 (0,1]")
    return p > float(alpha)


def target_projection(accept_mask, labels) -> List[int]:
    m = np.asarray(accept_mask, dtype=bool)
    lab = np.asarray(labels, dtype=np.int64)
    if lab.shape != m.shape:
        raise ContractError(f"投影形状不符: {m.shape} vs {lab.shape}")
    if not m.any():
        raise ContractError("空可行集合的目标投影无意义: 拒发 (空集合不得当读数)")
    return sorted({int(v) for v in lab[m]})


def composite_sup(p_by_theta, labels, alpha, sup_semantics: str, *,
                  assert_rejection_right: bool = False) -> dict:
    """目标层复合检验 p_c^sup = sup_{theta: C_Q(theta)=c, nuisance∈H} p_{theta,eta}(y)。

    只有 sup <= alpha 才排除 c; 数值最大化所得为下界 ⇒ 无拒绝权 (§6.3)。
    """
    p = np.asarray(p_by_theta, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if p.shape != lab.shape:
        raise ContractError(f"composite_sup 形状不符: {p.shape} vs {lab.shape}")
    try:
        sem = SupSemantics(sup_semantics)
    except ValueError:
        raise ContractError(f"sup_semantics 非法: {sup_semantics!r}")
    a = float(alpha)
    rejection_allowed = (sem == SupSemantics.EXACT_FINITE_COMPLETE_ENUMERATION)
    out = {}
    for c in (0, 1):
        vals = p[lab == c]
        if vals.size == 0:
            out[f"p_sup_c{c}"] = None
            out[f"n_theta_c{c}"] = 0
            out[f"excluded_c{c}"] = False
            out[f"argmax_theta_c{c}"] = None
            continue
        i = int(np.argmax(vals))
        sup_v = float(vals[i])
        out[f"p_sup_c{c}"] = sup_v
        out[f"n_theta_c{c}"] = int(vals.size)
        out[f"argmax_theta_c{c}"] = int(np.flatnonzero(lab == c)[i])
        out[f"would_exclude_if_misused_c{c}"] = bool(sup_v <= a)
        out[f"excluded_c{c}"] = bool(rejection_allowed and sup_v <= a)
    out.update({
        "sup_semantics": sem.value,
        "rejection_allowed": bool(rejection_allowed),
        "alpha": a,
        "guarantee_note": ("EXACT_FINITE: 声明域为有限册 ⇒ 完整穷举即真上确界, 排除有保证"
                           if rejection_allowed else
                           "LOWER_BOUND_ONLY: 数值最大化为下界, 只可作信息性参照, 不得用于拒绝 (§6.3)"),
    })
    if assert_rejection_right and not rejection_allowed:
        raise ContractError(
            "拒绝权不成立: sup_semantics=LOWER_BOUND_ONLY 的上确界不得用于排除答案 (§6.3 负例抓现行)")
    return out


# ---------------------------------------------------------------------------
# ② 误差域反演 (第二条路): 已知固定维高斯误差 + 协方差的卡方椭球
# ---------------------------------------------------------------------------

def chi_square_radius(d: int, alpha: float, sigma: float) -> float:
    """r = sigma * sqrt(chi2_ppf(1-alpha, d)); 卡方分位用 scipy (禁学习临界值)。"""
    from scipy.stats import chi2
    if not (isinstance(d, int) and d >= 1):
        raise ContractError(f"维数非法: {d!r}")
    a = float(alpha)
    s = float(sigma)
    if not (np.isfinite(a) and 0.0 < a < 1.0):
        raise ContractError(f"alpha 非法: {alpha!r}")
    if not (np.isfinite(s) and s > 0.0):
        raise ContractError(f"sigma 非法: {sigma!r}")
    return float(s * np.sqrt(chi2.ppf(1.0 - a, d)))


def ellipsoid_applicability(kernel: NoiseKernel, *, d_pred: int, z_len: int,
                            observed_mask=None, certify: bool = True) -> dict:
    """显式适用域检查; 违规即列出 (certify=True 时拒运行, False 时只标注无效)。"""
    kernel.validate()
    viol = []
    if kernel.family != "GAUSSIAN_ADDITIVE_IID":
        viol.append(f"NON_GAUSSIAN_ERROR_LAW({kernel.family})")
    if not kernel.known:
        viol.append("COVARIANCE_UNKNOWN(known=False)")
    if int(d_pred) != int(z_len):
        viol.append(f"DIMENSION_MISMATCH(pred {d_pred} vs obs {z_len})")
    if observed_mask is not None:
        om = np.asarray(observed_mask, dtype=bool)
        if om.shape != (int(z_len),):
            viol.append("OBSERVED_MASK_SHAPE")
        elif not om.all():
            viol.append(f"MISSED_OBSERVATION({int((~om).sum())} 坐标未曝光)")
    out = {"applicability_domain": list(ELLIPSOID_APPLICABILITY),
           "violations": viol,
           "in_domain": not viol,
           "certify": bool(certify)}
    if viol and certify:
        raise ContractError(
            "卡方椭球误用 (超出登记适用假设域): " + ", ".join(viol)
            + " —— 未知误差/数量变化/漏检禁直套 (§6.2); 接口拒运行不降级")
    return out


def ellipsoid_accept(*, kernel: NoiseKernel, pred_matrix, z, alpha: float,
                     observed_mask=None, certify: bool = True) -> dict:
    """S_alpha(z) = {theta: F_O(theta) - z ∈ E_O}; E_O = 卡方椭球 (解析覆盖 1-alpha)。"""
    app = ellipsoid_applicability(kernel, d_pred=np.asarray(pred_matrix).shape[-1],
                                  z_len=len(np.asarray(z, dtype=np.float64)),
                                  observed_mask=observed_mask, certify=certify)
    P = np.asarray(pred_matrix, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != zz.shape[0]:
        raise ContractError(f"椭球反演形状不符: pred {P.shape} vs z {zz.shape}")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(zz))):
        raise ContractError("预测/观测含非有限项")
    d = int(P.shape[1])
    r = chi_square_radius(d, alpha, float(kernel.sigma))
    dist2 = np.einsum("ij,ij->i", P - zz[None, :], P - zz[None, :])
    mask = dist2 <= r * r
    return {"accept_mask": mask, "radius": r, "radius_sq": r * r, "d": d,
            "alpha": float(alpha), "sigma": float(kernel.sigma),
            "n_accept": int(mask.sum()), "n_theta": int(P.shape[0]),
            "analytic_coverage_claim": (1.0 - float(alpha)) if app["in_domain"] else None,
            "guarantee_kind": (GuaranteeKind.UNIFORM_MODEL_CONDITIONAL.value if app["in_domain"]
                               else GuaranteeKind.NONE.value),
            "validity": "IN_DOMAIN" if app["in_domain"] else "INVALID_OUT_OF_DOMAIN",
            "applicability": app}


# ---------------------------------------------------------------------------
# ⑤ 顺序更新: 嵌套 S_t / 错误预算账 / 真实单调性与外包单调性分列
# ---------------------------------------------------------------------------

def sequential_alpha_schedule(n_new_rounds: int, alpha_initial=0.025,
                              alpha_total=0.05) -> dict:
    """alpha_0 = alpha_initial (初始观测); alpha_t = alpha_initial/[t(t+1)] (第 t 次新观测)。

    提案设置 (任务源 §8 末段), 非既有判据; 用 Fraction 精确记账再回浮点。
    """
    from fractions import Fraction
    if not (isinstance(n_new_rounds, int) and n_new_rounds >= 0):
        raise ContractError(f"n_new_rounds 非法: {n_new_rounds!r}")
    ai, at = Fraction(str(alpha_initial)), Fraction(str(alpha_total))
    rows = [{"t": 0, "kind": "INITIAL_OBSERVATION", "alpha_frac": str(ai),
             "alpha": float(ai)}]
    used = ai
    for t in range(1, n_new_rounds + 1):
        a = ai / Fraction(t * (t + 1))
        used += a
        rows.append({"t": t, "kind": "NEW_OBSERVATION", "rule": f"0.025/[{t}*{t+1}]",
                     "alpha_frac": str(a), "alpha": float(a)})
    remaining = at - used
    return {"rows": rows, "alpha_total": float(at),
            "used": float(used), "used_frac": str(used),
            "reserved_tail": float(remaining), "reserved_tail_frac": str(remaining),
            "sum_le_total": bool(used <= at),
            "ledger_closed": bool((used + remaining) == at),
            "infinite_tail_rule": ("Σ_{t>=1} 0.025/[t(t+1)] = 0.025 (裂项) ⇒ 0.025+0.025 = 0.05 "
                                  "= alpha_total; 未花费尾部随件上报, 禁当免费预算")}


def sequential_update(s_prev_mask, accept_mask_t, labels=None) -> dict:
    """S_t = S_{t-1} ∩ {theta: p_{theta,t} > alpha_t}; 嵌套为结构性事实 ⇒ 如实标 VACUOUS。"""
    prev = np.asarray(s_prev_mask, dtype=bool)
    acc = np.asarray(accept_mask_t, dtype=bool)
    if prev.shape != acc.shape:
        raise ContractError(f"顺序更新形状不符: {prev.shape} vs {acc.shape}")
    if not prev.any():
        raise ContractError("S_{t-1} 为空集: 拒发 (空集合不得当读数; 空可行集应走 MODEL_CONFLICT 路)")
    new = prev & acc
    if not new.any():
        raise ContractError("S_t 为空集: 全部允许状态被剔除 ⇒ 调用方须发 MODEL_CONFLICT, 不得在此发单值")
    out = {"S_t": new, "n_S_prev": int(prev.sum()), "n_S_t": int(new.sum()),
           "n_rejected_this_step": int(prev.sum() - new.sum()),
           "nested_structural": bool(np.all(~new | prev)),
           "nested_assertion_kind": "VACUOUS_STRUCTURAL (交集定义即嵌套; 按铁律 8 不作证据)",
           "strictly_shrank_this_step": bool(new.sum() < prev.sum())}
    if labels is not None:
        lab = np.asarray(labels, dtype=np.int64)
        ts_prev = target_projection(prev, lab)
        ts_t = target_projection(new, lab)
        out.update({"target_set_prev": ts_prev, "target_set_t": ts_t,
                    "projection_not_widened": bool(set(ts_t) <= set(ts_prev))})
    return out


def envelope_intersect(env_prev_mask, envelope_t_mask, true_t_mask=None) -> dict:
    """上一步有效外包 ∩ 本步有效外包 ⇒ 保守保留既有信息 (§8 末段必实现路径)。"""
    e0 = np.asarray(env_prev_mask, dtype=bool)
    e1 = np.asarray(envelope_t_mask, dtype=bool)
    if e0.shape != e1.shape:
        raise ContractError("外包形状不符")
    inter = e0 & e1
    if not inter.any():
        raise ContractError("外包求交为空: 须走 MODEL_CONFLICT 路, 禁在此发单值")
    out = {"envelope_intersect": inter, "n_env_prev": int(e0.sum()),
           "n_envelope_t": int(e1.sum()), "n_intersect": int(inter.sum()),
           "replace_path_n": int(e1.sum()),
           "replace_path_widens": bool(e1.sum() > e0.sum()),
           "intersect_path_monotone": bool(inter.sum() <= e0.sum()),
           "intersect_monotone_assertion_kind": "MEASURED (交集定义下应成立; 计数由本件现算, 不硬编码)",
           "monotonicity_kind": ("真实 S_t 单调性 = 由嵌套保证; 求解外包单调性 = 只在求交路径下成立; "
                                 "两者分列禁混写 (§8)")}
    if true_t_mask is not None:
        tr = np.asarray(true_t_mask, dtype=bool)
        if tr.shape != inter.shape:
            raise ContractError("真实集形状不符")
        out["true_inside_envelope"] = bool(np.all(~tr | inter))
        out["true_inside_envelope_kind"] = "MEASURED (可失败: 外包必须保守含真)"
    return out


# ---------------------------------------------------------------------------
# ⑥ 不可能性基线 (解析下界; 穷举验证在评估侧件)
# ---------------------------------------------------------------------------

def impossibility_dual_bound(alpha, tv_distance: float = 0.0) -> dict:
    """等观测分布相反答案对: P(输出 {0,1}) >= 1 - 2*alpha - eta_TV (负值按 0)。"""
    a = float(alpha)
    tv = float(tv_distance)
    if not (np.isfinite(a) and 0.0 < a < 0.5):
        raise ContractError(f"alpha 非法: {alpha!r} (需 ∈ (0, 0.5) 否则下界无意义)")
    if not (np.isfinite(tv) and 0.0 <= tv <= 1.0):
        raise ContractError(f"TV 距离非法: {tv_distance!r}")
    raw = 1.0 - 2.0 * a - tv
    return {"alpha": a, "eta_tv": tv, "bound_raw": raw,
            "bound_reported": max(0.0, raw),
            "clamped_to_zero": bool(raw < 0.0),
            "semantics": ("TV=0 (分布完全等价) ⇒ 1-2alpha; 负值按 0 解释 = 无任何双值强制",
                          "union bound: P(含 0) 与 P(含 1) 各 >= 1-alpha ⇒ 交 >= 1-2alpha"),
            "usage": "本函数只给下界主张; 是否达到该下界由不可能性穷举件实测"}


# ---------------------------------------------------------------------------
# ⑦ 认证用 ThetaSpec 的数据依赖守卫
# ---------------------------------------------------------------------------

OBSERVATION_ANCHORED = "OBSERVATION_ANCHORED"
INDEPENDENTLY_DECLARED = "INDEPENDENTLY_DECLARED"


def certified_theta_space(*, space_id: str, provenance: str, n_theta: int,
                          certify: bool = True, proof_token: Optional[str] = None) -> dict:
    """数据依赖 (观测锚定) 的窄支撑不得未经证明缩小认证用 ThetaSpec (§6.5)。"""
    if not space_id or n_theta <= 0:
        raise ContractError("ThetaSpec 标识/状态数非法")
    if provenance not in (OBSERVATION_ANCHORED, INDEPENDENTLY_DECLARED):
        raise ContractError(f"provenance 未在册: {provenance!r}")
    anchored = (provenance == OBSERVATION_ANCHORED)
    if certify and anchored and not proof_token:
        raise ContractError(
            "观测锚定空间作认证用 ThetaSpec 但无独立证明件 (独立分割/选择后有效检验/联合保证) "
            "⇒ 拒: 不得把数据依赖的窄支撑套入固定状态空间覆盖定理 (§6.5)")
    return {"space_id": space_id, "provenance": provenance, "n_theta": int(n_theta),
            "certify": bool(certify), "proof_token_present": bool(proof_token),
            "guarantee_kind": (GuaranteeKind.UNIFORM_MODEL_CONDITIONAL.value
                               if (not anchored or proof_token) else GuaranteeKind.NONE.value),
            "note": ("观测锚定只可改善搜索效率; 认证域仍按独立声明空间出保证" if anchored else
                     "声明域独立于待检观测")}


# ---------------------------------------------------------------------------
# ⑧ 失败模式: 找不到窄界的三原因分列 (禁统一写"方法不行")
# ---------------------------------------------------------------------------

def classify_unnarrowed(*, narrow_bound_exists: bool, coverage_valid: Optional[bool],
                        search_complete: bool, opposite_witness_present: bool,
                        budget_exhausted: bool) -> dict:
    """三原因分开出件: 结构歧义 / 统计无效 / 计算未完 (§14.2 第 2 行, §0.4 分支)。"""
    if not isinstance(narrow_bound_exists, bool) or not isinstance(search_complete, bool) \
            or not isinstance(opposite_witness_present, bool) or not isinstance(budget_exhausted, bool):
        raise ContractError("classify_unnarrowed 的布尔形参必须显式给 bool (禁 int 混用)")
    if coverage_valid is not None and not isinstance(coverage_valid, bool):
        raise ContractError("coverage_valid 必须为 bool 或 None (未审计)")
    competing = []
    if coverage_valid is False:
        competing.append(NarrowBoundCause.STATISTICAL_INVALID.value)
    if budget_exhausted or not search_complete:
        competing.append(NarrowBoundCause.COMPUTATION_INCOMPLETE.value)
    if opposite_witness_present:
        competing.append(NarrowBoundCause.STRUCTURAL_AMBIGUITY.value)
    if narrow_bound_exists:
        primary = NarrowBoundCause.NARROW_BOUND_OBTAINED.value
        disposition = "本项已有窄界, 不属失败模式"
    elif competing:
        primary = competing[0]  # 固定优先序: 统计无效 > 计算未完 > 结构歧义
        disposition = {
            NarrowBoundCause.STATISTICAL_INVALID.value: "修观测/检验模型后重验; 不报告有效界 (§0.4)",
            NarrowBoundCause.COMPUTATION_INCOMPLETE.value: "改算法或收窄计算问题; 不报告物理不可识别 (§0.4)",
            NarrowBoundCause.STRUCTURAL_AMBIGUITY.value: "保留双值或改动作; 不得包装成方法优越 (§3.5)",
        }[primary]
    else:
        primary = NarrowBoundCause.INSUFFICIENT_EVIDENCE.value
        disposition = "既无窄界又无三类成因证据 ⇒ 不得宣称结构不可识别, 补证据后再判"
    return {"primary_cause": primary, "competing_causes": competing,
            "precedence": ["STATISTICAL_INVALID", "COMPUTATION_INCOMPLETE", "STRUCTURAL_AMBIGUITY"],
            "precedence_applied": bool(len(competing) > 1),
            "disposition": disposition,
            "evidence": {"coverage_valid": coverage_valid, "search_complete": search_complete,
                         "opposite_witness_present": opposite_witness_present,
                         "budget_exhausted": budget_exhausted}}


# ---------------------------------------------------------------------------
# ⑨ 四轨出件 (有效性/信息性/计算性/观测性) + 永远双值基线
# ---------------------------------------------------------------------------

RATE_NUMERATOR_DENOMINATOR_KEYS = ("n_", "count", "rate")


def _check_rate_fields(track: str, fields: Mapping) -> None:
    """每个 rate 必须有在场分母且 > 0 (禁空对空; 空比较面不得发相等结论)。

    分母键名两种形都收: ``<rate键>_den`` 与 ``<词干>_den`` (把 _rate 换成 _den)。
    """
    for k, v in fields.items():
        if k.endswith("_rate") or k.startswith("rate_"):
            cands = [k + "_den", k.replace("_rate", "_den").replace("rate_", "den_")]
            den_key = next((c for c in cands if c in fields), None)
            if den_key is None:
                raise ContractError(
                    f"轨道 {track}: 比率 {k!r} 缺分母 (合法键名 {cands}): 拒发")
            den = fields[den_key]
            if not (isinstance(den, (int, np.integer)) and int(den) > 0):
                raise ContractError(
                    f"轨道 {track}: 分母 {den_key!r}={den!r} 非正整数 ⇒ 空比较面不得发结论 (§14.3-7)")


def validate_cert2_experiment(doc: Mapping) -> None:
    """§14.1 四轨必填 + 全绿随行双读数 (铁律 8) + 基线保留。"""
    if doc.get("schema") != SCHEMA_CERT2:
        raise ContractError(f"schema 非法: {doc.get('schema')!r}")
    for k in ("experiment_id", "instrument_id", "model_id", "observation_id", "measured_at"):
        if not isinstance(doc.get(k), str) or not doc.get(k):
            raise ContractError(f"必填字段 {k!r} 缺失/为空")
    tr = doc.get("tracks")
    if not isinstance(tr, dict):
        raise ContractError("tracks 必须为 dict (四轨)")
    need = {t.value for t in TrackKind}
    missing = need - set(tr.keys())
    if missing:
        raise ContractError(f"缺轨: {sorted(missing)}")
    for name, fields in tr.items():
        if not isinstance(fields, dict) or not fields:
            raise ContractError(f"轨道 {name} 为空容器: 不得当读数")
        _check_rate_fields(name, fields)
    v = doc.get("verdict_face")
    if not isinstance(v, dict) or "all_green" not in v:
        raise ContractError("verdict_face.all_green 缺失")
    if v.get("all_green"):
        for companion in ("vacuousness", "informativeness_companion"):
            if companion not in doc:
                raise ContractError(
                    f"全绿结论缺随行 {companion!r}: 铁律 8 (任何全绿/全收必带 vacuous/信息性双读数)")
    if not doc.get("baselines_present"):
        raise ContractError("baselines_present 必须为真并登记在盘基线件 (§14.1 永远双值基线必须保留)")
    g = doc.get("guarantee")
    if not isinstance(g, dict) or g.get("guarantee_kind") not in {x.value for x in GuaranteeKind}:
        raise ContractError("guarantee.guarantee_kind 非法")
    if g.get("guarantee_kind") == GuaranteeKind.NONE.value and doc.get("claims_effective_bound"):
        raise ContractError("guarantee_kind=NONE 却宣称有效界: 拒 (覆盖无效不报有效界)")
    return None


def build_cert2_experiment(*, experiment_id: str, instrument_id: str, model_id: str,
                           observation_id: str, measured_at: str, tracks: Dict[str, dict],
                           verdict_face: dict, guarantee: dict,
                           baselines_present: bool, alpha: Optional[float] = None,
                           vacuousness: Optional[dict] = None,
                           informativeness_companion: Optional[dict] = None,
                           claims_effective_bound: bool = False,
                           extra: Optional[dict] = None) -> dict:
    doc = {"schema": SCHEMA_CERT2, "experiment_id": experiment_id,
           "instrument_id": instrument_id, "model_id": model_id,
           "observation_id": observation_id, "measured_at": measured_at,
           "alpha": alpha, "tracks": tracks, "verdict_face": verdict_face,
           "guarantee": guarantee, "baselines_present": bool(baselines_present),
           "claims_effective_bound": bool(claims_effective_bound),
           "vacuousness": vacuousness, "informativeness_companion": informativeness_companion}
    doc.update(extra or {})
    validate_cert2_experiment(doc)
    return doc


def always_double_target_set(n_items: int) -> List[int]:
    """『永远双值』基线: 有效性可好而信息性为零, 不判作算法胜利 (§14.1/§14.3-2)。"""
    if int(n_items) <= 0:
        raise ContractError("永远双值基线的比较面为空: 拒发")
    return [0, 1]


def majority_target_set(n_c0: int, n_c1: int) -> List[int]:
    """『永远多数类』基线 (§14.3-3); 少数成员在场时其单值必错 (反例族暴露)。"""
    if int(n_c0) < 0 or int(n_c1) < 0 or (int(n_c0) + int(n_c1)) == 0:
        raise ContractError("多数类基线计数非法 (空容器不得当读数)")
    return [1] if n_c1 >= n_c0 else [0]


def gamma_coverage_premises_ok(p_dual, p_contains0, p_contains1, alpha) -> bool:
    """不可能性穷举的前提检查: 两侧覆盖皆 >= 1-alpha; 不满足者不得计入候选 Γ。

    标量与向量形参都接受 (穷举面按 4^n 向量一次算, 逐行检查会慢两个量级)。
    """
    a = float(alpha)
    d = np.asarray(p_dual, dtype=np.float64)
    c0 = np.asarray(p_contains0, dtype=np.float64)
    c1 = np.asarray(p_contains1, dtype=np.float64)
    if not (np.isfinite(d).all() and np.isfinite(c0).all() and np.isfinite(c1).all()):
        raise ContractError("Γ 枚举读数含非有限项")
    ok = (c0 >= 1.0 - a) & (c1 >= 1.0 - a)
    if ok.ndim == 0:
        return bool(ok)
    return ok




__all__ += [
    "SCHEMA_CERT2",
    "THRESHOLD_SOURCES",
    "BINARY_KERNEL_FAMILIES",
    "CONTINUOUS_KERNEL_FAMILIES",
    "ELLIPSOID_APPLICABILITY",
    "SupSemantics",
    "NarrowBoundCause",
    "TrackKind",
    "NoiseKernel",
    "bits_matrix",
    "popcount_table",
    "hamming_statistic",
    "outcome_prob_table",
    "statistic_pmf",
    "exact_p_value",
    "mc_statistic_draws",
    "mc_p_value_from_table",
    "resolution_guard",
    "require_threshold_source",
    "require_resolution",
    "confidence_mask",
    "target_projection",
    "composite_sup",
    "chi_square_radius",
    "ellipsoid_applicability",
    "ellipsoid_accept",
    "sequential_alpha_schedule",
    "sequential_update",
    "envelope_intersect",
    "impossibility_dual_bound",
    "OBSERVATION_ANCHORED",
    "INDEPENDENTLY_DECLARED",
    "certified_theta_space",
    "classify_unnarrowed",
    "validate_cert2_experiment",
    "build_cert2_experiment",
    "always_double_target_set",
    "majority_target_set",
    "gamma_coverage_premises_ok",
]

# ===========================================================================
# CERT-3 (G29) · 拓扑认证层: 区域外包 / 五状态证书 / 验证器 / 分诊
#
# 本节实现《技术方案 v1.0》§7 的机器面 (与 results/g29_cert3_topological_certification/
# prereg.json 同步):
#
# * 三要素各自成栏并分别外包 (节点存在性 NODE / 盘间边 EDGE / 端点附着 ATTACHMENT,
#   §7.2)。附着不并进图边: 只外包盘间边而把附着当既定的仪器, 在 region_label 处
#   必然与完整枚举反例相矛盾, 由验证器拒。
# * may 边永不当连通见证: 单值证书的见证必按该状态自身的三要素重放 (§7.2 末段)。
# * 空可行集合只能 MODEL_CONFLICT (§7.1: 空外包上的全连/全断是恒真谓词)。
# * 几何落在未认证数值边界 ⇒ GEOMETRY_NUMERIC_UNRESOLVED, 禁猜边得单值 (§7.1)。
# * 一切外包由完整枚举区域状态得到, 故 geometry_semantics 恒 DECLARED_COMPUTATIONAL;
#   连续分支定界不在本层 (§7.4: 有限状态完整枚举是首期硬交付)。
# ===========================================================================

SCHEMA_CERT3 = "connectivity-cert3-topological/v1"

GEOMETRY_SEMANTICS_KINDS = ("DECLARED_COMPUTATIONAL", "CERTIFIED_CONTINUOUS")

SINGLE_VALUE_PROOF_ROUTES = ("MUST_ENVELOPE", "MAY_ENVELOPE_EXCLUSION")


class ObjectKind(str, Enum):
    """§7.2 三要素的枚举面: 逐要素分别外包, 禁合并成一栏。"""
    NODE = "NODE"
    EDGE = "EDGE"
    ATTACHMENT = "ATTACHMENT"


class RegionLabel(str, Enum):
    ALL_CONNECTED = "ALL_CONNECTED"
    ALL_DISCONNECTED = "ALL_DISCONNECTED"
    UNRESOLVED = "UNRESOLVED"


class ProofRoute(str, Enum):
    NONE = "NONE"
    MUST_ENVELOPE = "MUST_ENVELOPE"
    MAY_ENVELOPE_EXCLUSION = "MAY_ENVELOPE_EXCLUSION"
    WITNESS_PAIR = "WITNESS_PAIR"


class UnresolvedCause(str, Enum):
    WITNESS_BUDGET_EXHAUSTED = "WITNESS_BUDGET_EXHAUSTED"
    SPLIT_BUDGET_EXHAUSTED = "SPLIT_BUDGET_EXHAUSTED"
    GEOMETRY_NUMERIC_UNRESOLVED = "GEOMETRY_NUMERIC_UNRESOLVED"
    TIME_LIMIT = "TIME_LIMIT"


class TriageKind(str, Enum):
    """128 类分诊三档 + 非双值档 (任务单 §1④)。"""
    STRUCTURAL_AMBIGUITY = "STRUCTURAL_AMBIGUITY"
    OBSERVATION_LIMITED = "OBSERVATION_LIMITED"
    COMPUTATION_UNRESOLVED = "COMPUTATION_UNRESOLVED"
    NOT_AMBIGUOUS = "NOT_AMBIGUOUS"


@dataclass(frozen=True)
class StateDerivation:
    """单状态由模型现场派生的三要素 + 目标标签 (识别消费路径, 不读隐藏真值)。

    a_side / b_side 是端点各自附着的网络对象集合: 附着栏独立成列, 不并入 edges。
    """

    state_id: int
    nodes: frozenset
    edges: frozenset
    a_side: frozenset
    b_side: frozenset
    attach_a: bool
    attach_b: bool
    label: int

    def __post_init__(self):
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        object.__setattr__(self, "edges", _canon(self.edges))
        object.__setattr__(self, "a_side", frozenset(self.a_side))
        object.__setattr__(self, "b_side", frozenset(self.b_side))
        if int(self.label) not in (0, 1):
            raise ContractError(f"label 非法: {self.label!r}")
        for (u, v) in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ContractError(
                    f"状态 {self.state_id}: 盘间边 ({u},{v}) 含不在在场节点内的对象 —— 三要素派生不自洽")
        for x in (self.a_side | self.b_side):
            if x not in self.nodes:
                raise ContractError(
                    f"状态 {self.state_id}: 附着对象 {x!r} 不在在场节点内 —— 三要素派生不自洽")

    def to_dict(self) -> dict:
        return {"state_id": self.state_id,
                "nodes": sorted(str(x) for x in self.nodes),
                "edges": sorted([str(u), str(v)] for (u, v) in self.edges),
                "a_side": sorted(str(x) for x in self.a_side),
                "b_side": sorted(str(x) for x in self.b_side),
                "attach_a": bool(self.attach_a), "attach_b": bool(self.attach_b),
                "label": int(self.label)}


def _canon(edges: Iterable[Tuple[object, object]]) -> frozenset:
    return frozenset((min(u, v), max(u, v)) for (u, v) in edges)


def check_attachment_consistency(derivation: StateDerivation) -> None:
    """附着声明必与其附着对象集合同空非空 (§7.1: 未附着与内部断链要能分开报)。"""
    if derivation.attach_a != (len(derivation.a_side) > 0):
        raise ContractError(f"状态 {derivation.state_id}: A 附着声明与其附着对象集不自洽")
    if derivation.attach_b != (len(derivation.b_side) > 0):
        raise ContractError(f"状态 {derivation.state_id}: B 附着声明与其附着对象集不自洽")


def _sides_connected(nodes: frozenset, edges: frozenset, side_a, side_b) -> bool:
    """多源邻接遍历: side_a 中任一对象能否沿 edges 到 side_b 中任一对象。"""
    sa = frozenset(side_a) & nodes
    sb = frozenset(side_b) & nodes
    if not sa or not sb:
        return False
    adj: Dict[object, List[object]] = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = set(sa)
    stack = list(sa)
    while stack:
        x = stack.pop()
        if x in sb:
            return True
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def state_connected(derivation: StateDerivation) -> bool:
    """按该状态自身三要素重放目标 (见证重验与忠实性核对都用此路)。"""
    return bool(derivation.attach_a and derivation.attach_b
                and _sides_connected(derivation.nodes, derivation.edges,
                                     derivation.a_side, derivation.b_side))


@dataclass(frozen=True)
class Envelope:
    n_states: int
    must_nodes: frozenset
    may_nodes: frozenset
    must_edges: frozenset
    may_edges: frozenset
    must_a_side: frozenset
    may_a_side: frozenset
    must_b_side: frozenset
    may_b_side: frozenset
    must_attach_a: bool
    may_attach_a: bool
    must_attach_b: bool
    may_attach_b: bool

    def element_counts(self) -> dict:
        """逐要素计数 (§7.2 三要素都外包的在场证据: 每要素 must/may 各一栏)。"""
        return {
            ObjectKind.NODE.value: {"must": len(self.must_nodes), "may": len(self.may_nodes)},
            ObjectKind.EDGE.value: {"must": len(self.must_edges), "may": len(self.may_edges)},
            ObjectKind.ATTACHMENT.value: {
                "must": int(self.must_attach_a) + int(self.must_attach_b),
                "may": int(self.may_attach_a) + int(self.may_attach_b),
                "must_a": self.must_attach_a, "may_a": self.may_attach_a,
                "must_b": self.must_attach_b, "may_b": self.may_attach_b},
        }

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "must_nodes": sorted(str(x) for x in self.must_nodes),
            "may_nodes": sorted(str(x) for x in self.may_nodes),
            "must_edges": sorted([str(u), str(v)] for (u, v) in self.must_edges),
            "may_edges": sorted([str(u), str(v)] for (u, v) in self.may_edges),
            "must_a_side": sorted(str(x) for x in self.must_a_side),
            "may_a_side": sorted(str(x) for x in self.may_a_side),
            "must_b_side": sorted(str(x) for x in self.must_b_side),
            "may_b_side": sorted(str(x) for x in self.may_b_side),
            "must_attach_a": self.must_attach_a, "may_attach_a": self.may_attach_a,
            "must_attach_b": self.must_attach_b, "may_attach_b": self.may_attach_b,
        }


def envelope_of(derivations: Sequence[StateDerivation]) -> Envelope:
    """集合运算路: 三要素各自取区域全体状态的 ∀/∃ 外包。

    空区域拒发 (§7.1): 所有区域被剔除只能是 MODEL_CONFLICT, 空外包上的
    "全部连通/全部不连通"是恒真谓词, 不是证据。
    """
    ds = list(derivations)
    if len(ds) == 0:
        raise ContractError("空区域不得建外包 (恒真谓词禁当证据): 由调用方走 MODEL_CONFLICT")
    for d in ds:
        check_attachment_consistency(d)
    return _env_from_sets(ds)


def _env_from_sets(ds: Sequence[StateDerivation]) -> Envelope:
    must_nodes = frozenset.intersection(*(frozenset(d.nodes) for d in ds))
    may_nodes = frozenset.union(*(frozenset(d.nodes) for d in ds))
    must_edges = _canon(frozenset.intersection(*(frozenset(d.edges) for d in ds)))
    may_edges = _canon(frozenset.union(*(frozenset(d.edges) for d in ds)))
    must_a = frozenset.intersection(*(frozenset(d.a_side) for d in ds))
    may_a = frozenset.union(*(frozenset(d.a_side) for d in ds))
    must_b = frozenset.intersection(*(frozenset(d.b_side) for d in ds))
    may_b = frozenset.union(*(frozenset(d.b_side) for d in ds))
    stray = sorted([str((u, v)) for (u, v) in must_edges
                    if u not in must_nodes or v not in must_nodes])
    if stray:
        raise ContractError(f"must 盘间边 {stray} 的端点不在 must 节点内: 节点存在性未外包")
    stray_a = sorted(str(x) for x in (must_a | must_b) if x not in must_nodes)
    if stray_a:
        raise ContractError(f"must 附着对象 {stray_a} 不在 must 节点内: 附着与存在性不自洽")
    return Envelope(
        len(ds), must_nodes, may_nodes, must_edges, may_edges, must_a, may_a, must_b, may_b,
        all(d.attach_a for d in ds), any(d.attach_a for d in ds),
        all(d.attach_b for d in ds), any(d.attach_b for d in ds))


def envelope_of_quantifier(derivations: Sequence[StateDerivation]) -> Envelope:
    """逐元素 ∀/∃ 循环路 (验证器的独立第二路, 与集合运算路不共用实现)。"""
    ds = list(derivations)
    if len(ds) == 0:
        raise ContractError("空区域不得建外包")
    node_u = set().union(*(set(d.nodes) for d in ds))
    edge_u = set().union(*(set(d.edges) for d in ds))
    a_u = set().union(*(set(d.a_side) for d in ds))
    b_u = set().union(*(set(d.b_side) for d in ds))
    mn = {x for x in node_u if all(x in d.nodes for d in ds)}
    yn = {x for x in node_u if any(x in d.nodes for d in ds)}
    me = {e for e in edge_u if all(e in d.edges for d in ds)}
    ye = {e for e in edge_u if any(e in d.edges for d in ds)}
    ma = {x for x in a_u if all(x in d.a_side for d in ds)}
    ya = {x for x in a_u if any(x in d.a_side for d in ds)}
    mb = {x for x in b_u if all(x in d.b_side for d in ds)}
    yb = {x for x in b_u if any(x in d.b_side for d in ds)}
    stray = sorted([str((u, v)) for (u, v) in me if u not in mn or v not in mn])
    if stray:
        raise ContractError(f"must 盘间边 {stray} 的端点不在 must 节点内: 节点存在性未外包")
    return Envelope(
        len(ds), frozenset(mn), frozenset(yn), _canon(me), _canon(ye),
        frozenset(ma), frozenset(ya), frozenset(mb), frozenset(yb),
        all(d.attach_a for d in ds), any(d.attach_a for d in ds),
        all(d.attach_b for d in ds), any(d.attach_b for d in ds))


def must_connected(env: Envelope, endpoints=("A", "B")) -> bool:
    """must 有路 ⇒ 区域内全连通 (§7.2 第一条; 附着栏单独把守)。"""
    return bool(env.must_attach_a and env.must_attach_b
                and _sides_connected(env.must_nodes, env.must_edges,
                                     env.must_a_side, env.must_b_side))


def may_connected(env: Envelope, endpoints=("A", "B")) -> bool:
    """may 有路是可能性外包, 不是连通见证 (§7.2 第二条与末段)。"""
    return bool(env.may_attach_a and env.may_attach_b
                and _sides_connected(env.may_nodes, env.may_edges,
                                     env.may_a_side, env.may_b_side))


def region_label(env: Envelope) -> str:
    """§7.2 三条方向规则: must 有路⇒全连; may 无路⇒全断; 否则未解决。"""
    m = must_connected(env)
    y = may_connected(env)
    if m and not y:
        raise ContractError("外包单调性破裂 (must 路真而 may 路假): 仪器缺陷, 禁出读数")
    if m:
        return RegionLabel.ALL_CONNECTED.value
    if not y:
        return RegionLabel.ALL_DISCONNECTED.value
    return RegionLabel.UNRESOLVED.value


def numeric_boundary_flags(margins: Sequence[Tuple[object, float]], tol: float) -> List[object]:
    """几何判据余量 < tol 的对象列入未认证数值边界 (§7.1 原因位)。"""
    if not (isinstance(tol, (int, float)) and np.isfinite(float(tol)) and float(tol) > 0):
        raise ContractError(f"tol 非法: {tol!r}")
    out = []
    for key, margin in margins:
        if not np.isfinite(float(margin)):
            raise ContractError(f"余量非有限: {key!r}/{margin!r}")
        if abs(float(margin)) < float(tol):
            out.append(key)
    return out


def reverify_witness(state: StateDerivation, claimed_label: int,
                     claimed_path: Optional[Sequence[object]] = None) -> dict:
    """见证重验: 只按该状态自身的三要素重放; may 边与跨区拼接边在此处必然失败 (§7.2 末段)。"""
    if int(claimed_label) not in (0, 1):
        raise ContractError(f"见证声明标签非法: {claimed_label!r}")
    if int(state.label) != int(claimed_label):
        return {"ok": False, "reason": "LABEL_MISMATCH",
                "recomputed_label": int(state.label), "claimed_label": int(claimed_label)}
    replay = state_connected(state)
    if int(claimed_label) == 1 and not replay:
        return {"ok": False, "reason": "CLAIMED_CONNECTED_BUT_STATE_DISCONNECTED",
                "recomputed_label": 1, "claimed_label": 1}
    if int(claimed_label) == 0 and replay:
        return {"ok": False, "reason": "CLAIMED_DISCONNECTED_BUT_STATE_CONNECTED",
                "recomputed_label": 0, "claimed_label": 0}
    if int(claimed_label) == 1 and claimed_path is not None:
        if len(claimed_path) < 2:
            raise ContractError(f"claimed_path 不足两端点: {claimed_path!r}")
        nodes = frozenset(claimed_path)
        if any(x not in state.nodes for x in nodes):
            return {"ok": False, "reason": "PATH_USES_ABSENT_NODE",
                    "detail": sorted(str(x) for x in nodes - state.nodes),
                    "recomputed_label": 1, "claimed_label": 1}
        pedges = _canon(zip(list(claimed_path)[:-1], list(claimed_path)[1:]))
        bad = sorted([str(u), str(v)] for (u, v) in pedges if (u, v) not in state.edges)
        if bad:
            return {"ok": False, "reason": "PATH_USES_ABSENT_EDGE", "detail": bad,
                    "recomputed_label": 1, "claimed_label": 1}
        if claimed_path[0] not in state.a_side:
            return {"ok": False, "reason": "PATH_START_NOT_ATTACHED",
                    "detail": [str(claimed_path[0])], "recomputed_label": 1, "claimed_label": 1}
        if claimed_path[-1] not in state.b_side:
            return {"ok": False, "reason": "PATH_END_NOT_ATTACHED",
                    "detail": [str(claimed_path[-1])], "recomputed_label": 1, "claimed_label": 1}
    return {"ok": True, "reason": "REPLAYED", "recomputed_label": int(state.label),
            "claimed_label": int(claimed_label)}


def witness_scan(states: Sequence[StateDerivation], *, scan_cap: Optional[int] = None) -> dict:
    """可行见证搜索: 升序扫描找 0/1 两标签见证; scan_cap=None 即预算不 binding。"""
    if len(states) == 0:
        raise ContractError("空可行集合不找见证: 由调用方走 MODEL_CONFLICT")
    order = sorted(states, key=lambda d: d.state_id)
    cap = len(order) if scan_cap is None else int(scan_cap)
    if cap <= 0:
        raise ContractError(f"scan_cap 非法: {cap}")
    seen = 0
    w0 = w1 = None
    exhausted = False
    for d in order:
        if seen >= cap:
            exhausted = True
            break
        seen += 1
        if d.label == 0 and w0 is None:
            w0 = d
        elif d.label == 1 and w1 is None:
            w1 = d
        if w0 is not None and w1 is not None:
            break
    return {"scanned": seen, "cap": cap,
            "exhausted_cap": bool(exhausted and (w0 is None or w1 is None)),
            "witness_0": None if w0 is None else int(w0.state_id),
            "witness_1": None if w1 is None else int(w1.state_id),
            "found_both": bool(w0 is not None and w1 is not None),
            "enumerated_states": len(order)}


def triage_of(*, batch_status: str, unresolved_cause: Optional[str] = None) -> str:
    """分诊档 (§1④): 计算未完优先, 再按 B* 是否有限分结构歧义 / 观测受限。"""
    if unresolved_cause is not None:
        if unresolved_cause not in {c.value for c in UnresolvedCause}:
            raise ContractError(f"unresolved_cause 非法: {unresolved_cause!r}")
        return TriageKind.COMPUTATION_UNRESOLVED.value
    if batch_status == "UNRESOLVED_UNSEPARABLE_PAIRS":
        return TriageKind.STRUCTURAL_AMBIGUITY.value
    if batch_status == "OPTIMAL":
        return TriageKind.OBSERVATION_LIMITED.value
    if batch_status in ("NO_CROSS_PAIRS", "NOT_EVALUATED"):
        return TriageKind.NOT_AMBIGUOUS.value
    raise ContractError(f"batch_status 非法: {batch_status!r}")


def build_cert3_certificate(*, certificate_id: str, model_id: str, query_id: str,
                            observation_id: str, status: str, target_set: Sequence[int],
                            feasible_state_ids: Sequence[int],
                            regions: Sequence[Mapping],
                            witnesses: Sequence[Mapping],
                            derivation_of_state: Callable[[int], StateDerivation],
                            geometry_semantics: str,
                            unresolved_cause: Optional[str] = None,
                            proof_route: str = ProofRoute.NONE.value,
                            search_complete: bool = True,
                            unresolved_region_count: Optional[int] = None,
                            continuous_certification_evidence: Optional[Mapping] = None,
                            guarantee_kind: str = "EXACT_FINITE_COMPLETE_ENUMERATION",
                            alpha_total: Optional[float] = None,
                            source_fingerprints: Optional[Dict[str, str]] = None,
                            reason: str = "") -> dict:
    """五状态证书机 (§7.1): 单值证书的两个合取前提在此处机械把守。

    合取前提 = (至少一个对应答案的有效可行见证) ∧ (完整允许空间的相反答案被保守排除)。
    空可行集合 ⇒ 只能 MODEL_CONFLICT; 找不到见证又未证空 ⇒ 只能 SEARCH_UNRESOLVED。
    """
    fs = sorted(int(s) for s in feasible_state_ids)
    a_side = [bool(derivation_of_state(int(s)).attach_a) for s in fs]
    b_side = [bool(derivation_of_state(int(s)).attach_b) for s in fs]
    if len(fs) == 0:
        attach_a_sum, attach_b_sum = UNATTACHED, UNATTACHED
    else:
        attach_a_sum = attachment_summary(a_side)
        attach_b_sum = attachment_summary(b_side)
    doc = {
        "schema": SCHEMA_CERT3,
        "certificate_id": certificate_id,
        "model_id": model_id, "query_id": query_id, "observation_id": observation_id,
        "guarantee_kind": guarantee_kind, "alpha_total": alpha_total,
        "status": status, "target_set": sorted(set(int(t) for t in target_set)),
        "attachment_status": f"A={attach_a_sum};B={attach_b_sum}",
        "attachment_counts": {
            "feasible": len(fs),
            "a_attached_true": int(sum(1 for x in a_side if x)),
            "b_attached_true": int(sum(1 for x in b_side if x)),
        },
        "proof_kind": (ProofKind.COMPLETE_ENUMERATION_EXCLUSION.value
                       if status.startswith("CERTIFIED") else ProofKind.NONE_YET.value),
        "proof_route": proof_route,
        "feasible_state_ids": fs,
        "regions": [dict(r) for r in regions],
        "witnesses": [dict(w) for w in witnesses],
        "unresolved_region_count": (None if unresolved_region_count is None
                                    else int(unresolved_region_count)),
        "search_complete": bool(search_complete),
        "unresolved_cause": unresolved_cause,
        "numeric_semantics": "EXACT_FINITE_COMPLETE_ENUMERATION",
        "geometry_semantics": geometry_semantics,
        "continuous_certification_evidence": (dict(continuous_certification_evidence)
                                              if continuous_certification_evidence else None),
        "source_fingerprints": dict(source_fingerprints or {}),
        "reason": reason,
    }
    validate_cert3_certificate(doc, derivation_of_state=derivation_of_state)
    return doc


def validate_cert3_certificate(doc: Mapping,
                               derivation_of_state: Optional[
                                   Callable[[int], StateDerivation]] = None) -> None:
    """证书条件必填检查 (§7.1 + §16.4); 合取前提不满足即 ContractError (禁静默降级)。"""
    if doc.get("schema") != SCHEMA_CERT3:
        raise ContractError(f"schema 非法: {doc.get('schema')!r}")
    for k in ("certificate_id", "model_id", "query_id", "observation_id", "reason"):
        if not isinstance(doc.get(k), str) or not doc.get(k):
            raise ContractError(f"必填字段 {k!r} 缺失/为空")
    gs = doc.get("geometry_semantics")
    if gs not in GEOMETRY_SEMANTICS_KINDS:
        raise ContractError(f"geometry_semantics 必须二值之一 {GEOMETRY_SEMANTICS_KINDS}: {gs!r}")
    if gs == "CERTIFIED_CONTINUOUS" and not doc.get("continuous_certification_evidence"):
        raise ContractError("CERTIFIED_CONTINUOUS 是更强断言, 必须带连续几何认证证据; "
                            "空手误标被抓 (§7.4 不得省略区别)")
    st = doc.get("status")
    if st not in {s.value for s in IdentificationStatus}:
        raise ContractError(f"status 非法: {st!r}")
    ts = doc.get("target_set")
    if not isinstance(ts, list) or sorted(set(ts)) != ts or any(t not in (0, 1) for t in ts):
        raise ContractError(f"target_set 非法: {ts!r}")
    if not isinstance(doc.get("regions"), list):
        raise ContractError("regions 必须为 list")
    wit = doc.get("witnesses")
    if not isinstance(wit, list):
        raise ContractError("witnesses 必须为 list")
    feasible = [int(s) for s in doc.get("feasible_state_ids", [])]
    if len(feasible) == 0 and st != IdentificationStatus.MODEL_CONFLICT.value:
        raise ContractError("空可行集合只能 MODEL_CONFLICT: 空外包上的全连/全断是恒真谓词 (§7.1)")
    labels = sorted({int(w["label"]) for w in wit})
    single = st in (IdentificationStatus.CERTIFIED_CONNECTED.value,
                    IdentificationStatus.CERTIFIED_DISCONNECTED.value)
    if single:
        want = 1 if st == IdentificationStatus.CERTIFIED_CONNECTED.value else 0
        if ts != [want]:
            raise ContractError(f"{st} 必须 target_set=[{want}]")
        if doc.get("proof_kind") != ProofKind.COMPLETE_ENUMERATION_EXCLUSION.value:
            raise ContractError("单值证书必须带可重验排除证据 (proof_kind)")
        if doc.get("proof_route") not in SINGLE_VALUE_PROOF_ROUTES:
            raise ContractError(f"单值证书必须由外包路证得相反答案排除: {doc.get('proof_route')!r}")
        if labels != [want]:
            raise ContractError(f"单值见证前提不满足: 需 label={want} 的有效见证, 现 {labels}")
        if not doc.get("search_complete"):
            raise ContractError("单值证书必须 search_complete=true")
        excl = (RegionLabel.ALL_CONNECTED.value if want == 1
                else RegionLabel.ALL_DISCONNECTED.value)
        regs = doc.get("regions")
        if not regs:
            raise ContractError("单值证书必须有区域登记 (相反答案的排除逐区域记账)")
        if any(r.get("recomputed_label") != excl for r in regs):
            raise ContractError("单值证书的相反答案排除必须覆盖全部剩余区域 (逐区 recomputed_label 齐同)")
    elif st == IdentificationStatus.WITNESSED_AMBIGUOUS.value:
        if ts != [0, 1]:
            raise ContractError("WITNESSED_AMBIGUOUS 必须 target_set=[0,1]")
        if labels != [0, 1]:
            raise ContractError("WITNESSED_AMBIGUOUS 必须两种合法见证齐")
        if doc.get("proof_route") != ProofRoute.WITNESS_PAIR.value:
            raise ContractError("歧义证书的 proof_route 必须是 WITNESS_PAIR")
    elif st == IdentificationStatus.SEARCH_UNRESOLVED.value:
        if ts != [0, 1]:
            raise ContractError("SEARCH_UNRESOLVED 保守保留 [0,1]")
        if doc.get("search_complete"):
            raise ContractError("SEARCH_UNRESOLVED 必须 search_complete=false")
        if doc.get("unresolved_cause") is None:
            raise ContractError("SEARCH_UNRESOLVED 必须给未解决原因 (禁空对空)")
        if doc.get("unresolved_region_count") is None:
            raise ContractError("SEARCH_UNRESOLVED 必须给未解决区域数")
    elif st == IdentificationStatus.MODEL_CONFLICT.value:
        if ts != []:
            raise ContractError("MODEL_CONFLICT 的 target_set 必须为空 (不计单值)")
        if labels:
            raise ContractError("MODEL_CONFLICT 不得携带见证 (空可行集合上无可行见证)")
        if doc.get("unresolved_region_count") != 0:
            raise ContractError("MODEL_CONFLICT 必须登记全部区域被剔除 (unresolved_region_count=0)")
    uc = doc.get("unresolved_cause")
    if uc is not None and uc not in {c.value for c in UnresolvedCause}:
        raise ContractError(f"unresolved_cause 非法: {uc!r}")
    if st != IdentificationStatus.SEARCH_UNRESOLVED.value and uc is not None:
        raise ContractError("未解决原因只属 SEARCH_UNRESOLVED: 与 MODEL_CONFLICT 不得混 (§7.1)")
    for w in wit:
        if int(w.get("label", -1)) not in (0, 1):
            raise ContractError(f"见证标签非法: {w!r}")
        if "state_id" not in w:
            raise ContractError("见证必须带可重放的 state_id")
        if int(w["state_id"]) not in feasible:
            raise ContractError(f"见证状态 {w['state_id']} 不在可行集合内: 非有效可行见证 (§7.1)")
        if derivation_of_state is not None:
            state = derivation_of_state(int(w["state_id"]))
            rep = reverify_witness(state, int(w["label"]), w.get("path"))
            if not rep["ok"]:
                raise ContractError(f"见证重放失败 {rep}: {w!r}")
    ident_164 = dict(
        schema=SCHEMA_IDENTIFICATION, model_id=doc["model_id"], query_id=doc["query_id"],
        observation_id=doc["observation_id"],
        guarantee_kind=doc["guarantee_kind"], alpha_total=doc["alpha_total"],
        status=doc["status"], target_set=list(doc["target_set"]),
        attachment_status=doc["attachment_status"], proof_kind=doc["proof_kind"],
        witnesses=[dict(w) for w in doc["witnesses"]],
        unresolved_region_count=doc.get("unresolved_region_count"),
        search_complete=doc["search_complete"], numeric_semantics=doc["numeric_semantics"],
        source_fingerprints=dict(doc["source_fingerprints"]), reason=doc["reason"])
    validate_identification_result(ident_164)


def verify_regions(*, regions: Sequence[Mapping],
                   derivation_of_state: Callable[[int], StateDerivation],
                   allowed_state_ids: Sequence[int],
                   optimizer_success: Optional[bool] = None) -> dict:
    """验证器 (§7.3 末句): 区域剔除与图外包都要重算; 优化器 success=True 不是证书。"""
    reasons: List[str] = []
    covered: List[int] = []
    per_region = []
    for r in regions:
        states = [derivation_of_state(int(s)) for s in r["state_ids"]]
        if len(states) == 0:
            reasons.append(f"区域 {r.get('region_id')} 为空: 剔除只能由观测约束做, 禁空区域产读数")
            continue
        covered.extend(int(s) for s in r["state_ids"])
        env = envelope_of(states)
        env_q = envelope_of_quantifier(states)
        two_route_equal = env.to_dict() == env_q.to_dict()
        if not two_route_equal:
            reasons.append(f"区域 {r.get('region_id')} 两路外包逐要素不等 (仪器缺陷)")
        lab = region_label(env)
        if r.get("claimed_label") != lab:
            reasons.append(f"区域 {r.get('region_id')} 外包方向不符: 声明 "
                           f"{r.get('claimed_label')} 重算 {lab}")
        truth = {int(1 if state_connected(d) else 0) for d in states}
        if lab == RegionLabel.ALL_CONNECTED.value and truth != {1}:
            reasons.append(f"区域 {r.get('region_id')} must 有路但完整枚举含不连通状态 {sorted(truth)}")
        if lab == RegionLabel.ALL_DISCONNECTED.value and truth != {0}:
            reasons.append(f"区域 {r.get('region_id')} may 无路但完整枚举含连通状态 {sorted(truth)}")
        for d in states:
            if int(d.label) != int(1 if state_connected(d) else 0):
                reasons.append(f"区域 {r.get('region_id')} 状态 {d.state_id} 三要素派生与标签不忠实")
        for w in r.get("witnesses", []):
            rep = reverify_witness(derivation_of_state(int(w["state_id"])), int(w["label"]),
                                   w.get("path"))
            if not rep["ok"]:
                reasons.append(f"区域 {r.get('region_id')} 见证不可重放 {rep}")
        per_region.append({"region_id": r.get("region_id"), "recomputed_label": lab,
                           "enumerated_states": len(states), "connected_labels_seen": sorted(truth),
                           "two_route_envelope_equal": two_route_equal,
                           "element_counts": env.element_counts()})
    allowed = sorted(int(s) for s in allowed_state_ids)
    cov = sorted(covered)
    if len(cov) != len(set(cov)):
        reasons.append("区域划分不互斥: 禁发全空间结论")
    if cov != allowed:
        reasons.append(f"区域划分不覆盖完整允许空间 (覆盖 {len(cov)} / 允许 {len(allowed)}): "
                       f"单区域判定不代表整体")
    whole: Optional[int] = None
    if not reasons:
        labs = {p["recomputed_label"] for p in per_region}
        if labs == {RegionLabel.ALL_CONNECTED.value}:
            whole = 1
        elif labs == {RegionLabel.ALL_DISCONNECTED.value}:
            whole = 0
    return {"ok": not reasons, "reasons": reasons, "per_region": per_region,
            "whole_space_label": whole, "coverage_enumerated": len(cov),
            "allowed_enumerated": len(allowed),
            "optimizer_success_ignored": optimizer_success is True}


__all__ += [
    "SCHEMA_CERT3",
    "GEOMETRY_SEMANTICS_KINDS",
    "SINGLE_VALUE_PROOF_ROUTES",
    "ObjectKind",
    "RegionLabel",
    "ProofRoute",
    "UnresolvedCause",
    "TriageKind",
    "StateDerivation",
    "check_attachment_consistency",
    "state_connected",
    "envelope_of",
    "envelope_of_quantifier",
    "must_connected",
    "may_connected",
    "region_label",
    "numeric_boundary_flags",
    "reverify_witness",
    "witness_scan",
    "triage_of",
    "build_cert3_certificate",
    "validate_cert3_certificate",
    "verify_regions",
]
