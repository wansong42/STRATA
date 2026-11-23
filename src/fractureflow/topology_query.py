# -*- coding: utf-8 -*-
"""G61 · Γ_Q 拓扑查询: 包络投影、五状态输出与独立验证器 (QUERY_V1)。

本模块是技术书 §3.2 贡献点 2 / §4.2 查询契约 / §5.2 E1 有限模型全轨迹 /
§6.1 topology_query.py 的机器面实现。口径:

* Γ_Q(Y) = {Q(D) : D ∈ E(Y)}; 本单的 E(Y) 用显式可行集全枚举承载
  (FeasibleSet.structures), 不引入连续/半解析外包, 也不含采样近似。
* 五状态 (与 §4.2 状态表逐条对应):
  {True} -> CERTIFIED_TRUE, {False} -> CERTIFIED_FALSE,
  {True, False} -> AMBIGUOUS, E=∅ -> MODEL_CONFLICT,
  预算耗尽 -> SEARCH_UNRESOLVED。空包络禁输出单值; 预算耗尽禁写成
  MODEL_CONFLICT/AMBIGUOUS (保守 γ 记 {False, True} 且 search_complete=False)。
* 查询族三族: TWO_POINT_CONNECTIVITY / INTERSECTS / ADJACENT。
  api_evaluate 走"结构关系图 + 附着"路, reference_evaluate 走"从对象几何
  重算"路; verify 只用 reference 路重投影, **不调用 answer**。
* 复用既有 semantics 但**不改**: identification / connectivity_qoi /
  intersect_disk 只 import 只读; 与 identification.Envelope 的语义对照入口
  是 identification_view (仅 TWO_POINT 族适用, 见 E10)。
* 本模块零 json 写盘; 读数件一律由脚本侧以 allow_nan=False 写出。
* 零研究读数: 合成有限小世界, 不构成真实岩体/真实观测/采购结论。
* 无窗纪律: 本模块不启动任何子进程。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_TOPOLOGY_QUERY = "topology-query/v1"
SCHEMA_QUERY_ANSWER = "topology-query.answer.v1"

FAMILY_TWO_POINT_CONNECTIVITY = "TWO_POINT_CONNECTIVITY"
FAMILY_INTERSECTS = "INTERSECTS"
FAMILY_ADJACENT = "ADJACENT"
FAMILIES = (FAMILY_TWO_POINT_CONNECTIVITY, FAMILY_INTERSECTS, FAMILY_ADJACENT)

STATUS_CERTIFIED_TRUE = "CERTIFIED_TRUE"
STATUS_CERTIFIED_FALSE = "CERTIFIED_FALSE"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_MODEL_CONFLICT = "MODEL_CONFLICT"
STATUS_SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"
STATUSES = (STATUS_CERTIFIED_TRUE, STATUS_CERTIFIED_FALSE, STATUS_AMBIGUOUS,
            STATUS_MODEL_CONFLICT, STATUS_SEARCH_UNRESOLVED)

STATUS_OF_GAMMA = {
    frozenset({True}): STATUS_CERTIFIED_TRUE,
    frozenset({False}): STATUS_CERTIFIED_FALSE,
    frozenset({True, False}): STATUS_AMBIGUOUS,
    frozenset(): STATUS_MODEL_CONFLICT,
}

SINGLE_VALUE_STATUSES = (STATUS_CERTIFIED_TRUE, STATUS_CERTIFIED_FALSE)
CONSERVATIVE_GAMMA = frozenset({True, False})

PROOF_COMPLETE_ENUMERATION = "COMPLETE_ENUMERATION_EXCLUSION"
PROOF_NONE = "NONE_YET"

RELATION_KINDS = ("intersects", "continues", "offsets", "contains", "adjacent",
                  "barrier_candidate")
OBJECT_TYPE_DISK = "DISK"
_NORMAL_TOL = 1e-8


class ContractError(ValueError):
    """契约违规 (缺字段/非法形/参数未冻结/真值语义冲突): 拒运行, 不降级。"""


class NotApplicableError(ContractError):
    """查询语义不适用 (如 n=0 圆盘世界): 必须显式 NOT_APPLICABLE, 禁静默 PASS。"""


# ---------------------------------------------------------------------------
# 标量 / 向量 / 结构校验
# ---------------------------------------------------------------------------

def _finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} 必须为数字, 收到 {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ContractError(f"{name} 含 NaN/inf: 本仪器禁静默传播")
    return out


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} 必须为非空字符串, 收到 {value!r}")
    return value


def _as_vec3(name: str, value: Any) -> Tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise ContractError(f"{name} 必须为 3 维数值向量, 收到 {type(value).__name__}")
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} 不可解析为坐标: {exc}") from exc
    if arr.shape != (3,):
        raise ContractError(f"{name} 形状非法: {arr.shape} (应为 (3,))")
    if not np.all(np.isfinite(arr)):
        raise ContractError(f"{name} 含 NaN/inf")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _unit_norm(vec: Sequence[float]) -> float:
    return math.sqrt(float(vec[0]) ** 2 + float(vec[1]) ** 2 + float(vec[2]) ** 2)


def _check_unit_normal(name: str, vec: Tuple[float, float, float]) -> None:
    nrm = _unit_norm(vec)
    if abs(nrm - 1.0) > _NORMAL_TOL:
        raise ContractError(f"{name} 非单位向量 (|n|={nrm:.12g}, 容差 {_NORMAL_TOL:g})")


# ---------------------------------------------------------------------------
# 预算 / 查询
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyBudget:
    """查询预算: 预算耗尽只许 SEARCH_UNRESOLVED。

    max_nodes 计已展开的可行结构数 (每评估一个结构 +1); max_seconds 为墙钟
    上限 (本单读数一律用 max_nodes 触发, 墙钟分支只作代码路径, 见 own_defects)。
    """
    max_nodes: int
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if isinstance(self.max_nodes, bool) or not isinstance(self.max_nodes, int):
            raise ContractError(f"TopologyBudget.max_nodes 必须为整数, 收到 {self.max_nodes!r}")
        if self.max_nodes < 0:
            raise ContractError(f"TopologyBudget.max_nodes 必须 >= 0, 收到 {self.max_nodes!r}")
        sec = _finite_scalar("TopologyBudget.max_seconds", self.max_seconds)
        if sec <= 0.0:
            raise ContractError(f"TopologyBudget.max_seconds 必须 > 0, 收到 {sec!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"max_nodes": int(self.max_nodes), "max_seconds": float(self.max_seconds),
                "node_unit": "FEASIBLE_STRUCTURE_EXPANDED"}


@dataclass(frozen=True)
class TopologyQuery:
    """冻结的拓扑查询: 族名 + params (A/B 坐标、r_attach、对象 id、邻接容差)。

    params 必须冻结: 构造时做深拷贝并按族的精确键集校验 (缺键/多键都拒),
    避免"改了坐标但证书没变"的静默漂移。
    """
    query_id: str
    family: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        qid = _as_str("query_id", self.query_id)
        fam = _as_str("family", self.family)
        if fam not in FAMILIES:
            raise ContractError(f"family 非法: {fam!r} (允许 {FAMILIES})")
        params = dict(self.params) if isinstance(self.params, Mapping) else None
        if params is None:
            raise ContractError("params 必须为 Mapping")
        if fam == FAMILY_TWO_POINT_CONNECTIVITY:
            keys = {"A", "B", "r_attach"}
            if set(params) != keys:
                raise ContractError(f"TWO_POINT_CONNECTIVITY.params 必须恰为 {sorted(keys)}, 收到 {sorted(params)}")
            na = _as_vec3("params.A", params["A"])
            nb = _as_vec3("params.B", params["B"])
            r = _finite_scalar("params.r_attach", params["r_attach"])
            if r < 0.0:
                raise ContractError(f"r_attach 必须非负, 收到 {r!r}")
            frozen = {"A": list(na), "B": list(nb), "r_attach": r}
        elif fam == FAMILY_INTERSECTS:
            keys = {"id_a", "id_b"}
            if set(params) != keys:
                raise ContractError(f"INTERSECTS.params 必须恰为 {sorted(keys)}, 收到 {sorted(params)}")
            a = _as_str("params.id_a", params["id_a"])
            b = _as_str("params.id_b", params["id_b"])
            if a == b:
                raise ContractError("INTERSECTS 查询的两个对象 id 必须不同")
            frozen = {"id_a": a, "id_b": b}
        else:  # ADJACENT
            keys = {"id_a", "id_b", "adjacency_gap"}
            if set(params) != keys:
                raise ContractError(f"ADJACENT.params 必须恰为 {sorted(keys)}, 收到 {sorted(params)}")
            a = _as_str("params.id_a", params["id_a"])
            b = _as_str("params.id_b", params["id_b"])
            if a == b:
                raise ContractError("ADJACENT 查询的两个对象 id 必须不同")
            gap = _finite_scalar("params.adjacency_gap", params["adjacency_gap"])
            if gap < 0.0:
                raise ContractError(f"adjacency_gap 必须非负, 收到 {gap!r}")
            frozen = {"id_a": a, "id_b": b, "adjacency_gap": gap}
        object.__setattr__(self, "query_id", qid)
        object.__setattr__(self, "family", fam)
        object.__setattr__(self, "params", frozen)

    def to_dict(self) -> Dict[str, Any]:
        return {"query_id": self.query_id, "family": self.family,
                "params": {k: (list(v) if isinstance(v, list) else v)
                           for k, v in self.params.items()}}

    @staticmethod
    def from_mapping(doc: Mapping[str, Any]) -> "TopologyQuery":
        if not isinstance(doc, Mapping):
            raise ContractError("TopologyQuery 需要 Mapping")
        for key in ("query_id", "family", "params"):
            if key not in doc:
                raise ContractError(f"TopologyQuery 缺字段 {key!r}: 拒运行")
        return TopologyQuery(doc["query_id"], doc["family"], doc["params"])


def _as_query(query: Any) -> TopologyQuery:
    if isinstance(query, TopologyQuery):
        return query
    if isinstance(query, Mapping):
        return TopologyQuery.from_mapping(query)
    raise ContractError(f"query 必须为 TopologyQuery 或 Mapping, 收到 {type(query).__name__}")


# ---------------------------------------------------------------------------
# 结构校验 (Mapping 鸭子类型; 不 import G59 schema 模块)
# ---------------------------------------------------------------------------

def validate_structure(structure: Any) -> Dict[str, Any]:
    """校验 Structure={objects, relations} 的最小契约; 非法即拒 (禁静默当 False)。"""
    if not isinstance(structure, Mapping):
        raise ContractError(f"structure 必须为 Mapping, 收到 {type(structure).__name__}")
    sid = _as_str("structure_id", structure.get("structure_id"))
    objects = structure.get("objects")
    relations = structure.get("relations")
    if not isinstance(objects, list) or not isinstance(relations, list):
        raise ContractError(f"structure {sid!r} 的 objects/relations 必须为 list")
    ids: List[str] = []
    for k, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            raise ContractError(f"structure {sid!r} objects[{k}] 非 Mapping")
        oid = _as_str("object.id", obj.get("id"))
        if oid in ids:
            raise ContractError(f"structure {sid!r} 对象 id 重复: {oid!r}")
        ids.append(oid)
        otype = _as_str("object.type", obj.get("type"))
        if otype != OBJECT_TYPE_DISK:
            raise ContractError(f"本单只支持 {OBJECT_TYPE_DISK}, 收到 {otype!r}")
        _as_vec3("object.center", obj.get("center"))
        normal = _as_vec3("object.normal", obj.get("normal"))
        _check_unit_normal("object.normal", normal)
        radius = _finite_scalar("object.radius", obj.get("radius"))
        if radius <= 0.0:
            raise ContractError(f"object.radius 必须 > 0, 收到 {radius!r}")
    id_set = frozenset(ids)
    for k, rel in enumerate(relations):
        if not isinstance(rel, Mapping):
            raise ContractError(f"structure {sid!r} relations[{k}] 非 Mapping")
        src = _as_str("relation.src", rel.get("src"))
        dst = _as_str("relation.dst", rel.get("dst"))
        kind = _as_str("relation.kind", rel.get("kind"))
        if kind not in RELATION_KINDS:
            raise ContractError(f"relation.kind 非法: {kind!r} (允许 {RELATION_KINDS})")
        if src == dst:
            raise ContractError("relation 的两端不得相同")
        if src not in id_set or dst not in id_set:
            raise ContractError(f"relation ({src},{dst}) 含不在 objects 内的对象: 结构不自洽")
    return {"structure_id": sid, "objects": objects, "relations": relations, "id_set": id_set}


def _object_arrays(structure: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    objects = structure["objects"]
    centers = np.asarray([obj["center"] for obj in objects], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray([obj["normal"] for obj in objects], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray([obj["radius"] for obj in objects], dtype=np.float64).reshape(-1)
    ids = [str(obj["id"]) for obj in objects]
    return centers, normals, radii, ids


def _relation_set(structure: Mapping[str, Any]) -> frozenset:
    out = set()
    for rel in structure["relations"]:
        u, v = str(rel["src"]), str(rel["dst"])
        out.add((min(u, v), max(u, v), str(rel["kind"])))
    return frozenset(out)


def _has_relation(structure: Mapping[str, Any], id_a: str, id_b: str, kind: str) -> bool:
    key = (min(id_a, id_b), max(id_a, id_b), kind)
    return key in _relation_set(structure)


# ---------------------------------------------------------------------------
# 几何基元 (两条实现路: 纯 Python 闭式 / 既有仪器)
# ---------------------------------------------------------------------------

def _disk_point_distance_py(center: Sequence[float], normal: Sequence[float],
                            radius: float, point: Sequence[float]) -> float:
    """点到圆盘距离闭式: sqrt(vn^2 + max(0, d_planar - r)^2)。"""
    vx = float(point[0]) - float(center[0])
    vy = float(point[1]) - float(center[1])
    vz = float(point[2]) - float(center[2])
    vn = vx * float(normal[0]) + vy * float(normal[1]) + vz * float(normal[2])
    px, py, pz = vx - vn * float(normal[0]), vy - vn * float(normal[1]), vz - vn * float(normal[2])
    d_planar = math.sqrt(px * px + py * py + pz * pz)
    gap = max(d_planar - float(radius), 0.0)
    return math.sqrt(vn * vn + gap * gap)


def _attached_ids(objects: Sequence[Mapping[str, Any]], point: Sequence[float],
                  r_attach: float) -> List[str]:
    out = []
    for obj in objects:
        dist = _disk_point_distance_py(obj["center"], obj["normal"], float(obj["radius"]), point)
        if dist <= float(r_attach):
            out.append(str(obj["id"]))
    return out


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, keys: Iterable[str]):
        self.parent = {k: k for k in keys}

    def find(self, x: str) -> str:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _sides_connected_py(present_ids: Sequence[str], edges: Iterable[Tuple[str, str]],
                        side_a: Sequence[str], side_b: Sequence[str]) -> bool:
    if not side_a or not side_b:
        return False
    uf = _UnionFind(present_ids)
    for (u, v) in edges:
        uf.union(u, v)
    roots_a = {uf.find(x) for x in side_a if x in uf.parent}
    roots_b = {uf.find(x) for x in side_b if x in uf.parent}
    return bool(roots_a) and bool(roots_b) and not roots_a.isdisjoint(roots_b)


def _pair_intersects_ref(centers, normals, radii, i: int, j: int) -> bool:
    from fractureflow.intersect_disk import disk_pair_intersect
    res = disk_pair_intersect(centers, normals, radii,
                              np.asarray([i], dtype=np.intp), np.asarray([j], dtype=np.intp))
    return bool(res[0])


def _intersects_matrix_ref(centers, normals, radii) -> List[Tuple[int, int]]:
    n = int(centers.shape[0])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return []
    from fractureflow.intersect_disk import disk_pair_intersect
    ii = np.asarray([p[0] for p in pairs], dtype=np.intp)
    jj = np.asarray([p[1] for p in pairs], dtype=np.intp)
    res = disk_pair_intersect(centers, normals, radii, ii, jj)
    return [pair for pair, hit in zip(pairs, res) if bool(hit)]


def pair_geometry(objects: Sequence[Mapping[str, Any]], adjacency_gap: float) -> Dict[str, Any]:
    """一次性算全对象对的判交/邻接矩阵 (供世界构造与结构派生; 非评估路径)。"""
    if not isinstance(objects, list):
        raise ContractError("pair_geometry 需要 objects list")
    gap = _finite_scalar("adjacency_gap", adjacency_gap)
    if gap < 0.0:
        raise ContractError("adjacency_gap 必须非负")
    centers = np.asarray([obj["center"] for obj in objects], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray([obj["normal"] for obj in objects], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray([obj["radius"] for obj in objects], dtype=np.float64).reshape(-1)
    ids = [str(obj["id"]) for obj in objects]
    n = len(ids)
    inter_pairs = set(_intersects_matrix_ref(centers, normals, radii))
    intersects = [[False] * n for _ in range(n)]
    adjacent = [[False] * n for _ in range(n)]
    gaps = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            hit = (i, j) in inter_pairs
            gap_ij = float(np.linalg.norm(centers[j] - centers[i])) - float(radii[i]) - float(radii[j])
            gaps[i][j] = gaps[j][i] = gap_ij
            intersects[i][j] = intersects[j][i] = hit
            adjacent[i][j] = adjacent[j][i] = bool((not hit) and gap_ij <= gap)
    return {"n": n, "ids": ids, "intersects": intersects, "adjacent": adjacent,
            "surface_gap": gaps, "adjacency_gap": gap}


def structure_from_mask(world_id: str, objects: Sequence[Mapping[str, Any]], mask: int,
                        adjacency_gap: float, *,
                        geometry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """由 world 对象表 + 在场位掩码派生 Structure (relations 由几何冻结规则生成)。

    bit i = 1 ⇔ objects[i] 在场; structure_id = "<world>:m<宽度 n 的 0/1 串>"。
    relations 只记在场对象之间的 intersects / adjacent 两类 (其余四类本单不产出)。
    """
    if isinstance(mask, bool) or not isinstance(mask, int):
        raise ContractError(f"mask 必须为整数, 收到 {mask!r}")
    n = len(objects)
    if mask < 0 or mask >= (1 << n):
        raise ContractError(f"mask 越界: {mask} 不在 [0, 2^{n})")
    geo = dict(geometry) if geometry is not None else pair_geometry(objects, adjacency_gap)
    if int(geo["n"]) != n:
        raise ContractError("pair_geometry.n 与 objects 数不一致")
    present = [i for i in range(n) if (mask >> i) & 1]
    sub_objects = [dict(objects[i]) for i in present]
    # relations 必须在**本结构自身**的对象集合上重算判交 (与 reference_evaluate 同一
    # 输入口径); 共享判交核对 scale 敏感, 用全世界的配对矩阵会与子集重算不一致。
    centers = np.asarray([obj["center"] for obj in sub_objects], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray([obj["normal"] for obj in sub_objects], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray([obj["radius"] for obj in sub_objects], dtype=np.float64).reshape(-1)
    hit_pairs = set(_intersects_matrix_ref(centers, normals, radii))
    relations: List[Dict[str, Any]] = []
    for iy in range(len(sub_objects)):
        for jx in range(iy + 1, len(sub_objects)):
            oi, oj = sub_objects[iy], sub_objects[jx]
            if (iy, jx) in hit_pairs:
                relations.append({"src": str(oi["id"]), "dst": str(oj["id"]), "kind": "intersects"})
            else:
                gap_ij = (math.sqrt((float(centers[jx][0]) - float(centers[iy][0])) ** 2
                                    + (float(centers[jx][1]) - float(centers[iy][1])) ** 2
                                    + (float(centers[jx][2]) - float(centers[iy][2])) ** 2)
                          - float(radii[iy]) - float(radii[jx]))
                if gap_ij <= float(adjacency_gap):
                    relations.append({"src": str(oi["id"]), "dst": str(oj["id"]),
                                      "kind": "adjacent"})
    return {"structure_id": f"{world_id}:m{mask:0{n}b}",
            "mask": int(mask),
            "objects": sub_objects,
            "relations": relations,
            "n_universe_objects": n}


def feasible_masks(n: int, records: Sequence[Mapping[str, Any]],
                   object_ids: Sequence[str]) -> List[int]:
    """观测一致性全枚举: 返回与全部观测记录一致的全部在场掩码 (升序)。

    一条记录 = {"object_id": str, "present": bool}; 同一对象的两条记录互相矛盾
    ⇒ 可行集为空 (这是 MODEL_CONFLICT 的构造来源, 不是物理不存在)。
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ContractError(f"n 非法: {n!r}")
    ids = [str(x) for x in object_ids]
    if len(set(ids)) != len(ids) or len(ids) != n:
        raise ContractError("object_ids 与 n 不一致或含重复")
    index = {oid: i for i, oid in enumerate(ids)}
    pinned: Dict[int, bool] = {}
    for rec in records:
        if not isinstance(rec, Mapping):
            raise ContractError("观测记录必须为 Mapping")
        oid = _as_str("observation.object_id", rec.get("object_id"))
        if oid not in index:
            raise ContractError(f"观测记录引用未知对象 {oid!r}")
        val = rec.get("present")
        if not isinstance(val, bool):
            raise ContractError(f"observation.present 必须为 bool, 收到 {val!r}")
        idx = index[oid]
        if idx in pinned and pinned[idx] != val:
            return []  # 观测冲突: 可行集为空
        pinned[idx] = val
    out: List[int] = []
    for mask in range(1 << n):
        if all(bool((mask >> idx) & 1) == val for idx, val in pinned.items()):
            out.append(mask)
    return out


def make_feasible_set(world_id: str, objects: Sequence[Mapping[str, Any]],
                      masks: Sequence[int], adjacency_gap: float, *,
                      geometry: Optional[Mapping[str, Any]] = None,
                      envelope_id: Optional[str] = None) -> "FeasibleSet":
    geo = dict(geometry) if geometry is not None else pair_geometry(objects, adjacency_gap)
    structures = [structure_from_mask(world_id, objects, int(m), adjacency_gap, geometry=geo)
                  for m in masks]
    return FeasibleSet(structures=structures, universe_size=len(objects),
                       envelope_id=envelope_id or f"{world_id}:env")


# ---------------------------------------------------------------------------
# FeasibleSet / QueryAnswer
# ---------------------------------------------------------------------------

@dataclass
class FeasibleSet:
    """显式可行集 E(Y): 逐结构全枚举承载 (空列表 = MODEL_CONFLICT 的唯一来源)。

    universe_size 是允许空间的对象总数 (2^n 全枚举的 n); n=0 的"空世界查询"
    语义不存在 ⇒ NotApplicableError (显式 NOT_APPLICABLE, 禁静默 PASS)。
    """
    structures: List[Mapping[str, Any]]
    universe_size: int
    envelope_id: str = "envelope"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if isinstance(self.universe_size, bool) or not isinstance(self.universe_size, int):
            raise ContractError(f"FeasibleSet.universe_size 必须为整数, 收到 {self.universe_size!r}")
        _as_str("envelope_id", self.envelope_id)
        if self.universe_size <= 0:
            raise NotApplicableError(
                "n=0 圆盘世界无查询语义: NOT_APPLICABLE (禁静默 PASS; 空包络只由非空允许空间的"
                "观测冲突产生)")
        if not isinstance(self.structures, list):
            raise ContractError("FeasibleSet.structures 必须为 list")
        seen = set()
        for s in self.structures:
            info = validate_structure(s)
            sid = info["structure_id"]
            if sid in seen:
                raise ContractError(f"可行集内 structure_id 重复: {sid!r}")
            seen.add(sid)

    @property
    def is_empty(self) -> bool:
        return len(self.structures) == 0

    def ids(self) -> List[str]:
        return [str(s["structure_id"]) for s in self.structures]

    def by_id(self) -> Dict[str, Mapping[str, Any]]:
        return {str(s["structure_id"]): s for s in self.structures}

    def to_dict(self) -> Dict[str, Any]:
        return {"envelope_id": self.envelope_id, "universe_size": int(self.universe_size),
                "n_structures": len(self.structures),
                "is_empty": self.is_empty,
                "structure_ids": self.ids()}


@dataclass(frozen=True)
class QueryAnswer:
    """五状态查询答案; gamma 为投影得到的答案集合 (frozenset[bool])。"""
    schema: str
    query_id: str
    status: str
    gamma: frozenset
    witness: Mapping[str, Any]
    reason: str
    n_feasible: int
    nodes_expanded: int
    elapsed_s: float
    search_complete: bool
    budget: Mapping[str, Any]
    proof_kind: str
    query: Mapping[str, Any]
    partial_projection: Tuple[Tuple[str, bool], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "query_id": self.query_id,
            "status": self.status,
            "gamma": sorted(bool(x) for x in self.gamma),
            "witness": {"positive": [str(x) for x in self.witness.get("positive", [])],
                        "negative": [str(x) for x in self.witness.get("negative", [])]},
            "reason": self.reason,
            "n_feasible": int(self.n_feasible),
            "nodes_expanded": int(self.nodes_expanded),
            "elapsed_s": round(float(self.elapsed_s), 6),
            "search_complete": bool(self.search_complete),
            "budget": dict(self.budget),
            "proof_kind": self.proof_kind,
            "query": dict(self.query),
            "partial_projection": [[str(sid), bool(lab)] for sid, lab in self.partial_projection],
        }


def _answer_doc(answer: Any) -> Dict[str, Any]:
    if isinstance(answer, QueryAnswer):
        return answer.to_dict()
    if isinstance(answer, Mapping):
        doc = dict(answer)
        if "gamma" not in doc:
            raise ContractError("answer 缺 gamma 字段")
        return doc
    raise ContractError(f"answer 必须为 QueryAnswer 或 Mapping, 收到 {type(answer).__name__}")


# ---------------------------------------------------------------------------
# 两条评估路
# ---------------------------------------------------------------------------

def api_evaluate(structure: Any, query: Any) -> bool:
    """API 路: 只消费结构关系图 + 附着闭式 (不重算判交几何, 作为独立两路之一)。"""
    info = validate_structure(structure)
    q = _as_query(query)
    objects = info["objects"]
    rels = _relation_set(info)
    if q.family == FAMILY_TWO_POINT_CONNECTIVITY:
        attach_a = _attached_ids(objects, q.params["A"], float(q.params["r_attach"]))
        attach_b = _attached_ids(objects, q.params["B"], float(q.params["r_attach"]))
        if not attach_a or not attach_b:
            return False
        edges = [(u, v) for (u, v, kind) in rels if kind == "intersects"]
        return _sides_connected_py([o["id"] for o in objects], edges, attach_a, attach_b)
    ids = [str(o["id"]) for o in objects]
    id_a = str(q.params["id_a"])
    id_b = str(q.params["id_b"])
    if id_a not in ids or id_b not in ids:
        return False
    kind = "intersects" if q.family == FAMILY_INTERSECTS else "adjacent"
    return _has_relation(info, id_a, id_b, kind)


def reference_evaluate(structure: Any, query: Any) -> bool:
    """参照路: 从对象几何重算 (判交用既有 intersect_disk, 附着用 connectivity_qoi 的闭式)。

    这是 verify 的独立重投影路: 它**不读** structure 里缓存的 relations。
    """
    info = validate_structure(structure)
    q = _as_query(query)
    centers, normals, radii, ids = _object_arrays(info)
    if q.family == FAMILY_TWO_POINT_CONNECTIVITY:
        if len(ids) == 0:
            return False
        from fractureflow.connectivity_qoi import disk_point_distance
        r_attach = float(q.params["r_attach"])
        pa = np.asarray(q.params["A"], dtype=np.float64).reshape(1, 3)
        pb = np.asarray(q.params["B"], dtype=np.float64).reshape(1, 3)
        da = disk_point_distance(centers, normals, radii, pa)[0]
        db = disk_point_distance(centers, normals, radii, pb)[0]
        attach_a = [ids[i] for i in range(len(ids)) if float(da[i]) <= r_attach]
        attach_b = [ids[i] for i in range(len(ids)) if float(db[i]) <= r_attach]
        if not attach_a or not attach_b:
            return False
        pairs = _intersects_matrix_ref(centers, normals, radii)
        edges = [(ids[i], ids[j]) for (i, j) in pairs]
        return _sides_connected_py(ids, edges, attach_a, attach_b)
    id_a = str(q.params["id_a"])
    id_b = str(q.params["id_b"])
    if id_a not in ids or id_b not in ids:
        return False
    i = ids.index(id_a)
    j = ids.index(id_b)
    hit = _pair_intersects_ref(centers, normals, radii, i, j)
    if q.family == FAMILY_INTERSECTS:
        return hit
    if hit:
        return False
    gap = float(np.linalg.norm(centers[j] - centers[i])) - float(radii[i]) - float(radii[j])
    return bool(gap <= float(q.params["adjacency_gap"]))


def evaluate(structure: Any, query: Any, evaluator: str = "api") -> bool:
    if evaluator == "api":
        return api_evaluate(structure, query)
    if evaluator == "reference":
        return reference_evaluate(structure, query)
    raise ContractError(f"evaluator 非法: {evaluator!r} (api|reference)")


# ---------------------------------------------------------------------------
# 投影 / 答案
# ---------------------------------------------------------------------------

def project_stats(envelope: Any, query: Any, evaluator: str = "api") -> Dict[str, Any]:
    """全枚举投影 + 枚举计数 (E1 完备性的"枚举数必须 = 可行结构数"由调用方断言)。"""
    if not isinstance(envelope, FeasibleSet):
        raise ContractError(f"project 需要 FeasibleSet, 收到 {type(envelope).__name__}")
    envelope.validate()
    q = _as_query(query)
    if evaluator not in ("api", "reference"):
        raise ContractError(f"evaluator 非法: {evaluator!r}")
    per_structure: List[Dict[str, Any]] = []
    labels = set()
    for structure in envelope.structures:
        label = bool(evaluate(structure, q, evaluator))
        labels.add(label)
        per_structure.append({"structure_id": str(structure["structure_id"]), "label": label})
    return {"gamma": frozenset(labels), "n_evaluated": len(per_structure),
            "n_feasible": len(envelope.structures), "evaluator": evaluator,
            "per_structure": per_structure}


def project(envelope: Any, query: Any, evaluator: str = "api") -> frozenset:
    """§4.2 冻结入口: project(envelope, query) -> frozenset[bool] (完整枚举)。"""
    return project_stats(envelope, query, evaluator)["gamma"]


def _make_answer(*, envelope: FeasibleSet, query: TopologyQuery, status: str,
                 gamma: frozenset, witness: Mapping[str, Any], reason: str,
                 n_feasible: int, nodes_expanded: int, elapsed_s: float,
                 search_complete: bool, budget: TopologyBudget,
                 proof_kind: str,
                 partial_projection: Sequence[Tuple[str, bool]] = ()) -> QueryAnswer:
    if status not in STATUSES:
        raise ContractError(f"status 非法: {status!r}")
    return QueryAnswer(schema=SCHEMA_QUERY_ANSWER, query_id=query.query_id, status=status,
                       gamma=frozenset(bool(x) for x in gamma),
                       witness={"positive": [str(x) for x in witness.get("positive", [])],
                                "negative": [str(x) for x in witness.get("negative", [])]},
                       reason=str(reason), n_feasible=int(n_feasible),
                       nodes_expanded=int(nodes_expanded), elapsed_s=float(elapsed_s),
                       search_complete=bool(search_complete), budget=budget.to_dict(),
                       proof_kind=str(proof_kind), query=query.to_dict(),
                       partial_projection=tuple((str(s), bool(l)) for s, l in partial_projection))


def answer(envelope: Any, query: Any, budget: Optional[TopologyBudget] = None) -> QueryAnswer:
    """§4.2 冻结入口: answer(envelope, query, budget) -> QueryAnswer (五状态)。

    判序: 空包络 → MODEL_CONFLICT (先于预算判定, 空集不经预算); 逐结构展开,
    展开数 > max_nodes 或墙钟 > max_seconds ⇒ SEARCH_UNRESOLVED (保守 γ={F,T},
    search_complete=False, 附已展开的 partial_projection); 否则按 γ 映射状态。
    """
    if not isinstance(envelope, FeasibleSet):
        raise ContractError(f"answer 需要 FeasibleSet, 收到 {type(envelope).__name__}")
    envelope.validate()
    q = _as_query(query)
    b = budget if budget is not None else TopologyBudget(max_nodes=max(len(envelope.structures), 1))
    if not isinstance(b, TopologyBudget):
        raise ContractError("budget 必须为 TopologyBudget")
    started = time.perf_counter()
    if envelope.is_empty:
        elapsed = time.perf_counter() - started
        return _make_answer(envelope=envelope, query=q, status=STATUS_MODEL_CONFLICT,
                            gamma=frozenset(), witness={},
                            reason="空可行集: 观测/核/数值界冲突 (不是物理不存在, 也不是 CERTIFIED_FALSE)",
                            n_feasible=0, nodes_expanded=0, elapsed_s=elapsed,
                            search_complete=True, budget=b, proof_kind=PROOF_NONE)
    order = sorted(envelope.structures, key=lambda s: str(s["structure_id"]))
    labels: List[Tuple[str, bool]] = []
    for structure in order:
        label = bool(api_evaluate(structure, q))
        labels.append((str(structure["structure_id"]), label))
        elapsed = time.perf_counter() - started
        if len(labels) > b.max_nodes or elapsed > b.max_seconds:
            return _make_answer(
                envelope=envelope, query=q, status=STATUS_SEARCH_UNRESOLVED,
                gamma=CONSERVATIVE_GAMMA, witness={},
                reason=("预算耗尽 (nodes_expanded=%d > max_nodes=%d 或 elapsed_s=%.6f > max_seconds=%.6f): "
                        "未完成全枚举, 只能 SEARCH_UNRESOLVED" % (len(labels), b.max_nodes, elapsed,
                                                                   b.max_seconds)),
                n_feasible=len(order), nodes_expanded=len(labels), elapsed_s=elapsed,
                search_complete=False, budget=b, proof_kind=PROOF_NONE,
                partial_projection=labels)
    elapsed = time.perf_counter() - started
    gamma = frozenset(lab for _, lab in labels)
    status = STATUS_OF_GAMMA[gamma]
    positive = [sid for sid, lab in labels if lab]
    negative = [sid for sid, lab in labels if not lab]
    if status == STATUS_CERTIFIED_TRUE:
        witness = {"positive": positive[:1], "negative": []}
        proof = PROOF_COMPLETE_ENUMERATION
    elif status == STATUS_CERTIFIED_FALSE:
        witness = {"positive": [], "negative": negative[:1]}
        proof = PROOF_COMPLETE_ENUMERATION
    elif status == STATUS_AMBIGUOUS:
        witness = {"positive": positive[:1], "negative": negative[:1]}
        proof = PROOF_NONE
    else:  # 不可达 (非空可行集的 gamma 至少 1 个元素)
        raise ContractError(f"内部错误: 非空可行集 gamma 非法 {gamma!r}")
    reason = ("完整枚举 %d 个可行结构, 投影 γ 有 %d 个不同答案 ⇒ %s"
              % (len(labels), len(gamma), status))
    return _make_answer(envelope=envelope, query=q, status=status, gamma=gamma,
                        witness=witness, reason=reason, n_feasible=len(order),
                        nodes_expanded=len(labels), elapsed_s=elapsed,
                        search_complete=True, budget=b, proof_kind=proof)


# ---------------------------------------------------------------------------
# 独立验证器 (§4.2: 不调用 answer, 从 envelope 重新投影)
# ---------------------------------------------------------------------------

def verify(answer_obj: Any, envelope: Any, query: Any) -> Dict[str, Any]:
    """独立重投影验证: 只用 reference 路重算; 不调用 answer()。

    检查面: 状态↔γ 映射; 单值证书的 γ 与独立投影一致; 见证在包络内且按其
    自身结构重放; 空包络只许 MODEL_CONFLICT; 预算耗尽只许 SEARCH_UNRESOLVED
    且确由 recorded budget 触发; 查询参数与答案快照一致 (抓"改坐标不换证书")。
    """
    if not isinstance(envelope, FeasibleSet):
        raise ContractError("verify 需要 FeasibleSet")
    doc = _answer_doc(answer_obj)
    q = _as_query(query)
    mismatch: List[str] = []
    stats = project_stats(envelope, q, evaluator="reference")
    ref_gamma = stats["gamma"]
    feasible_ids = set(envelope.ids())
    status = doc.get("status")
    if status not in STATUSES:
        mismatch.append(f"UNKNOWN_STATUS:{status!r}")
    try:
        gamma = frozenset(bool(x) for x in doc.get("gamma", []))
    except (TypeError, ValueError):
        gamma = frozenset()
        mismatch.append("GAMMA_NOT_BOOL_SEQUENCE")
    witness = doc.get("witness") or {}
    pos = [str(x) for x in witness.get("positive", []) or []]
    neg = [str(x) for x in witness.get("negative", []) or []]
    if doc.get("query_id") != q.query_id:
        mismatch.append(f"QUERY_ID_MISMATCH:{doc.get('query_id')!r}!={q.query_id!r}")
    snapshot = doc.get("query")
    if snapshot != q.to_dict():
        mismatch.append("QUERY_SNAPSHOT_STALE: 答案记录的查询参数与验证查询不一致 (改了参数没换证书)")
    budget = doc.get("budget") or {}
    try:
        max_nodes = int(budget.get("max_nodes"))
        max_seconds = float(budget.get("max_seconds"))
    except (TypeError, ValueError):
        max_nodes, max_seconds = None, None
        mismatch.append("BUDGET_UNREADABLE")
    nodes_expanded = doc.get("nodes_expanded")
    elapsed_s = doc.get("elapsed_s")
    exhausted = None
    if max_nodes is not None and isinstance(nodes_expanded, int) and not isinstance(nodes_expanded, bool):
        time_exhausted = (isinstance(elapsed_s, (int, float)) and not isinstance(elapsed_s, bool)
                          and float(elapsed_s) > max_seconds)
        exhausted = bool(nodes_expanded > max_nodes or time_exhausted)

    complete_statuses = (STATUS_CERTIFIED_TRUE, STATUS_CERTIFIED_FALSE, STATUS_AMBIGUOUS)
    if status in complete_statuses:
        if doc.get("search_complete") is not True:
            mismatch.append("COMPLETE_STATUS_MUST_HAVE_search_complete_true")
        if exhausted is True:
            mismatch.append("COMPLETE_STATUS_BUT_BUDGET_EXHAUSTED")

    def _check_witness_label(sid: str, want: bool) -> None:
        if sid not in feasible_ids:
            mismatch.append(f"WITNESS_NOT_IN_ENVELOPE:{sid}")
            return
        structure = envelope.by_id()[sid]
        got = bool(reference_evaluate(structure, q))
        if got != want:
            mismatch.append(f"WITNESS_RECOMPUTE_MISMATCH:{sid}:claimed={want}:recomputed={got}")

    if status == STATUS_SEARCH_UNRESOLVED:
        if exhausted is not True:
            mismatch.append("SEARCH_UNRESOLVED_WITHOUT_BUDGET_EXHAUSTION")
        if envelope.is_empty:
            mismatch.append("SEARCH_UNRESOLVED_ON_EMPTY_ENVELOPE")
        if gamma != CONSERVATIVE_GAMMA:
            mismatch.append("SEARCH_UNRESOLVED_GAMMA_MUST_BE_CONSERVATIVE_FT")
        if doc.get("search_complete") is not False:
            mismatch.append("SEARCH_UNRESOLVED_MUST_HAVE_search_complete_false")
        if ref_gamma == frozenset():
            mismatch.append("SEARCH_UNRESOLVED_WHILE_COMPLETE_PROJECTION_EMPTY")
    elif status == STATUS_MODEL_CONFLICT:
        if not envelope.is_empty:
            mismatch.append("MODEL_CONFLICT_ON_NONEMPTY_ENVELOPE")
        if ref_gamma != frozenset():
            mismatch.append("MODEL_CONFLICT_BUT_INDEPENDENT_PROJECTION_NONEMPTY")
        if gamma != frozenset():
            mismatch.append("MODEL_CONFLICT_GAMMA_MUST_BE_EMPTY")
        if pos or neg:
            mismatch.append("MODEL_CONFLICT_MUST_NOT_CARRY_WITNESS")
        if doc.get("search_complete") is not True:
            mismatch.append("MODEL_CONFLICT_MUST_HAVE_search_complete_true")
    elif status == STATUS_CERTIFIED_TRUE:
        if gamma != frozenset({True}):
            mismatch.append("CERTIFIED_TRUE_GAMMA_MUST_BE_TRUE_ONLY")
        if ref_gamma != frozenset({True}):
            mismatch.append("CERTIFIED_TRUE_INDEPENDENT_PROJECTION_NOT_TRUE_ONLY")
        if len(pos) != 1 or neg:
            mismatch.append("CERTIFIED_TRUE_NEEDS_EXACTLY_ONE_POSITIVE_WITNESS_AND_NO_NEGATIVE")
        for sid in pos:
            _check_witness_label(sid, True)
        if doc.get("proof_kind") != PROOF_COMPLETE_ENUMERATION:
            mismatch.append("CERTIFIED_MUST_CARRY_COMPLETE_ENUMERATION_PROOF")
    elif status == STATUS_CERTIFIED_FALSE:
        if gamma != frozenset({False}):
            mismatch.append("CERTIFIED_FALSE_GAMMA_MUST_BE_FALSE_ONLY")
        if ref_gamma != frozenset({False}):
            mismatch.append("CERTIFIED_FALSE_INDEPENDENT_PROJECTION_NOT_FALSE_ONLY")
        if len(neg) != 1 or pos:
            mismatch.append("CERTIFIED_FALSE_NEEDS_EXACTLY_ONE_NEGATIVE_WITNESS_AND_NO_POSITIVE")
        for sid in neg:
            _check_witness_label(sid, False)
        if doc.get("proof_kind") != PROOF_COMPLETE_ENUMERATION:
            mismatch.append("CERTIFIED_MUST_CARRY_COMPLETE_ENUMERATION_PROOF")
    elif status == STATUS_AMBIGUOUS:
        if gamma != CONSERVATIVE_GAMMA:
            mismatch.append("AMBIGUOUS_GAMMA_MUST_BE_BOTH")
        if ref_gamma != CONSERVATIVE_GAMMA:
            mismatch.append("AMBIGUOUS_INDEPENDENT_PROJECTION_NOT_BOTH")
        if len(pos) != 1 or len(neg) != 1:
            mismatch.append("AMBIGUOUS_NEEDS_BOTH_WITNESSES")
        for sid in pos:
            _check_witness_label(sid, True)
        for sid in neg:
            _check_witness_label(sid, False)
    return {"ok": not mismatch, "independent_projection": stats["per_structure"],
            "independent_gamma": sorted(ref_gamma), "mismatch": mismatch,
            "n_independent": stats["n_evaluated"], "evaluator": "REFERENCE_GEOMETRY",
            "budget_exhausted": exhausted}


# ---------------------------------------------------------------------------
# 与 identification.Envelope 的语义对照 (E10; 仅 TWO_POINT 族有对应语义)
# ---------------------------------------------------------------------------

def connection_view(structure: Any, query: Any) -> Dict[str, Any]:
    """把结构在 TWO_POINT 查询下的连接证据投影成 ident 三要素 (reference 几何路)。"""
    info = validate_structure(structure)
    q = _as_query(query)
    if q.family != FAMILY_TWO_POINT_CONNECTIVITY:
        raise NotApplicableError("identification.Envelope 对照只对 TWO_POINT_CONNECTIVITY 族适用")
    centers, normals, radii, ids = _object_arrays(info)
    if len(ids) == 0:
        return {"present": [], "edges": [], "attach_a": [], "attach_b": [], "connected": False}
    from fractureflow.connectivity_qoi import disk_point_distance
    r_attach = float(q.params["r_attach"])
    da = disk_point_distance(centers, normals, radii,
                             np.asarray(q.params["A"], dtype=np.float64).reshape(1, 3))[0]
    db = disk_point_distance(centers, normals, radii,
                             np.asarray(q.params["B"], dtype=np.float64).reshape(1, 3))[0]
    attach_a = [ids[i] for i in range(len(ids)) if float(da[i]) <= r_attach]
    attach_b = [ids[i] for i in range(len(ids)) if float(db[i]) <= r_attach]
    edges = [(ids[i], ids[j]) for (i, j) in _intersects_matrix_ref(centers, normals, radii)]
    connected = bool(attach_a and attach_b
                     and _sides_connected_py(ids, edges, attach_a, attach_b))
    return {"present": ids, "edges": edges, "attach_a": attach_a,
            "attach_b": attach_b, "connected": connected}


def identification_view(envelope: Any, query: Any) -> Dict[str, Any]:
    """§4.2 语义对照: ident.Envelope 的 must/may 区域标签 vs 完整枚举投影 γ。

    region_label 是**保守抽象**: ALL_CONNECTED ⇒ 全部可行结构连通 (γ={T});
    ALL_DISCONNECTED ⇒ 全部不连通 (γ={F}); UNRESOLVED ⇒ 不作断言。
    该路可能给出比完整枚举更保守的 UNRESOLVED (sound but incomplete), 故
    只作语义对照与描述统计, 不替代 Γ_Q。
    """
    if not isinstance(envelope, FeasibleSet):
        raise ContractError("identification_view 需要 FeasibleSet")
    q = _as_query(query)
    gamma = project(envelope, q, evaluator="reference")
    if envelope.is_empty:
        return {"applicable": True, "model_conflict": True, "region_label": None,
                "n_states": 0, "gamma": [], "violations": [], "incomplete_but_single_value": False}
    from fractureflow import identification as ident
    derivations = []
    for k, structure in enumerate(envelope.structures):
        view = connection_view(structure, q)
        derivations.append(ident.StateDerivation(
            state_id=k,
            nodes=frozenset(view["present"]),
            edges=frozenset(view["edges"]),
            a_side=frozenset(view["attach_a"]),
            b_side=frozenset(view["attach_b"]),
            attach_a=bool(view["attach_a"]),
            attach_b=bool(view["attach_b"]),
            label=int(bool(view["connected"])),
        ))
    env = ident.envelope_of(derivations)
    label = ident.region_label(env)
    violations: List[str] = []
    if label == ident.RegionLabel.ALL_CONNECTED.value and gamma != frozenset({True}):
        violations.append("REGION_ALL_CONNECTED_BUT_GAMMA_NOT_TRUE")
    if label == ident.RegionLabel.ALL_DISCONNECTED.value and gamma != frozenset({False}):
        violations.append("REGION_ALL_DISCONNECTED_BUT_GAMMA_NOT_FALSE")
    complete_single = len(gamma) == 1
    return {"applicable": True, "model_conflict": False, "region_label": label,
            "must_connected": bool(ident.must_connected(env)),
            "may_connected": bool(ident.may_connected(env)),
            "n_states": int(env.n_states), "gamma": sorted(gamma),
            "violations": violations,
            "incomplete_but_single_value": bool(label == ident.RegionLabel.UNRESOLVED.value
                                                and complete_single),
            "element_counts": env.element_counts()}


__all__ = [
    "SCHEMA_TOPOLOGY_QUERY", "SCHEMA_QUERY_ANSWER",
    "FAMILY_TWO_POINT_CONNECTIVITY", "FAMILY_INTERSECTS", "FAMILY_ADJACENT", "FAMILIES",
    "STATUS_CERTIFIED_TRUE", "STATUS_CERTIFIED_FALSE", "STATUS_AMBIGUOUS",
    "STATUS_MODEL_CONFLICT", "STATUS_SEARCH_UNRESOLVED", "STATUSES",
    "STATUS_OF_GAMMA", "SINGLE_VALUE_STATUSES", "CONSERVATIVE_GAMMA",
    "PROOF_COMPLETE_ENUMERATION", "PROOF_NONE",
    "RELATION_KINDS", "OBJECT_TYPE_DISK",
    "ContractError", "NotApplicableError",
    "TopologyBudget", "TopologyQuery", "FeasibleSet", "QueryAnswer",
    "validate_structure", "pair_geometry", "structure_from_mask", "feasible_masks",
    "make_feasible_set", "api_evaluate", "reference_evaluate", "evaluate",
    "project", "project_stats", "answer", "verify",
    "connection_view", "identification_view",
]
