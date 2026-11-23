# -*- coding: utf-8 -*-
"""S2 · 部分赋值 + 区域细化 + must/may 外包 + 独立证书验证器（N0 完整字典）。

本件是任务单 S2-N0（sha12 ``e1949ba6ad48``）§2.1 要求的**新建模块**；``finite_scope.py`` 与
``identification.py`` 一字未改（只 import）。

口径（判据权威 = 任务单 §4 + ``results/s2_certificate_solver/prereg_s2.json``）：

- 区域 = **有限个互斥子立方的并**（部分赋值）。A12 的 128 个基线类与 B13 的五个情景的允许集
  实测都是**单个**子立方；多立方只为一般性保留。
- 外包从部分赋值**符号化**推出，不逐状态枚举：must 边 = 区域内每个状态都为 1 的边；
  may 边 = 区域内存在状态为 1 的边。这条是"不逐状态全枚举"的全部实质；显式枚举只用于
  对账（§2.1 第 4 条）与验证侧，成本分栏报。
- 方向规则用 ``identification.region_label``（must 有路 ⇒ 全连；may 无路 ⇒ 全断；否则未解决）。
  两条方向的可靠性：must 路 ⇒ 区域内每个状态都含该路 ⇒ 全连通；may 无路 ⇒ 不存在连通状态
  （否则其路径边全为"未固定为 0"，构成 may 路，矛盾）⇒ 全不连通。
- 验证器 ``verify_certificate`` **不调用** ``solve_partial`` / ``certify``，不读任何 ``certified``
  布尔；它自己重算允许集、树结构、外包、标签、见证与覆盖，并用三路外包互校。
- 本层不写盘；输出只含整数、布尔、字符串与整数对，不产浮点科学读数（耗时除外，标 MEASURED_ONLY）。
"""
from __future__ import annotations

import itertools
import json
import math
import time
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fractureflow import finite_scope as fs
from fractureflow import identification as ident

SCHEMA_CERT = "s2.certificate/v1"
SCHEMA_VERIFY = "s2.verify/v1"
SCHEMA_PROBE = "s2.b_layer_probe/v1"
VERIFIER_VERSION = "s2.verifier/1"
SELECTION_PROVENANCE = "SYMBOLIC_ENVELOPE"
CERT_KEYS = ("schema", "certificate_id", "model_id", "table_id", "query_id", "n_bits",
             "allowed_state_ids", "label_claim", "proof_route", "regions", "witness",
             "verifier_version", "selection_provenance", "budget", "costs", "stats", "watermark")
FORBIDDEN_KEY_TOKENS = ("truth", "oracle", "hidden")

REPO_ROOT = Path(__file__).resolve().parents[2]


class SolverError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 表结构描述（适配器填，求解器与验证器都只读它）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableStructure:
    """有限表的**结构性事实**：位 → 对象 的映射方式。

    ``EDGE_BITS``：位 = 一条边的存在性，节点恒在场（A12 型）。
    ``DISK_BITS``：位 = 一个不确定盘的存在性，边由在场盘诱导（B13 型）。
    """

    kind: str
    n_bits: int
    nodes: Tuple[str, ...] = ()
    edges: Tuple[Tuple[int, str, str], ...] = ()
    a_nodes: Tuple[str, ...] = ()
    b_nodes: Tuple[str, ...] = ()
    a_incident_bits: Tuple[int, ...] = ()
    b_incident_bits: Tuple[int, ...] = ()
    certain: Tuple[str, ...] = ()
    uncertain: Tuple[str, ...] = ()
    contact_edges: Tuple[Tuple[int, int], ...] = ()
    attach_a_disks: Tuple[int, ...] = ()
    attach_b_disks: Tuple[int, ...] = ()

    def __post_init__(self):
        if self.kind not in ("EDGE_BITS", "DISK_BITS"):
            raise SolverError(f"未知表结构 kind={self.kind!r}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_bits": int(self.n_bits), "nodes": list(self.nodes),
                "edges": [[int(b), u, v] for (b, u, v) in self.edges],
                "a_nodes": list(self.a_nodes), "b_nodes": list(self.b_nodes),
                "a_incident_bits": [int(x) for x in self.a_incident_bits],
                "b_incident_bits": [int(x) for x in self.b_incident_bits],
                "certain": list(self.certain), "uncertain": list(self.uncertain),
                "contact_edges": [[int(i), int(j)] for (i, j) in self.contact_edges],
                "attach_a_disks": [int(x) for x in self.attach_a_disks],
                "attach_b_disks": [int(x) for x in self.attach_b_disks]}

    @staticmethod
    def from_dict(d: Mapping) -> "TableStructure":
        return TableStructure(
            kind=str(d["kind"]), n_bits=int(d["n_bits"]),
            nodes=tuple(str(x) for x in d.get("nodes", ())),
            edges=tuple((int(b), str(u), str(v)) for b, u, v in d.get("edges", ())),
            a_nodes=tuple(str(x) for x in d.get("a_nodes", ())),
            b_nodes=tuple(str(x) for x in d.get("b_nodes", ())),
            a_incident_bits=tuple(int(x) for x in d.get("a_incident_bits", ())),
            b_incident_bits=tuple(int(x) for x in d.get("b_incident_bits", ())),
            certain=tuple(str(x) for x in d.get("certain", ())),
            uncertain=tuple(str(x) for x in d.get("uncertain", ())),
            contact_edges=tuple((int(i), int(j)) for i, j in d.get("contact_edges", ())),
            attach_a_disks=tuple(int(x) for x in d.get("attach_a_disks", ())),
            attach_b_disks=tuple(int(x) for x in d.get("attach_b_disks", ())))


def structure_of(table: fs.FiniteTable) -> TableStructure:
    raw = table.tags.get("s2_structure")
    if not raw:
        raise SolverError(f"{table.table_id}: 表未登记 s2_structure")
    return TableStructure.from_dict(raw if isinstance(raw, dict) else json.loads(raw))


# ---------------------------------------------------------------------------
# 子立方（部分赋值）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cube:
    """部分赋值：fixed_mask 上的位固定为 fixed_vals 对应值，其余位自由。"""

    fixed_mask: int
    fixed_vals: int

    def __post_init__(self):
        if self.fixed_vals & ~self.fixed_mask:
            raise SolverError("子立方非法：固定值越出固定掩码")

    def varying(self, n_bits: int) -> List[int]:
        full = (1 << n_bits) - 1
        free = full & ~self.fixed_mask
        return [b for b in range(n_bits) if (free >> b) & 1]

    def count(self, n_bits: int) -> int:
        return 1 << len(self.varying(n_bits))

    def states(self, n_bits: int) -> List[int]:
        free = [b for b in range(n_bits) if not (self.fixed_mask >> b) & 1]
        out = [self.fixed_vals]
        for b in free:
            out = out + [s | (1 << b) for s in out]
        return sorted(out)

    def min_state(self) -> int:
        return int(self.fixed_vals)

    def forces_one(self, bit: int) -> bool:
        return bool((self.fixed_mask >> bit) & 1) and bool((self.fixed_vals >> bit) & 1)

    def forces_zero(self, bit: int) -> bool:
        return bool((self.fixed_mask >> bit) & 1) and not bool((self.fixed_vals >> bit) & 1)

    def allows_one(self, bit: int) -> bool:
        return not self.forces_zero(bit)

    def to_dict(self) -> dict:
        return {"fixed_mask": int(self.fixed_mask), "fixed_vals": int(self.fixed_vals)}


def region_states(cubes: Sequence[Cube], n_bits: int) -> List[int]:
    seen: set = set()
    for c in cubes:
        seen.update(c.states(n_bits))
    return sorted(seen)


def region_count(cubes: Sequence[Cube], n_bits: int) -> int:
    return sum(c.count(n_bits) for c in cubes)


def cubes_disjoint(a: Cube, b: Cube) -> bool:
    common = a.fixed_mask & b.fixed_mask
    return bool((a.fixed_vals ^ b.fixed_vals) & common)


def cubes_partition(parent: Cube, children: Sequence[Cube], n_bits: int) -> Tuple[bool, List[str]]:
    """符号化判分：子立方两两不交、各 ⊆ 父、计数和 = 父计数。"""
    reasons: List[str] = []
    if not children:
        return False, ["无子区域"]
    for ch in children:
        if ch.fixed_mask & parent.fixed_mask != parent.fixed_mask:
            reasons.append("子区域未继承父区域的固定位")
        elif (ch.fixed_vals ^ parent.fixed_vals) & parent.fixed_mask:
            reasons.append("子区域与父区域在父固定位上取值不同")
        if ch.count(n_bits) <= 0:
            reasons.append("子区域为空")
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            if not cubes_disjoint(children[i], children[j]):
                reasons.append("兄弟区域重叠")
    if region_count(children, n_bits) != parent.count(n_bits):
        reasons.append(f"子区域计数和 {region_count(children, n_bits)} != 父区域 "
                       f"{parent.count(n_bits)}")
    return not reasons, reasons


# ---------------------------------------------------------------------------
# 符号化 must/may 外包（求解侧的唯一算力来源）
# ---------------------------------------------------------------------------

def _cube_forces_attach(c: Cube, incident_bits: Sequence[int]) -> bool:
    return any(c.forces_one(b) for b in incident_bits)


def _cube_may_attach(c: Cube, incident_bits: Sequence[int]) -> bool:
    return any(c.allows_one(b) for b in incident_bits)


def symbolic_envelope(struct: TableStructure, cubes: Sequence[Cube], n_bits: int) -> ident.Envelope:
    """从部分赋值直接推 must/may；不做任何逐状态遍历。"""
    if not cubes:
        raise SolverError("空区域不得建外包（恒真谓词禁当证据）")
    n_states = region_count(cubes, n_bits)
    if n_states <= 0:
        raise SolverError("空区域不得建外包")
    if struct.kind == "EDGE_BITS":
        must_edges = frozenset((u, v) for (b, u, v) in struct.edges
                               if all(c.forces_one(b) for c in cubes))
        may_edges = frozenset((u, v) for (b, u, v) in struct.edges
                              if any(c.allows_one(b) for c in cubes))
        must_nodes = frozenset(struct.nodes)
        may_nodes = frozenset(struct.nodes)
        must_attach_a = all(_cube_forces_attach(c, struct.a_incident_bits) for c in cubes)
        may_attach_a = any(_cube_may_attach(c, struct.a_incident_bits) for c in cubes)
        must_attach_b = all(_cube_forces_attach(c, struct.b_incident_bits) for c in cubes)
        may_attach_b = any(_cube_may_attach(c, struct.b_incident_bits) for c in cubes)
        must_a = frozenset(struct.a_nodes) if must_attach_a else frozenset()
        may_a = frozenset(struct.a_nodes) if may_attach_a else frozenset()
        must_b = frozenset(struct.b_nodes) if must_attach_b else frozenset()
        may_b = frozenset(struct.b_nodes) if may_attach_b else frozenset()
        return ident.Envelope(n_states, must_nodes, may_nodes, must_edges, may_edges,
                              must_a, may_a, must_b, may_b,
                              must_attach_a, may_attach_a, must_attach_b, may_attach_b)
    # DISK_BITS
    n_unc = len(struct.uncertain)
    base = len(struct.certain)
    must_nodes: set = set(range(base))
    may_nodes: set = set(range(base))
    for j in range(n_unc):
        idx = base + j
        if all(c.forces_one(j) for c in cubes):
            must_nodes.add(idx)
        if any(c.allows_one(j) for c in cubes):
            may_nodes.add(idx)
    must_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                           if i in must_nodes and j in must_nodes)
    may_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                          if i in may_nodes and j in may_nodes)
    must_a = frozenset(x for x in must_nodes if x in struct.attach_a_disks)
    may_a = frozenset(x for x in may_nodes if x in struct.attach_a_disks)
    must_b = frozenset(x for x in must_nodes if x in struct.attach_b_disks)
    may_b = frozenset(x for x in may_nodes if x in struct.attach_b_disks)
    return ident.Envelope(n_states, frozenset(must_nodes), frozenset(may_nodes),
                          must_edges, may_edges, must_a, may_a, must_b, may_b,
                          bool(must_a), bool(may_a), bool(must_b), bool(may_b))


def _bfs_path(adj: Mapping, start: Iterable, goals: Iterable) -> Optional[List[Tuple]]:
    goals = set(goals)
    prev: Dict = {}
    seen = set(start)
    frontier = list(start)
    if not frontier or not goals:
        return None
    for x in frontier:
        if x in goals:
            return []
    while frontier:
        nxt = []
        for x in frontier:
            for (y, tag) in adj.get(x, ()):
                if y in seen:
                    continue
                seen.add(y)
                prev[y] = (x, tag)
                if y in goals:
                    path = []
                    cur = y
                    while cur in prev:
                        p, t = prev[cur]
                        path.append((p, cur, t))
                        cur = p
                    return list(reversed(path))
                nxt.append(y)
        frontier = nxt
    return None


def _adjacency(edges: Iterable[Tuple], tags: Optional[Mapping] = None) -> Dict:
    adj: Dict = {}
    for e in edges:
        u, v = e[0], e[1]
        tag = tags.get((u, v)) if tags else None
        adj.setdefault(u, []).append((v, tag))
        adj.setdefault(v, []).append((u, tag))
    return adj


def may_path_bits(struct: TableStructure, env: ident.Envelope) -> List[int]:
    """may 图上一条 A→B 路径所经过的**位**（用于挑分裂位；不看标签、不看真值）。"""
    if not (env.may_attach_a and env.may_attach_b):
        return []
    adj = _adjacency(env.may_edges)
    path = _bfs_path(adj, env.may_a_side & env.may_nodes, env.may_b_side & env.may_nodes)
    if not path:
        return []
    bit_of = {(u, v): b for (b, u, v) in struct.edges}
    out = []
    for (u, v, _t) in path:
        b = bit_of.get((u, v), bit_of.get((v, u)))
        if b is not None:
            out.append(int(b))
    return out


def choose_split_bit(struct: TableStructure, cubes: Sequence[Cube], env: ident.Envelope,
                     n_bits: int) -> int:
    """确定性分裂位：优先 may 路径上最低位未固定位；否则最低位未固定位。"""
    if len(cubes) != 1:
        raise SolverError("多立方区域应按立方拆分，不走位分裂")
    cube = cubes[0]
    varying = set(cube.varying(n_bits))
    on_path = [b for b in may_path_bits(struct, env) if b in varying]
    if on_path:
        return min(on_path)
    if not varying:
        raise SolverError("区域已固定全部位却仍未解决：仪器缺陷")
    return min(varying)


# ---------------------------------------------------------------------------
# 查询解析（允许集）
# ---------------------------------------------------------------------------

def query_kind(table: fs.FiniteTable) -> str:
    q = table.tags.get("s2_queries")
    if not q:
        raise SolverError(f"{table.table_id}: 表未登记 s2_queries")
    return str(q["kind"] if isinstance(q, dict) else json.loads(q)["kind"])


def _queries_meta(table: fs.FiniteTable) -> dict:
    q = table.tags.get("s2_queries")
    if not q:
        raise SolverError(f"{table.table_id}: 表未登记 s2_queries")
    return q if isinstance(q, dict) else json.loads(q)


def resolve_query_cubes(table: fs.FiniteTable, query_id: str) -> List[Cube]:
    """query_id → 允许集的子立方分解（求解侧；验证器另有独立实现）。"""
    meta = _queries_meta(table)
    kind = str(meta["kind"])
    if kind == "A12_CLASS":
        prefix = "A12:class:"
        if not query_id.startswith(prefix):
            raise SolverError(f"query_id 形态不符: {query_id!r}")
        code = int(query_id[len(prefix):])
        codes = table.baseline_codes()
        members = np.flatnonzero(codes == code)
        if members.size == 0:
            return []
        return decompose_states([int(s) for s in members.tolist()], table.n_bits)
    if kind == "B13_SCENARIO":
        prefix = "B13:scenario:"
        if not query_id.startswith(prefix):
            raise SolverError(f"query_id 形态不符: {query_id!r}")
        name = query_id[len(prefix):]
        sc = meta["scenarios"].get(name)
        if sc is None:
            raise SolverError(f"未知情景 {name!r}")
        if not sc.get("observation_in_forward_image", True):
            return []
        mask = 0
        vals = 0
        for (bit, val) in sc["reads"]:
            mask |= 1 << int(bit)
            if int(val):
                vals |= 1 << int(bit)
        return [Cube(mask, vals)]
    raise SolverError(f"未知查询类型 {kind!r}")


def decompose_states(states: Sequence[int], n_bits: int) -> List[Cube]:
    """状态集 → 互斥子立方（子立方输入返回单立方）。"""
    ss = sorted(int(s) for s in states)
    if not ss:
        return []
    fixed = (1 << n_bits) - 1
    for s in ss[1:]:
        fixed &= ~(s ^ ss[0])
    vals = ss[0] & fixed
    cube = Cube(fixed, vals)
    if cube.count(n_bits) == len(ss):
        return [cube]
    varying = [b for b in range(n_bits) if not (fixed >> b) & 1]
    for b in varying:
        z = [s for s in ss if not (s >> b) & 1]
        o = [s for s in ss if (s >> b) & 1]
        if z and o:
            return decompose_states(z, n_bits) + decompose_states(o, n_bits)
    raise SolverError("状态集无法分解为子立方")


# ---------------------------------------------------------------------------
# 区域 / 证书
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """区域树节点。``cubes`` 是互斥子立方；``label_claim`` 只许 "0"/"1"/None。"""

    region_id: str
    parent_id: Optional[str]
    depth: int
    cubes: Tuple[Cube, ...]
    label_claim: Optional[str] = None
    proof_route: Optional[str] = None
    witness: Optional[dict] = None
    envelope_counts: Optional[dict] = None

    @property
    def is_terminal(self) -> bool:
        return self.label_claim is not None

    def state_count(self, n_bits: int) -> int:
        return region_count(self.cubes, n_bits)

    def to_dict(self, n_bits: int) -> dict:
        st = region_states(self.cubes, n_bits)
        out = {"region_id": self.region_id, "parent_id": self.parent_id,
               "depth": int(self.depth), "is_terminal": bool(self.is_terminal),
               "cubes": [c.to_dict() for c in self.cubes],
               "state_count": len(st)}
        if self.is_terminal:
            out["state_ids"] = st
        else:
            out["state_ids_head"] = st[:8]
        out["label_claim"] = self.label_claim
        out["proof_route"] = self.proof_route
        out["envelope_counts"] = self.envelope_counts
        out["witness"] = self.witness
        return out


@dataclass
class Certificate:
    certificate_id: str
    model_id: str
    table_id: str
    query_id: str
    n_bits: int
    allowed_state_ids: Tuple[int, ...]
    label_claim: str
    proof_route: str
    regions: Tuple[Region, ...]
    witness: dict
    budget: dict
    costs: dict
    stats: dict
    watermark: dict = field(default_factory=dict)
    verifier_version: str = VERIFIER_VERSION
    selection_provenance: str = SELECTION_PROVENANCE

    def to_dict(self) -> dict:
        return {"schema": SCHEMA_CERT, "certificate_id": self.certificate_id,
                "model_id": self.model_id, "table_id": self.table_id,
                "query_id": self.query_id, "n_bits": int(self.n_bits),
                "allowed_state_ids": [int(s) for s in self.allowed_state_ids],
                "label_claim": self.label_claim, "proof_route": self.proof_route,
                "regions": [r.to_dict(self.n_bits) for r in self.regions],
                "witness": self.witness, "verifier_version": self.verifier_version,
                "selection_provenance": self.selection_provenance,
                "budget": self.budget, "costs": self.costs, "stats": self.stats,
                "watermark": self.watermark}


@dataclass
class Unresolved:
    query_id: str
    reason: str
    detail: dict
    partition: Tuple[Region, ...]
    stats: dict

    def to_dict(self, n_bits: int) -> dict:
        return {"query_id": self.query_id, "status": "UNRESOLVED", "reason": self.reason,
                "detail": self.detail,
                "partition": [r.to_dict(n_bits) for r in self.partition],
                "stats": self.stats}


def _claim_of(label: str) -> Optional[str]:
    if label == ident.RegionLabel.ALL_CONNECTED.value:
        return "1"
    if label == ident.RegionLabel.ALL_DISCONNECTED.value:
        return "0"
    return None


def _state_path(struct: TableStructure, derivation: ident.StateDerivation) -> List:
    """见证路径：节点保留原始类型（A12 为字符串，B13 为盘索引整数），否则 reverify 判不在场。"""
    adj = _adjacency(derivation.edges)
    path = _bfs_path(adj, derivation.a_side & derivation.nodes, derivation.b_side & derivation.nodes)
    if path is None:
        return []
    nodes = [path[0][0]] if path else []
    for (_u, v, _t) in path:
        nodes.append(v)
    return nodes


def _witness_for(table: fs.FiniteTable, struct: TableStructure, qid: str, cubes: Sequence[Cube],
                 claim: str, n_bits: int) -> dict:
    if table.derivation is None:
        raise SolverError("表未提供三要素派生：不能出见证")
    s = min(c.min_state() for c in cubes)
    d = table.derivation(qid, int(s))
    if not ident.state_connected(d) == (claim == "1"):
        raise SolverError(f"见证状态 {s} 的派生标签与区域声明 {claim} 不符：仪器缺陷")
    w = {"state_id": int(s), "label": int(claim), "replay_ok": True}
    if claim == "1":
        w["path"] = _state_path(struct, d)
    return w


def certify(table: fs.FiniteTable, query_id: str, region_tree: Sequence[Region], *,
            budget: Optional[Mapping] = None, stats: Optional[Mapping] = None,
            certificate_id: Optional[str] = None) -> "Certificate | Unresolved":
    """给定区域树，逐区域重算外包并出证书；任何不合规一律降 UNRESOLVED。"""
    t0 = time.perf_counter()
    struct = structure_of(table)
    n_bits = table.n_bits
    root_cubes = resolve_query_cubes(table, query_id)
    if not root_cubes:
        return Unresolved(query_id, "MODEL_CONFLICT_EMPTY_ALLOWED",
                          {"note": "允许集为空：观测不在前向像内，无区域可建外包"}, (), dict(stats or {}))
    allowed = region_states(root_cubes, n_bits)
    allowed_set = set(allowed)
    by_id = {r.region_id: r for r in region_tree}
    roots = [r for r in region_tree if r.parent_id is None]
    if len(roots) != 1:
        return Unresolved(query_id, "TREE_ROOT_NOT_UNIQUE", {"n_roots": len(roots)}, (), dict(stats or {}))
    root = roots[0]
    if region_states(root.cubes, n_bits) != allowed:
        return Unresolved(query_id, "ROOT_NOT_ALLOWED_SET",
                          {"root_count": root.state_count(n_bits), "allowed": len(allowed)},
                          (), dict(stats or {}))
    children: Dict[str, List[Region]] = {}
    for r in region_tree:
        if r.parent_id is not None:
            children.setdefault(r.parent_id, []).append(r)
    # 树结构与覆盖（符号化）：终末与否只看"有没有子节点"，不看 label_claim
    for r in region_tree:
        kids = children.get(r.region_id, [])
        if kids:
            if r.label_claim is not None:
                return Unresolved(query_id, "TERMINAL_WITH_CHILDREN", {"region_id": r.region_id},
                                  (), dict(stats or {}))
            if any(len(k.cubes) != 1 for k in kids) or len(r.cubes) != 1:
                return Unresolved(query_id, "MULTI_CUBE_NODE_UNSUPPORTED",
                                  {"region_id": r.region_id}, (), dict(stats or {}))
            ok, why = cubes_partition(r.cubes[0], [k.cubes[0] for k in kids], n_bits)
            if not ok:
                return Unresolved(query_id, "CHILDREN_NOT_PARTITION",
                                  {"region_id": r.region_id, "why": why}, (), dict(stats or {}))
    # 叶区域：外包、标签、见证
    for r in region_tree:
        if children.get(r.region_id):
            continue
        if r.state_count(n_bits) <= 0:
            return Unresolved(query_id, "EMPTY_TERMINAL_REGION", {"region_id": r.region_id},
                              (), dict(stats or {}))
        if not set(region_states(r.cubes, n_bits)) <= allowed_set:
            return Unresolved(query_id, "REGION_OUTSIDE_ALLOWED", {"region_id": r.region_id},
                              (), dict(stats or {}))
        env = symbolic_envelope(struct, r.cubes, n_bits)
        lab = ident.region_label(env)
        claim = _claim_of(lab)
        if claim is None:
            return Unresolved(query_id, "MIXED_REGION_IN_TREE", {"region_id": r.region_id},
                              (), dict(stats or {}))
        r.label_claim = claim
        r.proof_route = (ident.ProofRoute.MUST_ENVELOPE.value if claim == "1"
                         else ident.ProofRoute.MAY_ENVELOPE_EXCLUSION.value)
        r.envelope_counts = env.element_counts()
        r.witness = _witness_for(table, struct, query_id, r.cubes, claim, n_bits)
    claims = {r.label_claim for r in region_tree if r.is_terminal}
    if len(claims) != 1:
        return Unresolved(query_id, "MIXED_LABELS",
                          {"labels": sorted(claims),
                           "terminal_regions": sum(1 for r in region_tree if r.is_terminal)},
                          tuple(r for r in region_tree if r.is_terminal), dict(stats or {}))
    claim = claims.pop()
    terminals = [r for r in region_tree if r.is_terminal]
    w = terminals[0].witness if terminals else None
    st = dict(stats or {})
    st["certify_seconds"] = round(time.perf_counter() - t0, 6)
    st["states_materialized_allowed"] = len(allowed)
    st["states_materialized_terminal"] = sum(r.state_count(n_bits) for r in terminals)
    st["materialization_note"] = "落件需把允许集与终末区域 state_ids 写全；这是簿记成本，不参与标签判定"
    return Certificate(
        certificate_id=certificate_id or f"S2C:{query_id}",
        model_id=table.table_id, table_id=table.table_id, query_id=query_id,
        n_bits=n_bits, allowed_state_ids=tuple(allowed), label_claim=str(claim),
        proof_route=(ident.ProofRoute.MUST_ENVELOPE.value if claim == "1"
                     else ident.ProofRoute.MAY_ENVELOPE_EXCLUSION.value),
        regions=tuple(region_tree), witness=dict(w or {}),
        budget=dict(budget or {}), costs={"basis": "MEASURED_ONLY", **st},
        stats=st, watermark={"research_line": "S2_N0_certificate_solver",
                             "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
                             "noise": "N0", "dictionary": "COMPLETE_KNOWN",
                             "claim_scope": "DECLARED_FINITE_TABLE"})


def solve_partial(table: fs.FiniteTable, query_id: str, budget: Optional[Mapping] = None
                  ) -> "Certificate | Unresolved":
    """部分赋值 + 区域细化搜索；不逐状态枚举（外包由部分赋值符号化推出）。"""
    t0 = time.perf_counter()
    struct = structure_of(table)
    n_bits = table.n_bits
    bud = {"max_nodes": 4096, "region_cap": 4096, "max_depth": n_bits}
    bud.update(dict(budget or {}))
    root_cubes = resolve_query_cubes(table, query_id)
    if not root_cubes:
        return Unresolved(query_id, "MODEL_CONFLICT_EMPTY_ALLOWED",
                          {"note": "允许集为空：观测不在前向像内"}, (), {"solve_seconds": 0.0})
    counter = {"nodes": 0, "envelope_calls": 0, "split_calls": 0, "witness_calls": 0}
    regions: List[Region] = []
    stack: List[Region] = [Region("R0", None, 0, tuple(root_cubes))]
    seq = 1
    while stack:
        r = stack.pop()
        counter["nodes"] += 1
        regions.append(r)
        if counter["nodes"] > int(bud["max_nodes"]):
            return Unresolved(query_id, "SPLIT_BUDGET_EXHAUSTED",
                              {"max_nodes": int(bud["max_nodes"]), "nodes": counter["nodes"]},
                              tuple(x for x in regions if x.is_terminal),
                              _stats(counter, t0))
        env = symbolic_envelope(struct, r.cubes, n_bits)
        counter["envelope_calls"] += 1
        claim = _claim_of(ident.region_label(env))
        if claim is not None:
            r.label_claim = claim
            continue
        if r.depth >= int(bud["max_depth"]):
            return Unresolved(query_id, "DEPTH_EXHAUSTED", {"depth": r.depth},
                              tuple(x for x in regions if x.is_terminal), _stats(counter, t0))
        if len(r.cubes) != 1:
            kids = [Region(f"R{seq + i}", r.region_id, r.depth + 1, (c,))
                    for i, c in enumerate(r.cubes)]
            seq += len(kids)
        else:
            b = choose_split_bit(struct, r.cubes, env, n_bits)
            counter["split_calls"] += 1
            m0 = r.cubes[0].fixed_mask | (1 << b)
            v0 = r.cubes[0].fixed_vals & ~(1 << b)
            kids = [Region(f"R{seq}", r.region_id, r.depth + 1, (Cube(m0, v0),)),
                    Region(f"R{seq + 1}", r.region_id, r.depth + 1, (Cube(m0, v0 | (1 << b)),))]
            seq += 2
        if len(regions) + len(stack) + len(kids) > int(bud["region_cap"]):
            return Unresolved(query_id, "REGION_CAP_EXHAUSTED",
                              {"region_cap": int(bud["region_cap"])},
                              tuple(x for x in regions if x.is_terminal), _stats(counter, t0))
        stack.extend(reversed(kids))
    counter["witness_calls"] = sum(1 for r in regions if r.is_terminal)
    return certify(table, query_id, regions, budget=bud, stats=_stats(counter, t0),
                   certificate_id=f"S2C:{query_id}")


def _stats(counter: Mapping, t0: float) -> dict:
    return {"nodes": int(counter["nodes"]), "envelope_calls": int(counter["envelope_calls"]),
            "split_calls": int(counter["split_calls"]),
            "witness_calls": int(counter["witness_calls"]),
            "states_touched_solve": 0,
            "solve_seconds": round(time.perf_counter() - t0, 6)}


# ---------------------------------------------------------------------------
# 独立验证器（不调用 solve_partial / certify，不读任何 certified 布尔）
# ---------------------------------------------------------------------------

def _v_connected(nodes: Iterable, edges: Iterable, side_a: Iterable, side_b: Iterable) -> bool:
    """验证器自带的可达性实现（独立于 identification._sides_connected）。"""
    sa = set(side_a) & set(nodes)
    sb = set(side_b) & set(nodes)
    if not sa or not sb:
        return False
    adj: Dict = {}
    for (u, v) in edges:
        if u not in nodes or v not in nodes:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen = set(sa)
    queue = list(sa)
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        if x in sb:
            return True
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return False


def _v_direction(env: ident.Envelope) -> str:
    """验证器自带的 must/may 方向规则（与 region_label 同一判据，另一份实现）。"""
    must = bool(env.must_attach_a and env.must_attach_b
                and _v_connected(env.must_nodes, env.must_edges, env.must_a_side, env.must_b_side))
    may = bool(env.may_attach_a and env.may_attach_b
               and _v_connected(env.may_nodes, env.may_edges, env.may_a_side, env.may_b_side))
    if must and not may:
        return "INSTRUMENT_DEFECT_MONOTONICITY"
    if must:
        return ident.RegionLabel.ALL_CONNECTED.value
    if not may:
        return ident.RegionLabel.ALL_DISCONNECTED.value
    return ident.RegionLabel.UNRESOLVED.value


def _v_allowed(table: fs.FiniteTable, query_id: str) -> Tuple[List[int], List[str]]:
    """验证器自带的允许集重算：不复用 ReadAction.codes，也不复用求解器解析器。"""
    reasons: List[str] = []
    meta = _queries_meta(table)
    kind = str(meta["kind"])
    n_bits = table.n_bits
    all_states = list(range(1 << n_bits))
    if kind == "A12_CLASS":
        prefix = "A12:class:"
        if not query_id.startswith(prefix):
            return [], [f"query_id 形态不符: {query_id!r}"]
        code = int(query_id[len(prefix):])
        out = []
        for s in all_states:
            acc = 0
            for a in table.baseline:
                radix = 1
                c = 0
                for g in a.groups:
                    c = c * (bin(int(g)).count("1") + 1) + bin(s & int(g)).count("1")
                radix = a.radix
                acc = acc * radix + c
            if acc == code:
                out.append(s)
        if not out:
            reasons.append(f"验证器重算的允许集为空 (code={code})")
        return out, reasons
    if kind == "B13_SCENARIO":
        prefix = "B13:scenario:"
        if not query_id.startswith(prefix):
            return [], [f"query_id 形态不符: {query_id!r}"]
        name = query_id[len(prefix):]
        sc = meta["scenarios"].get(name)
        if sc is None:
            return [], [f"未知情景 {name!r}"]
        if not sc.get("observation_in_forward_image", True):
            return [], []
        reads = [(int(b), int(v)) for (b, v) in sc["reads"]]
        out = [s for s in all_states if all(((s >> b) & 1) == v for (b, v) in reads)]
        return out, reasons
    return [], [f"未知查询类型 {kind!r}"]


def _v_envelope_from_states(struct: TableStructure, states: Sequence[int],
                            n_bits: int) -> ident.Envelope:
    """验证器自带的显式外包（纯 python 集合循环，不经 identification）。"""
    n = len(states)
    if n == 0:
        raise SolverError("空区域不得建外包")
    if struct.kind == "EDGE_BITS":
        must_edges = set()
        may_edges = set()
        for (b, u, v) in struct.edges:
            vals = [(s >> b) & 1 for s in states]
            if all(vals):
                must_edges.add((u, v))
            if any(vals):
                may_edges.add((u, v))
        nodes = set(struct.nodes)
        def att(bits):
            return [any((s >> b) & 1 for b in bits) for s in states]
        fa = att(struct.a_incident_bits)
        fb = att(struct.b_incident_bits)
        ma, ya = all(fa), any(fa)
        mb, yb = all(fb), any(fb)
        return ident.Envelope(n, frozenset(nodes), frozenset(nodes),
                              frozenset(must_edges), frozenset(may_edges),
                              frozenset(struct.a_nodes) if ma else frozenset(),
                              frozenset(struct.a_nodes) if ya else frozenset(),
                              frozenset(struct.b_nodes) if mb else frozenset(),
                              frozenset(struct.b_nodes) if yb else frozenset(),
                              ma, ya, mb, yb)
    base = len(struct.certain)
    n_unc = len(struct.uncertain)
    must_nodes = set(range(base))
    may_nodes = set(range(base))
    for j in range(n_unc):
        vals = [(s >> j) & 1 for s in states]
        if all(vals):
            must_nodes.add(base + j)
        if any(vals):
            may_nodes.add(base + j)
    must_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                           if i in must_nodes and j in must_nodes)
    may_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                          if i in may_nodes and j in may_nodes)
    must_a = frozenset(x for x in must_nodes if x in struct.attach_a_disks)
    may_a = frozenset(x for x in may_nodes if x in struct.attach_a_disks)
    must_b = frozenset(x for x in must_nodes if x in struct.attach_b_disks)
    may_b = frozenset(x for x in may_nodes if x in struct.attach_b_disks)
    return ident.Envelope(n, frozenset(must_nodes), frozenset(may_nodes),
                          must_edges, may_edges, must_a, may_a, must_b, may_b,
                          bool(must_a), bool(may_a), bool(must_b), bool(may_b))


def verify_certificate(table: fs.FiniteTable, query_id: str, certificate: Mapping) -> dict:
    """独立重放：允许集、树、外包、标签、见证、覆盖全部重算；不合规一律拒收。"""
    t0 = time.perf_counter()
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if not isinstance(certificate, Mapping):
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": ["证书不是映射"],
                "verifier_version": VERIFIER_VERSION, "verify_seconds": 0.0}
    cert = dict(certificate)
    unknown = [k for k in cert if k not in CERT_KEYS]
    if unknown:
        reasons.append(f"证书含未知顶层字段 {unknown}")
    bad_keys = [k for k in cert if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad_keys:
        reasons.append(f"证书含真值/神谕字段 {bad_keys}")
    prov = cert.get("selection_provenance")
    if prov != SELECTION_PROVENANCE:
        reasons.append(f"选点来源非法 {prov!r}（只许 {SELECTION_PROVENANCE}）")
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != table.n_bits:
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    struct = structure_of(table)
    n_bits = table.n_bits

    allowed, why = _v_allowed(table, query_id)
    checks["allowed_recomputed"] = len(allowed)
    if why:
        reasons.extend(why)
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if declared_allowed != sorted(allowed):
        reasons.append(f"允许集不符：声明 {len(declared_allowed)} / 重算 {len(allowed)}")
    allowed_set = set(allowed)

    regions = list(cert.get("regions", []))
    if not regions:
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons + ["证书无区域"],
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    ids = [r.get("region_id") for r in regions]
    if len(set(ids)) != len(ids):
        reasons.append("区域 id 重复")
    by_id = {r["region_id"]: r for r in regions}
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        reasons.append(f"根区域不唯一 ({len(roots)})")
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    root = roots[0]
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append(f"区域 {r.get('region_id')} 的父区域不存在")
            kids.setdefault(p, []).append(r)

    def states_of(r: Mapping) -> List[int]:
        out: set = set()
        for c in r.get("cubes", []):
            m, v = int(c["fixed_mask"]), int(c["fixed_vals"])
            if v & ~m:
                reasons.append(f"区域 {r.get('region_id')} 子立方非法")
                continue
            free = [b for b in range(n_bits) if not (m >> b) & 1]
            ss = [v]
            for b in free:
                ss = ss + [s | (1 << b) for s in ss]
            out.update(ss)
        return sorted(out)

    root_states = states_of(root)
    if root_states != sorted(allowed):
        reasons.append(f"根区域 != 允许集（{len(root_states)} vs {len(allowed)}）")

    # 树：内部节点子区域必须两两不交且并集 = 父
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = states_of(r)
        if r.get("is_terminal"):
            if ch:
                reasons.append(f"终末区域 {r.get('region_id')} 仍有子区域")
            if not st:
                reasons.append(f"终末区域 {r.get('region_id')} 为空")
            continue
        if not ch:
            reasons.append(f"内部区域 {r.get('region_id')} 无子区域")
            continue
        union: List[int] = []
        for c in ch:
            union.extend(states_of(c))
        if len(union) != len(set(union)):
            reasons.append(f"区域 {r.get('region_id')} 子区域重叠")
        if sorted(set(union)) != st:
            reasons.append(f"区域 {r.get('region_id')} 子区域并集 != 父区域")

    # 终末区域：三路外包 + 标签 + 见证 + 忠实性
    audit_states = 0
    terminal = [r for r in regions if r.get("is_terminal")]
    claims: set = set()
    for r in terminal:
        rid = r.get("region_id")
        st = states_of(r)
        claim = r.get("label_claim")
        if claim not in ("0", "1"):
            reasons.append(f"终末区域 {rid} 标签非法 {claim!r}")
            continue
        claims.add(claim)
        if not st:
            reasons.append(f"终末区域 {rid} 为空：空区域不得建外包、不得发证书")
            continue
        if not set(st) <= allowed_set:
            reasons.append(f"终末区域 {rid} 含允许集外状态")
        if table.derivation is None:
            reasons.append("表未提供派生：无法重算外包")
            continue
        derivs = [table.derivation(query_id, int(s)) for s in st]
        audit_states += len(st)
        env_s = ident.envelope_of(derivs)
        env_q = ident.envelope_of_quantifier(derivs)
        env_v = _v_envelope_from_states(struct, st, n_bits)
        if env_s.to_dict() != env_q.to_dict():
            reasons.append(f"区域 {rid} 集合路与逐元素路不等")
        if env_v.to_dict() != env_s.to_dict():
            reasons.append(f"区域 {rid} 验证器显式外包与 identification 集合路不等")
        declared = r.get("envelope_counts")
        if declared is not None and declared != env_s.element_counts():
            reasons.append(f"区域 {rid} 声明外包计数与重算不等")
        lab_v = _v_direction(env_v)
        lab_s = ident.region_label(env_s)
        if lab_v != lab_s:
            reasons.append(f"区域 {rid} 验证器方向规则与 region_label 不等 ({lab_v} vs {lab_s})")
        if lab_v == "INSTRUMENT_DEFECT_MONOTONICITY":
            reasons.append(f"区域 {rid} 外包单调性破裂")
            continue
        want = ident.RegionLabel.ALL_CONNECTED.value if claim == "1" else ident.RegionLabel.ALL_DISCONNECTED.value
        if lab_v != want:
            reasons.append(f"区域 {rid} 声明 {claim} 但重算 {lab_v}")
        # 忠实性 + 显式标签（逐状态审计）
        for d in derivs:
            if int(d.label) != int(1 if ident.state_connected(d) else 0):
                reasons.append(f"区域 {rid} 状态 {d.state_id} 派生与标签不忠实")
                break
            if int(d.label) != int(claim):
                reasons.append(f"区域 {rid} 状态 {d.state_id} 标签 {d.label} != 声明 {claim}")
                break
        w = r.get("witness")
        if not w or "state_id" not in w:
            reasons.append(f"区域 {rid} 无见证")
        else:
            sid = int(w["state_id"])
            if sid not in set(st):
                reasons.append(f"区域 {rid} 见证 {sid} 不在区域内")
            elif int(w.get("label", -1)) != int(claim):
                reasons.append(f"区域 {rid} 见证标签与区域标签不符")
            else:
                rep = ident.reverify_witness(table.derivation(query_id, sid), int(claim),
                                             w.get("path"))
                if not rep["ok"]:
                    reasons.append(f"区域 {rid} 见证不可重放 {rep.get('reason')}")

    # 覆盖：终末区域两两不交、并集 = 允许集
    all_states: List[int] = []
    for r in terminal:
        all_states.extend(states_of(r))
    if len(all_states) != len(set(all_states)):
        reasons.append("终末区域划分不互斥")
    if sorted(set(all_states)) != sorted(allowed):
        missing = sorted(set(allowed) - set(all_states))
        extra = sorted(set(all_states) - set(allowed))
        reasons.append(f"终末区域覆盖 != 允许集 (缺 {len(missing)} / 多 {len(extra)})")
    if len(claims) > 1:
        reasons.append(f"终末区域标签不一致 {sorted(claims)}：混合查询不得发单标签证书")
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    elif int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
        reasons.append("证书顶层见证标签与 label_claim 不符")
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION,
            "checks": {"allowed_recomputed": len(allowed),
                       "regions": len(regions), "terminal_regions": len(terminal),
                       "audit_states": audit_states,
                       "claims": sorted(claims)},
            "verify_seconds": round(time.perf_counter() - t0, 6)}


# ---------------------------------------------------------------------------
# 适配器：A12（复用 finite_scope）+ B13（新建）
# ---------------------------------------------------------------------------

def _load(p: Path) -> dict:
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def a12_table_with_derivation(root: Optional[Path] = None) -> Tuple[fs.FiniteTable, dict]:
    """复用 ``finite_scope.a12_table()``，补三要素派生与 S2 结构登记（不改原模块）。"""
    base = Path(root) if root else REPO_ROOT
    table, meta = fs.a12_table(base)
    cc = base / "results" / "connectivity_certification"
    ms = _load(cc / "model_spec.json")["a_layer"]
    # 端点规范化 (min,max)：与 identification._canon 同口径，否则三路外包逐要素不可比
    edges = [(str(min(e[1], e[2])), str(max(e[1], e[2]))) for e in ms["edges"]]
    pos = {e[0]: i for i, e in enumerate(ms["edges"])}
    nodes = tuple(str(x) for x in ms["nodes"])
    inc_a = tuple(i for i, (u, v) in enumerate(edges) if "A" in (u, v))
    inc_b = tuple(i for i, (u, v) in enumerate(edges) if "B" in (u, v))
    struct = TableStructure(
        kind="EDGE_BITS", n_bits=table.n_bits, nodes=nodes,
        edges=tuple((pos[ms["edges"][i][0]], u, v) for i, (u, v) in enumerate(edges)),
        a_nodes=("A",), b_nodes=("B",),
        a_incident_bits=inc_a, b_incident_bits=inc_b)
    labels = np.asarray(table.labels["A12"], dtype=np.int64)
    attach_a = table.attach["A12"][0]
    attach_b = table.attach["A12"][1]

    def deriv(qid: str, s: int) -> ident.StateDerivation:
        s = int(s)
        es = frozenset(edges[i] for i in range(len(edges)) if (s >> i) & 1)
        return ident.StateDerivation(
            state_id=s, nodes=frozenset(nodes), edges=es,
            a_side=frozenset({"A"}) if bool(attach_a[s]) else frozenset(),
            b_side=frozenset({"B"}) if bool(attach_b[s]) else frozenset(),
            attach_a=bool(attach_a[s]), attach_b=bool(attach_b[s]), label=int(labels[s]))

    table.derivation = deriv
    table.tags = dict(table.tags)
    table.tags["s2_structure"] = struct.to_dict()
    table.tags["s2_queries"] = {"kind": "A12_CLASS"}
    return table, {"structure": struct, "n_classes": 128}


def b13_table(root: Optional[Path] = None) -> Tuple[fs.FiniteTable, dict]:
    """从 model_spec + state_table + scenarios 构造 B13 有限表（新建适配器）。"""
    from fractureflow.connectivity_qoi import disk_point_distance
    from fractureflow.intersect_disk import candidate_pairs, disk_pair_intersect

    base = Path(root) if root else REPO_ROOT
    cc = base / "results" / "connectivity_certification"
    spec = _load(cc / "model_spec.json")["b_layer"]
    st = _load(cc / "finite_reference" / "b_layer" / "state_table.json")
    scen = _load(cc / "finite_reference" / "b_layer" / "scenarios.json")["scenarios"]
    certain = [str(d["id"]) for d in spec["certain_disks"]]
    uncertain = [str(x) for x in st["uncertain_ids"]]
    if uncertain != [str(d["id"]) for d in spec["uncertain_disks"]]:
        raise SolverError("uncertain_ids 顺序与 model_spec.uncertain_disks 不一致")
    disks = list(spec["certain_disks"]) + list(spec["uncertain_disks"])
    C = np.asarray([d["center"] for d in disks], dtype=float)
    N = np.tile(np.array([0.0, 0.0, 1.0]), (len(disks), 1))
    R = np.asarray([d["radius"] for d in disks], dtype=float)
    A = np.asarray(spec["endpoints"]["A"], dtype=float)
    B = np.asarray(spec["endpoints"]["B"], dtype=float)
    ra = float(spec["r_attach"])
    pairs = candidate_pairs(C, R)
    hit = disk_pair_intersect(C, N, R, pairs[:, 0], pairs[:, 1], scale=float(np.median(R)))
    contact = tuple((int(i), int(j)) for (i, j) in pairs[hit].tolist())
    aatt = disk_point_distance(C, N, R, A)[0] <= ra
    batt = disk_point_distance(C, N, R, B)[0] <= ra
    attach_a_disks = tuple(int(i) for i in np.flatnonzero(aatt).tolist())
    attach_b_disks = tuple(int(i) for i in np.flatnonzero(batt).tolist())
    labels = np.asarray(st["c_labels"], dtype=np.int64)
    attach_a = np.asarray(st["attach_a"], dtype=bool)
    attach_b = np.asarray(st["attach_b"], dtype=bool)
    base_i = len(certain)
    n_unc = len(uncertain)
    bit_of_disk = {base_i + j: j for j in range(n_unc)}

    def deriv(qid: str, s: int) -> ident.StateDerivation:
        s = int(s)
        present = set(range(base_i))
        for j in range(n_unc):
            if (s >> j) & 1:
                present.add(base_i + j)
        es = frozenset((i, j) for (i, j) in contact if i in present and j in present)
        return ident.StateDerivation(
            state_id=s, nodes=frozenset(present), edges=es,
            a_side=frozenset(x for x in present if x in attach_a_disks),
            b_side=frozenset(x for x in present if x in attach_b_disks),
            attach_a=bool(attach_a[s]), attach_b=bool(attach_b[s]), label=int(labels[s]))

    struct = TableStructure(
        kind="DISK_BITS", n_bits=n_unc, certain=tuple(certain), uncertain=tuple(uncertain),
        contact_edges=contact, attach_a_disks=attach_a_disks, attach_b_disks=attach_b_disks)

    def _ra(wid: str, bit: int, cost_m: float) -> fs.ReadAction:
        cost_int = int(round(float(cost_m) * 10))
        if abs(float(cost_m) * 10 - cost_int) > 1e-9:
            raise SolverError(f"{wid}: cost_m={cost_m} 不是 0.1 m 的整数倍")
        return fs.ReadAction(wid, (1 << int(bit),), cost_int)

    cross_of = {str(w["id"]): str(w["crosses"][0]) for w in spec["wells"]["baseline"] + spec["wells"]["candidates"]}
    cost_of = {str(w["id"]): float(w["cost_m"]) for w in spec["wells"]["baseline"] + spec["wells"]["candidates"]}
    bit_of_name = {name: bit_of_disk[base_i + j] for j, name in enumerate(uncertain)}
    baseline = tuple(_ra(str(w["id"]), bit_of_name[cross_of[str(w["id"])]], cost_of[str(w["id"])])
                     for w in spec["wells"]["baseline"])
    actions = tuple(_ra(str(w["id"]), bit_of_name[cross_of[str(w["id"])]], cost_of[str(w["id"])])
                    for w in spec["wells"]["candidates"])

    sc_meta: Dict[str, dict] = {}
    for name, s in scen.items():
        reads = []
        ok = bool(s.get("status") != "MODEL_CONFLICT")
        for wid in s["wells"]:
            disk = cross_of[str(wid)]
            reads.append([bit_of_name[disk], int(s["true_state"][disk])])
        sc_meta[name] = {"wells": [str(w) for w in s["wells"]], "reads": reads,
                         "observation_in_forward_image": ok,
                         "declared_consistent_count": int(s["consistent_count"]),
                         "declared_target_set": list(s["target_set"]),
                         "status": s.get("status")}

    table = fs.FiniteTable(
        table_id=spec["model_id"], n_bits=n_unc,
        labels={"B13": labels},
        baseline=baseline, actions=actions,
        attach={"B13": (attach_a, attach_b)},
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N0",
              "model": "B13 声明有限模型", "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "B13_SCENARIO", "scenarios": sc_meta}})
    table.derivation = deriv
    meta = {"structure": struct, "contact_edges": len(contact), "certain": certain,
            "uncertain": uncertain, "attach_a_disks": attach_a_disks,
            "attach_b_disks": attach_b_disks, "scenarios": sc_meta,
            "baseline": [a.to_dict() for a in baseline], "actions": [a.to_dict() for a in actions]}
    return table, meta


# ---------------------------------------------------------------------------
# B 层探针：声明几何族上的箱 × 在场位 分支细化（must/may 用区间算术）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxFamily:
    """声明几何族：共面盘，盘心 x 在声明区间内，y/z 与半径固定；可选在场位自由。

    ``free_presence`` 是**可能不在场**的盘索引（对应离散位）；不在其中者恒在场。
    """

    family_id: str
    disks: Tuple[Tuple[str, float, float, float, float, float], ...]  # id, x_lo, x_hi, y, z, r
    A: Tuple[float, float, float]
    B: Tuple[float, float, float]
    r_attach: float
    free_presence: Tuple[int, ...] = ()
    note: str = ""

    def to_dict(self) -> dict:
        return {"family_id": self.family_id,
                "disks": [{"id": d[0], "x_lo": d[1], "x_hi": d[2], "y": d[3], "z": d[4],
                           "r": d[5]} for d in self.disks],
                "A": list(self.A), "B": list(self.B), "r_attach": self.r_attach,
                "free_presence": [int(x) for x in self.free_presence], "note": self.note}

    def box(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((d[1], d[2]) for d in self.disks)


def _fr(x) -> Fraction:
    """十进制精确有理数：声明族的端点按字面十进制取值，避免浮点把 1.8 判成 1.7999999999999998。"""
    return Fraction(str(float(x)))


def _geom_range(lo, hi, dy, dz) -> Tuple[Fraction, Fraction]:
    """区间算术（平方距离，精确有理数）：返回 (dmin^2, dmax^2)。"""
    lo, hi, dy, dz = _fr(lo), _fr(hi), _fr(dy), _fr(dz)
    base = dy * dy + dz * dz
    if lo <= 0 <= hi:
        dmin2 = base
    else:
        dmin2 = min(abs(lo), abs(hi)) ** 2 + base
    dmax2 = max(abs(lo), abs(hi)) ** 2 + base
    return dmin2, dmax2


def _box_geometry(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                  fixed: Mapping[int, int]) -> dict:
    """区间算术外包：fixed[盘索引] = 0/1 表示该盘必不在场/必在场；未出现者自由。"""
    n = len(fam.disks)
    must_present = {i for i in range(n) if i not in fam.free_presence}
    may_present = set(range(n))
    for i, v in fixed.items():
        if v == 1:
            must_present.add(i)
        else:
            may_present.discard(i)
    must_e, may_e = set(), set()
    for i in range(n):
        for j in range(i + 1, n):
            dmin2, dmax2 = _geom_range(_fr(xs[i][0]) - _fr(xs[j][1]),
                                       _fr(xs[i][1]) - _fr(xs[j][0]),
                                       _fr(fam.disks[i][3]) - _fr(fam.disks[j][3]),
                                       _fr(fam.disks[i][4]) - _fr(fam.disks[j][4]))
            thr = _fr(fam.disks[i][5]) + _fr(fam.disks[j][5])
            thr2 = thr * thr
            if dmax2 < thr2 and i in must_present and j in must_present:
                must_e.add((i, j))
            if dmin2 < thr2 and i in may_present and j in may_present:
                may_e.add((i, j))
    ma, ya, mb, yb = set(), set(), set(), set()
    for i in range(n):
        thr = _fr(fam.disks[i][5]) + _fr(fam.r_attach)
        thr2 = thr * thr
        for P, mm, yy in ((fam.A, ma, ya), (fam.B, mb, yb)):
            dmin2, dmax2 = _geom_range(_fr(P[0]) - _fr(xs[i][1]), _fr(P[0]) - _fr(xs[i][0]),
                                       _fr(P[1]) - _fr(fam.disks[i][3]),
                                       _fr(P[2]) - _fr(fam.disks[i][4]))
            if dmax2 <= thr2 and i in must_present:
                mm.add(i)
            if dmin2 <= thr2 and i in may_present:
                yy.add(i)
    return {"must_nodes": frozenset(must_present), "may_nodes": frozenset(may_present),
            "must_edges": frozenset(must_e), "may_edges": frozenset(may_e),
            "must_a": frozenset(ma), "may_a": frozenset(ya),
            "must_b": frozenset(mb), "may_b": frozenset(yb)}


def box_envelope(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                 fixed: Optional[Mapping[int, int]] = None) -> ident.Envelope:
    g = _box_geometry(fam, xs, dict(fixed or {}))
    return ident.Envelope(0, g["must_nodes"], g["may_nodes"], g["must_edges"], g["may_edges"],
                          g["must_a"], g["may_a"], g["must_b"], g["may_b"],
                          bool(g["must_a"]), bool(g["may_a"]), bool(g["must_b"]), bool(g["may_b"]))


def _region_measure(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                    fixed: Mapping[int, int]) -> float:
    vol = 1.0
    for (lo, hi) in xs:
        vol *= max(0.0, hi - lo)
    free = len([i for i in fam.free_presence if i not in fixed])
    return vol * float(1 << free)


def refine_family(fam: BoxFamily, max_nodes: int = 512) -> dict:
    """分支细化：先按在场位分裂（离散），再按最宽 x 区间二分（连续）。

    预算到即停（不再弹栈），未展开的箱子计入 ``frontier_boxes``；``terminated`` 只在
    栈自然清空时为真。跨阈值的连续族在纯箱式细化下不会自然终止——这是要如实报的结果。
    """
    xs0 = fam.box()
    stack: List[Tuple[Tuple[Tuple[float, float], ...], Dict[int, int], int]] = [(xs0, {}, 0)]
    terminal: List[dict] = []
    nodes = 0
    budget_hit = False
    degenerate = 0
    while stack:
        if nodes >= int(max_nodes):
            budget_hit = True
            break
        xs, fixed, depth = stack.pop()
        nodes += 1
        env = box_envelope(fam, xs, fixed)
        lab = ident.region_label(env)
        if lab in (ident.RegionLabel.ALL_CONNECTED.value, ident.RegionLabel.ALL_DISCONNECTED.value):
            terminal.append({"xs": [list(x) for x in xs], "fixed": dict(fixed),
                             "label": "1" if lab == ident.RegionLabel.ALL_CONNECTED.value else "0",
                             "depth": depth, "must_edges": sorted(env.must_edges),
                             "may_edges": sorted(env.may_edges),
                             "must_a": sorted(env.must_a_side), "may_a": sorted(env.may_a_side),
                             "must_b": sorted(env.must_b_side), "may_b": sorted(env.may_b_side)})
            continue
        free_bits = [i for i in fam.free_presence if i not in fixed]
        if free_bits:
            b = min(free_bits)
            for v in (0, 1):
                f2 = dict(fixed)
                f2[b] = v
                stack.append((xs, f2, depth + 1))
            continue
        widths = [x[1] - x[0] for x in xs]
        k = int(np.argmax(widths))
        if widths[k] <= 0.0:
            degenerate += 1
            continue
        mid = 0.5 * (xs[k][0] + xs[k][1])
        left = list(xs)
        right = list(xs)
        left[k] = (xs[k][0], mid)
        right[k] = (mid, xs[k][1])
        stack.append((tuple(right), dict(fixed), depth + 1))
        stack.append((tuple(left), dict(fixed), depth + 1))
    total = sum(_region_measure(fam, [tuple(x) for x in t["xs"]], t["fixed"]) for t in terminal)
    full = _region_measure(fam, fam.box(), {})
    return {"family_id": fam.family_id, "nodes": nodes, "terminal_regions": len(terminal),
            "frontier_boxes": len(stack), "budget_hit": budget_hit,
            "degenerate_boxes": degenerate,
            "terminated": (not budget_hit) and (not stack) and degenerate == 0,
            "labels": sorted({t["label"] for t in terminal}),
            "measure_terminal": total, "measure_family": full,
            "measure_closed": abs(total - full) <= 1e-9 * max(1.0, full),
            "terminal": terminal}


def _v_box_geometry(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                    fixed: Mapping[int, int]) -> dict:
    """验证器自带的区间算术（另一份实现，不调 _box_geometry）；同样用精确有理数。"""
    n = len(fam.disks)
    must_present, may_present = set(), set()
    for i in range(n):
        if i not in fam.free_presence:
            must_present.add(i)
            may_present.add(i)
        else:
            v = fixed.get(i, None)
            if v == 1:
                must_present.add(i)
            if v != 0:
                may_present.add(i)
    me, ye, ma, ya, mb, yb = [], [], [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            xi = (_fr(xs[i][0]), _fr(xs[i][1]))
            xj = (_fr(xs[j][0]), _fr(xs[j][1]))
            lo = xi[0] - xj[1]
            hi = xi[1] - xj[0]
            dy = _fr(fam.disks[i][3]) - _fr(fam.disks[j][3])
            dz = _fr(fam.disks[i][4]) - _fr(fam.disks[j][4])
            base = dy * dy + dz * dz
            dmin2 = base if (lo <= 0 <= hi) else min(abs(lo), abs(hi)) ** 2 + base
            dmax2 = max(abs(lo), abs(hi)) ** 2 + base
            thr = _fr(fam.disks[i][5]) + _fr(fam.disks[j][5])
            if dmax2 < thr * thr and i in must_present and j in must_present:
                me.append((i, j))
            if dmin2 < thr * thr and i in may_present and j in may_present:
                ye.append((i, j))
    for i in range(n):
        thr = _fr(fam.disks[i][5]) + _fr(fam.r_attach)
        thr2 = thr * thr
        for P, mm, yy in ((fam.A, ma, ya), (fam.B, mb, yb)):
            lo = _fr(P[0]) - _fr(xs[i][1])
            hi = _fr(P[0]) - _fr(xs[i][0])
            base = (_fr(P[1]) - _fr(fam.disks[i][3])) ** 2 + (_fr(P[2]) - _fr(fam.disks[i][4])) ** 2
            dmin2 = base if (lo <= 0 <= hi) else min(abs(lo), abs(hi)) ** 2 + base
            dmax2 = max(abs(lo), abs(hi)) ** 2 + base
            if dmax2 <= thr2 and i in must_present:
                mm.append(i)
            if dmin2 <= thr2 and i in may_present:
                yy.append(i)
    return {"must_nodes": frozenset(must_present), "may_nodes": frozenset(may_present),
            "must_edges": frozenset(me), "may_edges": frozenset(ye),
            "must_a": frozenset(ma), "may_a": frozenset(ya),
            "must_b": frozenset(mb), "may_b": frozenset(yb)}


# ---------------------------------------------------------------------------
# G64 · P2-C 连续覆盖门：自由在场位抽样修复 + 逐项覆盖验证
# ---------------------------------------------------------------------------

BOX_VERIFY_SCHEMA = "s2.box_verify/v2"
BOX_VERIFY_WATERMARK = "P2C_CONTINUOUS_COVER_AUDIT / DIAGNOSTIC_ONLY / NO_CERTIFIED_CLAIMS"
BOX_DEFAULT_TOL = 1e-9
BOX_MAX_PRESENCE_PATTERNS = 4096
BOX_MAX_BRANCH_COMBOS = 1024

#: 区域边界与阈值归属的显式约定（任务书 §3.4-4）。改约定必须同时改回归夹具。
BOX_BOUNDARY_CONVENTION = {
    "x_interval": "HALF_OPEN_[lo,hi)",
    "contact": "STRICT_d_lt_r_i_plus_r_j",
    "attach": "NON_STRICT_dist_minus_r_le_r_attach",
    "coverage": "HALF_OPEN_TOUCH_IS_DISJOINT",
    "measure": "LEBESGUE_x_COUNTING_ON_PRESENCE_BITS",
}

#: 原因码 -> 中文说明（报告映射表由本表重算，不手打）
BOX_REASON_TEXT = {
    "EMPTY_CERTIFICATE": "箱证书无区域",
    "COVERAGE_NOT_DECLARED": "证书未声明覆盖完备（coverage_declared != true），不得当全域已证",
    "FRONTIER_UNRESOLVED": "存在未展开 frontier / 未终止 / 预算耗尽（SEARCH_UNRESOLVED）",
    "FAMILY_BOX_DEGENERATE": "声明族 x 盒本身退化或零测度",
    "REGION_SHAPE_MISMATCH": "区域 x 区间维数与声明族不符",
    "DEGENERATE_INTERVAL": "区域存在 hi <= lo 的退化区间",
    "ZERO_MEASURE_REGION": "区域乘积测度为 0",
    "REGION_OUT_OF_BOX": "区域 x 区间越出声明族",
    "REGION_FIXED_INVALID_BIT": "区域固定了非自由在场位",
    "REGION_FIXED_VALUE_INVALID": "区域固定位的值不是 0/1",
    "REGION_LABEL_CLAIM_INVALID": "区域 label_claim 不是 '0'/'1'",
    "LABEL_MISMATCH": "区间外包重算标签与声明不符",
    "UNRESOLVED_REGION": "区域未被区间外包判明（UNRESOLVED / 单调性缺陷）",
    "PRODUCT_BRANCH_LABEL_DIFFERS": "未固定自由位的分支标签与声明不一致",
    "PRODUCT_BRANCH_MISSING": "某在场位组合没有任何区域覆盖（乘积空间缺分支）",
    "PRODUCT_SPACE_LIMIT": "在场位/未固定位组合数超枚举上限，未核实",
    "COVER_CHECK_LIMIT": "并集覆盖判定预算耗尽，未核实",
    "REGIONS_OVERLAP": "两区域在（离散在场位 × 连续坐标）乘积空间重叠",
    "UNION_NOT_COVERING": "某在场位组合下区域 x 盒并集不覆盖族盒",
    "MEASURE_MISMATCH": "区域测度和 != 族测度",
    "TOP_LABEL_MISMATCH": "顶层标签与区域标签不一致",
    "NUMERIC_BOUNDARY": "存在余量 < tol 的几何判据对象（NUMERIC_BOUNDARY）",
}


def _bv_exact_measure(box: Sequence[Tuple[Fraction, Fraction]], n_free: int) -> Fraction:
    """乘积空间精确测度 = x 盒 Lebesgue 测度 × 2^(未固定自由位数)。"""
    m = Fraction(1)
    for lo, hi in box:
        if hi <= lo:
            return Fraction(0)
        m *= (hi - lo)
    return m * (1 << max(0, int(n_free)))


def _bv_boxes_disjoint(a: Sequence[Tuple[Fraction, Fraction]],
                       b: Sequence[Tuple[Fraction, Fraction]]) -> bool:
    """半开区间约定：存在一维满足 a.hi <= b.lo 或 b.hi <= a.lo 即不相交。"""
    for k in range(len(a)):
        if a[k][1] <= b[k][0] or b[k][1] <= a[k][0]:
            return True
    return False


def _bv_pattern_compatible(fixed: Mapping[int, int], pattern: Mapping[int, int]) -> bool:
    """区域的固定值与该在场位组合一致（未固定位对任何组合都兼容）。"""
    for b, v in fixed.items():
        if int(pattern.get(int(b), -1)) != int(v):
            return False
    return True


def _bv_presence_conflict(fa: Mapping[int, int], fb: Mapping[int, int]) -> bool:
    """两区域在同一位上给出不同固定值 => 在乘积空间必然不相交。"""
    for b, v in fa.items():
        if int(b) in fb and int(fb[int(b)]) != int(v):
            return True
    return False


def _bv_rects_cover(box, rects, dims=None, budget=None) -> bool:
    """精确有理数判定：矩形集是否覆盖 ``box``（逐维剖分；不比较测度和）。"""
    if budget is None:
        budget = {"left": 400000, "exhausted": False}
    budget["left"] -= 1
    if budget["left"] < 0:
        budget["exhausted"] = True
        return False
    if any(lo >= hi for lo, hi in box):
        return True
    for r in rects:
        if all(r[k][0] <= box[k][0] and r[k][1] >= box[k][1] for k in range(len(box))):
            return True
    dims = tuple(range(len(box))) if dims is None else tuple(dims)
    if not dims:
        return False
    best = None
    for k in dims:
        lo, hi = box[k]
        cuts = {lo, hi}
        for r in rects:
            a, b = r[k]
            if lo < a < hi:
                cuts.add(a)
            if lo < b < hi:
                cuts.add(b)
        if len(cuts) > 2 and (best is None or len(cuts) > best[0]):
            best = (len(cuts), k, sorted(cuts))
    if best is None:
        return False
    _, k, cuts = best
    rest = tuple(d for d in dims if d != k)
    for a, b in zip(cuts, cuts[1:]):
        sub = list(box)
        sub[k] = (a, b)
        sub_rects = [r for r in rects if r[k][0] <= a and r[k][1] >= b]
        if not sub_rects:
            return False
        if not _bv_rects_cover(tuple(sub), sub_rects, rest, budget):
            return False
    return True


def _bv_margins(fam: BoxFamily, xs: Sequence[Tuple[float, float]]) -> List[dict]:
    """区域每个几何判据的精确平方距离区间与阈值（与 _v_box_geometry 同一公式，另一用途）。"""
    n = len(fam.disks)
    out: List[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            lo = _fr(xs[i][0]) - _fr(xs[j][1])
            hi = _fr(xs[i][1]) - _fr(xs[j][0])
            dy = _fr(fam.disks[i][3]) - _fr(fam.disks[j][3])
            dz = _fr(fam.disks[i][4]) - _fr(fam.disks[j][4])
            base = dy * dy + dz * dz
            dmin2 = base if (lo <= 0 <= hi) else min(abs(lo), abs(hi)) ** 2 + base
            dmax2 = max(abs(lo), abs(hi)) ** 2 + base
            out.append({"kind": "CONTACT", "i": i, "j": j,
                        "thr": _fr(fam.disks[i][5]) + _fr(fam.disks[j][5]),
                        "dmin2": dmin2, "dmax2": dmax2})
    for i in range(n):
        thr = _fr(fam.disks[i][5]) + _fr(fam.r_attach)
        for P, tag in ((fam.A, "A"), (fam.B, "B")):
            lo = _fr(P[0]) - _fr(xs[i][1])
            hi = _fr(P[0]) - _fr(xs[i][0])
            base = (_fr(P[1]) - _fr(fam.disks[i][3])) ** 2 + (_fr(P[2]) - _fr(fam.disks[i][4])) ** 2
            dmin2 = base if (lo <= 0 <= hi) else min(abs(lo), abs(hi)) ** 2 + base
            dmax2 = max(abs(lo), abs(hi)) ** 2 + base
            out.append({"kind": "ATTACH_" + tag, "i": i, "j": None,
                        "thr": thr, "dmin2": dmin2, "dmax2": dmax2})
    return out


def _bv_boundary_flags(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                       tol: float) -> List[dict]:
    """余量 < tol 的判据对象列 NUMERIC_BOUNDARY（余量按距离单位，平方量开方后相减）。"""
    out: List[dict] = []
    for e in _bv_margins(fam, xs):
        thr = float(e["thr"])
        m = min(abs(math.sqrt(float(e["dmin2"])) - thr),
                abs(math.sqrt(float(e["dmax2"])) - thr))
        if m < float(tol):
            out.append({"kind": e["kind"], "i": int(e["i"]), "j": e["j"],
                        "margin": float(m), "tol": float(tol),
                        "strictness": ("STRICT_LT" if e["kind"] == "CONTACT"
                                       else "NON_STRICT_LE"),
                        "dmin2_sq": float(e["dmin2"]), "dmax2_sq": float(e["dmax2"]),
                        "thr": thr})
    return out


def _bv_branch_labels(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                      fixed: Mapping[int, int], bits: Sequence[int],
                      max_combos: int) -> dict:
    """未固定的自由位逐一固定成 0/1，分别重算标签（乘积空间的直接证明，不靠推断）。"""
    bits = [int(b) for b in bits]
    if not bits:
        return {"ok": True, "n_combos": 0, "combos": []}
    n = 1 << len(bits)
    if n > int(max_combos):
        return {"ok": False, "n_combos": int(n), "combos": [], "reason": "PRODUCT_SPACE_LIMIT"}
    combos = []
    for combo in itertools.product((0, 1), repeat=len(bits)):
        f2 = dict(fixed)
        for b, v in zip(bits, combo):
            f2[int(b)] = int(v)
        g = _v_box_geometry(fam, xs, f2)
        env = ident.Envelope(0, g["must_nodes"], g["may_nodes"], g["must_edges"], g["may_edges"],
                             g["must_a"], g["may_a"], g["must_b"], g["may_b"],
                             bool(g["must_a"]), bool(g["may_a"]),
                             bool(g["must_b"]), bool(g["may_b"]))
        combos.append({"assignment": {int(b): int(v) for b, v in zip(bits, combo)},
                       "label": _v_direction(env)})
    return {"ok": True, "n_combos": int(n), "combos": combos}


def sample_region(fam: BoxFamily, xs: Sequence[Tuple[float, float]],
                  fixed: Mapping[int, int], n_samples: int = 200, seed: int = 20261018) -> dict:
    """区域内均匀采样，用精确距离判据重算标签（抽样审计，不是证明）。

    G64 修复：修复前用 ``fixed.get(i, 1)`` 先把自由位当在场，随后
    ``present.add(b) if rng.integers(0, 2) == 1 else None`` 只加不删，缺席分支概率为 0。
    现在每个自由位逐样本做真 Bernoulli(0/1)；``fixed`` 中显式的 0/1 被尊重；
    非自由位恒在场。返回值保留旧字段，另加分支计数（``branch_counts`` 等）。
    """
    rng = np.random.default_rng(seed)
    n = len(fam.disks)
    fixed_map = {int(k): int(v) for k, v in dict(fixed or {}).items()}
    for k, v in fixed_map.items():
        if v not in (0, 1):
            raise SolverError(f"fixed[{k}]={v} 不是 0/1")
        if k < 0 or k >= n:
            raise SolverError(f"fixed 键 {k} 越界（盘数 {n}）")
    free = [int(i) for i in fam.free_presence if int(i) not in fixed_map]
    labels: List[int] = []
    absent_counts = {b: 0 for b in free}
    present_counts = {b: 0 for b in free}
    hist: Dict[str, int] = {}
    for _ in range(int(n_samples)):
        pts = {i: rng.uniform(xs[i][0], xs[i][1]) for i in range(n)}
        present = set()
        for i in range(n):
            if i in fam.free_presence:
                if i in fixed_map:
                    if fixed_map[i] == 1:
                        present.add(i)
                elif int(rng.integers(0, 2)) == 1:
                    present.add(i)
            else:
                present.add(i)
        for b in free:
            if b in present:
                present_counts[b] += 1
            else:
                absent_counts[b] += 1
        key = ",".join(f"{b}={int(b in present)}" for b in free)
        hist[key] = hist.get(key, 0) + 1
        edges = set()
        for i in present:
            for j in present:
                if i >= j:
                    continue
                d = math.dist((pts[i], fam.disks[i][3], fam.disks[i][4]),
                              (pts[j], fam.disks[j][3], fam.disks[j][4]))
                if d < fam.disks[i][5] + fam.disks[j][5]:
                    edges.add((i, j))
        a_side = [i for i in present
                  if math.dist((pts[i], fam.disks[i][3], fam.disks[i][4]), fam.A)
                  - fam.disks[i][5] <= fam.r_attach]
        b_side = [i for i in present
                  if math.dist((pts[i], fam.disks[i][3], fam.disks[i][4]), fam.B)
                  - fam.disks[i][5] <= fam.r_attach]
        labels.append(1 if (a_side and b_side and _v_connected(present, edges, a_side, b_side)) else 0)
    return {"n_samples": int(n_samples), "n_connected": int(sum(labels)),
            "n_disconnected": int(len(labels) - sum(labels)), "labels_seen": sorted(set(labels)),
            # --- G64 新增字段（旧字段一个未删）
            "seed": int(seed), "sampler": "BERNOULLI_PER_FREE_BIT/v2",
            "free_bits": free, "fixed": {int(k): int(v) for k, v in fixed_map.items()},
            "absent_counts": {int(k): int(v) for k, v in absent_counts.items()},
            "present_counts": {int(k): int(v) for k, v in present_counts.items()},
            "branch_counts": {int(b): {"0": int(absent_counts[b]), "1": int(present_counts[b])}
                              for b in free},
            "presence_histogram": {k: int(v) for k, v in sorted(hist.items())},
            "sampling_role": "DIAGNOSTIC_ONLY_NOT_A_PROOF"}



def verify_box_certificate(fam: BoxFamily, certificate: Mapping, *,
                           n_samples: int = 200, seed: int = 20261018,
                           tol: float = BOX_DEFAULT_TOL,
                           max_presence_patterns: int = BOX_MAX_PRESENCE_PATTERNS,
                           max_branch_combos: int = BOX_MAX_BRANCH_COMBOS) -> dict:
    """箱级证书验证（G64 修复版）：区间重算 + 六项覆盖检查，抽样只作诊断。

    修复前只做「区间外包重算 + 抽样标签 + 区域测度和 == 族测度」，缺两两不相交、并集覆盖、
    乘积空间分支、frontier 计账、边界/NUMERIC_BOUNDARY，且把抽样纳入 pass 的充分条件。
    现在 ``pass == formal_pass``，只由本节所列判据决定（``sampling_used_in_pass`` 恒 False）。

    新增检查（对应任务书 §3.4）：(1) 乘积空间两两不相交（半开区间）;(2) 每个在场位组合下
    x 盒并集精确覆盖族盒（不比较测度和）;(3) 自由位 0/1 两分支都出现，未固定位逐分支证明
    同标签;(4) 边界约定 + 余量 < tol 的 NUMERIC_BOUNDARY;(5) frontier 计账 + 未声明覆盖
    一律 formal_pass=false;(6) 抽样不进充分条件。旧返回字段一个未删。
    """
    t0 = time.perf_counter()
    if not (isinstance(tol, (int, float)) and math.isfinite(float(tol)) and float(tol) > 0):
        raise SolverError(f"tol 非法: {tol!r}")
    tol = float(tol)
    reasons: List[str] = []
    codes: List[str] = []

    def _bad(code: str, text: str) -> None:
        if code not in codes:
            codes.append(code)
        reasons.append(text)

    regions_in = list(certificate.get("regions", []) or [])
    declared = certificate.get("coverage_declared") is True
    frontier_raw = certificate.get("frontier_boxes")
    unresolved_raw = certificate.get("unresolved")
    n_frontier = (len(list(frontier_raw)) if isinstance(frontier_raw, (list, tuple))
                  else (1 if frontier_raw else 0))
    n_unresolved = (len(list(unresolved_raw)) if isinstance(unresolved_raw, (list, tuple))
                    else (1 if unresolved_raw else 0))
    budget_hit = bool(certificate.get("budget_hit", False))
    terminated = certificate.get("terminated", None)
    # --- G100 frontier 类型/声明契约（只插在下方 frontier 记账段之前；四条覆盖判据与抽样一字未动） ---
    # 类型层唯一归属：缺键/空值 = FRONTIER_FIELDS_MISSING 或 FRONTIER_TERMINATED_MISSING；
    # 类型不对 = FRONTIER_TYPE_INVALID；terminated 与 frontier_boxes 互相矛盾 = FRONTIER_DECLARATION_INCONSISTENT。
    _g100_terminated_present = "terminated" in certificate
    _g100_frontier_present = "frontier_boxes" in certificate
    _g100_unresolved_present = "unresolved" in certificate
    _g100_type_ok = True
    _g100_fields_ok = True
    if not _g100_terminated_present:
        _bad("FRONTIER_TERMINATED_MISSING",
             "证书缺 terminated 键：frontier 终止状态未声明，类型契约不允许按缺省 None 放行")
        _g100_type_ok = False
    elif not isinstance(terminated, bool):
        _bad("FRONTIER_TYPE_INVALID",
             f"terminated={terminated!r} 不是严格 bool：0/1/None/字符串/容器一律拒")
        _g100_type_ok = False
    if not _g100_frontier_present or frontier_raw is None:
        _bad("FRONTIER_FIELDS_MISSING",
             "证书缺 frontier_boxes 键或值为 null：frontier 未声明，不得按空 frontier 放行")
        _g100_fields_ok = False
    elif not isinstance(frontier_raw, (list, tuple)):
        _bad("FRONTIER_TYPE_INVALID",
             f"frontier_boxes 类型 {type(frontier_raw).__name__} 非法：必须是 list/tuple")
        _g100_fields_ok = False
    if not _g100_unresolved_present or unresolved_raw is None:
        _bad("FRONTIER_FIELDS_MISSING",
             "证书缺 unresolved 键或值为 null：未决集未声明，不得按空集放行")
        _g100_fields_ok = False
    elif not isinstance(unresolved_raw, (list, tuple)):
        _bad("FRONTIER_TYPE_INVALID",
             f"unresolved 类型 {type(unresolved_raw).__name__} 非法：必须是 list/tuple")
        _g100_fields_ok = False
    if _g100_type_ok and _g100_fields_ok:
        if terminated is True and n_frontier > 0:
            _bad("FRONTIER_DECLARATION_INCONSISTENT",
                 f"terminated=true 但 frontier_boxes 非空（{n_frontier} 项）：终止声明与待扩张 frontier 矛盾")
        elif terminated is False and n_frontier == 0:
            _bad("FRONTIER_DECLARATION_INCONSISTENT",
                 "terminated=false 但 frontier_boxes 为空：未终止声明没有任何待扩张 frontier")
    accept_boundary = str(certificate.get("numeric_boundary_policy", "") or "") == "ACCEPT_EXACT"

    box = tuple((_fr(lo), _fr(hi)) for lo, hi in fam.box())
    ndim = len(box)
    free_bits = tuple(int(i) for i in fam.free_presence)
    if any(hi <= lo for lo, hi in box):
        _bad("FAMILY_BOX_DEGENERATE", "声明族 x 盒本身退化/零测度，不能发箱证书")
    if not regions_in:
        _bad("EMPTY_CERTIFICATE", "箱证书无区域")
    if not declared:
        _bad("COVERAGE_NOT_DECLARED",
             "证书未声明 coverage_declared=true：缺 frontier 声明的新证书不得当全域已证")
    if n_frontier or n_unresolved or budget_hit or terminated is False:
        _bad("FRONTIER_UNRESOLVED",
             f"frontier 未清空（frontier={n_frontier}, unresolved={n_unresolved}, "
             f"budget_hit={budget_hit}, terminated={terminated}）-> SEARCH_UNRESOLVED")

    per: List[dict] = []
    usable: List[dict] = []
    boundary: List[dict] = []
    for idx, r in enumerate(regions_in):
        rid = r.get("region_id", f"REGION{idx}")
        claim = str(r.get("label_claim"))
        raw_xs = list(r.get("xs", []) or [])
        rec: dict = {"region_id": rid, "label_claim": claim, "xs": raw_xs, "fixed": {},
                     "recomputed": None, "excluded_from_coverage": False,
                     "exclusion_reason": None, "measure": 0.0, "branch_proof": None,
                     "boundary": [], "must_edges": [], "may_edges": [], "samples": None}
        fixed: Dict[int, int] = {}
        for k, v in dict(r.get("fixed", {}) or {}).items():
            ki, vi = int(k), int(v)
            if vi not in (0, 1):
                _bad("REGION_FIXED_VALUE_INVALID", f"区域 {rid} 固定位 {ki} 的值 {vi} 不是 0/1")
                continue
            if ki not in free_bits:
                _bad("REGION_FIXED_INVALID_BIT", f"区域 {rid} 固定了非自由在场位 {ki}")
                continue
            fixed[ki] = vi
        rec["fixed"] = {int(k): int(v) for k, v in fixed.items()}
        if claim not in ("0", "1"):
            _bad("REGION_LABEL_CLAIM_INVALID", f"区域 {rid} 的 label_claim {claim!r} 不是 '0'/'1'")
        if len(raw_xs) != ndim:
            _bad("REGION_SHAPE_MISMATCH",
                 f"区域 {rid} 的 x 区间维数 {len(raw_xs)} != 族维数 {ndim}")
            rec["excluded_from_coverage"] = True
            rec["exclusion_reason"] = "SHAPE_MISMATCH"
            per.append(rec)
            continue
        xs = [(float(lo), float(hi)) for lo, hi in raw_xs]
        exact = tuple((_fr(lo), _fr(hi)) for lo, hi in xs)
        degen = [i for i, (lo, hi) in enumerate(exact) if hi <= lo]
        zero = [i for i, (lo, hi) in enumerate(exact) if hi == lo]
        if degen or zero:
            if degen:
                _bad("DEGENERATE_INTERVAL", f"区域 {rid} 存在 hi <= lo 的退化区间（维 {degen}）")
            if zero:
                _bad("ZERO_MEASURE_REGION", f"区域 {rid} 乘积测度为 0（维 {zero} 上 lo == hi）")
            rec["excluded_from_coverage"] = True
            rec["exclusion_reason"] = "DEGENERATE" if degen else "ZERO_MEASURE"
        out = [i for i, (lo, hi) in enumerate(exact) if lo < box[i][0] or hi > box[i][1]]
        if out:
            _bad("REGION_OUT_OF_BOX", f"区域 {rid} 的 x 区间越出声明族（维 {out}）")
            rec["excluded_from_coverage"] = True
            rec["exclusion_reason"] = rec["exclusion_reason"] or "OUT_OF_BOX"
        # 区间外包重算（_v_box_geometry 本单未改，仍是独立实现）
        g = _v_box_geometry(fam, xs, fixed)
        env = ident.Envelope(0, g["must_nodes"], g["may_nodes"], g["must_edges"], g["may_edges"],
                             g["must_a"], g["may_a"], g["must_b"], g["may_b"],
                             bool(g["must_a"]), bool(g["may_a"]),
                             bool(g["must_b"]), bool(g["may_b"]))
        lab = _v_direction(env)
        rec["recomputed"] = lab
        rec["must_edges"] = sorted(g["must_edges"])
        rec["may_edges"] = sorted(g["may_edges"])
        if lab in (ident.RegionLabel.UNRESOLVED.value, "INSTRUMENT_DEFECT_MONOTONICITY"):
            _bad("UNRESOLVED_REGION", f"区域 {rid} 未被区间外包判明（{lab}）")
        want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                else ident.RegionLabel.ALL_DISCONNECTED.value)
        if claim in ("0", "1") and lab != want:
            _bad("LABEL_MISMATCH", f"区域 {rid} 声明 {claim} 但验证器区间外包重算 {lab}")
        # 乘积空间：未固定的自由位必须逐分支证明标签一致
        unfixed = [b for b in free_bits if b not in fixed]
        bp = _bv_branch_labels(fam, xs, fixed, unfixed, max_branch_combos)
        rec["branch_proof"] = bp
        if not bp["ok"]:
            _bad("PRODUCT_SPACE_LIMIT",
                 f"区域 {rid} 的未固定自由位组合数 {bp.get('n_combos')} 超上限，未核实")
        elif claim in ("0", "1"):
            wrong = [c for c in bp["combos"] if c["label"] != want]
            if wrong:
                _bad("PRODUCT_BRANCH_LABEL_DIFFERS",
                     f"区域 {rid} 未固定自由位 {unfixed} 的分支标签与声明 {claim} 不一致：{wrong}")
        # 数值边界
        bflags = _bv_boundary_flags(fam, xs, tol)
        rec["boundary"] = bflags
        for f in bflags:
            boundary.append({"region_id": rid, **f})
        # 精确测度（乘积空间）
        m = _bv_exact_measure(exact, len(unfixed))
        rec["measure"] = float(m)
        per.append(rec)
        if not rec["excluded_from_coverage"]:
            usable.append({"region_id": rid, "exact": exact, "fixed": dict(fixed),
                           "measure": m, "claim": claim, "recomputed": lab})
    if boundary:
        _bad("NUMERIC_BOUNDARY",
             f"{len(boundary)} 个几何判据对象的余量 < tol={tol:g}：标 NUMERIC_BOUNDARY，"
             f"默认不得静默通过（显式 numeric_boundary_policy=ACCEPT_EXACT 才放行）")

    # (1) 乘积空间两两不相交
    overlaps: List[dict] = []
    n_pairs = 0
    n_conflict = 0
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            n_pairs += 1
            a, b = usable[i], usable[j]
            if _bv_presence_conflict(a["fixed"], b["fixed"]):
                n_conflict += 1
                continue
            if not _bv_boxes_disjoint(a["exact"], b["exact"]):
                if len(overlaps) < 50:
                    overlaps.append({"a": a["region_id"], "b": b["region_id"],
                                     "kind": "PRODUCT_SPACE_BOX_OVERLAP",
                                     "a_fixed": a["fixed"], "b_fixed": b["fixed"]})
    if overlaps:
        _bad("REGIONS_OVERLAP",
             f"{len(overlaps)} 对区域在（在场位 × 连续坐标）乘积空间重叠：{overlaps[:3]}")

    # (2)(3) 逐在场位组合的并集覆盖 + 分支在场
    patterns: List[dict] = []
    cover_exhausted = False
    if 2 ** len(free_bits) > int(max_presence_patterns):
        _bad("PRODUCT_SPACE_LIMIT",
             f"自由在场位 {len(free_bits)} 个 -> {2 ** len(free_bits)} 个组合超上限")
    else:
        for combo in itertools.product((0, 1), repeat=len(free_bits)):
            pattern = {int(b): int(v) for b, v in zip(free_bits, combo)}
            compat = [u for u in usable if _bv_pattern_compatible(u["fixed"], pattern)]
            prec: dict = {"pattern": {str(b): int(v) for b, v in pattern.items()},
                          "n_regions": len(compat)}
            if not compat:
                _bad("PRODUCT_BRANCH_MISSING", f"在场位组合 {pattern} 无任何区域覆盖")
                prec["covered"] = False
            else:
                budget = {"left": 400000, "exhausted": False}
                covered = _bv_rects_cover(box, [u["exact"] for u in compat], None, budget)
                cover_exhausted = cover_exhausted or bool(budget["exhausted"])
                if budget["exhausted"] and not covered:
                    _bad("COVER_CHECK_LIMIT", f"在场位组合 {pattern} 的覆盖判定预算耗尽，未核实")
                elif not covered:
                    _bad("UNION_NOT_COVERING",
                         f"在场位组合 {pattern} 下区域 x 盒并集不覆盖族盒")
                prec["covered"] = bool(covered)
            patterns.append(prec)

    # 测度和（保留旧字段；不再是唯一覆盖判据）
    full_measure = _bv_exact_measure(box, len(free_bits))
    total_measure = sum((u["measure"] for u in usable), Fraction(0))
    if total_measure != full_measure:
        _bad("MEASURE_MISMATCH",
             f"区域测度和 {float(total_measure):.6g} != 族测度 {float(full_measure):.6g}")

    top = certificate.get("label_claim")
    claims = {str(r.get("label_claim")) for r in regions_in}
    if top is not None and claims != {str(top)}:
        _bad("TOP_LABEL_MISMATCH", "顶层标签与区域标签不一致")

    # (6) 抽样只作诊断：绝不进 pass 的充分条件
    sampling: List[dict] = []
    for rec in per:
        if len(rec["xs"]) != ndim or rec["excluded_from_coverage"]:
            rec["samples"] = {"skipped": True, "reason": rec["exclusion_reason"] or "NOT_USABLE"}
        else:
            rec["samples"] = sample_region(fam, [(float(a), float(b)) for a, b in rec["xs"]],
                                           rec["fixed"], n_samples=n_samples, seed=seed)
        sampling.append({"region_id": rec["region_id"], "samples": rec["samples"]})

    blocking = [c for c in codes if not (c == "NUMERIC_BOUNDARY" and accept_boundary)]
    formal_pass = bool(regions_in) and declared and (not blocking)
    if not regions_in:
        status = "EMPTY_CERTIFICATE"
    elif not declared:
        status = "COVERAGE_UNDECLARED"
    elif "FRONTIER_UNRESOLVED" in codes:
        status = "SEARCH_UNRESOLVED"
    elif "NUMERIC_BOUNDARY" in codes and not accept_boundary:
        status = "NUMERIC_BOUNDARY"
    elif blocking:
        status = "REJECTED"
    else:
        status = "FORMAL_PASS"
    return {"schema": BOX_VERIFY_SCHEMA, "watermark": BOX_VERIFY_WATERMARK,
            "pass": bool(formal_pass), "formal_pass": bool(formal_pass),
            "coverage_declared": bool(declared), "status": status,
            "reasons": reasons, "reason_codes": codes,
            "reason_text": {c: BOX_REASON_TEXT.get(c, "") for c in codes},
            "boundary_convention": dict(BOX_BOUNDARY_CONVENTION), "tol": tol,
            "per_region": per,
            "measure_terminal": float(total_measure), "measure_family": float(full_measure),
            "measure_exact": {"terminal": str(total_measure), "family": str(full_measure),
                              "equal": bool(total_measure == full_measure)},
            "pairwise": {"n_pairs": int(n_pairs), "n_presence_conflict_pairs": int(n_conflict),
                         "n_overlaps": len(overlaps), "overlaps": overlaps},
            "coverage": {"n_patterns": len(patterns),
                         "n_covered": sum(1 for p in patterns if p.get("covered")),
                         "patterns": patterns,
                         "cover_check_budget_exhausted": bool(cover_exhausted)},
            "numeric_boundary": boundary,
            "numeric_boundary_policy": ("ACCEPT_EXACT" if accept_boundary else "REJECT"),
            "frontier": {"coverage_declared": bool(declared),
                         "frontier_boxes_declared": frontier_raw is not None,
                         "n_frontier_boxes": int(n_frontier),
                         "n_unresolved": int(n_unresolved),
                         "budget_hit": budget_hit, "terminated": terminated},
            "sampling_used_in_pass": False,
            "sampling_note": "抽样只作诊断，不构成证明，也不参与 pass 的充分条件",
            "sampling_diagnostics": sampling,
            "n_samples_per_region": int(n_samples),
            "verify_seconds": round(time.perf_counter() - t0, 6)}



__all__ = [
    "SCHEMA_CERT", "SCHEMA_VERIFY", "SCHEMA_PROBE", "VERIFIER_VERSION",
    "SELECTION_PROVENANCE", "CERT_KEYS", "SolverError",
    "TableStructure", "Cube", "Region", "Certificate", "Unresolved",
    "structure_of", "region_states", "region_count", "cubes_partition", "decompose_states",
    "symbolic_envelope", "may_path_bits", "choose_split_bit",
    "resolve_query_cubes", "query_kind",
    "solve_partial", "certify", "verify_certificate",
    "a12_table_with_derivation", "b13_table",
    "BoxFamily", "box_envelope", "refine_family", "sample_region", "verify_box_certificate",
    "BOX_VERIFY_SCHEMA", "BOX_VERIFY_WATERMARK", "BOX_BOUNDARY_CONVENTION", "BOX_REASON_TEXT",
    "BOX_DEFAULT_TOL", "BOX_MAX_PRESENCE_PATTERNS", "BOX_MAX_BRANCH_COMBOS",
]
