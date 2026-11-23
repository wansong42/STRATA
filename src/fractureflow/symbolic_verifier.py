# -*- coding: utf-8 -*-
"""S2-X · 符号/闭式验证器（第二实现；原验证器仍为权威）。

口径与判据：``results/architect_main_20261012_dispatch/task_order_S2_X_symbolic_verifier_20261019.md``
§4 + ``results/s2x_symbolic_verifier/prereg_s2x.json``（``prereg_fingerprint`` = bd015b9b2286）。

三条不可越的边界
----------------
1. **独立性**：本模块**不 import** ``certificate_solver`` / ``statistical_certificate`` /
   ``identification`` 中的任何判定或包络算法函数（见 prereg#protocol.independence 的禁用名册）。
   只读 ``FiniteTable`` 的字段（n_bits / table_id / tags / labels / baseline[].groups /
   derivation）与 ``PublicEvidence`` 的字段；似然只用 ``sequential_evidence.PredictiveMixture``
   （模型似然，非判定）。调算法函数即 ``CIRCULAR_NOT_INDEPENDENT``。
2. **同判**：逐张比 ``pass`` / ``reasons``（逐串同序）/ 允许集 / ``checks``。任一条不成立即
   ``EQUIVALENCE_FAILED``。为此本模块把冻结件的**四个已知洞原样保留**（proof_route /
   state_ids+state_count / 顶层 label_claim 与终末标签 / budget+costs+stats+watermark 内容）。
3. **成本口径**：删掉的是表级 ``range(1 << n_bits)`` 全状态扫描（``_v_allowed``）与
   ``_v_envelope_from_states`` 的逐状态集合循环；保留面（区域状态集物化、见证重放、
   统计后验递推）逐项声明在 cost_table 的 retained 栏，禁算进提速。

符号化实质（为什么等价）
------------------------
* **域**：A12 的观测码是混合进制唯一分解，给定 code 逐动作/逐组反解出 ``popcount(s&g)==digit``
  的**精确子立方并**；与逐状态过滤 ``code(s)==code`` 的状态集逐元素相同。B13 的 reads 直接成
  一个子立方。全程不扫 ``1<<n_bits``。
* **包络**：互斥子立方并上，"所有立方 forces_one(b)" ⟺ "区域内每个状态 b=1"；
  "某立方 allows_one(b)" ⟺ "存在状态 b=1"。与逐状态 intersection/union 同义。
* **方向**：must 有路 ⇒ 全连；may 无路 ⇒ 全断；必须真而 may 假 ⇒ 仪器缺陷（冻结件同序）。

与冻结件的**已声明差异**（不进同判面，见 own_defects）
----------------------------------------------------
* ``verifier_version`` / ``verify_seconds`` 不计入同判。
* 冻结件在同区域上做三路外包互校（集合路 / 逐元素路 / 显式路）。符号面只有**一路**
  （结构 + 子立方代数）。三路互校在语料上恒等（56 张 pass 即证），若某天表的三要素派生与
  ``s2_structure`` 不自洽，冻结件会报"集合路与逐元素路不等"而本实现不会 —— 该分支本单不覆盖。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量（数据契约的字段名/值域，非算法）
# ---------------------------------------------------------------------------

SCHEMA_VERIFY = "s2.verify/v1"
SCHEMA_STAT_VERIFY = "s2n.verify/v1"
VERIFIER_VERSION = "s2x.symbolic/1"
SELECTION_PROVENANCE = "SYMBOLIC_ENVELOPE"
STAT_SELECTION_PROVENANCE = "SYMBOLIC_ENVELOPE+STATISTICAL_BOUND"
BOUND_RULE = "EXACT_SUP_DENOMINATOR"
STAT_JOINT = "JOINT_LIKELIHOOD"
STAT_STEPWISE = "STEPWISE_ENVELOPE"
INF = "INF"
CERT_KEYS = ("schema", "certificate_id", "model_id", "table_id", "query_id", "n_bits",
             "allowed_state_ids", "label_claim", "proof_route", "regions", "witness",
             "verifier_version", "selection_provenance", "budget", "costs", "stats", "watermark")
STAT_CERT_KEYS = ("schema", "certificate_id", "model_id", "table_id", "query_id", "n_bits",
                  "domain_mode", "allowed_state_ids", "label_claim", "proof_route", "regions",
                  "witness", "verifier_version", "selection_provenance", "budget", "costs",
                  "stats", "s2n", "watermark")
FORBIDDEN_KEY_TOKENS = ("truth", "oracle", "hidden")
SCHEMA_CERT = "s2.certificate/v1"
SCHEMA_STAT_CERT = "s2.statistical_certificate/v1"

MAX_CUBES = 4096           # 子立方并的硬上限：越界抛 SymbolicTooComplex，禁静默枚举
MAX_REGION_MATERIALIZE = 1 << 20   # 区域状态集物化硬上限（与 2^n 无关，只防病态输入）


class SymbolicError(ValueError):
    """本模块内部契约破裂：一律抛出，不静默降级。"""


class SymbolicTooComplex(SymbolicError):
    """子立方代数越界（禁以枚举兜底）。"""


# ---------------------------------------------------------------------------
# 子立方代数（全部位运算，不物化状态）
# ---------------------------------------------------------------------------

def _pc(x: int) -> int:
    return int(x).bit_count()


@dataclass(frozen=True)
class SCube:
    """部分赋值：``mask`` 上的位固定为 ``vals``，其余自由。"""

    mask: int
    vals: int

    def __post_init__(self):
        if self.vals & ~self.mask:
            raise SymbolicError("子立方非法：固定值越出固定掩码")

    def count(self, n_bits: int) -> int:
        free = int(n_bits) - _pc(self.mask)
        if free < 0 or (self.mask >> n_bits):
            raise SymbolicTooComplex("子立方的固定位越出 n_bits：拒以物化兜底")
        return 1 << free

    def contains(self, state: int) -> bool:
        s = int(state)
        return (s & self.mask) == self.vals

    def forces_one(self, bit: int) -> bool:
        return bool((self.mask >> bit) & 1) and bool((self.vals >> bit) & 1)

    def forces_zero(self, bit: int) -> bool:
        return bool((self.mask >> bit) & 1) and not bool((self.vals >> bit) & 1)

    def allows_one(self, bit: int) -> bool:
        return not self.forces_zero(bit)

    def states(self, n_bits: int) -> List[int]:
        if self.mask >> n_bits:
            raise SymbolicTooComplex("子立方的固定位越出 n_bits：拒以物化兜底")
        free = [b for b in range(int(n_bits)) if not (self.mask >> b) & 1]
        out = [self.vals]
        for b in free:
            out = out + [s | (1 << b) for s in out]
        return sorted(out)


def cube_disjoint(a: SCube, b: SCube) -> bool:
    return bool((a.vals ^ b.vals) & a.mask & b.mask)


def cube_subset(a: SCube, b: SCube) -> bool:
    return (b.mask & a.mask) == b.mask and ((a.vals ^ b.vals) & b.mask) == 0


def cube_intersect(a: SCube, b: SCube) -> Optional[SCube]:
    if cube_disjoint(a, b):
        return None
    mask = a.mask | b.mask
    vals = (a.vals & a.mask) | (b.vals & b.mask)
    if vals & ~mask:
        return None
    return SCube(mask, vals)


def cube_subtract(a: SCube, b: SCube, n_bits: int) -> List[SCube]:
    """``a \\ b``：逐固定位切分，返回互斥子立方并（不物化）。"""
    if cube_intersect(a, b) is None:
        return [a]
    out: List[SCube] = []
    cur = [a]
    for bit in range(int(n_bits)):
        if not (b.mask >> bit) & 1:
            continue
        bval = (b.vals >> bit) & 1
        nxt: List[SCube] = []
        for c in cur:
            if (c.mask >> bit) & 1:
                if ((c.vals >> bit) & 1) == bval:
                    nxt.append(c)
                else:
                    out.append(c)
            else:
                out.append(SCube(c.mask | (1 << bit), c.vals | ((1 - bval) << bit)))
                nxt.append(SCube(c.mask | (1 << bit), c.vals | (bval << bit)))
        cur = nxt
        if not cur:
            break
    return out


def union_count(cubes: Sequence[SCube], n_bits: int) -> int:
    return sum(c.count(n_bits) for c in cubes)


def union_contains(cubes: Sequence[SCube], state: int) -> bool:
    return any(c.contains(state) for c in cubes)


def union_covers(a: Sequence[SCube], b: Sequence[SCube], n_bits: int) -> bool:
    return all(_cube_covered(x, b, n_bits) for x in a)


def _cube_covered(c: SCube, cover: Sequence[SCube], n_bits: int) -> bool:
    cur = [c]
    for e in cover:
        if not cur:
            return True
        nxt: List[SCube] = []
        for p in cur:
            if cube_intersect(p, e) is None:
                nxt.append(p)
            else:
                nxt.extend(cube_subtract(p, e, n_bits))
        cur = nxt
    return not cur


def union_equal(a: Sequence[SCube], b: Sequence[SCube], n_bits: int) -> bool:
    if union_count(a, n_bits) != union_count(b, n_bits):
        return False
    return union_covers(a, b, n_bits) and union_covers(b, a, n_bits)


def union_overlap(a: Sequence[SCube], b: Sequence[SCube]) -> bool:
    return any(cube_intersect(x, y) is not None for x in a for y in b)


def all_states_domain(n_bits: int) -> List[SCube]:
    """全状态域 = 一个零固定位子立方（不物化 2^n）。"""
    return [SCube(0, 0)]


class SUnion:
    """互斥子立方并（规范化）。所有判定在立方并上做，不物化状态。"""

    __slots__ = ("n_bits", "cubes")

    def __init__(self, cubes: Iterable[SCube] = (), n_bits: int = 0):
        self.n_bits = int(n_bits)
        self.cubes: List[SCube] = []
        for c in cubes:
            self.add(c)

    @staticmethod
    def from_states(states: Iterable[int], n_bits: int) -> "SUnion":
        """升序状态集 → 子立方并（只用于"声明允许集是否等于域"的对照，不做 2^n 扫描）。"""
        ss = sorted(int(s) for s in states)
        out = SUnion((), n_bits)
        stack = [ss]
        while stack:
            part = stack.pop()
            if not part:
                continue
            fixed = (1 << int(n_bits)) - 1
            for s in part[1:]:
                fixed &= ~(s ^ part[0])
            if (1 << (int(n_bits) - _pc(fixed))) == len(part):
                out.add(SCube(fixed, part[0] & fixed))
                continue
            for b in range(int(n_bits)):
                if not (fixed >> b) & 1:
                    z = [s for s in part if not (s >> b) & 1]
                    o = [s for s in part if (s >> b) & 1]
                    if z and o:
                        stack.append(o)
                        stack.append(z)
                        break
            else:
                raise SymbolicError("状态集无法分解为子立方")
        return out

    def add(self, c: SCube) -> None:
        pieces = [c]
        for e in self.cubes:
            nxt: List[SCube] = []
            for p in pieces:
                if cube_intersect(p, e) is None:
                    nxt.append(p)
                else:
                    nxt.extend(cube_subtract(p, e, self.n_bits))
            pieces = nxt
            if not pieces:
                return
        self.cubes.extend(pieces)
        if len(self.cubes) > MAX_CUBES:
            raise SymbolicTooComplex("子立方并超过 %d 个：拒以枚举兜底" % MAX_CUBES)

    def count(self) -> int:
        return union_count(self.cubes, self.n_bits)

    def contains(self, state: int) -> bool:
        return union_contains(self.cubes, state)

    def covers(self, other: "SUnion") -> bool:
        return all(_cube_covered(c, other.cubes, self.n_bits) for c in self.cubes)

    def equal(self, other: "SUnion") -> bool:
        return self.count() == other.count() and self.covers(other) and other.covers(self)

    def overlaps(self, other: "SUnion") -> bool:
        return union_overlap(self.cubes, other.cubes)

    def difference_count(self, other: "SUnion") -> int:
        rest = SUnion(self.cubes, self.n_bits)
        for e in other.cubes:
            nxt = SUnion((), self.n_bits)
            for p in rest.cubes:
                for q in (cube_subtract(p, e, self.n_bits)
                          if cube_intersect(p, e) is not None else [p]):
                    nxt.add(q)
            rest = nxt
        return rest.count()

    def intersection_count(self, other: "SUnion") -> int:
        n = 0
        for x in self.cubes:
            for y in other.cubes:
                z = cube_intersect(x, y)
                if z is not None:
                    n += z.count(self.n_bits)
        return n

    def states(self, limit: int = MAX_REGION_MATERIALIZE) -> List[int]:
        if self.count() > int(limit):
            raise SymbolicTooComplex("子立方并状态数 %d 超过物化上限 %d" % (self.count(), limit))
        out: set = set()
        for c in self.cubes:
            out.update(c.states(self.n_bits))
        return sorted(out)


# ---------------------------------------------------------------------------
# 表结构（自建结构体；只读 tags["s2_structure"]）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SStructure:
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
            raise SymbolicError("未知表结构 kind=%r" % (self.kind,))

    @staticmethod
    def from_dict(d: Mapping) -> "SStructure":
        return SStructure(
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

    @staticmethod
    def from_table(table) -> "SStructure":
        raw = (getattr(table, "tags", None) or {}).get("s2_structure")
        if not raw:
            raise SymbolicError("%s: 表未登记 s2_structure" % getattr(table, "table_id", "?"))
        return SStructure.from_dict(raw if isinstance(raw, dict) else json.loads(raw))


def _queries_meta(table) -> dict:
    raw = (getattr(table, "tags", None) or {}).get("s2_queries")
    if not raw:
        raise SymbolicError("%s: 表未登记 s2_queries" % getattr(table, "table_id", "?"))
    return raw if isinstance(raw, dict) else json.loads(raw)


# ---------------------------------------------------------------------------
# 域反解（把 O(2^n) 换成 O(组数)）
# ---------------------------------------------------------------------------

@dataclass
class SDomain:
    cubes: Tuple[SCube, ...]
    n_bits: int
    reasons: List[str] = field(default_factory=list)
    kind: str = ""
    code: Optional[int] = None

    @property
    def count(self) -> int:
        return union_count(self.cubes, self.n_bits)

    @property
    def union(self) -> SUnion:
        return SUnion(self.cubes, self.n_bits)


def _group_bases(groups: Sequence[int]) -> List[int]:
    return [int(g).bit_count() + 1 for g in groups]


def _action_radix(groups: Sequence[int]) -> int:
    r = 1
    for base in _group_bases(groups):
        r *= base
    return r


def _refine_cube_for_group(c: SCube, g: int, digit: int, n_bits: int) -> List[SCube]:
    """把 ``c`` 细化为 ``popcount(s & g) == digit`` 的精确子立方并。"""
    g = int(g)
    fixed_one = c.mask & c.vals & g
    fixed_zero = c.mask & (~c.vals) & g
    free = g & ~c.mask & ((1 << int(n_bits)) - 1)
    need = int(digit) - _pc(fixed_one)
    free_bits = [b for b in range(int(n_bits)) if (free >> b) & 1]
    if need < 0 or need > len(free_bits):
        return []
    out: List[SCube] = []
    base_mask = c.mask | fixed_zero | free      # 组内自由位全部固定（先按"全 0"）
    base_vals = c.vals
    for combo in _combinations(free_bits, need):
        m = base_mask
        v = base_vals
        for b in combo:
            v |= (1 << b)
        out.append(SCube(m, v))
        if len(out) > MAX_CUBES:
            raise SymbolicTooComplex("域细化子立方数超过 %d" % MAX_CUBES)
    return out


def _combinations(bits: Sequence[int], k: int) -> List[Tuple[int, ...]]:
    if k == 0:
        return [()]
    if k > len(bits):
        return []
    out: List[Tuple[int, ...]] = []

    def rec(i: int, acc: Tuple[int, ...]) -> None:
        if len(acc) == k:
            out.append(acc)
            return
        for j in range(i, len(bits)):
            rec(j + 1, acc + (bits[j],))

    rec(0, ())
    return out


def symbolic_domain_a12(table, query_id: str) -> SDomain:
    """A12_CLASS：混合进制唯一分解 → 精确子立方并（零 2^n 扫描）。"""
    n_bits = int(table.n_bits)
    prefix = "A12:class:"
    if not query_id.startswith(prefix):
        return SDomain((), n_bits, ["query_id 形态不符: %r" % (query_id,)], "A12_CLASS")
    code = int(query_id[len(prefix):])
    baseline = tuple(table.baseline or ())
    # 反解：acc = ((...)*radix_a + c_a)，c_a = (...*(base)+digit)
    digits: List[List[Tuple[int, int]]] = []     # 每动作 → [(group, digit), ...]
    acc = int(code)
    ok = True
    for a in reversed(baseline):
        groups = tuple(int(g) for g in a.groups)
        radix = _action_radix(groups)
        c = acc % radix
        acc //= radix
        rows: List[Tuple[int, int]] = []
        for g in reversed(groups):
            base = int(g).bit_count() + 1
            rows.append((int(g), c % base))
            c //= base
        if c != 0:
            ok = False
        digits.append(list(reversed(rows)))
    digits.reverse()
    if not ok or acc != 0:
        return SDomain((), n_bits, ["验证器重算的允许集为空 (code=%d)" % code], "A12_CLASS", code)
    cubes: List[SCube] = [SCube(0, 0)]
    for rows in digits:
        for (g, d) in rows:
            nxt: List[SCube] = []
            for c in cubes:
                nxt.extend(_refine_cube_for_group(c, g, d, n_bits))
            cubes = nxt
            if not cubes:
                break
        if not cubes:
            break
    cubes = [c for c in cubes if c.count(n_bits) > 0]
    if not cubes:
        return SDomain((), n_bits, ["验证器重算的允许集为空 (code=%d)" % code], "A12_CLASS", code)
    return SDomain(tuple(cubes), n_bits, [], "A12_CLASS", code)


def symbolic_domain(table, query_id: str) -> SDomain:
    """query_id → 允许集的子立方分解（与 ``_v_allowed`` 同义，不扫全状态）。"""
    n_bits = int(table.n_bits)
    meta = _queries_meta(table)
    kind = str(meta["kind"])
    if kind == "A12_CLASS":
        return symbolic_domain_a12(table, query_id)
    if kind == "B13_SCENARIO":
        prefix = "B13:scenario:"
        if not query_id.startswith(prefix):
            return SDomain((), n_bits, ["query_id 形态不符: %r" % (query_id,)], kind)
        name = query_id[len(prefix):]
        sc = (meta.get("scenarios") or {}).get(name)
        if sc is None:
            return SDomain((), n_bits, ["未知情景 %r" % (name,)], kind)
        if not sc.get("observation_in_forward_image", True):
            return SDomain((), n_bits, [], kind)
        mask = 0
        vals = 0
        bad = False
        for (bit, val) in sc["reads"]:
            b, v = int(bit), int(val) & 1
            if (mask >> b) & 1 and ((vals >> b) & 1) != v:
                bad = True
            mask |= 1 << b
            if v:
                vals |= 1 << b
        if bad or (vals & ~mask) or (mask >> n_bits):
            return SDomain((), n_bits, [], kind)
        return SDomain((SCube(mask, vals),), n_bits, [], kind)
    return SDomain((), n_bits, ["未知查询类型 %r" % (kind,)], kind)


# ---------------------------------------------------------------------------
# 包络（子立方代数；与逐状态 ∀/∃ 同义）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SEnvelope:
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
        """与冻结件 ``Envelope.element_counts()`` 逐键同形（外包计数的对外面）。"""
        return {
            "NODE": {"must": len(self.must_nodes), "may": len(self.may_nodes)},
            "EDGE": {"must": len(self.must_edges), "may": len(self.may_edges)},
            "ATTACHMENT": {"must": int(self.must_attach_a) + int(self.must_attach_b),
                           "may": int(self.may_attach_a) + int(self.may_attach_b),
                           "must_a": bool(self.must_attach_a), "may_a": bool(self.may_attach_a),
                           "must_b": bool(self.must_attach_b), "may_b": bool(self.may_attach_b)},
        }


def _canon_pair(u, v):
    return (u, v) if not (v < u) else (v, u)


def envelope_of_cubes(struct: SStructure, cubes: Sequence[SCube], n_bits: int,
                      n_states: int) -> SEnvelope:
    """从部分赋值直接推 must/may；不做任何逐状态遍历。"""
    if not cubes:
        raise SymbolicError("空区域不得建外包")
    if n_states <= 0:
        raise SymbolicError("空区域不得建外包")
    if struct.kind == "EDGE_BITS":
        must_edges = frozenset(_canon_pair(u, v) for (b, u, v) in struct.edges
                               if all(c.forces_one(b) for c in cubes))
        may_edges = frozenset(_canon_pair(u, v) for (b, u, v) in struct.edges
                              if any(c.allows_one(b) for c in cubes))
        must_nodes = frozenset(struct.nodes)
        may_nodes = frozenset(struct.nodes)
        must_attach_a = all(any(c.forces_one(b) for b in struct.a_incident_bits) for c in cubes)
        may_attach_a = any(any(c.allows_one(b) for b in struct.a_incident_bits) for c in cubes)
        must_attach_b = all(any(c.forces_one(b) for b in struct.b_incident_bits) for c in cubes)
        may_attach_b = any(any(c.allows_one(b) for b in struct.b_incident_bits) for c in cubes)
        return SEnvelope(
            n_states, must_nodes, may_nodes, must_edges, may_edges,
            frozenset(struct.a_nodes) if must_attach_a else frozenset(),
            frozenset(struct.a_nodes) if may_attach_a else frozenset(),
            frozenset(struct.b_nodes) if must_attach_b else frozenset(),
            frozenset(struct.b_nodes) if may_attach_b else frozenset(),
            must_attach_a, may_attach_a, must_attach_b, may_attach_b)
    base = len(struct.certain)
    n_unc = len(struct.uncertain)
    must_nodes = set(range(base))
    may_nodes = set(range(base))
    for j in range(n_unc):
        if all(c.forces_one(j) for c in cubes):
            must_nodes.add(base + j)
        if any(c.allows_one(j) for c in cubes):
            may_nodes.add(base + j)
    must_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                           if i in must_nodes and j in must_nodes)
    may_edges = frozenset((i, j) for (i, j) in struct.contact_edges
                          if i in may_nodes and j in may_nodes)
    must_a = frozenset(x for x in must_nodes if x in struct.attach_a_disks)
    may_a = frozenset(x for x in may_nodes if x in struct.attach_a_disks)
    must_b = frozenset(x for x in must_nodes if x in struct.attach_b_disks)
    may_b = frozenset(x for x in may_nodes if x in struct.attach_b_disks)
    return SEnvelope(n_states, frozenset(must_nodes), frozenset(may_nodes),
                     must_edges, may_edges, must_a, may_a, must_b, may_b,
                     bool(must_a), bool(may_a), bool(must_b), bool(may_b))


# ---------------------------------------------------------------------------
# 方向（自带可达性实现）
# ---------------------------------------------------------------------------

def reachable(nodes: Iterable, edges: Iterable, side_a: Iterable, side_b: Iterable) -> bool:
    na, nb = set(side_a) & set(nodes), set(side_b) & set(nodes)
    if not na or not nb:
        return False
    adj: Dict = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen = set(na)
    queue = list(na)
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        if x in nb:
            return True
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return False


ALL_CONNECTED = "ALL_CONNECTED"
ALL_DISCONNECTED = "ALL_DISCONNECTED"
UNRESOLVED = "UNRESOLVED"
MONOTONICITY_DEFECT = "INSTRUMENT_DEFECT_MONOTONICITY"


def direction(env: SEnvelope) -> str:
    """must 有路 ⇒ 全连；may 无路 ⇒ 全断；必须真而 may 假 ⇒ 仪器缺陷（与冻结件同序）。"""
    must = bool(env.must_attach_a and env.must_attach_b
                and reachable(env.must_nodes, env.must_edges, env.must_a_side, env.must_b_side))
    may = bool(env.may_attach_a and env.may_attach_b
               and reachable(env.may_nodes, env.may_edges, env.may_a_side, env.may_b_side))
    if must and not may:
        return MONOTONICITY_DEFECT
    if must:
        return ALL_CONNECTED
    if not may:
        return ALL_DISCONNECTED
    return UNRESOLVED


# ---------------------------------------------------------------------------
# 逐状态回退审计（只在方向不符/未解的红证书上触发；不进计时语料）
# ---------------------------------------------------------------------------

def derivation_connected(d) -> bool:
    """单状态三要素重放（自实现 DFS，不调 identification.state_connected）。"""
    return bool(d.attach_a and d.attach_b
                and reachable(d.nodes, d.edges, d.a_side, d.b_side))


def reverify_witness(d, claimed_label: int, claimed_path: Optional[Sequence[object]]) -> dict:
    """见证重验（逐条镜像 reverify_witness 的判据与原因串）。"""
    if int(d.label) != int(claimed_label):
        return {"ok": False, "reason": "LABEL_MISMATCH"}
    replay = derivation_connected(d)
    if int(claimed_label) == 1 and not replay:
        return {"ok": False, "reason": "CLAIMED_CONNECTED_BUT_STATE_DISCONNECTED"}
    if int(claimed_label) == 0 and replay:
        return {"ok": False, "reason": "CLAIMED_DISCONNECTED_BUT_STATE_CONNECTED"}
    if int(claimed_label) == 1 and claimed_path is not None:
        if len(claimed_path) < 2:
            raise SymbolicError("claimed_path 不足两端点")
        nodes = frozenset(claimed_path)
        if any(x not in d.nodes for x in nodes):
            return {"ok": False, "reason": "PATH_USES_ABSENT_NODE"}
        pedges = frozenset(_canon_pair(u, v) for u, v in zip(list(claimed_path)[:-1],
                                                             list(claimed_path)[1:]))
        if any((u, v) not in d.edges for (u, v) in pedges):
            return {"ok": False, "reason": "PATH_USES_ABSENT_EDGE"}
        if claimed_path[0] not in d.a_side:
            return {"ok": False, "reason": "PATH_START_NOT_ATTACHED"}
        if claimed_path[-1] not in d.b_side:
            return {"ok": False, "reason": "PATH_END_NOT_ATTACHED"}
    return {"ok": True, "reason": "REPLAYED"}


def _label_array(table):
    """表只有一张查询标签阵列（A12/B13/S1T 各一）；多张则退回逐状态派生。"""
    labs = getattr(table, "labels", None) or {}
    if len(labs) == 1:
        return list(labs.values())[0]
    return None


def _cube_list(r: Mapping, n_bits: int, reasons: List[str], fmt: str = "f") -> List[SCube]:
    """区域 cubes → SCube 列表；非法子立方**静默跳过**（原因串由 ``_states_of`` 负责记）。"""
    out: List[SCube] = []
    for c in r.get("cubes", []):
        m, v = int(c["fixed_mask"]), int(c["fixed_vals"])
        if v & ~m:
            continue
        out.append(SCube(m, v))
    return out


def _states_of(r: Mapping, n_bits: int, reasons: List[str], fmt: str = "f") -> List[int]:
    """区域状态集（集合语义；与冻结件同名同调用点的物化路径，规模 = |区域|）。"""
    out: set = set()
    rid = r.get("region_id")
    for c in r.get("cubes", []):
        m, v = int(c["fixed_mask"]), int(c["fixed_vals"])
        if v & ~m:
            reasons.append(("区域 %s 子立方非法" % rid) if fmt == "f"
                           else ("区域 %r 子立方非法" % (rid,)))
            continue
        if m >> n_bits:
            raise SymbolicTooComplex("区域 %r 的固定位越出 n_bits %d" % (rid, n_bits))
        free = [b for b in range(n_bits) if not (m >> b) & 1]
        ss = [v]
        for b in free:
            ss = ss + [s | (1 << b) for s in ss]
        if len(out) + len(ss) > MAX_REGION_MATERIALIZE:
            raise SymbolicTooComplex("区域 %r 状态数超过物化上限" % (rid,))
        out.update(ss)
    return sorted(out)


# ---------------------------------------------------------------------------
# 解析面验证器
# ---------------------------------------------------------------------------

def verify_certificate_symbolic(table, query_id: str, certificate: Mapping) -> dict:
    """独立重放（符号面）：允许集/树/外包/方向/标签/见证/覆盖全部重算。

    原因串与顺序逐条对齐 ``certificate_solver.verify_certificate``（含其四个已知洞）。
    """
    t0 = time.perf_counter()
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if not isinstance(certificate, Mapping):
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": ["证书不是映射"],
                "verifier_version": VERIFIER_VERSION, "verify_seconds": 0.0}
    cert = dict(certificate)
    unknown = [k for k in cert if k not in CERT_KEYS]
    if unknown:
        reasons.append("证书含未知顶层字段 %s" % (unknown,))
    bad_keys = [k for k in cert if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad_keys:
        reasons.append("证书含真值/神谕字段 %s" % (bad_keys,))
    prov = cert.get("selection_provenance")
    if prov != SELECTION_PROVENANCE:
        reasons.append("选点来源非法 %r（只许 %s）" % (prov, SELECTION_PROVENANCE))
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != table.n_bits:
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    struct = SStructure.from_table(table)
    n_bits = int(table.n_bits)

    domain = symbolic_domain(table, query_id)
    checks["allowed_recomputed"] = domain.count
    if domain.reasons:
        reasons.extend(domain.reasons)
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if not _declared_matches_domain(declared_allowed, domain):
        reasons.append("允许集不符：声明 %d / 重算 %d" % (len(declared_allowed), domain.count))
    domain_union = domain.union
    domain_states = (domain_union.states() if domain.count <= MAX_REGION_MATERIALIZE else None)
    domain_set = set(domain_states) if domain_states is not None else None

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
        reasons.append("根区域不唯一 (%d)" % (len(roots),))
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    root = roots[0]
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append("区域 %s 的父区域不存在" % (r.get("region_id"),))
            kids.setdefault(p, []).append(r)

    root_states = _states_of(root, n_bits, reasons)
    if domain.count <= MAX_REGION_MATERIALIZE:
        root_ok = (root_states == domain_states)
    else:
        root_ok = _set_equals_domain(root_states, domain_union)
    if not root_ok:
        reasons.append("根区域 != 允许集（%d vs %d）" % (len(root_states), domain.count))

    # 树：内部节点子区域必须两两不交且并集 = 父
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = _states_of(r, n_bits, reasons)
        if r.get("is_terminal"):
            if ch:
                reasons.append("终末区域 %s 仍有子区域" % (r.get("region_id"),))
            if not st:
                reasons.append("终末区域 %s 为空" % (r.get("region_id"),))
            continue
        if not ch:
            reasons.append("内部区域 %s 无子区域" % (r.get("region_id"),))
            continue
        union: List[int] = []
        for c in ch:
            union.extend(_states_of(c, n_bits, reasons))
        if len(union) != len(set(union)):
            reasons.append("区域 %s 子区域重叠" % (r.get("region_id"),))
        if sorted(set(union)) != st:
            reasons.append("区域 %s 子区域并集 != 父区域" % (r.get("region_id"),))

    # 终末区域：外包 + 方向 + 标签 + 见证
    audit_states = 0
    terminal = [r for r in regions if r.get("is_terminal")]
    claims: set = set()
    labels = _label_array(table)
    for r in terminal:
        rid = r.get("region_id")
        st = _states_of(r, n_bits, reasons)
        claim = r.get("label_claim")
        if claim not in ("0", "1"):
            reasons.append("终末区域 %s 标签非法 %r" % (rid, claim))
            continue
        claims.add(claim)
        if not st:
            reasons.append("终末区域 %s 为空：空区域不得建外包、不得发证书" % (rid,))
            continue
        if domain_set is not None:
            outside = not set(st) <= domain_set
        else:
            outside = not all(domain_union.contains(s) for s in st)
        if outside:
            reasons.append("终末区域 %s 含允许集外状态" % (rid,))
        if getattr(table, "derivation", None) is None:
            reasons.append("表未提供派生：无法重算外包")
            continue
        cubes = _cube_list(r, n_bits, reasons)
        env = envelope_of_cubes(struct, cubes, n_bits, len(st))
        audit_states += len(st)
        declared = r.get("envelope_counts")
        if declared is not None and declared != env.element_counts():
            reasons.append("区域 %s 声明外包计数与重算不等" % (rid,))
        lab = direction(env)
        if lab == MONOTONICITY_DEFECT:
            reasons.append("区域 %s 外包单调性破裂" % (rid,))
            continue
        want = ALL_CONNECTED if claim == "1" else ALL_DISCONNECTED
        if lab != want:
            reasons.append("区域 %s 声明 %s 但重算 %s" % (rid, claim, lab))
            # 方向不符 ⇒ 连通性非齐一 ⇒ 退回逐状态审计（与冻结件同判据）
            _audit_states(table, query_id, rid, st, claim, reasons)
        else:
            # 包络已证区域内连通性恒等于 claim ⇒ 忠实性 ⟺ labels[region] == claim
            first_bad = None
            if labels is None:
                first_bad = _first_label_bad_by_derivation(table, query_id, st, claim)
            else:
                for s in st:
                    if int(labels[int(s)]) != int(claim):
                        first_bad = s
                        break
            if first_bad is not None:
                reasons.append("区域 %s 状态 %s 派生与标签不忠实" % (rid, first_bad))
        w = r.get("witness")
        if not w or "state_id" not in w:
            reasons.append("区域 %s 无见证" % (rid,))
        else:
            sid = int(w["state_id"])
            if sid not in set(st):
                reasons.append("区域 %s 见证 %s 不在区域内" % (rid, sid))
            elif int(w.get("label", -1)) != int(claim):
                reasons.append("区域 %s 见证标签与区域标签不符" % (rid,))
            else:
                rep = reverify_witness(table.derivation(query_id, sid), int(claim), w.get("path"))
                if not rep["ok"]:
                    reasons.append("区域 %s 见证不可重放 %s" % (rid, rep.get("reason")))

    # 覆盖：终末区域两两不交、并集 = 允许集
    all_states: List[int] = []
    for r in terminal:
        all_states.extend(_states_of(r, n_bits, reasons))
    if len(all_states) != len(set(all_states)):
        reasons.append("终末区域划分不互斥")
    if domain_states is not None:
        missing = len(domain_set - set(all_states))
        extra = len(set(all_states) - domain_set)
    else:
        u = SUnion((), n_bits)
        for s in all_states:
            u.add(SCube(((1 << n_bits) - 1), int(s)))
        inter = domain_union.intersection_count(u)
        missing = domain.count - inter
        extra = u.count() - inter
    if missing or extra:
        reasons.append("终末区域覆盖 != 允许集 (缺 %d / 多 %d)" % (missing, extra))
    if len(claims) > 1:
        reasons.append("终末区域标签不一致 %s：混合查询不得发单标签证书" % (sorted(claims),))
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    elif int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
        reasons.append("证书顶层见证标签与 label_claim 不符")
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION,
            "checks": {"allowed_recomputed": domain.count,
                       "regions": len(regions), "terminal_regions": len(terminal),
                       "audit_states": audit_states,
                       "claims": sorted(claims)},
            "verify_seconds": round(time.perf_counter() - t0, 6)}


def _first_label_bad_by_derivation(table, query_id: str, st: Sequence[int], claim: str):
    for s in st:
        if int(table.derivation(query_id, int(s)).label) != int(claim):
            return s
    return None


def _audit_states(table, query_id: str, rid, st: Sequence[int], claim: str,
                  reasons: List[str]) -> None:
    """红证书上的逐状态审计（忠实性 + 标签），与冻结件逐条同序。"""
    derivs = [table.derivation(query_id, int(s)) for s in st]
    for d in derivs:
        if int(d.label) != int(1 if derivation_connected(d) else 0):
            reasons.append("区域 %s 状态 %s 派生与标签不忠实" % (rid, d.state_id))
            break
        if int(d.label) != int(claim):
            reasons.append("区域 %s 状态 %s 标签 %d != 声明 %s" % (rid, d.state_id, d.label, claim))
            break


def _declared_matches_domain(declared: Sequence[int], domain: SDomain) -> bool:
    """``declared == sorted(recomputed)`` 的符号判：严格递增 + 计数相等 + 逐元素落域。"""
    if len(declared) != domain.count:
        return False
    full = 1 << domain.n_bits
    prev = -1
    du = domain.union
    for s in declared:
        if s <= prev or s < 0 or s >= full:
            return False
        prev = s
        if not du.contains(int(s)):
            return False
    return True


def _set_equals_domain(states: Sequence[int], u: SUnion) -> bool:
    if len(states) != u.count():
        return False
    return all(u.contains(int(s)) for s in states)


# ---------------------------------------------------------------------------
# 统计面：重放（保留面）+ 历史摘要（自写）
# ---------------------------------------------------------------------------

def record_key_public(rec: Mapping) -> tuple:
    return (str(rec.get("design_id")), int(rec.get("well_id", -1)),
            tuple((int(c), int(n)) for c, n in rec.get("counts", [])))


def history_sha12_symbolic(public) -> str:
    doc = {"model_id": str(public.model_id), "tier": str(public.tier),
           "states": [int(s) for s in public.states],
           "steps": [{"action_id": str(a["action_id"]),
                      "record_key": list(record_key_public(r))} for a, r in public.steps]}
    raw = json.dumps(doc, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _is_inf(x) -> bool:
    return isinstance(x, str) and x == INF


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(str(x))


def _max_e(vals: Sequence[object]):
    best = Fraction(0, 1)
    for v in vals:
        if _is_inf(v):
            return INF
        if v > best:
            best = v
    return best


def replay_symbolic(public, reasons: Optional[List[str]] = None) -> dict:
    """保留面：自实现后验递推（Q_u / D_u / E_u / P），与冻结件同一递推。

    只用 ``PredictiveMixture``（模型似然，非判定）；权重更新顺序与冻结件逐行相同：
    先取 q 与 p 行，再用本步记录更新权重。
    """
    from fractureflow.sequential_evidence import PredictiveMixture

    reasons = [] if reasons is None else reasons
    states = tuple(int(s) for s in public.states)
    if not public.steps:
        reasons.append("公开历史为空：无法验证统计界")
        return {"Q": [Fraction(1, 1)], "D": [[Fraction(1, 1)] * len(states)],
                "E": [[Fraction(1, 1)] * len(states)], "P": [], "states": states}
    mix = PredictiveMixture(states, public.kernel, public.steps[0][0],
                            mode="posterior", tier=public.tier, cache=public.cache)
    Q = [Fraction(1, 1)]
    D: List[List[Fraction]] = [[Fraction(1, 1)] * len(states)]
    P: List[List[Fraction]] = []
    for (action, record) in public.steps:
        q = mix.prob(record, action)
        if q == 0:
            reasons.append("公开历史上 q=0：验证器不建界（0/0 禁发证书）")
            break
        Q.append(Q[-1] * q)
        p_row = [mix.state_probability(record, s, action) for s in states]
        P.append(p_row)
        d_row = [D[-1][i] * p_row[i] for i in range(len(states))]
        D.append(d_row)
        w = {}
        tot = Fraction(0, 1)
        for i, s in enumerate(states):
            w[s] = mix.weights[s] * mix.state_probability(record, s, action)
            tot += w[s]
        if tot == 0:
            reasons.append("验证器重放遇到全零似然：MODEL_CONFLICT")
            break
        mix.weights = {s: v / tot for s, v in w.items()}
    E: List[List[object]] = []
    for u in range(len(D)):
        E.append([INF if d == 0 else Q[u] / d for d in D[u]])
    return {"Q": Q, "D": D, "E": E, "P": P, "states": states}


# ---------------------------------------------------------------------------
# 统计面验证器
# ---------------------------------------------------------------------------

def verify_statistical_symbolic(table, public, query_id: str, certificate: Mapping) -> dict:
    """独立重放（统计面符号版）：统计界方向/阈值/逐状态排除性、区域树覆盖、must/may、见证。

    判据与 ``statistical_certificate.verify_statistical_certificate`` 逐条同序；
    ``s2.certificate/v1`` 转交本模块的解析面（等价于冻结件的接口复用）。
    """
    t0 = time.perf_counter()
    if not isinstance(certificate, Mapping):
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False, "reasons": ["证书不是映射"],
                "verifier_version": VERIFIER_VERSION, "verify_seconds": 0.0}
    cert = dict(certificate)
    if str(cert.get("schema")) == SCHEMA_CERT:
        out = dict(verify_certificate_symbolic(table, query_id, cert))
        out["s2n"] = {"reused_s2_verifier": True,
                      "note": "s2.certificate/v1 原样转交 S2-N0 符号解析面（接口复用）"}
        out["verify_seconds"] = round(time.perf_counter() - t0, 6)
        return out

    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if str(cert.get("schema")) != SCHEMA_STAT_CERT:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": ["未知证书 schema %r" % (cert.get("schema"),)],
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    unknown = [k for k in cert if k not in STAT_CERT_KEYS]
    if unknown:
        reasons.append("证书含未知顶层字段 %s" % (unknown,))
    bad = [k for k in cert if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad:
        reasons.append("证书含真值/神谕字段 %s" % (bad,))
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("selection_provenance") != STAT_SELECTION_PROVENANCE:
        reasons.append("选点来源非法 %r" % (cert.get("selection_provenance"),))
    if cert.get("verifier_version") != "s2n.verifier/1":
        reasons.append("证书 verifier_version 不是 %s" % "s2n.verifier/1")
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != int(table.n_bits):
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    public_obj = public
    if not (hasattr(public_obj, "steps") and hasattr(public_obj, "kernel")
            and hasattr(public_obj, "states")):
        raise SymbolicError("验证器只接受带有公开历史字段的对象（duck-typed PublicEvidence）")
    if str(cert.get("model_id")) != public_obj.model_id:
        reasons.append("证书 model_id 与公开历史不符")
    s2n = dict(cert.get("s2n") or {})
    if str(s2n.get("history_sha12")) != history_sha12_symbolic(public_obj):
        reasons.append("公开历史摘要不符：证书不是对这段观测出的")
    if str(s2n.get("tier")) != str(public_obj.tier):
        reasons.append("噪声档不符 (%r vs %r)" % (s2n.get("tier"), public_obj.tier))
    alpha_seq = _frac(s2n.get("alpha_seq", "0"))
    if not (0 < alpha_seq < 1):
        reasons.append("证书 alpha_seq 非法 %r" % (s2n.get("alpha_seq"),))
    thr = 1 / alpha_seq
    n_bits = int(table.n_bits)
    struct = SStructure.from_table(table)
    rep = replay_symbolic(public_obj, reasons)
    Q, D, E, P, vstates = rep["Q"], rep["D"], rep["E"], rep["P"], rep["states"]
    t_hist = len(Q) - 1
    checks["history_t"] = t_hist
    if int(s2n.get("t", -1)) != t_hist:
        reasons.append("证书声明的 t 与公开历史长度不符")

    # 域
    mode = str(cert.get("domain_mode"))
    if mode == "ALL_STATES":
        domain = SDomain((SCube(0, 0),), n_bits, [], "ALL_STATES")
    elif mode == "CONSISTENT_SET":
        domain = symbolic_domain(table, query_id)
    else:
        reasons.append("未知 domain_mode %r" % (mode,))
        domain = SDomain((), n_bits, [], "EMPTY")
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if not _declared_matches_domain(declared_allowed, domain):
        reasons.append("域名不符：声明 %d / 重算 %d" % (len(declared_allowed), domain.count))
    if domain.count == 0:
        reasons.append("空域：空集不得发证书")
    domain_states = domain.union.states()

    regions = list(cert.get("regions", []))
    checks["regions"] = len(regions)
    if not regions:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": reasons + ["证书无区域"], "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    ids = [r.get("region_id") for r in regions]
    if len(set(ids)) != len(ids):
        reasons.append("区域 id 重复")
    by_id = {r["region_id"]: r for r in regions}
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": reasons + ["根区域不唯一 (%d)" % len(roots)],
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append("区域 %r 的父区域不存在" % (r.get("region_id"),))
            kids.setdefault(p, []).append(r)
    st_of = {r["region_id"]: _states_of(r, n_bits, reasons, fmt="r") for r in regions}
    if st_of[roots[0]["region_id"]] != domain_states:
        reasons.append("根区域 != 域（%d vs %d）"
                       % (len(st_of[roots[0]["region_id"]]), domain.count))
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = st_of[r["region_id"]]
        if r.get("is_terminal"):
            if ch:
                reasons.append("终末区域 %r 仍有子区域" % (r.get("region_id"),))
            if not st:
                reasons.append("终末区域 %r 为空" % (r.get("region_id"),))
            continue
        if not ch:
            reasons.append("内部区域 %r 无子区域" % (r.get("region_id"),))
            continue
        union: List[int] = []
        for c in ch:
            union.extend(st_of[c["region_id"]])
        if len(union) != len(set(union)):
            reasons.append("区域 %r 子区域重叠" % (r.get("region_id"),))
        if sorted(set(union)) != st:
            reasons.append("区域 %r 子区域并集 != 父区域" % (r.get("region_id"),))

    # 逐节点：剔除节点重算界；存活节点重算外包与见证
    idx_of = {s: i for i, s in enumerate(vstates)}
    audit_states = 0
    n_excl = 0
    n_surv = 0
    claims: set = set()
    witness_holes: List[str] = []
    labels = _label_array(table)
    for r in regions:
        if not r.get("is_terminal"):
            continue
        rid = r.get("region_id")
        st = st_of[rid]
        excluded = bool(r.get("excluded"))
        if excluded:
            n_excl += 1
            stat = dict(r.get("statistical") or {})
            if not stat.get("excluded"):
                reasons.append("区域 %r 标了剔除但 statistical 段非剔除" % (rid,))
                continue
            if r.get("label_claim") is not None:
                reasons.append("剔除区域 %r 不得同时标标签" % (rid,))
                continue
            u = int(stat.get("u", 0))
            if not (1 <= u <= t_hist):
                reasons.append("区域 %r 的 u=%r 越出历史 (1..%d)" % (rid, u, t_hist))
                continue
            statistic = str(stat.get("statistic"))
            if statistic not in (STAT_JOINT, STAT_STEPWISE):
                reasons.append("区域 %r 的统计量 %r 非法" % (rid, statistic))
            if str(stat.get("bound_rule")) != BOUND_RULE:
                reasons.append("区域 %r 的界规则 %r 非法" % (rid, stat.get("bound_rule")))
            d_vals = [D[u][idx_of[s]] for s in st if s in idx_of]
            if len(d_vals) != len(st):
                reasons.append("区域 %r 含公开历史状态集外状态" % (rid,))
                continue
            sup_d = max(d_vals)
            if statistic == STAT_STEPWISE:
                step = Fraction(1, 1)
                for v in range(1, u + 1):
                    step *= max(P[v - 1][idx_of[s]] for s in st)
                target = step
            else:
                target = sup_d
            q = Q[u]
            if not _is_inf(stat.get("sup_D_u")):
                declared = _frac(stat.get("sup_D_u"))
                if declared != target:
                    reasons.append("区域 %r 声明的 sup_D_u 与重算（%s）不等"
                                   % (rid, statistic))
                for s, dv in zip(st, d_vals):
                    if dv > declared:
                        reasons.append("区域 %r 的界不是上界：状态 %d 的 D_u=%s > %s"
                                       % (rid, s, dv, declared))
                        break
                if statistic == STAT_STEPWISE and sup_d > declared:
                    reasons.append("区域 %r 用联合上确界去界逐步统计量：联合 sup=%s > 声明 %s"
                                   % (rid, sup_d, declared))
            if not _is_inf(stat.get("Q_u")) and _frac(stat.get("Q_u")) != q:
                reasons.append("区域 %r 声明的 Q_u 与重算不等" % (rid,))
            survivors = []
            for s in st:
                i = idx_of[s]
                emax = _max_e([E[v][i] for v in range(1, t_hist + 1)])
                if not (_is_inf(emax) or _frac(emax) >= thr):
                    survivors.append(s)
            if survivors:
                reasons.append("剔除区域 %r 仍含未被统计排除的状态 %s"
                               % (rid, survivors[:8]))
            if target != 0 and q / target < thr:
                reasons.append("区域 %r 的 Q_u/界 未达 1/alpha（重算口径 %s）" % (rid, statistic))
            declared_emax = dict(stat.get("per_state_emax") or {})
            for s in st:
                if str(s) not in declared_emax:
                    reasons.append("区域 %r 缺状态 %d 的 per_state_emax" % (rid, s))
                    break
            audit_states += len(st)
            continue
        n_surv += 1
        claim = r.get("label_claim")
        if claim not in ("0", "1"):
            reasons.append("终末区域 %r 标签非法 %r（既非剔除也非 0/1）" % (rid, claim))
            continue
        claims.add(claim)
        if not st:
            reasons.append("终末区域 %r 为空" % (rid,))
            continue
        if not set(st) <= set(domain_states):
            reasons.append("终末区域 %r 含域外状态" % (rid,))
        if getattr(table, "derivation", None) is None:
            reasons.append("表未提供派生：无法重算外包")
            continue
        cubes = _cube_list(r, n_bits, reasons, fmt="r")
        env = envelope_of_cubes(struct, cubes, n_bits, len(st))
        audit_states += len(st)
        declared_env = r.get("envelope_counts")
        if declared_env is not None and declared_env != env.element_counts():
            reasons.append("区域 %r 声明外包计数与重算不等" % (rid,))
        lab = direction(env)
        if lab == MONOTONICITY_DEFECT:
            reasons.append("区域 %r 外包单调性破裂" % (rid,))
            continue
        want = ALL_CONNECTED if claim == "1" else ALL_DISCONNECTED
        if lab != want:
            reasons.append("区域 %r 声明 %s 但重算 %s" % (rid, claim, lab))
            _audit_states(table, query_id, rid, st, claim, reasons)
        else:
            first_bad = None
            if labels is None:
                first_bad = _first_label_bad_by_derivation(table, query_id, st, claim)
            else:
                for s in st:
                    if int(labels[int(s)]) != int(claim):
                        first_bad = s
                        break
            if first_bad is not None:
                reasons.append("区域 %r 状态 %d 派生与标签不忠实" % (rid, first_bad))
        w = r.get("witness")
        if not w or "state_id" not in w:
            reasons.append("区域 %r 无见证" % (rid,))
        else:
            sid = int(w["state_id"])
            if sid not in set(st):
                reasons.append("区域 %r 见证 %d 不在区域内" % (rid, sid))
            elif int(w.get("label", -1)) != int(claim):
                reasons.append("区域 %r 见证标签与区域标签不符" % (rid,))
            else:
                rep_w = reverify_witness(table.derivation(query_id, sid),
                                         int(claim), w.get("path"))
                if not rep_w["ok"]:
                    reasons.append("区域 %r 见证不可重放 %r" % (rid, rep_w.get("reason")))
            if sid in idx_of:
                i = idx_of[sid]
                emax_w = _max_e([E[v][i] for v in range(1, t_hist + 1)])
                if not _is_inf(emax_w) and _frac(emax_w) >= thr:
                    witness_holes.append(rid)
    surv_states: List[int] = []
    for r in regions:
        if r.get("is_terminal") and not r.get("excluded"):
            surv_states.extend(st_of[r["region_id"]])
    all_terminal: List[int] = []
    for r in regions:
        if r.get("is_terminal"):
            all_terminal.extend(st_of[r["region_id"]])
    if len(all_terminal) != len(set(all_terminal)):
        reasons.append("终末区域划分不互斥")
    if sorted(set(all_terminal)) != sorted(domain_states):
        missing = sorted(set(domain_states) - set(all_terminal))
        extra = sorted(set(all_terminal) - set(domain_states))
        reasons.append("终末区域覆盖 != 域 (缺 %d / 多 %d)" % (len(missing), len(extra)))
    if len(claims) > 1:
        reasons.append("存活终末区域标签不一致 %s：混合查询不得发单标签证书" % (sorted(claims),))
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    else:
        if int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
            reasons.append("证书顶层见证标签与 label_claim 不符")
        sid = int(top["state_id"])
        if sid in idx_of:
            i = idx_of[sid]
            emax_t = _max_e([E[v][i] for v in range(1, t_hist + 1)])
            if not _is_inf(emax_t) and _frac(emax_t) >= thr:
                reasons.append("证书顶层见证状态已被统计排除")
    possible = []
    for s in domain_states:
        if s not in idx_of:
            continue
        i = idx_of[s]
        emax = _max_e([E[v][i] for v in range(1, t_hist + 1)])
        if not (_is_inf(emax) or _frac(emax) >= thr):
            possible.append(s)
    if not possible:
        reasons.append("统计可行集为空（S_t=∅）：空集不得发证书")
    if int(s2n.get("n_statistical_exclusions", -1)) != n_excl:
        reasons.append("证书声明的剔除节点数与重算不符")
    if int(s2n.get("n_possible_states", -1)) != len(possible):
        reasons.append("证书声明的可行状态数 %r 与重算 %d 不符"
                       % (s2n.get("n_possible_states"), len(possible)))
    if int(s2n.get("n_surviving_states", -1)) != len(set(surv_states)):
        reasons.append("证书声明的存活区域状态数与重算不符")
    if bool(cert.get("label_claim")) is False and cert.get("label_claim") not in ("0", "1"):
        reasons.append("证书 label_claim 非法 %r" % (cert.get("label_claim"),))
    if claims and str(cert.get("label_claim")) not in claims:
        reasons.append("证书 label_claim 与存活区域标签不符")
    return {"schema": SCHEMA_STAT_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION,
            "checks": {"history_t": t_hist, "domain": domain.count, "regions": len(regions),
                       "excluded_terminals": n_excl, "surviving_terminals": n_surv,
                       "surviving_states": len(set(surv_states)),
                       "possible_states": len(possible),
                       "regions_with_excluded_witness": len(witness_holes),
                       "audit_states": audit_states, "claims": sorted(claims),
                       "alpha_seq": str(alpha_seq), "threshold": str(thr)},
            "verify_seconds": round(time.perf_counter() - t0, 6),
            "s2n": {"reused_s2_verifier": False, "independent_replay": True,
                    "calls_production_solver": False}}


__all__ = [
    "SCHEMA_VERIFY", "SCHEMA_STAT_VERIFY", "VERIFIER_VERSION", "SymbolicError",
    "SymbolicTooComplex", "SCube", "SUnion", "SStructure", "SDomain", "SEnvelope",
    "cube_disjoint", "cube_subset", "cube_intersect", "cube_subtract",
    "union_count", "union_contains", "union_covers", "union_equal", "union_overlap",
    "all_states_domain", "symbolic_domain", "envelope_of_cubes", "direction",
    "reachable", "derivation_connected", "reverify_witness",
    "verify_certificate_symbolic", "verify_statistical_symbolic",
    "replay_symbolic", "history_sha12_symbolic",
]
