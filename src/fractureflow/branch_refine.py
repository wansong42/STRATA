# -*- coding: utf-8 -*-
"""S2-B · 跨阈值几何的分支细化与终止性（精确有理区间算术 + 阈值对齐三分裂）。

只承诺**声明几何族**内的行为（同 S2-N0 的 B 层探针口径）：
- 区间/外包全部用 ``fractions.Fraction`` 精确计算，声明族的端点是十进制字面值
  （``Fraction(str(float(x)))``），不用浮点做比较；
- must/may 由区间界**推出**，不用采样代替；采样只作矛盾发现与见证搜索（见证本身
  逐点精确重算，见 ``point_label``）；
- 跨阈值族若能终止，发 ``s2.certificate/v1`` 皮的分区证书；若不能终止，给出
  「未覆盖测度上界 + 前沿件数 + 节点数」，并**不**把未终止写成物理不可识别。

用法（不消费 R/W/G 号位；不改 ``certificate_solver.py`` / ``identification.py``）：:

    spec = GeometrySpec.from_box_family(cs.BoxFamily(...))
    art = certify_or_bound(None, spec, budget=512, split_rule="threshold")
    rep = verify_bound(art, spec)

写面：仅本文件（新建）。``certificate_solver`` / ``identification`` 只读复用。
"""
from __future__ import annotations

import hashlib
import itertools
import json
from functools import cmp_to_key
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import fractureflow.identification as ident
from fractureflow import certificate_solver as cs

SCHEMA_CERT = cs.SCHEMA_CERT                 # s2.certificate/v1（复用皮）
SCHEMA_EXT = "s2b.geometry/v1"
VERIFIER_VERSION = "s2b.verifier/1"
LINE = "S2B_branch_refine"
SELECTION_PROVENANCE = "DECLARED_GEOMETRY_ONLY"

#: 声明的分裂规则（禁裸称最优；四种全部逐族报告）
SPLIT_RULES: Tuple[str, ...] = ("box_widest", "gap_widest", "threshold", "hybrid")

#: 声明容差（相对族测度）。未覆盖测度 ≤ 该值才算「可验证界」；否则记 UNRESOLVED。
REL_TOL_MEASURE = Fraction(1, 10 ** 9)
#: 无理阈值的有理夹逼宽度（十进制位数）。仅 hybrid 规则使用。
BRACKET_DIGITS = 18

WATERMARK = {
    "research_line": LINE,
    "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
    "noise": "N0",
    "dictionary": "COMPLETE_KNOWN",
    "claim_scope": "DECLARED_BOUNDED_GEOMETRY",
    "real_domain_claims": False,
    "physical_unidentifiability_claims": False,
    "external_commitment_text": False,
}


class BranchRefineError(RuntimeError):
    """仪器/契约缺陷：一律停，不出读数。"""


# ---------------------------------------------------------------------------
# 精确标量与线性方程
# ---------------------------------------------------------------------------

def Q(x) -> Fraction:
    """十进制字面值 → 精确有理数（1.8 得 9/5，不是 1.7999999999999998）。"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(str(float(x)))


def _dot(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _solve_linear(rows: Sequence[Sequence[Fraction]],
                  rhs: Sequence[Fraction]) -> Optional[Tuple[Fraction, ...]]:
    """n×n 精确高斯消元；奇异返回 None（组合退化，不是缺陷）。"""
    n = len(rows)
    if n == 0:
        return ()
    m = [list(rows[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if m[r][col] != 0:
                piv = r
                break
        if piv is None:
            return None
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [m[r][c] - f * m[col][c] for c in range(n + 1)]
    return tuple(m[i][n] for i in range(n))


def _isqrt_exact(q: Fraction) -> Optional[Fraction]:
    """若 q 是有理数的完全平方，返回其非负平方根；否则 None（精确判定）。"""
    if q < 0:
        return None
    num, den = q.numerator, q.denominator
    rn, rd = math.isqrt(num), math.isqrt(den)
    if rn * rn == num and rd * rd == den and Fraction(rn, rd) ** 2 == q:
        return Fraction(rn, rd)
    return None


def _bracket_sqrt(q: Fraction, digits: int = BRACKET_DIGITS) -> Tuple[Fraction, Fraction]:
    """lo < sqrt(q) < hi，宽度 ≤ 10**-digits；两端都精确有理，且当场断言 lo² < q < hi²。"""
    if q <= 0:
        raise BranchRefineError("q ≤ 0 无实根：调用方应先判 q>0")
    scale = 10 ** int(digits)
    scaled = q * scale * scale
    root = math.isqrt(scaled.numerator // scaled.denominator)
    lo = Fraction(root, scale)
    hi = Fraction(root + 1, scale)
    while not (lo * lo < q < hi * hi):      # 整除下取整的边界兜底
        root += 1
        hi = Fraction(root + 1, scale)
        lo = Fraction(root, scale)
        if root > 2 * scale:
            raise BranchRefineError("夹逼失败：q 过大")
    return lo, hi


# ---------------------------------------------------------------------------
# 域：盒 ∩ 有限半空间（含严格/非严格），预设在场位
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HalfSpace:
    """``coeffs·x ≤ rhs``；``strict=True`` 表示 ``coeffs·x < rhs``（开半空间）。"""
    coeffs: Tuple[Fraction, ...]
    rhs: Fraction
    strict: bool = False
    tag: str = ""

    def to_dict(self) -> dict:
        return {"coeffs": [str(c) for c in self.coeffs], "rhs": str(self.rhs),
                "strict": bool(self.strict), "tag": self.tag}

    @staticmethod
    def from_dict(d: Mapping) -> "HalfSpace":
        return HalfSpace(tuple(Fraction(str(c)) for c in d["coeffs"]),
                         Fraction(str(d["rhs"])), bool(d.get("strict", False)),
                         str(d.get("tag", "")))


@dataclass(frozen=True)
class Domain:
    """凸域 = 盒 ∩ 半空间；``fixed[i] ∈ {-1,0,1}`` 为在场位（-1 = 自由）。"""
    n: int
    box: Tuple[Tuple[Fraction, Fraction], ...]
    halfspaces: Tuple[HalfSpace, ...] = ()
    fixed: Tuple[int, ...] = ()

    def __post_init__(self):
        if len(self.box) != self.n:
            raise BranchRefineError("盒维度与 n 不符")
        if not self.fixed:
            object.__setattr__(self, "fixed", tuple([-1] * self.n))
        if len(self.fixed) != self.n:
            raise BranchRefineError("在场位维度与 n 不符")

    # -- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"n": int(self.n),
                "box": [[str(lo), str(hi)] for (lo, hi) in self.box],
                "halfspaces": [h.to_dict() for h in self.halfspaces],
                "fixed": [int(v) for v in self.fixed]}

    @staticmethod
    def from_dict(d: Mapping) -> "Domain":
        return Domain(int(d["n"]),
                      tuple((Fraction(str(a)), Fraction(str(b))) for a, b in d["box"]),
                      tuple(HalfSpace.from_dict(h) for h in d.get("halfspaces", ())),
                      tuple(int(v) for v in d.get("fixed", ())))

    def key(self) -> tuple:
        return (self.n, self.box, tuple((h.coeffs, h.rhs, h.strict) for h in self.halfspaces),
                self.fixed)

    def volume_box(self) -> Fraction:
        v = Fraction(1)
        for lo, hi in self.box:
            v *= (hi - lo)
        return v

    def n_free_presence(self, free: Sequence[int]) -> int:
        """尚未落定的自由在场位数（只数声明 ``free_presence`` 里那些盘）。"""
        return sum(1 for i in free if self.fixed[i] < 0)

    def with_halfspace(self, hs: HalfSpace) -> "Domain":
        return Domain(self.n, self.box, self.halfspaces + (hs,), self.fixed)

    def with_fixed(self, i: int, v: int) -> "Domain":
        f = list(self.fixed)
        f[i] = int(v)
        return Domain(self.n, self.box, self.halfspaces, tuple(f))


@dataclass(frozen=True)
class GeometrySpec:
    """声明几何族：共面盘（y/z 与半径固定）、盘心 x 在声明区间、可选在场位自由。"""
    family_id: str
    disks: Tuple[Tuple[str, Fraction, Fraction, Fraction, Fraction, Fraction], ...]
    A: Tuple[Fraction, Fraction, Fraction]
    B: Tuple[Fraction, Fraction, Fraction]
    r_attach: Fraction
    free_presence: Tuple[int, ...] = ()
    note: str = ""

    @staticmethod
    def from_box_family(fam: "cs.BoxFamily") -> "GeometrySpec":
        return GeometrySpec(
            fam.family_id,
            tuple((d[0], Q(d[1]), Q(d[2]), Q(d[3]), Q(d[4]), Q(d[5])) for d in fam.disks),
            tuple(Q(v) for v in fam.A), tuple(Q(v) for v in fam.B), Q(fam.r_attach),
            tuple(int(i) for i in fam.free_presence), fam.note)

    @staticmethod
    def from_dict(d: Mapping) -> "GeometrySpec":
        if "disks" in d and d["disks"] and isinstance(d["disks"][0], Mapping):
            disks = tuple((x["id"], Fraction(str(x["x_lo"])), Fraction(str(x["x_hi"])),
                           Fraction(str(x["y"])), Fraction(str(x["z"])),
                           Fraction(str(x["r"]))) for x in d["disks"])
        else:
            disks = tuple((x[0], Fraction(str(x[1])), Fraction(str(x[2])),
                           Fraction(str(x[3])), Fraction(str(x[4])),
                           Fraction(str(x[5]))) for x in d["disks"])
        return GeometrySpec(str(d["family_id"]), disks,
                            tuple(Fraction(str(v)) for v in d["A"]),
                            tuple(Fraction(str(v)) for v in d["B"]),
                            Fraction(str(d["r_attach"])),
                            tuple(int(i) for i in d.get("free_presence", ())),
                            str(d.get("note", "")))

    def to_dict(self) -> dict:
        return {"family_id": self.family_id,
                "disks": [{"id": d[0], "x_lo": float(d[1]), "x_hi": float(d[2]),
                           "y": float(d[3]), "z": float(d[4]), "r": float(d[5])}
                          for d in self.disks],
                "A": [float(v) for v in self.A], "B": [float(v) for v in self.B],
                "r_attach": float(self.r_attach),
                "free_presence": [int(i) for i in self.free_presence], "note": self.note}

    def to_box_family(self) -> "cs.BoxFamily":
        return cs.BoxFamily(self.family_id,
                            tuple((d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]),
                                   float(d[5])) for d in self.disks),
                            tuple(float(v) for v in self.A), tuple(float(v) for v in self.B),
                            float(self.r_attach), tuple(int(i) for i in self.free_presence),
                            self.note)

    @property
    def n(self) -> int:
        return len(self.disks)

    def root_domain(self) -> Domain:
        return Domain(self.n, tuple((d[1], d[2]) for d in self.disks))

    def spec_sha12(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def family_volume(self) -> Fraction:
        v = Fraction(1)
        for d in self.disks:
            v *= (d[2] - d[1])
        return v

    def family_measure(self) -> Fraction:
        return self.family_volume() * (2 ** len(self.free_presence))


# ---------------------------------------------------------------------------
# 顶点 / 测度（全部精确有理；n ≤ 3）
# ---------------------------------------------------------------------------

_VERT_CACHE: Dict[tuple, Tuple[Tuple[Fraction, ...], ...]] = {}
_CACHE_LIMIT = 20000


def _planes(dom: Domain) -> List[Tuple[Tuple[Fraction, ...], Fraction]]:
    out = []
    for i, (lo, hi) in enumerate(dom.box):
        e = [Fraction(0)] * dom.n
        e[i] = Fraction(1)
        out.append((tuple(e), lo))
        out.append((tuple(e), hi))
    for h in dom.halfspaces:
        out.append((h.coeffs, h.rhs))
    return out


def _feasible(dom: Domain, x: Sequence[Fraction]) -> bool:
    for i, (lo, hi) in enumerate(dom.box):
        if x[i] < lo or x[i] > hi:
            return False
    for h in dom.halfspaces:
        if _dot(h.coeffs, x) > h.rhs:     # 顶点枚举按闭包做（开半空间的边界仍属闭包）
            return False
    return True


def vertices(dom: Domain) -> Tuple[Tuple[Fraction, ...], ...]:
    """闭包顶点集（组合枚举 + 精确求解）；空域返回 ()。"""
    k = dom.key()
    if k in _VERT_CACHE:
        return _VERT_CACHE[k]
    planes = _planes(dom)
    n = dom.n
    found = set()
    for comb in itertools.combinations(range(len(planes)), n):
        sol = _solve_linear([planes[c][0] for c in comb], [planes[c][1] for c in comb])
        if sol is None:
            continue
        if _feasible(dom, sol):
            found.add(tuple(sol))
    out = tuple(sorted(found))
    if len(_VERT_CACHE) < _CACHE_LIMIT:
        _VERT_CACHE[k] = out
    return out


def _face_vertices(dom: Domain, plane: Tuple[Tuple[Fraction, ...], Fraction]
                   ) -> List[Tuple[Fraction, ...]]:
    coeffs, rhs = plane
    return [v for v in vertices(dom) if _dot(coeffs, v) == rhs]


def _angle_order(pts2: Sequence[Tuple[Fraction, Fraction]]) -> List[int]:
    """凸多边形的环序：精确角度排序（先半平面、再叉积），**不经过浮点**。

    细长盒（宽度 1e-27 量级）用 float 角度排序会把不同顶点压成同一点，实测会让
    面积算成 0 —— 所以这里全程 Fraction 比较。
    """
    n = len(pts2)
    cx = sum(p[0] for p in pts2) / n
    cy = sum(p[1] for p in pts2) / n
    rel = [(p[0] - cx, p[1] - cy) for p in pts2]

    def half(v):
        x, y = v
        return 0 if (y > 0 or (y == 0 and x >= 0)) else 1

    def cmp(i, j):
        hi, hj = half(rel[i]), half(rel[j])
        if hi != hj:
            return -1 if hi < hj else 1
        cross = rel[i][0] * rel[j][1] - rel[i][1] * rel[j][0]
        if cross > 0:
            return -1
        if cross < 0:
            return 1
        di = rel[i][0] ** 2 + rel[i][1] ** 2
        dj = rel[j][0] ** 2 + rel[j][1] ** 2
        return -1 if di < dj else (0 if di == dj else 1)

    return sorted(range(n), key=cmp_to_key(cmp))


def _order_polygon(pts: Sequence[Tuple[Fraction, ...]]
                   ) -> Tuple[List[Tuple[Fraction, ...]], Optional[Tuple[Fraction, ...]]]:
    """凸多边形顶点环序（精确）；法向与面积同样用精确有理算。"""
    n = len(pts)
    if n < 3:
        return [], None
    dim = len(pts[0])
    if dim >= 3:
        a = [pts[1][i] - pts[0][i] for i in range(dim)]
        b = [pts[2][i] - pts[0][i] for i in range(dim)]
        nrm = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
               a[0] * b[1] - a[1] * b[0]]
        drop = max(range(3), key=lambda i: abs(nrm[i]))
        keep = [i for i in range(3) if i != drop]
    else:
        keep = [0, 1]
    pts2 = [(p[keep[0]], p[keep[1]]) for p in pts]
    order = _angle_order(pts2)
    return [pts[i] for i in order], None


def volume(dom: Domain) -> Fraction:
    """闭包测度（开/闭半空间同测度）；n=1 长、n=2 面积、n=3 体积。"""
    v = vertices(dom)
    n = dom.n
    if n == 1:
        if not v:
            return Fraction(0)
        return max(p[0] for p in v) - min(p[0] for p in v)
    if n == 2:
        if len(v) < 3:
            return Fraction(0)
        ordered, _ = _order_polygon(v)
        acc = Fraction(0)
        for i in range(len(ordered)):
            x1, y1 = ordered[i]
            x2, y2 = ordered[(i + 1) % len(ordered)]
            acc += x1 * y2 - x2 * y1
        return abs(acc) / 2
    if n == 3:
        if len(v) < 4:
            return Fraction(0)
        tot = Fraction(0)
        for plane in _planes(dom):
            pts = _face_vertices(dom, plane)
            if len(pts) < 3:
                continue
            ordered, _ = _order_polygon(pts)
            if len(ordered) < 3:
                continue
            area_vec = [Fraction(0), Fraction(0), Fraction(0)]
            for i in range(len(ordered)):
                p = ordered[i]
                q = ordered[(i + 1) % len(ordered)]
                area_vec[0] += p[1] * q[2] - p[2] * q[1]
                area_vec[1] += p[2] * q[0] - p[0] * q[2]
                area_vec[2] += p[0] * q[1] - p[1] * q[0]
            area_vec = [a / 2 for a in area_vec]
            if all(a == 0 for a in area_vec):
                continue
            cent = tuple(sum(p[i] for p in ordered) / len(ordered) for i in range(3))
            # 外法向定向：面积向量指向内部则翻转
            inner = tuple(sum(p[i] for p in v) / len(v) for i in range(3))
            if _dot(area_vec, tuple(cent[i] - inner[i] for i in range(3))) < 0:
                area_vec = [-a for a in area_vec]
            tot += _dot(area_vec, cent)
        return abs(tot) / 3
    raise BranchRefineError(f"维度 {n} 超出本实现的测度上限（声明 n ≤ 3）")


# ---------------------------------------------------------------------------
# 线性型上下界：三路（顶点 / 傅里叶–莫茨金 / 逐元素）
# ---------------------------------------------------------------------------

def form_bounds(dom: Domain, coeffs: Sequence[Fraction]) -> Tuple[Fraction, Fraction]:
    """顶点路：线性型在凸多面体上的极值必在顶点取到。"""
    v = vertices(dom)
    if not v:
        raise BranchRefineError("空域无界可算（空区域禁发读数）")
    vals = [_dot(coeffs, p) for p in v]
    return min(vals), max(vals)


def form_bounds_fm(dom: Domain, coeffs: Sequence[Fraction]
                   ) -> Tuple[Optional[Fraction], Optional[Fraction]]:
    """集合路：傅里叶–莫茨金消元求投影（与顶点路不共用实现）。"""
    n = dom.n
    cons: List[Tuple[List[Fraction], Fraction]] = []

    def add(c, r):
        cons.append(([Fraction(x) for x in c], Fraction(r)))

    for i, (lo, hi) in enumerate(dom.box):
        c = [Fraction(0)] * (n + 1)
        c[i] = Fraction(1)
        add(c, hi)
        c = [Fraction(0)] * (n + 1)
        c[i] = Fraction(-1)
        add(c, -lo)
    for h in dom.halfspaces:
        add(list(h.coeffs) + [Fraction(0)], h.rhs)
    add([-Fraction(c) for c in coeffs] + [Fraction(1)], Fraction(0))   # w - c·x ≤ 0
    add([Fraction(c) for c in coeffs] + [Fraction(-1)], Fraction(0))   # -w + c·x ≤ 0

    def norm(cons_in):
        seen = {}
        for c, r in cons_in:
            if all(x == 0 for x in c):
                if r < 0:
                    return None
                continue
            k = tuple(c)
            if k not in seen or r < seen[k]:
                seen[k] = r
        return [(list(k), v) for k, v in seen.items()]

    cons = norm(cons)
    if cons is None:
        return None, None
    for k in range(n):
        pos, neg, zer = [], [], []
        for c, r in cons:
            if c[k] > 0:
                pos.append((c, r))
            elif c[k] < 0:
                neg.append((c, r))
            else:
                zer.append((c, r))
        new = list(zer)
        for cp, bp in pos:
            for cn, bn in neg:
                ka, kb = -cn[k], cp[k]
                c = [ka * cp[j] + kb * cn[j] for j in range(n + 1)]
                c[k] = Fraction(0)
                new.append((c, ka * bp + kb * bn))
        cons = norm(new)
        if cons is None:
            return None, None
    lo = hi = None
    for c, r in cons:
        if c[n] > 0:
            b = r / c[n]
            hi = b if hi is None else min(hi, b)
        elif c[n] < 0:
            b = r / c[n]
            lo = b if lo is None else max(lo, b)
        else:
            if r < 0:
                return None, None
    if lo is not None and hi is not None and lo > hi:
        return None, None
    return lo, hi


def form_bounds_elementwise(dom: Domain, coeffs: Sequence[Fraction]) -> dict:
    """逐元素路：枚举顶点与盒角点，逐点精确取值（凸函数极值在顶点取到的审计）。"""
    v = vertices(dom)
    if not v:
        raise BranchRefineError("空域无界可算")
    corners = []
    for combo in itertools.product(*[(lo, hi) for lo, hi in dom.box]):
        if _feasible(dom, combo):
            corners.append(tuple(combo))
    vals_v = [_dot(coeffs, p) for p in v]
    vals_c = [_dot(coeffs, p) for p in corners] if corners else []
    lo, hi = min(vals_v), max(vals_v)
    clamped_lo = Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return {"vertex_max": hi, "vertex_min": lo, "n_vertices": len(v),
            "n_corners_feasible": len(corners),
            "corner_max": (max(vals_c) if vals_c else None),
            "corner_min": (min(vals_c) if vals_c else None),
            "clamped_min_abs": clamped_lo}


# ---------------------------------------------------------------------------
# 外包（must/may）：区间界推出，严格接触单独处理
# ---------------------------------------------------------------------------

def _face_excluded(dom: Domain, coeffs: Sequence[Fraction], val: Fraction) -> bool:
    """闭包面 {coeffs·x = val} 是否被某条**严格**半空间整片排除。"""
    face = dom.with_halfspace(HalfSpace(tuple(coeffs), val, False, "face_le"))
    face = face.with_halfspace(HalfSpace(tuple(-c for c in coeffs), -val, False, "face_ge"))
    fv = vertices(face)
    if not fv:
        return True
    for h in dom.halfspaces:
        if not h.strict:
            continue
        vals = [_dot(h.coeffs, p) for p in fv]
        if min(vals) == h.rhs and max(vals) == h.rhs:
            return True
    return False


def _side_below_strict(dom: Domain, coeffs: Sequence[Fraction], q: Fraction, side: str,
                       stats: Dict[str, int]) -> bool:
    """单侧严格判据：``hi`` 侧 ∀x: u < t；``lo`` 侧 ∀x: u > -t（t = sqrt(q)）。"""
    lo, hi = form_bounds(dom, coeffs)
    stats["interval_ops"] += 2
    if side == "hi":
        if hi * hi > q:
            return False
        if hi * hi == q and not _face_excluded(dom, coeffs, hi):
            return False
        return True
    if lo * lo > q:
        return False
    if lo * lo == q and not _face_excluded(dom, coeffs, lo):
        return False
    return True


def _below_strict(dom: Domain, coeffs: Sequence[Fraction], q: Fraction,
                  stats: Dict[str, int]) -> bool:
    """是否 ∀x∈dom（开域，严格半空间边界不在内）: (coeffs·x)² < q。"""
    return (_side_below_strict(dom, coeffs, q, "hi", stats)
            and _side_below_strict(dom, coeffs, q, "lo", stats))


def _may_strict(dom: Domain, coeffs: Sequence[Fraction], q: Fraction,
                stats: Dict[str, int]) -> bool:
    """是否 ∃x∈dom: (coeffs·x)² < q。"""
    lo, hi = form_bounds(dom, coeffs)
    stats["interval_ops"] += 2
    if lo <= 0 <= hi:
        return q > 0
    return min(abs(lo), abs(hi)) ** 2 < q


@dataclass
class EnvelopeInfo:
    envelope: "ident.Envelope"
    label: str
    pairs: List[dict]
    attach: List[dict]
    three_route: dict

    def counts(self) -> dict:
        return self.envelope.element_counts()


def present_sets(spec: GeometrySpec, dom: Domain) -> Tuple[set, set]:
    must = {i for i in range(spec.n) if i not in spec.free_presence}
    may = set(range(spec.n))
    for i, v in enumerate(dom.fixed):
        if v == 1:
            must.add(i)
        elif v == 0:
            may.discard(i)
    must &= may
    return must, may


def envelope_of_domain(spec: GeometrySpec, dom: Domain,
                       stats: Optional[Dict[str, int]] = None) -> EnvelopeInfo:
    """区间算术外包：must/may 全部由精确界推出（接触严格 <，附着 ≤，同 S2-N0）。"""
    st = stats if stats is not None else {"interval_ops": 0, "envelope_calls": 0}
    st["envelope_calls"] += 1
    must_nodes, may_nodes = present_sets(spec, dom)
    n = spec.n
    must_e, may_e = set(), set()
    pairs = []
    three = {"checked": 0, "mismatch": 0, "routes": ["vertex", "fourier_motzkin",
                                                     "elementwise"]}
    for i in range(n):
        for j in range(i + 1, n):
            coeffs = tuple((Fraction(1) if k == i else (Fraction(-1) if k == j else Fraction(0)))
                           for k in range(n))
            dy = spec.disks[i][3] - spec.disks[j][3]
            dz = spec.disks[i][4] - spec.disks[j][4]
            base = dy * dy + dz * dz
            thr = spec.disks[i][5] + spec.disks[j][5]
            q = thr * thr - base
            lo, hi = form_bounds(dom, coeffs)
            st["interval_ops"] += 2
            must = False
            may = False
            if q > 0:
                must = _below_strict(dom, coeffs, q, st)
                may = _may_strict(dom, coeffs, q, st)
            if must:
                must_e.add((i, j))
            if may:
                may_e.add((i, j))
            # 三路对账（只在区间路给出的界上做；不做证明，只发现矛盾）
            fm_lo, fm_hi = form_bounds_fm(dom, coeffs)
            el = form_bounds_elementwise(dom, coeffs)
            three["checked"] += 1
            inf2 = Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi)) ** 2
            ok = (fm_lo == lo and fm_hi == hi
                  and el["vertex_max"] == hi and el["vertex_min"] == lo
                  and (el["corner_max"] is None or el["corner_max"] <= hi)
                  and (el["corner_min"] is None or el["corner_min"] >= lo)
                  and el["clamped_min_abs"] ** 2 == inf2)
            if not ok:
                three["mismatch"] += 1
            pairs.append({"pair": [i, j], "u_lo": str(lo), "u_hi": str(hi),
                          "q": str(q), "must_contact": must, "may_contact": may,
                          "routes": {"vertex": [str(lo), str(hi)],
                                     "fourier_motzkin": [str(fm_lo), str(fm_hi)],
                                     "elementwise": {"vertex_max": str(el["vertex_max"]),
                                                     "clamped_min_abs":
                                                         str(el["clamped_min_abs"])}},
                          "three_route_ok": bool(ok)})
    must_e = {(i, j) for (i, j) in must_e if i in must_nodes and j in must_nodes}
    may_e = {(i, j) for (i, j) in may_e if i in may_nodes and j in may_nodes}
    ma, ya, mb, yb = set(), set(), set(), set()
    attach = []
    for i in range(n):
        for P, mm, yy in ((spec.A, ma, ya), (spec.B, mb, yb)):
            coeffs = tuple(Fraction(1) if k == i else Fraction(0) for k in range(n))
            dy = P[1] - spec.disks[i][3]
            dz = P[2] - spec.disks[i][4]
            base = dy * dy + dz * dz
            thr = spec.disks[i][5] + spec.r_attach
            q = thr * thr - base
            x_lo, x_hi = form_bounds(dom, coeffs)
            st["interval_ops"] += 2
            lo, hi = x_lo - P[0], x_hi - P[0]          # u = x_i - P_x 的精确区间
            top = max(hi * hi, lo * lo)
            mid = Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi)) ** 2
            must = (q >= 0) and top <= q
            may = (q >= 0) and mid <= q
            if must:
                mm.add(i)
            if may:
                yy.add(i)
            attach.append({"disk": i, "endpoint": "A" if P is spec.A else "B",
                           "u_lo": str(lo), "u_hi": str(hi), "q": str(q),
                           "must": bool(must), "may": bool(may)})
    ma &= must_nodes
    mb &= must_nodes
    ya &= may_nodes
    yb &= may_nodes
    if not (must_e <= may_e and ma <= ya and mb <= yb):
        raise BranchRefineError("外包单调性破裂（must 真而 may 假）：仪器缺陷")
    env = ident.Envelope(0,
                         frozenset(must_nodes), frozenset(may_nodes),
                         frozenset(sorted(must_e)), frozenset(sorted(may_e)),
                         frozenset(ma), frozenset(ya), frozenset(mb), frozenset(yb),
                         bool(ma), bool(ya), bool(mb), bool(yb))
    try:
        lab_raw = ident.region_label(env)
    except ident.ContractError as exc:                     # noqa: F821
        raise BranchRefineError(f"region_label 拒绝本外包: {exc}") from exc
    lab = _CLAIM.get(lab_raw, lab_raw)
    lab2 = direction_rule(env)
    if lab2 != lab:
        raise BranchRefineError(f"两路方向规则不等：region_label={lab} 本模块={lab2}")
    three["direction_two_route_equal"] = True
    return EnvelopeInfo(envelope=env, label=lab, pairs=pairs, attach=attach,
                        three_route=three)


_CLAIM = {ident.RegionLabel.ALL_CONNECTED.value: "1",
          ident.RegionLabel.ALL_DISCONNECTED.value: "0"}


def direction_rule(env: "ident.Envelope") -> str:
    """本模块自带的 must/may 方向规则（与 region_label 同判据、另一份实现）。"""
    def connected(nodes, edges, side_a, side_b) -> bool:
        sa, sb = set(side_a) & set(nodes), set(side_b) & set(nodes)
        if not sa or not sb:
            return False
        adj: Dict = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        seen, queue = set(sa), list(sa)
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

    must = bool(env.must_attach_a and env.must_attach_b
                and connected(env.must_nodes, env.must_edges, env.must_a_side, env.must_b_side))
    may = bool(env.may_attach_a and env.may_attach_b
               and connected(env.may_nodes, env.may_edges, env.may_a_side, env.may_b_side))
    if must and not may:
        raise BranchRefineError("方向规则单调性破裂")
    if must:
        return "1"
    if not may:
        return "0"
    return ident.RegionLabel.UNRESOLVED.value


# ---------------------------------------------------------------------------
# 逐点标签（见证用；不是证明，见证逐点精确重算）
# ---------------------------------------------------------------------------

def point_label(spec: GeometrySpec, x: Sequence[Fraction],
                fixed: Optional[Sequence[int]] = None) -> int:
    n = spec.n
    fx = list(fixed) if fixed is not None else [-1] * n
    present = []
    for i in range(n):
        if i in spec.free_presence:
            if fx[i] == 0:
                continue
            if fx[i] == -1:
                present.append(i)     # 见证取「在场」补全（外向补全，标签不受影响时由调用方断言）
                continue
        present.append(i)
    pset = set(present)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if i not in pset or j not in pset:
                continue
            d2 = ((x[i] - x[j]) ** 2 + (spec.disks[i][3] - spec.disks[j][3]) ** 2
                  + (spec.disks[i][4] - spec.disks[j][4]) ** 2)
            thr = spec.disks[i][5] + spec.disks[j][5]
            if d2 < thr * thr:
                edges.add((i, j))
    sides = []
    for P in (spec.A, spec.B):
        s = set()
        for i in pset:
            d2 = ((P[0] - x[i]) ** 2 + (P[1] - spec.disks[i][3]) ** 2
                  + (P[2] - spec.disks[i][4]) ** 2)
            thr = spec.disks[i][5] + spec.r_attach
            if d2 <= thr * thr:
                s.add(i)
        sides.append(s)
    a_side, b_side = sides
    if not a_side or not b_side:
        return 0
    adj: Dict = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen, queue = set(a_side), list(a_side)
    head = 0
    while head < len(queue):
        z = queue[head]
        head += 1
        if z in b_side:
            return 1
        for y in adj.get(z, ()):
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return 0


def _in_domain(dom: Domain, x: Sequence[Fraction]) -> bool:
    for i, (lo, hi) in enumerate(dom.box):
        if x[i] < lo or x[i] > hi:
            return False
    for h in dom.halfspaces:
        v = _dot(h.coeffs, x)
        if v > h.rhs:
            return False
        if h.strict and v == h.rhs:
            return False
    return True


def find_witness(spec: GeometrySpec, dom: Domain, want_label: str,
                 limit: int = 256) -> Optional[dict]:
    """在域内找一个**精确重算**后标签等于 ``want_label`` 的点；找不到返回 None。"""
    v = vertices(dom)
    if not v:
        return None
    cands: List[Tuple[Fraction, ...]] = []
    cent = tuple(sum(p[i] for p in v) / len(v) for i in range(dom.n))
    cands.append(cent)
    for a in v:
        for b in v:
            cands.append(tuple((a[i] + b[i]) / 2 for i in range(dom.n)))
    # 决定性格点（不掺随机；不读任何隐藏真值）
    m = 7
    for combo in itertools.product(*[range(1, m) for _ in range(dom.n)]):
        cands.append(tuple(dom.box[i][0] + (dom.box[i][1] - dom.box[i][0]) * Fraction(combo[i], m)
                           for i in range(dom.n)))
    seen = set()
    for p in cands[:limit * 8]:
        if p in seen:
            continue
        seen.add(p)
        if not _in_domain(dom, p):
            continue
        lab = str(point_label(spec, p, dom.fixed))
        if lab == want_label:
            return {"x": [str(c) for c in p], "label": int(lab), "exact": True}
        if len(seen) >= limit:
            break
    return None


# ---------------------------------------------------------------------------
# 分裂：在场位 / 盒二分 / 阈值对齐三分裂
# ---------------------------------------------------------------------------

def _coeffs_pair(n: int, i: int, j: int) -> Tuple[Fraction, ...]:
    return tuple(Fraction(1) if k == i else (Fraction(-1) if k == j else Fraction(0))
                 for k in range(n))


def _open_empty(dom: Domain) -> bool:
    """开域（严格半空间边界剔除后）是否为空：任一条严格半空间在闭包上恒取等即空。"""
    if not vertices(dom):
        return True
    for h in dom.halfspaces:
        if not h.strict:
            continue
        vals = [_dot(h.coeffs, p) for p in vertices(dom)]
        # 闭包上该型的最小值都到了边界 ⇒ 没有任何点严格小于 rhs ⇒ 开域空。
        # 注意「某个极点触边」不算空（那时 min < rhs），这是 F_STRADDLE 关键的一刀。
        if min(vals) >= h.rhs:
            return True
    return False


def _ge_t(v: Fraction, q: Fraction, t: Optional[Fraction]) -> bool:
    """v >= sqrt(q)（精确；t 为有理根时直接用，否则按平方判，带符号保护）。"""
    if t is not None:
        return v >= t
    return v >= 0 and v * v >= q


def _lt_t(v: Fraction, q: Fraction, t: Optional[Fraction]) -> bool:
    if t is not None:
        return v < t
    return v < 0 or v * v < q


def _le_negt(v: Fraction, q: Fraction, t: Optional[Fraction]) -> bool:
    if t is not None:
        return v <= -t
    return v <= 0 and v * v >= q


def _gt_negt(v: Fraction, q: Fraction, t: Optional[Fraction]) -> bool:
    if t is not None:
        return v > -t
    return v > 0 or v * v < q


def _ambiguity(spec: GeometrySpec, dom: Domain) -> List[dict]:
    """当前域内「真正跨阈值」的接触侧（上下分开列），附精确阈值信息。

    跨阈值 = 该侧存在越界点（u ≥ t 或 u ≤ -t）**且**存在接触点（u < t 或 u > -t）；
    只满足一侧不叫跨阈值（那会做出「整块滑过去」的白分裂，实测会空转）。
    """
    out = []
    n = spec.n
    st = {"interval_ops": 0}
    for i in range(n):
        for j in range(i + 1, n):
            coeffs = _coeffs_pair(n, i, j)
            dy = spec.disks[i][3] - spec.disks[j][3]
            dz = spec.disks[i][4] - spec.disks[j][4]
            base = dy * dy + dz * dz
            thr = spec.disks[i][5] + spec.disks[j][5]
            q = thr * thr - base
            if q <= 0:
                continue
            lo, hi = form_bounds(dom, coeffs)
            root = _isqrt_exact(q)
            for side in ("hi", "lo"):
                if side == "hi":
                    if not (_ge_t(hi, q, root) and _lt_t(lo, q, root)):
                        continue
                else:
                    if not (_le_negt(lo, q, root) and _gt_negt(hi, q, root)):
                        continue
                if _side_below_strict(dom, coeffs, q, side, st):
                    continue          # 该侧已被严格规则排除 ⇒ 不构成歧义
                out.append({"kind": "contact", "pair": [i, j], "side": side,
                            "coeffs": coeffs, "q": q, "t": root,
                            "u_lo": lo, "u_hi": hi,
                            "gap2": (hi * hi - q) if side == "hi" else (lo * lo - q)})
    return out


def _make_children_threshold(spec: GeometrySpec, dom: Domain, amb: dict,
                             bracket: Optional[Tuple[Fraction, Fraction]] = None
                             ) -> List[Tuple[Domain, dict]]:
    """阈值三分裂：{s·u < t} / {s·u = t} / {s·u > t}（无理阈值退化为四裂 + 夹逼带）。"""
    coeffs = amb["coeffs"]
    side = amb["side"]
    sign = 1 if side == "hi" else -1
    if bracket is None:
        t = amb["t"]
        if t is None:
            raise BranchRefineError("无理阈值不得走精确三分裂")
        t_lo = t_hi = t
        exact = True
    else:
        t_lo, t_hi = bracket
        exact = False
    s = sign
    scoeffs = tuple(sign * c for c in coeffs)
    out: List[Tuple[Domain, dict]] = []
    tag = f"contact{tuple(amb['pair'])}:{'hi' if side == 'hi' else 'lo'}"
    # 开侧：s·u < t_lo（该侧接触必然成立）
    out.append((dom.with_halfspace(HalfSpace(scoeffs, t_lo, True, tag + ":lt")),
                {"kind": "open_below", "tag": tag}))
    if exact:
        out.append((dom.with_halfspace(HalfSpace(scoeffs, t_hi, False, tag + ":eq"))
                    .with_halfspace(HalfSpace(tuple(-c for c in scoeffs), -t_hi, False,
                                              tag + ":eq2")),
                    {"kind": "surface", "tag": tag}))
        out.append((dom.with_halfspace(HalfSpace(tuple(-c for c in scoeffs), -t_hi, True,
                                                 tag + ":gt")),
                    {"kind": "open_above", "tag": tag}))
    else:
        # 夹逼带 [t_lo, t_hi]：显式容忍带，测度精确可算；不做证明，按界报
        band = (dom.with_halfspace(HalfSpace(scoeffs, t_hi, False, tag + ":band_hi"))
                .with_halfspace(HalfSpace(tuple(-c for c in scoeffs), -t_lo, False,
                                          tag + ":band_lo")))
        out.append((band, {"kind": "tolerance_band", "tag": tag}))
        out.append((dom.with_halfspace(HalfSpace(tuple(-c for c in scoeffs), -t_hi, True,
                                                 tag + ":gt")),
                    {"kind": "open_above", "tag": tag}))
    kept = []
    for d, meta in out:
        if _open_empty(d):
            continue
        kept.append((d, meta))
    return kept


def _make_children_box(spec: GeometrySpec, dom: Domain, k: int, mid: Fraction
                       ) -> List[Tuple[Domain, dict]]:
    lo, hi = dom.box[k]
    left = list(dom.box)
    right = list(dom.box)
    left[k] = (lo, mid)
    right[k] = (mid, hi)
    out = []
    for bx, tag in ((tuple(left), "left"), (tuple(right), "right")):
        d = Domain(dom.n, bx, dom.halfspaces, dom.fixed)
        if not _open_empty(d):
            out.append((d, {"kind": "box_bisect", "dim": k, "mid": str(mid), "tag": tag}))
    return out


def _straddle_coords(ambs: Sequence[dict]) -> List[int]:
    out = []
    for a in ambs:
        for k, c in enumerate(a["coeffs"]):
            if c != 0:
                out.append(k)
    return sorted(set(out))


def choose_split(spec: GeometrySpec, dom: Domain, rule: str) -> Optional[dict]:
    """返回分裂决定（只读声明几何，不读任何隐藏真值）；None = 无可分裂。"""
    if rule not in SPLIT_RULES:
        raise BranchRefineError(f"未声明的分裂规则 {rule!r}")
    free_bits = [i for i in spec.free_presence if dom.fixed[i] < 0]
    if free_bits:
        b = min(free_bits)
        return {"kind": "presence", "bit": b,
                "provenance": f"rule={rule}; declared free_presence; lowest index"}
    ambs = _ambiguity(spec, dom)
    if rule == "threshold":
        if not ambs:
            return None
        exacts = [a for a in ambs if a["t"] is not None]
        if not exacts:
            return {"kind": "none", "reason": "IRRATIONAL_THRESHOLD_NOT_EXACTLY_SPLITTABLE",
                    "provenance": f"rule={rule}"}
        a = max(exacts, key=lambda z: abs(z["gap2"]))
        return {"kind": "threshold", "amb": a,
                "provenance": f"rule={rule}; largest |gap| among exact-threshold sides"}
    if rule == "hybrid":
        if not ambs:
            return None
        a = max(ambs, key=lambda z: abs(z["gap2"]))
        if a["t"] is None:
            return {"kind": "threshold_bracket", "amb": a,
                    "bracket": _bracket_sqrt(a["q"]),
                    "provenance": f"rule={rule}; irrational threshold → rational bracket "
                                  f"{BRACKET_DIGITS} digits"}
        return {"kind": "threshold", "amb": a,
                "provenance": f"rule={rule}; exact threshold preferred"}
    if rule == "gap_widest":
        if ambs:
            coords = _straddle_coords(ambs)
            k = max(coords, key=lambda i: (dom.box[i][1] - dom.box[i][0]))
            w = dom.box[k][1] - dom.box[k][0]
            if w <= 0:
                return None
            return {"kind": "box_bisect", "dim": k,
                    "mid": (dom.box[k][0] + dom.box[k][1]) / 2,
                    "provenance": f"rule={rule}; widest coord among straddling constraints"}
        return _choose_box_widest(dom, rule)
    return _choose_box_widest(dom, rule)


def _choose_box_widest(dom: Domain, rule: str) -> Optional[dict]:
    k = max(range(dom.n), key=lambda i: (dom.box[i][1] - dom.box[i][0]))
    w = dom.box[k][1] - dom.box[k][0]
    if w <= 0:
        return None
    return {"kind": "box_bisect", "dim": k, "mid": (dom.box[k][0] + dom.box[k][1]) / 2,
            "provenance": f"rule={rule}; widest coordinate"}


def split_domain(spec: GeometrySpec, dom: Domain, dec: Mapping
                 ) -> List[Tuple[Domain, dict]]:
    kind = dec["kind"]
    if kind == "presence":
        b = int(dec["bit"])
        out = []
        for v in (0, 1):
            d = dom.with_fixed(b, v)
            if not _open_empty(d):
                out.append((d, {"kind": "presence", "bit": b, "value": v}))
        return out
    if kind == "box_bisect":
        return _make_children_box(spec, dom, int(dec["dim"]), dec["mid"])
    if kind == "threshold":
        return _make_children_threshold(spec, dom, dec["amb"])
    if kind == "threshold_bracket":
        return _make_children_threshold(spec, dom, dec["amb"], dec["bracket"])
    raise BranchRefineError(f"未实现的分裂 {kind!r}")


# ---------------------------------------------------------------------------
# 分支细化主循环
# ---------------------------------------------------------------------------

def domain_measure(spec: GeometrySpec, dom: Domain) -> Fraction:
    """区域在族内的测度 = 体积 × 未落定在场位的 2 倍率（同 S2-N0 的 ``_region_measure``）。"""
    return volume(dom) * (2 ** dom.n_free_presence(spec.free_presence))


def _region_record(spec: GeometrySpec, dom: Domain, info: EnvelopeInfo, depth: int,
                   rid: str, stats: Dict[str, int]) -> dict:
    v = volume(dom)
    meas = domain_measure(spec, dom)
    stats["witness_calls"] += 1
    wit = find_witness(spec, dom, info.label) if info.label != ident.RegionLabel.UNRESOLVED.value else None
    if info.label != ident.RegionLabel.UNRESOLVED.value and wit is None:
        raise BranchRefineError(f"区域 {rid} 判明却找不到可精确重算的见证 ⇒ 仪器缺陷")
    rec = {
        "region_id": rid,
        "label_claim": (None if info.label == ident.RegionLabel.UNRESOLVED.value
                        else info.label),
        "domain": dom.to_dict(),
        "volume": str(v),
        "measure": str(meas),
        "depth": int(depth),
        "must_edges": [list(e) for e in sorted(info.envelope.must_edges)],
        "may_edges": [list(e) for e in sorted(info.envelope.may_edges)],
        "must_a": sorted(info.envelope.must_a_side), "may_a": sorted(info.envelope.may_a_side),
        "must_b": sorted(info.envelope.must_b_side), "may_b": sorted(info.envelope.may_b_side),
        "must_nodes": sorted(info.envelope.must_nodes),
        "may_nodes": sorted(info.envelope.may_nodes),
        "witness": wit,
        "three_route_ok": bool(info.three_route["mismatch"] == 0),
        "three_route": {"pairs_checked": info.three_route["checked"],
                        "mismatch": info.three_route["mismatch"]},
    }
    return rec


def refine_region(region, geometry_spec, split_rule: str = "threshold",
                  budget: Optional[Mapping] = None) -> dict:
    """单区域上的分支定界。``region`` 可为 None（根）、``Domain`` 或 ``to_dict()``。"""
    spec = geometry_spec if isinstance(geometry_spec, GeometrySpec) \
        else GeometrySpec.from_dict(geometry_spec)
    if region is None:
        root = spec.root_domain()
    elif isinstance(region, Domain):
        root = region
    else:
        root = Domain.from_dict(region)
    b = dict(budget or {})
    max_nodes = int(b.get("max_nodes", 512))
    max_depth = int(b.get("max_depth", 64))
    t0 = time.perf_counter()
    stats = {"interval_ops": 0, "envelope_calls": 0, "vertex_solves": 0,
             "split_calls": 0, "witness_calls": 0}
    stack: List[Tuple[Domain, int]] = [(root, 0)]
    terminals: List[dict] = []
    frontier: List[dict] = []
    nodes = 0
    budget_hit = False
    no_split_reasons: List[str] = []
    while stack:
        if nodes >= max_nodes:
            budget_hit = True
            break
        dom, depth = stack.pop()
        nodes += 1
        info = envelope_of_domain(spec, dom, stats)
        if info.label in ("1", "0"):
            terminals.append(_region_record(spec, dom, info, depth,
                                            f"R{len(terminals)}", stats))
            continue
        if depth >= max_depth:
            frontier.append({"reason": "DEPTH_LIMIT", "depth": depth,
                             "domain": dom.to_dict(),
                             "measure": str(domain_measure(spec, dom))})
            no_split_reasons.append("DEPTH_LIMIT")
            continue
        dec = choose_split(spec, dom, split_rule)
        if dec is None or dec["kind"] == "none":
            why = (dec or {}).get("reason", "NO_APPLICABLE_SPLIT")
            frontier.append({"reason": why, "depth": depth, "domain": dom.to_dict(),
                             "measure": str(domain_measure(spec, dom)),
                             "label_interval": "UNRESOLVED"})
            no_split_reasons.append(why)
            continue
        stats["split_calls"] += 1
        if dec["kind"] == "threshold_bracket":
            stats["bracket_splits"] = stats.get("bracket_splits", 0) + 1
        kids = split_domain(spec, dom, dec)
        if not kids:
            frontier.append({"reason": "SPLIT_PRODUCED_EMPTY", "depth": depth,
                             "domain": dom.to_dict(),
                             "measure": str(domain_measure(spec, dom))})
            no_split_reasons.append("SPLIT_PRODUCED_EMPTY")
            continue
        for d, meta in reversed(kids):     # 逆序入栈 ⇒ 先展开第一个子域（左/下支优先，同 S2-N0）
            if meta.get("kind") == "tolerance_band":
                frontier.append({"reason": "TOLERANCE_BAND_DECLARED", "depth": depth + 1,
                                 "domain": d.to_dict(),
                                 "measure": str(domain_measure(spec, d)),
                                 "label_interval": "UNRESOLVED"})
                no_split_reasons.append("TOLERANCE_BAND_DECLARED")
                continue
            stack.append((d, depth + 1))
    if stack:      # 预算耗尽时栈里剩的件必须计入前沿（否则测度不闭合）
        budget_hit = True
        for d, dep in stack:
            frontier.append({"reason": "BUDGET_EXHAUSTED", "depth": dep,
                             "domain": d.to_dict(),
                             "measure": str(domain_measure(spec, d)),
                             "label_interval": "UNRESOLVED"})
        no_split_reasons.append("BUDGET_EXHAUSTED")
    wall = time.perf_counter() - t0
    covered = sum(Fraction(r["measure"]) for r in terminals)
    residual = sum(Fraction(f["measure"]) for f in frontier)
    fam = spec.family_measure()
    return {"schema": "s2b.refine/v1", "family_id": spec.family_id,
            "split_rule": split_rule, "budget": {"max_nodes": max_nodes,
                                                 "max_depth": max_depth},
            "nodes": nodes, "terminal_regions": len(terminals),
            "frontier_regions": len(frontier),
            "budget_hit": bool(budget_hit),
            # 终止 = 没有测度残差（零测度面若未决，单独计入 zero_measure_frontier）
            "terminated": bool(residual == 0 and not budget_hit),
            "zero_measure_frontier": sum(1 for f in frontier
                                         if Fraction(f["measure"]) == 0),
            "labels": sorted({r["label_claim"] for r in terminals}),
            "measure_family": str(fam), "measure_terminal": str(covered),
            "measure_residual": str(residual),
            "measure_closed": bool(covered + residual == fam),
            "measure_residual_within_tol": bool(residual <= REL_TOL_MEASURE * fam),
            "regions": terminals, "frontier": frontier,
            "no_split_reasons": sorted(set(no_split_reasons)),
            "stats": stats, "wall_s": round(wall, 6)}


# ---------------------------------------------------------------------------
# 证书 / 界 / 未解决：s2.certificate/v1 皮 + s2b.geometry/v1 扩展
# ---------------------------------------------------------------------------

def _cert_id(spec: GeometrySpec, rule: str) -> str:
    return f"s2b:{spec.family_id}:{rule}"


def _certificate_bones(spec: GeometrySpec, rule: str, res: Mapping) -> dict:
    return {"schema": SCHEMA_CERT, "schema_extension": SCHEMA_EXT,
            "certificate_id": _cert_id(spec, rule),
            "model_id": "fractureflow.branch_refine",
            "table_id": f"geom:{spec.family_id}",
            "query_id": f"GEOM:{spec.family_id}:A->B",
            "n_bits": int(spec.n), "allowed_state_ids": [],
            "verifier_version": VERIFIER_VERSION,
            "selection_provenance": SELECTION_PROVENANCE,
            "split_rule": rule,
            "geometry": {"family_id": spec.family_id, "spec_sha12": spec.spec_sha12(),
                         "spec": spec.to_dict(),
                         "measure_family": str(spec.family_measure()),
                         "volume_family": str(spec.family_volume())},
            "budget": res["budget"],
            "costs": {"nodes": res["nodes"], "wall_s": res["wall_s"],
                      "interval_ops": res["stats"]["interval_ops"],
                      "envelope_calls": res["stats"]["envelope_calls"],
                      "split_calls": res["stats"]["split_calls"]},
            "stats": {"terminal_regions": res["terminal_regions"],
                      "frontier_regions": res["frontier_regions"],
                      "budget_hit": res["budget_hit"],
                      "measure_terminal": res["measure_terminal"],
                      "measure_residual": res["measure_residual"],
                      "measure_closed": res["measure_closed"]},
            "watermark": WATERMARK}


@dataclass
class GeomCertificate:
    """终止且覆盖闭合（残差为 0）的分区证书；混合分区 ``label_claim=None``。"""
    body: dict
    kind: str = "CERTIFICATE"

    def to_dict(self) -> dict:
        return dict(self.body, artifact_kind=self.kind)

    @property
    def label_claim(self):
        return self.body.get("label_claim")

    @property
    def regions(self) -> List[dict]:
        return self.body.get("regions", [])


@dataclass
class Bound:
    """未终止但残差测度 ≤ 声明容差：可验证的未解决界（不是物理不可识别）。"""
    body: dict
    kind: str = "BOUND"

    def to_dict(self) -> dict:
        return dict(self.body, artifact_kind=self.kind)

    @property
    def regions(self) -> List[dict]:
        return self.body.get("regions", [])


@dataclass
class Unresolved:
    """残差测度 > 容差：只报上界与前沿件数；禁写成不可识别。"""
    body: dict
    kind: str = "UNRESOLVED"

    def to_dict(self) -> dict:
        return dict(self.body, artifact_kind=self.kind)

    @property
    def regions(self) -> List[dict]:
        return self.body.get("regions", [])


def certify_or_bound(region, geometry_spec, budget: Optional[Mapping] = None, *,
                     split_rule: str = "threshold",
                     rel_tol: Optional[Fraction] = None):
    """终止 ⇒ 证书；未终止 ⇒ 界（残差 ≤ 容差）或未解决（残差 > 容差）。"""
    spec = geometry_spec if isinstance(geometry_spec, GeometrySpec) \
        else GeometrySpec.from_dict(geometry_spec)
    tol = Fraction(rel_tol) if rel_tol is not None else REL_TOL_MEASURE
    res = refine_region(region, spec, split_rule=split_rule, budget=budget)
    fam = spec.family_measure()
    body = _certificate_bones(spec, split_rule, res)
    body["regions"] = res["regions"]
    body["frontier"] = res["frontier"]
    body["residual"] = {"measure": res["measure_residual"],
                        "n_frontier": res["frontier_regions"],
                        "tolerance_measure": str(tol * fam),
                        "relative_tolerance": str(tol),
                        "within_tolerance": bool(Fraction(res["measure_residual"]) <= tol * fam)}
    labs = {r["label_claim"] for r in res["regions"] if r["label_claim"] is not None}
    body["witness"] = {"per_region": [{"region_id": r["region_id"],
                                       "witness": r["witness"]} for r in res["regions"]],
                      "n_witness": sum(1 for r in res["regions"] if r["witness"])}
    if res["terminated"] and Fraction(res["measure_residual"]) == 0:
        body["label_claim"] = (labs.pop() if len(labs) == 1 else None)
        body["proof_route"] = ("INTERVAL_REFINEMENT_TOLERANCE"
                               if int(res["stats"].get("bracket_splits", 0)) > 0
                               else "INTERVAL_REFINEMENT_EXACT")
        body["a_e_reading"] = {
            "rule": "残差测度为 0 ⇒ 分区在集合上覆盖整族；a.e. 标签按测度取，"
                    "单点/低维面不改变 a.e. 读数（只作派生读数，不是证书标签）",
            "labels_measure": {lab: str(sum(Fraction(r["measure"]) for r in res["regions"]
                                            if r["label_claim"] == lab)) for lab in labs}}
        return GeomCertificate(body)
    if Fraction(res["measure_residual"]) <= tol * fam:
        body["label_claim"] = None
        body["proof_route"] = "INTERVAL_REFINEMENT_TOLERANCE_BOUND"
        body["a_e_reading"] = {
            "rule": "残差只到声明容差：分解决定部分已判明，未决部分测度 ≤ 容差；"
                    "不得据此写物理不可识别",
            "labels_measure": {lab: str(sum(Fraction(r["measure"]) for r in res["regions"]
                                            if r["label_claim"] == lab)) for lab in labs}}
        return Bound(body)
    body["label_claim"] = None
    body["proof_route"] = "INTERVAL_REFINEMENT_UNRESOLVED_BOUND"
    body["a_e_reading"] = {
        "rule": "预算耗尽且残差超容差：只报未覆盖测度上界/前沿件数/节点；"
                "禁写成物理不可识别，禁作族外推",
        "labels_measure": {lab: str(sum(Fraction(r["measure"]) for r in res["regions"]
                                        if r["label_claim"] == lab)) for lab in labs}}
    return Unresolved(body)


def verify_bound(bound, geometry_spec) -> dict:
    """界/未解决件的自校验（生产侧镜像；独立验证器见 scripts/s2b_verifier.py）。"""
    spec = geometry_spec if isinstance(geometry_spec, GeometrySpec) \
        else GeometrySpec.from_dict(geometry_spec)
    art = bound.to_dict() if hasattr(bound, "to_dict") else dict(bound)
    reasons: List[str] = []
    regions = art.get("regions", [])
    frontier = art.get("frontier", [])
    covered = Fraction(0)
    for r in regions:
        dom = Domain.from_dict(r["domain"])
        v = volume(dom)
        if str(v) != str(r["volume"]):
            reasons.append(f"区域 {r['region_id']} 体积重算不符")
        if not vertices(dom):
            reasons.append(f"区域 {r['region_id']} 为空域却带声明")
        info = envelope_of_domain(spec, dom)
        if info.label != r["label_claim"]:
            reasons.append(f"区域 {r['region_id']} 标签重算 {info.label} != 声明 "
                           f"{r['label_claim']}")
        covered += Fraction(r["measure"])
    residual = Fraction(0)
    for f in frontier:
        dom = Domain.from_dict(f["domain"])
        residual += domain_measure(spec, dom)
    fam = spec.family_measure()
    if covered + residual != fam:
        reasons.append("已判部分 + 残差 != 族测度（覆盖不闭合）")
    decl = Fraction(art.get("residual", {}).get("measure", "0"))
    if decl != residual:
        reasons.append(f"声明残差 {decl} != 重算残差 {residual}")
    tol = Fraction(art.get("residual", {}).get("tolerance_measure", "0"))
    if art.get("artifact_kind") == "BOUND" and residual > tol:
        reasons.append("BOUND 件残差超声明容差")
    return {"pass": not reasons, "reasons": reasons,
            "measure_family": str(fam), "measure_covered": str(covered),
            "measure_residual": str(residual), "tolerance_measure": str(tol),
            "n_regions": len(regions), "n_frontier": len(frontier),
            "label_claim": art.get("label_claim")}


__all__ = [
    "SCHEMA_CERT", "SCHEMA_EXT", "VERIFIER_VERSION", "LINE", "SPLIT_RULES",
    "REL_TOL_MEASURE", "WATERMARK", "BranchRefineError",
    "Q", "HalfSpace", "Domain", "GeometrySpec",
    "vertices", "volume", "form_bounds", "form_bounds_fm", "form_bounds_elementwise",
    "envelope_of_domain", "direction_rule", "point_label", "find_witness",
    "choose_split", "split_domain", "refine_region",
    "GeomCertificate", "Bound", "Unresolved", "certify_or_bound", "verify_bound",
]
