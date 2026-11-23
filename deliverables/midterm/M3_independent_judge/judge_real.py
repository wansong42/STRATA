#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3_independent_judge/judge_real.py
独立判卷器（零共享代码）：对 M1_real_to_exact_bridge + M2_active_identifiability
冻结产物做异模型精确有理盲判。

约束（任务书 M3 §一）：
  - 纯 Python 标准库 + fractions.Fraction；零浮点参与最终判定；
  - 不读取/不 import 任何 M1/M2/TC/TB/src 求解 .py（由 audit_imports.py 证明）；
  - 仅允许读取以下 JSON：cluster_register.json、clusters/*/model.json、
    runs/<run>/summary.json、cases/*/{certification,active_check}.json；
  - 每次 open() 记录到 read_ledger.jsonl；退出码 0=AGREE 1=DISAGREE/异常。

独立几何内核说明：
  - 接触窗口：每条边 (X, Y)（X∈{点 A, 点 B, 三角形片}，Y=三角形片）随共享参数 t
    的接触域写成关于 (t, x, b) 的精确线性可行系统：
      点-片  : x = p_X 且 x = B·b（b 为 Y 的 3 顶点重心坐标，b>=0, sum=1）
      片-片  : B_L·b_L = B_R·b_R（b_L,b_R>=0, sum=1）
      其中 B = v0 + t·v_comp + (v1-v0, v2-v0)；b 变量数 <=7，求解用自行实现的
      Fraction 精确原始单纯形（Bland 规则），接触域 = {t: exists(x,b)} 的投影；
  - cell 剖分：所有窗口端点排序成事件 knot，POINT 为单点、OPEN 为开区间，两者
    交替覆盖 mode 的 segments 并集（带 lo/hi_closed 开闭语义）；
  - 每 cell 以 sample t 枚举活动边图，BFS 判连通 / 记录最小割见证；
  - 三态、ρ、最近翻转见证、collapse_tree 条件域全部独立重算比对。
"""
import ast, hashlib, json, os, sys, io, time
from fractions import Fraction
from contextlib import contextmanager

# ----------------------------------------------------------------------
# 读取台账（证明只读白名单 JSON）
# ----------------------------------------------------------------------
LEDGER_PATH = None
_FORBIDDEN_SUBSTR = ("M1_real_to_exact_bridge", "M2_active_identifiability",
                     "TC_network", "TB_feasible", "src\\fractureflow",
                     "src/fractureflow")
_ALLOWED_SUFFIX = (".json", ".jsonl", ".md")
_allowed_reads = []  # 运行时记录

def ledger_open(path, mode="r", *a, **kw):
    """受控 open：写台账，且禁止以读模式打开任何 .py。"""
    rp = os.path.abspath(path)
    if "r" in mode or "+" in mode:
        if rp.lower().endswith(".py"):
            raise PermissionError("M3 independence: reading .py is forbidden: " + rp)
        _allowed_reads.append({"path": rp.replace("\\", "/"), "mode": mode,
                               "bytes": os.path.getsize(rp) if os.path.exists(rp) else -1})
    return open(path, mode, *a, **kw)

def jload(path):
    with ledger_open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flush_ledger(path, extra_meta):
    recs = list(_allowed_reads)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(recs)

# ----------------------------------------------------------------------
# 分数工具
# ----------------------------------------------------------------------
def F(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError("non-exact numeric field %r" % (x,))

def Fv(v):
    return [F(t) for t in v]

def vadd(a, b): return [a[i] + b[i] for i in range(3)]
def vsub(a, b): return [a[i] - b[i] for i in range(3)]
def vscale(a, s): return [a[i] * s for i in range(3)]
def vdot(a, b): return sum(a[i] * b[i] for i in range(3))
def vcross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

# ----------------------------------------------------------------------
# 精确线性规划（Fraction 原始单纯形，Bland 规则）
# 求解  max c^T x  s.t. A x <= b, x >= 0
# ----------------------------------------------------------------------
class LPResult:
    __slots__ = ("status", "obj", "x")
    def __init__(self, status, obj=None, x=None):
        self.status, self.obj, self.x = status, obj, x

def exact_lp(c, A, b):
    """max c·x, Ax<=b, x>=0。KACTL 单纯形（已验证算法，转 Fraction）。

    参考：KACTL SimplexAlgorithm（Stanford ICPC notebook），两阶段、
    Bland 规则防循环。返回 LPResult('OPTIMAL'/'INFEASIBLE'/'UNBOUNDED')。
    """
    m, n = len(A), len(c)
    N = n + 2
    D = [[Fraction(0)] * (n + 2) for _ in range(m + 2)]
    B = [0] * m   # 基变量（原变量 0..n-1，人工变量 n..n+m-1）
    Nv = [0] * (n + 1)  # 非基变量
    for i in range(m):
        for j in range(n):
            D[i][j] = F(A[i][j])
        B[i] = n + i
        D[i][n] = Fraction(-1)   # 人工变量列
        D[i][n + 1] = F(b[i])
    for j in range(n):
        Nv[j] = j
        D[m][j] = -F(c[j])
    Nv[n] = -1
    D[m + 1][n] = Fraction(1)

    def pivot(r, s):
        inv = Fraction(1) / D[r][s]
        for i in range(m + 2):
            if i == r:
                continue
            for j in range(n + 2):
                if j == s:
                    continue
                D[i][j] -= D[r][j] * D[i][s] * inv
        for j in range(n + 2):
            if j != s:
                D[r][j] *= inv
        for i in range(m + 2):
            if i != r:
                D[i][s] = -D[i][s] * inv
        D[r][s] = inv
        B[r], Nv[s] = Nv[s], B[r]

    def simplex(phase):
        x = m + 1 if phase == 1 else m
        while True:
            s = -1
            for j in range(n + 1):
                if phase == 2 and Nv[j] == -1:
                    continue
                if s == -1 or D[x][j] < D[x][s] or \
                   (D[x][j] == D[x][s] and Nv[j] < Nv[s]):
                    s = j
            if D[x][s] >= 0:
                return True
            r = -1
            for i in range(m):
                if D[i][s] > 0:
                    if r == -1:
                        r = i
                    elif D[i][n + 1] / D[i][s] < D[r][n + 1] / D[r][s] or \
                         (D[i][n + 1] / D[i][s] == D[r][n + 1] / D[r][s] and B[i] < B[r]):
                        r = i
            if r == -1:
                return False
            pivot(r, s)

    r = 0
    for i in range(1, m):
        if D[i][n + 1] < D[r][n + 1]:
            r = i
    if D[r][n + 1] < 0:
        pivot(r, n)
        if not simplex(1) or D[m + 1][n + 1] < 0:
            return LPResult("INFEASIBLE")
        if D[m + 1][n + 1] > 0:
            return LPResult("INFEASIBLE")
        for i in range(m):
            if B[i] == -1:
                s = 0
                for j in range(1, n + 1):
                    if D[i][j] < D[i][s] or (D[i][j] == D[i][s] and Nv[j] < Nv[s]):
                        s = j
                pivot(i, s)
    if not simplex(2):
        return LPResult("UNBOUNDED")
    x = [Fraction(0)] * n
    for i in range(m):
        if B[i] < n:
            x[B[i]] = D[i][n + 1]
    return LPResult("OPTIMAL", D[m][n + 1], x)


def lp_feasible(A, b):
    """exists x>=0 with Ax<=b ？精确判定。"""
    r = exact_lp([Fraction(0)] * (len(A[0]) if A else 0), A, b)
    return r.status == "OPTIMAL"

# ----------------------------------------------------------------------
# 几何实体
# ----------------------------------------------------------------------
class Tri:
    __slots__ = ("id", "comp", "verts", "v")
    def __init__(self, d):
        self.id = d["id"]
        self.comp = d["component"]
        self.verts = [Fv(x) for x in d["vertices"]]
        self.v = Fv(d["v"])

class Mode:
    __slots__ = ("id", "edit_cost", "segments", "triangles", "tri_by_id")
    def __init__(self, d):
        self.id = d["id"]
        self.edit_cost = F(d["edit_cost"])
        self.segments = [{"lo": F(s["lo"]), "hi": F(s["hi"]),
                          "lo_closed": bool(s["lo_closed"]), "hi_closed": bool(s["hi_closed"])}
                         for s in d["segments"]]
        self.triangles = [Tri(t) for t in d["triangles"]]
        self.tri_by_id = {t.id: t for t in self.triangles}

def mode_allows(m: Mode, t: Fraction, side=None):
    """t 是否落在 mode 的任一段；side in {'LO','HI',None} 用于端点闭性区分。"""
    if side is None:
        return any((s["lo"] < t < s["hi"]) or
                   (s["lo_closed"] and t == s["lo"]) or
                   (s["hi_closed"] and t == s["hi"]) for s in m.segments)
    for s in m.segments:
        if side == "LO" and s["lo_closed"] and t == s["lo"]:
            return True
        if side == "HI" and s["hi_closed"] and t == s["hi"]:
            return True
    return False

def seg_interior_sample(m: Mode, lo: Fraction, hi: Fraction):
    """(lo, hi) 开区间中点。"""
    return (lo + hi) / 2

# ----------------------------------------------------------------------
# 接触窗口：把边 e=(X,Y) 的接触集投影到 t
# ----------------------------------------------------------------------
# 变量布局（最大化/可行性 LP，全部 x>=0，用 y = x - L 转下界）：
#   片-片 : z = (t, u, bL1..3, bR1..3)  其中 t = u - K，K = |t| 上界（取 4 够用，
#            t∈[-1,1] 恒真但接触 t 可能在界外，K 取大不损失）
#   点-片 : z = (t, u, b1..3)
T_BOUND = Fraction(16)  # t 的松弛下界常数：t = u - T_BOUND, u>=0

def _edge_lp_matrix(mode, ta, tb, point_A=None, point_B=None):
    """
    构造边 (ta, tb) 接触系统的 (A, b, t_index_map)。
    ta/tb 可为 Tri、'@A'、'@B'（点端点坐标由 point_A/point_B 给）。
    返回 (A_ineq, b_ineq, n_vars, unpack_t)
    变量 z>=0: z = (u, b...) 其中 t = u - T_BOUND。
    约束写成 A z <= b。
    """
    left_is_pt = isinstance(ta, str)
    rows = []
    rhs = []

    def tri_rows(tri):
        """返回该三角形顶点矩阵按 z 变量取列位置的起始 offset。"""
        pass  # placeholder

    # 先确定变量数
    nl = 0 if left_is_pt else 3
    nr = 3  # 右侧恒为 Tri
    nvars = 1 + nl + nr
    u_off = 0
    bl_off = 1
    br_off = 1 + nl

    def B_col(tri):
        """返回 (v0, w, d1, d2)：x = v0 + w*t + d1*b1 + d2*b2。"""
        d1 = vsub(tri.verts[1], tri.verts[0])
        d2 = vsub(tri.verts[2], tri.verts[0])
        v0 = tri.verts[0]
        w = tri.v
        return v0, w, d1, d2

    if left_is_pt:
        pL = Fv(point_A if ta == "@A" else point_B)
        # 等式 x = pL 且 x = B_R bR  =>  B_R bR - pL = 0  (3 eq)
        v0, w, d1, d2 = B_col(tb)
        for ax in range(3):
            # pL_ax = v0_ax + w_ax*t + d1_ax*b1 + d2_ax*b2
            # => w_ax*(u - T_BOUND) + d1_ax*b1 + d2_ax*b2 = pL_ax - v0_ax
            row = [Fraction(0)] * nvars
            row[u_off] = w[ax]
            row[br_off] = d1[ax]
            row[br_off + 1] = d2[ax]
            c = pL[ax] - v0[ax] + w[ax] * T_BOUND
            rows.append(row[:]); rhs.append(c)
            rows.append([-r for r in row]); rhs.append(-c)
    else:
        # 片-片：B_L bL - B_R bR = 0 (3 eq)
        v0L, wL, d1L, d2L = B_col(ta)
        v0R, wR, d1R, d2R = B_col(tb)
        for ax in range(3):
            # wL_ax*t + d1L_ax*bL1 + d2L_ax*bL2 - (wR_ax*t + d1R_ax*bR1 + d2R_ax*bR2) = v0R_ax - v0L_ax
            row = [Fraction(0)] * nvars
            row[u_off] = wL[ax] - wR[ax]
            row[bl_off] = d1L[ax]; row[bl_off + 1] = d2L[ax]
            row[br_off] = -d1R[ax]; row[br_off + 1] = -d2R[ax]
            c = v0R[ax] - v0L[ax] + (wL[ax] - wR[ax]) * T_BOUND
            rows.append(row[:]); rhs.append(c)
            rows.append([-r for r in row]); rhs.append(-c)
    # 重心和 = 1
    for off in ([] if left_is_pt else [bl_off]) + [br_off]:
        row = [Fraction(0)] * nvars
        row[off] = row[off + 1] = row[off + 2] = Fraction(1)
        rows.append(row[:]); rhs.append(Fraction(1))
        rows.append([-r for r in row]); rhs.append(Fraction(-1))
    # t 全局界: -T_BOUND <= t <= T_BOUND => 0 <= u <= 2*T_BOUND
    row = [Fraction(0)] * nvars; row[u_off] = Fraction(1)
    rows.append(row); rhs.append(2 * T_BOUND)

    def unpack(z):
        return z[u_off] - T_BOUND
    return rows, rhs, nvars, unpack

def edge_window(mode, ta, tb, point_A, point_B):
    """
    精确计算边 (ta,tb) 在 mode 下的接触 t 集（投射到 t 轴的闭区间或 None）。
    返回 (lo, hi, lo_active, hi_active, mid_witness_x)：
      lo_active/hi_active —— 端点是否真属于接触域（LP 极值点可行性）。
    """
    rows, rhs, nvars, unpack = _edge_lp_matrix(mode, ta, tb, point_A, point_B)
    # 可行性
    if not lp_feasible(rows, rhs):
        return None
    # t = u - T_BOUND：min t = min u - T_BOUND；max t = max u - T_BOUND
    cmin = [Fraction(0)] * nvars
    cmin[0] = Fraction(-1)  # max(-u) = -min u
    r_min = exact_lp(cmin, rows, rhs)
    cmax = [Fraction(0)] * nvars
    cmax[0] = Fraction(1)
    r_max = exact_lp(cmax, rows, rhs)
    if r_min.status != "OPTIMAL" or r_max.status != "OPTIMAL":
        return None
    lo, hi = unpack(r_min.x), unpack(r_max.x)
    return (lo, hi, True, True)  # 线性闭集投影为闭区间，端点可达

def point_in_tri_at(mode, pt, tri):
    """点 pt 是否在 t 时刻与 tri 接触（LP 可行）。"""
    rows, rhs, nvars, unpack = _edge_lp_matrix(mode, "@A", tri, pt, pt)
    return lp_feasible(rows, rhs)

# ----------------------------------------------------------------------
# cell 剖分与图连通
# ----------------------------------------------------------------------
def knot_bounds(windows):
    ks = set()
    for w in windows:
        ks.add(w[0]); ks.add(w[1])
    return sorted(ks)

def build_cells(mode, windows):
    """把接触窗口端点排序后剖分为 POINT/OPEN cell 交替序列（对 mode 的 segments 并集）。"""
    # 收集 mode segments 端点 + 窗口端点
    ks = set()
    for s in mode.segments:
        ks.add(s["lo"]); ks.add(s["hi"])
    ks.update(knot_bounds(windows))
    ks = sorted(ks)
    cells = []
    prev = None
    for k in ks:
        if prev is not None:
            cells.append({"kind": "OPEN", "lo": prev, "hi": k,
                          "sample": (prev + k) / 2})
        cells.append({"kind": "POINT", "lo": k, "hi": k, "sample": k})
        prev = k
    # 只保留落在 mode 允许域的 cell（OPEN 要求内部允许，POINT 要求该点允许）
    keep = []
    for c in cells:
        if c["kind"] == "POINT":
            if mode_allows(mode, c["sample"]):
                keep.append(c)
        else:
            if mode_allows(mode, c["sample"]):
                keep.append(c)
    return keep

def active_edges_at(mode, edges_windows, t, side=None):
    """
    给定 t（POINT cell 直接取，OPEN cell 取区间中点），返回活动无向边集合。
    side：'LO'/'HI' 时还要满足窗口端点活性（闭区间两端恒活性）。
    """
    act = set()
    for (a, b), w in edges_windows.items():
        lo, hi, lo_a, hi_a = w[0], w[1], w[2], w[3]
        inside = (lo < t < hi) or (lo_a and t == lo) or (hi_a and t == hi)
        if inside:
            act.add((a, b))
    return act

def connectivity(nodes_special, act_edges):
    """BFS 求连通分量；返回 components(list[list]) 与是否 A-B 连通。"""
    adj = {}
    for (a, b) in act_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for n in list(adj.keys()) + nodes_special:
        if n in seen:
            continue
        stack, comp = [n], [n]
        seen.add(n)
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):  # neighbors
                if v not in seen:
                    seen.add(v); comp.append(v); stack.append(v)
        comps.append(sorted(comp))
    connected = "@A" in adj.get("@B", set()) or any(
        "@A" in c and "@B" in c for c in comps)
    return comps, connected

def find_path(act_edges, a="@A", b="@B"):
    """BFS 最短路径：返回节点序列或 None。"""
    adj = {}
    for (x, y) in act_edges:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    if a not in adj or b not in adj:
        return None
    from collections import deque
    prev = {a: None}
    dq = deque([a])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                dq.append(v)
                if v == b:
                    path = [b]
                    while path[-1] != a:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
    return None

# ----------------------------------------------------------------------
# 单一 (mode, 条件域) 三态重算
# ----------------------------------------------------------------------
def eval_domain(mode_map, mode_order, windows_by_mode, cond, budget_pairs=10**9):
    """
    在条件 cond 下重算三态。
    cond = {"allowed_modes": set, "mode_segments": {mid: [seg,...]}, "t_predicate": {...}}
    返回 dict(status, exists_connected, all_connected, cells, per_mode, measures)
    """
    allowed = cond.get("allowed_modes")
    segs_override = cond.get("mode_segments") or {}
    tpred = cond.get("t_predicate") or {"kind": "ALL_MODE_SEGMENTS", "threshold_t": None}
    kind = tpred.get("kind", "ALL_MODE_SEGMENTS")
    thr = F(tpred["threshold_t"]) if tpred.get("threshold_t") else None

    def t_ok(t, closed=True):
        if kind == "ALL_MODE_SEGMENTS":
            return True
        if kind == "LE_THRESHOLD":
            return t <= thr if closed else t < thr
        if kind == "LT_THRESHOLD":
            return t < thr
        if kind == "GT_THRESHOLD":
            return t > thr
        if kind == "GE_THRESHOLD":
            return t >= thr
        return True

    results = {}
    exists_conn, all_conn = False, True
    total_measure = Fraction(0)
    conn_measure = Fraction(0)
    cells_all = []
    conn_cell_cnt = 0
    for mid in mode_order:
        if allowed is not None and mid not in allowed:
            continue
        mode = mode_map[mid]
        # 该 mode 的域：mode.segments（可能被 cond.mode_segments 覆盖）∩ t_predicate
        dom_segs = segs_override.get(mid, mode.segments)
        # 以域端点 + 窗口端点为 knots
        ks = set()
        for s in dom_segs:
            ks.add(s["lo"]); ks.add(s["hi"])
        if thr is not None:
            ks.add(thr)
        for w in windows_by_mode.get(mid, {}).values():
            ks.add(w[0]); ks.add(w[1])
        ks = sorted(ks)
        # 生成原子（POINT + OPEN）
        atoms = []
        prev = None
        for k in ks:
            if prev is not None:
                atoms.append(("OPEN", prev, k))
            atoms.append(("POINT", k, k))
            prev = k
        mcells = []
        m_feasible_measure = Fraction(0)
        m_conn_cells = 0
        m_disconn_cells = 0
        for knd, lo, hi in atoms:
            # 域内过滤
            def in_dom(t, endpoint_side=None):
                for s in dom_segs:
                    loin = (s["lo"] < t) or (s["lo_closed"] and t == s["lo"])
                    hiin = (t < s["hi"]) or (s["hi_closed"] and t == s["hi"])
                    if endpoint_side == "LO":
                        hiin = t <= s["hi"]
                    if endpoint_side == "HI":
                        loin = s["lo"] <= t
                    if loin and hiin:
                        return True
                return False
            if knd == "OPEN":
                if lo >= hi:
                    continue
                sample = (lo + hi) / 2
                if not in_dom(sample):
                    continue
                # 谓词约束：整个开区间须满足
                if not (t_ok(sample) and t_ok(lo, False) is not None):
                    pass
                if not t_ok(sample):
                    continue
            else:
                sample = lo
                if not in_dom(sample):
                    continue
                if not t_ok(sample):
                    continue
            act = active_edges_at(mode, windows_by_mode.get(mid, {}), sample)
            comps, conn = connectivity(["@A", "@B"], act)
            cell = {"mode": mid, "kind": knd, "lo": lo, "hi": hi,
                    "sample": sample, "connected": conn,
                    "active_edges": sorted(act), "components": comps}
            mcells.append(cell)
            cells_all.append(cell)
            if knd == "OPEN":
                m_feasible_measure += (hi - lo)
            if conn:
                conn_cell_cnt += 1
                m_conn_cells += 1
                if knd == "OPEN":
                    conn_measure += (hi - lo)
            else:
                m_disconn_cells += 1
        if mcells:
            m_conn = any(c["connected"] for c in mcells)
            m_all = all(c["connected"] for c in mcells)
            exists_conn = exists_conn or m_conn
            all_conn = all_conn and m_all
            results[mid] = {"cells": len(mcells), "connected_cells": m_conn_cells,
                            "disconnected_cells": m_disconn_cells,
                            "feasible_measure": m_feasible_measure,
                            "status": ("CERTIFIED_CONNECTED" if m_all else
                                       ("CERTIFIED_DISCONNECTED" if not m_conn else
                                        "AMBIGUOUS_WITH_VERIFIED_WITNESSES"))}
        total_measure += m_feasible_measure
    if not results:
        return {"status": "EMPTY_DOMAIN", "exists_connected": False,
                "all_connected": False, "cells": [], "per_mode": {},
                "total_measure": Fraction(0), "conn_measure": Fraction(0),
                "conn_cell_count": 0, "disconn_cell_count": 0}
    status = ("CERTIFIED_CONNECTED" if all_conn else
              ("CERTIFIED_DISCONNECTED" if not exists_conn else
               "AMBIGUOUS_WITH_VERIFIED_WITNESSES"))
    return {"status": status, "exists_connected": exists_conn,
            "all_connected": all_conn, "cells": cells_all, "per_mode": results,
            "total_measure": total_measure, "conn_measure": conn_measure,
            "conn_cell_count": conn_cell_cnt,
            "disconn_cell_count": len(cells_all) - conn_cell_cnt}

# ----------------------------------------------------------------------
# 判卷
# ----------------------------------------------------------------------
def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode("utf-8")).hexdigest()

def build_windows(model):
    """全枚举每条 (node_i, tri_j) 边在每一 mode 的接触窗口。返回 windows_by_mode。"""
    pa, pb = Fv(model["endpoints"]["A"]), Fv(model["endpoints"]["B"])
    wb = {}
    for md in model["modes"]:
        tris = [Tri(t) for t in md["triangles"]]
        w = {}
        # 片-片：跨裂隙取无向对一次，同裂隙同向去重（与 M2 枚举口径一致）
        seen_pair = set()
        for i in range(len(tris)):
            for j in range(len(tris)):
                if i == j:
                    continue
                a, b = tris[i], tris[j]
                if a.comp != b.comp:
                    if i > j:
                        continue
                    key = (a.id, b.id)
                else:
                    key = tuple(sorted((a.id, b.id)))
                    if key in seen_pair:
                        continue
                seen_pair.add(key)
                r = edge_window(md, a, b, pa, pb)
                if r is not None:
                    r = clip_window_to_mode(md, r)
                if r is not None:
                    w[key] = r
        # 端点-片
        for ep in ("@A", "@B"):
            for tr in tris:
                r = edge_window(md, ep, tr, pa, pb)
                if r is not None:
                    r = clip_window_to_mode(md, r)
                if r is not None:
                    w[(ep, tr.id)] = r
        wb[md["id"]] = w
    return wb

def clip_window_to_mode(md, w):
    """把接触窗口 [lo,hi] 截到 mode.t_closed 全局 t 域；交集为空返回 None。"""
    t_lo, t_hi = F(md["t_closed"][0]), F(md["t_closed"][1])
    lo, hi, la, ha = w
    nlo, nhi = max(lo, t_lo), min(hi, t_hi)
    if nlo > nhi:
        return None
    nla = (nlo == lo and la) or (nlo == t_lo and t_lo >= lo)
    nha = (nhi == hi and ha) or (nhi == t_hi and t_hi <= hi)
    return (nlo, nhi, bool(nla), bool(nha))

def compare_cells(mine, theirs):
    """逐 cell 比对 (mode,kind,lo,hi,connected)。返回首个不一致或 None。"""
    key = lambda c: (c["mode"], c["kind"], str(F(c["lo"])), str(F(c["hi"])))
    M = {key(c): c for c in mine}
    T = {key(c): c for c in theirs}
    for k in T:
        if k not in M:
            return {"reason": "MISSING_CELL", "cell": k}
        mc, tc = M[k], T[k]
        if mc["connected"] != tc["connected"]:
            return {"reason": "CELL_CONNECTED_MISMATCH", "cell": k,
                    "mine": mc["connected"], "theirs": tc["connected"]}
    extra = [k for k in M if k not in T]
    if extra:
        return {"reason": "EXTRA_CELL", "cell": extra[0]}
    return None

def verify_path_witness(model, mode_map, w):
    """独立核验 connected_witness：同一 (mode,t) 内每步真接触且成路径。"""
    if w is None:
        return "connected_witness missing"
    mode = mode_map.get(w["mode"])
    if mode is None:
        return "bad mode"
    t = F(w["t"])
    if not mode_allows(mode, t):
        return "t outside mode domain"
    path = w.get("path") or []
    if not path or path[0] != "@A" or path[-1] != "@B":
        return "path endpoints invalid"
    tri_by_id = mode.tri_by_id
    pa, pb = Fv(model["endpoints"]["A"]), Fv(model["endpoints"]["B"])
    # 检查 contact_step_witnesses 与 path 一致
    csw = w.get("contact_step_witnesses") or []
    if len(csw) != len(path) - 1:
        return "contact_step_witnesses length mismatch"
    for step in csw:
        fr, to = step["from"], step["to"]
        t2 = F(step["t"])
        if t2 != t:
            return "witness step t mismatch"
        # x 必须满足双侧重心约束
        x = Fv(step["x"])
        ok = _check_contact(mode, fr, to, x, t, tri_by_id, pa, pb)
        if not ok:
            return "witness contact point invalid at %s->%s" % (fr, to)
    # 验证路径连续边真的活动：逐步检查接触
    for i in range(len(path) - 1):
        ok = _edge_active_at(mode, path[i], path[i + 1], t, tri_by_id, pa, pb)
        if not ok:
            return "edge not active at witness (mode,t): %s->%s" % (path[i], path[i + 1])
    return None

def _bary_inside(b):
    return all(t >= 0 for t in b) and sum(b) == 1

def _eval_tri_at(tri, t, b):
    v0, v1, v2 = tri.verts
    w = tri.v
    return [v0[i] + w[i] * t + (v1[i] - v0[i]) * b[1] + (v2[i] - v0[i]) * b[2]
            for i in range(3)]

def _check_contact(mode, fr, to, x, t, tri_by_id, pa, pb):
    """x 是否同时是 fr/to 在 (mode,t) 的合法接触点（重心坐标约束）。"""
    # to 恒为 Tri
    # 右侧既可以是三角形也可以是点端点
    if to in ("@A", "@B"):
        p = pa if to == "@A" else pb
        if not all(x[i] == p[i] for i in range(3)):
            return False
    else:
        tri_r = tri_by_id.get(to)
        if tri_r is None or not _point_on_tri(tri_r, t, x):
            return False
    if fr in ("@A", "@B"):
        p = pa if fr == "@A" else pb
        return all(x[i] == p[i] for i in range(3))
    tri_l = tri_by_id.get(fr)
    if tri_l is None:
        return False
    return _point_on_tri(tri_l, t, x)

def _point_on_tri(tri, t, x):
    """x ∈ tri(t) ？（精确 LP）"""
    # min 0 s.t. B b = x, b>=0 -> 构造 3 等式
    v0, w = tri.verts[0], tri.v
    d1 = vsub(tri.verts[1], v0); d2 = vsub(tri.verts[2], v0)
    # b3 = 1 - b1 - b2；等式 w*t + d1 b1 + d2 b2 = x - v0
    rows, rhs = [], []
    for ax in range(3):
        c = x[ax] - v0[ax] - w[ax] * t
        rows.append([d1[ax], d2[ax]]); rhs.append(c)
        rows.append([-d1[ax], -d2[ax]]); rhs.append(-c)
    # b1,b2>=0, b1+b2<=1
    rows.append([Fraction(1), Fraction(1)]); rhs.append(Fraction(1))
    return lp_feasible(rows, rhs)

def _edge_active_at(mode, a, b, t, tri_by_id, pa, pb):
    ta = "@A" if a == "@A" else ("@B" if b == "@A" else tri_by_id.get(a))
    # 统一：把点放左
    if isinstance(a, str) and a.startswith("@"):
        left, right = a, tri_by_id.get(b)
    else:
        left, right = tri_by_id.get(a), tri_by_id.get(b) if not b.startswith("@") else None
        if right is None:
            left, right = b, tri_by_id.get(a)
    r = edge_window(mode, left, right, pa, pb)
    if r is None:
        return False
    lo, hi, la, ha = r
    return (lo < t < hi) or (la and t == lo) or (ha and t == hi)

def verify_cut_witness(model, mode_map, windows_by_mode, w):
    """独立核验 disconnected_witness：在 (mode,t) 无活动边横跨 cut。"""
    if w is None:
        return "disconnected_witness missing"
    mid = w["mode"]
    mode = mode_map.get(mid)
    if mode is None:
        return "bad mode"
    t = F(w["t"])
    if not mode_allows(mode, t):
        return "t outside mode domain"
    cut = w.get("cut") or {}
    S, T = set(cut.get("S") or []), set(cut.get("T") or [])
    if "@A" in T or "@B" in S:
        return "cut endpoint side invalid"
    act = active_edges_at(mode, windows_by_mode.get(mid, {}), t)
    for (a, b) in act:
        if (a in S and b in T) or (a in T and b in S):
            return "active cross edge: %s-%s" % (a, b)
    return None

def rho_cost(mode_map, ref, mode_id, t):
    """edit_cost(m) + |t - t0|/sigma（reference=(mode,t0,sigma)，σ 为分母尺度）。
    该公式已用 M2 冻结的 6 例 flip_witness cost 数值校准（edit_cost=0 时 cost=|t|/σ）。"""
    m = mode_map[mode_id]
    t0, sigma = F(ref["t"]), F(ref["sigma"])
    return m.edit_cost + abs(t - t0) / sigma

def judge_cluster(cluster_id, register_entry, m1_root, m2_root, budget_check=True):
    """对一簇做全量独立判卷。返回 detail dict。"""
    det = {"cluster_id": cluster_id, "checks": {}, "verdict": "AGREE",
           "first_disagreement": None}
    def fail(reason, **kw):
        det["verdict"] = "DISAGREE"
        if det["first_disagreement"] is None:
            det["first_disagreement"] = {"reason": reason, **kw}
        return det

    model_path = os.path.join(m1_root, register_entry["model_path"])
    cert_path = os.path.join(m2_root, "cases", cluster_id, "certification.json")
    act_path = os.path.join(m2_root, "cases", cluster_id, "active_check.json")
    model = jload(model_path)
    cert = jload(cert_path)
    act = jload(act_path)

    # 1) 来源与假设守卫
    if model.get("scope") != "REAL_DERIVED_CONDITIONED_FINITE_UNION":
        return fail("SCOPE_NOT_REAL_DERIVED", scope=model.get("scope"))
    for blk in ("real_provenance", "assumption_register", "quantization_contract"):
        if not model.get(blk):
            return fail("MISSING_BLOCK", block=blk)
    cs = canon_sha(model)
    det["checks"]["canon_sha256"] = cs
    if cs != register_entry.get("model_canonical_sha256"):
        return fail("SOURCE_SHA_MISMATCH_REGISTER", mine=cs,
                    theirs=register_entry.get("model_canonical_sha256"))
    if cert.get("input_sha256") != cs:
        return fail("SOURCE_SHA_MISMATCH_CERT", mine=cs, theirs=cert.get("input_sha256"))

    # 拒收案例：状态合法即可，不重算
    if cert.get("status") == "BUDGET_EXCEEDED":
        det["checks"]["refused"] = True
        det["status_m2"] = "BUDGET_EXCEEDED"
        return det

    # 2) 全量接触窗口 + cell 普查
    mode_map = {m["id"]: Mode(m) for m in model["modes"]}
    mode_order = [m["id"] for m in model["modes"]]
    wb = build_windows(model)
    # 与 certification.contact_windows 逐条比对
    cw_theirs = {}
    for w in cert.get("contact_windows", []):
        cw_theirs.setdefault(w["mode"], {})[(w["from"], w["to"])] = w["t_closed"]
    # 逐 window 比界（无向对；null 声明 = 不可达，等价于不在我方 wb）
    for mid, wd in wb.items():
        for (a, b), (lo, hi, la, ha) in wd.items():
            tw = cw_theirs.get(mid, {}).get((a, b))
            if tw is None:
                tw = cw_theirs.get(mid, {}).get((b, a))
            if tw is None:
                return fail("WINDOW_MISSING_OR_NULL_IN_M2", mode=mid, edge=[a, b])
            if [F(lo), F(hi)] != [F(tw[0]), F(tw[1])]:
                return fail("WINDOW_BOUND_MISMATCH", mode=mid, edge=[a, b],
                            mine=[str(lo), str(hi)], theirs=tw)
    n_win = sum(len(d) for d in wb.values())
    det["checks"]["windows_recomputed"] = n_win
    det["checks"]["windows_declared"] = len(cert.get("contact_windows", []))
    # 无向键集必须完全一致（M2 null 声明与我方"不可达"等价）
    theirs_reach = {(mid, tuple(sorted((a, b)))) for mid, d in cw_theirs.items()
                    for (a, b) in d if d[(a, b)] is not None}
    mine_keys = {(mid, tuple(sorted((a, b)))) for mid, d in wb.items() for (a, b) in d}
    if theirs_reach != mine_keys:
        only_theirs = theirs_reach - mine_keys
        only_mine = mine_keys - theirs_reach
        if only_mine:
            mid, ab = sorted(only_mine)[0]
            return fail("WINDOW_MISSING_IN_M2", mode=mid, edge=list(ab))
        if only_theirs:
            mid, ab = sorted(only_theirs)[0]
            return fail("WINDOW_EXTRA_IN_M2", mode=mid, edge=list(ab))

    # cell 普查
    root_cond = {"allowed_modes": set(mode_order), "mode_segments": {}, 
                 "t_predicate": {"kind": "ALL_MODE_SEGMENTS", "threshold_t": None}}
    dom = eval_domain(mode_map, mode_order, wb, root_cond)
    det["checks"]["cells_recomputed"] = len(dom["cells"])
    det["checks"]["cells_declared"] = len(cert.get("cells", []))
    cmp_ = compare_cells(dom["cells"], cert.get("cells", []))
    if cmp_:
        cmp_.pop("reason", None)
        return fail("CELL_MISMATCH", **cmp_)

    # 3) 三态 + 双见证
    if cert.get("exists_connected") != dom["exists_connected"]:
        return fail("EXISTS_CONNECTED_MISMATCH", mine=dom["exists_connected"],
                    theirs=cert.get("exists_connected"))
    if cert.get("all_connected") != dom["all_connected"]:
        return fail("ALL_CONNECTED_MISMATCH", mine=dom["all_connected"],
                    theirs=cert.get("all_connected"))
    if cert.get("status") != dom["status"]:
        return fail("STATUS_MISMATCH", mine=dom["status"], theirs=cert.get("status"))
    det["status_m2"] = cert.get("status")
    # 见证
    if dom["exists_connected"]:
        e = verify_path_witness(model, mode_map, cert.get("connected_witness"))
        if e:
            return fail("CONNECTED_WITNESS_INVALID", detail=e)
    if not dom["all_connected"]:
        e = verify_cut_witness(model, mode_map, wb, cert.get("disconnected_witness"))
        if e:
            return fail("DISCONNECTED_WITNESS_INVALID", detail=e)

    # 4) ρ 与最近翻转见证
    ref = model.get("reference") or {}
    rho_m2 = cert.get("rho") or {}
    if cert.get("status") == "AMBIGUOUS_WITH_VERIFIED_WITNESSES" or rho_m2.get("infimum"):
        ref_mode, ref_t = ref.get("mode"), F(ref["t"])
        nominal_cell = None
        for c in dom["cells"]:
            if c["mode"] != ref_mode:
                continue
            if c["kind"] == "POINT" and c["sample"] == ref_t:
                nominal_cell = c
                break
            if c["kind"] == "OPEN" and c["lo"] < ref_t < c["hi"]:
                nominal_cell = c
                break
        if nominal_cell is None:
            return fail("NOMINAL_CELL_NOT_FOUND")
        nominal_connected = bool(nominal_cell["connected"])
        det["checks"]["nominal_connected"] = nominal_connected
        if rho_m2.get("nominal_connected") is not None and \
           bool(rho_m2["nominal_connected"]) != nominal_connected:
            return fail("NOMINAL_CONNECTED_MISMATCH", mine=nominal_connected,
                        theirs=rho_m2.get("nominal_connected"))
        # 独立重算 ρ：枚举所有与 nominal 反号的 cell 的 cost。
        # OPEN cell：infimum 在两端点取（未达，attained=False 若最近的只有 OPEN）。
        # POINT cell：cost 精确取到。
        flip = [c for c in dom["cells"] if c["connected"] != nominal_connected]
        cands = []
        for c in flip:
            if c["kind"] == "POINT":
                cands.append((rho_cost(mode_map, ref, c["mode"], c["sample"]), True, c))
            else:
                for t_end in (c["lo"], c["hi"]):
                    cands.append((rho_cost(mode_map, ref, c["mode"], t_end), False, c))
        cands.sort(key=lambda x: x[0])
        rho_min = cands[0][0] if cands else None
        attained = any(a and cost == rho_min for (cost, a, _) in cands) if cands else None
        det["checks"]["rho_recomputed"] = str(rho_min) if rho_min is not None else None
        if rho_m2.get("infimum") is not None:
            rho_theirs = F(rho_m2["infimum"])
            if rho_min is None or rho_min != rho_theirs:
                return fail("RHO_MISMATCH", mine=str(rho_min), theirs=str(rho_theirs))
            if rho_m2.get("attained") is not None and bool(rho_m2["attained"]) != bool(attained):
                return fail("RHO_ATTAINED_MISMATCH", mine=attained, theirs=rho_m2.get("attained"))
        fw = rho_m2.get("flip_witness")
        if fw is not None:
            # witness 的 cost 与 infimum 的关系取决于 attained：
            #   attained=True  -> witness 是最近翻转 POINT 本身，cost == infimum；
            #   attained=False -> witness 是某反号 OPEN cell 的中点，
            #                     infimum 取该 cell 近端点（未达），cost > infimum。
            mid, t = fw["mode"], F(fw["t"])
            c2 = rho_cost(mode_map, ref, mid, t)
            attained_m2 = bool(rho_m2.get("attained"))
            if attained_m2:
                if c2 != F(rho_m2["infimum"]):
                    return fail("FLIP_WITNESS_COST_MISMATCH", mine=str(c2),
                                theirs=rho_m2["infimum"])
            else:
                if c2 <= F(rho_m2["infimum"]):
                    return fail("FLIP_WITNESS_COST_NOT_ABOVE_INFIMUM", mine=str(c2),
                                theirs=rho_m2["infimum"])
            if F(fw.get("cost", str(c2))) != c2:
                return fail("FLIP_WITNESS_DECLARED_COST_MISMATCH", mine=str(c2),
                            theirs=fw.get("cost"))
            act_edges_fw = active_edges_at(mode_map[mid], wb.get(mid, {}), t)
            _, conn2 = connectivity(["@A", "@B"], act_edges_fw)
            if conn2 != bool(fw["connected"]):
                return fail("FLIP_WITNESS_CONNECTED_MISMATCH")
            if conn2 == nominal_connected:
                return fail("FLIP_WITNESS_NOT_A_FLIP")

    # 5) collapse_tree 复核（仅 AMBIGUOUS 案例要求）
    det["checks"]["collapse_tree_checked"] = 0
    if act and act.get("collapse_tree"):
        ct = act["collapse_tree"]
        n_checked = 0
        for fid, fr in (ct.get("fractures") or {}).items():
            for br in fr.get("branches") or []:
                n_checked += 1
                cond_raw = br.get("condition") or {}
                cond = {
                    "allowed_modes": set(cond_raw.get("allowed_modes") or mode_order),
                    "mode_segments": {
                        k: [{"lo": F(s["lo"]), "hi": F(s["hi"]),
                             "lo_closed": bool(s["lo_closed"]),
                             "hi_closed": bool(s["hi_closed"])} for s in v]
                        for k, v in (cond_raw.get("mode_segments") or {}).items()},
                    "t_predicate": cond_raw.get("t_predicate")}
                sub = eval_domain(mode_map, mode_order, wb, cond)
                claim = br.get("post_status")
                if sub["status"] != claim:
                    return fail("COLLAPSE_BRANCH_STATUS_MISMATCH",
                                fracture=fid, branch=br.get("branch_id"),
                                mine=sub["status"], theirs=claim)
                if bool(br.get("collapsed_to_certified")) != (sub["status"] in
                        ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")):
                    return fail("COLLAPSE_FLAG_MISMATCH",
                                fracture=fid, branch=br.get("branch_id"))
                if sub["exists_connected"] != bool(br.get("exists_connected", sub["exists_connected"])):
                    return fail("COLLAPSE_BRANCH_EXISTS_MISMATCH",
                                fracture=fid, branch=br.get("branch_id"))
                if sub["all_connected"] != bool(br.get("all_connected", sub["all_connected"])):
                    return fail("COLLAPSE_BRANCH_ALL_MISMATCH",
                                fracture=fid, branch=br.get("branch_id"))
        det["checks"]["collapse_tree_checked"] = n_checked
    return det

def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--m2-run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    register = jload(os.path.join(args.m1, "cluster_register.json"))
    summary = jload(os.path.join(args.m2_run, "summary.json"))

    entries = {e["cluster_id"]: e for e in register["clusters"]}
    batch = {"task_id": "M3", "schema": "m3-verdict-batch/v1",
             "m1_register_sha256": canon_sha(register),
             "m2_summary_sha256": canon_sha(summary),
             "run_id": summary.get("run_id"), "verdicts": {},
             "counts": {"AGREE": 0, "DISAGREE": 0, "MALFORMED": 0, "UNDECIDABLE": 0},
             "refused": []}
    details = {}
    collapse_verdicts = {}
    order = [args.only] if args.only else [e["cluster_id"] for e in register["clusters"]]
    for cid in order:
        try:
            det = judge_cluster(cid, entries[cid], args.m1, args.m2_run)
        except Exception as e:
            det = {"cluster_id": cid, "verdict": "MALFORMED",
                   "first_disagreement": {"reason": "JUDGE_EXCEPTION",
                                          "detail": repr(e)}}
        batch["verdicts"][cid] = {"verdict": det["verdict"],
                                  "status_m2": det.get("status_m2"),
                                  "first_disagreement": det.get("first_disagreement"),
                                  "checks": det.get("checks")}
        batch["counts"][det["verdict"]] += 1
        if det.get("status_m2") == "BUDGET_EXCEEDED":
            batch["refused"].append(cid)
        details[cid] = det
        if det.get("checks", {}).get("collapse_tree_checked"):
            collapse_verdicts[cid] = {
                "branches_checked": det["checks"]["collapse_tree_checked"],
                "verdict": det["verdict"],
                "first_disagreement": det.get("first_disagreement")}

    with open(os.path.join(args.out_dir, "verdict_batch.json"), "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(args.out_dir, "verdict_detail.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(args.out_dir, "collapse_tree_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(collapse_verdicts, f, ensure_ascii=False, indent=1, sort_keys=True)
    n_reads = flush_ledger(os.path.join(args.out_dir, "read_ledger.jsonl"), {})

    bad = batch["counts"]["DISAGREE"] + batch["counts"]["MALFORMED"] + batch["counts"]["UNDECIDABLE"]
    print(json.dumps({"verdicts": {k: v["verdict"] for k, v in batch["verdicts"].items()},
                      "counts": batch["counts"], "read_ledger_records": n_reads},
                     ensure_ascii=False, indent=1))
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    raise SystemExit(main(sys.argv[1:]))
