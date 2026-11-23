# -*- coding: utf-8 -*-
"""W90 · 两点贯通 QoI 仪器(新主线 estimand 的唯一合法计算入口)。

estimand(首期唯一, 技术方案 §1.1)
----------------------------------
C(A, B; F) ∈ {0, 1}:给定域内两点 A、B 与圆盘网络 F = {(c_i, n_i, r_i)},
存在裂隙序列 i_1..i_m 使 A ∈ attach(i_1)、B ∈ attach(i_m),且相邻圆盘
**真判交**(``intersect_disk.disk_pair_intersect``,严格不等式,非球近似)。
``attach(i)`` = 点到圆盘 i 的距离 ≤ ``r_attach`` 的捕获规则。

判交核逐字搬运自 ``scripts/w15_intersect.py``(J0.4 血统:两文件 sha256 逐字节
相同),本模块只做"图 + 端点捕获 + 查询/频率/子盒"的仪器外壳,零几何重推导。

设计选择(全部显式预注册, ``results/w90_connectivity_qoi_instrument/prereg.json``)
--------------------------------------------------------------------------------
* ``r_attach`` 是必须显式预注册的工程特征尺度(一段钻孔间隔或一个巷道轮廓),
  **API 不设默认值**,调用方必须显式传入,禁调参救结论。
* 判交取严格 ``<``(W15 政策,子集定理严格成立);捕获取含等号 ``≤``——
  两者是不同算子,政策不同,不得混谈。
* 局部子盒:盘 i 入盒 ⟺ center ∈ box ⊕ r_max(各轴外扩 r_max,使任何能触及
  盒体的盘都被纳入)。子盒外的连接被截断 ⇒ 子盒读数是 C 的**下界性质**,
  ``subbox_of`` 必须随行返回被截断的连接数(技术方案 §3-G3)。
* W24 死结消解:本 estimand 是局部两点贯通,不是墙-墙 spanning
  (wp_register#WP-0.notes);W24 两教训(电导归一化 / Dirichlet 全局抵消)
  归 Phase 2 水力层,本模块不消费电导。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from fractureflow.intersect_disk import (  # 逐字搬运的判交核(J0.4 血统)
    W15InputError,
    build_disk_graph,
    check_inputs,
)

__all__ = [
    "attach_points",
    "two_point_connected",
    "pair_connection_batch",
    "connection_frequency",
    "subbox_of",
    "disk_point_distance",
    "W15InputError",
]

_WILSON_Z = 1.959963984540054  # 95% 双侧,prereg#wilson_z 锁定


# ---------------------------------------------------------------------------
# 基元:点到圆盘集合的距离与捕获
# ---------------------------------------------------------------------------

def disk_point_distance(centers, normals, radii, P) -> np.ndarray:
    """逐盘点到圆盘集合的距离(闭式,prereg#attach_rule)。

    dist(p, D_i) = sqrt( vn² + max(0, d_planar − r_i)² ),
    vn = (p − c_i)·n_i,d_planar = |(p − c_i) − vn·n_i|。
    投影落盘内 ⇒ 距离 = |法向分量|;投影落盘外 ⇒ 补上径向缺口。

    返回 (K, M) 数组。归一化检查由调用方(attach_points)统一做。
    """
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    if P.ndim != 2 or P.shape[1] != 3:
        raise W15InputError(f"端点形状非法: {P.shape}(应为 (K,3) 或 (3,))")
    if not np.all(np.isfinite(P)):
        raise W15InputError("端点含 NaN/inf:本仪器禁静默传播")
    v = P[:, None, :] - c[None, :, :]                     # (K, M, 3)
    vn = np.einsum("kmj,mj->km", v, n)                    # 法向分量
    v_planar = v - vn[:, :, None] * n[None, :, :]
    d_planar = np.linalg.norm(v_planar, axis=2)
    gap = np.maximum(d_planar - r[None, :], 0.0)
    return np.sqrt(vn * vn + gap * gap)


def attach_points(centers, normals, radii, P, r_attach: float):
    """点-盘捕获邻接阵(csr,(K 点 × M 盘),0/1 float32)。

    attach ⟺ dist(点, 盘) ≤ r_attach(含等号)。候选剪枝:|P − c_i| ≤
    r_attach + r_i(三角不等式,无损),再逐候选做闭式距离精确过滤。
    """
    check_inputs(centers, normals, radii)
    if not (np.isscalar(r_attach) and np.isfinite(r_attach) and r_attach >= 0.0):
        raise W15InputError(f"r_attach 必须是非负有限标量且显式传入,收到 {r_attach!r}")
    c = np.asarray(centers, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    K = P.shape[0]
    M = c.shape[0]
    if M == 0 or K == 0:
        return sparse.csr_matrix((K, M), dtype=np.float32)
    # 候选剪枝:盘心到点距离 ≤ r_attach + r_i ≤ r_attach + max(r)
    tree = cKDTree(c)
    cand = tree.query_ball_point(P, r_attach + float(r.max()))
    rows, cols = [], []
    for k, idx in enumerate(cand):
        if not idx:
            continue
        idx_arr = np.asarray(idx, dtype=np.intp)
        d = disk_point_distance(c[idx_arr], normals[idx_arr], r[idx_arr], P[k])[0]
        hit = idx_arr[d <= r_attach]
        if hit.size:
            rows.append(np.full(hit.size, k, dtype=np.intp))
            cols.append(hit)
    if not rows:
        return sparse.csr_matrix((K, M), dtype=np.float32)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(K, M))


# ---------------------------------------------------------------------------
# 子盒
# ---------------------------------------------------------------------------

def _as_box(subbox) -> Tuple[np.ndarray, np.ndarray]:
    """接受 (lo, hi) 数组对或 subbox_of 返回的 dict。"""
    if isinstance(subbox, dict):
        lo = np.asarray(subbox["lo"], dtype=np.float64)
        hi = np.asarray(subbox["hi"], dtype=np.float64)
    else:
        arr = np.asarray(subbox, dtype=np.float64)
        if arr.shape != (2, 3):
            raise W15InputError(f"subbox 形状非法: {arr.shape}(应为 (2,3) 或 subbox_of 返回值)")
        lo, hi = arr[0], arr[1]
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise W15InputError("subbox 含 NaN/inf")
    if np.any(hi < lo):
        raise W15InputError("subbox hi < lo")
    return lo, hi


def _included_mask(centers, lo: np.ndarray, hi: np.ndarray, r_max: float) -> np.ndarray:
    """入盒判据:center ∈ box ⊕ r_max(prereg#subbox_inclusion)。"""
    return np.all((centers >= lo - r_max) & (centers <= hi + r_max), axis=1)


def subbox_of(A, B, r_max: float, margin: float, *,
              centers=None, radii=None, true_graph=None) -> Dict:
    """构造 A、B 的局部评估子盒 + 截断连接数随行(技术方案 §3-G3)。

    盒规则:包含 A、B 的最小轴对齐盒各侧 pad ``margin``,再对称增长至每轴
    边长 ≥ 2·|AB| + 2·r_max(G3 下界规则)。

    截断读数(给 ``centers``/``radii`` 才计算):
    * ``true_graph``(全域真判交图)给定时 → ``n_cut`` 为精确值;
    * 否则 → 球近似候选对跨盒计数,记 ``cut_level="candidate_sphere_upper_bound"``,
      按子集定理 E_true ⊆ E_sphere 它是 n_cut 的**上界**。

    返回 dict:lo/hi/ab_distance/side_lengths/… + 截断段。
    """
    A = np.asarray(A, dtype=np.float64).reshape(3)
    B = np.asarray(B, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        raise W15InputError("端点含 NaN/inf")
    if not (np.isscalar(r_max) and np.isfinite(r_max) and r_max >= 0.0):
        raise W15InputError(f"r_max 非法: {r_max!r}")
    if not (np.isscalar(margin) and np.isfinite(margin) and margin >= 0.0):
        raise W15InputError(f"margin 非法: {margin!r}")
    ab = float(np.linalg.norm(B - A))
    lo0 = np.minimum(A, B)
    hi0 = np.maximum(A, B)
    lo = lo0 - margin
    hi = hi0 + margin
    min_side = 2.0 * ab + 2.0 * r_max
    center = 0.5 * (lo + hi)
    side = hi - lo
    grew = (side < min_side).tolist()
    side_grown = np.where(grew, min_side, side)
    lo = np.where(grew, center - 0.5 * side_grown, lo)   # 增长以盒心对称, 严格包含 pad 盒
    hi = np.where(grew, center + 0.5 * side_grown, hi)

    out: Dict = {
        "lo": lo,
        "hi": hi,
        "ab_distance": ab,
        "r_max": float(r_max),
        "margin": float(margin),
        "min_side_rule": float(min_side),
        "side_lengths": hi - lo,
        "grown_to_min_side": grew,
    }
    if centers is None:
        return out
    c = np.asarray(centers, dtype=np.float64)
    if radii is not None:
        r = np.asarray(radii, dtype=np.float64)
        r_max_eff = float(r.max()) if r.size else 0.0
        out["r_max_covers_radii"] = bool(r_max >= r_max_eff)
        if not out["r_max_covers_radii"]:
            raise W15InputError(
                f"r_max={r_max} 小于数据最大半径 {r_max_eff}:入盒判据失效,拒绝出读数")
    inc = _included_mask(c, lo, hi, r_max)
    out["n_disks_total"] = int(c.shape[0])
    out["n_disks_included"] = int(np.count_nonzero(inc))
    out["n_disks_excluded"] = int(c.shape[0] - out["n_disks_included"])
    if true_graph is not None:
        coo = true_graph.tocoo()
        i, j = coo.row, coo.col
        keep = i < j
        i, j = i[keep], j[keep]
        cut = int(np.count_nonzero(inc[i] != inc[j]))
        out["n_cut"] = cut
        out["cut_level"] = "true_graph_exact"
        out["n_edges_full"] = int(i.size)
    elif radii is not None:
        # 子集定理:E_true ⊆ E_sphere ⇒ 跨盒候选对数是精确截断数的上界
        r = np.asarray(radii, dtype=np.float64)
        r_search = 2.0 * float(r.max()) if r.size else 0.0
        ti = cKDTree(c[inc])
        te = cKDTree(c[~inc])
        n_cut_cand = 0
        if inc.any() and (~inc).any() and r_search > 0.0:
            for idx_i in ti.query_ball_tree(te, r_search):
                n_cut_cand += len(idx_i)
        out["n_cut"] = int(n_cut_cand)
        out["cut_level"] = "candidate_sphere_upper_bound"
    return out


# ---------------------------------------------------------------------------
# 两点贯通查询
# ---------------------------------------------------------------------------

def _restrict_subbox(centers, normals, radii, subbox) -> Tuple[np.ndarray, ...]:
    lo, hi = _as_box(subbox)
    r_max = float(np.asarray(radii, dtype=np.float64).max())
    inc = _included_mask(centers, lo, hi, r_max)
    if not inc.any():
        raise W15InputError("子盒内无任何圆盘:C 下界读数为 0,请检查子盒/半径")
    return centers[inc], normals[inc], radii[inc], inc


def _attach_comp_sets(centers, normals, radii, endpoints, r_attach):
    """连通分量 + 每端点捕获盘的 comp 集合。返回 (comp_ids, sets, n_comp)。"""
    G = build_disk_graph(centers, normals, radii)
    n_comp, comp_ids = csgraph.connected_components(G, directed=False)
    sets = []
    if endpoints is not None and len(endpoints) > 0:
        att = attach_points(centers, normals, radii,
                            np.asarray(endpoints, dtype=np.float64), r_attach)
        for k in range(att.shape[0]):
            disks = att.indices[att.indptr[k]:att.indptr[k + 1]]
            sets.append(frozenset(comp_ids[disks].tolist()) if disks.size else frozenset())
    return comp_ids, sets, n_comp


def two_point_connected(centers, normals, radii, A, B, *, r_attach: float,
                        subbox=None) -> bool:
    """C(A,B) 单对查询(bool)。

    ``subbox=None`` 全域评估;给定时只在局部子盒内评估(下界性质,截断读数
    由 ``subbox_of`` 随行,本函数保持纯布尔)。
    """
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    check_inputs(c, n, r)
    if subbox is not None:
        c, n, r, _ = _restrict_subbox(c, n, r, subbox)
    _, sets, _ = _attach_comp_sets(
        c, n, r, [np.asarray(A, dtype=np.float64).reshape(3),
                  np.asarray(B, dtype=np.float64).reshape(3)], r_attach)
    return _sets_connected(sets[0], sets[1])


def _sets_connected(set_a: frozenset, set_b: frozenset) -> bool:
    if not set_a or not set_b:
        return False
    return not set_a.isdisjoint(set_b)


def pair_connection_batch(centers, normals, radii, pairs, *, r_attach: float,
                          subbox=None,
                          subboxes: Optional[Sequence] = None) -> np.ndarray:
    """多对批量查询 → np.ndarray[bool]。

    同一子盒(或全域)只建一次图;``subboxes`` 给定时逐对子盒,按盒分组建图。
    端点盘必须落在(或等于全域)入盒集内,否则显式报错。
    """
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    check_inputs(c, n, r)
    pairs = np.asarray(pairs, dtype=np.intp)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise W15InputError(f"pairs 形状非法: {pairs.shape}(应为 (P,2))")
    out = np.zeros(pairs.shape[0], dtype=bool)
    if pairs.shape[0] == 0:
        return out

    if subboxes is not None:
        if len(subboxes) != pairs.shape[0]:
            raise W15InputError("subboxes 长度须与 pairs 一致")
        groups: Dict[bytes, list] = {}
        for k, sb in enumerate(subboxes):
            key = b"full" if sb is None else np.asarray(
                _as_box(sb), dtype=np.float64).tobytes()
            groups.setdefault(key, []).append(k)
        for key, idxs in groups.items():
            sb = None if key == b"full" else np.frombuffer(key, dtype=np.float64).reshape(2, 3)
            out[np.asarray(idxs, dtype=np.intp)] = pair_connection_batch(
                c, n, r, pairs[np.asarray(idxs, dtype=np.intp)],
                r_attach=r_attach, subbox=sb)
        return out

    if subbox is not None:
        c, n, r, inc = _restrict_subbox(c, n, r, subbox)
        remap = np.full(inc.size, -1, dtype=np.intp)
        remap[inc] = np.arange(int(inc.sum()), dtype=np.intp)
        local_pairs = remap[pairs]
        if np.any(local_pairs < 0):
            raise W15InputError("存在端点盘不在子盒入盒集内:端点盘必须先入盒")
        pairs_use = local_pairs
    else:
        pairs_use = pairs

    P = np.stack([c[pairs_use[:, 0]], c[pairs_use[:, 1]]], axis=1).reshape(-1, 3)  # (2P,3)
    G = build_disk_graph(c, n, r)
    _, comp_ids = csgraph.connected_components(G, directed=False)
    att = attach_points(c, n, r, P, r_attach)   # (2P, M)
    comp_sets = []
    for k in range(att.shape[0]):
        disks = att.indices[att.indptr[k]:att.indptr[k + 1]]
        comp_sets.append(comp_ids[disks] if disks.size else np.zeros(0, dtype=np.int64))
    for k in range(pairs_use.shape[0]):
        sa, sb = comp_sets[2 * k], comp_sets[2 * k + 1]
        if sa.size and sb.size:
            out[k] = bool(np.intersect1d(sa, sb).size > 0)
    return out


# ---------------------------------------------------------------------------
# 集合频率(WP-1/2 的 p̂ 机制;本单只建仪器 + 冒烟)
# ---------------------------------------------------------------------------

def connection_frequency(ensemble, A, B, *, r_attach: float, subbox=None,
                         weights: Optional[Sequence] = None,
                         z: float = _WILSON_Z) -> Dict:
    """集合成员的 C(A,B) 频率 → dict(p_hat, wilson_lo, wilson_hi, ess, …)。

    p̂ 是**模型条件频率,不是概率**(§6.3-1):必随先验标识、ε、ESS——先验/ε
    由上游(WP-1)随行,本函数只负责把频率与 Wilson 区间、Kish ESS 算对。

    ``ensemble``:可迭代的 (centers, normals, radii) 三元组或含这三键的 dict。
    ``weights``:Kish ESS = (Σw)²/Σw²;等权时 ESS = n,Wilson 用 n。
    加权时 Wilson 以 Kish ESS 作有效 n 的近似区间,``wilson_basis`` 字段如实标注。
    """
    members = []
    for m in ensemble:
        if isinstance(m, dict):
            members.append((m["centers"], m["normals"], m["radii"]))
        else:
            members.append(m)
    if not members:
        raise W15InputError("ensemble 为空:拒绝空对空频率(军规⑭)")
    flags = np.zeros(len(members), dtype=bool)
    for k, (c, n, r) in enumerate(members):
        flags[k] = two_point_connected(c, n, r, A, B, r_attach=r_attach, subbox=subbox)
    n_mem = len(members)
    if weights is None:
        w = np.ones(n_mem, dtype=np.float64)
        ess = float(n_mem)
        wilson_basis = "n_members"
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_mem,) or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise W15InputError("weights 形状非法或含负/非有限值")
        sw = float(w.sum())
        if sw <= 0:
            raise W15InputError("weights 全零:Kish ESS 无定义")
        ess = float(sw * sw / np.square(w).sum())
        wilson_basis = "kish_ess_approx"
    p_hat = float(np.average(flags.astype(np.float64), weights=w))
    if ess > 0:
        z2 = z * z
        denom = 1.0 + z2 / ess
        center = (p_hat + z2 / (2.0 * ess)) / denom
        half = z * np.sqrt(p_hat * (1.0 - p_hat) / ess + z2 / (4.0 * ess * ess)) / denom
        lo = float(max(0.0, center - half))
        hi = float(min(1.0, center + half))
    else:
        lo = hi = 0.0
    return {
        "p_hat": p_hat,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "ess": ess,
        "n_members": n_mem,
        "n_connected": int(np.count_nonzero(flags)),
        "wilson_z": float(z),
        "wilson_basis": wilson_basis,
        "flags": [bool(x) for x in flags],
    }
