"""W15 · 方向感知的真实圆盘-圆盘相交判交仪器（独立实现，**不改** src/fractureflow/dfn.py）。

现役 ``fractureflow.dfn.build_connectivity_graph`` 的判交条件是
``|c_i - c_j| < r_i + r_j``（dfn.py:282 定义，快速路径 :357-358 / 回退 :381-383），
即把圆盘当球体的排除体积近似；该函数不消费 ``dfn.normals``，故对产状误差结构性不可见。
本模块另造一件**方向感知**的判交仪器，让"产状误差是否改变连通/渗流"变得可测。

几何命题（解析依据，非经验阈值）
--------------------------------
圆盘 D_i = { x ∈ R^3 : (x - c_i)·n_i = 0, |x - c_i| ≤ r_i }，n_i 为单位法向。

**子集定理**（本仪器的结构不变量，也是实现正确性的可判红闸）：
若 D_i ∩ D_j ≠ ∅，取公共点 p，则
    |c_i - c_j| ≤ |c_i - p| + |p - c_j| ≤ r_i + r_j。
故真相交边集 E_new ⊆ 球近似边集 E_old = { |c_i-c_j| < r_i + r_j }。
推论：同一实现下 新图边数 ≤ 旧图边数，且 p32_crit(new) ≥ p32_crit(old)（单调）。
⇒ 候选对生成沿用旧仪器的搜索半径 ``2·max(radii)`` 即**完备**（无近似损失），
  且两图之差可归因于取向本身而非候选集不同。

分支与推导
----------
φ = n_i 与 n_j 的夹角，sinφ = |n_i × n_j|。

(A) 平行分支 sinφ ≤ τ_sin：
    A1 共面 |Δc·n_i| ≤ τ_len：两圆盘在同一平面内 ⟺ 面内距离 < r_i + r_j。
        （包含情形 d < |r_i − r_j| 亦被同一不等式覆盖：被含圆盘上每点都属于交集，
          特别地圆心距最小者的边界点在另一圆盘内，非空。）
    A2 严格异面：两平面无公共点 ⇒ 圆盘无公共点 ⇒ 不相交。

(B) 相交分支 sinφ > τ_sin：两平面交于唯一直线 L，方向 t̂ = (n_i × n_j)/sinφ。
    每个圆盘与 L 的交是其上的一条弦（可能空），两圆盘相交 ⟺ 两弦重叠：
        |Δs| < ℓ_i + ℓ_j，  ℓ_k = sqrt(r_k² − u_k²)，  且 u_i < r_i、u_j < r_j
    其中 u_k = 圆心到 L 的距离**在其自身平面内**，Δs = 两圆心在 t̂ 上的投影差。
    关键：无需显式求 L 上的点（避免 1/sin²φ 的 p0 表达式在掠射时发散）：
        u_i = (Δc·n_j)/sinφ，  u_j = (Δc·n_i)/sinφ，  Δs = −Δc·t̂，  Δc = c_j − c_i
    推导：m̂_i := n_i × t̂ 是单位向量（t̂ ⊥ n_i ⇒ |n_i × t̂| = 1）且落在 P_i 内、垂直于 L。
    对 x ∈ P_i：x·n_j = c_i·n_j + ((x − c_i)·m̂_i)(m̂_i·n_j)，而
    m̂_i·n_j = (n_i × t̂)·n_j = t̂·(n_j × n_i) = −sinφ。L ⊂ P_j 上的点满足 x·n_j = h_j := c_j·n_j，
    解出 (c_i − x)·m̂_i = (h_j − c_i·n_j)/sinφ = (Δc·n_j)/sinφ，即 u_i（带号）。u_j 同法。

数值卫生
--------
* 退化输入显式处置，禁 NaN/inf 静默：零法向 / 非有限量 ⇒ 抛错；零半径 ⇒ 不产生边
  （点圆盘与另一圆盘相接需圆心恰落在对方圆盘内，属测度零，按严格不等式政策记无交）。
* 所有不等式取**严格** <（与旧仪器的 ``<`` 同政策），使子集定理严格成立而非依赖
  切触测度零的运气。
* τ_sin / τ_len 是**分支切换**用的数值阈而非物理参数：其取值不改变边集这件事
  由 instrument_forensics 的"阈值不敏感"实测证明（τ_sin 三档 × τ_len 三档边集逐位相同），
  且掠射构型在切换处与解析极限一致（连续性检验）。

接口与 build_connectivity_graph 对齐（dfn 入参、pbc、batch_size、domain、返回 csr_matrix），
但为独立文件、独立实现；pbc=True 明确拒绝而非静默降级（W13 口径 pbc=false，
周期边界的圆盘跨越处理属另一件事，未授权在此偷做）。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 注册常量（值 + 出处；凡决定判据读数的常量必须与判据同册登记）
# ---------------------------------------------------------------------------

#: 平行分支切换阈：sinφ ≤ TAU_SIN 走 (A)。float64 机器 eps = 2.22e-16，
#: 取 1e-8 留 8 个数量级余量；不敏感性由 forensics 实测（{1e-6,1e-8,1e-10} 边集逐位相同）。
TAU_SIN = 1e-8

#: 共面判定相对阈：|Δc·n_i| ≤ TAU_LEN_REL × scale 记共面。scale 见 _scale_of()。
TAU_LEN_REL = 1e-9

#: 逐块处理的候选对上限（内存卫生，不改变结果：块间无耦合，见 _pair_mask_chunked 文档）
PAIR_CHUNK = 4_000_000

#: 球近似搜索半径倍数：子集定理给出真相交 ⇒ |Δc| < r_i + r_j ≤ 2·max(radii)，故 2.0 完备。
SEARCH_RADIUS_MULT = 2.0


class W15InputError(ValueError):
    """本仪器拒绝的退化/非法输入（区别于几何上合法的"不相交"）。"""


def _scale_of(radii: np.ndarray, centers: np.ndarray) -> float:
    """长度容差的标尺 = 半径中位数（有正半径时），否则取坐标跨度中位数。

    为什么用中位数而非最大值：τ_len 要判的是"两个圆心到对方平面的偏离是否属于同一平面"，
    与典型圆盘尺寸同量级；个别巨无霸裂隙不应把容差整体放大。
    """
    pos = radii[radii > 0.0]
    if pos.size > 0:
        return float(np.median(pos))
    span = centers.max(axis=0) - centers.min(axis=0)
    return float(np.median(span))


def check_inputs(centers, normals, radii) -> None:
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3 or n.shape != c.shape or r.ndim != 1 or r.shape[0] != c.shape[0]:
        raise W15InputError(f"形状不匹配: centers {c.shape} normals {n.shape} radii {r.shape}")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(n)) and np.all(np.isfinite(r))):
        raise W15InputError("输入含 NaN/inf：本仪器禁静默传播，调用方须先清洗")
    if np.any(r < 0.0):
        raise W15InputError("半径不得为负")
    nrm = np.linalg.norm(n, axis=1)
    if c.shape[0] > 0 and np.any(nrm < 0.5):
        bad = int(np.argmax(nrm < 0.5))
        raise W15InputError(f"法向长度 {nrm[bad]:.6g} 非单位向量（索引 {bad}）：请先归一化")
    if c.shape[0] > 0 and np.any(np.abs(nrm - 1.0) > 1e-8):
        raise W15InputError("法向未归一（偏离 1 超过 1e-8），拒绝以近似单位向量出读数")


def disk_pair_intersect(centers, normals, radii, i_idx, j_idx, *,
                        tau_sin: float = TAU_SIN, tau_len_rel: float = TAU_LEN_REL,
                        scale: float | None = None, want_stats: bool = False):
    """对给定下标对 (i_idx, j_idx) 判两圆盘是否相交（严格不等式）。

    返回 bool 数组（长度 = 对数）。want_stats=True 时另返回逐分支计数与
    ``containment_violations``（= 真相交却违反球近似的对数，按子集定理必须为 0）。
    """
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    i_idx = np.asarray(i_idx, dtype=np.intp)
    j_idx = np.asarray(j_idx, dtype=np.intp)
    if i_idx.shape != j_idx.shape:
        raise W15InputError("i_idx 与 j_idx 长度不同")
    if i_idx.size == 0:
        empty = np.zeros(0, dtype=bool)
        if want_stats:
            return empty, {"n_pairs": 0, "n_parallel_branch": 0, "n_crossing_branch": 0,
                           "n_coplanar": 0, "n_skew_parallel": 0, "n_chord_overlap": 0,
                           "containment_violations": 0, "tau_sin": tau_sin,
                           "tau_len": 0.0, "scale": 0.0}
        return empty

    n_i = n[i_idx]
    n_j = n[j_idx]
    dc = c[j_idx] - c[i_idx]
    ri = r[i_idx]
    rj = r[j_idx]

    cos_phi = np.einsum("ij,ij->i", n_i, n_j)
    np.clip(cos_phi, -1.0, 1.0, out=cos_phi)
    cross = np.cross(n_i, n_j)
    sin_phi = np.linalg.norm(cross, axis=1)

    if scale is None:
        scale = _scale_of(r, c)
    tau_len = tau_len_rel * scale

    parallel = sin_phi <= tau_sin
    crossing = ~parallel

    out = np.zeros(i_idx.size, dtype=bool)
    # --- (A) 平行分支 ---------------------------------------------------------
    if parallel.any():
        ip = np.nonzero(parallel)[0]
        sgn = np.einsum("ij,ij->i", dc[ip], n_i[ip])      # 有号法向分量
        sep = np.abs(sgn)
        cop = sep <= tau_len
        in_plane = dc[ip] - sgn[:, None] * n_i[ip]          # 去掉法向分量（必须用有号值）
        d_planar = np.linalg.norm(in_plane, axis=1)
        hit = cop & (d_planar < ri[ip] + rj[ip])
        out[ip] = hit
    # --- (B) 相交分支：弦重叠判据 ---------------------------------------------
    if crossing.any():
        iq = np.nonzero(crossing)[0]
        s = sin_phi[iq]
        t_hat = cross[iq] / s[:, None]
        u_i = np.einsum("ij,ij->i", dc[iq], n_j[iq]) / s
        u_j = np.einsum("ij,ij->i", dc[iq], n_i[iq]) / s
        ds = np.abs(np.einsum("ij,ij->i", dc[iq], t_hat))
        ok_i = np.abs(u_i) < ri[iq]
        ok_j = np.abs(u_j) < rj[iq]
        half_i = np.sqrt(np.maximum(ri[iq] ** 2 - u_i ** 2, 0.0))
        half_j = np.sqrt(np.maximum(rj[iq] ** 2 - u_j ** 2, 0.0))
        out[iq] = ok_i & ok_j & (ds < half_i + half_j)

    if not want_stats:
        return out

    d_center = np.linalg.norm(dc, axis=1)
    viol = int(np.count_nonzero(out & ~(d_center < ri + rj)))
    stats = {
        "n_pairs": int(i_idx.size),
        "n_parallel_branch": int(parallel.sum()),
        "n_crossing_branch": int(crossing.sum()),
        "n_coplanar": 0,
        "n_skew_parallel": 0,
        "n_chord_overlap": 0,
        "containment_violations": viol,
        "tau_sin": float(tau_sin),
        "tau_len": float(tau_len),
        "scale": float(scale),
    }
    if parallel.any():
        ip = np.nonzero(parallel)[0]
        sep = np.abs(np.einsum("ij,ij->i", dc[ip], n_i[ip]))
        stats["n_coplanar"] = int(np.count_nonzero(sep <= tau_len))
        stats["n_skew_parallel"] = int(ip.size - stats["n_coplanar"])
    if crossing.any():
        stats["n_chord_overlap"] = int(np.count_nonzero(out[crossing]))
    return out, stats


_COUNTER_KEYS = ("n_pairs", "n_parallel_branch", "n_crossing_branch", "n_coplanar",
                 "n_skew_parallel", "n_chord_overlap", "containment_violations")


def _pair_mask_chunked(centers, normals, radii, pairs, *, tau_sin, tau_len_rel, scale):
    """逐块判交并把命中对累计回来。

    块间零耦合（每对独立判定）⇒ 分块不改变结果，只界住内存。
    """
    acc = {k: 0 for k in _COUNTER_KEYS}
    acc.update(tau_sin=float(tau_sin), tau_len=0.0, scale=float(scale))
    hits = []
    for st in range(0, pairs.shape[0], PAIR_CHUNK):
        blk = pairs[st:st + PAIR_CHUNK]
        m, stats = disk_pair_intersect(centers, normals, radii, blk[:, 0], blk[:, 1],
                                       tau_sin=tau_sin, tau_len_rel=tau_len_rel,
                                       scale=scale, want_stats=True)
        for k in _COUNTER_KEYS:
            acc[k] += stats[k]
        acc["tau_len"] = stats["tau_len"]
        if m.any():
            hits.append(blk[m])
    hit_pairs = (np.concatenate(hits, axis=0) if hits
                 else np.zeros((0, 2), dtype=np.intp))
    acc["n_edges"] = int(hit_pairs.shape[0])
    return hit_pairs, acc


def candidate_pairs(centers, radii, *, search_radius_mult: float = SEARCH_RADIUS_MULT):
    """球近似搜索半径内的全部候选对（i < j）。

    由子集定理，这是真相交边集的**完备候选超集**，与旧仪器 query_pairs 同集合。
    """
    c = np.asarray(centers, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if c.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.intp)
    r_max = float(r.max()) if r.size else 0.0
    if r_max <= 0.0:
        return np.zeros((0, 2), dtype=np.intp)
    tree = cKDTree(c)
    pairs = tree.query_pairs(search_radius_mult * r_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.intp)
    return np.asarray(pairs, dtype=np.intp)


def build_disk_graph(centers, normals, radii, *, tau_sin: float = TAU_SIN,
                     tau_len_rel: float = TAU_LEN_REL, want_stats: bool = False):
    """方向感知邻接阵（csr_matrix, 对称, 0/1 float32），无 PBC。"""
    check_inputs(centers, normals, radii)
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    M = c.shape[0]
    pairs = candidate_pairs(c, r)
    if pairs.shape[0] == 0:
        G = sparse.csr_matrix((M, M), dtype=np.float32)
        if want_stats:
            return G, {"n_fractures": M, "n_candidates": 0, "n_edges": 0,
                       "containment_violations": 0, "n_parallel_branch": 0,
                       "n_crossing_branch": 0, "n_coplanar": 0, "n_skew_parallel": 0,
                       "n_chord_overlap": 0, "tau_sin": float(tau_sin),
                       "tau_len": 0.0, "scale": _scale_of(r, c)}
        return G
    hit, stats = _pair_mask_chunked(c, n, r, pairs, tau_sin=tau_sin,
                                    tau_len_rel=tau_len_rel, scale=_scale_of(r, c))
    rows = np.concatenate([hit[:, 0], hit[:, 1]])
    cols = np.concatenate([hit[:, 1], hit[:, 0]])
    data = np.ones(rows.size, dtype=np.float32)
    G = sparse.csr_matrix((data, (rows, cols)), shape=(M, M))
    stats["n_fractures"] = M
    stats["n_candidates"] = int(pairs.shape[0])
    if want_stats:
        return G, stats
    return G


def build_direction_aware_graph(dfn, pbc: bool = False, batch_size: int = 5000,
                                domain=None, *, tau_sin: float = TAU_SIN,
                                tau_len_rel: float = TAU_LEN_REL, want_stats: bool = False):
    """与 ``fractureflow.dfn.build_connectivity_graph`` 同签名的方向感知版。

    dfn 需提供 .centers / .normals / .radii（即 DFNRealization）。
    ``batch_size`` 仅为签名兼容，本实现按 PAIR_CHUNK 分块。
    pbc=True 明确拒绝（不静默按非周期处理 —— 那是会产生错误边的降级）。
    """
    if pbc:
        raise W15InputError(
            "W15 仪器仅实现非周期判交（与 W13 口径 pbc=false 一致）。"
            "PBC 下圆盘跨域壁需镜像处理，属未授权范围，拒绝静默降级。")
    return build_disk_graph(dfn.centers, dfn.normals, dfn.radii,
                            tau_sin=tau_sin, tau_len_rel=tau_len_rel,
                            want_stats=want_stats)


def sphere_approx_graph(centers, radii, *, search_radius_mult: float = SEARCH_RADIUS_MULT):
    """复刻旧仪器判交（d < r_i + r_j）用于 A2 边集差与 J1 对照 —— 不 import 冻结件、不改它。

    与 build_connectivity_graph 的快速路径同原语（同 cKDTree + 同 query_pairs 半径 +
    同严格不等式），故两图之差只可能来自取向，不来自候选集或边界政策。
    """
    c = np.asarray(centers, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    M = c.shape[0]
    pairs = candidate_pairs(c, r, search_radius_mult=search_radius_mult)
    if pairs.shape[0] == 0:
        return sparse.csr_matrix((M, M), dtype=np.float32)
    d = np.linalg.norm(c[pairs[:, 1]] - c[pairs[:, 0]], axis=1)
    mask = d < r[pairs[:, 0]] + r[pairs[:, 1]]
    hit = pairs[mask]
    rows = np.concatenate([hit[:, 0], hit[:, 1]])
    cols = np.concatenate([hit[:, 1], hit[:, 0]])
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, M))


def edge_set(G):
    """上三角边集（frozenset of (i,j)）。"""
    coo = G.tocoo()
    i = coo.row
    j = coo.col
    keep = i < j
    return frozenset(zip(i[keep].tolist(), j[keep].tolist()))
