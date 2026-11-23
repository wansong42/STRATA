# -*- coding: utf-8 -*-
"""R110 · 重构评分卡 v1 纯函数模块 (冻结判据实现).

协议权威: results/v_r110_recon_protocol.json (预注册, mtime 早于一切结果).
本模块只实现协议 §scorecard_v1 的公式, 不含任何拟合/生成逻辑 (那是 B1 的职责),
也不读任何数据文件——全部函数接受显式参数, 可复现、可单测 (评分卡纯函数守卫).

数值卫生: 全程 float64 + 域盒中心化 (继承 segmentation.py 红线; HRF 坐标 ~4e6 量级).

面 (face) 表示: dict(c=[3] 中心, n=[3] 单位法向, R=等效半径 m)。真值网络与
重构网络一律先渲染为点云再走同一条 extract_faces 参数提取路径, 两侧口径对称。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

# --------------------------------------------------------------------------- #
# 冻结常数 (协议 §masking_platform / §scorecard_v1)
# --------------------------------------------------------------------------- #
SCANLINE_SEED = 20260828     # held-out 扫描线布点 seed (协议冻结)
SCANLINE_GRID = (5, 4)       # 5x4 = 20 条垂直线
SCANLINE_JITTER = 8.0        # 网格抖动幅度 (m)
SCANLINE_MIN_DIST_TO_BOREHOLE = 10.0  # 与任何钻孔位置的最小距离 (m)
THETA_MERGE_DEG = 20.0       # leader 聚类合并阈值
RENDER_DENSITY = 0.212       # pts/m^2 (= HRF 实测)
RENDER_CAP = 8000            # 单面点数上限
RENDER_MIN = 50              # 单面点数下限
RIPLEY_R_GRID = np.arange(10.0, 81.0, 10.0)   # 10..80 m
DENSITY_GRID = 8             # xy 网格 8x8
DENSITY_SIGMA = 80.0         # 高斯平滑 (m)
ANGLE_MATCH_DEG = 90.0       # 组系匹配罚项 (未匹配)
CONSISTENCY_Z_TOL_NOT_USED = None  # 一致性层无逐裂隙容差 (窗口级统计, 见协议)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


# --------------------------------------------------------------------------- #
# 冻结 held-out 扫描线 (唯一正演一致性窗口; 遮蔽器布孔时必须避开)
# --------------------------------------------------------------------------- #
def frozen_scanlines(box) -> np.ndarray:
    """20 条垂直扫描线 (x,y) 位置。box=[[x0,y0,z0],[x1,y1,z1]]。

    5x4 网格 + seed 20260828 抖动, 再整体推离域边界 (线必须在域内)。
    确定性: 同 box 双跑逐位一致。
    """
    box = np.asarray(box, dtype=np.float64)
    lo, hi = box[0], box[1]
    rng = np.random.default_rng(SCANLINE_SEED)
    nx, ny = SCANLINE_GRID
    xs = np.linspace(lo[0], hi[0], nx + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], ny + 2)[1:-1]
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    pts += rng.uniform(-SCANLINE_JITTER, SCANLINE_JITTER, pts.shape)
    # 推回域内 (防止抖动越界)
    margin = 5.0
    pts[:, 0] = np.clip(pts[:, 0], lo[0] + margin, hi[0] - margin)
    pts[:, 1] = np.clip(pts[:, 1], lo[1] + margin, hi[1] - margin)
    return pts


def draw_borehole_positions(box, k, seed=SCANLINE_SEED,
                            min_dist=SCANLINE_MIN_DIST_TO_BOREHOLE,
                            max_tries=10000):
    """钻孔 (x,y) 布点: seed 拒绝采样, 与冻结扫描线距离 > min_dist。确定性。"""
    box = np.asarray(box, dtype=np.float64)
    sl = frozen_scanlines(box)
    rng = np.random.default_rng(seed + 1)
    margin = 15.0
    out = []
    tries = 0
    while len(out) < k and tries < max_tries:
        tries += 1
        p = np.array([rng.uniform(box[0][0] + margin, box[1][0] - margin),
                      rng.uniform(box[0][1] + margin, box[1][1] - margin)])
        if np.min(np.hypot(sl[:, 0] - p[0], sl[:, 1] - p[1])) > min_dist:
            out.append(p)
    if len(out) < k:
        raise RuntimeError("borehole placement failed (max_tries)")
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# 网络提取与渲染 (双侧对称口径)
# --------------------------------------------------------------------------- #
def extract_faces(pos, fid):
    """点云 -> 面参数列表 [{c,n,R,n_pts}]。与真值提取同一路径 (协议口径)。"""
    pos = np.asarray(pos, dtype=np.float64)
    fid = np.asarray(fid)
    faces = []
    for f in np.unique(fid[fid >= 0]):
        m = fid == f
        P = pos[m]
        c = P.mean(0)
        X = P - c
        w, v = np.linalg.eigh(X.T @ X)
        n = _unit(v[:, 0])
        r_in = X - np.outer(X @ n, n)
        R = float(np.sqrt(2.0 * (r_in ** 2).sum(1).mean()))
        faces.append(dict(c=c, n=n, R=R, n_pts=int(m.sum())))
    return faces


def render_faces(faces, seed=0, density=RENDER_DENSITY,
                 cap=RENDER_CAP, min_pts=RENDER_MIN):
    """面列表 -> 点云 (pos, fid), 均匀圆盘采样, 确定性。"""
    rng = np.random.default_rng(seed)
    pos_list, fid_list = [], []
    for i, f in enumerate(faces):
        area = np.pi * f["R"] ** 2
        n = int(np.clip(area * density, min_pts, cap))
        n = max(n, 3)
        r = f["R"] * np.sqrt(rng.random(n))
        t = 2 * np.pi * rng.random(n)
        nvec = f["n"]
        a = np.array([1.0, 0.0, 0.0]) if abs(nvec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(nvec, a))
        e2 = np.cross(nvec, e1)
        off = (r * np.cos(t))[:, None] * e1 + (r * np.sin(t))[:, None] * e2
        pos_list.append(f["c"][None, :] + off)
        fid_list.append(np.full(n, i, dtype=np.int64))
    if not pos_list:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(pos_list), np.concatenate(fid_list)


# --------------------------------------------------------------------------- #
# 组系层
# --------------------------------------------------------------------------- #
def _sign_align_z(n):
    n = np.asarray(n, dtype=np.float64)
    s = np.where((n[:, 2:3] < 0) | ((n[:, 2:3] == 0) & (n[:, 0:1] < 0)), -1.0, 1.0)
    return n * s


def leader_set_clusters(faces, theta_deg=THETA_MERGE_DEG):
    """冻结组系估计器: leader 聚类 (按面积权重降序取领袖, θ=20°)。

    返回 (labels[K_face], means[K,3], props[K], order)。
    """
    if len(faces) == 0:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0), []
    N = _sign_align_z(np.array([f["n"] for f in faces]))
    wts = np.array([np.pi * f["R"] ** 2 for f in faces])
    order = np.argsort(-wts)
    theta = np.cos(np.radians(theta_deg))
    labels = np.full(len(faces), -1, dtype=int)
    means = []
    for i in order:
        if labels[i] >= 0:
            continue
        members = [i]
        lab = len(means)
        labels[i] = lab
        mu = N[i]
        # 迭代吸收: 以当前领袖为心, 一遍贪婪
        for j in order:
            if labels[j] >= 0:
                continue
            if abs(float(N[j] @ mu)) >= theta:
                labels[j] = lab
                members.append(j)
        mu = _unit(_sign_align_z(np.array([N[m] for m in members])).mean(0))
        means.append(mu)
    means = np.asarray(means)
    props = np.array([wts[labels == k].sum() for k in range(len(means))])
    props = props / max(props.sum(), 1e-12)
    return labels, means, props, order


def set_layer_metrics(faces_rec, faces_true):
    _, m_r, p_r, _ = leader_set_clusters(faces_rec)
    _, m_t, p_t, _ = leader_set_clusters(faces_true)
    k_r, k_t = len(m_r), len(m_t)
    set_count_abs_diff = abs(k_r - k_t)
    if k_r == 0 or k_t == 0:
        return dict(set_count_abs_diff=set_count_abs_diff,
                    fisher_mean_direction_distance_deg=90.0,
                    set_proportion_L1=2.0)
    # 匹配 (角距代价, 方形填充 90°)
    cosang = np.abs(m_r @ m_t.T)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    n = max(k_r, k_t)
    C = np.full((n, n), ANGLE_MATCH_DEG)
    C[:k_r, :k_t] = ang
    ri, ci = linear_sum_assignment(C)
    matched = [(int(a), int(b)) for a, b in zip(ri, ci)
               if a < k_r and b < k_t and ang[a, b] < ANGLE_MATCH_DEG]
    if matched:
        dists = [ang[a, b] for a, b in matched]
        # 未匹配组按 90° 罚项计入均值 (协议)
        n_un = (k_r - len(matched)) + (k_t - len(matched))
        fisher = float(np.mean(dists + [ANGLE_MATCH_DEG] * n_un)) if (len(matched) + n_un) else 0.0
    else:
        fisher = ANGLE_MATCH_DEG
    # 比例 L1
    l1 = 0.0
    m_r_set = {a for a, _ in matched}
    m_t_set = {b for _, b in matched}
    for a, b in matched:
        l1 += abs(float(p_r[a]) - float(p_t[b]))
    l1 += float(p_r[[i for i in range(k_r) if i not in m_r_set]].sum())
    l1 += float(p_t[[i for i in range(k_t) if i not in m_t_set]].sum())
    return dict(set_count_abs_diff=int(set_count_abs_diff),
                fisher_mean_direction_distance_deg=fisher,
                set_proportion_L1=float(l1))


# --------------------------------------------------------------------------- #
# 尺度层
# --------------------------------------------------------------------------- #
def _powerlaw_exponent(vals):
    """冻结粗口径: 互补 CDF 在 pooled [q10,q90] 内 log-log OLS。"""
    v = np.sort(np.asarray(vals, dtype=np.float64))
    if len(v) < 8:
        return float("nan")
    q10, q90 = np.quantile(v, [0.1, 0.9])
    if not (q90 > q10 > 0):
        return float("nan")
    grid = np.linspace(q10, q90, 10)
    S = np.array([max(1.0 / len(v), (v > g).mean()) for g in grid])
    b = np.polyfit(np.log(grid), np.log(S), 1)[0]
    return float(-b)


def scale_layer_metrics(faces_rec, faces_true):
    L_r = np.array([2 * f["R"] for f in faces_rec])
    L_t = np.array([2 * f["R"] for f in faces_true])
    if len(L_r) == 0 or len(L_t) == 0:
        return dict(trace_length_w1_rel=1.0, powerlaw_exponent_diff=float("nan"))
    w1 = wasserstein_distance(L_r, L_t) / max(float(L_t.mean()), 1e-9)
    a_r, a_t = _powerlaw_exponent(L_r), _powerlaw_exponent(L_t)
    d = float(a_r - a_t) if (np.isfinite(a_r) and np.isfinite(a_t)) else float("nan")
    return dict(trace_length_w1_rel=float(w1), powerlaw_exponent_diff=d)


# --------------------------------------------------------------------------- #
# 空间层
# --------------------------------------------------------------------------- #
def spatial_layer_metrics(faces_rec, faces_true, box):
    box = np.asarray(box, dtype=np.float64)
    c_r = np.array([f["c"] for f in faces_rec])
    c_t = np.array([f["c"] for f in faces_true])

    # 密度场: 8x8 xy + σ=80m 高斯平滑, TV
    def _density(c):
        gx = np.linspace(box[0][0], box[1][0], DENSITY_GRID + 2)[1:-1]
        gy = np.linspace(box[0][1], box[1][1], DENSITY_GRID + 2)[1:-1]
        G = np.zeros((DENSITY_GRID, DENSITY_GRID))
        if len(c) == 0:
            return G
        for i, x in enumerate(gx):
            for j, y in enumerate(gy):
                d2 = ((c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2)
                G[i, j] = np.exp(-d2 / (2 * DENSITY_SIGMA ** 2)).sum()
        s = G.sum()
        return G / s if s > 0 else G
    tv = 0.5 * float(np.abs(_density(c_r) - _density(c_t)).sum())

    # Ripley K (无边界校正, 双侧同简化)
    V = float(np.prod(box[1] - box[0]))

    def _k(c):
        if len(c) < 2:
            return np.zeros(len(RIPLEY_R_GRID))
        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        D = D[np.triu_indices(len(c), 1)]
        return np.array([float((D < r).sum()) * V / (len(c) * (len(c) - 1))
                         for r in RIPLEY_R_GRID])
    k_r, k_t = _k(c_r), _k(c_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(k_r - k_t) / np.where(k_t > 0, k_t, np.inf)
    ripley = float(np.nanmax(dev)) if np.isfinite(dev).any() else float("inf")
    return dict(density_field_l1_rel=float(tv),
                ripley_k_max_rel_dev=ripley)


# --------------------------------------------------------------------------- #
# 拓扑层
# --------------------------------------------------------------------------- #
def disc_line_intersects(face, x0, y0):
    """垂直线 (x0,y0) 与盘 (c,n,R) 是否相交 (协议冻结判据)。"""
    n = face["n"]
    if abs(n[2]) < 1e-9:
        return False
    dx, dy = x0 - face["c"][0], y0 - face["c"][1]
    s = -(n[0] * dx + n[1] * dy) / n[2]
    z = face["c"][2] + s
    d_in_plane = np.sqrt(dx ** 2 + dy ** 2 + (z - face["c"][2]) ** 2)
    return bool(d_in_plane <= face["R"])


def disc_disc_intersects(f1, f2):
    """盘-盘精确相交判据 (协议冻结)。

    两平面交线 x = p0 + t·u (u=n1×n2 归一化); disc i 内的线段区间满足
    |t·u + (p0−c_i)|² ≤ R_i², 即 t² + 2t(u·w_i) + |w_i|² − R_i² ≤ 0,
    区间端点 t = −(u·w_i) ± sqrt((u·w_i)² + R_i² − |w_i|²); 两区间有交集即相交。
    """
    n1, n2 = f1["n"], f2["n"]
    u = np.cross(n1, n2)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        # 平行面: 相交 iff 共面且盘重叠
        d = f2["c"] - f1["c"]
        if abs(float(d @ n1)) > 1e-6:
            return False
        return bool(np.linalg.norm(d) <= f1["R"] + f2["R"])
    u = u / nu
    d = f2["c"] - f1["c"]
    # p0: 两平面交线上距 (c1+c2)/2 最近的点: x = m + a*n1 + b*n2
    m = 0.5 * (f1["c"] + f2["c"])
    A = np.array([[float(n1 @ n1), float(n1 @ n2)],
                  [float(n2 @ n1), float(n2 @ n2)]])
    b = np.array([float(n1 @ (f1["c"] - m)), float(n2 @ (f2["c"] - m))])
    try:
        ab = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False
    p0 = m + ab[0] * n1 + ab[1] * n2

    def _seg(w, R):
        bb = float(u @ w)
        rad = bb * bb + R * R - float(w @ w)
        if rad <= 0:
            return None
        s = np.sqrt(rad)
        return (-bb - s, -bb + s)

    s1 = _seg(p0 - f1["c"], f1["R"])
    s2 = _seg(p0 - f2["c"], f2["R"])
    if s1 is None or s2 is None:
        return False
    return max(s1[0], s2[0]) <= min(s1[1], s2[1])


def _largest_cluster_fraction(faces):
    n = len(faces)
    if n == 0:
        return 0.0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if disc_disc_intersects(faces[i], faces[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    from collections import Counter
    cnt = Counter(find(i) for i in range(n))
    return max(cnt.values()) / n


def topology_layer_metrics(faces_rec, faces_true, box, scanlines):
    box = np.asarray(box, dtype=np.float64)
    H = float(box[1][2] - box[0][2])
    V = float(np.prod(box[1] - box[0]))

    def _p10(faces):
        if not faces:
            return 0.0
        cnt = sum(1 for f in faces for (x0, y0) in scanlines
                  if disc_line_intersects(f, x0, y0))
        return cnt / (len(scanlines) * H)

    def _p32(faces):
        return sum(np.pi * f["R"] ** 2 for f in faces) / V

    def _inter_density(faces):
        n = len(faces)
        cnt = sum(1 for i in range(n) for j in range(i + 1, n)
                  if disc_disc_intersects(faces[i], faces[j]))
        return cnt / V

    p10r, p10t = _p10(faces_rec), _p10(faces_true)
    p32r, p32t = _p32(faces_rec), _p32(faces_true)
    idr, idt = _inter_density(faces_rec), _inter_density(faces_true)

    def _rel(a, b):
        return abs(a - b) / b if b > 0 else (0.0 if a == 0 else float("inf"))

    return dict(
        p10_rel_err=float(_rel(p10r, p10t)),
        p32_rel_err=float(_rel(p32r, p32t)),
        intersection_density_rel_err=float(_rel(idr, idt)),
        largest_cluster_fraction_diff=abs(
            _largest_cluster_fraction(faces_rec) - _largest_cluster_fraction(faces_true)),
        _diagnostics=dict(p10_rec=p10r, p10_true=p10t, p32_rec=p32r, p32_true=p32t,
                          inter_rec=idr, inter_true=idt),
    )


# --------------------------------------------------------------------------- #
# 正演一致性层 (held-out 扫描线窗口级统计)
# --------------------------------------------------------------------------- #
def _line_pick(faces, scanlines):
    """返回 (z 坐标数组, 法向数组)——held-out 线组上的交切观测。

    判据与 disc_line_intersects 完全一致 (精确盘测试)。
    """
    zs, ns = [], []
    for f in faces:
        n = f["n"]
        if abs(n[2]) < 1e-9:
            continue
        for (x0, y0) in scanlines:
            dx, dy = x0 - f["c"][0], y0 - f["c"][1]
            s = -(n[0] * dx + n[1] * dy) / n[2]
            if dx * dx + dy * dy + s * s <= f["R"] ** 2:
                zs.append(f["c"][2] + s)
                ns.append(n)
    return np.array(zs), np.array(ns).reshape(-1, 3)


def consistency_layer_metrics(faces_rec, faces_true, box, scanlines):
    box = np.asarray(box, dtype=np.float64)
    H = float(box[1][2] - box[0][2])
    z_t, n_t = _line_pick(faces_true, scanlines)
    z_r, n_r = _line_pick(faces_rec, scanlines)
    p10_t = len(z_t) / (len(scanlines) * H)
    p10_r = len(z_r) / (len(scanlines) * H)
    p10_rel = abs(p10_r - p10_t) / p10_t if p10_t > 0 else (0.0 if p10_r == 0 else float("inf"))

    def _zw1(z):
        if len(z) == 0 or len(z_t) == 0:
            return 1.0
        return wasserstein_distance(z, z_t) / H

    def _circ_std(n):
        if len(n) < 2:
            return 0.0
        Nv = _sign_align_z(n)
        R = float(np.linalg.norm(Nv.mean(0)))
        R = min(max(R, 1e-9), 1.0 - 1e-12)
        return float(np.degrees(np.sqrt(-2 * np.log(R))))

    return dict(
        heldout_p10_rel_err=float(p10_rel),
        heldout_z_w1_rel=float(_zw1(z_r)),
        heldout_normal_circ_std_diff_deg=abs(_circ_std(n_r) - _circ_std(n_t)),
        _diagnostics=dict(picks_true=int(len(z_t)), picks_rec=int(len(z_r)),
                          p10_true=p10_t, p10_rec=p10_r,
                          circ_std_true=_circ_std(n_t), circ_std_rec=_circ_std(n_r)),
    )


# --------------------------------------------------------------------------- #
# 总评分卡
# --------------------------------------------------------------------------- #
def scorecard(faces_rec, faces_true, box, scanlines):
    """五层评分卡 (协议 §scorecard_v1 全量度量)。纯函数、确定性。"""
    out = dict(
        set=set_layer_metrics(faces_rec, faces_true),
        scale=scale_layer_metrics(faces_rec, faces_true),
        spatial=spatial_layer_metrics(faces_rec, faces_true, box),
        topology=topology_layer_metrics(faces_rec, faces_true, box, scanlines),
        consistency=consistency_layer_metrics(faces_rec, faces_true, box, scanlines),
    )
    return out


def scorecard_from_points(pos_rec, fid_rec, pos_true, fid_true, box, scanlines):
    """点云入口: 两侧同走 extract_faces (协议对称口径)。"""
    return scorecard(extract_faces(pos_rec, fid_rec),
                     extract_faces(pos_true, fid_true), box, scanlines)


# --------------------------------------------------------------------------- #
# vMF 切向高斯采样 (B1 用; 近似精度对 κ≥5 足够, 协议冻结)
# --------------------------------------------------------------------------- #
def vmf_tangent_gaussian(rng, mu, kappa):
    mu = _unit(np.asarray(mu, dtype=np.float64))
    kappa = float(np.clip(kappa, 5.0, 2000.0))
    sigma = 1.0 / np.sqrt(kappa)
    a = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(mu, a))
    e2 = np.cross(mu, e1)
    t = rng.normal(0.0, sigma, 2)
    v = mu + t[0] * e1 + t[1] * e2
    return _unit(v)
