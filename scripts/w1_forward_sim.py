# -*- coding: utf-8 -*-
"""W1 · 岩心展开图正向数字孪生 (Phase 0)。

几何权威 = results/pitfalls_and_evidence_20260925/技术方案_..._v1.0.md §1.1
         + results/w1_core_unfold_twin/prereg.json
         + results/w1_core_unfold_twin/prereg_amendments_1.json

闭式几何 (岩心系: 轴 = +z, 柱面点 P(t,z) = (R cos t, R sin t, z)):
    n = (sin d sin a, sin d cos a, cos d)            # 冻结件 fractureflow.geometry
    n . (P - p0) = 0  =>  z(t) = z0 - [nx(R cos t - x0) + ny(R sin t - y0)] / nz
    即 z(t) = c + u cos t + v sin t, u = -R tan d sin a, v = -R tan d cos a
    => A = hypot(u,v) = R tan d  =>  d = atan(A/R)  (d in [0,90) 单调无歧义)
       a = atan2(-u, -v) mod 360

法向一律委托冻结件 (T58/十五期 _nd 修复后口径, nz = cos dip); 本文件**不**自行
组装 sin/cos, 由结构上排除镜像 bug; 另自带 ST-1 三层往返自测 + 90-dip 镜像负例。

用法:
  python scripts/w1_forward_sim.py --selftest
  python scripts/w1_forward_sim.py --smoke [--dpi 150] [--seed 1] [--save-png]
水印: 研究线 · 禁入对外承诺文本
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # prereg determinism_env (须在 numpy 之前)
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 本机 libiomp5md 重复链接(AGENTS.md 门禁注记); 该开关是"unsafe workaround", 故其风险由 J5 跨进程逐位一致兜底

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from fractureflow.geometry import dip_dipdir_to_normal as _gd2n  # noqa: E402
from fractureflow.geometry import normal_to_dip_dipdir as _gn2d  # noqa: E402
from fractureflow.dfn import SetTable, generate_dfn  # noqa: E402

WATERMARK = "研究线数字 · 禁入对外承诺文本"
OUT = ROOT / "results" / "w1_core_unfold_twin"
CELLS = OUT / "cells"

MM_PER_M = 1000.0
TWO_PI = 2.0 * math.pi
MAX_PX = 130_000_000


# ---------------------------------------------------------------------------
# 产状 <-> 法向 (冻结约定的薄封装)
# ---------------------------------------------------------------------------

def normal_from_dip(dip_deg, dipdir_deg):
    return np.asarray(_gd2n(float(dip_deg), float(dipdir_deg)),
                      dtype=np.float64).reshape(3)


def dip_from_normal(n):
    d, a = _gn2d(np.asarray(n, dtype=np.float64).reshape(1, 3))
    return float(np.atleast_1d(d)[0]), float(np.atleast_1d(a)[0])


def circ_diff_deg(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.where(d > 180.0, d - 360.0, d)


def circ_abs_diff_deg(a, b):
    return np.abs(circ_diff_deg(a, b))


def sine_to_attitude(u, v, R_m):
    """(c,u,v) 线性参数 -> (dip, dipdir)。逆向提取器与 ST-1 共用同一函数。"""
    amp = math.hypot(float(u), float(v))
    dip = math.degrees(math.atan2(amp, float(R_m)))
    az = math.degrees(math.atan2(-float(u), -float(v))) % 360.0
    return dip, az


def attitude_to_sine(dip_deg, az_deg, R_m):
    """(dip, az) -> (A, u, v)。ST-1 双向锁。"""
    t = math.tan(math.radians(float(dip_deg)))
    a = math.radians(float(az_deg))
    u = -float(R_m) * t * math.sin(a)
    v = -float(R_m) * t * math.cos(a)
    return float(math.hypot(u, v)), u, v


# ---------------------------------------------------------------------------
# ST-1 · 三层往返自测 + 90-dip 镜像负例 (新代码自写, 不改 selfcheck S32)
# ---------------------------------------------------------------------------

def st1_roundtrip_selftest(k=400, seed=20260926):
    """容差按**解析条件数**给, 不用固定死数:
      dip = arccos|nz|  =>  ddip/dnz = -1/sin(dip), 故 |nz| 的相对误差 er
      在 dip 处放大为 er/sin(dip) (弧度)。冻结件归一化带 +1e-12 => er ~ 1e-12。
    断言 err <= bound (= 只允许条件数解释的误差, 多一个数量级即红)。
    """
    rng = np.random.default_rng(seed)
    dips = rng.uniform(0.5, 89.0, k)
    azs = rng.uniform(0.0, 360.0, k)
    R = 0.05
    ER = 2e-12          # 1e-12 归一化项 + 三角/舍入留量 (1 倍留量曾差 0.002%, 见执行自曝)
    ratio_n = ratio_1a = ratio_2d = 0.0
    e2a = e3 = 0.0
    for d, a in zip(dips, azs):
        sin_d = max(math.sin(math.radians(d)), 1e-9)
        bound_d_deg = ER / sin_d * 180.0 / math.pi
        n = normal_from_dip(d, a)
        d2, a2 = dip_from_normal(n)                       # 层 1: n -> 产状
        ratio_n = max(ratio_n, abs(d - d2) / bound_d_deg)
        ratio_1a = max(ratio_1a, float(circ_abs_diff_deg(a, a2)) / 1e-9)
        A, u, v = attitude_to_sine(d, a, R)               # 层 2: 正弦参数 -> 产状
        d3, a3 = sine_to_attitude(u, v, R)
        ratio_2d = max(ratio_2d, abs(d - d3) / 1e-12)
        e2a = max(e2a, float(circ_abs_diff_deg(a, a3)))
    # 层 3: 闭式曲线本身 —— 取 t 上若干点, 断言 n.(P-p0)==0
    for d, a in list(zip(dips, azs))[:60]:
        n = normal_from_dip(d, a)
        p0 = rng.normal(0.0, 0.01, 3)
        if abs(n[2]) < 1e-6:
            continue
        for t in rng.uniform(0.0, TWO_PI, 20):
            P = np.array([R * math.cos(t), R * math.sin(t),
                          p0[2] - (n[0] * (R * math.cos(t) - p0[0])
                                   + n[1] * (R * math.sin(t) - p0[1])) / n[2]])
            e3 = max(e3, abs(float(np.dot(n, P - p0))))
    # 层 4: 90-dip 镜像**负例** —— 证明层 3 的统计量对镜像**敏感**
    # (即: 若本文件某处把 sin/cos 互换, 层 3 必红)。不是证明 arccos 能分辨两者。
    # 两处口径纠正 (执行自曝): (a) 聚合用 max 不用 min —— min 会被"迹线恰好擦过镜像面"
    #     的单点翻转 (坑 P8 同族, 首版因此 6 例假失手);
    #  (b) 排除 |dip-45| < 5 的病态区 —— 那里镜像面与原面**夹角趋零**, 比较本身无意义。
    mirror_caught = 0
    mirror_tested = 0
    mirror_skipped_near_45 = 0
    mirror_min_resid_m = 1e9
    true_max_resid_m = 0.0
    t_probe = (0.3, 1.7, 2.9, 4.2, 5.1, 5.9)
    for d, a in list(zip(dips, azs)):
        if not (20.0 <= d <= 70.0):
            continue
        if abs(d - 45.0) < 5.0:
            mirror_skipped_near_45 += 1
            continue
        mirror_tested += 1
        n_ok = normal_from_dip(d, a)
        n_mir = normal_from_dip(90.0 - d, a)       # T58 前的犯案法向
        p0 = np.array([0.01, -0.007, 0.02])
        pts = []
        for t in t_probe:
            P = np.array([R * math.cos(t), R * math.sin(t),
                          p0[2] - (n_ok[0] * (R * math.cos(t) - p0[0])
                                   + n_ok[1] * (R * math.sin(t) - p0[1])) / n_ok[2]])
            pts.append(P - p0)
        pts = np.array(pts)
        res_mir = float(np.abs(pts @ n_mir).max())
        res_true = float(np.abs(pts @ n_ok).max())
        true_max_resid_m = max(true_max_resid_m, res_true)
        mirror_min_resid_m = min(mirror_min_resid_m, res_mir)
        if res_mir > 1e-4:
            mirror_caught += 1
    ok = (ratio_n <= 1.0 and ratio_1a <= 1.0 and ratio_2d <= 1.0
          and e2a < 1e-9 and e3 < 1e-12
          and true_max_resid_m < 1e-15
          and mirror_tested >= 50 and mirror_caught == mirror_tested)
    return {
        "ok": bool(ok), "k": k,
        "L1_dip_err_over_conditioning_bound_max": ratio_n,
        "L1_azim_err_over_1e-9_max": ratio_1a,
        "L2_dip_err_over_1e-12_max": ratio_2d,
        "L2_max_azim_err_deg": e2a,
        "L3_max_plane_residual_m": e3,
        "L4_mirror_tested": mirror_tested, "L4_mirror_caught": mirror_caught,
        "L4_mirror_skipped_near_45deg": mirror_skipped_near_45,
        "L4_mirror_min_residual_on_true_trace_m": mirror_min_resid_m,
        "L4_true_normal_max_residual_m": true_max_resid_m,
        "L4_requires": "max|n_mir.(P-p0)| > 1e-4 m 且 max|n_true.(P-p0)| < 1e-15 m => L3 对 sin/cos 互换有牙",
        "tolerance_law": "err_dip <= (1e-12/sin(dip)) rad; 冻结件 normal_to_dip_dipdir 用 arccos|nz|, 小倾角条件数 1/sin(dip)",
        "convention": "n=(sin d sin a, sin d cos a, cos d) via fractureflow.geometry (frozen)",
    }


# ---------------------------------------------------------------------------
# 裂隙总体 (大域 DFN -> 取其中一孔岩心; 不是在岩心内撒点)
# ---------------------------------------------------------------------------

REALISTIC_SETS = [   # (dip, dipdir, weight)
    (62.0, 25.0, 0.34), (74.0, 118.0, 0.22), (28.0, 300.0, 0.22),
    (81.0, 205.0, 0.12), (12.0, 145.0, 0.10),
]


def make_population(seed, domain_m=(4.0, 4.0, 2.5), p32=12.0, beta=3.0,
                    kappa=20.0, r_min=0.30, r_max=2.0):
    """域内裂隙总体 (generate_dfn 冻结生成器)。

    口径声明 (prereg amendment): generate_dfn 的 P32 按**域内中心**计数, 故跨域边界的
    大圆盘会低估 realized P32; 因此本线的密度轴以**实测每米迹线数**为准随表同行,
    名义 p32 只作单调旋钮。
    """
    ctr = np.array([normal_from_dip(d, a) for d, a, _ in REALISTIC_SETS])
    prop = np.array([w for _, _, w in REALISTIC_SETS])
    st = SetTable(centers=ctr, concentrations=np.full(len(ctr), float(kappa)),
                  proportions=prop)
    dfn = generate_dfn(st, p32=float(p32), beta=float(beta), domain=domain_m,
                       seed=int(seed), n_min=50,
                       r_min=float(r_min), r_max=float(r_max))
    return dfn, st


def rot_matrix(tilt_deg=0.0, tilt_azim_deg=0.0):
    ca, sa = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    cb, sb = math.cos(math.radians(tilt_azim_deg)), math.sin(math.radians(tilt_azim_deg))
    Ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    Rz = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry


def fracture_arcs_on_core(nrm, center, radius, R_core, z_lo, z_hi, n_theta,
                          min_arc_theta_rad=0.03):
    """圆盘 ∩ 圆柱面 -> [(theta[], z[], wrap_flag)]。解析, 单点判据精确。"""
    nx, ny, nz = (float(nrm[0]), float(nrm[1]), float(nrm[2]))
    x0, y0, z0 = (float(center[0]), float(center[1]), float(center[2]))
    horiz = math.hypot(nx, ny)

    if abs(nz) > 1e-6:
        th = np.linspace(0.0, TWO_PI, int(n_theta), endpoint=False)
        ct, st_ = np.cos(th), np.sin(th)
        z = z0 - (nx * (R_core * ct - x0) + ny * (R_core * st_ - y0)) / nz
        dxy2 = (R_core * ct - x0) ** 2 + (R_core * st_ - y0) ** 2
        keep = (dxy2 <= radius * radius) & (z >= z_lo) & (z <= z_hi)
        return _split_runs(th, z, keep, min_arc_theta_rad)

    # 平面含轴向: 交线 = 两条竖线
    if horiz < 1e-12:
        return []
    phi = math.atan2(ny, nx)
    d = (nx * x0 + ny * y0) / horiz
    if abs(d) >= R_core:
        return []
    dt = math.acos(d / R_core)
    out = []
    for t in (phi + dt, phi - dt):
        px, py = R_core * math.cos(t), R_core * math.sin(t)
        perp2 = (px - x0) ** 2 + (py - y0) ** 2
        half2 = radius * radius - perp2
        if half2 <= 0:
            continue
        half = math.sqrt(half2)
        lo, hi = max(z0 - half, z_lo), min(z0 + half, z_hi)
        if hi - lo <= (z_hi - z_lo) * 0.02:
            continue
        m = max(16, int(round((hi - lo) / ((z_hi - z_lo) / 600.0))))
        zz = np.linspace(lo, hi, m)
        tt = np.full(zz.shape, t % TWO_PI)
        out.append((tt, zz, False))
    return out


def _split_runs(th, z, keep, min_arc_theta_rad):
    idx = np.where(keep)[0]
    if idx.size < 2:
        return []
    n = th.size
    brk = np.where(np.diff(idx) > 1)[0]
    groups = []
    s = 0
    for b in brk:
        groups.append(idx[s:b + 1])
        s = b + 1
    groups.append(idx[s:])
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
        groups[0] = np.concatenate([groups[-1], groups[0]])
        groups.pop()
    out = []
    for g in groups:
        if g.size < 2:
            continue
        th_g = th[g].copy()
        wrap = bool(g[0] > g[-1])            # 索引非单调 => 跨接缝的环形段
        if wrap:
            # cohort-2 (AMEND-37): 旧实现按 **θ>π 数值阈值** 切 hi/lo 再拼接, 但合并段的正确切分点
            # 是索引断点。低端越过 180° 时 (如覆盖 340-360 + 0-200), 181-200° 也被划进 hi,
            # 于是 θ 数组非单调、与**未重排的 z 错位**, 记录窗口 != 实画覆盖 (构造反例见
            # results/w1_core_unfold_twin/cohort2_wrap_mechanism_probe.json)。
            # 现改为: 保持 (θ, z) 配对不动, 只在**真实下降处** +2π 展开 —— 非绕缝段无下降, 逐位不变。
            dth = np.diff(th_g)
            cum = np.concatenate([[0.0], np.cumsum(np.where(dth < 0.0, TWO_PI, 0.0))])
            th_g = th_g + cum
        if abs(th_g[-1] - th_g[0]) < min_arc_theta_rad:
            continue
        out.append((th_g, z[g].copy(), wrap))
    return out


# ---------------------------------------------------------------------------
# 栅格化: 1px 骨架 -> 圆盘膨胀成缝宽 (cv2, 环形列)
# ---------------------------------------------------------------------------

def _imwrite(path, arr):
    """中文绝对路径安全的 PNG 写 (cv2.imwrite 在含非 ASCII 的 CWD 下**静默返回 False**;
    本仓首次触发即本单 --save-png 零落盘, 沿用 outcrop_trace._imwrite_robust 同法)。"""
    ext = os.path.splitext(str(path))[1] or ".png"
    ok, enc = cv2.imencode(ext, arr)
    if not ok:
        raise SystemExit(f"W1: imencode 失败 {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(enc.tobytes())
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SystemExit(f"W1: 声称写盘但未落盘 {path}")
    return os.path.getsize(path)


def _disk_kernel(r_px):
    r = max(0.0, float(r_px))
    k = int(2 * math.ceil(r) + 1)
    if k <= 1:
        return np.ones((1, 1), dtype=np.uint8)
    o = np.arange(k) - (k - 1) / 2.0
    U, V = np.meshgrid(o, o)
    return (np.hypot(U, V) <= r + 0.25).astype(np.uint8)


def _paint_1px(cols, rows, W, H, row_ok=None):
    """把 (col, row) 点集涂成 1px 宽的整数位置; 列绕接缝, 行出界丢弃。"""
    c = np.rint(cols).astype(np.int64) % W
    r = np.rint(rows).astype(np.int64)
    ok = (r >= 0) & (r < H)
    if row_ok is not None:
        ok &= row_ok[r.clip(0, H - 1)]
    return c[ok], r[ok]


def _set_lab(lab, cs, rs, id_val):
    if cs.size == 0:
        return
    flat = (rs * lab.shape[1] + cs).astype(np.int64)
    flat = np.unique(flat)
    lab.reshape(-1)[flat] = id_val


def _set_mask(mask, cs, rs, val):
    if cs.size == 0:
        return
    flat = (rs * mask.shape[1] + cs).astype(np.int64)
    mask.reshape(-1)[np.unique(flat)] = val


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------

def base_cfg():
    return {
        "R_mm": 50.0, "L_mm": 500.0, "dpi": 600, "snr_db": "inf",
        "rubble_pct": 0.0, "loss_pct": 0.0, "fake_per_m": 0.0,
        "scribe_err": 0.0, "p32": 12.0, "aperture_mm": 0.5, "blur_sigma_px": 0.6,
        "illum_amp": [0.15, 0.05, 0.0], "tilt_deg": 0.0, "tilt_azim_deg": 0.0,
        "pop_seed_offset": 0, "r_min_pop": 0.30, "r_max_pop": 2.0,
    }


def _intervals(frac, span_hi, rng, min_frac=0.02):
    if frac <= 0:
        return []
    total = span_hi * frac
    out = []
    covered = 0.0
    for _ in range(200):
        if covered >= total - 1e-12:
            break
        L = min(span_hi * rng.uniform(min_frac, 4 * min_frac + 0.05), total - covered)
        if L <= 0:
            break
        lo = rng.uniform(0.0, span_hi - L)
        out.append((lo, lo + L))
        covered += L
    return out


def _value_noise(rows, cols, rng, cells_x=8.0, octaves=4):
    """多倍频值噪声, float32 出 (内存敏感: 单格可达 130M px)。

    上采样必须 order>=3: order=1 的双线性插值出**分片线性面**, 其二级导在格点
    处有非物理折痕, 会被脊/Hessian 滤波器当成真边缘 (J0 停扫定位到的实现缺陷,
    不是物理难度)。这里用样条插值 + 按 cell 尺寸低通, 使纹理带限与其声称的
    空间尺度一致。
    """
    acc = np.zeros((rows, cols), np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        cx = max(2, int(cells_x * (2 ** o)))
        cy = max(2, int(cx * rows / max(cols, 1)))
        base = rng.random((cy, cx)).astype(np.float32)
        z = ndimage.zoom(base, (rows / cy, cols / cx), order=3, mode="nearest")
        if z.shape != (rows, cols):
            z = np.resize(z, (rows, cols))
        z = ndimage.gaussian_filter(z, (0.5 * rows / cy, 0.5 * cols / cx),
                                    mode="nearest")
        acc += np.float32(amp) * (z - np.float32(0.5))
        tot += amp
        amp *= 0.55
    return acc / np.float32(max(tot, 1e-12))


def render_core(cfg, seed):
    """一孔岩心展开图 + 逐像素真值掩膜 + 逐迹线真值。"""
    rng = np.random.default_rng(int(seed))
    R = cfg["R_mm"] / MM_PER_M
    L = cfg["L_mm"] / MM_PER_M
    dpi = int(cfg["dpi"])
    px_mm = dpi / 25.4
    W = int(round(TWO_PI * cfg["R_mm"] * px_mm))
    H = int(round(cfg["L_mm"] * px_mm))
    if W * H > MAX_PX:
        raise SystemExit(f"W1 forward: 图像 {W}x{H} > {MAX_PX} px, 拒绝 (防内存爆)")
    z_hi, z_lo = L / 2.0, -L / 2.0

    # ---- 定向线误差: 图像列号 -> 提取器以为的 theta; 真实 theta 多转 eps
    if cfg["scribe_err"] == "NONE":
        eps = float(rng.uniform(0.0, 360.0))
        scribe_branch = False
    else:
        eps = float(rng.normal(0.0, float(cfg["scribe_err"]))) % 360.0
        scribe_branch = True

    # 缓冲全部走 8/16 bit: 单格最大 130M px, float32 版会到 GB 级 (内存实测后定)
    lab_frac = np.zeros((H, W), np.uint16)     # 真裂隙实例号
    lab_bed = np.zeros((H, W), np.uint16)      # 层理假迹线实例号
    lab_vein = np.zeros((H, W), np.uint16)     # 矿脉假迹线实例号
    a_frac = np.zeros((H, W), np.uint8)        # 变暗量 (255)
    a_bed = np.zeros((H, W), np.uint8)         # 层理变暗量 (242)
    a_vein = np.zeros((H, W), np.uint8)        # 矿脉变亮量 (242)
    truth_arcs = []
    fakes = []

    def render_label(cs, rs):
        """A① (架构师 round17): **标签由实际渲染几何导出**, 禁止反过来改像素凑已存标签。

        对**真的被画出来的**像素做普通最小二乘 (闭式线性拟合, 不引逆向提取器的任何实现),
        再按定向线偏移 eps 转回柱面坐标 (与 analytic 标签同一参考系)。None = 像素太少。
        """
        if cs.size < 16:
            return None
        th_i = (cs.astype(np.float64) + 0.5) / W * TWO_PI
        z_i = z_hi - (rs.astype(np.float64) + 0.5) / H * (z_hi - z_lo)
        Md = np.column_stack([np.ones(th_i.size), np.cos(th_i), np.sin(th_i)])
        coef, _res, _rk, _sv = np.linalg.lstsq(Md, z_i, rcond=None)
        c_f, u_i, v_i = (float(x) for x in coef)
        e = math.radians(eps)
        u_f = u_i * math.cos(e) - v_i * math.sin(e)
        v_f = u_i * math.sin(e) + v_i * math.cos(e)
        resid = Md @ coef - z_i
        d_f, a_f = sine_to_attitude(u_f, v_f, R)
        return {"dip": float(d_f), "az": float(a_f), "c": c_f, "u": u_f, "v": v_f,
                "rms_m": float(np.sqrt(np.mean(resid * resid))), "n_px": int(cs.size),
                # median |published curve - own painted pixels| 是最有用的那个数:
                # 若**没有**任何正弦能描述被画出的像素, 它必然变大 (首跑 s1 实测 36.55 mm)
                "median_resid_m": float(np.median(np.abs(resid)))}

    def emit(th_true, z, id_val, is_frac, kind, extra=None):
        cols = (((np.degrees(np.asarray(th_true, float)) - eps) % 360.0) / 360.0) * W
        rows = ((z_hi - np.asarray(z, float)) / (z_hi - z_lo)) * H
        cs, rs = _paint_1px(cols, rows, W, H)
        if kind == "frac":
            _set_lab(lab_frac, cs, rs, id_val)
            _set_mask(a_frac, cs, rs, 255)
        elif kind == "bedding":
            _set_lab(lab_bed, cs, rs, id_val)
            _set_mask(a_bed, cs, rs, 242)
        else:                                    # vein = 亮线
            _set_lab(lab_vein, cs, rs, id_val)
            _set_mask(a_vein, cs, rs, 242)
        # 单位: R 是**米** (cfg["R_mm"]/MM), 旧变量名 step_mm 因此把 arc_px 少算 1000 倍
        # (真值表里每条弧都报 area_px_total=2, 首跑实测到)。改名 step_m 并同时给出
        # 两种口径: 弧长像素 与 实际覆盖的列数。
        step_m = TWO_PI * R / max(1, len(th_true) - 1) if len(th_true) > 1 else 0.0
        n_col = int(np.unique(cs).size)
        rec = {"fracture_id": int(id_val), "is_fracture": bool(is_frac),
               "kind": kind, "arc_theta_deg": float(abs(np.degrees(
                   np.asarray(th_true, float)[-1] - np.asarray(th_true, float)[0]))),
               "th0_rad": float(np.asarray(th_true, float)[0]),
               "th1_rad": float(np.asarray(th_true, float)[-1]),
               "arc_len_px": int(round(len(th_true) * step_m * px_mm)),
               "n_cols_covered": n_col,
               "z_mid_m": float(0.5 * (np.asarray(z).min() + np.asarray(z).max())),
               "z_lo_m": float(np.asarray(z).min()), "z_hi_m": float(np.asarray(z).max()),
               "n_theta_samples": int(len(th_true))}
        if extra:
            rec.update(extra)
        if is_frac:
            rl = render_label(cs, rs)
            if rl is None:
                rec["label_source"] = "analytic (render_label 像素不足)"
            else:
                rec["dip_deg_analytic"] = rec.get("dip_deg")
                rec["dipdir_deg_analytic"] = rec.get("dipdir_deg")
                rec["c_truth_m_analytic"] = rec.get("c_truth_m")
                rec["u_truth_analytic"] = rec.get("u_truth")
                rec["v_truth_analytic"] = rec.get("v_truth")
                # 官方字段 = 渲染几何导出 (A①); analytic 保留作对照与诊断
                rec["dip_deg"] = rl["dip"]
                rec["dipdir_deg"] = rl["az"]
                rec["c_truth_m"] = rl["c"]
                rec["label_source"] = "render_derived"
                rec["render_label_rms_m"] = rl["rms_m"]
                rec["render_label_median_resid_m"] = rl["median_resid_m"]
                rec["render_label_n_px"] = rl["n_px"]
                rec["label_delta_dip_deg"] = abs(float(rec["dip_deg"])
                                                 - float(rec["dip_deg_analytic"]))
                rec["label_delta_az_deg"] = float(circ_abs_diff_deg(
                    rec["dipdir_deg"], rec["dipdir_deg_analytic"]))
                rec["label_delta_c_mm"] = abs(float(rec["c_truth_m"])
                                              - float(rec["c_truth_m_analytic"])) * 1000.0
                rec["label_delta_max_deg"] = max(rec["label_delta_dip_deg"],
                                                 rec["label_delta_az_deg"])
        (truth_arcs if is_frac else fakes).append(rec)

    # ---- 真裂隙 (紧域总体 + 向量化候选筛选)
    dfn, _st = make_population(int(seed) + int(cfg["pop_seed_offset"]),
                              p32=float(cfg["p32"]),
                              domain_m=(4.0, 4.0, cfg["L_mm"] / MM_PER_M + 2.0),
                              r_min=float(cfg.get("r_min_pop", 0.30)),
                              r_max=float(cfg.get("r_max_pop", 2.0)))
    M = rot_matrix(float(cfg["tilt_deg"]), float(cfg["tilt_azim_deg"]))
    off = rng.uniform(-0.3, 0.3, 3) * np.array([1.0, 1.0, 0.6])
    n_theta = max(8192, 4 * W)
    fid = 0
    if dfn.n_fractures:
        C_loc = (dfn.centers - off) @ M              # (N,3) 岩心系中心
        N_loc = dfn.normals @ M                      # (N,3) 岩心系法向
        rr = np.asarray(dfn.radii, float)
        dperp = np.hypot(C_loc[:, 0], C_loc[:, 1])
        cand = np.where((dperp <= rr + R)
                        & (C_loc[:, 2] - rr <= z_hi)
                        & (C_loc[:, 2] + rr >= z_lo))[0]
        for i in cand:
            rad = float(rr[i])
            c_loc = C_loc[i]
            n_loc = N_loc[i]
            near_vert = abs(n_loc[2]) < 1e-3
            d_true, a_true = dip_from_normal(n_loc)
            # 解析闭式常数项 (真值 c,u,v) —— 同时用作**格内一致性断言**:
            # 由 (u,v) 反解的产状必须等于由法向反解的产状, 否则仿真器内部口径分裂。
            u_t = -R * n_loc[0] / n_loc[2]
            v_t = -R * n_loc[1] / n_loc[2]
            c_t = c_loc[2] + (n_loc[0] * c_loc[0] + n_loc[1] * c_loc[1]) / n_loc[2]
            d_uv, a_uv = sine_to_attitude(u_t, v_t, R)
            dd = abs(d_true - d_uv)
            da = float(circ_abs_diff_deg(a_true, a_uv))
            if dd > 1e-6 or da > 1e-4:
                raise SystemExit(
                    f"W1 内部一致性断言失败 frac{i}: dip 法向={d_true:.6f} "
                    f"正弦={d_uv:.6f} (d={dd:.2e}) az d={da:.2e}")
            arcs = fracture_arcs_on_core(n_loc, c_loc, rad, R, z_lo, z_hi, n_theta)
            if not arcs:
                continue
            fid += 1
            for (th, z, wrap) in arcs:
                emit(th, z, fid, True, "frac",
                     extra={"dip_deg": d_true, "dipdir_deg": a_true,
                            "c_truth_m": float(c_t), "u_truth": float(u_t),
                            "v_truth": float(v_t),
                            "set_id": int(dfn.sets[i]), "wrap": bool(wrap),
                            "near_vertical": bool(near_vert),
                            "disc_radius_m": rad})

    # ---- 假迹线: 层理 (近水平暗线) + 矿脉 (亮线)
    if cfg["fake_per_m"] > 0:
        n_fake = max(1, int(round(float(cfg["fake_per_m"]) * L)))
        n_bed = (n_fake + 1) // 2
        th_true_ax = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
        for _ in range(n_bed):
            fid += 1
            zc = float(rng.uniform(z_lo, z_hi))
            zz = zc + 0.004 * np.sin(3.0 * th_true_ax + float(rng.uniform(0, 6)))
            emit(th_true_ax, zz, fid, False, "bedding",
                 extra={"dip_deg": 1.5, "dipdir_deg": 0.0})
        for _ in range(n_fake - n_bed):
            fid += 1
            d = float(rng.uniform(20, 70))
            a = float(rng.uniform(0, 360))
            nl = normal_from_dip(d, a)
            cl = np.array([rng.normal(0, R), rng.normal(0, R),
                           rng.uniform(z_lo, z_hi)])
            for (th, z, wrap) in fracture_arcs_on_core(nl, cl, max(0.6, 3 * R), R,
                                                       z_lo, z_hi, n_theta):
                emit(th, z, fid, False, "vein",
                     extra={"dip_deg": d, "dipdir_deg": a})

    # ---- 缝宽膨胀 (真值掩膜与图像同一形态学操作 => 无"骨架 vs 图"不一致)
    apx = float(cfg["aperture_mm"]) * px_mm
    k_f = _disk_kernel(apx / 2.0)
    k_b = _disk_kernel(apx * 1.3 / 2.0)
    k_v = _disk_kernel(apx * 1.8 / 2.0)
    lab_frac = cv2.dilate(lab_frac, k_f, iterations=1)
    a_frac = cv2.dilate(a_frac, k_f, iterations=1)
    if a_bed.any():
        lab_bed = cv2.dilate(lab_bed, k_b, iterations=1)
        a_bed = cv2.dilate(a_bed, k_b, iterations=1)
    if a_vein.any():
        lab_vein = cv2.dilate(lab_vein, k_v, iterations=1)
        a_vein = cv2.dilate(a_vein, k_v, iterations=1)

    # ---- 背景 (float32, 光照梯度为显式低阶式, 与噪声/模糊一同构成退化模型)
    img = (np.float32(0.52) + np.float32(0.10) * _value_noise(
        H, W, rng, cells_x=max(3.0, W / 900.0 * 8.0))).astype(np.float32)
    ii = ((np.arange(H) + 0.5) / H - 0.5)[:, None].astype(np.float32)
    jj = ((np.arange(W) + 0.5) / W - 0.5)[None, :].astype(np.float32)
    ia = np.asarray(cfg["illum_amp"], float).astype(np.float32)
    img += ia[0] * ii
    img += (ia[1] * jj + ia[2] * ii * jj).astype(np.float32)

    # ---- 迹线变暗/变亮: 分行块合成, 峰值内存与格大小解耦
    B = 2048
    for r0 in range(0, H, B):
        sl = slice(r0, min(H, r0 + B))
        d = np.maximum(a_frac[sl], a_bed[sl])
        blk = img[sl]
        blk *= (np.float32(1.0) - np.float32(0.55 / 255.0) * d)
        blk += np.float32(0.35 / 255.0) * a_vein[sl]
        img[sl] = blk
    del a_frac, a_bed, a_vein

    # ---- rubble / loss
    mask_rub = np.zeros(H, bool)
    mask_los = np.zeros(H, bool)
    for lo, hi in _intervals(cfg["rubble_pct"] / 100.0, H, rng):
        mask_rub[int(lo):int(hi)] = True
    for lo, hi in _intervals(cfg["loss_pct"] / 100.0, H, rng):
        mask_los[int(lo):int(hi)] = True
    for m in (lab_frac, lab_bed, lab_vein):
        if mask_rub.any():
            m[mask_rub] = 0
        if mask_los.any():
            m[mask_los] = 0
    if mask_rub.any():
        for r0 in range(0, H, B):
            sl = slice(r0, min(H, r0 + B))
            sel = mask_rub[sl]
            if sel.any():
                img[sl][sel] = (0.5 + 0.30 * rng.standard_normal(
                    (int(sel.sum()), W))).astype(np.float32)
    if mask_los.any():
        img[mask_los] = np.float32(0.80)

    if cfg["blur_sigma_px"] > 0:
        img = ndimage.gaussian_filter(img, float(cfg["blur_sigma_px"]),
                                      mode="nearest")
    if cfg["snr_db"] != "inf":
        sig = float(np.var(img))
        nv = sig / (10.0 ** (float(cfg["snr_db"]) / 10.0))
        sd = math.sqrt(max(nv, 0.0))
        for r0 in range(0, H, B):
            sl = slice(r0, min(H, r0 + B))
            img[sl] += rng.normal(0.0, sd, img[sl].shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    u8 = np.rint(img * 255.0).astype(np.uint8)

    mask_any = ((lab_frac > 0) | (lab_bed > 0) | (lab_vein > 0)).astype(np.uint8)
    lab_fake = np.maximum(lab_bed, lab_vein)
    for rec in truth_arcs + fakes:                 # 被 rubble/loss 抹掉的弧如实标记
        rr = int(round((z_hi - rec["z_mid_m"]) / (z_hi - z_lo) * H))
        rec["visible_after_degradation"] = bool(
            0 <= rr < H and not (mask_rub[rr] or mask_los[rr]))

    return {
        "image_u8": u8, "mask_instance": lab_frac, "mask_fake_instance": lab_fake,
        "mask_any": mask_any, "mask_dark": mask_any,
        "truth_arcs": truth_arcs, "truth_fakes": fakes, "W": W, "H": H,
        "z_lo": z_lo, "z_hi": z_hi, "R_m": R, "px_mm": px_mm,
        "meta": {
            "seed": int(seed), "cfg": {k: (list(map(float, v))
                                           if isinstance(v, (list, np.ndarray)) else v)
                                       for k, v in cfg.items()},
            "scribe_eps_deg": eps, "scribe_branch": scribe_branch,
            "n_fractures_in_pop": int(dfn.n_fractures),
            "n_fractures_visible": int(len({r["fracture_id"] for r in truth_arcs})),
            "n_arcs": len(truth_arcs), "n_fakes": len(fakes),
            "rubble_rows_px": int(mask_rub.sum()), "loss_rows_px": int(mask_los.sum()),
            "W": W, "H": H, "peak_px": W * H,
        },
    }


def sha_of_render(out):
    """J5 用: 图像原始字节 + 掩膜原始字节 + 真值 JSON 字节的 sha256。"""
    h = {}
    h["image"] = hashlib.sha256(out["image_u8"].tobytes()).hexdigest()
    h["mask_instance"] = hashlib.sha256(out["mask_instance"].tobytes()).hexdigest()
    h["mask_any"] = hashlib.sha256(out["mask_any"].tobytes()).hexdigest()
    h["truth"] = hashlib.sha256(json.dumps(
        {"arcs": out["truth_arcs"], "fakes": out["truth_fakes"]},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    return h


def calib_counts(p32_levels=(3.0, 6.0, 12.0, 25.0, 50.0), n_seed=8, dpi=150,
                 out_path=None):
    """prereg.calibration: **只读计数**, 禁触任何误差量 (事前划界)。

    目的 = 把 P32 四档定在"每芯可见裂隙中位数落在 [10,40]"。
    """
    rows = []
    for p in p32_levels:
        cfg = base_cfg()
        cfg["dpi"] = dpi
        cfg["p32"] = float(p)
        cnt, arcs, ext, vis = [], [], [], []
        for s in range(1, n_seed + 1):
            o = render_core(cfg, s)
            ids = {r["fracture_id"] for r in o["truth_arcs"]}
            cnt.append(len(ids))
            arcs.append(o["meta"]["n_arcs"])
            vis.append(sum(1 for r in o["truth_arcs"]
                           if r["visible_after_degradation"]))
            ext += [r["arc_theta_deg"] for r in o["truth_arcs"]]
        e = np.array(ext) if ext else np.array([0.0])
        rows.append({
            "p32_nominal": float(p),
            "median_fractures_per_core": float(np.median(cnt)),
            "mean_fractures_per_core": float(np.mean(cnt)),
            "median_arcs_per_core": float(np.median(arcs)),
            "median_arc_theta_deg": float(np.median(e)),
            "frac_arc_theta_lt_90deg": float(np.mean(e < 90.0)),
            "frac_arc_theta_lt_30deg": float(np.mean(e < 30.0)),
            "traces_per_m_realized": float(np.mean(cnt) / (cfg["L_mm"] / 1000.0)),
        })
    usable = [r["p32_nominal"] for r in rows
              if 10.0 <= r["median_fractures_per_core"] <= 40.0]
    out = {"watermark": WATERMARK, "read_only_counts_no_error_metric": True,
           "rows": rows, "p32_levels_in_target_band": usable,
           "target_band_median_fractures_per_core": [10, 40],
           "note": "P32 为名义输入; realized 口径 = traces_per_m_realized 随表同行"}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("CALIB " + json.dumps(out, ensure_ascii=False, allow_nan=False)[:1600],
          flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)

    st = st1_roundtrip_selftest()
    print("ST1 " + json.dumps(st, sort_keys=True, allow_nan=False))
    if not st["ok"]:
        raise SystemExit("ST-1 往返自测不过 => 产状/法向约定被破坏, 拒绝继续")
    if args.calib:
        OUT.mkdir(parents=True, exist_ok=True)
        calib_counts(dpi=args.dpi, out_path=str(OUT / "calib_counts.json"))
        return 0
    if not args.smoke:
        return 0

    import time
    cfg = base_cfg()
    cfg["dpi"] = args.dpi
    t0 = time.time()
    out = render_core(cfg, args.seed)
    info = dict(out["meta"])
    info.update({"secs": round(time.time() - t0, 2),
                 "mask_frac_px": int((out["mask_instance"] > 0).sum()),
                 "mask_any_px": int(out["mask_any"].sum()),
                 "img_mean": float(out["image_u8"].mean()),
                 "img_std": float(out["image_u8"].std()),
                 "sha": sha_of_render(out)})
    print("SMOKE " + json.dumps(info, sort_keys=True, allow_nan=False))
    if args.save_png:
        CELLS.mkdir(parents=True, exist_ok=True)
        s1 = _imwrite(CELLS / f"smoke_dpi{args.dpi}_{args.seed}.png", out["image_u8"])
        s2 = _imwrite(CELLS / f"smoke_dpi{args.dpi}_{args.seed}_mask.png",
                      out["mask_any"] * 255)
        print("PNG", s1, s2, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
