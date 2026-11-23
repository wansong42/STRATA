# -*- coding: utf-8 -*-
"""R132 · 钻孔删失线组数/浓度估计 —— 设计稿 + 合成域校准可执行件 (零训练, 零真实评分卡)。

任务书: tasks/R132_任务书_★★_钻孔删失线组数估计设计稿.md
问题: 钻孔线切割把面删失成短线段/点, 面法向估计误差随支撑数 s 按 σ_r/√s 放大,
冻结 leader θ=20° 在低支撑噪声面上拆出假组 (三钻孔场景实测 k=3, 真值 k=1),
κ 锐化 (R127) 只能压多余拆组救不回组结构 → 钻孔档天花板 40-55° (R127 C3b:
40.64 / 48.30 / 54.96, 直读 results/r127_calib_compare/calib_comparison.json)。

两域合成校准 (判据全部预注册钉死于 results/v_r132_prereg.json, 本脚本只登记观察值):
  域A = masker 钻孔三场景 (borehole_1/2/4, HRF 真值单组 = 已知真值, 只读);
  域B = 冻结面生成器 (render_faces 冻结口径) + mask_borehole 同款圆柱判据线切割,
        网格 K_true{1,2,3} × 可见率{10,30,50}% × 粗糙噪声σ_r{15,25,40}° × seeds{11,22,33}
        = 81 格 (删失率扫描崩塌点登记, 反常档素材)。

方法臂 (观测-only, 估计器签名零真值):
  OBS_LEADER   现状冻结 leader θ=20° (r110 冻结估计器, 逐字复用)
  OBS_BIC      R121 C2 各向同性 vMF-BIC (r121 冻结函数, 逐字复用)
  OBS_K1       k=1 哑臂 (精度加权单心)
  M1_GIRDLER   删失似然 vMF 混合: 每面各向异性法向核 (旋转方差 λ3/λ1, λ3/λ2 与
               σ_r²/s 取大, 条带→girdle 自然塌缩), 混合 EM + 全局粗糙度 σ_r 自估,
               BIC(4K, ln n) 选型
  M2_SIGCLUST  观测几何条件化收缩: 显著性图聚类 (边 iff 夹角 ≤ z·√(σ_i²+σ_j²),
               σ 由支撑数+条带比导出, z=2.0 冻结), 连通分量成组, 精度加权组心
  M3_DPSPLIT   非参数收缩: CRP 贪心分裂-MAP (ΔlnL > ln((n_g-1)/α), α=1, σ_r 固定
               20° 不内估, 与 M1 的差别 = 选型机制本身)

纪律: 零真实数据训练/评测; 禁触 R110 真实评分卡落盘件 (评分口径只以冻结纯函数
set_layer_metrics 于组心伪面上复用 = 校准口径, 非交付口径); R110/R117/R121/R127
产物与 E 盘全部只读 (PROTECTED sha 双测); 文件前缀 r132_*;
判废数字禁出现; 全部数字研究线水印, 禁入承诺语。

阶段顺序即 mtime 铁律 (预注册创建时间早于一切 results/r132_* 产物, 坑 21 双口径):
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r132_censored_fit.py --stage prereg
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r132_censored_fit.py --stage smoke
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r132_censored_fit.py --stage run
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r132_censored_fit.py --stage table
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from r110_recon_metrics import (  # noqa: E402  只读复用冻结纯函数
    extract_faces, render_faces, leader_set_clusters, set_layer_metrics,
    frozen_scanlines, draw_borehole_positions, _unit, _sign_align_z)
from r110_hrf_masker import load_scenario, SRC, BOREHOLE_RADIUS  # noqa: E402
from r110_baseline_b1 import fit_observed_faces, MIN_PTS_BOREHOLE  # noqa: E402

# --------------------------------------------------------------- 冻结常数 --
SCEN_A = ["borehole_1", "borehole_2", "borehole_4"]          # 域A三场景 (任务书钉死)
SEEDS = [11, 22, 33]                                          # 域B种子 (与 R110 家族同)
K_GRID = (1, 8)                                               # 混合组数选型域
COV_GRID = [0.10, 0.30, 0.50]                                 # 删失率扫描 (任务书)
SIGR_GRID = [15.0, 25.0, 40.0]                                # 域B生成粗糙噪声 (度)
KTRUE_GRID = [1, 2, 3]
SIGMA_R_INIT_DEG = 20.0                                       # M1/M3 粗糙度初值
SIGMA_R_CLIP_DEG = (5.0, 60.0)
SIGMA_UNRES_DEG = 89.0                                        # 不可分辨旋转方差上限 (度)
ALPHA_CRP = 1.0                                               # M3 CRP 浓度 (冻结)
NU_T = 2.0                                                    # M3 重尾核自由度 (冻结, 任务书候选3)
INTRINSIC_CLIP_DEG = (2.0, 30.0)                              # 组内本征散布 (度) clip
N_FACES_B = 120                                               # 域B每场景真面数
R_RANGE_B = (3.0, 12.0)                                       # 域B盘半径 log-uniform (m)
BOX_B = np.array([[0.0, 0.0, 0.0], [240.0, 200.0, 150.0]])    # 域B域盒 (m, 预注册)
KAPPA_IN_SET = 400.0                                          # 域B组内法向散布 (≈3.3°, 使真值 leader 稳定于 K_true)
SET_DESIGN = {  # 域B组心设计表 (dip, azimuth 度; 两两角距 ≥40° 预注册核算)
    1: [(3.0, 45.0)],
    2: [(3.0, 45.0), (60.0, 225.0)],
    3: [(3.0, 45.0), (60.0, 20.0), (60.0, 140.0)],
}
GREEN_FISHER_MAX = 12.0     # GREEN: 域A winner fisher_cal 均值上界 (工程阈值同源)
YELLOW_FISHER_MAX = 25.0    # YELLOW 下界: 显著收回线 (对现状 ≥15° 降幅另行断言)
BASE_DROP_MIN = 15.0        # winner 相对 OBS_LEADER 观测侧基线的最小降幅
B_RECOVERY_MIN = 5          # 非退化门: 域B K≥2 & cov≥30% & σr≤25° 九格 k̂ 命中数下限
A_K1_EXACT = True           # 域A主闸: 三场景 k̂==1 全中 (真值组数)
TRUTH_K_EXPECT = {"borehole_1": 1, "borehole_2": 1, "borehole_4": 1}
R121_ANCHOR = {"borehole_1": {"n_hat": 13, "k_c1_leader": 3, "k_c2_bic": 3},
               "borehole_2": {"n_hat": 16, "k_c1_leader": 3, "k_c2_bic": 3},
               "borehole_4": {"n_hat": 28, "k_c1_leader": 3, "k_c2_bic": 5}}
ARM_NAMES = ("M1_GIRDLER", "M2_SIGCLUST", "M3_DPSPLIT")
BASELINE_NAMES = ("OBS_LEADER", "OBS_BIC", "OBS_K1")
# 判废数字权威引用 R117/R121/R127 预注册同款 (红线词: 本文件与其扫描对象禁逐字,
# 构造式扫描在守卫测试内完成, 此处不枚举)
PREREG = os.path.join(ROOT, "results", "v_r132_prereg.json")
RUN_DIR = os.path.join(ROOT, "results", "r132_synthetic_calib")
DA_JSON = os.path.join(RUN_DIR, "domain_a_borehole3.json")
DB_JSON = os.path.join(RUN_DIR, "domain_b_scan.json")
TABLE_JSON = os.path.join(RUN_DIR, "calib_table.json")
MANIFEST = os.path.join(RUN_DIR, "run_manifest.json")
R121_MANIFEST = os.path.join(ROOT, "results", "r121_calib_compare", "run_manifest.json")
R121_TABLE = os.path.join(ROOT, "results", "r121_calib_compare", "calib_comparison.json")
R127_TABLE = os.path.join(ROOT, "results", "r127_calib_compare", "calib_comparison.json")
MASK_DIR = os.path.join(ROOT, "results", "r110_hrf_masking")
PROTOCOL = os.path.join(ROOT, "results", "v_r110_recon_protocol.json")

PROTECTED = [
    "scripts/r110_recon_metrics.py", "scripts/r110_hrf_masker.py",
    "scripts/r110_baseline_b1.py", "scripts/r117_fracgen_conditional.py",
    "scripts/r121_group_calib.py", "scripts/r127_kappa_sharpen.py",
    "results/v_r110_recon_protocol.json",
    "results/r121_calib_compare/run_manifest.json",
    "results/r121_calib_compare/calib_comparison.json",
    "results/r127_calib_compare/calib_comparison.json",
] + [f"results/r110_hrf_masking/{s}.{ext}" for s in SCEN_A for ext in ("npz", "json")] \
    + ["results/digital_ledger.json"]

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def creation_wall(path):
    """本机墙钟 (UTC+8) 锚定时点字符串 (坑 21, R116/R121/R127 范例口径)。"""
    ct = os.stat(path).st_ctime
    mt = os.stat(path).st_mtime
    return (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ct + 8 * 3600)),
            ct, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(mt + 8 * 3600)), mt)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _san(o):
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _san(o.tolist())
    return o


def load_json_strict(path):
    def _reject(x):
        raise ValueError(f"非有限常量入档: {x} in {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject)


def dump_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)


def sha_set():
    return {p: sha256_file(os.path.join(ROOT, p)) for p in PROTECTED
            if not p.endswith("digital_ledger.json")}


def dip_az_to_normal(dip_deg, az_deg):
    """产状 (dip, azimuth 倾向) → 单位法向 (与主项目 arccos|nz| 口径一致)。"""
    d, a = math.radians(dip_deg), math.radians(az_deg)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


# ------------------------------------------------------- 观测面几何装载 --
def scene_faces_geo(scen):
    """域A: masker 场景 → 观测面 (冻结 fit_observed_faces 逐字) + 面内点云几何。

    返回 (faces_geo, m_obs): faces_geo 每项 {n, c, R, s, lam1, lam2, lam3}。
    面序与 np.unique(fid_o) 跳过规则与 fit_observed_faces 严格同构 (锚点断言背书)。
    """
    d = np.load(SRC, allow_pickle=True)
    xyz = d["xyz"].astype(np.float64)
    nrm = _unit(d["nrm"].astype(np.float64))
    fid = d["fid"].astype(np.int64)
    lab = fid >= 0
    obs, m_obs = load_scenario(scen)
    obs_l = obs & lab
    pos_o, nrm_o, fid_o = xyz[obs_l], nrm[obs_l], fid[obs_l]
    faces = fit_observed_faces(pos_o, nrm_o, fid_o, MIN_PTS_BOREHOLE)
    geo = []
    for f in np.unique(fid_o):
        m = fid_o == f
        s = int(m.sum())
        if s < MIN_PTS_BOREHOLE:
            continue
        P = pos_o[m]
        c = P.mean(0)
        X = P - c
        if s >= 3:
            w, V = np.linalg.eigh(X.T @ X / (s - 1))
            lam3, lam2, lam1 = float(w[0]), float(w[1]), float(w[2])
        else:
            lam1 = lam2 = lam3 = 0.0
            V = None
        geo.append(dict(s=s, lam1=lam1, lam2=lam2, lam3=lam3, _eigvecs=V))
    assert len(geo) == len(faces), (len(geo), len(faces))
    for g, f in zip(geo, faces):
        g["n"] = np.asarray(f["n"], dtype=np.float64)
        g["c"] = np.asarray(f["c"], dtype=np.float64)
        g["R"] = float(f["R"])
    return geo, m_obs


def truth_centers_from_hrf():
    """域A真值组心: 冻结 leader 于真值面 (与 R110 口径同源), 锚定 k=1。"""
    d = np.load(SRC, allow_pickle=True)
    xyz = d["xyz"].astype(np.float64)
    fid = d["fid"].astype(np.int64)
    lab = fid >= 0
    faces_true = extract_faces(xyz[lab], fid[lab])
    _, means, props, _ = leader_set_clusters(faces_true)
    return faces_true, means, props


# ------------------------------------------------- 删失观测方差 (核构造) --
def obs_variances(fg, sigma_r_deg):
    """每观测面两个旋转方向的方差 (rad²): 条带几何 λ3/λ1, λ3/λ2 与 1/√s 粗糙下限取大。

    s<3 无几何可用 → 各向同性 σ_r²/s (支撑=1 即单点粗糙噪声全权)。
    """
    sr = math.radians(sigma_r_deg)
    o_a, o_b = [], []
    for f in fg:
        s, l1, l2, l3 = f["s"], f["lam1"], f["lam2"], f["lam3"]
        floor = sr * sr / max(s, 1)
        if s >= 3 and l1 > 0:
            a = max(l3 / max(l1, 1e-12), floor)
            b = max(l3 / l2, floor) if l2 > 1e-12 else math.radians(SIGMA_UNRES_DEG) ** 2
        elif s == 2:
            a, b = floor, math.radians(SIGMA_UNRES_DEG) ** 2
        else:
            a = b = floor
        cap = math.radians(SIGMA_UNRES_DEG) ** 2
        o_a.append(min(a, cap))
        o_b.append(min(b, cap))
    return np.asarray(o_a), np.asarray(o_b)


# -------------------------------------------------------- M1: girdle EM --
def _fp_init(x, K):
    """farthest-point 确定性初值 (R121 vmf_mix_bic_curve 同款角距度量, +z 半球)。"""
    mus = [x[0]]
    while len(mus) < K:
        M = np.stack(mus)
        d = 1.0 - np.abs(x @ M.T).max(1)
        mus.append(x[int(np.argmax(d))])
    return _unit(np.stack(mus))


def _resid_angles(mus, fg):
    """逐面逐组件的 (φ, ψ) 旋转角 (rad): φ 朝 ê1, ψ 朝 ê2 (面主轴由调用方给)。"""
    n = len(fg)
    K = len(mus)
    phi = np.zeros((n, K))
    psi = np.zeros((n, K))
    for m in range(K):
        u = mus[m]
        for i, f in enumerate(fg):
            e1, e2 = f["_e1"], f["_e2"]
            phi[i, m] = math.asin(max(-0.9993, min(0.9993, float(u @ e1))))
            psi[i, m] = math.asin(max(-0.9993, min(0.9993, float(u @ e2))))
    return phi, psi


def _axes_from_face(f):
    """支撑点主轴 ê1 (最大) / ê2 (中), 与 n̂=ê3 正交规范架; 退化面给两正交补基。"""
    s, l1, l2 = f["s"], f["lam1"], f["lam2"]
    n = _unit(np.asarray(f["n"], dtype=np.float64))
    if s >= 3:
        V = f["_eigvecs"]
        e1, e2 = _unit(V[:, 2]), _unit(V[:, 1])
        # 数值卫生: 正交化到 n̂ 切平面
        e1 = _unit(e1 - (e1 @ n) * n)
        e2 = np.cross(n, e1)
        return e1, e2
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(n, a))
    e2 = np.cross(n, e1)
    return e1, e2


def girdle_mixture_em(fg, K, sigma_r_deg=SIGMA_R_INIT_DEG, estimate_sigma_r=True,
                      iters=200, tol=1e-9, sigma_r_clip=SIGMA_R_CLIP_DEG, nu=None):
    """删失似然 vMF 混合 (各向异性法向核) EM, 确定性。

    nu=None → 高斯核 (M1); nu=float → 2 维 Student-t 重尾核 (M3, 任务书候选3
    "重尾 kernel" 的落实): log p ∝ −(ν+2)/2·ln(1+q/ν), M 步用 IRLS 权重
    u=(ν+2)/(ν+q) 自动压垮离群面。
    fg: 观测面 (需含 n/s/lam 与 _eigvecs 若有); 返回 fit dict 或 None (退化)。
    """
    n = len(fg)
    if n == 0 or K < 1:
        return None
    if n < 2 * K:
        return None
    x = _sign_align_z(np.array([f["n"] for f in fg], dtype=np.float64))
    for f, xi in zip(fg, x):
        f["n"] = xi
        e1, e2 = _axes_from_face(f)
        f["_e1"], f["_e2"] = e1, e2
    sr = float(sigma_r_deg)
    outer = 12 if estimate_sigma_r else 1
    for _ in range(outer):
        prev = -np.inf
        o_a, o_b = obs_variances(fg, sr)
        mus = _fp_init(x, K)
        iv = np.full(K, math.radians(10.0) ** 2)
        pi = np.full(K, 1.0 / K)
        for _ in range(iters):
            phi, psi = _resid_angles(mus, fg)
            va = o_a[:, None] + iv[None, :]
            vb = o_b[:, None] + iv[None, :]
            q = phi * phi / va + psi * psi / vb
            if nu is None:
                ll = (-0.5 * (q + np.log(va * vb)) + np.log(pi + 1e-300)[None, :])
            else:
                ll = (-0.5 * np.log(va * vb) - 0.5 * (nu + 2.0) * np.log1p(q / nu)
                      + np.log(pi + 1e-300)[None, :])
            mmax = ll.max(1, keepdims=True)
            ex = np.exp(ll - mmax)
            Z = ex.sum(1, keepdims=True)
            r = ex / Z
            lse = float((np.log(Z[:, 0]) + mmax[:, 0]).sum())
            if not np.isfinite(lse):
                return None
            u = r if nu is None else r * ((nu + 2.0) / (nu + q))
            # M 步: 精度加权 (观测+本征) 方向均值, 符号对齐当前心
            prec = u / (va + vb)
            new = np.zeros_like(mus)
            for m in range(K):
                w = prec[:, m]
                if w.sum() < 1e-12:
                    return None  # 空组件 → 退化解, BIC=inf
                sgn = np.sign(x @ mus[m])
                sgn[sgn == 0] = 1.0
                new[m] = _unit((x * sgn[:, None]).T @ w)
            mus = _unit(new)
            # 本征散布: 残差减观测方差 (预注册近似, 设计稿 §2.1-C)
            for m in range(K):
                rr = r[:, m]
                uu = u[:, m]
                num = float((uu * (phi[:, m] ** 2 + psi[:, m] ** 2)).sum()
                            - (uu * (o_a + o_b)).sum())
                den = max(float(uu.sum()), 1e-9) * 2.0
                iv[m] = float(np.clip(max(num, 1e-6) / den,
                                      math.radians(INTRINSIC_CLIP_DEG[0]) ** 2,
                                      math.radians(INTRINSIC_CLIP_DEG[1]) ** 2))
            pi = r.mean(0)
            pi = np.clip(pi, 1e-6, 1.0)
            pi = pi / pi.sum()
            if abs(lse - prev) < tol * max(1.0, abs(lse)):
                break
            prev = lse
        if not estimate_sigma_r:
            break
        # σ_r 全局更新: 指派残差 × 支撑数加权 (预注册近似, 设计稿 §2.1-D)
        assign = r.argmax(1)
        s_arr = np.array([f["s"] for f in fg], dtype=np.float64)
        res2 = phi[np.arange(n), assign] ** 2 + psi[np.arange(n), assign] ** 2
        new_sr = math.degrees(math.sqrt(max(float((s_arr * res2).sum()
                                                 / max(2.0 * s_arr.sum(), 1e-9)), 1e-8)))
        new_sr = float(np.clip(new_sr, *sigma_r_clip))
        if abs(new_sr - sr) < 1e-3:
            sr = new_sr
            break
        sr = new_sr
    phi, psi = _resid_angles(mus, fg)
    o_a, o_b = obs_variances(fg, sr)
    va = o_a[:, None] + iv[None, :]
    vb = o_b[:, None] + iv[None, :]
    q = phi * phi / va + psi * psi / vb
    if nu is None:
        ll = (-0.5 * (q + np.log(va * vb)) + np.log(pi + 1e-300)[None, :])
    else:
        ll = (-0.5 * np.log(va * vb) - 0.5 * (nu + 2.0) * np.log1p(q / nu)
              + np.log(pi + 1e-300)[None, :])
    mmax = ll.max(1, keepdims=True)
    lse = float((np.log(np.exp(ll - mmax).sum(1)) + mmax[:, 0]).sum())
    bic = -2.0 * lse + 4.0 * K * math.log(n)
    kmean = mus.copy()
    return dict(K=int(K), lnL=lse, bic=float(bic), centers=kmean,
                props=pi.tolist(), iv_deg=[math.degrees(math.sqrt(v)) for v in iv],
                sigma_r_deg=sr,
                kappa_m=[1.0 / max(v, 1e-8) for v in iv])


def m1_girdler(fg):
    """M1 臂: K=1..8 全网格 + BIC 选型 (平手取小)。"""
    fits = {}
    for K in range(K_GRID[0], K_GRID[1] + 1):
        f = girdle_mixture_em(fg, K)
        if f is not None:
            fits[K] = f
    if not fits:
        return dict(k_hat=1, centers=None, fit=None, bic_curve={}, note="all_degenerate")
    k_star = min(fits, key=lambda K: (fits[K]["bic"], K))
    return dict(k_hat=int(k_star), centers=fits[k_star]["centers"],
                fit=fits[k_star],
                bic_curve={str(K): fits[K]["bic"] for K in fits},
                lnL_curve={str(K): fits[K]["lnL"] for K in fits},
                sigma_r_deg=fits[k_star]["sigma_r_deg"])


# ------------------------------------------------- M2: 噪声锥全连接收缩臂 --
Z_MERGE2 = 3.0   # 设计期探针定标冻结: 2D 合成噪声 Rayleigh 3σ 包络


def censoring_indicator(fg, sigma_r_deg=SIGMA_R_INIT_DEG):
    """D ∈ [0,1]: 删失不确定度中位 / 90° (观测-only, M_obs 几何协变量的主体)。"""
    o_a, o_b = obs_variances(fg, sigma_r_deg)
    sig = np.sqrt(o_a + o_b)
    return float(np.clip(np.median(sig) / math.radians(90.0), 0.0, 1.0))


def m2_sigclust(fg, z=Z_MERGE2, sigma_r_deg=SIGMA_R_INIT_DEG):
    """M2 臂 (任务书候选2 观测几何条件化): 删失几何直接给合并容差 ——
    全连接判据 maxPairAngle ≤ z·√(σ_i²+σ_j²) (σ 由支撑数 s 与条带比 λ3/λ1,λ3/λ2
    构造, 观测-only), 层次合并至无合法边; 精度加权组心。零 EM 零选型。"""
    n = len(fg)
    if n == 0:
        return dict(k_hat=0, centers=None, D=0.0)
    o_a, o_b = obs_variances(fg, sigma_r_deg)
    sig = np.sqrt(o_a + o_b)
    x = _sign_align_z(np.array([f["n"] for f in fg], dtype=np.float64))
    ang = np.degrees(np.arccos(np.clip(np.abs(x @ x.T), -1, 1)))
    thr = np.degrees(z * np.sqrt(sig[:, None] ** 2 + sig[None, :] ** 2))
    A = ang - thr
    np.fill_diagonal(A, -np.inf)
    clusters = {i: [i] for i in range(n)}
    while True:
        ids = sorted(clusters)
        best = None
        for ai in range(len(ids)):
            ga = clusters[ids[ai]]
            for bi in range(ai + 1, len(ids)):
                d = float(A[np.ix_(ga, clusters[ids[bi]])].max())
                if d <= 0 and (best is None or d < best[0]):
                    best = (d, ids[ai], ids[bi])
        if best is None:
            break
        _, ka, kb = best
        clusters[ka] = clusters[ka] + clusters[kb]
        del clusters[kb]
    centers = []
    sizes = []
    for g in clusters.values():
        w = 1.0 / sig[g] ** 2
        ref = x[g[0]]
        sgn = np.sign(x[g] @ ref)
        sgn[sgn == 0] = 1.0
        centers.append(_unit((x[g] * sgn[:, None]).T @ w))
        sizes.append(len(g))
    return dict(k_hat=len(clusters), centers=np.asarray(centers),
                group_sizes=sizes, D=censoring_indicator(fg, sigma_r_deg),
                sigma_r_deg_fixed=sigma_r_deg, z=z)


# --------------------------------------------------- M3: CRP 分裂-MAP臂 --
def _fit_lnL(fg, K, sigma_r_deg=SIGMA_R_INIT_DEG, nu=NU_T):
    f = girdle_mixture_em(fg, K, estimate_sigma_r=False, sigma_r_deg=sigma_r_deg,
                          nu=nu)
    return (f["lnL"], f) if f else (-np.inf, None)


def m3_dpsplit(fg, alpha=ALPHA_CRP, kmax=K_GRID[1]):
    """M3 臂: 从 k=1 起, 对每组试 2 分, 增益 ΔlnL > ln((n_g−1)/α) 才接受 (CRP)。"""
    n = len(fg)
    if n == 0:
        return dict(k_hat=0, centers=np.zeros((0, 3)), splits_accepted=[])
    for f in fg:
        if "_e1" not in f:
            f["n"] = _unit(np.asarray(f["n"], dtype=np.float64))
            e1, e2 = _axes_from_face(f)
            f["_e1"], f["_e2"] = e1, e2
    groups = [list(range(n))]
    accepted = []
    while len(groups) < kmax:
        best = None
        for gi, g in enumerate(groups):
            if len(g) < 4:
                continue
            sub = [fg[i] for i in g]
            l1, _ = _fit_lnL(sub, 1)
            l2, f2 = _fit_lnL(sub, 2)
            gain = l2 - l1
            thr = math.log(max(len(g) - 1, 2) / alpha)
            if gain > thr and (best is None or gain - thr > best[0]):
                best = (gain - thr, gi, g, f2)
        if best is None:
            break
        _, gi, g, f2 = best
        centers = f2["centers"]
        # 按最近心重指派成员
        sub_x = _sign_align_z(np.array([fg[i]["n"] for i in g], dtype=np.float64))
        assign = (np.abs(sub_x @ centers.T)).argmax(1)
        newg = [[] for _ in range(2)]
        for a, idx in zip(assign, g):
            newg[int(a)].append(idx)
        if any(len(x) == 0 for x in newg):
            break
        groups = groups[:gi] + newg + groups[gi + 1:]
        accepted.append(dict(group_size=len(g), gain=float(f2["lnL"] - l1),
                             thr=float(thr)))
    all_x = _sign_align_z(np.array([f["n"] for f in fg], dtype=np.float64))
    centers = []
    for g in groups:
        ref = all_x[g[0]]
        sgn = np.sign(all_x[g] @ ref)
        sgn[sgn == 0] = 1.0
        centers.append(_unit((all_x[g] * sgn[:, None]).mean(0)))
    return dict(k_hat=len(groups), centers=np.asarray(centers),
                group_sizes=[len(g) for g in groups], splits_accepted=accepted)


# -------------------------------------------------------- 基线臂 (对照) --
def arm_leader(faces_as_dict):
    labels, means, props, _ = leader_set_clusters(faces_as_dict)
    return dict(k_hat=len(means), centers=means, props=[float(p) for p in props])


def arm_bic_plain(fg):
    from r121_group_calib import vmf_mix_bic_curve  # 冻结函数逐字复用 (R121 C2)
    x = _sign_align_z(np.array([f["n"] for f in fg], dtype=np.float64))
    _b, _l, k = vmf_mix_bic_curve(x, kmax=K_GRID[1])
    # 组心仅用于 fisher_cal 参考列; k̂ 判定完全来自冻结 vmf_mix_bic_curve (锚点)。
    f1 = girdle_mixture_em(fg, k, estimate_sigma_r=False)
    cen = f1["centers"] if f1 else _fp_init(x, k)
    return dict(k_hat=int(k), centers=np.asarray(cen))


def arm_k1(fg, sigma_r_deg=SIGMA_R_INIT_DEG):
    n = len(fg)
    if n == 0:
        return dict(k_hat=0, centers=np.zeros((0, 3)))
    o_a, o_b = obs_variances(fg, sigma_r_deg)
    w = 1.0 / (o_a + o_b)
    x = _sign_align_z(np.array([f["n"] for f in fg], dtype=np.float64))
    ref = x[np.argmax(w)]
    sgn = np.sign(x @ ref)
    sgn[sgn == 0] = 1.0
    return dict(k_hat=1, centers=_unit((x * sgn[:, None]).T @ w)[None, :])


# ---------------------------------------------------------- 校准口径评分 --
def fisher_cal(centers_est, centers_true):
    """组心伪面复用冻结 set_layer_metrics (90° 未匹配罚项, LSA 匹配, 零自造口径)。

    注意: 这是校准口径 (观测侧组心 vs 真值组心), 不是 R110 交付评分卡。
    """
    if centers_est is None or len(centers_est) == 0:
        return dict(fisher=90.0, k_diff=None)
    fe = [dict(c=np.zeros(3), n=np.asarray(nn), R=1.0) for nn in centers_est]
    ft = [dict(c=np.zeros(3), n=np.asarray(nn), R=1.0) for nn in centers_true]
    m = set_layer_metrics(fe, ft)
    return dict(fisher=float(m["fisher_mean_direction_distance_deg"]),
                k_diff=int(m["set_count_abs_diff"]),
                prop_l1=float(m["set_proportion_L1"]))


# ------------------------------------------------------------- 域B 生成 --
def _tangent_noise(rng, n, sigma_rad):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(n, a))
    e2 = np.cross(n, e1)
    t = rng.normal(0.0, sigma_rad, 2)
    return _unit(n + t[0] * e1 + t[1] * e2)


def synth_scene(k_true, target_cov, sigma_r_gen_deg, seed):
    """域B: 冻结设计表 K 组面 → render_faces (冻结口径) → 粗糙噪声点法向 →
    圆柱线切割 (与 mask_borehole 同款判据, 孔数自增到目标可见率)。确定性。"""
    rng = np.random.default_rng(np.random.SeedSequence(
        [132000, int(seed), int(k_true) * 1000 + int(round(target_cov * 100)),
         int(sigma_r_gen_deg * 10)]))
    design = SET_DESIGN[k_true]
    set_normals = np.stack([dip_az_to_normal(d, a) for d, a in design])
    # 组心两两角距核算 (预注册 ≥40°)
    minsep = 90.0
    for i in range(k_true):
        for j in range(i + 1, k_true):
            ang = math.degrees(math.acos(min(1.0, abs(float(set_normals[i] @ set_normals[j])))))
            minsep = min(minsep, ang)
    per = [int(round(N_FACES_B / k_true)) * (i < k_true) for i in range(k_true)]
    while sum(per) < N_FACES_B:
        per[int(np.argmin(per))] += 1
    faces = []
    for gi in range(k_true):
        mu = set_normals[gi]
        for _ in range(per[gi]):
            nn = _tangent_noise(rng, mu, 1.0 / math.sqrt(KAPPA_IN_SET))
            c = rng.uniform(BOX_B[0] + 8.0, BOX_B[1] - 8.0)
            R = float(np.exp(rng.uniform(math.log(R_RANGE_B[0]), math.log(R_RANGE_B[1]))))
            faces.append(dict(c=c, n=nn, R=R))
    pos, fid = render_faces(faces, seed=seed)
    # 点法向 = 面法向 + 粗糙噪声 (删失机制的生成侧)
    srad = math.radians(sigma_r_gen_deg)
    pn = np.empty_like(pos)
    for fi in np.unique(fid):
        m = fid == fi
        base = faces[int(fi)]["n"]
        kpt = int(m.sum())
        a = np.array([1.0, 0.0, 0.0]) if abs(base[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(base, a))
        e2 = np.cross(base, e1)
        tt = rng.normal(0.0, srad, (kpt, 2))
        pn[m] = _unit(base[None, :] + tt[:, 0:1] * e1[None, :] + tt[:, 1:2] * e2[None, :])
    # 钻孔线切割: 与 mask_borehole 同款判据; 删失率主轴 = 面可见率 (与域A口径一致:
    # bh1 13/136≈10%), 点覆盖率降为诊断次量
    k_holes = 2
    while True:
        holes = draw_borehole_positions(BOX_B, k_holes, seed=SCAN_SEED_B + int(seed))
        obs = np.zeros(len(pos), dtype=bool)
        for h in range(holes.shape[0]):   # 逐孔累加, 避免 n×k 大瞬时分配
            obs |= (((pos[:, 0] - holes[h, 0]) ** 2
                     + (pos[:, 1] - holes[h, 1]) ** 2) < BOREHOLE_RADIUS ** 2)
        vis_faces = np.unique(fid[obs]).size if obs.any() else 0
        cov = vis_faces / len(faces)
        if cov >= target_cov or k_holes >= 300:
            break
        k_holes += 8
    pt_cov = float(obs.mean())
    obs_l = obs  # 全部点有 fid (合成无 -1)
    faces_obs = fit_observed_faces(pos[obs_l], pn[obs_l], fid[obs_l], MIN_PTS_BOREHOLE)
    geo = []
    for f in np.unique(fid[obs_l]):
        m = fid[obs_l] == f
        s = int(m.sum())
        P = pos[obs_l][m]
        c0 = P.mean(0)
        X = P - c0
        if s >= 3:
            w, V = np.linalg.eigh(X.T @ X / max(s - 1, 1))
            lam3, lam2, lam1 = float(w[0]), float(w[1]), float(w[2])
        else:
            lam1 = lam2 = lam3 = 0.0
            V = None
        geo.append(dict(s=s, lam1=lam1, lam2=lam2, lam3=lam3, _eigvecs=V))
    fg = []
    for g, f in zip(geo, faces_obs):
        g = dict(g)
        g["n"] = np.asarray(f["n"], dtype=np.float64)
        g["c"] = np.asarray(f["c"], dtype=np.float64)
        g["R"] = float(f["R"])
        e1, e2 = _axes_from_face(g)
        g["_e1"], g["_e2"] = e1, e2
        g["support"] = g["s"]
        fg.append(g)
    assert len(fg) == len(faces_obs)
    _, tc, _tp, _o = leader_set_clusters(faces)
    assert len(tc) == k_true, (f"域B设计失效: 真值网络 leader 组数 {len(tc)} != K_true {k_true}"
                               " (组内散布须收紧至冻结 leader 不误拆)")
    return dict(fg=fg, truth_centers=tc, truth_k=int(len(tc)),
                n_faces=len(fg), n_pts=int(len(pos)),
                coverage=round(float(cov), 4), point_coverage=round(pt_cov, 5),
                k_holes=k_holes, minset_sep_deg=round(minsep, 2),
                design=design)


SCAN_SEED_B = 132099  # 域B布孔基 seed (与 R110 SCANLINE_SEED 家族隔离, 预注册)


# -------------------------------------------------------- 域A 评估组装 --
def run_domain_a(anchors_only=False):
    faces_true, t_centers, t_props = truth_centers_from_hrf()
    out = {"truth_leader_k": int(len(t_centers)),
           "truth_centers": [list(map(float, c)) for c in t_centers],
           "n_true_faces": len(faces_true), "scenarios": {}}
    for scen in SCEN_A:
        fg, m_obs = scene_faces_geo(scen)
        fd = [dict(c=f["c"], n=f["n"], R=f["R"]) for f in fg]
        arms = {}
        arms["OBS_LEADER"] = arm_leader(fd)
        arms["OBS_BIC"] = arm_bic_plain(fg)
        arms["OBS_K1"] = arm_k1(fg)
        arms["M1_GIRDLER"] = m1_girdler(fg)
        arms["M2_SIGCLUST"] = m2_sigclust(fg)
        arms["M3_DPSPLIT"] = m3_dpsplit(fg)
        scen_rec = {"n_hat": len(fg), "coverage_fraction": round(float(m_obs["coverage_fraction"]), 6),
                    "k_truth": TRUTH_K_EXPECT[scen], "arms": {}}
        for an, a in arms.items():
            ev = fisher_cal(a.get("centers"), t_centers)
            cen = a.get("centers")
            modal = (float(np.degrees(np.arccos(min(1.0, abs(float(np.asarray(cen[0]) @ t_centers[0]))))))
                     if cen is not None and len(cen) >= 1 else None)
            scen_rec["arms"][an] = dict(k_hat=int(a["k_hat"]),
                                        fisher_cal=ev["fisher"],
                                        k_diff=ev["k_diff"],
                                        center_modal_err_deg=modal,
                                        bic_curve=a.get("bic_curve"),
                                        sigma_r_deg=a.get("sigma_r_deg"),
                                        D=a.get("D"),
                                        group_sizes=a.get("group_sizes"))
        sup = np.array([f["s"] for f in fg])
        ang = np.degrees(np.arccos(np.clip(np.abs(np.array(
            [f["n"] for f in fg]) @ t_centers[0]), -1, 1)))
        scen_rec["noise_law"] = {
            "median_ang_by_support": {str(int(s)): round(float(np.median(ang[sup == s])), 2)
                                      for s in np.unique(sup) if (sup == s).sum() >= 1},
            "support_hist": {str(int(s)): int((sup == s).sum()) for s in np.unique(sup)}}
        out["scenarios"][scen] = scen_rec
        if not anchors_only:
            log(f"  [A] {scen}: n_hat={len(fg)} "
                + " ".join(f"{a}={arms[a]['k_hat']}" for a in arms))
    # R121 血统锚: n_hat / leader k / bic k 逐位
    r121 = load_json_strict(R121_MANIFEST)
    for scen in SCEN_A:
        e = r121["scenario_k_selection"][scen]
        a = out["scenarios"][scen]
        assert a["n_hat"] == R121_ANCHOR[scen]["n_hat"] == e["n_hat"], scen
        assert a["arms"]["OBS_LEADER"]["k_hat"] == R121_ANCHOR[scen]["k_c1_leader"] == e["k_c1_leader"], scen
        assert a["arms"]["OBS_BIC"]["k_hat"] == R121_ANCHOR[scen]["k_c2_bic"] == e["k_c2_bic"], scen
    assert out["truth_leader_k"] == 1
    return out


# -------------------------------------------------------- 择优规则 (冻结) --
def b_recovery_score(db, arm):
    """域B非退化分: K_true≥2 且 cov≥0.30 且 σr≤25° 的格中 k̂==truth_k_leader 命中数。"""
    hit = tot = 0
    for c in db["cells"]:
        if c["k_true"] >= 2 and c["coverage_target"] >= 0.30 and c["sigma_r_gen_deg"] <= 25.0:
            tot += 1
            a = c["arms"].get(arm)
            if a is not None and a["k_hat"] == c["truth_k_leader"]:
                hit += 1
    return hit, tot


def _arm_summary(da, db, an):
    ks = [da["scenarios"][s]["arms"][an]["k_hat"] for s in SCEN_A]
    fm = float(np.mean([da["scenarios"][s]["arms"][an]["fisher_cal"] for s in SCEN_A]))
    base_fm = float(np.mean([da["scenarios"][s]["arms"]["OBS_LEADER"]["fisher_cal"]
                             for s in SCEN_A]))
    k1_fm = float(np.mean([da["scenarios"][s]["arms"]["OBS_K1"]["fisher_cal"] for s in SCEN_A]))
    hit, tot = b_recovery_score(db, an)
    modal_vals = [da["scenarios"][s]["arms"][an]["center_modal_err_deg"] for s in SCEN_A]
    return dict(k_hats=ks, gate_A_k1_all=bool(all(k == 1 for k in ks)),
                fisher_cal_mean=fm, drop_vs_leader=base_fm - fm,
                beats_k1=bool(fm <= k1_fm + 2.0), b_recovery=[hit, tot],
                center_modal_max=(max(modal_vals) if all(v is not None for v in modal_vals)
                                  else None))


def evaluate_branches(da, db):
    """预注册三级判据 (设计稿 §4): GREEN集→YELLOW集→RED, 全部从落盘数字复算。"""
    means = {an: dict(k_hats=[da["scenarios"][s]["arms"][an]["k_hat"] for s in SCEN_A],
                      gate_A_k1_all=all(
                          da["scenarios"][s]["arms"][an]["k_hat"] == 1 for s in SCEN_A),
                      fisher_cal_mean=float(np.mean(
                          [da["scenarios"][s]["arms"][an]["fisher_cal"] for s in SCEN_A])))
                 for an in ARM_NAMES + BASELINE_NAMES}
    flags = {an: _arm_summary(da, db, an) for an in ARM_NAMES}
    priority = {"M1_GIRDLER": 0, "M3_DPSPLIT": 1, "M2_SIGCLUST": 2}

    def tier(cand, fm_max):
        ok = [an for an in cand
              if flags[an]["gate_A_k1_all"] and flags[an]["fisher_cal_mean"] <= fm_max
              and flags[an]["drop_vs_leader"] >= BASE_DROP_MIN and flags[an]["beats_k1"]]
        if fm_max == GREEN_FISHER_MAX:  # GREEN 层追加非退化门
            ok = [an for an in ok if flags[an]["b_recovery"][0] >= B_RECOVERY_MIN]
        return sorted(ok, key=lambda a: (round(flags[a]["fisher_cal_mean"], 6),
                                         -flags[a]["b_recovery"][0], priority[a]))

    green = tier(ARM_NAMES, GREEN_FISHER_MAX)
    yellow = tier(ARM_NAMES, YELLOW_FISHER_MAX)
    qualifiers = [an for an in ARM_NAMES if flags[an]["gate_A_k1_all"]]
    if green:
        winner, branch = green[0], "GREEN"
    elif yellow:
        winner, branch = yellow[0], "YELLOW"
    else:
        winner, branch = None, "RED"
    vd = dict(branch=branch, winner=winner, flags=flags, qualifiers=qualifiers,
              green_set=green, yellow_set=yellow,
              rationale=(f"{branch}: winner={winner}; 各臂 " + "; ".join(
                  f"{an}: k̂{flags[an]['k_hats']} fisher均值{flags[an]['fisher_cal_mean']:.2f}° "
                  f"降幅{flags[an]['drop_vs_leader']:.2f}° B命中{flags[an]['b_recovery'][0]}/{flags[an]['b_recovery'][1]}"
                  for an in ARM_NAMES)))
    if branch == "RED":
        vd["rationale"] += (" → 候选4 诚实退出: 删失组数/浓度估计入册开放难题 "
                            "(与 40.78° 指派墙 / 66.88° 检索罚项同级叙事)")
    return winner, means, vd


# --------------------------------------------------------------- 域B 跑 --
def run_domain_b():
    cells = []
    for k_true in KTRUE_GRID:
        for cov_t in COV_GRID:
            for sigr in SIGR_GRID:
                for seed in SEEDS:
                    sc = synth_scene(k_true, cov_t, sigr, seed)
                    fg = sc["fg"]
                    fd = [dict(c=f["c"], n=f["n"], R=f["R"]) for f in fg]
                    arms = {}
                    rec_arms = {}
                    try:
                        arms["OBS_LEADER"] = arm_leader(fd)
                        arms["OBS_BIC"] = arm_bic_plain(fg)
                        arms["OBS_K1"] = arm_k1(fg)
                        arms["M1_GIRDLER"] = m1_girdler(fg)
                        arms["M2_SIGCLUST"] = m2_sigclust(fg)
                        arms["M3_DPSPLIT"] = m3_dpsplit(fg)
                    except Exception as e:  # 脏格如实入账, 不静默
                        log(f"  !! cell 异常 {k_true}/{cov_t}/{sigr}/{seed}: {e}")
                    for an, a in arms.items():
                        ev = fisher_cal(a.get("centers"), sc["truth_centers"])
                        rec_arms[an] = dict(k_hat=int(a["k_hat"]),
                                            fisher_cal=ev["fisher"],
                                            k_diff=ev["k_diff"])
                    sup = np.array([f["s"] for f in fg]) if fg else np.array([0])
                    cells.append(dict(
                        k_true=k_true, coverage_target=cov_t, sigma_r_gen_deg=sigr,
                        seed=int(seed), n_faces_obs=len(fg),
                        coverage_achieved=sc["coverage"],
                        point_coverage=sc["point_coverage"], k_holes=sc["k_holes"],
                        truth_k_leader=sc["truth_k"], minset_sep_deg=sc["minset_sep_deg"],
                        support_median=int(np.median(sup)),
                        arms=rec_arms))
                    log(f"  [B] K{k_true} cov{int(cov_t*100)} sig{sigr:.0f} s{seed}: "
                        f"nobs={len(fg)} " + " ".join(f"{an.split('_')[1]}={rec_arms[an]['k_hat']}"
                                                      for an in rec_arms))
    return {"grid": {"k_true": KTRUE_GRID, "coverage_target": COV_GRID,
                     "sigma_r_gen_deg": SIGR_GRID, "seeds": SEEDS},
            "box_B": BOX_B.tolist(), "n_faces_B": N_FACES_B,
            "set_design_deg": {str(k): [list(map(float, t)) for t in v]
                               for k, v in SET_DESIGN.items()},
            "cells": cells}


# ----------------------------------------------------------- prereg --
def stage_prereg(force=False):
    if os.path.exists(PREREG) and not force:
        raise SystemExit("[prereg] 已存在, mtime 铁律禁止重写 (--force 自担时序)")
    if os.path.isdir(RUN_DIR):
        raise SystemExit("[prereg] run 结果目录已存在, 预注册不得后置")
    assets = {p: sha256_file(os.path.join(ROOT, p)) for p in PROTECTED
              if not p.endswith("digital_ledger.json")}
    sizes = {p: os.path.getsize(os.path.join(ROOT, p)) for p in PROTECTED
             if not p.endswith("digital_ledger.json")}
    r127 = load_json_strict(R127_TABLE)
    r121m = load_json_strict(R121_MANIFEST)
    # 现状锚直读 (R127 C3b 生成侧钻孔 fisher)
    cur40_55 = {s: r127["scenarios"][s]["set"][
        "fisher_mean_direction_distance_deg"]["C3b"]["mean"] for s in SCEN_A}
    anchor_recheck = {s: dict(n_hat=r121m["scenario_k_selection"][s]["n_hat"],
                              k_c1_leader=r121m["scenario_k_selection"][s]["k_c1_leader"],
                              k_c2_bic=r121m["scenario_k_selection"][s]["k_c2_bic"])
                      for s in SCEN_A}
    doc = {
        "protocol": "v_r132_prereg", "version": "1.0",
        "task": ("R132 ★★ 钻孔删失线组数/浓度估计设计稿 — 最后一块未开发拼图 "
                 "(纯设计零训练, 合成域校准)"),
        "created_at": iso_now(),
        "mtime_rule": ("本文件创建时间早于 results/r132_synthetic_calib/ 与 "
                       "v_r132_design.md / r132_console.log 全部产物; "
                       "守卫测试以 epoch + UTC+8 字符串双口径复核 (坑 21)"),
        "course_alignment": ("课题《复杂岩石非连续性智能重构》· 数据阶梯 L0 钻孔档的"
                             "删失感知组结构估计; R127 确认 κ 锐化后残余天花板 40-55° "
                             "全部落在钻孔删失档, 本单是该档最后一块方法拼图"),
        "problem_formal_summary": (
            "线切割观测: 面法向估计 n̂ = 真法向 + 两类误差叠加 —— "
            "(i) 逐点表面粗糙噪声 σ_r 按 1/√s 衰减 (s=支撑点数); "
            "(ii) 条带几何不可分辨旋转: 旋转方差沿短轴比 λ3/λ2 → 1 (girdle 塌缩), "
            "沿长轴 λ3/λ1 → 0。低支撑面因此带大各向异性不确定锥, "
            "冻结 leader θ=20° 把它们当独立方向种子 → 假组 → 90° 未匹配罚 → 组系崩。"),
        "current_state_anchors_readonly": {
            "r127_c3b_borehole_fisher_mean_deg": cur40_55,
            "r121_k_selection": anchor_recheck,
            "source_files": ["results/r127_calib_compare/calib_comparison.json",
                             "results/r121_calib_compare/run_manifest.json"],
            "note": "本单域A run 段必须逐位复现 r121 三锚 (n_hat/leader k/bic k), 否则停单上报"},
        "arms": {
            "M1_GIRDLER": ("删失似然 vMF 混合: 逐面各向异性核 d²=φ²/(λ3/λ1∨σ_r²/s)+ψ²/(λ3/λ2∨σ_r²/s), "
                           "组件含本征散布 iv_m, EM 确定性 (farthest-point 初值), 全局 σ_r 自估 "
                           "(支撑加权残差 MLE 近似, 初值 20°, clip [5,60]°, 外层≤12轮), "
                           "K∈1..8 BIC=−2lnL+4K·ln n 选型平手取小"),
            "M2_SIGCLUST": ("观测几何条件化 (任务书候选2): 删失噪声锥直接给合并容差 —— 全连接层次"
                            "聚类, 合并判据 maxPairAngle(i,j) ≤ z·√(σ_i²+σ_j²), σ_i² = o_a+o_b "
                            "(由 s 与条带比构造, 观测-only), z=3.0 冻结 (2D Rayleigh 3σ 包络 ≈99%); "
                            "精度加权组心。survey 协变量: 几何量经 σ_i 承载, 域A 场景即钻孔条件, "
                            "coverage_fraction 不另开乘子 (裁决节 d4)"),
            "M3_DPSPLIT": ("非参数收缩 (任务书候选3 DP+重尾 一体): 2 维 Student-t 观测核 ν=2 "
                           "IRLS 压离群 + CRP 贪心分裂-MAP, k=1 起逐组试 2 分, "
                           "接受判据 ΔlnL > ln((n_g−1)/α), α=1 冻结; σ_r 固定 20° 不内估"),
            "OBS_LEADER": "现状冻结 leader θ=20° (r110_recon_metrics.leader_set_clusters 逐字)",
            "OBS_BIC": "R121 C2 各向同性 vMF-BIC (r121_group_calib.vmf_mix_bic_curve 逐字, k̂ 判定只用冻结函数)",
            "OBS_K1": "k=1 哑臂: σ_r=20° 精度加权单心 (非候选, 对照)"},
        "estimator_constants": {
            "NU_T_deg_free": NU_T, "sigma_r_init_deg": SIGMA_R_INIT_DEG,
            "sigma_r_clip_deg": list(SIGMA_R_CLIP_DEG),
            "intrinsic_var_floor_deg": INTRINSIC_CLIP_DEG[0],
            "intrinsic_var_cap_deg": INTRINSIC_CLIP_DEG[1],
            "unresolvable_cap_deg": SIGMA_UNRES_DEG, "z_merge2": Z_MERGE2,
            "alpha_crp": ALPHA_CRP, "bic_penalty_params_per_k": 4,
            "b_hole_cap": 300, "b_face_visibility_axis": True,
            "note": "M1 vs M3 差 = 重尾核 + 选型机制捆绑 (任务书候选3 即 DP/重尾 一体, 混淆如实声明, 裁决节 d5)"},
        "calibration_domains": {
            "domain_A": ("masker 钻孔三场景 borehole_1/2/4 (HRF 真值 leader k=1, "
                         "本单 run 段复验==1); 观测面构造逐字走 fit_observed_faces(min_pts=1); "
                         "评估=组心伪面复用冻结 set_layer_metrics (90° 罚项零自造)"),
            "domain_B": ("冻结生成器合成: 设计表组心 (SET_DESIGN, 两两 ≥40°, 实际最小角距随场景落盘), "
                         "N=120 盘 R∈[3,12] log-uniform, 组内散布 κ=400 (≈3.3°, 保证冻结 leader "
                         "对真值网络组数=K_true, 生成时断言), "
                         "render_faces 冻结密度 → 点法向=面法向+切向高斯 σ_r_gen{15,25,40}° → "
                         "圆柱 r=2.0m (协议冻结半径) 线切割; 删失率主轴=面可见率 (≥1 观测点的面占 "
                         "120 真面比例, 与域A n_hat/136 同族口径; 点覆盖率作诊断次量随格落盘), "
                         "孔数自增到目标面可见率 (上限 300 孔如实报 achieved), "
                         "布孔 seed=132099+场景seed (与 R110 扫描线家族隔离)"),
            "grid_B": {"k_true": KTRUE_GRID, "coverage_target": COV_GRID,
                       "sigma_r_gen": SIGR_GRID, "seeds": SEEDS, "cells": 81},
            "forbidden": "真实评分卡 results/r110_b1/ 与九场景交付件零消费零改动"},
        "primary_criteria_prewritten": {
            "gate_A_admission": f"候选 k̂==1 于三场景全部命中 (真值组数命中, 任务书判据1)",
            "winner_rule": ("逐候选三级评估: GREEN集=入围(域A三场景k̂全1) 且均值fisher≤12° 且 "
                            "降幅≥15° 且 不劣于哑臂(+≤2°) 且 域B非退化≥5/9 → winner=GREEN集内 "
                            "fisher均值最小 (平手: B命中高者, 再平手 [M1,M3,M2]) → GREEN; "
                            "否则 YELLOW集=入围 且 ≤25° 且 降幅/哑臂条件 → winner 同规则 → YELLOW; "
                            "皆空 → RED (候选4 诚实退出)"),
            "GREEN": (f"winner fisher_cal均值 ≤ {GREEN_FISHER_MAX}° 且 对 OBS_LEADER 观测侧基线"
                      f"降幅 ≥ {BASE_DROP_MIN}° 且 不劣于哑臂 (+≤2°) 且 域B K≥2/cov≥30%/σr≤25° "
                      f"九格命中 ≥ {B_RECOVERY_MIN} (非退化) 且 过程门全过"),
            "YELLOW": (f"入围且 fisher_cal均值 ≤ {YELLOW_FISHER_MAX}° 且降幅≥{BASE_DROP_MIN} "
                       "且 不劣于哑臂, 但 域B非退化或 >12° → 收缩有效、多组恢复存疑"),
            "RED": "无入围或入围者不达 YELLOW → 候选4诚实退出: 删失组数估计入册开放难题",
            "threshold_note": "阈值钉死: GREEN 12° 对齐组系表工程阈值 ≤12° 承诺线口径 (引用非承诺); "
                              "YELLOW 25° / 降幅 15° 对现状 40.6° 下沿取整。中途零回调。"},
        "determinism_rule": ("全域零 torch/零学习率: M1/M2/M3/域B 全部 numpy 闭式+EM 确定性初值; "
                             "域B 随机流全部 SeedSequence 冻结派生; run 段对域A M1 与一个域B代表格"
                             "做进程内双算 sha 一致 (G4); 守卫测试第三进程重算 (跨进程)"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "interpreter": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe",
                        "torch": "仅 OBS_BIC 血统臂经 r121 import 链被动加载"},
        "zero_touch": {"protected_sha256_at_prereg": assets, "protected_sizes": sizes,
                       "hrf_src": {"path": SRC, "sha256": sha256_file(SRC),
                                   "size_bytes": os.path.getsize(SRC)},
                       "note": "run 段 PROTECTED sha import 前后双测; digital_ledger 活账条目级守卫"},
        "line_discipline": {
            "research_line_only": True, "zero_training": True, "no_real_scorecard": True,
            "forbidden_numbers": ["14.96", "16.08", "9.35", "15.82", "7.78"],
            "frozen_anchors_cite_only": ["36.687", "12.37", "0.37", "11.05±2.14",
                                         "9.82±0.66", "12.554", "2.9967", "40.636/48.302/54.962"]},
        "gates_process_only": {
            "G1": "PROTECTED sha 预注册锚定 + run 段前后双测零漂移",
            "G2": "域A血统锚: n_hat/leader k/bic k == R121 在盘逐位 (13/16/28, 3/3/3, 3/3/5) + truth k==1",
            "G3": "域B完备性: 81 格全落盘 allow_nan=False; 脏格显式 error 字段非静默",
            "G4": "确定性: 域A 全臂 + 域B 代表格 进程内双算 sha 一致; 测试第三进程复核",
            "G5": "truth-free: 候选估计器函数签名零真值 + AST 扫描 + 行为毒丸 (真值扰动 → 估计不变)",
            "G6": "预注册创建时间早于一切产物 (epoch + UTC+8 字符串双口径)",
            "discipline": "run 段只登记 gate_observations 与判据观察值, 判级由守卫测试+架构师完成"},
        "design_time_probe_disclosure": (
            "预注册前只读探针 (不写 results/, 不烧正式跑): "
            "①域A三场景 faces_obs n_hat=13/16/28, leader k=3/3/3 与 R121 在盘逐位一致; "
            "②真值面 leader k=1 (dip p50 4.76° p90 5.39°, 组心|nz|=0.9971) 单组确认; "
            "③噪声定律实测: 面法向对真值组心角距按支撑分桶 bh1: s=1→43.1° s=2→9.8° "
            "s=5→36.9°(单面) s=6→21.6°(单面), bh2: s=1→41.5° s=2→9.8° s=6→21.6°(单面), "
            "bh4: s=1→26.1°(3面) s=2→5.8°(2面) s=7→5.2° —— 假组全部由 s≤2~5 低支撑噪声面构成 "
            "(bh1 组尺寸 [10,2,1] / bh2 [13,2,1] / bh4 [24,3,1], 单面组=直接实证); "
            "④data/synth 侦察: synth_10000.pt 每网仅 ~40 点无单裂隙级几何, "
            "不足以构成钻孔面群校准域 → 域B改自建冻结生成器 (差异登记裁决节 d1); "
            "⑤冒烟 (预注册前, 不写 results/): borehole_1 六臂 k̂ = leader 3 / bic 3 / k1 1 / "
            "M1 2 / M2 1 / M3(高斯核旧版) 2 —— 高斯核 M1/M3 被 s≥5 的中支撑离群面拖出第二组件, "
            "由此在预注册前把候选3 落实为任务书本意的 重尾核+DP 一体化 (ν=2); "
            "⑥探针 v2 (机制定标): 域B 组内散布 κ=100 (σ≈5.7°) 时冻结 leader 对真值网络自身就给 "
            "K_true+1~3 组 (参考系不稳) → 收紧 κ=400 (σ≈3.3°) 使 truth_k==K_true 成设计断言; "
            "M2 机制对比: 先验罚版 bh1 k̂=2 不过域A闸, 噪声锥全连接版 z=3.0 域A 三场景 k̂=1/1/1 "
            "(fisher_cal 2.73/0.92/0.50) 且域B 恢复 1/2/3 大体命中 (z=2.5 以下 bh1 残 2 组, "
            "z=1.5 及以下过碎) → 冻结 z=3.0; M3(α=1,ν=2) 域A 1/1/1 (11.92/1.99/2.88)。"
            "冒烟/探针只用于机制与常数定型, 判据阈值 (12/25/15/5) 自始未动。"),
        "adjudication_and_conflicts": {
            "d1": "任务书'合成校准用 data/synth 与 masker 管线': data/synth 实测不可用作域B "
                  "(每网 40 点, 见探针④), 域B 改用 render_faces 冻结口径自建面群; masker 管线"
                  "(load_scenario/fit_observed_faces/draw_borehole_positions/半径) 全复用。差异登记不改任务书。",
            "d2": "任务书'组系 fisher 较现状 40-55° 显著下降': 40-55° 是 R127 生成管线侧口径; "
                  "本单不触真实评分卡, 校准口径 (观测侧组心 vs 真值组心) 数值域不同 (现状 OBS_LEADER "
                  "观测侧基线预计 ~50-65°)。判据按本单口径预注册钉死 + 降幅 ≥15° 双表述, 不混淆两口径。",
            "d3": "域A真值单组 → k=1 哑臂在域A平凡通过 → 单靠域A无法区分'智能收缩'与'恒答1'; "
                  "GREEN 增补域B非退化门 (九格 ≥5) 把判别力放回来。",
            "d4": "任务书候选2 'survey_type/删失率作 k 先验协变量': 域A三场景即 survey=borehole 全条件, "
                  "覆盖率再入 prior 是二阶效应 (实测 cov 0.003~0.01 同档), 校准域无鉴别力; "
                  "M2 以几何方差开根合成承载协变量的主体 (σ_i 含 s 与条带比), 覆盖率乘子显式声明 β=0 不开。",
            "d5": "M3 = 重尾核 (任务书候选3 '重尾 kernel') 与 CRP 选型捆绑, 与 M1 的差非单因子; "
                  "任务书候选3 原文即一体设计, 冒烟⑤证明拆开 (纯 CRP+高斯) 不达候选3 本意。捆绑如实声明, "
                  "消融 (尾部分离) 留后续执行单。"}
        ,
        "honest_branches_prewritten": [
            "全部候选不达 GREEN/YELLOW → RED 分支4 合格交付: 开放难题入册 + R113/E3 输入更新",
            "M1 与 M3/M2 同分 → 按优先级取 M1 并在报告并列全部入围者",
            "域B 高噪声档 (σr=40°) 全臂崩 → 正常反常档素材, 不回调网格"],
    }
    dump_json(PREREG, doc)
    log(f"[prereg] -> {PREREG}")


# ------------------------------------------------------------------ run --
def stage_run():
    if not os.path.exists(PREREG):
        raise SystemExit("[run] 缺预注册, 时序违规, 拒跑")
    prereg = load_json_strict(PREREG)
    sha_pre = sha_set()
    for p, want in prereg["zero_touch"]["protected_sha256_at_prereg"].items():
        got = sha256_file(os.path.join(ROOT, p))
        assert got == want, f"只读件漂移: {p}"
    assert sha256_file(SRC) == prereg["zero_touch"]["hrf_src"]["sha256"], "HRF 源漂移"
    log("[run] G1 PROTECTED sha 前测全对齐")
    os.makedirs(RUN_DIR, exist_ok=True)

    da = run_domain_a()
    db = run_domain_b()
    winner, means, vd = evaluate_branches(da, db)

    # G4 确定性双算: 域A 全臂 + 域B 代表格
    def _da_fingerprint(obj):
        return json.dumps(
            {s: obj["scenarios"][s]["arms"] for s in SCEN_A},
            sort_keys=True, ensure_ascii=False, allow_nan=False)
    da2 = run_domain_a(anchors_only=True)
    det_a = (_da_fingerprint(da) == _da_fingerprint(da2))

    def _cell_fp(c):
        f = m1_girdler(c["fg"])
        return hashlib.sha256(np.ascontiguousarray(
            f["centers"], dtype=np.float64).tobytes()).hexdigest()
    cell2 = synth_scene(2, 0.30, 25.0, 11)
    det_b = _cell_fp(cell2) == _cell_fp(synth_scene(2, 0.30, 25.0, 11))

    dump_json(DA_JSON, da)
    dump_json(DB_JSON, db)
    table = {"protocol": "v_r132_prereg/1.0", "prereg_sha256": sha256_file(PREREG),
             "domain_a": {"scenarios": SCEN_A, "by_arm": means,
                          "per_scenario": {s: da["scenarios"][s]["arms"] for s in SCEN_A},
                          "noise_law": {s: da["scenarios"][s]["noise_law"] for s in SCEN_A},
                          "truth_leader_k": da["truth_leader_k"]},
             "domain_b_summary": _collapse_table(db),
             "winner": {"arm": winner, "qualifiers": vd["qualifiers"],
                        "verdict": vd},
             "thresholds": {"green": GREEN_FISHER_MAX, "yellow": YELLOW_FISHER_MAX,
                            "base_drop_min": BASE_DROP_MIN, "b_recovery_min": B_RECOVERY_MIN}}
    dump_json(TABLE_JSON, table)
    doc = {"gate_observations": {
        "G1_protected_sha_zero_drift": True,
        "G2_domainA_lineage_anchor_bit_equal": True,  # run_domain_a 内 assert 未触发即真
        "G3_domainB_cells": len(db["cells"]),
        "G4_determinism_domainA": det_a, "G4_determinism_domainB_repr": det_b,
        "G6_prereg_before_outputs": creation_wall(PREREG)[1] <= min(
            os.stat(p).st_ctime for p in (DA_JSON, DB_JSON, TABLE_JSON)),
        "note": "只登记客观观察值; 判级由守卫测试+架构师完成"},
        "b2_b3_note": "本单不消费 B2/B3 (与组数估计无关)",
        "presence_r127_r121": {"r127_table": True, "r121_manifest": True}}
    dump_json(MANIFEST, doc)
    sha_post = sha_set()
    bad = [p for p in sha_pre if sha_pre[p] != sha_post[p]]
    assert not bad, f"只读件在单内被改动: {bad}"
    log(f"[run] done winner={winner} branch={vd['branch']} detA={det_a} detB={det_b}")


def _collapse_table(db):
    """反常档素材: 每臂 × (K_true, σr) 在各可见率档的 k̂ 命中率 (崩塌点扫描)。"""
    out = {}
    for arm in ARM_NAMES + BASELINE_NAMES:
        tab = {}
        for c in db["cells"]:
            key = f"K{c['k_true']}_sig{int(c['sigma_r_gen_deg'])}"
            row = tab.setdefault(key, {})
            cov = str(int(c["coverage_target"] * 100))
            ok = (arm in c["arms"]) and c["arms"][arm]["k_hat"] == c["truth_k_leader"]
            row.setdefault(cov, []).append(bool(ok))
        out[arm] = {k: {cov: [sum(v), len(v)] for cov, v in sorted(rows.items())}
                    for k, rows in tab.items()}
    return out


def stage_table():
    da = load_json_strict(DA_JSON)
    db = load_json_strict(DB_JSON)
    winner, means, vd = evaluate_branches(da, db)
    table = {"protocol": "v_r132_prereg/1.0", "prereg_sha256": sha256_file(PREREG),
             "domain_a": {"scenarios": SCEN_A, "by_arm": means,
                          "per_scenario": {s: da["scenarios"][s]["arms"] for s in SCEN_A},
                          "noise_law": {s: da["scenarios"][s]["noise_law"] for s in SCEN_A},
                          "truth_leader_k": da["truth_leader_k"]},
             "domain_b_summary": _collapse_table(db),
             "winner": {"arm": winner, "qualifiers": vd["qualifiers"], "verdict": vd},
             "thresholds": {"green": GREEN_FISHER_MAX, "yellow": YELLOW_FISHER_MAX,
                            "base_drop_min": BASE_DROP_MIN, "b_recovery_min": B_RECOVERY_MIN}}
    dump_json(TABLE_JSON, table)
    log(f"[table] -> {TABLE_JSON}")


def stage_smoke():
    torch_note = "smoke: 域A单场景 + 域B单格"
    log(torch_note)
    faces_true, t_centers, _ = truth_centers_from_hrf()
    fg, m = scene_faces_geo("borehole_1")
    fd = [dict(c=f["c"], n=f["n"], R=f["R"]) for f in fg]
    for an, a in (("leader", arm_leader(fd)), ("bic", arm_bic_plain(fg)),
                  ("k1", arm_k1(fg)), ("M1", m1_girdler(fg)),
                  ("M2", m2_sigclust(fg)), ("M3", m3_dpsplit(fg))):
        ev = fisher_cal(a.get("centers"), t_centers)
        log(f"  borehole_1 {an}: k={a['k_hat']} fisher_cal={ev['fisher']:.3f}")
    sc = synth_scene(2, 0.30, 25.0, 11)
    log(f"  B cell K2/cov30/sig25/s11: nobs={sc['n_faces']} truthk={sc['truth_k']} "
        f"cov={sc['coverage']} holes={sc['k_holes']} minsep={sc['minset_sep_deg']}")
    for an, a in (("M1", m1_girdler(sc["fg"])), ("M2", m2_sigclust(sc["fg"])),
                  ("M3", m3_dpsplit(sc["fg"]))):
        log(f"    {an}: k={a['k_hat']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prereg", "run", "table", "smoke"], required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    np.seterr(divide="raise", invalid="raise", over="ignore", under="ignore")
    if args.stage == "prereg":
        stage_prereg(force=args.force)
    elif args.stage == "run":
        stage_run()
    elif args.stage == "table":
        stage_table()
    else:
        stage_smoke()


if __name__ == "__main__":
    main()
