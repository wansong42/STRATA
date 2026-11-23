# -*- coding: utf-8 -*-
"""组系轴向区间构造 + 覆盖率校准 (R185 T3)。

为什么区间是**椭圆**而不是锥
----------------------------
产状的不确定度天然是切空间上的二维分布。把它压成一个「锥半角」会强制各向同性:

    - polar 型组系 (极点图上是一团): 两个方向散布接近 → 锥是合理近似;
    - girdle 型 (极点图上是大圆弧): 沿大圆方向几乎不可辨识, 垂直大圆方向很稳
      → 不确定度是**极度拉长的椭圆**, 任何单一锥半角要么在一个方向过度保守、
      要么在另一个方向严重低估覆盖率。

本模块因此以 **Wald 椭圆 (长半轴/短半轴/方位)** 为主口径, 锥半角作为
「保守外接圆」并列输出, 两条覆盖率都报。这是本项目 L4 既有
`bootstrap_confidence_cone` (只有锥) 的直接补强。

三种区间
--------
1. `wald`         观测 Fisher 信息 → 4 参数 (2 个轴向角 + 2 个浓度) Hessian →
                  取轴向块边际协方差 → 椭圆/锥。用到了「拟合了参数模型」这一点,
                  是 Bingham 相对纯 bootstrap 的**主要收益来源**。
2. `boot_param`   参数化 bootstrap: 从**拟合出的** Bingham 重采样, 并且
                  **重放 Terzaghi 观测偏置** (接受概率 ∝ |n·a|), 再走同一条估计管线。
                  重放观测模型是关键 —— 不重放就会低估方差 (Terzaghi 加权是
                  自归一化重要性采样, 方差放大来自权重长尾, 不是来自母体)。
3. `boot_nonparam`非参 bootstrap: 重抽观测条目 (项目 L4 现役口径, 作基线对照)。
                  已知弱点 (T46): 它度量的是「样本内精度」, 拿来当预测区间会
                  系统性欠覆盖 —— 本单把这条用合成真值量化出来。

信息量缩放 (诚实处理加权)
------------------------
负对数似然 ℓ = -W·nll₁, 故 Cov = inv(H₁)/W。加权时 W 该取 N 还是 ESS?
自归一化重要性采样的信息量按 ESS 折损, 因此本模块统一取 **W = ESS**
(等权时 ESS = N, 自动退化为经典结果)。这个选择对覆盖率有实质影响,
写在 `meta.information_scale` 里供审计。

校准与评估必须分半 (防泄漏)
--------------------------
标称 95% 的区间实测覆盖率往往不是 95%。要「校准」就必须用**一半试验**拟合
缩放因子, 在**另一半**上报覆盖率 —— 同集校准再同集评估是自证, 本项目禁止。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._sphere import HAS_TORCH, log_bingham_norm, hessian_of_scalar
from .bingham import Bingham

if HAS_TORCH:
    import torch


__all__ = [
    "NOMINAL_LEVELS",
    "chi2_2",
    "wald_uncertainty",
    "wald_cover",
    "bootstrap_cone",
    "fit_calibration_scale",
    "sample_observed",
    "IntervalResult",
]


NOMINAL_LEVELS = (0.60, 0.80, 0.95)


def chi2_2(level: float) -> float:
    """2 自由度卡方分位数 (闭式): χ²₂(1-α) = -2·ln(α)。"""
    a = 1.0 - float(level)
    if not (0.0 < a < 1.0):
        raise ValueError(f"chi2_2: level 必须在 (0,1), got {level}")
    return -2.0 * np.log(a)


def _skew_basis(p: int = 3):
    """旋转生成元: E12 把轴 1 转向轴 2, E13 把轴 1 转向轴 3。"""
    E12 = np.zeros((p, p))
    E12[0, 1] = -1.0
    E12[1, 0] = 1.0
    E13 = np.zeros((p, p))
    E13[0, 2] = -1.0
    E13[2, 0] = 1.0
    return E12, E13


@dataclass
class IntervalResult:
    method: str
    level: float
    semi_major_deg: Optional[float] = None
    semi_minor_deg: Optional[float] = None
    orient_deg: Optional[float] = None      # 长轴在 (m₂,m₃) 切基内的方位
    cone_deg: Optional[float] = None        # 保守外接锥半角
    covered_ellipse: Optional[bool] = None
    covered_cone: Optional[bool] = None
    meta: dict = None


def sample_observed(truth: Bingham, n: int, axis, terzaghi: bool, rng,
                    max_batches: int = 400):
    """按「真分布 → Terzaghi 观测偏置」生成一条观测样本 (条件于观测条数 n)。

    观测偏置: 一条法向为 n 的裂隙被钻孔切到的概率 ∝ |n·a| (≤ 1), 故可直接用
    拒绝采样, 接受概率就是 |n·a|。terzaghi=False 时直接返回真分布样本。
    """
    if not terzaghi:
        return truth.sample(n, rng)
    a = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    out, got = [], 0
    for _ in range(max_batches):
        m = max(int((n - got) * 3) + 64, 256)
        cand = truth.sample(m, rng)
        keep = cand[rng.random(m) < np.abs(cand @ a)]
        out.append(keep)
        got += int(keep.shape[0])
        if got >= n:
            break
    if got < n:
        raise RuntimeError(f"sample_observed: 只采到 {got}/{n}, 拒绝静默补样")
    return np.concatenate(out, axis=0)[:n]


def _generators_for(idx: int, p: int = 3):
    """绕「除 idx 外的两个轴」旋转 idx 轴的两个生成元。

    ⚠️ 被估轴必须可辨识, 否则 Hessian 奇异:
      - polar/triaxial 估 m₁ (λ 唯一最小) → 绕 (m₂,m₃) 转 m₁ ✓ 可辨识
      - girdle 估 m₃ (λ 唯一最大, 环带极) → 必须绕 (m₁,m₂) 转 m₃;
        若仍按 m₁ 参数化, 因 girdle 的 λ₁=λ₂=0 导致 m₁ 在环带内不可辨识,
        Hessian 直接奇异 (实测 torch.linalg.inv 报 diagonal element 4 is zero)。
    """
    others = [i for i in range(p) if i != idx]
    j, k = others[0], others[1]
    G1 = np.zeros((p, p))
    G1[j, idx] = 1.0
    G1[idx, j] = -1.0
    G2 = np.zeros((p, p))
    G2[k, idx] = 1.0
    G2[idx, k] = -1.0
    return G1, G2, others


def wald_uncertainty(S, axes_hat, lmbda_hat, n_eff,
                     levels=NOMINAL_LEVELS, idx: int = 0):
    """观测信息 → 被估轴的切空间 2×2 协方差 → 各标称水平的椭圆/锥。**支持批量**。

    S         : (B,3,3) 或 (3,3) 加权二阶矩
    axes_hat  : (B,3,3) 或 (3,3) Bingham 主轴 (列)
    lmbda_hat : (B,3) 或 (3,) 升序浓度
    n_eff     : (B,) 或 float 信息量 (ESS; 等权时 = N)
    idx       : 被估轴下标 (0 = 众数轴, p-1 = 环带极)

    返回 {"Sigma2": (B,2,2), "levels": {lv: {..arrays..}}, "singular": (B,), "cond": (B,)}
    """
    if not HAS_TORCH:
        raise RuntimeError("wald_uncertainty 需要 torch (要 autograd Hessian)")
    St = torch.as_tensor(np.asarray(S, dtype=np.float64))
    Mh = torch.as_tensor(np.asarray(axes_hat, dtype=np.float64))
    Lh = torch.as_tensor(np.asarray(lmbda_hat, dtype=np.float64))
    if St.ndim == 2:
        St, Mh, Lh = St.unsqueeze(0), Mh.unsqueeze(0), Lh.unsqueeze(0)
    Bsz = St.shape[0]
    G1, G2, others = _generators_for(idx, 3)
    G1t = torch.as_tensor(G1, dtype=torch.float64)
    G2t = torch.as_tensor(G2, dtype=torch.float64)

    def _nll(z):
        # z: (B,4) = (θ₁, θ₂, λ₂, λ₃)
        # ⚠️ 形状铁律: 必须得到 (B,1,1) 再乘 (3,3) → (B,3,3)。
        # 曾写成 z[:, 0:1, None, None] 得到 (B,1,3,3), 与 (B,3,3) 的 Mh 做 matmul
        # 时 torch 广播成 (B,B,3,3), nll 返回 (B,B) 而不是 (B,) —— B=1 时碰巧
        # 正确、B>1 时 Hessian 全错 (实测偏 15 倍, 覆盖率从 95% 掉到 4%)。
        # 这类 bug 只有"批量==单条"一致性测试能抓住, 见 tests/test_v_r185_.py。
        G = z[:, 0].reshape(-1, 1, 1) * G1t + z[:, 1].reshape(-1, 1, 1) * G2t
        assert G.shape == (z.shape[0], 3, 3), f"G shape {tuple(G.shape)} != (B,3,3)"
        R = torch.linalg.matrix_exp(G)                    # (B,3,3)
        M = Mh @ R
        lam = torch.cat([torch.zeros(z.shape[0], 1, dtype=z.dtype), z[:, 2:4]], dim=-1)
        MtSM = M.transpose(-1, -2) @ St @ M
        out = (lam * MtSM.diagonal(dim1=-2, dim2=-1)).sum(-1) + log_bingham_norm(lam)
        assert out.shape == (z.shape[0],), f"nll shape {tuple(out.shape)} != (B,)"
        return out

    def _grad(z):
        return torch.autograd.grad(_nll(z).sum(), z, create_graph=True)[0]

    z0 = torch.cat([torch.zeros(Bsz, 2, dtype=torch.float64), Lh[:, 1:3]], dim=-1)
    H = hessian_of_scalar(_grad, z0)                      # (B,4,4)
    ev_H = torch.linalg.eigvalsh(H)                       # (B,4) 升序
    singular = ev_H[:, 0] <= 1e-10 * torch.clamp(ev_H[:, -1], min=1e-300)
    Hr = H + torch.eye(4, dtype=H.dtype).unsqueeze(0) * (
        1e-8 * torch.clamp(ev_H[:, -1:], min=1e-12).unsqueeze(-1))
    Cov = torch.inverse(Hr)
    ne = torch.as_tensor(np.atleast_1d(np.asarray(n_eff, dtype=np.float64)),
                         dtype=torch.float64).reshape(-1)
    if ne.numel() == 1:
        ne = ne.expand(Bsz)
    Cov = Cov / ne.clamp_min(1e-12).reshape(-1, 1, 1)
    Sig2 = Cov[:, :2, :2].detach().cpu().numpy()

    ev, evec = np.linalg.eigh(Sig2)                       # (B,2) 升序
    ev = ev[:, ::-1]
    evec = evec[:, :, ::-1]
    orient = np.degrees(np.arctan2(evec[:, 1, 0], evec[:, 0, 0])) % 180.0

    levels_out = {}
    for lv in levels:
        c2 = chi2_2(lv)
        semi = np.degrees(np.sqrt(np.clip(ev, 0.0, None) * c2))
        levels_out[float(lv)] = {
            "chi2_2": float(c2),
            "semi_major_deg": semi[:, 0].tolist(),
            "semi_minor_deg": semi[:, 1].tolist(),
            "orient_deg": orient.tolist(),
            "cone_deg": semi[:, 0].tolist(),              # 外接圆 = 长半轴
        }
    return {"Sigma2": Sig2, "levels": levels_out,
            "singular": singular.detach().cpu().numpy().tolist(),
            "cond": (ev[:, 0] / np.clip(ev[:, 1], 1e-300, None)).tolist(),
            "tangent_basis": others}


def wald_cover(Sig2, axes_hat, m_true, level: float, idx: int = 0) -> Tuple[bool, bool]:
    """用 Wald 椭圆 / 外接锥判定真轴是否被覆盖。切基 = 除 idx 外的两个主轴。"""
    m_true = np.asarray(m_true, dtype=np.float64).reshape(3)
    A = np.asarray(axes_hat, dtype=np.float64)
    others = [i for i in range(3) if i != idx]
    m1 = A[:, idx]
    b1 = A[:, others[0]]
    b2 = A[:, others[1]]
    if float(m_true @ m1) < 0:                 # 轴数据: 符号无意义
        m_true = -m_true
    v = np.array([float(m_true @ b1), float(m_true @ b2)])
    ang = float(np.degrees(np.arccos(np.clip(abs(float(m_true @ m1)), 0, 1))))
    c2 = chi2_2(level)
    q = float(v @ np.linalg.solve(Sig2, v))
    ev = np.linalg.eigvalsh(Sig2)
    cone = float(np.degrees(np.sqrt(max(ev.max(), 0.0) * c2)))
    return bool(np.isfinite(q) and q <= c2), bool(ang <= cone)


def bootstrap_angles(fit0: Bingham, X, W, axis, terzaghi: bool, B: int,
                     mode: str, rng, idx: int = 0) -> np.ndarray:
    """返回 (B,) 重拟合主轴相对点估计主轴的角距 (度) —— 一次抽样本, 多档分位。

    idx: 估目标轴的下标 (0 = 最小 λ 轴 = 众数轴; p-1 = 最大 λ 轴 = 环带极)。
    """
    from .bingham import fit_bingham_batch
    if not HAS_TORCH:
        raise RuntimeError("bootstrap_angles 需要 torch")

    N = int(X.shape[0])
    if mode == "nonparam":
        # ⚠️ 变量名不要叫 idx —— 会遮蔽形参 idx (估目标轴下标), 曾直接导致
        # IndexError。重抽样下标一律用 bidx。
        bidx = rng.integers(0, N, size=(B, N))
        Xb = X[bidx]                                 # (B,N,3)
        Wb = (W[bidx] if W is not None else np.ones((B, N)))
    elif mode == "param":
        # 重采样; **只有当估计管线本身带 Terzaghi 校正时**才重放观测偏置。
        # ⚠️ 曾无条件重放: terzaghi=False 时 bootstrap 世界带偏置而估计世界不带,
        # 拟合轴被井轴拉走, 轴离散度虚增 4 倍 (median 1.1° → 4.5°), 覆盖率全错。
        # 参数化 bootstrap 必须重放「与真实观测完全相同」的采样设计, 否则它度量的
        # 不是估计量的抽样分布。
        a = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-12)
        Xb = np.empty((B, N, 3), dtype=np.float64)
        Wb = np.ones((B, N), dtype=np.float64)
        for b in range(B):
            if terzaghi:
                got = np.empty((0, 3))
                while got.shape[0] < N:
                    m = max(int((N - got.shape[0]) * 3) + 64, 256)
                    cand = fit0.sample(m, rng)
                    keep = cand[rng.random(m) < np.abs(cand @ a)]
                    got = keep if got.shape[0] == 0 else np.concatenate([got, keep], 0)
                Xb[b] = got[:N]
            else:
                Xb[b] = fit0.sample(N, rng)
        if terzaghi:
            Wb = 1.0 / np.clip(np.abs(Xb @ a), 1e-6, None)
            Wb = Wb / Wb.mean(axis=1, keepdims=True)
            Wb = np.clip(Wb, 0.1, 5.0)
    else:
        raise ValueError(f"bootstrap_cone: mode 必须是 param/nonparam, got {mode!r}")

    u0 = np.tile(fit0.lmbda[1:], (B, 1))          # 热启动: 各重抽样解彼此很近
    Xbt = torch.as_tensor(Xb, dtype=torch.float64)
    Wbt = torch.as_tensor(Wb, dtype=torch.float64)
    u0t = torch.as_tensor(u0, dtype=torch.float64)
    # 分块: B·N 很大时一次前向会占大量内存, 且热启动后每块都很快收敛
    ang = np.empty(B, dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(N, 1)))
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        fits = fit_bingham_batch(Xbt[s:e], Wbt[s:e], u_init=u0t[s:e])
        for b in range(s, e):
            ang[b] = np.degrees(np.arccos(np.clip(
                abs(float(fit0.axes[:, idx] @ fits[b - s].axes[:, idx])), 0, 1)))
    return ang


def bootstrap_cone(fit0: Bingham, X, W, axis, terzaghi: bool, B: int,
                   level: float, mode: str, rng, idx: int = 0):
    """bootstrap 锥半角: 取角距分布的 (100·level) 分位。"""
    ang = bootstrap_angles(fit0, X, W, axis, terzaghi, B, mode, rng, idx=idx)
    return float(np.percentile(ang, 100.0 * level)), {"n_boot": B, "mode": mode}


def fit_calibration_scale(errs, cones, level: float, grid=None) -> float:
    """在**校准半集**上拟合缩放因子 γ, 使经验覆盖率 ≥ 标称。

    输入 (errs, cones): 每个试验的真实轴向误差角 (度) 与标称锥半角 (度)。
    判据: coverage(γ) = mean(err ≤ γ·cone) 对 γ 单调不减, 取满足
    coverage ≥ level 的**最小** γ (保守侧取等, 不做插值美化)。
    网格上界仍达不到时返回网格上界并在调用方标记 `saturated`。
    """
    errs = np.asarray(errs, dtype=np.float64)
    cones = np.asarray(cones, dtype=np.float64)
    if errs.shape != cones.shape:
        raise ValueError(f"fit_calibration_scale: errs{errs.shape} != cones{cones.shape}")
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 4.0, 396), [6.0, 10.0, 100.0]])
    for g in grid:
        if float(np.mean(errs <= g * cones)) >= level:
            return float(g)
    return float(grid[-1])
