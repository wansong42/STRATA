# -*- coding: utf-8 -*-
"""角中心高斯 (Angular Central Gaussian, ACG) — R185 T1。

    f(x; Σ) = |Σ|^{-1/2} · (xᵀΣ⁻¹x)^{-p/2} / σ_p,   x ∈ S^{p-1},  Σ ≻ 0

为什么需要 ACG (而不是继续用 Bingham)
------------------------------------
1. **采样天然可微**: x = Lg / ‖Lg‖, g ~ N(0,I), Σ = LLᵀ (Cholesky)。
   这就是「重参数化采样」, 采样本身就是一次前向, 梯度直接流回 Σ。
   Bingham 只能拒绝采样, 拒绝步是不可微的 —— 若未来 B 层要让网络输出分布并
   对采样结果求梯度, 只能走 ACG。
2. **尺度不变**: Σ 与 c·Σ 同一分布, 故按 trace(Σ) = p 归一。
3. 与 Bingham 一样满足 f(x) = f(-x) (轴分布), 可直接用于法向数据。

代价: ACG 是**对数二次**族, Bingham 是**二次**族, 两者只在「高浓度 + 模态
附近」局部等价。本模块把这条等价关系写成**两种显式映射**并量化它们的矩差
(`bingham_acg_moment_report`), 不做「看起来差不多」的含糊对接。

两种 Bingham→ACG 映射 (重要, 别混用)
------------------------------------
设 Bingham 浓度 λ (升序, λ₁=0), 共同主轴 M, 记 s_i 为 ACG 的 Σ 特征值。

- `moment`  (矩匹配):  s_i = 1 / (1 + 2λ_i / p)
    推导: 在模态 x≈±m₁ 附近, 令 y = Mᵀx, y₁² = 1-Σ_{i≥2}y_i²,
        xᵀΣ⁻¹x = 1 + Σ_{i≥2} y_i²(1/s_i - 1)
        log f_ACG = -(p/2)log(xᵀΣ⁻¹x) ≈ -(p/2)Σ_{i≥2} y_i²(1/s_i - 1)
        log f_B   = -Σ_{i≥2} λ_i y_i²
    逐项匹配得 (p/2)(1/s_i - 1) = λ_i。**这是矩/曲率匹配, 用于「用 ACG 近似
    Bingham 的不确定性」。**

- `envelope` (最优拒绝包络):  s_i = 1 / (1 + 2λ_i)
    Kent–Ganeiber–Mardia (2018) 的包络: 令 Σ⁻¹ = I - 2A (A = -MΛMᵀ), 则
        f_B/f_ACG ∝ e^{-q}(1+2q)^{p/2},  q = Σλ_i y_i²
    其最大值有闭式, 该包络**最大化拒绝采样接受率**。`Bingham.sample` 用的就是它。

⚠️ 两者在 p=3 时差一个因子 3 (2λ/3 vs 2λ)。把包络 ACG 当成 Bingham 的矩代理
会**系统性低估散布** —— 这是本模块要明确钉住的一条可追溯偏差。

HONEST BOUNDARY
---------------
ACG 的 MLE (Tyler 不动点) 并非总存在: 数据过于集中时目标函数无界, Σ 会退化。
本模块**不做**截断救场, 而是在条件数超限时把 `converged=False` 标出来由调用方
决定 (退路: 改用 Bingham 拟合)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._sphere import sphere_area, HAS_TORCH

if HAS_TORCH:
    import torch

__all__ = [
    "ACG",
    "fit_acg",
    "acg_from_bingham",
    "acg_from_bingham_calibrated",
    "bingham_acg_moment_report",
    "rms_opening_deg",
    "COND_LIMIT",
]


def rms_opening_deg(w_desc):
    """公开入口: 二阶矩特征值 (降序) → RMS 张角 (度)。"""
    return _rms_opening_deg(w_desc)


COND_LIMIT = 1.0e10     # Σ 条件数上限: 超过即判定 MLE 退化 (数据过集中)


@dataclass
class ACG:
    """角中心高斯。Sigma 为 (p,p) 正定, 按 trace = p 归一 (尺度不变)。"""

    Sigma: np.ndarray
    n_eff: float = 0.0
    converged: bool = True
    n_iter: int = 0
    cond: float = 1.0
    meta: dict = None

    @property
    def p(self) -> int:
        return int(self.Sigma.shape[0])

    @property
    def eig(self):
        """(spreads 升序, axes 列)。spread 越小 = 该方向越集中。"""
        w, V = np.linalg.eigh(self.Sigma)
        return w, V

    @property
    def axes(self) -> np.ndarray:
        return self.eig[1]

    @property
    def spreads(self) -> np.ndarray:
        return self.eig[0]

    def log_prob(self, x) -> np.ndarray:
        xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
        w, V = self.eig
        y = xa @ V                                   # 主轴坐标
        q = (y ** 2) @ (1.0 / w)                     # xᵀΣ⁻¹x
        return (-0.5 * float(np.sum(np.log(w)))
                - 0.5 * self.p * np.log(np.maximum(q, 1e-300))
                - np.log(sphere_area(self.p)))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng(0)
        L = np.linalg.cholesky(self.Sigma)
        g = rng.normal(size=(int(n), self.p))
        z = g @ L.T
        return z / np.linalg.norm(z, axis=1, keepdims=True)

    def rsample(self, n: int, generator=None):
        """torch 重参数化采样 —— 梯度可从采样结果反传到 Sigma。

        Cholesky 在 torch 中可微 (`torch.linalg.cholesky`), 归一化也是可微的,
        因此整条采样路径对 Σ 可导。这就是「采样即前向」。
        """
        if not HAS_TORCH:
            raise RuntimeError("rsample 需要 torch")
        St = torch.as_tensor(self.Sigma, dtype=torch.float64)
        L = torch.linalg.cholesky(St)
        g = torch.randn(int(n), self.p, dtype=St.dtype, generator=generator)
        z = g @ L.T
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def to_dict(self) -> dict:
        w, V = self.eig
        return {
            "p": self.p,
            "spreads": [float(v) for v in w],
            "axes": [[float(v) for v in row] for row in V],
            "Sigma": [[float(v) for v in row] for row in self.Sigma],
            "n_eff": float(self.n_eff),
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
            "cond": float(self.cond),
        }


def fit_acg(x, weights=None, iters: int = 500, tol: float = 1e-12) -> ACG:
    """ACG 的 Tyler 不动点 MLE。

        Σ ← p · Σ_i w_i (x_i x_iᵀ) / (x_iᵀ Σ⁻¹ x_i)  /  Σ_i w_i

    不动点存在性有前提 (数据不能过于集中); 违反时 Σ 会退化, 本函数通过条件数
    识别并标 converged=False, 不做截断救场。
    """
    X = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("fit_acg: 输入需 (N,p) 且 N>=1, 拒绝静默降级")
    if not np.isfinite(X).all():
        raise ValueError("fit_acg: 输入含 NaN/Inf, 拒绝消费")
    nrm = np.linalg.norm(X, axis=1)
    if (nrm <= 0).any():
        raise ValueError("fit_acg: 输入含零向量, 拒绝消费")
    X = X / nrm[:, None]
    N, p = X.shape

    if weights is None:
        W = np.ones(N)
    else:
        W = np.asarray(weights, dtype=np.float64).reshape(-1)
        if W.shape[0] != N:
            raise ValueError(f"fit_acg: weights 长度 {W.shape[0]} != 样本数 {N}")
        if not np.isfinite(W).all() or (W < 0).any() or W.sum() <= 0:
            raise ValueError("fit_acg: weights 非法, 拒绝消费")
    W = W / W.sum()

    S = (W[:, None, None] * X[:, :, None] * X[:, None, :]).sum(0)
    S /= np.trace(S)
    Sig = S * p                       # 用二阶矩矩阵起步
    n_iter = 0
    converged = False
    for it in range(iters):
        n_iter = it + 1
        Sinv = np.linalg.pinv(Sig)
        q = np.einsum("ij,jk,ik->i", X, Sinv, X)
        q = np.maximum(q, 1e-300)
        new = p * (W[:, None, None] * X[:, :, None] * X[:, None, :] / q[:, None, None]).sum(0)
        new /= (np.trace(new) / p)
        new = 0.5 * (new + new.T)
        ev = np.linalg.eigvalsh(new)
        cond = float(ev[-1] / max(ev[0], 1e-300))
        if not np.isfinite(new).all():
            break
        delta = float(np.max(np.abs(new - Sig)))
        Sig = new
        if delta < tol:
            converged = True
            break
    ev = np.linalg.eigvalsh(Sig)
    cond = float(ev[-1] / max(ev[0], 1e-300))
    if cond > COND_LIMIT or not np.isfinite(cond):
        converged = False
    return ACG(Sigma=Sig, n_eff=float(N), converged=converged, n_iter=n_iter, cond=cond)


def acg_from_bingham(b, mode: str = "moment") -> ACG:
    """Bingham → ACG (同轴、按选定映射定散布)。mode ∈ {moment, envelope}。"""
    if mode not in ("moment", "envelope"):
        raise ValueError(f"acg_from_bingham: mode 必须是 moment/envelope, got {mode!r}")
    lam = np.asarray(b.lmbda, dtype=np.float64)
    p = b.p
    if mode == "moment":
        s = 1.0 / (1.0 + 2.0 * lam / p)
    else:
        s = 1.0 / (1.0 + 2.0 * lam)
    # 若 λ 全 0 (均匀), s 全 1 → Σ = I ✓
    Sigma = b.axes @ np.diag(s) @ b.axes.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma * (p / np.trace(Sigma))
    return ACG(Sigma=Sigma, n_eff=float(getattr(b, "n_eff", 0.0)),
               converged=True, n_iter=0, cond=float(s.max() / max(s.min(), 1e-300)),
               meta={"source": "bingham", "mode": mode, "lmbda": [float(v) for v in lam]})


def acg_from_bingham_calibrated(b, n_mc: int = 200_000, seed: int = 0,
                                lo: float = 0.25, hi: float = 8.0,
                                iters: int = 40, tol_deg: float = 0.02):
    """**数值标定**的 Bingham→ACG 映射: 让 ACG 的 RMS 张角匹配 Bingham 的。

    为什么需要: 曲率匹配 (moment) 只保证模态附近的二次项一致, 但 ACG 是
    **多项式重尾** ((xᵀΣ⁻¹x)^{-p/2}), Bingham 是指数尾; 尾部主导二阶矩,
    于是曲率匹配的 ACG 在整体上明显**更弥散** (实测 κ=10~40 时 RMS 张角
    高估 10~20°)。把它当矩代理用会把不确定度系统性放大。

    标定族:  s(t) = s_curv ** t        (t=1 即曲率匹配, t↑ → 更紧)
    单调性: t↑ ⇒ s↓ ⇒ 更集中, 故可用二分法。

    用**公共随机数 (CRN)**: 固定标准正态矩阵 G, 采样写作 normalize(G L(t)ᵀ),
    于是 f(t) 对 t 确定且连续 —— 二分法才收敛得干净 (否则 MC 噪声会让
    分位数估计抖动, 得到的是噪声而不是标定)。

    返回 (ACG, info)。info 含标定指数 t、残差、是否撞到边界。
    """
    from .bingham import Bingham as _B

    s_curv = 1.0 / (1.0 + 2.0 * np.asarray(b.lmbda, dtype=np.float64) / b.p)
    target = _rms_opening_deg(np.linalg.eigvalsh(b.moment())[::-1])
    axes = np.asarray(b.axes, dtype=np.float64)
    G = np.random.default_rng(seed).normal(size=(int(n_mc), b.p))

    def rms_at(t):
        s = s_curv ** t
        L = axes @ np.diag(np.sqrt(np.maximum(s, 1e-12)))
        z = G @ L.T
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        S = (z[:, :, None] * z[:, None, :]).mean(0)
        return _rms_opening_deg(np.linalg.eigvalsh(S)[::-1])

    f_lo, f_hi = rms_at(lo) - target, rms_at(hi) - target
    saturated = False
    if f_lo < 0:                       # 最松端已经比目标紧
        t_best, saturated = lo, True
    elif f_hi > 0:                     # 最紧端仍比目标松
        t_best, saturated = hi, True
    else:
        a, bb = lo, hi
        for _ in range(iters):
            m = 0.5 * (a + bb)
            if rms_at(m) - target > 0:
                a = m
            else:
                bb = m
            if bb - a < 1e-4:
                break
        t_best = 0.5 * (a + bb)

    s = s_curv ** t_best
    Sigma = axes @ np.diag(s) @ axes.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma * (b.p / np.trace(Sigma))
    a_out = ACG(Sigma=Sigma, n_eff=float(getattr(b, "n_eff", 0.0)), converged=not saturated,
                n_iter=0, cond=float(s.max() / max(s.min(), 1e-300)),
                meta={"source": "bingham", "mode": "calibrated", "exponent_t": float(t_best)})
    return a_out, {
        "exponent_t": float(t_best),
        "target_rms_deg": float(target),
        "calibrated_rms_deg": float(rms_at(t_best)),
        "residual_deg": float(rms_at(t_best) - target),
        "saturated": bool(saturated),
        "n_mc": int(n_mc),
    }


def _rms_opening_deg(w_desc):
    """二阶矩特征值 (降序) → 相对主轴的 RMS 张角 (度)。

    对 polar 型: cos θ_rms = sqrt(d₁)。用度而不是无量纲特征值, 是为了让矩差
    可以直接和产状精度 (°) 对照。
    """
    d = np.sort(np.asarray(w_desc, dtype=np.float64))[::-1]
    return float(np.degrees(np.arccos(np.clip(np.sqrt(np.clip(d[0], 0.0, 1.0)), 0.0, 1.0))))


def _moment_axes_deg(w):
    """把二阶矩特征值 (降序) 换成「相对主轴的 RMS 张角 (度)」—— 物理可比量。

    对 polar 型: θ_rms = sqrt(1 - d₁)。用度而不是无量纲特征值, 是为了让矩差
    可以直接和产状精度 (°C) 对照。
    """
    d = np.sort(np.asarray(w, dtype=np.float64))[::-1]
    return float(np.degrees(np.arccos(np.clip(np.sqrt(np.clip(d[0], 0, 1)), 0, 1))))


def bingham_acg_moment_report(b, n_mc: int = 400_000, seed: int = 0,
                              modes=("moment", "envelope")) -> dict:
    """同一 Bingham 下, 两种 ACG 映射的矩差报告 (T1 交付物)。

    对每种 mode 报告:
      - 主轴角差 (度): 应 ≈ 0 (同轴构造保证, 用于抓实现 bug)
      - 二阶矩特征值最大绝对差
      - **RMS 张角差 (度)**: 物理可比的矩差 —— 这是真正要报给下游的数字
      - MC 标准误 (不隐瞒 MC 噪声)
    """
    rng = np.random.default_rng(seed)
    Bm = b.moment()
    wB = np.linalg.eigvalsh(Bm)[::-1]
    theta_B = _moment_axes_deg(wB)

    rows = {}
    for mode in modes:
        a = acg_from_bingham(b, mode=mode)
        X = a.sample(n_mc, rng)
        S = (X[:, :, None] * X[:, None, :]).mean(0)
        wA = np.linalg.eigvalsh(S)[::-1]
        VA = np.linalg.eigh(S)[1][:, ::-1]
        # MC 标准误: Var(x_i x_j) 的估计 / sqrt(n)
        M2 = (X[:, :, None] * X[:, None, :])
        se = float(np.max(M2.std(axis=0) / np.sqrt(n_mc)))
        # 主轴 (最大特征值方向) 与 Bingham 主轴的角差
        iB = int(np.argmax(wB))
        ax_b = np.linalg.eigh(Bm)[1][:, ::-1][:, 0]
        ang = float(np.degrees(np.arccos(np.clip(abs(float(ax_b @ VA[:, 0])), 0, 1))))
        theta_A = _moment_axes_deg(wA)
        rows[mode] = {
            "spreads": [float(v) for v in a.spreads],
            "principal_axis_angle_diff_deg": ang,
            "moment_eig_max_abs_diff": float(np.max(np.abs(np.sort(wA)[::-1] - wB))),
            "rms_opening_angle_deg": theta_A,
            "rms_opening_angle_diff_deg": theta_A - theta_B,
            "mc_n": int(n_mc),
            "mc_max_se": se,
        }
    return {
        "bingham_lmbda": [float(v) for v in b.lmbda],
        "bingham_kind": b.kind,
        "bingham_rms_opening_angle_deg": theta_B,
        "acg": rows,
        "n_mc": int(n_mc),
        "note": ("envelope 映射是 Kent-Ganeiber-Mardia 最优拒绝包络 (给 Bingham.sample 用), "
                 "不是矩匹配; 把它当矩代理会系统性低估散布。矩匹配请用 moment。"),
    }
