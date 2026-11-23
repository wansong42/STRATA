# -*- coding: utf-8 -*-
"""Bingham 分布 (S^{p-1} 上的轴分布) — 拟合 / 密度 / 精确采样 (R185 T0).

为什么是 Bingham 而不是 Fisher/vMF
---------------------------------
裂隙产状是**轴数据** (法向 n 与 -n 是同一条结构面), 必须落在射影球面
S^{p-1}/{±1} 上。vMF 是**有向**分布 (钟在单一方向上), 拿它建模轴数据会把
±n 当成两个不同的峰, 聚类与均值都会被符号分裂污染 —— 这正是本项目八期
`setlabel._sign_align` 那一类 bug 的统计根源。Bingham 的密度

    f(x; M, λ) = exp( -Σ_k λ_k (m_kᵀx)² ) / c(λ),   x ∈ S^{p-1}

天然满足 f(x) = f(-x), 且**同一族同时表达两种地质形态**:

    polar  (极点簇)  λ = (0, κ, κ)      → 极点图上是一团 (最常见的组系形态)
    girdle (环带)    λ = (0, 0, κ)      → 极点图上是一条大圆弧 (褶皱/层面相关)
    triaxial         λ = (0, a, κ)      → 拉长的簇 (介于两者之间)

参数化 (口径锁定)
----------------
- `axes`  M = [m_1 … m_p] 正交, 列为特征方向;
- `lmbda` λ **升序**且 **min(λ) = 0** (因 Σx_k²=1, λ 与 λ-s·1 同分布, 故最小值
  钉 0 是唯一的规范形)。λ 升序 ⇔ M 的列按「模态优先」排列: m_1 = 众数轴
  (polar), m_p = 环带极 (girdle)。

MLE 是凸问题 (关键性质)
----------------------
对数似然只通过**二阶矩矩阵** S = Σw·xxᵀ/Σw 依赖数据:

    ℓ(M, λ) / (-Σw) = tr(Λ · Mᵀ S M) + log c(λ)

- 给定 λ, 由 von Neumann 迹不等式 + 重排不等式, M̂ 的列就是 S 的特征向量,
  且 λ 升序 ↔ S 特征值降序配对。**M̂ 有解析解, 不需要迭代。**
- 给定 M̂, 剩下 (p-1) 个浓度: 目标函数 F(λ) = Σλ_k d_k + log c(λ) 中 log c 是
  凸函数 (对数 Laplace 变换), 线性项凸 → **F 严格凸**, 牛顿法全局收敛、
  解唯一。这是本模块敢把拟合器用于无人值守批处理的唯一理由。

可微性
------
`log c(λ)` 走 `_sphere.log_bingham_norm` 的 torch 路径, 全程可微。浓度求解的
牛顿迭代用 `torch.where` 做回溯线搜索 (**不用 Python float 比较**, 否则计算图
被切断), 因此 `create_graph=True` 时梯度可从 λ̂ 一路反传到二阶矩矩阵 S,
再到原始法向 / 钻孔轴 —— 这是 T2 可微 Terzaghi 层的地基。

HONEST BOUNDARY
---------------
Bingham 是**椭圆对称**族, 不能表达斜/多峰。真实组系若本身双峰, 拟合结果是
两者的"折中椭圆", concentrations 会被低估, 且本模块**不会**给出任何双峰警告
(要做请走混合 Bingham, 另单)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ._sphere import (
    HAS_TORCH,
    log_bingham_norm,
    log_bingham_norm_np,
    sphere_area,
    hessian_of_scalar,
)

if HAS_TORCH:
    import torch

__all__ = [
    "Bingham",
    "bingham_from_scatter",
    "fit_bingham",
    "fit_bingham_batch",
    "LAMBDA_CAP",
]


LAMBDA_CAP = 1.0e4        # 浓度上界: 再高数值积分节点预算失控, 且无地质意义
_NEWTON_MAX_ITER = 30
_NEWTON_TOL = 1e-9
_LS_MARGIN = 1e-13        # 线搜索「严格改进裕度」(见 _newton_step 内的说明)


# ---------------------------------------------------------------------------
# 分布对象
# ---------------------------------------------------------------------------

@dataclass
class Bingham:
    """Bingham 分布。axes 列为特征方向, lmbda 升序且 min=0。"""

    axes: np.ndarray              # (p, p) 正交矩阵, 列为 m_1..m_p
    lmbda: np.ndarray             # (p,) 升序, lmbda[0] == 0
    log_norm: float               # log c(λ)
    n_eff: float = 0.0            # Σw (有效样本量)
    ess: Optional[float] = None   # 重要性加权有效样本量 (Terzaghi 下 < n_eff)
    converged: bool = True
    n_iter: int = 0
    grad_max: float = 0.0
    meta: dict = field(default_factory=dict)

    # -- 基本属性 ----------------------------------------------------------
    @property
    def p(self) -> int:
        return int(self.axes.shape[0])

    @property
    def kind(self) -> str:
        """polar / girdle / triaxial / uniform。

        判据是 λ 的**零元个数** (λ 升序, min=0):
            全零          → uniform (整球等密度)
            唯一最大元    → girdle  (λ=(0,…,0,κ), 密度集中在大圆上)
            唯一最小元    → polar   (λ=(0,κ,…,κ), 密度集中在 ±m₁)
            唯一最小元且次小≠最大 → triaxial (拉长的簇)
        ⚠️ 容易搞反: polar 是「唯一最小元」(密度 exp(-Σλ_k y_k²) 在 λ 最小的
        方向上最大), girdle 是「唯一最大元」。八期曾因类似方向性约定翻车。
        """
        # 用**相对**阈值 z = λ/λmax: 拟合 λ 的采样噪声是相对量 (κ=30 时 λ₃ 的
        # 标准误就是 ~1), 任何绝对阈值都会把真·girdle 误判成 triaxial。
        # 5% 是描述性分档, 不影响 distinguished() 的判定 (见下)。
        lam = np.asarray(self.lmbda, dtype=np.float64)
        if float(lam.max()) <= 1e-8:
            return "uniform"
        z = lam / float(lam.max())
        n_small = int(np.sum(z <= 0.05))
        if n_small >= self.p - 1:
            return "girdle"
        if n_small == 1:
            return "polar" if z[1] >= 0.95 else "triaxial"
        return "triaxial"

    def distinguished(self) -> Tuple[int, str]:
        """返回 (轴下标, 语义)。唯一最小元 → 众数轴; 唯一最大元 → 环带极。

        girdle 的「众数轴」不唯一 (整条大圆), 因此 girdle 族唯一可估的轴向
        目标是环带极。这是统计可辨识性, 不是实现取舍。
        """
        k = self.kind
        if k in ("polar", "triaxial"):
            return 0, "polar_axis"
        if k == "girdle":
            return self.p - 1, "girdle_pole"
        raise ValueError("uniform 分布没有可辨识的特征轴 (整球等密度)")

    def axis(self, i: int) -> np.ndarray:
        return np.asarray(self.axes[:, i], dtype=np.float64)

    # -- 密度 --------------------------------------------------------------
    def log_prob(self, x) -> np.ndarray:
        xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = xa @ self.axes
        return -((y ** 2) @ self.lmbda) - self.log_norm

    def prob(self, x) -> np.ndarray:
        return np.exp(self.log_prob(x))

    def moment(self) -> np.ndarray:
        """E[x xᵀ] = M diag(d) Mᵀ, d_k = E[x_k²] = -∂log c/∂λ_k。"""
        h = 1e-6
        d = np.empty(self.p)
        for k in range(self.p):
            lp = self.lmbda.copy()
            lm = self.lmbda.copy()
            lp[k] += h
            lm[k] -= h
            d[k] = -(log_bingham_norm_np(lp) - log_bingham_norm_np(lm)) / (2 * h)
        return self.axes @ np.diag(d) @ self.axes.T

    # -- 采样 --------------------------------------------------------------
    def sample(self, n: int, rng: Optional[np.random.Generator] = None,
               max_batches: int = 200) -> np.ndarray:
        """精确采样 (Kent–Ganeiber–Mardia 的 ACG 包络拒绝采样)。

        包络 ACG(Σ), Σ = M diag(1/(1+2λ_k)) Mᵀ。密度比只依赖 q = Σλ_k y_k²:
            f_B / f_ACG ∝ e^{-q}·(1+2q)^{p/2},   q ∈ [0, λmax]
        最大值有闭式 (驻点 q* = (p-1)/2), 故接受概率可**解析**给出,
        不需要经验调参, 也不会在低接受率时静默补样。
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        p = self.p
        lam = self.lmbda
        s = 1.0 / (1.0 + 2.0 * lam)
        L = self.axes @ np.diag(np.sqrt(s))

        def _g(q):
            return np.exp(-q) * (1.0 + 2.0 * q) ** (0.5 * p)

        acc_rate = self.acceptance_rate()
        G = self._envelope_G()

        out, got = [], 0
        for _ in range(max_batches):
            m = int(min(max(n - got, 1) / max(acc_rate, 1e-9) * 1.3 + 64, 2_000_000))
            g = rng.normal(size=(m, p))
            z = g @ L.T
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            q = (z @ self.axes) ** 2 @ lam
            keep = z[rng.random(m) < _g(q) / G]
            out.append(keep)
            got += int(keep.shape[0])
            if got >= n:
                break
        if got < n:
            raise RuntimeError(
                f"Bingham.sample: 仅采到 {got}/{n} (解析接受率 {acc_rate:.4f}), "
                f"拒绝预算耗尽 —— 拒绝静默补样")
        return np.concatenate(out, axis=0)[:n]

    def _envelope_G(self) -> float:
        """包络常数 G = max_{q∈[0,λmax]} e^{-q}(1+2q)^{p/2} (驻点 q*=(p-1)/2)。"""
        p = self.p
        lam = float(self.lmbda.max())
        qstar = float(np.clip((p - 1) / 2.0, 0.0, lam))

        def _g(q):
            return np.exp(-q) * (1.0 + 2.0 * q) ** (0.5 * p)
        return float(np.max(_g(np.array([0.0, lam, qstar]))))

    def acceptance_rate(self) -> float:
        """Kent–Ganeiber–Mardia 包络的**解析**接受概率 (无需试探)。

        接受率太低意味着拟合出的分布极集中, 是数值/建模问题的早期信号,
        因此把它做成一等公民而不是藏在采样循环里。
        """
        p = self.p
        s = 1.0 / (1.0 + 2.0 * self.lmbda)
        log_accept = (self.log_norm - np.log(self._envelope_G())
                      - 0.5 * float(np.sum(np.log(s))) - np.log(sphere_area(p)))
        return float(np.clip(np.exp(log_accept), 1e-12, 1.0))

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "p": self.p,
            "lmbda": [float(v) for v in self.lmbda],
            "axes": [[float(v) for v in row] for row in self.axes],
            "log_norm": float(self.log_norm),
            "kind": self.kind,
            "n_eff": float(self.n_eff),
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
            "grad_max": float(self.grad_max),
        }
        if self.ess is not None:
            d["ess"] = float(self.ess)
        if self.meta:
            d["meta"] = self.meta
        return d


# ---------------------------------------------------------------------------
# 拟合: 二阶矩矩阵 → M (解析) + λ (凸牛顿)
# ---------------------------------------------------------------------------

def _solve_lambda_torch(d_desc, u_init=None, max_iter=_NEWTON_MAX_ITER,
                        tol=_NEWTON_TOL, create_graph: bool = False):
    """批量求解 min_λ [ Σλ_k d_k + log c(λ) ], λ_1 = 0 钉住。

    d_desc : (B, p) **降序**的 S 特征值 (与 λ 升序配对)
    u_init : (B, p-1) 可选热启动。**bootstrap 重拟合必须热启动** —— 各重抽样本
             的解彼此很近, 热启动后 3~5 轮即收敛, 比冷启动快一个量级。
    create_graph : True 时保留计算图 (供可微层反传), 此时固定迭代不使用早停

    返回 (lam (B,p) 升序/min0, order (B,p) 列置换, converged (B,), n_iter, grad_max)
    """
    if not HAS_TORCH:
        raise RuntimeError("_solve_lambda_torch 需要 torch")
    Bsz, p = d_desc.shape
    dev, dt = d_desc.device, d_desc.dtype
    d_det = d_desc.detach()

    if u_init is not None:
        u = torch.as_tensor(u_init, dtype=dt, device=dev).reshape(Bsz, p - 1).clone()
    else:
        # 初值: 大浓度下 d_k ≈ 1/(2λ_k); 均匀 (d 全等) 时给 0 —— 两个极限都对
        lam_init = 0.5 * (1.0 / d_det.clamp_min(1e-6)
                          - 1.0 / d_det[:, :1].clamp_min(1e-6))
        u = lam_init[:, 1:].clone().clamp_(0.0, 50.0)
    u = u.requires_grad_(True)

    def _F(z):
        lam = torch.cat([torch.zeros(z.shape[0], 1, dtype=dt, device=dev), z], dim=-1)
        return (lam * d_desc).sum(-1) + log_bingham_norm(lam)

    def _grad(z):
        f = _F(z)
        return torch.autograd.grad(f.sum(), z, create_graph=True)[0]

    def _newton_step(d_a, u_a, use_linesearch: bool):
        """对一批活跃元素做一步牛顿, 返回 (新 u, 该步梯度)。

        use_linesearch=False (保图路径, 供可微层): **纯牛顿**, 不做任何分支选择。
            原因: 线搜索的 `torch.where(better, cand, best_u)` 是硬分支, 收敛点附近
            各候选的 F 差落到 roundoff 级别, 分支选择在 θ±h 两点可能不同 → autograd
            与有限差分差出 O(step/h)。实测 λ 通道误差因此达 2.5e-3 (S/eigh 通道
            只有 1e-8)。纯牛顿是光滑复合映射, 梯度精确; 目标函数严格凸 + 初值好,
            全步长牛顿本就收敛 (line search 只在初值很差时才有必要)。
        use_linesearch=True (批量实验路径): 保留回溯, 只要**值**不要梯度,
            对初值差的元素更鲁棒。
        """
        def _Fa(z):
            lam = torch.cat([torch.zeros(z.shape[0], 1, dtype=dt, device=dev), z], -1)
            return (lam * d_a).sum(-1) + log_bingham_norm(lam)

        def _ga(z):
            return torch.autograd.grad(_Fa(z).sum(), z, create_graph=True)[0]

        g = _ga(u_a)
        if not bool(torch.isfinite(g).all()):
            return u_a, g.detach()
        H = hessian_of_scalar(_ga, u_a)                      # (B, p-1, p-1)
        eye = torch.eye(p - 1, dtype=dt, device=dev).unsqueeze(0)
        tr = H.diagonal(dim1=-2, dim2=-1).sum(-1).abs()
        H = H + eye * (1e-10 * (tr + 1.0)).unsqueeze(-1).unsqueeze(-1)
        step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
        step = step.clamp(-LAMBDA_CAP, LAMBDA_CAP)

        if not use_linesearch:
            return u_a - step, g.detach()

        # 回溯线搜索: 全程 torch.where —— 不用 Python float 比较。
        # `better` 要求**严格改进裕度** (_LS_MARGIN): 收敛后 step→0, 各候选的 F 差
        # 落到 roundoff (1e-16) 级别, 只写 fc<best_f 会变成抛硬币。真实下降步的
        # 改进量远大于裕度, 不受影响。
        best_u, best_f = u_a, _Fa(u_a)
        alpha = 1.0
        for _ in range(3):
            cand = (u_a - alpha * step).clamp(-LAMBDA_CAP, LAMBDA_CAP)
            fc = _Fa(cand)
            better = torch.isfinite(fc) & (fc < best_f - _LS_MARGIN * (best_f.abs() + 1.0))
            best_u = torch.where(better.unsqueeze(-1), cand, best_u)
            best_f = torch.where(better, fc, best_f)
            alpha *= 0.25
        return best_u, g.detach()

    n_iter = 0
    grad_max = torch.full((Bsz,), float("inf"), dtype=dt, device=dev)
    converged = torch.zeros(Bsz, dtype=torch.bool, device=dev)

    if create_graph:
        # 需要保图: 不做早停/不改批量形状, 固定迭代, **纯牛顿** (无线搜索分支,
        # 见 _newton_step 内说明) —— 这是可微层梯度能过 gradcheck 的前提。
        for it in range(max_iter):
            n_iter = it + 1
            u, g = _newton_step(d_desc, u, use_linesearch=False)
            grad_max = g.abs().amax(dim=-1)
    else:
        # 活跃集收缩: 已收敛的元素**退出批量**。大批量里若等最慢的元素收敛,
        # 整批会被拖到 max_iter (实测 4000 条时 30 轮), 而绝大多数条目 4~5 轮
        # 就到位 —— 收缩后后续每轮的批量迅速变小, 这是覆盖率实验能跑完的关键。
        u = u.detach()
        active = torch.ones(Bsz, dtype=torch.bool, device=dev)
        for it in range(max_iter):
            n_iter = it + 1
            idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break
            u_a = u[idx].clone().requires_grad_(True)
            u_new, g_a = _newton_step(d_desc[idx], u_a, use_linesearch=True)
            u = u.index_copy(0, idx, u_new.detach())
            gm = g_a.abs().amax(dim=-1)
            grad_max = grad_max.index_copy(0, idx, gm)
            done = gm < tol
            active = active.index_copy(0, idx, ~done)
        converged = ~active

    lam = torch.cat([torch.zeros(u.shape[0], 1, dtype=dt, device=dev), u], dim=-1)
    # 规范化: 平移使 min = 0 (F 对 λ-s·1 不变, 无损), 再升序并给出列置换
    lam = torch.clamp(lam - lam.amin(dim=-1, keepdim=True), 0.0, LAMBDA_CAP)
    order = torch.argsort(lam, dim=-1)
    lam = torch.gather(lam, 1, order)
    if not create_graph:
        grad_max = grad_max.detach()
    return lam, order, converged, n_iter, grad_max


def _canonical_from_scatter_torch(S, u_init=None, create_graph: bool = False):
    """S: (B,p,p) torch → (lam (B,p), axes (B,p,p), converged, n_iter, grad_max)"""
    S = 0.5 * (S + S.transpose(-1, -2))
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    S = S / tr[:, None, None]
    evals, evecs = torch.linalg.eigh(S)                 # 升序
    d_desc = torch.flip(evals, dims=[-1])               # 降序
    M_desc = torch.flip(evecs, dims=[-1])               # 列与 d_desc 配对
    lam, order, conv, n_iter, gmax = _solve_lambda_torch(
        d_desc, u_init=u_init, create_graph=create_graph)
    M = torch.gather(M_desc, 2, order.unsqueeze(1).expand(-1, evals.shape[-1], -1))
    return lam, M, conv, n_iter, gmax


def bingham_params_from_scatter_t(S, u_init=None, create_graph: bool = False):
    """torch 版统一入口: 二阶矩矩阵 → (λ, M) 张量。

    `create_graph=True` 时保留计算图, 梯度可从 (λ, M) 反传到 S —— 可微
    Terzaghi 层 (terzaghi_layer.TerzaghiBinghamLayer) 依赖这条路径。
    """
    if not (HAS_TORCH and torch.is_tensor(S)):
        raise TypeError("bingham_params_from_scatter_t 需要 torch 输入")
    lam, M, conv, n_iter, gmax = _canonical_from_scatter_torch(
        S, u_init=u_init, create_graph=create_graph)
    return lam, M


def bingham_from_scatter(S, n_eff: float = 0.0, ess: Optional[float] = None) -> "Bingham":
    """从二阶矩矩阵 S 拟合 Bingham (单条)。S 可为 numpy 或 torch。

    这是**统一入口**: 普通拟合与 Terzaghi 加权拟合的差别, 仅仅是把 S 换成
    重要性加权版本 (见 terzaghi_layer.terzaghi_scatter)。
    """
    is_t = HAS_TORCH and torch.is_tensor(S)
    Snp = S.detach().cpu().numpy() if is_t else np.asarray(S, dtype=np.float64)
    Snp = np.atleast_2d(np.asarray(Snp, dtype=np.float64))
    if Snp.ndim != 2 or Snp.shape[0] != Snp.shape[1]:
        raise ValueError(f"bingham_from_scatter: S 必须方阵, got {Snp.shape}")
    if not np.isfinite(Snp).all():
        raise ValueError("bingham_from_scatter: S 含 NaN/Inf, 拒绝消费")
    tr = float(np.trace(Snp))
    if abs(tr) < 1e-12:
        raise ValueError("bingham_from_scatter: S 迹为 0, 拒绝消费")

    if HAS_TORCH:
        lam, M, conv, n_iter, gmax = _canonical_from_scatter_torch(
            torch.as_tensor(Snp, dtype=torch.float64).unsqueeze(0))
        lam_np = lam[0].detach().cpu().numpy()
        M_np = M[0].detach().cpu().numpy()
        converged = bool(conv[0].item())
        gmax_v = float(gmax.reshape(-1)[0])
    else:  # pragma: no cover
        Snp = 0.5 * (Snp + Snp.T) / tr
        evals, evecs = np.linalg.eigh(Snp)
        d_desc = evals[::-1]
        M_full = evecs[:, ::-1]
        lam_full = np.concatenate([[0.0],
                                   np.clip(0.5 * (1.0 / np.clip(d_desc, 1e-6, None)
                                                  - 1.0 / max(d_desc[0], 1e-6)), 0, LAMBDA_CAP)[1:]])
        lam_full = np.clip(lam_full - lam_full.min(), 0, LAMBDA_CAP)
        order = np.argsort(lam_full)
        lam_np, M_np = lam_full[order], M_full[:, order]
        converged, n_iter, gmax_v = False, 0, float("nan")

    return Bingham(axes=np.ascontiguousarray(M_np), lmbda=np.ascontiguousarray(lam_np),
                   log_norm=float(log_bingham_norm_np(lam_np)),
                   n_eff=float(n_eff), ess=ess, converged=converged,
                   n_iter=int(n_iter), grad_max=gmax_v)


def fit_bingham_batch(X, W=None, n_eff: Optional[float] = None,
                      ess: Optional[np.ndarray] = None, u_init=None):
    """批量拟合。X: (B,N,p) torch, W: (B,N) 可选权重。返回 list[Bingham]。

    覆盖率实验用: 一次前向在 (B·N) 规模上算配分函数, 比逐条拟合快 1~2 个量级。
    `u_init` 为 (B, p-1) 热启动 (bootstrap 重拟合强烈建议传, 见 _solve_lambda_torch)。
    """
    if not (HAS_TORCH and torch.is_tensor(X)):
        raise TypeError("fit_bingham_batch 需要 torch 输入")
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if W is None:
        Wt = torch.ones_like(Xn[..., 0])
    else:
        Wt = W.to(Xn.dtype)
    wsum = Wt.sum(-1).clamp_min(1e-12)
    S = torch.einsum("bn,bni,bnj->bij", Wt, Xn, Xn) / wsum[:, None, None]

    lam, M, conv, n_iter, gmax = _canonical_from_scatter_torch(S, u_init=u_init)
    lam_np = lam.detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()
    logn = log_bingham_norm(lam).detach().cpu().numpy()
    conv_np = conv.detach().cpu().numpy()
    gmax_np = gmax.reshape(-1).detach().cpu().numpy()
    neff_np = (wsum.detach().cpu().numpy() if n_eff is None
               else np.full(lam_np.shape[0], float(n_eff)))

    out = []
    for i in range(lam_np.shape[0]):
        out.append(Bingham(
            axes=np.ascontiguousarray(M_np[i]),
            lmbda=np.ascontiguousarray(lam_np[i]),
            log_norm=float(logn[i]), n_eff=float(neff_np[i]),
            ess=(float(ess[i]) if ess is not None else None),
            converged=bool(conv_np[i]), n_iter=int(n_iter),
            grad_max=float(gmax_np[i])))
    return out


def fit_bingham(x, weights=None, ess: Optional[float] = None) -> "Bingham":
    """从观测 (N,p) 拟合单条 Bingham。x: numpy 或 torch。"""
    is_t = HAS_TORCH and torch.is_tensor(x)
    X = x.detach().cpu().numpy() if is_t else np.asarray(x, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("fit_bingham: 输入需 (N,p) 且 N>=1, 拒绝静默降级")
    if not np.isfinite(X).all():
        raise ValueError("fit_bingham: 输入含 NaN/Inf, 拒绝消费")
    if (np.linalg.norm(X, axis=1) <= 0).any():
        raise ValueError("fit_bingham: 输入含零向量 (无方向), 拒绝消费")

    n, p = X.shape
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    if weights is None:
        W = np.ones(n, dtype=np.float64)
    else:
        W = np.asarray(weights, dtype=np.float64).reshape(-1)
        if W.shape[0] != n:
            raise ValueError(f"fit_bingham: weights 长度 {W.shape[0]} != 样本数 {n}")
        if not np.isfinite(W).all() or (W < 0).any():
            raise ValueError("fit_bingham: weights 含负值/非有限值, 拒绝消费")
        if W.sum() <= 0:
            raise ValueError("fit_bingham: weights 全零, 拒绝消费")
    S = (W[:, None, None] * Xn[:, :, None] * Xn[:, None, :]).sum(0) / W.sum()
    return bingham_from_scatter(S, n_eff=float(W.sum()), ess=ess)
