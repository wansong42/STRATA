# -*- coding: utf-8 -*-
"""球面指数族配分函数数值底座 (R185).

本模块只解决一件事: 精确、可微、向量化地算出 Bingham 分布的配分函数

    c(λ) = ∫_{S^{p-1}} exp( -Σ_k λ_k x_k² ) dS(x),      λ ∈ R^p

并由此支撑 Bingham 的密度/采样/凸 MLE, 以及未来 E3GT 分布头的反向传播。

设计要点
--------
1. **单算法 + 双后端**: 同一个递推在 numpy / torch 上跑, 由输入类型分派。
   两条路径不可能"悄悄漂移", 一致性由 `tests/test_v_r185_.py` 钉死 (≤1e-10)。
   numpy 路径用于打分/审计, torch 路径用于 autograd (拟合与可微层共用)。

2. **禁查表、禁硬编码域外值**: 配分函数走**自适应 Gauss-Legendre 数值积分**,
   不做域截断, 不做插值表。节点数随浓度自适应 (`_node_count`)。

3. **降维递推**: 利用球面坐标的嵌套结构
      x = (cosθ, sinθ·u),  u ∈ S^{p-2},  dS_p = sin^{p-2}θ dθ dS_{p-1}
   把 p 维球面积分降成 (p-2) 维积分:
      p=2 (圆)   闭式: 2π·e^{-min(λ)}·I₀e(|λ₁-λ₂|/2)     ← 精确, 指数缩放防溢出
      p=3 (球面) 1 维积分 (裂隙产状主用路径)
      p=4 (四元数/旋转) 2 维积分  (E3GT 对接面)
   全程在 log 空间做 logsumexp, 浓度再大也不溢出。

   ⚠️ 换元到 θ 是**必须的**: 若直接在 t=cosθ 上积分, Jacobian 含 (1-t²)^{(p-3)/2},
   p≥4 时在端点有平方根/分数次奇点, Gauss-Legendre 退化为代数收敛
   (实测 p=4, λ=0 相对误差 2e-6)。换成 sin^{p-2}θ 后被积函数解析, 指数收敛。
   这是本模块最关键的一处数值决定, 勿改回 t 参数化。

4. **变换不变性**: c(λ) 对 λ 的置换不变; 且 λ 与 λ - s·1 给出同一密度
   (因为 Σx_k² = 1), 故 c(λ - s) = e^{s}·c(λ)。节点数用 **极差 λmax-λmin**
   定标 (唯一有意义的尺度), 保证平移不变性不被数值策略破坏。

HONEST BOUNDARY
---------------
数值积分不是闭式。p=3 用 Dawson 闭式 (对称/Watson 族) 独立校验到 ~1e-12;
p≥4 用 λ=0 的球面积分闭式 + 1 维降阶独立校验。残差随浓度升高而增大,
`node_count` 的自适应系数即据此标定, 超出标定范围请如实登记而不静默接受。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import i0e as _i0e_np

try:  # torch 是项目主依赖; 缺失时退化为纯 numpy (仅打分, 不能拟合)
    import torch
except Exception:  # pragma: no cover
    torch = None


__all__ = [
    "sphere_area",
    "log_bingham_norm",
    "log_bingham_norm_np",
    "grad_log_bingham_norm",
    "hessian_of_scalar",
    "HAS_TORCH",
    "DOUBLE_BACKWARD_OK",
]


HAS_TORCH = torch is not None

# 单位球面 S^{p-1} ⊂ R^p 的表面积 (前几档硬编码避免 gamma 误差, 其余走公式)
_SPHERE_AREA_CACHE = {
    2: 2.0 * math.pi,            # 圆
    3: 4.0 * math.pi,            # 普通球面
    4: 2.0 * math.pi ** 2,       # S³ (四元数单位球)
    5: 8.0 * math.pi ** 2 / 3.0,
    6: math.pi ** 3,
}

_TINY = 1e-300


def sphere_area(p: int) -> float:
    """单位球面 S^{p-1} (R^p 中) 的表面积。"""
    if p < 2:
        raise ValueError(f"sphere_area: p 必须 >= 2, got {p}")
    if p in _SPHERE_AREA_CACHE:
        return _SPHERE_AREA_CACHE[p]
    return 2.0 * math.pi ** (p / 2.0) / math.gamma(p / 2.0)


# ---------------------------------------------------------------------------
# 后端分派: 同一份递推代码跑在 numpy 与 torch 上
# ---------------------------------------------------------------------------

class _NpBackend:
    name = "numpy"

    @staticmethod
    def amax(x, dim):
        return np.max(x, axis=dim)

    @staticmethod
    def sum(x, dim):
        return np.sum(x, axis=dim)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def sin(x):
        return np.sin(x)

    @staticmethod
    def log_i0e(x):
        # I₀e(x) = I₀(x)·e^{-x}, scipy 已做指数缩放, x 很大时也不会溢出/下溢
        return np.log(np.maximum(_i0e_np(x), _TINY))

    @staticmethod
    def const(arr, like):
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def is_tensor(x):
        return False

    @staticmethod
    def to_float_scalar(x):
        return float(np.max(np.abs(np.asarray(x))))


class _TorchBackend:
    name = "torch"

    @staticmethod
    def amax(x, dim):
        return torch.amax(x, dim=dim)

    @staticmethod
    def sum(x, dim):
        return torch.sum(x, dim=dim)

    @staticmethod
    def log(x):
        return torch.log(x)

    @staticmethod
    def exp(x):
        return torch.exp(x)

    @staticmethod
    def minimum(a, b):
        return torch.minimum(a, b)

    @staticmethod
    def abs(x):
        return torch.abs(x)

    @staticmethod
    def sin(x):
        return torch.sin(x)

    @staticmethod
    def log_i0e(x):
        # torch.special.i0e 与 scipy 同义 (指数缩放贝塞尔), 支持一阶 autograd
        return torch.log(torch.clamp(torch.special.i0e(x), min=_TINY))

    @staticmethod
    def const(arr, like):
        return torch.as_tensor(arr, dtype=like.dtype, device=like.device)

    @staticmethod
    def is_tensor(x):
        return torch.is_tensor(x)

    @staticmethod
    def to_float_scalar(x):
        return float(torch.amax(torch.abs(x.detach())).item())


def _backend_of(x):
    return _TorchBackend if (HAS_TORCH and torch.is_tensor(x)) else _NpBackend


# ---------------------------------------------------------------------------
# 自适应节点
# ---------------------------------------------------------------------------

def _node_count(spread: float, p: int) -> int:
    """配分函数积分的 Gauss-Legendre 节点数。

    spread = λmax - λmin (平移不变的唯一尺度)。被积函数在 θ 方向的特征宽度
    ~ 1/√spread, 故节点数按 √spread 增长。

    系数由 Dawson 闭式扫描**实测标定** (p=3, 对称族; 另用 n=1024 自参考扫了
    非对称族), 目标残差 ~1e-13:

        spread   5     20     60     200    600
        够用 n   24     40     48     64     96

    拟合得 n ≈ 24 + 4·√spread, 并留 ~30% 余量取 32 下限。
    ⚠️ 旧公式 48+8√spread 过于保守 (spread=15 时给 79 个节点, 实测 40 个
    就够 1e-13), 白烧 2 倍算力 —— 覆盖率实验要跑百万次拟合, 这里是主要成本项。
    """
    coef = 4.0 if p <= 3 else 8.0     # p>=4 是嵌套 (p-2) 维积分, 误差按维数累积
    n = int(24.0 + coef * math.sqrt(max(float(spread), 0.0)))
    return int(min(max(n, 32), 2048))


_GL_CACHE: dict = {}


def _gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """[-1,1] 上 n 点 Gauss-Legendre 节点/权重 (权重和为 2)。带缓存。

    ⚠️ leggauss 是 O(n²) 的特征值求解, 每次配分函数调用都重算会让单点耗时
    从 ~30us 涨到 ~2.6ms (实测)。缓存后节点/权重只算一次。
    """
    got = _GL_CACHE.get(n)
    if got is None:
        got = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = got
    return got


def _gauss_legendre_on(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """[a,b] 区间上的 Gauss-Legendre 节点/权重 (权重和为 b-a)。"""
    x, w = _gauss_legendre(n)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return mid + half * x, half * w


# ---------------------------------------------------------------------------
# 核心递推
# ---------------------------------------------------------------------------

def _log_norm_rec(l, B, n: int, p: int):
    """递推体。l: (..., p) 实数浓度 (可为负, 平移不变); 返回 (...)。"""
    if p == 2:
        l1 = l[..., 0]
        l2 = l[..., 1]
        # c₂(λ) = ∫₀^{2π} e^{-λ₁cos²φ-λ₂sin²φ}dφ = 2π·e^{-min(λ)}·I₀e(|λ₁-λ₂|/2)
        # 恒等变形 (-(λ₁+λ₂)/2 + |λ₁-λ₂|/2 = -min(λ₁,λ₂)) 使任意大浓度都不溢出。
        return (math.log(2.0 * math.pi)
                - B.minimum(l1, l2)
                + B.log_i0e(B.abs(l1 - l2) * 0.5))

    th_np, w_np = _gauss_legendre_on(0.0, math.pi, n)
    th = B.const(th_np, l).reshape((1,) * (l.ndim - 1) + (n,))
    w = B.const(w_np, l).reshape((1,) * (l.ndim - 1) + (n,))

    sn = B.sin(th)                        # (..., n), >0 (GL 节点严格在 (0,π) 内)
    s = sn * sn                           # sin²θ
    rest = l[..., 1:]                     # (..., p-1)
    inner = _log_norm_rec(rest[..., None, :] * s[..., :, None], B, n, p - 1)  # (..., n)

    # Jacobian sin^{p-2}θ  (解析, 无端点奇点) + exp(-λ₁cos²θ) = exp(-λ₁(1-sin²θ))
    g = (p - 2) * B.log(sn) - l[..., 0:1] * (1.0 - s) + inner

    gw = g + B.log(w)
    m = B.amax(gw, -1)
    return m + B.log(B.sum(B.exp(gw - m[..., None]), -1))


def log_bingham_norm(lmbda, n_nodes: Optional[int] = None):
    """log c(λ) = log ∫_{S^{p-1}} exp(-Σ_k λ_k x_k²) dS(x)。

    Parameters
    ----------
    lmbda : (..., p) array / tensor
        浓度向量。可为任意实数 (平移 λ-s·1 只改变 c 的 e^s 因子, 分布不变)。
    n_nodes : int, optional
        固定 Gauss-Legendre 节点数; 默认按 λ 极差自适应。

    Returns
    -------
    (...,) 与输入同后端、同 dtype 的 log 配分函数。
    """
    if not hasattr(lmbda, "shape"):
        lmbda = np.asarray(lmbda, dtype=np.float64)
    p = lmbda.shape[-1]
    if p < 2:
        raise ValueError(f"log_bingham_norm: λ 最后一维必须 >= 2, got {p}")

    B = _backend_of(lmbda)
    spread = 0.0
    if n_nodes is None:
        lam = lmbda.detach() if (HAS_TORCH and torch.is_tensor(lmbda)) else lmbda
        if hasattr(lam, "numpy"):
            lam = lam.detach().cpu().numpy()
        amax = float(np.max(np.asarray(lam)))
        amin = float(np.min(np.asarray(lam)))
        spread = amax - amin
        n = _node_count(spread, p)
    else:
        n = int(n_nodes)

    x = lmbda
    if HAS_TORCH and torch.is_tensor(x) and x.dtype != torch.float64:
        x = x.double()

    # ⚠️ 升序排序后再递推 —— 这不是美化, 是 7 个量级的精度差:
    # c(λ) 本就置换不变, 但递推把 λ₁ 放在 exp(-λ₁cos²θ) 外层。若 λ₁ 大,
    # 外层因子在 θ=π/2 处尖锐集中, 与内层 c_{p-1}(sin²θ·λ_rest) 的集中叠加,
    # Gauss-Legendre 精度崩掉 (实测 λ=(60,60,0) 在 n=48 时误差 8.7e-7,
    # 排序成 (0,60,60) 后 1.8e-14)。最小 λ 放外层 → 外层因子恒为 1,
    # 集中全部由内层的**闭式** p=2 基例承担 (i0e 精确), 数值上最优。
    # 升序在递推的每一层都保持 (内层只乘正数 (1-t²)…此处为 sin²θ)。
    if HAS_TORCH and torch.is_tensor(x):
        x = torch.sort(x, dim=-1).values
    else:
        x = np.sort(np.asarray(x, dtype=np.float64), axis=-1)

    # 高维球 (p>=4) 递推会产生 (..., n^{p-2}) 中间量, 按预算分块防爆内存
    if p >= 4:
        budget = 4_000_000
        per = int(n ** (p - 2))
        chunk = max(1, budget // max(per, 1))
        if x.ndim >= 2 and x.shape[0] > chunk:
            outs = [log_bingham_norm(x[i:i + chunk], n_nodes=n) for i in range(0, x.shape[0], chunk)]
            return (torch.cat(outs, 0) if (HAS_TORCH and torch.is_tensor(x)) else np.concatenate(outs, 0))

    # 统一成 (Bsz, p) 再还原形状, 简化广播
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, p) if x.ndim > 1 else x.reshape(1, p)
    out = _log_norm_rec(flat, B, n, p)
    if x.ndim == 1:
        return out.reshape(())
    return out.reshape(orig_shape)


def log_bingham_norm_np(lmbda, n_nodes: Optional[int] = None) -> np.ndarray:
    """强制 numpy 路径 (审计/独立复算用)。"""
    arr = np.asarray(lmbda, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] < 2:
        raise ValueError("log_bingham_norm_np: λ 长度必须 >= 2")
    return np.asarray(log_bingham_norm(arr, n_nodes=n_nodes), dtype=np.float64)


# ---------------------------------------------------------------------------
# autograd 辅助
# ---------------------------------------------------------------------------

def _probe_double_backward() -> bool:
    """探测 i0e / 递推链是否支持二阶反向 (决定 Hessian 走解析还是有限差分)。"""
    if not HAS_TORCH:
        return False
    try:
        lam = torch.tensor([0.0, 1.3, 2.7], dtype=torch.float64, requires_grad=True)
        f = log_bingham_norm(lam)
        g = torch.autograd.grad(f, lam, create_graph=True)[0]
        torch.autograd.grad(g[1], lam, retain_graph=True)[0]
        return True
    except Exception:
        return False


"""
DOUBLE_BACKWARD_OK 已惰性化 (R187-T1, 坑 33): 探测不再在 import 时执行 ——
模块级探测曾在 import 期跑 autograd+leggauss, 撞 torch/MKL 双 OpenMP 即
OMP Error #15 硬崩。现通过模块级 __getattr__ (PEP 562) 惰性转发, 或用
显式入口 verify_double_backward() 触发; 首次访问后缓存, 语义与旧常量一致。
"""

_DOUBLE_BACKWARD_CACHE: Optional[bool] = None


def verify_double_backward() -> bool:
    """显式探测入口: i0e / 递推链是否支持二阶反向 (结果缓存)。

    由测试或显式调用触发; import 本模块零副作用 (R187-T1)。
    """
    global _DOUBLE_BACKWARD_CACHE
    if _DOUBLE_BACKWARD_CACHE is None:
        _DOUBLE_BACKWARD_CACHE = _probe_double_backward()
    return _DOUBLE_BACKWARD_CACHE


def __getattr__(name: str):
    # PEP 562: 保持 `from ._sphere import DOUBLE_BACKWARD_OK` 兼容,
    # 探测延迟到真正访问该属性时。
    if name == "DOUBLE_BACKWARD_OK":
        return verify_double_backward()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def grad_log_bingham_norm(lmbda_t):
    """d log c / dλ, 即 **-E[x_k²]** (Bingham 二阶矩恒等式)。返回同形状 tensor。"""
    if not (HAS_TORCH and torch.is_tensor(lmbda_t)):
        raise TypeError("grad_log_bingham_norm 仅支持 torch 输入 (需要 autograd)")
    lam = lmbda_t
    if lam.dtype != torch.float64:
        lam = lam.double()
    need = lam.requires_grad
    lam = lam.detach().requires_grad_(True)
    f = log_bingham_norm(lam)
    g = torch.autograd.grad(f, lam)[0]
    if not need:
        g = g.detach()
    return g


def hessian_of_scalar(grad_fn, x, h: float = 1e-5):
    """批量 Hessian: 对 (..., d) 的 x, 返回 (..., d, d)。

    优先走「对 autograd 梯度再做一次批量反向」(精确); 若算子不支持二阶反向
    (DOUBLE_BACKWARD_OK=False), 自动退化为**梯度的中心差分** (误差 ~h² + eps/h),
    并在返回值语义上保持一致 (调用方无需感知)。
    """
    if not (HAS_TORCH and torch.is_tensor(x)):
        raise TypeError("hessian_of_scalar 仅支持 torch 输入")
    d = x.shape[-1]
    orig_shape = x.shape[:-1]
    xf = x.detach().reshape(-1, d).double()

    if verify_double_backward():
        xg = xf.clone().requires_grad_(True)
        g = grad_fn(xg)                      # (Bsz, d)
        # 逐列反向: e_jᵀ·J 给出 Hessian 第 j 行。d 很小 (p-1 ≤ 3), 无需 vmap,
        # 也绕开 is_grads_batched 的 shape 语义坑。
        rows = []
        for j in range(d):
            e = torch.zeros_like(g)
            e[:, j] = 1.0
            (rj,) = torch.autograd.grad(g, xg, grad_outputs=e,
                                        retain_graph=(j < d - 1))
            rows.append(rj)
        H = torch.stack(rows, dim=-2)        # (Bsz, d, d)
        return 0.5 * (H + H.transpose(-1, -2)).reshape(orig_shape + (d, d))

    # 退化路径: 梯度的中心差分 (算子不支持二阶反向时自动启用)
    H = torch.empty(xf.shape[0], d, d, dtype=xf.dtype, device=xf.device)
    for j in range(d):
        xp = xf.clone()
        xm = xf.clone()
        xp[:, j] += h
        xm[:, j] -= h
        H[:, :, j] = (grad_fn(xp) - grad_fn(xm)) / (2.0 * h)
    return 0.5 * (H + H.transpose(-1, -2)).reshape(orig_shape + (d, d))
