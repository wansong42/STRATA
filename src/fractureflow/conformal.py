# -*- coding: utf-8 -*-
"""R210 · 球面共形不确定度原语（支柱二实现层）。

设计来源（**verbatim 消费，本层不得改判据**）：
  - `docs/R205_球面共形设计稿.md`（§1 复用审计 / §3 三分数 / §4 加权协议 / §5 数学核）
  - `results/r205_conformal_design/r205_prereg_draft.json`（R205 预注册草案 = 判据来源）
  - `results/r201_identifiability_theory/formalization.md`（记号 v1.0：`d(u,v)=acos|⟨u,v⟩|`）
  - `results/r210_conformal_impl/prereg.json`（本单**实现读法裁定**，落盘早于全部结果件）

本模块提供四类**新增原语**（`orthostats/_sphere.py` 无 conformal / quantile / cone / cover）：

  1. **加权共形分位**：自归一化加权经验分位，阈值口径取 `k/(n+1)`（`k=⌈(1-α)(n+1)⌉`）——
     等权时**逐位退化为教科书 split-conformal 分位** `s_(k)`（见 `weighted_quantile` 注）。
  2. **对径球面帽**：产状是轴数据（`n ≡ -n`），帽 `{u: acos|⟨u,c⟩| ≤ ρ}` 天然对径，
     与项目「法向无向、±n 同面」铁律一致（AGENTS 路线 A/B 关键 bug 修复）。
  3. **帽集合的球面测度 / 并集 / 冗余度**：固定 Fibonacci 格点（**无 rng，确定性**），
     逐位可复。
  4. **三分数可算面 + 三协议构造**：S1 逐点帽 / S2 组心帽 / S3 conformal K-set；
     P-WELL 留组块 / P-STRAT Mondrian / P-COV 协变量倾斜。

三条铁律（越界即停手上报）：
  - 只消费 `nrm_full[hid]` 形式的**点级**真值；组系真值仅用于 TRUTH 侧评测，**不进校准**；
  - 三分数只输出**集合/帽**，绝不输出点精度（分级表 §0 规则 6 / §3 规则 3）；
  - CIRCULAR 侧禁覆盖率字段（R205 §6 毒丸 G4）。本模块的 `cap_*` 一律返回
    **球面测度 / 宽度**（不需真值即可算的量），结构上不能被冒充成覆盖率。

权重与 ESS 直接复用 `orthostats.terzaghi_layer`（`terzaghi_weights_t` / `terzaghi_ess`），
Fréchet 中位数直接复用 `inference._l1_median_batch`（**禁重实现**，与现役后端逐位同件）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .orthostats.terzaghi_layer import (
    terzaghi_ess,
    terzaghi_weights_t,
    W_HI_DEFAULT,
    W_LO_DEFAULT,
)

__all__ = [
    # 球面/角度
    "unit",
    "ang_deg",
    "cap_contains",
    "cap_mass",
    "fib_sphere",
    "cap_set_stats",
    # 共形分位
    "conformal_k",
    "conformal_level",
    "weighted_quantile",
    "weighted_coverage",
    "family_alpha",
    # 权重与自洽
    "ess_frac",
    "truncation_rate",
    # 分数面
    "frechet_median",
    "mean_member_angle",
    "phi_bic",
    "k_set",
    # 协议
    "lowo_folds",
    "stratum_by_nearest",
    "tilt_weights",
    # 常量
    "ALPHA_DEFAULT",
    "CAP_GRID_DEFAULT",
]

ALPHA_DEFAULT = 0.10                 # 名义 1-α = 0.90（R205 prereg `nominal`）
CAP_GRID_DEFAULT = 2048              # Fibonacci 格点数（确定性，非随机）


# ---------------------------------------------------------------------------
# 0. 基础：单位化 / 无向角距
# ---------------------------------------------------------------------------

def unit(v, axis: int = -1) -> np.ndarray:
    """float64 单位化（零向量以 1e-12 兜底，与项目其余模块同约定）。"""
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.where(n < 1e-12, 1e-12, n)


def ang_deg(a, b) -> np.ndarray:
    """**无向**角距 `d(a,b)=acos|⟨a,b⟩|`（度），全项目唯一口径（R201 §1 度量裁定）。

    支持广播；输出 float64。`|cos|` 先 clip 到 [0,1] 再 arccos（防 1+1e-16 触发 NaN）。
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    c = np.abs((aa * bb).sum(axis=-1))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# 1. 对径球面帽
# ---------------------------------------------------------------------------

def cap_mass(rho_deg) -> np.ndarray:
    """对径帽的**均匀球面测度**（= 均匀随机轴上落入帽内的概率）。

    `{u ∈ S²: acos|⟨u,c⟩| ≤ ρ}` 是绕 `±c` 的两个测地半径 ρ 的普通帽，
    面积 `4π(1-cos ρ)`，故 `P = 1 - cos ρ`（ρ=90° ⇒ 1，全空间）。
    ρ 超过 90° 时帽已是全空间，返回 1。
    """
    r = np.radians(np.asarray(rho_deg, dtype=np.float64))
    return np.clip(1.0 - np.cos(r), 0.0, 1.0)


def cap_contains(center, rho_deg, u) -> np.ndarray:
    """`u`（(M,3) 或 (3,)）是否落入中心 `center`、半角 `rho_deg` 的对径帽。"""
    return ang_deg(u, center) <= float(rho_deg)


def fib_sphere(m: int = CAP_GRID_DEFAULT) -> np.ndarray:
    """Fibonacci 球面格点 (m,3) —— **确定性**（无 rng），供测度/重叠度计算。

    用 `i+0.5` 偏移而非 `i`：避免极点重复，且 m 固定时结果逐位可复。
    """
    m = int(m)
    if m < 2:
        raise ValueError("fib_sphere: m 需 >= 2, got %r" % (m,))
    i = np.arange(m, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / m
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def cap_set_stats(centers, rhos, m: int = CAP_GRID_DEFAULT) -> Dict[str, float]:
    """帽集合的球面测度统计（**不需真值**，CIRCULAR 可用；**不是覆盖率**）。

    返回：
      n_caps        帽数
      mean_mass     平均帽测度 `mean(1-cos ρ)`
      union_mass    并集测度（格点上被至少一帽覆盖的比例）
      redundancy    `(Σ_i mass_i - union_mass) / Σ_i mass_i`（重叠冗余度；0=完全无重叠）
      overlap_mass  格点上被 **≥2** 顶帽覆盖的比例（“帽间重叠率”）
      max_mult      格点上最大覆盖重数
    """
    c = unit(np.atleast_2d(np.asarray(centers, dtype=np.float64)))
    r = np.asarray(rhos, dtype=np.float64).reshape(-1)
    if c.shape[0] != r.shape[0]:
        raise ValueError("cap_set_stats: centers %d != rhos %d" % (c.shape[0], r.shape[0]))
    if c.shape[0] == 0:
        raise ValueError("cap_set_stats: 空帽集合，拒绝静默返回")
    g = fib_sphere(m)
    # (m, n) 角距矩阵；m=2048, n~700 → 11 MB float64，可接受
    ang = np.degrees(np.arccos(np.clip(np.abs(g @ c.T), 0.0, 1.0)))
    inside = ang <= r[None, :]
    mult = inside.sum(axis=1)
    masses = cap_mass(r)
    union = float(np.mean(mult > 0))
    total = float(masses.sum())
    return {
        "n_caps": int(c.shape[0]),
        "mean_mass": float(masses.mean()),
        "mass_median": float(np.median(masses)),
        "mass_p90": float(np.percentile(masses, 90.0)),
        "union_mass": union,
        "redundancy": float((total - union) / total) if total > 0 else 0.0,
        "overlap_mass": float(np.mean(mult >= 2)),
        "max_mult": int(mult.max()),
        "grid_m": int(m),
    }


# ---------------------------------------------------------------------------
# 2. 共形分位（含加权）
# ---------------------------------------------------------------------------

def conformal_k(n: int, alpha: float = ALPHA_DEFAULT) -> int:
    """`k = ⌈(1-α)(n+1)⌉`（R205 prereg `conformal_quantile_order`）。`k > n` ⇒ 半径 +∞。"""
    n = int(n)
    if n < 1:
        raise ValueError("conformal_k: n 需 >= 1, got %r" % (n,))
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("conformal_k: alpha 需在 (0,1), got %r" % (alpha,))
    return int(math.ceil((1.0 - float(alpha)) * (n + 1)))


def conformal_level(n: int, alpha: float = ALPHA_DEFAULT) -> float:
    """加权分位使用的**累积质量阈值** `k/(n+1)` ∈ (0,1)。

    等权时它精确复现 split-conformal：`#{s_i ≤ ρ}/n ≥ k/(n+1)` ⟺ `#{s_i ≤ ρ} ≥ k`
    （因 `n·k/(n+1) = k - k/(n+1) > k-1`），故 `ρ = s_(k)` 逐位一致。
    """
    n = int(n)
    return conformal_k(n, alpha) / (n + 1.0)


def weighted_quantile(scores, weights=None, alpha: float = ALPHA_DEFAULT,
                      level: Optional[float] = None,
                      level_mode: str = "conformal") -> Tuple[float, dict]:
    """自归一化**加权**共形分位（R205 §5.2）。

    ρ = min{ ρ' : Σ_i w_i·1[s_i ≤ ρ'] / Σ_i w_i ≥ L }，`L` 默认取共形阈值 `k/(n+1)`。

    level_mode:
      "conformal"  默认。`L = ⌈(1-α)(n+1)⌉/(n+1)` —— 等权时逐位等于 `s_(k)`，
                   有限样本界 `1-α ≤ P(s ≤ ρ) ≤ 1-α + 1/(n+1)` 成立（R205 §5.1 [PROVED]）；
                   加权时该界退化为 [SKETCH]（Tibshirani et al. 2019 Thm 1，需 w 有界）。
      "ratio"      `L = 1-α` —— 纯加权分位（无 +1/(n+1) 修正），仅作敏感性对照，
                   **不作主口径**（它丢掉有限样本修正，等权时也不等于 conformal 分位）。

    返回 `(rho, info)`；`rho = +inf` 且 `info["degenerate"]=True` 表示 `k > n`。
    权重不允许 NaN/非有限/非正（**拒绝静默降级**）；全部为 0 同样拒绝。
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(s.size)
    if n < 1:
        raise ValueError("weighted_quantile: 空分数，拒绝静默返回")
    if not np.isfinite(s).all():
        raise ValueError("weighted_quantile: 分数含 NaN/Inf，拒绝消费")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError("weighted_quantile: weights %d != scores %d" % (w.size, n))
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weighted_quantile: 权重含负值/非有限值，拒绝消费")
        if float(w.sum()) <= 0.0:
            raise ValueError("weighted_quantile: 权重全零，拒绝消费")

    if level_mode not in ("conformal", "ratio"):
        raise ValueError("weighted_quantile: level_mode 需 conformal/ratio, got %r" % level_mode)
    if level is None:
        L = conformal_level(n, alpha) if level_mode == "conformal" else (1.0 - float(alpha))
    else:
        L = float(level)

    k = conformal_k(n, alpha)
    info = {"n": n, "k": k, "level_used": float(L), "level_mode": level_mode,
            "n_eff": float(terzaghi_ess(w)), "degenerate": bool(k > n),
            "w_min": float(w.min()), "w_max": float(w.max())}
    if k > n:
        return float("inf"), info

    order = np.argsort(s, kind="stable")
    s_s, w_s = s[order], w[order]
    cum = np.cumsum(w_s)
    W = float(cum[-1])
    idx = int(np.searchsorted(cum, L * W, side="left"))
    idx = min(idx, n - 1)
    return float(s_s[idx]), info


def weighted_coverage(covered, weights=None) -> float:
    """加权经验覆盖（R205 §5.2 目标测度 `P` 侧的读数）。`covered` 为 bool 数组。"""
    c = np.asarray(covered)
    if c.size == 0:
        raise ValueError("weighted_coverage: 空覆盖数组")
    if weights is None:
        return float(np.mean(c))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != c.size:
        raise ValueError("weighted_coverage: weights %d != covered %d" % (w.size, c.size))
    tot = float(w.sum())
    if tot <= 0:
        raise ValueError("weighted_coverage: 权重全零")
    return float((w * c).sum() / tot)


def family_alpha(alpha_fam: float, G: int, method: str = "bonferroni") -> float:
    """家族错误率 → 单组 α（R205 §5.5）。

    "bonferroni": `α_fam/G`  —— [PROVED]（无需成员独立，安全默认）。
    "sidak":      `1-(1-α_fam)^(1/G)` —— [SKETCH]（需近独立，仅作对照）。
    """
    G = int(G)
    if G < 1:
        raise ValueError("family_alpha: G 需 >= 1, got %r" % (G,))
    if method == "bonferroni":
        return float(alpha_fam) / G
    if method == "sidak":
        return float(1.0 - (1.0 - float(alpha_fam)) ** (1.0 / G))
    raise ValueError("family_alpha: method 需 bonferroni/sidak, got %r" % (method,))


# ---------------------------------------------------------------------------
# 3. 权重自洽量（ESS / 截断率）
# ---------------------------------------------------------------------------

def ess_frac(weights) -> float:
    """`ESS/N`（terzaghi_layer docstring：< 0.5 该 Terzaghi 校正基本不值得信）。"""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("ess_frac: 空权重")
    return float(terzaghi_ess(w) / w.size)


def truncation_rate(normals, axis=None) -> Dict[str, float]:
    """Terzaghi soft_clamp `[0.1, 5.0]` 的**截断率**（R205 §4.2 的必报自洽量）。

    诊断量（**不使用**）：按 `terzaghi_layer` 同一公式先算自归一化前的
    `w_raw = (1/|n·a|) / mean(1/|n·a|)`，再统计其越出 `[W_LO, W_HI]` 的比例。
    该量纯粹刻画「近水平面 vs 垂井」的倾斜不足面，不参与任何半径计算。
    """
    n = unit(np.atleast_2d(np.asarray(normals, dtype=np.float64)))
    if n.shape[0] == 0:
        raise ValueError("truncation_rate: 空法向")
    a = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    cos_wa = np.sqrt((n @ a) ** 2 + 1e-18)
    raw = 1.0 / np.clip(cos_wa, 1e-6, None)
    raw = raw / max(float(raw.mean()), 1e-12)
    lo = float((raw < W_LO_DEFAULT).mean())
    hi = float((raw > W_HI_DEFAULT).mean())
    return {"w_lo": W_LO_DEFAULT, "w_hi": W_HI_DEFAULT,
            "rate_below_lo": lo, "rate_above_hi": hi, "rate_truncated": lo + hi,
            "mean_abs_na": float(np.abs(n @ a).mean())}


# ---------------------------------------------------------------------------
# 4. 分数面
# ---------------------------------------------------------------------------

def frechet_median(pts, weights=None, iters: int = 40) -> np.ndarray:
    """加权球面射影 L1（Fréchet）中位数 —— **复用现役实现，禁重实现**。

    `inference._l1_median_batch` 是 `l1_local` 后端的同一件（R108 落地），
    等变（欧氏等距下共变）且与 `acos|⟨·,·⟩|` 指标同构，故本层直接消费。
    """
    from .inference import _l1_median_batch

    p = unit(np.atleast_2d(np.asarray(pts, dtype=np.float64)))
    if p.shape[0] == 0:
        raise ValueError("frechet_median: 空点集")
    w = np.ones((1, p.shape[0]), dtype=np.float64) if weights is None else \
        np.asarray(weights, dtype=np.float64).reshape(1, -1)
    if w.shape[1] != p.shape[0]:
        raise ValueError("frechet_median: weights %d != pts %d" % (w.shape[1], p.shape[0]))
    S = (w.reshape(-1)[:, None] * p).T @ p
    v0 = np.linalg.eigh(S)[1][:, -1][None]
    v = _l1_median_batch(p, w, v0, iters=iters)
    return unit(v)[0]


def mean_member_angle(pts, centers, assign, weights=None) -> np.ndarray:
    """逐组成员到其**指派中心**的平均无向角距（S2 的核形式，度）。

    理论依据（R205 §5.5 的加强版）：「均值的距离 ≤ 距离的均值」——对测地凸球冠内
    的 Fréchet 均值，`d(mean{a_i}, mean{b_i}) ≤ Σ w_i d(a_i,b_i)/Σ w_i`。
    故把**逐组成员平均残差**当作组心误差的**保守代理**是有据的（本单在 TRUTH 上
    实测该不等式，违反率如实上报，不假设成立）。
    """
    p = unit(np.atleast_2d(np.asarray(pts, dtype=np.float64)))
    c = unit(np.atleast_2d(np.asarray(centers, dtype=np.float64)))
    a = np.asarray(assign, dtype=int).reshape(-1)
    w = np.ones(p.shape[0], dtype=np.float64) if weights is None else \
        np.asarray(weights, dtype=np.float64).reshape(-1)
    out = np.full(c.shape[0], np.nan)
    for k in range(c.shape[0]):
        sel = a == k
        if not sel.any():
            continue
        d = ang_deg(p[sel], c[k])
        out[k] = float((w[sel] * d).sum() / max(float(w[sel].sum()), 1e-300))
    return out


def phi_bic(nrm_obs, Ks: Sequence[int], weights=None, seed: int = 42) -> Dict[str, float]:
    """S3 的**不依赖真值**拟合优度 `φ(K)`（R205 §3.3，重标 BIC 形式）。

    在**观测行**（真值已知，故 φ 可算）上做球面 k-means（复用 `setlabel.spherical_kmeans`），
    取加权角残差平方和 SSE(K)，令

        φ(K) = n·ln(SSE(K)/n) + 2K·ln(n)          （2K = 每组心 2 个自由角参数）

    越小越好。K=1 时 SSE 用全局 Fréchet 中心（退化基准）。返回 `{str(K): φ}`。
    """
    from .setlabel import spherical_kmeans

    x = unit(np.atleast_2d(np.asarray(nrm_obs, dtype=np.float64)))
    n = int(x.shape[0])
    if n < 3:
        raise ValueError("phi_bic: 观测行不足 3，φ(K) 无定义")
    w = np.ones(n, dtype=np.float64) if weights is None else \
        np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n:
        raise ValueError("phi_bic: weights %d != 观测行 %d" % (w.size, n))
    out: Dict[str, float] = {}
    for K in Ks:
        Ke = int(min(int(K), n))
        if Ke <= 1:
            c = frechet_median(x, w)[None]
            a = np.zeros(n, dtype=int)
        else:
            c, a = spherical_kmeans(x, Ke, seed=int(seed))
        d = ang_deg(x, c[a])
        sse = float((w * d * d).sum() / max(float(w.sum()), 1e-300))
        sse = max(sse, 1e-12)
        out[str(int(K))] = float(n * math.log(sse / n) + 2.0 * Ke * math.log(n))
    return out


def k_set(phi_row: Dict[str, float], rho: float) -> List[int]:
    """`C(K) = {K : φ(K) ≤ ρ}`（R205 §3.3；ρ 为 TRUTH 侧校准出的分位阈）。"""
    if not np.isfinite(rho):
        return sorted(int(k) for k in phi_row)
    return sorted(int(k) for k, v in phi_row.items() if float(v) <= float(rho))


# ---------------------------------------------------------------------------
# 5. 交换性协议构造
# ---------------------------------------------------------------------------

def lowo_folds(block_ids) -> List[Tuple[np.ndarray, np.ndarray]]:
    """留一**组块**（leave-one-well-out）折：返回 `[(cal_idx, test_idx), ...]`。

    组块定义见 R205 §2.2/§4.5：R207 聚合目录的组块 = 反解的 36 口实井；
    DECOVALEX 的组块 = `fracture_id`（同裂隙多顶点法向逐位相同，是天然不可交换单元）。
    `cal_idx ∩ test_idx = ∅`（A7 硬令），且并集 = 全集。
    """
    b = np.asarray(block_ids).reshape(-1)
    if b.size == 0:
        raise ValueError("lowo_folds: 空组块标签")
    out = []
    for u in np.unique(b):
        test = np.where(b == u)[0]
        cal = np.where(b != u)[0]
        if cal.size == 0:
            continue                      # 单块数据无多余块可校准，跳过并如实报数
        out.append((cal, test))
    return out


def stratum_by_nearest(dirs, centers) -> np.ndarray:
    """把方向指派到最近层心（无向 `|cos|`，与 `spherical_kmeans` 同口径）。"""
    d = unit(np.atleast_2d(np.asarray(dirs, dtype=np.float64)))
    c = unit(np.atleast_2d(np.asarray(centers, dtype=np.float64)))
    return np.abs(d @ c.T).argmax(axis=1).astype(int)


def tilt_weights(cov_cal, cov_test, bw: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """P-COV 的**协变量倾斜**权重（R205 §2.3，仅作敏感性对照）。

    `g(w) = exp(-||z_w - z_test||² / (2·bw²))`，`z` = 井级可观测量（标准化）。
    `z` **只允许**含观测侧可算量（观测法向分布形状 / 观测条数 / 井轴），禁含真值。
    bw 默认取「中位启发式」：全体井对距离中位（无真值可依，故为工程默认，
    其误差由 TRUTH 靶场实测，**不假装有免估红利**）。
    返回 `(g, info)`，`g` 已归一化到均值 1。
    """
    zc = np.atleast_2d(np.asarray(cov_cal, dtype=np.float64))
    zt = np.asarray(cov_test, dtype=np.float64).reshape(-1)
    if zc.size == 0:
        raise ValueError("tilt_weights: 空协变量")
    if not (np.isfinite(zc).all() and np.isfinite(zt).all()):
        raise ValueError("tilt_weights: 协变量含非有限值，拒绝消费")
    if bw is None:
        d = np.linalg.norm(zc[:, None, :] - zc[None, :, :], axis=-1)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        bw = max(med, 1e-6)
    bw = float(bw)
    dd = np.linalg.norm(zc - zt[None, :], axis=1)
    g = np.exp(-(dd * dd) / (2.0 * bw * bw))
    s = float(g.sum())
    if s <= 0:
        raise ValueError("tilt_weights: 核权重全零（bw 过小）")
    g = g / (s / g.size)
    return g, {"bw": bw, "bandwidth_rule": "median heuristic (工程默认, 无真值可依)"}
