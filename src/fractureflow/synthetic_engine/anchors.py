# -*- coding: utf-8 -*-
"""R183 锚包与数值标定（T1 支撑）。

一切科学锚出自 results/r183_synthetic_engine/anchors.json（prereg 先行，
来源 r176_loviisa / r161_gdr / mine_viz PoC / 本任务 BEISHAN+Åland 现算）。
RANDOMIZATION_PARAM 与科学锚严格分账（prereg honesty_policy #3）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANCHOR_PATH = _ROOT / "results" / "r183_synthetic_engine" / "anchors.json"


@dataclass
class AnchorPack:
    """编译锚包（全部带 verification_status，禁改数值）。"""

    loviisa_alpha_all: float
    loviisa_alpha_subsite_range: list
    loviisa_median_m: float
    loviisa_k4_azimuth_deg: list
    forge_disp_deg: list
    beishan_p32_design: float
    beishan_p32_poc_realized: float
    beishan_sets: list  # anchor_beishan_orientation.json 的 sets
    aland_alpha: float
    raw: dict = field(repr=False, default_factory=dict)


def load_anchor_pack(path: Path | str | None = None) -> AnchorPack:
    p = Path(path) if path else DEFAULT_ANCHOR_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(p.parent / "anchor_beishan_orientation.json", "r", encoding="utf-8") as f:
        bs = json.load(f)
    aland = raw.get("aland") or {}
    return AnchorPack(
        loviisa_alpha_all=float(raw["loviisa"]["alpha_all_merged"]),
        loviisa_alpha_subsite_range=[float(x) for x in raw["loviisa"]["alpha_subsite_range"]],
        loviisa_median_m=float(raw["loviisa"]["trace_median_m"]),
        loviisa_k4_azimuth_deg=[float(x) for x in raw["loviisa"]["k4_azimuth_deg"]],
        forge_disp_deg=[float(x) for x in raw["forge_gdr1750"]["set_dispersion_deg"]],
        beishan_p32_design=float(raw["beishan"]["p32_design"]),
        beishan_p32_poc_realized=float(raw["beishan"]["p32_poc_realized"]),
        beishan_sets=bs["sets"],
        aland_alpha=float(aland["alpha_20m"]) if aland.get("alpha_20m") is not None else float("nan"),
        raw={"anchors": raw, "beishan_orientation": bs},
    )


# ---------------------------------------------------------------- vMF κ 标定
@lru_cache(maxsize=256)
def _fisher_mean_angle_rad(kappa: float, mc_seed: int = 7, n_mc: int = 200_000) -> float:
    """Fisher/vMF 极角期望 E[θ]（arccos|cos| 同义：无向法向下 θ∈[0,π/2]）。

    MC 固定 seed（缓存复用），保证标定确定性。κ→0 时 E[θ]→π/4（均匀球面的
    无向平均极角，数值验证用）。
    """
    rng = np.random.default_rng(mc_seed)
    if kappa < 1e-10:
        u = rng.random(n_mc)
        cos_t = 1.0 + 2.0 * u  # 均匀球面 cosθ∈[-1,1]
    else:
        u = rng.random(n_mc)
        # 逆变换采样：cosθ = 1 + ln(1 - u(1-e^{-2κ}))/κ
        cos_t = 1.0 + np.log1p(-u * (1.0 - np.exp(-2.0 * kappa))) / kappa
        cos_t = np.clip(cos_t, -1.0, 1.0)
    ang = np.arccos(np.abs(cos_t))
    return float(np.mean(ang))


def kappa_from_dispersion_deg(disp_deg: float, tol: float = 1e-4) -> float:
    """由目标离散度（arccos|cos| 均值，度）数值反解 vMF κ（二分）。

    E[θ](κ) 单调递减：κ→0 为 ~45°（无向均匀），κ→∞ 为 0。
    """
    target = np.radians(float(disp_deg))
    lo, hi = 1e-8, 1e5
    if _fisher_mean_angle_rad(lo) < target:  # 已比目标更散（理论不达）
        return lo
    if _fisher_mean_angle_rad(hi) > target:
        return hi
    for _ in range(80):
        mid = np.sqrt(lo * hi)  # 对数中点（κ 跨数量级）
        if _fisher_mean_angle_rad(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < tol:
            break
    return float(np.sqrt(lo * hi))


def xmin_from_median(median_m: float, alpha: float) -> float:
    """截断前 Pareto 中位数反推 xmin：median = xmin·2^(1/(α-1))。

    DERIVED_FROM_MEASURED（Loviisa 实测 α 与中位长）。DERIVATION 标注随锚走。
    """
    if not (alpha > 1.0):
        raise ValueError("alpha must be > 1")
    return float(median_m / 2.0 ** (1.0 / (alpha - 1.0)))


def sample_trunc_pareto(rng: np.random.Generator, n: int, xmin: float, xmax: float, alpha: float):
    """截断 Pareto [xmin, xmax] 精确采样（逆 CDF，截断归一化）。"""
    if not (xmax > xmin > 0 and alpha > 1.0):
        raise ValueError("bad trunc-pareto params")
    u = rng.random(n)
    p = 1.0 - u * (1.0 - (xmin / xmax) ** (alpha - 1.0))  # P(X>x)/(P(xmin)) 形式
    return xmin * p ** (-1.0 / (alpha - 1.0))


def fit_alpha_truncated(samples: np.ndarray, lo: float, hi: float) -> float:
    """截断 Pareto MLE：ln L = n ln(α-1) - αΣln x - n ln(lo^{1-α} - hi^{1-α})。

    对 dlnL/dα 用 brentq 求根。samples 全部落在 [lo, hi]。
    """
    from scipy.optimize import brentq

    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    s = float(np.log(x).sum())
    c_lo, c_hi = float(np.log(lo)), float(np.log(hi))

    def dll(alpha: float) -> float:
        em = np.exp(-(alpha - 1.0) * c_lo) - np.exp(-(alpha - 1.0) * c_hi)
        d_em = -c_lo * np.exp(-(alpha - 1.0) * c_lo) + c_hi * np.exp(-(alpha - 1.0) * c_hi)
        return n / (alpha - 1.0) - s - n * d_em / em

    a_lo, a_hi = 1.0 + 1e-6, 10.0
    f_lo, f_hi = dll(a_lo), dll(a_hi)
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(dll, a_lo, a_hi, xtol=1e-8))


def sample_vmf(rng: np.random.Generator, mu: np.ndarray, kappa: float, size: int):
    """3D vMF 精确采样（逆 CDF，零拒绝）。mu 单位向量；返回 (size,3)。"""
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / (np.linalg.norm(mu) + 1e-12)
    if kappa < 1e-10:
        v = rng.normal(size=(size, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v
    u = rng.random(size)
    cos_t = 1.0 + np.log1p(-u * (1.0 - np.exp(-2.0 * kappa))) / kappa
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    phi = 2.0 * np.pi * rng.random(size)
    # 局部正交基
    ref = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(mu, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(mu, e1)
    return (
        cos_t[:, None] * mu[None, :]
        + sin_t[:, None] * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])
    )
