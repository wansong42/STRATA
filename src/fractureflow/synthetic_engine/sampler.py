# -*- coding: utf-8 -*-
"""R183-T1 DFN 采样器（Baecher 盘 + vMF 组产状 + 截断 Pareto 盘径 + 泊松板域位置）。

锚纪律（prereg）：
  - 产状组心/离散度：BEISHAN 880 法向实测（anchor_beishan_orientation.json）；
  - 离散度→κ 数值标定（anchors.kappa_from_dispersion_deg）；
  - α：Loviisa 1.277（eval）/ U[1.296,1.345] 亚区实测值域（train）；
  - p32：PoC 设计值 0.5（BEISHAN PoC，非原位实测——prereg honesty_policy #1）；
  - 口径：P32 = Σπr²/V（与冻结件 dfn.compute_p32 逐字同式，锚脚本断言 PASS）。
Bingham 数值困难 → vMF 替代并如实标注（任务书授权；正式层 R185）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fractureflow.synthetic_engine.anchors import (
    kappa_from_dispersion_deg,
    sample_trunc_pareto,
    sample_vmf,
)


@dataclass
class SceneConfig:
    """场景与采样配置（全部可 JSON 化；params_hash 溯源）。"""

    scene_id: str
    domain: str  # "eval" | "train"
    seed: int
    res_px: int = 1024
    width_m: float = 20.0
    height_m: float = 20.0
    # 露头窗口姿态（世界系 X东 Y北 Z上）
    origin: tuple = (0.0, 0.0, 0.0)      # 窗口 (u=0,v=0) 角点
    u_axis: tuple = (1.0, 0.0, 0.0)      # 图像横向（米）
    v_axis: tuple = (0.0, 0.0, 1.0)      # 图像纵向（米，row 向下递减）
    n_out: tuple = (0.0, -1.0, 0.0)      # 表面外法向（朝相机）
    # DFN
    K: int = 4
    set_centers: Optional[list] = None   # (K,3) 单位法向（无向）
    set_disp_deg: Optional[list] = None  # (K,) arccos|cos| 目标离散度
    set_weights: Optional[list] = None   # (K,)
    alpha: float = 1.277
    r_min: float = 0.0535
    r_max: float = 10.0
    p32: float = 0.5
    slab_half: float = 10.0
    terzaghi_censor: float = 0.08
    # 基面 / 渲染
    amplitude: float = 0.03
    surface_lattice: int = 257
    # 域随机化四件（None → eval 温和默认；train 全开）
    rand: Optional[dict] = None

    def frame_vectors(self):
        eu = np.asarray(self.u_axis, dtype=np.float64)
        ev = np.asarray(self.v_axis, dtype=np.float64)
        no = np.asarray(self.n_out, dtype=np.float64)
        O = np.asarray(self.origin, dtype=np.float64)
        return O, eu / np.linalg.norm(eu), ev / np.linalg.norm(ev), no / np.linalg.norm(no)


@dataclass
class DFN:
    """Baecher 盘实现。"""

    centers: np.ndarray   # (N,3)
    normals: np.ndarray   # (N,3) 单位（无向）
    radii: np.ndarray     # (N,)
    set_ids: np.ndarray   # (N,)
    p32_realized: float
    domain_volume: float
    config: dict = field(repr=False, default_factory=dict)


def realized_p32(radii: np.ndarray, volume: float) -> float:
    """P32 = Σπr²/V —— 与冻结件 dfn.compute_p32 逐字同式（锚脚本断言 PASS）。"""
    if volume <= 0:
        return 0.0
    return float(np.sum(np.pi * np.asarray(radii) ** 2) / volume)


def sample_dfn(cfg: SceneConfig) -> DFN:
    rng = np.random.default_rng(cfg.seed)
    O, eu, ev, no = cfg.frame_vectors()

    K = int(cfg.K)
    centers = np.asarray(cfg.set_centers, dtype=np.float64)
    disp = np.asarray(cfg.set_disp_deg, dtype=np.float64)
    w = np.asarray(cfg.set_weights, dtype=np.float64)
    w = w / w.sum()

    # 组采样数（多项式）→ 盘径 → 面积期望 → 盘总数（p32 闭环）
    n_pre = 200_000
    r_pre = sample_trunc_pareto(rng, n_pre, cfg.r_min, cfg.r_max, cfg.alpha)
    mean_area = float(np.mean(np.pi * r_pre**2))
    volume = cfg.width_m * cfg.height_m * (2.0 * cfg.slab_half)
    n_total = int(round(cfg.p32 * volume / mean_area))
    n_total = max(n_total, 1)
    set_ids = rng.choice(K, size=n_total, p=w)
    radii = sample_trunc_pareto(rng, n_total, cfg.r_min, cfg.r_max, cfg.alpha)

    # 位置：板域均匀（窗口内 u,v ∈ [0,W]×[0,H]，法向偏移 ∈ [-slab_half, +slab_half]）
    uu = rng.random(n_total) * cfg.width_m
    vv = rng.random(n_total) * cfg.height_m
    off = (rng.random(n_total) * 2.0 - 1.0) * cfg.slab_half
    centers3d = O + uu[:, None] * eu[None, :] + vv[:, None] * ev[None, :] - off[:, None] * no[None, :]

    # 产状：vMF（无向法向：κ 标定于 arccos|cos|）
    normals = np.empty((n_total, 3), dtype=np.float64)
    kap = [kappa_from_dispersion_deg(float(d)) for d in disp]
    for k in range(K):
        m = set_ids == k
        if not m.any():
            continue
        normals[m] = sample_vmf(rng, centers[k], kap[k], int(m.sum()))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    return DFN(
        centers=centers3d,
        normals=normals,
        radii=radii,
        set_ids=set_ids,
        p32_realized=realized_p32(radii, volume),
        domain_volume=float(volume),
        config={
            "kappa_used": kap,
            "n_disks": int(n_total),
            "mean_area_pre": mean_area,
            "volume_m3": volume,
        },
    )
