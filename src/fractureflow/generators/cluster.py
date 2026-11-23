# -*- coding: utf-8 -*-
"""W99 机制 M1 · Neyman-Scott/Thomas 簇状中心过程 (技术方案 §3-G4 优先级最高)。

模型: 均匀撒 P 个父点 -> N 个后代各自独立指派到某父点 (离散均匀) 并以
N(0, sigma_c^2 I3) 偏移散布, 域外拒绝重抽; 产状/半径边际由 MarginalSpec 转移
(只换中心采样 —— G4 机制1 原文)。

RNG 消费次序 (固定, determinism pill 依据):
  parents -> parent_idx -> offsets -> (域外拒绝重抽轮) -> sets -> normals -> radii
"""

import numpy as np

from .base import (MarginalSpec, assemble_catalog, check_n, check_proportions_iid,
                   assert_contained)

_MAX_REDRAW_ROUNDS = 200


def generate_neyman_scott(spec: MarginalSpec, n_parents: int, sigma_c: float,
                          seed: int, enforce_proportion_pill: bool = True):
    """生成一个簇状中心过程目录 (N 精确 = spec.n_pop)。

    n_parents: 父点数 P (校准自由参数);
    sigma_c:   高斯核带宽 m (校准自由参数)。
    """
    n = check_n(spec.n_pop)
    n_parents = int(n_parents)
    if n_parents <= 0 or n_parents > n:
        raise ValueError("neyman_scott: n_parents=%d 非法 (须 1..N)" % n_parents)
    if not (sigma_c > 0):
        raise ValueError("neyman_scott: sigma_c=%r 须为正" % sigma_c)
    rng = np.random.default_rng(seed)
    half = np.asarray(spec.domain, dtype=np.float64) / 2.0

    parents = rng.uniform(-half, half, size=(n_parents, 3))
    parent_idx = rng.integers(0, n_parents, size=n)
    centers = parents[parent_idx] + rng.normal(0.0, float(sigma_c), size=(n, 3))
    for _ in range(_MAX_REDRAW_ROUNDS):
        out = np.any((centers < -half) | (centers > half), axis=1)
        if not np.any(out):
            break
        centers[out] = parents[parent_idx[out]] + rng.normal(0.0, float(sigma_c),
                                                             size=(int(out.sum()), 3))
    else:
        raise RuntimeError("neyman_scott: 域内拒绝重抽 %d 轮未收敛 (sigma_c=%.3g)"
                           % (_MAX_REDRAW_ROUNDS, sigma_c))

    sets = rng.choice(len(spec.proportions), size=n, p=spec.proportions)
    dfn = assemble_catalog(spec, centers, sets, rng)
    assert_contained(dfn.centers, spec.domain)
    if enforce_proportion_pill:
        check_proportions_iid(spec, dfn.sets)
    return dfn
