# -*- coding: utf-8 -*-
"""W99 机制 M3 · 尺寸-产状耦合 + 显式大断裂对象 (技术方案 §3-G4 机制3, 打 H2 稳健性)。

模型: n_fault = round(phi*N) 个确定性个数大盘 r ~ Uniform(0.75, 1.0)*r_max_meta;
断裂盘法向取自该组经验池的收紧子池 (w = n*mu >= 组内 w 的 w_q 分位) —— 经验版
尺寸-浓度耦合 (amendment 0a: 原 vMF kappa_mult 旋钮随边际转移改经验池一并更换,
耦合语义不变: 越大越紧); 其余背景盘照 MarginalSpec 边际抽取。位置全部域内均匀。

RNG 消费次序 (固定): fault_radii -> fault_sets -> fault_centers -> bg_centers ->
bg_sets -> fault_normals(逐组收紧池) -> bg_normals(逐组池) -> bg_radii。
"""

import numpy as np

from ..dfn import DFNRealization
from .base import (MarginalSpec, check_n, check_proportions_iid, assert_contained,
                   sample_normals_by_set, _unit)


def tight_pool_mask(spec, w_q):
    """逐组收紧子池掩码: w = n*mu_k >= 组内 w 的 w_q 分位 (w_q=0 即全池)。"""
    if spec.vmf_mus is None:
        raise ValueError("fault_coupled: MarginalSpec 缺诊断 vmf_mus, 收紧耦合需要组均值方向")
    masks = []
    for k in range(spec.K):
        w = spec.normals_by_set[k] @ spec.vmf_mus[k]
        q = float(np.quantile(w, float(w_q))) if 0 < float(w_q) <= 1 else -np.inf
        masks.append(w >= q)
    return masks


def generate_fault_coupled(spec: MarginalSpec, fault_frac: float, w_q: float,
                           seed: int, enforce_proportion_pill: bool = True):
    """生成一个显式大断裂 + 尺寸-产状耦合目录 (N 精确 = spec.n_pop)。

    fault_frac: 断裂盘占比 phi (校准自由参数);
    w_q:        断裂盘组内法向收紧分位 (校准自由参数; 1=只取最贴组均值的方向)。
    """
    n = check_n(spec.n_pop)
    if not (0.0 < fault_frac < 1.0):
        raise ValueError("fault_coupled: fault_frac=%r 须在 (0,1)" % fault_frac)
    if not (0.0 <= w_q <= 1.0):
        raise ValueError("fault_coupled: w_q=%r 须在 [0,1]" % w_q)
    if spec.r_max_meta is None or not (spec.r_max_meta > 0):
        raise ValueError("fault_coupled: MarginalSpec.r_max_meta 缺失, "
                         "显式大断裂档需要站点 meta 的 dfn_params.r_max")
    rng = np.random.default_rng(seed)
    half = np.asarray(spec.domain, dtype=np.float64) / 2.0
    K = spec.K

    n_fault = int(round(fault_frac * n))
    n_bg = n - n_fault
    fault_radii = rng.uniform(0.75, 1.0, size=n_fault) * float(spec.r_max_meta)
    fault_sets = rng.choice(K, size=n_fault, p=spec.proportions)
    fault_centers = rng.uniform(-half, half, size=(n_fault, 3))
    bg_centers = rng.uniform(-half, half, size=(n_bg, 3))
    bg_sets = rng.choice(K, size=n_bg, p=spec.proportions)

    tight = tight_pool_mask(spec, w_q)
    fault_normals = np.empty((n_fault, 3), dtype=np.float64)
    for k in range(K):
        sel = fault_sets == k
        m = int(sel.sum())
        if m == 0:
            continue
        pool = spec.normals_by_set[k][tight[k]]
        if pool.shape[0] == 0:
            pool = spec.normals_by_set[k]
        fault_normals[sel] = pool[rng.integers(0, pool.shape[0], size=m)]
    bg_normals = sample_normals_by_set(spec, bg_sets, rng)

    centers = np.vstack([fault_centers, bg_centers])
    sets = np.concatenate([fault_sets, bg_sets]).astype(np.int64)
    radii = np.concatenate([fault_radii,
                            rng.choice(spec.radii_sample, size=n_bg, replace=True)])
    normals = _unit(np.vstack([fault_normals, bg_normals]))

    dfn = DFNRealization(centers=centers, normals=normals, radii=radii,
                         sets=sets, domain=tuple(spec.domain))
    assert_contained(dfn.centers, spec.domain)
    if enforce_proportion_pill:
        check_proportions_iid(spec, dfn.sets)
    dfn.n_fault = n_fault  # 诊断随行
    return dfn
