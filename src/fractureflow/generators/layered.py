# -*- coding: utf-8 -*-
"""W99 机制 M2 · 层状/地层约束中心过程 (技术方案 §3-G4 机制2, 沉积岩型)。

模型: L 个等厚水平层铺满 z 轴, 盘中心在域内均匀 (层归属由 z 决定, 边际与均匀
过程一致); 每层组占比从 Dirichlet(alpha * 全局占比) 独立抽一次 —— 层间组构成
不同、池化边际期望仍为全局占比; 产状/半径边际由 MarginalSpec 转移。

RNG 消费次序 (固定): centers(x,y,z) -> 逐层 dirichlet(层升序) -> 逐层组指派
(层升序) -> normals(按组) -> radii。
"""

import numpy as np

from .base import MarginalSpec, assemble_catalog, check_n, assert_contained

_DIRICHLET_FLOOR = 1e-12  # 防零占比组件 (Generator.dirichlet 要求 alpha>0)


def generate_layered(spec: MarginalSpec, n_layers: int, dirichlet_alpha: float,
                     seed: int):
    """生成一个层状目录 (N 精确 = spec.n_pop)。

    n_layers:         水平等厚层数 L (校准自由参数);
    dirichlet_alpha:  层内组占比 Dirichlet 浓度 (校准自由参数; ->inf 退化为均匀)。
    """
    n = check_n(spec.n_pop)
    n_layers = int(n_layers)
    if n_layers <= 0:
        raise ValueError("layered: n_layers=%d 非法" % n_layers)
    if not (dirichlet_alpha > 0):
        raise ValueError("layered: dirichlet_alpha=%r 须为正" % dirichlet_alpha)
    rng = np.random.default_rng(seed)
    half = np.asarray(spec.domain, dtype=np.float64) / 2.0
    K = len(spec.proportions)

    centers = rng.uniform(-half, half, size=(n, 3))
    bounds = np.linspace(-half[2], half[2], n_layers + 1)
    layer_id = np.clip(np.searchsorted(bounds, centers[:, 2], side="right") - 1,
                       0, n_layers - 1)

    sets = np.empty(n, dtype=np.int64)
    layer_props = np.empty((n_layers, K), dtype=np.float64)
    for l in range(n_layers):
        p_l = rng.dirichlet(np.maximum(dirichlet_alpha * np.asarray(spec.proportions),
                                       _DIRICHLET_FLOOR))
        layer_props[l] = p_l
        sel = layer_id == l
        if np.any(sel):
            sets[sel] = rng.choice(K, size=int(sel.sum()), p=p_l)

    dfn = assemble_catalog(spec, centers, sets, rng)
    assert_contained(dfn.centers, spec.domain)

    # proportion_fidelity (M2 口径, prereg 澄清): 每层实现占比与该目录自抽层内占比之差
    # 在多项抽样噪声 5 sigma 内 (实现正确性检查, 非池化边际约束)
    for l in range(n_layers):
        sel = layer_id == l
        m = int(sel.sum())
        if m == 0:
            continue
        realized = np.bincount(dfn.sets[sel], minlength=K) / float(m)
        dev = float(np.max(np.abs(realized - layer_props[l])))
        sigma_max = float(np.sqrt(np.max(layer_props[l] * (1.0 - layer_props[l])) / m))
        if dev > 5.0 * sigma_max + 1e-12:
            raise ValueError("layered: 层 %d 实现组占比偏离自抽 spec 超 5sigma 多项噪声 "
                             "(dev=%.4g, proportion_fidelity pill, M2 口径)" % (l, dev))
    dfn.layer_props = layer_props  # 诊断随行 (非 DFNRealization 标准字段)
    return dfn
