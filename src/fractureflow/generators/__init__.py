# -*- coding: utf-8 -*-
"""W99 替代 DFN 生成机制包 (WP-4 前置)。

三套替代机制 (技术方案 §3-G4 / wp_register#WP-4.mechanisms):
  generate_neyman_scott  M1 簇状中心过程 (Neyman-Scott/Thomas) —— 优先级最高
  generate_layered       M2 层状/地层约束
  generate_fault_coupled M3 尺寸-产状耦合 + 显式大断裂对象

公共底座 base.MarginalSpec / fit_marginals: 边际转移校准的 API 载体 —— 生成器
只吃边际规格 (逐组经验法向池, amendment 0a), 不吃任何目标中心数组 (spec_guard),
匹配只能经由观测摘要统计实现。禁改 dfn.py 既有行为; 只复用其公开容器 DFNRealization。
"""

from .base import (MarginalSpec, fit_marginals, fit_vmf_kappa,
                   sample_normals_by_set, assemble_catalog, check_n,
                   check_proportions_iid, assert_contained, N_MAX)
from .cluster import generate_neyman_scott
from .layered import generate_layered
from .fault_coupled import generate_fault_coupled

__version__ = "w99-v1.1-amendment0a"

__all__ = [
    "MarginalSpec", "fit_marginals", "fit_vmf_kappa",
    "sample_normals_by_set", "assemble_catalog", "check_n",
    "check_proportions_iid", "assert_contained", "N_MAX",
    "generate_neyman_scott", "generate_layered", "generate_fault_coupled",
    "__version__",
]
