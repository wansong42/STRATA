# -*- coding: utf-8 -*-
"""R183 合成视觉引擎 v0（模块 C · Phase 0 脊柱）。

纯生成件：零训练、零下载、零 GPU 硬依赖（numpy/scipy/PIL）。
组件：
  anchors     实测锚包加载 + vMF κ 标定 + xmin 推导
  sampler     T1 DFN 采样器（vMF 组产状 + 截断 Pareto 盘半径 + 泊松板域位置）
  surface     fBm value-noise 基面（DEM tile 接口预留）
  intersector 平面×基面解析求交（逐顶点双线性权重 = 可微结构留线）
  renderer    正交光栅化 + 程序纹理 RGB + 域随机化四件（光照/噪声/雾度/植被）
  degrader    T3 碎片化降质四算子（断点/端蚀/漏检/粘连）→ GT/OBS 双契约
  g1          G1 等效对齐门（像素对齐 / 吞吐 / 交付集 / 采样保真）
  cli         python -m fractureflow.synthetic_engine ...
冻结件 dfn.py / site_model.py 只读 import（锚推导）；e3gt_v4.py 零触碰。
"""
from fractureflow.synthetic_engine.anchors import AnchorPack, load_anchor_pack  # noqa: F401
from fractureflow.synthetic_engine.sampler import SceneConfig, sample_dfn  # noqa: F401
from fractureflow.synthetic_engine.surface import BaseSurface  # noqa: F401

__version__ = "0.1.0"
