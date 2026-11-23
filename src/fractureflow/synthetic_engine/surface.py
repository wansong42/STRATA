# -*- coding: utf-8 -*-
"""R183-T2a fBm 基面（value-noise 高度场，双线性，解析梯度）。

设计：
  - 格点值 noise 多八度叠加（gain/lacunarity 预注册），双线性插值 →
    沿格边线性 → 求交器沿边解析求根精确（intersector.py）；
  - 解析梯度（每像素）→ 法向图 / Lambert 光照；
  - DEM tile 接口预留不实现（任务书 §2-T2）。
一切 relief 振幅为 RANDOMIZATION_PARAM（prereg honesty_policy #3）。
"""
from __future__ import annotations

import numpy as np


class BaseSurface:
    """局部面参数化：S(u,v) = O + u·eu + v·ev + n_out·(A·r(u,v))，r∈[-1,1] 峰约 1。"""

    def __init__(
        self,
        width_m: float,
        height_m: float,
        lattice: int = 257,
        amplitude: float = 0.03,
        octaves: int = 6,
        gain: float = 0.5,
        lacunarity: float = 2.0,
        seed: int = 0,
        base_cells: int = 4,
    ):
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.lattice = int(lattice)
        self.amplitude = float(amplitude)
        self.octaves = int(octaves)
        self.gain = float(gain)
        self.lacunarity = float(lacunarity)
        self.seed = int(seed)
        self.cell_u = self.width_m / (self.lattice - 1)
        self.cell_v = self.height_m / (self.lattice - 1)

        # 多八度格点值噪声叠加（按 std 归一）
        ii = np.arange(self.lattice)
        U, V = np.meshgrid(ii / (self.lattice - 1) * self.width_m, ii / (self.lattice - 1) * self.height_m, indexing="ij")
        rng = np.random.default_rng(seed)
        acc = np.zeros((self.lattice, self.lattice), dtype=np.float64)
        wsum = 0.0
        for o in range(self.octaves):
            cells = int(round(base_cells * self.lacunarity**o))
            cells = max(1, min(cells, self.lattice - 1))
            w = self.gain**o
            wsum += w
            g = rng.normal(0.0, 1.0, size=(cells + 1, cells + 1))
            step_u = self.width_m / cells
            step_v = self.height_m / cells
            fu = np.clip(U / step_u, 0, cells - 1e-9)
            fv = np.clip(V / step_v, 0, cells - 1e-9)
            iu = fu.astype(np.int64)
            iv = fv.astype(np.int64)
            au = fu - iu
            av = fv - iv
            g00 = g[iu, iv]
            g10 = g[iu + 1, iv]
            g01 = g[iu, iv + 1]
            g11 = g[iu + 1, iv + 1]
            acc += w * ((1 - au) * (1 - av) * g00 + au * (1 - av) * g10 + (1 - au) * av * g01 + au * av * g11)
        acc /= wsum
        self.R = acc  # (lattice,lattice) 格点高度值，index [i_u, j_v]

    # ------------------------------------------------------------- 基础查询
    def eval_at(self, u: np.ndarray, v: np.ndarray):
        """双线性插值 r(u,v) 与解析梯度 (r_u, r_v)。u,v 米制，形状任意。"""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        fu = np.clip(u / self.cell_u, 0.0, self.lattice - 1.0 - 1e-9)
        fv = np.clip(v / self.cell_v, 0.0, self.lattice - 1.0 - 1e-9)
        iu = fu.astype(np.int64)
        iv = fv.astype(np.int64)
        au = fu - iu
        av = fv - iv
        R = self.R
        g00 = R[iu, iv]
        g10 = R[iu + 1, iv]
        g01 = R[iu, iv + 1]
        g11 = R[iu + 1, iv + 1]
        r = (1 - au) * (1 - av) * g00 + au * (1 - av) * g10 + (1 - au) * av * g01 + au * av * g11
        r_u = ((g10 - g00) * (1 - av) + (g11 - g01) * av) / self.cell_u
        r_v = ((g01 - g00) * (1 - au) + (g11 - g10) * au) / self.cell_v
        return r, r_u, r_v

    def lattice_uv(self):
        """格点 (u,v) 网格（米）。"""
        xs = np.arange(self.lattice) * self.cell_u
        ys = np.arange(self.lattice) * self.cell_v
        return xs, ys

    # --------------------------------------------------------- DEM 预留接口
    def load_dem_tile(self, tile_path: str, crs: str | None = None) -> None:  # pragma: no cover
        """预留接口：真实 DEM tile 替换 fBm（任务书 §2-T2『预留接口不实现』）。"""
        raise NotImplementedError("DEM tile 接口预留（R183 不实现）—— 见 contracts/prereg")
