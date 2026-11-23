# -*- coding: utf-8 -*-
"""R110 评分卡冻结常数 + 重构评分统一入口.

协议权威: results/v_r110_recon_protocol.json (预注册, mtime 早于一切结果).
本模块只引用冻结常数与纯函数接口, 不重复实现公式 (全部在 scripts/r110_recon_metrics.py).

数值卫生: 全程 float64 + 域盒中心化 (继承 segmentation.py 红线; HRF 坐标 ~4e6 量级).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 冻结常数 (协议 §masking_platform / §scorecard_v1) — 单一权威来源
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RECON_PROTOCOL_PATH = PROJECT_ROOT / "results" / "v_r110_recon_protocol.json"

SCANLINE_SEED = 20260828
SCANLINE_GRID = (5, 4)
SCANLINE_JITTER = 8.0
SCANLINE_MIN_DIST_TO_BOREHOLE = 10.0
THETA_MERGE_DEG = 20.0
RENDER_DENSITY = 0.212
RENDER_CAP = 8000
RENDER_MIN = 50
RIPLEY_R_GRID = np.arange(10.0, 81.0, 10.0)
DENSITY_GRID = 8
DENSITY_SIGMA = 80.0
ANGLE_MATCH_DEG = 90.0

# 五层 14 指标冻结名称 (R110 协议)
FROZEN_METRIC_NAMES = [
    # 方向层 (4)
    "fisher_gen_circ", "modal_err", "fisher_circ_margin", "circ_std",
    # 空间层 (3)
    "density_field_l1_rel", "ripley_k_max_rel_dev", "anisotropy_l1_rel",
    # 拓扑层 (2)
    "interdens", "intersection_jaccard",
    # 正演层 (3)
    "heldout_p10_rel_err", "scanline_p10_rel_dev", "scanline_p12_rel_dev",
    # 数据效率层 (2)
    "efficiency_half_budget_slope", "efficiency_quarters_budget_slope",
]

# 五层权重 (R110 协议冻结)
TIER_WEIGHTS = {
    "direction": 0.30,
    "spatial": 0.25,
    "topology": 0.20,
    "forward": 0.15,
    "efficiency": 0.10,
}

# 冻结诚实四锚 (B-1 引用口径)
FROZEN_ANCHORS = {
    "l1_local": 36.687,
    "oracle_floor": 12.37,
    "pointcloud": 0.37,
    "info_ceiling": 31.7,
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScorecardResult:
    """单方法 / 单 seed 的评分卡容器."""
    method: str                          # 方法标识符 (e.g. "B1", "fracgen_k4")
    seed: int                            # 随机种子
    metrics: Dict[str, float]            # FROZEN_METRIC_NAMES -> float
    tiers: Dict[str, float]              # 五层分
    composite: float                     # 加权总分 (越小越好)
    provenance: Dict[str, Any]           # sha / path / mtime
    verdict: str                         # aggregated verdict

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "seed": self.seed,
            "metrics": self.metrics,
            "tiers": self.tiers,
            "composite": self.composite,
            "provenance": self.provenance,
            "verdict": self.verdict,
        }


# --------------------------------------------------------------------------- #
# 协议加载
# --------------------------------------------------------------------------- #
def load_protocol() -> dict:
    """加载 R110 协议 JSON (冻结)."""
    if not RECON_PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"R110 协议JSON缺: {RECON_PROTOCOL_PATH}")
    return json.loads(RECON_PROTOCOL_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 评分卡计算入口
# --------------------------------------------------------------------------- #
def score_reconstruction(
    pred: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
    method: str = "anon",
    seed: int = 42,
    layout: str = "full",
) -> ScorecardResult:
    """对 pred 与 truth 走 R110 评分卡 v1, 返回 ScorecardResult.

    输入:
      pred/truth: 含键 "pos"[N,3], "normals"[N,3], "seg_id"[N] / "set_id"[N]
      method: 方法名
      seed: 随机种子
      layout: 场景名 (full/borehole_1/outcrop_10/...)

    实现 = 委托 scripts/r110_recon_metrics.py 的纯函数, 本函数只做统一封装.
    """
    # lazy import 避免循环 + 让 R110 公式的单测包住 metrics 模块
    import importlib.util
    metrics_path = PROJECT_ROOT / "scripts" / "r110_recon_metrics.py"
    spec = importlib.util.spec_from_file_location("r110_recon_metrics", metrics_path)
    r110 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r110)

    metrics: Dict[str, float] = {}
    # 方向层
    metrics["fisher_gen_circ"] = float(r110.fisher_gen_circ(pred, truth))
    metrics["modal_err"] = float(r110.modal_err(pred, truth))
    metrics["fisher_circ_margin"] = float(r110.fisher_circ_margin(pred, truth))
    metrics["circ_std"] = float(r110.circ_std(pred))
    # 空间层
    metrics["density_field_l1_rel"] = float(r110.density_field_l1_rel(pred, truth))
    metrics["ripley_k_max_rel_dev"] = float(r110.ripley_k_max_rel_dev(pred, truth))
    metrics["anisotropy_l1_rel"] = float(r110.anisotropy_l1_rel(pred, truth))
    # 拓扑层
    metrics["interdens"] = float(r110.interdens(pred, truth))
    metrics["intersection_jaccard"] = float(r110.intersection_jaccard(pred, truth))
    # 正演层
    metrics["heldout_p10_rel_err"] = float(r110.heldout_p10_rel_err(pred, truth))
    metrics["scanline_p10_rel_dev"] = float(r110.scanline_p10_rel_dev(pred, truth, layout=layout))
    metrics["scanline_p12_rel_dev"] = float(r110.scanline_p12_rel_dev(pred, truth, layout=layout))
    # 效率层 (optional: pred 粒度够时可算, 否则标记 NaN)
    try:
        metrics["efficiency_half_budget_slope"] = float(r110.efficiency_half_budget_slope(pred, truth))
    except Exception:
        metrics["efficiency_half_budget_slope"] = float("nan")
    try:
        metrics["efficiency_quarters_budget_slope"] = float(r110.efficiency_quarters_budget_slope(pred, truth))
    except Exception:
        metrics["efficiency_quarters_budget_slope"] = float("nan")

    # 五层分域聚合 (simple mean)
    tiers: Dict[str, float] = {
        "direction": np.mean([metrics["fisher_gen_circ"], metrics["modal_err"],
                              metrics["fisher_circ_margin"], metrics["circ_std"]]),
        "spatial": np.mean([metrics["density_field_l1_rel"], metrics["ripley_k_max_rel_dev"],
                            metrics["anisotropy_l1_rel"]]),
        "topology": np.mean([metrics["interdens"], metrics["intersection_jaccard"]]),
        "forward": np.mean([metrics["heldout_p10_rel_err"], metrics["scanline_p10_rel_dev"],
                            metrics["scanline_p12_rel_dev"]]),
        "efficiency": np.mean([v for k, v in metrics.items() if "efficiency" in k
                               if not np.isnan(v)]) if any(not np.isnan(v) for k, v in metrics.items()
                                                            if "efficiency" in k) else float("nan"),
    }
    composite = sum(TIERS[t] * tiers[t] for t in tiers if not np.isnan(tiers[t]))

    verdict = _aggregate_verdict(metrics)
    provenance = {
        "method": method, "seed": seed, "layout": layout,
        "scorecard_version": "v1", "protocol": str(RECON_PROTOCOL_PATH),
    }
    return ScorecardResult(method=method, seed=seed, metrics=metrics,
                           tiers=tiers, composite=composite,
                           provenance=provenance, verdict=verdict)


def scorecard_metrics() -> List[str]:
    """返回冻结指标名列表."""
    return list(FROZEN_METRIC_NAMES)


# --------------------------------------------------------------------------- #
# 内部
# --------------------------------------------------------------------------- #
def _aggregate_verdict(metrics: Dict[str, float]) -> str:
    """由指标聚合 verdict (简单三分: ok/warn/fail, 基于方向层 & 空间层)."""
    base = [metrics.get("modal_err", 1.0), metrics.get("fisher_gen_circ", 1.0)]
    mean_base = np.mean(base)
    if mean_base < 0.3:
        return "OK"
    elif mean_base < 0.7:
        return "WARN"
    return "FAIL"


# --------------------------------------------------------------------------- #
# CLI helper
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    proto = load_protocol()
    print(f"protocol v{proto.get('schema_version', '?')} with {proto.get('n_metrics', '?')} metrics")
