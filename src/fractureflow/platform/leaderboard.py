# -*- coding: utf-8 -*-
"""多方法对比表聚合."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

TIER_ORDER = ["direction", "spatial", "topology", "forward", "efficiency"]
METRIC_LABELS = {
    "fisher_gen_circ": "Fisher 组系",
    "modal_err": "组系 Modal",
    "fisher_circ_margin": "Fisher Margin",
    "circ_std": "圆 STD",
    "density_field_l1_rel": "密度场 L1",
    "ripley_k_max_rel_dev": "Ripley K max",
    "anisotropy_l1_rel": "各向异性 L1",
    "interdens": "InterDens",
    "intersection_jaccard": "Intersection",
    "heldout_p10_rel_err": "P10 heldout",
    "scanline_p10_rel_dev": "Scanline P10",
    "scanline_p12_rel_dev": "Scanline P12",
    "efficiency_half_budget_slope": "Eff half",
    "efficiency_quarters_budget_slope": "Eff quarters",
}

def aggregate_methods(result_paths, method_labels=None):
    """聚合多张结果件为对比表."""
    rows = []
    for p in result_paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = (method_labels or {}).get(str(p), obj.get("method", obj.get("label", p.stem)))
        metrics = obj.get("metrics", {})
        tiers = obj.get("tiers", {})
        composite = obj.get("composite")
        verdict = obj.get("verdict", "?")
        provenance = obj.get("provenance", {})
        rows.append({
            "label": label, "metrics": metrics, "tiers": tiers,
            "composite": composite, "verdict": verdict,
            "sha": provenance.get("sha256", ""), "source_path": str(p),
        })

    metric_names = sorted(set(m for r in rows for m in r["metrics"].keys()))
    table = []
    for r in rows:
        row = {"method": r["label"], "composite": r["composite"], "verdict": r["verdict"]}
        for m in metric_names:
            v = r["metrics"].get(m)
            row[m] = v if v is not None else float("nan")
        table.append(row)

    sorted_rows = sorted([r for r in rows if r["composite"] is not None],
                         key=lambda r: r["composite"])
    ranking = [{"rank": i+1, "method": r["label"], "composite": r["composite"]}
               for i, r in enumerate(sorted_rows)]
    return {
        "methods": [r["label"] for r in rows],
        "metric_names": metric_names,
        "table": table,
        "tiers": TIER_ORDER,
        "ranking": ranking,
        "n_compared": len(rows),
    }

def compare_table(agg):
    """打印对比表可读版."""
    metric_names = agg["metric_names"]
    lines = []
    header = f"{'Method':<20}" + "".join(f"{METRIC_LABELS.get(m, m):>14}" for m in metric_names) + f"{'Comp':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in agg["table"]:
        line = f"{row['method']:<20}"
        for m in metric_names:
            v = row.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                line += f"{chr(2014):>14}"
            else:
                line += f"{v:>14.4f}"
        c = row["composite"]
        cs = f"{c:.4f}" if c is not None else chr(2014)
        line += f"{cs:>10}"
        lines.append(line)
    if agg["ranking"]:
        lines.append("")
        lines.append("=== Ranking (composite, small=better) ===")
        for r in agg["ranking"]:
            lines.append(f"  #{r['rank']}: {r['method']} = {r['composite']:.4f}")
    return "\n".join(lines)
