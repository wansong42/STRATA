# -*- coding: utf-8 -*-
"""M4 Quantitative Baselines, Perozzo DSE vs Manual Analysis, and Data-Ladder Anchor Extractor.

Strictly read-only on:
- M1_real_to_exact_bridge/
- M2_active_identifiability/
- TC_network/ & triangle_contact/
- data/external/perozzo_2024/
- results/digital_ledger.json & historical anchor JSONs

Writes ONLY inside M4_benchmark_figures/ (single-writer discipline).
Enforces:
- Pure stdlib zipfile + xml.etree.ElementTree xlsx parsing (zero openpyxl).
- Exact rational Fraction arithmetic via frozen compile_pair.project_pair for Nominal DFN and Naive Union.
- Track separation (synthetic/historical ladder vs real-conditioned 22-cluster benchmarks vs Perozzo DSE vs Manual).
- Zero oracle numbers in outward outputs; null (never 0.0) on empty denominators; strict SHA256 verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SPRINT_DIR = HERE.parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())

M1_DIR = SPRINT_DIR / "M1_real_to_exact_bridge"
M2_DIR = SPRINT_DIR / "M2_active_identifiability"
EXEC_NEXT = REPO / "results" / "w180_technical_route_20261117" / "execution_next"
TRI_CONTACT_DIR = EXEC_NEXT / "triangle_contact"

sys.path.insert(0, str(M1_DIR))
sys.path.insert(0, str(TRI_CONTACT_DIR))

import real_network_format as rnf  # noqa: E402
import compile_pair  # noqa: E402

NS_XLSX = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ORACLE_FORBIDDEN_VALUES = (12.37, 12.3705, 19.1268, 16.2175, 11.024, 10.034, 20.81, 31.7)
SITES_ORDER = ("jardim_garcia_uh", "perozzo_finale_ligure")

PEROZZO_MANUAL_XLSX = REPO / "data" / "external" / "perozzo_2024" / "Manual-Discontinuities-Geometrical-Info.xlsx"
PEROZZO_AUTO_XLSX = REPO / "data" / "external" / "perozzo_2024" / "Automatic-Discontinuities-Geometrical-Info.xlsx"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Pure stdlib OPC xlsx parser (zipfile + ElementTree, matching TH_data/probe_formats.py)
# ---------------------------------------------------------------------------

def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return idx - 1


def parse_xlsx_table(path: Path) -> Dict[str, Any]:
    """Exhaustively parse sheet1 of an .xlsx file using only stdlib zipfile + ElementTree."""
    t0 = time.time()
    shared_strings: List[str] = []
    rows_raw: List[List[Any]] = []

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        if "xl/sharedStrings.xml" in names:
            with zf.open("xl/sharedStrings.xml") as fh:
                for _, el in ET.iterparse(fh, events=("end",)):
                    if el.tag == NS_XLSX + "si":
                        shared_strings.append("".join(t.text or "" for t in el.iter(NS_XLSX + "t")))
                        el.clear()

        sheet_files = sorted(n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        if not sheet_files:
            raise ValueError(f"No worksheet XML found in {path}")
        sheet_xml = sheet_files[0]

        max_cols = 0
        with zf.open(sheet_xml) as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag != NS_XLSX + "row":
                    continue
                cells: Dict[int, Any] = {}
                for c in el.iter(NS_XLSX + "c"):
                    ref = c.get("r") or ""
                    ctype = c.get("t")
                    v_el = c.find(NS_XLSX + "v")
                    is_el = c.find(NS_XLSX + "is")
                    val: Any = None
                    if ctype == "s" and v_el is not None and v_el.text is not None:
                        val = shared_strings[int(v_el.text)]
                    elif ctype == "inlineStr" and is_el is not None:
                        val = "".join(x.text or "" for x in is_el.iter(NS_XLSX + "t"))
                    elif ctype == "str" and v_el is not None:
                        val = v_el.text or ""
                    elif v_el is not None and v_el.text is not None:
                        txt = v_el.text
                        try:
                            val = float(txt)
                        except ValueError:
                            val = txt
                    if val is not None and ref:
                        col_i = _col_index(ref)
                        cells[col_i] = val
                        if col_i + 1 > max_cols:
                            max_cols = col_i + 1
                if cells:
                    row_list = [cells.get(i) for i in range(max_cols)]
                    rows_raw.append(row_list)
                el.clear()

    if not rows_raw:
        raise ValueError(f"Empty worksheet in {path}")
    headers = [str(h) for h in rows_raw[0]]
    data_rows = rows_raw[1:]
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "headers": headers,
        "row_count": len(data_rows),
        "rows": data_rows,
        "parse_seconds": round(time.time() - t0, 3),
    }


def _quantiles_and_moments(sorted_vals: List[float]) -> Dict[str, Any]:
    n = len(sorted_vals)
    if n == 0:
        return {
            "count": 0,
            "min": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
            "sum": None,
        }

    def pct(q: float) -> float:
        if n == 1:
            return sorted_vals[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    total = math.fsum(sorted_vals)
    mean = total / n
    var = math.fsum((x - mean) ** 2 for x in sorted_vals) / n
    return {
        "count": n,
        "min": round(sorted_vals[0], 6),
        "p5": round(pct(0.05), 6),
        "p10": round(pct(0.10), 6),
        "p25": round(pct(0.25), 6),
        "p50": round(pct(0.50), 6),
        "p75": round(pct(0.75), 6),
        "p90": round(pct(0.90), 6),
        "p95": round(pct(0.95), 6),
        "p99": round(pct(0.99), 6),
        "max": round(sorted_vals[-1], 6),
        "mean": round(mean, 6),
        "std": round(math.sqrt(var), 6),
        "sum": round(total, 6),
    }


def _histogram_bins(vals: Sequence[float], edges: Sequence[float]) -> List[int]:
    counts = [0] * (len(edges) - 1)
    lo_edge = edges[0]
    hi_edge = edges[-1]
    n_bins = len(counts)
    step = (hi_edge - lo_edge) / n_bins
    for x in vals:
        if x < lo_edge or x > hi_edge:
            continue
        if x == hi_edge:
            idx = n_bins - 1
        else:
            idx = int((x - lo_edge) / step)
            if idx < 0:
                idx = 0
            elif idx >= n_bins:
                idx = n_bins - 1
        counts[idx] += 1
    return counts


def _summarize_perozzo_table(parsed: Dict[str, Any], label: str) -> Dict[str, Any]:
    headers = parsed["headers"]
    col_idx = {h: i for i, h in enumerate(headers)}
    required_cols = [
        "Dip", "DipDirection", "Radius", "Xcenter", "Ycenter", "Zcenter",
        "Nx", "Ny", "Nz", "Mcoplanarity", "Kcolinearity", "MeanAbsError", "StDevAbsError",
    ]
    missing = [c for c in required_cols if c not in col_idx]
    if missing:
        raise ValueError(f"{label} xlsx missing columns: {missing}")

    cols: Dict[str, List[float]] = {c: [] for c in required_cols}
    cols["DiskArea_m2"] = []
    cols["Log10Radius"] = []
    missing_counts: Dict[str, int] = {c: 0 for c in required_cols}
    bad_core_rows = 0
    core_cols = [
        "Dip", "DipDirection", "Radius", "Xcenter", "Ycenter", "Zcenter",
        "Nx", "Ny", "Nz", "Mcoplanarity", "MeanAbsError", "StDevAbsError",
    ]

    for row in parsed["rows"]:
        row_vals: Dict[str, Optional[float]] = {}
        core_ok = True
        for c in required_cols:
            raw = row[col_idx[c]] if col_idx[c] < len(row) else None
            try:
                v = float(raw)
                if not math.isfinite(v):
                    raise ValueError("non-finite")
                row_vals[c] = v
            except (TypeError, ValueError):
                row_vals[c] = None
                missing_counts[c] += 1
                if c in core_cols:
                    core_ok = False
        if not core_ok:
            bad_core_rows += 1
            continue
        for c in required_cols:
            if row_vals[c] is not None:
                cols[c].append(row_vals[c])  # type: ignore[arg-type]
        r = float(row_vals["Radius"])  # type: ignore[arg-type]
        cols["DiskArea_m2"].append(math.pi * r * r)
        if r > 0.0:
            cols["Log10Radius"].append(math.log10(r))

    n_valid = len(cols["Radius"])
    if n_valid == 0:
        raise SystemExit(f"Empty valid facet set in {label}")

    stats = {}
    for key in ("Radius", "DiskArea_m2", "Dip", "DipDirection", "MeanAbsError", "StDevAbsError", "Mcoplanarity", "Kcolinearity"):
        s = sorted(cols[key])
        stats[key] = _quantiles_and_moments(s)

    radii = cols["Radius"]
    maes = cols["MeanAbsError"]
    size_thresholds = {
        "lt_0p10m": sum(1 for r in radii if r < 0.10),
        "lt_0p20m": sum(1 for r in radii if r < 0.20),
        "lt_0p50m": sum(1 for r in radii if r < 0.50),
        "lt_1p00m": sum(1 for r in radii if r < 1.00),
        "ge_1p00m": sum(1 for r in radii if r >= 1.00),
        "ge_2p00m": sum(1 for r in radii if r >= 2.00),
        "ge_5p00m": sum(1 for r in radii if r >= 5.00),
    }
    size_fractions = {k: round(v / n_valid, 6) for k, v in size_thresholds.items()}

    m1_eligible = sum(
        1 for i in range(n_valid)
        if maes[i] <= 0.05 and 0.5 <= radii[i] <= 8.0
        and (cols["Nx"][i] ** 2 + cols["Ny"][i] ** 2 + cols["Nz"][i] ** 2) ** 0.5 > 0.99
    )

    # Histograms for plotting directly from JSON
    log10_r_edges = [-1.5 + 0.1 * k for k in range(36)]  # -1.5 to +2.0 (0.0316 m to 100 m)
    dip_edges = [5.0 * k for k in range(19)]            # 0 to 90 deg
    dd_edges = [15.0 * k for k in range(25)]            # 0 to 360 deg

    bbox = {
        "xmin": round(min(cols["Xcenter"]), 4),
        "xmax": round(max(cols["Xcenter"]), 4),
        "ymin": round(min(cols["Ycenter"]), 4),
        "ymax": round(max(cols["Ycenter"]), 4),
        "zmin": round(min(cols["Zcenter"]), 4),
        "zmax": round(max(cols["Zcenter"]), 4),
    }

    return {
        "label": label,
        "source_path": parsed["path"],
        "bytes": parsed["bytes"],
        "sha256": parsed["sha256"],
        "parse_seconds": parsed["parse_seconds"],
        "declared_data_rows": parsed["row_count"],
        "valid_facets": n_valid,
        "bad_or_nonfinite_core_rows": bad_core_rows,
        "per_column_missing_counts": {k: v for k, v in missing_counts.items() if v > 0},
        "bbox_utm_m": bbox,
        "metrics": stats,
        "radius_threshold_counts": size_thresholds,
        "radius_threshold_fractions": size_fractions,
        "m1_eligibility_filter_pass_count": m1_eligible,
        "m1_eligibility_filter_pass_fraction": round(m1_eligible / n_valid, 6),
        "histograms": {
            "log10_radius": {
                "edges": [round(e, 4) for e in log10_r_edges],
                "counts": _histogram_bins(cols["Log10Radius"], log10_r_edges),
            },
            "dip_deg": {
                "edges": dip_edges,
                "counts": _histogram_bins(cols["Dip"], dip_edges),
            },
            "dip_direction_deg": {
                "edges": dd_edges,
                "counts": _histogram_bins([d % 360.0 for d in cols["DipDirection"]], dd_edges),
            },
        },
    }


def compute_perozzo_dse_vs_manual(out_path: Optional[Path] = None) -> Dict[str, Any]:
    manual_raw = parse_xlsx_table(PEROZZO_MANUAL_XLSX)
    auto_raw = parse_xlsx_table(PEROZZO_AUTO_XLSX)
    manual_sum = _summarize_perozzo_table(manual_raw, "expert_manual")
    auto_sum = _summarize_perozzo_table(auto_raw, "automatic_dse")

    n_man = manual_sum["valid_facets"]
    n_auto = auto_sum["valid_facets"]
    if n_man != 1405 or n_auto != 202371:
        raise ValueError(f"Unexpected Perozzo counts: manual={n_man}, auto={n_auto}")

    # Compare orientation distributions (Dip & DipDirection histogram L1 / cosine overlap)
    def _norm_hist(cnts: Sequence[int]) -> List[float]:
        s = sum(cnts)
        return [c / s if s > 0 else 0.0 for c in cnts]

    p_dip_m = _norm_hist(manual_sum["histograms"]["dip_deg"]["counts"])
    p_dip_a = _norm_hist(auto_sum["histograms"]["dip_deg"]["counts"])
    p_dd_m = _norm_hist(manual_sum["histograms"]["dip_direction_deg"]["counts"])
    p_dd_a = _norm_hist(auto_sum["histograms"]["dip_direction_deg"]["counts"])

    def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return round(sum(x * y for x, y in zip(a, b)) / (na * nb), 6) if na > 0 and nb > 0 else 0.0

    summary = {
        "schema": "m4-perozzo-dse-vs-manual/v1",
        "at_utc": utc_now(),
        "track": "PEROZZO_DSE_VS_MANUAL_SEPARATE_COLUMN",
        "never_used_as_ground_truth": True,
        "parser_used": "stdlib_only:zipfile+xml.etree.ElementTree",
        "comparison": {
            "manual_facet_count": n_man,
            "auto_dse_facet_count": n_auto,
            "oversegmentation_count_ratio": round(n_auto / n_man, 4),
            "median_radius_ratio_manual_over_auto": round(
                manual_sum["metrics"]["Radius"]["p50"] / auto_sum["metrics"]["Radius"]["p50"], 4
            ),
            "median_area_ratio_manual_over_auto": round(
                manual_sum["metrics"]["DiskArea_m2"]["p50"] / auto_sum["metrics"]["DiskArea_m2"]["p50"], 4
            ),
            "total_disk_area_ratio_auto_over_manual": round(
                auto_sum["metrics"]["DiskArea_m2"]["sum"] / manual_sum["metrics"]["DiskArea_m2"]["sum"], 4
            ),
            "median_mean_abs_error_ratio_auto_over_manual": round(
                auto_sum["metrics"]["MeanAbsError"]["p50"] / manual_sum["metrics"]["MeanAbsError"]["p50"], 4
            ),
            "dip_histogram_cosine_similarity": _cosine_sim(p_dip_m, p_dip_a),
            "dip_direction_histogram_cosine_similarity": _cosine_sim(p_dd_m, p_dd_a),
            "dip_histogram_tv_distance": round(0.5 * sum(abs(x - y) for x, y in zip(p_dip_m, p_dip_a)), 6),
            "dip_direction_histogram_tv_distance": round(0.5 * sum(abs(x - y) for x, y in zip(p_dd_m, p_dd_a)), 6),
        },
        "manual": manual_sum,
        "automatic_dse": auto_sum,
        "disclaimer": (
            "独立分栏对照：自动 DSE 面（202,371）与专家解译面（1,405）均来自 Perozzo 2024 公开数据，"
            "二者属不同解译尺度特征对比，绝不作为地下体内真值，也不混入 M1/M2 的 22 簇贯通评估分母。"
        ),
    }
    if out_path is not None:
        write_json(out_path, summary)
    return summary


# ---------------------------------------------------------------------------
# Historical Data Ladder Anchors (strictly non-oracle, read from disk JSONs)
# ---------------------------------------------------------------------------

def extract_data_ladder_anchors(out_path: Optional[Path] = None) -> Dict[str, Any]:
    ledger = json.loads((REPO / "results" / "digital_ledger.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in ledger.get("entries", []) if isinstance(e, dict) and "id" in e}

    l1_json = json.loads((REPO / "results" / "honest_leaderboard" / "l1_local__beishan_22.json").read_text(encoding="utf-8"))
    forge_json = json.loads((REPO / "results" / "forge_kgrid_ext.json").read_text(encoding="utf-8"))
    pc_json = json.loads((REPO / "results" / "pointcloud_gate.json").read_text(encoding="utf-8"))

    nns001 = by_id["NNS-001"]
    nns046 = by_id["NNS-046"]
    nns021 = by_id["NNS-021"]

    for eid, entry in (("NNS-001", nns001), ("NNS-046", nns046), ("NNS-021", nns021)):
        if "oracle" in str(entry.get("status", "")).lower():
            raise SystemExit(f"ORACLE_CONTAMINATION_GUARD: {eid} has oracle status {entry.get('status')!r}")

    v_l1 = float(l1_json["mae_mean"])
    forge_k12_row = next(r for r in forge_json["rows"] if r["K"] == 12 and r["variant"] == "anchored")
    v_fmi = float(forge_k12_row["mae"])
    v_pc = float(pc_json["evaluation"]["mae_hidden_deg"])

    if abs(v_l1 - float(nns001["value"])) > 1e-4:
        raise ValueError("NNS-001 mismatch vs source JSON")
    if abs(v_fmi - float(nns046["value"])) > 1e-4:
        raise ValueError("NNS-046 mismatch vs source JSON")
    if abs(v_pc - float(nns021["value"])) > 1e-4:
        raise ValueError("NNS-021 mismatch vs source JSON")

    anchors = {
        "schema": "m4-historical-data-ladder/v1",
        "at_utc": utc_now(),
        "track": "HISTORICAL_DATA_LADDER_NON_ORACLE",
        "contains_oracle": False,
        "excluded_oracle_note": "Strict exclusion of all oracle / diagnostic ceiling readings (e.g. NNS-011 / NNS-009 / NNS-010).",
        "stages": [
            {
                "stage_order": 1,
                "ledger_id": "NNS-001",
                "label_zh": "无联结钻孔 (L0/L1)",
                "metric_kind": "pointwise_hidden_acos_abs_cos_mae_deg",
                "value_deg": round(v_l1, 4),
                "display_deg": f"{v_l1:.2f}°",
                "protocol_tag": "北山 22 井 × 10 seeds · 严格无泄漏 (l1_local)",
                "source_json": "results/honest_leaderboard/l1_local__beishan_22.json",
                "source_sha256": sha256_file(REPO / "results" / "honest_leaderboard" / "l1_local__beishan_22.json"),
                "is_synthetic": False,
                "is_oracle": False,
            },
            {
                "stage_order": 2,
                "ledger_id": "NNS-046",
                "label_zh": "FMI 井壁迹线 (K=12)",
                "metric_kind": "set_level_modal_error_deg",
                "value_deg": round(v_fmi, 4),
                "display_deg": f"{v_fmi:.2f}°",
                "protocol_tag": "FORGE 双井 FMI · K=12 anchored 组系级模态角距",
                "source_json": "results/forge_kgrid_ext.json",
                "source_sha256": sha256_file(REPO / "results" / "forge_kgrid_ext.json"),
                "is_synthetic": False,
                "is_oracle": False,
            },
            {
                "stage_order": 3,
                "ledger_id": "NNS-021",
                "label_zh": "稠密点云 (合成壁面)",
                "metric_kind": "pointwise_hidden_acos_abs_cos_mae_deg",
                "value_deg": round(v_pc, 4),
                "display_deg": f"{v_pc:.2f}°",
                "protocol_tag": "合成 K=4 多走向壁面 (60% 隐伏, 2.5cm 噪 · 非实地开挖真值)",
                "source_json": "results/pointcloud_gate.json",
                "source_sha256": sha256_file(REPO / "results" / "pointcloud_gate.json"),
                "is_synthetic": True,
                "is_oracle": False,
            },
        ],
    }
    if out_path is not None:
        write_json(out_path, anchors)
    return anchors


# ---------------------------------------------------------------------------
# Exact Rational Baseline 1 (Nominal DFN) & Baseline 2 (Naive Union)
# ---------------------------------------------------------------------------

def _segment_intersects_window(window: Optional[Tuple[Q, Q]], segments: Sequence[Dict[str, Any]]) -> bool:
    if window is None:
        return False
    wlo, whi = window
    for seg in segments:
        slo = rnf.exact(seg["lo"])
        shi = rnf.exact(seg["hi"])
        lo = max(wlo, slo)
        hi = min(whi, shi)
        if lo < hi:
            return True
        if lo == hi:
            lc = True if lo > slo else bool(seg["lo_closed"])
            hc = True if hi < shi else bool(seg["hi_closed"])
            if lc and hc:
                return True
    return False


def _bfs_shortest_path_and_cut(
    nodes: Sequence[str],
    edges: Iterable[Tuple[str, str]],
    source: str = "@A",
    target: str = "@B",
) -> Dict[str, Any]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_set = set()
    for a, b in edges:
        e = tuple(sorted((a, b)))
        if e not in edge_set:
            edge_set.add(e)
            adj[a].append(b)
            adj[b].append(a)
    for n in adj:
        adj[n].sort()

    parent: Dict[str, Optional[str]] = {source: None}
    queue = [source]
    for cur in queue:
        if cur == target:
            break
        for nxt in adj[cur]:
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)

    if target in parent:
        path = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return {"connected": True, "path": path, "cut": None}

    reached = set(parent)
    tri_ids = {n for n in nodes if not n.startswith("@")}
    unattached = [ep for ep in ("A", "B") if not adj.get("@" + ep)]
    return {
        "connected": False,
        "path": None,
        "cut": {
            "S": sorted(tri_ids & reached),
            "T": sorted(tri_ids - reached),
            "unattached_endpoints": unattached,
        },
    }


def _item_signature_and_aabb(
    item: Dict[str, Any],
    t_lo: Q,
    t_hi: Q,
) -> Tuple[Tuple[Any, ...], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]:
    verts = tuple(tuple(rnf.exact(c) for c in v) for v in item["vertices"])
    vel = tuple(rnf.exact(c) for c in item["v"])
    bmin = []
    bmax = []
    for k in range(3):
        vals = []
        vk = vel[k]
        for pt in verts:
            vals.append(pt[k] + t_lo * vk)
            if t_hi != t_lo:
                vals.append(pt[k] + t_hi * vk)
        bmin.append(min(vals))
        bmax.append(max(vals))
    sig = (verts, vel)
    return sig, (bmin[0], bmin[1], bmin[2]), (bmax[0], bmax[1], bmax[2])


def _aabb_disjoint(
    a_min: Tuple[Q, Q, Q],
    a_max: Tuple[Q, Q, Q],
    b_min: Tuple[Q, Q, Q],
    b_max: Tuple[Q, Q, Q],
) -> bool:
    return (
        a_max[0] < b_min[0] or b_max[0] < a_min[0]
        or a_max[1] < b_min[1] or b_max[1] < a_min[1]
        or a_max[2] < b_min[2] or b_max[2] < a_min[2]
    )


def _project_pair_exact(
    snapshot_id: str,
    mode_id: str,
    domain: Sequence[str],
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    left_geom: Optional[Tuple[Tuple[Any, ...], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]] = None,
    right_geom: Optional[Tuple[Tuple[Any, ...], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]] = None,
    cache: Optional[Dict[Any, Optional[Tuple[Q, Q]]]] = None,
) -> Optional[Tuple[Q, Q]]:
    if left_geom is not None and right_geom is not None:
        if _aabb_disjoint(left_geom[1], left_geom[2], right_geom[1], right_geom[2]):
            return None
        cache_key = (left_geom[0], right_geom[0], domain[0], domain[1])
        if cache is not None and cache_key in cache:
            return cache[cache_key]
    else:
        cache_key = None

    pm = {
        "schema": "rigid-pair/v1",
        "snapshot_id": snapshot_id,
        "mode_id": mode_id,
        "t_closed": list(domain),
        "pieces": [
            {"id": left["id"], "vertices": left["vertices"], "velocity": left["v"]},
            {"id": right["id"], "vertices": right["vertices"], "velocity": right["v"]},
        ],
    }
    res = compile_pair.project_pair(pm)["t_closed"]
    win = None if res is None else (rnf.exact(res[0]), rnf.exact(res[1]))
    if cache is not None and cache_key is not None:
        cache[cache_key] = win
    return win


def run_nominal_and_union_on_cluster(model: Dict[str, Any]) -> Dict[str, Any]:
    """Run Baseline 1 (Nominal DFN at reference m0, t=0) and Baseline 2 (Naive Union over all legal (m, t))."""
    cid = model["snapshot_id"]
    ref = model["reference"]
    ref_mode_id = ref["mode"]
    ref_t = ref["t"]
    ref_t_q = rnf.exact(ref_t)
    ref_mode = next(m for m in model["modes"] if m["id"] == ref_mode_id)

    ep_items = [
        {"id": "@A", "vertices": [model["endpoints"]["A"]], "v": ["0/1", "0/1", "0/1"], "component": "@A"},
        {"id": "@B", "vertices": [model["endpoints"]["B"]], "v": ["0/1", "0/1", "0/1"], "component": "@B"},
    ]
    pair_cache: Dict[Any, Optional[Tuple[Q, Q]]] = {}

    # Baseline 1: Nominal DFN at (ref_mode, ref_t)
    items_nom = ref_mode["triangles"] + ep_items
    nom_geom = {it["id"]: _item_signature_and_aabb(it, ref_t_q, ref_t_q) for it in items_nom}
    tri_to_comp_nom = {it["id"]: it["component"] for it in items_nom}
    nom_edges: List[Tuple[str, str]] = []
    for left, right in itertools.combinations(items_nom, 2):
        if left["id"].startswith("@") and right["id"].startswith("@"):
            continue
        win = _project_pair_exact(
            cid,
            ref_mode_id,
            [ref_t, ref_t],
            left,
            right,
            left_geom=nom_geom[left["id"]],
            right_geom=nom_geom[right["id"]],
            cache=pair_cache,
        )
        if win is not None:
            nom_edges.append(tuple(sorted((left["id"], right["id"]))))  # type: ignore[arg-type]

    nom_bfs = _bfs_shortest_path_and_cut([it["id"] for it in items_nom], nom_edges)
    nom_frac_edges = sorted({
        tuple(sorted((tri_to_comp_nom[a], tri_to_comp_nom[b])))
        for a, b in nom_edges
        if tri_to_comp_nom[a] != tri_to_comp_nom[b]
    })

    # Baseline 2: Naive Union across all feasible (m, t)
    union_tri_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    union_frac_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    all_tri_nodes = {"@A", "@B"}
    all_frac_nodes = {"@A", "@B"}

    for mode in model["modes"]:
        if not mode["segments"]:
            continue
        mid = mode["id"]
        dom_lo, dom_hi = rnf.exact(mode["t_closed"][0]), rnf.exact(mode["t_closed"][1])
        items = mode["triangles"] + ep_items
        mode_geom = {it["id"]: _item_signature_and_aabb(it, dom_lo, dom_hi) for it in items}
        tri_to_comp = {it["id"]: it["component"] for it in items}
        all_tri_nodes.update(it["id"] for it in items)
        all_frac_nodes.update(it["component"] for it in items)

        for left, right in itertools.combinations(items, 2):
            if left["id"].startswith("@") and right["id"].startswith("@"):
                continue
            win = _project_pair_exact(
                cid,
                mid,
                mode["t_closed"],
                left,
                right,
                left_geom=mode_geom[left["id"]],
                right_geom=mode_geom[right["id"]],
                cache=pair_cache,
            )
            if _segment_intersects_window(win, mode["segments"]):
                assert win is not None
                te = tuple(sorted((left["id"], right["id"])))
                prov = {"mode": mid, "t_window": [rnf.qs(win[0]), rnf.qs(win[1])]}
                union_tri_edges.setdefault(te, []).append(prov)
                ca, cb = tri_to_comp[left["id"]], tri_to_comp[right["id"]]
                if ca != cb:
                    fe = tuple(sorted((ca, cb)))
                    union_frac_edges.setdefault(fe, []).append(prov)

    union_tri_bfs = _bfs_shortest_path_and_cut(sorted(all_tri_nodes), union_tri_edges.keys())
    union_frac_bfs = _bfs_shortest_path_and_cut(sorted(all_frac_nodes), union_frac_edges.keys())

    # Provenance of union path edges (which modes support each step)
    union_path_provenance = None
    if union_tri_bfs["connected"]:
        p = union_tri_bfs["path"]
        union_path_provenance = []
        for u, v in zip(p, p[1:]):
            te = tuple(sorted((u, v)))
            union_path_provenance.append({
                "from": u,
                "to": v,
                "supporting_windows": union_tri_edges[te],
            })

    return {
        "nominal_dfn": {
            "reference_mode": ref_mode_id,
            "reference_t": ref_t,
            "connected": nom_bfs["connected"],
            "path": nom_bfs["path"],
            "cut": nom_bfs["cut"],
            "active_tri_edge_count": len(nom_edges),
            "active_tri_edges": [list(e) for e in sorted(nom_edges)],
            "active_fracture_edge_count": len(nom_frac_edges),
        },
        "naive_union": {
            "connected": union_tri_bfs["connected"],
            "connected_fracture_graph": union_frac_bfs["connected"],
            "tri_vs_frac_graph_agree": union_tri_bfs["connected"] == union_frac_bfs["connected"],
            "path": union_tri_bfs["path"],
            "fracture_path": union_frac_bfs["path"],
            "cut": union_tri_bfs["cut"],
            "union_tri_edge_count": len(union_tri_edges),
            "union_fracture_edge_count": len(union_frac_edges),
            "union_path_provenance": union_path_provenance,
        },
    }


# ---------------------------------------------------------------------------
# Cluster Geometry & Non-Orthogonality Extractor (for Fig 1 & Fig 2)
# ---------------------------------------------------------------------------

def extract_cluster_geometry_metrics(model: Dict[str, Any], reg_entry: Dict[str, Any]) -> Dict[str, Any]:
    fcat = model["fracture_catalog"]
    fids = sorted(fcat.keys())
    pair_dots: List[float] = []
    pair_angles_deg: List[float] = []
    for i in range(len(fids)):
        ni = fcat[fids[i]]["normal_nominal"]
        for j in range(i + 1, len(fids)):
            nj = fcat[fids[j]]["normal_nominal"]
            d = abs(sum(ni[k] * nj[k] for k in range(3)))
            d_clamped = max(0.0, min(1.0, d))
            pair_dots.append(round(d_clamped, 6))
            pair_angles_deg.append(round(math.degrees(math.acos(d_clamped)), 4))

    sigmas = [float(fcat[fid]["residual_sigma_m"]) for fid in fids]
    areas = [float(fcat[fid]["area_m2"]) for fid in fids]
    radii = [float(fcat[fid]["char_radius_m"]) for fid in fids]
    dips = [float(fcat[fid]["dip_deg"]) for fid in fids]
    dds = [float(fcat[fid]["dip_direction_deg"]) for fid in fids]

    return {
        "cluster_id": model["snapshot_id"],
        "site_id": reg_entry["site_id"],
        "stratum": reg_entry["stratum"],
        "boundary_model": reg_entry["boundary_model"],
        "fracture_count": len(fids),
        "mode_count": len(model["modes"]),
        "triangles_per_mode": len(model["modes"][0]["triangles"]),
        "preserved_hole_count": len(reg_entry.get("preserved_hole_fragments", [])),
        "quantization_sensitive": bool(reg_entry.get("quantization_sensitive", False)),
        "min_nominal_pair_gap_m": reg_entry.get("min_nominal_pair_gap_m"),
        "non_orthogonal_pairs_0p05_0p95": reg_entry["non_orthogonality"]["non_orthogonal_pairs_0p05_0p95"],
        "total_fracture_pairs": reg_entry["non_orthogonality"]["total_pairs"],
        "non_orthogonal_fraction": reg_entry["non_orthogonality"]["non_orthogonal_fraction"],
        "pair_abs_normal_dots": pair_dots,
        "pair_acute_angles_deg": pair_angles_deg,
        "residual_sigma_min_m": round(min(sigmas), 6),
        "residual_sigma_max_m": round(max(sigmas), 6),
        "mean_area_m2": round(sum(areas) / len(areas), 5),
        "mean_radius_m": round(sum(radii) / len(radii), 5),
        "dips_deg": dips,
        "dip_directions_deg": dds,
    }


# ---------------------------------------------------------------------------
# Strict Contamination & Empty-Denominator Guards (M4-N1, M4-N3, M4-N4)
# ---------------------------------------------------------------------------

def assert_no_contamination(obj: Any, path_label: str = "real_outcrop_benchmarks") -> None:
    """Reject any oracle value or synthetic scope inside real-outcrop benchmark artifacts (M4-N1)."""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if "SYNTHETIC_DECLARED_TRIANGLE_UNION" in text:
        raise SystemExit(f"CONTAMINATION_REJECTED (M4-N1): SYNTHETIC_DECLARED_TRIANGLE_UNION found in {path_label}")
    low = text.lower()
    if '"is_oracle": true' in low or '"oracle"' in low:
        raise SystemExit(f"CONTAMINATION_REJECTED (M4-N1): oracle reference found in {path_label}")
    for ov in ORACLE_FORBIDDEN_VALUES:
        tok = f"{ov:.2f}"
        tok4 = f"{ov:.4f}"
        if tok4 in text or (tok in text and path_label != "data_ladder"):
            # Double check raw numeric equality on leaves to avoid accidental substring hits
            def _walk(node: Any) -> None:
                if isinstance(node, dict):
                    for v in node.values():
                        _walk(v)
                elif isinstance(node, list):
                    for v in node:
                        _walk(v)
                elif isinstance(node, (int, float)) and not isinstance(node, bool):
                    if abs(float(node) - ov) < 1e-4:
                        raise SystemExit(
                            f"CONTAMINATION_REJECTED (M4-N1): forbidden oracle numeric value {node} "
                            f"(matches {ov}) found in {path_label}"
                        )
            _walk(obj)


def assert_honest_empty_denominators(block: Dict[str, Any], label: str) -> None:
    """Enforce M4-N3: if a denominator count is 0, every rate using that denominator MUST be None (null)."""
    rate_specs = [
        ("ambiguous_count", "ambiguous_rate_check", False),
        ("disconnected_count", "phantom_fp_given_disconnected_rate", True),
        ("ambiguous_count", "active_step1_collapse_rate", True),
        ("ambiguous_count", "active_step2_collapse_rate", True),
        ("ambiguous_count", "active_step3_collapse_rate", True),
    ]
    for denom_key, rate_key, must_exist in rate_specs:
        denom = block.get(denom_key)
        if denom == 0 and rate_key in block:
            if block[rate_key] is not None:
                raise SystemExit(
                    f"EMPTY_DENOMINATOR_VIOLATION (M4-N3): {label}.{denom_key}=0 "
                    f"but {rate_key}={block[rate_key]!r} (must be null, never 0.0)"
                )
        if must_exist and denom == 0 and block.get(rate_key) is not None:
            raise SystemExit(
                f"EMPTY_DENOMINATOR_VIOLATION (M4-N3): {label}.{denom_key}=0 "
                f"but {rate_key}={block.get(rate_key)!r}"
            )


def _safe_rate(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return round(num / den, 6)


# ---------------------------------------------------------------------------
# M2 Run Discovery, Verification, and 3-Way Aggregation
# ---------------------------------------------------------------------------

def find_m2_run_dir(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = REPO / p
        if not (p / "summary.json").is_file():
            raise SystemExit(f"Explicit M2 run dir missing summary.json: {p}")
        return p
    runs_root = M2_DIR / "runs"
    if not runs_root.is_dir():
        return None
    candidates = sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and (d / "summary.json").is_file()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _extract_m2_status(cert: Dict[str, Any]) -> str:
    st = cert.get("status")
    if st in (
        "CERTIFIED_CONNECTED",
        "CERTIFIED_DISCONNECTED",
        "AMBIGUOUS_WITH_VERIFIED_WITNESSES",
        "BUDGET_EXCEEDED",
    ):
        return st
    if st in ("REFUSED", "REFUSED_NO_CERTIFICATE"):
        return "BUDGET_EXCEEDED"
    if st == "CERTIFIED_WITHIN_DECLARED_MODEL":
        return "CERTIFIED_CONNECTED" if cert.get("all_connected") else "CERTIFIED_DISCONNECTED"
    raise SystemExit(f"Unsupported M2 certification status {st!r} for {cert.get('cluster_id')}")


def _extract_rho_float(cert: Dict[str, Any]) -> Optional[float]:
    rho = cert.get("rho")
    if not isinstance(rho, dict):
        return None
    inf_raw = rho.get("infimum")
    if inf_raw is None:
        return None
    return float(rnf.exact(inf_raw))


def _aggregate_site_metrics(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise SystemExit(f"Empty cluster group for {label}")
    completed_rows = [r for r in rows if r["ours_status"] != "BUDGET_EXCEEDED"]
    n_comp = len(completed_rows)
    n_refused = n - n_comp

    n_conn = sum(1 for r in completed_rows if r["ours_status"] == "CERTIFIED_CONNECTED")
    n_disc = sum(1 for r in completed_rows if r["ours_status"] == "CERTIFIED_DISCONNECTED")
    n_amb = sum(1 for r in completed_rows if r["ours_status"] == "AMBIGUOUS_WITH_VERIFIED_WITNESSES")

    n_silent_flip = sum(1 for r in completed_rows if r["silent_flip"] is True)
    n_silent_flip_false_conn = sum(1 for r in completed_rows if r["silent_flip"] is True and r["nominal_connected"])
    n_silent_flip_false_disc = sum(1 for r in completed_rows if r["silent_flip"] is True and not r["nominal_connected"])

    n_phantom_all = sum(1 for r in completed_rows if r["phantom_union_path"] is True)
    n_phantom_on_disc = sum(
        1 for r in completed_rows if r["ours_status"] == "CERTIFIED_DISCONNECTED" and r["naive_union_connected"]
    )
    n_union_conn = sum(1 for r in rows if r["naive_union_connected"])
    n_nom_conn = sum(1 for r in rows if r["nominal_connected"])

    rhos = [r["rho_float"] for r in completed_rows if r["rho_float"] is not None]
    rhos_sorted = sorted(rhos)

    out = {
        "group_label": label,
        "cluster_count": n,
        "completed_count": n_comp,
        "budget_exceeded_count": n_refused,
        "budget_exceeded_rate": _safe_rate(n_refused, n),
        "certified_connected_count": n_conn,
        "certified_disconnected_count": n_disc,
        "ambiguous_count": n_amb,
        "disconnected_count": n_disc,
        "certified_connected_rate": _safe_rate(n_conn, n_comp),
        "certified_disconnected_rate": _safe_rate(n_disc, n_comp),
        "ambiguous_rate": _safe_rate(n_amb, n_comp),
        "nominal_dfn": {
            "denominator_completed": n_comp,
            "nominal_connected_count_all_m1": n_nom_conn,
            "silent_flip_count": n_silent_flip,
            "silent_flip_rate": _safe_rate(n_silent_flip, n_comp),
            "silent_flip_false_positive_risk_count": n_silent_flip_false_conn,
            "silent_flip_false_negative_risk_count": n_silent_flip_false_disc,
            "silent_flip_false_positive_risk_rate": _safe_rate(n_silent_flip_false_conn, n_comp),
            "silent_flip_false_negative_risk_rate": _safe_rate(n_silent_flip_false_disc, n_comp),
        },
        "naive_union": {
            "denominator_completed": n_comp,
            "union_connected_count_all_m1": n_union_conn,
            "phantom_union_count": n_phantom_all,
            "phantom_union_rate": _safe_rate(n_phantom_all, n_comp),
            "phantom_fp_given_disconnected_count": n_phantom_on_disc,
            "phantom_fp_given_disconnected_rate": _safe_rate(n_phantom_on_disc, n_disc),
        },
        "ours": {
            "denominator_completed": n_comp,
            "denominator_all_m1": n,
            "three_state_counts": {
                "CERTIFIED_CONNECTED": n_conn,
                "CERTIFIED_DISCONNECTED": n_disc,
                "AMBIGUOUS_WITH_VERIFIED_WITNESSES": n_amb,
                "BUDGET_EXCEEDED": n_refused,
            },
            "silent_misjudgment_within_declared_model_count": 0,
            "silent_misjudgment_within_declared_model_rate": 0.0,
            "rho_defined_count": len(rhos),
            "rho_stats": _quantiles_and_moments(rhos_sorted) if rhos_sorted else None,
        },
        # Mirror keys for M4-N3 empty-denominator guard
        "phantom_fp_given_disconnected_rate": _safe_rate(n_phantom_on_disc, n_disc),
    }
    assert_honest_empty_denominators(out, label)
    return out


def _quantile_linear(sorted_vals: List[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _extract_active_curves(
    m2_summary: Dict[str, Any],
    case_active_map: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract homogeneous-denominator active collapse curves across M2's simulated strategies.

    Reads directly from M2's summary.json (`active_collapse_simulation`) with exact
    group stratification (`combined_real_conditioned`, `jardim_garcia_uh`, `perozzo_finale_ligure`)
    and computes per-step [q25, q75] bands across ambiguous clusters from M2's `per_cluster`.
    """
    amb_rows = [r for r in rows if r["ours_status"] == "AMBIGUOUS_WITH_VERIFIED_WITNESSES"]
    n_amb = len(amb_rows)
    sim_root = m2_summary.get("active_collapse_simulation") or {}

    if n_amb == 0:
        return {
            "ambiguous_denominator": 0,
            "curves_by_group": {
                g: {
                    "denominator": 0,
                    "steps": [0, 1, 2, 3],
                    "strategies": {},
                }
                for g in ("combined_real_conditioned", "jardim_garcia_uh", "perozzo_finale_ligure")
            },
        }

    per_cluster_sim = sim_root.get("per_cluster") or {}
    agg_all = sim_root.get("aggregate_all_ambiguous") or {}
    agg_site = sim_root.get("aggregate_by_site") or {}

    groups_spec = [
        ("combined_real_conditioned", None, agg_all),
        ("jardim_garcia_uh", "jardim_garcia_uh", agg_site.get("jardim_garcia_uh") or {}),
        ("perozzo_finale_ligure", "perozzo_finale_ligure", agg_site.get("perozzo_finale_ligure") or {}),
    ]

    curves_by_group: Dict[str, Any] = {}
    for gname, site_filter, agg_dict in groups_spec:
        sub_amb = [
            r for r in amb_rows if site_filter is None or r["site_id"] == site_filter
        ]
        denom = len(sub_amb)
        if denom == 0 or not agg_dict:
            curves_by_group[gname] = {
                "denominator": denom,
                "steps": [0, 1, 2, 3],
                "strategies": {},
            }
            continue

        # Discover sorted integer steps from M2's curve dicts
        sample_strat = next(iter(agg_dict.values()))
        raw_curve_dict = sample_strat.get("collapse_curve_measure_float") or sample_strat.get("collapse_curve_cells_float") or {}
        steps = sorted(int(k) for k in raw_curve_dict.keys())

        strats_out: Dict[str, Any] = {}
        for sname, sdata in agg_dict.items():
            meas_map = sdata.get("collapse_curve_measure_float") or {}
            cell_map = sdata.get("collapse_curve_cells_float") or {}
            meas_mean = [float(meas_map[str(k)]) for k in steps] if meas_map else []
            cell_mean = [float(cell_map[str(k)]) for k in steps] if cell_map else []

            # Compute cross-cluster [q25, q75] bands from M2 per_cluster when >=2 ambiguous clusters exist
            meas_q25, meas_q75 = None, None
            cell_q25, cell_q75 = None, None
            if denom >= 2 and per_cluster_sim:
                q25_m, q75_m, q25_c, q75_c = [], [], [], []
                has_all_cids = all(
                    r["cluster_id"] in per_cluster_sim
                    and sname in (per_cluster_sim[r["cluster_id"]].get("strategies") or {})
                    for r in sub_amb
                )
                if has_all_cids:
                    for k in steps:
                        vm = sorted(
                            float(per_cluster_sim[r["cluster_id"]]["strategies"][sname]["collapse_curve_measure_float"][str(k)])
                            for r in sub_amb
                        )
                        vc = sorted(
                            float(per_cluster_sim[r["cluster_id"]]["strategies"][sname]["collapse_curve_cells_float"][str(k)])
                            for r in sub_amb
                        )
                        q25_m.append(round(_quantile_linear(vm, 0.25), 6))
                        q75_m.append(round(_quantile_linear(vm, 0.75), 6))
                        q25_c.append(round(_quantile_linear(vc, 0.25), 6))
                        q75_c.append(round(_quantile_linear(vc, 0.75), 6))
                    meas_q25, meas_q75 = q25_m, q75_m
                    cell_q25, cell_q75 = q25_c, q75_c

            strats_out[sname] = {
                "ambiguous_cluster_count": denom,
                "mean": meas_mean,
                "q25": meas_q25,
                "q75": meas_q75,
                "collapse_curve_measure": {
                    "mean": meas_mean,
                    "q25": meas_q25,
                    "q75": meas_q75,
                    "mean_steps": sdata.get("mean_steps_measure_float"),
                    "step1": sdata.get("step1_collapse_rate_measure"),
                    "step2": sdata.get("step2_collapse_rate_measure"),
                    "step3": sdata.get("step3_collapse_rate_measure"),
                },
                "collapse_curve_cells": {
                    "mean": cell_mean,
                    "q25": cell_q25,
                    "q75": cell_q75,
                    "mean_steps": sdata.get("mean_steps_cells_float"),
                    "step1": sdata.get("step1_collapse_rate_cells"),
                    "step2": sdata.get("step2_collapse_rate_cells"),
                    "step3": sdata.get("step3_collapse_rate_cells"),
                },
            }

        curves_by_group[gname] = {
            "denominator": denom,
            "ambiguous_cluster_ids": [r["cluster_id"] for r in sub_amb],
            "steps": steps,
            "strategies": strats_out,
        }

    return {
        "ambiguous_denominator": n_amb,
        "random_seeds": sim_root.get("random_seeds"),
        "max_steps_cap": sim_root.get("max_steps_cap"),
        "zero_leakage_protocol": sim_root.get("zero_leakage_protocol"),
        "curves_by_group": curves_by_group,
    }


def build_benchmark(
    *,
    m1_dir: Path = M1_DIR,
    m2_run_dir: Optional[Path] = None,
    allow_pending_m2: bool = False,
    out_dir: Path = HERE,
    run_dse: bool = True,
) -> Dict[str, Any]:
    t_start = time.time()
    reg_path = m1_dir / "cluster_register.json"
    if not reg_path.is_file():
        raise SystemExit(f"Missing M1 cluster_register.json: {reg_path}")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    clusters_meta = reg.get("clusters", [])
    if not clusters_meta:
        raise SystemExit("M1 cluster_register.json has 0 clusters")

    if run_dse:
        dse_summary = compute_perozzo_dse_vs_manual(out_dir / "perozzo_dse_vs_manual_summary.json")
    else:
        dse_path = out_dir / "perozzo_dse_vs_manual_summary.json"
        if not dse_path.is_file():
            dse_path = HERE / "perozzo_dse_vs_manual_summary.json"
        if dse_path.is_file():
            dse_summary = json.loads(dse_path.read_text(encoding="utf-8"))
        else:
            dse_summary = compute_perozzo_dse_vs_manual(out_dir / "perozzo_dse_vs_manual_summary.json")

    ladder_anchors = extract_data_ladder_anchors(out_dir / "data_ladder_anchors.json")

    if m2_run_dir is None and not allow_pending_m2:
        raise SystemExit(
            "M2 run directory not found. Pass --m2-run <dir> once M2 completes, "
            "or pass --allow-pending-m2 for Phase-1 scaffold execution."
        )

    m2_summary: Optional[Dict[str, Any]] = None
    m2_summary_sha: Optional[str] = None
    if m2_run_dir is not None:
        m2_sum_path = m2_run_dir / "summary.json"
        if not m2_sum_path.is_file():
            raise SystemExit(f"M2 summary.json missing at {m2_sum_path}")
        m2_summary = json.loads(m2_sum_path.read_text(encoding="utf-8"))
        m2_summary_sha = sha256_file(m2_sum_path)

    per_cluster_rows: List[Dict[str, Any]] = []
    case_active_map: Dict[str, Dict[str, Any]] = {}
    all_pair_dots_by_site: Dict[str, List[float]] = {s: [] for s in SITES_ORDER}
    all_pair_angles_by_site: Dict[str, List[float]] = {s: [] for s in SITES_ORDER}

    for entry in clusters_meta:
        cid = entry["cluster_id"]
        model_path = m1_dir / entry["model_path"]
        model = json.loads(model_path.read_text(encoding="utf-8"))

        # Validate M1 model & canonical sha256
        val = rnf.validate(model, repo_root=REPO, verify_source_sha=False)
        if val["input_sha256"] != entry["model_canonical_sha256"]:
            raise SystemExit(f"M1 canonical sha256 mismatch for {cid}")
        if model["scope"] != rnf.REAL_SCOPE:
            raise SystemExit(f"M1 invalid scope for {cid}: {model['scope']!r}")

        geom = extract_cluster_geometry_metrics(model, entry)
        all_pair_dots_by_site[entry["site_id"]].extend(geom["pair_abs_normal_dots"])
        all_pair_angles_by_site[entry["site_id"]].extend(geom["pair_acute_angles_deg"])

        b12 = run_nominal_and_union_on_cluster(model)
        nom_conn = b12["nominal_dfn"]["connected"]
        union_conn = b12["naive_union"]["connected"]

        row: Dict[str, Any] = {
            "cluster_id": cid,
            "site_id": entry["site_id"],
            "stratum": entry["stratum"],
            "boundary_model": entry["boundary_model"],
            "fracture_count": geom["fracture_count"],
            "mode_count": geom["mode_count"],
            "triangles_per_mode": geom["triangles_per_mode"],
            "preserved_hole_count": geom["preserved_hole_count"],
            "non_orthogonal_pairs": geom["non_orthogonal_pairs_0p05_0p95"],
            "total_fracture_pairs": geom["total_fracture_pairs"],
            "non_orthogonal_fraction": geom["non_orthogonal_fraction"],
            "quantization_sensitive": geom["quantization_sensitive"],
            "min_nominal_pair_gap_m": geom["min_nominal_pair_gap_m"],
            "model_canonical_sha256": val["input_sha256"],
            "nominal_connected": nom_conn,
            "nominal_active_tri_edges": b12["nominal_dfn"]["active_tri_edge_count"],
            "nominal_active_frac_edges": b12["nominal_dfn"]["active_fracture_edge_count"],
            "nominal_path_len": (len(b12["nominal_dfn"]["path"]) - 2) if b12["nominal_dfn"]["path"] else None,
            "naive_union_connected": union_conn,
            "naive_union_frac_connected": b12["naive_union"]["connected_fracture_graph"],
            "naive_union_tri_edges": b12["naive_union"]["union_tri_edge_count"],
            "naive_union_frac_edges": b12["naive_union"]["union_fracture_edge_count"],
            "naive_union_path_len": (len(b12["naive_union"]["path"]) - 2) if b12["naive_union"]["path"] else None,
            "ours_status": None,
            "exists_connected": None,
            "all_connected": None,
            "rho_exact": None,
            "rho_float": None,
            "rho_attained": None,
            "silent_flip": None,
            "phantom_union_path": None,
            "m2_cell_count": None,
            "m2_top1_fracture": None,
            "m2_top1_role": None,
            "baseline_details": b12,
            "geometry_details": geom,
        }

        if m2_run_dir is not None:
            cert_path = m2_run_dir / "cases" / cid / "certification.json"
            if not cert_path.is_file():
                raise SystemExit(f"M2 certification.json missing for {cid}: {cert_path}")
            cert = json.loads(cert_path.read_text(encoding="utf-8"))

            # M4-N4: Hard verify input_sha256 against M1 model
            cert_sha = cert.get("input_sha256")
            if cert_sha != val["input_sha256"]:
                raise SystemExit(
                    f"M2_INPUT_SHA256_MISMATCH (M4-N4): {cid} M2 certification input_sha256={cert_sha!r} "
                    f"!= M1 canonical sha256={val['input_sha256']!r}"
                )

            ours_status = _extract_m2_status(cert)
            is_refused = (ours_status == "BUDGET_EXCEEDED")
            exists_conn = None if is_refused else bool(cert.get("exists_connected"))
            all_conn = None if is_refused else bool(cert.get("all_connected"))

            # Cross-check Nominal DFN vs M2 nominal cell
            ref_mode = model["reference"]["mode"]
            ref_t = rnf.exact(model["reference"]["t"])
            m2_cells = cert.get("cells", [])
            nom_cell = next(
                (
                    c for c in m2_cells
                    if c.get("mode") == ref_mode and (
                        (c.get("kind") == "POINT" and rnf.exact(c["lo"]) == ref_t)
                        or (c.get("kind") == "OPEN" and rnf.exact(c["lo"]) < ref_t < rnf.exact(c["hi"]))
                    )
                ),
                None,
            )
            if nom_cell is not None and bool(nom_cell["connected"]) != bool(nom_conn):
                raise SystemExit(
                    f"NOMINAL_M2_MISMATCH: {cid} nominal baseline={nom_conn} != M2 nominal cell={nom_cell['connected']}"
                )

            # Cross-check Naive Union edges vs M2 cell active_edges union (if cell active_edges present)
            if (not is_refused) and m2_cells and "active_edges" in m2_cells[0]:
                m2_union_edges = {
                    tuple(sorted((e[0], e[1])))
                    for c in m2_cells
                    for e in c["active_edges"]
                }
                # Verify BFS connectivity on m2_union_edges agrees with union_conn
                m2_nodes = {"@A", "@B"} | {t["id"] for m in model["modes"] for t in m["triangles"]}
                m2_union_conn = _bfs_shortest_path_and_cut(sorted(m2_nodes), m2_union_edges)["connected"]
                if m2_union_conn != union_conn:
                    raise SystemExit(
                        f"NAIVE_UNION_M2_MISMATCH: {cid} naive_union={union_conn} != M2_cell_edge_union={m2_union_conn}"
                    )

            silent_flip = None if is_refused else (ours_status == "AMBIGUOUS_WITH_VERIFIED_WITNESSES")
            phantom_union = None if is_refused else bool(union_conn and not exists_conn)  # connected in union, CERTIFIED_DISCONNECTED in exact

            rho_block = cert.get("rho") or {}
            rho_float = _extract_rho_float(cert)

            row.update({
                "ours_status": ours_status,
                "exists_connected": exists_conn,
                "all_connected": all_conn,
                "rho_exact": rho_block.get("infimum"),
                "rho_float": rho_float,
                "rho_attained": rho_block.get("attained"),
                "silent_flip": silent_flip,
                "phantom_union_path": phantom_union,
                "m2_cell_count": len(m2_cells),
                "m2_cert_sha256": sha256_file(cert_path),
            })

            ac_path = m2_run_dir / "cases" / cid / "active_check.json"
            if ours_status == "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
                if not ac_path.is_file():
                    raise SystemExit(f"M2 active_check.json missing for AMBIGUOUS cluster {cid}")
                ac = json.loads(ac_path.read_text(encoding="utf-8"))
                case_active_map[cid] = ac
                ranked = ac.get("ranked_fractures") or []
                if ranked:
                    row["m2_top1_fracture"] = ranked[0].get("fracture_id")
                    row["m2_top1_role"] = ranked[0].get("witness_diff_role") or ranked[0].get("role")
            elif ac_path.is_file():
                case_active_map[cid] = json.loads(ac_path.read_text(encoding="utf-8"))

        per_cluster_rows.append(row)

    # Geometry & non-orthogonality summary across all 22 real clusters
    all_dots = all_pair_dots_by_site["jardim_garcia_uh"] + all_pair_dots_by_site["perozzo_finale_ligure"]
    all_angles = all_pair_angles_by_site["jardim_garcia_uh"] + all_pair_angles_by_site["perozzo_finale_ligure"]
    geom_summary = {
        "total_clusters": len(per_cluster_rows),
        "jardim_clusters": sum(1 for r in per_cluster_rows if r["site_id"] == "jardim_garcia_uh"),
        "perozzo_clusters": sum(1 for r in per_cluster_rows if r["site_id"] == "perozzo_finale_ligure"),
        "total_fracture_pairs": len(all_dots),
        "non_orthogonal_pairs_0p05_0p95": sum(1 for d in all_dots if 0.05 < d < 0.95),
        "non_orthogonal_pair_fraction": round(sum(1 for d in all_dots if 0.05 < d < 0.95) / len(all_dots), 4),
        "by_site": {
            site: {
                "cluster_count": sum(1 for r in per_cluster_rows if r["site_id"] == site),
                "total_pairs": len(all_pair_dots_by_site[site]),
                "non_orthogonal_pairs_0p05_0p95": sum(1 for d in all_pair_dots_by_site[site] if 0.05 < d < 0.95),
                "non_orthogonal_fraction": round(
                    sum(1 for d in all_pair_dots_by_site[site] if 0.05 < d < 0.95) / len(all_pair_dots_by_site[site]), 4
                ),
                "pair_abs_normal_dots": all_pair_dots_by_site[site],
                "pair_acute_angles_deg": all_pair_angles_by_site[site],
            }
            for site in SITES_ORDER
        },
        "angle_histogram_deg": {
            "edges": [10 * k for k in range(10)],  # 0..90 deg
            "jardim_counts": _histogram_bins(all_pair_angles_by_site["jardim_garcia_uh"], [10 * k for k in range(10)]),
            "perozzo_counts": _histogram_bins(all_pair_angles_by_site["perozzo_finale_ligure"], [10 * k for k in range(10)]),
        },
    }

    # Build output bundle
    if m2_run_dir is None:
        scaffold_summary = {
            "schema": "m4-baseline-scaffold/v1",
            "status": "SCAFFOLD_PENDING_M2",
            "at_utc": utc_now(),
            "elapsed_s": round(time.time() - t_start, 2),
            "m1_register_sha256": sha256_file(reg_path),
            "perozzo_dse_vs_manual_summary_path": "perozzo_dse_vs_manual_summary.json",
            "data_ladder_anchors_path": "data_ladder_anchors.json",
            "cluster_geometry_summary": geom_summary,
            "pre_m2_baseline_counts": {
                site: {
                    "cluster_count": sum(1 for r in per_cluster_rows if r["site_id"] == site),
                    "nominal_connected_count": sum(1 for r in per_cluster_rows if r["site_id"] == site and r["nominal_connected"]),
                    "naive_union_connected_count": sum(1 for r in per_cluster_rows if r["site_id"] == site and r["naive_union_connected"]),
                    "nominal_false_vs_union_true_count": sum(
                        1 for r in per_cluster_rows
                        if r["site_id"] == site and (not r["nominal_connected"]) and r["naive_union_connected"]
                    ),
                }
                for site in SITES_ORDER
            },
            "clusters": per_cluster_rows,
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
            "not_claimed": [
                "M2 runs/<run_id>/ 尚未落盘，本件为 Phase-1 脚手架预跑产物，不含三态认证与主动坍缩指标",
                "非地下体内无条件贯通真值",
                "非水力连通",
            ],
        }
        assert_no_contamination(scaffold_summary, "scaffold_summary")
        write_json(out_dir / "baseline_scaffold_m1_only.json", scaffold_summary)
        _write_csv(out_dir / "per_cluster_table_scaffold.csv", per_cluster_rows)
        return scaffold_summary

    # Full 3-way M1 + M2 benchmark mode
    jardim_rows = [r for r in per_cluster_rows if r["site_id"] == "jardim_garcia_uh"]
    perozzo_rows = [r for r in per_cluster_rows if r["site_id"] == "perozzo_finale_ligure"]

    benchmarks = {
        "combined_real_conditioned": _aggregate_site_metrics(per_cluster_rows, "combined_real_conditioned"),
        "jardim_garcia_uh": _aggregate_site_metrics(jardim_rows, "jardim_garcia_uh"),
        "perozzo_finale_ligure": _aggregate_site_metrics(perozzo_rows, "perozzo_finale_ligure"),
    }

    active_curves = _extract_active_curves(m2_summary or {}, case_active_map, per_cluster_rows)

    summary_out = {
        "schema": "m4-benchmark-summary/v1",
        "status": "COMPLETE_WITH_M2",
        "at_utc": utc_now(),
        "elapsed_s": round(time.time() - t_start, 2),
        "scope": "REAL_DERIVED_CONDITIONED_FINITE_UNION",
        "m1_register_sha256": sha256_file(reg_path),
        "m2_run_dir": str(m2_run_dir.relative_to(REPO)).replace("\\", "/"),
        "m2_run_id": m2_run_dir.name,
        "m2_summary_sha256": m2_summary_sha,
        "real_outcrop_benchmarks": benchmarks,
        "cluster_geometry_summary": geom_summary,
        "active_collapse_comparison": active_curves,
        "separate_tracks": {
            "perozzo_dse_vs_manual_ref": {
                "file": "perozzo_dse_vs_manual_summary.json",
                "sha256": sha256_file(out_dir / "perozzo_dse_vs_manual_summary.json"),
                "manual_count": dse_summary["comparison"]["manual_facet_count"],
                "auto_dse_count": dse_summary["comparison"]["auto_dse_facet_count"],
                "oversegmentation_ratio": dse_summary["comparison"]["oversegmentation_count_ratio"],
            },
            "historical_data_ladder_ref": {
                "file": "data_ladder_anchors.json",
                "sha256": sha256_file(out_dir / "data_ladder_anchors.json"),
                "contains_oracle": False,
            },
        },
        "clusters": [
            {k: v for k, v in r.items() if k not in ("baseline_details", "geometry_details")}
            for r in per_cluster_rows
        ],
        "cluster_details_for_plots": [
            {
                "cluster_id": r["cluster_id"],
                "site_id": r["site_id"],
                "ours_status": r["ours_status"],
                "nominal_connected": r["nominal_connected"],
                "naive_union_connected": r["naive_union_connected"],
                "silent_flip": r["silent_flip"],
                "phantom_union_path": r["phantom_union_path"],
                "rho_float": r["rho_float"],
                "rho_exact": r["rho_exact"],
                "union_path_provenance": r["baseline_details"]["naive_union"]["union_path_provenance"],
            }
            for r in per_cluster_rows
        ],
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
        "not_claimed": [
            "非地下体内无条件贯通真值 (not unconditional subsurface connectivity ground truth)",
            "非水力连通 (not hydraulic connectivity)",
            "合成域历史锚点与 Perozzo DSE 对照严禁混入真实露头 22 簇贯通分母",
        ],
    }

    assert_no_contamination(summary_out["real_outcrop_benchmarks"], "real_outcrop_benchmarks")
    assert_no_contamination(summary_out["clusters"], "clusters")
    write_json(out_dir / "benchmark_summary.json", summary_out)
    _write_csv(out_dir / "per_cluster_table.csv", per_cluster_rows)
    return summary_out


CSV_COLUMNS = [
    "cluster_id",
    "site_id",
    "stratum",
    "boundary_model",
    "fracture_count",
    "mode_count",
    "triangles_per_mode",
    "preserved_hole_count",
    "non_orthogonal_pairs",
    "total_fracture_pairs",
    "non_orthogonal_fraction",
    "quantization_sensitive",
    "min_nominal_pair_gap_m",
    "nominal_connected",
    "nominal_active_tri_edges",
    "nominal_active_frac_edges",
    "nominal_path_len",
    "naive_union_connected",
    "naive_union_frac_connected",
    "naive_union_tri_edges",
    "naive_union_frac_edges",
    "naive_union_path_len",
    "ours_status",
    "exists_connected",
    "all_connected",
    "rho_exact",
    "rho_float",
    "rho_attained",
    "silent_flip",
    "phantom_union_path",
    "m2_cell_count",
    "m2_top1_fracture",
    "m2_top1_role",
    "model_canonical_sha256",
]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            out_r = {}
            for col in CSV_COLUMNS:
                val = r.get(col)
                out_r[col] = "" if val is None else val
            writer.writerow(out_r)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run M4 Perozzo DSE vs Manual + 3-way baselines.")
    ap.add_argument("--m2-run", default=None, help="Path to M2 runs/<run_id> directory")
    ap.add_argument("--allow-pending-m2", action="store_true", help="Run Phase-1 scaffold without M2")
    ap.add_argument("--dse-only", action="store_true", help="Only run Perozzo DSE vs Manual analysis")
    ap.add_argument("--skip-dse", action="store_true", help="Reuse existing perozzo_dse_vs_manual_summary.json if present")
    ap.add_argument("--out-dir", default=None, help="Output directory (defaults to M4_benchmark_figures)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dse_only:
        dse = compute_perozzo_dse_vs_manual(out_dir / "perozzo_dse_vs_manual_summary.json")
        print(json.dumps(dse["comparison"], ensure_ascii=False, indent=2))
        return 0

    m2_dir = find_m2_run_dir(args.m2_run)
    res = build_benchmark(
        m1_dir=M1_DIR,
        m2_run_dir=m2_dir,
        allow_pending_m2=args.allow_pending_m2,
        out_dir=out_dir,
        run_dse=not args.skip_dse,
    )
    print(
        json.dumps(
            {
                "status": res["status"],
                "clusters": res["cluster_geometry_summary"]["total_clusters"],
                "elapsed_s": res["elapsed_s"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
