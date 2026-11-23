# -*- coding: utf-8 -*-
"""Export verified static JSON bundle for STRATA 3D Public Showcase.

Strictly read-only against frozen M1 / M2 / M3 artifacts and existing
`IdentifiabilityBridge`:
  - Re-verifies every cluster's `model.json` file SHA256 & canonical input_sha256.
  - Re-verifies M2 `certification.json` & `active_check.json` input_sha256 & artifact SHA256.
  - Re-verifies M3 `verdict_batch.json` (`22/22 AGREE`, zero disagreement).
  - Extracts 3D topological outer boundary rings and inner hole contour loops.
  - Precomputes all Depth-1 and Depth-2 active collapse responses & receipts.
  - Exports compact static JSONs to `dist/data/cases.json`, `dist/data/cases/<cluster_id>.json`,
    and `dist/data/manifest_sha256.json` (+ `dist/manifest_sha256.json`).
  - Zero console window popups (AGENTS.md §11 compliant).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fractureflow.surface_product.identifiability_bridge import (  # noqa: E402
    DEFAULT_M1_ROOT,
    DEFAULT_M2_ROOT,
    DEFAULT_M3_ROOT,
    REAL_SCOPE,
    IdentifiabilityBridge,
    _canonical_sha256,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _loop_perimeter(loop: Sequence[Sequence[float]]) -> float:
    if len(loop) < 2:
        return 0.0
    total = 0.0
    for i in range(len(loop)):
        total += _dist3(loop[i], loop[(i + 1) % len(loop)])
    return total


def _extract_fracture_contours(
    triangles: Sequence[Mapping[str, Any]],
    fractures_meta: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Decompose each fracture's triangles into outer boundary ring and inner hole contour rings."""
    hole_count_by_fid = {
        str(f["id"]): int(f.get("preserved_hole_count") or 0)
        for f in fractures_meta
    }
    tris_by_comp: Dict[str, List[Mapping[str, Any]]] = {}
    for tri in triangles:
        comp = str(tri["component"])
        tris_by_comp.setdefault(comp, []).append(tri)

    def _vkey(pt: Sequence[float]) -> str:
        return f"{round(pt[0] * 1e5)},{round(pt[1] * 1e5)},{round(pt[2] * 1e5)}"

    out: Dict[str, Dict[str, Any]] = {}
    for comp, comp_tris in tris_by_comp.items():
        edge_counts: Dict[Tuple[str, str], int] = {}
        directed_edges: Dict[Tuple[str, str], Tuple[str, str]] = {}
        coords: Dict[str, List[float]] = {}
        vel: List[float] = [0.0, 0.0, 0.0]

        for tri in comp_tris:
            vs = tri["vertices"]
            vel = list(tri.get("v") or [0.0, 0.0, 0.0])
            ks = [_vkey(p) for p in vs]
            for k, p in zip(ks, vs):
                coords[k] = [round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6)]
            for i in range(3):
                ka, kb = ks[i], ks[(i + 1) % 3]
                if ka == kb:
                    continue
                undir = (ka, kb) if ka < kb else (kb, ka)
                edge_counts[undir] = edge_counts.get(undir, 0) + 1
                directed_edges[undir] = (ka, kb)

        adj: Dict[str, List[str]] = {}
        boundary_segments: List[List[List[float]]] = []
        for undir, cnt in edge_counts.items():
            if cnt == 1:
                ka, kb = directed_edges[undir]
                adj.setdefault(ka, []).append(kb)
                adj.setdefault(kb, []).append(ka)
                boundary_segments.append([coords[ka], coords[kb]])

        # Trace closed loops from boundary adjacency graph
        visited_edges: set[Tuple[str, str]] = set()
        loops: List[List[List[float]]] = []
        for start_k in list(adj.keys()):
            for nxt_k in adj[start_k]:
                e0 = (start_k, nxt_k) if start_k < nxt_k else (nxt_k, start_k)
                if e0 in visited_edges:
                    continue
                loop_keys = [start_k]
                visited_edges.add(e0)
                prev_k, cur_k = start_k, nxt_k
                guard = 0
                while cur_k != start_k and guard < 2000:
                    guard += 1
                    loop_keys.append(cur_k)
                    candidates = [nb for nb in adj.get(cur_k, []) if nb != prev_k]
                    chosen = None
                    for nb in candidates:
                        ek = (cur_k, nb) if cur_k < nb else (nb, cur_k)
                        if ek not in visited_edges:
                            chosen = nb
                            visited_edges.add(ek)
                            break
                    if chosen is None:
                        break
                    prev_k, cur_k = cur_k, chosen
                if len(loop_keys) >= 3:
                    loops.append([coords[k] for k in loop_keys])

        loops.sort(key=_loop_perimeter, reverse=True)
        outer_ring = loops[0] if loops else []
        hole_rings = loops[1:] if len(loops) > 1 else []
        out[comp] = {
            "fracture_id": comp,
            "v": [round(float(vel[0]), 6), round(float(vel[1]), 6), round(float(vel[2]), 6)],
            "preserved_hole_count": hole_count_by_fid.get(comp, 0),
            "outer_ring": outer_ring,
            "hole_rings": hole_rings,
            "boundary_segments": boundary_segments,
        }
    return out


def export_bundle(
    *,
    dist_dir: Optional[Path] = None,
    m1_root: Optional[Path] = None,
    m2_root: Optional[Path] = None,
    m3_root: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    target_dist = (dist_dir or (THIS_DIR / "dist")).resolve()
    data_dir = target_dist / "data"
    cases_dir = data_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    m1_dir = (m1_root or DEFAULT_M1_ROOT).resolve()
    m2_dir = (m2_root or DEFAULT_M2_ROOT).resolve()
    m3_dir = (m3_root or DEFAULT_M3_ROOT).resolve()

    bridge = IdentifiabilityBridge(m1_root=m1_dir, m2_root=m2_dir, m3_root=m3_dir)

    # Load M1 register, M2 summary, and M3 verdict_batch directly for cross-audit
    m1_reg_path = m1_dir / "cluster_register.json"
    m2_run_dir = bridge._discover_m2_run_dir()
    assert m2_run_dir is not None, f"M2 run directory not found under {m2_dir}"
    m2_sum_path = m2_run_dir / "summary.json"
    m3_batch_path = m3_dir / "verdict_batch.json"

    assert m1_reg_path.is_file(), f"Missing M1 register: {m1_reg_path}"
    assert m2_sum_path.is_file(), f"Missing M2 summary: {m2_sum_path}"
    assert m3_batch_path.is_file(), f"Missing M3 verdict_batch: {m3_batch_path}"

    m1_reg_sha = _file_sha256(m1_reg_path)
    m2_sum_sha = _file_sha256(m2_sum_path)
    m3_batch_sha = _file_sha256(m3_batch_path)

    m1_reg = json.loads(m1_reg_path.read_text(encoding="utf-8"))
    m2_sum = json.loads(m2_sum_path.read_text(encoding="utf-8"))
    m3_batch = json.loads(m3_batch_path.read_text(encoding="utf-8"))

    m1_reg_canon_sha = _canonical_sha256(m1_reg)
    m2_sum_canon_sha = _canonical_sha256(m2_sum)

    assert m3_batch.get("m1_register_sha256") == m1_reg_canon_sha, "M3 -> M1 register canonical SHA mismatch"
    assert m3_batch.get("m2_summary_sha256") == m2_sum_canon_sha, "M3 -> M2 summary canonical SHA mismatch"
    assert m3_batch.get("counts", {}).get("AGREE") == 22, f"Expected 22/22 AGREE, got {m3_batch.get('counts')}"
    assert m3_batch.get("counts", {}).get("DISAGREE") == 0, "Expected 0 DISAGREE in M3"

    # Cache `_load_and_verify_cluster` and `_scan_real_directory_scope_guard` after first verified pass
    cases_payload = bridge.list_cases()
    assert cases_payload["ok"] is True
    assert cases_payload["total_cases"] == 22
    assert cases_payload["sha_verified_count"] == 22
    assert cases_payload["scope"] == REAL_SCOPE
    assert cases_payload["field_error_rate"] is None
    assert cases_payload["human_time"] is None
    assert cases_payload["hydraulic"] is None

    verified_cluster_cache: Dict[str, Any] = {}
    orig_load_verify = bridge._load_and_verify_cluster
    orig_scan_guard = bridge._scan_real_directory_scope_guard

    def _cached_load_verify(cid: str, reg: Any = None, m2_run_dir: Any = None) -> Any:
        if cid not in verified_cluster_cache:
            verified_cluster_cache[cid] = orig_load_verify(cid, reg=reg, m2_run_dir=m2_run_dir)
        return verified_cluster_cache[cid]

    bridge._load_and_verify_cluster = _cached_load_verify  # type: ignore[method-assign]
    bridge._scan_real_directory_scope_guard = lambda _ids: None  # type: ignore[method-assign]

    m3_verdicts = m3_batch.get("verdicts") or {}
    m3_artifacts = m3_batch.get("case_artifacts") or {}

    exported_files_manifest: Dict[str, Dict[str, Any]] = {}
    source_artifacts_manifest: Dict[str, Dict[str, Any]] = {
        "m1_cluster_register": {
            "path": "results/midterm_sprint_20261120/M1_real_to_exact_bridge/cluster_register.json",
            "sha256": m1_reg_sha,
            "canonical_sha256": m1_reg_canon_sha,
            "bytes": m1_reg_path.stat().st_size,
        },
        "m2_run_summary": {
            "path": f"results/midterm_sprint_20261120/M2_active_identifiability/runs/{m2_run_dir.name}/summary.json",
            "sha256": m2_sum_sha,
            "canonical_sha256": m2_sum_canon_sha,
            "bytes": m2_sum_path.stat().st_size,
        },
        "m3_verdict_batch": {
            "path": "results/midterm_sprint_20261120/M3_independent_judge/verdict_batch.json",
            "sha256": m3_batch_sha,
            "bytes": m3_batch_path.stat().st_size,
        },
    }

    cases_summary_by_id = {c["cluster_id"]: c for c in cases_payload["cases"]}

    for summary_item in cases_payload["cases"]:
        cid = str(summary_item["cluster_id"])
        case_detail = bridge.get_case(cid, allow_live=False)

        # Assert academic boundaries & SHA integrity
        assert case_detail["ok"] is True
        assert case_detail["cluster_id"] == cid
        assert case_detail["scope"] == REAL_SCOPE
        assert case_detail["input_sha256"] == summary_item["input_sha256"]
        assert case_detail["model_file_sha256"] == summary_item["model_file_sha256"]
        assert case_detail["m3_judge_verdict"] == "AGREE", f"{cid} M3 verdict is not AGREE"
        assert case_detail["field_error_rate"] is None
        assert case_detail["human_time"] is None
        assert case_detail["hydraulic"] is None

        m3_v = m3_verdicts.get(cid) or {}
        assert m3_v.get("verdict") == "AGREE"
        assert m3_v.get("first_disagreement") is None
        assert (m3_v.get("checks") or {}).get("canon_sha256") == case_detail["input_sha256"]

        # Enrich M3 judge detail
        case_detail["m3_judge_detail"] = {
            "verdict": m3_v.get("verdict"),
            "status_m2": m3_v.get("status_m2"),
            "first_disagreement": m3_v.get("first_disagreement"),
            "checks": m3_v.get("checks") or {},
            "case_artifacts_sha256": m3_artifacts.get(cid) or {},
            "verdict_batch_sha256": m3_batch_sha,
            "judge_real_sha256": m3_batch.get("judge_real_sha256"),
            "m1_register_sha256": m1_reg_sha,
            "m2_summary_sha256": m2_sum_sha,
        }

        # Extract 3D topological outer boundary rings + inner hole contour rings for nominal mode
        nom_mode = str((case_detail.get("nominal_reference") or {}).get("mode") or "m0")
        nom_geom = (case_detail.get("modes_geometry") or {}).get(nom_mode) or {}
        case_detail["fracture_contours"] = _extract_fracture_contours(
            nom_geom.get("triangles") or [],
            case_detail.get("fractures") or [],
        )

        # Precompute all Depth-1 and Depth-2 collapse responses
        collapse_responses: Dict[str, Dict[str, Any]] = {}
        ranked = (case_detail.get("active_check") or {}).get("ranked_fractures") or []
        for rf in ranked:
            fid = str(rf["fracture_id"])
            for br in rf.get("branches") or []:
                bid = str(br["branch_id"])
                res1 = bridge.apply_check(
                    {
                        "cluster_id": cid,
                        "fracture_id": fid,
                        "branch_id": bid,
                    }
                )
                assert res1["field_error_rate"] is None
                assert res1["human_time"] is None
                assert res1["hydraulic"] is None
                key1 = f"{fid}::{bid}"
                collapse_responses[key1] = res1

                ns = res1.get("next_step")
                if isinstance(ns, Mapping) and isinstance(ns.get("branches"), Mapping):
                    fid2 = str(ns.get("fracture_id") or "")
                    for bid2 in ns["branches"].keys():
                        res2 = bridge.apply_check(
                            {
                                "cluster_id": cid,
                                "fracture_id": fid,
                                "branch_id": bid,
                                "step2_fracture_id": fid2,
                                "step2_branch_id": str(bid2),
                            }
                        )
                        assert res2["field_error_rate"] is None
                        assert res2["human_time"] is None
                        assert res2["hydraulic"] is None
                        key2 = f"{fid}::{bid}::{fid2}::{bid2}"
                        collapse_responses[key2] = res2

        case_detail["collapse_responses"] = collapse_responses

        # Enrich summary item in cases.json for richer sidebar display
        dw = case_detail.get("disconnected_witness") or {}
        cut_gap = dw.get("cut_gap") or {}
        cw = case_detail.get("connected_witness") or {}
        top1 = ranked[0] if ranked else None
        summary_item["cut_gap_distance_m"] = cut_gap.get("distance_m")
        summary_item["connected_witness_mode"] = cw.get("mode")
        summary_item["connected_witness_t_float"] = cw.get("t_float")
        summary_item["disconnected_witness_mode"] = dw.get("mode")
        summary_item["disconnected_witness_t_float"] = dw.get("t_float")
        summary_item["top1_fracture_id"] = top1["fracture_id"] if top1 else None
        summary_item["top1_short_label"] = top1["short_label"] if top1 else None
        summary_item["top1_action_type"] = top1["action_type"] if top1 else None
        summary_item["m2_cert_sha256"] = case_detail.get("m2_cert_sha256")
        summary_item["m2_active_check_sha256"] = case_detail.get("m2_active_check_sha256")
        summary_item["m3_checks"] = m3_v.get("checks") or {}
        summary_item["cell_count"] = case_detail.get("cell_count")
        summary_item["pair_count"] = case_detail.get("pair_count")

        # Write dist/data/cases/<cluster_id>.json
        case_file = cases_dir / f"{cid}.json"
        case_bytes = json.dumps(
            case_detail,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        case_file.write_bytes(case_bytes)

        exported_files_manifest[f"data/cases/{cid}.json"] = {
            "cluster_id": cid,
            "sha256": _bytes_sha256(case_bytes),
            "bytes": len(case_bytes),
            "input_sha256": case_detail["input_sha256"],
            "model_file_sha256": case_detail["model_file_sha256"],
            "m2_cert_sha256": case_detail.get("m2_cert_sha256"),
            "m2_active_check_sha256": case_detail.get("m2_active_check_sha256"),
            "status": case_detail["status"],
            "m3_judge_verdict": case_detail["m3_judge_verdict"],
            "collapse_response_count": len(collapse_responses),
        }

    # Enrich top-level cases.json with M1/M2/M3 global credentials
    cases_payload["m1_register_sha256"] = m1_reg_sha
    cases_payload["m2_summary_sha256"] = m2_sum_sha
    cases_payload["m3_verdict_sha256"] = m3_batch_sha
    cases_payload["m3_judge_real_sha256"] = m3_batch.get("judge_real_sha256")
    cases_payload["m3_counts"] = m3_batch.get("counts")
    cases_payload["m3_enumeration"] = m3_batch.get("enumeration")

    # Assert exact 22-cluster distribution
    sc = cases_payload["status_counts"]
    assert sc.get("CERTIFIED_CONNECTED") == 1, f"Expected 1 CERTIFIED_CONNECTED, got {sc}"
    assert sc.get("CERTIFIED_DISCONNECTED") == 14, f"Expected 14 CERTIFIED_DISCONNECTED, got {sc}"
    assert sc.get("AMBIGUOUS_WITH_VERIFIED_WITNESSES") == 6, f"Expected 6 AMBIGUOUS, got {sc}"
    assert sc.get("BUDGET_EXCEEDED") == 1, f"Expected 1 BUDGET_EXCEEDED, got {sc}"
    assert cases_summary_by_id["jardim_c01"]["status"] == "BUDGET_EXCEEDED"

    cases_file = data_dir / "cases.json"
    cases_bytes = json.dumps(
        cases_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    cases_file.write_bytes(cases_bytes)
    exported_files_manifest["data/cases.json"] = {
        "sha256": _bytes_sha256(cases_bytes),
        "bytes": len(cases_bytes),
        "total_cases": 22,
    }

    # Include static web assets in manifest if present
    for asset_rel in ("index.html", "styles.css", "app.js", "viewer3d.js", "vendor/three.min.js"):
        ap = target_dist / asset_rel
        if ap.is_file():
            exported_files_manifest[asset_rel] = {
                "sha256": _file_sha256(ap),
                "bytes": ap.stat().st_size,
            }

    elapsed = round(time.perf_counter() - t0, 3)
    manifest = {
        "schema": "strata-showcase-manifest-sha256/v1",
        "generated_at_utc": _now_utc(),
        "elapsed_seconds": elapsed,
        "scope": REAL_SCOPE,
        "m2_run_id": m2_run_dir.name,
        "total_clusters": 22,
        "status_counts": cases_payload["status_counts"],
        "site_counts": cases_payload["site_counts"],
        "m3_counts": m3_batch.get("counts"),
        "academic_boundaries": {
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
            "not_claimed": cases_payload["not_claimed"],
        },
        "source_artifacts": source_artifacts_manifest,
        "exported_files": exported_files_manifest,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    (data_dir / "manifest_sha256.json").write_bytes(manifest_bytes)
    (target_dist / "manifest_sha256.json").write_bytes(manifest_bytes)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export STRATA Showcase verified static bundle.")
    parser.add_argument("--dist", default=str(THIS_DIR / "dist"), help="Output dist directory")
    args = parser.parse_args(argv)
    manifest = export_bundle(dist_dir=Path(args.dist))
    print(
        json.dumps(
            {
                "ok": True,
                "total_clusters": manifest["total_clusters"],
                "status_counts": manifest["status_counts"],
                "m3_counts": manifest["m3_counts"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "exported_file_count": len(manifest["exported_files"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
