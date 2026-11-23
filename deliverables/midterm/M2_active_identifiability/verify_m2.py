"""M2 Comprehensive Verification Suite, Summary Assembler, and >=6 Teeth-Bearing Negative Controls.

Performs:
1. Full consistency verification of every cluster's `certification.json` and `active_check.json`
   in a run directory (including from-scratch branch recomputation on Top-3 collapse_tree branches).
2. Exact bit-for-bit equivalence audit between accelerated `solve_real_clusters` and raw
   unmodified `TC_network/compile_network.py` on real clusters.
3. >=7 teeth-bearing negative controls (`M2-N1a`, `M2-N1b`, `M2-N2`, `M2-N3`, `M2-N4`, `M2-N5`,
   `M2-N6a`, `M2-N6b`, `M2-N6c`, `M2-N7`) executed via real CLI subprocesses with
   `creationflags=subprocess.CREATE_NO_WINDOW` (134217728), asserting non-zero exit codes.
4. Assembly of immutable `runs/<run_id>/summary.json` combining certification stats, active
   recommendations, and 4-strategy leak-free simulation results.
5. Audit that zero `__pycache__` or modified files were created inside M1 or frozen TC dirs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SPRINT_DIR = HERE.parent
REPO_ROOT = SPRINT_DIR.parent.parent
M1_DIR = SPRINT_DIR / "M1_real_to_exact_bridge"
TC_DIR = REPO_ROOT / "results" / "w180_technical_route_20261117" / "execution_next" / "dispatch_20261118" / "TC_network"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import active_identifiability as ai
import simulate_active_loop as sim
import solve_real_clusters as solver

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
PYTHON_EXE = Path(sys.executable).resolve()


def _run_py_no_window(
    args: Sequence[str],
    *,
    env_extra: Optional[Dict[str, str]] = None,
    cwd: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Invoke Python subprocess strictly without creating any console window (AGENTS.md §11)."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update(env_extra)
    cmd = [str(PYTHON_EXE), "-B", "-X", "utf8"] + [str(a) for a in args]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    return {
        "cmd": " ".join(cmd),
        "interpreter": str(PYTHON_EXE),
        "executed": True,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "creationflags": CREATE_NO_WINDOW,
    }


def build_summary_json(
    register_path: Path,
    run_dir: Path,
    sim_result: Dict[str, Any],
) -> Dict[str, Any]:
    reg = solver.load_strict_json(register_path)
    cases_dir = run_dir / "cases"

    status_counts_all: Dict[str, int] = {}
    status_counts_by_site: Dict[str, Dict[str, int]] = {}
    complete_counts_all: Dict[str, int] = {}
    complete_counts_by_site: Dict[str, Dict[str, int]] = {}

    per_cluster_rows: List[Dict[str, Any]] = []
    rho_values: List[Dict[str, Any]] = []
    silent_flip_count = 0
    phantom_union_count = 0
    certified_disconnected_count = 0

    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        sid = entry["site_id"]
        cert = solver.load_strict_json(cases_dir / cid / "certification.json")
        ac = solver.load_strict_json(cases_dir / cid / "active_check.json")
        st = cert["status"]
        status_counts_all[st] = status_counts_all.get(st, 0) + 1
        status_counts_by_site.setdefault(sid, {})
        status_counts_by_site[sid][st] = status_counts_by_site[sid].get(st, 0) + 1

        if cert["search_complete"]:
            complete_counts_all[st] = complete_counts_all.get(st, 0) + 1
            complete_counts_by_site.setdefault(sid, {})
            complete_counts_by_site[sid][st] = complete_counts_by_site[sid].get(st, 0) + 1
            if cert["nominal_summary"]["silent_flip_under_declared_model"]:
                silent_flip_count += 1
            if cert["nominal_summary"]["phantom_union_path"]:
                phantom_union_count += 1
            if st == "CERTIFIED_DISCONNECTED":
                certified_disconnected_count += 1
            if cert["rho"] and cert["rho"]["infimum"] is not None:
                rho_values.append(
                    {
                        "cluster_id": cid,
                        "site_id": sid,
                        "rho_infimum": cert["rho"]["infimum"],
                        "rho_infimum_float": cert["rho_infimum_float"],
                        "attained": cert["rho"]["attained"],
                        "nominal_connected": cert["rho"]["nominal_connected"],
                    }
                )

        top = ac.get("top_recommendation")
        per_cluster_rows.append(
            {
                "cluster_id": cid,
                "site_id": sid,
                "stratum": entry["stratum"],
                "fracture_count": entry["fracture_count"],
                "triangles_per_mode": entry["triangles_per_mode"],
                "mode_count": entry["mode_count"],
                "input_sha256": cert["input_sha256"],
                "status": st,
                "search_complete": cert["search_complete"],
                "cells": cert["enumeration"]["cells"] if cert["enumeration"] else 0,
                "pairs": cert["enumeration"]["pairs"] if cert["enumeration"] else 0,
                "rho_infimum": cert["rho"]["infimum"] if cert["rho"] else None,
                "rho_infimum_float": cert["rho_infimum_float"],
                "nominal_connected": cert["nominal_summary"]["connected"] if cert["nominal_summary"] else None,
                "silent_flip_under_declared_model": (
                    cert["nominal_summary"]["silent_flip_under_declared_model"] if cert["nominal_summary"] else None
                ),
                "naive_union_connected": (
                    cert["nominal_summary"]["naive_union_connected"] if cert["nominal_summary"] else None
                ),
                "phantom_union_path": (
                    cert["nominal_summary"]["phantom_union_path"] if cert["nominal_summary"] else None
                ),
                "active_status": ac["status"],
                "top_recommended_fracture": top["fracture_id"] if top else None,
                "top_action_type": top["best_action_type"] if top else None,
                "top_witness_diff_role": top["witness_diff_role"] if top else None,
                "top_minimax_reduction_fraction": top["minimax_reduction_fraction"] if top else None,
                "top_resolved_measure_fraction": top["resolved_measure_fraction"] if top else None,
                "top_guarantees_1step": top["guarantees_1step_certification"] if top else None,
                "certification_path": f"cases/{cid}/certification.json",
                "active_check_path": f"cases/{cid}/active_check.json",
            }
        )

    n_total = len(per_cluster_rows)
    n_complete = sum(1 for r in per_cluster_rows if r["search_complete"])
    n_refused = n_total - n_complete

    # Exact three-state rates over the COMPLETE denominator (excluding refusal envelopes!)
    three_state_rates_complete = {}
    for st in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED", "AMBIGUOUS_WITH_VERIFIED_WITNESSES"):
        cnt = complete_counts_all.get(st, 0)
        three_state_rates_complete[st] = {
            "count": cnt,
            "denominator": n_complete,
            "fraction_exact": solver.qs(Q(cnt, n_complete)) if n_complete > 0 else None,
            "fraction_float": round(cnt / n_complete, 6) if n_complete > 0 else None,
        }

    summary = {
        "schema": "m2-run-summary/v1",
        "scope": solver.REAL_SCOPE,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "m1_cluster_register_path": str(register_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "m1_cluster_register_sha256": solver.file_sha256(register_path),
        "counts": {
            "total_clusters_from_m1": n_total,
            "completed_exact_certification": n_complete,
            "refused_or_budget_exceeded": n_refused,
            "refused_cluster_ids": [r["cluster_id"] for r in per_cluster_rows if not r["search_complete"]],
            "status_counts_all": status_counts_all,
            "status_counts_by_site": status_counts_by_site,
            "complete_denominator_counts": complete_counts_all,
            "complete_denominator_counts_by_site": complete_counts_by_site,
        },
        "three_state_rates_complete_denominator": three_state_rates_complete,
        "baseline_diagnostics_complete_denominator": {
            "denominator_completed": n_complete,
            "nominal_silent_flip_count": silent_flip_count,
            "nominal_silent_flip_rate": round(silent_flip_count / n_complete, 6) if n_complete > 0 else None,
            "phantom_union_path_count": phantom_union_count,
            "certified_disconnected_denominator": certified_disconnected_count,
            "phantom_union_rate_among_disconnected": (
                round(phantom_union_count / certified_disconnected_count, 6)
                if certified_disconnected_count > 0
                else None
            ),
        },
        "rho_distribution_ambiguous": rho_values,
        "active_collapse_simulation": sim_result,
        "clusters": per_cluster_rows,
        "not_claimed": list(solver.NOT_CLAIMED_STANDARD),
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
    }
    solver.write_json(run_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Exact equivalence audit vs raw unmodified TC_network/compile_network.py
# ---------------------------------------------------------------------------


def audit_raw_tc_equivalence(
    register_path: Path,
    run_dir: Path,
    cluster_ids: Sequence[str] = ("perozzo_c05", "perozzo_c02"),
) -> Dict[str, Any]:
    rnf, cn, _, _, _ = solver.load_frozen_deps()
    reg = solver.load_strict_json(register_path)
    reg_dir = register_path.parent
    by_cid = {c["cluster_id"]: c for c in reg["clusters"]}
    checked: List[Dict[str, Any]] = []

    for cid in cluster_ids:
        entry = by_cid[cid]
        m = solver.load_strict_json(reg_dir / entry["model_path"])
        cert = solver.load_strict_json(run_dir / "cases" / cid / "certification.json")
        payload = rnf.to_core_kernel_payload(m, verify_source_sha=False)
        t0 = time.perf_counter()
        raw_tc = cn.compile_network(payload)
        elapsed = round(time.perf_counter() - t0, 3)

        # Compare contact_windows, cells (mode, kind, lo, hi, connected, active_edges, components), rho
        w_m2 = {(w["mode"], w["from"], w["to"]): w["t_closed"] for w in cert["contact_windows"]}
        w_tc = {(w["mode"], w["from"], w["to"]): w["t_closed"] for w in raw_tc["contact_windows"]}
        windows_equal = w_m2 == w_tc

        c_m2 = {
            (c["mode"], c["kind"], c["lo"], c["hi"]): (
                c["connected"],
                c["active_edges"],
                c["components"],
            )
            for c in cert["cells"]
        }
        c_tc = {
            (c["mode"], c["kind"], c["lo"], c["hi"]): (
                c["connected"],
                c["active_edges"],
                c["components"],
            )
            for c in raw_tc["cells"]
        }
        cells_equal = c_m2 == c_tc
        rho_equal = cert["rho"] == raw_tc["rho"]
        status_equal = cert["tc_compat_status"] == raw_tc["status"]
        ok = bool(windows_equal and cells_equal and rho_equal and status_equal)
        checked.append(
            {
                "cluster_id": cid,
                "windows_equal": windows_equal,
                "cells_equal": cells_equal,
                "rho_equal": rho_equal,
                "tc_status_equal": status_equal,
                "windows_count": len(w_tc),
                "cells_count": len(c_tc),
                "raw_tc_seconds": elapsed,
                "ok": ok,
            }
        )
        if not ok:
            raise RuntimeError(f"raw TC_network equivalence failed on {cid}: {checked[-1]}")

    return {"ok": True, "checked_clusters": checked}


# ---------------------------------------------------------------------------
# >=6 Teeth-Bearing Negative Controls (all real CLI non-zero exits!)
# ---------------------------------------------------------------------------


def _make_phantom_union_real_model(base_model: Dict[str, Any], *, cross_mode: bool) -> Dict[str, Any]:
    """Construct a valid TC-network/real-conditioned/v1 model with a phantom union path.

    Three fractures inside Jardim site bbox:
      - F_A (fixed at x=1, y in [1,2], z in [1,2], endpoint A at (1, 3/2, 3/2))
      - F_B (fixed at x=3, y in [1,2], z in [1,2], endpoint B at (3, 3/2, 3/2))
      - F_M (bridge plane of width x in [1, 2], translating along +x with v=(1,0,0) in cross_t mode,
             or sitting at x in [1, 2] in m0 and x in [2, 3] in m1 for cross_mode!)
    In BOTH cases:
      - Union graph across (m, t) has edges @A--F_A--F_M--F_B--@B -> NAIVE UNION CONNECTED!
      - At EVERY single (m, t), F_M touches at most one of F_A or F_B -> CERTIFIED_DISCONNECTED!
    """
    m = copy.deepcopy(base_model)
    # Use jardim_c03 (no holes) as valid site-provenance carrier
    m["snapshot_id"] = f"m2_nc_phantom_{'cross_mode' if cross_mode else 'cross_t'}"
    m["endpoints"] = {"A": ["1/1", "3/2", "3/2"], "B": ["3/1", "3/2", "3/2"]}

    def tri(tid: str, comp: str, verts: List[List[str]], vel: List[str]) -> Dict[str, Any]:
        return {"id": tid, "component": comp, "vertices": verts, "v": vel, "label": "REAL_ALPHA_TRI"}

    # F_A at x = 1 (square split into 2 tris), v = 0
    fa_tris = [
        tri("t_a0", "f_A", [["1/1", "1/1", "1/1"], ["1/1", "2/1", "1/1"], ["1/1", "2/1", "2/1"]], ["0/1", "0/1", "0/1"]),
        tri("t_a1", "f_A", [["1/1", "1/1", "1/1"], ["1/1", "2/1", "2/1"], ["1/1", "1/1", "2/1"]], ["0/1", "0/1", "0/1"]),
    ]
    # F_B at x = 3 (square split into 2 tris), v = 0
    fb_tris = [
        tri("t_b0", "f_B", [["3/1", "1/1", "1/1"], ["3/1", "2/1", "1/1"], ["3/1", "2/1", "2/1"]], ["0/1", "0/1", "0/1"]),
        tri("t_b1", "f_B", [["3/1", "1/1", "1/1"], ["3/1", "2/1", "2/1"], ["3/1", "1/1", "2/1"]], ["0/1", "0/1", "0/1"]),
    ]

    if not cross_mode:
        # Cross-t phantom path: F_M has x-width 1/2 (spanning [7/4, 9/4] at t=0), v = (1, 0, 0).
        # At t in [-1, -3/4]: F_M spans [3/4..5/4] to [1..3/2] -> touches F_A (x=1)! Does NOT touch F_B (x=3)!
        # At t in [3/4, 1]:   F_M spans [5/2..3] to [11/4..13/4] -> touches F_B (x=3)! Does NOT touch F_A (x=1)!
        # Never touches both simultaneously because width = 1/2 < 2 = dist(F_A, F_B)!
        fm_m0 = [
            tri("t_m0", "f_M", [["7/4", "1/1", "3/2"], ["9/4", "1/1", "3/2"], ["9/4", "2/1", "3/2"]], ["1/1", "0/1", "0/1"]),
            tri("t_m1", "f_M", [["7/4", "1/1", "3/2"], ["9/4", "2/1", "3/2"], ["7/4", "2/1", "3/2"]], ["1/1", "0/1", "0/1"]),
        ]
        fm_m1 = [
            tri("t_m0", "f_M", [["7/4", "1/1", "3/2"], ["9/4", "1/1", "3/2"], ["9/4", "2/1", "8/5"]], ["1/1", "0/1", "0/1"]),
            tri("t_m1", "f_M", [["7/4", "1/1", "3/2"], ["9/4", "2/1", "8/5"], ["7/4", "2/1", "8/5"]], ["1/1", "0/1", "0/1"]),
        ]
    else:
        # Cross-mode phantom path: in m0 F_M stays near F_A (x in [1, 3/2], v=(1/10,0,0)),
        # in m1 F_M stays near F_B (x in [5/2, 3], v=(1/10,0,0)).
        fm_m0 = [
            tri("t_m0", "f_M", [["1/1", "1/1", "3/2"], ["3/2", "1/1", "3/2"], ["3/2", "2/1", "3/2"]], ["1/10", "0/1", "0/1"]),
            tri("t_m1", "f_M", [["1/1", "1/1", "3/2"], ["3/2", "2/1", "3/2"], ["1/1", "2/1", "3/2"]], ["1/10", "0/1", "0/1"]),
        ]
        fm_m1 = [
            tri("t_m0", "f_M", [["5/2", "1/1", "3/2"], ["3/1", "1/1", "3/2"], ["3/1", "2/1", "3/2"]], ["1/10", "0/1", "0/1"]),
            tri("t_m1", "f_M", [["5/2", "1/1", "3/2"], ["3/1", "2/1", "3/2"], ["5/2", "2/1", "3/2"]], ["1/10", "0/1", "0/1"]),
        ]

    seg = [{"lo": "-1/1", "hi": "1/1", "lo_closed": True, "hi_closed": True}]
    m["modes"] = [
        {"id": "m0", "edit_cost": "0/1", "t_closed": ["-1/1", "1/1"], "segments": seg, "triangles": fa_tris + fb_tris + fm_m0},
        {"id": "m1", "edit_cost": "1/2", "t_closed": ["-1/1", "1/1"], "segments": seg, "triangles": fa_tris + fb_tris + fm_m1},
    ]
    m["reference"] = {"mode": "m0", "t": "0/1", "sigma": "1/2"}
    m["fracture_catalog"] = {
        "f_A": {"role": "ANCHOR_A", "area_m2": 1.0, "char_radius_m": 0.56, "center_local_m": [1.0, 1.5, 1.5], "original_hole_count": 0},
        "f_B": {"role": "ANCHOR_B", "area_m2": 1.0, "char_radius_m": 0.56, "center_local_m": [3.0, 1.5, 1.5], "original_hole_count": 0},
        "f_M": {"role": "INTERIOR_F1", "area_m2": 0.5, "char_radius_m": 0.4, "center_local_m": [2.0, 1.5, 1.5], "original_hole_count": 0},
    }
    m["mode_semantics"] = {
        "m0": {"description": "branch_left", "edit_cost": "0/1", "feature_assignments": {"f_M": "BRANCH_A"}},
        "m1": {"description": "branch_right", "edit_cost": "1/2", "feature_assignments": {"f_M": "BRANCH_B"}},
    }
    m["quantization_contract"]["hole_preservation_audit"] = []
    return m


def _inject_isolated_fracture(base_model: Dict[str, Any]) -> Dict[str, Any]:
    """Inject a valid, non-degenerate, completely isolated irrelevant fracture into a real AMBIGUOUS model."""
    m = copy.deepcopy(base_model)
    m["snapshot_id"] = base_model["snapshot_id"] + "__nc_isolated"
    iso_id = "perozzo_isolated_irrelevant_dummy"
    # Place 2 non-degenerate triangles well inside Perozzo world bbox but 15m away from the cluster!
    # Check cluster_local_origin so origin + pt stays safely inside [446790..447070, 4890360..4890730, 45..210].
    origin = [float(solver.q_exact(c)) for c in m["real_provenance"]["cluster_local_origin"]]
    # Find safe offset +15m or -15m on X axis
    dx = 15 if origin[0] + 18 < 447065 else -15
    v0 = [f"{dx}/1", "0/1", "0/1"]
    v1 = [f"{dx + 1}/1", "0/1", "0/1"]
    v2 = [f"{dx + 1}/1", "1/1", "0/1"]
    v3 = [f"{dx}/1", "1/1", "0/1"]
    iso_tris = [
        {"id": "t_iso_0", "component": iso_id, "vertices": [v0, v1, v2], "v": ["0/1", "0/1", "0/1"], "label": "CONDITIONED_DISK_TRI"},
        {"id": "t_iso_1", "component": iso_id, "vertices": [v0, v2, v3], "v": ["0/1", "0/1", "0/1"], "label": "CONDITIONED_DISK_TRI"},
    ]
    for mode in m["modes"]:
        mode["triangles"].extend(copy.deepcopy(iso_tris))
    m["fracture_catalog"][iso_id] = {
        "area_m2": 1.0,
        "center_local_m": [float(dx) + 0.5, 0.5, 0.0],
        "char_radius_m": 0.564,
        "dip_deg": 0.0,
        "dip_direction_deg": 0.0,
        "dxf_binding": None,
        "normal_nominal": [0.0, 0.0, 1.0],
        "original_hole_count": 0,
        "preserved_hole_count": 0,
        "residual_sigma_m": 0.01,
        "role": "INTERIOR_ISOLATED_DUMMY",
        "site_id": m["real_provenance"]["site_id"],
        "source_id": "m2_negative_control_isolated_injection",
        "velocity_nominal_m": [0.0, 0.0, 0.0],
        "velocity_sign": 0,
    }
    return m


def run_negative_controls(
    register_path: Path,
    run_dir: Path,
    scratch_dir: Path,
) -> Dict[str, Any]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    reg = solver.load_strict_json(register_path)
    reg_dir = register_path.parent
    by_cid = {c["cluster_id"]: c for c in reg["clusters"]}
    cases_dir = run_dir / "cases"

    controls: List[Dict[str, Any]] = []

    # ---- M2-N1a & M2-N1b: Cross-t and Cross-mode Phantom Union Paths ----
    base_no_hole = solver.load_strict_json(reg_dir / by_cid["jardim_c03"]["model_path"])
    for nc_id, cross_mode, desc in (
        ("M2-N1a", False, "真实坐标域异时(cross-t)并集假通路：单看并集连通，但同一(m,t)均断，篡改证书伪报连通必须非零拒收"),
        ("M2-N1b", True, "真实坐标域异模式(cross-mode)并集假通路：单看并集连通，但同一(m,t)均断，篡改证书伪报连通必须非零拒收"),
    ):
        pm = _make_phantom_union_real_model(base_no_hole, cross_mode=cross_mode)
        pm_path = scratch_dir / f"{nc_id}_model.json"
        solver.write_json(pm_path, pm)
        cert, _ = solver.solve_real_model(pm, cluster_id=nc_id, verify_source_sha=False)
        solver_correct = (
            cert["status"] == "CERTIFIED_DISCONNECTED"
            and cert["exists_connected"] is False
            and cert["nominal_summary"]["naive_union_connected"] is True
            and cert["nominal_summary"]["phantom_union_path"] is True
        )
        # Forge a fake CERTIFIED_CONNECTED certificate using the naive union path!
        forged = copy.deepcopy(cert)
        forged["status"] = "CERTIFIED_CONNECTED"
        forged["exists_connected"] = True
        forged["all_connected"] = True
        forged["connected_witness"] = {
            "mode": "m0",
            "t": "0/1",
            "path": cert["nominal_summary"]["naive_union_path"],
        }
        forged_path = scratch_dir / f"{nc_id}_forged_cert.json"
        solver.write_json(forged_path, forged)
        proc = _run_py_no_window([str(HERE / "solve_real_clusters.py"), "--verify-cert", str(pm_path), str(forged_path)])
        passed = bool(solver_correct and proc["exit_code"] != 0)
        controls.append(
            {
                "id": nc_id,
                "description": desc,
                "solver_certified_disconnected": cert["status"] == "CERTIFIED_DISCONNECTED",
                "naive_union_connected": cert["nominal_summary"]["naive_union_connected"],
                "phantom_union_path_detected": cert["nominal_summary"]["phantom_union_path"],
                "forged_union_cert_exit_code": proc["exit_code"],
                "forged_union_cert_stdout": proc["stdout"],
                "cmd": proc["cmd"],
                "rejected_with_teeth": passed,
            }
        )

    # ---- M2-N2: Tamper connected_witness of AMBIGUOUS cluster to invalid t ----
    m_c02_path = reg_dir / by_cid["jardim_c02"]["model_path"]
    cert_c02 = solver.load_strict_json(cases_dir / "jardim_c02" / "certification.json")
    forged_n2 = copy.deepcopy(cert_c02)
    # In jardim_c02, m0 excludes (-3/5, -2/5), so t = -1/2 is strictly infeasible!
    forged_n2["connected_witness"]["mode"] = "m0"
    forged_n2["connected_witness"]["t"] = "-1/2"
    n2_path = scratch_dir / "M2-N2_tampered_witness_cert.json"
    solver.write_json(n2_path, forged_n2)
    proc_n2 = _run_py_no_window([str(HERE / "solve_real_clusters.py"), "--verify-cert", str(m_c02_path), str(n2_path)])
    controls.append(
        {
            "id": "M2-N2",
            "description": "将真实 AMBIGUOUS 簇 jardim_c02 的 connected_witness 篡改为非法排除带内的 t=-1/2，本地证书自检必须非零拒收",
            "exit_code": proc_n2["exit_code"],
            "stdout": proc_n2["stdout"],
            "cmd": proc_n2["cmd"],
            "rejected_with_teeth": proc_n2["exit_code"] != 0,
        }
    )

    # ---- M2-N3: Forge active recommendation on CERTIFIED_CONNECTED cluster perozzo_c04 ----
    m_p04_path = reg_dir / by_cid["perozzo_c04"]["model_path"]
    c_p04_path = cases_dir / "perozzo_c04" / "certification.json"
    ac_p04 = solver.load_strict_json(cases_dir / "perozzo_c04" / "active_check.json")
    natural_refused = ac_p04["status"] == "ALREADY_CERTIFIED" and len(ac_p04["ranked_fractures"]) == 0
    forged_n3 = copy.deepcopy(ac_p04)
    forged_n3["status"] = "ACTIVE_RECOMMENDATIONS_READY"
    forged_n3["ranked_fractures"] = [
        {"fracture_id": "perozzo_row_0687", "rank": 1, "is_informative": True, "witness_diff_role": "FAKE"}
    ]
    n3_path = scratch_dir / "M2-N3_forged_active_on_certified.json"
    solver.write_json(n3_path, forged_n3)
    proc_n3 = _run_py_no_window(
        [str(HERE / "active_identifiability.py"), "--verify-active", str(m_p04_path), str(c_p04_path), str(n3_path)]
    )
    controls.append(
        {
            "id": "M2-N3",
            "description": "对真实 CERTIFIED_CONNECTED 簇 perozzo_c04 伪造主动复核推荐列表，守卫必须非零拒收",
            "natural_active_status": ac_p04["status"],
            "natural_ranked_empty": natural_refused,
            "exit_code": proc_n3["exit_code"],
            "stdout": proc_n3["stdout"],
            "cmd": proc_n3["cmd"],
            "rejected_with_teeth": bool(natural_refused and proc_n3["exit_code"] != 0),
        }
    )

    # ---- M2-N4: Inject isolated irrelevant fracture into real AMBIGUOUS cluster perozzo_c02 ----
    m_p02 = solver.load_strict_json(reg_dir / by_cid["perozzo_c02"]["model_path"])
    m_n4 = _inject_isolated_fracture(m_p02)
    m_n4_path = scratch_dir / "M2-N4_model_with_isolated.json"
    solver.write_json(m_n4_path, m_n4)
    cert_n4, wbm_n4 = solver.solve_real_model(m_n4, cluster_id="M2-N4", verify_source_sha=False)
    c_n4_path = scratch_dir / "M2-N4_cert.json"
    solver.write_json(c_n4_path, cert_n4)
    ac_n4 = ai.build_active_check(m_n4, cert_n4, windows_by_mode=wbm_n4)
    iso_row = next(r for r in ac_n4["ranked_fractures"] if r["fracture_id"] == "perozzo_isolated_irrelevant_dummy")
    first_row = ac_n4["ranked_fractures"][0]
    natural_zero_and_behind = (
        iso_row["witness_diff_role"] == "ISOLATED_IRRELEVANT_FRACTURE"
        and iso_row["is_informative"] is False
        and iso_row["minimax_reduction_fraction"] == "0/1"
        and iso_row["resolved_measure_fraction"] == "0/1"
        and iso_row["certified_branch_fraction"] == "0/1"
        and first_row["is_informative"] is True
        and iso_row["rank"] > first_row["rank"]
    )
    # Tamper active_check to illegally promote isolated fracture to rank 1 with fake gain
    forged_n4 = copy.deepcopy(ac_n4)
    forged_n4["ranked_fractures"].remove(
        next(r for r in forged_n4["ranked_fractures"] if r["fracture_id"] == "perozzo_isolated_irrelevant_dummy")
    )
    fake_iso = copy.deepcopy(iso_row)
    fake_iso["rank"] = 1
    fake_iso["is_informative"] = True
    fake_iso["minimax_reduction_fraction"] = "1/2"
    forged_n4["ranked_fractures"].insert(0, fake_iso)
    n4_ac_path = scratch_dir / "M2-N4_tampered_isolated_rank1.json"
    solver.write_json(n4_ac_path, forged_n4)
    proc_n4 = _run_py_no_window(
        [str(HERE / "active_identifiability.py"), "--verify-active", str(m_n4_path), str(c_n4_path), str(n4_ac_path)]
    )
    controls.append(
        {
            "id": "M2-N4",
            "description": "在真实 AMBIGUOUS 簇 perozzo_c02 注入完全孤立无关裂隙：自然收益严格为 0/1 且排在有效裂隙之后；若强行篡改提升至第 1 名，守卫必须非零拒收",
            "isolated_natural_rank": iso_row["rank"],
            "total_fractures": len(ac_n4["ranked_fractures"]),
            "isolated_natural_mrf": iso_row["minimax_reduction_fraction"],
            "isolated_natural_rmf": iso_row["resolved_measure_fraction"],
            "isolated_natural_cbf": iso_row["certified_branch_fraction"],
            "natural_zero_and_behind_effective": natural_zero_and_behind,
            "exit_code": proc_n4["exit_code"],
            "stdout": proc_n4["stdout"],
            "cmd": proc_n4["cmd"],
            "rejected_with_teeth": bool(natural_zero_and_behind and proc_n4["exit_code"] != 0),
        }
    )

    # ---- M2-N5: Tamper post_check_outcome / collapse_tree branch to contradict exact subdomain ----
    ac_c02 = solver.load_strict_json(cases_dir / "jardim_c02" / "active_check.json")
    forged_n5 = copy.deepcopy(ac_c02)
    first_tree_fid = forged_n5["collapse_tree"]["top_fracture_ids"][0]
    first_br = forged_n5["collapse_tree"]["fractures"][first_tree_fid]["branches"][0]
    orig_st = first_br["post_status"]
    first_br["post_status"] = "CERTIFIED_DISCONNECTED" if orig_st != "CERTIFIED_DISCONNECTED" else "CERTIFIED_CONNECTED"
    first_br["collapsed_to_certified"] = True
    n5_path = scratch_dir / "M2-N5_tampered_collapse_branch.json"
    solver.write_json(n5_path, forged_n5)
    proc_n5 = _run_py_no_window(
        [
            str(HERE / "active_identifiability.py"),
            "--verify-active",
            str(m_c02_path),
            str(cases_dir / "jardim_c02" / "certification.json"),
            str(n5_path),
        ]
    )
    controls.append(
        {
            "id": "M2-N5",
            "description": f"篡改 jardim_c02 collapse_tree 首推裂隙分支 post_status（原 {orig_st} → 伪造 {first_br['post_status']}），一致性守卫必须非零拒收",
            "exit_code": proc_n5["exit_code"],
            "stdout": proc_n5["stdout"],
            "cmd": proc_n5["cmd"],
            "rejected_with_teeth": proc_n5["exit_code"] != 0,
        }
    )

    # ---- M2-N6a / M2-N6b / M2-N6c: Illegal / cross-frame / empty-denominator models ----
    # N6a: Synthetic scope impersonation
    m_n6a = copy.deepcopy(m_p02)
    m_n6a["scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
    n6a_in = scratch_dir / "M2-N6a_synthetic_impersonation.json"
    n6a_out = scratch_dir / "M2-N6a_out.json"
    solver.write_json(n6a_in, m_n6a)
    proc_n6a = _run_py_no_window(
        [str(HERE / "solve_real_clusters.py"), "--model", str(n6a_in), "--out", str(n6a_out), "--no-verify-source-sha"]
    )
    env_n6a = solver.load_strict_json(n6a_out)
    controls.append(
        {
            "id": "M2-N6a",
            "description": "输入 scope 冒充合成域（SYNTHETIC_DECLARED_TRIANGLE_UNION）的非法模型，必须非零退出且输出拒收信封",
            "exit_code": proc_n6a["exit_code"],
            "refusal_status": env_n6a["status"],
            "search_complete": env_n6a["search_complete"],
            "stdout": proc_n6a["stdout"],
            "cmd": proc_n6a["cmd"],
            "rejected_with_teeth": bool(proc_n6a["exit_code"] != 0 and env_n6a["search_complete"] is False),
        }
    )

    # N6b: Cross-site coordinate frame contamination
    m_n6b = copy.deepcopy(m_p02)
    m_n6b["real_provenance"]["coordinate_frame"] = "jardim_garcia_quarry_uh_source_coordinates_v1"
    n6b_in = scratch_dir / "M2-N6b_cross_frame.json"
    n6b_out = scratch_dir / "M2-N6b_out.json"
    solver.write_json(n6b_in, m_n6b)
    proc_n6b = _run_py_no_window(
        [str(HERE / "solve_real_clusters.py"), "--model", str(n6b_in), "--out", str(n6b_out), "--no-verify-source-sha"]
    )
    env_n6b = solver.load_strict_json(n6b_out)
    controls.append(
        {
            "id": "M2-N6b",
            "description": "输入跨场地坐标系污染模型（Perozzo 簇挂 Jardim 坐标系），必须非零退出且输出拒收信封",
            "exit_code": proc_n6b["exit_code"],
            "refusal_status": env_n6b["status"],
            "search_complete": env_n6b["search_complete"],
            "stdout": proc_n6b["stdout"],
            "cmd": proc_n6b["cmd"],
            "rejected_with_teeth": bool(proc_n6b["exit_code"] != 0 and env_n6b["search_complete"] is False),
        }
    )

    # N6c: Empty feasible segments across all modes (empty denominator!)
    m_n6c = copy.deepcopy(m_p02)
    for md in m_n6c["modes"]:
        md["segments"] = []
    n6c_in = scratch_dir / "M2-N6c_empty_segments.json"
    n6c_out = scratch_dir / "M2-N6c_out.json"
    solver.write_json(n6c_in, m_n6c)
    proc_n6c = _run_py_no_window(
        [str(HERE / "solve_real_clusters.py"), "--model", str(n6c_in), "--out", str(n6c_out), "--no-verify-source-sha"]
    )
    env_n6c = solver.load_strict_json(n6c_out)
    controls.append(
        {
            "id": "M2-N6c",
            "description": "输入全模式空合法段（空分母）模型，必须非零退出且输出拒收信封，不产伪 PASS",
            "exit_code": proc_n6c["exit_code"],
            "refusal_status": env_n6c["status"],
            "search_complete": env_n6c["search_complete"],
            "stdout": proc_n6c["stdout"],
            "cmd": proc_n6c["cmd"],
            "rejected_with_teeth": bool(proc_n6c["exit_code"] != 0 and env_n6c["search_complete"] is False),
        }
    )

    # ---- M2-N7: Genuine runtime budget exceeded (jardim_c01: 5024 > 4096 cells) must refuse cleanly ----
    m_c01_path = reg_dir / by_cid["jardim_c01"]["model_path"]
    n7_out = scratch_dir / "M2-N7_jardim_c01_budget_refusal.json"
    proc_n7 = _run_py_no_window(
        [str(HERE / "solve_real_clusters.py"), "--model", str(m_c01_path), "--out", str(n7_out), "--no-verify-source-sha"]
    )
    env_n7 = solver.load_strict_json(n7_out)
    controls.append(
        {
            "id": "M2-N7",
            "description": "真实簇 jardim_c01 超出声明 budget.cells=4096（需 5024 cells）：CLI 必须非零退出(4)并输出 BUDGET_EXCEEDED 拒收信封，绝不静默扩预算或截断细胞",
            "exit_code": proc_n7["exit_code"],
            "refusal_status": env_n7["status"],
            "search_complete": env_n7["search_complete"],
            "budget_telemetry": env_n7["budget_telemetry"],
            "stdout": proc_n7["stdout"],
            "cmd": proc_n7["cmd"],
            "rejected_with_teeth": bool(
                proc_n7["exit_code"] == 4
                and env_n7["status"] == "BUDGET_EXCEEDED"
                and env_n7["search_complete"] is False
                and env_n7["cells"] == []
            ),
        }
    )

    n_total = len(controls)
    n_rejected = sum(1 for c in controls if c["rejected_with_teeth"])
    return {
        "schema": "m2-negative-controls/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "total_controls": n_total,
        "correctly_rejected_with_teeth": n_rejected,
        "all_rejected": n_total == n_rejected and n_total >= 6,
        "controls": controls,
    }


def audit_m1_final_sha_alignment(register_path: Path, run_dir: Path) -> Dict[str, Any]:
    """Align M2 run outputs against finalized M1 result.json and cluster_register.json SHAs."""
    m1_dir = register_path.parent
    m1_res_path = m1_dir / "result.json"
    if not m1_res_path.is_file():
        return {"ok": False, "reason": "M1 result.json missing (M1 not finalized)"}
    m1_res = solver.load_strict_json(m1_res_path)
    ev_by_file = {row["file"]: row["sha256"] for row in m1_res.get("evidence", [])}
    reg = solver.load_strict_json(register_path)
    reg_sha = solver.file_sha256(register_path)
    rnf_sha = solver.file_sha256(m1_dir / "real_network_format.py")
    mismatches: List[str] = []

    if m1_res.get("status") != "PASS":
        mismatches.append(f"M1 result.json status={m1_res.get('status')!r} != 'PASS'")
    if ev_by_file.get("cluster_register.json") != reg_sha:
        mismatches.append("cluster_register.json SHA mismatch vs M1 result.json")
    if ev_by_file.get("real_network_format.py") != rnf_sha:
        mismatches.append("real_network_format.py SHA mismatch vs M1 result.json")

    cases_dir = run_dir / "cases"
    aligned_clusters = 0
    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        rel_m = entry["model_path"]
        disk_sha = solver.file_sha256(m1_dir / rel_m)
        if ev_by_file.get(rel_m) != disk_sha:
            mismatches.append(f"{cid}: disk model.json SHA != M1 result.json")
        if entry["model_file_sha256"] != disk_sha:
            mismatches.append(f"{cid}: disk model.json SHA != cluster_register.json")
        cert = solver.load_strict_json(cases_dir / cid / "certification.json")
        ac = solver.load_strict_json(cases_dir / cid / "active_check.json")
        if cert["model_file_sha256"] != disk_sha:
            mismatches.append(f"{cid}: certification.json#model_file_sha256 mismatch")
        if cert["input_sha256"] != entry["model_canonical_sha256"]:
            mismatches.append(f"{cid}: certification.json#input_sha256 mismatch")
        if ac["input_sha256"] != entry["model_canonical_sha256"]:
            mismatches.append(f"{cid}: active_check.json#input_sha256 mismatch")
        aligned_clusters += 1

    return {
        "ok": len(mismatches) == 0,
        "m1_status": m1_res.get("status"),
        "m1_result_sha256": solver.file_sha256(m1_res_path),
        "m1_cluster_register_sha256": reg_sha,
        "m1_real_network_format_sha256": rnf_sha,
        "aligned_clusters": aligned_clusters,
        "mismatches": mismatches,
    }


def audit_no_pycache_pollution() -> Dict[str, Any]:
    """Ensure M2 created zero __pycache__ folders inside M1 or frozen TC/triangle_contact dirs."""
    watch_dirs = [
        M1_DIR,
        TC_DIR,
        REPO_ROOT / "results" / "w180_technical_route_20261117" / "execution_next" / "triangle_contact",
        REPO_ROOT / "results" / "w180_technical_route_20261117" / "execution_next" / "kernel_r3",
    ]
    found: List[str] = []
    for d in watch_dirs:
        if d.is_dir():
            for p in d.rglob("__pycache__"):
                found.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return {"ok": len(found) == 0, "pycache_dirs_found": found}


def verify_all(
    register_path: Path,
    run_dir: Path,
    *,
    scratch_recompute_top3: bool = True,
) -> Dict[str, Any]:
    reg = solver.load_strict_json(register_path)
    reg_dir = register_path.parent
    cases_dir = run_dir / "cases"
    t0 = time.perf_counter()

    # 1. Verify every cluster certification.json & active_check.json (including scratch recompute on AMBIGUOUS!)
    per_cluster_checks: List[Dict[str, Any]] = []
    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        m_path = reg_dir / entry["model_path"]
        c_path = cases_dir / cid / "certification.json"
        a_path = cases_dir / cid / "active_check.json"
        m_raw = solver.load_strict_json(m_path)
        cert = solver.load_strict_json(c_path)
        ac = solver.load_strict_json(a_path)

        sha_ok = cert["input_sha256"] == entry["model_canonical_sha256"] and ac["input_sha256"] == entry["model_canonical_sha256"]
        v_cert = solver.verify_certification_local(m_raw, cert, recheck_windows_from_scratch=False)
        v_ac = ai.verify_active_check_local(
            m_raw,
            cert,
            ac,
            recompute_from_scratch_branches=bool(scratch_recompute_top3 and cert["status"] == "AMBIGUOUS_WITH_VERIFIED_WITNESSES"),
        )
        ok = bool(sha_ok and v_cert["ok"] and v_ac["ok"])
        per_cluster_checks.append(
            {
                "cluster_id": cid,
                "status": cert["status"],
                "active_status": ac["status"],
                "sha_aligned_with_m1_register": sha_ok,
                "cert_check": v_cert,
                "active_check": v_ac,
                "ok": ok,
            }
        )
        if not ok:
            raise RuntimeError(f"verification failed on {cid}: {per_cluster_checks[-1]}")

    # 2. Exact equivalence audit vs raw TC_network/compile_network.py
    tc_eq = audit_raw_tc_equivalence(register_path, run_dir, ("perozzo_c05", "perozzo_c02"))

    # 3. Run >=6 teeth-bearing negative controls
    nc = run_negative_controls(register_path, run_dir, HERE / "_nc_scratch")
    solver.write_json(HERE / "negative_controls.json", nc)
    if not nc["all_rejected"]:
        raise RuntimeError(f"negative controls failed: {nc['correctly_rejected_with_teeth']}/{nc['total_controls']}")

    # 4. M1 final SHA alignment + Pycache pollution audit
    m1_align = audit_m1_final_sha_alignment(register_path, run_dir)
    if not m1_align["ok"]:
        raise RuntimeError(f"M1 final SHA alignment failed: {m1_align}")
    py_audit = audit_no_pycache_pollution()

    report = {
        "schema": "m2-verification-report/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "positive_clusters_total": len(per_cluster_checks),
        "positive_clusters_passed": sum(1 for r in per_cluster_checks if r["ok"]),
        "m1_final_sha_alignment": m1_align,
        "raw_tc_equivalence_audit": tc_eq,
        "negative_controls_total": nc["total_controls"],
        "negative_controls_rejected": nc["correctly_rejected_with_teeth"],
        "pycache_audit": py_audit,
        "all_passed": bool(
            all(r["ok"] for r in per_cluster_checks)
            and m1_align["ok"]
            and tc_eq["ok"]
            and nc["all_rejected"]
            and py_audit["ok"]
        ),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "per_cluster_checks": per_cluster_checks,
    }
    solver.write_json(HERE / "verification_report.json", report)
    return report


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2 End-to-End Verification, Summary Assembly, and Negative Controls")
    parser.add_argument("--register", required=True, help="Path to M1 cluster_register.json")
    parser.add_argument("--run-dir", required=True, help="Target M2 run directory")
    parser.add_argument("--run-pipeline", action="store_true", help="Also execute solve + active + simulate before verifying")
    parser.add_argument("--sim-json", help="Reuse an already computed simulation_detail.json instead of re-running simulation")
    parser.add_argument("--no-scratch-recompute", action="store_true", help="Skip from-scratch branch recomputation")
    args = parser.parse_args(argv)

    reg_path = Path(args.register)
    run_dir = Path(args.run_dir)

    if args.run_pipeline:
        solver.run_batch(reg_path, run_dir, verify_source_sha=True)
        ai.run_batch_active(reg_path, run_dir)
    if args.sim_json:
        sim_res = solver.load_strict_json(Path(args.sim_json))
    else:
        sim_res = sim.run_simulation_batch(reg_path, run_dir)
        solver.write_json(run_dir / "simulation_detail.json", sim_res)
    summary = build_summary_json(reg_path, run_dir, sim_res)
    v_rep = verify_all(reg_path, run_dir, scratch_recompute_top3=not args.no_scratch_recompute)

    print(
        json.dumps(
            {
                "all_passed": v_rep["all_passed"],
                "positive_passed": f"{v_rep['positive_clusters_passed']}/{v_rep['positive_clusters_total']}",
                "m1_sha_aligned": v_rep["m1_final_sha_alignment"]["ok"],
                "negative_controls": f"{v_rep['negative_controls_rejected']}/{v_rep['negative_controls_total']}",
                "status_counts_all": summary["counts"]["status_counts_all"],
                "pycache_clean": v_rep["pycache_audit"]["ok"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if v_rep["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
