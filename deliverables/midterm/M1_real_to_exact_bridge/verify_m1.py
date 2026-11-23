"""M1 Full Verification Suite: Positive Census, Exact Endpoint/Kernel Audit,
7 Classes of Toothed Negative Controls (M1-N1..M1-N7), and Frozen-Dir Integrity Audit.

Strictly obeys AGENTS.md §11: all subprocess invocations use
`creationflags=subprocess.CREATE_NO_WINDOW` (134217728) and `shell=False`.
Never writes to `TC_network/`, `TB_feasible/`, `TE_judge/`, or `data/`.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
INTERPRETER = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
CREATE_NO_WINDOW = 0x08000000  # 134217728

sys.path.insert(0, str(HERE))
import real_network_format as rnf


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_no_window(argv: List[str], cwd: Path, timeout: int = 120) -> Dict[str, Any]:
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(a) for a in argv],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
        shell=False,
    )
    return {
        "cmd": " ".join(str(a) for a in argv),
        "argv": [str(a) for a in argv],
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "creationflags": CREATE_NO_WINDOW,
        "shell": False,
    }


def load_frozen_tc_validator():
    """Load frozen TC_network/network_format.py read-only (no bytecode written)."""
    tc_path = REPO / "results/w180_technical_route_20261117/execution_next/dispatch_20261118/TC_network/network_format.py"
    spec = importlib.util.spec_from_file_location("frozen_tc_network_format_readonly", tc_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_exact_endpoint_attachment(model: Dict[str, Any], anchor_a: str, anchor_b: str) -> Tuple[bool, str]:
    pt_a = rnf.vector(model["endpoints"]["A"])
    pt_b = rnf.vector(model["endpoints"]["B"])
    for mode in model["modes"]:
        a_tris = [t for t in mode["triangles"] if t["component"] == anchor_a]
        b_tris = [t for t in mode["triangles"] if t["component"] == anchor_b]
        if not a_tris or not b_tris:
            return False, f"mode {mode['id']} missing anchor triangles"
        a_ok = any(
            rnf.point_in_triangle_projected_3d(
                pt_a,
                rnf.vector(t["vertices"][0]),
                rnf.vector(t["vertices"][1]),
                rnf.vector(t["vertices"][2]),
                max_plane_dist_sq=Q(0),  # exact coplanarity == 0
            )
            for t in a_tris
        )
        b_ok = any(
            rnf.point_in_triangle_projected_3d(
                pt_b,
                rnf.vector(t["vertices"][0]),
                rnf.vector(t["vertices"][1]),
                rnf.vector(t["vertices"][2]),
                max_plane_dist_sq=Q(0),  # exact coplanarity == 0
            )
            for t in b_tris
        )
        if not a_ok:
            return False, f"mode {mode['id']}: endpoint A not exactly on anchor {anchor_a}"
        if not b_ok:
            return False, f"mode {mode['id']}: endpoint B not exactly on anchor {anchor_b}"
    return True, "exact_zero_distance_attached_all_modes"


def audit_frozen_dirs(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Verify ZERO historical bytes modified or deleted in TC_network/, TB_feasible/,
    TE_judge/, and data/external/perozzo_2024/ relative to pre-build baseline.
    Separately records any append-only new run dirs created externally by parallel sessions."""
    disp = REPO / "results/w180_technical_route_20261117/execution_next/dispatch_20261118"
    drifts: List[Dict[str, Any]] = []
    external_appends: List[str] = []
    checked_files = 0

    for key in ("TC_network", "TB_feasible", "TE_judge"):
        base_entry = baseline[key]
        dpath = disp / key
        # 1. Verify EVERY historical baseline .py and top-level metadata file is byte-identical
        for group_name in ("py_files", "top_meta_files"):
            for relp, expected in base_entry[group_name].items():
                fp = REPO / relp
                checked_files += 1
                if not fp.is_file():
                    drifts.append({"dir": key, "kind": "historical_file_deleted", "file": relp})
                    continue
                actual_bytes = fp.stat().st_size
                actual_sha = rnf.sha_file(fp)
                if actual_bytes != expected["bytes"] or actual_sha != expected["sha256"]:
                    drifts.append({
                        "dir": key,
                        "kind": "historical_file_modified",
                        "file": relp,
                        "expected_sha256": expected["sha256"],
                        "actual_sha256": actual_sha,
                    })
        # 2. Check for any newly added .py or top-level files; allow append-only external runs/ subdirs
        #    created by parallel sessions while still flagging any unexpected edits outside runs/.
        known_py = set(base_entry["py_files"].keys())
        known_top = set(base_entry["top_meta_files"].keys())
        for p in sorted(dpath.rglob("*")):
            if not p.is_file():
                continue
            relp = str(p.relative_to(REPO)).replace("\\", "/")
            rel_inside = str(p.relative_to(dpath)).replace("\\", "/")
            if p.suffix == ".py" and relp not in known_py:
                if rel_inside.startswith("runs/"):
                    external_appends.append(relp)
                else:
                    drifts.append({"dir": key, "kind": "unexpected_new_py_outside_runs", "file": relp})
            elif p.parent == dpath and p.suffix in (".json", ".md") and relp not in known_top:
                drifts.append({"dir": key, "kind": "unexpected_new_top_file", "file": relp})

    # Check data/external/perozzo_2024 untouched
    cur_data = {
        p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted((REPO / "data/external/perozzo_2024").iterdir())
        if p.is_file()
    }
    checked_files += len(cur_data)
    if cur_data != baseline["data_perozzo_files"]:
        drifts.append({"dir": "data/external/perozzo_2024", "kind": "data_file_bytes_or_mtime_drift"})

    return {
        "schema": "m1-frozen-dirs-audit/v1",
        "at_utc": utc(),
        "checked_historical_files": checked_files,
        "drifts_count": len(drifts),
        "drifts": drifts,
        "concurrent_external_runs_appends": external_appends,
        "zero_modification_confirmed": len(drifts) == 0,
    }


def run_negative_controls(jardim_sample_path: Path, perozzo_sample_path: Path, perozzo_large_path: Path) -> Dict[str, Any]:
    scratch = HERE / "_nc_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    validator_script = HERE / "real_network_format.py"

    jardim_base = json.loads(jardim_sample_path.read_text(encoding="utf-8"))
    perozzo_base = json.loads(perozzo_sample_path.read_text(encoding="utf-8"))
    perozzo_large = json.loads(perozzo_large_path.read_text(encoding="utf-8"))

    def write_case(name: str, obj: Dict[str, Any]) -> Path:
        p = scratch / f"{name}.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    controls_spec: List[Tuple[str, str, Path, List[str], List[str]]] = []

    # M1-N1a: Tampered xlsx SHA256 in source_files
    n1a = copy.deepcopy(perozzo_base)
    orig_sha = n1a["real_provenance"]["source_files"][0]["sha256"]
    flipped = ("0" if orig_sha[0] != "0" else "1") + orig_sha[1:]
    n1a["real_provenance"]["source_files"][0]["sha256"] = flipped
    controls_spec.append((
        "M1-N1a",
        "Tamper 1 hex character of Perozzo xlsx SHA256 -> must fail with SOURCE_SHA256_MISMATCH",
        write_case("nc_m1_n1a_tampered_xlsx_sha", n1a),
        ["--verify-source-sha"],
        ["SOURCE_SHA256_MISMATCH"],
    ))

    # M1-N1b: 1-byte tampered copy of Jardim project.json referenced with original SHA
    tampered_proj = scratch / "tampered_project.json"
    tampered_proj.write_bytes(b'{"tampered": true}\n')
    n1b = copy.deepcopy(jardim_base)
    n1b["real_provenance"]["source_files"][0]["path"] = str(tampered_proj.relative_to(REPO)).replace("\\", "/")
    controls_spec.append((
        "M1-N1b",
        "Point Jardim cluster to tampered project.json while retaining original SHA/bytes -> hard reject",
        write_case("nc_m1_n1b_tampered_project_file", n1b),
        ["--verify-source-sha"],
        ["SOURCE_BYTES_MISMATCH", "SOURCE_SHA256_MISMATCH"],
    ))

    # M1-N2: Silently relabel real-derived scope to SYNTHETIC_DECLARED_TRIANGLE_UNION
    n2 = copy.deepcopy(jardim_base)
    n2["scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
    controls_spec.append((
        "M1-N2",
        "Change real cluster scope to SYNTHETIC_DECLARED_TRIANGLE_UNION -> hard reject (track separation)",
        write_case("nc_m1_n2_synthetic_scope_spoof", n2),
        [],
        ["REAL_SCOPE_VIOLATION"],
    ))

    # M1-N3a: Clear ASSUMED_NOT_OBSERVED tag and empty not_claimed
    n3a = copy.deepcopy(perozzo_base)
    n3a["assumption_register"]["assumed_edges_tag"] = ""
    n3a["assumption_register"]["not_claimed"] = []
    controls_spec.append((
        "M1-N3a",
        "Empty assumption_register.assumed_edges_tag and not_claimed -> hard reject",
        write_case("nc_m1_n3a_empty_assumptions", n3a),
        [],
        ["ASSUMED_NOT_OBSERVED"],
    ))

    # M1-N3b: Keep ASSUMED_NOT_OBSERVED but strip hydraulic & subsurface disclaimers from not_claimed
    n3b = copy.deepcopy(perozzo_base)
    n3b["assumption_register"]["not_claimed"] = ["only a harmless placeholder note", "second placeholder note"]
    controls_spec.append((
        "M1-N3b",
        "Remove mandatory subsurface/hydraulic disclaimers from not_claimed -> hard reject",
        write_case("nc_m1_n3b_missing_disclaimers", n3b),
        [],
        ["not_claimed"],
    ))

    # M1-N4a: Stuff Jardim coordinates into a Perozzo cluster (cross-site coordinate mixing)
    n4a = copy.deepcopy(perozzo_base)
    # Jardim world point ~ [700, 553, 628] expressed in Perozzo local coords relative to Perozzo origin
    p_orig = rnf.vector(n4a["real_provenance"]["cluster_local_origin"])
    j_world = (Q(700), Q(553), Q(628))
    j_in_p = rnf.sub(j_world, p_orig)
    n4a["modes"][0]["triangles"][0]["vertices"][0] = [rnf.qs(x) for x in j_in_p]
    controls_spec.append((
        "M1-N4a",
        "Inject Jardim world coordinates [700,553,628] into Perozzo cluster -> cross-site bbox hard reject",
        write_case("nc_m1_n4a_cross_site_coords", n4a),
        [],
        ["CROSS_SITE_COORDINATE_OUT_OF_SITE_BBOX"],
    ))

    # M1-N4b: Swap Jardim coordinate_frame label into Perozzo cluster
    n4b = copy.deepcopy(perozzo_base)
    n4b["real_provenance"]["coordinate_frame"] = "jardim_garcia_quarry_uh_source_coordinates_v1"
    controls_spec.append((
        "M1-N4b",
        "Swap Jardim coordinate_frame into Perozzo cluster -> cross-site frame hard reject",
        write_case("nc_m1_n4b_cross_site_frame", n4b),
        [],
        ["CROSS_SITE_FRAME_MISMATCH"],
    ))

    # M1-N5a: Replace Jardim holed fragment (frag_c8f4...) in m0 with a convex hull triangulation filling the real hole!
    n5a = copy.deepcopy(jardim_base)
    hole_entry = n5a["quantization_contract"]["hole_preservation_audit"][0]
    holed_fid = hole_entry["fragment_id"]
    m0_tris = n5a["modes"][0]["triangles"]
    comp_tris = [t for t in m0_tris if t["component"] == holed_fid]
    other_tris = [t for t in m0_tris if t["component"] != holed_fid]
    # Collect all unique 3D vertices of holed_fid and fan-triangulate its 2D convex hull in its plane!
    import numpy as np
    from shapely.geometry import MultiPoint

    pts3 = []
    seen_v = set()
    for t in comp_tris:
        for v in t["vertices"]:
            key = tuple(v)
            if key not in seen_v:
                seen_v.add(key)
                pts3.append((v, np.array([float(rnf.exact(c)) for c in v], dtype=np.float64)))
    p0 = pts3[0][1]
    t0_pts = [np.array([float(rnf.exact(c)) for c in v], dtype=np.float64) for v in comp_tris[0]["vertices"]]
    n_vec = np.cross(t0_pts[1] - t0_pts[0], t0_pts[2] - t0_pts[0])
    n_vec = n_vec / float(np.linalg.norm(n_vec))
    u_vec = t0_pts[1] - t0_pts[0]
    u_vec = u_vec / float(np.linalg.norm(u_vec))
    v_vec = np.cross(n_vec, u_vec)
    pts2_map = {}
    pts2_list = []
    for raw_v, p_np in pts3:
        d = p_np - p0
        xy = (round(float(np.dot(d, u_vec)), 7), round(float(np.dot(d, v_vec)), 7))
        pts2_map[xy] = raw_v
        pts2_list.append(xy)
    hull = MultiPoint(pts2_list).convex_hull
    hull_coords = list(hull.exterior.coords)[:-1]
    hull_raw_verts = [pts2_map[(round(x, 7), round(y, 7))] for x, y in hull_coords]
    hull_tris = []
    vel_orig = comp_tris[0]["v"]
    for k in range(1, len(hull_raw_verts) - 1):
        hull_tris.append({
            "id": f"hull_fill_{k:02d}",
            "component": holed_fid,
            "vertices": [hull_raw_verts[0], hull_raw_verts[k], hull_raw_verts[k + 1]],
            "v": vel_orig,
            "label": "CONVEX_HULL_HOLE_FILL_ADVERSARY",
        })
    n5a["modes"][0]["triangles"] = other_tris + hull_tris
    controls_spec.append((
        "M1-N5a",
        f"Replace Jardim real-hole fragment {holed_fid} with convex hull filling its hole -> topology/area guard hard reject",
        write_case("nc_m1_n5a_convex_hull_hole_fill", n5a),
        [],
        ["HOLE_FILLED_AREA_GUARD", "HOLE_AREA_MISMATCH", "HOLE_FILLED_PROBE_COVERED"],
    ))

    # M1-N5b: Subtle adversary who patches a single triangle over the hole interior probe and spoofs declared area
    n5b = copy.deepcopy(jardim_base)
    probe0 = rnf.vector(hole_entry["hole_interior_probe_points_local"][0])
    # Build a tiny triangle centered directly on probe0 in the fragment's plane
    eps = Q(1, 50)
    t_ref = comp_tris[0]
    a_r, b_r, c_r = (rnf.vector(v) for v in t_ref["vertices"])
    e1 = rnf.sub(b_r, a_r)
    e2 = rnf.sub(c_r, a_r)
    # Normalize roughly by rational scale
    e1_s = tuple(x * Q(1, 20) for x in e1)
    e2_s = tuple(x * Q(1, 20) for x in e2)
    pa = tuple(probe0[i] - e1_s[i] - e2_s[i] for i in range(3))
    pb = tuple(probe0[i] + Q(2) * e1_s[i] - e2_s[i] for i in range(3))
    pc = tuple(probe0[i] - e1_s[i] + Q(2) * e2_s[i] for i in range(3))
    patch_tri = {
        "id": "hole_patch_adversary",
        "component": holed_fid,
        "vertices": [[rnf.qs(x) for x in pa], [rnf.qs(x) for x in pb], [rnf.qs(x) for x in pc]],
        "v": t_ref["v"],
        "label": "SUBTLE_HOLE_PATCH",
    }
    n5b["modes"][0]["triangles"].append(patch_tri)
    # Even if adversary updates declared triangulated_net_area_m2 to dodge area mismatch!
    import math
    patch_area = 0.5 * math.sqrt(float(rnf.norm2(rnf.cross(rnf.sub(pb, pa), rnf.sub(pc, pa)))))
    n5b["quantization_contract"]["hole_preservation_audit"][0]["triangulated_net_area_m2"] += patch_area
    controls_spec.append((
        "M1-N5b",
        f"Subtle hole-patch adversary covering interior probe of {holed_fid} even with spoofed area -> exact probe guard hard reject",
        write_case("nc_m1_n5b_subtle_hole_probe_cover", n5b),
        [],
        ["HOLE_FILLED_PROBE_COVERED"],
    ))

    # M1-N6a: Triangle budget overflow (> 64 triangles, no silent truncation)
    n6a = copy.deepcopy(perozzo_large)  # 54 triangles -> add 11 copies -> 65 > 64
    base_tri = n6a["modes"][0]["triangles"][0]
    for extra_k in range(11):
        t_dup = copy.deepcopy(base_tri)
        t_dup["id"] = f"overflow_tri_{extra_k:02d}"
        n6a["modes"][0]["triangles"].append(t_dup)
    controls_spec.append((
        "M1-N6a",
        "Inject 65 triangles (>64 budget) in mode m0 -> Exhausted hard reject (no silent truncation)",
        write_case("nc_m1_n6a_triangle_budget_overflow", n6a),
        [],
        ["triangle budget exceeded"],
    ))

    # M1-N6b: Mode budget overflow (> 16 modes, no silent truncation)
    n6b = copy.deepcopy(perozzo_base)  # 6 modes -> expand to 17 modes > 16
    for extra_m in range(6, 17):
        m_dup = copy.deepcopy(n6b["modes"][0])
        m_dup["id"] = f"m{extra_m}"
        m_dup["edit_cost"] = f"{extra_m}/1"
        n6b["modes"].append(m_dup)
        n6b["mode_semantics"][f"m{extra_m}"] = copy.deepcopy(n6b["mode_semantics"]["m0"])
    controls_spec.append((
        "M1-N6b",
        "Inject 17 modes (>16 budget) -> Exhausted hard reject (no silent truncation)",
        write_case("nc_m1_n6b_mode_budget_overflow", n6b),
        [],
        ["mode budget exceeded"],
    ))

    # M1-N7: Rational bit-length budget overflow (> 8192 bits, while text length <= 4096)
    n7 = copy.deepcopy(perozzo_base)
    huge_num = (1 << 8195) + 1  # 8196 bits > 8192, ~2468 decimal chars <= 4096
    n7["modes"][0]["triangles"][0]["vertices"][0][0] = f"{huge_num}/1"
    controls_spec.append((
        "M1-N7",
        "Inject 8196-bit rational coordinate (>8192 bit budget, text <=4096) -> Exhausted bit-budget hard reject",
        write_case("nc_m1_n7_bit_budget_overflow", n7),
        [],
        ["rational bit budget exceeded"],
    ))

    results = []
    all_rejected = True
    for nc_id, desc, fpath, extra_flags, expected_tokens in controls_spec:
        argv = [INTERPRETER, "-B", "-X", "utf8", str(validator_script), "--validate", str(fpath), "--repo-root", str(REPO)] + extra_flags
        res = run_no_window(argv, REPO)
        failed_as_expected = res["exit_code"] != 0 and any(tok in res["stdout"] for tok in expected_tokens)
        if not failed_as_expected:
            all_rejected = False
        first_counterexample = None
        try:
            first_counterexample = json.loads(res["stdout"]) if res["stdout"] else {"raw_stderr": res["stderr"]}
        except json.JSONDecodeError:
            first_counterexample = {"raw_stdout": res["stdout"], "raw_stderr": res["stderr"]}
        results.append({
            "id": nc_id,
            "scenario": desc,
            "mutated_file": str(fpath.relative_to(REPO)).replace("\\", "/"),
            "command": res["cmd"],
            "expected_nonzero_exit": True,
            "actual_exit_code": res["exit_code"],
            "expected_reason_tokens": expected_tokens,
            "rejected_with_teeth": failed_as_expected,
            "first_counterexample": first_counterexample,
        })

    return {
        "schema": "m1-negative-controls/v1",
        "task_id": "M1",
        "at_utc": utc(),
        "total_controls": len(results),
        "rejected_with_teeth": sum(1 for r in results if r["rejected_with_teeth"]),
        "distinct_control_classes": 7,
        "all_passed_with_teeth": all_rejected,
        "controls": results,
    }


def main() -> int:
    freeze = json.loads((HERE / "prereg_freeze_receipt.json").read_text(encoding="utf-8"))
    prereg_sha = rnf.sha_file(HERE / "prereg.json")
    assert prereg_sha == freeze["prereg_file_sha256"], "PREREG_SHA_DRIFT"

    reg = json.loads((HERE / "cluster_register.json").read_text(encoding="utf-8"))
    assert reg["prereg_file_sha256"] == prereg_sha, "REGISTER_PREREG_SHA_MISMATCH"
    assert freeze["frozen_at_utc"] < reg["built_at_utc"], "PREREG_FREEZE_TIMESTAMP_NOT_EARLIER_THAN_BUILD"

    tc_mod = load_frozen_tc_validator()
    validator_script = HERE / "real_network_format.py"

    positive_checks = []
    positive_passed = 0
    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        mpath = HERE / entry["model_path"]
        assert mpath.is_file(), f"missing {mpath}"
        file_sha = rnf.sha_file(mpath)
        assert file_sha == entry["model_file_sha256"], f"{cid}: file sha mismatch"
        model = json.loads(mpath.read_text(encoding="utf-8"))
        canon_sha = rnf.real_input_sha256(model)
        assert canon_sha == entry["model_canonical_sha256"], f"{cid}: canonical sha mismatch"

        # 1. Real CLI subprocess validation with on-disk source SHA verification
        cli_res = run_no_window(
            [INTERPRETER, "-B", "-X", "utf8", str(validator_script), "--validate", str(mpath), "--repo-root", str(REPO), "--verify-source-sha"],
            REPO,
        )
        cli_ok = cli_res["exit_code"] == 0

        # 2. Exact endpoint attachment across all modes
        ep_ok, ep_note = check_exact_endpoint_attachment(model, entry["anchor_A_fracture"], entry["anchor_B_fracture"])

        # 3. Non-orthogonality floor check
        min_req = 2 if entry["site_id"] == "jardim_garcia_uh" else 3
        nonorth_ok = entry["non_orthogonality"]["non_orthogonal_pairs_0p05_0p95"] >= min_req

        # 4. In-memory lossless projection to frozen TC_network/network_format.validate()
        core_payload = rnf.to_core_kernel_payload(model, repo_root=REPO, verify_source_sha=False)
        tc_budget = tc_mod.validate(core_payload)
        tc_ok = isinstance(tc_budget, dict) and set(tc_budget) == {"cells", "pairs"}

        all_ok = cli_ok and ep_ok and nonorth_ok and tc_ok
        if all_ok:
            positive_passed += 1
        positive_checks.append({
            "cluster_id": cid,
            "site_id": entry["site_id"],
            "cli_exit_code": cli_res["exit_code"],
            "endpoint_attachment_exact": ep_ok,
            "endpoint_note": ep_note,
            "non_orthogonality_ok": nonorth_ok,
            "tc_kernel_in_memory_admitted": tc_ok,
            "passed": all_ok,
        })

    # Run all 7 classes (11 subcases) of toothed negative controls
    nc_report = run_negative_controls(
        HERE / "clusters/jardim_c01/model.json",
        HERE / "clusters/perozzo_c01/model.json",
        HERE / "clusters/perozzo_c04/model.json",
    )
    (HERE / "negative_controls.json").write_text(
        json.dumps(nc_report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Audit zero modification to frozen historical directories
    frozen_audit = audit_frozen_dirs(freeze["frozen_dirs_baseline"])
    (HERE / "frozen_dirs_audit.json").write_text(
        json.dumps(frozen_audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    ver_summary = {
        "schema": "m1-verification-summary/v1",
        "task_id": "M1",
        "at_utc": utc(),
        "prereg_frozen_before_build": freeze["frozen_at_utc"] < reg["built_at_utc"],
        "prereg_frozen_at_utc": freeze["frozen_at_utc"],
        "clusters_built_at_utc": reg["built_at_utc"],
        "positive_total": len(positive_checks),
        "positive_passed": positive_passed,
        "jardim_count": reg["counts"]["jardim_garcia_uh"],
        "perozzo_count": reg["counts"]["perozzo_finale_ligure"],
        "negative_controls_total": nc_report["total_controls"],
        "negative_controls_rejected_with_teeth": nc_report["rejected_with_teeth"],
        "frozen_dirs_zero_modification": frozen_audit["zero_modification_confirmed"],
        "all_checks_passed": (
            positive_passed == len(positive_checks)
            and len(positive_checks) >= 15
            and reg["counts"]["jardim_garcia_uh"] >= 5
            and reg["counts"]["perozzo_finale_ligure"] >= 10
            and nc_report["all_passed_with_teeth"]
            and frozen_audit["zero_modification_confirmed"]
        ),
        "positive_checks": positive_checks,
    }
    (HERE / "verification_summary.json").write_text(
        json.dumps(ver_summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_checks_passed": ver_summary["all_checks_passed"],
                "positive": f"{positive_passed}/{len(positive_checks)}",
                "negative_with_teeth": f"{nc_report['rejected_with_teeth']}/{nc_report['total_controls']}",
                "frozen_dirs_zero_mod": frozen_audit["zero_modification_confirmed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if ver_summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
