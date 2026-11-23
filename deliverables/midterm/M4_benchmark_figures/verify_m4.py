# -*- coding: utf-8 -*-
"""M4 Verification, Recompute-Consistency Guard, and 4-Class Toothed Negative Controls.

Supports:
1. `--audit-summary <path> [--recompute-check]` :
   Audits benchmark_summary.json for track separation (M4-N1), recomputation consistency (M4-N2),
   honest empty denominators (M4-N3), null unverified fields, and figure presence.
2. `--run-negative-controls` :
   Runs all 4 required negative control classes (M4-N1, M4-N2, M4-N3, M4-N4) as real
   no-window subprocesses and writes `negative_controls.json`.
3. `--phase1-check` :
   Audits Phase-1 outputs (Perozzo DSE vs Manual, data_ladder_anchors, baseline_scaffold, fig1 & fig2).

Strict AGENTS.md §11 compliance: `creationflags=subprocess.CREATE_NO_WINDOW` (134217728), `shell=False`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SPRINT_DIR = HERE.parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
INTERPRETER = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
CREATE_NO_WINDOW = 0x08000000  # 134217728

sys.path.insert(0, str(HERE))
import run_baselines as rb  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_no_window(argv: Sequence[str], cwd: Path, timeout: int = 180) -> Dict[str, Any]:
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUTF8"] = "1"
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
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "creationflags": CREATE_NO_WINDOW,
        "shell": False,
    }


def _safe_rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else round(num / den, 6)


def _check_close(a: Optional[float], b: Optional[float], label: str, tol: float = 1e-5) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        raise SystemExit(f"RECOMPUTE_MISMATCH (M4-N2): {label} expected={b!r} got={a!r}")
    if abs(float(a) - float(b)) > tol:
        raise SystemExit(f"RECOMPUTE_MISMATCH (M4-N2): {label} expected={b!r} got={a!r}")


def audit_summary_file(summary_path: Path, *, recompute_check: bool = False, check_figures: bool = True) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    if data.get("scope") != "REAL_DERIVED_CONDITIONED_FINITE_UNION":
        raise SystemExit(f"INVALID_SCOPE (M4-N1): expected REAL_DERIVED_CONDITIONED_FINITE_UNION, got {data.get('scope')!r}")

    for null_key in ("field_error_rate", "human_time", "hydraulic"):
        if data.get(null_key) is not None:
            raise SystemExit(f"NON_NULL_UNVERIFIED_FIELD: {null_key}={data.get(null_key)!r} must be null")

    # M4-N1: Strict oracle & synthetic scope ban inside real-outcrop benchmark statistics
    rb_block = data.get("real_outcrop_benchmarks")
    if not isinstance(rb_block, dict):
        raise SystemExit("Missing real_outcrop_benchmarks block")
    rb.assert_no_contamination(rb_block, "real_outcrop_benchmarks")
    rb.assert_no_contamination(data.get("clusters"), "clusters")

    # Check data ladder track also has zero oracle
    ladder_ref = (data.get("separate_tracks") or {}).get("historical_data_ladder_ref") or {}
    if ladder_ref.get("contains_oracle") is not False:
        raise SystemExit("ORACLE_CONTAMINATION (M4-N1): historical_data_ladder_ref.contains_oracle must be false")

    # M4-N3: Empty denominator null checks across all groups
    for gname in ("combined_real_conditioned", "jardim_garcia_uh", "perozzo_finale_ligure"):
        g = rb_block.get(gname)
        if not isinstance(g, dict):
            raise SystemExit(f"Missing benchmark group: {gname}")
        rb.assert_honest_empty_denominators(g, gname)

        # Also check active collapse curve block if present
        ac_block = (data.get("active_collapse_comparison") or {}).get("curves_by_group", {}).get(gname)
        if isinstance(ac_block, dict) and ac_block.get("denominator") == 0:
            if ac_block.get("strategies"):
                for sname, sdata in ac_block["strategies"].items():
                    vals = sdata.get("mean") if isinstance(sdata, dict) else sdata
                    if vals and any(v is not None for v in vals):
                        raise SystemExit(
                            f"EMPTY_DENOMINATOR_VIOLATION (M4-N3): active curve {gname}.{sname} has denominator=0 "
                            f"but non-null rate values {vals!r}"
                        )

    if recompute_check:
        csv_path = HERE / "per_cluster_table.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing {csv_path} for recompute check")
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        if len(csv_rows) != 22:
            raise SystemExit(f"RECOMPUTE_MISMATCH (M4-N2): expected 22 rows in per_cluster_table.csv, got {len(csv_rows)}")

        # Recompute per-group metrics directly from per_cluster_table.csv
        def _bool_opt(s: str) -> Optional[bool]:
            if s == "True":
                return True
            if s == "False":
                return False
            if s in ("", "None", "null"):
                return None
            raise SystemExit(f"Invalid bool in CSV: {s!r}")

        def _recompute_group(sub_rows: List[Dict[str, str]], gkey: str) -> None:
            n = len(sub_rows)
            g = rb_block[gkey]
            if g.get("cluster_count") != n:
                raise SystemExit(f"RECOMPUTE_MISMATCH (M4-N2): {gkey}.cluster_count={g.get('cluster_count')} != {n}")
            comp_rows = [r for r in sub_rows if r["ours_status"] != "BUDGET_EXCEEDED"]
            n_comp = len(comp_rows)
            n_ref = n - n_comp
            if g.get("completed_count") != n_comp or g.get("budget_exceeded_count") != n_ref:
                raise SystemExit(
                    f"RECOMPUTE_MISMATCH (M4-N2): {gkey}.completed/refused=({g.get('completed_count')},{g.get('budget_exceeded_count')}) != ({n_comp},{n_ref})"
                )
            n_conn = sum(1 for r in comp_rows if r["ours_status"] == "CERTIFIED_CONNECTED")
            n_disc = sum(1 for r in comp_rows if r["ours_status"] == "CERTIFIED_DISCONNECTED")
            n_amb = sum(1 for r in comp_rows if r["ours_status"] == "AMBIGUOUS_WITH_VERIFIED_WITNESSES")
            n_flip = sum(1 for r in comp_rows if _bool_opt(r["silent_flip"]) is True)
            n_flip_fp = sum(
                1 for r in comp_rows
                if _bool_opt(r["silent_flip"]) is True and _bool_opt(r["nominal_connected"]) is True
            )
            n_flip_fn = sum(
                1 for r in comp_rows
                if _bool_opt(r["silent_flip"]) is True and _bool_opt(r["nominal_connected"]) is False
            )
            n_phan = sum(1 for r in comp_rows if _bool_opt(r["phantom_union_path"]) is True)
            n_phan_disc = sum(
                1 for r in comp_rows
                if r["ours_status"] == "CERTIFIED_DISCONNECTED" and _bool_opt(r["naive_union_connected"]) is True
            )

            _check_close(g.get("certified_connected_rate"), _safe_rate(n_conn, n_comp), f"{gkey}.certified_connected_rate")
            _check_close(g.get("certified_disconnected_rate"), _safe_rate(n_disc, n_comp), f"{gkey}.certified_disconnected_rate")
            _check_close(g.get("ambiguous_rate"), _safe_rate(n_amb, n_comp), f"{gkey}.ambiguous_rate")
            _check_close(g.get("budget_exceeded_rate"), _safe_rate(n_ref, n), f"{gkey}.budget_exceeded_rate")
            _check_close(g["nominal_dfn"].get("silent_flip_rate"), _safe_rate(n_flip, n_comp), f"{gkey}.silent_flip_rate")
            _check_close(
                g["nominal_dfn"].get("silent_flip_false_positive_risk_rate"),
                _safe_rate(n_flip_fp, n_comp),
                f"{gkey}.silent_flip_fp_rate",
            )
            _check_close(
                g["nominal_dfn"].get("silent_flip_false_negative_risk_rate"),
                _safe_rate(n_flip_fn, n_comp),
                f"{gkey}.silent_flip_fn_rate",
            )
            _check_close(g["naive_union"].get("phantom_union_rate"), _safe_rate(n_phan, n_comp), f"{gkey}.phantom_union_rate")
            _check_close(
                g["naive_union"].get("phantom_fp_given_disconnected_rate"),
                _safe_rate(n_phan_disc, n_disc),
                f"{gkey}.phantom_fp_given_disconnected_rate",
            )

        _recompute_group(csv_rows, "combined_real_conditioned")
        _recompute_group([r for r in csv_rows if r["site_id"] == "jardim_garcia_uh"], "jardim_garcia_uh")
        _recompute_group([r for r in csv_rows if r["site_id"] == "perozzo_finale_ligure"], "perozzo_finale_ligure")

        # Recompute DSE reference match
        dse_path = HERE / "perozzo_dse_vs_manual_summary.json"
        dse = json.loads(dse_path.read_text(encoding="utf-8"))
        dse_ref = (data.get("separate_tracks") or {}).get("perozzo_dse_vs_manual_ref") or {}
        if dse_ref.get("sha256") != rb.sha256_file(dse_path):
            raise SystemExit("RECOMPUTE_MISMATCH (M4-N2): perozzo_dse_vs_manual_summary.json sha256 mismatch")
        _check_close(dse_ref.get("oversegmentation_ratio"), dse["comparison"]["oversegmentation_count_ratio"], "dse.oversegmentation_ratio", tol=1e-4)

    if check_figures:
        import ast
        plot_src = (HERE / "plot_figures.py").read_text(encoding="utf-8")
        if 'matplotlib.use("Agg")' not in plot_src and "matplotlib.use('Agg')" not in plot_src:
            raise SystemExit("plot_figures.py must enforce Agg backend")
        tree = ast.parse(plot_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "show":
                raise SystemExit("plot_figures.py must never call plt.show()")

        stems = [
            "fig1_data_ladder_and_architecture",
            "fig2_real_outcrop_clusters_overview",
            "fig3_baseline_failure_modes",
            "fig4_stability_rho_and_witness_anatomy",
            "fig5_active_next_check_collapse",
        ]
        for st in stems:
            for ext in ("png", "svg"):
                fp = HERE / "figures" / f"{st}.{ext}"
                if not fp.is_file() or fp.stat().st_size < 1024:
                    raise SystemExit(f"Missing or empty figure deliverable: {fp}")

    return {"ok": True, "summary_path": str(summary_path), "recompute_checked": recompute_check}


def run_negative_controls() -> Dict[str, Any]:
    summary_path = HERE / "benchmark_summary.json"
    if not summary_path.is_file():
        raise SystemExit("Cannot run full M4 negative controls before benchmark_summary.json exists")
    base = json.loads(summary_path.read_text(encoding="utf-8"))
    scratch = HERE / "_nc_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def _write_nc(name: str, obj: Any) -> Path:
        p = scratch / f"{name}.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    specs = []

    # M4-N1a: Inject SYNTHETIC_DECLARED_TRIANGLE_UNION into real_outcrop_benchmarks
    n1a = copy.deepcopy(base)
    n1a["real_outcrop_benchmarks"]["combined_real_conditioned"]["injected_scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
    p_n1a = _write_nc("nc_m4_n1a_synthetic_scope", n1a)
    specs.append((
        "M4-N1a",
        "Inject SYNTHETIC_DECLARED_TRIANGLE_UNION into real_outcrop_benchmarks -> audit must exit non-zero",
        [INTERPRETER, "-B", "-X", "utf8", str(HERE / "verify_m4.py"), "--audit-summary", str(p_n1a), "--skip-figures-check"],
        ["CONTAMINATION_REJECTED"],
    ))

    # M4-N1b: Inject oracle reading 12.3705 into real_outcrop_benchmarks
    n1b = copy.deepcopy(base)
    n1b["real_outcrop_benchmarks"]["jardim_garcia_uh"]["oracle_leak_deg"] = 12.3705
    p_n1b = _write_nc("nc_m4_n1b_oracle_leak", n1b)
    specs.append((
        "M4-N1b",
        "Inject oracle 12.3705 into real_outcrop_benchmarks -> audit must exit non-zero",
        [INTERPRETER, "-B", "-X", "utf8", str(HERE / "verify_m4.py"), "--audit-summary", str(p_n1b), "--skip-figures-check"],
        ["CONTAMINATION_REJECTED"],
    ))

    # M4-N2: Tamper one metric in benchmark_summary.json by 1% -> recompute check must catch & exit non-zero
    n2 = copy.deepcopy(base)
    comb = n2["real_outcrop_benchmarks"]["combined_real_conditioned"]
    orig_rate = comb["nominal_dfn"]["silent_flip_rate"]
    comb["nominal_dfn"]["silent_flip_rate"] = round((orig_rate if orig_rate else 0.5) * 1.01 + 0.01, 6)
    p_n2 = _write_nc("nc_m4_n2_one_percent_tamper", n2)
    specs.append((
        "M4-N2",
        "Perturb benchmark_summary.json metric by >=1% -> --recompute-check must exit non-zero",
        [INTERPRETER, "-B", "-X", "utf8", str(HERE / "verify_m4.py"), "--audit-summary", str(p_n2), "--recompute-check", "--skip-figures-check"],
        ["RECOMPUTE_MISMATCH"],
    ))

    # M4-N3: Force empty denominator (ambiguous_count=0 or disconnected_count=0) with 0.0 instead of null
    n3 = copy.deepcopy(base)
    jgrp = n3["real_outcrop_benchmarks"]["jardim_garcia_uh"]
    jgrp["disconnected_count"] = 0
    jgrp["certified_disconnected_count"] = 0
    jgrp["phantom_fp_given_disconnected_rate"] = 0.0
    p_n3 = _write_nc("nc_m4_n3_empty_denom_zero", n3)
    specs.append((
        "M4-N3",
        "Fill 0.0 instead of null when disconnected_count=0 -> audit must exit non-zero",
        [INTERPRETER, "-B", "-X", "utf8", str(HERE / "verify_m4.py"), "--audit-summary", str(p_n3), "--skip-figures-check"],
        ["EMPTY_DENOMINATOR_VIOLATION"],
    ))

    # M4-N4: Mirror M2 run directory and tamper 1 hex char of one cluster's certification.json input_sha256
    m2_src = REPO / base["m2_run_dir"]
    m2_mirror = scratch / "tampered_m2_run"
    n4_out = scratch / "n4_out"
    if m2_mirror.exists():
        shutil.rmtree(m2_mirror)
    if n4_out.exists():
        shutil.rmtree(n4_out)
    shutil.copytree(m2_src, m2_mirror)
    first_cid = base["clusters"][0]["cluster_id"]
    cert_p = m2_mirror / "cases" / first_cid / "certification.json"
    cert_obj = json.loads(cert_p.read_text(encoding="utf-8"))
    orig_sha = cert_obj["input_sha256"]
    cert_obj["input_sha256"] = ("0" if orig_sha[0] != "0" else "1") + orig_sha[1:]
    cert_p.write_text(json.dumps(cert_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    specs.append((
        "M4-N4",
        f"Tamper M2 {first_cid}/certification.json input_sha256 -> run_baselines.py must refuse non-zero",
        [INTERPRETER, "-B", "-X", "utf8", str(HERE / "run_baselines.py"), "--m2-run", str(m2_mirror), "--skip-dse", "--out-dir", str(n4_out)],
        ["M2_INPUT_SHA256_MISMATCH"],
    ))

    controls = []
    all_rejected = True
    for nc_id, desc, argv, expected_tokens in specs:
        res = run_no_window(argv, REPO)
        combined_out = res["stdout"] + "\n" + res["stderr"]
        ok = res["exit_code"] != 0 and any(tok in combined_out for tok in expected_tokens)
        if not ok:
            all_rejected = False
        controls.append({
            "id": nc_id,
            "scenario": desc,
            "command": res["cmd"],
            "expected_nonzero_exit": True,
            "actual_exit_code": res["exit_code"],
            "expected_reason_tokens": expected_tokens,
            "rejected_with_teeth": ok,
            "first_counterexample": combined_out[:500],
        })

    out = {
        "schema": "m4-negative-controls/v1",
        "task_id": "M4",
        "at_utc": utc_now(),
        "total_controls": len(controls),
        "rejected_with_teeth": sum(1 for c in controls if c["rejected_with_teeth"]),
        "distinct_control_classes": 4,
        "all_passed_with_teeth": all_rejected,
        "controls": controls,
    }
    rb.write_json(HERE / "negative_controls.json", out)
    return out


def phase1_check() -> Dict[str, Any]:
    dse = json.loads((HERE / "perozzo_dse_vs_manual_summary.json").read_text(encoding="utf-8"))
    assert dse["comparison"]["manual_facet_count"] == 1405
    assert dse["comparison"]["auto_dse_facet_count"] == 202371
    assert dse["never_used_as_ground_truth"] is True

    ladder = json.loads((HERE / "data_ladder_anchors.json").read_text(encoding="utf-8"))
    assert ladder["contains_oracle"] is False
    assert len(ladder["stages"]) == 3

    scaf = json.loads((HERE / "baseline_scaffold_m1_only.json").read_text(encoding="utf-8"))
    assert scaf["cluster_geometry_summary"]["total_clusters"] == 22
    rb.assert_no_contamination(scaf, "scaffold_summary")

    for stem in ("fig1_data_ladder_and_architecture", "fig2_real_outcrop_clusters_overview"):
        for ext in ("png", "svg"):
            fp = HERE / "figures" / f"{stem}.{ext}"
            if not fp.is_file() or fp.stat().st_size < 1024:
                raise SystemExit(f"Missing Phase-1 figure: {fp}")

    return {
        "ok": True,
        "phase": "PHASE1_SCAFFOLD_AND_FIG1_FIG2",
        "manual_facets": dse["comparison"]["manual_facet_count"],
        "auto_dse_facets": dse["comparison"]["auto_dse_facet_count"],
        "oversegmentation_ratio": dse["comparison"]["oversegmentation_count_ratio"],
        "clusters_validated": scaf["cluster_geometry_summary"]["total_clusters"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify M4 deliverables and run negative controls.")
    ap.add_argument("--audit-summary", default=None, help="Path to benchmark_summary.json to audit")
    ap.add_argument("--recompute-check", action="store_true", help="Recompute benchmark metrics from CSV/sources")
    ap.add_argument("--skip-figures-check", action="store_true", help="Skip figures presence check during negative control audit")
    ap.add_argument("--run-negative-controls", action="store_true", help="Run all 4 negative control classes and emit negative_controls.json")
    ap.add_argument("--phase1-check", action="store_true", help="Verify Phase-1 artifacts (DSE vs Manual, scaffold, fig1, fig2)")
    args = ap.parse_args(argv)

    if args.audit_summary:
        res = audit_summary_file(
            Path(args.audit_summary),
            recompute_check=args.recompute_check,
            check_figures=not args.skip_figures_check,
        )
        print(json.dumps(res, ensure_ascii=False))
        return 0

    if args.phase1_check:
        res = phase1_check()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.run_negative_controls:
        res = run_negative_controls()
        print(json.dumps({
            "all_passed_with_teeth": res["all_passed_with_teeth"],
            "rejected": f"{res['rejected_with_teeth']}/{res['total_controls']}",
        }, ensure_ascii=False))
        return 0 if res["all_passed_with_teeth"] else 1

    # Default full check
    audit_res = audit_summary_file(HERE / "benchmark_summary.json", recompute_check=True, check_figures=True)
    nc_res = run_negative_controls()
    ok = audit_res["ok"] and nc_res["all_passed_with_teeth"]
    print(json.dumps({"audit": audit_res, "negative_controls": f"{nc_res['rejected_with_teeth']}/{nc_res['total_controls']}"}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
