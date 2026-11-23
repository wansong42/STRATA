# -*- coding: utf-8 -*-
"""One-command windowless runner for all 4 project gates + midterm verify_delivery.

Usage:
    E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/run_gates.py
    E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/run_gates.py --out results/active_runs/gates_receipt.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY_TORCH = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
PY_BASE = "E:/app2/anacoda/Anacoda/python.exe"
CF = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_one(name: str, cmd: list[str], env: dict[str, str]) -> dict:
    t0 = datetime.datetime.now(datetime.timezone.utc)
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, creationflags=CF)
    dt = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = lines[-4:] if lines else []
    return {
        "gate": name,
        "command": " ".join(cmd),
        "returncode": p.returncode,
        "seconds": round(dt, 2),
        "stdout_tail": tail,
        "stderr_tail": [ln for ln in p.stderr.splitlines() if ln.strip()][-4:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all 4 project gates + delivery verification headlessly.")
    ap.add_argument("--out", default="", help="Optional path to write JSON receipt.")
    ap.add_argument("--draft", default="results/midterm_sprint_20261120/架构师交接文档_20261122_中期收官与下半程接手.md",
                    help="Markdown/docx draft path for G4 r303_draft_check.")
    ap.add_argument("--skip-delivery", action="store_true", help="Skip midterm verify_delivery.py.")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUTF8"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    tmp_dir = ROOT / "results/active_runs/_gate_runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = [
        ("G1_repro_certify", [PY_TORCH, "scripts/repro_certify.py", "--out", str(tmp_dir / f"g1_{ts}.json")]),
        ("G2_selfcheck_quick", [PY_TORCH, "-m", "fractureflow.selfcheck", "--quick", "--out", str(tmp_dir / f"g2_{ts}.json")]),
        ("G3_r263_delivery", [PY_TORCH, "scripts/r263_gain_mislabel_scan.py", "--scope", "delivery", "--fail-on-hit"]),
        ("G4_r303_draft_check", [PY_TORCH, "scripts/r303_draft_check.py", "--draft", args.draft, "--json-only"]),
    ]
    if not args.skip_delivery:
        jobs.append((
            "G5_verify_delivery",
            [PY_BASE, "results/midterm_sprint_20261120/答辩配套文档_20261122/verify_delivery.py"],
        ))

    results = []
    all_ok = True
    for name, cmd in jobs:
        res = run_one(name, cmd, env)
        results.append(res)
        ok = res["returncode"] == 0
        if not ok:
            all_ok = False
        print(f"[{'PASS' if ok else 'FAIL'}] {name} (rc={res['returncode']}, {res['seconds']}s)")
        for ln in res["stdout_tail"][-2:]:
            print(f"    {ln}")
        if not ok and res["stderr_tail"]:
            for ln in res["stderr_tail"][-2:]:
                print(f"    ERR: {ln}")

    summary = {
        "schema": "fractureflow.gates_run.v1",
        "executed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_passed": all_ok,
        "results": results,
    }
    out_path = pathlib.Path(args.out) if args.out else (tmp_dir / f"gates_summary_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReceipt saved: {out_path} | all_passed={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
