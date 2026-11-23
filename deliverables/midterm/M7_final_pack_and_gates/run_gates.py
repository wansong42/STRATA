# -*- coding: utf-8 -*-
"""M7 · 全仓终态四道现役门禁实跑（无窗、原始退出码、不接管道）。

用法：
  run_gates.py --only g1,g2,g3,g4m6
  run_gates.py --only g4report
每道门禁独立的退出码记录写入 gates/exit_codes_<tag>.json，便于长作业分段续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

CREATE_NO_WINDOW = 134217728
ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
PY = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
M7 = ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates"
M6 = ROOT / "results" / "midterm_sprint_20261120" / "M6_midterm_defense_deck"
GATES = M7 / "gates"
LOGS = M7 / "logs"


def run(tag: str, cmd: List[str], env_extra: Dict[str, str] | None = None, enumerated: int = 0) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if env_extra:
        env.update(env_extra)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{tag}.log"
    t0 = datetime.now(timezone.utc)
    with open(log, "wb") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                              stdout=fh, stderr=subprocess.STDOUT,
                              creationflags=CREATE_NO_WINDOW, shell=False, timeout=3600)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    rec = {"tag": tag, "cmd": " ".join(cmd), "interpreter": PY, "executed": True,
           "exit_code": int(proc.returncode), "seconds": round(dt, 2),
           "stdout_file": str(log.relative_to(M7)).replace("\\", "/"), "enumerated": enumerated}
    print(f"[{tag}] exit={proc.returncode} {dt:.1f}s -> {log}", flush=True)
    return rec


def build(tag: str) -> Dict[str, Any]:
    if tag == "g1":
        return run("g1", [PY, "-B", "-X", "utf8", "scripts/repro_certify.py",
                          "--out", str(GATES / "g1.json")], enumerated=9)
    if tag == "g2":
        return run("g2", [PY, "-B", "-X", "utf8", "-m", "fractureflow.selfcheck", "--quick",
                          "--out", str(GATES / "g2.json")],
                   env_extra={"PYTHONPATH": str(ROOT / "src"), "KMP_DUPLICATE_LIB_OK": "TRUE"},
                   enumerated=40)
    if tag == "g3":
        return run("g3", [PY, "-B", "-X", "utf8", "scripts/r263_gain_mislabel_scan.py",
                          "--scope", "delivery", "--fail-on-hit", "--out", str(GATES / "g3.json")],
                   enumerated=9)
    if tag == "g4m6":
        recs = []
        for name, fname in [("m6_ppt_outline", "PPT大纲_大创中期_15页.md"),
                            ("m6_spoken_script", "口头讲稿_8分钟实战版.md"),
                            ("m6_qa_20", "评委最尖锐20问_标准应答卡.md")]:
            recs.append(run(f"g4_{name}", [PY, "-B", "-X", "utf8", "scripts/r303_draft_check.py",
                                           "--draft", str(M6 / fname), "--doc-class", "paper",
                                           "--json-only", "--out", str(GATES / "g4" / name)],
                            enumerated=1))
        return {"tag": "g4m6", "combined_exit_code": max(r["exit_code"] for r in recs),
                "sub": recs, "executed": True, "enumerated": 3}
    if tag == "g4report":
        rep = M7 / "report.md"
        if not rep.is_file():
            raise SystemExit("M7_REPORT_MISSING")
        return run("g4_m7_report", [PY, "-B", "-X", "utf8", "scripts/r303_draft_check.py",
                                    "--draft", str(rep), "--doc-class", "governance",
                                    "--json-only", "--out", str(GATES / "g4" / "m7_report")],
                   enumerated=1)
    raise SystemExit(f"unknown gate tag: {tag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, help="comma list: g1,g2,g3,g4m6,g4report")
    args = ap.parse_args(argv)
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / "g4").mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.only.split(",") if t.strip()]
    all_recs = []
    for t in tags:
        rec = build(t)
        all_recs.append(rec)
        (GATES / f"exit_codes_{t}.json").write_text(
            json.dumps({"schema": "m7-gate-exit-codes/v1",
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                        "gate": t, "record": rec}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    bad = []
    for r in all_recs:
        if "combined_exit_code" in r:
            if r["combined_exit_code"] != 0:
                bad.append(r["tag"])
        elif r["exit_code"] != 0:
            bad.append(r["tag"])
    print(json.dumps({"gates": tags, "nonzero": bad}, ensure_ascii=False), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
