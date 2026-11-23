# -*- coding: utf-8 -*-
"""M7 · 三类有牙负控（每类都带正基线，负变异必须非零退出）。

M7-N1  假交付目录缺 own_defects.json  → MIDTERM_MANIFEST 构建器非零退出
M7-N2  临时副本 index.html 注入外部 CDN → 离线审计非零退出
M7-N3  M3 verdict_batch 副本 DISAGREE=1 → 终审汇总器拒签（exit 7）
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

CREATE_NO_WINDOW = 134217728
PY = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
M7 = ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates"
SCRATCH = M7 / "_nc_scratch"
M3VB = ROOT / "results" / "midterm_sprint_20261120" / "M3_independent_judge" / "verdict_batch.json"
STATIC = ROOT / "demos" / "workbench"


def run(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW, shell=False, timeout=900)
    return {"cmd": " ".join(cmd), "exit_code": int(p.returncode),
            "output_tail": p.stdout.decode("utf-8", errors="replace")[-900:]}


def n1() -> Dict[str, Any]:
    base = SCRATCH / "n1"
    if base.exists():
        shutil.rmtree(base)
    good = base / "n1_pos" / "M0_fakeunit"
    good.mkdir(parents=True)
    art = good / "artifact.bin"
    art.write_bytes(b"m7-n1-artifact\n")
    import hashlib
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    (good / "result.json").write_text(json.dumps({
        "schema": "dispatch-result/v1", "task_id": "M0", "status": "PASS",
        "evidence": [{"file": "artifact.bin", "bytes": art.stat().st_size, "sha256": sha}],
        "enumeration": {"negative_controls": 1, "negative_correctly_rejected": 1},
        "field_error_rate": None, "human_time": None, "hydraulic": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (good / "report.md").write_text("# fake\n", encoding="utf-8")
    (good / "run_receipt.json").write_text("{}\n", encoding="utf-8")
    (good / "own_defects.json").write_text(json.dumps({
        "not_verified": ["x"], "absence_of_problem_is_not_proof": ["x"],
        "found_while_working": ["x"], "scope_creep_refused": ["x"],
    }, ensure_ascii=False), encoding="utf-8")
    (good / "negative_controls.json").write_text(json.dumps({"total": 1, "correctly_rejected": 1}, ensure_ascii=False), encoding="utf-8")

    pos_dir = base / "n1_pos"
    pos = run([PY, "-B", "-X", "utf8", str(M7 / "build_manifest.py"), "--units-dir", str(pos_dir),
               "--out", str(base / "pos_manifest.json"), "--no-cross-check"])
    bad_dir = base / "n1_neg"
    shutil.copytree(pos_dir, bad_dir)
    (bad_dir / "M0_fakeunit" / "own_defects.json").unlink()
    neg = run([PY, "-B", "-X", "utf8", str(M7 / "build_manifest.py"), "--units-dir", str(bad_dir),
               "--out", str(base / "neg_manifest.json"), "--no-cross-check"])
    return {"id": "M7-N1", "scenario": "假交付目录缺 own_defects.json",
            "expected_failure": "build_manifest 非零退出（缺规范四件）",
            "baseline": pos, "negative": neg,
            "correctly_rejected": pos["exit_code"] == 0 and neg["exit_code"] == 2,
            "observed_rule": "missing_spec → exit 2"}


def n2() -> Dict[str, Any]:
    base = SCRATCH / "n2"
    if base.exists():
        shutil.rmtree(base)
    copy = base / "workbench_copy"
    shutil.copytree(STATIC, copy)
    idx = copy / "index.html"
    html = idx.read_text(encoding="utf-8")
    injected = html.replace("</body>", '<script src="https://cdn.jsdelivr.net/npm/evil@1.0.0/dist/evil.js"></script>\n</body>')
    if injected == html:
        raise RuntimeError("index.html has no </body> to inject")
    idx.write_text(injected, encoding="utf-8")
    pos = run([PY, "-B", "-X", "utf8", str(M7 / "offline_asset_audit.py"),
               "--root", str(STATIC), "--out", str(base / "pos_offline.json")])
    neg = run([PY, "-B", "-X", "utf8", str(M7 / "offline_asset_audit.py"),
               "--root", str(copy), "--out", str(base / "neg_offline.json")])
    return {"id": "M7-N2", "scenario": "临时副本 index.html 注入 https://cdn.jsdelivr.net 外链",
            "expected_failure": "offline_asset_audit 非零退出且命中 app_owned external",
            "baseline": pos, "negative": neg,
            "correctly_rejected": pos["exit_code"] == 0 and neg["exit_code"] == 2}


def n3() -> Dict[str, Any]:
    base = SCRATCH / "n3"
    base.mkdir(parents=True, exist_ok=True)
    tampered = base / "verdict_batch_tampered.json"
    d = json.loads(M3VB.read_text(encoding="utf-8"))
    d.setdefault("counts", {})
    d["counts"]["AGREE"] = 21
    d["counts"]["DISAGREE"] = 1
    first = sorted((d.get("verdicts") or {}).keys())[0]
    d["verdicts"][first]["verdict"] = "DISAGREE"
    tampered.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    pos = run([PY, "-B", "-X", "utf8", str(M7 / "assess_readiness.py"),
               "--m3-verdict", str(M3VB), "--out", str(base / "pos_signoff.json")])
    neg = run([PY, "-B", "-X", "utf8", str(M7 / "assess_readiness.py"),
               "--m3-verdict", str(tampered), "--out", str(base / "neg_signoff.json")])
    return {"id": "M7-N3", "scenario": "M3 verdict_batch 副本 counts.DISAGREE=1 且首簇改 DISAGREE",
            "expected_failure": "终审汇总器拒签并给出 REFUSE_SIGNOFF 告警（exit 7）",
            "baseline": pos, "negative": neg,
            "correctly_rejected": pos["exit_code"] == 0 and neg["exit_code"] == 7}


def main() -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cases = [n1(), n2(), n3()]
    out = {
        "schema": "m7-negative-controls/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "no_console_window": True,
        "creationflags": CREATE_NO_WINDOW,
        "scratch_dir": str(SCRATCH.relative_to(ROOT)).replace("\\", "/"),
        "total": len(cases),
        "correctly_rejected": sum(1 for c in cases if c["correctly_rejected"]),
        "all_passed": all(c["correctly_rejected"] for c in cases),
        "cases": cases,
    }
    (M7 / "negative_controls.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"total": out["total"], "correctly_rejected": out["correctly_rejected"],
                      "all_passed": out["all_passed"]}, ensure_ascii=False))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
