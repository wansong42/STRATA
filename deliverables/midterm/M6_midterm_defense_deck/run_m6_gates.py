# -*- coding: utf-8 -*-
"""M6 统一门禁运行器：四道现役门禁 + verify_m6 全量校验，全部无窗子进程，逐命令记录原始退出码。

纪律（AGENTS.md §5/§11、总控单 §五）：
- 唯一 Python: E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe
- --out 指向本单目录；禁管道吃退出码；子进程一律 shell=False + CREATE_NO_WINDOW
- selfcheck 需 KMP_DUPLICATE_LIB_OK=TRUE 与 PYTHONPATH=src
- g4（r303 --draft）必须逐份实跑：3 份核心文稿（paper）+ 本报告（governance）

用法:  python run_m6_gates.py
产出:  gates/g1.json gates/g2.json gates/g3.json gates/g4/ gates/g4_ppt_outline/
       gates/g4_spoken_script/ gates/g4_qa_20/ logs/*.txt run_receipt.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
GATES = SCRIPT_DIR / "gates"
LOGS = SCRIPT_DIR / "logs"
PYTHON_EXE = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
CREATE_NO_WINDOW = 134217728


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    for k, v in (env_extra or {}).items():
        if k == "PYTHONPATH":
            # 追加而非替换：保留宿主注入的运行时守护路径，否则子进程会被 CHAT_REFERENCE_PROTECTED 拒绝
            cur = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = v + os.pathsep + cur if cur else v
        else:
            env[k] = v
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=CREATE_NO_WINDOW,
    )


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    GATES.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    commands = []

    def record(tag: str, cmd: list[str], proc, enumerated: int, stdout_file=None, stderr_file=None):
        commands.append(
            {
                "cmd": " ".join(f'"{c}"' if " " in c else c for c in cmd),
                "interpreter": PYTHON_EXE,
                "executed": True,
                "exit_code": proc.returncode,
                "enumerated": enumerated,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "tag": tag,
            }
        )
        print(f"[{tag}] exit={proc.returncode} enumerated={enumerated}")
        if proc.returncode != 0:
            print((proc.stderr or proc.stdout)[-1500:])

    # ---- gate 0: verify_m6 全量校验 + 负控（先刷 negative_controls.json / g4_*/ 三份 paper 实跑）
    cmd = [PYTHON_EXE, "-B", "-X", "utf8", str(SCRIPT_DIR / "verify_m6.py"), "--run-negative-controls"]
    proc = run(cmd)
    (LOGS / "verify_m6.txt").write_text(
        "===== CMD =====\n" + " ".join(cmd) + f"\n===== EXIT =====\n{proc.returncode}\n===== STDOUT =====\n{proc.stdout}\n===== STDERR =====\n{proc.stderr}\n",
        encoding="utf-8",
    )
    enum_v = 0
    import re as _re
    m = _re.findall(r'"extracted_token_checks": (\d+)', proc.stdout)
    enum_v = sum(int(x) for x in m)
    record("verify_m6_full", cmd, proc, enum_v, "logs/verify_m6.txt")
    if proc.returncode != 0:
        print("VERIFY_M6_FAILED: 停止后续门禁")
    else:
        # ---- gate 1: repro_certify
        cmd = [PYTHON_EXE, "-B", "-X", "utf8", str(ROOT / "scripts" / "repro_certify.py"), "--out", str(GATES / "g1.json")]
        proc = run(cmd)
        (LOGS / "g1.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        n = len(jload(GATES / "g1.json").get("checks", {})) if (GATES / "g1.json").is_file() else 0
        record("g1_repro_certify", cmd, proc, n, "logs/g1.txt", "gates/g1.json")

        # ---- gate 3: r263 delivery scan
        cmd = [
            PYTHON_EXE, "-B", "-X", "utf8", str(ROOT / "scripts" / "r263_gain_mislabel_scan.py"),
            "--scope", "delivery", "--fail-on-hit", "--out", str(GATES / "g3.json"),
        ]
        proc = run(cmd)
        (LOGS / "g3.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        n = jload(GATES / "g3.json").get("files_scanned", 0) if (GATES / "g3.json").is_file() else 0
        record("g3_r263_delivery", cmd, proc, n, "logs/g3.txt", "gates/g3.json")

        # ---- gate 4: r303 --draft，逐份（3 核心文稿 paper 已由 verify_m6 实跑并落 gates/g4_*；此处补本报告 governance）
        for tag, draft, doc_class, outdir in [
            ("g4_ppt", SCRIPT_DIR / "PPT大纲_大创中期_15页.md", "paper", GATES / "g4_ppt_outline"),
            ("g4_script", SCRIPT_DIR / "口头讲稿_8分钟实战版.md", "paper", GATES / "g4_spoken_script"),
            ("g4_qa", SCRIPT_DIR / "评委最尖锐20问_标准应答卡.md", "paper", GATES / "g4_qa_20"),
            ("g4_report", SCRIPT_DIR / "report.md", "governance", GATES / "g4"),
        ]:
            cmd = [
                PYTHON_EXE, "-B", "-X", "utf8", str(ROOT / "scripts" / "r303_draft_check.py"),
                "--draft", str(draft), "--doc-class", doc_class, "--json-only", "--out", str(outdir),
            ]
            proc = run(cmd)
            djp = outdir / "draft_check.json"
            verdict = ""
            if djp.is_file():
                r = jload(djp)
                verdict = f"{r.get('verdict')}/n_red={r.get('n_red')}"
            (LOGS / (tag + ".txt")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            print(f"  [{tag}] {verdict}")
            record(tag, cmd, proc, 1, f"logs/{tag}.txt", str(djp.relative_to(SCRIPT_DIR)).replace("\\", "/"))

        # ---- gate 2: selfcheck --quick（错峰最后跑）
        cmd = [PYTHON_EXE, "-B", "-X", "utf8", "-m", "fractureflow.selfcheck", "--quick", "--out", str(GATES / "g2.json")]
        proc = run(cmd, env_extra={"PYTHONPATH": str(ROOT / "src")})
        (LOGS / "g2.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        n = 0
        if (GATES / "g2.json").is_file():
            r = jload(GATES / "g2.json")
            s = r.get("summary") or {}
            n = int(s.get("total") or s.get("checks_total") or len(r.get("groups", [])) or 0)
        record("g2_selfcheck_quick", cmd, proc, n, "logs/g2.txt", "gates/g2.json")

    receipt = {
        "at_utc": _now(),
        "interpreter": PYTHON_EXE,
        "cwd": str(ROOT),
        "env_overrides": {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONPATH": str(ROOT / "src") + " (仅 g2)"},
        "commands": commands,
        "no_console_window": True,
        "creationflags": CREATE_NO_WINDOW,
        "g4_core_draft_note": "gates/g4_ppt_outline|g4_spoken_script|g4_qa_20 由 verify_m6.py --run-negative-controls 内的 run_r303_on_core_drafts 实跑；M6-N3 亦独立复跑并记录于 negative_controls.json",
    }
    (SCRIPT_DIR / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bad = [c for c in commands if c["exit_code"] != 0]
    print("TOTAL", len(commands), "FAILED", len(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
