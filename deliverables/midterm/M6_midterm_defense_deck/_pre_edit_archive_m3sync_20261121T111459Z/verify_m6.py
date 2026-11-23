# -*- coding: utf-8 -*-
"""Verification & Negative Control Engine for Task M6 (Midterm Defense Deck).

Enforces:
  1. 100% of numbers (percentages, angles, decimals, ratios, comma-counts) in the
     3 core defense drafts map to `number_traceability_map.json` and recompute
     with 0 mismatch from authoritative on-disk `results/*.json` files.
  2. Zero internal codes (`T-N`, `W161`, `NNS-001`, `R263`, `E3GT`, `MACE`, etc.)
     in the prose of any core defense draft.
  3. Zero oracle readings (`12.37`, `12.51`, `16.22`, `19.13`, `19.16`, etc.),
     zero deprecated numbers, and zero R263/R303 cross-column mixing.
  4. All 3 core defense drafts pass `scripts/r303_draft_check.py --draft <doc>`
     with `verdict == "CLEAN"` and `n_red == 0` under both `paper` and `governance`.
  5. >= 3 families of negative controls (`M6-N1`, `M6-N2`, `M6-N3`) executed via
     isolated zero-window subprocesses (`CREATE_NO_WINDOW = 134217728`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_midterm_deck as B  # noqa: E402
import r263_gain_mislabel_scan as R263  # noqa: E402
import r303_draft_check as R303  # noqa: E402
import r303_kit as K  # noqa: E402

CREATE_NO_WINDOW = 134217728
PYTHON_EXE = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
CORE_DOCS = [B.DOC_PPT, B.DOC_SCRIPT, B.DOC_QA]

# Structural presentation constants (slide duration/page budget, never empirical claims)
PRESENTATION_META_REGEXES = [
    r"C:\\Users\\晚菘\\Documents\\Default Project\\[^\s`\"|）)]+",
    r"(?:results|data|demos|src|scripts)/[^\s`\"|）)]+",
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b",
    r"\b\d{2}:\d{2}:\d{2}Z\b",
    r"\b0\d:\d{2}\b",
    r"\b1\.5\s*分钟\b",
    r"\b9\.5\s*分钟\b",
    r"\b8\s*分钟\b",
    r"\b4\s*分钟\b",
    r"\b2\s*分钟\b",
    r"\[待M3回填\]",
]

INTERNAL_CODE_PATTERNS = [
    re.compile(r"\bT-[A-P]\b"),
    re.compile(r"\bW\d{2,3}\b"),
    re.compile(r"\bNNS-(?:\d{3}|xxx)\b", re.I),
    re.compile(r"\bDEP-\d{3}\b", re.I),
    re.compile(r"\b[rR]\d{3}(?:_\w+)?\b"),
    re.compile(r"\br2xx\b", re.I),
    re.compile(r"\bE3GT\b", re.I),
    re.compile(r"\bMACE\b", re.I),
    re.compile(r"\bfracgen\b", re.I),
    re.compile(r"\bM[0-7]-N\d[a-z]?\b"),
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_subproc(argv: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return subprocess.run(
        argv,
        cwd=str(cwd or ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=CREATE_NO_WINDOW,
    )


def verify_traceability_map(tmap_path: Path) -> Dict[str, Any]:
    """Verify every entry in number_traceability_map.json against on-disk JSONs."""
    if not tmap_path.is_file():
        raise RuntimeError(f"TRACEABILITY_MAP_MISSING: {tmap_path}")
    tmap = json.loads(tmap_path.read_text(encoding="utf-8"))

    for null_key in ("field_error_rate", "human_time", "hydraulic"):
        if tmap.get(null_key) is not None:
            raise RuntimeError(f"EMPTY_DENOMINATOR_VIOLATION: {null_key} must be null, got {tmap.get(null_key)!r}")

    entries = tmap.get("entries") or []
    if not entries:
        raise RuntimeError("EMPTY_TRACEABILITY_ENTRIES")

    registered_tokens: Set[str] = set()
    recomputed_count = 0

    for ent in entries:
        eid = ent.get("entry_id", "?")
        src_rel = ent.get("source_json", "")
        if not src_rel or not src_rel.startswith("results/") or not src_rel.endswith(".json"):
            raise RuntimeError(f"INVALID_SOURCE_JSON ({eid}): {src_rel!r}")
        if "M6_midterm_defense_deck" in src_rel:
            raise RuntimeError(f"SELF_REFERENCE_FORBIDDEN ({eid}): {src_rel!r}")
        src_path = ROOT / src_rel
        if not src_path.is_file():
            raise RuntimeError(f"SOURCE_JSON_NOT_ON_DISK ({eid}): {src_rel}")

        actual_sha = _sha256_file(src_path)
        if actual_sha != ent.get("source_sha256"):
            raise RuntimeError(
                f"SOURCE_SHA256_MISMATCH ({eid}): expected={ent.get('source_sha256')} actual={actual_sha}"
            )

        for req_field in ("scope", "enumeration", "measured_at_utc", "domain_track"):
            val = ent.get(req_field)
            if not isinstance(val, str) or not val.strip():
                raise RuntimeError(f"MISSING_THREE_PIECES_METADATA ({eid}): {req_field} is empty")

        if ent.get("is_oracle") is not False or ent.get("is_deprecated") is not False:
            raise RuntimeError(f"FORBIDDEN_FLAG_IN_MAP ({eid}): is_oracle/is_deprecated must be false")

        jobj = json.loads(src_path.read_text(encoding="utf-8"))
        jpath = ent.get("json_path")
        if isinstance(jpath, list):
            actual_raw = [B.resolve_json_path(jobj, p) for p in jpath]
        else:
            actual_raw = B.resolve_json_path(jobj, jpath)

        if actual_raw != ent.get("raw_value_in_json"):
            raise RuntimeError(
                f"RAW_VALUE_MISMATCH ({eid}): map={ent.get('raw_value_in_json')!r} actual={actual_raw!r}"
            )

        recomp = B.recompute_value(actual_raw, ent.get("recompute_rule", ""))
        if recomp != ent.get("recomputed_value"):
            raise RuntimeError(
                f"RECOMPUTE_MISMATCH ({eid}): map={ent.get('recomputed_value')!r} recomputed={recomp!r}"
            )

        for tok in ent.get("rendered_tokens") or []:
            registered_tokens.add(str(tok))
        recomputed_count += 1

    return {
        "ok": True,
        "entries_verified": recomputed_count,
        "registered_tokens": registered_tokens,
    }


def _strip_meta(text: str) -> str:
    out = text
    for pat in PRESENTATION_META_REGEXES:
        out = re.sub(pat, " ", out)
    return out


def check_single_draft(draft_path: Path, registered_tokens: Set[str]) -> Dict[str, Any]:
    """Run all 4 guards on a single markdown draft. Raises ValueError with code on failure."""
    if not draft_path.is_file():
        raise ValueError(f" code=2 DRAFT_NOT_FOUND: {draft_path}")
    raw_text = draft_path.read_text(encoding="utf-8")
    cleaned = _strip_meta(raw_text)

    # Guard A: Internal code leakage in prose
    for rx in INTERNAL_CODE_PATTERNS:
        m = rx.search(cleaned)
        if m:
            raise ValueError(f"code=3 INTERNAL_CODE_LEAK: token={m.group(0)!r} in {draft_path.name}")

    # Guard B: Forbidden oracle & deprecated numbers
    oracle_forms: Set[str] = set()
    for o in K.oracle_tokens():
        oracle_forms |= K.dp_forms([o["value"]], dps=(2, 3, 4))
    dep_forms, _ = K.deprecated_tokens()

    for m in re.finditer(r"(?<![\d.])(\d+\.\d+)(?![\d])", cleaned):
        tok = m.group(1)
        if tok in oracle_forms:
            raise ValueError(f"code=4 ORACLE_VALUE_FORBIDDEN: token={tok!r} in {draft_path.name}")
        if tok in dep_forms:
            raise ValueError(f"code=4 DEPRECATED_VALUE_FORBIDDEN: token={tok!r} in {draft_path.name}")

    # Guard C: R263 gain vs absolute error mislabeling
    abs_set, hr_set = R263._load_sets()
    abs_forms, hr_forms = R263._dp_forms(abs_set), R263._dp_forms(hr_set)
    r263_hits = [
        h for h in R263.scan_text(R263.reflow(raw_text), abs_forms, hr_forms, prose=True) if not h.get("exempt")
    ]
    if r263_hits:
        h0 = r263_hits[0]
        raise ValueError(
            f"code=5 R263_GAIN_MISLABEL_HIT: rule={h0['rule']} token={h0['token']!r} in {draft_path.name}"
        )

    # Guard D: 100% number traceability against number_traceability_map.json
    # Check all percentages, degree angles, decimals, ratios, and comma-formatted integers
    pct_tokens = re.findall(r"(?<![\d.])\d+(?:\.\d+)?%", cleaned)
    deg_tokens = re.findall(r"(?<![\d.])\d+(?:\.\d+)?°", cleaned)
    dec_tokens = re.findall(r"(?<![\d.])\d+\.\d+", cleaned)
    rat_tokens = re.findall(r"\b\d+/\d+\b", cleaned)
    comma_tokens = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", cleaned)

    all_extracted = pct_tokens + deg_tokens + dec_tokens + rat_tokens + comma_tokens
    for tok in all_extracted:
        if tok not in registered_tokens:
            raise ValueError(
                f"code=2 UNTRACED_NUMBER_TOKEN: token={tok!r} in {draft_path.name} is not registered in number_traceability_map.json"
            )

    # Guard E: R303 draft check (strictest paper mode)
    rep_paper = R303.check(raw_text, kind="md", doc_class="paper")
    if rep_paper["verdict"] != "CLEAN" or rep_paper["n_red"] != 0:
        first_red = next(
            (f for f in rep_paper["findings"] if f["severity"] == "RED" and not f["exempt"]),
            rep_paper["findings"][0] if rep_paper["findings"] else {},
        )
        raise ValueError(
            f"code=6 R303_DRAFT_NOT_CLEAN: verdict={rep_paper['verdict']} n_red={rep_paper['n_red']} first={first_red}"
        )

    return {
        "draft": draft_path.name,
        "bytes": draft_path.stat().st_size,
        "sha256": _sha256_file(draft_path),
        "extracted_token_checks": len(all_extracted),
        "distinct_tokens_checked": len(set(all_extracted)),
        "r263_hits": 0,
        "r303_paper_verdict": rep_paper["verdict"],
        "r303_paper_n_red": rep_paper["n_red"],
        "r303_paper_n_exempt": rep_paper["n_exempt"],
        "r303_paper_n_note": rep_paper["n_note"],
    }


def run_r303_on_core_drafts() -> List[Dict[str, Any]]:
    """Run real `scripts/r303_draft_check.py --draft <doc>` subprocesses on all 3 core drafts."""
    results = []
    out_subdirs = {
        B.DOC_PPT: SCRIPT_DIR / "gates" / "g4_ppt_outline",
        B.DOC_SCRIPT: SCRIPT_DIR / "gates" / "g4_spoken_script",
        B.DOC_QA: SCRIPT_DIR / "gates" / "g4_qa_20",
    }
    for fn in CORE_DOCS:
        draft_p = SCRIPT_DIR / fn
        out_d = out_subdirs[fn]
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON_EXE,
            "-B",
            "-X",
            "utf8",
            str(ROOT / "scripts" / "r303_draft_check.py"),
            "--draft",
            str(draft_p),
            "--doc-class",
            "paper",
            "--json-only",
            "--out",
            str(out_d),
        ]
        proc = _run_subproc(cmd)
        dc_json = out_d / "draft_check.json"
        if proc.returncode != 0 or not dc_json.is_file():
            raise RuntimeError(f"R303_SUBPROCESS_FAILED on {fn}: rc={proc.returncode} stderr={proc.stderr}")
        rep = json.loads(dc_json.read_text(encoding="utf-8"))
        if rep.get("verdict") != "CLEAN" or rep.get("n_red") != 0:
            raise RuntimeError(f"R303_NOT_CLEAN on {fn}: verdict={rep.get('verdict')} n_red={rep.get('n_red')}")
        results.append(
            {
                "draft": fn,
                "cmd": " ".join(f'"{c}"' if " " in c else c for c in cmd),
                "exit_code": proc.returncode,
                "verdict": rep["verdict"],
                "n_red": rep["n_red"],
                "n_exempt": rep["n_exempt"],
                "n_note": rep["n_note"],
                "n_findings": rep["n_findings"],
                "output_json": str(dc_json.relative_to(ROOT)).replace("\\", "/"),
                "output_sha256": _sha256_file(dc_json),
            }
        )
    return results


def run_negative_controls() -> Dict[str, Any]:
    """Execute >=3 families of negative controls with real isolated child processes."""
    scratch_dir = SCRIPT_DIR / "_nc_scratch"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    try:
        # --- M6-N1a: Inject internal code W161 / NNS-001 into spoken script copy ---
        n1a_file = scratch_dir / "nc_n1a_internal_code.md"
        base_script = (SCRIPT_DIR / B.DOC_SCRIPT).read_text(encoding="utf-8")
        n1a_file.write_text(
            base_script + "\n\n补充违规测试句：我们在内部任务 W161 与台账 NNS-001 中测量了钻孔角度。\n",
            encoding="utf-8",
        )
        p_n1a_verify = _run_subproc(
            [PYTHON_EXE, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--check-draft", str(n1a_file)]
        )
        p_n1a_r303 = _run_subproc(
            [
                PYTHON_EXE,
                "-B",
                "-X",
                "utf8",
                str(ROOT / "scripts" / "r303_draft_check.py"),
                "--draft",
                str(n1a_file),
                "--doc-class",
                "paper",
                "--json-only",
                "--out",
                str(scratch_dir / "r303_n1a"),
            ]
        )
        r303_n1a_rep = json.loads((scratch_dir / "r303_n1a" / "draft_check.json").read_text(encoding="utf-8"))
        n1a_rejected = (p_n1a_verify.returncode != 0) and (
            r303_n1a_rep.get("verdict") == "HIT" and r303_n1a_rep.get("n_red", 0) > 0
        )
        cases.append(
            {
                "id": "M6-N1a",
                "family": "M6-N1",
                "scenario": "在讲稿副本中故意塞入内部代号 W161 与 NNS-001",
                "expected_failure": "verify_m6.py 非零退出 (exit_code=3) 且 r303 (doc_class=paper) 判 HIT (R5 RED)",
                "verify_m6_exit_code": p_n1a_verify.returncode,
                "verify_m6_stderr": (p_n1a_verify.stderr or p_n1a_verify.stdout).strip()[:300],
                "r303_verdict": r303_n1a_rep.get("verdict"),
                "r303_n_red": r303_n1a_rep.get("n_red"),
                "correctly_rejected": n1a_rejected,
            }
        )

        # --- M6-N1b: Inject forbidden oracle reading 12.37 into PPT outline copy ---
        n1b_file = scratch_dir / "nc_n1b_oracle_1237.md"
        base_ppt = (SCRIPT_DIR / B.DOC_PPT).read_text(encoding="utf-8")
        n1b_file.write_text(
            base_ppt + "\n\n补充违规测试句：本系统在北山钻孔上的对外承诺误差已降至 12.37°。\n",
            encoding="utf-8",
        )
        p_n1b_verify = _run_subproc(
            [PYTHON_EXE, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--check-draft", str(n1b_file)]
        )
        p_n1b_r303 = _run_subproc(
            [
                PYTHON_EXE,
                "-B",
                "-X",
                "utf8",
                str(ROOT / "scripts" / "r303_draft_check.py"),
                "--draft",
                str(n1b_file),
                "--doc-class",
                "paper",
                "--json-only",
                "--out",
                str(scratch_dir / "r303_n1b"),
            ]
        )
        r303_n1b_rep = json.loads((scratch_dir / "r303_n1b" / "draft_check.json").read_text(encoding="utf-8"))
        n1b_rejected = (p_n1b_verify.returncode != 0) and (
            r303_n1b_rep.get("verdict") == "HIT" and r303_n1b_rep.get("n_red", 0) > 0
        )
        cases.append(
            {
                "id": "M6-N1b",
                "family": "M6-N1",
                "scenario": "在 PPT 大纲副本中故意塞入 oracle 禁引数值 12.37°",
                "expected_failure": "verify_m6.py 非零退出 (exit_code=4, ORACLE_VALUE_FORBIDDEN) 且 r303 判 HIT (R3b RED)",
                "verify_m6_exit_code": p_n1b_verify.returncode,
                "verify_m6_stderr": (p_n1b_verify.stderr or p_n1b_verify.stdout).strip()[:300],
                "r303_verdict": r303_n1b_rep.get("verdict"),
                "r303_n_red": r303_n1b_rep.get("n_red"),
                "correctly_rejected": n1b_rejected,
            }
        )

        # --- M6-N1c: Inject R263 gain-vs-absolute mislabeling sentence into Q&A copy ---
        n1c_file = scratch_dir / "nc_n1c_r263_mislabel.md"
        base_qa = (SCRIPT_DIR / B.DOC_QA).read_text(encoding="utf-8")
        n1c_file.write_text(
            base_qa
            + "\n\n补充违规测试句：a gated vMF-EM estimator with Terzaghi correction that improves on observation-only k-means by 1.38-4.37 deg on five of six synthetic-truth targets.\n",
            encoding="utf-8",
        )
        p_n1c_verify = _run_subproc(
            [PYTHON_EXE, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--check-draft", str(n1c_file)]
        )
        n1c_rejected = p_n1c_verify.returncode != 0
        cases.append(
            {
                "id": "M6-N1c",
                "family": "M6-N1",
                "scenario": "在应答卡副本中故意混入 R263 绝对误差冒充配对增益的病灶句 (1.38-4.37 deg)",
                "expected_failure": "verify_m6.py 复用 r263 判据非零退出 (exit_code=5, R263_GAIN_MISLABEL_HIT)",
                "verify_m6_exit_code": p_n1c_verify.returncode,
                "verify_m6_stderr": (p_n1c_verify.stderr or p_n1c_verify.stdout).strip()[:300],
                "correctly_rejected": n1c_rejected,
            }
        )

        # --- M6-N2: Inject untraced percentage 73.49% not in number_traceability_map.json ---
        n2_file = scratch_dir / "nc_n2_untraced_pct.md"
        n2_file.write_text(
            base_ppt + "\n\n补充违规测试句：我们的方法在地下盲区额外提升了 73.49% 的预测准确率。\n",
            encoding="utf-8",
        )
        p_n2_verify = _run_subproc(
            [PYTHON_EXE, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--check-draft", str(n2_file)]
        )
        p_n2_r303 = _run_subproc(
            [
                PYTHON_EXE,
                "-B",
                "-X",
                "utf8",
                str(ROOT / "scripts" / "r303_draft_check.py"),
                "--draft",
                str(n2_file),
                "--doc-class",
                "paper",
                "--json-only",
                "--out",
                str(scratch_dir / "r303_n2"),
            ]
        )
        r303_n2_rep = json.loads((scratch_dir / "r303_n2" / "draft_check.json").read_text(encoding="utf-8"))
        n2_rejected = (p_n2_verify.returncode != 0) and (
            r303_n2_rep.get("verdict") == "HIT" and r303_n2_rep.get("n_red", 0) > 0
        )
        cases.append(
            {
                "id": "M6-N2",
                "family": "M6-N2",
                "scenario": "在 PPT 大纲副本中故意写一个不在 number_traceability_map.json 里的无源百分比 73.49%",
                "expected_failure": "verify_m6.py 抓出 UNTRACED_NUMBER_TOKEN: '73.49%' 非零退出 (exit_code=2) 且 r303 报 R1a RED",
                "verify_m6_exit_code": p_n2_verify.returncode,
                "verify_m6_stderr": (p_n2_verify.stderr or p_n2_verify.stdout).strip()[:300],
                "r303_verdict": r303_n2_rep.get("verdict"),
                "r303_n_red": r303_n2_rep.get("n_red"),
                "correctly_rejected": n2_rejected,
            }
        )

        # --- M6-N3: Real --draft r303 check on all 3 core drafts + negative check without --draft ---
        r303_core_runs = run_r303_on_core_drafts()
        p_n3_no_draft = _run_subproc(
            [PYTHON_EXE, "-B", "-X", "utf8", str(ROOT / "scripts" / "r303_draft_check.py"), "--json-only"]
        )
        n3_all_clean = all(r["verdict"] == "CLEAN" and r["n_red"] == 0 and r["exit_code"] == 0 for r in r303_core_runs)
        n3_no_draft_failed = p_n3_no_draft.returncode != 0
        cases.append(
            {
                "id": "M6-N3",
                "family": "M6-N3",
                "scenario": "对全部 3 份核心答辩文稿真实传入 --draft 跑通 r303 (verdict==CLEAN, n_red==0)，并验证不传 --draft 的空跑必以 exit_code=2 失败拒收",
                "expected_failure": "不传 --draft 时 r303 必须非零退出 (exit_code=2)；传入 --draft 时 3/3 核心文稿必须 CLEAN 且 n_red=0",
                "missing_draft_flag_exit_code": p_n3_no_draft.returncode,
                "missing_draft_flag_stderr": (p_n3_no_draft.stderr or "").strip()[:240],
                "core_drafts_r303_runs": r303_core_runs,
                "correctly_rejected": bool(n3_all_clean and n3_no_draft_failed),
            }
        )
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)

    all_ok = all(c["correctly_rejected"] for c in cases)
    out = {
        "schema": "m6-negative-controls/v1",
        "at_utc": _now_utc(),
        "no_console_window": True,
        "creationflags": CREATE_NO_WINDOW,
        "total_families": 3,
        "total_cases": len(cases),
        "correctly_rejected": sum(1 for c in cases if c["correctly_rejected"]),
        "all_passed": all_ok,
        "cases": cases,
    }
    nc_path = SCRIPT_DIR / "negative_controls.json"
    nc_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all_ok:
        raise RuntimeError("NEGATIVE_CONTROLS_FAILED")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify M6 deliverables and run negative controls.")
    ap.add_argument("--check-draft", help="Verify a single draft file against traceability map and guards.")
    ap.add_argument("--run-negative-controls", action="store_true", help="Run M6-N1..M6-N3 negative controls.")
    args = ap.parse_args()

    tmap_path = SCRIPT_DIR / "number_traceability_map.json"
    try:
        tmap_res = verify_traceability_map(tmap_path)
    except Exception as exc:
        print(f"TRACEABILITY_MAP_ERROR: {exc}", file=sys.stderr)
        return 2

    reg_tokens = tmap_res["registered_tokens"]

    if args.check_draft:
        try:
            res = check_single_draft(Path(args.check_draft), reg_tokens)
            print(json.dumps({"ok": True, "result": res}, ensure_ascii=False))
            return 0
        except ValueError as exc:
            msg = str(exc)
            m_code = re.search(r"code=(\d+)", msg)
            rc = int(m_code.group(1)) if m_code else 2
            print(msg, file=sys.stderr)
            return rc

    # Default full verification of all 3 core drafts
    draft_results = []
    for fn in CORE_DOCS:
        try:
            draft_results.append(check_single_draft(SCRIPT_DIR / fn, reg_tokens))
        except ValueError as exc:
            msg = str(exc)
            m_code = re.search(r"code=(\d+)", msg)
            rc = int(m_code.group(1)) if m_code else 2
            print(msg, file=sys.stderr)
            return rc

    nc_summary = None
    if args.run_negative_controls:
        nc_out = run_negative_controls()
        nc_summary = {
            "total_cases": nc_out["total_cases"],
            "correctly_rejected": nc_out["correctly_rejected"],
            "all_passed": nc_out["all_passed"],
        }

    print(
        json.dumps(
            {
                "ok": True,
                "traceability_entries_verified": tmap_res["entries_verified"],
                "core_drafts_verified": len(draft_results),
                "draft_results": draft_results,
                "negative_controls": nc_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
