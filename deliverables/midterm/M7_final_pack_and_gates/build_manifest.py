# -*- coding: utf-8 -*-
"""M7 · 全链路资产与哈希终检（只读 M0–M6，输出 MIDTERM_MANIFEST.json）。

退出码：
  0  全链 PASS
  2  缺规范四件 / 目录为空
  3  证据 SHA256 不符
  4  某单 result.json#status != PASS
  5  负控未全部被拒 / 自有缺陷为空
  9  枚举到 0 个单元（禁空对空）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
SPRINT = ROOT / "results" / "midterm_sprint_20261120"
DEFAULT_UNITS = [
    "M0_cleanup_baseline",
    "M1_real_to_exact_bridge",
    "M2_active_identifiability",
    "M3_independent_judge",
    "M4_benchmark_figures",
    "M5_strata_workbench_demo",
    "M6_midterm_defense_deck",
]
SPEC_FILES = ["result.json", "report.md", "own_defects.json", "run_receipt.json"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_sha(p: Path) -> str:
    obj = json.loads(p.read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", default=str(SPRINT))
    ap.add_argument("--out", default=str(SPRINT / "M7_final_pack_and_gates" / "MIDTERM_MANIFEST.json"))
    ap.add_argument("--no-cross-check", action="store_true", help="负控用：跳过 M3/M6 跨单一致性")
    args = ap.parse_args(argv)

    base = Path(args.units_dir)
    if not base.is_dir():
        print(f"UNITS_DIR_MISSING: {base}", file=sys.stderr)
        return 9

    dirs = sorted([d for d in base.iterdir() if d.is_dir() and re.match(r"^M[0-6]_", d.name)])
    if not dirs:
        print("NO_UNITS_ENUMERATED: 0 个 M0–M6 目录，拒签", file=sys.stderr)
        return 9

    units: List[Dict[str, Any]] = []
    total_files = 0
    total_checked = 0
    total_mismatch = 0
    missing_spec: List[str] = []
    not_pass: List[str] = []
    nc_bad: List[str] = []
    od_empty: List[str] = []

    for d in dirs:
        u: Dict[str, Any] = {"unit": d.name, "dir": str(d), "spec": {}, "status": None,
                             "evidence": {"count": 0, "verified": 0, "mismatch": []},
                             "negative_controls": {}, "gates": {}, "own_defects_keys": {}}
        for f in SPEC_FILES:
            p = d / f
            ok = p.is_file() and p.stat().st_size > 0
            u["spec"][f] = {"present": ok, "bytes": p.stat().st_size if p.is_file() else 0,
                            "sha256": sha256_file(p) if ok else None}
            if not ok:
                missing_spec.append(f"{d.name}/{f}")
        if not (d / "result.json").is_file():
            units.append(u)
            continue
        r = load_json(d / "result.json")
        u["status"] = r.get("status")
        if u["status"] != "PASS":
            not_pass.append(f"{d.name}={u['status']}")
        for e in r.get("evidence") or []:
            total_checked += 1
            u["evidence"]["count"] += 1
            p = Path(e["file"])
            if not p.is_absolute():
                cand = d / p
                p = cand if cand.exists() else (ROOT / p)
            if not p.is_file():
                u["evidence"]["mismatch"].append({"file": e["file"], "reason": "MISSING"})
                total_mismatch += 1
                continue
            got = sha256_file(p)
            bytes_ok = ("bytes" not in e) or (p.stat().st_size == e["bytes"])
            if got != e.get("sha256") or not bytes_ok:
                u["evidence"]["mismatch"].append({"file": e["file"], "want": e.get("sha256"),
                                                  "got": got, "bytes_ok": bytes_ok})
                total_mismatch += 1
            else:
                u["evidence"]["verified"] += 1
            total_files += 1
        if (d / "own_defects.json").is_file():
            od = load_json(d / "own_defects.json")
            keys = ["not_verified", "absence_of_problem_is_not_proof", "found_while_working", "scope_creep_refused"]
            u["own_defects_keys"] = {k: len(od.get(k) or []) for k in keys}
            if any(v == 0 for v in u["own_defects_keys"].values()):
                od_empty.append(d.name)
        # 负控以 result.json#enumeration 为统一口径（交付规范字段），NC 文件只作旁证
        enum = r.get("enumeration") or {}
        nc_tot = enum.get("negative_controls")
        nc_rej = enum.get("negative_correctly_rejected")
        u["negative_controls"] = {"total": nc_tot, "correctly_rejected": nc_rej,
                                  "source": "result.json#enumeration"}
        if not nc_tot or nc_rej != nc_tot:
            nc_bad.append(f"{d.name}={nc_rej}/{nc_tot}")
        if (d / "negative_controls.json").is_file():
            nc = load_json(d / "negative_controls.json")
            u["negative_controls"]["file_total"] = (
                nc.get("total") or nc.get("total_cases") or nc.get("total_controls")
                or nc.get("n_controls") or (nc.get("controls_summary") or {}).get("total_checks")
                or len(nc.get("cases") or [])
            )
            u["negative_controls"]["file_all_passed"] = bool(
                nc.get("all_passed") or nc.get("all_passed_with_teeth")
                or (nc.get("controls_summary") or {}).get("all_toothed_rejections_passed")
            )
        u["gates"] = r.get("gates") or {}
        units.append(u)

    # ---- 跨单一致性 ----
    cross: Dict[str, Any] = {}
    if args.no_cross_check:
        cross = {"skipped": True}
        cross_ok = True
    else:
        try:
            m3d = base / "M3_independent_judge"
            m2run = base / "M2_active_identifiability" / "runs" / "run_20261120_exact_v1"
            vb = load_json(m3d / "verdict_batch.json")
            cross["m3_counts"] = vb.get("counts")
            cross["m3_refused"] = vb.get("refused")
            cross["m3_agree_22"] = (vb.get("counts") or {}).get("AGREE") == 22
            cross["m3_disagree_zero"] = (vb.get("counts") or {}).get("DISAGREE") == 0
            cross["m3_m1_input_match"] = canon_sha(base / "M1_real_to_exact_bridge" / "cluster_register.json") == vb.get("m1_register_sha256")
            cross["m3_m2_input_match"] = canon_sha(m2run / "summary.json") == vb.get("m2_summary_sha256")
            m6 = load_json(base / "M6_midterm_defense_deck" / "number_traceability_map.json")
            ms = m6.get("m3_sync_status") or {}
            cross["m6_m3_sync_sha_match"] = ms.get("sha256") == sha256_file(m3d / "verdict_batch.json")
            cross["m6_m3_agree"] = ms.get("agree")
            cross["m6_m3_disagree"] = ms.get("disagree")
            cross["m6_placeholder_zero"] = (load_json(base / "M6_midterm_defense_deck" / "result.json")
                                             .get("enumeration", {}).get("m3_pending_placeholders_remaining") == 0)
        except Exception as exc:  # 缺件即判死，不静默
            cross["error"] = f"{type(exc).__name__}: {exc}"
        cross_ok = bool(cross) and "error" not in cross and all([
            cross.get("m3_agree_22"), cross.get("m3_disagree_zero"),
            cross.get("m3_m1_input_match"), cross.get("m3_m2_input_match"),
            cross.get("m6_m3_sync_sha_match"), cross.get("m6_m3_agree") == 22,
            cross.get("m6_m3_disagree") == 0, cross.get("m6_placeholder_zero"),
        ])

    verdict = "PASS"
    reasons = []
    if missing_spec:
        verdict = "FAILED"; reasons.append(f"缺规范四件 {len(missing_spec)} 件")
    if total_mismatch:
        verdict = "FAILED"; reasons.append(f"证据 SHA 不符 {total_mismatch} 件")
    if not_pass:
        verdict = "FAILED"; reasons.append(f"非 PASS 单元 {not_pass}")
    if nc_bad:
        verdict = "FAILED"; reasons.append(f"负控未全拒 {nc_bad}")
    if od_empty:
        verdict = "FAILED"; reasons.append(f"own_defects 有空栏 {od_empty}")
    if not cross_ok:
        verdict = "FAILED"; reasons.append("跨单一致性不成立")
    if total_checked == 0:
        verdict = "FAILED"; reasons.append("证据枚举 0 件（禁空对空）")

    out = {
        "schema": "midterm-manifest/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "M0–M6 七个交付目录规范四件在盘性 + result.json evidence 全量 SHA256 重算 + 跨单输入/回填一致性",
        "units_dir": str(base),
        "enumeration": {
            "units": len(dirs),
            "spec_files_expected": len(dirs) * len(SPEC_FILES),
            "spec_files_present": len(dirs) * len(SPEC_FILES) - len(missing_spec),
            "evidence_entries": total_checked,
            "evidence_verified": total_checked - total_mismatch,
            "evidence_mismatch": total_mismatch,
        },
        "units": units,
        "cross_checks": cross,
        "verdict": verdict,
        "reasons": reasons,
        "not_claimed": [
            "哈希一致只证明交付件未被改动，不证明其科学结论正确",
            "M0–M6 的负控结论由其各自 result.json/negative_controls.json 承载，本单只做汇总核对",
        ],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "units": len(dirs), "evidence": total_checked,
                      "mismatch": total_mismatch, "missing_spec": missing_spec,
                      "reasons": reasons, "out": str(outp)}, ensure_ascii=False, indent=1))

    if missing_spec:
        return 2
    if total_mismatch:
        return 3
    if not_pass:
        return 4
    if nc_bad or od_empty or not cross_ok:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
