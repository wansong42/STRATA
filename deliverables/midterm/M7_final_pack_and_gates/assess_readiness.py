# -*- coding: utf-8 -*-
"""M7 · 终审汇总器：所有分项必须 PASS 且 M3 判决零分歧，才允许签全 PASS。

M3 检查同时看 counts 与逐簇 verdict（只改 counts 或只改单簇都抓得住）。
退出码：0 可签；7 M3 分歧/缺件；8 其它分项未过；9 无输入。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
M7 = ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates"


def load(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--m3-verdict")
    ap.add_argument("--gates-dir")
    ap.add_argument("--smoke")
    ap.add_argument("--offline")
    ap.add_argument("--out", default=str(M7 / "signoff.json"))
    args = ap.parse_args(argv)

    checks: List[Dict[str, Any]] = []
    m3_bad = False

    def add(name: str, ok: bool, detail: Any = None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if args.m3_verdict:
        vb = load(args.m3_verdict)
        counts = vb.get("counts") or {}
        verdicts = vb.get("verdicts") or {}
        vals = [str(v.get("verdict")) for v in verdicts.values() if isinstance(v, dict)]
        n_agree = sum(1 for v in vals if v == "AGREE")
        n_dis = sum(1 for v in vals if v == "DISAGREE")
        add("M3 counts.AGREE==22", counts.get("AGREE") == 22, counts)
        add("M3 counts.DISAGREE==0", counts.get("DISAGREE") == 0, counts.get("DISAGREE"))
        add("M3 逐簇 AGREE==22", n_agree == 22, {"agree": n_agree, "total": len(vals)})
        add("M3 逐簇 DISAGREE==0", n_dis == 0, n_dis)
        add("M3 MALFORMED/UNDECIDABLE==0",
            counts.get("MALFORMED", 0) == 0 and counts.get("UNDECIDABLE", 0) == 0, counts)
        add("M3 拒收仅 jardim_c01", vb.get("refused") == ["jardim_c01"], vb.get("refused"))
        if counts.get("DISAGREE") or n_dis or counts.get("AGREE") != 22 or n_agree != 22:
            m3_bad = True

    if args.manifest:
        m = load(args.manifest)
        add("MANIFEST verdict==PASS", m.get("verdict") == "PASS", m.get("verdict"))
        add("MANIFEST 证据 0 不符", (m.get("enumeration") or {}).get("evidence_mismatch") == 0,
            (m.get("enumeration") or {}).get("evidence_mismatch"))
        add("MANIFEST 跨单一致", bool(m.get("cross_checks")) and "error" not in (m.get("cross_checks") or {}),
            m.get("cross_checks"))

    if args.gates_dir:
        gd = Path(args.gates_dir)
        expected = ["g1", "g2", "g3", "g4m6", "g4report"]
        found = {}
        for tag in expected:
            p = gd / f"exit_codes_{tag}.json"
            if not p.is_file():
                found[tag] = "MISSING"
                continue
            rec = load(str(p))["record"]
            code = rec.get("combined_exit_code") if "combined_exit_code" in rec else rec.get("exit_code")
            found[tag] = code
        add("四道门禁五段原始退出码全 0", all(v == 0 for v in found.values()), found)
        try:
            g1 = load(str(gd / "g1.json")).get("summary") or {}
            add("g1 9/9", g1.get("pass_") == 9 and g1.get("fail") == 0, g1)
        except Exception as exc:
            add("g1 9/9", False, str(exc))
        try:
            g2 = load(str(gd / "g2.json"))
            d2 = g2.get("summary") or g2
            add("g2 40/40", (d2.get("passed") == 40 and d2.get("failed") == 0), d2)
        except Exception as exc:
            add("g2 40/40", False, str(exc))
        try:
            g3 = load(str(gd / "g3.json"))
            add("g3 非豁免混标 0", not g3.get("mislabel_hits"), {"mislabel": g3.get("mislabel_hits"),
                                                              "exempt": len(g3.get("exempt_hits") or [])})
        except Exception as exc:
            add("g3 非豁免混标 0", False, str(exc))
        g4ok, g4detail = True, {}
        for name in ["m6_ppt_outline", "m6_spoken_script", "m6_qa_20", "m7_report"]:
            p = gd / "g4" / name / "draft_check.json"
            if not p.is_file():
                g4ok = False; g4detail[name] = "MISSING"; continue
            d = load(str(p))
            g4detail[name] = {"verdict": d.get("verdict"), "n_red": d.get("n_red")}
            if d.get("verdict") != "CLEAN" or d.get("n_red") != 0:
                g4ok = False
        add("g4 四稿 CLEAN", g4ok, g4detail)

    if args.smoke:
        s = load(args.smoke)
        add("SMOKE verdict==PASS", s.get("verdict") == "PASS", s.get("verdict"))
        add("SMOKE 冷启动 2 场地 ≥3 案例", (s.get("enumeration") or {}).get("cases_loaded", 0) >= 3
            and (s.get("enumeration") or {}).get("jardim_cases", 0) >= 1
            and (s.get("enumeration") or {}).get("perozzo_cases", 0) >= 2, s.get("enumeration"))
        add("SMOKE 零外部请求", (s.get("enumeration") or {}).get("external_requests_runtime") == 0,
            (s.get("enumeration") or {}).get("external_requests_runtime"))
        add("SMOKE 无残留进程", (s.get("shutdown") or {}).get("port_closed") is True
            and (s.get("shutdown") or {}).get("residual_pid") is None, s.get("shutdown"))

    if args.offline:
        o = load(args.offline)
        add("OFFLINE verdict==PASS", o.get("verdict") == "PASS", o.get("verdict"))
        add("OFFLINE app 外部引用 0", ((o.get("strata") or {}).get("app_external_hits", 1) == 0)
            and len(((o.get("m6_deck") or {}).get("external_hits") or [])) == 0, o.get("verdict"))
        add("OFFLINE 本地引用 0 缺失", not ((o.get("strata") or {}).get("missing_local_refs")
                                        or (o.get("m6_deck") or {}).get("missing_local_refs")))

    if not checks:
        print("NO_INPUTS", file=sys.stderr)
        return 9

    failed = [c["check"] for c in checks if not c["ok"]]
    verdict = "SIGNED_PASS" if not failed else "REFUSED"
    out = {
        "schema": "m7-signoff/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "failed": failed,
        "alert": None if not failed else (
            f"REFUSE_SIGNOFF: M3 DISAGREE={load(args.m3_verdict).get('counts', {}).get('DISAGREE')}，"
            "终审不得签全 PASS" if m3_bad and args.m3_verdict else f"REFUSE_SIGNOFF: {failed}"
        ),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "failed": failed, "alert": out["alert"]}, ensure_ascii=False, indent=1))
    if failed and m3_bad:
        return 7
    return 0 if not failed else 8


if __name__ == "__main__":
    raise SystemExit(main())
