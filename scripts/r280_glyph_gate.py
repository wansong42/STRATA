#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R280 常驻字形闸（R263 A11 的泛化）：PDF 文本层不许出现 .notdef 空位。

判据（预注册 criteria.T2_fix_and_gate.gate_artifact）：
  1. 全库自有 *.pdf 制品文本层 U+0000 == 0 即绿；
  2. 仍带 NUL 的文件必须在 nul_disposition.json 里有属主与登记值，且实测值**只准减不准增**；
  3. 未登记的脏文件 = 直接红（新增即拦，不给"以后再修"留静默通道）；
  4. 第三方文献 PDF 只列不参与判据（外部件不归本单修，且实测它们本来就干净）。

判定核心 evaluate() 是纯函数（输入=实测行+登记表），因此负例毒丸不需要重扫全库。

用法：
  python scripts/r280_glyph_gate.py --mode pdf          # 全库扫描 + 落证据
  python scripts/r280_glyph_gate.py --mode source <md>  # 单份稿件的预渲染普查
  python scripts/r280_glyph_gate.py --poison            # 负例自测（注入必红 / 干净必绿）
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import glyphcover as GC  # noqa: E402

EVID = os.path.join(ROOT, "results", "r280_portal_typography")
DISPOSITION = os.path.join(EVID, "nul_disposition.json")
EXPECT = os.path.join(EVID, "symbol_expectation.json")
SCAN_ROOTS = ("docs", "advisor_pack", "results", "demos", "app", "viz")
EXTERNAL_MARK = ("格式参照", "r149_data_beachhead")


def is_external(path):
    return any(m in path for m in EXTERNAL_MARK)


def find_pdfs():
    """Repo-relative POSIX paths, so they can be registered and re-resolved anywhere."""
    out = []
    for rel in SCAN_ROOTS:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.lower().endswith(".pdf"):
                    out.append(os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/"))
    return sorted(out)


def measure(paths):
    rows = []
    for p in paths:
        r = GC.scan_pdf(os.path.join(ROOT, p))
        r["path"] = p
        rows.append(r)
    return rows


def evaluate(own_rows, register):
    """Pure decision: measured dirty rows vs the owner register. 只准减不准增."""
    reg = {r["path"]: r for r in register}
    clean = [r for r in own_rows if r["nul"] == 0]
    dirty = [r for r in own_rows if r["nul"] > 0]
    unregistered, worsened, fixed, ok_reg = [], [], [], []
    for r in dirty:
        d = reg.get(r["path"])
        if d is None:
            unregistered.append({"path": r["path"], "nul": r["nul"], "pages": r["pages"]})
        elif r["nul"] > d["registered_nul"]:
            worsened.append({"path": r["path"], "measured": r["nul"],
                             "registered": d["registered_nul"], "owner": d.get("owner")})
        else:
            ok_reg.append({"path": r["path"], "nul": r["nul"], "owner": d.get("owner"),
                           "note": d.get("note")})
    for r in clean:
        d = reg.get(r["path"])
        if d and d["registered_nul"] > 0:
            fixed.append({"path": r["path"], "was": d["registered_nul"], "owner": d.get("owner")})
    stale = [p for p in reg if p not in {r["path"] for r in own_rows}]
    verdict = "PASS" if not unregistered and not worsened else "FAIL"
    return {"own_pdfs": len(own_rows), "clean": len(clean), "dirty": len(dirty),
            "verdict": verdict,
            "unregistered_new_dirty": unregistered,
            "worsened_beyond_registration": worsened,
            "repaired_since_registration": fixed,
            "register_paths_no_longer_present": stale,
            "dirty_detail": ok_reg}


def load_register(path=None):
    p = path or DISPOSITION
    if not os.path.exists(p):
        return []
    return json.load(io.open(p, encoding="utf-8"))["register"]


def write_disposition(rows, path=None):
    p = path or DISPOSITION
    payload = {"unit": "R280", "artifact": "nul_disposition",
               "rule": "只准减不准增；实测值高于登记值、或出现未登记的脏 PDF，即红",
               "register": rows, "watermark": "治理判据, 禁入对外承诺文本"}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    io.open(tmp, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False))
    os.replace(tmp, p)


def semantic_report(dirty_rows):
    """Two things the NUL count alone cannot say:

    (1) for every still-dirty PDF, how many of its holes sit in a *meaning* position
        (a minus, an exponent, a hat) - that is what makes the handed-off papers P0, not cosmetic;
    (2) the repaired deliverable must keep its registered symbol multiset.
    """
    per_file = []
    for r in dirty_rows:
        p = os.path.join(ROOT, r["path"])
        try:
            t, _ = measure_symbols(p)
        except Exception as e:
            per_file.append({"path": r["path"], "error": type(e).__name__})
            continue
        hits = shape_scan(t)
        per_file.append({"path": r["path"], "nul": r["nul"],
                         "p0_positions": sum(v["count"] for k, v in hits.items() if v["severity"] == "P0"),
                         "shapes": {k: v["count"] for k, v in hits.items()}})
    exp = {}
    gate = {"pass": True, "missing": {}, "note": "no expectation file on disk yet"}
    if os.path.exists(EXPECT):
        exp = json.load(io.open(EXPECT, encoding="utf-8"))
        _, counts = measure_symbols(os.path.join(ROOT, exp["target_pdf"]))
        gate = symbol_expectation_gate(counts, exp["expect"])
        gate["target_pdf"] = exp["target_pdf"]
        gate["measured"] = counts
    return {"dirty_files_semantic_shape": per_file,
            "deliverable_symbol_expectation": gate,
            "p0_positions_total": sum(f.get("p0_positions", 0) for f in per_file),
            "verdict": "PASS" if gate["pass"] else "FAIL"}


def scan_all():
    all_rows = measure(find_pdfs())
    own = [r for r in all_rows if not is_external(r["path"])]
    external = [r for r in all_rows if is_external(r["path"])]
    dirty = [r for r in own if r["nul"] > 0]
    out = {"unit": "R280", "artifact": "t2_pdf_gate", "scan_roots": list(SCAN_ROOTS),
           "external_third_party_pdfs": len(external),
           "external_with_nul": [{"path": r["path"], "nul": r["nul"]}
                                 for r in external if r["nul"]]}
    out.update(evaluate(own, load_register()))
    out["semantic"] = semantic_report(dirty)
    if out["semantic"]["verdict"] != "PASS":
        out["verdict"] = "FAIL"
    out["watermark"] = "治理判据, 禁入对外承诺文本"
    return out


# ----------------------------------------------------------------- semantic shape rules (裁定79)
# A NUL is not merely an unreadable byte: where it sits tells you which meaning was swallowed.
# These rules are wording-agnostic (they key on the position shape, not on this document's
# sentences), so they keep working for any future manuscript rendered by this toolchain.
SHAPE_RULES = [
    ("sign_before_reading", re.compile(r"\x00\s*\d+(?:\.\d+)?\s*[°%]"),
     "读数前的负号位被吞（『−21.12°』印成『21.12°』= 符号翻转）", "P0"),
    ("exponent_position", re.compile(r"×\s*10\s*\x00"),
     "科学计数法指数位被吞（『1.48×10⁻⁶°』的数量级消失）", "P0"),
    ("one_minus_alpha", re.compile(r"1\s*\x00\s*[αβγ\u0391-\u03c9]"),
     "1−α 的减号位被吞（置信水平读错）", "P0"),
    ("digit_hole_digit", re.compile(r"\d\s*\x00\s*\d"),
     "两数之间的算符位被吞（『1−1/K』读成『11/K』）", "P0"),
    ("combining_or_hat", re.compile(r"[A-Za-z\u0391-\u03c9]\x00"),
     "字母后的尖号/帽子位被吞（K̂ κ̂ n̂ 的估计号消失）", "P1"),
    ("bare_nul", re.compile(r"\x00"), "其余 .notdef 空位", "P1"),
]


def shape_scan(text):
    hits = {}
    for name, rx, meaning, sev in SHAPE_RULES:
        n = text.count(chr(0)) if name == "bare_nul" else len(rx.findall(text))
        if n:
            hits[name] = {"count": n, "meaning": meaning, "severity": sev}
    return hits


def symbol_expectation_gate(counts, expect):
    """Pure: measured codepoint multiplicities vs the registered expectation. 少一个即红。"""
    short = {k: {"expect": v, "measured": counts.get(k, 0)} for k, v in expect.items()
             if counts.get(k, 0) < v}
    grown = {k: {"expect": expect[k], "measured": v} for k, v in counts.items()
             if k in expect and v > expect[k]}
    return {"missing": short, "grown": grown,
            "pass": not short, "note": "grown 只提示不判红（新增符号属内容增长，需登记）"}


def measure_symbols(pdf_path):
    from pypdf import PdfReader
    t = "\n".join((p.extract_text() or "") for p in PdfReader(pdf_path).pages)
    exp = json.load(io.open(EXPECT, encoding="utf-8"))["expect"] if os.path.exists(EXPECT) else {}
    counts = {("U+%04X" % ord(c)): t.count(c) for c in
              {chr(int(k.replace("U+", ""), 16)) for k in exp}}
    return t, counts


def poison():
    """Negatives that must each turn something red, positives that must stay green.

    F-R277-1 discipline: an injection that leaves the bytes unchanged is NOT a pass - the pill
    itself is invalid and is reported red.
    """
    cases = []
    rows = [{"path": "a.pdf", "nul": 5, "pages": 3}, {"path": "b.pdf", "nul": 0, "pages": 2}]
    base = evaluate(rows, [{"path": "a.pdf", "registered_nul": 5, "owner": "X"}])
    cases.append({"pill": "N1 dirty file without owner registration", "must": "FAIL",
                  "got": evaluate(rows, [])["verdict"],
                  "valid": base["verdict"] == "PASS" and evaluate(rows, [])["verdict"] == "FAIL"})
    worse = [{"path": "a.pdf", "nul": 6, "pages": 3}]
    cases.append({"pill": "N2 measured NUL above registered value", "must": "FAIL",
                  "got": evaluate(worse, [{"path": "a.pdf", "registered_nul": 5, "owner": "X"}])["verdict"],
                  "valid": evaluate(worse, [{"path": "a.pdf", "registered_nul": 5, "owner": "X"}])["verdict"] == "FAIL"})
    fixed = [{"path": "a.pdf", "nul": 0, "pages": 3}, {"path": "b.pdf", "nul": 0, "pages": 2}]
    e3 = evaluate(fixed, [{"path": "a.pdf", "registered_nul": 5, "owner": "X"}])
    cases.append({"pill": "N3 registered defect disappears (repaired)", "must": "PASS + recorded",
                  "got": e3["verdict"], "repaired": e3["repaired_since_registration"],
                  "valid": e3["verdict"] == "PASS" and len(e3["repaired_since_registration"]) == 1})
    cases.append({"pill": "P0 clean baseline", "must": "PASS", "got": base["verdict"],
                  "valid": base["verdict"] == "PASS"})

    # shape rules: the defective v1.1 text must light up, the repaired v1.2 must not
    p11 = os.path.join(ROOT, "docs", "成果总报告", "研究工作报告_v1.1.pdf")
    p12 = os.path.join(ROOT, "docs", "成果总报告", "研究工作报告_v1.2.pdf")
    h1 = {}
    if os.path.exists(p11):
        t11, _ = measure_symbols(p11)
        h1 = shape_scan(t11)
    if os.path.exists(p12):
        t12, c12 = measure_symbols(p12)
        h2 = shape_scan(t12)
        cases.append({"pill": "N4 shape rules catch the real defect in v1.1",
                      "must": ">=2 rule classes hit", "got": sorted(h1),
                      "valid": len([k for k in h1 if k != "bare_nul"]) >= 2})
        cases.append({"pill": "P1 repaired v1.2 has no shape hit", "must": "no hits",
                      "got": sorted(h2), "valid": not h2})
        # N5: delete one minus from the ARTIFACT text (renderer swallowing it) -> C1 red
        exp = json.load(io.open(EXPECT, encoding="utf-8"))["expect"]
        minus = chr(0x2212)
        cut = t12.replace(minus, "", 1)
        mutated = cut != t12
        c_cut = dict(c12)
        c_cut["U+2212"] = c_cut.get("U+2212", 0) - 1
        g = symbol_expectation_gate(c_cut, exp)
        cases.append({"pill": "N5 delete one minus from the artifact", "must": "missing U+2212",
                      "injected_differs_from_base": mutated, "got": g["missing"],
                      "valid": mutated and "U+2212" in g["missing"]})
        # N6: delete one minus from the SOURCE manuscript -> gate red on the same expectation
        md12 = io.open(os.path.join(ROOT, "docs", "成果总报告", "研究工作报告_v1.2.md"),
                       encoding="utf-8").read()
        src_cut = md12.replace(minus, "", 1)
        sc = {("U+%04X" % ord(c)): src_cut.count(c) for c in
              {chr(int(k.replace("U+", ""), 16)) for k in exp}}
        g2 = symbol_expectation_gate(sc, exp)
        cases.append({"pill": "N6 delete one minus from the manuscript", "must": "missing U+2212",
                      "injected_differs_from_base": src_cut != md12, "got": g2["missing"],
                      "valid": src_cut != md12 and "U+2212" in g2["missing"]})
    # N7: a codepoint no font carries must make the renderer refuse
    GC.DEFECTS.clear()
    GC.route("\u2be0", "Math")
    raised = False
    try:
        GC.assert_clean("poison pill N7")
    except SystemExit:
        raised = True
    cases.append({"pill": "N7 codepoint no installed font carries", "must": "SystemExit",
                  "got": "raised" if raised else "silently accepted", "valid": raised})
    ok = all(c["valid"] for c in cases)
    return cases, ok


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv else "--mode"
    if mode == "--mode" and len(argv) > 2 and argv[2] == "source":
        md = argv[3]
        rows = GC.audit_source(io.open(md, encoding="utf-8").read())
        bad = [r for r in rows if "NOT_ROUTABLE" in (r["as_Song"], r["as_Hei"], r["as_Times"])]
        for r in rows:
            print("U+%04X x%-4d Song=%-8s Hei=%-8s Times=%-8s"
                  % (r["cp"], r["n"], r["as_Song"], r["as_Hei"], r["as_Times"]))
        print("source", md, "rows", len(rows), "unroutable", len(bad))
        return 0 if not bad else 1
    if mode == "--poison":
        cases, ok = poison()
        for c in cases:
            print(json.dumps(c, ensure_ascii=False, allow_nan=False))
        print("POISON", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    out = scan_all()
    os.makedirs(EVID, exist_ok=True)
    dest = os.path.join(EVID, "t2_pdf_gate.json")
    tmp = dest + ".tmp"
    io.open(tmp, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False))
    os.replace(tmp, dest)
    print("own pdfs %d | clean %d | dirty %d | external %d (nul in external: %d) | verdict %s"
          % (out["own_pdfs"], out["clean"], out["dirty"], out["external_third_party_pdfs"],
             len(out["external_with_nul"]), out["verdict"]))
    for d in out["dirty_detail"]:
        print("   NUL %-4d %-62s owner=%s" % (d["nul"], d["path"][:62], d["owner"]))
    for d in out["unregistered_new_dirty"]:
        print("   UNREGISTERED NUL %-4d %s" % (d["nul"], d["path"]))
    sm = out["semantic"]
    print("semantic: P0 meaning-positions still swallowed = %d | deliverable expectation = %s"
          % (sm["p0_positions_total"], sm["deliverable_symbol_expectation"]["pass"]))
    for f in sorted(sm["dirty_files_semantic_shape"], key=lambda x: -x.get("p0_positions", 0))[:6]:
        if f.get("p0_positions"):
            print("   P0-positions %-3d nul %-3d %s" % (f["p0_positions"], f["nul"], f["path"][:60]))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
