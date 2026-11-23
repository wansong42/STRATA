"""R308 T3 常驻把守 —— 「K=4→16 平坦」端点的三标签同段闸 (有牙).

    python scripts/r308_check_flat_label.py [--target 判红面文件...]
                                           [--governance-target 治理面文件...]
                                           [--json 输出.json] [--fail-on-red]

规则本体 (任务书 T3 + 预注册 J4): 出现「K=4→16 平坦」类**断言** ⇒ 同段必须同时在场三标签
  L1 非精度声明 ｜ L2 列名与口径 (coverage_honest_assign_orient / 逐点方向 / 最近组心)
  ｜ L3 与缺口曲线的主从关系 (指派缺口 + 旁证/主论证类词) —— 缺任一即红, 无部分分。

判据不内联: 触发 token / 三标签 / 豁免标记 100% 从 results/r308_kflat_adjudication/prereg.json 解析,
解析不到即 SystemExit (宁红不放过)。触发定位法见 prereg_amendments_4 (同小句共现 + 引用识别 + 标记相邻)。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = "results/r308_kflat_adjudication/prereg.json"
DEFAULT_JUDGED = ["docs/论文_墙_发现过程技术报告.md"]

QUOTE_PAIRS = [(u"“", u"”"), ('"', '"'), (u"「", u"」"), ("`", "`"), (u"‘", u"’")]
CLAUSE_RX = re.compile(u"[。；;！!？，,、]")


def _pre():
    p = os.path.join(ROOT, PREREG)
    if not os.path.exists(p):
        raise SystemExit("预注册不在盘 (%s) —— 本闸拒绝以内置默认判据运行" % PREREG)
    d = json.load(io.open(p, encoding="utf-8"))
    for k in ("labels", "triggers", "exemptions"):
        if not d.get(k):
            raise SystemExit("预注册缺节 %s" % k)
    for k in ("T_flat", "T_num", "T_range"):
        if k not in d["triggers"]:
            raise SystemExit("预注册 triggers 缺 %s" % k)
    return d


def split_units(text):
    """unit = 空行分隔的段; Markdown 表格按**行**成 unit (同表不同行不得互为限定语)。"""
    units, buf = [], []

    def flush():
        if buf:
            body = "\n".join(x[1] for x in buf)
            if body.strip():
                units.append((buf[0][0], body))
            del buf[:]

    for i, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            flush()
            continue
        if ln.lstrip().startswith("|"):
            flush()
            units.append((i, ln))
            continue
        buf.append((i, ln))
    flush()
    return units


def clauses_of(unit):
    """小句: 表格行先按 | 分格, 再按句末标点与顿逗断句 (修正案 4 (a)(c) 的『同小句』口径)。"""
    cells = unit.split("|") if unit.lstrip().startswith("|") else [unit]
    out = []
    for cell in cells:
        for part in CLAUSE_RX.split(cell):
            for ln in part.splitlines():
                if ln.strip():
                    out.append(ln.strip())
    return out


def positions(text, toks):
    hits = []
    for t in toks:
        s = text.find(t)
        while s >= 0:
            hits.append((s, t))
            s = text.find(t, s + len(t))
    return hits


def in_quote(text, pos):
    for op, cl in QUOTE_PAIRS:
        i = 0
        while True:
            o = text.find(op, i)
            if o < 0:
                break
            c = text.find(cl, o + len(op))
            if c < 0:
                break
            if o <= pos < c:
                return op + cl
            i = o + len(op)
    return None


def has_any(text, marks):
    return next((m for m in marks if m in text), None)


def label_present(text, spec):
    main = has_any(text, spec.get("regex") or [])
    if not main:
        return False, None
    also = spec.get("regex_also")
    if also:
        sub = has_any(text, also)
        return (sub is not None), [main, sub]
    return True, [main]


def classify(unit, pre):
    """返回 (state, evidence)。state ∈ NOT_TRIGGERED / MENTION / ASSERTION。

    触发条件 (修正案 4 (a)): 同小句内 ①平坦词 且 ②(K 区间形 或 端点读数);
                             或 ③K 区间形 与 端点读数 同小句共现 (裸数字形断言, 不靠"平坦"二字)。
    豁免 (修正案 4 (b)(c)): 触发串落在引号内 = 转述; 或**同小句**内有已登记否定/禁写标记。
    """
    trig = {g: positions(unit, toks) for g, toks in pre["triggers"].items()}
    if not any(trig.values()):
        return "NOT_TRIGGERED", {"trigger_positions": {}}
    marks = [m for ml in pre["exemptions"].values() for m in ml]
    base = unit.find("")  # 0
    asserts, mentions, esc = [], [], []
    for cl in clauses_of(unit):
        off = unit.find(cl, base)
        off = 0 if off < 0 else off
        base = off + max(len(cl), 1)
        flats = positions(cl, pre["triggers"]["T_flat"])
        nums = positions(cl, pre["triggers"]["T_num"])
        rngs = positions(cl, pre["triggers"]["T_range"])
        forms = []
        if flats and (nums or rngs):
            forms.append(("平坦词×(K区间|端点)", flats + nums + rngs))
        if rngs and nums:
            forms.append(("K区间×端点(裸数字形)", flats + nums + rngs))
        if not forms:
            if flats:
                esc.append({"clause": cl[:160], "why": "孤立平坦词: 同小句内无 K 区间形与端点读数共现"})
            continue
        for label, toks in forms:
            for pos, tok in sorted(set(toks)):
                abs_pos = off + pos
                q = in_quote(unit, abs_pos)
                neg = has_any(cl, marks)
                rec = {"clause": cl[:160], "form": label, "token": tok,
                       "quote_pair": q, "negation_mark_in_clause": neg}
                (mentions if (q or neg) else asserts).append(rec)
    state = "ASSERTION" if asserts else ("MENTION" if mentions else "NOT_TRIGGERED")
    ev = {"assertions": asserts, "mentions": mentions, "escaped_forms": esc,
          "trigger_positions": {g: sorted({t for _p, t in v}) for g, v in trig.items() if v}}
    return state, ev


def scan_text(text, pre, source=u"<text>"):
    rows = []
    for ln, unit in split_units(text):
        state, ev = classify(unit, pre)
        if state == "NOT_TRIGGERED":
            continue
        labs = {k: label_present(unit, spec) for k, spec in pre["labels"].items()}
        missing = sorted(k for k, ok in labs.items() if not ok[0])
        rec = {"source": source, "unit_first_line": ln, "state": state,
               "labels_present": {k: (v[1] if v[0] else None) for k, v in labs.items()},
               "labels_missing": missing, "evidence": ev, "quote": unit[:400]}
        if state == "MENTION":
            rec["severity"] = "MENTION_DISCLOSED"
        else:
            rec["severity"] = "OK" if not missing else "RED"
        rows.append(rec)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", help="判红面 (交付正文/作者面)")
    ap.add_argument("--governance-target", action="append", help="治理语境面 (只计数不判红, R287 D-4 设计)")
    ap.add_argument("--json")
    ap.add_argument("--fail-on-red", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    pre = _pre()
    judged, gov = [], []
    for t in (a.target if a.target is not None else DEFAULT_JUDGED):
        judged.extend(scan_text(_read(t), pre, t))
    for t in (a.governance_target or []):
        gov.extend(scan_text(_read(t), pre, t))
    red = [r for r in judged if r["severity"] == "RED"]
    payload = {"gate": u"R308-A 平坦端点三标签同段闸", "prereg": PREREG,
               "prereg_sha8": hashlib.sha256(open(os.path.join(ROOT, PREREG), "rb").read()).hexdigest()[:8],
               "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "faces": {"judged": a.target if a.target is not None else DEFAULT_JUDGED,
                         "governance": a.governance_target or []},
               "counts": {"judged_triggered": len(judged),
                          "judged_ok": sum(1 for r in judged if r["severity"] == "OK"),
                          "judged_mention": sum(1 for r in judged if r["severity"] == "MENTION_DISCLOSED"),
                          "judged_red": len(red),
                          "governance_triggered": len(gov),
                          "governance_would_be_red": sum(1 for r in gov if r["severity"] == "RED")},
               "rows_judged": judged, "rows_governance": gov, "watermark": pre["watermark"]}
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        tmp = dst + ".tmp"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, dst)
    if not a.quiet:
        print(json.dumps(payload["counts"], ensure_ascii=False, allow_nan=False))
        for r in red[:12]:
            print(u"RED", r["source"], u"L%s" % r["unit_first_line"], u"缺", r["labels_missing"],
                  u"|", r["quote"][:90].replace("\n", u" ⏎ "))
    if a.fail_on_red and red:
        return 1
    return 0


def _read(t):
    p = t if os.path.isabs(t) else os.path.join(ROOT, t)
    return io.open(p, encoding="utf-8", errors="replace").read()


if __name__ == "__main__":
    sys.exit(main())
