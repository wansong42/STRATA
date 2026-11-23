"""R308 T3b 常驻把守 —— oracle 覆盖列 ↔ 指派缺口列 身份互斥闸 (有牙, 零内联常数).

    python scripts/r308_identity_gate.py [--target 文件...] [--json 输出.json] [--fail-on-red]

背景 (变更令 R308-2 要求 2, 判据本体见 prereg_amendments_5.json):
  forensics#refutation/R4_assignment_gap.K.4 同一节点相邻给出
      coverage_oracle_assign = 19.157   |   gap_orient = 19.233      ⇒ 只差 0.076°
  两列在 K=4 几乎同值 ⇒ 『把 oracle 覆盖当成指派缺口』是可抄出来的错, 且抄错即语义翻转。

规则 R308-B: 同一小句内 (身份标签, 数值) 成对出现 ——
  标签 ∈ gap 族 且 数值 ∈ oracle覆盖集  ⇒ RED
  标签 ∈ oracle覆盖族 且 数值 ∈ gap集   ⇒ RED
两侧值集**运行时直读** forensics / r204 两件 (零内联数字); 若两集出现同名 token ⇒ SystemExit
(= 该处不可判, 拒绝猜)。豁免按小句配对, 不做全文集合。
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import r308_check_flat_label as G                      # noqa: E402  (复用 unit/clause 切分, 不另写一套)

FORENSICS = "results/r202_mace_mixture/forensics.json"
T1JSON = "results/r204_set_metric_truth_gate/t1_dual_caliber.json"
V204 = "results/r204_set_metric_truth_gate/verdict.json"

GAP_LABELS = [u"gap_k", "gap_orient", "assignment_gap", u"指派缺口", u"缺口", u"泄漏量", "leak_amount"]
ORACLE_LABELS = ["coverage_oracle_assign", u"oracle 指派覆盖", u"oracle指派覆盖", u"oracle 覆盖",
                 u"oracle覆盖", "oracle_K", "on_disk_t54", u"oracle 档", u"oracle", u"真值指派",
                 u"oracle 指派"]


def _read(rel):
    return json.load(io.open(os.path.join(ROOT, rel), encoding="utf-8"))


def _forms(v):
    out = set()
    for s in (repr(v), str(v), u"%.3f" % v, u"%.4f" % v):
        out.add(s.rstrip(u"0").rstrip(u"."))
        out.add(s)
    return {x for x in out if x and len(x) >= 6}       # 短 token 会撞普通数字, 不入集


def value_sets():
    gap, oracle = set(), set()
    f = _read(FORENSICS)
    for node in (f["refutation"]["R4_assignment_gap"]["K"], f["A_curve_beishan_K"]):
        for _k, cell in node.items():
            if not isinstance(cell, dict):
                continue
            for key, bag in ((u"gap_orient", gap), ("coverage_oracle_assign", oracle)):
                if isinstance(cell.get(key), (int, float)):
                    bag |= _forms(cell[key])
    t = _read(T1JSON)
    for _k, cell in t["curve_beishan"].items():
        if isinstance(cell, dict):
            for key, bag in ((u"assignment_gap", gap), (u"oracle_caliber", oracle)):
                if isinstance(cell.get(key), (int, float)):
                    bag |= _forms(cell[key])
    for _k, v in t.get("leak_amount", {}).items():
        if isinstance(v, (int, float)):
            gap |= _forms(v)
    if isinstance(t.get("headline", {}).get("on_disk_t54"), (int, float)):
        oracle |= _forms(t["headline"]["on_disk_t54"])
    e1 = _read(V204)["criteria"]["E1"]
    for _k, v in (e1.get("gaps") or {}).items():
        if isinstance(v, (int, float)):
            gap |= _forms(v)
    if isinstance(e1.get("oracle_K12"), (int, float)):
        oracle |= _forms(e1["oracle_K12"])
    amb = sorted(gap & oracle)
    if amb:
        raise SystemExit(u"两列值集出现同名 token %s ⇒ 该处不可判身份, 拒绝猜测" % amb)
    return gap, oracle


def _all_positions(text, toks):
    hits = []
    for t in toks:
        s = text.find(t)
        while s >= 0:
            hits.append((s, t))
            s = text.find(t, s + len(t))
    return hits


def _label_rx():
    lab = (r"coverage_oracle_assign|assignment_gap|leak_amount|gap_orient|gap_k\d+|oracle_K\d+"
           + r"|on_disk_t54|" + u"指派缺口|真值指派|泄漏量|缺口|oracle\\s*指派\\s*覆盖|oracle\\s*覆盖"
           + r"|oracle\\s*档|oracle")
    return (re.compile(u"(?P<label>%s)(?P<gap>[^0-9\\n]{0,14}?)(?P<val>\\d+\\.\\d+)" % lab),
            re.compile(u"(?P<val>\\d+\\.\\d+)\\s*(?:%s)\\s*(?P<label>%s)"
                       % (u"是|为|属于|属|等于|=", lab)))


_BIND_F, _BIND_B = _label_rx()


_BIND_F, _BIND_B = _label_rx()

# 豁免只认『标签与数值之间』那段文字里的否定词: "缺口三元不得写成 19.157…" 是**禁止句式**不是错配;
# 否定词在标签之外 (如 "别写 gap_k12 = 12.5152") 仍判红 —— 那种写法本身就把错配值原样断言了出来。
NEG_MARKS = [u"不得", u"不是", u"并非", u"不可", u"勿", u"而非", u"而不是", u"误作", u"误将",
             u"混淆", u"错当", u"当成", u"当作", u"别把", u"不要把", u"不应", u"误读", u"误写"]


def _negated(between):
    return next((m for m in NEG_MARKS if m in (between or u"")), None)


def _family(tok, gap, oracle):
    if tok in gap:
        return u"gap"
    if tok in oracle:
        return u"oracle"
    return None


def judge_clause(cl, gap, oracle):
    """身份绑定式判定 (不用『最近标签』: 合规句『…− oracle 19.1571° (与 … 3 位版 19.233 同源)』
    会被最近标签法误杀, 已实测)。只在**标签与数值直接配对**时判:
      标签 + ≤14 个非数字字符 + 数值   (正接)
      数值 + 是/为/= + 标签             (倒装)
    配对不上就不判 ⇒ 精确优先; 牙由毒丸钉死 (见 t3b_identity_dual_test.json)。"""
    out = []
    for rx in (_BIND_F, _BIND_B):
        for m in rx.finditer(cl):
            lab, val = m.group(u"label"), m.group(u"val")
            v0, v1 = m.span(u"val")
            l0, l1 = m.span(u"label")
            between = cl[l1:v0] if v0 >= l1 else cl[v1:l0]
            if _negated(between):
                continue
            lf = u"oracle" if any(k in lab for k in
                                  (u"oracle", u"真值指派", u"coverage_oracle_assign", u"on_disk_t54")) \
                else u"gap"
            vf = _family(val, gap, oracle)
            if vf and vf != lf:
                out.append({u"token": val, u"value_family": vf, u"label": lab,
                            u"label_family": lf, u"span": [m.start(), m.end()],
                            u"binding": u"forward" if rx is _BIND_F else u"backward"})
    seen, uniq = set(), []
    for r in out:
        k = (r[u"token"], r[u"label"], r[u"binding"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def scan_text(text, gap, oracle, source=u"<text>"):
    rows = []
    for ln, unit in G.split_units(text):
        for cl in G.clauses_of(unit):
            bad = judge_clause(cl, gap, oracle)
            if bad:
                rows.append({u"source": source, u"line": ln, u"clause": cl[:220],
                             u"mismatches": bad, u"severity": u"RED"})
    return rows


def _sha8(rel):
    return hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()[:8]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append")
    ap.add_argument("--json")
    ap.add_argument("--fail-on-red", action="store_true")
    a = ap.parse_args(argv)
    gap, oracle = value_sets()
    targets = a.target or [u"docs/论文_墙_发现过程技术报告.md",
                           u"results/r308_kflat_adjudication/carrier_append_pending_R308-E1.md"]
    rows = []
    for t in targets:
        rows.extend(scan_text(G._read(t), gap, oracle, t))
    payload = {u"gate": u"R308-B oracle覆盖列 ↔ 指派缺口列 身份互斥闸",
               u"measured_at": time.strftime(u"%Y-%m-%dT%H:%M:%S%z", time.localtime()),
               u"value_sets_from_disk": {u"n_gap_tokens": len(gap), u"n_oracle_tokens": len(oracle),
                                         u"gap_sample": sorted(gap)[:6], u"oracle_sample": sorted(oracle)[:6]},
               u"sources": [FORENSICS, T1JSON, V204],
               u"sources_sha8": {p: _sha8(p) for p in (FORENSICS, T1JSON, V204)},
               u"targets": targets, u"rows": rows, u"n_red": len(rows),
               u"excluded_self_referential_artifacts": [
                   {u"path": u"results/r308_kflat_adjudication/author_claim_ruling.json",
                    u"why": u"裁定件按任务书要求**必须逐条列出被禁句式** (CA-09), 属转述形; "
                             u"其自合规性由 t3b 双测的 PP_TRIO(必红) 与 POS_INSIDE_NEGATION(必绿) 钉死, "
                             u"不是靠从扫描面里删掉来变绿"}],
               u"watermark": u"研究线判据 · 禁入对外承诺文本"}
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        tmp = dst + u".tmp"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with io.open(tmp, u"w", encoding=u"utf-8", newline=u"\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, dst)
    print(u"n_red=%d (gap tokens %d / oracle tokens %d, 全部运行时直读)"
          % (len(rows), len(gap), len(oracle)))
    for r in rows[:8]:
        print(u"RED", r[u"source"], u"L%s" % r[u"line"], u"|", r[u"clause"][:90])
    return 1 if (a.fail_on_red and rows) else 0


if __name__ == "__main__":
    sys.exit(main())
