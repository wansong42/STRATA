"""R303 作者成稿体检器 —— 九规则 (R1–R9) 实现 + CLI + 人读报告.

    python scripts/r303_draft_check.py --draft <稿子.md|.docx> [--out <目录>] [--json-only]

判级: 任何 **RED 且未获语境豁免** 的命中 ⇒ 稿件非 CLEAN。豁免必须带类名 + 匹配到的语境
标记原文 (未登记 = 误杀, 见 prereg#judges.J3)。每条命中给四件: 规则号 / 原文 / 为什么错 /
正确写法指针。规则定义与豁免类由 `results/r303_draft_check/prereg.json` 冻结; 本件是其实现。

R317 接入: 另有两条 **整条 import 属主判据** 的契约 (R308-A / R308-B, 见 r308_contracts),
与 R4 复用 R263 同族 —— 判据与值集不在本件, 本件只做条目搬运。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r303_kit as K                                            # noqa: E402
from r303_kit import (AUTH, DOI_IN_TEXT_RX, IDENT_BEFORE_RX, INTERVAL_RX,  # noqa: E402
                      NUMBER_RX, PATH_RX, UNIT_CTX_RX, Whitelist, abspath, dp_forms,
                      exemption_for, jload, line_sections, now_iso, units_of, write_json,
                      write_text)


# --------------------------------------------------------------------------- 命中构造
def _f(rid, u, why, pointer, tokens, severity="RED"):
    cls, mark, reason = exemption_for(rid, u)
    return {"rule": rid, "line": u.line, "quote": u.sent[:300], "why": why,
            "pointer": pointer, "tokens": sorted(set(tokens))[:8], "severity": severity,
            "exempt": cls is not None, "exempt_class": cls, "exempt_mark": mark,
            "exempt_reason": reason, "section": u.section}


def _tok_out_of_scope(sc, m):
    """标识符位 (DOI / arXiv / 路径内数字) 不是测量读数 —— 坑 90 的作用域排除。"""
    tok = m.group(1)
    if IDENT_BEFORE_RX.search(sc[max(0, m.start() - 14):m.start()]):
        return True
    if DOI_IN_TEXT_RX.search(sc[max(0, m.start() - 6):m.end() + 8]):
        return True
    return bool(PATH_RX.search(sc[max(0, m.start() - 40):m.end()]))


# --------------------------------------------------------------------------- R1 数字闭包
def r1_closure(text, wl):
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        for m in NUMBER_RX.finditer(sc):
            tok = m.group(1)
            if _tok_out_of_scope(sc, m) or wl.has(tok):
                continue
            tail = sc[m.end():m.end() + 3]
            near, d = wl.nearest(tok)
            if near:
                rid = "R1b"
                why = ("疑似舍入/抄写偏差: 最近的在册值是 %s (源 %s), 相对差 %.2f%%, "
                       "但本读数不等于任何在册值的合规舍入" % (near["value"], near["src"], d * 100))
            else:
                rid = "R1a"
                why = ("无源小数读数: 六层白名单 (number_kit / 台账 / 取料表 / 在册源件 / "
                       "稿内引用件 / 授权散文载体) 均无此值, 亦非其合规舍入")
            ptr = AUTH["number_kit"] + " (按 key 取行) + 白名单生长层见 prereg#whitelist"
            if not UNIT_CTX_RX.match(tail):
                f = _f(rid, u, why, ptr, [tok], severity="NOTE")
                f["exempt"] = False
                out.append(f)
                continue
            out.append(_f(rid, u, why, ptr, [tok]))
    return out


# --------------------------------------------------------------------------- R2 五处混拼形状
def r2a_ci_mismatch(text):
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        toks = NUMBER_RX.findall(sc)
        if len(toks) < 3:
            continue
        for p in K.ci_pairs():
            lo_f = dp_forms([p["lo"]], dps=(1, 2, 3, 4))
            hi_f = dp_forms([p["hi"]], dps=(1, 2, 3, 4))
            if not (any(t in lo_f for t in toks) and any(t in hi_f for t in toks)):
                continue
            pt_f = dp_forms([p["pt"]], dps=(1, 2, 3, 4))
            if set(toks) & pt_f:
                continue        # 本句自带该区间所属点估计 ⇒ 不是"把别的点估计拼上这条区间"
            for b in toks:
                if b in lo_f or b in hi_f or float(b) <= 1.0:
                    continue
                if abs(float(b) - p["pt"]) / p["pt"] > 0.25:
                    continue
                out.append(_f("R2a", u,
                              "P0-1 跨件混拼: 区间 [%s, %s] 由点估计 %s (%s) 同一次重抽得出, "
                              "本句却把 %s 当点估计与它相拼 ⇒ 点估计与区间不同源" % (
                                  p["lo"], p["hi"], p["pt"], p["key"], b),
                              p["src"] + " (点估计与其 CI 在同一行配对)", [b]))
    return out


def r2b_cross_corpus(text):
    pair = K.routeb_pair()
    if "routeb_with_id" not in pair or "routeb_without_id" not in pair:
        return []
    with_id, wo = pair["routeb_with_id"], pair["routeb_without_id"]
    wf = dp_forms([with_id["value"]], dps=(2, 3, 4, 5, 6))
    of = dp_forms([wo["value"]], dps=(1, 2, 3, 4))
    COMPARE_RX = re.compile(r"vs|对比|对照|夹逼|相比|有标签|无标签|有 ?id|无 ?id|fracture_id", re.I)
    blocks = K.table_blocks(text)
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        toks = set(NUMBER_RX.findall(sc))
        if not (toks & wf) or not COMPARE_RX.search(sc):
            continue
        around = sc + " " + u.para_clean + " " + blocks.get(u.line, "")
        if set(NUMBER_RX.findall(around)) & of:
            continue
        out.append(_f("R2b", u,
                      "P0-2 跨语料夹逼: 同语料配对的另一半 %s (无 fracture_id, 源 %s) 未在本句/本段"
                      "给出 ⇒ 无法排除是拿别的语料的墙高来夹" % (wo["value"], wo["src"]),
                      with_id["src"], sorted(toks & wf)))
    return out


FLAT_RX = re.compile(r"平坦|起伏|flat|整段", re.I)
KRANGE_RX = re.compile(r"K\s*=\s*\d+\s*(?:[到–—~～\-]|至|→)\s*\d+|阶[^。]{0,14}(?:四|4)[^。]{0,10}"
                       r"(?:十六|16)|从[一二三四五六七八九十\d]+到[一二三四五六七八九十\d]+")


def r2c_kgrid_claim(text):
    """判"整段平坦"这类断言的区间端点是否等于在盘 K 档注册端点 (只看区间, 不牵连同句其他数)。"""
    bounds = K.k_grid_bounds()
    bf = dp_forms(bounds, dps=(1, 2, 3))
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        if not (KRANGE_RX.search(sc) and FLAT_RX.search(sc)):
            continue
        iv = INTERVAL_RX.findall(sc)
        if not iv:
            out.append(_f("R2c", u,
                          "P0-3 整段断言未给出在盘区间: 在盘 K 档注册端点为 %s, 本句只作定性断言"
                          % bounds, AUTH["number_table"] + " rows[src 含 min_K/max_K]", [],
                          severity="NOTE"))
            continue
        bad = [t for ab in iv for t in ab if t not in bf]
        if not bad:
            continue
        out.append(_f("R2c", u,
                      "P0-3 整段断言与盘上区间不符: 在盘 K 档注册端点 %s, 本句区间端点却是 %s"
                      % (bounds, sorted(set(bad))),
                      AUTH["number_table"] + " rows[src 含 min_K/max_K]", bad))
    return out


def r2d_protocol_mix(text):
    alts = K.protocol_alternates()
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        toks = set(NUMBER_RX.findall(sc))
        for a in alts:
            hit = sorted(toks & dp_forms([a["value"]], dps=(1, 2, 3, 4)))
            if not hit:
                continue
            out.append(_f("R2d", u,
                          "P1-1 跨掩码协议混比: %s (=%s, 口径「%s」) 与诚实榜不同协议, 只允许出现在"
                          "协议敏感性说明位; 在册注记: %s" % (a["key"], a["rendered"],
                                                            a["sample"], a["note"]),
                          a["src"], hit))
    return out


def r2e_heading(text):
    out = []
    for u in units_of(text):
        if not re.match(r"^\s*(#{1,6}\s|\*\*)", u.sent):
            continue
        if "下界" in u.sent and re.search(r"验证|verify|validat", u.sent, re.I):
            out.append(_f("R2e", u,
                          "P1-2 标题把理论当已验证: 理论下界是数学对象 (证明状态另行如实披露), "
                          "实测墙只谈证据链; 建议改「经验墙的证据链」",
                          AUTH["carrier"] + " 第 0 章 P1-2 行", ["下界+验证"]))
    return out


# --------------------------------------------------------------------------- R3 禁引面
def r3a_deprecated(text):
    dep, dep_srcs = K.deprecated_tokens()
    dec = {t for t in dep if not t.endswith("°")}
    deg_ints = {t for t in dep if t.endswith("°")}
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        hits = [t for t in NUMBER_RX.findall(sc) if t in dec]
        for t in deg_ints:
            core = re.escape(t[:-1])
            hits.extend(t for _ in re.finditer(r"(?<![\d.])%s(?![\d.])\s*°" % core, sc))
        for h in sorted(set(hits)):
            out.append(_f("R3a", u,
                          "判废旧值 %s 出现在稿中 (载体第 3 章口径: 出现即错; 清单本体与沿革留档除外)" % h,
                          " ∪ ".join(dep_srcs), [h]))
    return out


def r3b_oracle(text):
    out = []
    rows = [(dp_forms([o["value"]], dps=(2, 3, 4)), o) for o in K.oracle_tokens()]
    for u in units_of(text):
        sc = u.sent_clean
        toks = set(NUMBER_RX.findall(sc))
        for fs, o in rows:
            hit = sorted(toks & fs)
            if hit:
                out.append(_f("R3b", u,
                              "oracle 面读数 %s (=%s) 裸用: %s ⇒ 只能作归因证据并显式标注不可部署"
                              % (hit[0], o["rendered"], o["note"]), o["src"], hit))
    return out


# --------------------------------------------------------------------------- R4 两栏混引 (整条 import)
def r4_two_column(text, wl=None):
    import r263_gain_mislabel_scan as g
    abs_set, hr_set = g._load_sets()
    abs_forms, hr_forms = g._dp_forms(abs_set), g._dp_forms(hr_set)
    hits = g.scan_text(g.reflow(text), abs_forms, hr_forms, prose=True)
    out = []
    for h in hits:
        loc = _locate(text, h["sentence"])
        # R-A 是"前缀模式"匹配 (防换精度规避)。若该 token 精确等于另一个在册量的值,
        # 则属**同值不同物**撞值, 不是两栏混引 —— 分诊为带双指针的语境豁免, 判据本身不放宽。
        collide = (h["rule"] == "R-A" and wl is not None and not h["exempt"]
                   and h["token"] not in abs_forms and h["token"] not in hr_forms
                   and wl.has(h["token"]))
        if collide:
            canon, dp = wl.identity(h["token"])
            src = next((e["src"] for e in wl.entries if abs(e["value"] - float(h["token"])) < 1e-9), "?")
            out.append({"rule": "R4-R-A", "line": loc, "quote": h["sentence"][:300],
                        "why": "R263 R-A 前缀模式撞上同值不同物: %s 是 %s 的在册值, 不是绝对误差集成员"
                               % (h["token"], src),
                        "pointer": AUTH["r263_scan"] + " (R-A) ∪ 撞值源 " + str(src),
                        "tokens": [h["token"]], "severity": "RED", "exempt": True,
                        "exempt_class": "X-R263-COLLISION", "exempt_mark": h["token"],
                        "exempt_reason": "同值不同物 (在册另一量的原值), 非两栏混引", "section": ""})
            continue
        out.append({"rule": "R4-" + h["rule"], "line": loc, "quote": h["sentence"][:300],
                    "why": "两栏混引 (配对改善 vs 绝对误差水平): 增益词与绝对读数同句 "
                           "[R263 判据 %s, 整条 import 复用其判据与豁免标记]" % h["rule"],
                    "pointer": AUTH["r263_scan"] + " ∪ results/r208_mixture_reboot/diag_em_baseline.json",
                    "tokens": [h["token"]], "severity": "RED", "exempt": bool(h["exempt"]),
                    "exempt_class": "X-R263-MARKER" if h["exempt"] else None,
                    "exempt_mark": h["token"] if h["exempt"] else None,
                    "exempt_reason": "R263 显式 absolute-level 措辞 (其 EXEMPTION_MARKERS)"
                    if h["exempt"] else None, "section": ""})
    return out


def _locate(text, sentence):
    """把 R263 命中 (其行号按 reflow 后文本计) 重新定位到原文件行 —— 报告要给作者能跳到的行。"""
    key = re.sub(r"\s+", " ", sentence).strip()[:60]
    if not key:
        return 0
    for i, line in enumerate(text.split("\n"), 1):
        if key[:40] in re.sub(r"\s+", " ", line):
            return i
    return 0


# --------------------------------------------------------------------------- R5 内部代号
# W81 rev4：裁定 94.2 的代号禁令**对象是论文/对外成稿**，不是治理件——治理面必须逐字写脚本名
# 才能安装与复核（在册 F-W81-5）。修法走**文档类别分轨**而非加豁免表：
#   doc_class 由调用式**显式声明**，默认 paper = fail-closed（未声明 = 全严格，绝不因新参数变宽）；
#   governance 档把红降为 NOTE 并保留在产物里（数与限定语同去同留；降级不是丢弃）。
DOC_CLASS_CHOICES = ("paper", "governance")


def r5_codes(text, doc_class="paper"):
    rxs, src_line = K.code_patterns()
    gov = doc_class == "governance"
    out = []
    for u in units_of(text):
        sc = u.sent_clean
        found = sorted({r.search(sc).group(0) for r in rxs if r.search(sc)})
        for fnd in found:
            out.append(_f("R5", u,
                          ("[doc_class=governance 降级为 NOTE，规则本体未改] " if gov else "")
                          + "内部代号「%s」入稿 (裁定 94.2: 代号一律不入稿)" % fnd,
                          AUTH["carrier"] + " 第 3 章「内部代号与治理计数」行", [fnd],
                          severity="NOTE" if gov else "RED"))
    return out


# --------------------------------------------------------------------------- R6 合成域限定语
SYNTH_QUAL_RX = re.compile(r"合成|孪生|基准|壁面级|DFN|数值实验|synthetic|仿真|虚拟")


def domain_markers(rd):
    """该读数**登记口径所属的语料/场地名** (从行内 key/src/sample 取, 零内联)。
    点名所属语算一种域限定 —— 它告诉读者这个数出自哪个信息制度 (任务书 R6 的意图)。"""
    blob = " ".join([str(rd.get("key") or ""), str(rd.get("src") or ""), str(rd.get("sample") or "")])
    return {x for x in re.findall(r"[A-Za-z][A-Za-z0-9]{4,}", blob) if x.lower() not in
            ("results", "json", "leaderboard", "summary", "point", "eval", "modal", "value")}


def r6_synthetic(text, readings):
    """R6 = 缺合成域限定语;  R6c = 缺在册随行口径 (行内 must_travel)。两件事**分开发号**。

    并成一条号会让"句中带『合成』"经 X-SYNTH-QUALIFIED 按规则号等值匹配把随行义务一并豁免掉
    (F-R317-2, 由 R319 修): 数据侧本就分成 synth_readings(need_synth=True) 与
    travel_obligations(need_synth=False, "不分合成域一律生效") 两个源, 病灶只在发出层。
    """
    out = []
    prepared = []
    for r in readings:
        fs = dp_forms([r["value"]], dps=(2, 3, 4)) if r["value"] >= 0.01             else dp_forms([r["value"]], dps=(3, 4, 5, 6))
        fs |= {x for x in r.get("exact") or [] if re.match(r"^\d+\.\d+$", x)}
        prepared.append((fs, r))
    for u in units_of(text):
        sc = u.sent_clean
        toks = set(NUMBER_RX.findall(sc))
        for fs, rd in prepared:
            hit = [t for t in sorted(toks & fs)
                   if UNIT_CTX_RX.match(sc[sc.find(t) + len(t):sc.find(t) + len(t) + 3])]
            if not hit:
                continue
            qualified = bool(SYNTH_QUAL_RX.search(sc)) or any(
                m.lower() in sc.lower() for m in domain_markers(rd))
            miss_synth = (["合成域限定语 (合成/孪生/基准/壁面级/DFN, 或点名所属语料场地)"]
                          if rd.get("need_synth") and not qualified else [])
            # 随行口径按**段**判 —— 取料表 R259-W1 自己的措辞即"必须同段注明"
            miss_travel = [mt for mt in rd["must"] if mt and mt not in u.para_clean]
            if miss_synth:
                out.append(_f("R6", u,
                              "合成/自证域读数 %s (=%s, 登记口径「%s」) 缺 %s" % (
                                  hit[0], rd["rendered"], rd["sample"], "、".join(miss_synth)),
                              rd["src"], hit))
            if miss_travel:
                out.append(_f("R6c", u,
                              "在册随行口径义务未满足: 读数 %s (=%s, 登记口径「%s」) 所在段缺 %s "
                              "(义务由该行自己的 must_travel 登记, 不分合成域一律生效; "
                              "句中带「合成/基准」等域限定语不构成豁免)" % (
                                  hit[0], rd["rendered"], rd["sample"],
                                  "、".join("随行口径「%s」" % m for m in miss_travel)),
                              rd["src"], hit))
    return out


# --------------------------------------------------------------------------- R7 字形与渲染
_FENCE_MARK = "`" * 3
_INLINE_GUARD_MAX = 20000


def code_hidden_regions(text):
    """返回 (整行被 ``` 围栏覆盖的行集, 行号→行内代码列区间表)。

    W81 rev4：CommonMark 里围栏内与行内代码中的 `#` 是字面量，不是标题标记。
    判 R7-heading 必须先剥这两类，否则对任何含代码块的件必产假红（F-W78-1；
    实测 W78 材料包 16/16、交接文档 6/6 全在围栏内）。
    行内代码必须按**列区间**判，不能「整行含反引号就跳过」——那会把同行混合形态
    （围栏/行内代码之后同一行或同段仍有真粘连标题）一并放过 = 覆盖面反向失守。
    """
    lines = text.split("\n")
    fenced, on = set(), False
    for i, raw in enumerate(lines, 1):
        if raw.strip().startswith(_FENCE_MARK):
            fenced.add(i)
            on = not on
            continue
        if on:
            fenced.add(i)
    spans = {}
    for i, raw in enumerate(lines, 1):
        found, pos, guard = [], 0, 0
        while pos < len(raw):
            guard += 1
            if guard > _INLINE_GUARD_MAX:
                raise SystemExit("R303 INLINE_SCAN_RUNAWAY line %d" % i)
            j = raw.find("`", pos)
            if j < 0:
                break
            k = raw.find("`", j + 1)
            if k < 0:
                break
            if k <= j:
                raise SystemExit("R303 INLINE_SCAN_NONMONOTONIC line %d" % i)
            found.append((j + 1, k + 1))
            pos = k + 1
        spans[i] = found
    return fenced, spans


def r7_glyph(text, kind="md"):
    import r280_glyph_gate as gg
    import r295_title_gate as tg
    out = []
    secs = line_sections(text)

    def mk(rid, line, quote, why, ptr, toks):
        # 与 r3cd_wording_bans 共用 _UnitLite: R7* 目前不在任何豁免类的 applies_to 里, 所以
        # 旧的裸桩 (无 sent_clean/para_clean) 恰好没被触发 —— 一旦有人给 R7 立豁免类就会同样崩栈。
        return _f(rid, _UnitLite(line, quote, quote, secs.get(line, "")), why, ptr, toks)

    if text.count("\x00"):
        for name, info in gg.shape_scan(text).items():
            out.append(mk("R7-shape", 0, "(全文)",
                          "%s: %d 处 —— %s [%s] (判据出处 scripts/r280_glyph_gate.SHAPE_RULES)"
                          % (name, info["count"], info["meaning"], info["severity"]),
                          AUTH["r280_glyph"], [str(info["count"])]))
        # 无独立 R7-nul 分支: r280 shape_scan 的 bare_nul 规则已覆盖一切未归类 NUL, 单列即不可达死代码

    nrep = text.count("\ufffd")
    if nrep:
        out.append(mk("R7-fffd", 0, "(全文)", "U+FFFD 替换符 %d 处 (编码丢失)" % nrep,
                      AUTH["r280_glyph"], [str(nrep)]))
    src_lines = text.split(chr(10))
    fenced, inline_spans = code_hidden_regions(text)
    for v in tg.scan_md(text).get("violations", []):
        raw = src_lines[v["line"] - 1] if 0 < v["line"] <= len(src_lines) else ""
        at = max(0, v.get("col", 1) - 1)
        if v["line"] in fenced:
            continue        # ``` 围栏内的 # 是字面量 (W81 rev4 / F-W78-1)
        if any(a <= v.get("col", 1) <= b for a, b in inline_spans.get(v["line"], [])):
            continue        # 行内代码跨度内的 # 同样是字面量 (rev3 补，按列判不按整行)
        if raw.strip().startswith("|") and re.match(r"#[^#]", raw[at:] or "#"):
            continue        # 表格单元里的字面单 # (列名) 非标题标记; "##" 粘连标题行仍判红
        kindtxt = "标题前有它物 (粘在表格行尾等)" if v.get("col", 1) != 1 else "前一行非空 ⇒ 渲染器不认它是标题"
        out.append(mk("R7-heading", v["line"], v["text"],
                      "未渲染标题: %s (判据出处 scripts/r295_title_gate.scan_md)" % kindtxt,
                      AUTH["r295_title"], ["#"]))
    if kind == "docx":
        nb = len(tg.BARE_RUN_RX.findall(text))
        if nb:
            out.append(mk("R7-bare-hash", 0, "(全文)",
                          "docx 正文里残留裸 `#` 连续 ≥2 共 %d 处 (未渲染痕迹)" % nb,
                          AUTH["r295_title"], [str(nb)]))
    return out


# --------------------------------------------------------------------------- R8 引用双向对齐
# W81 rev4：编号数字之后紧跟 [半/全角小数点]+数字 ⇒ 那是**量值邻接**不是图表编号。
# 原式 `([A-D]?\d{1,2})` 会在 "表 56.15" 上取出 token「表56」⇒ 报孤儿引用假红（交接文档 §10.3
# 记为「数值误析」）。`(?!\d)` 关掉 \d{1,2} 的回溯（否则 "56" 失败会退化成 "5" 再配 ".15"，
# 假红换个 token 复活），两枚前视合起来才封住该形态。真孤儿引用（"另见表57。"）照旧必抓。
REF_RX = re.compile(r"(图|表)\s*([A-D]?\d{1,2})(?!\d)(?![.．]\d)(?:\s*[（(][a-z\d]+[)）])?")
# 英文投稿形态: Fig. 3 / Figure 3 / Table 2 —— 与中文题注成对登记, 漏一侧即孤儿
# （同一缺陷在英文侧同形：Table 3.2 是量值邻接 ⇒ 与中文侧同规则加严，不放松）
REF_EN_RX = re.compile(r"(Fig\.?|Figure|Table|Eq\.?|Equation)\s*(\d{1,2})(?!\d)(?![.．]\d)")
# 题行 = 整行去掉可选 ** / #前缀后以 "图N/表N" 开头 (表内行除外)
CAP_LINE_RX = re.compile(r"^(?:\*\*\s*(图|表)\s*([A-D]?\d{1,2})"
                         r"|(图|表)\s*([A-D]?\d{1,2})\s*(?:｜|[:：]))")

def _ban_hit(phrase, text):
    """禁写词的变体形: 精确子串, 或"首尾字同、字符集几乎同"的缩略形。

    实测触发原因: 架构师夹具写的是"一次污染"而登记禁令是"一次性污染" ⇒ 精确匹配会漏掉
    同一个断言的缩略写法。判据取保守: 首字与末字必须相同, 且短语字符集被窗口覆盖 ≥ 4/5,
    窗口长度不超过短语 +2 字 ⇒ "一次污染"命中, 而语义无关的短句不命中 (双测各配一枚)。
    """
    if not phrase:
        return False
    if phrase in text:
        return True
    n = len(phrase)
    if n < 4:
        return False
    lo, hi = n - 1, n + 2   # 允许删一个修饰字 (一次性污染 ↔ 一次污染)
    for L in range(lo, hi + 1):
        for i in range(0, max(0, len(text) - L) + 1):
            w = text[i:i + L]
            if w[0] != phrase[0] or w[-1] != phrase[-1]:
                continue
            # 窗口必须是"短语删字"形态: 每个字符都属于短语, 且是短语的子序列 (防任意句子误命中)
            ps = set(phrase)
            if any(ch not in ps for ch in w):
                continue
            it = iter(phrase)
            if not all(ch in it for ch in w):
                continue
            if len(w) >= n - 1:
                return True
    return False


def _en_key(kind, num):
    k = kind.rstrip(".").lower()
    alias = {"fig": "Figure", "figure": "Figure", "table": "Table",
             "eq": "Equation", "equation": "Equation"}
    return alias.get(k, kind) + num


def r8_refs(text, manifest_ids):
    refs, defs, aliases = {}, {}, {}
    src_lines = text.split(chr(10))
    cap_lines = set()
    for i, raw in enumerate(src_lines, 1):
        cm = CAP_LINE_RX.match(raw.strip())
        if cm and not raw.strip().startswith("|"):
            grp = [x for x in cm.groups() if x]
            defs.setdefault(grp[0] + grp[1], i)
            cap_lines.add(i)
            for em in REF_EN_RX.finditer(raw):
                # 题行里的英文对照名只作**别名**: 正文引用中文编号即视为已引用,
                # 否则每个中英双语题注都会凭空造出"列而未引"假红
                aliases.setdefault(_en_key(em.group(1), em.group(2)), grp[0] + grp[1])
    for u in units_of(text):
        if u.line in cap_lines:
            continue
        sc = u.sent_clean
        for m in REF_RX.finditer(sc):
            refs.setdefault(m.group(1) + m.group(2), u.line)
        for em in REF_EN_RX.finditer(sc):           # 变更令 R303-2: Fig./Table 等英文引用形态
            refs.setdefault(_en_key(em.group(1), em.group(2)), u.line)
    out = []
    # 英文对照名与中文编号互为别名: 引用任一形态即算该本体已被引用 (双向归一, 不造双语假红)
    ref_line = {}
    for r, ln in refs.items():
        key = aliases.get(r, r)
        ref_line[key] = min(ref_line.get(key, ln), ln)
    for k in sorted(set(ref_line) - set(defs)):
        num = re.sub(r"^[ABCD]", "", k[1:])
        known = (k[0] == u"图" and any(num in x for x in manifest_ids))
        ptr = (AUTH["figure_manifest"] + u" (取料件图面在场, 但稿内无题行)" if known
               else u"稿件自身 (图/表题行未找到)")
        out.append({"rule": "R8-orphan-ref", "line": ref_line[k], "quote": k,
                    "why": u"正文引用「%s」但稿内找不到对应题行 ⇒ 孤儿引用" % k,
                    "pointer": ptr, "tokens": [k], "severity": "RED",
                    "exempt": False, "exempt_class": None, "exempt_mark": None,
                    "exempt_reason": None, "section": u""})
    for k in sorted(set(defs) - set(ref_line)):
        out.append({"rule": "R8-unref-object", "line": defs[k], "quote": k,
                    "why": u"「%s」在稿中却从未在正文被引用 (R258 F-1 同族: 列而未引)" % k,
                    "pointer": u"稿件自身", "tokens": [k], "severity": "RED",
                    "exempt": False, "exempt_class": None, "exempt_mark": None,
                    "exempt_reason": None, "section": u""})
    return out, {"refs": sorted(refs), "defs": sorted(defs),
                 "aliases": {a: aliases[a] for a in sorted(aliases)}}


# --------------------------------------------------------------------------- R9 精度统一
def r9_precision(text, wl):
    """同一在册量在稿中出现多位数: 含"在册登记位数"者记 NOTE, 否则 RED (R263 rendering_policy)。"""
    seen = {}
    for u in units_of(text):
        sc = u.sent_clean
        for m in NUMBER_RX.finditer(sc):
            tok = m.group(1)
            if _tok_out_of_scope(sc, m):
                continue
            canon, dp = wl.identity(tok)
            if canon is None:
                continue
            seen.setdefault(round(canon, 6), {}).setdefault(dp, (tok, u.line))
    out = []
    for val, forms in sorted(seen.items()):
        if len(forms) < 2:
            continue
        reg = wl.registered_decimals.get(val, set())
        dp_list = sorted(forms)
        sample = "、".join("%s(L%d)" % (forms[d][0], forms[d][1]) for d in dp_list)
        why = ("同一在册量 (%s 量级) 在稿中出现 %d 种位数 [%s]; 在册登记位数为 %s" % (
            val, len(dp_list), ", ".join(str(d) for d in dp_list), sorted(reg) or "未登记"))
        sev = "NOTE" if reg & set(dp_list) else "RED"
        out.append({"rule": "R9", "line": forms[dp_list[0]][1], "quote": sample,
                    "why": why + ("" if sev == "NOTE" else " ⇒ 无任何一档等于在册位数"),
                    "pointer": AUTH["r263_prereg"] + "#rendering_policy", "tokens": sorted(
                        {forms[d][0] for d in dp_list}), "severity": sev,
                    "exempt": False, "exempt_class": None, "exempt_mark": None,
                    "exempt_reason": None, "section": ""})
    return out


# --------------------------------------------------------------------------- 总控
def check(text, wl=None, kind="md", doc_class="paper"):
    if doc_class not in DOC_CLASS_CHOICES:
        raise SystemExit("R303: 未知 doc_class %r (可选 %s)" % (doc_class, "/".join(DOC_CLASS_CHOICES)))
    wl = wl or Whitelist(text)
    try:
        manifest_ids = sorted(jload(AUTH["figure_manifest"]).get("figures", {}).keys())
    except Exception:                                          # noqa: BLE001
        manifest_ids = []
    findings = []
    for fn, args in ((r1_closure, (text, wl)), (r2a_ci_mismatch, (text,)),
                     (r2b_cross_corpus, (text,)), (r2c_kgrid_claim, (text,)),
                     (r2d_protocol_mix, (text,)), (r2e_heading, (text,)),
                     (r3a_deprecated, (text,)), (r3b_oracle, (text,)),
                     (r4_two_column, (text, wl)), (r5_codes, (text, doc_class)),
                     (r6_synthetic, (text, K.synth_readings() + K.travel_obligations())),
                     (r7_glyph, (text, kind)),
                     (r2f_ci_endpoint_swap, (text,)), (r3cd_wording_bans, (text,)),
                     (r6b_synthetic_extrapolation, (text, K.synth_endpoints_all())),
                     (r308_contracts, (text,)),
                     (r7_extra, (text,))):
        findings += fn(*args)
    r8, r8info = r8_refs(text, manifest_ids)
    findings += r8
    findings += r9_precision(text, wl)
    red = [f for f in findings if f["severity"] == "RED" and not f["exempt"]]
    exempted = [f for f in findings if f["severity"] == "RED" and f["exempt"]]
    notes = [f for f in findings if f["severity"] == "NOTE"]
    by_rule = {}
    for f in findings:
        c = by_rule.setdefault(f["rule"], {"total": 0, "red": 0, "exempt": 0, "note": 0})
        c["total"] += 1
        c["red" if (f["severity"] == "RED" and not f["exempt"]) else
          ("exempt" if f["exempt"] else "note")] += 1
    return {"verdict": "CLEAN" if not red else "HIT", "n_findings": len(findings),
            "n_red": len(red), "n_exempt": len(exempted), "n_note": len(notes),
            "by_rule": by_rule, "whitelist_layers": wl.layers,
            "doc_class": doc_class, "kind": kind,
            "whitelist_missing_carriers": wl.missing_carriers,
            "whitelist_unreadable_sources": wl.unreadable_sources[:20],
            "r8_objects": r8info, "findings": findings, "generated_at": now_iso()}


def render_md(report, draft):
    L = ["# R303 成稿体检报告", "",
         "- 稿件: `%s` (%s, %d B, sha256 %s…)" % (draft["path"], draft["kind"], draft["bytes"],
                                                  draft["sha256"][:12]),
         "- 判级: **%s** ｜ 必改 RED %d ｜ 语境豁免 %d ｜ NOTE %d ｜ 总命中 %d" % (
             report["verdict"], report["n_red"], report["n_exempt"], report["n_note"],
             report["n_findings"]),
         "- 生成 %s ｜ 规则与豁免类冻结于 `results/r303_draft_check/prereg.json`" %
         report["generated_at"], ""]
    if report["n_red"]:
        L += ["## 必须改 (RED, 未获语境豁免)", "", "| 规则 | 行 | 原文 | 为什么错 | 正确写法指针 |",
              "|---|---|---|---|---|"]
        for f in report["findings"]:
            if f["severity"] == "RED" and not f["exempt"]:
                L.append("| %s | %s | %s | %s | `%s` |" % (
                    f["rule"], f["line"], f["quote"].replace("|", "\\|")[:90],
                    f["why"].replace("|", "\\|")[:200], (f["pointer"] or "")[:70]))
        L.append("")
    if report["n_exempt"]:
        L += ["## 语境豁免 (逐条给理由)", "", "| 规则 | 行 | 豁免类 | 语境标记 | 理由 |",
              "|---|---|---|---|---|"]
        for f in report["findings"]:
            if f["exempt"]:
                L.append("| %s | %s | %s | `%s` | %s |" % (
                    f["rule"], f["line"], f["exempt_class"], (f["exempt_mark"] or "")[:24],
                    (f["exempt_reason"] or "")[:90]))
        L.append("")
    if report["n_note"]:
        L += ["## NOTE (登记不判死)", ""]
        for f in report["findings"]:
            if f["severity"] == "NOTE":
                L.append("- L%s `%s` %s ｜ %s" % (f["line"], ",".join(f["tokens"]), f["rule"],
                                                  f["why"][:160]))
        L.append("")
    L += ["## 白名单生长层 (零内联数值)", ""]
    for name, info in report["whitelist_layers"].items():
        L.append("- `%s`: forms=%s values=%s files=%s ｜ src=%s" % (
            name, info.get("forms"), info.get("values"), info.get("files", "-"),
            info.get("src", "-")))
    if report["whitelist_missing_carriers"]:
        L += ["", "⚠ 授权散文载体缺件 (白名单缩面 ⇒ 可能制造假误杀): %s"
              % report["whitelist_missing_carriers"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    # 文档类别必须**显式声明**且进产物（可审计）；默认 paper = 全严格，禁按文件名猜类别
    ap.add_argument("--doc-class", dest="doc_class", choices=DOC_CLASS_CHOICES, default="paper",
                    help="paper=论文/对外成稿（默认，全严格）；governance=治理件，R5 代号规则降为 NOTE")
    a = ap.parse_args(argv)
    t0 = time.time()
    draft = K.load_draft(a.draft)
    rep = check(draft["text"], kind=draft["kind"], doc_class=a.doc_class)
    rep["draft"] = {k: draft[k] for k in ("path", "kind", "bytes", "sha256")}
    out = a.out or os.path.join(K.OUT_DIR, "run")
    jp = write_json(os.path.join(out, "draft_check.json"), rep)
    mp = "" if a.json_only else write_text(os.path.join(out, "draft_check.md"),
                                           render_md(rep, draft))
    if not a.quiet:
        print("verdict=%s red=%d exempt=%d note=%d total=%d -> %s%s (%.1fs)" % (
            rep["verdict"], rep["n_red"], rep["n_exempt"], rep["n_note"], rep["n_findings"],
            jp, (" " + mp) if mp else "", time.time() - t0))
    return rep



# ---------------------------------------------------- 变更令 R303-2 (R2f): CI 端点被替换成他组重抽
def r2f_ci_endpoint_swap(text):
    """同段出现墙锚系读数, 而 CI 端点不等于主口径成套值、却等于盘上他组重抽值 ⇒ 红 (裁定 82 P0)。

    判据是**成套身份**: 端点对来自源件字段, 点估计与区间必须同件同次重抽; 不用字符串巧合。
    """
    from r303_kit import INTERVAL_RX
    st = K.anchor_ci_set()
    prim, rivals = st["primary"], st["rivals"]
    out = []
    if prim is None or not rivals:
        return out
    pf = dp_forms([prim["lo"], prim["hi"]], dps=(2, 3, 4))
    meanf = dp_forms([prim["mean"]], dps=(1, 2, 3, 4))
    FAMILY_RX = re.compile(r"墙|锚|置信|CI|区间")
    blocks = K.table_blocks(text)
    for u in units_of(text):
        sc = u.sent_clean
        ctx = " ".join([sc, u.para_clean, blocks.get(u.line, "")])
        toks = set(NUMBER_RX.findall(sc))
        if not (toks & pf):
            continue                        # 端点不是主口径 ⇒ 与本判据无关
        if set(NUMBER_RX.findall(ctx)) & meanf:
            continue                        # 点估计在同段/同表格块在场且成套 ⇒ 合法
        if not FAMILY_RX.search(sc):
            continue
        out.append(_f("R2f", u,
                      "成套身份错位: 区间 %s 的主口径端点是 %s/%s (源 %s#%s), 但本句只给了它而缺同次"
                      "重抽的点估计 %s ⇒ 若替换成他组重抽端点即裁定 82 的 P0 红线" % (
                          sorted(toks & pf), prim["lo"], prim["hi"], prim["src"], prim["pointer"],
                          prim["mean"]),
                      prim["src"] + "#" + prim["pointer"], sorted(toks & pf)))
        del sc, toks
    # 他组端点单独出现且被当主口径 ⇒ 同样红 (这才是"改写成 27.12–34.08"的原形)
    for rv in rivals:
        rf = dp_forms([rv["lo"], rv["hi"]], dps=(2, 3, 4))
        for u in units_of(text):
            sc = u.sent_clean
            toks = set(NUMBER_RX.findall(sc))
            if not (toks & rf) or not FAMILY_RX.search(sc):
                continue
            if toks & pf:
                continue                    # 同**句**并列两值 = 对照陈述, 由 X-REDRAW-CONTEXT 认显式声明
            out.append(_f("R2f", u,
                          "端点被替换成他组重抽: 本句区间取 %s (源 %s#%s), 而现役主口径应为 %s/%s "
                          "(源 %s#%s); 同段未声明这是他组重抽 ⇒ 裁定 82 P0 红线" % (
                              sorted(toks & rf), rv["src"], rv["pointer"], rv["lo"], rv["hi"],
                              prim["src"], prim["pointer"]),
                          rv["src"] + "#" + rv["pointer"] + " ∪ " + prim["src"], sorted(toks & rf)))
    return out


# ---------------------------------------------------- 变更令 R303-2 (R3c/R3d): 措辞禁令与禁写形
def r3cd_wording_bans(text):
    """R3c = 字面禁写词; R3d = 读数 × 极限词共现且缺在册要求的限定语。

    禁令本体全部由 K.wording_bans() 从在盘件解析 (载体第 3 章禁写形行 ∪ R301 claim_forbidden),
    本函数不写任何词表; 源读不到时返回一条 R3c-INPUT-MISSING 红, 而不是静默放行。
    """
    wb = K.wording_bans()
    out = []
    for miss in wb["missing_sources"]:
        # 必须用 _UnitLite (带 sent_clean/para_clean): 原写法用裸 _U 桩, 而 "R3c"  registered 在
        # 豁免类的 applies_to 里 ⇒ exemption_for 取 u.sent_clean 直接 AttributeError, "响亮缺失"
        # 支实际是把整个体检器打崩 (R317 T2 普查实测, 见 results/r317_checker_wiring/verdict.json)
        out.append(_f("R3c-input-missing",
                      _UnitLite(0, u"(规则输入缺失)", u"(规则输入缺失)", u""),
                      "措辞禁令的权威源读不到: %s ⇒ 本判据不可判绿 (响亮缺失优于静默放行)" % miss,
                      miss, [miss]))
    for u in units_of(text):
        sc = u.sent_clean
        for b in wb["bans"]:
            if b["kind"] == "literal" and b["phrase"] and _ban_hit(b["phrase"], sc):
                out.append(_f("R3c-wording-ban", u,
                              "禁写措辞「%s」入稿 (由 %s 声明, 源 %s); 应改用其登记替代口径: %s" % (
                                  b["phrase"], b["source"], b["declared_from"],
                                  b["replacement"] or "(见源件)"),
                              b["source"], [b["phrase"]]))
            elif b["kind"] == "cooccur":
                nums = [n for n in b["numbers"] if n in sc.replace(u" ", u"")]
                lims = [w for w in b["limit_words"] if w in sc]
                if not (nums and lims):
                    continue
                if any(q in u.para_clean for q in b["qualifier"]):
                    continue
                out.append(_f("R3d-forbidden-claim", u,
                              "禁写形: 读数 %s 与极限词 %s 共现, 而同段缺限定语 %s ⇒ 该读数是"
                              "已达成量不是界 (源 %s); 其登记可写表述: %s" % (
                                  nums, lims, b["qualifier"], b["source"],
                                  b["replacement"] or "(见源件 T4_statement/claim_allowed)"),
                              b["source"], nums + lims))
    return out


# ---------------------------------------------------- 变更令 R303-2 (R6b): 合成端点外推禁令
EXTRAP_RX = re.compile(r"工程精度|可达精度|精度可达|可推广|推广到|外推到|可见.{0,12}可达|因此可|足以说明|投产|现场可用|真实场地可达|按工程可用")


def r6b_synthetic_extrapolation(text, endpoints):
    """合成/自证域端点与"工程可达"类推论词共现 ⇒ 红 (限定语在场也不能外推)。

    这条补的是 R6 的另一半: R6 管"没带限定语", R6b 管"带了限定语仍往工程域推"。
    端点集由 number_kit / number_table 的登记域字段运行时生成 (K.synth_endpoints_all)。
    """
    out = []
    forms = []
    for v in endpoints:
        dps = (3, 4, 5, 6) if v < 0.01 else ((2, 3, 4) if v >= 0.01 else (3, 4))
        fs = dp_forms([v], dps=dps)
        if 0.01 <= v < 1.0:
            fs |= dp_forms([v], dps=(1,))
        forms.append((fs, v))
    for u in units_of(text):
        sc = u.sent_clean
        toks = set(NUMBER_RX.findall(sc))
        hit = [(t, v) for fs, v in forms for t in sorted(toks & fs)
               if UNIT_CTX_RX.match(sc[sc.find(t) + len(t):sc.find(t) + len(t) + 3])]
        if not hit:
            continue
        ex = EXTRAP_RX.search(sc)
        if not ex:
            continue
        out.append(_f("R6b-synthetic-extrapolation", u,
                      "合成/自证域端点 %s (登记域 %s) 与外推词「%s」共现 ⇒ 合成基准上的读数不能当"
                      "工程可达精度 (任务书 R6 的反例形; 0.37° 系合成四壁空间分离场景)" % (
                          hit[0][0], hit[0][1], ex.group(0)),
                      K.AUTH["number_kit"] + " / " + K.AUTH["number_table"] + " 行内 truth_class/ev",
                      [hit[0][0], ex.group(0)]))
    return out


# ---------------------------------------------------- R317 T1 接入: R308 双契约 (整条 import)
def _r308_input_missing(rid, owner, err):
    return _f(rid + u"-input-missing",
              _UnitLite(0, u"(契约输入缺失)", u"(契约输入缺失)", u""),
              u"属主契约入口读不到它自己的预注册/源件 ⇒ 本判据不可判绿 (响亮缺失优于静默放行, "
              u"也优于把作者侧 CLI 打崩): %s: %s" % (type(err).__name__, str(err)[:160]),
              owner, [owner])


def r308_contracts(text):
    """R308-A「K=4→16 平坦」三标签同段闸 + R308-B oracle覆盖列↔指派缺口列身份互斥闸。

    与 R4 复用 R263 完全同一形态: 判据、触发词、三标签、两列值集 100% 留在属主件 ——
    `r308_check_flat_label` 自带 prereg 解析器 (读不到即 SystemExit 拒绝内置默认),
    `r308_identity_gate.value_sets()` 运行时直读 forensics / r204 三件。本函数零判据、
    零阈值、零值集, 只把属主命中搬成体检器条目 (原行号 + 原文 + 为什么错 + 指针)。
    """
    out = []
    try:
        import r308_check_flat_label as FA
        ptr = u"scripts/r308_check_flat_label.py ∪ " + FA.PREREG + u"#labels/triggers"
        for r in [x for x in FA.scan_text(text, FA._pre(), u"<draft>")
                  if x["severity"] == "RED"]:
            u = _UnitLite(r["unit_first_line"], r["quote"], r["quote"], u"")
            forms = sorted({a["form"] for a in r["evidence"]["assertions"]})
            out.append(_f("R308-A", u,
                          u"「K=4→16 平坦」类断言 (触发形 %s) 同段缺标签 %s ⇒ 必须同时在场 "
                          u"L1 非精度声明 / L2 列名与口径 / L3 与缺口曲线的主从关系 "
                          u"(属主判据缺任一即红, 无部分分; 转述与禁写语境由其自带豁免处理)" % (
                              u"、".join(forms), u"、".join(r["labels_missing"])),
                          ptr, r["labels_missing"]))
    except (SystemExit, Exception) as e:                                    # noqa: BLE001
        out.append(_r308_input_missing("R308-A", "scripts/r308_check_flat_label.py", e))
    try:
        import r308_identity_gate as IG
        gap, orc = IG.value_sets()
        ptr = (u"scripts/r308_identity_gate.py ∪ " + u" ∪ ".join(
            (IG.FORENSICS, IG.T1JSON, IG.V204)))
        for r in IG.scan_text(text, gap, orc, u"<draft>"):
            for m in r["mismatches"]:
                u = _UnitLite(r["line"], r["clause"], r["clause"], u"")
                out.append(_f("R308-B", u,
                              u"列身份错位: 标签「%s」(%s 族) 与数值 %s (%s 族) 直接配对 —— "
                              u"oracle 指派覆盖列与指派缺口列互斥, 两列值集由属主运行时直读 "
                              u"(K=4 两列只差 0.076°, 抄错即语义翻转)" % (
                                  m["label"], m["label_family"], m["token"], m["value_family"]),
                              ptr, [m["token"], m["label"]]))
    except (SystemExit, Exception) as e:                                    # noqa: BLE001
        out.append(_r308_input_missing("R308-B", "scripts/r308_identity_gate.py", e))
    return out


# ---------------------------------------------------- 变更令 R303-2 (R7): 无空格标题标记 + 代码围栏
def r7_extra(text):
    """`###4.1` 这类"井号后无空格"在 CommonMark 里不渲染成标题 —— 与 R295 的粘连标题同族。

    只加判定不放宽: 需 (a) 行首或表格单元黏连, (b) 连续 >=2 个 #, (c) 紧跟非空白非标点,
    (d) 不在 ``` 围栏内。合法夹具里的 "| # |" 列名 (单 #) 仍不误报。
    """
    out = []
    fence = False
    lines = text.split(chr(10))
    secs = line_sections(text)
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        for m in re.finditer(r"(?<!#)#{2,}(?![#\s])", raw):
            tail = raw[m.end():m.end() + 1]
            if not tail or tail in u"）)】.":
                continue
            if raw.lstrip().startswith(u"|") and m.start() < raw.index(u"|") + 1:
                pass
            q = _UnitLite(i, raw, raw, secs.get(i, u""))
            out.append(_f("R7-heading-nospace", q,
                          "标题井号后缺空格 (%s) ⇒ 渲染器不视作标题 (判据与 r295 scan_md 同族扩展, "
                          "只加判定不放宽)" % raw[:40], K.AUTH["r295_title"], [m.group(0)]))
            break
    return out


class _UnitLite(object):
    __slots__ = ("line", "sent", "para", "section")

    def __init__(self, line, sent, para, section):
        self.line, self.sent, self.para, self.section = line, sent, para, section

    @property
    def sent_clean(self):
        from r303_kit import CODE_RX, SUP_RX
        return CODE_RX.sub(" ", SUP_RX.sub(" ", self.sent))

    @property
    def para_clean(self):
        from r303_kit import CODE_RX, SUP_RX
        return CODE_RX.sub(" ", SUP_RX.sub(" ", self.para))


if __name__ == "__main__":
    main()
