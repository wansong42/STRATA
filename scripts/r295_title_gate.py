# -*- coding: utf-8 -*-
"""R295 · 标题渲染双侧常驻闸（L5 铁律：R293 的 P0 换一个永久规则）。

病灶（R293 九道闸测不出自己看不见的那一条）：本项目 PDF 由 reportlab 自渲染
（`r277_render.build_flowables_v11` 把正文按 `\\n\\s*\\n` 切块，只有**以 `#` 起手的整块**
才认成标题），不是 CommonMark。于是「行首有 `#` 且后跟空格」**不充分**：

  ① 前一行必须是空行 —— 否则标题与上文同块，整块走正文分支，
     而正文分支的 `re.sub(r"^#\\s+", "", …, re.M)` 只剥一个 `#`，`## ` 原样印到纸上；
  ② 该行除标题外不得有其它内容 —— 否则整块以 `|` 起手走表格分支，标题被吞进末格。

⇒ 双侧都必须把守，缺一侧不算做完：
  * **md 侧**（`scan_md`）：逐行定位 `#{1,6} `，报出「同行有它物」与「前一行非空」两类违规 + 行号，
    答的是「哪一行错」；
  * **PDF 侧**（`scan_pdf_text`）：数文本层残留的 `#`，答的是「修没修好」。
    只查 md ⇒ 渲染器一改就静默退化（裁定 79 的 .notdef 吞负号已为此付过一次 P0）。

口径注（坑 69 的反向形态：架构师按自造正则扫出 116 处假阳，真数 2）：
  本模块要求 `#` 后**紧跟空格**，故 `x.json#rows/7/value` 这类路径、`NNS-`、`#`-无空格
  一律不计。负例毒丸必须证明这条正则确有牙（见 tests/test_v_r295_.py::test_05）。
"""
from __future__ import annotations

import re

# 标题标记：行内任意位置的「1–6 个 # 紧跟一个空格」，前面不得还有 #（避开 ##### 重复计数）
HEAD_MARK_RX = re.compile(r"(?<!#)#{1,6} ")
# PDF 文本层残留：任务书口径 = 连续 ≥2 个 #；本单主闸更严 = 全文零个 #（见 prereg C3）
BARE_RUN_RX = re.compile(r"#{2,}")
LONE_HASH_RX = re.compile(r"(?<!#)#(?!#)")
RESULTS_PATH_RX = re.compile(r"results[\\/]")
SUBKEY_RX = re.compile(r"results/[^\s|]*#[^\s|]*")
PATHLIKE_JSON_RX = re.compile(r"[\w./\-]*results/[^\s|]*\.json")
ANCHOR_INTACT_RX = re.compile(r"NNS-\d{3}")
# 断行形：`NNS-1` 换行 `87`（CJK 逐字换行把七枚字符压断）
ANCHOR_BROKEN_RX = re.compile(r"NNS-\d{1,2}\n\d{1,3}(?!\d)")


def scan_md(md_text: str, require_marks: int | None = None) -> dict:
    """md 侧：返回每一处「未被渲染器认作标题」的标题标记及成因。

    `require_marks`（本军规要求：判据不得空跑判绿）：给定时期望的标题标记总数。
    空串或退化输入 ⇒ head_marks_total 掉到该值以下 ⇒ 直接判红，而不是「零违规 = 绿」。
    """
    lines = md_text.split("\n")
    same_line_pollution = []   # ② 标题前有它物（粘在表格行尾等）
    no_blank_before = []       # ① 前一行非空 ⇒ 与上文同块
    headings_total = 0
    for i, ln in enumerate(lines):
        for m in HEAD_MARK_RX.finditer(ln):
            headings_total += 1
            if m.start() != 0:
                same_line_pollution.append({
                    "line": i + 1, "col": m.start() + 1,
                    "text": ln[:120], "prev_tail": lines[i - 1][-40:] if i else ""})
            elif i > 0 and lines[i - 1].strip() != "":
                no_blank_before.append({
                    "line": i + 1, "col": 1, "text": ln[:120],
                    "prev_line": i, "prev_tail": lines[i - 1][-40:]})
    violations = same_line_pollution + no_blank_before
    degenerate = (require_marks is not None and headings_total < require_marks)
    return {
        "head_marks_total": headings_total,
        "require_marks": require_marks,
        "degenerate_input": bool(degenerate),
        "n_same_line_pollution": len(same_line_pollution),
        "n_no_blank_before": len(no_blank_before),
        "n_violations": len(violations),
        "violations": sorted(violations, key=lambda d: (d["line"], d["col"])),
        "same_line_pollution": same_line_pollution,
        "no_blank_before": no_blank_before,
        "pass": (not violations) and not degenerate,
        "rule": "每一处 `#{1,6} ` 必须 (a) 位于行首 (b) 其前一行是空行（或本行为文件首行）；"
                "且标题标记总数不得低于 require_marks（防空跑判绿）",
    }


# 未渲染标记全类（架构师 19:5x 第 3 条：同一形状不止 `##`）
MARKER_CLASSES = {
    "bold_double_star": re.compile(r"\*\*"),
    "any_asterisk": re.compile(r"\*"),
    "code_backtick": re.compile(r"`"),
    "unrendered_quote_start": re.compile(r"(?m)^>[ \t]"),
    "unrendered_bullet_start": re.compile(r"(?m)^[ \t]*[-*][ \t]"),
    "bare_hr_line": re.compile(r"(?m)^-{3,}[ \t]*$"),
}
# 散落/整格竖排的锚形态：允许 N N S - 之间夹任意空白（裁定 90 的构造性漏检这一形）
ANCHOR_SCATTERED_RX = re.compile(r"N\s*N\s*S\s*-\s*(?:\d\s*){3}")


def classify_hash_occurrences(pdf_text: str) -> dict:
    """把文本层每一枚 `#` 按形状归类。

    为什么单列：架构师 19:5x 提醒的「PDF 计数不能推引入者」，其机制就在这里——
    v1.3/v1.4 的 PDF 里 `#` 有 128 枚，但**几乎全是路径子键** `x.json#rows/7/value`
    （R290 移走路径墙前的形态），只有 1 枚是未渲染标题。把两类混在一个计数里，
    就会得出「v1.4 比 v1.5 更坏」这种假结论。故本闸按类分列，主闸只认 heading_like。
    """
    rows, counts = [], {"heading_like": 0, "path_subkey_like": 0, "other": 0}
    for m in re.finditer(r"#+", pdf_text):
        s, e = m.start(), m.end()
        before = pdf_text[max(0, s - 30):s]
        after = pdf_text[e:e + 30]
        run = m.group(0)
        glued_to_word = bool(re.search(r"[\w.／/]\Z", before))          # 前面紧贴字母/数字/点 ⇒ 像路径子键
        followed_by_space = bool(re.match(r"[ \t\n　]", after))
        if run == "#" and glued_to_word and not followed_by_space:
            form = "path_subkey_like"
        elif len(run) >= 2 or (glued_to_word is False and followed_by_space):
            form = "heading_like"
        else:
            form = "other"
        counts[form] += 1
        rows.append({"run": run, "form": form, "before_tail": before[-24:], "after_head": after[:24]})
    return {"counts_by_form": counts, "n_occurrences": len(rows), "occurrences": rows}


def scan_pdf_text(pdf_text: str, require_chars: int = 200, baseline_markers: dict | None = None) -> dict:
    """PDF 侧：文本层残留的 # 计数（从制品读，不从渲染器读）。

    `require_chars`：文本层少于该字符数即视为**缺件/空输出** ⇒ 判红（同上新军规）。
    `baseline_markers`：给定基线（现役前一身）的分类型标记计数时，附加一条
    「本件不得**新引入**任何一类未渲染标记」。存量形态不判红而单列 `pre_existing`
    （零扩展原则：登记移交，不把本单撑大）。
    """
    runs = list(BARE_RUN_RX.finditer(pdf_text))
    lone = list(LONE_HASH_RX.finditer(pdf_text))
    ctx = [{"run": m.group(0),
            "before": pdf_text[max(0, m.start() - 45):m.start()],
            "after": pdf_text[m.end():m.end() + 30]} for m in runs]
    degenerate = len(pdf_text.strip()) < require_chars
    cls = classify_hash_occurrences(pdf_text)
    markers = {k: len(rx.findall(pdf_text)) for k, rx in MARKER_CLASSES.items()}
    introduced, pre_existing = {}, {}
    if baseline_markers is not None:
        for k, v in markers.items():
            b = baseline_markers.get(k, 0)
            if v > b:
                introduced[k] = {"baseline": b, "now": v, "delta": v - b}
            elif v > 0:
                pre_existing[k] = {"baseline": b, "now": v}
    return {
        "hash_chars_total": pdf_text.count("#"),
        "n_bare_runs_ge2": len(runs),
        "n_lone_hash": len(lone),
        "text_chars": len(pdf_text),
        "require_chars": require_chars,
        "degenerate_input": bool(degenerate),
        "contexts": ctx,
        "hash_form_classification": cls,
        "marker_class_counts": markers,
        "marker_class_introduced": introduced,
        "marker_class_pre_existing": pre_existing,
        "marker_class_baseline_given": baseline_markers is not None,
        "pass_taskbook_caliber": (len(runs) == 0) and not degenerate,
        "pass_strict_caliber": (pdf_text.count("#") == 0) and not degenerate,
        "pass_no_new_markers": (baseline_markers is None) or (not introduced and not degenerate),
        "rule": "PDF 文本层不得出现任何 `#`（主闸，prereg R-3 加严）；任务书原口径 `#{2,}`==0 被其蕴含并同报；"
                "未渲染标记（** / 反引号 / 行首列表引用 / 裸 ---）不得较基线新增；文本层退化输入直接判红",
    }


# ---------------------------------------------------------------- 制品级读数
def pdf_artifact_readings(path: str, baseline_markers: dict | None = None) -> dict:
    """R290 判定项 + 本单双侧新闸，一次采齐（全部从 PDF 制品直读）。"""
    import glyphcover as G
    import r295_common as K
    t = K.pdf_text(path)
    g = G.scan_pdf(path)
    scattered = len(ANCHOR_SCATTERED_RX.findall(t))
    intact = len(ANCHOR_INTACT_RX.findall(t))
    return {
        "pages": g["pages"], "nul": g["nul"], "fffd": g["fffd"],
        "image_xobjects": g["image_xobjects"], "bytes": g["bytes"],
        "sha256": K.sha256(path),
        "results_hits": len(RESULTS_PATH_RX.findall(t)),
        "pathlike_json_hits": len(PATHLIKE_JSON_RX.findall(t)),
        "hash_subkey_hits": len(SUBKEY_RX.findall(t)),
        "anchor_intact": intact,
        "anchor_broken": len(ANCHOR_BROKEN_RX.findall(t)),
        "anchor_scattered_forms": scattered,
        "anchor_scattered_minus_intact": scattered - intact,
        "anchor_two_calibers_agree": scattered == intact,
        "decimal_multiset_sha_raw": K.multiset_sha(K.decimal_multiset_of_text(t)),
        "decimal_multiset_sha_stripped": K.multiset_sha(
            K.decimal_multiset_of_text(K.structural_stripped(t))),
        "pdf_side_gate": scan_pdf_text(t, baseline_markers=baseline_markers),
        "scope": "本件 PDF 全部页面（pypdf 逐页 extract_text + glyphcover.scan_pdf）",
        "enumerate_method": "进程内全量 re.findall（坑 73：不走被截断的 stdout）",
        "measured_at": K.now_iso(),
    }
