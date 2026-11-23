"""R272 shared measurements (《研究工作报告》= 结题报告 v2.0 正文底座 + 中文学报外壳).

Everything prereg / build / machine-check / dual-test-matrix / verdict assert is computed
here so the four cannot disagree. No threshold is inlined twice; no research reading is
hand-typed (all decimals must resolve against the R259 number table, which itself reads
the on-disk result JSONs).
"""
import hashlib
import io
import json
import os
import re
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(ROOT, "docs", "成果总报告")
FIG_DIR = os.path.join(OUT_DIR, "figures")
EVID = os.path.join(ROOT, "results", "r272_report_merge")
SRC_DIR = os.path.join(EVID, "manuscript_src")

# ---- read-only sources -------------------------------------------------
BASE_MD = os.path.join(ROOT, "docs", "结题报告", "结题报告_v2.0.md")
APPENDIX_A = os.path.join(ROOT, "docs", "结题报告", "附录A_现役数字总表.md")
NUMBER_TABLE = os.path.join(ROOT, "results", "r259_thesis_draft", "number_table.json")
PROCESS_FACTS = os.path.join(ROOT, "results", "r259_thesis_draft", "process_facts.json")
JCIP_BASELINE = os.path.join(ROOT, "results", "r270_cn_rewrite", "jcip_format_baseline.md")
REF_REGISTRY_CN = os.path.join(ROOT, "results", "r270_cn_rewrite", "reference_registry_cn.json")
R270_PREREG = os.path.join(ROOT, "results", "r270_cn_rewrite", "prereg.json")
OUTER_TONGSHEN = os.path.join(ROOT, "docs", "对外口径冻结页_R213.md")

FIGS_V30 = os.path.join(ROOT, "docs", "paper1", "figures_v30")
FIGS_V10 = os.path.join(ROOT, "docs", "paper2", "figures_v10")

REF_PDFS = {
    "JCIP-A": os.path.join("E:", os.sep, "edge", "煤矿安全知识图谱构建及智能查询方法研究_刘鹏.pdf"),
    "JCIP-B": os.path.join("E:", os.sep, "edge", "基于新型卷积神经网络构建矿山灾害事件检测模型_刘鹏.pdf"),
    "NKU": os.path.join("E:", os.sep, "edge", "基于Spark的LIBSVM参数优选并行化算法_李坤.pdf"),
}

# ---- deliverables ------------------------------------------------------
MANUSCRIPT = os.path.join(OUT_DIR, "研究工作报告_v1.0.md")
ONEPAGER_MD = os.path.join(OUT_DIR, "导师十分钟版_v1.0.md")
PDF_MAIN = os.path.join(OUT_DIR, "研究工作报告_v1.0.pdf")
PDF_ONEPAGER = os.path.join(OUT_DIR, "导师十分钟版_v1.0.pdf")

CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
WATERMARK = "研究线与治理数字, 禁入对外承诺文本"

# 判废数字家族 (直读 results/r263_gain_erratum/prereg.json 的 DEP 域 + R270 prereg M8 清单)
FORBIDDEN_NUMBERS_FALLBACK = ["14.22", "11.01", "9.35", "9.83", "7.23", "6.60", "15.82",
                              "12.39", "12.5152", "16.08", "14.96", "7.50", "4.88",
                              "3.81", "0.78", "12.512"]

# ---------------------------------------------------------------- basics


def read(path):
    return io.open(path, encoding="utf-8").read()


def read_bytes(path):
    return io.open(path, "rb").read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def sha256(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True))
    os.replace(tmp, path)
    return path


def jread(path):
    return json.loads(read(path))


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def tree_sha(base):
    acc = []
    for root, _d, fs in os.walk(base):
        for f in fs:
            fp = os.path.join(root, f)
            acc.append((os.path.relpath(fp, base).replace("\\", "/"), sha256(fp)))
    acc.sort()
    return hashlib.sha256("\n".join("%s %s" % t for t in acc).encode("utf-8")).hexdigest()


def env_guard():
    """坑 43: refuse to run under the base Anaconda interpreter."""
    import numpy
    import scipy
    exe = os.path.abspath(sys_executable())
    if "pytorch_env" not in exe.lower() or numpy.__version__ != "2.2.5" or scipy.__version__ != "1.15.3":
        raise SystemExit("FATAL 坑43: 环境毒丸 —— 当前 %s numpy=%s scipy=%s，须用 pytorch_env 且 numpy 2.2.5/scipy 1.15.3"
                         % (exe, numpy.__version__, scipy.__version__))
    return {"executable": exe, "numpy": numpy.__version__, "scipy": scipy.__version__}


def sys_executable():
    import sys
    return sys.executable


# ------------------------------------------- number conservation (T6 core)

DEC_RE = re.compile(r"-?\d+\.\d+")
INT_RE = re.compile(r"(?<![\d.])\d+(?![\d.])")

# patterns stripped before the multiset comparison (structural / bibliographic numbers)
STRIP_PATTERNS = [
    ("section_number", re.compile(r"^\s*[#>\-]*\s*[１-９1-9]{1,2}(?:．|\.)[１-９0-91-9]*(?:．[０-９0-9]+)?\s*[^\n]*$")),
    ("version", re.compile(r"[vV]\d+\.\d+(?:\.\d+)?")),
    ("fig_tab_number", re.compile(r"[图表式]\s*\d+(?:[ab]\b)?(?:[-–]\d+)?")),
    ("doi", re.compile(r"\b\d{2,4}\.\d{4,}(?:/\S+)?")),
    ("volume_issue_page", re.compile(r"\b\d{4}\s*[;,，]\s*\d{1,3}\s*\(?\d{0,3}\)?\s*:?\s*\d{0,4}\s*[-–]{1,2}\s*\d{0,4}")),
    ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("year_bare", re.compile(r"\b(19|20)\d{2}\b(?=[^\d]{0,3}(?:[，,、）)\]]|年|$))")),
    ("ledger_id", re.compile(r"\b(?:NNS|DEP|R\d{3}|M\d{2})[-\w]*")),
    ("project_code", re.compile(r"\b\d{4}[A-Z]{2}\d{6,}-\d{2}\b")),
    ("bracket_cite", re.compile(r"\[\d{1,3}(?:[,，-]\s*\d{1,3})*\]")),
    ("latin_id", re.compile(r"\b[A-Za-z]+[-_]?\d+[A-Za-z0-9_\-]*")),
]


def strip_structural(text):
    """Return text with structural/bibliographic number spans blanked out."""
    text = text.replace("\u2212", "-").translate(_FULLWIDTH_DIGITS)
    out = []
    for line in text.split("\n"):
        keep = True
        for name, rx in STRIP_PATTERNS:
            if name == "section_number" and rx.match(line.strip()):
                keep = False
                break
        if not keep:
            out.append("")
            continue
        t = line
        for name, rx in STRIP_PATTERNS:
            if name != "section_number":
                t = rx.sub(" ", t)
        out.append(t)
    return "\n".join(out)


def decimal_tokens(text):
    return sorted(DEC_RE.findall(strip_structural(text)))


def integer_tokens(text):
    return sorted(INT_RE.findall(strip_structural(text)))


def token_multiset_diff(before, after):
    from collections import Counter
    ca, cb = Counter(after), Counter(before)
    added = sorted(t for t in ca if ca[t] > cb.get(t, 0))
    removed = sorted(t for t in cb if cb[t] > ca.get(t, 0))
    return {"added": added, "removed": removed,
            "n_added_types": len(added), "n_removed_types": len(removed)}


# ------------------------------------------------------------- allowed set

def load_allowed_numbers():
    """allowed decimals = R259 number table (printed forms + resolved value at printed
    precision) ∪ 附录A md ∪ 结题报告 v2.0 body.  Every entry carries provenance."""
    rows = jread(NUMBER_TABLE)["rows"]
    allowed = {}
    for r in rows:
        key = r["key"]
        forms = set(r.get("printed_forms") or [])
        if r.get("printed"):
            forms.add(str(r["printed"]))
        rv = r.get("resolved_value")
        if isinstance(rv, (int, float)):
            dec = int(r.get("decimals") or 0)
            forms.add(("%.*f" % (dec, rv)).rstrip() if dec else str(int(rv)))
        for f in forms:
            allowed.setdefault(f, []).append(key)
    for src in (APPENDIX_A, BASE_MD):
        for t in decimal_tokens(read(src)):
            allowed.setdefault(t, []).append("SOURCE_DOC:%s" % rel(src))
    ints = {}
    for src in (APPENDIX_A, BASE_MD):
        for t in integer_tokens(read(src)):
            ints.setdefault(t, []).append("SOURCE_DOC:%s" % rel(src))
    for r in rows:
        key = r["key"]
        for f in (r.get("printed_forms") or []):
            if "." not in f:
                ints.setdefault(f, []).append(key)
        if isinstance(r.get("resolved_value"), int):
            ints.setdefault(str(r["resolved_value"]), []).append(key)
    return {"decimals": allowed, "integers": ints, "n_table_rows": len(rows)}


# --------------------------------------------------------------- utilities

def cn_abstract_char_count(text):
    """摘要体例按字符计（去空白），180–250 字。"""
    body = re.sub(r"^\s*摘\s*要[:：]\s*", "", text.strip())
    body = re.sub(r"\s+", "", body)
    return len(body)


def fullwidthize(num):
    table = str.maketrans("0123456789", "０１２３４５６７８９")
    return str(num).translate(table)


def split_sections(md_text):
    """Return {title: body} for '## ' level sections plus a 'front' entry."""
    parts = {}
    cur, buf = "front", []
    for line in md_text.split("\n"):
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            parts[cur] = "\n".join(buf)
            cur, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    parts[cur] = "\n".join(buf)
    return parts


def body_text(md_text):
    """正文（参考文献之前的主体，不含附录与参考文献）。"""
    lines = md_text.split("\n")
    out, stop = [], False
    for l in lines:
        if re.match(r"^##\s*(参考文献|附录)", l.strip()):
            stop = True
        if not stop:
            out.append(l)
    return "\n".join(out)


def references_block(md_text):
    m = re.search(r"^##\s*参考文献\s*$", md_text, re.M)
    if not m:
        return ""
    nxt = re.search(r"^##\s", md_text[m.end():], re.M)
    return md_text[m.end(): m.end() + (nxt.start() if nxt else len(md_text))]


def citation_marks(md_text):
    """正文（不含文献表/附录）里的上标引用位顺序表；接受 [n]、[n,m] 与 [n-m] 三种写法。"""
    body = body_text(md_text)
    out = []
    for grp in re.findall(r"(?<![A-Za-z0-9])\[(\d{1,3}(?:\s*[,，\-–]\s*\d{1,3})*)\](?!\()", body):
        for part in grp.split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d{1,3})\s*[-–]\s*(\d{1,3})", part)
            if m:
                out += list(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part:
                out.append(int(part))
    return out


def scannable_text(md_text):
    """参与数字守恒扫描的面：整稿去掉参考文献块（书目数字属注册表，不属读数）。"""
    return md_text.replace(references_block(md_text), " ")


def ref_display(gb_line):
    """Render a registry line for display: whitespace collapsed, emphasis tags dropped.
    A pure function of the registry bytes — no field value is editable from here."""
    import html as _html
    s = _html.unescape(gb_line)
    s = re.sub(r"</?(?:i|em|b|strong)>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def caption_numbers(md_text, prefix):
    caps = []
    for m in re.finditer(r"^\*\*%s\s*(\d{1,3})\s" % prefix, md_text, re.M):
        caps.append(int(m.group(1)))
    return caps


def equation_numbers(md_text):
    """Numbered display equations in the artifact: a math-operator line ending with (n)."""
    found = set()
    for ln in md_text.splitlines():
        m = re.search(r"[=≥≤≠].{0,160}?\((\d+)\)\s*$", ln)
        if m:
            found.add(m.group(1))
    return sorted(found, key=int)


def asset_counts(md_text):
    """Counts taken from the artifact itself, never from a receipt (R258 lesson)."""
    eqs = equation_numbers(md_text)
    return {"figures": len(caption_numbers(md_text, "图")),
            "tables": len(caption_numbers(md_text, "表")),
            "equations": len(eqs),
            "algorithms": len(re.findall(r"^\*\*算法\s*(\d+)", md_text, re.M))}


# --------------------------------------------------- governance-trace scan

TRACE_PATTERNS = {
    "mod_comment": re.compile(r"<!--\s*MOD\b"),
    "html_comment": re.compile(r"<!--"),
    "pointer_symbol": re.compile(r"[⟨⟦][A-Za-z0-9_\-]+[⟩⟧]|\[R259-[A-Z]|\[R2\d{2}-"),
    "watermark_line": re.compile(r"^\s*>?\s*\*{0,2}(本稿|水印|研究线数字|禁入对外|本单不代|机检\s?C\d)"),
    "internal_path": re.compile(r"(?:docs|results|scripts|src|tests)/[\w\-./\u4e00-\u9fff]+\.(?:md|py|json|pt|pdf|png)"),
    "audit_armour": re.compile(r"(不代答|不代填|本稿不代|已判废待另立|机检\s?C\d\d|唯一豁免位|预登记红线)"),
}


def text_hygiene(text):
    hits = {}
    for k, rx in TRACE_PATTERNS.items():
        if k == "internal_path":
            hits[k] = sorted(set(rx.findall(text)))[:40]
        else:
            hits[k] = [m.group(0)[:60] for m in rx.finditer(text)][:40]
    return {k: v for k, v in hits.items() if v}


# ------------------------------------------------- gain/absolute redline
# 唯一实现 = scripts/r263_gain_mislabel_scan.py（只读复用，本单不复制判据）

def gain_scanner():
    import importlib
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    return importlib.import_module("r263_gain_mislabel_scan")


HERE = os.path.dirname(os.path.abspath(__file__))


def gain_scan(text):
    scan = gain_scanner()
    abs_forms, hr_forms = scan._load_sets()
    items = scan.scan_text(text, abs_forms, hr_forms, prose=True)
    return [i for i in items if not i.get("exempt")]


# -------------------------------------------------------- forbidden numbers

def forbidden_number_hits(text, forbidden):
    """Boundary-matched scan: a retired literal only counts as a standalone reading."""
    out = []
    for tok in forbidden:
        rx = re.compile(r"(?<![\d.])" + re.escape(tok) + r"(?![\d.])")
        for m in rx.finditer(text):
            out.append({"token": tok, "context": text[max(0, m.start() - 40):m.end() + 40]})
    return out


def retired_tokens():
    """Retired literals, read from the R263 prereg (authoritative DEP domain) when present,
    union the R270 M8 list (frozen read-only input of this unit)."""
    toks = set(FORBIDDEN_NUMBERS_FALLBACK)
    r270 = jread(R270_PREREG)
    try:
        toks |= set(r270["criteria"]["T7_artifact_checks"]["M8_forbidden_numbers"]["params"]["forbidden"])
    except KeyError:
        raise SystemExit("FATAL: R270 prereg M8 list not resolvable")
    return sorted(toks, key=lambda s: (-len(s), s))


# --------------------------------------------------- reference registry

def load_ref_registry():
    d = jread(REF_REGISTRY_CN)
    toks = {}
    for e in d["entries"]:
        toks[e["token"].upper()] = e
    return d, toks


def render_reference_table(order_tokens, registry):
    """Render the GB/T 7714 table by first-citation order. Fields come from the registry
    verbatim — zero hand typing; an unregistered token is a hard error, never a skip."""
    by_tok = registry[1] if isinstance(registry, tuple) else {
        e["token"].upper(): e for e in registry["entries"]}
    lines = []
    for i, tok in enumerate(order_tokens, 1):
        e = by_tok.get(tok.upper())
        if e is None:
            raise SystemExit("FATAL: citation token not in registry: %s" % tok)
        lines.append({"n": i, "token": e["token"], "gb_line": e["gb_line"],
                      "year": e["fields"]["year"], "kind": e["fields"]["kind"]})
    return lines


# ------------------------------------------------------ format gate (16件)

def twelve_check(md_text):
    """件序在场性: 16 件按基线件 §1 行序逐项定位, 返回 present/absent/order."""
    front = md_text.split("\n## ")[0]
    pieces = {
        "1_刊眉卷期占位": (r"卷期[:：]\s*待编辑部分配", front),
        "2_文章编号占位": (r"文章编号[:：]\s*待编辑部分配", front),
        "3_中文标题": (r"^#\s+\S", md_text),
        "4_作者与单位": (r"中国矿业大学（北京）", front),
        "5_中文摘要": (r"摘\s*要[:：]", front),
        "6_中文关键词": (r"关键词[:：][^。]*；", front),
        "7_中图分类号与文献标识码": (r"中图分类号[:：]待核.*文献标识码[:：]Ａ", front),
        "8_英文对照题录": (r"\*\*Abstract:\*\*[\s\S]{0,3000}?\*\*Key words:\*\*", front),
        "9_正文分节全角编号": (r"^##\s*[１-９]{1,2}(?:．[０-９]{1,2})?\s", md_text),
        "10_末章总结与展望": (r"^##\s*[１-９]+\s*总结与展望", md_text),
        "11_首页脚注三件": (r"基金项目[:：][\s\S]{0,300}?收稿日期[:：]待编辑部分配[\s\S]{0,300}?Ｅ－ｍａｉｌ", front),
        "12_参考文献顺序编码": (r"^##\s*参考文献", md_text),
        "13_图题注": (r"^\*\*图\s*\d+\s", md_text),
        "14_表题注三线表": (r"^\*\*表\s*\d+\s", md_text),
        "15_页眉占位": (r"页眉[:：]?\s*待编辑部分配|页眉（待编辑部分配）", md_text),
        "16_无英文刊体件": (r"^\*\*中图分类号：待核\*\*　\*\*文献标识码：Ａ\*\*", front),
    }
    present, absent = [], []
    for name, (rx, hay) in pieces.items():
        (present if re.search(rx, hay, re.M) else absent).append(name)
    return {"present": present, "absent": absent, "n_present": len(present), "n_pieces": len(pieces)}


# -------------------------------------------------- math delimiter check

BRACKET_PAIRS = [("(", ")"), ("（", "）"), ("[", "]"), ("［", "］"), ("{", "}"),
                 ("⟨", "⟩"), ("〈", "〉"), ("“", "”"), ("‘", "’")]

_FULLWIDTH_DIGITS = {ord(c): str(i) for i, c in enumerate("０１２３４５６７８９")}


def delimiter_defects(md_text):
    """A delimiter pair whose left side sits directly on the right opener = empty operand.
    Catches the D-a class: arccos|⟨, t⟩| ."""
    body = body_text(md_text)
    defects = []
    for i, line in enumerate(body.split("\n"), 1):
        for a, b in BRACKET_PAIRS:
            for m in re.finditer(re.escape(a) + r"(?=[,;；，、\s]*" + re.escape(b) + r")", line):
                defects.append({"line": i, "pair": a + b, "text": line.strip()[:120],
                                "why": "empty operand between delimiters"})
            # D-a class proper: an operand slot is empty when a separator sits directly on
            # the opener, or a separator doubles up ("arccos|<, T>|").
            for m in re.finditer(re.escape(a) + r"\s*[,;；，、]", line):
                defects.append({"line": i, "pair": a + "->comma", "text": line.strip()[:120],
                                "why": "missing left operand: separator directly after opener"})
            for m in re.finditer(r"[,;；，、]\s*[,;；，、]", line):
                defects.append({"line": i, "pair": "double-separator", "text": line.strip()[:120],
                                "why": "two separators with an empty operand between them"})
        # operator left dangling before a closing delimiter: "×)", "− ]", ", |"
        for m in re.finditer(r"[=+\-−×·/,]\s*[)）\]］⟩〉]", line):
            defects.append({"line": i, "pair": "operator-then-close", "text": line.strip()[:120],
                            "why": "dangling operator before closing delimiter"})
    return defects


# --------------------------------------------------- figure/table citations

def figure_table_citations(md_text):
    caps = {"图": caption_numbers(md_text, "图"), "表": caption_numbers(md_text, "表")}
    body = body_text(md_text)
    out = {}
    for kind, nums in caps.items():
        for n in nums:
            # a mention counts only if it is not the caption line itself
            pat = re.compile(r"%s\s*%d(?!\d)" % (kind, n))
            mentions = [l for l in body.split("\n")
                        if pat.search(l) and not l.strip().startswith("**" + kind + str(n))]
            out["%s%d" % (kind, n)] = len(mentions)
    return out, caps
