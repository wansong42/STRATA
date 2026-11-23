"""R303 体检器底座 —— 稿件读取 / 白名单生长 / 语境豁免引擎 / 权威源直读.

红线 (任务书 §2 表头右列): **本文件不内联任何读数数值**。所有 token 集一律运行时
从在盘权威件 (AUTH 表登记的路径) 直读; 任何一层生长失败必须显式上报, 不得静默缩面
(缩面 = 制造假误杀)。

单元模型 (坑 90 + 裁定 89 的落地): 每条命中带三层语境 —— 句 (sent) / 段 (para) /
所在章节标题链 (section)。豁免判定必须声明**看哪一层**, 且必须给出匹配到的语境标记原文。
行号一律指向稿件原文件行 (不复用 R263 reflow 后的行号 —— 那会把作者送到错误的位置)。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import zipfile

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = os.path.join(ROOT, "results", "r303_draft_check")
DECIMALS_DP = tuple(range(0, 7))
MAX_JSON_BYTES = 12_000_000

# 只登记**路径**; 数值一律运行时直读。
AUTH = {
    "number_kit": "results/r296_writing_kit/number_kit.json",
    "ledger": "results/digital_ledger.json",
    "number_table": "results/r259_thesis_draft/number_table.json",
    "carrier": "docs/论文_墙_发现过程技术报告.md",
    "r263_prereg": "results/r263_gain_erratum/prereg.json",
    "r263_scan": "scripts/r263_gain_mislabel_scan.py",
    "r280_glyph": "scripts/r280_glyph_gate.py",
    "r295_title": "scripts/r295_title_gate.py",
    "figure_manifest": "results/r296_writing_kit/figures/figure_manifest.json",
    "r93_deprecated": "results/r93_deprecated_scan.json",
    "r246_holdout": "results/r246_paper1_rescue/wall_stratified_holdout.json",
    "r300_verdict": "results/r300_borehole1_thirdparty/verdict.json",
    "r301_verdict": "results/r301_rev_ceiling/verdict.json",
}
# R1 白名单第六层 = 在册读数的散文层出处 (缺件必须如实报, 见 Whitelist.missing_carriers)
PROSE_CARRIERS = [
    "docs/13度目标可达性审计.md",
    "docs/评测标准与口径锁定.md",
    "docs/可达性与数据补强方案.md",
    "docs/八期新旧数字对照总表.md",
    "docs/论文素材包/关键数字.md",
    "AGENTS.md",
]

NUMBER_RX = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d])")
UNIT_CTX_RX = re.compile(r"^\s*(°|deg|度|%|±)")
IDENT_BEFORE_RX = re.compile(r"(10\.\d{4}|arXiv[:.]?|\d{4}\.\d{4,5})$")
PATH_RX = re.compile(r"(?:results|docs|data|scripts|src|tasks|E:)/\S*")
SUP_RX = re.compile(r"<sup>.*?</sup>", re.S)
CODE_RX = re.compile(r"`[^`]*`")
DOI_IN_TEXT_RX = re.compile(r"\b10\.\d{4,5}|\b\d{4}\.\d{4,5}")
INTERVAL_RX = re.compile(r"(?<![\d.])(\d+\.\d+)\s*[°deg度]?\s*(?:[–—~～\-]|至|到)\s*"+r"(\d+\.\d+)(?![\d])")

HEAD_RX = re.compile(r"^\s*(#{1,6})\s+(.*)$")
HARD_UNIT_RX = re.compile(r"^\s*(\||#|>|\*\s|-\s|\d+[.)]\s|[-=]{3,})")
# 小数点位不是句界: [.!?] 之后必须真有空白 (r263 同款语义); 中文句号/分号无需空白
SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9*\[「])|(?<=[；。])")


# --------------------------------------------------------------------------- 小工具
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def abspath(relp):
    return os.path.join(ROOT, *relp.split("/"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def jload(relp):
    with io.open(abspath(relp), encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path, data):
    """坑 87: 先在内存序列化成功, 再 tempfile + os.replace; 绝不先截断后序列化。"""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".r303tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path, obj):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1,
                                   allow_nan=False).encode("utf-8"))
    return rel(path)


def write_text(path, text):
    _atomic_write(path, text.encode("utf-8"))
    return rel(path)


def dp_forms(values, dps=DECIMALS_DP):
    """R263 rendering_policy: 渲染值必须等于 round(源值, 该位数)。"""
    out = set()
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        for dp in dps:
            out.add(("%%.%df" % dp) % abs(f))
    return out


def numbers_in(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in(v, acc)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(float(x) for x in NUMBER_RX.findall(obj))
    return acc


def source_paths_inside(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("results/") and v.endswith(".json"):
                out.add(v)
            elif isinstance(v, str) and k in ("source_json", "file", "src", "source_path"):
                out.add(v)
            else:
                source_paths_inside(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            source_paths_inside(v, out)
    elif isinstance(obj, str):
        out.update(re.findall(r"results/[^\s`\"|)），。;；]+?\.json", obj))


# --------------------------------------------------------------------------- 稿件读取 (J4)
W_T_RX = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
W_P_RX = re.compile(r"<w:p(?:\s[^>]*)?>(.*?)</w:p>", re.S)
_XML_ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'"}


def _xml_unescape(s):
    for k, v in _XML_ENT.items():
        s = s.replace(k, v)
    s = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), s)
    return re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), s)


def read_docx_text(path):
    """J4: 真从文档体解文本 (python-docx 不在 pytorch_env ⇒ 直解 word/document.xml)。"""
    with zipfile.ZipFile(path) as z:
        if "word/document.xml" not in z.namelist():
            raise SystemExit("R303: docx 缺 word/document.xml —— 拒按文件名/元数据糊弄")
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    paras = ["\n".join(_xml_unescape(x) for x in W_T_RX.findall(pm.group(1)))
             for pm in W_P_RX.finditer(xml)]
    if not any(p.strip() for p in paras):
        raise SystemExit("R303: docx 解出正文为空 —— 判红而非判绿")
    return "\n".join(paras)


def load_draft(path):
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise SystemExit("R303: 稿件不存在: %s" % path)
    with open(p, "rb") as f:
        raw = f.read()
    ext = os.path.splitext(p)[1].lower()
    if ext == ".docx":
        text, kind = read_docx_text(p), "docx"
    elif ext in (".md", ".txt", ".markdown"):
        text, kind = raw.decode("utf-8", "replace"), "md"
    else:
        raise SystemExit("R303: 不支持的扩展名 %s (本单只做 md/docx)" % ext)
    return {"path": rel(p) if p.startswith(ROOT) else p, "abs": p, "kind": kind,
            "text": text, "bytes": len(raw), "sha256": sha256_bytes(raw)}


class Unit(object):
    """一个判定单元: 句原文 + 原文件行号 + 所在段 + 所在章节标题链。"""
    __slots__ = ("line", "sent", "para", "section")

    def __init__(self, line, sent, para, section):
        self.line = line
        self.sent = sent
        self.para = para
        self.section = section

    @property
    def sent_clean(self):
        return CODE_RX.sub(" ", SUP_RX.sub(" ", self.sent))

    @property
    def para_clean(self):
        return CODE_RX.sub(" ", SUP_RX.sub(" ", self.para))


def units_of(text):
    """行号准确的句单元 (表格行/标题/列表项自成单元; 正文软折行并段后再切句)。"""
    lines = text.split("\n")
    trail, units = [], []
    buf, buf_start = [], [None]

    def section_str():
        return " > ".join(t for _, t in trail)

    def flush():
        if not buf:
            buf_start[0] = None
            return
        para = " ".join(buf)
        first = buf_start[0] or 1
        pos = 0
        for piece in [x for x in SENT_SPLIT_RX.split(para) if x and x.strip()]:
            p = piece.strip()                     # 分片带前导空格会把定位推到上一行行尾
            idx = para.find(p, pos)
            if idx < 0:
                idx = pos
            pos = idx + len(p)
            acc, ln = 0, first
            for j, seg in enumerate(buf):
                nxt = acc + len(seg)
                if idx < nxt or j == len(buf) - 1:   # 严格小于: 落在本行内才算本行
                    ln = first + j
                    break
                acc = nxt + 1
            units.append(Unit(ln, piece.strip(), para, section_str()))
        del buf[:]
        buf_start[0] = None

    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            flush()
            continue
        hm = HEAD_RX.match(raw)
        if hm:
            flush()
            depth = len(hm.group(1))
            while trail and trail[-1][0] >= depth:
                trail.pop()
            trail.append((depth, hm.group(2).strip()))
            units.append(Unit(i, line, line, section_str()))
            continue
        if HARD_UNIT_RX.match(raw):
            flush()
            units.append(Unit(i, line, line, section_str()))
            continue
        if buf_start[0] is None:
            buf_start[0] = i
        buf.append(line)
    flush()
    return units


def table_blocks(text):
    """{行号: 所在连续表格块的全文} —— 表内相邻行视为同一对照语境 (R2b)。"""
    lines = text.split(chr(10))
    blocks, cur, start = {}, [], None
    for i, raw in enumerate(lines, 1):
        if raw.strip().startswith("|"):
            if start is None:
                start, cur = i, []
            cur.append(raw)
        else:
            if start is not None:
                blob = " ".join(cur)
                for j in range(start, i):
                    blocks[j] = blob
                start, cur = None, []
    if start is not None:
        blob = " ".join(cur)
        for j in range(start, len(lines) + 1):
            blocks[j] = blob
    return blocks


def line_sections(text):
    """{行号: 章节标题链} —— 供按行判定的规则 (R7) 取语境。"""
    out, trail = {}, []
    for i, raw in enumerate(text.split("\n"), 1):
        hm = HEAD_RX.match(raw)
        if hm:
            depth = len(hm.group(1))
            while trail and trail[-1][0] >= depth:
                trail.pop()
            trail.append((depth, hm.group(2).strip()))
        out[i] = " > ".join(t for _, t in trail)
    return out


# --------------------------------------------------------------------------- R1 白名单
class Whitelist(object):
    """六层机器生长; 每层记 forms/values/files 与源, 缺件与不可读源件如实入列。"""

    def __init__(self, draft_text=""):
        self.forms = set()
        self.entries = []
        self.layers = {}
        self.missing_carriers = []
        self.unreadable_sources = []
        self.registered_decimals = {}       # 值(六位取整) -> {在册 decimals}
        self._build(draft_text)

    def _add_values(self, name, values, key, src, extra_forms=None, decimals=None):
        vals = [abs(float(v)) for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        forms = dp_forms(vals)
        if extra_forms:
            forms |= set(extra_forms)
        self.layers[name] = {"values": len(set(vals)), "forms": len(forms), "src": src}
        self.forms |= forms
        for v in set(vals):
            self.entries.append({"value": float(v), "key": key, "src": src})
        for val, dp in (decimals or []):
            self.registered_decimals.setdefault(round(abs(float(val)), 6), set()).add(int(dp))

    @staticmethod
    def _json_forms(relp):
        p = abspath(relp)
        if not os.path.isfile(p):
            return set(), "missing"
        if os.path.getsize(p) > MAX_JSON_BYTES:
            return set(), "too_large"
        try:
            return dp_forms(numbers_in(json.load(io.open(p, encoding="utf-8")), [])), "ok"
        except Exception as e:                                # noqa: BLE001
            return set(), type(e).__name__

    def _build(self, draft_text):
        kit = jload(AUTH["number_kit"])
        kv, kdec = [], []
        for r in kit["rows"]:
            for k in ("value", "value_raw"):
                if isinstance(r.get(k), (int, float)):
                    kv.append(float(r[k]))
                    if isinstance(r.get("decimals"), int):
                        kdec.append((abs(float(r[k])), r["decimals"]))
            ci = r.get("ci") or {}
            for k in ("a", "b"):
                if isinstance(ci.get(k), (int, float)):
                    kv.append(float(ci[k]))
        self._add_values("W1_number_kit", kv, "number_kit", AUTH["number_kit"], decimals=kdec)

        led = jload(AUTH["ledger"])
        lv = numbers_in(led.get("entries", []), []) + numbers_in(led.get("deprecated", []), [])
        self._add_values("W2_digital_ledger", lv, "digital_ledger", AUTH["ledger"])

        nt = jload(AUTH["number_table"])
        nv, nf, ndec = [], set(), []
        for r in nt["rows"]:
            for k in ("resolved_raw", "resolved_value"):
                if isinstance(r.get(k), (int, float)):
                    nv.append(float(r[k]))
                    if isinstance(r.get("decimals"), int):
                        ndec.append((abs(float(r[k])), r["decimals"]))
            for pf in list(r.get("printed_forms") or []) + [str(r.get("printed") or "")]:
                if re.match(r"^\d+\.\d+$", pf):
                    nf.add(pf)
                    if isinstance(r.get("decimals"), int):
                        ndec.append((abs(float(pf)), r["decimals"]))
        self._add_values("W3_number_table", nv, "number_table", AUTH["number_table"],
                         extra_forms=nf, decimals=ndec)

        srcs = set()
        source_paths_inside(kit, srcs)
        source_paths_inside(led.get("entries", []), srcs)
        source_paths_inside(led.get("deprecated", []), srcs)
        source_paths_inside(nt, srcs)
        w4, n4 = set(), 0
        for s in sorted(x for x in srcs if x.startswith("results/") and x.endswith(".json")):
            f, st = self._json_forms(s)
            n4 += 1
            if st != "ok":
                self.unreadable_sources.append({"path": s, "why": st})
            w4 |= f
        self.layers["W4_registered_sources"] = {"values": 0, "forms": len(w4), "files": n4,
                                                "src": "在册件点名的 results/*.json"}
        self.forms |= w4

        cited = sorted(set(re.findall(r"results/[^\s`\"|)），。;；]+?\.json", draft_text)))
        w5 = set()
        for c in cited:
            f, st = self._json_forms(c)
            if st != "ok":
                self.unreadable_sources.append({"path": c, "why": "draft-cited:" + st})
            w5 |= f
        self.layers["W5_draft_cited_jsons"] = {"values": 0, "forms": len(w5), "files": len(cited),
                                               "src": "稿件行内点名的 results/*.json"}
        self.forms |= w5

        w6 = set()
        for c in PROSE_CARRIERS:
            p = abspath(c)
            if not os.path.isfile(p):
                self.missing_carriers.append(c)
                continue
            w6.update(NUMBER_RX.findall(io.open(p, encoding="utf-8", errors="replace").read()))
        self.layers["W6_prose_carriers"] = {"values": 0, "forms": len(w6),
                                            "files": len(PROSE_CARRIERS), "src": "PROSE_CARRIERS"}
        self.forms |= w6

    def has(self, token):
        return token in self.forms

    def identity(self, token):
        """token 在其位数下**唯一**对应的在册值; dp<2 或多值歧义 ⇒ None (坑 90)。"""
        dp = len(token.split(".")[1])
        if dp < 2:
            return None, dp
        fv = float(token)
        cand = {e["value"] for e in self.entries
                if abs(round(e["value"], dp) - fv) < 10 ** (-dp) / 2.0 + 1e-12}
        return (cand.pop() if len(cand) == 1 else None), dp

    def nearest(self, token, lo=5e-4, hi=1e-2):   # 0.05%–1% 才算"同一量的舍入/抄写漂移"
        fv = abs(float(token))
        best, bestd = None, None
        for e in self.entries:
            v = e["value"]
            if v <= 0:
                continue
            d = abs(v - fv) / max(v, fv)
            if lo <= d <= hi and (bestd is None or d < bestd):
                best, bestd = e, d
        return best, bestd


# --------------------------------------------------------------------------- 语境豁免引擎
# 每个类声明: 适用规则前缀 + **看哪一层语境** (sent/para/section) + 语境标记正则 + 理由。
# 未匹配到任何类 = 未登记 = 对合法稿件即误杀 (J3)。
EXEMPT_CLASSES = {
    "X-FORBID-LIST": {
        "where": "sent",
        "applies_to": ["R3a", "R3b", "R1", "R6", "R2", "R2f", "R3c", "R3d", "R5"],
        "rx": re.compile(r"判废旧值|禁引清单|出现即错|一票否决|禁入论文承诺线|不得再读作"),
        "reason": "本单元就是「哪些数不能用」的清单本体, 列出废值属其职责 (R93/R263: 废弃值须可检索)",
    },
    "X-QUOTE-AS-WRONG": {
        "where": "sent",
        "applies_to": ["R3a", "R3b", "R2", "R1", "R6", "R4"],
        "rx": re.compile(r"错误写法|你写的|原稿|病灶|作废|已废|判废|旧值|曾写|曾把|曾经|"
                         r"误用|误将|错把|写错|问题[:：]|而非|不是.{0,14}(增益|精度|承诺|基线)|"
                         r"不得引用|不可引用|勿引用"),
        "reason": "该读数作为「错法示例/沿革留档」被引用, 不是作为本文结论被引用",
    },
    "X-HISTORY-ARCHIVE": {
        "where": "section",
        "applies_to": ["R3a", "R3b", "R6", "R2", "R1"],
        "rx": re.compile(r"阶段 \d|发现过程|时间线|历史|沿革|审计|更正|复核|复盘|禁引清单|"
                         r"数字错误钉死|八期|泄漏|对账"),
        "reason": "所在章节是发现史/审计记录位 (取料与治理载体本身), 非论文正文承诺位",
    },
    "X-REGISTRY-SURFACE": {
        "where": "section",
        "applies_to": ["R5", "R3a", "R3b", "R9", "R1", "R6", "R2"],
        "rx": re.compile(r"附录[甲乙丙丁]|数字总表|现役数字|取数表|台账|登记"),
        "reason": "所在章节是数字总表/附录登记位 (在册读数的内部对照表), 非论文正文承诺位",
    },
    "X-NOT-DEPLOYABLE": {
        "where": "sent",
        "applies_to": ["R3b", "R1", "R2"],
        "rx": re.compile(r"不可部署|不可用|仅.{0,4}归因|参考线|须真值指派|真值指派|oracle|"
                         r"not deployable|禁止进客户"),
        "reason": "oracle 面允许出现在「显式标注不可部署」的归因语境 (裁定 83/92 口径)",
    },
    "X-SYNTH-QUALIFIED": {
        "where": "sent",
        "applies_to": ["R6", "R3b"],
        "rx": re.compile(r"合成|孪生|基准|壁面级|DFN|数值实验|synthetic|仿真|虚拟"),
        "reason": "合成域读数已带「这是合成基准」限定 (任务书 R6)",
    },
    "X-PROTOCOL-FOOTNOTE": {
        "where": "sent",
        "applies_to": ["R2", "R9", "R1"],
        "rx": re.compile(r"协议敏感|另一协议|不同协议|不可比较|不可互比|不得同表|脚注|"
                         r"fresh rng|rng999|另一语料|另一口径"),
        "reason": "两套掩码协议的对照只允许出现在协议敏感性说明位 (傍晚轮 §3.3 P1-1)",
    },
    "X-GOVERNANCE-META": {
        "where": "sent",
        "applies_to": ["R9", "R1", "R3a", "R2"],
        "rx": re.compile(r"任务书|台账|门禁|裁定 \d|回执|守卫|水印|附录[甲乙丙丁]|沿革|治理|"
                         r"登记|归属"),
        "reason": "治理/登记文字位, 非论文正文 (R255 治理盔甲只进机读件的同族判定)",
    },
    "X-INLINE-POINTER": {
        "where": "sent",
        "applies_to": ["R1", "R6"],
        "rx": re.compile(r"results/[^\s`\"|)）]+|number_kit|number_table|NNS-\d{3}"),
        "reason": "同一单元自带可回源出处指针 (坑 90 要求语境分类而非一律放行)",
    },
}


def exemption_for(rule_key, u):
    """返回 (类名, 语境标记原文, 理由)。只有声明适用且正则真匹配才算豁免。"""
    for cls in sorted(EXEMPT_CLASSES):
        spec = EXEMPT_CLASSES[cls]
        if rule_key.split("-")[0] not in spec["applies_to"]:   # 规则元精确等值
            continue
        hay = {"sent": u.sent_clean, "section": u.section, "para": u.para_clean}[spec["where"]]
        m = spec["rx"].search(hay)
        if m:
            return cls, m.group(0)[:40], spec["reason"]
    return None, None, None


# --------------------------------------------------------------------------- 权威 token 集
def deprecated_tokens():
    """直读: 载体第 3 章「判废旧值（出现即错）」行 ∪ r93 权威 patterns。零内联。"""
    toks, srcs = set(), []
    try:
        txt = io.open(abspath(AUTH["carrier"]), encoding="utf-8", errors="replace").read()
        m = re.search(r"判废旧值[^\n]*", txt)
        if m:
            row = m.group(0)
            toks.update(NUMBER_RX.findall(row))
            toks.update(x + "°" for x in re.findall(r"(?<![\d.])(\d{1,3})°", row))
            srcs.append(AUTH["carrier"] + " 第 3 章「判废旧值」行")
    except OSError:
        pass
    try:
        for p in jload(AUTH["r93_deprecated"]).get("patterns", []):
            toks.add(str(p))
        srcs.append(AUTH["r93_deprecated"] + "#patterns")
    except Exception:                                             # noqa: BLE001
        pass
    return toks, srcs


def oracle_tokens():
    rows = []
    for r in jload(AUTH["number_kit"])["rows"]:
        if r.get("truth_class") == "oracle":
            rows.append({"value": float(r["value"]), "key": r["key"],
                         "rendered": r.get("value_rendered"),
                         "src": r["src"]["file"], "note": r.get("what_it_is_not")})
    return rows


# 合成域「精度读数」集 = kit 的 CIRCULAR 行 + number_table 中 ev 明确为合成基准/孪生/校准/
# 场地标定/合成站者。刻意排除 `TRUTH(synthetic)`(合成站上量出的真值统计量) 与
# `CIRCULAR(一致性only)`(真实场地一致性档): 要求它们同句带「合成」二字属过度延伸,
# 设计期实测会造 20+ 处假误杀 (该裁定入 prereg#rulings_and_conflicts)。
SYNTH_EV_RX = re.compile(r"合成基准|合成孪生|合成校准|合成场地标定|合成站")


def synth_readings():
    out = []
    for r in jload(AUTH["number_kit"])["rows"]:
        if r.get("truth_class") == "CIRCULAR":
            out.append({"value": abs(float(r["value"])), "key": r["key"],
                        "rendered": r.get("value_rendered"), "src": r["src"]["file"],
                        "sample": r.get("sample"), "must": [],
                        "exact": [str(r.get("value_rendered") or "")]})
    for r in jload(AUTH["number_table"])["rows"]:
        ev = str(r.get("ev") or "")
        v = r.get("resolved_value")
        src = r.get("src")
        if SYNTH_EV_RX.search(ev) and isinstance(v, (int, float)) and isinstance(src, str):
            out.append({"value": abs(float(v)), "key": r["key"], "rendered": r.get("printed"),
                        "src": src, "sample": ev, "must": list(r.get("must_travel") or []),
                        "exact": [str(x) for x in (list(r.get("printed_forms") or [])
                                                   + [str(r.get("printed") or "")])]})
    for rd in out:
        rd["need_synth"] = True
    seen, dedup = set(), []
    for rd in out:
        if rd["value"] < 0.01 or rd["key"] in seen:
            continue                        # 微小值 (p 值/占比) 高撞值 ⇒ 不入 R6 (坑 90)
        seen.add(rd["key"])
        dedup.append(rd)
    return dedup


def travel_obligations():
    """随行义务: 在册行自己登记的 must_travel (取料表行内字段) —— 不分合成域, 一律生效。"""
    out, seen = [], set()
    for r in jload(AUTH["number_table"])["rows"]:
        must = list(r.get("must_travel") or [])
        v = r.get("resolved_value")
        if not must or not isinstance(v, (int, float)):
            continue
        if abs(float(v)) < 0.01 or r["key"] in seen:
            continue
        seen.add(r["key"])
        out.append({"value": abs(float(v)), "key": r["key"], "rendered": r.get("printed"),
                    "src": r.get("src"), "sample": str(r.get("ev") or ""), "must": must,
                    "need_synth": False,
                    "exact": [str(x) for x in (list(r.get("printed_forms") or [])
                                               + [str(r.get("printed") or "")])]})
    return out


def ci_pairs():
    rows = []
    for r in jload(AUTH["number_kit"])["rows"]:
        ci = r.get("ci") or {}
        if ci.get("kind") == "ci" and isinstance(ci.get("a"), (int, float)) \
                and isinstance(ci.get("b"), (int, float)) \
                and isinstance(r.get("value"), (int, float)):
            rows.append({"key": r["key"], "pt": float(r["value"]), "lo": float(ci["a"]),
                         "hi": float(ci["b"]), "src": r["src"]["file"]})
    return rows


def routeb_pair():
    d = {}
    for r in jload(AUTH["number_kit"])["rows"]:
        if str(r["key"]).startswith("routeb_"):
            d[r["key"]] = {"value": float(r["value"]), "rendered": r.get("value_rendered"),
                           "src": r["src"]["file"]}
    return d


def protocol_alternates():
    out = []
    for r in jload(AUTH["number_kit"])["rows"]:
        w = str(r.get("what_it_is_not") or "")
        if "不得同表比较" in w or "不同协议" in w:
            out.append({"key": r["key"], "value": float(r["value"]),
                        "rendered": r.get("value_rendered"), "sample": r.get("sample"),
                        "src": r["src"]["file"], "note": w})
    return out


def k_grid_bounds():
    """在盘的 K 档整段注册端点 (min_K4_16 / max_K4_16 一类), 零内联。"""
    vals = []
    for r in jload(AUTH["number_table"])["rows"]:
        src = str(r.get("src") or "")
        cal = str(r.get("caliber") or "")
        v = r.get("resolved_value")
        if isinstance(v, (int, float)) and (re.search(r"min_K|max_K", src)
                                            or ("K=4" in cal and "16" in cal)):
            vals.append(float(v))
    return sorted(set(vals))


def code_patterns():
    """代号模式取自载体第 3 章「内部代号与治理计数」行 + 通用 r\\d{3}/NNS-\\d{3} 形状。"""
    txt = io.open(abspath(AUTH["carrier"]), encoding="utf-8", errors="replace").read()
    line = next((ln for ln in txt.splitlines() if "内部代号与治理计数" in ln), "")
    named = [n for n in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", line)
             if n.lower() not in ("oracle", "xxx")]
    rxs = [re.compile(r"\b%s\b" % re.escape(n), re.I) for n in sorted(set(named))]
    rxs += [re.compile(r"\bNNS-\d{3}\b"), re.compile(r"\br\d{3}\b(?![\d])"),
            re.compile(r"\bR\d{3}_\w+\b")]
    return rxs, line


# ---------------------------------------------------------------- J5 测量法 (单实现)
# test_20 断言的是"判定件自洽", 而判定件的 J5 读数又来自包含 test_20 的全量跑 ⇒
# "全量必 0 红"对它自己构造性永红 (R257 病根: 把自身某一时刻状态当冻结面)。
# 测量法改为: 全量照跑, 但判据只看**外来失败集**; 自身节点红必须显式登记, 不隐藏也不参与永红。
J5_SELF_NODE = u"tests/test_v_r303_.py::test_20_verdict_and_handback_present_and_self_consistent"


def j5_readout(failed_nodes, passed=None, failed=None, self_node=J5_SELF_NODE):
    """返回 J5 full 支的读数: 外来红节点必须为空; 自身节点红另记字段, 由收口退出码兜底。"""
    fn = [x for x in (failed_nodes or [])]
    foreign = [x for x in fn if x != self_node]
    return {u"self_node": self_node,
            u"failed_nodes": sorted(fn),
            u"foreign_failures": sorted(foreign),
            u"self_node_failed": self_node in fn,
            u"pass": not foreign,
            u"measurement": u"全量收集条数照跑 (条数见 collected 字段, 不手写); "
                            u"判据 = 失败集 ⊆ {自身节点} 且集合外 0 红 "
                            u"(自身节点红另由收口退出码与 self_node_failed 字段兜底, 不隐藏)"}

# ---------------------------------------------------- 变更令 R303-2 (1): 代号只认显式禁写标记
EXEMPT_CLASSES["X-EXPLICIT-BAN-MARKER"] = {
    "where": "sent",
    "applies_to": ["R5", "R3c", "R3d"],
    "rx": re.compile(r"禁写|禁用|不得入稿|不许入稿|一律不入稿|勿使用|勿引用|禁引清单|判废旧值"
                     r"|错误写法|你写的|反例|一票否决|下列代号|代号清单|禁引用|禁单独引用"),
    "reason": "仅当同行含**显式**禁写/清单/示例标记才豁免 (R263 豁免只认显式措辞的先例)。"
              "代号入稿是裁定 94.2 的一票否决项, 误豁免等于把红线变绿色通道。",
}

# ---------------------------------------------------- 变更令 R303-2 (2): 三条禁令的机器读数
QUOTE_CHARS = chr(0x201C) + chr(0x201D) + chr(0x0022) + chr(0x0027) + chr(0x300C) + chr(0x300D)
QUOTED_RX = re.compile("[" + QUOTE_CHARS + r"]([^" + QUOTE_CHARS + r"]{2,40})[" + QUOTE_CHARS + r"]")
BAN_LINE_RX = re.compile(r"禁写形\s*(X-\d+)（(R\d+)）[:：]\s*(.+)")
LIMIT_WORDS = (r"天花板", r"上限", r"不变量", r"极限")
QUAL_CANDIDATES = (r"发射序", r"组数口径", r"truth_k_leader", r"k\u0302 ==")
NUMPAIR_RX = re.compile(r"\d+/\d+")


def anchor_ci_set():
    """墙锚 CI 的**成套身份**: 主口径 (r226 聚类 bootstrap) 与盘上他组重抽 (r246 分层留一)。

    裁定 82 把'把正文值 27.01/34.25 改写成 27.12/34.08'定为 P0 红线 => 判据用同件成套身份,
    不用字符串巧合。两组各自成对、各自绑源件; 读不到的源必须入 missing (不得静默放行)。
    """
    prim, missing = None, []
    for r in jload(AUTH["number_kit"])["rows"]:
        ci = r.get("ci") or {}
        if r.get("key") == "wall_anchor_mean" and ci.get("kind") == "ci":
            prim = {"lo": float(ci["a"]), "hi": float(ci["b"]), "mean": float(r["value"]),
                    "src": r["src"]["file"], "pointer": "statistics/ci95_cluster_bootstrap"}
    if prim is None:
        missing.append(AUTH["number_kit"])
    rivals = []
    try:
        v = ((jload(AUTH["r246_holdout"]).get("anchor_ci_propagation") or {})
             .get("anchor_ci_deg")) or []
        if len(v) == 2:
            rivals.append({"lo": float(v[0]), "hi": float(v[1]), "src": AUTH["r246_holdout"],
                           "pointer": "anchor_ci_propagation/anchor_ci_deg",
                           "name": "分层留一重抽 (R246)"})
        else:
            missing.append(AUTH["r246_holdout"])
    except Exception:
        missing.append(AUTH["r246_holdout"])
    return {"primary": prim, "rivals": rivals, "missing": missing}


def _r301_qualifiers():
    try:
        t4 = (jload(AUTH["r301_verdict"]) or {}).get("T4_statement", {}) or {}
    except Exception:
        t4 = {}
    blob = repr(t4)          # 只作子串检索, 不做序列化交付 ⇒ 无需 json.dumps/allow_nan
    return [w for w in QUAL_CANDIDATES if w in blob] or [QUAL_CANDIDATES[0], QUAL_CANDIDATES[1]]


def wording_bans():
    """措辞禁令输入: 载体第 3 章'禁写形 X-n（Rxxx）'行 ∪ R301 claim_forbidden (全部直读)。

    每条 -> {id, kind: literal|cooccur, phrase|numbers+limit_words, qualifier, replacement, source}
    源读不到 ⇒ 入 missing_sources, 规则侧必须响亮报输入缺失而非静默放行。
    """
    out, missing, seen = [], [], set()
    try:
        carrier = io.open(abspath(AUTH["carrier"]), encoding="utf-8", errors="replace").read()
    except OSError:
        carrier = ""
        missing.append(AUTH["carrier"])
    for bid, decl, body in BAN_LINE_RX.findall(carrier):
        for q in QUOTED_RX.findall(body):
            if q not in seen:
                seen.add(q)
                out.append({"id": bid, "kind": "literal", "phrase": q, "numbers": [],
                            "limit_words": [], "source": AUTH["carrier"], "declared_from": decl,
                            "qualifier": _r301_qualifiers(), "replacement": "",
                            "line_body": body[:160]})
        nums = NUMPAIR_RX.findall(body)
        lims = [w for w in LIMIT_WORDS if w in body]
        key = (tuple(sorted(nums)), tuple(sorted(lims)))
        if nums and lims and key not in seen:
            seen.add(key)
            out.append({"id": bid, "kind": "cooccur", "phrase": "", "numbers": nums,
                        "limit_words": lims, "source": AUTH["carrier"], "declared_from": decl,
                        "qualifier": _r301_qualifiers(), "replacement": "", "line_body": body[:160]})
    try:
        t4 = (jload(AUTH["r301_verdict"]) or {}).get("T4_statement", {}) or {}
    except Exception:
        t4 = {}
        missing.append(AUTH["r301_verdict"])
    qual = _r301_qualifiers()
    for item in (t4.get("claim_forbidden") or []):
        nums, lims = NUMPAIR_RX.findall(item), [w for w in LIMIT_WORDS if w in item]
        key = (tuple(sorted(nums)), tuple(sorted(lims)))
        if not (nums and lims) or key in seen:
            continue
        seen.add(key)
        out.append({"id": "R301", "kind": "cooccur", "phrase": "", "numbers": nums,
                    "limit_words": lims, "source": AUTH["r301_verdict"], "declared_from": "R301",
                    "qualifier": qual, "replacement": (t4.get("claim_allowed") or "")[:200],
                    "line_body": item[:160]})
    rep = ""
    try:
        v300 = jload(AUTH["r300_verdict"]) or {}
        rep = str((((v300.get("adjudication") or {}).get("recommendation_to_architect")
                    or [""])[0]))
    except Exception:
        missing.append(AUTH["r300_verdict"])
    for b in out:
        if b["declared_from"] == "R300":
            m = re.search(r"NOT_COMPARABLE_\w+", rep)
            b["replacement"] = ((m.group(0) if m else "NOT_COMPARABLE_*")
                                + " 源 " + AUTH["r300_verdict"] + "#/adjudication")
    return {"bans": out, "missing_sources": missing}


def synth_endpoints_all():
    """合成/自证域端点**全集** (供 R6b 外推禁令) —— 覆盖全部在册合成域行, 不止带 must_travel 者。"""
    vals = []
    for r in jload(AUTH["number_kit"])["rows"]:
        if r.get("truth_class") in ("CIRCULAR", "oracle"):
            vals.append(abs(float(r["value"])))
    for r in jload(AUTH["number_table"])["rows"]:
        ev, v = str(r.get("ev") or ""), r.get("resolved_value")
        if re.search(r"合成|synthetic", ev, re.I) and isinstance(v, (int, float)):
            vals.append(abs(float(v)))
    return sorted({round(v, 6) for v in vals if v >= 0.01})   # 不纳入 CIRCULAR(一致性only)

# R6b 只认"显式声明不可外推"的标记 (带合成限定语仍往工程域推, 恰是 R6b 的病灶形)
EXEMPT_CLASSES["X-SCOPE-DISCLOSURE"] = {
    "where": "sent",
    "applies_to": ["R6b"],
    "rx": re.compile(r"不可推广|不得推广|不能推广|勿推广|仅合成|只适用|适用域|限于合成|不外推|"
                     r"不等同于真实|与真实场地不可比|仅为原理性"),
    "reason": "显式声明了该读数不可外推到真实/工程域 (任务书 R6 的反例条款), 才算合格表述",
}

# 他组重抽的合法对照陈述必须**显式**声明身份 (裁定 82 只禁"把正文值改写成他组端点",
# 不禁"说明有两组重抽并给出差异") —— 只认显式措辞, 不吃泛历史语境。
EXEMPT_CLASSES["X-REDRAW-CONTEXT"] = {
    "where": "sent",
    "applies_to": ["R2f"],
    "rx": re.compile(r"重抽|稳健性|同一估计量|另一组|对照组|敏感性|非现役|未采用|历史值|"
                     r"种子级|蒙特卡洛|噪声|差\s*0\.|并列"),
    "reason": "该句显式把自己标成他组重抽/稳健性对照 (而非现役主口径), 属合法并列陈述",
}


# ------------------------------------------------- 变更令 R303-2 (收口): 豁免适用面归一化
# 旧写法用族字母 ("R1"/"R2"/"R3"/"R4") 表示"该族所有子规则"。前缀匹配改精确等值后,
# 在此把它们**显式**展开, 保持原有覆盖面一字不减 (只准加严, 不准因重构变松)。
# 展开表刻意不含 R2f / R3c / R3d / R6b / R7* / R8*: 它们是变更令新立的收紧项,
# 只有指名登记它们的豁免类才适用 (架构师抓到的正是"泛语境前缀把它们一并吃掉")。
_RULE_FAMILY_EXPAND = {
    "R1": ["R1a", "R1b"],
    "R2": ["R2a", "R2b", "R2c", "R2d", "R2e"],
    "R3": ["R3a", "R3b"],
    "R4": ["R4"],
}

for _spec in EXEMPT_CLASSES.values():
    _tok = set()
    for _a in _spec["applies_to"]:
        _tok.update(_RULE_FAMILY_EXPAND.get(_a, [_a]))
    _spec["applies_to"] = sorted(_tok)

# -------------------------------------- 变更令 R303-2 (收口): 禁令"定义位"从载体直读
def ban_list_site_labels():
    """读载体 `禁入论文承诺线` 块下每条 bullet 冒号前的**标签词** (清单本体的显式标记)。

    返回 (labels, source_pointer)。只认 bullet 首标签, 不认正文任意位置 ——
    否则又退化成泛语境豁免 (架构师抓到的正是这个)。
    """
    try:
        txt = io.open(abspath(AUTH["carrier"]), encoding="utf-8", errors="replace").read()
    except OSError:
        return [], AUTH["carrier"] + " (读不到)"
    m = re.search(r"禁入论文承诺线[^\n]*\n", txt)
    if not m:
        return [], AUTH["carrier"] + " (无『禁入论文承诺线』块)"
    labels = []
    for ln in txt[m.end():].split("\n"):
        if not ln.strip():
            if labels:
                break
            continue
        if not ln.lstrip().startswith(("-", "*", "|")):
            break
        body = ln.lstrip("-*| ").strip()
        head = re.split(r"[:：]", body, maxsplit=1)[0]
        head = re.sub(r"[*</`#]", "", head).strip()
        if 4 <= len(head) <= 24 and head not in labels:   # 只认"短标签：正文"形, 长句不是标签
            labels.append(head)
    return labels, AUTH["carrier"] + " 第 3 章『禁入论文承诺线』块首标签"


_BAN_SITE_LABELS, _BAN_SITE_SRC = ban_list_site_labels()

if _BAN_SITE_LABELS:                      # 直读到才立类; 读不到 ⇒ 不静默造一个空豁免
    EXEMPT_CLASSES["X-BAN-LIST-SITE"] = {
        "where": "sent",
        "applies_to": ["R5", "R3a", "R3b", "R3c", "R3d", "R6"],
        "rx": re.compile("|".join(re.escape(x) for x in _BAN_SITE_LABELS)),
        "reason": "该行是禁令清单的**定义位** (载体『禁入论文承诺线』块首标签, 盘面直读: "
                  + "、".join(_BAN_SITE_LABELS) + ")。清单本体必须能点名被禁对象, 否则无法执行 "
                  "(R93/R263『废弃值须可检索』同族)。来源 " + _BAN_SITE_SRC,
    }

