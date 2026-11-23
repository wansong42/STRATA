"""R263 混标扫描核 —— "集合档绝对误差区间被当作对基线的增益引用" 永久机检.

三条规则 (唯一实现, 供 scripts/tests/报告共同消费):
  R-A  预注册具体形态: 增益词 与 绝对误差读数 token (1.38*/4.37*/15.37*) 同句共现
  R-B  泛化形态: 增益词 + 区间 "A–B", 其端点命中 em_terzaghi_K/kmeans_obs 的**绝对读数集**
       而**不命中逐靶 headroom 集** (集合由源 JSON 运行时直读, 零硬编码)
  R-C  计数口径: 六靶语境里把过线数写成 5 (five of six / 5 of 6 / n = 5 targets)

豁免标记 (exemption markers): 同句显式声明"这是绝对误差而非增益"才允许共现。
用法:  python scripts/r263_gain_mislabel_scan.py --self-test   # 负例必红 + 正确句必绿
       python scripts/r263_gain_mislabel_scan.py --out FILE     # 全库清单 JSON
       python scripts/r263_gain_mislabel_scan.py --scope delivery --fail-on-hit
             # 作用域 = 交付面 (清单由在盘权威件运行时直读, 见 delivery_face_authority)
             # --fail-on-hit: 作用域内非豁免命中 > 0 或作用域断言不成立 ⇒ 退出码 1
             # --scope full (默认) 行为与加作用域之前逐位一致

作用域层 (终局期值守工作令 R263-SCOPE): 判据本体 (R-A/R-B/R-C 词表与判定函数) 一字未改,
新增的只有"扫哪些文件"与"命中了怎么Exit"两层 —— 由 --scope / --fail-on-hit 承载。
"""
import argparse
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDRM_FILE = "results/r208_mixture_reboot/diag_em_baseline.json"
R212_FILE = "results/r212_metric_aligned_loss/verdict.json"

GAIN_RX = re.compile(
    r"\b(improv\w*|gain\w*|headroom|outperform\w*|beat\w*|beats)\b"
    r"|增益|优于|反超|降低|收益"
    , re.I)
# R-A: 具体病灶 token (允许任意尾随小数位, 使 1.3833 与 1.38 同样入网)
ABS_RX = re.compile(r"(?<![\d.])(1\.38\d*|4\.37\d*|15\.37\d*)(?![\d])")
# R-C: 计数口径病灶
COUNT_RX = re.compile(r"(five of (the )?six|5 of (the )?6|n\s*=\s*5 targets?|five of six)", re.I)
SET_CONTEXT_RX = re.compile(r"k-means|km|vMF|EM|target|truth|synthetic|headroom|增益|靶|站点", re.I)
DEGRADATION_NEUTRAL_RX = re.compile(r"degraded|excluded|not dominant|反向|降级|剔除", re.I)

EXEMPTION_MARKERS = [
    "absolute level", "absolute error", "absolute modal", "not a gain", "not gains",
    "not the gain", "not headroom", "rather than a gain", "is an absolute",
    "绝对误差", "非增益", "不是增益", "而非增益", "绝对水平",
]

NUMBER_RX = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d])")
INTERVAL_RX = re.compile(r"(?<![\d.])(\d+\.\d+)\s*(?:[–\u2013\u2014-]|to)\s*(\d+\.\d+)(?![\d])")


def _load_sets():
    """运行时直读源件: 绝对读数集 + headroom 集 (含 2/3/4 位小数舍入形态)。"""
    hr = json.load(io.open(os.path.join(ROOT, HDRM_FILE), encoding="utf-8"))
    hd = hr["summary"]["per_site_headroom_terzaghi_minus_kmeans_deg"]
    v = json.load(io.open(os.path.join(ROOT, R212_FILE), encoding="utf-8"))
    sites = v["four_arm"]["sites"]
    t207 = [k for k in sites if k.startswith("r207_")]
    absolute, baseline = set(), set()
    for k in t207:
        absolute.add(float(sites[k]["em_terzaghi_K"]))
        baseline.add(float(sites[k]["kmeans_obs"]))
    decov = [k for k in sites if k.startswith("decovalex_")]
    for k in decov:
        absolute.add(float(sites[k]["em_terzaghi_K"]))
        baseline.add(float(sites[k]["kmeans_obs"]))
    headroom = {float(x) for x in hd.values()}
    return absolute | baseline, headroom


def _dp_forms(values):
    """把源值展开成 2/3/4 位小数的可匹配字符串集。"""
    out = set()
    for v in values:
        for dp in (2, 3, 4):
            out.add(("%." + str(dp) + "f") % abs(v))
    return out


HARD_UNIT_RX = re.compile(r"^\s*(\||#|>|\*\s|-\s|\d+[.)]\s|[-=]{3,})")


def split_sentences(text, prose=False):
    """按句切单元。prose=True 时先把 markdown 软折行并回整段再切句;
    表格行 / 列表项 / 标题 / 引用块 保持独立单元; 机读格式 (json/js/csv/py) 逐行为单元。
    """
    sents = []
    if not prose:
        for ln, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9*\[])|(?<=）)\s+", s):
                p = p.strip()
                if p:
                    sents.append((ln, p))
        return sents
    buf, buf_ln = [], None
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            if buf:
                for p in _sent_split(" ".join(buf)):
                    sents.append((buf_ln, p))
                buf, buf_ln = [], None
            continue
        if HARD_UNIT_RX.match(line):
            if buf:
                for p in _sent_split(" ".join(buf)):
                    sents.append((buf_ln, p))
                buf, buf_ln = [], None
            sents.append((ln, s))
            continue
        if buf_ln is None:
            buf_ln = ln
        buf.append(s)
    if buf:
        for p in _sent_split(" ".join(buf)):
            sents.append((buf_ln, p))
    return sents


def _sent_split(s):
    return [x.strip() for x in
            re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9*\[])", s) if x.strip()]


def has_marker(s):
    low = s.lower()
    return any(m.lower() in low for m in EXEMPTION_MARKERS)


def scan_text(text, abs_forms, hr_forms, prose=False):
    """核心判据: 返回命中列表 (dict)。prose=True 用软折行整段单元。"""
    hits = []
    for ln, s in split_sentences(text, prose=prose):
        if not GAIN_RX.search(s):
            continue
        markers = has_marker(s)
        for tok in ABS_RX.findall(s):
            hits.append({"rule": "R-A", "line": ln, "token": tok,
                         "exempt": markers, "sentence": s[:400]})
        for a, b in INTERVAL_RX.findall(s):
            in_abs = a in abs_forms or b in abs_forms
            in_hr = a in hr_forms or b in hr_forms
            if in_abs and not in_hr:
                hits.append({"rule": "R-B", "line": ln, "token": "%s-%s" % (a, b),
                             "exempt": markers, "sentence": s[:400]})
        if COUNT_RX.search(s) and SET_CONTEXT_RX.search(s) and not DEGRADATION_NEUTRAL_RX.search(s):
            hits.append({"rule": "R-C", "line": ln, "token": COUNT_RX.search(s).group(0),
                         "exempt": markers, "sentence": s[:400]})
    return hits


TEXT_EXT = (".md", ".txt", ".html", ".json", ".js", ".yaml", ".yml", ".csv")
SKIP_NAME_RX = re.compile(r"plotly\.min\.js|node_modules|\.git/|__pycache__", re.I)

CORPUS = [
    "docs/paper1", "docs/结题报告", "docs/论文素材包", "docs/论文",
    "demos/portal", "results/r254_paper1_v21", "results/r258_refs",
    "tasks", "archive/交接文档", "advisor_pack", "src",
]
CORPUS_ROOT_DOCS = True  # A2: docs/ 根级 .md 一并入扫
CORPUS_FILES = ["AGENTS.md", "README.md", "docs/对外口径冻结页_R213.md", "docs/论文素材总表_R213.md",
                "docs/论文①技术与实现详解_20260918.md"]
PDF_FACES = ["docs/paper1", "advisor_pack"]  # 渲染件同样扫 (制品面)
CODE_EXT = (".py",)

# ── 作用域层常量 (R263-SCOPE; 判据本体在以上, 以下只决定"扫哪些文件"与退出码) ──
SCOPE_FULL = "full"
SCOPE_DELIVERY = "delivery"
SCOPE_CHOICES = (SCOPE_FULL, SCOPE_DELIVERY)
SCAN_MODE = "in_process_run_scan"          # 口径标签: 进程内枚举, 非 stdout 解析
# 交付面清单的两份在盘权威件 (禁在本文件硬编码第二份清单)
AUTHORITY_MANIFEST = "results/r322_final_settlement/retirement_manifest.json"
AUTHORITY_DECLARATION = "results/_retired_narrative_20261001/治理退役声明.md"
DECL_SEC_START = u"## 二 ·"
DECL_SEC_END = u"## 三 ·"
DOC_PATH_RX = re.compile(u"docs/[^\\s`）)、，,]+?\\.md")
# 作用域自测用: 内存投毒的靶文件与毒句 (毒句取自 NEGATIVE_PILLS, 见 self_test)
PILL_TARGET = "results/_retired_narrative_20261001/治理退役声明.md"
PILL_ID = "NP-4"
# 加作用域时唯一准变的三个既有函数 (其余判据节点 AST 必须逐位不变)
MUTABLE_NODES = frozenset(["run_scan", "self_test", "main"])



def iter_targets():
    seen = set()
    for rel in CORPUS_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p) and rel not in seen:
            seen.add(rel)
            yield rel
    for rel in sorted(os.listdir(os.path.join(ROOT, "docs"))):
        if not CORPUS_ROOT_DOCS:
            break
        fp = os.path.join(ROOT, "docs", rel)
        r = "docs/" + rel
        if os.path.isfile(fp) and rel.lower().endswith(TEXT_EXT) and r not in seen:
            seen.add(r)
            yield r
    for rel in CORPUS:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if "__pycache__" not in d]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                r = os.path.relpath(fp, ROOT).replace("\\", "/")
                if r in seen or SKIP_NAME_RX.search(r):
                    continue
                if rel == "src" and not fn.lower().endswith(CODE_EXT + TEXT_EXT):
                    continue
                if fn.lower().endswith(TEXT_EXT):
                    seen.add(r)
                    yield r
    for rel in PDF_FACES:
        base = os.path.join(ROOT, rel)
        for dirpath, dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.lower().endswith(".pdf"):
                    r = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/")
                    if r not in seen and not SKIP_NAME_RX.search(r):
                        seen.add(r)
                        yield r


def reflow(text):
    """把 PDF 抽取的换行折回段落: 单换行 = 软折行 (合并), 空行 = 段落界。

    不做这一步, 一句会被排版拆成碎片, 句内限定语 (如 the degraded sixth) 就落在另一个
    扫描单元里, 判据会假红 —— 这是修扫描单元, 不是放宽判据。
    """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def read_text(rel):
    if rel.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception:
            return None, "pypdf-unavailable"
        try:
            r = PdfReader(os.path.join(ROOT, rel))
            return reflow("\n".join((pg.extract_text() or "") for pg in r.pages)), None
        except Exception as exc:
            return None, "pdf-read-error:%s" % type(exc).__name__
    try:
        t = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
    except UnicodeDecodeError:
        try:
            t = io.open(os.path.join(ROOT, rel), encoding="utf-8-sig").read()
        except Exception as exc:
            return None, "decode-error:%s" % type(exc).__name__
    except Exception as exc:
        return None, "read-error:%s" % type(exc).__name__
    return t, None


FACE_RULES = [
    (re.compile(r"submission_pack_v2|/extracted/|zenodo_deposit"), "OUTWARD_PACK"),
    (re.compile(r"^advisor_pack"), "ADVISOR_PACK"),
    (re.compile(r"^src/"), "DELIVERY_CODE"),
    (re.compile(r"paper1_v3\.2"), "OUTWARD_LIVE"),
    (re.compile(r"paper1_v(0\.1|0\.2|0\.2\.1|1\.0|2\.0|2\.1|3\.0|3\.1)|draft_v|\.pdf$"),
     "HISTORY_FROZEN"),
    (re.compile(r"^tasks/|^archive/交接文档|^AGENTS\.md|^docs/架构师交接"), "INTERNAL_GOVERNANCE"),
    (re.compile(r"^demos/portal"), "PORTAL"),
    (re.compile(r"^docs/结题报告"), "FINAL_REPORT"),
    (re.compile(r"^docs/论文|^docs/对外口径|^docs/论文素材"), "INTERNAL_MATERIAL"),
    (re.compile(r"^results/"), "EVIDENCE"),
]


def classify(rel):
    for rx, face in FACE_RULES:
        if rx.search(rel):
            return face
    return "OTHER"


# ═══════════════════════════════════════════════════════════════════════════
# 作用域层 (R263-SCOPE) —— 把 D3 裁定「混标闸只对交付面」从人工分面变成带退出码的断言。
# 本层只调用上面的判据, 不新增、不修改、不放宽任何判据。
# ═══════════════════════════════════════════════════════════════════════════

_RELOC_MANIFEST = "results/_retired_narrative_20261001/MANIFEST.json"
_RELOC_CACHE = None


def _relocation_map():
    """甲方定调 2026-10-01 治理叙事件整件移档后的寻址修复表。

    单一事实来源 = 移档 MANIFEST (moved[].from -> moved[].to), 本函数零硬编码清单。
    语义 = 同一文件的新落位, 非清单扩权; 每条命中在 provenance 里带 relocated_from。
    """
    global _RELOC_CACHE
    if _RELOC_CACHE is None:
        m = {}
        try:
            with io.open(os.path.join(ROOT, _RELOC_MANIFEST), encoding="utf-8") as f:
                man = json.load(f)
            for rec in man.get("moved", []):
                m[rec["from"].replace("\\", "/")] = rec["to"].replace("\\", "/")
        except Exception:
            pass
        _RELOC_CACHE = m
    return _RELOC_CACHE


def delivery_face_authority():
    """运行时直读两份在盘权威件得出交付面清单 (单一实现点, 禁第二份硬编码清单)。

    源 A = retirement_manifest.json 的 guards.G-4.extraction_docs_frozen ∪ G-5.whitelist_new_allowed
    源 B = results/_retired_narrative_20261001/治理退役声明.md §二 内出现的 docs/*.md 路径 + 本声明自身 (文中以「本声明」自称)
    两源不同 ⇒ 不调和: 取并集 (更严), 差集与逐件出处如实随结果输出。
    """
    try:
        with io.open(os.path.join(ROOT, AUTHORITY_MANIFEST), encoding="utf-8") as f:
            man = json.load(f)
        guards = man["guards"]
        src_a = sorted(set(list(guards["G-4"]["extraction_docs_frozen"])
                           + list(guards["G-5"]["whitelist_new_allowed"])))
    except Exception as exc:
        raise SystemExit("交付面权威源 A 读取失败 (%s): %s: %s"
                         % (AUTHORITY_MANIFEST, type(exc).__name__, exc))
    try:
        with io.open(os.path.join(ROOT, AUTHORITY_DECLARATION), encoding="utf-8") as f:
            decl = f.read()
    except Exception as exc:
        raise SystemExit("交付面权威源 B 读取失败 (%s): %s: %s"
                         % (AUTHORITY_DECLARATION, type(exc).__name__, exc))
    try:
        i0 = decl.index(DECL_SEC_START)
        i1 = decl.index(DECL_SEC_END, i0)
    except ValueError:
        raise SystemExit("交付面权威源 B 解析失败: %s 内找不到 %r..%r 分节界"
                         % (AUTHORITY_DECLARATION, DECL_SEC_START, DECL_SEC_END))
    src_b = sorted(set(DOC_PATH_RX.findall(decl[i0:i1])) | {AUTHORITY_DECLARATION})
    # 空集合 = 作用域无牙 (军规⑭), 宁可不跑也不报 0 红
    if not src_a or not src_b:
        raise SystemExit("交付面权威源解析为空 (A=%d B=%d) = 作用域无牙, 拒绝输出"
                         % (len(src_a), len(src_b)))
    reloc = _relocation_map()

    def _res(p):
        return reloc.get(p.replace("\\", "/"), p)

    union = sorted({_res(x) for x in (set(src_a) | set(src_b))})
    prov = {}
    for rel in (set(src_a) | set(src_b)):
        srcs = []
        if rel in src_a:
            srcs.append("manifest:G-4+G-5")
        if rel in src_b:
            srcs.append("declaration:§二" + ("+self" if rel == AUTHORITY_DECLARATION else ""))
        if _res(rel) != rel:
            srcs.append("relocated_from:" + rel)
        prov[_res(rel)] = srcs
    return {"sources": {"A_manifest": {"file": AUTHORITY_MANIFEST,
                                       "pointers": ["guards.G-4.extraction_docs_frozen",
                                                    "guards.G-5.whitelist_new_allowed"],
                                       "files": src_a},
                        "B_declaration": {"file": AUTHORITY_DECLARATION,
                                          "method": "regex docs/*.md in section 二 + self",
                                          "files": src_b}},
            "sets_identical": src_a == src_b,
            "A_minus_B": sorted(set(src_a) - set(src_b)),
            "B_minus_A": sorted(set(src_b) - set(src_a)),
            "ruling": u"两源不同 ⇒ 不调和, 按更严 (并集) 实现; 差集如实随结果输出",
            "union": union, "provenance": prov}


def delivery_face_files():
    return delivery_face_authority()["union"]


def scope_targets(scope, files=None):
    """返回 (作用域内目标清单, 权威源读数或 None)。full = 现盘原样全库。"""
    if files is not None:
        return sorted(files), (delivery_face_authority() if scope == SCOPE_DELIVERY else None)
    if scope == SCOPE_DELIVERY:
        auth = delivery_face_authority()
        return list(auth["union"]), auth
    if scope == SCOPE_FULL:
        return list(iter_targets()), None
    raise SystemExit("未知 --scope: %r (合法值 %s)" % (scope, list(SCOPE_CHOICES)))


def criteria_ast_fingerprint(path=None):
    """判据本体 AST 指纹 —— 只取模块级节点 (tree.body), 禁 ast.walk。

    为什么不用 walk: walk 会把函数体内的局部赋值也记成 "cst:<名>", 同名局部变量互相覆盖,
    后写者胜 ⇒ 新增一个函数就能改变既有键的取值, "无漂移" 变成不可证。模块级每个名字至多一次
    赋值, 而函数体已被整颗 FunctionDef 的 dump 覆盖, 覆盖面不减、歧义归零。
    """
    import ast
    import hashlib
    p = path or os.path.join(ROOT, _self_rel())
    with io.open(p, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name not in MUTABLE_NODES:
                out["fn:" + n.name] = hashlib.sha256(
                    ast.dump(n, include_attributes=False).encode("utf-8")).hexdigest()
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out["cst:" + t.id] = hashlib.sha256(
                        ast.dump(n, include_attributes=False).encode("utf-8")).hexdigest()
    return out


def criteria_ast_diff(before, after):
    """把两份指纹比成三类: 共有键漂移 / 新增 / 消失。只有第一类是改判据, 第三类是删判据。"""
    shared = sorted(set(before) & set(after))
    return {"n_before": len(before), "n_after": len(after),
            "drifted": [k for k in shared if before[k] != after[k]],
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after))}



def _self_rel():
    return os.path.relpath(os.path.abspath(__file__), ROOT).replace("\\", "/")



def build_scope_detail(scope, targets, auth, res, text_source_note=None):
    """作用域断言 bundle: 在场性 / 命中 / 求和自洽 / 退出码 —— 全实测, 不接受人工分面。"""
    red = res["mislabel_hits"]
    scanned = res["files_scanned"]
    by_file = {}
    for h in red:
        by_file[h["file"]] = by_file.get(h["file"], 0) + 1
    per_file = ({rel: by_file.get(rel, 0) for rel in targets}
                if scope == SCOPE_DELIVERY else None)

    missing = [rel for rel in targets if not os.path.isfile(os.path.join(ROOT, rel))]
    unreadable = [{"file": s["file"], "reason": s["reason"]} for s in res["files_skipped"]
                  if s["file"] in set(targets)]
    by_rule_sum = sum(res["by_rule"].values())
    by_face_sum = sum(res["by_face"].values())
    violations = []
    if not targets:
        violations.append("作用域为空 = 0 命中是真空通过而非判定 (军规⑭)")
    if by_rule_sum != len(red) or by_face_sum != len(red):
        violations.append("两维求和与命中总数不等 (R-B 整类被丢的形状): by_rule=%d by_face=%d red=%d"
                          % (by_rule_sum, by_face_sum, len(red)))
    if missing:
        violations.append("作用域文件不在盘 (0 命中不成立): %s" % missing)
    if unreadable:
        violations.append("作用域文件不可读/不可解析: %s" % unreadable)
    if scanned != len(targets) - len(unreadable):
        violations.append("实扫文件数与作用域清单不闭合: scanned=%d targets=%d unreadable=%d"
                          % (scanned, len(targets), len(unreadable)))

    return {"scope": scope, "口径": SCAN_MODE,
            "n_files_in_scope": len(targets),
            "files_scanned": scanned,
            "files_in_scope": list(targets) if scope == SCOPE_DELIVERY else None,
            "authority": auth,
            "per_file_red": per_file if scope == SCOPE_DELIVERY else None,
            "missing_on_disk": missing, "unreadable": unreadable,
            "non_exempt_hits": len(red),
            "exempt_hits": len(res["exempt_hits"]),
            "by_rule": res["by_rule"], "by_rule_sum": by_rule_sum,
            "by_face": res["by_face"], "by_face_sum": by_face_sum,
            "sum_reconcile_ok": by_rule_sum == len(red) and by_face_sum == len(red),
            "violations": violations,
            "text_source_note": text_source_note,
            "exit_code": 1 if violations else 0}


def evaluate_scope(scope=SCOPE_FULL, fail_on_hit=False, text_source=None, files=None):
    """一次作用域判定: 扫描 + 断言 + 退出码。禁改写文件本体 (投毒一律走内存 text_source)。"""
    targets, auth = scope_targets(scope, files=files)
    note = None
    if text_source is not None:
        note = "in_memory_text_source_override (未触盘)"
    res = _scan_targets(targets, text_source=text_source)
    detail = build_scope_detail(scope, targets, auth, res, text_source_note=note)
    rc = detail["exit_code"]
    if fail_on_hit and detail["non_exempt_hits"] > 0:
        rc = 1
    detail["fail_on_hit"] = bool(fail_on_hit)
    detail["rc"] = rc
    return {"scan": res, "scope_detail": detail}



NEGATIVE_PILLS = [
    {"id": "NP-1", "why": "任务书原病灶句 (摘要位)",
     "text": "a gated vMF-EM estimator with Terzaghi correction that improves on "
             "observation-only k-means by 1.38-4.37 deg on five of six synthetic-truth targets"},
    {"id": "NP-2", "why": "把绝对区间称作 headroom (投稿包图注形态)",
     "text": "the dual headroom intervals (1.38-4.37 deg excluding the named degraded target) "
             "are the set-level gains"},
    {"id": "NP-3", "why": "R-B 泛化: 4 位小数绝对值被写成增益 (防换精度规避)",
     "text": "the achiever outperforms the baseline by 2.2399-15.3737 deg across truth targets"},
    {"id": "NP-4", "why": "R-C 计数口径单独病灶 (无绝对 token 也须红)",
     "text": "the set-level estimator beats observation-only k-means on five of six targets"},
    {"id": "NP-5", "why": "v3.1 摘要原句 (含 PDF 软折行) 经 reflow 后仍须红 = 闸没被卸牙",
     "text": "a gated estimator with Terzaghi correction that improves on observation-only "
             "k-means by 1.38-4.37 deg on five of six\nsynthetic-truth targets and reads 4.01 deg "
             "on the external DECOVALEX benchmark"},
]

POSITIVE_CONTROLS = [
    {"id": "PC-1", "why": "正确增益栏: headroom 值 + 反向靶披露",
     "text": "the gated estimator improves on observation-only k-means by 0.6769-2.2745 deg "
             "on the five targets where it is dominant, with headroom >= 1 deg on 4 of 6, "
             "and a reversed headroom of -1.5835 deg at the hardest target"},
    {"id": "PC-2", "why": "正确绝对栏: 绝对值与增益词共现但带显式豁免标记",
     "text": "its own masked modal error spans 1.3833-15.3737 deg, an absolute level, "
             "not a gain over the baseline"},
    {"id": "PC-3", "why": "只谈绝对读数、无区间误标 (结题报告两栏用法)",
     "text": "em_terzaghi_K reads 1.3833-4.3738 deg in absolute level (not gains); the baseline "
             "reads 2.7974-13.7902 deg"},
]


def scope_selftest():
    """作用域层双测: 正例必绿 (现盘交付面 0 红) + 两枚负例必红 (内存投毒 / 文件缺席)。

    全程只在内存操作, 严禁改写交付面文件本体 (投毒一律走 text_source)。
    """
    out = []
    ev = evaluate_scope(SCOPE_DELIVERY, fail_on_hit=True)
    d = ev["scope_detail"]
    in_scope = d.get("files_in_scope") or []
    good = (d["non_exempt_hits"] == 0 and d["n_files_in_scope"] > 0
            and not d["violations"] and d["rc"] == 0 and PILL_TARGET in in_scope)
    out.append({"kind": "scope_positive", "id": "SP-1",
                "why": "delivery 作用域对现盘必须 0 红, 且作用域非空、逐件在盘、退出码 0",
                "must_be": "GREEN", "n_files_in_scope": d["n_files_in_scope"],
                "files_scanned": d["files_scanned"], "non_exempt_hits": d["non_exempt_hits"],
                "violations": d["violations"], "rc": d["rc"],
                "union_from": {"A_minus_B": d["authority"]["A_minus_B"],
                               "B_minus_A": d["authority"]["B_minus_A"],
                               "sets_identical": d["authority"]["sets_identical"]},
                "verdict": "OK" if good else "SCOPE-POSITIVE-RED"})

    base_t, err = read_text(PILL_TARGET)
    if base_t is None:
        out.append({"kind": "scope_poison", "id": "SP-2", "must_be": "RED",
                    "verdict": "PILL-TARGET-UNREADABLE:%s" % err})
        return out
    pill = [p for p in NEGATIVE_PILLS if p["id"] == PILL_ID][0]["text"]
    inj = base_t + "\n\n" + pill + "\n"
    identical = inj.encode("utf-8") == base_t.encode("utf-8")
    ev2 = evaluate_scope(SCOPE_DELIVERY, fail_on_hit=True,
                         text_source=lambda rel: ((inj, None) if rel == PILL_TARGET
                                                   else read_text(rel)))
    d2 = ev2["scope_detail"]
    before, after = d["non_exempt_hits"], d2["non_exempt_hits"]
    good2 = (not identical) and after == before + 1 and d2["rc"] == 1
    out.append({"kind": "scope_poison", "id": "SP-2",
                "why": "向交付面文件 (内存) 注入 %s 病灶句: 命中必须 +1 且该作用域 rc=1" % PILL_ID,
                "must_be": "RED", "pill_target": PILL_TARGET,
                "injection_bytes_differ": not identical,
                "hits_before": before, "hits_after": after, "rules": sorted(
                    {h["rule"] for h in ev2["scan"]["mislabel_hits"]}),
                "rc": d2["rc"],
                "verdict": ("OK" if good2 else
                            ("GATE-TEETH-LOSS:注入与底本字节相同 (禁记 no-op)" if identical
                             else "POISON-NOT-CAUGHT"))})
    ghost = "docs/__r263_scope_ghost_file__.md"
    files = [ghost] + [f for f in in_scope if f != PILL_TARGET]
    ev3 = evaluate_scope(SCOPE_DELIVERY, files=files)
    d3 = ev3["scope_detail"]
    good3 = bool(d3["violations"]) and d3["rc"] == 1 and ghost in d3["missing_on_disk"]
    out.append({"kind": "scope_absence_teeth", "id": "SP-3",
                "why": "作用域清单里放一个不在盘的文件: 必须判红 (证明 0 红不是真空通过)",
                "must_be": "RED", "n_violations": len(d3["violations"]),
                "violations": d3["violations"][:3], "rc": d3["rc"],
                "verdict": "OK" if good3 else "ABSENCE-GATE-TEETH-LOSS"})
    # SP-4: 权威件缺席时两路行为合同 —— delivery 必硬失败, full 附属计数必如实降级不静默
    saved = globals()["AUTHORITY_MANIFEST"]
    try:
        globals()["AUTHORITY_MANIFEST"] = "results/__r263_scope_no_such_authority__.json"
        try:
            evaluate_scope(SCOPE_DELIVERY)
            hard_failed = False
            why = None
        except SystemExit as exc:
            hard_failed, why = True, str(exc)[:160]
        deg = delivery_face_counts({"mislabel_hits": []})
        degraded = "authority_unavailable" in deg
    finally:
        globals()["AUTHORITY_MANIFEST"] = saved
    out.append({"kind": "scope_authority_teeth", "id": "SP-4",
                "why": "权威件缺席: delivery 作用域必须硬失败 (不许悄悄扫 0 件), "
                       "而全库路径的交付面计数必须显式标 unavailable (不许静默出 0)",
                "must_be": "RED", "delivery_hard_fail": hard_failed,
                "full_face_degrades_explicitly": degraded, "reason": why,
                "verdict": "OK" if (hard_failed and degraded) else "AUTHORITY-GATE-TEETH-LOSS"})
    return out



def self_test():
    abs_set, hr_set = _load_sets()
    abs_forms = _dp_forms(abs_set)
    hr_forms = _dp_forms(hr_set)
    report = []
    ok = True
    for pill in NEGATIVE_PILLS:
        hits = [h for h in scan_text(reflow(pill["text"]), abs_forms, hr_forms, prose=True)
                if not h["exempt"]]
        good = bool(hits)
        ok = ok and good
        report.append({"kind": "negative_pill", "id": pill["id"], "why": pill["why"],
                       "must_be": "RED", "rules": sorted({h["rule"] for h in hits}),
                       "verdict": "OK" if good else "GATE-TEETH-LOSS"})
    for ctl in POSITIVE_CONTROLS:
        hits = [h for h in scan_text(reflow(ctl["text"]), abs_forms, hr_forms, prose=True)
                if not h["exempt"]]
        good = not hits
        ok = ok and good
        report.append({"kind": "positive_control", "id": ctl["id"], "why": ctl["why"],
                       "must_be": "GREEN", "rules": sorted({h["rule"] for h in hits}),
                       "sentences_hung": [h["sentence"][:120] for h in hits],
                       "verdict": "OK" if good else "FALSE-POSITIVE"})
    for case in scope_selftest():
        ok = ok and case.get("verdict") == "OK"
        report.append(case)
    return ok, report, {"absolute_forms_n": len(abs_forms), "headroom_forms_n": len(hr_forms),
                        "absolute_values": sorted(abs_set), "headroom_values": sorted(hr_set)}



PROSE_EXT = (".md", ".txt", ".html")


def is_prose(rel):
    return rel.lower().endswith(PROSE_EXT) or rel.lower().endswith(".pdf")


def scan_file(rel, text, abs_forms, hr_forms):
    return scan_text(text, abs_forms, hr_forms, prose=is_prose(rel))


def _scan_targets(targets, text_source=None):
    """判据消费环: 逐件读文本 → scan_file → 汇总。text_source 仅供内存投毒自测, 生产路径为 None。"""
    abs_set, hr_set = _load_sets()
    abs_forms = _dp_forms(abs_set)
    hr_forms = _dp_forms(hr_set)
    rows, skipped = [], []
    n_files = 0
    for rel in targets:
        text, err = read_text(rel) if text_source is None else text_source(rel)
        if text is None:
            skipped.append({"file": rel, "reason": err})
            continue
        n_files += 1
        for h in scan_file(rel, text, abs_forms, hr_forms):
            rows.append(dict(file=rel, face=classify(rel), **h))
    red = [r for r in rows if not r["exempt"]]
    return {"files_scanned": n_files, "files_skipped": skipped,
            "hits_total": len(rows), "mislabel_hits": red,
            "exempt_hits": [r for r in rows if r["exempt"]],
            "by_rule": {k: sum(1 for r in red if r["rule"] == k) for k in ("R-A", "R-B", "R-C")},
            "by_face": {k: sum(1 for r in red if r["face"] == k)
                        for k in sorted({r["face"] for r in red})},
            "sets": {"absolute_values": sorted(abs_set), "headroom_values": sorted(hr_set)}}


def run_scan(scope=SCOPE_FULL, files=None, text_source=None):
    """scope=full (默认) 与加作用层之前逐位一致: 同 iter_targets 顺序、同键集、同读数。"""
    targets, _auth = scope_targets(scope, files=files)
    out = _scan_targets(targets, text_source=text_source)
    out["scope"] = scope
    return out


def _set_digest(rows):
    import hashlib
    key = lambda x: tuple(str(v) for v in (x["file"], x["face"], x["rule"], x["line"],
                                           x["token"], x["sentence"]))
    return hashlib.sha256("\n".join("\x1f".join(key(x))
                                    for x in sorted(rows, key=key)).encode("utf-8")).hexdigest()


BEFORE_SNAPSHOT = "results/_architect_watch_20260924/r263_scope/before_snapshot.json"
RECONCILE_EVIDENCE = "results/_architect_watch_20260924/r263_scope/full_baseline_reconcile.json"


def criteria_ast_before_after(baseline_path):
    """现件 vs 改前整件备份 (scanner_before/) 的判据 AST 三类差。备份不在盘则如实报。"""
    before_copy = os.path.join(os.path.dirname(os.path.join(ROOT, baseline_path)),
                               "scanner_before", os.path.basename(_self_rel()))
    if not os.path.isfile(before_copy):
        return {"before_copy": before_copy, "present": False,
                "conclusion": "before_copy_absent"}
    bf = criteria_ast_fingerprint(before_copy)
    af = criteria_ast_fingerprint()
    diff = criteria_ast_diff(bf, af)
    diff.update({"before_copy": os.path.relpath(before_copy, ROOT).replace("\\", "/"),
                 "present": True,
                 "conclusion": "PASS" if not diff["drifted"] and not diff["removed"] else "FAIL"})
    return diff



def default_path_reconcile(scope, res, baseline=None):
    """T4: 改前/改后对账结论 —— 默认 (full) 路径按命中集摘要与在盘改前基线逐位比。"""
    bl = baseline or BEFORE_SNAPSHOT
    out = {"scope": scope, "baseline_file": bl,
           "criteria_ast_before_after": criteria_ast_before_after(bl)}
    p = os.path.join(ROOT, bl)
    if not os.path.isfile(p):
        out.update({"applicable": scope == SCOPE_FULL, "baseline_on_disk": False,
                    "conclusion": "no_baseline_on_disk"})
        return out
    base = json.load(io.open(p, encoding="utf-8"))
    if scope != SCOPE_FULL:
        out.update({"applicable": False, "baseline_on_disk": True,
                    "conclusion": ("not_applicable_to_new_scope (改前不存在 delivery 路径); "
                                   "默认路径逐位对账留痕 = %s, 实测读数 = %s"
                                   % (RECONCILE_EVIDENCE,
                                      base.get("full", {}).get("mislabel_hits"))),
                    "baseline_full_red_hit_set_sha256": base.get("red_hit_set_sha256")})
        return out
    same_red = _set_digest(res["mislabel_hits"]) == base.get("red_hit_set_sha256")
    same_ex = _set_digest(res["exempt_hits"]) == base.get("exempt_hit_set_sha256")
    agg = base.get("full", {})
    same_agg = (res["files_scanned"] == agg.get("files_scanned")
                and res["hits_total"] == agg.get("hits_total")
                and res["by_rule"] == agg.get("by_rule")
                and res["by_face"] == agg.get("by_face")
                and len(res["mislabel_hits"]) == agg.get("mislabel_hits")
                and len(res["exempt_hits"]) == agg.get("exempt_hits"))
    ast_ok = out["criteria_ast_before_after"].get("conclusion") == "PASS"
    out.update({"applicable": True, "baseline_on_disk": True,
                "red_hit_set_identical": same_red,
                "exempt_hit_set_identical": same_ex,
                "aggregates_identical": same_agg,
                "criteria_ast_identical_on_shared_names": ast_ok,
                "conclusion": "PASS" if (same_red and same_ex and same_agg and ast_ok) else "FAIL"})
    return out



def delivery_face_counts(res):
    """从一次全库扫描里直取交付面逐件命中 (不再扫第二遍), 并核对每件确在盘且确被枚举。

    默认 (full) 路径的失败面不得因权威件缺席而变差: 读不到就如实标 unavailable,
    全库扫描照常出件; delivery 作用域则相反 —— 定不出清单必须硬失败 (见 delivery_face_authority)。
    """
    try:
        auth = delivery_face_authority()
    except (OSError, ValueError, KeyError, SystemExit) as exc:
        return {"authority_unavailable": "%s: %s" % (type(exc).__name__, exc)}
    by_file = {}

    for h in res["mislabel_hits"]:
        by_file[h["file"]] = by_file.get(h["file"], 0) + 1
    enum = set(iter_targets())
    per = {}
    for rel in auth["union"]:
        per[rel] = {"red_hits": by_file.get(rel, 0),
                    "on_disk": os.path.isfile(os.path.join(ROOT, rel)),
                    "in_corpus_enumeration": rel in enum,
                    "sources": auth["provenance"][rel]}
    return {"union": auth["union"], "per_file": per,
            "sets_identical": auth["sets_identical"],
            "A_minus_B": auth["A_minus_B"], "B_minus_A": auth["B_minus_A"],
            "total_red_on_delivery_face": sum(v["red_hits"] for v in per.values()),
            "any_union_file_not_enumerated": [k for k, v in per.items()
                                              if not v["in_corpus_enumeration"]],
            "any_union_file_off_disk": [k for k, v in per.items() if not v["on_disk"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--scope", choices=SCOPE_CHOICES, default=SCOPE_FULL,
                    help="full = 现盘原样全库 (默认); delivery = 只扫交付面 (清单由权威件直读)")
    ap.add_argument("--fail-on-hit", action="store_true",
                    help="作用域内非豁免命中 > 0 ⇒ 退出码 1 (默认不改退出码行为)")
    ap.add_argument("--baseline", default=BEFORE_SNAPSHOT,
                    help="改前基线件 (默认 = 本单登记的 before_snapshot.json; 不在盘则如实报)")
    a = ap.parse_args()
    if a.self_test:
        ok, report, sets = self_test()
        print(json.dumps({"self_test": "PASS" if ok else "FAIL", "report": report,
                          "sets": sets}, ensure_ascii=False, indent=1, allow_nan=False))
        return 0 if ok else 1
    ev = evaluate_scope(a.scope, fail_on_hit=a.fail_on_hit)
    res, detail = ev["scan"], ev["scope_detail"]
    if not detail["sum_reconcile_ok"]:
        raise SystemExit("求和不闭合, 拒绝输出: by_rule=%d by_face=%d red=%d"
                         % (detail["by_rule_sum"], detail["by_face_sum"],
                            detail["non_exempt_hits"]))
    payload = dict(res)
    payload["scope"] = a.scope
    payload["enumeration_mode"] = SCAN_MODE
    payload["sum_reconcile"] = {"by_rule_sum": detail["by_rule_sum"],
                                "by_face_sum": detail["by_face_sum"],
                                "non_exempt_total": detail["non_exempt_hits"],
                                "ok": detail["sum_reconcile_ok"]}
    payload["scope_detail"] = detail
    payload["delivery_face_per_file"] = delivery_face_counts(res)
    payload["default_path_reconcile"] = default_path_reconcile(a.scope, res, baseline=a.baseline)
    payload["criteria_ast_fingerprint"] = criteria_ast_fingerprint()

    txt = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False)
    if a.out:
        tmp = a.out + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            f.write(txt)
        os.replace(tmp, a.out)
        print("wrote", a.out)
    if a.scope == SCOPE_DELIVERY:
        print("scope=delivery in_scope=%d scanned=%d red=%d exempt=%d violations=%d"
              % (detail["n_files_in_scope"], detail["files_scanned"],
                 detail["non_exempt_hits"], detail["exempt_hits"], len(detail["violations"])))
        for rel in sorted(detail["per_file_red"]):
            print("  FILE red=%-3d %s" % (detail["per_file_red"][rel], rel))
        for v in detail["violations"]:
            print("  VIOLATION %s" % v)
    else:
        print("files=%d hits=%d mislabel=%d exempt=%d by_rule=%s by_face=%s"
              % (res["files_scanned"], res["hits_total"], len(res["mislabel_hits"]),
                 len(res["exempt_hits"]), res["by_rule"], res["by_face"]))
    for r in res["mislabel_hits"][:60]:
        print("  RED %-9s %-58s L%-5d %s | %s" % (r["rule"], r["file"][:58], r["line"],
                                                  r["token"], r["sentence"][:110]))
    return detail["rc"]


if __name__ == "__main__":
    sys.exit(main())
