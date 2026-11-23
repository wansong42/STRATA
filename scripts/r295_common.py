# -*- coding: utf-8 -*-
"""R295 公共层 · 递出前最后一步（标题渲染双闸 + 导师包现役化）。

写面（本单唯一可写）：
    docs/成果总报告/研究工作报告_v1.6.{md,pdf}      （新版，禁覆写 v1.5 及以前）
    docs/成果总报告/导师十分钟版_v1.3.{md,pdf}      （承接 R293 §7「十分钟版对齐」未竟项）
    docs/成果总报告/README.md                       （§一 归一 + 末尾 append-only 作废登记）
    docs/成果总报告/current.json                    （由既有派生链重跑得到，非手改）
    results/r295_final_polish/**, results/r295_final_report.md
    scripts/r295_*.py, tests/test_v_r295_.py
    archive/交接文档/工程师交接文档_R295_*.md

零写：台账、看板、AGENTS.md、advisor_pack/、submission_pack_v2/、docs/paper1|paper2/**、
      tests/ 既有守卫、渲染器（r272_render / r277_render / r280_render / glyphcover）。

纪律锚：坑 43（入口断言解释器）/ S31（allow_nan=False）/ 坑 73（枚举走进程内全量）/
        坑 21（prereg 严格早于判定件）/ 计数三件套（scope + enumerate_method + measured_at）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from r206_env_guard import assert_pinned_env, env_signature  # noqa: E402  （坑 43）

UNIT = "R295"
OUT_DIR = os.path.join(ROOT, "results", "r295_final_polish")
REPORT_DIR_REL = "docs/成果总报告"
REPORT_DIR = os.path.join(ROOT, "docs", "成果总报告")
CAND_DIR = os.path.join(OUT_DIR, "candidate")
SANDBOX_DIR = os.path.join(OUT_DIR, "sandbox")

BASE_REPORT_MD = os.path.join(REPORT_DIR, "研究工作报告_v1.5.md")
BASE_REPORT_PDF = os.path.join(REPORT_DIR, "研究工作报告_v1.5.pdf")
BASE_ONEPAGE_MD = os.path.join(REPORT_DIR, "导师十分钟版_v1.2.md")
BASE_ONEPAGE_PDF = os.path.join(REPORT_DIR, "导师十分钟版_v1.2.pdf")

# 未渲染标题的两处病灶（prereg 前只读取证所得，见 state_snapshot.json）
SITE_A = 408   # ## 参考文献          —— 前一行是正文，无空行分隔
SITE_B = 775   # … 非手填） |## 附录乙 …  —— 标题粘在表格行尾同一行

BLOCK_BEGIN = "<!-- R293-CURRENT-PTR-BEGIN -->"
BLOCK_END = "<!-- R293-CURRENT-PTR-END -->"

# 与 r293_common.DEC_RX 同字面（不 import 它，避免把本单判据钉在他单脚本的可改性上；
# 一致性由 tests/test_v_r295_.py::test_10 逐字对账把守）
DEC_RX = re.compile(r"\d+\.\d+")
NNS_RX = re.compile(r"NNS-\d{3}")
# 坑 69 的反例：路径里的 `#rows/7/value` 不是标题；必须要求 # 后紧跟空格
HASH_HEAD_RX = re.compile(r"(?<!#)#{1,6} ")
BARE_HASH_RUN_RX = re.compile(r"#{2,}")


def now_iso() -> str:
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def epoch() -> float:
    import time
    return time.time()


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def abspath(relp: str) -> str:
    return os.path.join(ROOT, relp.replace("/", os.sep))


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jdump(path: str, obj) -> str:
    """S31：allow_nan=False；UTF-8 无 BOM；键序固定 ⇒ 双跑逐字节可复算。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def jload(path: str):
    return json.loads(read_text(path))


def out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def counted(scope: str, enumerate_method: str, **kw) -> dict:
    """计数三件套（红线）：口径范围 + 枚举方式 + 实测时刻。"""
    return dict(scope=scope, enumerate_method=enumerate_method, measured_at=now_iso(), **kw)


# ------------------------------------------------------------------ PDF 读数
def pdf_text(path: str) -> str:
    from pypdf import PdfReader
    r = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in r.pages)


def pdf_page_texts(path: str):
    from pypdf import PdfReader
    r = PdfReader(path)
    return [(i, p.extract_text() or "") for i, p in enumerate(r.pages, 1)]


def image_xobjects_pagewise(path: str) -> int:
    """页面级图像 XObject 计数。判定一律走在册唯一实现 glyphcover.scan_pdf（R280/R290/R293 同口径），
    本函数只作交叉核对：两口径不等 ⇒ 说明有人在中间改了制品读数，须查明。"""
    import glyphcover as G
    return int(G.scan_pdf(path)["image_xobjects"])


def nul_and_fffd(path: str) -> dict:
    t = pdf_text(path)
    return {"nul": t.count("\x00"), "fffd": t.count("\ufffd")}


# ------------------------------------------------------------------ 读数守恒
def decimal_multiset_of_text(text: str) -> Counter:
    return Counter(DEC_RX.findall(text))


def decimal_multiset_of_file(path: str) -> Counter:
    return decimal_multiset_of_text(read_text(path))


def structural_stripped(text: str) -> str:
    """项目在册的结构性剥离（版本号/图表号/DOI/台账号…）。

    唯一实现 = r272_common.strip_structural（直接复用，不另造口径），
    这样「不新增小数读数」这条红线与 R290 token_guard 判的是同一件事。
    """
    import r272_common as C
    return C.strip_structural(text)


def multiset_sha(counter: Counter) -> str:
    return hashlib.sha256(
        json.dumps(sorted(counter.items()), ensure_ascii=False, sort_keys=True,
                   allow_nan=False).encode("utf-8")).hexdigest()


def multiset_diff(a: Counter, b: Counter) -> dict:
    return {"added": {k: b[k] for k in b if b[k] > a.get(k, 0)},
            "removed": {k: a[k] for k in a if a[k] > b.get(k, 0)},
            "equal": a == b,
            "sha_a": multiset_sha(a), "sha_b": multiset_sha(b),
            "n_a": sum(a.values()), "n_b": sum(b.values())}


# ------------------------------------------------------------------ 空号探测
def free_version(pack_dir: str, stem: str) -> str:
    """从盘面实测取空号（禁手写）：返回「目录里该系列最大版本 + 0.1」，且断言该槽位为空。"""
    rx = re.compile(re.escape(stem) + r"_v(\d+)\.(\d+)\.(md|pdf)$")
    best = (0, 0)
    for fn in sorted(os.listdir(pack_dir)):
        m = rx.match(fn)
        if m:
            best = max(best, (int(m.group(1)), int(m.group(2))))
    nxt = (best[0], best[1] + 1)
    tag = "v%d.%d" % nxt
    clash = [fn for fn in os.listdir(pack_dir)
             if fn.startswith("%s_%s." % (stem, tag))]
    if clash:
        raise SystemExit("FATAL: 期望空号 %s 已被占用：%s" % (tag, clash))
    return tag


# ------------------------------------------------------------------ 写日志（并发协调）
JOURNAL = os.path.join(ROOT, ".cowork-temp", "r295_writes.jsonl")


def journal(kind: str, path: str, note: str = "") -> None:
    """本单每一次落盘都记一行（沿用仓内 .cowork-temp/rNNN_writes.jsonl 约定）。

    作用有两个，都不是装饰：
      ① 别的会话可据此判断交付面归谁写（R285 已用同一约定）；
      ② 本单可**自证**「某文件不是我改的」——precheck 抓到外部漂移时，
         用「日志里零命中该路径」代替「我发誓没写」。
    """
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    relp = rel(path) if os.path.isabs(path) else path
    line = "%s\t%s\t%s%s\n" % (epoch(), kind, relp, ("\t" + note) if note else "")
    with open(JOURNAL, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def journal_paths() -> list:
    if not os.path.isfile(JOURNAL):
        return []
    outp = []
    with open(JOURNAL, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 3:
                outp.append(parts[2])
    return outp


def journalled_touch(path: str) -> bool:
    """本单是否写过该路径（用于外部漂移归因）。"""
    r = rel(path) if os.path.isabs(path) else path
    return r in set(journal_paths())
