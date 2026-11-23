# -*- coding: utf-8 -*-
"""R281 治理收口 · 公共件 (单一实现, 供 prereg / 重锚 / 登记 / 守卫复用).

CPU only. 一切 json 落盘 allow_nan=False + tempfile/os.replace 原子写 (坑 87).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "r281_governance_close")
RES_REL = "results/r281_governance_close"

PY = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
NP_VER = "2.2.5"
SP_VER = "1.15.3"

# AGENTS.md 里 R257 权威节的行首标记 (section header), 用于有界跨度判定
R257_HEADER = "## \U0001F9F1 R257"
# R257 入册的五条磁盘治理纪律 (与 tests/test_v_r257_.py DISCIPLINES 逐字一致)
DISCIPLINES = ("PYTHONDONTWRITEBYTECODE", "禁全 workspace 清缓存", "robocopy",
               "镜像不得排除", "git 禁全量还原")


def ensure_env():
    """坑 43: 环境毒丸断言, 不符即 SystemExit."""
    exe = os.path.realpath(sys.executable).replace("\\", "/").lower()
    if "pytorch_env" not in exe:
        raise SystemExit("环境闸: sys.executable 非 pytorch_env: %s" % sys.executable)
    import numpy
    import scipy
    if numpy.__version__ != NP_VER:
        raise SystemExit("环境闸: numpy %s != %s" % (numpy.__version__, NP_VER))
    if scipy.__version__ != SP_VER:
        raise SystemExit("环境闸: scipy %s != %s" % (scipy.__version__, SP_VER))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(rel):
    return io.open(os.path.join(ROOT, rel.replace("/", os.sep)), encoding="utf-8").read()


def atomic_write_json(path, obj):
    """tempfile + os.replace 原子写, allow_nan=False (坑 87 / S31)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2,
                         sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def now_stamp():
    import time
    t = time.time()
    lt = time.localtime(t)
    # 坑: 时序一律走 epoch, 不信 TZ 字符串 (R268 TZ-EMPTY-TRAP)
    utc8 = time.strftime("%Y-%m-%dT%H:%M:%S", lt) + "+08:00"
    return {"epoch": t, "utc8": utc8,
            "strftime_raw": time.strftime("%Y-%m-%d %H:%M:%S", lt)}


# ───────────────────────────────────────────────────────────── T4 把守核心 ──────────
# 实案例: R298 red_list_measured.json 自相矛盾 —— summary "5 failed" 而枚举只 4;
#          neighbor r264 "4 failed"→枚举 3; neighbor r291 "2 failed"→枚举 0.
# 通则: 凡从 pytest 运行器 stdout 枚举失败节点, 必须与 summary 计数强制对账,
#       不等即 SystemExit (漏计 = 静默丢红 = 假绿).

_SUMMARY_FAILED = re.compile(r"(\d+)\s+failed")
_SUMMARY_ERROR = re.compile(r"(\d+)\s+errors?")
_SUMMARY_ANY = re.compile(r"\d+\s+(?:passed|failed|errors?|deselected|skipped|warnings?)")
_NODE_FAILED = re.compile(r"^FAILED\s+(\S+?)(?:\s|$)", re.M)
_NODE_ERROR = re.compile(r"^ERROR\s+(\S+?)(?:\s|$)", re.M)
# 短格式失败行 (无 FAILED 前缀, 形如 "path::test - Err"), --tb=line 常见
_LINE_NODE = re.compile(r"^(?:tests[/\\][^\s]+?::[^\s:]+)\s+-\s", re.M)


def parse_pytest_summary(stdout_text):
    """从 pytest stdout 提取 summary 计数的失败/错误数 (无任何汇总行时返回 has_summary=False)."""
    m = _SUMMARY_FAILED.search(stdout_text)
    n_failed = int(m.group(1)) if m else 0
    e = _SUMMARY_ERROR.search(stdout_text)
    n_errors = int(e.group(1)) if e else 0
    # 合法汇总的最低门槛: 出现任一 "N passed/failed/error/..." 或 "no tests ran".
    has_summary = bool(_SUMMARY_ANY.search(stdout_text)) or ("no tests ran" in stdout_text)
    return {"has_summary": has_summary, "n_failed": n_failed, "n_errors": n_errors}


def enumerate_failed_nodes(stdout_text):
    """枚举被报告的失败/错误节点 id 集合 (FAILED + ERROR + 短格式行)."""
    nodes = set()
    nodes |= set(_NODE_FAILED.findall(stdout_text))
    nodes |= set(_NODE_ERROR.findall(stdout_text))
    # 短格式去重: 只在没有 FAILED 前缀版本时补, 避免同节点双计
    if not _NODE_FAILED.search(stdout_text):
        nodes |= set(_LINE_NODE.findall(stdout_text))
    return sorted(nodes)


def reconcile_pytest_failures(stdout_text, rc):
    """T4 把守判据体: 枚举红项必须与运行器汇总强制对账.

    返回 dict; 任何不变量破坏 -> SystemExit (响亮失败, 禁静默丢红).
    不变量:
      I1  summary 缺失 -> 不可审计, SystemExit (禁对无 summary 输出下"零红"结论);
      I2  len(nodes) == n_failed (+ n_errors), 否则 = 漏计, SystemExit;
      I3  rc==0 <=> (n_failed==0 and n_errors==0), 否则 rc 与计数不一致, SystemExit.
    """
    s = parse_pytest_summary(stdout_text)
    nodes = enumerate_failed_nodes(stdout_text)
    expected = s["n_failed"] + (s["n_errors"] or 0)
    res = {
        "summary": s,
        "n_nodes": len(nodes),
        "nodes": nodes,
        "rc": rc,
        "expected_total": expected,
    }
    if not s["has_summary"]:
        raise SystemExit("T4 对账失败: pytest 输出无 summary, 不可审计 (禁止对无汇总输出判'零红'): %r"
                         % res)
    if len(nodes) != expected:
        raise SystemExit(
            "T4 对账失败: 枚举到 %d 个失败节点 != summary 报的 %d (漏计=静默丢红). nodes=%s"
            % (len(nodes), expected, nodes))
    zero = (s["n_failed"] == 0 and (s["n_errors"] or 0) == 0)
    if (rc == 0) != zero:
        raise SystemExit(
            "T4 对账失败: rc=%s 与计数 (failed=%s errors=%s) 不一致"
            % (rc, s["n_failed"], s["n_errors"]))
    res["consistent"] = True
    return res


# ─────────────────────────────────────────────────────────── T1 重锚判据体 ──────────
def count_r257_section_headers(agents_text):
    """test_12 重锚: 权威节以"行首 header"计, 而非全串 (散文引用/回指不计)."""
    return sum(1 for L in agents_text.split("\n") if L.startswith(R257_HEADER))


def bounded_r257_section(agents_text):
    """test_11 重锚: 取 R257 权威节本身 (从行首 header 到下一个行首 '## '), 不到 EOF."""
    lines = agents_text.split("\n")
    si = next((i for i, L in enumerate(lines) if L.startswith(R257_HEADER)), None)
    if si is None:
        raise SystemExit("重锚失效: AGENTS.md 无 R257 权威节 header")
    ei = next((i for i in range(si + 1, len(lines)) if lines[i].startswith("## ")),
              len(lines))
    return "\n".join(lines[si:ei])


def touched_within_window(mtime, prereg_mt, verdict_mt):
    """test_08 重锚: 文件是否落在某单执行窗口 [prereg, verdict] 内 (mtime 只写变).

    True = 该文件在该单窗口内被写 -> 属该单动过; False = 窗口外 (更早/更晚他单).
    """
    return prereg_mt <= mtime <= verdict_mt
