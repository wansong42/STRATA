# -*- coding: utf-8 -*-
"""R281 T5d/e · 编号账与队列 = 脚本生成 (坑 72/78: 手写编号账必然周期性说谎).

读 tasks/RN_*.md 与 results/rN_*/verdict.json 的存在性 + mtime, 机器推导:
  * 已签发 (任务书在盘) / 已收口 (verdict 在盘) / 在飞 (任务书在盘但 verdict 不在) / 下一号;
  * 队列前置可验: require_precondition(N, [前置...]) 断言 results/<前置>/verdict.json 在盘,
    不在盘即 SystemExit (裁定 88 落地: "队列前置"必须机器可验, 不靠人肉观察/会话注册表 坑 85).

用法:
    python scripts/r281_queue_registry.py            # 打印 + 落 results/r281_governance_close/queue_registry.json
    python scripts/r281_queue_registry.py --assert-pre R290 R283   # 开工前断言这些前置 verdict 在盘
"""
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r281_common as C  # noqa: E402

TASK_RE = re.compile(r"^R(\d+)_任务书")
RESVER_RE = re.compile(r"^r(\d+)_")


def scan():
    issued = {}   # N -> taskbook path (max mtime if multiple)
    for p in glob.glob(os.path.join(ROOT, "tasks", "R*_任务书*.md")):
        m = TASK_RE.match(os.path.basename(p))
        if not m:
            continue
        n = int(m.group(1))
        mt = os.path.getmtime(p)
        if n not in issued or mt > issued[n]["mtime"]:
            issued[n] = {"taskbook": os.path.relpath(p, ROOT).replace("\\", "/"), "mtime": mt}
    closed = {}   # N -> verdict path
    for d in glob.glob(os.path.join(ROOT, "results", "r*_*/verdict.json")):
        base = os.path.basename(os.path.dirname(d))
        m = RESVER_RE.match(base)
        if not m:
            continue
        n = int(m.group(1))
        closed[n] = os.path.relpath(d, ROOT).replace("\\", "/")
    rows = []
    for n in sorted(set(issued) | set(closed)):
        status = ("已收口" if n in closed else "未收口")
        if n in issued and n in closed:
            state = "已收口"
        elif n in issued:
            state = "已签发未收口(在飞或停摆)"
        else:
            state = "有 verdict 无任务书(异常)"
        rows.append({
            "N": n, "state": state,
            "taskbook": issued.get(n, {}).get("taskbook"),
            "taskbook_mtime": (time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(issued[n]["mtime"])) if n in issued else None),
            "verdict": closed.get(n),
        })
    maxn = max(list(issued) + [0])
    return {"stamped": C.now_stamp(), "max_issued_N": maxn,
            "next_number": maxn + 1, "rows": rows}


def verdict_on_disk(n):
    return os.path.isfile(os.path.join(ROOT, "results", "r%d_*" % n, "verdict.json")) or \
        any(glob.glob(os.path.join(ROOT, "results", "r%d_*/verdict.json" % n)))


def require_precondition(pre_ns):
    """开工前置机器可验: 每个前置须有 verdict 在盘, 否则 SystemExit (拒写并退出)."""
    missing = [n for n in pre_ns if not verdict_on_disk(int(n))]
    if missing:
        raise SystemExit("队列前置未满足 (results/r<前置>/verdict.json 不在盘): %s "
                         "—— 不得开工 (裁定 88: 前置须机器可验, 不靠人肉观察)" % missing)
    return {"ok": True, "preconditions_met": list(pre_ns)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--assert-pre":
        print(json.dumps(require_precondition(sys.argv[2:]), ensure_ascii=False, allow_nan=False))
        sys.exit(0)
    reg = scan()
    reg["watermark"] = "编号账脚本生成件, 禁入对外"
    reg["how_to_use_precondition"] = \
        "python scripts/r281_queue_registry.py --assert-pre <N1> <N2>  (开工前跑; 缺前置即拒)"
    C.atomic_write_json(os.path.join(C.RES, "queue_registry.json"), reg)
    print("已用至 N=%d, 下一号=%d, 行数=%d" % (reg["max_issued_N"], reg["next_number"], len(reg["rows"])))
    for r in reg["rows"]:
        if r["N"] >= 280:
            print("  R%d %s verdict=%s" % (r["N"], r["state"], bool(r["verdict"])))
