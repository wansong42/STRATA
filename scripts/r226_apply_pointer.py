# -*- coding: utf-8 -*-
"""R226-T3 · PENDING-ANCHOR⟦R226⟧ → ⟦R226⟧ 回填器（错峰用, 幂等, 默认 dry-run）

## 为什么是脚本而不是本席直接改
R226 与 R225 **并行在飞**（R225 正在写 `docs/paper1/draft_v0.2.md`）。按任务书
§1-T3「与 R225 错峰, 锚件落盘后通知」, 本席**不抢占 R225 独占的草稿文件**;
改为交付一个可复现的、幂等的回填器 + 交接通知, 由 R225 会话（或架构师）在
其修订落定后一次性执行。

## 授权边界
只做 **token 级替换**（`PENDING-ANCHOR⟦R226⟧` → `⟦R226⟧`）与图例定义句替换。
**不改写任何命题表述** —— L103 那句 "Anchor provenance is pending" 的成文改写
属 R225 WP-1 职权（草稿正文），本器只在报告里给出建议措辞, 不动笔。

## 用法
  # ① 查看将要发生什么（默认, 零写入）
  python scripts/r226_apply_pointer.py
  # ② 真正回填（自动备份 + 写日志）
  python scripts/r226_apply_pointer.py --apply
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRAFT = os.path.join(ROOT, "docs", "paper1", "draft_v0.2.md")
LOG = os.path.join(ROOT, "results", "r226_anchor_305", "pointer_apply_log.json")
BACKUP_DIR = os.path.join(ROOT, "results", "r226_anchor_305", "backup")

OLD_TOKEN = "PENDING-ANCHOR⟦R226⟧"
NEW_TOKEN = "⟦R226⟧"

# 图例定义句（draft_v0.2.md 的 Annotation legend 行内片段）—— 必须一并替换,
# 否则替换后图例会留下一个未定义的 ID 说明。
OLD_LEGEND = ("`PENDING-ANCHOR⟦R226⟧` = a pointer whose on-disk statistical metadata "
              "is being traced in a parallel ticket.")
NEW_LEGEND = ("`⟦R226⟧` = the R226 calibration-anchor provenance pointer; its statistical "
              "metadata (pair count, clustered 95% interval, inclusion/exclusion rule, "
              "corpus composition, tilt and observed-versus-all-point calibers) is on disk "
              "in `results/r226_anchor_305/anchor_stats.json`.")

# 建议（**不自动执行**）: L103 的 pending 语义句改写, 供 R225 WP-1 采择。
ADVISED_PROSE = {
    "target": "docs/paper1/draft_v0.2.md#S4.2",
    "current_fragment": "**Anchor provenance is pending**: the statistical metadata of that anchor ...",
    "advisory": "R225 落定后宜把 pending 语义改为 resolved, 并写入锚件实测字段 —— "
                "N=1652 对 / 37 簇网络簇 bootstrap 95%CI [27.0, 34.2]° / 纳排=对网络零筛选, "
                "全网点真值法向, 未施 Terzaghi / 语料=loaded_real test 折 40 网 8 场地, "
                "LINE 型占主臂成对 72.5%。是否入正文由 R225/架构师裁定, 本器不改写。",
    "auto_applied": False,
}


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main(apply: bool):
    if not os.path.isfile(DRAFT):
        raise SystemExit("草稿不在盘: %s" % DRAFT)
    src = open(DRAFT, encoding="utf-8").read()
    sha_before = sha256_text(src)

    n_token = src.count(OLD_TOKEN)
    n_legend = src.count(OLD_LEGEND)
    out = src.replace(OLD_TOKEN, NEW_TOKEN)
    if n_legend:
        out = out.replace(OLD_LEGEND, NEW_LEGEND)
    sha_after = sha256_text(out)

    report = {
        "tool": "scripts/r226_apply_pointer.py",
        "mode": "apply" if apply else "dry-run",
        "target": "docs/paper1/draft_v0.2.md",
        "run_at": datetime.datetime.now().astimezone().isoformat(),
        "token": {"old": OLD_TOKEN, "new": NEW_TOKEN, "occurrences": n_token},
        "legend_definition": {"replacements": n_legend},
        "sha256_before": sha_before,
        "sha256_after": sha_after,
        "changed": bool(sha_before != sha_after),
        "advisory_not_applied": ADVISED_PROSE,
        "idempotent": n_token == 0,
    }

    if n_token == 0:
        print("[R226-T3] 已无 %s —— 回填已完成或从未挂载（幂等, 零写入）。" % OLD_TOKEN)
        if apply:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        return report

    if not apply:
        print("[R226-T3] DRY-RUN: 将替换 %d 处 %s → %s (+图例定义 %d 处)。"
              % (n_token, OLD_TOKEN, NEW_TOKEN, n_legend))
        print("[R226-T3] 加 --apply 执行（自动备份 + 写日志）。")
        return report

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(BACKUP_DIR, "draft_v0.2.md.pre_r226_pointer_%s" % stamp)
    shutil.copy2(DRAFT, bak)
    with open(DRAFT, "w", encoding="utf-8") as f:
        f.write(out)
    report["backup"] = os.path.relpath(bak, ROOT).replace(os.sep, "/")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    check = open(DRAFT, encoding="utf-8").read()
    report["verify_zero_residual_old_token"] = (check.count(OLD_TOKEN) == 0)
    print("[R226-T3] APPLIED: %d 处 token + %d 处图例; 备份 %s"
          % (n_token, n_legend, report["backup"]))
    print("[R226-T3] 复验: 残留 %s = %d (应为 0)"
          % (OLD_TOKEN, check.count(OLD_TOKEN)))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    a = ap.parse_args()
    main(a.apply)
