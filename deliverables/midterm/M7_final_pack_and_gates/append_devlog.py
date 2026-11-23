# -*- coding: utf-8 -*-
"""M7 · 开发日志尾追：逐字节归档 + SHA256 断言 + 追加两行（M3 验收 / M7 终检）。"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
LOG = ROOT / "results" / "devlog" / "开发日志.md"
M7 = ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates"
ARCH = M7 / "_pre_edit_archive"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ARCH.mkdir(parents=True, exist_ok=True)
    before_sha = sha(LOG)
    before_bytes = LOG.stat().st_size
    archived = ARCH / "开发日志_20261121_pre_M3M7.md"
    shutil.copyfile(LOG, archived)
    if sha(archived) != before_sha:
        raise SystemExit("ARCHIVE_SHA_MISMATCH: 归档件与改前字节不一致，拒绝追加")
    with open(LOG, "rb") as f:
        tail = f.read()[-1:]
    prefix = "" if tail == b"\n" else "\n"

    m3 = (
        "2026-11-21 | M3 真实派生域异模型独立判卷（架构师验收） | ACCEPTED | "
        "读盘重算 24 件证据 SHA256 零不符；22/22 簇 AGREE、0 DISAGREE、0 MALFORMED、0 UNDECIDABLE，"
        "10,551 细胞 / 132,000 对 / 75 排序分支 / 74 二阶分支 / 840 期望动作全复算；"
        "jardim_c01 超预算诚实拒收、未混入完备分母；6/6 负控被拒；四道门禁原始退出码全 0；"
        "独立性：零共享代码 + AST 导入审计 0 违规；抽 perozzo_c02 单簇独立复跑，verdict_batch 与存档字节级一致；"
        "输入哈希为 judge_real.canon_sha 口径（首轮按原始字节比对出现 2 条假报警，已按口径复算 0 不符）。 | "
        "results/midterm_sprint_20261120/M3_independent_judge/{result.json,verdict_batch.json,import_audit.json} · "
        "results/midterm_sprint_20261120/_architect_audit_20261121/{acceptance_M3_M6.json,acceptance_spot_runs.json}\n"
    )
    m7 = (
        "2026-11-21 | M7 答辩前夜终检与离线演示打包 | PASS | "
        "M0–M6 七个交付目录 28 件规范文件全部在盘，216 条 evidence 逐件重算 SHA256 零不符；"
        "四道现役门禁五段原始退出码全 0（g1 9/9、g2 40/40 单段 833.7 秒、g3 9 文件非豁免命中 0、g4 四稿 CLEAN）；"
        "STRATA 127.0.0.1:8866 无窗冷启动，4 案例（2 Jardim + 2 Perozzo）GET 全 200、3 次 apply_check 全 200、2 次第二步坍缩分别到 CERTIFIED_CONNECTED / CERTIFIED_DISCONNECTED，"
        "4 簇 M3 徽章均绿色 AGREE（rgb(209,250,229)），运行期外部请求 0，关停后端口与进程零残留；"
        "离线审计 app-owned 外部引用 0、本地引用缺失 0（plotly vendor 84 条字面量单列为未执行）；3/3 负控非零退出（0→2、0→2、0→7）。"
        "两处未达标单列：GET /cases 热击 4.2 秒 > 2 秒；M6 P12 所称 60 秒 B 级录屏全仓 0 个 mp4/webm，速查包已改为 A→C 两级。 | "
        "results/midterm_sprint_20261120/M7_final_pack_and_gates/{result.json,report.md,MIDTERM_MANIFEST.json,cold_start_smoke.json,offline_asset_audit.json,negative_controls.json,答辩现场清单_先读我.md}\n"
    )

    with open(LOG, "ab") as f:
        f.write(prefix.encode("utf-8"))
        f.write(m3.encode("utf-8"))
        f.write(m7.encode("utf-8"))

    after_sha = sha(LOG)
    after_bytes = LOG.stat().st_size
    text = LOG.read_text(encoding="utf-8")
    receipt = {
        "schema": "devlog-append-receipt/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(LOG),
        "archived": str(archived),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "archive_sha_equal": sha(archived) == before_sha,
        "appended_lines": 2,
        "m3_line_present": "M3 真实派生域异模型独立判卷（架构师验收）" in text,
        "m7_line_present": "M7 答辩前夜终检与离线演示打包" in text,
        "prefix_newline_added": prefix == "\n",
    }
    (ARCH / "devlog_append_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["before_bytes", "after_bytes", "archive_sha_equal",
                                              "m3_line_present", "m7_line_present"]}, ensure_ascii=False, indent=1))
    return 0 if receipt["archive_sha_equal"] and receipt["m3_line_present"] and receipt["m7_line_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
