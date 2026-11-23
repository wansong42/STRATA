# -*- coding: utf-8 -*-
"""生成 M6 result.json：枚举规范四件 + 核心交付物与门禁产物的真实字节数与 sha256。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ev(rel: str) -> dict:
    p = D / rel
    assert p.is_file(), f"missing {rel}"
    return {"file": rel.replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}


CORE = [
    "PPT大纲_大创中期_15页.md",
    "口头讲稿_8分钟实战版.md",
    "评委最尖锐20问_标准应答卡.md",
    "number_traceability_map.json",
    "build_midterm_deck.py",
    "midterm_deck.html",
    "verify_m6.py",
    "negative_controls.json",
    "own_defects.json",
    "run_receipt.json",
    "run_m6_gates.py",
    "report.md",
    "gates/g1.json",
    "gates/g2.json",
    "gates/g3.json",
    "gates/g4/draft_check.json",
    "gates/g4_ppt_outline/draft_check.json",
    "gates/g4_spoken_script/draft_check.json",
    "gates/g4_qa_20/draft_check.json",
]

NC = json.loads((D / "negative_controls.json").read_text(encoding="utf-8"))
RR = json.loads((D / "run_receipt.json").read_text(encoding="utf-8"))
TMAP = json.loads((D / "number_traceability_map.json").read_text(encoding="utf-8"))

# verify_m6 统计（从 run_receipt 的 verify_m6_full enumerated 取 613 处）
enum_tokens = next(c["enumerated"] for c in RR["commands"] if c["tag"] == "verify_m6_full")

result = {
    "schema": "dispatch-result/v1",
    "task_id": "M6",
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "scope": "大创中期答辩四幕 15 页 PPT 大纲、8 分钟逐页实战讲稿、A/B/C 三档 20 问应答卡、88 项数字溯源映射、离线 HTML 演示包；M3 已落盘并完成回填（22/22 簇独立核验 AGREE、0 分歧，jardim_c01 仅做超预算拒答状态一致性核验）；四道现役门禁与 verify_m6 全量校验、5 项负控全部通过",
    "enumeration": {
        "positive_cases": 3,
        "positive_passed": 3,
        "negative_controls": NC["total_cases"],
        "negative_correctly_rejected": NC["correctly_rejected"],
        "traceability_entries_verified": TMAP["summary"]["total_registered_entries"],
        "distinct_tokens_registered": TMAP["summary"]["total_distinct_tokens"],
        "numeric_occurrences_checked": enum_tokens,
        "ppt_pages": 15,
        "script_pages": 15,
        "qa_cards": 20,
        "html_slides": 15,
        "m3_pending_placeholders_remaining": 0,
        "m3_agree": TMAP["m3_sync_status"]["agree"],
        "m3_disagree": TMAP["m3_sync_status"]["disagree"],
        "m3_refused_budget_only": ["jardim_c01"],
    },
    "commands": RR["commands"],
    "gates": {
        "g1_repro_certify": "PASS 9/9 (exit 0)",
        "g2_selfcheck_quick": "PASS 40/40 (exit 0)",
        "g3_r263_delivery": "PASS files_scanned=9, 非豁免混标命中=0（mislabel_hits=[]；8 条均为已豁免命中；sum_reconcile.ok=true；exit 0）",
        "g4_r303_draft": "CLEAN n_red=0 × 4（3 份核心文稿 paper + 本报告 governance）",
        "verify_m6": "PASS 88 条目 / 613 处数字 / 3 核心文稿 / 5 负控 (exit 0)",
    },
    "honesty_boundary": {
        "naive_union_on_14_disconnected": "0/14（如实报告，主要危害在 6/6 歧义簇上 MASKS_DISCONNECTED_WITNESS；跨状态假通路牙口由 M2-N1a/b 定向负控实证）",
        "jardim_c01": "按声明 4,096 细胞预算诚实输出 BUDGET_EXCEEDED，不静默截断、不混入 21 簇完备认证分母",
        "m3_independent_judge": "22/22 AGREE、DISAGREE=0；其中 21 簇几何复算，jardim_c01 仅拒答状态一致性核验（不冒充几何重算）",
    },
    "evidence": [ev(r) for r in CORE],
    "not_claimed": [
        "非地下体内无条件贯通真值",
        "非水力连通/渗流导通",
        "非野外地质师实测工时（human_time 为信念树仿真步数）",
        "未把 jardim_c01 超预算拒答混入 21 簇完备认证分母，也未声称对该簇做了几何重算",
        "独立判卷 22/22 AGREE 不等于证明代码绝对无 bug",
        "离线 HTML 仅做结构/占位/资源约束程序核验，未完成逐页浏览器视觉验收，不宣称投影排版零缺陷",
        "HTML 为加分项演示包，非原生 PPTX 成品",
    ],
    "field_error_rate": None,
    "human_time": None,
    "hydraulic": None,
}

(D / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("result.json written;", len(result["evidence"]), "evidence files")
