# -*- coding: utf-8 -*-
"""R117 · 数字台账补录 (重构首战三锚) — 指针式条目, 可溯源, 编号动态防撞。

三条 (append-only, 写前备份):
  A) 组系层落败主锚: full 场景 fisher 方向距离 fracgen mean (vs B1/B2 参考带)
  B) 拓扑层胜出锚 (反常档定性: 误差对误差抵消, 非学到连通性): full interdens
  C) 总判级锚: 对 B1 判分格 胜/负 计数

状态标 active-research: 研究线对照数字, 禁入客户承诺语; 不触 S21 值校验分支。

执行:
  $env:PYTHONUTF8="1"; E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r117_append_ledger.py
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "digital_ledger.json")
SOURCE = "results/r117_recon_compare/comparison_table.json"
SRC_ABS = os.path.join(ROOT, SOURCE)


def _atomic_save(ledger):
    import tempfile as _tf
    fd, tmp = _tf.mkstemp(dir=os.path.dirname(LEDGER), suffix=".tmp", prefix="ledger_write_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, LEDGER)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def append_b3_supplement(ledger, t):
    """R116 B3 于本单执行窗口内归队 (合并件形态) → 三臂表已重建, 补充判级入册。
    append-only: 不动 NNS-067~069, 本条为其口径增补 (其 caliber 中 'B3 在飞' 表述由本条更正覆盖)。"""
    c = t["summary"]["by_baseline"]["B3"]["counts"]
    cb1 = t["summary"]["by_baseline"]["B1"]["counts"]
    cb2 = t["summary"]["by_baseline"]["B2"]["counts"]
    mx = max(int(re.match(r"NNS-(\d+)", e["id"]).group(1))
             for e in ledger["entries"] if re.match(r"NNS-\d+", e.get("id", "")))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, "results", f"_r117_backup_digital_ledger_{ts}.json")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2, allow_nan=False)
    entry = {
        "id": f"NNS-{mx + 1:03d}",
        "value": c["fracgen_win"],
        "unit": "count",
        "ci": [c["fracgen_win"], c["fracgen_win"]],
        "caliber": ("R117 重构首战 · B3 并入补充 (更正 NNS-069 caliber 中『B3 在飞』表述: "
                    "R116 于本单窗口内关账, 合并件 arms.ARM_MAIN 形态加载器适配后 --stage table 并入): "
                    f"126 判分格 vs B3 胜{c['fracgen_win']}/负{c['fracgen_loss']}/带内{c['within_2sigma']} "
                    f"(vs B1 胜{cb1['fracgen_win']}/负{cb1['fracgen_loss']}, "
                    f"vs B2 胜{cb2['fracgen_win']}/负{cb2['fracgen_loss']} 不变); "
                    "生成臂显著强于检索臂但两组主战场方向层仍全败 —— 首战终局判级以本条三臂口径为准"),
        "source_json": SOURCE,
        "source_path": "/summary/by_baseline/B3/counts/fracgen_win",
        "lineage": ("R117 三臂对照 (B1 R110 / B2 R115 / B3 R116 合并件形态); "
                    "加载器双形态支持 (逐场景件 / *scorecard*.json arms.ARM_MAIN); "
                    "表与报告重建确定性可复跑 (--stage table + r117_make_report.py)"),
        "status": "active-research",
        "verification": "tests/test_v_r117_.py 7 passed (B3 并入后 T4 完备性重算含 B3 臂)",
    }
    ledger["entries"].append(entry)
    ledger.setdefault("changelog", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "R117 fracgen 首战 · B3 归队并入补充",
        "changes": [
            f"新增 {entry['id']}: vs B3 胜{c['fracgen_win']}/负{c['fracgen_loss']}/带内{c['within_2sigma']} "
            "(三臂终局口径; NNS-069 的『B3 在飞』表述由本条覆盖更正)",
            "B3 以 R116 实际交付合并件 (b3_scorecard.json arms.ARM_MAIN) 适配加载, "
            "flag 规则/结构等值逻辑零改动; comparison_table + report 已按三臂重建",
        ],
    })
    counts = ledger.get("counts", {})
    counts["active"] = counts.get("active", 0) + 1
    counts["total"] = sum([len(ledger["entries"]),
                           len(ledger.get("deprecated", [])),
                           len(ledger.get("selfcheck", []))])
    ledger["counts"] = counts
    _atomic_save(ledger)
    print(f"[r117_ledger] B3 supplement appended {entry['id']} (value={c['fracgen_win']})")


def main():
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    t = json.load(open(SRC_ABS, encoding="utf-8"))
    has_main = any("重构首战" in e.get("caliber", "") for e in ledger["entries"])
    has_supp = any("B3 并入补充" in e.get("caliber", "") for e in ledger["entries"])
    if has_main and not has_supp and t["baselines_presence"].get("B3", {}).get("present"):
        append_b3_supplement(ledger, t)
        return
    if has_main:
        print("[r117_ledger] R117 条目已存在, 幂等退出")
        return

    def next_ids(n):
        mx = max(int(re.match(r"NNS-(\d+)", e["id"]).group(1))
                 for e in ledger["entries"] if re.match(r"NNS-\d+", e.get("id", "")))
        return [f"NNS-{mx + 1 + i:03d}" for i in range(n)]

    ids = next_ids(3)

    full = t["scenarios"]["full"]["metrics"]
    fisher = full["set"]["fisher_mean_direction_distance_deg"]
    inter = full["topology"]["intersection_density_rel_err"]
    cB1 = t["summary"]["by_baseline"]["B1"]["counts"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, "results", f"_r117_backup_digital_ledger_{ts}.json")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[r117_ledger] backup -> {backup}")

    lin = ("R117 fracgen 条件生成 vs 统计基线 · 重构首战; 预注册 results/v_r117_prereg.json (mtime 铁律合规); "
           "冻结评分卡 R110 五层 14 指标直读; 生成器 models/fracgen_v1_v9.pt 只读 (R111 全忠实); "
           "位置/尺寸与 B1 逐 seed 配对 (G4 27/27 逐位), 差异归因方向重构; 全部数字研究线, 禁入承诺语")
    ver = ("tests/test_v_r117_.py 7 passed (schema/跨进程确定性+R111接线锚/协议sha/完备性重算/资产绑定+时序/结构等值界/禁入数字)")

    entries = [
        {
            "id": ids[0],
            "value": round(fisher["fracgen_mean"], 4),
            "unit": "deg",
            "ci": [round(fisher["fracgen_mean"] - fisher["fracgen_std"], 4),
                   round(fisher["fracgen_mean"] + fisher["fracgen_std"], 4)],
            "caliber": ("R117 重构首战 · 组系层落败主锚: fracgen 条件生成 full 全观测场景 "
                        "fisher_mean_direction_distance (3 seeds mean±std) = "
                        f"{fisher['fracgen_mean']:.4f}±{fisher['fracgen_std']:.4f}° vs "
                        f"B1 {fisher['baselines']['B1']['mean']:.4f}° / "
                        f"B2 {fisher['baselines']['B2']['mean']:.4f}° [判级 {fisher['baselines']['B1']['flag']}/"
                        f"{fisher['baselines']['B2']['flag']}]; 机制=方向过弥散→组数过配 "
                        "(setdiff≈5.7 vs 真值1组, 详见 results/r117_compare_report.md §6-1), 组系层对两臂全败"),
            "source_json": SOURCE,
            "source_path": "/scenarios/full/metrics/set/fisher_mean_direction_distance_deg/fracgen_mean",
            "lineage": lin, "status": "active-research", "verification": ver,
        },
        {
            "id": ids[1],
            "value": round(inter["fracgen_mean"], 4),
            "unit": "rel",
            "ci": [round(inter["fracgen_mean"] - inter["fracgen_std"], 4),
                   round(inter["fracgen_mean"] + inter["fracgen_std"], 4)],
            "caliber": ("R117 重构首战 · 拓扑层胜出锚: fracgen full 场景 intersection_density_rel_err = "
                        f"{inter['fracgen_mean']:.4f}±{inter['fracgen_std']:.4f} vs "
                        f"B1 {inter['baselines']['B1']['mean']:.4f} / B2 {inter['baselines']['B2']['mean']:.4f} "
                        "[两臂皆胜]; ⚠ 反常档定性=方向弥散制造的交点增多对消 B1/B2 连通低估 "
                        "(误差对误差抵消, 非模型学到连通性, 报告 §6-2), 引用必携此注"),
            "source_json": SOURCE,
            "source_path": "/scenarios/full/metrics/topology/intersection_density_rel_err/fracgen_mean",
            "lineage": lin, "status": "active-research", "verification": ver,
        },
        {
            "id": ids[2],
            "value": cB1["fracgen_loss"],
            "unit": "count",
            "ci": [cB1["fracgen_loss"], cB1["fracgen_loss"]],
            "caliber": ("R117 重构首战 · 总判级: 126 判分格 (9场景×14指标) 对 B1 "
                        f"胜{cB1['fracgen_win']}/负{cB1['fracgen_loss']}/带内{cB1['within_2sigma']}/"
                        f"结构等值{cB1['structural_tie_vs_B1']}; 对 B2 "
                        f"胜{t['summary']['by_baseline']['B2']['counts']['fracgen_win']}/"
                        f"负{t['summary']['by_baseline']['B2']['counts']['fracgen_loss']}/"
                        f"带内{t['summary']['by_baseline']['B2']['counts']['within_2sigma']}; "
                        "2σ=B1/B2 自身 seed 带 (协议 §aggregation); B3 (R116) 在飞缺席注明, 归队后 --stage table 一键并入"),
            "source_json": SOURCE,
            "source_path": "/summary/by_baseline/B1/counts/fracgen_loss",
            "lineage": lin, "status": "active-research", "verification": ver,
        },
    ]
    for e in entries:
        ledger["entries"].append(e)
        print(f"[r117_ledger] append {e['id']}: {e['caliber'][:60]}...")

    ledger.setdefault("changelog", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "R117 fracgen 条件生成 vs 统计基线 · 重构首战对照成绩单 · 台账补录",
        "changes": [
            f"新增 {ids[0]}~{ids[-1]} (3 条): 组系层落败主锚 fisher {fisher['fracgen_mean']:.4f}° / "
            f"拓扑层胜出锚 interdens {inter['fracgen_mean']:.4f} (误差抵消定性) / "
            f"总判级 对B1 胜{cB1['fracgen_win']}负{cB1['fracgen_loss']}",
            "接线: R111 锚逐位复现 + B1 几何流 27/27 逐位配对 (尺度/空间层结构性等值)",
            "反常档 6 条: 方向过弥散→组数过配 (组系层对两臂全败主机制); 拓扑胜与组系败同源 "
            "(误差对误差); borehole_4 饱和伪差; 钻孔线型锚摆动; 域偏移登记; circ_std 过弥散投影",
            "B3 在飞 (R116) 缺席如实注明; R115 水位令『同时击败 B1 与 B2』口径下两臂皆胜 8 格; "
            "tests/test_v_r117_.py 7 passed; append-only 零删改",
        ],
    })
    counts = ledger.get("counts", {})
    counts["active"] = counts.get("active", 0) + len(entries)
    counts["total"] = sum([len(ledger["entries"]),
                           len(ledger.get("deprecated", [])),
                           len(ledger.get("selfcheck", []))])
    ledger["counts"] = counts

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(LEDGER), suffix=".tmp", prefix="ledger_write_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, LEDGER)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(f"[r117_ledger] ok, total entries: {len(ledger['entries'])}")


if __name__ == "__main__":
    main()
