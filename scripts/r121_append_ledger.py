# -*- coding: utf-8 -*-
"""R121 · 数字台账补录 (组数校准消融 RED 双锚) — 指针式条目, 动态编号防撞, 写前备份。

两条:
  A) 校准臂 full fisher 主锚 (RED 观察值, 阈值与三条件在 caliber 内)
  B) 因果检验锚: k=1 下 interdens 拓扑胜保留 (R117 机制表述精化的数值证据)
状态 active-research (S21 不校验该状态, 但指针仍保持可直读)。
执行: E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r121_append_ledger.py
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "digital_ledger.json")
SOURCE = "results/r121_calib_compare/calib_comparison.json"
SRC_ABS = os.path.join(ROOT, SOURCE)


def main():
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    t = json.load(open(SRC_ABS, encoding="utf-8"))
    if any("R121" in e.get("caliber", "")[:20] or "组数校准消融" in e.get("caliber", "")[:30]
           for e in ledger["entries"]):
        print("[r121_ledger] 已存在, 幂等退出")
        return
    mx = max(int(re.match(r"NNS-(\d+)", e["id"]).group(1))
             for e in ledger["entries"] if re.match(r"NNS-\d+", e.get("id", "")))
    ids = [f"NNS-{mx + 1 + i:03d}" for i in range(2)]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, "results", f"_r121_backup_digital_ledger_{ts}.json")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[r121_ledger] backup -> {backup}")

    fish = t["scenarios"]["full"]["set"]["fisher_mean_direction_distance_deg"]
    idm = t["scenarios"]["full"]["topology"]["intersection_density_rel_err"]
    po = t["summary"]["primary_observations"]["C1"]
    lin = ("R121 生成臂组数校准消融 (R117 发现① 因果验证, 零训练零调参); "
           "预注册 results/v_r121_prereg.json (铁律 4 + 坑 21 双口径); "
           "评分口径=R110 冻结五层, 与 R117 完全一致; "
           "换轨锚: C0 27/27 gen_dir sha 逐位=R117 在盘 (T2); R117 全家 PROTECTED sha 双测零漂移; "
           "B1/B2/B3 与 R117 gen 列全部直读引用不重跑; 研究线数字禁入承诺语")
    ver = "tests/test_v_r121_.py 8 passed (时序/换轨锚/评分列一致/零触碰/k_eff 自洽/完备重算/对账/禁入+实测)"

    e1 = {
        "id": ids[0],
        "value": round(fish["C1"]["mean"], 4),
        "unit": "deg",
        "ci": [round(fish["C1"]["mean"] - fish["C1"]["std"], 4),
               round(fish["C1"]["mean"] + fish["C1"]["std"], 4)],
        "caliber": ("R121 组数校准消融 · RED 主锚: C1 (观测 leader 组数钉死, full 场景 k=1) "
                    f"组系 fisher = {fish['C1']['mean']:.4f}±{fish['C1']['std']:.4f}° vs C0(R117 复跑) "
                    f"{fish['C0']['mean']:.4f}°; 显著回落 (-14.2°, 超 C0 2σ 带) 但未达预写恢复线 "
                    f"≤{po['threshold_max']}° 且 k_eff 仍 2–5 ∈/ [1,3] → 主断言 RED: 泄漏主通道 = "
                    "被保留原型的逐点浓度 κ_i 过弥散, 混合原型数只是放大器/下游症状"),
        "source_json": SOURCE,
        "source_path": "/scenarios/full/set/fisher_mean_direction_distance_deg/C1/mean",
        "lineage": lin, "status": "active-research", "verification": ver,
    }
    e2 = {
        "id": ids[1],
        "value": round(idm["C1"]["mean"], 4),
        "unit": "rel",
        "ci": [round(idm["C1"]["mean"] - idm["C1"]["std"], 4),
               round(idm["C1"]["mean"] + idm["C1"]["std"], 4)],
        "caliber": ("R121 因果检验锚: 钉 k=1 后 full 拓扑层 interdens 误差 = "
                    f"{idm['C1']['mean']:.4f}±{idm['C1']['std']:.4f} (C0 {idm['C0']['mean']:.4f} / "
                    f"B1 {idm['refs']['B1']['mean']:.4f}) → 预写『拓扑胜回摆向 B1』预期未发生, "
                    "R117『拓扑胜与组数过配同源』的表述精化为『同源=逐点方向过弥散 (κ_i 校准)』; "
                    "E3 条件机制输入从组数校准更新为浓度/不确定性校准"),
        "source_json": SOURCE,
        "source_path": "/scenarios/full/topology/intersection_density_rel_err/C1/mean",
        "lineage": lin, "status": "active-research", "verification": ver,
    }
    ledger["entries"] += [e1, e2]
    ledger.setdefault("changelog", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "R121 生成臂组数校准消融 · RED 双锚入册",
        "changes": [
            f"新增 {ids[0]}: C1 full fisher {fish['C1']['mean']:.4f}° (vs C0 {fish['C0']['mean']:.4f}°, "
            "-14.2° 超带未达 15° 恢复线, 主断言 RED)",
            f"新增 {ids[1]}: k=1 下 interdens {idm['C1']['mean']:.4f} (拓扑胜保留) —— "
            "R117 机制表述精化: 源=κ_i 逐点过弥散, 非混合原型数",
            "过程勘误: C0 二次单位化致落盘 sha 字节失配 (指标零影响), 修复重跑后 C1/C2 件与首跑"
            "逐字节一致 (确定性自证); tests/test_v_r121_.py 8 passed; R117 全家零触碰双测通过",
        ],
    })
    counts = ledger.get("counts", {})
    counts["active"] = counts.get("active", 0) + 2
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
    print(f"[r121_ledger] appended {ids}")


if __name__ == "__main__":
    main()
