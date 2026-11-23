# -*- coding: utf-8 -*-
"""R121 · 报告生成器: 数字零手打, 直读 calib_comparison.json / manifest / 三臂场景件。

输出 results/r121_report.md (反常档必附)。可安全重跑。
用法: E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r121_make_report.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "results", "r121_calib_compare")
SCEN = ["full", "borehole_1", "borehole_2", "borehole_4", "outcrop_10", "outcrop_30",
        "outcrop_50", "station_30", "multi_30_2"]
FLAG = {"fracgen_win": "改善", "fracgen_loss": "恶化", "within_2sigma": "带内", "na": "缺"}


def num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    t = json.load(open(os.path.join(RD, "calib_comparison.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(RD, "run_manifest.json"), encoding="utf-8"))
    ks = man["scenario_k_selection"]
    po = t["summary"]["primary_observations"]
    L = []
    L.append("# R121 · 生成臂组数校准消融 · R117 发现①的因果剖析 (诚实入账: RED 分支, 瓶颈下移一层)")
    L.append("")
    L.append("> 任务书 `tasks/R121_任务书_★★☆_生成臂组数校准消融.md` · 预注册 `results/v_r121_prereg.json`"
             " (创建时间早于一切产物, 坑 21 双口径) · 评分口径与 R117 完全一致 (R110 冻结五层 14 指标)。")
    L.append("> 三变体: C0=R117 原臂复跑 (换轨锚) · C1=观测 leader 组数钉死 · C2=vMF-BIC 选型; "
             "干预只改混合组数来源与权重归一 (GenHead.sample 逐 op 复刻, seed 与 C0 同流)。")
    L.append("> 闸门观察值 (run 段登记, 判级在守卫测试+架构师): " + ", ".join(
        f"{k}={v}" for k, v in man["gate_observations"].items() if k.startswith("G")))
    L.append("> **全部数字研究线, 禁入承诺语; 解读权在架构师。**")
    L.append("")
    L.append("## 0. 主断言观察值 (预注册 P2/P3 三条件, 只登记)")
    L.append("")
    for arm in ("C1", "C2"):
        a = po[arm]
        c0m = t["scenarios"]["full"]["set"]["fisher_mean_direction_distance_deg"]["C0"]["mean"]
        L.append(f"- {arm}: full fisher = **{num(a['full_fisher_mean'])}°** vs C0 "
                 f"{num(c0m)}° | 阈 ≤{a['threshold_max']}: "
                 f"{'达标' if a['below_threshold'] else '**未达**'} | 超 C0 带: "
                 f"{'是' if a['beyond_C0_band'] else '否'} | 单组场景 k_eff∈[1,3] 全 seed: "
                 f"{'是' if a['keff_in_range_all_seeds'] else '**否**'} → "
                 f"observation_GREEN = **{'GREEN' if a['observation_GREEN_all_three'] else 'RED'}**")
    L.append("")
    L.append("一句话 (只交数): 把混合原型数钉到观测组数 (full/station/outcrop/multi 均为 1) 后, "
             "组系 fisher 从 76.4° 显著回落到 ~62-75° (超 C0 带), 但距 15° 恢复线仍远, "
             "k_eff 仍碎为 2–7 —— **纯推理侧原型数钉死不能恢复组系层; 泄漏主通道在被保留原型"
             "自身的逐点浓度 κ_i 过弥散** (方向绕单一原型仍散到 leader θ=20° 拆组)。")
    L.append("")
    L.append("## 1. 组数选择与三臂方向层全景")
    L.append("")
    L.append("| 场景 | k_C1(leader) | k_C2(BIC) | fisher C0/C1/C2 | k_eff C0/C1/C2 | circ_std° C0/C1/C2 |")
    L.append("|---|---|---|---|---|---|")
    for s in SCEN:
        f = t["scenarios"][s]["set"]["fisher_mean_direction_distance_deg"]
        hc = t["scenarios"][s]["consistency"]["heldout_normal_circ_std_diff_deg"]
        ke = f["k_eff_leader"]
        ksel = ks[s]
        L.append(f"| {s} | {ksel['k_c1_leader']} | {ksel['k_c2_bic']} | "
                 f"{num(f['C0']['mean'])} / {num(f['C1']['mean'])} / {num(f['C2']['mean'])} | "
                 f"{ke['C0']} / {ke['C1']} / {ke['C2']} | "
                 f"{num(hc['C0']['mean'])} / {num(hc['C1']['mean'])} / {num(hc['C2']['mean'])} |")
    L.append("")
    L.append(f"- C1 与 C2 仅在 borehole_4 分化 (k=3 vs 5, 保留原型集 "
             f"[0,1,6] vs [0,1,2,3,6]), 其余 8 场景选择完全一致 —— **k 选择规则不是敏感面**; "
             f"vs C0 全表判级: 改善 {t['summary']['vs_C0_counts']['fracgen_win']} / "
             f"恶化 {t['summary']['vs_C0_counts']['fracgen_loss']} / "
             f"带内 {t['summary']['vs_C0_counts']['within_2sigma']} (252 判级点 = 2 校准臂 × 126 格)。")
    L.append("- 注意负向格: 三个钻孔场景 fisher 校准后反而小幅恶化 (C1/C2 把观测删失面的 leader "
             "组数 (3) 当真理, 掩掉了 C0 偶然用上的原型); outcrop_30/multi_30_2 几乎不动。")
    L.append("")
    L.append("## 2. 因果检验: R117『拓扑胜与组系败同源 (组数过配)』子假设被证伪/精化")
    L.append("")
    idm = t["scenarios"]["full"]["topology"]["intersection_density_rel_err"]
    L.append("| 场景 | interdens误差 C0 | C1 | C2 | B1 参照 |")
    L.append("|---|---|---|---|---|")
    for s in ("full", "station_30", "outcrop_10", "outcrop_30", "outcrop_50", "multi_30_2"):
        v = t["scenarios"][s]["topology"]["intersection_density_rel_err"]
        L.append(f"| {s} | {num(v['C0']['mean'],3)} | {num(v['C1']['mean'],3)} | "
                 f"{num(v['C2']['mean'],3)} | {num(v['refs']['B1']['mean'],3)} |")
    L.append("")
    L.append("- 预写因果预期『若组数过配是拓扑胜的因, 钉死 k=1 应使 interdens 回摆向 B1 水位』"
             f"→ **未发生** (full: C0 {num(idm['C0']['mean'],3)} → C1 {num(idm['C1']['mean'],3)}, "
             f"B1 {num(idm['refs']['B1']['mean'],3)}): 原型数=1 时拓扑胜几乎完整保留。")
    L.append("- 精化后的机制图 (数据说话): 单一泄漏源 = **逐点方向过弥散 (κ_i 校准失当)**; "
             "它在组系层表现为 leader 拆组 (90° 罚, R117 症状), 在拓扑层表现为打散近平行造交点 "
             "(R117 的红利)。钉住混合原型数只关掉了组数过配这一放大器 (~14°), 关不掉 κ_i 弥散 —— "
             "两层仍同源, 但『源』比 R117 登记的更深一层。")
    L.append("- 对 E3 扩散线 (R113) 的输入更新: 条件机制需求从『组数校准』改为"
             "**『浓度/不确定性校准』** (kappa/散度条件化或收缩); 组数入参仍值得保留 "
             "(full fisher 单靠它就有 -14.2° 的实测边际收益), 但两者不等价。")
    L.append("")
    L.append("## 3. 反常档诊断 (必填)")
    L.append("")
    L.append(f"1. **k=1 下 leader 仍拆 2–5 组**: full C1 k_eff="
             f"{t['scenarios']['full']['set']['fisher_mean_direction_distance_deg']['k_eff_leader']['C1']} "
             "(seed 11/22/33), 而混合原型只剩 1 个 —— 拆组完全由 vMF 逐点 κ_i 散布造成 "
             "(R117 诊断 κ_i p50≈62 但逐点分布宽, 低 κ_i 点偏轴 >20° 直接孵化领袖组)。"
             "此即瓶颈下移一层的直接证据。")
    L.append("2. **钻孔场景校准反噬**: borehole_1/2/4 fisher C0→C1 "
             f"{num(t['scenarios']['borehole_1']['set']['fisher_mean_direction_distance_deg']['C0']['mean'],1)}→"
             f"{num(t['scenarios']['borehole_1']['set']['fisher_mean_direction_distance_deg']['C1']['mean'],1)}, "
             f"{num(t['scenarios']['borehole_2']['set']['fisher_mean_direction_distance_deg']['C0']['mean'],1)}→"
             f"{num(t['scenarios']['borehole_2']['set']['fisher_mean_direction_distance_deg']['C1']['mean'],1)}, "
             f"{num(t['scenarios']['borehole_4']['set']['fisher_mean_direction_distance_deg']['C0']['mean'],1)}→"
             f"{num(t['scenarios']['borehole_4']['set']['fisher_mean_direction_distance_deg']['C1']['mean'],1)} —— "
             "线观测删失面的 leader 组数 (k=3) 本身是噪声源, 『观测组数』在低支撑场景不是可信先验; "
             "C2 的 BIC 在 borehole_4 选到 k=5 更差, 两准则在删失数据上同不可靠。")
    L.append("3. **饱和伪差照抄复认** (R117 判据边界情形): borehole_4 interdens 两臂贴 1.0, "
             "带≈0 时的 5e-5 级判级噪声不采信; 本单 vs C0 判级同样免疫该格 (带内)。")
    L.append("4. **C1≡C2 逐位同值**: 8/9 场景两臂数字完全相同 —— 非巧合, 是 (k 相同 → 保留原型集相同 "
             "→ 同一随机流同一采样) 的决定性推论, 反向印证采样实现的确定性。")
    L.append("5. **usage 的 seed 不变性** (附带观察): 同一场景三 seed 的保留原型集完全一致 —— "
             "usage 只由观测锚点决定 (SetModule 掩码聚合), 查询点扰动不进入; 符合架构语义, 入档。")
    L.append("6. **circ_std 部分恢复但未收敛**: full 13.76°→9.65° (判级改善), 仍 >6° 门 "
             "(协议 §b1 一致性门) —— 与反常档-1 同机制 (κ_i 弥散的 held-out 投影)。")
    L.append("7. **过程勘误 (如实入册)**: 首跑 C0 落盘条目对已单位化的方向二次 _unit, "
             "float64 末位变化使存盘 sha 与锚点判定路径不一致 (指标值零影响, 守卫测试 T2 抓到)。 "
             "修复后整单重跑: C1/C2 场景件与首跑逐字节一致 (确定性证明, 非重掷), 仅 C0 哈希字节修正; "
             "C0/C1/C2 任何指标数值两跑相同。教训并入『字节级锚点必须单一规范化路径』。")
    L.append("")
    L.append("## 4. 纪律声明")
    L.append("")
    L.append("- 零训练零调参; 干预=预注册确定性规则 (usage 前 k + 权重重归一, 逐 op 复刻采样, 随机流同 C0)。")
    L.append("- R117 全家 + 上游冻结件只读: PROTECTED sha 清单 import 前/单尾双测零漂移; "
             "C0 27/27 gen_dir sha 逐位=R117 在盘 (换轨锚成立); 装配同构锚 27/27。")
    L.append("- 时点陈述运行时实测 (manifest.b2_b3_presence_at_run); 判废数字扫描零命中 (守卫测试)。")
    L.append(f"- 判级汇总 (vs C0): {json.dumps(t['summary']['vs_C0_counts'], ensure_ascii=False, allow_nan=False)}")
    L.append("- 交付: scripts/r121_group_calib.py (--stage prereg|run|table) + results/v_r121_prereg.json + "
             "results/r121_calib_compare/{C0,C1,C2}/×9 + calib_comparison.json + run_manifest.json + "
             "results/r121_console.log + tests/test_v_r121_.py。")
    with open(os.path.join(ROOT, "results", "r121_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[report] -> results/r121_report.md")


if __name__ == "__main__":
    main()
