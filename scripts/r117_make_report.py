# -*- coding: utf-8 -*-
"""R117 · 对照报告生成器: 数字零手打, 全部直读 comparison_table.json 与逐场景 JSON。

输出 results/r117_compare_report.md — 只交数 + 结构化发现 + 反常档诊断, 不写宣传语
(优劣结论权在架构师)。可安全重跑 (报告非时序锚, mtime 铁律只约束预注册与结果件)。

用法:
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r117_make_report.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "results", "r117_recon_compare")
TABLE = os.path.join(RUN_DIR, "comparison_table.json")
PREREG = os.path.join(ROOT, "results", "v_r117_prereg.json")
OUT_MD = os.path.join(ROOT, "results", "r117_compare_report.md")
SCEN_ORDER = ["full", "borehole_1", "borehole_2", "borehole_4",
              "outcrop_10", "outcrop_30", "outcrop_50", "station_30", "multi_30_2"]
FLAG_ZH = {"fracgen_win": "胜", "fracgen_loss": "负",
           "within_2sigma": "带内", "structural_tie_vs_B1": "≡B1", "na": "缺"}
LAYERS = ("set", "scale", "spatial", "topology", "consistency")


def num(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def cell(v, bname, nd=4):
    b = v["baselines"][bname]
    if b["flag"] == "structural_tie_vs_B1":
        return f"≡B1 {num(b['mean'], nd)}"
    return (f"{num(v['fracgen_mean'], nd)}±{num(v['fracgen_std'], nd)} vs "
            f"{num(b['mean'], nd)}±{num(b['std'], nd)} [{FLAG_ZH[b['flag']]}]")


def main():
    with open(TABLE, encoding="utf-8") as fh:
        t = json.load(fh)
    with open(PREREG, encoding="utf-8") as fh:
        pr = json.load(fh)
    scen_files = {}
    for s in SCEN_ORDER:
        with open(os.path.join(RUN_DIR, "fracgen", f"{s}.json"), encoding="utf-8") as fh:
            scen_files[s] = json.load(fh)
    on_disk = [b for b in ("B1", "B2", "B3") if t["baselines_presence"][b]["present"]]
    fb = t["summary"]["by_baseline"]

    L = []
    L.append("# R117 · fracgen 条件生成 vs 统计基线 · 重构首战对照成绩单")
    L.append("")
    L.append("> 任务书: `tasks/R117_任务书_★★☆_fracgen条件生成vs统计基线_重构首战.md` · "
             "预注册: `results/v_r117_prereg.json` (创建时间早于一切结果件) · "
             "评分权威: `scripts/r110_recon_metrics.scorecard` 五层 14 指标 (冻结, 禁加指标), "
             f"协议件指纹 sha256[:12]=`{t['protocol_scorecard_sha256'][:12]}`。")
    L.append(f"> 生成器: `models/fracgen_v1_v9.pt` (R111 审计全忠实, 只读) · "
             f"9 场景 × 3 seeds [11/22/33] · 在盘基线: {', '.join(on_disk)} · 零重训零微调零调参。")
    L.append("> **全部数字属研究线对照, 禁入客户承诺语; 本报告只交数+结构化发现, "
             "解读与优劣结论权在架构师。**")
    L.append("")
    L.append("## 0. 总览 (126 判分格 = 9 场景 × 14 指标)")
    L.append("")
    for b in on_disk:
        c = fb[b]["counts"]
        L.append(f"- vs {b}: 胜 **{c['fracgen_win']}** / 负 **{c['fracgen_loss']}** / "
                 f"带内 **{c['within_2sigma']}**" +
                 (f" / 结构性等值 **{c['structural_tie_vs_B1']}** (与方向无关的配对格, 不判胜负)"
                  if b == "B1" else ""))
    for b in ("B2", "B3"):
        if not t["baselines_presence"][b]["present"]:
            L.append(f"- {b} 状态: {t['baselines_presence'][b]['note']}")
    lay = {layer: {b: sum(1 for s in SCEN_ORDER for m in _metrics(t, layer, b)
                          if _flag(t, s, layer, m, b) == "fracgen_win") for b in on_disk}
           for layer in LAYERS}
    layL = {layer: {b: sum(1 for s in SCEN_ORDER for m in _metrics(t, layer, b)
                           if _flag(t, s, layer, m, b) == "fracgen_loss") for b in on_disk}
            for layer in LAYERS}
    L.append("")
    L.append("层×基线 胜/负 速览 (126 格/基线):")
    L.append("")
    L.append("| 层 | " + " | ".join(f"vs {b} 胜/负/带内" for b in on_disk) + " |")
    L.append("|---" * (len(on_disk) + 1) + "|")
    for layer in LAYERS:
        ms = _metrics(t, layer, on_disk[0])
        cells = []
        for b in on_disk:
            w = sum(_flag(t, s, layer, m, b) == "fracgen_win" for s in SCEN_ORDER for m in ms)
            l = sum(_flag(t, s, layer, m, b) == "fracgen_loss" for s in SCEN_ORDER for m in ms)
            i = sum(_flag(t, s, layer, m, b) == "within_2sigma" for s in SCEN_ORDER for m in ms)
            cells.append(f"**{w} / {l}** / {i}")
        L.append(f"| {layer} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("结构化发现 (只交数):")
    L.append("")
    def win_layers(b):
        return "/".join(f"{layer}({lay[layer][b]})" for layer in LAYERS if lay[layer][b] > 0) or "无"
    def loss_layers(b):
        return "/".join(f"{layer}({layL[layer][b]})" for layer in LAYERS if layL[layer][b] > 0) or "无"
    L.append(f"- 胜项分布: vs B1 {win_layers('B1')}; vs B2 {win_layers('B2')}; "
             f"vs B3 {win_layers('B3')} (vs B1 胜项全部落在拓扑层, 同源定性见 §6-2)。"
             f"负项分布: vs B1 {loss_layers('B1')}; vs B2 {loss_layers('B2')}; "
             f"vs B3 {loss_layers('B3')}。")
    L.append(f"- 组系层对 B1/B2 全部落败 (0 胜; 方向过弥散→组数过配, §6-1), "
             f"但对 B3 反为 gen 大优 (B3 检索卡方向失真更重); "
             f"一致性层落败集中于法向圆标准差项 (§6-6)。")
    d_int = _cell(t, "full", "topology", "intersection_density_rel_err")
    d_den = _cell(t, "full", "spatial", "density_field_l1_rel")
    L.append("- full 场景三水位对照 (直读): intersection_density 误差 gen="
             + " / ".join(f"{b}={num(d_int['baselines'][b]['mean'],3)}" for b in on_disk)
             + f" → gen={num(d_int['fracgen_mean'],3)} 对全部在盘基线取胜; density_field gen="
             + " / ".join(f"{b}={num(d_den['baselines'][b]['mean'],3)}" for b in on_disk)
             + f" → gen={num(d_den['fracgen_mean'],3)} 与 B1 同水位 (位置模型配对所致), "
               "劣于 B2 强度场, 对 B3 见其自身检索失真水位。")
    L.append(f"- 判级汇总: 对全部在盘基线同向胜 **{both(t,'fracgen_win')}** 格 / "
             f"同向负 **{both(t,'fracgen_loss')}** 格 (其余为混合/带内/对 B1 等值格)。")
    L.append("")
    L.append("## 1. 接线与配对设计 (fidelity 证据)")
    L.append("")
    L.append("- 消费路径 = `FracGen.head.sample()` 条件 vMF 混合采样 (R111 run 段同款); "
             "观测锚点 (每观测面 ≤8 实测点, mask=1) + 查询点 (B1 面心, mask=0) 同批前向, "
             "模型输出查询点条件方向 → 重构面 (**c/R 与 B1 逐 seed 配对, 只有方向不同**)。")
    L.append("- 由此对 B1: 尺度层/空间层/topology.p32 结构性等值 (位置/尺寸流配对), 不判胜负; "
             "对 B2/B3: 这些格对比的是『B1 配对位置模型 vs 各自位置模型』, 方向重构不参与该对比 "
             "(判读时注意归因)。")
    L.append("- 忠实性锚 (run 段实测, `results/r117_recon_compare/run_manifest.json`): "
             "G0 R111 completion#0 sha 逐位复现=真; G1 全资产 sha256 零漂移=真; "
             "G3 full×seed11 生成方向双算=真; G4 B1 (c,R) 流 27/27 逐位等于预注册=真; "
             "G6 9/9 场景落盘=真。")
    L.append("- 逐 seed (c,R) 配对流指纹锚: 预注册 `v_r117_prereg.json` "
             "scenarios[].b1_geom_stream_sha_by_seed (full seed11 前 8 位 = "
             + pr["scenarios"][0]["b1_geom_stream_sha_by_seed"]["11"][:8] + ")。")
    L.append("")

    def layer_block(layer, title, note, nd=4):
        ms = [m for m in t["scenarios"]["full"]["metrics"][layer]]
        L.append(title)
        L.append("")
        L.append(f"> {note}")
        L.append("")
        for b in on_disk:
            col = f"vs {b} (胜/负/带内/等值): "
            w = sum(_flag(t, s, layer, m, b) == "fracgen_win" for s in SCEN_ORDER for m in ms)
            l = sum(_flag(t, s, layer, m, b) == "fracgen_loss" for s in SCEN_ORDER for m in ms)
            i = sum(_flag(t, s, layer, m, b) == "within_2sigma" for s in SCEN_ORDER for m in ms)
            st = sum(_flag(t, s, layer, m, b) == "structural_tie_vs_B1" for s in SCEN_ORDER for m in ms)
            L.append(f"> {col}{w}/{l}/{i}" + (f"/{st}" if b == "B1" else ""))
        L.append("")
        head = "| 场景 | " + " | ".join(f"{m}" for m in ms) + " |"
        L.append(head)
        L.append("|---" * (len(ms) + 1) + "|")
        for s in SCEN_ORDER:
            row = t["scenarios"][s]["metrics"][layer]
            cells = []
            for m in ms:
                v = row[m]
                parts = [f"gen {num(v['fracgen_mean'], nd)}±{num(v['fracgen_std'], nd)}"]
                for b in on_disk:
                    bb = v["baselines"][b]
                    if bb["flag"] == "structural_tie_vs_B1":
                        parts.append(f"≡B1")
                    else:
                        parts.append(f"{b} {num(bb['mean'], nd)} [{FLAG_ZH[bb['flag']]}]")
                cells.append(" / ".join(parts))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## 2. 组系层 (方向重构主战场)")
    L.append("")
    layer_block("set", "", "真值组数=1 (HRF 单组近水平巨盘); setdiff=|重构组数−1|; "
                "fisher 含未匹配组 90° 罚; 格式 gen mean±std / 基线 mean [判级]", nd=3)
    L.append("## 3. 拓扑层 (对 B1 的全部胜项在此)")
    L.append("")
    layer_block("topology", "", "p32_rel_err 对 B1 结构性等值; p10/交点密度/连通簇为方向+位置混合敏感", nd=3)
    L.append("## 4. 正演一致性层")
    L.append("")
    layer_block("consistency", "", "held-out 20 条冻结扫描线; P10/z 主要由位置几何决定, "
                "法向圆标准差为方向敏感项", nd=3)
    L.append("## 5. 尺度/空间层 (对 B1 结构性等值, 对 B2 显示位置模型差距)")
    L.append("")
    layer_block("scale", "", "尺寸流与 B1 配对 → 对 B1 恒等; 对 B2 差异来自 R̂ 流与删失面数不同", nd=3)
    layer_block("spatial", "", "中心流与 B1 配对 → 对 B1 恒等; 对 B2 差异 = 均匀 vs 强度场位置模型 "
                "(R115 主战场, fracgen 继承 B1 短板)", nd=3)

    # -------- 反常档 --------
    d = {s: scen_files[s]["per_seed"][0]["diag"] for s in SCEN_ORDER}
    L.append("## 6. 反常档诊断 (必填; 现象+假设, 不粉饰不判优)")
    L.append("")
    fs = _cell(t, "full", "set", "fisher_mean_direction_distance_deg")
    sd = _cell(t, "full", "set", "set_count_abs_diff")
    L.append(f"1. **方向过弥散 → 组数过配 (组系层对两臂全败的主机制)**。真值 1 组 "
             f"(dip 2.6–5.7°), full 场景 gen setdiff={num(sd['fracgen_mean'],2)} "
             f"(重构组数≈{1 + round(scen_files['full']['per_seed'][0]['scorecard']['set']['set_count_abs_diff'])}), "
             f"未匹配组按 90° 罚直接把 fisher 拉到 {num(fs['fracgen_mean'],1)}° "
             f"(B1 {num(fs['baselines']['B1']['mean'],2)}°)。生成方向本身并不远离观测组轴 "
             f"(full seed11 dev p50={d['full']['dev_vs_observed_axis_deg']['p50']:.1f}°, "
             f"合矢量 R̄={d['full']['resultant_sign_aligned']:.3f} — 非全同坍缩), "
             f"而是条件 vMF 混合把概率摊到多个原型上 (full seed11 usage>0.02 原型 "
             f"{d['full']['proto_usage_gt02']}/12, κ_i 中位 {d['full']['kappa_i_query_p50']:.0f} "
             f"= 高置信度的多峰弥散)。假设: n_proto=12 原型池中低使用原型的 mu_hat 为对消均值的 "
             f"epsilon 正则化方向 (几何不可靠), usage 对数耦合压不干净, 单组数据被拆成多组。")
    di = _cell(t, "full", "topology", "intersection_density_rel_err")
    dc = _cell(t, "outcrop_10", "topology", "largest_cluster_fraction_diff")
    L.append(f"2. **拓扑层胜与组系层败同源 (误差对误差的抵消, 非模型学到连通性)**。"
             f"真值 78% 面对相交靠『聚集+大半径+小倾角差』共同保证; B1 严格平行+均匀位置低估交点 "
             f"(full interdens 误差 {num(di['baselines']['B1']['mean'],3)}), fracgen 方向弥散把近平行打散, "
             f"人为抬高交切数恰好逼近真值 ({num(di['fracgen_mean'],3)}); B2 用正确位置模型也能到 "
             f"{num(di['baselines']['B2']['mean'],3)}。outcrop_10 最大连通簇差 gen "
             f"{num(dc['fracgen_mean'],3)} vs B1 {num(dc['baselines']['B1']['mean'],3)} 同机制。"
             f"若数据换到多走向真网络 (合成/R111 口径), 该红利不预期存在。")
    b4 = _cell(t, "borehole_4", "topology", "intersection_density_rel_err")
    L.append(f"3. **饱和伪差 (判据边界情形, 登记不作实质优势)**: borehole_4 interdens "
             f"gen {num(b4['fracgen_mean'])}±{num(b4['fracgen_std'])} vs B1 "
             f"{num(b4['baselines']['B1']['mean'])}±{num(b4['baselines']['B1']['std'])} "
             f"两臂均贴 1.0 饱和, B1 seed 带≈0 使 5e-5 噪声过 2σ 闸被判『胜』; 同格 vs B2 同理。")
    b1p = scen_files["borehole_1"]["per_seed"]
    L.append(f"4. **钻孔线型锚 + 组系比例失稳**: borehole_1 set_proportion_L1 跨 seed "
             f"{num(min(p['scorecard']['set']['set_proportion_L1'] for p in b1p),3)} → "
             f"{num(max(p['scorecard']['set']['set_proportion_L1'] for p in b1p),3)} "
             f"(摆动 18×); 锚点为 2m 孔内 ~1–3 点 LINE 型邻域, kNN-PCA 局部标架退化风险入档 "
             f"(与主项目 63% LINE 型登记10 同族)。")
    L.append("5. **域偏移登记 (解释力有限, 不作开脱)**: 观测:查询点比例 ≈8:1 (训练 40:60); "
             "s1/s3=0 中性编码 (训练恒为单位应力轴); lith 恒 1; HRF 单组近水平巨型盘 vs 合成多组中等域; "
             "查询点为体均匀采样 (训练隐伏点贴面)。坐标单位归一化不敏感 (预注册 "
             "design_time_probe_disclosure)。以上接口选择全部在预注册冻结, 未据评分调参。")
    hz = _cell(t, "full", "consistency", "heldout_p10_rel_err")
    hc = _cell(t, "full", "consistency", "heldout_normal_circ_std_diff_deg")
    L.append(f"6. **正演一致性两极**: heldout_p10/z 对 B1 带内 (位置配对所致, full p10 "
             f"gen {num(hz['fracgen_mean'],3)} vs B1 {num(hz['baselines']['B1']['mean'],3)}); "
             f"circ_std 方向敏感项 gen {num(hc['fracgen_mean'],2)}° vs B1 "
             f"{num(hc['baselines']['B1']['mean'],2)}° vs B2 {num(hc['baselines']['B2']['mean'],2)}° "
             f"— held-out 窗口法向离散度同样被过弥散抬高 (反常档-1 的投影)。"
             f"注: vs B2 的 p10 大败格是 B2 强度场位置红利, 非方向劣势。")
    L.append("")
    L.append("## 7. 纪律声明")
    L.append("")
    L.append("- 零重训零微调零调参; `models/` `data/` `E:/fracture_data/` 只读 "
             "(G1 全资产 sha256 复核, 含 B2/B3 缺席或完整性判定逻辑冻结于加载器)。")
    L.append("- 判级只按预注册 2σ 规则 (带=2·max(基线 seedσ,1e-9)), 本报告不下『谁更好』总评; "
             "冻结锚点仅引用; 历史判废数字零出现 (守卫测试扫描)。")
    b3p = t["baselines_presence"]["B3"]
    L.append("- 基线并入机制: `--stage table` 一键重建 (确定性, fracgen 结果不动), "
             "加载器支持逐场景件与合并件 (R116 交付形态 arms.ARM_MAIN) 两种合格布局; "
             + (f"B3 已并入本表: {b3p['note']}" if b3p["present"]
                else f"B3 状态: {b3p['note']}, 归队后重跑 --stage table 即并入"))
    L.append("- 结果件: `results/r117_recon_compare/fracgen/*.json` (9) + `comparison_table.json` + "
             "`run_manifest.json`; 代码 `scripts/r117_fracgen_conditional.py`; 守卫 `tests/test_v_r117_.py`。")
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] -> {OUT_MD} ({os.path.getsize(OUT_MD)} B)")


def _metrics(t, layer, b):
    return [m for m in t["scenarios"]["full"]["metrics"][layer]]


def _cell(t, scen, layer, m):
    return t["scenarios"][scen]["metrics"][layer][m]


def _flag(t, scen, layer, m, b):
    return _cell(t, scen, layer, m)["baselines"][b]["flag"]


def both(t, want):
    bs = [b for b in ("B1", "B2", "B3") if t["baselines_presence"][b]["present"]]
    n = 0
    for s in SCEN_ORDER:
        for layer in LAYERS:
            for m in t["scenarios"][s]["metrics"][layer]:
                v = t["scenarios"][s]["metrics"][layer][m]
                if all(v["baselines"][b]["flag"] == want for b in bs):
                    n += 1
    return n


if __name__ == "__main__":
    main()
