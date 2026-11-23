# -*- coding: utf-8 -*-
"""论文图表包生成脚本 (十三期 T88 重制版, 图账同源制度化).

生成 6 张期刊规范 Figure, 输出 docs/论文/figures/.

== S28-AUDIT ==
本脚本内【禁止出现任何未申报的浮点字面量】: 所有展示用精度数字一律从
results/*.json 直读 (数字变动 -> 重跑本脚本即同步, 根治"图与台账漂移");
纯布局常数 (figsize/alpha/线宽等) 必须在下方 ALLOWED_LAYOUT_FLOATS 申报,
由 src/fractureflow/selfcheck.py::chk_s28_figure_script_numbers 机检。

Figure 列表与数据源 (全部落盘真数据):
  Fig1_data_ladder     — 数据阶梯示意图      <- pointcloud_gate / trace_network_aland /
                                                t58_b2b3_forge_rebuild / l4_evidence_grading
  Fig2_mar_vs_ladder   — 角度误差 vs 数据档位 <- honest_leaderboard / p1_honest_decoder /
                                                phase0_oracle_floor_strict / t58_b2b3_forge_rebuild /
                                                pointcloud_gate / l4_evidence_grading
  Fig3_triple_source   — 三源组系表对比       <- set_table_decovalex / set_table_beishan /
                                                t58_b2b3_forge_rebuild (DEP-006 修复后值)
  Fig4_pointcloud_pipe — L3 管线评测          <- pointcloud_gate
  Fig5_blindinput      — 协议流程示意图       (无数字)
  Fig6_multiwell_audit — 多井联合适用性审计    <- multiwell_joint_audit (真实 231 井对角距 +
                                                22 井判定, 不再使用任何模拟数据)
"""

import os
import sys
import json

import numpy as np

# === S28-AUDIT: 布局浮点常数白名单 (唯一允许出现浮点字面量的位置) ===
# 注意: 必须保持纯集合字面量写法 (selfcheck 用 ast.literal_eval 提取)
ALLOWED_LAYOUT_FLOATS = {
    # figsize 小数
    "4.5", "3.5", "3.6", "6.5", "4.2",
    # 线宽
    "2.5",
    # fig1 阶梯几何
    "6.0", "4.8", "2.4", "1.2", "0.8", "1.5", "0.2", "0.5", "5.5", "9.3", "3.2", "7.5",
    # fig2 线宽/透明度/边距
    "0.7", "1.2", "0.5", "0.3",
    # fig3 误差棒与轴余量
    "0.03", "1.25",
    # fig4 散点透明度
    "0.6",
    # fig5 流程框坐标
    "0.4", "1.4", "0.1", "2.2", "2.8", "4.7", "5.3", "2.6", "7.2", "7.8",
    "4.5", "4.0", "0.02",
    # fig6 色条/文本变换/轴余量/图注位置
    "0.16", "0.06", "0.05", "0.12", "0.01", "0.97", "0.98",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "论文", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

_RES = os.path.join(ROOT, "results")


def _load(rel_path):
    with open(os.path.join(_RES, rel_path), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 数据绑定 (唯一事实来源 = results 落盘 JSON)
# ---------------------------------------------------------------------------
LB_L1LOCAL = _load("honest_leaderboard/l1_local__beishan_22.json")          # NNS-001/026
DECODER_CMP = _load("p1_honest_decoder.json")                               # kmeans_obs 等
ORACLE_FLOOR = _load("phase0_oracle_floor_strict.json")                     # oracle (非可部署)
L4_GRADE = _load("l4_evidence_grading.json")                                # NNS-045 信息墙
SET_BEISHAN = _load("set_table_beishan.json")                               # NNS-017
SET_DECOVALEX = _load("set_table_decovalex.json")                           # NNS-018
FORGE_REBUILD = _load("t58_b2b3_forge_rebuild.json")                        # DEP-006/NNS-019
PC_GATE = _load("pointcloud_gate.json")                                     # NNS-021
ALAND = _load("trace_network/trace_network_aland.json")                     # T32 确定性统计
MW_AUDIT = _load("multiwell_joint_audit.json")                              # T34 多井审计
P32_FULL = _load("dfn_pipeline_beishan_full.json")                          # 渗流曲线 (口径一致 6/6)
DFN_FINAL = _load("dfn_pipeline_final.json")                                # 20实现精化值 (曲线数组有缺陷, 仅引用其数字)
P0_AUDIT = _load("p0_full_audit.json")                                      # 泄漏 vs 诚实 K 网格

# 常用取值 (全部源自上方 JSON, 此处只做引用不做运算造数)
V_L1LOCAL = LB_L1LOCAL["mae_mean"]
V_KMEANS_OBS = DECODER_CMP["results"]["kmeans_obs"]["mean"]
V_KMEANS_OBS_STD = DECODER_CMP["results"]["kmeans_obs"]["std"]
_ORA_BEISHAN = ORACLE_FLOOR["results"]["beishan_22"]["12"]
V_ORACLE = _ORA_BEISHAN["mae_mean"]
V_ORACLE_STD = _ORA_BEISHAN["mae_std"]
V_CEIL = L4_GRADE["threshold_sources"]["information_ceiling_31_7deg"]["value"]
V_THRESH_12 = L4_GRADE["threshold_sources"]["grade_C_center_12deg"]["value"]
V_SET_BEISHAN = SET_BEISHAN["12"]["modal_err_mean"]
V_SET_BEISHAN_STD = SET_BEISHAN["12"]["modal_err_std"]
V_SET_DECO = SET_DECOVALEX["4"]["modal_err_mean"]
V_SET_DECO_STD = SET_DECOVALEX["4"]["modal_err_std"]
_FORGE_NEW = FORGE_REBUILD["new_numbers"]["K12"]
V_FORGE_NEW = _FORGE_NEW["mean"]
V_FORGE_NEW_STD = _FORGE_NEW["std"]
V_PC_MAE = PC_GATE["evaluation"]["mae_hidden_deg"]
V_PC_P50 = PC_GATE["evaluation"]["p50_hidden_deg"]
V_PC_P90 = PC_GATE["evaluation"]["p90_hidden_deg"]
V_PC_MISCLASS = PC_GATE["evaluation"]["misclass_rate_hidden"]
V_ALAND_BETA = ALAND["fit_beta"]["beta"]
V_ALAND_P21 = ALAND["P21_m_per_m2"]

# 全局期刊规范 + 中文字体 (Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

_cn_fonts = [f.name for f in fm.fontManager.ttflist if "Hei" in f.name or "YaHei" in f.name
             or "SimHei" in f.name or "SimSun" in f.name or "Song" in f.name
             or "Kai" in f.name or "Ming" in f.name or "STHeiti" in f.name
             or "PingFang" in f.name or "Source Han" in f.name]
if _cn_fonts:
    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["font.sans-serif"] = [_cn_fonts[0], "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def _fmt(v, nd=2):
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Figure 1: 数据阶梯示意图
# ---------------------------------------------------------------------------
def fig1_data_ladder():
    fig, ax = plt.subplots(figsize=(8, 5))

    levels = [
        ("L4\n融合数字孪生", 6.0, "#2ecc71", "待试点数据"),
        ("L3\n稠密 3D 点云", 4.8, "#27ae60", f"{_fmt(V_PC_MAE)}°"),
        ("L2\n露头 2D 迹线", 3.6, "#3498db",
         f"β={_fmt(V_ALAND_BETA)}\nP21={_fmt(V_ALAND_P21, 1)}"),
        ("L1\nFMI 井壁迹线", 2.4, "#9b59b6", f"组系表 {_fmt(V_FORGE_NEW)}°"),
        ("L0\n钻孔编录表", 1.2, "#e74c3c", f"信息墙 ~{_fmt(V_CEIL, 1)}°"),
    ]

    for i, (label, y, color, note) in enumerate(levels):
        ax.barh(y, 6, height=0.8, left=1, color=color, alpha=0.7,
                edgecolor="black", linewidth=0.8)
        ax.text(4, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        ax.text(7.5, y, note, ha="left", va="center", fontsize=9,
                color=color, fontweight="bold")

    ax.annotate(f"信息墙\n~{_fmt(V_CEIL, 1)}°", xy=(1, 1.2), xytext=(0.2, 0.5),
                fontsize=9, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

    ax.annotate("", xy=(9, 5.5), xytext=(9, 0.8),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2, linestyle="--"))
    ax.text(9.3, 3.2, "升维", fontsize=10, color="gray", rotation=90, va="center")

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("信息量 →")
    ax.set_title("Figure 1. 数据阶梯框架 (Data Ladder): 精度由数据档位决定, 非算法复杂度")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(bottom=0.2)

    fig.text(
        0.5, 0.01,
        "Fig.1  Data Ladder framework. Annotations are measured anchors from committed JSONs: "
        f"L3 gate {_fmt(V_PC_MAE)} deg [NNS-021]; L2 deterministic stats (beta/P21, not angle "
        f"error); L1 set-table modal err {_fmt(V_FORGE_NEW)} deg [NNS-019, post pole-convention "
        f"fix DEP-006]; L0 information ceiling ~{_fmt(V_CEIL, 1)} deg [NNS-045].",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "Fig1_data_ladder.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Figure 2: 角度误差 vs 数据档位 (统一角度量纲; 口径见图例)
# ---------------------------------------------------------------------------
def fig2_mar_vs_ladder():
    fig, ax = plt.subplots(figsize=(7, 5))

    # 天花板参考虚线 (不占散点)
    ax.axhline(y=V_CEIL, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"L0 信息墙 ~{_fmt(V_CEIL, 1)}° [NNS-045]")

    # 散点: (x, y, yerr|None, 标注, 样式)
    points = [
        (0, V_L1LOCAL, None, f"l1_local 无标签 {_fmt(V_L1LOCAL, 1)}°\n[诚实榜 NNS-001]",
         dict(s=100, color="#3498db")),
        (0, V_KMEANS_OBS, V_KMEANS_OBS_STD,
         f"路线A kmeans_obs K=6 {_fmt(V_KMEANS_OBS, 1)}°\n[strict 口径]",
         dict(s=100, color="#8e44ad")),
        (0, V_ORACLE, V_ORACLE_STD,
         f"Oracle 地板 {_fmt(V_ORACLE, 1)}°\n[非可部署]",
         dict(s=80, marker="s", color="#f39c12")),
        (1, V_FORGE_NEW, V_FORGE_NEW_STD,
         f"L1 组系表 modal_err {_fmt(V_FORGE_NEW, 1)}°\n[NNS-019, DEP-006 修复后]",
         dict(s=100, color="#9b59b6")),
        (3, V_PC_MAE, None,
         f"L3 平面法向 MAE {_fmt(V_PC_MAE)}°\n[NNS-021 合成闸门]",
         dict(s=100, color="#27ae60")),
    ]

    for x, y, yerr, label, style in points:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="gray",
                    elinewidth=1.2, capsize=4) if yerr else None
        ax.scatter(x, y, zorder=5, **style)
        # 最右侧散点标注向左偏移, 防止超出画布裁切
        dx, dy = (-150, 10) if x >= 3 else (10, 5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8)

    ax.annotate("升维\n(换档)", xy=(3, V_PC_MAE), xytext=(2, 15),
                fontsize=9, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=2))

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(0, 45)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["L0\n编录表", "L1\nFMI迹线", "L2\n露头迹线", "L3\n稠密点云", "L4\n数字孪生"])
    ax.set_ylabel("角度误差 (°)")
    ax.set_title("Figure 2. 重构精度 vs 数据档位 (各散点口径见图例标注)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5, 0.01,
        f"Fig.2  Angle error vs data level (single angular unit; caliber per point in legend). "
        f"L2 Aland traces yield deterministic statistics only (beta={_fmt(V_ALAND_BETA)}, "
        f"P21={_fmt(V_ALAND_P21, 1)} m/m², not angle error) hence omitted from scatter. "
        f"Values auto-loaded from committed results JSONs.",
        ha="center", fontsize=8, style="italic")

    out = os.path.join(OUT_DIR, "Fig2_mar_vs_ladder.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Figure 3: 三源组系表 modal_err 对比 (FORGE 用 DEP-006 修复后值)
# ---------------------------------------------------------------------------
def fig3_triple_source():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    sources = ["DECOVALEX\n(K=4)", "FORGE\n(K=12)", "beishan\n(K=12)"]
    modal_errs = [V_SET_DECO, V_FORGE_NEW, V_SET_BEISHAN]
    stds = [V_SET_DECO_STD, V_FORGE_NEW_STD, V_SET_BEISHAN_STD]
    colors = ["#27ae60", "#3498db", "#9b59b6"]

    bars = ax.bar(sources, modal_errs, yerr=stds, capsize=5, color=colors,
                  alpha=0.8, edgecolor="black", linewidth=0.8,
                  error_kw={"linewidth": 1.5})

    ymax = max(m + s for m, s in zip(modal_errs, stds))
    for bar, val, std in zip(bars, modal_errs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + ymax * 0.03,
                f"{val:.2f}°", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=V_THRESH_12, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"{_fmt(V_THRESH_12)}° 阈值")

    ax.set_ylabel("modal_err (°)")
    ax.set_title("Figure 3. 三源组系表验证 (全部 <12° 阈值)")
    ax.legend()
    ax.set_ylim(0, max(15, ymax * 1.25))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.text(
        0.5, 0.01,
        "Fig.3  Triple-source set-table validation (all pass the 12 deg threshold). "
        "FORGE bar revised upward after pole-convention correction (DEP-006; see paper "
        "appendix leakage audit); error bars = 1 std across seeds. Sources: "
        "set_table_decovalex.json / t58_b2b3_forge_rebuild.json / set_table_beishan.json.",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "Fig3_triple_source.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Figure 4: L3 管线评测 (数值全部来自 pointcloud_gate.json)
# ---------------------------------------------------------------------------
def fig4_pointcloud_pipe():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    rng = np.random.default_rng(42)
    K = PC_GATE["synth_config"]["K"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for k in range(K):
        n = 200  # 降采样显示
        u1 = rng.uniform(-4, 4, n)
        u2 = rng.uniform(-4, 4, n)
        offset = k * 6  # 空间分离
        axes[0].scatter(u1 + offset, u2, c=colors[k], s=1, alpha=0.6,
                        label=f"Group {k + 1}")
    axes[0].set_title("(a) 合成多壁点云示意 (K 由 JSON 配置)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].legend(fontsize=7, markerscale=5)
    axes[0].set_aspect("equal")

    ev = PC_GATE["evaluation"]
    metrics = ["MAE", "P50", "P90"]
    values = [ev["mae_hidden_deg"], ev["p50_hidden_deg"], ev["p90_hidden_deg"]]
    bars = axes[1].bar(metrics, values, color=["#27ae60", "#2ecc71", "#3498db"],
                       edgecolor="black", linewidth=0.8, alpha=0.8)
    vtop = max(values)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + vtop * 0.03,
                     f"{val:.3f}°", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Angle error (°)")
    axes[1].set_title("(b) 隐伏点评测 (obs_frac 由 JSON)")
    axes[1].set_ylim(0, vtop * 1.25)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 4. L3 稠密点云免标签重构管线 (T30 多走向闸门 PASS)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.text(
        0.5, 0.01,
        f"Fig.4  L3 dense-pointcloud label-free pipeline. (a) Synthetic K walls "
        f"(illustrative subsample; config in pointcloud_gate.json). (b) Hidden-point "
        f"errors MAE={values[0]:.3f} / P50={values[1]:.3f} / P90={values[2]:.3f} deg; "
        f"misclassification (hidden) = {V_PC_MISCLASS * 100:.2f}%. All values loaded from "
        f"pointcloud_gate.json.",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "Fig4_pointcloud_pipe.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Figure 5: 诚实评测协议流程 (BlindInput) — 示意图, 无数字
# ---------------------------------------------------------------------------
def fig5_blindinput():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (1, 3, "输入\n(真值遮蔽)", "#3498db"),
        (3.5, 3, "模型预测", "#9b59b6"),
        (6, 3, "评测\n(隐伏 acos 均值)", "#27ae60"),
        (1, 1, "毒丸测试\n(误用真值 → 报警)", "#e74c3c"),
        (3.5, 1, "红旗审计\n(检测泄漏)", "#f39c12"),
        (6, 1, "结果报告\n(10-seed ± std)", "#2ecc71"),
    ]

    for x, y, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, alpha=0.7,
                                       edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")

    arrows = [(2.2, 3, 2.8, 3), (4.7, 3, 5.3, 3),
              (3.5, 2.6, 1, 1.4), (3.5, 2.6, 3.5, 1.4), (3.5, 2.6, 6, 1.4),
              (7.2, 3, 7.8, 3), (1, 1.4, 1, 2.6), (3.5, 1.4, 3.5, 2.6)]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.text(5, 4.5, "Figure 5. BlindInput 诚实评测协议", fontsize=12,
            fontweight="bold", ha="center")
    ax.text(5, 4.0, "(Poison-pill test + Red-flag audit + Hidden-truth evaluation)",
            fontsize=9, ha="center", style="italic", color="gray")

    fig.text(0.5, 0.02,
             "Fig.5  BlindInput honest evaluation protocol. Ground truth is hidden from models. "
             "Poison-pill test detects accidental truth leakage. Red-flag audit monitors for "
             "protocol violations.",
             ha="center", fontsize=8, style="italic")

    out = os.path.join(OUT_DIR, "Fig5_blindinput.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Figure 6: 多井联合适用性审计 —— 全部真数据 (multiwell_joint_audit.json)
# ---------------------------------------------------------------------------
def fig6_multiwell_audit():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # ---- (a) 真实 n_wells x n_wells 组心角距矩阵 (全部井对的 matched 均值对称化) ----
    n_wells = MW_AUDIT["n_wells"]
    pw = MW_AUDIT["pairwise_consistency"]
    assert len(pw) == n_wells * (n_wells - 1) // 2, "井对数与 n_wells 不一致"
    mat = np.zeros((n_wells, n_wells), dtype=float)
    for pair, rec in pw.items():
        i, j = (int(t) for t in pair.split("-"))
        v = rec["mean_matched"]
        mat[i][j] = v
        mat[j][i] = v

    mean_ang = MW_AUDIT["summary"]["mean_pairwise_angle"]
    # 自检锚 (v1.1.1-G1 验收, γ线 2026-09-01 补): 矩阵非对角均值必须与 JSON 自带
    # summary 字段一致, 对不上 = 对 JSON 结构理解错误, 宁可不画也不硬凑。
    offdiag = mat[mat > 0]
    assert abs(float(offdiag.mean()) - float(mean_ang)) < 0.05, (
        f"pairwise mean mismatch: matrix={offdiag.mean():.4f} vs summary={mean_ang:.4f}")

    # 色标上限由数据推导 (原硬编码 60 属未申报分析字面量)
    im = axes[0].imshow(mat, cmap="RdYlGn_r", vmin=0,
                        vmax=float(np.ceil(mat.max())), aspect="auto")
    axes[0].set_xlabel("Well index")
    axes[0].set_ylabel("Well index")
    axes[0].set_title("(a) 组心角距矩阵 (实测)")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Angular distance (°)")

    axes[0].text(0.98, 0.97, f"pairwise mean = {mean_ang:.1f}°",
                 transform=axes[0].transAxes, fontsize=8, color="black",
                 ha="right", va="top",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # ---- (b) 最佳伙伴一致性三档计数 (阈值全部 JSON 直读, 零硬编码) ----
    t_lo = MW_AUDIT["decision_rule"]["threshold_deg"]        # 决策阈值 (JSON 字段)
    t_hi = MW_AUDIT["_g2_annotation"]["tier_upper_deg"]      # 分档上界 (JSON 注记字段)
    cats = [f"<{t_lo:g}°", f"{t_lo:g}–{t_hi:g}°", f">{t_hi:g}°"]
    counts = [0, 0, 0]
    for w in MW_AUDIT["per_well"]:
        c = w["partner_consistency"]
        if c < t_lo:
            counts[0] += 1
        elif c < t_hi:
            counts[1] += 1
        else:
            counts[2] += 1
    colors_b = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = axes[1].bar(cats, counts, color=colors_b, edgecolor="black",
                       linewidth=0.8, alpha=0.8)
    for bar, val in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(counts) * 0.03,
                     str(val), ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of wells")
    axes[1].set_title("(b) 最佳伙伴一致性分档")
    axes[1].set_xlabel(f"per_well.partner_consistency, edges {t_lo}/{t_hi}° from JSON")
    axes[1].set_ylim(0, max(counts) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)

    # ---- (c) 最终联合决策计数 (JSON summary 字段直读) ----
    summ = MW_AUDIT["summary"]
    vcats = [f"Joint\n(<{t_lo:g}°且联合更优)", "Pool\n(全池化)", "Independent"]
    vcounts = [summ["n_recommend_joint"], summ["n_recommend_pool"],
               summ["n_recommend_independent"]]
    vcolors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = axes[2].bar(vcats, vcounts, color=vcolors, edgecolor="black",
                       linewidth=0.8, alpha=0.8)
    for bar, val in zip(bars, vcounts):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vcounts) * 0.03,
                     str(val), ha="center", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Number of wells")
    axes[2].set_title("(c) 审计最终建议")
    axes[2].set_ylim(0, max(vcounts) * 1.2)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Figure 6. 多井联合适用性审计 (beishan {n_wells} 井, 真实数据)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    pct_ind = summ["n_recommend_independent"] / n_wells * 100
    fig.text(
        0.5, 0.01,
        "Fig.6  Multi-well joint audit on REAL data (source: multiwell_joint_audit.json; "
        "no synthetic values). (a) inter-well set-mode angular distance, symmetrized from "
        f"{len(pw)} well-pairs. (b) wells binned by best-partner consistency "
        f"(edges {t_lo}/{t_hi} deg read from JSON decision_rule/_g2_annotation). "
        f"(c) final verdicts: joint={vcounts[0]}, pool={vcounts[1]}, independent={vcounts[2]} "
        f"({pct_ind:.0f}% better processed independently); decision rule = best-partner "
        f"consistency <{t_lo} deg AND joint improves >0.5 deg.",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "Fig6_multiwell_audit.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Paper Fig.6: 筛查级渗流曲线 (曲线取自口径一致的 beishan_full 6/6 配对;
# 20 实现精化阈值以文字引用自 dfn_pipeline_final.json —— 该文件曲线数组
# 存在 5/6 记录缺陷, 不用于绘图, 只引用其 crit 数字)
# ---------------------------------------------------------------------------
def paper_fig6_percolation():
    fig, ax = plt.subplots(figsize=(6, 4.2))

    perc = P32_FULL["percolation"]
    grid = perc["p32_grid"]
    conn = perc["p_conn"]
    assert len(grid) == len(conn), "渗流源文件 grid/conn 长度不一致, 禁止绘图"

    crit = perc["p32_crit"]
    lo = perc["p32_crit_lower"]
    hi = perc["p32_crit_upper"]

    ax.plot(grid, conn, marker="o", color="#3498db", linewidth=2.5,
            markersize=6, label="p_conn(P32)")
    ax.axvspan(lo, hi, alpha=0.3, color="#f39c12",
               label=f"crit band [{lo:.2f}, {hi:.2f}]")
    ax.axvline(x=crit, color="red", linestyle="--", linewidth=1.5,
               label=f"p32_crit = {crit:.2f}")

    n_real = perc.get("n_realizations_per_point", "?")
    beta = perc.get("beta", "?")
    ax.set_xlabel("P32 (m²/m³)")
    ax.set_ylabel("Percolation probability")
    ax.set_title("Paper Fig.6. 筛查级渗流曲线 (beishan Auto-K 组系表驱动)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fin = DFN_FINAL["percolation"]
    fig.text(
        0.5, 0.01,
        f"Fig.6 (paper) Screening percolation for Beishan set-table-conditioned Baecher discs "
        f"(beta={beta}, domain 50 m cube, {n_real} realizations per point; source: "
        f"dfn_pipeline_beishan_full.json — internally consistent curve). The finer "
        f"{fin['n_realizations_per_point']}-realization run refines p32_crit to "
        f"{fin['p32_crit']:.2f} [{fin['p32_crit_lower']:.2f}, {fin['p32_crit_upper']:.2f}] "
        f"(dfn_pipeline_final.json; its recorded curve arrays are length-defective and thus "
        f"not plotted). Excluded-volume anchor n*V_ex ~ 2.7 quoted in text.",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "PaperFig6_percolation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Paper Fig.8: 泄漏 vs 诚实 K 网格 (leaky / strict / honest oracle floor)
# 数据: p0_full_audit.json + phase0_oracle_floor_strict.json 直读
# ---------------------------------------------------------------------------
def paper_fig8_leak_kgrid():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    kg = P0_AUDIT["results"]["kgrid_beishan"]
    ks = sorted(kg.keys(), key=int)

    leaky_m = [kg[k]["leaky"]["mean"] for k in ks]
    leaky_s = [kg[k]["leaky"]["std"] for k in ks]
    strict_m = [kg[k]["strict"]["mean"] for k in ks]
    strict_s = [kg[k]["strict"]["std"] for k in ks]
    ora = ORACLE_FLOOR["results"]["beishan_22"]
    ora_m = [ora[k]["mae_mean"] for k in ks]
    ora_s = [ora[k]["mae_std"] for k in ks]
    xs = [int(k) for k in ks]

    ax.errorbar(xs, strict_m, yerr=strict_s, marker="o", color="#3498db",
                linewidth=1.5, capsize=3, label="strict (honest baseline)")
    ax.errorbar(xs, leaky_m, yerr=leaky_s, marker="s", color="#e74c3c",
                linewidth=1.5, capsize=3, label="leaky baseline")
    ax.errorbar(xs, ora_m, yerr=ora_s, marker="^", color="#f39c12",
                linewidth=1.5, capsize=3,
                label="honest oracle floor (observed centroids)")

    k_last = xs[-1]
    gap = kg[ks[-1]]["leakage"]
    ax.annotate(f"leak magnitude at K={k_last}: {gap:.1f}°",
                xy=(k_last, leaky_m[-1]), xytext=(-250, -50),
                textcoords="offset points", fontsize=9,
                color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

    ax.set_xlabel("K (number of sets)")
    ax.set_ylabel("point MAE (°)")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(max(strict_m), max(leaky_m), max(ora_m)) * 1.2)
    ax.set_title("Paper Fig.8. K 网格的泄漏假象 vs 诚实口径 (beishan)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.text(
        0.5, 0.01,
        "Fig.8 (paper) Leaky-vs-honest K-grid on Beishan (sources: p0_full_audit.json, "
        "phase0_oracle_floor_strict.json; auto-loaded). Monotone 'improvement' with K in the "
        "leaky column is the canonical self-prediction artifact: finer partitions shrink "
        "fallback groups and enlarge the leak channel, reaching "
        f"{gap:.1f} deg at K={k_last}. Oracle series is quarantined from claims (policy §4.3).",
        ha="center", fontsize=8, style="italic", wrap=True)

    out = os.path.join(OUT_DIR, "PaperFig8_leak_kgrid.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("论文图表包生成 (T88 图账同源版)")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 60)

    fig1_data_ladder()
    fig2_mar_vs_ladder()
    fig3_triple_source()
    fig4_pointcloud_pipe()
    fig5_blindinput()
    fig6_multiwell_audit()
    paper_fig6_percolation()
    paper_fig8_leak_kgrid()

    print("=" * 60)
    print("全部 6 张 Figure + 2 张论文规划图 (PaperFig6/8) 已生成")
    print("=" * 60)


if __name__ == "__main__":
    main()
