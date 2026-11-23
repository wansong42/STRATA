# -*- coding: utf-8 -*-
"""R100 稳定性指标实件图 — N07A 页素材生成脚本.

只读消费既有 results/ 真值 JSON, 渲染两张"真实输出" PNG:
  1) results/v_r100_ui_evidence.png   K=4 vs K=6 组系表对照 (主图, 软件输出页风格)
  2) results/v_r100_conflict_card.png 双源对账冲突处理规则卡 (副图)
并写出可溯源元数据 results/v_r100_evidence.json (每个登图数字带 [源: path::key]).

铁律:
  - 禁止创造数字; 所有数值来自 set_table_a_k4/k6.json 与 v_r96_fusion_report.json.
  - 倾向/倾角由法向换算 (右手法则, 与 v_r21_demo 报告一致):
        dip     = degrees(arccos(|nz|))
        dip_dir = degrees(atan2(nx, ny)) mod 360

运行:
  $env:PYTHONUTF8="1"; $env:KMP_DUPLICATE_LIB_OK="TRUE"
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/v_r100_ui_evidence.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SRC_K4 = os.path.join(RESULTS, "v_r21_demo", "set_table_a_k4.json")
SRC_K6 = os.path.join(RESULTS, "v_r21_demo", "set_table_a_k6.json")
SRC_R96 = os.path.join(RESULTS, "v_r96_fusion_report.json")
SRC_R21_VERDICT = os.path.join(RESULTS, "v_r21_demo", "verdict.md")

OUT_MAIN = os.path.join(RESULTS, "v_r100_ui_evidence.png")
OUT_CARD = os.path.join(RESULTS, "v_r100_conflict_card.png")
OUT_EVIDENCE = os.path.join(RESULTS, "v_r100_evidence.json")

# ---------------------------------------------------------------------------
# 中文字体 (Windows 环境用微软雅黑)
# ---------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "Noto Sans CJK SC",
]


def _setup_cjk_font():
    """挑选一个可用中文字体, 加入 rcParams font.sans-serif 首位."""
    # 先确保字体注册 (ttc/otf 需显式 addfont)
    for f in font_manager.findSystemFonts():
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            continue
    avail = {fm.name: fm.fname for fm in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        # 宽松匹配 (微软雅黑/宋体/黑体/Noto 等)
        matched = [n for n in avail if name.lower() in n.lower()] or \
                  [n for n in avail if any(k in n.lower() for k in
                   ("yahei", "simhei", "noto sans cjk", "noto sans sc"))]
        if matched:
            chosen = sorted(matched, key=len)[0]
            plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 真值读取 + 换算
# ---------------------------------------------------------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normal_to_dip_dir_dip(n):
    """右手法则: dip_dir = atan2(nx, ny) mod 360; dip = arccos(|nz|)."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    dip_dir = math.degrees(math.atan2(nx, ny)) % 360.0
    dip = math.degrees(math.acos(abs(nz)))
    return dip_dir, dip


def _fmt_ratio(p):
    return f"{p * 100:.1f}%"


def _build_k_table(src_path, K):
    """返回 (rows, trace_centers, trace_kappa, trace_prop).
    rows: [ [组号, dip_dir, dip, kappa, 占比] ]
    """
    data = _load_json(src_path)
    centers = data["centers"]
    concentrations = data["concentrations"]
    proportions = data["proportions"]
    assert len(centers) == len(concentrations) == len(proportions) == K, \
        f"{os.path.basename(src_path)} 长度与 K={K} 不符"
    rows = []
    for i in range(K):
        dip_dir, dip = _normal_to_dip_dir_dip(centers[i])
        kappa = concentrations[i]
        prop = proportions[i]
        rows.append({
            "group": i,
            "dip_dir": round(dip_dir, 1),
            "dip": round(dip, 1),
            "kappa": round(kappa, 1),
            "prop_pct": round(prop * 100.0, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# 主图: K=4 vs K=6 组系表对照 (软件输出页风格)
# ---------------------------------------------------------------------------
def render_main(rows4, rows6):
    _setup_cjk_font()
    n_rows = max(len(rows4), len(rows6))
    # 用 figure-relative 坐标布局; 固定 fig 高度, 表格行高按 n_rows 自动缩
    fig_w, fig_h = 10.8, 7.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("#f4f6f9")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#f4f6f9")
    ax.axis("off")

    # ---- 顶栏: 软件输出页标题 ----
    title = "FractureFlow · 结构面组系统计表 (SetTable)"
    ax.text(0.5, 0.965, title, ha="center", va="top",
            fontsize=19, fontweight="bold", color="#1f2937")
    sub = ("方案: 实测输出对照   对象: FORGE 16B 井 (FMI picks, 2087 条)   "
           "管线: v_r21_demo v1.0")
    ax.text(0.5, 0.915, sub, ha="center", va="top",
            fontsize=11.5, color="#4b5563")

    # 两列宽度: 0.5 - left_gap ... 每列左右留白
    col_x = [0.03, 0.53]
    col_w = 0.44
    col_titles = ["档位 K = 4", "档位 K = 6"]

    for col_idx, (rows, K, ctitle) in enumerate(zip([rows4, rows6], [4, 6], col_titles)):
        x0 = col_x[col_idx]
        y_top = 0.85
        # 列标题 (档位标签)
        ax.text(x0 + col_w / 2, y_top, ctitle, ha="center", va="center",
                fontsize=14, fontweight="bold", color="#0f766e")
        y = y_top - 0.045
        headers = ["组", "倾向°", "倾角°", "浓度 κ", "占比"]
        # 表头底色条
        hx0, hx1 = x0, x0 + col_w
        ax.add_patch(plt.Rectangle((hx0, y - 0.032), hx1 - hx0, 0.045,
                                   facecolor="#0f766e", edgecolor="none", zorder=1))
        col_centers = [x0 + col_w * (i + 0.5) / len(headers) for i in range(len(headers))]
        for cx, htxt in zip(col_centers, headers):
            ax.text(cx, y, htxt, ha="center", va="center",
                    fontsize=11.5, color="white", fontweight="bold", zorder=2)

        y -= 0.07
        for r in rows:
            # 交替行底色
            if r["group"] % 2 == 0:
                ax.add_patch(plt.Rectangle((x0, y - 0.032), col_w, 0.05,
                                           facecolor="#e5f1ef", edgecolor="none", zorder=0))
            vals = [str(r["group"]), f"{r['dip_dir']:.1f}", f"{r['dip']:.1f}",
                    f"{r['kappa']:.1f}", f"{r['prop_pct']:.1f}%"]
            for cx, v in zip(col_centers, vals):
                ax.text(cx, y, v, ha="center", va="center",
                        fontsize=11, color="#111827", zorder=2)
            y -= 0.055

    # ---- 图注 (逐档裁决依据) ----
    caption = ("同一输入、两档 K 的组系表对照（实测输出）；κ=组内集中度（越大组内越稳定）"
               "——工程师逐档裁决的依据")
    ax.text(0.5, 0.022, caption, ha="center", va="bottom",
            fontsize=12, color="#374151")

    fig.savefig(OUT_MAIN, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 副图: 冲突处理规则卡
# ---------------------------------------------------------------------------
def render_conflict_card(r96, trace):
    _setup_cjk_font()
    fig = plt.figure(figsize=(10.4, 7.4), dpi=200)
    fig.patch.set_facecolor("#fdf6f0")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#fdf6f0")
    ax.axis("off")

    ax.text(0.5, 0.97, "双源对账 · 冲突处理规则卡", ha="center", va="top",
            fontsize=19, fontweight="bold", color="#7f1d1d")

    ax.text(0.5, 0.925,
            "来源: R96 融合对账（人工编录 vendor × 机器视觉自动） · 同口井 16B · K=6",
            ha="center", va="top", fontsize=11.5, color="#6b7280")

    # ---- 冲突实例 (标红) ----
    ax.text(0.045, 0.855, "冲突实例（双源方位实质冲突 → 转人工仲裁）", ha="left", va="top",
            fontsize=13.5, fontweight="bold", color="#111827")

    # 从 r96 找 red 对
    red = next(p for p in r96["reconcile"]["pairs"] if p["decision"] == "red")
    ia, ib = red["ia"], red["ib"]
    n_a, n_b = red["n_a"], red["n_b"]
    # 源表组方向
    vg = r96["source_tables"]["vendor"]["groups"][ia]
    vgn = r96["source_tables"]["vision"]["groups"][ib]
    ang = red["angdeg"]

    card_x0, card_y0, card_w, card_h = 0.045, 0.42, 0.91, 0.39
    ax.add_patch(plt.Rectangle((card_x0, card_y0), card_w, card_h,
                               facecolor="#fff5f5", edgecolor="#dc2626",
                               linewidth=2.0, zorder=1))

    ax.text(card_x0 + 0.02, card_y0 + card_h - 0.04,
            "■ 冲突对 · 自动标记 · 待人工仲裁", ha="left", va="top",
            fontsize=13, color="#b91c1c", fontweight="bold", zorder=2)
    ax.text(card_x0 + 0.02, card_y0 + card_h - 0.115,
            f"源 A (vendor 组 {ia}):  倾角 {vg['dip']:.1f}° / 倾向 {vg['dip_dir']:.1f}° "
            f"/ n={n_a}",
            ha="left", va="top", fontsize=12, color="#111827", zorder=2)
    ax.text(card_x0 + 0.02, card_y0 + card_h - 0.175,
            f"源 B (视觉组 {ib}):  倾角 {vgn['dip']:.1f}° / 倾向 {vgn['dip_dir']:.1f}° "
            f"/ n={n_b}",
            ha="left", va="top", fontsize=12, color="#111827", zorder=2)
    ax.text(card_x0 + 0.02, card_y0 + card_h - 0.235,
            f"双源角距 = {ang:.2f}°  ≥ 35°  →  判定红 · 转人工仲裁",
            ha="left", va="top", fontsize=13, color="#b91c1c", fontweight="bold", zorder=2)

    # ---- 规则三行 ----
    ax.text(0.045, 0.34, "对账规则（三色门控）", ha="left", va="top",
            fontsize=13.5, fontweight="bold", color="#111827")
    rules = [
        ("< 20°", "联合（双源互证同方位）", "#15803d"),
        ("20 – 35°", "最一致对（单源或半重合 · 待人工复核）", "#b45309"),
        (">= 35°", "独立 或 转仲裁（双源方位实质冲突）", "#b91c1c"),
    ]
    y = 0.285
    for band, txt, color in rules:
        ax.add_patch(plt.Rectangle((0.045, y - 0.028), 0.145, 0.045,
                                   facecolor=color, edgecolor="none", zorder=1))
        ax.text(0.045 + 0.0725, y, band, ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold", zorder=2)
        ax.text(0.215, y, txt, ha="left", va="center",
                fontsize=12, color="#111827", zorder=2)
        y -= 0.065

    # ---- 对账计数 ----
    n_g = r96["reconcile"]["n_green"]
    n_y = r96["reconcile"]["n_yellow"]
    n_r = r96["reconcile"]["n_red"]
    n_pair = r96["reconcile"]["n_matched_pairs"]
    ax.text(0.045, 0.095,
            f"本井对账: 匹配 {n_pair} 对 → 绿(互证) {n_g} / 黄(存疑) {n_y} / 红(冲突) {n_r}",
            ha="left", va="center", fontsize=12.5, color="#111827")

    ax.text(0.5, 0.028, "规则全文见演示包 · 附规则册", ha="center", va="center",
            fontsize=11, style="italic", color="#6b7280")

    fig.savefig(OUT_CARD, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 证据元数据 (每个登图数字 → 溯源)
# ---------------------------------------------------------------------------
def build_evidence(rows4, rows6, r96):
    def krow_src(K, i, key):
        fname = "set_table_a_k4.json" if K == 4 else "set_table_a_k6.json"
        if key in ("dip_dir", "dip"):
            return f"results/v_r21_demo/{fname}::centers[{i}] (法向换算 右手法则)"
        if key == "kappa":
            return f"results/v_r21_demo/{fname}::concentrations[{i}]"
        return f"results/v_r21_demo/{fname}::proportions[{i}]"

    tables = {"K4": [], "K6": []}
    for K, rows, tag in ((4, rows4, "K4"), (6, rows6, "K6")):
        for r in rows:
            tables[tag].append({
                "组号": r["group"],
                "倾向°": {"value": r["dip_dir"], "源": krow_src(K, r["group"], "dip_dir")},
                "倾角°": {"value": r["dip"], "源": krow_src(K, r["group"], "dip")},
                "浓度κ": {"value": r["kappa"], "源": krow_src(K, r["group"], "kappa")},
                "占比": {"value": r["prop_pct"], "源": krow_src(K, r["group"], "prop")},
            })

    # 冲突卡数字
    red = next(p for p in r96["reconcile"]["pairs"] if p["decision"] == "red")
    ia, ib = red["ia"], red["ib"]
    vg = r96["source_tables"]["vendor"]["groups"][ia]
    vgn = r96["source_tables"]["vision"]["groups"][ib]

    evidence = {
        "meta": {
            "task": "R100 稳定性指标实件图 · N07A 页素材",
            "mode": "只读消费既有 results/, 无评测/重跑; 图由真实产物渲染, 非手绘",
            "interpreter": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe",
            "convention": "dip=arccos(|nz|); dip_dir=atan2(nx,ny) mod 360 (右手法则, 与 v_r21_demo 一致)",
        },
        "images": {
            "main": {
                "file": "results/v_r100_ui_evidence.png",
                "caption": "同一输入、两档 K 的组系表对照（实测输出）；κ=组内集中度（越大组内越稳定）——工程师逐档裁决的依据",
                "source_files": ["results/v_r21_demo/set_table_a_k4.json",
                                 "results/v_r21_demo/set_table_a_k6.json"],
                "object": "FORGE 16B 井 (FMI picks, 2087 条)",
                "pipeline": "v_r21_demo v1.0",
            },
            "conflict_card": {
                "file": "results/v_r100_conflict_card.png",
                "caption": "规则全文在演示包·附规则册",
                "source_files": ["results/v_r96_fusion_report.json"],
                "object": "同口井 16B · 人工编录(vendor) × 机器视觉自动 · K=6",
            },
        },
        "main_table": tables,
        "conflict_card_numbers": {
            "双源角距": {"value": red["angdeg"],
                       "源": "results/v_r96_fusion_report.json::reconcile.pairs（decision=red）.angdeg"},
            "源A组号": {"value": ia,
                      "源": "results/v_r96_fusion_report.json::reconcile.pairs（decision=red）.ia"},
            "源B组号": {"value": ib,
                      "源": "results/v_r96_fusion_report.json::reconcile.pairs（decision=red）.ib"},
            "源A倾角": {"value": vg["dip"],
                      "源": f"results/v_r96_fusion_report.json::source_tables.vendor.groups[{ia}].dip"},
            "源A倾向": {"value": vg["dip_dir"],
                      "源": f"results/v_r96_fusion_report.json::source_tables.vendor.groups[{ia}].dip_dir"},
            "源A点数": {"value": red["n_a"],
                      "源": f"results/v_r96_fusion_report.json::reconcile.pairs（decision=red）.n_a"},
            "源B倾角": {"value": vgn["dip"],
                      "源": f"results/v_r96_fusion_report.json::source_tables.vision.groups[{ib}].dip"},
            "源B倾向": {"value": vgn["dip_dir"],
                      "源": f"results/v_r96_fusion_report.json::source_tables.vision.groups[{ib}].dip_dir"},
            "源B点数": {"value": red["n_b"],
                      "源": f"results/v_r96_fusion_report.json::reconcile.pairs（decision=red）.n_b"},
            "匹配对数": {"value": r96["reconcile"]["n_matched_pairs"],
                       "源": "results/v_r96_fusion_report.json::reconcile.n_matched_pairs"},
            "绿组数": {"value": r96["reconcile"]["n_green"],
                     "源": "results/v_r96_fusion_report.json::reconcile.n_green"},
            "黄组数": {"value": r96["reconcile"]["n_yellow"],
                     "源": "results/v_r96_fusion_report.json::reconcile.n_yellow"},
            "红组数": {"value": r96["reconcile"]["n_red"],
                     "源": "results/v_r96_fusion_report.json::reconcile.n_red"},
            "阈值20°": {"value": r96["config"]["green_deg"],
                       "源": "results/v_r96_fusion_report.json::config.green_deg"},
            "阈值35°": {"value": r96["config"]["yellow_deg"],
                       "源": "results/v_r96_fusion_report.json::config.yellow_deg"},
        },
    }
    return evidence


def main():
    rows4 = _build_k_table(SRC_K4, 4)
    rows6 = _build_k_table(SRC_K6, 6)
    r96 = _load_json(SRC_R96)

    render_main(rows4, rows6)
    render_conflict_card(r96, trace=None)

    evidence = build_evidence(rows4, rows6, r96)
    with open(OUT_EVIDENCE, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, ensure_ascii=False, indent=2, allow_nan=False)

    print("R100 实件图已生成:")
    print(f"  主图  : {OUT_MAIN}")
    print(f"  规则卡: {OUT_CARD}")
    print(f"  溯源  : {OUT_EVIDENCE}")


if __name__ == "__main__":
    main()
