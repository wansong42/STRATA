# -*- coding: utf-8 -*-
"""T5/T20: 文献真值对照.

目的: 没有客户时,「自动 vs 人工」的对照证据. 北山是高放处置预选区,
公开论文里的组系解释表就是地质师免费标注.

管线: 输入 data/external/literature_sets.csv (paper, site, K, set_no, azimuth, dip),
对同场地我们自己的数据跑 set_table, 再用 set_table_score 对文献组心打分,
输出对照表 md.

T20 扩充: 纳入北山沙枣园(霍亮等2024)数据. 无本地原始数据的场地仅报告文献值.

验收: 管线对样例 csv 跑通, 输出对照表格式正确.
"""
import os, sys, json, csv
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fractureflow.set_table_eval import set_table_score, estimate_set_table
from fractureflow.setlabel import _sign_align

# ============================================================
# 辅助函数
# ============================================================

def azimuth_dip_to_normal(azimuth_deg, dip_deg):
    """方位角+倾角 → ENU 单位法向.

    azimuth: 倾向方位角 (从北顺时针, °)
    dip: 倾角 (°)
    ENU = (sin(dip)*sin(azimuth), sin(dip)*cos(azimuth), cos(dip))
    """
    az_rad = np.radians(azimuth_deg)
    dip_rad = np.radians(dip_deg)
    n = np.array([
        np.sin(dip_rad) * np.sin(az_rad),
        np.sin(dip_rad) * np.cos(az_rad),
        np.cos(dip_rad),
    ])
    return n / (np.linalg.norm(n) + 1e-12)


def load_literature_csv(path):
    """读 literature_sets.csv, 返回 list of dict."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["K"] = int(row["K"])
            row["set_no"] = int(row["set_no"])
            row["azimuth"] = float(row["azimuth"])
            row["dip"] = float(row["dip"])
            rows.append(row)
    return rows


def load_site_data(site_name):
    """根据 site 名加载我们自己的数据."""
    # FORGE
    if site_name in ("16B", "58-32"):
        forge_path = "data/external/utah_forge_fmi/forge_fmi_2wells.pt"
        data = torch.load(forge_path, weights_only=False)
        idx = data["widx"].get(site_name)
        if idx is not None:
            return data["nets"][idx]
        return None
    # beishan (若以后支持)
    return None


def run_site_comparison(lit_rows, site_net):
    """对单个 site 做完整对照."""
    # lit_rows: 同一 paper+site 的所有组
    K_lit = lit_rows[0]["K"]
    site = lit_rows[0]["site"]

    # 文献组心
    lit_centers = np.zeros((K_lit, 3))
    for row in lit_rows:
        sn = row["set_no"]
        lit_centers[sn] = azimuth_dip_to_normal(row["azimuth"], row["dip"])

    # 我们自己的数据: 全量 set_table
    pos = np.asarray(site_net["pos"], dtype=float)
    nrm = np.asarray(site_net.get("nrm_full", site_net["nrm"]), dtype=float)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    L = len(pos)
    occ = np.ones(L, dtype=bool)  # 全量

    # 用文献的 K 作为我们的 K (同样口径)
    pred_table = estimate_set_table(pos, nrm, occ, K=K_lit, seed=42, strict=True)

    # 评分
    score = set_table_score(pred_table["centers"], lit_centers)

    return {
        "site": site,
        "K_lit": K_lit,
        "K_pred": pred_table["centers"].shape[0],
        "modal_err_deg": score["modal_err_deg"],
        "coverage": score["coverage"],
        "matched_pairs": score["matched_pairs"],
        "lit_centers": lit_centers,
        "pred_centers": pred_table["centers"],
        "pred_assign": pred_table["assign"],
    }


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("T5: 文献真值对照")
    print("=" * 70)

    csv_path = "data/external/literature_sets.csv"
    if not os.path.exists(csv_path):
        print(f"[FAIL] {csv_path} not found")
        return

    rows = load_literature_csv(csv_path)
    print(f"Loaded {len(rows)} literature sets")

    # 按 paper+site 分组
    groups = {}
    for row in rows:
        key = (row["paper"], row["site"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    results = {}
    md_lines = []
    md_lines.append("# 自动 vs 文献组系对照表 (T20 升级版)\n")
    md_lines.append("> 对照是「量级互证」不是「逐组打分」。文献数据条件与我们不同。\n")
    md_lines.append("## 有本地数据可对照的场地\n")
    md_lines.append("| 场地 | K(文献) | K(自动) | 模态误差(°) | 覆盖率 | 逐组角距(°) |")
    md_lines.append("|------|---------|---------|-------------|--------|-------------|")

    # 收集无本地数据的文献组系
    lit_only_groups = {}

    for (paper, site), lit_rows in groups.items():
        print(f"\n  Paper: {paper}, Site: {site}")
        site_net = load_site_data(site)
        if site_net is None:
            print(f"    [LIT-ONLY] no local data for {site}, reporting literature values")
            lit_only_groups[(paper, site)] = lit_rows
            continue

        comp = run_site_comparison(lit_rows, site_net)
        key = f"{paper}__{site}"
        results[key] = {
            "paper": paper,
            "site": site,
            "K_lit": comp["K_lit"],
            "K_pred": comp["K_pred"],
            "modal_err_deg": comp["modal_err_deg"],
            "coverage": comp["coverage"],
            "matched_pairs": comp["matched_pairs"],
        }

        # 逐组角距
        pair_strs = []
        for pi, ti, err in comp["matched_pairs"]:
            pair_strs.append(f"P{pi}-T{ti}:{err:.1f}")

        print(f"    K_lit={comp['K_lit']}, K_pred={comp['K_pred']}, "
              f"modal_err={comp['modal_err_deg']:.2f} deg, coverage={comp['coverage']:.2f}")
        print(f"    Matched: {', '.join(pair_strs)}")

        md_lines.append(
            f"| {site} | {comp['K_lit']} | {comp['K_pred']} | "
            f"{comp['modal_err_deg']:.2f} | {comp['coverage']:.2f} | {', '.join(pair_strs)} |"
        )

    # 无本地数据的文献组系表
    if lit_only_groups:
        md_lines.append("")
        md_lines.append("## 无本地数据 — 仅报告文献组系\n")
        md_lines.append("| 论文 | 场地 | K | 组号 | 倾向(°) | 倾角(°) | 来源注记 |")
        md_lines.append("|------|------|---|------|---------|---------|----------|")
        for (paper, site), lit_rows in lit_only_groups.items():
            for row in lit_rows:
                md_lines.append(
                    f"| {paper} | {site} | {row['K']} | {row['set_no']} | "
                    f"{row['azimuth']} | {row['dip']} | {row['source_note']} |"
                )
            # 也存入 results
            key = f"{paper}__{site}"
            results[key] = {
                "paper": paper,
                "site": site,
                "K_lit": lit_rows[0]["K"],
                "K_pred": None,
                "modal_err_deg": None,
                "coverage": None,
                "matched_pairs": [],
                "literature_values": [
                    {"set_no": r["set_no"], "azimuth": r["azimuth"], "dip": r["dip"]}
                    for r in lit_rows
                ],
            }

    # 诚实注记
    md_lines.append("")
    md_lines.append("## 诚实注记\n")
    md_lines.append("- 文献数据条件与我们不同（测量深度范围、采样密度、解译标准），对照是「量级互证」不是「逐组打分」。")
    md_lines.append("- 模态误差 = 匈牙利匹配后匹配组的方向误差均值。")
    md_lines.append("- 覆盖率 = 匹配角距 <30° 的真值组占比。")
    md_lines.append("- 自动组系基于法向几何聚类（球面 k-means），非地质成因分组。")
    md_lines.append("- 无本地数据的场地仅报告文献组系，无法做自动对照。")
    md_lines.append("- 文献组系多为地质师解译产物（含地质成因判断），对照仅作量级互证。")

    md_text = "\n".join(md_lines)

    # 落盘
    out_json = os.path.join(ROOT, "results/literature_truth_compare.json")
    with open(out_json, "w") as f:
        json.dump({"task": "T20", "results": results,
                    "assumptions": [
                        "文献组心来自论文明列的走向/倾向, 自动组心来自全量球面 k-means",
                        "对照是量级互证, 非逐组打分",
                        "无本地数据的场地仅报告文献组系, 无法做自动对照",
                        "文献组系多为地质师解译产物(含地质成因判断), 对照仅作量级互证",
                    ],
                    "reproducibility": {
                        "environment": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe",
                        "command": "python scripts/literature_truth_compare.py",
                    }}, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n  JSON: {out_json}")

    out_md = os.path.join(ROOT, "results/literature_truth_compare.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  MD: {out_md}")

    print("\n" + "=" * 70)
    print("T5 PASS" if results else "T5 FAIL (no results)")
    print("=" * 70)


if __name__ == "__main__":
    main()
