# -*- coding: utf-8 -*-
"""公开勘察报告复现管线 —— T24 核心脚本 (轻量框架版).

从真实勘察项目 PDF / 附录提取编录表 (深度+倾向+倾角), 跑标准路线 A 管线,
产自动组系表 + 玫瑰图 + HTML 报告章节, 并与地质师手工统计做对照.

管线:
  加载 CSV (深度/dip/dip_direction 必填)
  -> dip_dipdir_to_normal 转法向
  -> spherical k-means Auto-K 打标 (复用 setlabel.generate_set_ids)
  -> 组系表 CSV (复用 borehole_report.build_group_table)
  -> 玫瑰图 PNG (复用 borehole_report.plot_rose_diagram)
  -> HTML 报告 (auto_report.html)
  -> [可选] 与 ground truth 对照 (comparison.json)

用法:
  python scripts/report_replay.py --csv data/x.csv --well-id XK01 --output-dir results/replay/XK01
  python scripts/report_replay.py --csv data/x.csv --well-id XK01 --ground-truth data/x_truth.json --output-dir results/replay/XK01
  python scripts/report_replay.py --csv data/x.csv --well-id XK01 --K 4 --output-dir results/replay/XK01
  python scripts/report_replay.py --batch --manifest data/external/published_reports/manifest.csv --output-dir results/replay/

输入 CSV 列名约定 (大小写不敏感, 自动识别):
  深度: depth / md / 深度(m) / 深度
  倾角: dip / 倾角 / 倾角(deg) / 倾角(°)
  倾向: dip_direction / dip_dir / 倾向 / 倾向(deg) / 倾向(°)

地面真值 JSON 格式 (可选):
  {"K": 4, "groups": [{"group_id":0, "strike":135.0, "dip_dir":225.0, "dip":45.0, "count":30}, ...]}

依赖: numpy, matplotlib, torch (纯 numpy 实现). 不引入新第三方包.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
_CN_FONT = None
for _fn in ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC", "Heiti SC", "Arial Unicode MS"]:
    if any(f.name == _fn for f in font_manager.fontManager.ttflist):
        _CN_FONT = _fn
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 复用项目已有模块
from fractureflow.setlabel import generate_set_ids
from fractureflow.geometry import dip_dipdir_to_normal as _central_dip_dipdir_to_normal
from fractureflow.borehole_report import (
    build_group_table, plot_rose_diagram, normal_to_dip_dir,
)


# ============================================================
# 1. 数据加载与列名识别
# ============================================================

DEPTH_ALIASES = {"depth", "md", "深度(m)", "深度（m）", "深度", "m", "depth(m)"}
DIP_ALIASES = {"dip", "倾角(deg)", "倾角（°）", "倾角", "倾角(°)", "dip(deg)", "dip_angle"}
DIPDIR_ALIASES = {"dip_direction", "dip_dir", "倾向(deg)", "倾向（°）", "倾向",
                  "倾向(°)", "azimuth", "trend", "dip_dir(deg)"}


def _normalize_col(name):
    name = name.strip()
    name = name.replace("（", "(").replace("）", ")").replace("［", "[").replace("］", "]")
    return name.lower().strip()


def _find_col(headers, aliases):
    for h in headers:
        if _normalize_col(h) in aliases:
            return h
    return None


def _dip_dipdir_to_normal(dip_dir, dip):
    """倾向/倾角 -> 单位法向 (ENU). 与 read_forge_las.py 同口径 (参数序同样反序).

    ⚠ 参数序 = **(倾向, 倾角)**, 与函数名字面顺序相反。唯一权威 =
    fractureflow.geometry.dip_dipdir_to_normal(dip_deg=, dip_dir_deg=);
    本函数是显式关键字薄封装, 与权威逐位相同 (W38 20000 点复算 max diff 0.0),
    由 selfcheck S35 常驻把守。

    约定: nz = cos(dip), 水平分量幅值 = sin(dip) (无向 ±n 同面).
    历史 bug (2026-08-28 修复): 曾把 sin/cos 互换, dip 被镜像成 90°-dip,
    复现框架产出的组系表倾角列整体错误 (浅层地层被聚成陡倾假组系).
    """
    return _central_dip_dipdir_to_normal(dip_deg=np.asarray(dip, float),
                                         dip_dir_deg=np.asarray(dip_dir, float))


def load_borehole_csv(csv_path):
    """加载钻孔编录表 CSV. 返回 dict 含 depth/nrm/n_valid/n_skipped."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample) if csv.Sniffer().has_header(sample) else csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))

    headers = list(rows[0].keys())
    dip_col = _find_col(headers, DIP_ALIASES)
    dipdir_col = _find_col(headers, DIPDIR_ALIASES)
    depth_col = _find_col(headers, DEPTH_ALIASES)

    if not dip_col:
        raise ValueError(f"未识别倾角列. 表头: {headers}")
    if not dipdir_col:
        raise ValueError(f"未识别倾向列. 表头: {headers}")

    dip_vals, dipdir_vals, depth_vals = [], [], []
    skipped = 0
    for r in rows:
        try:
            d, dd = float(r[dip_col]), float(r[dipdir_col])
        except (ValueError, TypeError, KeyError):
            skipped += 1
            continue
        if not (0 <= d <= 90) or not (0 <= dd <= 360):
            skipped += 1
            continue
        dip_vals.append(d)
        dipdir_vals.append(dd)
        if depth_col:
            try:
                depth_vals.append(float(r[depth_col]))
            except (ValueError, TypeError, KeyError):
                depth_vals.append(float(len(dip_vals)))
        else:
            depth_vals.append(float(len(dip_vals)))

    if len(dip_vals) < 3:
        raise ValueError(f"有效数据仅 {len(dip_vals)} 条 (<3). CSV: {csv_path}")

    nrm = _dip_dipdir_to_normal(np.asarray(dipdir_vals), np.asarray(dip_vals))
    return {
        "depth": np.asarray(depth_vals, float),
        "dip": np.asarray(dip_vals, float),
        "dip_direction": np.asarray(dipdir_vals, float),
        "nrm": nrm,
        "n_valid": len(dip_vals),
        "n_skipped": skipped,
        "csv_path": csv_path,
    }


# ============================================================
# 2. Ground Truth 对照
# ============================================================

def load_ground_truth(json_path):
    """加载地质师手工统计 JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    if "groups" in gt:
        gt["K"] = len(gt["groups"])
    return gt


def compare_with_truth(auto_table, gt):
    """自动组系表 vs 地质师手工统计. 返回对照 dict."""
    K_auto, K_truth = len(auto_table), gt.get("K", 0)
    gt_groups = gt.get("groups", [])

    # 真值法向
    tn_list = []
    for g in gt_groups:
        n = _dip_dipdir_to_normal(np.array([g.get("dip_dir", g.get("dip_direction", 0))]),
                                   np.array([g.get("dip", 0)]))
        tn_list.append(n[0])
    truth_normals = np.array(tn_list) if tn_list else np.zeros((0, 3))

    # 自动组法向
    an_list = []
    for row in auto_table:
        n = _dip_dipdir_to_normal(np.array([row["倾向(°)"]]), np.array([row["倾角(°)"]]))
        an_list.append(n[0])
    auto_normals = np.array(an_list) if an_list else np.zeros((0, 3))

    # 贪心匹配 (按最小角距)
    used = set()
    matches = []
    for i, an in enumerate(auto_normals):
        best_j, best_err = -1, 180.0
        for j, tn in enumerate(truth_normals):
            if j in used:
                continue
            err = float(np.degrees(np.arccos(np.clip(np.abs(an @ tn), 0, 1))))
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            used.add(best_j)
            matches.append({
                "auto_group": i + 1, "truth_group": best_j + 1,
                "angle_error(°)": round(best_err, 2),
                "auto_count": auto_table[i]["条数"],
                "truth_count": gt_groups[best_j].get("count", "?"),
            })

    angle_errors = [m["angle_error(°)"] for m in matches]
    return {
        "K_auto": K_auto, "K_truth": K_truth, "K_match": K_auto == K_truth,
        "n_matched": len(matches), "group_matches": matches,
        "angle_errors(°)": angle_errors,
        "mean_angle_error(°)": round(np.mean(angle_errors), 2) if angle_errors else None,
        "max_angle_error(°)": round(max(angle_errors), 2) if angle_errors else None,
        "notes": gt.get("notes", ""), "source": gt.get("source", ""),
    }


# ============================================================
# 3. HTML 报告
# ============================================================

def generate_html_report(well_id, data, auto_table, K, out_dir,
                         comparison=None, project_name=None):
    """生成 auto_report.html."""
    site_name = project_name or well_id
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_total, n_skipped = data["n_valid"], data["n_skipped"]

    table_rows = ""
    for row in auto_table:
        table_rows += (f"<tr><td>{row['组号']}</td><td>{row['走向(°)']}</td>"
                       f"<td>{row['倾向(°)']}</td><td>{row['倾角(°)']}</td>"
                       f"<td>{row['条数']}</td><td>{row['占比(%)']}</td>"
                       f"<td>{row['组内离散(°)']}</td></tr>\n")

    comparison_html = ""
    if comparison:
        comp_rows = ""
        for m in comparison["group_matches"]:
            comp_rows += (f"<tr><td>自动组{m['auto_group']}</td><td>真值组{m['truth_group']}</td>"
                          f"<td>{m['angle_error(°)']}</td><td>{m['auto_count']}</td>"
                          f"<td>{m['truth_count']}</td></tr>\n")
        comparison_html = f"""
    <h2>与地质师手工统计对照</h2>
    <table><tr><th>自动组</th><th>真值组</th><th>角距(°)</th><th>自动条数</th><th>真值条数</th></tr>
    {comp_rows}</table>
    <p>组数一致: {'是' if comparison['K_match'] else '否'} (自动 K={comparison['K_auto']} vs 真值 K={comparison['K_truth']})<br>
    平均角距: {comparison['mean_angle_error(°)']}° | 最大角距: {comparison['max_angle_error(°)']}°<br>
    来源: {comparison.get('source', '')}</p>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>{site_name} 裂隙组系统计报告</title>
<style>
body {{ font-family: "Microsoft YaHei","SimHei",sans-serif; max-width:900px; margin:0 auto; padding:20px; color:#333; }}
h1 {{ color:#1a5276; border-bottom:2px solid #1a5276; padding-bottom:10px; }}
h2 {{ color:#2c3e50; margin-top:30px; }}
table {{ border-collapse:collapse; width:100%; margin:15px 0; }}
th,td {{ border:1px solid #ddd; padding:8px 12px; text-align:center; }}
th {{ background:#1a5276; color:white; }} tr:nth-child(even) {{ background:#f2f2f2; }}
img {{ max-width:100%; border:1px solid #ddd; margin:10px 0; }}
.note {{ background:#fef9e7; border-left:4px solid #f1c40f; padding:12px; margin:15px 0; }}
.footer {{ margin-top:40px; padding-top:10px; border-top:1px solid #ddd; color:#888; font-size:0.85em; }}
</style></head>
<body>
<h1>{site_name} — 裂隙组系统计报告</h1>
<p><strong>井号/项目</strong>: {well_id} | <strong>生成时间</strong>: {ts}</p>
<div class="note"><strong>重要说明</strong> (诚实边界):<br>
- 组系划分基于法向几何聚类 (球面 k-means), 非地质成因分组<br>
- 法向基于钻孔产状测量, 可能存在测量误差<br>- 本报告为自动复现结果, 仅供初步参考</div>
<h2>1. 输入数据摘要</h2>
<table><tr><th>参数</th><th>值</th></tr>
<tr><td>有效裂隙数</td><td>{n_total} 条</td></tr>
<tr><td>剔除(无效/越界)</td><td>{n_skipped} 条</td></tr>
<tr><td>识别组数 K</td><td>{K} 组</td></tr></table>
<h2>2. 走向玫瑰图</h2>
<img src="auto_rose.png" alt="走向玫瑰图">
<h2>3. 裂隙组系表</h2>
<table><tr><th>组号</th><th>走向(°)</th><th>倾向(°)</th><th>倾角(°)</th><th>条数</th><th>占比(%)</th><th>组内离散(°)</th></tr>
{table_rows}</table>
<p>注: 走向 = 倾向 - 90° (mod 360). 组内离散 = 与组心的平均角距.</p>
{comparison_html}
<div class="footer">T24 公开勘察报告复现 | 入口: scripts/report_replay.py | 环境: pytorch_env</div>
</body></html>"""
    html_path = os.path.join(out_dir, "auto_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ============================================================
# 4. 主流程
# ============================================================

def replay_single(csv_path, well_id, output_dir, ground_truth_path=None,
                  K=None, Krange=(2, 7), seed=42, project_name=None):
    """单井/单项目复现管线."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 加载 CSV
    print(f"[1/5] 加载编录表: {csv_path}")
    data = load_borehole_csv(csv_path)
    nrm = data["nrm"]
    print(f"  有效裂隙: {data['n_valid']} 条, 剔除: {data['n_skipped']} 条")

    # Step 2: 聚类打标 (复用 setlabel.generate_set_ids)
    print(f"[2/5] 球面 k-means 聚类...")
    if K is not None:
        from fractureflow.setlabel import spherical_kmeans
        centers_raw, assign = spherical_kmeans(nrm, K, seed=seed)
        print(f"  指定 K={K}")
    else:
        assign = generate_set_ids(nrm, Krange=Krange, seed=seed)
        K = int(assign.max()) + 1
        # 反算组心用于可视化
        from fractureflow.setlabel import spherical_kmeans
        centers_raw, _ = spherical_kmeans(nrm, K, seed=seed)
        print(f"  Auto-K = {K} (范围 {Krange})")

    # 组心方向 (用于 build_group_table / plot_stereonet)
    centers = centers_raw

    # Step 3: 组系表 (复用 borehole_report.build_group_table -> DataFrame)
    print(f"[3/5] 构建组系表...")
    df_table = build_group_table(nrm, assign, centers, K)
    # DataFrame -> list[dict] 用于 HTML / 对照 / 打印
    table = df_table.to_dict(orient="records")
    table_csv_path = os.path.join(output_dir, "auto_group_table.csv")
    df_table.to_csv(table_csv_path, index=False, encoding="utf-8-sig")
    for row in table:
        print(f"  组{row['组号']}: 走向{row['走向(°)']}° 倾向{row['倾向(°)']}° "
              f"倾角{row['倾角(°)']}° | {row['条数']}条 ({row['占比(%)']}%)")

    # Step 4: 玫瑰图 (复用 borehole_report.plot_rose_diagram)
    print(f"[4/5] 绘制玫瑰图...")
    rose_path = os.path.join(output_dir, "auto_rose.png")
    plot_rose_diagram(nrm, assign, K, rose_path, title=f"{well_id} 走向玫瑰图")

    # Step 5: 对照 + HTML 报告
    comparison = None
    if ground_truth_path:
        print(f"[5/5] 加载地面真值对照: {ground_truth_path}")
        gt = load_ground_truth(ground_truth_path)
        comparison = compare_with_truth(table, gt)
        comp_path = os.path.join(output_dir, "comparison.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"  组数一致: {comparison['K_match']} | "
              f"平均角距: {comparison['mean_angle_error(°)']}°")
    else:
        print(f"[5/5] 无地面真值, 跳过对照")

    html_path = generate_html_report(
        well_id, data, table, K, output_dir,
        comparison=comparison, project_name=project_name)

    summary = {
        "well_id": well_id, "project_name": project_name,
        "csv_path": csv_path, "n_valid": data["n_valid"], "n_skipped": data["n_skipped"],
        "K": K, "group_table": table, "output_dir": output_dir,
        "files": {
            "auto_report.html": html_path,
            "auto_group_table.csv": table_csv_path,
            "auto_rose.png": rose_path,
            "comparison.json": os.path.join(output_dir, "comparison.json") if comparison else None,
        },
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)

    print(f"\n=== 复现完成 ===")
    print(f"  输出: {output_dir}")
    print(f"  报告: {html_path}")
    return summary


def replay_batch(manifest_path, output_dir):
    """批量模式: 扫描 manifest 中 status=extracted 的条目逐个复现."""
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "extracted"]
    print(f"[batch] 已提取条目: {len(rows)}")
    results = []
    for row in rows:
        csv_path = row.get("csv_path", "")
        well_id = row.get("id", "unknown")
        gt_path = row.get("ground_truth", "") or None
        if not csv_path or not os.path.exists(csv_path):
            print(f"  [skip] {well_id}: CSV 不存在")
            continue
        try:
            summary = replay_single(csv_path, well_id,
                                    os.path.join(output_dir, well_id),
                                    ground_truth_path=gt_path)
            results.append(summary)
        except Exception as e:
            print(f"  [error] {well_id}: {e}")
            results.append({"well_id": well_id, "error": str(e)})
    batch_path = os.path.join(output_dir, "batch_summary.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    print(f"[batch] 完成 {len(results)} 项 -> {batch_path}")
    return results


# ============================================================
# 5. CLI 入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="T24 公开勘察报告复现管线 — 编录表 CSV -> 组系表 + 玫瑰图 + HTML 报告")
    ap.add_argument("--csv", default=None, help="输入编录表 CSV 路径")
    ap.add_argument("--well-id", default=None, help="井号/项目标识")
    ap.add_argument("--output-dir", default=None,
                    help="输出目录 (默认 results/replay/<well_id>)")
    ap.add_argument("--ground-truth", default=None,
                    help="地质师手工统计 JSON (可选, 用于对照)")
    ap.add_argument("--K", type=int, default=None, help="指定组数 K (不指定则 Auto-K)")
    ap.add_argument("--Kmin", type=int, default=2, help="Auto-K 下限")
    ap.add_argument("--Kmax", type=int, default=7, help="Auto-K 上限")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--project-name", default=None, help="项目显示名称")
    ap.add_argument("--batch", action="store_true", help="批量模式")
    ap.add_argument("--manifest", default=None, help="批量模式 manifest CSV 路径")
    args = ap.parse_args()

    if args.batch:
        if not args.manifest:
            ap.error("--batch 需要 --manifest")
        output_dir = args.output_dir or os.path.join(ROOT, "results", "replay")
        replay_batch(args.manifest, output_dir)
        return

    if not args.csv:
        ap.error("需要 --csv (或 --batch 模式)")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 不存在: {args.csv}")

    well_id = args.well_id or os.path.splitext(os.path.basename(args.csv))[0]
    output_dir = args.output_dir or os.path.join(ROOT, "results", "replay", well_id)
    replay_single(csv_path=args.csv, well_id=well_id, output_dir=output_dir,
                  ground_truth_path=args.ground_truth, K=args.K,
                  Krange=(args.Kmin, args.Kmax), seed=args.seed,
                  project_name=args.project_name)


if __name__ == "__main__":
    main()
