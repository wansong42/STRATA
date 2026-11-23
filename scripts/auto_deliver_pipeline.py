# -*- coding: utf-8 -*-
"""端到端业务交付管线 — 从**仓内示例数据**到分级交付报告.

⚠ 输入通道: 本件**没有**。全部取数固定来自
  data/real/loaded_real_nets_setid.pt (见 SOURCE_REL 与 run_full_pipeline 里的字面路径),
  --out-dir 只决定产物写到哪儿。历史 docstring 曾示例 `--single-well data/my_well.pt
  --grade-only` —— 这两个旗标在本件 argparse 里从来不存在 (会直接 argparse 报错),
  照抄者会以为它能读自己的数据, 故删去并改为下面的真实用法。

整合 Phase 2.3 质量闸门 + Phase 4 主动学习分级 + Phase 3 后验 CI.

流程:
  1. 输入: 仓内示例井数据 (法向 + 位置)
  2. 自动打标: spherical k-means → set_aware_dirs
  3. 质量闸门: 组点数 + 组心角距防御
  4. 不确定性: 边界点占比 + K 分歧度
  5. 产品分级: A / B / C (判据所依据的 MAE = 自掩码全量调查口径, 见 MAE_CALIBER_LABEL)
  6. 后验 CI: bootstrap 90% 置信区间 (显式播种 BOOTSTRAP_SEED)
  7. 输出: 组系表 + 报告 (Markdown, 首部带演示声明) + 路由决策

用法:
  python scripts/auto_deliver_pipeline.py --dataset beishan_22 --out-dir results/delivery
  python scripts/auto_deliver_pipeline.py --dataset all --max-wells 5 --out-dir <仅输出目录>

要处理客户自己的编录表, 请用现役入口 scripts/borehole_excel_entry.py (表 → net .pt)
再接 scripts/auto_label_borehole.py --data real --real-path <.pt> --out-dir <目录>。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ---------------------------------------------------------------------------
# guichain_20260925 · 演示口径声明与 bootstrap 播种 (T1 路线 b / T2 / T3)
#
# 事实基础 (实测, 非措辞): 本管线没有任何客户数据输入通道 —— argparse 五个旗标全是
# 参数/输出侧, 取数处唯一来源是下面 SOURCE_REL 的字面常量。故凡由本件产出的报告,
# 描述的对象恒为仓内示例数据, 与调用方选的 out-dir 是哪个文件夹无关。
# 声明必须落在**产物品**上 (报告首部 + 汇总 json), 而不是只写在源码注释里。
# ---------------------------------------------------------------------------
SOURCE_REL = "data/real/loaded_real_nets_setid.pt"
DEMO_DISCLOSURE = (
    "⚠ 演示口径: 本报告描述的是仓内示例数据 (%s), **非你提供的数据**。"
    "本管线无客户数据输入通道; 你选择的目录只用作输出。" % SOURCE_REL
)
DEMO_DISCLOSURE_TOKEN = "非你提供的数据"
# 分级所依据 MAE 的协议标签 (R263 两栏纪律同款: 读数不得脱离口径标签单独出现)
MAE_CALIBER_LABEL = ("口径=自掩码全量调查 (聚类与组心可见全部法向, 含被掩掉的隐伏点); "
                     "该读数是自洽统计, 不是 BlindInput 诚实榜口径, 不构成对外精度承诺")
# bootstrap 播种: 仓内日期种子惯例 (R226 20260916 / R249 20260918)
BOOTSTRAP_SEED = 20260925

# 疑似编录表探测 (单一实现, GUI 与本件共用; 产品侧禁出现第二份判定逻辑)
_TABLE_EXTS = (".csv", ".xlsx", ".xls")
# 自家产物名 (避免二次运行时把自己写的表当成"客户没被读的输入")
_PRODUCT_OUTPUT_PATTERNS = ("_group_table.csv", "legacy_borehole.csv")


def find_candidate_input_tables(folder, exclude_subdirs=()):
    """列出 folder 下疑似编录表的文件 (相对路径, 排序确定)。

    用途: 本管线**没有**输入通道, 所以"目录里有表"这一事实本身就是危险信号 —— 客户会
    以为报告说的是那些表。GUI 据此在启动前拒办, 本件据此在日志与汇总里响亮告警。
    判定口径: 扩展名 ∈ {.csv,.xlsx,.xls}; 递归但跳过 exclude_subdirs 与点开头目录;
    跳过自家产物名; 排序后返回 ⇒ 同一份盘面两次调用逐字相同 (不含任何 dict 序依赖)。
    """
    hits = []
    if not folder or not os.path.isdir(folder):
        return hits
    skip_dirs = set(exclude_subdirs or ())
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in skip_dirs and not d.startswith("."))
        for fn in sorted(filenames):
            low = fn.lower()
            if not low.endswith(_TABLE_EXTS):
                continue
            if any(low.endswith(p) for p in _PRODUCT_OUTPUT_PATTERNS):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), folder)
            hits.append(rel.replace(os.sep, "/"))
    return sorted(hits)


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _acos_err(pred, true):
    p, t = _unit(pred), _unit(true)
    c = np.clip(np.abs((p * t).sum(-1)), 0, 1)
    return np.rad2deg(np.arccos(c))


class WellDeliverable:
    """单口井的交付物."""

    def __init__(self, net, K=12):
        self.wid = net.get('wid', 'unknown')
        self.nrm = np.asarray(net.get('nrm_full', net.get('nrm')), float)
        self.pos = np.asarray(net.get('pos', np.random.randn(len(self.nrm), 3)), float)
        self.K = K
        self.n_pts = len(self.nrm)
        self.grade = None
        self.set_ids = None
        self.group_table = []
        self.uncertainty = {}
        self.mae_validated = None

    def auto_label(self):
        """自动打标 + 质量闸门."""
        from fractureflow.setlabel import generate_set_ids
        from phase2_quality_gates import quality_gate_filter

        K_eff = min(self.K, self.n_pts)
        if K_eff < 2:
            return False

        sid = generate_set_ids(self.nrm, Krange=(K_eff, K_eff), seed=42)

        # 质量闸门
        new_sid, qgate_info = quality_gate_filter(self.nrm, sid, K_eff)
        self.set_ids = new_sid
        self.quality_gate_info = qgate_info
        return True

    def validate_mae(self, n_seeds=10):
        """隐伏点 MAE 验证."""
        from fractureflow.inference import set_aware_dirs as sad

        errors = []
        for s in range(n_seeds):
            ms = 999 + 1000 * s
            L = self.n_pts
            rng = np.random.default_rng(ms)
            mask = rng.random(L) < 0.4
            if mask.sum() < 2 or (~mask).sum() < 1:
                continue
            K_loc = min(self.set_ids.max() + 1, int(mask.sum()))
            if K_loc < 2:
                continue
            if self.set_ids.max() >= K_loc:
                from fractureflow.setlabel import generate_set_ids
                sid = generate_set_ids(self.nrm, Krange=(K_loc, K_loc), seed=42)
            else:
                sid = self.set_ids
            dirs, _ = sad(self.pos, self.nrm, mask, sid)
            e = _acos_err(dirs[~mask], self.nrm[~mask])
            errors.append(float(e.mean()))

        self.mae_validated = float(np.mean(errors)) if errors else 999
        self.mae_std = float(np.std(errors)) if errors else 0
        return self.mae_validated

    def compute_group_table(self, n_bootstrap=100, seed=BOOTSTRAP_SEED):
        """计算组系表 + 后验 CI.

        bootstrap 走**显式播种**的 rng (guichain_20260925 T2): 原实现用全局
        `np.random.choice` (未播种), 而同件 :84 的掩码走 `default_rng(ms)` 是播种的 ——
        后果是报告里 90% CI 那一列同输入两次运行不可逐位复现 (实测五井全中,
        端点最大漂移 3.39°)。逐组派生子种子 `default_rng([seed, group_id])` ⇒ 某一组
        的 CI 只由该组自身数据决定, 别组点数变化不会把它挪走。
        """
        seed = int(seed)
        self.bootstrap_seed = seed
        nrm_u = _unit(self.nrm)
        unique_sids = np.unique(self.set_ids)

        self.group_table = []
        for k in unique_sids:
            sel = nrm_u[self.set_ids == k]
            n = len(sel)
            if n < 2:
                continue

            # 组心 (符号对齐)
            ref = sel.mean(0)
            ref /= np.linalg.norm(ref) + 1e-12
            sgn = np.sign((sel * ref).sum(-1, keepdims=True))
            sgn[sgn == 0] = 1
            center = _unit((sel * sgn).mean(0))

            # 组内离散
            cos = np.abs(sel @ center)
            dispersion = float(np.median(np.rad2deg(np.arccos(np.clip(cos, 0, 1)))))

            # Bootstrap CI
            rng = np.random.default_rng([seed, int(k)])
            angles = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                boot = sel[idx]
                ref_b = boot.mean(0)
                ref_b /= np.linalg.norm(ref_b) + 1e-12
                sgn_b = np.sign((boot * ref_b).sum(-1, keepdims=True))
                sgn_b[sgn_b == 0] = 1
                center_b = _unit((boot * sgn_b).mean(0))
                ang = np.rad2deg(np.arccos(np.clip(np.abs(center @ center_b), 0, 1)))
                angles.append(ang)

            ci_lo = float(np.percentile(angles, 5))
            ci_hi = float(np.percentile(angles, 95))

            self.group_table.append({
                'group_id': int(k),
                'n_points': n,
                'center': [round(float(x), 4) for x in center],
                'dispersion_deg': round(dispersion, 2),
                'ci_90': [round(ci_lo, 2), round(ci_hi, 2)],
                'bootstrap_seed': seed,
            })

    def compute_uncertainty(self, clf=None):
        """不确定性指标."""
        nrm_u = _unit(self.nrm)
        unique_sids = np.unique(self.set_ids)

        # 边界点占比 (基于组指派的熵)
        boundary_count = 0
        for i in range(self.n_pts):
            # 到各组心的距离
            dists = []
            for g in self.group_table:
                c = np.array(g['center'])
                cos = abs(nrm_u[i] @ c)
                dists.append(1 - cos)
            if len(dists) >= 2:
                sorted_d = sorted(dists)
                ratio = sorted_d[1] / (sorted_d[0] + 1e-6)
                if 0.5 < ratio < 2.0:
                    boundary_count += 1

        self.uncertainty['boundary_frac'] = boundary_count / max(self.n_pts, 1)

        # K 分歧度
        from fractureflow.setlabel import generate_set_ids
        assignments = []
        for k in [4, 6, 8, 10, 12]:
            if k < self.n_pts:
                sid = generate_set_ids(self.nrm, Krange=(k, k), seed=42)
                assignments.append(sid)

        agree = 0
        total = 0
        if len(assignments) >= 2:
            for i in range(min(self.n_pts, 30)):
                for j in range(i + 1, min(self.n_pts, 30)):
                    same_count = sum(1 for a in assignments if a[i] == a[j])
                    agree += same_count / len(assignments)
                    total += 1

        self.uncertainty['k_agreement'] = agree / max(total, 1)
        self.uncertainty['k_disagreement'] = 1.0 - self.uncertainty['k_agreement']

    def assign_grade(self):
        """产品分级 (基于 MAE + 不确定性)."""
        mae = self.mae_validated if self.mae_validated is not None else 9.0
        bf = self.uncertainty.get('boundary_frac', 0.5)

        if mae < 7.0 and bf < 0.2:
            self.grade = 'A'
        elif mae < 8.5:
            self.grade = 'B'
        else:
            self.grade = 'C'

    def generate_report(self, out_dir, well_idx=0):
        """生成 Markdown 报告."""
        os.makedirs(out_dir, exist_ok=True)
        wid_short = self.wid.replace('.csv', '').replace('/', '_').replace('\\', '_')
        if len(wid_short) > 20:
            wid_short = wid_short[-20:]
        report_path = os.path.join(out_dir, f"{well_idx:03d}_{wid_short}_report.md")

        lines = [
            f"# {self.wid} — 裂隙组系自动打标报告",
            "",
            DEMO_DISCLOSURE,
            "",
            f"**数据来源**: `{SOURCE_REL}` (仓内示例数据, 由本管线唯一取数路径加载)",
            f"**本件对应井**: {self.wid} (取数文件内的记录名, 非输入文件名)",
            f"**级别**: {self.grade}",
            f"**数据**: {self.n_pts} 个裂隙点",
            f"**协议**: K={self.K}, 10 mask seeds",
            f"**bootstrap seed**: {BOOTSTRAP_SEED}",
            "",
            "## 精度验证",
            "",
            f"> **{MAE_CALIBER_LABEL}**",
            "",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 隐伏点 MAE (均值±std) | {self.mae_validated:.2f}° ± {self.mae_std:.2f}° |",
            f"| 有效组数 | {len(self.group_table)} |",
            f"| 质量闸门 | 已应用 |",
            "",
            "## 不确定性",
            "",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 边界点占比 | {self.uncertainty.get('boundary_frac', 0):.1%} |",
            f"| K 分歧度 | {self.uncertainty.get('k_disagreement', 0):.3f} |",
            "",
            "## 组系表",
            "",
            "| 组 ID | 点数 | 离散度 (°) | 90% CI (°) | 中心方向 |",
            "|---|---|---|---|---|",
        ]

        for g in self.group_table:
            ci = g['ci_90']
            center_str = f"({g['center'][0]:.3f}, {g['center'][1]:.3f}, {g['center'][2]:.3f})"
            lines.append(
                f"| {g['group_id']} | {g['n_points']} | {g['dispersion_deg']:.1f} | "
                f"[{ci[0]:.1f}, {ci[1]:.1f}] | {center_str} |"
            )

        lines.extend([
            "",
            "## 处置建议",
            "",
        ])

        # 处置建议 (guichain_20260925 T3): 只弱化不加严语义 ——
        # 原 A/B 两档句首写的是「全自动交付」并带「无需人工复核」, 而触发它的 mae 是自掩码
        # 全量调查口径的自洽统计 (分级阈值落定 2026-08-18, 未随后续判废重标定), 不足以支撑
        # 那种语气。分级机制与阈值本身一字未动 (改判据语义 = 红线), 改的只是对外措辞。
        if self.grade == 'A':
            lines.append("- **自动打标完成**: 组内一致性与质量闸门自检通过 (判据 mae<7.0 且 boundary_frac<0.2)")
            lines.append("  - 该判据里的 mae 是**自掩码自洽统计**, 不是对真实隐伏情况的部署精度标定;")
            lines.append("    分级阈值尚未用带真值的独立样本重标定 ⇒ 本档不免除人工抽查。")
        elif self.grade == 'B':
            lines.append("- **自动打标完成**: 中置信组系 (判据 mae<8.5), 建议例行抽查")
            lines.append("  - 同上: 口径为自掩码自洽统计, 分级阈值未独立重标定。")
        else:
            lines.append("- **⚠ 人工复核推荐**: 组边界模糊或组系不稳定")
            lines.append(f"  - 边界点占比 {self.uncertainty.get('boundary_frac', 0):.1%} 超阈值")
            # 原「预计人工耗时: ~8 分钟」是仓内无任何标定依据的量化承诺 (全库 grep: 该数只在
            # 本件出现), 按裁定 58 对外零量化承诺移除, 不做同义替换。
            lines.append("  - 复核工作量取决于组数与边界点条数, 本件不给出耗时预估")

        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by auto_deliver_pipeline.py at {time.strftime('%Y-%m-%d %H:%M')}*")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return report_path


def run_full_pipeline(dataset, out_dir, K=12, n_seeds=10, max_wells=None):
    """完整管线."""
    ROOT_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据
    real_pt = os.path.join(ROOT_, "data/real/loaded_real_nets_setid.pt")
    d = torch.load(real_pt, weights_only=False)

    nets = []
    for n in d:
        wid = str(n.get('wid', ''))
        if dataset == 'beishan_22' and 'BS34' in wid:
            nets.append(n)
        elif dataset == 'loaded_real_30' and 'BS34' not in wid:
            nets.append(n)
        elif dataset == 'all':
            nets.append(n)

    if max_wells:
        nets = nets[:max_wells]

    # out_dir 里躺着疑似编录表 = 最容易出事的形态 (客户以为读的是自己的数据)。
    # 本件不改 CLI 的成败语义 (禁改判据语义), 只做响亮告警 + 把清单落进汇总 json 供机检。
    candidate_inputs = find_candidate_input_tables(out_dir, exclude_subdirs=("fractureflow_output",))

    print("=" * 70)
    print("端到端业务交付管线")
    print("=" * 70)
    print(DEMO_DISCLOSURE)
    if candidate_inputs:
        print("⚠⚠ 目录内检测到 %d 个疑似编录表文件**未被本管线读取** (它没有输入通道):"
              % len(candidate_inputs))
        for rel in candidate_inputs[:10]:
            print("     - %s" % rel)
        print("   产物描述的是 %s, 不是这些表。需要跑你自己的数据请走现役入口 "
              "scripts/borehole_excel_entry.py + scripts/auto_label_borehole.py。" % SOURCE_REL)
    print(f"数据集: {dataset}  井数: {len(nets)}  K={K}")
    print()

    # 处理每口井
    grades = {'A': [], 'B': [], 'C': []}

    for i, net in enumerate(nets):
        wid_short = str(net.get('wid', f'well_{i}'))[-30:]
        print(f"[{i+1}/{len(nets)}] {wid_short}...", end=' ')

        deliver = WellDeliverable(net, K)

        # 1. 自动打标
        if not deliver.auto_label():
            print("SKIP (too few points)")
            continue

        # 2. MAE 验证
        mae = deliver.validate_mae(n_seeds)

        # 3. 组系表 + CI
        deliver.compute_group_table()

        # 4. 不确定性
        deliver.compute_uncertainty()

        # 5. 分级
        deliver.assign_grade()

        # 6. 报告
        report_path = deliver.generate_report(out_dir, i)

        grades[deliver.grade].append({
            'wid': deliver.wid,
            'mae': round(mae, 2),
            'n_groups': len(deliver.group_table),
            'boundary_frac': round(deliver.uncertainty.get('boundary_frac', 0), 3),
            'report': report_path,
        })

        print(f"MAE={mae:.2f}° Grade={deliver.grade}")

    # 汇总
    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    for g in ['A', 'B', 'C']:
        n = len(grades[g])
        pct = n / len(nets) * 100 if nets else 0
        mae_str = ""
        if grades[g]:
            mae_avg = np.mean([w['mae'] for w in grades[g]])
            mae_str = f"  avg MAE={mae_avg:.2f}°"
        print(f"  {g} 级: {n} 井 ({pct:.0f}%){mae_str}")

    # 保存汇总 (只增键; 既有键名与数值一字未动 —— 现役五入口读的就是这几键)
    summary = {
        "dataset": dataset,
        "K": K,
        "n_wells": len(nets),
        "n_seeds": n_seeds,
        "grade_distribution": {g: len(grades[g]) for g in 'ABC'},
        "grade_details": grades,
        "disclosure": DEMO_DISCLOSURE,
        "mae_caliber_label": MAE_CALIBER_LABEL,
        "data_provenance": {
            "source_file_rel": SOURCE_REL,
            "source_file_abs": os.path.relpath(real_pt, ROOT).replace(os.sep, "/"),
            "reads_user_input": False,
            "input_channel_flags": [],
            "dataset_filter": dataset,
            "max_wells": max_wells,
            "well_ids_in_report": [str(n.get('wid', '')) for n in nets],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "out_dir_role": "output only (本管线无输入通道)",
            "out_dir_candidate_input_files_unread": candidate_inputs,
        },
    }
    summary_path = os.path.join(out_dir, "pipeline_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    print(f"\n汇总: {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="端到端业务交付管线")
    ap.add_argument("--dataset", default="beishan_22")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out-dir", default="results/delivery")
    ap.add_argument("--max-wells", type=int, default=None)
    args = ap.parse_args()

    run_full_pipeline(args.dataset, args.out_dir, args.K, args.seeds, args.max_wells)


if __name__ == "__main__":
    main()
