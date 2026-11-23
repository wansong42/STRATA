# -*- coding: utf-8 -*-
"""R98: 北山稳健性快照与逐井明细表（PPT N08 注脚与演示包备查件）。

任务 (tasks/R98_任务书_★☆_北山稳健性快照与逐井明细表.md):
1. 观测比例敏感性快照: beishan 22 井, obs_frac ∈ {0.2,0.4,0.6,0.8,1.0} × 10 seeds,
   K=12 modal_err; 与 results/deployment_bonus.json 的 beishan_22 行对照 (容差 ±0.5°)。
2. 逐井对比明细表: 每井 条数/K=12 modal_err/主方向 vs 真组前三组对照/leave-one-well-out 拉动。
3. 一张敏感性曲线 PNG (0.2→1.0 五档 + 1σ 误差带)。

口径 100% 复刻 r66_three_source_verification.eval_beishan / deployment_bonus_curve.eval_beishan_obsfrac:
- 真值 = set_ids (离线弱真值), 逐点 build_truth_from_assign
- 掩码 = make_blind(mask_seed=999+1000*s, obs_frac)
- 预测 = estimate_set_table(pos, truth_nrm, occ, K=12, seed=42, strict=True)  (观测-only, 无泄漏)
- 指标 = set_table_score.modal_err_deg

产物全部 v_r98_ 前缀, 零触碰冻结件。
"""
import os, sys, json, math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fractureflow.set_table_eval import set_table_score, estimate_set_table
from fractureflow.honest_eval import make_blind
from fractureflow.setlabel import _sign_align

OBS_FRAC_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 10
K = 12
BLIND_CALIBER = 0.4  # "40% 盲测档" 口径
TOLERANCE_DEG = 0.5   # 与 deployment_bonus ±0.5° 容差

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. beishan 数据
# ---------------------------------------------------------------------------

def load_beishan():
    """beishan 22 井 (BS34 网, 每井 40 条)。"""
    d = torch.load(os.path.join(ROOT, "data/real/loaded_real_nets_setid.pt"),
                   weights_only=False)
    beishan = [n for n in d if "BS34" in str(n.get("wid", ""))]
    # 排序: 按 widx 稳定输出
    beishan = sorted(beishan, key=lambda n: int(n.get("widx", 0)))
    return beishan


def well_label(net):
    """井号: BS34#{widx}。"""
    return f"BS34#{int(net.get('widx', 0))}"


def build_truth_from_assign(assign, nrm, K_truth):
    """从逐点指派算组系表中心 (与 r66 同款)。"""
    centers = np.zeros((K_truth, 3))
    for k in range(K_truth):
        sel = assign == k
        if sel.sum():
            pts = nrm[sel]
            pts_s = _sign_align(pts)
            centers[k] = pts_s.mean(0)
            centers[k] /= np.linalg.norm(centers[k]) + 1e-12
    return centers


def well_modal_err(net, mask_seed, obs_frac):
    """单井在给定掩码下的 K=12 modal_err (度)。

    口径对齐 deployment_bonus_curve.eval_beishan_obsfrac:
    仅要求 occ.sum() >= 2 (不含 r66 的 (~occ).sum()==0 守卫),
    以便 obs_frac=1.0 (全量建表 vs 真值组心) 也能评测。
    """
    x = make_blind(net, mask_seed=mask_seed, obs_frac=obs_frac)
    if x.occ.sum() < 2:
        return None
    truth_nrm = np.asarray(net.get("nrm_full", net["nrm"]), dtype=float)
    truth_nrm = truth_nrm / (np.linalg.norm(truth_nrm, axis=1, keepdims=True) + 1e-12)
    sid = np.asarray(net["set_ids"], dtype=int)
    K_truth = int(sid.max()) + 1
    pred = estimate_set_table(x.pos, truth_nrm, x.occ, K=K, seed=42, strict=True)
    truth_centers = build_truth_from_assign(sid, truth_nrm, K_truth)
    score = set_table_score(pred["centers"], truth_centers)
    return float(score["modal_err_deg"])


# ---------------------------------------------------------------------------
# 2. 敏感性快照 + 逐井明细
# ---------------------------------------------------------------------------

def normal_to_dip_dipdir(n):
    """法向 -> (dip, dip_direction) 度。

    约定 (全项目统一, 见 forge_fmi_pipeline._nd / 八期 arcsin→arccos 修复):
        n = [sin d * sin a, sin d * cos a, cos d]
    无向法向用 |cos| 约定: dip = arccos(|nz|), a = arctan2(nx, ny)。
    """
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    dip = math.degrees(math.acos(np.clip(abs(n[2]), 0, 1)))
    a = math.degrees(math.atan2(n[0], n[1])) % 360.0
    return dip, a


def top3_group_compare(net):
    """该井预测 K=12 组心 vs 真值组心, 取误差最小的前三组。

    返回 list[dict]: {truth_group, truth_n, pred_dip, pred_dipdir,
                      truth_dip, truth_dipdir, err_deg}
    """
    truth_nrm = np.asarray(net.get("nrm_full", net["nrm"]), dtype=float)
    truth_nrm = truth_nrm / (np.linalg.norm(truth_nrm, axis=1, keepdims=True) + 1e-12)
    sid = np.asarray(net["set_ids"], dtype=int)
    K_truth = int(sid.max()) + 1
    truth_centers = build_truth_from_assign(sid, truth_nrm, K_truth)

    # 预测用全量 (obs_frac=1.0, 建表 vs 真值组心, 无泄漏) —— 与 top3 对照同口径
    pos = np.asarray(net["pos"], dtype=float)
    occ = np.ones(len(truth_nrm), dtype=bool)
    pred = estimate_set_table(pos, truth_nrm, occ, K=K, seed=42, strict=True)
    pred_centers = pred["centers"]

    score = set_table_score(pred_centers, truth_centers)
    pairs = score["matched_pairs"]  # [(pred_i, truth_j, err_deg), ...]
    pairs = sorted(pairs, key=lambda p: p[2])[:3]

    out = []
    for pi, tj, e in pairs:
        truth_grp = int(tj)
        t_dip, t_ddir = normal_to_dip_dipdir(truth_centers[tj])
        p_dip, p_ddir = normal_to_dip_dipdir(pred_centers[pi])
        n_truth = int((sid == truth_grp).sum())
        out.append({
            "truth_group": truth_grp,
            "truth_n": n_truth,
            "truth_dip": round(t_dip, 2),
            "truth_dipdir": round(t_ddir, 2),
            "pred_dip": round(p_dip, 2),
            "pred_dipdir": round(p_ddir, 2),
            "err_deg": round(float(e), 3),
        })
    return out


def main():
    print("=" * 72)
    print("R98: 北山稳健性快照与逐井明细表")
    print("=" * 72)
    beishan = load_beishan()
    print(f"beishan 井数 = {len(beishan)}")
    for n in beishan:
        L = len(np.asarray(n.get('nrm_full', n['nrm'])))
        assert L == 40, f"{well_label(n)} 条数 {L} != 40 (任务书断言)"
    print("每井条数 = 40 断言 PASS")

    # ---- 敏感性快照 ----
    print("\n[1/3] 观测比例敏感性 (K=12 modal_err, 10 seeds) ...")
    sensitivity = {}
    for of in OBS_FRAC_LIST:
        seed_means = []   # 每 seed 的 22 井均值
        for s in range(N_SEEDS):
            mask_seed = 999 + 1000 * s
            well_errs = []
            for net in beishan:
                e = well_modal_err(net, mask_seed, of)
                if e is not None:
                    well_errs.append(e)
            if well_errs:
                seed_means.append(float(np.mean(well_errs)))
        mean = float(np.mean(seed_means))
        std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        sensitivity[str(of)] = {
            "modal_err_mean": round(mean, 4),
            "modal_err_std": round(std, 4),
            "n_seeds": len(seed_means),
        }
        print(f"  obs_frac={of:.1f}: {mean:.4f} ± {std:.4f} deg")

    # 与 deployment_bonus.json 对照
    db_path = os.path.join(ROOT, "results/deployment_bonus.json")
    db = json.load(open(db_path, encoding="utf-8"))
    db_beishan = db["deployment_bonus"]["beishan_22"]
    diffs = {}
    all_in_tol = True
    print("\n  对照 deployment_bonus.json (beishan_22, 容差 ±0.5°):")
    for of in OBS_FRAC_LIST:
        live = sensitivity[str(of)]["modal_err_mean"]
        ref = db_beishan[str(of)]["modal_err_mean"]
        diff = live - ref
        ok = abs(diff) <= TOLERANCE_DEG
        all_in_tol &= ok
        diffs[str(of)] = {"live": round(live, 4), "ref": round(ref, 4),
                          "diff": round(diff, 4), "within_tolerance": ok}
        print(f"  obs_frac={of:.1f}: live={live:.4f} ref={ref:.4f} diff={diff:+.4f} [{'OK' if ok else 'OUT-OF-TOL'}]")
    if not all_in_tol:
        print("  ⚠️ 存在超容差档, 需自查后向架构师报告 (禁止静默改数)。")

    # ---- 逐井明细 ----
    print("\n[2/3] 逐井明细表 (K=12, 40% 盲测档) ...")
    well_rows = []
    per_well_seed_means = {}
    for net in beishan:
        wid = well_label(net)
        L = len(np.asarray(net.get('nrm_full', net['nrm'])))
        errs_over_seeds = []
        for s in range(N_SEEDS):
            e = well_modal_err(net, 999 + 1000 * s, BLIND_CALIBER)
            if e is not None:
                errs_over_seeds.append(e)
        w_mean = float(np.mean(errs_over_seeds))
        per_well_seed_means[wid] = w_mean
        well_rows.append({
            "well": wid,
            "n_records": int(L),
            "modal_err_deg": round(w_mean, 4),
            "modal_err_std": round(float(np.std(errs_over_seeds, ddof=1)), 4)
            if len(errs_over_seeds) > 1 else 0.0,
            "top3_groups": top3_group_compare(net),
        })

    # leave-one-well-out 敏感性
    all_means = np.array([per_well_seed_means[r["well"]] for r in well_rows])
    overall_mean = float(all_means.mean())
    loo_by_well = {}
    for i, r in enumerate(well_rows):
        wid = r["well"]
        rest = np.delete(all_means, i)
        loo_mean = float(rest.mean())
        loo_by_well[wid] = {
            "mean_without_well": round(loo_mean, 4),
            "pull_deg": round(overall_mean - loo_mean, 4),  # 剔除后全井均值变化 (正=该井拉高均值)
            "overall_mean": round(overall_mean, 4),
        }
    for r in well_rows:
        r["leave_one_out"] = loo_by_well[r["well"]]
    print(f"  全井均值 (40% 盲测档) = {overall_mean:.4f}°")

    # ---- 落盘 JSON ----
    print("\n[3/3] 落盘 ...")
    output = {
        "task": "R98",
        "task_name": "北山稳健性快照与逐井明细表",
        "script": "scripts/v_r98_sensitivity.py",
        "date": "2026-08-27",
        "K": K,
        "n_seeds": N_SEEDS,
        "obs_frac_list": OBS_FRAC_LIST,
        "blind_caliber_deg": BLIND_CALIBER,
        "caliber_note": f"{BLIND_CALIBER*100:.0f}% = {BLIND_CALIBER*100:.0f}% 盲测档 (只给看 {BLIND_CALIBER*100:.0f}%、藏 {100-BLIND_CALIBER*100:.0f}% 的刁难口径)",
        "data": {
            "n_wells": len(beishan),
            "records_per_well": 40,
            "truth": "set_ids (离线弱真值, 非掩码真值)",
            "strict": True,
            "note": "逐点 build_truth_from_assign 真值组心; 观测-only k-means 建表 (无泄漏)",
        },
        "sensitivity": sensitivity,
        "deployment_bonus_compare": {
            "source": "results/deployment_bonus.json -> deployment_bonus.beishan_22",
            "tolerance_deg": TOLERANCE_DEG,
            "all_within_tolerance": all_in_tol,
            "per_frac_diff": diffs,
        },
        "per_well_table": well_rows,
        "overall_mean_deg_40pct": round(overall_mean, 4),
    }

    json_path = os.path.join(ROOT, "results/v_r98_sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  JSON: {json_path}")

    # ---- 逐井 CSV + MD ----
    csv_lines = ["well,n_records,modal_err_deg,modal_err_std,loo_mean_without,pull_deg,overall_mean,"
                 "tg1_truth_dip,tg1_truth_dipdir,tg1_pred_dip,tg1_pred_dipdir,tg1_err,"
                 "tg2_truth_dip,tg2_truth_dipdir,tg2_pred_dip,tg2_pred_dipdir,tg2_err,"
                 "tg3_truth_dip,tg3_truth_dipdir,tg3_pred_dip,tg3_pred_dipdir,tg3_err"]
    for r in well_rows:
        loo = r["leave_one_out"]
        tg = r["top3_groups"]
        pad = lambda t, i: (t[i] if i < len(t) else {})
        row = [str(r["well"]), str(r["n_records"]), f"{r['modal_err_deg']:.4f}", f"{r['modal_err_std']:.4f}",
               f"{loo['mean_without_well']:.4f}", f"{loo['pull_deg']:+.4f}", f"{loo['overall_mean']:.4f}"]
        for i in range(3):
            g = pad(tg, i)
            row += [f"{g.get('truth_dip', 'NA')}", f"{g.get('truth_dipdir', 'NA')}",
                    f"{g.get('pred_dip', 'NA')}", f"{g.get('pred_dipdir', 'NA')}",
                    f"{g.get('err_deg', 'NA')}"]
        csv_lines.append(",".join(row))
    csv_path = os.path.join(ROOT, "results/v_r98_per_well_table.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"  CSV: {csv_path}")

    # MD 渲染
    md = []
    md.append("# R98 北山逐井对比明细表")
    md.append("")
    md.append(f"- K = {K}, 10 seeds × 22 井, 每井 40 条")
    md.append(f"- **{BLIND_CALIBER*100:.0f}% 盲测档**: 只给看 {BLIND_CALIBER*100:.0f}% 观测、藏 {100-BLIND_CALIBER*100:.0f}% (刁难口径), "
              f"全井均值 = **{overall_mean:.4f}°** (锚点 9.82°±0.7)")
    md.append(f"- leave-one-well-out 拉动: pull = 全井均值 − 剔除该井后均值 (正 = 该井拉高均值 / 负 = 拉低)")
    md.append("")
    md.append("| 井号 | 条数 | modal_err(°) | 剔除后均值(°) | 拉动(°) | 主方向 vs 真值前三组 (倾向/倾角/差角) |")
    md.append("|---|---|---|---|---|---|")
    for r in well_rows:
        loo = r["leave_one_out"]
        tg = r["top3_groups"]
        tg_strs = []
        for g in tg:
            tg_strs.append(f"真组{g['truth_group']}(n={g['truth_n']}): "
                           f"倾{g['truth_dip']}°/向{g['truth_dipdir']}° vs 预{g['pred_dip']}°/向{g['pred_dipdir']}° 差{g['err_deg']}°")
        md.append(f"| {r['well']} | {r['n_records']} | {r['modal_err_deg']:.4f} | "
                  f"{loo['mean_without_well']:.4f} | {loo['pull_deg']:+.4f} | {'; '.join(tg_strs)} |")
    md.append("")
    md_path = os.path.join(ROOT, "results/v_r98_per_well_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"  MD: {md_path}")

    # ---- 敏感性曲线 PNG ----
    of_arr = np.array(OBS_FRAC_LIST)
    mean_arr = np.array([sensitivity[str(o)]["modal_err_mean"] for o in OBS_FRAC_LIST])
    std_arr = np.array([sensitivity[str(o)]["modal_err_std"] for o in OBS_FRAC_LIST])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(of_arr, mean_arr, yerr=std_arr, marker="o", color="#d62728",
                capsize=5, linewidth=2, markersize=6, label="modal_err (K=12)")
    # 1σ 误差带
    ax.fill_between(of_arr, mean_arr - std_arr, mean_arr + std_arr,
                    color="#d62728", alpha=0.15, label="±1σ")
    # 锚点标注 40% 盲测档
    ax.axvline(BLIND_CALIBER, color="#888888", linestyle="--", alpha=0.6)
    ax.text(BLIND_CALIBER + 0.01, ax.get_ylim()[1] if False else max(mean_arr) + 0.3,
            f"{BLIND_CALIBER*100:.0f}% 盲测档", color="#555555", fontsize=10)
    ax.set_xlabel("观测比例 obs_frac")
    ax.set_ylabel("组系表模态误差 (°)")
    ax.set_title("R98 北山观测比例敏感性 (K=12, 22 井 × 10 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    png_path = os.path.join(ROOT, "results/v_r98_sensitivity.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG: {png_path}")

    print("\n" + "=" * 72)
    print("R98 完成。敏感性与 deployment_bonus 对照 diff 见 JSON "
          f"(all_within_tolerance={all_in_tol})。")
    print("=" * 72)


if __name__ == "__main__":
    main()
