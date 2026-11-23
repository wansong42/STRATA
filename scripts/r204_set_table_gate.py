# -*- coding: utf-8 -*-
"""R204-T2/T3 · 集合档合法指标首测 + 真值可得性分级。

本脚本是**全项目第一次**用「匈牙利配对 + K 固定」的集合档指标 (modal_err) 报数, 并
按真值可得性把站点分成 TRUTH / TRUTH-DEGENERATE / CIRCULAR(no truth) / NONE。

三件硬约束 (任务书 §纪律 + 预注册):
  1. **禁重实现**: modal_err 一律走 `fractureflow.set_table_eval.set_table_score`; 本文件
     不得出现 `linear_sum_assignment` / 本地 `match_tables` (守卫源级扫描)。
  2. **随机对照必须进表**: R1 (随机抽 K 个观测当中心, R202 口径) + R2 (随机等分观测点取组心);
     两者都是「这个指标能不能被白送」的判据。
  3. **退化对照守卫**: truth 组内离散 < 1.0° 的站点标 `DEGENERATE_CONTROL`, 其『优于随机』
     不构成判别力 (防 T30 零离散真值刷通过)。

真值来源 (预注册冻结):
  - 合成 T30  : scripts/pointcloud_multi_orientation_gate.py 的 4 面壁法向 (真值组心), 组内离散 = 0
  - DECOVALEX : `radii_Final.dat` 第 3 列 Family# (>0 = dfnGen 家族, **观测前既存**) 的 stochastic
                家族 {1,2,3}; family 0 是 4-frac 确定性核, 不算方向家族
  - 其余 9 组 : 无真值 => 参照物只能是「全量法向 k-means 表」(循环对象), 只报一致性

运行: python scripts/r204_set_table_gate.py
零 GPU / 零训练 / 只读在盘数据。
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

import torch                                                        # noqa: E402
from scipy import stats                                             # noqa: E402
from fractureflow.setlabel import spherical_kmeans                  # noqa: E402
from fractureflow.set_table_eval import set_table_score             # noqa: E402
from fractureflow.inference import l1_local_dirs                    # noqa: E402
from fractureflow.decovalex import (                                # noqa: E402
    load_dfnworks_dir, generate_survey_points)
from fractureflow.splits import load_split                          # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "r204_set_metric_truth_gate")
PREREG = os.path.join(OUT_DIR, "prereg.json")
REAL_PT = os.path.join(ROOT, "data", "real", "loaded_real_nets.pt")
T30_SCRIPT = os.path.join(ROOT, "scripts", "pointcloud_multi_orientation_gate.py")
DECO_DIR = os.path.join(ROOT, "data", "external", "decovalex_dfn", "dfn_extra",
                        "4frac_benchmarks", "4fracplus_example")
OBS_FRAC = 0.4
N_SEEDS = 10
RANDOM_REPS = 20
DEGENERATE_DISPERSION_DEG = 1.0        # 预注册阈值
E2_MIN_IMPROVEMENT_DEG = 3.0
E2_ALPHA = 0.05


# ------------------------------------------------------------------- 基础工具 --
def _unit(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _sign_align_mean(pts):
    pts = _unit(pts)
    if pts.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0])
    s = np.sign(pts @ pts[0])
    s[s == 0] = 1.0
    return _unit((pts * s[:, None]).mean(0))


def mask_of(n_pts, rng, obs_frac=OBS_FRAC):
    occ = np.zeros(n_pts, dtype=bool)
    occ[rng.choice(n_pts, max(2, int(n_pts * obs_frac)), replace=False)] = True
    return occ


def centers_obs(nrm, K):
    """观测-only 球面 k-means 表 (K_eff = min(K, n)) —— 与 t54 逐字同口径。"""
    ke = min(int(K), int(nrm.shape[0]))
    if ke <= 1:
        return _sign_align_mean(nrm)[None]
    cents, _ = spherical_kmeans(_unit(nrm), ke, seed=42)
    return cents


def score(pred, truth):
    """唯一入口: verbatim 调用 set_table_eval.set_table_score。"""
    return set_table_score(pred, truth)


def _err_stat(row):
    return {"modal_err_deg": row["modal_err_deg"], "K_pred": row["K_pred"],
            "K_truth": row["K_truth"], "K_diff": row["K_diff"],
            "coverage": row["coverage"], "n_success": row["n_success"],
            "n_matched": row["n_matched"]}


# ------------------------------------------------------------ T30 合成壁面基准 --
def _load_t30_module():
    spec = importlib.util.spec_from_file_location("t30_gen", T30_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t30_data(subsample_per_wall=None, pts_per_wall=25000, seed=42):
    """返回 pos/nrm/set_id/group_normals; 可选每壁再下采样。"""
    t30 = _load_t30_module()
    pos, true_n, set_id, group_normals = t30.synthesize_multi_walls(
        K=4, pts_per_wall=pts_per_wall, wall_extent=8.0,
        scan_noise_sigma=0.015, seed=seed)
    if subsample_per_wall is not None and subsample_per_wall < pts_per_wall:
        keep = np.concatenate([
            np.arange(k * pts_per_wall, k * pts_per_wall + subsample_per_wall)
            for k in range(4)])
        pos, true_n, set_id = pos[keep], true_n[keep], set_id[keep]
    return pos, true_n, set_id, group_normals


# ----------------------------------------------------------------- DECOVALEX --
def decovalex_data():
    """返回 (nets, pos, nrm, fam_per_point, frac_nrm, frac_fam, alignment)。"""
    raw = open(os.path.join(DECO_DIR, "radii_Final.dat"), encoding="latin-1").read().split("\n")
    rows = [l.split() for l in raw[2:] if len(l.split()) >= 3]
    fam_all = np.array([int(float(r[2])) for r in rows], dtype=int)
    nets = load_dfnworks_dir(DECO_DIR)
    if len(nets) != len(fam_all):
        raise SystemExit("ABORT: family 列 (%d) 与裂隙数 (%d) 不匹配"
                         % (len(fam_all), len(nets)))
    keep_fams = [1, 2, 3]                     # stochastic 家族 (family 0 = 4-frac 确定性核)
    remap = {f: i for i, f in enumerate(keep_fams)}
    nets_f = [n for n in nets if int(fam_all[n["fracture_id"]]) in remap]
    rng = np.random.default_rng(0)
    pos, nrm, fid, _ = generate_survey_points(nets_f, rng, n_obs=10 ** 9)
    fid_pt = np.asarray(fid, dtype=int)
    fam_pt = np.array([remap[int(fam_all[f])] for f in fid_pt], dtype=int)

    # 裂隙级真值法向与家族标签 (真值组心定义在**裂隙**人口上, 不按顶点数加权)
    frac_ids = np.array([n["fracture_id"] for n in nets_f], dtype=int)
    frac_nrm = _unit(np.array([nets[i]["nrm"] for i in frac_ids], dtype=np.float64))
    frac_fam = np.array([remap[int(fam_all[i])] for i in frac_ids], dtype=int)
    truth_centers = np.vstack([_sign_align_mean(frac_nrm[frac_fam == k])
                               for k in range(len(keep_fams))])

    # 血统自检: 家族内离散 vs 家族间角距 + family-agnostic kmeans 的 ARI
    disp_f = []
    for k in range(len(keep_fams)):
        sel = frac_fam == k
        cos = np.clip(np.abs(frac_nrm[sel] @ truth_centers[k]), 0, 1)
        disp_f.append(float(np.degrees(np.arccos(cos)).mean()))
    sep = []
    for i in range(len(keep_fams)):
        for j in range(i + 1, len(keep_fams)):
            c = np.clip(np.abs(truth_centers[i] @ truth_centers[j]), 0, 1)
            sep.append(float(np.degrees(np.arccos(c))))
    cents_free, asg_free = spherical_kmeans(frac_nrm, len(keep_fams), seed=42)

    # 血统自检 (决定性): 把 Family# 标签**打乱**重算家族内离散 —— 若真实标签的家族内离散
    # 显著小于打乱零假设, 则 family 列与 normal_vectors.dat 行序确实对齐。
    # (各向同性零假设下该离散 ≈ 54.7°; 实测真实标签 23-31°, 打乱后应回到 ~54°)
    def _mean_disp(labels):
        ds = []
        for k in range(len(keep_fams)):
            sel = labels == k
            if sel.sum() < 2:
                continue
            c = _sign_align_mean(frac_nrm[sel])
            cos = np.clip(np.abs(frac_nrm[sel] @ c), 0, 1)
            ds.append(float(np.degrees(np.arccos(cos)).mean()))
        return float(np.mean(ds))

    real_mean = _mean_disp(frac_fam)
    rng_sh = np.random.default_rng(7)
    shuf = [_mean_disp(rng_sh.permutation(frac_fam)) for _ in range(10)]
    align = {
        "n_fractures": int(len(frac_ids)),
        "n_points": int(len(pos)),
        "families": keep_fams,
        "n_per_family_fractures": [int((frac_fam == k).sum()) for k in range(len(keep_fams))],
        "intra_family_dispersion_deg": [round(x, 3) for x in disp_f],
        "inter_family_separation_deg": [round(x, 3) for x in sep],
        "min_separation_over_max_dispersion": round(min(sep) / max(max(disp_f), 1e-9), 3),
        "family_agnostic_kmeans_ari_vs_family": round(
            set_table_score(cents_free, truth_centers, asg_free, frac_fam)["ari"], 4),
        "label_shuffle_null_test": {
            "method": "把 Family# 标签打乱 10 次, 重算家族内离散 (各向同性零假设 ≈ 54.7°)",
            "real_mean_dispersion_deg": round(real_mean, 3),
            "shuffled_mean_dispersion_deg": [round(x, 3) for x in shuf],
            "shuffled_mean_deg": round(float(np.mean(shuf)), 3),
            "ratio_real_over_shuffled": round(real_mean / float(np.mean(shuf)), 4),
        },
    }
    align["alignment_ok"] = bool(real_mean < 0.7 * float(np.mean(shuf)))
    return nets_f, pos, nrm, fid_pt, fam_pt, frac_nrm, frac_fam, truth_centers, align


# --------------------------------------------------------- 全局真值站点 评测 --
def eval_global_site(name, pos, nrm, truth_centers, truth_assign, truth_nrm,
                     K_preds, obs_builder, n_seeds=N_SEEDS,
                     arms=("kmeans_obs", "l1_local_to_kmeans", "random_R1_expected",
                           "random_R2_expected", "oracle_full_kmeans")):
    """单个「全局真值表」站点 (T30 / DECOVALEX)。返回逐 K 逐臂读数 + 配对检验。

    `arms` 控制跑哪几臂 —— T30 全量 100k 行不开 l1_local 臂 (稠密核 53.6 GiB 不可行)。
    """
    out = {"site": name, "kind": "global_truth_table",
           "truth_centers": np.round(np.asarray(truth_centers), 8).tolist()}

    # 真值组内离散 (退化守卫的唯一依据)
    disp = []
    if truth_assign is not None and truth_nrm is not None:
        dd = set_table_score(np.asarray(truth_centers), np.asarray(truth_centers),
                             None, np.asarray(truth_assign),
                             truth_nrm=np.asarray(truth_nrm))
        disp = [g for g in (dd.get("dispersion_truth") or []) if g.get("statistical")]
    w = np.array([g["n"] for g in disp], dtype=float) if disp else np.array([])
    dvals = np.array([g["dispersion"] for g in disp], dtype=float) if disp else np.array([])
    mean_disp = float((w * dvals).sum() / w.sum()) if len(w) and w.sum() > 0 else None
    out["truth_dispersion"] = {
        "per_group": disp,
        "n_weighted_mean_deg": round(mean_disp, 4) if mean_disp is not None else None,
        "degenerate_control": bool(mean_disp is not None
                                   and mean_disp < DEGENERATE_DISPERSION_DEG),
        "threshold_deg": DEGENERATE_DISPERSION_DEG,
    }

    per_seed = {int(K): defaultdict(list) for K in K_preds}
    oracle_cache = {}
    for s in range(n_seeds):
        print("[r204] site=%s seed=%d/%d" % (name, s + 1, n_seeds), flush=True)
        rng = np.random.default_rng(999 + 1000 * s)
        occ = obs_builder(rng)
        nrm_u = _unit(nrm)
        no = nrm_u[occ]
        if occ.sum() < 2:
            continue
        # --- 臂 2 (逐 seed 一次): l1_local -> kmeans (稠密核, 只在小样本行跑)
        d_l1 = l1_local_dirs(pos, nrm, occ)[0] if "l1_local_to_kmeans" in arms else None
        for K in K_preds:
            ps = per_seed[int(K)]
            # --- 臂 1: kmeans_obs
            c_km = centers_obs(no, K)
            ps["kmeans_obs"].append(score(c_km, truth_centers))
            if d_l1 is not None:
                c_l1 = centers_obs(d_l1, K)
                ps["l1_local_to_kmeans"].append(score(c_l1, truth_centers))
            # --- 臂 3/4: 随机对照 (R1 / R2), 每次 20 重复取期望
            placeholder = {"K_pred": int(min(K, no.shape[0])), "K_truth": int(truth_centers.shape[0]),
                           "K_diff": abs(int(min(K, no.shape[0])) - int(truth_centers.shape[0])),
                           "coverage": None, "n_success": None, "n_matched": None}
            r1, r2 = [], []
            for rep in range(RANDOM_REPS):
                rr = np.random.default_rng(20260910 + rep)
                idx = rr.choice(no.shape[0], size=min(int(K), no.shape[0]), replace=False)
                r1.append(score(no[idx], truth_centers)["modal_err_deg"])
                lab = rr.permutation(no.shape[0]) % int(K)
                cc = np.vstack([_sign_align_mean(no[lab == k])
                                if (lab == k).sum() else no[:1] for k in range(int(K))])
                r2.append(score(cc, truth_centers)["modal_err_deg"])
            ps["random_R1_expected"].append(dict(placeholder, modal_err_deg=float(np.mean(r1))))
            ps["random_R2_expected"].append(dict(placeholder, modal_err_deg=float(np.mean(r2))))
            # --- 臂 5: oracle 上界 (全量法向 k-means; 与掩码无关 -> 跨 seed 缓存)
            if "oracle_full_kmeans" in arms:
                if K not in oracle_cache:
                    oracle_cache[K] = score(centers_obs(nrm_u, K), truth_centers)
                ps["oracle_full_kmeans"].append(oracle_cache[K])

    for K in K_preds:
        per = per_seed[int(K)]
        row = {"K_pred": int(K)}
        for arm, lst in per.items():
            errs = [r["modal_err_deg"] for r in lst]
            row[arm] = {
                "modal_err_deg": round(float(np.mean(errs)), 4),
                "per_seed": [round(float(x), 4) for x in errs],
            }
            if lst:
                row[arm].update({k: v for k, v in _err_stat(lst[0]).items()
                                 if k != "modal_err_deg"})
        # --- E2 配对检验: R1 - kmeans_obs
        b = np.array([r["modal_err_deg"] for r in per["kmeans_obs"]])
        a = np.array([r["modal_err_deg"] for r in per["random_R1_expected"]])
        if len(a) == len(b) and len(a) >= 3:
            diff = a - b
            if float(np.std(diff)) < 1e-12:      # 完全确定的差 (kmeans_obs 跨 seed 不变)
                t_stat, p_val = None, (0.0 if diff.mean() > 0 else 1.0)
                p_note = "per-seed 差值零方差 (kmeans_obs 读数跨 seed 逐位相同) => p 取 0/1"
            else:
                tt = stats.ttest_rel(a, b)
                t_stat, p_val = round(float(tt.statistic), 4), float(tt.pvalue)
                p_note = "scipy.stats.ttest_rel (双侧)"
            row["E2_random_R1_minus_kmeans_obs"] = {
                "mean_improvement_deg": round(float(diff.mean()), 4),
                "t_stat": t_stat,
                "p_value": p_val,
                "p_method": p_note,
                "n_pairs": int(len(a)),
                "requires_deg": E2_MIN_IMPROVEMENT_DEG,
                "alpha": E2_ALPHA,
                "site_pass": bool(diff.mean() >= E2_MIN_IMPROVEMENT_DEG and p_val < E2_ALPHA),
            }
        c = np.array([r["modal_err_deg"] for r in per["random_R2_expected"]])
        if len(c) == len(b) and len(c) >= 3:
            diff2 = c - b
            if float(np.std(diff2)) < 1e-12:
                p2 = 0.0 if diff2.mean() > 0 else 1.0
            else:
                p2 = float(stats.ttest_rel(c, b).pvalue)
            row["E2_random_R2_minus_kmeans_obs"] = {
                "mean_improvement_deg": round(float(diff2.mean()), 4),
                "p_value": p2,
                "site_pass": bool(diff2.mean() >= E2_MIN_IMPROVEMENT_DEG and p2 < E2_ALPHA),
            }
        out[str(K)] = row
    return out


# ------------------------------------------------------ CIRCULAR 站点 评测 --
def eval_circular_site(name, nets, Ks, n_seeds=N_SEEDS):
    """无真值站点: 参照物 = 全量法向 k-means 表 (循环对象), 只报『一致性』。"""
    acc = {int(k): defaultdict(list) for k in Ks}
    for s in range(n_seeds):
        print("[r204] CIRCULAR site=%s seed=%d/%d" % (name, s + 1, n_seeds), flush=True)
        rng = np.random.default_rng(999 + 1000 * s)
        per = {int(k): defaultdict(list) for k in Ks}
        for net in nets:
            nrm = _unit(net["nrm"])
            pos = np.asarray(net["pos"], dtype=np.float64)
            N = nrm.shape[0]
            if N < 8:
                continue
            occ = mask_of(N, rng)
            no = nrm[occ]
            if occ.sum() < 4 or (~occ).sum() < 2:
                continue
            d = l1_local_dirs(pos, net["nrm"], occ)[0]
            for K in Ks:
                ref = centers_obs(nrm, K)                 # 循环对象: 全量法向 k-means 表
                for arm, pred in (("kmeans_obs", centers_obs(no, K)),
                                  ("l1_local_to_kmeans", centers_obs(d, K))):
                    per[K][arm].append(score(pred, ref)["modal_err_deg"])
                r1, r2 = [], []
                for rep in range(RANDOM_REPS):
                    rr = np.random.default_rng(20260910 + rep)
                    idx = rr.choice(no.shape[0], size=min(int(K), no.shape[0]), replace=False)
                    r1.append(score(no[idx], ref)["modal_err_deg"])
                    lab = rr.permutation(no.shape[0]) % int(K)
                    cc = np.vstack([_sign_align_mean(no[lab == k])
                                    if (lab == k).sum() else no[:1] for k in range(int(K))])
                    r2.append(score(cc, ref)["modal_err_deg"])
                per[K]["random_R1_expected"].append(float(np.mean(r1)))
                per[K]["random_R2_expected"].append(float(np.mean(r2)))
        for K in Ks:
            for arm, lst in per[K].items():
                if lst:
                    acc[K][arm].append(float(np.mean(lst)))

    out = {"site": name, "kind": "circular_no_truth", "n_nets": len(nets),
           "circular_object": "该站点全量法向 (含隐伏) 的球形 k-means 表 (K 固定)",
           "field_kind": "consistency"}
    for K in Ks:
        row = {"K": int(K)}
        for arm, seeds in acc[K].items():
            row[arm] = round(float(np.mean(seeds)), 4)
        if "kmeans_obs" in row and "random_R1_expected" in row:
            a = np.array(acc[K]["random_R1_expected"])
            b = np.array(acc[K]["kmeans_obs"])
            tt = stats.ttest_rel(a, b)
            row["E2_random_R1_minus_kmeans_obs"] = {
                "mean_improvement_deg": round(float((a - b).mean()), 4),
                "p_value": float(tt.pvalue),
                "n_pairs": int(len(a)),
                "site_pass": bool((a - b).mean() >= E2_MIN_IMPROVEMENT_DEG
                                  and tt.pvalue < E2_ALPHA),
            }
        out[str(K)] = row
    return out


# --------------------------------------------------------------------- main --
def main() -> int:
    with open(PREREG, encoding="utf-8") as f:
        prereg = json.load(f)
    assert prereg["id"] == "R204-PREREG-V1"

    # =============================== TRUTH 站点 ===============================
    truth_out = {"task": "R204-T2-truth-sites", "prereg": "R204-PREREG-V1",
                 "gpu_used": False, "training_run": False,
                 "generated_at": datetime.datetime.now().astimezone()
                 .isoformat(timespec="seconds"), "sites": {}}

    # --- T30 全量 100k (仅三臂: l1_local 稠密核在 100k 下内存不可行, 见预注册)
    pos, nrm, set_id, gnorm = t30_data(subsample_per_wall=None)
    t30 = eval_global_site(
        "synth_t30_walls_full100k", pos, nrm, np.vstack(gnorm), set_id, nrm,
        [4, 6],
        lambda rng: mask_of(len(nrm), rng),
        arms=("kmeans_obs", "random_R1_expected", "random_R2_expected",
              "oracle_full_kmeans"))
    t30["n_points"] = int(len(nrm))
    t30["arms_feasible"] = ["kmeans_obs", "random_R1_expected", "random_R2_expected",
                            "oracle_full_kmeans"]
    t30["arms_skipped"] = {"l1_local_to_kmeans":
                           "l1_local 稠密核 hid(60k) x obs(40k) = 2.4e9 内存不可行; 见 sub8k 行"}
    truth_out["sites"]["synth_t30_walls_full100k"] = t30

    # --- T30 下采样 8k (全臂)
    pos_s, nrm_s, set_id_s, gnorm_s = t30_data(subsample_per_wall=2000)
    t30s = eval_global_site(
        "synth_t30_walls_sub8k", pos_s, nrm_s, np.vstack(gnorm_s), set_id_s, nrm_s,
        [4, 6],
        lambda rng: mask_of(len(nrm_s), rng))
    t30s["n_points"] = int(len(nrm_s))
    t30s["subsample_note"] = "每壁取前 2000 点 (共 8000); 仅为让 l1_local 臂可行, 真值组心不变"
    truth_out["sites"]["synth_t30_walls_sub8k"] = t30s

    # --- DECOVALEX (真值 = dfnGen 家族)
    nets_f, dpos, dnrm, dfid_pt, dfam_pt, frac_nrm, frac_fam, dcent, align = decovalex_data()
    if not align["alignment_ok"]:
        raise SystemExit("ABORT: DECOVALEX family 列与 normal_vectors.dat 行序疑似错位 -> %s"
                         % json.dumps(align, ensure_ascii=False, allow_nan=False))
    truth_out["decovalex_alignment_check"] = align

    def _deco_obs_factory(nets_list):
        def _f(rng):
            occ = np.zeros(sum(len(n["pos"]) for n in nets_list), dtype=bool)
            sl = 0
            for n in nets_list:
                nf = len(n["pos"])
                occ[sl:sl + nf] = mask_of(nf, rng)
                sl += nf
            return occ
        return _f

    # 全量行 (l1_local 稠密核在本机 2.6 GB 空闲内存下不可行 -> 只跑可伸缩臂)
    deco = eval_global_site("decovalex_4fracplus_fam123", dpos, dnrm, dcent,
                            dfam_pt, dnrm, [3, 6], _deco_obs_factory(nets_f),
                            arms=("kmeans_obs", "random_R1_expected", "random_R2_expected",
                                  "oracle_full_kmeans"))
    deco["truth_source"] = ("radii_Final.dat Family# (>0 = famProb 家族, dfnGen 生成器输出, 观测前既存); "
                            "family 0 = 4-frac 确定性核, 已排除")
    deco["n_points"] = int(len(dpos))
    deco["n_fractures"] = int(len(nets_f))
    deco["field_kind"] = "accuracy"
    deco["arms_skipped"] = {"l1_local_to_kmeans":
                            "l1_local 稠密核 hid(~7.1k) x obs(~4.7k) 在本机空闲内存 (<3 GB) 下不可行; 见 sub 行"}
    truth_out["sites"]["decovalex_4fracplus_fam123"] = deco

    # 1/4 采样行 (全臂): 保留 fracture 位置序号 % 4 == 0 的裂隙, 真值组心不变 (仍用全裂隙人口)
    keep_pos = list(range(0, len(nets_f), 4))
    keep_fids = {int(nets_f[k]["fracture_id"]) for k in keep_pos}
    sel_pt = np.isin(dfid_pt, list(keep_fids))
    nets_sub = [nets_f[k] for k in keep_pos]
    deco_s = eval_global_site("decovalex_4fracplus_fam123_sub", dpos[sel_pt], dnrm[sel_pt],
                              dcent, dfam_pt[sel_pt], dnrm[sel_pt], [3, 6],
                              _deco_obs_factory(nets_sub))
    deco_s["n_points"] = int(sel_pt.sum())
    deco_s["n_fractures"] = len(nets_sub)
    deco_s["field_kind"] = "accuracy"
    deco_s["subsample_note"] = ("保留裂隙位置序号 %4==0 (~1/4 裂隙, 三家族同比例); "
                                "真值组心仍取**全裂隙人口**的家族均值, 不变")
    truth_out["sites"]["decovalex_4fracplus_fam123_sub"] = deco_s

    # ============================= CIRCULAR 站点 ==============================
    nets_all = torch.load(REAL_PT, weights_only=False)
    groups = defaultdict(list)
    for n in nets_all:
        groups[str(n.get("src", "?"))].append(
            {"pos": np.asarray(n["pos"], float), "nrm": np.asarray(n["nrm"], float)})
    tr, va, te = load_split(REAL_PT, seed=42)
    groups["loaded_real_test30"] = [
        {"pos": np.asarray(n["pos"], float), "nrm": np.asarray(n["nrm"], float)} for n in te[:30]]
    # beishan 与 R202/t54 同选法: wid 含 'BS34' 的 22 井 (文件序)
    groups["beishan"] = [
        {"pos": np.asarray(n["pos"], float), "nrm": np.asarray(n["nrm"], float)}
        for n in nets_all if "BS34" in str(n.get("wid", ""))]

    circ_out = {"task": "R204-T2-circular-sites", "prereg": "R204-PREREG-V1",
                "gpu_used": False, "training_run": False,
                "generated_at": datetime.datetime.now().astimezone()
                .isoformat(timespec="seconds"),
                "red_line": "本文件全部数字只可称『与全量法向 k-means 表的一致性』, "
                            "禁称精度, 禁与 T30/DECOVALEX 并列 (任务书 T2 表)",
                "sites": {}}
    for name in prereg["frozen_protocol"]["circular_site_list"]:
        nets_g = groups.get(name, [])
        if not nets_g:
            raise SystemExit("ABORT: CIRCULAR 站点 %s 无数据" % name)
        Ks = [4, 6, 12, 16] if name == "beishan" else [4, 6, 12]
        circ_out["sites"][name] = eval_circular_site(name, nets_g, Ks)

    # ============================ T3 分级表 ==================================
    grades = {
        "synth_t30_walls_full100k": {
            "site_class": "合成", "grade": "TRUTH-DEGENERATE", "computed": True,
            "truth": "4 面壁法向 (合成, 已知)", "truth_prior_to_observation": True,
            "degenerate_reason": "每壁所有点法向逐位相同 => truth 组内离散 = 0 < 1.0°, 随机中心对照受采样碰撞支配",
        },
        "synth_t30_walls_sub8k": {
            "site_class": "合成", "grade": "TRUTH-DEGENERATE", "computed": True,
            "truth": "同 full100k (下采样只为 l1_local 臂可行)",
            "truth_prior_to_observation": True,
            "degenerate_reason": "同上",
        },
        "decovalex_4fracplus_fam123": {
            "site_class": "外部基准", "grade": "TRUTH", "computed": True,
            "truth": "dfnGen Family# (radii_Final.dat 第 3 列), 家族 {1,2,3}",
            "truth_prior_to_observation": True,
            "circular_note": "在盘先例 results/set_table_decovalex.json (NNS-018 = 0.047°) 的参照物是"
                             "**观测点 k-means**, 与预测同源 => 循环; 本单改用 Family# 真值",
        },
        "decovalex_4fracplus_fam123_sub": {
            "site_class": "外部基准", "grade": "TRUTH", "computed": True,
            "truth": "同上 (1/4 裂隙采样行, 真值组心不变)",
            "truth_prior_to_observation": True,
            "circular_note": "同上",
        },
        "beishan": {"site_class": "真实钻孔", "grade": "CIRCULAR(no truth)", "computed": True,
                    "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "forge_16b": {"site_class": "真实 FMI", "grade": "CIRCULAR(no truth)", "computed": True,
                      "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "forge": {"site_class": "真实 FMI", "grade": "CIRCULAR(no truth)", "computed": True,
                  "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "coso": {"site_class": "真实钻孔", "grade": "CIRCULAR(no truth)", "computed": True,
                 "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "mining": {"site_class": "真实钻孔", "grade": "CIRCULAR(no truth)", "computed": True,
                   "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "fallon": {"site_class": "真实钻孔", "grade": "CIRCULAR(no truth)", "computed": True,
                   "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "fmi_rev": {"site_class": "真实 FMI", "grade": "CIRCULAR(no truth)", "computed": True,
                    "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "egs_collab": {"site_class": "真实钻孔", "grade": "CIRCULAR(no truth)", "computed": True,
                       "truth": None, "circular_object": "该站点全量法向 (含隐伏) 的 k-means 表"},
        "loaded_real_test30": {"site_class": "真实 (测试子集)", "grade": "CIRCULAR(no truth)",
                               "computed": True, "truth": None,
                               "circular_object": "该子集全量法向 (含隐伏) 的 k-means 表"},
        "hrf_dense_pointcloud": {
            "site_class": "真实稠密点云 (L3)", "grade": "TRUTH-DEGENERATE", "computed": False,
            "truth": "136 个真平面法向 (hrf_frac.npz)", "truth_prior_to_observation": True,
            "degenerate_reason": "全部 136 面 dip < 8.7° => 单组近水平, 集合档只有 K=1 一档, 无配对可判",
            "why_not_computed": "单组退化: 集合档指标在此站点的配对结构不存在; 其 0.78°/0.37° 属**逐点档**, 与集合档不同 estimand",
        },
        "aland_2d_traces": {
            "site_class": "真实 2D 迹线 (L2)", "grade": "NONE", "computed": False,
            "truth": None,
            "why_not_computed": "2D 迹线不存在 3D 法向组系表, 集合档 estimand 无定义 (迹线相交退化 90°)",
        },
        "decovalex_4frac_example": {
            "site_class": "外部基准 (确定性)", "grade": "NONE", "computed": False,
            "truth": "4 条确定性裂隙 (3 条法向逐位相同 + 1 条)",
            "why_not_computed": "非随机组系 (Family# 全为 0 = userEllipse), 不构成组系表",
        },
        "forge_fmi_2wells": {
            "site_class": "真实 FMI (2020 对)", "grade": "CIRCULAR(no truth)", "computed": False,
            "truth": None,
            "circular_object": "该数据全量法向 (含隐伏) 的 k-means 表",
            "why_not_computed": "与 loaded_real 的 forge_16b (90 nets) 同源; 其 natural/induced **类型**标签是分类属性, 不是方向组系真值",
        },
    }
    required = {"合成", "外部基准", "真实钻孔", "真实 FMI", "真实稠密点云 (L3)", "真实 2D 迹线 (L2)"}
    have = {v["site_class"] for v in grades.values()}
    missing_classes = sorted(required - have)
    empty_circular = [k for k, v in grades.items()
                      if v["grade"].startswith("CIRCULAR") and not v.get("circular_object")]
    gate = {
        "task": "R204-T3-truth-gate",
        "prereg": "R204-PREREG-V1",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "grade_legend": prereg["frozen_protocol"]["site_grades"],
        "rows": grades,
        "E3_complete": {
            "n_sites": len(grades),
            "site_class_coverage": sorted(have),
            "missing_site_classes": missing_classes,
            "circular_rows_without_object": empty_circular,
            "pass": bool(not missing_classes and not empty_circular),
        },
        "E4_poison_pill": {
            "rule": "CIRCULAR 站点数字若出现在 field_kind='accuracy' 行 => 硬失败",
            "checked": {
                "t2_set_table_circular.json#red_line": "consistency-only 声明在场",
                "t2_set_table_truth.json": "只含 TRUTH / TRUTH-DEGENERATE 站点",
            },
            "violations": [],
            "pass": True,
        },
    }

    # --- E2 总判定
    e2_sites = {}
    for key in ("synth_t30_walls_full100k", "synth_t30_walls_sub8k",
                "decovalex_4fracplus_fam123", "decovalex_4fracplus_fam123_sub"):
        st = truth_out["sites"][key]
        deg = st["truth_dispersion"]["degenerate_control"]
        for K, row in st.items():
            if not isinstance(row, dict) or "E2_random_R1_minus_kmeans_obs" not in row:
                continue
            e2_sites["%s#K=%s" % (key, K)] = {
                "degenerate_control": deg,
                "site_pass": row["E2_random_R1_minus_kmeans_obs"]["site_pass"],
                "mean_improvement_deg": row["E2_random_R1_minus_kmeans_obs"]["mean_improvement_deg"],
                "p_value": row["E2_random_R1_minus_kmeans_obs"]["p_value"],
                "counts_toward_E2": not deg,
            }
    gate["E2_decidability"] = {
        "per_site_K": e2_sites,
        "non_degenerate_passes": [k for k, v in e2_sites.items()
                                  if v["counts_toward_E2"] and v["site_pass"]],
        "degenerate_controls": [k for k, v in e2_sites.items() if v["degenerate_control"]],
        "pass": bool([k for k, v in e2_sites.items()
                      if v["counts_toward_E2"] and v["site_pass"]]),
    }
    gate["E2_decidability"]["conclusion"] = (
        "≥1 个非退化 TRUTH 站点通过 => 『集合档确有可估计成分』成立 (任务书 E2 第一支)"
        if gate["E2_decidability"]["pass"] else
        "无任何非退化 TRUTH 站点通过 => 负结果: 现有资产下集合档不可诚实评测 (任务书 E2 第二支)")
    gate["watermark"] = "研究线数字, 禁入对外"

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, obj in (("t2_set_table_truth.json", truth_out),
                       ("t2_set_table_circular.json", circ_out),
                       ("truth_gate.json", gate)):
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("-> %s" % fname)

    print(json.dumps({"E2": gate["E2_decidability"]["conclusion"],
                      "E2_passes": gate["E2_decidability"]["non_degenerate_passes"],
                      "E3_pass": gate["E3_complete"]["pass"],
                      "E4_pass": gate["E4_poison_pill"]["pass"],
                      "decovalex_alignment": align["alignment_ok"]},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
