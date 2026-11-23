# -*- coding: utf-8 -*-
"""B2-B3: DECOVALEX 4-frac_plus 组系表评测."""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fractureflow.set_table_eval import set_table_score, estimate_set_table
from fractureflow.decovalex import load_dfnworks_dir, generate_survey_points
from fractureflow.setlabel import _sign_align, spherical_kmeans

print("=" * 60)
print("B2-B3: DECOVALEX 4-frac_plus")
print("=" * 60)

base = "data/external/decovalex_dfn/dfn_extra/4frac_benchmarks"
nets = load_dfnworks_dir(os.path.join(base, "4fracplus_example"))
print(f"Fractures: {len(nets)}")

# 生成调查点
n_obs = 5
n_seeds = 10
K_list = [4, 6, 8, 12]

results = {}
for K in K_list:
    seed_errs = []
    for s in range(n_seeds):
        rng_s = np.random.default_rng(999 + 1000 * s)
        all_pos, all_nrm, all_fid, mask_s = generate_survey_points(nets, rng_s, n_obs=n_obs)
        occ = mask_s.astype(bool)

        if occ.sum() < 2 or (~occ).sum() == 0:
            continue

        # 预测: 观测-only kmeans (盲化法向)
        nrm_blind = all_nrm.copy()
        nrm_blind[~occ] = 0.0  # 隐伏点设为 0 (不参与 k-means)
        pred = estimate_set_table(all_pos, nrm_blind, occ, K=K, seed=42, strict=True)

        # 真值: fracture_id 定义真组
        # 1089 个 fracture -> 太多组; 归并到 K 组评测
        n_frac = len(nets)
        truth_by_frac = np.zeros((n_frac, 3))
        for fi in range(n_frac):
            sel = all_fid == fi
            if sel.sum():
                pts = all_nrm[sel]
                ps = _sign_align(pts)
                truth_by_frac[fi] = ps.mean(0)
                truth_by_frac[fi] /= np.linalg.norm(truth_by_frac[fi]) + 1e-12

        # 把 1089 个 fracture 法向归并到 K 组 (用观测点)
        obs_nrm_by_frac = []
        obs_fi = []
        for fi in range(n_frac):
            sel = (all_fid == fi) & occ
            if sel.sum():
                pts = all_nrm[sel]
                ps = _sign_align(pts)
                obs_nrm_by_frac.append(ps.mean(0))
                obs_fi.append(fi)
        if len(obs_nrm_by_frac) < K:
            continue
        obs_nrm_by_frac = np.array(obs_nrm_by_frac)
        obs_nrm_by_frac /= np.linalg.norm(obs_nrm_by_frac, axis=1, keepdims=True) + 1e-12

        # 观测-only k-means 把 fracture 归并到 K 组
        _, assign_frac = spherical_kmeans(obs_nrm_by_frac, K, seed=42)
        truth_centers = np.zeros((K, 3))
        for k in range(K):
            sel = assign_frac == k
            if sel.sum():
                # 这些 fracture 的真值中心
                frac_centers = obs_nrm_by_frac[sel]
                ref = frac_centers[0]
                sgn = np.sign((frac_centers * ref).sum(-1, keepdims=True))
                sgn[sgn == 0] = 1
                truth_centers[k] = (frac_centers * sgn).mean(0)
                truth_centers[k] /= np.linalg.norm(truth_centers[k]) + 1e-12

        score = set_table_score(pred["centers"], truth_centers)
        seed_errs.append(score["modal_err_deg"])

    if seed_errs:
        mean_err = float(np.mean(seed_errs))
        std_err = float(np.std(seed_errs, ddof=1))
        results[K] = {"modal_err_mean": round(mean_err, 3),
                      "modal_err_std": round(std_err, 3),
                      "n_seeds": len(seed_errs)}
        print(f"  K={K:2d}: modal_err = {mean_err:.2f} +/- {std_err:.2f} deg")
    else:
        print(f"  K={K:2d}: no valid seeds")

# 保存
out_path = os.path.join(ROOT, "results/set_table_decovalex.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, allow_nan=False)
print(f"\n结果: {json.dumps(results, indent=2, allow_nan=False)}")
print(f"保存: {out_path}")
