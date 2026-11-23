# -*- coding: utf-8 -*-
"""Phase 0.1: Oracle 地板曲线 (误差预算总表).

在 beishan 22 井 + loaded_real 抽样 30 井上, 计算泄漏 oracle
(真值分组 + 组心解码) 在 K = 4,6,8,10,12,14,16,20 下的理论底,
10 掩码 seed 稳健性.

这是整个攻坚计划的误差预算总表 —— 后续每个阶段宣称的增益,
都能在这条曲线上找到对应位置.

用法:
  python scripts/phase0_oracle_floor.py
  python scripts/phase0_oracle_floor.py --seeds 10 --Ks 4,6,8,10,12,14,16,20
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _sign_align(pts):
    """无向法向符号对齐."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def spherical_kmeans(nrm, K, iters=80, restarts=10, seed=42):
    """球面 k-means, 多重启取最优. 返回 (centers[K,3], assign[L])."""
    rng = np.random.default_rng(seed)
    pts = _unit(nrm)
    N = len(pts)
    K = min(K, N)
    if K <= 1:
        c = _unit(_sign_align(pts).mean(0))[None]
        return c, np.zeros(N, dtype=int)

    best_obj = -1e18
    best_centers = None
    for r in range(restarts):
        idx = rng.choice(N, size=K, replace=False)
        mu = pts[idx].copy()
        for _ in range(iters):
            cos = np.abs(pts @ mu.T)
            lab = cos.argmax(1)
            newmu = np.zeros_like(mu)
            for k in range(K):
                m = lab == k
                if m.sum() == 0:
                    newmu[k] = pts[rng.integers(N)]
                else:
                    newmu[k] = _unit(_sign_align(pts[m]).mean(0))
            if np.allclose(newmu, mu, atol=1e-8):
                mu = newmu
                break
            mu = newmu
        obj = np.abs(pts @ mu.T).max(1).sum()
        if obj > best_obj:
            best_obj = obj
            best_centers = mu.copy()

    # 最终指派
    cos = np.abs(pts @ best_centers.T)
    assign = cos.argmax(1)
    return best_centers, assign


def oracle_floor_single(nrm_full, K, mask_rng_seed=999, obs_frac=0.4, strict=False):
    """单井单掩码 seed 的 oracle 误差.

    算法:
      1. 用全量法向 nrm_full 做球面 k-means 分组 (分组本身仍用全量)
      2. 组心 = 组内符号对齐均值
         - strict=False (泄漏 oracle, 默认): 用全量成员算组心 (含隐伏点)
         - strict=True (诚实 oracle): 仅用观测点算组心
      3. 隐伏点 oracle 指派 = argmax |<n_j, T_i>| (用真值挑最近组心)
      4. 误差 = acos(|<group_center, T_i>|)
    """
    nrm_full = _unit(np.asarray(nrm_full, dtype=np.float64))
    L = len(nrm_full)
    if L < K + 1:
        return np.array([])

    rng = np.random.default_rng(mask_rng_seed)
    obs = rng.random(L) < obs_frac
    if obs.sum() < 2 or (~obs).sum() < 1:
        return np.array([])

    # 用全量法向做 k-means (分组本身仍用全量 - 这是 grouping oracle)
    centers, assign = spherical_kmeans(nrm_full, K, seed=42)

    # 组心: strict 仅用观测点, 否则用全量
    if strict:
        centers_obs = np.zeros_like(centers)
        have = np.zeros(K, dtype=bool)
        for k in range(K):
            sel = (assign == k) & obs
            if sel.sum() >= 1:
                centers_obs[k] = _unit(_sign_align(nrm_full[sel]).mean(0))
                have[k] = True
        # 对无观测点的组, 用全局观测均值
        gmean = _unit(nrm_full[obs].mean(0))
        for k in range(K):
            if not have[k]:
                centers_obs[k] = gmean
        centers = centers_obs

    # 隐伏点的 oracle 指派: 用真值法向挑最近组心
    hid_mask = ~obs
    nrm_hid = nrm_full[hid_mask]
    cos_to_centers = np.abs(nrm_hid @ centers.T)  # [Nh, K]
    best_center_idx = cos_to_centers.argmax(1)
    pred = centers[best_center_idx]

    # 误差
    cos_err = np.clip(np.abs((pred * nrm_hid).sum(-1)), 0, 1)
    err = np.rad2deg(np.arccos(cos_err))
    return err


def load_datasets():
    """加载 beishan 22 井 + loaded_real 抽样 30 井."""
    # beishan
    real_pt = os.path.join(ROOT, "data/real/loaded_real_nets_setid.pt")
    d = torch.load(real_pt, weights_only=False)
    bnets = []
    for n in d:
        wid = str(n.get('wid', ''))
        if 'BS34' in wid:
            nn = dict(n)
            if 'nrm_full' not in nn:
                nn['nrm_full'] = np.asarray(nn['nrm'], float).copy()
            bnets.append(nn)

    # loaded_real 抽样 30 井 (test split 的前 30 个)
    from fractureflow.splits import load_split
    real_pt_orig = os.path.join(ROOT, "data/real/loaded_real_nets.pt")
    tr, va, te = load_split(real_pt_orig, seed=42)
    lrnets = []
    for n in te[:30]:
        nn = dict(n)
        if 'nrm_full' not in nn:
            nn['nrm_full'] = np.asarray(nn['nrm'], float).copy()
        lrnets.append(nn)

    return bnets, lrnets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="掩码种子数")
    ap.add_argument("--Ks", type=str, default="4,6,8,10,12,14,16,20",
                    help="K 网格 (逗号分隔)")
    ap.add_argument("--obs-frac", type=float, default=0.4)
    args = ap.parse_args()

    Ks = [int(x) for x in args.Ks.split(",")]

    print("=" * 70)
    print("Phase 0.1: Oracle 地板曲线 (leaky vs strict 对照)")
    print("=" * 70)

    bnets, lrnets = load_datasets()
    print(f"beishan 井数: {len(bnets)}  |  loaded_real 抽样: {len(lrnets)}")
    print(f"K 网格: {Ks}")
    print(f"掩码种子数: {args.seeds}")
    print()

    # 对每种口径 (leaky / strict) 算 oracle floor
    for strict_mode, mode_name in [(False, "LEAKY (全量组心)"), (True, "STRICT (仅观测组心)")]:
        print(f"\n{'='*70}")
        print(f"--- {mode_name} ---")
        print(f"{'='*70}")
        results = {}

        for label, nets in [("beishan_22", bnets), ("loaded_real_30", lrnets)]:
            print(f"\n--- {label} ({len(nets)} nets) ---")
            k_results = {}
            for K in Ks:
                seed_errs = []
                for seed_idx in range(args.seeds):
                    mask_seed = 999 + 1000 * seed_idx
                    net_errs = []
                    for net in nets:
                        nrm_full = net.get('nrm_full', net.get('nrm'))
                        if nrm_full is None:
                            continue
                        e = oracle_floor_single(nrm_full, K, mask_seed, args.obs_frac, strict=strict_mode)
                        if len(e) > 0:
                            net_errs.append(e)
                    if net_errs:
                        all_e = np.concatenate(net_errs)
                        seed_errs.append(all_e.mean())

                if seed_errs:
                    k_results[K] = {
                        "mae_mean": round(float(np.mean(seed_errs)), 3),
                        "mae_std": round(float(np.std(seed_errs)), 3),
                        "mae_min": round(float(np.min(seed_errs)), 3),
                        "mae_max": round(float(np.max(seed_errs)), 3),
                        "n_seeds": len(seed_errs),
                    }
                    s = k_results[K]
                    print(f"  K={K:2d}  oracle MAE = {s['mae_mean']:.3f}° "
                          f"± {s['mae_std']:.3f}°  "
                          f"[{s['mae_min']:.3f} ~ {s['mae_max']:.3f}]")
                else:
                    print(f"  K={K:2d}  (no valid data)")

            results[label] = k_results

        # 合并
        print(f"\n--- 合并 (beishan + loaded_real) ---")
        combined = {}
        for K in Ks:
            all_seed_errs = []
            for seed_idx in range(args.seeds):
                mask_seed = 999 + 1000 * seed_idx
                seed_err_list = []
                for nets in [bnets, lrnets]:
                    for net in nets:
                        nrm_full = net.get('nrm_full', net.get('nrm'))
                        if nrm_full is None:
                            continue
                        e = oracle_floor_single(nrm_full, K, mask_seed, args.obs_frac, strict=strict_mode)
                        if len(e) > 0:
                            seed_err_list.append(e.mean())
                if seed_err_list:
                    all_seed_errs.append(np.mean(seed_err_list))
            if all_seed_errs:
                combined[K] = {
                    "mae_mean": round(float(np.mean(all_seed_errs)), 3),
                    "mae_std": round(float(np.std(all_seed_errs)), 3),
                }
                s = combined[K]
                print(f"  K={K:2d}  oracle MAE = {s['mae_mean']:.3f}° "
                      f"± {s['mae_std']:.3f}°")

        results["combined"] = combined

        # 保存
        tag = "strict" if strict_mode else "leaky"
        out_path = os.path.join(ROOT, f"results/phase0_oracle_floor_{tag}.json")
        with open(out_path, "w") as f:
            json.dump({
                "description": f"Phase 0.1 Oracle 地板 ({tag}): {'仅观测组心' if strict_mode else '全量组心(泄漏)'}",
                "strict": strict_mode,
                "Ks": Ks,
                "n_seeds": args.seeds,
                "obs_frac": args.obs_frac,
                "results": results,
            }, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"\n结果已保存: {out_path}")

    # 打印对比总表
    print("\n" + "=" * 70)
    print("对比总表: leaky (泄漏) vs strict (诚实)")
    print("=" * 70)
    leaky = json.load(open(os.path.join(ROOT, "results/phase0_oracle_floor_leaky.json")))
    strict = json.load(open(os.path.join(ROOT, "results/phase0_oracle_floor_strict.json")))
    print(f"{'K':>4s}  {'leaky':>10s}  {'strict':>10s}  {'Δ (泄漏幅度)':>14s}")
    print("-" * 45)
    for K in Ks:
        if K in leaky["results"]["combined"] and K in strict["results"]["combined"]:
            l = leaky["results"]["combined"][K]["mae_mean"]
            s = strict["results"]["combined"][K]["mae_mean"]
            print(f"{K:4d}  {l:10.3f}  {s:10.3f}  {s-l:14.3f}")
    print("\n注: strict = 诚实 oracle 地板 (仅观测点算组心 + 真值指派)")


if __name__ == "__main__":
    main()
