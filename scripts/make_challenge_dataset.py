# -*- coding: utf-8 -*-
"""复现练习出题器: 生成带隐藏真值的合成编录表 (全新种子, 项目从未使用).

用途: 给答辩人/导师一份"可对答案"的练习卷 —— 输入编录表 (depth/dip/
dip_direction, 与客户标准格式同方言), 真值组系封存在 truth_SEALED.json,
只有跑完打标+判分后才允许打开。换自己的 --seed 即可无限出新题。

生成模型 (全部如实登记在 truth 里):
  K 个裂隙组: 组法向拒绝采样 (两两角距 >= min_sep 度);
  每组 Fisher 集中度 kappa ∈ [kappa_lo, kappa_hi]; 组占比 ~ Dirichlet(alpha);
  每口井 n ~ U[n_min, n_max]; 井内组计数 ~ Multinomial(n, share);
  每观测 = 组中心法向 + Fisher(kappa) 噪声 (精确逆 CDF 采样, 非近似);
  产状约定与项目一致: n = [sin d sin a, sin d cos a, cos d], dip_dir=atan2(nx,ny).
  生成后做 产状->法向->产状 往返断言 (<1e-9), 保证出题本身不带坐标 bug.

零新依赖: 仅 numpy + 标准库。输出:
  data/challenge_<seed>/synthetic/well_XX.csv   (输入卷, 不含真值)
  data/challenge_<seed>/synthetic/truth_SEALED.json  (答案, 判分后才看)
  data/challenge_<seed>/synthetic/manifest.json (题面参数, 无答案细节可看)
"""
import argparse
import io
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def n_to_dipdir(n):
    nz = float(np.clip(n[2], -1.0, 1.0))
    dip = math.degrees(math.acos(nz))
    dd = math.degrees(math.atan2(n[0], n[1])) % 360.0
    return dip, dd


def dipdir_to_n(dip_deg, dd_deg):
    d, a = math.radians(dip_deg), math.radians(dd_deg)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def pick_axes(mu, rng):
    """选与 mu 最不共线的坐标轴做叉乘基 (argmin|mu| 法, 防 cross=0 退化)."""
    k = int(np.argmin(np.abs(mu)))
    axis = np.zeros(3)
    axis[k] = 1.0
    e1 = np.cross(mu, axis)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(mu, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def fisher_sample(mu, kappa, n, rng):
    """vMF3 精确采样: t=(1/k)*ln(u*2*sinh(k)+e^{-k}), phi~U(0,2pi)."""
    if kappa < 1e-6:
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.random(n)
    # 数值稳: ln(u*2*sinh k + e^-k) = ln(u*(e^k - e^-k) + e^-k) = k + ln(u + (1-u)e^-2k)
    t = 1.0 + np.log(np.clip(u + (1.0 - u) * math.exp(-2.0 * kappa), 1e-300, None)) / kappa
    t = np.clip(t, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    e1, e2 = pick_axes(mu, rng)
    return (np.outer(sin_t * np.cos(phi), e1) + np.outer(sin_t * np.sin(phi), e2)
            + np.outer(t, mu.reshape(1, 3)))


def sample_set_normals(rng, k, min_sep_deg):
    dirs = []
    tries = 0
    while len(dirs) < k and tries < 10000:
        tries += 1
        z = rng.normal(size=3)
        z /= np.linalg.norm(z)
        if dirs:
            dots = np.abs(np.array([float(d @ z) for d in dirs]))
            if math.degrees(math.acos(float(np.clip(dots.max(), -1, 1)))) < min_sep_deg:
                continue
        dirs.append(z)
    if len(dirs) < k:
        raise SystemExit("[出题失败] 组方向拒绝采样超预算, 换 seed 或降 min_sep")
    return np.array(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--wells", type=int, default=3)
    ap.add_argument("--n-min", type=int, default=55)
    ap.add_argument("--n-max", type=int, default=85)
    ap.add_argument("--kappa-lo", type=float, default=60.0)
    ap.add_argument("--kappa-hi", type=float, default=250.0)
    ap.add_argument("--min-sep", type=float, default=35.0)
    ap.add_argument("--dirichlet-alpha", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="默认 data/challenge_<seed>/synthetic")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out or os.path.join(ROOT, "data", f"challenge_{args.seed}", "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    K = int(rng.integers(args.kmin, args.kmax + 1))
    centers = sample_set_normals(rng, K, args.min_sep_deg if hasattr(args, "min_sep_deg") else args.min_sep)
    kappas = rng.uniform(args.kappa_lo, args.kappa_hi, K)
    shares = rng.dirichlet(np.full(K, args.dirichlet_alpha))
    shares = shares / shares.sum()

    truth_sets, n_total = [], 0
    well_ns = [int(rng.integers(args.n_min, args.n_max + 1)) for _ in range(args.wells)]
    per_well_vecs = [[] for _ in range(args.wells)]
    for s in range(K):
        counts = np.array([rng.multinomial(n, shares)[s] for n in well_ns])
        truth_sets.append({
            "set_id": s,
            "normal": [round(float(x), 6) for x in centers[s]],
            "n_obs": int(counts.sum()),
        })
        n_total += int(counts.sum())
        for w in range(args.wells):
            if counts[w] > 0:
                per_well_vecs[w].append(fisher_sample(centers[s], float(kappas[s]), int(counts[w]), rng))

    files = []
    for w in range(args.wells):
        vecs = (np.concatenate(per_well_vecs[w], axis=0) if per_well_vecs[w]
                else np.zeros((0, 3)))
        rng.shuffle(vecs)
        rows = [n_to_dipdir(v) for v in vecs]
        # 往返断言: 产状->法向->产状 误差 < 1e-9 (除 dd 的 360 回绕)
        for (d0, a0), v in zip(rows, vecs):
            d1, a1 = n_to_dipdir(dipdir_to_n(d0, a0))
            assert abs(d1 - d0) < 1e-9 and (abs(a1 - a0) % 360) < 1e-9, "出题器坐标往返失败"
        fname = f"well_{w + 1:02d}.csv"
        with io.open(os.path.join(out_dir, fname), "w", encoding="utf-8", newline="") as f:
            f.write("id,depth,dip,dip_direction\n")
            depth = 0.0
            for i, (d, a) in enumerate(rows):
                depth += float(rng.uniform(1.5, 3.5))
                f.write(f"obs_{i:03d},{depth:.2f},{d:.2f},{a:.2f}\n")
        files.append({"file": fname, "n_obs": len(rows)})

    with io.open(os.path.join(out_dir, "truth_SEALED.json"), "w", encoding="utf-8") as f:
        json.dump({
            "WARNING": "判分 (scripts/check_challenge.py) 跑完之前不要打开本文件",
            "seed": args.seed, "K_true": K,
            "sets": [
                {"set_id": ts["set_id"],
                 "dip": round(ts["normal"] and n_to_dipdir(np.array(ts["normal"]))[0], 3),
                 "dip_direction": round(n_to_dipdir(np.array(ts["normal"]))[1], 3),
                 "kappa": round(float(kappas[ts["set_id"]]), 2),
                 "share": round(float(shares[ts["set_id"]]), 4),
                 "n_obs": ts["n_obs"]}
                for ts in truth_sets],
            "well_files": files, "n_total": n_total,
            "model": "球面拒绝采样组心(角距>=min_sep)+Dirichlet占比+Fisher(kappa)精确逆CDF噪声",
        }, f, ensure_ascii=False, indent=2, allow_nan=False)

    with io.open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "n_total": n_total, "wells": args.wells,
                   "files": files,
                   "note": "题面参数; 真值封存在 truth_SEALED.json"},
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[make_challenge] seed={args.seed} K_true=<sealed> n_total={n_total}")
    for fl in files:
        print(f"  {fl['file']}: {fl['n_obs']} 条")
    print(f"-> {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
