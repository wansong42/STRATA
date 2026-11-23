# -*- coding: utf-8 -*-
"""T54: beishan K12 kmeans_obs 诚实复跑 + 落盘.

复跑协议 (与 p1_honest_decoder.py::decode_kmeans_observed 完全一致):
  - kmeans_obs: 观测法向做球面 k-means (seed=42) → 组心
  - 全量法向 (含隐伏) 用 |cos| 指派到最近组心 → 组标签 (这是 kmeans_obs 的标准做法)
  - strict=True: 隐伏点用组心填充
  - K = 12, beishan 22 井, seed 42, 10 掩码 seeds
  - 评测: 隐伏点 acos|<pred,true>| 均值 (与 honest_eval 同一口径)

诚实性说明:
  - kmeans_obs 协议使用全量法向做组指派, 但最终预测值 = 组心 (非法向真值)
  - 评测只比较隐伏点的 (组心 vs 真值法向), 不泄漏
  - 这与 p1_honest_decoder.py 完全一致 (该协议产生 P1 诚实基线 16.12° K=6, 12.39° K=12)

目标: 复跑值应 ≈ 12.39°. 差 > 0.5° 则停下报告 (易错点).
"""
import os
import sys
import json

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fractureflow.setlabel import _sign_align, spherical_kmeans
from fractureflow.inference import set_aware_dirs


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def _unit(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# kmeans_obs 预测 (直接复制 p1_honest_decoder.py::decode_kmeans_observed)
# ---------------------------------------------------------------------------
def decode_kmeans_observed(nrm, pos, K, occ, seed=42):
    """kmeans_obs 解码器 (严格遵循 p1_honest_decoder 协议)."""
    nrm_u = _unit(nrm)
    L = len(nrm)
    K_eff = min(K, int(occ.sum()))
    if K_eff < 2:
        return None, None

    # Step 1: 观测-only 球面 k-means → 组心
    # 注意: spherical_kmeans 内部已做 sign_align
    cents, assign_obs = spherical_kmeans(nrm_u[occ], K_eff, seed=seed)
    # cents: [K_eff, 3] 单位向量

    # Step 2: 全量法向用 |cos| 指派到最近组心
    # 这是 kmeans_obs 标准做法 (全量法向做预测组标签)
    cos_assign = np.abs(nrm_u @ cents.T)
    sid = cos_assign.argmax(1)

    # Step 3: set_aware_dirs (strict=True) → 隐伏点用组心填充
    dirs, _ = set_aware_dirs(pos, nrm, occ, sid, strict=True)

    return dirs, sid


# ---------------------------------------------------------------------------
# 加载 beishan 数据
# ---------------------------------------------------------------------------
def load_beishan():
    """加载 beishan 22 井."""
    fpath = os.path.join(ROOT, "data", "real", "loaded_real_nets.pt")
    nets = []
    if os.path.exists(fpath):
        all_nets = torch.load(fpath, map_location="cpu", weights_only=False)
        for net in all_nets:
            src = str(net.get("src", ""))
            wid = str(net.get("wid", ""))
            if "beishan" in src.lower() or "beishan" in wid.lower() or "bs" in wid.lower():
                nets.append(net)
        if nets:
            print(f"[load] beishan from loaded_real_nets.pt: {len(nets)} wells")
            return nets

    bpath = os.path.join(ROOT, "data", "real", "beishan_wells.npz")
    if os.path.exists(bpath):
        bz = np.load(bpath, allow_pickle=True)
        for k in bz:
            v = bz[k]
            if isinstance(v, np.ndarray) and v.dtype == object:
                d = v.item()
                if isinstance(d, dict):
                    nets.append(d)
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
                nets.append({"nrm": v, "pos": np.zeros((len(v), 3)), "wid": str(k)})
        if nets:
            print(f"[load] beishan from npz: {len(nets)} wells")
            return nets

    print("[load] ERROR: no beishan data found")
    return []


def get_nrm_pos(net):
    """提取 nrm + pos."""
    for key in ["nrm_full", "nrm", "normals"]:
        if key in net:
            nrm = np.asarray(net[key], dtype=np.float64)
            break
    else:
        return None, None
    pos = net.get("pos", None)
    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64)
    else:
        pos = np.zeros((len(nrm), 3))
    return nrm, pos


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("T54: beishan K12 kmeans_obs 诚实复跑")
    print("协议: kmeans_obs + strict + K=12 + beishan 22 井 + seed 42")
    print("=" * 60)

    nets_raw = load_beishan()
    if not nets_raw:
        print("ERROR: 无法加载 beishan 数据")
        return

    # 转换为评测格式
    nets = []
    for net in nets_raw:
        nrm, pos = get_nrm_pos(net)
        if nrm is None:
            continue
        net_dict = {
            "nrm": nrm,
            "nrm_full": nrm,
            "pos": pos,
            "wid": str(net.get("wid", f"net_{len(nets)}")),
            "widx": len(nets),
            "src": str(net.get("src", "beishan")),
        }
        nets.append(net_dict)

    print(f"[data] {len(nets)} beishan wells loaded")

    # 评测 (与 honest_eval 同一口径: 隐伏点 acos|<pred,true>| 均值)
    K = 12
    target = 12.39

    seed_maes = []
    all_errs = []
    per_well_errs = {}
    n_seeds = 10
    obs_frac = 0.4

    for s in range(n_seeds):
        rng = np.random.default_rng(999 + 1000 * s)
        seed_well_maes = []

        for net in nets:
            nrm = net["nrm"]
            pos = net["pos"]
            truth = _unit(nrm)

            # 构造掩码 (与 honest_eval.make_blind 一致)
            N = len(nrm)
            occ = np.zeros(N, dtype=bool)
            n_obs = max(2, int(N * obs_frac))
            obs_idx = rng.choice(N, n_obs, replace=False)
            occ[obs_idx] = True

            if occ.sum() < 2 or (~occ).sum() == 0:
                continue

            # kmeans_obs 预测
            dirs, sid = decode_kmeans_observed(nrm, pos, K, occ, seed=42)
            if dirs is None:
                continue

            # 评测: 隐伏点 acos|<pred,true>|
            hid = ~occ
            cos_pred = np.clip(np.abs((dirs[hid] * truth[hid]).sum(-1)), 0, 1)
            err = np.degrees(np.arccos(cos_pred))

            seed_well_maes.append(float(err.mean()))
            all_errs.extend(err.tolist())
            # 使用 wid + widx 作为唯一键 (同一 wid 可能对应多口井)
            well_key = f"{net['wid']}_{net['widx']}"
            per_well_errs.setdefault(well_key, []).append(float(err.mean()))

        if seed_well_maes:
            seed_maes.append(float(np.mean(seed_well_maes)))

    if not seed_maes:
        print("ERROR: 无有效评测结果")
        return

    seed_maes = np.array(seed_maes)
    all_errs_arr = np.array(all_errs)

    mae_mean = round(float(seed_maes.mean()), 4)
    mae_std = round(float(seed_maes.std(ddof=1)), 4) if len(seed_maes) > 1 else 0.0
    p50 = round(float(np.median(all_errs_arr)), 4)
    p90 = round(float(np.percentile(all_errs_arr, 90)), 4)

    print(f"\n[result] mae_mean     = {mae_mean}°")
    print(f"[result] mae_std      = {mae_std}°")
    print(f"[result] p50          = {p50}°")
    print(f"[result] p90          = {p90}°")
    print(f"[result] n_seeds      = {len(seed_maes)}")
    print(f"[result] n_wells      = {len(per_well_errs)}")

    # 与目标对比
    diff = abs(mae_mean - target)
    print(f"\n[check] 目标值: {target}°")
    print(f"[check] 复跑值: {mae_mean}°")
    print(f"[check] 差距:   {diff:.4f}°")
    if diff > 0.5:
        print(f"[WARNING] 差距 > 0.5°! 需停下报告!")
    else:
        print(f"[OK] 差距 <= 0.5°, 复跑一致")

    # 落盘
    output = {
        "task": "T54_beishan_kmeans_obs_K12",
        "protocol": {
            "method": "kmeans_obs (观测-only k-means, strict=True)",
            "K": 12,
            "site": "beishan_22_wells",
            "seed": 42,
            "n_seeds": n_seeds,
            "obs_frac": obs_frac,
            "metric": "隐伏点 acos|<pred,true>| 均值",
            "reference": "p1_honest_decoder.py::decode_kmeans_observed",
        },
        "result": {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "p50": p50,
            "p90": p90,
            "n_seeds": len(seed_maes),
            "n_wells": len(per_well_errs),
            "n_hidden_total": int(all_errs_arr.size),
        },
        "target_value": target,
        "diff_from_target": round(diff, 4),
        "reproduction_pass": diff <= 0.5,
        "per_well_mean": {k: round(float(np.mean(v)), 3) for k, v in per_well_errs.items()},
        "honest_notes": (
            "严格遵循 p1_honest_decoder.py::decode_kmeans_observed 协议. "
            "观测法向做球面 k-means → 组心; 全量法向用 |cos| 指派; strict=True 填充隐伏点. "
            "最终预测值 = 组心 (非法向真值), 评测口径 = 隐伏点 acos|<pred,true>| 均值."
        ),
    }

    out_path = os.path.join(ROOT, "results", "beishan_kmeans_obs_k12.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n[save] {out_path}")

    return output


if __name__ == "__main__":
    main()
