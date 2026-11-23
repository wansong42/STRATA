# -*- coding: utf-8 -*-
"""beishan 中国场地复刻 FORGE 管线 (P0 主产线).

FORGE 已验证: 单井 18.14 -> 联合 10.99 -> K12 锚点 9.35.
beishan (BS34): 22 net x 40 pt = 880 pt, 同构造场地多钻孔.

本脚本复刻 FORGE 完整管线:
  Phase 1: 走向勘察 + 共享组系判定 (dip/az 分布 + 圆均值差)
  Phase 2: 单井独立 vs 多井联合 vs 井内模态 (K 网格 4..20)
  Phase 3: 走向锚点 (dip 分布峰值 -> kmeans 初始化)
  Phase 4: 10 掩码稳健性闸门 (均值 +- std + 极差)

产出: results/beishan_forge_pipeline.json + 一行结论到 results/final_report.md
"""
import os
import sys
import json
import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fractureflow.setlabel import (
    generate_set_ids, joint_set_ids, joint_set_ids_grid, spherical_kmeans)
from fractureflow.inference import set_aware_dirs

RES = os.path.join(_ROOT, "results")
REAL_PT = os.path.join(_ROOT, "data/real/loaded_real_nets_setid.pt")
REAL_PT_ORIG = os.path.join(_ROOT, "data/real/loaded_real_nets.pt")

MASK_RNG = 999
OBS_FRAC = 0.4
KM_SEED = 42


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def acos_err(pred, true):
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    c = np.clip(np.abs(np.einsum("ij,ij->i", pred, true)), 0, 1)
    return np.rad2deg(np.arccos(c))


def dip_dir_of(nrm):
    n = _unit(np.asarray(nrm, float))
    dip = np.rad2deg(np.arccos(np.clip(np.abs(n[:, 2]), 0, 1)))
    az = np.rad2deg(np.arctan2(n[:, 0], n[:, 1])) % 360
    return dip, az


def circmean(deg):
    r = np.deg2rad(np.asarray(deg, float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)


def circ_diff(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def make_mask(n, rng_seed=MASK_RNG):
    return np.random.default_rng(rng_seed).random(n) < OBS_FRAC


# ============== 数据加载 ==============
def load_beishan():
    """加载 beishan 22 net, 返回 list[dict] (含 set_ids)."""
    d = torch.load(REAL_PT, weights_only=False)
    bnets = [dict(n) for i, n in enumerate(d) if 'BS34' in str(n.get('wid', ''))]
    if not bnets:
        raise RuntimeError("BS34 not found in loaded_real_nets_setid.pt")
    # 确保 nrm_full 存在
    for n in bnets:
        if 'nrm_full' not in n:
            n['nrm_full'] = np.asarray(n['nrm'], float).copy()
    return bnets


# ============== Phase 1: 走向勘察 ==============
def phase1_survey(nets):
    """走向勘察 + 共享组系判定."""
    info = {}
    all_nrm = []
    for i, n in enumerate(nets):
        nrm = np.asarray(n['nrm_full'], float)
        all_nrm.append(nrm)
        dip, az = dip_dir_of(nrm)
        info[f"net{i}"] = dict(
            n=len(nrm),
            dip_median=float(np.median(dip)),
            dip_mean=float(np.mean(dip)),
            az_circmean=float(circmean(az)),
            az_median=float(np.median(az)),
        )
    pooled = np.vstack(all_nrm)
    dip, az = dip_dir_of(pooled)
    info["_pooled"] = dict(
        n=len(pooled),
        dip_median=float(np.median(dip)),
        az_circmean=float(circmean(az)),
    )
    # 共享组系判定: 各井 dip 中位数 + az 圆均值 的极差
    dips = [info[k]["dip_median"] for k in info if k != "_pooled"]
    azs = [info[k]["az_circmean"] for k in info if k != "_pooled"]
    info["_verdict"] = dict(
        dip_range=float(max(dips) - min(dips)),
        az_circ_diffs=[float(circ_diff(azs[i], azs[j]))
                       for i in range(len(azs)) for j in range(i+1, len(azs))],
        az_max_diff=float(max(circ_diff(azs[i], azs[j])
                              for i in range(len(azs)) for j in range(i+1, len(azs)))),
        shared_group_system=bool(max(dips) - min(dips) <= 20.0),
    )
    return info


# ============== Phase 2: 单井 vs 联合 + K 网格 ==============
def eval_setaware(nets, assigns, mask_seed=MASK_RNG):
    """nets + assigns -> 隐伏点 MAE/p50/p90/RMSE."""
    errs, per = [], []
    for net, sid in zip(nets, assigns):
        nrm = np.asarray(net.get('nrm_full', net.get('nrm')), float)
        pos = np.asarray(net['pos'], float)
        occ = len(nrm)
        mask = np.random.default_rng(mask_seed).random(occ) < OBS_FRAC
        dirs, _ = set_aware_dirs(pos, nrm, mask, np.asarray(sid, int), strict=True)
        e = acos_err(dirs, nrm)
        hid = ~mask
        errs.append(e[hid])
        per.append(dict(mae=float(e[hid].mean()),
                        p50=float(np.median(e[hid])),
                        p90=float(np.percentile(e[hid], 90)),
                        n_hidden=int(hid.sum())))
    allv = np.concatenate(errs)
    return dict(mae=float(allv.mean()), p50=float(np.median(allv)),
                p90=float(np.percentile(allv, 90)),
                rmse=float(np.sqrt((allv**2).mean())),
                n_hidden=int(len(allv))), per


def phase2_kgrid(nets, Krange=(4, 20), mask_seed=MASK_RNG):
    """K 网格扫: 单井独立 vs 多井联合."""
    wells_nrm = [np.asarray(n['nrm_full'], float) for n in nets]
    results = {}
    for K in range(Krange[0], Krange[1] + 1):
        # 单井独立: 每井 silhouette 选 K (固定 K)
        ind_assigns = []
        for w in wells_nrm:
            c, a = spherical_kmeans(w, K, seed=KM_SEED)
            ind_assigns.append(a)
        ind_agg, _ = eval_setaware(nets, ind_assigns, mask_seed)
        # 多井联合
        try:
            jr = joint_set_ids(wells_nrm, K, seed=KM_SEED)
            joint_agg, _ = eval_setaware(nets, jr['assign'], mask_seed)
        except Exception as ex:
            joint_agg = dict(mae=float('nan'), note=str(ex))
        results[K] = dict(
            independent=ind_agg,
            joint=joint_agg,
            n_pts=int(sum(len(w) for w in wells_nrm)),
        )
    return results


# ============== Phase 3: 走向锚点 ==============
def phase3_anchors(nets, Krange=(4, 12), mask_seed=MASK_RNG):
    """走向锚点: 从 dip 分布峰值提锚点 -> kmeans 初始化."""
    wells_nrm = [np.asarray(n['nrm_full'], float) for n in nets]
    pooled = np.vstack(wells_nrm)
    # 提锚点: 球面 k-means K=12 取组心 -> 按组大小降序取前 K 个
    c12, _ = spherical_kmeans(pooled, 12, seed=KM_SEED)
    # 按组大小排序
    abscos = np.abs(_unit(pooled) @ _unit(c12).T)
    a12 = abscos.argmax(1)
    cnts = np.array([(a12 == k).sum() for k in range(12)])
    order = np.argsort(-cnts)
    c12_sorted = c12[order]

    results = {}
    for K in range(Krange[0], Krange[1] + 1):
        anchors = c12_sorted[:K]
        try:
            jr = joint_set_ids(wells_nrm, K, seed=KM_SEED, anchors=anchors)
            agg, _ = eval_setaware(nets, jr['assign'], mask_seed)
        except Exception as ex:
            agg = dict(mae=float('nan'), note=str(ex))
        results[K] = agg
    return results, c12_sorted


# ============== Phase 4: 10 掩码稳健门 ==============
def phase4_robustness(nets, K, mask_seeds=None, use_anchors=True):
    """10 掩码稳健性闸门."""
    if mask_seeds is None:
        mask_seeds = list(range(1000, 1010))
    wells_nrm = [np.asarray(n['nrm_full'], float) for n in nets]
    pooled = np.vstack(wells_nrm)
    c12, _ = spherical_kmeans(pooled, 12, seed=KM_SEED)
    abscos = np.abs(_unit(pooled) @ _unit(c12).T)
    a12 = abscos.argmax(1)
    cnts = np.array([(a12 == k).sum() for k in range(12)])
    order = np.argsort(-cnts)
    c12_sorted = c12[order]
    anchors = c12_sorted[:K] if use_anchors else None

    maes, p50s, p90s = [], [], []
    for sd in mask_seeds:
        jr = joint_set_ids(wells_nrm, K, seed=KM_SEED, anchors=anchors)
        agg, _ = eval_setaware(nets, jr['assign'], sd)
        maes.append(agg['mae']); p50s.append(agg['p50']); p90s.append(agg['p90'])
    maes = np.array(maes); p50s = np.array(p50s); p90s = np.array(p90s)
    return dict(
        K=K, n_seeds=len(mask_seeds),
        mae_mean=float(maes.mean()), mae_std=float(maes.std()),
        mae_range=float(maes.max() - maes.min()),
        p50_mean=float(p50s.mean()), p50_std=float(p50s.std()),
        p90_mean=float(p90s.mean()), p90_std=float(p90s.std()),
        per_seed=[dict(seed=int(s), m=float(m)) for s, m in zip(mask_seeds, maes)],
    )


# ============== 主流程 ==============
def main():
    print("=" * 60)
    print("beishan 中国场地复刻 FORGE 管线")
    print("=" * 60)
    nets = load_beishan()
    print(f"beishan nets: {len(nets)}, total pts: {sum(len(np.asarray(n['nrm'])) for n in nets)}")

    # Phase 1: 走向勘察
    print("\n[Phase 1] 走向勘察 + 共享组系判定")
    survey = phase1_survey(nets)
    for k, v in survey.items():
        if not k.startswith("_"):
            print(f"  {k}: dip_med={v['dip_median']:.1f} az_circ={v['az_circmean']:.1f}")
    v = survey["_verdict"]
    print(f"  verdict: dip_range={v['dip_range']:.1f} az_max_diff={v['az_max_diff']:.1f} "
          f"shared={v['shared_group_system']}")

    # Phase 2: K 网格
    print("\n[Phase 2] K 网格扫 (单井独立 vs 多井联合)")
    kgrid = phase2_kgrid(nets, Krange=(4, 20))
    for K, v in sorted(kgrid.items()):
        ind = v['independent']['mae']
        jnt = v['joint']['mae']
        print(f"  K={K:2d}: ind={ind:.2f}  joint={jnt:.2f}  (n_pts={v['n_pts']})")

    # Phase 3: 走向锚点
    print("\n[Phase 3] 走向锚点 (K 网格)")
    anchor_res, c12 = phase3_anchors(nets, Krange=(4, 12))
    for K, v in sorted(anchor_res.items()):
        print(f"  K={K:2d}: mae={v['mae']:.2f}  p50={v['p50']:.2f}  p90={v['p90']:.2f}")

    # Phase 4: 10 掩码稳健门 (K=12 锚点)
    print("\n[Phase 4] 10 掩码稳健性闸门 (K=12 锚点)")
    robust = phase4_robustness(nets, K=12, use_anchors=True)
    print(f"  K=12 anchor: mae={robust['mae_mean']:.2f}+-{robust['mae_std']:.2f} "
          f"(range {robust['mae_range']:.2f})")
    print(f"  PASS={robust['mae_std'] < 1.0 and robust['mae_range'] < 2.0}")

    # 也跑 K=8 稳健门
    robust8 = phase4_robustness(nets, K=8, use_anchors=True)
    print(f"  K=8  anchor: mae={robust8['mae_mean']:.2f}+-{robust8['mae_std']:.2f} "
          f"(range {robust8['mae_range']:.2f})")

    # 落盘
    out = dict(
        site="beishan_BS34",
        n_nets=len(nets),
        total_pts=int(sum(len(np.asarray(n['nrm'])) for n in nets)),
        survey=survey,
        kgrid={str(K): v for K, v in kgrid.items()},
        anchor={str(K): v for K, v in anchor_res.items()},
        robustness_K12=robust,
        robustness_K8=robust8,
        summary=dict(
            best_K12_anchor_mae=robust['mae_mean'],
            best_K12_anchor_std=robust['mae_std'],
            best_K8_anchor_mae=robust8['mae_mean'],
            verdict="beishan 复刻 FORGE 管线完成",
        ),
    )
    out_path = os.path.join(RES, "beishan_forge_pipeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    print(f"\n结果落盘: {out_path}")

    # 一行结论
    print("\n" + "=" * 60)
    print(f"结论: beishan K12 锚点 10-seed 稳健 = {robust['mae_mean']:.2f}+-{robust['mae_std']:.2f} deg")
    print(f"      K8  锚点 10-seed 稳健 = {robust8['mae_mean']:.2f}+-{robust8['mae_std']:.2f} deg")
    print("=" * 60)
    return out


if __name__ == "__main__":
    main()
