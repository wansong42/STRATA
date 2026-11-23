# -*- coding: utf-8 -*-
"""挑战判分器: 把打标结果 (.pt) 与封存真值 (truth_SEALED.json) 对答案.

口径与项目完全一致 (不引入新指标):
  - 组中心 = 组内观测法向的符号对齐均值 (先 |cos| 对齐再平均, ±n 不抵消);
  - 组方向误差 = acos|<pred_center, true_center>| (度) —— 即组系表 modal_err 同款;
  - 匹配 = 全 (label,truth) 对角阵按误差升序贪心一对一, 角距 > 45° 视为未匹配;
  - 未匹配的真值组计"漏", 未匹配的标注组计"碎"(过分裂), 分列报告不合并粉饰。

判级参考带 (练习用, 研究口径, 非商务承诺):
  mean modal_err < 3° 优秀 / < 7° 良好 / < 12° 达自设承诺线 / 否则未达标。
每文件与池化 (所有井合并) 各报一次。用法:
  python scripts/check_challenge.py --truth <truth_SEALED.json> --labeled a.pt [b.pt ...]
"""
import argparse
import io
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _sign_align_mean(vecs):
    """迭代符号对齐后求均值 (两轮), 返回单位均值向量."""
    v = np.asarray(vecs, float)
    if len(v) == 0:
        return np.zeros(3)
    ref = _unit(v[0])
    for _ in range(2):
        w = v * np.sign(v @ ref).reshape(-1, 1)
        w[np.isnan(w).any(axis=1)] = 0.0
        ref = _unit(w.sum(axis=0))
    return ref


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_unit(a), _unit(b)))), -1.0, 1.0))))


def centers_from_labeled(nrm, ids):
    ids = np.asarray(ids, int)
    nrm = np.asarray(nrm, float)
    return {int(s): _sign_align_mean(nrm[ids == s]) for s in np.unique(ids)}


def dipdir_to_n(dip_deg, dd_deg):
    d, a = np.radians(float(dip_deg)), np.radians(float(dd_deg))
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def greedy_match(pred_centers, truth_centers, max_ang=45.0):
    pairs = sorted(
        ((ang, pi, ti) for pi, p in pred_centers.items() for ti, t in truth_centers.items()
         for ang in [angle_deg(p, t)]),
        key=lambda x: x[0])
    used_p, used_t, matched = set(), set(), []
    for ang, pi, ti in pairs:
        if ang > max_ang or pi in used_p or ti in used_t:
            continue
        matched.append((pi, ti, ang))
        used_p.add(pi)
        used_t.add(ti)
    splits = [pi for pi in pred_centers if pi not in used_p]
    misses = [ti for ti in truth_centers if ti not in used_t]
    return matched, splits, misses


def report(tag, pred_centers, truth_centers):
    matched, splits, misses = greedy_match(pred_centers, truth_centers)
    errs = [a for _, _, a in matched]
    mean_e = float(np.mean(errs)) if errs else float("nan")
    verdict = ("优秀" if mean_e < 3 else "良好" if mean_e < 7 else
               "达自设承诺线（12°）" if mean_e < 12 else "未达标")
    print(f"[{tag}] K_pred={len(pred_centers)} K_true={len(truth_centers)} "
          f"matched={len(matched)} 碎组={len(splits)} 漏组={len(misses)}")
    for pi, ti, ang in matched:
        print(f"   pred#{pi} <-> true#{ti}: {ang:.3f}°")
    if errs:
        print(f"   mean modal_err = {np.mean(errs):.3f}° | max = {np.max(errs):.3f}° -> {verdict} (练习参考带, 研究口径)")
    return mean_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--labeled", nargs="+", required=True, help="auto_label_borehole --csv --out 产出的 .pt")
    args = ap.parse_args()

    with io.open(args.truth, encoding="utf-8") as f:
        truth = json.load(f)
    truth_centers = {int(s["set_id"]): _unit(dipdir_to_n(s["dip"], s["dip_direction"]))
                     for s in truth["sets"]}

    all_nrm, all_ids, offset = [], [], 0
    for pt in args.labeled:
        nets = torch.load(pt, map_location="cpu", weights_only=False)
        nets = nets if isinstance(nets, list) else [nets]
        for net in nets:
            nrm = np.asarray(net.get("nrm_full", net["nrm"]), float)
            ids = np.asarray(net["set_ids"], int)
            label = os.path.basename(pt).replace(".pt", "")
            report(label, centers_from_labeled(nrm, ids), truth_centers)
            all_nrm.append(nrm)
            all_ids.append(ids + offset)
            offset = int(all_ids[-1].max()) + 2
    report("池化(全部井合并)", centers_from_labeled(np.concatenate(all_nrm), np.concatenate(all_ids)),
           truth_centers)
    print("(真值已开封: 请只在完成上行打印之后查看 truth_SEALED.json)")


if __name__ == "__main__":
    main()
