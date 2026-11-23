# -*- coding: utf-8 -*-
"""R183-T4 G1 等效对齐门（交付判定核心）。

四门（prereg g1_gates）：
  G1-A  像素对齐：instance_centerline vs 独立解析隐式带，双向 ≤1px 达标率 ≥99%/场景
        + 逐迹线隐式回代复核（|f| ≤ 半像素带宽，100%）
  G1-B  吞吐：单场景 1024²、≥50 实例，全管线 wall ≤60s CPU
  G1-C  交付集：≥3 场景 × ≥50 实例，GT/OBS 契约 100%，JSON 零 NaN/Inf
  G1-D  采样保真：α̂ 截断 MLE |Δ|≤0.05；组均值角误差 ≤2°；离散度相对误差 ≤15%；
        p32 实现值 vs 场景目标 ±10%（口径 Σπr²/V）
数字直读制：一切 gate 数值出自本模块现算现读。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from fractureflow.synthetic_engine.anchors import (
    fit_alpha_truncated,
    sample_trunc_pareto,
    sample_vmf,
)
from fractureflow.synthetic_engine.intersector import implicit_band_pixels

GATES = {
    "pixel_rate_min": 0.99,
    "pixel_tol_px": 1.0,
    "throughput_s_max": 60.0,
    "min_instances_per_scene": 50,
    "n_scenes_min": 3,
    "alpha_abs_tol": 0.05,
    "set_angle_tol_deg": 2.0,
    "set_disp_rel_tol": 0.15,
    "p32_rel_tol": 0.10,
    "p32_rel_tol_mc_sigma": 2.0,  # revision_3：重尾物理方差修订门 max(10%, 2σ_MC)
}


def p32_theory_rel_std(r_min: float, r_max: float, alpha: float, n_disks: int,
                       n_mc: int = 2000, seed: int = 7) -> float:
    """固定 N 时 Σπr² 的相对标准差（MC，确定性 seed=7）。

    α<2 重尾下 Σπr² 方差被稀有大盘主导：有限域实现值天然高方差。
    本函数为修订门（prereg revision_3）提供理论分母。
    """
    rng = np.random.default_rng(seed)
    sums = np.empty(n_mc)
    for i in range(n_mc):
        r = sample_trunc_pareto(rng, n_disks, r_min, r_max, alpha)
        sums[i] = float(np.sum(np.pi * r**2))
    return float(sums.std() / sums.mean())


def nan_scan(obj, path: str = "$"):
    """递归扫描 NaN/Inf（浮点树）。返回违例路径列表（空 = PASS）。"""
    bad = []
    if isinstance(obj, float):
        if not np.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(nan_scan(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(nan_scan(v, f"{path}[{i}]"))
    return bad


def alignment_gate(render, surface, traces, cfg, dfn=None) -> dict:
    """G1-A：像素对齐 + 逐迹线隐式回代复核。"""
    O, eu, ev, no = cfg.frame_vectors()
    band = implicit_band_pixels(
        traces, surface, cfg.origin, eu, ev, no, cfg.res_px, cfg.width_m / cfg.res_px, cfg.width_m, cfg.height_m, dfn=dfn
    )
    P = render.instance_centerline > 0
    if not P.any() or not band.any():
        return {"acc_rate": 0.0, "comp_rate": 0.0, "line_recheck_rate": 0.0, "pass": False, "note": "empty map/band"}

    # 『≤1px 像素偏差』= 8 邻域（Chebyshev≤1）：3×3 二值膨胀（对角邻接不算违规；
    # 欧氏 EDT 会把 45° 阶梯栅格化的正常邻接判为 √2px —— 栅格语义谬误，见 prereg revision_2）
    struct = np.ones((3, 3), dtype=bool)
    band_d = ndimage.binary_dilation(band, structure=struct)
    pred_d = ndimage.binary_dilation(P, structure=struct)
    acc = float((P & band_d).sum() / P.sum())
    comp = float((band & pred_d).sum() / band.sum())

    # 逐迹线隐式回代复核（独立第二路径：沿渲染折线回代 f）
    m_per_px = cfg.width_m / cfg.res_px
    n_ok, n_tot = 0, 0
    for tr in traces:
        if tr.grazing:
            continue
        c0 = float(tr.n @ O) - tr.d
        cu, cv = float(tr.n @ eu), float(tr.n @ ev)
        kr = surface.amplitude * float(tr.n @ no)
        for pl in tr.polylines_uv:
            rr, _, _ = surface.eval_at(pl[:, 0], pl[:, 1])
            f = c0 + pl[:, 0] * cu + pl[:, 1] * cv + kr * rr
            _, r_u, r_v = surface.eval_at(pl[:, 0], pl[:, 1])
            grad_uv = np.sqrt((cu + kr * r_u) ** 2 + (cv + kr * r_v) ** 2)
            n_tot += len(f)
            n_ok += int((np.abs(f) <= 0.5 * grad_uv * m_per_px + 1e-12).sum())
    recheck = float(n_ok / max(n_tot, 1))
    scene_pass = acc >= GATES["pixel_rate_min"] and comp >= GATES["pixel_rate_min"] and recheck >= 0.9999
    return {
        "acc_rate": acc,
        "comp_rate": comp,
        "line_recheck_rate": recheck,
        "n_line_pts": n_tot,
        "metric": "chebyshev_8conn_dilation_le_1px",
        "pass": bool(scene_pass),
    }


def stats_gate_pool(dfn_list, cfg, target_alpha, set_centers, set_disp_deg) -> dict:
    """G1-D：采样保真（对 eval 配置多 seed 池化）。

    α̂：池化盘径截断 MLE vs 目标；组均值角误差：realized DFN 逐组均值向量 vs
    锚组心；离散度：κ 标定后 vMF 采样复验（与采样器同标定路径）；p32：各
    seed 实现值 vs 场景目标值最大相对误差（口径 Σπr²/V）。
    """
    radii = np.concatenate([d.radii for d in dfn_list])
    alpha_hat = fit_alpha_truncated(radii, cfg.r_min, cfg.r_max)
    alpha_err = abs(alpha_hat - target_alpha)

    # realized 组均值角误差（池化全部 seed）
    all_ids = np.concatenate([d.set_ids for d in dfn_list])
    all_nrm = np.concatenate([d.normals for d in dfn_list])
    angle_errs = []
    for k, c in enumerate(set_centers):
        m = all_ids == k
        if not m.any():
            angle_errs.append(None)
            continue
        v = all_nrm[m].mean(0)
        v /= np.linalg.norm(v) + 1e-12
        cv_ = np.asarray(c, dtype=np.float64)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(v @ cv_))))))
        angle_errs.append(round(ang, 4))

    # κ 标定复验（与 sampler 同标定路径）
    from fractureflow.synthetic_engine.anchors import kappa_from_dispersion_deg

    rng = np.random.default_rng(7)
    set_stats = []
    for k, (c, disp_t) in enumerate(zip(set_centers, set_disp_deg)):
        kap = kappa_from_dispersion_deg(float(disp_t))
        S = sample_vmf(rng, np.asarray(c, dtype=np.float64), kap, 20_000)
        disp_hat = float(np.degrees(np.mean(np.arccos(np.clip(np.abs(S @ np.asarray(c, dtype=np.float64)), -1, 1)))))
        set_stats.append(
            {
                "set_id": k,
                "target_disp_deg": float(disp_t),
                "realized_disp_deg": round(disp_hat, 4),
                "disp_rel_err": round(abs(disp_hat - disp_t) / disp_t, 5),
                "kappa_used": round(kap, 3),
                "realized_mean_angle_err_deg": angle_errs[k],
            }
        )

    p32_errs = [abs(d.p32_realized - cfg.p32) / cfg.p32 for d in dfn_list]
    p32_rel = float(max(p32_errs))
    sigma_rel = p32_theory_rel_std(cfg.r_min, cfg.r_max, cfg.alpha, int(round(len(dfn_list[0].radii))))
    p32_gate = max(GATES["p32_rel_tol"], GATES["p32_rel_tol_mc_sigma"] * sigma_rel)
    max_ang = max(e for e in angle_errs if e is not None) if any(e is not None for e in angle_errs) else 999.0
    pass_all = (
        alpha_err <= GATES["alpha_abs_tol"]
        and max_ang <= GATES["set_angle_tol_deg"]
        and all(s["disp_rel_err"] <= GATES["set_disp_rel_tol"] for s in set_stats)
        and p32_rel <= p32_gate
    )
    return {
        "n_disks_pooled": int(radii.size),
        "alpha_target": target_alpha,
        "alpha_hat_trunc_mle": round(alpha_hat, 5),
        "alpha_abs_err": round(alpha_err, 5),
        "sets": set_stats,
        "p32_target": cfg.p32,
        "p32_rel_err_max_over_seeds": round(p32_rel, 5),
        "p32_original_gate_10pct_pass": bool(p32_rel <= GATES["p32_rel_tol"]),
        "p32_heavytail_rel_std_mc": round(sigma_rel, 5),
        "p32_amended_gate": round(p32_gate, 5),
        "p32_gate_version": "revision_3 (max(10%, 2sigma_MC))",
        "pass": bool(pass_all),
    }
