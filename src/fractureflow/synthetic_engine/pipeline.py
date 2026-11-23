# -*- coding: utf-8 -*-
"""R183 场景管线：采样 → 基面 → 求交 → 渲染 → 降质 → GT/OBS 双契约（计时）。

供 scripts/r183_build_scenes.py 与 tests/test_v_r183_.py 共用（同一路径，禁双标）。
RNG 纪律：采样/基面用 cfg.seed；渲染 rng = seed+1；降质 rng = seed+2。
"""
from __future__ import annotations

import time

import numpy as np

from fractureflow.fracture_instance import (
    Extent,
    FractureInstance,
    FractureInstanceSet,
    Plane,
    Provenance,
    SegmentQuality,
    TraceSegment,
    params_hash,
    utc_now,
)
from fractureflow.synthetic_engine.degrader import DegradeConfig, degrade
from fractureflow.synthetic_engine.intersector import intersect_fractures
from fractureflow.synthetic_engine.renderer import render_scene
from fractureflow.synthetic_engine.sampler import sample_dfn
from fractureflow.synthetic_engine.surface import BaseSurface


def cfg_hash(cfg) -> str:
    return params_hash(cfg.__dict__)


def build_scene(cfg, rand: dict, degrade_cfg: DegradeConfig | None = None):
    """构建场景。返回 bundle dict（不落盘）。"""
    t0 = time.perf_counter()
    dfn = sample_dfn(cfg)
    t1 = time.perf_counter()

    surface = BaseSurface(
        cfg.width_m,
        cfg.height_m,
        lattice=cfg.surface_lattice,
        amplitude=cfg.amplitude,
        seed=cfg.seed,
    )
    O, eu, ev, no = cfg.frame_vectors()
    traces, grazing = intersect_fractures(dfn, surface, O, eu, ev, no, cfg.terzaghi_censor)
    t2 = time.perf_counter()

    rng_render = np.random.default_rng(cfg.seed + 1)
    render = render_scene(cfg, surface, traces, rng_render, rand)
    t3 = time.perf_counter()

    mpp = cfg.width_m / cfg.res_px
    H = cfg.height_m
    widths = render.meta["trace_widths"]

    gt_instances: list[FractureInstance] = []
    for tr in traces:
        if tr.grazing or not tr.polylines_world:
            continue
        idx = tr.fracture_idx
        iid = f"{cfg.scene_id}_F{idx:04d}"
        w = float(widths.get(idx, 1))
        segs = []
        pts_all = []
        resid_sq = []
        inliers = 0
        n_pts = 0
        for k, (wm, wpx) in enumerate(zip(tr.polylines_world, tr.polylines_uv)):
            col = wpx[:, 0] / mpp
            row = (H - wpx[:, 1]) / mpp
            px = np.stack([col, row], axis=1)
            fdev = np.abs(wm @ tr.n - tr.d)
            resid_sq.append(float(np.mean(fdev**2)))
            tol = max(2.0 * float(np.sqrt(np.mean(fdev**2))), 1e-3)
            inliers += int((fdev <= tol).sum())
            n_pts += len(fdev)
            segs.append(
                TraceSegment(
                    seg_id=f"{iid}#s{k}",
                    pts_m=wm.tolist(),
                    pts_px=px.tolist(),
                    width_px=w,
                    mark="clean",
                    quality=SegmentQuality(
                        inlier_rate=float((fdev <= tol).mean()),
                        residual_m=float(np.sqrt(np.mean(fdev**2))),
                        gate="pass" if float(np.sqrt(np.mean(fdev**2))) <= max(cfg.amplitude, 0.02) else "warn",
                    ),
                )
            )
            pts_all.append(wm)
        cat = np.vstack(pts_all)
        rms = float(np.sqrt(np.mean(np.array(resid_sq) ** 2)))
        inlier_rate = inliers / max(n_pts, 1)
        gt_instances.append(
            FractureInstance(
                id=iid,
                plane=Plane(n=[float(x) for x in tr.n], d=tr.d),
                extent=Extent(bbox_min=cat.min(0).tolist(), bbox_max=cat.max(0).tolist(), segments=segs),
                quality=SegmentQuality(
                    inlier_rate=inlier_rate,
                    residual_m=rms,
                    gate="pass" if rms <= max(cfg.amplitude, 0.02) else "warn",
                ),
                provenance=Provenance(
                    source="synthetic_engine_v0",
                    method="analytic_intersection",
                    method_version="0.1.0",
                    created_at=utc_now(),
                    scene_id=cfg.scene_id,
                    seed=cfg.seed,
                    params_hash=cfg_hash(cfg),
                    notes=f"set_id={tr.set_id}; disk_radius_m={float(dfn.radii[idx]):.4f}; terzaghi_censor={cfg.terzaghi_censor}",
                ),
            )
        )
    gt_set = FractureInstanceSet(
        scene_id=cfg.scene_id,
        m_per_px=mpp,
        frame={
            "origin": list(map(float, cfg.origin)),
            "u_axis": list(map(float, eu)),
            "v_axis": list(map(float, ev)),
            "n_out": list(map(float, no)),
            "window_w_m": cfg.width_m,
            "window_h_m": cfg.height_m,
            "frame_note": "世界系 X东 Y北 Z上；pts_m 位于基面表面（含 fBm 起伏）；px=(col,row) 左上原点",
        },
        instances=gt_instances,
    )
    t4 = time.perf_counter()

    dcfg = degrade_cfg or DegradeConfig()
    rng_deg = np.random.default_rng(cfg.seed + 2)
    obs_set, diff = degrade(gt_set, render.visibility, rng_deg, dcfg, cfg.scene_id)
    t5 = time.perf_counter()

    return {
        "cfg": cfg,
        "rand": rand,
        "dfn": dfn,
        "surface": surface,
        "traces": traces,
        "grazing_count": grazing,
        "render": render,
        "gt_set": gt_set,
        "obs_set": obs_set,
        "diff": diff,
        "timings_s": {
            "sample": round(t1 - t0, 3),
            "intersect": round(t2 - t1, 3),
            "render": round(t3 - t2, 3),
            "contract_gt": round(t4 - t3, 3),
            "degrade": round(t5 - t4, 3),
            "total": round(t5 - t0, 3),
        },
    }
