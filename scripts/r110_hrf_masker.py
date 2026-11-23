# -*- coding: utf-8 -*-
"""R110 · HRF 虚拟观测遮蔽平台 (四类观测 × 预算梯度, 冻结判据实现).

协议权威: results/v_r110_recon_protocol.json §masking_platform / §m_obs_spec。

四类虚拟观测 (全部点级布尔掩码, 确定性):
  1. full                全观测 (对照/自洽验证场景)
  2. borehole_{1,2,4}    虚拟钻孔线切割: k 条垂直孔 (r=2.0 m), 布点拒绝采样
                         保证与 held-out 扫描线距离 > 10 m
  3. outcrop_{10,30,50}  单侧露头半空间裁剪: (p−center)·d̂ ≥ quantile(1−f), d̂=+东
  4. station_30          站位遮挡+视角剔除: 固定相机位, 面向剔除 (轴向法向
                         |cos(视线,法向)|>cos80°) + 512² 深度缓冲单回波遮挡,
                         z_tol 二分校准到可见率 30%±0.5%
  5. multi_30_2          多源联合 = outcrop_30 ∪ borehole_2

输出 (results/r110_hrf_masking/):
  <scenario>.npz  : obs_packed (packbits uint8) + face_support[136 int32]
  <scenario>.json : M_obs 元数据 (协议 §m_obs_spec 字段, 无时间戳——确定性)
  manifest.json   : 源文件 sha256/size/mtime + 场景清单 + 生成时刻

红线: E:/fracture_data/hrf_frac.npz 只读; 本脚本不写 E 盘任何文件。

用法 (src 下不必, 直接):
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r110_hrf_masker.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from r110_recon_metrics import (  # noqa: E402
    frozen_scanlines, draw_borehole_positions, _unit)

SRC = r"E:/fracture_data/hrf_frac.npz"
OUT_DIR = os.path.join(ROOT, "results", "r110_hrf_masking")
BOREHOLE_RADIUS = 2.0        # m (协议冻结)
STATION_GRID = 512
STATION_CAMERA_OFFSET = np.array([150.0, 200.0, 260.0])  # 相对域中心 (冻结)
FACING_COS = float(np.cos(np.radians(80.0)))              # 轴向面向剔除阈 (冻结)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- #
# 各遮蔽器核心 (纯函数, 可单测/可复算对拍)
# --------------------------------------------------------------------------- #
def mask_full(xyz, fid, **_):
    return np.ones(len(xyz), dtype=bool)


def mask_borehole(xyz, fid, box, k, seed=20260828):
    holes = draw_borehole_positions(box, k, seed=seed)
    dx = xyz[:, 0][:, None] - holes[None, :, 0]
    dy = xyz[:, 1][:, None] - holes[None, :, 1]
    d2 = dx * dx + dy * dy
    obs = (d2.min(1) < BOREHOLE_RADIUS ** 2)
    return obs, holes


def mask_outcrop(xyz, fid, box, frac, direction=(1.0, 0.0, 0.0)):
    d = _unit(np.asarray(direction, dtype=np.float64))
    center = 0.5 * (np.asarray(box[0]) + np.asarray(box[1]))
    s = (xyz - center) @ d
    thr = float(np.quantile(s, 1.0 - frac))
    return (s >= thr), dict(direction_d=[float(x) for x in d], threshold_m=thr)


def _station_project(xyz, nrm, box):
    """站位几何投影 (一次性): 返回 (px, py, dep, facing)。纯函数。"""
    box = np.asarray(box, dtype=np.float64)
    center = 0.5 * (box[0] + box[1])
    cam = center + STATION_CAMERA_OFFSET
    view = _unit(center - cam)
    a = np.array([1.0, 0.0, 0.0]) if abs(view[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_ax = _unit(np.cross(view, a))
    v_ax = np.cross(view, u_ax)
    rel = xyz - cam
    dep = rel @ view
    aa = rel @ u_ax
    bb = rel @ v_ax
    # 面向剔除 (轴向: |cos| 大于阈, 即非侧视)
    facing = np.abs(nrm @ view) / np.maximum(
        np.linalg.norm(nrm, axis=1), 1e-12) > FACING_COS
    a0, a1 = float(aa.min()), float(aa.max())
    b0, b1 = float(bb.min()), float(bb.max())
    px = np.clip(((aa - a0) / max(a1 - a0, 1e-9) * STATION_GRID).astype(int),
                 0, STATION_GRID - 1)
    py = np.clip(((bb - b0) / max(b1 - b0, 1e-9) * STATION_GRID).astype(int),
                 0, STATION_GRID - 1)
    return px, py, dep, facing


def _station_visibility(proj, z_tol):
    """深度缓冲单回波遮挡 (给定投影与 z_tol)。纯函数。"""
    px, py, dep, facing = proj
    depth_buf = np.full((STATION_GRID, STATION_GRID), np.inf)
    np.minimum.at(depth_buf, (py, px), dep)
    dmin = depth_buf[py, px]
    return facing & (dep - dmin < z_tol)


def mask_station(xyz, nrm, fid, box, frac, tol=5e-3, iters=28):
    """z_tol 二分校准到目标可见率。确定性 (投影缓存, 迭代只比深度)。"""
    proj = _station_project(xyz, nrm, box)
    lo, hi = 0.0, float(np.linalg.norm(box[1] - box[0]))
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = _station_visibility(proj, mid)
        f = float(m.mean())
        best = (mid, m, f)
        if abs(f - frac) < tol:
            break
        if f < frac:
            lo = mid
        else:
            hi = mid
    z_tol, m, f = best
    return m, dict(camera_pos=[float(x) for x in (0.5 * (np.asarray(box[0]) + np.asarray(box[1])) + STATION_CAMERA_OFFSET)],
                   grid_res=STATION_GRID, z_tol_m=float(z_tol),
                   achieved_fraction=f)


# --------------------------------------------------------------------------- #
# M_obs 组装与落盘
# --------------------------------------------------------------------------- #
def build_m_obs(scenario, obs, fid, extra, box, survey_box):
    """协议 §m_obs_spec 字段。per_face_confidence = min(1, support/300) (无真值信息)。"""
    obs = np.asarray(obs, dtype=bool)
    fid = np.asarray(fid)
    labeled = fid >= 0
    obs_l = obs & labeled
    support = np.zeros(int(fid.max()) + 1, dtype=np.int64)
    for f, cnt in zip(*np.unique(fid[obs_l], return_counts=True)):
        support[int(f)] = cnt
    observed_faces = [int(i) for i in range(len(support)) if support[i] > 0]
    per_face_conf = {int(i): float(min(1.0, support[i] / 300.0))
                     for i in observed_faces}
    m = dict(
        scenario=scenario,
        observation_type=_obs_type(scenario),
        view_geometry=dict(extra or {}),
        survey_box=[[float(x) for x in box[0]], [float(x) for x in box[1]]],
        coverage_fraction=float(obs_l.sum() / max(int(labeled.sum()), 1)),
        observed_faces=observed_faces,
        per_face_confidence=per_face_conf,
        n_obs_points=int(obs_l.sum()),
        diagnostics_only=dict(
            observed_point_fraction_per_face={
                int(i): float(support[i]) for i in observed_faces}),
        heldout_scanlines_note="20 条冻结垂直扫描线 (seed 20260828) 永不进入任何输入观测; 见协议 §masking_platform",
    )
    return m, support


def _obs_type(scenario):
    if scenario == "full":
        return "full"
    if scenario.startswith("borehole"):
        return "borehole"
    if scenario.startswith("outcrop"):
        return "outcrop_halfspace"
    if scenario.startswith("station"):
        return "station_occlusion"
    if scenario.startswith("multi"):
        return "multi"
    raise ValueError(scenario)


def save_scenario(scenario, obs, fid, m_obs, support):
    os.makedirs(OUT_DIR, exist_ok=True)
    packed = np.packbits(np.asarray(obs, dtype=bool))
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{scenario}.npz"),
        obs_packed=packed, n_points=len(obs), face_support=support)
    with open(os.path.join(OUT_DIR, f"{scenario}.json"), "w", encoding="utf-8") as f:
        json.dump(m_obs, f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"saved {scenario}: cov={m_obs['coverage_fraction']:.4f} "
        f"faces={len(m_obs['observed_faces'])} pts={m_obs['n_obs_points']}")


def load_scenario(scenario):
    """供 B1 / 测试复用: 返回 (obs bool[L], M_obs dict)。"""
    z = np.load(os.path.join(OUT_DIR, f"{scenario}.npz"))
    obs = np.unpackbits(z["obs_packed"], count=int(z["n_points"])).astype(bool)
    with open(os.path.join(OUT_DIR, f"{scenario}.json"), encoding="utf-8") as f:
        m = json.load(f)
    return obs, m


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    import datetime
    log(f"loading {SRC} (read-only)")
    with open(SRC, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    d = np.load(SRC, allow_pickle=True)
    xyz = d["xyz"].astype(np.float64)
    nrm = _unit(d["nrm"].astype(np.float64))
    fid = d["fid"].astype(np.int64)
    L = len(xyz)
    box = np.array([xyz.min(0), xyz.max(0)])
    log(f"L={L}, box={box[0].round(1)} .. {box[1].round(1)}")

    # 冻结扫描线 (只作校验声明, 不产出)
    sl = frozen_scanlines(box)
    log(f"held-out scanlines: {len(sl)} (min pairwise dist "
        f"{np.min(np.linalg.norm(sl[:, None] - sl[None], axis=-1) + np.eye(len(sl)) * 1e9):.1f} m)")

    manifest = dict(
        task="R110", platform="hrf_masking_v1",
        source=dict(path=SRC, sha256=sha, size_bytes=os.path.getsize(SRC)),
        scenarios=[],
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        deterministic=True,
        note="M_obs json 无时间戳字段, 同环境双跑逐位一致; manifest 含时间戳仅供溯源",
    )

    def run(scenario, obs, extra):
        m, support = build_m_obs(scenario, obs, fid, extra, box, box)
        save_scenario(scenario, obs, fid, m, support)
        manifest["scenarios"].append(dict(
            scenario=scenario, coverage_fraction=m["coverage_fraction"],
            n_observed_faces=len(m["observed_faces"])))

    # 1) full
    run("full", mask_full(xyz, fid), {})

    # 2) borehole 1/2/4
    for k in (1, 2, 4):
        obs, holes = mask_borehole(xyz, fid, box, k)
        run(f"borehole_{k}", obs, dict(
            holes=[dict(x=float(h[0]), y=float(h[1]),
                        z_top=float(box[1][2] + 5), z_bot=float(box[0][2] - 5),
                        radius=BOREHOLE_RADIUS) for h in holes]))

    # 3) outcrop 10/30/50
    for f in (0.10, 0.30, 0.50):
        obs, geo = mask_outcrop(xyz, fid, box, f)
        run(f"outcrop_{int(f * 100)}", obs, geo)

    # 4) station 30
    obs, geo = mask_station(xyz, nrm, fid, box, 0.30)
    run("station_30", obs, geo)

    # 5) multi = outcrop_30 ∪ borehole_2
    obs_o, geo_o = mask_outcrop(xyz, fid, box, 0.30)
    obs_b, holes = mask_borehole(xyz, fid, box, 2)
    run("multi_30_2", obs_o | obs_b, dict(
        components=["outcrop_30", "borehole_2"],
        outcrop=geo_o,
        boreholes=[dict(x=float(h[0]), y=float(h[1]),
                        z_top=float(box[1][2] + 5), z_bot=float(box[0][2] - 5),
                        radius=BOREHOLE_RADIUS) for h in holes]))

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"manifest saved -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
