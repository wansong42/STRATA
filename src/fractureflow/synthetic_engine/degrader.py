# -*- coding: utf-8 -*-
"""R183-T3 碎片化模拟器（可控降质算子四类 → GT/OBS 双契约）。

四算子（任务书 §2-T3）：
  1 segment_break    泊松断点
  2 endpoint_erosion 端点侵蚀
  3 missed_detection 随机丢段（植被遮挡与短段偏置耦合）
  4 adhesion         近距粘连（异实例段合并为一条观测记录）
输出：降质前 GT 契约 + 降质后 OBS 契约（段级 id）+ diff_manifest
（R185 碎片化校正层的训练场）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fractureflow.fracture_instance import (
    Extent,
    FractureInstance,
    FractureInstanceSet,
    Plane,
    Provenance,
    SegmentQuality,
    TraceSegment,
    utc_now,
)


@dataclass
class DegradeConfig:
    lam_break_per_m: float = 0.7
    erosion_frac: float = 0.12
    p_drop_base: float = 0.12
    drop_occlusion_mult: float = 3.0
    short_bias_px: float = 12.0
    merge_gap_px: float = 3.0
    max_merge_members: int = 4


def _resample_polyline(pl: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """折线弧长参数化取点。positions 为弧长（px）。"""
    seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_out = np.clip(positions, 0.0, s[-1])
    idx = np.searchsorted(s, s_out, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (s_out - s[idx]) / np.maximum(seg[idx], 1e-12)
    return pl[idx] + t[:, None] * (pl[idx + 1] - pl[idx])


def _occlusion_frac(pts_px: np.ndarray, visibility: np.ndarray, res: int) -> float:
    c = np.clip(np.round(pts_px).astype(np.int64), 0, res - 1)
    vis = visibility[c[:, 1], c[:, 0]] > 0
    return float(1.0 - vis.mean()) if len(vis) else 0.0


def degrade(
    gt: FractureInstanceSet,
    visibility: np.ndarray,
    rng: np.random.Generator,
    cfg: DegradeConfig,
    scene_id: str,
) -> tuple[FractureInstanceSet, dict]:
    res = visibility.shape[0]
    obs_instances: list[FractureInstance] = []
    diff = {"scene_id": scene_id, "params": cfg.__dict__, "per_parent": {}}

    # 第一遍：断点 + 侵蚀 + 丢段
    obs_pool = []  # (parent_id, seg_id, pts_px, mark, occ, parent_plane, parent_resid)
    for inst in gt.instances:
        parent_segs = inst.extent.segments
        n_children = 0
        n_dropped = 0
        breaks_total = 0
        erosion_total = 0.0
        for seg in parent_segs:
            if seg.pts_px is None or len(seg.pts_px) < 2:
                continue
            pl = np.asarray(seg.pts_px, dtype=np.float64)
            seglen = float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
            len_m = seglen * gt.m_per_px
            # 1) 泊松断点
            n_br = int(rng.poisson(cfg.lam_break_per_m * max(len_m, 1e-9)))
            n_br = min(n_br, max(int(seglen // 4), 0))
            breaks_total += n_br
            if n_br > 0:
                cuts = np.sort(rng.random(n_br) * seglen)
                bounds = np.concatenate([[0.0], cuts, [seglen]])
                pieces = [(_resample_polyline(pl, np.linspace(bounds[i], bounds[i + 1], 8))) for i in range(len(bounds) - 1)]
            else:
                pieces = [pl]
            # 2) 侵蚀 + 3) 丢段
            for k, piece in enumerate(pieces):
                plen = float(np.sum(np.linalg.norm(np.diff(piece, axis=0), axis=1)))
                e0 = float(rng.random() * cfg.erosion_frac * plen)
                e1 = float(rng.random() * cfg.erosion_frac * plen)
                s = np.linspace(0.0, plen, max(int(plen // 2) + 2, 8))
                if plen > e0 + e1 + 2.0:
                    piece = _resample_polyline(piece, s[(s >= e0) & (s <= plen - e1)])
                erosion_total += e0 + e1
                if len(piece) < 2:
                    n_dropped += 1
                    continue
                occ = _occlusion_frac(piece, visibility, res)
                short_bias = float(np.clip(plen / cfg.short_bias_px, 0.3, 1.0))
                p_drop = float(np.clip(cfg.p_drop_base * (1.0 + (cfg.drop_occlusion_mult - 1.0) * occ) * short_bias, 0.0, 0.95))
                if rng.random() < p_drop:
                    n_dropped += 1
                    continue
                mark = "clean"
                if n_br > 0:
                    mark = "broken"
                if e0 + e1 > 0.5:
                    mark = "eroded" if mark != "broken" else "broken"
                if occ > 0.5:
                    mark = "occluded"
                n_children += 1
                obs_pool.append(
                    {
                        "parent": inst.id,
                        "seg_id": f"{inst.id}#obs{n_children - 1}",
                        "pts_px": piece,
                        "mark": mark,
                        "occ": occ,
                        "plane": inst.plane,
                        "resid": inst.quality.residual_m,
                    }
                )
        diff["per_parent"][inst.id] = {
            "breaks": breaks_total,
            "n_children": n_children,
            "n_dropped": n_dropped,
            "erosion_px_total": round(erosion_total, 2),
        }

    # 4) 粘连：异父段端点近距（≤ merge_gap_px）合并（union-find，限成员数）
    parent_of = {i: i for i in range(len(obs_pool))}

    def find(x):
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_of[max(ra, rb)] = min(ra, rb)

    ends = []
    for i, o in enumerate(obs_pool):
        pl = o["pts_px"]
        ends.append((i, 0, pl[0]))
        ends.append((i, -1, pl[-1]))
    gap2 = cfg.merge_gap_px**2
    for a in range(len(ends)):
        ia, _, pa = ends[a]
        for b in range(a + 1, len(ends)):
            ib, _, pb = ends[b]
            if find(ia) == find(ib):
                continue
            if obs_pool[ia]["parent"] == obs_pool[ib]["parent"]:
                continue
            if float(np.sum((pa - pb) ** 2)) <= gap2:
                union(ia, ib)

    groups: dict = {}
    for i in range(len(obs_pool)):
        groups.setdefault(find(i), []).append(i)

    merged_pairs = 0
    for root, members in groups.items():
        if len(members) > cfg.max_merge_members:
            members = members[: cfg.max_merge_members]
        ids = [obs_pool[i]["seg_id"] for i in members]
        parents = sorted({obs_pool[i]["parent"] for i in members})
        oid = "+".join(ids) if len(members) > 1 else ids[0]
        pts = np.vstack([obs_pool[i]["pts_px"] for i in members])
        mark = "merged" if len(members) > 1 else obs_pool[members[0]]["mark"]
        occ = float(np.mean([obs_pool[i]["occ"] for i in members]))
        lo2 = pts.min(0).tolist()
        hi2 = pts.max(0).tolist()
        obs_instances.append(
            FractureInstance(
                id=oid,
                plane=obs_pool[members[0]]["plane"],
                extent=Extent(
                    bbox_min=lo2 + [0.0],
                    bbox_max=hi2 + [0.0],
                    segments=[
                        TraceSegment(
                            seg_id=oid,
                            pts_px=pts.tolist(),
                            width_px=2.0,
                            mark=mark,
                            quality=SegmentQuality(1.0 - occ, obs_pool[members[0]]["resid"], "pass" if occ < 0.5 else "warn"),
                        )
                    ],
                ),
                quality=SegmentQuality(1.0 - occ, obs_pool[members[0]]["resid"], "pass" if occ < 0.5 else "warn"),
                provenance=Provenance(
                    source="synthetic_engine_v0",
                    method="degrader_v0",
                    method_version="0.1.0",
                    created_at=utc_now(),
                    scene_id=scene_id,
                    notes=(f"coordinate_frame=pixel; parents={parents}; members={ids}" if len(members) > 1 else "coordinate_frame=pixel"),
                ),
            )
        )
        if len(members) > 1:
            merged_pairs += 1
            diff["per_parent"].setdefault(parents[0], {})["merged_with"] = ids[1:]

    diff["n_obs_instances"] = len(obs_instances)
    diff["n_merged_records"] = merged_pairs
    obs_set = FractureInstanceSet(
        scene_id=scene_id + "_obs",
        m_per_px=gt.m_per_px,
        frame=dict(gt.frame),
        instances=obs_instances,
    )
    return obs_set, diff
