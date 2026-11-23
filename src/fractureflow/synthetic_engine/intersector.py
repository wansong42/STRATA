# -*- coding: utf-8 -*-
"""R183-T2b 平面×基面解析求交器。

数学：S(u,v) = O + u·eu + v·ev + n_out·(A·r(u,v))；裂缝平面 n·x = d。
  f(u,v) := n·S(u,v) − d = c0 + u·(n·eu) + v·(n·ev) + A·(n·n_out)·r(u,v) − d
基面 r 沿格边线性（双线性）→ f 沿格边线性 → 边交点解析精确：
  t = F0/(F0−F1)，交点严格落在基面与裂缝平面上。
可微结构（任务书：本单不验可微，只留结构）：逐顶点记录 (edge_kind,i,j,t)
及两端格点线性权重 (1−t, t)，后续可装配 (n,d,格值)→顶点 的雅可比。
等值线配对 = marching-squares 中心值规则（含鞍点）+ 循环序兜底。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace3D:
    """一条 3D 迹线（可含多个折线分支）。"""

    fracture_idx: int
    set_id: int
    n: np.ndarray
    d: float
    polylines_uv: list = field(default_factory=list)     # list of (M,2) u,v 米
    polylines_world: list = field(default_factory=list)  # list of (M,3)
    vertex_weights: list = field(default_factory=list)   # 可微结构留线：[seg][{edge,i,j,t,w}]
    grazing: bool = False


def _edge_crossings(F: np.ndarray, cell_u: float, cell_v: float, R: np.ndarray):
    """全部格边交点。

    ("h", i, j) = 节点(i,j)-(i+1,j)（u 向边）；("v", i, j) = (i,j)-(i,j+1)（v 向边）。
    每条边至多一个交点（f 沿边线性）。
    """
    Fs = np.where(F == 0.0, 1e-300, F)  # 恰零退化守卫
    h_idx = np.argwhere(Fs[:-1, :] * Fs[1:, :] < 0)
    v_idx = np.argwhere(Fs[:, :-1] * Fs[:, 1:] < 0)
    edges = {}
    for i, j in h_idx:
        f0, f1 = F[i, j], F[i + 1, j]
        t = float(f0 / (f0 - f1))
        edges[("h", int(i), int(j))] = {
            "t": t,
            "uv": ((int(i) + t) * cell_u, int(j) * cell_v),
            "w": (1.0 - t, t),
        }
    for i, j in v_idx:
        f0, f1 = F[i, j], F[i, j + 1]
        t = float(f0 / (f0 - f1))
        edges[("v", int(i), int(j))] = {
            "t": t,
            "uv": (int(i) * cell_u, (int(j) + t) * cell_v),
            "w": (1.0 - t, t),
        }
    return edges


def _cell_pairs(i, j, edges, F):
    """单元 (i,j) 内交点配对。返回 [(key_a, key_b), ...]。"""
    bl = ("h", i, j) in edges or ("v", i, j) in edges  # noqa: F841（可读性）
    cand = {}
    if ("h", i, j) in edges:
        cand["bottom"] = ("h", i, j)
    if ("h", i, j + 1) in edges:
        cand["top"] = ("h", i, j + 1)
    if ("v", i, j) in edges:
        cand["left"] = ("v", i, j)
    if ("v", i + 1, j) in edges:
        cand["right"] = ("v", i + 1, j)
    if len(cand) < 2:
        return []
    Fc = 0.25 * (F[i, j] + F[i + 1, j] + F[i, j + 1] + F[i + 1, j + 1])
    center_pos = bool(Fc > 0)
    signs = {
        "bl": F[i, j] > 0,
        "br": F[i + 1, j] > 0,
        "tr": F[i + 1, j + 1] > 0,
        "tl": F[i, j + 1] > 0,
    }
    adj = {
        "bl": ("bottom", "left"),
        "br": ("bottom", "right"),
        "tr": ("top", "right"),
        "tl": ("top", "left"),
    }
    pairs, used = [], set()
    for corner, s in signs.items():
        if s == center_pos:
            continue
        e1, e2 = adj[corner]
        if e1 in cand and e2 in cand and e1 not in used and e2 not in used:
            pairs.append((cand[e1], cand[e2]))
            used.update((e1, e2))
    if not pairs:
        order = [e for e in ("bottom", "right", "top", "left") if e in cand]
        pairs = [(cand[a], cand[b]) for a, b in zip(order[0::2], order[1::2])]
    return pairs


def intersect_fractures(dfn, surface, origin, eu, ev, no, terzaghi_censor: float = 0.08):
    """对每个盘求 3D 迹线。返回 (Trace3D 列表, 掠射计数)。"""
    O = np.asarray(origin, dtype=np.float64)
    traces: list[Trace3D] = []
    grazing_count = 0
    nu, nv = surface.lattice, surface.lattice

    for idx in range(len(dfn.radii)):
        n = dfn.normals[idx]
        d = float(n @ dfn.centers[idx])
        if abs(float(n @ no)) < terzaghi_censor:
            grazing_count += 1
            traces.append(Trace3D(fracture_idx=idx, set_id=int(dfn.set_ids[idx]), n=n, d=d, grazing=True))
            continue
        c0 = float(n @ O) - d
        cu = float(n @ eu)
        cv = float(n @ ev)
        kr = surface.amplitude * float(n @ no)
        xs, ys = surface.lattice_uv()
        U, V = np.meshgrid(xs, ys, indexing="ij")
        F = c0 + U * cu + V * cv + kr * surface.R
        if F.min() > 0 or F.max() < 0:
            continue  # 平面不切窗口内基面
        edges = _edge_crossings(F, surface.cell_u, surface.cell_v, surface.R)
        if not edges:
            continue
        # 单元配对 → 线段池
        segs = {}  # k -> (key_a, key_b, pa, pb)
        adj: dict = {}
        cells = set()
        for (_, i, j) in edges:
            cells.update(((i, j), (i, max(j - 1, 0)), (max(i - 1, 0), j)))
        for (i, j) in cells:
            if not (0 <= i < nu - 1 and 0 <= j < nv - 1):
                continue
            for ka, kb in _cell_pairs(i, j, edges, F):
                k = len(segs)
                segs[k] = (ka, kb, edges[ka]["uv"], edges[kb]["uv"], edges[ka]["w"], edges[kb]["w"])
                adj.setdefault(ka, []).append(k)
                adj.setdefault(kb, []).append(k)
                ka, kb = ka, kb
        if not segs:
            continue
        # 折线重建：从边界端（度 1）出发走链，再补闭环
        visited = {k: False for k in segs}
        start_order = [k for k in segs if len(adj[segs[k][0]]) == 1 or len(adj[segs[k][1]]) == 1]
        start_order += [k for k in segs if not visited[k]]

        def walk(k0: int, key0):
            chain, wrec = [], []
            k, key = k0, key0
            while True:
                if visited[k]:
                    break
                visited[k] = True
                ka, kb, pa, pb, wa, wb = segs[k]
                if key == ka:
                    chain.append(pa)
                    chain.append(pb)
                    wrec.append((ka, wa[0], wa[1], pb))
                    nxt = kb
                else:
                    chain.append(pb)
                    chain.append(pa)
                    wrec.append((kb, wb[0], wb[1], pa))
                    nxt = ka
                key = nxt  # 出口边 = 下一段共享边（入口端）
                nxt_segs = [m for m in adj[key] if not visited[m]]
                if not nxt_segs:
                    break
                k = nxt_segs[0]
            return chain, wrec

        polylines_uv, weights_all = [], []
        for k0 in start_order:
            if visited[k0]:
                continue
            ka, kb = segs[k0][0], segs[k0][1]
            key0 = ka if len(adj[ka]) == 1 else kb
            chain, wrec = walk(k0, key0)
            ded = [chain[0]]
            for p in chain[1:]:
                if np.linalg.norm(np.asarray(p) - np.asarray(ded[-1])) > 1e-9:
                    ded.append(p)
            if len(ded) >= 2:
                polylines_uv.append(np.asarray(ded, dtype=np.float64))
                weights_all.append(wrec)
        if not polylines_uv:
            continue
        # 盘域裁剪（迹线在盘平面上：到盘心欧氏距离 = 弦半径）
        C = dfn.centers[idx]
        Rad = float(dfn.radii[idx])
        final_uv, final_world, final_w = [], [], []
        for pl, wrec in zip(polylines_uv, weights_all):
            u, v = pl[:, 0], pl[:, 1]
            rr, _, _ = surface.eval_at(u, v)
            world = (
                O
                + u[:, None] * eu[None, :]
                + v[:, None] * ev[None, :]
                + no[None, :] * (surface.amplitude * rr)[:, None]
            )
            dist = np.linalg.norm(world - C[None, :], axis=1)
            inside = dist <= Rad
            if not inside.any():
                continue
            k = 0
            runs = []
            while k < len(pl):
                if inside[k]:
                    m = k
                    while m + 1 < len(pl) and inside[m + 1]:
                        m += 1
                    runs.append((k, m))
                    k = m + 1
                else:
                    k += 1
            for (a, b) in runs:
                seg_pts = []
                if a > 0 and dist[a] != dist[a - 1]:
                    t = (Rad - dist[a - 1]) / (dist[a] - dist[a - 1])
                    seg_pts.append(pl[a - 1] + t * (pl[a] - pl[a - 1]))
                seg_pts.extend(pl[a : b + 1])  # run 内全部顶点（修复：此前丢失中间点）
                if b < len(pl) - 1 and dist[b + 1] != dist[b]:
                    t = (Rad - dist[b]) / (dist[b + 1] - dist[b])
                    seg_pts.append(pl[b] + t * (pl[b + 1] - pl[b]))
                if len(seg_pts) < 2:
                    continue
                sp = np.asarray(seg_pts, dtype=np.float64)
                su, sv = sp[:, 0], sp[:, 1]
                rr2, _, _ = surface.eval_at(su, sv)
                wpts = (
                    O
                    + su[:, None] * eu[None, :]
                    + sv[:, None] * ev[None, :]
                    + no[None, :] * (surface.amplitude * rr2)[:, None]
                )
                final_uv.append(sp)
                final_world.append(wpts)
        if not final_uv:
            continue
        traces.append(
            Trace3D(
                fracture_idx=idx,
                set_id=int(dfn.set_ids[idx]),
                n=n,
                d=d,
                polylines_uv=final_uv,
                polylines_world=final_world,
                vertex_weights=weights_all,
                grazing=False,
            )
        )
    return traces, grazing_count


def implicit_band_pixels(traces, surface, origin, eu, ev, no, res_px, m_per_px, width_m, height_m, dfn=None):
    """独立解析 GT：逐裂缝在像素网格回代 f=n·S−d，符号变边像素带，
    并回代世界坐标按盘半径过滤（迹线 = 盘∩基面，非整平面零集）。

    与 marching 几何完全独立的第二评估路径（G1-A 独立性来源）。
    返回 bool 图 (res_px,res_px)，True = 迹线带像素。
    """
    O = np.asarray(origin, dtype=np.float64)
    ii = np.arange(res_px)
    cols = (ii + 0.5) * m_per_px
    rows_m = height_m - (ii + 0.5) * m_per_px  # row0 = 顶 = v=height
    U, V = np.meshgrid(cols, rows_m, indexing="xy")  # (row,col)
    r_grid, _, _ = surface.eval_at(U, V)
    band = np.zeros((res_px, res_px), dtype=bool)
    for tr in traces:
        if tr.grazing or not tr.polylines_uv:
            continue
        n = tr.n
        c0 = float(n @ O) - tr.d
        cu = float(n @ eu)
        cv = float(n @ ev)
        kr = surface.amplitude * float(n @ no)
        F = c0 + U * cu + V * cv + kr * r_grid
        cx = (F[:, :-1] * F[:, 1:]) < 0
        cy = (F[:-1, :] * F[1:, :]) < 0
        tb = np.zeros((res_px, res_px), dtype=bool)
        rr, cc = np.where(cx)
        tb[rr, cc] = True
        tb[rr, cc + 1] = True
        rr, cc = np.where(cy)
        tb[rr, cc] = True
        tb[rr + 1, cc] = True
        # 盘半径过滤（+1px 余量）：迹线 = 盘∩基面，不是整平面零集
        if dfn is not None and tb.any():
            ys, xs = np.where(tb)
            um = (xs + 0.5) * m_per_px
            vm = height_m - (ys + 0.5) * m_per_px
            rm, _, _ = surface.eval_at(um, vm)
            wm = (
                O
                + um[:, None] * eu[None, :]
                + vm[:, None] * ev[None, :]
                + no[None, :] * (surface.amplitude * rm)[:, None]
            )
            Rad = float(dfn.radii[tr.fracture_idx])
            Cc = dfn.centers[tr.fracture_idx][None, :]
            keep = np.linalg.norm(wm - Cc, axis=1) <= Rad + m_per_px
            tb[ys[~keep], xs[~keep]] = False
        band |= tb
    return band
