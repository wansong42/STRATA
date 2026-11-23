# -*- coding: utf-8 -*-
"""Read-only verification & active-identifiability bridge for the STRATA workbench.

Strictly enforces M5 / Master Plan Amendment B2 invariants:
  * Real-derived conditioned scope (`REAL_DERIVED_CONDITIONED_FINITE_UNION`) is
    permanently separated from synthetic scope (`SYNTHETIC_DECLARED_TRIANGLE_UNION`).
  * Every cluster load recomputes `model.json` canonical & file SHA256 fresh on
    disk and checks it against `cluster_register.json` and (when present)
    M2 `certification.json#input_sha256` — any 1-byte tamper raises HTTP 409.
  * Automatically mounts frozen M2 certification/active_check packages and M3
    independent judge verdicts when present on disk (`PRECOMPUTED_VERIFIED_TREE`),
    and transparently falls back to the exact rational solver (`LIVE_EXACT_SOLVER`,
    verified via `recheck.verify`) when M2 artifacts are not yet on disk or when
    an explicit multi-step check goes beyond the precomputed tree.
  * Never fabricates results, never mutates M1/M2/M3 artifacts, never invokes
    external shells or pops up console windows.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .contracts import ContractError
from .workbench_service import ROOT

REAL_SCHEMA = "TC-network/real-conditioned/v1"
REAL_SCOPE = "REAL_DERIVED_CONDITIONED_FINITE_UNION"
SYNTHETIC_SCOPE = "SYNTHETIC_DECLARED_TRIANGLE_UNION"

DEFAULT_M1_ROOT = ROOT / "results" / "midterm_sprint_20261120" / "M1_real_to_exact_bridge"
DEFAULT_M2_ROOT = ROOT / "results" / "midterm_sprint_20261120" / "M2_active_identifiability"
DEFAULT_M3_ROOT = ROOT / "results" / "midterm_sprint_20261120" / "M3_independent_judge"
TC_NETWORK_DIR = (
    ROOT
    / "results"
    / "w180_technical_route_20261117"
    / "execution_next"
    / "dispatch_20261118"
    / "TC_network"
)


class IdentifiabilityError(ContractError):
    """Structured error with explicit HTTP status for /api/identifiability/*."""

    def __init__(
        self,
        message: str,
        code: str = "IDENTIFIABILITY_ERROR",
        http_status: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code)
        self.http_status = int(http_status)
        self.details = details or {}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frac(val: Any) -> Fraction:
    if isinstance(val, Fraction):
        return val
    if isinstance(val, (int, str)):
        return Fraction(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            raise IdentifiabilityError(f"non-finite numeric value: {val!r}", "INVALID_RATIONAL", 409)
        return Fraction(str(val))
    raise IdentifiabilityError(f"cannot parse rational: {val!r}", "INVALID_RATIONAL", 409)


def _qs(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def _vec_float(raw: Sequence[Any]) -> List[float]:
    return [float(_frac(c)) for c in raw]


def _vec_add_scaled(p: Sequence[float], v: Sequence[float], t: float) -> List[float]:
    return [p[0] + t * v[0], p[1] + t * v[1], p[2] + t * v[2]]


def _vsub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vscale(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vdist2(a: Sequence[float], b: Sequence[float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _closest_pt_point_triangle(
    p: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> Tuple[float, float, float]:
    """Exact Euclidean closest point on closed triangle (a,b,c) to point p (Ericson RTCD)."""
    ab = _vsub(b, a)
    ac = _vsub(c, a)
    ap = _vsub(p, a)
    d1 = _vdot(ab, ap)
    d2 = _vdot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return (a[0], a[1], a[2])

    bp = _vsub(p, b)
    d3 = _vdot(ab, bp)
    d4 = _vdot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return (b[0], b[1], b[2])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if (d1 - d3) != 0.0 else 0.0
        return _vadd(a, _vscale(ab, v))

    cp = _vsub(p, c)
    d5 = _vdot(ab, cp)
    d6 = _vdot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return (c[0], c[1], c[2])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if (d2 - d6) != 0.0 else 0.0
        return _vadd(a, _vscale(ac, w))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        return _vadd(b, _vscale(_vsub(c, b), w))

    denom = va + vb + vc
    if denom == 0.0:
        return (a[0], a[1], a[2])
    v = vb / denom
    w = vc / denom
    return (
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    )


def _closest_pts_seg_seg(
    p1: Sequence[float],
    q1: Sequence[float],
    p2: Sequence[float],
    q2: Sequence[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    d1 = _vsub(q1, p1)
    d2 = _vsub(q2, p2)
    r = _vsub(p1, p2)
    a = _vdot(d1, d1)
    e = _vdot(d2, d2)
    f = _vdot(d2, r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return (p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = _vdot(d1, r)
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = _vdot(d1, d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    return _vadd(p1, _vscale(d1, s)), _vadd(p2, _vscale(d2, t))


def _closest_pts_tri_tri(
    t1: Sequence[Sequence[float]],
    t2: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """Approximate/exact closest point pair between two 3D triangles for 3D visualization."""
    best_d2 = float("inf")
    best_a = (t1[0][0], t1[0][1], t1[0][2])
    best_b = (t2[0][0], t2[0][1], t2[0][2])

    # Vertex-to-triangle checks (6 pairs)
    for v in t1:
        q = _closest_pt_point_triangle(v, t2[0], t2[1], t2[2])
        d2 = _vdist2(v, q)
        if d2 < best_d2:
            best_d2, best_a, best_b = d2, (v[0], v[1], v[2]), q
    for v in t2:
        q = _closest_pt_point_triangle(v, t1[0], t1[1], t1[2])
        d2 = _vdist2(q, v)
        if d2 < best_d2:
            best_d2, best_a, best_b = d2, q, (v[0], v[1], v[2])

    # Edge-to-edge checks (9 pairs)
    edges1 = ((t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0]))
    edges2 = ((t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0]))
    for e1a, e1b in edges1:
        for e2a, e2b in edges2:
            pa, pb = _closest_pts_seg_seg(e1a, e1b, e2a, e2b)
            d2 = _vdist2(pa, pb)
            if d2 < best_d2:
                best_d2, best_a, best_b = d2, pa, pb

    # Edge-triangle intersection check for penetrating non-coplanar triangles
    def _seg_tri_hit(p0: Sequence[float], p1: Sequence[float], tri: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
        d = _vsub(p1, p0)
        e1 = _vsub(tri[1], tri[0])
        e2 = _vsub(tri[2], tri[0])
        h = (d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0])
        a = _vdot(e1, h)
        if abs(a) < 1e-12:
            return None
        f = 1.0 / a
        s = _vsub(p0, tri[0])
        u = f * _vdot(s, h)
        if u < -1e-7 or u > 1.0 + 1e-7:
            return None
        q = (s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0])
        v = f * _vdot(d, q)
        if v < -1e-7 or u + v > 1.0 + 1e-7:
            return None
        t = f * _vdot(e2, q)
        if -1e-7 <= t <= 1.0 + 1e-7:
            t_clamped = max(0.0, min(1.0, t))
            return _vadd(p0, _vscale(d, t_clamped))
        return None

    hits: List[Tuple[float, float, float]] = []
    for e1a, e1b in edges1:
        hit = _seg_tri_hit(e1a, e1b, t2)
        if hit is not None:
            hits.append(hit)
    for e2a, e2b in edges2:
        hit = _seg_tri_hit(e2a, e2b, t1)
        if hit is not None:
            hits.append(hit)
    if hits:
        mid = (
            sum(p[0] for p in hits) / len(hits),
            sum(p[1] for p in hits) / len(hits),
            sum(p[2] for p in hits) / len(hits),
        )
        return mid, mid, 0.0

    return best_a, best_b, math.sqrt(max(0.0, best_d2))


def _tri_centroid(verts: Sequence[Sequence[float]]) -> List[float]:
    return [
        (verts[0][0] + verts[1][0] + verts[2][0]) / 3.0,
        (verts[0][1] + verts[1][1] + verts[2][1]) / 3.0,
        (verts[0][2] + verts[1][2] + verts[2][2]) / 3.0,
    ]


def _short_fracture_label(fid: str, role: str) -> str:
    tag = {
        "ANCHOR_A": "A锚",
        "ANCHOR_B": "B锚",
        "INTERIOR_F1": "F1",
        "INTERIOR_F2": "F2",
        "INTERIOR": "内",
    }.get(role, "")
    if fid.startswith("perozzo_row_"):
        core = "P" + fid.split("_")[-1]
    elif fid.startswith("frag_"):
        core = fid[:10]
    else:
        core = fid[:12]
    return f"{tag}·{core}" if tag else core


def _normalize_status(raw_status: Optional[str], exists_conn: Optional[bool], all_conn: Optional[bool]) -> str:
    if raw_status in (
        "CERTIFIED_CONNECTED",
        "CERTIFIED_DISCONNECTED",
        "AMBIGUOUS_WITH_VERIFIED_WITNESSES",
        "REFUSED",
        "INFEASIBLE",
        "BUDGET_EXCEEDED",
        "INFEASIBLE_OBSERVATIONS",
    ):
        return str(raw_status)
    if raw_status == "CERTIFIED_WITHIN_DECLARED_MODEL":
        if all_conn is True:
            return "CERTIFIED_CONNECTED"
        if exists_conn is False:
            return "CERTIFIED_DISCONNECTED"
    if exists_conn is True and all_conn is True:
        return "CERTIFIED_CONNECTED"
    if exists_conn is False and all_conn is False:
        return "CERTIFIED_DISCONNECTED"
    if exists_conn is True and all_conn is False:
        return "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
    return str(raw_status or "UNKNOWN")


def _bfs_connected(edges: Sequence[Sequence[str]], source: str = "@A", target: str = "@B") -> Optional[List[str]]:
    adj: Dict[str, Set[str]] = {}
    for edge in edges:
        if len(edge) != 2:
            continue
        u, v = str(edge[0]), str(edge[1])
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    if source not in adj or target not in adj:
        return None
    queue: List[str] = [source]
    parent: Dict[str, Optional[str]] = {source: None}
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == target:
            path: List[str] = []
            node: Optional[str] = target
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None


def _cell_measure(cell: Mapping[str, Any]) -> Fraction:
    if cell.get("kind") == "OPEN":
        lo = _frac(cell["lo"])
        hi = _frac(cell["hi"])
        return max(Fraction(0), hi - lo)
    return Fraction(0)


def _compute_stability_from_cells(
    model: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    ref = model.get("reference")
    if not ref or not cells:
        return None
    t0 = _frac(ref["t"])
    sigma = _frac(ref["sigma"])
    edits = {m["id"]: _frac(m["edit_cost"]) for m in model["modes"]}

    nominal = None
    for r in cells:
        if r.get("mode") != ref["mode"]:
            continue
        lo, hi = _frac(r["lo"]), _frac(r["hi"])
        if (r.get("kind") == "POINT" and lo == t0) or (r.get("kind") == "OPEN" and lo < t0 < hi):
            nominal = r
            break
    if nominal is None:
        # Reference point excluded by conditioning; measure flip distance against first available cell's polarity
        return {
            "infimum": None,
            "infimum_float": None,
            "attained": None,
            "nominal_connected": None,
            "nominal_in_conditioned_domain": False,
            "flip_witness": None,
        }

    flips = []
    for r in cells:
        if bool(r.get("connected")) == bool(nominal.get("connected")):
            continue
        lo, hi = _frac(r["lo"]), _frac(r["hi"])
        closest = min(hi, max(lo, t0))
        mode_id = str(r["mode"])
        edit_c = edits.get(mode_id, Fraction(0))
        infimum = max(abs(closest - t0) / sigma, edit_c)
        witness_t = closest
        if r.get("kind") == "OPEN" and witness_t in (lo, hi):
            if edit_c > abs(closest - t0) / sigma:
                step = min((hi - lo) / 2, (edit_c * sigma - abs(closest - t0)) / 2)
                witness_t += step if witness_t == lo else -step
            else:
                witness_t = (lo + hi) / 2
        cost = max(abs(witness_t - t0) / sigma, edit_c)
        flips.append((infimum, cost == infimum, cost, mode_id, witness_t, bool(r.get("connected"))))

    if not flips:
        return {
            "infimum": None,
            "infimum_float": None,
            "attained": None,
            "nominal_connected": bool(nominal.get("connected")),
            "nominal_in_conditioned_domain": True,
            "flip_witness": None,
        }
    lower = min(row[0] for row in flips)
    attained = any(row[0] == lower and row[1] for row in flips)
    best = min(flips, key=lambda row: (row[2], row[3], row[4]))
    return {
        "infimum": _qs(lower),
        "infimum_float": round(float(lower), 6),
        "attained": attained,
        "nominal_connected": bool(nominal.get("connected")),
        "nominal_in_conditioned_domain": True,
        "flip_witness": {
            "mode": best[3],
            "t": _qs(best[4]),
            "t_float": round(float(best[4]), 6),
            "connected": best[5],
            "cost": _qs(best[2]),
            "cost_float": round(float(best[2]), 6),
        },
    }


def _shifted_mode_triangles(
    model: Mapping[str, Any],
    mode_id: str,
    t_float: float,
) -> Dict[str, Dict[str, Any]]:
    mode_obj = next((m for m in model["modes"] if m["id"] == mode_id), None)
    if mode_obj is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for tri in mode_obj["triangles"]:
        verts0 = [_vec_float(p) for p in tri["vertices"]]
        vel = _vec_float(tri["v"])
        verts_t = [_vec_add_scaled(p, vel, t_float) for p in verts0]
        out[str(tri["id"])] = {
            "id": str(tri["id"]),
            "component": str(tri["component"]),
            "vertices": verts_t,
            "centroid": _tri_centroid(verts_t),
        }
    return out


def _enrich_connected_witness(
    model: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> Dict[str, Any]:
    mode_id = str(cell.get("mode") or cell.get("mode_id") or "m0")
    t_raw = cell.get("sample", cell.get("t", "0/1"))
    t_q = _frac(t_raw)
    t_f = float(t_q)
    raw_path = list(cell.get("path") or cell.get("path_triangles") or [])
    if raw_path and raw_path[0] != "@A":
        raw_path = ["@A"] + raw_path
    if raw_path and raw_path[-1] != "@B":
        raw_path = raw_path + ["@B"]

    tri_map = _shifted_mode_triangles(model, mode_id, t_f)
    ep_a = _vec_float(model["endpoints"]["A"])
    ep_b = _vec_float(model["endpoints"]["B"])

    path_triangles = [nid for nid in raw_path if not str(nid).startswith("@")]
    path_fractures: List[str] = []
    for tid in path_triangles:
        info = tri_map.get(tid)
        if info and info["component"] not in path_fractures:
            path_fractures.append(info["component"])

    polyline: List[List[float]] = [ep_a]
    contact_points: List[Dict[str, Any]] = []
    for i in range(len(raw_path) - 1):
        u, v = str(raw_path[i]), str(raw_path[i + 1])
        if u == "@A" and v in tri_map:
            cp = _closest_pt_point_triangle(ep_a, *tri_map[v]["vertices"])
            pt = [cp[0], cp[1], cp[2]]
            contact_points.append({"from": u, "to": v, "point": pt})
            polyline.append(pt)
            polyline.append(tri_map[v]["centroid"])
        elif v == "@B" and u in tri_map:
            cp = _closest_pt_point_triangle(ep_b, *tri_map[u]["vertices"])
            pt = [cp[0], cp[1], cp[2]]
            contact_points.append({"from": u, "to": v, "point": pt})
            polyline.append(pt)
            polyline.append(ep_b)
        elif u in tri_map and v in tri_map:
            pa, pb, _ = _closest_pts_tri_tri(tri_map[u]["vertices"], tri_map[v]["vertices"])
            pt = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0]
            contact_points.append({"from": u, "to": v, "point": pt})
            polyline.append(pt)
            polyline.append(tri_map[v]["centroid"])
    if len(polyline) == 1:
        polyline.append(ep_b)

    return {
        "mode": mode_id,
        "t": _qs(t_q),
        "t_float": round(t_f, 6),
        "cell_kind": str(cell.get("kind") or cell.get("cell_kind") or "POINT"),
        "cell_lo": str(cell.get("lo", _qs(t_q))),
        "cell_hi": str(cell.get("hi", _qs(t_q))),
        "path_nodes": raw_path,
        "path_triangles": path_triangles,
        "path_fractures": path_fractures,
        "active_edges": [list(e) for e in (cell.get("active_edges") or [])],
        "contact_points": contact_points,
        "path_polyline_3d": polyline,
    }


def _enrich_disconnected_witness(
    model: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> Dict[str, Any]:
    mode_id = str(cell.get("mode") or cell.get("mode_id") or "m0")
    t_raw = cell.get("sample", cell.get("t", "0/1"))
    t_q = _frac(t_raw)
    t_f = float(t_q)
    cut = cell.get("cut") or {}
    s_tris = list(cut.get("S") or cell.get("cut_S_triangles") or [])
    t_tris = list(cut.get("T") or cell.get("cut_T_triangles") or [])
    unattached = list(cut.get("unattached_endpoints") or cell.get("unattached_endpoints") or [])

    tri_map = _shifted_mode_triangles(model, mode_id, t_f)
    ep_a = _vec_float(model["endpoints"]["A"])
    ep_b = _vec_float(model["endpoints"]["B"])

    s_frags = sorted({tri_map[tid]["component"] for tid in s_tris if tid in tri_map})
    t_frags = sorted({tri_map[tid]["component"] for tid in t_tris if tid in tri_map})

    # Compute closest physical separation gap across S/T (or unattached endpoint)
    best_dist = float("inf")
    gap: Optional[Dict[str, Any]] = None
    if s_tris and t_tris:
        for sid in s_tris:
            if sid not in tri_map:
                continue
            for tid in t_tris:
                if tid not in tri_map:
                    continue
                pa, pb, dist = _closest_pts_tri_tri(tri_map[sid]["vertices"], tri_map[tid]["vertices"])
                if dist < best_dist:
                    best_dist = dist
                    gap = {
                        "from_id": sid,
                        "to_id": tid,
                        "from_fracture": tri_map[sid]["component"],
                        "to_fracture": tri_map[tid]["component"],
                        "from_pt": [pa[0], pa[1], pa[2]],
                        "to_pt": [pb[0], pb[1], pb[2]],
                        "distance_m": round(dist, 5),
                        "kind": "ST_TRIANGLE_GAP",
                    }
    elif unattached:
        for ep_name in unattached:
            ep_pt = ep_a if ep_name == "A" else ep_b
            for tid, info in tri_map.items():
                cp = _closest_pt_point_triangle(ep_pt, *info["vertices"])
                dist = math.sqrt(_vdist2(ep_pt, cp))
                if dist < best_dist:
                    best_dist = dist
                    gap = {
                        "from_id": "@" + ep_name,
                        "to_id": tid,
                        "from_fracture": "@" + ep_name,
                        "to_fracture": info["component"],
                        "from_pt": ep_pt,
                        "to_pt": [cp[0], cp[1], cp[2]],
                        "distance_m": round(dist, 5),
                        "kind": "UNATTACHED_ENDPOINT_GAP",
                    }

    return {
        "mode": mode_id,
        "t": _qs(t_q),
        "t_float": round(t_f, 6),
        "cell_kind": str(cell.get("kind") or cell.get("cell_kind") or "POINT"),
        "cell_lo": str(cell.get("lo", _qs(t_q))),
        "cell_hi": str(cell.get("hi", _qs(t_q))),
        "cut_S_triangles": s_tris,
        "cut_T_triangles": t_tris,
        "cut_S_fractures": s_frags,
        "cut_T_fractures": t_frags,
        "unattached_endpoints": unattached,
        "active_edges": [list(e) for e in (cell.get("active_edges") or [])],
        "cut_gap": gap,
    }


def _pick_representative_cells(
    model: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    rho: Optional[Mapping[str, Any]],
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """Select clean representative connected and disconnected cells (preferring nominal/lowest edit cost)."""
    ref = model.get("reference") or {"mode": "m0", "t": "0/1"}
    t0 = _frac(ref["t"])
    edits = {m["id"]: _frac(m["edit_cost"]) for m in model["modes"]}

    def _cell_rank(r: Mapping[str, Any]) -> Tuple[int, Fraction, Fraction, int, str]:
        mode_id = str(r.get("mode", "m0"))
        lo, hi = _frac(r["lo"]), _frac(r["hi"])
        sample = _frac(r.get("sample", lo))
        is_nom = 0 if (mode_id == ref["mode"] and ((r.get("kind") == "POINT" and lo == t0) or (r.get("kind") == "OPEN" and lo < t0 < hi))) else 1
        dist_t0 = abs(sample - t0)
        kind_pref = 0 if r.get("kind") == "OPEN" else 1
        return (is_nom, edits.get(mode_id, Fraction(999)), dist_t0, kind_pref, mode_id)

    conn_cells = [r for r in cells if bool(r.get("connected"))]
    disc_cells = [r for r in cells if not bool(r.get("connected"))]

    conn_best = min(conn_cells, key=_cell_rank) if conn_cells else None
    disc_best = min(disc_cells, key=_cell_rank) if disc_cells else None

    # If flip_witness exists, prefer the cell matching the nearest flip witness for the opposite polarity
    fw = (rho or {}).get("flip_witness") if isinstance(rho, Mapping) else None
    if isinstance(fw, Mapping) and fw.get("mode") is not None and fw.get("t") is not None:
        fw_t = _frac(fw["t"])
        fw_mode = str(fw["mode"])
        for r in cells:
            if str(r.get("mode")) != fw_mode:
                continue
            lo, hi = _frac(r["lo"]), _frac(r["hi"])
            hit = (r.get("kind") == "POINT" and lo == fw_t) or (r.get("kind") == "OPEN" and lo < fw_t < hi)
            if hit:
                if bool(r.get("connected")):
                    conn_best = r
                else:
                    disc_best = r
                break

    return conn_best, disc_best


def _compute_baseline_comparison(
    model: Mapping[str, Any],
    status: str,
    cells: Sequence[Mapping[str, Any]],
    contact_windows: Sequence[Mapping[str, Any]],
    rho: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute exact Naive Union Graph and Nominal (m0, t=0) baseline contrasts."""
    # Map triangle id -> component across modes
    tri_to_comp: Dict[str, str] = {}
    for m in model["modes"]:
        for tri in m["triangles"]:
            tri_to_comp[str(tri["id"])] = str(tri["component"])

    # Naive union edges: ANY edge active in ANY mode or ANY t
    union_edges: Set[Tuple[str, str]] = set()
    for w in contact_windows:
        if w.get("t_closed") is not None:
            u = str(w.get("from", ""))
            v = str(w.get("to", ""))
            if u and v:
                union_edges.add((u, v) if u <= v else (v, u))
    for r in cells:
        for edge in r.get("active_edges") or []:
            if len(edge) == 2:
                u, v = str(edge[0]), str(edge[1])
                union_edges.add((u, v) if u <= v else (v, u))

    union_path = _bfs_connected(sorted(union_edges))
    union_connected = union_path is not None
    union_frags: List[str] = []
    if union_path:
        for nid in union_path:
            if not nid.startswith("@"):
                comp = tri_to_comp.get(nid, nid)
                if comp not in union_frags:
                    union_frags.append(comp)

    if union_connected and status == "CERTIFIED_DISCONNECTED":
        union_failure = "FALSE_UNION_PATH"
        union_msg = (
            "朴素并集图将不同模式/不同参数时刻的接触边错误拼成一条贯通假路（误报连通），"
            "而精确同刻整网判定证明在全部合法 (m, t) 上均不贯通（CERTIFIED_DISCONNECTED）。"
        )
    elif union_connected and status == "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
        union_failure = "MASKS_DISCONNECTED_WITNESS"
        union_msg = (
            "朴素并集图单边报「连通」，彻底掩盖了合法参数域内存在的精确断割反例 (m-, t-)；"
            "若据此决策将忽略多解歧义风险。"
        )
    elif (not union_connected) and status == "CERTIFIED_DISCONNECTED":
        union_failure = "AGREES_DISCONNECTED"
        union_msg = "朴素并集图与精确整网结论一致（均判不贯通），本簇无跨时假通路。"
    elif union_connected and status == "CERTIFIED_CONNECTED":
        union_failure = "AGREES_CONNECTED"
        union_msg = "朴素并集图与精确整网结论一致（全部合法 (m, t) 均贯通）。"
    else:
        union_failure = "OBSERVED_CONTRAST"
        union_msg = f"朴素并集图判{'连通' if union_connected else '不贯通'}；精确整网状态为 {status}。"

    nom_conn = (rho or {}).get("nominal_connected") if isinstance(rho, Mapping) else None
    rho_str = (rho or {}).get("infimum") if isinstance(rho, Mapping) else None
    rho_f = None
    if rho_str is not None:
        try:
            rho_f = round(float(_frac(rho_str)), 4)
        except Exception:
            rho_f = None

    if status == "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
        if nom_conn is True:
            nom_failure = "SILENT_FALSE_SAFE_FLIP"
            nom_msg = (
                f"名义单一模型 (m0, t=0) 仅给出「连通」，但在距名义点仅 ρ={rho_f} 处存在合法断割反例，"
                "会产生名义静默假安全翻转。"
            )
        elif nom_conn is False:
            nom_failure = "SILENT_MISSED_CONNECTION_FLIP"
            nom_msg = (
                f"名义单一模型 (m0, t=0) 仅给出「不贯通」，但在距名义点仅 ρ={rho_f} 处存在合法贯通反例，"
                "会产生名义静默漏报连通翻转。"
            )
        else:
            nom_failure = "AMBIGUOUS_NOMINAL_EXCLUDED"
            nom_msg = "当前条件化子域已排除原名义点，但剩余域内仍存在双见证歧义。"
    else:
        nom_failure = "AGREES_WITH_CERTIFIED"
        nom_msg = (
            f"名义单一模型 (m0, t=0) 判{'连通' if nom_conn else '不贯通'}，与精确整网确定认证一致（全合法域无翻转）。"
            if nom_conn is not None
            else f"精确整网已认证为 {status}。"
        )

    return {
        "naive_union": {
            "connected": union_connected,
            "path_triangles": [n for n in (union_path or []) if not n.startswith("@")],
            "path_fractures": union_frags,
            "union_edge_count": len(union_edges),
            "failure_type": union_failure,
            "explanation": union_msg,
        },
        "nominal_single_point": {
            "mode": str((model.get("reference") or {}).get("mode", "m0")),
            "t": str((model.get("reference") or {}).get("t", "0/1")),
            "connected": nom_conn,
            "rho_float": rho_f,
            "failure_type": nom_failure,
            "explanation": nom_msg,
        },
    }


def _describe_feature_value(feat_val: str) -> str:
    mapping = {
        "ATTRIB_NONE": "基准归属（未向单侧偏置）",
        "ATTRIB_TO_F1": "边界支持归属 F1（F1 延展 1.20x / F2 收缩 0.92x）",
        "ATTRIB_TO_F2": "边界支持归属 F2（F2 延展 1.20x / F1 收缩 0.92x）",
        "EXT_1.00X": "名义有限边界（1.00x 不外延）",
        "EXT_1.15X": "边界外延 1.15x（覆盖遮挡端缘）",
        "TILT_0DEG": "名义倾角（0° 无偏转）",
        "TILT_POS_3.95DEG": "倾角正偏 +3.95°",
        "TILT_NEG_3.95DEG": "倾角负偏 -3.95°",
    }
    return mapping.get(feat_val, feat_val)


def _compute_exact_active_check(
    model: Mapping[str, Any],
    status: str,
    cells: Sequence[Mapping[str, Any]],
    contact_windows: Sequence[Mapping[str, Any]],
    conn_witness: Optional[Mapping[str, Any]],
    disc_witness: Optional[Mapping[str, Any]],
    input_sha256: str,
) -> Dict[str, Any]:
    """Exact zero-leakage active next-best-check ranker & collapse_tree builder.

    Operates strictly on exact rational cell partitions from `compile_network`,
    never peeking at hidden ground-truth labels.
    """
    if status != "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
        return {
            "status": "ALREADY_CERTIFIED",
            "provenance_mode": "LIVE_EXACT_SOLVER",
            "input_sha256": input_sha256,
            "ranked_fractures": [],
            "collapse_tree": {},
        }

    fcat = model.get("fracture_catalog") or {}
    msem = model.get("mode_semantics") or {}
    all_frags = sorted(fcat.keys())
    if not all_frags:
        all_frags = sorted({str(tri["component"]) for m in model["modes"] for tri in m["triangles"]})

    tri_to_comp: Dict[str, str] = {}
    for m in model["modes"]:
        for tri in m["triangles"]:
            tri_to_comp[str(tri["id"])] = str(tri["component"])

    # Pre-parse all cells ONCE so branch splits avoid repeated Fraction() parsing
    parsed_cells: List[Tuple[Mapping[str, Any], Fraction, Fraction, bool, Fraction, bool, str]] = []
    cells_by_mode: Dict[str, List[Tuple[Fraction, Fraction, bool, bool]]] = {}
    total_measure = Fraction(0)
    for r in cells:
        lo_q = _frac(r["lo"])
        hi_q = _frac(r["hi"])
        is_pt = r.get("kind") == "POINT"
        meas_q = Fraction(0) if is_pt else max(Fraction(0), hi_q - lo_q)
        conn_b = bool(r.get("connected"))
        mode_s = str(r["mode"])
        total_measure += meas_q
        parsed_cells.append((r, lo_q, hi_q, is_pt, meas_q, conn_b, mode_s))
        cells_by_mode.setdefault(mode_s, []).append((lo_q, hi_q, is_pt, conn_b))
    total_cells = max(1, len(parsed_cells))

    conn_path_frags = set((conn_witness or {}).get("path_fractures") or [])
    cut_s_frags = set((disc_witness or {}).get("cut_S_fractures") or [])
    cut_t_frags = set((disc_witness or {}).get("cut_T_fractures") or [])
    m_plus = (conn_witness or {}).get("mode")
    m_minus = (disc_witness or {}).get("mode")

    # Identify contact boundary knots per fracture (excluding same-fracture triangles)
    knots_by_frag: Dict[str, Set[Fraction]] = {fid: set() for fid in all_frags}
    for w in contact_windows:
        if w.get("t_closed") is None:
            continue
        u, v = str(w.get("from", "")), str(w.get("to", ""))
        cu = tri_to_comp.get(u)
        cv = tri_to_comp.get(v)
        if cu == cv and cu is not None:
            continue
        lo_q, hi_q = _frac(w["t_closed"][0]), _frac(w["t_closed"][1])
        for comp in (cu, cv):
            if comp in knots_by_frag:
                knots_by_frag[comp].add(lo_q)
                knots_by_frag[comp].add(hi_q)

    # Also find flip knots in cells (knots adjacent to a polarity flip)
    flip_knots_global: Set[Fraction] = set()
    for m_cells in cells_by_mode.values():
        m_sorted = sorted(m_cells, key=lambda c: (c[0], 0 if c[2] else 1))
        for a, b in zip(m_sorted, m_sorted[1:]):
            if a[3] != b[3]:
                flip_knots_global.add(a[1])
                flip_knots_global.add(b[0])

    def _eval_parsed_branch(
        branch_id: str,
        label: str,
        condition_summary: str,
        action_type: str,
        sub_parsed: List[Tuple[Mapping[str, Any], Fraction, Fraction, bool, Fraction, bool, str]],
        filter_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        has_true = False
        has_false = False
        sub_measure = Fraction(0)
        for row in sub_parsed:
            sub_measure += row[4]
            if row[5]:
                has_true = True
            else:
                has_false = True
            if has_true and has_false and sub_measure > 0:
                pass
        if not sub_parsed:
            b_status = "INFEASIBLE"
            exists_c, all_c = False, False
        elif has_true and not has_false:
            b_status = "CERTIFIED_CONNECTED"
            exists_c, all_c = True, True
        elif has_false and not has_true:
            b_status = "CERTIFIED_DISCONNECTED"
            exists_c, all_c = False, False
        else:
            b_status = "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
            exists_c, all_c = True, False

        return {
            "branch_id": branch_id,
            "label": label,
            "action_type": action_type,
            "condition_summary": condition_summary,
            "filter_spec": filter_spec,
            "resulting_status": b_status,
            "collapsed_to_certified": b_status in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"),
            "exists_connected": exists_c,
            "all_connected": all_c,
            "cell_count": len(sub_parsed),
            "measure_exact": _qs(sub_measure),
            "measure_float": round(float(sub_measure), 6),
            "_meas_q": sub_measure,
            "_sub_cells": [row[0] for row in sub_parsed],
        }

    def _materialize_branch_3d(b: Dict[str, Any]) -> Dict[str, Any]:
        b.pop("_meas_q", None)
        sub_cells = b.pop("_sub_cells", [])
        sub_rho = _compute_stability_from_cells(model, sub_cells) if sub_cells else None
        c_cell, d_cell = _pick_representative_cells(model, sub_cells, sub_rho)
        b["rho"] = sub_rho
        b["connected_witness"] = _enrich_connected_witness(model, c_cell) if c_cell is not None else None
        b["disconnected_witness"] = _enrich_disconnected_witness(model, d_cell) if d_cell is not None else None
        return b

    candidates: List[Dict[str, Any]] = []
    collapse_tree: Dict[str, Dict[str, Any]] = {}

    for fid in all_frags:
        meta = fcat.get(fid) or {}
        role = str(meta.get("role") or "INTERIOR")
        action_options: List[Tuple[str, List[Dict[str, Any]], str]] = []

        # 1) RESOLVE_MODE_FEATURE from model.mode_semantics
        modes_by_feat: Dict[str, List[str]] = {}
        for mid, sem in msem.items():
            fassign = (sem or {}).get("feature_assignments") or {}
            if fid in fassign:
                fval = str(fassign[fid])
                modes_by_feat.setdefault(fval, []).append(str(mid))
        if len(modes_by_feat) >= 2:
            branches: List[Dict[str, Any]] = []
            for fval in sorted(modes_by_feat.keys()):
                allowed_modes = set(modes_by_feat[fval])
                sub_parsed = [row for row in parsed_cells if row[6] in allowed_modes]
                if not sub_parsed:
                    continue
                branches.append(
                    _eval_parsed_branch(
                        branch_id=f"mode_{fval}",
                        label=f"分支 · 确认特征：{fval}",
                        condition_summary=f"{_describe_feature_value(fval)}（限定合法模式 {', '.join(sorted(allowed_modes))}）",
                        action_type="RESOLVE_MODE_FEATURE",
                        sub_parsed=sub_parsed,
                        filter_spec={"kind": "modes", "allowed_modes": sorted(allowed_modes), "feature_value": fval},
                    )
                )
            if len(branches) >= 2:
                action_options.append(("RESOLVE_MODE_FEATURE", branches, "核查离散模式特征归属/边界延展"))

        # 2) REFINE_PARAMETER_BAND at exact contact knots of fid (LT_GE & LE_GT exact 2-way partitions)
        cand_knots = sorted(k for k in knots_by_frag.get(fid, set()) if k in flip_knots_global)
        if not cand_knots:
            cand_knots = sorted(knots_by_frag.get(fid, set()))
        best_band_option: Optional[Tuple[Tuple[int, Fraction, Fraction, Fraction], List[Dict[str, Any]], str]] = None

        def _score_branches(brs: List[Dict[str, Any]]) -> Tuple[int, Fraction, Fraction, Fraction]:
            wc = max(
                (Fraction(0) if b["collapsed_to_certified"] else b["_meas_q"])
                for b in brs
            )
            res_m = sum((b["_meas_q"] for b in brs if b["collapsed_to_certified"]), Fraction(0))
            cert_f = Fraction(sum(1 for b in brs if b["collapsed_to_certified"]), len(brs))
            any_cert = 0 if res_m > 0 or cert_f > 0 else 1
            return (any_cert, wc, -res_m, -cert_f)

        def _split_lt_ge(knot: Fraction) -> Tuple[List[Any], List[Any]]:
            lt_list: List[Any] = []
            ge_list: List[Any] = []
            for row in parsed_cells:
                r, lo_q, hi_q, is_pt, meas_q, conn_b, mode_s = row
                if is_pt:
                    if lo_q < knot:
                        lt_list.append(row)
                    else:
                        ge_list.append(row)
                else:
                    if hi_q <= knot:
                        lt_list.append(row)
                    elif lo_q >= knot:
                        ge_list.append(row)
                    else:
                        # Exact split of invariant OPEN cell (lo_q, hi_q) at interior knot
                        mid_l = (lo_q + knot) / 2
                        mid_r = (knot + hi_q) / 2
                        r_l = dict(r, lo=_qs(lo_q), hi=_qs(knot), sample=_qs(mid_l), kind="OPEN")
                        r_p = dict(r, lo=_qs(knot), hi=_qs(knot), sample=_qs(knot), kind="POINT")
                        r_r = dict(r, lo=_qs(knot), hi=_qs(hi_q), sample=_qs(mid_r), kind="OPEN")
                        lt_list.append((r_l, lo_q, knot, False, knot - lo_q, conn_b, mode_s))
                        ge_list.append((r_p, knot, knot, True, Fraction(0), conn_b, mode_s))
                        ge_list.append((r_r, knot, hi_q, False, hi_q - knot, conn_b, mode_s))
            return lt_list, ge_list

        def _split_le_gt(knot: Fraction) -> Tuple[List[Any], List[Any]]:
            le_list: List[Any] = []
            gt_list: List[Any] = []
            for row in parsed_cells:
                r, lo_q, hi_q, is_pt, meas_q, conn_b, mode_s = row
                if is_pt:
                    if lo_q <= knot:
                        le_list.append(row)
                    else:
                        gt_list.append(row)
                else:
                    if hi_q <= knot:
                        le_list.append(row)
                    elif lo_q >= knot:
                        gt_list.append(row)
                    else:
                        mid_l = (lo_q + knot) / 2
                        mid_r = (knot + hi_q) / 2
                        r_l = dict(r, lo=_qs(lo_q), hi=_qs(knot), sample=_qs(mid_l), kind="OPEN")
                        r_p = dict(r, lo=_qs(knot), hi=_qs(knot), sample=_qs(knot), kind="POINT")
                        r_r = dict(r, lo=_qs(knot), hi=_qs(hi_q), sample=_qs(mid_r), kind="OPEN")
                        le_list.append((r_l, lo_q, knot, False, knot - lo_q, conn_b, mode_s))
                        le_list.append((r_p, knot, knot, True, Fraction(0), conn_b, mode_s))
                        gt_list.append((r_r, knot, hi_q, False, hi_q - knot, conn_b, mode_s))
            return le_list, gt_list

        for knot in cand_knots:
            # 2a) LT_GE: t < knot vs t >= knot (contact onset boundary)
            lt_parsed, ge_parsed = _split_lt_ge(knot)
            if lt_parsed and ge_parsed:
                b_lt = _eval_parsed_branch(
                    branch_id=f"band_lt:{_qs(knot)}",
                    label=f"分支 A · 确认 t < {float(knot):.4f}",
                    condition_summary=f"局部点云复核确认该裂隙接触处于 t < {_qs(knot)} ({float(knot):.4f}) 侧",
                    action_type="REFINE_PARAMETER_BAND",
                    sub_parsed=lt_parsed,
                    filter_spec={"kind": "t_lt", "knot": _qs(knot)},
                )
                b_ge = _eval_parsed_branch(
                    branch_id=f"band_ge:{_qs(knot)}",
                    label=f"分支 B · 确认 t ≥ {float(knot):.4f}",
                    condition_summary=f"局部点云复核确认该裂隙接触处于 t ≥ {_qs(knot)} ({float(knot):.4f}) 侧",
                    action_type="REFINE_PARAMETER_BAND",
                    sub_parsed=ge_parsed,
                    filter_spec={"kind": "t_ge", "knot": _qs(knot)},
                )
                brs = [b_lt, b_ge]
                sc = _score_branches(brs)
                if best_band_option is None or sc < best_band_option[0]:
                    best_band_option = (
                        sc,
                        brs,
                        f"局部点云复核该裂隙接触起始临界点 t* = {_qs(knot)} ({float(knot):.4f})",
                    )

            # 2b) LE_GT: t <= knot vs t > knot (contact offset boundary)
            le_parsed, gt_parsed = _split_le_gt(knot)
            if le_parsed and gt_parsed:
                b_le = _eval_parsed_branch(
                    branch_id=f"band_le:{_qs(knot)}",
                    label=f"分支 A · 确认 t ≤ {float(knot):.4f}",
                    condition_summary=f"局部点云复核确认该裂隙位移位于 t ≤ {_qs(knot)} ({float(knot):.4f}) 侧",
                    action_type="REFINE_PARAMETER_BAND",
                    sub_parsed=le_parsed,
                    filter_spec={"kind": "t_leq", "knot": _qs(knot)},
                )
                b_gt = _eval_parsed_branch(
                    branch_id=f"band_gt:{_qs(knot)}",
                    label=f"分支 B · 确认 t > {float(knot):.4f}",
                    condition_summary=f"局部点云复核确认该裂隙位移位于 t > {_qs(knot)} ({float(knot):.4f}) 侧",
                    action_type="REFINE_PARAMETER_BAND",
                    sub_parsed=gt_parsed,
                    filter_spec={"kind": "t_gt", "knot": _qs(knot)},
                )
                brs = [b_le, b_gt]
                sc = _score_branches(brs)
                if best_band_option is None or sc < best_band_option[0]:
                    best_band_option = (
                        sc,
                        brs,
                        f"局部点云复核该裂隙接触终止临界点 t* = {_qs(knot)} ({float(knot):.4f})",
                    )

        if best_band_option is not None:
            _, band_branches, band_desc = best_band_option
            action_options.append(("REFINE_PARAMETER_BAND", band_branches, band_desc))

        # Pick best action option for this fracture (prefer options that actually certify at least one branch!)
        best_action_type = "NO_DISTINGUISHING_ACTION"
        best_branches: List[Dict[str, Any]] = []
        best_action_desc = "该裂隙在当前合法域内不产生歧义分裂（收益为 0）"
        best_wc = total_measure
        best_res_frac = Fraction(0)
        best_cert_frac = Fraction(0)
        best_any_cert = 1

        for act_type, brs, act_desc in action_options:
            wc = max(
                (Fraction(0) if b["collapsed_to_certified"] else b["_meas_q"])
                for b in brs
            )
            res_m = sum((b["_meas_q"] for b in brs if b["collapsed_to_certified"]), Fraction(0))
            res_frac = (res_m / total_measure) if total_measure > 0 else Fraction(
                sum(b["cell_count"] for b in brs if b["collapsed_to_certified"]), total_cells
            )
            cert_frac = Fraction(sum(1 for b in brs if b["collapsed_to_certified"]), len(brs))
            any_cert = 0 if (res_frac > 0 or cert_frac > 0) else 1
            better = (
                (not best_branches)
                or (any_cert < best_any_cert)
                or (any_cert == best_any_cert and wc < best_wc)
                or (any_cert == best_any_cert and wc == best_wc and res_frac > best_res_frac)
                or (any_cert == best_any_cert and wc == best_wc and res_frac == best_res_frac and cert_frac > best_cert_frac)
            )
            if better:
                best_action_type = act_type
                best_branches = brs
                best_action_desc = act_desc
                best_wc = wc
                best_res_frac = res_frac
                best_cert_frac = cert_frac
                best_any_cert = any_cert

        # Witness divergence attribution
        in_path = fid in conn_path_frags
        in_cut_side = (fid in cut_s_frags) or (fid in cut_t_frags)
        distinguishes_modes = False
        if m_plus and m_minus and m_plus in msem and m_minus in msem:
            fp = (msem[m_plus].get("feature_assignments") or {}).get(fid)
            fm = (msem[m_minus].get("feature_assignments") or {}).get(fid)
            if fp is not None and fm is not None and fp != fm:
                distinguishes_modes = True

        if distinguishes_modes and in_path:
            wrole = "MODE_WITNESS_SPLITTER_AND_CONNECTED_PATH"
            reason = "既位于连通见证路径上，又是直接区分连通模式 m+ 与断割模式 m- 的核心分支裂隙。"
        elif in_path and in_cut_side:
            wrole = "IN_CONNECTED_PATH_AND_CUT_BOUNDARY"
            reason = "恰位于连通见证 (m+, t+) 贯通链上，且处于断割见证 (m-, t-) 的 S/T 隔离边界。"
        elif distinguishes_modes:
            wrole = "MODE_WITNESS_SPLITTER"
            reason = "核查该裂隙的离散归属/延展分支可直接区分 m+ 与 m- 所在模式子集。"
        elif in_path:
            wrole = "ON_CONNECTED_WITNESS_PATH"
            reason = "位于连通见证 (m+, t+) 的核心贯通裂隙链中。"
        elif best_res_frac > 0 or best_cert_frac > 0:
            wrole = "CRITICAL_CONTACT_BOUNDARY"
            reason = "其局部接触事件节点直接决定子域的连通性翻转。"
        else:
            wrole = "NON_DISCRIMINATING_FRACTURE"
            reason = "该裂隙在双见证间无区分力，复核后不消除当前多解歧义。"

        wc_ratio = (best_wc / total_measure) if total_measure > 0 else Fraction(1)
        candidates.append(
            {
                "fracture_id": fid,
                "short_label": _short_fracture_label(fid, role),
                "role": role,
                "action_type": best_action_type,
                "action_description": best_action_desc,
                "certified_branch_fraction": round(float(best_cert_frac), 4),
                "certified_branch_fraction_exact": _qs(best_cert_frac),
                "resolved_measure_fraction": round(float(best_res_frac), 4),
                "resolved_measure_fraction_exact": _qs(best_res_frac),
                "worst_case_remaining_ambiguity": round(float(wc_ratio), 4),
                "worst_case_remaining_measure_exact": _qs(best_wc),
                "witness_diff_role": wrole,
                "geometric_attribution": reason,
                "in_connected_path": in_path,
                "in_cut_S": fid in cut_s_frags,
                "in_cut_T": fid in cut_t_frags,
                "branches": [
                    {
                        "branch_id": b["branch_id"],
                        "label": b["label"],
                        "action_type": b["action_type"],
                        "condition_summary": b["condition_summary"],
                        "resulting_status": b["resulting_status"],
                        "collapsed_to_certified": b["collapsed_to_certified"],
                        "cell_count": b["cell_count"],
                        "measure_float": b["measure_float"],
                    }
                    for b in best_branches
                ],
                "_full_branches": best_branches,
            }
        )

    # Sort by: actually resolves ambiguity > Minimax remaining ambiguity ASC > resolved measure DESC > certified branch fraction DESC
    candidates.sort(
        key=lambda item: (
            0 if (_frac(item["resolved_measure_fraction_exact"]) > 0 or _frac(item["certified_branch_fraction_exact"]) > 0) else 1,
            _frac(item["worst_case_remaining_measure_exact"]),
            -_frac(item["resolved_measure_fraction_exact"]),
            -_frac(item["certified_branch_fraction_exact"]),
            0 if item["witness_diff_role"] != "NON_DISCRIMINATING_FRACTURE" else 1,
            item["fracture_id"],
        )
    )

    ranked_out: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates, start=1):
        full_branches = [_materialize_branch_3d(b) for b in item.pop("_full_branches")]
        item["rank"] = idx
        ranked_out.append(item)
        if full_branches:
            collapse_tree[item["fracture_id"]] = {
                "fracture_id": item["fracture_id"],
                "rank": idx,
                "action_type": item["action_type"],
                "branches": {b["branch_id"]: b for b in full_branches},
            }

    return {
        "status": "ACTIVE_CHECK_RECOMMENDED",
        "provenance_mode": "LIVE_EXACT_SOLVER",
        "input_sha256": input_sha256,
        "total_measure_exact": _qs(total_measure),
        "ranked_fractures": ranked_out,
        "collapse_tree": collapse_tree,
    }


class IdentifiabilityBridge:
    """Thread-safe read-only bridge serving /api/identifiability/* for STRATA."""

    def __init__(
        self,
        *,
        m1_root: Optional[Path | str] = None,
        m2_root: Optional[Path | str] = None,
        m3_root: Optional[Path | str] = None,
        repo_root: Optional[Path | str] = None,
        live_timeout_seconds: float = 60.0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else ROOT.resolve()
        self.m1_root = Path(
            m1_root or os.environ.get("FRACTUREFLOW_M1_ROOT") or DEFAULT_M1_ROOT
        ).resolve()
        self.m2_root = Path(
            m2_root or os.environ.get("FRACTUREFLOW_M2_ROOT") or DEFAULT_M2_ROOT
        ).resolve()
        self.m3_root = Path(
            m3_root or os.environ.get("FRACTUREFLOW_M3_ROOT") or DEFAULT_M3_ROOT
        ).resolve()
        self.live_timeout_seconds = float(live_timeout_seconds)

        self._lock = threading.Lock()
        self._live_cache: Dict[str, Dict[str, Any]] = {}  # keyed by model canonical sha256
        self._rnf_mod: Any = None
        self._cn_mod: Any = None
        self._recheck_mod: Any = None

    def _get_rnf(self) -> Any:
        if self._rnf_mod is not None:
            return self._rnf_mod
        # Prefer M1 validator from m1_root if present, else default M1 root
        candidates = [self.m1_root / "real_network_format.py", DEFAULT_M1_ROOT / "real_network_format.py"]
        target = next((p for p in candidates if p.is_file()), None)
        if target is None:
            raise IdentifiabilityError(
                "real_network_format.py not found in M1 directory",
                "M1_VALIDATOR_MISSING",
                503,
            )
        mod_name = "m1_real_network_format_bridge"
        mod = sys.modules.get(mod_name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(mod_name, target)
            if spec is None or spec.loader is None:
                raise IdentifiabilityError("failed to load real_network_format", "M1_VALIDATOR_LOAD_ERROR", 500)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        self._rnf_mod = mod
        return mod

    def _get_exact_solver(self) -> Tuple[Any, Any]:
        if self._cn_mod is not None and self._recheck_mod is not None:
            return self._cn_mod, self._recheck_mod
        tc_dir = str(TC_NETWORK_DIR)
        if tc_dir not in sys.path:
            sys.path.insert(0, tc_dir)
        import compile_network as cn_mod  # type: ignore[import-not-found]
        import recheck as recheck_mod  # type: ignore[import-not-found]

        self._cn_mod = cn_mod
        self._recheck_mod = recheck_mod
        return cn_mod, recheck_mod

    def _discover_m2_run_dir(self) -> Optional[Path]:
        runs_dir = self.m2_root / "runs"
        if not runs_dir.is_dir():
            return None
        candidates = [
            p for p in runs_dir.iterdir()
            if p.is_dir() and (p / "cases").is_dir()
        ]
        if not candidates:
            return None
        # Prefer runs that have summary.json, newest first
        candidates.sort(
            key=lambda p: (
                1 if (p / "summary.json").is_file() else 0,
                p.stat().st_mtime,
                p.name,
            ),
            reverse=True,
        )
        return candidates[0]

    def _load_m3_verdicts(self) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        vpath = self.m3_root / "verdict_batch.json"
        if not vpath.is_file():
            return {}, None
        try:
            raw = json.loads(vpath.read_text(encoding="utf-8"))
            batch_sha = _file_sha256(vpath)
        except Exception:
            return {}, None
        by_id: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            cases = raw.get("cases") or raw.get("verdicts") or raw.get("results") or []
            if isinstance(cases, dict):
                for cid, entry in cases.items():
                    if isinstance(entry, dict):
                        by_id[str(cid)] = entry
            elif isinstance(cases, list):
                for entry in cases:
                    if isinstance(entry, dict):
                        cid = entry.get("cluster_id") or entry.get("case_id") or entry.get("id")
                        if cid:
                            by_id[str(cid)] = entry
        return by_id, batch_sha

    def _load_register(self) -> Dict[str, Any]:
        reg_path = self.m1_root / "cluster_register.json"
        if not reg_path.is_file():
            raise IdentifiabilityError(
                f"M1 cluster_register.json missing at {reg_path}",
                "M1_REGISTER_MISSING",
                503,
            )
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IdentifiabilityError(f"invalid cluster_register.json: {exc}", "M1_REGISTER_INVALID", 409) from exc
        if not isinstance(reg, dict) or reg.get("scope") != REAL_SCOPE:
            raise IdentifiabilityError(
                f"cluster_register scope must be {REAL_SCOPE!r}, got {reg.get('scope')!r}",
                "SCOPE_VIOLATION",
                409,
            )
        clusters = reg.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise IdentifiabilityError("cluster_register has empty clusters list", "EMPTY_CLUSTER_REGISTER", 409)
        return reg

    def _scan_real_directory_scope_guard(self, registered_ids: Set[str]) -> None:
        """Hard-reject any synthetic impostor or unregistered rogue model dropped into clusters/ (M5-N3)."""
        clusters_dir = self.m1_root / "clusters"
        if not clusters_dir.is_dir():
            return
        for child in sorted(clusters_dir.iterdir()):
            if not child.is_dir():
                continue
            model_file = child / "model.json"
            if not model_file.is_file():
                continue
            try:
                raw = json.loads(model_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise IdentifiabilityError(
                    f"malformed model.json in real directory ({child.name}): {exc}",
                    "MODEL_JSON_INVALID",
                    409,
                    {"cluster_id": child.name},
                ) from exc
            scope = raw.get("scope") if isinstance(raw, dict) else None
            if scope == SYNTHETIC_SCOPE or scope != REAL_SCOPE:
                raise IdentifiabilityError(
                    f"REAL_SCOPE_VIOLATION in real clusters directory ({child.name}): "
                    f"scope={scope!r} is forbidden (required {REAL_SCOPE!r})",
                    "SCOPE_VIOLATION",
                    409,
                    {"cluster_id": child.name, "found_scope": scope},
                )
            if child.name not in registered_ids:
                raise IdentifiabilityError(
                    f"unregistered cluster {child.name!r} present in real clusters directory",
                    "UNREGISTERED_REAL_CLUSTER",
                    409,
                    {"cluster_id": child.name},
                )

    def _load_and_verify_cluster(
        self,
        cluster_id: str,
        reg: Optional[Dict[str, Any]] = None,
        m2_run_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load cluster model.json + optional M2 files, verifying scope & SHA256 fresh on disk."""
        if not cluster_id or any(ch in cluster_id for ch in ("/", "\\", "..")):
            raise IdentifiabilityError(f"invalid cluster_id: {cluster_id!r}", "INVALID_CLUSTER_ID", 400)
        register = reg if reg is not None else self._load_register()
        reg_by_id = {str(c["cluster_id"]): c for c in register["clusters"]}
        entry = reg_by_id.get(cluster_id)
        if entry is None:
            # Check if an unregistered impostor file sits on disk under that name
            rogue = self.m1_root / "clusters" / cluster_id / "model.json"
            if rogue.is_file():
                try:
                    raw_rogue = json.loads(rogue.read_text(encoding="utf-8"))
                    if isinstance(raw_rogue, dict) and raw_rogue.get("scope") != REAL_SCOPE:
                        raise IdentifiabilityError(
                            f"synthetic/non-real scope {raw_rogue.get('scope')!r} rejected for {cluster_id!r}",
                            "SCOPE_VIOLATION",
                            409,
                            {"cluster_id": cluster_id, "found_scope": raw_rogue.get("scope")},
                        )
                except IdentifiabilityError:
                    raise
                except Exception:
                    pass
                raise IdentifiabilityError(
                    f"unregistered cluster file found at {cluster_id!r}",
                    "UNREGISTERED_REAL_CLUSTER",
                    409,
                )
            raise IdentifiabilityError(
                f"cluster not found: {cluster_id!r}",
                "CLUSTER_NOT_FOUND",
                404,
                {"cluster_id": cluster_id},
            )

        model_path = (self.m1_root / entry["model_path"]).resolve()
        if self.m1_root not in model_path.parents or not model_path.is_file():
            raise IdentifiabilityError(
                f"model.json missing for {cluster_id!r}",
                "MODEL_FILE_MISSING",
                404,
            )

        actual_file_sha = _file_sha256(model_path)
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IdentifiabilityError(
                f"corrupted model.json for {cluster_id!r}: {exc}",
                "MODEL_JSON_INVALID",
                409,
            ) from exc

        # Hard scope check (M5-N3)
        if not isinstance(model, dict) or model.get("scope") != REAL_SCOPE:
            raise IdentifiabilityError(
                f"REAL_SCOPE_VIOLATION for {cluster_id!r}: expected {REAL_SCOPE!r}, got {model.get('scope') if isinstance(model, dict) else type(model).__name__!r}",
                "SCOPE_VIOLATION",
                409,
                {"cluster_id": cluster_id, "found_scope": model.get("scope") if isinstance(model, dict) else None},
            )

        # Full M1 validator check
        rnf = self._get_rnf()
        try:
            val_summary = rnf.validate(model, repo_root=self.repo_root, verify_source_sha=False)
        except Exception as exc:
            code = "SCOPE_VIOLATION" if type(exc).__name__ == "Outside" else "MODEL_VALIDATION_FAILED"
            raise IdentifiabilityError(
                f"M1 validation rejected {cluster_id!r}: {exc}",
                code,
                409,
                {"cluster_id": cluster_id, "validator_error": type(exc).__name__},
            ) from exc

        actual_canon_sha = str(val_summary["input_sha256"])

        # Register SHA256 integrity check (M5-N1)
        expected_canon_sha = entry.get("model_canonical_sha256")
        if expected_canon_sha and actual_canon_sha != expected_canon_sha:
            raise IdentifiabilityError(
                f"MODEL_SHA256_MISMATCH for {cluster_id!r}: register expects {expected_canon_sha}, disk is {actual_canon_sha}",
                "MODEL_SHA256_MISMATCH",
                409,
                {
                    "cluster_id": cluster_id,
                    "expected_canonical_sha256": expected_canon_sha,
                    "actual_canonical_sha256": actual_canon_sha,
                },
            )
        expected_file_sha = entry.get("model_file_sha256")
        if expected_file_sha and actual_file_sha != expected_file_sha:
            raise IdentifiabilityError(
                f"MODEL_FILE_SHA256_MISMATCH for {cluster_id!r}: register expects {expected_file_sha}, disk is {actual_file_sha}",
                "MODEL_FILE_SHA256_MISMATCH",
                409,
                {
                    "cluster_id": cluster_id,
                    "expected_file_sha256": expected_file_sha,
                    "actual_file_sha256": actual_file_sha,
                },
            )

        # Optional M2 certification.json & active_check.json integrity check (M5-N1)
        run_dir = m2_run_dir if m2_run_dir is not None else self._discover_m2_run_dir()
        m2_cert: Optional[Dict[str, Any]] = None
        m2_ac: Optional[Dict[str, Any]] = None
        if run_dir is not None:
            case_dir = run_dir / "cases" / cluster_id
            cert_path = case_dir / "certification.json"
            ac_path = case_dir / "active_check.json"
            if cert_path.is_file():
                try:
                    m2_cert = json.loads(cert_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise IdentifiabilityError(
                        f"corrupted M2 certification.json for {cluster_id!r}: {exc}",
                        "M2_CERT_INVALID",
                        409,
                    ) from exc
                cert_sha = m2_cert.get("input_sha256")
                if not cert_sha or cert_sha != actual_canon_sha:
                    raise IdentifiabilityError(
                        f"CERT_INPUT_SHA256_MISMATCH for {cluster_id!r}: certification.json expects {cert_sha!r}, model.json is {actual_canon_sha}",
                        "CERT_INPUT_SHA256_MISMATCH",
                        409,
                        {
                            "cluster_id": cluster_id,
                            "cert_input_sha256": cert_sha,
                            "actual_model_sha256": actual_canon_sha,
                        },
                    )
                m2_cert["_artifact_sha256"] = _file_sha256(cert_path)
                m2_cert["_run_id"] = run_dir.name
            if ac_path.is_file():
                try:
                    m2_ac = json.loads(ac_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise IdentifiabilityError(
                        f"corrupted M2 active_check.json for {cluster_id!r}: {exc}",
                        "M2_ACTIVE_CHECK_INVALID",
                        409,
                    ) from exc
                ac_sha = m2_ac.get("input_sha256")
                if ac_sha and ac_sha != actual_canon_sha:
                    raise IdentifiabilityError(
                        f"ACTIVE_CHECK_INPUT_SHA256_MISMATCH for {cluster_id!r}: active_check.json expects {ac_sha!r}, model.json is {actual_canon_sha}",
                        "ACTIVE_CHECK_INPUT_SHA256_MISMATCH",
                        409,
                    )
                m2_ac["_artifact_sha256"] = _file_sha256(ac_path)
                m2_ac["_run_id"] = run_dir.name

        return entry, model, actual_canon_sha, actual_file_sha, m2_cert, m2_ac

    def _run_live_exact_solver(self, model: Dict[str, Any], actual_canon_sha: str) -> Dict[str, Any]:
        with self._lock:
            cached = self._live_cache.get(actual_canon_sha)
        if cached is not None:
            return cached

        rnf = self._get_rnf()
        cn_mod, recheck_mod = self._get_exact_solver()
        core_payload = rnf.to_core_kernel_payload(model, repo_root=self.repo_root, verify_source_sha=False)

        holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                raw_res = cn_mod.compile_network(core_payload)
                if raw_res.get("search_complete"):
                    # Fast exact witness graph self-check on every returned cell (zero brute-force pair re-enumeration)
                    for cell in raw_res.get("cells") or []:
                        edge_set = {
                            tuple(sorted((str(e[0]), str(e[1]))))
                            for e in (cell.get("active_edges") or [])
                            if len(e) == 2
                        }
                        if cell.get("connected"):
                            path = cell.get("path") or []
                            if len(path) < 2 or path[0] != "@A" or path[-1] != "@B":
                                holder["error"] = IdentifiabilityError("invalid connected witness endpoints", "LIVE_RECHECK_FAILED", 500)
                                return
                            for u, v in zip(path, path[1:]):
                                if tuple(sorted((str(u), str(v)))) not in edge_set:
                                    holder["error"] = IdentifiabilityError("connected witness uses inactive edge", "LIVE_RECHECK_FAILED", 500)
                                    return
                        else:
                            cut = cell.get("cut") or {}
                            s_set = set(cut.get("S") or [])
                            t_set = set(cut.get("T") or [])
                            for u, v in edge_set:
                                if (u in s_set and v in t_set) or (v in s_set and u in t_set):
                                    holder["error"] = IdentifiabilityError("disconnected cut crossed by active edge", "LIVE_RECHECK_FAILED", 500)
                                    return
                holder["result"] = raw_res
            except Exception as exc:
                holder["error"] = exc

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=self.live_timeout_seconds)
        if th.is_alive():
            raise IdentifiabilityError(
                f"live exact solver exceeded timeout ({self.live_timeout_seconds:.1f}s)",
                "LIVE_SOLVER_TIMEOUT",
                504,
            )
        if "error" in holder:
            err = holder["error"]
            if isinstance(err, IdentifiabilityError):
                raise err
            raise IdentifiabilityError(f"live exact solver error: {err}", "LIVE_SOLVER_ERROR", 500)
        raw_res = holder["result"]
        if not raw_res.get("search_complete"):
            raise IdentifiabilityError(
                f"live exact solver incomplete ({raw_res.get('status')}): {raw_res.get('reason')}",
                "LIVE_SOLVER_INCOMPLETE",
                422,
            )

        status = _normalize_status(
            raw_res.get("status"),
            raw_res.get("exists_connected"),
            raw_res.get("all_connected"),
        )
        cells = list(raw_res.get("cells") or [])
        windows = list(raw_res.get("contact_windows") or [])
        rho_raw = raw_res.get("rho")
        rho = _compute_stability_from_cells(model, cells) if rho_raw is not None else None
        c_cell, d_cell = _pick_representative_cells(model, cells, rho)
        conn_wit = _enrich_connected_witness(model, c_cell) if c_cell is not None else None
        disc_wit = _enrich_disconnected_witness(model, d_cell) if d_cell is not None else None
        baseline = _compute_baseline_comparison(model, status, cells, windows, rho)
        active_check = _compute_exact_active_check(
            model, status, cells, windows, conn_wit, disc_wit, actual_canon_sha
        )

        bundle = {
            "provenance_mode": "LIVE_EXACT_SOLVER",
            "cert_source": "LIVE_EXACT_SOLVER",
            "input_sha256": actual_canon_sha,
            "local_recheck_verified": True,
            "status": status,
            "exists_connected": bool(raw_res.get("exists_connected")),
            "all_connected": bool(raw_res.get("all_connected")),
            "cell_count": len(cells),
            "pair_count": int((raw_res.get("enumeration") or {}).get("pairs", 0)),
            "cells": cells,
            "contact_windows": windows,
            "rho": rho,
            "connected_witness": conn_wit,
            "disconnected_witness": disc_wit,
            "baseline_comparison": baseline,
            "active_check": active_check,
            "solved_at_utc": _now_utc(),
        }
        with self._lock:
            self._live_cache[actual_canon_sha] = bundle
        return bundle

    def _adapt_m2_bundle(
        self,
        model: Dict[str, Any],
        actual_canon_sha: str,
        m2_cert: Dict[str, Any],
        m2_ac: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize M2 on-disk certification.json + active_check.json into STRATA contract."""
        cache_key = f"{actual_canon_sha}:{m2_cert.get('_artifact_sha256') or ''}:{m2_ac.get('_artifact_sha256') if m2_ac else ''}"
        with self._lock:
            cached = self._live_cache.get(cache_key)
        if cached is not None:
            return cached
        cells = list(m2_cert.get("cells") or [])
        windows = list(m2_cert.get("contact_windows") or [])
        status = _normalize_status(
            m2_cert.get("status"),
            m2_cert.get("exists_connected"),
            m2_cert.get("all_connected"),
        )
        exists_c = m2_cert.get("exists_connected")
        all_c = m2_cert.get("all_connected")
        if exists_c is None:
            exists_c = status in ("CERTIFIED_CONNECTED", "AMBIGUOUS_WITH_VERIFIED_WITNESSES")
        if all_c is None:
            all_c = status == "CERTIFIED_CONNECTED"

        # Normalize rho
        raw_rho = m2_cert.get("rho")
        rho: Optional[Dict[str, Any]] = None
        if isinstance(raw_rho, Mapping):
            inf_raw = raw_rho.get("infimum", raw_rho.get("rho_infimum"))
            fw = raw_rho.get("flip_witness") or m2_cert.get("nearest_flip_witness")
            rho = {
                "infimum": str(inf_raw) if inf_raw is not None else None,
                "infimum_float": round(float(_frac(inf_raw)), 6) if inf_raw is not None else None,
                "attained": raw_rho.get("attained"),
                "nominal_connected": raw_rho.get("nominal_connected"),
                "nominal_in_conditioned_domain": True,
                "flip_witness": fw,
            }
        elif raw_rho is not None:
            rho = {
                "infimum": str(raw_rho),
                "infimum_float": round(float(_frac(raw_rho)), 6),
                "attained": m2_cert.get("attained"),
                "nominal_connected": m2_cert.get("nominal_connected"),
                "nominal_in_conditioned_domain": True,
                "flip_witness": m2_cert.get("nearest_flip_witness"),
            }
        elif cells:
            rho = _compute_stability_from_cells(model, cells)

        # Witnesses: prefer explicit M2 witnesses, fall back to cell extraction
        raw_cw = m2_cert.get("connected_witness")
        raw_dw = m2_cert.get("disconnected_witness")
        c_cell, d_cell = _pick_representative_cells(model, cells, rho) if cells else (None, None)
        conn_wit = _enrich_connected_witness(model, raw_cw) if isinstance(raw_cw, Mapping) else (
            _enrich_connected_witness(model, c_cell) if c_cell is not None else None
        )
        disc_wit = _enrich_disconnected_witness(model, raw_dw) if isinstance(raw_dw, Mapping) else (
            _enrich_disconnected_witness(model, d_cell) if d_cell is not None else None
        )

        baseline = _compute_baseline_comparison(model, status, cells, windows, rho)

        # Adapt M2 active_check.json + collapse_tree if present
        if status in ("BUDGET_EXCEEDED", "REFUSED", "INFEASIBLE", "INFEASIBLE_OBSERVATIONS"):
            active_check = {
                "status": "NOT_APPLICABLE_REFUSED",
                "reason": m2_cert.get("reason") or (m2_ac.get("reason") if m2_ac else f"certification incomplete ({status})"),
                "provenance_mode": "PRECOMPUTED_VERIFIED_TREE" if m2_ac is not None else "M2_PRECOMPUTED_CERT",
                "input_sha256": actual_canon_sha,
                "m2_active_check_sha256": m2_ac.get("_artifact_sha256") if m2_ac else None,
                "ranked_fractures": [],
                "collapse_tree": {},
            }
        elif m2_ac is not None:
            active_check = self._adapt_m2_active_check(model, status, m2_ac, actual_canon_sha, cells)
        else:
            active_check = _compute_exact_active_check(
                model, status, cells, windows, conn_wit, disc_wit, actual_canon_sha
            )

        bundle = {
            "provenance_mode": "PRECOMPUTED_VERIFIED_TREE" if m2_ac is not None else "LIVE_EXACT_SOLVER",
            "cert_source": "M2_PRECOMPUTED_CERT",
            "m2_run_id": m2_cert.get("_run_id"),
            "m2_cert_sha256": m2_cert.get("_artifact_sha256"),
            "m2_active_check_sha256": m2_ac.get("_artifact_sha256") if m2_ac else None,
            "input_sha256": actual_canon_sha,
            "local_recheck_verified": bool((m2_cert.get("local_recheck") or {}).get("ok", True)),
            "status": status,
            "reason": m2_cert.get("reason"),
            "exists_connected": bool(exists_c),
            "all_connected": bool(all_c),
            "cell_count": len(cells) if cells else int((m2_cert.get("enumeration") or {}).get("cells", m2_cert.get("cell_count") or 0)),
            "pair_count": int((m2_cert.get("enumeration") or {}).get("pairs", m2_cert.get("pair_count") or 0)),
            "cells": cells,
            "contact_windows": windows,
            "rho": rho,
            "connected_witness": conn_wit,
            "disconnected_witness": disc_wit,
            "baseline_comparison": baseline,
            "active_check": active_check,
        }
        with self._lock:
            self._live_cache[cache_key] = bundle
        return bundle

    def _adapt_m2_active_check(
        self,
        model: Dict[str, Any],
        status: str,
        m2_ac: Dict[str, Any],
        actual_canon_sha: str,
        cells: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if status != "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
            return {
                "status": str(m2_ac.get("status") or "ALREADY_CERTIFIED"),
                "reason": m2_ac.get("reason"),
                "provenance_mode": "PRECOMPUTED_VERIFIED_TREE",
                "input_sha256": actual_canon_sha,
                "m2_active_check_sha256": m2_ac.get("_artifact_sha256"),
                "ranked_fractures": [],
                "collapse_tree": {},
            }

        fcat = model.get("fracture_catalog") or {}
        raw_ranked = m2_ac.get("ranked_fractures") or []
        raw_tree = m2_ac.get("collapse_tree") or {}
        norm_ranked: List[Dict[str, Any]] = []
        norm_tree: Dict[str, Dict[str, Any]] = {}

        def _fval(obj: Mapping[str, Any], float_key: str, rat_key: str, default: float = 0.0) -> float:
            if obj.get(float_key) is not None:
                try:
                    return float(obj[float_key])
                except Exception:
                    pass
            if obj.get(rat_key) is not None:
                try:
                    return float(_frac(obj[rat_key]))
                except Exception:
                    pass
            return default

        # Normalize collapse_tree entries matching M2 schema `collapse_tree.fractures[fid].branches`
        tree_by_fid: Dict[str, Any] = {}
        if isinstance(raw_tree, Mapping):
            if isinstance(raw_tree.get("fractures"), Mapping):
                for k, v in raw_tree["fractures"].items():
                    if isinstance(v, Mapping):
                        tree_by_fid[str(v.get("fracture_id") or k)] = v
            elif isinstance(raw_tree.get("branches"), Mapping) and raw_tree.get("fracture_id"):
                tree_by_fid[str(raw_tree["fracture_id"])] = raw_tree
            elif isinstance(raw_tree.get("nodes"), list):
                for node in raw_tree["nodes"]:
                    if isinstance(node, Mapping) and node.get("fracture_id"):
                        tree_by_fid[str(node["fracture_id"])] = node
            else:
                for k, v in raw_tree.items():
                    if isinstance(v, Mapping):
                        tree_by_fid[str(v.get("fracture_id") or k)] = v
        elif isinstance(raw_tree, list):
            for node in raw_tree:
                if isinstance(node, Mapping) and node.get("fracture_id"):
                    tree_by_fid[str(node["fracture_id"])] = node

        def _cond_summary(b: Mapping[str, Any]) -> str:
            if b.get("condition_summary"):
                return str(b["condition_summary"])
            if b.get("label_en"):
                return str(b["label_en"])
            cond = b.get("condition")
            if isinstance(cond, Mapping):
                parts = []
                if cond.get("feature_constraint"):
                    parts.append(str(cond["feature_constraint"]))
                if cond.get("t_predicate"):
                    parts.append(str(cond["t_predicate"]))
                if cond.get("allowed_modes"):
                    parts.append("模式: " + ",".join(str(m) for m in cond["allowed_modes"]))
                if parts:
                    return " · ".join(parts)
            return str(b.get("description") or b.get("branch_id") or "")

        def _norm_next_step(ns_raw: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(ns_raw, Mapping):
                return None
            ns_fid = str(ns_raw.get("recommended_fracture_id") or ns_raw.get("fracture_id") or "")
            ns_role = str((fcat.get(ns_fid) or {}).get("role") or "INTERIOR") if ns_fid else "INTERIOR"
            ns_act = str(ns_raw.get("action_type") or "ACTIVE_CHECK")
            sub_branches = _norm_branches(ns_raw.get("branches") or {}, ns_act, enrich_next=False)
            if not sub_branches:
                return None
            return {
                "fracture_id": ns_fid,
                "short_label": _short_fracture_label(ns_fid, ns_role) if ns_fid else "",
                "role": ns_role,
                "action_type": ns_act,
                "action_description": str(
                    ns_raw.get("action_summary_zh")
                    or ns_raw.get("action_summary_en")
                    or ns_raw.get("action_description")
                    or ns_act
                ),
                "certified_branch_fraction": round(
                    _fval(ns_raw, "certified_branch_fraction_float", "certified_branch_fraction", 0.0), 4
                ),
                "resolved_measure_fraction": round(
                    _fval(ns_raw, "resolved_measure_fraction_float", "resolved_measure_fraction", 0.0), 4
                ),
                "branches": sub_branches,
            }

        def _norm_branch_obj(bid: str, b: Mapping[str, Any], action_type: str, enrich_next: bool = True) -> Dict[str, Any]:
            res_status = _normalize_status(
                b.get("post_status") or b.get("resulting_status") or b.get("status") or (b.get("certification") or {}).get("status"),
                b.get("exists_connected"),
                b.get("all_connected"),
            )
            cert_sub = b.get("certification") if isinstance(b.get("certification"), Mapping) else b
            cw_raw = cert_sub.get("connected_witness")
            dw_raw = cert_sub.get("disconnected_witness")
            cw = _enrich_connected_witness(model, cw_raw) if isinstance(cw_raw, Mapping) else None
            dw = _enrich_disconnected_witness(model, dw_raw) if isinstance(dw_raw, Mapping) else None
            ns_obj = _norm_next_step(b.get("next_step")) if enrich_next else None
            return {
                "branch_id": str(bid),
                "label": str(b.get("label_zh") or b.get("label") or b.get("outcome_label") or bid),
                "label_en": str(b.get("label_en") or ""),
                "action_type": str(b.get("action_type") or action_type),
                "condition_summary": _cond_summary(b),
                "resulting_status": res_status,
                "collapsed_to_certified": res_status in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"),
                "cell_count": int(b.get("branch_cells_count") or b.get("cell_count") or len(b.get("cells") or []) or 0),
                "measure_float": _fval(b, "branch_measure_float", "branch_measure", 0.0),
                "rho": cert_sub.get("rho"),
                "connected_witness": cw,
                "disconnected_witness": dw,
                "next_step": ns_obj,
            }

        def _norm_branches(raw_src: Any, action_type: str, enrich_next: bool = True) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw_src, Mapping):
                for bk, bv in raw_src.items():
                    if isinstance(bv, Mapping):
                        out[str(bk)] = _norm_branch_obj(str(bk), bv, action_type, enrich_next=enrich_next)
            elif isinstance(raw_src, list):
                for bidx, bv in enumerate(raw_src):
                    if isinstance(bv, Mapping):
                        bk = str(bv.get("branch_id") or bv.get("outcome_id") or f"branch_{bidx + 1}")
                        out[bk] = _norm_branch_obj(bk, bv, action_type, enrich_next=enrich_next)
            return out

        for idx, item in enumerate(raw_ranked, start=1):
            if not isinstance(item, Mapping):
                continue
            fid = str(item.get("fracture_id") or item.get("component") or "")
            if not fid:
                continue
            role = str((fcat.get(fid) or {}).get("role") or item.get("catalog_role") or item.get("role") or "INTERIOR")
            action_type = str(item.get("best_action_type") or item.get("action_type") or "ACTIVE_CHECK")
            # Prefer full materialized branches from collapse_tree.fractures[fid].branches (which carry witnesses!)
            tree_node = tree_by_fid.get(fid) or {}
            raw_branches_src = (
                tree_node.get("branches")
                or tree_node.get("outcomes")
                or tree_node.get("children")
                or item.get("post_check_outcomes")
                or item.get("branches")
                or []
            )
            branch_dict = _norm_branches(raw_branches_src, action_type, enrich_next=True)

            mrf = _fval(item, "minimax_reduction_fraction_float", "minimax_reduction_fraction", -1.0)
            wc_ratio = round(max(0.0, 1.0 - mrf), 4) if mrf >= 0.0 else _fval(
                item, "worst_case_remaining_ambiguity", "worst_case_remaining_ambiguity_measure", 0.0
            )
            norm_ranked.append(
                {
                    "rank": int(item.get("rank") or idx),
                    "fracture_id": fid,
                    "short_label": _short_fracture_label(fid, role),
                    "role": role,
                    "action_type": action_type,
                    "action_description": str(item.get("best_action_summary_zh") or item.get("action_description") or item.get("description") or action_type),
                    "certified_branch_fraction": round(_fval(item, "certified_branch_fraction_float", "certified_branch_fraction", 0.0), 4),
                    "resolved_measure_fraction": round(_fval(item, "resolved_measure_fraction_float", "resolved_measure_fraction", 0.0), 4),
                    "worst_case_remaining_ambiguity": wc_ratio,
                    "witness_diff_role": str(item.get("witness_diff_role") or "CRITICAL_FRACTURE"),
                    "geometric_attribution": str(
                        item.get("geometric_attribution_zh")
                        or item.get("geometric_attribution")
                        or item.get("reason")
                        or item.get("witness_diff_role")
                        or ""
                    ),
                    "branches": [
                        {
                            "branch_id": b["branch_id"],
                            "label": b["label"],
                            "action_type": b["action_type"],
                            "condition_summary": b["condition_summary"],
                            "resulting_status": b["resulting_status"],
                            "collapsed_to_certified": b["collapsed_to_certified"],
                            "cell_count": b["cell_count"],
                            "measure_float": b["measure_float"],
                            "has_next_step": bool(b.get("next_step")),
                            "next_step_fracture_label": (b["next_step"].get("short_label") if isinstance(b.get("next_step"), Mapping) else None),
                        }
                        for b in branch_dict.values()
                    ],
                }
            )
            if branch_dict:
                norm_tree[fid] = {
                    "fracture_id": fid,
                    "rank": int(item.get("rank") or idx),
                    "action_type": action_type,
                    "branches": branch_dict,
                }

        return {
            "status": str(m2_ac.get("status") or "ACTIVE_CHECK_RECOMMENDED"),
            "provenance_mode": "PRECOMPUTED_VERIFIED_TREE",
            "input_sha256": actual_canon_sha,
            "m2_active_check_sha256": m2_ac.get("_artifact_sha256"),
            "ranked_fractures": norm_ranked,
            "collapse_tree": norm_tree,
        }

    # ------------------------------------------------------------------
    # Public HTTP Route Implementations
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        m2_run = self._discover_m2_run_dir()
        m3_verdicts, m3_sha = self._load_m3_verdicts()
        return {
            "ok": True,
            "m1_present": (self.m1_root / "cluster_register.json").is_file(),
            "m2_present": m2_run is not None,
            "m2_run_id": m2_run.name if m2_run else None,
            "m3_present": bool(m3_verdicts),
            "m3_verdict_count": len(m3_verdicts),
            "m3_verdict_sha256": m3_sha,
            "scope": REAL_SCOPE,
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
        }

    def list_cases(self) -> Dict[str, Any]:
        """GET /api/identifiability/cases — strictly validates every cluster SHA256 & scope."""
        reg = self._load_register()
        registered_ids = {str(c["cluster_id"]) for c in reg["clusters"]}
        # M5-N3 guard: reject synthetic impostor dropped into real clusters dir
        self._scan_real_directory_scope_guard(registered_ids)

        m2_run = self._discover_m2_run_dir()
        m3_verdicts, m3_sha = self._load_m3_verdicts()

        items: List[Dict[str, Any]] = []
        site_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}

        for reg_entry in reg["clusters"]:
            cid = str(reg_entry["cluster_id"])
            entry, model, canon_sha, file_sha, m2_cert, m2_ac = self._load_and_verify_cluster(
                cid, reg=reg, m2_run_dir=m2_run
            )
            site_id = str(entry["site_id"])
            site_counts[site_id] = site_counts.get(site_id, 0) + 1

            status = "AWAITING_M2_OR_LIVE_SOLVE"
            rho_val = None
            rho_float = None
            cert_source = "NOT_YET_SOLVED"
            if m2_cert is not None:
                status = _normalize_status(
                    m2_cert.get("status"),
                    m2_cert.get("exists_connected"),
                    m2_cert.get("all_connected"),
                )
                cert_source = "M2_PRECOMPUTED_CERT"
                raw_rho = m2_cert.get("rho")
                if isinstance(raw_rho, Mapping):
                    rho_val = raw_rho.get("infimum", raw_rho.get("rho_infimum"))
                elif raw_rho is not None:
                    rho_val = str(raw_rho)
                if rho_val is not None:
                    try:
                        rho_float = round(float(_frac(rho_val)), 6)
                    except Exception:
                        rho_float = None
            else:
                with self._lock:
                    cached = self._live_cache.get(canon_sha)
                if cached is not None:
                    status = cached["status"]
                    cert_source = "LIVE_EXACT_SOLVER"
                    if isinstance(cached.get("rho"), Mapping):
                        rho_val = cached["rho"].get("infimum")
                        rho_float = cached["rho"].get("infimum_float")

            status_counts[status] = status_counts.get(status, 0) + 1
            m3_entry = m3_verdicts.get(cid)
            m3_verdict = (
                str(m3_entry.get("verdict") or m3_entry.get("status") or "PENDING_M3")
                if isinstance(m3_entry, Mapping)
                else "PENDING_M3"
            )

            areg = model.get("assumption_register") or {}
            um = areg.get("uncertainty_model") or {}
            items.append(
                {
                    "cluster_id": cid,
                    "site_id": site_id,
                    "stratum": str(entry.get("stratum") or ""),
                    "scope": model["scope"],
                    "status": status,
                    "cert_source": cert_source,
                    "has_precomputed_active_check": m2_ac is not None,
                    "rho": rho_val,
                    "rho_float": rho_float,
                    "m3_judge_verdict": m3_verdict,
                    "input_sha256": canon_sha,
                    "model_file_sha256": file_sha,
                    "fracture_count": int(entry.get("fracture_count") or 0),
                    "mode_count": int(entry.get("mode_count") or 0),
                    "triangles_per_mode": int(entry.get("triangles_per_mode") or 0),
                    "preserved_hole_count": len(entry.get("preserved_hole_fragments") or []),
                    "quantization_sensitive": bool(entry.get("quantization_sensitive", False)),
                    "assumption_summary": {
                        "boundary_model": areg.get("boundary_model"),
                        "assumed_edges_tag": areg.get("assumed_edges_tag"),
                        "parameter_t_meaning": um.get("parameter_t_meaning"),
                        "not_claimed": list(areg.get("not_claimed") or []),
                    },
                }
            )

        return {
            "ok": True,
            "schema": "strata-identifiability-cases/v1",
            "at_utc": _now_utc(),
            "scope": REAL_SCOPE,
            "m2_run_id": m2_run.name if m2_run else None,
            "m3_verdict_sha256": m3_sha,
            "total_cases": len(items),
            "site_counts": site_counts,
            "status_counts": status_counts,
            "sha_verified_count": len(items),
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
            "not_claimed": list(reg.get("not_claimed") or []),
            "cases": items,
        }

    def get_case(self, cluster_id: str, *, allow_live: bool = True) -> Dict[str, Any]:
        """GET /api/identifiability/case?id=<cluster_id> — full 3D geometry + witnesses + active_check."""
        reg = self._load_register()
        registered_ids = {str(c["cluster_id"]) for c in reg["clusters"]}
        self._scan_real_directory_scope_guard(registered_ids)
        m2_run = self._discover_m2_run_dir()
        m3_verdicts, m3_sha = self._load_m3_verdicts()

        entry, model, canon_sha, file_sha, m2_cert, m2_ac = self._load_and_verify_cluster(
            cluster_id, reg=reg, m2_run_dir=m2_run
        )

        if m2_cert is not None:
            cert_bundle = self._adapt_m2_bundle(model, canon_sha, m2_cert, m2_ac)
        elif allow_live:
            cert_bundle = self._run_live_exact_solver(model, canon_sha)
        else:
            cert_bundle = {
                "provenance_mode": "AWAITING_M2_CERT",
                "cert_source": "NOT_YET_SOLVED",
                "input_sha256": canon_sha,
                "local_recheck_verified": False,
                "status": "AWAITING_M2_OR_LIVE_SOLVE",
                "exists_connected": None,
                "all_connected": None,
                "cell_count": 0,
                "pair_count": 0,
                "rho": None,
                "connected_witness": None,
                "disconnected_witness": None,
                "baseline_comparison": None,
                "active_check": {"status": "AWAITING_M2_CERT", "ranked_fractures": [], "collapse_tree": {}},
            }

        m3_entry = m3_verdicts.get(cluster_id)
        m3_verdict = (
            str(m3_entry.get("verdict") or m3_entry.get("status") or "PENDING_M3")
            if isinstance(m3_entry, Mapping)
            else "PENDING_M3"
        )

        # Build 3D geometry payload (float coordinates for canvas rendering, exact strings preserved)
        fcat = model.get("fracture_catalog") or {}
        msem = model.get("mode_semantics") or {}
        fractures_out: List[Dict[str, Any]] = []
        for fid in sorted(fcat.keys()):
            meta = fcat[fid]
            role = str(meta.get("role") or "INTERIOR")
            fractures_out.append(
                {
                    "id": fid,
                    "short_label": _short_fracture_label(fid, role),
                    "role": role,
                    "area_m2": meta.get("area_m2"),
                    "char_radius_m": meta.get("char_radius_m"),
                    "dip_deg": meta.get("dip_deg"),
                    "dip_direction_deg": meta.get("dip_direction_deg"),
                    "residual_sigma_m": meta.get("residual_sigma_m"),
                    "center_local_m": meta.get("center_local_m"),
                    "normal_nominal": meta.get("normal_nominal"),
                    "preserved_hole_count": int(meta.get("preserved_hole_count") or 0),
                    "original_hole_count": int(meta.get("original_hole_count") or 0),
                    "source_id": meta.get("source_id"),
                    "dxf_binding": meta.get("dxf_binding"),
                }
            )

        modes_geometry: Dict[str, Dict[str, Any]] = {}
        for m in model["modes"]:
            mid = str(m["id"])
            modes_geometry[mid] = {
                "id": mid,
                "edit_cost": str(m["edit_cost"]),
                "t_closed": [str(m["t_closed"][0]), str(m["t_closed"][1])],
                "segments": m["segments"],
                "semantics": msem.get(mid),
                "triangles": [
                    {
                        "id": str(tri["id"]),
                        "component": str(tri["component"]),
                        "label": str(tri.get("label") or ""),
                        "vertices": [_vec_float(p) for p in tri["vertices"]],
                        "v": _vec_float(tri["v"]),
                    }
                    for tri in m["triangles"]
                ],
            }

        hole_probes: List[Dict[str, Any]] = []
        for h in (model.get("quantization_contract") or {}).get("hole_preservation_audit") or []:
            hole_probes.append(
                {
                    "fragment_id": h["fragment_id"],
                    "preserved_hole_count": h["preserved_hole_count"],
                    "preserved_hole_area_m2": h["preserved_hole_area_m2"],
                    "probes": [_vec_float(p) for p in (h.get("hole_interior_probe_points_local") or [])],
                }
            )

        # Strip raw massive cell list from HTTP payload while keeping compact summary
        ac_public = {
            k: v for k, v in (cert_bundle.get("active_check") or {}).items()
            if k != "collapse_tree"
        }
        # Include compact collapse_tree preview (without duplicating huge raw cell arrays)
        ac_public["collapse_tree_summary"] = {
            fid: {
                "fracture_id": fid,
                "rank": node.get("rank"),
                "action_type": node.get("action_type"),
                "branch_ids": sorted((node.get("branches") or {}).keys()),
            }
            for fid, node in ((cert_bundle.get("active_check") or {}).get("collapse_tree") or {}).items()
        }

        return {
            "ok": True,
            "schema": "strata-identifiability-case/v1",
            "at_utc": _now_utc(),
            "cluster_id": cluster_id,
            "site_id": entry["site_id"],
            "stratum": entry.get("stratum"),
            "scope": model["scope"],
            "input_sha256": canon_sha,
            "model_file_sha256": file_sha,
            "provenance_mode": cert_bundle["provenance_mode"],
            "cert_source": cert_bundle["cert_source"],
            "m2_run_id": cert_bundle.get("m2_run_id"),
            "m2_cert_sha256": cert_bundle.get("m2_cert_sha256"),
            "m2_active_check_sha256": cert_bundle.get("m2_active_check_sha256"),
            "local_recheck_verified": bool(cert_bundle.get("local_recheck_verified", False)),
            "m3_judge_verdict": m3_verdict,
            "m3_verdict_sha256": m3_sha,
            "status": cert_bundle["status"],
            "reason": cert_bundle.get("reason"),
            "exists_connected": cert_bundle["exists_connected"],
            "all_connected": cert_bundle["all_connected"],
            "cell_count": cert_bundle["cell_count"],
            "pair_count": cert_bundle["pair_count"],
            "rho": cert_bundle["rho"],
            "connected_witness": cert_bundle["connected_witness"],
            "disconnected_witness": cert_bundle["disconnected_witness"],
            "baseline_comparison": cert_bundle["baseline_comparison"],
            "active_check": ac_public,
            "nominal_reference": model["reference"],
            "endpoints": {
                "A": _vec_float(model["endpoints"]["A"]),
                "B": _vec_float(model["endpoints"]["B"]),
            },
            "endpoints_exact": model["endpoints"],
            "fractures": fractures_out,
            "modes_geometry": modes_geometry,
            "hole_probes": hole_probes,
            "real_provenance": model["real_provenance"],
            "assumption_register": model["assumption_register"],
            "quantization_contract": model["quantization_contract"],
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
        }

    def apply_check(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """POST /api/identifiability/apply_check — consume verified collapse_tree or live exact re-solve."""
        cluster_id = str(body.get("cluster_id") or "").strip()
        fracture_id = str(body.get("fracture_id") or "").strip()
        branch_id = str(body.get("branch_id") or "").strip()
        if not cluster_id:
            raise IdentifiabilityError("cluster_id is required", "CLUSTER_ID_REQUIRED", 400)
        if not fracture_id:
            raise IdentifiabilityError("fracture_id is required", "FRACTURE_ID_REQUIRED", 400)
        if not branch_id:
            raise IdentifiabilityError("branch_id is required", "BRANCH_ID_REQUIRED", 400)

        reg = self._load_register()
        registered_ids = {str(c["cluster_id"]) for c in reg["clusters"]}
        self._scan_real_directory_scope_guard(registered_ids)
        m2_run = self._discover_m2_run_dir()

        _, model, canon_sha, _, m2_cert, m2_ac = self._load_and_verify_cluster(
            cluster_id, reg=reg, m2_run_dir=m2_run
        )

        # Validate fracture_id exists in model
        known_frags = set((model.get("fracture_catalog") or {}).keys())
        if not known_frags:
            known_frags = {str(tri["component"]) for m in model["modes"] for tri in m["triangles"]}
        if fracture_id not in known_frags:
            raise IdentifiabilityError(
                f"unknown fracture_id {fracture_id!r} for cluster {cluster_id!r}",
                "FRACTURE_NOT_FOUND",
                404,
                {"cluster_id": cluster_id, "fracture_id": fracture_id},
            )

        if m2_cert is not None:
            cert_bundle = self._adapt_m2_bundle(model, canon_sha, m2_cert, m2_ac)
        else:
            cert_bundle = self._run_live_exact_solver(model, canon_sha)

        ac = cert_bundle.get("active_check") or {}
        tree = ac.get("collapse_tree") or {}
        node = tree.get(fracture_id)
        if not isinstance(node, Mapping):
            raise IdentifiabilityError(
                f"fracture {fracture_id!r} has no declared active-check branches in cluster {cluster_id!r} (status={cert_bundle['status']})",
                "INVALID_CHECK_TARGET",
                400,
                {"cluster_id": cluster_id, "fracture_id": fracture_id, "status": cert_bundle["status"]},
            )
        branches = node.get("branches") or {}
        if branch_id not in branches:
            raise IdentifiabilityError(
                f"illegal branch_id {branch_id!r} for fracture {fracture_id!r}; allowed={sorted(branches.keys())}",
                "INVALID_BRANCH_ID",
                400,
                {
                    "cluster_id": cluster_id,
                    "fracture_id": fracture_id,
                    "branch_id": branch_id,
                    "allowed_branches": sorted(branches.keys()),
                },
            )

        step1_branch = branches[branch_id]
        step2_fracture_id = str(body.get("step2_fracture_id") or "").strip()
        step2_branch_id = str(body.get("step2_branch_id") or "").strip()
        provenance_mode = (
            "PRECOMPUTED_VERIFIED_TREE"
            if cert_bundle.get("provenance_mode") == "PRECOMPUTED_VERIFIED_TREE"
            else "LIVE_EXACT_SOLVER"
        )

        if step2_branch_id:
            ns = step1_branch.get("next_step")
            if not isinstance(ns, Mapping) or not isinstance(ns.get("branches"), Mapping):
                raise IdentifiabilityError(
                    f"step-1 branch {branch_id!r} on {fracture_id!r} has no next_step subtree",
                    "INVALID_STEP2_TARGET",
                    400,
                    {"cluster_id": cluster_id, "fracture_id": fracture_id, "branch_id": branch_id},
                )
            ns_fid = str(ns.get("fracture_id") or "")
            if step2_fracture_id and ns_fid and step2_fracture_id != ns_fid:
                raise IdentifiabilityError(
                    f"step2_fracture_id {step2_fracture_id!r} does not match expected {ns_fid!r}",
                    "INVALID_STEP2_FRACTURE",
                    400,
                    {"expected_fracture_id": ns_fid, "step2_fracture_id": step2_fracture_id},
                )
            eff_fid2 = step2_fracture_id or ns_fid or fracture_id
            branches2 = ns["branches"]
            if step2_branch_id not in branches2:
                raise IdentifiabilityError(
                    f"illegal step2_branch_id {step2_branch_id!r} for next_step {eff_fid2!r}; allowed={sorted(branches2.keys())}",
                    "INVALID_BRANCH_ID",
                    400,
                    {"cluster_id": cluster_id, "fracture_id": eff_fid2, "branch_id": step2_branch_id, "allowed_branches": sorted(branches2.keys())},
                )
            active_branch = branches2[step2_branch_id]
            prev_status = step1_branch["resulting_status"]
            eff_fid = eff_fid2
            eff_bid = step2_branch_id
            step_count = 2
            receipt_body = {
                "cluster_id": cluster_id,
                "input_sha256": canon_sha,
                "fracture_id": fracture_id,
                "branch_id": branch_id,
                "step2_fracture_id": eff_fid2,
                "step2_branch_id": step2_branch_id,
                "previous_status": prev_status,
                "resulting_status": active_branch["resulting_status"],
                "provenance_mode": provenance_mode,
            }
        else:
            active_branch = step1_branch
            prev_status = cert_bundle["status"]
            eff_fid = fracture_id
            eff_bid = branch_id
            step_count = 1
            receipt_body = {
                "cluster_id": cluster_id,
                "input_sha256": canon_sha,
                "fracture_id": fracture_id,
                "branch_id": branch_id,
                "previous_status": prev_status,
                "resulting_status": active_branch["resulting_status"],
                "provenance_mode": provenance_mode,
            }

        check_receipt_sha = _canonical_sha256(receipt_body)

        return {
            "ok": True,
            "schema": "strata-identifiability-apply-check/v1",
            "at_utc": _now_utc(),
            "cluster_id": cluster_id,
            "step_count": step_count,
            "root_status": cert_bundle["status"],
            "step1_fracture_id": fracture_id,
            "step1_branch_id": branch_id,
            "step1_branch_label": step1_branch.get("label"),
            "step2_fracture_id": (eff_fid if step_count == 2 else None),
            "step2_branch_id": step2_branch_id or None,
            "fracture_id": eff_fid,
            "branch_id": eff_bid,
            "branch_label": active_branch.get("label"),
            "action_type": active_branch.get("action_type"),
            "condition_summary": active_branch.get("condition_summary"),
            "previous_status": prev_status,
            "resulting_status": active_branch["resulting_status"],
            "collapsed_to_certified": bool(active_branch.get("collapsed_to_certified", False)),
            "provenance_mode": provenance_mode,
            "input_sha256": canon_sha,
            "m2_cert_sha256": cert_bundle.get("m2_cert_sha256"),
            "m2_active_check_sha256": cert_bundle.get("m2_active_check_sha256"),
            "check_receipt_sha256": check_receipt_sha,
            "cell_count": active_branch.get("cell_count"),
            "rho": active_branch.get("rho"),
            "connected_witness": active_branch.get("connected_witness"),
            "disconnected_witness": active_branch.get("disconnected_witness"),
            "next_step": active_branch.get("next_step") if step_count == 1 else None,
            "field_error_rate": None,
            "human_time": None,
            "hydraulic": None,
        }
