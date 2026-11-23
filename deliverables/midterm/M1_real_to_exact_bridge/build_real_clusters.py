"""M1 Real Outcrop Multi-Fracture Cluster Builder (Jardim Garcia + Perozzo 2024).

Mechanically executes the frozen pre-registered pure-geometry rules in `prereg.json`
after asserting `prereg.json` SHA256 matches `prereg_freeze_receipt.json` and all
source files match their registered SHA256 digests.

Zero connectivity computation is performed during selection or construction.
Outputs:
- `clusters/<cluster_id>/model.json` (schema = TC-network/real-conditioned/v1)
- `cluster_register.json`

Execution discipline (AGENTS.md §11): zero subprocess console windows, zero modification
to historical frozen directories or source data.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(HERE))

import real_network_format as rnf

GRID_DEN = 10000
GRID_STEP = Q(1, GRID_DEN)
HALF_STEP = Q(1, 2 * GRID_DEN)
COS_TILT = Q(420, 421)  # exact Pythagorean: 420^2 + 29^2 = 421^2 (~3.95 deg)
SIN_TILT = Q(29, 421)
OCT_DIAG = Q(7071, 10000)

JARDIM_PROJECT_REL = (
    "results/w160_full_pointcloud_roi/work/w160_uh_roi_patch_20261113_141632/exports/"
    "proj_pf02_w160_uh_roi_patch_16817b5928_v000005_caa65f8da7d9_0c1bedc9/project.json"
)
JARDIM_PLY_REL = "results/w160_full_pointcloud_roi/roi/uh_roi_patch/points.ply"
JARDIM_OBS_REL = "results/w160_full_pointcloud_roi/roi/uh_roi_patch/observation.json"
JARDIM_IDX_REL = "results/w160_full_pointcloud_roi/roi/uh_roi_patch/source_index.bin"
JARDIM_MAN_REL = "results/w160_full_pointcloud_roi/roi_manifest.json"
JARDIM_REG_REL = "results/w180_technical_route_20261117/execution_next/dispatch_20261118/TM_real_product/real_case_register.json"

PEROZZO_XLSX_REL = "data/external/perozzo_2024/Manual-Discontinuities-Geometrical-Info.xlsx"
PEROZZO_DXF_REL = "data/external/perozzo_2024/Mapped_Discontinuity_Traces-Planes.dxf"
PEROZZO_LAS_REL = "data/external/perozzo_2024/Pointcloud_Full-Resolution.las"
PEROZZO_MAN_REL = "results/w180_technical_route_20261117/execution_next/dispatch_20261118/TH_data/download_manifest.json"
PEROZZO_REG_REL = "results/w180_technical_route_20261117/execution_next/dispatch_20261118/TH_data/registration_check.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def snap_q(val: float, den: int = GRID_DEN) -> Q:
    return Q(int(round(float(val) * den)), den)


def snap_vec3(v: Sequence[float], den: int = GRID_DEN) -> Tuple[Q, Q, Q]:
    return (snap_q(v[0], den), snap_q(v[1], den), snap_q(v[2], den))


def qs_vec3(v: Tuple[Q, Q, Q]) -> List[str]:
    return [rnf.qs(v[0]), rnf.qs(v[1]), rnf.qs(v[2])]


def canonical_basis_np(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Identical deterministic orthonormal basis to fractureflow.surface_product.support_geometry."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / float(np.linalg.norm(n))
    axes = np.eye(3, dtype=np.float64)
    idx = int(np.argmin(np.abs(n)))
    u = np.cross(axes[idx], n)
    if float(np.linalg.norm(u)) <= 1e-12:
        u = np.cross(axes[(idx + 1) % 3], n)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def normal_to_dip_dipdir(n: np.ndarray) -> Tuple[float, float]:
    """Convert unit normal (pointing upward nz>=0) to geological dip and dip direction (deg)."""
    nv = np.asarray(n, dtype=np.float64)
    nv = nv / float(np.linalg.norm(nv))
    if nv[2] < 0.0:
        nv = -nv
    dip = math.degrees(math.acos(max(-1.0, min(1.0, float(nv[2])))))
    dd = math.degrees(math.atan2(float(nv[0]), float(nv[1]))) % 360.0
    return round(dip, 4), round(dd, 4)


# ---------------------------------------------------------------------------
# Exact Rational 2D Geometry & Hole-Preserving Triangulation
# ---------------------------------------------------------------------------

def cross2(o: Tuple[Q, Q], a: Tuple[Q, Q], b: Tuple[Q, Q]) -> Q:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def signed_area2(poly: Sequence[Tuple[Q, Q]]) -> Q:
    s = Q(0)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2


def _on_segment(a: Tuple[Q, Q], b: Tuple[Q, Q], p: Tuple[Q, Q]) -> bool:
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        and cross2(a, b, p) == 0
    )


def segments_intersect(
    p1: Tuple[Q, Q], p2: Tuple[Q, Q], p3: Tuple[Q, Q], p4: Tuple[Q, Q], *, allow_shared_endpoint: bool = True
) -> bool:
    shared = {p1, p2} & {p3, p4}
    if shared:
        if not allow_shared_endpoint:
            return True
        if len({p1, p2, p3, p4}) <= 2:
            return True
        # Check collinear overlap beyond the shared endpoint
        if cross2(p1, p2, p3) == 0 and cross2(p1, p2, p4) == 0:
            other1 = p2 if p1 in shared else p1
            other2 = p4 if p3 in shared else p3
            sh = next(iter(shared))
            dot_dir = (other1[0] - sh[0]) * (other2[0] - sh[0]) + (other1[1] - sh[1]) * (other2[1] - sh[1])
            return dot_dir > 0
        return False
    d1 = cross2(p3, p4, p1)
    d2 = cross2(p3, p4, p2)
    d3 = cross2(p1, p2, p3)
    d4 = cross2(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def is_simple_ring(poly: Sequence[Tuple[Q, Q]]) -> bool:
    n = len(poly)
    if n < 3 or len(set(poly)) != n:
        return False
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        if a1 == a2:
            return False
        for j in range(i + 1, n):
            b1, b2 = poly[j], poly[(j + 1) % n]
            if j == (i + 1) % n or i == (j + 1) % n:
                if segments_intersect(a1, a2, b1, b2, allow_shared_endpoint=True):
                    return False
                continue
            if segments_intersect(a1, a2, b1, b2, allow_shared_endpoint=False):
                return False
    return True


def point_strictly_inside_ring(pt: Tuple[Q, Q], ring: Sequence[Tuple[Q, Q]]) -> bool:
    """Exact rational ray-casting point-in-polygon; returns False on boundary."""
    x, y = pt
    n = len(ring)
    inside = False
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if _on_segment(a, b, pt):
            return False
        ax, ay = a
        bx, by = b
        if (ay > y) != (by > y):
            x_int = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_int:
                inside = not inside
    return inside


def point_in_tri_closed_2d(p: Tuple[Q, Q], a: Tuple[Q, Q], b: Tuple[Q, Q], c: Tuple[Q, Q]) -> bool:
    d1 = cross2(a, b, p)
    d2 = cross2(b, c, p)
    d3 = cross2(c, a, p)
    return d1 >= 0 and d2 >= 0 and d3 >= 0


def ear_clip_exact(poly: Sequence[Tuple[Q, Q]]) -> List[Tuple[int, int, int]]:
    """Exact rational ear-clipping triangulation of a simple CCW 2D polygon."""
    n = len(poly)
    area = signed_area2(poly)
    if area <= 0:
        raise ValueError("ear_clip_exact requires CCW positive-area simple polygon")
    # Remove strictly collinear vertices if any would prevent ear finding
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10000:
            raise ValueError("ear_clip_exact guard exceeded")
        m = len(idx)
        clipped = False
        for k in range(m):
            ip, ic, inx = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = poly[ip], poly[ic], poly[inx]
            if cross2(a, b, c) <= 0:
                continue
            ok = True
            for j in idx:
                if j in (ip, ic, inx):
                    continue
                if point_in_tri_closed_2d(poly[j], a, b, c):
                    ok = False
                    break
            if not ok:
                continue
            tris.append((ip, ic, inx))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            raise ValueError("ear_clip_exact found no valid ear")
    a, b, c = poly[idx[0]], poly[idx[1]], poly[idx[2]]
    if cross2(a, b, c) <= 0:
        raise ValueError("final ear triangle is degenerate")
    tris.append((idx[0], idx[1], idx[2]))
    tri_area_sum = sum((cross2(poly[i], poly[j], poly[k]) / 2 for i, j, k in tris), Q(0))
    if tri_area_sum != area:
        raise ValueError(f"ear_clip_exact area conservation failed: {tri_area_sum} != {area}")
    return tris


def triangulate_polygon_with_one_hole_exact(
    ext_ccw: Sequence[Tuple[Q, Q]],
    hole_ccw: Sequence[Tuple[Q, Q]],
) -> List[Tuple[Tuple[Q, Q], Tuple[Q, Q], Tuple[Q, Q]]]:
    """Exact rational triangulation of a simple CCW exterior polygon with one simple
    strictly-interior hole (given as CCW ring), via two disjoint valid diagonals."""
    assert is_simple_ring(ext_ccw) and signed_area2(ext_ccw) > 0
    assert is_simple_ring(hole_ccw) and signed_area2(hole_ccw) > 0
    for hv in hole_ccw:
        assert point_strictly_inside_ring(hv, ext_ccw), "hole vertex not strictly inside exterior"

    ne, nh = len(ext_ccw), len(hole_ccw)
    ext_edges = [(ext_ccw[i], ext_ccw[(i + 1) % ne]) for i in range(ne)]
    hole_edges = [(hole_ccw[j], hole_ccw[(j + 1) % nh]) for j in range(nh)]

    valid_diags: List[Tuple[Q, int, int]] = []
    for i in range(ne):
        pe = ext_ccw[i]
        for j in range(nh):
            ph = hole_ccw[j]
            mid = ((pe[0] + ph[0]) / 2, (pe[1] + ph[1]) / 2)
            if not point_strictly_inside_ring(mid, ext_ccw):
                continue
            if point_strictly_inside_ring(mid, hole_ccw):
                continue
            blocked = False
            for a, b in ext_edges:
                if segments_intersect(pe, ph, a, b, allow_shared_endpoint=True):
                    blocked = True
                    break
            if blocked:
                continue
            for a, b in hole_edges:
                if segments_intersect(pe, ph, a, b, allow_shared_endpoint=True):
                    blocked = True
                    break
            if not blocked:
                d2 = (pe[0] - ph[0]) ** 2 + (pe[1] - ph[1]) ** 2
                valid_diags.append((d2, i, j))

    valid_diags.sort()
    chosen = None
    for idx_a in range(len(valid_diags)):
        d2a, i1, j1 = valid_diags[idx_a]
        for idx_b in range(idx_a + 1, len(valid_diags)):
            d2b, i2, j2 = valid_diags[idx_b]
            if i1 == i2 or j1 == j2:
                continue
            if segments_intersect(ext_ccw[i1], hole_ccw[j1], ext_ccw[i2], hole_ccw[j2], allow_shared_endpoint=False):
                continue
            # Build P1 and P2 and verify both are strictly simple & positive area
            def walk_fwd(n_ring: int, start: int, end: int) -> List[int]:
                out = [start]
                cur = start
                while cur != end:
                    cur = (cur + 1) % n_ring
                    out.append(cur)
                return out

            def walk_bwd(n_ring: int, start: int, end: int) -> List[int]:
                out = [start]
                cur = start
                while cur != end:
                    cur = (cur - 1) % n_ring
                    out.append(cur)
                return out

            p1 = [ext_ccw[k] for k in walk_fwd(ne, i1, i2)] + [hole_ccw[k] for k in walk_bwd(nh, j2, j1)]
            p2 = [ext_ccw[k] for k in walk_fwd(ne, i2, i1)] + [hole_ccw[k] for k in walk_bwd(nh, j1, j2)]
            if is_simple_ring(p1) and is_simple_ring(p2):
                a1, a2 = signed_area2(p1), signed_area2(p2)
                if a1 > 0 and a2 > 0 and (a1 + a2) == (signed_area2(ext_ccw) - signed_area2(hole_ccw)):
                    chosen = (p1, p2)
                    break
        if chosen is not None:
            break

    if chosen is None:
        raise ValueError("failed to find valid two-diagonal hole decomposition")

    p1, p2 = chosen
    tris: List[Tuple[Tuple[Q, Q], Tuple[Q, Q], Tuple[Q, Q]]] = []
    for sub_poly in (p1, p2):
        for ia, ib, ic in ear_clip_exact(sub_poly):
            tris.append((sub_poly[ia], sub_poly[ib], sub_poly[ic]))

    total_area = sum((cross2(a, b, c) / 2 for a, b, c in tris), Q(0))
    expected_area = signed_area2(ext_ccw) - signed_area2(hole_ccw)
    assert total_area == expected_area, f"hole triangulation area mismatch: {total_area} != {expected_area}"
    return tris


def _clean_ring_2d(coords: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for xy in coords:
        p = (float(xy[0]), float(xy[1]))
        if not pts or abs(p[0] - pts[-1][0]) > 1e-12 or abs(p[1] - pts[-1][1]) > 1e-12:
            pts.append(p)
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) <= 1e-12 and abs(pts[0][1] - pts[-1][1]) <= 1e-12:
        pts.pop()
    # Ensure CCW
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    if area < 0.0:
        pts.reverse()
    return pts


def _simplify_ring_vw(
    ring: List[Tuple[float, float]],
    target_max_verts: int,
    *,
    must_contain: Optional[Polygon] = None,
    must_be_inside: Optional[Polygon] = None,
    min_verts: int = 4,
) -> List[Tuple[float, float]]:
    """Visvalingam-Whyatt iterative vertex removal preserving polygon simplicity and hole topology.

    Sorts vertices by effective triangle area each step and breaks on the first valid removal
    (O(1) checks per step instead of O(n^3) exact-Fraction scans); exact Fraction simplicity
    is enforced on snapped coordinates whenever n <= target_max_verts + 6.
    """
    pts = list(ring)
    must_contain_buf = must_contain.buffer(0.005) if must_contain is not None else None
    while len(pts) > target_max_verts and len(pts) > min_verts:
        n = len(pts)
        order = []
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            tri_area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            order.append((tri_area, i))
        order.sort()
        removed = False
        check_exact = n <= target_max_verts + 6
        for _, i in order:
            cand = pts[:i] + pts[i + 1 :]
            if len(cand) < min_verts:
                continue
            poly = Polygon(cand)
            if not poly.is_valid or not poly.is_simple or poly.area <= 1e-6:
                continue
            if must_contain_buf is not None and not poly.contains(must_contain_buf):
                continue
            if must_be_inside is not None and not must_be_inside.contains(poly.buffer(0.005)):
                continue
            if check_exact:
                cand_q = [(snap_q(p[0]), snap_q(p[1])) for p in cand]
                if not is_simple_ring(cand_q) or signed_area2(cand_q) <= 0:
                    continue
            pts = cand
            removed = True
            break
        if not removed:
            break
    return pts


def simplify_and_triangulate_jardim_region(
    exterior_raw: Sequence[Sequence[float]],
    holes_raw: Sequence[Sequence[Sequence[float]]],
    *,
    max_ext_verts: int,
    max_hole_verts: int = 6,
    hole_min_area: float = 0.01,
) -> Dict[str, Any]:
    """Simplify primary Jardim region (preserving concavity & largest hole if present)
    and triangulate in exact rational arithmetic on Tier-2 1e-4 m grid."""
    ext_clean = _clean_ring_2d(exterior_raw)
    orig_holes_clean = [_clean_ring_2d(h) for h in holes_raw if len(h) >= 4]
    orig_poly_full = Polygon(ext_clean, orig_holes_clean)
    orig_hull_area = float(orig_poly_full.convex_hull.area)
    orig_net_area = float(orig_poly_full.area)

    # Pick largest real hole >= hole_min_area (if any)
    kept_hole_raw = None
    if orig_holes_clean:
        by_area = sorted(orig_holes_clean, key=lambda h: Polygon(h).area, reverse=True)
        if Polygon(by_area[0]).area >= hole_min_area:
            kept_hole_raw = by_area[0]

    if kept_hole_raw is None:
        ext_simp = _simplify_ring_vw(ext_clean, max_ext_verts, min_verts=4)
        ext_q = [(snap_q(p[0]), snap_q(p[1])) for p in ext_simp]
        assert is_simple_ring(ext_q) and signed_area2(ext_q) > 0
        tri_idx = ear_clip_exact(ext_q)
        tris_2d = [(ext_q[i], ext_q[j], ext_q[k]) for i, j, k in tri_idx]
        simp_poly = Polygon([(float(x), float(y)) for x, y in ext_q])
        orig_compare = Polygon(ext_clean)
        hausdorff = float(orig_compare.hausdorff_distance(simp_poly))
        hole_q = None
        hole_probes_2d: List[Tuple[Q, Q]] = []
        preserved_hole_area = 0.0
    else:
        ext_poly_orig = Polygon(ext_clean)
        # First simplify hole while keeping it inside original exterior
        hole_simp = _simplify_ring_vw(
            kept_hole_raw,
            max_hole_verts,
            must_be_inside=ext_poly_orig,
            min_verts=4,
        )
        hole_poly_simp = Polygon(hole_simp)
        ext_simp = _simplify_ring_vw(
            ext_clean,
            max_ext_verts,
            must_contain=hole_poly_simp,
            min_verts=5,
        )
        ext_q = [(snap_q(p[0]), snap_q(p[1])) for p in ext_simp]
        hole_q = [(snap_q(p[0]), snap_q(p[1])) for p in hole_simp]
        # Verify exact rational strict containment & non-intersection
        assert is_simple_ring(ext_q) and signed_area2(ext_q) > 0
        assert is_simple_ring(hole_q) and signed_area2(hole_q) > 0
        tris_2d = triangulate_polygon_with_one_hole_exact(ext_q, hole_q)
        # Compute exact interior probe points strictly inside hole_q via ear-clipping the hole itself!
        hole_tris = ear_clip_exact(hole_q)
        hole_tris_sorted = sorted(
            hole_tris,
            key=lambda t: abs(cross2(hole_q[t[0]], hole_q[t[1]], hole_q[t[2]])),
            reverse=True,
        )
        hole_probes_2d = []
        for ia, ib, ic in hole_tris_sorted[:2]:
            pa, pb, pc = hole_q[ia], hole_q[ib], hole_q[ic]
            probe = ((pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3)
            assert point_strictly_inside_ring(probe, hole_q), "probe not strictly inside hole"
            hole_probes_2d.append(probe)
        simp_poly = Polygon(
            [(float(x), float(y)) for x, y in ext_q],
            [[(float(x), float(y)) for x, y in hole_q]],
        )
        orig_compare = Polygon(ext_clean, [kept_hole_raw])
        hausdorff = float(orig_compare.hausdorff_distance(simp_poly))
        preserved_hole_area = float(signed_area2(hole_q))

    # Compute exact 2D pivot for scaling: hole probe[0] if holed (keeps probe inside hole for all alpha>0!), else centroid
    if hole_probes_2d:
        pivot_2d = hole_probes_2d[0]
    else:
        c2 = simp_poly.centroid
        pivot_2d = (snap_q(c2.x), snap_q(c2.y))

    tri_area_2d = float(sum((cross2(a, b, c) / 2 for a, b, c in tris_2d), Q(0)))
    return {
        "ext_q": ext_q,
        "hole_q": hole_q,
        "tris_2d": tris_2d,
        "pivot_2d": pivot_2d,
        "hole_probes_2d": hole_probes_2d,
        "original_hole_count": len(orig_holes_clean),
        "preserved_hole_count": 1 if hole_q is not None else 0,
        "original_net_area_m2": orig_net_area,
        "original_convex_hull_area_m2": orig_hull_area,
        "preserved_hole_area_m2": preserved_hole_area,
        "triangulated_area_2d_m2": tri_area_2d,
        "hausdorff_2d_m": hausdorff,
    }


# ---------------------------------------------------------------------------
# 3D Distance Helpers (Purely Descriptive for Quantization Sensitivity Audit)
# ---------------------------------------------------------------------------

def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _seg_seg_dist(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b
    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D
    if D < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    diff = w + sc * u - tc * v
    return float(np.linalg.norm(diff))


def _point_tri_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(p - a))
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(p - b))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + v * ab)))
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(p - c))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return float(np.linalg.norm(p - (a + ab * v + ac * w)))


def _tri_tri_dist(t1: Tuple[np.ndarray, np.ndarray, np.ndarray], t2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    # Check intersection via Moller-like segment-triangle test first
    for tri_a, tri_b in ((t1, t2), (t2, t1)):
        a, b, c = tri_b
        n = np.cross(b - a, c - a)
        nn = float(np.dot(n, n))
        if nn <= 1e-18:
            continue
        for i in range(3):
            p0, p1 = tri_a[i], tri_a[(i + 1) % 3]
            d0 = float(np.dot(p0 - a, n))
            d1 = float(np.dot(p1 - a, n))
            if (d0 <= 0.0 <= d1) or (d1 <= 0.0 <= d0):
                denom = d0 - d1
                if abs(denom) > 1e-15:
                    s = d0 / denom
                    x = p0 + s * (p1 - p0)
                    # Barycentric check
                    c0 = float(np.dot(np.cross(b - a, x - a), n))
                    c1 = float(np.dot(np.cross(c - b, x - b), n))
                    c2 = float(np.dot(np.cross(a - c, x - c), n))
                    if c0 >= -1e-12 * nn and c1 >= -1e-12 * nn and c2 >= -1e-12 * nn:
                        return 0.0
    best = float("inf")
    for p in t1:
        best = min(best, _point_tri_dist(p, t2[0], t2[1], t2[2]))
    for p in t2:
        best = min(best, _point_tri_dist(p, t1[0], t1[1], t1[2]))
    for i in range(3):
        a1, a2 = t1[i], t1[(i + 1) % 3]
        for j in range(3):
            b1, b2 = t2[j], t2[(j + 1) % 3]
            best = min(best, _seg_seg_dist(a1, a2, b1, b2))
    return best


def compute_nominal_pair_gap_summary(m0_triangles: List[Dict[str, Any]]) -> Tuple[Optional[float], int]:
    """Compute minimum positive nominal fracture-pair 3D distance and touching pair count."""
    by_comp: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for tri in m0_triangles:
        pts = tuple(np.array([float(rnf.exact(c)) for c in v], dtype=np.float64) for v in tri["vertices"])
        by_comp.setdefault(tri["component"], []).append(pts)  # type: ignore[arg-type]
    comps = sorted(by_comp)
    touching = 0
    min_pos: Optional[float] = None
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            d_ij = float("inf")
            for ta in by_comp[comps[i]]:
                for tb in by_comp[comps[j]]:
                    d_ij = min(d_ij, _tri_tri_dist(ta, tb))
                    if d_ij == 0.0:
                        break
                if d_ij == 0.0:
                    break
            if d_ij <= 1e-10:
                touching += 1
            else:
                min_pos = d_ij if min_pos is None else min(min_pos, d_ij)
    return min_pos, touching


# ---------------------------------------------------------------------------
# Source Loaders (Read-Only, Exact Text Parsing for Perozzo)
# ---------------------------------------------------------------------------

def load_perozzo_xlsx(path: Path) -> List[Dict[str, Any]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
        sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        root = ET.fromstring(z.read(sheets[0]))
        raw_rows: List[Tuple[int, Dict[str, Optional[str]]]] = []
        for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            cells: Dict[str, Optional[str]] = {}
            for c in row.findall("m:c", ns):
                ref = c.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                v = c.find("m:v", ns)
                val = None if v is None else v.text
                if c.get("t") == "s" and val is not None:
                    val = shared[int(val)]
                cells[col] = val
            raw_rows.append((int(row.get("r", "0")), cells))

    hdr = raw_rows[0][1]
    col_by_name = {v: k for k, v in hdr.items() if v is not None}
    planes: List[Dict[str, Any]] = []
    for row_num, cells in raw_rows[1:]:
        def g(name: str) -> str:
            val = cells[col_by_name[name]]
            assert val is not None
            return val.strip()

        cx_raw, cy_raw, cz_raw = g("Xcenter"), g("Ycenter"), g("Zcenter")
        nx_raw, ny_raw, nz_raw = g("Nx"), g("Ny"), g("Nz")
        r_raw, mae_raw, std_raw = g("Radius"), g("MeanAbsError"), g("StDevAbsError")
        n_np = np.array([float(nx_raw), float(ny_raw), float(nz_raw)], dtype=np.float64)
        n_norm = float(np.linalg.norm(n_np))
        planes.append({
            "row": row_num,
            "fracture_id": f"perozzo_row_{row_num:04d}",
            "dip_deg": float(g("Dip")),
            "dip_direction_deg": float(g("DipDirection")),
            "radius": float(r_raw),
            "radius_raw": r_raw,
            "center_raw": [cx_raw, cy_raw, cz_raw],
            "center_q": (Q(cx_raw), Q(cy_raw), Q(cz_raw)),
            "center_np": np.array([float(cx_raw), float(cy_raw), float(cz_raw)], dtype=np.float64),
            "normal_raw": [nx_raw, ny_raw, nz_raw],
            "normal_q": (Q(nx_raw), Q(ny_raw), Q(nz_raw)),
            "normal_np": n_np / n_norm if n_norm > 0 else n_np,
            "normal_norm": n_norm,
            "mae": float(mae_raw),
            "mae_raw": mae_raw,
            "std": float(std_raw),
            "std_raw": std_raw,
            "sigma": float(mae_raw) + float(std_raw),
        })
    return planes


def load_perozzo_dxf(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    polys: List[Dict[str, Any]] = []
    i = 0
    in_entities = False
    while i < len(pairs):
        code, val = pairs[i]
        if code == "2" and val == "ENTITIES":
            in_entities = True
        elif code == "0" and val == "ENDSEC":
            in_entities = False
        elif in_entities and code == "0" and val in ("POLYLINE", "LWPOLYLINE"):
            cur: Dict[str, Any] = {"type": val, "layer": None, "vertices_raw": [], "entity_index": len(polys)}
            polys.append(cur)
            i += 1
            while i < len(pairs) and pairs[i][0] != "0":
                c2, v2 = pairs[i]
                if c2 == "8":
                    cur["layer"] = v2
                elif c2 == "10" and val == "LWPOLYLINE":
                    cur["vertices_raw"].append([v2, None, None])
                elif c2 == "20" and val == "LWPOLYLINE" and cur["vertices_raw"]:
                    cur["vertices_raw"][-1][1] = v2
                elif c2 == "30" and val == "LWPOLYLINE" and cur["vertices_raw"]:
                    cur["vertices_raw"][-1][2] = v2
                i += 1
            if val == "POLYLINE":
                while i < len(pairs):
                    c2, v2 = pairs[i]
                    if c2 == "0" and v2 == "VERTEX":
                        vx = [None, None, None]
                        i += 1
                        while i < len(pairs) and pairs[i][0] != "0":
                            c3, v3 = pairs[i]
                            if c3 == "10":
                                vx[0] = v3
                            elif c3 == "20":
                                vx[1] = v3
                            elif c3 == "30":
                                vx[2] = v3
                            i += 1
                        cur["vertices_raw"].append(vx)
                        continue
                    if c2 == "0" and v2 == "SEQEND":
                        i += 1
                        break
                    break
            continue
        i += 1

    traces: List[Dict[str, Any]] = []
    for p in polys:
        vr = p["vertices_raw"]
        uniq = vr[:-1] if len(vr) >= 2 and vr[0] == vr[-1] else vr
        pts_np = np.array([[float(a), float(b), float(c)] for a, b, c in uniq], dtype=np.float64)
        gap = float(np.linalg.norm(pts_np[0] - pts_np[-1])) if len(pts_np) >= 2 else 0.0
        traces.append({
            "entity_index": p["entity_index"],
            "layer": p["layer"],
            "vertices_raw": uniq,
            "pts_np": pts_np,
            "centroid_np": pts_np.mean(axis=0),
            "open_endpoint_gap_m": gap,
        })
    return traces


# ---------------------------------------------------------------------------
# Cluster Construction Shared Kinematics & Mode Builder
# ---------------------------------------------------------------------------

def build_cluster_model(
    *,
    cluster_id: str,
    site_id: str,
    coordinate_frame: str,
    stratum: str,
    prereg: Dict[str, Any],
    prereg_sha: str,
    source_files: List[Dict[str, Any]],
    cluster_local_origin: Tuple[Q, Q, Q],
    fractures: List[Dict[str, Any]],
    anchor_a_idx: int,
    anchor_b_idx: int,
    boundary_model: str,
    residual_source_desc: str,
    dxf_binding_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build exact rational model.json + register summary for one real cluster."""
    k_coeff = rnf.exact(prereg["uncertainty_and_modes_rule"]["continuous_parameter"]["design_coefficient_k"])
    n_f = len(fractures)
    fa = fractures[anchor_a_idx]
    fb = fractures[anchor_b_idx]

    pt_a: Tuple[Q, Q, Q] = fa["anchor_point_q"]
    pt_b: Tuple[Q, Q, Q] = fb["anchor_point_q"]
    a_np = np.array([float(x) for x in pt_a], dtype=np.float64)
    b_np = np.array([float(x) for x in pt_b], dtype=np.float64)
    ab_vec = b_np - a_np
    ab_len = float(np.linalg.norm(ab_vec))
    ab_unit = ab_vec / ab_len if ab_len > 1e-12 else np.array([1.0, 0.0, 0.0])

    # Interior fracture ordering for velocity signs & critical corridor ranking
    interior_indices = [i for i in range(n_f) if i not in (anchor_a_idx, anchor_b_idx)]
    interior_by_ab = sorted(
        interior_indices,
        key=lambda i: (float(np.dot(fractures[i]["center_local_np"] - a_np, ab_unit)), fractures[i]["fracture_id"]),
    )
    vel_sign: Dict[int, int] = {anchor_a_idx: 0, anchor_b_idx: 0}
    for rank, idx in enumerate(interior_by_ab):
        vel_sign[idx] = 1 if (rank % 2 == 0) else -1

    # Rank interior fractures by geometric corridor centrality (purely geometric!)
    corridor_ranked = sorted(
        interior_indices,
        key=lambda i: (
            _point_seg_dist(fractures[i]["center_local_np"], a_np, b_np) / (fractures[i]["char_radius_m"] + 0.1),
            -fractures[i]["sigma_m"],
            fractures[i]["fracture_id"],
        ),
    )
    f1_idx = corridor_ranked[0]
    f2_idx = corridor_ranked[1]

    mode_specs = prereg["uncertainty_and_modes_rule"]["strata_assignment_by_ordinal_index"][stratum]["modes"]

    def eval_fracture_geometry(
        f_idx: int,
        scale_q: Q,
        tilt_dir: int,
    ) -> Tuple[List[Tuple[Tuple[Q, Q, Q], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]], Tuple[Q, Q, Q]]:
        fr = fractures[f_idx]
        c_q: Tuple[Q, Q, Q] = fr["c_q"]
        u_q: Tuple[Q, Q, Q] = fr["u_q"]
        v_q: Tuple[Q, Q, Q] = fr["v_q"]
        n_q: Tuple[Q, Q, Q] = fr["n_q"]
        piv_u, piv_v = fr["pivot_2d"]
        piv_3d = (
            c_q[0] + piv_u * u_q[0] + piv_v * v_q[0],
            c_q[1] + piv_u * u_q[1] + piv_v * v_q[1],
            c_q[2] + piv_u * u_q[2] + piv_v * v_q[2],
        )

        if tilt_dir == 0:
            v_eff = v_q
            n_eff = n_q
        else:
            st = SIN_TILT if tilt_dir > 0 else -SIN_TILT
            v_eff = tuple(COS_TILT * v_q[k] + st * n_q[k] for k in range(3))
            n_eff = tuple(-st * v_q[k] + COS_TILT * n_q[k] for k in range(3))

        # Velocity vector along effective plane normal (exact rational)
        sgn = vel_sign[f_idx]
        if sgn == 0:
            vel_q = (Q(0), Q(0), Q(0))
        else:
            mag_q = snap_q(float(k_coeff) * fr["sigma_m"])
            if mag_q == 0:
                mag_q = Q(1, 1000)  # floor 1 mm if published residual is exact 0
            signed_mag = mag_q if sgn > 0 else -mag_q
            vel_q = (signed_mag * n_eff[0], signed_mag * n_eff[1], signed_mag * n_eff[2])

        tris_3d: List[Tuple[Tuple[Q, Q, Q], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]] = []
        for a2, b2, c2 in fr["tris_2d"]:
            pts3 = []
            for pu, pv in (a2, b2, c2):
                du = (pu - piv_u) * scale_q
                dv = (pv - piv_v) * scale_q
                p3 = (
                    piv_3d[0] + du * u_q[0] + dv * v_eff[0],
                    piv_3d[1] + du * u_q[1] + dv * v_eff[1],
                    piv_3d[2] + du * u_q[2] + dv * v_eff[2],
                )
                pts3.append(p3)
            tris_3d.append((pts3[0], pts3[1], pts3[2]))
        return tris_3d, vel_q

    modes_out: List[Dict[str, Any]] = []
    mode_semantics: Dict[str, Any] = {}
    vel_nominal_by_idx: Dict[int, Tuple[Q, Q, Q]] = {}

    for mspec in mode_specs:
        mid = mspec["id"]
        f1_scale = rnf.exact(mspec["f1_scale"])
        f2_scale = rnf.exact(mspec.get("f2_scale", "1/1"))
        f2_tilt = int(mspec["f2_tilt"])

        tris_json: List[Dict[str, Any]] = []
        tri_counter = 0
        for fi in range(n_f):
            sc = Q(1)
            td = 0
            if fi == f1_idx:
                sc = f1_scale
            elif fi == f2_idx:
                sc = f2_scale
                td = f2_tilt
            tris_3d, vel_q = eval_fracture_geometry(fi, sc, td)
            if mid == "m0":
                vel_nominal_by_idx[fi] = vel_q
            label = "REAL_ALPHA_TRI" if site_id == "jardim_garcia_uh" else "CONDITIONED_OCT_DISK_TRI"
            if fi in (f1_idx, f2_idx) and (sc != 1 or td != 0):
                label = label + "_ASSUMED_PERTURBED"
            for local_ti, (va, vb, vc) in enumerate(tris_3d):
                tri_counter += 1
                tris_json.append({
                    "id": f"t_{fi:02d}_{local_ti:02d}",
                    "component": fractures[fi]["fracture_id"],
                    "vertices": [qs_vec3(va), qs_vec3(vb), qs_vec3(vc)],
                    "v": qs_vec3(vel_q),
                    "label": label,
                })

        segs_json = [
            {"lo": s[0], "hi": s[1], "lo_closed": True, "hi_closed": True}
            for s in mspec["segments"]
        ]
        modes_out.append({
            "id": mid,
            "t_closed": list(prereg["uncertainty_and_modes_rule"]["continuous_parameter"]["t_closed"]),
            "triangles": tris_json,
            "segments": segs_json,
            "edit_cost": mspec["edit_cost"],
        })

        f1_id = fractures[f1_idx]["fracture_id"]
        f2_id = fractures[f2_idx]["fracture_id"]
        tilt_tag = "TILT_0DEG" if f2_tilt == 0 else ("TILT_POS_3.95DEG" if f2_tilt > 0 else "TILT_NEG_3.95DEG")
        if stratum == "STANDARD_EXT_TILT":
            ext_tag = "EXT_1.00" if f1_scale == 1 else "EXT_1.15"
            f_assign = {f1_id: ext_tag, f2_id: tilt_tag}
            desc = f"{f1_id}:{ext_tag}, {f2_id}:{tilt_tag}"
        else:
            attrib = mspec["attrib"]
            f_assign = {f1_id: attrib, f2_id: tilt_tag}
            desc = f"mutually_exclusive:{attrib}, {f2_id}:{tilt_tag}, segments={mspec['segments']}"
        mode_semantics[mid] = {
            "edit_cost": mspec["edit_cost"],
            "stratum": stratum,
            "description": desc,
            "feature_assignments": f_assign,
        }

    # Build fracture catalog & hole preservation audit
    fracture_catalog: Dict[str, Any] = {}
    hole_audit: List[Dict[str, Any]] = []
    max_hausdorff = 0.0
    max_normal_angle_deg = 0.0
    max_tier2_euclid = 0.0

    for fi, fr in enumerate(fractures):
        role = "INTERIOR"
        if fi == anchor_a_idx:
            role = "ANCHOR_A"
        elif fi == anchor_b_idx:
            role = "ANCHOR_B"
        elif fi == f1_idx:
            role = "INTERIOR_F1"
        elif fi == f2_idx:
            role = "INTERIOR_F2"

        dip_deg, dd_deg = normal_to_dip_dipdir(fr["normal_np"])
        vn = vel_nominal_by_idx[fi]
        fracture_catalog[fr["fracture_id"]] = {
            "source_id": fr["source_id"],
            "site_id": site_id,
            "role": role,
            "normal_nominal": [round(float(x), 6) for x in fr["normal_np"]],
            "dip_deg": dip_deg,
            "dip_direction_deg": dd_deg,
            "center_local_m": [round(float(x), 5) for x in fr["center_local_np"]],
            "char_radius_m": round(float(fr["char_radius_m"]), 5),
            "area_m2": round(float(fr["area_m2"]), 5),
            "residual_sigma_m": round(float(fr["sigma_m"]), 6),
            "velocity_sign": vel_sign[fi],
            "velocity_nominal_m": [round(float(x), 6) for x in vn],
            "original_hole_count": int(fr.get("original_hole_count", 0)),
            "preserved_hole_count": int(fr.get("preserved_hole_count", 0)),
            "dxf_binding": fr.get("dxf_binding"),
        }
        max_hausdorff = max(max_hausdorff, float(fr.get("hausdorff_2d_m", 0.0)))
        max_normal_angle_deg = max(max_normal_angle_deg, float(fr.get("normal_angle_dev_deg", 0.0)))
        max_tier2_euclid = max(max_tier2_euclid, float(fr.get("tier2_vertex_euclid_bound_m", 0.0005)))

        if fr.get("original_hole_count", 0) > 0:
            # Lift 2D hole probes to exact 3D local coordinates
            probes_3d = []
            c_q, u_q, v_q = fr["c_q"], fr["u_q"], fr["v_q"]
            for pu, pv in fr["hole_probes_2d"]:
                p3 = (
                    c_q[0] + pu * u_q[0] + pv * v_q[0],
                    c_q[1] + pu * u_q[1] + pv * v_q[1],
                    c_q[2] + pu * u_q[2] + pv * v_q[2],
                )
                probes_3d.append(qs_vec3(p3))
            # Compute actual triangulated 3D area in m0 for this component
            m0_comp_tris = [t for t in modes_out[0]["triangles"] if t["component"] == fr["fracture_id"]]
            area_3d = 0.0
            for t in m0_comp_tris:
                pa, pb, pc = (rnf.vector(v) for v in t["vertices"])
                area_3d += 0.5 * math.sqrt(float(rnf.norm2(rnf.cross(rnf.sub(pb, pa), rnf.sub(pc, pa)))))
            hole_audit.append({
                "fragment_id": fr["fracture_id"],
                "original_hole_count": int(fr["original_hole_count"]),
                "preserved_hole_count": int(fr["preserved_hole_count"]),
                "original_net_area_m2": round(float(fr["original_net_area_m2"]), 6),
                "original_convex_hull_area_m2": round(float(fr["original_convex_hull_area_m2"]), 6),
                "preserved_hole_area_m2": round(float(fr["preserved_hole_area_m2"]), 6),
                "triangulated_net_area_m2": round(area_3d, 6),
                "hole_interior_probe_points_local": probes_3d,
            })

    # Non-orthogonality metrics across all distinct fracture pairs
    abs_dots: List[float] = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            abs_dots.append(abs(float(np.dot(fractures[i]["normal_np"], fractures[j]["normal_np"]))))
    non_orth_cnt = sum(1 for d in abs_dots if 0.05 < d < 0.95)

    # Quantization sensitivity check on nominal m0 geometry
    min_pos_gap, touching_cnt = compute_nominal_pair_gap_summary(modes_out[0]["triangles"])
    q_sensitive = bool(min_pos_gap is not None and min_pos_gap <= 2.0 * max_tier2_euclid)

    quantization_contract = {
        "schema": "m1-quantization-contract/v1",
        "tier1_exact_source_lift": {
            "method": (
                "Exact rational lift of workbench_scene local_origin from project.json"
                if site_id == "jardim_garcia_uh"
                else "Exact Fraction(raw_decimal_string) lift of published xlsx Xcenter/Ycenter/Zcenter/Radius/Nx/Ny/Nz and DXF trace vertices"
            ),
            "cluster_local_origin_perturbation_m": 0.0,
        },
        "tier2_rational_grid": {
            "grid_step_m": rnf.qs(GRID_STEP),
            "per_axis_bound_m": rnf.qs(HALF_STEP),
            "max_euclidean_bound_m": round(max_tier2_euclid, 6),
            "coplanarity_preserved_exact": True,
        },
        "tier3_geometry_approximation": {
            "method": boundary_model,
            "max_hausdorff_deviation_m": round(max_hausdorff, 6),
            "max_normal_angle_deviation_deg": round(max_normal_angle_deg, 6),
            "dxf_trace_deviation_summary": dxf_binding_summary,
        },
        "min_nominal_pair_gap_m": None if min_pos_gap is None else round(min_pos_gap, 6),
        "nominal_touching_pair_count": touching_cnt,
        "quantization_sensitive": q_sensitive,
        "hole_preservation_audit": hole_audit,
    }

    assumption_register = {
        "boundary_model": boundary_model,
        "assumed_edges_tag": "ASSUMED_NOT_OBSERVED",
        "assumed_edges_excluded_from_observation_support": True,
        "amendments_applied": ["B1_REAL_OUTCROP_CONDITIONED_FINITE_CLUSTER_ADMISSION", "B2_CONSUMER_REAL_DERIVED_SCOPE_EXTENSION"],
        "uncertainty_model": {
            "parameter_t_meaning": "Shared dimensionless residual-displacement parameter t in [-1, 1] translating interior fractures rigidly along their fitted plane normals",
            "velocity_formula": "v_i = sign_i * k * sigma_i * n_i for interior fractures (anchor fractures f_A, f_B fixed at v=0)",
            "design_coefficient_k": rnf.qs(k_coeff),
            "residual_source": residual_source_desc,
            "discrete_modes_mechanism": (
                "Stratum STANDARD_EXT_TILT: 2x3 modes combining f1 radial/planar boundary extension (1.00x vs 1.15x) "
                "and f2 exact Pythagorean dip tilt (0 vs +/-3.95 deg)."
                if stratum == "STANDARD_EXT_TILT"
                else "Stratum MUTUAL_EXCL_ATTRIB_MULTISEG: 6 modes combining mutually exclusive boundary-support attribution "
                "(f1 1.20x/f2 0.92x on t in [-1,-1/10] vs f2 1.20x/f1 0.92x on t in [1/10,1]) with multi-segment nominal domain [-1,-3/5]U[-2/5,1] and f2 tilt +/-3.95 deg."
            ),
        },
        "not_claimed": [
            "非地下体内无条件贯通真值 (not unconditional subsurface/in-rock connectivity ground truth)",
            "非水力连通与非渗透率结论 (not hydraulic connectivity or permeability)",
            "非直接观测边界边均标记 ASSUMED_NOT_OBSERVED 且不计入直接观测支持",
        ],
    }

    real_provenance: Dict[str, Any] = {
        "site_id": site_id,
        "cluster_id": cluster_id,
        "stratum": stratum,
        "seed_fracture_id": fractures[0]["fracture_id"],
        "coordinate_frame": coordinate_frame,
        "cluster_local_origin": qs_vec3(cluster_local_origin),
        "prereg_sha256": prereg_sha,
        "source_files": source_files,
        "built_at_utc": utc(),
    }
    if dxf_binding_summary is not None:
        real_provenance["dxf_binding_summary"] = dxf_binding_summary

    model: Dict[str, Any] = {
        "schema": rnf.SCHEMA,
        "snapshot_id": cluster_id,
        "scope": rnf.REAL_SCOPE,
        "union_complete": True,
        "region_origin": "REAL_DECLARED_RESIDUAL_DOMAIN",
        "endpoints": {
            "A": qs_vec3(pt_a),
            "B": qs_vec3(pt_b),
        },
        "modes": modes_out,
        "reference": dict(prereg["uncertainty_and_modes_rule"]["continuous_parameter"]["reference"]),
        "budget": dict(prereg["global_budgets"]["default_solver_budget"]),
        "real_provenance": real_provenance,
        "assumption_register": assumption_register,
        "quantization_contract": quantization_contract,
        "fracture_catalog": fracture_catalog,
        "mode_semantics": mode_semantics,
    }

    val_summary = rnf.validate(model, repo_root=REPO, verify_source_sha=False)

    reg_entry = {
        "cluster_id": cluster_id,
        "site_id": site_id,
        "coordinate_frame": coordinate_frame,
        "stratum": stratum,
        "model_path": f"clusters/{cluster_id}/model.json",
        "model_canonical_sha256": val_summary["input_sha256"],
        "fracture_count": n_f,
        "mode_count": len(modes_out),
        "triangles_per_mode": len(modes_out[0]["triangles"]),
        "anchor_A_fracture": fa["fracture_id"],
        "anchor_B_fracture": fb["fracture_id"],
        "critical_f1_fracture": fractures[f1_idx]["fracture_id"],
        "critical_f2_fracture": fractures[f2_idx]["fracture_id"],
        "preserved_hole_fragments": [h["fragment_id"] for h in hole_audit],
        "non_orthogonality": {
            "total_pairs": len(abs_dots),
            "non_orthogonal_pairs_0p05_0p95": non_orth_cnt,
            "non_orthogonal_fraction": round(non_orth_cnt / len(abs_dots), 4) if abs_dots else 0.0,
            "min_abs_normal_dot": round(min(abs_dots), 6) if abs_dots else None,
            "max_abs_normal_dot": round(max(abs_dots), 6) if abs_dots else None,
        },
        "min_nominal_pair_gap_m": quantization_contract["min_nominal_pair_gap_m"],
        "nominal_touching_pair_count": touching_cnt,
        "quantization_sensitive": q_sensitive,
        "max_hausdorff_deviation_m": round(max_hausdorff, 6),
        "max_normal_angle_deviation_deg": round(max_normal_angle_deg, 6),
        "boundary_model": boundary_model,
        "multi_segment_modes": [m["id"] for m in modes_out if len(m["segments"]) > 1],
    }
    return model, reg_entry


# ---------------------------------------------------------------------------
# Site A: Jardim Garcia Cluster Builder
# ---------------------------------------------------------------------------

def build_jardim_clusters(prereg: Dict[str, Any], prereg_sha: str, source_meta: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cfg = prereg["site_A_jardim_garcia"]
    proj = json.loads((REPO / JARDIM_PROJECT_REL).read_text(encoding="utf-8"))
    with open(REPO / JARDIM_PLY_REL, "rb") as fh:
        while fh.readline().strip() != b"end_header":
            pass
        pts = np.frombuffer(fh.read(), dtype="<f4").reshape(-1, 3).astype(np.float64)

    ws_origin_raw = proj["artifacts"]["workbench_scene"]["frame"]["origin"]
    ws_origin_np = np.array(ws_origin_raw, dtype=np.float64)
    # Exact Tier-1 binary float lift (for float32 origin e.g. 697.92095947265625 = 11434737/16384)
    ws_origin_q = tuple(Q(float(x)) for x in ws_origin_raw)
    pts_local = pts - ws_origin_np
    cands = proj["artifacts"]["workbench_scene"]["candidates"]

    frags_all: List[Dict[str, Any]] = []
    for frag in proj["fragments"]:
        fid = frag["fragment_id"]
        idx = np.array(frag["support_ref"]["indices"], dtype=np.int64)
        sp = pts_local[idx]
        if "support_refit" in frag["candidate_generator"]:
            unique = np.unique(np.round(sp, 12), axis=0)
            ctr = np.mean(unique, axis=0)
            u, v = canonical_basis_np(frag["plane_local"]["normal"])
            frame_src = "reconstructed_refit_mean_canonical"
        else:
            c = cands[fid]
            ctr = np.array(c["center_local"], dtype=np.float64)
            u = np.array(c["basis_u"], dtype=np.float64)
            v = np.array(c["basis_v"], dtype=np.float64)
            frame_src = "workbench_scene_candidates"
        n = np.cross(u, v)
        n = n / float(np.linalg.norm(n))

        regs = frag["finite_regions_local"]
        main_reg = max(regs, key=lambda r: Polygon(r["exterior"], r.get("holes", [])).area)
        poly = Polygon(main_reg["exterior"], main_reg.get("holes", []))
        if poly.area < cfg["fragment_eligibility"]["min_primary_region_area_m2"]:
            continue

        ext_uv = np.array(main_reg["exterior"], dtype=np.float64)
        pts3 = ctr[None, :] + ext_uv[:, 0:1] * u[None, :] + ext_uv[:, 1:2] * v[None, :]
        resids = np.abs(np.array(frag["residuals"], dtype=np.float64))
        holes = main_reg.get("holes", [])
        hole_areas = [Polygon(h).area for h in holes]

        frags_all.append({
            "fracture_id": fid,
            "source_id": f"jardim_project_v5:{fid}",
            "frame_source": frame_src,
            "ctr_np": ctr,
            "u_np": u,
            "v_np": v,
            "n_np": n,
            "center_local_np": pts3.mean(axis=0),
            "bmin": pts3.min(axis=0),
            "bmax": pts3.max(axis=0),
            "main_reg": main_reg,
            "area_m2": float(poly.area),
            "char_radius_m": math.sqrt(float(poly.area) / math.pi),
            "sigma_m": float(resids.mean() + resids.std()),
            "hole_count": len(holes),
            "max_hole_area": max(hole_areas) if hole_areas else 0.0,
        })

    # Sort deterministically by fracture_id
    frags_all.sort(key=lambda x: x["fracture_id"])
    nf = len(frags_all)

    # Precompute pairwise 3D AABB gap and centroid distance
    aabb_gap = np.zeros((nf, nf), dtype=np.float64)
    cent_dist = np.zeros((nf, nf), dtype=np.float64)
    for i in range(nf):
        for j in range(i + 1, nf):
            g = np.maximum(0.0, np.maximum(frags_all[i]["bmin"] - frags_all[j]["bmax"], frags_all[j]["bmin"] - frags_all[i]["bmax"]))
            d = float(np.linalg.norm(g))
            cd = float(np.linalg.norm(frags_all[i]["center_local_np"] - frags_all[j]["center_local_np"]))
            aabb_gap[i, j] = aabb_gap[j, i] = d
            cent_dist[i, j] = cent_dist[j, i] = cd

    max_gap = float(cfg["spatial_adjacency_rule"]["max_aabb_gap_m"])
    target_n = int(cfg["target_cluster_count"])
    k_size = int(cfg["cluster_size_k"])
    min_nonorth = int(cfg["non_orthogonality_rule"]["min_non_orthogonal_pairs"])
    ndot_lo, ndot_hi = cfg["non_orthogonality_rule"]["abs_normal_dot_open_interval"]

    def jardim_members_for_seed(seed_idx: int) -> List[int]:
        nbrs = [j for j in range(nf) if j != seed_idx and aabb_gap[seed_idx, j] <= max_gap]
        nbrs.sort(key=lambda j: (aabb_gap[seed_idx, j], cent_dist[seed_idx, j], frags_all[j]["fracture_id"]))
        return [seed_idx] + nbrs[: k_size - 1]

    def count_nonorth(indices: Sequence[int]) -> int:
        cnt = 0
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                d = abs(float(np.dot(frags_all[indices[a_pos]]["n_np"], frags_all[indices[b_pos]]["n_np"])))
                if ndot_lo < d < ndot_hi:
                    cnt += 1
        return cnt

    # Deterministic seed selection per prereg.json (enforcing min_non_orthogonal_pairs >= 2)
    single_hole_cands = [
        i for i, f in enumerate(frags_all)
        if f["hole_count"] == 1
        and len(jardim_members_for_seed(i)) == k_size
        and count_nonorth(jardim_members_for_seed(i)) >= min_nonorth
    ]
    first_seed = max(single_hole_cands, key=lambda i: (frags_all[i]["max_hole_area"], frags_all[i]["fracture_id"]))
    seeds = [first_seed]
    while len(seeds) < target_n:
        best_i = None
        best_key = (-1.0, "")
        for i in range(nf):
            if i in seeds:
                continue
            members = jardim_members_for_seed(i)
            if len(members) < k_size or count_nonorth(members) < min_nonorth:
                continue
            min_d = min(cent_dist[i, s] for s in seeds)
            key = (min_d, frags_all[i]["fracture_id"])
            if key > best_key:
                best_key = key
                best_i = i
        assert best_i is not None, "could not find enough Jardim seeds"
        seeds.append(best_i)

    source_files = [
        {"path": JARDIM_PROJECT_REL, "sha256": source_meta[JARDIM_PROJECT_REL]["sha256"], "bytes": source_meta[JARDIM_PROJECT_REL]["bytes"], "role": "immutable_jardim_project_v5"},
        {"path": JARDIM_PLY_REL, "sha256": source_meta[JARDIM_PLY_REL]["sha256"], "bytes": source_meta[JARDIM_PLY_REL]["bytes"], "role": "roi_pointcloud_ply"},
        {"path": JARDIM_OBS_REL, "sha256": source_meta[JARDIM_OBS_REL]["sha256"], "bytes": source_meta[JARDIM_OBS_REL]["bytes"], "role": "observation_descriptor"},
        {"path": JARDIM_IDX_REL, "sha256": source_meta[JARDIM_IDX_REL]["sha256"], "bytes": source_meta[JARDIM_IDX_REL]["bytes"], "role": "source_index_map"},
        {"path": JARDIM_MAN_REL, "sha256": source_meta[JARDIM_MAN_REL]["sha256"], "bytes": source_meta[JARDIM_MAN_REL]["bytes"], "role": "roi_manifest"},
    ]

    strata_cfg = prereg["uncertainty_and_modes_rule"]["strata_assignment_by_ordinal_index"]
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for ord_idx, seed_i in enumerate(seeds, 1):
        nbrs = [j for j in range(nf) if j != seed_i and aabb_gap[seed_i, j] <= max_gap]
        nbrs.sort(key=lambda j: (aabb_gap[seed_i, j], cent_dist[seed_i, j], frags_all[j]["fracture_id"]))
        member_indices = [seed_i] + nbrs[: k_size - 1]

        # Triangulate members within 64-triangle budget
        tri_cfg = cfg["polygon_triangulation_rule"]
        no_hole_max_v = int(tri_cfg["no_hole_target_max_vertices"])
        while True:
            built_frags: List[Dict[str, Any]] = []
            total_tris = 0
            for mi in member_indices:
                f = frags_all[mi]
                is_holed = f["hole_count"] > 0 and f["max_hole_area"] >= float(tri_cfg["hole_min_area_m2"])
                max_ev = int(tri_cfg["hole_exterior_target_max_vertices"]) if is_holed else no_hole_max_v
                res = simplify_and_triangulate_jardim_region(
                    f["main_reg"]["exterior"],
                    f["main_reg"].get("holes", []),
                    max_ext_verts=max_ev,
                    max_hole_verts=int(tri_cfg["hole_interior_target_max_vertices"]),
                    hole_min_area=float(tri_cfg["hole_min_area_m2"]),
                )
                c_q = snap_vec3(f["ctr_np"])
                u_q = snap_vec3(f["u_np"])
                v_q = snap_vec3(f["v_np"])
                n_q = snap_vec3(f["n_np"])
                # Measure normal angle deviation from quantized basis
                n_cross = np.cross(
                    np.array([float(x) for x in u_q]),
                    np.array([float(x) for x in v_q]),
                )
                n_cross = n_cross / float(np.linalg.norm(n_cross))
                cos_ang = max(-1.0, min(1.0, abs(float(np.dot(n_cross, f["n_np"])))))
                ang_dev_deg = math.degrees(math.acos(cos_ang))

                # Compute exact Tier-2 vertex bound against unquantized frame on simplified vertices
                max_v_disp = 0.0
                for pu, pv in res["ext_q"]:
                    p_raw = f["ctr_np"] + float(pu) * f["u_np"] + float(pv) * f["v_np"]
                    p_q = np.array([float(c_q[k] + pu * u_q[k] + pv * v_q[k]) for k in range(3)])
                    max_v_disp = max(max_v_disp, float(np.linalg.norm(p_raw - p_q)))
                # Add 2D snapping Euclidean bound sqrt(2)*(1/20000)
                max_v_disp += math.sqrt(2.0) * float(HALF_STEP)

                # Anchor point = exact rational centroid of largest-area triangle of this fragment
                best_t = max(res["tris_2d"], key=lambda t: abs(cross2(t[0], t[1], t[2])))
                au = (best_t[0][0] + best_t[1][0] + best_t[2][0]) / 3
                av = (best_t[0][1] + best_t[1][1] + best_t[2][1]) / 3
                anchor_pt_q = (
                    c_q[0] + au * u_q[0] + av * v_q[0],
                    c_q[1] + au * u_q[1] + av * v_q[1],
                    c_q[2] + au * u_q[2] + av * v_q[2],
                )

                total_tris += len(res["tris_2d"])
                built_frags.append({
                    "fracture_id": f["fracture_id"],
                    "source_id": f["source_id"],
                    "c_q": c_q,
                    "u_q": u_q,
                    "v_q": v_q,
                    "n_q": n_q,
                    "normal_np": f["n_np"],
                    "center_local_np": f["center_local_np"],
                    "bmin": f["bmin"],
                    "bmax": f["bmax"],
                    "char_radius_m": f["char_radius_m"],
                    "area_m2": f["area_m2"],
                    "sigma_m": f["sigma_m"],
                    "tris_2d": res["tris_2d"],
                    "pivot_2d": res["pivot_2d"],
                    "anchor_point_q": anchor_pt_q,
                    "hole_probes_2d": res["hole_probes_2d"],
                    "original_hole_count": res["original_hole_count"],
                    "preserved_hole_count": res["preserved_hole_count"],
                    "original_net_area_m2": res["original_net_area_m2"],
                    "original_convex_hull_area_m2": res["original_convex_hull_area_m2"],
                    "preserved_hole_area_m2": res["preserved_hole_area_m2"],
                    "hausdorff_2d_m": res["hausdorff_2d_m"],
                    "normal_angle_dev_deg": ang_dev_deg,
                    "tier2_vertex_euclid_bound_m": max_v_disp,
                    "dxf_binding": None,
                })
            if total_tris <= int(prereg["global_budgets"]["max_triangles_per_mode"]):
                break
            assert no_hole_max_v > int(tri_cfg["no_hole_fallback_max_vertices_if_budget_tight"]), "triangle budget exceeded"
            no_hole_max_v = int(tri_cfg["no_hole_fallback_max_vertices_if_budget_tight"])

        # Endpoint pair selection (pure pairwise geometry: farthest non-overlapping AABB pair, fallback farthest)
        non_ov_pairs = []
        all_pairs = []
        for i in range(len(built_frags)):
            for j in range(i + 1, len(built_frags)):
                fi_m, fj_m = member_indices[i], member_indices[j]
                cd = cent_dist[fi_m, fj_m]
                key = (cd, built_frags[i]["fracture_id"], built_frags[j]["fracture_id"])
                all_pairs.append((key, i, j))
                if aabb_gap[fi_m, fj_m] > 0.0:
                    non_ov_pairs.append((key, i, j))
        chosen_ep = max(non_ov_pairs if non_ov_pairs else all_pairs, key=lambda x: x[0])
        ep_a_idx, ep_b_idx = chosen_ep[1], chosen_ep[2]

        stratum = (
            "STANDARD_EXT_TILT"
            if ord_idx in strata_cfg["STANDARD_EXT_TILT"]["jardim_indices"]
            else "MUTUAL_EXCL_ATTRIB_MULTISEG"
        )
        cid = f"jardim_c{ord_idx:02d}"
        model, reg = build_cluster_model(
            cluster_id=cid,
            site_id=cfg["site_id"],
            coordinate_frame=cfg["coordinate_frame"],
            stratum=stratum,
            prereg=prereg,
            prereg_sha=prereg_sha,
            source_files=source_files,
            cluster_local_origin=ws_origin_q,  # type: ignore[arg-type]
            fractures=built_frags,
            anchor_a_idx=ep_a_idx,
            anchor_b_idx=ep_b_idx,
            boundary_model=tri_cfg["boundary_model"],
            residual_source_desc="Jardim project.json real point-to-plane residuals: sigma_i = mean(|r_i|) + std(|r_i|)",
            dxf_binding_summary=None,
        )
        out.append((model, reg))
    return out


# ---------------------------------------------------------------------------
# Site B: Perozzo 2024 Cluster Builder
# ---------------------------------------------------------------------------

def build_perozzo_clusters(prereg: Dict[str, Any], prereg_sha: str, source_meta: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cfg = prereg["site_B_perozzo_2024"]
    planes_all = load_perozzo_xlsx(REPO / PEROZZO_XLSX_REL)
    traces_all = load_perozzo_dxf(REPO / PEROZZO_DXF_REL)
    assert len(planes_all) == 1405 and len(traces_all) == 1405

    # Precompute trace centroids for geometric binding
    trace_ctrs = np.array([t["centroid_np"] for t in traces_all], dtype=np.float64)
    plane_ctrs_all = np.array([p["center_np"] for p in planes_all], dtype=np.float64)

    pe = cfg["plane_eligibility"]
    filt = [
        p for p in planes_all
        if p["mae"] <= float(pe["max_mean_abs_error_m"])
        and float(pe["min_radius_m"]) <= p["radius"] <= float(pe["max_radius_m"])
        and p["normal_norm"] > float(pe["min_normal_norm"])
    ]
    nf = len(filt)
    cs = np.array([p["center_np"] for p in filt], dtype=np.float64)
    rs = np.array([p["radius"] for p in filt], dtype=np.float64)
    ns = np.array([p["normal_np"] for p in filt], dtype=np.float64)

    dmat = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
    gapmat = dmat - (rs[:, None] + rs[None, :])
    clearance = float(cfg["spatial_adjacency_rule"]["clearance_m"])
    adj = (gapmat <= clearance) & (~np.eye(nf, dtype=bool))
    degs = adj.sum(axis=1)

    seed_cfg = cfg["candidate_seed_rule"]
    cands: List[int] = []
    for i in range(nf):
        if int(degs[i]) < int(seed_cfg["min_spatial_degree"]):
            continue
        nbrs = list(np.where(adj[i])[0])
        nbrs.sort(key=lambda j: (float(gapmat[i, j]), float(dmat[i, j]), filt[j]["row"]))
        sub = [i] + nbrs[:5]
        sub_n = ns[sub]
        dots = np.abs(sub_n @ sub_n.T)
        triu = dots[np.triu_indices(len(sub), 1)]
        nonorth = int(((triu > 0.05) & (triu < 0.95)).sum())
        if nonorth >= int(seed_cfg["min_non_orthogonal_pairs_in_top6"]):
            cands.append(i)

    med_y = float(np.median([filt[i]["center_np"][1] for i in cands]))
    med_z = float(np.median([filt[i]["center_np"][2] for i in cands]))
    quadrants: Dict[str, List[int]] = {
        "low_Y_low_Z": [],
        "low_Y_high_Z": [],
        "high_Y_low_Z": [],
        "high_Y_high_Z": [],
    }
    for i in cands:
        y, z = filt[i]["center_np"][1], filt[i]["center_np"][2]
        qk = ("low_Y" if y <= med_y else "high_Y") + "_" + ("low_Z" if z <= med_z else "high_Z")
        quadrants[qk].append(i)

    strat_cfg = cfg["seed_stratification_rule"]
    sizes_per_q = [int(x) for x in strat_cfg["cluster_sizes_per_quadrant"]]

    def grow_cluster(seed_i: int, target_k: int) -> List[int]:
        nbrs = list(np.where(adj[seed_i])[0])
        nbrs.sort(key=lambda j: (float(gapmat[seed_i, j]), float(dmat[seed_i, j]), filt[j]["row"]))
        members = [seed_i]
        for j in nbrs:
            if len(members) >= target_k:
                break
            members.append(int(j))
        # If direct neighbors < target_k - 1, expand 2-hop by nearest gap to cluster
        while len(members) < target_k:
            mset = set(members)
            best_j = None
            best_k = (float("inf"), float("inf"), 10**9)
            for j in range(nf):
                if j in mset:
                    continue
                g_min = min(float(gapmat[m, j]) for m in members)
                d_min = min(float(dmat[m, j]) for m in members)
                cand_k = (g_min, d_min, filt[j]["row"])
                if cand_k < best_k:
                    best_k = cand_k
                    best_j = j
            assert best_j is not None
            members.append(best_j)
        return members

    selected_clusters: List[Tuple[str, int, List[int]]] = []
    chosen_sets: List[set] = []
    chosen_seed_indices: List[int] = []

    for qk in strat_cfg["quadrant_order"]:
        q_cands = quadrants[qk]
        q_med_deg = float(np.median([degs[i] for i in q_cands]))
        for pick_pos, target_k in enumerate(sizes_per_q):
            picked = None
            for max_jaccard in (0.15, 0.30, 0.50):
                if pick_pos == 0:
                    ordered = sorted(q_cands, key=lambda i: (abs(float(degs[i]) - q_med_deg), filt[i]["row"]))
                else:
                    ordered = sorted(
                        q_cands,
                        key=lambda i: (-min(float(dmat[i, s]) for s in chosen_seed_indices), filt[i]["row"]),
                    )
                for cand_i in ordered:
                    if cand_i in chosen_seed_indices:
                        continue
                    members = grow_cluster(cand_i, target_k)
                    mset = set(members)
                    if all(len(mset & s) / len(mset | s) <= max_jaccard for s in chosen_sets):
                        picked = (cand_i, members)
                        break
                if picked is not None:
                    break
            assert picked is not None, f"failed to pick Perozzo cluster in quadrant {qk}"
            s_i, members = picked
            chosen_seed_indices.append(s_i)
            chosen_sets.append(set(members))
            selected_clusters.append((qk, s_i, members))

    source_files = [
        {"path": PEROZZO_XLSX_REL, "sha256": source_meta[PEROZZO_XLSX_REL]["sha256"], "bytes": source_meta[PEROZZO_XLSX_REL]["bytes"], "role": "manual_discontinuities_xlsx"},
        {"path": PEROZZO_DXF_REL, "sha256": source_meta[PEROZZO_DXF_REL]["sha256"], "bytes": source_meta[PEROZZO_DXF_REL]["bytes"], "role": "manual_traces_dxf"},
        {"path": PEROZZO_LAS_REL, "sha256": source_meta[PEROZZO_LAS_REL]["sha256"], "bytes": source_meta[PEROZZO_LAS_REL]["bytes"], "role": "full_resolution_las_reference"},
        {"path": PEROZZO_MAN_REL, "sha256": source_meta[PEROZZO_MAN_REL]["sha256"], "bytes": source_meta[PEROZZO_MAN_REL]["bytes"], "role": "th_download_manifest"},
        {"path": PEROZZO_REG_REL, "sha256": source_meta[PEROZZO_REG_REL]["sha256"], "bytes": source_meta[PEROZZO_REG_REL]["bytes"], "role": "th_registration_check"},
    ]

    strata_cfg = prereg["uncertainty_and_modes_rule"]["strata_assignment_by_ordinal_index"]
    poly_cfg = cfg["conditioned_polygon_model"]
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for ord_idx, (qk, seed_i, member_indices) in enumerate(selected_clusters, 1):
        seed_plane = filt[seed_i]
        # Tier-1 exact rational local origin = published center of seed fracture
        origin_q: Tuple[Q, Q, Q] = seed_plane["center_q"]
        origin_np = seed_plane["center_np"]

        built_frags: List[Dict[str, Any]] = []
        dxf_max_dev_cluster = 0.0
        dxf_mutual_count = 0

        for mi in member_indices:
            p = filt[mi]
            # Geometric DXF trace binding (nearest trace centroid in 3D, never row order)
            dists_to_traces = np.linalg.norm(trace_ctrs - p["center_np"][None, :], axis=1)
            best_t_idx = int(np.argmin(dists_to_traces))
            best_t = traces_all[best_t_idx]
            best_dist = float(dists_to_traces[best_t_idx])
            # Check mutual nearest over ALL 1405 planes
            dists_t_to_planes = np.linalg.norm(plane_ctrs_all - best_t["centroid_np"][None, :], axis=1)
            nearest_plane_row = planes_all[int(np.argmin(dists_t_to_planes))]["row"]
            mutual = nearest_plane_row == p["row"]
            if mutual:
                dxf_mutual_count += 1

            # Exact rational vertex-to-plane signed distances for bound DXF trace
            c_world_q = p["center_q"]
            n_world_q = p["normal_q"]
            nn_q = rnf.norm2(n_world_q)
            inv_norm_f = 1.0 / math.sqrt(float(nn_q))
            devs = []
            for vx, vy, vz in best_t["vertices_raw"]:
                vq = (Q(vx), Q(vy), Q(vz))
                res_q = rnf.dot(rnf.sub(vq, c_world_q), n_world_q)
                devs.append(abs(float(res_q)) * inv_norm_f)
            max_dev = max(devs) if devs else 0.0
            mean_dev = sum(devs) / len(devs) if devs else 0.0
            dxf_max_dev_cluster = max(dxf_max_dev_cluster, max_dev)

            # In-plane basis aligned with projected principal axis of bound DXF trace
            n_unit = p["normal_np"]
            pts_t = best_t["pts_np"] - best_t["centroid_np"][None, :]
            proj_t = pts_t - np.outer(pts_t @ n_unit, n_unit)
            u_np = None
            if len(proj_t) >= 2:
                _, svals, vt = np.linalg.svd(proj_t, full_matrices=False)
                if svals[0] > 1e-6:
                    cand_u = vt[0]
                    cand_u = cand_u - float(np.dot(cand_u, n_unit)) * n_unit
                    un = float(np.linalg.norm(cand_u))
                    if un > 1e-6:
                        u_np = cand_u / un
            if u_np is None:
                u_np, v_np = canonical_basis_np(n_unit)
            else:
                v_np = np.cross(n_unit, u_np)
                v_np = v_np / float(np.linalg.norm(v_np))

            # Local exact rational center and quantized orthonormal basis
            c_local_exact = rnf.sub(p["center_q"], origin_q)
            c_q = (snap_q(float(c_local_exact[0])), snap_q(float(c_local_exact[1])), snap_q(float(c_local_exact[2])))
            u_q = snap_vec3(u_np)
            v_q = snap_vec3(v_np)
            n_q = snap_vec3(n_unit)

            n_cross = np.cross(np.array([float(x) for x in u_q]), np.array([float(x) for x in v_q]))
            n_cross = n_cross / float(np.linalg.norm(n_cross))
            ang_dev_deg = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(n_cross, n_unit)))))))

            # Exact rational regular octagon (8 vertices -> 6 fan triangles from v0)
            R_q = snap_q(p["radius"])
            D_q = snap_q(float(R_q * OCT_DIAG))
            oct_2d: List[Tuple[Q, Q]] = [
                (R_q, Q(0)),
                (D_q, D_q),
                (Q(0), R_q),
                (-D_q, D_q),
                (-R_q, Q(0)),
                (-D_q, -D_q),
                (Q(0), -R_q),
                (D_q, -D_q),
            ]
            tris_2d = [(oct_2d[0], oct_2d[k], oct_2d[k + 1]) for k in range(1, 7)]

            # Tier-2 Euclidean vertex bound vs raw exact center + unquantized basis
            c_local_np = p["center_np"] - origin_np
            max_v_disp = 0.0
            for pu, pv in oct_2d:
                p_raw = c_local_np + float(pu) * u_np + float(pv) * v_np
                p_q_np = np.array([float(c_q[k] + pu * u_q[k] + pv * v_q[k]) for k in range(3)])
                max_v_disp = max(max_v_disp, float(np.linalg.norm(p_raw - p_q_np)))
            max_v_disp += math.sqrt(2.0) * float(HALF_STEP)

            # Tier-3 radial Hausdorff deviation of octagon vs circle of radius R = R*(1 - cos(22.5 deg))
            oct_hausdorff = p["radius"] * (1.0 - math.cos(math.pi / 8.0))
            oct_area = 2.0 * math.sqrt(2.0) * (p["radius"] ** 2)

            built_frags.append({
                "fracture_id": p["fracture_id"],
                "source_id": f"perozzo_manual_xlsx_row_{p['row']}",
                "c_q": c_q,
                "u_q": u_q,
                "v_q": v_q,
                "n_q": n_q,
                "normal_np": n_unit,
                "center_local_np": c_local_np,
                "char_radius_m": p["radius"],
                "area_m2": oct_area,
                "sigma_m": p["sigma"],
                "tris_2d": tris_2d,
                "pivot_2d": (Q(0), Q(0)),
                "anchor_point_q": c_q,  # exact center lies on octagon diagonal (v0, v4)
                "hole_probes_2d": [],
                "original_hole_count": 0,
                "preserved_hole_count": 0,
                "hausdorff_2d_m": oct_hausdorff,
                "normal_angle_dev_deg": ang_dev_deg,
                "tier2_vertex_euclid_bound_m": max_v_disp,
                "dxf_binding": {
                    "dxf_layer": best_t["layer"],
                    "dxf_entity_index": best_t["entity_index"],
                    "center_to_trace_centroid_dist_m": round(best_dist, 6),
                    "mutual_nearest": mutual,
                    "max_abs_vertex_to_plane_dist_m": round(max_dev, 6),
                    "mean_abs_vertex_to_plane_dist_m": round(mean_dev, 6),
                    "trace_open_endpoint_gap_m": round(best_t["open_endpoint_gap_m"], 6),
                },
            })

        # Endpoint pair selection: farthest pair with center_dist > R_i + R_j (pure pairwise geometry), fallback farthest
        non_ov = []
        all_p = []
        for i in range(len(built_frags)):
            for j in range(i + 1, len(built_frags)):
                fi_m, fj_m = member_indices[i], member_indices[j]
                cd = float(dmat[fi_m, fj_m])
                key = (cd, built_frags[i]["fracture_id"], built_frags[j]["fracture_id"])
                all_p.append((key, i, j))
                if gapmat[fi_m, fj_m] > 0.0:
                    non_ov.append((key, i, j))
        chosen_ep = max(non_ov if non_ov else all_p, key=lambda x: x[0])
        ep_a_idx, ep_b_idx = chosen_ep[1], chosen_ep[2]

        stratum = (
            "STANDARD_EXT_TILT"
            if ord_idx in strata_cfg["STANDARD_EXT_TILT"]["perozzo_indices"]
            else "MUTUAL_EXCL_ATTRIB_MULTISEG"
        )
        cid = f"perozzo_c{ord_idx:02d}"
        dxf_summary = {
            "quadrant": qk,
            "fractures_bound": len(built_frags),
            "mutual_nearest_count": dxf_mutual_count,
            "max_trace_to_plane_deviation_m": round(dxf_max_dev_cluster, 6),
        }
        model, reg = build_cluster_model(
            cluster_id=cid,
            site_id=cfg["site_id"],
            coordinate_frame=cfg["coordinate_frame"],
            stratum=stratum,
            prereg=prereg,
            prereg_sha=prereg_sha,
            source_files=source_files,
            cluster_local_origin=origin_q,
            fractures=built_frags,
            anchor_a_idx=ep_a_idx,
            anchor_b_idx=ep_b_idx,
            boundary_model=poly_cfg["boundary_model"],
            residual_source_desc="Perozzo 2024 published xlsx plane-fit residuals: sigma_i = MeanAbsError_i + StDevAbsError_i",
            dxf_binding_summary=dxf_summary,
        )
        out.append((model, reg))
    return out


def main() -> int:
    # 1. Verify preregistration freeze integrity BEFORE building anything
    prereg_path = HERE / "prereg.json"
    freeze_path = HERE / "prereg_freeze_receipt.json"
    if not prereg_path.is_file() or not freeze_path.is_file():
        print("ERROR: prereg.json or prereg_freeze_receipt.json missing", file=sys.stderr)
        return 2
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual_prereg_sha = rnf.sha_file(prereg_path)
    if actual_prereg_sha != freeze["prereg_file_sha256"]:
        print(f"ERROR: PREREG_SHA_DRIFT actual={actual_prereg_sha} frozen={freeze['prereg_file_sha256']}", file=sys.stderr)
        return 3
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    source_meta = freeze["source_files_verified"]

    # 2. Re-verify source file SHAs (all non-large files) before building
    for relp, info in source_meta.items():
        full = REPO / relp
        if not full.is_file() or full.stat().st_size != info["bytes"]:
            print(f"ERROR: SOURCE_SIZE_OR_MISSING {relp}", file=sys.stderr)
            return 4
        if info["bytes"] <= rnf.LARGE_FILE_BYTES:
            if rnf.sha_file(full) != info["sha256"]:
                print(f"ERROR: SOURCE_SHA_MISMATCH {relp}", file=sys.stderr)
                return 4

    # 3. Build Jardim & Perozzo real multi-fracture clusters
    jardim_items = build_jardim_clusters(prereg, actual_prereg_sha, source_meta)
    perozzo_items = build_perozzo_clusters(prereg, actual_prereg_sha, source_meta)
    all_items = jardim_items + perozzo_items

    clusters_dir = HERE / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    register_entries: List[Dict[str, Any]] = []

    for model, reg_entry in all_items:
        cid = model["snapshot_id"]
        cdir = clusters_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        mpath = cdir / "model.json"
        mpath.write_text(
            json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Re-read and validate with full on-disk source SHA verification
        reloaded = json.loads(mpath.read_text(encoding="utf-8"))
        vinfo = rnf.validate(reloaded, repo_root=REPO, verify_source_sha=True)
        reg_entry["model_file_bytes"] = mpath.stat().st_size
        reg_entry["model_file_sha256"] = rnf.sha_file(mpath)
        assert vinfo["input_sha256"] == reg_entry["model_canonical_sha256"]
        register_entries.append(reg_entry)

    nonorth_total_pairs = sum(e["non_orthogonality"]["total_pairs"] for e in register_entries)
    nonorth_hits = sum(e["non_orthogonality"]["non_orthogonal_pairs_0p05_0p95"] for e in register_entries)
    cluster_register = {
        "schema": "m1-cluster-register/v1",
        "task_id": "M1",
        "built_at_utc": utc(),
        "prereg_frozen_at_utc": freeze["frozen_at_utc"],
        "prereg_file_sha256": actual_prereg_sha,
        "scope": rnf.REAL_SCOPE,
        "target_model_schema": rnf.SCHEMA,
        "counts": {
            "total_clusters": len(register_entries),
            "jardim_garcia_uh": len(jardim_items),
            "perozzo_finale_ligure": len(perozzo_items),
            "standard_ext_tilt_clusters": sum(1 for e in register_entries if e["stratum"] == "STANDARD_EXT_TILT"),
            "mutual_excl_attrib_multiseg_clusters": sum(
                1 for e in register_entries if e["stratum"] == "MUTUAL_EXCL_ATTRIB_MULTISEG"
            ),
            "clusters_with_preserved_holes": sum(1 for e in register_entries if e["preserved_hole_fragments"]),
            "quantization_sensitive_clusters": sum(1 for e in register_entries if e["quantization_sensitive"]),
        },
        "non_orthogonality_aggregate": {
            "total_fracture_pairs": nonorth_total_pairs,
            "non_orthogonal_pairs_0p05_0p95": nonorth_hits,
            "non_orthogonal_pair_fraction": round(nonorth_hits / nonorth_total_pairs, 4) if nonorth_total_pairs else 0.0,
        },
        "source_sha256": {
            relp: {"sha256": info["sha256"], "bytes": info["bytes"]}
            for relp, info in source_meta.items()
        },
        "not_claimed": [
            "非地下体内无条件贯通真值",
            "非水力连通",
            "条件几何模型结论不等同于合成域也不等同于现场开挖真值",
        ],
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
        "clusters": register_entries,
    }

    reg_path = HERE / "cluster_register.json"
    reg_path.write_text(
        json.dumps(cluster_register, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "counts": cluster_register["counts"],
                "non_orthogonality_aggregate": cluster_register["non_orthogonality_aggregate"],
                "register_sha256": rnf.sha_file(reg_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
