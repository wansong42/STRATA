# -*- coding: utf-8 -*-
"""PF02 deterministic geometry kernel: multi-scale candidates, finite support
domains and finite predicates.

Design boundaries (do not cross them):

* The kernel is NumPy/SciPy/Shapely only.  No learned weights, no training,
  no hidden evaluation references.
* An infinite-plane intersection, a 2D projection crossing or an AABB overlap
  is NEVER sufficient for a ``yes`` relation.  Both finite domains must be
  clipped against the intersection line and supported by observed points.
* A spatial index is an efficiency device, not a definition of truth.  Every
  index result is audited against an exhaustive small-scace enumeration.
* Missing edges stay ``NOT_EVALUATED`` / ``unknown``; they are never ``no``.
* Source points are only read.  Every candidate keeps indices back into the
  original point array.
* Real-domain quality metrics stay ``undefined`` (JSON ``null``) until a human
  verified 3D reference exists.  Upstream 2D records are not a substitute.

The module reuses PF00 contract objects without changing their meaning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .contracts import FiniteRegion, canonical_sha256, file_sha256, stable_id

try:  # shapely 2.x vectorised scalar function; 1.x geometry method fallback
    from shapely import hausdorff_distance as _hausdorff
except ImportError:  # pragma: no cover - environment dependent
    def _hausdorff(left: Any, right: Any) -> float:
        return float(left.hausdorff_distance(right))


GEOMETRY_SCHEMA_VERSION = "fractureflow.surface_product.geometry_kernel.v1"
SOFTWARE_VERSION = "pf02_geometry_kernel_v1"
RECOMPUTE_SCRIPT = "src/fractureflow/surface_product/geometry_kernel.py"

EPS = 1e-12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise GeometryError("non-finite number cannot enter a public geometry payload")
    return number


def _i(value: Any) -> int:
    return int(value)


def _v3(value: Any) -> List[float]:
    return [_f(item) for item in value]


class GeometryError(ValueError):
    """A deterministic geometry boundary violation."""


class EmptyInputError(GeometryError):
    """Empty inputs must abort instead of producing a vacuous PASS."""


# ---------------------------------------------------------------------------
# PLY input (read-only)
# ---------------------------------------------------------------------------

_PLY_SCALAR = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "int64": "<i8", "uint64": "<u8",
    "float32": "<f4", "float64": "<f8",
}


def read_ply_points(source_path: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read xyz from a binary little-endian PLY.  The source is opened read-only."""
    if source_path is None:
        raise EmptyInputError("point cloud path is missing")
    target = Path(source_path)
    if not target.is_file():
        raise EmptyInputError("point cloud file is missing: %s" % target)
    with target.open("rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise GeometryError("not a PLY file: %s" % target)
        fmt = ""
        vertex_count = None
        properties: List[Tuple[str, str]] = []
        current = None
        while True:
            line = handle.readline()
            if not line:
                raise GeometryError("PLY header has no end_header")
            text = line.decode("ascii", "replace").strip()
            if text == "end_header":
                break
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current = parts[1]
                if current == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                # PLY grammar: ``property <scalar-type> <name>`` (or
                # ``property list <count-type> <item-type> <name>``).
                if parts[1] == "list":
                    raise GeometryError("vertex list properties are not supported by this reader")
                if parts[1] not in _PLY_SCALAR:
                    raise GeometryError("unsupported PLY scalar type %r" % parts[1])
                properties.append((parts[1], parts[2]))
        if fmt != "binary_little_endian":
            raise GeometryError("only binary_little_endian PLY is supported, got %r" % fmt)
        if vertex_count is None or vertex_count <= 0:
            raise EmptyInputError("PLY vertex count is empty")
        names = [name for _, name in properties]
        if not {"x", "y", "z"}.issubset(set(names)):
            raise GeometryError("PLY vertex lacks x/y/z")
        seen: Dict[str, int] = {}
        dtype_fields = []
        for ply_type, name in properties:
            count = seen.get(name, 0)
            seen[name] = count + 1
            key = name if count == 0 else "%s_%d" % (name, count)
            dtype_fields.append((key, _PLY_SCALAR[ply_type]))
        dtype = np.dtype(dtype_fields)
        records = np.fromfile(handle, dtype=dtype, count=vertex_count)
    if records.size == 0:
        raise EmptyInputError("PLY contains no vertices")
    points = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
    header = {
        "path": str(target),
        "format": fmt,
        "vertex_count": int(vertex_count),
        "property_names": names,
        "record_bytes": int(dtype.itemsize),
        "sha256": file_sha256(target),
        "byte_count": int(target.stat().st_size),
    }
    return points, header


# ---------------------------------------------------------------------------
# Local frame, input audit, parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalFrame:
    origin: List[float]
    scale: float
    unit_status: str
    coordinate_frame: str

    def __post_init__(self) -> None:
        if len(self.origin) != 3 or not all(math.isfinite(float(v)) for v in self.origin):
            raise GeometryError("local origin must be three finite numbers")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise GeometryError("local scale must be a positive finite number")
        if not self.unit_status or not self.coordinate_frame:
            raise GeometryError("unit_status and coordinate_frame are mandatory")

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - np.asarray(self.origin, dtype=np.float64)) / self.scale

    def to_source(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * self.scale + np.asarray(self.origin, dtype=np.float64)

    def roundtrip_max_error(self, points: np.ndarray) -> float:
        source = np.asarray(points, dtype=np.float64)
        return float(np.max(np.abs(self.to_source(self.to_local(source)) - source)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": _v3(self.origin),
            "scale": _f(self.scale),
            "unit_status": self.unit_status,
            "coordinate_frame": self.coordinate_frame,
        }


def make_local_frame(points: np.ndarray, *, scale: float = 1.0,
                     unit_status: str = "UNKNOWN_SOURCE_COORDINATE_UNIT",
                     coordinate_frame: str = "source_coordinates") -> LocalFrame:
    """Default scale is 1; only an explicit configuration may change it."""
    origin = np.min(np.asarray(points, dtype=np.float64), axis=0)
    return LocalFrame(_v3(origin), _f(scale), unit_status, coordinate_frame)


@dataclass(frozen=True)
class InputAudit:
    point_count: int
    finite_point_count: int
    duplicate_point_count: int
    bounds_min: List[float]
    bounds_max: List[float]
    unit: str
    unit_status: str
    coordinate_frame: str
    registration_evidence: str
    license_status: str
    scale: float
    source_sha256: str
    median_point_spacing: float
    noise_sigma: float
    working_set_size: int
    decimation_lineage: Dict[str, Any]
    degraded: bool
    degradation_reasons: List[str]
    measured_at: str
    recompute_script: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": _i(self.point_count),
            "finite_point_count": _i(self.finite_point_count),
            "duplicate_point_count": _i(self.duplicate_point_count),
            "bounds_min": _v3(self.bounds_min),
            "bounds_max": _v3(self.bounds_max),
            "unit": self.unit,
            "unit_status": self.unit_status,
            "coordinate_frame": self.coordinate_frame,
            "registration_evidence": self.registration_evidence,
            "license_status": self.license_status,
            "scale": _f(self.scale),
            "source_sha256": self.source_sha256,
            "median_point_spacing": _f(self.median_point_spacing),
            "noise_sigma": _f(self.noise_sigma),
            "working_set_size": _i(self.working_set_size),
            "decimation_lineage": dict(self.decimation_lineage),
            "degraded": bool(self.degraded),
            "degradation_reasons": list(self.degradation_reasons),
            "measured_at": self.measured_at,
            "recompute_script": self.recompute_script,
        }


@dataclass(frozen=True)
class CandidateParameters:
    """Every threshold is derived from the estimated point spacing, never fixed
    across units.  All values are recorded into the candidate batch hash."""

    seed: int = 42
    scale_labels: Tuple[str, ...] = ("fine", "coarse")
    seed_neighbor_ks: Tuple[int, ...] = (12, 48)
    seed_radius_ratios: Tuple[float, ...] = (4.0, 12.0)
    distance_tolerance_ratios: Tuple[float, ...] = (1.0, 1.6)
    adjacency_radius_ratios: Tuple[float, ...] = (2.2, 4.4)
    min_stable_cross_norm: float = 0.25
    seed_count: int = 24
    spacing_k: int = 8
    normal_k: int = 24
    normal_cos_threshold: float = 0.92
    min_inliers: int = 24
    adjacency_radius_ratio: float = 2.2
    alpha_radius_ratio: float = 3.0
    min_region_area_ratio: float = 4.0
    simplify_tolerance_ratio: float = 0.25
    support_grid_side: int = 32
    support_radius_ratio: float = 1.8
    nms_normal_cos: float = 0.985
    nms_offset_ratio: float = 1.5
    nms_overlap_ratio: float = 0.55
    max_candidates: int = 48
    working_set_size: int = 20000
    max_inliers_per_proposal: int = 6000
    near_parallel_cos: float = 0.985
    coplanar_offset_ratio: float = 2.0
    min_shared_length_ratio: float = 1.0
    min_support_coverage: float = 0.5
    min_pair_support: int = 8
    support_samples: int = 12
    index_radius_ratio: float = 1.0  # 1.0 == conservative enclosing-ball radius
    explained_radius_ratio: float = 2.5

    def config_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": _i(self.seed),
            "scale_labels": list(self.scale_labels),
            "seed_neighbor_ks": [_i(v) for v in self.seed_neighbor_ks],
            "seed_radius_ratios": [_f(v) for v in self.seed_radius_ratios],
            "distance_tolerance_ratios": [_f(v) for v in self.distance_tolerance_ratios],
            "adjacency_radius_ratios": [_f(v) for v in self.adjacency_radius_ratios],
            "min_stable_cross_norm": _f(self.min_stable_cross_norm),
            "seed_count": _i(self.seed_count),
            "spacing_k": _i(self.spacing_k),
            "normal_k": _i(self.normal_k),
            "normal_cos_threshold": _f(self.normal_cos_threshold),
            "min_inliers": _i(self.min_inliers),
            "adjacency_radius_ratio": _f(self.adjacency_radius_ratio),
            "alpha_radius_ratio": _f(self.alpha_radius_ratio),
            "min_region_area_ratio": _f(self.min_region_area_ratio),
            "simplify_tolerance_ratio": _f(self.simplify_tolerance_ratio),
            "support_grid_side": _i(self.support_grid_side),
            "support_radius_ratio": _f(self.support_radius_ratio),
            "nms_normal_cos": _f(self.nms_normal_cos),
            "nms_offset_ratio": _f(self.nms_offset_ratio),
            "nms_overlap_ratio": _f(self.nms_overlap_ratio),
            "max_candidates": _i(self.max_candidates),
            "working_set_size": _i(self.working_set_size),
            "max_inliers_per_proposal": _i(self.max_inliers_per_proposal),
            "near_parallel_cos": _f(self.near_parallel_cos),
            "coplanar_offset_ratio": _f(self.coplanar_offset_ratio),
            "min_shared_length_ratio": _f(self.min_shared_length_ratio),
            "min_support_coverage": _f(self.min_support_coverage),
            "min_pair_support": _i(self.min_pair_support),
            "support_samples": _i(self.support_samples),
            "index_radius_ratio": _f(self.index_radius_ratio),
            "explained_radius_ratio": _f(self.explained_radius_ratio),
        }


# ---------------------------------------------------------------------------
# Finite support domains
# ---------------------------------------------------------------------------


def _clean_ring(coords: Iterable[Any], *, tol: float = 1e-10) -> List[List[float]]:
    points: List[List[float]] = []
    for item in coords:
        pair = [_f(item[0]), _f(item[1])]
        if points and math.hypot(pair[0] - points[-1][0], pair[1] - points[-1][1]) <= tol:
            continue
        points.append(pair)
    if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol:
        points.pop()
    if len(points) < 3:
        raise GeometryError("degenerate ring with fewer than three distinct positions")
    points.append(list(points[0]))
    return points


def _as_regions(geometry: Any, *, min_area: float) -> List[FiniteRegion]:
    if geometry.is_empty:
        return []
    polygons = list(geometry.geoms) if isinstance(geometry, (MultiPolygon,)) else [geometry]
    regions: List[FiniteRegion] = []
    for polygon in polygons:
        if not isinstance(polygon, Polygon) or polygon.area < min_area or not polygon.is_valid:
            continue
        exterior = _clean_ring(polygon.exterior.coords)
        holes = [_clean_ring(ring.coords) for ring in polygon.interiors]
        try:
            regions.append(FiniteRegion(exterior=exterior, holes=holes))
        except Exception:
            # A near-degenerate ring is a declared failure, never a silently
            # filled or convexified domain.
            continue
    return regions


def finite_region_from_polygon(polygon: Polygon, *, spacing: float, simplify_tol: float) -> "FiniteDomain":
    """Build a FiniteDomain from an explicitly supplied 2D polygon (synthetic
    cases and exact geometric counterexamples)."""
    regions = _as_regions(polygon, min_area=EPS)
    if not regions:
        raise EmptyInputError("explicit finite domain is empty")
    return _build_domain(regions, uv=None, spacing=spacing,
                         simplify_tol=simplify_tol, construction="explicit_polygon",
                         degraded=False, degradation_reasons=[])


def _build_domain(regions: List[FiniteRegion], *, uv: Optional[np.ndarray],
                  spacing: float, simplify_tol: float, construction: str,
                  degraded: bool, degradation_reasons: Sequence[str]) -> "FiniteDomain":
    merged = unary_union([item.to_shapely() for item in regions])
    concave_area = float(merged.area)
    if uv is not None and len(uv):
        hull_pts = np.asarray(uv, dtype=np.float64)
    else:
        hull_pts = np.vstack([np.asarray(item.exterior, dtype=np.float64)
                              for item in regions])
    try:
        convex_area = float(ConvexHull(hull_pts).volume) if len(hull_pts) >= 3 else concave_area
    except Exception:
        convex_area = concave_area
    simplified = merged.simplify(max(simplify_tol, EPS), preserve_topology=True)
    area_delta = abs(concave_area - float(simplified.area))
    try:
        hausdorff = _f(_hausdorff(merged, simplified))
    except Exception:
        hausdorff = 0.0
    holes = sum(len(item.holes) for item in regions)
    topology_changed = (len(getattr(simplified, "geoms", [merged])) != len(regions)
                        or sum(len(getattr(part, "interiors", [])) for part in
                               getattr(simplified, "geoms", [simplified])) != holes)
    # Actual supported area: grid samples inside the concave domain that have
    # an observed support point within one sampling radius.  Denominators are
    # exported so the ratio is never a bare number.
    grid_total, grid_covered = 0, 0
    if uv is not None and len(uv) and concave_area > EPS:
        minx, miny, maxx, maxy = merged.bounds
        side = 32
        xs = np.linspace(minx, maxx, side)
        ys = np.linspace(miny, maxy, side)
        grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
        try:
            from shapely import contains_xy
            inside = contains_xy(merged, grid[:, 0], grid[:, 1])
        except ImportError:  # pragma: no cover
            inside = np.asarray([merged.covers(Point(x, y)) for x, y in grid])
        grid = grid[inside]
        grid_total = int(len(grid))
        if grid_total:
            tree = cKDTree(uv)
            nearest, _ = tree.query(grid, k=1)
            grid_covered = int(np.count_nonzero(nearest <= spacing * 1.8))
    coverage = (grid_covered / grid_total) if grid_total else 0.0
    return FiniteDomain(
        regions=regions,
        convex_area=_f(convex_area),
        concave_area=_f(concave_area),
        actual_support_area=_f(concave_area * coverage),
        support_grid_total=grid_total,
        support_grid_covered=grid_covered,
        support_coverage_ratio=_f(coverage),
        hole_count=holes,
        component_count=len(regions),
        simplify_tolerance=_f(simplify_tol),
        simplify_area_delta=_f(area_delta),
        simplify_hausdorff=hausdorff,
        simplify_topology_changed=bool(topology_changed),
        construction=construction,
        degraded=bool(degraded),
        degradation_reasons=list(degradation_reasons),
    )


@dataclass
class FiniteDomain:
    regions: List[FiniteRegion]
    convex_area: float
    concave_area: float
    actual_support_area: float
    support_grid_total: int
    support_grid_covered: int
    support_coverage_ratio: float
    hole_count: int
    component_count: int
    simplify_tolerance: float
    simplify_area_delta: float
    simplify_hausdorff: float
    simplify_topology_changed: bool
    construction: str
    degraded: bool
    degradation_reasons: List[str]

    def polygon(self) -> Any:
        return unary_union([item.to_shapely() for item in self.regions])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regions": [item.to_dict() for item in self.regions],
            "convex_area": _f(self.convex_area),
            "concave_area": _f(self.concave_area),
            "actual_support_area": _f(self.actual_support_area),
            "support_grid_total": _i(self.support_grid_total),
            "support_grid_covered": _i(self.support_grid_covered),
            "support_coverage_ratio": _f(self.support_coverage_ratio),
            "hole_count": _i(self.hole_count),
            "component_count": _i(self.component_count),
            "simplify_tolerance": _f(self.simplify_tolerance),
            "simplify_area_delta": _f(self.simplify_area_delta),
            "simplify_hausdorff": _f(self.simplify_hausdorff),
            "simplify_topology_changed": bool(self.simplify_topology_changed),
            "construction": self.construction,
            "degraded": bool(self.degraded),
            "degradation_reasons": list(self.degradation_reasons),
        }


def alpha_shape_domain(uv: np.ndarray, *, spacing: float, params: CandidateParameters,
                       allow_convex_fallback: bool = True) -> FiniteDomain:
    """Concave finite support domain.  Triangles whose circumradius exceeds
    alpha are removed; the removed interior triangles remain holes instead of
    being silently filled by a convex hull.

    ``allow_convex_fallback=False`` makes an empty alpha domain a hard refusal
    rather than a declared degradation.  W151 support refits use this to keep
    the production path free of convex-hull fill.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) < 3:
        raise EmptyInputError("fewer than three support points")
    if len(np.unique(uv, axis=0)) < 3:
        raise GeometryError("support points are collinear; no finite area")
    alpha = spacing * params.alpha_radius_ratio
    simplify_tol = spacing * params.simplify_tolerance_ratio
    min_area = (spacing ** 2) * params.min_region_area_ratio
    degraded = False
    reasons: List[str] = []
    try:
        triangulation = Delaunay(uv)
        corners = uv[triangulation.simplices]           # (T, 3, 2)
        a, b, c = corners[:, 0, :], corners[:, 1, :], corners[:, 2, :]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            circum = (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1)
                      * np.linalg.norm(a - c, axis=1)) / (4.0 * area)
        selected = np.flatnonzero((area > EPS) & np.isfinite(circum) & (circum <= alpha))
        kept = [Polygon(corners[index]) for index in selected]
        merged = unary_union(kept) if kept else Polygon()
    except Exception as exc:
        degraded = True
        reasons.append("alpha_shape_failed:%s" % type(exc).__name__)
        merged = Polygon()
    if merged.is_empty:
        if not allow_convex_fallback:
            raise EmptyInputError(
                "alpha domain is empty and convex-hull fallback is disabled"
            )
        # A convex hull is only a declared degradation with an explicit reason;
        # it is never used as the default finite domain.
        degraded = True
        reasons.append("ALPHA_DOMAIN_EMPTY_CONVEX_HULL_FALLBACK")
        try:
            hull = ConvexHull(uv)
            merged = Polygon(uv[hull.vertices])
        except Exception:
            raise EmptyInputError("no finite support domain can be constructed")
    regions = _as_regions(merged, min_area=min_area)
    if not regions:
        degraded = True
        reasons.append("ALL_REGIONS_BELOW_MIN_AREA")
        regions = _as_regions(merged, min_area=EPS)
    if not regions:
        raise EmptyInputError("finite support domain produced no valid region")
    return _build_domain(regions, uv=uv, spacing=spacing, simplify_tol=simplify_tol,
                         construction="alpha_shape_union", degraded=degraded,
                         degradation_reasons=reasons)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _unit_normal(normal: np.ndarray) -> np.ndarray:
    value = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        raise GeometryError("degenerate plane normal")
    return value / norm


def _basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _unit_normal(normal)
    reference = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = _unit_normal(vt[-1])
    residual = (points - center) @ normal
    return normal, center, float(np.sqrt(np.mean(residual ** 2))), residual


def _connected_parts(points: np.ndarray, radius: float) -> List[np.ndarray]:
    if len(points) < 2:
        return [np.arange(len(points))]
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.size == 0:
        return [np.arange(len(points))]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


@dataclass
class FragmentCandidate:
    candidate_id: str
    observation_id: str
    roi_id: str
    support_indices: List[int]
    center_local: List[float]
    normal_local: List[float]
    offset_local: float
    basis_u: List[float]
    basis_v: List[float]
    domain: FiniteDomain
    residuals: List[float]
    scale_label: str
    lineage: List[str]
    nms_suppressed: List[str]
    quality: Dict[str, Any]
    aabb_min: List[float]
    aabb_max: List[float]

    def polygon(self) -> Any:
        return self.domain.polygon()

    def radius(self) -> float:
        return 0.5 * float(np.linalg.norm(np.asarray(self.aabb_max) - np.asarray(self.aabb_min)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "observation_id": self.observation_id,
            "roi_id": self.roi_id,
            "support_indices": [_i(v) for v in self.support_indices],
            "center_local": _v3(self.center_local),
            "normal_local": _v3(self.normal_local),
            "offset_local": _f(self.offset_local),
            "basis_u": _v3(self.basis_u),
            "basis_v": _v3(self.basis_v),
            "domain": self.domain.to_dict(),
            "residuals": [_f(v) for v in self.residuals],
            "scale_label": self.scale_label,
            "lineage": list(self.lineage),
            "nms_suppressed": list(self.nms_suppressed),
            "quality": dict(self.quality),
            "aabb_min": _v3(self.aabb_min),
            "aabb_max": _v3(self.aabb_max),
        }


def candidate_from_plane(*, candidate_id: str, normal: Sequence[float],
                         plane_point: Sequence[float], polygon: Polygon,
                         spacing: float = 0.25, roi_id: str = "roi_main",
                         observation_id: str = "obs_main",
                         domain_basis_u: Optional[Sequence[float]] = None,
                         support_indices: Optional[Sequence[int]] = None,
                         scale_label: str = "explicit") -> FragmentCandidate:
    """Exact finite fragment for synthetic counterexamples.  Support points are
    a deterministic grid inside the declared polygon, so pair decisions have a
    real, inspectable support denominator."""
    n = _unit_normal(np.asarray(normal, dtype=np.float64))
    center = np.asarray(plane_point, dtype=np.float64)
    u = _unit_normal(np.asarray(domain_basis_u, dtype=np.float64)) if domain_basis_u is not None else _basis(n)[0]
    u = u - np.dot(u, n) * n
    u = _unit_normal(u)
    v = np.cross(n, u)
    minx, miny, maxx, maxy = polygon.bounds
    count = 24
    xs = np.linspace(minx, maxx, count)
    ys = np.linspace(miny, maxy, count)
    uv = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    try:
        from shapely import contains_xy
        mask = contains_xy(polygon, uv[:, 0], uv[:, 1])
    except ImportError:  # pragma: no cover
        mask = np.asarray([polygon.covers(Point(x, y)) for x, y in uv])
    uv = uv[mask]
    if len(uv) < 3:
        raise EmptyInputError("explicit plane has too few grid support points")
    local_points = center + np.outer(uv[:, 0], u) + np.outer(uv[:, 1], v)
    domain = finite_region_from_polygon(polygon, spacing=spacing,
                                        simplify_tol=spacing * 0.25)
    rings = []
    for region in domain.regions:
        rings.extend([np.asarray(region.exterior, dtype=np.float64)])
        rings.extend(np.asarray(hole, dtype=np.float64) for hole in region.holes)
    world = np.vstack([center + np.outer(r[:, 0], u) + np.outer(r[:, 1], v) for r in rings])
    aabb_min, aabb_max = np.min(world, axis=0), np.max(world, axis=0)
    indices = ([_i(v) for v in support_indices] if support_indices is not None
               else list(range(len(uv))))
    return FragmentCandidate(
        candidate_id=candidate_id, observation_id=observation_id, roi_id=roi_id,
        support_indices=indices, center_local=_v3(center),
        normal_local=_v3(n), offset_local=_f(float(np.dot(n, center))),
        basis_u=_v3(u), basis_v=_v3(v), domain=domain,
        residuals=[0.0] * len(uv), scale_label=scale_label,
        lineage=[], nms_suppressed=[],
        quality={"support_point_count": _i(len(uv)), "rmse": 0.0, "source": "explicit_synthetic_plane"},
        aabb_min=_v3(aabb_min), aabb_max=_v3(aabb_max),
    )


def _make_candidate(proposal_id: str, points_local: np.ndarray,
                    global_indices: np.ndarray, spacing: float,
                    params: CandidateParameters, *, observation_id: str, roi_id: str,
                    scale_label: str) -> Optional[FragmentCandidate]:
    if len(global_indices) < params.min_inliers:
        return None
    pts = points_local[global_indices]
    try:
        normal, center, rmse, residual = _fit_plane(pts)
        # One robust refit: drop points whose residual is too large for the
        # fitted plane.  This prevents a second nearby plane from tilting the
        # normal when a coarse-scale tolerance collected a few of its points.
        keep = np.abs(residual) <= max(2.0 * rmse, 0.75 * spacing)
        if params.min_inliers <= int(keep.sum()) < len(pts):
            global_indices = global_indices[keep]
            pts = points_local[global_indices]
            normal, center, rmse, residual = _fit_plane(pts)
        u, v = _basis(normal)
        uv = np.column_stack([(pts - center) @ u, (pts - center) @ v])
        domain = alpha_shape_domain(uv, spacing=spacing, params=params)
    except (EmptyInputError, GeometryError):
        return None
    rings = []
    for region in domain.regions:
        rings.append(np.asarray(region.exterior, dtype=np.float64))
        rings.extend(np.asarray(hole, dtype=np.float64) for hole in region.holes)
    world = np.vstack([center + np.outer(r[:, 0], u) + np.outer(r[:, 1], v) for r in rings])
    aabb_min, aabb_max = np.min(world, axis=0), np.max(world, axis=0)
    payload = {
        "proposal": proposal_id,
        "scale_label": scale_label,
        "support": sorted(_i(item) for item in global_indices),
    }
    return FragmentCandidate(
        candidate_id=stable_id("frag", payload),
        observation_id=observation_id, roi_id=roi_id,
        support_indices=[_i(item) for item in global_indices],
        center_local=_v3(center), normal_local=_v3(normal),
        offset_local=_f(float(np.dot(normal, center))),
        basis_u=_v3(u), basis_v=_v3(v), domain=domain,
        residuals=[_f(item) for item in residual[:512]],
        scale_label=scale_label, lineage=[proposal_id], nms_suppressed=[],
        quality={
            "support_point_count": _i(len(global_indices)),
            "rmse": _f(rmse),
            "source": "deterministic_multiscale_seed",
        },
        aabb_min=_v3(aabb_min), aabb_max=_v3(aabb_max),
    )


def _nms(candidates: List[FragmentCandidate], params: CandidateParameters,
         spacing: float) -> Tuple[List[FragmentCandidate], int]:
    """Suppress duplicate proposals with explicit geometric evidence; the merge
    lineage is kept on the survivor."""
    ordered = sorted(candidates, key=lambda item: (-len(item.support_indices), item.candidate_id))
    kept: List[FragmentCandidate] = []
    suppressed = 0
    for candidate in ordered:
        duplicate_of = None
        for survivor in kept:
            n_dot = abs(float(np.dot(np.asarray(candidate.normal_local), np.asarray(survivor.normal_local))))
            offset_gap = abs(candidate.offset_local - survivor.offset_local)
            if n_dot < params.nms_normal_cos or offset_gap > params.nms_offset_ratio * spacing:
                continue
            left, right = candidate.polygon(), survivor.polygon()
            intersection = left.intersection(right).area
            denominator = min(left.area, right.area)
            geometric_overlap = intersection / denominator if denominator > EPS else 0.0
            smaller = min(len(candidate.support_indices), len(survivor.support_indices))
            shared_support = (len(set(candidate.support_indices) & set(survivor.support_indices)) / smaller
                              if smaller else 0.0)
            if geometric_overlap >= params.nms_overlap_ratio or shared_support >= 0.5:
                duplicate_of = survivor
                break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            suppressed += 1
            duplicate_of.nms_suppressed.append(candidate.candidate_id)
            duplicate_of.lineage.extend(candidate.lineage)
        if len(kept) >= params.max_candidates:
            break
    return kept, suppressed


@dataclass
class CandidateBatch:
    batch_id: str
    parameters: CandidateParameters
    config_hash: str
    seed: int
    input_sha256: str
    software_version: str
    frame: LocalFrame
    audit: InputAudit
    fragments: List[FragmentCandidate]
    residual_pool_indices: List[int]
    unexplained_indices: List[int]
    proposal_total: int
    suppressed_total: int
    failures: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    measured_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "parameters": self.parameters.to_dict(),
            "config_hash": self.config_hash,
            "seed": _i(self.seed),
            "input_sha256": self.input_sha256,
            "software_version": self.software_version,
            "frame": self.frame.to_dict(),
            "audit": self.audit.to_dict(),
            "fragments": [item.to_dict() for item in self.fragments],
            "residual_pool_indices": [_i(v) for v in self.residual_pool_indices],
            "unexplained_indices_sample": [_i(v) for v in self.unexplained_indices[:200]],
            "proposal_total": _i(self.proposal_total),
            "suppressed_total": _i(self.suppressed_total),
            "failures": list(self.failures),
            "statistics": self.statistics,
            "measured_at": self.measured_at,
            "recompute_script": RECOMPUTE_SCRIPT,
        }


def estimate_local_geometry(points_local: np.ndarray, params: CandidateParameters
                            ) -> Tuple[float, float, np.ndarray]:
    """Median knn spacing, robust plane-fit noise and local PCA normals."""
    if len(points_local) < 4:
        raise EmptyInputError("working set is too small for local geometry estimation")
    tree = cKDTree(points_local)
    k = min(params.spacing_k + 1, len(points_local))
    nk = min(params.normal_k, len(points_local))
    max_k = max(k, nk, 2)
    distances, neighbours = tree.query(points_local, k=max_k)
    rho = float(np.median(distances[:, k - 1]))
    if not math.isfinite(rho) or rho <= EPS:
        raise GeometryError("degenerate point spacing; scale is undefined")
    local = points_local[neighbours[:, :nk]]
    centered = local - local.mean(axis=1, keepdims=True)
    try:
        # Batched SVD: one call for the whole working set instead of one per
        # point.  This is the difference between seconds and minutes on a
        # 300k-point real cloud.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normals = vt[:, -1, :].astype(np.float64)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.tile(
            np.array([[0.0, 0.0, 1.0]]), (len(normals), 1)), where=lengths > EPS)
        noise = np.median(np.abs(np.einsum("nki,ni->nk", centered, normals)), axis=1)
    except np.linalg.LinAlgError:
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points_local), 1))
        noise = np.zeros(len(points_local), dtype=np.float64)
    sigma = float(np.median(noise))
    return rho, sigma, normals


def reference_match_report(fragments: Sequence[FragmentCandidate],
                           reference_planes: Optional[Sequence[Mapping[str, Any]]],
                           *, frame: LocalFrame, spacing: float,
                           params: CandidateParameters) -> Dict[str, Any]:
    """Synthetic-domain candidate recall with an explicit denominator.

    A reference plane counts as matched only when a candidate has a near
    identical plane, a close offset and enough support points.  This is a
    functional probe, not a research quality measurement."""
    criterion = ("abs(n_candidate . n_ref) >= nms_normal_cos AND "
                 "abs(n_ref . centre - d_ref) <= 3*spacing AND "
                 "support_point_count >= min_pair_support")
    if not reference_planes:
        return {"reference_total": 0, "matched_total": 0, "recall": None,
                "recall_status": "undefined_empty_reference_set",
                "matching_criterion": criterion, "per_reference": []}
    per_reference = []
    for ref in reference_planes:
        normal = _unit_normal(np.asarray(ref["normal"], dtype=np.float64))
        point_local = frame.to_local(np.asarray(ref["plane_point"], dtype=np.float64).reshape(1, 3))[0]
        offset = float(np.dot(normal, point_local))
        matched = []
        for fragment in fragments:
            if abs(float(np.dot(np.asarray(fragment.normal_local), normal))) < params.nms_normal_cos:
                continue
            if abs(float(np.dot(normal, np.asarray(fragment.center_local))) - offset) > 3.0 * spacing:
                continue
            if len(fragment.support_indices) < params.min_pair_support:
                continue
            matched.append(fragment.candidate_id)
        per_reference.append({
            "label": str(ref["label"]),
            "matched": bool(matched),
            "matched_candidate_count": _i(len(matched)),
            "matched_candidate_ids": matched[:5],
        })
    total = len(per_reference)
    matched_total = sum(1 for item in per_reference if item["matched"])
    return {
        "reference_total": _i(total),
        "matched_total": _i(matched_total),
        "recall": _f(matched_total / total) if total else None,
        "recall_status": "enumerated" if total else "undefined_empty_reference_set",
        "matching_criterion": criterion,
        "per_reference": per_reference,
        "domain": "synthetic_functional_probe",
    }


def build_candidates(points_source: np.ndarray, *, params: CandidateParameters,
                     frame: LocalFrame, audit: InputAudit,
                     observation_id: str, roi_id: str,
                     input_sha256: str,
                     reference_planes: Optional[Sequence[Mapping[str, Any]]] = None,
                     working_indices: Optional[np.ndarray] = None) -> CandidateBatch:
    """Deterministic multi-scale candidate generation.  No learned weights."""
    source = np.asarray(points_source, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise EmptyInputError("point cloud must be a non-empty N x 3 array")
    if not np.all(np.isfinite(source)):
        raise GeometryError("point cloud contains non-finite coordinates")
    before = source.tobytes()
    local = frame.to_local(source)
    if working_indices is None:
        if len(local) > params.working_set_size:
            step = int(math.ceil(len(local) / params.working_set_size))
            working_indices = np.arange(0, len(local), step, dtype=np.int64)
        else:
            working_indices = np.arange(len(local), dtype=np.int64)
    if len(working_indices) == 0:
        raise EmptyInputError("working set is empty")
    work = local[working_indices]
    rho, sigma, normals = estimate_local_geometry(work, params)
    tree = cKDTree(work)
    proposals: List[FragmentCandidate] = []
    failures: List[Dict[str, Any]] = []
    proposal_total = 0

    for label, k_neighbors, radius_ratio, tol_ratio, adjacency_ratio in zip(
            params.scale_labels, params.seed_neighbor_ks,
            params.seed_radius_ratios, params.distance_tolerance_ratios,
            params.adjacency_radius_ratios):
        seed_radius = rho * radius_ratio
        distance_tol = max(rho * tol_ratio, 2.0 * sigma)
        seeds = np.linspace(0, len(work) - 1, num=min(params.seed_count, len(work)), dtype=np.int64)
        for seed_index in seeds:
            proposal_total += 1
            proposal_id = "%s_s%03d" % (label, int(seed_index))
            axis = normals[seed_index]
            # Inliers are collected over the whole working set by plane and
            # normal consistency; spatial distance is handled by the adjacency
            # split below, so distant coplanar points can never be merged.
            plane_distance = np.abs((work - work[seed_index]) @ axis)
            normal_agreement = np.abs(normals @ axis)
            inliers = np.flatnonzero((plane_distance <= distance_tol) &
                                     (normal_agreement >= params.normal_cos_threshold))
            if len(inliers) < params.min_inliers:
                failures.append({"proposal_id": proposal_id, "scale_label": label,
                                 "seed_local_index": _i(seed_index),
                                 "seed_radius": _f(seed_radius),
                                 "reason": "PLANAR_INLIERS_BELOW_MIN_INLIERS",
                                 "inlier_count": _i(len(inliers)),
                                 "required": _i(params.min_inliers)})
                continue
            if len(inliers) > params.max_inliers_per_proposal:
                # Bounded deterministic stride keeps the concave-domain
                # construction tractable on a dense real cloud.  The cap is a
                # declared degradation, not a silent downsample.
                stride = int(math.ceil(len(inliers) / params.max_inliers_per_proposal))
                original_count = int(len(inliers))
                inliers = inliers[::stride]
                failures.append({"proposal_id": proposal_id, "scale_label": label,
                                 "seed_local_index": _i(seed_index),
                                 "reason": "INLIER_SET_CAPPED_DETERMINISTIC_STRIDE",
                                 "original_inlier_count": _i(original_count),
                                 "kept_inlier_count": _i(len(inliers)),
                                 "stride": _i(stride)})
            parts = _connected_parts(work[inliers], rho * adjacency_ratio)
            for part_index, part in enumerate(parts):
                global_indices = working_indices[inliers[part]]
                candidate = _make_candidate(
                    "%s_p%02d" % (proposal_id, part_index), local, global_indices,
                    rho, params, observation_id=observation_id, roi_id=roi_id,
                    scale_label=label)
                if candidate is None:
                    failures.append({"proposal_id": proposal_id, "scale_label": label,
                                     "part_index": _i(part_index),
                                     "reason": "FINITE_DOMAIN_CONSTRUCTION_FAILED",
                                     "point_count": _i(len(part))})
                else:
                    proposals.append(candidate)

    kept, suppressed = _nms(proposals, params, rho)
    assigned = set(itertools.chain.from_iterable(item.support_indices for item in kept))
    residual_pool = [_i(item) for item in working_indices if _i(item) not in assigned]

    # Full-point denominator: a point is explained only when an assigned
    # support point lies within an explicitly declared radius.
    explained_radius = rho * params.explained_radius_ratio
    unexplained: List[int] = []
    if assigned:
        support_indices = np.fromiter(assigned, dtype=np.int64)
        support_tree = cKDTree(local[support_indices])
        nearest, _ = support_tree.query(local, k=1)
        unexplained = [_i(item) for item in np.flatnonzero(nearest > explained_radius)]
        explained_count = len(local) - len(unexplained)
    else:
        unexplained = list(range(len(local)))
        explained_count = 0

    if source.tobytes() != before:
        raise GeometryError("source point array was mutated during candidate generation")

    statistics = {
        "point_denominator": _i(len(local)),
        "working_set_denominator": _i(len(working_indices)),
        "explained_point_count": _i(explained_count),
        "unexplained_point_count": _i(len(unexplained)),
        "unexplained_ratio": _f(len(unexplained) / len(local)),
        "explained_radius": _f(explained_radius),
        "explained_criterion": "nearest assigned support point within explained_radius, full point array",
        "candidate_count": _i(len(kept)),
        "candidate_per_point_denominator": _i(len(local)),
        "proposal_denominator": _i(proposal_total),
        "suppressed_duplicate_count": _i(suppressed),
        "residual_pool_count": _i(len(residual_pool)),
        "residual_pool_denominator": _i(len(working_indices)),
        "median_point_spacing": _f(rho),
        "noise_sigma": _f(sigma),
        # Synthetic reference matching is explicit; real-domain quality is undefined.
        "reference_surface_total": (None if reference_planes is None else _i(len(reference_planes))),
        "matched_reference_surface_count": None,
        "candidate_recall": None,
        "candidate_recall_status": "undefined",
        "candidate_recall_reason": "HUMAN_VERIFIED_3D_REFERENCE_MISSING",
        "real_candidate_quality": None,
        "real_candidate_quality_status": "undefined",
        "real_candidate_quality_reason": "HUMAN_VERIFIED_3D_REFERENCE_MISSING",
        "unit_status": frame.unit_status,
        "metric_or_geographic_conclusions_emitted": False,
    }
    if reference_planes:
        match = reference_match_report(kept, reference_planes, frame=frame,
                                       spacing=rho, params=params)
        statistics["matched_reference_surface_count"] = match["matched_total"]
        statistics["candidate_recall"] = match["recall"]
        statistics["candidate_recall_status"] = ("synthetic_functional_probe__" + match["recall_status"])
        statistics["reference_match_report"] = match

    if not kept:
        failures.append({"proposal_id": "batch", "scale_label": "all",
                         "reason": "NO_CANDIDATE_SURVIVED",
                         "note": "missing structure is not evidence of absent structure"})

    payload = {
        "input_sha256": input_sha256,
        "config_hash": params.config_hash(),
        "seed": params.seed,
        "software_version": SOFTWARE_VERSION,
        "frame": frame.to_dict(),
        "work_size": len(working_indices),
    }
    return CandidateBatch(
        batch_id=stable_id("batch", payload),
        parameters=params, config_hash=params.config_hash(), seed=params.seed,
        input_sha256=input_sha256, software_version=SOFTWARE_VERSION,
        frame=frame, audit=audit, fragments=kept,
        residual_pool_indices=residual_pool, unexplained_indices=unexplained,
        proposal_total=proposal_total, suppressed_total=suppressed,
        failures=failures, statistics=statistics, measured_at=_now(),
    )


# ---------------------------------------------------------------------------
# Spatial index screening + exhaustive audit
# ---------------------------------------------------------------------------


def exhaustive_pairs(count: int) -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(count), 2))


def _enclosing_radius(item: FragmentCandidate) -> float:
    """Radius of the sphere centred on the plane origin that contains the whole
    finite AABB.  Using the sum of two such radii makes an AABB-overlapping
    pair impossible to miss in a centre-based index."""
    low = np.asarray(item.aabb_min, dtype=np.float64)
    high = np.asarray(item.aabb_max, dtype=np.float64)
    center = np.asarray(item.center_local, dtype=np.float64)
    corners = np.asarray([[x, y, z] for x, y, z in itertools.product(
        (low[0], high[0]), (low[1], high[1]), (low[2], high[2]))], dtype=np.float64)
    return float(np.max(np.linalg.norm(corners - center, axis=1)))


def coarse_pair_candidates(fragments: Sequence[FragmentCandidate], *,
                           params: CandidateParameters, spacing: float) -> Tuple[List[Tuple[int, int]], float]:
    """cKDTree screening on fragment centres.  The radius is the sum of the
    largest enclosing-ball radii, which makes AABB-overlapping pairs
    mathematically impossible to miss.  The index never defines truth."""
    if len(fragments) < 2:
        return [], 0.0
    centers = np.asarray([item.center_local for item in fragments], dtype=np.float64)
    radii = np.asarray([_enclosing_radius(item) for item in fragments], dtype=np.float64)
    radius = float(np.max(radii) * 2.0 * params.index_radius_ratio)
    radius = max(radius, spacing * 1e-6, EPS)
    pairs = cKDTree(centers).query_pairs(r=radius, output_type="ndarray")
    result = sorted((int(a), int(b)) for a, b in pairs if a != b)
    return result, radius


def aabb_overlap_3d(left: FragmentCandidate, right: FragmentCandidate, *,
                    tolerance: float = 0.0) -> bool:
    lo = np.asarray(left.aabb_min) - tolerance
    hi = np.asarray(left.aabb_max) + tolerance
    lo2 = np.asarray(right.aabb_min) - tolerance
    hi2 = np.asarray(right.aabb_max) + tolerance
    return bool(np.all(lo <= hi2) and np.all(lo2 <= hi))


@dataclass
class SpatialIndexAudit:
    exhaustive_pair_total: int
    coarse_pair_total: int
    reference_relation_total: int
    missed_reference_relations: int
    coarse_recall: Optional[float]
    coarse_recall_status: str
    index_radius: float
    index_radius_ratio: float
    enumerated_by: str
    missed_examples: List[List[int]]
    negative_control: Dict[str, Any]
    measured_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exhaustive_pair_total": _i(self.exhaustive_pair_total),
            "coarse_pair_total": _i(self.coarse_pair_total),
            "reference_relation_total": _i(self.reference_relation_total),
            "missed_reference_relations": _i(self.missed_reference_relations),
            "coarse_recall": (None if self.coarse_recall is None else _f(self.coarse_recall)),
            "coarse_recall_status": self.coarse_recall_status,
            "index_radius": _f(self.index_radius),
            "index_radius_ratio": _f(self.index_radius_ratio),
            "enumerated_by": self.enumerated_by,
            "missed_examples": [list(map(_i, pair)) for pair in self.missed_examples],
            "negative_control": dict(self.negative_control),
            "measured_at": self.measured_at,
            "recompute_script": RECOMPUTE_SCRIPT,
        }


def audit_spatial_index(fragments: Sequence[FragmentCandidate], *,
                        params: CandidateParameters, spacing: float,
                        negative_ratio: float = 0.25) -> Tuple[SpatialIndexAudit, List[Tuple[int, int]]]:
    """Exhaustive small-scene audit.  Reference relations are defined by an
    independent 3D AABB enumeration, never by the index itself."""
    all_pairs = exhaustive_pairs(len(fragments))
    coarse, radius = coarse_pair_candidates(fragments, params=params, spacing=spacing)
    coarse_set = set(coarse)
    reference = [pair for pair in all_pairs if aabb_overlap_3d(fragments[pair[0]], fragments[pair[1]])]
    missed = [pair for pair in reference if pair not in coarse_set]
    # An empty reference set is undefined, not a recall of 1.0.
    recall: Optional[float] = (1.0 - len(missed) / len(reference)) if reference else None
    recall_status = "enumerated" if reference else "undefined_empty_reference_set"
    # Capability probe: progressively undersized index radii must expose
    # missed reference relations.  A single fixed ratio can be inconclusive on
    # a compact scene, so the sweep is part of the evidence.
    probes = []
    for ratio in (negative_ratio, negative_ratio * 0.4, negative_ratio * 0.16):
        small = dataclasses.replace(params, index_radius_ratio=max(ratio, 1e-6))
        small_coarse, small_radius = coarse_pair_candidates(fragments, params=small, spacing=spacing)
        small_missed = [pair for pair in reference if pair not in set(small_coarse)]
        probes.append({
            "index_radius_ratio": _f(ratio),
            "index_radius": _f(small_radius),
            "coarse_pair_total": _i(len(small_coarse)),
            "missed_reference_relations": _i(len(small_missed)),
            "missed_examples": [list(pair) for pair in small_missed[:10]],
        })
    detectable = any(probe["missed_reference_relations"] > 0 for probe in probes)
    audit = SpatialIndexAudit(
        exhaustive_pair_total=len(all_pairs),
        coarse_pair_total=len(coarse),
        reference_relation_total=len(reference),
        missed_reference_relations=len(missed),
        coarse_recall=recall,
        coarse_recall_status=recall_status,
        index_radius=_f(radius),
        index_radius_ratio=_f(params.index_radius_ratio),
        enumerated_by="full combination enumeration + independent 3D AABB overlap; index is not the criterion",
        missed_examples=[list(pair) for pair in missed[:20]],
        negative_control={
            "probes": probes,
            "detects_missed_relations": bool(detectable),
            "reference_relation_denominator": _i(len(reference)),
            "note": ("progressively undersized index radii; an empty reference set cannot "
                     "demonstrate detection capability and is not a PASS"),
        },
        measured_at=_now(),
    )
    return audit, coarse


# ---------------------------------------------------------------------------
# Finite relation predicates
# ---------------------------------------------------------------------------


@dataclass
class RelationEvidence:
    src_id: str
    dst_id: str
    relation_type: str
    state: str
    unknown_reason: Optional[str]
    infinite_plane_intersects: bool
    finite_domain_intersects: bool
    finite_shared_length: float
    support_coverage: Optional[float]
    support_coverage_status: str
    support_sample_total: int
    support_sample_covered: int
    overlap_area: Optional[float]
    reason_codes: List[str]
    evidence_refs: List[str]
    measured_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_id": self.src_id, "dst_id": self.dst_id,
            "relation_type": self.relation_type, "state": self.state,
            "unknown_reason": self.unknown_reason,
            "infinite_plane_intersects": bool(self.infinite_plane_intersects),
            "finite_domain_intersects": bool(self.finite_domain_intersects),
            "finite_shared_length": _f(self.finite_shared_length),
            "support_coverage": (None if self.support_coverage is None else _f(self.support_coverage)),
            "support_coverage_status": self.support_coverage_status,
            "support_sample_total": _i(self.support_sample_total),
            "support_sample_covered": _i(self.support_sample_covered),
            "overlap_area": (None if self.overlap_area is None else _f(self.overlap_area)),
            "reason_codes": list(self.reason_codes),
            "evidence_refs": list(self.evidence_refs),
            "denominator": _i(self.support_sample_total),
            "measured_at": self.measured_at,
            "recompute_script": RECOMPUTE_SCRIPT,
        }


def _line_intervals_in_polygon(polygon: Any, u0: float, v0: float,
                               du: float, dv: float, span: float,
                               t_center: float = 0.0) -> List[Tuple[float, float]]:
    """Clip an infinite line to a finite (possibly holed) polygon and return
    global parameter intervals along that line.  ``t_center`` only moves the
    sampling window; it never changes the parameter origin."""
    norm2 = du * du + dv * dv
    if norm2 <= EPS:
        return []
    low_t, high_t = t_center - span, t_center + span
    line = LineString([(u0 + du * low_t, v0 + dv * low_t),
                       (u0 + du * high_t, v0 + dv * high_t)])
    clipped = line.intersection(polygon)
    segments: List[Any] = []
    if isinstance(clipped, LineString):
        segments = [clipped]
    elif isinstance(clipped, MultiLineString):
        segments = list(clipped.geoms)
    elif hasattr(clipped, "geoms"):
        segments = [part for part in clipped.geoms if isinstance(part, LineString)]
    intervals: List[Tuple[float, float]] = []
    for segment in segments:
        coords = np.asarray(segment.coords, dtype=np.float64)
        if len(coords) < 2:
            continue
        parameters = ((coords[:, 0] - u0) * du + (coords[:, 1] - v0) * dv) / norm2
        low, high = float(np.min(parameters)), float(np.max(parameters))
        if high - low > EPS:
            intervals.append((low, high))
    return intervals


def _interval_overlap(left: Sequence[Tuple[float, float]],
                      right: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    result = []
    for a0, a1 in left:
        for b0, b1 in right:
            low, high = max(a0, b0), min(a1, b1)
            if high - low > EPS:
                result.append((low, high))
    return result


def _support_points_local(candidate: FragmentCandidate,
                          points_local: np.ndarray) -> np.ndarray:
    indices = np.asarray(candidate.support_indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(points_local))]
    return points_local[indices] if len(indices) else np.empty((0, 3), dtype=np.float64)


def _coverage_along_intervals(intervals: Sequence[Tuple[float, float]],
                              p0: np.ndarray, direction: np.ndarray,
                              left: FragmentCandidate, right: FragmentCandidate,
                              points_local: np.ndarray, params: CandidateParameters,
                              support_radius: float) -> Tuple[Optional[float], int, int]:
    total = max(1, params.support_samples)
    if not intervals:
        # No shared finite segment: coverage is undefined, not 0.
        return None, 0, 0
    length = sum(high - low for low, high in intervals)
    samples: List[np.ndarray] = []
    for low, high in intervals:
        share = max(1, int(round(total * (high - low) / length)))
        for value in np.linspace(low, high, share, endpoint=False):
            samples.append(p0 + value * direction)
    samples_array = np.asarray(samples, dtype=np.float64)
    left_points = _support_points_local(left, points_local)
    right_points = _support_points_local(right, points_local)
    covered = 0
    if len(left_points) and len(right_points):
        left_tree, right_tree = cKDTree(left_points), cKDTree(right_points)
        left_dist, _ = left_tree.query(samples_array, k=1)
        right_dist, _ = right_tree.query(samples_array, k=1)
        covered = int(np.count_nonzero((left_dist <= support_radius) & (right_dist <= support_radius)))
    return (covered / len(samples_array)), len(samples_array), covered


def evaluate_pair(left: FragmentCandidate, right: FragmentCandidate, *,
                  params: CandidateParameters, spacing: float,
                  points_local: np.ndarray) -> RelationEvidence:
    """Finite predicate.  Infinite-plane intersection is only an intermediate
    computation; a yes requires a shared finite segment with support on both
    sides.  Near-parallel / coplanar objects take a separate branch."""
    n1 = np.asarray(left.normal_local, dtype=np.float64)
    n2 = np.asarray(right.normal_local, dtype=np.float64)
    c1 = np.asarray(left.center_local, dtype=np.float64)
    c2 = np.asarray(right.center_local, dtype=np.float64)
    u1, v1 = np.asarray(left.basis_u), np.asarray(left.basis_v)
    u2, v2 = np.asarray(right.basis_u), np.asarray(right.basis_v)
    d1, d2 = float(np.dot(n1, c1)), float(np.dot(n2, c2))
    alignment = abs(float(np.dot(n1, n2)))
    support_radius = spacing * params.support_radius_ratio
    min_shared = spacing * params.min_shared_length_ratio
    codes: List[str] = []
    overlap_area: Optional[float] = None
    shared_length = 0.0
    finite_intersects = False
    coverage: Optional[float] = None
    sample_total, sample_covered = 0, 0
    coverage_status = "undefined_no_shared_finite_segment"
    infinite_intersects = False

    if alignment >= params.near_parallel_cos:
        # Near-parallel / coplanar branch: overlap, adjacent, separated.
        # An infinite-plane intersection far away is never used here.
        offset_gap = abs(float(np.dot(n1, c2 - c1)))
        coplanar = offset_gap <= params.coplanar_offset_ratio * spacing
        if coplanar:
            projected_parts = []
            for region in right.domain.regions:
                ring = np.asarray(region.exterior, dtype=np.float64)
                world = c2 + np.outer(ring[:, 0], u2) + np.outer(ring[:, 1], v2)
                projected = np.column_stack([(world - c1) @ u1, (world - c1) @ v1])
                try:
                    projected_parts.append(Polygon(projected).buffer(0))
                except Exception:
                    continue
            overlap_area = (float(left.polygon().intersection(unary_union(projected_parts)).area)
                            if projected_parts else 0.0)
        if not aabb_overlap_3d(left, right, tolerance=support_radius):
            codes.append("FINITE_DOMAINS_SEPARATED")
            state, reason, relation_type = "no", None, "intersects_in_roi"
        elif coplanar and overlap_area is not None and overlap_area > EPS:
            # Overlapping coplanar fragments remain separate instances; a
            # merge needs human confirmation, not an automatic yes.
            codes.append("COPLANAR_OVERLAP_NEEDS_HUMAN_REVIEW")
            state, reason, relation_type = "unknown", "AMBIGUOUS", "same_observed_surface"
        else:
            codes.append("NEAR_PARALLEL_NO_SHARED_FINITE_SEGMENT")
            if not coplanar:
                codes.append("NON_COPLANAR_NEAR_PARALLEL_OVERLAP_UNDEFINED")
            state, reason, relation_type = "unknown", "INSUFFICIENT_SUPPORT", "intersects_in_roi"
    else:
        cross = np.cross(n1, n2)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm <= EPS:
            state, reason, relation_type = "unknown", "AMBIGUOUS", "intersects_in_roi"
            codes.append("DEGENERATE_PLANE_PAIR")
        else:
            infinite_intersects = True
            direction = cross / cross_norm
            matrix = np.vstack([n1, n2])
            p0 = matrix.T @ np.linalg.solve(matrix @ matrix.T, np.array([d1, d2], dtype=np.float64))
            du1, dv1 = float(np.dot(direction, u1)), float(np.dot(direction, v1))
            du2, dv2 = float(np.dot(direction, u2)), float(np.dot(direction, v2))

            def _uv_extent(poly: Any) -> float:
                minx, miny, maxx, maxy = poly.bounds
                return math.hypot(maxx - minx, maxy - miny)

            span = 2.0 * max(
                _uv_extent(left.polygon()) / max(math.hypot(du1, dv1), EPS),
                _uv_extent(right.polygon()) / max(math.hypot(du2, dv2), EPS),
            ) + 1.0
            t_center = float(np.dot((c1 + c2) * 0.5 - p0, direction))
            u01, v01 = float(np.dot(p0 - c1, u1)), float(np.dot(p0 - c1, v1))
            u02, v02 = float(np.dot(p0 - c2, u2)), float(np.dot(p0 - c2, v2))
            intervals1 = _line_intervals_in_polygon(
                left.polygon(), u01, v01, du1, dv1, span, t_center)
            intervals2 = _line_intervals_in_polygon(
                right.polygon(), u02, v02, du2, dv2, span, t_center)
            shared = _interval_overlap(intervals1, intervals2)
            shared_length = sum(high - low for low, high in shared)
            finite_intersects = shared_length > EPS
            if not finite_intersects:
                codes.append("INFINITE_PLANES_INTERSECT_BUT_FINITE_DOMAINS_DISJOINT")
            coverage, sample_total, sample_covered = _coverage_along_intervals(
                shared, p0, direction, left, right, points_local, params, support_radius)
            coverage_status = "enumerated" if coverage is not None else coverage_status
            left_support = len(left.support_indices)
            right_support = len(right.support_indices)
            coverage_ok = coverage is not None and coverage >= params.min_support_coverage
            # A small angle makes the intersection line position extremely
            # sensitive to fitting noise.  Such a pair can never be promoted
            # to yes by finite-domain clipping alone.
            unstable = cross_norm < params.min_stable_cross_norm
            if unstable:
                codes.append("SMALL_ANGLE_INTERSECTION_LINE_UNSTABLE")
            if (finite_intersects and shared_length >= min_shared and coverage_ok
                    and left_support >= params.min_pair_support
                    and right_support >= params.min_pair_support and not unstable):
                codes.append("SHARED_FINITE_SEGMENT_WITH_TWO_SIDED_SUPPORT")
                state, reason, relation_type = "yes", None, "intersects_in_roi"
            elif not finite_intersects:
                state, reason, relation_type = "no", None, "intersects_in_roi"
            elif unstable:
                state, reason, relation_type = "unknown", "AMBIGUOUS", "intersects_in_roi"
            elif (left_support < params.min_pair_support or right_support < params.min_pair_support
                  or not coverage_ok):
                codes.append("SUPPORT_BELOW_MINIMUM")
                state, reason, relation_type = "unknown", "INSUFFICIENT_SUPPORT", "intersects_in_roi"
            else:
                codes.append("SHARED_SEGMENT_BELOW_MIN_LENGTH")
                state, reason, relation_type = "unknown", "AMBIGUOUS", "intersects_in_roi"

    evidence_refs = ["%s:%s" % (code, "finite_kernel") for code in codes] or ["finite_kernel:no_codes"]
    return RelationEvidence(
        src_id=left.candidate_id, dst_id=right.candidate_id,
        relation_type=relation_type, state=state, unknown_reason=reason,
        infinite_plane_intersects=bool(infinite_intersects),
        finite_domain_intersects=bool(finite_intersects),
        finite_shared_length=_f(shared_length),
        support_coverage=coverage,
        support_coverage_status=coverage_status,
        support_sample_total=_i(sample_total),
        support_sample_covered=_i(sample_covered),
        overlap_area=overlap_area,
        reason_codes=codes, evidence_refs=evidence_refs, measured_at=_now(),
    )


@dataclass
class RelationReport:
    decisions: List[RelationEvidence]
    evaluated_pair_total: int
    unevaluated_pair_total: int
    exhaustive_pair_total: int
    index_audit: SpatialIndexAudit
    statistics: Dict[str, Any]
    measured_at: str

    def state_for(self, src_id: str, dst_id: str) -> str:
        for decision in self.decisions:
            if {decision.src_id, decision.dst_id} == {src_id, dst_id}:
                return decision.state
        return "NOT_EVALUATED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": [item.to_dict() for item in self.decisions],
            "evaluated_pair_total": _i(self.evaluated_pair_total),
            "unevaluated_pair_total": _i(self.unevaluated_pair_total),
            "exhaustive_pair_total": _i(self.exhaustive_pair_total),
            "index_audit": self.index_audit.to_dict(),
            "statistics": self.statistics,
            "measured_at": self.measured_at,
            "recompute_script": RECOMPUTE_SCRIPT,
        }


def build_relation_report(fragments: Sequence[FragmentCandidate], *,
                          params: CandidateParameters, spacing: float,
                          points_local: np.ndarray,
                          index_audit: SpatialIndexAudit,
                          coarse_pairs: Sequence[Tuple[int, int]]) -> RelationReport:
    coarse_set = set(coarse_pairs)
    decisions: List[RelationEvidence] = []
    for a, b in coarse_pairs:
        decisions.append(evaluate_pair(
            fragments[a], fragments[b], params=params, spacing=spacing,
            points_local=points_local))
    counts = {"yes": 0, "no": 0, "unknown": 0}
    for decision in decisions:
        counts[decision.state] += 1

    all_pairs = exhaustive_pairs(len(fragments))
    unevaluated = [pair for pair in all_pairs if pair not in coarse_set]
    statistics = {
        "exhaustive_pair_denominator": _i(len(all_pairs)),
        "coarse_pair_denominator": _i(len(coarse_pairs)),
        "evaluated_pair_count": _i(len(decisions)),
        "unevaluated_pair_count": _i(len(unevaluated)),
        "yes_count": _i(counts["yes"]),
        "no_count": _i(counts["no"]),
        "unknown_count": _i(counts["unknown"]),
        "missing_edge_is_no": False,
        "missing_edge_policy": "unevaluated pairs are NOT_EVALUATED / unknown, never no",
        "intersection_transitivity_used": False,
        "reference_relation_denominator": _i(index_audit.reference_relation_total),
        "missed_reference_relations": _i(index_audit.missed_reference_relations),
    }
    return RelationReport(
        decisions=decisions, evaluated_pair_total=len(decisions),
        unevaluated_pair_total=len(unevaluated), exhaustive_pair_total=len(all_pairs),
        index_audit=index_audit, statistics=statistics, measured_at=_now(),
    )


# ---------------------------------------------------------------------------
# Synthetic scenes (explicitly synthetic, never research references)
# ---------------------------------------------------------------------------


def _grid_plane(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray,
                u_range: Tuple[float, float], v_range: Tuple[float, float],
                side: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    us = np.linspace(u_range[0], u_range[1], side)
    vs = np.linspace(v_range[0], v_range[1], side)
    uv = np.asarray([[u, v] for v in vs for u in us], dtype=np.float64)
    normal = np.cross(axis_u, axis_v)
    normal = normal / np.linalg.norm(normal)
    jitter = rng.normal(0.0, noise, size=(len(uv), 1))
    return origin + np.outer(uv[:, 0], axis_u) + np.outer(uv[:, 1], axis_v) + jitter * normal


def synthetic_scene(name: str, *, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Deterministic synthetic point clouds.  They are functional probes, not
    research references."""
    rng = np.random.default_rng(seed)
    x, y, z = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    if name == "same_direction_distinct_instances":
        a = _grid_plane(np.array([0.0, 0.0, 0.0]), x, y, (-2, 2), (-2, 2), 34, 0.004, rng)
        b = _grid_plane(np.array([0.0, 0.0, 6.0]), x, y, (-2, 2), (-2, 2), 34, 0.004, rng)
        points = np.vstack([a, b])
        meta = {"reference_surface_total": 2, "synthetic": True,
                "semantic": "two same-direction planes far apart must stay two instances",
                "reference_planes": [
                    {"label": "plane_a", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 0.0]},
                    {"label": "plane_b", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 6.0]},
                ]}
    elif name == "ring_with_hole":
        uv = []
        side = 46
        for value_u in np.linspace(-3.0, 3.0, side):
            for value_v in np.linspace(-3.0, 3.0, side):
                if abs(value_u) < 1.1 and abs(value_v) < 1.1:
                    continue
                uv.append([value_u, value_v])
        uv = np.asarray(uv, dtype=np.float64)
        points = np.column_stack([uv[:, 0], uv[:, 1], rng.normal(0.0, 0.004, len(uv))])
        meta = {"reference_surface_total": 1, "synthetic": True,
                "semantic": "finite support domain with a hole must not be filled",
                "reference_planes": [
                    {"label": "ring", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 0.0]},
                ]}
    elif name == "near_parallel":
        tilt = np.array([0.06, 0.0, 1.0])
        tilt = tilt / np.linalg.norm(tilt)
        u_tilt = np.cross(y, tilt)
        u_tilt = u_tilt / np.linalg.norm(u_tilt)
        a = _grid_plane(np.array([0.0, 0.0, 0.0]), x, y, (-2, 2), (-2, 2), 34, 0.004, rng)
        b = _grid_plane(np.array([0.0, 0.0, 0.5]), u_tilt, y, (-2, 2), (-2, 2), 34, 0.004, rng)
        points = np.vstack([a, b])
        meta = {"reference_surface_total": 2, "synthetic": True,
                "semantic": "near-parallel planes must not yield an infinite-plane yes",
                "reference_planes": [
                    {"label": "plane_a", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 0.0]},
                    {"label": "plane_b", "normal": [0.0599, 0.0, 0.9982], "plane_point": [0.0, 0.0, 0.5]},
                ]}
    elif name == "finite_disjoint":
        a = _grid_plane(np.array([0.0, 0.0, 0.0]), x, y, (-2, 2), (-2, 2), 32, 0.004, rng)
        b = _grid_plane(np.array([0.0, 5.0, 0.0]), x, z, (-2, 2), (-2, 2), 32, 0.004, rng)
        points = np.vstack([a, b])
        meta = {"reference_surface_total": 2, "synthetic": True,
                "semantic": "infinite planes intersect but finite supports do not",
                "reference_planes": [
                    {"label": "plane_a", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 0.0]},
                    {"label": "plane_b", "normal": [0.0, -1.0, 0.0], "plane_point": [0.0, 5.0, 0.0]},
                ]}
    elif name == "transitive_triplet":
        a = _grid_plane(np.array([0.0, 0.0, 0.0]), y, z, (-2, 2), (-2, 2), 32, 0.004, rng)
        b = _grid_plane(np.array([0.0, 0.0, 0.0]), x, y, (-2, 2), (-2, 4), 32, 0.004, rng)
        c = _grid_plane(np.array([0.0, 3.0, 0.0]), x, z, (-2, 2), (-2, 2), 32, 0.004, rng)
        points = np.vstack([a, b, c])
        meta = {"reference_surface_total": 3, "synthetic": True,
                "semantic": "A-B and B-C intersect but A-C finite supports are disjoint",
                "reference_planes": [
                    {"label": "plane_a", "normal": [1.0, 0.0, 0.0], "plane_point": [0.0, 0.0, 0.0]},
                    {"label": "plane_b", "normal": [0.0, 0.0, 1.0], "plane_point": [0.0, 0.0, 0.0]},
                    {"label": "plane_c", "normal": [0.0, -1.0, 0.0], "plane_point": [0.0, 3.0, 0.0]},
                ]}
    else:
        raise GeometryError("unknown synthetic scene %r" % name)
    if len(points) == 0:
        raise EmptyInputError("synthetic scene produced no points")
    return points, meta


SYNTHETIC_SCENES = ("same_direction_distinct_instances", "ring_with_hole",
                    "near_parallel", "finite_disjoint", "transitive_triplet")
