# -*- coding: utf-8 -*-
"""W151: real support-point geometry for surface-product edit actions.

This module is the single place where an edit action turns a *selected set of
original support points* into real local geometry:

* a least-squares plane (normal, offset, local basis, residuals),
* a concave finite support domain (holes and separate components preserved),
* explicit rejection when the selection cannot support a plane.

The production path never substitutes a synthetic plane or a convex hull for a
failed fit.  A caller that cannot resolve the original points must refuse the
edit; the only fixture reconstruction lives in
``fixture_points_for_fragment`` and every record it produces is labelled
``SYNTHETIC_FIXTURE_RECONSTRUCTION``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from .contracts import ContractError, canonical_sha256
from .geometry_kernel import CandidateParameters, EmptyInputError, GeometryError, alpha_shape_domain

MIN_SUPPORT_POINTS = 3
DEFAULT_PLANE_TOLERANCE_RATIO = 0.05
DEFAULT_ALPHA_RADIUS_RATIO = 2.0
DEFAULT_MIN_REGION_AREA_RATIO = 0.10
DEFAULT_SIMPLIFY_TOLERANCE_RATIO = 0.05
DEFAULT_COPLANAR_ANGLE_DEG = 1.0
DEFAULT_COPLANAR_OFFSET_RATIO = 0.02
MAX_RESIDUALS = 512


class SupportGeometryError(ContractError):
    """An explicit refusal, never a silently fabricated plane."""

    def __init__(self, message: str, code: str = "SUPPORT_GEOMETRY_INVALID", **details: Any) -> None:
        super().__init__(message, code)
        self.details = dict(details)


def _unit(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not math.isfinite(norm) or norm <= 1e-12:
        raise SupportGeometryError("degenerate plane normal", "PLANE_DEGENERATE")
    return array / norm


def canonical_basis(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal basis used by this module and the workbench.

    The reference axis is the least aligned coordinate axis, which keeps the
    basis stable for every normal except a measure-zero tie.
    """
    n = _unit(normal)
    axes = np.eye(3, dtype=np.float64)
    reference = axes[int(np.argmin(np.abs(n)))]
    u = np.cross(reference, n)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        alternate = axes[(int(np.argmin(np.abs(n))) + 1) % 3]
        u = np.cross(alternate, n)
        u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        raise SupportGeometryError("cannot construct a local basis", "PLANE_DEGENERATE")
    u /= u_norm
    v = np.cross(n, u)
    v /= float(np.linalg.norm(v))
    return u, v


def frame_of(normal: Sequence[float], offset: Optional[float] = None,
             origin: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    n = _unit(normal)
    if origin is None:
        if offset is None:
            raise SupportGeometryError("frame needs an origin or an offset", "FRAME_INVALID")
        origin_array = n * float(offset)
    else:
        origin_array = np.asarray(origin, dtype=np.float64)
        if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)):
            raise SupportGeometryError("frame origin is not finite", "FRAME_INVALID")
    u, v = canonical_basis(n)
    return {
        "normal": [float(item) for item in n],
        "origin": [float(item) for item in origin_array],
        "basis_u": [float(item) for item in u],
        "basis_v": [float(item) for item in v],
        "offset": float(n @ origin_array),
    }


def normalise_frame(frame: Mapping[str, Any]) -> Dict[str, Any]:
    normal = _unit(frame["normal"])
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u = np.asarray(frame["basis_u"], dtype=np.float64)
    v = np.asarray(frame["basis_v"], dtype=np.float64)
    if origin.shape != (3,) or u.shape != (3,) or v.shape != (3,):
        raise SupportGeometryError("frame vectors must be 3D", "FRAME_INVALID")
    if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise SupportGeometryError("frame vectors are not finite", "FRAME_INVALID")
    u = u - normal * float(u @ normal)
    if float(np.linalg.norm(u)) <= 1e-12:
        u, _ = canonical_basis(normal)
    else:
        u = u / float(np.linalg.norm(u))
    v = np.cross(normal, u)
    v = v / float(np.linalg.norm(v))
    return {
        "normal": [float(item) for item in normal],
        "origin": [float(item) for item in origin],
        "basis_u": [float(item) for item in u],
        "basis_v": [float(item) for item in v],
        "offset": float(normal @ origin),
    }


def project_uv(points: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u = np.asarray(frame["basis_u"], dtype=np.float64)
    v = np.asarray(frame["basis_v"], dtype=np.float64)
    delta = np.asarray(points, dtype=np.float64) - origin
    return np.column_stack([delta @ u, delta @ v])


def estimate_spacing(uv: np.ndarray) -> float:
    unique = np.unique(np.round(np.asarray(uv, dtype=np.float64), 12), axis=0)
    if len(unique) < MIN_SUPPORT_POINTS:
        raise SupportGeometryError("fewer than three distinct support points", "INSUFFICIENT_SUPPORT")
    tree = cKDTree(unique)
    distances, _ = tree.query(unique, k=2)
    positive = distances[:, 1][distances[:, 1] > 0]
    if not len(positive):
        raise SupportGeometryError("support points have no positive spacing", "INSUFFICIENT_SUPPORT")
    spacing = float(np.median(positive))
    if not math.isfinite(spacing) or spacing <= 1e-12:
        raise SupportGeometryError("support spacing is degenerate", "SUPPORT_GEOMETRY_INVALID")
    return spacing


def _plane_tolerance(extent: float, options: Mapping[str, Any]) -> float:
    explicit = options.get("plane_tolerance")
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value) or value <= 0:
            raise SupportGeometryError("plane_tolerance must be positive and finite", "FIT_OPTION_INVALID")
        return value
    ratio = float(options.get("plane_tolerance_ratio", DEFAULT_PLANE_TOLERANCE_RATIO))
    if not math.isfinite(ratio) or ratio <= 0:
        raise SupportGeometryError("plane_tolerance_ratio must be positive and finite", "FIT_OPTION_INVALID")
    return max(1e-6, ratio * max(extent, 1e-9))


@dataclass(frozen=True)
class SupportFit:
    normal: List[float]
    offset: float
    origin: List[float]
    basis_u: List[float]
    basis_v: List[float]
    residuals: List[float]
    residual_count: int
    rmse: float
    max_abs_residual: float
    plane_tolerance: float
    support_point_count: int
    regions: List[Dict[str, Any]]
    domain: Dict[str, Any]
    spacing: float
    fit_config: Dict[str, Any]

    def frame(self) -> Dict[str, Any]:
        return {
            "normal": list(self.normal),
            "origin": list(self.origin),
            "basis_u": list(self.basis_u),
            "basis_v": list(self.basis_v),
            "offset": float(self.offset),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normal": list(self.normal),
            "offset": float(self.offset),
            "origin": list(self.origin),
            "basis_u": list(self.basis_u),
            "basis_v": list(self.basis_v),
            "residuals": list(self.residuals),
            "residual_count": int(self.residual_count),
            "rmse": float(self.rmse),
            "max_abs_residual": float(self.max_abs_residual),
            "plane_tolerance": float(self.plane_tolerance),
            "support_point_count": int(self.support_point_count),
            "regions": [dict(item) for item in self.regions],
            "domain": dict(self.domain),
            "spacing": float(self.spacing),
            "fit_config": dict(self.fit_config),
        }


def fit_support_geometry(
    points_local: Sequence[Sequence[float]],
    *,
    options: Optional[Mapping[str, Any]] = None,
    previous_normal: Optional[Sequence[float]] = None,
) -> SupportFit:
    """Fit a plane and a concave finite domain to one real support selection.

    Refusals are explicit and typed: too few points, collinear points, a
    residual above the declared plane-model tolerance, or an empty/degraded
    finite domain never become a default plane or a convex hull.
    """
    fit_options = dict(options or {})
    points = np.asarray(points_local, dtype=np.float64)
    if points.size == 0:
        raise SupportGeometryError(
            "support selection is empty; at least %d points are required" % MIN_SUPPORT_POINTS,
            "INSUFFICIENT_SUPPORT", distinct_points=0,
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise SupportGeometryError("support points must be an N x 3 array", "SUPPORT_SHAPE_INVALID")
    if not np.all(np.isfinite(points)):
        raise SupportGeometryError("support points contain a non-finite coordinate", "SUPPORT_NON_FINITE")
    unique = np.unique(np.round(points, 12), axis=0)
    if len(unique) < MIN_SUPPORT_POINTS:
        raise SupportGeometryError(
            "support selection has %d distinct point(s); at least %d are required"
            % (len(unique), MIN_SUPPORT_POINTS),
            "INSUFFICIENT_SUPPORT", distinct_points=int(len(unique)),
        )
    center = np.mean(unique, axis=0)
    centered = unique - center
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SupportGeometryError("plane fit did not converge", "PLANE_FIT_FAILED") from exc
    if len(singular) < 2 or singular[0] <= 1e-12 or singular[1] <= max(1e-12, singular[0] * 1e-10):
        raise SupportGeometryError("support points are collinear; no plane can be fitted", "PLANE_DEGENERATE")
    normal = _unit(vt[-1])
    if previous_normal is not None:
        previous = np.asarray(previous_normal, dtype=np.float64)
        if float(previous @ normal) < 0.0:
            normal = -normal
    elif float(normal[2]) < 0.0:
        # Stable orientation for the common sub-horizontal fracture plane.
        normal = -normal
    residual = centered @ normal
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_abs = float(np.max(np.abs(residual)))
    extent = float(np.linalg.norm(np.ptp(unique, axis=0)))
    tolerance = _plane_tolerance(extent, fit_options)
    if max_abs > tolerance:
        raise SupportGeometryError(
            "support selection does not satisfy the plane model: max |residual| %.6g > tolerance %.6g"
            % (max_abs, tolerance),
            "PLANE_MODEL_REJECTED", max_abs_residual=max_abs, plane_tolerance=tolerance,
            support_point_count=int(len(unique)),
        )
    u, v = canonical_basis(normal)
    frame = {
        "normal": [float(item) for item in normal],
        "origin": [float(item) for item in center],
        "basis_u": [float(item) for item in u],
        "basis_v": [float(item) for item in v],
        "offset": float(normal @ center),
    }
    uv = project_uv(unique, frame)
    spacing = estimate_spacing(uv)
    params = CandidateParameters(
        alpha_radius_ratio=float(fit_options.get("alpha_radius_ratio", DEFAULT_ALPHA_RADIUS_RATIO)),
        min_region_area_ratio=float(fit_options.get("min_region_area_ratio", DEFAULT_MIN_REGION_AREA_RATIO)),
        simplify_tolerance_ratio=float(fit_options.get("simplify_tolerance_ratio", DEFAULT_SIMPLIFY_TOLERANCE_RATIO)),
    )
    try:
        domain = alpha_shape_domain(uv, spacing=spacing, params=params, allow_convex_fallback=False)
    except EmptyInputError as exc:
        raise SupportGeometryError(
            "finite support domain is empty for this selection: %s" % exc,
            "FINITE_DOMAIN_EMPTY", spacing=spacing,
        ) from exc
    except GeometryError as exc:
        raise SupportGeometryError(
            "finite support domain cannot be built: %s" % exc, "FINITE_DOMAIN_INVALID", spacing=spacing,
        ) from exc
    if domain.degraded:
        raise SupportGeometryError(
            "finite support domain is degraded; refusing to fill unobserved area: %s"
            % list(domain.degradation_reasons),
            "FINITE_DOMAIN_DEGRADED", degradation_reasons=list(domain.degradation_reasons),
        )
    regions = [region.to_dict() for region in domain.regions]
    if not regions:
        raise SupportGeometryError("finite support domain has no valid region", "FINITE_DOMAIN_EMPTY")
    residuals = [float(item) for item in residual[:MAX_RESIDUALS]]
    config = {
        "method": "svd_plane_plus_alpha_shape",
        "plane_tolerance": float(tolerance),
        "plane_tolerance_ratio": float(fit_options.get("plane_tolerance_ratio", DEFAULT_PLANE_TOLERANCE_RATIO)),
        "alpha_radius_ratio": float(params.alpha_radius_ratio),
        "min_region_area_ratio": float(params.min_region_area_ratio),
        "simplify_tolerance_ratio": float(params.simplify_tolerance_ratio),
        "spacing": float(spacing),
        "allow_convex_fallback": False,
    }
    config["config_hash"] = canonical_sha256(config)
    return SupportFit(
        normal=[float(item) for item in normal],
        offset=float(normal @ center),
        origin=[float(item) for item in center],
        basis_u=[float(item) for item in u],
        basis_v=[float(item) for item in v],
        residuals=residuals,
        residual_count=int(len(unique)),
        rmse=rmse,
        max_abs_residual=max_abs,
        plane_tolerance=float(tolerance),
        support_point_count=int(len(points)),
        regions=regions,
        domain=domain.to_dict(),
        spacing=spacing,
        fit_config=config,
    )


def region_metrics(regions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    areas: List[float] = []
    holes = 0
    for region in regions:
        polygon = Polygon(region["exterior"], region.get("holes") or [])
        if polygon.is_empty or not polygon.is_valid:
            raise SupportGeometryError("fitted finite region is invalid", "FINITE_DOMAIN_INVALID")
        areas.append(float(polygon.area))
        holes += len(region.get("holes") or [])
    return {
        "region_count": len(areas),
        "hole_count": holes,
        "region_areas": [float(item) for item in areas],
        "total_area": float(sum(areas)),
    }


def transform_region(region: Mapping[str, Any], source: Mapping[str, Any],
                     target: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-express one 2D region from one fragment frame into another frame."""
    source = normalise_frame(source)
    target = normalise_frame(target)

    def convert(ring: Sequence[Sequence[float]]) -> List[List[float]]:
        u_hat = np.asarray(source["basis_u"], dtype=np.float64)
        v_hat = np.asarray(source["basis_v"], dtype=np.float64)
        origin = np.asarray(source["origin"], dtype=np.float64)
        t_u = np.asarray(target["basis_u"], dtype=np.float64)
        t_v = np.asarray(target["basis_v"], dtype=np.float64)
        t_origin = np.asarray(target["origin"], dtype=np.float64)
        result = []
        for item in ring:
            xyz = origin + float(item[0]) * u_hat + float(item[1]) * v_hat
            delta = xyz - t_origin
            result.append([float(delta @ t_u), float(delta @ t_v)])
        return result

    return {
        "exterior": convert(region["exterior"]),
        "holes": [convert(hole) for hole in region.get("holes") or []],
    }


def coplanarity_report(
    frames: Sequence[Mapping[str, Any]],
    *,
    angle_tolerance_deg: float = DEFAULT_COPLANAR_ANGLE_DEG,
    offset_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    if not frames:
        raise SupportGeometryError("no member frame to compare", "SUPPORT_FRAME_UNAVAILABLE")
    normalised = [normalise_frame(frame) for frame in frames]
    base = normalised[0]
    base_normal = np.asarray(base["normal"], dtype=np.float64)
    base_origin = np.asarray(base["origin"], dtype=np.float64)
    max_angle = 0.0
    max_gap = 0.0
    for frame in normalised[1:]:
        normal = np.asarray(frame["normal"], dtype=np.float64)
        dot = max(-1.0, min(1.0, float(abs(base_normal @ normal))))
        max_angle = max(max_angle, float(math.degrees(math.acos(dot))))
        origin = np.asarray(frame["origin"], dtype=np.float64)
        max_gap = max(max_gap, float(abs((origin - base_origin) @ base_normal)))
    combined = [np.asarray(frame["origin"], dtype=np.float64) for frame in normalised]
    span = float(np.linalg.norm(np.ptp(np.asarray(combined), axis=0)))
    tolerance = float(offset_tolerance) if offset_tolerance is not None else max(1e-6, DEFAULT_COPLANAR_OFFSET_RATIO * max(span, 1e-9))
    ok = max_angle <= float(angle_tolerance_deg) and max_gap <= tolerance
    return {
        "ok": bool(ok),
        "max_angle_deg": max_angle,
        "max_offset_gap": max_gap,
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "offset_tolerance": tolerance,
        "member_count": len(normalised),
        "policy": "COPLANAR_SUPPORT_REQUIRED_NO_CONVEX_FILL",
    }


def fixture_points_for_fragment(fragment: Mapping[str, Any]) -> np.ndarray:
    """Deterministic fixture-only reconstruction from a fragment's own region.

    This is used only when ``EditEngine`` is constructed without a real point
    provider (legacy synthetic contract tests).  Every caller must label the
    resulting geometry ``SYNTHETIC_FIXTURE_RECONSTRUCTION``.  It is never a
    production fallback.
    """
    region = (fragment.get("finite_regions_local") or [None])[0]
    if not region:
        raise SupportGeometryError("fixture fragment has no finite region", "SUPPORT_POINTS_UNAVAILABLE")
    polygon = Polygon(region["exterior"], region.get("holes") or [])
    if polygon.is_empty or polygon.area <= 0:
        raise SupportGeometryError("fixture finite region is empty", "SUPPORT_POINTS_UNAVAILABLE")
    minx, miny, maxx, maxy = polygon.bounds
    side = 16
    candidates = []
    for iy in range(side):
        for ix in range(side):
            x = minx + (maxx - minx) * (ix + 0.5) / side
            y = miny + (maxy - miny) * (iy + 0.5) / side
            if polygon.covers(Point(x, y)):
                candidates.append((x, y))
    if len(candidates) < MIN_SUPPORT_POINTS:
        # A thin polygon can contain fewer grid centres; fall back to a small
        # grid over its exterior bounds (still a declared fixture, not real).
        candidates = [
            (minx + (maxx - minx) * (ix + 0.5) / 4.0,
             miny + (maxy - miny) * (iy + 0.5) / 4.0)
            for iy in range(4) for ix in range(4)
        ]
    normal = np.asarray(fragment["plane_local"]["normal"], dtype=np.float64)
    offset = float(fragment["plane_local"]["offset"])
    u, v = canonical_basis(normal)
    origin = normal * offset
    points = [origin + x * u + y * v for x, y in candidates]
    return np.asarray(points, dtype=np.float64)


__all__ = [
    "DEFAULT_ALPHA_RADIUS_RATIO", "DEFAULT_COPLANAR_ANGLE_DEG", "DEFAULT_COPLANAR_OFFSET_RATIO",
    "DEFAULT_MIN_REGION_AREA_RATIO", "DEFAULT_PLANE_TOLERANCE_RATIO", "MAX_RESIDUALS",
    "MIN_SUPPORT_POINTS", "SupportFit", "SupportGeometryError", "canonical_basis",
    "coplanarity_report", "estimate_spacing", "fit_support_geometry",
    "fixture_points_for_fragment", "frame_of", "normalise_frame", "project_uv",
    "region_metrics", "transform_region",
]
