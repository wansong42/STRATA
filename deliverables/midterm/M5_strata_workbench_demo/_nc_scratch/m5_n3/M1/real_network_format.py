"""M1 Real-Derived Conditioned Whole-Network Format Validator & Kernel Adapter.

Implements Amendment B2 (`00_总控单与架构师裁定_20261120.md`):
- Independent versioned admission validator for `schema = "TC-network/real-conditioned/v1"`
  and `scope = "REAL_DERIVED_CONDITIONED_FINITE_UNION"`.
- Zero modification to historically frozen `TC_network/network_format.py`.
- Permanent separation between real-derived conditioned models and synthetic models
  (`SYNTHETIC_DECLARED_TRIANGLE_UNION`): cross-labeling in either direction is hard-rejected.
- Strict enforcement of:
  1. Source provenance & SHA256 verification (M1-N1)
  2. Real-derived scope non-substitution (M1-N2)
  3. Non-empty `assumption_register` with `ASSUMED_NOT_OBSERVED` and mandatory `not_claimed` (M1-N3)
  4. Same-site coordinate frame & physical world bounding-box guard (M1-N4)
  5. Exact rational hole-preservation & anti-convex-hull-fill topology guard (M1-N5)
  6. Mode (<=16), triangle (<=64), segment (<=2048), and rational bit (<=8192) budgets (M1-N6)
- Lossless in-memory projection `to_core_kernel_payload(model)` for exact solver consumption;
  never persisted to disk as synthetic.

Stdlib only (fractions, hashlib, json, math, pathlib). Zero floating-point final decisions
for geometric degeneracy, segment membership, rational bounds, or hole-probe containment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "TC-network/real-conditioned/v1"
REAL_SCOPE = "REAL_DERIVED_CONDITIONED_FINITE_UNION"
SYNTHETIC_SCOPE = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
CORE_KERNEL_SCHEMA = "TC-network/shared-t/v1"

ALLOWED_REGION_ORIGINS = frozenset({
    "REAL_DECLARED_RESIDUAL_DOMAIN",
    "TB_FEASIBLE_BOUND",
})

ALLOWED_BOUNDARY_MODELS = frozenset({
    "REAL_ALPHA_POLYGON_CONSTRAINED_TRIANGULATION",
    "CONDITIONED_HEX_OCT_DISK_FROM_EXPERT_PLANE",
    "CONDITIONED_FINITE_POLYGON_FROM_EXPERT_RADIUS",
})

MAX_MODES = 16
MAX_TRIANGLES = 64
MAX_SEGMENTS = 2048
MAX_RATIONAL_TEXT = 4096
MAX_RATIONAL_BITS = 8192
LARGE_FILE_BYTES = 100 * 1024 * 1024  # 100 MiB

ZERO = Q(0)
ONE = Q(1)

SITE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "jardim_garcia_uh": {
        "coordinate_frame": "jardim_garcia_quarry_uh_source_coordinates_v1",
        "world_bbox_min": (Q(680), Q(545), Q(620)),
        "world_bbox_max": (Q(721), Q(561), Q(636)),
        "required_source_substrings": ("project.json", "points.ply"),
        "forbidden_source_substrings": (
            "perozzo_2024",
            "Manual-Discontinuities-Geometrical-Info.xlsx",
            "Mapped_Discontinuity_Traces-Planes.dxf",
        ),
    },
    "perozzo_finale_ligure": {
        "coordinate_frame": "perozzo_2024_finale_ligure_las_utm_v1",
        "world_bbox_min": (Q(446790), Q(4890360), Q(45)),
        "world_bbox_max": (Q(447070), Q(4890730), Q(210)),
        "required_source_substrings": (
            "Manual-Discontinuities-Geometrical-Info.xlsx",
            "Mapped_Discontinuity_Traces-Planes.dxf",
        ),
        "forbidden_source_substrings": (
            "w160_full_pointcloud_roi",
            "jardim_garcia",
            "uh_roi_patch",
        ),
    },
}


class Invalid(ValueError):
    """Malformed input, missing mandatory disclosure, SHA mismatch, or topology violation."""


class Outside(ValueError):
    """Scope violation, cross-site frame contamination, or undeclared extension."""


class Exhausted(TimeoutError):
    """Explicit mode, triangle, segment, or rational bit-length budget exceeded."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Invalid(reason)


def exact(x: Any) -> Q:
    require(type(x) in (int, str, float, Q), f"rational required, got {type(x).__name__}")
    if isinstance(x, str):
        if len(x) > MAX_RATIONAL_TEXT:
            raise Exhausted("rational text budget exceeded")
        raw = x.strip()
        require(bool(raw), "empty rational string")
    else:
        raw = x
    try:
        val = Q(raw)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise Invalid(f"invalid finite rational: {x!r}") from exc
    if max(abs(val.numerator).bit_length(), val.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise Exhausted("rational bit budget exceeded (>8192)")
    return val


def qs(x: Any) -> str:
    v = exact(x)
    return f"{v.numerator}/{v.denominator}"


def canonical(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def real_input_sha256(model: Dict[str, Any]) -> str:
    """Canonical SHA256 of the REAL model.json (must be used by M2/M3/M4/M5)."""
    return fingerprint(model)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(value: Any, label: str = "sha256") -> str:
    require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        f"{label}: 64 lowercase hex chars required",
    )
    return value


def vector(x: Any) -> Tuple[Q, Q, Q]:
    require(type(x) is list and len(x) == 3, "three rational coordinates required")
    return (exact(x[0]), exact(x[1]), exact(x[2]))


def sub(a: Tuple[Q, Q, Q], b: Tuple[Q, Q, Q]) -> Tuple[Q, Q, Q]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Tuple[Q, Q, Q], b: Tuple[Q, Q, Q]) -> Tuple[Q, Q, Q]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dot(a: Tuple[Q, Q, Q], b: Tuple[Q, Q, Q]) -> Q:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Tuple[Q, Q, Q], b: Tuple[Q, Q, Q]) -> Tuple[Q, Q, Q]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm2(a: Tuple[Q, Q, Q]) -> Q:
    return dot(a, a)


def point_in_triangle_projected_3d(
    p: Tuple[Q, Q, Q],
    a: Tuple[Q, Q, Q],
    b: Tuple[Q, Q, Q],
    c: Tuple[Q, Q, Q],
    max_plane_dist_sq: Q = Q(1, 10000),  # (0.01 m)^2
) -> bool:
    """Exact rational check: does the orthogonal projection of p onto plane(a,b,c)
    lie on/inside triangle (a,b,c) while p is within sqrt(max_plane_dist_sq) of the plane?"""
    ab = sub(b, a)
    ac = sub(c, a)
    n = cross(ab, ac)
    nn = norm2(n)
    if nn == 0:
        return False
    ap = sub(p, a)
    signed = dot(ap, n)
    # Orthogonal distance squared = signed^2 / nn
    if signed * signed > max_plane_dist_sq * nn:
        return False
    # Project p orthogonally onto plane(a,b,c) in exact rational arithmetic
    scale_t = signed / nn
    p_proj = (p[0] - scale_t * n[0], p[1] - scale_t * n[1], p[2] - scale_t * n[2])
    c0 = dot(cross(sub(b, a), sub(p_proj, a)), n)
    c1 = dot(cross(sub(c, b), sub(p_proj, b)), n)
    c2 = dot(cross(sub(a, c), sub(p_proj, c)), n)
    return c0 >= 0 and c1 >= 0 and c2 >= 0


def _segments(raw: Any, domain: Tuple[Q, Q]) -> List[Tuple[Q, Q, bool, bool]]:
    require(type(raw) is list, "segments list required")
    if len(raw) > MAX_SEGMENTS:
        raise Exhausted("segment budget exceeded")
    result: List[Tuple[Q, Q, bool, bool]] = []
    for item in raw:
        require(
            type(item) is dict and set(item) == {"lo", "hi", "lo_closed", "hi_closed"},
            "segment schema must be {lo, hi, lo_closed, hi_closed}",
        )
        lo, hi = exact(item["lo"]), exact(item["hi"])
        lc, hc = item["lo_closed"], item["hi_closed"]
        require(type(lc) is bool and type(hc) is bool, "explicit bool closures required")
        require(lo <= hi, "inverted segment")
        require(lo < hi or (lc and hc), "empty singleton segment")
        if lo < domain[0] or hi > domain[1]:
            raise Outside("feasible segment outside declared t_closed domain")
        result.append((lo, hi, lc, hc))
    result.sort()
    merged: List[Tuple[Q, Q, bool, bool]] = []
    for lo, hi, lc, hc in result:
        if merged and (lo < merged[-1][1] or (lo == merged[-1][1] and (lc or merged[-1][3]))):
            a, b, ac, bc = merged[-1]
            merged[-1] = (
                a,
                max(b, hi),
                (ac or lc) if a == lo else ac,
                (bc or hc) if b == hi else (hc if hi > b else bc),
            )
        else:
            merged.append((lo, hi, lc, hc))
    return merged


def admitted(t: Q, intervals: Sequence[Tuple[Q, Q, bool, bool]]) -> bool:
    return any(
        (lo < t or (lo == t and lc)) and (t < hi or (t == hi and hc))
        for lo, hi, lc, hc in intervals
    )


def _check_world_bbox(pt_local: Tuple[Q, Q, Q], origin: Tuple[Q, Q, Q], site_id: str, label: str) -> None:
    spec = SITE_REGISTRY[site_id]
    bmin = spec["world_bbox_min"]
    bmax = spec["world_bbox_max"]
    world = add(origin, pt_local)
    for axis, name in enumerate(("X", "Y", "Z")):
        if world[axis] < bmin[axis] or world[axis] > bmax[axis]:
            raise Outside(
                f"CROSS_SITE_COORDINATE_OUT_OF_SITE_BBOX: {label} {name}={float(world[axis]):.4f} "
                f"outside site '{site_id}' range [{float(bmin[axis])}, {float(bmax[axis])}]"
            )


def _find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    base = Path(start).resolve() if start is not None else Path(__file__).resolve().parent
    for cand in (base,) + tuple(base.parents):
        if (cand / "AGENTS.md").is_file():
            return cand
    return None


def _validate_real_provenance(
    prov: Any,
    *,
    repo_root: Optional[Path],
    verify_source_sha: bool,
    hash_large_files: bool,
) -> Tuple[str, Tuple[Q, Q, Q]]:
    require(type(prov) is dict, "real_provenance object required")
    req_keys = {"site_id", "coordinate_frame", "cluster_local_origin", "source_files", "prereg_sha256"}
    opt_keys = {"cluster_id", "stratum", "seed_fracture_id", "built_at_utc", "dxf_binding_summary"}
    missing = req_keys - set(prov)
    require(not missing, f"real_provenance missing required fields: {sorted(missing)}")
    extra = set(prov) - req_keys - opt_keys
    if extra:
        raise Outside(f"undeclared real_provenance fields: {sorted(extra)}")

    site_id = prov["site_id"]
    if type(site_id) is not str or site_id not in SITE_REGISTRY:
        raise Outside(f"unregistered site_id: {site_id!r}")
    spec = SITE_REGISTRY[site_id]

    frame = prov["coordinate_frame"]
    if frame != spec["coordinate_frame"]:
        raise Outside(
            f"CROSS_SITE_FRAME_MISMATCH: site_id='{site_id}' requires frame "
            f"'{spec['coordinate_frame']}', got {frame!r}"
        )

    sha_text(prov["prereg_sha256"], "real_provenance.prereg_sha256")
    origin = vector(prov["cluster_local_origin"])
    _check_world_bbox((ZERO, ZERO, ZERO), origin, site_id, "cluster_local_origin")

    src_files = prov["source_files"]
    require(type(src_files) is list and len(src_files) >= 1, "real_provenance.source_files must be nonempty")
    seen_paths: List[str] = []
    root = _find_repo_root(repo_root) if verify_source_sha else None
    if verify_source_sha and root is None:
        raise Invalid("verify_source_sha=True requested but repo_root could not be resolved")

    for idx, entry in enumerate(src_files):
        require(type(entry) is dict, f"source_files[{idx}] must be dict")
        require({"path", "sha256", "bytes", "role"} <= set(entry), f"source_files[{idx}] missing path/sha256/bytes/role")
        p_str = entry["path"]
        require(type(p_str) is str and bool(p_str.strip()), f"source_files[{idx}].path must be nonempty string")
        norm_p = p_str.replace("\\", "/")
        for bad_sub in spec["forbidden_source_substrings"]:
            if bad_sub.lower() in norm_p.lower():
                raise Outside(
                    f"CROSS_SITE_SOURCE_CONTAMINATION: source '{norm_p}' forbidden for site '{site_id}'"
                )
        expected_sha = sha_text(entry["sha256"], f"source_files[{idx}].sha256")
        expected_bytes = entry["bytes"]
        require(type(expected_bytes) is int and expected_bytes > 0, f"source_files[{idx}].bytes must be positive int")
        seen_paths.append(norm_p)

        if verify_source_sha:
            assert root is not None
            full_path = Path(norm_p) if Path(norm_p).is_absolute() else (root / norm_p)
            require(full_path.is_file(), f"SOURCE_FILE_MISSING: {norm_p}")
            actual_bytes = full_path.stat().st_size
            require(
                actual_bytes == expected_bytes,
                f"SOURCE_BYTES_MISMATCH: {norm_p} expected={expected_bytes} actual={actual_bytes}",
            )
            if actual_bytes <= LARGE_FILE_BYTES or hash_large_files:
                actual_sha = sha_file(full_path)
                require(
                    actual_sha == expected_sha,
                    f"SOURCE_SHA256_MISMATCH: {norm_p} expected={expected_sha} actual={actual_sha}",
                )

    for req_sub in spec["required_source_substrings"]:
        if not any(req_sub in p for p in seen_paths):
            raise Invalid(f"real_provenance.source_files missing required site source containing '{req_sub}'")

    return site_id, origin


def _validate_assumption_register(reg: Any) -> None:
    require(type(reg) is dict, "assumption_register object required")
    req_keys = {
        "boundary_model",
        "assumed_edges_tag",
        "assumed_edges_excluded_from_observation_support",
        "uncertainty_model",
        "not_claimed",
    }
    opt_keys = {"notes", "amendments_applied"}
    missing = req_keys - set(reg)
    require(not missing, f"assumption_register missing required fields: {sorted(missing)}")
    extra = set(reg) - req_keys - opt_keys
    if extra:
        raise Outside(f"undeclared assumption_register fields: {sorted(extra)}")

    require(
        reg["boundary_model"] in ALLOWED_BOUNDARY_MODELS,
        f"assumption_register.boundary_model invalid: {reg['boundary_model']!r}",
    )
    require(
        reg["assumed_edges_tag"] == "ASSUMED_NOT_OBSERVED",
        "assumption_register.assumed_edges_tag must be exact string 'ASSUMED_NOT_OBSERVED'",
    )
    require(
        reg["assumed_edges_excluded_from_observation_support"] is True,
        "assumption_register.assumed_edges_excluded_from_observation_support must be true",
    )

    um = reg["uncertainty_model"]
    require(type(um) is dict, "assumption_register.uncertainty_model must be dict")
    um_req = {
        "parameter_t_meaning",
        "velocity_formula",
        "design_coefficient_k",
        "residual_source",
        "discrete_modes_mechanism",
    }
    require(um_req <= set(um), f"uncertainty_model missing fields: {sorted(um_req - set(um))}")
    for key in um_req:
        val = um[key]
        require(type(val) is str and bool(val.strip()), f"uncertainty_model.{key} must be nonempty string")
    exact(um["design_coefficient_k"])

    nc = reg["not_claimed"]
    require(
        type(nc) is list and len(nc) >= 2 and all(type(s) is str and bool(s.strip()) for s in nc),
        "assumption_register.not_claimed must be list of >=2 nonempty disclaimer strings",
    )
    joined = " ".join(nc).lower()
    has_subsurface = any(tok in joined for tok in ("地下体内", "subsurface", "unconditional", "无条件真值"))
    has_hydraulic = any(tok in joined for tok in ("水力", "hydraulic"))
    require(
        has_subsurface,
        "assumption_register.not_claimed must explicitly disclaim unconditional subsurface/in-rock ground truth",
    )
    require(
        has_hydraulic,
        "assumption_register.not_claimed must explicitly disclaim hydraulic connectivity",
    )


def _validate_quantization_contract(
    qc: Any,
    m0_triangles: List[Dict[str, Any]],
    fracture_catalog: Optional[Dict[str, Any]],
) -> None:
    require(type(qc) is dict, "quantization_contract object required")
    req_keys = {
        "schema",
        "tier1_exact_source_lift",
        "tier2_rational_grid",
        "tier3_geometry_approximation",
        "min_nominal_pair_gap_m",
        "quantization_sensitive",
        "hole_preservation_audit",
    }
    missing = req_keys - set(qc)
    require(not missing, f"quantization_contract missing fields: {sorted(missing)}")
    require(qc["schema"] == "m1-quantization-contract/v1", "quantization_contract.schema invalid")
    require(type(qc["tier1_exact_source_lift"]) is dict and bool(qc["tier1_exact_source_lift"]), "tier1 required")

    t2 = qc["tier2_rational_grid"]
    require(type(t2) is dict, "tier2_rational_grid must be dict")
    require({"grid_step_m", "per_axis_bound_m", "max_euclidean_bound_m"} <= set(t2), "tier2 fields missing")
    step = exact(t2["grid_step_m"])
    per_axis = exact(t2["per_axis_bound_m"])
    require(step > 0 and per_axis == step / 2, "tier2 per_axis_bound_m must equal grid_step_m / 2")
    max_euclid = float(t2["max_euclidean_bound_m"])
    require(math.isfinite(max_euclid) and max_euclid >= 0.0, "invalid tier2 max_euclidean_bound_m")

    t3 = qc["tier3_geometry_approximation"]
    require(type(t3) is dict and bool(t3), "tier3_geometry_approximation required")
    require(
        {"method", "max_hausdorff_deviation_m", "max_normal_angle_deviation_deg"} <= set(t3),
        "tier3 missing method/max_hausdorff_deviation_m/max_normal_angle_deviation_deg",
    )

    min_gap = qc["min_nominal_pair_gap_m"]
    qsens = qc["quantization_sensitive"]
    require(type(qsens) is bool, "quantization_sensitive must be bool")
    if min_gap is not None:
        require(type(min_gap) in (int, float) and math.isfinite(min_gap) and min_gap >= 0.0, "invalid min_nominal_pair_gap_m")
        if min_gap <= 2.0 * max_euclid and not qsens:
            raise Invalid(
                f"quantization_sensitive must be true when min_nominal_pair_gap_m ({min_gap}) "
                f"<= 2 * tier2_max_euclidean_bound_m ({2.0 * max_euclid})"
            )

    # Exact Hole-Preservation & Anti-Convex-Hull-Fill Topology Guard (M1-N5)
    hole_audit = qc["hole_preservation_audit"]
    require(type(hole_audit) is list, "hole_preservation_audit must be list")
    audited_ids = set()
    tris_by_comp: Dict[str, List[Tuple[Tuple[Q, Q, Q], Tuple[Q, Q, Q], Tuple[Q, Q, Q]]]] = {}
    for tri in m0_triangles:
        pts = tuple(vector(p) for p in tri["vertices"])
        tris_by_comp.setdefault(tri["component"], []).append(pts)  # type: ignore[arg-type]

    for entry in hole_audit:
        require(type(entry) is dict, "hole_preservation_audit entry must be dict")
        hreq = {
            "fragment_id",
            "original_hole_count",
            "preserved_hole_count",
            "original_net_area_m2",
            "original_convex_hull_area_m2",
            "preserved_hole_area_m2",
            "triangulated_net_area_m2",
            "hole_interior_probe_points_local",
        }
        require(hreq <= set(entry), f"hole_preservation_audit missing fields: {sorted(hreq - set(entry))}")
        fid = entry["fragment_id"]
        audited_ids.add(fid)
        require(entry["original_hole_count"] >= 1, f"hole_audit {fid}: original_hole_count must be >=1")
        require(
            entry["preserved_hole_count"] >= 1,
            f"HOLE_FILLED_TOPOLOGY_VIOLATION: {fid} has preserved_hole_count < 1",
        )
        probes = entry["hole_interior_probe_points_local"]
        require(type(probes) is list and len(probes) >= 1, f"hole_audit {fid}: nonempty hole probes required")

        comp_tris = tris_by_comp.get(fid, [])
        require(bool(comp_tris), f"hole_audit {fid}: fragment triangles missing from m0")

        # 1. Exact triangle area verification vs convex hull / no-hole fill
        actual_area = 0.0
        for a, b, c in comp_tris:
            n = cross(sub(b, a), sub(c, a))
            actual_area += 0.5 * math.sqrt(float(norm2(n)))

        hull_area = float(entry["original_convex_hull_area_m2"])
        hole_area = float(entry["preserved_hole_area_m2"])
        declared_area = float(entry["triangulated_net_area_m2"])
        require(hole_area > 0.0, f"hole_audit {fid}: preserved_hole_area_m2 must be > 0")
        if abs(actual_area - declared_area) > max(0.02, 0.05 * declared_area):
            raise Invalid(
                f"HOLE_AREA_MISMATCH: {fid} actual m0 triangle area={actual_area:.5f} "
                f"differs from declared triangulated_net_area_m2={declared_area:.5f}"
            )
        if actual_area >= hull_area - 0.5 * hole_area:
            raise Invalid(
                f"HOLE_FILLED_AREA_GUARD: {fid} actual m0 area={actual_area:.5f} approaches/exceeds "
                f"convex_hull_area={hull_area:.5f} (hole_area={hole_area:.5f})"
            )

        # 2. Exact rational geometric probe check: NO triangle of fid may cover a real hole interior probe!
        for p_idx, raw_probe in enumerate(probes):
            probe = vector(raw_probe)
            for t_idx, (a, b, c) in enumerate(comp_tris):
                if point_in_triangle_projected_3d(probe, a, b, c):
                    raise Invalid(
                        f"HOLE_FILLED_PROBE_COVERED: {fid} triangle[{t_idx}] covers real hole "
                        f"interior probe[{p_idx}] (convex-hull or hole-fill detected)"
                    )

    if fracture_catalog is not None:
        for fid, meta in fracture_catalog.items():
            if meta.get("original_hole_count", 0) > 0:
                require(
                    fid in audited_ids,
                    f"HOLE_AUDIT_MISSING: holed fragment {fid} missing from hole_preservation_audit",
                )


def _triangle(raw: Any) -> Tuple[Tuple[Tuple[Q, Q, Q], Tuple[Q, Q, Q], Tuple[Q, Q, Q]], Tuple[Q, Q, Q]]:
    require(type(raw) is dict, "triangle object required")
    if set(raw) - {"id", "component", "vertices", "v", "label"}:
        raise Outside("undeclared triangle geometry or moving normal")
    require(type(raw.get("id")) is str and bool(raw["id"]) and not raw["id"].startswith("@"), "triangle id")
    require(type(raw.get("component")) is str and bool(raw["component"]), "rigid component required")
    require(type(raw.get("vertices")) is list and len(raw["vertices"]) == 3, "closed triangle required")
    points = (vector(raw["vertices"][0]), vector(raw["vertices"][1]), vector(raw["vertices"][2]))
    velocity = vector(raw.get("v"))
    require(any(cross(sub(points[1], points[0]), sub(points[2], points[0]))), f"degenerate triangle: {raw['id']}")
    return points, velocity


def validate(
    model: Any,
    *,
    repo_root: Optional[Path] = None,
    verify_source_sha: bool = False,
    hash_large_files: bool = False,
    require_non_static: bool = True,
) -> Dict[str, Any]:
    """Validate a M1 real-derived conditioned multi-fracture cluster model.

    Raises Invalid / Outside / Exhausted on any violation.
    Returns summary dict with validated budget, site_id, counts, and canonical sha256.
    """
    require(type(model) is dict, "network model must be JSON object")
    allowed = {
        "schema",
        "snapshot_id",
        "scope",
        "union_complete",
        "region_origin",
        "region_provenance",
        "endpoints",
        "modes",
        "reference",
        "budget",
        "real_provenance",
        "assumption_register",
        "quantization_contract",
        "fracture_catalog",
        "mode_semantics",
    }
    extra_top = set(model) - allowed
    if extra_top:
        raise Outside(f"undeclared top-level fields: {sorted(extra_top)}")

    require(model.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}, got {model.get('schema')!r}")
    require(type(model.get("snapshot_id")) is str and bool(model["snapshot_id"]), "nonempty snapshot_id required")

    scope = model.get("scope")
    if scope == SYNTHETIC_SCOPE:
        raise Outside(
            "REAL_SCOPE_VIOLATION: SYNTHETIC_DECLARED_TRIANGLE_UNION is strictly forbidden "
            "for real-derived conditioned models (permanent track separation)"
        )
    if scope != REAL_SCOPE or model.get("union_complete") is not True:
        raise Outside(f"only {REAL_SCOPE!r} with union_complete=true admitted")

    origin_kind = model.get("region_origin")
    if origin_kind not in ALLOWED_REGION_ORIGINS:
        raise Outside(f"region_origin must be in {sorted(ALLOWED_REGION_ORIGINS)}, got {origin_kind!r}")
    if origin_kind == "REAL_DECLARED_RESIDUAL_DOMAIN":
        require("region_provenance" not in model, "REAL_DECLARED_RESIDUAL_DOMAIN must not carry TB region_provenance")

    site_id, local_origin = _validate_real_provenance(
        model.get("real_provenance"),
        repo_root=repo_root,
        verify_source_sha=verify_source_sha,
        hash_large_files=hash_large_files,
    )
    _validate_assumption_register(model.get("assumption_register"))

    endpoints = model.get("endpoints")
    require(type(endpoints) is dict and set(endpoints) == {"A", "B"}, "endpoints must be exact keys {A, B}")
    for ep_name, ep_pt in endpoints.items():
        pt_q = vector(ep_pt)
        _check_world_bbox(pt_q, local_origin, site_id, f"endpoint.{ep_name}")

    modes = model.get("modes")
    require(type(modes) is list and len(modes) >= 1, "nonempty modes list required")
    if len(modes) > MAX_MODES:
        raise Exhausted(f"mode budget exceeded: {len(modes)} > {MAX_MODES}")

    mode_names = set()
    all_components = set()
    any_nonzero_velocity = False
    m0_triangles: Optional[List[Dict[str, Any]]] = None
    max_tri_count = 0

    for mode in modes:
        require(
            type(mode) is dict and set(mode) == {"id", "t_closed", "triangles", "segments", "edit_cost"},
            "each mode must have exact keys {id, t_closed, triangles, segments, edit_cost}",
        )
        mid = mode["id"]
        require(type(mid) is str and bool(mid) and mid not in mode_names, f"duplicate/invalid mode id: {mid!r}")
        mode_names.add(mid)
        require(exact(mode["edit_cost"]) >= 0, f"mode {mid}: nonnegative edit_cost required")
        require(type(mode["t_closed"]) is list and len(mode["t_closed"]) == 2, f"mode {mid}: bounded closed domain")
        domain = (exact(mode["t_closed"][0]), exact(mode["t_closed"][1]))
        require(domain[0] <= domain[1], f"mode {mid}: inverted t_closed domain")
        _segments(mode["segments"], domain)

        tris = mode["triangles"]
        require(type(tris) is list and len(tris) >= 1, f"mode {mid}: nonempty triangle list required")
        if len(tris) > MAX_TRIANGLES:
            raise Exhausted(f"triangle budget exceeded in mode {mid}: {len(tris)} > {MAX_TRIANGLES}")
        max_tri_count = max(max_tri_count, len(tris))
        if mid == "m0":
            m0_triangles = tris

        tri_ids = set()
        comp_vel: Dict[str, Tuple[Q, Q, Q]] = {}
        for raw_tri in tris:
            pts, vel = _triangle(raw_tri)
            tid = raw_tri["id"]
            comp = raw_tri["component"]
            require(tid not in tri_ids, f"mode {mid}: duplicate triangle id {tid}")
            tri_ids.add(tid)
            all_components.add(comp)
            if vel != (ZERO, ZERO, ZERO):
                any_nonzero_velocity = True
            require(
                comp not in comp_vel or comp_vel[comp] == vel,
                f"mode {mid}: rigid component {comp} has inconsistent velocities",
            )
            comp_vel[comp] = vel
            for v_idx, pt_q in enumerate(pts):
                _check_world_bbox(pt_q, local_origin, site_id, f"{mid}.{tid}.v{v_idx}")

    require(m0_triangles is not None, "nominal mode 'm0' is required")
    if require_non_static:
        require(any_nonzero_velocity, "non-static requirement violated: all fracture velocities are zero")
        require(len(modes) >= 2, "non-static requirement violated: at least 2 explicit modes required")

    budget = model.get("budget", {"cells": 4096, "pairs": 65536})
    require(type(budget) is dict and set(budget) == {"cells", "pairs"}, "budget must have exact keys {cells, pairs}")
    require(all(type(v) is int and 0 <= v <= 1000000 for v in budget.values()), "invalid solver budget")

    ref = model.get("reference")
    require(ref is not None and type(ref) is dict and set(ref) == {"mode", "t", "sigma"}, "reference {mode,t,sigma} required")
    require(exact(ref["sigma"]) > 0, "reference.sigma must be positive")
    ref_mode = next((m for m in modes if m["id"] == ref["mode"]), None)
    if ref_mode is None or exact(ref_mode["edit_cost"]) != 0:
        raise Outside("reference mode missing or has nonzero edit_cost")
    ref_domain = (exact(ref_mode["t_closed"][0]), exact(ref_mode["t_closed"][1]))
    if not admitted(exact(ref["t"]), _segments(ref_mode["segments"], ref_domain)):
        raise Outside("reference t is outside feasible segments of reference mode")

    fcat = model.get("fracture_catalog")
    if fcat is not None:
        require(type(fcat) is dict and set(fcat) == all_components, "fracture_catalog keys must match triangle components")

    msem = model.get("mode_semantics")
    if msem is not None:
        require(type(msem) is dict and set(msem) == mode_names, "mode_semantics keys must match declared mode ids")
        for mid, sem in msem.items():
            require(
                type(sem) is dict and "feature_assignments" in sem and type(sem["feature_assignments"]) is dict,
                f"mode_semantics[{mid}] must contain feature_assignments dict",
            )

    _validate_quantization_contract(model.get("quantization_contract"), m0_triangles, fcat)

    return {
        "ok": True,
        "schema": SCHEMA,
        "scope": REAL_SCOPE,
        "site_id": site_id,
        "snapshot_id": model["snapshot_id"],
        "input_sha256": fingerprint(model),
        "mode_count": len(modes),
        "max_triangles": max_tri_count,
        "fracture_count": len(all_components),
        "budget": budget,
    }


def to_core_kernel_payload(
    model: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
    verify_source_sha: bool = False,
) -> Dict[str, Any]:
    """Lossless IN-MEMORY projection of a validated real-conditioned model onto
    the frozen `TC-network/shared-t/v1` exact solver kernel input contract.

    WARNING: Returned object is strictly for in-memory consumption by the exact
    linear-translation kernel (`TC_network/compile_network.py`). It MUST NEVER be
    persisted to disk as a real-derived deliverable or mislabelled as synthetic.
    """
    validate(model, repo_root=repo_root, verify_source_sha=verify_source_sha)
    core_origin = (
        "DECLARED_DEVELOPMENT_DOMAIN"
        if model["region_origin"] == "REAL_DECLARED_RESIDUAL_DOMAIN"
        else model["region_origin"]
    )
    payload: Dict[str, Any] = {
        "schema": CORE_KERNEL_SCHEMA,
        "snapshot_id": model["snapshot_id"],
        "scope": SYNTHETIC_SCOPE,
        "union_complete": True,
        "region_origin": core_origin,
        "endpoints": copy.deepcopy(model["endpoints"]),
        "modes": copy.deepcopy(model["modes"]),
        "reference": copy.deepcopy(model["reference"]),
        "budget": copy.deepcopy(model.get("budget", {"cells": 4096, "pairs": 65536})),
    }
    if "region_provenance" in model:
        payload["region_provenance"] = copy.deepcopy(model["region_provenance"])
    return payload


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a TC-network/real-conditioned/v1 model JSON.")
    parser.add_argument("--validate", required=True, help="Path to model.json")
    parser.add_argument("--repo-root", default=None, help="Repository root for source SHA verification")
    parser.add_argument("--verify-source-sha", action="store_true", help="Verify on-disk SHA256 of source files")
    parser.add_argument("--hash-large-files", action="store_true", help="Also full-hash >100 MiB source files")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(Path(args.validate).read_text(encoding="utf-8"))
        summary = validate(
            raw,
            repo_root=Path(args.repo_root) if args.repo_root else None,
            verify_source_sha=args.verify_source_sha,
            hash_large_files=args.hash_large_files,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except Exhausted as exc:
        print(json.dumps({"ok": False, "error_class": "Exhausted", "reason": str(exc)}, ensure_ascii=False))
        return 4
    except Outside as exc:
        print(json.dumps({"ok": False, "error_class": "Outside", "reason": str(exc)}, ensure_ascii=False))
        return 3
    except (Invalid, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error_class": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())

