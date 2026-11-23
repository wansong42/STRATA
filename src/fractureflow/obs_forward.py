# -*- coding: utf-8 -*-
"""DFN observation geometry operators (R158, G66 noise-order repair).

Three operators that transform a DFN (set of oriented discs) into synthetic
observations with per-face ground-truth labels.

1. ``virtual_borehole``   -- line x disc intersection -> per-fracture logs
2. ``virtual_fmi``        -- disc x cylinder intersection -> sinusoidal traces
3. ``virtual_outcrop_scan`` -- plane intersection + point cloud + roughness noise

All pure numpy geometry.  Zero install, zero training, zero mechanics.

G66 noise-order contract (P1-L)
-------------------------------
Every operator follows one path:

1. geometric ``obs_mask`` (no RNG involved);
2. public observation order (depth / chord-position ascending, public fields only);
3. **one** noise draw over the observed normals in public order
   (:func:`_noise_observed`, ``n_observed`` rows only -- never ``M`` rows);
4. every record angle (``dip``/``dipdir``/``terzaghi_angle`` and FMI
   ``amplitude``/``phase_deg``) is derived from that same noisy normal;
5. positions/depths stay geometric.

Because the RNG only ever sees the observed set in public order, adding or
permuting hidden objects cannot change the public numbers.

Legacy ``Observation`` still carries hidden-truth fields (``frac_ids``,
``truth_sets``, unobserved arrays, ``provenance['dfn_seed']``, record
``frac_id``) for consumer compatibility.  They are labelled HIDDEN_TRUTH and
never enter :mod:`fractureflow.public_payload`.

Consumes (read-only):
    fractureflow.dfn.DFNRealization  -- centers / normals / radii / sets / domain

Produces:
    Observation -- unified observation package with obs_mask, obs_normals,
    obs_positions, frac_ids, truth_sets, provenance, operator records.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .geometry import dip_dipdir_to_normal as _central_dip_dipdir_to_normal
from .geometry import normal_to_dip_dipdir as _central_normal_to_dip_dipdir


_NOISE_MODEL = "isotropic_tangent_gaussian_v1"
_OBS_VERSION = "G66-v1.1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    """Normalize to unit length without systematic shrinkage for unit inputs."""
    v = np.asarray(v, dtype=np.float64)
    nl = np.linalg.norm(v, axis=axis, keepdims=True)
    # Only add epsilon when norm is near zero (avoid divide-by-zero);
    # adding it unconditionally would shrink unit-length inputs by ~1e-12.
    nl = np.where(nl < 1e-12, 1e-12, nl)
    return v / nl


def _sign_align(pts):
    """Sign-align unoriented normals to a common hemisphere.

    [M, 3] -> [M, 3], aligned to the first vector.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def normal_to_dip_dipdir(n):
    """Convert a unit normal to (dip_deg, dipdir_deg).

    Delegates to the centralized frozen convention in geometry.py
    (single source of truth; unsigned-safe: n and -n yield one pair).
    Exactly-axial inputs are snapped to dip 0/180 with dipdir 0:
    geometry's unconditional +1e-12 normalizer otherwise returns
    dip ~8.1e-5 deg for a perfectly horizontal plane.
    """
    n = _unit(np.asarray(n, dtype=np.float64).reshape(-1))
    if np.hypot(n[0], n[1]) < 1e-9:
        return (0.0, 0.0) if n[2] >= 0 else (180.0, 0.0)
    dip, dipdir = _central_normal_to_dip_dipdir(n)
    return float(dip), float(dipdir)


def dip_dipdir_to_normal(dip_deg, dipdir_deg):
    """Inverse of normal_to_dip_dipdir (centralized in geometry.py)."""
    return _central_dip_dipdir_to_normal(dip_deg, dipdir_deg)


def _apply_normal_noise(normals, roughness_deg, rng):
    """Add isotropic angular noise to an array of unit normals.

    Parameters
    ----------
    normals : (M, 3) array of unit normals
    roughness_deg : float -- std of angular deviation in degrees (0 = clean)
    rng : np.random.Generator

    Returns
    -------
    (M, 3) noisy unit normals (or copy if roughness_deg <= 0).
    """
    normals = np.asarray(normals, dtype=np.float64).copy()
    if roughness_deg <= 0:
        return normals
    M = normals.shape[0]
    sigma = np.radians(roughness_deg)
    n = _unit(normals)
    # Build two tangent vectors per normal
    # Choose a helper vector not parallel to n
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (M, 1))
    mask = np.abs(n[:, 2]) > 0.9
    helper[mask] = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u = _unit(u)
    v = np.cross(n, u)
    v = _unit(v)
    dx = rng.normal(0, sigma, size=(M, 1))
    dy = rng.normal(0, sigma, size=(M, 1))
    n_noisy = n + dx * u + dy * v
    return _unit(n_noisy)


def _new_noise_rng(seed):
    return np.random.default_rng(int(seed))


def _noise_observed(normals, roughness_deg, rng):
    """Single post-observation noise path shared by all three operators.

    ``normals`` must already be the observed rows in public order.  The RNG
    never sees hidden/unobserved rows, so the draw is invariant to the hidden
    world size and to hidden-ID permutation.
    """
    arr = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    return _apply_normal_noise(_unit(arr), float(roughness_deg), rng)


def _rp(x, ndigits):
    """Round exactly like the public record does (stable public sort keys)."""
    v = round(float(x), ndigits)
    return 0.0 if v == 0.0 else v


# ---------------------------------------------------------------------------
# data structure
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Unified observation package produced by all three operators.

    Attributes
    ----------
    operator : str -- "virtual_borehole" / "virtual_fmi" / "virtual_outcrop_scan"
    obs_mask : (M,) bool -- True for observed fractures in the original DFN
    obs_normals : (M, 3) -- observed rows = post-noise normals; unobserved rows
        retain the true normal.  The unobserved rows are HIDDEN_TRUTH and are
        excluded from ``public_payload``.
    obs_positions : (M, 3) -- observation positions (intersection points,
        trace centers, or scan-plane chord centers); unobserved rows = disc
        centres (HIDDEN_TRUTH).
    frac_ids : (n_obs,) int -- indices of observed fractures (HIDDEN_TRUTH:
        indices into the hidden object list)
    truth_sets : (M,) int -- true set assignment from dfn.sets (HIDDEN_TRUTH)
    roughness_deg : float
    provenance : dict -- {operator, public params, noise_seed[, dfn_seed],
        n_observed, n_total, version}.  ``dfn_seed`` is HIDDEN_TRUTH.
    records : list -- operator-specific public log entries; each keeps a
        hidden ``frac_id`` used only for matching (stripped by public_payload)
    point_normals : list | None -- outcrop only: per-fracture post-noise point
        normals in public record order.  PUBLIC observed point cloud
        (sign-aligned mean equals obs_normals for that fracture).
    """
    operator: str
    obs_mask: np.ndarray
    obs_normals: np.ndarray
    obs_positions: np.ndarray
    frac_ids: np.ndarray
    truth_sets: np.ndarray
    roughness_deg: float
    provenance: dict
    records: list = field(default_factory=list)
    point_normals: Optional[list] = None


# ---------------------------------------------------------------------------
# 1. virtual_borehole -- line x disc intersection
# ---------------------------------------------------------------------------

def virtual_borehole(x0, d, dfn, roughness_deg=0.0, seed=0):
    """Simulate borehole logging: line intersects DFN discs.

    Parameters
    ----------
    x0 : (3,) borehole entry point
    d : (3,) borehole direction (will be normalized)
    dfn : DFNRealization
    roughness_deg : float -- angular noise on observed normals
    seed : int -- RNG seed for noise

    Returns
    -------
    Observation
        records: list of dict per intersected fracture, sorted by depth, keys
            depth, frac_id, dip, dipdir, terzaghi_angle.  dip/dipdir and
            terzaghi_angle are derived from the post-noise normal.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    d = _unit(np.asarray(d, dtype=np.float64))
    M = dfn.normals.shape[0]

    obs_mask = np.zeros(M, dtype=bool)
    obs_normals = dfn.normals.copy()
    obs_positions = dfn.centers.copy().astype(np.float64)
    truth_sets = dfn.sets.copy()

    hits = []  # (depth, row index, intersection point)
    for i in range(M):
        n = dfn.normals[i]
        C = dfn.centers[i]
        R = dfn.radii[i]

        nd = float(n @ d)
        if abs(nd) < 1e-12:
            # Disc parallel to borehole -- no intersection (or tangent)
            continue

        t = float(n @ (C - x0) / nd)
        if t < 0:
            # Intersection behind entry point
            continue

        P = x0 + t * d  # intersection point
        dist = np.linalg.norm(P - C)
        if dist > R:
            # Intersection outside disc
            continue

        obs_mask[i] = True
        obs_positions[i] = P
        hits.append((t, i, P))

    # Public order: depth first, then public position (hidden-ID free).
    hits.sort(key=lambda h: (round(h[0], 4), _rp(h[2][0], 12),
                             _rp(h[2][1], 12), _rp(h[2][2], 12)))
    observed_idx = [h[1] for h in hits]

    rng = _new_noise_rng(seed)
    noisy = _noise_observed(dfn.normals[observed_idx] if observed_idx
                            else np.empty((0, 3)), roughness_deg, rng)
    if observed_idx:
        obs_normals[observed_idx] = noisy

    items = []
    for k, (t, i, P) in enumerate(hits):
        n = noisy[k]
        dip, dipdir = normal_to_dip_dipdir(n)
        terzaghi_angle = float(np.degrees(np.arccos(np.clip(abs(float(n @ d)), 0, 1))))
        rec = {
            "depth": round(t, 4),
            "frac_id": int(i),
            "dip": round(dip, 2),
            "dipdir": round(dipdir, 2),
            "terzaghi_angle": round(terzaghi_angle, 2),
        }
        items.append(((rec["depth"], _rp(P[0], 12), _rp(P[1], 12), _rp(P[2], 12),
                       rec["dip"], rec["dipdir"], rec["terzaghi_angle"]), rec))
    items.sort(key=lambda kv: kv[0])
    records = [rec for _, rec in items]

    frac_ids = np.where(obs_mask)[0]

    provenance = {
        "operator": "virtual_borehole",
        "dfn_seed": getattr(dfn, "_seed", None),
        "x0": list(x0),
        "d": list(d),
        "roughness_deg": roughness_deg,
        "noise_seed": seed,
        "noise_model": _NOISE_MODEL,
        "public_order": "depth_ascending_then_position",
        "n_observed": int(obs_mask.sum()),
        "n_total": int(M),
        "version": _OBS_VERSION,
    }

    return Observation(
        operator="virtual_borehole",
        obs_mask=obs_mask,
        obs_normals=obs_normals,
        obs_positions=obs_positions,
        frac_ids=frac_ids,
        truth_sets=truth_sets,
        roughness_deg=roughness_deg,
        provenance=provenance,
        records=records,
    )


def virtual_borehole_multi(wells, dfn, roughness_deg=0.0, seed=0):
    """Multi-well borehole: combine observations from several wells.

    Wrapper over :func:`virtual_borehole` (same G66 noise path per well).
    Not covered by ``public_payload`` v1 (several records per object).

    Parameters
    ----------
    wells : list of (x0, d) tuples
    dfn : DFNRealization
    roughness_deg : float
    seed : int

    Returns
    -------
    Observation (merged: a fracture observed by multiple wells is marked once)
    """
    M = dfn.normals.shape[0]
    obs_mask = np.zeros(M, dtype=bool)
    obs_normals = dfn.normals.copy()
    obs_positions = dfn.centers.copy().astype(np.float64)
    truth_sets = dfn.sets.copy()
    all_records = []

    for wi, (x0, d) in enumerate(wells):
        obs = virtual_borehole(x0, d, dfn, roughness_deg, seed + wi)
        # Merge: take first observation if multiple wells hit same fracture
        newly = obs.obs_mask & ~obs_mask
        obs_mask |= obs.obs_mask
        obs_normals[newly] = obs.obs_normals[newly]
        obs_positions[newly] = obs.obs_positions[newly]
        for rec in obs.records:
            rec["well_id"] = wi
            all_records.append(rec)

    # Public-ish ordering across wells: well id (public) then depth.
    all_records.sort(key=lambda r: (int(r.get("well_id", 0)), r["depth"],
                                    r["dip"], r["dipdir"]))
    frac_ids = np.where(obs_mask)[0]
    provenance = {
        "operator": "virtual_borehole_multi",
        "n_wells": len(wells),
        "wells": [{"x0": list(x0), "d": list(d)} for x0, d in wells],
        "roughness_deg": roughness_deg,
        "noise_seed": seed,
        "noise_model": _NOISE_MODEL,
        "n_observed": int(obs_mask.sum()),
        "n_total": int(M),
        "version": _OBS_VERSION,
    }
    return Observation(
        operator="virtual_borehole",
        obs_mask=obs_mask,
        obs_normals=obs_normals,
        obs_positions=obs_positions,
        frac_ids=frac_ids,
        truth_sets=truth_sets,
        roughness_deg=roughness_deg,
        provenance=provenance,
        records=all_records,
    )


# ---------------------------------------------------------------------------
# 2. virtual_fmi -- disc x cylinder intersection -> sinusoidal trace
# ---------------------------------------------------------------------------

def virtual_fmi(dfn, well_center, well_axis, well_radius,
                roughness_deg=0.0, seed=0):
    """Simulate FMI (Fullbore Micro Imager): disc x cylinder intersection.

    For a vertical well (well_axis = [0,0,1]) through well_center, each
    intersecting disc produces a sinusoidal trace on the unfolded borehole
    wall with parameters (amplitude, phase, depth_center).

    G66: amplitude/phase_deg are derived from the post-noise normal; the trace
    centre depth stays geometric.

    Parameters
    ----------
    dfn : DFNRealization
    well_center : (3,) point on the well axis
    well_axis : (3,) well direction (normalized)
    well_radius : float -- borehole radius (m)
    roughness_deg : float
    seed : int

    Returns
    -------
    Observation
        records: list of dict per intersected fracture, sorted by trace-centre
            depth, keys frac_id, amplitude, phase_deg, depth_center
    """
    well_center = np.asarray(well_center, dtype=np.float64)
    well_axis = _unit(np.asarray(well_axis, dtype=np.float64))
    rw = float(well_radius)
    M = dfn.normals.shape[0]

    ax = well_axis
    C_perp = dfn.centers - well_center
    C_perp = C_perp - np.outer(C_perp @ ax, ax)  # remove axial component
    dist_to_axis = np.linalg.norm(C_perp, axis=1)  # (M,)

    n_axial = dfn.normals @ ax  # (M,)

    obs_mask = np.zeros(M, dtype=bool)
    obs_normals = dfn.normals.copy()
    obs_positions = dfn.centers.copy().astype(np.float64)
    truth_sets = dfn.sets.copy()

    hits = []  # (depth_along, row index, trace centre)
    for i in range(M):
        R = dfn.radii[i]
        if dist_to_axis[i] > R + rw:
            continue
        if abs(n_axial[i]) < 1e-10:
            continue
        n = dfn.normals[i]
        C = dfn.centers[i]
        depth_along = float(np.dot(n, C - well_center) / np.dot(n, ax))
        trace_center = well_center + depth_along * ax
        if dfn.domain is not None:
            Lx, Ly, Lz = dfn.domain
            if not (-Lx / 2 - 0.5 <= trace_center[0] <= Lx / 2 + 0.5 and
                    -Ly / 2 - 0.5 <= trace_center[1] <= Ly / 2 + 0.5 and
                    -Lz / 2 - 0.5 <= trace_center[2] <= Lz / 2 + 0.5):
                continue

        obs_mask[i] = True
        obs_positions[i] = trace_center
        hits.append((depth_along, i, trace_center))

    hits.sort(key=lambda h: (round(h[0], 4), _rp(h[2][0], 12),
                             _rp(h[2][1], 12), _rp(h[2][2], 12)))
    observed_idx = [h[1] for h in hits]

    rng = _new_noise_rng(seed)
    noisy = _noise_observed(dfn.normals[observed_idx] if observed_idx
                            else np.empty((0, 3)), roughness_deg, rng)
    if observed_idx:
        obs_normals[observed_idx] = noisy

    # Public well-perpendicular frame (depends only on the public well axis).
    if abs(ax[2]) < 0.9:
        e1 = _unit(np.cross(ax, [0, 0, 1]))
    else:
        e1 = _unit(np.cross(ax, [1, 0, 0]))
    e2 = _unit(np.cross(ax, e1))

    items = []
    for k, (depth_along, i, trace_center) in enumerate(hits):
        n = noisy[k]
        n_ax = float(n @ ax)
        n_perp_vec = n - n_ax * ax
        n_perp_mag = float(np.sqrt(1.0 - n_ax ** 2 + 1e-30))
        denom = max(abs(n_ax), 1e-12)
        amplitude = rw * n_perp_mag / denom
        if np.linalg.norm(n_perp_vec) > 1e-10:
            phase_rad = np.arctan2(float(n_perp_vec @ e2),
                                   float(n_perp_vec @ e1))
        else:
            phase_rad = 0.0
        phase_deg = float(np.degrees(phase_rad))
        if phase_deg < 0:
            phase_deg += 360.0
        rec = {
            "frac_id": int(i),
            "amplitude": round(float(amplitude), 4),
            "phase_deg": round(float(phase_deg), 2),
            "depth_center": round(float(depth_along), 4),
        }
        items.append(((rec["depth_center"], _rp(trace_center[0], 12),
                       _rp(trace_center[1], 12), _rp(trace_center[2], 12),
                       rec["amplitude"], rec["phase_deg"]), rec))
    items.sort(key=lambda kv: kv[0])
    records = [rec for _, rec in items]

    frac_ids = np.where(obs_mask)[0]

    provenance = {
        "operator": "virtual_fmi",
        "well_center": list(well_center),
        "well_axis": list(well_axis),
        "well_radius": rw,
        "roughness_deg": roughness_deg,
        "noise_seed": seed,
        "noise_model": _NOISE_MODEL,
        "public_order": "depth_center_ascending_then_position",
        "n_observed": int(obs_mask.sum()),
        "n_total": int(M),
        "version": _OBS_VERSION,
    }
    return Observation(
        operator="virtual_fmi",
        obs_mask=obs_mask,
        obs_normals=obs_normals,
        obs_positions=obs_positions,
        frac_ids=frac_ids,
        truth_sets=truth_sets,
        roughness_deg=roughness_deg,
        provenance=provenance,
        records=records,
    )


# ---------------------------------------------------------------------------
# 3. virtual_outcrop_scan -- plane intersection + point cloud + roughness
# ---------------------------------------------------------------------------

def virtual_outcrop_scan(dfn, scan_z, bbox_xy, density,
                         roughness_deg=0.0, seed=0):
    """Simulate digital outcrop scan: plane intersection + point cloud.

    Scans a horizontal plane at z = scan_z.  For each disc that intersects
    this plane, samples points along the chord (intersection line segment)
    at the given density.  Each point gets the fracture's normal plus
    angular noise controlled by roughness_deg.

    G66: discs are processed in public order (chord centre x, y, chord length)
    and the per-point noise is drawn from one RNG that only sees observed
    discs, so hidden-object insertion/permutation cannot shift the stream.

    Parameters
    ----------
    dfn : DFNRealization
    scan_z : float -- z-coordinate of the scan plane
    bbox_xy : ((xmin, xmax), (ymin, ymax)) -- bounding box on the scan plane
    density : float -- points per meter along each trace
    roughness_deg : float -- angular noise on normals
    seed : int

    Returns
    -------
    Observation
        records: list of dict per observed fracture, sorted by chord centre,
            keys frac_id, n_points, chord_length, trace_points is not stored.
        The observed normal per fracture is the sign-aligned mean of the
        noisy point normals (averaging reduces effective noise).
    """
    (xmin, xmax), (ymin, ymax) = bbox_xy
    M = dfn.normals.shape[0]

    obs_mask = np.zeros(M, dtype=bool)
    obs_normals = dfn.normals.copy()
    obs_positions = dfn.centers.copy().astype(np.float64)
    truth_sets = dfn.sets.copy()

    hits = []  # (chord_cx, chord_cy, chord_length, row index, point arrays)
    for i in range(M):
        n = dfn.normals[i]
        C = dfn.centers[i]
        R = dfn.radii[i]

        nx, ny, nz = n
        denom = np.sqrt(nx ** 2 + ny ** 2)
        if denom < 1e-10:
            # Disc is horizontal (parallel to scan plane) -- skip or tangent
            continue
        d = abs(nz * (scan_z - C[2])) / denom
        if d > R:
            # Disc does not intersect scan plane
            continue

        half_len = np.sqrt(max(R ** 2 - d ** 2, 0.0))
        chord_dir = np.array([-ny, nx]) / denom
        t0 = -(nz / (denom ** 2)) * (scan_z - C[2])
        chord_cx = C[0] + t0 * nx
        chord_cy = C[1] + t0 * ny

        chord_length = 2.0 * half_len
        n_pts = max(int(np.ceil(chord_length * density)), 3)

        ts = np.linspace(-half_len, half_len, n_pts)
        pts_x = chord_cx + ts * chord_dir[0]
        pts_y = chord_cy + ts * chord_dir[1]
        pts_z = np.full(n_pts, scan_z, dtype=np.float64)

        in_bbox = (pts_x >= xmin) & (pts_x <= xmax) & \
                  (pts_y >= ymin) & (pts_y <= ymax)
        if not np.any(in_bbox):
            continue
        pts = np.stack([pts_x[in_bbox], pts_y[in_bbox], pts_z[in_bbox]], axis=1)
        hits.append((chord_cx, chord_cy, chord_length, i, n, pts))

    hits.sort(key=lambda h: (_rp(h[0], 12), _rp(h[1], 12),
                             round(h[2], 4), h[4].shape[0]))

    rng = _new_noise_rng(seed)
    records = []
    point_normals = []
    for chord_cx, chord_cy, chord_length, i, n, pts in hits:
        n_pts_f = int(pts.shape[0])
        pt_normals = _noise_observed(np.tile(n, (n_pts_f, 1)), roughness_deg, rng)
        pts_s = _sign_align(pt_normals)
        obs_n = _unit(pts_s.mean(0))
        obs_normals[i] = obs_n
        obs_positions[i] = np.array([chord_cx, chord_cy, scan_z])
        obs_mask[i] = True
        point_normals.append(pt_normals)
        records.append({
            "frac_id": int(i),
            "n_points": n_pts_f,
            "chord_length": round(float(chord_length), 4),
        })

    frac_ids = np.where(obs_mask)[0]

    provenance = {
        "operator": "virtual_outcrop_scan",
        "scan_z": float(scan_z),
        "bbox_xy": [[xmin, xmax], [ymin, ymax]],
        "density": float(density),
        "roughness_deg": float(roughness_deg),
        "noise_seed": int(seed),
        "noise_model": _NOISE_MODEL,
        "public_order": "chord_center_x_then_y_then_chord_length",
        "n_observed": int(obs_mask.sum()),
        "n_total": int(M),
        "n_scan_points": int(sum(r["n_points"] for r in records)),
        "version": _OBS_VERSION,
    }
    return Observation(
        operator="virtual_outcrop_scan",
        obs_mask=obs_mask,
        obs_normals=obs_normals,
        obs_positions=obs_positions,
        frac_ids=frac_ids,
        truth_sets=truth_sets,
        roughness_deg=roughness_deg,
        provenance=provenance,
        records=records,
        point_normals=point_normals,
    )


# ---------------------------------------------------------------------------
# utility: compute oracle centers from truth set_ids
# ---------------------------------------------------------------------------

def oracle_centers_from_obs(obs):
    """Compute oracle group modal directions from an Observation.

    Uses the true set_ids (obs.truth_sets) to group observed normals,
    then computes sign-aligned mean per group.  Only groups with at
    least one observed fracture produce a center.

    Returns
    -------
    (K_obs, 3) array of unit modal directions (sorted by group id).
    """
    mask = obs.obs_mask
    nrm = obs.obs_normals[mask]
    sets = obs.truth_sets[mask]
    classes = np.unique(sets)
    centers = []
    for k in classes:
        sel = sets == k
        if sel.sum() == 0:
            continue
        pts = nrm[sel]
        pts_s = _sign_align(pts)
        c = _unit(pts_s.mean(0))
        centers.append(c)
    if not centers:
        return np.empty((0, 3))
    return np.array(centers)


# ---------------------------------------------------------------------------
# utility: compute honest centers via observation-only k-means
# ---------------------------------------------------------------------------

def honest_centers_from_obs(obs, K, seed=42):
    """Compute honest group modal directions via observation-only k-means.

    Uses spherical k-means (|cos| assignment + sign-aligned mean) on
    observed normals only.  This mirrors the R117/R127 pipeline.

    Returns
    -------
    (K_eff, 3) array of unit modal directions.
    """
    mask = obs.obs_mask
    nrm = obs.obs_normals[mask]
    if len(nrm) == 0:
        return np.empty((0, 3))
    # Inline spherical k-means (same caliber as setlabel.spherical_kmeans)
    rng = np.random.default_rng(seed)
    nrm = _unit(nrm)
    n = len(nrm)
    K_eff = min(K, n)
    if K_eff <= 1:
        c = _unit(_sign_align(nrm).mean(0))
        return c[None]
    centers = nrm[rng.choice(n, K_eff, replace=False)].copy()
    for _ in range(80):
        cos = np.abs(nrm @ centers.T)
        assign = cos.argmax(1)
        changed = False
        for k in range(K_eff):
            sel = assign == k
            if sel.any():
                new_c = _unit(_sign_align(nrm[sel]).mean(0))
                if not np.allclose(new_c, centers[k], atol=1e-12):
                    changed = True
                centers[k] = new_c
        if not changed:
            break
    return centers
