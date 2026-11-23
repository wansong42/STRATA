# -*- coding: utf-8 -*-
"""P2 exact baselines (G68 / G62R): B1 RANSAC, B2 region growing, B3 CT components.

Contract
--------
* Every hypothesis set is enumerated over a **declared finite object universe**
  produced by one deterministic baseline.  ``hypothesis_scope`` is
  ``EXACT_FINITE_ENUMERATION`` for the presence/absence masks of that universe;
  the universe itself is ``DATA_DRIVEN_CANDIDATE_ONLY`` (no continuous-cover
  claim about the plane parameters).
* The only query family consumed here is ``PLANAR_UNION_CONNECTIVITY``; every
  other semantic must be rejected upstream by
  ``identification.require_supported_query_semantics``.
* must/may are derived hypothesis-by-hypothesis from the finite set H; the
  envelope is the element-wise AND/OR over H (validated against
  ``identification.envelope_of`` on a deterministic subsample when H is large).
* Status priority is fixed: empty/contradictory H -> MODEL_CONFLICT; budget or
  wall-clock exhaustion -> SEARCH_UNRESOLVED (even if one label was already
  observed); {0} -> CERTIFIED_DISCONNECTED; {1} -> CERTIFIED_CONNECTED;
  {0,1} -> WITNESSED_AMBIGUOUS.

This module never imports torch and never touches a GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

HYPOTHESIS_SCOPE = "EXACT_FINITE_ENUMERATION"
CANDIDATE_SCOPE = "DATA_DRIVEN_CANDIDATE_ONLY"
QUERY_KIND = "PLANAR_UNION_CONNECTIVITY"
READING_DIAGNOSTIC = "BASELINE_DIAGNOSTIC"

STATUS_MODEL_CONFLICT = "MODEL_CONFLICT"
STATUS_SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"
STATUS_CERTIFIED_DISCONNECTED = "CERTIFIED_DISCONNECTED"
STATUS_CERTIFIED_CONNECTED = "CERTIFIED_CONNECTED"
STATUS_WITNESSED_AMBIGUOUS = "WITNESSED_AMBIGUOUS"

DIAGNOSTIC_LABEL = {
    STATUS_MODEL_CONFLICT: "BASELINE_DIAGNOSTIC_MODEL_CONFLICT",
    STATUS_SEARCH_UNRESOLVED: "BASELINE_DIAGNOSTIC_SEARCH_UNRESOLVED",
    STATUS_CERTIFIED_DISCONNECTED: "BASELINE_DIAGNOSTIC_DISCONNECTED",
    STATUS_CERTIFIED_CONNECTED: "BASELINE_DIAGNOSTIC_CONNECTED",
    STATUS_WITNESSED_AMBIGUOUS: "BASELINE_DIAGNOSTIC_AMBIGUOUS",
}

PUBLIC_ENTRY_KEYS = frozenset({"position", "normal", "record"})
FORBIDDEN_KEYS = frozenset({
    "frac_id", "frac_ids", "truth_sets", "dfn_seed", "hidden_truth",
    "true_normal", "true_normals", "oracle", "world_seed", "full_network",
    "truth", "truth_mask", "centers_true", "normals_true", "radii_true",
    "pos_true", "nrm_full", "set_ids", "raw_xyz", "xyz_raw", "fracture_id",
    "fid", "frac_normals", "frac_size",
})
ALLOWED_TOP_KEYS = frozenset({
    "schema", "operator", "operator_version", "n_observed", "noise_spec",
    "measurement", "public_order", "observations", "case_id", "face", "site",
})
ALLOWED_MEASUREMENT_KEYS = frozenset({
    "x0", "d", "well_center", "well_axis", "well_radius", "scan_z",
    "bbox_xy", "density", "world_box",
})


class BaselineContractError(ValueError):
    """Explicit contract violation (never silently repaired)."""


class EmptyObservationError(SystemExit):
    """n_observed == 0: fail loudly, never emit an empty reading."""


class ScaleGuardError(BaselineContractError):
    """Observed coordinates fall outside the declared metric world box."""


class TruthLeakError(BaselineContractError):
    """A forbidden (truth / hidden / restricted) key reached a baseline input."""


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def _f(x: float) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise BaselineContractError("non-finite float in output")
    return round(v, 12)


def _arr(a: Sequence[float]) -> List[float]:
    return [_f(x) for x in np.asarray(a, dtype=np.float64).reshape(-1)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1e-12:
        raise BaselineContractError("degenerate normal")
    return np.asarray(v, dtype=np.float64) / n


def _sign_fix(n: np.ndarray) -> np.ndarray:
    """Deterministic orientation: largest |component| is positive."""
    idx = int(np.argmax(np.abs(n)))
    return -n if n[idx] < 0 else n


def _plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(n, a))
    e2 = _unit(np.cross(n, e1))
    return e1, e2


# ---------------------------------------------------------------------------
# Input contract / public view
# ---------------------------------------------------------------------------

def assert_public_only(obj: Any, path: str = "payload") -> int:
    """Reject forbidden truth / hidden / restricted keys anywhere in the tree."""
    seen = 0
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            ks = str(k)
            if ks in FORBIDDEN_KEYS:
                raise TruthLeakError("forbidden key %r at %s" % (ks, path))
            seen += 1 + assert_public_only(v, "%s.%s" % (path, ks))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            seen += 1 + assert_public_only(v, "%s[%d]" % (path, i))
    return seen


def assert_public_payload(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise BaselineContractError("public payload must be a mapping")
    extra = set(payload) - ALLOWED_TOP_KEYS
    if extra:
        raise TruthLeakError("non-whitelisted top-level keys: %r" % sorted(extra))
    obs = payload.get("observations")
    if not isinstance(obs, list):
        raise BaselineContractError("payload.observations must be a list")
    if len(obs) == 0:
        raise EmptyObservationError("G68: empty public observation, refusing reading")
    n = 0
    for i, entry in enumerate(obs):
        if not isinstance(entry, Mapping) or set(entry) != PUBLIC_ENTRY_KEYS:
            raise TruthLeakError("observation %d must carry exactly %r"
                                 % (i, sorted(PUBLIC_ENTRY_KEYS)))
        p = np.asarray(entry["position"], dtype=np.float64).reshape(-1)
        nn = np.asarray(entry["normal"], dtype=np.float64).reshape(-1)
        if p.shape != (3,) or nn.shape != (3,):
            raise BaselineContractError("observation %d position/normal must be 3-vectors" % i)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(nn))):
            raise BaselineContractError("observation %d has non-finite values" % i)
        if abs(float(np.linalg.norm(nn)) - 1.0) > 1e-6:
            raise BaselineContractError("observation %d normal is not unit" % i)
        n += 1
    if int(payload.get("n_observed", -1)) != n:
        raise BaselineContractError("n_observed mismatch")
    measurement = payload.get("measurement") or {}
    if "world_box" in measurement:
        wb = measurement["world_box"]
        if not (isinstance(wb, Mapping) and "lo" in wb and "hi" in wb):
            raise BaselineContractError("measurement.world_box must carry lo/hi")
    assert_public_only(payload)
    return {"n_observed": n, "n_key_nodes": assert_public_only(payload)}


def payload_points(payload: Mapping) -> Tuple[np.ndarray, np.ndarray]:
    assert_public_payload(payload)
    entries = payload["observations"]
    pts = np.asarray([e["position"] for e in entries], dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray([e["normal"] for e in entries], dtype=np.float64).reshape(-1, 3)
    return pts, nrm


def scale_guard(points: np.ndarray, world_box: Mapping, margin_frac: float = 0.01) -> dict:
    """Metric-scale guard: observed points must lie in the declared world box."""
    lo = np.asarray(world_box["lo"], dtype=np.float64).reshape(3)
    hi = np.asarray(world_box["hi"], dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)):
        raise ScaleGuardError("declared world_box is not a finite proper box")
    span = float(np.max(hi - lo))
    margin = margin_frac * span
    if not (np.all(points >= lo - margin) and np.all(points <= hi + margin)):
        raise ScaleGuardError(
            "observed points outside declared metric world box [%s, %s] "
            "(unit/scale error suspected)" % (lo.tolist(), hi.tolist()))
    return {"world_box": {"lo": _arr(lo), "hi": _arr(hi)}, "span": _f(span),
            "margin": _f(margin)}


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def median_nn_spacing(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 4:
        raise BaselineContractError("median_nn_spacing needs >= 4 points, got %d" % n)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    if not np.all(np.isfinite(nn)) or float(np.median(nn)) <= 0.0:
        raise BaselineContractError("degenerate point set (zero/NaN NN spacing)")
    return float(np.median(nn))


def plane_distances(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(points, dtype=np.float64) - center) @ normal)


def fit_plane(points: np.ndarray, ref_normal: Optional[np.ndarray] = None) -> dict:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 1:
        raise BaselineContractError("fit_plane on empty support")
    center = pts.mean(axis=0)
    centered = pts - center
    if pts.shape[0] >= 3:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(np.sum(s > 1e-9 * max(float(s[0]), 1e-12)))
    else:
        s = np.zeros(3)
        vt = np.eye(3)
        rank = 0
    if rank >= 2:
        normal = _unit(vt[2])
        fit_mode = "SVD_PLANE"
    elif ref_normal is not None:
        normal = _unit(np.asarray(ref_normal, dtype=np.float64))
        fit_mode = "OBSERVED_NORMAL_DEGENERATE_POSITIONS"
    else:
        raise BaselineContractError("degenerate collinear support and no reference normal")
    if ref_normal is not None and float(normal @ _unit(np.asarray(ref_normal, dtype=np.float64))) < 0:
        normal = -normal
    else:
        normal = _sign_fix(normal)
    e1, e2 = _plane_basis(normal)
    a = centered @ e1
    b = centered @ e2
    radius = float(np.max(np.sqrt(a * a + b * b))) if pts.shape[0] else 0.0
    d = plane_distances(pts, center, normal)
    rms = float(np.sqrt(np.mean(d * d)))
    return {"center": center, "normal": normal, "radius": max(radius, 1e-9),
            "rms": rms, "rank": rank, "fit_mode": fit_mode,
            "singular_values": [float(x) for x in s]}


def make_object(object_id: str, points: np.ndarray, observed_normals: Optional[np.ndarray],
                support_indices: Sequence[int]) -> dict:
    idx = np.asarray(list(support_indices), dtype=np.int64)
    ref = None
    if observed_normals is not None and idx.size:
        ref = observed_normals[idx].mean(axis=0)
        if np.linalg.norm(ref) <= 1e-12:
            ref = None
    fit = fit_plane(points[idx], ref_normal=ref)
    disp = None
    if observed_normals is not None and idx.size:
        n_fit = fit["normal"]
        cos = np.clip(np.abs(observed_normals[idx] @ n_fit), -1.0, 1.0)
        disp = float(np.degrees(np.arccos(cos)).mean())
    return {"object_id": object_id,
            "center": _arr(fit["center"]), "normal": _arr(fit["normal"]),
            "radius": _f(fit["radius"]), "rms": _f(fit["rms"]),
            "support": int(idx.size), "fit_mode": fit["fit_mode"],
            "rank": int(fit["rank"]),
            "normal_dispersion_deg": None if disp is None else _f(disp),
            "_support_indices": idx}


def public_object(obj: Mapping) -> dict:
    return {k: v for k, v in obj.items() if not str(k).startswith("_")}


# ---------------------------------------------------------------------------
# B1: deterministic RANSAC planar family
# ---------------------------------------------------------------------------

def ransac_planar_family(points: np.ndarray, normals: Optional[np.ndarray], *,
                         seed: int, tol: float, iterations: int = 5000,
                         min_inliers: int = 3, max_objects: int = 13,
                         refit_rounds: int = 2,
                         max_seconds: float = 20.0) -> dict:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < min_inliers:
        return {"objects": [], "unexplained_indices": list(range(n)),
                "n_iterations_total": 0, "stop_reason": "TOO_FEW_POINTS"}
    rng = np.random.default_rng(int(seed))
    remaining = np.arange(n, dtype=np.int64)
    objects: List[dict] = []
    used_iterations = 0
    n_degenerate = 0
    stop_reason = "MAX_OBJECTS"
    t0 = time.perf_counter()
    while remaining.size >= min_inliers and len(objects) < max_objects:
        if time.perf_counter() - t0 > max_seconds:
            stop_reason = "TIME_LIMIT"
            break
        P = pts[remaining]
        m = P.shape[0]
        idx = rng.integers(0, m, size=(int(iterations), 3), dtype=np.int64)
        ok = ((idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2]))
        idx = idx[ok]
        used_iterations += int(idx.shape[0])
        if idx.shape[0] == 0:
            stop_reason = "NO_VALID_SAMPLES"
            break
        p0 = P[idx[:, 0]]
        p1 = P[idx[:, 1]]
        p2 = P[idx[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        nrm_norm = np.linalg.norm(nrm, axis=1)
        valid = nrm_norm > 1e-12
        idx, p0, nrm, nrm_norm = idx[valid], p0[valid], nrm[valid], nrm_norm[valid]
        if idx.shape[0] == 0:
            stop_reason = "ALL_SAMPLES_DEGENERATE"
            break
        nrm = nrm / nrm_norm[:, None]
        diff = P[None, :, :] - p0[:, None, :]
        dist = np.abs(np.einsum("ijk,ik->ij", diff, nrm))
        inlier = dist <= tol
        counts = inlier.sum(axis=1).astype(np.int64)
        mean_d = np.where(counts > 0,
                          (dist * inlier).sum(axis=1) / np.maximum(counts, 1),
                          np.inf)
        order = np.lexsort((np.arange(counts.size), mean_d, -counts))
        best = int(order[0])
        if int(counts[best]) < min_inliers:
            stop_reason = "NO_PLANE_ABOVE_MIN_INLIERS"
            break
        mask = inlier[best].copy()
        for _ in range(int(refit_rounds)):
            if int(mask.sum()) < min_inliers:
                break
            try:
                fit = fit_plane(P[mask], ref_normal=None if normals is None else
                                normals[remaining][mask].mean(axis=0))
            except BaselineContractError:
                break
            new_mask = plane_distances(P, fit["center"], fit["normal"]) <= tol
            if int(new_mask.sum()) < min_inliers:
                break
            if np.array_equal(new_mask, mask):
                mask = new_mask
                break
            mask = new_mask
        support_global = remaining[mask]
        try:
            obj = make_object("O%02d" % len(objects), pts, normals, support_global)
        except BaselineContractError:
            n_degenerate += 1
            remaining = remaining[~mask]
            continue
        obj["baseline"] = "B1_RANSAC_PLANAR_FAMILY"
        objects.append(obj)
        remaining = remaining[~mask]
    if len(objects) >= max_objects:
        stop_reason = "MAX_OBJECTS"
    elif remaining.size < min_inliers and stop_reason == "MAX_OBJECTS":
        stop_reason = "EXHAUSTED"
    return {"objects": objects, "unexplained_indices": remaining.tolist(),
            "n_iterations_total": used_iterations, "stop_reason": stop_reason,
            "degenerate_dropped": int(n_degenerate)}


# ---------------------------------------------------------------------------
# B2: deterministic region growing (kNN, observed-normal 15 deg)
# ---------------------------------------------------------------------------

def region_growing(points: np.ndarray, normals: np.ndarray, *,
                   k: int = 30, angle_deg: float = 15.0,
                   curvature_max: float = 0.2, min_size: int = 3,
                   max_objects: int = 13,
                   max_seconds: float = 20.0) -> dict:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < min_size:
        return {"objects": [], "unexplained_indices": list(range(n)),
                "stop_reason": "TOO_FEW_POINTS", "local_curvature_median": None}
    kk = int(min(max(int(k), 1), n - 1))
    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=kk + 1)
    nbr = np.asarray(nbr, dtype=np.int64)[:, 1:]
    curv = np.zeros(n, dtype=np.float64)
    for i in range(n):
        Q = pts[nbr[i]] - pts[nbr[i]].mean(axis=0)
        ev = np.linalg.eigvalsh(Q.T @ Q)
        ev = np.clip(ev, 0.0, None)
        total = float(ev.sum())
        curv[i] = 0.0 if total <= 1e-15 else float(ev[0] / total)
    adj: List[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in nbr[i]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    order = np.lexsort((np.arange(n), curv))
    label = np.full(n, -1, dtype=np.int64)
    regions: List[List[int]] = []
    t0 = time.perf_counter()
    stop_reason = "COMPLETE"
    for seed in order.tolist():
        if time.perf_counter() - t0 > max_seconds:
            stop_reason = "TIME_LIMIT"
            break
        if label[seed] != -1 or curv[seed] > curvature_max:
            continue
        rid = len(regions)
        label[seed] = rid
        members = [seed]
        region_normal = nrm[seed].copy()
        queue = [seed]
        while queue:
            cur = queue.pop()
            for nb in sorted(adj[cur]):
                if label[nb] != -1 or curv[nb] > curvature_max:
                    continue
                cosang = float(np.clip(abs(nrm[nb] @ _unit(region_normal)), -1.0, 1.0))
                if float(np.degrees(np.arccos(cosang))) > angle_deg:
                    continue
                label[nb] = rid
                members.append(nb)
                region_normal = region_normal + nrm[nb]
                queue.append(nb)
        regions.append(members)
    kept = [r for r in regions if len(r) >= min_size]
    kept.sort(key=lambda r: (-len(r), int(min(r))))
    kept = kept[:max_objects]
    kept_ids = set(tuple(r) for r in kept)
    unexplained = [i for i in range(n)
                   if all(tuple(r) not in kept_ids or i not in r for r in kept)]
    objects = []
    n_degenerate = 0
    for j, members in enumerate(kept):
        try:
            obj = make_object("O%02d" % j, pts, nrm, members)
        except BaselineContractError:
            n_degenerate += 1
            continue
        obj["baseline"] = "B2_REGION_GROWING"
        objects.append(obj)
    dropped = {i for i in range(n) if all(i not in r for r in [o["_support_indices"].tolist() for o in objects])}
    return {"objects": objects,
            "unexplained_indices": sorted(set(unexplained) | dropped),
            "stop_reason": stop_reason, "degenerate_dropped": int(n_degenerate),
            "local_curvature_median": _f(float(np.median(curv)))}


# ---------------------------------------------------------------------------
# B3: CT connected components (6-adjacency voxels)
# ---------------------------------------------------------------------------

def ct_components(points: np.ndarray, normals: np.ndarray, *, voxel: float,
                  min_points: int = 3, max_objects: int = 13,
                  max_seconds: float = 20.0) -> dict:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < min_points:
        return {"objects": [], "unexplained_indices": list(range(n)),
                "stop_reason": "TOO_FEW_POINTS", "n_voxels": 0}
    if not (math.isfinite(voxel) and voxel > 0):
        raise BaselineContractError("voxel size must be positive and finite")
    keys = np.floor((pts - pts.min(axis=0)) / float(voxel)).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    owner: Dict[Tuple[int, int, int], int] = {
        tuple(int(x) for x in uniq[i]): i for i in range(uniq.shape[0])}
    adj: List[set] = [set() for _ in range(uniq.shape[0])]
    for i in range(uniq.shape[0]):
        x, y, z = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            j = owner.get((x + d[0], y + d[1], z + d[2]))
            if j is not None:
                adj[i].add(j)
    seen = np.zeros(uniq.shape[0], dtype=bool)
    comps: List[List[int]] = []
    t0 = time.perf_counter()
    stop_reason = "COMPLETE"
    for i in range(uniq.shape[0]):
        if seen[i]:
            continue
        if time.perf_counter() - t0 > max_seconds:
            stop_reason = "TIME_LIMIT"
            break
        seen[i] = True
        stack = [i]
        vox = []
        while stack:
            cur = stack.pop()
            vox.append(cur)
            for nb in sorted(adj[cur]):
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        point_idx = np.flatnonzero(np.isin(inverse, vox)).tolist()
        if len(point_idx) >= min_points:
            comps.append(point_idx)
    comps.sort(key=lambda r: (-len(r), int(min(r))))
    comps = comps[:max_objects]
    used = set()
    for r in comps:
        used.update(r)
    unexplained = [i for i in range(n) if i not in used]
    objects = []
    n_degenerate = 0
    for j, members in enumerate(comps):
        try:
            obj = make_object("O%02d" % j, pts, nrm, members)
        except BaselineContractError:
            n_degenerate += 1
            continue
        obj["baseline"] = "B3_CT_CONNECTED_COMPONENTS"
        objects.append(obj)
    used2 = set()
    for o in objects:
        used2.update(int(x) for x in o["_support_indices"])
    unexplained = [i for i in range(n) if i not in used2]
    return {"objects": objects, "unexplained_indices": unexplained,
            "stop_reason": stop_reason, "n_voxels": int(uniq.shape[0]),
            "degenerate_dropped": int(n_degenerate)}


# ---------------------------------------------------------------------------
# Finite hypothesis enumeration
# ---------------------------------------------------------------------------

def prefix_object_ids(objects: Sequence[Mapping], tag: str) -> List[dict]:
    out = []
    for obj in objects:
        d = dict(obj)
        d["object_id"] = "%s:%s" % (tag, str(d["object_id"]))
        out.append(d)
    return out


def merge_universes(universes: Sequence[Sequence[Mapping]], *, tau: float,
                    angle_deg: float = 5.0,
                    center_frac: float = 0.5) -> dict:
    """Deduplicate candidate objects across baselines into one finite universe.

    Two objects are the same candidate when |angle(n_i, n_j)| <= angle_deg,
    |(c_j - c_i) . n_i| <= tau and the in-plane centre distance is
    <= center_frac * (r_i + r_j).  Deterministic: the first occurrence (input
    order) wins; every merge is recorded.
    """
    kept: List[dict] = []
    merged: List[dict] = []
    for group in universes:
        for obj in group:
            c = np.asarray(obj["center"], dtype=np.float64)
            nv = _unit(np.asarray(obj["normal"], dtype=np.float64))
            r = float(obj["radius"])
            dup_of = None
            for j, prev in enumerate(kept):
                cp = np.asarray(prev["center"], dtype=np.float64)
                npv = _unit(np.asarray(prev["normal"], dtype=np.float64))
                cosang = float(np.clip(abs(nv @ npv), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosang)))
                plane_d = abs(float((c - cp) @ npv))
                delta = c - cp
                inplane = float(np.linalg.norm(delta - float(delta @ npv) * npv))
                if (ang <= angle_deg and plane_d <= tau
                        and inplane <= center_frac * (r + float(prev["radius"]))):
                    dup_of = j
                    break
            if dup_of is None:
                d = dict(obj)
                d["merged_from"] = [str(obj["object_id"])]
                kept.append(d)
            else:
                merged.append({"object_id": str(obj["object_id"]),
                               "merged_into": str(kept[dup_of]["object_id"])})
                kept[dup_of]["merged_from"].append(str(obj["object_id"]))
    return {"objects": kept, "n_input": sum(len(g) for g in universes),
            "n_kept": len(kept), "n_merged": len(merged), "merged": merged}


def _adj_bits(hit: np.ndarray) -> List[int]:
    k = int(hit.shape[0])
    return [sum(1 << j for j in range(k) if bool(hit[i, j])) for i in range(k)]


def _bits_label(mask: int, adj_bits: Sequence[int], a_bits: int, b_bits: int) -> int:
    sa = mask & a_bits
    sb = mask & b_bits
    if not sa or not sb:
        return 0
    seen = sa
    frontier = sa
    while frontier:
        nxt = 0
        f = frontier
        while f:
            lsb = f & -f
            nxt |= adj_bits[lsb.bit_length() - 1]
            f ^= lsb
        nxt &= mask & ~seen
        if nxt & sb:
            return 1
        seen |= nxt
        frontier = nxt
    return 1 if (seen & sb) else 0


def bitsets_for_query(objects: Sequence[Mapping], hit: np.ndarray,
                      point: Sequence[float], r_attach: float) -> int:
    flags = attach_flags(objects, point, r_attach)
    return sum(1 << j for j in range(len(objects)) if bool(flags[j]))


def project_masks(masks: Sequence[int], objects: Sequence[Mapping], hit: np.ndarray,
                  a_bits: int, b_bits: int) -> List[int]:
    adj = _adj_bits(hit)
    return [int(_bits_label(int(m), adj, a_bits, b_bits)) for m in masks]


def envelope_from_masks(masks: Sequence[int], objects: Sequence[Mapping],
                        hit: np.ndarray, a_bits: int, b_bits: int) -> dict:
    """Element-wise AND/OR over every mask of H, computed as integer bitsets."""
    k = len(objects)
    if len(masks) == 0:
        return envelope_from_hypotheses([])
    and_all = (1 << k) - 1
    or_all = 0
    any_a = any_b = False
    all_a = all_b = True
    for m in masks:
        m = int(m)
        and_all &= m
        or_all |= m
        ha = bool(m & a_bits)
        hb = bool(m & b_bits)
        any_a = any_a or ha
        any_b = any_b or hb
        all_a = all_a and ha
        all_b = all_b and hb
    pairs_must = 0
    pairs_may = 0
    marr = np.asarray([int(m) for m in masks], dtype=np.int64)
    for i in range(k):
        for j in range(i + 1, k):
            if not bool(hit[i, j]):
                continue
            both = (1 << i) | (1 << j)
            if (and_all & both) == both:
                pairs_must |= both
            if bool(np.any((marr & both) == both)):
                pairs_may |= both
    def ids(bits: int) -> List[str]:
        return [str(objects[j]["object_id"]) for j in range(k) if (bits >> j) & 1]
    def edge_list(bits: int) -> List[List[str]]:
        out = []
        for i in range(k):
            for j in range(i + 1, k):
                if (bits >> i) & 1 and (bits >> j) & 1 and bool(hit[i, j]):
                    a, b = str(objects[i]["object_id"]), str(objects[j]["object_id"])
                    out.append(sorted([a, b]))
        return sorted(out)
    return {"n_hypotheses": len(masks),
            "must_nodes": sorted(ids(and_all)), "may_nodes": sorted(ids(or_all)),
            "must_edges": edge_list(pairs_must), "may_edges": edge_list(pairs_may),
            "must_a_side": sorted(ids(and_all & a_bits)),
            "may_a_side": sorted(ids(or_all & a_bits)),
            "must_b_side": sorted(ids(and_all & b_bits)),
            "may_b_side": sorted(ids(or_all & b_bits)),
            "must_attach_a": bool(all_a), "may_attach_a": bool(any_a),
            "must_attach_b": bool(all_b), "may_attach_b": bool(any_b)}


def _cap_universe(objects: Sequence[Mapping], max_universe: int) -> Tuple[List[dict], int, List[str]]:
    order = sorted(range(len(objects)),
                   key=lambda i: (-int(objects[i]["support"]), str(objects[i]["object_id"])))
    kept = [dict(objects[i]) for i in order[:max_universe]]
    dropped = [str(objects[i]["object_id"]) for i in order[max_universe:]]
    for j, obj in enumerate(kept):
        obj["universe_index"] = j
        obj["object_id"] = str(obj["object_id"])
    return kept, len(dropped), dropped


def enumerate_hypotheses(objects: Sequence[Mapping], points: np.ndarray, *,
                         tau: float, min_support: int = 3,
                         max_universe: int = 13, max_masks: int = 8192,
                         max_seconds: float = 60.0) -> dict:
    """All presence masks over the declared universe that explain the observation.

    Coverage rule (preregistered): an observed point is explained by an object
    when its distance to the object's **infinite supporting plane** is <= tau.
    Points explained by no candidate form the *unexplained set*; they are
    counted, never silently deleted, and they do not constrain H (baseline
    model limitation).  A present object must keep at least ``min_support``
    observed points within tau.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    kept, n_dropped_cap, dropped_ids = _cap_universe(objects, int(max_universe))
    t0 = time.perf_counter()
    keep_flags = []
    for obj in kept:
        c = np.asarray(obj["center"], dtype=np.float64)
        nv = np.asarray(obj["normal"], dtype=np.float64)
        cover = plane_distances(pts, c, nv) <= tau
        obj["_cover_count"] = int(cover.sum())
        keep_flags.append(int(cover.sum()) >= int(min_support))
    supported = [obj for obj, ok in zip(kept, keep_flags) if ok]
    n_dropped_support = len(kept) - len(supported)
    kept = supported
    k = len(kept)
    if k == 0:
        return {"objects": [], "masks": [], "n_masks_enumerated": 0,
                "n_feasible": 0, "n_unexplained_points": int(pts.shape[0]),
                "unexplained_indices": list(range(pts.shape[0])),
                "n_dropped_by_cap": int(n_dropped_cap),
                "n_dropped_unsupported": int(n_dropped_support),
                "dropped_by_cap_ids": dropped_ids,
                "search_complete": True, "stop_reason": "EMPTY_UNIVERSE",
                "elapsed_s": _f(time.perf_counter() - t0)}
    cover = np.zeros((pts.shape[0], k), dtype=bool)
    for j, obj in enumerate(kept):
        c = np.asarray(obj["center"], dtype=np.float64)
        nv = np.asarray(obj["normal"], dtype=np.float64)
        cover[:, j] = plane_distances(pts, c, nv) <= tau
    cover_bits = []
    unexplained = []
    for p in range(pts.shape[0]):
        b = 0
        for j in range(k):
            if cover[p, j]:
                b |= (1 << j)
        if b == 0:
            unexplained.append(p)
        cover_bits.append(b)
    constrained = [b for b in cover_bits if b != 0]
    total = (1 << k) - 1
    limit = int(total) if int(total) <= int(max_masks) else int(max_masks)
    stop_reason = "COMPLETE" if limit == total else "MAX_MASKS_EXCEEDED"
    if time.perf_counter() - t0 > max_seconds:
        return {"objects": kept, "masks": [], "n_masks_enumerated": 0,
                "n_masks_possible": total, "n_feasible": 0,
                "n_unexplained_points": len(unexplained),
                "unexplained_indices": unexplained,
                "n_dropped_by_cap": int(n_dropped_cap),
                "n_dropped_unsupported": int(n_dropped_support),
                "dropped_by_cap_ids": dropped_ids,
                "search_complete": False, "stop_reason": "TIME_LIMIT",
                "elapsed_s": _f(time.perf_counter() - t0)}
    marr = np.arange(1, limit + 1, dtype=np.int64)
    valid = np.ones(limit, dtype=bool)
    for b in constrained:
        valid &= (marr & np.int64(b)) != 0
    masks = marr[valid].tolist()
    enumerated = int(limit)
    return {"objects": kept, "masks": masks, "n_masks_enumerated": enumerated,
            "n_masks_possible": total, "n_feasible": len(masks),
            "n_unexplained_points": len(unexplained),
            "unexplained_indices": unexplained,
            "n_dropped_by_cap": int(n_dropped_cap),
            "n_dropped_unsupported": int(n_dropped_support),
            "dropped_by_cap_ids": dropped_ids,
            "search_complete": stop_reason == "COMPLETE",
            "stop_reason": stop_reason,
            "elapsed_s": _f(time.perf_counter() - t0)}


def pair_hit_matrix(objects: Sequence[Mapping]) -> np.ndarray:
    k = len(objects)
    hit = np.zeros((k, k), dtype=bool)
    if k < 2:
        return hit
    from fractureflow.intersect_disk import disk_pair_intersect
    centers = np.asarray([o["center"] for o in objects], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray([o["normal"] for o in objects], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray([o["radius"] for o in objects], dtype=np.float64)
    ii, jj = np.triu_indices(k, 1)
    res = disk_pair_intersect(centers, normals, radii, ii, jj)
    for a, b, r in zip(ii.tolist(), jj.tolist(), res.tolist()):
        hit[a, b] = hit[b, a] = bool(r)
    return hit


def attach_flags(objects: Sequence[Mapping], point: Sequence[float], r_attach: float) -> np.ndarray:
    k = len(objects)
    if k == 0:
        return np.zeros(0, dtype=bool)
    from fractureflow.connectivity_qoi import disk_point_distance
    centers = np.asarray([o["center"] for o in objects], dtype=np.float64).reshape(-1, 3)
    normals = np.asarray([o["normal"] for o in objects], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray([o["radius"] for o in objects], dtype=np.float64)
    p = np.asarray(point, dtype=np.float64).reshape(1, 3)
    d = disk_point_distance(centers, normals, radii, p)[0]
    return np.asarray(d <= float(r_attach), dtype=bool)


def mask_elements(mask: int, objects: Sequence[Mapping], hit: np.ndarray,
                  attach_a: np.ndarray, attach_b: np.ndarray) -> dict:
    present = [j for j in range(len(objects)) if (mask >> j) & 1]
    nodes = frozenset(str(objects[j]["object_id"]) for j in present)
    edges = set()
    for x in present:
        for y in present:
            if x < y and bool(hit[x, y]):
                u, v = str(objects[x]["object_id"]), str(objects[y]["object_id"])
                edges.add((min(u, v), max(u, v)))
    a_side = frozenset(str(objects[j]["object_id"]) for j in present if bool(attach_a[j]))
    b_side = frozenset(str(objects[j]["object_id"]) for j in present if bool(attach_b[j]))
    return {"mask": int(mask), "nodes": nodes, "edges": frozenset(edges),
            "a_side": a_side, "b_side": b_side}


def state_label(elements: Mapping) -> int:
    from fractureflow.identification import _sides_connected
    return int(bool(elements["a_side"]) and bool(elements["b_side"])
               and _sides_connected(elements["nodes"], elements["edges"],
                                    elements["a_side"], elements["b_side"]))


def envelope_from_hypotheses(hypotheses: Sequence[Mapping]) -> dict:
    """Element-wise AND/OR over every hypothesis (must/may from H itself)."""
    if len(hypotheses) == 0:
        return {"n_hypotheses": 0, "must_nodes": [], "may_nodes": [],
                "must_edges": [], "may_edges": [], "must_a_side": [], "may_a_side": [],
                "must_b_side": [], "may_b_side": [],
                "must_attach_a": False, "may_attach_a": False,
                "must_attach_b": False, "may_attach_b": False}
    nodes_u = set().union(*(set(h["nodes"]) for h in hypotheses))
    edges_u = set().union(*(set(h["edges"]) for h in hypotheses))
    a_u = set().union(*(set(h["a_side"]) for h in hypotheses))
    b_u = set().union(*(set(h["b_side"]) for h in hypotheses))
    must_nodes = {x for x in nodes_u if all(x in h["nodes"] for h in hypotheses)}
    may_nodes = set(nodes_u)
    must_edges = {e for e in edges_u if all(e in h["edges"] for h in hypotheses)}
    may_edges = set(edges_u)
    must_a = {x for x in a_u if all(x in h["a_side"] for h in hypotheses)}
    may_a = set(a_u)
    must_b = {x for x in b_u if all(x in h["b_side"] for h in hypotheses)}
    may_b = set(b_u)
    must_attach_a = all(bool(h["a_side"]) for h in hypotheses)
    may_attach_a = any(bool(h["a_side"]) for h in hypotheses)
    must_attach_b = all(bool(h["b_side"]) for h in hypotheses)
    may_attach_b = any(bool(h["b_side"]) for h in hypotheses)
    return {"n_hypotheses": len(hypotheses),
            "must_nodes": sorted(must_nodes), "may_nodes": sorted(may_nodes),
            "must_edges": [list(e) for e in sorted(must_edges)],
            "may_edges": [list(e) for e in sorted(may_edges)],
            "must_a_side": sorted(must_a), "may_a_side": sorted(may_a),
            "must_b_side": sorted(must_b), "may_b_side": sorted(may_b),
            "must_attach_a": bool(must_attach_a), "may_attach_a": bool(may_attach_a),
            "must_attach_b": bool(must_attach_b), "may_attach_b": bool(may_attach_b)}


def envelope_connectivity(envelope: Mapping) -> dict:
    """must/may connectivity on the derived envelope (identification authority)."""
    from fractureflow.identification import Envelope, must_connected, may_connected
    env = Envelope(
        n_states=int(envelope["n_hypotheses"]),
        must_nodes=frozenset(envelope["must_nodes"]),
        may_nodes=frozenset(envelope["may_nodes"]),
        must_edges=frozenset((min(e), max(e)) for e in envelope["must_edges"]),
        may_edges=frozenset((min(e), max(e)) for e in envelope["may_edges"]),
        must_a_side=frozenset(envelope["must_a_side"]),
        may_a_side=frozenset(envelope["may_a_side"]),
        must_b_side=frozenset(envelope["must_b_side"]),
        may_b_side=frozenset(envelope["may_b_side"]),
        must_attach_a=bool(envelope["must_attach_a"]),
        may_attach_a=bool(envelope["may_attach_a"]),
        must_attach_b=bool(envelope["must_attach_b"]),
        may_attach_b=bool(envelope["may_attach_b"]),
    )
    return {"must_connected": bool(must_connected(env)),
            "may_connected": bool(may_connected(env))}


def validate_envelope_reference(hypotheses: Sequence[Mapping], envelope: Mapping,
                                sample_limit: int = 128) -> dict:
    """Compare the bitset envelope against identification.envelope_of on a sample.

    The sample is deterministic (first/last plus a fixed stride).
    """
    from fractureflow.identification import StateDerivation, envelope_of
    h = list(hypotheses)
    if not h:
        return {"checked": 0, "mismatch": 0, "ok": True}
    if len(h) <= sample_limit:
        sample = h
    else:
        stride = max(1, len(h) // (sample_limit - 2))
        idx = list(range(0, len(h), stride))[:sample_limit - 2] + [len(h) - 1]
        sample = [h[i] for i in sorted(set(idx))]
    ds = [StateDerivation(state_id=i, nodes=x["nodes"], edges=x["edges"],
                          a_side=x["a_side"], b_side=x["b_side"],
                          attach_a=bool(x["a_side"]), attach_b=bool(x["b_side"]),
                          label=int(x["label"]))
          for i, x in enumerate(sample)]
    ref = envelope_of(ds)
    pairs = [("must_nodes", ref.must_nodes, set(envelope["must_nodes"])),
             ("may_nodes", ref.may_nodes, set(envelope["may_nodes"])),
             ("must_edges", set(ref.must_edges),
              set((min(e), max(e)) for e in envelope["must_edges"])),
             ("may_edges", set(ref.may_edges),
              set((min(e), max(e)) for e in envelope["may_edges"])),
             ("must_a_side", ref.must_a_side, set(envelope["must_a_side"])),
             ("may_a_side", ref.may_a_side, set(envelope["may_a_side"])),
             ("must_b_side", ref.must_b_side, set(envelope["must_b_side"])),
             ("may_b_side", ref.may_b_side, set(envelope["may_b_side"]))]
    mismatch = 0
    detail = []
    for name, a, b in pairs:
        if set(a) != set(b):
            mismatch += 1
            detail.append(name)
    if bool(ref.must_attach_a) != bool(envelope["must_attach_a"]):
        mismatch += 1
        detail.append("must_attach_a")
    if bool(ref.may_attach_a) != bool(envelope["may_attach_a"]):
        mismatch += 1
        detail.append("may_attach_a")
    if bool(ref.must_attach_b) != bool(envelope["must_attach_b"]):
        mismatch += 1
        detail.append("must_attach_b")
    if bool(ref.may_attach_b) != bool(envelope["may_attach_b"]):
        mismatch += 1
        detail.append("may_attach_b")
    return {"checked": len(sample), "mismatch": mismatch, "detail": detail,
            "ok": mismatch == 0, "reference": "identification.envelope_of"}


# ---------------------------------------------------------------------------
# Query projection under the fixed status priority
# ---------------------------------------------------------------------------

def classify_status(labels: Iterable[int], *, empty_h: bool, budget_exhausted: bool) -> str:
    labs = frozenset(int(x) for x in labels)
    if empty_h:
        return STATUS_MODEL_CONFLICT
    if budget_exhausted:
        return STATUS_SEARCH_UNRESOLVED
    if labs == frozenset({0}):
        return STATUS_CERTIFIED_DISCONNECTED
    if labs == frozenset({1}):
        return STATUS_CERTIFIED_CONNECTED
    if labs == frozenset({0, 1}):
        return STATUS_WITNESSED_AMBIGUOUS
    raise BaselineContractError("gamma neither empty, {0}, {1} nor {0,1}: %r" % (labs,))


def project_query(hypotheses: Sequence[Mapping], *, max_nodes: int,
                  elapsed_s: float, max_seconds: float) -> dict:
    """Project one PLANAR_UNION_CONNECTIVITY query over H with fixed priority."""
    n = len(hypotheses)
    exhausted = bool(n > int(max_nodes) or float(elapsed_s) > float(max_seconds))
    if exhausted:
        return {"labels": [], "gamma": [], "status": STATUS_SEARCH_UNRESOLVED,
                "n_hypotheses": n, "nodes_expanded": n,
                "search_complete": False,
                "pressure": {"max_nodes": int(max_nodes), "max_seconds": float(max_seconds),
                             "elapsed_s": _f(elapsed_s)}}
    labels = [int(h["label"]) for h in hypotheses]
    status = classify_status(labels, empty_h=(n == 0), budget_exhausted=False)
    return {"labels": labels, "gamma": sorted(set(labels)), "status": status,
            "n_hypotheses": n, "nodes_expanded": n, "search_complete": True,
            "pressure": {"max_nodes": int(max_nodes), "max_seconds": float(max_seconds),
                         "elapsed_s": _f(elapsed_s)}}


def status_reading_fields(status: str) -> dict:
    return {"status": status, "reading_class": READING_DIAGNOSTIC,
            "certificate_label": DIAGNOSTIC_LABEL[status],
            "certification_eligible": "NOT_GRANTED_BY_THIS_UNIT",
            "p2_gate_signed_by_this_unit": False}


__all__ = [
    "HYPOTHESIS_SCOPE", "CANDIDATE_SCOPE", "QUERY_KIND", "READING_DIAGNOSTIC",
    "STATUS_MODEL_CONFLICT", "STATUS_SEARCH_UNRESOLVED",
    "STATUS_CERTIFIED_DISCONNECTED", "STATUS_CERTIFIED_CONNECTED",
    "STATUS_WITNESSED_AMBIGUOUS", "DIAGNOSTIC_LABEL",
    "BaselineContractError", "EmptyObservationError", "ScaleGuardError", "TruthLeakError",
    "canonical_json", "canonical_sha256", "assert_public_only", "assert_public_payload",
    "payload_points", "scale_guard", "median_nn_spacing", "fit_plane", "make_object",
    "public_object", "ransac_planar_family", "region_growing", "ct_components",
    "enumerate_hypotheses", "pair_hit_matrix", "attach_flags", "mask_elements",
    "state_label", "envelope_from_hypotheses", "envelope_connectivity",
    "validate_envelope_reference", "classify_status", "project_query",
    "status_reading_fields",
]
