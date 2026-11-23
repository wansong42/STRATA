# -*- coding: utf-8 -*-
"""Public observation payload for P1-L (G66).

Contract
--------
``public_payload(obs)`` returns a JSON-safe dict that contains **only**
publicly observable quantities of one Observation:

* operator name and operator version,
* number of observed objects,
* noise specification (roughness, noise seed, noise model),
* public measurement geometry (well / scan parameters),
* public observation order key,
* one entry per observed object: public position, post-noise normal, and the
  public record (no ``frac_id``).

Never serialized: ``frac_ids`` / ``truth_sets`` / unobserved arrays /
``dfn_seed`` / any hidden object identifier.  The sha256 of the canonical JSON
bytes is the only cache key.

Canonical form: ``json.dumps(sort_keys=True, ensure_ascii=False,
allow_nan=False, separators=(",", ":"))``; every float is finite-checked,
rounded to 12 decimals and ``-0.0`` is folded to ``0.0``.

This module is new in G66 and does not import ``obs_forward`` (the Observation
is duck-typed), so it can be reused by G63R without a cycle.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SCHEMA_ID = "fractureflow.public_payload.v1"
NOISE_MODEL = "isotropic_tangent_gaussian_v1"

FORBIDDEN_KEYS = frozenset({
    "frac_id", "frac_ids", "truth_sets", "dfn_seed", "hidden_truth",
    "true_normal", "oracle", "world_seed", "full_network",
})

RECORD_WHITELIST = {
    "virtual_borehole": ("depth", "dip", "dipdir", "terzaghi_angle"),
    "virtual_fmi": ("amplitude", "phase_deg", "depth_center"),
    "virtual_outcrop_scan": ("n_points", "chord_length"),
}
MEASUREMENT_WHITELIST = {
    "virtual_borehole": ("x0", "d"),
    "virtual_fmi": ("well_center", "well_axis", "well_radius"),
    "virtual_outcrop_scan": ("scan_z", "bbox_xy", "density"),
}
PUBLIC_ORDER_KEY = {
    "virtual_borehole": "depth_ascending_then_position",
    "virtual_fmi": "depth_center_ascending_then_position",
    "virtual_outcrop_scan": "chord_center_x_then_y_then_chord_length",
}
UNIT_TOL = 1e-6
FLOAT_NDIGITS = 12


class PublicPayloadError(ValueError):
    """Public payload contract violation (explicit rejection)."""


class EmptyObservationError(SystemExit):
    """M=0 / n_observed=0: fail loudly, never emit an empty public payload."""


def _f(x, where):
    v = float(x)
    if not math.isfinite(v):
        raise PublicPayloadError("non-finite float at %s: %r" % (where, x))
    v = round(v, FLOAT_NDIGITS)
    if v == 0.0:
        v = 0.0
    return v


def _arr3(a, where):
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise PublicPayloadError("%s must be a 3-vector, got %r" % (where, arr.shape))
    return [_f(arr[i], "%s[%d]" % (where, i)) for i in range(3)]


def _unit_checked(a, where):
    arr = np.asarray(a, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise PublicPayloadError("non-finite normal at %s" % where)
    n = float(np.linalg.norm(arr))
    if abs(n - 1.0) > UNIT_TOL:
        raise PublicPayloadError("non-unit normal at %s: |n|=%.12g" % (where, n))
    return _arr3(arr, where)


def _record_value(v, where):
    if isinstance(v, (bool, np.bool_)):
        raise PublicPayloadError("boolean not allowed in record at %s" % where)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return _f(v, where)


def _check_key_tree(obj, path="payload"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in FORBIDDEN_KEYS:
                raise PublicPayloadError("forbidden key %r at %s" % (k, path))
            _check_key_tree(v, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_key_tree(v, "%s[%d]" % (path, i))


def public_payload(obs) -> dict:
    """Build the public payload dict from an Observation (duck-typed)."""
    op = str(obs.operator)
    if op not in RECORD_WHITELIST:
        raise PublicPayloadError(
            "operator %r is not covered by public_payload v1 "
            "(virtual_borehole_multi is deferred to G63R)" % (op,))

    mask = np.asarray(obs.obs_mask, dtype=bool).reshape(-1)
    n_obs = int(mask.sum())
    if n_obs == 0:
        raise EmptyObservationError("G66: empty observation (n_observed=0), refusing payload")

    normals = np.asarray(obs.obs_normals, dtype=np.float64)
    positions = np.asarray(obs.obs_positions, dtype=np.float64)
    if normals.shape != (mask.shape[0], 3) or positions.shape != (mask.shape[0], 3):
        raise PublicPayloadError("obs_normals/obs_positions shape mismatch: %r %r"
                                 % (normals.shape, positions.shape))

    frac_ids = np.asarray(obs.frac_ids, dtype=int).reshape(-1)
    if not np.array_equal(frac_ids, np.flatnonzero(mask)):
        raise PublicPayloadError("obs.frac_ids is not flatnonzero(obs_mask)")

    records = list(obs.records or [])
    if len(records) != n_obs:
        raise PublicPayloadError(
            "records length %d != n_observed %d (unobserved or foreign record?)"
            % (len(records), n_obs))
    by_id = {}
    allowed = RECORD_WHITELIST[op]
    for rec in records:
        if not isinstance(rec, dict) or "frac_id" not in rec:
            raise PublicPayloadError("record without frac_id cannot be matched")
        fid = int(rec["frac_id"])
        if fid in by_id:
            raise PublicPayloadError("duplicate record frac_id %d" % fid)
        if fid < 0 or fid >= mask.shape[0] or not bool(mask[fid]):
            raise PublicPayloadError(
                "record frac_id %d is not an observed object (hidden record injection)" % fid)
        extra = set(rec) - set(allowed) - {"frac_id", "well_id"}
        if extra:
            raise PublicPayloadError("record has non-whitelisted keys %r" % sorted(extra))
        out_rec = {}
        for k in allowed:
            if k not in rec:
                raise PublicPayloadError("record missing key %r" % k)
            out_rec[k] = _record_value(rec[k], "records[%d].%s" % (fid, k))
        by_id[fid] = out_rec

    prov = dict(obs.provenance or {})
    measurement = {}
    for k in MEASUREMENT_WHITELIST[op]:
        if k not in prov:
            raise PublicPayloadError("provenance missing public measurement key %r" % k)
        measurement[k] = prov[k]
    measurement = _canonical_measurement(op, measurement)

    roughness = _f(obs.roughness_deg, "roughness_deg")
    if roughness < 0.0:
        raise PublicPayloadError("negative roughness_deg")
    if "noise_seed" not in prov:
        raise PublicPayloadError("provenance missing noise_seed")
    noise_seed = prov["noise_seed"]
    if isinstance(noise_seed, (bool, np.bool_)) or not isinstance(noise_seed, (int, np.integer)):
        raise PublicPayloadError("noise_seed must be an integer")

    entries = []
    for fid in frac_ids.tolist():
        fid = int(fid)
        pos = positions[fid]
        nor = normals[fid]
        if not np.all(np.isfinite(pos)):
            raise PublicPayloadError("non-finite observed position at frac row %d" % fid)
        p = _arr3(pos, "position")
        n = _unit_checked(nor, "normal")
        rec = by_id[fid]
        entries.append({"position": p, "normal": n, "record": rec,
                        "_sort": _sort_key(op, rec, p)})
    entries.sort(key=lambda e: e["_sort"])
    for e in entries:
        del e["_sort"]

    payload = {
        "schema": SCHEMA_ID,
        "operator": op,
        "operator_version": str(prov.get("version", "UNKNOWN")),
        "n_observed": n_obs,
        "noise_spec": {"roughness_deg": roughness,
                       "noise_seed": int(noise_seed),
                       "noise_model": NOISE_MODEL},
        "measurement": measurement,
        "public_order": PUBLIC_ORDER_KEY[op],
        "observations": entries,
    }
    _check_key_tree(payload)
    return payload


def _sort_key(op, rec, pos):
    if op == "virtual_borehole":
        return (float(rec["depth"]), pos[0], pos[1], pos[2])
    if op == "virtual_fmi":
        return (float(rec["depth_center"]), pos[0], pos[1], pos[2])
    return (pos[0], pos[1], float(rec["chord_length"]))


def _canonical_measurement(op, m):
    if op == "virtual_borehole":
        return {"x0": _arr3(m["x0"], "measurement.x0"),
                "d": _unit_checked(m["d"], "measurement.d")}
    if op == "virtual_fmi":
        return {"well_center": _arr3(m["well_center"], "measurement.well_center"),
                "well_axis": _unit_checked(m["well_axis"], "measurement.well_axis"),
                "well_radius": _f(m["well_radius"], "measurement.well_radius")}
    bbox = m["bbox_xy"]
    return {"scan_z": _f(m["scan_z"], "measurement.scan_z"),
            "bbox_xy": [[_f(bbox[0][0], "bbox.xmin"), _f(bbox[0][1], "bbox.xmax")],
                        [_f(bbox[1][0], "bbox.ymin"), _f(bbox[1][1], "bbox.ymax")]],
            "density": _f(m["density"], "measurement.density")}


def public_payload_bytes(obs) -> bytes:
    """Canonical UTF-8 JSON bytes of the public payload."""
    return json.dumps(public_payload(obs), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def public_payload_sha256(obs) -> str:
    return hashlib.sha256(public_payload_bytes(obs)).hexdigest()


def cache_key(obs) -> str:
    """Only cache key source: the public payload sha256. No dfn_seed/frac_id."""
    return public_payload_sha256(obs)
