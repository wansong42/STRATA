# -*- coding: utf-8 -*-
"""G69 · G63R P1 unified observation bundle (E_t) + finite hypothesis envelope.

Contract
--------
* ``build_et(payload, registry_row)`` consumes **only** a G66 public payload
  (:mod:`fractureflow.public_payload`) plus public registry columns.  It never
  reads ``frac_ids`` / ``truth_sets`` / unobserved arrays / ``dfn_seed`` /
  record ``frac_id``; the payload is key-scanned before use.
* Missingness is the six-value enumeration of the P1 execution plan:
  ``NOT_EXPOSED / EXPOSED_NOT_DETECTED / DEVICE_FAILED / OCCLUDED /
  OUT_OF_RESOLUTION / UNKNOWN_MISSING``.  Only ``EXPOSED_NOT_DETECTED``
  belongs to the negative-observation core; every other value is
  *unconstrained* (never "object absent").
* ``compile_hypotheses`` enumerates a finite world set H over the observed
  entries plus the unconstrained missing cells, and derives must/may with
  :func:`fractureflow.identification.envelope_of` (intersection path) and
  re-derives with ``envelope_of_quantifier`` (independent path).  Every H and
  every result carries ``hypothesis_scope = EXACT_FINITE_ENUMERATION`` only;
  no continuous coverage claim is made here.
* Status priority: empty envelope -> ``MODEL_CONFLICT``; ``|H| > 20000`` or
  budget exhausted -> ``SEARCH_UNRESOLVED``; then ``{T}`` -> CERTIFIED_TRUE,
  ``{F}`` -> CERTIFIED_FALSE, ``{T,F}`` -> AMBIGUOUS.  These strings are
  mechanical labels of a finite envelope and are emitted with
  ``issuance = DIAGNOSTIC_ONLY_NOT_A_CERTIFICATE``.

No GPU, no training, no hidden-truth access.  The module is pure Python/numpy
and never imports :mod:`fractureflow.obs_forward`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import public_payload as PP
from . import identification as ID
from . import topology_query as TQ

SCHEMA_ET = "g69.et.v1"
SCHEMA_LEDGER = "g69.exposure_ledger.v1"
SCHEMA_H = "g69.h_envelope.v1"
SCHEMA_MUSTMAY = "g69.must_may.v1"
BUILDER_VERSION = "g69.observation_pipeline.v1"
HYPOTHESIS_SCOPE = "EXACT_FINITE_ENUMERATION"
H_LIMIT = 20000
MAX_FREE_SLOTS = 14  # 2^14 = 16384 <= 20000; 2^15 = 32768 must go SEARCH_UNRESOLVED

MISSINGNESS_VALUES = (
    "NOT_EXPOSED",
    "EXPOSED_NOT_DETECTED",
    "DEVICE_FAILED",
    "OCCLUDED",
    "OUT_OF_RESOLUTION",
    "UNKNOWN_MISSING",
)
NEGATIVE_OBSERVATION_VALUES = frozenset({"EXPOSED_NOT_DETECTED"})
UNCONSTRAINED_MISSING_VALUES = frozenset(MISSINGNESS_VALUES) - NEGATIVE_OBSERVATION_VALUES

STATUS_CERTIFIED_TRUE = "CERTIFIED_TRUE"
STATUS_CERTIFIED_FALSE = "CERTIFIED_FALSE"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_MODEL_CONFLICT = "MODEL_CONFLICT"
STATUS_SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"
STATUSES = (STATUS_CERTIFIED_TRUE, STATUS_CERTIFIED_FALSE, STATUS_AMBIGUOUS,
            STATUS_MODEL_CONFLICT, STATUS_SEARCH_UNRESOLVED)

RELATION_TYPES = ("NODE_PRESENT", "NODE_ABSENT", "EDGE_INTERSECTS",
                  "EDGE_ADJACENT", "ATTACH_A", "ATTACH_B")

ET_FORBIDDEN_KEYS = frozenset(set(PP.FORBIDDEN_KEYS) | {
    "frac_id", "frac_index", "dfn_index", "disc_id", "truth_id", "true_sets",
    "truth_set", "unobserved_normals", "unobserved_positions",
    "unobserved_centers", "hidden_objects", "hidden_side",
})
ET_FORBIDDEN_SUBSTRINGS = ("truth_", "hidden_", "oracle_", "ground_truth",
                           "full_network", "unobserved_")

#: public adapter declaration; every field is instrument/design metadata, never
#: a hidden object property.  Defaults are declared here, not read from truth.
DEFAULT_SUPPORT = {
    "virtual_borehole": {
        "support_kind": "BOREHOLE_DEPTH_CELLS",
        "cell_size": 0.5,
        "tail_unknown_cells": 4,
        "resolution": 0.5,
        "device_failed_offsets": [],
        "occluded_offsets": [],
        "out_of_resolution_offsets": [],
        "override_by_offset": {},
    },
    "virtual_fmi": {
        "support_kind": "FMI_DEPTH_CELLS",
        "cell_size": 0.5,
        "tail_unknown_cells": 4,
        "resolution": 0.5,
        "device_failed_offsets": [],
        "occluded_offsets": [],
        "out_of_resolution_offsets": [],
        "override_by_offset": {},
    },
    "virtual_outcrop_scan": {
        "support_kind": "OUTCROP_GRID",
        "grid": 3,
        "ring_corner_cells": 4,
        "resolution": 0.25,
        "device_failed_offsets": [],
        "occluded_offsets": [],
        "out_of_resolution_offsets": [],
        "override_by_offset": {},
    },
}


class ETPipelineError(ValueError):
    """E_t construction / H compilation contract violation."""


class ETIsolationError(ETPipelineError):
    """Forbidden hidden key or hidden value reached the public E_t."""


def _finite(x, where):
    v = float(x)
    if not math.isfinite(v):
        raise ETPipelineError("non-finite float at %s: %r" % (where, x))
    v = round(v, 12)
    return 0.0 if v == 0.0 else v


def _vec3(x, where):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ETPipelineError("%s must be a 3-vector, got %r" % (where, arr.shape))
    return [_finite(arr[i], "%s[%d]" % (where, i)) for i in range(3)]


def _key_paths(obj, path="et"):
    out = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.extend(_key_paths(v, "%s.%s" % (path, k)))
            out.append((str(k), "%s.%s" % (path, k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_key_paths(v, "%s[%d]" % (path, i)))
    return out


def scan_forbidden(obj) -> list:
    """Recursive forbidden-key scan; returns every hit (empty = clean)."""
    hits = []
    for key, path in _key_paths(obj):
        if key in ET_FORBIDDEN_KEYS:
            hits.append({"key": key, "path": path, "class": "FORBIDDEN_KEY"})
        elif any(s in key.lower() for s in ET_FORBIDDEN_SUBSTRINGS):
            hits.append({"key": key, "path": path, "class": "SOURCE_PATTERN"})
    return hits


def validate_missingness(status) -> str:
    s = str(status)
    if s not in MISSINGNESS_VALUES:
        raise ETPipelineError(
            "missingness value %r not in the six-value enumeration %r"
            % (status, list(MISSINGNESS_VALUES)))
    return s


def missingness_partition(units: Sequence[Mapping[str, Any]]) -> dict:
    """Split units into detection / negative core / unconstrained.

    Only ``EXPOSED_NOT_DETECTED`` may enter the negative core; asking the
    negative core to include anything else raises (no silent widening).
    """
    negative, unconstrained, positive = [], [], []
    for u in units:
        if bool(u.get("detected")):
            positive.append(u["unit_id"])
            continue
        status = validate_missingness(u.get("missing"))
        (negative if status in NEGATIVE_OBSERVATION_VALUES else unconstrained).append(u["unit_id"])
    return {"detected_positive": positive, "negative_core": negative,
            "unconstrained": unconstrained,
            "n_negative_core": len(negative), "n_unconstrained": len(unconstrained)}


def assert_negative_core_only(statuses: Sequence[str]) -> None:
    bad = sorted({str(s) for s in statuses} - NEGATIVE_OBSERVATION_VALUES)
    if bad:
        raise ETPipelineError(
            "negative-observation core may contain only EXPOSED_NOT_DETECTED; got %r" % bad)


def _default_support(payload) -> dict:
    op = str(payload.get("operator"))
    if op not in DEFAULT_SUPPORT:
        raise ETPipelineError("operator %r has no declared public support" % op)
    return json.loads(json.dumps(DEFAULT_SUPPORT[op], allow_nan=False))


def _depth_positions(payload, support):
    m = payload["measurement"]
    if payload["operator"] == "virtual_borehole":
        x0 = np.asarray(m["x0"], dtype=np.float64)
        d = np.asarray(m["d"], dtype=np.float64)
        key = "depth"
        kind = "BOREHOLE_DEPTH_CELLS"
    else:
        x0 = np.asarray(m["well_center"], dtype=np.float64)
        d = np.asarray(m["well_axis"], dtype=np.float64)
        key = "depth_center"
        kind = "FMI_DEPTH_CELLS"
    return x0, d, key, kind


def _build_depth_ledger(payload, support) -> dict:
    x0, axis, key, kind = _depth_positions(payload, support)
    cs = float(support["cell_size"])
    if not (math.isfinite(cs) and cs > 0.0):
        raise ETPipelineError("cell_size must be positive finite")
    depths = [float(o["record"][key]) for o in payload["observations"]]
    if not depths:
        raise ETPipelineError("empty observation ledger: refuse to build E_t")
    i_min = int(math.floor(min(depths) / cs))
    i_max = int(math.ceil(max(depths) / cs))
    tail = int(support.get("tail_unknown_cells", 0))
    resolution = float(support.get("resolution", cs))
    detected_cells = {int(math.floor(d / cs)) for d in depths}
    overrides = {int(k): str(v) for k, v in dict(support.get("override_by_offset") or {}).items()}
    device = {int(x) for x in support.get("device_failed_offsets") or []}
    occluded = {int(x) for x in support.get("occluded_offsets") or []}
    outres = {int(x) for x in support.get("out_of_resolution_offsets") or []}
    units = []
    for off, cell in enumerate(range(i_min, i_max + tail + 1)):
        depth = (cell + 0.5) * cs
        center = x0 + depth * axis
        detected = cell in detected_cells
        if detected:
            missing = None
        elif off in overrides:
            missing = validate_missingness(overrides[off])
        elif off in device:
            missing = "DEVICE_FAILED"
        elif off in occluded:
            missing = "OCCLUDED"
        elif off in outres:
            missing = "OUT_OF_RESOLUTION"
        elif cell > i_max:
            missing = "UNKNOWN_MISSING"
        elif cs < resolution - 1e-12:
            missing = "OUT_OF_RESOLUTION"
        else:
            missing = "EXPOSED_NOT_DETECTED"
        units.append({"unit_id": "u%03d" % off, "index": off, "cell": [cell],
                      "center": _vec3(center, "unit.center"),
                      "detected": bool(detected), "missing": missing,
                      "source": "PUBLIC_MEASUREMENT_GEOMETRY"})
    return {"support_kind": kind, "cell_size": cs, "resolution": resolution,
            "cell_range": [i_min, i_max + tail], "units": units}


def _build_outcrop_ledger(payload, support) -> dict:
    m = payload["measurement"]
    (xmin, xmax), (ymin, ymax) = m["bbox_xy"]
    z = float(m["scan_z"])
    g = int(support["grid"])
    density = float(m["density"])
    cw = (float(xmax) - float(xmin)) / g
    ch = (float(ymax) - float(ymin)) / g
    if not (cw > 0.0 and ch > 0.0):
        raise ETPipelineError("outcrop bbox is degenerate")
    resolution = float(support.get("resolution", 1.0 / density if density > 0 else cw))
    pts = [(float(o["position"][0]), float(o["position"][1])) for o in payload["observations"]]
    if not pts:
        raise ETPipelineError("empty outcrop ledger: refuse to build E_t")
    overrides = {int(k): str(v) for k, v in dict(support.get("override_by_offset") or {}).items()}
    device = {int(x) for x in support.get("device_failed_offsets") or []}
    occluded = {int(x) for x in support.get("occluded_offsets") or []}
    outres = {int(x) for x in support.get("out_of_resolution_offsets") or []}
    units = []
    off = 0
    for i in range(g):
        for j in range(g):
            cx = float(xmin) + (i + 0.5) * cw
            cy = float(ymin) + (j + 0.5) * ch
            det = any(abs(px - cx) <= cw / 2.0 and abs(py - cy) <= ch / 2.0 for px, py in pts)
            if det:
                missing = None
            elif off in overrides:
                missing = validate_missingness(overrides[off])
            elif off in device:
                missing = "DEVICE_FAILED"
            elif off in occluded:
                missing = "OCCLUDED"
            elif off in outres:
                missing = "OUT_OF_RESOLUTION"
            elif min(cw, ch) < resolution - 1e-12:
                missing = "OUT_OF_RESOLUTION"
            else:
                missing = "EXPOSED_NOT_DETECTED"
            units.append({"unit_id": "u%03d" % off, "index": off, "cell": [i, j],
                          "center": _vec3([cx, cy, z], "unit.center"),
                          "detected": bool(det), "missing": missing,
                          "source": "PUBLIC_MEASUREMENT_GEOMETRY"})
            off += 1
    corners = [(-0.5, -0.5), (g - 0.5, -0.5), (-0.5, g - 0.5), (g - 0.5, g - 0.5)]
    for (fi, fj) in corners[: int(support.get("ring_corner_cells", 0))]:
        cx = float(xmin) + (fi + 0.5) * cw
        cy = float(ymin) + (fj + 0.5) * ch
        status = overrides.get(off, "NOT_EXPOSED")
        units.append({"unit_id": "u%03d" % off, "index": off, "cell": [fi, fj],
                      "center": _vec3([cx, cy, z], "unit.center"),
                      "detected": False, "missing": validate_missingness(status),
                      "source": "PUBLIC_MEASUREMENT_GEOMETRY"})
        off += 1
    return {"support_kind": "OUTCROP_GRID", "cell_size": [cw, ch], "resolution": resolution,
            "cell_range": [[xmin, xmax], [ymin, ymax]], "units": units}


def build_exposure_ledger(payload, support=None) -> dict:
    support = dict(support or _default_support(payload))
    op = str(payload["operator"])
    if op in ("virtual_borehole", "virtual_fmi"):
        core = _build_depth_ledger(payload, support)
    elif op == "virtual_outcrop_scan":
        core = _build_outcrop_ledger(payload, support)
    else:
        raise ETPipelineError("operator %r has no exposure ledger" % op)
    units = core["units"]
    counts = {v: 0 for v in MISSINGNESS_VALUES}
    n_detected = 0
    for u in units:
        if u["detected"]:
            n_detected += 1
        else:
            counts[validate_missingness(u["missing"])] += 1
    part = missingness_partition(units)
    return {"schema": SCHEMA_LEDGER, "support_kind": core["support_kind"],
            "cell_size": core["cell_size"], "resolution": core["resolution"],
            "cell_range": core["cell_range"], "n_units": len(units),
            "counts": counts, "n_detected_positive": n_detected,
            "units": units, "missingness_partition": part,
            "rule": ("detected cell = positive record; EXPOSED_NOT_DETECTED is the only "
                     "negative-observation value; every other missing value leaves the "
                     "cell unconstrained and is never read as object absence")}


def validate_payload(payload) -> dict:
    if not isinstance(payload, Mapping):
        raise ETPipelineError("payload must be a Mapping")
    if payload.get("schema") != PP.SCHEMA_ID:
        raise ETPipelineError("payload schema %r != %r" % (payload.get("schema"), PP.SCHEMA_ID))
    hits = scan_forbidden(payload)
    if hits:
        raise ETIsolationError("public payload carries forbidden keys: %r" % hits[:5])
    for key in ("obs_normals", "obs_positions", "obs_mask", "frac_ids", "truth_sets"):
        if key in payload:
            raise ETIsolationError("payload carries hidden observation array %r" % key)
    obs = payload.get("observations")
    if not isinstance(obs, list) or not obs:
        raise ETPipelineError("payload has no observations (empty payload refused)")
    if int(payload.get("n_observed", -1)) != len(obs):
        raise ETPipelineError("n_observed != len(observations)")
    for k, o in enumerate(obs):
        if set(o) - {"position", "normal", "record"}:
            raise ETPipelineError("observation entry %d has extra keys %r" % (k, sorted(o)))
        _vec3(o["position"], "observations[%d].position" % k)
        _vec3(o["normal"], "observations[%d].normal" % k)
        rec = o["record"]
        if not isinstance(rec, Mapping) or "frac_id" in rec:
            raise ETPipelineError("record %d missing or carries frac_id" % k)
        allowed = set(PP.RECORD_WHITELIST[str(payload["operator"])])
        if set(rec) - allowed:
            raise ETPipelineError("record %d has non-whitelisted keys %r" % (k, sorted(set(rec) - allowed)))
    return {"n_observed": len(obs), "operator": str(payload["operator"])}


def build_et(payload, registry_row, support=None) -> dict:
    """Construct E_t from a G66 public payload + public registry row only."""
    info = validate_payload(payload)
    row = registry_row or {}
    for key in ("site_id", "split_group", "license_status", "coordinate_frame", "dataset_id"):
        if key not in row:
            raise ETPipelineError("registry row missing public column %r" % key)
    ledger = build_exposure_ledger(payload, support)
    entries = []
    for k, o in enumerate(payload["observations"]):
        entries.append({"entry_index": k, "position": _vec3(o["position"], "entry.position"),
                        "normal": _vec3(o["normal"], "entry.normal"),
                        "record": dict(o["record"])})
    et = {
        "schema": SCHEMA_ET,
        "site_id": str(row["site_id"]),
        "split_group": str(row["split_group"]),
        "license_status": str(row["license_status"]),
        "frame": {"coordinate_frame": str(row["coordinate_frame"]),
                  "operator": info["operator"],
                  "public_order": str(payload["public_order"]),
                  "position_role": {
                      "virtual_borehole": "borehole-disc intersection point",
                      "virtual_fmi": "FMI trace centre on the well axis",
                      "virtual_outcrop_scan": "scan-plane chord centre",
                  }[info["operator"]]},
        "points": [e["position"] for e in entries],
        "normals": [e["normal"] for e in entries],
        "traces": [dict(e["record"]) for e in entries],
        "entries": entries,
        "masks": ledger,
        "provenance": {
            "payload_schema": str(payload["schema"]),
            "payload_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "operator": info["operator"],
            "operator_version": str(payload.get("operator_version")),
            "noise_spec": dict(payload["noise_spec"]),
            "measurement": json.loads(json.dumps(payload["measurement"], allow_nan=False)),
            "registry_dataset_id": str(row["dataset_id"]),
            "registry_license": str(row.get("license")),
            "builder": BUILDER_VERSION,
            "support_declaration": json.loads(json.dumps(ledger["support_kind"], allow_nan=False)),
            "model_assumptions": [
                "radius_model is a declared public model parameter, not a measured radius",
                "support cells come from public measurement geometry; no instrument "
                "failure/occlusion log exists in public_payload v1",
            ],
        },
    }
    et["content_sha256"] = et_content_sha(et)
    hits = scan_forbidden(et)
    if hits:
        raise ETIsolationError("E_t carries forbidden keys: %r" % hits[:5])
    return et


def et_content_sha(et) -> str:
    body = {k: v for k, v in et.items() if k != "content_sha256"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_et_bytes(et) -> bytes:
    return json.dumps(et, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def et_from_json(text: str) -> dict:
    doc = json.loads(text)
    hits = scan_forbidden(doc)
    if hits:
        raise ETIsolationError("E_t JSON carries forbidden keys: %r" % hits[:5])
    if doc.get("content_sha256") != et_content_sha(doc):
        raise ETIsolationError("E_t content_sha256 mismatch on roundtrip")
    return doc


# ---------------------------------------------------------------------------
# finite hypothesis set H -> must/may
# ---------------------------------------------------------------------------

def default_query_for_et(et, radius_model: float = 0.5) -> dict:
    """Public query: connect the first and last observed entry centres."""
    if len(et["entries"]) < 2:
        raise ETPipelineError("at least two observed entries are required for a two-point query")
    A = list(et["entries"][0]["position"])
    B = list(et["entries"][-1]["position"])
    return {"A": A, "B": B, "r_attach": 0.0, "radius_model": float(radius_model),
            "adjacency_gap": 0.1}


def build_universe(et, radius_model: float = 0.5) -> dict:
    """Universe = observed entries (pinned present) + negative core (pinned absent)
    + unconstrained missing cells (free bits).  Negative core never becomes free."""
    ledger = et["masks"]
    part = ledger["missingness_partition"]
    entries = et["entries"]
    normals = np.asarray([e["normal"] for e in entries], dtype=np.float64)
    objects, observed_ids, negative_ids, free_ids, records = [], [], [], [], []
    for k, e in enumerate(entries):
        oid = "o%03d" % k
        observed_ids.append(oid)
        objects.append({"id": oid, "type": "DISK", "center": list(e["position"]),
                        "normal": list(e["normal"]),
                        "radius": float(radius_model), "role": "OBSERVED",
                        "source": "PUBLIC_PAYLOAD"})
        records.append({"object_id": oid, "present": True})
    units = {u["unit_id"]: u for u in ledger["units"]}
    for k, uid in enumerate(part["negative_core"]):
        u = units[uid]
        oid = "n%03d" % k
        negative_ids.append(oid)
        objects.append({"id": oid, "type": "DISK", "center": list(u["center"]),
                        "normal": _free_normal(u["center"], entries, normals),
                        "radius": float(radius_model), "role": "NEGATIVE_CORE",
                        "source": "PUBLIC_EXPOSURE_LEDGER"})
        records.append({"object_id": oid, "present": False})
    for k, uid in enumerate(part["unconstrained"]):
        u = units[uid]
        oid = "f%03d" % k
        free_ids.append(oid)
        objects.append({"id": oid, "type": "DISK", "center": list(u["center"]),
                        "normal": _free_normal(u["center"], entries, normals),
                        "radius": float(radius_model), "role": "UNCONSTRAINED_MISSING",
                        "source": "PUBLIC_EXPOSURE_LEDGER"})
    return {"objects": objects, "observed_ids": observed_ids,
            "negative_ids": negative_ids, "free_ids": free_ids,
            "records": records, "radius_model": float(radius_model),
            "n_universe": len(objects), "n_free": len(free_ids),
            "negative_core_rule": "pinned_absent",
            "unconstrained_rule": "free_bit_never_absent_by_default"}


def _free_normal(center, entries, normals):
    if not entries:
        return [0.0, 0.0, 1.0]
    c = np.asarray(center, dtype=np.float64)
    best, best_d = 0, float("inf")
    for k, e in enumerate(entries):
        d = float(np.linalg.norm(c - np.asarray(e["position"], dtype=np.float64)))
        if d < best_d:
            best, best_d = k, d
    n = normals[best] / max(float(np.linalg.norm(normals[best])), 1e-12)
    return [_finite(n[i], "free_normal") for i in range(3)]


def derive_status(*, empty: bool, n_hypotheses: int, h_limit: int = H_LIMIT,
                  budget_exhausted: bool = False, gamma: Sequence[int]) -> str:
    """Mechanical priority: empty > (over-limit | budget) > gamma singleton/pair."""
    if empty:
        return STATUS_MODEL_CONFLICT
    if int(n_hypotheses) > int(h_limit) or budget_exhausted:
        return STATUS_SEARCH_UNRESOLVED
    g = sorted({int(x) for x in gamma})
    if g == [1]:
        return STATUS_CERTIFIED_TRUE
    if g == [0]:
        return STATUS_CERTIFIED_FALSE
    if g == [0, 1]:
        return STATUS_AMBIGUOUS
    raise ETPipelineError("gamma %r has no five-state mapping" % (g,))


def _mask_for(universe, free_subset) -> int:
    idx = {o["id"]: i for i, o in enumerate(universe["objects"])}
    mask = 0
    for oid in universe["observed_ids"]:
        mask |= 1 << idx[oid]
    for oid in free_subset:
        mask |= 1 << idx[oid]
    return mask


def _derivation_from_structure(structure, query, state_id: int):
    objects = structure["objects"]
    nodes = frozenset(str(o["id"]) for o in objects)
    edges = frozenset((str(r["src"]), str(r["dst"]))
                      for r in structure["relations"] if r["kind"] == "intersects")
    c = np.asarray([o["center"] for o in objects], dtype=np.float64).reshape(-1, 3)
    n = np.asarray([o["normal"] for o in objects], dtype=np.float64).reshape(-1, 3)
    r = np.asarray([o["radius"] for o in objects], dtype=np.float64).reshape(-1)
    from .connectivity_qoi import disk_point_distance
    a_side, b_side = frozenset(), frozenset()
    if len(objects):
        da = disk_point_distance(c, n, r, np.asarray(query["A"], dtype=np.float64).reshape(1, 3))[0]
        db = disk_point_distance(c, n, r, np.asarray(query["B"], dtype=np.float64).reshape(1, 3))[0]
        a_side = frozenset(str(objects[i]["id"]) for i in range(len(objects))
                           if float(da[i]) <= float(query["r_attach"]))
        b_side = frozenset(str(objects[i]["id"]) for i in range(len(objects))
                           if float(db[i]) <= float(query["r_attach"]))
    label = 1 if TQ.api_evaluate(structure, TQ.TopologyQuery(
        "q", "TWO_POINT_CONNECTIVITY",
        {"A": query["A"], "B": query["B"], "r_attach": query["r_attach"]})) else 0
    return ID.StateDerivation(state_id=state_id, nodes=nodes, edges=edges,
                              a_side=a_side, b_side=b_side,
                              attach_a=bool(a_side), attach_b=bool(b_side), label=label)


def compile_hypotheses(universe: Mapping[str, Any], query: Mapping[str, Any], *,
                       h_limit: int = H_LIMIT, max_nodes_budget: int | None = None,
                       forced_empty: bool = False) -> dict:
    """Exact finite enumeration over free bits; must/may derived per hypothesis."""
    n_free = int(universe["n_free"])
    n_h = 1 << n_free if not forced_empty else 0
    base = {"schema": SCHEMA_H, "hypothesis_scope": HYPOTHESIS_SCOPE,
            "n_universe": int(universe["n_universe"]), "n_free": n_free,
            "n_hypotheses": int(n_h), "h_limit": int(h_limit),
            "budget": {"max_nodes": max_nodes_budget,
                       "node_unit": "FEASIBLE_STRUCTURE_EXPANDED"},
            "search_truncated": False,
            "issuance": "DIAGNOSTIC_ONLY_NOT_A_CERTIFICATE"}
    if forced_empty:
        status = derive_status(empty=True, n_hypotheses=0, h_limit=h_limit, gamma=[])
        return dict(base, status=status, empty=True, enumerated=0, gamma=[],
                    reason="constructed observation conflict: feasible set is empty",
                    must_may=None, states=[])
    if n_h > int(h_limit):
        status = derive_status(empty=False, n_hypotheses=n_h, h_limit=h_limit, gamma=[])
        return dict(base, status=status, empty=False, enumerated=0, gamma=[],
                    reason=("|H|=%d exceeds limit %d; enumeration refused, not truncated"
                            % (n_h, h_limit)),
                    must_may=None, states=[])
    budget = n_h if max_nodes_budget is None else int(max_nodes_budget)
    if budget < 0:
        raise ETPipelineError("max_nodes_budget must be >= 0")
    budget_exhausted = n_h > budget
    labels, states, structures = [], [], []
    free_ids = list(universe["free_ids"])
    enumerated = 0
    for r in range(n_free + 1):
        for subset in itertools.combinations(free_ids, r):
            if enumerated >= budget:
                break
            mask = _mask_for(universe, subset)
            structure = TQ.structure_from_mask("H", universe["objects"], mask,
                                               float(query["adjacency_gap"]))
            states.append(_derivation_from_structure(structure, query, len(states)))
            labels.append(1 if states[-1].label else 0)
            structures.append(structure)
            enumerated += 1
        if enumerated >= budget:
            break
    if budget_exhausted:
        status = derive_status(empty=False, n_hypotheses=n_h, h_limit=h_limit,
                               budget_exhausted=True, gamma=labels)
        return dict(base, status=status, empty=False, enumerated=enumerated,
                    gamma=sorted(set(labels)),
                    reason="budget exhausted (%d of %d structures expanded)" % (enumerated, n_h),
                    must_may=None, states=[s.to_dict() for s in states],
                    search_truncated=True,
                    budget={"max_nodes": budget, "node_unit": "FEASIBLE_STRUCTURE_EXPANDED",
                            "exhausted": True})
    env = ID.envelope_of(states)
    env2 = ID.envelope_of_quantifier(states)
    if env.to_dict() != env2.to_dict():
        raise ETPipelineError("envelope_of and envelope_of_quantifier disagree")
    must_may = _must_may_doc(universe, structures, states, env, query)
    status = derive_status(empty=False, n_hypotheses=n_h, h_limit=h_limit,
                           gamma=sorted(set(labels)))
    return dict(base, status=status, empty=False, enumerated=enumerated,
                gamma=sorted(set(labels)),
                reason="complete enumeration of %d finite hypotheses" % n_h,
                must_may=must_may, states=[s.to_dict() for s in states],
                envelope=env.to_dict())


def _must_may_doc(universe, structures, states, env, query) -> dict:
    must_nodes = set(env.must_nodes)
    may_nodes = set(env.may_nodes)
    must_absent = {o["id"] for o in universe["objects"] if o["id"] not in may_nodes}
    may_absent = {o["id"] for o in universe["objects"] if o["id"] not in must_nodes}
    must_edges = sorted([min(e), max(e), "intersects"] for e in env.must_edges)
    may_edges = sorted([min(e), max(e), "intersects"] for e in env.may_edges)
    adj_must, adj_may = None, set()
    for st in structures:
        rel = {(min(r["src"], r["dst"]), max(r["src"], r["dst"]))
               for r in st["relations"] if r["kind"] == "adjacent"}
        adj_may |= rel
        adj_must = rel if adj_must is None else (adj_must & rel)
    adj_must = adj_must or set()
    must_edges += sorted([u, v, "adjacent"] for u, v in adj_must)
    may_edges += sorted([u, v, "adjacent"] for u, v in adj_may)
    checks = _check_must_may(must_nodes, may_nodes, must_absent, may_absent,
                             env, must_edges, may_edges)
    return {
        "schema": SCHEMA_MUSTMAY,
        "relation_types": list(RELATION_TYPES),
        "derivation": {"source": "PER_HYPOTHESIS_ENUMERATION",
                       "n_states_used": len(states),
                       "intersection_path": "identification.envelope_of",
                       "verification_path": "identification.envelope_of_quantifier",
                       "independent_paths_equal": True},
        "must": {"nodes": sorted(must_nodes), "absent": sorted(must_absent),
                 "edges": must_edges, "a_side": sorted(env.must_a_side),
                 "b_side": sorted(env.must_b_side),
                 "attach_a": bool(env.must_attach_a), "attach_b": bool(env.must_attach_b)},
        "may": {"nodes": sorted(may_nodes), "absent": sorted(may_absent),
                "edges": may_edges, "a_side": sorted(env.may_a_side),
                "b_side": sorted(env.may_b_side),
                "attach_a": bool(env.may_attach_a), "attach_b": bool(env.may_attach_b)},
        "region_label": ID.region_label(env),
        "must_connected": ID.must_connected(env),
        "may_connected": ID.may_connected(env),
        "checks": checks,
        "query": {"A": list(query["A"]), "B": list(query["B"]),
                  "r_attach": float(query["r_attach"]),
                  "adjacency_gap": float(query["adjacency_gap"])},
        "hypothesis_scope": HYPOTHESIS_SCOPE,
        "may_path_is_not_a_physical_network": True,
    }


def _check_must_may(must_nodes, may_nodes, must_absent, may_absent, env,
                    must_edges, may_edges) -> dict:
    errs = []
    if not must_nodes <= may_nodes:
        errs.append("must_nodes_not_subset_may_nodes")
    if must_nodes & must_absent:
        errs.append("node_both_must_present_and_must_absent")
    if not set(env.must_a_side) <= may_nodes:
        errs.append("must_a_side_outside_may_nodes")
    if not set(env.must_b_side) <= may_nodes:
        errs.append("must_b_side_outside_may_nodes")
    may_edge_set = {(u, v, k) for u, v, k in may_edges}
    for u, v, k in must_edges:
        if (u, v, k) not in may_edge_set:
            errs.append("must_edge_%s_not_in_may:%s-%s" % (k, u, v))
        if u not in must_nodes or v not in must_nodes:
            errs.append("must_edge_endpoint_missing:%s-%s" % (u, v))
    for u, v, k in may_edges:
        if k not in ("intersects", "adjacent"):
            errs.append("illegal_relation_kind:%s" % k)
        if u not in may_nodes or v not in may_nodes:
            errs.append("may_edge_endpoint_missing:%s-%s" % (u, v))
    if env.must_attach_a and not env.may_attach_a:
        errs.append("attach_a_must_but_not_may")
    if env.must_attach_b and not env.may_attach_b:
        errs.append("attach_b_must_but_not_may")
    if ID.must_connected(env) and not ID.may_connected(env):
        errs.append("must_connected_but_may_disconnected")
    return {"ok": not errs, "errors": errs,
            "must_subset_may": not errs,
            "relation_types_legal": not any(e.startswith("illegal_") for e in errs),
            "endpoints_exist": not any("endpoint_missing" in e for e in errs)}


def must_may_artifact_ok(doc: Mapping[str, Any]) -> dict:
    """Independent checker for must⊆may + endpoint existence on a serialized doc."""
    errs = []
    mn = set(doc["must"]["nodes"])
    yn = set(doc["may"]["nodes"])
    if not mn <= yn:
        errs.append("must_nodes_not_subset_may_nodes")
    must_edges = {(u, v, k) for u, v, k in doc["must"]["edges"]}
    may_edges = {(u, v, k) for u, v, k in doc["may"]["edges"]}
    if not must_edges <= may_edges:
        errs.append("must_edges_not_subset_may_edges")
    for u, v, k in doc["must"]["edges"]:
        if u not in mn or v not in mn:
            errs.append("must_edge_endpoint_missing")
        if k not in ("intersects", "adjacent"):
            errs.append("illegal_relation_kind")
    for u, v, k in doc["may"]["edges"]:
        if u not in yn or v not in yn:
            errs.append("may_edge_endpoint_missing")
    if not set(doc["must"]["a_side"]) <= yn or not set(doc["must"]["b_side"]) <= yn:
        errs.append("must_attachment_outside_may_nodes")
    return {"ok": not errs, "errors": errs}


def hypotheses_derivation_check(doc: Mapping[str, Any], states: Sequence[Mapping[str, Any]]) -> dict:
    """Verify every must element appears in ALL enumerated states (H is authoritative)."""
    if not states:
        return {"ok": False, "errors": ["no_states"]}
    errs = []
    node_sets = [set(s["nodes"]) for s in states]
    must_nodes = set(doc["must"]["nodes"])
    for n in must_nodes:
        if not all(n in ns for ns in node_sets):
            errs.append("must_node_not_in_all_states:%s" % n)
    for u, v, k in doc["must"]["edges"]:
        if k != "intersects":
            continue
        for s in states:
            se = {frozenset((a, b)) for a, b in s["edges"]}
            if frozenset((u, v)) not in se:
                errs.append("must_edge_not_in_all_states:%s-%s" % (u, v))
                break
    return {"ok": not errs, "errors": errs, "n_states": len(states)}
