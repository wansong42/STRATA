# -*- coding: utf-8 -*-
"""G73 · P2 integration chain on EXACT_FINITE_ENUMERATION / PLANAR_UNION_CONNECTIVITY.

``E_t -> HypothesisSet H (exact finite) -> Gamma_Q -> must/may -> status ->
certificate/witness -> budget``.  The module is the production path; the
independent replay lives in ``scripts/g73_replay.py`` and shares no code with
it (separate process, separate geometry implementation).

Boundaries
----------
* ``hypothesis_scope`` is ``EXACT_FINITE_ENUMERATION`` only.  The domain is a
  declared finite object inventory (or an observation ledger); no continuous
  parameter space is covered and no candidate-set enumeration is promoted to H.
* Only ``PLANAR_UNION_CONNECTIVITY`` is consumed.  Every query passes
  ``identification.require_supported_query_semantics``; other semantics raise
  ``UnsupportedQuerySemantics`` and never reach the geometry kernel.
* Single-value labels are mechanical finite-envelope readings.  Certificates
  are ``DIAGNOSTIC_CERTIFICATE`` because no P2 gate is signed by this unit.
* A may path is never a witness.  ``witness_from_may_path`` refuses.
* CPU only, no GPU, no training, no subprocess, no shell.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from . import identification as ID
from . import observation_pipeline as OP
from .connectivity_qoi import disk_point_distance
from .intersect_disk import disk_pair_intersect

SCHEMA_CASE = "g73.p2_case.v1"
SCHEMA_RESULT = "g73.p2_case_result.v1"
SCHEMA_CERT = "g73.p2_diagnostic_certificate.v1"
SCHEMA_MUSTMAY = "g73.must_may.v1"

HYPOTHESIS_SCOPE = "EXACT_FINITE_ENUMERATION"
CERTIFIABLE_SCOPES = frozenset({HYPOTHESIS_SCOPE})
H_LIMIT = 20000
MAX_FREE_SLOTS = 14
DEFAULT_MAX_SECONDS = 7200.0
ISSUANCE = "DIAGNOSTIC_ONLY_NOT_A_CERTIFICATE"
CERTIFICATE_CLASS = "DIAGNOSTIC_CERTIFICATE"

STATUS_MODEL_CONFLICT = "MODEL_CONFLICT"
STATUS_SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"
STATUS_CERTIFIED_DISCONNECTED = "CERTIFIED_DISCONNECTED"
STATUS_CERTIFIED_CONNECTED = "CERTIFIED_CONNECTED"
STATUS_WITNESSED_AMBIGUOUS = "WITNESSED_AMBIGUOUS"
SINGLE_VALUE = {
    STATUS_CERTIFIED_DISCONNECTED: 0,
    STATUS_CERTIFIED_CONNECTED: 1,
}
STATUS_OF_GAMMA = {
    (0,): STATUS_CERTIFIED_DISCONNECTED,
    (1,): STATUS_CERTIFIED_CONNECTED,
    (0, 1): STATUS_WITNESSED_AMBIGUOUS,
}

INTERSECTION_CONVENTION = (
    "connectivity_qoi two_point_connected: adjacent discs true-intersect "
    "(intersect_disk.disk_pair_intersect, strict inequality)")
ATTACHMENT_CONVENTION = (
    "endpoint captured by disc iff point-to-disc distance <= r_attach (inclusive)")

FORBIDDEN_TRUTH_KEYS = frozenset({
    "truth", "truth_mask", "truth_sets", "truth_set", "hidden_truth",
    "hidden_objects", "hidden_side", "ground_truth", "present_truth",
    "true_normal", "true_normals", "true_sets", "full_network", "oracle",
    "dfn_seed", "frac_id", "frac_ids", "world_seed", "unobserved_normals",
    "unobserved_positions", "unobserved_centers", "centers_true",
    "normals_true", "radii_true", "true_world",
})
FORBIDDEN_TRUTH_SUBSTRINGS = ("truth_", "hidden_", "oracle_", "ground_truth",
                              "unobserved_", "full_network")


class P2ContractError(ValueError):
    """Contract violation: refuse to run, never degrade silently."""


class P2TruthLeakError(P2ContractError):
    """A hidden-truth key reached the production input."""


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _f(x) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise P2ContractError("non-finite float: %r" % (x,))
    v = round(v, 12)
    return 0.0 if v == 0.0 else v


def assert_no_truth(obj, path: str = "input") -> int:
    """Recursive hidden-truth key scan; returns the number of scanned keys."""
    scanned = 0
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            scanned += 1
            key = str(k)
            if key in FORBIDDEN_TRUTH_KEYS:
                raise P2TruthLeakError("forbidden truth key %r at %s" % (key, path))
            low = key.lower()
            if any(s in low for s in FORBIDDEN_TRUTH_SUBSTRINGS):
                raise P2TruthLeakError("truth-like key %r at %s" % (key, path))
            scanned += assert_no_truth(v, path + "." + key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            scanned += assert_no_truth(v, "%s[%d]" % (path, i))
    return scanned


def make_query_spec(query: Mapping) -> ID.QuerySpec:
    kind = str(query.get("query_kind", ""))
    if kind != ID.SUPPORTED_QUERY_KIND:
        raise ID.UnsupportedQuerySemantics(
            "query_kind", kind, "only PLANAR_UNION_CONNECTIVITY is supported")
    spec = ID.QuerySpec(
        query_id=str(query["query_id"]),
        A=tuple(float(x) for x in query["A"]),
        B=tuple(float(x) for x in query["B"]),
        target_domain=str(query["target_domain"]),
        r_attach=float(query["r_attach"]),
        intersection_convention=str(query["intersection_convention"]),
        attachment_convention=str(query["attachment_convention"]),
        endpoint_form=str(query.get("endpoint_form", ID.WORLD_COORDINATE)),
    )
    ID.require_supported_query_semantics(spec)
    return spec


def cube_domain(points: Sequence, margin: float = 0.0) -> str:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = float(arr.min()) - float(margin)
    hi = float(arr.max()) + float(margin)
    return "WORLD_COORDINATE box [%.12g,%.12g]^3" % (lo, hi)


def _objects_sorted(objects: Sequence[Mapping]) -> List[dict]:
    out = []
    for o in objects:
        if str(o.get("type", "DISK")) != "DISK":
            raise P2ContractError("only DISK objects enter the finite universe")
        out.append({"id": str(o["id"]), "type": "DISK",
                    "center": [_f(x) for x in o["center"]],
                    "normal": [_f(x) for x in o["normal"]],
                    "radius": _f(o["radius"])})
    out.sort(key=lambda o: o["id"])
    if len({o["id"] for o in out}) != len(out):
        raise P2ContractError("duplicate object id in universe")
    return out


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 0.0:
        raise P2ContractError("degenerate vector")
    return a / n


def _match_entries_to_objects(objects, entries, scale):
    tol_p = 1e-6 * max(float(scale), 1e-9)
    tol_c = 1e-6 * max(float(scale), 1e-9)
    tol_a = math.radians(15.0)
    matches, issues = {}, []
    for k, e in enumerate(entries):
        P = np.asarray(e["position"], dtype=np.float64).reshape(3)
        ne = _unit(e["normal"])
        scored = []
        for o in objects:
            C = np.asarray(o["center"], dtype=np.float64)
            n = _unit(o["normal"])
            pd = abs(float((P - C) @ n))
            ang = math.acos(float(np.clip(abs(ne @ n), -1.0, 1.0)))
            contain = float(disk_point_distance(
                np.asarray([o["center"]], dtype=np.float64),
                np.asarray([o["normal"]], dtype=np.float64),
                np.asarray([o["radius"]], dtype=np.float64),
                P.reshape(1, 3))[0, 0])
            if pd <= tol_p and contain <= tol_c and ang <= tol_a:
                scored.append((pd / tol_p + ang / tol_a, pd, ang, contain, o["id"]))
        if not scored:
            issues.append({"class": "UNMATCHED_OBSERVATION", "entry_index": k})
            continue
        scored.sort()
        best = scored[0]
        if len(scored) > 1 and (scored[1][0] - best[0]) <= 1e-9:
            issues.append({"class": "AMBIGUOUS_OBSERVATION_MATCH", "entry_index": k,
                           "candidates": [s[4] for s in scored if abs(s[0] - best[0]) <= 1e-9]})
            continue
        oid = best[4]
        if oid in matches:
            issues.append({"class": "DUPLICATE_OBSERVATION_MATCH", "entry_index": k,
                           "object_id": oid})
            continue
        matches[oid] = {"entry_index": k, "plane_offset": best[1],
                        "angle_rad": best[2], "contain_distance": best[3]}
    return matches, issues


def derive_inventory_pins(case: Mapping) -> dict:
    """E_t series + declared finite inventory -> PRESENT / ABSENT / FREE.

    Each public ``virtual_borehole`` E_t contributes detections and a negative
    core over its exposed depth window.  Evidence is intersected across the
    series: any detection pins PRESENT, any exposed-not-detected crossing pins
    ABSENT, everything else stays FREE.  A present/absent conflict is a model
    conflict, never a silent override.
    """
    objects = _objects_sorted(case["inventory"])
    series = list(case.get("et_series") or [])
    if not series:
        raise P2ContractError("inventory case has an empty E_t series")
    det_ids, abs_ids = set(), set()
    issues, match_tables, detail = [], [], []
    for wi, item in enumerate(series):
        et, well = item["et"], item["well"]
        x0 = np.asarray(well["x0"], dtype=np.float64).reshape(3)
        d = _unit(well["d"])
        coords = [o["center"] for o in objects] + [list(e["position"]) for e in et["entries"]]
        arr = np.asarray(coords, dtype=np.float64)
        scale = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
        matches, m_issues = _match_entries_to_objects(objects, et["entries"], scale)
        for it in m_issues:
            issues.append(dict(it, well_index=wi))
        match_tables.append({"well_index": wi, "matches": matches})
        det_ids |= set(matches)
        masks = et["masks"]
        cs = float(masks["cell_size"])
        i_min, i_max = int(masks["cell_range"][0]), int(masks["cell_range"][1])
        for o in objects:
            C = np.asarray(o["center"], dtype=np.float64)
            n = _unit(o["normal"])
            nd = float(n @ d)
            if abs(nd) < 1e-12:
                continue
            t = float(n @ (C - x0) / nd)
            if t < 0.0:
                continue
            P = x0 + t * d
            if float(np.linalg.norm(P - C)) > float(o["radius"]):
                continue
            cell = int(math.floor((t + 1e-9) / cs))
            if not (i_min <= cell <= i_max):
                continue
            if o["id"] not in matches:
                abs_ids.add(o["id"])
                detail.append({"object_id": o["id"], "pin": "ABSENT",
                               "reason": "EXPOSED_NOT_DETECTED", "well_index": wi,
                               "cell": cell, "depth": _f(t)})
    conflict = sorted(det_ids & abs_ids)
    for o in objects:
        if o["id"] in det_ids:
            detail.append({"object_id": o["id"], "pin": "PRESENT",
                           "reason": "DETECTED_AND_MATCHED"})
    pins = {}
    for o in objects:
        if o["id"] in det_ids:
            pins[o["id"]] = "PRESENT"
        elif o["id"] in abs_ids:
            pins[o["id"]] = "ABSENT"
        else:
            pins[o["id"]] = "FREE"
    present = sorted([k for k, v in pins.items() if v == "PRESENT"])
    absent = sorted([k for k, v in pins.items() if v == "ABSENT"])
    free = sorted([k for k, v in pins.items() if v == "FREE"])
    return {"objects": objects, "pins": pins, "present_ids": present,
            "absent_ids": absent, "free_ids": free,
            "match_table": match_tables, "issues": issues, "detail": detail,
            "n_universe": len(objects), "n_free": len(free),
            "contradiction_ids": conflict,
            "universe_source": case.get("universe_source", "DECLARED_INVENTORY")}


def derive_ledger_pins(case: Mapping) -> dict:
    """G69 E_t -> ledger universe (observed present / negative absent / free)."""
    et = case["et"]
    u = OP.build_universe(et, radius_model=float(case.get("radius_model", 0.5)))
    objects = _objects_sorted(u["objects"])
    by_id = {o["id"]: o for o in objects}
    present = sorted([i for i in u["observed_ids"] if i in by_id])
    absent = sorted([i for i in u["negative_ids"] if i in by_id])
    free = sorted([i for i in u["free_ids"] if i in by_id])
    return {"objects": objects, "pins": {o["id"]: ("PRESENT" if o["id"] in present
                                                    else "ABSENT" if o["id"] in absent
                                                    else "FREE") for o in objects},
            "present_ids": present, "absent_ids": absent, "free_ids": free,
            "match_table": {}, "issues": [], "detail": [],
            "n_universe": len(objects), "n_free": len(free),
            "universe_source": "OBSERVATION_LEDGER_G69"}


def derive_explicit_pins(case: Mapping) -> dict:
    """Constructed control case: explicit objects + pins (not E_t derived)."""
    objects = _objects_sorted(case["objects"])
    pins = {str(k): str(v) for k, v in case["pins"].items()}
    missing = sorted(set(o["id"] for o in objects) - set(pins))
    if missing:
        raise P2ContractError("constructed case missing pins for %r" % missing)
    present = sorted([k for k, v in pins.items() if v == "PRESENT"])
    absent = sorted([k for k, v in pins.items() if v == "ABSENT"])
    free = sorted([k for k, v in pins.items() if v == "FREE"])
    return {"objects": objects, "pins": pins, "present_ids": present,
            "absent_ids": absent, "free_ids": free, "match_table": {},
            "issues": [], "detail": [], "n_universe": len(objects),
            "n_free": len(free), "universe_source": "CONSTRUCTED_CONTROL"}


def enumerate_hypotheses(universe: Mapping, budget: Mapping) -> dict:
    """All presence masks over the free bits; no truncation without a flag."""
    present = list(universe["present_ids"])
    absent = list(universe["absent_ids"])
    free = list(universe["free_ids"])
    max_nodes = int(budget.get("max_nodes", H_LIMIT))
    max_seconds = float(budget.get("max_seconds", DEFAULT_MAX_SECONDS))
    forced_empty = bool(budget.get("forced_empty", False))
    ids = [o["id"] for o in universe["objects"]]
    index = {oid: i for i, oid in enumerate(ids)}
    base = 0
    for oid in present:
        base |= 1 << index[oid]
    overlap = sorted(set(present) & set(absent))
    n_free = len(free)
    n_h = 0 if (forced_empty or overlap) else (1 << n_free)
    doc = {"n_universe": len(ids), "n_free": n_free, "n_hypotheses": int(n_h),
           "h_limit": H_LIMIT, "max_nodes": max_nodes,
           "max_seconds": _f(max_seconds), "hypothesis_scope": HYPOTHESIS_SCOPE,
           "enumeration_order": "ASCENDING_FREE_BITMASK",
           "pinned_present": present, "pinned_absent": absent, "free_ids": free,
           "contradiction": overlap}
    if forced_empty or overlap:
        return dict(doc, states=[], enumerated=0, search_complete=False,
                    stop_reason="MODEL_CONFLICT", budget_exhausted=False,
                    elapsed_s=0.0)
    if n_h > H_LIMIT:
        return dict(doc, states=[], enumerated=0, search_complete=False,
                    stop_reason="H_LIMIT_EXCEEDED", budget_exhausted=False,
                    elapsed_s=0.0)
    if max_seconds <= 0.0:
        return dict(doc, states=[], enumerated=0, search_complete=False,
                    stop_reason="TIME_LIMIT", budget_exhausted=False,
                    elapsed_s=0.0)
    t0 = time.perf_counter()
    states = []
    stopped = "COMPLETE"
    for i in range(n_h):
        if len(states) >= max_nodes:
            stopped = "MAX_NODES_EXCEEDED"
            break
        if (i & 63) == 0 and time.perf_counter() - t0 > max_seconds:
            stopped = "TIME_LIMIT"
            break
        mask = base
        for b, oid in enumerate(free):
            if (i >> b) & 1:
                mask |= 1 << index[oid]
        states.append({"state_id": len(states), "mask": int(mask),
                       "free_bits": int(i),
                       "present_ids": [oid for oid in ids if (mask >> index[oid]) & 1]})
    elapsed = time.perf_counter() - t0
    return dict(doc, states=states, enumerated=len(states),
                search_complete=(stopped == "COMPLETE"),
                stop_reason=stopped,
                budget_exhausted=(stopped == "MAX_NODES_EXCEEDED"),
                elapsed_s=_f(elapsed))


def _pair_matrix(objects) -> np.ndarray:
    k = len(objects)
    if k < 2:
        return np.zeros((k, k), dtype=bool)
    c = np.asarray([o["center"] for o in objects], dtype=np.float64)
    n = np.asarray([o["normal"] for o in objects], dtype=np.float64)
    r = np.asarray([o["radius"] for o in objects], dtype=np.float64)
    i_idx, j_idx = np.triu_indices(k, 1)
    hit = disk_pair_intersect(c, n, r, i_idx, j_idx)
    m = np.zeros((k, k), dtype=bool)
    m[i_idx[hit], j_idx[hit]] = True
    m = m | m.T
    return m


def _attach_sets(objects, spec: ID.QuerySpec) -> tuple:
    if not objects:
        return frozenset(), frozenset()
    c = np.asarray([o["center"] for o in objects], dtype=np.float64)
    n = np.asarray([o["normal"] for o in objects], dtype=np.float64)
    r = np.asarray([o["radius"] for o in objects], dtype=np.float64)
    P = np.asarray([spec.A, spec.B], dtype=np.float64)
    dist = disk_point_distance(c, n, r, P)
    a = frozenset(objects[i]["id"] for i in range(len(objects)) if dist[0, i] <= spec.r_attach)
    b = frozenset(objects[i]["id"] for i in range(len(objects)) if dist[1, i] <= spec.r_attach)
    return a, b


def project_labels(states, objects, spec: ID.QuerySpec) -> List[int]:
    index = {o["id"]: o for o in objects}
    labels = []
    for st in states:
        ids = st["present_ids"]
        if not ids:
            labels.append(0)
            continue
        c = np.asarray([index[i]["center"] for i in ids], dtype=np.float64)
        n = np.asarray([index[i]["normal"] for i in ids], dtype=np.float64)
        r = np.asarray([index[i]["radius"] for i in ids], dtype=np.float64)
        labels.append(1 if ID.two_point_target(spec, c, n, r) else 0)
    return labels


def derive_must_may(states, labels, objects, spec: ID.QuerySpec) -> dict:
    if not states:
        return {"available": False, "reason": "empty feasible set"}
    index = {o["id"]: i for i, o in enumerate(objects)}
    hit = _pair_matrix(objects)
    a_side, b_side = _attach_sets(objects, spec)
    derivations = []
    for st, lab in zip(states, labels):
        present = set(st["present_ids"])
        edges = set()
        for i in present:
            for j in present:
                if i < j and hit[index[i], index[j]]:
                    edges.add((min(i, j), max(i, j)))
        derivations.append(ID.StateDerivation(
            state_id=st["state_id"], nodes=frozenset(present), edges=frozenset(edges),
            a_side=frozenset(a_side & present), b_side=frozenset(b_side & present),
            attach_a=bool(a_side & present), attach_b=bool(b_side & present),
            label=int(lab)))
    env = ID.envelope_of(derivations)
    env2 = ID.envelope_of_quantifier(derivations)
    if env.to_dict() != env2.to_dict():
        raise P2ContractError("envelope_of and envelope_of_quantifier disagree")
    return {
        "available": True, "schema": SCHEMA_MUSTMAY,
        "derivation": {"source": "PER_HYPOTHESIS_ENUMERATION",
                       "n_states_used": len(derivations),
                       "intersection_path": "identification.envelope_of",
                       "verification_path": "identification.envelope_of_quantifier",
                       "independent_paths_equal": True},
        "must": {"nodes": sorted(env.must_nodes), "edges": sorted([u, v] for u, v in env.must_edges),
                 "a_side": sorted(env.must_a_side), "b_side": sorted(env.must_b_side),
                 "attach_a": bool(env.must_attach_a), "attach_b": bool(env.must_attach_b)},
        "may": {"nodes": sorted(env.may_nodes), "edges": sorted([u, v] for u, v in env.may_edges),
                "a_side": sorted(env.may_a_side), "b_side": sorted(env.may_b_side),
                "attach_a": bool(env.may_attach_a), "attach_b": bool(env.may_attach_b)},
        "region_label": ID.region_label(env),
        "must_connected": bool(ID.must_connected(env)),
        "may_connected": bool(ID.may_connected(env)),
        "must_subset_may": set(env.must_nodes) <= set(env.may_nodes)
                           and set(map(tuple, env.must_edges)) <= set(map(tuple, env.may_edges)),
        "may_path_is_not_a_physical_network": True,
    }


def decide_status(*, contradiction: bool, search_complete: bool,
                  n_hypotheses: int, gamma: Sequence[int],
                  scope: str = HYPOTHESIS_SCOPE) -> str:
    if contradiction or n_hypotheses == 0:
        return STATUS_MODEL_CONFLICT
    if not search_complete:
        return STATUS_SEARCH_UNRESOLVED
    g = tuple(sorted({int(x) for x in gamma}))
    if g not in STATUS_OF_GAMMA:
        raise P2ContractError("gamma %r has no state mapping" % (g,))
    status = STATUS_OF_GAMMA[g]
    if status in SINGLE_VALUE and scope not in CERTIFIABLE_SCOPES:
        return STATUS_SEARCH_UNRESOLVED
    return status


def build_certificate(*, case_id, query_id, status, gamma, n_hypotheses,
                      states, labels, search_complete, scope, n_universe):
    if status not in SINGLE_VALUE:
        return {"issued": False, "reason": "status is not single-valued"}
    if not search_complete:
        return {"issued": False, "reason": "search not complete"}
    if scope not in CERTIFIABLE_SCOPES:
        return {"issued": False, "reason": "hypothesis_scope not certifiable"}
    if sorted(set(labels)) != [SINGLE_VALUE[status]]:
        return {"issued": False, "reason": "all_surviving_hypotheses_same_label is false"}
    label = SINGLE_VALUE[status]
    witness = next(st for st, lab in zip(states, labels) if lab == label)
    return {"issued": True, "schema": SCHEMA_CERT,
            "certificate_id": "%s:%s" % (case_id, query_id),
            "certificate_class": CERTIFICATE_CLASS,
            "qualification": "P2_GATE_NOT_SIGNED_BY_THIS_UNIT",
            "issuance": ISSUANCE,
            "label": int(label), "status": status,
            "all_surviving_hypotheses_same_label": True,
            "n_surviving": int(n_hypotheses), "n_universe": int(n_universe),
            "hypothesis_scope": scope,
            "witness": {"state_id": int(witness["state_id"]),
                        "mask": int(witness["mask"]),
                        "present_ids": list(witness["present_ids"])},
            "may_path_is_not_a_witness": True}


def witness_from_may_path(*_args, **_kwargs):
    raise P2ContractError("a may path is not a witness; refusal is the correct reading")


def run_case(case: Mapping) -> dict:
    """Execute one case through the production chain."""
    t0 = time.perf_counter()
    assert_no_truth(case, "case")
    for k in ("case_id", "query", "budget"):
        if k not in case:
            raise P2ContractError("case missing %r" % k)
    source = str(case.get("universe_source", "DECLARED_INVENTORY"))
    if source in ("DECLARED_INVENTORY_FROM_SYNTHETIC_WORLD", "DECLARED_INVENTORY"):
        universe = derive_inventory_pins(case)
    elif source == "OBSERVATION_LEDGER_G69":
        universe = derive_ledger_pins(case)
    elif source == "CONSTRUCTED_CONTROL":
        universe = derive_explicit_pins(case)
    else:
        raise P2ContractError("unknown universe_source %r" % source)
    spec = make_query_spec(case["query"])
    scope = str(case.get("hypothesis_scope", HYPOTHESIS_SCOPE))
    evidence_issues = list(universe.get("issues") or [])
    h = enumerate_hypotheses(universe, case["budget"])
    n_h = int(h["n_hypotheses"])
    contradiction = (bool(h["contradiction"]) or bool(case["budget"].get("forced_empty", False))
                     or bool(universe.get("contradiction_ids")))

    if contradiction or n_h == 0:
        status = STATUS_MODEL_CONFLICT
        labels, gamma, must_may = [], [], {"available": False, "reason": "empty feasible set"}
        search_complete = False
    elif not h["search_complete"]:
        labels = project_labels(h["states"], universe["objects"], spec) if h["states"] else []
        gamma = sorted(set(labels))
        status = decide_status(contradiction=False, search_complete=False,
                               n_hypotheses=n_h, gamma=gamma, scope=scope)
        must_may = {"available": False, "reason": h["stop_reason"]}
        search_complete = False
    else:
        labels = project_labels(h["states"], universe["objects"], spec)
        gamma = sorted(set(labels))
        must_may = derive_must_may(h["states"], labels, universe["objects"], spec)
        search_complete = True
        status = decide_status(contradiction=False, search_complete=True,
                               n_hypotheses=n_h, gamma=gamma, scope=scope)
        if evidence_issues:
            status = STATUS_SEARCH_UNRESOLVED
            search_complete = False
            must_may = {"available": False,
                        "reason": "UNIVERSE_INCOMPLETE_OR_AMBIGUOUS_EVIDENCE"}

    cert = build_certificate(case_id=case["case_id"], query_id=spec.query_id,
                             status=status, gamma=gamma, n_hypotheses=n_h,
                             states=h["states"], labels=labels,
                             search_complete=search_complete, scope=scope,
                             n_universe=universe["n_universe"])
    elapsed = time.perf_counter() - t0
    return {
        "schema": SCHEMA_RESULT, "case_id": str(case["case_id"]),
        "query_id": spec.query_id, "case_class": case.get("case_class", "MAIN_QUERY"),
        "split": case.get("split", "NA"),
        "universe_source": universe["universe_source"],
        "hypothesis_scope": scope,
        "hypothesis_scope_is_exact_finite": scope == HYPOTHESIS_SCOPE,
        "hypothesis": {
            "pinned_present": list(universe["present_ids"]),
            "pinned_absent": list(universe["absent_ids"]),
            "free_ids": list(universe["free_ids"]),
            "masks": [int(st["mask"]) for st in h["states"]],
            "n_masks_listed": len(h["states"]),
        },
        "n_universe": universe["n_universe"], "n_free": universe["n_free"],
        "n_hypotheses": n_h, "enumerated": h["enumerated"],
        "search_complete": search_complete, "stop_reason": h["stop_reason"],
        "pins": {"n_present": len(universe["present_ids"]),
                 "n_absent": len(universe["absent_ids"]),
                 "n_free": len(universe["free_ids"])},
        "pin_issues": universe["issues"],
        "evidence_issues": evidence_issues,
        "gamma": gamma, "labels_digest": canonical_sha256(labels),
        "status": status,
        "all_surviving_hypotheses_same_label": len(gamma) == 1,
        "certificate": cert,
        "must_may": must_may,
        "has_truth": bool(case.get("has_truth", False)),
        "budget": {"max_nodes": h["max_nodes"], "max_seconds": h["max_seconds"],
                   "budget_exhausted": bool(h["budget_exhausted"]),
                   "enumeration_elapsed_s": h["elapsed_s"],
                   "case_elapsed_s": _f(elapsed)},
        "issuance": ISSUANCE,
        "certification_issued": False,
        "measured_at_utc": utc_now(),
    }
