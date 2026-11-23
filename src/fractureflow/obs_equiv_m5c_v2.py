"""M5C v2 two-locus pool generator and structural certificates.

Scope: software implementation + pool construction only. No scientific reading,
no strategy run, no selector call, no sealed access, no evaluator file read.

Frozen authority
----------------
``results/obs_equiv_20261024/m5c_v2_prereg_20261026/prereg_v2.json``
(schema ``obs_equiv.m5c_v2_prereg.v1``) and the architect ruling
``results/obs_equiv_20261024/w8_architect_20261026/ruling_v2_open_questions_20261026.md``
(R2V2_20261026).  This module implements the W8-03 pool-build order
``results/obs_equiv_20261024/w8_task_orders_20261026/TO-W8-02-03_v2冻结与建池.md``.

What is new here (and what is not)
----------------------------------
Reused, byte-for-byte, from the frozen v1 module
``fractureflow.obs_equiv_m5c`` (imported, never re-implemented and never
modified): the rational geometry helpers (``_round2``, ``_xy``, ``_pt``,
``_num``, ``canonical_direction``, ``disk_contact``), ``Grid``/``build_grid``,
``_public_object``/``_serialise_object``, ``signature_of``/``canonical_json``/
``sha256_text``/``sha256_bytes``, ``_contact_graph``/``_variant_query``,
``build_world`` (the frozen evaluator semantics), the C-NT-1..C-NT-5
certificate predicates, and ``pool_fingerprint``/``assert_dataset_sha``.

New in v2 (this module only):

* the **two-locus generator** ``visible_chain_v2`` + ``build_candidate``:
  the v1 visible staircase is kept (radii, disk counts, rise, step, cut), but
  the cut clearance is widened to ``CLEARANCE_MULTIPLIER * family max gap`` so
  that two decisive contact loci both fall outside the reach of either visible
  endpoint disk (the v1 clearance is too short: both loci would land in the same
  grid cell and one cell would resolve the query in one reveal);
* the declared Q structure: 4 ``Q=1`` variants (both loci connected) and 4
  ``Q=0`` variants (2 broken at L1, 2 broken at L2).  Each ``Q=0`` variant is
  its ``Q=1`` parent with **exactly one** locus altered, so that every locus's
  "connected" signature is shared with a ``Q=1`` signature (measured, see
  ``locus_signature_overlap``);
* the break at a locus is a *radius contraction at a fixed centre*: the locus
  element stays visible (never an EMPTY reveal) and never moves between grid
  cells, so the outcome partition of the locus cell keeps a cross-class
  (Q=1 u Q=0) class for the C-NT-6 predicate;
* C-NT-6 (binding): for every change unit and every one of the 36 public grid
  actions, ``V1(a|b0) > 0``;
* C-NT-7 (binding): every unit carries the same 36 public grid action ids at
  unit cost 1.0, with no per-unit action removal;
* C-NT-8 (diagnostic only): L1 and L2 fall in different grid cells and no cell
  contains both;
* the deterministic attempt sequence / rejection-and-resample loop with
  ``N_ATTEMPT_MAX`` and a per-candidate rejection log.

Exactness: coordinates and radii are multiples of 1/100 in ``Fraction``; every
contact, ``V1`` and certificate comparison is exact rational arithmetic.  The
only floats in this module are used to *choose* candidate parameters (they
never decide a predicate); every predicate is re-evaluated in exact arithmetic
before a candidate is accepted.
"""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from fractureflow import obs_equiv_m5c as _v1
from fractureflow.obs_equiv_m5a_reference import (
    initial_belief,
    risk_brier,
    v1 as one_step_value,
)

SCHEMA = "obs_equiv.m5c_v2_impl.core.v1"

POOL_ID = "m5c_v2_declared_pool_20261026"
POOL_VERSION = "v2"
GENERATION_NAMESPACE = "m5c_v2_20261026"
PROTOCOL_LABEL = "SYNTHETIC_DECLARED_DOMAIN_ONLY"
MECHANISM_VARIANT_TAG = "V2_TWO_LOCUS_VARIANT"
SCOPE_TAG = "K_STAR_GE_2_DECLARED_SYNTHETIC"

MECHANISMS = _v1.MECHANISMS
FAMILIES = _v1.FAMILIES
INDICES = _v1.INDICES
ROLE_CHANGE = _v1.ROLE_CHANGE
ROLE_CONTROL = _v1.ROLE_CONTROL
SPLIT_DEVELOPMENT = _v1.SPLIT_DEVELOPMENT
SPLIT_CONFIRMATORY = _v1.SPLIT_CONFIRMATORY

GRID_G = _v1.GRID_G
COST_PER_REVEAL = _v1.COST_PER_REVEAL
B_CAP = _v1.B_CAP
RHO_PRIMARY = _v1.RHO_PRIMARY
H_SIZE = 8
PRIOR_WEIGHTS = (1,) * 8  # OQ-11: (1,1,1,1) per Q class, two classes of mass 1/2

N_ATTEMPT_MAX = 10_000
UNITS_PER_CELL_PER_ROLE = 4

#: v2 cut clearance = multiplier x the family's declared maximum edge gap.
#: The multiplier is a *new v2 design constant* (declared, not a v1 threshold
#: change): with the v1 clearance both loci fall inside the reach of the visible
#: endpoint disks and land in one grid cell, which makes C-NT-6 unsatisfiable.
CLEARANCE_MULTIPLIER = Fraction(22, 10)

#: Declared rational grid for the two locus positions (design proposal
#: ``locus_positions_proposed``: L1 in [0.28, 0.34], L2 in [0.66, 0.72]).
LOCUS_T1_GRID = tuple(Fraction(k, 100) for k in (28, 29, 30, 31, 32, 33, 34))
LOCUS_T2_GRID = tuple(Fraction(k, 100) for k in (66, 67, 68, 69, 70, 71, 72))

#: Body (backbone) disk count per mechanism: the declared per-locus range in
#: ``v2_pool_parameters.json#hidden_object_counts``.
BODY_DISK_COUNT = {
    "CHAIN_VS_DIRECT": 2,
    "CROSSING_VS_PASSING": 2,
    "HIDDEN_BRIDGE": 3,
}

#: Perpendicular nuisance offsets, as multiples of the A->B vector length
#: (``_pt`` uses the un-normalised perpendicular, exactly as in v1).
OFFSET_GRID = (Fraction(-2, 100), Fraction(-1, 100), Fraction(0), Fraction(1, 100), Fraction(2, 100))

#: Normal-vector patterns used by the hidden elements (constant within a unit).
NORMAL_PATTERN_SIZE = 5

C_NT_BINDING = ("C-NT-1", "C-NT-2", "C-NT-3", "C-NT-4", "C-NT-5", "C-NT-6", "C-NT-7")
C_NT_DIAGNOSTIC = ("C-NT-8",)

REJECT_LOCUS_INSIDE_REACH = "LOCUS_INSIDE_VISIBLE_REACH"
REJECT_BODY_TOO_SMALL = "BODY_RADIUS_TOO_SMALL"
REJECT_DECOY_TOO_SMALL = "DECOY_RADIUS_TOO_SMALL"
REJECT_Q_LABEL = "INTENDED_Q_LABEL_MISMATCH"
REJECT_STATUS = "CHANGE_UNIT_STATUS_NOT_CHANGE"
REJECT_CNT6 = "C_NT_6_FAIL"
REJECT_CERT = "BINDING_CERTIFICATE_FAIL"

UNIT_STATUS_CHANGE = _v1.UNIT_STATUS_CHANGE
UNIT_STATUS_CONTROL = _v1.UNIT_STATUS_CONTROL


class ContractViolation(RuntimeError):
    """Implementation-level inconsistency (never a protocol result)."""


class CandidateRejected(Exception):
    """A candidate failed a declared build predicate and must be resampled."""

    def __init__(self, reason: str, detail: Optional[Mapping] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail or {})


def _fail_closed(reason: str) -> None:
    raise SystemExit(f"M5C-V2-FAIL-CLOSED: {reason}")


def canonical_json(document) -> str:
    return _v1.canonical_json(document)


def sha256_text(text: str) -> str:
    return _v1.sha256_text(text)


def sha256_bytes(data: bytes) -> str:
    return _v1.sha256_bytes(data)


def _sq(value: Fraction) -> Fraction:
    return value * value


# ---------------------------------------------------------------------------
# Declared v2 world geometry
# ---------------------------------------------------------------------------


def family_clearance(family: str) -> Fraction:
    """v2 cut clearance (edge gap) for a world family, in hundredths."""
    spec = _v1.FAMILY_SPECS[family]
    return _v1._round2(CLEARANCE_MULTIPLIER * spec["gap_width"][1])


def visible_chain_v2(mechanism: str, family: str, index: int) -> tuple:
    """v1 visible staircase with the widened v2 cut clearance.

    Same radii, disk counts, rise and step as ``obs_equiv_m5c._visible_chain``;
    only the along-x shift at the cut is recomputed so that the edge gap equals
    :func:`family_clearance`.  Returns ``(objects, a_disk, b_disk, a_end,
    b_end, spec)`` with the same conventions as v1.
    """
    spec = _v1.FAMILY_SPECS[family]
    n = spec["visible_chain_disks"][index % 2]
    radii = tuple(spec["visible_radii"][(i + index) % len(spec["visible_radii"])] for i in range(n))
    x0, y0 = spec["x0"], spec["y0"]
    base = [(x0 + spec["step_x"] * i, y0 + spec["rise"] * i) for i in range(n)]
    cut = index % (n - 1)
    r_a, r_b = radii[cut], radii[cut + 1]
    need = r_a + r_b + family_clearance(family)
    step_x, rise = spec["step_x"], spec["rise"]
    dx = step_x
    while dx * dx + rise * rise < need * need:
        dx += Fraction(1, 100)
        if dx > Fraction(30):
            raise ContractViolation(f"{mechanism}/{family}/{index}: no v2 cut shift found")
    shift = dx - step_x
    points = [(x + shift, y) if i > cut else (x, y) for i, (x, y) in enumerate(base)]
    objects = [
        {
            "id": f"v{i}",
            "center": (_v1._round2(x), _v1._round2(y), Fraction(0)),
            "normal": (0, 0, 1),
            "radius": radii[i],
            "type": "DISK",
        }
        for i, (x, y) in enumerate(points)
    ]
    return objects, objects[0], objects[-1], objects[cut], objects[cut + 1], spec


# ---------------------------------------------------------------------------
# Deterministic attempt sequence
# ---------------------------------------------------------------------------


def candidate_seed(mechanism: str, family: str, index: int, attempt: int) -> int:
    """Declared seed derivation (v2_pool_parameters.json#seed_derivation)."""
    material = f"{GENERATION_NAMESPACE}|{mechanism}|{family}|{index}|{attempt}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def candidate_parameters(mechanism: str, family: str, index: int, attempt: int) -> dict:
    """Fully determined candidate parameters for one attempt index."""
    rng = random.Random(candidate_seed(mechanism, family, index, attempt))
    t1 = LOCUS_T1_GRID[rng.randrange(len(LOCUS_T1_GRID))]
    t2 = LOCUS_T2_GRID[rng.randrange(len(LOCUS_T2_GRID))]
    body_count = BODY_DISK_COUNT[mechanism]
    offsets = [OFFSET_GRID[rng.randrange(len(OFFSET_GRID))] for _ in range(body_count)]
    skew = OFFSET_GRID[rng.randrange(len(OFFSET_GRID))]
    normal_base = rng.randrange(NORMAL_PATTERN_SIZE)
    return {
        "t1": t1,
        "t2": t2,
        "body_count": body_count,
        "body_offsets": offsets,
        "locus_skew": skew,
        "normal_base": normal_base,
        "seed_attempt": candidate_seed(mechanism, family, index, attempt),
    }


def candidate_fingerprint(mechanism: str, family: str, index: int, attempt: int, params: Mapping) -> str:
    """Stable identity of a candidate (logged for every rejection)."""
    payload = {
        "namespace": GENERATION_NAMESPACE,
        "mechanism": mechanism,
        "world_family": family,
        "index": index,
        "attempt": attempt,
        "t1": str(params["t1"]),
        "t2": str(params["t2"]),
        "body_count": params["body_count"],
        "body_offsets": [str(q) for q in params["body_offsets"]],
        "locus_skew": str(params["locus_skew"]),
        "normal_base": params["normal_base"],
        "seed_attempt": params["seed_attempt"],
    }
    return sha256_text(canonical_json(payload))


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _hundredths(value: float) -> Fraction:
    """Snap a positive float to the 1/100 grid (choice helper only)."""
    scaled = int(round(value * 100))
    return Fraction(max(scaled, 1), 100)


def build_candidate(mechanism: str, family: str, index: int, attempt: int) -> dict:
    """Build one candidate unit (change + control) or raise CandidateRejected.

    The predicates that can reject a candidate are all structural and declared
    in advance: locus reach, body/decoy radius admissibility, the intended Q
    label of every variant, the change-unit status, C-NT-1..C-NT-5, C-NT-6 and
    C-NT-7.  No predicate is relaxed after the fact.
    """
    params = candidate_parameters(mechanism, family, index, attempt)
    fingerprint = candidate_fingerprint(mechanism, family, index, attempt, params)
    visible, a_disk, b_disk, a_end, b_end, spec = visible_chain_v2(mechanism, family, index)

    ax, ay = _v1._xy(a_end["center"])
    bx, by = _v1._xy(b_end["center"])
    r_a = _v1.exact_fraction(a_end["radius"])
    r_b = _v1.exact_fraction(b_end["radius"])
    delta = (bx - ax, by - ay)
    perp = (-delta[1], delta[0])
    length = float(delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
    if length <= 0:
        raise ContractViolation("degenerate A->B segment")

    def point(t: Fraction, q: Fraction) -> tuple:
        return _v1._pt(a_end, b_end, perp, t, q)

    t1 = params["t1"]
    t2 = params["t2"]
    s1 = float(t1) * length
    s2 = float(1 - t2) * length
    margin1 = s1 - float(r_a)
    margin2 = s2 - float(r_b)
    if margin1 <= 0.03 or margin2 <= 0.03:
        raise CandidateRejected(
            REJECT_LOCUS_INSIDE_REACH,
            {"margin_L1": margin1, "margin_L2": margin2, "segment_length": length},
        )

    body_count = params["body_count"]
    gap = (float(t2 - t1) * length) / (body_count + 1)
    body_radius = _v1._round2(Fraction(int(0.62 * gap * 100), 100))
    if body_radius < Fraction(4, 100):
        raise CandidateRejected(REJECT_BODY_TOO_SMALL, {"body_radius": float(body_radius), "gap": gap})

    # Bridge radius must reach the visible endpoint disk and the first body disk.
    base1 = max(margin1 + 0.02, gap - float(body_radius) + 0.02)
    base2 = max(margin2 + 0.02, gap - float(body_radius) + 0.02)
    q1 = [_v1._round2(_hundredths(base1 + 0.02 * i)) for i in range(4)]
    # L2 band is kept disjoint from the L1 band so that every variant's
    # (radius) multiset is unique (C-NT-1 non-translation).
    q2 = [_v1._round2(_hundredths(base2 + 0.12 + 0.02 * i)) for i in range(4)]
    decoy1 = [_v1._round2(_hundredths(margin1 - 0.05)) - Fraction(1, 100) * i for i in range(2)]
    decoy2 = [_v1._round2(_hundredths(margin2 - 0.05)) - Fraction(1, 100) * i for i in range(2)]
    if min(decoy1 + decoy2) < Fraction(2, 100):
        raise CandidateRejected(
            REJECT_DECOY_TOO_SMALL,
            {"decoy_L1": [float(v) for v in decoy1], "decoy_L2": [float(v) for v in decoy2]},
        )

    body_positions = [
        point(t1 + (t2 - t1) * Fraction(i + 1, body_count + 1), params["body_offsets"][i])
        for i in range(body_count)
    ]
    p1 = point(t1, params["locus_skew"])
    p2 = point(t2, -params["locus_skew"])

    normal_index = params["normal_base"]

    def hidden_configuration(b1_radius: Fraction, b2_radius: Fraction) -> list:
        objects = [
            _v1._hidden_object("locus1", 0, p1, b1_radius, normal_index),
            _v1._hidden_object("locus2", 0, p2, b2_radius, normal_index + 1),
        ]
        objects[0]["id"] = "L1"
        objects[1]["id"] = "L2"
        for i, (position, radius) in enumerate(zip(body_positions, [body_radius] * body_count)):
            obj = _v1._hidden_object(f"body{i}", i, position, radius, normal_index + 2 + i)
            obj["id"] = f"body{i}"
            objects.append(obj)
        return objects

    # h0..h3 = Q=1 (both loci connected); h4,h5 = broken at L1; h6,h7 = broken at L2.
    declared = [
        (q1[0], q2[0], 1),
        (q1[1], q2[1], 1),
        (q1[2], q2[2], 1),
        (q1[3], q2[3], 1),
        (decoy1[0], q2[0], 0),
        (decoy1[1], q2[1], 0),
        (q1[0], decoy2[0], 0),
        (q1[1], decoy2[1], 0),
    ]
    hidden = []
    labels = []
    for b1_radius, b2_radius, expected in declared:
        configuration = hidden_configuration(b1_radius, b2_radius)
        label = _v1._variant_query(list(visible) + configuration, "v0", f"v{len(visible) - 1}")
        labels.append(int(label))
        hidden.append(configuration)
        if int(label) != expected:
            raise CandidateRejected(
                REJECT_Q_LABEL,
                {
                    "expected_label": [int(entry[2]) for entry in declared],
                    "measured_label": labels,
                    "segment_length": length,
                },
            )

    query = {
        "family": "TWO_POINT_CONNECTIVITY",
        "params": {
            "A": [_v1._num(a_disk["center"][0]), _v1._num(a_disk["center"][1]), _v1._num(a_disk["center"][2])],
            "B": [_v1._num(b_disk["center"][0]), _v1._num(b_disk["center"][1]), _v1._num(b_disk["center"][2])],
        },
        "query_id": f"q|{mechanism}|{family}|{index}",
    }
    observation = {
        "observed_objects": [_v1._public_object(obj, f"p{i}") for i, obj in enumerate(visible)],
        "query": query,
    }
    visible_ids = [obj["id"] for obj in visible]
    scenes = [
        {"objects": [_v1._serialise_object(obj) for obj in list(visible) + list(configuration)]}
        for configuration in hidden
    ]
    change_id = f"{mechanism}|{family}|{ROLE_CHANGE}|{index}"
    control_id = f"{mechanism}|{family}|{ROLE_CONTROL}|{index}"
    public_change = {"unit_id": change_id, "observation": observation}
    public_control = {"unit_id": control_id, "observation": observation}
    evaluator_change = {
        "unit_id": change_id,
        "twin_unit_id": control_id,
        "mechanism": mechanism,
        "world_family": family,
        "role": ROLE_CHANGE,
        "index": index,
        "visible_object_ids": visible_ids,
        "prior_weights": list(PRIOR_WEIGHTS),
        "query_labels": list(labels),
        "scenes": scenes,
        "generation": {
            "mechanism_variant": MECHANISM_VARIANT_TAG,
            "generation_namespace": GENERATION_NAMESPACE,
            "attempt": attempt,
            "candidate_fingerprint": fingerprint,
            "seed_attempt": params["seed_attempt"],
            "locus_t1": str(t1),
            "locus_t2": str(t2),
            "body_disk_count": body_count,
            "locus_radii_q1": [str(v) for v in q1],
            "locus_radii_q2": [str(v) for v in q2],
            "decoy_radii_l1": [str(v) for v in decoy1],
            "decoy_radii_l2": [str(v) for v in decoy2],
            "body_radius": str(body_radius),
        },
    }
    evaluator_control = {
        "unit_id": control_id,
        "twin_unit_id": change_id,
        "mechanism": mechanism,
        "world_family": family,
        "role": ROLE_CONTROL,
        "index": index,
        "visible_object_ids": visible_ids,
        "prior_weights": list(PRIOR_WEIGHTS),
        "query_labels": [0, 0, 0, 0, 0, 0, 0, 0],
        "scenes": scenes,
        "twin_hidden_point_identical": True,
        "generation": {
            "mechanism_variant": MECHANISM_VARIANT_TAG,
            "generation_namespace": GENERATION_NAMESPACE,
            "attempt": attempt,
            "candidate_fingerprint": fingerprint,
            "seed_attempt": params["seed_attempt"],
        },
    }

    world_change = _v1.build_world(public_change, evaluator_change)
    world_control = _v1.build_world(public_control, evaluator_control)
    if world_change.status != UNIT_STATUS_CHANGE:
        raise CandidateRejected(
            REJECT_STATUS, {"status": world_change.status, "unit_id": change_id}
        )
    if world_control.status != UNIT_STATUS_CONTROL:
        raise CandidateRejected(
            REJECT_STATUS, {"status": world_control.status, "unit_id": control_id}
        )

    locus_cells = {
        "L1": world_change.grid.cell_id_at(p1[0], p1[1]),
        "L2": world_change.grid.cell_id_at(p2[0], p2[1]),
    }
    loci = {
        "L1": {
            "cell_id": locus_cells["L1"],
            "center": [_v1._num(p1[0]), _v1._num(p1[1]), 0.0],
            "element_ids": ["L1"],
            "q1_radii": [str(v) for v in q1],
            "q0_radii": [str(v) for v in decoy1],
        },
        "L2": {
            "cell_id": locus_cells["L2"],
            "center": [_v1._num(p2[0]), _v1._num(p2[1]), 0.0],
            "element_ids": ["L2"],
            "q1_radii": [str(v) for v in q2],
            "q0_radii": [str(v) for v in decoy2],
        },
        "locus_cells_distinct": locus_cells["L1"] != locus_cells["L2"],
        "same_cell_contains_both": locus_cells["L1"] == locus_cells["L2"],
    }

    certificates = unit_certificates(world_change, None, loci)
    certificates_control = unit_certificates(world_control, world_change, loci)
    binding_failures = [
        cert_id
        for cert_id in C_NT_BINDING
        if not certificates["certificates"][cert_id]["ok"]
        or not certificates_control["certificates"][cert_id]["ok"]
    ]
    if binding_failures:
        reason = REJECT_CNT6 if binding_failures == ["C-NT-6"] else REJECT_CERT
        raise CandidateRejected(
            reason,
            {
                "unit_id": change_id,
                "failed": binding_failures,
                "c_nt_6": certificates["certificates"]["C-NT-6"],
                "locus_cells": locus_cells,
            },
        )

    return {
        "mechanism": mechanism,
        "world_family": family,
        "index": index,
        "attempt": attempt,
        "candidate_fingerprint": fingerprint,
        "public_change": public_change,
        "public_control": public_control,
        "evaluator_change": evaluator_change,
        "evaluator_control": evaluator_control,
        "certificates": certificates,
        "certificates_control": certificates_control,
        "loci": loci,
        "world_change": world_change,
        "segment_length": length,
        "parameters": {
            "t1": str(t1),
            "t2": str(t2),
            "body_count": body_count,
            "body_radius": str(body_radius),
            "locus_radii_q1": [str(v) for v in q1],
            "locus_radii_q2": [str(v) for v in q2],
            "decoy_radii_l1": [str(v) for v in decoy1],
            "decoy_radii_l2": [str(v) for v in decoy2],
        },
    }


# ---------------------------------------------------------------------------
# C-NT-6 / C-NT-7 / C-NT-8 and the cross-class signature overlap
# ---------------------------------------------------------------------------


def certificate_no_one_step_full_resolve(world) -> dict:
    """C-NT-6 (binding): every action has V1(a|b0) > 0 on a change unit."""
    if world.status != UNIT_STATUS_CHANGE:
        return {
            "id": "C-NT-6",
            "what": "no cost-1 one-step full-resolve action at b0 (change units)",
            "applicable": False,
            "ok": True,
            "reason": "control unit: Q is constant, V1(a)=R_Q(b0)=0 for every action by construction",
        }
    belief0 = initial_belief(world.problem)
    risk0 = risk_brier(belief0, world.problem.query)
    values = {}
    zero_actions = []
    for action in world.problem.actions:
        value = one_step_value(belief0, action, world.problem.query)
        values[action.action_id] = value
        if value <= 0:
            zero_actions.append(action.action_id)
    strict = sum(1 for value in values.values() if value < risk0)
    return {
        "id": "C-NT-6",
        "what": "no cost-1 one-step full-resolve action at b0 (change units)",
        "applicable": True,
        "actions_checked": len(values),
        "zero_or_negative_actions": sorted(zero_actions),
        "min_v1": str(min(values.values())) if values else None,
        "max_v1": str(max(values.values())) if values else None,
        "risk_at_b0": str(risk0),
        "strictly_informative_actions": strict,
        "verdict_definition": "PASS iff |{a : V1(a) <= 0}| == 0 and actions_checked == 36",
        "ok": (not zero_actions) and len(values) == GRID_G * GRID_G,
    }


def canonical_action_ids() -> tuple:
    return tuple(f"R{i}C{j}" for i in range(GRID_G) for j in range(GRID_G))


def certificate_action_set_uniform(worlds: Sequence) -> dict:
    """C-NT-7 (binding): same 36 public grid actions, unit cost, no removal."""
    expected = canonical_action_ids()
    failures = []
    cost_failures = []
    for world in worlds:
        ids = tuple(sorted(action.action_id for action in world.problem.actions))
        if ids != tuple(sorted(expected)):
            failures.append({"unit_id": world.unit_id, "n_actions": len(ids)})
        for action in world.problem.actions:
            if action.cost != COST_PER_REVEAL:
                cost_failures.append(
                    {"unit_id": world.unit_id, "action_id": action.action_id, "cost": str(action.cost)}
                )
    return {
        "id": "C-NT-7",
        "what": "every unit carries the same 36 public grid actions at unit cost 1.0",
        "units_checked": len(worlds),
        "expected_action_count": len(expected),
        "action_set_failures": failures,
        "cost_failures": cost_failures,
        "ok": (not failures) and (not cost_failures) and len(worlds) > 0,
    }


def certificate_locus_separation(loci: Mapping) -> dict:
    """C-NT-8 (diagnostic): L1 and L2 in different cells, no cell covers both."""
    cell1 = loci["L1"]["cell_id"]
    cell2 = loci["L2"]["cell_id"]
    overlap_cells = [cell1] if cell1 == cell2 else []
    distinct = cell1 != cell2
    detail = {
        "id": "C-NT-8",
        "what": "L1 and L2 fall in different grid cells; no single cell covers both",
        "L1_cell": cell1,
        "L2_cell": cell2,
        "cells_covering_both_loci": overlap_cells,
        "binding": False,
        "status": "CONSTRUCTION_DIAGNOSTIC_ONLY",
        "ok": distinct,
    }
    if not distinct:
        # Diagnostic only: record the consequence, never repair it here.
        detail["consequence"] = (
            "with one cell covering both loci every outcome class of that cell is Q-pure, "
            "so C-NT-6 fails on that cell; the unit is rejected by C-NT-6, not by C-NT-8"
        )
    return detail


def locus_signature_overlap(world, loci: Mapping) -> dict:
    """Measured cross-class overlap of the "connected" locus signatures.

    For each locus, the signature set of the intact locus over the Q=1 variants
    must intersect the intact-locus signature set over the Q=0 variants that are
    broken at the *other* locus.  Otherwise the locus separates Q on its own and
    C-NT-6 would fail.
    """
    hypotheses = tuple(f"h{i}" for i in range(H_SIZE))
    out = {}
    for locus_id, broken_indices, intact_indices in (
        ("L1", (4, 5), (6, 7)),
        ("L2", (6, 7), (4, 5)),
    ):
        cell = loci[locus_id]["cell_id"]
        per_h = world.signatures[cell]
        q1 = sorted({per_h[hypotheses[i]] for i in range(4)})
        q0_intact = sorted({per_h[hypotheses[i]] for i in intact_indices})
        q0_broken = sorted({per_h[hypotheses[i]] for i in broken_indices})
        out[locus_id] = {
            "cell_id": cell,
            "q1_signatures": q1,
            "q0_intact_signatures": q0_intact,
            "q0_broken_signatures": q0_broken,
            "intersection": sorted(set(q1) & set(q0_intact)),
            "overlap_ok": bool(set(q1) & set(q0_intact)),
        }
    return out


def unit_certificates(world, twin, loci: Mapping) -> dict:
    """C-NT-1..C-NT-8 for one unit, reusing the frozen v1 predicates 1-5."""
    base = _v1.unit_certificates(world, twin)
    certificates = dict(base["certificates"])
    certificates["C-NT-6"] = certificate_no_one_step_full_resolve(world)
    certificates["C-NT-7"] = {
        "id": "C-NT-7",
        "what": "unit carries the 36 public grid actions at unit cost 1.0",
        "actions": len(world.problem.actions),
        "expected_actions": GRID_G * GRID_G,
        "ok": len(world.problem.actions) == GRID_G * GRID_G
        and all(action.cost == COST_PER_REVEAL for action in world.problem.actions),
    }
    certificates["C-NT-8"] = certificate_locus_separation(loci)
    overlap = locus_signature_overlap(world, loci)
    counters = dict(base["counters"])
    counters.update(
        {
            "locus_cells": {"L1": loci["L1"]["cell_id"], "L2": loci["L2"]["cell_id"]},
            "locus_cells_distinct": loci["locus_cells_distinct"],
            "locus_signature_overlap": overlap,
            "c_nt_8_ok": certificates["C-NT-8"]["ok"],
        }
    )
    binding_ok = all(certificates[cert_id]["ok"] for cert_id in ("C-NT-1", "C-NT-2", "C-NT-3", "C-NT-4", "C-NT-5", "C-NT-6", "C-NT-7"))
    return {
        "unit_id": base["unit_id"],
        "mechanism": base["mechanism"],
        "world_family": base["world_family"],
        "role": base["role"],
        "status": base["status"],
        "eligible": base["eligible"],
        "certificates": certificates,
        "counters": counters,
        "all_certificates_pass": binding_ok,
        "all_certificates_including_diagnostic_pass": binding_ok and certificates["C-NT-8"]["ok"],
    }


# ---------------------------------------------------------------------------
# Pool assembly with the rejection-and-resample loop
# ---------------------------------------------------------------------------


def accept_unit(mechanism: str, family: str, index: int, max_attempts: int = N_ATTEMPT_MAX) -> dict:
    """First accepted candidate for one (mechanism, family, index) slot."""
    rejections = []
    for attempt in range(max_attempts):
        params = candidate_parameters(mechanism, family, index, attempt)
        fingerprint = candidate_fingerprint(mechanism, family, index, attempt, params)
        try:
            accepted = build_candidate(mechanism, family, index, attempt)
        except CandidateRejected as exc:
            rejections.append(
                {
                    "attempt": attempt,
                    "candidate_fingerprint": fingerprint,
                    "reason": exc.reason,
                    "detail": _jsonable(exc.detail),
                }
            )
            continue
        return {
            "accepted": accepted,
            "rejections": rejections,
            "attempts_used": attempt + 1,
        }
    return {"accepted": None, "rejections": rejections, "attempts_used": max_attempts}


def _jsonable(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def generate_pool(max_attempts: int = N_ATTEMPT_MAX) -> dict:
    """Deterministic v2 pool: 3 mechanisms x 3 families x 2 roles x 4 indices.

    Returns ``{"public", "evaluator", "units", "rejection_log", "stop_cells"}``
    where ``units`` carries the manifest-level per-unit data.  A slot that cannot
    be filled inside ``max_attempts`` raises :class:`ContractViolation` carrying
    the cell name so the caller can report ``STOP_CELL_YIELD``.
    """
    public_rows = []
    evaluator_rows = []
    units = []
    rejection_log = {}
    stop_cells = []
    for mechanism in MECHANISMS:
        for family in FAMILIES:
            cell = f"{mechanism}|{family}"
            cell_log = {
                "cell": cell,
                "mechanism": mechanism,
                "world_family": family,
                "per_index": [],
                "candidates_built": 0,
                "candidates_rejected": 0,
                "candidates_accepted": 0,
                "reason_counts": {},
                "attempt_index_range": None,
                "accepted_candidate_fingerprints": [],
            }
            for index in INDICES:
                outcome = accept_unit(mechanism, family, index, max_attempts=max_attempts)
                rejections = outcome["rejections"]
                cell_log["per_index"].append(
                    {
                        "index": index,
                        "attempts_used": outcome["attempts_used"],
                        "rejections": len(rejections),
                        "accepted_fingerprint": (
                            outcome["accepted"]["candidate_fingerprint"] if outcome["accepted"] else None
                        ),
                        "rejection_records": rejections,
                    }
                )
                cell_log["candidates_built"] += outcome["attempts_used"]
                cell_log["candidates_rejected"] += len(rejections)
                for record in rejections:
                    cell_log["reason_counts"][record["reason"]] = (
                        cell_log["reason_counts"].get(record["reason"], 0) + 1
                    )
                if outcome["accepted"] is None:
                    stop_cells.append(
                        {
                            "cell": cell,
                            "index": index,
                            "attempts": outcome["attempts_used"],
                            "status": "STOP_CELL_YIELD",
                        }
                    )
                    continue
                accepted = outcome["accepted"]
                cell_log["candidates_accepted"] += 1
                cell_log["accepted_candidate_fingerprints"].append(accepted["candidate_fingerprint"])
                public_rows.append(accepted["public_change"])
                public_rows.append(accepted["public_control"])
                evaluator_rows.append(accepted["evaluator_change"])
                evaluator_rows.append(accepted["evaluator_control"])
                units.append(accepted)
            if cell_log["per_index"]:
                cell_log["attempt_index_range"] = [
                    0,
                    max(entry["attempts_used"] for entry in cell_log["per_index"]) - 1,
                ]
            rejection_log[cell] = cell_log
    if stop_cells:
        try:
            _cells = json.dumps(stop_cells, ensure_ascii=False, allow_nan=False)
        except ValueError:  # 非有限值不得进消息体，但协议异常类型必须保住
            _cells = "<non-finite stop_cells: %r>" % (stop_cells,)
        raise ContractViolation(f"STOP_CELL_YIELD: {_cells}")
    return {
        "public": public_rows,
        "evaluator": evaluator_rows,
        "units": units,
        "rejection_log": rejection_log,
        "stop_cells": stop_cells,
    }


def pool_fingerprint(public_rows: Sequence[Mapping], evaluator_rows: Sequence[Mapping]) -> str:
    """Frozen v1 fingerprint rule, reused verbatim through the v1 module."""
    return _v1.pool_fingerprint(public_rows, evaluator_rows)


def assert_dataset_sha(public_rows: Sequence[Mapping], evaluator_rows: Sequence[Mapping], expected: str) -> str:
    return _v1.assert_dataset_sha(public_rows, evaluator_rows, expected)


# ---------------------------------------------------------------------------
# Split rule (inherited from the v1 mother protocol, materialised for v2)
# ---------------------------------------------------------------------------


def holdout_mechanism(eligible_mechanisms: Sequence[str]) -> Optional[str]:
    """Codepoint-order first eligible mechanism (v1 split protocol)."""
    eligible = sorted(set(eligible_mechanisms))
    return eligible[0] if eligible else None


def assign_split(mechanism: str, family: str, holdout: Optional[str]) -> str:
    if holdout is None:
        _fail_closed("split assignment without an eligible holdout mechanism")
    if mechanism == holdout:
        return SPLIT_CONFIRMATORY
    return SPLIT_CONFIRMATORY if family == min(FAMILIES) else SPLIT_DEVELOPMENT
