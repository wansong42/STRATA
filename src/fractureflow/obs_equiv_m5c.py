"""M5C non-translation mechanisms, mechanism holdout and scalable pruning.

Implements the frozen protocol
``results/obs_equiv_20261024/m5c_protocol_prereg_20261025/prereg.json``
(schema ``obs_equiv.m5c_protocol_prereg.v1``, sha256 ddfbb3da...b8a2ce) on top
of the shared M5A belief machinery (``fractureflow.obs_equiv_m5a_reference``),
which is imported and never reimplemented, and the seven frozen selectors of
``fractureflow.obs_equiv_m5b``, which are imported as objects (identity checked
in the unit tests) rather than re-typed here.

Scope: software implementation only. No scientific reading, no real data, no
evaluator file read from inside this module, no v6 artefact. H1 remains an
unverified target; nothing computed here is evidence for or against it.

Frozen pieces implemented here (cross-referenced to prereg.json):

* ``units``: a declared synthetic domain of 3 mechanisms x 3 world families x
  2 roles x 4 units = 72 units; each unit carries a public payload, eight
  hidden variants (4 x Q=1, 4 x Q=0 for a change unit) and a declared prior
  whose two Q classes each hold half the mass with positive integer
  within-class weights.
* ``units.certificates_per_unit``: C-NT-1 (no rational translation between any
  two complete hidden configurations), C-NT-2 (structural visibility), C-NT-3
  (query decisiveness, change units only), C-NT-4 (byte-identical public
  payload across H), C-NT-5 (twin control: hidden configurations point-identical
  to the change twin, Q identically zero). Failures are registered, never
  silently repaired.
* ``action_space``: public-only bbox addressing, G=6, half-open cells with the
  last row/column closed, per-axis degeneracy, OUTSIDE counting, canonical
  outcome signatures (``EMPTY`` for an empty reveal).
* ``cost_table`` / ``risk_definitions``: unit cost 1.0 per reveal; Brier risk
  primary, Bayes 0-1 secondary, never mixed.
* ``endpoints``: FIXED_RISK (rho=0, B_cap=6) censored expected cost,
  unresolved rate and resolved-only mean with count; FIXED_BUDGET (B=1..6)
  terminal primary risk and terminal undecided rate read off the same path.
* ``scalable_pruning``: LB_UNIT, LB_ISO (K_LB=2 subset enumeration with the
  K_LB+1 fallback), LB_B1 (one Bellman lifting step) and R_LB (K_SEP=2
  separation depth, degenerate to 0 above K_SEP*c), an admissible interval
  branch-and-bound for the capped fixed-risk and the fixed-budget endpoint, an
  exact full-tree reference with a state cap, the signature quotient used for
  memoisation only, and a two-phase lexicographic refinement that re-derives
  the policy on the un-quotiented problem.
* ``split_protocol``: codepoint-order mechanism holdout with one confirmatory
  family per non-holdout mechanism, applied at the cell level so that the
  change/control twins always land in the same split.
* aggregation with hard refusal of confirmatory input on the reporting path.

Exactness: every probability, risk, cost, bound and interval endpoint is a
``fractions.Fraction``. The only float path is the inherited entropy score of
QUERY_AGNOSTIC_IG / QUERY_BALD_LIKE inside the imported M5B selectors, exactly
as frozen. There is no epsilon, no tolerance and no float in pruning.

Rational geometry: all coordinates and radii are multiples of 1/100 and all
direction vectors are canonical coprime integer triples, as the prereg
requires; contact is the exact rational comparison
``dx*dx + dy*dy <= (r1 + r2)**2``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
# tracemalloc is imported lazily inside the search so the hot path stays cheap.

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from fractureflow import obs_equiv_m5b as _frozen_controls
from fractureflow.obs_equiv_m5a_reference import (
    NO_INFORMATIVE_ACTION,
    STATUS_NO_INFORMATIVE,
    STATUS_OK,
    STATUS_UNRESOLVED,
    Action,
    CostResult,
    OptResult,
    PolicyNode,
    Problem,
    exact_fraction,
    initial_belief,
    no_informative_action,
    opt_fixed_budget,
    outcome_cells,
    policy_key,
    posterior,
    replay_policy,
    risk_bayes_01,
    risk_brier,
    v1,
)

SCHEMA = "obs_equiv.m5c_impl.core.v1"
PREREG_SHA256 = "ddfbb3dad826b18662403f8b5a6fb632cc5c32551796f08ab49accec77b8a2ce"
M5A_MODULE_SHA256 = "e4919d6fd81a90862893f8132f1483ca2123f041fabd1e244b23dd696a0250ec"

GRID_G = 6
COST_PER_REVEAL = Fraction(1)
RHO_PRIMARY = Fraction(0)
B_CAP = 6
BUDGET_GRID = (1, 2, 3, 4, 5, 6)
REPLICATES_STOCHASTIC = 32
REPLICATES_DETERMINISTIC = 1

K_LB = 2
K_SEP = 2
K_BELL = 1
N_EXPAND = 200_000
N_FULL_STATES = 200_000
T_MAX_SECONDS = 600
M_MAX_BYTES = 1_073_741_824
MAX_TREE_NODES = 400_000

STOP = "STOP"
RISK_REACHED = "RISK_REACHED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
NO_AFFORDABLE_ACTION = "NO_AFFORDABLE_ACTION"
NO_INFORMATIVE_AT_STOP = "NO_INFORMATIVE_AT_STOP"
REFERENCE_POLICY = "REFERENCE_POLICY"

ROLE_CHANGE = "QUERY_CHANGED"
ROLE_CONTROL = "QUERY_INVARIANT"
ROLE_NO_INFORMATIVE = NO_INFORMATIVE_ACTION

UNIT_STATUS_CHANGE = "CHANGE"
UNIT_STATUS_CONTROL = "CONTROL"
UNIT_STATUS_NO_INFORMATIVE = "NO_INFORMATIVE"

SPLIT_DEVELOPMENT = "development"
SPLIT_CONFIRMATORY = "confirmatory"

CERTIFIED = "CERTIFIED"
INFEASIBLE = "INFEASIBLE"
CAPPED = "CAPPED"

CERTIFIED_OPTIMAL_CAPPED = "CERTIFIED_OPTIMAL_CAPPED"
CERTIFIED_INTERVAL = "CERTIFIED_INTERVAL"
TREE_INFEASIBLE = "TREE_INFEASIBLE"
NOT_COMPUTED = "NOT_COMPUTED"
RELATIVE_GAP_UNDEFINED_ZERO_LOWER = "RELATIVE_GAP_UNDEFINED_ZERO_LOWER"
RELATIVE_GAP_UNDEFINED_ZERO_REFERENCE = "RELATIVE_GAP_UNDEFINED_ZERO_REFERENCE"
IMPLEMENTATION_DEFECT = "IMPLEMENTATION_DEFECT"

MECHANISMS = ("CHAIN_VS_DIRECT", "CROSSING_VS_PASSING", "HIDDEN_BRIDGE")
FAMILIES = ("WF_COMPACT", "WF_STANDARD", "WF_WIDE")
INDICES = (0, 1, 2, 3)
ROLES = (ROLE_CHANGE, ROLE_CONTROL)

CONTROL_IDS = (
    "RANDOM",
    "GEOMETRIC_FROZEN",
    "SPACE_FILL",
    "QUERY_AGNOSTIC_IG",
    "QUERY_BALD_LIKE",
    "EC2_LIKE",
    "GREEDY_RISK_ONE_STEP",
    "FULL_TREE_OPT",
)
SELECTOR_IDS = CONTROL_IDS[:-1]

# The seven selectors are the frozen M5B objects, not re-typed pseudocode.
SELECTORS = {
    "RANDOM": (_frozen_controls.select_random, "PUBLIC_ONLY", True),
    "GEOMETRIC_FROZEN": (_frozen_controls.select_geometric_frozen, "PUBLIC_ONLY", False),
    "SPACE_FILL": (_frozen_controls.select_space_fill, "PUBLIC_ONLY", False),
    "QUERY_AGNOSTIC_IG": (
        _frozen_controls.select_query_agnostic_ig,
        "SHARED_MODEL_NO_QUERY",
        False,
    ),
    "QUERY_BALD_LIKE": (_frozen_controls.select_query_bald_like, "SHARED_MODEL", False),
    "EC2_LIKE": (_frozen_controls.select_ec2_like, "SHARED_MODEL", False),
    "GREEDY_RISK_ONE_STEP": (
        _frozen_controls.select_greedy_risk_one_step,
        "SHARED_MODEL",
        False,
    ),
}


class ContractViolation(RuntimeError):
    """Implementation-level inconsistency (never a protocol result)."""


def _fail_closed(reason: str) -> None:
    raise SystemExit(f"M5C-FAIL-CLOSED: {reason}")


# ---------------------------------------------------------------------------
# Canonical JSON, exact fractions, rational geometry
# ---------------------------------------------------------------------------


def canonical_json(document) -> str:
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signature_of(objects: Sequence[Mapping]) -> str:
    """Frozen outcome signature: sha256(canonical_json(payload list))[:16]."""
    if not objects:
        return "EMPTY"
    payloads = sorted((canonical_json(o) for o in objects))
    return sha256_text("[" + ",".join(payloads) + "]")[:16]


def _sq(value: Fraction) -> Fraction:
    return value * value


def _round2(value: Fraction) -> Fraction:
    """Nearest multiple of 1/100 (half up); keeps every coordinate rational."""
    scaled = value * 100
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    if remainder >= Fraction(1, 2):
        floor += 1
    return Fraction(floor, 100)


def _xy(center: Sequence) -> tuple:
    return exact_fraction(center[0], "x"), exact_fraction(center[1], "y")


def disk_contact(c1, r1, c2, r2) -> bool:
    """Exact contact: squared centre distance <= squared radius sum."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return _sq(dx) + _sq(dy) <= _sq(r1 + r2)


def canonical_direction(nx: int, ny: int) -> tuple:
    """Coprime integer direction pair with canonical sign; returns 3-tuple."""
    if nx == 0 and ny == 0:
        raise ValueError("zero direction is not a direction")
    g = math.gcd(abs(nx), abs(ny))
    nx, ny = nx // g, ny // g
    if nx < 0 or (nx == 0 and ny < 0):
        nx, ny = -nx, -ny
    return (nx, ny, 0)


def _num(value: Fraction) -> float:
    """Serialise a rational as a JSON number, asserting exact round-trip."""
    as_float = float(value)
    if Fraction(str(as_float)) != value:
        raise ContractViolation(f"rational {value} does not round-trip through float")
    return as_float


# ---------------------------------------------------------------------------
# Public-only grid (G=6) — same interface as the frozen M5B grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    x_edges: tuple
    y_edges: tuple
    x_degenerate: bool
    y_degenerate: bool

    @property
    def nx(self) -> int:
        return len(self.x_edges) - 1

    @property
    def ny(self) -> int:
        return len(self.y_edges) - 1

    def cell_ids(self) -> tuple:
        return tuple(f"R{i}C{j}" for i in range(self.ny) for j in range(self.nx))

    @staticmethod
    def parse(cell_id: str) -> tuple:
        if not (cell_id.startswith("R") and "C" in cell_id):
            raise ValueError(f"bad cell id {cell_id!r}")
        row, col = cell_id[1:].split("C", 1)
        return int(row), int(col)

    def index_of(self, x: Fraction, y: Fraction) -> tuple:
        if self.nx == 1:
            j = 0
        else:
            x0, x1 = self.x_edges[0], self.x_edges[-1]
            j = int((x - x0) * self.nx / (x1 - x0))
            j = 0 if j < 0 else (self.nx - 1 if j >= self.nx else j)
        if self.ny == 1:
            i = 0
        else:
            y0, y1 = self.y_edges[0], self.y_edges[-1]
            i = int((y - y0) * self.ny / (y1 - y0))
            i = 0 if i < 0 else (self.ny - 1 if i >= self.ny else i)
        return i, j

    def cell_id_at(self, x: Fraction, y: Fraction) -> str:
        i, j = self.index_of(x, y)
        return f"R{i}C{j}"

    def cell_bounds(self, cell_id: str) -> tuple:
        i, j = self.parse(cell_id)
        return (self.x_edges[j], self.x_edges[j + 1], self.y_edges[i], self.y_edges[i + 1])

    def cell_center(self, cell_id: str) -> tuple:
        x0, x1, y0, y1 = self.cell_bounds(cell_id)
        return ((x0 + x1) / 2, (y0 + y1) / 2)


def build_grid(public_observation: Mapping) -> Grid:
    """PUBLIC_ONLY addressing: bbox of visible disk centres plus query endpoints."""
    points = []
    for obj in public_observation.get("observed_objects", ()):
        points.append(_xy(obj["center"]))
    params = public_observation.get("query", {}).get("params", {})
    if "A" in params and "B" in params:
        points.append(_xy(params["A"]))
        points.append(_xy(params["B"]))
    if not points:
        _fail_closed("public observation has no addressable points (empty bbox)")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_degenerate = x0 == x1
    y_degenerate = y0 == y1
    if x_degenerate:
        x_edges = (x0, x0)
    else:
        x_edges = tuple(x0 + (x1 - x0) * Fraction(j, GRID_G) for j in range(GRID_G + 1))
    if y_degenerate:
        y_edges = (y0, y0)
    else:
        y_edges = tuple(y0 + (y1 - y0) * Fraction(i, GRID_G) for i in range(GRID_G + 1))
    return Grid(x_edges, y_edges, x_degenerate, y_degenerate)


def public_query_points(public_observation: Mapping) -> tuple:
    params = public_observation["query"]["params"]
    return _xy(params["A"]), _xy(params["B"])


def visible_centers(public_observation: Mapping) -> tuple:
    return tuple(_xy(o["center"]) for o in public_observation.get("observed_objects", ()))


# ---------------------------------------------------------------------------
# Declared world families and deterministic pool generation
# ---------------------------------------------------------------------------

FAMILY_SPECS = {
    "WF_COMPACT": {
        "bbox_xy": (Fraction(4), Fraction(3)),
        "visible_radii": (
            Fraction(36, 100),
            Fraction(44, 100),
            Fraction(40, 100),
            Fraction(42, 100),
            Fraction(38, 100),
            Fraction(44, 100),
        ),
        "visible_chain_disks": (3, 4),
        "step_x": Fraction(62, 100),
        "rise": Fraction(20, 100),
        "gap_width": (Fraction(30, 100), Fraction(40, 100)),
        "x0": Fraction(1),
        "y0": Fraction(60, 100),
        "hidden_radius": Fraction(24, 100),
        "chain_counts": (1, 2, 3, 4),
        "bridge_counts": (1, 2, 3, 2),
        "single_bridge_radius": Fraction(30, 100),
        "decoy_radius": Fraction(6, 100),
    },
    "WF_STANDARD": {
        "bbox_xy": (Fraction(6), Fraction(4)),
        "visible_radii": (
            Fraction(46, 100),
            Fraction(58, 100),
            Fraction(52, 100),
            Fraction(60, 100),
            Fraction(50, 100),
            Fraction(58, 100),
        ),
        "visible_chain_disks": (4, 5),
        "step_x": Fraction(82, 100),
        "rise": Fraction(26, 100),
        "gap_width": (Fraction(40, 100), Fraction(55, 100)),
        "x0": Fraction(1),
        "y0": Fraction(60, 100),
        "hidden_radius": Fraction(30, 100),
        "chain_counts": (2, 3, 4, 5),
        "bridge_counts": (2, 3, 4, 2),
        "single_bridge_radius": Fraction(36, 100),
        "decoy_radius": Fraction(8, 100),
    },
    "WF_WIDE": {
        "bbox_xy": (Fraction(8), Fraction(5)),
        "visible_radii": (
            Fraction(56, 100),
            Fraction(74, 100),
            Fraction(62, 100),
            Fraction(72, 100),
            Fraction(60, 100),
            Fraction(74, 100),
        ),
        "visible_chain_disks": (5, 6),
        "step_x": Fraction(102, 100),
        "rise": Fraction(32, 100),
        "gap_width": (Fraction(55, 100), Fraction(75, 100)),
        "x0": Fraction(1),
        "y0": Fraction(60, 100),
        "hidden_radius": Fraction(36, 100),
        "chain_counts": (2, 3, 4, 5),
        "bridge_counts": (2, 3, 4, 3),
        "single_bridge_radius": Fraction(42, 100),
        "decoy_radius": Fraction(10, 100),
    },
}

# Evenly spaced rational abscissae (multiples of 1/100) used to place hidden disks.
_CHAIN_T = {
    1: (Fraction(50, 100),),
    2: (Fraction(32, 100), Fraction(68, 100)),
    3: (Fraction(22, 100), Fraction(50, 100), Fraction(78, 100)),
    4: (Fraction(18, 100), Fraction(39, 100), Fraction(61, 100), Fraction(82, 100)),
    5: (
        Fraction(15, 100),
        Fraction(32, 100),
        Fraction(50, 100),
        Fraction(68, 100),
        Fraction(85, 100),
    ),
}

_NORMAL_PATTERN = (
    (0, 0, 1),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (1, -1, 0),
)


def _gap_shift(
    r_a: Fraction, r_b: Fraction, rise: Fraction, step_x: Fraction, spec: Mapping, index: int
) -> Fraction:
    """Smallest hundredth shift (along x) that makes the edge gap reach the target."""
    low, high = spec["gap_width"]
    target = low + (high - low) * Fraction(index, 4)
    need = r_a + r_b + target
    centre_dx = step_x
    while centre_dx <= Fraction(3):
        if _sq(centre_dx) + _sq(rise) >= _sq(need):
            break
        centre_dx += Fraction(1, 100)
    else:
        raise ContractViolation(f"no gap shift found in [{low}, {high}]")
    gap_ok_low = _sq(centre_dx) + _sq(rise) >= _sq(r_a + r_b + low)
    gap_ok_high = _sq(centre_dx) + _sq(rise) <= _sq(r_a + r_b + high)
    if not (gap_ok_low and gap_ok_high):
        raise ContractViolation(
            f"edge gap outside declared family range for index {index}: dx {centre_dx}"
        )
    return centre_dx - step_x


def _visible_chain(mechanism: str, family: str, index: int) -> tuple:
    """Visible staircase chain, cut once; returns (objects, A, B, A_end, B_end, spec)."""
    spec = FAMILY_SPECS[family]
    n = spec["visible_chain_disks"][index % 2]
    radii = tuple(spec["visible_radii"][(i + index) % len(spec["visible_radii"])] for i in range(n))
    x0, y0 = spec["x0"], spec["y0"]
    base = [
        (x0 + spec["step_x"] * i, y0 + spec["rise"] * i) for i in range(n)
    ]
    cut = index % (n - 1)
    r_a, r_b = radii[cut], radii[cut + 1]
    shift = _gap_shift(r_a, r_b, spec["rise"], spec["step_x"], spec, index)
    points = []
    for i, (x, y) in enumerate(base):
        points.append((x + shift, y) if i > cut else (x, y))
    objects = []
    for i, (x, y) in enumerate(points):
        objects.append(
            {
                "id": f"v{i}",
                "center": (_round2(x), _round2(y), Fraction(0)),
                "normal": (0, 0, 1),
                "radius": radii[i],
                "type": "DISK",
            }
        )
    return objects, objects[0], objects[-1], objects[cut], objects[cut + 1], spec


def _pt(a: Mapping, b: Mapping, perp, t: Fraction, q: Fraction) -> tuple:
    ax, ay, _az = a["center"]
    bx, by, _bz = b["center"]
    x = ax + t * (bx - ax) + q * perp[0]
    y = ay + t * (by - ay) + q * perp[1]
    return (_round2(x), _round2(y), Fraction(0))


def _hidden_object(unit_tag: str, j: int, center, radius: Fraction, normal_index: int) -> dict:
    nx, ny, nz = _NORMAL_PATTERN[normal_index % len(_NORMAL_PATTERN)]
    return {
        "id": f"x{j}",
        "center": center,
        "normal": canonical_direction(nx, ny) if (nx, ny) != (0, 0) else (0, 0, 1),
        "radius": radius,
        "type": "DISK",
    }


def _hidden_variants(
    mechanism: str,
    family: str,
    index: int,
    a_end: Mapping,
    b_end: Mapping,
    spec: Mapping,
) -> list:
    """Eight hidden configurations: four Q=1 then four Q=0 (declared classes)."""
    ax, ay, _ = a_end["center"]
    bx, by, _ = b_end["center"]
    delta = (bx - ax, by - ay)
    perp = (-delta[1], delta[0])
    base = spec["hidden_radius"]
    variants = []

    def side_chain(start, end, count, radii_list, q_start, q_step, tag):
        out = []
        ts = _CHAIN_T[count]
        for i, t in enumerate(ts):
            q = q_start + q_step * i
            center = _pt(start, end, perp, t, q)
            out.append(
                _hidden_object(tag, i, center, radii_list[i % len(radii_list)], index + i)
            )
        return out

    def alt_radii(count, extra=Fraction(0)):
        return tuple(base + extra + Fraction(3, 100) * (i % 2) for i in range(count))

    if mechanism == "CHAIN_VS_DIRECT":
        counts = spec["chain_counts"]
        for j in range(4):
            m = counts[j]
            curve = (Fraction(0), Fraction(6, 100), Fraction(-4, 100), Fraction(5, 100))[j]
            ts = _CHAIN_T[m]
            disks = []
            for i, t in enumerate(ts):
                q = curve * (t - Fraction(50, 100))
                disks.append(
                    _hidden_object(
                        f"chain{j}",
                        i,
                        _pt(a_end, b_end, perp, t, q),
                        base + Fraction(3, 100) * (i % 2),
                        index + j + i,
                    )
                )
            variants.append(disks)
        splits = (
            (Fraction(26, 100), Fraction(74, 100)),
            (Fraction(30, 100), Fraction(80, 100)),
            (Fraction(22, 100), Fraction(70, 100)),
            (Fraction(28, 100), Fraction(78, 100)),
        )
        shapes = ((1, 1), (2, 1), (1, 2), (3, 1))
        sub_radii = tuple(_round2(value * Fraction(3, 4)) for value in alt_radii(3))
        for j, ((p, qn), (sa, sb)) in enumerate(zip(shapes, splits), start=4):
            disks = []
            for i in range(p):
                t = (
                    Fraction(6, 100) + (sa - Fraction(6, 100)) * Fraction(i, p - 1)
                    if p > 1
                    else (Fraction(6, 100) + sa) / 2
                )
                disks.append(
                    _hidden_object(
                        f"gapA{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, Fraction(2, 100) * (i + 1)),
                        sub_radii[i],
                        index + j + i,
                    )
                )
            for i in range(qn):
                t = (
                    sb + (Fraction(94, 100) - sb) * Fraction(i, qn - 1)
                    if qn > 1
                    else (sb + Fraction(94, 100)) / 2
                )
                disks.append(
                    _hidden_object(
                        f"gapB{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, Fraction(-2, 100) * (i + 1)),
                        sub_radii[(i + 1) % 3],
                        index + j + i + 2,
                    )
                )
            variants.append(disks)
    elif mechanism == "CROSSING_VS_PASSING":
        shapes = ((1, 1), (1, 2), (2, 1), (2, 2))
        for j, (m1, m2) in enumerate(shapes):
            disks = []
            x_radius = base + Fraction(6, 100)
            disks.append(
                _hidden_object(
                    f"cross{j}",
                    0,
                    _pt(a_end, b_end, perp, Fraction(50, 100), Fraction(0)),
                    x_radius,
                    index + j,
                )
            )
            t1 = (Fraction(26, 100),) if m1 == 1 else (Fraction(16, 100), Fraction(36, 100))
            q1 = (Fraction(4, 100),) if m1 == 1 else (Fraction(5, 100), Fraction(2, 100))
            for i, (t, q) in enumerate(zip(t1, q1)):
                disks.append(
                    _hidden_object(
                        f"traceA{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, q),
                        base,
                        index + j + i + 1,
                    )
                )
            t2 = (Fraction(74, 100),) if m2 == 1 else (Fraction(64, 100), Fraction(84, 100))
            q2 = (Fraction(-4, 100),) if m2 == 1 else (Fraction(-2, 100), Fraction(-5, 100))
            for i, (t, q) in enumerate(zip(t2, q2)):
                disks.append(
                    _hidden_object(
                        f"traceB{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, q),
                        base,
                        index + j + i + 3,
                    )
                )
            variants.append(disks)
        splits = (
            (Fraction(34, 100), Fraction(66, 100)),
            (Fraction(38, 100), Fraction(70, 100)),
            (Fraction(30, 100), Fraction(62, 100)),
            (Fraction(36, 100), Fraction(68, 100)),
        )
        tip = _round2(base - Fraction(10, 100))
        for j, ((m1, m2), (sa, sb)) in enumerate(zip(shapes, splits), start=4):
            disks = []
            t1 = (sa,) if m1 == 1 else (sa - Fraction(20, 100), sa)
            q1 = (Fraction(6, 100),) if m1 == 1 else (Fraction(7, 100), Fraction(6, 100))
            for i, (t, q) in enumerate(zip(t1, q1)):
                radius = tip if i == len(t1) - 1 else base
                disks.append(
                    _hidden_object(
                        f"passA{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, q),
                        radius,
                        index + j + i + 1,
                    )
                )
            t2 = (sb,) if m2 == 1 else (sb, sb + Fraction(20, 100))
            q2 = (Fraction(-6, 100),) if m2 == 1 else (Fraction(-6, 100), Fraction(-7, 100))
            for i, (t, q) in enumerate(zip(t2, q2)):
                radius = tip if i == 0 else base
                disks.append(
                    _hidden_object(
                        f"passB{j}",
                        len(disks),
                        _pt(a_end, b_end, perp, t, q),
                        radius,
                        index + j + i + 4,
                    )
                )
            variants.append(disks)
    elif mechanism == "HIDDEN_BRIDGE":
        counts = spec["bridge_counts"]
        for j in range(4):
            b = counts[j]
            if b == 1:
                disks = [
                    _hidden_object(
                        f"bridge{j}",
                        0,
                        _pt(a_end, b_end, perp, Fraction(50, 100), Fraction(0)),
                        spec["single_bridge_radius"],
                        index + j,
                    )
                ]
            else:
                ts = _CHAIN_T[b]
                extra = Fraction(4, 100) if j == 3 else Fraction(0)
                disks = []
                for i, t in enumerate(ts):
                    q = Fraction(2, 100) * (i - Fraction(b - 1, 2)) if j == 2 else Fraction(0)
                    disks.append(
                        _hidden_object(
                            f"bridge{j}",
                            i,
                            _pt(a_end, b_end, perp, t, q),
                            base + extra + Fraction(3, 100) * (i % 2),
                            index + j + i,
                        )
                    )
            variants.append(disks)
        decoy_counts = counts
        for j in range(4):
            d = decoy_counts[j]
            ts = {
                1: (Fraction(50, 100),),
                2: (Fraction(44, 100), Fraction(56, 100)),
                3: (Fraction(42, 100), Fraction(50, 100), Fraction(58, 100)),
                4: (
                    Fraction(42, 100),
                    Fraction(47, 100),
                    Fraction(53, 100),
                    Fraction(58, 100),
                ),
            }[d]
            disks = []
            for i, t in enumerate(ts):
                q = Fraction(6, 100) * (1 if i % 2 == 0 else -1)
                radius = spec["decoy_radius"] + Fraction(2, 100) * ((j + i) % 2)
                disks.append(
                    _hidden_object(
                        f"decoy{j}",
                        i,
                        _pt(a_end, b_end, perp, t, q),
                        radius,
                        index + j + i + 1,
                    )
                )
            variants.append(disks)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    return variants


def _contact_graph(objects: Sequence[Mapping]) -> dict:
    parent = {o["id"]: o["id"] for o in objects}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, a in enumerate(objects):
        ca = _xy(a["center"])
        ra = exact_fraction(a["radius"])
        for b in objects[i + 1 :]:
            if disk_contact(ca, ra, _xy(b["center"]), exact_fraction(b["radius"])):
                union(a["id"], b["id"])
    return {o["id"]: find(o["id"]) for o in objects}


def _variant_query(objects: Sequence[Mapping], aid: str, bid: str) -> int:
    comp = _contact_graph(objects)
    return int(comp[aid] == comp[bid])


def _serialise_object(obj: Mapping) -> dict:
    cx, cy, cz = obj["center"]
    nx, ny, nz = obj["normal"]
    return {
        "id": obj["id"],
        "center": [_num(cx), _num(cy), _num(cz)],
        "normal": [int(nx), int(ny), int(nz)],
        "radius": _num(exact_fraction(obj["radius"])),
        "type": obj["type"],
    }


def _public_object(obj: Mapping, public_id: str) -> dict:
    cx, cy, cz = obj["center"]
    nx, ny, nz = obj["normal"]
    return {
        "center": [_num(cx), _num(cy), _num(cz)],
        "normal": [int(nx), int(ny), int(nz)],
        "radius": _num(exact_fraction(obj["radius"])),
        "type": obj["type"],
        "public_id": public_id,
    }


def _weights_for(index: int) -> tuple:
    """Two equal-mass Q classes with positive integer within-class weights."""
    rotations = (
        ((1, 2, 1, 2), (2, 1, 2, 1)),
        ((2, 1, 1, 2), (1, 2, 1, 2)),
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((2, 2, 1, 1), (1, 1, 2, 2)),
    )
    q_one, q_zero = rotations[index % 4]
    return (*q_one, *q_zero)


def _build_unit_records(mechanism: str, family: str, index: int) -> tuple:
    visible, a_disk, b_disk, a_end, b_end, spec = _visible_chain(mechanism, family, index)
    hidden = _hidden_variants(mechanism, family, index, a_end, b_end, spec)
    if len(hidden) != 8:
        raise ContractViolation("a unit must carry exactly eight hidden variants")
    query_change = [1, 1, 1, 1, 0, 0, 0, 0]
    for h, q in zip(hidden, query_change):
        label = _variant_query(list(visible) + list(h), "v0", f"v{len(visible) - 1}")
        if label != q:
            raise ContractViolation(
                f"{mechanism}/{family}/{index}: declared Q={q} but contact graph gives {label}"
            )
    query = {
        "family": "TWO_POINT_CONNECTIVITY",
        "params": {
            "A": [_num(a_disk["center"][0]), _num(a_disk["center"][1]), _num(a_disk["center"][2])],
            "B": [_num(b_disk["center"][0]), _num(b_disk["center"][1]), _num(b_disk["center"][2])],
        },
        "query_id": f"q|{mechanism}|{family}|{index}",
    }
    observation = {
        "observed_objects": [
            _public_object(obj, f"p{i}") for i, obj in enumerate(visible)
        ],
        "query": query,
    }
    weights = _weights_for(index)
    if len(weights) != 8 or any(value <= 0 for value in weights):
        _fail_closed("manifest prior weights must be eight positive integers")
    change_id = f"{mechanism}|{family}|{ROLE_CHANGE}|{index}"
    control_id = f"{mechanism}|{family}|{ROLE_CONTROL}|{index}"
    public_change = {"unit_id": change_id, "observation": observation}
    public_control = {"unit_id": control_id, "observation": observation}
    visible_ids = [o["id"] for o in visible]
    evaluator_change = {
        "unit_id": change_id,
        "twin_unit_id": control_id,
        "mechanism": mechanism,
        "world_family": family,
        "role": ROLE_CHANGE,
        "index": index,
        "visible_object_ids": visible_ids,
        "prior_weights": list(weights),
        "query_labels": list(query_change),
        "scenes": [
            {"objects": [_serialise_object(o) for o in list(visible) + list(h)]}
            for h in hidden
        ],
    }
    evaluator_control = {
        "unit_id": control_id,
        "twin_unit_id": change_id,
        "mechanism": mechanism,
        "world_family": family,
        "role": ROLE_CONTROL,
        "index": index,
        "visible_object_ids": visible_ids,
        "prior_weights": list(weights),
        "query_labels": [0, 0, 0, 0, 0, 0, 0, 0],
        "scenes": [
            {"objects": [_serialise_object(o) for o in list(visible) + list(h)]}
            for h in hidden
        ],
        "twin_hidden_point_identical": True,
    }
    return public_change, evaluator_change, public_control, evaluator_control


def generate_pool() -> dict:
    """Deterministic M5C declared domain: 72 units (36 change + 36 control)."""
    public_rows = []
    evaluator_rows = []
    for mechanism in MECHANISMS:
        for family in FAMILIES:
            for index in INDICES:
                rows = _build_unit_records(mechanism, family, index)
                public_rows.extend([rows[0], rows[2]])
                evaluator_rows.extend([rows[1], rows[3]])
    if len(public_rows) != 72 or len(evaluator_rows) != 72:
        raise ContractViolation("pool size is not 72")
    return {"public": public_rows, "evaluator": evaluator_rows}


def pool_fingerprint(public_rows: Sequence[Mapping], evaluator_rows: Sequence[Mapping]) -> str:
    by_id = {row["unit_id"]: row for row in evaluator_rows}
    paired = sorted(((row, by_id[row["unit_id"]]) for row in public_rows), key=lambda p: p[0]["unit_id"])
    return sha256_bytes(
        canonical_json({"public": [a for a, _b in paired], "evaluator": [b for _a, b in paired]}).encode(
            "utf-8"
        )
    )


def assert_dataset_sha(public_rows: Sequence[Mapping], evaluator_rows: Sequence[Mapping], expected: str) -> str:
    if not expected:
        _fail_closed("dataset_sha256 is not registered")
    got = pool_fingerprint(public_rows, evaluator_rows)
    if got != expected:
        _fail_closed(f"dataset_sha256 mismatch: expected {expected}, recomputed {got}")
    return got


# ---------------------------------------------------------------------------
# Evaluator-side unit construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reveal:
    cell_id: str
    signature: str
    objects: tuple


@dataclass(frozen=True)
class UnitWorld:
    unit_id: str
    mechanism: str
    world_family: str
    role: str
    index: int
    public_observation: Mapping
    grid: Grid
    problem: Problem
    signatures: Mapping[str, Mapping[str, str]]
    payloads: Mapping[str, Mapping[str, tuple]]
    hidden_objects: Mapping[str, tuple]
    hidden_counts: tuple
    outside_count: int
    pure_z_pairs: int
    informative_cells: tuple
    status: str
    twin_unit_id: Optional[str] = None

    @property
    def eligible(self) -> bool:
        return self.status == UNIT_STATUS_CHANGE


def _payload(obj: Mapping) -> dict:
    return {k: obj[k] for k in ("type", "center", "normal", "radius")}


def build_world(public_record: Mapping, evaluator_record: Mapping) -> UnitWorld:
    """Evaluator side: H / b0 / Q plus the frozen action-outcome contract."""
    if public_record["unit_id"] != evaluator_record["unit_id"]:
        raise ContractViolation("public/evaluator unit_id mismatch")
    scenes = evaluator_record["scenes"]
    if not scenes:
        _fail_closed("evaluator scene set is empty")
    if len(scenes) != 8:
        _fail_closed(f"unit must declare eight variants, got {len(scenes)}")
    observation = public_record["observation"]
    grid = build_grid(observation)
    visible = set(evaluator_record["visible_object_ids"])
    hypotheses = tuple(f"h{i}" for i in range(len(scenes)))
    visible_geometry = set()
    for scene in scenes:
        geometry = tuple(
            sorted(
                (
                    o["id"],
                    tuple(o["center"]),
                    tuple(o["normal"]),
                    o["radius"],
                    o["type"],
                )
                for o in scene["objects"]
                if o["id"] in visible
            )
        )
        visible_geometry.add(geometry)
    if len(visible_geometry) != 1:
        raise ContractViolation(f"{public_record['unit_id']}: visible geometry differs across H")
    prior = evaluator_record["prior_weights"]
    labels = evaluator_record["query_labels"]
    if len(prior) != len(hypotheses) or len(labels) != len(hypotheses):
        raise ContractViolation("prior/query length does not match |H|")
    if any(int(w) <= 0 for w in prior):
        _fail_closed("manifest prior weights must be positive integers")
    query = {h: int(bool(labels[i])) for i, h in enumerate(hypotheses)}
    actions = []
    signatures = {}
    payloads = {}
    hidden_counts = []
    pure_z_pairs = 0
    hidden_by_h = {}
    for h, scene in zip(hypotheses, scenes):
        hidden_objs = tuple(_payload(o) for o in scene["objects"] if o["id"] not in visible)
        hidden_by_h[h] = hidden_objs
        hidden_counts.append(len(hidden_objs))
    for cell_id in grid.cell_ids():
        per_h = {}
        per_h_payload = {}
        for h in hypotheses:
            objects = []
            for obj in hidden_by_h[h]:
                x, y = _xy(obj["center"])
                if x < grid.x_edges[0] or x > grid.x_edges[-1]:
                    continue
                if y < grid.y_edges[0] or y > grid.y_edges[-1]:
                    continue
                if grid.cell_id_at(x, y) == cell_id:
                    objects.append(obj)
            per_h[h] = signature_of(objects)
            per_h_payload[h] = tuple(sorted(objects, key=canonical_json))
        groups = {}
        for h in hypotheses:
            groups.setdefault(per_h[h], []).append(h)
        partition = [sorted(members) for members in groups.values()]
        actions.append(Action.of(cell_id, COST_PER_REVEAL, partition))
        signatures[cell_id] = per_h
        payloads[cell_id] = {
            sig: next(per_h_payload[h] for h in hypotheses if per_h[h] == sig) for sig in groups
        }
    outside_count = 0
    for h in hypotheses:
        for obj in hidden_by_h[h]:
            x, y = _xy(obj["center"])
            if (
                x < grid.x_edges[0]
                or x > grid.x_edges[-1]
                or y < grid.y_edges[0]
                or y > grid.y_edges[-1]
            ):
                outside_count += 1
    # pure-z variation: pairs whose x/y projection and radii/normals agree, z differs
    for i, h in enumerate(hypotheses):
        for h2 in hypotheses[i + 1 :]:
            left = hidden_by_h[h]
            right = hidden_by_h[h2]
            if len(left) != len(right):
                continue

            def flat(objects):
                out = []
                for o in objects:
                    cx, cy, cz = o["center"]
                    out.append(
                        (
                            o["type"],
                            exact_fraction(cx),
                            exact_fraction(cy),
                            tuple(int(v) for v in o["normal"]),
                            exact_fraction(o["radius"]),
                        )
                    )
                return sorted(out)

            def flat_full(objects):
                out = []
                for o in objects:
                    cx, cy, cz = o["center"]
                    out.append(
                        (
                            o["type"],
                            exact_fraction(cx),
                            exact_fraction(cy),
                            exact_fraction(cz),
                            tuple(int(v) for v in o["normal"]),
                            exact_fraction(o["radius"]),
                        )
                    )
                return sorted(out)

            if flat(left) == flat(right) and flat_full(left) != flat_full(right):
                pure_z_pairs += 1
    problem = Problem.build(hypotheses, prior, query, actions)
    informative = tuple(
        cell_id
        for cell_id in grid.cell_ids()
        if len({signatures[cell_id][h] for h in hypotheses}) >= 2
    )
    belief0 = initial_belief(problem)
    risk0 = risk_brier(belief0, query)
    if all(query[h] == 0 for h in hypotheses) or all(query[h] == 1 for h in hypotheses):
        status = UNIT_STATUS_CONTROL
        if risk0 != 0:
            raise ContractViolation("control unit has non-zero initial risk")
    elif no_informative_action(problem):
        status = UNIT_STATUS_NO_INFORMATIVE
    else:
        status = UNIT_STATUS_CHANGE
    return UnitWorld(
        unit_id=public_record["unit_id"],
        mechanism=evaluator_record["mechanism"],
        world_family=evaluator_record["world_family"],
        role=evaluator_record["role"],
        index=evaluator_record["index"],
        public_observation=observation,
        grid=grid,
        problem=problem,
        signatures=signatures,
        payloads=payloads,
        hidden_objects=hidden_by_h,
        hidden_counts=tuple(hidden_counts),
        outside_count=outside_count,
        pure_z_pairs=pure_z_pairs,
        informative_cells=informative,
        status=status,
        twin_unit_id=evaluator_record.get("twin_unit_id"),
    )


# ---------------------------------------------------------------------------
# C-NT-1..5 structural certificates
# ---------------------------------------------------------------------------


def _hidden_config_signature(objects: Sequence[Mapping]) -> tuple:
    out = []
    for o in objects:
        cx, cy, cz = o["center"]
        nx, ny, nz = o["normal"]
        out.append(
            (
                o["type"],
                exact_fraction(cx),
                exact_fraction(cy),
                exact_fraction(cz),
                int(nx),
                int(ny),
                int(nz),
                exact_fraction(o["radius"]),
            )
        )
    return tuple(sorted(out))


def cert_non_translation(objects_per_variant: Sequence[Sequence[Mapping]]) -> dict:
    """C-NT-1: no two complete hidden configurations are rational translations."""
    configs = [_hidden_config_signature(objs) for objs in objects_per_variant]
    pairs = []
    bad = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            pairs.append((i, j))
            a, b = configs[i], configs[j]
            if len(a) != len(b):
                continue
            shape_a = tuple(sorted((t, nx, ny, nz, r) for (t, _x, _y, _z, nx, ny, nz, r) in a))
            shape_b = tuple(sorted((t, nx, ny, nz, r) for (t, _x, _y, _z, nx, ny, nz, r) in b))
            if shape_a != shape_b:
                continue
            translated = False
            for src in a:
                for dst in b:
                    shift = (dst[1] - src[1], dst[2] - src[2], dst[3] - src[3])
                    moved = sorted(
                        (t, x + shift[0], y + shift[1], z + shift[2], nx, ny, nz, r)
                        for (t, x, y, z, nx, ny, nz, r) in a
                    )
                    if moved == list(b):
                        translated = True
                        break
                if translated:
                    break
            if translated:
                bad.append([i, j])
    return {
        "id": "C-NT-1",
        "what": "no two complete hidden configurations are rational translations",
        "checked_pairs": len(pairs),
        "translatable_pairs": bad,
        "ok": not bad,
    }


def cert_structural_visibility(signatures: Mapping[str, Mapping[str, str]]) -> dict:
    """C-NT-2: some addressable cell separates at least two hypotheses."""
    distinguishing = [
        cell
        for cell, per_h in signatures.items()
        if len(set(per_h.values())) >= 2
    ]
    return {
        "id": "C-NT-2",
        "what": "exists an addressable cell whose outcome signature varies over H",
        "distinguishing_cells": sorted(distinguishing),
        "count": len(distinguishing),
        "ok": bool(distinguishing),
    }


def cert_query_decisive(
    signatures: Mapping[str, Mapping[str, str]], query: Mapping[str, int], applicable: bool
) -> dict:
    """C-NT-3: some cell separates a Q-disagreeing pair (change units only)."""
    if not applicable:
        return {
            "id": "C-NT-3",
            "what": "exists a cell separating a Q-disagreeing hypothesis pair",
            "applicable": False,
            "ok": True,
            "reason": "control unit: Q is constant, the certificate is not applicable",
            "witnesses": [],
        }
    witnesses = []
    hyps = sorted(query)
    for cell, per_h in signatures.items():
        for i, h in enumerate(hyps):
            for h2 in hyps[i + 1 :]:
                if query[h] != query[h2] and per_h[h] != per_h[h2]:
                    witnesses.append([cell, h, h2])
                    break
    return {
        "id": "C-NT-3",
        "what": "exists a cell separating a Q-disagreeing hypothesis pair",
        "applicable": True,
        "ok": bool(witnesses),
        "witnesses": witnesses[:8],
        "witness_count": len(witnesses),
    }


def cert_public_identical(public_per_h: Sequence[Mapping]) -> dict:
    """C-NT-4: canonical public payload identical across H."""
    canon = [canonical_json(record) for record in public_per_h]
    hashes = [sha256_text(text) for text in canon]
    return {
        "id": "C-NT-4",
        "what": "public payload byte-identical across H",
        "variant_records": len(canon),
        "sha256": hashes[0] if hashes else None,
        "distinct_hashes": len(set(hashes)),
        "ok": len(set(hashes)) == 1 and bool(hashes),
    }


def cert_twin(
    change_hidden: Sequence[Sequence[Mapping]],
    control_hidden: Sequence[Sequence[Mapping]],
    control_query: Mapping[str, int],
) -> dict:
    """C-NT-5: twin control hidden configurations point-identical, Q identically zero."""
    same_count = len(change_hidden) == len(control_hidden)
    identical = same_count and all(
        _hidden_config_signature(a) == _hidden_config_signature(b)
        for a, b in zip(change_hidden, control_hidden)
    )
    q_zero = all(int(v) == 0 for v in control_query.values())
    return {
        "id": "C-NT-5",
        "what": "control twin hidden configurations point-identical to the change twin",
        "variants": len(control_hidden),
        "point_identical": bool(identical),
        "query_all_zero": bool(q_zero),
        "ok": bool(identical and q_zero),
    }


def hidden_configurations(world: UnitWorld) -> list:
    return [list(world.hidden_objects[h]) for h in sorted(world.hidden_objects)]


def unit_certificates(world: UnitWorld, twin: Optional[UnitWorld] = None) -> dict:
    """All five certificates for one unit, plus the three declared counters."""
    public_per_h = []
    for h in sorted(world.problem.hypotheses):
        public_per_h.append(
            {
                "query": world.public_observation["query"],
                "observed_objects": list(world.public_observation["observed_objects"]),
            }
        )
    certs = {
        "C-NT-1": cert_non_translation(hidden_configurations(world)),
        "C-NT-2": cert_structural_visibility(world.signatures),
        "C-NT-3": cert_query_decisive(
            world.signatures, world.problem.query, world.status == UNIT_STATUS_CHANGE
        ),
        "C-NT-4": cert_public_identical(public_per_h),
    }
    if world.status == UNIT_STATUS_CONTROL:
        if twin is None:
            raise ContractViolation("control unit certificate requires its change twin")
        certs["C-NT-5"] = cert_twin(
            hidden_configurations(twin), hidden_configurations(world), world.problem.query
        )
    else:
        certs["C-NT-5"] = {
            "id": "C-NT-5",
            "what": "twin certificate applies to the QUERY_INVARIANT member of the pair",
            "applicable": False,
            "ok": True,
            "reason": "change unit: C-NT-5 is carried by its control twin",
        }
    counters = {
        "outside_objects": world.outside_count,
        "degenerate_bbox_axes": int(world.grid.x_degenerate) + int(world.grid.y_degenerate),
        "pure_z_variant_pairs": world.pure_z_pairs,
        "grid_shape": [world.grid.nx, world.grid.ny],
        "candidate_actions": len(world.problem.actions),
        "informative_cells": list(world.informative_cells),
        "informative_cell_count": len(world.informative_cells),
        "hidden_counts": list(world.hidden_counts),
    }
    return {
        "unit_id": world.unit_id,
        "mechanism": world.mechanism,
        "world_family": world.world_family,
        "role": world.role,
        "status": world.status,
        "eligible": world.eligible,
        "certificates": certs,
        "counters": counters,
        "all_certificates_pass": all(c["ok"] for c in certs.values()),
    }


# ---------------------------------------------------------------------------
# Capability-restricted selector context and trajectory expansion
# ---------------------------------------------------------------------------


def _public_context(world: UnitWorld, revealed: tuple, affordable: tuple, remaining: int):
    a, b = public_query_points(world.public_observation)
    return _frozen_controls.PublicContext(
        grid=world.grid,
        query_a=a,
        query_b=b,
        visible_centers=visible_centers(world.public_observation),
        revealed=revealed,
        affordable=tuple(sorted(affordable)),
        remaining_budget=remaining,
    )


def _actions_by_id(world: UnitWorld) -> dict:
    return {a.action_id: a for a in world.problem.actions}


def _make_context(world, capability, belief, revealed, affordable, remaining):
    public = _public_context(world, revealed, affordable, remaining)
    actions = _actions_by_id(world)
    if capability == "PUBLIC_ONLY":
        return public
    if capability == "SHARED_MODEL_NO_QUERY":
        return _frozen_controls.InfoContext(public=public, belief=belief, actions=actions)
    return _frozen_controls.QueryModelContext(
        public=public, belief=belief, actions=actions, query=world.problem.query
    )


def select_action(world: UnitWorld, strategy_id: str, belief, revealed, affordable, remaining, rng):
    selector, capability, _stochastic = SELECTORS[strategy_id]
    ctx = _make_context(world, capability, belief, revealed, affordable, remaining)
    return selector(ctx, rng)


@dataclass(frozen=True)
class Leaf:
    path: tuple
    probability: Fraction
    cost: int
    risks: tuple
    resolved: bool
    stop_reason: str
    bayes: tuple = ()


def _affordable(world: UnitWorld, revealed: tuple, spent: int, cap: int) -> tuple:
    if cap - spent < COST_PER_REVEAL:
        return ()
    revealed_ids = {r.cell_id for r in revealed}
    return tuple(cell for cell in world.grid.cell_ids() if cell not in revealed_ids)


def _all_non_informative(belief, world: UnitWorld, affordable: Sequence[str]) -> bool:
    current = risk_brier(belief, world.problem.query)
    actions = _actions_by_id(world)
    return all(v1(belief, actions[cell], world.problem.query) == current for cell in affordable)


def _reveal(world: UnitWorld, cell_id: str, cell) -> Reveal:
    signature = world.signatures[cell_id][sorted(set(cell))[0]]
    return Reveal(cell_id, signature, world.payloads[cell_id][signature])


def expand_tree(world: UnitWorld, strategy_id: str, rho=RHO_PRIMARY, cap=B_CAP) -> list:
    """Exact conditional-policy tree for a deterministic selector."""
    problem = world.problem
    belief0 = initial_belief(problem)
    leaves = []
    nodes = 0

    def walk(belief, revealed, spent, path, probability, risks, bayes):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_TREE_NODES:
            _fail_closed(f"tree node cap {MAX_TREE_NODES} exceeded for {world.unit_id}")
        risk = risk_brier(belief, problem.query)
        affordable = _affordable(world, revealed, spent, cap)
        if risk <= rho:
            leaves.append(Leaf(path, probability, spent, tuple(risks), True, RISK_REACHED, tuple(bayes)))
            return
        if spent >= cap:
            leaves.append(Leaf(path, probability, spent, tuple(risks), False, BUDGET_EXHAUSTED, tuple(bayes)))
            return
        if not affordable:
            leaves.append(Leaf(path, probability, spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes)))
            return
        if _all_non_informative(belief, world, affordable):
            leaves.append(Leaf(path, probability, spent, tuple(risks), False, NO_INFORMATIVE_AT_STOP, tuple(bayes)))
            return
        action_id = select_action(world, strategy_id, belief, revealed, affordable, cap - spent, None)
        if action_id == STOP or action_id is None:
            leaves.append(Leaf(path, probability, spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes)))
            return
        if action_id not in affordable:
            raise ContractViolation(f"{world.unit_id}: selector picked unaffordable {action_id!r}")
        action = _actions_by_id(world)[action_id]
        for cell, mass in outcome_cells(belief, action):
            child = posterior(belief, action, cell)
            walk(
                child,
                revealed + (_reveal(world, action_id, cell),),
                spent + 1,
                path + (action_id,),
                probability * mass,
                risks + [risk_brier(child, problem.query)],
                bayes + [risk_bayes_01(child, problem.query)],
            )

    walk(belief0, (), 0, (), Fraction(1), [], [])
    total = sum((leaf.probability for leaf in leaves), Fraction(0))
    if total != 1:
        raise ContractViolation(f"{world.unit_id}: tree probabilities sum to {total}, not 1")
    return leaves


def _sample_outcome(belief, action, rng: random.Random):
    cells = outcome_cells(belief, action)
    if len(cells) == 1:
        return cells[0]
    draw = rng.random()
    cumulative = 0.0
    for cell, mass in cells:
        cumulative += float(mass)
        if draw < cumulative:
            return cell, mass
    return cells[-1]


def sample_path(world: UnitWorld, strategy_id: str, rng: random.Random, rho=RHO_PRIMARY, cap=B_CAP) -> Leaf:
    problem = world.problem
    belief = initial_belief(problem)
    revealed = ()
    spent = 0
    path = ()
    risks = []
    bayes = []
    while True:
        risk = risk_brier(belief, problem.query)
        affordable = _affordable(world, revealed, spent, cap)
        if risk <= rho:
            return Leaf(path, Fraction(1), spent, tuple(risks), True, RISK_REACHED, tuple(bayes))
        if spent >= cap:
            return Leaf(path, Fraction(1), spent, tuple(risks), False, BUDGET_EXHAUSTED, tuple(bayes))
        if not affordable:
            return Leaf(path, Fraction(1), spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes))
        if _all_non_informative(belief, world, affordable):
            return Leaf(path, Fraction(1), spent, tuple(risks), False, NO_INFORMATIVE_AT_STOP, tuple(bayes))
        action_id = select_action(world, strategy_id, belief, revealed, affordable, cap - spent, rng)
        if action_id == STOP or action_id is None:
            return Leaf(path, Fraction(1), spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes))
        if action_id not in affordable:
            raise ContractViolation(f"{world.unit_id}: selector picked unaffordable {action_id!r}")
        action = _actions_by_id(world)[action_id]
        cell, _mass = _sample_outcome(belief, action, rng)
        belief = posterior(belief, action, cell)
        revealed = revealed + (_reveal(world, action_id, cell),)
        spent += 1
        path = path + (action_id,)
        risks.append(risk_brier(belief, problem.query))
        bayes.append(risk_bayes_01(belief, problem.query))


def leaf_risk_at_k(leaf: Leaf, root_risk: Fraction, k: int) -> Fraction:
    if k == 0:
        return root_risk
    if k <= len(leaf.risks):
        return leaf.risks[k - 1]
    return leaf.risks[-1] if leaf.risks else root_risk


def leaf_bayes_at_k(leaf: Leaf, root_bayes: Fraction, k: int) -> Fraction:
    if k == 0:
        return root_bayes
    if k <= len(leaf.bayes):
        return leaf.bayes[k - 1]
    return leaf.bayes[-1] if leaf.bayes else root_bayes


def readings_from_leaves(world: UnitWorld, leaves: Sequence[Leaf], cap: int = B_CAP) -> dict:
    if not leaves:
        _fail_closed(f"empty trajectory stream for {world.unit_id}")
    problem = world.problem
    root_risk = risk_brier(initial_belief(problem), problem.query)
    root_bayes = risk_bayes_01(initial_belief(problem), problem.query)
    total = sum((leaf.probability for leaf in leaves), Fraction(0))
    resolved_mass = sum((leaf.probability for leaf in leaves if leaf.resolved), Fraction(0))
    censored_cost = sum(
        (leaf.probability * Fraction(leaf.cost if leaf.resolved else cap) for leaf in leaves),
        Fraction(0),
    ) / total
    resolved_cost = sum(
        (leaf.probability * Fraction(leaf.cost) for leaf in leaves if leaf.resolved), Fraction(0)
    )
    resolved_mean = resolved_cost / resolved_mass if resolved_mass else None
    risk_at = []
    bayes_at = []
    undecided = []
    for k in range(0, cap + 1):
        risk_at.append(
            sum((leaf.probability * leaf_risk_at_k(leaf, root_risk, k) for leaf in leaves), Fraction(0))
            / total
        )
        bayes_at.append(
            sum((leaf.probability * leaf_bayes_at_k(leaf, root_bayes, k) for leaf in leaves), Fraction(0))
            / total
        )
        if k >= 1:
            undecided.append(
                sum(
                    (
                        leaf.probability
                        for leaf in leaves
                        if leaf_risk_at_k(leaf, root_risk, k) > 0
                    ),
                    Fraction(0),
                )
                / total
            )
    return {
        "resolution_mass": resolved_mass / total,
        "unresolved_mass": (total - resolved_mass) / total,
        "censored_expected_cost": censored_cost,
        "resolved_only_mean_cost": resolved_mean,
        "risk_at_k": risk_at,
        "bayes_at_k": bayes_at,
        "undecided_mass_at_k": undecided,
        "root_risk": root_risk,
        "root_bayes": root_bayes,
        "leaves": leaves,
    }


def seed_for(dataset_sha: str, unit_id: str, strategy_id: str, rep: int) -> int:
    material = f"{dataset_sha}|{unit_id}|{strategy_id}|rep{rep}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def run_strategy(world: UnitWorld, strategy_id: str, dataset_sha: str) -> dict:
    if strategy_id == "FULL_TREE_OPT":
        raise ValueError("FULL_TREE_OPT is a reference, not a selector")
    _selector, _capability, stochastic = SELECTORS[strategy_id]
    seeds = []
    if stochastic:
        leaves = []
        for rep in range(REPLICATES_STOCHASTIC):
            seed = seed_for(dataset_sha, world.unit_id, strategy_id, rep)
            leaves.append(sample_path(world, strategy_id, random.Random(seed)))
            seeds.append(seed)
    else:
        leaves = expand_tree(world, strategy_id)
    if not leaves:
        _fail_closed(f"empty trajectory stream for {world.unit_id}/{strategy_id}")
    readings = readings_from_leaves(world, leaves)
    return {
        "strategy": strategy_id,
        "stochastic": stochastic,
        "replicates": (len(leaves) if stochastic else REPLICATES_DETERMINISTIC),
        "leaf_count": len(leaves),
        "seeds": seeds,
        "deterministic": not stochastic,
        "resolution_mass": str(readings["resolution_mass"]),
        "unresolved_mass": str(readings["unresolved_mass"]),
        "censored_expected_cost": str(readings["censored_expected_cost"]),
        "resolved_only_mean_cost": (
            None if readings["resolved_only_mean_cost"] is None
            else str(readings["resolved_only_mean_cost"])
        ),
        "risk_at_k": [str(v) for v in readings["risk_at_k"]],
        "bayes_at_k": [str(v) for v in readings["bayes_at_k"]],
        "undecided_mass_at_k": [str(v) for v in readings["undecided_mass_at_k"]],
        "root_risk": str(readings["root_risk"]),
        "root_bayes": str(readings["root_bayes"]),
        "leaves": [
            {
                "path": list(leaf.path),
                "probability": str(leaf.probability),
                "cost": leaf.cost,
                "risks": [str(v) for v in leaf.risks],
                "bayes": [str(v) for v in leaf.bayes],
                "resolved": leaf.resolved,
                "stop_reason": leaf.stop_reason,
            }
            for leaf in leaves
        ],
    }


# ---------------------------------------------------------------------------
# Admissible lower bounds (exact rational, zero epsilon)
# ---------------------------------------------------------------------------


def _support(belief: Mapping) -> tuple:
    return tuple(sorted(h for h, p in belief.items() if p > 0))


def _splitting_actions(problem: Problem, belief: Mapping, remaining) -> tuple:
    out = []
    for action in sorted(problem.actions, key=lambda a: a.action_id):
        if action.cost > remaining:
            continue
        if len(outcome_cells(belief, action)) >= 2:
            out.append(action)
    return tuple(out)


def _cell_index(action: Action) -> dict:
    return {h: i for i, cell in enumerate(action.partition) for h in cell}


def _separates(index_a: Mapping, h: str, h2: str) -> bool:
    return index_a[h] != index_a[h2]


def _partners(belief: Mapping, query: Mapping, h: str) -> tuple:
    qh = query[h]
    return tuple(h2 for h2 in _support(belief) if h2 != h and query[h2] != qh)


def _iso_bound(
    belief: Mapping, query: Mapping, actions: Sequence[Action], k_lb: int = K_LB
) -> tuple:
    """LB_ISO plus a feasibility flag; returns (value, feasible, detail)."""
    support = _support(belief)
    indexes = {a.action_id: _cell_index(a) for a in actions}
    partner_map = {h: _partners(belief, query, h) for h in support}
    # any Q-cross pair inseparable by every affordable action => infeasible
    for h in support:
        for h2 in partner_map[h]:
            if h < h2:
                if not any(_separates(indexes[a.action_id], h, h2) for a in actions):
                    return (None, False, f"pair {h},{h2} is inseparable")
    c_min = min((a.cost for a in actions), default=COST_PER_REVEAL)
    total = Fraction(0)
    detail = {}
    for h in support:
        partners = partner_map[h]
        if not partners:
            detail[h] = 0
            continue
        best = None
        for size in range(1, k_lb + 1):
            for subset in itertools.combinations(actions, size):
                ids = [a.action_id for a in subset]
                if all(
                    any(_separates(indexes[aid], h, h2) for aid in ids) for h2 in partners
                ):
                    best = size
                    break
            if best is not None:
                break
        if best is None:
            best = k_lb + 1
        detail[h] = best
        total += belief[h] * c_min * best
    return (total, True, detail)


def _lb0(
    problem: Problem,
    belief: Mapping,
    remaining: Fraction,
    k_lb: int,
    rho: Fraction,
    cache: dict,
):
    """LB_0 = max(LB_UNIT, LB_ISO); None means +INF (provably infeasible)."""
    if risk_brier(belief, problem.query) <= rho:
        return Fraction(0)
    key = (_support(belief), remaining)
    if key in cache:
        return cache[key]
    split = _splitting_actions(problem, belief, remaining)
    iso, feasible, detail = _iso_bound(belief, problem.query, split, k_lb)
    if not feasible or any(value > remaining for value in detail.values()):
        cache[key] = None
        return None
    c_min = min((a.cost for a in problem.actions), default=COST_PER_REVEAL)
    value = max(c_min, iso)
    cache[key] = value
    return value


def lower_bounds(
    problem: Problem,
    belief: Mapping,
    remaining: Fraction,
    k_lb: int = K_LB,
    rho: Fraction = RHO_PRIMARY,
) -> dict:
    """LB_UNIT / LB_ISO / LB_B1 / LB_COMBINED at one state (exact fractions)."""
    remaining = exact_fraction(remaining, "remaining")
    risk = risk_brier(belief, problem.query)
    if risk <= rho:
        return {
            "lb_unit": Fraction(0),
            "lb_iso": Fraction(0),
            "lb_b1": Fraction(0),
            "lb_combined": Fraction(0),
            "infeasible": False,
            "reason": "pure",
            "d_lb": {},
        }
    split = _splitting_actions(problem, belief, remaining)
    c_min = min((a.cost for a in problem.actions), default=COST_PER_REVEAL)
    lb_unit = c_min
    iso, feasible, detail = _iso_bound(belief, problem.query, split, k_lb)
    if not feasible:
        return {
            "lb_unit": lb_unit,
            "lb_iso": None,
            "lb_b1": None,
            "lb_combined": None,
            "infeasible": True,
            "reason": detail,
            "d_lb": {},
        }
    if any(c_min * value > remaining for value in detail.values()):
        return {
            "lb_unit": lb_unit,
            "lb_iso": iso,
            "lb_b1": None,
            "lb_combined": None,
            "infeasible": True,
            "reason": "necessary separation cost exceeds remaining budget",
            "d_lb": detail,
        }
    # LB_B1: one Bellman lifting step. Non-splitting actions are dominated, so
    # restricting the min to splitting actions keeps the bound admissible and
    # tighter than the frozen "min over affordable" template.
    cache = {}
    best = None
    for action in split:
        term = action.cost
        dead = False
        for cell, mass in outcome_cells(belief, action):
            child = posterior(belief, action, cell)
            sub = _lb0(problem, child, remaining - action.cost, k_lb, rho, cache)
            if sub is None:
                dead = True
                break
            term += mass * sub
        if dead:
            continue
        if best is None or term < best:
            best = term
    combined = max(lb_unit, iso)
    if best is not None and best > combined:
        combined = best
    return {
        "lb_unit": lb_unit,
        "lb_iso": iso,
        "lb_b1": best,
        "lb_combined": combined,
        "infeasible": False,
        "reason": "ok",
        "d_lb": detail,
    }


def r_lb(belief: Mapping, query: Mapping, actions: Sequence[Action], beta, k_sep: int = K_SEP):
    """R_LB: sum of masses of Q-cross pairs inseparable within ``beta`` actions."""
    beta = exact_fraction(beta, "beta")
    if beta > k_sep * COST_PER_REVEAL:
        return Fraction(0)
    support = _support(belief)
    indexes = {a.action_id: _cell_index(a) for a in actions}
    total = Fraction(0)
    for i, h in enumerate(support):
        for h2 in support[i + 1 :]:
            if query[h] == query[h2]:
                continue
            # A sequence separates the pair iff at least one of its actions does,
            # so the size-1 check is already exhaustive for every beta >= 1.
            separable = any(_separates(index, h, h2) for index in indexes.values())
            if not separable:
                total += belief[h] * belief[h2]
    return total


# ---------------------------------------------------------------------------
# Signature quotient (value abstraction; policy certificates never use it)
# ---------------------------------------------------------------------------


def _quotient_key(problem: Problem, belief: Mapping, remaining: Fraction) -> tuple:
    support = _support(belief)
    affordable = [a for a in sorted(problem.actions, key=lambda a: a.action_id) if a.cost <= remaining]
    indexes = [(a.action_id, _cell_index(a)) for a in affordable]
    classes = {}
    for h in support:
        sig = (problem.query[h], tuple(index[h] for _aid, index in indexes))
        classes[sig] = classes.get(sig, Fraction(0)) + belief[h]
    return tuple(sorted((sig, mass) for sig, mass in classes.items()))


def signature_quotient(problem: Problem, belief: Mapping, remaining) -> dict:
    key = _quotient_key(problem, belief, exact_fraction(remaining, "remaining"))
    return {
        "classes": len(key),
        "support": len(_support(belief)),
        "ratio": Fraction(len(key), len(_support(belief))),
        "key": key,
    }


# ---------------------------------------------------------------------------
# Search caps and statistics
# ---------------------------------------------------------------------------


@dataclass
class SearchCaps:
    n_expand: int = N_EXPAND
    t_max_seconds: float = T_MAX_SECONDS
    m_max_bytes: int = M_MAX_BYTES
    measure_memory: bool = False


@dataclass
class SearchStats:
    states_expanded: int = 0
    states_memo_hits: int = 0
    branches_generated: int = 0
    peak_frontier: int = 0
    max_depth: int = 0
    capped: bool = False
    cap_reason: str = ""
    runtime_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "states_expanded": self.states_expanded,
            "states_memo_hits": self.states_memo_hits,
            "branches_generated": self.branches_generated,
            "peak_frontier": self.peak_frontier,
            "max_depth": self.max_depth,
            "capped": self.capped,
            "cap_reason": self.cap_reason,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


class _Searcher:
    """Shared memo/caps machinery for the two interval searches."""

    def __init__(self, problem: Problem, rho, caps: SearchCaps, k_lb: int = K_LB):
        self.problem = problem
        self.rho = exact_fraction(rho, "rho")
        self.caps = caps
        self.k_lb = k_lb
        self.stats = SearchStats()
        self.memo = {}
        self.rollout_memo = {}
        self._started = time.perf_counter()
        self._trace = None
        if caps.measure_memory:
            import tracemalloc

            tracemalloc.start()
            self._trace = tracemalloc

    def close(self) -> None:
        self.stats.runtime_seconds = time.perf_counter() - self._started
        if self._trace is not None:
            current, peak = self._trace.get_traced_memory()
            self.stats.peak_memory_bytes = peak
            self._trace.stop()

    def _cap_hit(self) -> Optional[str]:
        if self.stats.states_expanded >= self.caps.n_expand:
            return "states"
        if time.perf_counter() - self._started >= self.caps.t_max_seconds:
            return "time"
        if self._trace is not None and self._trace.get_traced_memory()[1] >= self.caps.m_max_bytes:
            return "memory"
        return None


def _add_term(total, term):
    if total is None or term is None:
        return None
    return total + term


def _min_defined(values):
    present = [v for v in values if v is not None]
    return min(present) if present else None


# ---------------------------------------------------------------------------
# Branch-and-bound: capped fixed-risk endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BnbResult:
    status: str
    lo: Optional[Fraction]
    hi: Optional[Fraction]
    policy: Optional[PolicyNode]
    policy_key: Optional[tuple]
    replay: Optional[dict]
    stats: dict
    quotient_support_ratio: Fraction
    interval_status: str
    stats_full: dict = field(default_factory=dict)


def _rollout_cost(searcher: _Searcher, belief, remaining, memo=None) -> Optional[tuple]:
    """Cheapest found feasible policy under the cap; returns (expected_cost, node)."""
    if memo is None:
        memo = searcher.rollout_memo
    key = (_support(belief), remaining)
    if risk_brier(belief, searcher.problem.query) <= searcher.rho:
        return (Fraction(0), PolicyNode(None, ()))
    if key in memo:
        return memo[key]
    split = _splitting_actions(searcher.problem, belief, remaining)
    if not split:
        memo[support] = None
        return None
    ranked = sorted(
        split, key=lambda a: (v1(belief, a, searcher.problem.query), a.action_id)
    )
    result = None
    for action in ranked:
        children = outcome_cells(belief, action)
        total = action.cost
        nodes = []
        good = True
        for cell, mass in children:
            sub = _rollout_cost(
                searcher, posterior(belief, action, cell), remaining - action.cost, memo
            )
            if sub is None:
                good = False
                break
            total += mass * sub[0]
            nodes.append((cell, sub[1]))
        if good:
            result = (total, PolicyNode(action.action_id, tuple(nodes)))
            break
    memo[key] = result
    return result


def _solve_cost(searcher: _Searcher, belief, remaining: Fraction, depth: int):
    problem = searcher.problem
    searcher.stats.states_expanded += 1
    searcher.stats.max_depth = max(searcher.stats.max_depth, depth)
    searcher.stats.peak_frontier = max(searcher.stats.peak_frontier, depth)
    if risk_brier(belief, problem.query) <= searcher.rho:
        return (Fraction(0), Fraction(0), CERTIFIED)
    key = (_quotient_key(problem, belief, remaining), remaining)
    hit = searcher.memo.get(key)
    if hit is not None:
        searcher.stats.states_memo_hits += 1
        return hit
    split = _splitting_actions(problem, belief, remaining)
    if not split:
        searcher.memo[key] = (None, None, INFEASIBLE)
        return searcher.memo[key]
    bounds = lower_bounds(problem, belief, remaining, searcher.k_lb, searcher.rho)
    if bounds["infeasible"]:
        searcher.memo[key] = (None, None, INFEASIBLE)
        return searcher.memo[key]
    lo_state = bounds["lb_combined"]
    cap = searcher._cap_hit()
    if cap is not None:
        searcher.stats.capped = True
        searcher.stats.cap_reason = cap
        return (lo_state, None, CAPPED)
    rollout = _rollout_cost(searcher, belief, remaining)
    ub_state = None if rollout is None else rollout[0]
    if ub_state is not None:
        if lo_state > ub_state:
            raise ContractViolation(
                f"admissible bound {lo_state} exceeds a feasible policy value {ub_state}"
            )
        if lo_state == ub_state:
            searcher.memo[key] = (ub_state, ub_state, CERTIFIED)
            return searcher.memo[key]
    best_lo = None
    best_hi = ub_state
    for action in split:
        cap = searcher._cap_hit()
        if cap is not None:
            searcher.stats.capped = True
            searcher.stats.cap_reason = cap
            return (best_lo if best_lo is not None else lo_state, None, CAPPED)
        searcher.stats.branches_generated += 1
        children = outcome_cells(belief, action)
        act_lo = action.cost
        act_hi = action.cost
        dead = False
        for cell, mass in children:
            child = posterior(belief, action, cell)
            sub = _solve_cost(searcher, child, remaining - action.cost, depth + 1)
            if sub[0] is None:
                dead = True
                break
            act_lo += mass * sub[0]
            if act_hi is not None:
                if sub[1] is None:
                    act_hi = None
                else:
                    act_hi += mass * sub[1]
        if dead:
            continue
        if best_lo is None or act_lo < best_lo:
            best_lo = act_lo
        if act_hi is not None and (best_hi is None or act_hi < best_hi):
            best_hi = act_hi
    if best_lo is None:
        searcher.memo[key] = (None, None, INFEASIBLE)
        return searcher.memo[key]
    if best_hi is not None and best_lo == best_hi:
        searcher.memo[key] = (best_hi, best_hi, CERTIFIED)
    else:
        searcher.memo[key] = (best_lo, best_hi, CAPPED)
    return searcher.memo[key]


def _cost_value(problem: Problem, belief, remaining, rho, memo) -> Optional[Fraction]:
    support = _support(belief)
    key = (support, remaining)
    if key in memo:
        return memo[key]
    if risk_brier(belief, problem.query) <= rho:
        memo[key] = Fraction(0)
        return memo[key]
    split = _splitting_actions(problem, belief, remaining)
    best = None
    for action in split:
        total = action.cost
        dead = False
        for cell, mass in outcome_cells(belief, action):
            sub = _cost_value(
                problem, posterior(belief, action, cell), remaining - action.cost, rho, memo
            )
            if sub is None:
                dead = True
                break
            total += mass * sub
        if dead:
            continue
        if best is None or total < best:
            best = total
    memo[key] = best
    return best


def lex_refine_cost(problem: Problem, rho, cap, v_star: Fraction) -> PolicyNode:
    """Phase B: lexicographically smallest optimal policy on the un-quotiented problem."""
    rho = exact_fraction(rho, "rho")
    cap = exact_fraction(cap, "cap")
    memo = {}

    def build(belief, remaining):
        if risk_brier(belief, problem.query) <= rho:
            return PolicyNode(None, ())
        state_value = _cost_value(problem, belief, remaining, rho, memo)
        if state_value is None:
            raise ContractViolation("lex refinement reached an infeasible state")
        for action in sorted(problem.actions, key=lambda a: a.action_id):
            if action.cost > remaining:
                continue
            if len(outcome_cells(belief, action)) < 2:
                continue
            total = action.cost
            child_nodes = []
            dead = False
            for cell, mass in outcome_cells(belief, action):
                child = posterior(belief, action, cell)
                sub = _cost_value(problem, child, remaining - action.cost, rho, memo)
                if sub is None:
                    dead = True
                    break
                total += mass * sub
                child_nodes.append((cell, child))
            if dead:
                continue
            if total == state_value:
                children = tuple((cell, build(child, remaining - action.cost)) for cell, child in child_nodes)
                return PolicyNode(action.action_id, children)
        raise ContractViolation(
            f"lex refinement found no action reaching the certified value {state_value}"
        )

    node = build(initial_belief(problem), cap)
    replay = replay_policy(problem, node)
    if replay["expected_cost"] != v_star:
        raise ContractViolation(
            f"lex refinement replay {replay['expected_cost']} != certified {v_star}"
        )
    if replay["max_leaf_primary_brier"] > rho:
        raise ContractViolation("lex refinement policy has a leaf above rho")
    return node


def bnb_cost(
    problem: Problem,
    rho=RHO_PRIMARY,
    cap=B_CAP,
    caps: Optional[SearchCaps] = None,
    k_lb: int = K_LB,
) -> BnbResult:
    """BNB_SOLVE for the capped fixed-risk endpoint."""
    cap = exact_fraction(cap, "cap")
    caps = caps or SearchCaps()
    searcher = _Searcher(problem, rho, caps, k_lb)
    belief0 = initial_belief(problem)
    quotient = signature_quotient(problem, belief0, cap)
    try:
        if no_informative_action(problem):
            return BnbResult(
                STATUS_NO_INFORMATIVE, None, None, None, None, None, searcher.stats.as_dict(),
                quotient["ratio"], "NO_INFORMATIVE_ACTION", {},
            )
        lo, hi, status = _solve_cost(searcher, belief0, cap, 0)
    finally:
        searcher.close()
    stats = searcher.stats.as_dict()
    if lo is None:
        return BnbResult(TREE_INFEASIBLE, None, None, None, None, None, stats, quotient["ratio"],
                         TREE_INFEASIBLE, stats)
    policy = None
    policy_key_value = None
    replay = None
    if status == CERTIFIED and lo == hi:
        policy = lex_refine_cost(problem, rho, cap, lo)
        policy_key_value = policy_key(policy)
        replay = replay_policy(problem, policy)
        if replay["expected_cost"] != lo:
            raise ContractViolation(
                f"BNB certificate replay {replay['expected_cost']} != {lo}"
            )
        interval_status = CERTIFIED_OPTIMAL_CAPPED
    else:
        interval_status = CERTIFIED_INTERVAL
    return BnbResult(
        status,
        lo,
        hi,
        policy,
        policy_key_value,
        replay,
        stats,
        quotient["ratio"],
        interval_status,
        stats,
    )


# ---------------------------------------------------------------------------
# Branch-and-bound: fixed-budget endpoint (R_LB as the admissible risk bound)
# ---------------------------------------------------------------------------


def _rollout_budget(searcher: _Searcher, belief, remaining, memo=None):
    if memo is None:
        memo = searcher.rollout_memo
    key = (_support(belief), remaining)
    if key in memo:
        return memo[key]
    stop = risk_brier(belief, searcher.problem.query)
    split = _splitting_actions(searcher.problem, belief, remaining)
    if not split:
        memo[key] = (stop, PolicyNode(None, ()))
        return memo[key]
    ranked = sorted(
        split, key=lambda a: (v1(belief, a, searcher.problem.query), a.action_id)
    )
    best = (stop, PolicyNode(None, ()))
    for action in ranked[:1]:
        total = Fraction(0)
        nodes = []
        for cell, mass in outcome_cells(belief, action):
            sub = _rollout_budget(
                searcher, posterior(belief, action, cell), remaining - action.cost, memo
            )
            total += mass * sub[0]
            nodes.append((cell, sub[1]))
        if total < best[0]:
            best = (total, PolicyNode(action.action_id, tuple(nodes)))
    memo[key] = best
    return best


def _solve_budget(searcher: _Searcher, belief, remaining: Fraction, depth: int):
    problem = searcher.problem
    searcher.stats.states_expanded += 1
    searcher.stats.max_depth = max(searcher.stats.max_depth, depth)
    searcher.stats.peak_frontier = max(searcher.stats.peak_frontier, depth)
    stop = risk_brier(belief, problem.query)
    key = (_quotient_key(problem, belief, remaining), remaining, "BUDGET")
    hit = searcher.memo.get(key)
    if hit is not None:
        searcher.stats.states_memo_hits += 1
        return hit
    split = _splitting_actions(problem, belief, remaining)
    if not split:
        searcher.memo[key] = (stop, stop, CERTIFIED)
        return searcher.memo[key]
    affordable = tuple(a for a in problem.actions if a.cost <= remaining)
    lo = r_lb(belief, problem.query, affordable, remaining)
    rollout = _rollout_budget(searcher, belief, remaining)
    hi = min(rollout[0], stop)
    if lo >= hi:
        searcher.memo[key] = (hi, hi, CERTIFIED)
        return searcher.memo[key]
    best_lo = lo
    best_hi = hi
    for action in split:
        cap = searcher._cap_hit()
        if cap is not None:
            searcher.stats.capped = True
            searcher.stats.cap_reason = cap
            return (best_lo, None, CAPPED)
        searcher.stats.branches_generated += 1
        act_lo = Fraction(0)
        act_hi = Fraction(0)
        dead = False
        for cell, mass in outcome_cells(belief, action):
            sub = _solve_budget(
                searcher, posterior(belief, action, cell), remaining - action.cost, depth + 1
            )
            if sub[0] is None:
                dead = True
                break
            act_lo += mass * sub[0]
            if act_hi is None or sub[1] is None:
                act_hi = None
            else:
                act_hi += mass * sub[1]
        if dead:
            continue
        if act_lo < best_lo:
            best_lo = act_lo
        if act_hi is not None and act_hi < best_hi:
            best_hi = act_hi
    if best_hi is None:
        searcher.memo[key] = (best_lo, None, CAPPED)
    elif best_lo == best_hi:
        searcher.memo[key] = (best_hi, best_hi, CERTIFIED)
    else:
        searcher.memo[key] = (best_lo, best_hi, CAPPED)
    return searcher.memo[key]


def _budget_value(problem: Problem, belief, remaining, memo) -> Fraction:
    support = _support(belief)
    key = (support, remaining)
    if key in memo:
        return memo[key]
    stop = risk_brier(belief, problem.query)
    split = _splitting_actions(problem, belief, remaining)
    best = stop
    for action in split:
        total = Fraction(0)
        for cell, mass in outcome_cells(belief, action):
            total += mass * _budget_value(
                problem, posterior(belief, action, cell), remaining - action.cost, memo
            )
        if total < best:
            best = total
    memo[key] = best
    return best


def lex_refine_budget(problem: Problem, budget, v_star: Fraction) -> PolicyNode:
    budget = exact_fraction(budget, "budget")
    memo = {}

    def build(belief, remaining):
        state_value = _budget_value(problem, belief, remaining, memo)
        stop = risk_brier(belief, problem.query)
        if stop == state_value:
            # frozen tie-break: STOP contributes the empty tuple, which sorts
            # before any action id, so stopping wins an exact tie.
            return PolicyNode(None, ())
        for action in sorted(problem.actions, key=lambda a: a.action_id):
            if action.cost > remaining:
                continue
            if len(outcome_cells(belief, action)) < 2:
                continue
            total = Fraction(0)
            child_nodes = []
            for cell, mass in outcome_cells(belief, action):
                child = posterior(belief, action, cell)
                total += mass * _budget_value(problem, child, remaining - action.cost, memo)
                child_nodes.append((cell, child))
            if total == state_value:
                children = tuple((cell, build(child, remaining - action.cost)) for cell, child in child_nodes)
                return PolicyNode(action.action_id, children)
        if stop == state_value:
            return PolicyNode(None, ())
        raise ContractViolation("lex refinement (budget) found no optimal decision")

    node = build(initial_belief(problem), budget)
    replay = replay_policy(problem, node)
    if replay["expected_terminal_primary_brier"] != v_star:
        raise ContractViolation(
            f"budget lex refinement replay {replay['expected_terminal_primary_brier']} != {v_star}"
        )
    return node


def bnb_budget(problem: Problem, budget, caps: Optional[SearchCaps] = None) -> BnbResult:
    budget = exact_fraction(budget, "budget")
    caps = caps or SearchCaps()
    searcher = _Searcher(problem, RHO_PRIMARY, caps)
    belief0 = initial_belief(problem)
    quotient = signature_quotient(problem, belief0, budget)
    try:
        if no_informative_action(problem):
            return BnbResult(
                STATUS_NO_INFORMATIVE, None, None, None, None, None, searcher.stats.as_dict(),
                quotient["ratio"], "NO_INFORMATIVE_ACTION", {},
            )
        lo, hi, status = _solve_budget(searcher, belief0, budget, 0)
    finally:
        searcher.close()
    stats = searcher.stats.as_dict()
    if lo is None:
        return BnbResult(TREE_INFEASIBLE, None, None, None, None, None, stats, quotient["ratio"],
                         TREE_INFEASIBLE, stats)
    policy = None
    policy_key_value = None
    replay = None
    if status == CERTIFIED and lo == hi:
        policy = lex_refine_budget(problem, budget, lo)
        policy_key_value = policy_key(policy)
        replay = replay_policy(problem, policy)
        if replay["expected_terminal_primary_brier"] != lo:
            raise ContractViolation("budget certificate replay mismatch")
        interval_status = CERTIFIED_OPTIMAL_CAPPED
    else:
        interval_status = CERTIFIED_INTERVAL
    return BnbResult(
        status, lo, hi, policy, policy_key_value, replay, stats, quotient["ratio"],
        interval_status, stats,
    )


# ---------------------------------------------------------------------------
# Exact references: capped fixed-risk wrapper and full tree with state cap
# ---------------------------------------------------------------------------


def capped_cost_fixed_risk(problem: Problem, rho, cap) -> CostResult:
    """COST(rho) restricted to per-path cost <= cap (prereg extension wrapper)."""
    rho = exact_fraction(rho, "rho")
    cap = exact_fraction(cap, "cap")
    if no_informative_action(problem):
        return CostResult(STATUS_NO_INFORMATIVE, rho)
    memo = {}
    result = _capped_cost(problem, initial_belief(problem), rho, cap, memo)
    if result is None:
        return CostResult(STATUS_UNRESOLVED, rho)
    expected, worst, key, node = result
    return CostResult(
        STATUS_OK, rho, expected_cost=expected, worst_cost=worst, policy=node, key=key,
        replay=replay_policy(problem, node),
    )


def _capped_cost(problem, belief, rho, remaining, memo):
    key = (_support(belief), remaining)
    if key in memo:
        return memo[key]
    if risk_brier(belief, problem.query) <= rho:
        result = (Fraction(0), Fraction(0), (), PolicyNode(None, ()))
        memo[key] = result
        return result
    if remaining < COST_PER_REVEAL:
        memo[key] = None
        return None
    best = None
    for action in sorted(problem.actions, key=lambda a: a.action_id):
        if action.cost > remaining:
            continue
        if len(outcome_cells(belief, action)) < 2:
            continue
        expected = action.cost
        worst = action.cost
        children = []
        child_keys = []
        feasible = True
        for cell, mass in outcome_cells(belief, action):
            sub = _capped_cost(
                problem, posterior(belief, action, cell), rho, remaining - action.cost, memo
            )
            if sub is None:
                feasible = False
                break
            expected += mass * sub[0]
            worst = max(worst, action.cost + sub[1])
            children.append((cell, sub[3]))
            child_keys.append(sub[2])
        if not feasible:
            continue
        candidate_key = (action.action_id,) + tuple(child_keys)
        if best is None or (expected, candidate_key) < (best[0], best[2]):
            best = (expected, worst, candidate_key, PolicyNode(action.action_id, tuple(children)))
    memo[key] = best
    return best


@dataclass(frozen=True)
class FullTreeResult:
    status: str
    value: Optional[Fraction]
    worst: Optional[Fraction]
    policy: Optional[PolicyNode]
    policy_key: Optional[tuple]
    replay: Optional[dict]
    states: int
    peak_depth: int
    state_cap: int


class _StateCapExceeded(Exception):
    pass


def full_tree_cost_capped(
    problem: Problem, rho=RHO_PRIMARY, cap=B_CAP, state_cap: int = N_FULL_STATES
) -> FullTreeResult:
    """Exhaustive exact capped fixed-risk reference (no bounds, no pruning)."""
    rho = exact_fraction(rho, "rho")
    cap = exact_fraction(cap, "cap")
    if no_informative_action(problem):
        return FullTreeResult(STATUS_NO_INFORMATIVE, None, None, None, None, None, 0, 0, state_cap)
    memo = {}
    counts = {"states": 0, "depth": 0}

    def solve(belief, remaining, depth):
        counts["states"] += 1
        counts["depth"] = max(counts["depth"], depth)
        if counts["states"] > state_cap:
            raise _StateCapExceeded()
        key = (_support(belief), remaining)
        if key in memo:
            return memo[key]
        if risk_brier(belief, problem.query) <= rho:
            result = (Fraction(0), Fraction(0), (), PolicyNode(None, ()))
            memo[key] = result
            return result
        best = None
        for action in sorted(problem.actions, key=lambda a: a.action_id):
            if action.cost > remaining:
                continue
            if len(outcome_cells(belief, action)) < 2:
                continue
            expected = action.cost
            worst = action.cost
            children = []
            child_keys = []
            feasible = True
            for cell, mass in outcome_cells(belief, action):
                sub = solve(posterior(belief, action, cell), remaining - action.cost, depth + 1)
                if sub is None:
                    feasible = False
                    break
                expected += mass * sub[0]
                worst = max(worst, action.cost + sub[1])
                children.append((cell, sub[3]))
                child_keys.append(sub[2])
            if not feasible:
                continue
            candidate_key = (action.action_id,) + tuple(child_keys)
            if best is None or (expected, candidate_key) < (best[0], best[2]):
                best = (expected, worst, candidate_key, PolicyNode(action.action_id, tuple(children)))
        memo[key] = best
        return best

    try:
        result = solve(initial_belief(problem), cap, 0)
    except _StateCapExceeded:
        return FullTreeResult(
            NOT_COMPUTED, None, None, None, None, None, counts["states"], counts["depth"], state_cap
        )
    if result is None:
        return FullTreeResult(
            TREE_INFEASIBLE, None, None, None, None, None, counts["states"], counts["depth"], state_cap
        )
    expected, worst, key, node = result
    replay = replay_policy(problem, node)
    if replay["expected_cost"] != expected:
        raise ContractViolation("full-tree certificate replay mismatch")
    if replay["max_leaf_primary_brier"] > rho:
        raise ContractViolation("full-tree reference has a leaf above rho")
    return FullTreeResult(
        STATUS_OK, expected, worst, node, key, replay, counts["states"], counts["depth"], state_cap
    )


def full_tree_budget(problem: Problem, budget) -> OptResult:
    """Exact fixed-budget reference, delegated to the frozen M5A implementation."""
    return opt_fixed_budget(problem, exact_fraction(budget, "budget"))


# ---------------------------------------------------------------------------
# Gap arithmetic
# ---------------------------------------------------------------------------


def cert_gaps(lo, hi, v_full) -> dict:
    """CERT_GAP: exact certified interval and exact gap to the full-tree reference."""
    out = {
        "lo": None if lo is None else str(lo),
        "hi": None if hi is None else str(hi),
        "absolute_certified_gap": None,
        "relative_certified_gap": None,
        "relative_certified_gap_status": None,
        "absolute_gap_vs_full_tree": None,
        "relative_gap_vs_full_tree": None,
        "relative_gap_vs_full_tree_status": None,
        "full_tree_status": None,
        "defect": False,
    }
    if lo is not None and hi is not None:
        gap = hi - lo
        out["absolute_certified_gap"] = str(gap)
        if lo > 0:
            out["relative_certified_gap"] = str(gap / lo)
        else:
            out["relative_certified_gap_status"] = RELATIVE_GAP_UNDEFINED_ZERO_LOWER
    if v_full is None:
        out["full_tree_status"] = NOT_COMPUTED
        return out
    out["full_tree_status"] = STATUS_OK
    v_bnb = lo if lo is not None else hi
    if v_bnb is None:
        return out
    diff = v_bnb - v_full
    out["absolute_gap_vs_full_tree"] = str(diff)
    if v_full > 0:
        out["relative_gap_vs_full_tree"] = str(diff / v_full)
    else:
        out["relative_gap_vs_full_tree_status"] = RELATIVE_GAP_UNDEFINED_ZERO_REFERENCE
    if v_bnb < v_full:
        out["defect"] = True
        out["defect_kind"] = IMPLEMENTATION_DEFECT
    elif v_bnb != v_full:
        out["defect"] = True
        out["defect_kind"] = "CERTIFICATE_INCONSISTENT_WITH_FULL_TREE"
    return out


# ---------------------------------------------------------------------------
# Split assignment and manifest predicates
# ---------------------------------------------------------------------------


def assign_split(cells: Sequence[Mapping]) -> dict:
    """Frozen assignment rule applied to materialised cell eligibility."""
    eligible = [c for c in cells if c["eligible"]]
    if not eligible:
        _fail_closed("no eligible cells: the split would be empty")
    mechanisms = sorted({c["mechanism"] for c in eligible})
    holdout = mechanisms[0]
    decisions = {}
    for cell in cells:
        cell_key = (cell["mechanism"], cell["world_family"])
        if not cell["eligible"]:
            decisions[cell_key] = {
                "split": None,
                "reason": "ineligible: no b0-informative change unit; excluded from comparison",
            }
            continue
        if cell["mechanism"] == holdout:
            decisions[cell_key] = {
                "split": SPLIT_CONFIRMATORY,
                "reason": "mechanism holdout: first eligible mechanism in codepoint order",
            }
            continue
        families = sorted(
            c["world_family"] for c in eligible if c["mechanism"] == cell["mechanism"]
        )
        if cell["world_family"] == families[0]:
            decisions[cell_key] = {
                "split": SPLIT_CONFIRMATORY,
                "reason": "first eligible family in codepoint order within non-holdout mechanism",
            }
        else:
            decisions[cell_key] = {"split": SPLIT_DEVELOPMENT, "reason": "non-first family"}
    return {"holdout_mechanism": holdout, "eligible_mechanisms": mechanisms, "decisions": decisions}


def manifest_predicate_check(manifest: Mapping) -> dict:
    """MR-04 universal predicates over the materialised assignment manifest."""
    cells = manifest["cells"]
    units = manifest["units"]
    eligible_cells = [c for c in cells if c["eligible"]]
    holdout = manifest["holdout_mechanism"]
    checks = {}
    checks["holdout_has_no_development_cell"] = not any(
        c["mechanism"] == holdout and c["split"] == SPLIT_DEVELOPMENT for c in eligible_cells
    )
    checks["holdout_all_confirmatory"] = all(
        c["split"] == SPLIT_CONFIRMATORY for c in eligible_cells if c["mechanism"] == holdout
    )
    per_mechanism = {}
    for c in eligible_cells:
        per_mechanism.setdefault(c["mechanism"], []).append(c)
    counts = {}
    non_holdout_ok = True
    for mechanism, members in per_mechanism.items():
        if mechanism == holdout:
            continue
        families = sorted(m["world_family"] for m in members)
        got_confirmatory = [m for m in members if m["split"] == SPLIT_CONFIRMATORY]
        counts[mechanism] = {
            "eligible_cells": len(members),
            "confirmatory_cells": len(got_confirmatory),
            "development_cells": len(members) - len(got_confirmatory),
        }
        if len(got_confirmatory) != 1 or got_confirmatory[0]["world_family"] != families[0]:
            non_holdout_ok = False
    checks["non_holdout_one_confirmatory_per_mechanism"] = non_holdout_ok
    checks["ineligible_cells_are_registered"] = all(
        c.get("split") is None and bool(c.get("exclusion_reason"))
        for c in cells
        if not c["eligible"]
    )
    by_id = {u["unit_id"]: u for u in units}
    twins_ok = True
    twins_checked = 0
    for unit in units:
        twin = by_id.get(unit.get("twin_unit_id"))
        if twin is None:
            twins_ok = False
            continue
        twins_checked += 1
        if unit["split"] != twin["split"]:
            twins_ok = False
        if unit["mechanism"] != twin["mechanism"] or unit["world_family"] != twin["world_family"]:
            twins_ok = False
    checks["twins_share_split"] = twins_ok
    checks["twins_checked"] = twins_checked
    checks["holdout_has_zero_development_units"] = not any(
        u["mechanism"] == holdout and u["split"] == SPLIT_DEVELOPMENT for u in units
    )
    checks["confirmatory_units"] = sum(1 for u in units if u["split"] == SPLIT_CONFIRMATORY)
    checks["development_units"] = sum(1 for u in units if u["split"] == SPLIT_DEVELOPMENT)
    checks["confirmatory_change_units"] = sum(
        1 for u in units if u["split"] == SPLIT_CONFIRMATORY and u["role"] == ROLE_CHANGE
    )
    checks["development_change_units"] = sum(
        1 for u in units if u["split"] == SPLIT_DEVELOPMENT and u["role"] == ROLE_CHANGE
    )
    checks["confirmatory_control_units"] = sum(
        1 for u in units if u["split"] == SPLIT_CONFIRMATORY and u["role"] == ROLE_CONTROL
    )
    checks["development_control_units"] = sum(
        1 for u in units if u["split"] == SPLIT_DEVELOPMENT and u["role"] == ROLE_CONTROL
    )
    # control units stay in their cell's split lane (role is not a split level);
    # only ineligible CHANGE units must be registered out of the denominator.
    checks["ineligible_units_registered"] = all(
        u.get("split") is None and bool(u.get("exclusion_reason"))
        for u in units
        if u["role"] == ROLE_CHANGE and not u["eligible"]
    )
    return {
        "checks": checks,
        "per_mechanism": counts,
        "holdout_mechanism": holdout,
        "all_pass": bool(
            checks["holdout_has_no_development_cell"]
            and checks["holdout_all_confirmatory"]
            and checks["non_holdout_one_confirmatory_per_mechanism"]
            and checks["ineligible_cells_are_registered"]
            and checks["twins_share_split"]
            and checks["holdout_has_zero_development_units"]
            and checks["ineligible_units_registered"]
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation (development path only; refuses sealed/confirmatory records)
# ---------------------------------------------------------------------------


def _mean(values: Sequence[Fraction]) -> Optional[Fraction]:
    if not values:
        return None
    return sum(values, Fraction(0)) / len(values)


def _mean_optional(values: Sequence) -> tuple:
    present = [Fraction(v) for v in values if v is not None]
    return _mean(present), len(present)


def aggregate_cell(cell_record: Mapping, expected_split: str = SPLIT_DEVELOPMENT) -> dict:
    """Mean over a cell's eligible change units; refuses confirmatory records."""
    if cell_record["split"] != expected_split:
        _fail_closed(
            f"aggregate_cell refuses split {cell_record['split']!r} (expected {expected_split!r})"
        )
    denominator = [u for u in cell_record["units"] if u["role"] == ROLE_CHANGE and u["eligible"]]
    if not denominator:
        _fail_closed(f"empty denominator cell {cell_record['cell_id']}")
    out = {
        "cell_id": cell_record["cell_id"],
        "mechanism": cell_record["mechanism"],
        "world_family": cell_record["world_family"],
        "split": cell_record["split"],
        "units_in_denominator": len(denominator),
        "excluded_units": [
            {"unit_id": u["unit_id"], "role": u["role"], "eligible": u["eligible"]}
            for u in cell_record["units"]
            if u["role"] != ROLE_CHANGE or not u["eligible"]
        ],
        "strategies": {},
    }
    for strategy in CONTROL_IDS:
        rows = [u["readings"][strategy] for u in denominator]
        censored = [Fraction(r["censored_expected_cost"]) for r in rows]
        unresolved = [Fraction(r["unresolved_mass"]) for r in rows]
        resolved_costs = [
            Fraction(r["resolved_only_mean_cost"]) for r in rows if r["resolved_only_mean_cost"] is not None
        ]
        risk_at = [_mean_optional([r["risk_at_k"][k] for r in rows]) for k in range(B_CAP + 1)]
        bayes_at = [_mean_optional([r["bayes_at_k"][k] for r in rows]) for k in range(B_CAP + 1)]
        undecided = [_mean_optional([r["undecided_mass_at_k"][k] for r in rows]) for k in range(B_CAP)]
        out["strategies"][strategy] = {
            "censored_expected_total_cost": str(_mean(censored)),
            "unresolved_rate": str(_mean(unresolved)),
            "resolved_only_mean_cost": None if not resolved_costs else str(_mean(resolved_costs)),
            "resolved_only_units": len(resolved_costs),
            "risk_at_k": [None if v is None else str(v) for v, _n in risk_at],
            "risk_at_k_units": [n for _v, n in risk_at],
            "bayes_risk_at_k": [None if v is None else str(v) for v, _n in bayes_at],
            "undecided_rate_at_k": [None if v is None else str(v) for v, _n in undecided],
        }
    return out


__all__ = [
    "SCHEMA",
    "PREREG_SHA256",
    "M5A_MODULE_SHA256",
    "GRID_G",
    "COST_PER_REVEAL",
    "RHO_PRIMARY",
    "B_CAP",
    "BUDGET_GRID",
    "REPLICATES_STOCHASTIC",
    "K_LB",
    "K_SEP",
    "K_BELL",
    "N_EXPAND",
    "N_FULL_STATES",
    "T_MAX_SECONDS",
    "M_MAX_BYTES",
    "MECHANISMS",
    "FAMILIES",
    "INDICES",
    "ROLES",
    "ROLE_CHANGE",
    "ROLE_CONTROL",
    "UNIT_STATUS_CHANGE",
    "UNIT_STATUS_CONTROL",
    "UNIT_STATUS_NO_INFORMATIVE",
    "SPLIT_DEVELOPMENT",
    "SPLIT_CONFIRMATORY",
    "CONTROL_IDS",
    "SELECTOR_IDS",
    "SELECTORS",
    "CERTIFIED_OPTIMAL_CAPPED",
    "CERTIFIED_INTERVAL",
    "TREE_INFEASIBLE",
    "NOT_COMPUTED",
    "RELATIVE_GAP_UNDEFINED_ZERO_LOWER",
    "RELATIVE_GAP_UNDEFINED_ZERO_REFERENCE",
    "IMPLEMENTATION_DEFECT",
    "ContractViolation",
    "canonical_json",
    "sha256_text",
    "sha256_bytes",
    "signature_of",
    "disk_contact",
    "canonical_direction",
    "Grid",
    "build_grid",
    "public_query_points",
    "visible_centers",
    "FAMILY_SPECS",
    "generate_pool",
    "pool_fingerprint",
    "assert_dataset_sha",
    "Reveal",
    "UnitWorld",
    "build_world",
    "cert_non_translation",
    "cert_structural_visibility",
    "cert_query_decisive",
    "cert_public_identical",
    "cert_twin",
    "unit_certificates",
    "expand_tree",
    "sample_path",
    "readings_from_leaves",
    "seed_for",
    "run_strategy",
    "lower_bounds",
    "r_lb",
    "signature_quotient",
    "SearchCaps",
    "SearchStats",
    "BnbResult",
    "bnb_cost",
    "bnb_budget",
    "capped_cost_fixed_risk",
    "FullTreeResult",
    "full_tree_cost_capped",
    "full_tree_budget",
    "lex_refine_cost",
    "lex_refine_budget",
    "cert_gaps",
    "assign_split",
    "manifest_predicate_check",
    "aggregate_cell",
]
