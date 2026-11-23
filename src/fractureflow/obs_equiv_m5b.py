"""M5B active-measurement implementation core (stdlib only, no I/O).

Implements the frozen protocol
``results/obs_equiv_20261024/m5b_protocol_prereg_20261025/prereg.json``
(schema ``obs_equiv.m5b_protocol_prereg.v1``) on top of the shared M5A belief
machinery (``fractureflow.obs_equiv_m5a_reference``), which is imported and
never reimplemented.

Scope: software implementation only. No scientific reading, no real data, no
evaluator files read from inside this module. H1 remains an unverified target;
nothing computed here is evidence for or against it.

Frozen pieces implemented here (all cross-referenced to prereg.json):

* ``action_space``: public-only addressing (bbox of public disk centres plus
  query endpoints, G=4), half-open cells with the last row/column closed,
  per-axis degeneracy to a single row/column, and the frozen outcome
  signature ``Z_a(h) = sha256(canonical_json(revealed objects))[:16]`` (payload
  keys ``type``/``center``/``normal``/``radius``; empty reveal is ``EMPTY``).
  Hidden objects whose centre falls outside the bbox are never addressable and
  are counted (OUTSIDE) rather than dropped.
* ``controls``: the eight zero-training selectors with their frozen
  pseudocode, each receiving a capability-restricted context (PUBLIC_ONLY sees
  no belief and no partitions; QUERY_AGNOSTIC_IG sees belief and partitions but
  no query labels; BALD/EC2/GREEDY additionally see query labels).
* ``endpoints``: FIXED_RISK (rho=0, cap=4) with censored cost (unresolved
  paths cost the cap and stay in the denominator) and FIXED_BUDGET
  (B in 1..4) terminal risk / undecided rate, read off the same trajectory.
* ``extension_wrappers``: ``capped_cost_fixed_risk``, the only extension
  permitted by the prereg (prefix pruning at sum(cost) <= cap).
* ``stopping_rules`` / ``no_informative_action_and_denominator_rules``:
  RISK_REACHED, BUDGET_EXHAUSTED, NO_AFFORDABLE_ACTION,
  NO_INFORMATIVE_AT_STOP, group-level NO_INFORMATIVE_ACTION exclusion.
* ``split_assignment_rule``: deterministic mechanism/family assignment with
  the codepoint-order holdout, plus the MR-04 predicate check and eligibility.
* ``seeds_and_replicates``: seed derivation from
  ``sha256('{dataset_sha256}|{pair_id}|{strategy_id}|rep{r}')[:8]`` and the
  32-replicate RANDOM / single-run deterministic rule.

Interpretation gaps registered by the implementation unit (not protocol
changes): cell_center is the exact midpoint of the closed cell rectangle; the
frozen squared-distance ordering is used for the geometric selectors (order
equivalent to the Euclidean distance); GREEDY_RISK_ONE_STEP is applied at the
current belief through ``greedy_pick`` on a projected problem (the M5A
function is defined at the initial belief only); trajectories are evaluated as
conditional policy trees for deterministic selectors (exact) and as
outcome-sampled paths for RANDOM (32 seeded replicates), both estimating the
same path-level functional.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from fractureflow.obs_equiv_m5a_reference import (
    NO_INFORMATIVE_ACTION,
    STATUS_NO_INFORMATIVE,
    STATUS_OK,
    STATUS_UNRESOLVED,
    Action,
    CostResult,
    PolicyNode,
    Problem,
    exact_fraction,
    greedy_pick,
    initial_belief,
    no_informative_action,
    opt_fixed_budget,
    outcome_cells,
    posterior,
    query_probability,
    replay_policy,
    risk_bayes_01,
    risk_brier,
    risk_columns,
    v1,
)

SCHEMA = "obs_equiv.m5b_impl.core.v1"
PREREG_SHA256 = "e31e4c232c4795fe1f1095ef721eecd8a65a356556312e3a53f55f224a395735"
DATASET_SHA256 = "fbc63caec9468a4df552d735082196ac47d47cd6a097b6c9344af32478769d18"

GRID_G = 4
COST_PER_REVEAL = Fraction(1)
RHO_PRIMARY = Fraction(0)
B_CAP = 4
BUDGET_GRID = (1, 2, 3, 4)
REPLICATES_STOCHASTIC = 32
REPLICATES_DETERMINISTIC = 1
MAX_TREE_NODES = 200_000

STOP = "STOP"
RISK_REACHED = "RISK_REACHED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
NO_AFFORDABLE_ACTION = "NO_AFFORDABLE_ACTION"
NO_INFORMATIVE_AT_STOP = "NO_INFORMATIVE_AT_STOP"

ROLE_CHANGE = "CHANGE_GROUP"
ROLE_CONTROL = "CONTROL_GROUP"
ROLE_NO_INFORMATIVE = NO_INFORMATIVE_ACTION

SPLIT_DEVELOPMENT = "development"
SPLIT_CONFIRMATORY = "confirmatory"

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


class ContractViolation(RuntimeError):
    """Raised for implementation-level inconsistencies (not protocol results)."""


def _fail_closed(reason: str) -> None:
    raise SystemExit(f"M5B-FAIL-CLOSED: {reason}")


# ---------------------------------------------------------------------------
# Canonical JSON, signatures, exact geometry helpers
# ---------------------------------------------------------------------------


def canonical_json(document) -> str:
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def signature_of(objects: Sequence[Mapping]) -> str:
    """Frozen outcome signature: sha256(canonical_json(payload list))[:16]."""
    if not objects:
        return "EMPTY"
    payloads = sorted((canonical_json(o) for o in objects))
    return sha256_text("[" + ",".join(payloads) + "]")[:16]


def _sq(value: Fraction) -> Fraction:
    return value * value


def point_segment_dist2d_sq(p, a, b) -> Fraction:
    """Exact squared point-to-segment distance (monotone in Euclidean distance)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return _sq(px - ax) + _sq(py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    if t < 0:
        t = Fraction(0)
    elif t > 1:
        t = Fraction(1)
    cx = ax + t * dx
    cy = ay + t * dy
    return _sq(px - cx) + _sq(py - cy)


def point_dist2d_sq(p, q) -> Fraction:
    return _sq(p[0] - q[0]) + _sq(p[1] - q[1])


def _xy_point(center: Sequence) -> tuple:
    return exact_fraction(center[0], "x"), exact_fraction(center[1], "y")


# ---------------------------------------------------------------------------
# Public-only grid
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
            if j < 0:
                j = 0
            elif j >= self.nx:
                j = self.nx - 1
        if self.ny == 1:
            i = 0
        else:
            y0, y1 = self.y_edges[0], self.y_edges[-1]
            i = int((y - y0) * self.ny / (y1 - y0))
            if i < 0:
                i = 0
            elif i >= self.ny:
                i = self.ny - 1
        return i, j

    def cell_id_at(self, x: Fraction, y: Fraction) -> str:
        i, j = self.index_of(x, y)
        return f"R{i}C{j}"

    def cell_bounds(self, cell_id: str) -> tuple:
        i, j = self.parse(cell_id)
        return (
            self.x_edges[j],
            self.x_edges[j + 1],
            self.y_edges[i],
            self.y_edges[i + 1],
        )

    def cell_center(self, cell_id: str) -> tuple:
        x0, x1, y0, y1 = self.cell_bounds(cell_id)
        return (x0 + x1) / 2, (y0 + y1) / 2


def build_grid(public_observation: Mapping) -> Grid:
    """PUBLIC_ONLY addressing: bbox of visible disk centres plus query endpoints."""
    observation = public_observation
    points = []
    for obj in observation.get("observed_objects", ()):
        points.append(_xy_point(obj["center"]))
    query = observation.get("query", {})
    params = query.get("params", {})
    if "A" in params and "B" in params:
        points.append(_xy_point(params["A"]))
        points.append(_xy_point(params["B"]))
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
    return _xy_point(params["A"]), _xy_point(params["B"])


def visible_centers(public_observation: Mapping) -> tuple:
    return tuple(_xy_point(o["center"]) for o in public_observation.get("observed_objects", ()))


# ---------------------------------------------------------------------------
# Evaluator-side world construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reveal:
    cell_id: str
    signature: str
    objects: tuple


@dataclass(frozen=True)
class World:
    pair_id: str
    mechanism: str
    world_family: str
    public_observation: Mapping
    grid: Grid
    problem: Problem
    signatures: Mapping[str, Mapping[str, str]]
    payloads: Mapping[str, Mapping[str, tuple]]
    hidden_counts: tuple
    outside_count: int
    informative_cells: tuple
    role: str

    @property
    def eligible(self) -> bool:
        return bool(self.informative_cells)


def _payload(obj: Mapping) -> dict:
    return {k: obj[k] for k in ("type", "center", "normal", "radius")}


def build_world(public_record: Mapping, evaluator_record: Mapping) -> World:
    """Evaluator side: build H / b0 / Q and the frozen action-outcome contract.

    Selectors never call this function and never receive its inputs; they only
    see the constructed ``Problem`` through capability-restricted contexts.
    """
    if public_record["pair_id"] != evaluator_record["pair_id"]:
        raise ContractViolation("public/evaluator pair_id mismatch")
    scenes = evaluator_record["scenes"]
    if not scenes:
        _fail_closed("evaluator scene set is empty")
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
        raise ContractViolation(
            f"{public_record['pair_id']}: visible geometry differs across hypotheses"
        )
    prior = evaluator_record["prior_weights"]
    labels = evaluator_record["query_labels"]
    if len(prior) != len(hypotheses) or len(labels) != len(hypotheses):
        raise ContractViolation("prior/query length does not match |H|")
    query = {h: int(bool(labels[i])) for i, h in enumerate(hypotheses)}
    actions = []
    signatures = {}
    payloads = {}
    hidden_per_scene = tuple(
        sum(1 for obj in scene["objects"] if obj["id"] not in visible) for scene in scenes
    )
    outside_count = 0
    for cell_id in grid.cell_ids():
        per_h = {}
        per_h_payload = {}
        for h, scene in zip(hypotheses, scenes):
            objects = []
            for obj in scene["objects"]:
                if obj["id"] in visible:
                    continue
                x, y = _xy_point(obj["center"])
                if x < grid.x_edges[0] or x > grid.x_edges[-1]:
                    continue
                if y < grid.y_edges[0] or y > grid.y_edges[-1]:
                    continue
                if grid.cell_id_at(x, y) == cell_id:
                    objects.append(_payload(obj))
            per_h[h] = signature_of(objects)
            per_h_payload[h] = tuple(sorted(objects, key=canonical_json))
        groups = {}
        for h in hypotheses:
            groups.setdefault(per_h[h], []).append(h)
        partition = [sorted(members) for members in groups.values()]
        actions.append(Action.of(cell_id, COST_PER_REVEAL, partition))
        signatures[cell_id] = per_h
        payloads[cell_id] = {
            sig: next(per_h_payload[h] for h in hypotheses if per_h[h] == sig)
            for sig in groups
        }
    for scene in scenes:
        for obj in scene["objects"]:
            if obj["id"] in visible:
                continue
            x, y = _xy_point(obj["center"])
            if (
                x < grid.x_edges[0]
                or x > grid.x_edges[-1]
                or y < grid.y_edges[0]
                or y > grid.y_edges[-1]
            ):
                outside_count += 1
    problem = Problem.build(hypotheses, prior, query, actions)
    informative = tuple(
        cell_id
        for cell_id in grid.cell_ids()
        if len({signatures[cell_id][h] for h in hypotheses}) >= 2
    )
    belief0 = initial_belief(problem)
    risk0 = risk_brier(belief0, query)
    if all(query[h] == 0 for h in hypotheses) or all(query[h] == 1 for h in hypotheses):
        role = ROLE_CONTROL
        if risk0 != 0:
            raise ContractViolation("control group has non-zero initial risk")
    elif no_informative_action(problem):
        role = ROLE_NO_INFORMATIVE
    else:
        role = ROLE_CHANGE
    return World(
        pair_id=public_record["pair_id"],
        mechanism=evaluator_record["mechanism"],
        world_family=evaluator_record["world_family"],
        public_observation=observation,
        grid=grid,
        problem=problem,
        signatures=signatures,
        payloads=payloads,
        hidden_counts=hidden_per_scene,
        outside_count=outside_count,
        informative_cells=informative,
        role=role,
    )


# ---------------------------------------------------------------------------
# Capability-restricted selector contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicContext:
    """PUBLIC_ONLY: payload, revealed history and affordable ids. No belief, no Q."""

    grid: Grid
    query_a: tuple
    query_b: tuple
    visible_centers: tuple
    revealed: tuple
    affordable: tuple
    remaining_budget: int


@dataclass(frozen=True)
class InfoContext:
    """SHARED_MODEL without query labels (QUERY_AGNOSTIC_IG)."""

    public: PublicContext
    belief: Mapping
    actions: Mapping

    @property
    def affordable(self) -> tuple:
        return self.public.affordable


@dataclass(frozen=True)
class QueryModelContext:
    """SHARED_MODEL with query labels (BALD / EC2 / GREEDY)."""

    public: PublicContext
    belief: Mapping
    actions: Mapping
    query: Mapping

    @property
    def affordable(self) -> tuple:
        return self.public.affordable


def _public_context(world: World, revealed: tuple, affordable: tuple, remaining: int) -> PublicContext:
    a, b = public_query_points(world.public_observation)
    return PublicContext(
        grid=world.grid,
        query_a=a,
        query_b=b,
        visible_centers=visible_centers(world.public_observation),
        revealed=revealed,
        affordable=tuple(sorted(affordable)),
        remaining_budget=remaining,
    )


def _actions_by_id(world: World) -> dict:
    return {a.action_id: a for a in world.problem.actions}


# ---------------------------------------------------------------------------
# The eight frozen selectors
# ---------------------------------------------------------------------------


def select_random(ctx: PublicContext, rng: random.Random):
    """if not affordable: return STOP; return rng.choice(sorted(affordable))"""
    if not ctx.affordable:
        return STOP
    return rng.choice(list(ctx.affordable))


def select_geometric_frozen(ctx: PublicContext, rng: random.Random):
    """min(affordable, key=(dist_to_segment_AB(cell_center), action_id))"""
    if not ctx.affordable:
        return STOP
    a, b = ctx.query_a, ctx.query_b
    return min(
        ctx.affordable,
        key=lambda cell: (point_segment_dist2d_sq(ctx.grid.cell_center(cell), a, b), cell),
    )


def select_space_fill(ctx: PublicContext, rng: random.Random):
    """maximin over visible centres + revealed cell centres; ties by action id."""
    if not ctx.affordable:
        return STOP
    evidence = list(ctx.visible_centers)
    evidence.extend(ctx.grid.cell_center(r.cell_id) for r in ctx.revealed)
    if not evidence:
        return min(ctx.affordable)
    def gap(cell):
        centre = ctx.grid.cell_center(cell)
        return min(point_dist2d_sq(centre, e) for e in evidence)
    return min(ctx.affordable, key=lambda cell: (-gap(cell), cell))


def _belief_entropy(belief: Mapping) -> float:
    total = 0.0
    for value in belief.values():
        p = float(value)
        if p > 0.0:
            total -= p * math.log(p)
    return total


def select_query_agnostic_ig(ctx: InfoContext, rng: random.Random):
    """argmin_a sum_c p(c|b,a) * H(posterior(b,a,c)); Q is not read."""
    if not ctx.affordable:
        return STOP
    scores = {}
    for cell in ctx.affordable:
        action = ctx.actions[cell]
        score = 0.0
        for outcome_cell, mass in outcome_cells(ctx.belief, action):
            score += float(mass) * _belief_entropy(posterior(ctx.belief, action, outcome_cell))
        scores[cell] = score
    return min(ctx.affordable, key=lambda cell: (scores[cell], cell))


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def select_query_bald_like(ctx: QueryModelContext, rng: random.Random):
    """argmin_a sum_c p(c|b,a) * H2(query_probability(posterior(b,a,c)))."""
    if not ctx.affordable:
        return STOP
    scores = {}
    for cell in ctx.affordable:
        action = ctx.actions[cell]
        score = 0.0
        for outcome_cell, mass in outcome_cells(ctx.belief, action):
            p = float(query_probability(posterior(ctx.belief, action, outcome_cell), ctx.query))
            score += float(mass) * _bernoulli_entropy(p)
        scores[cell] = score
    return min(ctx.affordable, key=lambda cell: (scores[cell], cell))


def open_pairs(belief: Mapping, query: Mapping) -> int:
    support = sorted(belief)
    count = 0
    for i, h in enumerate(support):
        for h2 in support[i + 1:]:
            if query[h] != query[h2]:
                count += 1
    return count


def select_ec2_like(ctx: QueryModelContext, rng: random.Random):
    """argmin_a sum_c p(c|b,a) * #open Q-disagreeing support pairs."""
    if not ctx.affordable:
        return STOP
    scores = {}
    for cell in ctx.affordable:
        action = ctx.actions[cell]
        score = Fraction(0)
        for outcome_cell, mass in outcome_cells(ctx.belief, action):
            score += mass * open_pairs(posterior(ctx.belief, action, outcome_cell), ctx.query)
        scores[cell] = score
    return min(ctx.affordable, key=lambda cell: (scores[cell], cell))


def greedy_pick_at(ctx: QueryModelContext, budget=None) -> tuple:
    """Apply the frozen ``greedy_pick`` at the current belief.

    ``greedy_pick`` is defined at the initial belief only; projecting the
    current belief support and the affordable actions into a fresh ``Problem``
    feeds the frozen implementation without reimplementing its loop.
    """
    support = sorted(ctx.belief)
    support_set = set(support)
    actions = []
    for cell in ctx.affordable:
        action = ctx.actions[cell]
        cells = [sorted(set(part) & support_set) for part in action.partition]
        cells = [part for part in cells if part]
        if cells:
            actions.append(Action.of(action.action_id, action.cost, cells))
    if not actions:
        return None, risk_brier(ctx.belief, ctx.query)
    projected = Problem.build(
        support,
        [ctx.belief[h] for h in support],
        {h: ctx.query[h] for h in support},
        actions,
    )
    return greedy_pick(projected, budget)


def select_greedy_risk_one_step(ctx: QueryModelContext, rng: random.Random):
    """argmin_a V1(a); tie by action id (frozen M5A greedy_pick)."""
    if not ctx.affordable:
        return STOP
    pick, _value = greedy_pick_at(ctx)
    if pick is None:
        return STOP
    return pick


SELECTORS = {
    "RANDOM": (select_random, "PUBLIC_ONLY", True),
    "GEOMETRIC_FROZEN": (select_geometric_frozen, "PUBLIC_ONLY", False),
    "SPACE_FILL": (select_space_fill, "PUBLIC_ONLY", False),
    "QUERY_AGNOSTIC_IG": (select_query_agnostic_ig, "SHARED_MODEL_NO_QUERY", False),
    "QUERY_BALD_LIKE": (select_query_bald_like, "SHARED_MODEL", False),
    "EC2_LIKE": (select_ec2_like, "SHARED_MODEL", False),
    "GREEDY_RISK_ONE_STEP": (select_greedy_risk_one_step, "SHARED_MODEL", False),
}


def _make_context(world: World, capability: str, belief, revealed, affordable, remaining):
    public = _public_context(world, revealed, affordable, remaining)
    actions = _actions_by_id(world)
    if capability == "PUBLIC_ONLY":
        return public
    if capability == "SHARED_MODEL_NO_QUERY":
        return InfoContext(public=public, belief=belief, actions=actions)
    return QueryModelContext(public=public, belief=belief, actions=actions, query=world.problem.query)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    path: tuple
    probability: Fraction
    cost: int
    risks: tuple
    resolved: bool
    stop_reason: str
    bayes: tuple = ()


def _affordable(world: World, revealed: tuple, spent: int, cap: int) -> tuple:
    remaining = cap - spent
    if remaining < COST_PER_REVEAL:
        return ()
    revealed_ids = {r.cell_id for r in revealed}
    return tuple(cell for cell in world.grid.cell_ids() if cell not in revealed_ids)


def _all_non_informative(belief, world: World, affordable: Sequence[str]) -> bool:
    current = risk_brier(belief, world.problem.query)
    actions = _actions_by_id(world)
    return all(v1(belief, actions[cell], world.problem.query) == current for cell in affordable)


def _select(world: World, strategy_id: str, belief, revealed, affordable, remaining, rng):
    selector, capability, _stochastic = SELECTORS[strategy_id]
    ctx = _make_context(world, capability, belief, revealed, affordable, remaining)
    return selector(ctx, rng)


def _reveal(world: World, cell_id: str, cell) -> Reveal:
    signature = world.signatures[cell_id][sorted(set(cell))[0]]
    return Reveal(cell_id, signature, world.payloads[cell_id][signature])


def expand_tree(world: World, strategy_id: str, rho=RHO_PRIMARY, cap=B_CAP) -> list:
    """Exact conditional-policy tree for a deterministic selector."""
    problem = world.problem
    belief0 = initial_belief(problem)
    leaves = []
    nodes = 0

    def walk(belief, revealed, spent, path, probability, risks, bayes):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_TREE_NODES:
            _fail_closed(f"tree node cap {MAX_TREE_NODES} exceeded for {world.pair_id}")
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
        action_id = _select(world, strategy_id, belief, revealed, affordable, cap - spent, None)
        if action_id == STOP or action_id is None:
            leaves.append(Leaf(path, probability, spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes)))
            return
        if action_id not in affordable:
            raise ContractViolation(f"{world.pair_id}: selector picked unaffordable {action_id!r}")
        action = _actions_by_id(world)[action_id]
        for cell, mass in outcome_cells(belief, action):
            child = posterior(belief, action, cell)
            child_risks = risks + [risk_brier(child, problem.query)]
            child_bayes = bayes + [risk_bayes_01(child, problem.query)]
            walk(
                child,
                revealed + (_reveal(world, action_id, cell),),
                spent + 1,
                path + (action_id,),
                probability * mass,
                child_risks,
                child_bayes,
            )

    walk(belief0, (), 0, (), Fraction(1), [], [])
    total = sum((leaf.probability for leaf in leaves), Fraction(0))
    if total != 1:
        raise ContractViolation(f"{world.pair_id}: tree probabilities sum to {total}, not 1")
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


def sample_path(world: World, strategy_id: str, rng: random.Random, rho=RHO_PRIMARY, cap=B_CAP) -> Leaf:
    """One realized trajectory (outcome sampled from the current belief)."""
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
        action_id = _select(world, strategy_id, belief, revealed, affordable, cap - spent, rng)
        if action_id == STOP or action_id is None:
            return Leaf(path, Fraction(1), spent, tuple(risks), False, NO_AFFORDABLE_ACTION, tuple(bayes))
        if action_id not in affordable:
            raise ContractViolation(f"{world.pair_id}: selector picked unaffordable {action_id!r}")
        action = _actions_by_id(world)[action_id]
        cell, _mass = _sample_outcome(belief, action, rng)
        belief = posterior(belief, action, cell)
        revealed = revealed + (_reveal(world, action_id, cell),)
        spent += 1
        path = path + (action_id,)
        risks.append(risk_brier(belief, problem.query))
        bayes.append(risk_bayes_01(belief, problem.query))


# ---------------------------------------------------------------------------
# Endpoint readings
# ---------------------------------------------------------------------------


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


def readings_from_leaves(world: World, leaves: Sequence[Leaf], cap: int = B_CAP) -> dict:
    if not leaves:
        _fail_closed(f"empty trajectory stream for {world.pair_id}")
    root_risk = risk_brier(initial_belief(world.problem), world.problem.query)
    root_bayes = risk_bayes_01(initial_belief(world.problem), world.problem.query)
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
        value = sum(
            (leaf.probability * leaf_risk_at_k(leaf, root_risk, k) for leaf in leaves), Fraction(0)
        ) / total
        risk_at.append(value)
        secondary = sum(
            (leaf.probability * leaf_bayes_at_k(leaf, root_bayes, k) for leaf in leaves),
            Fraction(0),
        ) / total
        bayes_at.append(secondary)
        if k >= 1:
            mass = sum(
                (
                    leaf.probability
                    for leaf in leaves
                    if leaf_risk_at_k(leaf, root_risk, k) > 0
                ),
                Fraction(0),
            ) / total
            undecided.append(mass)
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


def seed_for(dataset_sha: str, pair_id: str, strategy_id: str, rep: int) -> int:
    material = f"{dataset_sha}|{pair_id}|{strategy_id}|rep{rep}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def run_strategy(world: World, strategy_id: str, dataset_sha: str = DATASET_SHA256) -> dict:
    """Run one selector on one group; returns serialisable group readings."""
    if strategy_id == "FULL_TREE_OPT":
        raise ValueError("FULL_TREE_OPT is a reference, not a selector")
    _selector, _capability, stochastic = SELECTORS[strategy_id]
    leaves = []
    seeds = []
    if stochastic:
        for rep in range(REPLICATES_STOCHASTIC):
            seed = seed_for(dataset_sha, world.pair_id, strategy_id, rep)
            leaves.append(sample_path(world, strategy_id, random.Random(seed)))
            seeds.append(seed)
    else:
        leaves = expand_tree(world, strategy_id)
    if not leaves:
        _fail_closed(f"empty trajectory stream for {world.pair_id}/{strategy_id}")
    readings = readings_from_leaves(world, leaves)
    return {
        "strategy": strategy_id,
        "stochastic": stochastic,
        "replicates": len(leaves),
        "seeds": seeds,
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


def run_full_tree_reference(world: World, rho=RHO_PRIMARY, cap=B_CAP) -> dict:
    """FULL_TREE_OPT: capped COST(rho), OPT(B) for B in 1..4, replay certificate."""
    problem = world.problem
    capped = capped_cost_fixed_risk(problem, rho, cap)
    opts = {}
    for budget in BUDGET_GRID:
        result = opt_fixed_budget(problem, budget)
        opts[budget] = {
            "status": result.status,
            "value": None if result.value is None else str(result.value),
            "secondary_bayes_01": (
                None if result.secondary_value is None else str(result.secondary_value)
            ),
            "policy_sequence": (
                None if result.policy is None else result.policy.flat_sequence()
            ),
        }
    certificate = None
    policy_leaves = []
    if capped.status == STATUS_OK:
        replay = replay_policy(problem, capped.policy)
        if replay["expected_cost"] != capped.expected_cost:
            raise ContractViolation("capped COST replay does not reach the certificate")
        if replay["max_leaf_primary_brier"] > rho:
            raise ContractViolation("capped COST certificate has a leaf above rho")
        certificate = {
            "expected_cost": str(replay["expected_cost"]),
            "worst_cost": str(replay["worst_cost"]),
            "max_leaf_primary_brier": str(replay["max_leaf_primary_brier"]),
            "policy_sequence": capped.policy.flat_sequence(),
        }
        policy_leaves = [
            {
                "path": list(record["path"]),
                "probability": str(record["path_probability"]),
                "cost": str(record["cost"]),
                "primary_brier": str(record["primary_brier"]),
                "resolved": record["primary_brier"] <= rho,
            }
            for record in replay["leaves"]
        ]
    for budget, record in opts.items():
        if record["status"] == STATUS_OK:
            result = opt_fixed_budget(problem, budget)
            replay = replay_policy(problem, result.policy)
            if replay["expected_terminal_primary_brier"] != result.value:
                raise ContractViolation(f"OPT({budget}) replay does not reach the certificate")
    return {
        "reference": "FULL_TREE_OPT",
        "capped_cost_status": capped.status,
        "capped_cost_expected": (
            None if capped.expected_cost is None else str(capped.expected_cost)
        ),
        "capped_cost_worst": None if capped.worst_cost is None else str(capped.worst_cost),
        "capped_cost_not_computed": capped.status != STATUS_OK,
        "opt": opts,
        "certificate": certificate,
        "policy_leaves": policy_leaves,
    }


def full_tree_group_readings(world: World) -> dict:
    """Normalise the FULL_TREE_OPT reference into the strategy-reading schema."""
    root_risk = risk_brier(initial_belief(world.problem), world.problem.query)
    root_bayes = risk_bayes_01(initial_belief(world.problem), world.problem.query)
    reference = run_full_tree_reference(world)
    opt_values = [reference["opt"][budget]["value"] for budget in BUDGET_GRID]
    opt_secondary = [
        reference["opt"][budget]["secondary_bayes_01"] for budget in BUDGET_GRID
    ]
    risk_at_k = [str(root_risk)] + [value for value in opt_values]
    bayes_at_k = [str(root_bayes)] + [value for value in opt_secondary]
    undecided_at_k = [
        None if value is None else str(Fraction(1) if Fraction(value) > 0 else Fraction(0))
        for value in opt_values
    ]
    base = {
        "strategy": "FULL_TREE_OPT",
        "stochastic": False,
        "reference": True,
        "replicates": 1,
        "seeds": [],
        "root_risk": str(root_risk),
        "root_bayes": str(root_bayes),
        "risk_at_k": risk_at_k,
        "bayes_at_k": bayes_at_k,
        "undecided_mass_at_k": undecided_at_k,
        "capped_cost_status": reference["capped_cost_status"],
        "capped_cost_not_computed": reference["capped_cost_not_computed"],
        "opt": reference["opt"],
        "certificate": reference["certificate"],
    }
    if world.role == ROLE_CONTROL:
        base.update(
            {
                "resolution_mass": "1",
                "unresolved_mass": "0",
                "censored_expected_cost": "0",
                "resolved_only_mean_cost": "0",
                "risk_at_k": ["0"] * (B_CAP + 1),
                "bayes_at_k": ["0"] * (B_CAP + 1),
                "undecided_mass_at_k": ["0"] * B_CAP,
                "leaves": [],
            }
        )
        return base
    if reference["capped_cost_status"] == STATUS_OK:
        leaves = [
            {
                "path": leaf["path"],
                "probability": leaf["probability"],
                "cost": leaf["cost"],
                "risks": [leaf["primary_brier"]] * len(leaf["path"]),
                "bayes": [],
                "resolved": leaf["resolved"],
                "stop_reason": "REFERENCE_POLICY",
            }
            for leaf in reference["policy_leaves"]
        ]
        base.update(
            {
                "resolution_mass": "1",
                "unresolved_mass": "0",
                "censored_expected_cost": reference["capped_cost_expected"],
                "resolved_only_mean_cost": reference["capped_cost_expected"],
                "leaves": leaves,
            }
        )
        return base
    base.update(
        {
            "resolution_mass": "0",
            "unresolved_mass": "1",
            "censored_expected_cost": str(B_CAP),
            "resolved_only_mean_cost": None,
            "leaves": [],
        }
    )
    return base


def run_group(world: World, dataset_sha: str = DATASET_SHA256) -> dict:
    """All eight controls on one group (seven selectors + the tree reference)."""
    readings = {sid: run_strategy(world, sid, dataset_sha) for sid in SELECTOR_IDS}
    readings["FULL_TREE_OPT"] = full_tree_group_readings(world)
    return {
        "pair_id": world.pair_id,
        "role": world.role,
        "mechanism": world.mechanism,
        "world_family": world.world_family,
        "hypotheses": len(world.problem.hypotheses),
        "informative_cells": list(world.informative_cells),
        "outside_objects": world.outside_count,
        "grid_shape": [world.grid.nx, world.grid.ny],
        "readings": readings,
    }


# ---------------------------------------------------------------------------
# Extension wrapper: capped COST (the only extension the prereg allows)
# ---------------------------------------------------------------------------


def _belief_key(belief: Mapping) -> tuple:
    return tuple(sorted(belief.items()))


def _capped_cost_recursive(problem: Problem, belief, rho, remaining, memo):
    key = (_belief_key(belief), remaining)
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
            continue  # exactness-preserving dominance skip (M5A search note)
        expected = action.cost
        worst = action.cost
        children = []
        child_keys = []
        feasible = True
        for cell, mass in outcome_cells(belief, action):
            sub = _capped_cost_recursive(
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


def capped_cost_fixed_risk(problem: Problem, rho, cap) -> CostResult:
    """COST(rho) restricted to per-path cost <= cap (prereg extension wrapper)."""
    rho = exact_fraction(rho, "rho")
    cap = exact_fraction(cap, "cap")
    if no_informative_action(problem):
        return CostResult(STATUS_NO_INFORMATIVE, rho)
    result = _capped_cost_recursive(problem, initial_belief(problem), rho, cap, {})
    if result is None:
        return CostResult(STATUS_UNRESOLVED, rho)
    expected, worst, key, node = result
    return CostResult(
        STATUS_OK,
        rho,
        expected_cost=expected,
        worst_cost=worst,
        policy=node,
        key=key,
        replay=replay_policy(problem, node),
    )


# ---------------------------------------------------------------------------
# Split assignment (MR-04)
# ---------------------------------------------------------------------------


def assign_split(cells: Sequence[Mapping]) -> dict:
    """Apply the frozen assignment rule to materialised cell eligibility."""
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
                "reason": "ineligible: no b0-informative action; excluded from comparison",
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
    """MR-04: universal predicate check over a materialised assignment manifest."""
    cells = manifest["cells"]
    eligible = [c for c in cells if c["eligible"]]
    holdout = manifest["holdout_mechanism"]
    checks = {}
    checks["holdout_has_no_development_cell"] = not any(
        c["mechanism"] == holdout and c["split"] == SPLIT_DEVELOPMENT for c in eligible
    )
    checks["holdout_all_confirmatory"] = all(
        c["split"] == SPLIT_CONFIRMATORY for c in eligible if c["mechanism"] == holdout
    )
    per_mechanism = {}
    for c in eligible:
        per_mechanism.setdefault(c["mechanism"], []).append(c)
    counts = {}
    non_holdout_ok = True
    for mechanism, members in per_mechanism.items():
        if mechanism == holdout:
            continue
        families = sorted(m["world_family"] for m in members)
        expected_first = families[0]
        got_confirmatory = [m for m in members if m["split"] == SPLIT_CONFIRMATORY]
        counts[mechanism] = {
            "eligible": len(members),
            "confirmatory": len(got_confirmatory),
            "development": len(members) - len(got_confirmatory),
        }
        if len(got_confirmatory) != 1:
            non_holdout_ok = False
        elif got_confirmatory[0]["world_family"] != expected_first:
            non_holdout_ok = False
        if any(m["split"] not in (SPLIT_CONFIRMATORY, SPLIT_DEVELOPMENT) for m in members):
            non_holdout_ok = False
    checks["non_holdout_one_confirmatory_per_mechanism"] = non_holdout_ok
    checks["ineligible_cells_are_registered"] = all(
        c.get("split") is None and bool(c.get("exclusion_reason"))
        for c in cells
        if not c["eligible"]
    )
    checks["confirmatory_count"] = sum(
        1 for c in eligible if c["split"] == SPLIT_CONFIRMATORY
    )
    checks["development_count"] = sum(1 for c in eligible if c["split"] == SPLIT_DEVELOPMENT)
    return {
        "checks": checks,
        "per_mechanism": counts,
        "holdout_mechanism": holdout,
        "all_pass": bool(
            checks["holdout_has_no_development_cell"]
            and checks["holdout_all_confirmatory"]
            and checks["non_holdout_one_confirmatory_per_mechanism"]
            and checks["ineligible_cells_are_registered"]
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
    """Mean over a cell's denominator groups; refuses confirmatory records."""
    if cell_record["split"] != expected_split:
        _fail_closed(
            f"aggregate_cell refuses split {cell_record['split']!r} "
            f"(expected {expected_split!r}): reporting path has no confirmatory input"
        )
    denominator = [g for g in cell_record["groups"] if g["role"] == ROLE_CHANGE]
    if not denominator:
        _fail_closed(f"empty denominator cell {cell_record['cell_id']}")
    out = {
        "cell_id": cell_record["cell_id"],
        "mechanism": cell_record["mechanism"],
        "world_family": cell_record["world_family"],
        "split": cell_record["split"],
        "groups_in_denominator": len(denominator),
        "excluded_groups": [
            {"pair_id": g["pair_id"], "role": g["role"]}
            for g in cell_record["groups"]
            if g["role"] != ROLE_CHANGE
        ],
        "strategies": {},
    }
    for strategy in CONTROL_IDS:
        rows = [g["readings"][strategy] for g in denominator]
        censored = [Fraction(r["censored_expected_cost"]) for r in rows]
        unresolved = [Fraction(r["unresolved_mass"]) for r in rows]
        resolved_costs = [
            Fraction(r["resolved_only_mean_cost"])
            for r in rows
            if r["resolved_only_mean_cost"] is not None
        ]
        risk_at = [_mean_optional([r["risk_at_k"][k] for r in rows]) for k in range(B_CAP + 1)]
        bayes_at = [_mean_optional([r["bayes_at_k"][k] for r in rows]) for k in range(B_CAP + 1)]
        undecided = [_mean_optional([r["undecided_mass_at_k"][k] for r in rows]) for k in range(B_CAP)]
        first = {}
        selection = {}
        costs = {}
        sequences = {}
        for row in rows:
            for leaf in row["leaves"]:
                probability = Fraction(leaf["probability"])
                if leaf["path"]:
                    first[leaf["path"][0]] = first.get(leaf["path"][0], Fraction(0)) + probability
                for cell in leaf["path"]:
                    selection[cell] = selection.get(cell, Fraction(0)) + probability
                key = str(leaf["cost"]) if leaf["resolved"] else str(B_CAP)
                costs[key] = costs.get(key, Fraction(0)) + probability
                seq = "->".join(leaf["path"]) if leaf["path"] else "STOP"
                sequences[seq] = sequences.get(seq, Fraction(0)) + probability
        out["strategies"][strategy] = {
            "censored_expected_total_cost": str(_mean(censored)),
            "unresolved_rate": str(_mean(unresolved)),
            "resolved_only_mean_cost": (
                None if not resolved_costs else str(_mean(resolved_costs))
            ),
            "resolved_only_cells": len(resolved_costs),
            "risk_at_k": [None if v is None else str(v) for v in (m for m, _n in risk_at)],
            "risk_at_k_groups": [n for _m, n in risk_at],
            "bayes_risk_at_k": [None if v is None else str(v) for v in (m for m, _n in bayes_at)],
            "undecided_rate_at_k": [
                None if v is None else str(v) for v in (m for m, _n in undecided)
            ],
            "first_action_counts": {
                k: str(v) for k, v in sorted(first.items())
            },
            "action_selection_counts": {
                k: str(v) for k, v in sorted(selection.items())
            },
            "censored_cost_histogram": {k: str(v) for k, v in sorted(costs.items())},
            "sequence_counts": {k: str(v) for k, v in sorted(sequences.items())},
        }
    return out


__all__ = [
    "SCHEMA",
    "PREREG_SHA256",
    "DATASET_SHA256",
    "GRID_G",
    "COST_PER_REVEAL",
    "RHO_PRIMARY",
    "B_CAP",
    "BUDGET_GRID",
    "REPLICATES_STOCHASTIC",
    "REPLICATES_DETERMINISTIC",
    "CONTROL_IDS",
    "SELECTOR_IDS",
    "STOP",
    "RISK_REACHED",
    "BUDGET_EXHAUSTED",
    "NO_AFFORDABLE_ACTION",
    "NO_INFORMATIVE_AT_STOP",
    "ROLE_CHANGE",
    "ROLE_CONTROL",
    "ROLE_NO_INFORMATIVE",
    "SPLIT_DEVELOPMENT",
    "SPLIT_CONFIRMATORY",
    "ContractViolation",
    "canonical_json",
    "sha256_text",
    "signature_of",
    "point_segment_dist2d_sq",
    "point_dist2d_sq",
    "Grid",
    "build_grid",
    "public_query_points",
    "visible_centers",
    "Reveal",
    "World",
    "build_world",
    "PublicContext",
    "InfoContext",
    "QueryModelContext",
    "select_random",
    "select_geometric_frozen",
    "select_space_fill",
    "select_query_agnostic_ig",
    "select_query_bald_like",
    "select_ec2_like",
    "select_greedy_risk_one_step",
    "open_pairs",
    "greedy_pick_at",
    "SELECTORS",
    "Leaf",
    "expand_tree",
    "sample_path",
    "leaf_risk_at_k",
    "readings_from_leaves",
    "seed_for",
    "run_strategy",
    "run_full_tree_reference",
    "capped_cost_fixed_risk",
    "assign_split",
    "manifest_predicate_check",
    "aggregate_cell",
]
