"""M5A exact reference core: finite-H active-measurement software verification.

Implements the frozen contract
``results/obs_equiv_20261024/m5a_exact_reference_design_20261025/contract.json``
(schema ``obs_equiv.m5a_exact_reference.contract.v1``). This is software
verification only: no scientific reading, no real data, no experiment. H1
(query-aware policies reaching decidability at lower cost) remains an
unverified target; nothing computed here is evidence for or against it.

Frozen semantics honoured by this module:

* Shared belief update ``p(z|b,a) = sum_{h in z} b(h)`` and
  ``b'(h|a,z) = b(h)*1[Z_a(h)=z] / p(z|b,a)``; zero-probability branches never
  enter a tree.
* Brier risk ``R_Q(b) = p(1-p)`` is the PRIMARY optimization risk; the 0-1
  Bayes risk ``min(p, 1-p)`` is a separate secondary column, never mixed into
  optimization and never post-hoc selected.
* One-step exact value ``V1(a|b) = sum_z p(z|b,a) * R_Q(b'|a,z)``; greedy is
  the myopic argmin (tie: smallest action id) and is never called optimal.
* Endpoints: fixed-budget ``OPT(B) = min_policy E[terminal R_Q]`` over
  conditional policies with per-path budget ``sum c(a) <= B`` (stopping is a
  policy), and fixed-risk ``COST(rho)`` = minimum expected total cost of a
  conditional policy whose EVERY branch ends with ``R_Q <= rho``; worst-path
  cost of the chosen policy is reported alongside. Infeasible rho yields
  status ``UNRESOLVED`` (kept in the denominator, never dropped).
* Fail-closed (SystemExit): empty hypothesis set; empty action set; empty
  declared outcome stream (partition with no non-empty cell; zero-total
  prior); replaying an outcome with zero probability under the belief.
* ``NO_INFORMATIVE_ACTION``: when at the INITIAL belief every action has
  ``V1(a|b0) == R_Q(b0)``, all strategy endpoints return that status and
  produce no policy comparison (diagnostics such as the V1 table remain
  allowed).
* Tie-break: among co-optimal conditional policies, the lexicographically
  smallest complete action sequence, operationalised as a nested DFS pre-order
  key over action ids with outcomes in sorted cell order; STOP contributes
  the empty tuple, which sorts before any action id.

Search note: both searches skip actions that do not split the current belief
support (fewer than two positive-mass outcome cells). Such an action leaves
the belief unchanged while consuming budget or adding cost, so acting
immediately dominates it; the skip is exactness-preserving, not a heuristic.
``_opt_enumerative`` additionally requires ``budget <= 2 * min(cost)`` so any
policy has at most two actions per path, making flat product enumeration
exact.

Engineering constraints: pure standard library, no PyTorch/JAX, no new
dependencies, no file I/O. Problems, query truth tables and outcome mappings
are injected by the caller (evaluator side); this module never reads
evaluator files. All arithmetic is exact ``fractions.Fraction`` (decimal
costs enter through their string form, so 0.6 means 3/5); there is no float
path, so the contract's 1e-12 float-tolerance clause is not exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Optional, Sequence

SCHEMA = "obs_equiv.m5a_exact_reference.contract.v1"
RISK_PRIMARY = "BRIER_RISK"
RISK_SECONDARY = "BAYES_0_1_RISK"
STATUS_OK = "OK"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_NO_INFORMATIVE = NO_INFORMATIVE_ACTION = "NO_INFORMATIVE_ACTION"
STRATEGY_COMPARISON_FORBIDDEN = "FORBIDDEN"
METHOD_RECURSIVE = "recursive"
METHOD_ITERATIVE = "iterative"
METHOD_ENUMERATIVE = "enumerative"
ENUMERATIVE_POLICY_CAP = 200_000

Belief = Mapping[str, Fraction]


def exact_fraction(value, name: str = "value") -> Fraction:
    """Exact conversion; floats are read through their decimal string form."""
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, float, str)):
        raise ValueError(f"{name} must be int/str/float/Fraction, got {type(value).__name__}")
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _fail_closed(reason: str) -> None:
    raise SystemExit(f"M5A-FAIL-CLOSED: {reason}")


@dataclass(frozen=True)
class Action:
    """One candidate measurement: deterministic outcome map over H plus cost."""

    action_id: str
    cost: Fraction
    partition: tuple  # tuple of cells; each cell is a sorted tuple of hypothesis ids

    @classmethod
    def of(cls, action_id: str, cost, partition: Sequence[Sequence[str]]) -> "Action":
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("action_id must be a non-empty string")
        cells = [cell for cell in (tuple(sorted(set(c))) for c in partition) if cell]
        if not cells:
            _fail_closed(
                f"action {action_id!r} declares an empty outcome stream "
                "(partition has no non-empty cell)"
            )
        seen = set()
        for cell in cells:
            overlap = seen.intersection(cell)
            if overlap:
                raise ValueError(f"action {action_id!r} cells overlap on {sorted(overlap)}")
            seen.update(cell)
        return cls(action_id, exact_fraction(cost, "cost"), tuple(sorted(cells)))


@dataclass(frozen=True)
class Problem:
    hypotheses: tuple
    prior: dict
    query: dict
    actions: tuple

    @classmethod
    def build(
        cls,
        hypotheses: Sequence[str],
        prior_weights: Sequence,
        query: Mapping[str, int],
        actions: Sequence[Action],
    ) -> "Problem":
        hyps = tuple(hypotheses)
        if not hyps:
            _fail_closed("hypothesis set H is empty")
        if len(set(hyps)) != len(hyps):
            raise ValueError("duplicate hypothesis ids")
        if not actions:
            _fail_closed("action set A is empty")
        if len(prior_weights) != len(hyps):
            raise ValueError("prior_weights length must match |H|")
        weights = [exact_fraction(w, "prior weight") for w in prior_weights]
        if any(w < 0 for w in weights):
            raise ValueError("prior weights must be non-negative")
        if sum(weights) == 0:
            _fail_closed("prior has zero total mass: the outcome stream is empty")
        if set(query) != set(hyps):
            raise ValueError("query must be defined exactly on H")
        if any(query[h] not in (0, 1) for h in hyps):
            raise ValueError("query values must be 0 or 1")
        ids = [a.action_id for a in actions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate action ids")
        universe = set(hyps)
        for action in actions:
            if action.cost <= 0:
                raise ValueError(f"action {action.action_id!r} cost must be > 0")
            covered = set()
            for cell in action.partition:
                extra = set(cell) - universe
                if extra:
                    raise ValueError(
                        f"action {action.action_id!r} cell references unknown "
                        f"hypotheses {sorted(extra)}"
                    )
                covered.update(cell)
            if covered != universe:
                raise ValueError(
                    f"action {action.action_id!r} partition must cover H exactly; "
                    f"missing {sorted(universe - covered)}"
                )
        return cls(hyps, dict(zip(hyps, weights)), dict(query), tuple(actions))


@dataclass(frozen=True)
class PolicyNode:
    """Conditional policy tree: STOP (action is None) or an action + subtrees."""

    action: Optional[str]
    children: tuple = ()

    def flat_sequence(self) -> list:
        """DFS pre-order action ids in sorted-outcome order (STOP adds none)."""
        if self.action is None:
            return []
        out = [self.action]
        for _cell, child in self.children:
            out.extend(child.flat_sequence())
        return out


def policy_key(node: PolicyNode) -> tuple:
    """Nested lexicographic key implementing the frozen tie-break rule."""
    if node.action is None:
        return ()
    return (node.action,) + tuple(policy_key(child) for _cell, child in node.children)


# ---------------------------------------------------------------------------
# Shared belief update and risks
# ---------------------------------------------------------------------------


def initial_belief(problem: Problem) -> dict:
    total = sum(problem.prior.values())
    return {h: p / total for h, p in problem.prior.items() if p > 0}


def query_probability(belief: Belief, query: Mapping[str, int]) -> Fraction:
    return sum((belief[h] * query[h] for h in belief), Fraction(0))


def risk_brier(belief: Belief, query: Mapping[str, int]) -> Fraction:
    p = query_probability(belief, query)
    return p * (1 - p)


def risk_bayes_01(belief: Belief, query: Mapping[str, int]) -> Fraction:
    p = query_probability(belief, query)
    return min(p, 1 - p)


def risk_columns(belief: Belief, query: Mapping[str, int]) -> dict:
    """Both risks as separate columns; only the primary ever drives a search."""
    return {
        "primary_brier": risk_brier(belief, query),
        "secondary_bayes_01": risk_bayes_01(belief, query),
    }


def outcome_cells(belief: Belief, action: Action) -> tuple:
    """Positive-mass outcome cells with probabilities (zero branches never enter)."""
    support = set(belief)
    out = []
    for cell in action.partition:
        mass = sum((belief[h] for h in cell if h in support), Fraction(0))
        if mass > 0:
            out.append((cell, mass))
    if not out:
        _fail_closed(
            f"action {action.action_id!r} yields an empty outcome stream on the "
            "current belief"
        )
    return tuple(out)


def posterior(belief: Belief, action: Action, cell: Sequence[str]) -> dict:
    """Shared exact update; a zero-probability observed outcome fails closed."""
    cell_set = set(cell)
    mass = sum((p for h, p in belief.items() if h in cell_set), Fraction(0))
    if mass == 0:
        _fail_closed(
            f"replayed outcome {sorted(cell_set)} has zero probability for action "
            f"{action.action_id!r} under the current belief: empty outcome stream"
        )
    return {h: p / mass for h, p in belief.items() if h in cell_set and p > 0}


def v1_columns(belief: Belief, action: Action, query: Mapping[str, int]) -> dict:
    """One-step exact value of ``action`` under both risk columns."""
    primary = Fraction(0)
    secondary = Fraction(0)
    for cell, mass in outcome_cells(belief, action):
        columns = risk_columns(posterior(belief, action, cell), query)
        primary += mass * columns["primary_brier"]
        secondary += mass * columns["secondary_bayes_01"]
    return {"primary_brier": primary, "secondary_bayes_01": secondary}


def v1(belief: Belief, action: Action, query: Mapping[str, int]) -> Fraction:
    return v1_columns(belief, action, query)["primary_brier"]


def greedy_pick(problem: Problem, budget: Optional[Fraction] = None) -> tuple:
    """Myopic argmin V1 at the initial belief (tie: smallest action id).

    Diagnostic only, never a global-optimum claim. Returns (action_id, V1),
    or (None, current risk) when no action is affordable.
    """
    belief = initial_belief(problem)
    best = (None, risk_brier(belief, problem.query))
    for action in sorted(problem.actions, key=lambda a: a.action_id):
        if budget is not None and action.cost > budget:
            continue
        value = v1(belief, action, problem.query)
        if best[0] is None or value < best[1]:
            best = (action.action_id, value)
    return best


def no_informative_action(problem: Problem) -> bool:
    """True iff every action leaves the one-step risk unchanged at the root."""
    belief = initial_belief(problem)
    current = risk_brier(belief, problem.query)
    return all(v1(belief, a, problem.query) == current for a in problem.actions)


# ---------------------------------------------------------------------------
# Optimality-certificate replay
# ---------------------------------------------------------------------------


def replay_policy(problem: Problem, policy: PolicyNode, belief: Optional[dict] = None) -> dict:
    """Branch-by-branch replay of a conditional policy (certificate evaluation).

    Walks the policy against the shared belief update, producing one record
    per terminal leaf: path actions, leaf cell, path probability, cost, and
    both risk columns. Aggregates: expected terminal risk (both columns),
    expected cost, worst-path cost, max leaf primary risk.
    """
    if belief is None:
        belief = initial_belief(problem)
    records = []

    def full_walk(node: PolicyNode, current: dict, prob: Fraction, path: list, cost: Fraction):
        if node.action is None:
            columns = risk_columns(current, problem.query)
            records.append(
                {
                    "path": list(path),
                    "cell": tuple(sorted(current)),
                    "path_probability": prob,
                    "cost": cost,
                    "primary_brier": columns["primary_brier"],
                    "secondary_bayes_01": columns["secondary_bayes_01"],
                }
            )
            return
        action = _action_by_id(problem, node.action)
        declared = {cell: child for cell, child in node.children}
        for cell, mass in outcome_cells(current, action):
            if cell not in declared:
                raise ValueError(f"policy lacks a subtree for observed cell {cell}")
            full_walk(
                declared[cell],
                posterior(current, action, cell),
                prob * mass,
                path + [node.action],
                cost + action.cost,
            )

    full_walk(policy, belief, Fraction(1), [], Fraction(0))
    return {
        "leaves": records,
        "expected_terminal_primary_brier": sum(
            (r["path_probability"] * r["primary_brier"] for r in records), Fraction(0)
        ),
        "expected_terminal_secondary_bayes_01": sum(
            (r["path_probability"] * r["secondary_bayes_01"] for r in records), Fraction(0)
        ),
        "expected_cost": sum((r["path_probability"] * r["cost"] for r in records), Fraction(0)),
        "worst_cost": max((r["cost"] for r in records), default=Fraction(0)),
        "max_leaf_primary_brier": max((r["primary_brier"] for r in records), default=Fraction(0)),
    }


def _action_by_id(problem: Problem, action_id: str) -> Action:
    for action in problem.actions:
        if action.action_id == action_id:
            return action
    raise ValueError(f"unknown action id {action_id!r}")


def _belief_key(belief: dict) -> tuple:
    return tuple(sorted(belief.items()))


def _splitters(problem: Problem, belief: dict, budget: Optional[Fraction]) -> list:
    """Budget-feasible actions that split the current support, id order."""
    out = []
    for action in sorted(problem.actions, key=lambda a: a.action_id):
        if budget is not None and action.cost > budget:
            continue
        if len(outcome_cells(belief, action)) >= 2:
            out.append(action)
    return out


# ---------------------------------------------------------------------------
# Greedy rollout (myopic baseline; conditional policy built step by step)
# ---------------------------------------------------------------------------


def _greedy_node(problem: Problem, belief: dict, budget: Fraction) -> PolicyNode:
    current = risk_brier(belief, problem.query)
    best = None
    for action in _splitters(problem, belief, budget):
        value = v1(belief, action, problem.query)
        if best is None or value < best[1]:
            best = (action, value)
    if best is None or best[1] >= current:
        return PolicyNode(None, ())
    action = best[0]
    children = tuple(
        (cell, _greedy_node(problem, posterior(belief, action, cell), budget - action.cost))
        for cell, _mass in outcome_cells(belief, action)
    )
    return PolicyNode(action.action_id, children)


# ---------------------------------------------------------------------------
# OPT(B): fixed-budget endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptResult:
    status: str
    budget: Fraction
    value: Optional[Fraction] = None
    policy: Optional[PolicyNode] = None
    key: Optional[tuple] = None
    replay: Optional[dict] = None
    secondary_value: Optional[Fraction] = None


def _opt_recursive(problem: Problem, belief: dict, budget: Fraction, memo: dict) -> tuple:
    state = (_belief_key(belief), budget)
    hit = memo.get(state)
    if hit is not None:
        return hit
    best = (risk_brier(belief, problem.query), (), PolicyNode(None, ()))
    for action in _splitters(problem, belief, budget):
        expected = Fraction(0)
        children = []
        child_keys = []
        for cell, mass in outcome_cells(belief, action):
            sub_value, sub_key, sub_node = _opt_recursive(
                problem, posterior(belief, action, cell), budget - action.cost, memo
            )
            expected += mass * sub_value
            children.append((cell, sub_node))
            child_keys.append(sub_key)
        candidate_key = (action.action_id,) + tuple(child_keys)
        if expected < best[0] or (expected == best[0] and candidate_key < best[1]):
            best = (expected, candidate_key, PolicyNode(action.action_id, tuple(children)))
    memo[state] = best
    return best


def _opt_iterative(problem: Problem, belief: dict, budget: Fraction) -> tuple:
    """Explicit-stack state discovery + bottom-up table DP (no recursion)."""
    root = (_belief_key(belief), budget)
    edges_of = {}
    stack = [root]
    while stack:
        state = stack.pop()
        if state in edges_of:
            continue
        current = dict(state[0])
        remaining = state[1]
        edges = []
        for action in _splitters(problem, current, remaining):
            branch = []
            for cell, _mass in outcome_cells(current, action):
                child = (_belief_key(posterior(current, action, cell)), remaining - action.cost)
                branch.append((cell, child))
                if child not in edges_of:
                    stack.append(child)
            edges.append((action, tuple(branch)))
        edges_of[state] = edges
    table = {}
    # bottom-up: children carry strictly smaller remaining budget, so ascending
    # remaining order guarantees every child is evaluated before its parents
    for state in sorted(edges_of, key=lambda s: (s[1], s[0])):
        current = dict(state[0])
        best = (risk_brier(current, problem.query), (), PolicyNode(None, ()))
        for action, branch in edges_of[state]:
            expected = Fraction(0)
            children = []
            child_keys = []
            for cell, child in branch:
                sub_value, sub_key, sub_node = table[child]
                expected += _cell_mass(current, action, cell) * sub_value
                children.append((cell, sub_node))
                child_keys.append(sub_key)
            candidate_key = (action.action_id,) + tuple(child_keys)
            if expected < best[0] or (expected == best[0] and candidate_key < best[1]):
                best = (expected, candidate_key, PolicyNode(action.action_id, tuple(children)))
        table[state] = best
    return table[root]


def _cell_mass(belief: dict, action: Action, cell: tuple) -> Fraction:
    for candidate, mass in outcome_cells(belief, action):
        if candidate == cell:
            return mass
    raise ValueError(f"cell {cell} is not a positive-mass outcome of {action.action_id!r}")


def _opt_enumerative(problem: Problem, belief: dict, budget: Fraction) -> tuple:
    """Flat product enumeration of every within-budget policy, replay-scored.

    Exact when ``budget <= 2 * min(cost)``: no path can afford three actions,
    so every policy is (root decision, per-cell second decision, forced stop).
    Iterates decisions with plain loops; no recursion.
    """
    min_cost = min(a.cost for a in problem.actions)
    if budget > 2 * min_cost:
        raise ValueError(
            "enumerative OPT requires budget <= 2 * min cost (three-action paths exist)"
        )
    root_splitters = _splitters(problem, belief, budget)

    def second_level(sub_belief: dict, remaining: Fraction) -> list:
        # candidate subtrees below one cell: STOP, or one action with forced-STOP children
        options = [PolicyNode(None, ())]
        for action in _splitters(problem, sub_belief, remaining):
            children = tuple(
                (cell, PolicyNode(None, ())) for cell, _m in outcome_cells(sub_belief, action)
            )
            options.append(PolicyNode(action.action_id, children))
        return options

    best = None
    count = 0

    def score(node: PolicyNode) -> None:
        nonlocal best, count
        count += 1
        if count > ENUMERATIVE_POLICY_CAP:
            raise RuntimeError(f"enumerative policy cap ({ENUMERATIVE_POLICY_CAP}) exceeded")
        replay = replay_policy(problem, node)
        value = replay["expected_terminal_primary_brier"]
        key = policy_key(node)
        if best is None or value < best[0] or (value == best[0] and key < best[1]):
            best = (value, key, node)

    score(PolicyNode(None, ()))  # stop-at-root policy
    for action in root_splitters:
        cells = outcome_cells(belief, action)
        option_lists = [
            second_level(posterior(belief, action, cell), budget - action.cost)
            for cell, _mass in cells
        ]
        indices = [0] * len(cells)
        total = 1
        for options in option_lists:
            total *= len(options)
        for _ in range(total):
            children = tuple(
                (cells[i][0], option_lists[i][indices[i]]) for i in range(len(cells))
            )
            score(PolicyNode(action.action_id, children))
            for i in range(len(cells) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < len(option_lists[i]):
                    break
                indices[i] = 0
    return best


def opt_fixed_budget(problem: Problem, budget, method: str = METHOD_RECURSIVE) -> OptResult:
    """``OPT(B) = min_policy E[terminal primary risk]`` with frozen tie-break."""
    budget = exact_fraction(budget, "budget")
    if no_informative_action(problem):
        return OptResult(STATUS_NO_INFORMATIVE, budget)
    belief = initial_belief(problem)
    if method == METHOD_RECURSIVE:
        value, key, node = _opt_recursive(problem, belief, budget, {})
    elif method == METHOD_ITERATIVE:
        value, key, node = _opt_iterative(problem, belief, budget)
    elif method == METHOD_ENUMERATIVE:
        value, key, node = _opt_enumerative(problem, belief, budget)
    else:
        raise ValueError(f"unknown method {method!r}")
    replay = replay_policy(problem, node)
    return OptResult(
        STATUS_OK,
        budget,
        value=value,
        policy=node,
        key=key,
        replay=replay,
        secondary_value=replay["expected_terminal_secondary_bayes_01"],
    )


# ---------------------------------------------------------------------------
# COST(rho): fixed-risk endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostResult:
    status: str
    rho: Fraction
    expected_cost: Optional[Fraction] = None
    worst_cost: Optional[Fraction] = None
    policy: Optional[PolicyNode] = None
    key: Optional[tuple] = None
    replay: Optional[dict] = None


def _cost_recursive(problem: Problem, belief: dict, rho: Fraction, memo: dict):
    """Returns (expected, worst, key, node) or None when rho is unreachable."""
    state = _belief_key(belief)
    if state in memo:
        return memo[state]
    result = None
    if risk_brier(belief, problem.query) <= rho:
        result = (Fraction(0), Fraction(0), (), PolicyNode(None, ()))
    else:
        best = None
        for action in _splitters(problem, belief, None):
            expected = action.cost
            worst = action.cost
            children = []
            child_keys = []
            feasible = True
            for cell, mass in outcome_cells(belief, action):
                sub = _cost_recursive(problem, posterior(belief, action, cell), rho, memo)
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
        result = best
    memo[state] = result
    return result


def _cost_iterative(problem: Problem, belief: dict, rho: Fraction):
    """Explicit-stack discovery + bottom-up DP ordered by support size (no recursion)."""
    root = _belief_key(belief)
    edges_of = {}
    stack = [root]
    while stack:
        state = stack.pop()
        if state in edges_of:
            continue
        current = dict(state)
        edges = []
        if risk_brier(current, problem.query) > rho:
            for action in _splitters(problem, current, None):
                children = tuple(
                    _belief_key(posterior(current, action, cell))
                    for cell, _mass in outcome_cells(current, action)
                )
                edges.append((action, children))
                for child in children:
                    if child not in edges_of:
                        stack.append(child)
        edges_of[state] = edges
    table = {}
    for state in sorted(edges_of, key=lambda s: (len(s), s)):
        current = dict(state)
        if risk_brier(current, problem.query) <= rho:
            table[state] = (Fraction(0), Fraction(0), (), PolicyNode(None, ()))
            continue
        best = None
        for action, children in edges_of[state]:
            if any(table[child] is None for child in children):
                continue
            expected = action.cost
            worst = action.cost
            child_keys = []
            child_nodes = []
            feasible = True
            for cell, child in zip(
                [c for c, _m in outcome_cells(current, action)], children
            ):
                sub = table[child]
                if sub is None:
                    feasible = False
                    break
                expected += _cell_mass(current, action, cell) * sub[0]
                worst = max(worst, action.cost + sub[1])
                child_keys.append(sub[2])
                child_nodes.append((cell, sub[3]))
            if not feasible:
                continue
            candidate_key = (action.action_id,) + tuple(child_keys)
            if best is None or (expected, candidate_key) < (best[0], best[2]):
                best = (
                    expected,
                    worst,
                    candidate_key,
                    PolicyNode(action.action_id, tuple(child_nodes)),
                )
        table[state] = best
    return table[root]


def cost_fixed_risk(problem: Problem, rho, method: str = METHOD_RECURSIVE) -> CostResult:
    """``COST(rho)``: min expected total cost s.t. every branch reaches R_Q <= rho.

    ``rho=0`` is exact decidability (remaining hypotheses all agree on Q).
    Returns status ``UNRESOLVED`` when no conditional policy reaches rho.
    """
    rho = exact_fraction(rho, "rho")
    if no_informative_action(problem):
        return CostResult(STATUS_NO_INFORMATIVE, rho)
    belief = initial_belief(problem)
    if method == METHOD_RECURSIVE:
        result = _cost_recursive(problem, belief, rho, {})
    elif method == METHOD_ITERATIVE:
        result = _cost_iterative(problem, belief, rho)
    else:
        raise ValueError(f"unknown method {method!r} for COST")
    if result is None:
        return CostResult(STATUS_UNRESOLVED, rho)
    expected, worst, key, node = result
    return CostResult(
        STATUS_OK, rho, expected_cost=expected, worst_cost=worst, policy=node, key=key,
        replay=replay_policy(problem, node),
    )


def greedy_rollout(problem: Problem, budget) -> dict:
    """Myopic rollout under a budget; conditional policy replayed for its value."""
    budget = exact_fraction(budget, "budget")
    if no_informative_action(problem):
        return {
            "status": STATUS_NO_INFORMATIVE,
            "policy": None,
            "replay": None,
            "terminal_expected_primary_brier": None,
        }
    node = _greedy_node(problem, initial_belief(problem), budget)
    replay = replay_policy(problem, node)
    return {
        "status": STATUS_OK,
        "policy": node,
        "replay": replay,
        "terminal_expected_primary_brier": replay["expected_terminal_primary_brier"],
    }


def run_reference(problem: Problem, budget, rho) -> dict:
    """Driver: diagnostics first; strategy endpoints only when comparison is allowed."""
    belief = initial_belief(problem)
    columns0 = risk_columns(belief, problem.query)
    budget = exact_fraction(budget, "budget")
    rho = exact_fraction(rho, "rho")
    out = {
        "schema": SCHEMA,
        "initial": {
            "belief": {h: str(p) for h, p in sorted(belief.items())},
            "primary_brier": str(columns0["primary_brier"]),
            "secondary_bayes_01": str(columns0["secondary_bayes_01"]),
        },
        "v1_primary_brier": {
            a.action_id: str(v1(belief, a, problem.query)) for a in problem.actions
        },
        "v1_secondary_bayes_01": {
            a.action_id: str(v1_columns(belief, a, problem.query)["secondary_bayes_01"])
            for a in problem.actions
        },
    }
    if no_informative_action(problem):
        out["output"] = NO_INFORMATIVE_ACTION
        out["strategy_comparison"] = STRATEGY_COMPARISON_FORBIDDEN
        return out
    out["output"] = "STRATEGY_COMPARISON_ALLOWED"
    pick_id, pick_value = greedy_pick(problem, budget)
    out["greedy_picks"] = pick_id
    out["greedy_v1"] = str(pick_value)
    rollout = greedy_rollout(problem, budget)
    out["greedy_terminal_primary_brier"] = str(rollout["terminal_expected_primary_brier"])
    opt_rec = opt_fixed_budget(problem, budget, METHOD_RECURSIVE)
    opt_it = opt_fixed_budget(problem, budget, METHOD_ITERATIVE)
    out["opt"] = {
        "budget": str(budget),
        "value": str(opt_rec.value),
        "policy_sequence": opt_rec.policy.flat_sequence(),
        "recursive_iterative_agree": opt_rec.value == opt_it.value
        and opt_rec.key == opt_it.key,
        "secondary_bayes_01": str(opt_rec.secondary_value),
    }
    cost_rec = cost_fixed_risk(problem, rho, METHOD_RECURSIVE)
    cost_it = cost_fixed_risk(problem, rho, METHOD_ITERATIVE)
    if cost_rec.status == STATUS_OK:
        out["cost_rho"] = {
            "rho": str(rho),
            "expected": str(cost_rec.expected_cost),
            "worst_case": str(cost_rec.worst_cost),
            "policy_sequence": cost_rec.policy.flat_sequence(),
            "recursive_iterative_agree": (
                cost_rec.expected_cost == cost_it.expected_cost
                and cost_rec.key == cost_it.key
                and cost_rec.worst_cost == cost_it.worst_cost
            ),
        }
    else:
        out["cost_rho"] = {"rho": str(rho), "status": cost_rec.status}
    return out


__all__ = [
    "SCHEMA",
    "RISK_PRIMARY",
    "RISK_SECONDARY",
    "STATUS_OK",
    "STATUS_UNRESOLVED",
    "NO_INFORMATIVE_ACTION",
    "STRATEGY_COMPARISON_FORBIDDEN",
    "METHOD_RECURSIVE",
    "METHOD_ITERATIVE",
    "METHOD_ENUMERATIVE",
    "Action",
    "Problem",
    "PolicyNode",
    "OptResult",
    "CostResult",
    "exact_fraction",
    "initial_belief",
    "query_probability",
    "risk_brier",
    "risk_bayes_01",
    "risk_columns",
    "outcome_cells",
    "posterior",
    "v1",
    "v1_columns",
    "greedy_pick",
    "no_informative_action",
    "replay_policy",
    "policy_key",
    "opt_fixed_budget",
    "cost_fixed_risk",
    "greedy_rollout",
    "run_reference",
]
