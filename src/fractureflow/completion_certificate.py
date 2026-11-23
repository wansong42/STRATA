# -*- coding: utf-8 -*-
"""Completion-certified connectivity answers and next-observation scoring.

Two input models are supported:

* finite candidate worlds: exact equivalence classes and exact answerability;
* partial voxel volumes: conservative certificates over all completions bounded by
  ``max_hidden_void``.  An UNDECIDABLE result is never converted into a guess.

Voxel state convention: 0 = observed matrix, 1 = observed void, -1 = hidden.
R3 convention: DISCONNECTED iff at least two 6-connected void bodies each occupy
at least ``tau`` of all void voxels; otherwise CONNECTED (empty void = INVALID_END).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage

CONNECTED = "CONNECTED"
DISCONNECTED = "DISCONNECTED"
UNDECIDABLE = "UNDECIDABLE"
INVALID_END = "INVALID_END"
_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def _fraction(x: Any) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))


def _number(x: Fraction) -> int | float:
    return x.numerator if x.denominator == 1 else float(x)


def _normalise_label(x: Any) -> int:
    if x in (0, False, DISCONNECTED):
        return 0
    if x in (1, True, CONNECTED):
        return 1
    raise ValueError("label must be 0/1 or DISCONNECTED/CONNECTED: %r" % (x,))


def r3_label(mask: np.ndarray, *, tau: Any = Fraction(1, 20)) -> dict:
    """Evaluate the frozen R3 predicate on a complete Boolean void volume."""
    tau_f = _fraction(tau)
    if not 0 < tau_f <= 1:
        raise ValueError("tau must be in (0,1]")
    a = np.asarray(mask, dtype=bool)
    if a.ndim == 2:
        a = a[np.newaxis, :, :]
    if a.ndim != 3:
        raise ValueError("mask must be a 2-D stack or 3-D volume")
    labels, n = ndimage.label(a, structure=_STRUCTURE_6)
    sizes = sorted((int(x) for x in np.bincount(labels.ravel(), minlength=n + 1)[1:]),
                   reverse=True)
    total = int(sum(sizes))
    if total == 0:
        return {"result": INVALID_END, "sizes_top5": [], "void_total": 0}
    n_large = sum(Fraction(s, total) >= tau_f for s in sizes)
    return {
        "result": CONNECTED if n_large == 1 else DISCONNECTED,
        "sizes_top5": sizes[:5],
        "void_total": total,
        "n_bodies_ge_threshold": n_large,
        "tau": str(tau_f),
    }


def _second_body_upper(component_sizes: Sequence[int], max_hidden_void: int) -> tuple[int, str]:
    """Upper-bound the second-largest body over all hidden completions.

    Visible components are indivisible; hidden void is divisible between two
    bodies.  For each visible subset sum s, the best balanced two-body split has
    second size at most min(floor((V+H)/2), min(s,V-s)+H).  Enumerating subset
    sums therefore gives a rigorous upper bound (not necessarily geometry-tight).
    """
    sizes = [int(s) for s in component_sizes]
    v = sum(sizes)
    h = int(max_hidden_void)
    t = v + h
    # Avoid a very large bitset shift series on production volumes.  The fallback
    # is looser but remains a valid upper bound.
    if len(sizes) > 128 or v > 2_000_000 or len(sizes) * max(v, 1) > 2_000_000:
        return t // 2, "TOTAL_HALF_UPPER"
    bits = 1
    for size in sizes:
        if size <= 0:
            raise ValueError("component sizes must be positive")
        bits |= bits << size
    best = 0
    for s in range(v + 1):
        if (bits >> s) & 1:
            candidate = min(t // 2, min(s, v - s) + h)
            best = max(best, candidate)
    return best, "EXACT_VISIBLE_SUBSET_SUM_RELAXATION"


def r3_certificate(
    observation: np.ndarray,
    *,
    tau: Any = Fraction(1, 20),
    max_hidden_void: Optional[int] = None,
) -> dict:
    """Return a conservative answer valid for every admissible hidden completion.

    CONNECTED certificate: second_body_upper < tau * visible_void_total.
    DISCONNECTED certificate: two visible bodies, each at least tau * void_total_max,
    lie in different hidden-wildcard components and therefore cannot merge.
    """
    tau_f = _fraction(tau)
    if not 0 < tau_f <= 1:
        raise ValueError("tau must be in (0,1]")
    a = np.asarray(observation)
    if a.ndim == 2:
        a = a[np.newaxis, :, :]
    if a.ndim != 3:
        raise ValueError("observation must be a 2-D stack or 3-D volume")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("observation dtype must be integer")
    if not np.isin(a, [-1, 0, 1]).all():
        raise ValueError("observation values must be -1/0/1")
    hidden = a == -1
    n_hidden = int(hidden.sum())
    h = n_hidden if max_hidden_void is None else int(max_hidden_void)
    if h < 0 or h > n_hidden:
        raise ValueError("max_hidden_void must be between 0 and the hidden voxel count")
    visible = a == 1
    v = int(visible.sum())
    t_max = v + h
    visible_labels, n_visible = ndimage.label(visible, structure=_STRUCTURE_6)
    sizes = [int(x) for x in np.bincount(visible_labels.ravel(), minlength=n_visible + 1)[1:]]
    if n_hidden == 0:
        exact = r3_label(visible, tau=tau_f)
        second_exact = sorted(sizes, reverse=True)[1] if len(sizes) > 1 else 0
        return {
            "schema": "completion-certificate/r3-v1",
            "tau": str(tau_f),
            "shape_zyx": list(a.shape),
            "visible_void_total": v,
            "hidden_voxels": 0,
            "max_hidden_void": 0,
            "void_total_min": v,
            "void_total_max": v,
            "visible_component_sizes": sorted(sizes, reverse=True),
            "second_body_upper": second_exact,
            "second_body_upper_method": "EXACT_NO_HIDDEN",
            "connected_margin_tau_v_minus_upper": _number(tau_f * v - second_exact),
            "decision": exact["result"],
            "certificate_kind": "EXACT_NO_HIDDEN",
            "exact_r3": exact,
            "is_valid_for_all_completions": True,
        }
    second_upper, bound_method = _second_body_upper(sizes, h)

    base = {
        "schema": "completion-certificate/r3-v1",
        "tau": str(tau_f),
        "shape_zyx": list(a.shape),
        "visible_void_total": v,
        "hidden_voxels": n_hidden,
        "max_hidden_void": h,
        "void_total_min": v,
        "void_total_max": t_max,
        "visible_component_sizes": sorted(sizes, reverse=True),
        "second_body_upper": second_upper,
        "second_body_upper_method": bound_method,
        "connected_margin_tau_v_minus_upper": _number(tau_f * v - second_upper),
    }
    if v == 0:
        return {**base, "decision": UNDECIDABLE, "certificate_kind": "NO_VISIBLE_VOID",
                "reason": "empty, connected and disconnected completions remain possible",
                "is_valid_for_all_completions": True}

    # Potential components treat every hidden voxel as void.  Two visible bodies
    # in different potential components cannot be merged by any hidden completion.
    potential, _ = ndimage.label(a != 0, structure=_STRUCTURE_6)
    large = []
    for cid in range(1, n_visible + 1):
        size = int((visible_labels == cid).sum())
        if Fraction(size, t_max) >= tau_f:
            pid = int(potential[visible_labels == cid][0])
            large.append({"component_id": cid, "size": size, "potential_component_id": pid})
    if len({row["potential_component_id"] for row in large}) >= 2:
        return {
            **base,
            "decision": DISCONNECTED,
            "certificate_kind": "TWO_SEPARATED_LARGE_BODIES",
            "disconnected_witnesses": large,
            "is_valid_for_all_completions": True,
        }

    if Fraction(second_upper, v) < tau_f:
        return {
            **base,
            "decision": CONNECTED,
            "certificate_kind": "SECOND_BODY_BELOW_THRESHOLD_FOR_ALL_COMPLETIONS",
            "is_valid_for_all_completions": True,
        }
    return {
        **base,
        "decision": UNDECIDABLE,
        "certificate_kind": "BOUNDS_DO_NOT_SEPARATE",
        "reason": "neither the connected upper bound nor the disconnected witness is decisive",
        "large_visible_components": large,
        "is_valid_for_all_completions": True,
    }


def equivalent_set(
    states: Sequence[Hashable],
    actions: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Any],
) -> list[Hashable]:
    """Filter candidate worlds by all observed action outcomes."""
    members = list(states)
    if not members:
        raise ValueError("states must be non-empty")
    for action_id, outcome in observed.items():
        if action_id not in actions:
            raise ValueError("observed action is absent from the action bank: %s" % action_id)
        outcomes = actions[action_id]["outcomes"]
        members = [s for s in members if outcomes[s] == outcome]
    if not members:
        raise ValueError("observations conflict with every candidate world")
    return members


def decide_equivalent_set(members: Sequence[Hashable], labels_of: Mapping[Hashable, Any]) -> dict:
    """Answer exactly when every member of the current equivalence class agrees."""
    if not members:
        raise ValueError("equivalence class must be non-empty")
    labels = [_normalise_label(labels_of[s]) for s in members]
    unique = sorted(set(labels))
    if len(unique) == 1:
        decision = CONNECTED if unique[0] else DISCONNECTED
        kind = "PURE_EQUIVALENCE_CLASS"
    else:
        decision = UNDECIDABLE
        kind = "LABEL_MIXED_EQUIVALENCE_CLASS"
    witnesses = []
    if len(unique) == 2:
        witnesses = [
            {"state": s, "label": DISCONNECTED if _normalise_label(labels_of[s]) == 0 else CONNECTED}
            for s in members[:1]
        ]
        other = next(s for s in members if _normalise_label(labels_of[s]) != _normalise_label(labels_of[members[0]]))
        witnesses.append({"state": other, "label": DISCONNECTED if _normalise_label(labels_of[other]) == 0 else CONNECTED})
    return {
        "decision": decision,
        "certificate_kind": kind,
        "equivalent_states": list(members),
        "n_equivalent_states": len(members),
        "label_witnesses": witnesses,
        "is_valid_for_all_completions": True,
    }


def rank_next_observations(
    members: Sequence[Hashable],
    labels_of: Mapping[Hashable, Any],
    actions: Mapping[str, Mapping[str, Any]],
    *,
    priors: Optional[Mapping[Hashable, Any]] = None,
    used_actions: Sequence[str] = (),
) -> list[dict]:
    """Rank actions by exact expected decisive mass per unit declared cost.

    The score is a planning criterion, not a certificate that the selected action
    will succeed.  Priors default to uniform over the current equivalence class.
    """
    if not members:
        raise ValueError("members must be non-empty")
    if priors is None:
        weights = {s: Fraction(1, len(members)) for s in members}
    else:
        weights = {s: _fraction(priors.get(s, 0)) for s in members}
        total = sum(weights.values(), Fraction(0))
        if total <= 0 or any(w < 0 for w in weights.values()):
            raise ValueError("priors must be non-negative with positive mass on members")
        weights = {s: w / total for s, w in weights.items()}
    rows = []
    used = set(used_actions)
    for action_id, spec in actions.items():
        if action_id in used:
            continue
        cost = _fraction(spec["cost"])
        if cost <= 0:
            raise ValueError("action cost must be positive: %s" % action_id)
        outcomes = spec["outcomes"]
        groups: dict[Any, list[Hashable]] = {}
        for state in members:
            groups.setdefault(outcomes[state], []).append(state)
        decisive_mass = Fraction(0)
        branch_rows = []
        for outcome, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
            group_labels = {_normalise_label(labels_of[s]) for s in group}
            pure = len(group_labels) == 1
            mass = sum((weights[s] for s in group), Fraction(0))
            if pure:
                decisive_mass += mass
            branch_rows.append({"outcome": outcome, "n_states": len(group),
                                "prior_mass": _number(mass), "decisive": pure})
        score = decisive_mass / cost
        rows.append({
            "action_id": action_id,
            "cost": _number(cost),
            "expected_decisive_mass": _number(decisive_mass),
            "decisive_for_every_outcome": all(r["decisive"] for r in branch_rows),
            "score_decisive_mass_per_cost": _number(score),
            "branches": branch_rows,
            "planning_only_not_answer_certificate": True,
        })
    rows.sort(key=lambda r: (
        -float(_fraction(r["score_decisive_mass_per_cost"])),
        -float(_fraction(r["expected_decisive_mass"])),
        not r["decisive_for_every_outcome"],
        float(_fraction(r["cost"])),
        str(r["action_id"]),
    ))
    return rows


def answer_finite(
    states: Sequence[Hashable],
    labels_of: Mapping[Hashable, Any],
    actions: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Any],
    *,
    priors: Optional[Mapping[Hashable, Any]] = None,
) -> dict:
    """End-to-end finite-world answer, certificate and next-action ranking."""
    members = equivalent_set(states, actions, observed)
    answer = decide_equivalent_set(members, labels_of)
    ranked = rank_next_observations(members, labels_of, actions, priors=priors,
                                    used_actions=list(observed))
    return {
        "schema": "completion-certificate/finite-v1",
        **answer,
        "observed_actions": dict(observed),
        "next_observation": ranked[0] if ranked else None,
        "ranked_next_observations": ranked,
    }
