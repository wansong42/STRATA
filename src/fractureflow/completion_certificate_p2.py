# -*- coding: utf-8 -*-
"""Scalable completion certificates for large 3-D volumes (P2 add-on).

This module never replaces ``fractureflow.completion_certificate``.  It reuses the
frozen ``r3_label`` predicate and the frozen DISCONNECTED witness rule, and only
substitutes a cheaper-but-still-rigorous computation of the CONNECTED bound.

Why it is needed
----------------
``completion_certificate._second_body_upper`` falls back to ``t // 2`` whenever
``len(sizes) * max(v, 1) > 2_000_000``.  On the real 3-D payload
(v ~ 1.3e6 .. 1.1e7, n up to ~5200 components) that fallback fires on nearly every
state, and ``t // 2`` sits far above ``tau * v`` for ``tau = 1/20``; CONNECTED can
therefore never be certified.  That fallback is a runtime safeguard, not part of the
R3 predicate.

Soundness (full argument in ``bounds_proof.md``)
-----------------------------------------------
Let ``s_1 >= s_2 >= ...`` be the visible 6-connected void bodies, ``V = sum s_i``,
``H`` an upper bound on hidden void, ``T = V + H``.  Every body of any admissible
completion is ``(a union of whole visible components) + (some hidden void)``, so a
body whose visible part has subset sum ``s`` is at most ``s + H``.

For the two largest bodies ``b1 >= b2``:

* ``b1 + b2 <= T``                       =>  ``b2 <= T // 2``;
* taking the subset sum ``s`` of ``b2``:  ``b2 <= min(s, V - s) + H``.

Hence, with ``S`` the set of reachable subset sums,

    U_sub = min( T // 2 ,  max_{s in S} min(s, V - s) + H )

is an upper bound on ``b2`` for every completion.  Since ``g(s) = min(s, V-s) = g(V-s)``
and ``V - s`` is reachable whenever ``s`` is, the maximum of ``g`` is attained at a
reachable ``s <= V // 2``; call it ``s*``.  So ``U_sub = min(T//2, s* + H)``.

Two corollaries:

* ``U_sub <= U_dom := min(T // 2, (V - s_1) + H)`` always, because ``s* <= V - s_1``
  (if ``s*`` excludes ``s_1`` then ``s* <= V - s_1``; if it includes ``s_1`` then
  ``s* <= V//2 <= V - s_1``).
* **If ``s_1 > V // 2`` then ``U_sub == U_dom`` exactly**: ``V - s_1 < V//2`` is a
  reachable sum, so ``s* >= V - s_1``, and ``s* <= V - s_1`` from the line above.
  This is the regime of every measured real state, so the bitset adds no power there
  and ``U_dom`` alone decides the CONNECTED predicate.

CONNECTED is certified iff ``U < tau * V`` with a **strict** inequality, because R3
counts a body that exactly reaches the threshold.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage

from fractureflow.completion_certificate import (
    CONNECTED,
    DISCONNECTED,
    UNDECIDABLE,
    _STRUCTURE_6,
    _fraction,
    _number,
    r3_label,
)

METHOD_EXACT = "EXACT_SUBSET_SUM_BITSET"
METHOD_FASTPATH = "DOMINANT_MAJORITY_EXACT"
METHOD_DOMINANT = "DOMINANT_REMAINDER_UPPER"
DEFAULT_MAX_BITS = 8_000_000


def _check_sizes(component_sizes: Sequence[int]) -> list[int]:
    sizes = [int(s) for s in component_sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError("component sizes must be positive")
    return sizes


def dominant_remainder_upper(component_sizes: Sequence[int], max_hidden_void: int) -> dict:
    """Return ``U_dom = min((V+H)//2, (V - s_1) + H)`` with provenance."""
    sizes = _check_sizes(component_sizes)
    h = int(max_hidden_void)
    if h < 0:
        raise ValueError("max_hidden_void must be non-negative")
    v = sum(sizes)
    s1 = max(sizes) if sizes else 0
    t = v + h
    return {
        "upper": int(min(t // 2, (v - s1) + h)),
        "visible_total": v,
        "largest_visible_component": s1,
        "void_total_max": t,
        "method": METHOD_DOMINANT,
    }


def exact_subset_sum_near_half(
    component_sizes: Sequence[int],
    *,
    max_bits: int = DEFAULT_MAX_BITS,
) -> dict:
    """Exact ``s* = max{s in S : s <= V//2}``.

    Fast path: when ``s_1 > V // 2`` the answer is ``V - s_1`` with no bitset at all
    (see the module docstring).  Otherwise a single truncated Python-int bitset over
    the non-dominant components is used, bounded by ``V // 2``.
    """
    sizes = _check_sizes(component_sizes)
    v = sum(sizes)
    half = v // 2
    s1 = max(sizes) if sizes else 0
    if half < 1:
        return {"exact": True, "s_star": 0, "visible_total": v, "half_window": half,
                "bits": 0, "over_budget": False, "mode": METHOD_FASTPATH}
    if s1 > half:
        return {"exact": True, "s_star": int(v - s1), "visible_total": v,
                "half_window": half, "bits": 0, "over_budget": False,
                "mode": METHOD_FASTPATH,
                "note": "dominant body exceeds V//2, so s* = V - s1 exactly"}

    # No majority body: build a bitset over every component, truncated at V//2.
    if half > max_bits:
        return {"exact": False, "s_star": None, "visible_total": v, "half_window": half,
                "bits": half, "over_budget": True, "mode": METHOD_EXACT,
                "reason": "V//2 exceeds the declared bit budget"}
    mask = (1 << (half + 1)) - 1
    bits = 1
    for size in sizes:
        if size <= half:
            bits |= bits << size
            bits &= mask
    return {"exact": True, "s_star": int(bits.bit_length() - 1), "visible_total": v,
            "half_window": half, "bits": half + 1, "over_budget": False,
            "mode": METHOD_EXACT}


def scalable_second_body_upper(
    component_sizes: Sequence[int],
    max_hidden_void: int,
    *,
    tau: Any = Fraction(1, 20),
    max_bits: int = DEFAULT_MAX_BITS,
) -> dict:
    """Upper-bound the second-largest body using ``U_sub`` (exact) or ``U_dom``."""
    tau_f = _fraction(tau)
    if not 0 < tau_f <= 1:
        raise ValueError("tau must be in (0,1]")
    dom = dominant_remainder_upper(component_sizes, max_hidden_void)
    h = int(max_hidden_void)
    v = dom["visible_total"]
    t = dom["void_total_max"]
    sub = exact_subset_sum_near_half(component_sizes, max_bits=max_bits)
    if sub["exact"]:
        s_star = int(sub["s_star"])
        upper = int(min(t // 2, s_star + h))
        method = sub["mode"]
    else:
        upper = int(dom["upper"])
        method = METHOD_DOMINANT + "_BITSET_OVER_BUDGET"
    return {
        "upper": upper,
        "method": method,
        "dominant_remainder_upper": int(dom["upper"]),
        "subset_sum": sub,
        "upper_subset_sum": int(upper) if sub["exact"] else None,
        "upper_never_looser_than_dominant": upper <= dom["upper"],
        "equals_dominant": upper == dom["upper"],
    }


def r3_certificate_scalable(
    observation: np.ndarray,
    *,
    tau: Any = Fraction(1, 20),
    max_hidden_void: Optional[int] = None,
    max_bits: int = DEFAULT_MAX_BITS,
) -> dict:
    """Conservative answer valid for every admissible completion (large volumes).

    Mirrors ``completion_certificate.r3_certificate`` case by case: the no-hidden
    path, the ``v == 0`` guard and the DISCONNECTED witness rule are reused
    unchanged; only the CONNECTED bound is substituted.
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
    counts = np.bincount(visible_labels.ravel(), minlength=n_visible + 1)[1:]
    sizes_in_label_order = [int(x) for x in counts]
    sizes_sorted = sorted(sizes_in_label_order, reverse=True)

    if n_hidden == 0:
        exact = r3_label(visible, tau=tau_f)
        second_exact = sizes_sorted[1] if len(sizes_sorted) > 1 else 0
        return {
            "schema": "completion-certificate/r3-v2-scalable",
            "tau": str(tau_f),
            "shape_zyx": list(a.shape),
            "visible_void_total": v,
            "hidden_voxels": 0,
            "max_hidden_void": 0,
            "void_total_min": v,
            "void_total_max": v,
            "visible_component_sizes": sizes_sorted,
            "second_body_upper": second_exact,
            "second_body_upper_method": "EXACT_NO_HIDDEN",
            "connected_margin_tau_v_minus_upper": _number(tau_f * v - second_exact),
            "decision": exact["result"],
            "certificate_kind": "EXACT_NO_HIDDEN",
            "exact_r3": exact,
            "is_valid_for_all_completions": True,
        }

    bound = scalable_second_body_upper(sizes_sorted, h, tau=tau_f, max_bits=max_bits)
    second_upper = bound["upper"]
    base = {
        "schema": "completion-certificate/r3-v2-scalable",
        "tau": str(tau_f),
        "shape_zyx": list(a.shape),
        "visible_void_total": v,
        "hidden_voxels": n_hidden,
        "max_hidden_void": h,
        "void_total_min": v,
        "void_total_max": t_max,
        "visible_component_sizes": sizes_sorted,
        "n_visible_components": len(sizes_sorted),
        "second_body_upper": second_upper,
        "second_body_upper_method": bound["method"],
        "connected_margin_tau_v_minus_upper": _number(tau_f * v - second_upper),
        "bound_audit": {
            "dominant_remainder_upper": bound["dominant_remainder_upper"],
            "subset_sum": bound["subset_sum"],
            "upper_subset_sum": bound["upper_subset_sum"],
            "equals_dominant": bound["equals_dominant"],
        },
    }
    if v == 0:
        return {**base, "decision": UNDECIDABLE, "certificate_kind": "NO_VISIBLE_VOID",
                "reason": "empty, connected and disconnected completions remain possible",
                "is_valid_for_all_completions": True}

    # Frozen DISCONNECTED rule, verbatim: two visible bodies, each at least
    # tau * void_total_max, lying in different hidden-wildcard components.
    # Short circuit: fewer than two bodies can reach the threshold, so no separated
    # witness can exist and the (expensive) potential-component labelling is skipped.
    candidates = [cid for cid in range(1, n_visible + 1)
                  if sizes_in_label_order[cid - 1] > 0
                  and Fraction(sizes_in_label_order[cid - 1], t_max) >= tau_f]
    large = []
    if len(candidates) >= 2:
        potential, _ = ndimage.label(a != 0, structure=_STRUCTURE_6)
        pid_of = ndimage.minimum(potential, visible_labels, candidates)
        for cid, pid in zip(candidates, pid_of):
            large.append({"component_id": int(cid), "size": sizes_in_label_order[cid - 1],
                          "potential_component_id": int(pid)})
    if len({row["potential_component_id"] for row in large}) >= 2:
        return {**base, "decision": DISCONNECTED,
                "certificate_kind": "TWO_SEPARATED_LARGE_BODIES",
                "disconnected_witnesses": large,
                "is_valid_for_all_completions": True}

    if Fraction(second_upper, v) < tau_f:
        return {**base, "decision": CONNECTED,
                "certificate_kind": "SECOND_BODY_BELOW_THRESHOLD_FOR_ALL_COMPLETIONS",
                "is_valid_for_all_completions": True}
    return {**base, "decision": UNDECIDABLE, "certificate_kind": "BOUNDS_DO_NOT_SEPARATE",
            "reason": "neither the connected upper bound nor the disconnected witness is decisive",
            "large_visible_components": large, "is_valid_for_all_completions": True}

def frozen_core_second_upper_by_formula(
    component_sizes: Sequence[int],
    max_hidden_void: int,
) -> dict:
    """Reproduce ``completion_certificate._second_body_upper`` in O(V/64).

    The frozen core either falls back to ``t // 2`` or runs an exact subset-sum
    relaxation.  Its exact branch enumerates ``range(v + 1)`` in Python, which costs
    ~72 s at v = 1.6e6; the closed form below returns the identical value using the
    bitset.  ``mirrors`` states which frozen branch was reproduced so a caller can
    sample-verify against the real function.
    """
    sizes = _check_sizes(component_sizes)
    h = int(max_hidden_void)
    if h < 0:
        raise ValueError("max_hidden_void must be non-negative")
    v = sum(sizes)
    t = v + h
    if len(sizes) > 128 or v > 2_000_000 or len(sizes) * max(v, 1) > 2_000_000:
        return {"upper": int(t // 2), "method": "TOTAL_HALF_UPPER",
                "mirrors": "fallback_branch", "visible_total": v}
    sub = exact_subset_sum_near_half(sizes)
    if not sub["exact"]:
        return {"upper": int(t // 2), "method": "TOTAL_HALF_UPPER",
                "mirrors": "fallback_branch_bitset_over_budget", "visible_total": v}
    s_star = int(sub["s_star"])
    best = min(t // 2, min(s_star, v - s_star) + h)
    return {"upper": int(best), "method": "EXACT_VISIBLE_SUBSET_SUM_RELAXATION",
            "mirrors": "exact_branch", "visible_total": v, "s_star": s_star}

def certify_p2_and_frozen(
    observation: np.ndarray,
    *,
    tau: Any = Fraction(1, 20),
    max_hidden_void: Optional[int] = None,
    max_bits: int = DEFAULT_MAX_BITS,
) -> dict:
    """Compute the frozen-core bound (A3) and the P2 bound (A4) from ONE labelling pass.

    Byte-identical results to calling :func:`r3_certificate_scalable` and reproducing
    ``completion_certificate.r3_certificate`` separately, but it labels the visible
    mask once instead of three times.  On the 8.1e7-voxel states each labelling pass
    costs seconds, so this is what keeps the 432-evaluation sweep tractable.
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
    in_order = [int(x) for x in np.bincount(visible_labels.ravel(), minlength=n_visible + 1)[1:]]
    sizes = sorted(in_order, reverse=True)

    if n_hidden == 0:
        mine = r3_certificate_scalable(observation, tau=tau_f, max_hidden_void=max_hidden_void,
                                      max_bits=max_bits)
        return {"p2": mine, "frozen": dict(mine), "n_visible_components": n_visible}

    p2_bound = scalable_second_body_upper(sizes, h, tau=tau_f, max_bits=max_bits)
    p2_upper = p2_bound["upper"]
    frozen_bound = frozen_core_second_upper_by_formula(sizes, h)
    frozen_upper = frozen_bound["upper"]

    # Both arms share the same observation, so the potential-component labelling is
    # done at most once here instead of inside each _decide call.
    cand_all = [cid for cid in range(1, n_visible + 1)
                if in_order[cid - 1] > 0 and Fraction(in_order[cid - 1], t_max) >= tau_f]
    sep_pids = None
    if len(cand_all) >= 2:
        potential, _ = ndimage.label(a != 0, structure=_STRUCTURE_6)
        sep_pids = {int(x) for x in ndimage.minimum(potential, visible_labels, cand_all)}

    def _decide(upper: int, method: str) -> dict:
        if v == 0:
            return {"decision": UNDECIDABLE, "certificate_kind": "NO_VISIBLE_VOID",
                    "second_body_upper": upper, "second_body_upper_method": method}
        if sep_pids is not None and len(sep_pids) >= 2:
            return {"decision": DISCONNECTED, "certificate_kind": "TWO_SEPARATED_LARGE_BODIES",
                    "second_body_upper": upper, "second_body_upper_method": method,
                    "disconnected_witness_component_ids": [int(c) for c in cand_all]}
        if Fraction(upper, v) < tau_f:
            return {"decision": CONNECTED,
                    "certificate_kind": "SECOND_BODY_BELOW_THRESHOLD_FOR_ALL_COMPLETIONS",
                    "second_body_upper": upper, "second_body_upper_method": method}
        return {"decision": UNDECIDABLE, "certificate_kind": "BOUNDS_DO_NOT_SEPARATE",
                "second_body_upper": upper, "second_body_upper_method": method}

    p2 = _decide(p2_upper, p2_bound["method"])
    fr = _decide(frozen_upper, frozen_bound["method"])
    p2.update({"tau": str(tau_f), "shape_zyx": list(a.shape), "visible_void_total": v,
               "hidden_voxels": n_hidden, "max_hidden_void": h, "void_total_max": t_max,
               "visible_component_sizes": sizes, "n_visible_components": n_visible,
               "is_valid_for_all_completions": True,
               "connected_margin_tau_v_minus_upper": _number(tau_f * v - p2_upper)})
    fr.update({"tau": str(tau_f), "shape_zyx": list(a.shape), "visible_void_total": v,
               "hidden_voxels": n_hidden, "max_hidden_void": h, "void_total_max": t_max,
               "visible_component_sizes": sizes, "n_visible_components": n_visible,
               "is_valid_for_all_completions": True,
               "connected_margin_tau_v_minus_upper": _number(tau_f * v - frozen_upper)})
    # A1 (naive visible R3) is exactly the frozen predicate evaluated on the visible
    # mask: DISCONNECTED iff two visible bodies each reach tau of the visible total.
    # Derived from the sizes already in hand, so no extra labelling pass is needed.
    n_large_visible = sum(1 for s_ in sizes if s_ > 0 and Fraction(s_, v) >= tau_f) if v else 0
    naive = CONNECTED if n_large_visible == 1 else DISCONNECTED
    return {"p2": p2, "frozen": fr, "naive_visible": naive,
            "n_large_visible_bodies": n_large_visible,
            "n_visible_components": n_visible,
            "p2_bound_audit": p2_bound, "frozen_bound": frozen_bound}
