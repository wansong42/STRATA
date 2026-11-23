"""M2 Leak-Free Multi-Step Counterfactual Active-Check Simulation.

Runs apples-to-apples, zero-leakage multi-step active collapse simulations on all
`AMBIGUOUS_WITH_VERIFIED_WITNESSES` real-derived clusters under identical domain-update
and exact-certification rules across strategies:
  1. `OURS_MINIMAX_EXACT`           (exact Minimax remaining ambiguity minimization)
  2. `OURS_MEASURE_SPLIT`           (exact Lebesgue measure-weighted collapse maximization)
  3. `BASELINE_RANDOM`              (20 fixed seeds, uniform random unchecked fracture/action)
  4. `BASELINE_HEURISTIC_SIZE_OR_DIST` (static geometric heuristic: nearest to A-B segment,
                                        tie-break largest area; zero solver peeking)
  5. `BASELINE_HEURISTIC_MAX_AREA`  (supplemental geometric heuristic: largest area / radius)

Zero-leakage guarantee by construction:
Every policy function receives ONLY the current public belief domain `D_k` and public static
model geometry; it executes top-down as an exact belief-domain decision tree, never inspecting
any hidden world `(m*, t*)` or hidden connectivity label. All collapse fractions and expected
steps are computed with exact `fractions.Fraction` Lebesgue measure conservation
(`sum(leaf_measure) == root_measure` asserted on every tree).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import active_identifiability as ai
import solve_real_clusters as solver

ZERO = Q(0)
ONE = Q(1)

PRIMARY_STRATEGIES = (
    "OURS_MINIMAX_EXACT",
    "OURS_MEASURE_SPLIT",
    "BASELINE_RANDOM",
    "BASELINE_HEURISTIC_SIZE_OR_DIST",
)
ALL_STRATEGIES = PRIMARY_STRATEGIES + ("BASELINE_HEURISTIC_MAX_AREA",)
DEFAULT_RANDOM_SEEDS = tuple(range(20))
DEFAULT_MAX_STEPS = 12


# ---------------------------------------------------------------------------
# Pure static geometric heuristics (ZERO solver peeking!)
# ---------------------------------------------------------------------------


def _point_to_segment_distance_3d(
    p: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
    if ab2 <= 0.0:
        return math.sqrt(ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2])
    tau = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2))
    proj = (a[0] + tau * ab[0], a[1] + tau * ab[1], a[2] + tau * ab[2])
    dx, dy, dz = p[0] - proj[0], p[1] - proj[1], p[2] - proj[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _static_fracture_geometry_order(model: Dict[str, Any], variant: str) -> List[str]:
    """Deterministic static fracture ordering using ONLY public fracture_catalog & endpoints."""
    fcat = model.get("fracture_catalog") or {}
    ep_a = [float(solver.q_exact(x)) for x in model["endpoints"]["A"]]
    ep_b = [float(solver.q_exact(x)) for x in model["endpoints"]["B"]]
    rows = []
    for fid in sorted(fcat.keys()):
        meta = fcat[fid]
        center = meta.get("center_local_m") or [0.0, 0.0, 0.0]
        area = float(meta.get("area_m2") or 0.0)
        radius = float(meta.get("char_radius_m") or 0.0)
        dist = _point_to_segment_distance_3d(center, ep_a, ep_b)
        if variant == "BASELINE_HEURISTIC_SIZE_OR_DIST":
            rows.append((dist, -area, -radius, fid))
        elif variant == "BASELINE_HEURISTIC_MAX_AREA":
            rows.append((-area, -radius, dist, fid))
        else:
            raise ValueError(f"unknown heuristic variant: {variant}")
    rows.sort()
    return [r[-1] for r in rows]


# ---------------------------------------------------------------------------
# Leak-free policy decision selectors (input = public belief domain D_k ONLY)
# ---------------------------------------------------------------------------


def _choose_next_step_exact(
    strategy: str,
    model: Dict[str, Any],
    wbm: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    domain_state: Dict[str, Any],
    cur_measure: Q,
    cur_cells: int,
    tri_to_frac: Dict[str, str],
    fracture_ids: Sequence[str],
    used_action_ids: frozenset[str],
    branch_cache: Dict[Any, Dict[str, Any]],
    lattice_index: Optional[solver.CellLatticeIndex] = None,
) -> Optional[Dict[str, Any]]:
    actions_by_frac = ai.enumerate_candidate_actions(
        model, wbm, domain_state, tri_to_frac=tri_to_frac, fracture_ids=fracture_ids
    )
    evals: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for fid in fracture_ids:
        for act in actions_by_frac.get(fid, []):
            if act["action_id"] in used_action_ids:
                continue
            ev = ai.evaluate_action_exact(
                model,
                wbm,
                cur_measure,
                cur_cells,
                act,
                branch_cache=branch_cache,
                lattice_index=lattice_index,
            )
            evals.append((ev, act))
    if not evals:
        return None
    if strategy == "OURS_MINIMAX_EXACT":
        best_ev, best_act = min(evals, key=lambda pair: ai.minimax_action_sort_key(pair[0]))
    elif strategy == "OURS_MEASURE_SPLIT":
        best_ev, best_act = min(evals, key=lambda pair: ai.measure_split_action_sort_key(pair[0]))
    else:
        raise ValueError(f"unsupported exact strategy: {strategy}")
    # Do not waste steps on strictly zero-reduction actions if none reduce anything
    if (
        best_ev["_q_mrf"] == ZERO
        and best_ev["_q_rmf"] == ZERO
        and best_ev["worst_case_remaining_ambiguity_cells"] >= cur_cells
    ):
        return None
    return best_act


def _choose_next_step_heuristic(
    strategy: str,
    model: Dict[str, Any],
    wbm: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    domain_state: Dict[str, Any],
    tri_to_frac: Dict[str, str],
    static_order: Sequence[str],
    used_action_ids: frozenset[str],
    exhausted_fracs: frozenset[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (chosen_action_or_None, inspected_fracture_id_or_None).

    If the top unchecked fracture has no splitting action, returns (None, fid)
    representing a real 1-step wasted field check on that fracture!
    """
    actions_by_frac = ai.enumerate_candidate_actions(
        model, wbm, domain_state, tri_to_frac=tri_to_frac, fracture_ids=static_order
    )
    t_bounds = ai.domain_t_bounds(domain_state)
    t_mid = (t_bounds[0] + t_bounds[1]) / 2 if t_bounds is not None else ZERO

    for fid in static_order:
        if fid in exhausted_fracs:
            continue
        avail = [a for a in actions_by_frac.get(fid, []) if a["action_id"] not in used_action_ids]
        if not avail:
            # Inspecting this fracture yields no observable split; costs 1 step & marks exhausted
            return None, fid
        mode_acts = [a for a in avail if a["action_type"] == "RESOLVE_MODE_FEATURE"]
        if mode_acts:
            return mode_acts[0], fid
        band_acts = [a for a in avail if a["action_type"] == "REFINE_PARAMETER_BAND"]
        if band_acts:
            # Pure geometric midpoint bisection (zero solver peeking!)
            chosen = min(
                band_acts,
                key=lambda a: (abs(solver.q_exact(a["threshold_t"]) - t_mid), a["action_id"]),
            )
            return chosen, fid
        return None, fid
    return None, None


def _choose_next_step_random(
    seed: int,
    cluster_id: str,
    step: int,
    branch_trace: str,
    model: Dict[str, Any],
    wbm: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    domain_state: Dict[str, Any],
    tri_to_frac: Dict[str, str],
    fracture_ids: Sequence[str],
    used_action_ids: frozenset[str],
    exhausted_fracs: frozenset[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    remaining_fracs = [fid for fid in fracture_ids if fid not in exhausted_fracs]
    if not remaining_fracs:
        return None, None
    digest = hashlib.sha256(f"{seed}|{cluster_id}|{step}|{branch_trace}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    fid = rng.choice(remaining_fracs)
    actions_by_frac = ai.enumerate_candidate_actions(
        model, wbm, domain_state, tri_to_frac=tri_to_frac, fracture_ids=[fid]
    )
    avail = [a for a in actions_by_frac.get(fid, []) if a["action_id"] not in used_action_ids]
    if not avail:
        return None, fid
    return rng.choice(avail), fid


# ---------------------------------------------------------------------------
# Exact belief-domain decision-tree simulator (100% full enumeration!)
# ---------------------------------------------------------------------------


def _cell_belongs_to_domain(cell: Dict[str, Any], domain_state: Dict[str, Any]) -> bool:
    mid = cell["mode"]
    segs = domain_state["mode_segments"].get(mid)
    if not segs:
        return False
    t_s = solver.q_exact(cell["sample"])
    return solver.admitted_in_intervals(t_s, segs)


def simulate_strategy_on_cluster(
    model: Dict[str, Any],
    cert: Dict[str, Any],
    wbm: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    strategy: str,
    *,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    branch_cache: Optional[Dict[Any, Dict[str, Any]]] = None,
    lattice_index: Optional[solver.CellLatticeIndex] = None,
) -> Dict[str, Any]:
    """Run exact leak-free belief-tree simulation of one strategy (and optional seed) on one AMBIGUOUS cluster."""
    cid = cert["cluster_id"]
    tri_to_frac = cert["triangle_to_fracture"]
    fracture_ids = list(cert["fracture_ids"])
    root_domain = ai.initial_domain_state(model)
    root_measure = ai.domain_measure(root_domain)
    all_root_cells = cert["cells"]
    n_root_cells = len(all_root_cells)
    cache = branch_cache if branch_cache is not None else {}
    lidx = lattice_index if lattice_index is not None else solver.CellLatticeIndex(all_root_cells)

    static_order = (
        _static_fracture_geometry_order(model, strategy)
        if strategy in ("BASELINE_HEURISTIC_SIZE_OR_DIST", "BASELINE_HEURISTIC_MAX_AREA")
        else fracture_ids
    )

    leaves: List[Dict[str, Any]] = []

    def recurse(
        domain_state: Dict[str, Any],
        step: int,
        used_actions: frozenset[str],
        exhausted_fracs: frozenset[str],
        branch_trace: str,
        tracked_cells: List[Dict[str, Any]],
    ) -> None:
        # Evaluate current domain exact status
        cache_key = tuple((m, tuple(domain_state["mode_segments"][m])) for m in domain_state["allowed_modes"])
        if cache_key in cache:
            res = cache[cache_key]
        else:
            res = lidx.evaluate_subdomain(
                domain_state["allowed_modes"],
                domain_state["mode_segments"],
            )
            cache[cache_key] = res

        st = res["status"]
        cur_meas: Q = res["total_measure"]
        cur_cells: int = res["cells_count"]

        if st in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"):
            leaves.append(
                {
                    "resolved": True,
                    "post_status": st,
                    "steps": step,
                    "measure": cur_meas,
                    "tracked_root_cells": len(tracked_cells),
                    "leaf_cells_count": cur_cells,
                }
            )
            return

        if step >= max_steps:
            leaves.append(
                {
                    "resolved": False,
                    "post_status": st,
                    "steps": step,
                    "measure": cur_meas,
                    "tracked_root_cells": len(tracked_cells),
                    "leaf_cells_count": cur_cells,
                }
            )
            return

        # Choose next action purely from current public belief domain (ZERO hidden-world access!)
        chosen_act: Optional[Dict[str, Any]] = None
        inspected_fid: Optional[str] = None

        if strategy in ("OURS_MINIMAX_EXACT", "OURS_MEASURE_SPLIT"):
            chosen_act = _choose_next_step_exact(
                strategy,
                model,
                wbm,
                domain_state,
                cur_meas,
                cur_cells,
                tri_to_frac,
                fracture_ids,
                used_actions,
                cache,
                lattice_index=lidx,
            )
        elif strategy in ("BASELINE_HEURISTIC_SIZE_OR_DIST", "BASELINE_HEURISTIC_MAX_AREA"):
            chosen_act, inspected_fid = _choose_next_step_heuristic(
                strategy,
                model,
                wbm,
                domain_state,
                tri_to_frac,
                static_order,
                used_actions,
                exhausted_fracs,
            )
        elif strategy == "BASELINE_RANDOM":
            assert seed is not None
            chosen_act, inspected_fid = _choose_next_step_random(
                seed,
                cid,
                step,
                branch_trace,
                model,
                wbm,
                domain_state,
                tri_to_frac,
                fracture_ids,
                used_actions,
                exhausted_fracs,
            )
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        if chosen_act is None:
            if inspected_fid is not None and inspected_fid not in exhausted_fracs:
                # Wasted 1 field check on a non-discriminating fracture -> continue at step + 1
                recurse(
                    domain_state,
                    step + 1,
                    used_actions,
                    exhausted_fracs | {inspected_fid},
                    f"{branch_trace}|noop:{inspected_fid}",
                    tracked_cells,
                )
                return
            # No actions or fractures left -> unresolved leaf
            leaves.append(
                {
                    "resolved": False,
                    "post_status": st,
                    "steps": step,
                    "measure": cur_meas,
                    "tracked_root_cells": len(tracked_cells),
                    "leaf_cells_count": cur_cells,
                }
            )
            return

        next_used = used_actions | {chosen_act["action_id"]}
        # Recurse over all mutually exclusive observable outcome branches
        assigned_root_cells = 0
        for br in chosen_act["branches"]:
            child_dom = {
                "allowed_modes": list(br["allowed_modes"]),
                "mode_segments": br["mode_segments"],
            }
            child_tracked = [c for c in tracked_cells if _cell_belongs_to_domain(c, child_dom)]
            assigned_root_cells += len(child_tracked)
            recurse(
                child_dom,
                step + 1,
                next_used,
                exhausted_fracs,
                f"{branch_trace}|{chosen_act['action_id']}={br['branch_id']}",
                child_tracked,
            )
        if assigned_root_cells != len(tracked_cells):
            raise RuntimeError(
                f"partition cell-conservation error on {cid} ({strategy}): "
                f"assigned={assigned_root_cells} != parent={len(tracked_cells)}"
            )

    recurse(root_domain, 0, frozenset(), frozenset(), "root", list(all_root_cells))

    # Strict exact conservation checks!
    sum_meas = sum((lf["measure"] for lf in leaves), ZERO)
    sum_cells = sum(lf["tracked_root_cells"] for lf in leaves)
    if sum_meas != root_measure:
        raise RuntimeError(f"measure conservation violated on {cid}/{strategy}: {sum_meas} != {root_measure}")
    if sum_cells != n_root_cells:
        raise RuntimeError(f"cell conservation violated on {cid}/{strategy}: {sum_cells} != {n_root_cells}")

    # Compute exact collapse curve by step k = 0..max_steps
    curve_meas: Dict[str, str] = {}
    curve_meas_flt: Dict[str, float] = {}
    curve_cells: Dict[str, str] = {}
    curve_cells_flt: Dict[str, float] = {}

    for k in range(0, max_steps + 1):
        m_k = sum((lf["measure"] for lf in leaves if lf["resolved"] and lf["steps"] <= k), ZERO)
        c_k = sum(lf["tracked_root_cells"] for lf in leaves if lf["resolved"] and lf["steps"] <= k)
        frac_m = (m_k / root_measure) if root_measure > ZERO else Q(c_k, n_root_cells)
        frac_c = Q(c_k, n_root_cells)
        curve_meas[str(k)] = solver.qs(frac_m)
        curve_meas_flt[str(k)] = round(float(frac_m), 6)
        curve_cells[str(k)] = solver.qs(frac_c)
        curve_cells_flt[str(k)] = round(float(frac_c), 6)

    resolved_meas = sum((lf["measure"] for lf in leaves if lf["resolved"]), ZERO)
    unresolved_meas = root_measure - resolved_meas
    resolved_cells = sum(lf["tracked_root_cells"] for lf in leaves if lf["resolved"])

    # Exact expected steps (note: unresolved leaves are capped at max_steps in censored mean,
    # and exact uncensored mean is reported when 100% resolved!)
    all_meas_resolved = unresolved_meas == ZERO
    all_cells_resolved = resolved_cells == n_root_cells
    all_leaves_resolved = all(lf["resolved"] for lf in leaves)

    mean_steps_meas_q = (
        sum((lf["measure"] * lf["steps"] for lf in leaves), ZERO) / root_measure
        if root_measure > ZERO
        else Q(sum(lf["tracked_root_cells"] * lf["steps"] for lf in leaves), n_root_cells)
    )
    mean_steps_cells_q = Q(sum(lf["tracked_root_cells"] * lf["steps"] for lf in leaves), n_root_cells)
    worst_case_steps = max(lf["steps"] for lf in leaves) if all_leaves_resolved else None

    return {
        "cluster_id": cid,
        "strategy": strategy,
        "seed": seed,
        "max_steps_cap": max_steps,
        "root_measure": solver.qs(root_measure),
        "root_cells_count": n_root_cells,
        "leaf_count": len(leaves),
        "all_measure_resolved": all_meas_resolved,
        "all_cells_resolved": all_cells_resolved,
        "all_leaves_resolved": all_leaves_resolved,
        "unresolved_measure_fraction": solver.qs(unresolved_meas / root_measure) if root_measure > ZERO else "0/1",
        "mean_steps_measure_exact": solver.qs(mean_steps_meas_q),
        "mean_steps_measure_float": round(float(mean_steps_meas_q), 6),
        "mean_steps_cells_exact": solver.qs(mean_steps_cells_q),
        "mean_steps_cells_float": round(float(mean_steps_cells_q), 6),
        "worst_case_steps": worst_case_steps,
        "collapse_curve_measure_exact": curve_meas,
        "collapse_curve_measure_float": curve_meas_flt,
        "collapse_curve_cells_exact": curve_cells,
        "collapse_curve_cells_float": curve_cells_flt,
    }


def _average_seed_runs(seed_runs: Sequence[Dict[str, Any]], max_steps: int) -> Dict[str, Any]:
    """Exact rational average across fixed random seeds for BASELINE_RANDOM."""
    n = len(seed_runs)
    assert n > 0
    curve_m: Dict[str, str] = {}
    curve_m_flt: Dict[str, float] = {}
    curve_c: Dict[str, str] = {}
    curve_c_flt: Dict[str, float] = {}
    for k in range(0, max_steps + 1):
        ks = str(k)
        avg_m = sum((solver.q_exact(r["collapse_curve_measure_exact"][ks]) for r in seed_runs), ZERO) / n
        avg_c = sum((solver.q_exact(r["collapse_curve_cells_exact"][ks]) for r in seed_runs), ZERO) / n
        curve_m[ks] = solver.qs(avg_m)
        curve_m_flt[ks] = round(float(avg_m), 6)
        curve_c[ks] = solver.qs(avg_c)
        curve_c_flt[ks] = round(float(avg_c), 6)

    mean_m = sum((solver.q_exact(r["mean_steps_measure_exact"]) for r in seed_runs), ZERO) / n
    mean_c = sum((solver.q_exact(r["mean_steps_cells_exact"]) for r in seed_runs), ZERO) / n
    wc_vals = [r["worst_case_steps"] for r in seed_runs if r["worst_case_steps"] is not None]
    return {
        "cluster_id": seed_runs[0]["cluster_id"],
        "strategy": "BASELINE_RANDOM",
        "seeds": [r["seed"] for r in seed_runs],
        "seed_count": n,
        "max_steps_cap": max_steps,
        "all_seeds_measure_resolved": all(r["all_measure_resolved"] for r in seed_runs),
        "all_seeds_leaves_resolved": all(r["all_leaves_resolved"] for r in seed_runs),
        "mean_steps_measure_exact": solver.qs(mean_m),
        "mean_steps_measure_float": round(float(mean_m), 6),
        "mean_steps_cells_exact": solver.qs(mean_c),
        "mean_steps_cells_float": round(float(mean_c), 6),
        "worst_case_steps_max_across_seeds": max(wc_vals) if len(wc_vals) == n else None,
        "worst_case_steps_mean_across_seeds": round(sum(wc_vals) / len(wc_vals), 4) if wc_vals else None,
        "collapse_curve_measure_exact": curve_m,
        "collapse_curve_measure_float": curve_m_flt,
        "collapse_curve_cells_exact": curve_c,
        "collapse_curve_cells_float": curve_c_flt,
    }


def _aggregate_across_clusters(
    cluster_rows: Sequence[Dict[str, Any]],
    strategy: str,
    max_steps: int,
) -> Optional[Dict[str, Any]]:
    if not cluster_rows:
        return None
    n = len(cluster_rows)
    curve_m: Dict[str, str] = {}
    curve_m_flt: Dict[str, float] = {}
    curve_c: Dict[str, str] = {}
    curve_c_flt: Dict[str, float] = {}
    for k in range(0, max_steps + 1):
        ks = str(k)
        avg_m = sum((solver.q_exact(r["collapse_curve_measure_exact"][ks]) for r in cluster_rows), ZERO) / n
        avg_c = sum((solver.q_exact(r["collapse_curve_cells_exact"][ks]) for r in cluster_rows), ZERO) / n
        curve_m[ks] = solver.qs(avg_m)
        curve_m_flt[ks] = round(float(avg_m), 6)
        curve_c[ks] = solver.qs(avg_c)
        curve_c_flt[ks] = round(float(avg_c), 6)

    mean_m = sum((solver.q_exact(r["mean_steps_measure_exact"]) for r in cluster_rows), ZERO) / n
    mean_c = sum((solver.q_exact(r["mean_steps_cells_exact"]) for r in cluster_rows), ZERO) / n
    return {
        "strategy": strategy,
        "ambiguous_cluster_count": n,
        "mean_steps_measure_exact": solver.qs(mean_m),
        "mean_steps_measure_float": round(float(mean_m), 6),
        "mean_steps_cells_exact": solver.qs(mean_c),
        "mean_steps_cells_float": round(float(mean_c), 6),
        "step1_collapse_rate_measure": curve_m_flt["1"],
        "step2_collapse_rate_measure": curve_m_flt["2"],
        "step3_collapse_rate_measure": curve_m_flt["3"],
        "step1_collapse_rate_cells": curve_c_flt["1"],
        "step2_collapse_rate_cells": curve_c_flt["2"],
        "step3_collapse_rate_cells": curve_c_flt["3"],
        "collapse_curve_measure_exact": curve_m,
        "collapse_curve_measure_float": curve_m_flt,
        "collapse_curve_cells_exact": curve_c,
        "collapse_curve_cells_float": curve_c_flt,
    }


def run_simulation_batch(
    register_path: Path,
    run_dir: Path,
    *,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    reg = solver.load_strict_json(register_path)
    reg_dir = register_path.parent
    cases_dir = run_dir / "cases"
    t_all = time.perf_counter()

    per_cluster: Dict[str, Dict[str, Any]] = {}
    by_strategy_all: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ALL_STRATEGIES}
    by_site_strategy: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    ambiguous_ids: List[str] = []

    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        sid = entry["site_id"]
        cert = solver.load_strict_json(cases_dir / cid / "certification.json")
        if cert["status"] != "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
            continue
        ambiguous_ids.append(cid)
        m_raw = solver.load_strict_json(reg_dir / entry["model_path"])
        wbm = ai.windows_by_mode_from_cert(cert)
        branch_cache: Dict[Any, Dict[str, Any]] = {}
        lidx = solver.CellLatticeIndex(cert["cells"])
        per_cluster[cid] = {"site_id": sid, "stratum": entry["stratum"], "strategies": {}}
        by_site_strategy.setdefault(sid, {s: [] for s in ALL_STRATEGIES})

        for strat in ALL_STRATEGIES:
            if strat == "BASELINE_RANDOM":
                seed_runs = [
                    simulate_strategy_on_cluster(
                        m_raw,
                        cert,
                        wbm,
                        strat,
                        seed=sd,
                        max_steps=max_steps,
                        branch_cache=branch_cache,
                        lattice_index=lidx,
                    )
                    for sd in random_seeds
                ]
                row = _average_seed_runs(seed_runs, max_steps)
            else:
                row = simulate_strategy_on_cluster(
                    m_raw,
                    cert,
                    wbm,
                    strat,
                    seed=None,
                    max_steps=max_steps,
                    branch_cache=branch_cache,
                    lattice_index=lidx,
                )
            per_cluster[cid]["strategies"][strat] = row
            by_strategy_all[strat].append(row)
            by_site_strategy[sid][strat].append(row)

        print(
            f"[M2-SIM] {cid:12s} | "
            f"MINIMAX={per_cluster[cid]['strategies']['OURS_MINIMAX_EXACT']['mean_steps_measure_float']:.3f} | "
            f"MSPLIT={per_cluster[cid]['strategies']['OURS_MEASURE_SPLIT']['mean_steps_measure_float']:.3f} | "
            f"RAND={per_cluster[cid]['strategies']['BASELINE_RANDOM']['mean_steps_measure_float']:.3f} | "
            f"HEUR_DIST={per_cluster[cid]['strategies']['BASELINE_HEURISTIC_SIZE_OR_DIST']['mean_steps_measure_float']:.3f} | "
            f"HEUR_AREA={per_cluster[cid]['strategies']['BASELINE_HEURISTIC_MAX_AREA']['mean_steps_measure_float']:.3f}",
            flush=True,
        )

    agg_all = {
        strat: _aggregate_across_clusters(by_strategy_all[strat], strat, max_steps)
        for strat in ALL_STRATEGIES
    }
    agg_by_site = {
        sid: {
            strat: _aggregate_across_clusters(by_site_strategy[sid][strat], strat, max_steps)
            for strat in ALL_STRATEGIES
        }
        for sid in sorted(by_site_strategy.keys())
    }

    return {
        "schema": "m2-active-simulation/v1",
        "zero_leakage_protocol": "belief-domain decision tree; policy receives only public domain D_k and static geometry, never hidden world or connectivity label",
        "max_steps_cap": max_steps,
        "random_seeds": list(random_seeds),
        "ambiguous_clusters_simulated": ambiguous_ids,
        "ambiguous_cluster_count": len(ambiguous_ids),
        "aggregate_all_ambiguous": agg_all,
        "aggregate_by_site": agg_by_site,
        "per_cluster": per_cluster,
        "simulation_seconds": round(time.perf_counter() - t_all, 3),
    }


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2 Leak-Free Multi-Step Active Collapse Simulator")
    parser.add_argument("--register", required=True, help="Path to M1 cluster_register.json")
    parser.add_argument("--run-dir", required=True, help="Run directory containing cases/<cluster_id>/certification.json")
    parser.add_argument("--out", help="Optional output JSON path (also embedded into summary.json by pipeline)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)

    sim = run_simulation_batch(Path(args.register), Path(args.run_dir), max_steps=args.max_steps)
    if args.out:
        solver.write_json(Path(args.out), sim)
    print(
        json.dumps(
            {
                "ambiguous_cluster_count": sim["ambiguous_cluster_count"],
                "aggregate_mean_steps_measure": {
                    k: (v["mean_steps_measure_float"] if v else None)
                    for k, v in sim["aggregate_all_ambiguous"].items()
                },
                "simulation_seconds": sim["simulation_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
