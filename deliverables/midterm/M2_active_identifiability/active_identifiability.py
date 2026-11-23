"""M2 Active Identifiability ("Next-Best-Check") Exact Ranker & Collapse Tree Builder.

Zero-leakage, exact-rational active check prioritization for real-derived conditioned
fracture clusters (`AMBIGUOUS_WITH_VERIFIED_WITNESSES`):
- Operates strictly on observable, model-declared checks per fracture:
  1. `RESOLVE_MODE_FEATURE(fracture_id)`: partitions active modes by `mode_semantics[m].feature_assignments[fid]`.
  2. `REFINE_PARAMETER_BAND(fracture_id)`: partitions the feasible `t` domain at exact rational
     fracture-pair contact boundary knots (`t < t*` vs `t >= t*` for contact onset, `t <= t*` vs
     `t > t*` for contact offset).
- For every candidate action branch `o`, genuinely invokes the exact whole-network solver on the
  conditioned subdomain `D(o)` (exact `Fraction` arithmetic, zero float final decisions).
- Computes exact, prior-free Minimax metrics (`worst_case_remaining_ambiguity_measure`,
  `minimax_reduction_fraction`) and exact Lebesgue measure-weighted metrics
  (`resolved_measure_fraction`, `certified_branch_fraction`, `expected_remaining_ambiguity_measure`).
- Produces human-readable `witness_diff_role` geometric attribution and a self-contained,
  M3-verifiable, M5-ready `collapse_tree` (depth 1 for all informative fractures, depth 2
  follow-up for Top-3 ambiguous branches).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import solve_real_clusters as solver

ZERO = Q(0)
ONE = Q(1)

ACTIVE_SCHEMA = "m2-active-identifiability/v1"
TREE_SCHEMA = "m2-collapse-tree/v1"

ROLE_PRIORITY = {
    "DISTINGUISHES_WITNESS_MODES_AND_CUT_BOUNDARY": 0,
    "IN_CONNECTED_PATH_AND_CUT_BOUNDARY": 1,
    "DISTINGUISHES_WITNESS_MODES": 2,
    "ON_CONNECTED_WITNESS_PATH": 3,
    "MODE_BRANCH_CANDIDATE": 4,
    "SHARED_T_CONTACT_PROXY": 5,
    "NON_CRITICAL_INVARIANT_FRACTURE": 6,
    "ISOLATED_IRRELEVANT_FRACTURE": 7,
}


def windows_by_mode_from_cert(cert: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]]:
    out: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]] = {}
    for w in cert.get("contact_windows", []):
        mid = w["mode"]
        iv = None if w["t_closed"] is None else (solver.q_exact(w["t_closed"][0]), solver.q_exact(w["t_closed"][1]))
        out.setdefault(mid, []).append((w["from"], w["to"], iv))
    return out


def initial_domain_state(model: Dict[str, Any]) -> Dict[str, Any]:
    allowed_modes: List[str] = []
    mode_segments: Dict[str, List[Tuple[Q, Q, bool, bool]]] = {}
    for m in model["modes"]:
        dom = (solver.q_exact(m["t_closed"][0]), solver.q_exact(m["t_closed"][1]))
        segs = solver.normalize_segments(m["segments"], dom)
        if segs:
            allowed_modes.append(m["id"])
            mode_segments[m["id"]] = segs
    return {"allowed_modes": sorted(allowed_modes), "mode_segments": mode_segments}


def domain_measure(domain_state: Dict[str, Any]) -> Q:
    return sum((solver.segments_measure(segs) for segs in domain_state["mode_segments"].values()), ZERO)


def domain_t_bounds(domain_state: Dict[str, Any]) -> Optional[Tuple[Q, Q]]:
    los, his = [], []
    for segs in domain_state["mode_segments"].values():
        for lo, hi, _, _ in segs:
            los.append(lo)
            his.append(hi)
    if not los:
        return None
    return (min(los), max(his))


# ---------------------------------------------------------------------------
# Witness Divergence Role Attribution (purely from public model witnesses)
# ---------------------------------------------------------------------------


def compute_witness_diff_roles(
    model: Dict[str, Any],
    cert: Dict[str, Any],
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
) -> Dict[str, Dict[str, Any]]:
    """Compute geometric role of each fracture relative to the dual witnesses and mode semantics."""
    tri_to_frac = cert["triangle_to_fracture"]
    fracture_ids = list(cert["fracture_ids"])
    cw = cert.get("connected_witness")
    dw = cert.get("disconnected_witness")
    msem = model.get("mode_semantics") or {}
    fcat = model.get("fracture_catalog") or {}

    conn_fracs = set(cw["fracture_path"]) if cw and cw.get("fracture_path") else set()
    s_tris = set(dw["cut"]["S"]) if dw and dw.get("cut") else set()
    t_tris = set(dw["cut"]["T"]) if dw and dw.get("cut") else set()

    # Which fractures sit directly on an edge of connected_witness.path crossing S <-> T of disconnected_witness?
    cut_crossing_fracs: set[str] = set()
    cut_crossing_partners: Dict[str, set[str]] = {}
    if cw and cw.get("path") and dw and dw.get("cut"):
        for a, b in zip(cw["path"], cw["path"][1:]):
            if a.startswith("@") or b.startswith("@"):
                continue
            crosses = (a in s_tris and b in t_tris) or (b in s_tris and a in t_tris)
            if crosses:
                fa, fb = tri_to_frac[a], tri_to_frac[b]
                cut_crossing_fracs.add(fa)
                cut_crossing_fracs.add(fb)
                cut_crossing_partners.setdefault(fa, set()).add(fb)
                cut_crossing_partners.setdefault(fb, set()).add(fa)

    # Check mode feature variation and witness mode separation
    active_modes = [m["id"] for m in model["modes"] if m["segments"]]
    m_plus = cw["mode"] if cw else None
    m_minus = dw["mode"] if dw else None

    # Check contact presence across all modes (for isolated-fracture detection)
    has_any_external_contact: Dict[str, bool] = {fid: False for fid in fracture_ids}
    has_any_interior_transition: Dict[str, bool] = {fid: False for fid in fracture_ids}
    dom_by_mode = {m["id"]: (solver.q_exact(m["t_closed"][0]), solver.q_exact(m["t_closed"][1])) for m in model["modes"]}

    for mid in active_modes:
        dom_lo, dom_hi = dom_by_mode[mid]
        pair_ivs: Dict[Tuple[str, str], List[Tuple[Q, Q]]] = {}
        for a, b, iv in windows_by_mode.get(mid, []):
            if iv is None:
                continue
            fa = a if a.startswith("@") else tri_to_frac[a]
            fb = b if b.startswith("@") else tri_to_frac[b]
            if fa == fb:
                continue
            if not fa.startswith("@"):
                has_any_external_contact[fa] = True
                pair_ivs.setdefault((fa, fb), []).append(iv)
            if not fb.startswith("@"):
                has_any_external_contact[fb] = True
                pair_ivs.setdefault((fb, fa), []).append(iv)
        for (fid, _other), ivs in pair_ivs.items():
            for u, v in solver.merge_closed_intervals(ivs):
                if dom_lo < u < dom_hi or dom_lo < v < dom_hi:
                    has_any_interior_transition[fid] = True

    roles: Dict[str, Dict[str, Any]] = {}
    for fid in fracture_ids:
        feat_vals = sorted(
            {
                msem[mid]["feature_assignments"].get(fid)
                for mid in active_modes
                if mid in msem and fid in msem[mid].get("feature_assignments", {})
            }
        )
        has_mode_feat = len(feat_vals) >= 2
        val_plus = msem.get(m_plus, {}).get("feature_assignments", {}).get(fid) if m_plus else None
        val_minus = msem.get(m_minus, {}).get("feature_assignments", {}).get(fid) if m_minus else None
        distinguishes_modes = bool(has_mode_feat and val_plus != val_minus)
        in_path = fid in conn_fracs
        crosses_cut = fid in cut_crossing_fracs
        isolated = (not has_any_external_contact[fid]) and (not has_mode_feat)
        has_trans = has_any_interior_transition[fid]

        if distinguishes_modes and crosses_cut:
            tag = "DISTINGUISHES_WITNESS_MODES_AND_CUT_BOUNDARY"
            zh = f"该裂隙既位于连通见证路径与断割见证 S/T 边界交界处，又直接区分连通模式({val_plus})与断割模式({val_minus})。"
        elif in_path and crosses_cut:
            tag = "IN_CONNECTED_PATH_AND_CUT_BOUNDARY"
            partners = sorted(cut_crossing_partners.get(fid, set()))
            zh = f"该裂隙位于连通见证路径上并直接跨越断割见证 S/T 临界断口（对侧邻片：{', '.join(partners)}）。"
        elif distinguishes_modes:
            tag = "DISTINGUISHES_WITNESS_MODES"
            zh = f"该裂隙的离散分支特征直接区分连通见证模式({val_plus})与断割见证模式({val_minus})。"
        elif in_path:
            tag = "ON_CONNECTED_WITNESS_PATH"
            zh = "该裂隙位于连通见证路径上，局部接触或分支复核可直接收窄贯通条件域。"
        elif has_mode_feat:
            tag = "MODE_BRANCH_CANDIDATE"
            zh = f"该裂隙携带显式离散分支特征（{'/'.join(feat_vals)}），复核后可消除模式级歧义。"
        elif has_trans:
            tag = "SHARED_T_CONTACT_PROXY"
            zh = "该裂隙虽不直接跨名义断口，但其与邻片的接触边界节点在共享参数 t 模型下可间接约束 t 区间。"
        elif isolated:
            tag = "ISOLATED_IRRELEVANT_FRACTURE"
            zh = "孤立无关裂隙：在全部合法模式下既无模式分支特征也无外部接触，复核收益严格为 0。"
        else:
            tag = "NON_CRITICAL_INVARIANT_FRACTURE"
            zh = "非临界不变裂隙：在当前合法域内既无模式分支也无内部接触切换事件，复核收益为 0。"

        roles[fid] = {
            "witness_diff_role": tag,
            "geometric_attribution_zh": zh,
            "in_connected_witness_path": in_path,
            "crosses_disconnected_cut_boundary": crosses_cut,
            "cut_crossing_partners": sorted(cut_crossing_partners.get(fid, set())),
            "distinguishes_witness_modes": distinguishes_modes,
            "has_mode_feature": has_mode_feat,
            "mode_feature_values": feat_vals,
            "has_external_contact": has_any_external_contact[fid],
            "has_interior_contact_transition": has_trans,
            "catalog_role": fcat.get(fid, {}).get("role"),
        }
    return roles


# ---------------------------------------------------------------------------
# Candidate observable action generation (strictly leak-free!)
# ---------------------------------------------------------------------------


def enumerate_candidate_actions(
    model: Dict[str, Any],
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    domain_state: Dict[str, Any],
    *,
    tri_to_frac: Optional[Dict[str, str]] = None,
    fracture_ids: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Enumerate all strictly observable, domain-partitioning actions per fracture.

    NEVER inspects connectivity labels; depends ONLY on declared `mode_semantics`
    and pairwise fracture contact geometry within `domain_state`.
    """
    t2f = tri_to_frac if tri_to_frac is not None else solver._build_tri_to_fracture(model)
    fids = list(fracture_ids) if fracture_ids is not None else (
        sorted(model["fracture_catalog"].keys()) if model.get("fracture_catalog") else sorted(set(t2f.values()))
    )
    active_modes = [m for m in domain_state["allowed_modes"] if domain_state["mode_segments"].get(m)]
    msem = model.get("mode_semantics") or {}
    actions_by_frac: Dict[str, List[Dict[str, Any]]] = {fid: [] for fid in fids}

    # Pre-collect merged fracture-pair contact intervals per active mode
    pair_merged_by_mode: Dict[str, Dict[Tuple[str, str], List[Tuple[Q, Q]]]] = {}
    for mid in active_modes:
        raw_pairs: Dict[Tuple[str, str], List[Tuple[Q, Q]]] = {}
        for a, b, iv in windows_by_mode.get(mid, []):
            if iv is None:
                continue
            fa = a if a.startswith("@") else t2f[a]
            fb = b if b.startswith("@") else t2f[b]
            if fa == fb:
                continue
            if not fa.startswith("@"):
                raw_pairs.setdefault((fa, fb), []).append(iv)
            if not fb.startswith("@"):
                raw_pairs.setdefault((fb, fa), []).append(iv)
        pair_merged_by_mode[mid] = {k: solver.merge_closed_intervals(v) for k, v in raw_pairs.items()}

    for fid in fids:
        # 1. RESOLVE_MODE_FEATURE(fid)
        groups: Dict[str, List[str]] = {}
        for mid in active_modes:
            val = msem.get(mid, {}).get("feature_assignments", {}).get(fid)
            if val is not None:
                groups.setdefault(str(val), []).append(mid)
        # Only valid if EVERY active mode has an assignment for fid AND >=2 distinct values exist
        covered_modes = sum(len(ms) for ms in groups.values())
        if len(groups) >= 2 and covered_modes == len(active_modes):
            branches = []
            for val in sorted(groups.keys()):
                b_modes = sorted(groups[val])
                b_segs = {m: list(domain_state["mode_segments"][m]) for m in b_modes}
                branches.append(
                    {
                        "branch_id": f"mode_{val}",
                        "label_zh": f"确认裂隙分支为 {val}",
                        "label_en": f"Confirm branch {val}",
                        "feature_value": val,
                        "allowed_modes": b_modes,
                        "mode_segments": b_segs,
                        "t_predicate": {"kind": "ALL_MODE_SEGMENTS", "threshold_t": None},
                        "feature_constraint": {"fracture_id": fid, "feature_value": val},
                    }
                )
            actions_by_frac[fid].append(
                {
                    "action_id": f"MODE::{fid}",
                    "action_type": "RESOLVE_MODE_FEATURE",
                    "fracture_id": fid,
                    "action_summary_zh": f"核查裂隙 {fid} 的边界延展/产状/归属分支（{' / '.join(sorted(groups))}）",
                    "action_summary_en": f"Verify discrete mode feature of {fid} ({', '.join(sorted(groups))})",
                    "threshold_t": None,
                    "split_mode": None,
                    "trigger_partners": [],
                    "branches": branches,
                }
            )

        # 2. REFINE_PARAMETER_BAND(fid) from fracture-pair contact boundary knots
        knot_meta: Dict[Tuple[Q, str], List[Dict[str, str]]] = {}
        for mid in active_modes:
            for (fa, other), ivs in pair_merged_by_mode[mid].items():
                if fa != fid:
                    continue
                for u, v in ivs:
                    knot_meta.setdefault((u, "LT_GE"), []).append(
                        {"mode": mid, "partner": other, "event": "CONTACT_ONSET"}
                    )
                    knot_meta.setdefault((v, "LE_GT"), []).append(
                        {"mode": mid, "partner": other, "event": "CONTACT_OFFSET"}
                    )

        for (t_star, split_mode), meta_list in sorted(knot_meta.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if split_mode == "LT_GE":
                left_kind, right_kind = "LT_THRESHOLD", "GE_THRESHOLD"
                left_sym, right_sym = "<", ">="
                left_phys, right_phys = "未接触 / 断开侧", "接触成立侧"
            else:
                left_kind, right_kind = "LE_THRESHOLD", "GT_THRESHOLD"
                left_sym, right_sym = "<=", ">"
                left_phys, right_phys = "接触成立侧", "未接触 / 断开侧"

            left_segs: Dict[str, List[Tuple[Q, Q, bool, bool]]] = {}
            right_segs: Dict[str, List[Tuple[Q, Q, bool, bool]]] = {}
            for mid in active_modes:
                cur_segs = domain_state["mode_segments"][mid]
                ls = solver.intersect_segments_halfline(cur_segs, left_kind, t_star)
                rs = solver.intersect_segments_halfline(cur_segs, right_kind, t_star)
                if ls:
                    left_segs[mid] = ls
                if rs:
                    right_segs[mid] = rs

            if not left_segs or not right_segs:
                continue  # Does not genuinely split current domain

            partners = sorted({row["partner"] for row in meta_list})
            t_str = solver.qs(t_star)
            t_flt = round(float(t_star), 5)
            branches = [
                {
                    "branch_id": "band_left",
                    "label_zh": f"现场复核确认 t {left_sym} {t_flt}（{left_phys}）",
                    "label_en": f"Confirm t {left_sym} {t_flt}",
                    "feature_value": None,
                    "allowed_modes": sorted(left_segs.keys()),
                    "mode_segments": left_segs,
                    "t_predicate": {"kind": left_kind, "threshold_t": t_str},
                    "feature_constraint": None,
                },
                {
                    "branch_id": "band_right",
                    "label_zh": f"现场复核确认 t {right_sym} {t_flt}（{right_phys}）",
                    "label_en": f"Confirm t {right_sym} {t_flt}",
                    "feature_value": None,
                    "allowed_modes": sorted(right_segs.keys()),
                    "mode_segments": right_segs,
                    "t_predicate": {"kind": right_kind, "threshold_t": t_str},
                    "feature_constraint": None,
                },
            ]
            actions_by_frac[fid].append(
                {
                    "action_id": f"BAND::{fid}::{split_mode}::{t_str}",
                    "action_type": "REFINE_PARAMETER_BAND",
                    "fracture_id": fid,
                    "action_summary_zh": (
                        f"局部复核裂隙 {fid} 与 {', '.join(partners)} 的接触临界点 t*={t_flt}"
                        f"（区分 t {left_sym} t* 与 t {right_sym} t*）"
                    ),
                    "action_summary_en": (
                        f"Verify contact threshold t*={t_flt} of {fid} with {', '.join(partners)} "
                        f"(t {left_sym} t* vs t {right_sym} t*)"
                    ),
                    "threshold_t": t_str,
                    "threshold_t_float": t_flt,
                    "split_mode": split_mode,
                    "trigger_partners": partners,
                    "branches": branches,
                }
            )

    return actions_by_frac


# ---------------------------------------------------------------------------
# Exact Action Evaluation & Scoring (Zero Fake Priors, Exact Fractions!)
# ---------------------------------------------------------------------------


def evaluate_action_exact(
    model: Dict[str, Any],
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    parent_total_measure: Q,
    parent_total_cells: int,
    action: Dict[str, Any],
    *,
    branch_cache: Optional[Dict[Any, Dict[str, Any]]] = None,
    lattice_index: Optional[solver.CellLatticeIndex] = None,
) -> Dict[str, Any]:
    """Evaluate a candidate action by genuinely invoking exact solver on every branch D(o)."""
    branch_evals: List[Dict[str, Any]] = []
    n_branches = len(action["branches"])
    n_cert = 0
    resolved_meas = ZERO
    resolved_cells = 0
    worst_rem_meas = ZERO
    worst_rem_cells = 0
    worst_rem_modes = 0
    exp_rem_meas_num = ZERO  # sum_i meas_i * rem_meas_i
    exp_rem_cells_num = ZERO  # sum_i cells_i * rem_cells_i

    for br in action["branches"]:
        cache_key = tuple(
            (m, tuple(br["mode_segments"][m]))
            for m in br["allowed_modes"]
        )
        if branch_cache is not None and cache_key in branch_cache:
            res = branch_cache[cache_key]
        elif lattice_index is not None:
            res = lattice_index.evaluate_subdomain(br["allowed_modes"], br["mode_segments"])
            if branch_cache is not None:
                branch_cache[cache_key] = res
        else:
            res = solver.solve_conditioned_subdomain(
                model,
                windows_by_mode,
                br["allowed_modes"],
                br["mode_segments"],
                include_witnesses=False,
            )
            if branch_cache is not None:
                branch_cache[cache_key] = res

        st = res["status"]
        is_cert = st in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")
        b_meas: Q = res["total_measure"]
        b_cells: int = res["cells_count"]
        b_modes: int = len(res["active_modes"])

        rem_meas = ZERO if is_cert else b_meas
        rem_cells = 0 if is_cert else b_cells
        rem_modes = 0 if is_cert else b_modes

        if is_cert:
            n_cert += 1
            resolved_meas += b_meas
            resolved_cells += b_cells
        if rem_meas > worst_rem_meas:
            worst_rem_meas = rem_meas
        if rem_cells > worst_rem_cells:
            worst_rem_cells = rem_cells
        if rem_modes > worst_rem_modes:
            worst_rem_modes = rem_modes
        exp_rem_meas_num += b_meas * rem_meas
        exp_rem_cells_num += Q(b_cells * rem_cells, 1)

        branch_evals.append(
            {
                "branch_id": br["branch_id"],
                "label_zh": br["label_zh"],
                "label_en": br["label_en"],
                "condition": {
                    "allowed_modes": list(br["allowed_modes"]),
                    "mode_segments": {
                        m: solver.segments_to_json(br["mode_segments"][m])
                        for m in br["allowed_modes"]
                    },
                    "t_predicate": br["t_predicate"],
                    "feature_constraint": br["feature_constraint"],
                },
                "post_status": st,
                "collapsed_to_certified": is_cert,
                "exists_connected": res["exists_connected"],
                "all_connected": res["all_connected"],
                "branch_measure": solver.qs(b_meas),
                "branch_measure_float": round(float(b_meas), 8),
                "branch_measure_fraction": (
                    solver.qs(b_meas / parent_total_measure)
                    if parent_total_measure > ZERO
                    else solver.qs(Q(b_cells, parent_total_cells))
                ),
                "branch_cells_count": b_cells,
                "branch_modes_count": b_modes,
                "remaining_ambiguity_measure": solver.qs(rem_meas),
                "remaining_ambiguity_cells": rem_cells,
                "remaining_ambiguity_modes": rem_modes,
                "_raw_allowed_modes": br["allowed_modes"],
                "_raw_mode_segments": br["mode_segments"],
            }
        )

    cbf = Q(n_cert, n_branches)
    if parent_total_measure > ZERO:
        rmf = resolved_meas / parent_total_measure
        mrf = ONE - (worst_rem_meas / parent_total_measure)
        exp_rem = exp_rem_meas_num / parent_total_measure
        exp_red_frac = ONE - (exp_rem / parent_total_measure)
    else:
        rmf = Q(resolved_cells, parent_total_cells)
        mrf = ONE - Q(worst_rem_cells, parent_total_cells)
        exp_rem = exp_rem_cells_num / Q(parent_total_cells, 1)
        exp_red_frac = ONE - (exp_rem / Q(parent_total_cells, 1))

    return {
        "action_id": action["action_id"],
        "action_type": action["action_type"],
        "fracture_id": action["fracture_id"],
        "action_summary_zh": action["action_summary_zh"],
        "action_summary_en": action["action_summary_en"],
        "threshold_t": action.get("threshold_t"),
        "split_mode": action.get("split_mode"),
        "trigger_partners": action.get("trigger_partners", []),
        "branch_count": n_branches,
        "certified_branch_count": n_cert,
        "certified_branch_fraction": solver.qs(cbf),
        "certified_branch_fraction_float": round(float(cbf), 8),
        "resolved_measure_fraction": solver.qs(rmf),
        "resolved_measure_fraction_float": round(float(rmf), 8),
        "worst_case_remaining_ambiguity_measure": solver.qs(worst_rem_meas),
        "worst_case_remaining_ambiguity_measure_float": round(float(worst_rem_meas), 8),
        "minimax_reduction_fraction": solver.qs(mrf),
        "minimax_reduction_fraction_float": round(float(mrf), 8),
        "worst_case_remaining_ambiguity_cells": worst_rem_cells,
        "worst_case_remaining_ambiguity_modes": worst_rem_modes,
        "expected_remaining_ambiguity_measure": solver.qs(exp_rem),
        "expected_remaining_ambiguity_measure_float": round(float(exp_rem), 8),
        "expected_ambiguity_reduction_fraction": solver.qs(exp_red_frac),
        "expected_ambiguity_reduction_fraction_float": round(float(exp_red_frac), 8),
        "guarantees_1step_certification": bool(worst_rem_cells == 0),
        "post_check_outcomes": branch_evals,
        # Exact rational sort keys (stripped before JSON serialization)
        "_q_wcra": worst_rem_meas,
        "_q_rmf": rmf,
        "_q_cbf": cbf,
        "_q_mrf": mrf,
        "_q_era": exp_rem,
    }


def minimax_action_sort_key(ev: Dict[str, Any]) -> Tuple[Any, ...]:
    """Exact lexicographic Minimax key (smaller is strictly better)."""
    return (
        ev["_q_wcra"],
        ev["worst_case_remaining_ambiguity_cells"],
        -ev["_q_rmf"],
        -ev["_q_cbf"],
        ev["worst_case_remaining_ambiguity_modes"],
        ev["_q_era"],
        ev["branch_count"],
        ev["action_id"],
    )


def measure_split_action_sort_key(ev: Dict[str, Any]) -> Tuple[Any, ...]:
    """Exact lexicographic Measure-Split key (smaller is strictly better)."""
    return (
        -ev["_q_rmf"],
        ev["_q_era"],
        -ev["_q_cbf"],
        ev["worst_case_remaining_ambiguity_cells"],
        ev["_q_wcra"],
        ev["branch_count"],
        ev["action_id"],
    )


def _strip_internal_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_internal_fields(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_internal_fields(x) for x in obj]
    return obj


def _build_highlight_3d(
    checked_fracture_id: str,
    branch_solve: Dict[str, Any],
) -> Dict[str, Any]:
    st = branch_solve["status"]
    cw = branch_solve.get("connected_witness")
    dw = branch_solve.get("disconnected_witness")
    if st == "CERTIFIED_CONNECTED" and cw is not None:
        return {
            "mode": cw["mode"],
            "t": cw["t"],
            "t_float": cw["t_float"],
            "checked_fracture_id": checked_fracture_id,
            "view_kind": "CERTIFIED_CONNECTED_PATH",
            "connected_fracture_ids": list(cw["fracture_path"]),
            "connected_triangle_ids": [n for n in cw["path"] if not n.startswith("@")],
            "cut_S_fracture_ids": [],
            "cut_T_fracture_ids": [],
            "cut_S_triangle_ids": [],
            "cut_T_triangle_ids": [],
            "active_edges": cw["active_edges"],
        }
    if st == "CERTIFIED_DISCONNECTED" and dw is not None:
        return {
            "mode": dw["mode"],
            "t": dw["t"],
            "t_float": dw["t_float"],
            "checked_fracture_id": checked_fracture_id,
            "view_kind": "CERTIFIED_DISCONNECTED_CUT",
            "connected_fracture_ids": [],
            "connected_triangle_ids": [],
            "cut_S_fracture_ids": list(dw["fracture_cut"]["S_fractures"]),
            "cut_T_fracture_ids": list(dw["fracture_cut"]["T_fractures"]),
            "cut_S_triangle_ids": list(dw["cut"]["S"]),
            "cut_T_triangle_ids": list(dw["cut"]["T"]),
            "active_edges": dw["active_edges"],
        }
    # Still AMBIGUOUS: provide both witnesses' 3D highlights
    return {
        "mode": cw["mode"] if cw else (dw["mode"] if dw else None),
        "t": cw["t"] if cw else (dw["t"] if dw else None),
        "t_float": cw["t_float"] if cw else (dw["t_float"] if dw else None),
        "checked_fracture_id": checked_fracture_id,
        "view_kind": "STILL_AMBIGUOUS_DUAL_WITNESS",
        "connected_fracture_ids": list(cw["fracture_path"]) if cw else [],
        "connected_triangle_ids": [n for n in cw["path"] if not n.startswith("@")] if cw else [],
        "cut_S_fracture_ids": list(dw["fracture_cut"]["S_fractures"]) if dw else [],
        "cut_T_fracture_ids": list(dw["fracture_cut"]["T_fractures"]) if dw else [],
        "cut_S_triangle_ids": list(dw["cut"]["S"]) if dw else [],
        "cut_T_triangle_ids": list(dw["cut"]["T"]) if dw else [],
        "active_edges": cw["active_edges"] if cw else [],
    }


def _materialize_tree_branches(
    model: Dict[str, Any],
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    action_eval: Dict[str, Any],
    *,
    expand_depth2: bool = False,
    tri_to_frac: Optional[Dict[str, str]] = None,
    fracture_ids: Optional[Sequence[str]] = None,
    branch_cache: Optional[Dict[Any, Dict[str, Any]]] = None,
    lattice_index: Optional[solver.CellLatticeIndex] = None,
) -> List[Dict[str, Any]]:
    """Materialize full branch certificates, 3D highlights, and optional depth-2 follow-up."""
    out_branches: List[Dict[str, Any]] = []
    fid = action_eval["fracture_id"]
    for br in action_eval["post_check_outcomes"]:
        b_modes = br["_raw_allowed_modes"]
        b_segs = br["_raw_mode_segments"]
        full_res = solver.solve_conditioned_subdomain(
            model,
            windows_by_mode,
            b_modes,
            b_segs,
            include_witnesses=True,
            include_contact_steps=True,
        )
        # Sanity invariant: full solve must agree identically with evaluation pass!
        if full_res["status"] != br["post_status"] or full_res["cells_count"] != br["branch_cells_count"]:
            raise RuntimeError("internal mismatch between fast evaluation and materialized collapse_tree branch")

        next_step = None
        if expand_depth2 and full_res["status"] == "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
            sub_domain = {"allowed_modes": b_modes, "mode_segments": b_segs}
            sub_actions = enumerate_candidate_actions(
                model, windows_by_mode, sub_domain, tri_to_frac=tri_to_frac, fracture_ids=fracture_ids
            )
            sub_evals = []
            for sfid, alist in sub_actions.items():
                for act in alist:
                    sub_evals.append(
                        evaluate_action_exact(
                            model,
                            windows_by_mode,
                            full_res["total_measure"],
                            full_res["cells_count"],
                            act,
                            branch_cache=branch_cache,
                            lattice_index=lattice_index,
                        )
                    )
            if sub_evals:
                best2 = min(sub_evals, key=minimax_action_sort_key)
                next_step = {
                    "recommended_fracture_id": best2["fracture_id"],
                    "recommended_action_id": best2["action_id"],
                    "action_type": best2["action_type"],
                    "action_summary_zh": best2["action_summary_zh"],
                    "action_summary_en": best2["action_summary_en"],
                    "certified_branch_fraction": best2["certified_branch_fraction"],
                    "resolved_measure_fraction": best2["resolved_measure_fraction"],
                    "minimax_reduction_fraction": best2["minimax_reduction_fraction"],
                    "guarantees_1step_certification": best2["guarantees_1step_certification"],
                    "branches": _materialize_tree_branches(
                        model,
                        windows_by_mode,
                        best2,
                        expand_depth2=False,
                        tri_to_frac=tri_to_frac,
                        fracture_ids=fracture_ids,
                        branch_cache=branch_cache,
                        lattice_index=lattice_index,
                    ),
                }

        out_branches.append(
            {
                "branch_id": br["branch_id"],
                "label_zh": br["label_zh"],
                "label_en": br["label_en"],
                "condition": br["condition"],
                "post_status": full_res["status"],
                "collapsed_to_certified": full_res["status"] in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"),
                "exists_connected": full_res["exists_connected"],
                "all_connected": full_res["all_connected"],
                "branch_measure": br["branch_measure"],
                "branch_measure_float": br["branch_measure_float"],
                "branch_measure_fraction": br["branch_measure_fraction"],
                "branch_cells_count": full_res["cells_count"],
                "branch_modes_count": len(full_res["active_modes"]),
                "connected_witness": full_res["connected_witness"],
                "disconnected_witness": full_res["disconnected_witness"],
                "highlight_3d": _build_highlight_3d(fid, full_res),
                "next_step": next_step,
            }
        )
    return out_branches


# ---------------------------------------------------------------------------
# Main entry: build `active_check.json` for a certified cluster
# ---------------------------------------------------------------------------


def build_active_check(
    model: Dict[str, Any],
    cert: Dict[str, Any],
    *,
    windows_by_mode: Optional[Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]]] = None,
    top_k_depth2: int = 3,
) -> Dict[str, Any]:
    """Compute full active next-best-check ranking and collapse_tree for one cluster."""
    cid = cert["cluster_id"]
    base_meta = {
        "schema": ACTIVE_SCHEMA,
        "scope": cert.get("scope"),
        "cluster_id": cid,
        "site_id": cert.get("site_id"),
        "stratum": cert.get("stratum"),
        "input_sha256": cert.get("input_sha256"),
        "root_certification_status": cert.get("status"),
        "not_claimed": list(solver.NOT_CLAIMED_STANDARD),
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
    }

    # Strict guard (M2-N3 & M2-N6): never invent recommendations on certified or refused clusters!
    if not cert.get("search_complete"):
        return {
            **base_meta,
            "status": "NOT_APPLICABLE_REFUSED",
            "reason": f"certification incomplete ({cert.get('status')}): {cert.get('reason')}",
            "ranked_fractures": [],
            "top_recommendation": None,
            "collapse_tree": None,
        }

    if cert["status"] in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"):
        return {
            **base_meta,
            "status": "ALREADY_CERTIFIED",
            "reason": f"cluster is already {cert['status']}; zero ambiguity remains, no active check required",
            "ranked_fractures": [],
            "top_recommendation": None,
            "collapse_tree": {
                "schema": TREE_SCHEMA,
                "cluster_id": cid,
                "input_sha256": cert["input_sha256"],
                "root_status": cert["status"],
                "already_certified": True,
                "top_k_included": 0,
                "top_fracture_ids": [],
                "fractures": {},
            },
        }

    if cert["status"] != "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
        raise ValueError(f"unexpected certification status: {cert['status']!r}")

    t0 = time.perf_counter()
    wbm = windows_by_mode if windows_by_mode is not None else windows_by_mode_from_cert(cert)
    tri_to_frac = cert["triangle_to_fracture"]
    fracture_ids = list(cert["fracture_ids"])
    root_domain = initial_domain_state(model)
    root_measure = solver.q_exact(cert["measure_summary"]["total_feasible_measure"])
    root_cells = cert["enumeration"]["cells"]
    roles = compute_witness_diff_roles(model, cert, wbm)
    actions_by_frac = enumerate_candidate_actions(
        model, wbm, root_domain, tri_to_frac=tri_to_frac, fracture_ids=fracture_ids
    )
    branch_cache: Dict[Any, Dict[str, Any]] = {}
    lattice_index = solver.CellLatticeIndex(cert["cells"])

    fcat = model.get("fracture_catalog") or {}
    frac_rows: List[Dict[str, Any]] = []
    best_eval_by_frac: Dict[str, Optional[Dict[str, Any]]] = {}

    for fid in fracture_ids:
        cand_actions = actions_by_frac.get(fid, [])
        evals = [
            evaluate_action_exact(
                model,
                wbm,
                root_measure,
                root_cells,
                act,
                branch_cache=branch_cache,
                lattice_index=lattice_index,
            )
            for act in cand_actions
        ]
        role_info = roles[fid]
        cat_role = fcat.get(fid, {}).get("role") or ""
        is_anchor = cat_role.startswith("ANCHOR")

        if evals:
            best_mm = min(evals, key=minimax_action_sort_key)
            best_ms = min(evals, key=measure_split_action_sort_key)
            # An action is informative iff it strictly reduces worst-case ambiguity OR resolves positive measure/cells
            is_informative = bool(
                best_mm["_q_mrf"] > ZERO
                or best_mm["_q_rmf"] > ZERO
                or best_mm["worst_case_remaining_ambiguity_cells"] < root_cells
            )
            best_eval_by_frac[fid] = best_mm if is_informative else None
            sort_key = (
                0 if is_informative else 1,
                best_mm["_q_wcra"],
                best_mm["worst_case_remaining_ambiguity_cells"],
                -best_mm["_q_rmf"],
                -best_mm["_q_cbf"],
                best_mm["worst_case_remaining_ambiguity_modes"],
                best_mm["_q_era"],
                ROLE_PRIORITY.get(role_info["witness_diff_role"], 99),
                1 if is_anchor else 0,
                fid,
            )
            row = {
                "fracture_id": fid,
                "catalog_role": cat_role,
                "is_informative": is_informative,
                "witness_diff_role": role_info["witness_diff_role"],
                "geometric_attribution_zh": role_info["geometric_attribution_zh"],
                "witness_role_details": role_info,
                "candidate_action_count": len(evals),
                "best_action_id": best_mm["action_id"],
                "best_action_type": best_mm["action_type"],
                "best_action_summary_zh": best_mm["action_summary_zh"],
                "best_action_summary_en": best_mm["action_summary_en"],
                "certified_branch_fraction": best_mm["certified_branch_fraction"],
                "certified_branch_fraction_float": best_mm["certified_branch_fraction_float"],
                "resolved_measure_fraction": best_mm["resolved_measure_fraction"],
                "resolved_measure_fraction_float": best_mm["resolved_measure_fraction_float"],
                "worst_case_remaining_ambiguity_measure": best_mm["worst_case_remaining_ambiguity_measure"],
                "worst_case_remaining_ambiguity_measure_float": best_mm["worst_case_remaining_ambiguity_measure_float"],
                "minimax_reduction_fraction": best_mm["minimax_reduction_fraction"],
                "minimax_reduction_fraction_float": best_mm["minimax_reduction_fraction_float"],
                "worst_case_remaining_ambiguity_cells": best_mm["worst_case_remaining_ambiguity_cells"],
                "worst_case_remaining_ambiguity_modes": best_mm["worst_case_remaining_ambiguity_modes"],
                "expected_remaining_ambiguity_measure": best_mm["expected_remaining_ambiguity_measure"],
                "expected_ambiguity_reduction_fraction": best_mm["expected_ambiguity_reduction_fraction"],
                "guarantees_1step_certification": best_mm["guarantees_1step_certification"],
                "post_check_outcomes": _strip_internal_fields(best_mm["post_check_outcomes"]),
                "measure_split_best_action_id": best_ms["action_id"],
                "measure_split_resolved_measure_fraction": best_ms["resolved_measure_fraction"],
                "all_candidate_actions_summary": [
                    {
                        "action_id": e["action_id"],
                        "action_type": e["action_type"],
                        "certified_branch_fraction": e["certified_branch_fraction"],
                        "resolved_measure_fraction": e["resolved_measure_fraction"],
                        "minimax_reduction_fraction": e["minimax_reduction_fraction"],
                        "worst_case_remaining_ambiguity_measure": e["worst_case_remaining_ambiguity_measure"],
                        "guarantees_1step_certification": e["guarantees_1step_certification"],
                    }
                    for e in sorted(evals, key=minimax_action_sort_key)
                ],
                "_sort_key": sort_key,
            }
        else:
            best_eval_by_frac[fid] = None
            sort_key = (
                1,
                root_measure,
                root_cells,
                ZERO,
                ZERO,
                len(root_domain["allowed_modes"]),
                root_measure,
                ROLE_PRIORITY.get(role_info["witness_diff_role"], 99),
                1 if is_anchor else 0,
                fid,
            )
            row = {
                "fracture_id": fid,
                "catalog_role": cat_role,
                "is_informative": False,
                "witness_diff_role": role_info["witness_diff_role"],
                "geometric_attribution_zh": role_info["geometric_attribution_zh"],
                "witness_role_details": role_info,
                "candidate_action_count": 0,
                "best_action_id": None,
                "best_action_type": None,
                "best_action_summary_zh": "无可用可观测划分动作（零收益）",
                "best_action_summary_en": "No observable splitting action (zero gain)",
                "certified_branch_fraction": "0/1",
                "certified_branch_fraction_float": 0.0,
                "resolved_measure_fraction": "0/1",
                "resolved_measure_fraction_float": 0.0,
                "worst_case_remaining_ambiguity_measure": solver.qs(root_measure),
                "worst_case_remaining_ambiguity_measure_float": round(float(root_measure), 8),
                "minimax_reduction_fraction": "0/1",
                "minimax_reduction_fraction_float": 0.0,
                "worst_case_remaining_ambiguity_cells": root_cells,
                "worst_case_remaining_ambiguity_modes": len(root_domain["allowed_modes"]),
                "expected_remaining_ambiguity_measure": solver.qs(root_measure),
                "expected_ambiguity_reduction_fraction": "0/1",
                "guarantees_1step_certification": False,
                "post_check_outcomes": [],
                "measure_split_best_action_id": None,
                "measure_split_resolved_measure_fraction": "0/1",
                "all_candidate_actions_summary": [],
                "_sort_key": sort_key,
            }
        frac_rows.append(row)

    frac_rows.sort(key=lambda r: r["_sort_key"])
    for idx, row in enumerate(frac_rows, start=1):
        row["rank"] = idx

    # Build collapse_tree: materialize ALL informative fractures (depth 2 on Top-3)
    informative_rows = [r for r in frac_rows if r["is_informative"]]
    top_ids = [r["fracture_id"] for r in informative_rows[:top_k_depth2]]
    tree_fractures: Dict[str, Any] = {}
    for row in informative_rows:
        fid = row["fracture_id"]
        best_ev = best_eval_by_frac[fid]
        assert best_ev is not None
        expand2 = fid in top_ids
        mat_branches = _materialize_tree_branches(
            model,
            wbm,
            best_ev,
            expand_depth2=expand2,
            tri_to_frac=tri_to_frac,
            fracture_ids=fracture_ids,
            branch_cache=branch_cache,
            lattice_index=lattice_index,
        )
        tree_fractures[fid] = {
            "rank": row["rank"],
            "fracture_id": fid,
            "catalog_role": row["catalog_role"],
            "best_action_id": row["best_action_id"],
            "action_type": row["best_action_type"],
            "action_summary_zh": row["best_action_summary_zh"],
            "action_summary_en": row["best_action_summary_en"],
            "witness_diff_role": row["witness_diff_role"],
            "geometric_attribution_zh": row["geometric_attribution_zh"],
            "certified_branch_fraction": row["certified_branch_fraction"],
            "resolved_measure_fraction": row["resolved_measure_fraction"],
            "minimax_reduction_fraction": row["minimax_reduction_fraction"],
            "guarantees_1step_certification": row["guarantees_1step_certification"],
            "depth2_expanded": expand2,
            "branches": mat_branches,
        }

    ranked_clean = _strip_internal_fields(frac_rows)
    top_rec = ranked_clean[0] if ranked_clean and ranked_clean[0]["is_informative"] else None
    collapse_tree = {
        "schema": TREE_SCHEMA,
        "cluster_id": cid,
        "input_sha256": cert["input_sha256"],
        "root_status": cert["status"],
        "already_certified": False,
        "root_feasible_measure": solver.qs(root_measure),
        "root_cells_count": root_cells,
        "top_k_included": len(top_ids),
        "top_fracture_ids": top_ids,
        "all_informative_fracture_ids": [r["fracture_id"] for r in informative_rows],
        "fractures": tree_fractures,
    }

    return {
        **base_meta,
        "status": "ACTIVE_RECOMMENDATIONS_READY",
        "reason": None,
        "root_feasible_measure": solver.qs(root_measure),
        "root_cells_count": root_cells,
        "informative_fracture_count": len(informative_rows),
        "zero_gain_fracture_count": len(ranked_clean) - len(informative_rows),
        "ranking_policy": "OURS_MINIMAX_EXACT",
        "top_recommendation": (
            {
                "rank": 1,
                "fracture_id": top_rec["fracture_id"],
                "best_action_id": top_rec["best_action_id"],
                "best_action_type": top_rec["best_action_type"],
                "best_action_summary_zh": top_rec["best_action_summary_zh"],
                "witness_diff_role": top_rec["witness_diff_role"],
                "geometric_attribution_zh": top_rec["geometric_attribution_zh"],
                "certified_branch_fraction": top_rec["certified_branch_fraction"],
                "resolved_measure_fraction": top_rec["resolved_measure_fraction"],
                "minimax_reduction_fraction": top_rec["minimax_reduction_fraction"],
                "guarantees_1step_certification": top_rec["guarantees_1step_certification"],
            }
            if top_rec is not None
            else None
        ),
        "ranked_fractures": ranked_clean,
        "collapse_tree": collapse_tree,
        "compute_seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Consistency guard for active_check.json (M2-N3, M2-N4, M2-N5!)
# ---------------------------------------------------------------------------


def _verify_exact_partition(
    parent_domain: Dict[str, Any],
    branches: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
) -> Tuple[bool, str]:
    """Verify that branches form an exact disjoint partition of parent_domain."""
    dom_by_mode = {m["id"]: (solver.q_exact(m["t_closed"][0]), solver.q_exact(m["t_closed"][1])) for m in model["modes"]}
    # Collect all knots per mode from parent + branches
    for mid in parent_domain["allowed_modes"]:
        p_segs = parent_domain["mode_segments"][mid]
        b_segs_list: List[List[Tuple[Q, Q, bool, bool]]] = []
        knots = {pt for lo, hi, _, _ in p_segs for pt in (lo, hi)}
        for br in branches:
            cond = br["condition"]
            raw = cond["mode_segments"].get(mid, [])
            b_segs = solver.normalize_segments(raw, dom_by_mode[mid]) if raw else []
            b_segs_list.append(b_segs)
            for lo, hi, _, _ in b_segs:
                knots.add(lo)
                knots.add(hi)
        ordered = sorted(knots)
        # Check every atomic knot and open midpoint
        probes: List[Q] = []
        for i, k in enumerate(ordered):
            probes.append(k)
            if i + 1 < len(ordered):
                probes.append((k + ordered[i + 1]) / 2)
        for pt in probes:
            in_p = solver.admitted_in_intervals(pt, p_segs)
            in_b_count = sum(1 for bs in b_segs_list if solver.admitted_in_intervals(pt, bs))
            if in_p and in_b_count != 1:
                return False, f"partition violation in {mid} at t={solver.qs(pt)}: parent=True, branch_hits={in_b_count}"
            if (not in_p) and in_b_count != 0:
                return False, f"branch leaks outside parent in {mid} at t={solver.qs(pt)}: branch_hits={in_b_count}"
    return True, "ok"


def verify_active_check_local(
    model: Dict[str, Any],
    cert: Dict[str, Any],
    active_check: Dict[str, Any],
    *,
    recompute_from_scratch_branches: bool = False,
) -> Dict[str, Any]:
    """Verify active_check.json consistency against model.json and certification.json."""

    def fail(msg: str) -> Dict[str, Any]:
        return {"ok": False, "status": "ACTIVE_CHECK_INVALID", "reason": msg}

    try:
        if active_check.get("schema") != ACTIVE_SCHEMA:
            return fail("invalid active_check schema")
        if active_check.get("input_sha256") != cert.get("input_sha256"):
            return fail("input_sha256 mismatch between active_check and certification")
        if active_check.get("field_error_rate") is not None or active_check.get("human_time") is not None or active_check.get("hydraulic") is not None:
            return fail("empty-denominator metrics must be null")

        cert_st = cert.get("status")
        if cert_st in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED"):
            if active_check.get("status") != "ALREADY_CERTIFIED":
                return fail(f"certified cluster must report ALREADY_CERTIFIED, got {active_check.get('status')!r}")
            if active_check.get("ranked_fractures"):
                return fail("certified cluster must NOT emit non-empty ranked_fractures (M2-N3)")
            if active_check.get("top_recommendation") is not None:
                return fail("certified cluster must have top_recommendation=null (M2-N3)")
            return {"ok": True, "status": "VERIFIED_ALREADY_CERTIFIED"}

        if not cert.get("search_complete"):
            if active_check.get("status") != "NOT_APPLICABLE_REFUSED" or active_check.get("ranked_fractures"):
                return fail("refused cluster must report NOT_APPLICABLE_REFUSED with empty ranked_fractures")
            return {"ok": True, "status": "VERIFIED_REFUSED_ACTIVE_SKIP"}

        if cert_st != "AMBIGUOUS_WITH_VERIFIED_WITNESSES" or active_check.get("status") != "ACTIVE_RECOMMENDATIONS_READY":
            return fail(f"mismatched statuses cert={cert_st!r} active={active_check.get('status')!r}")

        wbm = windows_by_mode_from_cert(cert)
        root_dom = initial_domain_state(model)
        dom_by_mode = {m["id"]: (solver.q_exact(m["t_closed"][0]), solver.q_exact(m["t_closed"][1])) for m in model["modes"]}

        # Check all ranked_fractures ordering & isolated-fracture zero-gain invariant (M2-N4)
        ranked = active_check.get("ranked_fractures", [])
        if not ranked:
            return fail("AMBIGUOUS cluster has empty ranked_fractures")
        seen_zero = False
        for row in ranked:
            if row["witness_diff_role"] == "ISOLATED_IRRELEVANT_FRACTURE":
                if (
                    row["minimax_reduction_fraction"] != "0/1"
                    or row["resolved_measure_fraction"] != "0/1"
                    or row["certified_branch_fraction"] != "0/1"
                    or row["is_informative"] is not False
                ):
                    return fail(f"isolated fracture {row['fracture_id']} has non-zero gain (M2-N4)")
            if not row["is_informative"]:
                seen_zero = True
            elif seen_zero:
                return fail(f"informative fracture {row['fracture_id']} ranked AFTER a zero-gain fracture (M2-N4)")

            # Check every post_check_outcomes branch in ranked_fractures
            outcomes = row.get("post_check_outcomes", [])
            if outcomes:
                ok_part, p_msg = _verify_exact_partition(root_dom, outcomes, model)
                if not ok_part:
                    return fail(f"ranked fracture {row['fracture_id']} invalid partition: {p_msg}")
                for br in outcomes:
                    cond = br["condition"]
                    m_segs = {
                        m: solver.normalize_segments(cond["mode_segments"][m], dom_by_mode[m])
                        for m in cond["allowed_modes"]
                    }
                    re = solver.solve_conditioned_subdomain(
                        model, wbm, cond["allowed_modes"], m_segs, include_witnesses=False
                    )
                    if re["status"] != br["post_status"]:
                        return fail(
                            f"post_check_outcome mismatch on {row['fracture_id']}/{br['branch_id']}: "
                            f"claimed={br['post_status']} actual={re['status']} (M2-N5)"
                        )
                    if (re["status"] in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")) != br["collapsed_to_certified"]:
                        return fail(f"collapsed_to_certified mismatch on {row['fracture_id']}/{br['branch_id']} (M2-N5)")
                    if re["cells_count"] != br["branch_cells_count"]:
                        return fail(f"branch_cells_count mismatch on {row['fracture_id']}/{br['branch_id']}")

        # Check collapse_tree branches (depth 1 and depth 2)
        tree = active_check.get("collapse_tree")
        if not isinstance(tree, dict) or tree.get("schema") != TREE_SCHEMA:
            return fail("missing or invalid collapse_tree")
        top1_fid = tree.get("top_fracture_ids", [None])[0] if tree.get("top_fracture_ids") else None
        branches_verified = 0
        for fid, fentry in tree.get("fractures", {}).items():
            branches = fentry.get("branches", [])
            if not branches:
                return fail(f"collapse_tree fracture {fid} has empty branches")
            ok_part, p_msg = _verify_exact_partition(root_dom, branches, model)
            if not ok_part:
                return fail(f"collapse_tree {fid} partition violation: {p_msg}")
            for br in branches:
                cond = br["condition"]
                m_segs = {
                    m: solver.normalize_segments(cond["mode_segments"][m], dom_by_mode[m])
                    for m in cond["allowed_modes"]
                }
                if recompute_from_scratch_branches and fid == top1_fid:
                    re = solver.recompute_conditioned_from_scratch(model, cond["allowed_modes"], cond["mode_segments"])
                else:
                    re = solver.solve_conditioned_subdomain(
                        model, wbm, cond["allowed_modes"], m_segs, include_witnesses=False
                    )
                if re["status"] != br["post_status"]:
                    return fail(
                        f"collapse_tree mismatch on {fid}/{br['branch_id']}: "
                        f"claimed={br['post_status']} actual={re['status']} (M2-N5)"
                    )
                if (re["status"] in ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")) != br["collapsed_to_certified"]:
                    return fail(f"collapse_tree collapsed flag mismatch on {fid}/{br['branch_id']} (M2-N5)")
                if re["cells_count"] != br["branch_cells_count"]:
                    return fail(f"collapse_tree cell count mismatch on {fid}/{br['branch_id']}")
                branches_verified += 1

                # Depth-2 verification if present
                ns = br.get("next_step")
                if ns is not None:
                    sub_dom = {"allowed_modes": cond["allowed_modes"], "mode_segments": m_segs}
                    ok_p2, p2_msg = _verify_exact_partition(sub_dom, ns["branches"], model)
                    if not ok_p2:
                        return fail(f"collapse_tree depth-2 partition violation on {fid}/{br['branch_id']}: {p2_msg}")
                    for br2 in ns["branches"]:
                        c2 = br2["condition"]
                        m2_segs = {
                            m: solver.normalize_segments(c2["mode_segments"][m], dom_by_mode[m])
                            for m in c2["allowed_modes"]
                        }
                        re2 = solver.solve_conditioned_subdomain(
                            model, wbm, c2["allowed_modes"], m2_segs, include_witnesses=False
                        )
                        if re2["status"] != br2["post_status"]:
                            return fail(
                                f"collapse_tree depth-2 mismatch on {fid}/{br['branch_id']}/{br2['branch_id']}: "
                                f"claimed={br2['post_status']} actual={re2['status']} (M2-N5)"
                            )
                        branches_verified += 1

        return {
            "ok": True,
            "status": "VERIFIED_ACTIVE_CHECK",
            "branches_verified": branches_verified,
            "ranked_fractures_count": len(ranked),
        }
    except Exception as exc:
        return fail(f"active_check verification exception: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI: batch over a run_dir's cases/ or single verification
# ---------------------------------------------------------------------------


def run_batch_active(register_path: Path, run_dir: Path) -> Dict[str, Any]:
    reg = solver.load_strict_json(register_path)
    reg_dir = register_path.parent
    cases_dir = run_dir / "cases"
    summary_rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()

    for entry in reg["clusters"]:
        cid = entry["cluster_id"]
        c_path = cases_dir / cid / "certification.json"
        if not c_path.is_file():
            raise FileNotFoundError(f"missing certification.json for {cid}: {c_path}")
        m_raw = solver.load_strict_json(reg_dir / entry["model_path"])
        cert = solver.load_strict_json(c_path)
        ac = build_active_check(m_raw, cert)
        verdict = verify_active_check_local(m_raw, cert, ac)
        if not verdict["ok"]:
            raise RuntimeError(f"active_check local verification failed on {cid}: {verdict}")
        ac["local_recheck"] = verdict
        solver.write_json(cases_dir / cid / "active_check.json", ac)

        top = ac.get("top_recommendation")
        summary_rows.append(
            {
                "cluster_id": cid,
                "site_id": entry["site_id"],
                "certification_status": cert["status"],
                "active_status": ac["status"],
                "top_fracture_id": top["fracture_id"] if top else None,
                "top_action_type": top["best_action_type"] if top else None,
                "top_witness_diff_role": top["witness_diff_role"] if top else None,
                "top_minimax_reduction": top["minimax_reduction_fraction"] if top else None,
                "top_resolved_measure_fraction": top["resolved_measure_fraction"] if top else None,
                "top_guarantees_1step": top["guarantees_1step_certification"] if top else None,
                "active_check_path": f"cases/{cid}/active_check.json",
            }
        )
        if top:
            print(
                f"[M2-ACTIVE] {cid:12s} | #1={top['fracture_id']} | {top['best_action_type']:21s} | "
                f"role={top['witness_diff_role']} | MRF={top['minimax_reduction_fraction']} | "
                f"RMF={top['resolved_measure_fraction']} | 1step={top['guarantees_1step_certification']}",
                flush=True,
            )
        else:
            print(f"[M2-ACTIVE] {cid:12s} | {ac['status']}", flush=True)

    return {
        "total_clusters": len(summary_rows),
        "ambiguous_active_ready": sum(1 for r in summary_rows if r["active_status"] == "ACTIVE_RECOMMENDATIONS_READY"),
        "already_certified": sum(1 for r in summary_rows if r["active_status"] == "ALREADY_CERTIFIED"),
        "not_applicable_refused": sum(1 for r in summary_rows if r["active_status"] == "NOT_APPLICABLE_REFUSED"),
        "total_seconds": round(time.perf_counter() - t_all, 3),
        "clusters": summary_rows,
    }


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2 Active Identifiability Next-Best-Check Ranker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", help="Path to M1 cluster_register.json")
    group.add_argument(
        "--verify-active",
        nargs=3,
        metavar=("MODEL_JSON", "CERT_JSON", "ACTIVE_JSON"),
        help="Verify active_check.json against model.json and certification.json",
    )
    parser.add_argument("--run-dir", help="Run directory containing cases/<cluster_id>/certification.json")
    parser.add_argument("--scratch-recompute", action="store_true", help="Recompute collapse_tree branches from scratch")
    args = parser.parse_args(argv)

    if args.verify_active:
        m = solver.load_strict_json(Path(args.verify_active[0]))
        c = solver.load_strict_json(Path(args.verify_active[1]))
        a = solver.load_strict_json(Path(args.verify_active[2]))
        verdict = verify_active_check_local(m, c, a, recompute_from_scratch_branches=args.scratch_recompute)
        print(json.dumps(verdict, ensure_ascii=False))
        return 0 if verdict["ok"] else 2

    if not args.run_dir:
        raise SystemExit("--run-dir required with --register")
    summary = run_batch_active(Path(args.register), Path(args.run_dir))
    print(json.dumps({k: v for k, v in summary.items() if k != "clusters"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
