"""M2 Exact Whole-Network Certification Engine for Real-Derived Multi-Fracture Clusters.

Consumes M1 real-derived conditioned models (`schema = "TC-network/real-conditioned/v1"`,
`scope = "REAL_DERIVED_CONDITIONED_FINITE_UNION"`):
- Strictly validates every input through M1 `real_network_format.validate()` before solving.
- Reuses frozen `TC_network` / `triangle_contact` exact rational pair projection & cell
  partitioning in-memory (NEVER writing synthetic-scope projections to disk, NEVER modifying
  any frozen directory or creating `__pycache__` bytecode).
- Zero floating-point final decisions: all contact windows, cell boundaries, connectivity
  paths/cuts, stability margin rho, and conditioned subdomain intersections use exact
  `fractions.Fraction` arithmetic.
- Explicitly distinguishes:
  * `CERTIFIED_CONNECTED`
  * `CERTIFIED_DISCONNECTED`
  * `AMBIGUOUS_WITH_VERIFIED_WITNESSES`
  * `BUDGET_EXCEEDED` / `INFEASIBLE` / `REFUSED` (strict refusal envelope, never mixed into
    connectivity denominators).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SPRINT_DIR = HERE.parent
REPO_ROOT = SPRINT_DIR.parent.parent
M1_DIR = SPRINT_DIR / "M1_real_to_exact_bridge"
EXECUTION_DIR = REPO_ROOT / "results" / "w180_technical_route_20261117" / "execution_next"
TC_DIR = EXECUTION_DIR / "dispatch_20261118" / "TC_network"
TRI_DIR = EXECUTION_DIR / "triangle_contact"

ZERO = Q(0)
ONE = Q(1)

CERT_SCHEMA = "m2-real-certification/v1"
REAL_SCOPE = "REAL_DERIVED_CONDITIONED_FINITE_UNION"
NOT_CLAIMED_STANDARD = [
    "非地下体内无条件贯通真值 (not unconditional subsurface/in-rock ground truth)",
    "非水力连通与非渗透率结论 (not hydraulic connectivity or permeability)",
    "非人工实测耗时结论 (human_time = null)",
    "条件几何模型结论不等同于合成域也不等同于现场开挖真值",
]

_LOADED_MODULES: Dict[str, Any] = {}


def _load_module(name: str, path: Path, extra_sys_paths: Sequence[Path] = ()) -> Any:
    if name in _LOADED_MODULES:
        return _LOADED_MODULES[name]
    for p in extra_sys_paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _LOADED_MODULES[name] = mod
    return mod


def load_frozen_deps() -> Tuple[Any, Any, Any, Any, Any]:
    """Load read-only M1 validator and frozen TC/triangle_contact exact modules."""
    rnf = _load_module("m1_real_network_format", M1_DIR / "real_network_format.py")
    cn = _load_module("tc_compile_network", TC_DIR / "compile_network.py", (TC_DIR,))
    cells_mod = _load_module("tc_cells", TC_DIR / "cells.py", (TC_DIR,))
    _, compile_pair, verify_pair = cn.dependencies()
    return rnf, cn, cells_mod, compile_pair, verify_pair


def _json_object_no_dups(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON object key: {k}")
        out[k] = v
    return out


def _json_no_nan(val: str) -> Any:
    raise ValueError(f"nonfinite JSON constant: {val}")


def load_strict_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(
            fh,
            object_pairs_hook=_json_object_no_dups,
            parse_constant=_json_no_nan,
        )


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qs(x: Q) -> str:
    return f"{x.numerator}/{x.denominator}"


def q_exact(x: Any) -> Q:
    if isinstance(x, Q):
        return x
    return Q(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Exact rational interval algebra (for conditioning & measure computation)
# ---------------------------------------------------------------------------


def normalize_segments(
    raw_segments: Sequence[Dict[str, Any]],
    domain: Tuple[Q, Q],
) -> List[Tuple[Q, Q, bool, bool]]:
    """Normalize explicit rational segments into sorted disjoint intervals."""
    items: List[Tuple[Q, Q, bool, bool]] = []
    for seg in raw_segments:
        lo, hi = q_exact(seg["lo"]), q_exact(seg["hi"])
        lc, hc = bool(seg["lo_closed"]), bool(seg["hi_closed"])
        if lo < domain[0]:
            lo, lc = domain[0], True
        if hi > domain[1]:
            hi, hc = domain[1], True
        if lo > hi or (lo == hi and not (lc and hc)):
            continue
        items.append((lo, hi, lc, hc))
    if not items:
        return []
    knots = sorted({pt for seg in items for pt in (seg[0], seg[1])})

    def inside(t: Q) -> bool:
        return any(
            (t > a or (t == a and ac)) and (t < b or (t == b and bc))
            for a, b, ac, bc in items
        )

    atoms: List[Tuple[Q, Q, bool, bool]] = []
    for i, k in enumerate(knots):
        if inside(k):
            atoms.append((k, k, True, True))
        if i + 1 < len(knots):
            hi_k = knots[i + 1]
            if inside((k + hi_k) / 2):
                atoms.append((k, hi_k, False, False))
    merged: List[Tuple[Q, Q, bool, bool]] = []
    for atom in atoms:
        if merged and merged[-1][1] == atom[0] and (merged[-1][3] or atom[2]):
            prev = merged.pop()
            merged.append((prev[0], atom[1], prev[2], atom[3]))
        else:
            merged.append(atom)
    return merged


def segments_to_json(intervals: Sequence[Tuple[Q, Q, bool, bool]]) -> List[Dict[str, Any]]:
    return [
        {"lo": qs(lo), "hi": qs(hi), "lo_closed": lc, "hi_closed": hc}
        for lo, hi, lc, hc in intervals
    ]


def intersect_segments_halfline(
    intervals: Sequence[Tuple[Q, Q, bool, bool]],
    pred_kind: str,
    threshold: Q,
) -> List[Tuple[Q, Q, bool, bool]]:
    """Exact intersection of normalized segments with a rational half-line.

    pred_kind in {'LT_THRESHOLD', 'LE_THRESHOLD', 'GT_THRESHOLD', 'GE_THRESHOLD'}.
    """
    out: List[Tuple[Q, Q, bool, bool]] = []
    for lo, hi, lc, hc in intervals:
        if pred_kind == "LT_THRESHOLD":
            if lo > threshold or (lo == threshold):
                continue
            if hi > threshold:
                n_hi, n_hc = threshold, False
            elif hi == threshold:
                n_hi, n_hc = threshold, False
            else:
                n_hi, n_hc = hi, hc
            n_lo, n_lc = lo, lc
        elif pred_kind == "LE_THRESHOLD":
            if lo > threshold or (lo == threshold and not lc):
                continue
            if hi > threshold:
                n_hi, n_hc = threshold, True
            else:
                n_hi, n_hc = hi, hc
            n_lo, n_lc = lo, lc
        elif pred_kind == "GT_THRESHOLD":
            if hi < threshold or (hi == threshold):
                continue
            if lo < threshold:
                n_lo, n_lc = threshold, False
            elif lo == threshold:
                n_lo, n_lc = threshold, False
            else:
                n_lo, n_lc = lo, lc
            n_hi, n_hc = hi, hc
        elif pred_kind == "GE_THRESHOLD":
            if hi < threshold or (hi == threshold and not hc):
                continue
            if lo < threshold:
                n_lo, n_lc = threshold, True
            else:
                n_lo, n_lc = lo, lc
            n_hi, n_hc = hi, hc
        else:
            raise ValueError(f"unsupported predicate kind: {pred_kind}")
        if n_lo < n_hi or (n_lo == n_hi and n_lc and n_hc):
            out.append((n_lo, n_hi, n_lc, n_hc))
    return out


def segments_measure(intervals: Sequence[Tuple[Q, Q, bool, bool]]) -> Q:
    return sum((hi - lo for lo, hi, _, _ in intervals), ZERO)


def admitted_in_intervals(t: Q, intervals: Sequence[Tuple[Q, Q, bool, bool]]) -> bool:
    return any(
        (t > lo or (t == lo and lc)) and (t < hi or (t == hi and hc))
        for lo, hi, lc, hc in intervals
    )


def merge_closed_intervals(intervals: Sequence[Tuple[Q, Q]]) -> List[Tuple[Q, Q]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: List[Tuple[Q, Q]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# ---------------------------------------------------------------------------
# Exact rational geometry helpers (provably identical to compile_pair)
# ---------------------------------------------------------------------------


def _parse_item_exact(item: Dict[str, Any]) -> Tuple[Tuple[Tuple[Q, Q, Q], ...], Tuple[Q, Q, Q]]:
    verts = tuple(tuple(q_exact(c) for c in v) for v in item["vertices"])
    vel = tuple(q_exact(c) for c in item["v"])
    return verts, vel


def _exact_aabb_disjoint(
    left_v: Tuple[Tuple[Q, Q, Q], ...],
    left_vel: Tuple[Q, Q, Q],
    right_v: Tuple[Tuple[Q, Q, Q], ...],
    right_vel: Tuple[Q, Q, Q],
    dom_lo: Q,
    dom_hi: Q,
) -> bool:
    """Exact rational 3-coordinate-axis separation test over [dom_lo, dom_hi].

    Mathematically sound: coordinate axes e_0, e_1, e_2 are valid separating axes.
    If projections onto X, Y, or Z fail to overlap for all t in [dom_lo, dom_hi],
    then the two convex sets cannot intersect at any t in [dom_lo, dom_hi].
    Zero floating-point arithmetic.
    """
    lo, hi = dom_lo, dom_hi
    for axis in (0, 1, 2):
        a_min = min(p[axis] for p in left_v)
        a_max = max(p[axis] for p in left_v)
        b_min = min(p[axis] for p in right_v)
        b_max = max(p[axis] for p in right_v)
        s = left_vel[axis] - right_vel[axis]
        # s * t <= b_max - a_min
        d1 = b_max - a_min
        if s > ZERO:
            cand = d1 / s
            if cand < hi:
                hi = cand
        elif s < ZERO:
            cand = d1 / s
            if cand > lo:
                lo = cand
        elif d1 < ZERO:
            return True
        # -s * t <= a_max - b_min
        d2 = a_max - b_min
        ns = -s
        if ns > ZERO:
            cand = d2 / ns
            if cand < hi:
                hi = cand
        elif ns < ZERO:
            cand = d2 / ns
            if cand > lo:
                lo = cand
        elif d2 < ZERO:
            return True
        if lo > hi:
            return True
    return False


def compute_pair_contact_window(
    payload: Dict[str, Any],
    mode: Dict[str, Any],
    left: Dict[str, Any],
    right: Dict[str, Any],
    left_exact: Tuple[Tuple[Tuple[Q, Q, Q], ...], Tuple[Q, Q, Q]],
    right_exact: Tuple[Tuple[Tuple[Q, Q, Q], ...], Tuple[Q, Q, Q]],
    dom: Tuple[Q, Q],
    dom_key: Tuple[str, str],
    cn: Any,
    compile_pair: Any,
    pair_cache: Dict[Any, Optional[Tuple[Q, Q]]],
    *,
    use_exact_shortcuts: bool = True,
) -> Optional[Tuple[Q, Q]]:
    """Compute exact closed rational contact window [t_lo, t_hi] or None."""
    lv, lvel = left_exact
    rv, rvel = right_exact
    if use_exact_shortcuts:
        # 1. Exact shared-vertex same-velocity identity: p + t*v == p + t*v for all t in dom.
        if lvel == rvel and any(p in rv for p in lv):
            return (dom[0], dom[1])
        # 2. Exact rational 3-axis AABB separating-plane pruning.
        if _exact_aabb_disjoint(lv, lvel, rv, rvel, dom[0], dom[1]):
            return None
    kl = (lv, lvel)
    kr = (rv, rvel)
    cache_key = (kl, kr, dom_key) if kl <= kr else (kr, kl, dom_key)
    if cache_key not in pair_cache:
        raw_win = compile_pair.project_pair(cn.pair_model(payload, mode, left, right))["t_closed"]
        pair_cache[cache_key] = None if raw_win is None else (q_exact(raw_win[0]), q_exact(raw_win[1]))
    return pair_cache[cache_key]


# ---------------------------------------------------------------------------
# Deterministic exact graph classification (identical semantics to TC_network)
# ---------------------------------------------------------------------------


def classify_slice_graph(
    tri_ids: Sequence[str],
    edges: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Exact deterministic BFS connectivity, shortest path, cut (S, T), components."""
    all_nodes = list(tri_ids) + ["@A", "@B"]
    adj: Dict[str, List[str]] = {n: [] for n in all_nodes}
    canon_edges = sorted({(a, b) if a < b else (b, a) for a, b in edges})
    for a, b in canon_edges:
        adj[a].append(b)
        adj[b].append(a)
    for n in adj:
        adj[n].sort()

    # BFS from @A
    parent: Dict[str, Optional[str]] = {"@A": None}
    queue = ["@A"]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt in adj[cur]:
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)

    connected = "@B" in parent
    path: Optional[List[str]] = None
    cut: Optional[Dict[str, Any]] = None
    tri_set = set(tri_ids)

    if connected:
        rev = []
        cur: Optional[str] = "@B"
        while cur is not None:
            rev.append(cur)
            cur = parent[cur]
        path = list(reversed(rev))
    else:
        reached = set(parent.keys())
        unattached = [ep for ep in ("A", "B") if not adj["@" + ep]]
        cut = {
            "S": sorted(tri_set & reached),
            "T": sorted(tri_set - reached),
            "unattached_endpoints": unattached,
        }

    # Connected components of the triangle-only graph (excluding @A, @B)
    tri_adj: Dict[str, List[str]] = {n: [] for n in tri_ids}
    for a, b in canon_edges:
        if not a.startswith("@") and not b.startswith("@"):
            tri_adj[a].append(b)
            tri_adj[b].append(a)
    remaining = set(tri_ids)
    components: List[List[str]] = []
    while remaining:
        start = min(remaining)
        visited = {start}
        q = [start]
        h = 0
        while h < len(q):
            u = q[h]
            h += 1
            for v in tri_adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        components.append(sorted(visited))
        remaining.difference_update(visited)

    return {
        "connected": connected,
        "path": path,
        "cut": cut,
        "components": components,
        "active_edges": [[a, b] for a, b in canon_edges],
    }


def compute_exact_stability(
    model: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Exact stability margin rho and nearest flip witness (matches TC_network/recheck.py)."""
    ref = model.get("reference")
    if ref is None:
        return None
    t0 = q_exact(ref["t"])
    sigma = q_exact(ref["sigma"])
    nominal = next(
        (
            r
            for r in records
            if r["mode"] == ref["mode"]
            and (
                (r["kind"] == "POINT" and q_exact(r["lo"]) == t0)
                or (r["kind"] == "OPEN" and q_exact(r["lo"]) < t0 < q_exact(r["hi"]))
            )
        ),
        None,
    )
    if nominal is None:
        raise ValueError("reference point is not covered by exact cells")
    edits = {m["id"]: q_exact(m["edit_cost"]) for m in model["modes"]}

    flips = []
    for r in records:
        if r["connected"] == nominal["connected"]:
            continue
        lo, hi = q_exact(r["lo"]), q_exact(r["hi"])
        edit = edits[r["mode"]]
        if r["kind"] == "POINT":
            dist = abs(lo - t0)
            temporal_attained = True
            witness_t = lo
        elif lo < t0 < hi:
            dist = ZERO
            temporal_attained = True
            witness_t = t0
        else:
            closest = lo if abs(lo - t0) <= abs(hi - t0) else hi
            dist = abs(closest - t0)
            temporal_attained = False
            if edit > dist / sigma:
                step = min((hi - lo) / 2, (edit * sigma - dist) / 2)
                witness_t = closest + step if closest == lo else closest - step
            else:
                witness_t = (lo + hi) / 2
        temporal_cost = dist / sigma
        infimum = max(temporal_cost, edit)
        attained_cell = temporal_attained or (edit > temporal_cost)
        witness_cost = max(abs(witness_t - t0) / sigma, edit)
        flips.append((infimum, attained_cell, witness_cost, r["mode"], witness_t, r["connected"], r))

    if not flips:
        return {
            "infimum": None,
            "attained": None,
            "nominal_connected": nominal["connected"],
            "flip_witness": None,
        }
    lower = min(row[0] for row in flips)
    attained = any(row[0] == lower and row[1] for row in flips)
    best = min(flips, key=lambda row: (row[2], row[3], row[4]))
    return {
        "infimum": qs(lower),
        "attained": attained,
        "nominal_connected": nominal["connected"],
        "flip_witness": {
            "mode": best[3],
            "t": qs(best[4]),
            "connected": best[5],
            "cost": qs(best[2]),
        },
    }


def _build_tri_to_fracture(model: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for mode in model["modes"]:
        for tri in mode["triangles"]:
            tid = tri["id"]
            comp = tri["component"]
            if tid in mapping and mapping[tid] != comp:
                raise ValueError(f"inconsistent component mapping for triangle {tid}")
            mapping[tid] = comp
    return mapping


def _fracture_path_from_tri_path(
    tri_path: Sequence[str],
    tri_to_frac: Dict[str, str],
) -> List[str]:
    out: List[str] = []
    for node in tri_path:
        if node.startswith("@"):
            continue
        fid = tri_to_frac[node]
        if not out or out[-1] != fid:
            out.append(fid)
    return out


def _fracture_cut_from_tri_cut(
    tri_cut: Dict[str, Any],
    tri_to_frac: Dict[str, str],
) -> Dict[str, Any]:
    s_fracs = sorted({tri_to_frac[tid] for tid in tri_cut["S"]})
    t_fracs = sorted({tri_to_frac[tid] for tid in tri_cut["T"]})
    return {
        "S_fractures": s_fracs,
        "T_fractures": t_fracs,
        "unattached_endpoints": list(tri_cut["unattached_endpoints"]),
    }


def _compute_contact_step_witnesses(
    payload: Dict[str, Any],
    mode: Dict[str, Any],
    tri_path: Sequence[str],
    t_q: Q,
    tri_to_frac: Dict[str, str],
    cn: Any,
    verify_pair: Any,
) -> List[Dict[str, Any]]:
    """Compute exact rational 3D contact points along every edge of a connected witness path."""
    items_by_id = {p["id"]: p for p in cn.nodes(payload, mode)}
    steps: List[Dict[str, Any]] = []
    for a, b in zip(tri_path, tri_path[1:]):
        ia, ib = items_by_id[a], items_by_id[b]
        pm = cn.pair_model(payload, mode, ia, ib, [t_q, t_q])
        v_res = verify_pair.vertices(pm)
        if v_res is None or not v_res.get("endpoint_witnesses"):
            raise ValueError(f"connected path edge ({a}, {b}) failed exact primal contact check at t={qs(t_q)}")
        w = v_res["endpoint_witnesses"][0]
        l_weights = [q_exact(x) for x in w["left_weights"]]
        point = [
            sum(l_weights[j] * q_exact(ia["vertices"][j][k]) for j in range(len(ia["vertices"])))
            + t_q * q_exact(ia["v"][k])
            for k in range(3)
        ]
        steps.append(
            {
                "from": a,
                "to": b,
                "from_fracture": None if a.startswith("@") else tri_to_frac[a],
                "to_fracture": None if b.startswith("@") else tri_to_frac[b],
                "t": qs(t_q),
                "x": [qs(c) for c in point],
                "x_float": [round(float(c), 6) for c in point],
                "bary_left": list(w["left_weights"]),
                "bary_right": list(w["right_weights"]),
            }
        )
    return steps


def _format_witness(
    cell: Dict[str, Any],
    payload: Dict[str, Any],
    tri_to_frac: Dict[str, str],
    cn: Any,
    verify_pair: Any,
    *,
    t_override: Optional[Q] = None,
    include_contact_steps: bool = True,
) -> Dict[str, Any]:
    t_q = t_override if t_override is not None else q_exact(cell["sample"])
    mode_obj = next(m for m in payload["modes"] if m["id"] == cell["mode"])
    base: Dict[str, Any] = {
        "mode": cell["mode"],
        "cell_kind": cell["kind"],
        "lo": cell["lo"],
        "hi": cell["hi"],
        "t": qs(t_q),
        "t_float": round(float(t_q), 8),
        "connected": cell["connected"],
        "active_edges": cell["active_edges"],
        "components": cell["components"],
    }
    if cell["connected"]:
        base["path"] = cell["path"]
        base["fracture_path"] = _fracture_path_from_tri_path(cell["path"], tri_to_frac)
        base["cut"] = None
        base["fracture_cut"] = None
        if include_contact_steps:
            base["contact_step_witnesses"] = _compute_contact_step_witnesses(
                payload, mode_obj, cell["path"], t_q, tri_to_frac, cn, verify_pair
            )
    else:
        base["path"] = None
        base["fracture_path"] = None
        base["cut"] = cell["cut"]
        base["fracture_cut"] = _fracture_cut_from_tri_cut(cell["cut"], tri_to_frac)
        base["contact_step_witnesses"] = None
    return base


def _cell_measure(cell: Dict[str, Any]) -> Q:
    if cell["kind"] == "POINT":
        return ZERO
    return q_exact(cell["hi"]) - q_exact(cell["lo"])


def _select_canonical_witnesses(
    model: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    rho: Optional[Dict[str, Any]],
) -> Tuple[Optional[Tuple[Dict[str, Any], Q]], Optional[Tuple[Dict[str, Any], Q]]]:
    """Select canonical (connected_cell, t) and (disconnected_cell, t) witnesses.

    For AMBIGUOUS cases, prefer the exact nominal cell for its side and the exact
    nearest flip witness cell for the opposite side.
    """
    ref = model.get("reference")
    nominal_cell = None
    t0 = None
    if ref is not None:
        t0 = q_exact(ref["t"])
        nominal_cell = next(
            (
                r
                for r in records
                if r["mode"] == ref["mode"]
                and (
                    (r["kind"] == "POINT" and q_exact(r["lo"]) == t0)
                    or (r["kind"] == "OPEN" and q_exact(r["lo"]) < t0 < q_exact(r["hi"]))
                )
            ),
            None,
        )

    flip_cell = None
    flip_t = None
    if rho is not None and rho.get("flip_witness") is not None:
        fw = rho["flip_witness"]
        flip_t = q_exact(fw["t"])
        flip_cell = next(
            (
                r
                for r in records
                if r["mode"] == fw["mode"]
                and (
                    (r["kind"] == "POINT" and q_exact(r["lo"]) == flip_t)
                    or (r["kind"] == "OPEN" and q_exact(r["lo"]) < flip_t < q_exact(r["hi"]))
                )
            ),
            None,
        )

    conn_pick: Optional[Tuple[Dict[str, Any], Q]] = None
    disc_pick: Optional[Tuple[Dict[str, Any], Q]] = None

    if nominal_cell is not None and t0 is not None:
        if nominal_cell["connected"]:
            conn_pick = (nominal_cell, t0)
        else:
            disc_pick = (nominal_cell, t0)
    if flip_cell is not None and flip_t is not None:
        if flip_cell["connected"] and conn_pick is None:
            conn_pick = (flip_cell, flip_t)
        elif (not flip_cell["connected"]) and disc_pick is None:
            disc_pick = (flip_cell, flip_t)

    if conn_pick is None:
        first_c = next((r for r in records if r["connected"]), None)
        if first_c is not None:
            conn_pick = (first_c, q_exact(first_c["sample"]))
    if disc_pick is None:
        first_d = next((r for r in records if not r["connected"]), None)
        if first_d is not None:
            disc_pick = (first_d, q_exact(first_d["sample"]))
    return conn_pick, disc_pick


def _compute_naive_union_diagnostics(
    model: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    tri_to_frac: Dict[str, str],
) -> Dict[str, Any]:
    """Compute naive cross-(m,t) union graph and per-mode cross-t union graph diagnostics."""
    all_tri_ids = sorted(tri_to_frac.keys())
    global_edges: set[Tuple[str, str]] = set()
    per_mode_edges: Dict[str, set[Tuple[str, str]]] = {m["id"]: set() for m in model["modes"]}
    for r in records:
        mid = r["mode"]
        for a, b in r["active_edges"]:
            edge = (a, b) if a < b else (b, a)
            global_edges.add(edge)
            per_mode_edges[mid].add(edge)

    global_union = classify_slice_graph(all_tri_ids, sorted(global_edges))
    per_mode_union: Dict[str, bool] = {}
    for m in model["modes"]:
        mid = m["id"]
        m_tri_ids = sorted(t["id"] for t in m["triangles"])
        m_res = classify_slice_graph(m_tri_ids, sorted(per_mode_edges[mid]))
        per_mode_union[mid] = m_res["connected"]

    exists_conn = any(r["connected"] for r in records)
    return {
        "naive_union_connected": global_union["connected"],
        "naive_union_path": global_union["path"],
        "naive_union_fracture_path": (
            _fracture_path_from_tri_path(global_union["path"], tri_to_frac)
            if global_union["connected"] and global_union["path"] is not None
            else None
        ),
        "phantom_union_path": bool(global_union["connected"] and (not exists_conn)),
        "per_mode_union_connected": per_mode_union,
    }


# ---------------------------------------------------------------------------
# Core solver entry points: full real model & conditioned subdomain
# ---------------------------------------------------------------------------


def _refusal_envelope(
    *,
    model: Any,
    cluster_id: Optional[str],
    model_path: Optional[str],
    model_file_sha: Optional[str],
    status: str,
    tc_compat_status: str,
    reason: str,
    budget_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    is_dict = isinstance(model, dict)
    prov = model.get("real_provenance") if is_dict else None
    site_id = prov.get("site_id") if isinstance(prov, dict) else None
    stratum = prov.get("stratum") if isinstance(prov, dict) else None
    try:
        in_sha = canonical_sha256(model) if is_dict else None
    except (TypeError, ValueError):
        in_sha = None
    return {
        "schema": CERT_SCHEMA,
        "scope": model.get("scope") if is_dict else None,
        "cluster_id": cluster_id or (prov.get("cluster_id") if isinstance(prov, dict) else None),
        "site_id": site_id,
        "stratum": stratum,
        "snapshot_id": model.get("snapshot_id") if is_dict else None,
        "input_sha256": in_sha,
        "model_file_sha256": model_file_sha,
        "model_path": model_path,
        "region_origin": model.get("region_origin") if is_dict else None,
        "observation_support": "REFUSED_NO_CERTIFICATE",
        "field_certificate": False,
        "status": status,
        "tc_compat_status": tc_compat_status,
        "search_complete": False,
        "reason": reason,
        "exists_connected": None,
        "all_connected": None,
        "enumeration": None,
        "budget_telemetry": budget_telemetry,
        "measure_summary": None,
        "nominal_summary": None,
        "per_mode_summary": None,
        "rho": None,
        "rho_infimum_float": None,
        "nearest_flip_witness": None,
        "connected_witness": None,
        "disconnected_witness": None,
        "triangle_to_fracture": None,
        "fracture_ids": None,
        "cells": [],
        "contact_windows": [],
        "not_claimed": list(NOT_CLAIMED_STANDARD),
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
    }


def solve_real_model(
    model: Any,
    *,
    cluster_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_file_sha: Optional[str] = None,
    repo_root: Optional[Path] = REPO_ROOT,
    verify_source_sha: bool = False,
    require_non_static: bool = True,
    use_exact_shortcuts: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]]]]:
    """Validate and solve a real-conditioned model with exact rational arithmetic.

    Returns (certification_dict, windows_by_mode_or_None).
    """
    rnf, cn, cells_mod, compile_pair, verify_pair = load_frozen_deps()

    # Step 1: Admission validation via M1 real_network_format
    try:
        val_summary = rnf.validate(
            model,
            repo_root=repo_root,
            verify_source_sha=verify_source_sha,
            require_non_static=require_non_static,
        )
    except rnf.Exhausted as exc:
        return (
            _refusal_envelope(
                model=model,
                cluster_id=cluster_id,
                model_path=model_path,
                model_file_sha=model_file_sha,
                status="BUDGET_EXCEEDED",
                tc_compat_status="SEARCH_UNRESOLVED",
                reason=str(exc),
            ),
            None,
        )
    except rnf.Outside as exc:
        return (
            _refusal_envelope(
                model=model,
                cluster_id=cluster_id,
                model_path=model_path,
                model_file_sha=model_file_sha,
                status="REFUSED",
                tc_compat_status="OUT_OF_SCOPE",
                reason=str(exc),
            ),
            None,
        )
    except (rnf.Invalid, ValueError, TypeError, KeyError) as exc:
        return (
            _refusal_envelope(
                model=model,
                cluster_id=cluster_id,
                model_path=model_path,
                model_file_sha=model_file_sha,
                status="REFUSED",
                tc_compat_status="INVALID_INPUT",
                reason=str(exc),
            ),
            None,
        )

    input_sha = val_summary["input_sha256"]
    budget = val_summary["budget"]
    prov = model["real_provenance"]
    cid = cluster_id or prov.get("cluster_id") or model["snapshot_id"]

    # Check non-empty feasible segments
    if not any(mode["segments"] for mode in model["modes"]):
        return (
            _refusal_envelope(
                model=model,
                cluster_id=cid,
                model_path=model_path,
                model_file_sha=model_file_sha,
                status="INFEASIBLE",
                tc_compat_status="INFEASIBLE_OBSERVATIONS",
                reason="empty declared feasible set across all modes",
            ),
            None,
        )

    # Step 2: Lossless in-memory projection onto frozen TC kernel contract (never persisted)
    payload = rnf.to_core_kernel_payload(model, repo_root=repo_root, verify_source_sha=False)
    tri_to_frac = _build_tri_to_fracture(model)

    windows_flat: List[Dict[str, Any]] = []
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]] = {}
    records: List[Dict[str, Any]] = []
    pair_cache: Dict[Any, Optional[Tuple[Q, Q]]] = {}
    pairs_done = 0
    modes_evaluated = 0

    for mode in payload["modes"]:
        if not mode["segments"]:
            windows_by_mode[mode["id"]] = []
            continue
        modes_evaluated += 1
        items = cn.nodes(payload, mode)
        tri_ids = [t["id"] for t in mode["triangles"]]
        parsed_items = [_parse_item_exact(it) for it in items]
        dom = (q_exact(mode["t_closed"][0]), q_exact(mode["t_closed"][1]))
        dom_key = (qs(dom[0]), qs(dom[1]))
        mode_windows: List[Tuple[str, str, Optional[Tuple[Q, Q]]]] = []

        for idx_l, idx_r in itertools.combinations(range(len(items)), 2):
            left, right = items[idx_l], items[idx_r]
            if left["id"].startswith("@") and right["id"].startswith("@"):
                continue
            if pairs_done >= budget["pairs"]:
                return (
                    _refusal_envelope(
                        model=model,
                        cluster_id=cid,
                        model_path=model_path,
                        model_file_sha=model_file_sha,
                        status="BUDGET_EXCEEDED",
                        tc_compat_status="SEARCH_UNRESOLVED",
                        reason=f"pair work budget exhausted ({pairs_done} >= {budget['pairs']}) in mode {mode['id']}",
                        budget_telemetry={
                            "limit_kind": "pairs",
                            "declared_budget_pairs": budget["pairs"],
                            "declared_budget_cells": budget["cells"],
                            "pairs_done_at_halt": pairs_done,
                            "cells_done_at_halt": len(records),
                            "halted_in_mode": mode["id"],
                        },
                    ),
                    None,
                )
            pairs_done += 1
            try:
                interval = compute_pair_contact_window(
                    payload,
                    mode,
                    left,
                    right,
                    parsed_items[idx_l],
                    parsed_items[idx_r],
                    dom,
                    dom_key,
                    cn,
                    compile_pair,
                    pair_cache,
                    use_exact_shortcuts=use_exact_shortcuts,
                )
            except Exception as exc:
                if type(exc).__name__ == "Exhausted":
                    return (
                        _refusal_envelope(
                            model=model,
                            cluster_id=cid,
                            model_path=model_path,
                            model_file_sha=model_file_sha,
                            status="BUDGET_EXCEEDED",
                            tc_compat_status="SEARCH_UNRESOLVED",
                            reason=str(exc),
                        ),
                        None,
                    )
                return (
                    _refusal_envelope(
                        model=model,
                        cluster_id=cid,
                        model_path=model_path,
                        model_file_sha=model_file_sha,
                        status="REFUSED",
                        tc_compat_status="INVALID_INPUT",
                        reason=str(exc),
                    ),
                    None,
                )
            a, b = (left["id"], right["id"]) if left["id"] < right["id"] else (right["id"], left["id"])
            mode_windows.append((a, b, interval))
            windows_flat.append(
                {
                    "mode": mode["id"],
                    "from": a,
                    "to": b,
                    "t_closed": None if interval is None else [qs(interval[0]), qs(interval[1])],
                }
            )
        windows_by_mode[mode["id"]] = mode_windows

        # Exact POINT/OPEN cell partition under declared cell budget
        remaining_cells = budget["cells"] - len(records)
        try:
            mode_cells = cells_mod.partition(mode, [row[2] for row in mode_windows], remaining_cells)
        except Exception as exc:
            if type(exc).__name__ == "Exhausted":
                # Also record exact uncapped cell count of this mode for diagnostic telemetry
                uncapped_mode_cells = len(cells_mod.partition(mode, [row[2] for row in mode_windows], 1_000_000))
                return (
                    _refusal_envelope(
                        model=model,
                        cluster_id=cid,
                        model_path=model_path,
                        model_file_sha=model_file_sha,
                        status="BUDGET_EXCEEDED",
                        tc_compat_status="SEARCH_UNRESOLVED",
                        reason=(
                            f"cell work budget exhausted during partition in mode {mode['id']} "
                            f"(accumulated {len(records)} + mode_cells {uncapped_mode_cells} > {budget['cells']})"
                        ),
                        budget_telemetry={
                            "limit_kind": "cells",
                            "declared_budget_cells": budget["cells"],
                            "declared_budget_pairs": budget["pairs"],
                            "pairs_done_at_halt": pairs_done,
                            "cells_accumulated_before_mode": len(records),
                            "mode_cells_required": uncapped_mode_cells,
                            "halted_in_mode": mode["id"],
                        },
                    ),
                    None,
                )
            raise

        for cell in mode_cells:
            if len(records) >= budget["cells"]:
                return (
                    _refusal_envelope(
                        model=model,
                        cluster_id=cid,
                        model_path=model_path,
                        model_file_sha=model_file_sha,
                        status="BUDGET_EXCEEDED",
                        tc_compat_status="SEARCH_UNRESOLVED",
                        reason=f"cell work budget exhausted ({len(records)} >= {budget['cells']})",
                        budget_telemetry={
                            "limit_kind": "cells",
                            "declared_budget_cells": budget["cells"],
                            "declared_budget_pairs": budget["pairs"],
                            "pairs_done_at_halt": pairs_done,
                            "cells_accumulated_before_mode": len(records),
                            "halted_in_mode": mode["id"],
                        },
                    ),
                    None,
                )
            t_q = q_exact(cell["sample"])
            active_edges = [(a, b) for a, b, iv in mode_windows if cells_mod.active(iv, t_q)]
            rec = {"mode": mode["id"], **cell, **classify_slice_graph(tri_ids, active_edges)}
            records.append(rec)

    if not records:
        return (
            _refusal_envelope(
                model=model,
                cluster_id=cid,
                model_path=model_path,
                model_file_sha=model_file_sha,
                status="INFEASIBLE",
                tc_compat_status="INFEASIBLE_OBSERVATIONS",
                reason="nonempty feasible set produced zero cells",
            ),
            None,
        )

    # Step 3: Exact 3-state verdict, stability margin rho, dual witnesses, and baseline diagnostics
    labels = {r["connected"] for r in records}
    exists_conn = True in labels
    all_conn = labels == {True}
    if exists_conn and not all_conn:
        status = "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
        tc_status = "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
    elif all_conn:
        status = "CERTIFIED_CONNECTED"
        tc_status = "CERTIFIED_WITHIN_DECLARED_MODEL"
    else:
        status = "CERTIFIED_DISCONNECTED"
        tc_status = "CERTIFIED_WITHIN_DECLARED_MODEL"

    rho = compute_exact_stability(model, records)
    conn_pick, disc_pick = _select_canonical_witnesses(model, records, rho)
    connected_witness = (
        _format_witness(conn_pick[0], payload, tri_to_frac, cn, verify_pair, t_override=conn_pick[1])
        if conn_pick is not None
        else None
    )
    disconnected_witness = (
        _format_witness(disc_pick[0], payload, tri_to_frac, cn, verify_pair, t_override=disc_pick[1])
        if disc_pick is not None
        else None
    )

    nearest_flip_witness = None
    if rho is not None and rho.get("flip_witness") is not None:
        fw = rho["flip_witness"]
        fw_t = q_exact(fw["t"])
        fw_cell = next(
            r
            for r in records
            if r["mode"] == fw["mode"]
            and (
                (r["kind"] == "POINT" and q_exact(r["lo"]) == fw_t)
                or (r["kind"] == "OPEN" and q_exact(r["lo"]) < fw_t < q_exact(r["hi"]))
            )
        )
        nearest_flip_witness = {
            **_format_witness(
                fw_cell,
                payload,
                tri_to_frac,
                cn,
                verify_pair,
                t_override=fw_t,
                include_contact_steps=fw_cell["connected"],
            ),
            "cost": fw["cost"],
            "cost_float": round(float(q_exact(fw["cost"])), 8),
        }

    # Exact rational measure statistics
    total_meas = sum((_cell_measure(r) for r in records), ZERO)
    conn_meas = sum((_cell_measure(r) for r in records if r["connected"]), ZERO)
    disc_meas = total_meas - conn_meas
    conn_cells = sum(1 for r in records if r["connected"])
    disc_cells = len(records) - conn_cells
    point_cells = sum(1 for r in records if r["kind"] == "POINT")
    open_cells = len(records) - point_cells

    per_mode_summary: Dict[str, Any] = {}
    for m in model["modes"]:
        mid = m["id"]
        m_recs = [r for r in records if r["mode"] == mid]
        if not m_recs:
            per_mode_summary[mid] = {"status": "INFEASIBLE", "cells": 0}
            continue
        m_labels = {r["connected"] for r in m_recs}
        m_st = (
            "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
            if len(m_labels) > 1
            else ("CERTIFIED_CONNECTED" if True in m_labels else "CERTIFIED_DISCONNECTED")
        )
        m_tot = sum((_cell_measure(r) for r in m_recs), ZERO)
        m_con = sum((_cell_measure(r) for r in m_recs if r["connected"]), ZERO)
        per_mode_summary[mid] = {
            "status": m_st,
            "cells": len(m_recs),
            "connected_cells": sum(1 for r in m_recs if r["connected"]),
            "disconnected_cells": sum(1 for r in m_recs if not r["connected"]),
            "feasible_measure": qs(m_tot),
            "connected_measure": qs(m_con),
            "disconnected_measure": qs(m_tot - m_con),
        }

    union_diag = _compute_naive_union_diagnostics(model, records, tri_to_frac)
    nominal_conn = rho["nominal_connected"] if rho is not None else None
    nominal_summary = {
        "mode": model["reference"]["mode"],
        "t": qs(q_exact(model["reference"]["t"])),
        "connected": nominal_conn,
        "silent_flip_under_declared_model": bool(status == "AMBIGUOUS_WITH_VERIFIED_WITNESSES"),
        **union_diag,
    }

    cert: Dict[str, Any] = {
        "schema": CERT_SCHEMA,
        "scope": REAL_SCOPE,
        "cluster_id": cid,
        "site_id": prov["site_id"],
        "stratum": prov.get("stratum"),
        "snapshot_id": model["snapshot_id"],
        "input_sha256": input_sha,
        "model_file_sha256": model_file_sha,
        "model_path": model_path,
        "region_origin": model["region_origin"],
        "observation_support": "CONDITIONED_OUTCROP_MODEL_NOT_SUBSURFACE_TRUTH",
        "field_certificate": False,
        "status": status,
        "tc_compat_status": tc_status,
        "search_complete": True,
        "reason": None,
        "exists_connected": exists_conn,
        "all_connected": all_conn,
        "enumeration": {
            "pairs": pairs_done,
            "cells": len(records),
            "point_cells": point_cells,
            "open_cells": open_cells,
            "modes_evaluated": modes_evaluated,
            "budget_cells": budget["cells"],
            "budget_pairs": budget["pairs"],
        },
        "budget_telemetry": None,
        "measure_summary": {
            "total_feasible_measure": qs(total_meas),
            "total_feasible_measure_float": round(float(total_meas), 8),
            "connected_measure": qs(conn_meas),
            "disconnected_measure": qs(disc_meas),
            "connected_measure_fraction": qs(conn_meas / total_meas) if total_meas > ZERO else None,
            "connected_measure_fraction_float": round(float(conn_meas / total_meas), 8) if total_meas > ZERO else None,
            "connected_cell_count": conn_cells,
            "disconnected_cell_count": disc_cells,
        },
        "nominal_summary": nominal_summary,
        "per_mode_summary": per_mode_summary,
        "rho": rho,
        "rho_infimum_float": (
            round(float(q_exact(rho["infimum"])), 8) if rho is not None and rho["infimum"] is not None else None
        ),
        "nearest_flip_witness": nearest_flip_witness,
        "connected_witness": connected_witness,
        "disconnected_witness": disconnected_witness,
        "triangle_to_fracture": tri_to_frac,
        "fracture_ids": sorted(model["fracture_catalog"].keys()) if model.get("fracture_catalog") else sorted(set(tri_to_frac.values())),
        "cells": records,
        "contact_windows": windows_flat,
        "assumption_register": model["assumption_register"],
        "real_provenance": model["real_provenance"],
        "quantization_contract": model["quantization_contract"],
        "not_claimed": list(NOT_CLAIMED_STANDARD),
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
    }
    return cert, windows_by_mode


class CellLatticeIndex:
    """Exact cell-restriction index over a certified cluster's POINT/OPEN cell lattice.

    By exact cell constancy (no contact window endpoint lies strictly inside any OPEN cell),
    restricting the precomputed cell lattice to any subdomain D(o) produces the EXACT SAME
    cells_count, total_measure, exists_connected, all_connected, and status as re-running
    `cells.partition()` + `classify_slice_graph()`, in O(|cells|) exact Fraction operations.
    """

    def __init__(self, cells: Sequence[Dict[str, Any]]) -> None:
        self.by_mode: Dict[str, List[Tuple[str, Q, Q, bool]]] = {}
        for c in cells:
            mid = c["mode"]
            self.by_mode.setdefault(mid, []).append(
                (c["kind"], q_exact(c["lo"]), q_exact(c["hi"]), bool(c["connected"]))
            )
        for mid in self.by_mode:
            self.by_mode[mid].sort(key=lambda r: (r[1], 0 if r[0] == "POINT" else 1, r[2]))

    def evaluate_subdomain(
        self,
        allowed_modes: Sequence[str],
        mode_segments: Dict[str, List[Tuple[Q, Q, bool, bool]]],
    ) -> Dict[str, Any]:
        cells_count = 0
        total_meas = ZERO
        conn_meas = ZERO
        exists_conn = False
        exists_disc = False
        active_modes: List[str] = []

        for mid in sorted(allowed_modes):
            segs = mode_segments.get(mid, [])
            if not segs:
                continue
            mode_cells = self.by_mode.get(mid, [])
            if not mode_cells:
                continue
            seg_min = segs[0][0]
            seg_max = segs[-1][1]
            seg_pts = sorted({pt for s_lo, s_hi, _, _ in segs for pt in (s_lo, s_hi)})
            mode_hit = False

            for kind, lo, hi, conn in mode_cells:
                if hi < seg_min or lo > seg_max:
                    continue
                if kind == "POINT":
                    if admitted_in_intervals(lo, segs):
                        mode_hit = True
                        cells_count += 1
                        if conn:
                            exists_conn = True
                        else:
                            exists_disc = True
                else:
                    # OPEN (lo, hi): split only at segment boundary points strictly inside (lo, hi)
                    inner = [p for p in seg_pts if lo < p < hi]
                    if not inner:
                        if admitted_in_intervals((lo + hi) / 2, segs):
                            mode_hit = True
                            cells_count += 1
                            dm = hi - lo
                            total_meas += dm
                            if conn:
                                exists_conn = True
                                conn_meas += dm
                            else:
                                exists_disc = True
                    else:
                        pts = [lo] + inner + [hi]
                        for i in range(len(pts) - 1):
                            a, b = pts[i], pts[i + 1]
                            if i > 0 and admitted_in_intervals(a, segs):
                                mode_hit = True
                                cells_count += 1
                                if conn:
                                    exists_conn = True
                                else:
                                    exists_disc = True
                            if admitted_in_intervals((a + b) / 2, segs):
                                mode_hit = True
                                cells_count += 1
                                dm = b - a
                                total_meas += dm
                                if conn:
                                    exists_conn = True
                                    conn_meas += dm
                                else:
                                    exists_disc = True
            if mode_hit:
                active_modes.append(mid)

        if cells_count == 0:
            return {
                "status": "INFEASIBLE",
                "search_complete": False,
                "exists_connected": None,
                "all_connected": None,
                "cells_count": 0,
                "active_modes": [],
                "total_measure": ZERO,
                "connected_measure": ZERO,
                "disconnected_measure": ZERO,
            }

        all_conn = exists_conn and (not exists_disc)
        if exists_conn and exists_disc:
            status = "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
        elif all_conn:
            status = "CERTIFIED_CONNECTED"
        else:
            status = "CERTIFIED_DISCONNECTED"

        return {
            "status": status,
            "search_complete": True,
            "exists_connected": exists_conn,
            "all_connected": all_conn,
            "cells_count": cells_count,
            "active_modes": active_modes,
            "total_measure": total_meas,
            "connected_measure": conn_meas,
            "disconnected_measure": total_meas - conn_meas,
        }


def solve_conditioned_subdomain(
    model: Dict[str, Any],
    windows_by_mode: Dict[str, List[Tuple[str, str, Optional[Tuple[Q, Q]]]]],
    allowed_modes: Sequence[str],
    mode_segments: Dict[str, List[Tuple[Q, Q, bool, bool]]],
    *,
    include_witnesses: bool = True,
    include_contact_steps: bool = False,
) -> Dict[str, Any]:
    """Genuinely re-run exact cell partition & slice graph classification on a conditioned subdomain D(o).

    Uses exact precomputed mode contact windows (mathematically invariant under domain restriction)
    and exact `cells.partition()` + `classify_slice_graph()`.
    """
    rnf, cn, cells_mod, _, verify_pair = load_frozen_deps()
    tri_to_frac = _build_tri_to_fracture(model)
    modes_by_id = {m["id"]: m for m in model["modes"]}
    budget_cells = model.get("budget", {"cells": 4096})["cells"]

    records: List[Dict[str, Any]] = []
    active_modes: List[str] = []
    for mid in sorted(allowed_modes):
        segs = mode_segments.get(mid, [])
        if not segs:
            continue
        active_modes.append(mid)
        mode_orig = modes_by_id[mid]
        cond_mode = {
            "id": mid,
            "t_closed": mode_orig["t_closed"],
            "segments": segments_to_json(segs),
        }
        mode_windows = windows_by_mode[mid]
        tri_ids = [t["id"] for t in mode_orig["triangles"]]
        mode_cells = cells_mod.partition(cond_mode, [row[2] for row in mode_windows], budget_cells - len(records))
        for cell in mode_cells:
            t_q = q_exact(cell["sample"])
            active_edges = [(a, b) for a, b, iv in mode_windows if cells_mod.active(iv, t_q)]
            records.append({"mode": mid, **cell, **classify_slice_graph(tri_ids, active_edges)})

    if not records:
        return {
            "status": "INFEASIBLE",
            "search_complete": False,
            "exists_connected": None,
            "all_connected": None,
            "cells": [],
            "cells_count": 0,
            "active_modes": [],
            "total_measure": ZERO,
            "connected_measure": ZERO,
            "disconnected_measure": ZERO,
            "connected_witness": None,
            "disconnected_witness": None,
        }

    labels = {r["connected"] for r in records}
    exists_conn = True in labels
    all_conn = labels == {True}
    if exists_conn and not all_conn:
        status = "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
    elif all_conn:
        status = "CERTIFIED_CONNECTED"
    else:
        status = "CERTIFIED_DISCONNECTED"

    total_meas = sum((_cell_measure(r) for r in records), ZERO)
    conn_meas = sum((_cell_measure(r) for r in records if r["connected"]), ZERO)
    disc_meas = total_meas - conn_meas

    connected_witness = None
    disconnected_witness = None
    if include_witnesses:
        payload = rnf.to_core_kernel_payload(model, verify_source_sha=False)
        first_c = next((r for r in records if r["connected"]), None)
        first_d = next((r for r in records if not r["connected"]), None)
        if first_c is not None:
            connected_witness = _format_witness(
                first_c,
                payload,
                tri_to_frac,
                cn,
                verify_pair,
                include_contact_steps=include_contact_steps,
            )
        if first_d is not None:
            disconnected_witness = _format_witness(
                first_d,
                payload,
                tri_to_frac,
                cn,
                verify_pair,
                include_contact_steps=False,
            )

    return {
        "status": status,
        "search_complete": True,
        "exists_connected": exists_conn,
        "all_connected": all_conn,
        "cells": records,
        "cells_count": len(records),
        "active_modes": active_modes,
        "total_measure": total_meas,
        "connected_measure": conn_meas,
        "disconnected_measure": disc_meas,
        "connected_witness": connected_witness,
        "disconnected_witness": disconnected_witness,
    }


def recompute_conditioned_from_scratch(
    model: Dict[str, Any],
    allowed_modes: Sequence[str],
    mode_segments_json: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build an independent conditioned model dict and run the full exact solver from scratch.

    Used by verify_m2.py and M2-N5 to independently cross-check conditioned branch results
    without reusing any precomputed window or cell cache.
    """
    sub = copy.deepcopy(model)
    allowed_set = set(allowed_modes)
    new_modes = []
    for m in sub["modes"]:
        if m["id"] in allowed_set and mode_segments_json.get(m["id"]):
            m_copy = copy.deepcopy(m)
            m_copy["segments"] = copy.deepcopy(mode_segments_json[m["id"]])
            new_modes.append(m_copy)
    if not new_modes:
        return {"status": "INFEASIBLE", "search_complete": False, "cells_count": 0}
    sub["modes"] = new_modes
    if sub.get("mode_semantics") is not None:
        sub["mode_semantics"] = {m["id"]: sub["mode_semantics"][m["id"]] for m in new_modes}
    # Reference may fall outside the conditioned subdomain; solve via core kernel directly
    rnf, cn, cells_mod, compile_pair, _ = load_frozen_deps()
    # Validate geometry/scope on the original model first
    rnf.validate(model, verify_source_sha=False)
    payload = rnf.to_core_kernel_payload(model, verify_source_sha=False)
    payload["modes"] = new_modes
    payload["reference"] = None
    # Run exact solver without sharing any cache with caller
    records = []
    pair_cache: Dict[Any, Optional[Tuple[Q, Q]]] = {}
    for mode in payload["modes"]:
        items = cn.nodes(payload, mode)
        tri_ids = [t["id"] for t in mode["triangles"]]
        parsed_items = [_parse_item_exact(it) for it in items]
        dom = (q_exact(mode["t_closed"][0]), q_exact(mode["t_closed"][1]))
        dom_key = (qs(dom[0]), qs(dom[1]))
        mw = []
        for idx_l, idx_r in itertools.combinations(range(len(items)), 2):
            l, r = items[idx_l], items[idx_r]
            if l["id"].startswith("@") and r["id"].startswith("@"):
                continue
            iv = compute_pair_contact_window(
                payload, mode, l, r, parsed_items[idx_l], parsed_items[idx_r], dom, dom_key, cn, compile_pair, pair_cache
            )
            a, b = (l["id"], r["id"]) if l["id"] < r["id"] else (r["id"], l["id"])
            mw.append((a, b, iv))
        for cell in cells_mod.partition(mode, [row[2] for row in mw], payload["budget"]["cells"] - len(records)):
            t_q = q_exact(cell["sample"])
            edges = [(a, b) for a, b, iv in mw if cells_mod.active(iv, t_q)]
            records.append({"mode": mode["id"], **cell, **classify_slice_graph(tri_ids, edges)})
    labels = {r["connected"] for r in records}
    st = (
        "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
        if len(labels) > 1
        else ("CERTIFIED_CONNECTED" if True in labels else "CERTIFIED_DISCONNECTED")
    )
    return {
        "status": st,
        "search_complete": True,
        "exists_connected": True in labels,
        "all_connected": labels == {True},
        "cells_count": len(records),
    }


# ---------------------------------------------------------------------------
# Local certificate self-check (catches tampered witnesses, wrong status, etc.)
# ---------------------------------------------------------------------------


def verify_certification_local(
    model: Dict[str, Any],
    cert: Dict[str, Any],
    *,
    recheck_windows_from_scratch: bool = False,
) -> Dict[str, Any]:
    """Strict local certificate verifier for `m2-real-certification/v1`."""
    rnf, cn, cells_mod, compile_pair, verify_pair = load_frozen_deps()
    errors: List[str] = []

    def fail(msg: str) -> Dict[str, Any]:
        return {"ok": False, "status": "CERTIFICATE_INVALID", "reason": msg}

    try:
        if not isinstance(cert, dict) or cert.get("schema") != CERT_SCHEMA:
            return fail("invalid certification schema")
        val = rnf.validate(model, verify_source_sha=False)
        if cert.get("scope") != REAL_SCOPE:
            return fail(f"wrong scope: {cert.get('scope')!r}")
        if cert.get("input_sha256") != val["input_sha256"]:
            return fail("input_sha256 mismatch against model.json")
        if cert.get("field_certificate") is not False:
            return fail("field_certificate must be false")
        if cert.get("field_error_rate") is not None or cert.get("human_time") is not None or cert.get("hydraulic") is not None:
            return fail("empty-denominator metrics must be null")

        if not cert.get("search_complete"):
            if cert.get("status") not in ("BUDGET_EXCEEDED", "INFEASIBLE", "REFUSED"):
                return fail(f"incomplete cert has non-refusal status: {cert.get('status')!r}")
            if cert.get("cells") or cert.get("contact_windows"):
                return fail("refusal envelope must not carry cells or contact_windows")
            if cert.get("exists_connected") is not None or cert.get("all_connected") is not None or cert.get("rho") is not None:
                return fail("refusal envelope must keep exists/all/rho null")
            return {"ok": True, "status": "VERIFIED_REFUSAL_ENVELOPE", "reason": cert.get("reason")}

        # Complete certificate checks
        cells = cert.get("cells")
        windows = cert.get("contact_windows")
        if not isinstance(cells, list) or not cells:
            return fail("complete cert requires nonempty cells list")
        if not isinstance(windows, list) or not windows:
            return fail("complete cert requires nonempty contact_windows list")

        modes_by_id = {m["id"]: m for m in model["modes"]}
        payload = rnf.to_core_kernel_payload(model, verify_source_sha=False)

        # Build mode feasible segment maps
        mode_feasible: Dict[str, List[Tuple[Q, Q, bool, bool]]] = {}
        for m in model["modes"]:
            dom = (q_exact(m["t_closed"][0]), q_exact(m["t_closed"][1]))
            mode_feasible[m["id"]] = normalize_segments(m["segments"], dom)

        # Build window lookup
        win_by_mode: Dict[str, Dict[Tuple[str, str], Optional[Tuple[Q, Q]]]] = {m["id"]: {} for m in model["modes"]}
        for w in windows:
            mid = w["mode"]
            if mid not in win_by_mode:
                return fail(f"unknown window mode {mid}")
            edge = (w["from"], w["to"]) if w["from"] < w["to"] else (w["to"], w["from"])
            if edge == ("@A", "@B"):
                return fail("forbidden direct @A--@B window")
            iv = None if w["t_closed"] is None else (q_exact(w["t_closed"][0]), q_exact(w["t_closed"][1]))
            if iv is not None and iv[0] > iv[1]:
                return fail(f"inverted window {mid} {edge}")
            win_by_mode[mid][edge] = iv

        # Optionally verify every contact window from scratch
        if recheck_windows_from_scratch:
            pair_cache: Dict[Any, Optional[Tuple[Q, Q]]] = {}
            for mode in payload["modes"]:
                if not mode["segments"]:
                    continue
                items = cn.nodes(payload, mode)
                parsed_items = [_parse_item_exact(it) for it in items]
                dom = (q_exact(mode["t_closed"][0]), q_exact(mode["t_closed"][1]))
                dom_key = (qs(dom[0]), qs(dom[1]))
                for il, ir in itertools.combinations(range(len(items)), 2):
                    l, r = items[il], items[ir]
                    if l["id"].startswith("@") and r["id"].startswith("@"):
                        continue
                    iv = compute_pair_contact_window(
                        payload, mode, l, r, parsed_items[il], parsed_items[ir], dom, dom_key, cn, compile_pair, pair_cache
                    )
                    edge = (l["id"], r["id"]) if l["id"] < r["id"] else (r["id"], l["id"])
                    if win_by_mode[mode["id"]].get(edge) != iv:
                        return fail(f"contact window mismatch on {mode['id']} {edge}")

        # Verify every cell's active graph, connectivity, path/cut, and segment admission
        cell_keys = set()
        for c in cells:
            mid = c["mode"]
            if mid not in modes_by_id:
                return fail(f"unknown cell mode {mid}")
            lo, hi, sample = q_exact(c["lo"]), q_exact(c["hi"]), q_exact(c["sample"])
            if c["kind"] == "POINT":
                if not (lo == hi == sample):
                    return fail(f"invalid POINT cell {mid} {lo} {hi} {sample}")
            elif c["kind"] == "OPEN":
                if not (lo < sample < hi):
                    return fail(f"invalid OPEN cell sample {mid} {lo} {hi} {sample}")
            else:
                return fail(f"invalid cell kind {c['kind']!r}")
            if not admitted_in_intervals(sample, mode_feasible[mid]):
                return fail(f"cell sample {qs(sample)} outside feasible segments of {mid}")
            key = (mid, c["kind"], lo, hi)
            if key in cell_keys:
                return fail(f"duplicate cell {key}")
            cell_keys.add(key)

            tri_ids = [t["id"] for t in modes_by_id[mid]["triangles"]]
            expected_edges = sorted(
                edge
                for edge, iv in win_by_mode[mid].items()
                if iv is not None and iv[0] <= sample <= iv[1]
            )
            claimed_edges = sorted((a, b) if a < b else (b, a) for a, b in c["active_edges"])
            if claimed_edges != expected_edges:
                return fail(f"cell active_edges mismatch at {mid} t={qs(sample)}")
            recls = classify_slice_graph(tri_ids, expected_edges)
            if recls["connected"] != c["connected"]:
                return fail(f"cell connected mismatch at {mid} t={qs(sample)}")
            if c["connected"]:
                path = c.get("path")
                if not isinstance(path, list) or len(path) < 3 or path[0] != "@A" or path[-1] != "@B":
                    return fail(f"invalid connected path at {mid} t={qs(sample)}")
                edge_set = set(expected_edges)
                for u, v in zip(path, path[1:]):
                    e = (u, v) if u < v else (v, u)
                    if e not in edge_set:
                        return fail(f"path uses inactive/cross-t edge {e} at {mid} t={qs(sample)}")
            else:
                cut = c.get("cut")
                if not isinstance(cut, dict) or set(cut) != {"S", "T", "unattached_endpoints"}:
                    return fail(f"invalid cut structure at {mid} t={qs(sample)}")
                s_set, t_set = set(cut["S"]), set(cut["T"])
                if (s_set & t_set) or (s_set | t_set != set(tri_ids)):
                    return fail(f"cut S/T does not partition triangles at {mid} t={qs(sample)}")
                for a, b in expected_edges:
                    if (a in s_set and b in t_set) or (b in s_set and a in t_set):
                        return fail(f"cut has crossing active edge {(a, b)} at {mid} t={qs(sample)}")

        # Verify status consistency
        labels = {c["connected"] for c in cells}
        exp_exists = True in labels
        exp_all = labels == {True}
        exp_status = (
            "AMBIGUOUS_WITH_VERIFIED_WITNESSES"
            if (exp_exists and not exp_all)
            else ("CERTIFIED_CONNECTED" if exp_all else "CERTIFIED_DISCONNECTED")
        )
        if cert.get("status") != exp_status or cert.get("exists_connected") != exp_exists or cert.get("all_connected") != exp_all:
            return fail(f"top-level status {cert.get('status')!r} disagrees with exact cells ({exp_status!r})")

        # Verify connected_witness & disconnected_witness independently against geometry at witness t!
        cw = cert.get("connected_witness")
        dw = cert.get("disconnected_witness")
        if exp_exists:
            if not isinstance(cw, dict):
                return fail("connected_witness required when exists_connected=True")
            w_mid = cw.get("mode")
            if w_mid not in modes_by_id:
                return fail(f"connected_witness unknown mode {w_mid!r}")
            w_t = q_exact(cw["t"])
            if not admitted_in_intervals(w_t, mode_feasible[w_mid]):
                return fail(f"connected_witness t={qs(w_t)} is NOT in feasible segments of mode {w_mid}")
            w_path = cw.get("path")
            if not isinstance(w_path, list) or len(w_path) < 3 or w_path[0] != "@A" or w_path[-1] != "@B":
                return fail("connected_witness path malformed")
            # Direct exact primal verification of EVERY hop at exact (w_mid, w_t)!
            mode_obj = next(m for m in payload["modes"] if m["id"] == w_mid)
            items_by_id = {p["id"]: p for p in cn.nodes(payload, mode_obj)}
            for a, b in zip(w_path, w_path[1:]):
                if a not in items_by_id or b not in items_by_id:
                    return fail(f"connected_witness references unknown node {a}/{b}")
                pm = cn.pair_model(payload, mode_obj, items_by_id[a], items_by_id[b], [w_t, w_t])
                if verify_pair.vertices(pm) is None:
                    return fail(f"connected_witness edge ({a}, {b}) is NOT in contact at ({w_mid}, t={qs(w_t)})")
        elif cw is not None:
            return fail("CERTIFIED_DISCONNECTED must not publish connected_witness")

        if not exp_all:
            if not isinstance(dw, dict):
                return fail("disconnected_witness required when all_connected=False")
            w_mid = dw.get("mode")
            if w_mid not in modes_by_id:
                return fail(f"disconnected_witness unknown mode {w_mid!r}")
            w_t = q_exact(dw["t"])
            if not admitted_in_intervals(w_t, mode_feasible[w_mid]):
                return fail(f"disconnected_witness t={qs(w_t)} is NOT in feasible segments of mode {w_mid}")
            # Verify actual graph at (w_mid, w_t) is disconnected and cut has no crossing active edges
            w_edges = sorted(
                edge
                for edge, iv in win_by_mode[w_mid].items()
                if iv is not None and iv[0] <= w_t <= iv[1]
            )
            tri_ids = [t["id"] for t in modes_by_id[w_mid]["triangles"]]
            if classify_slice_graph(tri_ids, w_edges)["connected"]:
                return fail(f"disconnected_witness at ({w_mid}, t={qs(w_t)}) is actually CONNECTED")
        elif dw is not None:
            return fail("CERTIFIED_CONNECTED must not publish disconnected_witness")

        # Verify rho
        exp_rho = compute_exact_stability(model, cells)
        if cert.get("rho") != exp_rho:
            return fail("rho disagrees with exact recomputation")

        return {
            "ok": True,
            "status": "VERIFIED_LOCAL_CERTIFICATION",
            "cells_verified": len(cells),
            "windows_verified": len(windows),
        }
    except Exception as exc:
        return fail(f"verification exception: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI: batch over M1 cluster_register.json or single model.json
# ---------------------------------------------------------------------------


def run_batch(
    register_path: Path,
    run_dir: Path,
    *,
    verify_source_sha: bool = True,
    cluster_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    reg = load_strict_json(register_path)
    reg_dir = register_path.parent
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    selected = reg["clusters"]
    if cluster_filter:
        wanted = set(cluster_filter)
        selected = [c for c in selected if c["cluster_id"] in wanted]

    results_index: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    site_counts: Dict[str, Dict[str, int]] = {}
    t_all = time.perf_counter()

    for entry in selected:
        cid = entry["cluster_id"]
        m_path = reg_dir / entry["model_path"]
        m_raw = load_strict_json(m_path)
        m_fsha = file_sha256(m_path)
        rel_model_path = str(m_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")

        t0 = time.perf_counter()
        cert, _ = solve_real_model(
            m_raw,
            cluster_id=cid,
            model_path=rel_model_path,
            model_file_sha=m_fsha,
            repo_root=REPO_ROOT,
            verify_source_sha=verify_source_sha,
        )
        elapsed = round(time.perf_counter() - t0, 3)

        # Cross-check against cluster_register SHA
        if cert["input_sha256"] != entry["model_canonical_sha256"]:
            raise RuntimeError(
                f"SHA mismatch vs cluster_register on {cid}: "
                f"{cert['input_sha256']} != {entry['model_canonical_sha256']}"
            )
        if m_fsha != entry["model_file_sha256"]:
            raise RuntimeError(
                f"file SHA mismatch vs cluster_register on {cid}: "
                f"{m_fsha} != {entry['model_file_sha256']}"
            )

        local_verdict = verify_certification_local(m_raw, cert, recheck_windows_from_scratch=False)
        if not local_verdict["ok"]:
            raise RuntimeError(f"local certificate check failed on {cid}: {local_verdict}")

        cert["solve_seconds"] = elapsed
        cert["local_recheck"] = local_verdict
        out_path = cases_dir / cid / "certification.json"
        write_json(out_path, cert)

        st = cert["status"]
        sid = entry["site_id"]
        status_counts[st] = status_counts.get(st, 0) + 1
        site_counts.setdefault(sid, {})
        site_counts[sid][st] = site_counts[sid].get(st, 0) + 1

        results_index.append(
            {
                "cluster_id": cid,
                "site_id": sid,
                "stratum": entry["stratum"],
                "status": st,
                "search_complete": cert["search_complete"],
                "input_sha256": cert["input_sha256"],
                "cells": cert["enumeration"]["cells"] if cert["enumeration"] else 0,
                "pairs": cert["enumeration"]["pairs"] if cert["enumeration"] else 0,
                "rho_infimum": cert["rho"]["infimum"] if cert["rho"] else None,
                "rho_infimum_float": cert["rho_infimum_float"],
                "silent_flip": cert["nominal_summary"]["silent_flip_under_declared_model"] if cert["nominal_summary"] else None,
                "phantom_union_path": cert["nominal_summary"]["phantom_union_path"] if cert["nominal_summary"] else None,
                "solve_seconds": elapsed,
                "certification_path": f"cases/{cid}/certification.json",
            }
        )
        print(
            f"[M2-SOLVE] {cid:12s} | {sid:21s} | {st:33s} | "
            f"cells={results_index[-1]['cells']:4d} | rho={str(cert['rho_infimum_float']):8s} | {elapsed:.2f}s",
            flush=True,
        )

    return {
        "total_clusters": len(results_index),
        "completed_clusters": sum(1 for r in results_index if r["search_complete"]),
        "refused_clusters": sum(1 for r in results_index if not r["search_complete"]),
        "status_counts_all": status_counts,
        "status_counts_by_site": site_counts,
        "total_solve_seconds": round(time.perf_counter() - t_all, 3),
        "clusters": results_index,
    }


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2 Exact Whole-Network Certification Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", help="Path to M1 cluster_register.json for batch solve")
    group.add_argument("--model", help="Path to a single model.json")
    group.add_argument("--verify-cert", nargs=2, metavar=("MODEL_JSON", "CERT_JSON"), help="Verify certification.json against model.json")
    parser.add_argument("--run-dir", help="Output run directory for batch mode")
    parser.add_argument("--out", help="Output JSON path for single-model mode")
    parser.add_argument("--no-verify-source-sha", action="store_true", help="Skip on-disk source SHA checks")
    parser.add_argument("--clusters", nargs="*", help="Optional subset of cluster_ids")
    args = parser.parse_args(argv)

    if args.verify_cert:
        m_path, c_path = Path(args.verify_cert[0]), Path(args.verify_cert[1])
        verdict = verify_certification_local(load_strict_json(m_path), load_strict_json(c_path))
        print(json.dumps(verdict, ensure_ascii=False))
        return 0 if verdict["ok"] else 2

    if args.model:
        m_path = Path(args.model)
        try:
            m_raw = load_strict_json(m_path)
            m_fsha = file_sha256(m_path)
        except Exception as exc:
            env = _refusal_envelope(
                model=None,
                cluster_id=None,
                model_path=str(m_path),
                model_file_sha=None,
                status="REFUSED",
                tc_compat_status="INVALID_INPUT",
                reason=f"failed to load JSON: {exc}",
            )
            if args.out:
                write_json(Path(args.out), env)
            print(json.dumps({"status": env["status"], "reason": env["reason"]}, ensure_ascii=False))
            return 2
        cert, _ = solve_real_model(
            m_raw,
            model_path=str(m_path),
            model_file_sha=m_fsha,
            repo_root=REPO_ROOT,
            verify_source_sha=not args.no_verify_source_sha,
        )
        if args.out:
            write_json(Path(args.out), cert)
        print(json.dumps({"status": cert["status"], "search_complete": cert["search_complete"], "reason": cert["reason"]}, ensure_ascii=False))
        if cert["search_complete"]:
            return 0
        return 4 if cert["status"] == "BUDGET_EXCEEDED" else 2

    if not args.run_dir:
        raise SystemExit("--run-dir is required with --register")
    summary = run_batch(
        Path(args.register),
        Path(args.run_dir),
        verify_source_sha=not args.no_verify_source_sha,
        cluster_filter=args.clusters,
    )
    print(json.dumps({"status_counts_all": summary["status_counts_all"], "total_solve_seconds": summary["total_solve_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
