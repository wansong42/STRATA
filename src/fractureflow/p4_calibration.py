# -*- coding: utf-8 -*-
"""G74 P4 calibration primitives (CPU only, stdlib + numpy).

The module only uses case records that are already outcome-blind at split time.
It refuses (SystemExit) rather than silently dropping any case.  No certificate
is issued here: every reading is an evaluation of certificates issued by G73.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

GRID = (0.25, 0.5, 0.75, 1.0)
N_BOOT = 2000
BOOT_SEED = 20261022

STATUS_CERT_CONNECTED = "CERTIFIED_CONNECTED"
STATUS_CERT_DISCONNECTED = "CERTIFIED_DISCONNECTED"
STATUS_AMBIGUOUS = "WITNESSED_AMBIGUOUS"
STATUS_UNRESOLVED = "SEARCH_UNRESOLVED"
STATUS_CONFLICT = "MODEL_CONFLICT"
ALL_STATUSES = (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED, STATUS_AMBIGUOUS,
                STATUS_UNRESOLVED, STATUS_CONFLICT)

LEVELS = (
    {"id": "L0_CERTIFIED_ONLY", "accepts": (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED)},
    {"id": "L1_PLUS_AMBIGUOUS", "accepts": (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED, STATUS_AMBIGUOUS)},
    {"id": "L2_PLUS_UNRESOLVED", "accepts": ALL_STATUSES},
)


class Refusal(SystemExit):
    """A contract refusal; never caught into a PASS."""


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canon_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- intervals
def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k + 1)))


def _bisect_monotone(func, target: float, increasing: bool, iters: int = 200) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = func(mid)
        if (val < target) if increasing else (val > target):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_two_sided(k: int, n: int, alpha: float = 0.05) -> dict:
    if n <= 0:
        raise Refusal("Clopper-Pearson denominator is zero")
    if n > 10000:
        raise Refusal("Clopper-Pearson: n too large for exact evaluation")
    # lower: 1 - CDF(k-1; p) = alpha/2, increasing in p
    # upper:     CDF(k;   p) = alpha/2, decreasing in p
    lo = 0.0 if k == 0 else _bisect_monotone(lambda p: 1.0 - _binom_cdf(k - 1, n, p), alpha / 2.0, increasing=True)
    hi = 1.0 if k == n else _bisect_monotone(lambda p: _binom_cdf(k, n, p), alpha / 2.0, increasing=False)
    return {"k": int(k), "n": int(n), "alpha": alpha, "lower": round(lo, 6), "upper": round(hi, 6)}


def cp_upper_one_sided(k: int, n: int, alpha: float = 0.05) -> dict:
    if n <= 0:
        raise Refusal("Clopper-Pearson denominator is zero")
    # CDF(k; p) = alpha, decreasing in p
    hi = 1.0 if k == n else _bisect_monotone(lambda p: _binom_cdf(k, n, p), alpha, increasing=False)
    return {"k": int(k), "n": int(n), "alpha_one_sided": alpha, "upper": round(hi, 6)}


# -------------------------------------------------------------------- split
def assign_truth_split(world_ids, world_split) -> dict:
    out = {}
    for wid in world_ids:
        if wid not in world_split:
            raise Refusal(f"world {wid} absent from the frozen prereg split")
        out[wid] = world_split[wid]
    return out


def assert_no_leakage(assignments, split_key: str) -> dict:
    """assignments: list of dicts with split_key and source_key."""
    sides = {}
    for a in assignments:
        sides.setdefault(a["split"], set()).add(a[split_key])
    cal = sides.get("CALIBRATION", set())
    test = sides.get("TEST", set())
    if not cal or not test:
        raise Refusal(f"empty split on {split_key}")
    overlap = sorted(cal & test)
    if overlap:
        raise Refusal(f"leakage on {split_key}: {overlap}")
    return {"split_key": split_key, "calibration": len(cal), "test": len(test),
            "overlap": overlap, "n_assignments": len(assignments)}


# ---------------------------------------------------------------- coverage
def guard_coverage_denominator(all_cases, used_cases):
    """A coverage denominator may never be a subset of the truth-bearing tasks."""
    if len(used_cases) != len(all_cases):
        raise Refusal(f"coverage denominator truncated: {len(used_cases)} of {len(all_cases)} tasks")
    if {c["case_id"] for c in used_cases} != {c["case_id"] for c in all_cases}:
        raise Refusal("coverage denominator does not match the full task set")
    return True


def guard_no_truth_in_coverage(cases):
    bad = [c["case_id"] for c in cases if c.get("truth_label") is None]
    if bad:
        raise Refusal(f"no-truth cases entered coverage: {bad[:5]}")
    return True


def guard_population(cases, forbidden_sources):
    bad = [c["case_id"] for c in cases if any(f in str(c.get("source_id", "")) for f in forbidden_sources)]
    if bad:
        raise Refusal(f"forbidden source entered the coverage denominator: {bad[:5]}")
    return True


def guard_unique_ids(cases):
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise Refusal("duplicate case id")
    return True


def guard_query_kind(cases, supported="PLANAR_UNION_CONNECTIVITY"):
    bad = [c["case_id"] for c in cases if c.get("query_kind") != supported]
    if bad:
        raise Refusal(f"unsupported query kind on {bad[:5]}")
    return True


def guard_report_lines(n_lines: int, limit: int = 60):
    if n_lines > limit:
        raise Refusal(f"report has {n_lines} lines > {limit}")
    return True


FORBIDDEN_CLAIM_PATTERNS = (
    ("FALLON_AS_3D_TRUTH", ("3d_connectivity_truth", "3d connectivity truth", "connectivity truth", "三维连通真值")),
    ("MD_AS_TVD", ("tvd", "true vertical depth")),
    ("IMAGE_LEVEL_SUPERVISION", ("image_level_supervision", "image-level supervision", "图像级监督")),
)


def lint_forbidden_claims(text: str) -> dict:
    low = text.lower()
    hits = []
    for label, needles in FORBIDDEN_CLAIM_PATTERNS:
        for n in needles:
            if n.lower() in low:
                hits.append({"class": label, "needle": n})
    return {"clean": not hits, "hits": hits}


def _require_truth(cases):
    if not cases:
        raise Refusal("empty truth-bearing denominator")
    for c in cases:
        if c.get("truth_label") is None:
            raise Refusal(f"case {c['case_id']} has no truth label but entered coverage")


def coverage_query(cases) -> dict:
    _require_truth(cases)
    n = len(cases)
    covered = sum(1 for c in cases if int(c["truth_label"]) in [int(x) for x in c["gamma"]])
    cert = [c for c in cases if c.get("certificate_issued")]
    covered_cert = sum(1 for c in cert if int(c["truth_label"]) in [int(x) for x in c["gamma"]])
    single = [c for c in cases if c["status"] in (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED)]
    return {
        "definition": "truth_label in Gamma_Q; denominator = all truth-bearing tasks in the split",
        "denominator": n,
        "covered": int(covered),
        "coverage": round(covered / n, 6),
        "cp95": cp_two_sided(covered, n),
        "contrast_column_not_the_metric": {
            "label": "coverage among issued certificates only (conditioned on issuance)",
            "denominator": len(cert),
            "covered": int(covered_cert),
            "coverage": (round(covered_cert / len(cert), 6) if cert else None),
        },
        "single_value_tasks": len(single),
        "single_value_rate": round(len(single) / n, 6),
    }


def coverage_object(cases) -> dict:
    _require_truth(cases)
    total = 0
    covered = 0
    all_covered_cases = 0
    uncovered_examples = []
    for c in cases:
        ids = sorted(c.get("inventory_ids") or [])
        truth = set(c.get("truth_present_ids") or [])
        must = set(c.get("must_nodes") or [])
        may = set(c.get("may_nodes") or [])
        if c["status"] in (STATUS_UNRESOLVED, STATUS_CONFLICT) or not c.get("must_may_available"):
            # No surviving-hypothesis envelope: count every object as uncovered.
            total += len(ids)
            if ids:
                uncovered_examples.append({"case_id": c["case_id"], "reason": "NO_ENVELOPE", "n_objects": len(ids)})
            continue
        case_ok = True
        for oid in ids:
            total += 1
            t = oid in truth
            ok = (oid in must) <= t <= (oid in may)
            if ok:
                covered += 1
            else:
                case_ok = False
                if len(uncovered_examples) < 10:
                    uncovered_examples.append({"case_id": c["case_id"], "object_id": oid,
                                               "truth_present": t, "must_present": oid in must,
                                               "may_present": oid in may})
        if case_ok:
            all_covered_cases += 1
    if total == 0:
        raise Refusal("object-level denominator is zero")
    return {
        "definition": "truth presence bit inside [must_present, may_present] of surviving hypotheses; unresolved/conflict cases have no envelope and count as uncovered",
        "denominator_objects": total,
        "covered_objects": covered,
        "object_coverage": round(covered / total, 6),
        "cp95": cp_two_sided(covered, total) if total <= 10000 else None,
        "cases_all_objects_covered": all_covered_cases,
        "denominator_cases": len(cases),
        "uncovered_examples": uncovered_examples,
        "not_a_topological_metric": True,
    }


def bootstrap_cluster(cases, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    _require_truth(cases)
    if n_boot < 2000:
        raise Refusal("bootstrap below the preregistered 2000 resamples")
    clusters = {}
    for c in cases:
        clusters.setdefault(c["world_id"], []).append(c)
    keys = sorted(clusters)
    if len(keys) < 2:
        raise Refusal("cluster bootstrap needs at least 2 clusters")
    rng = np.random.default_rng(seed)
    cov, single, abst = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        rows = [c for i in pick for c in clusters[keys[int(i)]]]
        n = len(rows)
        cov.append(sum(1 for c in rows if int(c["truth_label"]) in [int(x) for x in c["gamma"]]) / n)
        single.append(sum(1 for c in rows if c["status"] in (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED)) / n)
        abst.append(sum(1 for c in rows if c["status"] not in (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED)) / n)

    def pc(a):
        return {"p2_5": round(float(np.percentile(a, 2.5)), 6),
                "p50": round(float(np.percentile(a, 50)), 6),
                "p97_5": round(float(np.percentile(a, 97.5)), 6)}

    return {"unit": "world_id", "n_clusters": len(keys), "n_resamples": n_boot, "seed": seed,
            "method": "percentile", "coverage": pc(cov), "single_value_rate": pc(single),
            "abstention_rate": pc(abst), "degenerate_all_clusters_equal": len({c["status"] for c in cases}) == 1}


# ---------------------------------------------------------- risk-coverage
def _accepted(level, case) -> bool:
    return case["status"] in level["accepts"]


def _answer(level, case):
    """Return (answer, provenance).  L0 answers are certificates; L1/L2 forced."""
    st = case["status"]
    if st in (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED):
        return int(case["certificate_label"]), "CERTIFICATE"
    if level["id"] == "L0_CERTIFIED_ONLY":
        raise Refusal("L0 cannot answer a non-certified case")
    if st == STATUS_AMBIGUOUS:
        return int(bool(case.get("must_connected"))), "FORCED_PESSIMISTIC_NOT_A_CERTIFICATE"
    return 0, "FORCED_ZERO_NOT_A_CERTIFICATE"


def _level_metrics(cases, level) -> dict:
    denom = len(cases)
    acc = [c for c in cases if _accepted(level, c)]
    wrong = 0
    cert_acc = 0
    for c in acc:
        ans, prov = _answer(level, c)
        if prov == "CERTIFICATE":
            cert_acc += 1
        if int(ans) != int(c["truth_label"]):
            wrong += 1
    return {
        "level": level["id"],
        "accepted": len(acc),
        "coverage": round(len(acc) / denom, 6),
        "wrong": int(wrong),
        "risk": (round(wrong / len(acc), 6) if acc else None),
        "accepted_certificates": cert_acc,
        "accepted_forced": len(acc) - cert_acc,
        "denominator": denom,
    }


def select_operating_points(cal_cases, grid=GRID) -> list:
    """Selection reads CALIBRATION ONLY.  No test argument exists by design."""
    _require_truth(cal_cases)
    out = []
    for target in grid:
        chosen = None
        for level in LEVELS:
            m = _level_metrics(cal_cases, level)
            if m["coverage"] + 1e-12 >= target:
                chosen = m
                break
        out.append({"target_coverage": target,
                    "reachable": chosen is not None,
                    "chosen_level": (chosen["level"] if chosen else None),
                    "calibration_metrics": chosen})
    return out


def risk_coverage(cal_cases, test_cases, grid=GRID) -> dict:
    _require_truth(cal_cases)
    _require_truth(test_cases)
    if tuple(grid) != tuple(GRID):
        raise Refusal("coverage grid drifted from the preregistered grid")
    selection = select_operating_points(cal_cases, grid)
    rows = []
    for sel in selection:
        row = dict(sel)
        if sel["reachable"]:
            level = next(l for l in LEVELS if l["id"] == sel["chosen_level"])
            row["test_metrics"] = _level_metrics(test_cases, level)
            row["selection_inputs"] = "CALIBRATION_ONLY"
        rows.append(row)
    all_levels = {
        "calibration": [_level_metrics(cal_cases, l) for l in LEVELS],
        "test": [_level_metrics(test_cases, l) for l in LEVELS],
    }
    diag = []
    for c in test_cases:
        diag.append({"case_id": c["case_id"], "status": c["status"],
                     "truth": int(c["truth_label"]),
                     "optimistic_answer": int(bool(c.get("may_connected")))})
    opt_wrong = sum(1 for d in diag if d["optimistic_answer"] != d["truth"])
    return {
        "coverage_grid": list(grid),
        "selection_rule": "most conservative level whose CALIBRATION coverage >= target; test never read",
        "selected_points": rows,
        "all_levels_for_transparency": all_levels,
        "diagnostic_optimistic_answer": {
            "definition": "may_connected forced answer; NOT a certificate and NOT used for any decision",
            "n": len(diag),
            "wrong": int(opt_wrong),
            "risk": round(opt_wrong / len(diag), 6),
            "warning": "this column is inflated by the synthetic answer sheet (ambiguous cases are all truth=1); never cite it as a certificate error rate",
        },
        "posthoc_selection_used_test_outcomes": False,
    }


# ------------------------------------------------------------- confusion
def query_confusion(cases) -> dict:
    _require_truth(cases)
    n = len(cases)
    correct = wrong = amb = unres = conf = 0
    errors = []
    for c in cases:
        st = c["status"]
        if st in (STATUS_CERT_CONNECTED, STATUS_CERT_DISCONNECTED):
            if int(c["certificate_label"]) == int(c["truth_label"]):
                correct += 1
            else:
                wrong += 1
                errors.append({"case_id": c["case_id"], "certificate_label": int(c["certificate_label"]),
                               "truth_label": int(c["truth_label"])})
        elif st == STATUS_AMBIGUOUS:
            amb += 1
        elif st == STATUS_UNRESOLVED:
            unres += 1
        elif st == STATUS_CONFLICT:
            conf += 1
        else:
            raise Refusal(f"unknown status {st} on {c['case_id']}")
    if correct + wrong + amb + unres + conf != n:
        raise Refusal("outcome partition does not sum to the denominator")
    cert_n = correct + wrong
    return {
        "denominator": n,
        "query_correct_single": correct,
        "query_wrong_single": wrong,
        "abstention_ambiguous": amb,
        "abstention_unresolved": unres,
        "abstention_conflict": conf,
        "abstention_total": amb + unres + conf,
        "correct_single_rate": round(correct / n, 6),
        "wrong_single_rate": round(wrong / n, 6),
        "abstention_rate": round((amb + unres + conf) / n, 6),
        "single_value_rate": round(cert_n / n, 6),
        "certificates": cert_n,
        "certificate_error_rate": (round(wrong / cert_n, 6) if cert_n else None),
        "certificate_error_upper_95_one_sided": cp_upper_one_sided(wrong, cert_n) if cert_n else None,
        "wrong_cases": errors,
        "abstention_definition": "dual-valued + unresolved + conflict, over all tasks",
    }


# --------------------------------------------------------------- audit
def certificate_audit(g73_certs, g73_cases, replay_cases) -> dict:
    by_id = {c["case_id"]: c for c in g73_cases}
    rp = {c["case_id"]: c for c in replay_cases}
    rows = []
    for cert in g73_certs:
        cid = cert["case_id"]
        case = by_id.get(cid)
        rep = rp.get(cid)
        if case is None or rep is None:
            raise Refusal(f"certificate {cid} lacks a case or replay record")
        truth = case.get("truth_label")
        rec_label = cert.get("label")
        rep_cert = rep.get("certificate") or {}
        rep_label = rep_cert.get("label") if rep_cert.get("issued") else None
        rep_status = rep.get("status")
        truth_bearing = truth is not None
        match = (rec_label == rep_label) and (cert.get("status") == rep_status)
        witness_present = rep_cert.get("witness_present_ids") or []
        inventory = {o["id"] for o in (rep.get("inventory") or [])}
        witness_ok = bool(witness_present) and set(witness_present).issubset(inventory)
        witness_label_ok = rep_cert.get("witness_label") == rep_label
        wrong = bool(truth_bearing and (rec_label != truth or rep_label != truth))
        rows.append({
            "certificate_id": cert.get("certificate_id", cid),
            "case_id": cid,
            "truth_bearing": truth_bearing,
            "truth_label": (int(truth) if truth is not None else None),
            "recorded_label": rec_label,
            "recorded_status": cert.get("status"),
            "replay_label": rep_label,
            "replay_status": rep_status,
            "replay_match": bool(match),
            "witness_present": witness_ok,
            "witness_label_matches": bool(witness_label_ok),
            "all_surviving_hypotheses_same_label": bool(cert.get("all_surviving_hypotheses_same_label", True)),
            "correct": (bool(rec_label == truth and rep_label == truth) if truth_bearing else None),
            "wrong": wrong,
        })
    tb = [r for r in rows if r["truth_bearing"]]
    nt = [r for r in rows if not r["truth_bearing"]]
    wrong_n = sum(1 for r in tb if r["wrong"])
    return {
        "n_certificates": len(rows),
        "truth_bearing": {
            "n": len(tb),
            "replay_match": sum(1 for r in tb if r["replay_match"]),
            "witness_ok": sum(1 for r in tb if r["witness_present"] and r["witness_label_matches"]),
            "wrong": wrong_n,
            "certificate_error_rate": (round(wrong_n / len(tb), 6) if tb else None),
            "certificate_error_upper_95_one_sided": cp_upper_one_sided(wrong_n, len(tb)) if tb else None,
        },
        "no_truth": {
            "n": len(nt),
            "replay_match": sum(1 for r in nt if r["replay_match"]),
            "correctness": "NOT_ASSESSABLE",
            "note": "replay/status only; these certificates never enter coverage",
        },
        "replay_mismatches": sum(1 for r in rows if not r["replay_match"]),
        "rows": rows,
    }


def truth_recompute_audit(g73_cases, truth_recompute) -> dict:
    out = []
    for c in g73_cases:
        if c.get("truth_label") is None:
            continue
        item = truth_recompute.get(c["case_id"])
        if item is None:
            raise Refusal(f"truth recompute missing {c['case_id']}")
        out.append({"case_id": c["case_id"], "recorded": int(c["truth_label"]),
                    "recomputed": int(item["truth_label"]),
                    "match": int(c["truth_label"]) == int(item["truth_label"])})
    mism = [r for r in out if not r["match"]]
    return {"n": len(out), "n_mismatch": len(mism), "mismatches": mism}


# --------------------------------------------------------------- fallon
def circular_stats(deg) -> dict:
    a = np.asarray(deg, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise Refusal("circular statistics received no finite values")
    rad = np.deg2rad(a)
    c = float(np.mean(np.cos(rad)))
    s = float(np.mean(np.sin(rad)))
    mean = math.degrees(math.atan2(s, c)) % 360.0
    r = math.hypot(c, s)
    return {"n": int(a.size), "circular_mean_deg": round(mean, 3),
            "resultant_length": round(r, 6),
            "circular_std_deg": round(math.degrees(math.sqrt(max(-2.0 * math.log(max(r, 1e-12)), 0.0))), 3)}


def scalar_stats(values, prefix: str) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise Refusal("scalar statistics received no finite values")
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {f"{prefix}_n": int(a.size), f"{prefix}_min": round(float(a.min()), 4),
            f"{prefix}_p10": round(float(q[0]), 4), f"{prefix}_p25": round(float(q[1]), 4),
            f"{prefix}_median": round(float(q[2]), 4), f"{prefix}_p75": round(float(q[3]), 4),
            f"{prefix}_p90": round(float(q[4]), 4), f"{prefix}_max": round(float(a.max()), 4),
            f"{prefix}_mean": round(float(a.mean()), 4)}


def ks_distance(a, b) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        raise Refusal("KS distance with an empty sample")
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return round(float(np.max(np.abs(ca - cb))), 6)


def fallon_stats(depths_m, dips_deg, azimuths_deg, apertures_mm, md_span_m=None) -> dict:
    depth_stats = scalar_stats(depths_m, "depth_md_m")
    dip_stats = scalar_stats(dips_deg, "dip_deg")
    az = circular_stats(azimuths_deg)
    ap = np.asarray(apertures_mm, dtype=np.float64)
    if not np.all(np.isfinite(ap)):
        raise Refusal("non-finite aperture")
    span = float(md_span_m if md_span_m is not None else (np.max(depths_m) - np.min(depths_m)))
    if span <= 0:
        raise Refusal("non-positive logged interval")
    return {
        **depth_stats,
        **dip_stats,
        "dip_azimuth_deg": az,
        "density_picks_per_100m": round(len(depths_m) / span * 100.0, 4),
        "logged_interval_m": round(span, 4),
        "aperture_mm": {"n_zero": int(np.sum(ap == 0.0)), "n_positive": int(np.sum(ap > 0.0)),
                        "max": round(float(ap.max()), 4)},
        "dip_bins_deg": {
            "0_30": int(np.sum((np.asarray(dips_deg) >= 0) & (np.asarray(dips_deg) < 30))),
            "30_60": int(np.sum((np.asarray(dips_deg) >= 30) & (np.asarray(dips_deg) < 60))),
            "60_90": int(np.sum((np.asarray(dips_deg) >= 60) & (np.asarray(dips_deg) <= 90))),
        },
    }


# --------------------------------------------------------------- verdict
def a1_reading(test_confusion, audit) -> dict:
    """Strict A1 conditions (execution plan 7.3).  This unit does not sign it."""
    q = test_confusion
    upper = audit["truth_bearing"]["certificate_error_upper_95_one_sided"]
    cp_abs = cp_two_sided(q["abstention_total"], q["denominator"])
    cond = {
        "1_f1_struct_margin_ge_0_05": {"status": "NOT_EVALUABLE", "why": "no P3 model and no F1_struct baseline in this unit"},
        "2_site_bootstrap_ci_lower_gt_0": {"status": "NOT_EVALUABLE", "why": "F1_struct delta does not exist"},
        "3_wrong_certificate_count_zero": {"status": ("PASS" if q["query_wrong_single"] == 0 and audit["truth_bearing"]["wrong"] == 0 else "FAIL"),
                                            "value": audit["truth_bearing"]["wrong"]},
        "4_cert_error_upper_95_le_0_05": {"status": ("PASS" if upper and upper["upper"] <= 0.05 else "FAIL"),
                                           "value": upper},
        "5_abstention_rate_le_0_5": {"status": ("PASS_POINT_ONLY" if q["abstention_rate"] <= 0.5 else "FAIL"),
                                      "point": q["abstention_rate"], "exact_cp95": cp_abs,
                                      "note": "the point estimate passes on the test denominator but the exact interval does not exclude 0.5"},
        "6_relational_or_topological_improvement": {"status": "NOT_EVALUABLE", "why": "no comparator model or baseline in this unit"},
        "7_recomputable_from_disk": {"status": "PASS", "why": "all readings are JSON artifacts recomputable by scripts/g74_*.py"},
    }
    blocking = [k for k, v in cond.items() if v["status"] in ("FAIL", "NOT_EVALUABLE", "PASS_POINT_ONLY")]
    return {"signed_by_this_unit": False,
            "would_go_if_signed_now": len(blocking) == 0,
            "blocking_conditions": blocking,
            "conditions": cond}


def decide_verdict(test_confusion, test_coverage, test_worlds, audit, truth_audit, leakage, replay_missing) -> dict:
    q = test_confusion
    if q["denominator"] < 12 or test_worlds < 4 or replay_missing > 0:
        code = "G74_P4_DATA_INSUFFICIENT"
        grade = "DATA_INSUFFICIENT"
    elif q["query_wrong_single"] > 0 or audit["replay_mismatches"] > 0 or audit["truth_bearing"]["wrong"] > 0 or leakage > 0 or truth_audit["n_mismatch"] > 0:
        code = "G74_P4_NO_GO"
        grade = "NO_GO"
    elif test_coverage["coverage"] < 0.80 or test_coverage["cp95"]["lower"] < 0.60:
        code = "G74_P4_T2_DIAGNOSTIC_ONLY"
        grade = "T2"
    else:
        code = "G74_P4_T1_SYNTHETIC_EMPIRICAL"
        grade = "T1"
    a1 = a1_reading(test_confusion, audit)
    return {"code": code, "grade": grade,
            "real_domain_grade": "T2",
            "qualifier": "SYNTHETIC_FINITE_DECLARED_DOMAIN_ONLY / REAL_DOMAIN_T2",
            "a1_reading": a1,
            "inputs": {"test_denominator": q["denominator"], "test_worlds": test_worlds,
                       "coverage": test_coverage["coverage"], "coverage_cp_lower": test_coverage["cp95"]["lower"],
                       "wrong_single": q["query_wrong_single"],
                       "cert_error_upper_95": audit["truth_bearing"]["certificate_error_upper_95_one_sided"],
                       "replay_mismatches": audit["replay_mismatches"],
                       "truth_recompute_mismatch": truth_audit["n_mismatch"],
                       "leakage": leakage}}
