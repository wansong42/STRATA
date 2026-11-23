# -*- coding: utf-8 -*-
"""S3-B · 选点策略变体引擎（PLANNING_PROXY_ONLY ｜ DEVELOPMENT_ONLY ｜ N1）。

任务单：`results/architect_main_20261012_dispatch/task_order_S3_B_policy_variants_20261018.md`
（sha12 `caa630b6dcb3`）。本模块给 S3-A 的扩展池闭环提供**参数化选点引擎**：把 S3-A
`closed_loop_expanded._choose_new` 的管线拆成 7 个可声明旋钮，逐变体只改一个旋钮。

旋钮（默认值 = S3-A NEW 基线实现，逐位等价由 b0_ENGINE ≡ b0_NEW 检查监督）:
  * divergence       : HELLINGER（默认）/ KL_SYM（ε-平滑 Jeffreys）/ TV
  * share_floor      : 0.10（默认）/ 0.05 / 0.20 / NONE（c~ = c/c_max）
  * opponent_search  : ON（默认，预算 64 EXHAUST_PAIRS_UP_TO_BUDGET）/ OFF
  * prefilter_k      : 4（默认）/ 8 / FULL（不做前筛）
  * two_step         : ON（默认）/ OFF（选择只看第一步分数）
  * selection        : ARGMAX_TWO_STEP（默认）/ ARGMAX_FIRST_STEP / RANDOM_PI（π∝λ/c~）
  * weights          : EQUAL（默认）/ HARD（难对加权 W_ij ∝ 1/(δ+D̄_ij)，δ=0.05）

实现纪律
--------
* **能委托就委托**：基线可表达的子路径直接调用 `active_policy` 原函数
  （`hellinger` / `opponent_search` / `cost_share` / `top_k_filter` /
  `two_step_lookahead` / `_minmax_cost_share`），保证与 S3-A 逐位一致；只有无法用
  原函数表达的分量（KL_SYM/TV 分离度、NONE 成本份额、HARD 权重、两步评分的泛化）
  走本模块实现，并由 `_work/engine_selfcheck.json` 的镜像-原函数对照监督。
* **不是认证器**：本模块输出是规划代理分数，不是错误率/置信界；不构成策略优劣主张。
* **只看公开面**：`engine_plan` 只吃 `PublicStateExpanded` + 静态 ddist；真值不进入。
* **零硬删除、零改旧字节**：本模块只读消费 S3-A / S3-P 制品。
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import active_policy as ap
from . import closed_loop_expanded as cle

__all__ = [
    "WATERMARK", "DEFAULT_SPEC", "VARIANT_SPECS", "VARIANT_ORDER", "BASE_STRATEGIES",
    "EPS_KL_SM", "HARD_DELTA", "OPP_BUDGET", "MAX_WITNESS_PAIRS",
    "tv", "kl_sym", "pair_div", "witness_pairs", "cost_share_of",
    "hard_pair_weights", "opponent_search_any", "lp_lambda_any",
    "mirror_two_step", "engine_plan", "spec_delta", "branch_report_variant",
    "validators", "validate_ledger_variants", "canonical_json",
]

WATERMARK = {
    "research_line": "S3B_policy_variants",
    "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
    "noise": "N1",
    "dictionary": "COMPLETE_KNOWN",
    "claim_scope": "DEVELOPMENT_ONLY",
    "forbidden": ["真实贯通判定", "普遍策略优越", "成本优势主张", "对外承诺文本",
                  "确认实验"],
}

_TOL = 1e-9                       # 与 active_policy._TOL 同值（镜像对照用）
EPS_KL_SM = 1e-3                  # KL_SYM 的声明平滑常量（并支撑 ε-混合）
HARD_DELTA = 0.05                 # V_WEIGHT_HARD 的声明平滑常量
OPP_BUDGET = 64                   # 与 S3-A 对手搜索预算同值
MAX_WITNESS_PAIRS = 128           # 与 S3-A 见证对 cap 同值

BASE_STRATEGIES = ("NEW", "EC2_TWO_STEP")

#: 默认旋钮 = S3-A `_choose_new` 的实现选择（不是 prereg 文字声称的口径；见 own_defects）
DEFAULT_SPEC = {
    "divergence": "HELLINGER",
    "share_floor": 0.10,
    "opponent_search": "ON",
    "prefilter_k": 4,
    "two_step": "ON",
    "selection": "ARGMAX_TWO_STEP",
    "weights": "EQUAL",
}

VARIANT_SPECS: Dict[str, dict] = {
    "V_DIV_KL_SYM": {**DEFAULT_SPEC, "divergence": "KL_SYM"},
    "V_DIV_TV": {**DEFAULT_SPEC, "divergence": "TV"},
    "V_FLOOR_005": {**DEFAULT_SPEC, "share_floor": 0.05},
    "V_FLOOR_020": {**DEFAULT_SPEC, "share_floor": 0.20},
    "V_FLOOR_NONE": {**DEFAULT_SPEC, "share_floor": "NONE"},
    "V_OPP_OFF": {**DEFAULT_SPEC, "opponent_search": "OFF"},
    "V_K8": {**DEFAULT_SPEC, "prefilter_k": 8},
    "V_KPOOL": {**DEFAULT_SPEC, "prefilter_k": "FULL"},
    "V_TWO_OFF": {**DEFAULT_SPEC, "two_step": "OFF", "selection": "ARGMAX_FIRST_STEP"},
    "V_SELECT_RANDOM": {**DEFAULT_SPEC, "selection": "RANDOM_PI"},
    "V_WEIGHT_HARD": {**DEFAULT_SPEC, "weights": "HARD"},
}
VARIANT_ORDER = tuple(VARIANT_SPECS)

#: 名册策略次序（显式钉住；policy_seed 公式用 strategy_index）
STRATEGY_ORDER = BASE_STRATEGIES + VARIANT_ORDER


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def spec_delta(spec: Mapping[str, Any]) -> dict:
    """该 spec 相对默认基线的声明差（必须恰为任务单声明的单旋钮差）。"""
    return {k: v for k, v in spec.items() if k in DEFAULT_SPEC and DEFAULT_SPEC[k] != v}


# ---------------------------------------------------------------------------
# 分离度（KL_SYM / TV；HELLINGER 委托 active_policy）
# ---------------------------------------------------------------------------

def tv(p: Mapping[Any, float], q: Mapping[Any, float]) -> float:
    """总变差 `TV = 1/2 Σ_y |p(y)-q(y)| ∈ [0,1]`（并支撑求和；无平滑常量）。"""
    if not p or not q:
        raise ap.PlanningContractError("tv: 空分布 ⇒ 拒")
    keys = set(p) | set(q)
    return float(0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0)))
                           for k in keys))


def kl_sym(p: Mapping[Any, float], q: Mapping[Any, float],
           *, eps: float = EPS_KL_SM) -> float:
    """对称 KL（Jeffreys）在 **ε-混合**上的有限读数。

    声明口径（prereg 冻结）：先在并支撑 U 上做 ε-混合
    `p̃ = (1-ε)p + ε/|U|`、`q̃` 同式，再算 `D = Σ p̃ ln(p̃/q̃) + Σ q̃ ln(q̃/p̃)`。
    理由（结构探针 P1）：N1 下 81.4% 的对立对存在一方独有原子 ⇒ **原始 Jeffreys
    普遍为 +∞**，任何有限规划分数都必须声明混合常量；ε=1e-3 全案一致、不参与调参。
    """
    if not p or not q:
        raise ap.PlanningContractError("kl_sym: 空分布 ⇒ 拒")
    if not (0.0 < float(eps) < 1.0):
        raise ap.PlanningContractError("eps 必须 ∈ (0,1): %r" % (eps,))
    keys = set(p) | set(q)
    e = float(eps) / max(1, len(keys))
    d = 0.0
    for k in keys:
        pp = (1.0 - float(eps)) * float(p.get(k, 0.0)) + e
        qq = (1.0 - float(eps)) * float(q.get(k, 0.0)) + e
        d += pp * math.log(pp / qq) + qq * math.log(qq / pp)
    if not math.isfinite(d) or d < -1e-9:
        raise ap.PlanningContractError("kl_sym 非有限/负值（实现缺陷）: %r" % d)
    return float(max(0.0, d))


_DIV_MEMO: Dict[tuple, float] = {}
_DIV_MEMO_GUARD: Optional[tuple] = None


def _guard_of(dists: Mapping[str, Mapping[int, Mapping[Any, float]]]) -> tuple:
    aid0 = sorted(dists)[0]
    s0 = min(dists[aid0])
    return (id(dists[aid0][s0]), len(dists), len(dists[aid0]))


def pair_div(kind: str, dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
             aid: str, i: int, j: int) -> float:
    """(行动, 状态对) 的分离度；进程内记忆化（同一静态面重复调用不重算）。"""
    global _DIV_MEMO_GUARD
    key = (kind, str(aid), int(i), int(j))
    g = _guard_of(dists)
    if _DIV_MEMO_GUARD != g:
        _DIV_MEMO.clear()
        _DIV_MEMO_GUARD = g
    v = _DIV_MEMO.get(key)
    if v is None:
        p = dists[aid][int(i)]
        q = dists[aid][int(j)]
        if kind == "HELLINGER":
            v = float(ap.hellinger(p, q))
        elif kind == "TV":
            v = tv(p, q)
        elif kind == "KL_SYM":
            v = kl_sym(p, q)
        else:
            raise ap.PlanningContractError("未知分离度 %r" % (kind,))
        _DIV_MEMO[key] = v
    return float(v)


def _div_matrix(kind: str, dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                pairs: Sequence[Tuple[int, int]]) -> Dict[str, Dict[Tuple[int, int], float]]:
    out: Dict[str, Dict[Tuple[int, int], float]] = {}
    for aid in sorted(dists):
        out[aid] = {(int(i), int(j)): pair_div(kind, dists, aid, i, j)
                    for (i, j) in pairs}
    return out


# ---------------------------------------------------------------------------
# 见证对 / 成本份额 / 对手搜索
# ---------------------------------------------------------------------------

def witness_pairs(world, allowed) -> list:
    """统计可能集内对立标签对（字典序）；与 `cle._witness_pairs` 同构。"""
    by_label: Dict[int, list] = {0: [], 1: []}
    for s in allowed:
        by_label[world.labels[int(s)]].append(int(s))
    pairs = [(int(i), int(j)) for i in by_label[0] for j in by_label[1]]
    pairs.sort()
    return pairs


def cost_share_of(costs: Mapping[str, float], floor) -> Dict[str, float]:
    """成本份额 c~：标准 floor 走 `ap._minmax_cost_share`；NONE 走 c/c_max。"""
    vals = {a: float(c) for a, c in costs.items()}
    if isinstance(floor, str) and floor == "NONE":
        cmax = max(vals.values())
        if cmax <= 0.0:
            raise ap.PlanningContractError("NONE 成本份额: c_max ≤ 0")
        return {a: float(v / cmax) for a, v in vals.items()}
    return {a: float(v) for a, v in
            ap._minmax_cost_share(vals, floor=float(floor)).items()}


def hard_pair_weights(pairs: Sequence[Tuple[int, int]],
                      dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                      pool: Sequence[str], kind: str) -> Dict[Tuple[int, int], float]:
    """V_WEIGHT_HARD：W_ij ∝ 1/(δ+D̄_ij)，D̄ = 全合法池平均分离度；均值归一到 1。"""
    raw: Dict[Tuple[int, int], float] = {}
    for (i, j) in pairs:
        dbar = sum(pair_div(kind, dists, a, i, j) for a in pool) / float(len(pool))
        raw[(int(i), int(j))] = 1.0 / (HARD_DELTA + dbar)
    mean = sum(raw.values()) / float(len(raw))
    return {p: float(v / mean) for p, v in raw.items()}


def opponent_search_any(kind: str, witness_set: ap.WitnessSet,
                        dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                        label_of: Mapping[int, Any], budget: int) -> dict:
    """对手搜索：HELLINGER 直接委托 `ap.opponent_search`；其余分离度走镜像。"""
    if kind == "HELLINGER":
        out = ap.opponent_search(witness_set, dists, label_of, budget=budget,
                                 stop_rule="EXHAUST_PAIRS_UP_TO_BUDGET")
        out = dict(out)
        out["mode"] = "DELEGATE_AP"
        return out
    if budget <= 0:
        raise ap.PlanningContractError("预算必须为正: %r" % (budget,))
    if not label_of:
        raise ap.PlanningContractError("label_of 为空 ⇒ 拒")
    if not dists:
        raise SystemExit("[policy_variants] 空行动池 ⇒ 停")
    states = sorted(int(s) for s in label_of)
    covered = set(witness_set.pairs)
    cands: list = []
    n_same = 0
    for ii in range(len(states)):
        for jj in range(ii + 1, len(states)):
            a, b = states[ii], states[jj]
            pair = (min(a, b), max(a, b))
            if pair in covered:
                continue
            if label_of[a] == label_of[b]:
                n_same += 1
                continue
            cands.append(pair)
    aids = sorted(dists)
    rows = []
    for (i, j) in cands:
        per = {a: pair_div(kind, dists, a, i, j) for a in aids}
        worst_a = max(per, key=lambda a: (per[a], a))
        rows.append({"pair": [i, j], "labels": [label_of[i], label_of[j]],
                     "worst_action_id": worst_a,
                     "worst_divergence": float(per[worst_a]),
                     "mean_divergence": float(sum(per.values()) / len(per)),
                     "per_action_divergence": {a: float(v) for a, v in per.items()}})
    rows.sort(key=lambda r: (r["worst_divergence"], r["mean_divergence"],
                             tuple(r["pair"])))
    truncated = len(rows) > int(budget)
    evaluated = rows[:int(budget)] if truncated else rows
    harder_than = max((max(pair_div(kind, dists, a, i, j) for a in aids)
                       for (i, j) in witness_set.pairs), default=0.0)
    new_pairs = [r["pair"] for r in evaluated
                 if r["worst_divergence"] <= float(harder_than)]
    indist = [r["pair"] for r in evaluated if r["worst_divergence"] <= _TOL]
    return {
        "budget": int(budget), "stop_rule": "EXHAUST_PAIRS_UP_TO_BUDGET",
        "states_declared": len(states),
        "pairs_total_possible": len(states) * (len(states) - 1) // 2,
        "pairs_already_in_witness_set": len(covered),
        "pairs_same_label_skipped": int(n_same),
        "candidates_opposite_label_uncovered": len(cands),
        "candidates_evaluated": len(evaluated),
        "budget_exhausted": bool(truncated),
        "search_exact_over_declared_states": not truncated,
        "harder_than_threshold": None if harder_than is None else float(harder_than),
        "found": bool(evaluated),
        "new_pairs_this_strictly_harder_or_equal": new_pairs,
        "indistinguishable_pairs_found": indist,
        "n_indistinguishable_found": len(indist),
        "rows": evaluated,
        "claim": ("FOUND_COUNTEREXAMPLE_ROWS" if evaluated
                  else "NO_COUNTEREXAMPLE_FOUND_WITHIN_DECLARED_STATES_AND_BUDGET"),
        "not_claimed": ("「未找到」不等于「不存在」：只在声明状态集×声明预算内搜索"),
        "mode": "MIRROR_KIND",
        "kind": kind,
    }


def lp_lambda_any(kind: str, witness_set: ap.WitnessSet,
                  dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                  costs_raw: Mapping[str, float], floor, pool: Sequence[str],
                  *, force_mirror: bool = False) -> dict:
    """成本份额 LP（λ）：HELLINGER+等权+标准 floor 时委托 `ap.cost_share`；否则镜像。"""
    aids = sorted(pool)
    pairs = witness_set.pairs
    if (kind == "HELLINGER" and not witness_set.weights
            and not isinstance(floor, str) and not force_mirror):
        out = ap.cost_share(witness_set, {a: dists[a] for a in aids},
                            {a: float(costs_raw[a]) for a in aids},
                            share_floor=float(floor))
        return {"lambda": {a: float(v) for a, v in out["lambda"].items()},
                "degenerate": bool(out["degenerate"]),
                "worst_pair": list(out["worst_pair"]),
                "worst_score": float(out["worst_score"]),
                "solver": out["solver"], "mode": "DELEGATE_AP"}
    cshare = cost_share_of(costs_raw, floor)
    mat = np.zeros((len(pairs), len(aids)), dtype=np.float64)
    for r, (i, j) in enumerate(pairs):
        for c, a in enumerate(aids):
            mat[r, c] = pair_div(kind, dists, a, i, j) / float(cshare[a])
    wvec = witness_set.weight_vector()
    weighted = mat * wvec[:, None]
    try:
        from scipy.optimize import linprog
    except Exception as exc:                                # noqa: BLE001
        raise ap.CostShareBackendUnavailable(
            "scipy.optimize.linprog 不可用 (%s): 禁静默降级" % (exc,)) from exc
    n, m = len(aids), len(pairs)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0
    A_ub = np.zeros((m + 1, n + 1))
    b_ub = np.zeros(m + 1)
    A_ub[:m, :n] = -weighted
    A_ub[:m, -1] = 1.0
    A_ub[m, -1] = -1.0
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n + [(None, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status != 0 or res.x is None:
        raise ap.PlanningContractError("cost_share LP 未成功: status=%r msg=%r"
                                       % (res.status, res.message))
    t_opt = float(-res.fun)
    lam = np.asarray(res.x[:n], dtype=np.float64)
    if t_opt <= _TOL:
        row_sum = weighted.sum(axis=0)
        if float(row_sum.sum()) > 0.0:
            lam = row_sum / float(row_sum.sum())
    lam = np.where(np.abs(lam) < 1e-12, 0.0, lam)
    lam = lam / float(lam.sum())
    scores = weighted @ lam
    worst_r = int(np.argmin(scores))
    return {"lambda": {a: float(lam[c]) for c, a in enumerate(aids)},
            "degenerate": bool(float(scores[worst_r]) <= _TOL),
            "worst_pair": list(pairs[worst_r]),
            "worst_score": float(scores[worst_r]),
            "solver": {"backend": "scipy.optimize.linprog", "method": "highs",
                       "status": int(res.status), "message": str(res.message)},
            "mode": "MIRROR_KIND", "kind": kind}


# ---------------------------------------------------------------------------
# 两步前瞻（泛化镜像；HELLINGER+等权+标准 floor 时委托 ap.two_step_lookahead）
# ---------------------------------------------------------------------------

def _posterior_weights(states: Sequence[int], aid: str, y: Any,
                       dists) -> Tuple[Dict[int, float], float]:
    post: Dict[int, float] = {}
    denom = 0.0
    for s in states:
        like = float(dists[aid].get(s, {}).get(y, 0.0))
        post[s] = like
        denom += like
    return post, denom


def mirror_two_step(kind: str, pairs: Sequence[Tuple[int, int]],
                    pair_weights: Mapping[Tuple[int, int], float],
                    dists, cshare_sub: Mapping[str, float]) -> dict:
    """`ap.two_step_lookahead` 的泛化镜像（kind/权重参数化；数学逐式同构）。

    返回 per_action（first_step_score / expected_second_step_score / two_step_score）
    与 two_step_scores。镜像性由 `_work/engine_selfcheck.json` 与委托输出的逐对
    对照监督（|Δ| < 1e-9 才记合格）。
    """
    aids = sorted(dists)
    states = tuple(sorted({s for (i, j) in pairs for s in (int(i), int(j))}))
    if not states:
        raise SystemExit("[policy_variants] 见证集无状态 ⇒ 停")
    prior = {s: 1.0 for s in states}
    dmat = _div_matrix(kind, dists, pairs)

    def w_of(pair) -> float:
        return float(pair_weights.get((int(pair[0]), int(pair[1])), 1.0))

    def worst_score_under(weights: Mapping[int, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        tot_w = float(sum(weights.values()))
        if tot_w <= 0.0:
            raise ap.PlanningContractError("后验权重全零 ⇒ 拒")
        for aid in aids:
            num = 0.0
            for (i, j) in pairs:
                w = float(weights.get(i, 0.0)) * float(weights.get(j, 0.0))
                num += w * w_of((i, j)) * dmat[aid][(i, j)]
            out[aid] = (num / tot_w) / float(cshare_sub[aid])
        return out

    per_action = []
    for a1 in aids:
        outcomes = sorted({y for s in states for y in dists[a1].get(s, {})},
                          key=lambda y: (str(type(y)), str(y)))
        if not outcomes:
            raise ap.PlanningContractError("行动 %s 在见证集状态上无结局支撑 ⇒ 拒" % a1)
        exp2 = 0.0
        p_sum = 0.0
        null_mass = 0.0
        null_outcomes = [y for y in outcomes
                         if all(float(dists[a1].get(s, {}).get(y, 0.0)) >= 1.0 - _TOL
                                for s in states)]
        branches = []
        for y in outcomes:
            post, denom = _posterior_weights(states, a1, y, dists)
            if denom <= 0.0:
                branches.append({"outcome": y, "evidence_probability": 0.0,
                                 "skipped": "ZERO_EVIDENCE_PROB"})
                continue
            nw = {s: post[s] / denom for s in states}
            raw2 = worst_score_under(nw)
            scores2 = {a: float(raw2[a]) for a in aids if a != a1}
            best2 = min(scores2.items(), key=lambda kv: (kv[1], kv[0])) \
                if scores2 else (None, None)
            informative = (denom / len(states)) if y not in null_outcomes else 0.0
            if y in null_outcomes:
                null_mass += denom / len(states)
            exp2 += informative * float(best2[1] if best2[0] is not None else 0.0)
            p_sum += denom
            branches.append({"outcome": y, "evidence_probability": float(denom),
                             "second_step_best": best2[0],
                             "second_step_best_score": (None if best2[0] is None
                                                        else float(best2[1]))})
        first = worst_score_under(prior)[a1]
        per_action.append({"action": a1, "first_step_score": float(first),
                           "expected_second_step_score": float(exp2),
                           "null_outcome_mass_fraction": float(null_mass),
                           "two_step_score": float(first + exp2),
                           "evidence_probability_sum": float(p_sum),
                           "n_branches": len(branches),
                           "branches": branches})
    return {"per_action": per_action,
            "two_step_scores": {r["action"]: r["two_step_score"] for r in per_action},
            "first_step_scores": {r["action"]: r["first_step_score"]
                                  for r in per_action},
            "mode": "MIRROR_KIND"}


def _delegate_two_step(witness_set: ap.WitnessSet, sub, sub_costs, share_floor: float) -> dict:
    ts = ap.two_step_lookahead(witness_set, sub, sub_costs, None,
                               ks_compared=(4, None), share_floor=share_floor)
    return {"per_action": ts["per_action"],
            "two_step_scores": {a: float(v) for a, v in ts["two_step_scores"].items()},
            "first_step_scores": {r["action"]: float(r["first_step_score"])
                                  for r in ts["per_action"]},
            "mode": "DELEGATE_AP"}


# ---------------------------------------------------------------------------
# 引擎（唯一入口）
# ---------------------------------------------------------------------------

def _tie_pick(scored: Mapping[str, float], pool, world) -> str:
    return str(sorted(pool, key=lambda a: (-float(scored[a]), world.costs[a], a))[0])


def _draw_u01(pub) -> float:
    h = hashlib.sha256(canonical_json(pub.as_dict()).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return float(rng.random())


def engine_plan(world, pub, spec_or_name, *, force_mirror: bool = False) -> dict:
    """参数化选点：只吃 PublicStateExpanded；返回 {action_id, detail}。

    `spec_or_name` 可以是 `VARIANT_SPECS` 的名字，或直接给一个 spec 字典
    （身份检查用的默认规格即经此路径）。`force_mirror=True` 时跳过一切委托，
    全走本模块镜像（engine 自检用；不进名册）。
    """
    if isinstance(spec_or_name, str):
        if spec_or_name not in VARIANT_SPECS:
            raise ap.PlanningContractError("未知变体 %r" % (spec_or_name,))
        spec = VARIANT_SPECS[spec_or_name]
        spec_name = spec_or_name
    else:
        spec = dict(spec_or_name)
        for k in DEFAULT_SPEC:
            spec.setdefault(k, DEFAULT_SPEC[k])
        spec_name = "ENGINE_SPEC_DIRECT" if spec_delta(spec) else "ENGINE_DEFAULT"
    kind = spec["divergence"]
    part = tuple(int(s) for s in pub.allowed)
    pairs_all = witness_pairs(world, part)
    if not pairs_all:
        raise ap.PlanningContractError("%s: 允许集内无对立标签对（应由 fallback 处理）"
                                       % world.world_id)
    pairs = pairs_all[:MAX_WITNESS_PAIRS]
    label_of = {int(s): int(world.labels[int(s)]) for s in part}
    w = ap.WitnessSet(tuple(pairs), labels=label_of, name="S3B_W",
                      provenance="统计可能集内对立标签对（字典序前 N）")
    pool = list(pub.legal)
    if not pool:
        raise ap.PlanningContractError("%s: 空合法池" % world.world_id)
    dists = {a: {int(s): world.static.dists_float[a][int(s)] for s in part}
             for a in pool}
    costs_share = cost_share_of({a: float(world.costs_dl[a]) for a in pool},
                                spec["share_floor"])
    # 对手搜索
    if spec["opponent_search"] == "ON":
        opp = opponent_search_any(kind, w, dists, label_of, OPP_BUDGET)
        extra = [tuple(p) for p in opp["new_pairs_this_strictly_harder_or_equal"]
                 if tuple(p) not in set(pairs)]
    else:
        opp = {"budget": OPP_BUDGET, "stop_rule": "OFF_BY_VARIANT", "found": False,
               "candidates_evaluated": 0, "budget_exhausted": False,
               "harder_than_threshold": None, "claim": "OPPONENT_SEARCH_OFF_BY_VARIANT",
               "mode": "OFF_BY_VARIANT"}
        extra = []
    merged = sorted(set(list(pairs) + extra))[:MAX_WITNESS_PAIRS]
    pair_weights: Dict[Tuple[int, int], float] = {}
    if spec["weights"] == "HARD":
        pair_weights = hard_pair_weights(merged, dists, pool, kind)
    w2 = ap.WitnessSet(tuple(merged), labels=label_of, weights=pair_weights,
                       name="S3B_W_HARDENED",
                       provenance="对手搜索扩集（cap=%d）" % MAX_WITNESS_PAIRS)
    lam_res = lp_lambda_any(kind, w2, dists,
                            {a: float(world.costs_dl[a]) for a in pool},
                            spec["share_floor"], pool, force_mirror=force_mirror)
    k = spec["prefilter_k"]
    k_eff = len(pool) if (isinstance(k, str) and k == "FULL") else int(k)
    keep = ap.top_k_filter(costs_share, k_eff)
    kept = [a for a in keep["kept"] if a in dists]
    if not kept:
        raise ap.PlanningContractError("%s: 前筛后无候选" % world.world_id)
    sub = {a: dists[a] for a in kept}
    sub_costs = {a: float(world.costs_dl[a]) for a in kept}
    # 两步 / 第一步
    if spec["two_step"] == "ON":
        delegate_ok = (kind == "HELLINGER" and not pair_weights
                       and not isinstance(spec["share_floor"], str)
                       and not force_mirror)
        if delegate_ok:
            ts = _delegate_two_step(w2, sub, sub_costs, float(spec["share_floor"]))
        else:
            cshare_sub = cost_share_of(sub_costs, spec["share_floor"])
            ts = mirror_two_step(kind, merged, pair_weights, sub, cshare_sub)
    else:
        pairs_seq = merged
        cshare_sub = cost_share_of(sub_costs, spec["share_floor"])
        dmat = _div_matrix(kind, sub, pairs_seq)
        states = tuple(sorted({s for (i, j) in pairs_seq for s in (int(i), int(j))}))
        first: Dict[str, float] = {}
        for aid in sorted(sub):
            num = 0.0
            for (i, j) in pairs_seq:
                wv = float(pair_weights.get((int(i), int(j)), 1.0))
                num += wv * dmat[aid][(int(i), int(j))]
            first[aid] = (num / float(len(states))) / float(cshare_sub[aid])
        ts = {"per_action": [], "two_step_scores": None,
              "first_step_scores": first, "mode": "FIRST_STEP_ONLY"}
    # 选择
    sel = spec["selection"]
    pick_info: Dict[str, Any] = {"rule": sel, "pool_considered": kept}
    if sel == "ARGMAX_TWO_STEP":
        pick = _tie_pick(ts["two_step_scores"], kept, world)
        pick_info["score_used"] = {a: float(ts["two_step_scores"][a]) for a in kept}
    elif sel == "ARGMAX_FIRST_STEP":
        pick = _tie_pick(ts["first_step_scores"], kept, world)
        pick_info["score_used"] = {a: float(ts["first_step_scores"][a]) for a in kept}
    elif sel == "RANDOM_PI":
        # π ∝ λ/c~ 在 kept 候选池上抽样（预注册文字「在 kept 池按 π∝λ/c~」；实现细化，
        # 先于名册钉死并登记）：λ 优先取全池 LP 解；若 kept 上 λ 全零 ⇒ 在 kept 子池
        # 重解同一 cost_share LP；若仍全零（理论退化）⇒ 按基线确定性规则（two_step top）
        # 选择并置 fallback_zero_pi（逐条入账），不得据此宣称被抽中的行动分布。
        lam_full = lam_res["lambda"]
        raw = {a: float(lam_full.get(a, 0.0)) / float(costs_share[a]) for a in kept}
        lam_used = lam_full
        lam_stage = "FULL_POOL_LAMBDA"
        cshare_used = costs_share
        if sum(raw.values()) <= 0.0:
            sub_costs_map = {a: float(world.costs_dl[a]) for a in kept}
            lam_sub = lp_lambda_any(kind, w2, {a: dists[a] for a in kept},
                                    sub_costs_map, spec["share_floor"], kept,
                                    force_mirror=force_mirror)
            cshare_used = cost_share_of(sub_costs_map, spec["share_floor"])
            raw = {a: float(lam_sub["lambda"].get(a, 0.0)) / float(cshare_used[a])
                   for a in kept}
            lam_used = lam_sub["lambda"]
            lam_stage = "KEPT_POOL_RESOLVE"
        tot = sum(raw.values())
        if tot <= 0.0:
            pick = _tie_pick(ts["two_step_scores"], kept, world)
            pick_info.update({"pi": None, "u01_from_public_state": None,
                              "fallback_zero_pi": True,
                              "fallback_rule": "π 全零（两级 λ 均退化）⇒ 按基线 two_step top",
                              "lambda_stage": lam_stage})
        else:
            pi = {a: raw[a] / tot for a in sorted(kept)}
            u = _draw_u01(pub)
            acc = 0.0
            pick = sorted(pi)[-1]
            for a in sorted(pi):
                acc += pi[a]
                if u < acc:
                    pick = a
                    break
            pick_info.update({"pi": pi, "u01_from_public_state": u,
                              "fallback_zero_pi": False,
                              "lambda_stage": lam_stage,
                              "draw_rule": ("u01 = default_rng(sha256(canonical(pub))[:8]).random()；"
                                            "同一公开面重放得同一 u（分支/重放一致）"),
                              "lambda": {a: float(lam_used.get(a, 0.0)) for a in sorted(kept)}})
    else:
        raise ap.PlanningContractError("未知选择规则 %r" % (sel,))
    detail = {
        "variant_name": spec_name,
        "spec": dict(spec),
        "spec_delta_vs_default": spec_delta(spec),
        "divergence_kind": kind,
        "witness_pairs_total": len(pairs_all),
        "witness_pairs_used": len(merged),
        "witness_truncated": bool(len(pairs_all) > len(merged)),
        "opponent": {k2: opp.get(k2) for k2 in
                     ("budget", "stop_rule", "found", "candidates_evaluated",
                      "budget_exhausted", "harder_than_threshold", "claim", "mode")},
        "opponent_extra_pairs": len(extra),
        "cost_share": {a: float(v) for a, v in costs_share.items()},
        "cost_share_degenerate": bool(lam_res.get("degenerate")),
        "top_k_kept": kept,
        "top_k_filtered_out": keep["filtered_out"],
        "two_step_scores": ({a: float(v) for a, v in ts["two_step_scores"].items()}
                            if ts["two_step_scores"] is not None else None),
        "first_step_scores": {a: float(v) for a, v in ts["first_step_scores"].items()},
        "two_step_mode": ts["mode"],
        "selection": pick_info,
        "lambda": {a: float(v) for a, v in lam_res["lambda"].items()},
        "planning_pool": pool,
    }
    return {"action_id": pick, "detail": detail}


# ---------------------------------------------------------------------------
# 分支账（变体版；与 cle.branch_report_expanded 逐字段同构）
# ---------------------------------------------------------------------------

def branch_report_variant(world, trace: dict, spec_or_name) -> dict:
    """镜像 `cle.branch_report_expanded`；变体为确定性挑点 ⇒ 重放无 rng。"""
    spec_tag = spec_or_name if isinstance(spec_or_name, str) else "ENGINE_DEFAULT"
    acted = [x for x in trace["steps"] if x["action_id"] is not None]
    rows = []
    allowed = tuple(int(s) for s in world.initial_allowed)
    used = 0
    executed = ()
    read_so_far = 0
    for k, st in enumerate(acted):
        aid = str(st["action_id"])
        pre_hist = []
        for x in acted[:k]:
            pre_hist.append((str(x["action_id"]), cle._record_from_json(world, x["record_key"])))
        recs, truncated = world.q_support(pre_hist, aid)
        idx_of = {cle._rk_key(r): i for i, (si, r) in enumerate(recs)}
        next_actions, terminal_by_outcome, decisions = {}, {}, {}
        detail_cache = {}
        used2 = used + world.costs[aid]
        ex2 = executed + (aid,)
        read2 = read_so_far | int(world.static.read_masks[aid])
        for si, r in recs:
            idx = idx_of[cle._rk_key(r)]
            allowed_new = tuple(world.eprocess(pre_hist + [(aid, r)]).state_set())
            sb = cle._cheap_terminal(world, allowed_new, used2, ex2, k + 1)
            if sb is not None:
                terminal_by_outcome[idx] = sb
                next_actions[idx] = None
                decisions[idx] = "STOP:" + sb
                continue
            key_allowed = allowed_new
            if key_allowed in detail_cache:
                pick = detail_cache[key_allowed]
            else:
                if len({world.labels[int(s)] for s in allowed_new}) == 1:
                    pub = world.public(allowed_new, used2, ex2, k + 1)
                    import random as _r
                    ch = cle._choose_space_fill(world, pub, _r.Random(0), read2)
                else:
                    pub = world.public(allowed_new, used2, ex2, k + 1)
                    ch = engine_plan(world, pub, spec_or_name)
                pick = str(ch["action_id"])
                detail_cache[key_allowed] = pick
            next_actions[idx] = pick
            decisions[idx] = pick
        acts_only = sorted({v for v in next_actions.values() if v is not None})
        dec_set = sorted(set(decisions.values()))
        rows.append({"t": int(st["t"]), "action_id": aid,
                     "n_outcomes": len(recs), "outcomes_truncated": bool(truncated),
                     "observed_outcome_index": idx_of.get(
                         cle._rk_key(cle._record_from_json(world, st["record_key"]))),
                     "outcome_support_indices": [int(si) for si, _r in recs],
                     "support_catalog_ref": ("branch_diagnostics.json#support_catalog"
                                             "[action_id]（全盘支撑按 record_key 排序）"),
                     "next_action_by_outcome": {str(i): v for i, v in next_actions.items()},
                     "terminal_by_outcome": {str(i): v
                                             for i, v in terminal_by_outcome.items()},
                     "decision_by_outcome": {str(i): d for i, d in decisions.items()},
                     "n_next_distinct": len(acts_only),
                     "n_next_distinct_decisions": len(dec_set),
                     "branch_point": bool(len(acts_only) >= 2),
                     "branch_point_by_decision": bool(len(dec_set) >= 2),
                     "distinct_next_actions": acts_only,
                     "distinct_decisions": dec_set,
                     "caliber_note": ("决策层含廉价公开终末谓词；证书层终末不逐分支模拟"
                                      "（成本上限，见 prereg counterfactual）")})
        allowed = tuple(world.eprocess(
            pre_hist + [(aid, cle._record_from_json(world, st["record_key"]))]).state_set())
        used += world.costs[aid]
        executed = executed + (aid,)
        read_so_far |= int(world.static.read_masks[aid])
    return {"world_id": world.world_id, "strategy": spec_tag,
            "seed": int(trace.get("seed", -1)),
            "policy_seed": int(trace.get("policy_seed", -1)),
            "rows": rows,
            "has_branch_point": bool(any(r["branch_point"] for r in rows)),
            "has_branch_point_by_decision": bool(
                any(r["branch_point_by_decision"] for r in rows)),
            "n_branch_points": int(sum(1 for r in rows if r["branch_point"])),
            "n_branch_points_by_decision": int(
                sum(1 for r in rows if r["branch_point_by_decision"])),
            "all_steps_invariant": bool(all(r["n_next_distinct"] <= 1 for r in rows)),
            "pre_planned": bool(all(r["n_next_distinct"] <= 1 for r in rows)),
            "notes": {"rng_handling": "变体为公开面确定性挑点（RANDOM_PI 由公开面派生 u01）",
                      "enumeration": ("结局 = 该步行动在 q_t 下的正概率记录；"
                                      "全盘支撑为声明上界（cap=%d）"
                                      % world.branch_outcome_cap)}}


# ---------------------------------------------------------------------------
# 校验器（控制面用；与 cle.VALIDATORS 同签名的子集 + S3-B 专属）
# ---------------------------------------------------------------------------

def _iter_runs(ledger):
    return list(ledger.get("runs", []))


def v_ledger_nonempty(ledger, ctx=None):
    if not _iter_runs(ledger):
        raise SystemExit("[s3b] 空账：拒绝记 PASS")
    if not {r["strategy"] for r in _iter_runs(ledger)}:
        raise SystemExit("[s3b] 空策略面：拒绝记 PASS")
    return []


def v_config_uniform(ledger, ctx=None):
    pre = (ctx or {}).get("prereg") or {}
    cfg = pre.get("experimental_config") or {}
    bad = []
    for r in _iter_runs(ledger):
        if int(r["budget"]) != int(cfg["budget_int"]):
            bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "key": "budget"})
        if str(r["alpha_seq"]) != str(cfg["alpha_seq"]):
            bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "key": "alpha_seq"})
        if int(r["max_steps"]) != int(cfg["max_steps"]):
            bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "key": "max_steps"})
        if str(r["tier"]) != str(cfg["tier"]):
            bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "key": "tier"})
    return bad


def v_denominator(ledger, ctx=None):
    roster = ledger.get("roster") or {}
    if not roster:
        raise SystemExit("[s3b] 空名册：拒绝记 PASS")
    want = {(w, s, int(k), ss) for ss in roster.get("seed_sets", {})
            for w in roster.get("worlds", [])
            for s in roster.get("strategies", [])
            for k in roster["seed_sets"][ss]}
    got = [(r["world_id"], r["strategy"], int(r["seed"]), r["seed_set"])
           for r in _iter_runs(ledger)]
    bad = []
    if len(got) != len(set(got)):
        bad.append({"code": "DUPLICATE_RUN"})
    for m in sorted(want - set(got)):
        bad.append({"code": "MISSING_RUN", "run": list(m)})
    for e in sorted(set(got) - want):
        bad.append({"code": "EXTRA_RUN", "run": list(e)})
    for r in _iter_runs(ledger):
        cls = (r.get("stop") or {}).get("class")
        if cls not in cle.OUTCOME_CLASSES:
            bad.append({"code": "UNKNOWN_OUTCOME_CLASS", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "class": cls})
    return bad


def v_budget(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        used = 0
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            if used + int(s["cost"]) > int(r["budget"]):
                bad.append({"code": "OVER_BUDGET", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            used += int(s["cost"])
        if used != int(r["used"]):
            bad.append({"code": "USED_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    return bad


def v_certificates(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        terminal_t = None
        for s in r["steps"]:
            c = s.get("certificate")
            if not c:
                bad.append({"code": "MISSING_CERTIFICATE_FIELD", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
                continue
            if c["issued"] and not c["verified"]:
                bad.append({"code": "CERTIFICATE_NOT_VERIFIED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            if c["issued"] and int(c.get("allowed_n", 0)) <= 0:
                bad.append({"code": "CERTIFICATE_ON_EMPTY_SET", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            if c["issued"]:
                terminal_t = s["t"]
                if str(c.get("label_claim")) not in ("0", "1"):
                    bad.append({"code": "CERTIFICATE_LABEL_INVALID",
                                "world": r["world_id"], "strategy": r["strategy"],
                                "seed": r["seed"], "t": s["t"]})
                ids = s.get("allowed_state_ids")
                if int(r["truth_state"]) in set(ids if ids is not None else []) and \
                        int(c["label_claim"]) != int(r["truth_label"]):
                    bad.append({"code": "WRONG_CERT_WHILE_TRUTH_POSSIBLE",
                                "world": r["world_id"], "strategy": r["strategy"],
                                "seed": r["seed"], "t": s["t"]})
        if (r.get("stop") or {}).get("class") == "CERTIFIED_SINGLE" and terminal_t is None:
            bad.append({"code": "CERTIFIED_WITHOUT_CERTIFICATE", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    return bad


def v_certificate_caliber(ledger, ctx=None):
    """证书口径：α=3/40、domain=ALL_STATES、statistic/AO 口径、budget 三上限。"""
    certs = (ctx or {}).get("certificates") or {}
    bad = []
    if str(certs.get("cert_alpha_seq")) != "3/40":
        bad.append({"code": "CERT_CALIBER_DRIFT", "key": "cert_alpha_seq",
                    "got": certs.get("cert_alpha_seq")})
    for c in (certs.get("certificates") or []):
        body = c.get("certificate") or {}
        s2n = body.get("s2n") or {}
        checks = {
            "s2n.alpha_seq": str(s2n.get("alpha_seq")),
            "s2n.tier": str(s2n.get("tier")),
            "s2n.statistic": str(s2n.get("statistic")),
            "s2n.bound_rule": str(s2n.get("bound_rule")),
            "s2n.domain_mode": str(s2n.get("domain_mode")),
            "domain_mode": str(body.get("domain_mode")),
            "verifier_version": str(body.get("verifier_version")),
        }
        want = {"s2n.alpha_seq": "3/40", "s2n.tier": "N1",
                "s2n.statistic": "JOINT_LIKELIHOOD",
                "s2n.bound_rule": "EXACT_SUP_DENOMINATOR",
                "s2n.domain_mode": "ALL_STATES", "domain_mode": "ALL_STATES",
                "verifier_version": "s2n.verifier/1"}
        for k, w in want.items():
            if checks[k] != w:
                bad.append({"code": "CERT_CALIBER_DRIFT", "key": k,
                            "got": checks[k], "want": w})
        bud = body.get("budget") or {}
        for k, w in (("max_nodes", 64), ("region_cap", 256), ("max_depth", 4),
                     ("domain", "ALL_STATES"),
                     ("statistic", "JOINT_LIKELIHOOD")):
            if str(bud.get(k)) != str(w):
                bad.append({"code": "CERT_CALIBER_DRIFT", "key": "budget.%s" % k,
                            "got": bud.get(k), "want": w})
    return bad


def v_outcome_accounting(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        acted = [s for s in r["steps"] if s["action_id"] is not None]
        if len(acted) != r["n_steps"]:
            bad.append({"code": "STEP_COUNT_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
        for s in acted:
            if s.get("record_key") is None or s.get("record_sha12") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
    return bad


def v_cost_two_sided(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        c = r.get("cost") or {}
        if "observed" not in c:
            bad.append({"code": "COST_MISSING_OBSERVED", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
        if "compute_seconds" not in c:
            bad.append({"code": "COST_MISSING_COMPUTE", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    return bad


def v_variant_identity(ledger, ctx=None):
    """每个变体的 run 记录里 applied spec 与声明差必须恰为单旋钮声明差。"""
    pre = (ctx or {}).get("prereg") or {}
    variants = pre.get("variants") or {}
    if not variants:
        raise SystemExit("[s3b] 缺变体声明面：拒绝记 PASS")
    decl = {k: dict(v.get("delta_vs_default") or {}) for k, v in variants.items()}
    bad = []
    for r in _iter_runs(ledger):
        strat = r["strategy"]
        if strat in BASE_STRATEGIES:
            continue
        rec = r.get("variant") or {}
        spec = rec.get("spec") or {}
        if not spec:
            bad.append({"code": "VARIANT_SPEC_MISSING", "strategy": strat,
                        "world": r["world_id"], "seed": r["seed"]})
            continue
        got_delta = spec_delta(spec)
        want_delta = decl.get(strat, {})
        if got_delta != want_delta:
            bad.append({"code": "VARIANT_PARAMETER_DRIFT", "strategy": strat,
                        "world": r["world_id"], "seed": r["seed"],
                        "got": got_delta, "want": want_delta})
    return bad


def v_branch_consistency(ledger, ctx=None):
    """分支账逐行自洽（镜像 cle.v_counterfactual_consistency 的口径）。"""
    cf = (ctx or {}).get("branches") or ledger.get("branches") or {}
    runs = cf.get("runs") or []
    if not runs:
        raise SystemExit("[s3b] 分支面为空：拒绝记 PASS")
    bad = []
    for r in runs:
        for row in r["rows"]:
            nxt = row.get("next_action_by_outcome") or {}
            n_act = len({v for v in nxt.values() if v is not None})
            if int(row.get("n_next_distinct", -1)) != n_act:
                bad.append({"code": "NEXT_DISTINCT_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row.get("t")})
            if bool(row.get("branch_point")) != bool(n_act >= 2):
                bad.append({"code": "BRANCH_FLAG_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row.get("t")})
            dec = row.get("decision_by_outcome") or {}
            n_dec = len(set(dec.values()))
            if int(row.get("n_next_distinct_decisions", -1)) != n_dec:
                bad.append({"code": "NEXT_DECISIONS_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row.get("t")})
            if bool(row.get("branch_point_by_decision")) != bool(n_dec >= 2):
                bad.append({"code": "BRANCH_BY_DECISION_FLAG_MISMATCH",
                            "world": r["world_id"], "strategy": r["strategy"],
                            "seed": r["seed"], "t": row.get("t")})
        if bool(r.get("pre_planned")) != bool(all(
                int(x.get("n_next_distinct", 0)) <= 1 for x in r["rows"])):
            bad.append({"code": "PRE_PLANNED_FLAG_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    return bad


def v_public_reproducibility(ledger, ctx=None):
    """公开历史重放：每条 run 的逐步行动必须能被同一引擎从公开面复现。

    ctx: {"static":…, "prereg":…, "run_filter": callable|None,
          "make_world": callable(world_id, seed, seed_set)}
    """
    ctx = ctx or {}
    make_world = ctx.get("make_world")
    if make_world is None:
        raise SystemExit("[s3b] 校验缺少世界工厂：拒绝记 PASS")
    filt = ctx.get("run_filter")
    bad = []
    n_replayed = 0
    for r in _iter_runs(ledger):
        if filt is not None and not filt(r):
            continue
        n_replayed += 1
        world = make_world(r["world_id"], int(r["seed"]), r["seed_set"])
        import random as _r
        rng = _r.Random(int(r.get("policy_seed", 0)))
        allowed = tuple(int(s) for s in world.initial_allowed)
        used = 0
        executed = ()
        read_so_far = 0
        steps = []
        strat = r["strategy"]
        try:
            for s in r["steps"]:
                if s["action_id"] is None:
                    continue
                pub = world.public(allowed, used, executed, s["t"] - 1,
                                   world.history_sha12(steps))
                if strat in BASE_STRATEGIES:
                    ch = cle.choose_action_expanded(strat, world, pub, rng, read_so_far)
                elif strat == "ENGINE_DEFAULT":
                    ch = engine_plan(world, pub, dict(DEFAULT_SPEC))
                else:
                    if len({world.labels[int(x)] for x in allowed}) == 1:
                        ch = cle._choose_space_fill(world, pub, _r.Random(0), read_so_far)
                    else:
                        ch = engine_plan(world, pub, strat)
                if str(ch["action_id"]) != str(s["action_id"]):
                    bad.append({"code": "ACTION_NOT_REPRODUCIBLE_FROM_PUBLIC_HISTORY",
                                "world": r["world_id"], "strategy": strat,
                                "seed": r["seed"], "t": s["t"],
                                "recorded": s["action_id"],
                                "recomputed": ch["action_id"]})
                rec = cle._record_from_json(world, s["record_key"])
                steps.append((str(s["action_id"]), rec))
                allowed = tuple(world.eprocess(steps).state_set())
                used += int(s["cost"])
                executed = executed + (str(s["action_id"]),)
                read_so_far |= int(world.static.read_masks[str(s["action_id"])])
        except Exception as exc:                            # noqa: BLE001
            bad.append({"code": "REPLAY_RAISED", "world": r["world_id"],
                        "strategy": strat, "seed": r["seed"], "error": repr(exc)})
    return bad


VALIDATORS = {
    "v_ledger_nonempty": v_ledger_nonempty,
    "v_config_uniform": v_config_uniform,
    "v_denominator": v_denominator,
    "v_budget": v_budget,
    "v_certificates": v_certificates,
    "v_certificate_caliber": v_certificate_caliber,
    "v_outcome_accounting": v_outcome_accounting,
    "v_cost_two_sided": v_cost_two_sided,
    "v_variant_identity": v_variant_identity,
    "v_branch_consistency": v_branch_consistency,
    "v_public_reproducibility": v_public_reproducibility,
}


def validate_ledger_variants(ledger, ctx=None) -> dict:
    """跑全部校验器；空面 SystemExit；单器异常记 VALIDATOR_RAISED（不打死）。"""
    out = {}
    for name, fn in VALIDATORS.items():
        try:
            out[name] = fn(ledger, ctx)
        except SystemExit:
            raise
        except Exception as exc:                            # noqa: BLE001
            out[name] = [{"code": "VALIDATOR_RAISED", "validator": name,
                          "error": repr(exc)}]
    return out
