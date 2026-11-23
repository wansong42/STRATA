# -*- coding: utf-8 -*-
"""S2-N · 含噪证书：统计区域排除 + 拓扑 must/may + 独立验证器。

判据权威 = 任务单 S2-N（sha12 ``76c74ff4de77``）§4 + ``results/s2n_noisy_certificate/prereg_s2n.json``。
本件是 §2.1 要求的新建模块；``sequential_evidence.py``（S1 e-process）与
``certificate_solver.py``（S2-N0 证书/验证器）**一字未改**，只 import 与复用。

三条强制语义（与 v2.0 §7.3 逐条对应）：

1. **统计上界**。对区域 ``B`` 与时刻 ``u``，``U_u(B) = sup_{theta in B} log D_u(theta)``，
   其中 ``D_u(theta) = prod_{s<=u} p_theta(y_s | H_{s-1}, a_s)`` 是**同一个 theta 全程固定**的
   联合似然（不是逐步上确界的乘积）。精确实现：``D_u`` 用 ``Fraction`` 累乘，``sup`` 用精确
   有理数比较，阈值比较 ``Q_u / sup_D_u >= 1/alpha_seq`` 也是精确的。
   ``STEPWISE_ENVELOPE`` 只作外包校验（它 >= 联合 sup，用作界是保守的），不作主读数；
   反过来拿联合 sup 去界逐步统计量在本件里被显式拒绝。
2. **证书**。只有（剩余区域非空）且（全部剩余终末区域同标签）且（每区有可重放见证）才发证书。
   每个剔除节点写全 ``u / Q_u / sup_D_u / ratio / margin / per_state_emax / direction``。
3. **验证器**。``verify_statistical_certificate`` 不调用 ``solve_partial`` / ``certify`` /
   ``certify_with_statistics`` / ``statistical_exclude``，也不读任何 ``certified`` 布尔；
   它自己重放公开历史、重算每状态 ``D_u/E_u``、重算区域树与三路外包、重放见证。

N0 面：确定性核（eps=0）下 ``Q_u / sup_D_u <= 1 < 1/alpha_seq``，统计剔除恒为 0，
本机件退化为 S2-N0 的 ``solve_partial`` + ``certify``；``s2.certificate/v1`` 的证书由
``verify_statistical_certificate`` 直接转交 ``certificate_solver.verify_certificate``。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import certificate_solver as cs
from . import finite_scope as fs
from . import identification as ident
from .observation_kernel import canonical_sha12
from .sequential_evidence import (
    EProcess, PredictiveMixture, bits_of, record_key, INF,
)

SCHEMA_STAT_CERT = "s2.statistical_certificate/v1"
SCHEMA_STAT_VERIFY = "s2n.verify/v1"
SCHEMA_STAT_UNRESOLVED = "s2n.unresolved/v1"
VERIFIER_VERSION = "s2n.verifier/1"
SELECTION_PROVENANCE = "SYMBOLIC_ENVELOPE+STATISTICAL_BOUND"
BOUND_RULE = "EXACT_SUP_DENOMINATOR"
STAT_JOINT = "JOINT_LIKELIHOOD"
STAT_STEPWISE = "STEPWISE_ENVELOPE"
CERT_KEYS = ("schema", "certificate_id", "model_id", "table_id", "query_id", "n_bits",
             "domain_mode", "allowed_state_ids", "label_claim", "proof_route", "regions",
             "witness", "verifier_version", "selection_provenance", "budget", "costs",
             "stats", "s2n", "watermark")
FORBIDDEN_KEY_TOKENS = ("truth", "oracle", "hidden")
REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_CERTIFIED = "CERTIFIED"
STATUS_MIXED = "MIXED_LABELS"
STATUS_EMPTY = "EMPTY"
STATUS_CONFLICT = "MODEL_CONFLICT"
STATUS_UNRESOLVED = "UNRESOLVED"


class StatisticalCertificateError(ValueError):
    """本线内部契约破裂：一律抛出，不静默降级。"""


# ---------------------------------------------------------------------------
# 精确账本：Q_u（累积分子）与每状态 D_u / E_u（全部时间）
# ---------------------------------------------------------------------------

def _is_inf(x) -> bool:
    return isinstance(x, str) and x == INF


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(str(x))


def _log_float(x) -> Optional[float]:
    if _is_inf(x):
        return None
    v = _frac(x)
    if v <= 0:
        return None
    return math.log(v.numerator) - math.log(v.denominator)


def _steps_of(eprocess) -> Tuple[Tuple[dict, dict], ...]:
    """公开历史：``[(action_spec, public_record), ...]``，由构造方原样挂上。

    禁凭 ``log_rows`` 的 record_key 反推记录（会丢掉 q 的可预测性口径）。
    """
    steps = getattr(eprocess, "s2n_steps", None)
    if steps is None:
        raise StatisticalCertificateError(
            "e-process 未挂公开历史 s2n_steps：统计界必须由公开记录重算，禁凭键重建")
    return tuple(steps)


def attach_steps(eprocess, steps: Sequence[Tuple[dict, dict]]):
    eprocess.s2n_steps = tuple(steps)
    return eprocess


def exact_ledger(eprocess) -> dict:
    """重放公开历史得到 ``Q[u]`` / ``D[u][state]`` / ``E[u][state]``（精确）。

    并与 S1 e-process 自身的读数逐项绑定：``E_t`` 与 ``Emax`` 必须精确相等，
    不等即仪器缺陷（禁出读数）。
    """
    cached = getattr(eprocess, "_s2n_ledger", None)
    if cached is not None:
        return cached
    steps = _steps_of(eprocess)
    rows = list(eprocess.log_rows)
    if len(rows) != len(steps):
        raise StatisticalCertificateError(
            "公开历史步数与 e-process 记录不符 (%d vs %d)" % (len(steps), len(rows)))
    states = tuple(int(s) for s in eprocess.states)
    Q = [Fraction(1, 1)]
    D: List[List[Fraction]] = [[Fraction(1, 1)] * len(states)]
    E: List[List[object]] = [[Fraction(1, 1)] * len(states)]
    P: List[List[Fraction]] = []
    for u, (action, record) in enumerate(steps, start=1):
        row = rows[u - 1]
        if str(row.get("status")) == "MODEL_CONFLICT":
            raise StatisticalCertificateError("e-process 记录到 MODEL_CONFLICT：禁在此状态建界")
        q = _frac(row["q"])
        Q.append(Q[-1] * q)
        p_row: List[Fraction] = []
        d_row: List[Fraction] = []
        for i, s in enumerate(states):
            p = eprocess.mixture.state_probability(record, s, action)
            p_row.append(p)
            d_row.append(D[-1][i] * p)
        P.append(p_row)
        D.append(d_row)
        E.append([INF if d == 0 else Q[-1] / d for d in d_row])
    # 与 S1 现读绑定
    for i, s in enumerate(states):
        e_now = eprocess.E[s]
        mine = E[-1][i]
        if _is_inf(e_now) != _is_inf(mine):
            raise StatisticalCertificateError(
                "状态 %d 的 E_t 无穷性不符 (e-process=%r, ledger=%r)" % (s, e_now, mine))
        if not _is_inf(e_now) and _frac(e_now) != mine:
            raise StatisticalCertificateError("状态 %d 的 E_t 不精确相等" % (s,))
        # EProcess.Emax 初值为 E_0=1：历史最大证据必须含 t=0 这一项
        emax_mine = _max_e([Fraction(1, 1)] + [E[u][i] for u in range(1, len(E))])
        emax_now = eprocess.Emax[s]
        if _is_inf(emax_now) != _is_inf(emax_mine):
            raise StatisticalCertificateError("状态 %d 的 Emax 无穷性不符" % (s,))
        if not _is_inf(emax_now) and _frac(emax_now) != emax_mine:
            raise StatisticalCertificateError("状态 %d 的 Emax 不精确相等" % (s,))
    led = {"states": states, "t": len(steps), "Q": Q, "D": D, "E": E, "P": P,
           "action_ids": [str(a["action_id"]) for a, _ in steps],
           "record_keys": [record_key(r) for _, r in steps],
           "alpha_seq": str(eprocess.alpha_seq), "threshold": str(eprocess.threshold)}
    eprocess._s2n_ledger = led
    return led


def _max_e(vals: Sequence[object]):
    best = Fraction(0, 1)
    for v in vals:
        if _is_inf(v):
            return INF
        if v > best:
            best = v
    return best


def _index_of(ledger, state: int) -> int:
    try:
        return ledger["states"].index(int(state))
    except ValueError:
        raise StatisticalCertificateError("状态 %d 不在 e-process 状态集内" % (state,))


# ---------------------------------------------------------------------------
# 区域（StatisticalRegion）
# ---------------------------------------------------------------------------

@dataclass
class StatisticalRegion:
    """区域树节点：cubes（互斥子立方）+ 统计剔除证据 + must/may 外包标签。"""

    region_id: str
    parent_id: Optional[str]
    depth: int
    cubes: Tuple[cs.Cube, ...]
    label_claim: Optional[str] = None
    proof_route: Optional[str] = None
    witness: Optional[dict] = None
    envelope_counts: Optional[dict] = None
    excluded: bool = False
    exclusion: Optional[dict] = None
    logE_upper_by_state: Optional[dict] = None

    @property
    def is_terminal(self) -> bool:
        return self.label_claim is not None or bool(self.excluded)

    def state_ids(self, n_bits: int) -> List[int]:
        return cs.region_states(self.cubes, n_bits)

    def to_dict(self, n_bits: int) -> dict:
        st = self.state_ids(n_bits)
        out = {"region_id": self.region_id, "parent_id": self.parent_id,
               "depth": int(self.depth), "is_terminal": bool(self.is_terminal),
               "cubes": [c.to_dict() for c in self.cubes],
               "state_count": len(st), "label_claim": self.label_claim,
               "proof_route": self.proof_route, "envelope_counts": self.envelope_counts,
               "witness": self.witness, "excluded": bool(self.excluded),
               "statistical": self.exclusion,
               "logE_upper_by_state": self.logE_upper_by_state}
        if self.is_terminal:
            out["state_ids"] = st
        else:
            out["state_ids_head"] = st[:8]
        return out


@dataclass
class StatisticalCertificate:
    certificate_id: str
    model_id: str
    table_id: str
    query_id: str
    n_bits: int
    domain_mode: str
    allowed_state_ids: Tuple[int, ...]
    label_claim: str
    proof_route: str
    regions: Tuple[StatisticalRegion, ...]
    witness: dict
    budget: dict
    costs: dict
    stats: dict
    s2n: dict
    watermark: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_STAT_CERT, "certificate_id": self.certificate_id,
            "model_id": self.model_id, "table_id": self.table_id,
            "query_id": self.query_id, "n_bits": int(self.n_bits),
            "domain_mode": self.domain_mode,
            "allowed_state_ids": [int(s) for s in self.allowed_state_ids],
            "label_claim": self.label_claim, "proof_route": self.proof_route,
            "regions": [r.to_dict(self.n_bits) for r in self.regions],
            "witness": self.witness, "verifier_version": VERIFIER_VERSION,
            "selection_provenance": SELECTION_PROVENANCE,
            "budget": self.budget, "costs": self.costs, "stats": self.stats,
            "s2n": self.s2n, "watermark": self.watermark,
        }


@dataclass
class StatisticalUnresolved:
    query_id: str
    status: str
    reason: str
    detail: dict
    partition: Tuple[StatisticalRegion, ...]
    stats: dict
    s2n: dict = field(default_factory=dict)

    def to_dict(self, n_bits: int) -> dict:
        return {"schema": SCHEMA_STAT_UNRESOLVED, "query_id": self.query_id,
                "status": self.status, "reason": self.reason, "detail": self.detail,
                "partition": [r.to_dict(n_bits) for r in self.partition],
                "stats": self.stats, "s2n": self.s2n}


# ---------------------------------------------------------------------------
# 统计区域排除（主判据）
# ---------------------------------------------------------------------------

def statistical_exclude(region: StatisticalRegion, eprocess, alpha_seq) -> dict:
    """区域统计排除：存在 ``u<=t`` 使 ``Q_u / sup_{theta in B} D_u(theta) >= 1/alpha``。

    返回 ``excluded / bound_evidence / unresolved`` 三键。**禁止**把当前最大 E 当区域上界，
    也**禁止**把逐步上确界乘积当联合上确界（后者只作外包校验）。
    """
    alpha = _frac(alpha_seq)
    if not (0 < alpha < 1):
        raise StatisticalCertificateError("alpha_seq 非法: %r" % (alpha_seq,))
    thr = 1 / alpha
    ledger = exact_ledger(eprocess)
    n_bits = eprocess.mixture._d
    states = [int(s) for s in region.state_ids(n_bits)]
    if not states:
        return {"excluded": False, "bound_evidence": None,
                "unresolved": {"reason": "EMPTY_REGION",
                               "note": "空区域不得建外包、不得发证书（恒真谓词禁当证据）"}}
    idx = [_index_of(ledger, s) for s in states]
    per_state_emax = {str(s): (str(eprocess.Emax[int(s)]) if not _is_inf(eprocess.Emax[int(s)])
                               else INF) for s in states}
    per_state_excluded_at = {}
    for s, i in zip(states, idx):
        first = None
        for u in range(1, len(ledger["E"])):
            v = ledger["E"][u][i]
            if _is_inf(v) or _frac(v) >= thr:
                first = u
                break
        per_state_excluded_at[str(s)] = first
    # 逐 u 的精确界
    rows = []
    u_star = None
    best = None
    for u in range(1, len(ledger["D"])):
        d_vals = [ledger["D"][u][i] for i in idx]
        sup_d = max(d_vals)
        argmax_state = states[d_vals.index(sup_d)]
        q = ledger["Q"][u]
        if sup_d == 0:
            ratio = INF
            margin = None
        else:
            ratio = q / sup_d
            margin = ratio - thr
        passes = _is_inf(ratio) or _frac(ratio) >= thr
        # 逐步上确界外包（只作校验：它 >= 联合 sup，用作界保守）
        step_sup = Fraction(1, 1)
        for v in range(1, u + 1):
            step_sup *= max(ledger["P"][v - 1][i] for i in idx)
        step_ratio = INF if step_sup == 0 else q / step_sup
        if step_sup != 0 and (sup_d == 0 or step_sup < sup_d):
            raise StatisticalCertificateError(
                "逐步上确界小于联合 sup（%s < %s）：界方向破裂" % (step_sup, sup_d))
        rows.append({"u": u, "Q_u": str(q), "sup_D_u": str(sup_d),
                     "sup_D_argmax_state": int(argmax_state),
                     "ratio_exact": (INF if _is_inf(ratio) else str(ratio)),
                     "margin_exact": (None if margin is None else str(margin)),
                     "passes": bool(passes),
                     "logQ_u_float": _log_float(q), "U_u_float": _log_float(sup_d),
                     "stepwise": {"sup_D_u": str(step_sup),
                                  "support": bool(_is_inf(step_ratio) or _frac(step_ratio) >= thr),
                                  "note": "逐步上确界乘积（>= 联合 sup）：只作保守外包校验"}})
        if passes and u_star is None:
            u_star = u
            best = rows[-1]
            break
    if u_star is None:
        last = rows[-1]
        return {"excluded": False, "bound_evidence": None,
                "unresolved": {"reason": "REGION_BOUND_BELOW_THRESHOLD",
                               "u": last["u"], "ratio_exact": last["ratio_exact"],
                               "threshold": str(thr),
                               "note": "无历史时刻满足 Q_u/sup_D_u >= 1/alpha：该区域未被统计排除"}}
    evidence = {
        "excluded": True, "u": int(u_star), "statistic": STAT_JOINT,
        "bound_rule": BOUND_RULE, "direction": "UPPER_BOUND_ON_LOG_DENOMINATOR",
        "Q_u": best["Q_u"], "sup_D_u": best["sup_D_u"],
        "sup_D_argmax_state": best["sup_D_argmax_state"],
        "ratio_exact": best["ratio_exact"], "threshold": str(thr),
        "margin_exact": best["margin_exact"],
        "logQ_u_float": best["logQ_u_float"], "U_u_float": best["U_u_float"],
        "log_margin_float": (None if best["margin_exact"] is None
                             else _log_float(_frac(best["margin_exact"]) + thr)),
        "alpha_seq": str(alpha), "state_ids": states,
        "per_state_emax": per_state_emax,
        "per_state_excluded_at": per_state_excluded_at,
        "stepwise_envelope": best["stepwise"],
        "state_bound_evals": int(len(states) * u_star),
    }
    return {"excluded": True, "bound_evidence": evidence, "unresolved": None}


# ---------------------------------------------------------------------------
# 生产侧：区域细化 + 证书组装
# ---------------------------------------------------------------------------

def _claim_of(label: str) -> Optional[str]:
    if label == ident.RegionLabel.ALL_CONNECTED.value:
        return "1"
    if label == ident.RegionLabel.ALL_DISCONNECTED.value:
        return "0"
    return None


def _bfs_path_nodes(derivation: ident.StateDerivation) -> List:
    adj: Dict = {}
    for (u, v) in sorted(derivation.edges):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    starts = sorted(derivation.a_side & derivation.nodes, key=str)
    goals = derivation.b_side & derivation.nodes
    if not starts or not goals:
        return []
    prev: Dict = {}
    seen = set(starts)
    frontier = list(starts)
    for x in frontier:
        if x in goals:
            return [x]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj.get(x, ()):
                if y in seen:
                    continue
                seen.add(y)
                prev[y] = x
                if y in goals:
                    path = [y]
                    cur = y
                    while cur in prev:
                        cur = prev[cur]
                        path.append(cur)
                    return list(reversed(path))
                nxt.append(y)
        frontier = nxt
    return []


def _witness_for(table: fs.FiniteTable, query_id: str, cubes: Sequence[cs.Cube],
                 claim: str, n_bits: int, feasible: Optional[Sequence[int]] = None) -> dict:
    if table.derivation is None:
        raise StatisticalCertificateError("表未提供三要素派生：不得出见证")
    region_states = cs.region_states(cubes, n_bits)
    pool = sorted(set(int(x) for x in (feasible or [])) & set(region_states))
    s = pool[0] if pool else min(c.min_state() for c in cubes)
    d = table.derivation(query_id, int(s))
    if int(d.label) != int(claim):
        raise StatisticalCertificateError(
            "见证状态 %d 的派生标签 %r 与区域声明 %r 不符：仪器缺陷" % (s, d.label, claim))
    w = {"state_id": int(s), "label": int(claim), "replay_ok": True}
    if claim == "1":
        path = _bfs_path_nodes(d)
        if len(path) >= 2:
            w["path"] = path
        else:
            # 单元盘同时附着 A/B（三要素直接连通）：无 2 节点以上的边路，
            # 见证合法性由 state_connected 重放承担，path 置 None 且显式登记。
            w["path"] = None
            w["direct_attach"] = True
    return w


def _root_cubes(table: fs.FiniteTable, query_id: str, domain_mode: str) -> List[cs.Cube]:
    if domain_mode == "ALL_STATES":
        if query_id not in ("ALL", "*"):
            # 仍允许带查询名，但域一律是全状态：语义只看 domain_mode
            pass
        return [cs.Cube(0, 0)]
    if domain_mode == "CONSISTENT_SET":
        return cs.resolve_query_cubes(table, query_id)
    raise StatisticalCertificateError("未知 domain_mode=%r" % (domain_mode,))


def certify_with_statistics(table: fs.FiniteTable, eprocess, query_id: str,
                            budget: Optional[Mapping] = None,
                            alpha_seq=None) -> "StatisticalCertificate | StatisticalUnresolved":
    """统计剔除 + 符号化 must/may 细化；任何不合规一律降 UNRESOLVED（不猜、不删）。"""
    t0 = time.perf_counter()
    alpha = _frac(alpha_seq if alpha_seq is not None else eprocess.alpha_seq)
    bud = {"domain": "CONSISTENT_SET", "max_nodes": 24, "region_cap": 4096,
           "max_depth": int(table.n_bits), "statistic": STAT_JOINT}
    bud.update(dict(budget or {}))
    if bud["statistic"] != STAT_JOINT:
        raise StatisticalCertificateError("主读数只许 %s（逐步外包只作校验）" % STAT_JOINT)
    struct = cs.structure_of(table)
    n_bits = int(table.n_bits)
    root_cubes = _root_cubes(table, query_id, str(bud["domain"]))
    stats = {"nodes": 0, "regions": 0, "statistical_exclude_calls": 0,
             "statistical_exclusions": 0, "state_bound_evals": 0,
             "envelope_calls": 0, "split_calls": 0, "witness_calls": 0}
    if not root_cubes:
        stats = dict(stats, certify_seconds=round(time.perf_counter() - t0, 6))
        return StatisticalUnresolved(
            query_id=query_id, status=STATUS_CONFLICT,
            reason="MODEL_CONFLICT_EMPTY_ALLOWED",
            detail={"note": "允许集为空：观测不在前向像内，无区域可建外包"}, partition=(),
            stats=stats, s2n={"alpha_seq": str(alpha), "tier": str(eprocess.mixture.tier)})
    allowed = cs.region_states(root_cubes, n_bits)
    regions: List[StatisticalRegion] = []
    stack: List[StatisticalRegion] = [StatisticalRegion("R0", None, 0, tuple(root_cubes))]
    seq = 1
    while stack:
        r = stack.pop()
        stats["nodes"] += 1
        regions.append(r)
        stats["regions"] = len(regions)
        if stats["nodes"] > int(bud["max_nodes"]):
            stats = dict(stats, certify_seconds=round(time.perf_counter() - t0, 6))
            return StatisticalUnresolved(
                query_id=query_id, status=STATUS_UNRESOLVED,
                reason="SPLIT_BUDGET_EXHAUSTED",
                detail={"max_nodes": int(bud["max_nodes"]), "nodes": stats["nodes"]},
                partition=tuple(x for x in regions if x.is_terminal), stats=stats,
                s2n={"alpha_seq": str(alpha), "tier": str(eprocess.mixture.tier)})
        ex = statistical_exclude(r, eprocess, alpha)
        stats["statistical_exclude_calls"] += 1
        if ex["excluded"]:
            r.excluded = True
            r.exclusion = ex["bound_evidence"]
            stats["statistical_exclusions"] += 1
            stats["state_bound_evals"] += int(ex["bound_evidence"]["state_bound_evals"])
            continue
        stats["state_bound_evals"] += len(r.state_ids(n_bits)) * max(1, int(eprocess.t))
        env = cs.symbolic_envelope(struct, r.cubes, n_bits)
        stats["envelope_calls"] += 1
        claim = _claim_of(ident.region_label(env))
        r.logE_upper_by_state = {
            str(s): (None if _is_inf(eprocess.Emax[int(s)]) else _log_float(eprocess.Emax[int(s)]))
            for s in r.state_ids(n_bits)}
        if claim is not None:
            r.label_claim = claim
            r.proof_route = (ident.ProofRoute.MUST_ENVELOPE.value if claim == "1"
                             else ident.ProofRoute.MAY_ENVELOPE_EXCLUSION.value)
            r.envelope_counts = env.element_counts()
            thr_ = 1 / alpha
            feas = [s for s in r.state_ids(n_bits)
                    if not (_is_inf(eprocess.Emax[int(s)])
                            or _frac(eprocess.Emax[int(s)]) >= thr_)]
            r.witness = _witness_for(table, query_id, r.cubes, claim, n_bits, feasible=feas)
            r.witness["feasible"] = bool(feas)
            stats["witness_calls"] += 1
            continue
        if r.depth >= int(bud["max_depth"]):
            stats = dict(stats, certify_seconds=round(time.perf_counter() - t0, 6))
            return StatisticalUnresolved(
                query_id=query_id, status=STATUS_UNRESOLVED, reason="DEPTH_EXHAUSTED",
                detail={"depth": r.depth, "region_id": r.region_id},
                partition=tuple(x for x in regions if x.is_terminal), stats=stats,
                s2n={"alpha_seq": str(alpha), "tier": str(eprocess.mixture.tier)})
        if len(r.cubes) != 1:
            kids = [StatisticalRegion("R%d" % (seq + i), r.region_id, r.depth + 1, (c,))
                    for i, c in enumerate(r.cubes)]
            seq += len(kids)
        else:
            b = cs.choose_split_bit(struct, r.cubes, env, n_bits)
            stats["split_calls"] += 1
            m0 = r.cubes[0].fixed_mask | (1 << b)
            v0 = r.cubes[0].fixed_vals & ~(1 << b)
            kids = [StatisticalRegion("R%d" % seq, r.region_id, r.depth + 1,
                                      (cs.Cube(m0, v0),)),
                    StatisticalRegion("R%d" % (seq + 1), r.region_id, r.depth + 1,
                                      (cs.Cube(m0, v0 | (1 << b)),))]
            seq += 2
        if len(regions) + len(stack) + len(kids) > int(bud["region_cap"]):
            stats = dict(stats, certify_seconds=round(time.perf_counter() - t0, 6))
            return StatisticalUnresolved(
                query_id=query_id, status=STATUS_UNRESOLVED, reason="REGION_CAP_EXHAUSTED",
                detail={"region_cap": int(bud["region_cap"])},
                partition=tuple(x for x in regions if x.is_terminal), stats=stats,
                s2n={"alpha_seq": str(alpha), "tier": str(eprocess.mixture.tier)})
        stack.extend(reversed(kids))
    surviving = [r for r in regions if r.label_claim is not None]
    excluded = [r for r in regions if r.excluded]
    thr = 1 / alpha

    def _feasible(r: StatisticalRegion) -> List[int]:
        out = []
        for s in r.state_ids(n_bits):
            v = eprocess.Emax[int(s)]
            if not (_is_inf(v) or _frac(v) >= thr):
                out.append(int(s))
        return out

    feasible_by_region = {r.region_id: _feasible(r) for r in surviving}
    possible_states = sorted({s for v in feasible_by_region.values() for s in v})
    stats["surviving_terminals"] = len(surviving)
    stats["excluded_terminals"] = len(excluded)
    stats["possible_states"] = len(possible_states)
    stats["certify_seconds"] = round(time.perf_counter() - t0, 6)
    base_s2n = {"alpha_seq": str(alpha), "tier": str(eprocess.mixture.tier),
                "kernel_id": str(getattr(eprocess.mixture.kernel, "kernel_id", "")),
                "history_sha12": history_sha12(eprocess), "t": int(eprocess.t),
                "statistic": STAT_JOINT, "bound_rule": BOUND_RULE,
                "domain_mode": str(bud["domain"]),
                "n_statistical_exclusions": len(excluded),
                "n_possible_states": len(possible_states),
                "n_surviving_states": int(sum(len(r.state_ids(n_bits)) for r in surviving))}
    if not surviving:
        return StatisticalUnresolved(
            query_id=query_id, status=STATUS_EMPTY,
            reason="EMPTY_AFTER_STATISTICAL_EXCLUSION",
            detail={"note": "全域被统计排除 ⇒ S_t 为空：与 MODEL_CONFLICT 同口径，禁发证书",
                    "n_excluded_regions": len(excluded)},
            partition=tuple(excluded), stats=stats, s2n=base_s2n)
    if not possible_states:
        return StatisticalUnresolved(
            query_id=query_id, status=STATUS_EMPTY,
            reason="EMPTY_STATISTICALLY_POSSIBLE_SET",
            detail={"note": "所有声明标签的终末区域都不含统计可行状态 ⇒ S_t 为空：空集不发证书",
                    "n_excluded_regions": len(excluded),
                    "n_surviving_regions": len(surviving)},
            partition=tuple(surviving + excluded), stats=stats, s2n=base_s2n)
    labels = {r.label_claim for r in surviving}
    if len(labels) != 1:
        return StatisticalUnresolved(
            query_id=query_id, status=STATUS_MIXED, reason="MIXED_LABELS",
            detail={"labels": sorted(labels), "terminal_regions": len(surviving),
                    "n_excluded_regions": len(excluded),
                    "n_possible_states": len(possible_states),
                    "note": "剩余区域不同标签：统计歧义未消，禁发单标签证书"},
            partition=tuple(surviving + excluded), stats=stats, s2n=base_s2n)
    claim = labels.pop()
    # 顶层见证必须是统计可行状态（见证空洞 = 未发证书）
    top = None
    for r in surviving:
        if feasible_by_region.get(r.region_id):
            top = dict(r.witness or {})
            break
    if not top:
        raise StatisticalCertificateError("无统计可行见证：禁发证书")
    return StatisticalCertificate(
        certificate_id=str(bud.get("certificate_id") or ("S2NC:%s" % query_id)),
        model_id=table.table_id, table_id=table.table_id, query_id=query_id,
        n_bits=n_bits, domain_mode=str(bud["domain"]), allowed_state_ids=tuple(allowed),
        label_claim=str(claim),
        proof_route=(ident.ProofRoute.MUST_ENVELOPE.value if claim == "1"
                     else ident.ProofRoute.MAY_ENVELOPE_EXCLUSION.value),
        regions=tuple(regions), witness=dict(top), budget=dict(bud),
        costs={"basis": "MEASURED_ONLY", **stats}, stats=stats,
        s2n=dict(base_s2n, certificate_kind="STATISTICAL",
                 n_terminal_regions=len(regions)),
        watermark={"research_line": "S2N_noisy_certificate",
                   "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
                   "noise": str(eprocess.mixture.tier), "dictionary": "COMPLETE_KNOWN",
                   "claim_scope": "DECLARED_SMALL_MODEL"})


# ---------------------------------------------------------------------------
# 公开历史（验证器输入）
# ---------------------------------------------------------------------------

@dataclass
class PublicEvidence:
    """公开可给验证器的一切：核、噪声档、状态集、公开记录序列、标签。"""

    model_id: str
    kernel: object
    tier: str
    states: Tuple[int, ...]
    steps: Tuple[Tuple[dict, dict], ...]
    labels: Dict[int, int] = field(default_factory=dict)
    # 只做声明核似然的记忆化（纯函数缓存），不含任何生产判定
    cache: Optional[dict] = None

    def to_public(self) -> dict:
        return {"model_id": self.model_id, "tier": self.tier,
                "states": [int(s) for s in self.states],
                "steps": [{"action_id": str(a["action_id"]),
                           "record_key": list(record_key(r))} for a, r in self.steps]}


def history_sha12(eprocess_or_public) -> str:
    if isinstance(eprocess_or_public, PublicEvidence):
        doc = eprocess_or_public.to_public()
    else:
        ep = eprocess_or_public
        steps = _steps_of(ep)
        doc = {"model_id": str(getattr(ep, "s2n_model_id", "")),
               "tier": str(ep.mixture.tier),
               "states": [int(s) for s in ep.states],
               "steps": [{"action_id": str(a["action_id"]),
                          "record_key": list(record_key(r))} for a, r in steps]}
    return canonical_sha12(doc)


def run_eprocess(kernel, tier: str, states: Sequence[int], labels: Mapping[int, int],
                 alpha_seq, steps: Sequence[Tuple[dict, dict]], mode: str = "posterior",
                 prior: Optional[Mapping[int, Fraction]] = None,
                 cache: Optional[dict] = None, model_id: str = "") -> EProcess:
    """构造并跑完 S1 e-process；公开历史原样挂上供统计界重算。"""
    states = tuple(int(s) for s in states)
    if not steps:
        raise StatisticalCertificateError("公开历史为空：禁建 e-process")
    mix = PredictiveMixture(states, kernel, steps[0][0], prior=prior, mode=mode, tier=tier,
                            cache=cache)
    ep = EProcess(states, dict(labels), _frac(alpha_seq), mix, kernel)
    attach_steps(ep, steps)
    ep.s2n_model_id = str(model_id)
    for action, record in steps:
        out = ep.update(action, record)
        if str(out.get("status")) == "MODEL_CONFLICT":
            raise StatisticalCertificateError("e-process 进入 MODEL_CONFLICT：禁发证书")
    _ = exact_ledger(ep)   # 与 S1 读数绑定，失败即抛
    return ep


# ---------------------------------------------------------------------------
# 独立验证器
# ---------------------------------------------------------------------------

def _v_states_of(cubes: Iterable[Mapping], n_bits: int, reasons: List[str],
                 rid) -> List[int]:
    out: set = set()
    for c in cubes:
        m, v = int(c["fixed_mask"]), int(c["fixed_vals"])
        if v & ~m:
            reasons.append("区域 %r 子立方非法" % (rid,))
            continue
        ss = [v]
        for b in range(n_bits):
            if not (m >> b) & 1:
                ss = ss + [s | (1 << b) for s in ss]
        out.update(ss)
    return sorted(out)


def _v_explicit_env(struct: cs.TableStructure, states: Sequence[int],
                    n_bits: int) -> ident.Envelope:
    """验证器自带外包（逐状态显式路；不调 symbolic_envelope / envelope_of）。"""
    if not states:
        raise StatisticalCertificateError("空区域不得建外包")
    if struct.kind == "EDGE_BITS":
        node_sets = [frozenset(struct.nodes) for _ in states]
        edge_sets = [frozenset((u, v) for (b, u, v) in struct.edges if (s >> b) & 1)
                     for s in states]
        a_sets = [frozenset(struct.a_nodes) if any((s >> b) & 1 for b in struct.a_incident_bits)
                  else frozenset() for s in states]
        b_sets = [frozenset(struct.b_nodes) if any((s >> b) & 1 for b in struct.b_incident_bits)
                  else frozenset() for s in states]
        attach_a = [bool(x) for x in a_sets]
        attach_b = [bool(x) for x in b_sets]
    else:
        base = len(struct.certain)
        node_sets, edge_sets, a_sets, b_sets = [], [], [], []
        for s in states:
            present = set(range(base))
            for j in range(len(struct.uncertain)):
                if (s >> j) & 1:
                    present.add(base + j)
            node_sets.append(frozenset(present))
            edge_sets.append(frozenset((i, j) for (i, j) in struct.contact_edges
                                       if i in present and j in present))
            a_sets.append(frozenset(x for x in present if x in struct.attach_a_disks))
            b_sets.append(frozenset(x for x in present if x in struct.attach_b_disks))
        attach_a = [bool(x) for x in a_sets]
        attach_b = [bool(x) for x in b_sets]
    must_nodes = frozenset.intersection(*node_sets)
    may_nodes = frozenset.union(*node_sets)
    must_edges = frozenset.intersection(*edge_sets)
    may_edges = frozenset.union(*edge_sets)
    must_a = frozenset.intersection(*a_sets)
    may_a = frozenset.union(*a_sets)
    must_b = frozenset.intersection(*b_sets)
    may_b = frozenset.union(*b_sets)
    return ident.Envelope(len(states), must_nodes, may_nodes, must_edges, may_edges,
                          must_a, may_a, must_b, may_b,
                          all(attach_a), any(attach_a), all(attach_b), any(attach_b))


def _v_direction(env: ident.Envelope) -> str:
    """验证器自带方向规则（独立于 ident.must_connected/may_connected）。"""
    def reach(nodes, edges, sa, sb) -> bool:
        sa, sb = set(sa) & set(nodes), set(sb) & set(nodes)
        if not sa or not sb:
            return False
        adj: Dict = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        seen, queue = set(sa), list(sa)
        head = 0
        while head < len(queue):
            x = queue[head]
            head += 1
            if x in sb:
                return True
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        return False

    must = reach(env.must_nodes, env.must_edges, env.must_a_side, env.must_b_side)
    may = reach(env.may_nodes, env.may_edges, env.may_a_side, env.may_b_side)
    if must and not may:
        return "INSTRUMENT_DEFECT_MONOTONICITY"
    if must:
        return ident.RegionLabel.ALL_CONNECTED.value
    if not may:
        return ident.RegionLabel.ALL_DISCONNECTED.value
    return ident.RegionLabel.UNRESOLVED.value


def _v_consistent_states(table: fs.FiniteTable, query_id: str) -> List[int]:
    """验证器自带的 N0 一致集（位运算直算；不调 resolve_query_cubes / _v_allowed）。"""
    meta = table.tags.get("s2_queries")
    meta = meta if isinstance(meta, dict) else json.loads(meta or "{}")
    kind = str(meta.get("kind"))
    n = int(table.n_bits)
    if kind == "A12_CLASS":
        code = int(str(query_id).split(":")[-1])
        codes = table.baseline_codes()
        return [s for s in range(1 << n) if int(codes[s]) == code]
    if kind == "B13_SCENARIO":
        name = str(query_id).split(":")[-1]
        sc = (meta.get("scenarios") or {}).get(name)
        if sc is None:
            raise StatisticalCertificateError("未知情景 %r" % (name,))
        if not sc.get("observation_in_forward_image", True):
            return []
        out = []
        for s in range(1 << n):
            ok = True
            for bit, val in sc["reads"]:
                if ((s >> int(bit)) & 1) != int(val):
                    ok = False
                    break
            if ok:
                out.append(s)
        return out
    raise StatisticalCertificateError("未知查询类型 %r（验证器只认 A12_CLASS/B13_SCENARIO）" % (kind,))


def _v_replay(public: PublicEvidence, reasons: List[str]) -> dict:
    """验证器重放：自带 PredictiveMixture 重算 Q_u 与每状态 D_u/E_u（精确）。"""
    states = tuple(int(s) for s in public.states)
    if not public.steps:
        reasons.append("公开历史为空：无法验证统计界")
        return {"Q": [Fraction(1, 1)], "D": [[Fraction(1, 1)] * len(states)],
                "E": [[Fraction(1, 1)] * len(states)], "P": [],
                "states": states}
    mix = PredictiveMixture(states, public.kernel, public.steps[0][0],
                            mode="posterior", tier=public.tier, cache=public.cache)
    Q = [Fraction(1, 1)]
    D: List[List[Fraction]] = [[Fraction(1, 1)] * len(states)]
    P: List[List[Fraction]] = []
    for (action, record) in public.steps:
        q = mix.prob(record, action)
        if q == 0:
            reasons.append("公开历史上 q=0：验证器不建界（0/0 禁发证书）")
            break
        Q.append(Q[-1] * q)
        p_row = [mix.state_probability(record, s, action) for s in states]
        P.append(p_row)
        d_row = [D[-1][i] * p_row[i] for i in range(len(states))]
        D.append(d_row)
        w = {}
        tot = Fraction(0, 1)
        for i, s in enumerate(states):
            w[s] = mix.weights[s] * mix.state_probability(record, s, action)
            tot += w[s]
        if tot == 0:
            reasons.append("验证器重放遇到全零似然：MODEL_CONFLICT")
            break
        mix.weights = {s: v / tot for s, v in w.items()}
    E: List[List[object]] = []
    for u in range(len(D)):
        E.append([INF if d == 0 else Q[u] / d for d in D[u]])
    return {"Q": Q, "D": D, "E": E, "P": P, "states": states}


def verify_statistical_certificate(table: fs.FiniteTable, eprocess_public,
                                   query_id: str, certificate: Mapping) -> dict:
    """独立重放：统计界方向/阈值/逐状态排除性、区域树覆盖、must/may、见证、历史身份。

    - ``s2.certificate/v1``（S2-N0 原样证书）⇒ 直接转交 ``certificate_solver.verify_certificate``。
    - ``s2.statistical_certificate/v1`` ⇒ 本函数全程自算，不读任何生产布尔。
    """
    t0 = time.perf_counter()
    if not isinstance(certificate, Mapping):
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False, "reasons": ["证书不是映射"],
                "verifier_version": VERIFIER_VERSION, "verify_seconds": 0.0}
    cert = dict(certificate)
    if str(cert.get("schema")) == cs.SCHEMA_CERT:
        out = cs.verify_certificate(table, query_id, cert)
        out = dict(out)
        out["s2n"] = {"reused_s2_verifier": True,
                      "note": "s2.certificate/v1 原样转交 S2-N0 verify_certificate（接口复用）"}
        out["verify_seconds"] = round(time.perf_counter() - t0, 6)
        return out

    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if str(cert.get("schema")) != SCHEMA_STAT_CERT:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": ["未知证书 schema %r" % (cert.get("schema"),)],
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    unknown = [k for k in cert if k not in CERT_KEYS]
    if unknown:
        reasons.append("证书含未知顶层字段 %s" % (unknown,))
    bad = [k for k in cert if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad:
        reasons.append("证书含真值/神谕字段 %s" % (bad,))
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("selection_provenance") != SELECTION_PROVENANCE:
        reasons.append("选点来源非法 %r" % (cert.get("selection_provenance"),))
    if cert.get("verifier_version") != VERIFIER_VERSION:
        reasons.append("证书 verifier_version 不是 %s" % VERIFIER_VERSION)
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != int(table.n_bits):
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    public = eprocess_public
    if not isinstance(public, PublicEvidence):
        raise StatisticalCertificateError("验证器只接受 PublicEvidence（公开历史对象）")
    if str(cert.get("model_id")) != public.model_id:
        reasons.append("证书 model_id 与公开历史不符")
    s2n = dict(cert.get("s2n") or {})
    if str(s2n.get("history_sha12")) != history_sha12(public):
        reasons.append("公开历史摘要不符：证书不是对这段观测出的")
    if str(s2n.get("tier")) != str(public.tier):
        reasons.append("噪声档不符 (%r vs %r)" % (s2n.get("tier"), public.tier))
    alpha_seq = _frac(s2n.get("alpha_seq", "0"))
    if not (0 < alpha_seq < 1):
        reasons.append("证书 alpha_seq 非法 %r" % (s2n.get("alpha_seq"),))
    thr = 1 / alpha_seq
    n_bits = int(table.n_bits)
    struct = cs.structure_of(table)
    rep = _v_replay(public, reasons)
    Q, D, E, P, vstates = rep["Q"], rep["D"], rep["E"], rep["P"], rep["states"]
    t_hist = len(Q) - 1
    checks["history_t"] = t_hist
    if int(s2n.get("t", -1)) != t_hist:
        reasons.append("证书声明的 t 与公开历史长度不符")

    # 域
    mode = str(cert.get("domain_mode"))
    if mode == "ALL_STATES":
        domain = list(range(1 << n_bits))
    elif mode == "CONSISTENT_SET":
        domain = _v_consistent_states(table, query_id)
    else:
        reasons.append("未知 domain_mode %r" % (mode,))
        domain = []
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if declared_allowed != sorted(domain):
        reasons.append("域名不符：声明 %d / 重算 %d" % (len(declared_allowed), len(domain)))
    if not domain:
        reasons.append("空域：空集不得发证书")

    regions = list(cert.get("regions", []))
    checks["regions"] = len(regions)
    if not regions:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": reasons + ["证书无区域"], "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    ids = [r.get("region_id") for r in regions]
    if len(set(ids)) != len(ids):
        reasons.append("区域 id 重复")
    by_id = {r["region_id"]: r for r in regions}
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        return {"schema": SCHEMA_STAT_VERIFY, "pass": False,
                "reasons": reasons + ["根区域不唯一 (%d)" % len(roots)],
                "verifier_version": VERIFIER_VERSION,
                "verify_seconds": round(time.perf_counter() - t0, 6)}
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append("区域 %r 的父区域不存在" % (r.get("region_id"),))
            kids.setdefault(p, []).append(r)
    st_of = {r["region_id"]: _v_states_of(r.get("cubes", []), n_bits, reasons, r.get("region_id"))
             for r in regions}
    if st_of[roots[0]["region_id"]] != sorted(domain):
        reasons.append("根区域 != 域（%d vs %d）"
                       % (len(st_of[roots[0]["region_id"]]), len(domain)))
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = st_of[r["region_id"]]
        if r.get("is_terminal"):
            if ch:
                reasons.append("终末区域 %r 仍有子区域" % (r.get("region_id"),))
            if not st:
                reasons.append("终末区域 %r 为空" % (r.get("region_id"),))
            continue
        if not ch:
            reasons.append("内部区域 %r 无子区域" % (r.get("region_id"),))
            continue
        union: List[int] = []
        for c in ch:
            union.extend(st_of[c["region_id"]])
        if len(union) != len(set(union)):
            reasons.append("区域 %r 子区域重叠" % (r.get("region_id"),))
        if sorted(set(union)) != st:
            reasons.append("区域 %r 子区域并集 != 父区域" % (r.get("region_id"),))

    # 逐节点：剔除节点重算界；存活节点重算三路外包与见证
    idx_of = {s: i for i, s in enumerate(vstates)}
    audit_states = 0
    n_excl = 0
    n_surv = 0
    claims: set = set()
    witness_holes: List[str] = []
    for r in regions:
        if not r.get("is_terminal"):
            continue
        rid = r.get("region_id")
        st = st_of[rid]
        excluded = bool(r.get("excluded"))
        if excluded:
            n_excl += 1
            stat = dict(r.get("statistical") or {})
            if not stat.get("excluded"):
                reasons.append("区域 %r 标了剔除但 statistical 段非剔除" % (rid,))
                continue
            if r.get("label_claim") is not None:
                reasons.append("剔除区域 %r 不得同时标标签" % (rid,))
                continue
            u = int(stat.get("u", 0))
            if not (1 <= u <= t_hist):
                reasons.append("区域 %r 的 u=%r 越出历史 (1..%d)" % (rid, u, t_hist))
                continue
            statistic = str(stat.get("statistic"))
            if statistic not in (STAT_JOINT, STAT_STEPWISE):
                reasons.append("区域 %r 的统计量 %r 非法" % (rid, statistic))
            if str(stat.get("bound_rule")) != BOUND_RULE:
                reasons.append("区域 %r 的界规则 %r 非法" % (rid, stat.get("bound_rule")))
            d_vals = [D[u][idx_of[s]] for s in st if s in idx_of]
            if len(d_vals) != len(st):
                reasons.append("区域 %r 含公开历史状态集外状态" % (rid,))
                continue
            sup_d = max(d_vals)
            if statistic == STAT_STEPWISE:
                # 逐步上确界乘积：prod_{v<=u} sup_{theta in B} p_theta(y_v)
                step = Fraction(1, 1)
                for v in range(1, u + 1):
                    step *= max(P[v - 1][idx_of[s]] for s in st)
                target = step
            else:
                target = sup_d
            q = Q[u]
            if not _is_inf(stat.get("sup_D_u")):
                declared = _frac(stat.get("sup_D_u"))
                if declared != target:
                    reasons.append("区域 %r 声明的 sup_D_u 与重算（%s）不等"
                                   % (rid, statistic))
                for s, dv in zip(st, d_vals):
                    if dv > declared:
                        reasons.append("区域 %r 的界不是上界：状态 %d 的 D_u=%s > %s"
                                       % (rid, s, dv, declared))
                        break
                if statistic == STAT_STEPWISE and sup_d > declared:
                    reasons.append("区域 %r 用联合上确界去界逐步统计量：联合 sup=%s > 声明 %s"
                                   % (rid, sup_d, declared))
            if not _is_inf(stat.get("Q_u")) and _frac(stat.get("Q_u")) != q:
                reasons.append("区域 %r 声明的 Q_u 与重算不等" % (rid,))
            # 阈值：整区每个状态都必须已被排除（这一条同时打掉"当前最大 E 当上界"）
            survivors = []
            for s in st:
                i = idx_of[s]
                emax = _max_e([E[v][i] for v in range(1, t_hist + 1)])
                if not (_is_inf(emax) or _frac(emax) >= thr):
                    survivors.append(s)
            if survivors:
                reasons.append("剔除区域 %r 仍含未被统计排除的状态 %s"
                               % (rid, survivors[:8]))
            if target != 0 and q / target < thr:
                reasons.append("区域 %r 的 Q_u/界 未达 1/alpha（重算口径 %s）" % (rid, statistic))
            declared_emax = dict(stat.get("per_state_emax") or {})
            for s in st:
                if str(s) not in declared_emax:
                    reasons.append("区域 %r 缺状态 %d 的 per_state_emax" % (rid, s))
                    break
            audit_states += len(st)
            continue
        if not excluded:
            n_surv += 1
            claim = r.get("label_claim")
            if claim not in ("0", "1"):
                reasons.append("终末区域 %r 标签非法 %r（既非剔除也非 0/1）" % (rid, claim))
                continue
            claims.add(claim)
            if not st:
                reasons.append("终末区域 %r 为空" % (rid,))
                continue
            if not set(st) <= set(domain):
                reasons.append("终末区域 %r 含域外状态" % (rid,))
            if table.derivation is None:
                reasons.append("表未提供派生：无法重算外包")
                continue
            derivs = [table.derivation(query_id, int(s)) for s in st]
            audit_states += len(st)
            env_set = ident.envelope_of(derivs)
            env_q = ident.envelope_of_quantifier(derivs)
            env_exp = _v_explicit_env(struct, st, n_bits)
            if env_set.to_dict() != env_q.to_dict():
                reasons.append("区域 %r 集合路与逐元素路不等" % (rid,))
            if env_exp.to_dict() != env_set.to_dict():
                reasons.append("区域 %r 验证器显式路与集合路不等" % (rid,))
            declared_env = r.get("envelope_counts")
            if declared_env is not None and declared_env != env_set.element_counts():
                reasons.append("区域 %r 声明外包计数与重算不等" % (rid,))
            lab = _v_direction(env_exp)
            if lab == "INSTRUMENT_DEFECT_MONOTONICITY":
                reasons.append("区域 %r 外包单调性破裂" % (rid,))
                continue
            want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                    else ident.RegionLabel.ALL_DISCONNECTED.value)
            if lab != want:
                reasons.append("区域 %r 声明 %s 但重算 %s" % (rid, claim, lab))
            for d in derivs:
                if int(d.label) != int(1 if ident.state_connected(d) else 0):
                    reasons.append("区域 %r 状态 %d 派生与标签不忠实" % (rid, d.state_id))
                    break
                if int(d.label) != int(claim):
                    reasons.append("区域 %r 状态 %d 标签 %d != 声明 %s"
                                   % (rid, d.state_id, d.label, claim))
                    break
            w = r.get("witness")
            if not w or "state_id" not in w:
                reasons.append("区域 %r 无见证" % (rid,))
            else:
                sid = int(w["state_id"])
                if sid not in set(st):
                    reasons.append("区域 %r 见证 %d 不在区域内" % (rid, sid))
                elif int(w.get("label", -1)) != int(claim):
                    reasons.append("区域 %r 见证标签与区域标签不符" % (rid,))
                else:
                    rep_w = ident.reverify_witness(table.derivation(query_id, sid),
                                                   int(claim), w.get("path"))
                    if not rep_w["ok"]:
                        reasons.append("区域 %r 见证不可重放 %r" % (rid, rep_w.get("reason")))
                if sid in idx_of:
                    i = idx_of[sid]
                    emax_w = _max_e([E[v][i] for v in range(1, t_hist + 1)])
                    if not _is_inf(emax_w) and _frac(emax_w) >= thr:
                        # 见证的职责是证明该区域标签（区域已由三路外包证明同标签）；
                        # 该状态是否统计可行不影响标签证明，只影响"非空洞"——
                        # 非空洞由证书顶层见证单独承担（见下方硬检查）。
                        witness_holes.append(rid)
    # 覆盖：域内每个状态要么在被剔除节点，要么在被声明标签的存活节点
    surv_states: List[int] = []
    for r in regions:
        if r.get("is_terminal") and not r.get("excluded"):
            surv_states.extend(st_of[r["region_id"]])
    all_terminal: List[int] = []
    for r in regions:
        if r.get("is_terminal"):
            all_terminal.extend(st_of[r["region_id"]])
    if len(all_terminal) != len(set(all_terminal)):
        reasons.append("终末区域划分不互斥")
    if sorted(set(all_terminal)) != sorted(domain):
        missing = sorted(set(domain) - set(all_terminal))
        extra = sorted(set(all_terminal) - set(domain))
        reasons.append("终末区域覆盖 != 域 (缺 %d / 多 %d)" % (len(missing), len(extra)))
    if len(claims) > 1:
        reasons.append("存活终末区域标签不一致 %s：混合查询不得发单标签证书" % (sorted(claims),))
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    else:
        if int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
            reasons.append("证书顶层见证标签与 label_claim 不符")
        sid = int(top["state_id"])
        if sid in idx_of:
            i = idx_of[sid]
            emax_t = _max_e([E[v][i] for v in range(1, t_hist + 1)])
            if not _is_inf(emax_t) and _frac(emax_t) >= thr:
                reasons.append("证书顶层见证状态已被统计排除")
    possible = []
    for s in domain:
        if s not in idx_of:
            continue
        i = idx_of[s]
        emax = _max_e([E[v][i] for v in range(1, t_hist + 1)])
        if not (_is_inf(emax) or _frac(emax) >= thr):
            possible.append(s)
    if not possible:
        reasons.append("统计可行集为空（S_t=∅）：空集不得发证书")
    if int(s2n.get("n_statistical_exclusions", -1)) != n_excl:
        reasons.append("证书声明的剔除节点数与重算不符")
    if int(s2n.get("n_possible_states", -1)) != len(possible):
        reasons.append("证书声明的可行状态数 %r 与重算 %d 不符"
                       % (s2n.get("n_possible_states"), len(possible)))
    if int(s2n.get("n_surviving_states", -1)) != len(set(surv_states)):
        reasons.append("证书声明的存活区域状态数与重算不符")
    if bool(cert.get("label_claim")) is False and cert.get("label_claim") not in ("0", "1"):
        reasons.append("证书 label_claim 非法 %r" % (cert.get("label_claim"),))
    if claims and str(cert.get("label_claim")) not in claims:
        reasons.append("证书 label_claim 与存活区域标签不符")
    return {"schema": SCHEMA_STAT_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION,
            "checks": {"history_t": t_hist, "domain": len(domain), "regions": len(regions),
                       "excluded_terminals": n_excl, "surviving_terminals": n_surv,
                       "surviving_states": len(set(surv_states)),
                       "possible_states": len(possible),
                       "regions_with_excluded_witness": len(witness_holes),
                       "audit_states": audit_states, "claims": sorted(claims),
                       "alpha_seq": str(alpha_seq), "threshold": str(thr)},
            "verify_seconds": round(time.perf_counter() - t0, 6),
            "s2n": {"reused_s2_verifier": False, "independent_replay": True,
                    "calls_production_solver": False}}


# ---------------------------------------------------------------------------
# 声明 bit-flip 核（A12 / B13 含噪压力面）
# ---------------------------------------------------------------------------

class BitFlipChannelKernel:
    """声明观测核：每个被读位独立以 ``eps`` 翻转（tier 决定 eps）。

    状态位序与 ``finite_scope`` 一致（位 b = edge/disk b 的存在性）；
    ``reads[action_id]`` 给出该动作依次读的状态位下标。
    """

    def __init__(self, table_id: str, n_bits: int, reads: Mapping[str, Sequence[int]],
                 eps_by_tier: Mapping[str, float], kind: str = "BSC"):
        self.table_id = str(table_id)
        self.n_bits = int(n_bits)
        self.reads = {str(k): tuple(int(x) for x in v) for k, v in reads.items()}
        for aid, pos in self.reads.items():
            if len(set(pos)) != len(pos) or any(p < 0 or p >= n_bits for p in pos):
                raise StatisticalCertificateError("%s: 读位非法 %r" % (aid, pos))
        self.eps = {str(k): Fraction(str(v)) for k, v in eps_by_tier.items()}
        if any(not (0 <= e <= Fraction(1, 2)) for e in self.eps.values()):
            raise StatisticalCertificateError("eps 只在 [0, 1/2]：>1/2 会反转似然方向")
        self.kind = kind

    def candidate_geometry(self) -> list:
        return list(range(self.n_bits))

    def eps_of(self, tier: str) -> Fraction:
        if tier not in self.eps:
            raise StatisticalCertificateError("未知噪声档 %r" % (tier,))
        return self.eps[tier]

    def make_record(self, action_id: str, observed_bits: Sequence[int]) -> dict:
        pos = self.reads[str(action_id)]
        if len(observed_bits) != len(pos):
            raise StatisticalCertificateError("%s: 观测位数不符" % (action_id,))
        return {"design_id": str(action_id), "well_id": 0,
                "counts": [[i, int(v)] for i, v in enumerate(observed_bits)]}

    def record_probability_exact(self, public_record, state_bits, action_spec, *,
                                 tier: str = "N1", history=None) -> Fraction:
        aid = str(action_spec["action_id"])
        pos = self.reads[aid]
        obs = {int(k): int(v) for k, v in public_record.get("counts", [])}
        if len(obs) != len(pos):
            raise StatisticalCertificateError("%s: 记录位数不符" % (aid,))
        eps = self.eps_of(tier)
        p = Fraction(1, 1)
        for i, bitpos in enumerate(pos):
            v = obs[i]
            actual = int(state_bits[bitpos])
            p *= (1 - eps) if actual == v else eps
            if p == 0:
                return Fraction(0, 1)
        return p

    def simulate(self, state_bits, action_spec, tier: str, rng) -> dict:
        aid = str(action_spec["action_id"])
        eps = float(self.eps_of(tier))
        obs = []
        for bitpos in self.reads[aid]:
            b = int(state_bits[bitpos])
            if rng.random() < eps:
                b = 1 - b
            obs.append(b)
        return self.make_record(aid, obs)


def a12_channel_setup(root: Optional[Path] = None) -> dict:
    """A12：基线 = 逐边读位（finite_scope 把每个 read 拆成单元素组）⇒ 真 bit-flip。"""
    table, meta = cs.a12_table_with_derivation(root)
    base = Path(root) if root else REPO_ROOT
    acts = json.loads((base / "results/connectivity_certification/action_spec.json")
                      .read_text(encoding="utf-8"))["a_layer"]
    by_id = {a["id"]: a for a in acts["actions"]}
    ms = json.loads((base / "results/connectivity_certification/model_spec.json")
                    .read_text(encoding="utf-8"))["a_layer"]
    pos = {str(e[0]): i for i, e in enumerate(ms["edges"])}
    reads = {}
    for a in table.baseline:
        reads[str(a.action_id)] = tuple(pos[str(e)] for e in by_id[str(a.action_id)]["reads"])
        got = tuple(sorted(int(b).bit_length() - 1 for b in a.groups))
        if got != tuple(sorted(reads[str(a.action_id)])):
            raise StatisticalCertificateError(
                "A12 action %s 读位与 ReadAction 组不一致" % (a.action_id,))
    flat = [b for v in reads.values() for b in v]
    if len(flat) != len(set(flat)):
        raise StatisticalCertificateError("A12 基线读位重复：码不是位向量")
    return {"table": table, "reads": reads, "n_bits": table.n_bits,
            "states": tuple(range(table.n_states)), "meta": meta}


def b13_channel_setup(root: Optional[Path] = None) -> dict:
    """B13：每个井读一个不确定盘的存在位 ⇒ 真 bit-flip。"""
    table, meta = cs.b13_table(root)
    reads = {}
    for name, scn in meta["scenarios"].items():
        reads[str(name)] = tuple(int(bit) for bit, _ in scn["reads"])
    return {"table": table, "reads": reads, "n_bits": table.n_bits,
            "states": tuple(range(table.n_states)), "meta": meta}


# ---------------------------------------------------------------------------
# S1 小模型适配（16 状态 / 2 井）
# ---------------------------------------------------------------------------

def s1_tiny_setup(root: Optional[Path] = None) -> dict:
    """S1 tiny model → FiniteTable（DISK_BITS 结构 + 三要素派生）+ 观测核。"""
    from .connectivity_qoi import disk_point_distance
    from .intersect_disk import candidate_pairs, disk_pair_intersect
    from .observation_kernel import GridDetectionKernel

    base = Path(root) if root else REPO_ROOT
    s1 = base / "results" / "s1_sequential_validity"
    model = json.loads((s1 / "tiny_model" / "model.json").read_text(encoding="utf-8"))
    spec = json.loads((base / "results/s0s1_minimal_vertical_chain/s0/observation_spec.json")
                      .read_text(encoding="utf-8"))
    kernel = GridDetectionKernel(model, spec)
    discs = model["discs"]
    C = np.asarray([d["center"] for d in discs], dtype=float)
    N = np.asarray([d["normal"] for d in discs], dtype=float)
    R = np.asarray([d["radius"] for d in discs], dtype=float)
    pairs = candidate_pairs(C, R)
    hit = disk_pair_intersect(C, N, R, pairs[:, 0], pairs[:, 1],
                              scale=float(np.median(R)))
    contact = tuple((int(i), int(j)) for (i, j) in pairs[hit].tolist())
    q = model["query"]
    A = np.asarray(q["A"], dtype=float)
    B = np.asarray(q["B"], dtype=float)
    ra = float(q["r_attach"])
    aatt = [int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, A)[0] <= ra)]
    batt = [int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, B)[0] <= ra)]
    labels = {int(k): int(v) for k, v in model["state_labels"].items()}
    n_bits = len(discs)

    def deriv(qid: str, s: int) -> ident.StateDerivation:
        present = {i for i in range(n_bits) if (int(s) >> i) & 1}
        es = frozenset((i, j) for (i, j) in contact if i in present and j in present)
        a_side = frozenset(x for x in present if x in aatt)
        b_side = frozenset(x for x in present if x in batt)
        return ident.StateDerivation(state_id=int(s), nodes=frozenset(present), edges=es,
                                     a_side=a_side, b_side=b_side,
                                     attach_a=bool(a_side), attach_b=bool(b_side),
                                     label=labels[int(s)])

    struct = cs.TableStructure(kind="DISK_BITS", n_bits=n_bits, certain=(),
                               uncertain=tuple(str(d.get("candidate_index")) for d in discs),
                               contact_edges=contact, attach_a_disks=tuple(aatt),
                               attach_b_disks=tuple(batt))
    table = fs.FiniteTable(
        table_id=str(model["model_id"]), n_bits=n_bits, labels={"S1T": np.asarray(
            [labels[s] for s in range(1 << n_bits)], dtype=np.int64)},
        baseline=(), actions=(),
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N1",
              "model": "S1 声明小模型", "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "S1T_DOMAIN"}})
    table.derivation = deriv
    # 忠实性：逐状态派生标签必须等于模型登记标签
    mism = [s for s in range(1 << n_bits)
            if int(ident.state_connected(deriv("ALL", s))) != labels[s]]
    if mism:
        raise StatisticalCertificateError(
            "tiny model 派生与登记标签不符（float 附着边界）: %s" % (mism,))
    by_id = {a["action_id"]: a for a in model["action_pool"]}
    seq = [by_id[i] for i in model["action_sequence"]]
    return {"table": table, "kernel": kernel, "model": model, "spec": spec,
            "labels": labels, "states": tuple(range(1 << n_bits)), "seq": seq,
            "n_bits": n_bits, "contact_edges": contact,
            "attach_a_disks": tuple(aatt), "attach_b_disks": tuple(batt)}


def support_records(setup: Mapping, tier: str) -> Dict[str, List[dict]]:
    """该噪声档下的全盘在位状态支撑（任何子状态支撑的超集）：轨迹枚举的唯一来源。"""
    kernel = setup["kernel"]
    n_bits = int(setup["n_bits"])
    out: Dict[str, List[dict]] = {}
    for a in setup["seq"]:
        action = kernel._parse_action(a)
        pmf = kernel.count_pmf(bits_of((1 << n_bits) - 1, n_bits), a, tier=tier)
        recs = []
        for vec, p in pmf.items():
            if p == 0:
                continue
            counts = {i: int(c) for i, c in enumerate(vec) if c}
            recs.append(kernel.make_public_record(action, counts))
        recs.sort(key=lambda r: record_key(r))
        out[str(a["action_id"])] = recs
    return out


__all__ = [
    "SCHEMA_STAT_CERT", "SCHEMA_STAT_VERIFY", "SCHEMA_STAT_UNRESOLVED",
    "VERIFIER_VERSION", "SELECTION_PROVENANCE", "BOUND_RULE", "STAT_JOINT",
    "STAT_STEPWISE", "STATUS_CERTIFIED", "STATUS_MIXED", "STATUS_EMPTY",
    "STATUS_CONFLICT", "STATUS_UNRESOLVED", "StatisticalCertificateError",
    "StatisticalRegion", "StatisticalCertificate", "StatisticalUnresolved",
    "PublicEvidence", "BitFlipChannelKernel", "attach_steps", "exact_ledger",
    "statistical_exclude", "certify_with_statistics", "verify_statistical_certificate",
    "history_sha12", "run_eprocess", "a12_channel_setup", "b13_channel_setup",
    "s1_tiny_setup",
]
