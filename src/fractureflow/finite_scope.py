# -*- coding: utf-8 -*-
"""有限表核心层：分诊 / B*_batch / V(H) / 代理选点差距 / 范围汇总 / 五状态证书。

世界族（``world_family``）、A12、B13 共用这一层：只要给出「状态 → 标签」与「动作读组 + 整数代价」，
就按同一套口径出读数。本件是 2026-10-18 架构师晨轮按路线指导书 v1.1 §5 搭建的**预定设计**；
实验中可改，改动按 src 演进纪律执行（改前镜像按内容 sha 归档 + 前后 sha 双读写进该单 verdict）。

纪律：
- 方法面 truth-free：只吃有限表，不接「被实现的那个真状态」；标签是模型目标函数，对每个候选状态都可算。
- N0：动作在状态 s 上的结局 = 各读组内在场位的计数（读组全为单位时即 ``s & 读掩码``）。
  负观测只在零漏检声明下有效，调用方须让 N0 声明随读数同行。
- 完整枚举是唯一参照；超预算一律记 ``COMPUTATION_UNFINISHED`` 并写明原因，不删、不猜。
- 本层零写盘；输出只含整数、布尔、字符串与 ``[分子, 分母]`` 整数对，不产浮点读数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fractureflow import identification as ident

SCHEMA_TRIAGE = "finite-scope-triage/v1"

SINGLE = "SINGLE"
OBSERVATION_LIMITED = "OBSERVATION_LIMITED"
STRUCTURAL = "STRUCTURAL"
TRIAGE_KINDS = (SINGLE, OBSERVATION_LIMITED, STRUCTURAL)

COST_EXACT = "EXACT"
COMPUTATION_UNFINISHED = "COMPUTATION_UNFINISHED"

DEFAULT_BUDGET = {
    "vh_node_budget": 200_000,
    "batch_rep_cap": 2048,
    "proxy_rep_cap": 256,
    "region_cap": 4096,
}

REPO_ROOT = Path(__file__).resolve().parents[2]


class FiniteScopeError(ValueError):
    pass


class _BudgetExceeded(Exception):
    pass


def _popcount_int(x: int) -> int:
    return bin(int(x)).count("1")


def popcount(x) -> np.ndarray:
    return np.bitwise_count(np.asarray(x, dtype=np.uint64)).astype(np.int64)


# ---------------------------------------------------------------------------
# 有限表
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadAction:
    """一个观测动作：互不重叠的读组（位掩码）+ 正整数代价。组内只释放在场计数。"""

    action_id: str
    groups: Tuple[int, ...]
    cost_int: int

    def __post_init__(self):
        if not self.action_id:
            raise FiniteScopeError("action_id 不得为空")
        if not isinstance(self.cost_int, int) or isinstance(self.cost_int, bool) or self.cost_int <= 0:
            raise FiniteScopeError(f"{self.action_id}: cost_int 必须为正整数, 收到 {self.cost_int!r}")
        masks = tuple(int(g) for g in self.groups)
        acc = 0
        for g in masks:
            if g <= 0:
                raise FiniteScopeError(f"{self.action_id}: 读组掩码必须为正")
            if acc & g:
                raise FiniteScopeError(f"{self.action_id}: 读组互相重叠")
            acc |= g
        object.__setattr__(self, "groups", masks)

    @property
    def read_mask(self) -> int:
        m = 0
        for g in self.groups:
            m |= g
        return m

    @property
    def singleton_groups(self) -> bool:
        return all(_popcount_int(g) == 1 for g in self.groups)

    @property
    def radix(self) -> int:
        r = 1
        for g in self.groups:
            r *= _popcount_int(g) + 1
        return r

    def codes(self, states: np.ndarray) -> np.ndarray:
        s = np.asarray(states, dtype=np.uint64)
        code = np.zeros(s.shape[0], dtype=np.int64)
        for g in self.groups:
            code = code * (_popcount_int(g) + 1) + popcount(s & np.uint64(g))
        return code

    def to_dict(self) -> dict:
        return {"action_id": self.action_id, "groups": [int(g) for g in self.groups],
                "cost_int": int(self.cost_int), "read_mask": int(self.read_mask),
                "singleton_groups": bool(self.singleton_groups)}


def combine_codes(parts: Sequence[Tuple[np.ndarray, int]]) -> np.ndarray:
    """混合进制拼接 (codes, radix) 序列；进制积超过 2^62 即拒（防溢出静默）。"""
    if not parts:
        raise FiniteScopeError("combine_codes: 空输入")
    total = 1
    out = np.zeros(parts[0][0].shape[0], dtype=np.int64)
    for codes, radix in parts:
        total *= int(radix)
        if total >= (1 << 62):
            raise FiniteScopeError("观测码进制积越界 (≥2^62)：动作读组过多")
        out = out * int(radix) + codes
    return out


@dataclass
class FiniteTable:
    """有限表：n_bits 个二值对象的 2^n_bits 个状态 + 逐查询标签 + 基线读与候选动作。"""

    table_id: str
    n_bits: int
    labels: Mapping[str, np.ndarray]
    baseline: Tuple[ReadAction, ...]
    actions: Tuple[ReadAction, ...]
    attach: Mapping[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    derivation: Optional[Callable[[str, int], "ident.StateDerivation"]] = None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not (1 <= int(self.n_bits) <= 30):
            raise FiniteScopeError(f"n_bits={self.n_bits} 越界 (1..30)")
        n = self.n_states
        if not self.labels:
            raise FiniteScopeError("labels 为空：拒建表")
        for qid, lab in self.labels.items():
            arr = np.asarray(lab)
            if arr.shape != (n,):
                raise FiniteScopeError(f"{qid}: labels 长度 {arr.shape} != 2^n={n}")
            if not np.all((arr == 0) | (arr == 1)):
                raise FiniteScopeError(f"{qid}: labels 只许 0/1")
        ids = [a.action_id for a in self.actions]
        if len(set(ids)) != len(ids):
            raise FiniteScopeError("候选动作 id 重复")
        full = (1 << int(self.n_bits)) - 1
        for a in tuple(self.baseline) + tuple(self.actions):
            if a.read_mask & ~full:
                raise FiniteScopeError(f"{a.action_id}: 读掩码越出 n_bits")
        for qid, (aa, bb) in self.attach.items():
            if np.asarray(aa).shape != (n,) or np.asarray(bb).shape != (n,):
                raise FiniteScopeError(f"{qid}: attach 数组长度不符")

    @property
    def n_states(self) -> int:
        return 1 << int(self.n_bits)

    def states(self) -> np.ndarray:
        return np.arange(self.n_states, dtype=np.int64)

    def blind_mask(self) -> int:
        seen = 0
        for a in tuple(self.baseline) + tuple(self.actions):
            seen |= a.read_mask
        return ((1 << int(self.n_bits)) - 1) & ~seen

    def baseline_codes(self) -> np.ndarray:
        s = self.states()
        if not self.baseline:
            return np.zeros(self.n_states, dtype=np.int64)
        return combine_codes([(a.codes(s), a.radix) for a in self.baseline])

    def action_codes(self) -> Dict[str, np.ndarray]:
        s = self.states()
        return {a.action_id: a.codes(s) for a in self.actions}

    def full_pool_codes(self, action_codes: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.actions:
            return np.zeros(self.n_states, dtype=np.int64)
        return combine_codes([(action_codes[a.action_id], a.radix) for a in self.actions])

    def describe(self) -> dict:
        return {"table_id": self.table_id, "n_bits": int(self.n_bits), "n_states": self.n_states,
                "queries": sorted(self.labels), "baseline": [a.to_dict() for a in self.baseline],
                "actions": [a.to_dict() for a in self.actions], "blind_mask": int(self.blind_mask()),
                "tags": dict(self.tags)}


def partition_classes(table: FiniteTable) -> List[Tuple[int, np.ndarray]]:
    """按基线观测码分类：返回 [(基线码, 升序成员状态)]，按基线码升序。"""
    codes = table.baseline_codes()
    order = np.lexsort((table.states(), codes))
    sc = codes[order]
    cuts = np.flatnonzero(np.diff(sc)) + 1
    out = []
    for seg in np.split(order, cuts):
        out.append((int(codes[seg[0]]), np.sort(seg).astype(np.int64)))
    return out


# ---------------------------------------------------------------------------
# 成本：B*_batch（复用 identification）与 V(H)（带节点预算的本层实现）
# ---------------------------------------------------------------------------

def _cost_actions(table: FiniteTable) -> List["ident.CostAction"]:
    return [ident.CostAction(a.action_id, int(a.cost_int)) for a in table.actions]


def worst_case_tree_cost(rep_codes: Mapping[str, np.ndarray], rep_labels: np.ndarray,
                         costs: Mapping[str, int], *, node_budget: int) -> dict:
    """V(H)：最优最坏成本决策树（任务源 §11.2）。

    语义与 ``identification.optimal_worst_case_cost`` 相同（只取 ≥2 个非空分支的动作；
    min over 动作 of 代价 + max over 分支；同价取动作 id 字典序最小），另加节点预算以便把
    超时如实记成计算未完。输入是已按全池剖面折叠的代表元（每个代表元标签唯一）。
    """
    aids = sorted(rep_codes)
    labs = np.asarray(rep_labels, dtype=np.int64)
    memo: Dict[frozenset, Optional[int]] = {}
    counter = [0]

    def solve(gids: frozenset) -> Optional[int]:
        sub = labs[list(gids)]
        if sub.min() == sub.max():
            return 0
        if gids in memo:
            return memo[gids]
        counter[0] += 1
        if counter[0] > node_budget:
            raise _BudgetExceeded()
        idx = np.fromiter(gids, dtype=np.int64)
        best: Optional[int] = None
        for aid in aids:
            col = rep_codes[aid][idx]
            vals = np.unique(col)
            if vals.shape[0] < 2:
                continue
            c = int(costs[aid])
            if best is not None and c >= best:
                continue
            worst = 0
            ok = True
            for v in vals:
                sub_cost = solve(frozenset(idx[col == v].tolist()))
                if sub_cost is None:
                    ok = False
                    break
                worst = max(worst, sub_cost)
                if best is not None and c + worst >= best:
                    break
            if not ok:
                continue
            if best is None or c + worst < best:
                best = c + worst
        memo[gids] = best
        return best

    try:
        cost = solve(frozenset(range(labs.shape[0])))
    except _BudgetExceeded:
        return {"status": COMPUTATION_UNFINISHED, "cost_int": None, "nodes": counter[0],
                "cause": "VH_NODE_BUDGET", "node_budget": int(node_budget)}
    if cost is None:
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "nodes": counter[0]}
    return {"status": "OPTIMAL", "cost_int": int(cost), "nodes": counter[0],
            "node_budget": int(node_budget)}


def proxy_policy_cost(rep_codes: Mapping[str, np.ndarray], rep_labels: np.ndarray,
                      rep_weights: Sequence[int], cost_actions: Sequence["ident.CostAction"],
                      *, node_budget: int) -> dict:
    """代理选点策略（G31 的 EC2 型跨答案对质量代理，``active_query.ec2_proxy_ranking``）的最坏成本。

    每个结点取代理排序第一且确实分离跨答案对的动作，按结局分支递归到纯标签。
    这是 Phase A 语义的差距测量：只与 V(H) 比，不作任何策略优劣主张。
    """
    from fractureflow import active_query as aq

    labs = np.asarray(rep_labels, dtype=np.int64)
    w = [int(x) for x in rep_weights]
    counter = [0]

    def run(idx: np.ndarray) -> Optional[int]:
        sub = labs[idx]
        if sub.min() == sub.max():
            return 0
        counter[0] += 1
        if counter[0] > node_budget:
            raise _BudgetExceeded()
        tot = sum(w[i] for i in idx.tolist())
        belief = aq.Belief.build([int(i) for i in idx.tolist()],
                                 [Fraction(w[i], tot) for i in idx.tolist()],
                                 lambda g: int(labs[g]), model_note="finite_scope proxy node")
        rank = aq.ec2_proxy_ranking(belief, cost_actions,
                                    lambda g, aid: int(rep_codes[aid][g]))
        chosen = next((r for r in rank["rows"] if r.separated_pair_count > 0), None)
        if chosen is None:
            return None
        col = rep_codes[chosen.action_id][idx]
        worst = 0
        for v in np.unique(col):
            c = run(idx[col == v])
            if c is None:
                return None
            worst = max(worst, c)
        return int(chosen.cost_int) + worst

    try:
        cost = run(np.arange(labs.shape[0], dtype=np.int64))
    except _BudgetExceeded:
        return {"status": COMPUTATION_UNFINISHED, "cost_int": None, "nodes": counter[0],
                "cause": "PROXY_NODE_BUDGET"}
    if cost is None:
        return {"status": "UNRESOLVED_UNSEPARABLE_PAIRS", "cost_int": None, "nodes": counter[0]}
    return {"status": "OPTIMAL_POLICY_COST", "cost_int": int(cost), "nodes": counter[0],
            "claim_scope": "DEVELOPMENT_ONLY_GAP_MEASUREMENT"}


# ---------------------------------------------------------------------------
# 分诊
# ---------------------------------------------------------------------------

def _attach_summary(arr: Optional[np.ndarray], members: np.ndarray) -> Optional[str]:
    if arr is None:
        return None
    return ident.attachment_summary([bool(x) for x in np.asarray(arr)[members].tolist()])


def triage_class(table: FiniteTable, qid: str, members: np.ndarray, *,
                 action_codes: Mapping[str, np.ndarray], full_codes: np.ndarray,
                 budget: Mapping[str, int], with_costs: bool = True,
                 with_proxy: bool = True) -> dict:
    lab_all = np.asarray(table.labels[qid], dtype=np.int64)
    lab = lab_all[members]
    n1 = int(lab.sum())
    n0 = int(members.shape[0]) - n1
    target = sorted({int(x) for x in np.unique(lab)})
    aa, bb = table.attach.get(qid, (None, None))
    row = {
        "members_count": int(members.shape[0]), "min_state": int(members[0]),
        "n_c0": n0, "n_c1": n1, "target_set": target,
        "five_state": ident.classify_status(target),
        "attach_a": _attach_summary(aa, members), "attach_b": _attach_summary(bb, members),
    }
    if len(target) == 1:
        row.update({"triage": SINGLE, "cost_status": COST_EXACT, "batch_cost": 0,
                    "tree_cost": 0, "proxy_cost": 0})
        return row
    fc = full_codes[members]
    uniq, inv = np.unique(fc, return_inverse=True)
    g_min = np.full(uniq.shape[0], 2, dtype=np.int64)
    g_max = np.full(uniq.shape[0], -1, dtype=np.int64)
    np.minimum.at(g_min, inv, lab)
    np.maximum.at(g_max, inv, lab)
    mixed = np.flatnonzero(g_min != g_max)
    if mixed.shape[0]:
        sizes1 = np.bincount(inv, weights=lab, minlength=uniq.shape[0]).astype(np.int64)
        sizes = np.bincount(inv, minlength=uniq.shape[0]).astype(np.int64)
        unsep = int(np.sum((sizes1 * (sizes - sizes1))[mixed]))
        g0 = int(mixed[0])
        sel = members[inv == g0]
        sl = lab_all[sel]
        s0 = int(sel[sl == 0].min())
        s1 = int(sel[sl == 1].min())
        diff = s0 ^ s1
        nonsingle = 0
        for a in tuple(table.baseline) + tuple(table.actions):
            for g in a.groups:
                if _popcount_int(g) > 1:
                    nonsingle |= g
        lemma_ok = (diff & ~(table.blind_mask() | nonsingle)) == 0
        row.update({"triage": STRUCTURAL, "cost_status": COST_EXACT, "batch_cost": None,
                    "tree_cost": None, "proxy_cost": None,
                    "mixed_full_profile_groups": int(mixed.shape[0]),
                    "unseparable_cross_pairs": unsep,
                    "witness_pair": {"state_label0": s0, "state_label1": s1, "diff_mask": int(diff),
                                     "diff_within_blind_or_merged_groups": bool(lemma_ok)}})
        return row
    reps = np.array([int(members[inv == g].min()) for g in range(uniq.shape[0])], dtype=np.int64)
    rep_weights = np.bincount(inv, minlength=uniq.shape[0]).astype(np.int64)
    row.update({"triage": OBSERVATION_LIMITED, "n_full_profiles": int(reps.shape[0])})
    if not with_costs:
        row.update({"cost_status": "NOT_EVALUATED", "batch_cost": None, "tree_cost": None,
                    "proxy_cost": None})
        return row
    causes = []
    batch_cost = None
    batch_chosen = None
    if reps.shape[0] <= int(budget["batch_rep_cap"]):
        res = ident.minimum_batch_cost([int(x) for x in reps], lambda s: int(lab_all[s]),
                                       _cost_actions(table),
                                       lambda s, aid: int(action_codes[aid][s]))
        if res["status"] == "OPTIMAL":
            batch_cost, batch_chosen = int(res["cost_int"]), list(res["chosen"])
        else:
            raise FiniteScopeError(f"{qid}: 可分类的 B* 返回 {res['status']} —— 仪器缺陷")
    else:
        causes.append("BATCH_REP_CAP")
    rep_codes = {aid: action_codes[aid][reps] for aid in action_codes}
    rep_labels = lab_all[reps]
    costs = {a.action_id: int(a.cost_int) for a in table.actions}
    vh = worst_case_tree_cost(rep_codes, rep_labels, costs, node_budget=int(budget["vh_node_budget"]))
    tree_cost = vh["cost_int"] if vh["status"] == "OPTIMAL" else None
    if vh["status"] == COMPUTATION_UNFINISHED:
        causes.append(vh["cause"])
    elif vh["status"] != "OPTIMAL":
        raise FiniteScopeError(f"{qid}: 可分类的 V(H) 返回 {vh['status']} —— 仪器缺陷")
    proxy_cost = None
    proxy_status = "NOT_EVALUATED"
    if with_proxy and reps.shape[0] <= int(budget["proxy_rep_cap"]):
        px = proxy_policy_cost(rep_codes, rep_labels, rep_weights, _cost_actions(table),
                               node_budget=int(budget["vh_node_budget"]))
        proxy_status = px["status"]
        proxy_cost = px["cost_int"]
    elif with_proxy:
        proxy_status = "PROXY_REP_CAP"
    row.update({"cost_status": COMPUTATION_UNFINISHED if causes else COST_EXACT,
                "cost_unfinished_causes": causes, "batch_cost": batch_cost,
                "batch_chosen": batch_chosen, "tree_cost": tree_cost,
                "tree_nodes": int(vh["nodes"]), "proxy_cost": proxy_cost,
                "proxy_status": proxy_status})
    return row


def triage_table(table: FiniteTable, qid: str, *, budget: Optional[Mapping[str, int]] = None,
                 with_costs: bool = True, with_proxy: bool = True) -> List[dict]:
    if qid not in table.labels:
        raise FiniteScopeError(f"未知查询 {qid!r}")
    bud = dict(DEFAULT_BUDGET)
    bud.update(budget or {})
    ac = table.action_codes()
    fc = table.full_pool_codes(ac)
    rows = []
    for code, members in partition_classes(table):
        r = triage_class(table, qid, members, action_codes=ac, full_codes=fc, budget=bud,
                         with_costs=with_costs, with_proxy=with_proxy)
        r["baseline_code"] = code
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# 范围汇总（三口径分栏；世界为单位的汇总在驱动层做）
# ---------------------------------------------------------------------------

def _bernoulli_class_mass(members: np.ndarray, n_bits: int, p: Fraction) -> Fraction:
    hist = np.bincount(popcount(members), minlength=n_bits + 1)
    q = 1 - p
    return sum((Fraction(int(c)) * (p ** k) * (q ** (n_bits - k)) for k, c in enumerate(hist.tolist())),
               Fraction(0))


def scope_summary(table: FiniteTable, rows: Sequence[Mapping], *,
                  p_alt: Fraction = Fraction(3, 10)) -> dict:
    """单表的分诊汇总：类计数 / 状态等权计数（主口径）/ Bernoulli(p_alt) 精确质量，三栏分列。"""
    if not rows:
        raise FiniteScopeError("空分诊表：拒发汇总")
    classes = partition_classes(table)
    by_code = {c: m for c, m in classes}
    out = {k: {"class_count": 0, "state_count": 0, "bernoulli_mass": Fraction(0)}
           for k in TRIAGE_KINDS}
    unfinished = {"class_count": 0, "state_count": 0}
    for r in rows:
        k = r["triage"]
        members = by_code[r["baseline_code"]]
        out[k]["class_count"] += 1
        out[k]["state_count"] += int(r["members_count"])
        out[k]["bernoulli_mass"] += _bernoulli_class_mass(members, table.n_bits, p_alt)
        if r.get("cost_status") == COMPUTATION_UNFINISHED:
            unfinished["class_count"] += 1
            unfinished["state_count"] += int(r["members_count"])
    tot_states = sum(v["state_count"] for v in out.values())
    tot_mass = sum((v["bernoulli_mass"] for v in out.values()), Fraction(0))
    if tot_states != table.n_states or tot_mass != 1 or sum(v["class_count"] for v in out.values()) != len(rows):
        raise FiniteScopeError("分诊汇总不闭合：分母守恒失败")
    return {
        "schema": SCHEMA_TRIAGE, "n_classes": len(rows), "n_states": table.n_states,
        "p_alt": [p_alt.numerator, p_alt.denominator],
        "by_triage": {k: {"class_count": v["class_count"], "state_count": v["state_count"],
                          "bernoulli_mass": [v["bernoulli_mass"].numerator,
                                             v["bernoulli_mass"].denominator]}
                      for k, v in out.items()},
        "cost_unfinished": unfinished,
        "denominators": {"class_count": len(rows), "state_count": table.n_states,
                         "bernoulli_mass": [1, 1]},
    }


# ---------------------------------------------------------------------------
# 五状态证书（§7.1–§7.3：不决即按位分裂区域）
# ---------------------------------------------------------------------------

def _split_regions(members: np.ndarray, derive: Callable[[int], "ident.StateDerivation"],
                   expected: str, cap: int) -> Optional[List[np.ndarray]]:
    stack = [members]
    out: List[np.ndarray] = []
    while stack:
        reg = stack.pop()
        lab = ident.region_label(ident.envelope_of([derive(int(s)) for s in reg]))
        if lab == expected:
            out.append(reg)
        elif lab == ident.RegionLabel.UNRESOLVED.value:
            if reg.shape[0] == 1:
                raise FiniteScopeError("单状态区域外包不决：仪器缺陷")
            vary = int(np.bitwise_or.reduce(reg)) & ~int(np.bitwise_and.reduce(reg))
            j = (vary & -vary).bit_length() - 1
            bit = (reg >> j) & 1
            stack.append(reg[bit == 1])
            stack.append(reg[bit == 0])
        else:
            raise FiniteScopeError(f"单值类内出现相反方向区域 {lab}：仪器缺陷")
        if len(out) + len(stack) > cap:
            return None
    return sorted(out, key=lambda r: int(r[0]))


def class_certificate(table: FiniteTable, qid: str, members: np.ndarray, row: Mapping, *,
                      model_id: str, region_cap: int = DEFAULT_BUDGET["region_cap"],
                      include_doc: bool = False) -> dict:
    """为一个基线类出 G29 同款五状态证书，并用 ``verify_regions`` 独立重算区域外包。"""
    if table.derivation is None:
        raise FiniteScopeError("该表未提供三要素派生：不能出证书")
    cache: Dict[int, "ident.StateDerivation"] = {}

    def derive(s: int) -> "ident.StateDerivation":
        if s not in cache:
            cache[s] = table.derivation(qid, int(s))
        return cache[s]

    lab_all = np.asarray(table.labels[qid], dtype=np.int64)
    target = list(row["target_set"])
    status = row["five_state"]
    witnesses = []
    for want in target:
        s = int(members[lab_all[members] == want].min())
        witnesses.append({"state_id": s, "label": int(want)})
    regions: List[dict] = []
    proof_route = ident.ProofRoute.WITNESS_PAIR.value
    search_complete, unresolved_count, cause = True, None, None
    verify = None
    if len(target) == 1:
        expected = (ident.RegionLabel.ALL_CONNECTED.value if target == [1]
                    else ident.RegionLabel.ALL_DISCONNECTED.value)
        regs = _split_regions(members, derive, expected, int(region_cap))
        if regs is None:
            status = ident.IdentificationStatus.SEARCH_UNRESOLVED.value
            target = [0, 1]
            search_complete, cause = False, ident.UnresolvedCause.SPLIT_BUDGET_EXHAUSTED.value
            unresolved_count = 1
            proof_route = ident.ProofRoute.NONE.value
        else:
            proof_route = (ident.ProofRoute.MUST_ENVELOPE.value if target == [1]
                           else ident.ProofRoute.MAY_ENVELOPE_EXCLUSION.value)
            regions = [{"region_id": f"R{i}", "state_ids": [int(x) for x in r.tolist()],
                        "claimed_label": expected, "recomputed_label": expected}
                       for i, r in enumerate(regs)]
            verify = ident.verify_regions(regions=regions, derivation_of_state=derive,
                                          allowed_state_ids=[int(x) for x in members.tolist()])
            if not verify["ok"]:
                raise FiniteScopeError(f"区域验证失败: {verify['reasons'][:3]}")
    doc = ident.build_cert3_certificate(
        certificate_id=f"{table.table_id}:{qid}:{row['baseline_code']}", model_id=model_id,
        query_id=qid, observation_id=f"{table.table_id}:baseline:{row['baseline_code']}",
        status=status, target_set=target, feasible_state_ids=[int(x) for x in members.tolist()],
        regions=regions, witnesses=witnesses, derivation_of_state=derive,
        geometry_semantics="DECLARED_COMPUTATIONAL", unresolved_cause=cause,
        proof_route=proof_route, search_complete=search_complete,
        unresolved_region_count=unresolved_count,
        reason="finite_scope: 完整枚举 + 区域外包（不决即按位分裂）")
    out = {"status": doc["status"], "target_set": doc["target_set"], "proof_route": proof_route,
           "n_regions": len(regions), "witnesses": witnesses,
           "verify_ok": None if verify is None else bool(verify["ok"]),
           "whole_space_label": None if verify is None else verify["whole_space_label"],
           "search_complete": search_complete, "unresolved_cause": cause}
    if include_doc:
        out["doc"] = doc
    return out


# ---------------------------------------------------------------------------
# A12 回归适配（读 G26 冻结制品；只读）
# ---------------------------------------------------------------------------

FR = REPO_ROOT / "results" / "connectivity_certification" / "finite_reference"


def _load(p: Path) -> dict:
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def a12_table(root: Optional[Path] = None) -> Tuple[FiniteTable, dict]:
    """A12-GRAPH-SENSOR-V1 → 有限表（基线 = action_spec.baseline，候选 = action_spec.pool）。"""
    base = Path(root) if root else REPO_ROOT
    cc = base / "results" / "connectivity_certification"
    ms = _load(cc / "model_spec.json")["a_layer"]
    acts = _load(cc / "action_spec.json")["a_layer"]
    sl = _load(cc / "finite_reference" / "a_layer" / "state_labels.json")
    edge_ids = [e[0] for e in ms["edges"]]
    pos = {e: i for i, e in enumerate(edge_ids)}
    by_id = {a["id"]: a for a in acts["actions"]}

    def ra(aid: str) -> ReadAction:
        a = by_id[aid]
        return ReadAction(aid, tuple(1 << pos[e] for e in a["reads"]), int(a["cost"]))

    table = FiniteTable(
        table_id=ms["model_id"], n_bits=len(edge_ids),
        labels={"A12": np.asarray(sl["c_labels"], dtype=np.uint8)},
        baseline=tuple(ra(a) for a in acts["baseline"]), actions=tuple(ra(a) for a in acts["pool"]),
        attach={"A12": (np.asarray(sl["attach_a"], dtype=bool), np.asarray(sl["attach_b"], dtype=bool))}
        if "attach_a" in sl and isinstance(sl["attach_a"], list) else {},
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N0",
              "model": "A12 声明有限模型", "geometry_semantics": "DECLARED_COMPUTATIONAL"})
    meta = {"class_key_bits": list(sl["class_key_bits"]), "edge_pos": pos}
    return table, meta


def a12_regression(root: Optional[Path] = None) -> dict:
    """A12 回归锚：分诊计数与逐类 (B*, V(H)) 必须与 G26 costs_per_class 逐类相等。"""
    base = Path(root) if root else REPO_ROOT
    table, meta = a12_table(base)
    rows = triage_table(table, "A12", with_proxy=False)
    g26 = {r["class_key"]: r for r in _load(base / "results" / "connectivity_certification" /
                                            "finite_reference" / "a_layer" / "costs.json")["costs_per_class"]}
    status_map = {SINGLE: "NO_CROSS_PAIRS", STRUCTURAL: "UNRESOLVED_UNSEPARABLE_PAIRS",
                  OBSERVATION_LIMITED: "OPTIMAL"}
    mism = []
    for r in rows:
        s = r["min_state"]
        key = sum(((s >> meta["edge_pos"][e]) & 1) << i for i, e in enumerate(meta["class_key_bits"]))
        g = g26[key]
        mine = (status_map[r["triage"]], r["batch_cost"], r["tree_cost"])
        ref = (g["status"], g["batch_cost"] if g["status"] != "UNRESOLVED_UNSEPARABLE_PAIRS" else None,
               g["tree_cost"] if g["status"] != "UNRESOLVED_UNSEPARABLE_PAIRS" else None)
        if mine != ref:
            mism.append({"class_key": key, "mine": list(mine), "g26": list(ref)})
    counts = {k: sum(1 for r in rows if r["triage"] == k) for k in TRIAGE_KINDS}
    return {"table_id": table.table_id, "counts": counts, "n_classes": len(rows),
            "mismatches": mism, "equal": not mism and counts == {SINGLE: 53, OBSERVATION_LIMITED: 27,
                                                                 STRUCTURAL: 48}}


__all__ = [
    "SCHEMA_TRIAGE", "SINGLE", "OBSERVATION_LIMITED", "STRUCTURAL", "TRIAGE_KINDS",
    "COST_EXACT", "COMPUTATION_UNFINISHED", "DEFAULT_BUDGET", "FiniteScopeError",
    "ReadAction", "FiniteTable", "combine_codes", "partition_classes", "popcount",
    "worst_case_tree_cost", "proxy_policy_cost", "triage_class", "triage_table",
    "scope_summary", "class_certificate", "a12_table", "a12_regression",
]
