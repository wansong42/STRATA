# -*- coding: utf-8 -*-
"""CERT-4 主动补测 Phase A 开发面 (任务源 §10.1-10.7)。

水印: 研究线开发仪器 · 合成域 · 禁入对外承诺文本 · Phase B (有效界收益实验) 冻结中。

面纪律:
- 本文件零写 identification.py (G29 是唯一写者), 只 import 只读消费;
- 规划读数 (期望收益 U_M / 代理分 / 排序) 与有效性读数 (target_set / 证书) 分轨:
  本模块一切输出 claim_scope = DEVELOPMENT_ONLY, 不得充当目标界或有效性主张;
- 真值隔离: 消费路径只读已释放记录与模型表, 隐藏真值一律经 ident.make_poisoned_truth()。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fractureflow.identification import (
    ContractError,
    CostAction,
    TruthAccessViolation,
    assert_released_payload,
    graph_connected,
    make_poisoned_truth,
    optimal_worst_case_cost,
    project_target_set,
)

# ---------------------------------------------------------------------------
# 契约错误 (本面自有类型, 不外抛裸 ValueError)
# ---------------------------------------------------------------------------


class ActiveQueryContractError(ValueError):
    """本面输入契约违规。"""


class ExpectationNotDeclared(ActiveQueryContractError):
    """缺规划分布却要求数值期望 (§10.2『不能假装期望已经定义』)。"""


class ProxyNotBoundViolation(ActiveQueryContractError):
    """把便宜代理的数值当目标界/有效性主张使用 (§10.3)。"""


class NotAdaptiveViolation(ActiveQueryContractError):
    """预先选井形自称自适应闭环 (§10.5 / G22 基线)。"""


class EvidenceTokenError(ActiveQueryContractError):
    """停止判定的证据令牌类型不足 (§10.7『采样未找到差异不算』)。"""


class FutureRecordViolation(ActiveQueryContractError):
    """消费路径读到尚未由 sensor 释放的记录 (§14.3-12)。"""


DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
FORBIDDEN_PROVENANCE_PREFIXES = ("TEST_RESULT", "FITTED_ON", "TUNED_BY", "GRID_SEARCH_ON")
RULE_BAYES_TERNARY = "BAYES_TERNARY_PLANNING"
RULE_CERTIFICATE_TRACK = "CERTIFICATE_TRACK"
POLICY_ALWAYS_JUDGE = "ALWAYS_JUDGE_POSTERIOR_BYPASS"
POLICY_ALWAYS_ABSTAIN = "ALWAYS_ABSTAIN"
POLICY_MAJORITY = "ALWAYS_MAJORITY"
POLICY_RANDOM_UNIFORM = "RANDOM_UNIFORM"
DEGENERATE_POLICIES = (POLICY_ALWAYS_JUDGE, POLICY_ALWAYS_ABSTAIN, POLICY_MAJORITY,
                       POLICY_RANDOM_UNIFORM)


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise ActiveQueryContractError(f"非有限概率: {x!r}")
        return Fraction(str(x))
    if isinstance(x, str):
        return Fraction(x)
    raise ActiveQueryContractError(f"概率类型非法: {x!r} ({type(x).__name__})")


# ---------------------------------------------------------------------------
# §10.1 三值决策与 λ 损失
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossWeights:
    """λ_FP / λ_FN / λ_? 权重; 依据必须来自任务定义, 禁由测试结果反推。"""

    lambda_fp: float
    lambda_fn: float
    lambda_abstain: float
    provenance: str
    asymmetry_basis: Optional[str] = None

    def __post_init__(self):
        for name in ("lambda_fp", "lambda_fn", "lambda_abstain"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v != v or v < 0:
                raise ActiveQueryContractError(f"损失权重 {name} 非法: {v!r}")
        if self.lambda_fp <= 0 or self.lambda_fn <= 0:
            raise ActiveQueryContractError("错判权重必须 > 0, 否则『乱判』无代价, 判据失去牙齿")
        prov = str(self.provenance or "").strip().upper()
        if not prov:
            raise ActiveQueryContractError("损失权重必须带任务定义依据 (provenance), 拒发")
        for bad in FORBIDDEN_PROVENANCE_PREFIXES:
            if prov.startswith(bad):
                raise ActiveQueryContractError(
                    f"权重依据 {self.provenance!r} 以测试结果为准 (§10.1 禁以调 lambda 让主策略胜出)")

    @classmethod
    def symmetric_declared(cls, lambda_abstain: float, provenance: str) -> "LossWeights":
        return cls(1.0, 1.0, float(lambda_abstain), provenance, asymmetry_basis=None)

    def is_symmetric(self) -> bool:
        return float(self.lambda_fp) == float(self.lambda_fn)

    def to_dict(self) -> dict:
        return {"lambda_fp": self.lambda_fp, "lambda_fn": self.lambda_fn,
                "lambda_abstain": self.lambda_abstain, "provenance": self.provenance,
                "asymmetry_basis": self.asymmetry_basis, "symmetric_fp_fn": self.is_symmetric()}


def norm_decision(d: object) -> str:
    """决策 token 归一: 三值只许 {1, 0, ?} (int 与 str 两种写法等价收下)。"""
    if d in ("?", None):
        return "?"
    if d in (1, "1"):
        return "1"
    if d in (0, "0"):
        return "0"
    raise ActiveQueryContractError(f"决策取值非法: {d!r} (只许 1/0/?)")


def loss_of(d: object, c: object, w: LossWeights) -> float:
    """ell(d,C) = λ_FP·1[d=1,C=0] + λ_FN·1[d=0,C=1] + λ_?·1[d=?]。"""
    dd = norm_decision(d)
    if int(c) not in (0, 1):
        raise ActiveQueryContractError(f"真值取值非法: c={c!r}")
    if dd == "?":
        return float(w.lambda_abstain)
    if dd == "1" and int(c) == 0:
        return float(w.lambda_fp)
    if dd == "0" and int(c) == 1:
        return float(w.lambda_fn)
    return 0.0


def loss_table(w: LossWeights) -> List[dict]:
    out = []
    for d in ("1", "0", "?"):
        for c in (0, 1):
            out.append({"d": d, "c": c, "loss": loss_of(d, c, w)})
    return out


@dataclass(frozen=True)
class Belief:
    """规划用信念状态: 允许状态集 + 每状态概率 + 模型目标函数值 C(s)。

    labels 是模型定义的目标函数 (对每个候选状态都能算), 不是隐藏真值;
    隐藏的是『哪一个状态被实现』——那只在 sensor 释放后才可读到。
    """

    states: Tuple[int, ...]
    probs: Tuple[Fraction, ...]
    labels: Tuple[int, ...]
    model_note: str = ""

    def __post_init__(self):
        if len(self.states) == 0:
            raise ActiveQueryContractError("信念状态成员集为空: 拒发 (空集不得当读数)")
        if not (len(self.states) == len(self.probs) == len(self.labels)):
            raise ActiveQueryContractError("states/probs/labels 长度不一致")
        if len(set(self.states)) != len(self.states):
            raise ActiveQueryContractError("信念状态成员重复")
        if any(int(v) not in (0, 1) for v in self.labels):
            raise ActiveQueryContractError("目标值 C(s) 只允许 0/1")
        s = sum(self.probs, Fraction(0))
        if s != 1:
            raise ActiveQueryContractError(f"规划概率未归一 (Σp = {s} != 1): 期望未定义")

    @classmethod
    def build(cls, states: Sequence[int], probs: Sequence[object],
              labels_of: Callable[[int], int], model_note: str = "") -> "Belief":
        ps = tuple(_frac(p) for p in probs)
        return cls(tuple(int(s) for s in states), ps, tuple(int(labels_of(s)) for s in states),
                   model_note)

    @property
    def n(self) -> int:
        return len(self.states)

    def index_of(self, state: int) -> int:
        return self.states.index(int(state))

    def prob(self, state: int) -> Fraction:
        return self.probs[self.index_of(state)]

    def p_connected(self) -> Fraction:
        return sum((p for p, l in zip(self.probs, self.labels) if l == 1), Fraction(0))

    def target_set(self) -> List[int]:
        return project_target_set(list(range(self.n)), self.labels)

    def restrict_by_mask(self, keep: Sequence[bool]) -> "Belief":
        st, pr, lb = [], [], []
        for i, k in enumerate(keep):
            if k:
                st.append(self.states[i])
                pr.append(self.probs[i])
                lb.append(self.labels[i])
        if not st:
            raise EmptyBeliefAfterRestriction("限制后信念集为空: 走 MODEL_CONFLICT 路, 禁在此发单值")
        tot = sum(pr, Fraction(0))
        if tot == 0:
            raise EmptyBeliefAfterRestriction("限制后概率质量为零: 该分支不可达, 不得当已缩窄计收益")
        pr = tuple(p / tot for p in pr)
        return Belief(tuple(st), tuple(pr), tuple(lb), self.model_note)

    def branch_by_outcome(self, profile_of: Callable[[int], object]) -> List[Tuple[object, "Belief"]]:
        """按动作结局划分支 (只枚举非空分支; 空结局不当免费成功)。"""
        groups: Dict[object, List[int]] = {}
        for i, s in enumerate(self.states):
            groups.setdefault(profile_of(s), []).append(i)
        out = []
        for key in sorted(groups, key=repr):
            keep = [i in set(groups[key]) for i in range(self.n)]
            mass = sum((self.probs[i] for i in groups[key]), Fraction(0))
            out.append((key, mass, self.restrict_by_mask(keep)))
        return out


class EmptyBeliefAfterRestriction(ActiveQueryContractError):
    """空可行集合 (MODEL_CONFLICT 路), 不得在此产出单值。"""


def decision_risks(belief: Belief, w: LossWeights) -> Dict[str, float]:
    p = belief.p_connected()
    one_minus = 1 - p
    return {"1": float(w.lambda_fp * one_minus), "0": float(w.lambda_fn * p),
            "?": float(w.lambda_abstain)}


def risk_of_decision(d: object, belief: Belief, w: LossWeights) -> float:
    return sum((float(p) * loss_of(d, c, w) for p, c in zip(belief.probs, belief.labels)), 0.0)


def apply_rule(rule: str, belief: Belief, w: LossWeights) -> object:
    """规划侧决策规则 (只有贝叶斯三值与退化策略); 证书轨由 certified_decision 单独把守。"""
    if rule == RULE_BAYES_TERNARY:
        r = decision_risks(belief, w)
        return min(("1", "0", "?"), key=lambda k: (r[k], {"1": 0, "0": 1, "?": 2}[k]))
    if rule == POLICY_ALWAYS_JUDGE:
        r = decision_risks(belief, w)
        return min("1", "0", key=lambda k: (r[k], 0 if k == "1" else 1))
    if rule == POLICY_ALWAYS_ABSTAIN:
        return "?"
    if rule == POLICY_MAJORITY:
        n1 = sum(1 for l in belief.labels if l == 1)
        n0 = belief.n - n1
        return "1" if n1 > n0 else ("0" if n0 > n1 else "0")
    raise ActiveQueryContractError(f"未知规划规则: {rule!r} (证书轨不走本函数)")


def policy_expected_risk(policy: str, belief: Belief, w: LossWeights) -> float:
    """对 belief 上的真实 C 分布取期望; RANDOM_UNIFORM 用解析式, 不引 RNG。"""
    if policy == POLICY_RANDOM_UNIFORM:
        return 0.5 * risk_of_decision("1", belief, w) + 0.5 * risk_of_decision("0", belief, w)
    return risk_of_decision(apply_rule(policy, belief, w), belief, w)


@dataclass(frozen=True)
class Decision:
    d: object
    basis: str
    certificate_status: str
    claim_scope: str = DEVELOPMENT_ONLY

    def __post_init__(self):
        object.__setattr__(self, "d", norm_decision(self.d))

    def to_dict(self) -> dict:
        return {"d": self.d, "basis": self.basis, "certificate_status": self.certificate_status,
                "claim_scope": self.claim_scope}


def certified_decision(target_set: Sequence[int], w: LossWeights, *,
                        posterior: Optional[float] = None,
                        certificate_status: str = "SUBSTITUTED_FINITE_ENUMERATION",
                        search_complete: bool = True,
                        allow_posterior_bypass: bool = False) -> Decision:
    """证书轨 (§10.1): 只在有证据 (单点目标集) 时输出单值, 否则 d=?。

    高后验不得绕过证书: allow_posterior_bypass 是留给负例的显式门, 默认且唯一安全值为 False。
    """
    if allow_posterior_bypass:
        raise ActiveQueryContractError(
            "风险目标不授权以高后验概率绕过证书 (§10.1): 该参数只许负例触发, 不许生产路径置真")
    ts = sorted(set(int(t) for t in target_set)) if len(target_set) else []
    if not search_complete:
        return Decision("?", "SEARCH_UNRESOLVED_NO_SINGLE_VALUE_RIGHT", "SEARCH_UNRESOLVED")
    if len(ts) == 0:
        return Decision("?", "EMPTY_FEASIBLE_SET_MODEL_CONFLICT", "MODEL_CONFLICT")
    if len(ts) == 1:
        return Decision(str(ts[0]), "CERTIFICATE_SINGLE_VALUE", certificate_status)
    if posterior is not None and float(posterior) > 0.99:
        return Decision("?", "WITNESSED_AMBIGUOUS_POSTERIOR_NOT_AUTHORIZED", "WITNESSED_AMBIGUOUS")
    return Decision("?", "WITNESSED_AMBIGUOUS_ABSTAIN", "WITNESSED_AMBIGUOUS")


# ---------------------------------------------------------------------------
# §10.2 规划期望收益 U_M 与 min-M 稳健单位成本选点
# ---------------------------------------------------------------------------


def rule_risk_under_rule(rule: str, belief: Belief, w: LossWeights) -> float:
    return risk_of_decision(apply_rule(rule, belief, w), belief, w)


def expected_risk_after(rule: str, belief: Belief, w: LossWeights, actions: Sequence[CostAction],
                        outcome_fn: Callable[[int, str], object], action_id: str) -> float:
    """E_M[R_M(H, Y_a)|H] —— 按该动作的全部非空结局分支加权 (§10.2)。"""
    a = next((x for x in actions if x.action_id == action_id), None)
    if a is None:
        raise ActiveQueryContractError(f"动作 {action_id!r} 不在册")
    prof = lambda s: outcome_fn(s, action_id)
    tot = 0.0
    for _key, mass, sub in belief.branch_by_outcome(prof):
        tot += float(mass) * rule_risk_under_rule(rule, sub, w)
    return tot


def expected_gain(rule: str, belief: Belief, w: LossWeights, actions: Sequence[CostAction],
                  outcome_fn: Callable[[int, str], object], action_id: str) -> float:
    """U_M(a|H) = R_M(H) - E_M[R_M(H, Y_a)|H] (可为负: 如实报, 禁 clamp 成非负)。"""
    return rule_risk_under_rule(rule, belief, w) - expected_risk_after(
        rule, belief, w, actions, outcome_fn, action_id)


def robust_unit_cost_choice(models: Mapping[str, Belief], w: LossWeights,
                            actions: Sequence[CostAction],
                            outcome_fn: Callable[[int, str], object], *,
                            rule: str = RULE_BAYES_TERNARY,
                            candidates: Optional[Sequence[str]] = None,
                            require_min_models: int = 2) -> dict:
    """a* = argmax_a min_M U_M(a|H)/c(a)。族内成员不足 require_min_models ⇒ 标 VACUOUS。"""
    if not models:
        raise ExpectationNotDeclared("缺规划分布: 期望未定义, 只能走有限最坏成本退路或仅输出候选动作")
    ids = sorted(models)
    pool = sorted(actions, key=lambda a: a.action_id)
    cand = [a for a in pool if candidates is None or a.action_id in set(candidates)]
    if not cand:
        raise ActiveQueryContractError("候选动作集为空: 拒发")
    vacuous = {"flag": len(ids) < require_min_models,
               "kind": ("VACUOUS_SINGLE_MODEL (min-M 退化为普通期望选点, 不作稳健证据)"
                        if len(ids) < require_min_models else "MEASURED")}
    rows = []
    for a in cand:
        per = {}
        for mid in ids:
            per[mid] = expected_gain(rule, models[mid], w, pool, outcome_fn, a.action_id)
        worst = min(per.values())
        rows.append({"action_id": a.action_id, "cost_int": a.cost_int,
                     "U_per_model": {k: float(v) for k, v in per.items()},
                     "min_M_U": float(worst),
                     "unit_cost_gain": float(worst) / float(a.cost_int),
                     "ties_break": a.action_id})
    rows.sort(key=lambda r: (-round(r["unit_cost_gain"], 12), r["action_id"]))
    return {"chosen": rows[0]["action_id"], "rows": rows, "model_family": ids,
            "rule": rule, "min_M_semantics": "逐模型算 U 再取最小 (禁先平均概率再算期望)",
            "robustness_kind": vacuous, "claim_scope": DEVELOPMENT_ONLY,
            "validity_note": "本节全部读数属规划轨; 有效性只来自证书/置信集合, 不借用此处的 U"}


def finite_worst_case_candidates(belief: Belief, actions: Sequence[CostAction],
                                 outcome_fn: Callable[[int, str], object]) -> dict:
    """缺规划分布时的退路 (§10.2): 只给有限最坏成本候选序, 不给任何数字收益主张。"""
    res = optimal_worst_case_cost(list(range(belief.n)), lambda i: belief.labels[i], actions,
                                  lambda i, a: outcome_fn(belief.states[i], a))
    if res["cost_int"] is None:
        return {"status": "UNSEPARABLE_EXISTS", "candidates": [],
                "no_numeric_gain_claim": True, "uncovered_count": res["uncovered_count"],
                "note": "存在不可区分跨答案对: 有限最坏成本不存在, 不得伪造候选序"}
    return {"status": "CANDIDATES_ONLY", "candidates": [a.action_id for a in
                                                        sorted(actions, key=lambda x: x.action_id)],
            "worst_case_cost_int": res["cost_int"], "no_numeric_gain_claim": True,
            "note": "候选序按动作 id 规范并列给出, 不含期望收益主张"}


# ---------------------------------------------------------------------------
# §10.3 便宜代理: 跨答案解释对消除 (EC2 形) + 前 K 筛 + 两步前瞻
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyScore:
    action_id: str
    cost_int: int
    separated_pair_mass: float
    separated_pair_count: int
    total_pair_count: int
    score_unit_cost: float

    @property
    def is_validity_bound(self) -> bool:
        return False

    def as_validity_bound(self):
        raise ProxyNotBoundViolation(
            f"代理分 {self.action_id} = {self.score_unit_cost} 只是选点排序依据, "
            "不是目标界, 也不是任何有效性主张 (§10.3)")

    def to_dict(self) -> dict:
        return {"action_id": self.action_id, "cost_int": self.cost_int,
                "separated_pair_mass": self.separated_pair_mass,
                "separated_pair_count": self.separated_pair_count,
                "total_pair_count": self.total_pair_count,
                "score_unit_cost": self.score_unit_cost,
                "role": "SELECTION_ORDERING_ONLY", "is_validity_bound": False,
                "claim_scope": DEVELOPMENT_ONLY}


def cross_answer_pairs(belief: Belief) -> Tuple[List[Tuple[int, int, Fraction]], Fraction, int]:
    pairs = []
    tot = Fraction(0)
    for i in range(belief.n):
        for j in range(i + 1, belief.n):
            if belief.labels[i] == belief.labels[j]:
                continue
            w_ij = belief.probs[i] * belief.probs[j]
            pairs.append((i, j, w_ij))
            tot += w_ij
    return pairs, tot, len(pairs)


def ec2_proxy_ranking(belief: Belief, actions: Sequence[CostAction],
                      outcome_fn: Callable[[int, str], object]) -> dict:
    """w_i·w_j 跨答案对质量: 每动作可消除的对质量 / 成本 ⇒ 候选排序 (§10.3)。

    输出只许作选点排序; ProxyScore 结构上不是界 (as_validity_bound 必抛)。
    """
    pairs, tot_mass, n_pairs = cross_answer_pairs(belief)
    rows = []
    for a in sorted(actions, key=lambda x: x.action_id):
        mass = Fraction(0)
        cnt = 0
        for i, j, w_ij in pairs:
            if outcome_fn(belief.states[i], a.action_id) != outcome_fn(belief.states[j], a.action_id):
                mass += w_ij
                cnt += 1
        rows.append(ProxyScore(a.action_id, a.cost_int, float(mass), cnt, n_pairs,
                               float(mass) / float(a.cost_int)))
    rows.sort(key=lambda r: (-round(r.score_unit_cost, 12), r.action_id))
    return {"rows": rows, "n_cross_pairs": n_pairs, "total_pair_mass": float(tot_mass),
            "zero_mass_actions": [r.action_id for r in rows if r.separated_pair_count == 0],
            "role": "SELECTION_ORDERING_ONLY", "claim_scope": DEVELOPMENT_ONLY,
            "provenance_note": ("EC2 形跨答案解释对消除思路的类比使用; 本单不宣称自己的损失/宽度函数"
                                "继承其自适应次模保证 (任务源 §10.3 末段)")}


def top_k_filter(ranking_rows: Sequence[ProxyScore], K: int) -> dict:
    """前 K 筛: 覆盖率与被筛动作必须逐名在册, 禁悄悄省不利动作。"""
    if K <= 0:
        raise ActiveQueryContractError(f"K 必须为正: {K!r}")
    if K > len(ranking_rows):
        raise ActiveQueryContractError(f"K={K} 超过动作池 {len(ranking_rows)}: 拒发")
    selected = list(ranking_rows[:K])
    dropped = list(ranking_rows[K:])
    if len(selected) + len(dropped) != len(ranking_rows):
        raise ActiveQueryContractError("前 K 筛丢失动作条目: 禁悄悄省不利动作")
    return {"selected": [r.action_id for r in selected],
            "dropped": [{"action_id": r.action_id, "score_unit_cost": r.score_unit_cost,
                         "reason": "OUTSIDE_TOP_K"} for r in dropped],
            "K": K, "pool_size": len(ranking_rows),
            "action_coverage": len(selected) / float(len(ranking_rows)),
            "enumerated_all_actions": sorted(r.action_id for r in ranking_rows),
            "silent_drop_detected": False}


def two_step_lookahead(models: Mapping[str, Belief], w: LossWeights,
                       actions: Sequence[CostAction],
                       outcome_fn: Callable[[int, str], object], *,
                       candidates: Sequence[str],
                       rule: str = RULE_BAYES_TERNARY) -> dict:
    """两步前瞻: 第一步候选 a1 → 每个非空结局分支内再择优第二动作 (自适应二段)。

    收益 = R(H) - E[二段后的 R]; 成本 = E[c(a1)+c(a2_branch)]。与单步贪心同面并报。
    """
    if not models:
        raise ExpectationNotDeclared("缺规划分布: 两步前瞻的期望未定义")
    pool = sorted(actions, key=lambda a: a.action_id)
    cost_of = {a.action_id: a.cost_int for a in pool}
    rows = []
    for a1 in sorted(set(candidates)):
        if a1 not in cost_of:
            raise ActiveQueryContractError(f"候选 {a1!r} 不在动作册")
        per_model_gain, per_model_exp_cost, branch_detail = {}, {}, []
        for mid, bel in sorted(models.items()):
            r0 = rule_risk_under_rule(rule, bel, w)
            exp_gain, exp_cost = Fraction(0), Fraction(0)
            for key, mass, sub in bel.branch_by_outcome(lambda s: outcome_fn(s, a1)):
                subrows = []
                for a2 in pool:
                    if a2.action_id == a1:
                        continue
                    rr = expected_risk_after(rule, sub, w, pool, outcome_fn, a2.action_id)
                    subrows.append({"action_id": a2.action_id, "risk_after": rr})
                if not subrows:
                    best_risk, best_a2 = rule_risk_under_rule(rule, sub, w), None
                else:
                    pick = min(subrows, key=lambda r: (round(r["risk_after"], 12), r["action_id"]))
                    sub_r0 = rule_risk_under_rule(rule, sub, w)
                    if pick["risk_after"] >= sub_r0 - 1e-15:
                        best_risk, best_a2 = sub_r0, None  # 二段无改善 ⇒ 不花第二次成本
                    else:
                        best_risk, best_a2 = pick["risk_after"], pick["action_id"]
                exp_gain += mass * (Fraction(str(r0)) - Fraction(str(best_risk)))
                exp_cost += mass * (cost_of[a1] + (cost_of[best_a2] if best_a2 else 0))
                branch_detail.append({"model": mid, "branch": repr(key),
                                      "branch_mass": float(mass), "second_action": best_a2,
                                      "risk_after_two_steps": float(best_risk)})
            per_model_gain[mid] = float(exp_gain)
            per_model_exp_cost[mid] = float(exp_cost)
        worst_gain = min(per_model_gain.values())
        worst_cost = max(per_model_exp_cost.values())
        rows.append({"action_id": a1, "U2_per_model": per_model_gain,
                     "expected_cost_per_model": per_model_exp_cost,
                     "min_M_U2": float(worst_gain),
                     "min_M_unit_cost": (float(worst_gain) / float(worst_cost)
                                         if worst_cost else None),
                     "branch_detail": [b for b in branch_detail
                                       if b["model"] == min(per_model_gain,
                                                            key=lambda k: per_model_gain[k])]})
    rows.sort(key=lambda r: (-(round(r["min_M_unit_cost"], 12)
                              if r["min_M_unit_cost"] is not None else -float("inf")),
                             r["action_id"]))
    return {"chosen": rows[0]["action_id"], "rows": rows, "depth": 2, "rule": rule,
            "single_step_reference": "单步见 robust_unit_cost_choice (同 belief 同权重同动作册)",
            "claim_scope": DEVELOPMENT_ONLY,
            "action_pool_enumerated": [a.action_id for a in pool]}


def optimal_lookahead_reference(models: Mapping[str, Belief], w: LossWeights,
                                actions: Sequence[CostAction],
                                outcome_fn: Callable[[int, str], object], *,
                                rule: str = RULE_BAYES_TERNARY,
                                max_depth: Optional[int] = None) -> dict:
    """风险平面参照: 完整枚举动作序列的最优期望风险 (小动作集精确解)。

    只枚举非空分支, 空结局不当免费成功 (§10.6)。这是开发基准参照, 不是有效界。
    """
    if not models:
        raise ExpectationNotDeclared("缺规划分布: 最优风险树参照的期望未定义")
    ids = sorted(models)
    depth_cap = max_depth if max_depth is not None else len(actions)
    per_model = {}
    for mid in ids:
        bel = models[mid]

        def solve(b: Belief, used: Tuple[str, ...], d: int):
            r0 = rule_risk_under_rule(rule, b, w)
            options = [(r0, 0.0, (), "")]  # 「不再补测」始终是候选: 无改善即不花成本
            if d > 0:
                for a in actions:
                    if a.action_id in used:
                        continue
                    branches = b.branch_by_outcome(lambda s: outcome_fn(s, a.action_id))
                    exp_r, exp_c = Fraction(0), Fraction(0)
                    for _k, mass, sub in branches:
                        rr, cc, seq = solve(sub, used + (a.action_id,), d - 1)
                        exp_r += mass * Fraction(str(rr))
                        exp_c += mass * Fraction(str(a.cost_int + cc))
                    options.append((float(exp_r), float(exp_c), (a.action_id,) + seq, a.action_id))
            best = min(options, key=lambda t: (round(t[0], 12), t[1], t[3]))
            return best[0], best[1], best[2]

        r_star, c_star, seq = solve(bel, (), depth_cap)
        per_model[mid] = {"risk_before": rule_risk_under_rule(rule, bel, w),
                          "optimal_risk_after_tree": float(r_star),
                          "expected_cost_of_tree": float(c_star),
                          "first_action_of_optimal_tree": (seq[0] if seq else None),
                          "depth_cap": depth_cap}
    return {"per_model": per_model, "models": ids,
            "method": "exhaustive_over_action_sequences_nonempty_branches_only",
            "optimality": "有限动作集全枚举 (开发基准参照; 不外推为界)",
            "claim_scope": DEVELOPMENT_ONLY}


def assert_not_optimality_evidence(claim: Mapping) -> None:
    """顺序最优主张必须有完整决策树参照令牌; 『单步赢随机』不构成该证据 (§10.4)。"""
    kinds = set(claim.get("evidence_kinds") or [])
    if "SEQUENTIAL_OPTIMAL" in kinds and "EXHAUSTIVE_TREE_REFERENCE" not in kinds:
        raise ActiveQueryContractError(
            "顺序最优主张缺完整树参照令牌: 赢随机不是顺序最优证据 (§10.4)")
    if "BEATS_RANDOM" in kinds and not (kinds - {"BEATS_RANDOM"}):
        raise ActiveQueryContractError("只有赢随机一项证据: 不得据此作任何最优性主张")


# ---------------------------------------------------------------------------
# §10.5-10.6 真闭环骨架 + 全结局入账
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureModel:
    """预先给定的失败/未检出/超时模型 (§10.6: 不静默重抽)。"""

    p_action_failure: Fraction
    p_no_detection: Fraction
    p_timeout: Fraction
    provenance: str

    def __post_init__(self):
        for name in ("p_action_failure", "p_no_detection", "p_timeout"):
            v = getattr(self, name)
            if not isinstance(v, Fraction) or v < 0 or v > 1:
                raise ActiveQueryContractError(f"{name} 需 [0,1] 内 Fraction: {v!r}")
        if not str(self.provenance or "").strip():
            raise ActiveQueryContractError("失败模型必须声明依据 (禁把默认值当真值)")

    @classmethod
    def declared(cls, p_fail: object, p_nodet: object, p_timeout: object,
                 provenance: str) -> "FailureModel":
        return cls(_frac(p_fail), _frac(p_nodet), _frac(p_timeout), provenance)

    def to_dict(self) -> dict:
        return {"p_action_failure": str(self.p_action_failure),
                "p_no_detection": str(self.p_no_detection), "p_timeout": str(self.p_timeout),
                "provenance": self.provenance}


OUTCOME_SUCCESS = "SUCCESS_DETECTED"
OUTCOME_EMPTY_UPDATE = "EMPTY_UPDATE_SET_MODEL_CONFLICT"
OUTCOME_FAILURE = "ACTION_FAILED"
OUTCOME_NO_DETECTION = "NOT_DETECTED"
OUTCOME_TIMEOUT = "SEARCH_UNRESOLVED_TIMEOUT"
FULL_OUTCOME_KINDS = (OUTCOME_SUCCESS, OUTCOME_EMPTY_UPDATE, OUTCOME_FAILURE,
                      OUTCOME_NO_DETECTION, OUTCOME_TIMEOUT)


@dataclass(frozen=True)
class OutcomeBranch:
    kind: str
    prob: Fraction
    cost_int: int
    gain: float
    status: str
    informational: bool
    in_denominator: bool = True

    def to_dict(self) -> dict:
        return {"kind": self.kind, "prob": str(self.prob), "prob_float": float(self.prob),
                "cost_int": self.cost_int, "expected_gain": self.gain, "status": self.status,
                "informational": self.informational, "in_denominator": True}


class OutcomeLedger:
    """全结局账 (§10.6): 空集/失败/未检出/超时都进期望账, 分母不缩。"""

    def __init__(self, rows: Sequence[OutcomeBranch]):
        if not rows:
            raise ActiveQueryContractError("结局账为空面: 禁发任何期望读数 (空比较面纪律)")
        kinds = [r.kind for r in rows]
        if len(set(kinds)) != len(kinds):
            raise ActiveQueryContractError(f"结局账种类重复: {kinds}")
        self.rows = list(rows)

    @property
    def prob_sum(self) -> Fraction:
        return sum((r.prob for r in self.rows), Fraction(0))

    def close(self) -> dict:
        s = self.prob_sum
        return {"probability_closure": str(s), "closed_exactly": bool(s == 1),
                "branches": len(self.rows),
                "kinds_present": sorted(r.kind for r in self.rows),
                "kinds_absent": sorted(set(FULL_OUTCOME_KINDS) - {r.kind for r in self.rows}),
                "every_branch_in_denominator": all(r.in_denominator for r in self.rows),
                "note": "闭合 = Σp 精确等于 1 (Fraction 记账) 且无分支被过滤"}

    def full_expectation(self) -> dict:
        dropped = [r.kind for r in self.rows if not r.in_denominator]
        if dropped:
            raise ActiveQueryContractError(
                f"结局分支被剔出分母 {dropped}: 分母不得随结局改变 (§14.2『未解决格被删』)")
        g = sum((r.prob * Fraction(str(r.gain)) for r in self.rows), Fraction(0))
        c = sum((r.prob * r.cost_int for r in self.rows), Fraction(0))
        return {"expected_gain": float(g), "expected_cost": float(c),
                "expected_gain_frac": str(g), "expected_cost_frac": str(c),
                "branches_used": len(self.rows), "basis": "FULL_OUTCOME_LEDGER"}

    def naive_finite_only_mean(self) -> dict:
        """负例路 (§10.6): isfinite 过滤后只平均容易结局 —— 只用于暴露偏离, 禁入交付。"""
        easy = [r for r in self.rows if r.informational]
        if not easy:
            raise ActiveQueryContractError("容易结局为空: 该负例路本身无定义, 不得记均值")
        g = sum((r.gain for r in easy), 0.0) / len(easy)
        c = sum((float(r.cost_int) for r in easy), 0.0) / len(easy)
        return {"expected_gain": float(g), "expected_cost": float(c),
                "branches_used": len(easy), "branches_dropped": len(self.rows) - len(easy),
                "dropped_kinds": sorted(r.kind for r in self.rows if not r.informational),
                "basis": "EASY_ONLY_MEAN_FORBIDDEN_IN_DELIVERY"}


def build_outcome_ledger(*, detection_gain: float, failure_cost: int, no_detection_cost: int,
                         timeout_cost: int, failure_model: FailureModel,
                         p_empty_update: object, success_cost: int) -> OutcomeLedger:
    """按预登记结局概率造账 (§10.6): 五类结局齐, 概率和精确为 1。"""
    p_fail = failure_model.p_action_failure
    p_nd = failure_model.p_no_detection
    p_to = failure_model.p_timeout
    p_empty = _frac(p_empty_update)
    p_succ = 1 - p_fail - p_nd - p_to - p_empty
    if p_succ < 0:
        raise ActiveQueryContractError(
            f"结局概率超配: 1 - ({p_fail}+{p_nd}+{p_to}+{p_empty}) < 0")
    rows = [
        OutcomeBranch(OUTCOME_SUCCESS, p_succ, int(success_cost), float(detection_gain),
                      "NARROWED_OR_CERTIFIED", True),
        OutcomeBranch(OUTCOME_EMPTY_UPDATE, p_empty, int(success_cost), 0.0, "MODEL_CONFLICT", False),
        OutcomeBranch(OUTCOME_FAILURE, p_fail, int(failure_cost), 0.0, "ACTION_FAILED", False),
        OutcomeBranch(OUTCOME_NO_DETECTION, p_nd, int(no_detection_cost), 0.0, "NOT_DETECTED", False),
        OutcomeBranch(OUTCOME_TIMEOUT, p_to, int(timeout_cost), 0.0, "SEARCH_UNRESOLVED", False),
    ]
    return OutcomeLedger(rows)


class Sensor:
    """观测释放器: 每步只释放本步动作的记录; 隐藏实现状态经 poison 把守。

    outcome_schedule 由世界脚本 (审计面) 预先声明哪一步发生失败/未检出/超时,
    sensor 按步查表 —— 消费路径事先不知道, 只能从已释放记录里读到 (§10.5-10.6)。
    """

    def __init__(self, *, realized_state: int, action_ids: Sequence[str],
                 outcome_fn: Callable[[int, str], object],
                 outcome_schedule: Optional[Mapping[int, str]] = None):
        self._action_ids = list(action_ids)
        self._outcome_fn = outcome_fn
        self._realized = int(realized_state)
        self._schedule = {int(k): str(v) for k, v in (outcome_schedule or {}).items()}
        bad = set(self._schedule.values()) - {OUTCOME_FAILURE, OUTCOME_NO_DETECTION, OUTCOME_TIMEOUT}
        if bad:
            raise ActiveQueryContractError(f"结局日程含未知结局类型 {sorted(bad)}")
        self.hidden_truth = make_poisoned_truth()
        self._released_steps: List[int] = []

    def release(self, action_id: str, step: int) -> dict:
        if action_id not in self._action_ids:
            raise ContractError(f"动作 {action_id!r} 不在 sensor 可执行册内")
        if step != len(self._released_steps) + 1:
            raise ActiveQueryContractError(
                f"步序断裂: 请求 step={step}, 已释放 {len(self._released_steps)} 步")
        kind = self._schedule.get(step)
        payload = {"step": int(step), "action_id": action_id,
                   "execution": kind or "OK",
                   "outcome": None if kind else self._outcome_fn(self._realized, action_id),
                   "detected": kind is None}
        assert_released_payload(payload)
        self._released_steps.append(step)
        return payload

    def make_future_payload(self, action_id: str, step: int) -> dict:
        """负例专用: 造一条尚未到步的记录 (生产路径禁调; 供未来记录注入负例消费)。"""
        if step <= len(self._released_steps):
            raise ActiveQueryContractError("该步已释放, 不是未来记录: 负例无效 (NO_OP)")
        return {"step": int(step), "action_id": action_id, "execution": "OK",
                "outcome": self._outcome_fn(self._realized, action_id), "detected": True}


REQUIRED_RECORD_KEYS = ("step", "action_id", "execution", "outcome", "detected")


class ReleasedLedger:
    """消费侧账本: 只接受已到步且动作与本轨迹选择一致的真实释放。"""

    def __init__(self):
        self.view: List[dict] = []
        self.high_water = 0

    def absorb(self, payload: Mapping, *, chosen_action_id: str) -> dict:
        if payload.get("action_id") != chosen_action_id:
            raise ActiveQueryContractError(
                f"释放记录动作 {payload.get('action_id')!r} != 本步选定 {chosen_action_id!r}")
        step = int(payload["step"])
        if step > self.high_water + 1:
            raise FutureRecordViolation(
                f"读到 step={step} 的记录而账本高水位 = {self.high_water}: 未来记录注入")
        assert_released_payload(payload)
        if any(k not in payload for k in REQUIRED_RECORD_KEYS):
            raise ActiveQueryContractError("释放记录字段不全: 拒入账")
        rec = {k: payload[k] for k in REQUIRED_RECORD_KEYS}
        if rec["detected"] is False and rec["outcome"] is not None:
            raise ActiveQueryContractError("未检出/失败结局却带位值: 结局语义矛盾, 拒入账")
        self.view.append(rec)
        self.high_water = step
        return rec

    def outcome_sequence(self) -> List[object]:
        return [r["outcome"] for r in self.view]


def belief_after_released(belief: Belief, ledger: ReleasedLedger,
                          outcome_fn: Callable[[int, str], object]) -> Belief:
    """更新集合: 只按已释放且检出的记录逐步限制信念 (§10.5), 不读实现状态。

    未检出/失败记录不改变集合但成本照计 (§10.6: 结局不得从账上消失)。
    """
    cur = belief
    for rec in ledger.view:
        if rec["detected"] is False:
            continue
        keep = [outcome_fn(s, rec["action_id"]) == rec["outcome"] for s in cur.states]
        cur = cur.restrict_by_mask(keep)
    return cur


def run_closed_loop(*, belief: Belief, sensor: Sensor, ledger: ReleasedLedger,
                    actions: Sequence[CostAction],
                    outcome_fn: Callable[[int, str], object], weights: LossWeights,
                    chooser: Callable[[Belief, Sequence[str]], str],
                    budget_cost: int, search_budget_steps: int) -> dict:
    """真闭环 (§10.5): 选动作 → sensor 释放 → 更新集合 → 验证书 → 下一步。

    每步只读已释放记录; 预算/搜索步数耗尽或证书达成即停 (§10.7)。
    """
    w = weights
    pool_ids = [a.action_id for a in actions]
    cost_of = {a.action_id: a.cost_int for a in actions}
    cur = belief
    spent = 0
    steps = []
    status = "CONTINUE"
    for step in range(1, search_budget_steps + 1):
        target = cur.target_set()
        dec = certified_decision(target, w)
        if dec.d != "?":
            status = "CERTIFIED_SINGLE_VALUE"
            break
        if spent >= budget_cost:
            status = "BUDGET_EXHAUSTED"
            break
        remaining = [i for i in pool_ids if i not in [s["action_id"] for s in steps]]
        if not remaining:
            status = "ACTION_POOL_EXHAUSTED_AMBIGUOUS"
            break
        a = chooser(cur, remaining)
        if a not in remaining:
            raise ActiveQueryContractError(f"chooser 返回未在册/已用动作 {a!r}")
        payload = sensor.release(a, step)
        rec = ledger.absorb(payload, chosen_action_id=a)
        spent += cost_of[a]
        if rec["detected"] is False:
            row_status = rec["execution"]
            steps.append({"step": step, "action_id": a, "outcome": None,
                          "execution": row_status, "belief_members_after": cur.n,
                          "row_status": row_status, "cost_spent": spent, "narrowing": 0})
            if row_status == OUTCOME_TIMEOUT:
                status = "SEARCH_UNRESOLVED"
                break
            continue
        try:
            nxt = cur.restrict_by_mask([outcome_fn(s, a) == rec["outcome"] for s in cur.states])
            row_status = "NARROWED" if nxt.n < cur.n else "NO_NARROWING"
            cur = nxt
        except EmptyBeliefAfterRestriction:
            row_status = "MODEL_CONFLICT"
            status = "MODEL_CONFLICT"
            steps.append({"step": step, "action_id": a, "outcome": rec["outcome"],
                          "belief_members_after": cur.n, "row_status": row_status})
            break
        steps.append({"step": step, "action_id": a, "outcome": rec["outcome"],
                      "belief_members_after": cur.n, "row_status": row_status,
                      "cost_spent": spent})
    else:
        status = "SEARCH_UNRESOLVED"
    final = cur.target_set()
    dec = certified_decision(final, w)
    return {"steps": steps, "status": status, "final_target_set": final,
            "decision": dec.to_dict(), "cost_spent": spent, "budget_cost": int(budget_cost),
            "released_records": len(ledger.view),
            "search_budget_steps": int(search_budget_steps),
            "loop_semantics": "每步只读 sensor 已释放记录; 预计划轨迹由 probe_adaptivity 单独裁定"}


def probe_adaptivity(chooser: Callable[[Belief, Sequence[str]], str], belief: Belief,
                     action_pool: Sequence[CostAction],
                     outcome_fn: Callable[[int, str], object]) -> dict:
    """预计划探测器: 对第一步每个反事实结局分支重跑选点, 后继动作逐位不变 ⇒ 非自适应。"""
    ids = [a.action_id for a in action_pool]
    if not ids:
        raise ActiveQueryContractError("动作池为空: 无法探测自适应")
    base = chooser(belief, ids)
    if base not in ids:
        raise ActiveQueryContractError(f"chooser 返回不在册动作 {base!r}")
    picks = []
    for _key, _mass, sub in belief.branch_by_outcome(lambda s: outcome_fn(s, base)):
        remaining = [a.action_id for a in action_pool if a.action_id != base]
        picks.append(chooser(sub, remaining) if remaining else None)
    distinct = len(set(p for p in picks if p is not None))
    branches_with_choice = sum(1 for pr in picks if pr is not None)
    pool_size = len(ids)
    has_teeth = pool_size >= 3 and len(picks) >= 2 and branches_with_choice >= 2
    adaptive = distinct > 1
    return {"first_action": base, "second_action_per_branch": picks,
            "branches_probed": len(picks), "distinct_second_actions": distinct,
            "pool_size": pool_size,
            "probe_teeth": ("MEASURED" if has_teeth else
                            "NO_TEETH_VACUOUS (池过小或分支过少: 后继选择唯一, "
                            "该谓词在此夹具上无从成立, 不作自适应证据)"),
            "adaptation_kind": ("ADAPTIVE (第二动作随已释放结局改变)" if adaptive
                                else "PRE_PLANNED (结局无关, 同 G22 预先选井形)"),
            "may_claim_adaptive_closed_loop": bool(adaptive and has_teeth)}


def assert_adaptive_claim_allowed(probe: dict) -> None:
    if not probe["may_claim_adaptive_closed_loop"]:
        raise NotAdaptiveViolation(
            f"{probe['adaptation_kind']}: 不得自称已验证『看结果再选』的自适应闭环 (§10.5)")


# ---------------------------------------------------------------------------
# §10.7 停止条件五条
# ---------------------------------------------------------------------------

STOP_ALL_CERTIFIED = "ALL_TASKS_CERTIFIED"
STOP_NO_MEANINGFUL_GAIN = "BEST_ACTION_UPPER_BOUND_BELOW_MPE"
STOP_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
STOP_UNSEPARABLE_PAIR = "CROSS_ANSWER_PAIR_UNDISTINGUISHABLE"
STOP_MODEL_CONFLICT = "MODEL_CONFLICT_FIX_MODEL_FIRST"
STOP_CODES = (STOP_ALL_CERTIFIED, STOP_NO_MEANINGFUL_GAIN, STOP_BUDGET_EXHAUSTED,
              STOP_UNSEPARABLE_PAIR, STOP_MODEL_CONFLICT)

EVIDENCE_COMPLETE_FINITE = "COMPLETE_FINITE_REFERENCE"
EVIDENCE_VALID_UPPER_BOUND = "VALID_UPPER_BOUND"
EVIDENCE_BUDGET_LEDGER = "BUDGET_LEDGER"
EVIDENCE_UNSEPARABLE_WITNESS = "UNSEPARABLE_PAIR_WITNESS"
EVIDENCE_MODEL_CONFLICT_SET = "MODEL_CONFLICT_SET"
EVIDENCE_SAMPLING_SEARCH = "SAMPLING_SEARCH"

REQUIRED_EVIDENCE = {
    STOP_ALL_CERTIFIED: {EVIDENCE_COMPLETE_FINITE, EVIDENCE_VALID_UPPER_BOUND},
    STOP_NO_MEANINGFUL_GAIN: {EVIDENCE_COMPLETE_FINITE, EVIDENCE_VALID_UPPER_BOUND},
    STOP_BUDGET_EXHAUSTED: {EVIDENCE_BUDGET_LEDGER},
    STOP_UNSEPARABLE_PAIR: {EVIDENCE_UNSEPARABLE_WITNESS, EVIDENCE_COMPLETE_FINITE},
    STOP_MODEL_CONFLICT: {EVIDENCE_MODEL_CONFLICT_SET},
}


@dataclass(frozen=True)
class EvidenceToken:
    kind: str
    detail: str
    enumerated: Optional[int] = None

    def __post_init__(self):
        allowed = {EVIDENCE_COMPLETE_FINITE, EVIDENCE_VALID_UPPER_BOUND, EVIDENCE_BUDGET_LEDGER,
                   EVIDENCE_UNSEPARABLE_WITNESS, EVIDENCE_MODEL_CONFLICT_SET,
                   EVIDENCE_SAMPLING_SEARCH}
        if self.kind not in allowed:
            raise ActiveQueryContractError(f"证据令牌类型未知: {self.kind!r}")
        if not str(self.detail or "").strip():
            raise ActiveQueryContractError("证据令牌必须带说明")
        if self.kind == EVIDENCE_COMPLETE_FINITE and not self.enumerated:
            raise ActiveQueryContractError("完整有限参照令牌必须带枚举数 (缺伴随枚举面即拒发)")
        if self.kind == EVIDENCE_UNSEPARABLE_WITNESS and "PAIR" not in self.detail.upper():
            raise ActiveQueryContractError("不可区分令牌必须交出具名状态对")


def evaluate_stop(code: str, token: EvidenceToken, *, measured: Optional[float] = None,
                  mme: Optional[float] = None) -> dict:
    """停止判定 (§10.7): 证据类型不足 ⇒ 不停, 标 SEARCH_UNRESOLVED。"""
    if code not in STOP_CODES:
        raise ActiveQueryContractError(f"未知停止条件: {code!r}")
    ok_kinds = REQUIRED_EVIDENCE[code]
    if token.kind not in ok_kinds:
        return {"stop": False, "code": code, "decision": "CONTINUE", "status": "SEARCH_UNRESOLVED",
                "reason": f"证据类型 {token.kind} 不足以主张 {code}; 采样未找到差异不算",
                "required_kinds": sorted(ok_kinds)}
    if code == STOP_BUDGET_EXHAUSTED:
        if measured is None or mme is None:
            raise ActiveQueryContractError("预算耗尽判定需已花成本与预算两者")
        if not (measured >= mme):
            return {"stop": False, "code": code, "decision": "CONTINUE", "status": "PROCEED",
                    "reason": f"已花 {measured} 未达预算 {mme}: 不得称预算耗尽",
                    "required_kinds": sorted(ok_kinds)}
        return {"stop": True, "code": code, "decision": "STOP", "status": "STOP_CONDITION_MET",
                "reason": token.detail, "cost_spent": float(measured), "budget": float(mme),
                "required_kinds": sorted(ok_kinds)}
    if code == STOP_UNSEPARABLE_PAIR:
        if token.kind == EVIDENCE_UNSEPARABLE_WITNESS and token.enumerated is None:
            raise ActiveQueryContractError("不可区分对令牌必须带枚举动作数 (完整参照过的动作集大小)")
        return {"stop": True, "code": code, "decision": "STOP", "status": "STOP_CONDITION_MET",
                "reason": token.detail, "actions_enumerated": token.enumerated,
                "required_kinds": sorted(ok_kinds)}
    if code == STOP_MODEL_CONFLICT:
        return {"stop": True, "code": code, "decision": "STOP", "status": "STOP_CONDITION_MET",
                "reason": token.detail, "required_kinds": sorted(ok_kinds)}
    if code == STOP_NO_MEANINGFUL_GAIN:
        if measured is None or mme is None:
            raise ActiveQueryContractError("该停止条件需可信上界读数与 MPE 两者齐")
        if not (measured <= mme):
            return {"stop": False, "code": code, "decision": "CONTINUE", "status": "PROCEED",
                    "reason": f"最佳动作收益上界 {measured} 未低于 MPE {mme}",
                    "required_kinds": sorted(ok_kinds)}
        return {"stop": True, "code": code, "decision": "STOP", "status": "STOP_CONDITION_MET",
                "reason": token.detail, "upper_bound": float(measured), "mpe": float(mme),
                "required_kinds": sorted(ok_kinds)}
    return {"stop": True, "code": code, "decision": "STOP", "status": "STOP_CONDITION_MET",
            "reason": token.detail, "required_kinds": sorted(ok_kinds)}


def verify_unseparable_witness(belief: Belief, actions: Sequence[CostAction],
                               outcome_fn: Callable[[int, str], object],
                               pair: Tuple[int, int]) -> dict:
    """不可区分对令牌的可重验性 (§10.7): 现场重算,  separable 的对不得当结构歧义证据。"""
    i, j = int(pair[0]), int(pair[1])
    if not (0 <= i < belief.n and 0 <= j < belief.n) or i == j:
        raise ActiveQueryContractError(f"见证对下标非法: {pair}")
    if belief.labels[i] == belief.labels[j]:
        raise ActiveQueryContractError(f"见证 ({i},{j}) 同答案: 不是跨答案对, 令牌拒收")
    seps = [a.action_id for a in actions
            if outcome_fn(belief.states[i], a.action_id) != outcome_fn(belief.states[j], a.action_id)]
    if seps:
        raise ActiveQueryContractError(
            f"见证 ({i},{j}) 可被 {seps} 区分: 结构歧义主张不成立 (禁把可分对当不可分)")
    return {"pair": [belief.states[i], belief.states[j]], "labels": [belief.labels[i], belief.labels[j]],
            "actions_enumerated": len(actions), "separating_actions": [],
            "verdict": "UNSEPARABLE_CONFIRMED_BY_REPLAY"}


def verify_model_conflict(belief: Optional[Belief]) -> dict:
    """MODEL_CONFLICT 令牌: 只有空可行集合才可发, 非空即拒 (§14.3-5)。"""
    if belief is not None:
        raise ActiveQueryContractError(
            f"可行集合非空 (成员 {belief.n}): 不得发 MODEL_CONFLICT 令牌")
    return {"verdict": "MODEL_CONFLICT_CONFIRMED", "note": "观测签名不在允许状态空间像内"}


def verify_all_certified(certificates: Sequence[Mapping], required_status: Sequence[str]) -> dict:
    """全任务有证书: 逐任务现场核对状态, 缺任一即拒停 (§10.7 第一条)。"""
    if not certificates:
        raise ActiveQueryContractError("证书清单为空: 空比较面禁发相等/全覆盖结论")
    need = set(required_status)
    missing = [c.get("query_id") for c in certificates if c.get("status") not in need]
    if missing:
        raise ActiveQueryContractError(f"以下任务无所需证书: {missing[:5]} (共 {len(missing)})")
    return {"verdict": "ALL_CERTIFIED", "certificates": len(certificates),
            "statuses_required": sorted(need), "enumerated": len(certificates)}


def mechanism_divergence_row(*, mechanism_gain: float, target_risk_delta: float,
                             total_cost: float) -> dict:
    """机制收益方向相反时不机械停 (§10.7 末段): 仍按目标风险与总成本判。"""
    return {"mechanism_gain": float(mechanism_gain),
            "target_risk_delta": float(target_risk_delta), "total_cost": float(total_cost),
            "mechanical_stop_triggered": False,
            "rule": "机制栏与目标栏方向相反时不停; 区分动作仍可作候选, 判据是目标风险与总成本"}


# ---------------------------------------------------------------------------
# 出件纪律把守
# ---------------------------------------------------------------------------


def assert_development_only(doc: Mapping) -> None:
    """规划/代理类出件必须带 DEVELOPMENT_ONLY; 出现有效性主张字段即拒。"""
    if doc.get("claim_scope") != DEVELOPMENT_ONLY:
        raise ActiveQueryContractError("本单出件缺 claim_scope=DEVELOPMENT_ONLY: 拒发")
    for k in doc:
        if str(k).lower() in ("guarantee_kind", "coverage", "validity_bound", "bound",
                              "confidence_level"):
            raise ActiveQueryContractError(f"开发面出件含有效性主张字段 {k!r}: 拒发")


def assert_serializable(doc) -> bool:
    """NaN/inf 一律拒 (S31 同向); 非有限读数不得成为任何结论的载体。"""
    import math

    def walk(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            raise ActiveQueryContractError(f"读数含非有限浮点: {x!r}")
        if isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    walk(doc)
    return True


def serialize(obj) -> str:
    import json
    assert_serializable(obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)

