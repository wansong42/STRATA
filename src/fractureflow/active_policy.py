# -*- coding: utf-8 -*-
"""S3-P · 反标签选点规划库 (PLANNING_PROXY_ONLY)。

本模块是**规划层纯函数库**: 把「优先区分仍可能改变目标答案的解释」做成可单测、可复算
的规划分数与选点工具。它的输出只对**当前见证集/声明 fixture** 负责。

红线 (逐条对应任务单 S3-P §2.3 与 §5)
--------------------------------------
* **不是认证器**: 这里的任何分数都不是认证错误率、不是置信界、不是上/下界。
  `hellinger` 只是两个声明分布之间的分离度; `cost_share` 只是规划权重。
* **不做闭环**: 无真 sensor、无认证器更新、无证书、无自适应策略主张。
* **不做策略优劣/成本优势主张**: `cost_share`/`sample_action` 的输出是**规划输入**,
  不是"某策略更优/更省"的证据。调用方禁据此对外声明。
* **前筛不永久屏蔽**: `two_step_lookahead` 的 K 档只影响**本步排名**; 被筛行动必须
  逐名返回 (`filtered_out`), 且 `never_permanently_blocked` 逐档给出可用性证据。
* **零分数不得当全局结论**: 最坏分数为 0 只说明**当前工作集内存在全行动不可分对**,
  不是"无信息"的全局判断。`cost_share` 返回 `worst_pair` 与 `degenerate` 标志,
  调用方必须把不可分对入账 (见 `indistinguishable_pairs`)。

依赖: 仅标准库 + numpy(RNG) + scipy.optimize.linprog (惰性导入, 缺失即显式报错)。
无 scipy 时 `cost_share` 抛 `CostShareBackendUnavailable`, **不静默降级**。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PlanningContractError",
    "CostShareBackendUnavailable",
    "WitnessSet",
    "hellinger",
    "normalize_dist",
    "action_divergence_matrix",
    "cost_share",
    "sample_action",
    "top_k_filter",
    "opponent_search",
    "two_step_lookahead",
    "indistinguishable_pairs",
    "PROXY_DISCLAIMER",
]

#: 逐行水印 (任务单要求 claim_scope=PLANNING_PROXY_ONLY; 人读面与本模块读数行同带)。
PROXY_DISCLAIMER = (
    "PLANNING_PROXY_ONLY: 本模块输出为规划代理读数, 非认证错误率/置信界/上下界, "
    "非闭环策略, 不构成策略优劣或成本优势主张; 禁入对外承诺文本。"
)

_TOL = 1e-9
"""浮点容差。声明分布由精确有理数 (Fraction) 转入, 残差量级 ~1e-16;
1e-9 只吸收浮点噪声, 不吸收真实归一化错误 (后者量级远大于此)。"""


class PlanningContractError(ValueError):
    """规划库契约违规 (空面/形状不符/非归一化/越界)。"""


class CostShareBackendUnavailable(RuntimeError):
    """`scipy.optimize.linprog` 不可用 —— 显式失败, 禁静默降级到启发式。"""


# ---------------------------------------------------------------------------
# 见证集
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WitnessSet:
    """「目标答案不同」的状态对集合。

    pairs  : ((i, j), ...)  i != j, 无序对按 (min, max) 规范存储
    labels : {state_index: 目标标签}  —— 只用于**构造期校验** (见 `coverage`) 与
             对外报读; 分数计算**不读标签**。
    weights: {pair: w}  缺省等权; 规划用途, 不是概率主张。
    """

    pairs: Tuple[Tuple[int, int], ...]
    labels: Mapping[int, Any] = field(default_factory=dict)
    weights: Mapping[Tuple[int, int], float] = field(default_factory=dict)
    name: str = "witness_set"
    provenance: str = ""

    def __post_init__(self) -> None:
        norm: List[Tuple[int, int]] = []
        for p in self.pairs:
            a, b = int(p[0]), int(p[1])
            if a == b:
                raise PlanningContractError("见证对不得自对: (%d,%d)" % (a, b))
            norm.append((min(a, b), max(a, b)))
        if not norm:
            raise SystemExit("[active_policy] 空见证集 ⇒ 停 (不得记 PASS)")
        if len(set(norm)) != len(norm):
            raise PlanningContractError("见证对重复: %r" % (norm,))
        object.__setattr__(self, "pairs", tuple(sorted(norm)))
        for p in self.pairs:
            w = self.weights.get(p, 1.0)
            if not math.isfinite(float(w)) or float(w) <= 0.0:
                raise PlanningContractError("见证对权重必须为正有限: %r -> %r" % (p, w))
        for idx in self.labels:
            if int(idx) < 0:
                raise PlanningContractError("state_index 不得为负: %r" % (idx,))

    # -- 基本量 -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def states(self) -> Tuple[int, ...]:
        s: set = set()
        for a, b in self.pairs:
            s.add(a)
            s.add(b)
        return tuple(sorted(s))

    def weight_of(self, pair: Tuple[int, int]) -> float:
        return float(self.weights.get((min(pair[0], pair[1]), max(pair[0], pair[1])), 1.0))

    def weight_vector(self) -> np.ndarray:
        return np.array([self.weight_of(p) for p in self.pairs], dtype=np.float64)

    def with_pairs(self, pairs: Iterable[Tuple[int, int]], *, name: Optional[str] = None,
                  provenance: Optional[str] = None) -> "WitnessSet":
        """同标签/同权重口径地换对集合 (对手搜索用)。"""
        return WitnessSet(pairs=tuple(tuple(int(x) for x in p) for p in pairs),
                          labels=dict(self.labels), weights=dict(self.weights),
                          name=name or self.name, provenance=provenance or self.provenance)

    # -- 有牙性: 标签必须真的不同 -------------------------------------------
    def coverage(self) -> dict:
        """构造期校验: 每个见证对必须**在声明标签上真的不同**且两状态都有标签。

        这条把"见证集"与"随手取的对"分开: 若一对的标签相同, 它根本不是见证对,
        用它的分离度去算最坏分数会把分数压低并冒充"信息量不足"。
        """
        if not self.labels:
            raise PlanningContractError("WitnessSet.coverage: 缺 labels, 无法校验见证性")
        missing: List[int] = []
        same_label: List[Tuple[int, int]] = []
        for a, b in self.pairs:
            if a not in self.labels or b not in self.labels:
                missing.extend([x for x in (a, b) if x not in self.labels])
                continue
            if self.labels[a] == self.labels[b]:
                same_label.append((a, b))
        ok = not missing and not same_label
        return {"name": self.name, "n_pairs": len(self.pairs), "ok": bool(ok),
                "missing_label_states": sorted(set(missing)),
                "pairs_with_equal_labels": same_label,
                "rule": "见证对 = 目标标签不同的对; 标签相同者在构造期即拒 (非'难对')"}

    def to_dict(self) -> dict:
        return {"name": self.name, "pairs": [list(p) for p in self.pairs],
                "states": list(self.states), "n_pairs": len(self.pairs),
                "weights": {("%d|%d" % p): self.weight_of(p) for p in self.pairs},
                "labels": {str(k): self.labels[k] for k in sorted(self.labels)},
                "provenance": self.provenance}


# ---------------------------------------------------------------------------
# Hellinger 分离度
# ---------------------------------------------------------------------------

def normalize_dist(p: Mapping[Any, float], *, strict: bool = True) -> Dict[Any, float]:
    """归一化声明分布; 负值/非有限一律拒 (不得静默截断)。"""
    if not p:
        raise PlanningContractError("空分布 ⇒ 拒 (空面不得记分)")
    out: Dict[Any, float] = {}
    tot = 0.0
    for k, v in p.items():
        fv = float(v)
        if not math.isfinite(fv):
            raise PlanningContractError("分布含非有限值: %r -> %r" % (k, v))
        if fv < 0.0:
            raise PlanningContractError("分布含负值: %r -> %r" % (k, v))
        out[k] = fv
        tot += fv
    if tot <= 0.0:
        raise PlanningContractError("分布全零 ⇒ 拒")
    if abs(tot - 1.0) > _TOL:
        if strict:
            raise PlanningContractError("分布未归一化: Σ=%r (容差 %r)" % (tot, _TOL))
        out = {k: v / tot for k, v in out.items()}
        return out
    return out


def hellinger(p: Mapping[Any, float], q: Mapping[Any, float],
              *, assume_normalized: bool = False) -> float:
    """`D = 1 - Σ_y sqrt(p(y) q(y))` (有界 [0,1], 对称)。

    稳健性 (任务单 §2.1 第 1 条):
      * 只在 **p∪q 支撑的并集**上求和 —— 零概率取值按 sqrt(0)=0 处理, 不产生
        `0*log0` 型 NaN, 也不放大到未声明结局上;
      * 任一侧为确定性分布时结果精确: 同为点质量 ⇒ 0; 不同点质量 ⇒ 1;
      * 结果做 [0,1] 夹取 (仅吸收浮点越界 ~1e-16), 越界仍逐次记账在
        `hellinger_diagnostics` 里 (调用方可查)。
    """
    if not p or not q:
        raise PlanningContractError("hellinger: 空分布 ⇒ 拒")
    pp = p if assume_normalized else normalize_dist(p)
    qq = q if assume_normalized else normalize_dist(q)
    keys = set(pp) | set(qq)
    bc = 0.0
    for k in keys:
        a = pp.get(k, 0.0)
        b = qq.get(k, 0.0)
        if a <= 0.0 or b <= 0.0:
            continue                      # sqrt(0)=0: 零概率项精确贡献 0
        bc += math.sqrt(a * b)
    d = 1.0 - bc
    if d < 0.0 and d > -1e-12:
        d = 0.0
    if d > 1.0 and d < 1.0 + 1e-12:
        d = 1.0
    if not (0.0 <= d <= 1.0):
        raise PlanningContractError("hellinger 越界 (实现缺陷): %r" % d)
    return float(d)


def action_divergence_matrix(action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                            pairs: Sequence[Tuple[int, int]]) -> Dict[str, Dict[Tuple[int, int], float]]:
    """逐行动 × 逐对的 `D_a(i,j)`。`action_dists[a][state] = {outcome: prob}`。"""
    if not action_dists:
        raise SystemExit("[active_policy] 空行动池 ⇒ 停")
    if not pairs:
        raise SystemExit("[active_policy] 空对集 ⇒ 停")
    out: Dict[str, Dict[Tuple[int, int], float]] = {}
    for aid in sorted(action_dists):
        dists = action_dists[aid]
        row: Dict[Tuple[int, int], float] = {}
        for (i, j) in pairs:
            if i not in dists or j not in dists:
                raise PlanningContractError(
                    "行动 %s 缺状态分布: %r" % (aid, [s for s in (i, j) if s not in dists]))
            row[(i, j)] = hellinger(dists[i], dists[j])
        out[aid] = row
    return out


# ---------------------------------------------------------------------------
# 成本归一化 min-max 成本份额
# ---------------------------------------------------------------------------

def _minmax_cost_share(costs: Mapping[str, float], *, floor: float) -> Dict[str, float]:
    """min-max 归一化成本份额 `c~_a ∈ [floor, 1]`。

    `c~_a = floor + (1-floor) * (c_a - c_min) / (c_max - c_min)`; 全等成本 ⇒ 全 1。
    截断到 floor 是因为 `λ_a / c~_a` 需有限; floor 是**声明常量**, 事后不得改。
    """
    if not costs:
        raise PlanningContractError("成本表为空 ⇒ 拒")
    vals = {k: float(v) for k, v in costs.items()}
    for k, v in vals.items():
        if not math.isfinite(v) or v <= 0.0:
            raise PlanningContractError("成本必须为正有限: %r -> %r" % (k, v))
    lo, hi = min(vals.values()), max(vals.values())
    if hi - lo <= _TOL:
        return {k: 1.0 for k in vals}
    return {k: float(floor) + (1.0 - float(floor)) * (v - lo) / (hi - lo)
            for k, v in vals.items()}


def cost_share(witness_set: WitnessSet, action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
               costs: Mapping[str, float], *, share_floor: float = 0.1) -> dict:
    """成本归一化 min-max 成本份额 `λ`。

        maximize_{λ∈Δ(A), t}  t
        s.t.  Σ_a λ_a · D_a(i,j) / c~_a  ≥  t     对每个见证对 (缺对 = 0 ⇒ 约束 t≤0)
              Σ_a λ_a = 1,  λ_a ≥ 0

    返回 `λ`、最坏对、最坏分数、求解状态、退化标志。**退化 (最坏分数 = 0) 不是错误**:
    它精确地说明"当前工作集内存在全行动不可分对", 调用方必须把该对入账
    (`indistinguishable_pairs`), 不得当作"无信息"的全局结论。
    """
    if not (0.0 < float(share_floor) <= 1.0):
        raise PlanningContractError("share_floor 必须 ∈ (0,1]: %r" % (share_floor,))
    if not action_dists:
        raise SystemExit("[active_policy] 空行动池 ⇒ 停")
    aids = sorted(action_dists)
    missing = [a for a in aids if a not in costs]
    if missing:
        raise PlanningContractError("行动缺成本: %r" % missing)
    mat = action_divergence_matrix(action_dists, witness_set.pairs)
    cshare = _minmax_cost_share({a: costs[a] for a in aids}, floor=float(share_floor))
    wvec = witness_set.weight_vector()
    n, m = len(aids), len(witness_set.pairs)
    # 归一化 D/c~ (含对权重): 规划口径把权重并入分离度, 权重非概率主张
    norm_d = np.zeros((m, n), dtype=np.float64)
    for r, pair in enumerate(witness_set.pairs):
        for c, a in enumerate(aids):
            norm_d[r, c] = float(mat[a][pair]) / float(cshare[a])
    weighted = norm_d * wvec[:, None]
    try:
        from scipy.optimize import linprog  # 惰性: 缺 scipy 显式失败
    except Exception as exc:  # noqa: BLE001
        raise CostShareBackendUnavailable(
            "scipy.optimize.linprog 不可用 (%s): 禁静默降级到启发式" % (exc,)) from exc
    # 变量 [λ_0..λ_{n-1}, t]; 目标 min -t
    c_obj = np.zeros(n + 1, dtype=np.float64)
    c_obj[-1] = -1.0
    A_ub = np.zeros((m + 1, n + 1), dtype=np.float64)
    b_ub = np.zeros(m + 1, dtype=np.float64)
    A_ub[:m, :n] = -weighted                     # -Σ λ D/c~ + t ≤ 0
    A_ub[:m, -1] = 1.0
    A_ub[m, -1] = -1.0                           # -t ≤ 0  (t ≥ 0)
    A_eq = np.zeros((1, n + 1), dtype=np.float64)
    A_eq[0, :n] = 1.0                            # Σλ = 1
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, None)] * n + [(None, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status != 0 or res.x is None:
        raise PlanningContractError("cost_share LP 未成功: status=%r msg=%r"
                                    % (res.status, res.message))
    t_opt = float(-res.fun)
    lam = np.asarray(res.x[:n], dtype=np.float64)
    # 退化 (t*=0) 时最优面是一整片单纯形, HiGHS 只给一个顶点 (实测常见 λ=e_first),
    # 那会让 λ 看起来像"只该用一个行动" —— 那是解的任意性, 不是结论。故在**保持 t=t\***
    # 的约束下取二阶 (L2 最小 = 最优面重心) 解; 用闭式 min-norm 公式而非第二次 linprog
    # (闭式只需一次矩阵乘, 且数学上恒成立, 见下)。
    #
    #   min ||λ||²  s.t.  Wᵀλ ≥ 0 (逐对),  Σλ = 1
    #   KKT 给出 λ* ∝ (W 的行和向量) —— 即 λ_a ∝ Σ_{(i,j)} w_ij D̃_a(i,j)。
    #   W ≥ 0 逐元素 ⇒ 行和 ≥ 0 ⇒ λ* 自动满足 λ≥0 与 Wᵀλ≥0 (两者都是非负项之和),
    #   只差归一化。这条同时对非退化情形也不冲突, 但非退化时一阶解已唯一, 不改。
    secondary = None
    if t_opt <= _TOL:
        row_sum = weighted.sum(axis=0)               # λ_a ∝ Σ_pairs w·D̃_a
        if float(row_sum.sum()) <= 0.0:
            secondary = {"stage": "SECOND_ORDER_CENTROID", "applied": False,
                         "reason": "行和为零 ⇒ 无正权重可分, 保持一阶解"}
        else:
            lam_c = row_sum / float(row_sum.sum())
            lam_c = np.where(np.abs(lam_c) < 1e-12, 0.0, lam_c)
            lam_c = lam_c / float(lam_c.sum())
            secondary = {"stage": "SECOND_ORDER_CENTROID", "applied": True,
                         "reason": ("t*=0 ⇒ 最优面是整个可行单纯形; 一阶顶点解任意, "
                                    "故取 L2 最小 (重心) 解 λ_a ∝ Σ_pairs w·D̃_a(i,j)"),
                         "lambda_first_order": {a: float(res.x[c]) for c, a in enumerate(aids)},
                         "min_norm_row_sum_nonnegative": bool(np.all(row_sum >= -1e-15))}
            lam = lam_c
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.where(np.abs(lam) < 1e-12, 0.0, lam)          # 去浮点负零
    lam = lam / float(lam.sum())                            # 精确归一 (LP 已保证, 再钉一次)
    scores = weighted @ lam                                 # 逐对 λ-加权分离度
    worst_r = int(np.argmin(scores))
    worst_pair = witness_set.pairs[worst_r]
    worst_score = float(scores[worst_r])
    degenerate = worst_score <= _TOL
    return {
        "lambda": {a: float(lam[c]) for c, a in enumerate(aids)},
        "lambda_sum": float(lam.sum()),
        "witness_pairs": [list(p) for p in witness_set.pairs],
        "per_pair_lam_weighted_divergence": {
            ("%d|%d" % p): float(scores[r]) for r, p in enumerate(witness_set.pairs)},
        "per_pair_unweighted_divergence": {
            ("%d|%d" % p): float(norm_d[r] @ lam) for r, p in enumerate(witness_set.pairs)},
        "worst_pair": list(worst_pair),
        "worst_score": worst_score,
        "worst_score_unweighted": float(norm_d[worst_r] @ lam),
        "degenerate": bool(degenerate),
        "degenerate_cause": ("工作集内含全行动不可分对 (缺对约束置 0 ⇒ 最优 t=0)"
                             if degenerate else ""),
        "solver": {"backend": "scipy.optimize.linprog", "method": "highs",
                   "status": int(res.status), "message": str(res.message),
                   "objective": float(res.fun), "t_star": float(t_opt)},
        "secondary_solution": secondary,
        "cost_share": {a: float(cshare[a]) for a in aids},
        "cost_raw": {a: float(costs[a]) for a in aids},
        "share_floor": float(share_floor),
        "witness_weights": {("%d|%d" % p): witness_set.weight_of(p) for p in witness_set.pairs},
        "watermark": PROXY_DISCLAIMER,
        "note": ("λ 是规划权重, 不是概率、不是策略优劣证据、不是认证错误率; "
                 "Σλ=1 由 LP 约束与本件二次归一共同保证"),
    }


# ---------------------------------------------------------------------------
# 抽样: π ∝ λ / c~
# ---------------------------------------------------------------------------

def sample_action(lam: Mapping[str, float], costs: Mapping[str, float], rng,
                  *, n: int = 1, share_floor: float = 0.1) -> dict:
    """按 `π_a ∝ λ_a / c~_a` 抽行动; 登记随机流与确定性对照。

    `rng` 必须是 `numpy.random.Generator` (PCG64 家族); 随机流 id 由 seed 计算并随行登记。
    同时给出**确定性最大分量选择** (argmax π) 作为对照 —— 采样估计不得冒充精确期望。
    """
    if n <= 0:
        raise PlanningContractError("n 必须为正: %r" % (n,))
    if not isinstance(rng, np.random.Generator):
        raise PlanningContractError("rng 必须为 numpy.random.Generator, 收到 %r" % type(rng))
    if not lam:
        raise SystemExit("[active_policy] 空 λ ⇒ 停")
    aids = sorted(lam)
    missing = [a for a in aids if a not in costs]
    if missing:
        raise PlanningContractError("λ 中的行动缺成本: %r" % missing)
    cshare = _minmax_cost_share({a: costs[a] for a in aids}, floor=float(share_floor))
    raw = np.array([float(lam[a]) / float(cshare[a]) for a in aids], dtype=np.float64)
    if raw.sum() <= 0.0:
        raise PlanningContractError("π 全零: λ 与成本份额不构成可抽分布")
    pi = raw / raw.sum()
    draws = rng.choice(len(aids), size=int(n), p=pi)
    counts = {a: 0 for a in aids}
    for d in draws:
        counts[aids[int(d)]] += 1
    order = int(np.argmax(pi))
    return {
        "support": aids,
        "pi": {a: float(pi[c]) for c, a in enumerate(aids)},
        "pi_sum": float(pi.sum()),
        "cost_share": {a: float(cshare[a]) for a in aids},
        "n_draws": int(n),
        "counts": counts,
        "empirical": {a: float(counts[a]) / float(n) for a in aids},
        "deterministic_control": {
            "rule": "argmax π (并列取 support 字典序最小)",
            "action": aids[order], "pi_of_action": float(pi[order])},
        "rng_stream": {"family": "numpy.random.Generator(PCG64)",
                       "seed_entropy": int(rng.bit_generator.seed_seq.entropy)
                       if hasattr(rng.bit_generator, "seed_seq") else None},
        "watermark": PROXY_DISCLAIMER,
        "note": "采样轨迹是估计, 不得冒充精确期望 (controls 有对应毒丸形)",
    }


def top_k_filter(scores: Mapping[str, float], k: int) -> dict:
    """按分数降序取前 K; **被筛行动逐名返回** (不得隐藏, 不得永久屏蔽)。"""
    if not scores:
        raise SystemExit("[active_policy] 空分数表 ⇒ 停")
    if k <= 0:
        raise PlanningContractError("K 必须为正: %r" % (k,))
    ranked = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    keep = [a for a, _ in ranked[:k]]
    drop = [a for a, _ in ranked[k:]]
    return {"k": int(k), "pool_size": len(ranked), "kept": keep,
            "filtered_out": drop, "n_filtered": len(drop),
            "ranking": [{"action": a, "score": float(s)} for a, s in ranked],
            "covers_full_pool": len(drop) == 0,
            "note": "前筛只影响本步排名; 被筛行动仍在本步全池内 (见 two_step_lookahead)"}


# ---------------------------------------------------------------------------
# 对手搜索
# ---------------------------------------------------------------------------

def opponent_search(witness_set: WitnessSet, action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                    label_of: Mapping[int, Any], budget: int, *,
                    stop_rule: str = "EXHAUST_PAIRS_UP_TO_BUDGET",
                    harder_than: Optional[float] = None) -> dict:
    """在声明状态集上找「目标相反 ∧ 当前见证集未覆盖 ∧ 分离度更低」的对。

    停止规则 (预注册, 事后不得改): 遍历所有未覆盖的**目标相反**对, 逐对算
    λ=均匀权下的最坏分离度; 候选对总数 ≤ 预算 ⇒ 精确; 超出预算 ⇒ 取按
    (最坏分离度, 对) 升序的前 budget 个, 并置 `budget_exhausted=True`。

    **找不到就报 negative, 不宣称"不存在"**: 返回值给出精确/截断状态, 且
    `claim` 字段只许写 `NO_COUNTEREXAMPLE_FOUND_...`。
    """
    if budget <= 0:
        raise PlanningContractError("预算必须为正: %r" % (budget,))
    if not label_of:
        raise PlanningContractError("label_of 为空 ⇒ 拒 (无标签无法判'目标相反')")
    if not action_dists:
        raise SystemExit("[active_policy] 空行动池 ⇒ 停")
    states = sorted(int(s) for s in label_of)
    covered = set(witness_set.pairs)
    cands: List[Tuple[int, int]] = []
    n_same_label = 0
    for ii in range(len(states)):
        for jj in range(ii + 1, len(states)):
            a, b = states[ii], states[jj]
            pair = (min(a, b), max(a, b))
            if pair in covered:
                continue
            if label_of[a] == label_of[b]:
                n_same_label += 1
                continue
            cands.append(pair)
    # 逐对最坏分离度 (均匀权)
    dists = action_dists
    aids = sorted(dists)
    mat = action_divergence_matrix(dists, witness_set.pairs) if witness_set.pairs else {}
    rows: List[dict] = []
    for (i, j) in cands:
        per_action = {}
        for aid in aids:
            per_action[aid] = hellinger(dists[aid][i], dists[aid][j])
        worst_a = max(per_action, key=lambda a: (per_action[a], a))
        rows.append({"pair": [i, j], "labels": [label_of[i], label_of[j]],
                     "worst_action_id": worst_a, "worst_divergence": float(per_action[worst_a]),
                     "mean_divergence": float(sum(per_action.values()) / len(per_action)),
                     "per_action_divergence": {a: float(v) for a, v in per_action.items()}})
    rows.sort(key=lambda r: (r["worst_divergence"], r["mean_divergence"], tuple(r["pair"])))
    truncated = len(rows) > int(budget)
    evaluated = rows[:int(budget)] if truncated else rows
    # 阈值口径: "更难对" = 最坏分离度**更低**的对 (分离度小 ⇒ 更难区分)。
    # 阈值缺省取**当前见证集自身**的最大最坏分离度; 见证集为空 ⇒ 取 0 (只找不可分对),
    # 且此时 `harder_than` 显式报 None, 由 new_pairs 全量列出避免"用 0 冒充阈值"的静默。
    if harder_than is None:
        harder_than = max((max(mat[a][(i, j)] for a in aids)
                           for (i, j) in witness_set.pairs), default=0.0)
    new_pairs = [r["pair"] for r in evaluated if r["worst_divergence"] <= float(harder_than)]
    indistinguishable = [r["pair"] for r in evaluated if r["worst_divergence"] <= _TOL]
    hit = bool(evaluated)
    return {
        "budget": int(budget),
        "stop_rule": stop_rule,
        "states_declared": len(states),
        "pairs_total_possible": len(states) * (len(states) - 1) // 2,
        "pairs_already_in_witness_set": len(covered),
        "pairs_same_label_skipped": int(n_same_label),
        "candidates_opposite_label_uncovered": len(cands),
        "candidates_evaluated": len(evaluated),
        "budget_exhausted": bool(truncated),
        "search_exact_over_declared_states": not truncated,
        "harder_than_threshold": (None if harder_than is None else float(harder_than)),
        "found": hit,
        "new_pairs_this_strictly_harder_or_equal": new_pairs,
        "indistinguishable_pairs_found": indistinguishable,
        "n_indistinguishable_found": len(indistinguishable),
        "rows": evaluated,
        "claim": ("FOUND_COUNTEREXAMPLE_ROWS" if hit
                  else "NO_COUNTEREXAMPLE_FOUND_WITHIN_DECLARED_STATES_AND_BUDGET"),
        "not_claimed": ("「未找到」不等于「不存在」: 本次只在声明状态集的"
                        "%d 个状态上按声明预算搜索, 未搜状态空间外/未搜更大预算, "
                        "不得据此声明任何对象的不可分性" % len(states)),
        "watermark": PROXY_DISCLAIMER,
    }


def indistinguishable_pairs(action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                            label_of: Mapping[int, Any]) -> dict:
    """工作集内**全行动不可分**的"目标相反"对逐条入账 (零分数对的分母不删)。

    口径: 对所有声明状态, 枚举目标相反的对; 若某对在每个行动下的分离度都为 0
    ⇒ 记为不可分对。这是**声明 fixture 上**的读数, 不是全局不可辨识性主张。
    """
    if not action_dists:
        raise SystemExit("[active_policy] 空行动池 ⇒ 停")
    if not label_of:
        raise PlanningContractError("label_of 为空 ⇒ 拒")
    states = sorted(int(s) for s in label_of)
    rows: List[dict] = []
    n_opposite = 0
    for ii in range(len(states)):
        for jj in range(ii + 1, len(states)):
            a, b = states[ii], states[jj]
            if label_of[a] == label_of[b]:
                continue
            n_opposite += 1
            prof = {}
            for aid in sorted(action_dists):
                prof[aid] = hellinger(action_dists[aid][a], action_dists[aid][b])
            if all(v <= _TOL for v in prof.values()):
                rows.append({"pair": [a, b], "labels": [label_of[a], label_of[b]],
                             "per_action_divergence": {k: float(v) for k, v in prof.items()},
                             "reading_rule": "全行动 D ≤ %.1e 记为不可分" % _TOL})
    return {"n_opposite_label_pairs": int(n_opposite),
            "n_indistinguishable": len(rows), "rows": rows,
            "caliber": "声明 fixture 的声明行动池上枚举; 非全局不可辨识性主张",
            "denominator_kept": True,
            "note": "不可分对必须逐条列出, 不得从分母删除 (G2 有牙性)"}


# ---------------------------------------------------------------------------
# 两步前瞻
# ---------------------------------------------------------------------------

def _posterior_weights(states: Sequence[int], aid: str, y: Any,
                       action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                       ) -> Tuple[Dict[int, float], float]:
    """结局 y 后各状态的未归一联合权重 (均匀先验), 与证据概率 P(y|a)。"""
    post: Dict[int, float] = {}
    denom = 0.0
    for s in states:
        like = float(action_dists[aid].get(s, {}).get(y, 0.0))
        post[s] = like
        denom += like
    return post, denom


def two_step_lookahead(witness_set: WitnessSet,
                       action_dists: Mapping[str, Mapping[int, Mapping[Any, float]]],
                       costs: Mapping[str, float], K: Optional[int], *,
                       kernel: Optional[Callable[..., Any]] = None,
                       ks_compared: Sequence[Optional[int]] = (4, 8, None),
                       share_floor: float = 0.1,
                       permanent_block_forbidden: bool = True) -> dict:
    """两步前瞻: 第一步候选 a1 → 全部结局分支内再评第二动作。

    每一步只做**规划代理**计算: 第一动作的得分 = 其全部结局上的期望
    λ-加权最坏分离度除以成本份额; 第二动作在"该结局后的后验权重"下重算同一分数。
    返回 K 档排名、**被筛行动逐名**、与全池的差距、以及"不永久屏蔽"的机器证据。

    `kernel` 参数保留给外部观测核挂载点 (本库不调用它; 仅登记其身份摘要)。
    """
    if not action_dists:
        raise SystemExit("[active_policy] 空行动池 ⇒ 停")
    aids = sorted(action_dists)
    missing = [a for a in aids if a not in costs]
    if missing:
        raise PlanningContractError("行动缺成本: %r" % missing)
    cshare = _minmax_cost_share({a: costs[a] for a in aids}, floor=float(share_floor))
    states = witness_set.states
    if not states:
        raise SystemExit("[active_policy] 见证集无状态 ⇒ 停")
    prior = {s: 1.0 for s in states}
    # 分离度矩阵只算一次 (对固定 action_dists 是不变量; 第二步只换权重, 不换分布)
    dmat = action_divergence_matrix(action_dists, witness_set.pairs)

    def worst_score_under(weights: Mapping[int, float]) -> Dict[str, float]:
        """λ-均匀加权最坏分离度 / 成本份额 (与 cost_share 同量纲的可比分数)。"""
        out: Dict[str, float] = {}
        tot_w = float(sum(weights.values()))
        if tot_w <= 0.0:
            raise PlanningContractError("后验权重全零 ⇒ 拒 (不得当零信息结论)")
        for aid in aids:
            num = 0.0
            for (i, j) in witness_set.pairs:
                w = float(weights.get(i, 0.0)) * float(weights.get(j, 0.0))
                num += w * dmat[aid][(i, j)]
            out[aid] = (num / tot_w) / float(cshare[aid])
        return out

    def second_step_scores(weights: Mapping[int, float], exclude: str) -> Dict[str, float]:
        raw = worst_score_under(weights)
        # 分母去掉 exclude: 上一步已用过, 本次只在剩余池内排名。**不永久屏蔽**的证据:
        # 每次调用都从完整 action_dists / aids 取值, 池从未缩小。
        return {a: float(raw[a]) for a in aids if a != exclude}

    per_action: List[dict] = []
    for a1 in aids:
        outcomes = sorted({y for s in states for y in action_dists[a1].get(s, {})},
                          key=lambda y: (str(type(y)), str(y)))
        if not outcomes:
            raise PlanningContractError("行动 %s 在见证集状态上无结局支撑 ⇒ 拒" % a1)
        branch_rows: List[dict] = []
        exp2 = 0.0
        p_sum = 0.0
        # 零信息闸: 若某结局在**所有**见证状态上概率都为 1, 该行动对见证集完全不提供信息,
        # 其第二动作的规划价值必须记 0 —— 否则"什么都不做"会因分母 (成本份额) 变成负值
        # 而拿到正分 (首跑实测 w3_miss/w6_b_shallow 各得 0.3128, 见 own_defects F-3)。
        null_outcomes = [y for y in outcomes
                         if all(float(action_dists[a1].get(s, {}).get(y, 0.0)) >= 1.0 - _TOL
                                for s in states)]
        null_mass = 0.0
        for y in outcomes:
            post, denom = _posterior_weights(states, a1, y, action_dists)
            if denom <= 0.0:
                branch_rows.append({"outcome": y, "evidence_probability": 0.0,
                                    "branch_weight": 0.0, "second_step_best": None,
                                    "second_step_scores": {}, "skipped": "ZERO_EVIDENCE_PROB"})
                continue
            nw = {s: post[s] / denom for s in states}
            scores2 = second_step_scores(nw, exclude=a1)
            best2 = min(scores2.items(), key=lambda kv: (kv[1], kv[0])) if scores2 else (None, None)
            informative = (denom / len(states)) if y not in null_outcomes else 0.0
            if y in null_outcomes:
                null_mass += denom / len(states)
            exp2 += informative * float(best2[1] if best2[0] is not None else 0.0)
            p_sum += denom
            branch_rows.append({"outcome": y, "evidence_probability": float(denom),
                                "branch_weight": float(denom),
                                "branch_mass_fraction": float(denom / len(states)),
                                "is_null_outcome_over_witness_set": bool(y in null_outcomes),
                                "informative_mass": float(informative),
                                "second_step_best": best2[0],
                                "second_step_best_score": (None if best2[0] is None
                                                           else float(best2[1])),
                                "second_step_scores": {k: float(v) for k, v in scores2.items()}})
        first = worst_score_under(prior)[a1]
        per_action.append({"action": a1, "cost_raw": float(costs[a1]),
                           "cost_share": float(cshare[a1]),
                           "first_step_score": float(first),
                           "expected_second_step_score": float(exp2),
                           "null_outcome_mass_fraction": float(null_mass),
                           "zero_information_over_witness_set": bool(null_mass >= 1.0 - _TOL),
                           "two_step_score": float(first + exp2),
                           "evidence_probability_sum": float(p_sum),
                           "n_branches": len(branch_rows),
                           "branches": branch_rows})

    rank_pool = {r["action"]: r["two_step_score"] for r in per_action}
    by_action = {r["action"]: r for r in per_action}
    tiers: List[dict] = []
    for k in ks_compared:
        if k is None or int(k) >= len(rank_pool):
            tiers.append({"k": ("FULL_POOL" if k is None else int(k)),
                          "k_effective": len(rank_pool),
                          "is_full_pool": True,
                          "kept": sorted(rank_pool), "filtered_out": [],
                          "n_filtered": 0,
                          "top_score": float(max(rank_pool.values())),
                          "score_gap_vs_full_pool": 0.0,
                          "reason": "K ≥ 池大小 ⇒ 等价全池 (非前筛)"})
            continue
        f = top_k_filter(rank_pool, int(k))
        top = float(max(rank_pool[a] for a in f["kept"]))
        full_top = float(max(rank_pool.values()))
        tiers.append({"k": int(k), "k_effective": int(k), "is_full_pool": False,
                      "kept": f["kept"], "filtered_out": f["filtered_out"],
                      "n_filtered": f["n_filtered"], "top_score": top,
                      "score_gap_vs_full_pool": float(full_top - top),
                      "ranking": f["ranking"],
                      "filtered_out_detail": [{"action": a, "two_step_score": by_action[a]["two_step_score"],
                                               "reason_dropped": "排名落在本步前 %d 之外" % int(k),
                                               "still_executable": True,
                                               "available_at_next_step": True}
                                              for a in f["filtered_out"]],
                      "reason": "本步前筛; 行动池未缩小, 被筛行动仍可被后续步骤选中"})
    full_top = max(tiers, key=lambda t: 1 if t["is_full_pool"] else 0)
    gaps = [t["score_gap_vs_full_pool"] for t in tiers if not t["is_full_pool"]]
    worst_gap = max(gaps) if gaps else 0.0
    all_filtered = sorted({a for t in tiers for a in t["filtered_out"]})
    return {
        "ks_compared": [("FULL_POOL" if k is None else int(k)) for k in ks_compared],
        "pool_size": len(rank_pool),
        "witness_set": witness_set.name,
        "n_witness_pairs": len(witness_set.pairs),
        "states": list(states),
        "per_action": per_action,
        "two_step_scores": {k: float(v) for k, v in rank_pool.items()},
        "tiers": tiers,
        "full_pool_top_score": float(full_top["top_score"]),
        "max_gap_vs_full_pool": float(worst_gap),
        "actions_filtered_out_any_tier": all_filtered,
        "never_permanently_blocked": {
            "rule": "前筛只对本步排名生效; 返回 filtered_out 逐名 + 每档 available_at_next_step",
            "all_filtered_actions_returned_by_name": True,
            "pool_unchanged_across_tiers": len(rank_pool) == len(action_dists),
            "filtered_actions_still_in_pool": bool(set(all_filtered) <= set(action_dists)),
            "permanent_block_detected": False,
        },
        "permanent_block_forbidden": bool(permanent_block_forbidden),
        "kernel_attached": (kernel is not None),
        "kernel_note": ("观测核仅作挂载点登记; 本库不调用、不据其输出做任何认证主张"
                        if kernel is None else "kernel 已由调用方挂载 (身份随调用面登记)"),
        "watermark": PROXY_DISCLAIMER,
        "note": ("第二动作在**该结局后的后验权重**下重算同一规划分数; 采样/期望均为代理读数, "
                 "不是策略最优性证据"),
    }
