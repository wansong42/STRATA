# -*- coding: utf-8 -*-
"""S1 · 累积证据集合与 e-process（本单唯一新核心）。

判据权威 = 本线 ``prereg_s1.json`` 与本单 §2.1/§4；观测语义唯一来源 = S0 已冻结的
``results/s0s1_minimal_vertical_chain/s0/observation_spec.json`` 与
``src/fractureflow/observation_kernel.py``（kernel_id ``K-30041ee8cade-9b14eeea67f7``）。
本件不改 ``identification.py`` / ``observation_kernel.py`` 任一字节。

强制语义（与任务单 §2.1 逐条对应）:
  1. ``E_0(theta)=1``；``E_t(theta)=prod_{s<=t} q_s(Y_s|H_{s-1},a_s)/p_theta(Y_s|H_{s-1},a_s)``；
     ``S_t={theta: max_{u<=t} E_u(theta) < 1/alpha_seq}`` —— **必须保存历史最大证据**。
  2. ``q_t`` 是**看见本次观测之前**确定的正规化条件预测密度；``mode="posterior"`` 按历史
     贝叶斯权重混合，``mode="uniform"`` 用固定均匀混合（消融臂）。两者都可预测、可归一化，
     ``normalization_residual`` 用 Fraction 精确检查 ``sum_y q_t(y) == 1``。
  3. 零概率：某候选对已发生记录赋 0 ⇒ 该候选排除（``E=INF``）；**全部候选都赋 0**
     ⇒ ``MODEL_CONFLICT``，禁止 0/0 产生证书。
  4. 首个观测必须入账（``EProcess`` 在 ``t=1`` 就对第一条记录更新，不存在"先裁状态"路径）。
  5. 行动种子/历史条件进入 ``H``：``q_t`` 用 ``H_{t-1}`` 的后验权重；``shared_bias=false``、
     ``repeat_observations=false`` 由 prereg 声明，本件不默认独立以外的任何结构。
  6. 数值：全程 ``fractions.Fraction`` 精确算术；``log_rows`` 另记浮点 ``log q`` / ``log p`` /
     ``max log E`` 与归一化残差，供审计对照，不参与判定。
  7. 目标投影 ``Gamma_t={g_Q(theta): theta in S_t}``；只有 ``S_t`` 非空且 ``|Gamma_t|=1``
     才发单值证书；``S_t`` 空 ⇒ ``EMPTY``（= MODEL_CONFLICT）；``|Gamma_t|=2`` ⇒ 双值/弃权。
  8. 模型身份范围只做 ``KNOWN_SYNTHETIC``；未知 nuisance / 密度上包络 / 校准域预算不在本件。
"""
from __future__ import annotations

import math
from fractions import Fraction

from .observation_kernel import EMPTY_OUTCOME, KernelScopeError

__all__ = [
    "ModelConflict", "SearchUnresolved", "INF", "bits_of", "record_key",
    "record_from_counts", "PredictiveMixture", "EProcess",
    "STATUS_CERTIFIED", "STATUS_DUAL", "STATUS_EMPTY", "STATUS_CONFLICT",
    "STATUS_SEARCH_UNRESOLVED",
]

INF = "INF"

STATUS_CERTIFIED = "CERTIFIED"
STATUS_DUAL = "DUAL"
STATUS_EMPTY = "EMPTY"
STATUS_CONFLICT = "MODEL_CONFLICT"
STATUS_SEARCH_UNRESOLVED = "SEARCH_UNRESOLVED"


class ModelConflict(RuntimeError):
    """全部候选状态对已发生记录赋 0 概率（q_t=0）：禁止 0/0 发证书。"""


class SearchUnresolved(RuntimeError):
    """声明枚举预算内未解决（不猜、不删、不截断）。"""


def bits_of(state: int, d: int) -> tuple:
    return tuple((int(state) >> i) & 1 for i in range(d))


def record_key(public_record: dict) -> tuple:
    """公开记录的机读键：只含公开字段，不含任何真值/候选标识。"""
    return (str(public_record.get("design_id")),
            int(public_record.get("well_id", -1)),
            tuple((int(c), int(n)) for c, n in public_record.get("counts", [])))


def record_from_counts(kernel, action_spec: dict, counts: dict) -> dict:
    """由计数向量构造公开记录（与生成侧同一 make_public_record 码路）。"""
    action = kernel._parse_action(action_spec)
    return kernel.make_public_record(action, counts)


class PredictiveMixture:
    """``q_t(y | H_{t-1}, a_t)``：看见本次观测之前确定的混合预测密度（精确有理数）。

    Parameters
    ----------
    states : 候选状态整数序列（本单 0..15，全部列出，不删）
    kernel : GridDetectionKernel
    action_spec : 默认行动（``log_prob``/``observe`` 可逐次覆盖）
    prior : {state: Fraction}；缺省均匀 1/|states|
    mode : ``posterior``（主读数）| ``uniform``（消融臂）
    tier : 噪声档 N0/N1/N2
    """

    def __init__(self, states, kernel, action_spec, prior=None, mode="posterior",
                 tier="N1", cache=None):
        self.states = tuple(int(s) for s in states)
        if len(set(self.states)) != len(self.states):
            raise ValueError("states 含重复项")
        self.kernel = kernel
        self.action_spec = action_spec
        self.mode = mode
        if mode not in ("posterior", "uniform"):
            raise ValueError("mode 必须为 posterior/uniform: %r" % (mode,))
        self.tier = tier
        if prior is None:
            prior = {s: Fraction(1, len(self.states)) for s in self.states}
        self.prior = {int(k): Fraction(v) for k, v in prior.items()}
        if set(self.prior) != set(self.states):
            raise ValueError("prior 键集与 states 不一致")
        if sum(self.prior.values()) != 1:
            raise ValueError("prior 未归一化: %s" % (sum(self.prior.values()),))
        self.weights = dict(self.prior)
        self.history = []
        self._pcache = {} if cache is None else cache
        self._d = len(self.kernel.candidate_geometry())
        if any(s < 0 or s >= (1 << self._d) for s in self.states):
            raise ValueError("states 超出 d=%d 的位宽" % self._d)

    # -- 似然（精确）-------------------------------------------------------
    def state_probability(self, public_record, state, action_spec=None) -> Fraction:
        a = action_spec if action_spec is not None else self.action_spec
        key = (str(a["action_id"]), int(state), record_key(public_record))
        if key not in self._pcache:
            self._pcache[key] = self.kernel.record_probability_exact(
                public_record, bits_of(state, self._d), a, tier=self.tier)
        return self._pcache[key]

    def _weights(self) -> dict:
        if self.mode == "posterior":
            return self.weights
        return {s: Fraction(1, len(self.states)) for s in self.states}

    def prob(self, public_record, action_spec=None) -> Fraction:
        """q_t(y|H_{t-1},a_t) 的精确值（未归一化即为实现缺陷）。"""
        w = self._weights()
        q = Fraction(0, 1)
        for s in self.states:
            ws = w[s]
            if ws == 0:
                continue
            q += ws * self.state_probability(public_record, s, action_spec)
        return q

    def log_prob(self, public_record, action_spec=None) -> Fraction:
        """接口名保留（任务单 §2.1）：返回精确 q_t（非对数），浮点对数见 log_prob_float。"""
        return self.prob(public_record, action_spec)

    def log_prob_float(self, public_record, action_spec=None):
        q = self.prob(public_record, action_spec)
        if q == 0:
            return -math.inf
        return math.log(q.numerator) - math.log(q.denominator)

    def observe(self, public_record, action_spec=None) -> dict:
        """看到观测之后更新后验权重（H_t = H_{t-1} ∪ {(a_t, y_t)}）。"""
        a = action_spec if action_spec is not None else self.action_spec
        new = {}
        for s in self.states:
            new[s] = self.weights[s] * self.state_probability(public_record, s, a)
        tot = sum(new.values())
        if tot == 0:
            raise ModelConflict(
                "全部候选状态对记录 %r 赋 0 概率 (q_t=0): MODEL_CONFLICT"
                % (record_key(public_record),))
        self.weights = {s: v / tot for s, v in new.items()}
        self.history.append({"action_id": str(a["action_id"]),
                             "record": record_key(public_record),
                             "posterior": {str(s): str(v) for s, v in self.weights.items()}})
        return self.weights

    # -- 归一化自检 --------------------------------------------------------
    def support_records(self, action_spec=None) -> list:
        """该行动下混合支撑（= 全盘在位状态的支撑，是任何子状态支撑的超集）。"""
        a = action_spec if action_spec is not None else self.action_spec
        d = max(self.states).bit_length()
        full = (1 << d) - 1
        pmf = self.kernel.count_pmf(bits_of(full, d), a, tier=self.tier)
        out = []
        for vec, p in pmf.items():
            if p == 0:
                continue
            counts = {i: int(c) for i, c in enumerate(vec) if c}
            out.append(record_from_counts(self.kernel, a, counts))
        return out

    def normalization_residual(self, action_spec=None) -> dict:
        a = action_spec if action_spec is not None else self.action_spec
        recs = self.support_records(a)
        total = sum((self.prob(r, a) for r in recs), Fraction(0, 1))
        return {"action_id": str(a["action_id"]), "n_support": len(recs),
                "sum_q": str(total), "residual": str(total - 1),
                "exact_one": bool(total == 1)}


class EProcess:
    """时间一致置信集合 ``S_t`` 与目标投影 ``Gamma_t``（历史最大证据）。"""

    def __init__(self, states, labels, alpha_seq, mixture, kernel):
        self.states = tuple(int(s) for s in states)
        self.labels = {int(k): int(v) for k, v in labels.items()}
        if set(self.labels) != set(self.states):
            raise ValueError("labels 键集与 states 不一致")
        self.alpha_seq = Fraction(alpha_seq)
        if not (0 < self.alpha_seq < 1):
            raise ValueError("alpha_seq 非法: %r" % (alpha_seq,))
        self.threshold = 1 / self.alpha_seq
        self.mixture = mixture
        self.kernel = kernel
        self.t = 0
        self.E = {s: Fraction(1, 1) for s in self.states}
        self.Emax = {s: Fraction(1, 1) for s in self.states}
        self.excluded_at = {}
        self.records = []
        self.log_rows = []
        self.conflict = False
        self.conflict_at = None

    # -- 核心更新 ----------------------------------------------------------
    def update(self, action_spec: dict, public_record: dict) -> dict:
        q = self.mixture.prob(public_record, action_spec)   # 先预测（H_{t-1}）
        if q == 0:
            self.conflict = True
            self.conflict_at = self.t + 1
            self.log_rows.append({"t": self.t + 1, "action_id": str(action_spec["action_id"]),
                                  "q": "0", "status": STATUS_CONFLICT,
                                  "note": "全部候选赋 0 概率: 禁 0/0 发证书"})
            self.records.append((str(action_spec["action_id"]), record_key(public_record)))
            return {"q": Fraction(0, 1), "status": STATUS_CONFLICT}
        self.t += 1
        row = {"t": self.t, "action_id": str(action_spec["action_id"]),
               "record": record_key(public_record), "q": str(q),
               "log_q": (math.log(q.numerator) - math.log(q.denominator))}
        for s in self.states:
            if s in self.excluded_at:
                row.setdefault("skipped_excluded", []).append(s)
                continue
            p = self.mixture.state_probability(public_record, s, action_spec)
            if p == 0:
                self.E[s] = INF
                self.Emax[s] = INF
                self.excluded_at[s] = self.t
                row.setdefault("zero_probability_states", []).append(s)
                continue
            e_new = self.E[s] * q / p
            self.E[s] = e_new
            if e_new > self.Emax[s]:
                self.Emax[s] = e_new
        # 目标投影（历史最大证据）
        S = self.state_set()
        G = self.target_set()
        max_log_e = {}
        for s in self.states:
            v = self.Emax[s]
            if v == INF:
                max_log_e[str(s)] = None
            else:
                max_log_e[str(s)] = (math.log(v.numerator) - math.log(v.denominator))
        row.update({"n_S": len(S), "S": list(S), "Gamma": list(G),
                    "status": self.status(), "max_log_E": max_log_e})
        self.log_rows.append(row)
        self.records.append((str(action_spec["action_id"]), record_key(public_record)))
        self.mixture.observe(public_record, action_spec)     # 更新 H_t
        return {"q": q, "status": self.status(), "S": S, "Gamma": G}

    # -- 读数 --------------------------------------------------------------
    def state_set(self) -> tuple:
        """S_t = {theta: max_{u<=t} E_u(theta) < 1/alpha_seq}。"""
        out = []
        for s in self.states:
            if s in self.excluded_at:
                continue
            if self.Emax[s] == INF:
                continue
            if self.Emax[s] < self.threshold:
                out.append(s)
        return tuple(out)

    def target_set(self) -> tuple:
        return tuple(sorted({self.labels[s] for s in self.state_set()}))

    def status(self) -> str:
        if self.conflict:
            return STATUS_CONFLICT
        S = self.state_set()
        if not S:
            return STATUS_EMPTY
        G = self.target_set()
        if len(G) == 1:
            return STATUS_CERTIFIED
        if len(G) == 2:
            return STATUS_DUAL
        return STATUS_SEARCH_UNRESOLVED

    def singleton_value(self):
        G = self.target_set()
        return G[0] if (len(G) == 1 and self.state_set()) else None

    def ledger(self) -> dict:
        return {"alpha_seq": str(self.alpha_seq), "threshold": str(self.threshold),
                "T": self.t, "conflict": self.conflict, "conflict_at": self.conflict_at,
                "excluded_at": {str(k): int(v) for k, v in self.excluded_at.items()},
                "rows": self.log_rows,
                "final_S": list(self.state_set()),
                "final_Gamma": list(self.target_set()),
                "final_status": self.status()}
