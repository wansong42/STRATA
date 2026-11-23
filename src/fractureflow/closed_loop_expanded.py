# -*- coding: utf-8 -*-
"""S3-A · 扩展行动池（S1-X 7 井 / T=6 观测律）上的含噪闭环（本单唯一新核心模块）。

判据权威 = `results/s3a_expanded_closed_loop/prereg_s3a.json` 与任务单
`task_order_S3_A_expanded_closed_loop_20261018.md`（sha12 ``f621a14c5b3b``）§2.1/§4。
本件**只读**消费四个既有模块，不改 ``closed_loop.py`` / ``active_policy.py`` /
``statistical_certificate.py`` / ``sequential_evidence.py`` 任一字节：

* 观测核与行动池身份 = S1-X ``observation_kernel.py``（code sha12 ``9b14eeea67f7``）
  + ``results/s1x_action_expansion/observation_spec_s1x.json``（spec sha12 ``d7d914514865``）
  ⇒ kernel_id ``K-d7d914514865-9b14eeea67f7``（灌入即复核，STALE_KERNEL 直接抛）。
* 含噪证书 = S2-N ``statistical_certificate``（``run_eprocess`` /
  ``certify_with_statistics`` / ``verify_statistical_certificate``；α 口径 ``3/40``）。
* 规划库 = S3-P ``active_policy``（见证集 / Hellinger 成本份额 / 对手搜索 / 两步前瞻）
  与 ``active_query``（EC² 代理 + 两步前瞻）。

强制语义（与任务单 §2.1 逐条对应）:

1. **行动池身份**：7 井（``XS1,XS1M,XS2,XS3,XS4,XS5,XS6``）与 T=6 观测律身份；
   唯一来源 = S1-X 冻结件；本件不新增/改核。``XS1``/``XS1M`` 是重复观测对
   （逐盘结局表三档全等），在本件里记 ``duplicate_pair_both_executed`` 与
   ``T_by_signature=6``，不得按井数计 7 个独立行动。
2. **真闭环**：第 t 步行动只由 ``H_{t-1}``（公开记录历史、成本账、公开可算的
   统计可能集）决定；``choose_action_expanded`` 只吃 ``PublicStateExpanded``，
   真值只进 ``ExpandedWorld.observe``（噪声流单列，声明种子）。
3. **分支口径**：行动级分支 = 同一公开历史下的两个不同**可发生结局**（该步行动在
   当前混合预测分布 q_t 下的正概率记录）导向不同的下一步行动；决策级分支 =
   「下一步行动或廉价公开终末谓词」的变化。主判 = 行动级；决策级只作辅助读数。
4. **证书**：每步（t≥1）对当前公开历史调用 S2-N ``certify_with_statistics``
   （domain=ALL_STATES）；发出的每张证书逐张过 ``verify_statistical_certificate``；
   空集/混合/冲突不发证书（逐条记 reason）。t=0 不尝试（S2-N 协议要求非空公开历史）。
5. **噪声**：主读数 N1（``noise=N1`` 全账）；N2 只作声明子模型（缩短序列），
   子模型读数不得外推。
6. **基线**：``NEW`` / ``EC2_TWO_STEP`` / ``RANDOM`` / ``SPACE_FILL`` / ``ORIENTATION``
   同预算/同 α/同世界/同种子；全结局入账（六类；不删）。
7. **主张**：只报 ``DEVELOPMENT_ONLY``；不写普遍优越/成本优势/真实补测收益。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random as _random
import time
from fractions import Fraction

import numpy as np

from . import active_policy as ap
from . import active_query as aq
from . import certificate_solver as cs
from . import finite_scope as fs
from . import identification as ident
from . import statistical_certificate as sc
from .observation_kernel import GridDetectionKernel, kernel_identity, sha12_file
from .sequential_evidence import PredictiveMixture, bits_of, record_key

__all__ = [
    "WATERMARK", "STRATEGIES", "OUTCOME_CLASSES", "CERT_QUERY", "CERT_BUDGET_TEMPLATE",
    "DEFAULT_BUDGET_INT", "DEFAULT_MAX_STEPS", "DEFAULT_ALPHA", "KERNEL_ID_REGISTERED",
    "MODEL_X_SHA12_REGISTERED", "SPEC_S1X_SHA12_REGISTERED", "CODE_SHA12_REGISTERED",
    "ClosedLoopExpandedError", "StaticS1X", "ExpandedWorld", "PublicStateExpanded",
    "build_static_s1x", "legal_actions", "choose_action_expanded",
    "issue_stat_certificate", "trim_choice_detail", "run_policy_expanded",
    "branch_report_expanded", "VALIDATORS", "validate_ledger_expanded",
    "sha12_obj", "sha12_file12", "canonical_json", "record_key_json",
]

WATERMARK = {
    "research_line": "S3A_expanded_closed_loop",
    "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
    "noise": "N1",
    "dictionary": "COMPLETE_KNOWN",
    "claim_scope": "DEVELOPMENT_ONLY",
    "forbidden": ["真实贯通判定", "普遍策略优越", "成本优势主张", "对外承诺文本"],
}

STRATEGIES = ("NEW", "EC2_TWO_STEP", "RANDOM", "SPACE_FILL", "ORIENTATION")
OUTCOME_CLASSES = ("CERTIFIED_SINGLE", "DUAL_INDISTINGUISHABLE", "BUDGET_EXHAUSTED",
                   "POOL_EXHAUSTED", "MODEL_CONFLICT", "SEARCH_UNRESOLVED")

# 证书协议（与 S2-N N1 臂同一域口径：ALL_STATES + 同一节点/区域上限）
CERT_QUERY = "S1X:ALL"
CERT_BUDGET_TEMPLATE = {"domain": "ALL_STATES", "max_nodes": 64, "region_cap": 256,
                        "max_depth": 4, "statistic": sc.STAT_JOINT}

DEFAULT_ALPHA = Fraction(3, 40)          # = 0.075，与 S2-N / S3-N0 / S3-P 同口径
DEFAULT_BUDGET_INT = 16                  # 1e-2 域长单位；声明序列 14 + 最小行动 2
DEFAULT_MAX_STEPS = 4                    # = S1-X 声明序列长度
TABLE_ID = "S1X-T16-7WELL-V1"

KERNEL_ID_REGISTERED = "K-d7d914514865-9b14eeea67f7"
MODEL_X_SHA12_REGISTERED = "08e2fce787ec"
SPEC_S1X_SHA12_REGISTERED = "30cc51052f59"
CODE_SHA12_REGISTERED = "9b14eeea67f7"
SPEC_CONTENT_SHA12_REGISTERED = "d7d914514865"


class ClosedLoopExpandedError(RuntimeError):
    """闭环实现缺陷（不是读数）。"""


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def sha12_obj(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:12]


def sha12_file12(path: str) -> str:
    return sha12_file(path)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jsonable(obj):
    if isinstance(obj, Fraction):
        return "%d/%d" % (obj.numerator, obj.denominator)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def _rk_key(record) -> tuple:
    return record_key(record)


def _rk_json(record) -> list:
    aid, wid, counts = record_key(record)
    return [aid, int(wid), [[int(c), int(n)] for c, n in counts]]


def record_key_json(record) -> list:
    """公开记录 → 机读 record_key 形（[design_id, well_id, [[cell,count],...]]）。"""
    return _rk_json(record)


# ---------------------------------------------------------------------------
# 静态面：S1-X 世界装配（一次构建，多世界复用）
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StaticS1X:
    """S1-X 冻结身份 + 观测核 + 有限表 + 规划缓存（全部只读派生量）。"""

    root: str
    model: dict
    spec: dict
    kernel: GridDetectionKernel
    table: fs.FiniteTable
    labels: dict
    pool: list
    by_id: dict
    states: tuple
    cost_int: dict
    cost_domain_length: dict
    tier: str
    n_bits: int
    # 预计算（N 档口径）
    pmf_exact: dict            # {aid: {state: {record_key_tuple: Fraction}}}
    dists_float: dict          # {aid: {state: {record_key_tuple: float}}}
    law_tokens: dict           # {aid: {state: str}}  --- 观测律规范令牌（精确比较）
    support_records: dict      # {aid: [record dict, ...]} 全盘在位状态支撑（record_key 排序）
    support_sizes: dict        # {aid: int}
    separable: dict            # {aid: {(i,j): bool}} 两状态观测律是否不等
    read_masks: dict           # {aid: int} 暴露盘位掩码（公开几何）
    azimuths: dict             # {aid: float|None}
    ab_azimuth: float | None
    exposure_rows: list
    duplicate_proof: dict
    shared_cache: dict
    provenance: dict

    def pkey(self, record) -> tuple:
        return _rk_key(record)


def _build_table_s1x(model: dict, n_bits: int) -> fs.FiniteTable:
    from .connectivity_qoi import disk_point_distance
    from .intersect_disk import candidate_pairs, disk_pair_intersect

    discs = model["discs"]
    C = np.asarray([d["center"] for d in discs], dtype=float)
    N = np.asarray([d["normal"] for d in discs], dtype=float)
    R = np.asarray([d["radius"] for d in discs], dtype=float)
    pairs = candidate_pairs(C, R)
    hit = disk_pair_intersect(C, N, R, pairs[:, 0], pairs[:, 1], scale=float(np.median(R)))
    contact = tuple((int(i), int(j)) for (i, j) in pairs[hit].tolist())
    q = model["query"]
    A = np.asarray(q["A"], dtype=float)
    B = np.asarray(q["B"], dtype=float)
    ra = float(q["r_attach"])
    aatt = [int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, A)[0] <= ra)]
    batt = [int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, B)[0] <= ra)]
    labels = {int(k): int(v) for k, v in model["state_labels"].items()}

    def deriv(query_id: str, s: int) -> ident.StateDerivation:
        present = {i for i in range(n_bits) if (int(s) >> i) & 1}
        es = frozenset((i, j) for (i, j) in contact if i in present and j in present)
        a_side = frozenset(x for x in present if x in aatt)
        b_side = frozenset(x for x in present if x in batt)
        return ident.StateDerivation(state_id=int(s), nodes=frozenset(present), edges=es,
                                     a_side=a_side, b_side=b_side,
                                     attach_a=bool(a_side), attach_b=bool(b_side),
                                     label=labels[int(s)])

    mism = [s for s in range(1 << n_bits)
            if int(ident.state_connected(deriv("ALL", s))) != labels[s]]
    if mism:
        raise ClosedLoopExpandedError(
            "S1-X 派生标签与登记标签不符：%s" % (mism,))
    struct = cs.TableStructure(kind="DISK_BITS", n_bits=n_bits, certain=(),
                               uncertain=tuple(str(d.get("candidate_index"))
                                               for d in discs),
                               contact_edges=contact, attach_a_disks=tuple(aatt),
                               attach_b_disks=tuple(batt))
    table = fs.FiniteTable(
        table_id=TABLE_ID, n_bits=n_bits,
        labels={"S1X": np.asarray([labels[s] for s in range(1 << n_bits)],
                                  dtype=np.int64)},
        baseline=(), actions=(),
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N1",
              "model": "S1X 7-well expanded model (inherits S1 tiny geometry)",
              "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "S1X_DOMAIN"}})
    table.derivation = deriv
    return table


def _law_token(aid: str, pmf: dict) -> str:
    rows = sorted((_rk_json_key(k), "%d/%d" % (v.numerator, v.denominator))
                  for k, v in pmf.items())
    return canonical_json(rows)


def _rk_json_key(k) -> list:
    return [[int(c), int(n)] for c, n in k]


def build_static_s1x(root: str, model_rel: str, spec_rel: str, tier: str = "N1",
                     table_id: str = TABLE_ID) -> StaticS1X:
    """装配 S1-X 静态面：核身份复核 + 有限表 + 规划/证书缓存（tier 为预计算档）。"""
    model_path = os.path.join(root, model_rel)
    spec_path = os.path.join(root, spec_rel)
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    kernel = GridDetectionKernel(model, spec)
    if kernel.kernel_id != KERNEL_ID_REGISTERED:
        raise ClosedLoopExpandedError("STALE_KERNEL：%s != %s"
                                      % (kernel.kernel_id, KERNEL_ID_REGISTERED))
    n_bits = len(model["discs"])
    if n_bits != 4:
        raise ClosedLoopExpandedError("S1-X 应为 4 盘；现读 %d" % n_bits)
    table = _build_table_s1x(model, n_bits)
    labels = {int(k): int(v) for k, v in model["state_labels"].items()}
    pool = list(model["action_pool"])
    by_id = {a["action_id"]: a for a in pool}
    states = tuple(range(1 << n_bits))

    cache = {}
    pmf_exact, dists_float, law_tokens, support_records, separable = {}, {}, {}, {}, {}
    support_sizes = {}
    for a in pool:
        aid = str(a["action_id"])
        pmf_exact[aid], dists_float[aid], law_tokens[aid] = {}, {}, {}
        for s in states:
            pmf_raw = kernel.count_pmf(bits_of(s, n_bits), a, tier=tier)
            pmf = {}
            for vec, p in pmf_raw.items():
                if p == 0:
                    continue
                counts = {i: int(c) for i, c in enumerate(vec) if c}
                rk = tuple((int(i), int(c)) for i, c in counts.items())
                pmf[rk] = Fraction(p)
            pmf_exact[aid][int(s)] = pmf
            tot = 0.0
            dd = {}
            for rk, p in pmf.items():
                dd[rk] = float(p)
                tot += float(p)
            dists_float[aid][int(s)] = {k: v / tot for k, v in dd.items()}
            law_tokens[aid][int(s)] = _law_token(aid, pmf)
        full = kernel.count_pmf(bits_of((1 << n_bits) - 1, n_bits), a, tier=tier)
        recs = []
        for vec, p in full.items():
            if p == 0:
                continue
            counts = {i: int(c) for i, c in enumerate(vec) if c}
            recs.append(kernel.make_public_record(kernel._parse_action(a), counts))
        recs.sort(key=_rk_key)
        support_records[aid] = recs
        support_sizes[aid] = len(recs)
        sep = {}
        for i in states:
            for j in states:
                if i < j:
                    sep[(i, j)] = (law_tokens[aid][i] != law_tokens[aid][j])
        separable[aid] = sep

    read_masks, azimuths = {}, {}
    exposure_rows = []
    for a in pool:
        aid = str(a["action_id"])
        m = 0
        for i in range(n_bits):
            e = kernel.disc_exposure(a, i)
            exposure_rows.append({"action_id": aid, "disc_index": i,
                                  "exposed": bool(e["exposed"]), "reason": e["reason"],
                                  "t": e["t"]})
            if e["exposed"]:
                m |= 1 << i
        read_masks[aid] = m
        d = a["direction"]
        azimuths[aid] = math.degrees(math.atan2(float(d[1]), float(d[0])))
    q = model["query"]
    ab_azimuth = math.degrees(math.atan2(float(q["B"][1]) - float(q["A"][1]),
                                         float(q["B"][0]) - float(q["A"][0])))
    # 重复对证明：逐盘结局表全档相等 + 逐状态观测律令牌相等
    dup = {"action_ids": ["XS1", "XS1M"]}
    try:
        tables_equal_all_tiers = True
        for t in ("N0", "N1", "N2"):
            for i in range(n_bits):
                t1 = kernel.disc_outcome_table(by_id["XS1"], i, tier=t)
                t2 = kernel.disc_outcome_table(by_id["XS1M"], i, tier=t)
                if {k: Fraction(v) for k, v in t1.items()} != \
                        {k: Fraction(v) for k, v in t2.items()}:
                    tables_equal_all_tiers = False
    except Exception as exc:                            # noqa: BLE001
        tables_equal_all_tiers = False
        dup["exception"] = repr(exc)
    equal_tokens = all(law_tokens["XS1"][s] == law_tokens["XS1M"][s] for s in states)
    dup["per_disc_tables_equal_all_tiers"] = bool(tables_equal_all_tiers)
    dup["per_state_law_tokens_equal_at_tier"] = bool(equal_tokens)
    dup["tier"] = tier

    provenance = {
        "kernel_id": kernel.kernel_id,
        "kernel_formula_recompute": kernel_identity(
            {k: v for k, v in spec.items() if k != "kernel_identity"})["kernel_id"],
        "model_file": model_rel, "model_file_sha12": sha12_file(model_path),
        "model_file_sha12_registered": MODEL_X_SHA12_REGISTERED,
        "spec_file": spec_rel, "spec_file_sha12": sha12_file(spec_path),
        "spec_file_sha12_registered": SPEC_S1X_SHA12_REGISTERED,
        "spec_content_sha12": kernel.spec_content_sha12,
        "spec_content_sha12_registered": SPEC_CONTENT_SHA12_REGISTERED,
        "code_file": "src/fractureflow/observation_kernel.py",
        "code_sha12": kernel.code_sha12,
        "code_sha12_registered": CODE_SHA12_REGISTERED,
        "n_wells": len(pool), "n_bits": n_bits, "n_states": 1 << n_bits,
        "tier": tier,
    }
    # 签名独立性（T_by_signature）：按逐盘结局表签名（全档）分组
    sig_groups = {}
    for a in pool:
        rows = []
        for i in range(n_bits):
            tab = kernel.disc_outcome_table(a, i, tier=tier)
            rows.append(sorted((str(k), "%d/%d" % (v.numerator, v.denominator))
                               for k, v in tab.items()))
        sig = sha12_obj({"n_depth_bins": int(round(
            float(a["length"]) / kernel.depth_bin_width)), "per_disc": rows})
        sig_groups.setdefault(sig, []).append(str(a["action_id"]))
    T_by_signature = len(sig_groups)
    provenance["signature_groups"] = [{"signature_sha12": k, "action_ids": v}
                                      for k, v in sorted(sig_groups.items())]
    provenance["T_by_signature"] = T_by_signature
    provenance["sequence"] = [str(x) for x in model.get("action_sequence", [])]
    provenance["sequence_cost_int"] = int(sum(
        round(float(by_id[x]["cost"]) * 100) for x in model.get("action_sequence", [])))
    cost_int = {str(a["action_id"]): int(round(float(a["cost"]) * 100)) for a in pool}
    cost_dl = {str(a["action_id"]): float(a["cost"]) for a in pool}
    return StaticS1X(
        root=root, model=model, spec=spec, kernel=kernel, table=table, labels=labels,
        pool=pool, by_id=by_id, states=states, cost_int=cost_int,
        cost_domain_length=cost_dl, tier=tier, n_bits=n_bits,
        pmf_exact=pmf_exact, dists_float=dists_float, law_tokens=law_tokens,
        support_records=support_records, support_sizes=support_sizes,
        separable=separable, read_masks=read_masks, azimuths=azimuths,
        ab_azimuth=float(ab_azimuth), exposure_rows=exposure_rows,
        duplicate_proof=dup, shared_cache=cache, provenance=provenance)


# ---------------------------------------------------------------------------
# 公开面与世界
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PublicStateExpanded:
    """策略可见的全部信息（H_{t-1}）；不含真值、不含真值标签。"""

    world_id: str
    step: int
    allowed: tuple
    used: int
    budget: int
    executed: tuple
    legal: tuple
    alpha_seq: str
    history_sha12: str

    def as_dict(self) -> dict:
        return {"world_id": self.world_id, "step": self.step,
                "allowed_n": len(self.allowed), "used": self.used,
                "budget": self.budget, "executed": list(self.executed),
                "legal": list(self.legal), "alpha_seq": self.alpha_seq,
                "history_sha12": self.history_sha12}


class ExpandedWorld:
    """一个声明世界的运行面：真值 + 噪声流 + 公开历史重算 + 规划缓存。"""

    def __init__(self, static: StaticS1X, world_id: str, *, truth_state: int,
                 noise_seed: int, budget_int: int = DEFAULT_BUDGET_INT,
                 alpha_seq=DEFAULT_ALPHA, max_steps: int = DEFAULT_MAX_STEPS,
                 share_floor: float = 0.1, opponent_budget: int = 64,
                 max_witness_pairs: int = 128, branch_outcome_cap: int = 1024):
        self.static = static
        self.world_id = str(world_id)
        self.truth_state = int(truth_state)
        if not (0 <= self.truth_state < len(static.states)):
            raise ClosedLoopExpandedError("truth_state 越界：%r" % (truth_state,))
        self.noise_seed = int(noise_seed)
        self.noise_rng = np.random.default_rng(self.noise_seed)
        self.budget_int = int(budget_int)
        self.alpha_seq = Fraction(alpha_seq)
        self.max_steps = int(max_steps)
        self.share_floor = float(share_floor)
        self.opponent_budget = int(opponent_budget)
        self.max_witness_pairs = int(max_witness_pairs)
        self.branch_outcome_cap = int(branch_outcome_cap)
        self.pool_ids = tuple(str(a["action_id"]) for a in static.pool)
        self.costs = dict(static.cost_int)
        self.costs_dl = dict(static.cost_domain_length)
        self.labels = dict(static.labels)
        self.initial_allowed = tuple(int(s) for s in static.states)
        self._pcache = static.shared_cache
        self.n_eprocess = 0
        self.eprocess_seconds = 0.0
        self.loss_weights = aq.LossWeights(
            1.0, 1.0, 0.5, "S3A prereg 声明：0/1 对称损失 + 弃权 0.5；不由本单读数反推")

    # -- 公开面 ------------------------------------------------------------
    def equiprobable_prior(self):
        n = len(self.static.states)
        return {int(s): Fraction(1, n) for s in self.static.states}

    def mixture_for(self, steps_pre):
        """由公开历史构造混合预测（q_t 的来源）；空历史 = 均匀先验。"""
        mix = PredictiveMixture(self.static.states, self.static.kernel,
                                self.static.by_id[self.pool_ids[0]],
                                prior=self.equiprobable_prior(), mode="posterior",
                                tier=self.static.tier, cache=self._pcache)
        for aid, rec in steps_pre:
            mix.observe(rec, self.static.by_id[str(aid)])
        return mix

    def eprocess(self, steps):
        """由公开历史重算 e-process（S2-N 同码路）；空历史直接抛（协议要求非空）。"""
        if not steps:
            raise ClosedLoopExpandedError("公开历史为空：S2-N 协议禁建 e-process")
        t0 = time.perf_counter()
        try:
            ep = sc.run_eprocess(self.static.kernel, self.static.tier,
                                 self.static.states, self.labels, self.alpha_seq,
                                 [(self.static.by_id[str(a)], r) for a, r in steps],
                                 cache=self._pcache, model_id=self.static.table.table_id)
        finally:
            self.n_eprocess += 1
            self.eprocess_seconds += time.perf_counter() - t0
        return ep

    def state_set_for(self, steps) -> tuple:
        return tuple(self.eprocess(steps).state_set())

    def observe(self, action_id: str) -> dict:
        """执行侧：真值 + N1 噪声流 → 公开记录。"""
        if action_id not in self.static.by_id:
            raise ClosedLoopExpandedError("未知行动 %r" % (action_id,))
        return self.static.kernel.simulate(bits_of(self.truth_state, self.static.n_bits),
                                           self.static.by_id[action_id],
                                           rng=self.noise_rng, tier=self.static.tier)

    def public(self, allowed, used, executed, step, history_sha12: str = "") -> PublicStateExpanded:
        done = set(executed)
        legal = tuple(a for a in self.pool_ids
                      if a not in done and used + self.costs[a] <= self.budget_int)
        return PublicStateExpanded(self.world_id, int(step), tuple(allowed), int(used),
                                   self.budget_int, tuple(executed), legal,
                                   str(self.alpha_seq), str(history_sha12))

    def history_sha12(self, steps) -> str:
        rows = []
        for aid, rec in steps:
            rows.append([str(aid), _rk_json(rec)])
        return sha12_obj({"world_id": self.world_id, "tier": self.static.tier,
                          "steps": rows})

    def splits(self, allowed, action_id: str) -> bool:
        """该行动能否把 allowed 拆成 ≥2 个不同观测律类（S3-N0 `sim.splits` 的含噪同构）。"""
        tokens = {self.static.law_tokens[action_id][int(s)] for s in allowed}
        return len(tokens) > 1

    def q_support(self, steps_pre, action_id: str):
        """该行动在当前公开历史下的正概率结局记录（声明上界 = 全盘支撑，cap 记截断）。

        返回 ``([(support_index, record), ...], truncated)``；support_index 是该记录在
        该行动全盘支撑（``support_records[action_id]``，按 record_key 排序）中的序号，
        分支账用它做压引用（不逐行复制记录键）。
        """
        mix = self.mixture_for(steps_pre)
        action = self.static.by_id[action_id]
        recs = []
        for si, rec in enumerate(self.static.support_records[action_id]):
            if mix.prob(rec, action) > 0:
                recs.append((si, rec))
        cap = self.branch_outcome_cap
        truncated = len(recs) > cap
        if truncated:
            recs = recs[:cap]
        return recs, truncated


def legal_actions(world: ExpandedWorld, used: int, executed) -> tuple:
    done = set(executed)
    return tuple(a for a in world.pool_ids
                 if a not in done and used + world.costs[a] <= world.budget_int)


# ---------------------------------------------------------------------------
# 策略（只吃 PublicStateExpanded）
# ---------------------------------------------------------------------------

def _witness_pairs(world: ExpandedWorld, allowed) -> list:
    by_label = {0: [], 1: []}
    for s in allowed:
        by_label[world.labels[int(s)]].append(int(s))
    pairs = [(int(i), int(j)) for i in by_label[0] for j in by_label[1]]
    pairs.sort()
    return pairs


def _restrict_dists(world: ExpandedWorld, allowed, pool_ids) -> dict:
    out = {}
    for a in pool_ids:
        d = {}
        for s in allowed:
            d[int(s)] = world.static.dists_float[a][int(s)]
        out[a] = d
    return out


def _tie_pick(scored: dict, pool, world: ExpandedWorld) -> str:
    return str(sorted(pool, key=lambda a: (-float(scored[a]), world.costs[a], a))[0])


def _choose_new(world: ExpandedWorld, pub: PublicStateExpanded, rng) -> dict:
    part = tuple(pub.allowed)
    pairs_all = _witness_pairs(world, part)
    if not pairs_all:
        raise ClosedLoopExpandedError("%s NEW: 允许集内无对立标签对" % world.world_id)
    pairs = pairs_all[:world.max_witness_pairs]
    label_of = {int(s): int(world.labels[int(s)]) for s in part}
    w = ap.WitnessSet(tuple(pairs), labels=label_of, name="S3A_W",
                      provenance="统计可能集内对立标签对（字典序前 N）")
    pool = list(pub.legal)
    dists = _restrict_dists(world, part, pool)
    costs = {a: float(world.costs_dl[a]) for a in pool}
    opp = ap.opponent_search(w, dists, label_of, budget=world.opponent_budget,
                             stop_rule="EXHAUST_PAIRS_UP_TO_BUDGET")
    extra = [tuple(p) for p in opp["new_pairs_this_strictly_harder_or_equal"]
             if tuple(p) not in set(pairs)]
    merged = sorted(set(list(pairs) + extra))[:world.max_witness_pairs]
    w2 = ap.WitnessSet(tuple(merged), labels=label_of, name="S3A_W_HARDENED",
                       provenance="对手搜索扩集（cap=%d）" % world.max_witness_pairs)
    lam = ap.cost_share(w2, dists, costs, share_floor=world.share_floor)
    keep = ap.top_k_filter(lam["cost_share"], 4)
    kept = [a for a in keep["kept"] if a in dists]
    if not kept:
        raise ClosedLoopExpandedError("%s NEW: 前筛后无候选" % world.world_id)
    sub = {a: dists[a] for a in kept}
    sub_costs = {a: costs[a] for a in kept}
    ts = ap.two_step_lookahead(w2, sub, sub_costs, None, ks_compared=(4, None),
                               share_floor=world.share_floor)
    pick = _tie_pick(ts["two_step_scores"], kept, world)
    return {"action_id": pick,
            "detail": {
                "witness_pairs_total": len(pairs_all),
                "witness_pairs_used": len(merged),
                "witness_truncated": bool(len(pairs_all) > len(merged)),
                "opponent": {k: opp[k] for k in ("budget", "stop_rule", "found",
                                                 "candidates_evaluated",
                                                 "budget_exhausted",
                                                 "harder_than_threshold", "claim")},
                "opponent_extra_pairs": len(extra),
                "cost_share": {a: float(v) for a, v in lam["cost_share"].items()},
                "cost_share_degenerate": bool(lam.get("degenerate")),
                "top_k_kept": kept, "top_k_filtered_out": keep["filtered_out"],
                "two_step_scores": {a: float(v) for a, v in ts["two_step_scores"].items()},
                "planning_pool": pool}}


def _choose_ec2(world: ExpandedWorld, pub: PublicStateExpanded, rng) -> dict:
    part = tuple(pub.allowed)
    pool = list(pub.legal)
    belief = aq.Belief.build(tuple(int(s) for s in part),
                             [Fraction(1, len(part))] * len(part),
                             lambda s: world.labels[int(s)], "S3A:%s" % world.world_id)
    acts = [aq.CostAction(a, int(world.costs[a])) for a in pool]
    ofn = lambda s, aid: world.static.law_tokens[aid][int(s)]           # noqa: E731
    ec2 = aq.ec2_proxy_ranking(belief, acts, ofn)
    rows = sorted(ec2["rows"], key=lambda r: (-float(r.score_unit_cost), r.action_id))
    cands = [r.action_id for r in rows[:4]]
    if not cands:
        raise ClosedLoopExpandedError("%s EC2: 代理排序为空" % world.world_id)
    sub = [aq.CostAction(a, int(world.costs[a])) for a in cands]
    ts = aq.two_step_lookahead({world.world_id: belief}, world.loss_weights, sub, ofn,
                               candidates=cands)
    pick = str(ts["chosen"])
    if pick not in cands:
        raise ClosedLoopExpandedError("%s EC2: 两步选了候选池外的 %r"
                                      % (world.world_id, pick))
    return {"action_id": pick,
            "detail": {"ec2_order": [r.action_id for r in rows],
                       "ec2_scores": {r.action_id: float(r.score_unit_cost) for r in rows},
                       "ec2_n_cross_pairs": ec2["n_cross_pairs"],
                       "ec2_zero_mass_actions": ec2["zero_mass_actions"],
                       "candidates": cands, "two_step_chosen": pick,
                       "two_step_rows": len(ts["rows"]), "planning_pool": pool,
                       "outcome_fn": "LAW_TOKEN（逐状态观测律规范令牌；两状态可分 ⇔ 令牌不等）"}}


def _choose_random(world: ExpandedWorld, pub: PublicStateExpanded, rng) -> dict:
    pool = list(pub.legal)
    if not pool:
        raise ClosedLoopExpandedError("%s RANDOM: 空合法池" % world.world_id)
    idx = int(rng.randrange(len(pool)))
    return {"action_id": pool[idx], "detail": {"draw_index": idx, "pool": pool}}


def _choose_space_fill(world: ExpandedWorld, pub: PublicStateExpanded, rng,
                       read_so_far: int) -> dict:
    pool = list(pub.legal)
    if not pool:
        raise ClosedLoopExpandedError("%s SPACE_FILL: 空合法池" % world.world_id)
    best = sorted(pool, key=lambda a: (-bin(world.static.read_masks[a] & ~int(read_so_far))
                                       .count("1"), world.costs[a], a))[0]
    return {"action_id": best,
            "detail": {"newly_read_bits": {a: bin(world.static.read_masks[a]
                                                  & ~int(read_so_far)).count("1")
                                           for a in pool},
                       "read_mask_so_far": int(read_so_far), "planning_pool": pool}}


def _circ(a, b) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _choose_orientation(world: ExpandedWorld, pub: PublicStateExpanded, rng) -> dict:
    pool = list(pub.legal)
    if not pool:
        raise ClosedLoopExpandedError("%s ORIENTATION: 空合法池" % world.world_id)
    if world.static.ab_azimuth is not None:
        scored = {a: -_circ(world.static.azimuths[a], world.static.ab_azimuth)
                  for a in pool}
        best = sorted(pool, key=lambda a: (-scored[a], world.costs[a], a))[0]
        return {"action_id": best,
                "detail": {"basis": "AZIMUTH_TO_AB_AXIS",
                           "ab_azimuth_deg": float(world.static.ab_azimuth),
                           "azimuth_deg": {a: float(world.static.azimuths[a])
                                           for a in pool},
                           "circular_distance_deg":
                               {a: _circ(world.static.azimuths[a],
                                         world.static.ab_azimuth) for a in pool},
                           "planning_pool": pool}}
    order = [a for a in world.pool_ids if a in set(pool)]
    return {"action_id": order[0],
            "detail": {"basis": "DECLARED_INDEX_ORDER_NO_GEOMETRY",
                       "order": order, "planning_pool": pool}}


def choose_action_expanded(strategy: str, world: ExpandedWorld,
                           pub: PublicStateExpanded, rng, read_so_far: int = 0) -> dict:
    """**只看 PublicStateExpanded**（H_{t-1}）决定下一步行动。"""
    if strategy not in STRATEGIES:
        raise ClosedLoopExpandedError("未知策略 %r" % (strategy,))
    if not pub.legal:
        raise ClosedLoopExpandedError("%s/%s: 无合法行动却仍要求选点"
                                      % (world.world_id, strategy))
    if strategy in ("NEW", "EC2_TWO_STEP") and \
            len({world.labels[int(s)] for s in pub.allowed}) == 1:
        out = _choose_space_fill(world, pub, rng, read_so_far)
        out["detail"]["fallback"] = "NO_OPPOSITE_PAIRS_LABELS_UNIFORM"
        out["detail"]["fallback_rule"] = "SPACE_FILL（prereg 声明）"
        return out
    if strategy == "NEW":
        return _choose_new(world, pub, rng)
    if strategy == "EC2_TWO_STEP":
        return _choose_ec2(world, pub, rng)
    if strategy == "RANDOM":
        return _choose_random(world, pub, rng)
    if strategy == "SPACE_FILL":
        return _choose_space_fill(world, pub, rng, read_so_far)
    return _choose_orientation(world, pub, rng)


# ---------------------------------------------------------------------------
# 证书（S2-N 协议：每步对公开历史发证 + 独立验证）
# ---------------------------------------------------------------------------

def _cert_summary(rec: dict) -> dict:
    return {"issued": bool(rec["issued"]), "status": rec["status"],
            "reason": rec["reason"], "label_claim": rec["label_claim"],
            "verified": rec["verified"], "verify_reasons": list(rec["verify_reasons"]),
            "certificate_id": rec["certificate_id"],
            "certificate_sha12": rec["certificate_sha12"],
            "allowed_n": int(rec["allowed_n"]), "n_cubes": int(rec.get("n_cubes", 0)),
            "allowed_sha12": rec["allowed_sha12"],
            "s2n_status": rec.get("s2n_status"),
            "certify_seconds": round(rec["certify_seconds"], 6),
            "verify_seconds": round(rec["verify_seconds"], 6)}


def issue_stat_certificate(world: ExpandedWorld, steps, tag: str) -> dict:
    """对当前公开历史发 S2-N 含噪证书；空集/混合/冲突不发证，逐条记 reason。"""
    out = {"step_tag": str(tag), "kind": "S2N_STATISTICAL_CERTIFICATE",
           "issued": False, "status": "NOT_ISSUED", "reason": None, "label_claim": None,
           "allowed_n": len(world.static.states), "allowed_sha12": None,
           "n_cubes": 0, "verified": None, "verify_reasons": [],
           "certificate_id": None, "certificate_sha12": None,
           "certify_seconds": 0.0, "verify_seconds": 0.0,
           "s2n_status": None, "certificate": None, "verification": None}
    if not steps:
        out["status"] = "NOT_ATTEMPTED"
        out["reason"] = "T0_NO_PUBLIC_HISTORY_PER_S2N_PROTOCOL"
        return out
    t0 = time.perf_counter()
    try:
        ep = world.eprocess(steps)
        allowed = tuple(ep.state_set())
        out["allowed_n"] = len(allowed)
        out["allowed_sha12"] = sha12_obj(list(allowed))
        out["s2n_status"] = str(ep.status())
        res = sc.certify_with_statistics(world.static.table, ep, CERT_QUERY,
                                         dict(CERT_BUDGET_TEMPLATE), world.alpha_seq)
    except Exception as exc:                            # noqa: BLE001
        out["status"] = "EXCEPTION"
        out["reason"] = "CERTIFY_RAISED: %r" % (exc,)
        out["certify_seconds"] = round(time.perf_counter() - t0, 6)
        return out
    out["certify_seconds"] = round(time.perf_counter() - t0, 6)
    if isinstance(res, sc.StatisticalCertificate):
        cert = res.to_dict()
        out.update({"issued": True, "status": "CERTIFICATE",
                    "label_claim": str(cert.get("label_claim")),
                    "certificate_id": cert.get("certificate_id"),
                    "certificate_sha12": sha12_obj(cert), "certificate": cert,
                    "n_cubes": int(res.s2n.get("n_terminal_regions") or 0)})
        t1 = time.perf_counter()
        public = sc.PublicEvidence(model_id=world.static.table.table_id,
                                   kernel=world.static.kernel, tier=world.static.tier,
                                   states=world.static.states,
                                   steps=[(world.static.by_id[str(a)], r)
                                          for a, r in steps],
                                   labels=world.labels, cache=world._pcache)
        try:
            rep = sc.verify_statistical_certificate(world.static.table, public,
                                                    CERT_QUERY, cert)
            out["verified"] = bool(rep.get("pass"))
            out["verify_reasons"] = list(rep.get("reasons", []))
            out["verification"] = {"pass": bool(rep.get("pass")),
                                   "reasons": list(rep.get("reasons", [])),
                                   "checks": rep.get("checks"),
                                   "verifier_version": rep.get("verifier_version")}
        except Exception as exc:                        # noqa: BLE001
            out["verified"] = False
            out["verify_reasons"] = ["VERIFY_RAISED: %r" % (exc,)]
        out["verify_seconds"] = round(time.perf_counter() - t1, 6)
        return out
    out["status"] = str(res.status)                     # EMPTY / MIXED_LABELS / UNRESOLVED
    out["reason"] = str(res.reason)
    detail = res.detail if isinstance(res.detail, dict) else {}
    out["not_issued_detail"] = jsonable(detail)
    return out


# ---------------------------------------------------------------------------
# 终末判定
# ---------------------------------------------------------------------------

def _stop_and_why(world: ExpandedWorld, allowed, used, executed, step, max_steps,
                  cert: dict) -> dict:
    """按 prereg 声明序判定终末类；返回 {} 表示继续。"""
    if not allowed:
        return {"class": "MODEL_CONFLICT", "subreason": "ALLOWED_EMPTY",
                "detail": {"at_step": step}}
    if cert["issued"] and cert["verified"]:
        return {"class": "CERTIFIED_SINGLE", "subreason": "CERTIFICATE_VERIFIED",
                "detail": {"label_claim": cert["label_claim"],
                           "certificate_id": cert["certificate_id"],
                           "s2n_status": cert.get("s2n_status")}}
    if cert["issued"] and not cert["verified"]:
        return {"class": "SEARCH_UNRESOLVED", "subreason": "CERTIFICATE_UNVERIFIED",
                "detail": {"verify_reasons": list(cert["verify_reasons"])}}
    if str(cert.get("status")) == "EXCEPTION":
        return {"class": "SEARCH_UNRESOLVED", "subreason": "CERTIFICATE_EXCEPTION",
                "detail": {"reason": cert["reason"]}}
    legal = legal_actions(world, used, executed)
    if not legal:
        if len(executed) >= len(world.pool_ids):
            return {"class": "POOL_EXHAUSTED", "subreason": "NO_UNEXECUTED_ACTION",
                    "detail": {"executed": list(executed)}}
        return {"class": "BUDGET_EXHAUSTED", "subreason": "NO_AFFORDABLE_ACTION",
                "detail": {"used": used, "budget": world.budget_int,
                           "remaining": [a for a in world.pool_ids
                                         if a not in set(executed)]}}
    if step >= max_steps:
        return {"class": "SEARCH_UNRESOLVED", "subreason": "STEP_CAP",
                "detail": {"max_steps": max_steps}}
    if not any(world.splits(allowed, a) for a in legal):
        labels = sorted({world.labels[int(s)] for s in allowed})
        if len(labels) == 1:
            return {"class": "SEARCH_UNRESOLVED", "subreason": "PART_FROZEN_UNCERTIFIED",
                    "detail": {"legal": list(legal), "allowed_n": len(allowed),
                               "labels": labels}}
        return {"class": "DUAL_INDISTINGUISHABLE",
                "subreason": "NO_LEGAL_ACTION_SPLITS_ALLOWED_SET",
                "detail": {"legal": list(legal), "allowed_n": len(allowed),
                           "labels": labels}}
    return {}


# ---------------------------------------------------------------------------
# 闭环执行（任务单 §2.1 签名）
# ---------------------------------------------------------------------------

def run_policy_expanded(strategy: str, model_x, action_pool, budget: int, alpha_seq,
                        rng, max_steps: int) -> dict:
    """跑一条含噪闭环轨迹（S1-X 扩展行动池）。

    ``model_x`` = :class:`ExpandedWorld`（由 ``build_static_s1x`` + 声明
    world/truth/noise_seed 装配）。策略选择只吃 ``PublicStateExpanded``；
    真值只在 ``world.observe`` 内出现；噪声流 = ``world.noise_rng``。
    """
    world = model_x
    if not isinstance(world, ExpandedWorld):
        raise ClosedLoopExpandedError("model_x 必须是 ExpandedWorld")
    if strategy not in STRATEGIES:
        raise ClosedLoopExpandedError("未知策略 %r" % (strategy,))
    pool = tuple(str(a) for a in action_pool)
    if set(pool) != set(world.pool_ids):
        raise ClosedLoopExpandedError("%s: action_pool 与声明池不一致" % world.world_id)
    if int(budget) != world.budget_int:
        raise ClosedLoopExpandedError("%s: budget %r != 声明 %r（禁换预算）"
                                      % (world.world_id, budget, world.budget_int))
    if Fraction(alpha_seq) != world.alpha_seq:
        raise ClosedLoopExpandedError("%s: alpha %r != 声明 %r（禁换 α）"
                                      % (world.world_id, alpha_seq, world.alpha_seq))
    if int(max_steps) != world.max_steps:
        raise ClosedLoopExpandedError("%s: max_steps %r != 声明 %r"
                                      % (world.world_id, max_steps, world.max_steps))
    t_start = time.perf_counter()
    t_choice = t_cert = t_verify = 0.0
    n_choose = n_cert = n_verify = 0

    steps = []                    # [(action_id, record)] 公开历史
    allowed = tuple(int(s) for s in world.initial_allowed)
    used = 0
    executed = ()
    read_so_far = 0
    rows = []
    certs = []
    t_cert0 = time.perf_counter()
    cert0 = issue_stat_certificate(world, (), "t0")
    t_cert += time.perf_counter() - t_cert0
    rows.append({"t": 0, "action_id": None, "cost": 0, "used_before": 0,
                 "used_after": 0, "record_key": None, "record_sha12": None,
                 "allowed_before": len(allowed), "allowed_after": len(allowed),
                 "allowed_sha12": sha12_obj(list(allowed)),
                 "allowed_state_ids": list(allowed) if len(allowed) <= 256 else None,
                 "certificate": _cert_summary(cert0), "eprocess": None,
                 "choice_detail": None, "read_mask_so_far": 0,
                 "pin": "t0：S2-N 协议要求非空公开历史 ⇒ 不发证书也不尝试"})
    certs.append(cert0)
    stop = _stop_and_why(world, allowed, used, executed, 0, max_steps, cert0)

    while not stop:
        step = len(steps)
        pub = world.public(allowed, used, executed, step,
                           world.history_sha12(steps))
        ts = time.perf_counter()
        choice = choose_action_expanded(strategy, world, pub, rng, read_so_far)
        t_choice += time.perf_counter() - ts
        n_choose += 1
        aid = str(choice["action_id"])
        if aid not in pub.legal:
            raise ClosedLoopExpandedError("%s/%s: 选中非法行动 %r（合法池 %r）"
                                          % (world.world_id, strategy, aid, pub.legal))
        before = used
        allowed_before_n = len(allowed)
        used += world.costs[aid]
        if used > world.budget_int:
            raise ClosedLoopExpandedError("%s: 超预算 %d > %d"
                                          % (world.world_id, used, world.budget_int))
        record = world.observe(aid)
        executed = executed + (aid,)
        read_so_far |= int(world.static.read_masks[aid])
        steps.append((aid, record))
        ep = world.eprocess(steps)
        allowed = tuple(ep.state_set())
        tk = time.perf_counter()
        cert = issue_stat_certificate(world, steps, "t%d" % len(steps))
        t_cert += time.perf_counter() - tk
        if cert["issued"]:
            n_verify += 1
            t_verify += cert["verify_seconds"]
        n_cert += 1
        rows.append({"t": len(steps), "action_id": aid,
                     "cost": int(world.costs[aid]),
                     "cost_domain_length": float(world.costs_dl[aid]),
                     "used_before": int(before), "used_after": int(used),
                     "record_key": _rk_json(record),
                     "record_sha12": sha12_obj(record),
                     "record_counts": record.get("counts"),
                     "allowed_before": int(allowed_before_n),
                     "allowed_after": len(allowed),
                     "allowed_sha12": sha12_obj(list(allowed)),
                     "allowed_state_ids": list(allowed) if len(allowed) <= 256 else None,
                     "certificate": _cert_summary(cert),
                     "eprocess": {"t": int(ep.t), "status": str(ep.status()),
                                  "n_S": len(ep.state_set()),
                                  "Gamma": list(ep.target_set())},
                     "choice_detail": trim_choice_detail(choice["detail"]),
                     "read_mask_so_far": int(read_so_far)})
        certs.append(cert)
        stop = _stop_and_why(world, allowed, used, executed, len(steps), max_steps, cert)

    elapsed = time.perf_counter() - t_start
    answer = None
    for c in certs:
        if c["issued"] and c["verified"]:
            answer = int(c["label_claim"])
            break
    dup_used = ("XS1" in executed) and ("XS1M" in executed)
    return {
        "schema": "s3a.trace.v1",
        "world_id": world.world_id,
        "strategy": strategy,
        "truth_state": int(world.truth_state),
        "truth_label": int(world.labels[world.truth_state]),
        "tier": world.static.tier,
        "kernel_id": world.static.kernel.kernel_id,
        "budget": int(world.budget_int),
        "alpha_seq": str(world.alpha_seq),
        "max_steps": int(max_steps),
        "initial_allowed": len(world.initial_allowed),
        "noise_seed": int(world.noise_seed),
        "steps": rows,
        "stop": stop,
        "used": int(used),
        "n_steps": len(steps),
        "executed": list(executed),
        "answer": answer,
        "correct": (None if answer is None
                    else bool(answer == int(world.labels[world.truth_state]))),
        "certificate_attempts": int(n_cert),
        "certificates_issued": int(sum(1 for c in certs if c["issued"])),
        "certificates_verified": int(sum(1 for c in certs
                                         if c["issued"] and c["verified"])),
        "certificates_unverified": int(sum(1 for c in certs
                                           if c["issued"] and not c["verified"])),
        "certificates_not_attempted_t0": 1,
        "not_issued_count": int(sum(1 for c in certs if not c["issued"])),
        "duplicate_pair_both_executed": bool(dup_used),
        "history_sha12": world.history_sha12(steps),
        "cost": {"observed": int(used), "compute_seconds": round(elapsed, 6),
                 "choice_seconds": round(t_choice, 6),
                 "certify_seconds": round(t_cert, 6),
                 "verify_seconds": round(t_verify, 6),
                 "eprocess_seconds": round(world.eprocess_seconds, 6),
                 "n_choose_calls": int(n_choose),
                 "n_certify_calls": int(n_cert),
                 "n_verify_calls": int(n_verify),
                 "basis": "MEASURED_ONLY"},
        "certificates": [_cert_summary(c) for c in certs],
        "_certificates_full": [c for c in certs if c["issued"]],
        "watermark": dict(WATERMARK),
    }


def trim_choice_detail(detail):
    """规划明细入账口径：ap/aq 输出本已摘要化（两步只存分数表），此处去掉大字段。"""
    if not isinstance(detail, dict):
        return detail
    out = {}
    for k, v in detail.items():
        if k == "two_step_rows":
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {str(x): (round(float(y), 9) if isinstance(y, float) else y)
                      for x, y in v.items()}
        elif isinstance(v, list):
            out[k] = [str(x) if not isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 反事实分支（行动级主判 / 决策级辅助；证书层终末不逐分支模拟）
# ---------------------------------------------------------------------------

def _random_prefix_rng(world: ExpandedWorld, acted, policy_seed: int, k: int):
    r = _random.Random(policy_seed)
    used = 0
    executed = ()
    for _ in range(k):
        pool = [a for a in world.pool_ids if a not in set(executed)
                and used + world.costs[a] <= world.budget_int]
        if not pool:
            return None
        pick = pool[int(r.randrange(len(pool)))]
        used += world.costs[pick]
        executed = executed + (pick,)
    return r


def branch_report_expanded(strategy: str, world: ExpandedWorld, trace: dict,
                           policy_seed: int) -> dict:
    """逐步枚举该步行动在当前公开历史下的全部正概率结局，重算下一步行动。

    行动级分支 = 不同结局 → 不同下一步行动（主判）。
    决策级 = 下一步行动或廉价公开终末谓词（预算/池/步上限/无行动可再分）的变化（辅助）。
    证书层终末（获证与否）**不**逐分支模拟（成本上限，声明于 prereg）。
    """
    acted = [x for x in trace["steps"] if x["action_id"] is not None]
    rows = []
    allowed = tuple(int(s) for s in world.initial_allowed)
    used = 0
    executed = ()
    read_so_far = 0
    for k, st in enumerate(acted):
        aid = str(st["action_id"])
        # 公开历史（真实记录）——从 trace 重建
        pre_hist = []
        for x in acted[:k]:
            rid = x["record_key"]
            pre_hist.append((str(x["action_id"]), _record_from_json(world, rid)))
        recs, truncated = world.q_support(pre_hist, aid)
        idx_of = {_rk_key(r): i for i, (si, r) in enumerate(recs)}
        next_actions, terminal_by_outcome, decisions = {}, {}, {}
        detail_cache = {}
        base_rng = _random_prefix_rng(world, acted, policy_seed, k) \
            if strategy == "RANDOM" else None
        used2 = used + world.costs[aid]
        ex2 = executed + (aid,)
        read2 = read_so_far | int(world.static.read_masks[aid])
        for si, r in recs:
            idx = idx_of[_rk_key(r)]
            allowed_new = tuple(world.eprocess(pre_hist + [(aid, r)]).state_set())
            sb = _cheap_terminal(world, allowed_new, used2, ex2, k + 1)
            if sb is not None:
                terminal_by_outcome[idx] = sb
                next_actions[idx] = None
                decisions[idx] = "STOP:" + sb
                continue
            if strategy == "RANDOM":
                if base_rng is None:
                    next_actions[idx] = None
                    decisions[idx] = "STOP:NO_RNG_BASE"
                    continue
                rb = _random.Random()
                rb.setstate(base_rng.getstate())
                bp = list(legal_actions(world, used2, ex2))
                if not bp:
                    terminal_by_outcome[idx] = "NO_LEGAL"
                    next_actions[idx] = None
                    decisions[idx] = "STOP:NO_LEGAL"
                    continue
                pick = str(bp[int(rb.randrange(len(bp)))])
                next_actions[idx] = pick
                decisions[idx] = pick
                continue
            key_allowed = allowed_new
            if key_allowed in detail_cache:
                pick = detail_cache[key_allowed]
            else:
                pub = world.public(allowed_new, used2, ex2, k + 1)
                ch = choose_action_expanded(strategy, world, pub, _random.Random(0),
                                            read2)
                pick = str(ch["action_id"])
                detail_cache[key_allowed] = pick
            next_actions[idx] = pick
            decisions[idx] = pick
        acts_only = sorted({v for v in next_actions.values() if v is not None})
        dec_set = sorted(set(decisions.values()))
        rows.append({"t": int(st["t"]), "action_id": aid,
                     "n_outcomes": len(recs), "outcomes_truncated": bool(truncated),
                     "observed_outcome_index": idx_of.get(
                         _rk_key(_record_from_json(world, st["record_key"]))),
                     "outcome_support_indices": [int(si) for si, _r in recs],
                     "support_catalog_ref": ("branch_diagnostics.json#support_catalog"
                                             "[action_id]（全盘支撑按 record_key 排序）"),
                     "next_action_by_outcome": {str(i): v
                                                for i, v in next_actions.items()},
                     "terminal_by_outcome": {str(i): v
                                             for i, v in terminal_by_outcome.items()},
                     "decision_by_outcome": {str(i): d
                                             for i, d in decisions.items()},
                     "n_next_distinct": len(acts_only),
                     "n_next_distinct_decisions": len(dec_set),
                     "branch_point": bool(len(acts_only) >= 2),
                     "branch_point_by_decision": bool(len(dec_set) >= 2),
                     "distinct_next_actions": acts_only,
                     "distinct_decisions": dec_set,
                     "caliber_note": ("决策层含廉价公开终末谓词；证书层终末不逐分支模拟"
                                      "（成本上限，见 prereg counterfactual）")})
        allowed = tuple(world.eprocess(pre_hist + [(aid, _record_from_json(world, st["record_key"]))]).state_set())
        used += world.costs[aid]
        executed = executed + (aid,)
        read_so_far |= int(world.static.read_masks[aid])
    return {"world_id": world.world_id, "strategy": strategy,
            "seed": int(trace.get("seed", -1)),
            "policy_seed": int(policy_seed),
            "rows": rows,
            "has_branch_point": bool(any(r["branch_point"] for r in rows)),
            "has_branch_point_by_decision": bool(
                any(r["branch_point_by_decision"] for r in rows)),
            "n_branch_points": int(sum(1 for r in rows if r["branch_point"])),
            "n_branch_points_by_decision": int(
                sum(1 for r in rows if r["branch_point_by_decision"])),
            "all_steps_invariant": bool(all(r["n_next_distinct"] <= 1 for r in rows)),
            "pre_planned": bool(all(r["n_next_distinct"] <= 1 for r in rows)),
            "notes": {"rng_handling": ("RANDOM：每个分支从同一前缀 rng 状态取同一次抽取；"
                                       "确定性策略分支重算用 Random(0)（不用 rng）"),
                      "enumeration": ("结局 = 该步行动在 q_t（公开历史混合预测）下的"
                                      "正概率记录；全盘支撑为声明上界（cap=%d）"
                                      % world.branch_outcome_cap)}}


def _record_from_json(world: ExpandedWorld, rk_json) -> dict:
    """由 record_key 的 JSON 形复原公开记录（与生成侧同一 make_public_record 码路）。"""
    aid, wid, counts = rk_json[0], int(rk_json[1]), rk_json[2]
    action = world.static.kernel._parse_action(world.static.by_id[str(aid)])
    return world.static.kernel.make_public_record(
        action, {int(c): int(n) for c, n in counts})


def _cheap_terminal(world: ExpandedWorld, allowed, used, executed, step) -> str:
    """决策级分支用的**廉价公开终末谓词**（不含证书层）。"""
    if not allowed:
        return "MODEL_CONFLICT_EMPTY_ALLOWED"
    legal = legal_actions(world, used, executed)
    if not legal:
        if len(executed) >= len(world.pool_ids):
            return "POOL_EXHAUSTED"
        return "BUDGET_EXHAUSTED"
    if step >= world.max_steps:
        return "STEP_CAP"
    if not any(world.splits(allowed, a) for a in legal):
        labels = {world.labels[int(s)] for s in allowed}
        if len(labels) == 1:
            return "PART_FROZEN_UNCERTIFIED"
        return "DUAL_INDISTINGUISHABLE"
    return None


# ---------------------------------------------------------------------------
# 校验器（控制面）
# ---------------------------------------------------------------------------

def _iter_runs(ledger):
    return list(ledger.get("runs", []))


def v_ledger_nonempty(ledger, ctx=None):
    if not _iter_runs(ledger):
        raise SystemExit("[s3a] 空账：拒绝记 PASS")
    n_w = len({r["world_id"] for r in _iter_runs(ledger)})
    n_s = len({r["strategy"] for r in _iter_runs(ledger)})
    n_k = len({int(r["seed"]) for r in _iter_runs(ledger)})
    if n_w == 0 or n_s == 0 or n_k == 0:
        raise SystemExit("[s3a] 空世界/空策略/空种子面：拒绝记 PASS")
    return []


def v_budget(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        used = 0
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            if used + int(s["cost"]) > int(r["budget"]):
                bad.append({"code": "OVER_BUDGET", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": s["t"], "used": used, "cost": s["cost"],
                            "budget": r["budget"]})
            used += int(s["cost"])
        if used != int(r["used"]):
            bad.append({"code": "USED_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"],
                        "used": r["used"], "sum": used})
    return bad


def v_no_repeat(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        seen = []
        for s in r["steps"]:
            a = s["action_id"]
            if a is None:
                continue
            if a in seen:
                bad.append({"code": "REPEATED_ACTION", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "action": a})
            seen.append(a)
    return bad


def v_outcome_accounting(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        acted = [s for s in r["steps"] if s["action_id"] is not None]
        if len(acted) != r["n_steps"]:
            bad.append({"code": "STEP_COUNT_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"],
                        "rows": len(acted), "n_steps": r["n_steps"]})
        for s in acted:
            if s.get("record_key") is None or s.get("record_sha12") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            if s.get("used_after") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
        if not r["steps"] or r["steps"][-1].get("certificate") is None:
            bad.append({"code": "MISSING_TERMINAL_ROW", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    return bad


def v_denominator(ledger, ctx=None):
    roster = ledger.get("roster") or {}
    want = {(w, s, int(k)) for w in roster.get("worlds", [])
            for s in roster.get("strategies", []) for k in roster.get("seeds", [])}
    if not want:
        raise SystemExit("[s3a] 声明名册为空：拒绝记 PASS")
    got = [(r["world_id"], r["strategy"], int(r["seed"])) for r in _iter_runs(ledger)]
    bad = []
    if len(got) != len(set(got)):
        bad.append({"code": "DUPLICATE_RUN"})
    for m in sorted(want - set(got)):
        bad.append({"code": "MISSING_RUN", "run": list(m)})
    for e in sorted(set(got) - want):
        bad.append({"code": "EXTRA_RUN", "run": list(e)})
    for r in _iter_runs(ledger):
        cls = (r.get("stop") or {}).get("class")
        if cls not in OUTCOME_CLASSES:
            bad.append({"code": "UNKNOWN_OUTCOME_CLASS", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"], "class": cls})
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
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"],
                            "reasons": c.get("verify_reasons")})
            if c["issued"] and int(c.get("allowed_n", 0)) <= 0:
                bad.append({"code": "CERTIFICATE_ON_EMPTY_SET", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            if c["issued"]:
                terminal_t = s["t"]
                if str(c.get("label_claim")) not in ("0", "1"):
                    bad.append({"code": "CERTIFICATE_LABEL_INVALID",
                                "world": r["world_id"], "strategy": r["strategy"],
                                "seed": r["seed"], "t": s["t"]})
                # 「0 错证书当真相仍在」：真状态若仍在统计可能集，证书标签必须等于真标签
                ids = s.get("allowed_state_ids")
                sid = r["truth_state"]
                if int(sid) in set(ids if ids is not None else []) and \
                        int(c["label_claim"]) != int(r["truth_label"]):
                    bad.append({"code": "WRONG_CERT_WHILE_TRUTH_POSSIBLE",
                                "world": r["world_id"], "strategy": r["strategy"],
                                "seed": r["seed"], "t": s["t"]})
        if (r.get("stop") or {}).get("class") == "CERTIFIED_SINGLE" and terminal_t is None:
            bad.append({"code": "CERTIFIED_WITHOUT_CERTIFICATE",
                        "world": r["world_id"], "strategy": r["strategy"],
                        "seed": r["seed"]})
        if (r.get("stop") or {}).get("class") == "MODEL_CONFLICT" and terminal_t is not None:
            bad.append({"code": "MODEL_CONFLICT_WITH_CERTIFICATE",
                        "world": r["world_id"], "strategy": r["strategy"],
                        "seed": r["seed"]})
    return bad


def v_config_equal(ledger, ctx=None):
    cfg = ledger.get("configs") or {}
    if not cfg:
        raise SystemExit("[s3a] 空配置面：拒绝记 PASS")
    bad = []
    for w, spec in cfg.items():
        base = None
        for r in _iter_runs(ledger):
            if r["world_id"] != w:
                continue
            key = (r["budget"], r["alpha_seq"], r["max_steps"])
            if base is None:
                base = key
            elif key != base:
                bad.append({"code": "CONFIG_DRIFT_ACROSS_STRATEGIES", "world": w,
                            "strategy": r["strategy"], "seed": r["seed"],
                            "got": list(key), "want": list(base)})
        if base is not None and (spec.get("budget"), spec.get("alpha_seq"),
                                 spec.get("max_steps")) != base:
            bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": w,
                        "got": list(base), "want": [spec.get("budget"),
                                                    spec.get("alpha_seq"),
                                                    spec.get("max_steps")]})
    return bad


def v_noise_tier(ledger, ctx=None):
    """主账噪声档必须全为 prereg 声明档（N1）；N2 只许出现在子模型面。"""
    pre = (ctx or {}).get("prereg") or {}
    want = str((pre.get("watermark") or {}).get("noise") or "N1")
    bad = []
    for r in _iter_runs(ledger):
        tier = str(r.get("tier"))
        if tier != want:
            bad.append({"code": "NOISE_TIER_DRIFT", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"],
                        "got": tier, "want": want})
    return bad


def v_duplicate_pair_identity(ledger, ctx=None):
    """XS1/XS1M 不得当两个独立行动：T_by_signature 必须 = 重算值（且 = 6），重复对证明在场。"""
    ident_block = ledger.get("identity") or {}
    ctx = ctx or {}
    static = ctx.get("static")
    bad = []
    if static is None:
        raise SystemExit("[s3a] 校验缺少静态面：拒绝记 PASS")
    T_recomputed = int(static.provenance["T_by_signature"])
    if int(ident_block.get("T_by_signature", -1)) != T_recomputed:
        bad.append({"code": "INDEPENDENT_ACTION_COUNT_DRIFT",
                    "declared": ident_block.get("T_by_signature"),
                    "recomputed": T_recomputed})
    if T_recomputed != 6:
        bad.append({"code": "INDEPENDENT_ACTION_COUNT_DRIFT",
                    "recomputed": T_recomputed, "want": 6})
    dup = static.duplicate_proof
    if not (dup.get("per_disc_tables_equal_all_tiers")
            and dup.get("per_state_law_tokens_equal_at_tier")):
        bad.append({"code": "DUPLICATE_PAIR_NOT_PROVEN", "proof": jsonable(dup)})
    if int(ident_block.get("n_wells", -1)) != len(static.pool):
        bad.append({"code": "INDEPENDENT_ACTION_COUNT_DRIFT",
                    "declared_n_wells": ident_block.get("n_wells"),
                    "recomputed": len(static.pool)})
    return bad


def v_kernel_identity(ledger, ctx=None):
    """kernel_id / 文件 sha12 必须与现盘逐位一致（改核/加井不换 id 即红）。"""
    ident_block = ledger.get("identity") or {}
    ctx = ctx or {}
    static = ctx.get("static")
    bad = []
    if static is None:
        raise SystemExit("[s3a] 校验缺少静态面：拒绝记 PASS")
    prov = static.provenance
    if str(ident_block.get("kernel_id")) != str(static.kernel.kernel_id):
        bad.append({"code": "KERNEL_ID_DRIFT",
                    "declared": ident_block.get("kernel_id"),
                    "recomputed": static.kernel.kernel_id})
    if str(ident_block.get("kernel_id")) != KERNEL_ID_REGISTERED:
        bad.append({"code": "KERNEL_ID_DRIFT", "declared": ident_block.get("kernel_id"),
                    "registered": KERNEL_ID_REGISTERED})
    if str(prov.get("code_sha12")) != CODE_SHA12_REGISTERED:
        bad.append({"code": "KERNEL_CODE_DRIFT", "recomputed": prov.get("code_sha12"),
                    "registered": CODE_SHA12_REGISTERED})
    if str(prov.get("model_file_sha12")) != MODEL_X_SHA12_REGISTERED:
        bad.append({"code": "MODEL_FILE_DRIFT",
                    "recomputed": prov.get("model_file_sha12"),
                    "registered": MODEL_X_SHA12_REGISTERED})
    if str(ident_block.get("model_file_sha12")) != str(prov.get("model_file_sha12")):
        bad.append({"code": "MODEL_FILE_DRIFT",
                    "declared": ident_block.get("model_file_sha12"),
                    "recomputed": prov.get("model_file_sha12")})
    return bad


def v_public_choice(ledger, ctx=None):
    """逐步用公开历史重算策略选择；不可复现即判红（禁真值选点）。

    注入坏账时不得把校验器打死：重放中的次生异常记 `REPLAY_RAISED` /
    `HISTORY_ADVANCE_FAILED` 并保留此前已判出的码。
    """
    ctx = ctx or {}
    make_world = ctx.get("make_world")
    if make_world is None:
        raise SystemExit("[s3a] 校验缺少世界工厂：拒绝记 PASS")
    bad = []
    for r in _iter_runs(ledger):
        world = make_world(r["world_id"], int(r["seed"]))
        rng = _random.Random()
        rng.seed(int(r["policy_seed"]))
        allowed = tuple(int(s) for s in world.initial_allowed)
        used = 0
        executed = ()
        read_so_far = 0
        steps = []
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            try:
                pub = world.public(allowed, used, executed, s["t"] - 1,
                                   world.history_sha12(steps))
                ch = choose_action_expanded(r["strategy"], world, pub, rng, read_so_far)
            except Exception as exc:                    # noqa: BLE001
                bad.append({"code": "REPLAY_RAISED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": s["t"], "error": repr(exc)})
                break
            if str(ch["action_id"]) != str(s["action_id"]):
                bad.append({"code": "ACTION_NOT_REPRODUCIBLE_FROM_PUBLIC_HISTORY",
                            "world": r["world_id"], "strategy": r["strategy"],
                            "seed": r["seed"], "t": s["t"],
                            "recorded": s["action_id"], "recomputed": ch["action_id"]})
            if s.get("record_key") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
                break
            rec = _record_from_json(world, s["record_key"])
            steps.append((str(s["action_id"]), rec))
            try:
                allowed = tuple(world.eprocess(steps).state_set())
            except Exception as exc:                    # noqa: BLE001
                bad.append({"code": "HISTORY_ADVANCE_FAILED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": s["t"], "error": repr(exc)})
                break
            used += int(s["cost"])
            executed = executed + (str(s["action_id"]),)
            read_so_far |= int(world.static.read_masks[str(s["action_id"])])
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


def v_adaptivity(ledger, ctx=None):
    cf = ledger.get("counterfactual") or {}
    runs = cf.get("runs") or []
    if not runs:
        raise SystemExit("[s3a] 反事实面为空：拒绝记 PASS")
    if not any(r.get("has_branch_point") for r in runs):
        return [{"code": "PRE_PLANNED_NO_BRANCH_POINT", "runs": len(runs)}]
    return []


def v_counterfactual_consistency(ledger, ctx=None):
    cf = ledger.get("counterfactual") or (ctx or {}).get("counterfactual") or {}
    rows_all = cf.get("runs") or []
    if not rows_all:
        raise SystemExit("[s3a] 反事实面为空：拒绝记 PASS")
    bad = []
    for r in rows_all:
        for row in r["rows"]:
            nxt = row.get("next_action_by_outcome") or {}
            n_act = len({v for v in nxt.values() if v is not None})
            if int(row.get("n_next_distinct", -1)) != n_act:
                bad.append({"code": "NEXT_DISTINCT_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row["t"], "declared": row.get("n_next_distinct"),
                            "recomputed": n_act})
            if bool(row.get("branch_point")) != bool(n_act >= 2):
                bad.append({"code": "BRANCH_FLAG_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": row["t"]})
            dec = row.get("decision_by_outcome") or {}
            n_dec = len(set(dec.values()))
            if int(row.get("n_next_distinct_decisions", -1)) != n_dec:
                bad.append({"code": "NEXT_DECISIONS_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row["t"]})
            if set(dec) != set(row.get("next_action_by_outcome") or {}):
                bad.append({"code": "DECISION_INDEX_MISMATCH", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": row["t"]})
            if bool(row.get("branch_point_by_decision")) != bool(n_dec >= 2):
                bad.append({"code": "BRANCH_BY_DECISION_FLAG_MISMATCH",
                            "world": r["world_id"], "strategy": r["strategy"],
                            "seed": r["seed"], "t": row["t"]})
        inv = all(int(x.get("n_next_distinct", 0)) <= 1 for x in r["rows"])
        if bool(r.get("pre_planned")) != bool(inv):
            bad.append({"code": "PRE_PLANNED_FLAG_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
        if bool(r.get("has_branch_point")) != bool(
                any(x.get("branch_point") for x in r["rows"])):
            bad.append({"code": "HAS_BRANCH_FLAG_MISMATCH", "world": r["world_id"],
                        "strategy": r["strategy"], "seed": r["seed"]})
    agg = bool(all(r.get("pre_planned") for r in rows_all))
    if bool(cf.get("all_pre_planned")) != agg:
        bad.append({"code": "ALL_PRE_PLANNED_MISMATCH",
                    "declared": bool(cf.get("all_pre_planned")), "recomputed": agg})
    n_branch = sum(1 for r in rows_all if r.get("has_branch_point"))
    if int(cf.get("n_runs_with_branch_point", -1)) != n_branch:
        bad.append({"code": "N_BRANCH_RUNS_MISMATCH",
                    "declared": cf.get("n_runs_with_branch_point"),
                    "recomputed": n_branch})
    return bad


def v_branch_caliber(ledger, ctx=None):
    """行动级分支计数必须与逐行重算一致；决策级不得冒充行动级。"""
    cf = ledger.get("counterfactual") or (ctx or {}).get("counterfactual") or {}
    rows_all = cf.get("runs") or []
    if not rows_all:
        raise SystemExit("[s3a] 反事实面为空：拒绝记 PASS")
    bad = []
    n_act = sum(1 for r in rows_all for x in r["rows"] if x.get("branch_point"))
    if int(cf.get("n_action_level_branch_points", -1)) != n_act:
        bad.append({"code": "ACTION_LEVEL_COUNT_MISMATCH",
                    "declared": cf.get("n_action_level_branch_points"),
                    "recomputed": n_act})
    n_dec = sum(1 for r in rows_all for x in r["rows"]
                if x.get("branch_point_by_decision"))
    if int(cf.get("n_decision_level_branch_points", -1)) != n_dec:
        bad.append({"code": "DECISION_LEVEL_COUNT_MISMATCH",
                    "declared": cf.get("n_decision_level_branch_points"),
                    "recomputed": n_dec})
    examples = cf.get("branch_examples") or []
    for ex in examples:
        if not ex.get("branch_point"):
            bad.append({"code": "BRANCH_EXAMPLE_NOT_A_BRANCH_POINT",
                        "world": ex.get("world_id"), "strategy": ex.get("strategy"),
                        "t": ex.get("t")})
    return bad


def v_roster_matches_prereg(ledger, ctx=None):
    pre = (ctx or {}).get("prereg") or {}
    if not pre:
        raise SystemExit("[s3a] 校验缺少 prereg：拒绝记 PASS")
    roster = ledger.get("roster") or {}
    want_w = {w["world_id"] for w in pre["worlds"]}
    want_s = set(pre["strategies"])          # prereg 的 strategies 对象键集（声明面）
    want_k = {int(x) for x in pre["seeds"]}
    bad = []
    if set(roster.get("worlds", [])) != want_w:
        bad.append({"code": "ROSTER_WORLD_DRIFT",
                    "got": sorted(roster.get("worlds", [])), "want": sorted(want_w)})
    if set(roster.get("strategies", [])) != want_s:
        bad.append({"code": "ROSTER_STRATEGY_DRIFT",
                    "got": sorted(roster.get("strategies", [])), "want": sorted(want_s)})
    if {int(x) for x in roster.get("seeds", [])} != want_k:
        bad.append({"code": "ROSTER_SEED_DRIFT",
                    "got": sorted(int(x) for x in roster.get("seeds", [])),
                    "want": sorted(want_k)})
    return bad


def v_claim_scope(ledger, ctx=None):
    ctx = ctx or {}
    phrases = ctx.get("phrases") or ledger.get("forbidden_phrases") or []
    texts = ctx.get("texts") or []
    if not phrases:
        raise SystemExit("[s3a] 空禁用词表：拒绝记 PASS")
    bad = []
    for t in texts:
        for p in phrases:
            if p and p in t:
                bad.append({"code": "FORBIDDEN_CLAIM_PHRASE", "phrase": p})
    return bad


def v_grading_consistency(ledger, ctx=None):
    ctx = ctx or {}
    v = ctx.get("verdict") or {}
    bad = []
    if not v:
        raise SystemExit("[s3a] 判级面为空：拒绝记 PASS")
    cf = ledger.get("counterfactual") or {}
    if bool(cf.get("all_pre_planned")) and str(v.get("grading")) == "ACCEPTED":
        bad.append({"code": "PRE_PLANNED_WRITTEN_AS_ADAPTIVE", "grading": v.get("grading")})
    unver = sum(int(r.get("certificates_unverified", 0)) for r in _iter_runs(ledger))
    if unver and str(v.get("grading")) == "ACCEPTED":
        bad.append({"code": "UNVERIFIED_CERTIFICATES_WITH_ACCEPTED", "n": unver})
    return bad


VALIDATORS = {
    "v_ledger_nonempty": v_ledger_nonempty,
    "v_budget": v_budget,
    "v_no_repeat": v_no_repeat,
    "v_outcome_accounting": v_outcome_accounting,
    "v_denominator": v_denominator,
    "v_certificates": v_certificates,
    "v_config_equal": v_config_equal,
    "v_noise_tier": v_noise_tier,
    "v_duplicate_pair_identity": v_duplicate_pair_identity,
    "v_kernel_identity": v_kernel_identity,
    "v_public_choice": v_public_choice,
    "v_cost_two_sided": v_cost_two_sided,
    "v_adaptivity": v_adaptivity,
    "v_counterfactual_consistency": v_counterfactual_consistency,
    "v_branch_caliber": v_branch_caliber,
    "v_roster_matches_prereg": v_roster_matches_prereg,
    "v_claim_scope": v_claim_scope,
    "v_grading_consistency": v_grading_consistency,
}


def validate_ledger_expanded(ledger, ctx=None) -> dict:
    """跑全部校验器；空面 SystemExit；单器异常记 VALIDATOR_RAISED（不打死）。"""
    out = {}
    for name, fn in VALIDATORS.items():
        try:
            out[name] = fn(ledger, ctx)
        except SystemExit:
            raise
        except Exception as exc:                        # noqa: BLE001
            out[name] = [{"code": "VALIDATOR_RAISED", "validator": name,
                          "error": repr(exc)}]
    return out
