# -*- coding: utf-8 -*-
"""S3-N0 · 反标签选点真闭环（本单唯一新核心模块）。

判据权威 = 本线 ``prereg_s3n.json`` 与任务单 §2.1/§4。本件**只读**消费四个既有模块，
不改 ``active_policy.py`` / ``certificate_solver.py`` / ``sequential_evidence.py`` /
``active_query.py`` 任一字节。

强制语义（与任务单 §2.1 逐条对应）:
  1. 真闭环：第 t 步行动只由 ``H_{t-1}``（公开允许集、已执行行动、成本、公开记录）决定；
     不得预先把整段行动序列排完。本件把「决定」与「执行」物理分开：``choose_action``
     只吃 ``PublicState``，真值只进 ``WorldSim.observe``。
  2. 行动合法：执行前检查 ``used + cost(a) <= budget``；同一行动不得重复；六类结局照样入账。
  3. 停止条件：认证成功 / 双值不可区分 / 预算耗尽 / 动作池耗尽 / 模型冲突 / 计算未解决；
     六类全入账，不删任一类。
  4. 证书：每步用 S2-N0 ``certificate_solver`` 对**当前允许集**发证书；证书必须过
     ``verify_certificate``；空集/冲突不发证书。
  5. 策略：``NEW``（S3-P 见证集 + Hellinger 成本份额 + 对手搜索 + 两步前瞻）、
     ``EC2_TWO_STEP``（active_query EC² 代理 + 两步）、``RANDOM``、``SPACE_FILL``、
     ``ORIENTATION``；规则逐条在 prereg 声明。
  6. 分母：每个 (world, seed, strategy) 一条终末记录；不删超时/冲突/空集；配对同世界同种子。
  7. 主张范围：只报 ``DEVELOPMENT_ONLY``；不写普遍优越 / 成本优势 / 真实收益。

证书协议（prereg v3 §certificate_protocol）:
  允许集 ``A`` → ``decompose_states(A)`` 得互斥子立方；**逐子立方**用 S2-N0
  ``certify`` 发证：载体是「位等式读场景」（``B13_SCENARIO``，根 = 单立方），
  细分裂位一律走 S2-N0 自己的 ``choose_split_bit`` + ``symbolic_envelope`` +
  ``region_label``，直到每个叶区域 decisive；再由 ``certify`` 独立重算外包/见证/覆盖并出证。
  子证书逐张过 ``verify_certificate``；全部通过且标签一致 ⇒ 该允许集获单标签复合证书。
  这样绕开的是 ``certify`` 的接口限制（多立方父节点不得有子区域），**不是**判据：
  标签仍由 S2-N0 的 ``s2_structure`` + ``derivation`` 推出，验证器独立重放。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from fractions import Fraction

import numpy as np

from . import active_policy as ap
from . import active_query as aq
from . import certificate_solver as cs
from . import finite_scope as fs
from . import identification as ident

__all__ = [
    "CLASS_CODE", "BIT_READ_SCENARIO", "STRATEGIES", "OUTCOME_CLASSES", "INF",
    "ClosedLoopError", "PublicState", "WorldSim", "DeterministicEProcess",
    "action_codes", "combine_code", "legal_actions", "issue_certificate",
    "choose_action", "run_policy", "branch_report", "validate_ledger",
    "VALIDATORS", "WATERMARK", "refine_cube",
]

CLASS_CODE = "CLASS_CODE"
BIT_READ_SCENARIO = "BIT_READ_SCENARIO"
INF = "INF"
STRATEGIES = ("NEW", "EC2_TWO_STEP", "RANDOM", "SPACE_FILL", "ORIENTATION")
OUTCOME_CLASSES = ("CERTIFIED_SINGLE", "DUAL_INDISTINGUISHABLE", "BUDGET_EXHAUSTED",
                   "POOL_EXHAUSTED", "MODEL_CONFLICT", "SEARCH_UNRESOLVED")
WATERMARK = {
    "research_line": "S3N_closed_loop_n0",
    "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
    "noise": "N0",
    "dictionary": "COMPLETE_KNOWN",
    "claim_scope": "DEVELOPMENT_ONLY",
    "forbidden": ["真实贯通判定", "普遍策略优越", "成本优势主张", "对外承诺文本"],
}


class ClosedLoopError(RuntimeError):
    """闭环实现缺陷（不是读数）。"""


def _sha12(obj) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                   separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def action_codes(action, n_states: int) -> list:
    """ReadAction → 逐状态观测码（N0 确定性：每状态一个码）。"""
    states = np.arange(int(n_states), dtype=np.int64)
    return [int(x) for x in action.codes(states)]


def combine_code(acc: int, code: int, radix: int) -> int:
    return int(acc) * int(radix) + int(code)


def _popcount(x: int) -> int:
    return bin(int(x)).count("1")


def _claim_of(label: str):
    if label == ident.RegionLabel.ALL_CONNECTED.value:
        return "1"
    if label == ident.RegionLabel.ALL_DISCONNECTED.value:
        return "0"
    return None


# --------------------------------------------------------------------------- 公开面
@dataclasses.dataclass(frozen=True)
class PublicState:
    """策略可见的全部信息（H_{t-1}）。不含真值、不含真值标签。"""
    world_id: str
    step: int
    allowed: tuple
    used: int
    budget: int
    executed: tuple
    legal: tuple
    alpha_seq: str

    def as_dict(self) -> dict:
        return {"world_id": self.world_id, "step": self.step,
                "allowed_n": len(self.allowed), "used": self.used,
                "budget": self.budget, "executed": list(self.executed),
                "legal": list(self.legal), "alpha_seq": self.alpha_seq}


class DeterministicEProcess:
    """N0 精确 e-process：``E_t(theta)=prod_s q_s(Y_s)/p_theta(Y_s)``，Fraction 精确。

    混合 q_s 取「S_{s-1} 上的均匀」；p_theta 在 N0 确定性模型下是点质量 {0,1}。
    ``S_t = {theta: max_{u<=t} E_u < 1/alpha_seq}``。本件只作对账栏，不参与证书判定。
    """

    def __init__(self, states, labels, alpha_seq):
        self.states = tuple(int(s) for s in states)
        self.labels = {int(k): int(v) for k, v in labels.items()}
        self.alpha = Fraction(alpha_seq)
        if not (0 < self.alpha < 1):
            raise ClosedLoopError("alpha 非法: %r" % (alpha_seq,))
        self.threshold = 1 / self.alpha
        self.E = {s: Fraction(1, 1) for s in self.states}
        self.Emax = {s: Fraction(1, 1) for s in self.states}
        self.excluded_at = {}
        self.t = 0
        self.rows = []
        self.conflict = False

    def update(self, s_prev, codes, code) -> dict:
        prev = tuple(int(s) for s in s_prev)
        if not prev:
            raise ClosedLoopError("S_{t-1} 为空：不得更新 e-process")
        matched = [s for s in prev if int(codes[s]) == int(code)]
        q = Fraction(len(matched), len(prev))
        self.t += 1
        if not matched:
            self.conflict = True
        mset = set(matched)
        for s in self.states:
            if s in self.excluded_at:
                continue
            if s not in mset:
                self.E[s] = INF
                self.Emax[s] = INF
                self.excluded_at[s] = self.t
                continue
            e = self.E[s] * q
            self.E[s] = e
            if e > self.Emax[s]:
                self.Emax[s] = e
        S = self.state_set()
        G = tuple(sorted({self.labels[s] for s in S}))
        row = {"t": self.t, "q": str(q), "n_S": len(S), "S": list(S), "Gamma": list(G),
               "status": self.status()}
        self.rows.append(row)
        return row

    def state_set(self) -> tuple:
        return tuple(s for s in self.states
                     if s not in self.excluded_at and self.Emax[s] != INF
                     and self.Emax[s] < self.threshold)

    def status(self) -> str:
        if self.conflict:
            return "MODEL_CONFLICT"
        S = self.state_set()
        if not S:
            return "EMPTY"
        G = {self.labels[s] for s in S}
        if len(G) == 1:
            return "CERTIFIED"
        if len(G) == 2:
            return "DUAL"
        return "SEARCH_UNRESOLVED"


# --------------------------------------------------------------------------- 世界
class WorldSim:
    """声明世界的模拟器：持有真值、公开记录与行动账。

    ``table`` 必须是 S2-N0/finite_scope 的 ``FiniteTable``（含 s2_structure 与 derivation）。
    ``carrier`` 只作登记用途（观测码语义），不再决定查询载体。
    """

    def __init__(self, world_id: str, table: fs.FiniteTable, carrier: str, *,
                 budget: int, alpha_seq, max_steps: int, truth_state: int,
                 orientation_basis: str = "DECLARED_INDEX_ORDER_NO_GEOMETRY",
                 azimuths=None, ab_azimuth=None, max_witness_pairs: int = 128,
                 share_floor: float = 0.1, loss_weights=None,
                 opponent_budget: int = 64, refine_node_cap: int = 8192):
        self.world_id = str(world_id)
        self.table = table
        self.carrier = str(carrier)
        self.budget = int(budget)
        self.alpha_seq = Fraction(alpha_seq)
        self.max_steps = int(max_steps)
        self.truth = int(truth_state)
        self.orientation_basis = str(orientation_basis)
        self.azimuths = dict(azimuths or {})
        self.ab_azimuth = ab_azimuth
        self.max_witness_pairs = int(max_witness_pairs)
        self.share_floor = float(share_floor)
        self.opponent_budget = int(opponent_budget)
        self.refine_node_cap = int(refine_node_cap)
        self.loss_weights = loss_weights or aq.LossWeights(
            1.0, 1.0, 0.5, "prereg 声明：0/1 对称损失 + 弃权 0.5")
        self.n_states = int(table.n_states)
        self.n_bits = int(table.n_bits)
        self.label_query = sorted(table.labels)[0]
        self.labels = {s: int(table.labels[self.label_query][s])
                       for s in range(self.n_states)}
        self.initial_actions = tuple(table.baseline)
        self.pool = tuple(table.actions)
        aid_of = {a.action_id: a for a in self.initial_actions + self.pool}
        self.action_of = dict(aid_of)
        self.all_ids = tuple(aid_of)
        self.codes = {aid: action_codes(a, self.n_states) for aid, a in aid_of.items()}
        self.radix = {aid: int(a.radix) for aid, a in aid_of.items()}
        self.cost = {a.action_id: int(a.cost_int) for a in self.pool}
        self.read_mask = {a.action_id: int(a.read_mask) for a in self.pool}
        self.pool_ids = tuple(a.action_id for a in self.pool)
        self.struct = cs.structure_of(table)
        if set(self.pool_ids) != set(self.cost):
            raise ClosedLoopError("行动池与成本表不一致")
        init_ids = {a.action_id for a in self.initial_actions}
        if init_ids & set(self.pool_ids):
            raise ClosedLoopError("行动同时是 initial read 与闭环行动")
        allowed = tuple(range(self.n_states))
        for a in self.initial_actions:
            allowed = self.restrict(allowed, a.action_id, self.codes[a.action_id][self.truth])
        self.initial_allowed = allowed
        if not allowed:
            raise ClosedLoopError("%s: 初始允许集为空（观测不在前向像内）" % self.world_id)

    # -- 公开面 ------------------------------------------------------------
    def view(self, cube: cs.Cube, tag: str):
        """由单个子立方造求解器认得的表视图 + query（位等式读场景）。"""
        reads = [[int(b), int((cube.fixed_vals >> b) & 1)]
                 for b in range(self.n_bits) if (cube.fixed_mask >> b) & 1]
        scen = {"reads": reads, "observation_in_forward_image": True}
        tags = dict(self.table.tags)
        tags["s2_queries"] = {"kind": "B13_SCENARIO", "scenarios": {str(tag): scen}}
        view = dataclasses.replace(self.table, tags=tags)
        return view, "B13:scenario:%s" % str(tag)

    def observe(self, action_id: str) -> int:
        """执行侧：真值 + N0 确定性观测律 → 公开码。"""
        return int(self.codes[action_id][self.truth])

    def restrict(self, allowed, action_id: str, code: int) -> tuple:
        c = self.codes[action_id]
        return tuple(s for s in allowed if c[s] == int(code))

    def splits(self, allowed, action_id: str) -> bool:
        """该行动能否把当前允许集拆成 ≥2 个非空观测类（即还能再读出新信息）。"""
        return len({self.codes[action_id][s] for s in allowed}) > 1

    def separates(self, allowed, action_id: str) -> bool:
        """该行动能否把当前允许集里任一对立标签对拆开。"""
        c = self.codes[action_id]
        by_code = {}
        for s in allowed:
            by_code.setdefault(c[s], []).append(self.labels[s])
        return any(len(set(v)) > 1 for v in by_code.values())

    def public(self, allowed, used, executed, step) -> PublicState:
        done = set(executed)
        return PublicState(self.world_id, int(step), tuple(allowed), int(used),
                           self.budget, tuple(executed),
                           tuple(a for a in self.pool_ids if a not in done
                                 and used + self.cost[a] <= self.budget),
                           str(self.alpha_seq))


def legal_actions(sim: WorldSim, used: int, executed) -> tuple:
    done = set(executed)
    return tuple(a for a in sim.pool_ids if a not in done and used + sim.cost[a] <= sim.budget)


# --------------------------------------------------------------------------- 证书
def refine_cube(sim: WorldSim, cube: cs.Cube, top: str = "R0") -> list:
    """单立方内按 S2-N0 的 split-bit / envelope / region_label 细化到叶区域 decisive。"""
    n_bits = sim.n_bits
    struct = sim.struct
    regions = []
    seq = 1
    stack = [cs.Region(top, None, 0, (cube,))]
    while stack:
        r = stack.pop()
        regions.append(r)
        if len(regions) > sim.refine_node_cap:
            raise ClosedLoopError("%s: 细化节点超上限 %d" % (sim.world_id, sim.refine_node_cap))
        env = cs.symbolic_envelope(struct, r.cubes, n_bits)
        if _claim_of(ident.region_label(env)) is not None:
            continue
        b = cs.choose_split_bit(struct, r.cubes, env, n_bits)
        c0 = r.cubes[0]
        m0 = c0.fixed_mask | (1 << b)
        v0 = c0.fixed_vals & ~(1 << b)
        kids = [cs.Region("R%d" % seq, r.region_id, r.depth + 1, (cs.Cube(m0, v0),)),
                cs.Region("R%d" % (seq + 1), r.region_id, r.depth + 1,
                          (cs.Cube(m0, v0 | (1 << b)),))]
        seq += 2
        stack.extend(reversed(kids))
    return regions


def _certify_one_cube(sim: WorldSim, cube: cs.Cube, tag: str) -> dict:
    rec = {"cube": cube.to_dict(), "state_count": int(cube.count(sim.n_bits)),
           "issued": False, "verified": None, "label_claim": None,
           "reason": None, "regions": 0, "certificate_id": None,
           "certificate_sha12": None, "certify_seconds": 0.0, "verify_seconds": 0.0,
           "certificate": None}
    t0 = time.perf_counter()
    try:
        view, qid = sim.view(cube, tag)
        roots = cs.resolve_query_cubes(view, qid)
        if tuple(roots) != (cube,):
            raise ClosedLoopError("%s: 子立方查询重算不符 %r != %r"
                                  % (sim.world_id, roots, cube))
        tree = refine_cube(sim, cube, top="R0")
        res = cs.certify(view, qid, tree, certificate_id="S3NC:%s:%s" % (sim.world_id, tag))
    except Exception as exc:                        # noqa: BLE001
        rec["reason"] = "CERTIFY_RAISED: %r" % (exc,)
        rec["certify_seconds"] = round(time.perf_counter() - t0, 6)
        rec["status"] = "EXCEPTION"
        return rec
    rec["certify_seconds"] = round(time.perf_counter() - t0, 6)
    rec["regions"] = len(tree)
    if not isinstance(res, cs.Certificate):
        rec["status"] = "UNRESOLVED"
        rec["reason"] = str(getattr(res, "reason", "UNKNOWN"))
        return rec
    cert = res.to_dict()
    rec.update({"issued": True, "status": "CERTIFICATE",
                "label_claim": str(cert.get("label_claim")),
                "certificate_id": cert.get("certificate_id"),
                "certificate_sha12": _sha12(cert), "certificate": cert})
    t1 = time.perf_counter()
    try:
        rep = cs.verify_certificate(view, qid, cert)
        rec["verified"] = bool(rep.get("pass"))
        rec["verify_reasons"] = list(rep.get("reasons", []))
    except Exception as exc:                        # noqa: BLE001
        rec["verified"] = False
        rec["verify_reasons"] = ["VERIFY_RAISED: %r" % (exc,)]
    rec["verify_seconds"] = round(time.perf_counter() - t1, 6)
    return rec


def issue_certificate(sim: WorldSim, allowed, tag: str) -> dict:
    """对当前允许集发复合证书（逐子立方 S2-N0 出证 + 原验证器验收）。"""
    out = {"step_tag": str(tag), "kind": "CUBE_PARTITION_CERTIFICATE",
           "issued": False, "status": "NOT_ISSUED", "reason": None,
           "label_claim": None, "allowed_n": len(allowed),
           "allowed_sha12": _sha12(list(allowed)),
           "n_cubes": 0, "verified": None, "verify_reasons": [],
           "certificate_id": None, "certificate_sha12": None,
           "certify_seconds": 0.0, "verify_seconds": 0.0,
           "cubes": [], "certificates": [], "certificate": None}
    if not allowed:
        out["status"] = "NOT_ISSUED_EMPTY"
        out["reason"] = "MODEL_CONFLICT_EMPTY_ALLOWED"
        return out
    cubes = cs.decompose_states(list(allowed), sim.n_bits)
    out["n_cubes"] = len(cubes)
    if len(cs.region_states(cubes, sim.n_bits)) != len(allowed):
        raise ClosedLoopError("%s: 子立方分解计数不符" % sim.world_id)
    for i, cube in enumerate(cubes):
        r = _certify_one_cube(sim, cube, "%s_c%d" % (tag, i))
        out["cubes"].append({k: r.get(k) for k in
                             ("cube", "state_count", "issued", "verified",
                              "label_claim", "reason", "regions", "certificate_id",
                              "certificate_sha12", "status")})
        out["certify_seconds"] += r["certify_seconds"]
        out["verify_seconds"] += r.get("verify_seconds", 0.0)
        if r.get("certificate"):
            out["certificates"].append(r["certificate"])
    bad = [r for r in out["cubes"] if not r["issued"]]
    if bad:
        out["status"] = "UNRESOLVED"
        out["reason"] = "CUBE_NOT_DECISIVE:%s" % sorted({str(r["reason"]) for r in bad})
        return out
    claims = {r["label_claim"] for r in out["cubes"]}
    if len(claims) != 1:
        out["status"] = "UNRESOLVED"
        out["reason"] = "MIXED_LABELS_ACROSS_CUBES:%s" % sorted(claims)
        return out
    unver = [r for r in out["cubes"] if not r["verified"]]
    out["issued"] = True
    out["status"] = "CERTIFICATE"
    out["label_claim"] = str(claims.pop())
    out["verified"] = not unver
    out["verify_reasons"] = ["CUBE_NOT_VERIFIED:%s" % r["certificate_id"] for r in unver]
    out["certificate_id"] = "S3NC-PART:%s:%s" % (sim.world_id, tag)
    out["certificate_sha12"] = _sha12({"world": sim.world_id, "tag": str(tag),
                                       "allowed": list(allowed),
                                       "cubes": [r["certificate_sha12"] for r in out["cubes"]]})
    out["certificate"] = {
        "schema": "s3n.cube_partition_certificate.v1",
        "certificate_id": out["certificate_id"],
        "world_id": sim.world_id, "tag": str(tag),
        "allowed_state_ids": list(allowed),
        "n_cubes": len(cubes), "label_claim": out["label_claim"],
        "sub_certificates": out["certificates"],
        "sub_certificate_sha12": [r["certificate_sha12"] for r in out["cubes"]],
        "verified_by": "fractureflow.certificate_solver.verify_certificate",
        "all_sub_verified": bool(out["verified"]),
        "watermark": dict(WATERMARK)}
    return out


# --------------------------------------------------------------------------- 策略
def _witness_pairs(sim: WorldSim, allowed) -> list:
    by_label = {0: [], 1: []}
    for s in allowed:
        by_label[sim.labels[s]].append(s)
    pairs = [(int(i), int(j)) for i in by_label[0] for j in by_label[1]]
    pairs.sort()
    return pairs


def _action_dists(sim: WorldSim, allowed, pool) -> dict:
    return {a: {int(s): {int(sim.codes[a][s]): 1.0} for s in allowed} for a in pool}


def _tie_pick(scored: dict, pool, sim: WorldSim) -> str:
    return str(sorted(pool, key=lambda a: (-float(scored[a]), sim.cost[a], a))[0])


def _choose_new(sim: WorldSim, pub: PublicState, rng) -> dict:
    part = pub.allowed
    pairs_all = _witness_pairs(sim, part)
    if not pairs_all:
        raise ClosedLoopError("%s NEW: 允许集内无对立标签对" % sim.world_id)
    pairs = pairs_all[:sim.max_witness_pairs]
    label_of = {int(s): int(sim.labels[s]) for s in part}
    w = ap.WitnessSet(tuple(pairs), labels=label_of, name="S3N_W",
                      provenance="允许集内对立标签对（字典序前 N）")
    pool = list(pub.legal)
    dists = _action_dists(sim, part, pool)
    costs = {a: float(sim.cost[a]) for a in pool}
    opp = ap.opponent_search(w, dists, label_of, budget=sim.opponent_budget,
                             stop_rule="EXHAUST_PAIRS_UP_TO_BUDGET")
    extra = [tuple(p) for p in opp["new_pairs_this_strictly_harder_or_equal"]
             if tuple(p) not in set(pairs)]
    merged = sorted(set(list(pairs) + extra))[:sim.max_witness_pairs]
    w2 = ap.WitnessSet(tuple(merged), labels=label_of, name="S3N_W_HARDENED",
                       provenance="对手搜索扩集（cap=%d）" % sim.max_witness_pairs)
    lam = ap.cost_share(w2, dists, costs, share_floor=sim.share_floor)
    keep = ap.top_k_filter(lam["cost_share"], 4)
    kept = [a for a in keep["kept"] if a in dists]
    if not kept:
        raise ClosedLoopError("%s NEW: 前筛后无候选" % sim.world_id)
    sub = {a: dists[a] for a in kept}
    sub_costs = {a: costs[a] for a in kept}
    ts = ap.two_step_lookahead(w2, sub, sub_costs, None, ks_compared=(4, None),
                               share_floor=sim.share_floor)
    pick = _tie_pick(ts["two_step_scores"], kept, sim)
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


def _choose_ec2(sim: WorldSim, pub: PublicState, rng) -> dict:
    part = pub.allowed
    pool = list(pub.legal)
    belief = aq.Belief.build(tuple(part), [Fraction(1, len(part))] * len(part),
                             lambda s: sim.labels[s], "S3N:%s" % sim.world_id)
    acts = [aq.CostAction(a, int(sim.cost[a])) for a in pool]
    ofn = lambda s, aid: int(sim.codes[aid][int(s)])          # noqa: E731
    ec2 = aq.ec2_proxy_ranking(belief, acts, ofn)
    rows = sorted(ec2["rows"], key=lambda r: (-float(r.score_unit_cost), r.action_id))
    cands = [r.action_id for r in rows[:4]]
    if not cands:
        raise ClosedLoopError("%s EC2: 代理排序为空" % sim.world_id)
    sub = [aq.CostAction(a, int(sim.cost[a])) for a in cands]
    ts = aq.two_step_lookahead({sim.world_id: belief}, sim.loss_weights, sub, ofn,
                               candidates=cands)
    pick = str(ts["chosen"])
    if pick not in cands:
        raise ClosedLoopError("%s EC2: 两步选了候选池外的 %r" % (sim.world_id, pick))
    return {"action_id": pick,
            "detail": {"ec2_order": [r.action_id for r in rows],
                       "ec2_scores": {r.action_id: float(r.score_unit_cost) for r in rows},
                       "ec2_n_cross_pairs": ec2["n_cross_pairs"],
                       "ec2_zero_mass_actions": ec2["zero_mass_actions"],
                       "candidates": cands, "two_step_chosen": pick,
                       "two_step_rows": len(ts["rows"]), "planning_pool": pool}}


def _choose_random(sim: WorldSim, pub: PublicState, rng) -> dict:
    pool = list(pub.legal)
    if not pool:
        raise ClosedLoopError("%s RANDOM: 空合法池" % sim.world_id)
    idx = int(rng.randrange(len(pool)))
    return {"action_id": pool[idx], "detail": {"draw_index": idx, "pool": pool}}


def _choose_space_fill(sim: WorldSim, pub: PublicState, rng, read_so_far: int) -> dict:
    pool = list(pub.legal)
    best = sorted(pool, key=lambda a: (-_popcount(sim.read_mask[a] & ~int(read_so_far)),
                                       sim.cost[a], a))[0]
    return {"action_id": best,
            "detail": {"newly_read_bits": {a: _popcount(sim.read_mask[a] & ~int(read_so_far))
                                          for a in pool},
                       "read_mask_so_far": int(read_so_far), "planning_pool": pool}}


def _circ(a, b) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _choose_orientation(sim: WorldSim, pub: PublicState, rng) -> dict:
    pool = list(pub.legal)
    if sim.azimuths and sim.ab_azimuth is not None:
        scored = {a: -_circ(sim.azimuths[a], sim.ab_azimuth) for a in pool}
        best = sorted(pool, key=lambda a: (-scored[a], sim.cost[a], a))[0]
        return {"action_id": best,
                "detail": {"basis": "AZIMUTH_TO_AB_AXIS",
                           "ab_azimuth_deg": float(sim.ab_azimuth),
                           "azimuth_deg": {a: float(sim.azimuths[a]) for a in pool},
                           "circular_distance_deg": {a: _circ(sim.azimuths[a],
                                                              sim.ab_azimuth)
                                                     for a in pool},
                           "planning_pool": pool}}
    order = [a for a in sim.pool_ids if a in set(pool)]
    return {"action_id": order[0],
            "detail": {"basis": "DECLARED_INDEX_ORDER_NO_GEOMETRY",
                       "order": order, "planning_pool": pool}}


def choose_action(strategy: str, sim: WorldSim, pub: PublicState, rng,
                  read_so_far: int = 0) -> dict:
    """**只看 PublicState**（H_{t-1}）决定下一步行动。"""
    if strategy not in STRATEGIES:
        raise ClosedLoopError("未知策略 %r" % (strategy,))
    if not pub.legal:
        raise ClosedLoopError("%s/%s: 无合法行动却仍要求选点" % (sim.world_id, strategy))
    if strategy in ("NEW", "EC2_TWO_STEP") and len({sim.labels[s] for s in pub.allowed}) == 1:
        out = _choose_space_fill(sim, pub, rng, read_so_far)
        out["detail"]["fallback"] = "NO_OPPOSITE_PAIRS_LABELS_UNIFORM"
        out["detail"]["fallback_rule"] = "SPACE_FILL（prereg v2 声明）"
        return out
    if strategy == "NEW":
        return _choose_new(sim, pub, rng)
    if strategy == "EC2_TWO_STEP":
        return _choose_ec2(sim, pub, rng)
    if strategy == "RANDOM":
        return _choose_random(sim, pub, rng)
    if strategy == "SPACE_FILL":
        return _choose_space_fill(sim, pub, rng, read_so_far)
    return _choose_orientation(sim, pub, rng)


# --------------------------------------------------------------------------- 闭环
def _stop_and_why(sim, allowed, used, executed, step, max_steps, cert) -> dict:
    """按 prereg 声明序判定终末类。返回 {} 表示继续。"""
    if not allowed:
        return {"class": "MODEL_CONFLICT", "subreason": "ALLOWED_EMPTY",
                "detail": {"at_step": step}}
    if cert["issued"] and cert["verified"]:
        return {"class": "CERTIFIED_SINGLE", "subreason": "CERTIFICATE_VERIFIED",
                "detail": {"label_claim": cert["label_claim"],
                           "certificate_id": cert["certificate_id"],
                           "n_cubes": cert["n_cubes"]}}
    if cert["issued"] and not cert["verified"]:
        return {"class": "SEARCH_UNRESOLVED", "subreason": "CERTIFICATE_UNVERIFIED",
                "detail": {"verify_reasons": list(cert["verify_reasons"])}}
    if str(cert.get("status")) == "EXCEPTION":
        return {"class": "SEARCH_UNRESOLVED", "subreason": "CERTIFICATE_EXCEPTION",
                "detail": {"reason": cert["reason"]}}
    legal = legal_actions(sim, used, executed)
    if not legal:
        if len(executed) >= len(sim.pool_ids):
            return {"class": "POOL_EXHAUSTED", "subreason": "NO_UNEXECUTED_ACTION",
                    "detail": {"executed": list(executed)}}
        return {"class": "BUDGET_EXHAUSTED", "subreason": "NO_AFFORDABLE_ACTION",
                "detail": {"used": used, "budget": sim.budget,
                           "remaining": [a for a in sim.pool_ids
                                         if a not in set(executed)]}}
    if step >= max_steps:
        return {"class": "SEARCH_UNRESOLVED", "subreason": "STEP_CAP",
                "detail": {"max_steps": max_steps}}
    if not any(sim.splits(allowed, a) for a in legal):
        labels = sorted({sim.labels[s] for s in allowed})
        if len(labels) == 1:
            return {"class": "SEARCH_UNRESOLVED", "subreason": "PART_FROZEN_UNCERTIFIED",
                    "detail": {"legal": list(legal), "allowed_n": len(allowed),
                               "labels": labels}}
        return {"class": "DUAL_INDISTINGUISHABLE",
                "subreason": "NO_LEGAL_ACTION_SPLITS_ALLOWED_SET",
                "detail": {"legal": list(legal), "allowed_n": len(allowed),
                           "labels": labels}}
    return {}


def run_policy(strategy: str, model: WorldSim, action_pool, budget: int, alpha_seq,
               rng, max_steps: int) -> dict:
    """跑一条闭环轨迹。策略选择只吃 PublicState；真值只在 ``sim.observe`` 内出现。"""
    if strategy not in STRATEGIES:
        raise ClosedLoopError("未知策略 %r" % (strategy,))
    pool = tuple(str(a) for a in action_pool)
    if set(pool) != set(model.pool_ids):
        raise ClosedLoopError("%s: action_pool 与声明池不一致" % model.world_id)
    if int(budget) != model.budget:
        raise ClosedLoopError("%s: budget %r != 声明 %r（禁换预算）"
                              % (model.world_id, budget, model.budget))
    if Fraction(alpha_seq) != model.alpha_seq:
        raise ClosedLoopError("%s: alpha %r != 声明 %r（禁换 α）"
                              % (model.world_id, alpha_seq, model.alpha_seq))
    if int(max_steps) != model.max_steps:
        raise ClosedLoopError("%s: max_steps %r != 声明 %r"
                              % (model.world_id, max_steps, model.max_steps))
    t_start = time.perf_counter()
    t_choice = t_cert = t_verify = 0.0
    n_choose = n_cert = n_verify = 0

    allowed = tuple(model.initial_allowed)
    used = 0
    executed = ()
    read_so_far = 0
    for a in model.initial_actions:
        read_so_far |= int(a.read_mask)
    eproc = DeterministicEProcess(model.initial_allowed,
                                  {s: model.labels[s] for s in model.initial_allowed},
                                  model.alpha_seq)
    steps = []
    certs = []

    def _cert(tag):
        nonlocal t_cert, t_verify, n_cert, n_verify
        rec = issue_certificate(model, allowed, tag)
        n_cert += 1
        t_cert += rec["certify_seconds"]
        if rec["issued"]:
            n_verify += 1
            t_verify += rec["verify_seconds"]
        return rec

    cert0 = _cert("t0")
    steps.append({"t": 0, "action_id": None, "cost": 0, "outcome": None,
                  "allowed_before": len(model.initial_allowed),
                  "allowed_after": len(allowed), "allowed_sha12": _sha12(list(allowed)),
                  "allowed": list(allowed) if len(allowed) <= 256 else None,
                  "certificate": _cert_summary(cert0), "eprocess": None,
                  "read_mask_so_far": int(read_so_far)})
    certs.append(cert0)
    stop = _stop_and_why(model, allowed, used, executed, 0, max_steps, cert0)
    if not stop and cert0["issued"]:
        raise ClosedLoopError("%s: 证书已发但未判终末" % model.world_id)

    while not stop:
        step = len([x for x in steps if x["action_id"] is not None])
        pub = model.public(allowed, used, executed, step)
        ts = time.perf_counter()
        choice = choose_action(strategy, model, pub, rng, read_so_far)
        t_choice += time.perf_counter() - ts
        n_choose += 1
        aid = str(choice["action_id"])
        if aid not in pub.legal:
            raise ClosedLoopError("%s/%s: 选中非法行动 %r（合法池 %r）"
                                  % (model.world_id, strategy, aid, pub.legal))
        before = used
        used += model.cost[aid]
        if used > model.budget:
            raise ClosedLoopError("%s: 超预算 %d > %d" % (model.world_id, used, model.budget))
        if aid in executed:
            raise ClosedLoopError("%s: 重复行动 %r" % (model.world_id, aid))
        code = model.observe(aid)
        executed = executed + (aid,)
        read_so_far |= int(model.read_mask[aid])
        allowed_prev = allowed
        allowed = model.restrict(allowed_prev, aid, code)
        erow = eproc.update(allowed_prev, model.codes[aid], code)
        cert = _cert("t%d" % (step + 1))
        steps.append({"t": step + 1, "action_id": aid, "cost": int(model.cost[aid]),
                      "used_after": int(used), "used_before": int(before),
                      "outcome": int(code),
                      "allowed_before": len(allowed_prev), "allowed_after": len(allowed),
                      "allowed_sha12": _sha12(list(allowed)),
                      "allowed": list(allowed) if len(allowed) <= 256 else None,
                      "certificate": _cert_summary(cert), "eprocess": erow,
                      "choice_detail": choice["detail"],
                      "read_mask_so_far": int(read_so_far)})
        certs.append(cert)
        stop = _stop_and_why(model, allowed, used, executed, step + 1, max_steps, cert)

    elapsed = time.perf_counter() - t_start
    answer = None
    for s in steps:
        c = s["certificate"]
        if c["issued"] and c["verified"]:
            answer = int(c["label_claim"])
            break
    return {
        "schema": "s3n.trace.v1",
        "world_id": model.world_id,
        "strategy": strategy,
        "truth_state": int(model.truth),
        "truth_label": int(model.labels[model.truth]),
        "carrier": model.carrier,
        "budget": int(model.budget),
        "alpha_seq": str(model.alpha_seq),
        "max_steps": int(max_steps),
        "initial_allowed": len(model.initial_allowed),
        "steps": steps,
        "stop": stop,
        "used": int(used),
        "n_steps": len([x for x in steps if x["action_id"] is not None]),
        "answer": answer,
        "correct": (None if answer is None
                    else bool(answer == int(model.labels[model.truth]))),
        "certificate_attempts": int(n_cert),
        "certificates_issued": int(sum(1 for c in certs if c["issued"])),
        "certificates_verified": int(sum(1 for c in certs if c["issued"] and c["verified"])),
        "certificates_unverified": int(sum(1 for c in certs
                                           if c["issued"] and not c["verified"])),
        "certificate_not_issued_empty": int(sum(1 for c in certs
                                                if c["status"] == "NOT_ISSUED_EMPTY")),
        "eprocess_equals_allowed": bool(all(
            s["eprocess"] is None or s["allowed"] is None or
            tuple(sorted(s["eprocess"]["S"])) == tuple(sorted(s["allowed"]))
            for s in steps)),
        "eprocess_rows": eproc.rows,
        "cost": {"observed": int(used),
                 "compute_seconds": round(elapsed, 6),
                 "choice_seconds": round(t_choice, 6),
                 "certify_seconds": round(t_cert, 6),
                 "verify_seconds": round(t_verify, 6),
                 "n_choose_calls": int(n_choose),
                 "n_certify_calls": int(n_cert),
                 "n_verify_calls": int(n_verify),
                 "basis": "MEASURED_ONLY"},
        "certificates": [_cert_stub(c) for c in certs],
        "_certificates_full": [c["certificate"] for c in certs if c["issued"]],
        "watermark": dict(WATERMARK),
    }


def _cert_summary(rec: dict) -> dict:
    return {"issued": bool(rec["issued"]), "status": rec["status"],
            "reason": rec["reason"], "label_claim": rec["label_claim"],
            "verified": rec["verified"], "verify_reasons": list(rec["verify_reasons"]),
            "certificate_id": rec["certificate_id"],
            "certificate_sha12": rec["certificate_sha12"],
            "allowed_n": int(rec["allowed_n"]), "n_cubes": int(rec["n_cubes"]),
            "allowed_sha12": rec["allowed_sha12"],
            "certify_seconds": round(rec["certify_seconds"], 6),
            "verify_seconds": round(rec["verify_seconds"], 6)}


def _cert_stub(rec: dict) -> dict:
    out = _cert_summary(rec)
    out["tag"] = rec["step_tag"]
    return out


# --------------------------------------------------------------------------- 反事实
def _random_prefix_rng(sim: WorldSim, acted, policy_seed: int, k: int):
    """重建 RANDOM 第 k 次抽取前的 rng（前缀池由 recorded 行动账复原，与结局无关）。"""
    import random as _random

    r = _random.Random(policy_seed)
    used = 0
    executed = ()
    for i in range(k):
        pool = [a for a in sim.pool_ids if a not in set(executed)
                and used + sim.cost[a] <= sim.budget]
        if not pool:
            return None
        pick = pool[int(r.randrange(len(pool)))]
        used += sim.cost[pick]
        executed = executed + (pick,)
    return r


def branch_report(strategy: str, sim: WorldSim, trace: dict, policy_seed: int) -> dict:
    """逐步枚举当前允许集下的全部分支结局，用同一公开历史重算下一步行动。

    分支重算前对 rng 取 getstate()/setstate() 快照，保证分支只用同一起点。
    """
    import random as _random

    rows = []
    allowed = tuple(sim.initial_allowed)
    used = 0
    executed = ()
    acted = [x for x in trace["steps"] if x["action_id"] is not None]
    for k, st in enumerate(acted):
        aid = st["action_id"]
        codes_here = sorted({int(sim.codes[aid][s]) for s in allowed}) if allowed else []
        next_actions = {}
        terminal_by_outcome = {}
        read_so_far = 0
        for a in sim.initial_actions:
            read_so_far |= int(a.read_mask)
        for a in executed:
            read_so_far |= int(sim.read_mask[a])
        base_rng = _random_prefix_rng(sim, acted, policy_seed, k) \
            if strategy == "RANDOM" else None
        for code in codes_here:
            nxt = sim.restrict(allowed, aid, code)
            ex2 = executed + (aid,)
            used2 = used + sim.cost[aid]
            # 分支是否在该步就终末（终末分支没有"下一步行动"）
            cb = issue_certificate(sim, nxt, "%s_b%s" % (trace.get("tag", "cf"),
                                                         "%d_%d" % (k, code)))
            sb = _stop_and_why(sim, nxt, used2, ex2, k + 1, sim.max_steps, cb)
            if sb:
                terminal_by_outcome[int(code)] = str(sb["class"])
                next_actions[int(code)] = None
                continue
            pub = sim.public(nxt, used2, ex2, k + 1)
            if not pub.legal:
                terminal_by_outcome[int(code)] = "NO_LEGAL"
                next_actions[int(code)] = None
                continue
            if strategy == "RANDOM":
                if base_rng is None:
                    next_actions[int(code)] = None
                    continue
                rb = _random.Random()
                rb.setstate(base_rng.getstate())
                bp = list(pub.legal)
                next_actions[int(code)] = str(bp[int(rb.randrange(len(bp)))])
            else:
                ch = choose_action(strategy, sim, pub, _random.Random(0), read_so_far)
                next_actions[int(code)] = str(ch["action_id"])
        acts_only = sorted({v for v in next_actions.values() if v is not None})
        decisions = sorted({(v if v is not None else "STOP:" + str(terminal_by_outcome.get(kk)))
                            for kk, v in next_actions.items()})
        rows.append({"t": st["t"], "action_id": aid,
                     "n_outcomes": len(codes_here),
                     "observed_outcome": st["outcome"],
                     "next_action_by_outcome": {str(kk): v
                                                for kk, v in next_actions.items()},
                     "terminal_by_outcome": {str(kk): v
                                             for kk, v in terminal_by_outcome.items()},
                     "n_next_distinct": len(acts_only),
                     "n_next_distinct_decisions": len(decisions),
                     "branch_point": bool(len(acts_only) >= 2),
                     "branch_point_by_decision": bool(len(decisions) >= 2)})
        allowed = sim.restrict(allowed, aid, st["outcome"])
        used += sim.cost[aid]
        executed = executed + (aid,)
    return {"world_id": sim.world_id, "strategy": strategy,
            "seed": int(trace.get("seed", -1)), "policy_seed": int(policy_seed),
            "rows": rows,
            "has_branch_point": bool(any(r["branch_point"] for r in rows)),
            "has_branch_point_by_decision": bool(
                any(r["branch_point_by_decision"] for r in rows)),
            "all_steps_invariant": bool(all(r["n_next_distinct"] <= 1 for r in rows)),
            "pre_planned": bool(all(r["n_next_distinct"] <= 1 for r in rows))}


# --------------------------------------------------------------------------- 校验（控制面用）
def _iter_runs(ledger):
    return list(ledger.get("runs", []))


def v_ledger_nonempty(ledger, ctx=None):
    if not _iter_runs(ledger):
        raise SystemExit("[s3n] 空账：拒绝记 PASS")
    n_w = len({r["world_id"] for r in _iter_runs(ledger)})
    n_s = len({r["strategy"] for r in _iter_runs(ledger)})
    n_k = len({int(r["seed"]) for r in _iter_runs(ledger)})
    if n_w == 0 or n_s == 0 or n_k == 0:
        raise SystemExit("[s3n] 空世界/空策略/空种子面：拒绝记 PASS")
    return []


def v_budget(ledger, ctx=None):
    bad = []
    for r in _iter_runs(ledger):
        used = 0
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            if used + s["cost"] > r["budget"]:
                bad.append({"code": "OVER_BUDGET", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"],
                            "t": s["t"], "used": used, "cost": s["cost"],
                            "budget": r["budget"]})
            used += s["cost"]
        if used != r["used"]:
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
            if s.get("outcome") is None or s.get("used_after") is None:
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
        raise SystemExit("[s3n] 声明名册为空：拒绝记 PASS")
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
            if c["issued"] and not int(c.get("n_cubes", 0)):
                bad.append({"code": "CERTIFICATE_WITHOUT_CUBES", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
            if c["issued"]:
                terminal_t = s["t"]
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
        raise SystemExit("[s3n] 空配置面：拒绝记 PASS")
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


def v_public_choice(ledger, ctx=None):
    """逐步用公开历史重算策略选择；不可复现即判红（禁真值选点）。"""
    import random as _random

    sims = (ctx or {}).get("sims") or {}
    if not sims:
        raise SystemExit("[s3n] 校验缺少世界模拟器：拒绝记 PASS")
    bad = []
    for r in _iter_runs(ledger):
        sim = sims.get(r["world_id"])
        if sim is None:
            raise SystemExit("[s3n] 未知世界 %r" % (r["world_id"],))
        rng = _random.Random()
        rng.seed(int(r["policy_seed"]))
        allowed = tuple(sim.initial_allowed)
        used = 0
        executed = ()
        read_so_far = 0
        for a in sim.initial_actions:
            read_so_far |= int(a.read_mask)
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            pub = sim.public(allowed, used, executed, s["t"] - 1)
            ch = choose_action(r["strategy"], sim, pub, rng, read_so_far)
            if str(ch["action_id"]) != str(s["action_id"]):
                bad.append({"code": "ACTION_NOT_REPRODUCIBLE_FROM_PUBLIC_HISTORY",
                            "world": r["world_id"], "strategy": r["strategy"],
                            "seed": r["seed"], "t": s["t"],
                            "recorded": s["action_id"], "recomputed": ch["action_id"]})
            if s.get("outcome") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "world": r["world_id"],
                            "strategy": r["strategy"], "seed": r["seed"], "t": s["t"]})
                break
            allowed = sim.restrict(allowed, s["action_id"], s["outcome"])
            used += sim.cost[s["action_id"]]
            executed = executed + (s["action_id"],)
            read_so_far |= int(sim.read_mask[s["action_id"]])
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
    """G1：至少一个分支点；全案 PRE_PLANNED 即判红。"""
    cf = ledger.get("counterfactual") or {}
    runs = cf.get("runs") or []
    if not runs:
        raise SystemExit("[s3n] 反事实面为空：拒绝记 PASS")
    if not any(r.get("has_branch_point") for r in runs):
        return [{"code": "PRE_PLANNED_NO_BRANCH_POINT", "runs": len(runs)}]
    return []


def v_claim_scope(ledger, ctx=None):
    """报告面禁用词扫描（开发对照禁写成确认实验/普遍优越）。"""
    ctx = ctx or {}
    phrases = ctx.get("phrases") or ledger.get("forbidden_phrases") or []
    texts = ctx.get("texts") or []
    if not phrases:
        raise SystemExit("[s3n] 空禁用词表：拒绝记 PASS")
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
        raise SystemExit("[s3n] 判级面为空：拒绝记 PASS")
    cf = ledger.get("counterfactual") or {}
    if bool(cf.get("all_pre_planned")) and str(v.get("grading")) == "ACCEPTED":
        bad.append({"code": "PRE_PLANNED_WRITTEN_AS_ADAPTIVE", "grading": v.get("grading")})
    unver = sum(int(r.get("certificates_unverified", 0)) for r in _iter_runs(ledger))
    if unver and str(v.get("grading")) == "ACCEPTED":
        bad.append({"code": "UNVERIFIED_CERTIFICATES_WITH_ACCEPTED", "n": unver})
    return bad


def v_counterfactual_consistency(ledger, ctx=None):
    """反事实账自洽：n_next_distinct / branch_point / 汇总标志必须与逐分支明细一致。"""
    cf = ledger.get("counterfactual") or (ctx or {}).get("counterfactual") or {}
    rows_all = cf.get("runs") or []
    if not rows_all:
        raise SystemExit("[s3n] 反事实面为空：拒绝记 PASS")
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


def v_roster_matches_prereg(ledger, ctx=None):
    """名册必须与冻结 prereg 的 world/strategy/seed 集逐项相等（禁只报有利种子/世界）。"""
    pre = (ctx or {}).get("prereg") or {}
    if not pre:
        raise SystemExit("[s3n] 校验缺少 prereg：拒绝记 PASS")
    roster = ledger.get("roster") or {}
    want_w = {w["world_id"] for w in pre["worlds"]}
    want_s = set(pre["strategy_order"])
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


VALIDATORS = {
    "v_ledger_nonempty": v_ledger_nonempty,
    "v_budget": v_budget,
    "v_no_repeat": v_no_repeat,
    "v_outcome_accounting": v_outcome_accounting,
    "v_denominator": v_denominator,
    "v_certificates": v_certificates,
    "v_config_equal": v_config_equal,
    "v_public_choice": v_public_choice,
    "v_cost_two_sided": v_cost_two_sided,
    "v_adaptivity": v_adaptivity,
    "v_counterfactual_consistency": v_counterfactual_consistency,
    "v_roster_matches_prereg": v_roster_matches_prereg,
    "v_claim_scope": v_claim_scope,
    "v_grading_consistency": v_grading_consistency,
}


def validate_ledger(ledger, ctx=None) -> dict:
    """跑全部校验器；返回逐器读数。

    空面一律 SystemExit（不记 PASS）；单个校验器抛异常记成 ``VALIDATOR_RAISED``
    （被注入的坏账不许把校验器直接打死）。
    """
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
