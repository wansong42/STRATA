# -*- coding: utf-8 -*-
"""S4-W · 多机制开发世界族（BAECHER / CLUSTER）× 含噪闭环装配（本单唯一新核心模块）。

判据权威 = `results/s4w_world_family_dev/prereg_s4w.json`（冻结件）与任务单
`results/architect_main_20261012_dispatch/task_order_S4_W_world_family_dev_20261018.md`
（sha12 ``6a0499f0cacf``）§2 / §4。**DEVELOPMENT_ONLY**：本件产出的一切世界、读数与
效应均为开发级合成面，禁入确认 / 真实域 / 普遍策略优越 / 成本优势 / 对外承诺文本。

设计（先声明后生成；一切数量参数由 prereg 的 ``world_params`` 冻结，本件只实现机制）：

1. **域 / 盘数 / 井位模板（S1-X 同构）**：域 [0,1]^3；每世界 4 个候选盘（同 S1-X 盘数），
   从机制实现中按**只看几何**的规则选出：先按包围球域内性（center±r 在域内、含 1e-2 安全
   边界）筛选，再按盘心到锚点段 A0–B0 距离升序取前 4（并列按 x,y,z,source_index 字典序）。
2. **机制**（只换中心过程，边际同条件同源）：
   * ``BAECHER`` = ``dfn.generate_dfn``（体强度 P32 反解 N、域内均匀中心、vMF 产状、幂律半径）；
   * ``CLUSTER`` = ``generators.cluster.generate_neyman_scott``（父点均匀 + 高斯核后代；
     边际池 = 同条件同种子的 BAECHER 实现，只换中心采样）。
3. **端点**：A = 候选盘中最接近锚点 A0 者的盘心；B = 其余候选中最近锚点 B0 者的盘心；
   attach 由 ``r_attach`` 几何判定（``disk_point_distance``）；标签 = 接触图连通性派生
   （与 ``closed_loop_expanded._build_table_s1x`` 同一码路）。
4. **井（行动）**：每候选盘一口短井：方位 = 声明方向 w、长度 = 声明 L（深度格整数倍）、
   entry = 盘心 − t*·w ⇒ 目标盘必在 t* 处被曝光；**离轨曝光完全由核几何决定**，不作任何
   标签/存在性读取。成本 = L（域长单位），与 S1-X 同口径。
5. **真相**：truth_state 从世界自己的生成流中声明顺序抽取（在候选/端点/井之后、标签派生
   **之前**；均匀于 2^n 状态，不看标签）。
6. **闭环**：行动选择 / 证书 / 分支口径 / 规划数学**逐字复用** ``closed_loop_expanded``：
   本件只移植其**静态装配回路**（pmf / 观测律令牌 / 支撑 / 读取掩码 / 方位角），因为 S4-W
   世界逐世界几何不同且没有 XS1/XS1M 重复对；装配后仍返回 ``cle.StaticS1X`` 容器，交给
   ``cle.ExpandedWorld`` / ``cle.run_policy_expanded`` / ``cle.choose_action_expanded`` 原样使用。
7. **校验面**（供控制面与门禁）：家族分母 / 世界重放 / 种子独立 / 零标签生成 / 平凡保留 /
   预算 / 结局入账 / 证书 / 配置 / 噪声档 / 公开选点重放 / 成本两面 / 世界级聚合 /
   功效输入 / 机制对照 / 主张范围 / 判级一致性。

RNG 消费次序（逐世界固定，可复现凭据）：
  ``rng = default_rng(world_stream_seed)`` →
  （重抽环）每轮 ``attempt_seed = rng.integers(0, 2**63-1)`` → 机制实现；
  选定候选后 ``truth_state = rng.integers(0, 2**n_candidates)``。
"""
from __future__ import annotations

import hashlib
import json
import math
import random as _random
import time
from fractions import Fraction

import numpy as np

from . import certificate_solver as cs
from . import closed_loop_expanded as cle
from . import dfn
from . import finite_scope as fs
from . import identification as ident
from .generators import cluster as gcluster
from .generators.base import MarginalSpec
from .observation_kernel import GridDetectionKernel, kernel_identity
from .sequential_evidence import bits_of

__all__ = [
    "WORLD_SCHEMA", "FAMILY_SCHEMA", "MECHANISMS", "SELECTION_REASON_VOCAB",
    "SELECTION_FIELDS_ALLOWED", "PHASE_ORDER", "WorldFamilyError",
    "canonical_json", "sha12_obj", "world_id_of", "world_stream_seed",
    "generate_world", "replay_world", "select_candidates_of",
    "derive_labels_of_world", "geometry_facts_of", "build_world_static",
    "WorldFactory", "make_expanded_world", "FAMILY_VALIDATORS", "validate_family",
    "v_family_denominator", "v_world_replay", "v_world_seed_independence",
    "v_no_label_generation", "v_trivial_preserved", "v_family_budget",
    "v_outcome_accounting", "v_certificates", "v_config_vs_prereg",
    "v_noise_tier_consistency", "v_public_choice_replay", "v_cost_two_sided",
    "v_world_effects_aggregation", "v_power_probe_inputs", "v_mechanism_effects",
    "v_claim_scope", "v_grading_consistency",
]

WORLD_SCHEMA = "s4w.world.v1"
FAMILY_SCHEMA = "s4w.family_registry.v1"
MECHANISMS = ("BAECHER", "CLUSTER")

# 生成阶段序（标签必须在最后；任何生成/筛选输入必须先于标签存在）
PHASE_ORDER = ("REALIZATION", "CANDIDATE_SELECTION", "QUERY_ENDPOINTS",
               "WELLS", "TRUTH_STATE_DRAW", "LABELS_DERIVED")

# 选盘规则只许引用这些几何字段（出现其它字段 ⇒ 读标签/存在性嫌疑）
SELECTION_FIELDS_ALLOWED = ("center", "radius", "source_index",
                            "seg_dist_to_anchor_segment")
# 重抽理由词表（只含几何谓词；任何标签/答案词 ⇒ 红）
SELECTION_REASON_VOCAB = ("ACCEPT", "REJECT_INSUFFICIENT_ELIGIBLE",
                          "REJECT_FEWER_THAN_TWO_CANDIDATES")

TRIVIAL_WORLD_KINDS = ("ALL_LABELS_SAME", "NO_DISCS", "NO_WELLS", "FEWER_THAN_FOUR_CANDIDATES")
R_ATTACH_DEFAULT = 0.02
CONTAINMENT_MARGIN_DEFAULT = 0.01


class WorldFamilyError(RuntimeError):
    """世界族实现缺陷（不是读数）。"""


def canonical_json(obj) -> str:
    return cle.canonical_json(obj)


def sha12_obj(obj) -> str:
    return cle.sha12_obj(obj)


def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def world_id_of(mechanism: str, condition: str, tier: str, k: int) -> str:
    return "S4W-%s-%s-%s-%02d" % (str(mechanism), str(condition), str(tier), int(k))


def world_stream_seed(root_seed: int, mech_index: int, cond_index: int,
                      tier_index: int, k: int) -> int:
    """世界生成流种子（声明公式；与机器/平台无关的整数种子）。"""
    ss = np.random.SeedSequence([int(root_seed), int(mech_index), int(cond_index),
                                 int(tier_index), int(k)])
    return int(ss.generate_state(1)[0])


def _unit(v, name="v") -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not (n > 0 and math.isfinite(n)):
        raise WorldFamilyError("%s 非有限/零向量: %r" % (name, v))
    return a / n


def _seg_dist(p, a, b) -> float:
    """点 p 到线段 [a,b] 的欧氏距离（纯几何）。"""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 <= 0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


# ---------------------------------------------------------------------------
# 机制实现（只换中心过程）
# ---------------------------------------------------------------------------

def _baecher_realization(cond: dict, seed: int, window: dict):
    lo = np.asarray(window["lo"], dtype=np.float64)
    hi = np.asarray(window["hi"], dtype=np.float64)
    extents = hi - lo
    center = (lo + hi) / 2.0
    st = dfn.SetTable.from_centers(
        np.asarray([cond["set_center"]], dtype=np.float64), kappa=float(cond["kappa"]))
    real = dfn.generate_dfn(st, p32=float(cond["p32"]), beta=float(cond["beta"]),
                            domain=tuple(float(x) for x in extents), seed=int(seed),
                            n_min=int(cond["n_min"]), r_min=float(cond["r_min"]),
                            r_max=float(cond["r_max"]))
    real.centers = np.asarray(real.centers, dtype=np.float64) + center[None, :]
    real.domain = (1.0, 1.0, 1.0)
    return real, {"generator": "dfn.generate_dfn",
                  "call": {"p32": float(cond["p32"]), "beta": float(cond["beta"]),
                           "generation_window": {"lo": [float(x) for x in lo],
                                                 "hi": [float(x) for x in hi]},
                           "n_min": int(cond["n_min"]),
                           "r_min": float(cond["r_min"]), "r_max": float(cond["r_max"]),
                           "set_center": [float(x) for x in cond["set_center"]],
                           "kappa": float(cond["kappa"])}}


def _cluster_realization(cond: dict, mech_p: dict, seed: int, window: dict):
    lo = np.asarray(window["lo"], dtype=np.float64)
    hi = np.asarray(window["hi"], dtype=np.float64)
    extents = tuple(float(x) for x in (hi - lo))
    center = (lo + hi) / 2.0
    ss = np.random.SeedSequence(int(seed)).spawn(2)
    pool_seed = int(ss[0].generate_state(1)[0])
    cluster_seed = int(ss[1].generate_state(1)[0])
    pool_real, pool_note = _baecher_realization(cond, pool_seed, window)
    n_pop = int(len(pool_real.centers))
    spec = MarginalSpec(domain=extents, n_pop=n_pop,
                        normals_by_set=(np.asarray(pool_real.normals, dtype=np.float64),),
                        radii_sample=np.asarray(pool_real.radii, dtype=np.float64))
    real = gcluster.generate_neyman_scott(spec, int(mech_p["n_parents"]),
                                          float(mech_p["sigma_c"]), cluster_seed)
    real.centers = np.asarray(real.centers, dtype=np.float64) + center[None, :]
    real.domain = (1.0, 1.0, 1.0)
    return real, {"generator": "generators.cluster.generate_neyman_scott",
                  "call": {"n_parents": int(mech_p["n_parents"]),
                           "sigma_c": float(mech_p["sigma_c"]),
                           "generation_window": {"lo": [float(x) for x in lo],
                                                 "hi": [float(x) for x in hi]},
                           "marginal_pool": "同条件 BAECHER 实现（同一公共生成器）",
                           "marginal_pool_seed": pool_seed,
                           "marginal_pool_generator": pool_note["generator"],
                           "cluster_seed": cluster_seed,
                           "n_pop": n_pop}}


def _generate_realization(mechanism: str, cond: dict, mech_p: dict, seed: int, window: dict):
    if mechanism == "BAECHER":
        return _baecher_realization(cond, seed, window)
    if mechanism == "CLUSTER":
        return _cluster_realization(cond, mech_p, seed, window)
    raise WorldFamilyError("未知机制 %r" % (mechanism,))


# ---------------------------------------------------------------------------
# 选盘（只看几何）
# ---------------------------------------------------------------------------

def select_candidates_of(real, cond: dict, params: dict):
    """几何筛选 + 排序；返回 (rows, ranked_eligible)。

    只看：包围球域内性（center/radius）、到锚点段距离（center/anchors）。
    不读任何存在性/标签字段（这些字段此刻尚未派生）。real.centers 已在世界坐标。
    """
    pts = np.asarray(real.centers, dtype=np.float64)
    rads = np.asarray(real.radii, dtype=np.float64)
    margin = float(params.get("containment_margin", CONTAINMENT_MARGIN_DEFAULT))
    A0 = np.asarray(params["anchors"]["A"], dtype=np.float64)
    B0 = np.asarray(params["anchors"]["B"], dtype=np.float64)
    r_cap = float(cond["r_max"])
    rows = []
    for i in range(int(len(pts))):
        c = pts[i]
        r = float(rads[i])
        in_dom = bool(np.all(c - r >= margin - 1e-12) and
                      np.all(c + r <= 1.0 - margin + 1e-12))
        eligible = bool(in_dom and (r <= r_cap + 1e-12))
        rows.append({"source_index": int(i), "center": [float(x) for x in c],
                     "radius": r, "eligible": eligible,
                     "seg_dist_to_anchor_segment": float(_seg_dist(c, A0, B0))})
    elig = [x for x in rows if x["eligible"]]
    ranked = sorted(elig, key=lambda x: (x["seg_dist_to_anchor_segment"],
                                         x["center"][0], x["center"][1],
                                         x["center"][2], x["source_index"]))
    return rows, ranked


def geometry_facts_of(discs, query) -> dict:
    """候选盘几何的接触图与 A/B attach（公开几何的唯一派生源）。"""
    from .connectivity_qoi import disk_point_distance
    from .intersect_disk import candidate_pairs, disk_pair_intersect

    C = np.asarray([d["center"] for d in discs], dtype=float)
    N = np.asarray([d["normal"] for d in discs], dtype=float)
    R = np.asarray([d["radius"] for d in discs], dtype=float)
    pairs = candidate_pairs(C, R)
    if len(pairs):
        hit = disk_pair_intersect(C, N, R, pairs[:, 0], pairs[:, 1],
                                  scale=float(np.median(R)))
        contact = tuple((int(i), int(j)) for (i, j) in pairs[hit].tolist())
    else:
        contact = ()
    A = np.asarray(query["A"], dtype=float)
    B = np.asarray(query["B"], dtype=float)
    ra = float(query["r_attach"])
    aatt = tuple(int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, A)[0] <= ra))
    batt = tuple(int(i) for i in np.flatnonzero(disk_point_distance(C, N, R, B)[0] <= ra))
    return {"contact_edges": contact, "attach_a_disks": aatt, "attach_b_disks": batt}


def derive_labels_of_world(discs, query, n_bits: int) -> dict:
    """逐状态标签 = DISK_BITS 连通性派生（与 S1-X ``_build_table_s1x`` 同一码路）。"""
    facts = geometry_facts_of(discs, query)
    contact = facts["contact_edges"]
    aatt = set(facts["attach_a_disks"])
    batt = set(facts["attach_b_disks"])
    labels = {}
    for s in range(1 << int(n_bits)):
        present = {i for i in range(int(n_bits)) if (int(s) >> i) & 1}
        es = frozenset((i, j) for (i, j) in contact if i in present and j in present)
        a_side = frozenset(x for x in present if x in aatt)
        b_side = frozenset(x for x in present if x in batt)
        der = ident.StateDerivation(state_id=int(s), nodes=frozenset(present), edges=es,
                                    a_side=a_side, b_side=b_side,
                                    attach_a=bool(a_side), attach_b=bool(b_side),
                                    label=0)
        labels[int(s)] = int(ident.state_connected(der))
    return {"labels": labels, "facts": facts}


# ---------------------------------------------------------------------------
# 世界生成（先声明后生成；阶段序 = PHASE_ORDER）
# ---------------------------------------------------------------------------

def generate_world(*, world_id: str, mechanism: str, condition: str, tier: str,
                   k: int, root_seed: int, mech_index: int, cond_index: int,
                   tier_index: int, params: dict) -> dict:
    """生成一个开发世界（几何 + 井 + 真相 + 标签；不含任何读数）。"""
    if mechanism not in MECHANISMS:
        raise WorldFamilyError("未知机制 %r" % (mechanism,))
    cond = dict(params["conditions"][condition])
    mech_p = dict(params["mechanisms"][mechanism])
    window = dict(params["generation_window"])
    if str(tier) not in tuple(params["tier_order"]):
        raise WorldFamilyError("未知噪声档 %r" % (tier,))
    n_cand = int(params["n_candidates"])
    stream_seed = world_stream_seed(root_seed, mech_index, cond_index, tier_index, k)
    rng = np.random.default_rng(stream_seed)
    max_attempts = int(params.get("max_attempts", 8))

    attempts = []
    cands = None
    real_note = None
    real = None
    for attempt in range(max_attempts):
        attempt_seed = int(rng.integers(0, 2 ** 63 - 1))
        real, real_note = _generate_realization(mechanism, cond, mech_p, attempt_seed,
                                                window)
        rows, ranked = select_candidates_of(real, cond, params)
        n_elig = len(ranked)
        if n_elig >= n_cand:
            decision = "ACCEPT"
            attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                             "n_realization": int(len(rows)),
                             "n_eligible": int(n_elig),
                             "decision": decision,
                             "reason": "ACCEPT"})
            cands = ranked[:n_cand]
            break
        reason = ("REJECT_FEWER_THAN_TWO_CANDIDATES" if n_elig < 2
                  else "REJECT_INSUFFICIENT_ELIGIBLE")
        attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                         "n_realization": int(len(rows)),
                         "n_eligible": int(n_elig),
                         "decision": "REJECT", "reason": reason})
    if cands is None:
        # 声明回退：不删世界；保留现有合格盘（≥2 可跑，否则标不可跑）
        _rows, ranked = select_candidates_of(real, cond, params)
        cands = ranked[:min(n_cand, len(ranked))]

    n_bits = int(len(cands))
    # -- 端点：A = 最接近锚点 A0 的候选盘心；B = 其余中最近 B0 者 -----------------
    A0 = np.asarray(params["anchors"]["A"], dtype=np.float64)
    B0 = np.asarray(params["anchors"]["B"], dtype=np.float64)
    centers = [np.asarray(c["center"], dtype=np.float64) for c in cands]
    if n_bits >= 1:
        i_a = min(range(n_bits), key=lambda idx: (float(np.linalg.norm(centers[idx] - A0)), idx))
    else:
        i_a = None
    rest = [idx for idx in range(n_bits) if idx != i_a]
    if rest:
        i_b = min(rest, key=lambda idx: (float(np.linalg.norm(centers[idx] - B0)), idx))
    else:
        i_b = None
    query = None
    if i_a is not None and i_b is not None:
        query = {"query_id": "Q_S4W_%s" % world_id,
                 "A": [float(x) for x in centers[i_a]],
                 "B": [float(x) for x in centers[i_b]],
                 "target_domain": "WORLD_COORDINATE box [0,1]^3",
                 "r_attach": float(params.get("r_attach", R_ATTACH_DEFAULT)),
                 "intersection_convention": ("connectivity_qoi two_point_connected: "
                                             "adjacent discs true-intersect "
                                             "(intersect_disk.disk_pair_intersect, strict inequality)"),
                 "attachment_convention": ("endpoint captured by disc iff point-to-disc distance "
                                           "<= r_attach (inclusive)"),
                 "endpoint_form": "WORLD_COORDINATE",
                 "anchor_A": [float(x) for x in A0], "anchor_B": [float(x) for x in B0],
                 "a_disc": int(i_a) if i_a is not None else None,
                 "b_disc": int(i_b) if i_b is not None else None,
                 "endpoint_rule": ("A/B = 距锚点 A0/B0 最近的候选盘盘心（只看几何；"
                                   "B 取除 A 盘外的最近者）")}

    # -- 井（每候选盘一口）-----------------------------------------------------
    w = _unit(params["well_direction"], "well_direction")
    L = float(params["well_length"])
    tstar = float(params["well_t_star"])
    cost = float(params["well_cost"])
    wells = []
    for s, c in enumerate(cands):
        entry = centers[s] - tstar * w
        wells.append({"action_id": "W%d" % (s + 1), "well_id": int(s),
                      "action_class": "SHORT_SEGMENT_TARGETED",
                      "entry": [float(x) for x in entry],
                      "direction": [float(x) for x in w],
                      "length": L, "cost": cost, "cost_unit": "domain_length",
                      "target_candidate": int(s),
                      "t_star": tstar,
                      "executable_condition": "entry 在声明域内；length 为深度格宽整数倍",
                      "failure_model": "no device failure（同 S0/S1 声明）",
                      "note": ("瞄准候选盘 %d 盘心（t*=%.2f）；离轨曝光由核几何决定，"
                               "不读存在性/标签" % (s, tstar))})

    # -- 盘描述（几何；无标签）---------------------------------------------------
    discs = []
    for s, c in enumerate(cands):
        nrm = _normal_of(real, c)
        discs.append({"candidate_index": int(s),
                      "center": [float(x) for x in c["center"]],
                      "normal": [float(x) for x in nrm],
                      "radius": float(c["radius"]),
                      "role": ("endpoint_A" if s == i_a else
                               ("endpoint_B" if s == i_b else "interior")),
                      "source_index": int(c["source_index"])})
    if any(len(d["normal"]) != 3 for d in discs):
        raise WorldFamilyError("%s: 盘法向形态非法" % world_id)

    # -- 真相：在候选/端点/井之后、标签派生之前均匀抽取 --------------------------
    truth_state = int(rng.integers(0, 1 << max(n_bits, 1))) if n_bits >= 1 else 0

    # -- 标签派生（最后阶段）-----------------------------------------------------
    if query is not None:
        der = derive_labels_of_world(discs, query, n_bits)
        labels = der["labels"]
        facts = der["facts"]
    else:
        labels, facts = {}, {"contact_edges": (), "attach_a_disks": (), "attach_b_disks": ()}

    model = {"domain": {"lo": [0.0, 0.0, 0.0], "hi": [1.0, 1.0, 1.0]},
             "discs": discs, "action_pool": wells,
             "action_sequence": [wd["action_id"] for wd in wells],
             "query": query,
             "state_labels": {str(s): int(v) for s, v in sorted(labels.items())},
             "state_bits_rule": "bit i = 候选盘 i（LSB=第一候选）；state = Σ bit_i·2^i"}
    model["state_bits"] = {str(s): [int(b) for b in bits_of(s, n_bits)]
                           for s in range(1 << n_bits)} if n_bits >= 1 else {}
    # 几何指纹（不含标签）与整模型指纹（含标签）
    geo_obj = {"domain": model["domain"], "discs": model["discs"],
               "action_pool": model["action_pool"], "query": model["query"]}
    geometry_sha12 = sha12_obj(geo_obj)
    model_sha12 = sha12_obj(model)

    all_same = bool(len(set(labels.values())) <= 1) if labels else True
    trivial_kinds = []
    if all_same:
        trivial_kinds.append("ALL_LABELS_SAME")
    if n_bits == 0:
        trivial_kinds.append("NO_DISCS")
    if len(wells) == 0:
        trivial_kinds.append("NO_WELLS")
    if n_bits < n_cand:
        trivial_kinds.append("FEWER_THAN_FOUR_CANDIDATES")
    flags = {"all_labels_same": all_same,
             "n_candidates": int(n_bits),
             "fewer_than_four_candidates": bool(n_bits < n_cand),
             "no_wells": bool(len(wells) == 0),
             "no_discs": bool(n_bits == 0),
             "runnable": bool(n_bits >= 2 and len(wells) >= 2),
             "trivial": bool(trivial_kinds),
             "trivial_kinds": trivial_kinds}
    if not flags["runnable"]:
        flags["not_runnable_reason"] = ("NO_QUERY_ENDPOINTS" if query is None
                                        else "FEWER_THAN_TWO_CANDIDATES")

    spec = {
        "schema": WORLD_SCHEMA,
        "world_id": str(world_id),
        "mechanism": str(mechanism), "condition": str(condition), "tier": str(tier),
        "world_code": int(k),
        "phase_order": list(PHASE_ORDER),
        "generation": {
            "stream_seed": int(stream_seed),
            "stream_seed_rule": ("SeedSequence([root_seed, mech_index, cond_index, "
                                 "tier_index, k]).generate_state(1)"),
            "root_seed": int(root_seed),
            "mech_index": int(mech_index), "cond_index": int(cond_index),
            "tier_index": int(tier_index), "k": int(k),
            "conditions_params": cond,
            "mechanism_params": mech_p,
            "mechanism_realization": real_note,
            "n_realization": int(len(real.centers)),
            "attempts": attempts,
            "rng_consumption": ("stream -> [attempt_seed_i ...] -> truth_state；"
                                "标签在 truth 之后派生（phase_order）"),
        },
        "selection": {
            "rule": ("包围球域内性（center±r 在 [0,1]^3 内 + containment_margin）∧ "
                     "r ≤ 条件 r_max ⇒ 合格；按 (seg_dist_to_anchor_segment, x, y, z, "
                     "source_index) 升序取前 n_candidates"),
            "priority_fields": ["center", "radius", "source_index",
                                "seg_dist_to_anchor_segment"],
            "n_candidates_declared": n_cand,
            "eligible_rows": [{"source_index": r["source_index"], "eligible": bool(r["eligible"]),
                               "seg_dist_to_anchor_segment": r["seg_dist_to_anchor_segment"]}
                              for r in select_candidates_of(real, cond, params)[0]],
            "chosen_source_index": [int(c["source_index"]) for c in cands],
            "labels_available_at_selection": False,
        },
        "model": model,
        "truth_state": int(truth_state),
        "n_bits": int(n_bits), "n_states": int(1 << n_bits) if n_bits >= 1 else 0,
        "flags": flags,
        "facts": {"contact_edges": [[int(i), int(j)] for (i, j) in facts["contact_edges"]],
                  "attach_a_disks": [int(x) for x in facts["attach_a_disks"]],
                  "attach_b_disks": [int(x) for x in facts["attach_b_disks"]]},
        "geometry_sha12": geometry_sha12,
        "model_sha12": model_sha12,
    }
    return spec


def _normal_of(real, cand_row) -> np.ndarray:
    return np.asarray(real.normals[int(cand_row["source_index"])], dtype=np.float64)


def replay_world(*, world_id: str, mechanism: str, condition: str, tier: str, k: int,
                 root_seed: int, mech_index: int, cond_index: int, tier_index: int,
                 params: dict) -> dict:
    """重放（= 再生成一次；确定性由声明流保证）；校验器用它比对现盘世界。"""
    return generate_world(world_id=world_id, mechanism=mechanism, condition=condition,
                          tier=tier, k=k, root_seed=root_seed, mech_index=mech_index,
                          cond_index=cond_index, tier_index=tier_index, params=params)


# ---------------------------------------------------------------------------
# 静态装配（移植 closed_loop_expanded.build_static_s1x 的回路；规划/证书零改）
# ---------------------------------------------------------------------------

def _build_table_family(spec: dict, n_bits: int) -> fs.FiniteTable:
    facts = geometry_facts_of(spec["model"]["discs"], spec["model"]["query"])
    labels = {int(k): int(v) for k, v in spec["model"]["state_labels"].items()}
    contact = tuple((int(i), int(j)) for i, j in facts["contact_edges"])
    aatt = tuple(int(x) for x in facts["attach_a_disks"])
    batt = tuple(int(x) for x in facts["attach_b_disks"])

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
        raise WorldFamilyError("%s: 派生标签与登记标签不符：%s" % (spec["world_id"], mism))
    struct = cs.TableStructure(kind="DISK_BITS", n_bits=n_bits, certain=(),
                               uncertain=tuple(str(d["candidate_index"])
                                               for d in spec["model"]["discs"]),
                               contact_edges=contact, attach_a_disks=aatt,
                               attach_b_disks=batt)
    table = fs.FiniteTable(
        table_id=str(spec["world_id"]), n_bits=n_bits,
        labels={"S4W": np.asarray([labels[s] for s in range(1 << n_bits)], dtype=np.int64)},
        baseline=(), actions=(),
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
              "noise": str(spec["tier"]),
              "model": "S4W 开发世界族（%s/%s）" % (spec["mechanism"], spec["condition"]),
              "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "S4W_DOMAIN"}})
    table.derivation = deriv
    return table


def build_world_static(spec: dict, s1x_spec: dict, kernel_id_expected: str,
                       tier: str | None = None) -> cle.StaticS1X:
    """装配一个世界的静态面（返回 ``cle.StaticS1X`` 容器，供 cle 闭环原样使用）。"""
    tier = str(tier or spec["tier"])
    model = spec["model"]
    kernel = GridDetectionKernel(model, s1x_spec)
    if kernel.kernel_id != str(kernel_id_expected):
        raise WorldFamilyError("STALE_KERNEL：%s != %s"
                               % (kernel.kernel_id, kernel_id_expected))
    n_bits = int(spec["n_bits"])
    if n_bits != len(model["discs"]):
        raise WorldFamilyError("%s: n_bits 与 discs 数不符" % spec["world_id"])
    table = _build_table_family(spec, n_bits)
    labels = {int(k): int(v) for k, v in model["state_labels"].items()}
    pool = list(model["action_pool"])
    by_id = {a["action_id"]: a for a in pool}
    states = tuple(range(1 << n_bits))

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
            law_tokens[aid][int(s)] = cle._law_token(aid, pmf)
        full = kernel.count_pmf(bits_of((1 << n_bits) - 1, n_bits), a, tier=tier)
        recs = []
        for vec, p in full.items():
            if p == 0:
                continue
            counts = {i: int(c) for i, c in enumerate(vec) if c}
            recs.append(kernel.make_public_record(kernel._parse_action(a), counts))
        recs.sort(key=cle._rk_key)
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

    provenance = {
        "research_line": "S4W_world_family_dev",
        "world_id": str(spec["world_id"]),
        "mechanism": str(spec["mechanism"]), "condition": str(spec["condition"]),
        "kernel_id": kernel.kernel_id,
        "kernel_formula_recompute": kernel_identity(
            {k: v for k, v in s1x_spec.items() if k != "kernel_identity"})["kernel_id"],
        "spec_content_sha12": kernel.spec_content_sha12,
        "code_file": "src/fractureflow/observation_kernel.py",
        "code_sha12": kernel.code_sha12,
        "model_sha12": str(spec.get("model_sha12")),
        "geometry_sha12": str(spec.get("geometry_sha12")),
        "n_wells": len(pool), "n_bits": n_bits, "n_states": 1 << n_bits, "tier": tier,
        "T_by_signature": len(sig_groups),
        "signature_groups": [{"signature_sha12": k, "action_ids": v}
                             for k, v in sorted(sig_groups.items())],
        "sequence": [str(x) for x in model.get("action_sequence", [])],
        "sequence_cost_int": int(sum(
            round(float(by_id[x]["cost"]) * 100) for x in model.get("action_sequence", []))),
        "static_assembly": ("移植 closed_loop_expanded.build_static_s1x 的静态回路"
                            "（逐世界几何、无重复对）；规划数学与证书协议零改"),
    }
    cost_int = {str(a["action_id"]): int(round(float(a["cost"]) * 100)) for a in pool}
    cost_dl = {str(a["action_id"]): float(a["cost"]) for a in pool}
    dup = {"action_ids": [], "note": "S4-W 世界无重复对（每候选盘一口井）",
           "per_disc_tables_equal_all_tiers": None,
           "per_state_law_tokens_equal_at_tier": None, "tier": tier}
    return cle.StaticS1X(
        root="", model=model, spec=s1x_spec, kernel=kernel, table=table, labels=labels,
        pool=pool, by_id=by_id, states=states, cost_int=cost_int,
        cost_domain_length=cost_dl, tier=tier, n_bits=n_bits,
        pmf_exact=pmf_exact, dists_float=dists_float, law_tokens=law_tokens,
        support_records=support_records, support_sizes=support_sizes,
        separable=separable, read_masks=read_masks, azimuths=azimuths,
        ab_azimuth=float(ab_azimuth), exposure_rows=exposure_rows,
        duplicate_proof=dup, shared_cache={}, provenance=provenance)


def make_expanded_world(static: cle.StaticS1X, spec: dict, *, noise_seed: int,
                        budget_int: int, alpha_seq, max_steps: int,
                        truth_state: int | None = None):
    truth = int(spec["truth_state"] if truth_state is None else truth_state)
    return cle.ExpandedWorld(static, str(spec["world_id"]), truth_state=truth,
                             noise_seed=int(noise_seed), budget_int=int(budget_int),
                             alpha_seq=Fraction(alpha_seq), max_steps=int(max_steps))


class WorldFactory:
    """按 (world_id, tier) 惰性装配静态面（LRU=1）；供运行器与校验器共用。"""

    def __init__(self, spec_by_id: dict, s1x_spec: dict, kernel_id_expected: str,
                 *, cache_size: int = 1):
        self.spec_by_id = dict(spec_by_id)
        self.s1x_spec = s1x_spec
        self.kernel_id_expected = str(kernel_id_expected)
        self.cache_size = int(cache_size)
        self._cache = {}
        self._order = []
        self.builds = 0
        self.seconds = 0.0

    def static_of(self, world_id: str, tier: str | None = None) -> cle.StaticS1X:
        spec = self.spec_by_id[str(world_id)]
        tier = str(tier or spec["tier"])
        key = (str(world_id), tier)
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        st = build_world_static(spec, self.s1x_spec, self.kernel_id_expected, tier=tier)
        cost = time.perf_counter() - t0
        self.builds += 1
        self.seconds += cost
        self._cache[key] = st
        self._order.append(key)
        while len(self._order) > self.cache_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return st


# ---------------------------------------------------------------------------
# 校验器（S4-W 家族面；部分语义逐字沿用 closed_loop_expanded 的控制面）
# ---------------------------------------------------------------------------

RUN_OUTCOME_CLASSES = cle.OUTCOME_CLASSES


def _runs(led) -> list:
    return list((led or {}).get("runs", []))


def _registry_worlds(reg) -> list:
    return list((reg or {}).get("worlds", []))


def _iter_acted(r):
    return [s for s in r["steps"] if s["action_id"] is not None]


def _run_key(r):
    return "%s|%s|%d" % (r["world_id"], r["strategy"], int(r["seed"]))


def v_family_denominator(docs, ctx=None):
    """世界分母：registry vs prereg 名册；ledger 运行 vs 名册；六类结局。"""
    reg = docs.get("registry")
    pre = (ctx or {}).get("prereg") or {}
    if not reg:
        raise SystemExit("[s4w] 缺世界登记：拒绝记 PASS")
    worlds = _registry_worlds(reg)
    if not worlds:
        raise SystemExit("[s4w] 空世界面：拒绝记 PASS")
    want_w = set(str(x) for x in pre.get("roster_worlds", []))
    if not want_w:
        raise SystemExit("[s4w] prereg 名册为空：拒绝记 PASS")
    bad = []
    got = [str(w["world_id"]) for w in worlds]
    if len(got) != len(set(got)):
        bad.append({"code": "DUPLICATE_WORLD"})
    for m in sorted(want_w - set(got)):
        bad.append({"code": "MISSING_WORLD", "world": m})
    for e in sorted(set(got) - want_w):
        bad.append({"code": "EXTRA_WORLD", "world": e})
    # 逐格计数（每格 ≥ 声明值）
    per_cell = {}
    for w in worlds:
        per_cell.setdefault((w["mechanism"], w["condition"], w["tier"]), 0)
        per_cell[(w["mechanism"], w["condition"], w["tier"])] += 1
    for cell, n in sorted(per_cell.items()):
        if n < int(pre.get("n_worlds_per_cell", 0)):
            bad.append({"code": "CELL_BELOW_DECLARED", "cell": list(cell), "n": n})
    led = docs.get("ledger")
    if led is not None:
        runs = _runs(led)
        if not runs:
            raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
        strategies = list(pre.get("strategies", {}).keys())
        seeds = [int(x) for x in pre.get("seeds", [])]
        runnable = [str(w["world_id"]) for w in worlds if w["flags"]["runnable"]]
        want_runs = {(wid, s, k) for wid in runnable for s in strategies for k in seeds}
        got_runs = [(str(r["world_id"]), str(r["strategy"]), int(r["seed"]))
                    for r in runs]
        if len(got_runs) != len(set(got_runs)):
            bad.append({"code": "DUPLICATE_RUN"})
        for m in sorted(want_runs - set(got_runs)):
            bad.append({"code": "MISSING_RUN", "run": list(m)})
        for e in sorted(set(got_runs) - want_runs):
            bad.append({"code": "EXTRA_RUN", "run": list(e)})
        for r in runs:
            cls = (r.get("stop") or {}).get("class")
            if cls not in RUN_OUTCOME_CLASSES:
                bad.append({"code": "UNKNOWN_OUTCOME_CLASS", "run": _run_key(r),
                            "class": cls})
    return bad


def v_world_replay(docs, ctx=None):
    """逐世界从声明流/参数重放：几何指纹 / 模型指纹 / 标签 / 真相 / 旗标逐位相等。"""
    reg = docs.get("registry")
    ctx = ctx or {}
    pre = ctx.get("prereg") or {}
    params = pre.get("world_params")
    if not reg:
        raise SystemExit("[s4w] 缺世界登记：拒绝记 PASS")
    if not params:
        raise SystemExit("[s4w] 缺 world_params：拒绝记 PASS")
    mech_order = [str(x) for x in params["mechanism_order"]]
    cond_order = [str(x) for x in params["condition_order"]]
    tier_order = [str(x) for x in params["tier_order"]]
    root = int(params["root_seed"])
    bad = []
    for w in _registry_worlds(reg):
        try:
            fresh = replay_world(world_id=w["world_id"], mechanism=w["mechanism"],
                                 condition=w["condition"], tier=w["tier"],
                                 k=int(w["world_code"]), root_seed=root,
                                 mech_index=mech_order.index(w["mechanism"]),
                                 cond_index=cond_order.index(w["condition"]),
                                 tier_index=tier_order.index(w["tier"]),
                                 params=params)
        except Exception as exc:                          # noqa: BLE001
            bad.append({"code": "WORLD_REPLAY_RAISED", "world": w["world_id"],
                        "error": repr(exc)})
            continue
        for field in ("geometry_sha12", "model_sha12"):
            if str(w.get(field)) != str(fresh.get(field)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field, "declared": w.get(field),
                            "recomputed": fresh.get(field)})
        # 逐字节比对模型/生成/筛选/旗标/事实（sha 字段可能被伪造，语义面必须逐位相等）
        for field in ("model", "generation", "selection", "flags", "facts"):
            if canonical_json(w.get(field)) != canonical_json(fresh.get(field)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field})
        for field in ("n_bits", "n_states", "truth_state", "world_code"):
            if int(w.get(field, -1)) != int(fresh.get(field, -2)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field, "declared": w.get(field),
                            "recomputed": fresh.get(field)})
    faces = (ctx or {}).get("faces")
    if isinstance(faces, dict):
        faces["v_world_replay"] = {"n_worlds_replayed": len(_registry_worlds(reg))}
    return bad


def v_world_seed_independence(docs, ctx=None):
    """世界独立：生成流种子互异；几何指纹除声明重复外不得互撞。"""
    reg = docs.get("registry")
    if not reg:
        raise SystemExit("[s4w] 缺世界登记：拒绝记 PASS")
    worlds = _registry_worlds(reg)
    if not worlds:
        raise SystemExit("[s4w] 空世界面：拒绝记 PASS")
    bad = []
    seen_seed = {}
    seen_geo = {}
    for w in worlds:
        wid = str(w["world_id"])
        seed = int(w["generation"]["stream_seed"])
        if seed in seen_seed:
            bad.append({"code": "WORLD_STREAM_SEED_COLLISION", "world": wid,
                        "other": seen_seed[seed], "seed": seed})
        seen_seed.setdefault(seed, wid)
        geo = str(w.get("geometry_sha12"))
        if geo in seen_geo:
            bad.append({"code": "DUPLICATE_GEOMETRY", "world": wid,
                        "other": seen_geo[geo], "geometry_sha12": geo})
        seen_geo.setdefault(geo, wid)
    led = docs.get("ledger")
    # 噪声种子：同 (world, seed) 跨策略必须相同（配对设计）；跨 (world, seed) 必须互异
    per_pair = {}
    for r in _runs(led):
        pair = (str(r["world_id"]), int(r["seed"]))
        ns = int(r.get("noise_seed", -1))
        if pair in per_pair and per_pair[pair] != ns:
            bad.append({"code": "NOISE_SEED_INCONSISTENT_ACROSS_STRATEGIES",
                        "run": _run_key(r)})
        per_pair.setdefault(pair, ns)
    by_val = {}
    for pair, ns in sorted(per_pair.items()):
        if ns in by_val and by_val[ns] != pair:
            bad.append({"code": "DUPLICATE_NOISE_SEED", "pair": list(pair),
                        "other": list(by_val[ns]), "noise_seed": ns})
        by_val.setdefault(ns, pair)
    return bad


def v_no_label_generation(docs, ctx=None):
    """零标签生成：阶段序 / 选盘字段表 / 重抽理由词表 / 标签只由几何派生。"""
    reg = docs.get("registry")
    if not reg:
        raise SystemExit("[s4w] 缺世界登记：拒绝记 PASS")
    worlds = _registry_worlds(reg)
    if not worlds:
        raise SystemExit("[s4w] 空世界面：拒绝记 PASS")
    bad = []
    for w in worlds:
        wid = str(w["world_id"])
        phase = [str(x) for x in w.get("phase_order", [])]
        if phase != list(PHASE_ORDER):
            bad.append({"code": "LABEL_PHASE_ORDER", "world": wid, "phase_order": phase})
        sel = w.get("selection") or {}
        for f in sel.get("priority_fields", []):
            if str(f) not in SELECTION_FIELDS_ALLOWED:
                bad.append({"code": "LABEL_BASED_SELECTION", "world": wid, "field": str(f)})
        if sel.get("labels_available_at_selection") is not False:
            bad.append({"code": "LABEL_BASED_SELECTION", "world": wid,
                        "field": "labels_available_at_selection"})
        for att in (w.get("generation") or {}).get("attempts", []):
            if str(att.get("reason")) not in SELECTION_REASON_VOCAB:
                bad.append({"code": "SELECTION_REASON_UNKNOWN", "world": wid,
                            "reason": att.get("reason")})
        # 标签必需 = 由几何重新派生
        try:
            der = derive_labels_of_world(w["model"]["discs"], w["model"]["query"],
                                         int(w["n_bits"]))
        except Exception as exc:                          # noqa: BLE001
            bad.append({"code": "LABELS_NOT_DERIVED_FROM_GEOMETRY", "world": wid,
                        "error": repr(exc)})
            continue
        rec = {int(k): int(v) for k, v in w["model"]["state_labels"].items()}
        if canonical_json(rec) != canonical_json(der["labels"]):
            bad.append({"code": "LABELS_NOT_DERIVED_FROM_GEOMETRY", "world": wid})
        facts = der["facts"]
        want_facts = {"contact_edges": [[int(i), int(j)] for (i, j) in facts["contact_edges"]],
                      "attach_a_disks": [int(x) for x in facts["attach_a_disks"]],
                      "attach_b_disks": [int(x) for x in facts["attach_b_disks"]]}
        if canonical_json(w.get("facts")) != canonical_json(want_facts):
            bad.append({"code": "LABELS_NOT_DERIVED_FROM_GEOMETRY", "world": wid,
                        "field": "facts"})
        all_same = bool(len(set(rec.values())) <= 1) if rec else True
        if bool((w.get("flags") or {}).get("all_labels_same")) != bool(all_same):
            bad.append({"code": "TRIVIAL_FLAG_MISMATCH", "world": wid})
    return bad


def v_trivial_preserved(docs, ctx=None):
    """平凡世界保留：登记在册且（可跑时）出现在运行账；不可跑必须给理由。"""
    reg = docs.get("registry")
    if not reg:
        raise SystemExit("[s4w] 缺世界登记：拒绝记 PASS")
    worlds = _registry_worlds(reg)
    if not worlds:
        raise SystemExit("[s4w] 空世界面：拒绝记 PASS")
    bad = []
    n_trivial = 0
    for w in worlds:
        wid = str(w["world_id"])
        flags = w.get("flags") or {}
        if flags.get("trivial"):
            n_trivial += 1
            if not flags.get("trivial_kinds"):
                bad.append({"code": "TRIVIAL_NOT_LISTED", "world": wid})
        if not flags.get("runnable") and not flags.get("not_runnable_reason"):
            bad.append({"code": "NOT_RUNNABLE_WITHOUT_REASON", "world": wid})
    if n_trivial == 0:
        bad.append({"code": "TRIVIAL_FILTERED_ALL", "note": "本族无一平凡世界：疑按答案筛"})
    led = docs.get("ledger")
    if led is not None:
        runnable = {str(w["world_id"]) for w in worlds if (w.get("flags") or {}).get("runnable")}
        in_ledger = {str(r["world_id"]) for r in _runs(led)}
        for wid in sorted(runnable - in_ledger):
            bad.append({"code": "WORLD_FILTERED_AFTER_DRAW", "world": wid})
    return bad


def v_family_budget(docs, ctx=None):
    led = docs.get("ledger")
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    bad = []
    for r in runs:
        used = 0
        seen = []
        for s in r["steps"]:
            if s["action_id"] is None:
                continue
            if s["action_id"] in seen:
                bad.append({"code": "REPEATED_ACTION", "run": _run_key(r),
                            "action": s["action_id"]})
            seen.append(s["action_id"])
            if used + int(s["cost"]) > int(r["budget"]):
                bad.append({"code": "OVER_BUDGET", "run": _run_key(r), "t": s["t"],
                            "used": used, "cost": s["cost"], "budget": r["budget"]})
            used += int(s["cost"])
        if used != int(r["used"]):
            bad.append({"code": "USED_MISMATCH", "run": _run_key(r),
                        "used": r["used"], "sum": used})
    return bad


def v_outcome_accounting(docs, ctx=None):
    led = docs.get("ledger")
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    bad = []
    for r in runs:
        acted = _iter_acted(r)
        if len(acted) != int(r["n_steps"]):
            bad.append({"code": "STEP_COUNT_MISMATCH", "run": _run_key(r),
                        "rows": len(acted), "n_steps": r["n_steps"]})
        for s in acted:
            if s.get("record_key") is None or s.get("record_sha12") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r), "t": s["t"]})
            if s.get("used_after") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r), "t": s["t"]})
        if not r["steps"] or r["steps"][-1].get("certificate") is None:
            bad.append({"code": "MISSING_TERMINAL_ROW", "run": _run_key(r)})
    return bad


def v_certificates(docs, ctx=None):
    led = docs.get("ledger")
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    bad = []
    for r in runs:
        terminal_t = None
        for s in r["steps"]:
            c = s.get("certificate")
            if not c:
                bad.append({"code": "MISSING_CERTIFICATE_FIELD", "run": _run_key(r),
                            "t": s["t"]})
                continue
            if c["issued"] and not c["verified"]:
                bad.append({"code": "CERTIFICATE_NOT_VERIFIED", "run": _run_key(r),
                            "t": s["t"], "reasons": c.get("verify_reasons")})
            if c["issued"] and int(c.get("allowed_n", 0)) <= 0:
                bad.append({"code": "CERTIFICATE_ON_EMPTY_SET", "run": _run_key(r),
                            "t": s["t"]})
            if c["issued"]:
                terminal_t = s["t"]
                if str(c.get("label_claim")) not in ("0", "1"):
                    bad.append({"code": "CERTIFICATE_LABEL_INVALID", "run": _run_key(r),
                                "t": s["t"]})
                ids = s.get("allowed_state_ids")
                sid = int(r["truth_state"])
                if int(sid) in set(ids if ids is not None else []) and \
                        int(c["label_claim"]) != int(r["truth_label"]):
                    bad.append({"code": "WRONG_CERT_WHILE_TRUTH_POSSIBLE",
                                "run": _run_key(r), "t": s["t"]})
        if (r.get("stop") or {}).get("class") == "CERTIFIED_SINGLE" and terminal_t is None:
            bad.append({"code": "CERTIFIED_WITHOUT_CERTIFICATE", "run": _run_key(r)})
        if (r.get("stop") or {}).get("class") == "MODEL_CONFLICT" and terminal_t is not None:
            bad.append({"code": "MODEL_CONFLICT_WITH_CERTIFICATE", "run": _run_key(r)})
    return bad


def v_config_vs_prereg(docs, ctx=None):
    led = docs.get("ledger")
    pre = (ctx or {}).get("prereg") or {}
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    cfg = pre.get("experimental_config") or {}
    bad = []
    by_world = {}
    for r in runs:
        by_world.setdefault(str(r["world_id"]), []).append(r)
    for wid, rs in sorted(by_world.items()):
        base = None
        for r in rs:
            key = (int(r["budget"]), str(r["alpha_seq"]), int(r["max_steps"]))
            if base is None:
                base = key
            elif key != base:
                bad.append({"code": "CONFIG_DRIFT_ACROSS_STRATEGIES", "world": wid,
                            "run": _run_key(r), "got": list(key), "want": list(base)})
        if base is not None:
            want = (int(cfg.get("budget_int", -1)), str(cfg.get("alpha_seq")),
                    int(cfg.get("max_steps", -1)))
            if base != want:
                bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": wid,
                            "got": list(base), "want": list(want)})
    return bad


def v_noise_tier_consistency(docs, ctx=None):
    reg = docs.get("registry")
    led = docs.get("ledger")
    if reg is None or led is None:
        raise SystemExit("[s4w] 缺世界登记/运行账：拒绝记 PASS")
    tier_of = {str(w["world_id"]): str(w["tier"]) for w in _registry_worlds(reg)}
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    bad = []
    for r in runs:
        want = tier_of.get(str(r["world_id"]))
        if want is None:
            bad.append({"code": "NOISE_TIER_DRIFT", "run": _run_key(r),
                        "reason": "world_not_in_registry"})
            continue
        if str(r.get("tier")) != want:
            bad.append({"code": "NOISE_TIER_DRIFT", "run": _run_key(r),
                        "got": r.get("tier"), "want": want})
    return bad


def v_public_choice_replay(docs, ctx=None):
    """逐步用公开历史重放策略选择；不可复现即判红（禁真值选点）。

    ctx 需要 ``make_world(run) -> cle.ExpandedWorld``；``replay_sample``（可选）
    = 采样键集，None 表示全量。次生异常记 REPLAY_RAISED / HISTORY_ADVANCE_FAILED。
    """
    led = docs.get("ledger")
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    make_world = (ctx or {}).get("make_world")
    if make_world is None:
        raise SystemExit("[s4w] 校验缺少世界工厂：拒绝记 PASS")
    sample = (ctx or {}).get("replay_sample")
    bad = []
    n_replayed = 0
    n_steps_replayed = 0
    for r in runs:
        if sample is not None and _run_key(r) not in sample:
            continue
        n_replayed += 1
        world = make_world(r)
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
            n_steps_replayed += 1
            try:
                pub = world.public(allowed, used, executed, s["t"] - 1,
                                   world.history_sha12(steps))
                ch = cle.choose_action_expanded(r["strategy"], world, pub, rng, read_so_far)
            except Exception as exc:                      # noqa: BLE001
                bad.append({"code": "REPLAY_RAISED", "run": _run_key(r), "t": s["t"],
                            "error": repr(exc)})
                break
            if str(ch["action_id"]) != str(s["action_id"]):
                bad.append({"code": "ACTION_NOT_REPRODUCIBLE_FROM_PUBLIC_HISTORY",
                            "run": _run_key(r), "t": s["t"],
                            "recorded": s["action_id"], "recomputed": ch["action_id"]})
            if s.get("record_key") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r), "t": s["t"]})
                break
            rec = cle._record_from_json(world, s["record_key"])
            steps.append((str(s["action_id"]), rec))
            try:
                allowed = tuple(world.eprocess(steps).state_set())
            except Exception as exc:                      # noqa: BLE001
                bad.append({"code": "HISTORY_ADVANCE_FAILED", "run": _run_key(r),
                            "t": s["t"], "error": repr(exc)})
                break
            used += int(s["cost"])
            executed = executed + (str(s["action_id"]),)
            read_so_far |= int(world.static.read_masks[str(s["action_id"])])
    if n_replayed == 0:
        raise SystemExit("[s4w] 重放采样为空：拒绝记 PASS")
    faces = (ctx or {}).get("faces")
    if isinstance(faces, dict):
        faces["v_public_choice_replay"] = {"n_runs_replayed": n_replayed,
                                           "n_steps_replayed": n_steps_replayed,
                                           "sample_declared": sample is not None}
    return bad


def v_cost_two_sided(docs, ctx=None):
    led = docs.get("ledger")
    if led is None:
        raise SystemExit("[s4w] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s4w] 空运行账：拒绝记 PASS")
    bad = []
    for r in runs:
        c = r.get("cost") or {}
        if "observed" not in c:
            bad.append({"code": "COST_MISSING_OBSERVED", "run": _run_key(r)})
        if "compute_seconds" not in c:
            bad.append({"code": "COST_MISSING_COMPUTE", "run": _run_key(r)})
        if c.get("basis") != "MEASURED_ONLY":
            bad.append({"code": "COST_BASIS_NOT_MEASURED", "run": _run_key(r),
                        "basis": c.get("basis")})
    return bad


def v_world_effects_aggregation(docs, ctx=None):
    """世界级聚合：逐世界行齐全；配对差以世界为外层单位；σ 输入在场且与账一致。"""
    eff = docs.get("effects")
    led = docs.get("ledger")
    reg = docs.get("registry")
    if eff is None or led is None or reg is None:
        raise SystemExit("[s4w] 缺世界效果/运行账/登记：拒绝记 PASS")
    worlds = _registry_worlds(reg)
    runs = _runs(led)
    if not worlds or not runs:
        raise SystemExit("[s4w] 空面：拒绝记 PASS")
    bad = []
    per_world = eff.get("per_world") or {}
    runnable = [str(w["world_id"]) for w in worlds if (w.get("flags") or {}).get("runnable")]
    strategies = list((eff.get("roster") or {}).get("strategies", []))
    for wid in runnable:
        for s in strategies:
            row = (per_world.get(wid) or {}).get(s)
            if row is None:
                bad.append({"code": "WORLD_AGG_MISSING", "world": wid, "strategy": s})
                continue
            rs = [r for r in runs if str(r["world_id"]) == wid and r["strategy"] == s]
            if not rs:
                continue
            recomputed = sum(1 for r in rs if (r.get("stop") or {}).get("class") == "CERTIFIED_SINGLE"
                             and r.get("correct") is True)
            if int(row.get("certified_correct", -1)) != recomputed:
                bad.append({"code": "WORLD_EFFECT_MISMATCH", "world": wid, "strategy": s,
                            "declared": row.get("certified_correct"),
                            "recomputed": recomputed})
    for stratum, rows in (eff.get("paired") or {}).items():
        for metric, block in rows.items():
            diffs = block.get("world_diffs") or []
            if not diffs:
                bad.append({"code": "PAIRING_INCOMPLETE", "stratum": stratum,
                            "metric": metric, "reason": "empty"})
                continue
            if block.get("unit") != "WORLD":
                bad.append({"code": "PAIRING_UNIT_DRIFT", "stratum": stratum,
                            "metric": metric, "unit": block.get("unit")})
            if block.get("sigma_is_estimate") is not True:
                bad.append({"code": "SIGMA_INPUT_MISSING", "stratum": stratum,
                            "metric": metric})
            # 方向表与计数（复核①：lower-is-better 指标上负差 = NEW 更优）
            direction = block.get("direction")
            if direction not in ("HIGHER_IS_BETTER", "LOWER_IS_BETTER"):
                bad.append({"code": "PAIRING_DIRECTION_MISSING", "stratum": stratum,
                            "metric": metric, "direction": direction})
            else:
                want = eff.get("comparator_direction", {}).get("metric_direction", {}) or {}
                m = str(block.get("metric"))
                if m in want and want[m] != direction:
                    bad.append({"code": "PAIRING_DIRECTION_MISMATCH", "stratum": stratum,
                                "metric": metric, "declared": direction, "want": want[m]})
                if direction == "HIGHER_IS_BETTER":
                    nb = sum(1 for d in diffs if d > 0)
                    nw = sum(1 for d in diffs if d < 0)
                else:
                    nb = sum(1 for d in diffs if d < 0)
                    nw = sum(1 for d in diffs if d > 0)
                nti = sum(1 for d in diffs if d == 0)
                if (int(block.get("n_new_better", -1)) != nb
                        or int(block.get("n_new_worse", -1)) != nw
                        or int(block.get("n_new_tie", -1)) != nti):
                    bad.append({"code": "PAIRING_COUNT_MISMATCH", "stratum": stratum,
                                "metric": metric,
                                "declared": [block.get("n_new_better"),
                                             block.get("n_new_worse"),
                                             block.get("n_new_tie")],
                                "recomputed": [nb, nw, nti]})
    if not (eff.get("sigma_h") or {}):
        bad.append({"code": "SIGMA_INPUT_MISSING", "scope": "sigma_h"})
    return bad


def v_power_probe_inputs(docs, ctx=None):
    """功效/成本探针：Δ_min=0.02 固定；σ 为估计值（禁当已知真值）；禁承诺字段。"""
    powd = docs.get("power")
    if powd is None:
        raise SystemExit("[s4w] 缺功效探针：拒绝记 PASS")
    bad = []
    if powd.get("delta_min") != 0.02:
        bad.append({"code": "DELTA_MIN_DRIFT", "delta_min": powd.get("delta_min")})
    if powd.get("sigma_source_kind") != "ESTIMATED_FROM_DEV_WORLDS":
        bad.append({"code": "SIGMA_CLAIMED_KNOWN",
                    "sigma_source_kind": powd.get("sigma_source_kind")})
    if powd.get("sigma_is_estimate") is not True:
        bad.append({"code": "SIGMA_CLAIMED_KNOWN"})
    for k in ("n_required_final", "guaranteed_n", "commitment", "promised_sample_size"):
        if k in powd:
            bad.append({"code": "POWER_PROMISE_FIELD", "field": k})
    for stratum, block in (powd.get("strata") or {}).items():
        for metric, row in block.items():
            for f in ("sigma_d", "sigma_h", "n_worlds", "z_alpha", "z_beta", "n_formula"):
                if f not in row:
                    bad.append({"code": "POWER_INPUT_UNSOURCED", "stratum": stratum,
                                "metric": metric, "field": f})
    if not (powd.get("strata") or {}):
        bad.append({"code": "POWER_INPUT_UNSOURCED", "scope": "strata_empty"})
    if str(powd.get("claim_scope")) != "DEVELOPMENT_ONLY":
        bad.append({"code": "CLAIM_SCOPE_VIOLATION", "where": "power_cost_probe",
                    "claim_scope": powd.get("claim_scope")})
    return bad


def v_mechanism_effects(docs, ctx=None):
    """机制对照：逐（机制×条件×档）齐全；方向翻转由数据重算，隐藏即红。"""
    mech = docs.get("mechanism_effects")
    if mech is None:
        raise SystemExit("[s4w] 缺机制对照：拒绝记 PASS")
    cells = mech.get("cells") or {}
    if not cells:
        raise SystemExit("[s4w] 机制对照空面：拒绝记 PASS")
    bad = []
    pre = (ctx or {}).get("prereg") or {}
    params = pre.get("world_params") or {}
    want_cells = {"%s|%s|%s" % (m, c, t)
                  for m in params.get("mechanism_order", [])
                  for c in params.get("condition_order", [])
                  for t in params.get("tier_order", [])}
    for cell in sorted(want_cells - set(cells)):
        bad.append({"code": "MECH_CELL_MISSING", "cell": cell})
    # 方向翻转重算（对声明的一组比较）
    flips = mech.get("direction_flips") or []
    declared = {}
    for f in flips:
        flip_cells = f.get("cells")
        if flip_cells is None:
            flip_cells = [f["cell"]] if "cell" in f else []
        for c in flip_cells:
            declared.setdefault(str(f["comparison"]), []).append(str(c))
    for comp in (mech.get("flip_scan_comparisons") or []):
        want = []
        signs = {}
        for cell, block in cells.items():
            row = (block.get("paired_means") or {}).get(comp)
            if row is None:
                continue
            signs[cell] = 1 if float(row) > 0 else (-1 if float(row) < 0 else 0)
        pos = sorted(c for c, s in signs.items() if s > 0)
        neg = sorted(c for c, s in signs.items() if s < 0)
        if pos and neg:
            want = sorted(pos + neg)
        got = sorted(declared.get(comp, []))
        if sorted(set(got)) != want:
            bad.append({"code": "DIRECTION_FLIP_HIDDEN", "comparison": comp,
                        "declared": got, "recomputed_flip_cells": want})
    return bad


def v_claim_scope(docs=None, ctx=None):
    ctx = ctx or {}
    phrases = [str(x) for x in (ctx.get("phrases") or [])]
    texts = [str(x) for x in (ctx.get("texts") or [])]
    fields = ctx.get("claim_scopes") or []
    if not phrases:
        raise SystemExit("[s4w] 空禁用词表：拒绝记 PASS")
    if not texts and not fields:
        raise SystemExit("[s4w] 空主张面：拒绝记 PASS")
    bad = []
    for t in texts:
        for p in phrases:
            if p and p in t:
                bad.append({"code": "FORBIDDEN_CLAIM_PHRASE", "phrase": p})
    for f in fields:
        if str(f) != "DEVELOPMENT_ONLY":
            bad.append({"code": "CLAIM_SCOPE_VIOLATION", "claim_scope": f})
    return bad


def v_grading_consistency(docs=None, ctx=None):
    ctx = ctx or {}
    v = ctx.get("verdict") or {}
    if not v:
        raise SystemExit("[s4w] 判级面为空：拒绝记 PASS")
    reg = docs.get("registry") if docs else None
    ctl = docs.get("controls") if docs else None
    led = docs.get("ledger") if docs else None
    GRADING_VALUES = ("ACCEPTED", "REJECTED", "PARTIAL", "NOT_DELIVERED", "UNRESOLVED")
    bad = []
    g = str(v.get("grading"))
    if g not in GRADING_VALUES:
        bad.append({"code": "GRADING_VALUE_UNKNOWN", "grading": v.get("grading")})
        return bad
    if g == "ACCEPTED":
        if ctl is not None and not ctl.get("all_caught"):
            bad.append({"code": "GRADING_UNSUPPORTED", "reason": "controls_not_all_caught"})
        if ctl is not None and not (ctl.get("clean_control") or {}).get("clean_ok", False):
            bad.append({"code": "GRADING_UNSUPPORTED", "reason": "clean_control_not_silent"})
        if led is not None:
            unver = sum(int(r.get("certificates_unverified", 0)) for r in _runs(led))
            if unver:
                bad.append({"code": "GRADING_UNSUPPORTED", "reason": "unverified_certificates"})
    return bad


FAMILY_VALIDATORS = {
    "v_family_denominator": v_family_denominator,
    "v_world_replay": v_world_replay,
    "v_world_seed_independence": v_world_seed_independence,
    "v_no_label_generation": v_no_label_generation,
    "v_trivial_preserved": v_trivial_preserved,
    "v_family_budget": v_family_budget,
    "v_outcome_accounting": v_outcome_accounting,
    "v_certificates": v_certificates,
    "v_config_vs_prereg": v_config_vs_prereg,
    "v_noise_tier_consistency": v_noise_tier_consistency,
    "v_public_choice_replay": v_public_choice_replay,
    "v_cost_two_sided": v_cost_two_sided,
    "v_world_effects_aggregation": v_world_effects_aggregation,
    "v_power_probe_inputs": v_power_probe_inputs,
    "v_mechanism_effects": v_mechanism_effects,
    "v_claim_scope": v_claim_scope,
    "v_grading_consistency": v_grading_consistency,
}


def validate_family(docs, ctx=None, only=None) -> dict:
    """跑家族校验器；空面 SystemExit；单器异常记 VALIDATOR_RAISED（不打死）。"""
    names = list(FAMILY_VALIDATORS) if only is None else list(only)
    out = {}
    for name in names:
        fn = FAMILY_VALIDATORS[name]
        try:
            out[name] = fn(docs, ctx)
        except SystemExit:
            raise
        except Exception as exc:                          # noqa: BLE001
            out[name] = [{"code": "VALIDATOR_RAISED", "validator": name,
                          "error": repr(exc)}]
    return out
