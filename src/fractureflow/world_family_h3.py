# -*- coding: utf-8 -*-
"""S5-D1 · 绑定预算挑战族（H3 头部空间闸开发族；本单唯一新族核心模块）。

判据权威 = `results/s5d1_h3_family_redesign/prereg_s5d1.json`（冻结件）与任务单
`results/architect_main_20261012_dispatch/task_order_S5D1_binding_budget_family_20261020.md`
§2/§4/§5。**DEVELOPMENT_ONLY**：本件产出的一切世界、读数与效应均为开发级合成面，
禁入确认 / 真实域 / 普遍策略优越 / 成本优势 / 对外承诺文本；不写 S5 结论。

设计（先声明后生成；机制/条件/端点/井规则与 S4-W 同定义，只改规模与预算制度）：

1. **域 / 盘数 / 候选序（S4-W 同构）**：域 [0,1]^3；每世界生成机制实现后按
   **只看几何** 的规则选候选：包围球域内性（center±r 在域内 + containment_margin）
   ∧ r ≤ 条件 r_max ⇒ 合格；按 (到锚点段 A0–B0 距离, x, y, z, source_index) 升序，
   前 12 个进入**完整候选序**。d=12 用全部 12 个；预注册 fallback d=8 用完整候选序
   **前 8 个**（子字典规则声明在先、可复算）。
2. **机制**（只换中心过程，边际同条件同源）：
   * ``BAECHER`` = ``dfn.generate_dfn``（P32 反解 N、域内均匀中心、vMF 产状、幂律半径）；
   * ``CLUSTER`` = ``generators.cluster.generate_neyman_scott``（父点均匀 + 高斯核后代；
     边际池 = 同条件同种子的 BAECHER 实现，只换中心采样）。
3. **端点**：A = 候选盘中最接近锚点 A0 者的盘心；B = 其余候选中最近锚点 B0 者的盘心；
   attach 由 ``r_attach`` 几何判定；标签 = 接触图连通性派生（与 S1-X / S4-W 同一码路）。
4. **井（行动）**：每候选盘一口短井（方位 = 声明 w、长度 = 声明 L、cost = 声明值），
   entry = 盘心 − t*·w ⇒ 目标盘必在 t* 处被曝光；**离轨曝光完全由核几何决定**，不作任何
   标签/存在性读取。每口井的公开几何曝光位掩码在生成阶段由观测核几何逐盘判定并记录。
5. **真相**：truth_state 从世界自己的生成流中声明顺序抽取（在候选/端点/井/掩码之后、
   标签派生**之前**；均匀于 2^n 状态，不看标签）。
6. **闭环**：行动选择 / 证书 / 分支口径 / 规划数学**逐字复用** ``closed_loop_expanded``：
   本件只做**静态装配**（lazy mask-cached 观测律 + 稀疏精确 PMF），装配后仍返回
   ``cle.StaticS1X`` 容器，交给 ``cle.ExpandedWorld`` / ``cle.run_policy_expanded`` /
   ``cle.choose_action_expanded`` 原样使用。
7. **绑定预算**：n_bits 口井、每井 cost_int=4、budget_int=12、max_steps=3
   ⇒ 预算严格小于全池成本 4·n_bits（d=12 时 48；d=8 时 32），每 run 至多执行 3 口井。
8. **校验面**（供控制面与门禁）：分母 / 世界重放 / 种子独立 / 零标签生成 / 平凡保留 /
   绑定预算 / 井池与曝光掩码 / 结局入账 / 证书 / 配置 / 噪声档 / 公开选点重放 /
   成本两面 / 世界级聚合 / oracle 口径 / 功效输入 / 头闸判定 / 判级一致性。

RNG 消费次序（逐世界固定，可复现凭据）：
  ``rng = default_rng(world_stream_seed)`` →
  （重抽环）每轮 ``attempt_seed = rng.integers(0, 2**63-1)`` → 机制实现；
  选定候选后 ``truth_state = rng.integers(0, 2**dimension)``。
"""
from __future__ import annotations

import hashlib
import json
import math
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
    "WORLD_SCHEMA", "FAMILY_SCHEMA", "MECHANISMS", "PHASE_ORDER", "SELECTION_REASON_VOCAB",
    "SELECTION_FIELDS_ALLOWED", "RUN_OUTCOME_CLASSES", "WorldFamilyError",
    "canonical_json", "sha12_obj", "sha12_bytes", "sha12_file",
    "world_id_of", "world_stream_seed", "generate_world", "replay_world",
    "select_candidates_of", "derive_labels_of_world", "geometry_facts_of",
    "build_world_static", "make_expanded_world", "WorldFactory",
    "compute_headroom", "FAMILY_VALIDATORS", "validate_family",
]

WORLD_SCHEMA = "s5d1.world.v1"
FAMILY_SCHEMA = "s5d1.family_registry.v1"
MECHANISMS = ("BAECHER", "CLUSTER")

# 生成阶段序（标签必须在最后；掩码只读公开几何，先于真相抽取）
PHASE_ORDER = ("REALIZATION", "CANDIDATE_SELECTION", "QUERY_ENDPOINTS", "WELLS",
               "EXPOSURE_MASKS", "TRUTH_STATE_DRAW", "LABELS_DERIVED")

SELECTION_FIELDS_ALLOWED = ("center", "radius", "source_index",
                            "seg_dist_to_anchor_segment")
SELECTION_REASON_VOCAB = ("ACCEPT", "REJECT_INSUFFICIENT_ELIGIBLE")

TRIVIAL_WORLD_KINDS = ("ALL_LABELS_SAME", "NO_DISCS", "NO_WELLS",
                       "FEWER_THAN_DECLARED_CANDIDATES")
R_ATTACH_DEFAULT = 0.02
CONTAINMENT_MARGIN_DEFAULT = 0.01
RUN_OUTCOME_CLASSES = cle.OUTCOME_CLASSES


class WorldFamilyError(RuntimeError):
    """世界族实现缺陷（不是读数）。"""


def canonical_json(obj) -> str:
    return cle.canonical_json(obj)


def sha12_obj(obj) -> str:
    return cle.sha12_obj(obj)


def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def sha12_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha12_bytes(f.read())


def world_id_of(mechanism: str, condition: str, tier: str, k: int) -> str:
    return "S5D1-%s-%s-%s-%02d" % (str(mechanism), str(condition), str(tier), int(k))


def world_stream_seed(root_seed: int, mech_index: int, cond_index: int,
                      tier_index: int, k: int) -> int:
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
# 机制实现（与 S4-W 同定义，只换中心过程）
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


def _generate_realization(mechanism: str, cond: dict, mech_p: dict, seed: int,
                          window: dict):
    if mechanism == "BAECHER":
        return _baecher_realization(cond, seed, window)
    if mechanism == "CLUSTER":
        return _cluster_realization(cond, mech_p, seed, window)
    raise WorldFamilyError("未知机制 %r" % (mechanism,))


# ---------------------------------------------------------------------------
# 选盘（只看几何）
# ---------------------------------------------------------------------------

def select_candidates_of(real, cond: dict, params: dict):
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
# 井曝光掩码（公开几何；生成阶段记录）
# ---------------------------------------------------------------------------

def wells_of_candidates(cands, params: dict) -> list:
    w = _unit(params["well_direction"], "well_direction")
    L = float(params["well_length"])
    tstar = float(params["well_t_star"])
    cost = float(params["well_cost"])
    wells = []
    for s, c in enumerate(cands):
        center = np.asarray(c["center"], dtype=np.float64)
        entry = center - tstar * w
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
    return wells


def attach_exposure_masks(model: dict, obs_spec: dict) -> list:
    """用冻结观测核几何逐井记录公开曝光位掩码（不读标签/真相）。"""
    kernel = GridDetectionKernel(model, obs_spec)
    n_bits = len(model["discs"])
    rows_out = []
    for a in model["action_pool"]:
        mask = 0
        rows = []
        for i in range(n_bits):
            e = kernel.disc_exposure(a, i)
            rows.append({"disc_index": int(i), "exposed": bool(e["exposed"]),
                         "reason": str(e["reason"]), "t": e["t"]})
            if e["exposed"]:
                mask |= 1 << i
        a["exposure_mask"] = int(mask)
        a["exposure_bits"] = [int(i) for i in range(n_bits) if (mask >> i) & 1]
        a["exposure_rows"] = rows
        rows_out.append({"action_id": str(a["action_id"]), "mask": int(mask),
                         "mask_size": int(bin(mask).count("1")),
                         "target_candidate": int(a["target_candidate"]),
                         "target_exposed": bool((mask >> int(a["target_candidate"])) & 1)})
    return rows_out


# ---------------------------------------------------------------------------
# 世界生成（先声明后生成；阶段序 = PHASE_ORDER）
# ---------------------------------------------------------------------------

def generate_world(*, world_id: str, mechanism: str, condition: str, tier: str,
                   k: int, root_seed: int, mech_index: int, cond_index: int,
                   tier_index: int, dimension: int, params: dict,
                   obs_spec: dict) -> dict:
    """生成一个 H3 族世界（几何 + 井 + 掩码 + 真相 + 标签；不含任何读数）。"""
    if mechanism not in MECHANISMS:
        raise WorldFamilyError("未知机制 %r" % (mechanism,))
    cond = dict(params["conditions"][condition])
    mech_p = dict(params["mechanisms"][mechanism])
    window = dict(params["generation_window"])
    if str(tier) not in tuple(params["tier_order"]):
        raise WorldFamilyError("未知噪声档 %r" % (tier,))
    n_declared = int(params["n_candidates_declared"])
    if int(dimension) not in (12, 8):
        raise WorldFamilyError("dimension 只许 12（主）或 8（fallback）：%r" % (dimension,))
    if int(dimension) > n_declared:
        raise WorldFamilyError("dimension 超过完整候选数：%r > %r" % (dimension, n_declared))
    stream_seed = world_stream_seed(root_seed, mech_index, cond_index, tier_index, k)
    rng = np.random.default_rng(stream_seed)
    max_attempts = int(params.get("max_attempts", 8))

    attempts = []
    ranked_full = None
    real_note = None
    real = None
    for attempt in range(max_attempts):
        attempt_seed = int(rng.integers(0, 2 ** 63 - 1))
        real, real_note = _generate_realization(mechanism, cond, mech_p, attempt_seed,
                                                window)
        rows, ranked = select_candidates_of(real, cond, params)
        n_elig = len(ranked)
        if n_elig >= n_declared:
            attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                             "n_realization": int(len(rows)),
                             "n_eligible": int(n_elig),
                             "decision": "ACCEPT", "reason": "ACCEPT"})
            ranked_full = ranked[:n_declared]
            break
        attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                         "n_realization": int(len(rows)),
                         "n_eligible": int(n_elig),
                         "decision": "REJECT",
                         "reason": "REJECT_INSUFFICIENT_ELIGIBLE"})
    if ranked_full is None:
        raise WorldFamilyError(
            "%s: %d 次重抽后仍不足 %d 个合格候选（最后一次 n_eligible=%d）"
            % (world_id, max_attempts, n_declared,
               int(attempts[-1]["n_eligible"]) if attempts else -1))

    cands = ranked_full[:int(dimension)]
    n_bits = int(len(cands))
    # -- 端点 ---------------------------------------------------------------
    A0 = np.asarray(params["anchors"]["A"], dtype=np.float64)
    B0 = np.asarray(params["anchors"]["B"], dtype=np.float64)
    centers = [np.asarray(c["center"], dtype=np.float64) for c in cands]
    i_a = min(range(n_bits), key=lambda idx: (float(np.linalg.norm(centers[idx] - A0)), idx))
    rest = [idx for idx in range(n_bits) if idx != i_a]
    i_b = (min(rest, key=lambda idx: (float(np.linalg.norm(centers[idx] - B0)), idx))
           if rest else None)
    query = None
    if i_b is not None:
        query = {"query_id": "Q_S5D1_%s" % world_id,
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
                 "a_disc": int(i_a), "b_disc": int(i_b),
                 "endpoint_rule": ("A/B = 距锚点 A0/B0 最近的候选盘盘心（只看几何；"
                                   "B 取除 A 盘外的最近者）")}

    # -- 井 -----------------------------------------------------------------
    wells = wells_of_candidates(cands, params)
    # -- 盘描述（几何；无标签）---------------------------------------------
    discs = []
    for s, c in enumerate(cands):
        nrm = np.asarray(real.normals[int(c["source_index"])], dtype=np.float64)
        discs.append({"candidate_index": int(s),
                      "center": [float(x) for x in c["center"]],
                      "normal": [float(x) for x in nrm],
                      "radius": float(c["radius"]),
                      "role": ("endpoint_A" if s == i_a else
                               ("endpoint_B" if s == i_b else "interior")),
                      "source_index": int(c["source_index"])})
    model = {"domain": {"lo": [0.0, 0.0, 0.0], "hi": [1.0, 1.0, 1.0]},
             "discs": discs, "action_pool": wells,
             "action_sequence": [wd["action_id"] for wd in wells],
             "query": query,
             "state_labels": {}, "state_bits_rule":
                 "bit i = 候选盘 i（LSB=第一候选）；state = Σ bit_i·2^i"}
    # -- 曝光掩码（公开几何；先于真值）--------------------------------------
    mask_rows = attach_exposure_masks(model, obs_spec) if len(discs) else []

    # -- 真相：在候选/端点/井/掩码之后、标签派生之前均匀抽取 ----------------
    truth_state = int(rng.integers(0, 1 << max(n_bits, 1))) if n_bits >= 1 else 0
    # -- 标签派生（最后阶段）-----------------------------------------------
    if query is not None:
        der = derive_labels_of_world(discs, query, n_bits)
        labels = der["labels"]
        facts = der["facts"]
    else:
        labels, facts = {}, {"contact_edges": (), "attach_a_disks": (),
                             "attach_b_disks": ()}
    model["state_labels"] = {str(s): int(v) for s, v in sorted(labels.items())}
    model["state_bits"] = ({str(s): [int(b) for b in bits_of(s, n_bits)]
                            for s in range(1 << n_bits)} if n_bits >= 1 else {})
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
    if n_bits < int(dimension):
        trivial_kinds.append("FEWER_THAN_DECLARED_CANDIDATES")
    mask_sizes = [int(bin(int(a["exposure_mask"])).count("1")) for a in wells]
    distinct_masks = sorted({int(a["exposure_mask"]) for a in wells})
    flags = {"all_labels_same": all_same,
             "n_candidates": int(n_bits),
             "fewer_than_declared_candidates": bool(n_bits < int(dimension)),
             "no_wells": bool(len(wells) == 0),
             "no_discs": bool(n_bits == 0),
             "runnable": bool(n_bits >= 2 and len(wells) >= 2 and query is not None),
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
        "dimension_declared": 12,
        "dimension_used": int(dimension),
        "fallback_applied": bool(int(dimension) != 12),
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
                                "掩码与标签在 truth 之前/之后按 PHASE_ORDER"),
        },
        "selection": {
            "rule": ("包围球域内性（center±r 在 [0,1]^3 内 + containment_margin）∧ "
                     "r ≤ 条件 r_max ⇒ 合格；按 (seg_dist_to_anchor_segment, x, y, z, "
                     "source_index) 升序取完整候选序前 %d；d=8 fallback 取前 8" % n_declared),
            "priority_fields": list(SELECTION_FIELDS_ALLOWED),
            "n_candidates_declared": int(n_declared),
            "n_candidates_used": int(n_bits),
            "chosen_source_index_full": [int(c["source_index"]) for c in ranked_full],
            "chosen_source_index": [int(c["source_index"]) for c in cands],
            "eligible_rows": [{"source_index": int(r["source_index"]),
                               "eligible": bool(r["eligible"]),
                               "seg_dist_to_anchor_segment": r["seg_dist_to_anchor_segment"]}
                              for r in select_candidates_of(real, cond, params)[0]],
            "labels_available_at_selection": False,
        },
        "exposure": {
            "rule": ("每井 entry = 盘心 − t*·w；length/cost 为声明值；曝光位由冻结观测核"
                     "几何逐盘判定（t∈[0,L_eff] ∧ 点到盘心 ≤ r）"),
            "per_well": mask_rows,
            "mask_sizes": mask_sizes,
            "n_distinct_masks": int(len(distinct_masks)),
            "distinct_masks": [int(x) for x in distinct_masks],
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


def replay_world(*, world_id: str, mechanism: str, condition: str, tier: str, k: int,
                 root_seed: int, mech_index: int, cond_index: int, tier_index: int,
                 dimension: int, params: dict, obs_spec: dict) -> dict:
    return generate_world(world_id=world_id, mechanism=mechanism, condition=condition,
                          tier=tier, k=k, root_seed=root_seed, mech_index=mech_index,
                          cond_index=cond_index, tier_index=tier_index,
                          dimension=dimension, params=params, obs_spec=obs_spec)


# ---------------------------------------------------------------------------
# 静态装配（lazy mask-cached 观测律；d=12/d=8 可算性的关键）
# ---------------------------------------------------------------------------

def sparse_pmf(kernel: GridDetectionKernel, action: dict, mask: int, n_bits: int,
               tier: str = "N1", cap: int = 400000) -> dict:
    """(action, 在位掩码) 的计数向量 PMF 稀疏精确有理数；语义同 count_pmf。"""
    dp = {(): Fraction(1, 1)}
    for i in range(int(n_bits)):
        if not ((int(mask) >> i) & 1):
            continue
        tab = kernel.disc_outcome_table(action, i, tier=tier)
        ndp = {}
        for vec, p0 in dp.items():
            for outcome, p in tab.items():
                if outcome == "EMPTY":
                    key = vec
                else:
                    c = int(outcome)
                    pos = None
                    for j, (cc, _n) in enumerate(vec):
                        if cc == c:
                            pos = j
                            break
                    if pos is None:
                        key = tuple(sorted(vec + ((c, 1),)))
                    else:
                        lst = list(vec)
                        lst[pos] = (cc, lst[pos][1] + 1)
                        key = tuple(sorted(lst))
                ndp[key] = ndp.get(key, Fraction(0, 1)) + p0 * p
        dp = ndp
        if len(dp) > int(cap):
            raise WorldFamilyError(
                "稀疏精确 PMF 超出声明预算 cap=%d (action=%s, mask=%r)"
                % (cap, action["action_id"], mask))
    total = sum(dp.values())
    if total != 1:
        raise WorldFamilyError("稀疏 PMF 归一化失败: sum=%s != 1" % (total,))
    return dp


class _MaskCache:
    def __init__(self, kernel, n_bits: int, tier: str, cap: int = 400000):
        self.kernel = kernel
        self.n_bits = int(n_bits)
        self.tier = str(tier)
        self.cap = int(cap)
        self._pmf = {}
        self._token = {}
        self._dist = {}
        self.n_pmf_calls = 0
        self.seconds = 0.0

    def pmf(self, action: dict, mask: int) -> dict:
        key = (str(action["action_id"]), int(mask))
        if key not in self._pmf:
            t0 = time.perf_counter()
            self._pmf[key] = sparse_pmf(self.kernel, action, int(mask), self.n_bits,
                                        tier=self.tier, cap=self.cap)
            self.seconds += time.perf_counter() - t0
            self.n_pmf_calls += 1
        return self._pmf[key]

    def token(self, action: dict, mask: int) -> str:
        key = (str(action["action_id"]), int(mask))
        if key not in self._token:
            self._token[key] = cle._law_token(str(action["action_id"]),
                                              self.pmf(action, int(mask)))
        return self._token[key]

    def dist(self, action: dict, mask: int) -> dict:
        key = (str(action["action_id"]), int(mask))
        if key not in self._dist:
            pmf = self.pmf(action, int(mask))
            self._dist[key] = {k: float(v) for k, v in pmf.items()}
        return self._dist[key]


class _LazyByState(dict):
    def __init__(self, cache: _MaskCache, action: dict, mask: int, kind: str):
        super().__init__()
        self._cache = cache
        self._action = action
        self._mask = int(mask)
        self._kind = str(kind)

    def __missing__(self, state):
        m = int(state) & self._mask
        if self._kind == "token":
            return self._cache.token(self._action, m)
        return self._cache.dist(self._action, m)

    def __getitem__(self, state):
        return self.__missing__(int(state))


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
        raise WorldFamilyError("%s: 派生标签与登记标签不符：%s"
                               % (spec["world_id"], mism[:8]))
    struct = cs.TableStructure(kind="DISK_BITS", n_bits=n_bits, certain=(),
                               uncertain=tuple(str(d["candidate_index"])
                                               for d in spec["model"]["discs"]),
                               contact_edges=contact, attach_a_disks=aatt,
                               attach_b_disks=batt)
    table = fs.FiniteTable(
        table_id=str(spec["world_id"]), n_bits=n_bits,
        labels={"S5D1": np.asarray([labels[s] for s in range(1 << n_bits)],
                                   dtype=np.int64)},
        baseline=(), actions=(),
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
              "noise": str(spec["tier"]),
              "model": "S5-D1 绑定预算挑战族（%s/%s, d=%d）"
                       % (spec["mechanism"], spec["condition"], n_bits),
              "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "family": "BINDING_BUDGET_CHALLENGE",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "S5D1_DOMAIN"}})
    table.derivation = deriv
    return table


def build_world_static(spec: dict, obs_spec: dict, kernel_id_expected: str,
                       tier: str | None = None, pmf_cap: int = 400000) -> cle.StaticS1X:
    tier = str(tier or spec["tier"])
    model = spec["model"]
    kernel = GridDetectionKernel(model, obs_spec)
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

    cache = _MaskCache(kernel, n_bits, tier, cap=int(pmf_cap))
    dists_float, law_tokens = {}, {}
    read_masks, azimuths = {}, {}
    exposure_rows = []
    for a in pool:
        aid = str(a["action_id"])
        m = 0
        for i in range(n_bits):
            e = kernel.disc_exposure(a, i)
            exposure_rows.append({"action_id": aid, "disc_index": i,
                                  "exposed": bool(e["exposed"]),
                                  "reason": e["reason"], "t": e["t"]})
            if e["exposed"]:
                m |= 1 << i
        recorded = a.get("exposure_mask")
        if recorded is not None and int(recorded) != int(m):
            raise WorldFamilyError("%s/%s: 登记的曝光掩码与几何重算不符（%r vs %r）"
                                   % (spec["world_id"], aid, recorded, m))
        read_masks[aid] = m
        law_tokens[aid] = _LazyByState(cache, a, m, "token")
        dists_float[aid] = _LazyByState(cache, a, m, "dist")
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
        sig = sha12_obj({"n_depth_bins": int(round(float(a["length"])
                                                   / kernel.depth_bin_width)),
                         "per_disc": rows})
        sig_groups.setdefault(sig, []).append(str(a["action_id"]))

    mask_sizes = {aid: int(bin(m).count("1")) for aid, m in read_masks.items()}
    provenance = {
        "research_line": "S5D1_binding_budget_family",
        "family": "BINDING_BUDGET_CHALLENGE",
        "world_id": str(spec["world_id"]),
        "mechanism": str(spec["mechanism"]), "condition": str(spec["condition"]),
        "kernel_id": kernel.kernel_id,
        "kernel_formula_recompute": kernel_identity(
            {k: v for k, v in obs_spec.items() if k != "kernel_identity"})["kernel_id"],
        "spec_content_sha12": kernel.spec_content_sha12,
        "code_file": "src/fractureflow/observation_kernel.py",
        "code_sha12": kernel.code_sha12,
        "model_sha12": str(spec.get("model_sha12")),
        "geometry_sha12": str(spec.get("geometry_sha12")),
        "n_wells": len(pool), "n_bits": n_bits, "n_states": 1 << n_bits, "tier": tier,
        "dimension_declared": int(spec.get("dimension_declared", 12)),
        "dimension_used": int(spec.get("dimension_used", n_bits)),
        "fallback_applied": bool(spec.get("fallback_applied", False)),
        "read_masks": {k: int(v) for k, v in sorted(read_masks.items())},
        "mask_sizes": mask_sizes,
        "n_distinct_masks": int(len(set(read_masks.values()))),
        "T_by_signature": len(sig_groups),
        "signature_groups": [{"signature_sha12": k, "action_ids": v}
                             for k, v in sorted(sig_groups.items())],
        "sequence": [str(x) for x in model.get("action_sequence", [])],
        "static_assembly": ("lazy mask-cached dists/law tokens + sparse exact PMF；"
                            "规划数学与证书协议逐字复用 closed_loop_expanded"),
    }
    cost_int = {str(a["action_id"]): int(round(float(a["cost"]) * 100)) for a in pool}
    cost_dl = {str(a["action_id"]): float(a["cost"]) for a in pool}
    dup = {"action_ids": [], "note": "S5-D1 每候选盘一口定向井；无重复对",
           "per_disc_tables_equal_all_tiers": None,
           "per_state_law_tokens_equal_at_tier": None, "tier": tier}
    static = cle.StaticS1X(
        root="", model=model, spec=obs_spec, kernel=kernel, table=table, labels=labels,
        pool=pool, by_id=by_id, states=states, cost_int=cost_int,
        cost_domain_length=cost_dl, tier=tier, n_bits=n_bits,
        pmf_exact={}, dists_float=dists_float, law_tokens=law_tokens,
        support_records={}, support_sizes={}, separable={},
        read_masks=read_masks, azimuths=azimuths, ab_azimuth=float(ab_azimuth),
        exposure_rows=exposure_rows, duplicate_proof=dup, shared_cache={},
        provenance=provenance)
    static.mask_cache = cache
    return static


def make_expanded_world(static: cle.StaticS1X, spec: dict, *, noise_seed: int,
                        budget_int: int, alpha_seq, max_steps: int):
    return cle.ExpandedWorld(static, str(spec["world_id"]),
                             truth_state=int(spec["truth_state"]),
                             noise_seed=int(noise_seed), budget_int=int(budget_int),
                             alpha_seq=Fraction(alpha_seq), max_steps=int(max_steps))


class WorldFactory:
    def __init__(self, spec_by_id: dict, obs_spec: dict, kernel_id_expected: str,
                 *, cache_size: int = 1, pmf_cap: int = 400000):
        self.spec_by_id = dict(spec_by_id)
        self.obs_spec = obs_spec
        self.kernel_id_expected = str(kernel_id_expected)
        self.cache_size = int(cache_size)
        self.pmf_cap = int(pmf_cap)
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
        st = build_world_static(spec, self.obs_spec, self.kernel_id_expected,
                                tier=tier, pmf_cap=self.pmf_cap)
        self.seconds += time.perf_counter() - t0
        self.builds += 1
        self._cache[key] = st
        self._order.append(key)
        while len(self._order) > self.cache_size:
            self._cache.pop(self._order.pop(0), None)
        return st


# ---------------------------------------------------------------------------
# 头闸纯函数（effects / controls / gates 共用；禁各处手打）
# ---------------------------------------------------------------------------

def _runs(led) -> list:
    return list((led or {}).get("runs", []))


def _worlds(reg) -> list:
    return list((reg or {}).get("worlds", []))


def _run_key(r):
    return "%s|%s|%d" % (r["world_id"], r["strategy"], int(r["seed"]))


def _sd(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def compute_headroom(ledger: dict, registry: dict, prereg: dict) -> dict:
    """三项头闸的唯一计算入口（scope=全部声明世界，含平凡世界）。"""
    runs = _runs(ledger)
    worlds = _worlds(registry)
    if not runs:
        raise SystemExit("[s5d1] 空运行账：拒绝计算头闸")
    if not worlds:
        raise SystemExit("[s5d1] 空世界面：拒绝计算头闸")
    strategies = list(prereg["experiment"]["strategies"])
    seeds = [int(x) for x in prereg["experiment"]["seeds"]]
    wid_order = [str(w["world_id"]) for w in worlds]
    trivial_of = {str(w["world_id"]): bool(w["flags"]["trivial"]) for w in worlds}
    runnable = [str(w["world_id"]) for w in worlds if w["flags"]["runnable"]]
    if len(runnable) != len(wid_order):
        raise SystemExit("[s5d1] 存在不可跑世界：头闸分母不完整")

    by = {}
    for r in runs:
        by.setdefault((str(r["world_id"]), str(r["strategy"])), []).append(r)

    rate = {}
    for wid in wid_order:
        rate[wid] = {}
        for s in strategies:
            rs = by.get((wid, s), [])
            if len(rs) != len(seeds):
                raise SystemExit("[s5d1] 世界 %s/%s 运行数 %d != 声明 %d"
                                 % (wid, s, len(rs), len(seeds)))
            cc = sum(1 for r in rs if r["stop"]["class"] == "CERTIFIED_SINGLE"
                     and r.get("correct") is True)
            rate[wid][s] = cc / float(len(rs))

    diffs = [rate[wid]["NEW"] - rate[wid]["EC2_TWO_STEP"] for wid in wid_order]
    sigma_d = _sd(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_zero = sum(1 for d in diffs if d == 0)
    oracle = {}
    for wid in wid_order:
        vals = [rate[wid][s] for s in strategies]
        oracle[wid] = {"max": max(vals), "min": min(vals), "gap": max(vals) - min(vals),
                       "trivial": trivial_of[wid]}
    oracle_mean = sum(oracle[w]["gap"] for w in wid_order) / float(len(wid_order))
    oracle_nontrivial = [oracle[w]["gap"] for w in wid_order if not trivial_of[w]]

    cells = {}
    for w in worlds:
        cell = "%s|%s|%s" % (w["mechanism"], w["condition"], w["tier"])
        cells.setdefault(cell, []).append(str(w["world_id"]))
    cell_rows = {}
    n_nondegenerate = 0
    for cell in sorted(cells):
        wids = cells[cell]
        per = {}
        cell_nondegenerate = False
        any_mixed_rate = False
        for s in strategies:
            vals = [rate[w][s] for w in wids]
            all_zero = all(v == 0.0 for v in vals)
            all_one = all(v == 1.0 for v in vals)
            degenerate = bool(all_zero or all_one)
            if not degenerate:
                cell_nondegenerate = True
            if any(0.0 < v < 1.0 for v in vals):
                any_mixed_rate = True
            per[s] = {"values": vals, "all_zero": all_zero, "all_one": all_one,
                      "degenerate": degenerate, "mean": sum(vals) / len(vals)}
        if cell_nondegenerate:
            n_nondegenerate += 1
        cell_rows[cell] = {"n_worlds": len(wids), "by_strategy": per,
                           "nondegenerate": bool(cell_nondegenerate),
                           "any_mixed_within_world_rate": bool(any_mixed_rate)}
    n_cells = len(cell_rows)
    gate1 = bool(sigma_d >= 0.05)
    gate2 = bool(oracle_mean >= 0.10)
    gate3 = bool(n_nondegenerate >= 6)
    verdict = "HEADROOM_PASS" if (gate1 and gate2 and gate3) else "HEADROOM_FAIL"
    return {
        "schema": "s5d1.headroom.v1",
        "unit": "S5-D1",
        "scope": {"worlds": wid_order, "n_worlds": len(wid_order),
                  "includes_trivial": True,
                  "n_trivial": sum(1 for w in wid_order if trivial_of[w]),
                  "strategies": strategies, "seeds": seeds,
                  "enumeration": "每世界每策略 2 个种子的 CERTIFIED_SINGLE∧correct 计数率；"
                                 "oracle gap 逐世界 max−min 后对全部声明世界取均值"},
        "per_world_rate": rate,
        "sigma_d": {"contrast": "NEW-EC2_TWO_STEP", "sd": sigma_d,
                    "mean": sum(diffs) / len(diffs), "n": len(diffs),
                    "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
                    "both_signs": bool(n_pos > 0 and n_neg > 0),
                    "unit": "WORLD", "sigma_is_estimate": True,
                    "source_family": "S5D1_BINDING_BUDGET_CHALLENGE"},
        "oracle_gap": {"per_world": oracle, "mean": oracle_mean,
                       "n_worlds": len(wid_order), "includes_trivial": True,
                       "mean_nontrivial_only": (sum(oracle_nontrivial)
                                                / float(len(oracle_nontrivial))
                                                if oracle_nontrivial else None),
                       "n_nontrivial": len(oracle_nontrivial)},
        "nondegenerate_cells": {"per_cell": cell_rows, "n_cells": n_cells,
                                "n_nondegenerate": n_nondegenerate,
                                "rule": ("格内至少一个策略的 8 个世界级读数不全 0 且不全 1"
                                         " ⇒ 该格非退化")},
        "gate_readings": {
            "gate1_sigma_d_ge_0.05": {"pass": gate1, "value": sigma_d,
                                       "threshold": 0.05},
            "gate2_oracle_gap_ge_0.10": {"pass": gate2, "value": oracle_mean,
                                          "threshold": 0.10},
            "gate3_nondegenerate_cells_ge_6_of_8": {"pass": gate3,
                                                    "value": n_nondegenerate,
                                                    "threshold": 6, "n_cells": n_cells},
        },
        "verdict": verdict,
        "branch": ("S5_INPUTS_READY" if verdict == "HEADROOM_PASS"
                   else "H3_EXPLORATORY_ONLY"),
        "measured_at_utc": None,
        "measurement_note": ("本件由在盘脚本从 closed_loop_ledger.json#runs 重算；"
                             "报告字面不复制读数"),
    }


# ---------------------------------------------------------------------------
# 家族校验器
# ---------------------------------------------------------------------------

def _registry_required(reg):
    if not reg:
        raise SystemExit("[s5d1] 缺世界登记：拒绝记 PASS")
    worlds = _worlds(reg)
    if not worlds:
        raise SystemExit("[s5d1] 空世界面：拒绝记 PASS")
    return worlds


def _ledger_required(led):
    if led is None:
        raise SystemExit("[s5d1] 缺运行账：拒绝记 PASS")
    runs = _runs(led)
    if not runs:
        raise SystemExit("[s5d1] 空运行账：拒绝记 PASS")
    return runs


def _acted(r):
    return [s for s in r["steps"] if s.get("action_id") is not None]


def v_prereg_integrity(docs, ctx=None):
    pre = (ctx or {}).get("prereg")
    if not pre:
        raise SystemExit("[s5d1] 缺预注册：拒绝记 PASS")
    bad = []
    fp = pre.get("prereg_fingerprint", {}).get("canonical_sha12")
    if not fp:
        bad.append({"code": "PREREG_FINGERPRINT_MISSING"})
    want = (ctx or {}).get("prereg_fingerprint_recomputed")
    if want is not None and str(fp) != str(want):
        bad.append({"code": "PREREG_FINGERPRINT_DRIFT", "disk": fp, "recomputed": want})
    for key in ("root_seed", "budget_units", "max_steps", "dimension_primary",
                "dimension_fallback", "n_candidates_declared", "alpha_seq"):
        if key not in pre:
            bad.append({"code": "PREREG_FIELD_MISSING", "field": key})
    if int(pre.get("root_seed", -1)) != 20261021:
        bad.append({"code": "ROOT_SEED_DRIFT", "got": pre.get("root_seed")})
    if int(pre.get("budget_units", -1)) != 12:
        bad.append({"code": "BUDGET_DRIFT", "got": pre.get("budget_units")})
    if int(pre.get("max_steps", -1)) != 3:
        bad.append({"code": "MAX_STEPS_DRIFT", "got": pre.get("max_steps")})
    return bad


def v_family_denominator(docs, ctx=None):
    reg = docs.get("registry")
    pre = (ctx or {}).get("prereg") or {}
    worlds = _registry_required(reg)
    bad = []
    want_w = set(str(x) for x in pre.get("roster_worlds", []))
    if not want_w:
        raise SystemExit("[s5d1] prereg 名册为空：拒绝记 PASS")
    got = [str(w["world_id"]) for w in worlds]
    if len(got) != len(set(got)):
        bad.append({"code": "DUPLICATE_WORLD"})
    for m in sorted(want_w - set(got)):
        bad.append({"code": "MISSING_WORLD", "world": m})
    for e in sorted(set(got) - want_w):
        bad.append({"code": "EXTRA_WORLD", "world": e})
    per_cell = {}
    for w in worlds:
        cell = (w["mechanism"], w["condition"], w["tier"])
        per_cell[cell] = per_cell.get(cell, 0) + 1
    for cell, n in sorted(per_cell.items()):
        if n < int(pre.get("worlds_per_cell", 0)):
            bad.append({"code": "CELL_BELOW_DECLARED", "cell": list(cell), "n": n})
    for w in worlds:
        if not w["flags"]["runnable"]:
            bad.append({"code": "WORLD_NOT_RUNNABLE", "world": w["world_id"],
                        "reason": w["flags"].get("not_runnable_reason")})
    led = docs.get("ledger")
    if led is not None:
        runs = _ledger_required(led)
        strategies = list(pre.get("strategies", []))
        seeds = [int(x) for x in pre.get("seeds", [])]
        runnable = [str(w["world_id"]) for w in worlds if w["flags"]["runnable"]]
        want_runs = {(wid, s, k) for wid in runnable for s in strategies for k in seeds}
        got_runs = [(str(r["world_id"]), str(r["strategy"]), int(r["seed"])) for r in runs]
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
        if len(got_runs) != 64 * len(strategies) * len(seeds):
            bad.append({"code": "DENOMINATOR_SIZE", "got": len(got_runs),
                        "want": 64 * len(strategies) * len(seeds)})
    return bad


def v_world_replay(docs, ctx=None):
    reg = docs.get("registry")
    worlds = _registry_required(reg)
    ctx = ctx or {}
    pre = ctx.get("prereg") or {}
    params = pre.get("world_params")
    obs = ctx.get("obs_spec")
    decision = ctx.get("decision") or {}
    if not params:
        raise SystemExit("[s5d1] 缺 world_params：拒绝记 PASS")
    if not obs:
        raise SystemExit("[s5d1] 缺观测规格：拒绝记 PASS")
    dim = int(decision.get("decision", {}).get("dimension", 0))
    if dim not in (8, 12):
        raise SystemExit("[s5d1] 缺维度决策：拒绝记 PASS")
    mech_order = [str(x) for x in params["mechanism_order"]]
    cond_order = [str(x) for x in params["condition_order"]]
    tier_order = [str(x) for x in params["tier_order"]]
    root = int(params["root_seed"])
    bad = []
    for w in worlds:
        try:
            fresh = replay_world(world_id=w["world_id"], mechanism=w["mechanism"],
                                 condition=w["condition"], tier=w["tier"],
                                 k=int(w["world_code"]), root_seed=root,
                                 mech_index=mech_order.index(w["mechanism"]),
                                 cond_index=cond_order.index(w["condition"]),
                                 tier_index=tier_order.index(w["tier"]),
                                 dimension=dim, params=params, obs_spec=obs)
        except Exception as exc:                          # noqa: BLE001
            bad.append({"code": "WORLD_REPLAY_RAISED", "world": w["world_id"],
                        "error": repr(exc)})
            continue
        for field in ("geometry_sha12", "model_sha12"):
            if str(w.get(field)) != str(fresh.get(field)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field})
        for field in ("model", "generation", "selection", "flags", "facts", "exposure"):
            if canonical_json(w.get(field)) != canonical_json(fresh.get(field)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field})
        for field in ("n_bits", "n_states", "truth_state", "world_code",
                      "dimension_used"):
            if int(w.get(field, -1)) != int(fresh.get(field, -2)):
                bad.append({"code": "WORLD_REPLAY_MISMATCH", "world": w["world_id"],
                            "field": field})
    faces = ctx.get("faces")
    if isinstance(faces, dict):
        faces["v_world_replay"] = {"n_worlds_replayed": len(worlds)}
    return bad


def v_world_seed_independence(docs, ctx=None):
    reg = docs.get("registry")
    worlds = _registry_required(reg)
    bad = []
    seen_seed, seen_geo = {}, {}
    for w in worlds:
        wid = str(w["world_id"])
        seed = int(w["generation"]["stream_seed"])
        if seed in seen_seed:
            bad.append({"code": "WORLD_STREAM_SEED_COLLISION", "world": wid,
                        "other": seen_seed[seed]})
        seen_seed.setdefault(seed, wid)
        geo = str(w.get("geometry_sha12"))
        if geo in seen_geo:
            bad.append({"code": "DUPLICATE_GEOMETRY", "world": wid,
                        "other": seen_geo[geo]})
        seen_geo.setdefault(geo, wid)
    led = docs.get("ledger")
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
                        "other": list(by_val[ns])})
        by_val.setdefault(ns, pair)
    return bad


def v_no_label_generation(docs, ctx=None):
    reg = docs.get("registry")
    worlds = _registry_required(reg)
    bad = []
    for w in worlds:
        wid = str(w["world_id"])
        if [str(x) for x in w.get("phase_order", [])] != list(PHASE_ORDER):
            bad.append({"code": "LABEL_PHASE_ORDER", "world": wid})
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
        want_facts = {"contact_edges": [[int(i), int(j)] for i, j in facts["contact_edges"]],
                      "attach_a_disks": [int(x) for x in facts["attach_a_disks"]],
                      "attach_b_disks": [int(x) for x in facts["attach_b_disks"]]}
        if canonical_json(w.get("facts")) != canonical_json(want_facts):
            bad.append({"code": "LABELS_NOT_DERIVED_FROM_GEOMETRY", "world": wid,
                        "field": "facts"})
        all_same = bool(len(set(rec.values())) <= 1) if rec else True
        if bool(w["flags"].get("all_labels_same")) != bool(all_same):
            bad.append({"code": "TRIVIAL_FLAG_MISMATCH", "world": wid})
    return bad


def v_trivial_preserved(docs, ctx=None):
    reg = docs.get("registry")
    worlds = _registry_required(reg)
    bad = []
    for w in worlds:
        wid = str(w["world_id"])
        flags = w.get("flags") or {}
        if flags.get("trivial"):
            if not flags.get("trivial_kinds"):
                bad.append({"code": "TRIVIAL_NOT_LISTED", "world": wid})
        if not flags.get("runnable") and not flags.get("not_runnable_reason"):
            bad.append({"code": "NOT_RUNNABLE_WITHOUT_REASON", "world": wid})
    led = docs.get("ledger")
    if led is not None:
        runnable = {str(w["world_id"]) for w in worlds if w["flags"]["runnable"]}
        in_ledger = {str(r["world_id"]) for r in _runs(led)}
        for wid in sorted(runnable - in_ledger):
            bad.append({"code": "WORLD_FILTERED_AFTER_DRAW", "world": wid})
    return bad


def v_well_pool(docs, ctx=None):
    reg = docs.get("registry")
    pre = (ctx or {}).get("prereg") or {}
    worlds = _registry_required(reg)
    budget = int(pre.get("budget_units", -1))
    per_well = int(pre.get("well_cost_units", -1))
    n_declared = int(pre.get("n_candidates_declared", -1))
    bad = []
    n_worlds_multi_mask = 0
    global_masks = set()
    for w in worlds:
        wid = str(w["world_id"])
        n_bits = int(w["n_bits"])
        pool = list(w["model"].get("action_pool", []))
        if len(pool) != n_bits:
            bad.append({"code": "WELL_POOL_INCOMPLETE", "world": wid,
                        "n_wells": len(pool), "n_bits": n_bits})
        targets = [int(a.get("target_candidate", -1)) for a in pool]
        if sorted(targets) != list(range(n_bits)):
            bad.append({"code": "WELL_TARGET_MISMATCH", "world": wid,
                        "targets": sorted(targets)})
        masks = []
        for a in pool:
            if int(round(float(a["cost"]) * 100)) != per_well:
                bad.append({"code": "WELL_COST_DRIFT", "world": wid,
                            "action": a["action_id"], "cost": a["cost"]})
            m = a.get("exposure_mask")
            if m is None:
                bad.append({"code": "EXPOSURE_MASK_MISSING", "world": wid,
                            "action": a["action_id"]})
                continue
            masks.append(int(m))
            tgt = int(a["target_candidate"])
            if not ((int(m) >> tgt) & 1):
                bad.append({"code": "TARGET_NOT_EXPOSED", "world": wid,
                            "action": a["action_id"], "target": tgt})
        global_masks.update(masks)
        if len(set(masks)) > 1:
            n_worlds_multi_mask += 1
        if n_bits * per_well <= budget:
            bad.append({"code": "BUDGET_NOT_BINDING", "world": wid,
                        "pool_cost": n_bits * per_well, "budget": budget})
        if n_bits != n_declared and not w.get("fallback_applied"):
            bad.append({"code": "DIMENSION_NOT_DECLARED", "world": wid, "n_bits": n_bits})
    if len(global_masks) < 2:
        bad.append({"code": "EXPOSURE_MASKS_DEGENERATE", "n_distinct": len(global_masks)})
    if len(worlds) and n_worlds_multi_mask < int(pre.get("min_worlds_multi_mask", 0)):
        bad.append({"code": "EXPOSURE_MASKS_DEGENERATE",
                    "n_worlds_multi_mask": n_worlds_multi_mask,
                    "min_required": int(pre.get("min_worlds_multi_mask", 0))})
    faces = (ctx or {}).get("faces")
    if isinstance(faces, dict):
        faces["v_well_pool"] = {"n_worlds": len(worlds),
                                "n_worlds_multi_mask": n_worlds_multi_mask,
                                "n_distinct_masks_global": len(global_masks)}
    return bad


def v_family_budget(docs, ctx=None):
    led = docs.get("ledger")
    pre = (ctx or {}).get("prereg") or {}
    runs = _ledger_required(led)
    budget = int(pre.get("budget_units", -1))
    max_steps = int(pre.get("max_steps", -1))
    per_well = int(pre.get("well_cost_units", -1))
    bad = []
    for r in runs:
        used = 0
        acted = []
        for s in r["steps"]:
            if s.get("action_id") is None:
                continue
            acted.append(s["action_id"])
            if s["action_id"] in acted[:-1]:
                bad.append({"code": "REPEATED_ACTION", "run": _run_key(r),
                            "action": s["action_id"]})
            if int(s.get("cost", -1)) != per_well:
                bad.append({"code": "STEP_COST_DRIFT", "run": _run_key(r),
                            "cost": s.get("cost")})
            if used + int(s.get("cost", 0)) > budget:
                bad.append({"code": "OVER_BUDGET", "run": _run_key(r), "t": s["t"]})
            used += int(s.get("cost", 0))
        if used != int(r.get("used", -1)):
            bad.append({"code": "USED_MISMATCH", "run": _run_key(r),
                        "declared": r.get("used"), "sum": used})
        if len(acted) != int(r.get("n_steps", -1)):
            bad.append({"code": "STEP_COUNT_MISMATCH", "run": _run_key(r),
                        "rows": len(acted), "n_steps": r.get("n_steps")})
        if len(acted) > max_steps:
            bad.append({"code": "MAX_STEPS_EXCEEDED", "run": _run_key(r),
                        "n_steps": len(acted), "max_steps": max_steps})
        if int(r.get("budget", -1)) != budget:
            bad.append({"code": "RUN_BUDGET_DRIFT", "run": _run_key(r),
                        "budget": r.get("budget")})
        if int(r.get("max_steps", -1)) != max_steps:
            bad.append({"code": "RUN_MAX_STEPS_DRIFT", "run": _run_key(r),
                        "max_steps": r.get("max_steps")})
    return bad


def v_outcome_accounting(docs, ctx=None):
    led = docs.get("ledger")
    runs = _ledger_required(led)
    bad = []
    for r in runs:
        acted = _acted(r)
        if len(acted) != int(r.get("n_steps", -1)):
            bad.append({"code": "STEP_COUNT_MISMATCH", "run": _run_key(r)})
        for s in acted:
            if s.get("record_key") is None or s.get("record_sha12") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r),
                            "t": s["t"]})
            if s.get("used_after") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r),
                            "t": s["t"]})
        if not r["steps"] or r["steps"][-1].get("certificate") is None:
            bad.append({"code": "MISSING_TERMINAL_ROW", "run": _run_key(r)})
        if (r.get("stop") or {}).get("class") not in RUN_OUTCOME_CLASSES:
            bad.append({"code": "UNKNOWN_OUTCOME_CLASS", "run": _run_key(r)})
    return bad


def v_run_integrity(docs, ctx=None):
    led = docs.get("ledger")
    runs = _ledger_required(led)
    pre = (ctx or {}).get("prereg") or {}
    cap_run = float(pre.get("cost_caps", {}).get("per_run_wall_clock_seconds", 120))
    bad = []
    for r in runs:
        key = _run_key(r)
        attempts = r.get("attempts") or []
        failed = [a for a in attempts if str(a.get("status")) != "COMPLETED"]
        if failed and not bool(r.get("cap_retained", False)):
            bad.append({"code": "RERUN_REPLACED", "run": key,
                        "failed_attempts": len(failed)})
        if bool(r.get("cap_retained", False)) and str(r.get("cap_kind")) not in (
                "WALL_CLOCK_CAP_HIT", "COST_CEILING_HIT", "NONE"):
            bad.append({"code": "CAP_KIND_UNKNOWN", "run": key,
                        "cap_kind": r.get("cap_kind")})
        rs = r.get("run_seconds")
        if rs is None:
            bad.append({"code": "RUN_SECONDS_MISSING", "run": key})
        elif float(rs) > cap_run and r.get("cap_kind") != "WALL_CLOCK_CAP_HIT":
            bad.append({"code": "WALL_CLOCK_CAP_UNRECORDED", "run": key,
                        "run_seconds": rs, "cap": cap_run})
    return bad


def v_certificates(docs, ctx=None):
    led = docs.get("ledger")
    runs = _ledger_required(led)
    bad = []
    for r in runs:
        terminal = None
        for s in r["steps"]:
            c = s.get("certificate")
            if not c:
                bad.append({"code": "MISSING_CERTIFICATE_FIELD", "run": _run_key(r),
                            "t": s["t"]})
                continue
            if c.get("issued") and not c.get("verified"):
                bad.append({"code": "CERTIFICATE_NOT_VERIFIED", "run": _run_key(r),
                            "t": s["t"]})
            if c.get("issued") and int(c.get("allowed_n", 0)) <= 0:
                bad.append({"code": "CERTIFICATE_ON_EMPTY_SET", "run": _run_key(r),
                            "t": s["t"]})
            if c.get("issued"):
                terminal = s["t"]
                if str(c.get("label_claim")) not in ("0", "1"):
                    bad.append({"code": "CERTIFICATE_LABEL_INVALID", "run": _run_key(r),
                                "t": s["t"]})
                ids = s.get("allowed_state_ids")
                if ids is not None and int(r["truth_state"]) in set(int(x) for x in ids):
                    if int(c["label_claim"]) != int(r["truth_label"]):
                        bad.append({"code": "WRONG_CERT_WHILE_TRUTH_POSSIBLE",
                                    "run": _run_key(r), "t": s["t"]})
        if (r.get("stop") or {}).get("class") == "CERTIFIED_SINGLE" and terminal is None:
            bad.append({"code": "CERTIFIED_WITHOUT_CERTIFICATE", "run": _run_key(r)})
        if int(r.get("certificates_unverified", -1)) != 0:
            bad.append({"code": "UNVERIFIED_CERTIFICATE_COUNT", "run": _run_key(r),
                        "n": r.get("certificates_unverified")})
    return bad


def v_config_vs_prereg(docs, ctx=None):
    led = docs.get("ledger")
    pre = (ctx or {}).get("prereg") or {}
    runs = _ledger_required(led)
    bad = []
    by_world = {}
    for r in runs:
        by_world.setdefault(str(r["world_id"]), []).append(r)
    for wid, rs in sorted(by_world.items()):
        base = None
        for r in rs:
            key = (int(r["budget"]), str(r["alpha_seq"]), int(r["max_steps"]),
                   int(r["n_bits"]))
            if base is None:
                base = key
            elif key != base:
                bad.append({"code": "CONFIG_DRIFT_ACROSS_STRATEGIES", "world": wid,
                            "run": _run_key(r)})
        want = (int(pre.get("budget_units", -1)), str(pre.get("alpha_seq")),
                int(pre.get("max_steps", -1)), int(pre.get("dimension_primary", -1)))
        if base is not None:
            if base[0] != want[0] or base[1] != want[1] or base[2] != want[2]:
                bad.append({"code": "CONFIG_DRIFT_VS_PREREG", "world": wid,
                            "got": list(base[:3]), "want": list(want[:3])})
            dim = int((ctx or {}).get("decision", {}).get("decision", {}).get(
                "dimension", want[3]))
            if base[3] != dim:
                bad.append({"code": "DIMENSION_DRIFT", "world": wid,
                            "got": base[3], "want": dim})
    return bad


def v_noise_tier_consistency(docs, ctx=None):
    reg = docs.get("registry")
    led = docs.get("ledger")
    worlds = _registry_required(reg)
    runs = _ledger_required(led)
    tier_of = {str(w["world_id"]): str(w["tier"]) for w in worlds}
    bad = []
    for r in runs:
        want = tier_of.get(str(r["world_id"]))
        if want is None:
            bad.append({"code": "NOISE_TIER_DRIFT", "run": _run_key(r),
                        "reason": "world_not_in_registry"})
        elif str(r.get("tier")) != want:
            bad.append({"code": "NOISE_TIER_DRIFT", "run": _run_key(r),
                        "got": r.get("tier"), "want": want})
    return bad


def v_public_choice_replay(docs, ctx=None):
    led = docs.get("ledger")
    runs = _ledger_required(led)
    make_world = (ctx or {}).get("make_world")
    if make_world is None:
        raise SystemExit("[s5d1] 校验缺少世界工厂：拒绝记 PASS")
    sample = (ctx or {}).get("replay_sample")
    import random as _random
    bad = []
    n_replayed = 0
    n_steps = 0
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
            if s.get("action_id") is None:
                continue
            n_steps += 1
            try:
                pub = world.public(allowed, used, executed, s["t"] - 1,
                                   world.history_sha12(steps))
                ch = cle.choose_action_expanded(r["strategy"], world, pub, rng,
                                                read_so_far)
            except Exception as exc:                      # noqa: BLE001
                bad.append({"code": "REPLAY_RAISED", "run": _run_key(r), "t": s["t"],
                            "error": repr(exc)})
                break
            if str(ch["action_id"]) != str(s["action_id"]):
                bad.append({"code": "ACTION_NOT_REPRODUCIBLE_FROM_PUBLIC_HISTORY",
                            "run": _run_key(r), "t": s["t"],
                            "recorded": s["action_id"],
                            "recomputed": ch["action_id"]})
            if s.get("record_key") is None:
                bad.append({"code": "OUTCOME_NOT_BOOKED", "run": _run_key(r),
                            "t": s["t"]})
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
        raise SystemExit("[s5d1] 重放采样为空：拒绝记 PASS")
    faces = (ctx or {}).get("faces")
    if isinstance(faces, dict):
        faces["v_public_choice_replay"] = {"n_runs_replayed": n_replayed,
                                           "n_steps_replayed": n_steps,
                                           "sample_declared": sample is not None}
    return bad


def v_cost_two_sided(docs, ctx=None):
    led = docs.get("ledger")
    runs = _ledger_required(led)
    bad = []
    for r in runs:
        c = r.get("cost") or {}
        if "observed" not in c:
            bad.append({"code": "COST_MISSING_OBSERVED", "run": _run_key(r)})
        if "compute_seconds" not in c:
            bad.append({"code": "COST_MISSING_COMPUTE", "run": _run_key(r)})
        if c.get("basis") != "MEASURED_ONLY":
            bad.append({"code": "COST_BASIS_NOT_MEASURED", "run": _run_key(r)})
        if r.get("static_build_seconds") is None:
            bad.append({"code": "STATIC_SECONDS_MISSING", "run": _run_key(r)})
    return bad


def v_world_effects_aggregation(docs, ctx=None):
    eff = docs.get("effects")
    led = docs.get("ledger")
    reg = docs.get("registry")
    if eff is None or led is None or reg is None:
        raise SystemExit("[s5d1] 缺世界效果/运行账/登记：拒绝记 PASS")
    worlds = _registry_required(reg)
    runs = _ledger_required(led)
    pre = (ctx or {}).get("prereg") or {}
    strategies = list(pre.get("strategies", []))
    bad = []
    per_world = eff.get("per_world") or {}
    for w in worlds:
        wid = str(w["world_id"])
        if wid not in per_world:
            bad.append({"code": "WORLD_AGG_MISSING", "world": wid})
            continue
        for s in strategies:
            row = per_world[wid].get(s)
            if row is None:
                bad.append({"code": "WORLD_AGG_MISSING", "world": wid, "strategy": s})
                continue
            rs = [r for r in runs if str(r["world_id"]) == wid and r["strategy"] == s]
            if not rs:
                bad.append({"code": "ZERO_DENOMINATOR_ROW", "world": wid, "strategy": s})
                continue
            cc = sum(1 for r in rs if r["stop"]["class"] == "CERTIFIED_SINGLE"
                     and r.get("correct") is True)
            if int(row.get("certified_correct", -1)) != cc:
                bad.append({"code": "WORLD_EFFECT_MISMATCH", "world": wid,
                            "strategy": s, "declared": row.get("certified_correct"),
                            "recomputed": cc})
    paired = eff.get("paired") or {}
    strata_worlds = eff.get("strata_worlds") or {}
    if not paired:
        bad.append({"code": "PAIRING_MISSING"})
    for stratum, rows in paired.items():
        want = strata_worlds.get(stratum)
        if want is None:
            bad.append({"code": "PAIRING_SCOPE_MISSING", "stratum": stratum})
        for key, block in rows.items():
            diffs = block.get("world_diffs") or []
            if not diffs:
                bad.append({"code": "PAIRING_INCOMPLETE", "stratum": stratum, "key": key})
                continue
            if block.get("unit") != "WORLD":
                bad.append({"code": "PAIRING_UNIT_DRIFT", "stratum": stratum, "key": key})
            if block.get("sigma_is_estimate") is not True:
                bad.append({"code": "SIGMA_INPUT_MISSING", "stratum": stratum, "key": key})
            if want is not None:
                if sorted(block.get("world_ids") or []) != sorted(want):
                    bad.append({"code": "PAIRING_SCOPE_MISMATCH", "stratum": stratum,
                                "key": key})
                if len(diffs) != len(want):
                    bad.append({"code": "PAIRING_LENGTH_MISMATCH", "stratum": stratum,
                                "key": key})
    if not (eff.get("sigma_h") or {}):
        bad.append({"code": "SIGMA_INPUT_MISSING", "scope": "sigma_h"})
    return bad


def v_oracle_scope(docs, ctx=None):
    eff = docs.get("effects")
    led = docs.get("ledger")
    reg = docs.get("registry")
    pre = (ctx or {}).get("prereg") or {}
    if eff is None or led is None or reg is None:
        raise SystemExit("[s5d1] 缺效果/账/登记：拒绝记 PASS")
    worlds = _registry_required(reg)
    declared = (eff.get("oracle_gap") or {})
    recomputed = compute_headroom(led, reg, pre)
    bad = []
    if int(declared.get("n_worlds", -1)) != len(worlds):
        bad.append({"code": "ORACLE_SCOPE_MISMATCH", "declared": declared.get("n_worlds"),
                    "want": len(worlds)})
    if declared.get("includes_trivial") is not True:
        bad.append({"code": "ORACLE_SCOPE_MISMATCH", "field": "includes_trivial"})
    if abs(float(declared.get("mean", -1)) - recomputed["oracle_gap"]["mean"]) > 1e-9:
        bad.append({"code": "ORACLE_READING_MISMATCH",
                    "declared": declared.get("mean"),
                    "recomputed": recomputed["oracle_gap"]["mean"]})
    return bad


def v_power_inputs(docs, ctx=None):
    powd = docs.get("power")
    eff = docs.get("effects")
    pre = (ctx or {}).get("prereg") or {}
    if powd is None:
        raise SystemExit("[s5d1] 缺功效输入：拒绝记 PASS")
    if eff is None:
        raise SystemExit("[s5d1] 缺效应件：拒绝记 PASS")
    bad = []
    if float(powd.get("delta_min", -1)) != 0.02:
        bad.append({"code": "DELTA_MIN_DRIFT", "got": powd.get("delta_min")})
    if powd.get("sigma_source_kind") != "ESTIMATED_FROM_THIS_FAMILY":
        bad.append({"code": "SIGMA_SOURCE_FOREIGN",
                    "got": powd.get("sigma_source_kind")})
    if powd.get("sigma_is_estimate") is not True:
        bad.append({"code": "SIGMA_CLAIMED_KNOWN"})
    for k in ("n_required_final", "guaranteed_n", "commitment", "promised_sample_size",
              "sample_size_defined"):
        if k in powd:
            bad.append({"code": "POWER_PROMISE_FIELD", "field": k})
    if powd.get("cross_family_pooling") is not False:
        bad.append({"code": "CROSS_FAMILY_POOLING"})
    for q in ("sigma_h", "sigma_d", "c_h", "n_required"):
        if q not in powd or not powd.get(q):
            bad.append({"code": "POWER_INPUT_MISSING", "quantity": q})
    approval = powd.get("approval") or {}
    for q in ("sigma_h", "sigma_d", "c_h", "n_required"):
        if approval.get(q) is not None:
            bad.append({"code": "APPROVAL_FIELD_FILLED", "quantity": q,
                        "value": approval.get(q)})
    sig = (eff.get("headroom") or {}).get("sigma_d") or {}
    if sig:
        if abs(float(powd["sigma_d"].get("value", -1)) - float(sig.get("sd", -2))) > 1e-12:
            bad.append({"code": "SIGMA_D_MISMATCH",
                        "declared": powd["sigma_d"].get("value"),
                        "recomputed": sig.get("sd")})
        if powd["sigma_d"].get("source_family") != "S5D1_BINDING_BUDGET_CHALLENGE":
            bad.append({"code": "SIGMA_SOURCE_FOREIGN",
                        "got": powd["sigma_d"].get("source_family")})
    c_h = powd.get("c_h") or {}
    if int(c_h.get("n_worlds", -1)) != 64:
        bad.append({"code": "C_H_SCOPE_MISMATCH", "n_worlds": c_h.get("n_worlds")})
    if c_h.get("basis") != "MEASURED_ONLY":
        bad.append({"code": "C_H_BASIS_NOT_MEASURED", "basis": c_h.get("basis")})
    return bad


def v_headroom_gate(docs, ctx=None):
    eff = docs.get("effects")
    led = docs.get("ledger")
    reg = docs.get("registry")
    pre = (ctx or {}).get("prereg") or {}
    if eff is None or led is None or reg is None:
        raise SystemExit("[s5d1] 缺效果/账/登记：拒绝头闸判定")
    declared = eff.get("headroom") or {}
    if not declared:
        raise SystemExit("[s5d1] 缺头闸判定：拒绝记 PASS")
    recomputed = compute_headroom(led, reg, pre)
    bad = []
    if str(declared.get("verdict")) != recomputed["verdict"]:
        bad.append({"code": "HEADROOM_BRANCH_MISMATCH",
                    "declared": declared.get("verdict"),
                    "recomputed": recomputed["verdict"]})
    if str(declared.get("branch")) != recomputed["branch"]:
        bad.append({"code": "HEADROOM_BRANCH_MISMATCH", "field": "branch",
                    "declared": declared.get("branch"),
                    "recomputed": recomputed["branch"]})
    dg = declared.get("gate_readings") or {}
    rg = recomputed["gate_readings"]
    for k in ("gate1_sigma_d_ge_0.05", "gate2_oracle_gap_ge_0.10",
              "gate3_nondegenerate_cells_ge_6_of_8"):
        if bool((dg.get(k) or {}).get("pass")) != bool(rg[k]["pass"]):
            bad.append({"code": "HEADROOM_READING_MISMATCH", "gate": k,
                        "declared": (dg.get(k) or {}).get("pass"),
                        "recomputed": rg[k]["pass"]})
    scope = declared.get("scope") or {}
    if int(scope.get("n_worlds", -1)) != len(_worlds(reg)):
        bad.append({"code": "HEADROOM_SUBSET_SCOPE",
                    "declared": scope.get("n_worlds"), "want": len(_worlds(reg))})
    if declared.get("verdict") == "HEADROOM_PASS":
        if not all(rg[k]["pass"] for k in rg):
            bad.append({"code": "HEADROOM_PASS_WITH_FAILED_GATE"})
    return bad


def v_grading_consistency(docs=None, ctx=None):
    ctx = ctx or {}
    v = ctx.get("verdict") or {}
    if not v:
        raise SystemExit("[s5d1] 判级面为空：拒绝记 PASS")
    GRADING_VALUES = ("ACCEPTED", "REJECTED", "PARTIAL", "NOT_DELIVERED", "UNRESOLVED")
    bad = []
    g = str(v.get("grading"))
    if g not in GRADING_VALUES:
        bad.append({"code": "GRADING_VALUE_UNKNOWN", "grading": v.get("grading")})
        return bad
    ctl = (docs or {}).get("controls")
    led = (docs or {}).get("ledger")
    head = v.get("headroom_verdict")
    if head not in ("HEADROOM_PASS", "HEADROOM_FAIL", None):
        bad.append({"code": "HEADROOM_VERDICT_UNKNOWN", "value": head})
    if g == "ACCEPTED":
        if ctl is not None and not ctl.get("all_caught"):
            bad.append({"code": "GRADING_UNSUPPORTED", "reason": "controls_not_all_caught"})
        if ctl is not None and not (ctl.get("clean_control") or {}).get("clean_ok", False):
            bad.append({"code": "GRADING_UNSUPPORTED",
                        "reason": "clean_control_not_silent"})
        if ctl is not None and not (ctl.get("no_op") or {}).get(
                "identical_to_clean_sample", False):
            bad.append({"code": "GRADING_UNSUPPORTED", "reason": "noop_not_identical"})
        if led is not None:
            unver = sum(int(r.get("certificates_unverified", 0)) for r in _runs(led))
            if unver:
                bad.append({"code": "GRADING_UNSUPPORTED",
                            "reason": "unverified_certificates"})
        if head == "HEADROOM_PASS":
            eff = (docs or {}).get("effects") or {}
            h = eff.get("headroom") or {}
            gates = h.get("gate_readings") or {}
            if not all((gates.get(k) or {}).get("pass")
                       for k in ("gate1_sigma_d_ge_0.05",
                                 "gate2_oracle_gap_ge_0.10",
                                 "gate3_nondegenerate_cells_ge_6_of_8")):
                bad.append({"code": "GRADING_UNSUPPORTED",
                            "reason": "headroom_pass_without_three_gates"})
    return bad


def v_claim_scope(docs=None, ctx=None):
    ctx = ctx or {}
    phrases = [str(x) for x in (ctx.get("phrases") or [])]
    texts = [str(x) for x in (ctx.get("texts") or [])]
    fields = ctx.get("claim_scopes") or []
    if not phrases:
        raise SystemExit("[s5d1] 空禁用词表：拒绝记 PASS")
    if not texts and not fields:
        raise SystemExit("[s5d1] 空主张面：拒绝记 PASS")
    bad = []
    for t in texts:
        for p in phrases:
            if p and p in t:
                bad.append({"code": "FORBIDDEN_CLAIM_PHRASE", "phrase": p,
                            "context": t[:80]})
    for f in fields:
        if str(f) != "DEVELOPMENT_ONLY":
            bad.append({"code": "CLAIM_SCOPE_VIOLATION", "claim_scope": f})
    approval = (ctx.get("approval") or {})
    for q in ("sigma_h", "sigma_d", "c_h", "n_required"):
        if approval.get(q) is not None:
            bad.append({"code": "APPROVAL_FIELD_FILLED", "quantity": q})
    return bad


FAMILY_VALIDATORS = {
    "v_prereg_integrity": v_prereg_integrity,
    "v_family_denominator": v_family_denominator,
    "v_world_replay": v_world_replay,
    "v_world_seed_independence": v_world_seed_independence,
    "v_no_label_generation": v_no_label_generation,
    "v_trivial_preserved": v_trivial_preserved,
    "v_well_pool": v_well_pool,
    "v_family_budget": v_family_budget,
    "v_outcome_accounting": v_outcome_accounting,
    "v_run_integrity": v_run_integrity,
    "v_certificates": v_certificates,
    "v_config_vs_prereg": v_config_vs_prereg,
    "v_noise_tier_consistency": v_noise_tier_consistency,
    "v_public_choice_replay": v_public_choice_replay,
    "v_cost_two_sided": v_cost_two_sided,
    "v_world_effects_aggregation": v_world_effects_aggregation,
    "v_oracle_scope": v_oracle_scope,
    "v_power_inputs": v_power_inputs,
    "v_headroom_gate": v_headroom_gate,
    "v_claim_scope": v_claim_scope,
    "v_grading_consistency": v_grading_consistency,
}


def validate_family(docs, ctx=None, only=None) -> dict:
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
