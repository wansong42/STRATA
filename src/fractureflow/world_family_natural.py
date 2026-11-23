# -*- coding: utf-8 -*-
"""S4-E · d=12 自然世界族（固定世界坐标端点）+ 18 候选入口网格（本单唯一新核心模块）。

判据权威 = 任务单 `results/architect_main_20261012_dispatch/
task_order_S4_E_natural_family_d12_20261019.md` §2/§4 与本线
`results/s4e_natural_family_d12/prereg_s4e.json`（冻结件）。

**本件只产 S5 的输入（σ_h/σ_d/c_h/样本量依据），不做 S5 主比较、不出确认结论。**
水印 = SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS / COMPLETE_KNOWN / DEVELOPMENT_ONLY。

与 S4-W 挑战族的三处本质差别（任务单 §1 F1 / RULING-23 ⑭）：
  1. 盘数 d=12（状态空间 4096/世界），不是 4；
  2. 端点 = **固定世界坐标** A=(0.15,0.5,0.5)、B=(0.85,0.5,0.5)，不是候选盘盘心；
  3. 行动池 = 18 个**盲网格入口**（x∈{0.2,0.5,0.8} × y∈{0.3,0.7} × z=0 × 3 方向），
     不是"一候选盘一口定向井"。
⇒ 本族的井**不保证**曝光任何盘；曝光完全由核几何决定。世界 id 与 S4-W 不共享命名空间。

复用（逐字，不改既有线字节）：
  * 机制实现 / 几何事实 / 标签派生：``world_family_dev``（``e25e8c111ffe``）
  * 五策略闭环 + 证书协议：``closed_loop_expanded``（``81448d28336c``）
  * 证书求解器：``statistical_certificate`` / ``certificate_solver``
  * 观测核：``observation_kernel``（代码字节零改；本线只换 spec 并重算 kernel_id）

有限范围钉：``finite_scope``（``da62412af696``）语义电池由 S1-F 负责；在 S1-F 完成前，
本线一切读数带 ``FINITE_SCOPE_PINNED_UNVERIFIED`` 与现盘 sha12。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from fractions import Fraction
from typing import Mapping

import numpy as np

from . import certificate_solver as cs
from . import closed_loop_expanded as cle
from . import dfn
from . import finite_scope as fs
from . import identification as ident
from . import world_family_dev as wfd
from .generators import cluster as gcluster
from .generators.base import MarginalSpec
from .observation_kernel import (ComputationUnfinished, GridDetectionKernel,
                                 kernel_identity)
from .sequential_evidence import bits_of

__all__ = [
    "WORLD_SCHEMA", "FAMILY_SCHEMA", "TIER", "MECHANISMS", "PHASE_ORDER",
    "NaturalFamilyError", "canonical_json", "sha12_obj", "sha12_file",
    "world_id_of", "world_stream_seed", "generate_world", "replay_world",
    "select_candidates_of", "derive_labels", "geometry_facts",
    "build_world_static", "make_expanded_world", "WorldFactory",
    "sparse_pmf", "audit_sparse_pmf_matches_kernel",
    "FINITE_SCOPE_PIN", "FINITE_SCOPE_SHA12",
]

WORLD_SCHEMA = "s4e.world.v1"
FAMILY_SCHEMA = "s4e.family_registry.v1"
TIER = "N1"
MECHANISMS = ("BAECHER", "CLUSTER")

PHASE_ORDER = ("REALIZATION", "CANDIDATE_CROP", "QUERY_ENDPOINTS_FIXED",
               "WELL_GRID", "TRUTH_STATE_DRAW", "LABELS_DERIVED")

TRIVIAL_WORLD_KINDS = ("ALL_LABELS_SAME", "NO_DISCS", "NO_WELLS",
                       "FEWER_THAN_TWELVE_CANDIDATES")

FINITE_SCOPE_PIN = "FINITE_SCOPE_PINNED_UNVERIFIED"
_HERE = os.path.dirname(os.path.abspath(__file__))
FINITE_SCOPE_SHA12 = hashlib.sha256(
    open(os.path.join(_HERE, "finite_scope.py"), "rb").read()).hexdigest()[:12]


class NaturalFamilyError(RuntimeError):
    """自然族实现缺陷（不是读数）。"""


def canonical_json(obj) -> str:
    return cle.canonical_json(obj)


def sha12_obj(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:12]


def sha12_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def world_id_of(mechanism: str, condition: str, attach_code: str, k: int,
                tier: str = TIER) -> str:
    """本族 id 命名空间（与 S4W-* 不共享前缀）。"""
    return "S4E-%s-%s-%s-%s-%02d" % (str(mechanism), str(condition),
                                     str(attach_code), str(tier), int(k))


def world_stream_seed(root_seed: int, mech_index: int, cond_index: int,
                      attach_index: int, k: int) -> int:
    ss = np.random.SeedSequence([int(root_seed), int(mech_index), int(cond_index),
                                 int(attach_index), int(k)])
    return int(ss.generate_state(1)[0])


# ---------------------------------------------------------------------------
# 族构造（只看几何；标签最后派生）
# ---------------------------------------------------------------------------

def select_candidates_of(real, cond: dict, params: dict):
    """几何合格性 + 取前 n_candidates 个（source_index 升序）。

    **不按到 A–B 的距离排序**（v2.0 §10.2 明令禁止"只挑离 A–B 最近的若干盘"）。
    合格 = 包围球在 [0,1]^3 内（含 containment_margin）∧ r ≤ 条件 r_max。
    """
    pts = np.asarray(real.centers, dtype=np.float64)
    rads = np.asarray(real.radii, dtype=np.float64)
    margin = float(params.get("containment_margin", 0.01))
    r_cap = float(cond["r_max"])
    rows = []
    for i in range(int(len(pts))):
        c = pts[i]
        r = float(rads[i])
        in_dom = bool(np.all(c - r >= margin - 1e-12) and
                      np.all(c + r <= 1.0 - margin + 1e-12))
        eligible = bool(in_dom and (r <= r_cap + 1e-12))
        rows.append({"source_index": int(i), "center": [float(x) for x in c],
                     "radius": r, "eligible": eligible})
    elig = [x for x in rows if x["eligible"]]
    return rows, elig


def geometry_facts(discs, query) -> dict:
    """接触图 + A/B attach（逐字复用 world_family_dev 的公开几何派生）。"""
    return wfd.geometry_facts_of(discs, query)


def derive_labels(discs, query, n_bits: int) -> dict:
    """逐状态标签 = 接触图连通性（与 S1-X / S4-W 同一码路）。"""
    return wfd.derive_labels_of_world(discs, query, n_bits)


# ---------------------------------------------------------------------------
# 机制实现（K 组 vMF；与 world_family_dev 同一底层生成器）
# ---------------------------------------------------------------------------

def _realization_window(window: dict):
    lo = np.asarray(window["lo"], dtype=np.float64)
    hi = np.asarray(window["hi"], dtype=np.float64)
    return lo, hi, tuple(float(x) for x in (hi - lo)), (lo + hi) / 2.0


def _baecher_realization(cond: dict, seed: int, window: dict):
    lo, hi, extents, center = _realization_window(window)
    st = dfn.SetTable.from_centers(
        np.asarray(cond["set_centers"], dtype=np.float64), kappa=float(cond["kappa"]))
    real = dfn.generate_dfn(st, p32=float(cond["p32"]), beta=float(cond["beta"]),
                            domain=extents, seed=int(seed),
                            n_min=int(cond["n_min"]), r_min=float(cond["r_min"]),
                            r_max=float(cond["r_max"]))
    real.centers = np.asarray(real.centers, dtype=np.float64) + center[None, :]
    real.domain = (1.0, 1.0, 1.0)
    return real, {"generator": "dfn.generate_dfn",
                  "call": {"p32": float(cond["p32"]), "beta": float(cond["beta"]),
                           "n_sets": int(cond["n_sets"]), "kappa": float(cond["kappa"]),
                           "set_centers": cond["set_centers"],
                           "generation_window": {"lo": [float(x) for x in lo],
                                                 "hi": [float(x) for x in hi]},
                           "n_min": int(cond["n_min"]),
                           "r_min": float(cond["r_min"]), "r_max": float(cond["r_max"])}}


def _cluster_realization(cond: dict, mech_p: dict, seed: int, window: dict):
    lo, hi, extents, center = _realization_window(window)
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
                           "cluster_seed": cluster_seed, "n_pop": n_pop}}


def _generate_realization(mechanism: str, cond: dict, mech_p: dict, seed: int,
                          window: dict):
    if mechanism == "BAECHER":
        return _baecher_realization(cond, seed, window)
    if mechanism == "CLUSTER":
        return _cluster_realization(cond, mech_p, seed, window)
    raise NaturalFamilyError("未知机制 %r" % (mechanism,))


def _well_grid(params: dict) -> list:
    out = []
    k = 0
    for x in params["entry_x"]:
        for y in params["entry_y"]:
            for d in params["directions"]:
                k += 1
                v = np.asarray(d, dtype=np.float64)
                v = v / float(np.linalg.norm(v))
                out.append({"action_id": "G%02d" % k, "well_id": k,
                            "action_class": "SHORT_SEGMENT_GRID",
                            "entry": [float(x), float(y), float(params["entry_z"])],
                            "direction": [float(t) for t in v],
                            "length": float(params["well_length"]),
                            "cost": float(params["well_cost"]),
                            "cost_unit": "domain_length",
                            "target_candidate": None,
                            "executable_condition": "entry 在声明域内；length 为深度格宽整数倍",
                            "failure_model": "no device failure（同 S0/S1 声明）",
                            "note": "盲网格入口：不瞄准任何候选盘；曝光由核几何决定"})
    return out


def generate_world(*, world_id: str, mechanism: str, condition: str, k: int,
                   root_seed: int, mech_index: int, cond_index: int,
                   attach_code: str, attach_scale: float, attach_index: int,
                   params: dict) -> dict:
    """生成一个自然族世界（几何 + 井 + 真相 + 标签；不含任何读数）。"""
    if mechanism not in MECHANISMS:
        raise NaturalFamilyError("未知机制 %r" % (mechanism,))
    cond = dict(params["conditions"][condition])
    mech_p = dict(params["mechanisms"][mechanism])
    window = dict(params["generation_window"])
    n_cand = int(params["n_candidates"])
    stream_seed = world_stream_seed(root_seed, mech_index, cond_index,
                                    attach_index, k)
    rng = np.random.default_rng(stream_seed)
    max_attempts = int(params.get("max_attempts", 8))

    attempts = []
    cands = None
    real = None
    real_note = None
    for attempt in range(max_attempts):
        attempt_seed = int(rng.integers(0, 2 ** 63 - 1))
        real, real_note = _generate_realization(mechanism, cond, mech_p,
                                                attempt_seed, window)
        rows, elig = select_candidates_of(real, cond, params)
        if len(elig) >= n_cand:
            attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                             "n_realization": int(len(rows)),
                             "n_eligible": int(len(elig)),
                             "decision": "ACCEPT", "reason": "ACCEPT"})
            cands = elig[:n_cand]
            break
        attempts.append({"attempt": attempt, "attempt_seed": attempt_seed,
                         "n_realization": int(len(rows)),
                         "n_eligible": int(len(elig)),
                         "decision": "REJECT",
                         "reason": ("REJECT_FEWER_THAN_TWO_CANDIDATES"
                                    if len(elig) < 2
                                    else "REJECT_INSUFFICIENT_ELIGIBLE")})
    if cands is None:
        rows, elig = select_candidates_of(real, cond, params)
        cands = elig[:min(n_cand, len(elig))]

    n_bits = int(len(cands))
    discs = []
    for s, c in enumerate(cands):
        nrm = np.asarray(real.normals[int(c["source_index"])], dtype=np.float64)
        discs.append({"candidate_index": int(s),
                      "center": [float(x) for x in c["center"]],
                      "normal": [float(x) for x in nrm],
                      "radius": float(c["radius"]),
                      "source_index": int(c["source_index"])})

    qp = dict(params["query"])
    query = {"query_id": "Q_%s" % world_id,
             "A": [float(x) for x in qp["A"]], "B": [float(x) for x in qp["B"]],
             "target_domain": "WORLD_COORDINATE box [0,1]^3",
             "r_attach": float(attach_scale),
             "r_attach_tier": str(attach_code),
             "intersection_convention": ("connectivity_qoi two_point_connected: "
                                         "adjacent discs true-intersect "
                                         "(intersect_disk.disk_pair_intersect, strict inequality)"),
             "attachment_convention": ("endpoint captured by disc iff point-to-disc "
                                       "distance <= r_attach (inclusive)"),
             "endpoint_form": "WORLD_COORDINATE_FIXED",
             "endpoint_rule": "A/B = 固定世界坐标（不随世界/状态变化）"}

    wells = _well_grid(params)
    truth_state = int(rng.integers(0, 1 << max(n_bits, 1))) if n_bits >= 1 else 0

    if query is not None and n_bits >= 1:
        der = derive_labels(discs, query, n_bits)
        labels = der["labels"]
        facts = der["facts"]
    else:
        labels, facts = {}, {"contact_edges": (), "attach_a_disks": (),
                             "attach_b_disks": ()}

    model = {"domain": {"lo": [0.0, 0.0, 0.0], "hi": [1.0, 1.0, 1.0]},
             "discs": discs, "action_pool": wells,
             "action_sequence": [wd["action_id"] for wd in wells],
             "query": query,
             "state_labels": {str(s): int(v) for s, v in sorted(labels.items())},
             "state_bits_rule": "bit i = 候选盘 i（LSB=第一候选）；state = Σ bit_i·2^i"}
    geo_obj = {"domain": model["domain"], "discs": model["discs"],
               "action_pool": model["action_pool"], "query": model["query"]}
    geometry_sha12 = sha12_obj(geo_obj)
    model_sha12 = sha12_obj(model)

    label_counts = {"n_label0": int(sum(1 for v in labels.values() if int(v) == 0)),
                    "n_label1": int(sum(1 for v in labels.values() if int(v) == 1))}
    all_same = bool(len(set(labels.values())) <= 1) if labels else True
    trivial_kinds = []
    if all_same:
        trivial_kinds.append("ALL_LABELS_SAME")
    if n_bits == 0:
        trivial_kinds.append("NO_DISCS")
    if len(wells) == 0:
        trivial_kinds.append("NO_WELLS")
    if n_bits < n_cand:
        trivial_kinds.append("FEWER_THAN_TWELVE_CANDIDATES")
    flags = {"all_labels_same": all_same,
             "n_candidates": int(n_bits),
             "fewer_than_twelve_candidates": bool(n_bits < n_cand),
             "no_wells": bool(len(wells) == 0),
             "no_discs": bool(n_bits == 0),
             "n_attach_a": int(len(facts["attach_a_disks"])),
             "n_attach_b": int(len(facts["attach_b_disks"])),
             "n_contact_edges": int(len(facts["contact_edges"])),
             "runnable": bool(n_bits >= 2 and len(wells) >= 2),
             "trivial": bool(trivial_kinds),
             "trivial_kinds": trivial_kinds}
    if not flags["runnable"]:
        flags["not_runnable_reason"] = "FEWER_THAN_TWO_CANDIDATES"

    return {
        "schema": WORLD_SCHEMA,
        "world_id": str(world_id),
        "mechanism": str(mechanism), "condition": str(condition), "tier": TIER,
        "attach_code": str(attach_code), "attach_scale": float(attach_scale),
        "world_code": int(k),
        "family": "NATURAL_D12",
        "phase_order": list(PHASE_ORDER),
        "generation": {
            "stream_seed": int(stream_seed),
            "stream_seed_rule": ("SeedSequence([root_seed, mech_index, cond_index, "
                                 "attach_index, k]).generate_state(1)"),
            "root_seed": int(root_seed),
            "mech_index": int(mech_index), "cond_index": int(cond_index),
            "attach_index": int(attach_index), "k": int(k),
            "conditions_params": cond,
            "mechanism_params": mech_p,
            "mechanism_realization": real_note,
            "n_realization": int(len(real.centers)),
            "attempts": attempts,
            "rng_consumption": "stream -> [attempt_seed_i ...] -> truth_state；"
                               "标签在 truth 之后派生（phase_order）"},
        "selection": {
            "rule": ("包围球域内性（center±r 在 [0,1]^3 内 + containment_margin）∧ "
                     "r ≤ 条件 r_max ⇒ 合格；按 source_index 升序取前 n_candidates"
                     "（不按到 A–B 距离排序）"),
            "crop_declaration": ("目标 = 本 12 盘裁剪模型；生成窗内未入选的盘不进入候选"
                                 "字典，也不冒充原域目标（v2.0 §10.2 裁剪条款）"),
            "n_candidates_declared": n_cand,
            "eligible_rows": [{"source_index": r["source_index"],
                               "eligible": bool(r["eligible"])} for r in
                              select_candidates_of(real, cond, params)[0]],
            "chosen_source_index": [int(c["source_index"]) for c in cands],
            "labels_available_at_selection": False,
        },
        "model": model,
        "truth_state": int(truth_state),
        "n_bits": int(n_bits), "n_states": int(1 << n_bits) if n_bits >= 1 else 0,
        "label_counts": label_counts,
        "flags": flags,
        "facts": {"contact_edges": [[int(i), int(j)] for (i, j) in facts["contact_edges"]],
                  "attach_a_disks": [int(x) for x in facts["attach_a_disks"]],
                  "attach_b_disks": [int(x) for x in facts["attach_b_disks"]]},
        "geometry_sha12": geometry_sha12,
        "model_sha12": model_sha12,
        "finite_scope_pin": {"status": FINITE_SCOPE_PIN, "code_sha12": FINITE_SCOPE_SHA12},
    }


def replay_world(**kw) -> dict:
    return generate_world(**kw)


# ---------------------------------------------------------------------------
# 稀疏精确 PMF（与 kernel.count_pmf 同分布；只为 d=12 的可算性）
# ---------------------------------------------------------------------------

def sparse_pmf(kernel: GridDetectionKernel, action: dict, mask: int, n_bits: int,
               tier: str = TIER, cap: int = 400000) -> dict:
    """给定 (action, 在位掩码) 的计数向量 PMF（稀疏精确有理数）。

    语义与 ``GridDetectionKernel.count_pmf`` 完全相同（同一逐盘结局表、同一计数向量
    生成函数乘积、同一 λ=0），只是把稠密 cell 向量换成稀疏 ``((cell,count),...)``
    键，避免 d=12 时 200 格稠密元组的物化开销。等价性由
    ``audit_sparse_pmf_matches_kernel`` 在盘数 ≤4 的掩码上逐值对账。
    """
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
                        cc, n = vec[pos]
                        lst = list(vec)
                        lst[pos] = (cc, n + 1)
                        key = tuple(sorted(lst))
                ndp[key] = ndp.get(key, Fraction(0, 1)) + p0 * p
        dp = ndp
        if len(dp) > int(cap):
            raise ComputationUnfinished(
                "稀疏精确 PMF 超出声明预算 cap=%d (action=%s, mask=%r)"
                % (cap, action["action_id"], mask))
    total = sum(dp.values())
    if total != 1:
        raise NaturalFamilyError("稀疏 PMF 归一化失败: sum=%s != 1" % (total,))
    return dp


def audit_sparse_pmf_matches_kernel(kernel, action: dict, n_bits: int, max_popcount: int = 4,
                                    tier: str = TIER) -> dict:
    """稀疏 PMF vs 核 count_pmf 的逐值对账（只取 popcount ≤ max_popcount 的掩码）。"""
    rows = []
    for mask in range(1 << int(n_bits)):
        if bin(mask).count("1") > int(max_popcount):
            continue
        mine = sparse_pmf(kernel, action, mask, n_bits, tier=tier)
        ref = kernel.count_pmf(bits_of(mask, n_bits), action, tier=tier)
        ref_sparse = {}
        for vec, p in ref.items():
            counts = tuple((int(i), int(c)) for i, c in enumerate(vec) if c)
            ref_sparse[counts] = Fraction(p)
        ok = (mine == ref_sparse)
        rows.append({"mask": int(mask), "popcount": bin(mask).count("1"),
                     "n_support_sparse": len(mine), "n_support_kernel": len(ref_sparse),
                     "equal": bool(ok)})
        if not ok:
            break
    return {"action_id": str(action["action_id"]), "n_masks": len(rows),
            "all_equal": bool(all(r["equal"] for r in rows)), "rows": rows}


# ---------------------------------------------------------------------------
# 静态装配（lazy dists / law tokens；d=12 可算性的关键）
# ---------------------------------------------------------------------------

class _MaskCache:
    """(action, 在位掩码) → 稀疏 PMF / 观测律令牌 / 浮点分布；惰性 + 逐世界缓存。"""

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
    """按 state 取值的惰性视图（返回共享对象；调用方只读）。"""

    def __init__(self, cache: _MaskCache, action: dict, mask: int, kind: str):
        super().__init__()
        self._cache = cache
        self._action = action
        self._mask = int(mask)
        self._kind = str(kind)
        self._n_bits = int(cache.n_bits)

    def __missing__(self, state):
        m = int(state) & self._mask
        if self._kind == "token":
            return self._cache.token(self._action, m)
        return self._cache.dist(self._action, m)

    def __getitem__(self, state):
        return self.__missing__(int(state))


def _build_table_family(spec: dict, n_bits: int) -> fs.FiniteTable:
    facts = geometry_facts(spec["model"]["discs"], spec["model"]["query"])
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
        raise NaturalFamilyError("%s: 派生标签与登记标签不符：%s"
                                 % (spec["world_id"], mism[:8]))
    struct = cs.TableStructure(kind="DISK_BITS", n_bits=n_bits, certain=(),
                               uncertain=tuple(str(d["candidate_index"])
                                               for d in spec["model"]["discs"]),
                               contact_edges=contact, attach_a_disks=aatt,
                               attach_b_disks=batt)
    table = fs.FiniteTable(
        table_id=str(spec["world_id"]), n_bits=n_bits,
        labels={"S4E": np.asarray([labels[s] for s in range(1 << n_bits)],
                                  dtype=np.int64)},
        baseline=(), actions=(),
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
              "noise": str(spec["tier"]),
              "model": "S4E 自然族（%s/%s, d=%d）" % (spec["mechanism"],
                                                    spec["condition"], n_bits),
              "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "family": "NATURAL_D12",
              "s2_structure": struct.to_dict(),
              "s2_queries": {"kind": "S4E_DOMAIN"}})
    table.derivation = deriv
    return table


def build_world_static(spec: dict, obs_spec: dict, kernel_id_expected: str,
                       tier: str | None = None, pmf_cap: int = 400000) -> cle.StaticS1X:
    """装配一个世界的静态面（返回 ``cle.StaticS1X``，供 cle 闭环原样使用）。"""
    tier = str(tier or spec["tier"])
    model = spec["model"]
    kernel = GridDetectionKernel(model, obs_spec)
    if kernel.kernel_id != str(kernel_id_expected):
        raise NaturalFamilyError("STALE_KERNEL：%s != %s"
                                 % (kernel.kernel_id, kernel_id_expected))
    n_bits = int(spec["n_bits"])
    if n_bits != len(model["discs"]):
        raise NaturalFamilyError("%s: n_bits 与 discs 数不符" % spec["world_id"])
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

    provenance = {
        "research_line": "S4E_natural_family_d12",
        "family": "NATURAL_D12",
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
        "n_exposed_pairs": int(sum(1 for r in exposure_rows if r["exposed"])),
        "T_by_signature": len(sig_groups),
        "signature_groups": [{"signature_sha12": k, "action_ids": v}
                             for k, v in sorted(sig_groups.items())],
        "sequence": [str(x) for x in model.get("action_sequence", [])],
        "static_assembly": ("lazy mask-cached dists/law tokens + sparse exact PMF；"
                            "规划数学与证书协议逐字复用 closed_loop_expanded"),
        "finite_scope_pin": {"status": FINITE_SCOPE_PIN, "code_sha12": FINITE_SCOPE_SHA12},
    }
    cost_int = {str(a["action_id"]): int(round(float(a["cost"]) * 100)) for a in pool}
    cost_dl = {str(a["action_id"]): float(a["cost"]) for a in pool}
    dup = {"action_ids": [], "note": "S4-E 盲网格井；无重复对",
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
    static.mask_cache = cache  # 供成本读数（不参与规划语义）
    return static


def make_expanded_world(static: cle.StaticS1X, spec: dict, *, noise_seed: int,
                        budget_int: int, alpha_seq, max_steps: int):
    return cle.ExpandedWorld(static, str(spec["world_id"]),
                             truth_state=int(spec["truth_state"]),
                             noise_seed=int(noise_seed), budget_int=int(budget_int),
                             alpha_seq=Fraction(alpha_seq), max_steps=int(max_steps))


class WorldFactory:
    """按 world_id 惰性装配静态面（LRU=1）。"""

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
