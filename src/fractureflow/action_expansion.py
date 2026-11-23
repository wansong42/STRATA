# -*- coding: utf-8 -*-
"""S1-X · 行动类型扩展库件（偏心井 / 多 t 档段长 / 读数不扩展）。

本件只服务 `results/architect_main_20261012_dispatch/task_order_S1_X_action_expansion_20261018.md`：
把 S1 的「全部井穿过共同圆心的 2 口井」扩成含偏心井与多 t 档段长的行动池，并在新池上做
精确轨迹覆盖与结构可辨识判定。**不改** `observation_kernel.py`（只读复用，code sha12
`9b14eeea67f7`）与 `sequential_evidence.py`（只读复用 e-process 身份）。

强制语义（与任务单 §2 逐条对应）:
 1. 行动 = 有限井段 (entry, direction, length)；段长必须是深度格宽 `0.01` 的整数倍；
    entry 在声明域内。偏心井只改几何，**不改读数类型**。
 2. 读数 = `(深度格, 无向产状格)` 匿名计数向量，与 S0/S1 完全同一码路；本单**不引入**
    视倾角/方位等方向相关读数（任务单 §2.1-3 要求显式声明；声明为 NOT_INTRODUCED）。
    因此本单的信息增益只能来自「哪些盘被曝光、曝光在哪个深度格」，不能来自新读数。
 3. 身份：任何行动/读数/量化变化 => 新 `kernel_id` + 新 prereg。本件把行动池写进观测
    规格的 `s1x_extension` 段，用 `observation_kernel.kernel_identity` 同一公式重算身份
    （spec 内容变 => spec sha12 变 => kernel_id 变；代码字节不变，故 code sha12 仍为
    `9b14eeea67f7`）。旧 `K-30041ee8cade-9b14eeea67f7` 只作对照，禁混用。
 4. 独立行动数按**观测律签名**（逐盘精确结局表）计数，不按井数；重复对逐条列出。
 5. 结构判定：θ 与孪生 θ'（d1/d2 互换）在**每个**行动 × **每个**噪声档上的全计数向量
    PMF 精确相等 => 全序列同分布 => 结构性不可认证；只要有一条行动打破相等，即给见证
    记录与精确概率差。禁止用抽样代替全 PMF。
 6. 精确轨迹：路径空间 = Π_t support_{full}(a_t)（全盘在位状态的 count_pmf 支撑，是任何
    子状态支撑的超集）；叶数超 `LEAF_BUDGET` 即 `COMPUTATION_UNFINISHED`，不猜不删不截断。
 7. 数值：一切概率 `fractions.Fraction`；浮点只出现在耗时与审计字段。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from fractions import Fraction

import numpy as np

from .observation_kernel import (
    EMPTY_OUTCOME,
    GridDetectionKernel,
    canonical_sha12,
    kernel_identity,
)
from .sequential_evidence import (
    STATUS_CERTIFIED,
    STATUS_CONFLICT,
    STATUS_DUAL,
    STATUS_EMPTY,
    STATUS_SEARCH_UNRESOLVED,
    EProcess,
    PredictiveMixture,
    bits_of,
    record_from_counts,
    record_key,
)

__all__ = [
    "ROOT", "S0_SPEC_REL", "S1_MODEL_REL", "OUT_REL", "LEAF_BUDGET",
    "PLANE_NORMAL", "PLANE_E1", "PLANE_E2", "COMMON_CENTER",
    "POOL_DECLARATION", "SEQUENCE", "TARGET_STATES", "TWIN_PAIR",
    "load_json", "sha12_bytes", "sha12_file", "sha12_obj", "frac_str", "jsonable", "js",
    "build_pool", "build_extension", "build_spec", "build_model",
    "action_signature", "signature_sha12", "signature_groups",
    "state_pmf", "pmf_compare", "pmf_tv", "pmf_equal",
    "sweep_action", "enumerate_sequence", "exposure_table",
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S0_SPEC_REL = "results/s0s1_minimal_vertical_chain/s0/observation_spec.json"
S1_MODEL_REL = "results/s1_sequential_validity/tiny_model/model.json"
OUT_REL = "results/s1x_action_expansion"
LEAF_BUDGET = 200000

# 与 S1 tiny_model 同一几何（只读继承）：共同圆心 C 与 d1/d2 的盘面。
COMMON_CENTER = (0.5, 0.5, 0.5)
PLANE_NORMAL = (0.0, 0.3420201433256687, 0.9396926207859084)
PLANE_E1 = (1.0, 0.0, 0.0)
PLANE_E2 = (0.0, 0.9396926207859084, -0.3420201433256687)
# 继承模型 d0 的盘心（脚本会与 model.json 逐位对账，不一致即 SystemExit）。
D0_CENTER = (0.5494974746830583, 0.5494974746830583, 0.5)

TARGET_STATES = (2, 4, 10, 12)
TWIN_PAIR = ((2, 4), (10, 12))

# 行动池声明（先冻结后跑）。参数含义:
#   kind = CORE（线穿共同圆心）/ ECCENTRIC_SHELL（线与 d1/d2 盘面的交点偏离圆心）
#          / TANGENT_D0（线躺在 d1/d2 盘面内、过 d0 盘心：d1/d2 恒为 PARALLEL 未曝光）
#   rho  = 盘面内偏移（domain_length），方向由 in_plane 指定
#   t_cross = 交点沿井方向的深度（取深度格中点，避免格边界）
#   length  = 声明段长（格宽整数倍）
POOL_DECLARATION = (
    {"action_id": "XS1", "well_id": 10, "kind": "ECCENTRIC_SHELL",
     "in_plane": "e1", "rho": -0.08, "t_cross": 0.025, "length": 0.04,
     "note": "偏心壳井：与 d1/d2 盘面交点距 C 0.08 ∈ (r_d1=0.05, r_d2=0.12] "
             "=> d2 曝光、d1 未曝光；本侧 d0/d3 均不在井段内（纯壳井）"},
    {"action_id": "XS1M", "well_id": 11, "kind": "ECCENTRIC_SHELL",
     "in_plane": "e2", "rho": -0.08, "t_cross": 0.025, "length": 0.04,
     "note": "与 XS1 同 t 档、同段长、同样只曝光 d2 的另一口井：用于检验"
             "「按井数冒充独立行动数」（预期为重复观测）"},
    {"action_id": "XS2", "well_id": 12, "kind": "ECCENTRIC_SHELL",
     "in_plane": "e2", "rho": -0.08, "t_cross": 0.005, "length": 0.02,
     "note": "偏心壳井的第二 t 档 + 0.02 段长档：交点落在 bin0 中点，"
             "与 XS1 的 bin2 观测律不同"},
    {"action_id": "XS3", "well_id": 13, "kind": "CORE",
     "in_plane": None, "rho": 0.0, "t_cross": 0.025, "length": 0.04,
     "note": "穿共同圆心的核心井：四盘在 bin2 中点全部曝光（对照 S1 的 T=2 上限）"},
    {"action_id": "XS4", "well_id": 14, "kind": "CORE",
     "in_plane": None, "rho": 0.0, "t_cross": 0.055, "length": 0.06,
     "note": "穿共同圆心的长段井（0.06 = 6 格）：四盘在 bin5 中点曝光，段长档 0.06"},
    {"action_id": "XS5", "well_id": 15, "kind": "ECCENTRIC_SHELL",
     "in_plane": "e1", "rho": 0.03, "t_cross": 0.035, "length": 0.04,
     "note": "面内小偏移井（rho=0.03 < r_d1）：d1/d2 同时曝光且深度不同；"
             "预期**不能**打破孪生对（半径检验在两者内同时成立），作为壳井的反面对照"},
    {"action_id": "XS6", "well_id": 16, "kind": "TANGENT_D0",
     "in_plane": None, "rho": 0.0, "t_cross": 0.025, "length": 0.04,
     "note": "盘面内切向井：方向 e1 躺在 d1/d2 盘面内 ⇒ d1/d2 恒 PARALLEL 未曝光；"
             "过 d0 盘心 ⇒ 只曝光 d0（bin2 中点）。用于把 d0 与 d1 的格重叠拆开"},
)

# 声明序列：四个互异观测律（壳井 bin2 -> 核心井四盘 bin2 -> 壳井 bin0 -> d0 切向井）。
SEQUENCE = ("XS1", "XS3", "XS2", "XS6")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def sha12_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha12_bytes(f.read())


def sha12_obj(obj) -> str:
    return canonical_sha12(obj)


def frac_str(x):
    if isinstance(x, Fraction):
        return "%d/%d" % (x.numerator, x.denominator)
    return x


def js(obj, **kw):
    """json.dumps 的安全封装：allow_nan 字面量只在本调用点出现一次。

    本线脚本一律走本封装（属面 S31 守卫按 AST 查 json.dumps ∧ allow_nan 字面量，
    看不穿 `**kw` 置位；把字面量集中在实现件里可让写面审计保持 0 缺守卫）。
    """
    kw.setdefault("ensure_ascii", False)
    return json.dumps(obj, allow_nan=False, **kw)


def jsonable(obj):
    if isinstance(obj, Fraction):
        return frac_str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def _vec_add(a, b):
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def _vec_scale(v, s):
    return [float(v[0]) * float(s), float(v[1]) * float(s), float(v[2]) * float(s)]


def build_pool() -> list:
    """由 POOL_DECLARATION 计算行动规格（纯几何，不看任何标签/真值）。"""
    axes = {"e1": PLANE_E1, "e2": PLANE_E2}
    pool = []
    for row in POOL_DECLARATION:
        t = float(row["t_cross"])
        direction = list(PLANE_NORMAL)
        if row["kind"] == "CORE":
            plane_point = list(COMMON_CENTER)
        elif row["kind"] == "ECCENTRIC_SHELL":
            axis = axes[row["in_plane"]]
            plane_point = _vec_add(COMMON_CENTER, _vec_scale(axis, row["rho"]))
        elif row["kind"] == "TANGENT_D0":
            # 方向改为盘面内 e1；线过 d0 盘心，故曝光点 = d0 盘心，t = t_cross。
            direction = list(PLANE_E1)
            plane_point = list(D0_CENTER)
        else:
            raise SystemExit("未知行动类型: %r" % (row["kind"],))
        entry = [plane_point[k] - t * direction[k] for k in range(3)]
        pool.append({
            "action_id": row["action_id"],
            "well_id": int(row["well_id"]),
            "action_class": row["kind"],
            "entry": [round(x, 15) for x in entry],
            "direction": [round(x, 15) for x in direction],
            "length": float(row["length"]),
            "cost": float(row["length"]),
            "cost_unit": "domain_length",
            "plane_point": [round(x, 15) for x in plane_point],
            "in_plane_offset_rho": float(row["rho"]),
            "t_cross_declared": t,
            "depth_bin_of_cross": int(round(t / 0.01 - 0.5)),
            "executable_condition": "entry 在声明域 [0,1]^3 内；length 为深度格宽整数倍",
            "failure_model": "no device failure（与 S0/S1 同：失败面未开）",
            "note": row["note"],
        })
    return pool


def build_extension(pool: list) -> dict:
    """写进观测规格的 s1x_extension 段（身份的一部分）。"""
    return {
        "research_line": "S1X_action_expansion",
        "authority": "results/architect_main_20261012_dispatch/"
                     "task_order_S1_X_action_expansion_20261018.md",
        "inherits_geometry_from": S1_MODEL_REL,
        "readout": {
            "kind": "DEPTH_CELL_x_UNDIRECTED_ORIENTATION_CELL_COUNT_VECTOR",
            "direction_dependent_readings": "NOT_INTRODUCED",
            "why": "任务单 §2.1-3 要求显式声明；本单不引入视倾角/方位读数，"
                   "信息增益只能来自曝光几何（哪些盘被曝光、落在哪个深度格）。",
            "quantization_order": "true intersection -> declared measurement error -> "
                                  "quantize -> release (与 S0 同一码路)",
        },
        "action_pool": pool,
        "declared_sequence": list(SEQUENCE),
        "new_degrees_of_freedom": {
            "eccentric_wells": True,
            "multi_t_band_lengths": [0.03, 0.04, 0.06],
            "direction_dependent_readings": False,
        },
        "duplicate_policy": "独立行动数按逐盘精确结局表签名计数；签名相同即重复观测，"
                            "不得计入 T，也不得在同一序列中当第二份独立证据。",
    }


def build_spec(s0_spec: dict, pool: list) -> dict:
    """S0 观测规格 + s1x_extension => 新规格（重算 kernel_identity）。"""
    base = {k: v for k, v in s0_spec.items() if k != "kernel_identity"}
    base["spec_id"] = "S1X_OBS_SPEC_V1"
    base["s1x_extension"] = build_extension(pool)
    ki = kernel_identity(base)
    out = dict(base)
    out["kernel_identity"] = ki
    return out


def build_model(s1_model: dict, pool: list, kernel_id: str) -> dict:
    model = {
        "schema": "s1x.model.v1",
        "model_id": "S1X_MODEL_V1",
        "model_identity": "KNOWN_SYNTHETIC",
        "watermark": {
            "research_line": "S1X_action_expansion",
            "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
            "dictionary": "COMPLETE_KNOWN",
            "claim_scope": "DECLARED_SMALL_MODEL",
            "real_domain_claims": False,
            "external_commitment_text": False,
        },
        "inherits_geometry_from": {
            "path": S1_MODEL_REL,
            "sha12": sha12_file(os.path.join(ROOT, S1_MODEL_REL)),
            "note": "四盘几何、查询、目标规则逐字节继承 S1；行动池整体替换，"
                    "旧 K-30041ee8cade-9b14eeea67f7 读数只作对照",
        },
        "domain": dict(s1_model["domain"]),
        "discs": [dict(d) for d in s1_model["discs"]],
        "action_pool": pool,
        "action_sequence": [a for a in SEQUENCE],
        "query": dict(s1_model["query"]),
        "target_rule_declared": s1_model["target_rule_declared"],
        "state_index": s1_model["state_index"],
        "state_labels": {str(k): int(v) for k, v in s1_model["state_labels"].items()},
        "state_bits": {str(k): list(v) for k, v in s1_model["state_bits"].items()},
        "counterexample_pair": dict(s1_model["counterexample_pair"]),
        "positive_set": dict(s1_model["positive_set"]),
        "noise_tier_default": "N1",
        "tier_scope": ["N0", "N1", "N2"],
        "kernel_id": kernel_id,
    }
    return model


# ---------------------------------------------------------------- 签名与 PMF
def _fs_table(table: dict) -> list:
    return sorted([[str(k), frac_str(v)] for k, v in table.items()],
                  key=lambda row: row[0])


def action_signature(kernel: GridDetectionKernel, action: dict, tier: str = "N1") -> dict:
    """观测律签名：逐盘精确结局表（含 EMPTY 分支）。"""
    per_disc = []
    for i in range(len(kernel.candidate_geometry())):
        per_disc.append(_fs_table(kernel.disc_outcome_table(action, i, tier)))
    return {
        "action_id": action["action_id"],
        "well_id": int(action["well_id"]),
        "tier": tier,
        "n_depth_bins": int(round(action["length"] / kernel.depth_bin_width)),
        "per_disc": per_disc,
    }


def signature_sha12(sig: dict) -> str:
    return canonical_sha12({k: v for k, v in sig.items()
                            if k not in ("action_id", "well_id")})


def signature_groups(sigs: list) -> list:
    groups = {}
    for i, sig in enumerate(sigs):
        groups.setdefault(signature_sha12(sig), []).append(i)
    out = []
    for sha, idx in sorted(groups.items(), key=lambda kv: kv[1][0]):
        out.append({"signature_sha12": sha, "members": idx,
                    "n_members": len(idx),
                    "action_ids": [sigs[i]["action_id"] for i in idx]})
    return out


def state_pmf(kernel: GridDetectionKernel, action: dict, state: int,
              tier: str, d: int = 4) -> dict:
    return dict(kernel.count_pmf(bits_of(state, d), action, tier=tier))


def pmf_equal(p: dict, q: dict) -> bool:
    return p == q


def pmf_tv(p: dict, q: dict) -> Fraction:
    keys = set(p) | set(q)
    s = Fraction(0, 1)
    for k in keys:
        s += abs(p.get(k, Fraction(0, 1)) - q.get(k, Fraction(0, 1)))
    return s / 2


def _vec_to_counts(vec) -> list:
    return [[int(i), int(c)] for i, c in enumerate(vec) if int(c)]


def pmf_compare(kernel: GridDetectionKernel, action: dict, state_a: int, state_b: int,
                tier: str, n_cells: int, d: int = 4) -> dict:
    """同一行动下两状态的全计数向量 PMF 逐项比较（精确）。"""
    p = state_pmf(kernel, action, state_a, tier, d)
    q = state_pmf(kernel, action, state_b, tier, d)
    for pm in (p, q):
        if sum(pm.values(), Fraction(0, 1)) != 1:
            raise SystemExit("count_pmf 归一化失败: %s/%s" % (action["action_id"], tier))
    keys = set(p) | set(q)
    equal = (p == q)
    tv = pmf_tv(p, q)
    best = None
    best_diff = Fraction(-1, 1)
    for k in keys:
        diff = abs(p.get(k, Fraction(0, 1)) - q.get(k, Fraction(0, 1)))
        if diff > best_diff:
            best_diff = diff
            best = k
    witness = None
    if best is not None:
        witness = {
            "counts": _vec_to_counts(best),
            "p_state_a": frac_str(p.get(best, Fraction(0, 1))),
            "p_state_b": frac_str(q.get(best, Fraction(0, 1))),
            "abs_diff": frac_str(best_diff),
            "empty_record": not _vec_to_counts(best),
        }
    return {
        "action_id": action["action_id"],
        "well_id": int(action["well_id"]),
        "tier": tier,
        "state_a": int(state_a),
        "state_b": int(state_b),
        "equal": bool(equal),
        "tv": frac_str(tv),
        "tv_float": float(tv),
        "n_support_a": len(p),
        "n_support_b": len(q),
        "n_support_union": len(keys),
        "n_cells": int(n_cells),
        "witness": witness,
    }


# ---------------------------------------------------------------- 精确枚举
def sweep_action(kernel: GridDetectionKernel, action: dict, tier: str,
                 states=tuple(range(16)), d: int = 4) -> dict:
    """单行动扫面：全正概率记录 × 逐状态精确概率（记录由 count_pmf 直接给出）。"""
    parsed = kernel._parse_action(action)
    n_cells = parsed["n_depth_bins"] * kernel.n_orientation_cells
    pmfs = {}
    records = {}
    for s in states:
        pmfs[s] = kernel.count_pmf(bits_of(s, d), action, tier=tier)
        for vec, prob in pmfs[s].items():
            counts = {i: int(c) for i, c in enumerate(vec) if int(c)}
            rk = tuple((int(i), int(c)) for i, c in counts.items())
            if rk not in records:
                records[rk] = record_from_counts(kernel, action, counts)
    # 路径空间取 16 个状态支撑的并集：N0 档检测概率 1、无 EMPTY 分支，
    # 子状态的支撑可以**不在**全盘状态支撑里（例如 θ=0 的全 EMPTY 记录）。
    support = []
    seen = set()
    for s in states:
        for vec, prob in pmfs[s].items():
            if prob == 0:
                continue
            counts = {i: int(c) for i, c in enumerate(vec) if int(c)}
            rk = tuple((int(i), int(c)) for i, c in counts.items())
            if rk not in seen:
                seen.add(rk)
                support.append((rk, prob))
    support.sort(key=lambda kv: (len(kv[0]), kv[0]))
    pmat = {rk: [Fraction(0, 1)] * len(states) for rk, _ in support}
    for j, s in enumerate(states):
        for vec, prob in pmfs[s].items():
            counts = {i: int(c) for i, c in enumerate(vec) if int(c)}
            rk = tuple((int(i), int(c)) for i, c in counts.items())
            if rk in pmat:
                pmat[rk][j] = prob
    for rk, _ in support:
        tot = sum(pmat[rk], Fraction(0, 1))
        if tot == 0:
            raise SystemExit("扫面出现全状态零概率记录: %r" % (rk,))
    for j, s in enumerate(states):
        tot = sum((pmat[rk][j] for rk, _ in support), Fraction(0, 1))
        if tot != 1:
            raise SystemExit("扫面支撑对状态 %d 未闭合: sum=%s action=%s"
                             % (s, tot, action["action_id"]))
    return {"action_id": action["action_id"], "tier": tier, "n_cells": n_cells,
            "support": support, "pmat": pmat, "records": records,
            "state_support": {s: sum(1 for v in pmfs[s].values() if v > 0)
                              for s in states},
            "pmf_norm_ok": all(sum(pmfs[s].values(), Fraction(0, 1)) == 1
                               for s in states)}


def enumerate_sequence(kernel: GridDetectionKernel, actions: list, tier: str,
                       alpha_seq: Fraction, labels: dict, states=tuple(range(16)),
                       d: int = 4, leaf_budget: int = LEAF_BUDGET,
                       collect_first_exclusion: bool = True,
                       drop_final_statuses=None) -> dict:
    """全轨迹精确枚举（无抽样）：逐 θ* 的 P(∃t: θ*∉S_t) 与终步单值率。"""
    t0 = time.time()
    sweeps = [sweep_action(kernel, a, tier, states, d) for a in actions]
    sizes = [len(s["support"]) for s in sweeps]
    n_leaves = 1
    for n in sizes:
        n_leaves *= n
    if n_leaves > leaf_budget:
        return {"status": "COMPUTATION_UNFINISHED", "n_leaves": n_leaves,
                "leaf_budget": leaf_budget, "support_sizes": sizes}
    # 预填 PredictiveMixture 缓存：只读复用 e-process 身份，但 p 面由 count_pmf 精确给出。
    cache = {}
    for sw, action in zip(sweeps, actions):
        for rk, _prob in sw["support"]:
            rec_key = record_key(sw["records"][rk])
            for j, s in enumerate(states):
                cache[(str(action["action_id"]), int(s), rec_key)] = sw["pmat"][rk][j]
    alpha = Fraction(alpha_seq)
    res = {s: {"excluded_mass": Fraction(0, 1), "survive_mass": Fraction(0, 1),
               "certified_correct_mass": Fraction(0, 1),
               "certified_wrong_mass": Fraction(0, 1),
               "dual_mass": Fraction(0, 1), "empty_mass": Fraction(0, 1),
               "conflict_mass": Fraction(0, 1), "unresolved_mass": Fraction(0, 1),
               "first_exclusion_mass": {t: Fraction(0, 1) for t in range(1, len(actions) + 1)},
               "n_paths_nonzero": 0}
           for s in states}
    totals = {"n_paths": 0, "max_n_S": 0, "status_counts": {},
              "n_paths_all_states_excluded": 0}
    index_lists = [list(range(len(sw["support"]))) for sw in sweeps]
    for combo in itertools.product(*index_lists):
        totals["n_paths"] += 1
        mix = PredictiveMixture(states, kernel, actions[0], mode="posterior",
                                tier=tier, cache=cache)
        ep = EProcess(states, labels, alpha, mix, kernel)
        p_path = [Fraction(1, 1)] * len(states)
        excluded = [False] * len(states)
        first_t = [None] * len(states)
        status = None
        for t, (sw, action, idx) in enumerate(zip(sweeps, actions, combo), start=1):
            rk = sw["support"][idx][0]
            ep.update(action, sw["records"][rk])
            rows = sw["pmat"][rk]
            S = set(ep.state_set())
            for j, s in enumerate(states):
                p_path[j] *= rows[j]
                if not excluded[j] and s not in S:
                    excluded[j] = True
                    first_t[j] = t
            status = ep.status()
            totals["max_n_S"] = max(totals["max_n_S"], len(S))
        totals["status_counts"][status] = totals["status_counts"].get(status, 0) + 1
        if drop_final_statuses and status in drop_final_statuses:
            # 控制面专用：模拟"删失败轨迹"。概率账必然不闭合（用于 P07 的牙齿）。
            continue
        if all(excluded):
            totals["n_paths_all_states_excluded"] += 1
        for j, s in enumerate(states):
            if p_path[j] == 0:
                continue
            r = res[s]
            r["n_paths_nonzero"] += 1
            if excluded[j]:
                r["excluded_mass"] += p_path[j]
                if collect_first_exclusion:
                    r["first_exclusion_mass"][first_t[j]] += p_path[j]
            else:
                r["survive_mass"] += p_path[j]
            if status == STATUS_CERTIFIED:
                if ep.target_set() == (labels[s],):
                    r["certified_correct_mass"] += p_path[j]
                else:
                    r["certified_wrong_mass"] += p_path[j]
            elif status == STATUS_DUAL:
                r["dual_mass"] += p_path[j]
            elif status == STATUS_EMPTY:
                r["empty_mass"] += p_path[j]
            elif status == STATUS_CONFLICT:
                r["conflict_mass"] += p_path[j]
            else:
                r["unresolved_mass"] += p_path[j]
    out = {"status": "EXACT", "tier": tier, "alpha_seq": frac_str(alpha),
           "alpha_seq_float": float(alpha), "n_leaves": n_leaves,
           "leaf_budget": leaf_budget, "support_sizes": sizes,
           "support_actions": [a["action_id"] for a in actions],
           "elapsed_s": round(time.time() - t0, 3),
           "totals": totals, "per_state": {}}
    for s in states:
        r = res[s]
        tot = r["excluded_mass"] + r["survive_mass"]
        r["total_prob"] = frac_str(tot)
        r["prob_closed"] = bool(tot == 1)
        r["coverage_exact"] = frac_str(r["excluded_mass"])
        r["coverage_float"] = float(r["excluded_mass"])
        r["coverage_le_alpha"] = bool(r["excluded_mass"] <= alpha)
        r["certified_correct_rate"] = float(r["certified_correct_mass"] / tot) if tot else None
        r["dual_rate"] = float(r["dual_mass"] / tot) if tot else None
        r["empty_rate"] = float(r["empty_mass"] / tot) if tot else None
        r["conflict_rate"] = float(r["conflict_mass"] / tot) if tot else None
        r["unresolved_rate"] = float(r["unresolved_mass"] / tot) if tot else None
        r["first_exclusion_mass"] = {str(t): frac_str(v)
                                     for t, v in r["first_exclusion_mass"].items()}
        out["per_state"][str(s)] = jsonable(r)
    out["totals"] = jsonable(out["totals"])
    return out


def exposure_table(kernel: GridDetectionKernel, actions: list) -> list:
    rows = []
    for a in actions:
        for i in range(len(kernel.candidate_geometry())):
            e = kernel.disc_exposure(a, i)
            rows.append({"action_id": a["action_id"], "disc_index": i,
                         "exposed": bool(e["exposed"]), "reason": e["reason"],
                         "t": e["t"]})
    return rows
