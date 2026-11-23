# -*- coding: utf-8 -*-
"""R110 · B1 统计基线 (经典朴素参数化 DFN 重采样) + 协议自洽性验证.

协议权威: results/v_r110_recon_protocol.json (含 v1.1 预注册期修正: 密度=朴素枚举).
零训练零生成模型——B1 是参数化重采样, 是重构线的参考带 (reference band), 不是模型.

流程 (每场景 × 3 seeds 11/22/33):
  1. 载入遮蔽场景 obs 子集 (r110_hrf_masker 产物) + M_obs
  2. 观测面提取 (阈值: borehole ≥1 点, 面状 ≥30 点; 法向规则: ≥20 点 SVD, 否则 nrm 均值)
  3. N̂ = 观测面数 (朴素枚举, v1.1)
  4. Fisher 组系拟合 (leader 聚类 θ=20°, κ̂=(m−1)/(1−R̄) 截断 [5,2000])
  5. 生成: 均匀位置 + vMF 切向高斯方向 + R̂ bootstrap → 盘列表
  6. 渲染→同口径提取→五层评分卡 vs 真值网络
  7. identity check (真值渲染 vs 真值) + full 场景自洽门 (协议 §b1_selfconsistency_gates)

输出: results/r110_b1/b1_scorecard.json
红线: 源数据只读; 不调参掩盖异常 (反常档诊断节在 r110_report.md)。

用法:
  E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe scripts/r110_baseline_b1.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from r110_recon_metrics import (  # noqa: E402
    extract_faces, render_faces, scorecard, frozen_scanlines,
    leader_set_clusters, vmf_tangent_gaussian, _unit, _sign_align_z)
from r110_hrf_masker import load_scenario, SRC  # noqa: E402

PROTOCOL = os.path.join(ROOT, "results", "v_r110_recon_protocol.json")
OUT_DIR = os.path.join(ROOT, "results", "r110_b1")
OUT_JSON = os.path.join(OUT_DIR, "b1_scorecard.json")
SCENARIOS = ["full", "borehole_1", "borehole_2", "borehole_4",
             "outcrop_10", "outcrop_30", "outcrop_50", "station_30", "multi_30_2"]
SEEDS = [11, 22, 33]
MIN_PTS_AREAL = 30
MIN_PTS_BOREHOLE = 1
SVD_NORMAL_MIN_PTS = 20
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def _san(o):
    """JSON 序列化清洗: numpy 标量转 python, 非有限值转 None。"""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


# --------------------------------------------------------------------------- #
# B1 核心: 观测面拟合
# --------------------------------------------------------------------------- #
def fit_observed_faces(pos_o, nrm_o, fid_o, min_pts):
    """观测面提取 (协议冻结规则)。返回 faces 列表 [{c,n,R,support}]。"""
    faces = []
    for f in np.unique(fid_o):
        m = fid_o == f
        support = int(m.sum())
        if support < min_pts:
            continue
        P = pos_o[m]
        c = P.mean(0)
        X = P - c
        if support >= SVD_NORMAL_MIN_PTS:
            w, v = np.linalg.eigh(X.T @ X)
            n = _unit(v[:, 0])
        else:
            Nv = _sign_align_z(nrm_o[m])
            n = _unit(Nv.mean(0))
        r_in = X - np.outer(X @ n, n)
        R = float(np.sqrt(2.0 * (r_in ** 2).sum(1).mean()))
        faces.append(dict(c=c, n=n, R=R, support=support))
    return faces


def b1_generate(faces_obs, box, seed):
    """朴素参数化补全 (v1.1): N̂=观测面数; 均匀位置 + vMF + R̂ bootstrap。"""
    rng = np.random.default_rng(seed)
    box = np.asarray(box, dtype=np.float64)
    N_hat = len(faces_obs)
    if N_hat == 0:
        return []
    # 组系拟合 (面积加权 leader 聚类)
    labels, means, props, _ = leader_set_clusters(faces_obs)
    kappa = []
    for k in range(len(means)):
        Nn = _sign_align_z(np.array([f["n"] for f in faces_obs])[labels == k])
        Rbar = float(np.linalg.norm(Nn.mean(0)))
        m = int((labels == k).sum())
        kap = (m - 1) / max(1.0 - Rbar, 1e-6) if m > 1 else 100.0
        kappa.append(float(np.clip(kap, 5.0, 2000.0)))
    Rs = np.array([f["R"] for f in faces_obs])
    gen = []
    for _ in range(N_hat):
        k = int(rng.choice(len(means), p=props / props.sum()))
        n = vmf_tangent_gaussian(rng, means[k], kappa[k])
        if rng.random() < 0.5:
            n = -n
        R = float(Rs[rng.integers(0, len(Rs))])
        c = np.array([rng.uniform(box[0][0], box[1][0]),
                      rng.uniform(box[0][1], box[1][1]),
                      rng.uniform(box[0][2], box[1][2])])
        gen.append(dict(c=c, n=n, R=R))
    return gen


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    protocol = json.load(open(PROTOCOL, encoding="utf-8"))
    log(f"protocol loaded (version={protocol['version']}, "
        f"amendments={[a['amendment'] for a in protocol.get('preregister_amendments', [])]})")

    log(f"loading truth {SRC} (read-only)")
    d = np.load(SRC, allow_pickle=True)
    xyz = d["xyz"].astype(np.float64)
    nrm = _unit(d["nrm"].astype(np.float64))
    fid = d["fid"].astype(np.int64)
    lab = fid >= 0
    box = np.array([xyz.min(0), xyz.max(0)])
    scanlines = frozen_scanlines(box)
    faces_true = extract_faces(xyz[lab], fid[lab])
    log(f"truth: {len(faces_true)} faces, box diag "
        f"{np.linalg.norm(box[1] - box[0]):.0f} m")

    results = dict(
        task="R110", baseline="B1_statistical_resampling_v1.1",
        protocol="results/v_r110_recon_protocol.json",
        seeds=SEEDS, scenarios={}, identity_check={}, notes=[],
    )

    # ---- identity check: 真值渲染 vs 真值 (渲染-提取-评测路径零系统偏置) ----
    pos_r, fid_r = render_faces(faces_true, seed=0)
    sc = scorecard(extract_faces(pos_r, fid_r), faces_true, box, scanlines)
    results["identity_check"] = _san(sc)

    def _flat(d):
        for layer in d.values():
            for k, v in layer.items():
                if not k.startswith("_"):
                    yield k, v

    def _id_ok(k, v):
        if v is None or not np.isfinite(v):
            return False
        if k == "set_count_abs_diff":
            return v == 0
        if k == "set_proportion_L1" or k == "largest_cluster_fraction_diff":
            return v < 0.02
        if k == "powerlaw_exponent_diff":
            return abs(v) < 0.15
        if "deg" in k:
            return v < 1.0
        return v < 0.02  # 其余相对型度量

    flat_sc = dict(_flat(sc))
    id_ok = all(_id_ok(k, v) for k, v in flat_sc.items())
    log(f"identity check: {'PASS' if id_ok else 'FAIL'} "
        f"(max rel={max((v for k, v in flat_sc.items() if isinstance(v, (int, float)) and 'deg' not in k and k not in ('set_count_abs_diff', 'set_proportion_L1', 'largest_cluster_fraction_diff', 'powerlaw_exponent_diff')), default=0):.5f})")

    # ---- 各场景 × seeds ----
    for scen in SCENARIOS:
        obs, m_obs = load_scenario(scen)
        obs_l = obs & lab
        pos_o, nrm_o, fid_o = xyz[obs_l], nrm[obs_l], fid[obs_l]
        min_pts = MIN_PTS_BOREHOLE if scen.startswith("borehole") else MIN_PTS_AREAL
        faces_obs = fit_observed_faces(pos_o, nrm_o, fid_o, min_pts)
        N_hat = len(faces_obs)
        per_seed = []
        for seed in SEEDS:
            gen = b1_generate(faces_obs, box, seed)
            pos_g, fid_g = render_faces(gen, seed=seed)
            faces_rec = extract_faces(pos_g, fid_g) if len(gen) else []
            sc = scorecard(faces_rec, faces_true, box, scanlines) if faces_rec else None
            per_seed.append(dict(seed=seed, n_generated=len(gen),
                                 scorecard=_san(sc) if sc else None))
            log(f"{scen} seed={seed}: N_gen={len(gen)}")

        # 聚合 (跳过缺失 scorecard)
        valid = [p["scorecard"] for p in per_seed if p["scorecard"]]
        agg = {}
        if valid:
            for layer in ("set", "scale", "spatial", "topology", "consistency"):
                keys = [k for k in valid[0][layer] if not k.startswith("_")]
                agg[layer] = {}
                for k in keys:
                    vals = [float(v[layer][k]) for v in valid
                            if v[layer][k] is not None]
                    if vals:
                        agg[layer][k] = dict(mean=float(np.mean(vals)),
                                             std=float(np.std(vals)),
                                             n=len(vals))
        results["scenarios"][scen] = dict(
            observation_type=m_obs["observation_type"],
            coverage_fraction=m_obs["coverage_fraction"],
            n_observed_faces=N_hat,
            n_true_faces=len(faces_true),
            min_face_pts=min_pts,
            per_seed=per_seed,
            aggregate=_san(agg),
        )
        log(f"{scen}: obs_faces={N_hat}/{len(faces_true)} "
            f"cov={m_obs['coverage_fraction']:.4f}")

    # ---- full 场景自洽门 (协议 §b1_selfconsistency_gates) ----
    gates = protocol["b1_selfconsistency_gates"]["full_observation_gates"]
    full_agg = results["scenarios"]["full"]["aggregate"]
    checks = []
    gate_map = [
        ("set", "set_count_abs_diff", lambda v: v <= 1),
        ("set", "fisher_mean_direction_distance_deg", lambda v: v <= 8.0),
        ("set", "set_proportion_L1", lambda v: v <= 0.20),
        ("scale", "trace_length_w1_rel", lambda v: v <= 0.25),
        ("scale", "powerlaw_exponent_diff", lambda v: abs(v) <= 0.80),
        ("spatial", "density_field_l1_rel", lambda v: v <= 0.30),
        ("spatial", "ripley_k_max_rel_dev", lambda v: v <= 0.60),
        ("topology", "p10_rel_err", lambda v: v <= 0.30),
        ("topology", "p32_rel_err", lambda v: v <= 0.25),
        ("topology", "intersection_density_rel_err", lambda v: v <= 0.40),
        ("topology", "largest_cluster_fraction_diff", lambda v: v <= 0.25),
        ("consistency", "heldout_p10_rel_err", lambda v: v <= 0.30),
        ("consistency", "heldout_z_w1_rel", lambda v: v <= 0.35),
        ("consistency", "heldout_normal_circ_std_diff_deg", lambda v: v <= 6.0),
    ]
    n_pass = 0
    for layer, key, fn in gate_map:
        v = full_agg.get(layer, {}).get(key, {}).get("mean")
        ok = (v is not None) and fn(v)
        n_pass += bool(ok)
        checks.append(dict(layer=layer, metric=key, value=v,
                           gate=gates.get(key, gates.get(key + "(abs)", "")),
                           passed=bool(ok)))
    verdict = "SELF_CONSISTENT" if n_pass >= 12 else "HONEST_FAIL"
    results["selfconsistency"] = dict(
        identity_check_passed=bool(id_ok),
        gates_passed=f"{n_pass}/14",
        checks=checks,
        verdict=verdict,
    )
    results["notes"].append(
        "B1 = 朴素参数化重采样参考带 (v1.1, 见协议 preregister_amendments); "
        "遮蔽场景无门, 退化结构本身是平台产出物; 反常档诊断见 results/r110_report.md")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_san(results), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"saved {OUT_JSON}")
    log(f"identity={'PASS' if id_ok else 'FAIL'}  "
        f"full-obs gates={n_pass}/14  verdict={verdict}")


if __name__ == "__main__":
    main()
