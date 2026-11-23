# -*- coding: utf-8 -*-
"""R117 · fracgen 条件生成 vs 统计基线 B1 —— 重构首战对照成绩单。

零重训零微调零调参: fracgen_v1_v9 权重与采样口径一律 R111 原样 (head.sample 条件
vMF 混合采样, CPU 单线程逐位确定性); models/ data/ E 盘全目录只读。

评分: 只用冻结评分卡 scripts/r110_recon_metrics.scorecard (五层 14 指标, 禁加指标),
协议权威 results/v_r110_recon_protocol.json (预注册, mtime 早于一切 R110 结果)。

设计 (接口以 scripts/r111_fracgen_smoke.py run 段实测为准):
  逐场景 fit_observed_faces → b1_generate 得到 N̂ 张面 (中心 c, 尺寸 R) 与 B1 统计方向;
  fracgen 臂复用同批 c/R (位置/尺寸配对, 消除自由度), 只把方向替换成条件生成方向:
  观测点 (锚点, mask=1, 带实测法向) + N̂ 查询点 (B1 面心, mask=0) 一起喂模型,
  encode_features → head.sample → 取查询点条件 vMF 方向作为重构面法向。
  于是评分卡差异只归因"方向重构机制"这一维, 尺度层/空间层对 B1 结构性等值。
  反常档如实记录 (生成退化/全同/越界/方向弥散致组数过配)。

阶段顺序 = mtime 铁律:
  python scripts/r117_fracgen_conditional.py --stage prereg  # -> results/v_r117_prereg.json (只写一次)
  python scripts/r117_fracgen_conditional.py --stage run     # -> results/r117_recon_compare/*

诚实条款: run 段只登记客观观察值 (gate_observations), 不自验自过; fracgen 输给 B1 如实入账
(这就是预注册的价值); 统计属研究线, 禁入客户承诺语; 优劣结论权在架构师。
历史判废数字 (14.96/16.08/9.35/15.82/7.78) 禁出现。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from r110_recon_metrics import (  # noqa: E402
    extract_faces, render_faces, scorecard, frozen_scanlines, _unit)
from r110_hrf_masker import load_scenario, SRC  # noqa: E402
from r110_baseline_b1 import (  # noqa: E402
    fit_observed_faces, b1_generate, SCENARIOS, SEEDS,
    MIN_PTS_AREAL, MIN_PTS_BOREHOLE, _san)

DEVICE = "cpu"                    # 冻结口径: CPU 单线程逐位确定性 (同 R111)
THREADS = 1
PRIMARY_CKPT = "fracgen_v1_v9.pt"  # R111 审计=唯一 123/123 全忠实 ckpt
M_PER_FACE = 8                    # 每观测面锚点上限 (掩码侧观测点)
ANCHOR_SEED = 20260828            # 锚点采样基 seed (遮蔽平台冻结 seed 同族)
SEED_MAP = {11: 117011, 22: 117022, 33: 117033}   # 生成采样 torch seed (场景无关)
B2_DIR = os.path.join(ROOT, "results", "r110_b2_scores")
B3_DIR = os.path.join(ROOT, "results", "r110_b3_scores")

PROTOCOL = os.path.join(ROOT, "results", "v_r110_recon_protocol.json")
B1_JSON = os.path.join(ROOT, "results", "r110_b1", "b1_scorecard.json")
PREREG = os.path.join(ROOT, "results", "v_r117_prereg.json")
RUN_DIR = os.path.join(ROOT, "results", "r117_recon_compare")
FRACGEN_DIR = os.path.join(RUN_DIR, "fracgen")
TABLE_JSON = os.path.join(RUN_DIR, "comparison_table.json")
MASK_DIR = os.path.join(ROOT, "results", "r110_hrf_masking")
LAYERS = ("set", "scale", "spatial", "topology", "consistency")
# 冻结协议 §b1_selfconsistency_gates 的 14 指标全名 (方向重构敏感子集在诊断中标注)
METRICS = {
    "set": ["set_count_abs_diff", "fisher_mean_direction_distance_deg", "set_proportion_L1"],
    "scale": ["trace_length_w1_rel", "powerlaw_exponent_diff"],
    "spatial": ["density_field_l1_rel", "ripley_k_max_rel_dev"],
    "topology": ["p10_rel_err", "p32_rel_err", "intersection_density_rel_err",
                 "largest_cluster_fraction_diff"],
    "consistency": ["heldout_p10_rel_err", "heldout_z_w1_rel",
                    "heldout_normal_circ_std_diff_deg"],
}
# 结构性等值层 (位置/尺寸配对 B1, 与方向无关的度量): 报告与诊断标注
GEOM_ONLY = {"scale": METRICS["scale"], "spatial": METRICS["spatial"],
             "topology_p32": ["p32_rel_err"]}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _reject_nonfinite(x):
    raise ValueError(f"非有限常量入档: {x}")


def load_json_strict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject_nonfinite)


# --------------------------------------------------------------- 共享装配 --

def load_truth():
    """真值 HRF (只读) + 冻结扫描线 + 域盒 (协议 evaluation_data 口径)。"""
    d = np.load(SRC, allow_pickle=True)
    xyz = d["xyz"].astype(np.float64)
    nrm = _unit(d["nrm"].astype(np.float64))
    fid = d["fid"].astype(np.int64)
    lab = fid >= 0
    box = np.array([xyz.min(0), xyz.max(0)])
    scanlines = frozen_scanlines(box)
    faces_true = extract_faces(xyz[lab], fid[lab])
    return dict(xyz=xyz, nrm=nrm, fid=fid, lab=lab, box=box,
                scanlines=scanlines, faces_true=faces_true)


def scenario_faces_obs(truth, scen):
    obs, m_obs = load_scenario(scen)
    obs_l = obs & truth["lab"]
    pos_o = truth["xyz"][obs_l]
    nrm_o = truth["nrm"][obs_l]
    fid_o = truth["fid"][obs_l]
    min_pts = MIN_PTS_BOREHOLE if scen.startswith("borehole") else MIN_PTS_AREAL
    faces_obs = fit_observed_faces(pos_o, nrm_o, fid_o, min_pts)
    return faces_obs, m_obs, (pos_o, nrm_o, fid_o, min_pts)


def anchor_stream_sha(scen, pos_o, nrm_o, fid_o, min_pts):
    """锚点选择 (确定性) 的内容指纹: 与场景名 hash 绑定, 供 run 段复核。"""
    rng = np.random.default_rng(ANCHOR_SEED + zlib.crc32(scen.encode("utf-8")) % 10000)
    h = hashlib.sha256()
    for fv in np.unique(fid_o):
        idx = np.where(fid_o == int(fv))[0]
        if len(idx) < min_pts:
            continue
        pick = rng.choice(len(idx), size=min(M_PER_FACE, len(idx)), replace=False)
        sel = np.sort(idx[pick])
        h.update(pos_o[sel].astype(np.float64).tobytes())
        h.update(nrm_o[sel].astype(np.float64).tobytes())
    return h.hexdigest()


def gather_anchor_points(pos_o, nrm_o, fid_o, min_pts, scen, faces_obs=None):
    """每观测面确定性抽 ≤M_PER_FACE 个观测点 (升序索引) 作 mask=1 锚点。

    faces_obs 提供时同步返回逐锚点的归一化前长度 (2R̂_该面, 观测侧量, 无真值)。
    """
    rng = np.random.default_rng(ANCHOR_SEED + zlib.crc32(scen.encode("utf-8")) % 10000)
    a_pos, a_nrm, a_len = [], [], []
    fi = 0
    for fv in np.unique(fid_o):
        idx = np.where(fid_o == int(fv))[0]
        if len(idx) < min_pts:
            continue
        pick = rng.choice(len(idx), size=min(M_PER_FACE, len(idx)), replace=False)
        sel = np.sort(idx[pick])
        a_pos.append(pos_o[sel])
        a_nrm.append(nrm_o[sel])
        if faces_obs is not None:
            a_len.append(np.full(len(sel), 2.0 * faces_obs[fi]["R"]))
        fi += 1
    out = [np.concatenate(a_pos), np.concatenate(a_nrm)]
    if faces_obs is not None:
        out.append(np.concatenate(a_len))
    return tuple(out)


def b1_stream_sha(faces_obs, box, seed):
    gen = b1_generate(faces_obs, box, seed)
    c = np.ascontiguousarray(np.array([g["c"] for g in gen], dtype=np.float64))
    R = np.array([g["R"] for g in gen], dtype=np.float64)
    return gen, hashlib.sha256(c.tobytes() + R.tobytes()).hexdigest()


# ------------------------------------------------------------ 模型臂核心 --

def _load_model():
    from fractureflow.train import load_checkpoint
    torch.set_num_threads(THREADS)
    model, cfg, step, ema = load_checkpoint(
        os.path.join(ROOT, "models", PRIMARY_CKPT), DEVICE)
    model.eval()
    return model, cfg, step


def fracgen_orientations(model, cfg, faces_obs, gen, truth, scen, obs_pts, seed):
    """条件生成: 观测锚点(mask=1)+查询面心(mask=0) 同批喂模型 → 查询点条件方向。

    返回 (dirs[n_q,3], 诊断 dict)。坐标系按协议 float64 域盒中心化后除以尺度 s。
    """
    from fractureflow.data import collate
    pos_o, nrm_o, fid_o, min_pts = obs_pts
    a_pos, a_nrm, a_len = gather_anchor_points(pos_o, nrm_o, fid_o, min_pts, scen,
                                               faces_obs=faces_obs)
    Rs = np.array([f["R"] for f in faces_obs])
    s = float(np.median(Rs))
    if not (s > 0 and np.isfinite(s)):
        s = float(np.median([g["R"] for g in gen])) or 1.0
    q_pos = np.array([g["c"] for g in gen], dtype=np.float64)
    n_q = len(gen)
    box_cen = truth["box"].mean(0)
    pos = np.concatenate([a_pos, q_pos])
    nrmv = np.concatenate([a_nrm, np.zeros((n_q, 3))])
    nrmv = nrmv / np.maximum(np.linalg.norm(nrmv, axis=1, keepdims=True), 1e-9)
    lens_m = np.concatenate([a_len, np.array([g["R"] for g in gen]) * 2.0])
    maskv = np.concatenate([np.ones(len(a_pos)), np.zeros(n_q)]).astype(np.float32)
    pos_i = (pos - box_cen) / s
    net = {
        "pos": torch.as_tensor(pos_i, dtype=torch.float32),
        "nrm": torch.as_tensor(nrmv, dtype=torch.float32),
        "len": torch.as_tensor(lens_m / s, dtype=torch.float32).reshape(-1, 1),
        "lith": torch.ones(len(pos_i), dtype=torch.int64),
        "s1": np.zeros(3, dtype=np.float32),
        "s3": np.zeros(3, dtype=np.float32),
        "obs_mask": maskv,
    }
    rng999 = np.random.default_rng(999)
    b = collate([net], cfg.obs_frac, rng999, DEVICE)[0]
    assert b["pos"].shape[1] > cfg.k_knn, "L≤k_knn, kNN 退化 (9 场景实测 L≥92, 不应触发)"
    with torch.no_grad():
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        torch.manual_seed(SEED_MAP[int(seed)])
        smp = model.head.sample(h, alpha, mu_hat, n_samples=1, usage=usage)
        q_dir = _unit(smp[0][-n_q:, 0].double().numpy())
        pi, mu_p, kappa_k, kappa_i = model.head.params(h, alpha, mu_hat, usage)
    pi_np = pi[0].detach().cpu().numpy()
    kq = kappa_i[0].detach().cpu().numpy()
    usage_np = usage[0].detach().cpu().numpy()

    # 诊断 (模型侧 + 观测侧, 不消费真值): 方向弥散 / 原型使用 / 全同退化
    nd = _unit(q_dir)
    resultant = float(np.linalg.norm(_sign_align(nd).mean(0)))
    dev_obs = np.degrees(np.arccos(np.clip(
        np.abs(nd @ _sign_align_mean(a_nrm)), -1, 1)))
    return q_dir, {
        "n_anchor": int(len(a_pos)), "n_query": int(n_q), "L_model": int(b["pos"].shape[1]),
        "scale_m": s,
        "finite_ok": bool(np.isfinite(q_dir).all()),
        "max_unit_norm_dev": float(np.abs(np.linalg.norm(q_dir, axis=1) - 1).max()),
        "resultant_sign_aligned": resultant,
        "dev_vs_observed_axis_deg": {"p50": float(np.percentile(dev_obs, 50)),
                                     "p90": float(np.percentile(dev_obs, 90)),
                                     "mean": float(dev_obs.mean())},
        "proto_usage_gt02": int((usage_np > 0.02).sum()),
        "kappa_i_query_p50": float(np.percentile(kq[-n_q:], 50)),
        "pi_argmax_hist": {str(int(p)): int((pi_np[-n_q:].argmax(1) == p).sum())
                           for p in range(pi_np.shape[1])
                           if int((pi_np[-n_q:].argmax(1) == p).sum()) > 0},
    }


def _sign_align(pts):
    """法向无向铁律: 符号对齐到首向量半球后返回原样数组 (均值/内积用)。"""
    x = np.asarray(pts, dtype=np.float64)
    ref = _unit(x.mean(0))
    s = x @ ref
    s = np.where(s == 0.0, 1.0, s)
    return x * np.sign(s)[:, None]


def _sign_align_mean(pts):
    return _unit(_sign_align(np.asarray(pts, dtype=np.float64)).mean(0))


# --------------------------------------------------------------- prereg --

def stage_prereg(force=False):
    if os.path.exists(PREREG) and not force:
        raise SystemExit("[prereg] 预注册已存在, mtime 铁律禁止重写 (确需重建用 --force 自担时序责任)")
    if os.path.isdir(RUN_DIR):
        raise SystemExit("[prereg] run 结果目录已存在, 预注册不得后置")
    torch.set_num_threads(THREADS)
    truth = load_truth()
    b1 = load_json_strict(B1_JSON)
    log(f"truth faces={len(truth['faces_true'])} pts={len(truth['xyz'])}")

    scenarios = []
    for scen in SCENARIOS:
        faces_obs, m_obs, obs_pts = scenario_faces_obs(truth, scen)
        N_hat = len(faces_obs)
        assert N_hat == b1["scenarios"][scen]["n_observed_faces"], scen
        streams = {}
        for seed in SEEDS:
            _, sh = b1_stream_sha(faces_obs, truth["box"], seed)
            streams[str(seed)] = sh
        asc = anchor_stream_sha(scen, *obs_pts)
        scenarios.append(dict(
            scenario=scen, observation_type=m_obs["observation_type"],
            coverage_fraction=round(float(m_obs["coverage_fraction"]), 6),
            n_hat=N_hat, min_face_pts=obs_pts[3],
            survey_box=[[float(x) for x in truth["box"][0]],
                        [float(x) for x in truth["box"][1]]],
            b1_geom_stream_sha_by_seed=streams,
            anchor_stream_sha=asc))
        log(f"  {scen}: N_hat={N_hat} anchor_sha={asc[:10]}")

    def asset(rel):
        full = os.path.join(ROOT, rel)
        return {"file": rel, "sha256": sha256_file(full), "size_bytes": os.path.getsize(full)}

    ck = torch.load(os.path.join(ROOT, "models", PRIMARY_CKPT), map_location=DEVICE,
                    weights_only=False)
    masking_assets = {}
    for scen in SCENARIOS:
        masking_assets[scen] = {
            "npz": sha256_file(os.path.join(MASK_DIR, f"{scen}.npz")),
            "json": sha256_file(os.path.join(MASK_DIR, f"{scen}.json"))}

    doc = {
        "protocol": "v_r117_prereg", "version": "1.0",
        "task": ("R117 ★★☆ fracgen 条件生成 vs 统计基线 B1 · 重构首战 "
                 "(零重训零微调零调参, 纯推理接线)"),
        "created_at": iso_now(),
        "mtime_rule": ("本文件创建时间必须早于 results/r117_recon_compare/ 下一切产物; "
                       "守卫测试以文件系统 CreationTime 复核"),
        "course_alignment": "课题《复杂岩石非连续性智能重构》· 重构首战: 生成式 vs 统计对照成绩单",
        "scoring_authority": {
            "scorecard": "scripts/r110_recon_metrics.scorecard (冻结五层 14 指标, 禁加指标)",
            "protocol_file": "results/v_r110_recon_protocol.json",
            "protocol_sha256": sha256_file(PROTOCOL),
            "note": "真值提取/渲染/遮蔽口径全部继承 R110 冻结协议, 本单零改动"},
        "generation_asset": {
            "file": "models/" + PRIMARY_CKPT, **asset("models/" + PRIMARY_CKPT),
            "step": ck.get("step"), "tag": ck.get("tag"),
            "consumption_path": "FracGen.head.sample() 条件 vMF 混合采样 (R111 审计 123/123 全忠实)",
            "wiring_ref": "scripts/r111_fracgen_smoke.py run 段 (已验证可跑)"},
        "paired_design": {
            "position_size": "复用 B1 同 seed b1_generate 产出的面心 c 与尺寸 R (几何配对, 消除自由度)",
            "orientation": "fracgen 条件生成方向 (观测锚点 mask=1 + 查询面心 mask=0)",
            "implication": ("尺度层/空间层/topology.p32 与 B1 结构性等值 (与方向无关); "
                            "对照信息全部落在方向重构层 (set / topology 交点 / consistency 法向)"),
            "b1_geom_stream_anchor": ("每场景每 seed 的 (c,R) sha256 冻结于 scenarios[].b1_geom_stream_sha_by_seed, "
                                      "run 段必须逐位复现 (几何配对守卫)")},
        "model_io_freeze": {
            "device": DEVICE, "threads": THREADS,
            "M_PER_FACE": M_PER_FACE, "anchor_seed_base": ANCHOR_SEED,
            "anchor_seed_rule": "np.random.default_rng(ANCHOR_SEED + zlib.crc32(scenario)%10000), 面内升序索引取点",
            "coord_normalization": "float64 域盒中心化后除以 s=median(R̂_观测面) (协议数值卫生红线)",
            "sample_seed_map": {str(k): v for k, v in SEED_MAP.items()},
            "collate_obs_mask": "net 自带 obs_mask → prepare_net 幂等不再重掩码 (T77)",
            "stress_encoding": "s1=s3=0 (无应力字段, 中性编码; aniso 特征置零, 如实登记)"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "interpreter":
                        "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"},
        "seeds": SEEDS,
        "scenarios": scenarios,
        "baselines_on_disk": {
            "B1": {"file": "results/r110_b1/b1_scorecard.json", "sha256": sha256_file(B1_JSON),
                   "present": True},
            "B2": {"dir": "results/r110_b2_scores/", "present": os.path.isdir(B2_DIR),
                   "note": "缺席如实注明" if not os.path.isdir(B2_DIR) else "在盘"},
            "B3": {"dir": "results/r110_b3_scores/", "present": os.path.isdir(B3_DIR),
                   "note": "缺席如实注明" if not os.path.isdir(B3_DIR) else "在盘"}},
        "assets": {"protocol": asset("results/v_r110_recon_protocol.json"),
                   "recon_metrics": asset("scripts/r110_recon_metrics.py"),
                   "hrf_masker": asset("scripts/r110_hrf_masker.py"),
                   "baseline_b1": asset("scripts/r110_baseline_b1.py"),
                   "b1_scorecard": asset("results/r110_b1/b1_scorecard.json"),
                   "hrf_src": {"path": SRC, "size_bytes": os.path.getsize(SRC),
                               "sha256": sha256_file(SRC)},
                   "masking": masking_assets},
        "gates_process_only": {
            "G1_asset_zero_drift": "run 段复核 ckpt/协议/B1/遮蔽 9 场景 npz+json sha256 全等预注册",
            "G2_schema": "每 fracgen 逐场景 JSON 含五层全 14 指标 (allow_nan=False, 非有限→None)",
            "G3_determinism": "同 seed full×seed11 生成方向 float64 sha 双算一致 (run 内复算 + 测试跨进程)",
            "G4_geom_pairing": "run 段 b1_generate (c,R) 逐位复现预注册 stream sha",
            "G5_timing": "本 prereg 创建时间早于 r117_recon_compare/ 全部产物 (守卫测试复核)",
            "G6_completeness": "9 场景 × 3 seeds 全部落盘 + comparison_table 全基线在列或缺席声明",
            "discipline": "run 段只登记 gate_observations 客观观察值, 判级由守卫测试+架构师完成, 不自授"},
        "significance_rule": {
            "band": "2σ, σ 取 B1 同场景同指标 seed 标准差 (协议 §aggregation)",
            "win": "mean(fracgen) < mean(B1) - 2·max(std_B1,1e-9) 且方向正确 (指标越小越好)",
            "loss": "mean(fracgen) > mean(B1) + 2·max(std_B1,1e-9)",
            "tie_structural": "尺度/空间/topology.p32 因几何配对恒等, 不参与胜负判读 (报告标注)"},
        "wiring_fidelity_anchor": {
            "rule": ("run 段必须先复现 R111 已验证接线: r111_fracgen_smoke.probe_completion"
                     "(S1_natural, primary ckpt, k=0) 的 float32 sha256 == results/r111_smoke_stats.json "
                     "登记值 (环境+模型+采样路径三位一体忠实性检查, 失败即拒跑)"),
            "r111_stats_file": "results/r111_smoke_stats.json",
            "r111_stats_sha256": sha256_file(os.path.join(ROOT, "results", "r111_smoke_stats.json")),
            "r111_gate_sha256": sha256_file(os.path.join(ROOT, "results", "v_r111_mini_gate.json"))},
        "design_time_probe_disclosure": (
            "预注册前做过接口设计探针 (非评分调优): ①B1 几何流回放 9 场景 N̂/(c,R) 逐位锚定; "
            "②坐标单位归一化 (median R̂ vs 原始米制) 对 full×seed11 fisher 77.2 vs 75.9 — 不敏感, "
            "按协议数值卫生原则选定 median R̂ 归一; ③模型输出有限性/单位范数/R111 sha 复现。"
            "此三查只为接口定型, 未据此调任何参数; 主战结果以下方冻结规则跑出的全部数字为准。"),
        "honest_branches_prewritten": [
            "fracgen 输给 B1 → 如实入账 (生成方向弥散/组数过配属预期反常档)",
            "部分层赢部分层输 → 逐层如实; 结构性等值层不计胜负",
            "全部数字属研究线对照, 禁入客户承诺语"],
        "line_discipline": {
            "research_line_only": True, "forbidden_numbers": ["14.96", "16.08", "9.35", "15.82", "7.78"],
            "readonly": ["models/", "data/", "E:/fracture_data/"], "zero_training": True,
            "frozen_anchors_cite_only": ["36.687", "12.37", "0.37", "11.05±2.14",
                                         "9.82±0.66", "12.554", "P32*=0.2405"]},
    }
    os.makedirs(os.path.dirname(PREREG), exist_ok=True)
    with open(PREREG, "w", encoding="utf-8") as f:
        json.dump(_san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[prereg] -> {PREREG}")


# ------------------------------------------------------------------- run --

def verify_assets(prereg):
    def chk(name, path, want):
        got = sha256_file(path)
        assert got == want, f"资产漂移 {name}: {got[:12]} != 预注册 {want[:12]}"
    a = prereg["assets"]
    chk("protocol", PROTOCOL, a["protocol"]["sha256"])
    chk("recon_metrics", os.path.join(ROOT, "scripts/r110_recon_metrics.py"), a["recon_metrics"]["sha256"])
    chk("hrf_masker", os.path.join(ROOT, "scripts/r110_hrf_masker.py"), a["hrf_masker"]["sha256"])
    chk("baseline_b1", os.path.join(ROOT, "scripts/r110_baseline_b1.py"), a["baseline_b1"]["sha256"])
    chk("b1_scorecard", B1_JSON, a["b1_scorecard"]["sha256"])
    chk("hrf_src", SRC, a["hrf_src"]["sha256"])
    chk("ckpt", os.path.join(ROOT, "models/" + PRIMARY_CKPT), prereg["generation_asset"]["sha256"])
    for scen, h in a["masking"].items():
        chk(f"masking/{scen}.npz", os.path.join(MASK_DIR, f"{scen}.npz"), h["npz"])
        chk(f"masking/{scen}.json", os.path.join(MASK_DIR, f"{scen}.json"), h["json"])
    log("asset zero-drift verified (G1)")


def verify_r111_wiring_anchor():
    """复现 R111 已验证接线: S1_natural completion#0 sha 逐位一致, 否则拒跑。"""
    st_path = os.path.join(ROOT, "results", "r111_smoke_stats.json")
    want = load_json_strict(st_path)["results"]["S1_natural"]["by_ckpt"][PRIMARY_CKPT][
        "determinism"]["completion0_sha256"]
    import r111_fracgen_smoke as r111
    gate = load_json_strict(os.path.join(ROOT, "results", "v_r111_mini_gate.json"))
    nets = r111.load_val_nets()
    s1 = [s for s in gate["scenarios"] if s["id"] == "S1_natural"][0]
    net = nets[s1["net_index"]]
    mask = r111.build_mask(net, "S1_natural")
    got = hashlib.sha256(r111.probe_completion(
        net, mask, PRIMARY_CKPT, 0).tobytes()).hexdigest()
    assert got == want, f"R111 接线忠实性锚失败: {got[:12]} != {want[:12]}"
    log("R111 wiring-fidelity anchor reproduced (G0)")


def stage_run(single=None):
    if not os.path.exists(PREREG):
        raise SystemExit("[run] 缺预注册, 时序违规, 拒跑")
    prereg = load_json_strict(PREREG)
    verify_assets(prereg)
    wf = prereg["wiring_fidelity_anchor"]
    got_stats_sha = sha256_file(os.path.join(ROOT, wf["r111_stats_file"]))
    assert got_stats_sha == wf["r111_stats_sha256"], "r111_smoke_stats.json 漂移 (锚文件被改)"
    verify_r111_wiring_anchor()
    torch.set_num_threads(THREADS)
    os.makedirs(FRACGEN_DIR, exist_ok=True)
    truth = load_truth()
    model, cfg, step = _load_model()
    assert step == prereg["generation_asset"]["step"], f"ckpt step 漂移 {step}"
    scen_list = [single] if single else SCENARIOS

    scen_results = {}
    geom_pair_ok = True
    det_double = True
    for sc in prereg["scenarios"]:
        scen = sc["scenario"]
        if scen not in scen_list:
            continue
        faces_obs, m_obs, obs_pts = scenario_faces_obs(truth, scen)
        asc = anchor_stream_sha(scen, *obs_pts)
        assert asc == sc["anchor_stream_sha"], f"{scen} 锚点流漂移"
        per_seed = []
        for seed in SEEDS:
            gen, sh = b1_stream_sha(faces_obs, truth["box"], seed)
            if sh != sc["b1_geom_stream_sha_by_seed"][str(seed)]:
                geom_pair_ok = False
                log(f"  !! 几何配对 stream 漂移 {scen} seed{seed}")
            q_dir, diag = fracgen_orientations(model, cfg, faces_obs, gen, truth, scen, obs_pts, seed)
            faces_rec_src = [dict(c=gen[i]["c"], n=q_dir[i], R=gen[i]["R"]) for i in range(len(gen))]
            pos_g, fid_g = render_faces(faces_rec_src, seed=seed)
            faces_rec = extract_faces(pos_g, fid_g) if len(gen) else []
            sc_card = scorecard(faces_rec, truth["faces_true"], truth["box"], truth["scanlines"]) \
                if faces_rec else None
            dir_sha = hashlib.sha256(
                np.ascontiguousarray(q_dir, dtype=np.float64).tobytes()).hexdigest()
            per_seed.append(dict(seed=seed, n_generated=len(gen),
                                 gen_dir_sha256=dir_sha, diag=diag, scorecard=_san(sc_card)))
            log(f"  {scen} seed{seed}: N={len(gen)} fisher="
                f"{(sc_card or {'set':{}})['set'].get('fisher_mean_direction_distance_deg', float('nan')):.2f}")
        if scen == "full":
            # G3 run 内双算: full×seed11 独立重采样一次, sha 必须一致
            gen11, _ = b1_stream_sha(faces_obs, truth["box"], SEEDS[0])
            q2, _ = fracgen_orientations(model, cfg, faces_obs, gen11, truth, scen, obs_pts, SEEDS[0])
            sha2 = hashlib.sha256(np.ascontiguousarray(q2, dtype=np.float64).tobytes()).hexdigest()
            det_double = (sha2 == per_seed[0]["gen_dir_sha256"])
        scen_results[scen] = dict(
            scenario=scen, observation_type=m_obs["observation_type"],
            coverage_fraction=round(float(m_obs["coverage_fraction"]), 6),
            n_hat=len(faces_obs), n_true_faces=len(truth["faces_true"]),
            min_face_pts=obs_pts[3], per_seed=per_seed)
        with open(os.path.join(FRACGEN_DIR, f"{scen}.json"), "w", encoding="utf-8") as f:
            json.dump(_san(scen_results[scen]), f, ensure_ascii=False, indent=2, allow_nan=False)
    return prereg, truth, model, cfg, scen_results, geom_pair_ok, det_double


def load_fracgen_results():
    scen_results = {}
    for scen in SCENARIOS:
        with open(os.path.join(FRACGEN_DIR, f"{scen}.json"), encoding="utf-8") as f:
            scen_results[scen] = json.load(f)
    return scen_results


def load_baseline_arm(dirname, name):
    """B2/B3 臂加载 (两种合格形态):
    a) 逐场景 <scen>.json ×9, 每份 per_seed≥3 且含 scorecard (R110/B2 惯例);
    b) 合并件 (R116 实际交付形态): 目录内单一 *scorecard*.json,
       arms.ARM_MAIN[<scen>].per_seed[].scorecard 层嵌套同构。
    任一形态不完整/不合格 → 在飞半成品, R48 纪律登记移交不消费。"""
    if not os.path.isdir(dirname):
        return None, "不在盘 (未落盘), 缺席如实注明"
    # 形态 a
    files = {s: os.path.join(dirname, f"{s}.json") for s in SCENARIOS}
    if all(os.path.exists(p) for p in files.values()):
        data = {}
        try:
            for s, p in files.items():
                j = load_json_strict(p)
                ps = j.get("per_seed") or j.get("scenarios", {}).get(s, {}).get("per_seed")
                assert ps and len(ps) >= 3 and all("scorecard" in x for x in ps), s
                data[s] = {"per_seed": ps}
        except Exception as e:
            return None, f"逐场景件解析/结构不合格 ({e}), 登记移交不消费"
        return data, "在盘 (9 场景 × ≥3 seeds 完整, 已并入对照)"
    # 形态 b: 合并件
    merged = [f for f in sorted(os.listdir(dirname))
              if f.endswith(".json") and "scorecard" in f]
    for mf in merged:
        try:
            j = load_json_strict(os.path.join(dirname, mf))
            arm = j.get("arms", {}).get("ARM_MAIN")
            if not arm:
                continue
            data = {}
            for s in SCENARIOS:
                ps = arm[s]["per_seed"]
                assert ps and len(ps) >= 3 and all("scorecard" in x for x in ps), s
                data[s] = {"per_seed": ps}
            return data, (f"在盘 (R116 关账形态: 合并件 {mf} 主臂 ARM_MAIN, "
                          "9 场景 × 3 seeds 完整, 已并入对照)")
        except Exception:
            continue
    missing = [s for s, p in files.items() if not os.path.exists(p)]
    return None, (f"目录存在但不完整 (缺 {len(missing)}/9 逐场景件且无合格合并件), "
                  "判定为在飞半成品 — 登记移交, 不消费")


def build_table(prereg, scen_results):
    b1 = load_json_strict(B1_JSON)
    arms = {"B1": {s: b1["scenarios"][s] for s in SCENARIOS}}
    presence = {"B1": {"present": True, "note": "在盘 (results/r110_b1/b1_scorecard.json)"}}
    for name, dirname in (("B2", B2_DIR), ("B3", B3_DIR)):
        data, note = load_baseline_arm(dirname, name)
        presence[name] = {"present": data is not None, "note": note,
                          "path": os.path.relpath(dirname, ROOT)}
        if data is not None:
            arms[name] = data

    def agg(per_seed, layer, metric):
        vals = []
        for p in per_seed:
            sc = p.get("scorecard")
            if sc is None:
                continue
            v = sc[layer].get(metric)
            if v is None:
                continue
            vals.append(float(v))
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    def flag(fm, bm, bs, structural):
        if structural:
            return "structural_tie_vs_B1"
        if fm is None or bm is None:
            return "na"
        band = 2 * max(bs if bs is not None else 0.0, 1e-9)
        if fm < bm - band:
            return "fracgen_win"
        if fm > bm + band:
            return "fracgen_loss"
        return "within_2sigma"

    def is_structural_vs_b1(layer, metric):
        return (layer in ("scale", "spatial") or (layer == "topology" and metric == "p32_rel_err"))

    table = {
        "protocol": "v_r117_prereg/1.0", "prereg_sha256": sha256_file(PREREG),
        "protocol_scorecard_sha256": prereg["scoring_authority"]["protocol_sha256"],
        "ckpt": prereg["generation_asset"]["file"],
        "seeds": SEEDS,
        "baselines_presence": presence,
        "fracgen_geometry_paired_with": ("B1 (面心 c 与尺寸 R 逐 seed 配对复用; 方向为 fracgen 条件生成)。"
                                         "对 B1: 尺度/空间/topology.p32 结构性等值不判胜负; "
                                         "对 B2/B3: 这些格反映的是配对的 B1 位置/尺寸模型 vs 其自身位置模型, "
                                         "方向模型不参与 (判读权在架构师)"),
        "scenarios": {}, "summary": {"by_baseline": {}},
    }
    counts = {b: {"fracgen_win": 0, "fracgen_loss": 0, "within_2sigma": 0,
                  "structural_tie_vs_B1": 0, "na": 0} for b in arms}
    for scen, res in scen_results.items():
        row = {"observation_type": res["observation_type"],
               "n_hat": res["n_hat"], "metrics": {}}
        for layer in LAYERS:
            row["metrics"][layer] = {}
            for metric in METRICS[layer]:
                fa = agg(res["per_seed"], layer, metric)
                cell = {"fracgen_mean": fa["mean"], "fracgen_std": fa["std"],
                        "n_seeds": fa["n"], "baselines": {}}
                for bname, bdata in arms.items():
                    ps = bdata[scen]["per_seed"]
                    ba = agg(ps, layer, metric)
                    if bname == "B1":
                        st = is_structural_vs_b1(layer, metric)
                    else:
                        st = False
                    fl = flag(fa["mean"], ba["mean"], ba["std"], st)
                    counts[bname][fl] += 1
                    cell["baselines"][bname] = {
                        "mean": ba["mean"], "std": ba["std"], "n": ba["n"],
                        "diff_gen_minus_base": (None if (fa["mean"] is None or ba["mean"] is None)
                                                else fa["mean"] - ba["mean"]),
                        "band_2sigma": None if ba["std"] is None else 2 * max(ba["std"], 1e-9),
                        "flag": fl}
                row["metrics"][layer][metric] = cell
        table["scenarios"][scen] = row
    for bname, cc in counts.items():
        table["summary"]["by_baseline"][bname] = {"counts": cc, "on_disk": bname in arms}
    with open(TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(_san(table), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[table] -> {TABLE_JSON}  counts=" + json.dumps(
        {b: c["counts"] for b, c in table["summary"]["by_baseline"].items()},
        ensure_ascii=False, allow_nan=False))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prereg", "run", "table"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    if args.stage == "prereg":
        stage_prereg(force=args.force)
        return
    if args.stage == "table":
        # 表重建: fracgen 结果不动, 只按在盘基线刷新 comparison_table (确定性, 无时间戳)
        prereg = load_json_strict(PREREG)
        table = build_table(prereg, load_fracgen_results())
        man_path = os.path.join(RUN_DIR, "run_manifest.json")
        if os.path.exists(man_path):
            man = load_json_strict(man_path)
            man["flag_counts"] = table["summary"]["by_baseline"]
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(_san(man), f, ensure_ascii=False, indent=2, allow_nan=False)
        return
    (prereg, truth, model, cfg, scen_results,
     geom_pair_ok, det_double) = stage_run(single=args.scenario)
    if args.scenario is None:
        table = build_table(prereg, scen_results)
        doc = {"gate_observations": {
            "G0_r111_wiring_anchor_reproduced": True,
            "G1_asset_zero_drift": True,
            "G3_determinism_double_run": det_double,
            "G4_geom_pairing_bit_equal": geom_pair_ok,
            "G6_all_scenarios_present": len(scen_results) == 9,
            "note": "只登记客观观察值; G2/G5 由守卫测试复核 (schema/时序); 判级不自授"},
            "flag_counts": table["summary"]["by_baseline"]}
        with open(os.path.join(RUN_DIR, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[run] done scenarios={len(scen_results)}")


if __name__ == "__main__":
    main()
