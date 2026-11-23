# -*- coding: utf-8 -*-
"""R127 · κ 浓度条件化消融 —— R121 病灶 (被保留原型逐点浓度 κ_i 过弥散) 的因果验证, 零训练零调参。

臂 (阶梯 ×2/×4 为任务书预注册全集, 全部报告禁挑):
  C0   R121/R117 原臂配置复跑 (换轨锚: 27 个 gen_dir sha 与 R117 及 R121 C0 在盘件逐位一致);
  C3a  C1 基臂 (k=观测 leader 组数, usage 前 k, pi 掩后重归一) + 采样浓度 κ_i×2;
  C3b  C1 基臂 + κ_i×4;
  ×1 恒等检查 (不落盘目录): C3 代码路径 κ×1 必须逐位复现 R121 C1 在盘 gen_dir sha (27/27, 基座锚)。

干预唯一性: 在 restricted_sample 的 kk=gather(kappa_i) 之后、1e3 稳定 clamp 之前插入 kk=kk·F。
模型内部 kappa_max=80 的架构钳帽先于本干预 (锐化绕过表达帽, 该帽本身登记为 E3 架构观察)。
multinomial→rand→randn 随机流与 C1 完全同序同量 → C3a/C3b 与 C1 组件选择逐点相同,
唯一差异 = 绕同一原型的极角收缩 (w→1)。κ_i≤80 → ×4≤320 << 1e3, 稳定 clamp 不参与。

评分口径与 R117/R121 完全一致 (R110 冻结五层 14 指标, 9 场景 × seeds 11/22/33);
B1/B2/B3 与 R117/R121 在盘数字只引用不重跑; R117/R121 全家只读 (import 前后 sha256 双测)。

阶段顺序即 mtime 铁律:
  python scripts/r127_kappa_sharpen.py --stage prereg   # -> results/v_r127_prereg.json
  python scripts/r127_kappa_sharpen.py --stage run      # -> results/r127_calib_compare/**
  python scripts/r127_kappa_sharpen.py --stage table    # 重算对照表 (幂等)

诚实条款: run 段只登记主断言观察值, 判级由守卫测试+架构师完成; GREEN 不达标 → RED 分支
(弥散源自数据/观测本身, 原型座次/质量) 如实入册; 全部数字研究线, 禁入承诺语; 判废数字禁出现。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import r117_fracgen_conditional as X  # noqa: E402  只读复用 (sha 双测在 run 段)
from r110_recon_metrics import (  # noqa: E402
    extract_faces, render_faces, scorecard, leader_set_clusters, _unit)

SEEDS = X.SEEDS
SCENARIOS = X.SCENARIOS
SINGLE_SET_SCENARIOS = ["full", "outcrop_10", "outcrop_30", "outcrop_50",
                        "station_30", "multi_30_2"]   # 面状观测单组场景 (k_eff 判据域, R121 同集)
ARMS = ("C0", "C3a", "C3b")
KAPPA_SCALES = {"C3a": 2.0, "C3b": 4.0}   # 任务书预注册阶梯 (禁挑)
GREEN_FISHER_MAX = 40.0   # 主断言阈值 (任务书预写死: 较 R121 C1 的 62.14° 减半线, 不回调)
KEFF_RANGE = (1, 2)       # 任务书: 单组场景 k_eff 向 [1,2] 收敛
WEIGHT_GUARD_MAX = 1.2e-7
LAYERS = X.LAYERS
METRICS = X.METRICS

PREREG = os.path.join(ROOT, "results", "v_r127_prereg.json")
RUN_DIR = os.path.join(ROOT, "results", "r127_calib_compare")
TABLE_JSON = os.path.join(RUN_DIR, "calib_comparison.json")
MANIFEST = os.path.join(RUN_DIR, "run_manifest.json")
R117_DIR = os.path.join(ROOT, "results", "r117_recon_compare")
R121_DIR = os.path.join(ROOT, "results", "r121_calib_compare")
R121_TABLE = os.path.join(R121_DIR, "calib_comparison.json")

R117_PROTECTED = [
    "scripts/r117_fracgen_conditional.py", "scripts/r117_make_report.py",
    "scripts/r117_append_ledger.py", "results/v_r117_prereg.json",
    "results/r117_recon_compare/comparison_table.json",
    "results/r117_recon_compare/run_manifest.json", "results/digital_ledger.json",
    "results/v_r110_recon_protocol.json", "scripts/r110_recon_metrics.py",
    "results/r110_b1/b1_scorecard.json",
    "results/r110_b2_scores/full.json", "results/r110_b3_scores/b3_scorecard.json",
]
R121_PROTECTED = [
    "scripts/r121_group_calib.py", "scripts/r121_make_report.py",
    "scripts/r121_append_ledger.py", "results/v_r121_prereg.json",
    "results/r121_calib_compare/calib_comparison.json",
    "results/r121_calib_compare/run_manifest.json",
]
PROTECTED = R117_PROTECTED + R121_PROTECTED \
    + [f"results/r117_recon_compare/fracgen/{s}.json" for s in SCENARIOS] \
    + [f"results/r121_calib_compare/{a}/{s}.json" for a in ("C0", "C1", "C2")
       for s in SCENARIOS] \
    + [f"models/{X.PRIMARY_CKPT}"]

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def creation_wall(path):
    """本机墙钟 (UTC+8) 锚定时点字符串 (坑 21, R116/R121 范例口径)。"""
    ct = os.stat(path).st_ctime
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ct + 8 * 3600)), ct


def sha_set():
    return {p: X.sha256_file(os.path.join(ROOT, p)) for p in PROTECTED}


def load_json_strict(path):
    def _reject(x):
        raise ValueError(f"非有限常量入档: {x} in {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject)


def sha64(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------- 变体核心 --

def build_batch(faces_obs, gen, truth, scen, obs_pts):
    """镜像 R117 fracgen_orientations 的装配 (逐位同构, 由 C0 换轨锚断言背书)。"""
    from fractureflow.data import collate
    pos_o, nrm_o, fid_o, min_pts = obs_pts
    a_pos, a_nrm, a_len = X.gather_anchor_points(pos_o, nrm_o, fid_o, min_pts, scen,
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
    b = collate([net], 0.4, np.random.default_rng(999), X.DEVICE)[0]
    assert b["pos"].shape[1] > 16
    return b, n_q


def restricted_sample_sharp(model, b, n_q, keep_idx, seed, kappa_scale):
    """R121 restricted_sample 的逐 op 复刻 + 唯一干预: kk = gather(kappa_i)·kappa_scale。

    插入点在模型 kappa_max 钳帽之后、1e3 稳定 clamp 之前; 随机流 (multinomial→rand→randn)
    与 R121 完全同序同量 → scale=1.0 时逐位等于 R121 C1 (基座恒等检查), scale>1 时组件
    选择逐点不变, 唯一差异 = 绕原型的极角收缩。

    返回 (query 行 dirs [L,3] 全点 float64, 权重和最大偏差, usage, 诊断 dict)。
    """
    from fractureflow.gen_head import unit_norm, rotate_to_mu
    with torch.no_grad():
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        # 锚点: 同一 (h,alpha,mu,usage) 上原样 head.sample → 必须等于 R117 落盘方向 (装配同构证明)
        torch.manual_seed(X.SEED_MAP[int(seed)])
        smp0 = model.head.sample(h, alpha, mu_hat, n_samples=1, usage=usage)
        pi, mu, kappa_k, kappa_i = model.head.params(h, alpha, mu_hat, usage)
        B, L, K = pi.shape
        m = torch.zeros(K, dtype=pi.dtype)
        m[torch.as_tensor(keep_idx, dtype=torch.long)] = 1.0
        pi_keep = pi * m.view(1, 1, K)
        row_sum = pi_keep.sum(-1, keepdim=True)
        pi_r = pi_keep / row_sum.clamp_min(1e-30)
        wsum_dev = float((pi_r.sum(-1) - 1.0).abs().max())
        assert (m.sum() > 0).item()
        comps = torch.multinomial(pi_r.reshape(B * L, K), 1, replacement=True) \
            .reshape(B, L, 1)
        mu_c = torch.gather(mu.unsqueeze(1).expand(B, L, K, 3), 2,
                            comps.unsqueeze(-1).expand(B, L, 1, 3))
        kk = torch.gather(kappa_i, 2, comps).unsqueeze(-1)
        kk_raw_max = float(kk.max())
        kk = kk * float(kappa_scale)
        kk = torch.minimum(kk, torch.tensor(1e3, device=kk.device))
        u = torch.rand(B, L, 1, 1, device=kk.device)
        w = 1.0 + (1.0 / kk) * torch.log(u + (1.0 - u) * torch.exp(-2.0 * kk))
        w = w.clamp(-1.0, 1.0)
        s = torch.sqrt(torch.clamp(1.0 - w.pow(2), min=0.0))
        z = torch.randn(B, L, 1, 2, device=kk.device)
        zn = unit_norm(z)
        z2 = torch.cat([s * zn, w], dim=-1)
        out = rotate_to_mu(z2, mu_c)
    anchor_dirs = smp0[0, :, 0].double().numpy()
    dirs = out[0, :, 0].double().numpy()
    usage_np = usage[0].detach().cpu().numpy()
    kq = kappa_i[0, -n_q:, :].detach().cpu().numpy()
    kmax = float(getattr(model.head, "kappa_max", float("nan")))
    diag = {
        "kappa_scale": float(kappa_scale),
        "kappa_max_arch": kmax,
        "kk_raw_max": kk_raw_max,
        "kk_after_scale_max": float(kk.max()),
        "stability_clamp_binding": bool(kk_raw_max * float(kappa_scale) > 1e3),
        "kappa_i_query_p50": float(np.percentile(kq, 50)),
        "kappa_i_query_p90": float(np.percentile(kq, 90)),
        "kappa_i_cap_sat_frac": float(np.mean(kq >= 0.999 * kmax)) if np.isfinite(kmax) else None,
    }
    return dirs, anchor_dirs, wsum_dev, usage_np, diag


def keep_protos_from_usage(usage_np, k):
    k = int(min(max(k, 1), len(usage_np)))
    order = np.argsort(-usage_np, kind="stable")
    return sorted(int(i) for i in order[:k]), k


def proto_floor_diag(model, b, n_q, keep_idx, usage_np, truth_leader_means):
    """原型座次地板诊断 (oracle, 只登记不设闸): κ→∞ 时生成方向→保留原型加权均值。

    返回 (与真值 leader 均值的角距 deg, 保留原型间最大两两角距 deg)。
    """
    with torch.no_grad():
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        pi, mu, kappa_k, kappa_i = model.head.params(h, alpha, mu_hat, usage)
    pi_q = pi[0, -n_q:, :].detach().cpu().numpy()[:, list(keep_idx)]
    pi_q = pi_q / np.maximum(pi_q.sum(-1, keepdims=True), 1e-30)
    w_mean = pi_q.mean(0)
    mus = mu[0].detach().cpu().numpy()[list(keep_idx)]
    order = np.argsort(-usage_np, kind="stable")
    top = int(np.where(np.asarray(sorted(keep_idx)) == order[0])[0][0])
    ref = mus[top]
    aligned = np.array([v if float(v @ ref) >= 0 else -v for v in mus])
    d = _unit((w_mean[:, None] * aligned).sum(0))
    ang_t = min(float(np.degrees(np.arccos(np.clip(abs(float(d @ t)), -1, 1))))
                for t in np.atleast_2d(truth_leader_means))
    pw = [float(np.degrees(np.arccos(np.clip(abs(float(a @ b2)), -1, 1))))
          for i, a in enumerate(mus) for b2 in mus[i + 1:]]
    return {"proto_floor_deg": ang_t,
            "proto_pairwise_max_deg": max(pw) if pw else 0.0}


# ------------------------------------------------------------------ prereg --

def stage_prereg(force=False):
    if os.path.exists(PREREG) and not force:
        raise SystemExit("[prereg] 已存在, mtime 铁律禁止重写 (--force 自担时序)")
    if os.path.isdir(RUN_DIR):
        raise SystemExit("[prereg] run 结果已存在, 预注册不得后置")
    torch.set_num_threads(1)
    r117_tab = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
    r121_tab = load_json_strict(R121_TABLE)
    c0_anchor = {}
    c1_identity = {}
    for scen in SCENARIOS:
        j = load_json_strict(os.path.join(R117_DIR, "fracgen", f"{scen}.json"))
        c0_anchor[scen] = {str(p["seed"]): p["gen_dir_sha256"] for p in j["per_seed"]}
        jc1 = load_json_strict(os.path.join(R121_DIR, "C1", f"{scen}.json"))
        c1_identity[scen] = {str(p["seed"]): p["gen_dir_sha256"] for p in jc1["per_seed"]}
        jc0 = load_json_strict(os.path.join(R121_DIR, "C0", f"{scen}.json"))
        for p in jc0["per_seed"]:
            assert c0_anchor[scen][str(p["seed"])] == p["gen_dir_sha256"], \
                f"R121 C0 与 R117 在盘件不一致: {scen}/{p['seed']}"
    assets = {p: X.sha256_file(os.path.join(ROOT, p)) for p in PROTECTED}
    sizes = {p: os.path.getsize(os.path.join(ROOT, p)) for p in PROTECTED}
    ct_wall, _ = creation_wall(os.path.join(R117_DIR, "comparison_table.json"))
    r121_c1_full_fisher = r121_tab["scenarios"]["full"]["set"][
        "fisher_mean_direction_distance_deg"]["C1"]["mean"]

    doc = {
        "protocol": "v_r127_prereg", "version": "1.0",
        "task": ("R127 ★★☆ κ 浓度条件化消融 — R121 病灶 (被保留原型逐点浓度 κ_i 过弥散) "
                 "的因果验证 (零训练零调参, 纯推理侧采样浓度锐化, 阶梯 ×2/×4 预注册禁挑)"),
        "created_at": iso_now(),
        "mtime_rule": ("本文件创建时间早于 results/r127_calib_compare/ 与 r127_console.log "
                       "全部产物; 守卫测试以 epoch+UTC+8 字符串双口径复核 (坑 21)"),
        "upstream_frozen": {
            "r117_comparison_table": "results/r117_recon_compare/comparison_table.json",
            "r117_table_creation_wall": ct_wall,
            "r117_full_fisher_mean_ref":
                r117_tab["scenarios"]["full"]["metrics"]["set"]["fisher_mean_direction_distance_deg"]["fracgen_mean"],
            "r121_c1_full_fisher_ref": r121_c1_full_fisher,
            "r121_c1_ref_source": "results/r121_calib_compare/calib_comparison.json (直读)",
            "scorecard_authority": "results/v_r110_recon_protocol.json (五层 14 指标冻结)",
            "seeds": SEEDS, "scenarios": SCENARIOS},
        "arms": {
            "C0": "R117/R121 原臂配置原样复跑 (复用 fracgen_orientations 函数本体); "
                  "换轨锚=27 个 gen_dir sha 与 R117 及 R121 C0 在盘件逐位一致, 不一致即整单停跑上报",
            "C3a": "C1 基臂 (k = leader_set_clusters(faces_obs, θ=20°) 组数, usage 前 k 原型, "
                   "pi 逐点掩后重归一) + 采样浓度 κ_i×2",
            "C3b": "C1 基臂 + 采样浓度 κ_i×4"},
        "intervention_spec": {
            "sampler": "R121 restricted_sample 逐 op 同款同序复刻 (multinomial→rand→randn, "
                       "n_samples=1); 唯一干预 = kk=gather(kappa_i)·F, 插入点在模型 "
                       "kappa_max=80 架构钳帽之后、1e3 稳定 clamp 之前; κ_i≤80 → ×4≤320, "
                       "稳定 clamp 不参与 (run 段逐次登记 binding=false); "
                       "seed 与 C0 相同 (117011/117022/117033) → 组件选择逐点与 C1 相同, "
                       "唯一差异 = 绕同一原型的极角收缩 (w→1)",
            "identity_check": "κ×1 恒等: C3 代码路径 scale=1.0 逐位复现 R121 C1 在盘 "
                              "gen_dir_sha256 (27/27, 基座锚, 失败=整单停跑上报); "
                              "R121 C1 评分即本管线 κ×1 档剂量响应基点",
            "keep_rule": "usage 降序前 k (stable argsort, 平手取低索引), k = k_c1 (观测 leader 组数)",
            "weight_guard": f"每行采样断言 |Σπ−1| ≤ {WEIGHT_GUARD_MAX}",
            "assembly": "build_batch 镜像 R117 装配; C0 逐场景×seed 换轨锚 + κ×1 基座锚双保险",
            "geometry": "c/R 仍与 B1 逐 seed 配对 (R117 口径), k_eff_leader 用冻结 leader 估计器于重构面",
            "arch_observation": "GenHead.kappa_max=80 架构钳帽 (模型表达不出 κ>80 的锐度, "
                                "κ_i p50≈62 贴帽) —— 本干预在采样侧绕过该帽; 帽饱和率 "
                                "(κ_i≥0.999·cap 占比) 逐场景登记为 E3 架构观察, 不设闸"},
        "primary_assertions_prewritten": {
            "P1_hard_anchors": ("C0 27/27 sha == R117/R121 在盘 且 κ×1 恒等 27/27 sha == R121 C1 "
                                "在盘 (硬闸, 失败=停单上报非判级)"),
            "P2_GREEN_C3a": (f"C3a full 场景 fisher_mean ≤ {GREEN_FISHER_MAX}° 且 单组场景集 "
                             f"{SINGLE_SET_SCENARIOS} 的 k_eff_leader ∈ {list(KEFF_RANGE)} "
                             f"(3 seeds 全部) 且 零门回退 (过程门 G1–G6 全过)"),
            "P3_GREEN_C3b": "同 P2, 臂换 C3b (κ×4)",
            "green_semantics": ("GREEN = 至少一档达标 (任务书: 『C3 至少一档使 full 组系 fisher "
                                "≤40°(较 C1 的 62.14° 减半) 且单组场景 k_eff 向 [1,2] 收敛、"
                                "零门回退』); 零门回退的操作化 = 过程门全过 (PROTECTED sha 双测 / "
                                "C0 换轨锚 / κ×1 基座锚 / 权重和守卫 / 落盘完备 NaN 拒绝 / "
                                "预注册时序) —— 指标层的 interdens 回摆是 R121 预写因果预期, "
                                "属登记观察不属回退门; 钻孔删失线反噬按任务书分场景如实报告, 不设闸"),
            "causal_expectation_registered_not_gated": (
                "若浓度锐化收复组系层, 预期: interdens 的 R117 胜项回摆向 B1 水位 (弥散驱动假设"
                "反证, R121 已登记同款), circ_std 一致性层回落, 保留原型加权均值方向 ≈ 重构组心 "
                "(proto_floor 诊断与实测 fisher 合流) —— 只登记观察值, 不设闸"),
            "oracle_diag_registered_not_gated": (
                "proto_floor_diag: 保留原型 usage 加权均值方向与真值 leader 均值的角距 "
                "(κ→∞ 组系层地板的估计, 消费真值, oracle 级仅研究线) + 保留原型间最大两两角距"),
            "RED_branch": ("P2/P3 均不达 → 『采样浓度不是 (唯一) 病灶: 弥散源自数据/观测本身 "
                           "(原型座次/质量)』入册, proto_floor 与 cap 饱和诊断定量归因 "
                           "(同样是合格交付, E3 输入更新)")},
        "k_eff_definitions": {
            "k_eff_leader": "leader_set_clusters(重构面, θ=20°) 组数 (判据域=单组场景, 全 9 场景值均落盘)"},
        "comparison_rule": {
            "vs_arm": "C3a/C3b vs C0 (同管线换轨对照): band=2·max(C0 seedσ,1e-9), 全指标 lower-better",
            "reference_cols": ("R117 在盘 gen 列 + R121 C1 列 (κ×1 基点, 直读) + B1/B2/B3 直读引用, "
                               "不重跑")},
        "zero_touch": {
            "protected_sha256_at_prereg": assets,
            "protected_sizes": sizes,
            "note": "run 段 import 前/结束+落盘后双测全部 PROTECTED sha (R117+R121 全家), "
                    "任一漂移即拒跑/停写; digital_ledger.json 活账以条目级守卫接管 "
                    "(NNS-067~069 + NNS-079/080 指针直读)"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": X.DEVICE, "threads": 1,
                        "interpreter": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"},
        "c0_anchor_expectation": c0_anchor,
        "c1_identity_anchor_expectation": c1_identity,
        "design_time_probe_disclosure": (
            "本单零设计时探针 (R121 先例可走但本单无需): 判据全部由任务书预写 (≤40°, [1,2]), "
            "无可调参数, 探针无决策价值; 预注册前仅做只读文件侦察 (R121 在盘件结构/权重和实测 "
            "0.0/GenHead kappa_max=80 源码确认), 未烧一枪未写 results/。"),
        "b2_b3_presence_at_prereg": {
            "B2": os.path.isdir(X.B2_DIR), "B3": os.path.isdir(X.B3_DIR),
            "note": "时点陈述运行时实测 (R116/R117/R121 教训); 对照列以 run 段 build 时实读为准"},
        "line_discipline": {
            "research_line_only": True, "zero_training": True,
            "forbidden_numbers": ["14.96", "16.08"],
            "readonly": ["models/", "data/", "E:/fracture_data/",
                         "results/r117_recon_compare/", "results/v_r117_prereg.json",
                         "results/r121_calib_compare/", "results/v_r121_prereg.json"]},
        "gates_process_only": {
            "G1": "PROTECTED sha (R117+R121 全家) import 前后双测零漂移",
            "G2": "C0 27/27 逐位 = R117 与 R121 C0 在盘 (P1 前半)",
            "G3": "κ×1 恒等 27/27 逐位 = R121 C1 在盘 (P1 后半, 基座锚)",
            "G4": f"采样权重和守卫 max|Σπ−1| ≤ {WEIGHT_GUARD_MAX} 全通过 (含恒等检查路径)",
            "G5": "三臂 × 9 场景 × 3 seeds 全落盘 allow_nan=False, strict 复读通过; 表含参照列或缺席声明",
            "G6": "预注册创建时间早于一切产物 (epoch + UTC+8 字符串双口径)",
            "discipline": "run 段只登记观察值不自授判级"},
    }
    os.makedirs(os.path.dirname(PREREG), exist_ok=True)
    with open(PREREG, "w", encoding="utf-8") as f:
        json.dump(X._san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[prereg] -> {PREREG}")


# --------------------------------------------------------------------- run --

def ledger_entry_guard():
    """活账 append-only 口径: R117/R121 条目在场且指针值与源头一致 (字节恒转不误杀)。"""
    led = load_json_strict(os.path.join(ROOT, "results/digital_ledger.json"))
    r117t = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
    r121t = load_json_strict(R121_TABLE)
    cur = {e["id"]: e for e in led["entries"]}
    want = {
        "NNS-067": (r117t["scenarios"]["full"]["metrics"]["set"][
            "fisher_mean_direction_distance_deg"]["fracgen_mean"],
            "results/r117_recon_compare/comparison_table.json"),
        "NNS-068": (r117t["scenarios"]["full"]["metrics"]["topology"][
            "intersection_density_rel_err"]["fracgen_mean"],
            "results/r117_recon_compare/comparison_table.json"),
        "NNS-069": (r117t["summary"]["by_baseline"]["B1"]["counts"]["fracgen_loss"],
                    "results/r117_recon_compare/comparison_table.json"),
        "NNS-079": (r121t["scenarios"]["full"]["set"][
            "fisher_mean_direction_distance_deg"]["C1"]["mean"],
            "results/r121_calib_compare/calib_comparison.json"),
        "NNS-080": (r121t["scenarios"]["full"]["topology"][
            "intersection_density_rel_err"]["C1"]["mean"],
            "results/r121_calib_compare/calib_comparison.json"),
    }
    for eid, (wv, src) in want.items():
        assert eid in cur, f"台账条目丢失: {eid}"
        assert cur[eid]["source_json"] == src, eid
        assert abs(float(cur[eid]["value"]) - round(float(wv), 4)) < 5e-5, eid
    assert "NNS-071" in cur and cur["NNS-071"]["status"] == "active-research"


def stage_run():
    if not os.path.exists(PREREG):
        raise SystemExit("[run] 缺预注册, 时序违规, 拒跑")
    sha_pre = sha_set()
    prereg = load_json_strict(PREREG)
    for p, want in prereg["zero_touch"]["protected_sha256_at_prereg"].items():
        if p == "results/digital_ledger.json":
            continue  # 活账 append-only, 字节级恒转由 ledger_entry_guard 条目级接管
        got = X.sha256_file(os.path.join(ROOT, p))
        assert got == want, f"只读件漂移: {p}"
    log("PROTECTED import-前 sha 全对齐 (G1, R117+R121 全家)")
    ledger_entry_guard()
    truth = X.load_truth()
    model, cfg, step = X._load_model()
    os.makedirs(RUN_DIR, exist_ok=True)
    for arm in ARMS:
        os.makedirs(os.path.join(RUN_DIR, arm), exist_ok=True)

    r117_tab = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
    anchor117 = prereg["c0_anchor_expectation"]
    anchor121 = prereg["c1_identity_anchor_expectation"]
    results = {a: {} for a in ARMS}
    scen_meta = {}
    identity_table = {}
    g2_ok, g3_ok, g4_ok = True, True, True

    for scen in SCENARIOS:
        faces_obs, m_obs, obs_pts = X.scenario_faces_obs(truth, scen)
        labels, means, props, _ = leader_set_clusters(faces_obs)
        k_c1 = int(len(means))
        _, means_t, _, _ = leader_set_clusters(truth["faces_true"])
        meta = {"scenario": scen, "n_hat": len(faces_obs), "k_c1_leader": k_c1,
                "k_true_leader": int(len(means_t))}
        scen_meta[scen] = meta
        per_seed = {a: [] for a in ARMS}
        identity_table[scen] = {}

        for seed in SEEDS:
            gen, stream_sha = X.b1_stream_sha(faces_obs, truth["box"], seed)
            want117 = anchor117[scen][str(seed)]
            # ---- C0: 直接复用 R117 函数本体 (换轨锚) ----
            q_dir0, diag0 = X.fracgen_orientations(model, cfg, faces_obs, gen, truth,
                                                   scen, obs_pts, seed)
            if sha64(q_dir0) != want117:
                g2_ok = False
                log(f"  !! C0 换轨锚漂移 {scen}/{seed}")
            per_seed["C0"].append(dict(gen=gen, q_dir=q_dir0, seed=seed, keep=None))
            # ---- 公共批 + keep 规则 ----
            b, n_q = build_batch(faces_obs, gen, truth, scen, obs_pts)
            with torch.no_grad():
                h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
            usage_np = usage[0].detach().cpu().numpy()
            keep_idx, k_used = keep_protos_from_usage(usage_np, k_c1)
            # ---- κ×1 恒等检查 (基座锚, 不落盘目录) ----
            dirs_id, anchor_full, wdev_id, _, diag_id = restricted_sample_sharp(
                model, b, n_q, keep_idx, seed, 1.0)
            ash = sha64(_unit(anchor_full[-n_q:]))
            if ash != want117:
                g3_ok = False
                log(f"  !! 装配锚点漂移 {scen}/{seed}")
            sh_id = sha64(_unit(dirs_id[-n_q:]))
            identity_table[scen][str(seed)] = sh_id
            if sh_id != anchor121[scen][str(seed)]:
                g3_ok = False
                raise SystemExit(f"[run] κ×1 基座恒等失败 {scen}/{seed}: "
                                 f"{sh_id[:16]} != {anchor121[scen][str(seed)][:16]} (整单停跑)")
            if wdev_id > WEIGHT_GUARD_MAX:
                g4_ok = False
            # ---- C3a / C3b: 唯一差异 = κ 缩放 ----
            for arm, scale in KAPPA_SCALES.items():
                dirs, _, wdev, _, diag = restricted_sample_sharp(
                    model, b, n_q, keep_idx, seed, scale)
                if wdev > WEIGHT_GUARD_MAX:
                    g4_ok = False
                pd = proto_floor_diag(model, b, n_q, keep_idx, usage_np, means_t)
                per_seed[arm].append(dict(
                    gen=gen, q_dir=_unit(dirs[-n_q:]), seed=seed,
                    keep={"kept": keep_idx, "k_used": k_used,
                          "usage_kept": [round(float(usage_np[i]), 4) for i in keep_idx],
                          "wsum_dev": wdev},
                    kappa_diag={**diag, "wsum_dev": wdev},
                    proto_floor_diag=pd))

        for arm in ARMS:
            entries = []
            for e in sorted(per_seed[arm], key=lambda z: z["seed"]):
                gen, q_dir, seed = e["gen"], e["q_dir"], e["seed"]
                src = [dict(c=gen[i]["c"], n=q_dir[i], R=gen[i]["R"]) for i in range(len(gen))]
                pos_g, fid_g = render_faces(src, seed=seed)
                fr = extract_faces(pos_g, fid_g)
                sc = scorecard(fr, truth["faces_true"], truth["box"], truth["scanlines"])
                _, means_r, _, _ = leader_set_clusters(fr)
                entry = dict(seed=seed, n_generated=len(gen),
                             gen_dir_sha256=sha64(q_dir),
                             k_eff_leader=int(len(means_r)),
                             keep_protos=e["keep"], scorecard=X._san(sc))
                if arm in KAPPA_SCALES:
                    entry["kappa_diag"] = e["kappa_diag"]
                    entry["proto_floor_diag"] = e["proto_floor_diag"]
                entries.append(entry)
            results[arm][scen] = entries

        for arm in ARMS:
            j = {"arm": arm, "scenario": scen,
                 "observation_type": m_obs["observation_type"],
                 "coverage_fraction": round(float(m_obs["coverage_fraction"]), 6),
                 "n_hat": len(faces_obs), "k_c1_leader": k_c1,
                 "kappa_scales": {a: KAPPA_SCALES[a] for a in KAPPA_SCALES},
                 "per_seed": results[arm][scen]}
            with open(os.path.join(RUN_DIR, arm, f"{scen}.json"), "w", encoding="utf-8") as f:
                json.dump(X._san(j), f, ensure_ascii=False, indent=2, allow_nan=False)
        log(f"{scen}: k_C1={k_c1} done")

    doc = {"gate_observations": {
        "G1_protected_sha_zero_drift": True,
        "G2_C0_bit_anchor_27of27": g2_ok,
        "G3_kappa1_identity_27of27": g3_ok,
        "G4_weight_sum_guard": g4_ok,
        "G6_prereg_before_outputs": creation_wall(PREREG)[1] <= min(
            os.stat(os.path.join(RUN_DIR, a, s + ".json")).st_ctime
            for a in ARMS for s in SCENARIOS),
        "note": "只登记客观观察值; 判级由守卫测试+架构师完成"},
        "c1_identity_sha_table": identity_table,
        "scenario_k_selection": scen_meta,
        "kappa_max_arch": float(getattr(model.head, "kappa_max", float("nan"))),
        "b2_b3_presence_at_run": {"B2": os.path.isdir(X.B2_DIR),
                                  "B3": os.path.isdir(X.B3_DIR)}}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(X._san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)

    build_calib_table(prereg, r117_tab)
    # G5: 全部落盘 strict 复读
    for a in ARMS:
        for s in SCENARIOS:
            load_json_strict(os.path.join(RUN_DIR, a, f"{s}.json"))
    load_json_strict(TABLE_JSON)
    load_json_strict(MANIFEST)
    ledger_entry_guard()
    sha_post = sha_set()
    bad = [p for p in sha_pre if sha_pre[p] != sha_post[p]
           and not p.endswith("digital_ledger.json")]
    assert not bad, f"只读件在单内被改动: {bad}"
    log("[run] PROTECTED sha 结束复测零漂移; G5 strict 复读全过; done")


def agg(per_seed_entries, layer, metric):
    vals = []
    for e in per_seed_entries:
        sc = e.get("scorecard")
        if not sc:
            continue
        v = sc[layer].get(metric)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def build_calib_table(prereg, r117_tab):
    data = {a: {s: load_json_strict(os.path.join(RUN_DIR, a, f"{s}.json"))["per_seed"]
                for s in SCENARIOS} for a in ARMS}
    b1 = load_json_strict(X.B1_JSON)
    arms_ref = {"B1": {s: b1["scenarios"][s]["per_seed"] for s in SCENARIOS}}
    for nm, dirp in (("B2", X.B2_DIR), ("B3", X.B3_DIR)):
        data_p, note = X.load_baseline_arm(dirp, nm)
        if data_p is not None:
            arms_ref[nm] = {s: data_p[s]["per_seed"] for s in SCENARIOS}
    r121_tab = load_json_strict(R121_TABLE)
    tab = {"protocol": "v_r127_prereg/1.0", "prereg_sha256": X.sha256_file(PREREG),
           "seeds": SEEDS,
           "primary_thresholds": {"green_fisher_max": GREEN_FISHER_MAX,
                                  "keff_range": list(KEFF_RANGE),
                                  "weight_guard_max": WEIGHT_GUARD_MAX,
                                  "kappa_scales": KAPPA_SCALES,
                                  "single_set_scenarios": SINGLE_SET_SCENARIOS},
           "scenarios": {}, "summary": {"vs_C0_counts": {}, "primary_observations": {}},
           "reference_presence": {"R117_table": True, "R121_C1": True, "B1": True,
                                  "B2": "B2" in arms_ref, "B3": "B3" in arms_ref}}
    counts = {"fracgen_win": 0, "fracgen_loss": 0, "within_2sigma": 0, "na": 0}
    for scen in SCENARIOS:
        r121_c1_se = load_json_strict(os.path.join(R121_DIR, "C1", f"{scen}.json"))["per_seed"]
        rowm = {}
        for layer in LAYERS:
            rowm[layer] = {}
            for metric in METRICS[layer]:
                c0 = agg(data["C0"][scen], layer, metric)
                cell = {"C0": c0, "C3a": agg(data["C3a"][scen], layer, metric),
                        "C3b": agg(data["C3b"][scen], layer, metric), "refs": {}}
                r17 = r117_tab["scenarios"][scen]["metrics"][layer][metric]
                cell["refs"]["R117"] = {"mean": r17["fracgen_mean"], "std": r17["fracgen_std"],
                                        "n": r17.get("n_seeds", 3),
                                        "source": "results/r117_recon_compare/comparison_table.json (直读)"}
                cell["refs"]["R121_C1"] = {
                    "mean": r121_tab["scenarios"][scen][layer][metric]["C1"]["mean"],
                    "std": r121_tab["scenarios"][scen][layer][metric]["C1"]["std"],
                    "source": "results/r121_calib_compare/calib_comparison.json (κ×1 基点, 直读)"}
                for bn, bd in arms_ref.items():
                    cell["refs"][bn] = agg(bd[scen], layer, metric)
                for arm in ("C3a", "C3b"):
                    v = cell[arm]
                    if v["mean"] is None or c0["mean"] is None:
                        v["flag_vs_C0"] = "na"; counts["na"] += 1
                    else:
                        band = 2 * max(c0["std"] or 0.0, 1e-9)
                        if v["mean"] < c0["mean"] - band:
                            v["flag_vs_C0"] = "fracgen_win"; counts["fracgen_win"] += 1
                        elif v["mean"] > c0["mean"] + band:
                            v["flag_vs_C0"] = "fracgen_loss"; counts["fracgen_loss"] += 1
                        else:
                            v["flag_vs_C0"] = "within_2sigma"; counts["within_2sigma"] += 1
                cell["k_eff_leader"] = {a: [e["k_eff_leader"] for e in data[a][scen]]
                                        for a in ARMS}
                cell["k_eff_leader"]["R121_C1"] = [e["k_eff_leader"] for e in r121_c1_se]
                rowm[layer][metric] = cell
        tab["scenarios"][scen] = rowm
    tab["summary"]["vs_C0_counts"] = counts

    man = load_json_strict(MANIFEST)
    gates = man["gate_observations"]
    gates_all = bool(gates.get("G1_protected_sha_zero_drift")
                     and gates.get("G2_C0_bit_anchor_27of27")
                     and gates.get("G3_kappa1_identity_27of27")
                     and gates.get("G4_weight_sum_guard")
                     and gates.get("G6_prereg_before_outputs"))
    po = {}
    for arm in ("C3a", "C3b"):
        fisher = tab["scenarios"]["full"]["set"]["fisher_mean_direction_distance_deg"]
        c0m, va = fisher["C0"], fisher[arm]
        keff = [k for s in SINGLE_SET_SCENARIOS
                for k in tab["scenarios"][s]["set"]["fisher_mean_direction_distance_deg"]
                ["k_eff_leader"][arm]]
        below = va["mean"] is not None and va["mean"] <= GREEN_FISHER_MAX
        keff_ok = all(KEFF_RANGE[0] <= k <= KEFF_RANGE[1] for k in keff)
        po[arm] = {
            "full_fisher_mean": va["mean"],
            "threshold_max": GREEN_FISHER_MAX,
            "below_threshold": below,
            "c0_full_fisher_mean": c0m["mean"],
            "r121_c1_full_fisher_ref": fisher["refs"]["R121_C1"]["mean"],
            "improves_over_r121_c1": bool(va["mean"] is not None
                                          and va["mean"] < fisher["refs"]["R121_C1"]["mean"]),
            "keff_values_single_set": keff,
            "keff_in_range_all_seeds": keff_ok,
            "process_gates_all_pass": gates_all,
            "observation_GREEN_all_three": bool(below and keff_ok and gates_all),
            "causal_full_interdens": {
                "C0": tab["scenarios"]["full"]["topology"]["intersection_density_rel_err"]["C0"]["mean"],
                arm: tab["scenarios"]["full"]["topology"]["intersection_density_rel_err"][arm]["mean"],
                "R121_C1_ref": tab["scenarios"]["full"]["topology"]["intersection_density_rel_err"]["refs"]["R121_C1"]["mean"],
                "B1_ref": tab["scenarios"]["full"]["topology"]["intersection_density_rel_err"]["refs"]["B1"]["mean"]},
            "causal_full_circ_std": {
                "C0": tab["scenarios"]["full"]["consistency"]["heldout_normal_circ_std_diff_deg"]["C0"]["mean"],
                arm: tab["scenarios"]["full"]["consistency"]["heldout_normal_circ_std_diff_deg"][arm]["mean"],
                "R121_C1_ref": tab["scenarios"]["full"]["consistency"]["heldout_normal_circ_std_diff_deg"]["refs"]["R121_C1"]["mean"]},
        }
        pf = [e["proto_floor_diag"]["proto_floor_deg"] for s in SINGLE_SET_SCENARIOS
              for e in data[arm][s]]
        po[arm]["proto_floor_single_set_p50"] = float(np.percentile(pf, 50))
    po["GREEN_any_arm"] = bool(po["C3a"]["observation_GREEN_all_three"]
                               or po["C3b"]["observation_GREEN_all_three"])
    tab["summary"]["primary_observations"] = po
    with open(TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(X._san(tab), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[table] -> {TABLE_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prereg", "run", "table"], required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.stage == "prereg":
        stage_prereg(force=args.force)
    elif args.stage == "table":
        prereg = load_json_strict(PREREG)
        build_calib_table(prereg,
                          load_json_strict(os.path.join(R117_DIR, "comparison_table.json")))
    else:
        stage_run()


if __name__ == "__main__":
    main()
