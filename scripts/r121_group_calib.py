# -*- coding: utf-8 -*-
"""R121 · 生成臂组数校准消融 —— R117 发现① (组数过配) 的因果验证, 零训练零调参。

三变体 (全部纯推理侧, 与 R117 共用装配/评分链):
  C0  R117 原臂配置原样复跑 (换轨锚: 27 个 gen_dir sha 必须与 R117 在盘件逐位一致);
  C1  观测组数钉死: k = B1 同款 leader 聚类 (θ=20°) 对观测面的组数, 混合保留 usage 前 k 原型;
  C2  vMF 混合 BIC 选型: k = argmin BIC (K∈1..8, 观测面法向, +z 半球符号对齐), 同保留规则。
  干预 = 只改混合组数来源与权重归一: pi 掩到保留原型集后逐点重归一, 采样算法/随机流与
  GenHead.sample 逐 op 同款同序 (multinomial→rand→randn), seed 与 C0 一致 (117011/22/33)。

评分口径与 R117 完全一致 (R110 冻结评分卡五层 14 指标, 9 场景 × seeds 11/22/33);
B1/B2/B3 与 R177 在盘数字只引用不重跑。R117 全家只读 (import 前后 sha256 双测)。

阶段顺序即 mtime 铁律:
  python scripts/r121_group_calib.py --stage prereg   # -> results/v_r121_prereg.json
  python scripts/r121_group_calib.py --stage run      # -> results/r121_calib_compare/**

诚实条款: run 段只登记主断言观察值, 判级由守卫测试+架构师完成; GREEN 不达标 → RED 分支
如实入册 (瓶颈登记为原型质量/混合结构); 全部数字研究线, 禁入承诺语; 判废数字禁出现。
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
    extract_faces, render_faces, scorecard, leader_set_clusters, _unit, _sign_align_z)

SEEDS = X.SEEDS
SCENARIOS = X.SCENARIOS
SINGLE_SET_SCENARIOS = ["full", "outcrop_10", "outcrop_30", "outcrop_50",
                        "station_30", "multi_30_2"]   # 面状观测单组场景 (k_eff 判据域)
ARMS = ("C0", "C1", "C2")
K_RANGE = (1, 8)          # C2 BIC 选型域 (预注册冻结)
GREEN_FISHER_MAX = 15.0   # 主断言阈值 (预写死: dev p50~11° 的 k=1 恢复线, 不回调)
KEFF_RANGE = (1, 3)
LAYERS = X.LAYERS
METRICS = X.METRICS

PREREG = os.path.join(ROOT, "results", "v_r121_prereg.json")
RUN_DIR = os.path.join(ROOT, "results", "r121_calib_compare")
TABLE_JSON = os.path.join(RUN_DIR, "calib_comparison.json")
MANIFEST = os.path.join(RUN_DIR, "run_manifest.json")
R117_DIR = os.path.join(ROOT, "results", "r117_recon_compare")

# 只读保护清单 (import 前后 + run 末三道 sha 测)
PROTECTED = [
    "scripts/r117_fracgen_conditional.py", "scripts/r117_make_report.py",
    "scripts/r117_append_ledger.py", "results/v_r117_prereg.json",
    "results/r117_recon_compare/comparison_table.json",
    "results/r117_recon_compare/run_manifest.json", "results/digital_ledger.json",
    "results/v_r110_recon_protocol.json", "scripts/r110_recon_metrics.py",
    "results/r110_b1/b1_scorecard.json",
    "results/r110_b2_scores/full.json", "results/r110_b3_scores/b3_scorecard.json",
] + [f"results/r117_recon_compare/fracgen/{s}.json" for s in SCENARIOS] \
    + [f"models/{X.PRIMARY_CKPT}"]

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def creation_wall(path):
    """本机墙钟 (UTC+8) 锚定时点字符串 (坑 21, R116 范例口径)。"""
    ct = os.stat(path).st_ctime
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ct + 8 * 3600)), ct


def sha_set():
    return {p: X.sha256_file(os.path.join(ROOT, p)) for p in PROTECTED}


def load_json_strict(path):
    def _reject(x):
        raise ValueError(f"非有限常量入档: {x} in {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject)


# ---------------------------------------------------------------- 变体核心 --

def build_batch(faces_obs, gen, truth, scen, obs_pts):
    """镜像 R117 fracgen_orientations 的装配 (逐位同构, 由 C0/锚点 sha 断言背书)。"""
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


def restricted_sample(model, b, keep_idx, seed):
    """GenHead.sample 的掩码重归一复刻: 只改组数来源与权重归一, 逐 op 同款同序。

    返回 (query 行 dirs [L,3] 全点 float64, 权重和最大偏差, pi 行和诊断)。
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
        keep_mask_sum = (m.sum() > 0).item()
        assert keep_mask_sum
        comps = torch.multinomial(pi_r.reshape(B * L, K), 1, replacement=True) \
            .reshape(B, L, 1)
        mu_c = torch.gather(mu.unsqueeze(1).expand(B, L, K, 3), 2,
                            comps.unsqueeze(-1).expand(B, L, 1, 3))
        kk = torch.gather(kappa_i, 2, comps).unsqueeze(-1)
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
    return dirs, anchor_dirs, wsum_dev, usage_np


def _kappa_mle(rbar):
    rbar = min(max(float(rbar), 1e-9), 1.0 - 1e-9)
    return rbar * (3.0 - rbar * rbar) / (1.0 - rbar ** 3)


def vmf_mix_bic_curve(x, kmax=8, iters=300, tol=1e-9):
    """确定性 vMF 混合 EM BIC 曲线 (K=1..kmax)。

    x [n,3] 已 +z 半球符号对齐; init=farthest-point (角距); κ MLE 无 n 修正 (口径预注册);
    BIC = −2·lnL + (4K−1)·ln n; 退化 (空组件/NaN) → BIC=+inf。返回 (bic[K], lnL[K], k_sel)。
    """
    n = len(x)
    bics, lls = [], []
    for K in range(1, kmax + 1):
        if n < 2 * K:
            bics.append(float("inf")); lls.append(float("-inf")); continue
        # farthest-point init (确定性)
        mus = [x[0]]
        while len(mus) < K:
            M = np.stack(mus)
            d = 1.0 - np.abs(x @ M.T).max(1)
            mus.append(x[int(np.argmax(d))])
        mus = _unit(np.stack(mus))
        w = np.full(K, 1.0 / K)
        kap = np.full(K, 10.0)
        prev = -np.inf
        ok = True
        for _ in range(iters):
            t = np.einsum("nk,mk->nm", x, mus) * kap[None, :] \
                + np.log(w + 1e-300)[None, :] + (np.log(kap) - np.log(4 * np.pi)
                                                  - kap + np.log1p(-np.exp(-2 * kap)))[None, :]
            t -= t.max(1, keepdims=True)
            p = np.exp(t)
            Z = p.sum(1, keepdims=True)
            r = p / Z
            ll = float((np.log(Z[:, 0])).sum())
            w = r.mean(0)
            num = _unit(np.einsum("nm,nj->mj", r, x) + 1e-12)
            rbar = np.linalg.norm(np.einsum("nm,nj->mj", r, x) / np.maximum(r.sum(0)[:, None], 1e-12), axis=1)
            kap = np.clip([_kappa_mle(rb) for rb in rbar], 1e-4, 2000.0)
            mus = num
            if not np.isfinite(ll):
                ok = False
                break
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        if not ok or (r.sum(0) < 1e-6).any():
            bics.append(float("inf")); lls.append(float("-inf")); continue
        # 终态 lnL
        t = np.einsum("nk,mk->nm", x, mus) * kap[None, :] \
            + np.log(w + 1e-300)[None, :] + (np.log(kap) - np.log(4 * np.pi)
                                              - kap + np.log1p(-np.exp(-2 * kap)))[None, :]
        ll = float(np.log(np.exp(t - t.max(1, keepdims=True)).sum(1)).sum() + t.max(1).sum())
        bics.append(-2 * ll + (4 * K - 1) * np.log(n))
        lls.append(ll)
    bics = np.array(bics)
    k_sel = int(np.argmin(bics)) + 1 if np.isfinite(bics).any() else 1
    return [None if not np.isfinite(b) else float(b) for b in bics], \
        [None if not np.isfinite(l) else float(l) for l in lls], k_sel


def keep_protos_from_usage(usage_np, k):
    k = int(min(max(k, 1), len(usage_np)))
    order = np.argsort(-usage_np, kind="stable")
    return sorted(int(i) for i in order[:k]), k


# ------------------------------------------------------------------ prereg --

def stage_prereg(force=False):
    if os.path.exists(PREREG) and not force:
        raise SystemExit("[prereg] 已存在, mtime 铁律禁止重写 (--force 自担时序)")
    if os.path.isdir(RUN_DIR):
        raise SystemExit("[prereg] run 结果已存在, 预注册不得后置")
    torch.set_num_threads(1)
    r117_tab = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
    c0_anchor = {}
    for scen in SCENARIOS:
        j = load_json_strict(os.path.join(R117_DIR, "fracgen", f"{scen}.json"))
        c0_anchor[scen] = {str(p["seed"]): p["gen_dir_sha256"] for p in j["per_seed"]}
    assets = {p: X.sha256_file(os.path.join(ROOT, p)) for p in PROTECTED}
    sizes = {p: os.path.getsize(os.path.join(ROOT, p)) for p in PROTECTED}
    ct_wall, _ = creation_wall(os.path.join(R117_DIR, "comparison_table.json"))

    doc = {
        "protocol": "v_r121_prereg", "version": "1.0",
        "task": ("R121 ★★☆ 生成臂组数校准消融 — R117 发现① 因果验证 "
                 "(零训练零调参, 纯推理侧组数干预)"),
        "created_at": iso_now(),
        "mtime_rule": ("本文件创建时间早于 results/r121_calib_compare/ 与 r121_console.log "
                       "全部产物; 守卫测试以 epoch+UTC+8 字符串双口径复核 (坑 21)"),
        "upstream_frozen": {
            "r117_comparison_table": "results/r117_recon_compare/comparison_table.json",
            "r117_table_creation_wall": ct_wall,
            "r117_full_fisher_mean_ref":
                r117_tab["scenarios"]["full"]["metrics"]["set"]["fisher_mean_direction_distance_deg"]["fracgen_mean"],
            "scorecard_authority": "results/v_r110_recon_protocol.json (五层 14 指标冻结)",
            "seeds": SEEDS, "scenarios": SCENARIOS},
        "arms": {
            "C0": "R117 原臂配置原样复跑 (复用其 fracgen_orientations 函数本体); "
                  "换轨锚=27 个 gen_dir_sha256 与 R117 在盘件逐位一致, 不一致即整单停跑上报",
            "C1": "k = leader_set_clusters(faces_obs, θ=20°) 组数 (B1 同款冻结估计器, 观测侧); "
                  "保留 usage 前 k 原型, pi 逐点掩后重归一",
            "C2": "k = vMF 混合 BIC argmin (K∈1..8, 观测面法向 +z 半球符号对齐, EM farthest-point 初值, "
                  "κ MLE 无 n 修正, BIC=−2lnL+(4K−1)lnn, 平手取小 k); 保留/归一规则同 C1"},
        "intervention_spec": {
            "sampler": "GenHead.sample 逐 op 同款同序复刻 (multinomial→rand→randn, n_samples=1), "
                       "唯一差异 = pi 掩码+重归一; seed 与 C0 相同 (117011/117022/117033) → "
                       "C1/C2 与 C0 共享随机流起点, 差异只来自组数干预本身",
            "keep_rule": "usage 降序前 k (stable argsort, 平手取低索引); k 超界截 12",
            "weight_guard": "每行采样断言 |Σπ_r−1| ≤ 1e-5 (k_eff 计算自洽守卫)",
            "assembly": "build_batch 镜像 R117 装配; 每次 run 内以 head.sample 锚点 sha==R117 落盘值逐场景×seed 证明同构",
            "geometry": "c/R 仍与 B1 逐 seed 配对 (R117 口径), k_eff_leader 用冻结 leader 估计器于重构面"},
        "primary_assertions_prewritten": {
            "P1_C0_bit_anchor": "C0 27/27 gen_dir sha == R117 落盘 (硬闸, 失败=停单上报非判级)",
            "P2_GREEN_C1": (f"C1 full 场景 fisher_mean ≤ {GREEN_FISHER_MAX}° 且 "
                            f"< C0 均值 − 2·max(C0 std,1e-9) 且 单组场景集 "
                            f"{SINGLE_SET_SCENARIOS} 的 k_eff_leader ∈ {KEFF_RANGE} (3 seeds 全部)"),
            "P3_GREEN_C2": f"同 P2, 臂换 C2",
            "causal_expectation_registered_not_gated": (
                "若组系恢复成立, 预期拓扑层 interdens 的 R117 胜项回摆向 B1 水位 "
                "(胜→弥散驱动的假设反证), 及 circ_std 回落 —— 只登记观察值, 不设闸"),
            "RED_branch": "P2/P3 不达 → 瓶颈登记『原型质量/混合结构』(E3 输入更新), 同样是合格交付"},
        "k_eff_definitions": {
            "k_eff_leader": "leader_set_clusters(重构面, θ=20°) 组数 (判据域=单组场景, 全 9 场景值均落盘)",
            "k_eff_participation": "1/Σ_k w̄_k² , w̄=掩码重归一后 query 点平均混合权重 (结构自洽诊断)"},
        "comparison_rule": {
            "vs_arm": "C1/C2 vs C0 (同管线换轨对照): band=2·max(C0 seedσ,1e-9), 全指标 lower-better",
            "reference_cols": "R117 在盘 gen 列 (=C0 预期) + B1/B2/B3 直读引用, 不重跑"},
        "zero_touch": {
            "protected_sha256_at_prereg": assets,
            "protected_sizes": sizes,
            "note": "run 段 import 前/结束+落盘后双测全部 PROTECTED sha, 任一漂移即拒跑/停写; "
                    "R117 全家与台账 NNS-067~071 只读"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": X.DEVICE, "threads": 1,
                        "interpreter": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"},
        "c0_anchor_expectation": c0_anchor,
        "b2_b3_presence_at_prereg": {
            "B2": os.path.isdir(X.B2_DIR), "B3": os.path.isdir(X.B3_DIR),
            "note": "时点陈述运行时实测 (R116/R117 教训); 对照列以 run 段 build 时实读为准"},
        "line_discipline": {
            "research_line_only": True, "zero_training": True,
            "forbidden_numbers": ["14.96", "16.08"],
            "readonly": ["models/", "data/", "E:/fracture_data/", "results/r117_recon_compare/",
                         "results/v_r117_prereg.json"]},
        "gates_process_only": {
            "G1": "PROTECTED sha import 前后双测零漂移",
            "G2": "C0 27/27 逐位 = R117 在盘 (P1)",
            "G3": "C1/C2 每场景×seed 装配锚点 head.sample sha = R117 落盘 (装配同构逐次证明)",
            "G4": "采样权重和守卫 max|Σπ−1| ≤ 1e-5 全通过",
            "G5": "三变体 × 9 场景 × 3 seeds 全落盘 allow_nan=False; 表含参照列或缺席声明",
            "G6": "预注册创建时间早于一切产物 (epoch + UTC+8 字符串双口径)",
            "discipline": "run 段只登记观察值不自授判级"},
        "design_time_probe_disclosure": (
            "预注册前接口探针 (不烧一枪, 不写 results/, 口径同 R117 先例): ①borehole_1 上 "
            "C0 锚点/装配同构锚/权重和守卫全部通过, BIC 曲线形态合理 (n<2K 保护生效); "
            "②full+station_30 快查: k_C1=k_C2=1 时 fisher 76.4°→~62° (部分恢复, 超 15° 阈), "
            "k_eff 仍 2–5 —— 证实泄漏还有一条『被保留原型自身 κ_i 逐点浓度失校准』通道, "
            "GREEN 判据 (≤15°) 按原则预写不改, RED 分支的分析价值已在预期中; 探针未导致任何"
            "规则/阈值/采样参数改动, 一枪结果以下方冻结规则跑出为准。"),
    }
    os.makedirs(os.path.dirname(PREREG), exist_ok=True)
    with open(PREREG, "w", encoding="utf-8") as f:
        json.dump(X._san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
    log(f"[prereg] -> {PREREG}")


# --------------------------------------------------------------------- run --

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
    log("PROTECTED import-前 sha 全对齐 (G1)")

    torch.set_num_threads(1)
    LEDGER_REL = "results/digital_ledger.json"
    def ledger_entry_guard():
        """活账 append-only 口径: R117 条目 NNS-067~069 在场且指针值与
        comparison_table.json 源头一致 (字节恒转不误杀, 篡改可检出)。"""
        led = load_json_strict(os.path.join(ROOT, LEDGER_REL))
        r117t = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
        cur = {e["id"]: e for e in led["entries"]}
        want = {
            "NNS-067": r117t["scenarios"]["full"]["metrics"]["set"][
                "fisher_mean_direction_distance_deg"]["fracgen_mean"],
            "NNS-068": r117t["scenarios"]["full"]["metrics"]["topology"][
                "intersection_density_rel_err"]["fracgen_mean"],
            "NNS-069": r117t["summary"]["by_baseline"]["B1"]["counts"]["fracgen_loss"],
        }
        for eid, wv in want.items():
            assert eid in cur, f"R117 台账条目丢失: {eid}"
            assert abs(float(cur[eid]["value"]) - round(float(wv), 4)) < 5e-5, eid
    ledger_entry_guard()
    truth = X.load_truth()
    model, cfg, step = X._load_model()
    os.makedirs(RUN_DIR, exist_ok=True)
    for arm in ARMS:
        os.makedirs(os.path.join(RUN_DIR, arm), exist_ok=True)

    r117_tab = load_json_strict(os.path.join(R117_DIR, "comparison_table.json"))
    anchor_map = prereg["c0_anchor_expectation"]
    results = {a: {} for a in ARMS}
    scen_meta = {}
    g2_ok, g3_ok, g4_ok = True, True, True

    for scen in SCENARIOS:
        faces_obs, m_obs, obs_pts = X.scenario_faces_obs(truth, scen)
        labels, means, props, _ = leader_set_clusters(faces_obs)
        k_c1 = int(len(means))
        xdir = _sign_align_z(np.array([f["n"] for f in faces_obs]))
        bic, lls, k_c2 = vmf_mix_bic_curve(xdir, kmax=K_RANGE[1])
        usage_probe = None
        meta = {"scenario": scen, "n_hat": len(faces_obs),
                "k_c1_leader": k_c1, "k_c2_bic": k_c2,
                "bic_curve": {str(i + 1): b for i, b in enumerate(bic)},
                "lnL_curve": {str(i + 1): l for i, l in enumerate(lls)}}
        scen_meta[scen] = meta
        per_seed = {a: [] for a in ARMS}

        for seed in SEEDS:
            gen, stream_sha = X.b1_stream_sha(faces_obs, truth["box"], seed)
            want_sha = anchor_map[scen][str(seed)]
            # ---- C0: 直接复用 R117 函数本体 ----
            q_dir0, diag0 = X.fracgen_orientations(model, cfg, faces_obs, gen, truth, scen, obs_pts, seed)
            sh0 = hashlib.sha256(np.ascontiguousarray(q_dir0, dtype=np.float64).tobytes()).hexdigest()
            if sh0 != want_sha:
                g2_ok = False
                log(f"  !! C0 换轨锚漂移 {scen}/{seed}")
            per_seed["C0"].append(dict(gen=gen, q_dir=q_dir0, seed=seed, keep=None))  # q_dir0 已是 _unit 结果, 不再二次归一
            # ---- C1 / C2 ----
            b, n_q = build_batch(faces_obs, gen, truth, scen, obs_pts)
            all_keep = list(range(12))
            dirs12, anchor_full, wdev12, usage_np = restricted_sample(model, b, all_keep, seed)
            ash = hashlib.sha256(np.ascontiguousarray(
                _unit(anchor_full[-n_q:]), dtype=np.float64).tobytes()).hexdigest()
            if ash != want_sha:
                g3_ok = False
                log(f"  !! C1/C2 装配锚点漂移 {scen}/{seed}")
            for arm, kk in (("C1", k_c1), ("C2", k_c2)):
                keep_idx, k_used = keep_protos_from_usage(usage_np, kk)
                dirs, _, wdev, _ = restricted_sample(model, b, keep_idx, seed)
                if wdev > 1e-5:
                    g4_ok = False
                per_seed[arm].append(dict(gen=gen, q_dir=_unit(dirs[-n_q:]), seed=seed,
                                          keep={"kept": keep_idx, "k_used": k_used,
                                                "usage_kept": [round(float(usage_np[i]), 4)
                                                               for i in keep_idx],
                                                "wsum_dev": wdev}))

        # 落盘三臂 × seed 评分
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
                             gen_dir_sha256=hashlib.sha256(
                                 np.ascontiguousarray(q_dir, dtype=np.float64).tobytes()).hexdigest(),
                             k_eff_leader=int(len(means_r)),
                             keep_protos=e["keep"], scorecard=X._san(sc))
                entries.append(entry)
            results[arm][scen] = entries

        # 组装 per-scenario JSON
        for arm in ARMS:
            j = {"arm": arm, "scenario": scen,
                 "observation_type": m_obs["observation_type"],
                 "coverage_fraction": round(float(m_obs["coverage_fraction"]), 6),
                 "n_hat": len(faces_obs), "k_c1_leader": k_c1, "k_c2_bic": k_c2,
                 "per_seed": results[arm][scen]}
            with open(os.path.join(RUN_DIR, arm, f"{scen}.json"), "w", encoding="utf-8") as f:
                json.dump(X._san(j), f, ensure_ascii=False, indent=2, allow_nan=False)
        log(f"{scen}: k_C1={k_c1} k_C2={k_c2} done")

    doc = {"gate_observations": {
        "G1_protected_sha_zero_drift": True,
        "G2_C0_bit_anchor_27of27": g2_ok,
        "G3_assembly_anchor_all_match": g3_ok,
        "G4_weight_sum_guard": g4_ok,
        "G6_prereg_before_outputs": creation_wall(PREREG)[1] <= min(
            os.stat(os.path.join(RUN_DIR, a, s + ".json")).st_ctime for a in ARMS for s in SCENARIOS),
        "note": "只登记客观观察值; 判级由守卫测试+架构师完成"},
        "scenario_k_selection": scen_meta,
        "b2_b3_presence_at_run": {"B2": os.path.isdir(X.B2_DIR), "B3": os.path.isdir(X.B3_DIR)}}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(X._san(doc), f, ensure_ascii=False, indent=2, allow_nan=False)

    # 对照表
    build_calib_table(prereg, r117_tab)
    ledger_entry_guard()
    sha_post = sha_set()
    bad = [p for p in sha_pre if sha_pre[p] != sha_post[p]
           and not p.endswith("digital_ledger.json")]
    assert not bad, f"只读件在单内被改动: {bad}"
    log("[run] PROTECTED sha 结束复测零漂移; done")


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
    tab = {"protocol": "v_r121_prereg/1.0", "prereg_sha256": X.sha256_file(PREREG),
           "seeds": SEEDS,
           "primary_thresholds": {"green_fisher_max": GREEN_FISHER_MAX,
                                  "keff_range": list(KEFF_RANGE),
                                  "single_set_scenarios": SINGLE_SET_SCENARIOS},
           "scenarios": {}, "summary": {"vs_C0_counts": {}, "primary_observations": {}},
           "reference_presence": {"R117_table": True, "B1": True,
                                  "B2": "B2" in arms_ref, "B3": "B3" in arms_ref}}
    counts = {"fracgen_win": 0, "fracgen_loss": 0, "within_2sigma": 0, "na": 0}
    for scen in SCENARIOS:
        rowm = {}
        for layer in LAYERS:
            rowm[layer] = {}
            for metric in METRICS[layer]:
                c0 = agg(data["C0"][scen], layer, metric)
                cell = {"C0": c0, "C1": agg(data["C1"][scen], layer, metric),
                        "C2": agg(data["C2"][scen], layer, metric), "refs": {}}
                r17 = r117_tab["scenarios"][scen]["metrics"][layer][metric]
                cell["refs"]["R117"] = {"mean": r17["fracgen_mean"], "std": r17["fracgen_std"],
                                        "n": r17.get("n_seeds", 3),
                                        "source": "results/r117_recon_compare/comparison_table.json (直读)"}
                for bn, bd in arms_ref.items():
                    cell["refs"][bn] = agg(bd[scen], layer, metric)
                for arm in ("C1", "C2"):
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
                cell["k_eff_leader"] = {a: [e["k_eff_leader"] for e in data[a][scen]] for a in ARMS}
                rowm[layer][metric] = cell
        tab["scenarios"][scen] = rowm
    tab["summary"]["vs_C0_counts"] = counts

    # 主断言观察值 (只登记)
    po = {}
    for arm in ("C1", "C2"):
        fisher = tab["scenarios"]["full"]["set"]["fisher_mean_direction_distance_deg"]
        c0, va = fisher["C0"], fisher[arm]
        band = 2 * max(c0["std"] or 0.0, 1e-9)
        keff_ok = all(1 <= k <= 3 for s in SINGLE_SET_SCENARIOS
                      for k in tab["scenarios"][s]["set"]["fisher_mean_direction_distance_deg"]["k_eff_leader"][arm])
        po[arm] = {
            "full_fisher_mean": va["mean"],
            "threshold_max": GREEN_FISHER_MAX,
            "below_threshold": (va["mean"] is not None and va["mean"] <= GREEN_FISHER_MAX),
            "beyond_C0_band": (va["mean"] is not None and c0["mean"] is not None
                               and va["mean"] < c0["mean"] - band),
            "keff_in_range_all_seeds": bool(keff_ok),
            "observation_GREEN_all_three": bool((va["mean"] is not None and va["mean"] <= GREEN_FISHER_MAX)
                                                and (va["mean"] is not None and c0["mean"] is not None
                                                     and va["mean"] < c0["mean"] - band) and keff_ok)}
        # 因果预期观察: interdens 回摆
        idm = tab["scenarios"]["full"]["topology"]["intersection_density_rel_err"]
        po[arm]["causal_full_interdens"] = {"C0": idm["C0"]["mean"], arm: idm[arm]["mean"],
                                            "B1_ref": idm["refs"]["B1"]["mean"]}
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
        build_calib_table(prereg, load_json_strict(os.path.join(R117_DIR, "comparison_table.json")))
    else:
        stage_run()


if __name__ == "__main__":
    main()
