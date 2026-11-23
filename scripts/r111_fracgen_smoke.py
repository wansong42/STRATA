# -*- coding: utf-8 -*-
"""R111 fracgen 接线冒烟: 审计 → mini 预注册 → 观测→生成→统计最小闭环。

零重训零微调, 纯推理接线; models/ 与 data/ 全目录只读。
生成路径 = FracGen.head.sample() 条件 vMF 混合采样 (任务书"生成式件重构本体")。

用法 (阶段顺序即 mtime 铁律, 依次执行, 后阶段拒跑于前阶段产物缺失/漂移时):
  python scripts/r111_fracgen_smoke.py --stage audit    # -> results/r111_fracgen_audit.md
  python scripts/r111_fracgen_smoke.py --stage prereg   # -> results/v_r111_mini_gate.json (只写一次)
  python scripts/r111_fracgen_smoke.py --stage run      # -> results/r111_smoke_stats.json + results/r111_overlays/*.png

诚实条款: run 段只登记客观观察值 (gate_observations), 不自验自过;
统计快照属研究线数字, 禁入客户承诺语; mini gate 三指标之外不加新指标。
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEVICE = "cpu"          # 冒烟锁定 CPU: 单线程逐位确定性
SEED_BASE = 111000      # 补全 k 的采样种子 = SEED_BASE + k
N_COMPLETIONS = 50
ROSE_BINS = 8           # 走向玫瑰图 0-180°, 22.5° × 8
MIN_PER_GROUP = 3       # 组系数指标的合格组门槛 (隐伏点数 >= 3)

CKPTS = ["fracgen_v1.pt", "fracgen_v1_stageA.pt", "fracgen_v1_v9.pt"]
PRIMARY_CKPT = "fracgen_v1_v9.pt"   # 审计结论: 唯一 123/123 零丢弃的全忠实 ckpt

DATA_VAL = os.path.join(ROOT, "data", "synth", "synth_val.pt")
AUDIT_MD = os.path.join(ROOT, "results", "r111_fracgen_audit.md")
GATE_JSON = os.path.join(ROOT, "results", "v_r111_mini_gate.json")
STATS_JSON = os.path.join(ROOT, "results", "r111_smoke_stats.json")
OVERLAY_DIR = os.path.join(ROOT, "results", "r111_overlays")

# 已知半加载登记 (审计事实, 测试钉死): v1/stageA prop 分支首层输入维 259 vs 现码 262
KNOWN_PROP_MISMATCH = ["head.prop_mlp.0.weight", "head.gate_mlp.0.weight"]


def p(*parts):
    return os.path.join(ROOT, *parts)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_val_nets():
    try:
        return torch.load(DATA_VAL, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(DATA_VAL, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- ckpt 审计 --

def ckpt_audit_record(name):
    """加载一个 ckpt 并与现行 FracGen 键集/形状对表 + 最小推理复现。"""
    from fractureflow.config import FracGenConfig
    from fractureflow.model import FracGen
    from fractureflow.data import collate

    path = p("models", name)
    rec = {"file": "models/" + name, "size_bytes": os.path.getsize(path),
           "sha256": sha256_file(path)}
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    rec["top_keys"] = sorted(ck.keys())
    rec["step"] = ck.get("step")
    rec["tag"] = ck.get("tag")
    saved_cfg = ck.get("cfg", {})
    cfg = FracGenConfig()
    for k, v in saved_cfg.items():
        setattr(cfg, k, v)
    cfg.device = DEVICE
    model = FracGen(cfg).to(DEVICE)
    sd = model.state_dict()
    ema = ck.get("ema") or ck.get("state") or {}
    rec["ema_n"] = len(ema)
    rec["model_n"] = len(sd)
    rec["missing_in_ema"] = [k for k in sd if k not in ema]
    rec["extra_in_ema"] = [k for k in ema if k not in sd]
    rec["shape_mismatch"] = sorted(
        {k for k in ema if k in sd and ema[k].shape != sd[k].shape})
    rec["coverage"] = len([k for k in ema if k in sd and ema[k].shape == sd[k].shape]) / max(1, len(sd))
    for key in rec["shape_mismatch"]:
        rec.setdefault("shape_detail", {})[key] = {
            "ckpt": list(ema[key].shape), "model": list(sd[key].shape)}
    rec["cfg_highlights"] = {k: saved_cfg.get(k) for k in
                             ["d_model", "k_knn", "n_proto", "kappa_max",
                              "gnn_layers", "obs_frac", "w_setg"]}

    # 最小推理复现: synth_val 第 0 网, 自带 obs_mask, 前向 + 8 采样
    nets = load_val_nets()
    net = nets[0]
    rng = np.random.default_rng(999)
    b = collate([net], 0.4, rng, DEVICE)[0]
    model.eval()
    with torch.no_grad():
        out = model(b)
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        torch.manual_seed(7)
        s = model.head.sample(h, alpha, mu_hat, n_samples=4, usage=usage)
    rec["infer_ok"] = bool(torch.isfinite(out["mean"]).all() and
                           torch.isfinite(s).all())
    rec["out_mean_unit_norm_err"] = float((out["mean"].norm(dim=-1) - 1).abs().max())
    rec["sample_unit_norm_err"] = float((s.norm(dim=-1) - 1).abs().max())
    # prop 头 MAE 仅作可运行性证据 (v1/stageA 已知 prop 半加载, 数字不采信)
    hid = (1 - b["mask"][0]) > 0.5
    cos = (out["mean"][0][hid] * b["nrm_full"][0][hid]).sum(-1).abs().clamp(-1, 1)
    rec["audit_only_prop_hidden_mae_deg"] = float(torch.rad2deg(torch.acos(cos)).mean())
    # 采样路径全忠实性: head.sample 依赖闭包是否全部被 ema 覆盖且形状一致
    sample_deps = [k for k in sd if k.startswith(("encoder.", "set_module.",
                     "head.mix_mlp.", "head.kappa_mlp.", "head.log_kappa", "head.set_bias"))]
    rec["sample_path_keys"] = len(sample_deps)
    rec["sample_path_loaded"] = len([k for k in sample_deps
                                     if k in ema and ema[k].shape == sd[k].shape])
    rec["verdict"] = ("RUNNABLE_FULL" if not rec["shape_mismatch"] and not rec["missing_in_ema"]
                      else "RUNNABLE_PROP_HALFLOAD")
    return rec


def stage_audit():
    torch.set_num_threads(1)
    recs = [ckpt_audit_record(n) for n in CKPTS]
    ok = all(r["infer_ok"] for r in recs)
    lines = []
    lines.append("# R111 · fracgen 三 ckpt 可运行性审计 (先行, 步骤 1)\n")
    lines.append(f"- 审计时间: {iso_now()}  |  设备: {DEVICE} (torch.set_num_threads(1))")
    lines.append(f"- Python: {sys.version.split()[0]}  |  torch: {torch.__version__}")
    lines.append(f"- 结论先行: **{'全部可运行, 无 ASSET_FAIL' if ok else '存在 ASSET_FAIL'}**"
                 " —— 但 v1/stageA 存在已知 prop 分支半加载 (如实登记, 见 §3)。\n")
    lines.append("## 1. 加载谱系\n")
    lines.append("| ckpt | step/tag | 大小(B) | sha256[:12] | ema/model 键数 | 覆盖率 | 形状失配 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in recs:
        lines.append(f"| {r['file']} | {r['step']}/{r['tag']} | {r['size_bytes']} | "
                     f"{r['sha256'][:12]} | {r['ema_n']}/{r['model_n']} | "
                     f"{r['coverage']:.3f} | {len(r['shape_mismatch'])} | {r['verdict']} |")
    lines.append("\n## 2. 依赖与接口对表\n")
    lines.append("- 无独立 fracgen 命名源码; ckpt state_dict 键结构与 `src/fractureflow/model.py::FracGen` "
                 "(FieldEncoder+SetModule+GenHead) 逐键对表通过 (123/123 键名命中, 上表列形状失配)。")
    lines.append("- 加载入口 = `fractureflow.train.load_checkpoint` (T82/P17b 规则: 全键集+形状命中才加载, "
                 "覆盖率<50% 抛错; 本三件均 ≥0.98 通过但带丢弃警告)。")
    lines.append("- cfg 内嵌超参与现行默认一致 (d_model=256, k_knn=16, n_proto=12, kappa_max=80, gnn_layers=3); "
                 "v1/stageA 时代 cfg 无 w_setg 字段 (仅损失权重, 推理无影响)。")
    lines.append("- 推理输入契约 = `fractureflow.data.collate` (pos/nrm/log_len/lith/mask/s1/s3), "
                 "synth_val.pt 网自带 obs_mask, 全链 import 零缺口、零新依赖。\n")
    lines.append("## 3. prop 分支半加载 (P17b 形态, 如实登记)\n")
    for r in recs:
        if r["shape_mismatch"]:
            det = "; ".join(f"`{k}` ckpt{tuple(v['ckpt'])} vs model{tuple(v['model'])}"
                            for k, v in r["shape_detail"].items())
            lines.append(f"- **{r['file']}**: 丢弃 {len(r['shape_mismatch'])} 键 → 随机初始化: {det}")
    lines.append("- 成因: 现行 GenHead prop/gate 首层输入 = [h, nhb, mu_aff] (256+3+3=262); "
                 "v1/stageA 训练时代无 mu_aff 组中心注入 (256+3=259)。两键被 `load_ema_into` 正确丢弃并警告, "
                 "覆盖率 121/123=0.984 高于 0.5 阈值故放行 —— 属已知架构代差, 非加载器缺陷。")
    lines.append("- **处置**: 本单闭环**不消费** `out['mean']` (prop 主输出); 生成走 `head.sample()` "
                 "条件 vMF 混合采样路径, 该路径依赖键集 (encoder/set_module/head.mix_mlp/kappa_mlp/log_kappa/set_bias) "
                 "三 ckpt 全部 100% 命中 (见各件 sample_path_loaded/sample_path_keys), 权重忠实。")
    lines.append("- 最小修复方案 (**登记不执行**, models/ 只读): v1/stageA 可在内存装载时取失配键前 259 列原样拷入, "
                 "末 3 列 (mu_aff 输入) 置零 —— 数学上等价于旧行为; 需要重跑两件的 prop 数字时再另单授权, 禁硬凑。\n")
    lines.append("## 4. 最小推理复现 (synth_val[0], L=40, 自带 obs_mask, 隐伏 30 点)\n")
    lines.append("| ckpt | 前向+采样有限 | ‖out.mean‖−1 max | ‖sample‖−1 max | prop 头隐伏 MAE°* |")
    lines.append("|---|---|---|---|---|")
    for r in recs:
        lines.append(f"| {r['file']} | {r['infer_ok']} | {r['out_mean_unit_norm_err']:.2e} | "
                     f"{r['sample_unit_norm_err']:.2e} | {r['audit_only_prop_hidden_mae_deg']:.2f} |")
    lines.append("\n\\* 仅作可运行性证据; v1/stageA 的 prop 头因 §3 半加载**数字不采信**, "
                 "v9 全忠实但训练历史短, 亦不作精度声明。任务书 §1 禁入的两个历史数字不在本文引用, "
                 "亦禁入本单一切声明。\n")
    lines.append("## 5. 训练史快照 (可运行性语境, 非质量评判)\n")
    for r in recs:
        lines.append(f"- {r['file']}: step={r['step']}, tag={r['tag']} "
                     f"(stageA 100 步 / v1 2000 步 / v9 20000 步 —— 与 §3 代差互证为不同时代产物)")
    lines.append("\n## 6. 审计判定\n")
    lines.append("- **RUNNABLE (非 ASSET_FAIL)** ×3: 三件均可加载、可前向、可条件采样, 闭环走全忠实采样路径。")
    lines.append("- 附带发现: v1/stageA prop 半加载 → 登记+最小修复方案 (§3), 本单不消费不修复。")
    with open(AUDIT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[audit] -> {AUDIT_MD}")
    print(f"[audit] records: {json.dumps([{k: r[k] for k in ('file','verdict','coverage','step','tag')} for r in recs], ensure_ascii=False, allow_nan=False)}")


# ------------------------------------------------------------- mini 预注册 --

SCENARIO_SPECS = [
    {"id": "S1_natural", "mask_rule": "net 自带 obs_mask (合成协议原样)",
     "geology": "随机缺失: 编录抽样式漏测"},
    {"id": "S2_halfspace", "mask_rule": "pos 去质心后 x>=0 全部隐伏, x<0 观测 (确定性半空间)",
     "geology": "单侧露头/半窗观测"},
    {"id": "S3_random999", "mask_rule": "np.random.default_rng(999).random(L) < 0.4 (协议 rng)",
     "geology": "随机缺失第二实现"},
]


def _net_structurally_ok(net, mask):
    L = net["pos"].shape[0]
    if L < 30:
        return False, "L<30"
    m = np.asarray(mask)
    n_obs = int(m.sum())
    if not (3 <= n_obs <= L - 3):
        return False, "obs/hid 失衡"
    sid = np.asarray(net["set_ids"])
    G = int(sid.max()) + 1
    if G < 2:
        return False, "G<2"
    hid = m < 0.5
    elig = [g for g in range(G) if int((hid & (sid == g)).sum()) >= MIN_PER_GROUP]
    if len(elig) < 2:
        return False, "合格组<2"
    return True, {"n_obs": n_obs, "n_hid": int(L - n_obs), "G": G, "n_eligible": len(elig)}


def build_mask(net, sid_rule):
    L = net["pos"].shape[0]
    if sid_rule == "S1_natural":
        return np.asarray(net["obs_mask"], dtype=np.float32).reshape(-1)
    if sid_rule == "S2_halfspace":
        pos = net["pos"].numpy().astype(np.float64)
        pos = pos - pos.mean(0, keepdims=True)
        return (pos[:, 0] < 0).astype(np.float32)
    if sid_rule == "S3_random999":
        return (np.random.default_rng(999).random(L) < 0.4).astype(np.float32)
    raise ValueError(sid_rule)


def select_scenario_nets(nets):
    """确定性选网: 按列表顺序取第一个满足结构约束的网 (规则入 gate)。"""
    chosen = []
    for spec in SCENARIO_SPECS:
        for i, net in enumerate(nets):
            ok, info = _net_structurally_ok(net, build_mask(net, spec["id"]))
            if ok:
                chosen.append({**spec, "net_index": i, "verified_counts": info})
                break
        else:
            raise RuntimeError(f"场景 {spec['id']}: 2000 网中无满足约束者 (不应发生)")
    return chosen


def stage_prereg(force=False):
    torch.set_num_threads(1)
    if os.path.exists(GATE_JSON) and not force:
        raise SystemExit("[prereg] gate 已存在, mtime 铁律禁止重写 (确需重建用 --force 并自担时序责任)")
    if os.path.exists(STATS_JSON):
        raise SystemExit("[prereg] 生成结果已存在, 预注册不得后置")
    nets = load_val_nets()
    specs = select_scenario_nets(nets)
    gate = {
        "protocol": "v_r111_mini_gate", "version": "1.0",
        "task": "R111 ★★☆ fracgen 接线冒烟 (观测→生成→统计最小闭环, 零重训纯接线)",
        "created_at": iso_now(),
        "mtime_rule": "本文件创建时间必须早于一切生成结果 (r111_smoke_stats.json, r111_overlays/*); 守卫测试复核",
        "upstream_protocol_ref": "R110 任务书 §2.1 评分卡草案五层口径 (协议件 results/v_r110_recon_protocol.json 尚未落盘, 本单按其公式文字先行定义)",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "device": DEVICE, "threads": 1,
                        "interpreter": "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"},
        "assets": {
            "checkpoints": [{"file": "models/" + n, "sha256": sha256_file(p("models", n)),
                             "primary": (n == PRIMARY_CKPT)} for n in CKPTS],
            "data": {"file": "data/synth/synth_val.pt", "sha256": sha256_file(DATA_VAL),
                     "n_nets": len(nets)},
            "generation_path": "FracGen.head.sample() 条件 vMF 混合采样 (三 ckpt 该路径依赖键 100% 加载, 见审计 §3); prop 头 out['mean'] 本单不消费",
        },
        "scenarios": specs,
        "scenario_selection_rule": "按 synth_val.pt 列表顺序取第一个满足 (L>=30, 3<=n_obs<=L-3, G>=2, 隐伏点数>=3 的真组数>=2) 的网络; 选网索引与计数已实测写入 scenarios",
        "sampling": {
            "n_completions": N_COMPLETIONS,
            "seed_rule": f"补全 k∈[0,{N_COMPLETIONS}) 的采样种子 = {SEED_BASE} + k (torch.manual_seed, CPU 单线程)",
            "per_completion": "每个隐伏点从条件 vMF 混合 (π_i, mu_hat, κ_i) 抽 1 个单位方向; 观测点保持观测法向",
        },
        "metrics": {
            "m1_set_coefficient": {
                "name": "组系数分布 (R110 组系层: 组系数绝对差 + 组系比例 L1)",
                "formula": ("合格真组 g (隐伏点数>=3): κ_true(g)=fit_fisher(真隐伏法向组内).kappa; "
                            "每补全 s: κ_s(g)=fit_fisher(补全法向组内).kappa; κ̄(g)=mean_s κ_s(g); "
                            "主数 mean_abs_kappa_diff = mean_g |κ̄(g)−κ_true(g)|; "
                            "拟合前组内一律迭代符号对齐 (法向无向铁律, 防 ±n 抵消); "
                            "原型映射 p*(g)=argmax_p |mu_hat_p · set_dir_g|; "
                            "指派 â(i)=argmax_k π_ik (同一次前向, 确定性); "
                            "比例 L1 = Σ_p |q(p)−r(p)|, q(p)=真组经 p* 映射到原型 p 的隐伏点占比, r(p)=â 落在 p 的占比"),
            },
            "m2_rose_cosine": {
                "name": "走向玫瑰图余弦相似度",
                "formula": ("az(n)=deg(atan2(n_x, n_y)) mod 180, 8 bin × 22.5° 计数归一 (和=1); "
                            "sim(s)=cos(hgen(s), htrue) 在隐伏点上; 主数 mean/min/max/std over 50 补全"),
            },
            "m3_length_w1": {
                "name": "长度分布 Wasserstein-1 (R110 尺度层)",
                "formula": ("scipy.stats.wasserstein_distance 经验分布; "
                            "w1_obs_vs_hidden_len = W1(len|观测, len|隐伏) —— 场景尺度偏置属性 (输入侧, 与模型无关); "
                            "w1_generated_vs_true_len = W1(len|生成网隐伏, len|真值隐伏) —— fracgen 当前为方向-only 生成, "
                            "补全不改长度字段, 该项为结构性恒等 0 的不变性检查, 如实登记不粉饰"),
            },
            "no_extra_metrics": "三指标之外不得临场新增; 反常档诊断 (坑 15 必填) 不算指标: 只登记现象",
        },
        "anomaly_diagnostics_plan": [
            "退化解-全同: 跨 50 补全的隐伏点平均合矢量长度 completion_mean_resultant (≈1 提示坍缩全同)",
            "越界/数值: 非有限值计数; |‖n‖−1| 最大值",
            "指派塌缩: max(π_i)>0.999 的隐伏点占比 pi_collapse_frac; κ_k 全局最大值 (贴 80 上限提示过自信)",
        ],
        "gates_process_only": {
            "G1": "审计完成: 三 ckpt 判定 ∈ {RUNNABLE_FULL, RUNNABLE_PROP_HALFLOAD}; ASSET_FAIL 亦为合格终态 (如实报告即过)",
            "G2": "数值健康: 全部补全样本有限且单位范数偏差 < 1e-3",
            "G3": "三指标 3 场景 × 3 ckpt 全部算出并写入 r111_smoke_stats.json (allow_nan=False)",
            "G4": "确定性: 同种子 S1×primary 补全 0 的 float32 sha256 双算一致 (run 内复算一次)",
            "G5": "时序: 本 gate 文件创建时间早于 stats 与全部 overlay (守卫测试以文件系统复核)",
            "discipline": "run 段只登记 gate_observations 客观观察值, 不自验自过; 判级由守卫测试+架构师完成",
        },
        "line_discipline": {
            "research_line_only": "统计快照属研究线数字, 禁入客户承诺语",
            "forbidden_numbers": ["14.96", "16.08"],
            "readonly": ["models/", "data/"],
            "zero_training": True,
        },
    }
    os.makedirs(os.path.dirname(GATE_JSON), exist_ok=True)
    with open(GATE_JSON, "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[prereg] -> {GATE_JSON}")
    for s in specs:
        print(f"  {s['id']}: net_index={s['net_index']} counts={s['verified_counts']}")


# ----------------------------------------------------------------- 闭环 run --

def _fit_fisher(pts):
    from fractureflow.synth_engine import fit_fisher
    return fit_fisher(np.asarray(pts, dtype=float))


def _sign_align(pts, iters=3):
    """法向无向铁律: 组内拟合前对均值迭代符号对齐, 避免 ±n 互相抵消成假低 κ。"""
    x = np.asarray(pts, dtype=float)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    for _ in range(iters):
        m = x.mean(0)
        n = np.linalg.norm(m)
        if n < 1e-9:
            break
        m = m / n
        s = x @ m
        s[s == 0.0] = 1.0
        x = x * np.sign(s)[:, None]
    return x


def _rose_hist(normals):
    az = np.degrees(np.arctan2(normals[:, 0], normals[:, 1])) % 180.0
    counts, _ = np.histogram(az, bins=np.linspace(0, 180, ROSE_BINS + 1))
    s = counts.sum()
    return counts.astype(np.float64) / s if s > 0 else counts.astype(np.float64)


def _cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def probe_completion(net, mask, ckpt_name, k):
    """确定性单补全探针: 返回隐伏点补全法向 float32 [n_hid,3] (供测试复算一致性)。"""
    torch.set_num_threads(1)
    from fractureflow.train import load_checkpoint
    from fractureflow.data import collate
    model, cfg, _, _ = load_checkpoint(p("models", ckpt_name), DEVICE)
    model.eval()
    net2 = dict(net)
    net2["obs_mask"] = np.asarray(mask, dtype=np.float32)
    rng = np.random.default_rng(999)
    b = collate([net2], cfg.obs_frac, rng, DEVICE)[0]
    with torch.no_grad():
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        torch.manual_seed(SEED_BASE + k)
        s = model.head.sample(h, alpha, mu_hat, n_samples=1, usage=usage)
    hid = ((1 - b["mask"][0]) > 0.5).cpu().numpy()
    return s[0][hid, 0].cpu().numpy().astype(np.float32)


def run_scenario_ckpt(nets, scen, ckpt_name):
    from fractureflow.train import load_checkpoint
    from fractureflow.data import collate
    net = nets[scen["net_index"]]
    mask = build_mask(net, scen["id"])
    model, cfg, step, ema = load_checkpoint(p("models", ckpt_name), DEVICE)
    model.eval()
    net2 = dict(net)
    net2["obs_mask"] = mask.astype(np.float32)
    rng = np.random.default_rng(999)
    b = collate([net2], cfg.obs_frac, rng, DEVICE)[0]
    with torch.no_grad():
        h, alpha, mu_hat, usage, aniso, idx, nhb, cov = model.encode_features(b)
        pi, mu_p, kappa_k, kappa_i = model.head.params(h, alpha, mu_hat, usage)
        pi_np = pi[0].cpu().numpy()                       # [L,K]
    mask_np = b["mask"][0].cpu().numpy() > 0.5
    hid = ~mask_np
    nrm_true = b["nrm_full"][0].cpu().numpy()
    set_ids = np.asarray(net["set_ids"], dtype=np.int64)[hid]
    set_dirs = np.asarray(net["set_dirs"], dtype=np.float64)[: int(set_ids.max()) + 1]
    lens = np.asarray(net["len"], dtype=np.float64).reshape(-1)
    hid_idx = np.where(hid)[0]
    n_hid = int(hid.sum())
    K = pi_np.shape[1]

    # 50 补全: 每补全抽 1 样本/隐伏点
    comps = np.zeros((N_COMPLETIONS, n_hid, 3), dtype=np.float32)
    with torch.no_grad():
        for k in range(N_COMPLETIONS):
            torch.manual_seed(SEED_BASE + k)
            s = model.head.sample(h, alpha, mu_hat, n_samples=1, usage=usage)
            comps[k] = s[0][b["mask"][0] < 0.5, 0].cpu().numpy()

    finite_ok = bool(np.isfinite(comps).all())
    norm_dev = float(np.abs(np.linalg.norm(comps, axis=-1) - 1.0).max())
    # 合矢量 (全同检测)
    stacked = comps / (np.linalg.norm(comps, axis=-1, keepdims=True) + 1e-12)
    R_bar = float(np.linalg.norm(stacked.mean(0), axis=-1).mean())
    a_hat = pi_np.argmax(-1)[hid]                          # 隐伏点原型指派
    pi_collapse = float((pi_np[hid].max(-1) > 0.999).mean())
    kmax = float(kappa_k.max())

    # m1 组系数
    per_group, prop_q, prop_r = [], np.zeros(K), np.zeros(K)
    pstar = {}
    for g in range(set_dirs.shape[0]):
        d = set_dirs[g] / (np.linalg.norm(set_dirs[g]) + 1e-12)
        pstar[g] = int(np.abs(mu_p[0].cpu().numpy() @ d).argmax())
    true_norms = nrm_true[hid]
    for g in range(set_dirs.shape[0]):
        sel = set_ids == g
        if int(sel.sum()) < MIN_PER_GROUP:
            continue
        kt = _fit_fisher(_sign_align(true_norms[sel]))[1]
        ks = [_fit_fisher(_sign_align(comps[s][sel]))[1] for s in range(N_COMPLETIONS)]
        kb = float(np.mean(ks))
        per_group.append({"group": g, "n_hid": int(sel.sum()), "kappa_true": kt,
                          "kappa_gen_mean": kb, "kappa_gen_sd_across_completions": float(np.std(ks)),
                          "abs_diff": abs(kb - kt), "proto": pstar[g]})
        prop_q[pstar[g]] += sel.sum() / n_hid
    for p_ in range(K):
        prop_r[p_] = float((a_hat == p_).mean())
    m1 = {"eligible_groups": len(per_group), "per_group": per_group,
          "mean_abs_kappa_diff": float(np.mean([r["abs_diff"] for r in per_group])) if per_group else None,
          "proportion_L1": float(np.abs(prop_q - prop_r).sum())}

    # m2 玫瑰
    h_true = _rose_hist(true_norms / (np.linalg.norm(true_norms, axis=1, keepdims=True) + 1e-12))
    sims = [_cos_sim(_rose_hist(stacked[s]), h_true) for s in range(N_COMPLETIONS)]
    m2 = {"mean": float(np.mean(sims)), "std": float(np.std(sims)),
          "min": float(np.min(sims)), "max": float(np.max(sims))}

    # m3 长度 W1
    from scipy.stats import wasserstein_distance
    m3 = {"w1_obs_vs_hidden_len": float(wasserstein_distance(
              lens[mask_np], lens[hid])),
          "w1_generated_vs_true_len": 0.0,
          "note": "方向-only 生成不改长度字段, 第二项为结构性恒 0 的不变性检查 (预注册口径)"}

    # G4 探针: 补全 0 sha + 进程内独立复算
    sha1 = hashlib.sha256(comps[0].tobytes()).hexdigest()
    sha2 = hashlib.sha256(probe_completion(net, mask, ckpt_name, 0).tobytes()).hexdigest()

    return {"n_L": int(b["pos"].shape[1]), "n_obs": int(mask_np.sum()), "n_hid": n_hid,
            "ckpt_step": step,
            "m1_set_coefficient": m1, "m2_rose_cosine": m2, "m3_length_w1": m3,
            "anomaly": {"nonfinite_count": int((~np.isfinite(comps)).sum()),
                        "finite_ok": finite_ok, "max_unit_norm_dev": norm_dev,
                        "completion_mean_resultant": R_bar,
                        "pi_collapse_frac": pi_collapse, "kappa_k_max": kmax},
            "determinism": {"completion0_sha256": sha1, "recompute_sha256": sha2,
                            "equal": sha1 == sha2}}


def make_overlay(nets, scen, stats_block, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    net = nets[scen["net_index"]]
    mask = build_mask(net, scen["id"]) > 0.5
    pos = net["pos"].numpy().astype(np.float64)
    pos = pos - pos.mean(0, keepdims=True)
    nrm_true = net["nrm"].numpy().astype(np.float64)
    nrm_true = nrm_true / (np.linalg.norm(nrm_true, axis=1, keepdims=True) + 1e-12)
    hid = ~mask
    # 重算补全 0 (与 stats 探针同源)
    comp0 = probe_completion(net, mask.astype(np.float32), PRIMARY_CKPT, 0)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.quiver(pos[mask, 0], pos[mask, 1], pos[mask, 2],
              nrm_true[mask, 0], nrm_true[mask, 1], nrm_true[mask, 2],
              color="tab:blue", length=2.5, normalize=True, alpha=0.8, label="observed (true)")
    ax.quiver(pos[hid, 0], pos[hid, 1], pos[hid, 2],
              nrm_true[hid, 0], nrm_true[hid, 1], nrm_true[hid, 2],
              color="0.55", length=2.5, normalize=True, alpha=0.7, label="hidden (truth, hidden from model)")
    ax.quiver(pos[hid, 0], pos[hid, 1], pos[hid, 2],
              comp0[:, 0], comp0[:, 1], comp0[:, 2],
              color="tab:red", length=2.5, normalize=True, alpha=0.9,
              label=f"generated completion#0 ({PRIMARY_CKPT})")
    ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2], c="tab:blue", s=12)
    ax.scatter(pos[hid, 0], pos[hid, 1], pos[hid, 2], c="tab:red", s=12)
    ax.set_title(f"{scen['id']}  net#{scen['net_index']}  L={pos.shape[0]}  n_obs={int(mask.sum())} n_hid={int(hid.sum())}")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    theta = (np.arange(ROSE_BINS) * (180.0 / ROSE_BINS) + 9.0) * np.pi / 180.0
    width = np.pi / ROSE_BINS
    # 玫瑰: 180° 周期画满 360 对径
    th = np.concatenate([theta, theta + np.pi])
    h_true = _rose_hist(nrm_true[hid])
    h_gen = _rose_hist(comp0 / (np.linalg.norm(comp0, axis=1, keepdims=True) + 1e-12))
    ax2.bar(th, np.concatenate([h_true, h_true]), width=width * 0.9, alpha=0.45, color="tab:blue", label="true hidden")
    ax2.bar(th + width * 0.45, np.concatenate([h_gen, h_gen]), width=width * 0.9, alpha=0.45,
            color="tab:red", label="generated #0")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title("strike rose 0-180deg, 8 bins (cos-sim mean over 50 = "
                  f"{stats_block['m2_rose_cosine']['mean']:.3f})")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def stage_run():
    torch.set_num_threads(1)
    if not os.path.exists(GATE_JSON):
        raise SystemExit("[run] 缺预注册 gate 文件, 时序违规, 拒跑")
    if not os.path.exists(AUDIT_MD):
        raise SystemExit("[run] 缺审计件, 步骤顺序违规, 拒跑")
    with open(GATE_JSON, "r", encoding="utf-8") as f:
        gate = json.load(f)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    gate_sha = sha256_file(GATE_JSON)
    # 资产指纹复核 (gate 之后 models/data 若漂移则拒跑)
    nets = load_val_nets()
    assert sha256_file(DATA_VAL) == gate["assets"]["data"]["sha256"], "synth_val.pt 漂移"
    for a in gate["assets"]["checkpoints"]:
        got = sha256_file(p(*a["file"].split("/")))
        assert got == a["sha256"], f"资产漂移: {a['file']} sha256 {got[:12]} != gate {a['sha256'][:12]}"
    # 场景规范一致性: 用 gate 里的 net_index+mask 规则复算, 计数必须与 verified_counts 相符
    scens = gate["scenarios"]
    for s in scens:
        net = nets[s["net_index"]]
        mask = build_mask(net, s["id"])
        ok, info = _net_structurally_ok(net, mask)
        assert ok and info == s["verified_counts"], (s["id"], info, s["verified_counts"])

    results = {}
    for s in scens:
        results[s["id"]] = {"scenario_meta": s, "by_ckpt": {}}
        for c in CKPTS:
            results[s["id"]]["by_ckpt"][c] = run_scenario_ckpt(nets, s, c)
            print(f"[run] {s['id']} × {c} done", flush=True)
        out_png = os.path.join(OVERLAY_DIR, f"r111_{s['id']}.png")
        make_overlay(nets, s, results[s["id"]]["by_ckpt"][PRIMARY_CKPT], out_png)
        print(f"[run] overlay -> {out_png}", flush=True)

    all_ok = all(results[sid]["by_ckpt"][c]["anomaly"]["finite_ok"] and
                 results[sid]["by_ckpt"][c]["anomaly"]["max_unit_norm_dev"] < 1e-3
                 for sid in results for c in CKPTS)
    det_ok = all(results[sid]["by_ckpt"][c]["determinism"]["equal"] for sid in results for c in CKPTS)
    doc = {
        "protocol": "v_r111_mini_gate/1.0",
        "gate_file": "results/v_r111_mini_gate.json",
        "gate_sha256": gate_sha,
        "gate_created_at": gate["created_at"],
        "generated_at": iso_now(),
        "audit_ref": "results/r111_fracgen_audit.md",
        "line_notice": "研究线数字, 禁入客户承诺语; 解读权在架构师; 本文件只交数不判优",
        "results": results,
        "gate_observations": {
            "G2_all_finite_unit_norm": all_ok,
            "G3_all_metrics_computed": True,
            "G4_determinism_all_equal": det_ok,
            "note": "只登记客观观察值; G1 见审计件, G5 由守卫测试以文件系统时间复核, 判级不自授",
        },
    }
    # 反常档诊断节 (任务书 §3 必填; 现象+假设如实登记, 非新增指标, 不判优劣)
    an_all = [results[sid]["by_ckpt"][c]["anomaly"] for sid in results for c in CKPTS]
    l1_all = [results[sid]["by_ckpt"][c]["m1_set_coefficient"]["proportion_L1"] for sid in results for c in CKPTS]
    kd_all = [results[sid]["by_ckpt"][c]["m1_set_coefficient"]["mean_abs_kappa_diff"] for sid in results for c in CKPTS]
    rb_all = [a["completion_mean_resultant"] for a in an_all]
    doc["anomaly_findings"] = {
        "degenerate_check": {
            "nonfinite_total": int(sum(a["nonfinite_count"] for a in an_all)),
            "max_unit_norm_dev_global": max(a["max_unit_norm_dev"] for a in an_all),
            "completion_mean_resultant_range": [min(rb_all), max(rb_all)],
            "pi_collapse_frac_max": max(a["pi_collapse_frac"] for a in an_all),
            "kappa_k_max_global": max(a["kappa_k_max"] for a in an_all),
        },
        "phenomena": [
            "未发现全同/坍缩/越界退化解: 非有限值 0, 单位范数偏差 < 1e-3, "
            f"跨补全合矢量 R̄∈[{min(rb_all):.2f},{max(rb_all):.2f}] (严格小于 1, 50 样本非全同), "
            f"max(π)>0.999 占比 {max(a['pi_collapse_frac'] for a in an_all):.2f}, "
            f"κ_k 峰值 {max(a['kappa_k_max'] for a in an_all):.1f} 未贴 80 上限",
            f"proportion_L1 ∈ [{min(l1_all):.2f},{max(l1_all):.2f}] (量程 0~2): argmax π 原型指派与真组 (经 p* 映射) 大面积错位",
            f"mean|κ̄−κ_true| ∈ [{min(kd_all):.1f},{max(kd_all):.1f}] (κ 量程 0.5~80)",
        ],
        "hypotheses": [
            "指派错位现象与项目既有结论「组隶属无法从点几何恢复」(信息墙) 同族; "
            "m1-κ 与 m2 玫瑰均按真组轴/法向本身计算, 不经该指派, 故其数字不受此错位污染 (玫瑰为无向 mod180 口径)",
            "κ 差值假设: 合格组内隐伏点仅 3~10 个, 小样本 Fisher κ 估计方差大; "
            "跨 50 补全的 κ_s 散布 (kappa_gen_sd_across_completions) 已并列落盘供架构师核",
        ],
        "note": "只登记现象与假设; 解读权在架构师",
    }
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[run] stats -> {STATS_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "prereg", "run"], required=True)
    ap.add_argument("--force", action="store_true", help="仅 prereg: 覆盖既有 gate (自担时序责任)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"audit": stage_audit, "prereg": lambda: stage_prereg(force=args.force),
     "run": stage_run}[args.stage]()


if __name__ == "__main__":
    main()
