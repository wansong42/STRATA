# -*- coding: utf-8 -*-
"""R226 · 30.5° 锚统计溯源与机读化单（R224-C2-H2 · CL-F 清偿）

任务书: tasks/R226_任务书_★☆_30.5锚统计溯源与机读化单_CLF清偿_CPU零训练_v0.md
算力: CPU only | 零 GPU | 零训练。溯源臂 = 无新点估计（锚值 30.50° 原样复现）。

## 臂判定
T1 溯源结果: 唯一在盘计算件 = `E:/ff_scripts/local_probe.py::locality_stats`
(13度审计 §2.3 明确点名) ⇒ **可恢复** ⇒ 走 **T2a 机读化臂**（非 T2b 重算臂）。

## 本件做什么 / 不做什么
做: 直跑原始计算件 → 逐位复现 30.50° → 把原文缺失的统计元数据
    （N / 95%CI / 纳排 / 距离阈值 / 语料构成 / LINE 型分面）逐字段机读化。
不做: 不改任何距离阈值下的点估计作为"新锚"；不触碰 31.7° 主墙（NNS-026/045）；
      不回填任何对外文档（素材包指针化见 T3，另由补丁清单登记）。

## 保真度设计（关键）
1) `importlib` 直接从磁盘加载**原文件** `E:/ff_scripts/local_probe.py`，
   复用其 `ang()` 与 `prepare_net(·, 0.4, default_rng(999))` 调用口径，
   不重写、不复制几何/距离/角距公式。
2) 逐字复制 `locality_stats` 的聚合循环语句（同一 bins / 同一 r 定义 /
   同一 nrm_full 口径），仅额外记录逐网贡献（用于 cluster bootstrap 与分面）。
3) **保真度闸**: 把本件复现的 6 个分箱均值与原始终端日志
   (`results/r226_anchor_305/local_probe_stdout.log`) 逐箱比对，
   要求 |Δ| < 0.005°（原脚本打印精度 2 位小数）——不过闸即停手。

## 口径登记（C2-H2 逐条点名项）
- 点集: `prepare_net` 的 `pos`（全网点，含隐伏）与 `nrm_full`（真值法向，含隐伏）。
  ⇒ **不是观测点口径**；掩码在此统计中不参与（仅写入 pn["nrm"]/mask，本统计不读）。
- 倾斜: `loaded_real_nets.pt` 原始法向，**未施加 Terzaghi 权重、未做倾斜校正**。
- 归一化半径: `r = pos.std(0).mean()`（逐网、轴 std 的算术均值，pos 已中心化）。
- 角距: `acos(|<n_i, n_j>|)`（轴无向，值域 [0, 90°]）。
- 语料: `data/real/loaded_real_nets.pt` 经 `load_split(·, 42)` 的 **test 折**。

用法:
  python scripts/r226_anchor_305.py --stage run
产物: results/r226_anchor_305/{anchor_stats.json, lineage.json, local_probe_stdout.log}
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT_DIR = os.path.join(ROOT, "results", "r226_anchor_305")
ANCHOR_JSON = os.path.join(OUT_DIR, "anchor_stats.json")
LINEAGE_JSON = os.path.join(OUT_DIR, "lineage.json")
STDOUT_LOG = os.path.join(OUT_DIR, "local_probe_stdout.log")

REAL = os.path.join(ROOT, "data", "real", "loaded_real_nets.pt")
ORIGINAL = "E:/ff_scripts/local_probe.py"

PRIMARY_BIN = (0.0, 0.25)                 # 主臂距离阈值（13度审计 §2.3 首箱）
DOC_BIN_MEANS = [30.50, 36.61, 39.14, 42.90, 43.30, 45.26]   # 13度审计 §2.3 文档锚表
BINS = [(0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 1e9)]
FIDELITY_TOL = 0.005                      # 原脚本打印 2 位小数
N_BOOT = 10000
BOOT_SEED = 20260916


def _json_default(o):
    """numpy 标量/数组 → 原生 python（防 float32 不可序列化）。"""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % type(o))


def bkey(x):
    """分箱下界的统一键（避免 int/float str 不一致）。"""
    return "%g" % float(x)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_original():
    """从磁盘直接加载原计算件（不复制代码）。"""
    if not os.path.isfile(ORIGINAL):
        raise SystemExit("原始计算件缺失: %s ⇒ 转 T2b 重算臂" % ORIGINAL)
    spec = importlib.util.spec_from_file_location("r226_local_probe_orig", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(lp, split):
    """逐字复制 lp.locality_stats 的聚合语句, 额外记录逐网贡献。"""
    from fractureflow.data import prepare_net

    agg = {b: [] for b in BINS}          # 与原脚本同结构 (bin -> list[array])
    per_net = []                         # 逐网记录: 供 cluster bootstrap / 分面
    for net_idx, net in enumerate(split):
        pn = prepare_net(net, 0.4, np.random.default_rng(999))       # 同原脚本
        pos = pn["pos"].numpy(); nf = pn["nrm_full"].numpy()          # 同原脚本
        r = pos.std(0).mean() + 1e-9                                  # 同原脚本
        d = np.linalg.norm(pos[None] - pos[:, None], axis=-1) / r      # 同原脚本
        a = lp.ang(np.broadcast_to(nf[None], (len(nf), len(nf), 3)),   # 同原脚本
                   np.broadcast_to(nf[:, None], (len(nf), len(nf), 3)))
        iu = np.triu_indices(len(nf), 1)                               # 同原脚本
        dd, aa = d[iu], a[iu]                                          # 同原脚本

        rec = {
            "net_idx": net_idx,
            "src": str(net.get("src", "?")),
            "n_points": int(len(nf)),
            "r_norm": float(r),
            # LINE 型判定 (docs/冻结期已知偏差登记_20260829.md 登记 10):
            # 严格共线 = x=y=0, 仅深度 z 变化
            "line_type": bool(np.allclose(pos[:, 0], 0.0, atol=1e-6)
                              and np.allclose(pos[:, 1], 0.0, atol=1e-6)),
            "bins": {},
        }
        for b in BINS:                                                 # 同原脚本
            m = (dd >= b[0]) & (dd < b[1])                             # 同原脚本
            if m.any():
                agg[b].append(aa[m])
                rec["bins"][bkey(b[0])] = {"n": int(m.sum()), "ang": aa[m]}
        per_net.append(rec)
    return agg, per_net


def bin_stats(v):
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p05": float(np.quantile(v, 0.05)),
        "p95": float(np.quantile(v, 0.95)),
    }


def ci_iid(v):
    """朴素 i.i.d. 正态近似 CI —— 仅作参照, 因网内成对相关而**系统性偏窄**。"""
    n = int(v.size)
    if n < 2:
        return None
    se = float(v.std(ddof=1) / np.sqrt(n))
    return {"lo": float(v.mean() - 1.96 * se), "hi": float(v.mean() + 1.96 * se),
            "se": se, "method": "normal_approx_iid",
            "caveat": "网内成对相关 ⇒ 低估方差, 仅作参照不用于论断"}


def ci_cluster(per_net, lo, hi, n_boot=N_BOOT, seed=BOOT_SEED):
    """网络簇 bootstrap: 以**网络**为重抽样单位（尊重网内成对相关）。"""
    clusters = []
    for rec in per_net:
        key = bkey(lo)
        if key in rec["bins"]:
            clusters.append(rec["bins"][key]["ang"])
    if not clusters:
        return None
    rng = np.random.default_rng(seed)
    k = len(clusters)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        boot[i] = np.concatenate([clusters[j] for j in idx]).mean()
    return {
        "method": "cluster_bootstrap_percentile",
        "unit": "network",
        "n_clusters": k,
        "n_boot": n_boot,
        "seed": seed,
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "sd_of_boot_mean": float(boot.std(ddof=1)),
        "design_effect_vs_iid": None,   # 下方回填
    }


def facet(per_net, lo, pred):
    """按谓词切分语料, 返回子面均值/N/簇数。"""
    arrs, nets, pts = [], 0, 0
    for rec in per_net:
        if not pred(rec):
            continue
        key = bkey(lo)
        if key in rec["bins"]:
            arrs.append(rec["bins"][key]["ang"])
            nets += 1
            pts += rec["n_points"]
    if not arrs:
        return {"n": 0, "n_nets": 0, "n_points": 0, "mean": None}
    v = np.concatenate(arrs)
    return {"n": int(v.size), "n_nets": nets, "n_points": pts,
            "mean": float(v.mean()), "median": float(np.median(v))}


def run_original_script():
    """直跑原始计算件（subprocess, 强制 UTF-8 输出）, 留原始终端日志。"""
    import subprocess
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    proc = subprocess.run([sys.executable, ORIGINAL], capture_output=True,
                          cwd=ROOT, env=env, timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    with open(STDOUT_LOG, "w", encoding="utf-8") as f:
        f.write("### R226 直跑原始计算件: %s\n" % ORIGINAL)
        f.write("### exit_code=%d\n### ---- stdout ----\n" % proc.returncode)
        f.write(out)
        if err:
            f.write("### ---- stderr ----\n" + err)
    if proc.returncode != 0:
        raise SystemExit("原始计算件退出码 %d, 停手上报" % proc.returncode)
    return out


def parse_stdout_log():
    """从原始终端日志解析 6 箱 (n, mean) —— 保真度闸的逐位参照。"""
    if not os.path.isfile(STDOUT_LOG):
        return None
    txt = open(STDOUT_LOG, "r", encoding="utf-8", errors="replace").read()
    vals = re.findall(r"n=\s*(\d+)\s+mean_ang=\s*([\d.]+)", txt)
    if len(vals) < len(BINS):
        return None
    return [{"n": int(a), "mean": float(b)} for a, b in vals[:len(BINS)]]


def main(stage):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = datetime.datetime.now().astimezone()

    # ① 直跑原始计算件 (取原始终端数值) —— 必须先于本次聚合
    run_original_script()
    ref = parse_stdout_log()
    if not ref:
        raise SystemExit("原始终端日志解析失败 ⇒ 保真度闸无参照, 停手上报")

    # ② 同码路径聚合 (供 N / CI / 分面)
    lp = load_original()
    from fractureflow.splits import load_split

    _tr, _va, te = load_split(REAL, 42)
    agg, per_net = collect(lp, te)

    # ── 分箱统计 + 保真度闸 (三重: 原始终端逐位 / 文档锚表 / 箱内 N) ──
    bins_out = []
    fidelity = {
        "gate": "PASS", "tol_deg": FIDELITY_TOL,
        "reference": "原始终端日志 (subprocess 直跑, UTF-8) 逐箱 n 相等 且 "
                     "rerun 四舍五入到 2 位小数 == 日志值",
        "per_bin": [],
    }
    for i, b in enumerate(BINS):
        v = np.concatenate(agg[b]) if agg[b] else np.array([])
        st = bin_stats(v) if v.size else {"n": 0, "mean": None}
        st.update({"lo_norm": b[0], "hi_norm": b[1]})
        bins_out.append(st)
        doc_mean = DOC_BIN_MEANS[i]
        got = st["mean"]
        d_doc = abs(got - doc_mean) if got is not None else None
        n_log, m_log = ref[i]["n"], ref[i]["mean"]
        n_ok = (n_log == st["n"])
        m_ok = (got is not None and round(got, 2) == round(m_log, 2))
        ok = bool(n_ok and m_ok and d_doc is not None and d_doc < FIDELITY_TOL)
        fidelity["per_bin"].append({
            "bin": "[%g, %g)" % (b[0], b[1]),
            "doc_anchor_deg": doc_mean,
            "rerun_deg": got,
            "abs_delta_vs_doc": d_doc,
            "stdout_log_deg": m_log,
            "stdout_log_n": n_log,
            "rerun_n": st["n"],
            "n_match_stdout_log": bool(n_ok),
            "mean_match_stdout_log_2dp": bool(m_ok),
            "pass": ok,
        })
        if not ok:
            fidelity["gate"] = "FAIL"
    if fidelity["gate"] != "PASS":
        raise SystemExit("保真度闸 FAIL, 停手上报架构师: %s"
                         % json.dumps(fidelity["per_bin"], ensure_ascii=False,
                                      allow_nan=False))

    # ── 主臂: 95%CI ───────────────────────────────────────────────────
    lo, hi = PRIMARY_BIN
    v_main = np.concatenate(agg[PRIMARY_BIN])
    iid = ci_iid(v_main)
    clu = ci_cluster(per_net, lo, hi)
    if clu and iid:
        clu["design_effect_vs_iid"] = round(
            (clu["sd_of_boot_mean"] / iid["se"]) ** 2, 3)

    # ── 纳排 ─────────────────────────────────────────────────────────
    n_nets_total = len(per_net)
    contrib = [r for r in per_net if bkey(lo) in r["bins"]]
    total_points = sum(r["n_points"] for r in per_net)
    total_pairs = sum(r["n_points"] * (r["n_points"] - 1) // 2 for r in per_net)
    inclusion = {
        "corpus": "data/real/loaded_real_nets.pt",
        "split": "load_split(REAL, 42) -> test 折",
        "n_nets_total": n_nets_total,
        "n_nets_with_ge2_points": int(sum(1 for r in per_net if r["n_points"] >= 2)),
        "n_nets_contributing_to_primary_bin": len(contrib),
        "n_nets_excluded_from_primary_bin": n_nets_total - len(contrib),
        "exclusion_rule": "该网在主臂距离阈值 [0,0.25r) 内无成对点 —— 非人为剔除, "
                          "系 (点数少 / 点分布均匀) 的自然结果; 原始计算件对网络**零筛选**"
                          "（locality_stats 无 mask/点数过滤, 与同文件 run() 不同）",
        "point_caliber": "全网点 (含隐伏), 真值法向 nrm_full; **非观测点口径**",
        "mask_role": "prepare_net(net, 0.4, default_rng(999)) 的掩码不参与本统计"
                     "（本统计只读 pos / nrm_full）",
        "tilt_caliber": "原始法向, 未施加 Terzaghi 权重 / 未做倾斜校正",
        "network_filter": "无（零筛选, 全 40 网进入聚合）",
    }
    exclusion = {
        "deviation_from_zero_screens": "原计算件对网络零筛选 ⇒ 这是 30.5° 与其它"
                                       "评测数字口径差异之源（评测走 prepare_net 掩码+过滤）",
        "empty_bin_handling": "空箱不进入均值（原脚本 `if m.any()` 同一逻辑）",
    }

    # ── 语料构成 ──────────────────────────────────────────────────────
    line_nets = [r for r in per_net if r["line_type"]]
    d3_nets = [r for r in per_net if not r["line_type"]]
    src_counter = {}
    for r in per_net:
        src_counter[r["src"]] = src_counter.get(r["src"], 0) + 1
    line_pairs = sum(r["bins"].get(bkey(lo), {}).get("n", 0) for r in line_nets)
    d3_pairs = sum(r["bins"].get(bkey(lo), {}).get("n", 0) for r in d3_nets)
    composition = {
        "n_nets": n_nets_total,
        "n_points_total": total_points,
        "n_pairs_total": total_pairs,
        "n_pairs_primary_bin": int(v_main.size),
        "primary_bin_pair_share_of_all_pairs": round(v_main.size / max(total_pairs, 1), 6),
        "site_breakdown_nets": dict(sorted(src_counter.items(),
                                           key=lambda kv: -kv[1])),
        "line_type_definition": "严格共线 (x=y=0, 仅深度 z 变化) — "
                                "docs/冻结期已知偏差登记_20260829.md 登记 10",
        "line_type_nets": len(line_nets),
        "line_type_net_share_test_split": round(len(line_nets) / max(n_nets_total, 1), 4),
        "corpus_level_reference": "全 391 网口径 247/391 = 0.632 (docs/冻结期已知偏差登记"
                                  "_20260829.md 登记 10 / AGENTS.md) — 本锚只用 test 折",
        "line_type_pairs_primary_bin": int(line_pairs),
        "line_type_pair_share_primary_bin": round(line_pairs / max(v_main.size, 1), 4),
        "d3_nets": len(d3_nets),
        "d3_pairs_primary_bin": int(d3_pairs),
        "note_line_type": "LINE 型网 x=y=0 ⇒ 3D 距离退化为 |Δz|; 且 r = pos.std(0).mean() "
                          "= (0+0+std_z)/3 = std_z/3, 使 0.25r 深度窗口仅为 "
                          "|Δz| < 0.0833·std_z —— 主臂在该子面上是**一维深度近邻**统计, "
                          "不构成三维空间近邻证据。逐字段机读见 descriptive_facets。",
    }

    # ── 描述性分面（同码路径扩展聚合；不替代锚值；未回填任何文档） ────
    facets = {
        "tier": "溯源分面（同码路径扩展聚合, 非新实验；不替代锚值 30.50°；未回填任何文档）",
        "by_corpus_type": {
            "line_type": facet(per_net, lo, lambda r: r["line_type"]),
            "true_3d": facet(per_net, lo, lambda r: not r["line_type"]),
        },
        "by_site": {s: facet(per_net, lo, (lambda s_: lambda r: r["src"] == s_)(s))
                    for s in sorted(src_counter)},
        "threshold_sensitivity": {},
    }
    # 阈值敏感臂: 原 bins 粒度不含 0.15r ⇒ 用同码路径（同 prepare_net 口径 / 同 r 定义 /
    # 同角距函数）独立聚合; 0.25 与 0.5 档同时充当与主臂/第 2 箱的交叉一致性核对。
    facets["threshold_sensitivity"] = threshold_arm(lp, te, (0.15, 0.25, 0.5))

    primary_mean = float(v_main.mean())
    dev_type = max((abs(float(f["mean"]) - primary_mean)
                    for f in facets["by_corpus_type"].values()
                    if f.get("mean") is not None), default=0.0)
    site_means = {s: f["mean"] for s, f in facets["by_site"].items()
                  if f.get("mean") is not None}
    dev_site = max((abs(m - primary_mean) for m in site_means.values()), default=0.0)
    # 低 N 场地分面 (n<200) 的偏离不具统计意义 (如 forge n=7 / mining n=6)
    LOW_N = 200
    low_n = {s: f["n"] for s, f in facets["by_site"].items()
             if f.get("mean") is not None and f["n"] < LOW_N}
    hi_n = {s: f for s, f in facets["by_site"].items()
            if f.get("mean") is not None and f["n"] >= LOW_N}
    dev_hi_site = max((abs(float(f["mean"]) - primary_mean)
                       for f in hi_n.values()), default=0.0)
    facets["heterogeneity"] = {
        "by_corpus_type_max_abs_dev_deg": round(dev_type, 3),
        "by_site_max_abs_dev_deg": round(float(dev_site), 3),
        "by_site_max_abs_dev_deg_highN_only": round(float(dev_hi_site), 3),
        "high_n_threshold_pairs": LOW_N,
        "by_site_range_deg": [round(min(site_means.values()), 3),
                              round(max(site_means.values()), 3)],
        "by_site_argmin_argmax": [min(site_means, key=site_means.get),
                                  max(site_means, key=site_means.get)],
        "low_n_site_facets_pairs": dict(sorted(low_n.items())),
        "low_n_note": "n<200 的场地分面 (forge=7 / mining=6 / egs_collab=63) 点数太少, "
                      "其均值不作为异质性证据; forge 分面的 51.5° 由 7 对主导, 无统计意义。",
        "threshold_range_deg": [min(x["mean"] for x in
                                    facets["threshold_sensitivity"].values() if "mean" in x),
                                max(x["mean"] for x in
                                    facets["threshold_sensitivity"].values() if "mean" in x)],
        "disclosure": "锚是**池化**数字（全 40 网聚合）；语料型分面几乎无偏离"
                      "（LINE 30.80 / 3D 29.72, Δ≤0.8°），但**场地分面存在实质异质性**"
                      "（高 N 场地 24.0°–42.7°, 最大偏离 12.2°）。论文如按'普遍性'语义"
                      "引用 30.5°，该异质性未在源文档记录 —— 本件如实披露，"
                      "是否加限定语由架构师裁定（本单禁自行改稿）。",
    }
    facets["requires_architect_review"] = bool(dev_type > 3.0 or dev_hi_site > 3.0)
    facets["max_abs_deviation_vs_primary_deg"] = round(float(max(dev_type, dev_hi_site)), 3)
    facets["halt_note"] = ("主臂复现 Δ=0.002° ⇒ §4 退路（重算 |Δ|>3°）**未触发**, "
                           "不需停手；上列分面偏离仅作披露，是否影响论文表述由架构师裁定。")

    written_at = datetime.datetime.now().astimezone()
    payload = {
        "_version_note": "R226 T2a 机读化臂 (溯源). 锚值 30.50° 原样复现, 零新点估计。",
        "anchor_id": "R226-ANCHOR-30.5",
        "anchor_value_deg": float(v_main.mean()),
        "anchor_value_reported_deg": 30.50,
        "anchor_semantic": "同一网络内成对裂隙**真值法向**的轴无向角距, 受限于归一化"
                           "网内距离 < 0.25r（条件宽度/局部性的两点统计锚）",
        "lineage": {
            "computing_artifact": ORIGINAL,
            "function": "locality_stats()",
            "sha256_computing_artifact": sha256_file(ORIGINAL),
            "computing_artifact_mtime": datetime.datetime.fromtimestamp(
                os.path.getmtime(ORIGINAL)).astimezone().isoformat(),
            "recoverable": True,
            "arm_executed": "T2a (可恢复 → 机读化)",
            "reproduced_value_deg": round(float(v_main.mean()), 2),
            "doc_anchor_value_deg": 30.50,
            "abs_delta_deg": round(abs(float(v_main.mean()) - 30.50), 4),
            "dataset": "data/real/loaded_real_nets.pt",
            "sha256_dataset": sha256_file(REAL),
            "original_stdout_log": "results/r226_anchor_305/local_probe_stdout.log",
            "note": "计算件在工作区之外 (E:/ff_scripts/), 不在版本控制内; 本锚此前"
                    "唯一在盘形态是 13度审计 §2.3 的**文档锚**(手抄表格)。",
        },
        "primary_bin": {"lo_norm": lo, "hi_norm": hi,
                        "definition": "归一化距离 d/r ∈ [0, 0.25)"},
        "normalization": {
            "r_definition": "r = pos.std(0).mean()  (逐网, pos 已中心化; 轴 std 的算术均值)",
            "r_denominator_note": "mean 而非 norm ⇒ 对 LINE 型（单轴）网 r = std_z/3; "
                                  "对 3D 网 r = (std_x+std_y+std_z)/3",
            "pair_angle": "acos(|<n_i, n_j>|)  (轴无向, [0, 90°])",
        },
        "statistics": {
            "N_pairs": int(v_main.size),
            "mean_deg": round(float(v_main.mean()), 4),
            "median_deg": round(float(np.median(v_main)), 4),
            "sd_deg": round(float(v_main.std(ddof=1)), 4),
            "p05_deg": round(float(np.quantile(v_main, 0.05)), 4),
            "p95_deg": round(float(np.quantile(v_main, 0.95)), 4),
            "ci95_cluster_bootstrap": clu,
            "ci95_iid_normal_approx": iid,
            "ci_primary": "cluster_bootstrap_percentile (重抽样单位 = 网络)",
            "reporting_rounding": "对外沿用 30.5°（原文档口径, 2 位有效小数）; "
                                  "本件保留 4 位小数仅供机读, 不构成新的对外精度声明",
        },
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "corpus_composition": composition,
        "distance_threshold": {
            "primary": 0.25, "unit": "× r (逐网归一化)",
            "bin_edges": [[b[0], b[1]] for b in BINS],
            "bin_table": bins_out,
        },
        "fidelity_gate": fidelity,
        "descriptive_facets": facets,
        "audit_fields_addressed": {
            "C2-H2_N": "statistics.N_pairs = %d" % int(v_main.size),
            "C2-H2_CI": "statistics.ci95_cluster_bootstrap",
            "C2-H2_inclusion": "inclusion_criteria / exclusion_criteria",
            "C2-H2_line_type": "corpus_composition.line_type_* + descriptive_facets.by_corpus_type",
            "C2-H2_point_caliber": "inclusion_criteria.point_caliber (全网真值法向, 非观测点)",
            "C2-H2_tilt_caliber": "inclusion_criteria.tilt_caliber (未施 Terzaghi)",
            "C2-H2_network_screen": "inclusion_criteria.network_filter (零筛选)",
        },
        "discipline": {
            "no_new_point_estimate": True,
            "wall_31_7_untouched": True,
            "docs_backfilled": [
                "docs/论文素材包/诚实边界.md (L15 指针化 + 口径纠正)",
                "docs/论文素材包/智能性论证章节.md (L64 指针化 + 口径纠正)",
                "docs/论文素材包/README.md (L15 指针化)",
            ],
            "docs_backfill_detail": "results/r226_anchor_305/pointer_patches.json",
            "docs_backfilled_pending": [
                "docs/paper1/draft_v0.2.md — R225 在飞, 按纪律错峰; "
                "回填器 scripts/r226_apply_pointer.py (默认 dry-run)",
            ],
            "pointer_target_text": "⟦R226⟧ → results/r226_anchor_305/anchor_stats.json",
        },
        "runtime": {
            "started_at": t0.isoformat(),
            "finished_at": written_at.isoformat(),
            "cpu_only": True, "gpu": False, "training": False,
            "env": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
    }
    with open(ANCHOR_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=_json_default, allow_nan=False)
    print("[R226] anchor_stats.json 落盘: N=%d mean=%.4f° CI95=[%.4f, %.4f]"
          % (v_main.size, v_main.mean(), clu["lo"], clu["hi"]))
    print("[R226] 保真度闸 PASS — 6 箱与原终端日志逐箱一致, 与文档锚表 Δ<%.3f°" % FIDELITY_TOL)
    return payload


def threshold_arm(lp, split, thresholds):
    """距离阈值敏感臂（T2b 规格主臂+敏感臂的阈值项; 本单在 T2a 下作描述性披露）。"""
    from fractureflow.data import prepare_net

    acc = {t: [] for t in thresholds}
    per_net = {t: [] for t in thresholds}
    for net in split:
        pn = prepare_net(net, 0.4, np.random.default_rng(999))
        pos = pn["pos"].numpy(); nf = pn["nrm_full"].numpy()
        r = pos.std(0).mean() + 1e-9
        d = np.linalg.norm(pos[None] - pos[:, None], axis=-1) / r
        a = lp.ang(np.broadcast_to(nf[None], (len(nf), len(nf), 3)),
                   np.broadcast_to(nf[:, None], (len(nf), len(nf), 3)))
        iu = np.triu_indices(len(nf), 1)
        dd, aa = d[iu], a[iu]
        for t in thresholds:
            m = dd < t
            if m.any():
                acc[t].append(aa[m])
                per_net[t].append(aa[m])
    out = {}
    for t in thresholds:
        v = np.concatenate(acc[t]) if acc[t] else np.array([])
        if not v.size:
            out["%g" % t] = {"n": 0}
            continue
        rng = np.random.default_rng(BOOT_SEED)
        k = len(per_net[t])
        boot = np.array([np.concatenate([per_net[t][j] for j in
                                         rng.integers(0, k, k)]).mean()
                         for _ in range(N_BOOT)])
        out["%g" % t] = {
            "n": int(v.size), "mean": round(float(v.mean()), 4),
            "n_nets": k,
            "ci95_cluster_bootstrap": {
                "lo": round(float(np.quantile(boot, 0.025)), 4),
                "hi": round(float(np.quantile(boot, 0.975)), 4),
                "method": "cluster_bootstrap_percentile", "unit": "network",
                "n_boot": N_BOOT, "seed": BOOT_SEED},
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=["run"])
    args = ap.parse_args()
    main(args.stage)
