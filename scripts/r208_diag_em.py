# -*- coding: utf-8 -*-
"""R208 诊断 · 离线 polar-Bingham 混合 EM 基线 (CPU, 零训练) —— T0-P3 修正版。

【缺陷记录 T0-P3 —— 先修后跑, 原因见下】
原版 `em_mixture` 的 κ-MLE 二分**方向写反** (且 lo/hi 短路分支同向):
  * κ-MLE 条件: E_κ[x1²] = d1 (d1 = 加权散布最大特征值), E_κ[x1²] 对 κ **单调递增**
    (κ→0 ⇒ 1/3, κ→∞ ⇒ 1; 400 点 GL 求积实测 0.38 → 0.9980);
  * 原码 `grad(κ) = E_κ[x1²] − d1`, 但 `if grad(lo) < 0: kappa = lo` —— 实测
    真实站点 d1 ≈ 0.968~0.972 ⇒ grad(0.5) = 0.3797 − 0.97 ≈ −0.59 < 0 ⇒ **每次都取 lo**,
    κ 被钉死在 0.5 (最近点), 二分体从未执行。
  * 后果: 混合退化为"近均匀分量"⇒ 责任度趋同 ⇒ K 个组心在 2~3 轮内坍缩到全局主轴,
    modal_err 假高 (~35–45°, 恰等于 90°−α) —— 会被误读成「观测信息不够」而错误闭账。
  本版按正确方向二分并保留全轮对数似然单调自检 (EM 有效性硬证据)。

【本脚本回答的三个问题 (判据面之外, 纯诊断)】
  Q1 经典离线最优 (kmeans 初始化 + 责任度 EM + 精确 M-step) 在这些站点能到多少?
  Q2 它能否打赢 `kmeans_obs` ≥1.0°? —— 主臂 GO 判据的**可达性前提**:
     主臂损失 = 观测点上球面混合 NLL, 其全局最优**就是**该 NLL 的 MLE,
     故本表同时是主臂在无限容量/完美优化下的**上界**。
  Q3 主臂结构代价: K_max=6 固定 + top-K 选择, 在 K_true∈{3,4} 站点上过分裂损失多少?

【臂】kmeans_obs(官方基线逐位复现, 交叉校验) / em_plain_K / em_terzaghi_K /
      em_terzaghi_Kmax6_topK (主臂结构代理, 选择规则同模型 resp_sum)
      + diag_only_* (模型选择的 oracle 上界, **诊断专用, 禁入任何结果表**, 见 red_line_note)
【自检】① kmeans_obs 与在盘基线逐位一致; ② EM 加权对数似然逐轮单调不降。
用法: ...python.exe scripts/r208_diag_em.py [--sites ...] [--iters 60]
"""
from __future__ import annotations
import argparse, datetime, importlib.util, json, os, sys, time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fractureflow.set_table_eval import set_table_score          # noqa: E402
from fractureflow.setlabel import spherical_kmeans, _sign_align  # noqa: E402
from fractureflow.orthostats import log_bingham_norm_np          # noqa: E402

RES = os.path.join(ROOT, "results", "r208_mixture_reboot")
SITES = os.path.join(RES, "train_sites")
R207 = os.path.join(ROOT, "results", "r207_synth_truth_range")
R204J = os.path.join(ROOT, "results", "r204_set_metric_truth_gate")
WATERMARK = "研究线数字, 禁入对外"
AXIS = np.array([0.0, 0.0, 1.0])
KAPPA_LO, KAPPA_HI = 0.5, 2000.0          # 诊断不加模型侧 κ≤200 夹断, 取真 MLE
W_LO, W_HI = 0.1, 5.0                     # 与 terzaghi_weights_t(legacy 语义) 同口径

_GL_T, _GL_W = np.polynomial.legendre.leggauss(400)   # 一次预计算 (leggauss 确定性)


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


# --------------------------------------------------------------------- κ MLE --
def ex1(kappa):
    """E_κ[x1²] = ∫t²e^{κt²}dt / ∫e^{κt²}dt (对 κ 单调递增)。支持向量输入。

    数值: 写成 e^{κ(t²−1)} (≤1) 防 κ≈2000 时 e^{κ} 上溢 (比值里 e^{−κ} 因子相消)。
    """
    k = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    p = np.exp(k[:, None] * (_GL_T ** 2 - 1.0)) * _GL_W      # (m,400), 0 ≤ p ≤ w
    return (p * (_GL_T ** 2)).sum(1) / p.sum(1)


def solve_kappa(d1):
    """解 E_κ[x1²] = d1 (单调递增 ⇒ 二分方向: E<d1 抬 lo)。向量化, 60 步。"""
    d = np.atleast_1d(np.asarray(d1, dtype=np.float64))
    lo = np.full_like(d, KAPPA_LO)
    hi = np.full_like(d, KAPPA_HI)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = ex1(mid) < d                                     # 需要更大的 κ
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.clip(0.5 * (lo + hi), KAPPA_LO, KAPPA_HI)


# ------------------------------------------------------------------------- EM --
def em_mixture(nrm, K, w_pts=None, iters=60, seed=42, tol=1e-11):
    """polar Bingham 混合 EM (kmeans 初始化 + 精确 M-step)。

    返回 dict(centers, pi, kappa, resp_sum, ll_trace, n_iter, mono_ok)。
    resp_sum = 末轮**不加权**责任度质量和 Σ_i r_ik (与模型 resp_sum 选择规则同口径)。
    """
    nrm = _unit(np.asarray(nrm, dtype=np.float64))
    n = nrm.shape[0]
    K = int(min(K, n))
    w_pts = np.ones(n) if w_pts is None else np.asarray(w_pts, dtype=np.float64)
    sw_tot = float(w_pts.sum())
    cents, _ = spherical_kmeans(nrm, K, seed=seed)
    cents = _unit(cents)
    pi = np.full(K, 1.0 / K)
    kappa = np.full(K, 20.0)
    ll_trace, resp_sum, mono_ok = [], None, True
    for _ in range(iters):
        cos2 = (nrm @ cents.T) ** 2                                   # (n,K)
        logc = log_bingham_norm_np(np.stack([np.zeros(K), kappa, kappa], axis=1))
        logf = kappa[None, :] * cos2 - kappa[None, :] - logc[None, :]
        logr = np.log(pi)[None, :] + logf
        mx = logr.max(axis=1, keepdims=True)
        r = np.exp(logr - mx)
        rsum = r.sum(axis=1, keepdims=True)                           # exp(0) 归一
        r /= rsum
        ll = float((w_pts * (mx[:, 0] + np.log(rsum[:, 0]))).sum() / sw_tot)
        ll_trace.append(ll)
        if len(ll_trace) >= 2 and ll_trace[-1] < ll_trace[-2] - 1e-9:
            mono_ok = False
        resp_sum = r.sum(axis=0)
        # ---- M-step ----
        d1 = np.empty(K)
        for k in range(K):
            wk = r[:, k] * w_pts
            sw = float(wk.sum())
            if sw < 1e-12:
                d1[k] = ex1(kappa[k])
                continue
            S = (wk[:, None, None] * nrm[:, :, None] * nrm[:, None, :]).sum(0) / sw
            ev, evec = np.linalg.eigh(S)
            m = evec[:, -1]
            if m @ cents[k] < 0:
                m = -m
            cents[k] = _unit(m)
            d1[k] = float(ev[-1])
        kappa = solve_kappa(d1)
        pi = (w_pts[:, None] * r).sum(0) / sw_tot
        pi = np.clip(pi, 1e-9, None)
        pi = pi / pi.sum()
        if len(ll_trace) >= 2 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            break
    return {"centers": cents, "pi": pi, "kappa": kappa, "resp_sum": resp_sum,
            "ll_trace": ll_trace, "n_iter": len(ll_trace), "mono_ok": mono_ok}


def terzaghi_w(nrm, axis=AXIS):
    """1/|n·a| 自归一化后 clip [0.1,5] (与 terzaghi_weights_t legacy 语义同口径)。"""
    w = 1.0 / np.clip(np.abs(_unit(nrm) @ axis), 1e-6, None)
    return np.clip(w / w.mean(), W_LO, W_HI)


def best_sel(cents, truth, K, diag_only=True):
    """诊断专用: 在 C(Kf,K) 个子集中取 modal_err 最小的那组 (oracle 选择上界)。"""
    from itertools import combinations
    best = 1e9
    for idx in combinations(range(len(cents)), K):
        e = set_table_score(cents[list(idx)], truth)["modal_err_deg"]
        if e < best:
            best = e
    return float(best)


def run_site(nrm_full, masks, truth, K, has_axis=True, seed0=42, iters=60, tag=""):
    """一套掩码 × 多臂。返回逐臂 per-seed 与均值。"""
    truth = np.asarray(truth, dtype=np.float64)
    acc = defaultdict(list)
    ll_min_mono = True
    for s in range(len(masks)):
        occ = np.asarray(masks[s], dtype=bool)
        no = _unit(nrm_full[occ])
        # ---- 官方基线逐位复现 (centers_obs: kmeans(_unit(nrm), min(K,n), seed=42)) ----
        km = spherical_kmeans(no, min(int(K), len(no)), seed=seed0)[0]
        acc["kmeans_obs"].append(set_table_score(km, truth)["modal_err_deg"])
        w = terzaghi_w(no, AXIS) if has_axis else np.ones(len(no))
        arms = [("em_plain_K", int(K), np.ones(len(no)), None),
                ("em_terzaghi_K", int(K), w, None),
                ("em_terzaghi_Kmax6_topK", 6, w, int(K))]
        if has_axis:
            arms.append(("diag_only_em_terzaghi_Kmax6_bestsel", 6, w, int(K)))
        for name, Kfit, wv, topk in arms:
            if name.startswith("diag_only") and not has_axis:
                continue
            r = em_mixture(no, Kfit, wv, iters=iters, seed=seed0)
            ll_min_mono = ll_min_mono and r["mono_ok"]
            cents = r["centers"]
            if name.startswith("diag_only"):
                acc[name].append(best_sel(cents, truth, topk))
            elif topk is not None:
                sel = np.argsort(-r["resp_sum"])[:topk]
                acc[name].append(set_table_score(cents[sel], truth)["modal_err_deg"])
            else:
                acc[name].append(set_table_score(cents, truth)["modal_err_deg"])
    return {"site": tag, "K_true": int(K), "has_axis": bool(has_axis),
            "n_masks": len(masks), "n_obs_typical": int(np.sum(masks[0])),
            "arms": {k: {"per_seed": [round(float(x), 4) for x in v],
                         "mean_deg": round(float(np.mean(v)), 4)} for k, v in acc.items()},
            "em_ll_monotone_all": bool(ll_min_mono)}


def r208_site_ids():
    man = json.load(open(os.path.join(SITES, "manifest.json"), encoding="utf-8"))
    return [s["site_id"] for s in man["sites"] if s["kind"] == "val"]


def r207_site_ids():
    tbl = json.load(open(os.path.join(R207, "difficulty_table.json"), encoding="utf-8"))
    return list(tbl["summary"]["usable_site_ids"])


def load_r204():
    spec = importlib.util.spec_from_file_location(
        "r204_gate", os.path.join(ROOT, "scripts", "r204_set_table_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decovalex_rows(K_primary=3):
    """DECOVALEX 4-frac_plus (Family 真值, has_axis=False ⇒ 无井轴 ⇒ 无 Terzaghi 臂)。"""
    R204 = load_r204()
    t2 = json.load(open(os.path.join(R204J, "t2_set_table_truth.json"), encoding="utf-8"))
    st = t2["sites"]["decovalex_4fracplus_fam123"]
    nets_f, dpos, dnrm, *_rest, dcent, _al = R204.decovalex_data()
    truth = np.asarray(dcent, dtype=np.float64)
    N = len(dpos)
    masks = []
    for s in range(10):
        rng = np.random.default_rng(999 + 1000 * s)
        m = np.zeros(N, dtype=bool)
        sl = 0
        for n in nets_f:
            nf = len(n["pos"])
            m[sl:sl + nf] = R204.mask_of(nf, rng)
            sl += nf
        masks.append(m)
    row = run_site(dnrm, np.array(masks), truth, int(K_primary), has_axis=False,
                   tag="decovalex_4fracplus_fam123")
    row["kmeans_obs_stored_per_seed"] = \
        st[str(K_primary)]["kmeans_obs"]["per_seed"]
    row["kmeans_obs_stored_mean_deg"] = st[str(K_primary)]["kmeans_obs"]["modal_err_deg"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="all", help="all / val / r207 / 逗号分隔站点 id")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="diag_em_baseline.json")
    a = ap.parse_args()

    if a.sites == "all":
        ids = [("r208_val", s) for s in r208_site_ids()] + \
              [("r207", s) for s in r207_site_ids()]
    elif a.sites == "val":
        ids = [("r208_val", s) for s in r208_site_ids()]
    elif a.sites == "r207":
        ids = [("r207", s) for s in r207_site_ids()]
    else:
        ids = [("r208_val", s) for s in a.sites.split(",")]

    out = {"task": "R208-T0-P3", "prereg": "R208-PREREG-V1",
           "watermark": WATERMARK,
           "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "env": {"python": sys.executable, "numpy": np.__version__},
           "defect_record": {
               "id": "R208-T0-P3",
               "where": "scripts/r208_diag_em.py::em_mixture κ-MLE 二分",
               "what": "二分方向写反 + lo/hi 短路分支同向 ⇒ grad(κ=0.5)<0 恒成立 ⇒ κ 钉死 0.5",
               "evidence": "scripts/_r208_kappa_probe.py: E_κ[x1²] 0.3797(κ=0.5)→0.997998(κ=500) "
                           "单调递增; 真实站点 d1=0.968~0.972 ⇒ grad(0.5)≈−0.59<0",
               "impact": "κ 取最近点 ⇒ 责任度趋同 ⇒ 组心坍缩到全局主轴 ⇒ modal_err 假高 ~90°−α; "
                         "若直接采信会把『诊断器坏了』误判成『观测信息不够』并触发永久闭账",
               "fix": "按 E_κ[x1²]=d1 的正确方向二分 (60 步), 并加 EM 对数似然逐轮单调自检",
               "note": "本文件为诊断件, 不属判据面; 修复先于首次运行, 零历史结论受影响"},
           "protocol": {
               "estimand": "总体组心 (R207 truth_centers = 全总体符号对齐均值)",
               "metric": "fractureflow.set_table_eval.set_table_score (verbatim)",
               "kmeans_obs": "spherical_kmeans(_unit(nrm_obs), min(K,n), seed=42) —— R204 centers_obs 逐字",
               "em": "kmeans(seed=42) 初始化 + 责任度 EM + 加权散布主特征向量 + E_κ[x1²]=d1 的 κ MLE, "
                     "调用 orthostats.log_bingham_norm_np 取 log c (与主臂损失同一条配分函数)",
               "terzaghi": "1/|n·a| 自归一化 + clip[0.1,5]; has_axis=False 的站点不设 terzaghi 臂",
               "Kmax6_topK": "K=6 过参数 EM 后按 Σ_i r_ik (不加权, 同模型 resp_sum) 取 top-K",
               "red_line_note": "diag_only_* 为 oracle 分量选择上界, 仅用于定位失败层级, "
                                "**禁入 verdict/任何结果表** (任务书红线: 主判据禁 oracle)"},
           "sites": {}}

    t0 = time.time()
    for kind, sid in ids:
        if kind == "r208_val":
            z = np.load(os.path.join(SITES, sid + ".npz"))
            meta = json.load(open(os.path.join(SITES, sid + "_meta.json"), encoding="utf-8"))
            row = run_site(z["nrm_full"], z["masks"], z["truth_centers"],
                           int(meta["K_true"]), has_axis=True, iters=a.iters, tag=sid)
        else:
            z = np.load(os.path.join(R207, "sites", sid + ".npz"))
            meta = json.load(open(os.path.join(R207, "sites", sid + "_meta.json"),
                                  encoding="utf-8"))
            row = run_site(z["nrm_full"], z["masks"], z["truth_centers"],
                           int(meta["K_true"]), has_axis=True, iters=a.iters, tag=sid)
            basel = json.load(open(os.path.join(R207, "sites", sid + "_baselines.json"),
                                   encoding="utf-8"))
            stored = basel["eval"][str(int(meta["K_true"]))]["kmeans_obs"]["per_seed"]
            row["kmeans_obs_stored_per_seed"] = stored
            row["kmeans_obs_stored_mean_deg"] = basel["eval"][str(int(meta["K_true"]))][
                "kmeans_obs"]["modal_err_deg"]
            row["kmeans_obs_bitwise_match"] = bool(np.allclose(
                stored, row["arms"]["kmeans_obs"]["per_seed"], atol=1e-6))
        out["sites"][sid] = row
        print("[t0p3] %-22s K=%d %s" % (sid, row["K_true"],
              "  ".join("%s=%.3f" % (k, v["mean_deg"]) for k, v in row["arms"].items())),
              flush=True)

    try:
        out["sites"]["decovalex_4fracplus_fam123"] = decovalex_rows()
        r = out["sites"]["decovalex_4fracplus_fam123"]
        print("[t0p3] %-22s K=%d %s" % ("decovalex_4fracplus_fam123", r["K_true"],
              "  ".join("%s=%.3f" % (k, v["mean_deg"]) for k, v in r["arms"].items())),
              flush=True)
    except Exception as exc:                                    # noqa: BLE001
        out["decovalex_error"] = "%s: %s" % (type(exc).__name__, exc)
        print("[t0p3] decovalex 跳过: %s" % exc, flush=True)

    # ---- 汇总: 站点级 headroom (em_terzaghi_K vs kmeans_obs) ----
    summary = {"per_site_headroom_terzaghi_minus_kmeans_deg": {},
               "n_sites_terzaghi_beats_kmeans_by_1deg": 0,
               "n_sites_em_plain_beats_kmeans_by_1deg": 0,
               "kmeans_obs_crosscheck_all_match": True,
               "em_ll_monotone_all": True,
               "structural_cost_Kmax6_topK_mean_deg": []}
    for sid, row in out["sites"].items():
        ar = row["arms"]
        if "kmeans_obs" not in ar:
            continue
        km = ar["kmeans_obs"]["mean_deg"]
        if "em_terzaghi_K" in ar:
            d = km - ar["em_terzaghi_K"]["mean_deg"]
            summary["per_site_headroom_terzaghi_minus_kmeans_deg"][sid] = round(d, 4)
            if d >= 1.0:
                summary["n_sites_terzaghi_beats_kmeans_by_1deg"] += 1
            if "em_terzaghi_Kmax6_topK" in ar:
                summary["structural_cost_Kmax6_topK_mean_deg"].append(
                    [sid, round(ar["em_terzaghi_Kmax6_topK"]["mean_deg"]
                                - ar["em_terzaghi_K"]["mean_deg"], 4)])
        if "em_plain_K" in ar and km - ar["em_plain_K"]["mean_deg"] >= 1.0:
            summary["n_sites_em_plain_beats_kmeans_by_1deg"] += 1
        if row.get("kmeans_obs_bitwise_match") is False:
            summary["kmeans_obs_crosscheck_all_match"] = False
        if not row.get("em_ll_monotone_all", True):
            summary["em_ll_monotone_all"] = False
    out["summary"] = summary
    out["runtime_sec"] = round(time.time() - t0, 1)

    path = os.path.join(RES, a.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("[t0p3] -> %s (%.0fs)" % (path, out["runtime_sec"]), flush=True)
    print("[t0p3] kmeans 交叉校验全对 = %s; EM 似然单调 = %s"
          % (summary["kmeans_obs_crosscheck_all_match"], summary["em_ll_monotone_all"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
