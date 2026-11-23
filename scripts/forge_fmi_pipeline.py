# -*- coding: utf-8 -*-
"""FORGE 真实多井 路线 A 验证 + 产品化 主流程 (Phase 1.4 / 2 / 3).

依赖: scripts/read_forge_las.py (已产出 forge_fmi_2wells.pt)
协议: 掩码 default_rng(999).random(n)<0.4 ; KMeans seed=42 ; 口径 acos(|<pred,true>|) 度.

产出:
  - 1.4  forge_fmi_survey.json / .png  走向勘察 + 共享组系判定
  - 2.1  results/forge_multiwell_real.json  单井 vs 多井联合 vs 井内模态
  - 2.2  results/forge_kgrid.json          K 网格 + 锚点
  - 2.3  results/forge_typegate.json       类型门控复测
  - 2.4  results/forge_vs_loaded.json      与既有路线 A 横向对比 (八期前口径参照值已判废, 不再打印)
  - 3    data/external/utah_forge_fmi/forge_report_v2/{group_table.csv,rose.png,stereonet.png,confidence.png}
"""
import os
import sys
import json
import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fractureflow.setlabel import generate_set_ids, joint_set_ids, spherical_kmeans
from fractureflow.inference import set_aware_dirs, propagation_confidence

# --- local definitions (these were removed from report.py during refactor) ---


def dip_dir_of(nrm):
    """法向 -> (dip, dip_direction), 度.

    W43: 改为**委托权威反解** `fractureflow.geometry.normal_to_dip_dipdir`
    (含 nz<0 时方位 +180 的补偿; 无向铁律: n 与 -n 同解)。旧实现自写
    `atan2(nE, nN)` 且只对 dip 取 |nz|、对方位不给符号补偿 ⇒ 组内 sign-align
    (本文件 `group_table`:60-64 以组内首行为参照) 一旦把组心对到下半球,
    组系表 CSV 的倾向/走向列与玫瑰图整 180° 错。nz >= 0 的输入逐位不变。
    返回形态与旧实现一致 (2-D 输入 -> 1-D 数组)。
    """
    from fractureflow.geometry import normal_to_dip_dipdir as _n2dd
    return _n2dd(np.asarray(nrm, float))


def group_table(wells_nrm, assign, depth=None, ftype=None):
    """Build group-table rows (list[dict]) compatible with write_group_table_csv."""
    wells_nrm = [np.asarray(w, float) for w in wells_nrm]
    assign = [np.asarray(a, int) for a in assign]
    all_nrm = np.vstack(wells_nrm)
    all_assign = np.concatenate(assign)
    n_total = max(len(all_nrm), 1)
    K = int(all_assign.max()) + 1 if len(all_assign) else 0
    rows = []
    for k in range(K):
        mask = all_assign == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        nrm_k = all_nrm[mask]
        nrm_k = nrm_k / (np.linalg.norm(nrm_k, axis=-1, keepdims=True) + 1e-12)
        # sign-align
        ref = nrm_k[0]
        sgn = np.sign((nrm_k * ref).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1
        nrm_k = nrm_k * sgn
        center = nrm_k.mean(0)
        center = center / (np.linalg.norm(center) + 1e-12)
        dip, dip_dir = dip_dir_of(center.reshape(1, -1))
        strike = float((dip_dir[0] - 90.0) % 360.0)
        pct = n_k / n_total * 100.0
        if pct >= 20.0:
            grade = "很发育"
        elif pct >= 10.0:
            grade = "发育"
        elif pct >= 5.0:
            grade = "较发育"
        else:
            grade = "不发育"
        row = dict(
            group_id=k + 1,
            n_fractures=n_k,
            mean_dip_dir=round(float(dip_dir[0]), 1),
            mean_dip=round(float(dip[0]), 1),
            mean_strike=round(strike, 1),
            proportion=round(pct, 1),
            development_grade=grade,
        )
        if depth is not None:
            depth = np.asarray(depth, float)
            d_k = depth[mask]
            row["depth_min"] = round(float(d_k.min()), 1)
            row["depth_max"] = round(float(d_k.max()), 1)
            row["depth_mean"] = round(float(d_k.mean()), 1)
        if ftype is not None:
            ft_k = np.array(ftype)[mask]
            nat_pct = float((ft_k == "natural").sum()) / n_k * 100.0
            row["natural_pct"] = round(nat_pct, 1)
        rows.append(row)
    return rows


def write_group_table_csv(rows, path):
    """Write group-table rows to CSV."""
    import csv
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rose_diagram(path, wells_nrm, assign):
    """Per-group rose diagram (polar histogram of dip-direction)."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, 8)),
                      labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    cmap = plt.cm.tab10
    assign = [np.asarray(a, int) for a in assign]
    all_nrm = np.vstack([np.asarray(w, float) for w in wells_nrm])
    all_assign = np.concatenate(assign)
    K = int(all_assign.max()) + 1 if len(all_assign) else 0
    bins = np.linspace(0, 360, 37)
    for k in range(K):
        mask = all_assign == k
        if mask.sum() == 0:
            continue
        az, _ = dip_dir_of(all_nrm[mask])
        h, _ = np.histogram(az, bins=bins)
        ax.bar(np.radians(np.linspace(0, 360, 36)), h, width=2 * np.pi / 36,
               color=cmap(k % 10), alpha=0.55, align="edge", label=f"G{k + 1}")
    ax.set_title("FORGE group rose (dip-direction)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stereonet(path, wells_nrm, assign, centers=None):
    """Stereonet (equal-angle) scatter of group-colored poles + group centers."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, 8)),
                      labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_rgrids([30, 60], labels=["30°", "60°"], fontsize=8)
    cmap = plt.cm.tab10
    assign = [np.asarray(a, int) for a in assign]
    all_nrm = np.vstack([np.asarray(w, float) for w in wells_nrm])
    all_assign = np.concatenate(assign)
    _, az = dip_dir_of(all_nrm)
    dip, _ = dip_dir_of(all_nrm)  # dip=arccos(|nz|)
    # pole plunge = 90 - dip ( lowers ), theta = dip direction
    plunge = 90.0 - dip  # 0 = horizontal plane pole at center, 90 = vertical at rim
    K = int(all_assign.max()) + 1 if len(all_assign) else 0
    for k in range(K):
        mask = all_assign == k
        if mask.sum() == 0:
            continue
        ax.scatter(np.radians(az[mask]), plunge[mask],
                   s=3, color=cmap(k % 10), alpha=0.4, label=f"G{k + 1}")
    if centers is not None:
        centers = np.asarray(centers, float)
        caz, cdip = dip_dir_of(centers)
        cplunge = 90.0 - cdip
        ax.scatter(np.radians(caz), cplunge, s=80, color="black",
                   marker="*", zorder=5, label="centers")
    ax.set_title("FORGE stereonet (poles)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
PT = os.path.join(_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells.pt")
RES = os.path.join(_ROOT, "results")
OUT = os.path.join(_ROOT, "data/external/utah_forge_fmi")

MASK_RNG = 999
OBS = 0.4
KM_SEED = 42


def make_mask(n):
    return np.random.default_rng(MASK_RNG).random(n) < OBS


def acos_err(pred, true):
    """隐伏点角误差 (度) = acos(|<pred,true>|), 向量化."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    c = np.clip(np.abs(np.einsum("ij,ij->i", pred, true)), -1, 1)
    return np.rad2deg(np.arccos(c))


def load_forge():
    d = torch.load(PT, weights_only=False)
    return d["nets"], d["widx"], d.get("summary", {})


def eval_assign(nets, assigns):
    """nets(list) + assigns(list, 与 nets 对齐) -> 隐伏点 MAE 等. 跨井等权合并."""
    errs = []
    per = []
    for net, sid in zip(nets, assigns):
        nrm = np.asarray(net.get("nrm_full", net.get("nrm")), float)
        pos = np.asarray(net["pos"], float)
        occ = make_mask(len(nrm))
        dirs, _ = set_aware_dirs(pos, nrm, occ, sid, strict=True)
        e = acos_err(dirs, nrm)            # 度, 长度 n
        hid = ~occ
        errs.append(e[hid])
        per.append(dict(wid=net.get("wid", "pooled"), mae=float(e[hid].mean()),
                        p50=float(np.median(e[hid])),
                        p90=float(np.percentile(e[hid], 90)),
                        n_hidden=int(hid.sum())))
    allv = np.concatenate(errs)
    agg = dict(mae=float(allv.mean()), p50=float(np.median(allv)),
               p90=float(np.percentile(allv, 90)),
               rmse=float(np.sqrt((allv ** 2).mean())),
               n_hidden=int(len(allv)))
    return agg, per


def survey(nets):
    """Phase 1.4 走向勘察 + 共享组系判定."""
    wells_nrm = [np.asarray(n["nrm_full"], float) for n in nets]
    wells_wid = [n["wid"] for n in nets]
    info = {}
    centers_all = []
    for n in nets:
        nrm = np.asarray(n["nrm_full"], float)
        az, dip = dip_dir_of(nrm)
        # 主走向: 球面 k-means K=4 取主导组心
        c, _ = spherical_kmeans(nrm, 4, seed=KM_SEED)
        azc, dipc = dip_dir_of(c)
        order = np.argsort(-np.linalg.norm(c, axis=1))  # 任意, 仅展示
        peaks = sorted([(float(a), float(d)) for a, d in zip(azc, dipc)],
                       key=lambda x: -float(np.sum(np.abs(nrm @ _ud(_nd(x[0], x[1]))))))
        info[n["wid"]] = dict(
            n=len(nrm),
            dip_median=float(np.median(dip)),
            dip_dir_median=float(np.median(az)),
            dip_dir_mean_circ=float(circmean(az)),
            peaks=[dict(dip_dir=float(a), dip=float(d)) for a, d in peaks],
        )
        centers_all.append(c)
    # 共享组系判定: 两井主走向(圆均值)差 + 倾角中位差
    wids = list(info.keys())
    azm = {w: info[w]["dip_dir_mean_circ"] for w in wids}
    dpm = {w: info[w]["dip_median"] for w in wids}
    if len(wids) >= 2:
        diff_az = abs(circ_diff(azm[wids[0]], azm[wids[1]]))
        diff_dip = abs(dpm[wids[0]] - dpm[wids[1]])
    else:
        diff_az = diff_dip = None
    shared = bool(diff_az is not None and diff_az <= 15.0 and diff_dip <= 15.0)
    verdict = ("共享组系前提成立 -> 多井联合聚类有效" if shared else
               "共享组系前提不成立(走向/倾角差异超阈值) -> 主用单井独立, 联合作对照实验")
    info["_verdict"] = dict(
        diff_dip_dir_deg=None if diff_az is None else float(diff_az),
        diff_dip_deg=None if diff_dip is None else float(diff_dip),
        shared_group_system=shared,
        recommendation=verdict,
    )
    return info


def _nd(dip_dir, dip):
    """dip_dir(deg,倾向) + dip(deg) -> 单位法向 (East,North,Up).

    与 read_forge_las.dip_dipdir_to_normal 同口径 (T58 后):
      nz = cos(dip), 水平分量幅值 = sin(dip).
      历史 bug (2026-08-28 修复): 此处曾把 sin/cos 互换, dip 被镜像成
      90°-dip, 导致走向峰值锚点档全部建立在镜像法向上.
    """
    a = np.deg2rad(dip_dir); d = np.deg2rad(dip)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def _ud(v):
    v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)


def circmean(deg):
    r = np.deg2rad(deg)
    d = np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r))))
    return float(d % 360.0)


def circ_diff(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def phase1_survey(nets):
    info = survey(nets)
    with open(os.path.join(OUT, "forge_fmi_survey.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, allow_nan=False)
    # 玫瑰图对比 (两井)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, 8)),
                      labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    cmap = plt.cm.tab10
    for i, n in enumerate(nets):
        az, _ = dip_dir_of(np.asarray(n["nrm_full"], float))
        h, _ = np.histogram(az, bins=np.linspace(0, 360, 37))
        ax.bar(np.radians(np.linspace(0, 360, 36)), h, width=2 * np.pi / 36,
               color=cmap(i), alpha=0.55, align="edge", label=n["wid"])
    ax.set_title("FORGE dip-direction rose: 16B vs 58-32", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forge_fmi_survey.png"), dpi=150)
    plt.close(fig)
    print("[1.4] survey verdict:", info["_verdict"]["recommendation"])
    print(f"      az-diff={info['_verdict']['diff_dip_dir_deg']:.1f}  dip-diff={info['_verdict']['diff_dip_deg']:.1f}")
    return info


def phase2_multiwell(nets):
    """2.1 单井 / 多井联合 / 井内模态 四对照."""
    nrm16 = np.asarray(nets[0]["nrm_full"], float)
    nrm58 = np.asarray(nets[1]["nrm_full"], float)
    wells_nrm = [nrm16, nrm58]
    # A 单井独立 (auto K)
    a_assign = [generate_set_ids(w, Krange=(2, 7), seed=KM_SEED) for w in wells_nrm]
    a_agg, a_per = eval_assign(nets, a_assign)
    # 联合 K 选取: 取两井 auto K 之和封顶, 或固定 8. 用 survey peaks 数
    K = 8
    jres = joint_set_ids(wells_nrm, K=K, seed=KM_SEED)
    j_assign = jres["assign"]  # list per well
    # B 多井联合 + 全局组心 (池化观测)
    pooled_net = dict(
        pos=np.vstack([nets[0]["pos"], nets[1]["pos"]]),
        nrm_full=np.vstack([nets[0]["nrm_full"], nets[1]["nrm_full"]]),
    )
    pool_assign = np.concatenate(j_assign)
    b_agg, b_per = eval_assign([pooled_net], [pool_assign])
    # C 联合聚类 + 井内模态 (每井单独解码, 用联合 label)
    c_assign = j_assign
    c_agg, c_per = eval_assign(nets, c_assign)
    # D 单井但 K=对方峰值数 (简单对照): 用 K=6
    d_assign = [generate_set_ids(w, Krange=(2, 6), seed=KM_SEED) for w in wells_nrm]
    d_agg, d_per = eval_assign(nets, d_assign)
    out = dict(
        protocol=dict(obs_frac=OBS, mask_rng=MASK_RNG, km_seed=KM_SEED, metric="acos(|<pred,true>|) deg, hidden points"),
        joint_K=K,
        A_single_well=dict(agg=a_agg, per=a_per),
        B_joint_global=dict(agg=b_agg, per=b_per),
        C_joint_cluster_perwell_mode=dict(agg=c_agg, per=c_per),
        D_single_well_K6=dict(agg=d_agg, per=d_per),
        verdict=("联合(全局/井内)优于或等于单井 -> 多井联合有效" if
                 min(b_agg["mae"], c_agg["mae"]) <= a_agg["mae"] + 0.5 else
                 "多井联合未优于单井(组系差异/采样偏差) -> 主用单井, 联合仅作对照"),
    )
    with open(os.path.join(RES, "forge_multiwell_real.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("[2.1] A single-well MAE=%.2f | B joint-global=%.2f | C joint-cluster/perwell=%.2f | D K6=%.2f"
          % (a_agg["mae"], b_agg["mae"], c_agg["mae"], d_agg["mae"]))
    return out


def _mean_obs_per_group(assigns, occ_list):
    """每组平均观测点数 (组内点数 <30 开始警惕噪声)."""
    sizes = []
    for sid, occ in zip(assigns, occ_list):
        sid = np.asarray(sid)
        occ = np.asarray(occ)
        for g in np.unique(sid):
            sizes.append(int(((sid == g) & occ).sum()))
    return float(np.mean(sizes)) if sizes else 0.0


def phase2_kgrid(nets):
    """2.2 K 网格 + 锚点 (扩展 K∈{6,8,10,12,14,16,18,20})."""
    wells_nrm = [np.asarray(n["nrm_full"], float) for n in nets]
    occ_list = [make_mask(len(n["nrm_full"])) for n in nets]
    # 走向峰值 -> 锚点法向
    svm = survey(nets)
    peaks = []
    for w in nets:
        for p in svm[w["wid"]]["peaks"]:
            peaks.append(_nd(p["dip_dir"], p["dip"]))
    anchors = np.array(peaks)
    rows = []
    for K in [6, 8, 10, 12, 14, 16, 18, 20]:
        j = joint_set_ids(wells_nrm, K=K, seed=KM_SEED)
        agg, _ = eval_assign(nets, j["assign"])
        mpg = _mean_obs_per_group(j["assign"], occ_list)
        rows.append(dict(K=K, variant="joint_perwell", mae=agg["mae"], p90=agg["p90"],
                         mean_obs_per_group=mpg))
        # 锚点: 用该 K 个峰值 (截断/补)
        if len(anchors) >= K:
            anc = anchors[:K]
        else:
            anc = np.vstack([anchors, wells_nrm[0][:K - len(anchors)]])
        ja = joint_set_ids(wells_nrm, K=K, seed=KM_SEED, anchors=anc)
        agga, _ = eval_assign(nets, ja["assign"])
        mpga = _mean_obs_per_group(ja["assign"], occ_list)
        rows.append(dict(K=K, variant="anchored", mae=agga["mae"], p90=agga["p90"],
                         mean_obs_per_group=mpga))
    out = dict(rows=rows,
               note="anchored = 走向峰值法向作 kmeans 初始化锚点; "
                    "mean_obs_per_group = 联合聚类后每组平均观测点数")
    with open(os.path.join(RES, "forge_kgrid_ext.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    # 曲线 + knee 标注
    _plot_kgrid_curve(out, os.path.join(RES, "forge_kgrid_ext.png"))
    best = min(rows, key=lambda r: r["mae"])
    print("[2.2] K-grid best: K=%d %s MAE=%.2f (mean_obs/group=%.0f)"
          % (best["K"], best["variant"], best["mae"], best["mean_obs_per_group"]))
    return out


def _plot_kgrid_curve(out, path):
    Ks = sorted({r["K"] for r in out["rows"]})
    j_mae = [min(r["mae"] for r in out["rows"] if r["K"] == K and r["variant"] == "joint_perwell") for K in Ks]
    a_mae = [min(r["mae"] for r in out["rows"] if r["K"] == K and r["variant"] == "anchored") for K in Ks]
    j_mpg = [min(r["mean_obs_per_group"] for r in out["rows"] if r["K"] == K and r["variant"] == "joint_perwell") for K in Ks]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(Ks, j_mae, "o-", color="steelblue", label="joint_perwell MAE")
    ax1.plot(Ks, a_mae, "s--", color="darkorange", label="anchored MAE")
    ax1.axhline(9.0, color="red", ls=":", lw=1, label="9° 商用线")
    ax1.set_xlabel("K (组数)"); ax1.set_ylabel("隐伏点 MAE (°)")
    ax1.set_xticks(Ks)
    ax2 = ax1.twinx()
    ax2.plot(Ks, j_mpg, "^:", color="gray", label="每组平均观测点数")
    ax2.set_ylabel("每组平均观测点数", color="gray")
    ax1.set_title("FORGE kgrid: MAE vs K (rng=999, 40% obs)")
    ax1.legend(loc="upper right", fontsize=8); ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("[2.2] curve -> %s" % path)


def phase2_kgrid_robust(nets, best_K=None, best_variant="anchored"):
    """A2: 多掩码稳健性闸门. 固定方法(默认 K=12 anchored), 换掩码 seed 跑 10 遍.
    要求: 10 掩码均值 <=9.5° 且所有 <10.5° 才许写"达标/逼近"."""
    if best_K is None:
        # 取 ext 曲线里最优 K
        ext = json.load(open(os.path.join(RES, "forge_kgrid_ext.json"), encoding="utf-8"))
        best = min(ext["rows"], key=lambda r: r["mae"])
        best_K, best_variant = best["K"], best["variant"]
    wells_nrm = [np.asarray(n["nrm_full"], float) for n in nets]
    svm = survey(nets)
    peaks = []
    for w in nets:
        for p in svm[w["wid"]]["peaks"]:
            peaks.append(_nd(p["dip_dir"], p["dip"]))
    anchors = np.array(peaks)
    seeds = [999, 1001, 2003, 3001, 4007, 5009, 6007, 7001, 8009, 9001]
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        # 重新构造 mask: 用通用 obs_frac; make_mask 已锁 rng=999, 故此处手动覆盖
        def mask_fn(n):
            return rng.random(n) < OBS
        j = joint_set_ids(wells_nrm, K=best_K, seed=KM_SEED,
                          anchors=(anchors[:best_K] if best_variant == "anchored" and len(anchors) >= best_K else None))
        # 重算 eval_assign 但用本 seed 掩码
        errs = []
        for net, sid in zip(nets, j["assign"]):
            nrm = np.asarray(net.get("nrm_full", net.get("nrm")), float)
            pos = np.asarray(net["pos"], float)
            occ = mask_fn(len(nrm))
            dirs, _ = set_aware_dirs(pos, nrm, occ, sid, strict=True)
            e = acos_err(dirs, nrm)
            errs.append(e[~occ])
        allv = np.concatenate(errs)
        rows.append(dict(mask_seed=sd, mae=float(allv.mean()), p90=float(np.percentile(allv, 90))))
    maes = np.array([r["mae"] for r in rows])
    passed = bool(maes.mean() <= 9.5 and maes.max() < 10.5)
    out = dict(best_K=best_K, best_variant=best_variant,
               n_masks=len(seeds), mae_mean=float(maes.mean()),
               mae_std=float(maes.std()), mae_min=float(maes.min()),
               mae_max=float(maes.max()), p90_mean=float(np.mean([r["p90"] for r in rows])),
               verdict=("达标/逼近 (10掩码均值<=9.5且全<10.5)" if passed else "未达标, 如实写区间"),
               passed=passed, rows=rows)
    with open(os.path.join(RES, "forge_kgrid_robust.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("[A2] K=%d %s over %d masks: mean=%.2f±%.2f min=%.2f max=%.2f -> %s"
          % (best_K, best_variant, len(seeds), maes.mean(), maes.std(),
             maes.min(), maes.max(), "PASS" if passed else "FAIL"))
    return out


def phase2_typegate(nets):
    """2.3 类型门控复测 (natural vs induced vs 全部 vs 类型隔离)."""
    rows = []
    for n in nets:
        nrm = np.asarray(n["nrm_full"], float)
        ft = np.array(n["ftype"])
        nat = ft == "natural"
        ind = ft == "induced"
        occ = make_mask(len(nrm))
        # 全部
        sid_all = generate_set_ids(nrm, Krange=(2, 7), seed=KM_SEED)
        d_all, _ = set_aware_dirs(n["pos"], nrm, occ, sid_all, strict=True)
        # natural only
        if nat.sum() > 8:
            sid_nat = generate_set_ids(nrm[nat], Krange=(2, 7), seed=KM_SEED)
            # 把 induced 归到最近 natural 组
            full = np.zeros(len(nrm), int)
            full[nat] = sid_nat
            cand = np.unique(sid_nat)
            if len(cand):
                cen = [np.mean(_sign(nrm[nat][sid_nat == k]), 0) for k in cand]
                cen = np.array([c / (np.linalg.norm(c) + 1e-12) for c in cen])
                for i in np.where(ind)[0]:
                    full[i] = cand[np.abs(nrm[i] @ cen.T).argmax()]
            d_nat, _ = set_aware_dirs(n["pos"], nrm, occ, full, strict=True)
            mae_nat = float(acos_err(d_nat, nrm)[~occ].mean())
        else:
            mae_nat = None
        # induced only
        if ind.sum() > 8:
            sid_ind = generate_set_ids(nrm[ind], Krange=(2, 4), seed=KM_SEED)
            full = np.zeros(len(nrm), int)
            full[ind] = sid_ind
            cand = np.unique(sid_ind)
            if len(cand):
                cen = [np.mean(_sign(nrm[ind][sid_ind == k]), 0) for k in cand]
                cen = np.array([c / (np.linalg.norm(c) + 1e-12) for c in cen])
                for i in np.where(nat)[0]:
                    full[i] = cand[np.abs(nrm[i] @ cen.T).argmax()]
            d_ind, _ = set_aware_dirs(n["pos"], nrm, occ, full, strict=True)
            mae_ind = float(acos_err(d_ind, nrm)[~occ].mean())
        else:
            mae_ind = None
        # 类型隔离 (natural/induced 不同组空间)
        sid_iso = np.zeros(len(nrm), int)
        if nat.sum() > 4:
            sid_iso[nat] = generate_set_ids(nrm[nat], Krange=(2, 7), seed=KM_SEED)
        off = int(sid_iso[nat].max()) + 1 if nat.sum() else 0
        if ind.sum() > 4:
            sid_iso[ind] = generate_set_ids(nrm[ind], Krange=(2, 4), seed=KM_SEED) + off
        d_iso, _ = set_aware_dirs(n["pos"], nrm, occ, sid_iso, strict=True)
        mae_iso = float(acos_err(d_iso, nrm)[~occ].mean())
        mae_all = float(acos_err(d_all, nrm)[~occ].mean())
        rows.append(dict(wid=n["wid"], n=len(nrm), n_natural=int(nat.sum()),
                         n_induced=int(ind.sum()),
                         mae_all=mae_all, mae_natural_only=mae_nat,
                         mae_induced_only=mae_ind, mae_type_isolated=mae_iso))
    out = dict(rows=rows,
               note="类型隔离=natural/induced 强制分入不同组空间; 单类基线=该类定组其余点投最近组")
    with open(os.path.join(RES, "forge_typegate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    for r in rows:
        print("[2.3] %s all=%.2f nat=%.2f ind=%.2f iso=%.2f"
              % (r["wid"], r["mae_all"], r["mae_natural_only"] or -1,
                 r["mae_induced_only"] or -1, r["mae_type_isolated"]))
    return out


def _sign(v):
    v = np.asarray(v, float)
    ref = v[0]; s = np.sign((v * ref).sum(-1, keepdims=True)); s[s == 0] = 1
    return v * s


def phase2_vs_loaded():
    """2.4 与既有路线 A (loaded_real_nets_setid.pt) 对比.

    R268 T4: 这里曾把两个**判废参照值**当 reference 字段与提示打印出来 —— 它们是
    八期 `_nd` sin/cos 修复前口径的快照, 按 docs/八期新旧数字对照总表.md 已判废; 出现在产品
    脚本的控制台输出里 = 客户会抄进自己报告 (docs/伽马线_主项目关账报告_20260901.md 已经在
    解释"与 ref 的差距是预存口径差", 就是被这两行提示带出来的)。生成器不再重发该字段;
    历史对照值仍可在冻结证据件 results/forge_vs_loaded.json#reference 与其 _version_note 查到
    (该件的 "ref 字段保留原值不删, 待架构师裁决" 声明原样有效, 本单未重跑、未覆写)。
    """
    d = torch.load(os.path.join(_ROOT, "data/real/loaded_real_nets_setid.pt"), weights_only=False)
    nets = [n for n in d if ("nrm_full" in n or "nrm" in n)]
    # 标准 K (2,7)
    a_assign = [generate_set_ids(np.asarray(n.get("nrm_full", n["nrm"]), float), Krange=(2, 7), seed=KM_SEED) for n in nets]
    agg7, _ = eval_assign(nets, a_assign)
    # K<=12 细分
    a_assign12 = [generate_set_ids(np.asarray(n.get("nrm_full", n["nrm"]), float), Krange=(2, 12), seed=KM_SEED) for n in nets]
    agg12, _ = eval_assign(nets, a_assign12)
    out = dict(loaded_real=dict(standard_K_mae=agg7["mae"], fine_K12_mae=agg12["mae"]),
               note="复现既有路线 A 基线, 供 FORGE 真实多井横向对比; "
                    "八期前口径参照值已判废, 不再由本生成器重发 (R268 T4)")
    with open(os.path.join(RES, "forge_vs_loaded.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("[2.4] loaded_real stdK=%.2f | K12=%.2f (横向对比档, 无对外参照值)"
          % (agg7["mae"], agg12["mae"]))
    return out


def phase3_report(nets, survey_info):
    """3 组系表 + 玫瑰图 + 赤平投影 + 置信度 (真实 FORGE 数据)."""
    rep = os.path.join(OUT, "forge_report_v2")
    os.makedirs(rep, exist_ok=True)
    wells_nrm = [np.asarray(n["nrm_full"], float) for n in nets]
    wells_wid = [n["wid"] for n in nets]
    # 联合分组 (用 survey 推荐 K 或 8)
    K = 8
    jres = joint_set_ids(wells_nrm, K=K, seed=KM_SEED)
    assign = jres["assign"]
    # 组系表 (合并 + 每井深度/类型)
    rows = group_table(wells_nrm, assign,
                       depth=np.concatenate([nets[i]["md_m"] for i in range(len(nets))]),
                       ftype=np.concatenate([nets[i]["ftype"] for i in range(len(nets))]))
    write_group_table_csv(rows, os.path.join(rep, "group_table.csv"))
    # 玫瑰图 (按组着色)
    rose_diagram(os.path.join(rep, "rose.png"), wells_nrm, assign)
    # 赤平投影 (组心)
    centers = jres["centers"]
    stereonet(os.path.join(rep, "stereonet.png"), wells_nrm, assign, centers=centers)
    # 置信度: 每点 propagation_confidence (按井归一化坐标, 消除 ft 尺度饱和)
    conf_all = []
    for i, n in enumerate(nets):
        occ = make_mask(len(nets[i]["nrm_full"]))
        pos = np.asarray(nets[i]["pos"], float)
        mu, sd = pos.mean(0), pos.std(0) + 1e-9
        c = propagation_confidence((pos - mu) / sd, occ)
        conf_all.append(c)
    conf = np.concatenate(conf_all)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conf, bins=30, color="steelblue", alpha=0.8)
    ax.set_xlabel("propagation confidence (0-1)")
    ax.set_ylabel("count")
    ax.set_title("FORGE per-point prediction confidence", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(rep, "confidence.png"), dpi=150)
    plt.close(fig)
    out = dict(report_dir=rep, K=K, group_table=rows,
               confidence_mean=float(np.nanmean(conf)),
               confidence_median=float(np.nanmedian(conf)))
    with open(os.path.join(rep, "report_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("[3] report -> %s (groups=%d, conf_mean=%.2f)" % (rep, len(rows), out["confidence_mean"]))
    return out


def main():
    nets, widx, summ = load_forge()
    print("Loaded FORGE 2 wells: %d fractures" % sum(len(n["nrm_full"]) for n in nets))
    phase1_survey(nets)
    phase2_multiwell(nets)
    phase2_kgrid(nets)
    phase2_kgrid_robust(nets)
    phase2_typegate(nets)
    phase2_vs_loaded()
    phase3_report(nets, None)
    print("\nDONE. JSON in results/forge_*.json ; figures in data/external/utah_forge_fmi/")


if __name__ == "__main__":
    main()
