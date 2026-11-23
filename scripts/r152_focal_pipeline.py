# -*- coding: utf-8 -*-
"""R156 T3/T4 · r152_focal_pipeline.py — 产状预检 + 观测-only 两臂管线 (standalone).

判据/口径 (R152 v1.1 §2 T3/T4 + §4, 经 prereg.json 零改动继承, 2 项 amendment 见该文件):
  T3 预检闸: 每区域走向展布 >=30 度 + 事件数 >=80, 不过闸换区域禁硬做;
  T4 两臂:   观测-only 球面 k-means (K 网格 2..8) + 组系表 modal_err
             + 逐点 acos 均值/p50/p90;
             oracle (全量组心) vs honest (40% 观测掩码, 事件级, rng 固定, 10 种子)。

镜像冻结口径 (standalone 重实现, 不 import/修改 src/fractureflow):
  spherical_kmeans / _sign_align / _silhouette  <- src/fractureflow/setlabel.py
  make_blind 掩码约定 (rng.random(L) < obs_frac) <- src/fractureflow/honest_eval.py
  组系表匈牙利匹配 + modal_err + coverage@30    <- src/fractureflow/set_table_eval.py
  10 掩码种子 999+1000s + kmeans seed=42        <- scripts/t54_beishan_kmeans_obs_k12.py

红线: 零训练/零新依赖 (numpy+scipy); 输出 JSON allow_nan=False; 只写
results/r152_focal_migration/ 下新文件; 确定性: 无墙钟时间戳, 双跑逐位一致。

冻结状态 (R156 收刀关账, 2026-09-04):
  - 冻结日期: 2026-09-04
  - 确定性双跑已证: results/r152_focal_migration/determinism_report.json
  - prereg sha256: 5cec2b24351de1ea0c37557bab55e71c9cacbdf80583a80d1516d0edc0a2ed21
  - 本域已收刀, 不再投入新单
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT_DIR = "results/r152_focal_migration"
PREREG_PATH = os.path.join(OUT_DIR, "prereg.json")
GCMT_CSV = os.path.join(OUT_DIR, "gcmt_events.csv")
NCSN_CSV = os.path.join(OUT_DIR, "ncsn_events.csv")

KMEANS_SEED = 42
ITERS = 80


# ---------------------------------------------------------------------------
# 几何 (与 r152_catalog_adapter.py 同式; 编录表同构 (dip, dip_dir) -> ENU 法向)
# ---------------------------------------------------------------------------
def orientation_to_normal(dip_dir_deg, dip_deg):
    a = np.radians(np.asarray(dip_dir_deg, dtype=np.float64))
    d = np.radians(np.asarray(dip_deg, dtype=np.float64))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def normal_to_orientation(n):
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    dip = np.degrees(np.arccos(np.clip(n[..., 2], -1.0, 1.0)))
    dip_dir = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360.0
    return dip_dir, dip


def strike_axial(strike_deg):
    """走向折叠到 [0,180) (轴向: s 与 s+180 同面)."""
    return np.asarray(strike_deg, dtype=np.float64) % 180.0


def center_strike(c):
    dd, dp = normal_to_orientation(c)
    return float(strike_axial(dd - 90.0))


# ---------------------------------------------------------------------------
# 冻结口径 port: 球面 k-means + 轮廓系数 (src/fractureflow/setlabel.py)
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _sign_align(pts):
    """按首向量符号对齐 (无向法向), [M,3] -> [M,3]. (port of setlabel._sign_align)"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def spherical_kmeans(nrm, K, iters=ITERS, seed=KMEANS_SEED):
    """port of setlabel.spherical_kmeans: |cos| 指派, 符号对齐均值组心, iters=80."""
    rng = np.random.default_rng(seed)
    pts = _unit(nrm)
    n = len(pts)
    if n == 0:
        raise ValueError("spherical_kmeans: 空法向输入 (R80 PP-B3 响亮拒绝)")
    K = min(K, n)
    if K <= 1:
        c = _unit(_sign_align(pts).mean(0))[None]
        return c, np.zeros(n, dtype=int)
    centers = pts[rng.choice(n, K, replace=False)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        cos = np.abs(pts @ centers.T)
        assign = cos.argmax(1)
        for k in range(K):
            sel = pts[assign == k]
            if len(sel):
                centers[k] = _unit(_sign_align(sel).mean(0))
    return centers, assign


def silhouette(nrm, centers, assign):
    """port of setlabel._silhouette: 基于夹角 (无向 |cos|) 的轮廓系数均值."""
    pts = _unit(nrm)
    K = len(centers)
    abscos = np.abs(pts @ centers.T)
    a = np.zeros(len(pts))
    b = np.full(len(pts), -np.inf)
    for k in range(K):
        sel = assign == k
        if sel.sum() <= 1:
            a[sel] = 0.0
            continue
        a[sel] = 1.0 - abscos[sel, k]
    others = [k for k in range(K) if (assign == k).sum() > 1]
    for k in range(K):
        sel = assign == k
        if not sel.any() or len(others) <= 1:
            b[sel] = 0.0
            continue
        oc = np.full(len(pts), -np.inf)
        for k2 in others:
            if k2 == k:
                continue
            oc = np.maximum(oc, abscos[:, k2])
        b[sel] = 1.0 - oc[sel]
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-9)
    return float(np.nanmean(s))


# ---------------------------------------------------------------------------
# 组系表对比 (port of set_table_eval: 匈牙利匹配 + modal_err + coverage@30)
# ---------------------------------------------------------------------------
def strike_circle_dist(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def match_consensus(pred_strikes, consensus_axial):
    """匈牙利匹配 (代价 = 走向圆距离), 返回 (matched_pairs, modal_err, coverage30)."""
    Kp, Kt = len(pred_strikes), len(consensus_axial)
    if Kp == 0 or Kt == 0:
        return [], None, 0.0
    cost = np.zeros((Kp, Kt))
    for i in range(Kp):
        for j in range(Kt):
            cost[i, j] = strike_circle_dist(pred_strikes[i], consensus_axial[j])
    ri, ci = linear_sum_assignment(cost)
    pairs, errs = [], []
    for i, j in zip(ri.tolist(), ci.tolist()):
        e = float(cost[i, j])
        errs.append(e)
        pairs.append({"cluster": i, "consensus_j": j, "err_deg": round(e, 4)})
    modal_err = float(np.mean(errs)) if errs else None
    cov = sum(1 for e in errs if e < 30.0) / float(Kt)
    return pairs, modal_err, cov


# ---------------------------------------------------------------------------
# 数据加载 + 预注册过滤
# ---------------------------------------------------------------------------
def load_prereg():
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        prereg = json.load(f)
    return prereg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_box(lat, lon, box):
    return (box["lat_range"][0] <= lat <= box["lat_range"][1]
            and box["lon_range"][0] <= lon <= box["lon_range"][1])


def region_of(lat, lon, regions):
    for r in regions:
        if in_box(lat, lon, r):
            return r["id"]
    return None


def load_gcmt(regions):
    """GCMT 解析表 -> {region: [(nrm, evid), ...]}; 逐平面退化排除 (amendment#1)."""
    out = {r["id"]: [] for r in regions}
    out["_stats"] = {"rows": 0, "planes_kept": 0, "planes_dropped_dip0": 0}
    with open(GCMT_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out["_stats"]["rows"] += 1
            rid = region_of(float(row["c_lat"]), float(row["c_lon"]), regions)
            if rid is None:
                continue
            for i in (1, 2):
                dip = float(row[f"dip{i}"])
                if dip <= 0.0:
                    out["_stats"]["planes_dropped_dip0"] += 1
                    continue
                nrm = orientation_to_normal(
                    (float(row[f"strike{i}"]) + 90.0) % 360.0, dip)
                out[rid].append((nrm, row["evid"]))
                out["_stats"]["planes_kept"] += 1
    return out


def load_ncsn(regions):
    """NCSN 解析表 -> 预注册过滤 -> {region: [(nrm_plane1, nrm_plane2, evid), ...]}."""
    out = {r["id"]: [] for r in regions}
    out["_stats"] = {"rows": 0, "region_rows": 0, "kept": 0,
                     "drop_misfit": 0, "drop_nfm": 0, "drop_conv": 0,
                     "drop_dip": 0}
    with open(NCSN_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out["_stats"]["rows"] += 1
            rid = region_of(float(row["lat"]), float(row["lon"]), regions)
            if rid is None:
                continue
            out["_stats"]["region_rows"] += 1
            misfit = row["misfit1"]
            nfm = row["nfm1"]
            conv = row["conv1"]
            dip1 = float(row["dip1"])
            if not misfit or float(misfit) > 0.30:
                out["_stats"]["drop_misfit"] += 1
                continue
            if not nfm or float(nfm) < 10.0:
                out["_stats"]["drop_nfm"] += 1
                continue
            if conv == "C":
                out["_stats"]["drop_conv"] += 1
                continue
            if not (0.0 < dip1 <= 90.0):
                out["_stats"]["drop_dip"] += 1
                continue
            n1 = orientation_to_normal(
                (float(row["strike1"]) + 90.0) % 360.0, dip1)
            dip2 = float(row["dip2"])
            n2 = (orientation_to_normal(
                (float(row["strike2"]) + 90.0) % 360.0, dip2)
                if dip2 > 0.0 else None)
            out[rid].append((n1, n2, row["evid"]))
            out["_stats"]["kept"] += 1
    return out


# ---------------------------------------------------------------------------
# T3 预检
# ---------------------------------------------------------------------------
def axial_spread_deg(strikes_axial):
    """轴向圆离散 (prereg 定义, 单位修正后):

    倍角映射 2θ 上取 R2 = |mean e^{i2θ}|, 圆标准差 σ2 = sqrt(-2 ln R2) [rad],
    折回走向空间: spread_deg = degrees(σ2)/2。
    自检: strike∈[15,75] 均匀 (线性 std 17.3°) -> R2=0.827 -> spread=17.6° ✓
    集中 -> R2→1 -> spread→0; 均匀 -> R2→0 -> spread→∞。
    """
    th = np.radians(np.asarray(strikes_axial, dtype=np.float64) * 2.0)
    R2 = float(np.abs(np.mean(np.exp(1j * th))))
    if R2 >= 1.0:
        return 0.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(R2, 1e-300)))) / 2.0)


def precheck(name, n_events, strikes_axial, gate):
    spread = axial_spread_deg(strikes_axial) if n_events else 0.0
    ok = bool(n_events >= gate["min_events"]
              and spread >= gate["strike_spread_min_deg"])
    return {
        "catalog_region": name, "n_events": int(n_events),
        "strike_spread_deg": round(spread, 4),
        "gate": {"min_events": gate["min_events"],
                 "strike_spread_min_deg": gate["strike_spread_min_deg"]},
        "pass": ok,
    }


# ---------------------------------------------------------------------------
# T4 两臂
# ---------------------------------------------------------------------------
def kscan(pts, k_grid, consensus):
    """K 网格扫描: 每球面 k-means (seed=42) + 轮廓系数; 返回逐 K 记录与选中 K."""
    recs = []
    best_k, best_s = None, -np.inf
    for K in k_grid:
        if K > len(pts):
            break
        centers, assign = spherical_kmeans(pts, K, seed=KMEANS_SEED)
        s = silhouette(pts, centers, assign)
        recs.append({"K": K, "silhouette": round(s, 6)})
        if s > best_s:
            best_s, best_k = s, K
    return recs, best_k, best_s


def arm_tables(centers, consensus):
    strikes = [center_strike(c) for c in centers]
    pairs, modal_err, cov = match_consensus(strikes, consensus)
    table = []
    for i, c in enumerate(centers):
        dd, dp = normal_to_orientation(c)
        table.append({"cluster": i, "dip": round(float(dp), 4),
                      "dip_dir": round(float(dd), 4),
                      "axial_strike": round(strike_axial(dd - 90.0), 4)})
    return table, pairs, modal_err, cov


def per_point_stats(nrm, centers, assign):
    abscos = np.abs(_unit(nrm) @ centers.T)
    err = np.degrees(np.arccos(np.clip(abscos[np.arange(len(nrm)), assign], 0, 1)))
    return {"mean": round(float(err.mean()), 4),
            "p50": round(float(np.percentile(err, 50)), 4),
            "p90": round(float(np.percentile(err, 90)), 4),
            "n": int(err.size)}


def run_region(rid, planes_by_event, consensus_axial, k_grid):
    """单区域两臂. planes_by_event: [(n1, n2_or_None, evid)]."""
    normals = []
    ev_idx = []
    for ei, (n1, n2, _) in enumerate(planes_by_event):
        normals.append(n1)
        ev_idx.append(ei)
        if n2 is not None:
            normals.append(n2)
            ev_idx.append(ei)
    normals = np.array(normals)
    ev_idx = np.array(ev_idx)
    n_events = len(planes_by_event)
    out = {"region": rid, "n_events": int(n_events),
           "n_planes": int(len(normals)), "kmeans_seed": KMEANS_SEED}

    # ---- oracle 臂: 全量组心 ----
    recs, K_oracle, sil_oracle = kscan(normals, k_grid, consensus_axial)
    centers_o, assign_o = spherical_kmeans(normals, K_oracle, seed=KMEANS_SEED)
    table_o, pairs_o, modal_o, cov_o = arm_tables(centers_o, consensus_axial)
    stats_o = per_point_stats(normals, centers_o, assign_o)
    out["oracle"] = {"k_scan": recs, "K_selected": int(K_oracle),
                     "silhouette": round(sil_oracle, 6),
                     "group_table": table_o, "matched_pairs": pairs_o,
                     "modal_err_deg": round(modal_o, 4) if modal_o is not None else None,
                     "coverage_at_30deg": round(cov_o, 4),
                     "per_plane_acos_err": stats_o}

    # ---- honest 臂: 事件级 40% 掩码 (make_blind 约定), 10 种子 ----
    seeds = []
    pooled_hidden = []
    modal_h_list = []
    for s in range(10):
        mask_seed = 999 + 1000 * s
        rng = np.random.default_rng(mask_seed)
        occ_ev = rng.random(n_events) < 0.40
        occ = occ_ev[ev_idx]
        obs = normals[occ]
        hid = normals[~occ]
        recs_h, K_h, sil_h = kscan(obs, k_grid, consensus_axial)
        centers_h, assign_h = spherical_kmeans(obs, K_h, seed=KMEANS_SEED)
        # 盲预测: 隐伏平面 |cos| 最近组心
        cos_h = np.abs(_unit(hid) @ centers_h.T)
        pred = cos_h.argmax(1)
        err = np.degrees(np.arccos(
            np.clip(cos_h[np.arange(len(hid)), pred], 0, 1)))
        table_h, pairs_h, modal_h, cov_h = arm_tables(centers_h, consensus_axial)
        modal_h_list.append(modal_h if modal_h is not None else 90.0)
        pooled_hidden.append(err)
        seeds.append({"mask_seed": mask_seed, "n_obs_planes": int(occ.sum()),
                      "n_hidden_planes": int((~occ).sum()),
                      "k_scan": recs_h, "K_selected": int(K_h),
                      "silhouette_obs": round(sil_h, 6),
                      "hidden_acos_err": {
                          "mean": round(float(err.mean()), 4),
                          "p50": round(float(np.percentile(err, 50)), 4),
                          "p90": round(float(np.percentile(err, 90)), 4)},
                      "group_table": table_h, "matched_pairs": pairs_h,
                      "modal_err_deg": round(modal_h, 4) if modal_h is not None else None,
                      "coverage_at_30deg": round(cov_h, 4)})
    pooled = np.concatenate(pooled_hidden)
    modal_arr = np.array([m for m in modal_h_list if m is not None])
    out["honest"] = {
        "mask_rule": "event-level: rng(mask_seed).random(n_events) < 0.40; planes follow event",
        "mask_seeds": [999 + 1000 * s for s in range(10)],
        "n_seeds": 10,
        "per_seed": seeds,
        "hidden_acos_err_agg": {
            "mean_of_seed_means": round(float(np.mean(
                [sd["hidden_acos_err"]["mean"] for sd in seeds])), 4),
            "std_of_seed_means": round(float(np.std(
                [sd["hidden_acos_err"]["mean"] for sd in seeds], ddof=1)), 4),
            "pooled_p50": round(float(np.percentile(pooled, 50)), 4),
            "pooled_p90": round(float(np.percentile(pooled, 90)), 4),
            "n_hidden_total": int(pooled.size)},
        "modal_err_deg_mean": round(float(modal_arr.mean()), 4),
        "modal_err_deg_std": round(float(modal_arr.std(ddof=1)), 4),
    }
    out["honest_minus_oracle"] = {
        "acos_mean_delta_deg": round(
            out["honest"]["hidden_acos_err_agg"]["mean_of_seed_means"]
            - stats_o["mean"], 4),
        "modal_err_delta_deg": round(
            out["honest"]["modal_err_deg_mean"] - modal_o, 4),
    }
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def r6(x):
    return round(float(x), 6)


def main():
    prereg = load_prereg()
    prereg_sha = sha256_file(PREREG_PATH)
    regions = prereg["regions"]
    gate = prereg["precheck_gate_T3"]
    k_grid = prereg["pipeline_T4"]["k_grid"]

    gcmt = load_gcmt(regions)
    ncsn = load_ncsn(regions)
    print("[load] gcmt stats:", gcmt["_stats"])
    print("[load] ncsn stats:", ncsn["_stats"])

    # ---- T3 预检 (两目录 x 三区域) ----
    prechecks = []
    consensus_axial = {}
    for r in regions:
        consensus_axial[r["id"]] = [float(s) % 180.0
                                    for s in r["consensus_axial_strike_deg"]]
    for cat_name, data in (("gcmt", gcmt), ("ncsn", ncsn)):
        for r in regions:
            rid = r["id"]
            rows = data[rid]
            sa = []
            if cat_name == "ncsn":
                # 走向展布用全部有效平面 (节面1 + 有效共轭), 轴向折叠
                for n1, n2, _ in rows:
                    dd, _ = normal_to_orientation(n1)
                    sa.append(strike_axial(dd - 90.0))
                    if n2 is not None:
                        dd2, _ = normal_to_orientation(n2)
                        sa.append(strike_axial(dd2 - 90.0))
            else:
                for pl, _ in rows:
                    dd, _ = normal_to_orientation(pl)
                    sa.append(strike_axial(dd - 90.0))
            pc = precheck(f"{cat_name}:{rid}", len(rows), sa, gate)
            prechecks.append(pc)
            print(f"[precheck] {pc['catalog_region']}: n={pc['n_events']} "
                  f"spread={pc['strike_spread_deg']:.2f} pass={pc['pass']}")

    with open(os.path.join(OUT_DIR, "precheck_regions.json"), "w",
              encoding="utf-8") as f:
        json.dump({"prereg_sha256": prereg_sha,
                   "prechecks": prechecks,
                   "stats": {"gcmt": gcmt["_stats"], "ncsn": ncsn["_stats"]}},
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print("[save] precheck_regions.json")

    # ---- T4 两臂 (过闸区域, 机械执行) ----
    pass_map = {pc["catalog_region"]: pc["pass"] for pc in prechecks}
    green_candidates = {}
    for r in regions:
        rid = r["id"]
        cons = consensus_axial[rid]
        if pass_map.get(f"ncsn:{rid}"):
            arms = run_region(rid, ncsn[rid], cons, k_grid)
            arms["catalog"] = "ncsn"
            arms["prereg_sha256"] = prereg_sha
            arms["consensus"] = {
                "axial_strike_deg": cons,
                "label": "文献共识档·非人工实测",
                "refs": r["consensus_refs"]}
            out_path = os.path.join(OUT_DIR, f"arms_{rid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(arms, f, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"[save] {out_path}")
            green_candidates[rid] = {
                "honest_modal_err_deg": arms["honest"]["modal_err_deg_mean"],
                "honest_minus_oracle_acos_deg":
                    arms["honest_minus_oracle"]["acos_mean_delta_deg"],
                "honest_minus_oracle_modal_deg":
                    arms["honest_minus_oracle"]["modal_err_delta_deg"],
                "oracle_modal_err_deg": arms["oracle"]["modal_err_deg"]}
        else:
            print(f"[skip] ncsn:{rid} failed precheck gate")

    # ---- 机械 GREEN 判定 (R152 T5 判据零改动) ----
    green_regions = [rid for rid, v in green_candidates.items()
                     if v["honest_modal_err_deg"] is not None
                     and v["honest_modal_err_deg"] <= 15.0]
    verdict_branch = "GREEN" if green_regions else "PENDING_RED_REVIEW"
    summary = {
        "prereg_sha256": prereg_sha,
        "green_candidates": green_candidates,
        "green_regions": green_regions,
        "verdict_branch": verdict_branch,
        "criteria": "GREEN: >=1 region obs-only (honest-arm) centers vs consensus strike groups modal_err <= 15 deg AND honest degradation quantifiable; else RED with three-question attribution (R152 v1.1 T5 verbatim)",
    }
    with open(os.path.join(OUT_DIR, "pipeline_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("[save] pipeline_summary.json")
    print("[verdict-branch]", verdict_branch, green_regions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
