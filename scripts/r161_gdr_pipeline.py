# -*- coding: utf-8 -*-
"""R161 · r161_gdr_pipeline.py — 真实地质域首轮: GDR DFN 三件 x 组系管线对照 (standalone).

判据/口径全部预注册于 results/r161_gdr_pipeline/prereg.json (零改动继承), 本文件只负责机械执行。

流程:
  T1 适配: 各源产状列 -> 轴向法向 (权威列 = Strike + Dip_Angle; Trend/Plunge 只作逐行校验,
          实测其半球符号逐行随机, 直接建法向会注入 ~50% 倾向翻转)
  G0 锚复现闸: 先用 FORGE FMI (ftype 真值, K=12, 40% 掩码, 10 seed) 复现既有锚
          11.0453 ± 2.1422; 不过闸 -> 停单 (退出码 3), 不出任何 GDR 数字
  T3 两臂: oracle (全量组心 = T_self 真值) vs honest (40% 掩码盲预测), 10 种子
  T4 同场地对照: 全量 K=4 组心, FORGE FMI 每井 vs 每个 GDR 源, 匈牙利匹配角距

镜像冻结口径 (standalone 重实现, 不 import/修改 src/fractureflow):
  spherical_kmeans / _sign_align / _silhouette   <- src/fractureflow/setlabel.py
  estimate_set_table (strict=True, 观测-only)    <- src/fractureflow/set_table_eval.py
  set_table_score (匈牙利 + modal_err + cov@30)  <- src/fractureflow/set_table_eval.py
  make_blind 掩码约定 (rng(seed).random(L)<0.40) <- src/fractureflow/honest_eval.py
  10 掩码种子 999+1000s, kmeans seed=42          <- scripts/t54_*, scripts/b2b3_*

红线: 零训练/零新依赖 (numpy+scipy; torch 仅用于只读 FORGE 桥源, 缺失即 skip);
输出 JSON allow_nan=False; 只写 results/r161_gdr_pipeline/ 下新文件;
确定性: 输出内无墙钟时间戳, 双跑逐位一致 (determinism_report.json)。
"""
import csv
import glob
import hashlib
import json
import math
import os
import sys
import zipfile

# ---------------------------------------------------------------------------
# 确定性硬前提: BLAS 多线程归约序在不同进程间可能漂移 (同一 matmul 结果末位不同),
# 经 k-means 的 argmax 近并列点放大后会导致跨进程结果不一致 (实测本单即命中:
# 双跑 arms_decovalex_4fracplus.json sha256 不符)。故在 import numpy 之前锁定单线程。
# 代价: 大矩阵乘法变慢 (本单最大 146k x 4 / 19.5k x 12, 可接受)。
# ---------------------------------------------------------------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402  (必须在线程数锁定之后导入)
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT_DIR = "results/r161_gdr_pipeline"
PREREG_PATH = os.path.join(OUT_DIR, "prereg.json")

KMEANS_SEED = 42
ITERS = 80
K_GRID = [2, 3, 4, 5, 6, 7, 8]
ANCHOR_K = 12
MASK_SEEDS = [999 + 1000 * s for s in range(10)]
OBS_FRAC = 0.40
# K 参与资格: 每组至少 3 个观测点 (prereg amendment #1; 对齐
# set_table_eval._group_dispersion 的 statistical 底线 n>=3)
MIN_PTS_PER_K = 3
CROSSCHECK_K = 4

BEACHHEAD = "results/r149_data_beachhead"
GDR_DIR = os.path.join(BEACHHEAD, "extracted")

# 既有锚 (只引用, 零改写; 来源 results/set_table_eval.json::forge_2wells.K12
# 与 results/t58_b2b3_forge_rebuild.json::new_numbers.K12)
ANCHOR_FORGE_MODAL_MEAN = 11.0453
ANCHOR_FORGE_MODAL_STD = 2.14219
ANCHOR_GATE_TOL = 0.02


# ---------------------------------------------------------------------------
# 几何 (ENU, 方位角自北顺时针)
# ---------------------------------------------------------------------------
def normal_from_strike_dip(strike_deg, dip_deg):
    """(strike, dip) -> 单位法向; dip_dir = (strike + 90) mod 360 (右手法则)."""
    a = np.radians((np.asarray(strike_deg, dtype=np.float64) + 90.0) % 360.0)
    d = np.radians(np.asarray(dip_deg, dtype=np.float64))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def normal_to_orientation(n):
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    dip = np.degrees(np.arccos(np.clip(n[..., 2], -1.0, 1.0)))
    dip_dir = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360.0
    return dip_dir, dip


def axial_strike(dip_dir_deg):
    return np.asarray(dip_dir_deg, dtype=np.float64) % 180.0


def center_row(c, n_members):
    dd, dp = normal_to_orientation(c)
    return {"dip_deg": round(float(dp), 4), "dip_dir_deg": round(float(dd), 4),
            "axial_strike_deg": round(float(axial_strike(dd - 90.0)), 4),
            "n_members": int(n_members)}


# ---------------------------------------------------------------------------
# 冻结口径 port: 球面 k-means / 符号对齐 / 轮廓系数 (setlabel.py)
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _sign_align(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def spherical_kmeans(nrm, K, iters=ITERS, seed=KMEANS_SEED):
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


def estimate_set_table_obs(nrm, occ, K, seed=KMEANS_SEED):
    """port of set_table_eval.estimate_set_table(strict=True): 观测-only 组心."""
    nrm_use = nrm[occ]
    K_eff = min(K, len(nrm_use))
    if K_eff < 2:
        c = _unit(_sign_align(nrm_use).mean(0))[None]
        return c
    cents, assign_obs = spherical_kmeans(nrm_use, K_eff, seed=seed)
    modes = np.zeros((K_eff, 3))
    for k in range(K_eff):
        sel = assign_obs == k
        if sel.sum():
            pts_s = _sign_align(nrm_use[sel])
            modes[k] = pts_s.mean(0)
            modes[k] /= (np.linalg.norm(modes[k]) + 1e-12)
    return modes


# ---------------------------------------------------------------------------
# 组系表对比 (port of set_table_eval.set_table_score)
# ---------------------------------------------------------------------------
def set_table_score(pred_centers, truth_centers, match_threshold=30.0):
    pred = _unit(np.asarray(pred_centers, dtype=np.float64))
    truth = _unit(np.asarray(truth_centers, dtype=np.float64))
    Kp, Kt = pred.shape[0], truth.shape[0]
    if Kp == 0 or Kt == 0:
        return {"modal_err_deg": 90.0, "K_pred": Kp, "K_truth": Kt,
                "coverage": 0.0, "n_matched": 0, "matched_pairs": []}
    cost = np.zeros((Kp, Kt))
    for i in range(Kp):
        for j in range(Kt):
            cos = float(np.clip(np.abs(pred[i] @ truth[j]), 0.0, 1.0))
            cost[i, j] = np.degrees(np.arccos(cos))
    ri, ci = linear_sum_assignment(cost)
    errs, pairs = [], []
    for i, j in zip(ri.tolist(), ci.tolist()):
        e = float(cost[i, j])
        errs.append(e)
        pairs.append({"pred_i": i, "truth_j": j, "err_deg": round(e, 3)})
    modal = float(np.mean(errs)) if errs else 90.0
    n_ok = sum(1 for e in errs if e < match_threshold)
    return {"modal_err_deg": round(modal, 3),
            "K_pred": Kp, "K_truth": Kt,
            "coverage": round(n_ok / max(Kt, 1), 3),
            "n_matched": len(pairs),
            "matched_pairs": pairs}


def label_truth_centers(nrm, labels):
    """标签真值组心: 全量法向按标签分组, 符号对齐均值 (b2b3_forge.py 同构)."""
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    centers = np.zeros((len(uniq), 3))
    counts = []
    for k, u in enumerate(uniq):
        sel = labels == u
        counts.append(int(sel.sum()))
        pts_s = _sign_align(nrm[sel])
        centers[k] = pts_s.mean(0)
        centers[k] /= (np.linalg.norm(centers[k]) + 1e-12)
    return centers, uniq, counts


def truth_label_table(nrm, labels):
    """真值组表 (纯加法, 供 verdict 直读): 每组 组心产状 + 点数 + 组内离散。

    组内离散口径沿用 set_table_eval._group_dispersion: arccos(mean|cos|) (度)。
    """
    centers, uniq, counts = label_truth_centers(nrm, np.array(labels))
    lab = np.asarray(labels)
    rows = []
    for k, u in enumerate(uniq):
        sel = lab == u
        cos = np.clip(np.abs(_unit(nrm[sel]) @ centers[k]), 0.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        row = center_row(centers[k], counts[k])
        row["label"] = str(u)
        row["dispersion_deg"] = round(float(ang.mean()), 4)
        row["max_dev_deg"] = round(float(ang.max()), 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# T1 适配: 解析 + 逐行校验
# ---------------------------------------------------------------------------
def _new_counters():
    return {"rows": 0, "kept": 0, "parse_error": 0,
            "degenerate_dip": 0, "trend_plunge_inconsistent": 0,
            "label_missing": 0}


def parse_gdr_csv(path, strike_col, dip_col, label_rule):
    """GDR 产状表 -> 法向; 逐行校验 (V1 plunge / V2 trend / V3 dip / V4 parse)."""
    cnt = _new_counters()
    rec = {"strike": [], "dip": [], "trend": [], "plunge": [], "hemi": [], "label": []}
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            cnt["rows"] += 1
            try:
                st = float(r[strike_col])
                dp = float(r[dip_col])
                tr = float(r["Trend[deg]"])
                pl = float(r["Plunge[deg]"])
            except (TypeError, ValueError, KeyError):
                cnt["parse_error"] += 1
                continue
            if not all(math.isfinite(x) for x in (st, dp, tr, pl)):
                cnt["parse_error"] += 1
                continue
            if not (0.0 < dp <= 90.0):
                cnt["degenerate_dip"] += 1
                continue
            alpha = (st + 90.0) % 360.0
            delta = (tr - alpha) % 360.0
            dev0 = min(delta, 360.0 - delta)
            dev180 = abs(delta - 180.0)
            if abs(pl - (90.0 - dp)) > 0.01 or min(dev0, dev180) > 0.01:
                cnt["trend_plunge_inconsistent"] += 1
                continue
            lab = label_rule(r)
            if lab is None:
                cnt["label_missing"] += 1
            rec["strike"].append(st)
            rec["dip"].append(dp)
            rec["trend"].append(tr)
            rec["plunge"].append(pl)
            rec["hemi"].append("up" if dev0 <= dev180 else "down")
            rec["label"].append(lab)
            cnt["kept"] += 1
    nrm = normal_from_strike_dip(np.array(rec["strike"]), np.array(rec["dip"]))
    return {"nrm": nrm, "rec": rec, "counters": cnt}


def parse_decovalex(zip_path, member_suffix):
    """DECOVALEX normal_vectors.dat (每裂缝单条三分量法向) -> 法向表 (直读, 不往返)."""
    cnt = _new_counters()
    rec = {"strike": [], "dip": [], "trend": [], "plunge": [], "hemi": [], "label": []}
    vecs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist()
                   if n.endswith(member_suffix) and "__MACOSX" not in n]
        if not members:
            raise FileNotFoundError(f"zip 内未找到 {member_suffix}")
        payload = zf.read(sorted(members)[0]).decode("ascii", errors="replace")
    for line in payload.strip().split("\n"):
        if not line.strip():
            continue
        cnt["rows"] += 1
        parts = line.strip().split()
        if len(parts) < 3:
            cnt["parse_error"] += 1
            continue
        try:
            n = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            cnt["parse_error"] += 1
            continue
        if not np.all(np.isfinite(n)) or np.linalg.norm(n) <= 1e-12:
            cnt["parse_error"] += 1
            continue
        n = n / np.linalg.norm(n)
        if n[2] < 0.0:                      # 规范到上半球 (轴向等价, 仅报表可读)
            n = -n
        vecs.append(n)
        dd, dp = normal_to_orientation(n)
        rec["strike"].append(float(axial_strike(dd - 90.0)))
        rec["dip"].append(float(dp))
        rec["trend"].append(float("nan"))
        rec["plunge"].append(float("nan"))
        rec["hemi"].append("n/a")
        rec["label"].append(None)
        cnt["kept"] += 1
    return {"nrm": _unit(np.array(vecs)), "rec": rec, "counters": cnt}


def parse_forge_pt(path):
    """FORGE FMI 双井 (只读) -> 逐井法向表。缺 torch/缺文件 -> 返回 (None, reason)."""
    if not os.path.exists(path):
        return None, "MISSING_FILE"
    try:
        import torch
    except ImportError:
        return None, "NO_TORCH"
    data = torch.load(path, weights_only=False)
    nets = data["nets"] if isinstance(data, dict) and "nets" in data else data
    units = []
    for net in nets:
        cnt = _new_counters()
        nrm = np.asarray(net.get("nrm_full", net["nrm"]), dtype=np.float64)
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        ft = np.asarray(net.get("ftype")).astype(str) if "ftype" in net else None
        cnt["rows"] = int(nrm.shape[0])
        rec = {"strike": [], "dip": [], "trend": [], "plunge": [], "hemi": [],
               "label": [str(x) for x in ft] if ft is not None else []}
        dd, dp = normal_to_orientation(nrm)
        rec["dip"] = dp.tolist()
        rec["strike"] = axial_strike(dd - 90.0).tolist()
        rec["trend"] = [float("nan")] * len(dp)
        rec["plunge"] = [float("nan")] * len(dp)
        rec["hemi"] = ["n/a"] * len(dp)
        cnt["kept"] = int(nrm.shape[0])
        units.append({"unit_id": str(net.get("wid", "?")), "nrm": nrm,
                      "rec": rec, "counters": cnt})
    return units, None


# ---------------------------------------------------------------------------
# 数据源装配
# ---------------------------------------------------------------------------
def _label_none(r):
    return None


def _label_1750_stochastic(r):
    obj = (r.get("object") or "").strip()
    if not obj:
        return None
    return obj.rsplit("_", 1)[0]           # 'Set 1 Dipping South Min 20m Radius_1' -> 组名


def _label_1750_discrete(r):
    toks = (r.get("object") or "").split("_")
    for t in toks:
        if t in ("NF", "HF"):
            return t
    return None


def _label_1646_filename(r):
    fn = (r.get("Filename") or "").strip()
    return fn[:-4] if fn.lower().endswith(".csv") else (fn or None)


def load_sources():
    """返回 [(source_id, meta, units)]; units = [(unit_id, nrm, rec, counters)]."""
    specs = [
        ("gdr1222", {
            "kind": "gdr_glob",
            "glob": os.path.join(GDR_DIR, "gdr1222", "Filtered_800m_*.csv"),
            "strike_col": "Strike[deg]", "dip_col": "DipAngle[deg]",
            "label_rule": _label_none,
            "label_kind": None}),
        ("gdr1646", {
            "kind": "gdr_glob",
            "glob": os.path.join(GDR_DIR, "gdr1646", "2024 July Updated MEQ DFN",
                                 "Local Coordinates", "New MEQ DFN v2 July 2024.csv"),
            "strike_col": "Strike[deg]", "dip_col": "Dip_Angle[deg]",
            "label_rule": _label_1646_filename,
            "label_kind": "weak_provenance_cluster"}),
        ("gdr1750_stochastic", {
            "kind": "gdr_glob",
            "glob": os.path.join(GDR_DIR, "gdr1750", "Global Coordinates Feet",
                                 "FORGE 2025 v1 DFN Fracture Files",
                                 "DFN 2025 v1 Stochastic Fractures global ft.csv"),
            "strike_col": "Strike[deg]", "dip_col": "Dip_Angle[deg]",
            "label_rule": _label_1750_stochastic,
            "label_kind": "true_set"}),
        ("gdr1750_discrete", {
            "kind": "gdr_glob",
            "glob": os.path.join(GDR_DIR, "gdr1750", "Global Coordinates Feet",
                                 "FORGE 2025 v1 DFN Fracture Files",
                                 "DFN 2025 v1 Discrete Fractures global ft.csv"),
            "strike_col": "Strike[deg]", "dip_col": "Dip_Angle[deg]",
            "label_rule": _label_1750_discrete,
            "label_kind": "semantic_NF_HF"}),
    ]
    out = []
    for sid, meta in specs:
        files = sorted(glob.glob(meta["glob"]),
                       key=lambda p: (int(p.split("_")[-1].split(".")[0])
                                      if p.split("_")[-1].split(".")[0].isdigit() else 0, p))
        if not files:
            out.append((sid, dict(meta, status="MISSING"), []))
            continue
        units = []
        for i, fp in enumerate(files):
            parsed = parse_gdr_csv(fp, meta["strike_col"], meta["dip_col"],
                                   meta["label_rule"])
            units.append({"unit_id": os.path.basename(fp), "nrm": parsed["nrm"],
                          "rec": parsed["rec"], "counters": parsed["counters"]})
        out.append((sid, dict(meta, status="OK", files=[os.path.basename(f) for f in files]),
                    units))

    # DECOVALEX
    zpath = os.path.join(BEACHHEAD, "raw", "zen14873207_decovalex_d23_4frac_dfn.zip")
    if os.path.exists(zpath):
        p = parse_decovalex(zpath, "4fracplus_example/normal_vectors.dat")
        out.append(("decovalex_4fracplus",
                    {"kind": "decovalex_zip", "status": "OK",
                     "label_rule": _label_none, "label_kind": None},
                    [{"unit_id": "4fracplus_example", "nrm": p["nrm"],
                      "rec": p["rec"], "counters": p["counters"]}]))
    else:
        out.append(("decovalex_4fracplus", {"kind": "decovalex_zip", "status": "MISSING",
                                            "label_kind": None}, []))

    # FORGE FMI 桥源 (只读)
    units, reason = parse_forge_pt("data/external/utah_forge_fmi/forge_fmi_2wells.pt")
    if units is None:
        out.append(("forge_fmi_bridge",
                    {"kind": "forge_pt", "status": f"SKIP:{reason}", "label_kind": "ftype"}, []))
    else:
        out.append(("forge_fmi_bridge",
                    {"kind": "forge_pt", "status": "OK", "label_kind": "ftype"}, units))
    return out


# ---------------------------------------------------------------------------
# 两臂管线
# ---------------------------------------------------------------------------
def kscan(nrm, k_grid):
    recs, best_k, best_s = [], None, -np.inf
    for K in k_grid:
        if len(nrm) < MIN_PTS_PER_K * K:
            recs.append({"K": K, "silhouette": None, "eligible": False})
            continue
        centers, assign = spherical_kmeans(nrm, K, seed=KMEANS_SEED)
        s = silhouette(nrm, centers, assign)
        recs.append({"K": K, "silhouette": round(s, 6), "eligible": True})
        if s > best_s:
            best_s, best_k = s, K
    return recs, best_k, best_s


def per_point_stats(nrm, centers, assign):
    abscos = np.abs(_unit(nrm) @ _unit(centers).T)
    err = np.degrees(np.arccos(np.clip(abscos[np.arange(len(nrm)), assign], 0.0, 1.0)))
    return {"mean": round(float(err.mean()), 4),
            "p50": round(float(np.percentile(err, 50)), 4),
            "p90": round(float(np.percentile(err, 90)), 4),
            "n": int(err.size)}


def run_unit_arms(nrm, rec, ks_to_run):
    """单 unit: oracle 臂 + honest 臂 (每个 K in ks_to_run)。返回 (per-K 记录, 真值组心)。"""
    out = {}
    centers_o_full = {}
    stats_o_full = {}
    for K in ks_to_run:
        centers_o, assign_o = spherical_kmeans(nrm, K, seed=KMEANS_SEED)
        centers_o_full[K] = centers_o
        stats_o_full[K] = per_point_stats(nrm, centers_o, assign_o)
    L = len(nrm)
    nrm_u = _unit(nrm)
    labels = rec["label"]
    has_label = bool(labels) and len(labels) == L and all(x is not None for x in labels)
    truth_label = None
    if has_label:
        truth_label, _, _ = label_truth_centers(nrm_u, np.array(labels))
    for K in ks_to_run:
        centers_o = centers_o_full[K]
        assign_o = np.abs(nrm_u @ _unit(centers_o).T).argmax(1)
        seeds = []
        for ms in MASK_SEEDS:
            rng = np.random.default_rng(ms)
            occ = rng.random(L) < OBS_FRAC
            n_obs = int(occ.sum())
            if n_obs < MIN_PTS_PER_K * K or n_obs < 2 or (L - n_obs) == 0:
                seeds.append({"mask_seed": ms, "skipped": True,
                              "reason": "n_obs < %d*K 或退化" % MIN_PTS_PER_K,
                              "n_obs": n_obs})
                continue
            centers_h = estimate_set_table_obs(nrm_u, occ, K, seed=KMEANS_SEED)
            hid = nrm_u[~occ]
            cos_h = np.abs(hid @ _unit(centers_h).T)
            pred = cos_h.argmax(1)
            err_h = np.degrees(np.arccos(np.clip(
                cos_h[np.arange(len(hid)), pred], 0.0, 1.0)))
            s_self = set_table_score(centers_h, centers_o)
            row = {"mask_seed": ms, "n_obs": n_obs, "n_hidden": int(L - n_obs),
                   "T_self": {"modal_err_deg": s_self["modal_err_deg"],
                              "coverage_at_30deg": s_self["coverage"]},
                   "hidden_acos_err": {"mean": round(float(err_h.mean()), 4),
                                       "p50": round(float(np.percentile(err_h, 50)), 4),
                                       "p90": round(float(np.percentile(err_h, 90)), 4)}}
            if has_label:
                s_lab = set_table_score(centers_h, truth_label)
                row["T_label"] = {"modal_err_deg": s_lab["modal_err_deg"],
                                  "coverage_at_30deg": s_lab["coverage"]}
            seeds.append(row)
        out[K] = {"oracle": {"per_point_acos": stats_o_full[K],
                             "group_table": [center_row(centers_o[k],
                                                        int((assign_o == k).sum()))
                                             for k in range(len(centers_o))]},
                  "honest": seeds}
    return out, truth_label


def aggregate_arms(per_unit):
    """per_unit: {unit_id: {"oracle":..., "honest":[...]}} -> 先跨 unit 均值, 再跨 seed mean/std."""
    rows = list(per_unit.values())
    if not rows:
        return None
    n_units = len(rows)
    seed_keys = [r["mask_seed"] for r in rows[0]["honest"]]
    per_seed = {}
    for ms in seed_keys:
        vals_self, vals_lab, cov_self, cov_lab, hid_means, skipped = [], [], [], [], [], 0
        for r in rows:
            srow = next(x for x in r["honest"] if x["mask_seed"] == ms)
            if srow.get("skipped"):
                skipped += 1
                continue
            vals_self.append(srow["T_self"]["modal_err_deg"])
            cov_self.append(srow["T_self"]["coverage_at_30deg"])
            if "T_label" in srow:
                vals_lab.append(srow["T_label"]["modal_err_deg"])
                cov_lab.append(srow["T_label"]["coverage_at_30deg"])
            hid_means.append(srow["hidden_acos_err"]["mean"])
        if skipped == len(rows):
            continue
        per_seed[ms] = {
            "n_units_used": len(rows) - skipped,
            "n_units_skipped": skipped,
            "T_self_modal_err": round(float(np.mean(vals_self)), 4),
            "T_self_coverage": round(float(np.mean(cov_self)), 4),
            "T_label_modal_err": (round(float(np.mean(vals_lab)), 4) if vals_lab else None),
            "T_label_coverage": (round(float(np.mean(cov_lab)), 4) if cov_lab else None),
            "hidden_acos_mean_of_unit_means": round(float(np.mean(hid_means)), 4)}
    if not per_seed:
        return None
    v_self = [p["T_self_modal_err"] for p in per_seed.values()]
    v_lab = [p["T_label_modal_err"] for p in per_seed.values()
             if p["T_label_modal_err"] is not None]
    v_hid = [p["hidden_acos_mean_of_unit_means"] for p in per_seed.values()]
    v_ora = [r["oracle"]["per_point_acos"]["mean"] for r in rows]
    agg = {
        "n_units": n_units,
        "n_seeds": len(per_seed),
        "T_self_modal_err_mean": round(float(np.mean(v_self)), 4),
        "T_self_modal_err_std": (round(float(np.std(v_self, ddof=1)), 4)
                                 if len(v_self) > 1 else 0.0),
        "T_self_coverage_mean": round(float(np.mean(
            [p["T_self_coverage"] for p in per_seed.values()])), 4),
        "hidden_acos_mean_of_seed_means": round(float(np.mean(v_hid)), 4),
        "hidden_acos_std_of_seed_means": (round(float(np.std(v_hid, ddof=1)), 4)
                                          if len(v_hid) > 1 else 0.0),
        "oracle_per_point_acos_mean": round(float(np.mean(v_ora)), 4),
        "honest_minus_oracle_acos": round(float(np.mean(v_hid)) - float(np.mean(v_ora)), 4),
        "per_seed": per_seed,
    }
    if v_lab and len(v_lab) == len(v_self):
        agg["T_label_modal_err_mean"] = round(float(np.mean(v_lab)), 4)
        agg["T_label_modal_err_std"] = (round(float(np.std(v_lab, ddof=1)), 4)
                                        if len(v_lab) > 1 else 0.0)
        agg["T_label_coverage_mean"] = round(float(np.mean(
            [p["T_label_coverage"] for p in per_seed.values()
             if p["T_label_coverage"] is not None])), 4)
    return agg


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def write_parsed_csv(sid, units):
    path = os.path.join(OUT_DIR, f"parsed_{sid}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "unit", "row", "strike_deg", "dip_deg",
                    "dip_dir_deg", "trend_deg", "plunge_deg", "pole_hemisphere", "label"])
        for u in units:
            rec = u["rec"]
            dip = rec["dip"]
            strike = rec["strike"]
            for i in range(len(dip)):
                dd = (float(strike[i]) + 90.0) % 360.0 if np.isfinite(strike[i]) else float("nan")
                w.writerow([sid, u["unit_id"], i,
                            round(float(strike[i]), 4) if np.isfinite(strike[i]) else "",
                            round(float(dip[i]), 4),
                            round(dd, 4) if np.isfinite(dd) else "",
                            "" if not np.isfinite(rec["trend"][i]) else round(rec["trend"][i], 4),
                            "" if not np.isfinite(rec["plunge"][i]) else round(rec["plunge"][i], 4),
                            rec["hemi"][i], rec["label"][i] if i < len(rec["label"]) else ""])
    return path


def run_source(sid, meta, units, prereg_sha):
    print(f"\n[source] {sid}: status={meta.get('status')} units={len(units)}")
    if not units:
        return {"source": sid, "status": meta.get("status", "MISSING"),
                "prereg_sha256": prereg_sha, "units": [], "arms": None}
    write_parsed_csv(sid, units)
    counters = {}
    unit_reports = []
    per_unit_arms = {}
    n_units_seen = 0
    res_truth_table = None
    for u in units:
        nrm = _unit(u["nrm"])
        if len(nrm) < 10:
            unit_reports.append({"unit_id": u["unit_id"], "n_valid": len(nrm),
                                 "status": "SKIP_TOO_FEW"})
            continue
        recs_k, k_star, sil = kscan(nrm, K_GRID)
        ks_to_run = sorted(set([k_star] + ([ANCHOR_K] if len(nrm) >= MIN_PTS_PER_K * ANCHOR_K
                                           and len(nrm) * OBS_FRAC >= MIN_PTS_PER_K * ANCHOR_K
                                           else [])))
        arms, truth_label = run_unit_arms(nrm, u["rec"], ks_to_run)
        per_unit_arms[u["unit_id"]] = {}
        for K in ks_to_run:
            tag = "K12(anchor)" if K == ANCHOR_K else "K_star"
            per_unit_arms[u["unit_id"]][tag] = {"oracle": arms[K]["oracle"],
                                                "honest": arms[K]["honest"]}
        unit_reports.append({
            "unit_id": u["unit_id"], "n_valid": int(len(nrm)),
            "counters": u["counters"], "k_scan": recs_k,
            "K_star": int(k_star) if k_star else None,
            "silhouette_at_K_star": round(sil, 6) if k_star else None,
            "K_run": [int(x) for x in ks_to_run],
            "oracle_group_table_K_star": (arms[k_star]["oracle"]["group_table"]
                                          if k_star else None),
            "truth_label_groups": (None if truth_label is None else
                                   len(truth_label))})
        counters[u["unit_id"]] = u["counters"]
        has_lab = bool(u["rec"]["label"]) and all(x is not None for x in u["rec"]["label"])
        if has_lab and n_units_seen == 0:
            res_truth_table = truth_label_table(nrm, u["rec"]["label"])
        print(f"  unit={u['unit_id']}: n={len(nrm)} K*={k_star} sil={sil:.4f} "
              f"K_run={ks_to_run}")
        n_units_seen += 1
    arms_agg = {}
    for tag in ("K_star", "K12(anchor)"):
        d = {uid: v[tag] for uid, v in per_unit_arms.items() if tag in v}
        a = aggregate_arms(d)
        if a is not None:
            arms_agg[tag] = a
    res = {"source": sid, "status": "OK", "prereg_sha256": prereg_sha,
           "n_units": len(units),
           "counters_per_unit": counters,
           "counters_total": {k: int(sum(c[k] for c in counters.values()))
                              for k in _new_counters()},
           "units": unit_reports,
           "arms": arms_agg,
           "truth_label_table": res_truth_table,
           "label_kind": meta.get("label_kind"),
           "files": meta.get("files")}
    if sid == "decovalex_4fracplus":
        u0 = units[0]
        nrm0 = _unit(u0["nrm"])
        recs_k, k_star, _ = kscan(nrm0, K_GRID)
        ks = sorted(set([k_star, CROSSCHECK_K]))
        s_res = {int(k): run_s_arm(nrm0, [k])[k] for k in ks}
        all_zero = all(abs(v["modal_err_mean"]) < 1e-9 for v in s_res.values())
        res["S_arm"] = {
            "definition": "既有 0.047 deg 锚口径复现尝试: 真值组心 = 观测裂缝法向 k-means "
                          "归并 K 组 (b2b3_decovalex.py 配方); 预测组心 = 观测-only "
                          "estimate_set_table。本载体每裂缝单法向 -> 两侧同源恒等",
            "anchor_reference_readonly": {"modal_err": 0.047, "K": 4,
                                          "src": "results/set_table_decovalex.json "
                                                 "(只引用, 零改写)"},
            "K_run": [int(k) for k in ks],
            "result": {str(k): v for k, v in s_res.items()},
            "verdict": ("不可复现: 本载体下该口径退化为恒等式 (两侧同一批观测向量/同一 "
                        "seed/同一 K), 实测 modal_err = 0; 既有 0.047 的差异来自其每裂缝"
                        "多点调查点载体 (含噪, 真值侧取裂缝级均值), 本载体无该噪声源")
                       if all_zero else
                       "非恒等: 配方 port 与既有实现有差异, 需停查"}
    write_json(f"arms_{sid}.json", res)
    print(f"  [save] arms_{sid}.json")
    return res


def run_anchor_gate(units, prereg_sha):
    """G0 锚复现闸: FORGE FMI (ftype 真值, K=12) 复现 11.0453 ± 2.1422."""
    if not units:
        return {"gate": "GATE_SKIPPED", "reason": "forge_fmi_bridge 源不可用",
                "prereg_sha256": prereg_sha}
    seed_means = []
    per_seed = []
    for ms in MASK_SEEDS:
        well_vals = []
        for u in units:
            nrm = _unit(u["nrm"])
            L = len(nrm)
            rng = np.random.default_rng(ms)
            occ = rng.random(L) < OBS_FRAC
            if occ.sum() < 2 or (~occ).sum() == 0 or L < 5:
                continue
            pred = estimate_set_table_obs(nrm, occ, ANCHOR_K, seed=KMEANS_SEED)
            truth, uniq, counts = label_truth_centers(nrm, np.array(u["rec"]["label"]))
            sc = set_table_score(pred, truth)
            well_vals.append(sc["modal_err_deg"])
        if well_vals:
            seed_means.append(float(np.mean(well_vals)))
            per_seed.append({"mask_seed": ms, "per_well": well_vals,
                             "mean": round(float(np.mean(well_vals)), 4)})
    mean = float(np.mean(seed_means))
    std = float(np.std(seed_means, ddof=1))
    dev_mean = abs(mean - ANCHOR_FORGE_MODAL_MEAN)
    dev_std = abs(std - ANCHOR_FORGE_MODAL_STD)
    passed = bool(dev_mean <= ANCHOR_GATE_TOL and dev_std <= ANCHOR_GATE_TOL)
    out = {"gate": "PASS" if passed else "FAIL",
           "prereg_sha256": prereg_sha,
           "recipe": "b2b3_forge.py 同构: 逐井 estimate_set_table(strict=True,K=12,seed=42) "
                     "vs ftype 真值组心, 匈牙利 modal_err, 先井均后 seed 均, 10 seed",
           "anchor_reference_readonly": {"modal_err_mean": ANCHOR_FORGE_MODAL_MEAN,
                                         "modal_err_std": ANCHOR_FORGE_MODAL_STD,
                                         "src": "results/set_table_eval.json::forge_2wells.K12 "
                                                "[NNS-019] (只引用, 零改写)"},
           "replicated": {"modal_err_mean": round(mean, 4), "modal_err_std": round(std, 4),
                          "n_seeds": len(seed_means)},
           "deviation": {"mean_abs": round(dev_mean, 6), "std_abs": round(dev_std, 6),
                         "tol": ANCHOR_GATE_TOL},
           "per_seed": per_seed}
    return out


def run_s_arm(nrm, ks_to_run):
    """S 臂: 既有 DECOVALEX 0.047 deg 锚的口径复现尝试 (prereg truth_modes.S_arm).

    配方 (b2b3_decovalex.py): 真值组心 = 把**观测**裂缝法向用 k-means 归并成 K 组后取
    组内符号对齐均值; 预测组心 = 观测-only estimate_set_table。本单载体为每裂缝单法向,
    两条路径作用于同一批观测向量、同一 seed、同一 K, 故理论上恒等 -> modal_err = 0。
    实测到 0 即证实该锚口径在本载体上退化为恒等式 (不可复现); 实测非 0 则说明配方
    port 有误, 需停查。
    """
    L = len(nrm)
    nrm_u = _unit(nrm)
    out = {}
    for K in ks_to_run:
        seeds = []
        for ms in MASK_SEEDS:
            occ = np.random.default_rng(ms).random(L) < OBS_FRAC
            obs = nrm_u[occ]
            if len(obs) < MIN_PTS_PER_K * K or len(obs) < 2:
                seeds.append({"mask_seed": ms, "skipped": True, "n_obs": int(occ.sum())})
                continue
            pred = estimate_set_table_obs(nrm_u, occ, K, seed=KMEANS_SEED)
            _, assign = spherical_kmeans(obs, K, seed=KMEANS_SEED)
            truth = np.zeros((K, 3))
            for k in range(K):
                sel = assign == k
                if sel.sum():
                    pts_s = _sign_align(obs[sel])
                    truth[k] = pts_s.mean(0)
                    truth[k] /= (np.linalg.norm(truth[k]) + 1e-12)
            sc = set_table_score(pred, truth)
            seeds.append({"mask_seed": ms, "n_obs": int(occ.sum()),
                          "modal_err_deg": sc["modal_err_deg"],
                          "coverage_at_30deg": sc["coverage"]})
        vals = [s["modal_err_deg"] for s in seeds if not s.get("skipped")]
        out[K] = {"per_seed": seeds,
                  "modal_err_mean": round(float(np.mean(vals)), 4) if vals else None,
                  "modal_err_std": (round(float(np.std(vals, ddof=1)), 4)
                                    if len(vals) > 1 else 0.0),
                  "n_seeds": len(vals)}
    return out


def run_crosscheck(all_units_by_source):
    """同场地 K=4 全量组心对照 (FORGE FMI 每井 vs 每个 GDR 源)."""
    pools = {}
    for sid, units in all_units_by_source.items():
        if not units:
            continue
        if sid == "forge_fmi_bridge":
            for u in units:
                if len(u["nrm"]) >= MIN_PTS_PER_K * CROSSCHECK_K:
                    c, a = spherical_kmeans(_unit(u["nrm"]), CROSSCHECK_K, seed=KMEANS_SEED)
                    pools[f"forge_fmi:{u['unit_id']}"] = (c, a)
        else:
            nrm = np.concatenate([_unit(u["nrm"]) for u in units])
            if len(nrm) >= MIN_PTS_PER_K * CROSSCHECK_K:
                c, a = spherical_kmeans(nrm, CROSSCHECK_K, seed=KMEANS_SEED)
                pools[sid] = (c, a)
    keys = sorted(pools)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if not (keys[i].startswith("forge_fmi:") or keys[j].startswith("forge_fmi:")):
                continue
            ca, cb = pools[keys[i]][0], pools[keys[j]][0]
            cost = np.degrees(np.arccos(np.clip(
                np.abs(_unit(ca) @ _unit(cb).T), 0.0, 1.0)))
            ri, ci = linear_sum_assignment(cost)
            errs = [float(cost[a, b]) for a, b in zip(ri.tolist(), ci.tolist())]
            pairs.append({"a": keys[i], "b": keys[j],
                          "mean_deg": round(float(np.mean(errs)), 4),
                          "min_deg": round(float(np.min(errs)), 4),
                          "max_deg": round(float(np.max(errs)), 4),
                          "matched_pairs": [round(e, 3) for e in errs]})
    tables = {k: [center_row(c, int((pools[k][1] == i).sum()))
                  for i, c in enumerate(pools[k][0])] for k in keys}
    return {"K": CROSSCHECK_K, "kmeans_seed": KMEANS_SEED,
            "note": "全量 (无掩码) 组心; GDR 侧为该源全量池化, FORGE 侧为逐井; "
                    "轴向 |cos| 角距, 匈牙利匹配; 纯几何对照, 非精度指标",
            "group_tables": tables, "pairs": pairs}


def output_manifest():
    """results/r161_gdr_pipeline/ 下除 prereg.json 外所有文件的 sha256 (相对路径为键)。"""
    man = {}
    for name in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, name)
        if not os.path.isfile(p) or name == "prereg.json":
            continue
        man[name] = sha256_file(p)
    return man


def write_determinism_report(manifest_paths):
    """N 次独立运行互校: 只比较在所有清单中都出现且非本报告的文件的 sha256。"""
    ms = []
    for p in manifest_paths:
        with open(p, "r", encoding="utf-8") as f:
            ms.append(json.load(f))
    keys = sorted(set.intersection(*[set(m) for m in ms]) - {"determinism_report.json"})
    rows = []
    for k in keys:
        hs = [m.get(k) for m in ms]
        rows.append({"file": k, "hashes": hs, "match": len(set(hs)) == 1})
    rep = {"rule": "双跑(本例为 %d 次独立运行互校)逐位一致 (R152 §4 / R161 验收判据 ②); "
                   "跨进程独立运行, 比对除 prereg.json 与本报告外全部输出文件 sha256 "
                   "(本报告自身不参与比较)" % len(ms),
           "n_runs": len(ms), "n_files": len(rows),
           "all_match": bool(rows) and all(r["match"] for r in rows),
           "files": rows}
    write_json("determinism_report.json", rep)
    print(f"[determinism] n_runs={rep['n_runs']} all_match={rep['all_match']} "
          f"n_files={rep['n_files']}")
    return rep


def main(argv):
    if len(argv) >= 4 and argv[1] == "--determinism":
        os.makedirs(OUT_DIR, exist_ok=True)
        write_determinism_report(argv[2:])
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        json.load(f)
    prereg_sha = sha256_file(PREREG_PATH)
    print(f"[prereg] sha256={prereg_sha}")

    sources = load_sources()
    by_id = {sid: units for sid, _, units in sources}
    metas = {sid: meta for sid, meta, _ in sources}

    # ---- G0 锚复现闸 (先于任何 GDR 数字) ----
    gate = run_anchor_gate(by_id.get("forge_fmi_bridge", []), prereg_sha)
    write_json("anchor_replication.json", gate)
    print(f"[G0 锚复现闸] {gate['gate']} "
          f"replicated={gate.get('replicated')} dev={gate.get('deviation')}")
    if gate["gate"] == "FAIL":
        print("[FATAL] 锚复现闸未过: 本单 standalone port 与既有锚口径不同源, "
              "GDR 数字全部作废 (退出码 3)")
        return 3
    if gate["gate"] == "GATE_SKIPPED":
        print("[WARN] 锚复现闸跳过 (桥源不可用): 继续但 GDR 数字无口径自证")

    # ---- T3 两臂 ----
    summary_rows = []
    for sid, meta, units in sources:
        res = run_source(sid, meta, units, prereg_sha)
        if res.get("arms"):
            for tag, a in res["arms"].items():
                summary_rows.append({
                    "source": sid, "K_tag": tag,
                    "T_self_modal_err_mean": a.get("T_self_modal_err_mean"),
                    "T_self_modal_err_std": a.get("T_self_modal_err_std"),
                    "T_label_modal_err_mean": a.get("T_label_modal_err_mean"),
                    "T_label_modal_err_std": a.get("T_label_modal_err_std"),
                    "hidden_acos_mean": a.get("hidden_acos_mean_of_seed_means"),
                    "honest_minus_oracle_acos": a.get("honest_minus_oracle_acos"),
                    "n_units": a.get("n_units"), "n_seeds": a.get("n_seeds")})

    # ---- T4 同场地对照 ----
    cc = run_crosscheck(by_id)
    write_json("crosscheck_same_site.json", cc)
    print("[save] crosscheck_same_site.json")

    write_json("pipeline_summary.json", {
        "prereg_sha256": prereg_sha,
        "anchor_gate": gate["gate"],
        "crosscheck_pairs": cc["pairs"],
        "sources": {sid: {"status": metas[sid].get("status"),
                          "n_units": len(by_id.get(sid, [])),
                          "label_kind": metas[sid].get("label_kind")}
                    for sid in metas},
        "rows": summary_rows})
    print("[save] pipeline_summary.json")

    if len(argv) >= 3 and argv[1] == "--hash-manifest":
        man = output_manifest()
        with open(argv[2], "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"[manifest] -> {argv[2]} ({len(man)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
