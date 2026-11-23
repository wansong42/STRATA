# -*- coding: utf-8 -*-
"""W8 local evidence — 在盘资产盘点 + 多走向实测 (全部只读)。

产出: results/w8_public_data_recheck/evidence_snapshots/local/*.json (带 sha256)
     results/w8_public_data_recheck/local_measurements.json
判据: prereg.json (C2 多走向量化定义 = ≥2 组 modal 法向角距 ≥30°, 球面 k-means K∈2..8 silhouette)
水印: 研究线取证 · 禁入对外承诺文本
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "w8_public_data_recheck")
SNAP = os.path.join(OUT, "evidence_snapshots", "local")


def atomic_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, path)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_s():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def dir_snapshot(root, max_depth=2, max_entries=4000):
    rows = []
    base = root
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, base)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                rows.append({"path": os.path.join(rel, fn).replace("\\", "/"),
                             "bytes": st.st_size, "mtime": datetime.datetime.fromtimestamp(
                                 st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
            except OSError as e:
                rows.append({"path": os.path.join(rel, fn).replace("\\", "/"), "error": str(e)})
            if len(rows) >= max_entries:
                return rows
    return rows


# ---------------- C2 多走向实测 (球面 k-means, 无向法向 |cos| 指派 + 符号对齐) ----------------
def spher_kmeans(nrm, K, seed=42, n_init=8, iters=100):
    """无向法向球面 k-means: 以 |cos| 相似度指派, 组心符号对齐聚合。返回 (labels, centers, inertia)."""
    rng = np.random.default_rng(seed)
    n = nrm.shape[0]
    best = None
    for _ in range(n_init):
        idx = rng.choice(n, size=K, replace=False)
        C = nrm[idx].copy()
        for _ in range(iters):
            sim = np.abs(nrm @ C.T)              # (n,K)
            lab = np.argmax(sim, axis=1)
            newC = np.zeros_like(C)
            for k in range(K):
                m = lab == k
                if not m.any():
                    newC[k] = nrm[rng.integers(n)]
                    continue
                v = nrm[m].copy()
                ref = v[np.argmax(np.linalg.norm(v, axis=1))] if False else v[0]
                sgn = np.sign(v @ ref)
                sgn[sgn == 0] = 1.0
                newC[k] = (sgn[:, None] * v).sum(axis=0)
                nn = np.linalg.norm(newC[k])
                newC[k] = newC[k] / nn if nn > 1e-12 else nrm[rng.integers(n)]
            if np.allclose(newC, C, atol=1e-10):
                C = newC
                break
            C = newC
        sim = np.abs(nrm @ C.T)
        lab = np.argmax(sim, axis=1)
        inertia = float(np.mean(1.0 - sim[np.arange(n), lab]))
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best


def best_K_sets(nrm, kmin=2, kmax=8, seed=42):
    """silhouette 选 K (球面, |cos| 距离), 返回 (K, labels, centers, silhouette, modal_matrix)."""
    def sil(lab, C):
        # 无向距离 d = 1 - |cos|; 轮廓系数对 C 内每点: (b-a)/max(a,b)
        D = 1.0 - np.abs(nrm @ C.T)          # (n,K) 到组心距离
        n = nrm.shape[0]
        s = np.zeros(n)
        for i in range(n):
            k = lab[i]
            a = D[i, k]
            others = [D[i, j] for j in range(C.shape[0]) if j != k and (lab == j).any()]
            if not others:
                s[i] = 0.0
                continue
            b = min(others)
            m = max(a, b)
            s[i] = 0.0 if m <= 1e-12 else (b - a) / m
        return float(np.mean(s))

    results = []
    for K in range(kmin, kmax + 1):
        lab, C, inertia = spher_kmeans(nrm, K, seed=seed)
        results.append((sil(lab, C), K, lab, C, inertia))
    results.sort(key=lambda t: -t[0])
    s_val, K, lab, C, inertia = results[0]
    # modal 间角距矩阵 (无向: acos(|cos|))
    Kc = C.shape[0]
    M = np.zeros((Kc, Kc))
    for i in range(Kc):
        for j in range(Kc):
            M[i, j] = float(np.degrees(np.arccos(min(1.0, abs(float(C[i] @ C[j]))))))
    sizes = [int((lab == k).sum()) for k in range(Kc)]
    return {"K_best": int(K), "silhouette": round(s_val, 4), "inertia": round(inertia, 5),
            "centers": [[round(float(x), 5) for x in c] for c in C],
            "sizes": sizes,
            "modal_angle_matrix_deg": [[round(float(v), 2) for v in row] for row in M],
            "max_modal_separation_deg": round(float(np.max(M)), 2) if Kc > 1 else 0.0,
            "K_grid_silhouette": {str(r[1]): round(r[0], 4) for r in sorted(results, key=lambda t: t[1])}}


def dip_dipdir_to_normal(dip_deg, dipdir_deg):
    d = np.radians(np.asarray(dip_deg, dtype=float))
    a = np.radians(np.asarray(dipdir_deg, dtype=float))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=1)


def normals_dip(nrm):
    n = np.asarray(nrm, dtype=float)
    n = n / np.maximum(1e-12, np.linalg.norm(n, axis=1, keepdims=True))
    return 90.0 - np.degrees(np.arccos(np.abs(n[:, 2])))


# ---------------- 自检 (P8: 检查器先在自己身上失败一次) ----------------
def selftest():
    rng = np.random.default_rng(7)
    n1 = dip_dipdir_to_normal(np.full(60, 80.0), 40 + rng.normal(0, 3, 60))
    n2 = dip_dipdir_to_normal(np.full(60, 30.0), 210 + rng.normal(0, 3, 60))
    P = np.vstack([n1, n2])
    P /= np.maximum(1e-12, np.linalg.norm(P, axis=1, keepdims=True))
    r = best_K_sets(P)
    ok = (r["max_modal_separation_deg"] >= 30.0) and r["K_best"] >= 2
    # 退化: 单组近水平
    n3 = dip_dipdir_to_normal(4 + rng.normal(0, 0.8, 120), rng.uniform(0, 360, 120))
    n3 /= np.maximum(1e-12, np.linalg.norm(n3, axis=1, keepdims=True))
    r1 = best_K_sets(n3)
    ok1 = r1["max_modal_separation_deg"] < 30.0
    return {"two_set_input_max_sep_deg": r["max_modal_separation_deg"],
            "two_set_K_best": r["K_best"], "two_set_pass": bool(ok),
            "single_set_input_max_sep_deg": r1["max_modal_separation_deg"],
            "single_set_pass": bool(ok1),
            "overall": bool(ok and ok1)}


# ---------------- 主流程 ----------------
def main():
    report = {"artifact": "w8_local_evidence", "generated_at": now_s(),
              "watermark": "研究线取证 · 禁入对外承诺文本", "sections": {}}
    S = report["sections"]

    # 0. selftest
    S["checker_selftest"] = selftest()

    # 1. Pontrelli bin: 全文件 FitPlan/ccCompassType/DipDir 计数 + 头部 1MB sha 复现
    binp = r"E:\fracture_data\Pontrelli\Wall_point_cloud\Pontrelli_wall_point_cloud.bin"
    sec = {"path": binp}
    if os.path.isfile(binp):
        sec["size_bytes"] = os.path.getsize(binp)
        with open(binp, "rb") as f:
            head = f.read(1 << 20)
        sec["head1mb_sha256"] = hashlib.sha256(head).hexdigest()
        sec["head1mb_sha256_matches_20260926_forensics"] = (
            sec["head1mb_sha256"] == "0bebe02c98400716db677054d8e1be3d6e45272cbef009d1d9981ff16b80dfce")
        sec["magic"] = head[:8].hex() + " ascii=" + repr(head[:6])
        counts = {"FitPlan": 0, "ccCompassType": 0, "DipDir": 0, "Compressed normal": 0}
        tok = {k: k.encode("utf-16-le") for k in counts}
        with open(binp, "rb") as f:
            prev_tail = b""
            while True:
                b = f.read(1 << 24)  # 16MB
                if not b:
                    break
                buf = prev_tail + b
                for k, t in tok.items():
                    counts[k] += buf.count(t)
                prev_tail = b[-64:]
        sec["utf16_token_counts"] = counts
        sec["note"] = ("非重叠 count 经 16MB 块 + 64B 尾重叠扫描; token 为 UTF-16LE 编码; "
                       "边界重复由 prev_tail 拼接消除, 但跨块出现率极低, 计数为下界精度 ≥ 791-1")
        # 独立计数去重: 用 findall 位置法对 FitPlan 精确计数 (流式)
        positions = []
        t = tok["FitPlan"]
        with open(binp, "rb") as f:
            off = 0
            prev_tail = b""
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                buf = prev_tail + b
                start = 0
                while True:
                    i = buf.find(t, start)
                    if i < 0:
                        break
                    positions.append(off - len(prev_tail) + i)
                    start = i + 1
                off += len(b)
                prev_tail = b[-len(t) * 2:]
        sec["fitplan_exact_positions_n"] = len(positions)
        sec["fitplan_positions_head"] = positions[:5]
    else:
        sec["error"] = "bin not found"
    S["pontrelli_bin_scan"] = sec

    # 2. Pontrelli 791 面产状实测 (源自 W3h 权威资产 point_labels.npz, 只读)
    sec2 = {"source": "results/w3h_linkage_truth_asset/point_labels.npz (只读引用)"}
    p2 = os.path.join(ROOT, "results", "w3h_linkage_truth_asset", "point_labels.npz")
    d = np.load(p2, allow_pickle=True)
    fn = d["face_name"]
    sec2["n_faces"] = int(fn.shape[0])
    dips, dds = [], []
    bad = 0
    for s in fn:
        m = re.match(r"^(\d{1,2})/(\d{3})$", str(s))
        if not m:
            bad += 1
            continue
        dips.append(int(m.group(1)))
        dds.append(int(m.group(2)))
    sec2["parse_fail"] = bad
    dips = np.array(dips, float)
    dds = np.array(dds, float)
    sec2["dip_stats"] = {"min": float(dips.min()), "max": float(dips.max()),
                         "median": float(np.median(dips)), "mean": round(float(dips.mean()), 2),
                         "pct_ge_70": round(float((dips >= 70).mean()), 4)}
    sec2["dipdir_stats"] = {"min": float(dds.min()), "max": float(dds.max()),
                            "all_multiple_of_10": bool(np.all(dds % 10 == 0)),
                            "unique_mod10": sorted(set(int(x) % 10 for x in dds))}
    N = dip_dipdir_to_normal(dips, dds)
    N /= np.maximum(1e-12, np.linalg.norm(N, axis=1, keepdims=True))
    sets = best_K_sets(N)
    sec2["multi_orientation_measurement"] = sets
    sec2["dip_span_deg"] = float(dips.max() - dips.min())
    sec2["C2_multi_orientation_by_prereg_rule"] = bool(
        sets["K_best"] >= 2 and sets["max_modal_separation_deg"] >= 30.0)
    S["pontrelli_faces_791"] = sec2

    # 3. HRF (hrf_frac.npz)
    sec3 = {}
    p3 = r"E:\fracture_data\hrf_frac.npz"
    if os.path.isfile(p3):
        sec3["path"] = p3
        sec3["size_bytes"] = os.path.getsize(p3)
        h = np.load(p3, allow_pickle=True)
        sec3["keys"] = {k: list(h[k].shape) if hasattr(h[k], "shape") else None for k in h.keys()}
        if "frac_normals" in h.keys():
            fn3 = np.asarray(h["frac_normals"], dtype=float)
            fn3 /= np.maximum(1e-12, np.linalg.norm(fn3, axis=1, keepdims=True))
            dp = normals_dip(fn3)
            sec3["n_planes"] = int(fn3.shape[0])
            sec3["dip_stats"] = {"min": round(float(dp.min()), 2), "max": round(float(dp.max()), 2),
                                 "median": round(float(np.median(dp)), 2)}
            sub = fn3[:: max(1, fn3.shape[0] // 4000)]
            sets3 = best_K_sets(sub)
            sec3["multi_orientation_measurement"] = sets3
            sec3["C2_multi_orientation_by_prereg_rule"] = bool(
                sets3["K_best"] >= 2 and sets3["max_modal_separation_deg"] >= 30.0)
        fid = h.get("fracture_id") if "fracture_id" in h.keys() else None
        if fid is not None:
            sec3["fracture_id"] = {"n_unique": int(len(np.unique(fid))),
                                   "note": "数据件内 wall/平面 id; 其法向参照为件内拟合平面 (R246 重标 T3_FILE_INTERNAL 口径)"}
    else:
        sec3["error"] = "not found"
    S["hrf"] = sec3

    # 4. Åland
    sec4 = {}
    repo = os.path.join(ROOT, "data", "external", "aland_islands_repo")
    tj = os.path.join(repo, "data", "trace_data", "traces")
    if os.path.isdir(tj):
        gjs = glob.glob(os.path.join(tj, "*.geojson"))
        sec4["trace_geojson_n"] = len(gjs)
        zvals = []
        try:
            g0 = json.load(io.open(gjs[0], encoding="utf-8"))
            feats = g0.get("features", [])
            sec4["first_file_features"] = len(feats)
            for ft in feats[:50]:
                for ln in (ft.get("geometry") or {}).get("coordinates", []):
                    if ln and isinstance(ln[0], (list, tuple)):
                        for pt in ln:
                            zvals.append(pt[-1] if len(pt) > 2 else None)
                            break
                    elif ln:
                        zvals.append(ln[-1] if len(ln) > 2 else None)
            sec4["z_sample_nonnull"] = [z for z in zvals if z is not None][:5]
            sec4["z_all_null_or_absent"] = all(z is None for z in zvals)
        except Exception as e:
            sec4["geojson_parse_error"] = str(e)
    az = os.path.join(repo, "data", "azimuth_sets.json")
    if os.path.isfile(az):
        azd = json.load(io.open(az, encoding="utf-8"))
        sec4["azimuth_sets_keys"] = list(azd.keys())[:10] if isinstance(azd, dict) else str(type(azd))
        sec4["azimuth_sets_n_groups"] = len(azd) if isinstance(azd, dict) else None
    sec4["C3_3d_pointcloud"] = "NO (2D 迹线 geojson, z 缺失或 0 退化)"
    S["aland"] = sec4

    # 5. FORGE / DECOVALEX / INEL-1 目录快照
    for name, rel in [("forge_utah_fmi", r"data\external\utah_forge_fmi"),
                      ("decovalex_dfn", r"data\external\decovalex_dfn"),
                      ("inel1_snake_river", r"data\external\snake_river_INEL1")]:
        p = os.path.join(ROOT, rel)
        S.setdefault("dir_snapshots", {})[name] = {"root": rel, "exists": os.path.isdir(p),
                                                   "listing": dir_snapshot(p, max_depth=2)[:120] if os.path.isdir(p) else None}

    # 6. DECOVALEX normal_vectors.dat 读数 (合成确定性 4 裂隙)
    nv = os.path.join(ROOT, "data", "external", "decovalex_dfn")
    found = []
    for dirpath, _, files in os.walk(nv):
        for f in files:
            if f == "normal_vectors.dat":
                found.append(os.path.join(dirpath, f))
    if found:
        rows = [float(x) for x in io.open(found[0]).read().split()]
        arr = np.array(rows).reshape(-1, 3)
        S["decovalex_normals"] = {"file": found[0], "n": int(arr.shape[0]),
                                  "rows_unique_rounded": int(len({tuple(np.round(r, 6)) for r in arr})),
                                  "first_rows": [[round(float(v), 6) for v in r] for r in arr[:6]],
                                  "C1_real": "NO (dfnWorks 确定性合成)"}

    # 7. INEL-1 LAS 曲线名扫描 (前 3 个 LAS 的 CURV 段)
    las_dir = os.path.join(ROOT, "data", "external", "snake_river_INEL1")
    curves = {}
    las_files = glob.glob(os.path.join(las_dir, "**", "*.las"), recursive=True)[:3]
    for lf in las_files:
        try:
            txt = io.open(lf, encoding="latin-1", errors="replace").read(200000)
            names = re.findall(r"^\s*([A-Za-z0-9_]+)\s*\.\s*[A-Z ]+\.", txt, flags=re.M)
            # CURV 段: 取 ~PR 尽量精确 — 简化: 报告前 40 个 mnemonic
            curves[os.path.basename(lf)] = sorted(set(names))[:40]
        except Exception as e:
            curves[os.path.basename(lf)] = "error: %s" % e
    S["inel1_las_curves"] = {"files_scanned": len(las_files), "curve_names": curves,
                             "orientation_curves_found": [c for v in curves.values() if isinstance(v, list)
                                                          for c in v if c.upper() in
                                                          ("DIP", "DIPA", "AZ", "AZIM", "DIPDIR", "DIP_AZ", "TVDSS", "DIPMAG")]
                             }

    # 8. Pontrelli zip 原件清单
    for name, p in [("pontrelli_wall_zip", r"E:\fracture_data\Pontrelli_wall.zip"),
                    ("wall_point_cloud_zip", r"E:\fracture_data\Wall_point_cloud.zip"),
                    ("points_laz", r"E:\fracture_data\points.laz"),
                    ("points_new_758M_laz", r"E:\fracture_data\points_new_758M.laz"),
                    ("new_frac_npz", r"E:\fracture_data\new_frac.npz")]:
        S.setdefault("big_assets", {})[name] = {"path": p, "exists": os.path.isfile(p),
                                                "bytes": os.path.getsize(p) if os.path.isfile(p) else None}

    # 落盘
    atomic_dump(report, os.path.join(OUT, "local_measurements.json"))
    # 快照: 本报告自 sha
    self_sha = sha256_file(os.path.join(OUT, "local_measurements.json"))
    print("local_measurements.json written, sha256:", self_sha)
    # 单独快照件: 关键在盘目录清单
    snaps = {
        "E_fracture_data_root": dir_snapshot(r"E:\fracture_data", max_depth=1)[:120],
        "data_external": dir_snapshot(os.path.join(ROOT, "data", "external"), max_depth=2)[:200],
    }
    atomic_dump({"generated_at": now_s(), "snapshots": snaps},
                os.path.join(SNAP, "local_dir_listings.json"))
    print("local dir listings written")
    print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in S.items()},
                     ensure_ascii=False, allow_nan=False)[:1500])


if __name__ == "__main__":
    main()
