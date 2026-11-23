# -*- coding: utf-8 -*-
"""P1 诚实口径解码器对比: oracle 标签 + 多种解码策略.

核心假设: 亲和度模型已学到强跨井模式 (AUC 0.92-0.94),
瓶颈在解码器 (谱聚类/加权投票不如 k-means).

本脚本:
  1. 用 oracle 标签 (全量 k-means 10 重启) 重训 GBT (消除训练噪声)
  2. 对比 4 种解码器:
     a. k-means on observed (当前基线)
     b. 谱聚类 (原 Phase 1)
     c. 层次聚类 (AgglomerativeClustering on affinity)
     d. 软亲和度投票 (直接利用 affinity matrix)
  3. 全部用 strict=True 评测

产出: results/p1_honest_decoder.json.
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def _sign_align(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts
    ref = pts[0]
    s = np.sign((pts * ref).sum(-1, keepdims=True))
    s[s == 0] = 1
    return pts * s


def mae_deg(pred, true, hid):
    cos = np.clip(np.abs((pred[hid] * true[hid]).sum(-1)), 0, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def oracle_labels_for_well(nrm, K, seed=42, restarts=10):
    """用全量数据做多重启 k-means 得 oracle 标签."""
    from fractureflow.setlabel import spherical_kmeans
    nrm_u = _unit(nrm)
    best_obj = -1e18
    best_assign = None
    for r in range(restarts):
        c, a = spherical_kmeans(nrm_u, K, seed=seed + r)
        obj = np.abs(nrm_u @ c.T).max(1).sum()
        if obj > best_obj:
            best_obj = obj
            best_assign = a
    return best_assign


def build_pair_features(nrm, pos):
    """为单井构建 12 维成对特征 + pair_indices."""
    L = len(nrm)
    nrm = _unit(nrm)
    # 所有对 (i < j)
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    if not pairs:
        return np.zeros((0, 12), np.float32), np.zeros((0, 2), int)

    i_idx = np.array([p[0] for p in pairs])
    j_idx = np.array([p[1] for p in pairs])
    n_i = nrm[i_idx]
    n_j = nrm[j_idx]

    # 特征
    cos_abs = np.abs((n_i * n_j).sum(-1))
    acos_abs_cos = np.arccos(np.clip(cos_abs, 0, 1))
    cos_angle_sqrt = np.sqrt(1 - cos_abs ** 2)

    # 深度差 (用 pos[:, 2] 作为深度)
    depth_diff = np.abs(pos[i_idx, 2] - pos[j_idx, 2])
    depth_range = max(pos[:, 2].std(), 1e-6)
    depth_diff_norm = depth_diff / depth_range

    # 倾角 + 倾向 (从法向计算)
    dip_i = np.arccos(np.clip(np.abs(n_i[:, 2]), 0, 1))
    dip_j = np.arccos(np.clip(np.abs(n_j[:, 2]), 0, 1))
    dip_diff = np.abs(dip_i - dip_j)

    # 倾向 (azimuth)
    azi_i = np.arctan2(n_i[:, 1], n_i[:, 0])
    azi_j = np.arctan2(n_j[:, 1], n_j[:, 0])
    azimuth_diff_circ = np.abs(np.arctan2(np.sin(azi_i - azi_j), np.cos(azi_i - azi_j)))

    # 局部密度 (KNN)
    from scipy.spatial.distance import cdist
    D = cdist(pos, pos)
    k_nbr = min(8, L - 1)
    local_density = np.array([np.sort(D[i])[1:k_nbr + 1].mean() for i in range(L)])
    local_density = 1.0 / (local_density + 1e-6)
    local_density_i = local_density[i_idx]
    local_density_j = local_density[j_idx]
    density_ratio = local_density_i / (local_density_j + 1e-6)

    # 共享邻域
    knn_sets = [set(np.argsort(D[i])[1:k_nbr + 1]) for i in range(L)]
    shared_neighbor_ratio = np.array([
        len(knn_sets[i] & knn_sets[j]) / max(len(knn_sets[i] | knn_sets[j]), 1)
        for i, j in zip(i_idx, j_idx)
    ]) if k_nbr > 0 else np.zeros(len(pairs))

    # Terzaghi 采样率差 (简化: 用深度差近似)
    terzaghi_diff = depth_diff_norm

    # 3D 欧氏距离
    euclid_dist = np.linalg.norm(pos[i_idx] - pos[j_idx], axis=-1)
    euclid_dist_norm = euclid_dist / max(pos.std(), 1e-6)

    # angle_to_vertical
    angle_to_vertical_i = dip_i

    features = np.stack([
        acos_abs_cos, depth_diff_norm, dip_diff, azimuth_diff_circ,
        local_density_i, local_density_j, shared_neighbor_ratio,
        terzaghi_diff, euclid_dist_norm, angle_to_vertical_i,
        density_ratio, cos_angle_sqrt
    ], axis=-1).astype(np.float32)

    pair_indices = np.stack([i_idx, j_idx], axis=-1)
    return features, pair_indices


def decode_kmeans_observed(nrm, pos, K, occ):
    """解码器 a: k-means on observed points (当前基线)."""
    from fractureflow.inference import set_aware_dirs
    from fractureflow.setlabel import spherical_kmeans
    nrm_u = _unit(nrm)
    L = len(nrm)
    K_eff = min(K, int(occ.sum()))
    if K_eff < 2:
        return None
    cents, assign_obs = spherical_kmeans(nrm_u[occ], K_eff, seed=42)
    cos_assign = np.abs(nrm_u @ cents.T)
    sid = cos_assign.argmax(1)
    dirs, _ = set_aware_dirs(pos, nrm, occ, sid, strict=True)
    return dirs


def decode_spectral(pair_indices, probs, nrm, pos, K, occ):
    """解码器 b: 谱聚类 on affinity matrix."""
    from fractureflow.inference import set_aware_dirs
    L = len(nrm)
    K_eff = min(K, int(occ.sum()))
    if K_eff < 2:
        return None

    # 构建亲和矩阵
    A = np.zeros((L, L))
    for (i, j), p in zip(pair_indices, probs):
        A[i, j] = p
        A[j, i] = p
    np.fill_diagonal(A, 1.0)

    # 归一化拉普拉斯
    d = A.sum(1)
    d[d < 1e-10] = 1e-10
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(L) - D_inv_sqrt @ A @ D_inv_sqrt

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    k_actual = min(K_eff + 2, L - 1)
    try:
        eigvals, eigvecs = eigsh(csr_matrix(L_norm), k=k_actual, which='SM', tol=1e-6, maxiter=500)
    except Exception:
        return None

    X = eigvecs[:, 1:K_eff + 1]
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X_norm = X / norms

    from fractureflow.setlabel import spherical_kmeans
    cents, assign = spherical_kmeans(X_norm, K_eff, seed=42)
    dirs, _ = set_aware_dirs(pos, nrm, occ, assign, strict=True)
    return dirs


def decode_hierarchical(pair_indices, probs, nrm, pos, K, occ):
    """解码器 c: 层次聚类 on affinity distance."""
    from fractureflow.inference import set_aware_dirs
    L = len(nrm)
    K_eff = min(K, int(occ.sum()))
    if K_eff < 2:
        return None

    # 距离矩阵 = 1 - affinity
    D = np.ones((L, L))
    for (i, j), p in zip(pair_indices, probs):
        D[i, j] = 1 - p
        D[j, i] = 1 - p
    np.fill_diagonal(D, 0)

    from sklearn.cluster import AgglomerativeClustering
    try:
        clust = AgglomerativeClustering(n_clusters=K_eff, metric='precomputed', linkage='average')
        assign = clust.fit_predict(D)
    except Exception:
        return None

    dirs, _ = set_aware_dirs(pos, nrm, occ, assign, strict=True)
    return dirs


def decode_soft_voting(pair_indices, probs, nrm, pos, K, occ):
    """解码器 d: 软亲和度投票 (观测点 k-means → 隐伏点加权投票)."""
    from fractureflow.inference import set_aware_dirs
    from fractureflow.setlabel import spherical_kmeans
    nrm_u = _unit(nrm)
    L = len(nrm)
    K_eff = min(K, int(occ.sum()))
    if K_eff < 2:
        return None

    # 观测点 k-means
    cents_obs, assign_obs = spherical_kmeans(nrm_u[occ], K_eff, seed=42)

    # 建立 (i,j) → prob 映射
    pair_prob = {}
    for (i, j), p in zip(pair_indices, probs):
        pair_prob[(i, j)] = p
        pair_prob[(j, i)] = p

    obs_idx = np.where(occ)[0]
    assign_all = np.full(L, -1, dtype=int)
    assign_all[occ] = assign_obs

    # 隐伏点: 对每个组, 计算与组内观测点的平均亲和度
    for i in np.where(~occ)[0]:
        group_scores = np.zeros(K_eff)
        group_counts = np.zeros(K_eff)
        for oi, j in enumerate(obs_idx):
            p = pair_prob.get((i, j), 0.5)
            g = assign_obs[oi]
            group_scores[g] += p
            group_counts[g] += 1
        avg_scores = group_scores / np.maximum(group_counts, 1)
        assign_all[i] = int(np.argmax(avg_scores))

    dirs, _ = set_aware_dirs(pos, nrm, occ, assign_all, strict=True)
    return dirs


def main():
    print("=" * 70)
    print("P1 诚实口径解码器对比 (oracle 标签 + strict)")
    print("=" * 70)

    # 加载数据
    real_pt = os.path.join(ROOT, "data/real/loaded_real_nets_setid.pt")
    all_nets = torch.load(real_pt, weights_only=False)
    bnets = [n for n in all_nets if "BS34" in str(n.get("wid", ""))]
    print(f"beishan: {len(bnets)} wells")

    K = 6  # 诚实口径最优 K
    seeds = 10
    obs_frac = 0.4

    # 预计算: 每井的 oracle 标签 + 特征
    print("\n预计算 oracle 标签 + 特征...")
    well_data = []
    for net in bnets:
        nrm = np.asarray(net.get("nrm_full", net.get("nrm")), float)
        pos = np.asarray(net["pos"], float)
        sid = oracle_labels_for_well(nrm, K, restarts=10)
        feats, pair_idx = build_pair_features(nrm, pos)
        # 标签: 同组=1, 异组=0
        labels = (sid[pair_idx[:, 0]] == sid[pair_idx[:, 1]]).astype(np.int32)
        well_data.append({
            "nrm": nrm, "pos": pos, "sid": sid,
            "features": feats, "pair_indices": pair_idx, "labels": labels,
            "wid": str(net.get("wid", "")),
        })
    print(f"  完成: {len(well_data)} wells")

    # 解码器列表
    decoders = [
        ("kmeans_obs", decode_kmeans_observed),
        ("spectral", decode_spectral),
        ("hierarchical", decode_hierarchical),
        ("soft_voting", decode_soft_voting),
    ]

    # 留一井交叉验证
    all_results = {name: [] for name, _ in decoders}

    t0 = time.time()
    for left_out_idx in range(len(well_data)):
        # 训练 GBT (用 oracle 标签)
        train_wells = [w for i, w in enumerate(well_data) if i != left_out_idx]
        X_tr = np.concatenate([w['features'] for w in train_wells], axis=0)
        y_tr = np.concatenate([w['labels'] for w in train_wells], axis=0)

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            subsample=0.8, max_features='sqrt', random_state=42
        )
        clf.fit(X_tr, y_tr)

        # 测试井
        te = well_data[left_out_idx]
        X_te = te['features']
        probs = clf.predict_proba(X_te)[:, 1]

        # 每种掩码 seed
        for sd in range(seeds):
            rng = np.random.default_rng(999 + 1000 * sd)
            occ = rng.random(len(te['nrm'])) < obs_frac
            if occ.sum() < 2 or (~occ).sum() < 1:
                continue

            for dec_name, dec_fn in decoders:
                try:
                    if dec_name == "kmeans_obs":
                        dirs = dec_fn(te['nrm'], te['pos'], K, occ)
                    else:
                        dirs = dec_fn(te['pair_indices'], probs, te['nrm'], te['pos'], K, occ)
                    if dirs is not None:
                        mae = mae_deg(dirs, te['nrm'], ~occ)
                        all_results[dec_name].append(mae)
                except Exception:
                    pass

        if (left_out_idx + 1) % 5 == 0:
            print(f"  进度: {left_out_idx + 1}/{len(well_data)}")

    elapsed = time.time() - t0

    # 汇总
    print(f"\n{'='*70}")
    print(f"结果汇总 (K={K}, strict, {seeds} seeds, {len(well_data)} wells LOO)")
    print(f"{'='*70}")
    print(f"{'解码器':20s}  {'MAE_mean':>10s}  {'MAE_std':>10s}  {'n':>6s}")
    print("-" * 55)

    summary = {}
    for name, maes in all_results.items():
        if maes:
            mean_m = float(np.mean(maes))
            std_m = float(np.std(maes))
            summary[name] = {"mean": round(mean_m, 3), "std": round(std_m, 3), "n": len(maes)}
            print(f"{name:20s}  {mean_m:10.3f}  {std_m:10.3f}  {len(maes):6d}")

    print(f"\n耗时: {elapsed:.1f}s")

    # 保存
    out_path = os.path.join(ROOT, "results/p1_honest_decoder.json")
    with open(out_path, "w") as f:
        json.dump({
            "description": "P1 诚实口径解码器对比 (oracle 标签 + strict)",
            "K": K,
            "n_seeds": seeds,
            "obs_frac": obs_frac,
            "n_wells": len(well_data),
            "results": summary,
        }, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
