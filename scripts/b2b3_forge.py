# -*- coding: utf-8 -*-
"""B2-B3: FORGE 双井 FMI 组系表评测."""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fractureflow.set_table_eval import set_table_score, estimate_set_table
from fractureflow.setlabel import _sign_align

print("=" * 60)
print("B2-B3: FORGE 双井 FMI")
print("=" * 60)

forge_path = "data/external/utah_forge_fmi/forge_fmi_2wells.pt"
if not os.path.exists(forge_path):
    print(f"[SKIP] {forge_path} not found")
    sys.exit(0)

data = torch.load(forge_path, weights_only=False)
nets = data["nets"] if isinstance(data, dict) and "nets" in data else data
print(f"Wells: {len(nets)}")

# 检查ftype字段
if isinstance(nets[0], dict) and "ftype" in nets[0]:
    ftypes = [n["ftype"][0] if len(n["ftype"]) > 0 else "unknown" for n in nets]
    print(f"ftype values: {set(ftypes)}")

K_list = [4, 6, 8, 12]
n_seeds = 10

results = {}
for K in K_list:
    seed_errs = []
    for s in range(n_seeds):
        well_errs = []
        for net in nets:
            pos = np.asarray(net["pos"], dtype=float)
            nrm = np.asarray(net.get("nrm_full", net["nrm"]), dtype=float)
            nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
            L = len(pos)
            if L < 5:
                continue
            rng = np.random.default_rng(999 + 1000 * s)
            occ = rng.random(L) < 0.4
            if occ.sum() < 2 or (~occ).sum() == 0:
                continue

            # 预测: 观测-only kmeans
            pred = estimate_set_table(pos, nrm, occ, K=K, seed=42, strict=True)

            # 真值: ftype 标签 (如果有)
            truth_centers = None
            if "ftype" in net:
                types = np.asarray(net["ftype"])
                if types.dtype == object or types.dtype.kind in ("U", "S", "O"):
                    # 字符串标签 -> 数值编码
                    unique_types = list(set(types))
                    type_map = {t: i for i, t in enumerate(unique_types)}
                    types_num = np.array([type_map[t] for t in types])
                else:
                    types_num = types.astype(int)
                Kt = int(types_num.max()) + 1
                tc = np.zeros((Kt, 3))
                for k in range(Kt):
                    sel = types_num == k
                    if sel.sum():
                        pts = nrm[sel]
                        ps = _sign_align(pts)
                        tc[k] = ps.mean(0)
                        tc[k] /= np.linalg.norm(tc[k]) + 1e-12
                truth_centers = tc

            if truth_centers is not None:
                score = set_table_score(pred["centers"], truth_centers)
                well_errs.append(score["modal_err_deg"])
        if well_errs:
            seed_errs.append(float(np.mean(well_errs)))

    if seed_errs:
        mean_err = float(np.mean(seed_errs))
        std_err = float(np.std(seed_errs, ddof=1))
        results[K] = {"modal_err_mean": round(mean_err, 3),
                      "modal_err_std": round(std_err, 3),
                      "n_seeds": len(seed_errs)}
        print(f"  K={K:2d}: modal_err = {mean_err:.2f} +/- {std_err:.2f} deg")
    else:
        print(f"  K={K:2d}: no valid seeds")

# 保存
out_path = os.path.join(ROOT, "results/set_table_forge.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, allow_nan=False)
print(f"\n结果: {json.dumps(results, indent=2, allow_nan=False)}")
print(f"保存: {out_path}")
