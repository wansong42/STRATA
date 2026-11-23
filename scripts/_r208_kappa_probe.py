# -*- coding: utf-8 -*-
"""R208 临时探针: 验证 r208_diag_em.em_mixture 的 κ-MLE 二分方向 (用完即删)。

极性 Bingham λ=(0,κ,κ) 的 κ-MLE 条件: E_κ[x1²] = d1 (d1 = 加权散布最大特征值)。
E_κ[x1²] 对 κ 单调**递增** (κ→0 ⇒ 1/3, κ→∞ ⇒ 1)。据此判 grad/二分方向。
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fractureflow.setlabel import spherical_kmeans        # noqa: E402


def ex1_of_kappa(kap):
    """E_κ[x1²] (数值求积, 与脚本内 grad 同实现)。"""
    th_, w_ = np.polynomial.legendre.leggauss(400)
    p = np.exp(-kap * (1.0 - th_ ** 2)) * w_
    return float((p * (th_ ** 2)).sum() / p.sum())


def main():
    print("=== 1) E_kappa[x1^2] 单调性 ===", flush=True)
    ks = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    e = [ex1_of_kappa(k) for k in ks]
    for k, v in zip(ks, e):
        print("  kappa=%7.2f  E[x1^2]=%.6f" % (k, v))
    print("  单调递增:", all(e[i] < e[i + 1] for i in range(len(e) - 1)), flush=True)

    print("\n=== 2) 真实站点上的 d1 (E-step 后各类簇散布最大特征值) ===", flush=True)
    RES = os.path.join(ROOT, "results", "r208_mixture_reboot")
    R207 = os.path.join(ROOT, "results", "r207_synth_truth_range")
    for path, sid in ((os.path.join(RES, "train_sites", "r208t_val_37_K4.npz"), "r208t_val_37_K4"),
                      (os.path.join(R207, "sites", "r207_K4_sig15.npz"), "r207_K4_sig15"),
                      (os.path.join(R207, "sites", "r207_K6_sig24.npz"), "r207_K6_sig24")):
        z = np.load(path)
        obs = z["masks"][0]
        no = z["nrm_full"][obs]
        no = no / (np.linalg.norm(no, axis=1, keepdims=True) + 1e-12)
        K = len(z["truth_centers"])
        cents, _ = spherical_kmeans(no, K, seed=42)
        w = 1.0 / np.clip(np.abs(no @ np.array([0.0, 0.0, 1.0])), 1e-6, None)
        w = np.clip(w / w.mean(), 0.1, 5.0)
        d1s = []
        for k in range(K):
            sel = np.abs(no @ cents[k]) > np.cos(np.radians(15.0))
            if sel.sum() < 3:
                continue
            S = (w[sel, None, None] * no[sel][:, :, None] * no[sel][:, None, :]).sum(0) / w[sel].sum()
            ev = np.linalg.eigvalsh(S)
            d1s.append(float(ev[-1]))
        d1 = float(np.mean(d1s))
        g_lo, g_hi = ex1_of_kappa(0.5) - d1, ex1_of_kappa(500.0) - d1
        print("  %-16s K=%d  d1=%.4f  grad(lo=0.5)=%+.4f  grad(hi=500)=%+.4f  "
              "-> 现码分支: %s"
              % (sid, K, d1, g_lo, g_hi,
                 "kappa=lo=0.5 (卡死)" if g_lo < 0 else
                 ("kappa=hi=500" if g_hi > 0 else "进入二分")), flush=True)


if __name__ == "__main__":
    main()
