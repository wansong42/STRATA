# -*- coding: utf-8 -*-
"""W99 替代 DFN 生成机制 · 公共底座 (边际转移规格 + 经验法向池 + 目录组装)。

W99 纪律 (results/w99_generator_mechanisms/prereg.json + amendments[0a]):
  - 禁改 dfn.py 既有行为: 只 import 其公开容器 DFNRealization 作目录载体;
  - 生成器 API 只接受 MarginalSpec, 不接受任何目标中心数组 —— 机制目录与目标
    目录的空间布局独立, 匹配只能经由观测摘要统计实现 (§3-G4: 不匹配参数, 只匹配
    可观测摘要);
  - 边际转移 = 逐组经验重抽 (amendment 0a): 目标 pop 法向的 w 分布非 vMF 形状
    (q10 0.903 vs vMF kappa-hat 0.809, 均值同 0.957), vMF 重拟合丢形状;
    逐组经验池重抽逐字保留目标法向边际。vMF (mu, kappa-hat) 仅作诊断字段随行;
  - 不吃 generate_dfn 的 P32 反解 + N_max=20000 自适应 r_min pilot 环, 改为 N 精确
    计数; 同上限取舍: N > 20000 硬 raise (理由同 dfn.py);
  - RNG 消费次序逐机制固定并写在其 docstring, 同 seed 逐位可复 (determinism pill)。
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import special

from ..dfn import DFNRealization

N_MAX = 20000  # 与 dfn.generate_dfn 同上限取舍 (prereg#generator_constraints_locked.n_max_policy)


def check_n(n):
    """N 精确计数上限守卫 (同 dfn.py N_max=20000 取舍, 不走自适应 pilot 环)。"""
    n = int(n)
    if n > N_MAX:
        raise ValueError(
            "generators: N=%d 超过硬上限 %d (与 dfn.generate_dfn 同上限取舍: "
            "界定内存/图构建成本; 需要更大目录属新决策, 须登记而非静默放行)" % (n, N_MAX))
    if n <= 0:
        raise ValueError("generators: N=%d 非正" % n)
    return n


def _unit(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    nl = np.linalg.norm(v, axis=axis, keepdims=True)
    nl = np.where(nl < 1e-12, 1e-12, nl)
    return v / nl


def fit_vmf_kappa(rbar):
    """vMF 浓度 kappa 的 MLE: Newton 解 A(kappa) = I1(kappa)/I0(kappa) = rbar (仅诊断用)。

    注意: 平面近似公式 rbar*(3-rbar^2)/(1-rbar^2) 不是 A 的逆 (W99 实测: 对
    rbar=0.9575 该式给 23.9, 真解 12.03 —— Bessel 比值为准)。rbar -> 1 或 Newton
    不收敛时回退平面近似并如实返回 used_fallback。
    """
    rbar = float(np.clip(rbar, 1e-9, 1.0 - 1e-9))
    if rbar < 1e-6:
        return 0.0, True
    k0 = rbar * (3.0 - rbar ** 2) / (1.0 - rbar ** 2)
    k = max(k0, 1e-6)
    for _ in range(80):
        a = special.i1(k) / special.i0(k)
        ap = 1.0 - a / k - a * a
        step = (a - rbar) / ap if abs(ap) > 1e-14 else 0.0
        k_new = k - step
        if k_new <= 0:
            k_new = k / 2.0
        if abs(k_new - k) < 1e-12 * max(1.0, k):
            k = k_new
            break
        k = k_new
    else:
        return max(k0, 1e-9), True
    if not np.isfinite(k) or k <= 0:
        return max(k0, 1e-9), True
    return float(k), False


@dataclass
class MarginalSpec:
    """机制生成的边际规格 (不含任何中心/位置信息)。

    domain         (3,) 域 extents (中心在原点约定, 与 r207 pop_centers 一致)
    n_pop          目录精确盘数
    normals_by_set 逐组经验法向池 (K 个 (n_k,3) 单位数组; 生成时有放回重抽,
                   amendment 0a: 逐字保留目标法向边际, 替代 vMF 重拟合)
    radii_sample   (n_pop,) 半径经验样本 (生成时有放回重抽)
    proportions    (K,) 组占比 (与池大小一致, 和为 1)
    vmf_mus        (K,3) 诊断字段: 逐组符号对齐均值方向 (不用于生成)
    vmf_kappas     (K,) 诊断字段: 逐组 vMF kappa MLE (Bessel 比值 Newton; 不用于生成)
    r_max_meta     站点 meta 的 dfn_params.r_max (M3 显式大断裂档用; 可 None)

    schema 级守卫: 本 dataclass 无 centers 字段, fit_marginals 签名不收中心 ——
    目标中心永不进入生成侧 (spec_guard)。
    """

    domain: tuple
    n_pop: int
    normals_by_set: tuple
    radii_sample: np.ndarray
    proportions: np.ndarray = None
    vmf_mus: np.ndarray = None
    vmf_kappas: np.ndarray = None
    r_max_meta: float = None

    def __post_init__(self):
        self.radii_sample = np.asarray(self.radii_sample, dtype=np.float64)
        self.domain = tuple(float(x) for x in self.domain)
        pools = tuple(np.asarray(p, dtype=np.float64) for p in self.normals_by_set)
        for p in pools:
            if p.ndim != 2 or p.shape[1] != 3:
                raise ValueError("MarginalSpec: normals_by_set 池须为 (n_k,3), got %s"
                                 % (p.shape,))
            if p.shape[0] == 0:
                raise ValueError("MarginalSpec: 空法向池")
            nl = np.linalg.norm(p, axis=1)
            if np.max(np.abs(nl - 1.0)) > 1e-6:
                raise ValueError("MarginalSpec: 法向池存在非单位行")
        self.normals_by_set = pools
        K = len(pools)
        if self.proportions is None:
            sizes = np.array([p.shape[0] for p in pools], dtype=np.float64)
            self.proportions = sizes / sizes.sum()
        self.proportions = np.asarray(self.proportions, dtype=np.float64)
        if self.proportions.shape[0] != K:
            raise ValueError("MarginalSpec: proportions K=%d != 池数 %d"
                             % (self.proportions.shape[0], K))
        if abs(self.proportions.sum() - 1.0) > 1e-9 or np.any(self.proportions < 0):
            raise ValueError("MarginalSpec: proportions 非归一化或含负值")
        if K > 64:
            raise ValueError("MarginalSpec: K=%d > 64 不合理 (spec_guard)" % K)
        if self.radii_sample.ndim != 1:
            raise ValueError("MarginalSpec: radii_sample 须 1-D (spec_guard)")
        if self.radii_sample.shape[0] != self.n_pop:
            raise ValueError("MarginalSpec: radii_sample n=%d != n_pop=%d"
                             % (self.radii_sample.shape[0], self.n_pop))
        if np.any(self.radii_sample <= 0):
            raise ValueError("MarginalSpec: radii_sample 存在非正半径")
        if self.vmf_mus is not None:
            self.vmf_mus = np.asarray(self.vmf_mus, dtype=np.float64)
            if self.vmf_mus.shape != (K, 3):
                raise ValueError("MarginalSpec: 诊断 vmf_mus 形状须 (K,3)")
        if self.vmf_kappas is not None:
            self.vmf_kappas = np.asarray(self.vmf_kappas, dtype=np.float64)
            if self.vmf_kappas.shape != (K,):
                raise ValueError("MarginalSpec: 诊断 vmf_kappas 形状须 (K,)")
        if self.r_max_meta is not None and not (self.r_max_meta > 0):
            raise ValueError("MarginalSpec: r_max_meta=%r 非正" % self.r_max_meta)

    @property
    def K(self):
        return len(self.normals_by_set)


def fit_marginals(pop_normals, pop_radii, pop_sets, domain, r_max_meta=None):
    """从目标总体目录拟合边际规格 (签名无 centers —— 目标中心永不进入生成侧)。

    逐组经验池 = 该组 pop 法向行 (amendment 0a); 诊断 vMF: 组内首成员为参照做
    符号对齐后取均值 (无向法向铁律, 同 obs_forward._sign_align 约定),
    kappa = fit_vmf_kappa MLE (Bessel 比值 Newton)。
    """
    normals = _unit(np.asarray(pop_normals, dtype=np.float64))
    radii = np.asarray(pop_radii, dtype=np.float64)
    sets = np.asarray(pop_sets)
    if normals.ndim != 2 or normals.shape[1] != 3 or normals.shape[0] != radii.shape[0] \
            or radii.shape[0] != sets.shape[0]:
        raise ValueError("fit_marginals: normals/radii/sets 行数不一致")
    labels = {}
    for s in np.unique(sets):
        labels[int(s)] = np.where(sets == s)[0]
    K = len(labels)
    pools = tuple(normals[labels[k]].copy() for k in sorted(labels))
    sizes = np.array([p.shape[0] for p in pools], dtype=np.float64)
    proportions = sizes / sizes.sum()
    mus = np.empty((K, 3))
    kappas = np.empty(K)
    for k, idx in enumerate(sorted(labels)):
        X = normals[labels[idx]]
        ref = X[0]
        Xa = X * np.sign(X @ ref)[:, None]
        m = Xa.mean(axis=0)
        rbar = float(np.linalg.norm(m))
        mus[k] = m / max(rbar, 1e-12)
        kappas[k], _ = fit_vmf_kappa(rbar)
    return MarginalSpec(domain=tuple(float(x) for x in domain),
                        n_pop=int(radii.shape[0]),
                        normals_by_set=pools, radii_sample=radii.copy(),
                        proportions=proportions, vmf_mus=mus, vmf_kappas=kappas,
                        r_max_meta=r_max_meta)


def sample_normals_by_set(spec, sets, rng):
    """逐组经验池有放回重抽法向 (组标签 0..K-1; 组升序消费 RNG)。"""
    sets = np.asarray(sets, dtype=int)
    n = sets.shape[0]
    normals = np.empty((n, 3), dtype=np.float64)
    for k in range(spec.K):
        sel = sets == k
        m = int(sel.sum())
        if m == 0:
            continue
        pool = spec.normals_by_set[k]
        normals[sel] = pool[rng.integers(0, pool.shape[0], size=m)]
    return normals


def assemble_catalog(spec, centers, sets, rng):
    """边际组装: 法向逐组经验重抽 -> 半径经验重抽 -> DFNRealization (RNG 次序固定)。"""
    centers = np.asarray(centers, dtype=np.float64)
    sets = np.asarray(sets, dtype=int)
    check_n(centers.shape[0])
    normals = sample_normals_by_set(spec, sets, rng)
    radii = rng.choice(spec.radii_sample, size=centers.shape[0], replace=True)
    return DFNRealization(centers=centers, normals=_unit(normals), radii=radii,
                          sets=sets, domain=tuple(spec.domain))


def assert_contained(centers, domain, tol=1e-9):
    """domain_containment 毒丸断言 (生成器自证)。"""
    half = np.asarray(domain, dtype=np.float64) / 2.0
    c = np.asarray(centers, dtype=np.float64)
    if c.shape[0] and (np.any(c < -half - tol) or np.any(c > half + tol)):
        raise ValueError("generators: 目录中心越出域 (domain_containment pill)")


def check_proportions_iid(spec, sets, tol=0.02):
    """proportion_fidelity 毒丸 (iid 指派机制 M1/M3 用): 实现组占比与 spec 偏差 <= tol。

    tol 为真实目录尺度 (N≈1.7万, 0.02 ≈ 6 sigma 多项噪声) 的绝对阈; 小样本玩具
    目录加 5 sigma 统计地板防误触发 (地板只在 N 小时生效, 真实尺度仍由 0.02 绑定)。
    """
    sets = np.asarray(sets, dtype=int)
    K = spec.K
    n = sets.shape[0]
    realized = np.bincount(sets, minlength=K) / float(n)
    p = np.asarray(spec.proportions)
    dev = float(np.max(np.abs(realized - p)))
    floor = 5.0 * float(np.sqrt(np.max(p * (1.0 - p)) / max(n, 1)))
    tol_eff = max(float(tol), floor)
    if dev > tol_eff:
        raise ValueError("generators: 实现组占比偏差 %.4f > %.4f (proportion_fidelity pill)"
                         % (dev, tol_eff))
    return dev
