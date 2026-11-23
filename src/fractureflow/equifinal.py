# -*- coding: utf-8 -*-
"""W91 · WP-1 等解释集合构造器 (ABC pilot, 一站一制度)。

新主线 estimand 与集合定义 (技术方案 §1.1 / §3-G2; 判据权威 =
``results/mainline_plan_20261005/wp_register.json#work_packages[WP-1]``):

* 等解释集合 ``𝓕(D_obs, ε) = { F : ρ( S(O(F)), S(D_obs) ) ≤ ε }`` —— 等解释
  **只在观测空间定义**; 禁用『结构相似』当判据。
* 三腿构造: A 参数层 ABC 后验 (每个接受 θ 一个实现) / B 实现层 (每 θ 抽 M 个
  seed, 逐 seed 过 ε, 分离 epistemic/aleatoric) / C 结构扰动层 (局部增删旋转
  缩放, 逐扰动过 ε)。
* eps 规则: ε = 条件化臂自身先验池 ρ 的第 q 百分位 (q 预注册网格), 目标 ESS ≥ 200;
  ε 敏感性曲线是一等交付 (J1.5, G8-E2 出口判据载体)。

模块纪律 (wp_register#WP-1.hard_prohibition, AST 级守卫 + 毒丸):
  **本文件是条件化器唯一载体, 全文禁止出现真值键**
  ``pop_centers / pop_normals / pop_radii / pop_sets / truth_sets / truth_centers``
  (字符串字面量也算)。守卫 = ``scripts/w91_leakage_guard.py#ast_guard_scan``;
  运行时守卫 = :func:`assert_no_truth_keys` (条件化器消费的一切映射必须过闸)。
  真值侧参照 (J1.2 truth-forward / 泄漏有牙对照) 一律住在 ``scripts/w91_*.py``,
  不进本模块 —— 本文件连读取它们的资格都没有, 这就是"牙"。

观测制度 (r207 一站一制度, 全部预注册于 results/w91_equifinal_ensemble/prereg.json):
  站 = r207_K4_sig15; 制度 = 站点 catalog × mask s=0 (精确 0.4 均匀抽签);
  井位从 ``pos`` 的 (x,y) **精确**重构 (竖直井交点 (x,y) = 井位, r207_gen 同构),
  井底 z0 = -130 m (meta#observation "穿全高 260m"), 井轴 = (0,0,1)。
  条件化器只读 npz 的 ``pos / nrm_full / masks`` 三键 (wp_register#WP-1.inputs)。

W90 移交四条的吸收方式:
  ① r_attach 无默认 → 本模块常量 ``R_ATTACH_MAIN`` 显式预注册, API 不设默认;
  ② p̂ 随行限定语 → :func:`connection_stats` 返回值带 prior 标识 / eps / ESS;
  ③ conformal/coverage 在 WP-1/2 消费 → 本单只交付集合与 p̂, 不做覆盖主张;
  ④ r207 编录无半径 → 观测侧半径策略 = **四族摘要全部无半径** (成员网络半径
    只由生成器先验产生, 仅用于 C(A,B) 评估), 见 prereg#observation_radius_policy。

钻孔前向 = :func:`drill_wells` —— ``obs_forward.virtual_borehole`` 逐盘公式的
向量化逐算符镜像 (t = n·(C-x0)/nd, dist=|P-C|, 命中 dist ≤ R), 等价性验证见
``results/w91_equifinal_ensemble/instrument_equivalence.json`` (vs 库算子逐位)。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dfn import DFNRealization, SetTable, generate_dfn
from .geometry import normal_to_dip_dipdir
from .intersect_disk import W15InputError, build_disk_graph
from .obs_forward import virtual_borehole_multi
from .terzaghi import terzaghi_weights

__all__ = [
    "FORBIDDEN_KEYS",
    "ALLOWED_NPZ_KEYS",
    "assert_no_truth_keys",
    "load_station_observation",
    "wells_from_pos",
    "drill_wells",
    "simulate_catalog",
    "sample_theta",
    "theta_to_arrays",
    "catalog_summaries",
    "family_distances",
    "rho_of",
    "mad_of_pool",
    "abc_pool",
    "legB_members",
    "legC_members",
    "perturb_network",
    "ensemble_connection_flags",
    "connection_stats",
    "divergence_table",
    "ConditionerConfig",
    "EquifinalError",
]


class EquifinalError(ValueError):
    """条件化器输入/契约违例(含真值键闯入)。"""


# ---------------------------------------------------------------------------
# 泄漏守卫: 真值键清单 + 运行时闸
# ---------------------------------------------------------------------------

#: 条件化器禁读的真值键 (wp_register#WP-1.hard_prohibition 原文清单 + npz 真值心)。
FORBIDDEN_KEYS: Tuple[str, ...] = (
    "pop_centers", "pop_normals", "pop_radii", "pop_sets",
    "truth_sets", "truth_centers",
)
#: npz 允许读取的键 (wp_register#WP-1.inputs: 仅此三键)。
ALLOWED_NPZ_KEYS: Tuple[str, ...] = ("pos", "nrm_full", "masks")


def assert_no_truth_keys(obj) -> None:
    """运行时守卫: 条件化器消费的映射里出现任何真值键 ⇒ 立即 raise。

    递归面 (G2 · §5.1 修订): dict 的**键**与 dict/list/tuple 的**逐元素**都走。
    改前只递归 dict.values(), 于是 {"x": [{"pop_centers": ...}]} 这类"真值键藏在
    list 元素里"的载荷被静默放行 —— 那是构造性漏检, 不是低概率侥幸 (反例已由
    G2 pre_flight 的 truth_bearing_control 在旧实现上实测为"放行")。
    set/dict_keys/numpy 数组等容器**仍不递归**: 记为已知残余 (G2 §5.4 登记件
    results/g2_guardians_p2_batch/known_not_covered_register.json 条目 RN-1),
    禁把本守卫读成"任意嵌套都拦得住"。
    """
    if isinstance(obj, dict):
        for k in obj.keys():
            if isinstance(k, str) and k in FORBIDDEN_KEYS:
                raise EquifinalError(
                    "泄漏守卫: 条件化器输入含真值键 %r —— wp_register#WP-1 "
                    "hard_prohibition 违例, 拒绝消费" % k)
        for v in obj.values():
            assert_no_truth_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_no_truth_keys(v)


# ---------------------------------------------------------------------------
# 预注册常量 (与 results/w91_equifinal_ensemble/prereg.json 同值; 改判据须走修正案)
# ---------------------------------------------------------------------------

#: B2 解绑 (G15 relay-1): 站点身份不再是装载器的回落常量。
#: STATION_ID 保留为**已知站点册里的一名** (历史消费者仍读这个属性名 —— 删它会把
#: 无关的下游脚本一起打断, 那不是本单的授权范围), 但它不再决定 obs["station_id"]
#: 的取值。装载器要求显式入参或由站点规格 (npz 基名) 派生, 不在册即 SystemExit。
STATION_ID = "r207_K4_sig15"
#: 站点册成员 (合成册) = results/r207_synth_truth_range/sites/*_meta.json 的 site_id,
#: 逐名运行时核对 (核对件 = prereg_amendment_u1_reanchor.json 的 registry_provenance)。
#: R2 分册 (G27): 这一层就是原来的册, 成员与出处声明**零改**; 真实域站另立一层,
#: 不再往里塞 (把真实域站塞进 r207 册 = 违该册自身出处声明, 见 G25 决策账 R1 的代价)。
SYNTHETIC_STATION_IDS: Tuple[str, ...] = (
    "r207_K3_sig08", "r207_K3_sig15", "r207_K3_sig24", "r207_K4_sig08", "r207_K4_sig15", "r207_K4_sig24", "r207_K6_sig08", "r207_K6_sig15", "r207_K6_sig24",
)
#: ── R2 第二层: 真实域册 ──────────────────────────────────────────────────
#: 每名一律经各线 prereg 登记收录; 登记规程 =
#: results/g27_station_registry_split/prereg.json#r2_registry_semantics.tiers.REAL_DOMAIN,
#: 逐名出处核对件 = results/g27_station_registry_split/registry_provenance_check.json。
#: 禁凭空手打: 站名必须在 REAL_DOMAIN_STATION_PROVENANCE 有可解析 source_pointer,
#: 且指针取值 == 站名 (核对件现场断言)。装载器不为此新增任何 I/O ——
#: 解析仍只按 npz 基名与显式入参 (在册牙齿 P6 的口径不变)。
REAL_DOMAIN_STATION_IDS: Tuple[str, ...] = (
    "forge_58-32",
)
#: 站名 -> 出处指针 (路径#键)。收录授权归架构师排程 (G27 verdict#needs_owner N-1)。
REAL_DOMAIN_STATION_PROVENANCE: Dict[str, Dict] = {
    "forge_58-32": {
        "source_pointer": "results/g16_j60_envelope_entry/prereg.json#station_and_regime.station",
        "registered_by": "G27 · 甲方 2026-10-16 采纳 R2 分册 (与技术方案 §4.3 数据隔离同向)",
        "role": "real_domain",
    },
}
#: 合并册 (向后兼容名; 历史消费者读它) = 合成册 ∪ 真实域册。
KNOWN_STATION_IDS: Tuple[str, ...] = SYNTHETIC_STATION_IDS + REAL_DOMAIN_STATION_IDS
MASK_IDX = 0                      # 一站一制度: station × mask s=0
OBS_FRAC = 0.4                    # r207 masking 协议 (R204 verbatim)
WELL_Z0 = -130.0                  # meta#observation: 竖直井穿全高 260m, 井底 = -Lz/2
DOMAIN = (270.0, 270.0, 260.0)    # 巡查设计参数 (meta#dfn_params.domain), 非推断对象
N_WELLS_EXPECTED = 36             # 6×6 网格 (meta#observation); 重构校验用
MIN_HITS_FOR_RHO = 12             # 低于此命中数的模拟 catalog 直接判 ρ=哨兵 (拒绝)
RHO_SENTINEL = 1e6                # 退化 catalog 的距离哨兵 (确定性拒绝)
N_MAX_DOMAIN = 20000              # 模拟器有效域: generate_dfn N_max 语义上限
#   (r207 构建器同款断言 dfn.n_fractures<=20000) —— 超域 θ 自动拒绝 (修正案 W91-0b):
#   先验抽样后先解析估计 N (镜像 generate_dfn 自适应环, 独立 rng), N_est > 上限
#   ⇒ 不生成, ρ=哨兵; 生成后 n_network > 上限 ⇒ 同哨兵。两道筛都只按模拟器
#   语义域拒绝, 不看任何观测/真值。

R_ATTACH_MAIN = 10.0              # 工程特征尺度 (预注册主档, m)
R_ATTACH_SENSITIVITY = (5.0, 20.0)

AZ_BINS = 12                      # 产状族: 12 方位 × 3 倾角带
DIP_BINS = 3
SPACING_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

Q_GRID = (0.05, 0.02, 0.01, 0.005, 0.002)   # ε = 先验池 ρ 的 q 分位
Q_MAIN = 0.01                                # 主操作点 (ESS 目标 ≥200)

M_SEEDS_LEGB = 4                  # 腿 B: 每接受 θ 抽 M 个候选 seed (逐 seed 过 ε)
N_LEGB_PARENTS = 200              # 腿 B/C 只挂在按池序的前 N_LEGB_PARENTS 个接受 θ 上
N_PROPOSALS_LEGC = 4              # 腿 C: 每父成员的扰动提案数
N_TRUTHFWD_DRAWS = 200            # J1.2 真值前向的新鲜掩码抽签数 (评估侧, 不进本模块运行)

ENDPOINT_SEPARATIONS = (10.0, 20.0, 40.0, 80.0)
ENDPOINT_ORIENTATIONS = ("vertical", "horizontal", "diag45")
N_ENDPOINT_OFFSETS = 2
ENDPOINT_SEED = 20261006
ENDPOINT_ANCHOR = (0.0, 0.0, 0.0)
ENDPOINT_OFFSET_CUBE = 30.0       # 偏移均匀立方 [−30, 30]³

SEED_POOL_MAIN = 91001
SEED_POOL_HEAVY = 91002
SEED_POOL_LIGHT = 91003

#: 先验 (技术方案 G2: 宽先验; 尺寸尾部两套敏感性先验 尾重/尾轻)。
PRIORS: Dict[str, Dict] = {
    "wide": {
        "p32_loguniform": (0.02, 1.5),
        "beta_uniform": (1.2, 4.5),
        "r_max_loguniform": (2.0, 60.0),
        "r_min_loguniform": (0.2, 8.0),
        "kappa_loguniform": (2.0, 200.0),
        "K_choices": (1, 2, 3, 4, 5, 6),
        "note": "主臂: 全宽先验, 不以任何真值参数为中心",
    },
    "heavy_tail": {
        "p32_loguniform": (0.02, 1.5),
        "beta_uniform": (1.2, 2.6),
        "r_max_loguniform": (10.0, 60.0),
        "r_min_loguniform": (0.2, 8.0),
        "kappa_loguniform": (2.0, 200.0),
        "K_choices": (1, 2, 3, 4, 5, 6),
        "note": "敏感性臂: 尾部重 (β 小, r_max 大)",
    },
    "light_tail": {
        "p32_loguniform": (0.02, 1.5),
        "beta_uniform": (2.6, 4.5),
        "r_max_loguniform": (2.0, 15.0),
        "r_min_loguniform": (0.2, 8.0),
        "kappa_loguniform": (2.0, 200.0),
        "K_choices": (1, 2, 3, 4, 5, 6),
        "note": "敏感性臂: 尾部轻 (β 大, r_max 小)",
    },
}


# ---------------------------------------------------------------------------
# 观测侧装载 (白名单 + 运行时守卫)
# ---------------------------------------------------------------------------

def station_registry_tier(station_id: Optional[str]) -> str:
    """站名所属册层: SYNTHETIC / REAL_DOMAIN / UNREGISTERED (零 I/O, 只读模块内常量)。

    给分轨报告与下游守卫用 —— 合成域与真实域读数禁互当证据 (甲方红线)。不进 obs,
    装载器返回键集零改 (载荷八键不变是换代合格条件之一)。
    """
    if station_id in SYNTHETIC_STATION_IDS:
        return "SYNTHETIC"
    if station_id in REAL_DOMAIN_STATION_IDS:
        return "REAL_DOMAIN"
    return "UNREGISTERED"


def _resolve_station_id(npz_path: str, station_id: "Optional[str]") -> str:
    """站点身份 = 站点规格 (npz 基名) 派生; 显式入参只在与规格一致时放行, 未知即 SystemExit。

    两条禁令写死在这里:
      * 禁静默回落到 STATION_ID —— 那是 B2 缺陷本身 (标签与所载数据不挂钩时,
        任何"站点面读数"都是错标读数; 病因见 results/w122_station_id_decoupling/)。
      * 禁静默接受"显式入参 != 站点规格" —— 那等于把 B2 从常量搬到调用参数上。
    """
    derived = os.path.splitext(os.path.basename(npz_path))[0]
    if station_id is not None and station_id != derived:
        if derived in KNOWN_STATION_IDS:
            raise SystemExit(
                "站点身份冲突: 显式 station_id=%r 与站点规格派生 %r 不同 —— 拒绝。"
                " (解绑后标签必须与数据同源, 禁靠入参改写已注册的站身份)"
                % (station_id, derived))
        sid = station_id
    else:
        sid = derived
    if sid not in KNOWN_STATION_IDS:
        raise SystemExit(
            "未知站点 %r: 不在 KNOWN_STATION_IDS 册内 (B2 解绑后禁静默回落到 %r)。"
            " 处置 = 把该站登记进站点规格册 (sites/*_meta.json 的 site_id) 形, "
            " 或调用时显式传一个已注册的 station_id。" % (sid, STATION_ID))
    return sid


def load_station_observation(npz_path: str, mask_idx: int = MASK_IDX,
                        station_id: Optional[str] = None) -> Dict:
    """装载观测制度数据 —— 只读 ``ALLOWED_NPZ_KEYS`` 三键, 其余一律不碰。

    返回 dict 键: pos / nrm_full / mask / well_xy / well_index / depth。
    (well_index/depth 由 pos 精确重构 —— 竖直井交点性质, 见 :func:`wells_from_pos`)
    """
    sid = _resolve_station_id(npz_path, station_id)
    from numpy.lib.npyio import NpzFile
    with NpzFile(npz_path, allow_pickle=False) as z:
        missing = [k for k in ("pos", "nrm_full", "masks") if k not in z.files]
        if missing:
            raise EquifinalError("站点 npz 缺允许键: %s" % missing)
        pos = np.asarray(z["pos"], dtype=np.float64)
        nrm_full = np.asarray(z["nrm_full"], dtype=np.float64)
        masks = np.asarray(z["masks"], dtype=bool)
    if pos.ndim != 2 or pos.shape[1] != 3 or nrm_full.shape != pos.shape:
        raise EquifinalError("pos/nrm_full 形状非法: %s / %s" % (pos.shape, nrm_full.shape))
    if not (0 <= mask_idx < masks.shape[0]):
        raise EquifinalError("mask_idx 越界: %d / %d" % (mask_idx, masks.shape[0]))
    mask = masks[mask_idx]
    well_xy, well_index = wells_from_pos(pos)
    obs = {
        "station_id": _resolve_station_id(npz_path, station_id),
        "mask_idx": int(mask_idx),
        "pos": pos,
        "nrm_full": nrm_full,
        "mask": mask,
        "well_xy": well_xy,
        "well_index": well_index,
        "depth": pos[:, 2].copy(),
        "source_npz": os.path.basename(npz_path),
    }
    assert_no_truth_keys(obs)
    return obs


def wells_from_pos(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从交点坐标**精确**重构竖直井位 (只用 pos; r207_gen 同构性质)。

    竖直井 x0=(wx,wy,z0), d=(0,0,1) ⇒ 交点 P=(x0 + t·d) 的 (x,y) 与井位逐位相同,
    故 pos 的 (x,y) 精确等值类 = 井位集合。返回 (well_xy (W,2) 按字典序升序,
    well_index (N,) 每交点所属井号)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    xy = np.ascontiguousarray(pos[:, :2])
    view = xy.view([("x", np.float64), ("y", np.float64)]).ravel()
    uniq, inverse = np.unique(view, return_inverse=True)
    well_xy = np.stack([uniq["x"], uniq["y"]], axis=1)
    return well_xy, np.asarray(inverse, dtype=np.int64)


# ---------------------------------------------------------------------------
# 钻孔前向 (obs_forward.virtual_borehole 的向量化逐算符镜像)
# ---------------------------------------------------------------------------

def drill_wells(well_xy: np.ndarray, centers: np.ndarray, normals: np.ndarray,
                radii: np.ndarray, z0: float = WELL_Z0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """36 口竖直井 × 盘网络 → 命中记录 (井号, 交点 z, 交点坐标, 盘号)。

    与 ``obs_forward.virtual_borehole`` 逐盘公式逐算符同构:
    ``nd = n·d``; ``|nd| < 1e-12`` 跳过; ``t = n·(C-x0)/nd``; ``t < 0`` 跳过;
    ``P = x0 + t·d``; 命中 ⟺ ``|P-C| ≤ R`` (含等号, 与库一致)。
    竖直井 ⇒ nd = nz, t = n·(C-x0)/nz, P = (wx, wy, z0+t)。
    返回按 (井序, 盘号升序) 排列。判交与捕获口径不含本函数 ——
    这里只产出 catalog (观测层), 不是 QoI。
    """
    c = np.asarray(centers, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3 or n.shape != c.shape or r.shape != (c.shape[0],):
        raise EquifinalError("网络数组形状非法: %s / %s / %s" % (c.shape, n.shape, r.shape))
    well_xy = np.asarray(well_xy, dtype=np.float64)
    if well_xy.ndim != 2 or well_xy.shape[1] != 2:
        raise EquifinalError("well_xy 形状非法: %s" % (well_xy.shape,))
    nd = n[:, 2]
    ok_base = np.abs(nd) >= 1e-12
    hit_w: List[np.ndarray] = []
    hit_p: List[np.ndarray] = []
    hit_i: List[np.ndarray] = []
    for w in range(well_xy.shape[0]):
        wx = float(well_xy[w, 0])
        wy = float(well_xy[w, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            # t = n·(C-x0)/nd, x0=(wx,wy,z0): 逐项加和顺序与 np.dot(n, C-x0) 一致
            t = (n[:, 0] * (c[:, 0] - wx)
                 + n[:, 1] * (c[:, 1] - wy)
                 + n[:, 2] * (c[:, 2] - z0)) / nd
        ok = ok_base & (t >= 0.0)
        pz = z0 + t
        dx = wx - c[:, 0]
        dy = wy - c[:, 1]
        dz = pz - c[:, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        ok = ok & (dist <= r)
        idx = np.nonzero(ok)[0]
        if idx.size:
            hit_w.append(np.full(idx.size, w, dtype=np.int64))
            hit_i.append(idx)
            hit_p.append(np.stack([np.full(idx.size, wx), np.full(idx.size, wy),
                                   pz[idx]], axis=1))
    if hit_w:
        return (np.concatenate(hit_w), np.concatenate(hit_p)[:, 2],
                np.concatenate(hit_p), np.concatenate(hit_i))
    e = np.empty(0, dtype=np.int64)
    return e, np.empty(0), np.empty((0, 3)), e.copy()


def _merge_first_well(hit_w: np.ndarray, hit_z: np.ndarray, hit_p: np.ndarray,
                      hit_i: np.ndarray) -> Tuple[np.ndarray, ...]:
    """跨井命中去重, 首井获胜 (virtual_borehole_multi 合并语义:
    catalog = 唯一盘集合, 位置取第一个命中它的井的交点)。"""
    if hit_i.size == 0:
        return hit_w, hit_z, hit_p, hit_i
    _, first_pos = np.unique(hit_i, return_index=True)
    sel = np.sort(first_pos)
    return hit_w[sel], hit_z[sel], hit_p[sel], hit_i[sel]


def simulate_catalog(theta: "Theta", well_xy: np.ndarray, seed: int
                     ) -> Dict[str, np.ndarray]:
    """θ → 网络 → 钻孔 → 首井合并 → 精确 0.4 均匀掩码 → 模拟 catalog (观测空间)。

    掩码公式与 r207 masking 协议逐字同构: ``max(2, int(n_hits*OBS_FRAC))`` 个
    命中均匀保留 (R204 verbatim); 抽签 rng = default_rng([20261006, seed])。
    返回 dict: well_index / depth / normals / positions (全部 mask 后子集)。
    """
    dfn = theta_to_network(theta, seed)
    hit_w, hit_z, hit_p, hit_i = drill_wells(well_xy, dfn.centers, dfn.normals,
                                             dfn.radii)
    hit_w, hit_z, hit_p, hit_i = _merge_first_well(hit_w, hit_z, hit_p, hit_i)
    n_hits = int(hit_w.size)
    if n_hits == 0:
        return {"well_index": np.empty(0, dtype=np.int64),
                "depth": np.empty(0), "normals": np.empty((0, 3)),
                "positions": np.empty((0, 3)), "n_hits": 0, "n_kept": 0,
                "n_network": int(dfn.n_fractures)}
    rng = np.random.default_rng([20261006, int(seed)])
    k = max(2, int(n_hits * OBS_FRAC))
    keep = np.sort(rng.choice(n_hits, size=min(k, n_hits), replace=False))
    return {"well_index": hit_w[keep],
            "depth": hit_z[keep],
            "normals": dfn.normals[hit_i[keep]],
            "positions": hit_p[keep],
            "n_hits": n_hits, "n_kept": int(keep.size),
            "n_network": int(dfn.n_fractures)}


def estimate_network_size(theta: "Theta", pool_seed: int, i: int) -> int:
    """生成前 N 估计 —— 镜像 generate_dfn 的自适应 r_min 环 (同数学, 独立 rng)。

    修正案 W91-0b 的模拟器定义域筛: 返回的 N 估计 > N_MAX_DOMAIN ⇒ 该 θ 落在
    模拟器有效域外 (generate_dfn 的 N_max 循环会放弃收敛), 不生成直接拒绝。
    """
    from .dfn import _sample_powerlaw
    rng = np.random.default_rng([int(pool_seed), 91, int(i)])
    r_min, r_max, beta = theta.r_min, theta.r_max, theta.beta
    V = DOMAIN[0] * DOMAIN[1] * DOMAIN[2]
    N = 200
    for _ in range(10):
        pilot = _sample_powerlaw(beta, r_min, r_max, 10000, rng)
        mean_area = np.pi * np.mean(pilot ** 2)
        N = max(200, int(round(theta.p32 * V / mean_area)))
        if N <= N_MAX_DOMAIN or r_min >= r_max * 0.5:
            break
        r_min = min(r_min * 1.5, r_max * 0.5)
    return N


def theta_to_network(theta: "Theta", seed: int) -> DFNRealization:
    """θ → DFNRealization (deterministic given seed; 禁用默认 r_min/r_max 截断)。"""
    st = SetTable(centers=theta.set_centers[:theta.K],
                  concentrations=theta.kappas[:theta.K],
                  proportions=theta.props[:theta.K])
    return generate_dfn(st, p32=theta.p32, beta=theta.beta, domain=DOMAIN,
                        seed=int(seed), r_min=theta.r_min, r_max=theta.r_max)


# ---------------------------------------------------------------------------
# θ 先验
# ---------------------------------------------------------------------------

@dataclass
class Theta:
    """腿 A 参数向量 θ = (K, {μ_k, κ_k, π_k}, P32, β, r_min, r_max)。"""
    K: int
    set_centers: np.ndarray          # (6,3) pad
    kappas: np.ndarray               # (6,) pad
    props: np.ndarray                # (6,) pad
    p32: float
    beta: float
    r_min: float
    r_max: float
    prior: str = "wide"

    def as_vector(self) -> np.ndarray:
        """定长数值向量 (K, 18 个 μ 分量 pad, 6κ, 6π, p32, β, r_min, r_max)。"""
        return np.concatenate([
            np.array([self.K, self.p32, self.beta, self.r_min, self.r_max]),
            self.set_centers.reshape(-1), self.kappas, self.props])


def sample_theta(rng: np.random.Generator, prior: str = "wide") -> Theta:
    """从预注册先验抽 θ。μ_k ~ 球面均匀, κ_k ~ LogU, π ~ Dirichlet(1)。"""
    spec = PRIORS[prior]
    K = int(rng.choice(np.asarray(spec["K_choices"])))
    mu = rng.normal(size=(6, 3))
    mu = mu / np.clip(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12, None)
    kappa = np.exp(rng.uniform(np.log(spec["kappa_loguniform"][0]),
                               np.log(spec["kappa_loguniform"][1]), size=6))
    props = rng.dirichlet(np.ones(6))
    # 前 K 组吃掉比例, 其余 pad 为 0 (SetTable 内部会再归一)
    props_k = props[:K] / props[:K].sum()
    props_full = np.zeros(6)
    props_full[:K] = props_k
    p32 = float(np.exp(rng.uniform(np.log(spec["p32_loguniform"][0]),
                                   np.log(spec["p32_loguniform"][1]))))
    beta = float(rng.uniform(*spec["beta_uniform"]))
    r_max = float(np.exp(rng.uniform(np.log(spec["r_max_loguniform"][0]),
                                     np.log(spec["r_max_loguniform"][1]))))
    for _ in range(100):
        r_min = float(np.exp(rng.uniform(np.log(spec["r_min_loguniform"][0]),
                                         np.log(spec["r_min_loguniform"][1]))))
        if r_min <= 0.8 * r_max:
            break
    else:
        r_min = 0.5 * r_max
    return Theta(K=K, set_centers=mu, kappas=kappa, props=props_full,
                 p32=p32, beta=beta, r_min=r_min, r_max=r_max, prior=prior)


def theta_to_arrays(theta: Theta) -> Dict[str, np.ndarray]:
    return {"K": np.int64(theta.K), "set_centers": theta.set_centers,
            "kappas": theta.kappas, "props": theta.props,
            "p32": np.float64(theta.p32), "beta": np.float64(theta.beta),
            "r_min": np.float64(theta.r_min), "r_max": np.float64(theta.r_max)}


# ---------------------------------------------------------------------------
# 四族摘要 (观测空间; 全族无半径 —— prereg#observation_radius_policy)
# ---------------------------------------------------------------------------

def catalog_summaries(well_index: np.ndarray, depth: np.ndarray,
                      normals: np.ndarray, n_wells: int) -> Dict[str, np.ndarray]:
    """四族摘要 (register#method.summary_families):

    计数族 = 每井命中数 (n_wells,); 产状族 = Terzaghi 校正前后 12方位×3倾角带
    直方图 (两套概率直方图); 间距族 = 沿井相邻交点间距的 q10/q25/q50/q75/q90;
    各向异性族 = 交点法向二阶矩张量特征值 (降序, 和为 1)。
    """
    n = int(well_index.size)
    counts = np.bincount(well_index, minlength=n_wells).astype(np.float64)
    if n >= 3:
        dip, dd = normal_to_dip_dipdir(normals)
        az_edges = np.linspace(0.0, 360.0, AZ_BINS + 1)
        dip_edges = np.linspace(0.0, 90.0, DIP_BINS + 1)
        h_raw, _, _ = np.histogram2d(dd, dip, bins=[az_edges, dip_edges])
        w_tz = terzaghi_weights(normals)          # 全竖直井 ⇒ 轴 (0,0,1)
        h_tz, _, _ = np.histogram2d(dd, dip, bins=[az_edges, dip_edges], weights=w_tz)
        hist_raw = (h_raw / h_raw.sum()).reshape(-1)
        hist_tz = (h_tz / h_tz.sum()).reshape(-1)
    else:
        hist_raw = np.zeros(AZ_BINS * DIP_BINS)
        hist_tz = np.zeros(AZ_BINS * DIP_BINS)
    if n >= 2:
        order = np.lexsort((depth, well_index))
        d_sorted = depth[order]
        w_sorted = well_index[order]
        gaps = d_sorted[1:] - d_sorted[:-1]
        gaps = gaps[w_sorted[1:] == w_sorted[:-1]]   # 只取同井相邻
        if gaps.size >= 1:
            spacing_q = np.quantile(gaps, SPACING_QUANTILES)
        else:
            spacing_q = np.full(len(SPACING_QUANTILES), np.nan)
    else:
        spacing_q = np.full(len(SPACING_QUANTILES), np.nan)
    if n >= 3:
        t_mat = (normals.T @ normals) / float(n)
        eig = np.linalg.eigvalsh(t_mat)[::-1]        # 降序
        eig = np.clip(eig, 0.0, None)
        eig = eig / eig.sum()
    else:
        eig = np.full(3, np.nan)
    return {"counts": counts, "hist_raw": hist_raw, "hist_tz": hist_tz,
            "spacing_q": spacing_q, "eigvals": eig, "n_obs": n}


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 1.0
    bc = float(np.sqrt(p / sp * q / sq).sum())
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def family_distances(s_obs: Dict[str, np.ndarray],
                     s_sim: Dict[str, np.ndarray]) -> np.ndarray:
    """四族距离 (d1 计数, d2 产状 Hellinger 均值, d3 间距分位相对差, d4 特征值)。"""
    if s_sim["n_obs"] < MIN_HITS_FOR_RHO or s_obs["n_obs"] < MIN_HITS_FOR_RHO:
        return np.full(4, RHO_SENTINEL)
    d1 = float(np.linalg.norm(s_sim["counts"] - s_obs["counts"])) \
        / np.sqrt(float(s_obs["n_obs"]))
    d2 = 0.5 * (_hellinger(s_sim["hist_tz"], s_obs["hist_tz"])
                + _hellinger(s_sim["hist_raw"], s_obs["hist_raw"]))
    q_obs = s_obs["spacing_q"]
    q_sim = s_sim["spacing_q"]
    scale = float(q_obs[2]) if np.isfinite(q_obs[2]) and q_obs[2] > 0 else 1.0
    if not (np.all(np.isfinite(q_sim)) and np.all(np.isfinite(q_obs))):
        return np.full(4, RHO_SENTINEL)
    d3 = float(np.linalg.norm(q_sim - q_obs)) / scale
    d4 = float(np.linalg.norm(s_sim["eigvals"] - s_obs["eigvals"]))
    if not all(np.isfinite(x) for x in (d1, d2, d3, d4)):
        return np.full(4, RHO_SENTINEL)
    return np.array([d1, d2, d3, d4])


def rho_of(d: np.ndarray, mad: np.ndarray) -> float:
    """ρ = 各族标准化后的 L2 (MAD 标准化即族权; 全部来自先验池, 不看后验)。"""
    d = np.asarray(d, dtype=np.float64)
    mad = np.asarray(mad, dtype=np.float64)
    safe = np.where(mad > 0, mad, 1.0)
    if np.any(mad <= 0):
        # 退化族 (先验池内零散) → 整族降权为 0 距离并登记; 不会发生在主臂
        pass
    z = d / safe
    return float(np.sqrt(np.sum(z * z)))


def mad_of_pool(dist_matrix: np.ndarray) -> np.ndarray:
    """先验池四族距离的 MAD (robust 尺度; 全零族回退 1.0 并由调用方登记)。"""
    med = np.median(dist_matrix, axis=0)
    mad = np.median(np.abs(dist_matrix - med), axis=0)
    mad = np.where(mad > 0, mad, 1.0)
    return mad


# ---------------------------------------------------------------------------
# ABC 池 (拒绝采样; 断点续跑由 runner 逐格落盘)
# ---------------------------------------------------------------------------

def abc_pool(n_sims: int, well_xy: np.ndarray, s_obs: Dict[str, np.ndarray],
             n_wells: int, seed: int, prior: str = "wide",
             checkpoint: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """先验池: n_sims 个 (θ, seed) → 模拟 → 四族距离。

    返回数组集: theta_vec (n,35) / sim_seed (n,) / dist (n,4) / n_hits / n_kept /
    valid (n,) —— valid=False 的行 = 模拟器定义域外 (N>N_max, 修正案 W91-0b),
    距离为哨兵, 不进 ε 分位与接受集。
    ``checkpoint`` 给定时续跑 (跳过前 m 行)。确定性: 池内第 i 个模拟的
    (θ rng, dfn seed, mask rng) 全由 (seed, i) 派生。
    """
    i0 = 0
    if checkpoint is not None and len(checkpoint.get("sim_seed", [])) > 0:
        i0 = int(len(checkpoint["sim_seed"]))
    tv = np.zeros((n_sims, 35), dtype=np.float64)
    sd = np.zeros(n_sims, dtype=np.int64)
    dm = np.full((n_sims, 4), np.nan)
    nh = np.zeros(n_sims, dtype=np.int64)
    nk = np.zeros(n_sims, dtype=np.int64)
    vd = np.zeros(n_sims, dtype=bool)
    if i0:
        tv[:i0] = checkpoint["theta_vec"][:i0]
        sd[:i0] = checkpoint["sim_seed"][:i0]
        dm[:i0] = checkpoint["dist"][:i0]
        nh[:i0] = checkpoint["n_hits"][:i0]
        nk[:i0] = checkpoint["n_kept"][:i0]
        vd[:i0] = checkpoint["valid"][:i0]
    for i in range(i0, n_sims):
        rng_theta = np.random.default_rng([seed, 17, i])
        theta = sample_theta(rng_theta, prior=prior)
        tv[i] = theta.as_vector()
        sd[i] = int(np.random.default_rng([seed, 29, i]).integers(0, 2**31 - 1))
        # 模拟器定义域筛 (修正案 W91-0b): 先验 N 估计超 N_max ⇒ 不生成直接拒绝
        if estimate_network_size(theta, seed, i) > N_MAX_DOMAIN:
            dm[i] = RHO_SENTINEL
            continue
        cat = simulate_catalog(theta, well_xy, sd[i])
        if cat["n_network"] > N_MAX_DOMAIN:
            dm[i] = RHO_SENTINEL
            continue
        vd[i] = True
        s_sim = catalog_summaries(cat["well_index"], cat["depth"],
                                  cat["normals"], n_wells)
        dm[i] = family_distances(s_obs, s_sim)
        nh[i] = cat["n_hits"]
        nk[i] = cat["n_kept"]
    return {"theta_vec": tv, "sim_seed": sd, "dist": dm, "n_hits": nh,
            "n_kept": nk, "valid": vd}


# ---------------------------------------------------------------------------
# 腿 B / 腿 C
# ---------------------------------------------------------------------------

def legB_members(theta_vec: np.ndarray, sim_seed: int, well_xy: np.ndarray,
                 s_obs: Dict[str, np.ndarray], n_wells: int, mad: np.ndarray,
                 eps: float, m_seeds: int = M_SEEDS_LEGB) -> List[Dict]:
    """腿 B 实现层: 同 θ 抽 M 个新 seed, **逐 seed 过 ε** (𝓕 成员资格)。"""
    theta = theta_from_vector(theta_vec)
    out = []
    for j in range(m_seeds):
        child_seed = int(np.random.default_rng([770031, int(sim_seed), j]).integers(0, 2**31 - 1))
        cat = simulate_catalog(theta, well_xy, child_seed)
        s_sim = catalog_summaries(cat["well_index"], cat["depth"],
                                  cat["normals"], n_wells)
        d = family_distances(s_obs, s_sim)
        r = rho_of(d, mad)
        if r <= eps:
            out.append({"leg": "B", "theta_vec": theta_vec, "network_seed": child_seed,
                        "rho": r, "dist": d.tolist(), "n_hits": int(cat["n_hits"]),
                        "n_kept": int(cat["n_kept"])})
    return out


def theta_from_vector(vec: np.ndarray) -> Theta:
    K = int(round(float(vec[0])))
    mu = vec[5:23].reshape(6, 3)
    return Theta(K=K, set_centers=mu.copy(), kappas=vec[23:29].copy(),
                 props=vec[29:35].copy(), p32=float(vec[1]), beta=float(vec[2]),
                 r_min=float(vec[3]), r_max=float(vec[4]))


def perturb_network(centers: np.ndarray, normals: np.ndarray, radii: np.ndarray,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """腿 C 结构扰动: 局部 增/删/旋转/缩放 (register#method.three_legs[2])。

    k_edit ~ {1..6}; 每步等概率四类:
      删 = 去随机一盘; 增 = 从成员自身边缘分布抽新盘 (位置域内均匀);
      旋 = 单盘法向绕随机切向轴转 N(0,10°); 缩 = 单盘半径 × exp(N(0,0.25))。
    """
    centers = centers.copy()
    normals = normals.copy()
    radii = radii.copy()
    k_edit = int(rng.integers(1, 7))
    for _ in range(k_edit):
        op = rng.integers(0, 4)
        m = centers.shape[0]
        if op == 0 and m > 1:                      # 删
            i = int(rng.integers(0, m))
            centers = np.delete(centers, i, axis=0)
            normals = np.delete(normals, i, axis=0)
            radii = np.delete(radii, i, axis=0)
        elif op == 1:                              # 增 (成员自身边缘分布)
            j = int(rng.integers(0, m)) if m else 0
            axis = normals[j] if m else np.array([0.0, 0.0, 1.0])
            new_n = axis + rng.normal(0, 0.2, size=3)
            nn = np.linalg.norm(new_n)
            new_n = new_n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
            centers = np.vstack([centers, rng.uniform(-0.5, 0.5, size=3)
                                 * np.array(DOMAIN)])
            normals = np.vstack([normals, new_n[None, :]])
            radii = np.append(radii, radii[j] if m
                              else float(np.exp(rng.uniform(np.log(1.0), np.log(10.0)))))
        elif op == 2 and m > 0:                    # 旋
            i = int(rng.integers(0, m))
            n_i = normals[i]
            helper = np.array([0.0, 0.0, 1.0]) if abs(n_i[2]) < 0.9 \
                else np.array([1.0, 0.0, 0.0])
            u = np.cross(n_i, helper)
            nu = np.linalg.norm(u)
            if nu < 1e-12:
                continue
            u = u / nu
            v = np.cross(n_i, u)
            ang = float(rng.normal(0.0, np.radians(10.0)))
            new_n = np.cos(ang) * n_i + np.sin(ang) * (np.cos(
                rng.uniform(0, 2 * np.pi)) * u + np.sin(
                rng.uniform(0, 2 * np.pi)) * v)
            nn = np.linalg.norm(new_n)
            normals[i] = new_n / nn if nn > 1e-12 else n_i
        elif op == 3 and m > 0:                    # 缩
            i = int(rng.integers(0, m))
            radii[i] = radii[i] * float(np.exp(rng.normal(0.0, 0.25)))
    return centers, normals, radii


def legC_members(theta_vec: np.ndarray, sim_seed: int, well_xy: np.ndarray,
                 s_obs: Dict[str, np.ndarray], n_wells: int, mad: np.ndarray,
                 eps: float, n_proposals: int = N_PROPOSALS_LEGC) -> List[Dict]:
    """腿 C: 从接受成员出发局部扰动, 只保留模拟目录仍在 ε 内的提案。"""
    theta = theta_from_vector(theta_vec)
    base = theta_to_network(theta, sim_seed)
    out = []
    for j in range(n_proposals):
        rng = np.random.default_rng([770097, int(sim_seed), j])
        c2, n2, r2 = perturb_network(base.centers, base.normals, base.radii, rng)
        if c2.shape[0] == 0:
            continue
        hit_w, hit_z, hit_p, hit_i = drill_wells(well_xy, c2, n2, r2)
        hit_w, hit_z, hit_p, hit_i = _merge_first_well(hit_w, hit_z, hit_p, hit_i)
        n_hits = int(hit_w.size)
        if n_hits == 0:
            continue
        mrng = np.random.default_rng([20261006, int(sim_seed) + 1000003 + j])
        k = max(2, int(n_hits * OBS_FRAC))
        keep = np.sort(mrng.choice(n_hits, size=min(k, n_hits), replace=False))
        s_sim = catalog_summaries(hit_w[keep], hit_z[keep], n2[hit_i[keep]],
                                  n_wells)
        d = family_distances(s_obs, s_sim)
        r = rho_of(d, mad)
        if r <= eps:
            out.append({"leg": "C", "theta_vec": theta_vec, "network_seed": int(sim_seed),
                        "perturb_j": j, "rho": r, "dist": d.tolist(),
                        "n_hits": n_hits, "n_kept": int(keep.size)})
    return out


# ---------------------------------------------------------------------------
# 端点对 + 集合贯通频率 (消费 W90 仪器)
# ---------------------------------------------------------------------------

def endpoint_pairs() -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """预注册端点对: 分离度 {10,20,40,80} × 方位 {垂直,水平,45°} × 2 偏移 = 24 对。"""
    rng = np.random.default_rng(ENDPOINT_SEED)
    offs = rng.uniform(-ENDPOINT_OFFSET_CUBE, ENDPOINT_OFFSET_CUBE,
                       size=(len(ENDPOINT_SEPARATIONS) * len(ENDPOINT_ORIENTATIONS)
                             * N_ENDPOINT_OFFSETS, 3))
    dirs = {"vertical": np.array([0.0, 0.0, 1.0]),
            "horizontal": np.array([1.0, 0.0, 0.0]),
            "diag45": np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)}
    A, B, meta = [], [], []
    k = 0
    for sep in ENDPOINT_SEPARATIONS:
        for oname in ENDPOINT_ORIENTATIONS:
            u = dirs[oname]
            for _ in range(N_ENDPOINT_OFFSETS):
                off = offs[k]
                k += 1
                a = np.asarray(ENDPOINT_ANCHOR) + off - 0.5 * sep * u
                b = np.asarray(ENDPOINT_ANCHOR) + off + 0.5 * sep * u
                A.append(a)
                B.append(b)
                meta.append({"sep_m": sep, "orient": oname})
    return np.asarray(A), np.asarray(B), meta


def ensemble_connection_flags(members: Sequence[Dict], well_xy_unused=None,
                              pairs_AB: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              r_attach: float = R_ATTACH_MAIN) -> np.ndarray:
    """集合成员 × 端点对 的 C(A,B) 布尔阵 (真判交核, W90 仪器)。

    members: [{theta_vec, network_seed, ...}]; 网络由 (θ, seed) 确定性重建,
    腿 C 成员 = 父成员 (θ, seed) 网络加 perturb_j 扰动 (扰动 rng 确定性重建)。
    逐成员建图一次, 48 个端点 attach 一次 (build_disk_graph + attach_points)。
    """
    from scipy.sparse.csgraph import connected_components
    from .connectivity_qoi import attach_points
    A_pts, B_pts = pairs_AB if pairs_AB is not None else endpoint_pairs()[:2]
    P_all = np.vstack([A_pts, B_pts])                    # (2P,3)
    n_pair = A_pts.shape[0]
    out = np.zeros((len(members), n_pair), dtype=bool)
    for mi, m in enumerate(members):
        theta = theta_from_vector(m["theta_vec"])
        dfn = theta_to_network(theta, int(m["network_seed"]))
        c, n, r = dfn.centers, dfn.normals, dfn.radii
        if m.get("leg") == "C":
            rng = np.random.default_rng([770097, int(m["network_seed"]),
                                         int(m["perturb_j"])])
            c, n, r = perturb_network(c, n, r, rng)
        if c.shape[0] == 0:
            continue
        G = build_disk_graph(c, n, r)
        n_comp, comp_ids = connected_components(G, directed=False)
        att = attach_points(c, n, r, P_all, r_attach)     # (2P, M)
        for k in range(n_pair):
            da = att.indices[att.indptr[k]:att.indptr[k + 1]]
            db = att.indices[att.indptr[n_pair + k]:att.indptr[n_pair + k + 1]]
            if da.size == 0 or db.size == 0:
                continue
            sa = np.unique(comp_ids[da])
            sb = np.unique(comp_ids[db])
            out[mi, k] = bool(np.intersect1d(sa, sb).size > 0)
    return out


def connection_stats(flags_col: np.ndarray, prior: str, eps: float,
                     r_attach: float) -> Dict:
    """单端点对的 p̂ 读数 + 随行限定语 (W90 移交②: p̂ 是模型条件频率, 不是概率)。"""
    flags = np.asarray(flags_col, dtype=bool)
    n = int(flags.size)
    if n == 0:
        raise EquifinalError("空集合: 拒绝空对空 p̂ (军规⑭)")
    p = float(flags.mean())
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return {
        "p_hat": p,
        "wilson_lo": float(max(0.0, center - half)),
        "wilson_hi": float(min(1.0, center + half)),
        "D_divergence": float(1.0 - abs(2.0 * p - 1.0)),
        "bernoulli_entropy_bits": float(0.0 if p in (0.0, 1.0)
                                        else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))),
        "n_members": n,
        "n_connected": int(flags.sum()),
        "qualifiers": {
            "p_hat_is_model_conditional_frequency_not_probability": True,
            "prior_label": prior,
            "eps": float(eps),
            "ess_param_is_n_members_equal_weights": int(n),
            "r_attach_m": float(r_attach),
            "subbox": "full-domain graph (W90: zero truncation)",
            "truncated_connections": 0,
            "lower_bound_note": "full-domain: C 为精确局部两点贯通 (无子盒截断)",
        },
    }


def divergence_table(flags: np.ndarray, pair_meta: List[Dict], prior: str,
                     eps: float, r_attach: float) -> List[Dict]:
    rows = []
    for k in range(flags.shape[1]):
        st = connection_stats(flags[:, k], prior, eps, r_attach)
        rows.append({**pair_meta[k], **st})
    return rows
