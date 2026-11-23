# -*- coding: utf-8 -*-
"""一键自检体系 — 八期 T63 核心资产 + 九期 T67 管道断裂探测器.

用法 (R188 起两个入口等价, 结果必须逐字段一致):
    cd <project_root> && python src/fractureflow/selfcheck.py            # 直跑, 零环境变量
    cd <project_root> && set PYTHONPATH=src && python -m fractureflow.selfcheck   # 权威口径
    追加参数:
      --with-tests   额外再跑一次 pytest 并汇总 (在 S20 之外)
      --quick        轻量模式: 跳过 S20 全量 pytest 慢检查 (只减执行集, 不改判据)
      --out <path>   报告落盘位置, 默认 results/selfcheck_report.json
                     (并行在飞时用临时目录跑守卫, 避免覆盖他线读到的现报)
    崩溃取证: 每次运行额外写 <--out>.progress.jsonl (逐检查 JSONL, check_start
                    先落盘再执行检查体); 可用 FF_SELFCHECK_PROGRESS 覆盖路径。
                    只观测, 不改判据、不改报告 schema。

无网络、无 GPU。默认含 S20 (全量 pytest, 预算 1500s, 现约 15-20 min);
--quick 约 1 min。输出 PASS/FAIL 表 + 落盘 JSON。

坑 35 史: 直跑形态 sys.path[0]=脚本目录, src/fractureflow/platform (R153 子包)
遮蔽 stdlib platform, torch import 期必崩 —— 各方跑法不一、计数口径混乱的根源,
故下方入口卫生必须早于任何三方库 import。

六组检查:
  1. 几何约定组: dip/dd↔法向往返 + NED 约定 + 法向无向 + 符号对齐
  2. 泄漏毒丸组: 几何/Terzaghi/strict 三组毒丸
  3. 口径扫描组: 废弃数字出现在非标注语境即 FAIL
  4. 资产完整性组: 关键 results/*.json 存在性 + 关键数字一致性抽查
  5. 单元回归组: B1-B9 各一条最小复现用例
  6. 管道断裂探测组 (T67): 冲突检测 / 聚类管道 / SetTable 生成 / pytest 整合
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# R188 入口卫生 (坑 35) — 必须早于任何三方库 import, 否则 torch 已按影子 platform 点火
#   直跑时 sys.path[0] = 脚本目录 = src/fractureflow, 其下的 platform/ 子包 (R153)
#   遮蔽 stdlib platform → torch/_load_dll_libraries() 里 platform.machine() 崩。
#   修法 = 直跑形态把脚本目录从 sys.path 摘掉, 只留其父目录 src/ (与 -m 形态同构)。
#   platform 子包语义零改动: 它仍以 fractureflow.platform 正常导入, 只是不再冒充顶层。
#   探针实测 (子进程, 影子在场): json/argparse/subprocess/time/traceback/typing 六个
#   stdlib 都不会 import platform, 故无需清 sys.modules (R188-T2 守卫再兜一次)。
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if __package__ in (None, ""):  # 直跑形态; -m / import 形态下 __package__ == "fractureflow"
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

# 同属入口修复: 非 UTF-8 控制台 (本机 GBK) 下 print("✓") 直接 UnicodeEncodeError,
# 崩在报告阶段 —— 与坑 35 一样让"直跑不可用"。errors=replace 只把不可编码字符降级
# 为占位符, 不改任何判据/数字; UTF-8 环境 (PYTHONUTF8=1) 行为逐字节不变。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # 已被替换为非 TextIOWrapper (如 pytest capture) 时静默跳过
        pass

import json
import argparse
import re
import subprocess
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SELF_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")

# 确保 fractureflow 包和 scripts/ 可导入
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# JSON 序列化辅助
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# 废弃数字清单 (来自 docs/评测标准与口径锁定.md §4.3)
# ---------------------------------------------------------------------------

_DEPRECATED_NUMBERS = {
    "15.82": "泄漏 oracle",
    "14.22": "泄漏 oracle p50",
    "12.39": "泄漏 oracle K12",
    "7.50": "泄漏 oracle strict=False",
    "9.83": "泄漏多井联合",
    "7.78": "FORGE modal_err (buggy)",
    "7.52": "FORGE 联合 (buggy)",
    "9.35": "FORGE K12 锚点 (buggy)",
    "43.2": "dip_only 误差 (arcsin)",
    "42": "beishan dip 中位 (arcsin, 应为 47.7)",
}


# ---------------------------------------------------------------------------
# 检查结果
# ---------------------------------------------------------------------------

class CheckResult:
    """单条检查结果."""
    def __init__(self, name: str, group: str, passed: bool, message: str = "", detail: str = ""):
        self.name = name
        self.group = group
        self.passed = bool(passed)  # 确保是 Python bool
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "passed": self.passed,
            "message": self.message,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# 检查注册
# ---------------------------------------------------------------------------

_ALL_CHECKS: List[Tuple[str, str, Callable[[], CheckResult]]] = []


def check(name: str, group: str):
    """装饰器: 注册一条检查."""
    def decorator(fn: Callable[[], CheckResult]):
        _ALL_CHECKS.append((group, name, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 1. 几何约定组
# ---------------------------------------------------------------------------

@check("dip_dipdir_to_normal 水平面", "几何约定")
def chk_horizontal_plane():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([0.0]))
    ok = abs(abs(float(n[0, 2])) - 1.0) < 1e-6
    return CheckResult("dip_dipdir_to_normal 水平面", "几何约定", ok,
                       "水平面法向竖直" if ok else f"nU={n[0,2]} (应为 ±1)")


@check("dip_dipdir_to_normal 垂直面", "几何约定")
def chk_vertical_plane():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([90.0]))
    ok = abs(float(n[0, 2])) < 1e-6 and abs(float(n[0, 1]) - 1.0) < 1e-6
    return CheckResult("dip_dipdir_to_normal 垂直面", "几何约定", ok,
                       "垂直面法向水平" if ok else f"nU={n[0,2]}, nN={n[0,1]}")


@check("dip_dipdir_to_normal 45°", "几何约定")
def chk_45_degree():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([45.0]))
    expected = np.cos(np.deg2rad(45))
    ok = abs(abs(float(n[0, 2])) - expected) < 1e-6
    return CheckResult("dip_dipdir_to_normal 45°", "几何约定", ok,
                       "|nz|=cos45" if ok else f"|nz|={abs(n[0,2])} (应为 {expected})")


@check("FORGE 16B 血统锚点", "几何约定")
def chk_lineage_anchor():
    try:
        import torch
        pt_path = os.path.join(_PROJECT_ROOT, "data/external/utah_forge_fmi/forge_fmi_2wells.pt")
        if not os.path.exists(pt_path):
            return CheckResult("FORGE 16B 血统锚点", "几何约定", True, "跳过 (pt 不存在)")
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        for net in d["nets"]:
            if net["wid"] == "16B":
                nrm = net["nrm"]
                nz = np.abs(nrm[:, 2])
                implied = np.degrees(np.arccos(np.clip(nz, 0, 1)))
                raw = net["dip"]
                diff = float(np.median(np.abs(implied - raw)))
                ok = diff < 0.01
                return CheckResult("FORGE 16B 血统锚点", "几何约定", ok,
                                   f"median |implied - raw| = {diff:.6f}°" if ok else f"diff={diff:.4f}°")
        return CheckResult("FORGE 16B 血统锚点", "几何约定", False, "16B not found")
    except Exception as e:
        return CheckResult("FORGE 16B 血统锚点", "几何约定", False, str(e))


@check("Terzaghi 权重方向", "几何约定")
def chk_terzaghi_direction():
    from fractureflow.terzaghi import terzaghi_weights
    n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w = terzaghi_weights(n, well_axis=np.array([0.0, 0.0, 1.0]))   # 竖直井是这条约定的前提, 必须点名
    ok = float(w[1]) > float(w[0])
    return CheckResult("Terzaghi 权重方向", "几何约定", ok,
                       f"垂直面 w={w[1]:.3f} > 水平面 w={w[0]:.3f}" if ok else f"方向反了")


@check("arccos 前 abs+clip", "几何约定")
def chk_arccos_clip():
    cos_vals = np.array([0.0, 0.5, 1.0, -0.0])
    clipped = np.clip(np.abs(cos_vals), 0, 1)
    ok = bool(np.all(clipped >= 0) and np.all(clipped <= 1))
    return CheckResult("arccos 前 abs+clip", "几何约定", ok, "arccos 输入安全")


# ---------------------------------------------------------------------------
# 2. 泄漏毒丸组
# ---------------------------------------------------------------------------

@check("毒丸: 几何约定 (水平面→竖直法向)", "泄漏毒丸")
def chk_poison_geometry():
    from scripts.read_forge_las import dip_dipdir_to_normal
    n = dip_dipdir_to_normal(np.array([0.0]), np.array([0.0]))
    ok = abs(float(n[0, 2])) > 0.99
    return CheckResult("毒丸: 几何约定", "泄漏毒丸", ok,
                       "几何约定正确" if ok else f"几何约定可能错误: nU={n[0,2]}")


@check("毒丸: strict 默认值", "泄漏毒丸")
def chk_poison_strict_default():
    import inspect
    from fractureflow.inference import set_aware_dirs
    sig = inspect.signature(set_aware_dirs)
    default = sig.parameters["strict"].default
    ok = default == True
    return CheckResult("毒丸: strict 默认值", "泄漏毒丸", ok,
                       f"strict 默认值 = {default}" if ok else f"strict 默认值仍为 {default} (应为 True)")


@check("毒丸: Terzaghi 方向", "泄漏毒丸")
def chk_poison_terzaghi():
    from fractureflow.terzaghi import terzaghi_weights
    n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w = terzaghi_weights(n, well_axis=np.array([0.0, 0.0, 1.0]))
    # 如果方向反了, 水平面权重会 > 垂直面
    ok = float(w[0]) < float(w[1])
    return CheckResult("毒丸: Terzaghi 方向", "泄漏毒丸", ok,
                       "Terzaghi 方向正确" if ok else "Terzaghi 方向可能反了")


@check("S23 数据构造毒丸: nrm_full 污染不影响 set_dirs (T76)", "泄漏毒丸")
def chk_s23_data_construction_poison():
    """T76 毒丸: 把 0-观测组的 nrm_full 填垃圾值, 断言 set_dirs 不变.

    构造: 40 点 K=4 组, 第 4 组全部隐伏 (0 观测). 修复前 set_dirs[3] 由
    nrm_full[sel] 构造 (泄漏隐伏真值), 污染 nrm_full 后 set_dirs 变.
    修复后 set_dirs[3] 由全局观测均值构造 (仅 nrm_obs), 污染 nrm_full 不影响.
    """
    from fractureflow.data import prepare_net

    L, K = 40, 4
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(L, 3)).astype(np.float32)
    # 4 组正交方向 + 小噪声
    base = np.eye(3, dtype=np.float32)
    nrm_list = []
    for k in range(K):
        noise = rng.normal(0, 0.05, (L // K, 3)).astype(np.float32)
        pts = base[k % 3] + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
        nrm_list.append(pts)
    nrm = np.concatenate(nrm_list, axis=0)
    set_ids = np.repeat(np.arange(K), L // K)
    # 第 4 组 (idx 30-39) 全部隐伏
    obs_mask = np.ones(L, dtype=np.float32)
    obs_mask[30:40] = 0.0

    net_base = {
        "pos": torch.as_tensor(pos),
        "nrm": torch.as_tensor(nrm),
        "len": torch.ones(L, 1),
        "lith": torch.zeros(L, dtype=torch.int64),
        "s1": torch.eye(3)[0],
        "s3": torch.eye(3)[2],
        "set_ids": torch.as_tensor(set_ids, dtype=torch.int64),
        "obs_mask": torch.as_tensor(obs_mask),
    }

    # 基线
    b1 = prepare_net(dict(net_base), 0.4, rng=np.random.default_rng(999))
    sd1 = b1["set_dirs"].clone()

    # 毒丸: 把第 4 组 (隐伏) 的 nrm_full 改成完全不同的方向
    net_poison = dict(net_base)
    nrm_poison = nrm.copy()
    nrm_poison[30:40] = np.array([0.0, 0.0, 1.0])  # 改成竖直向上 (原组方向是 z)
    net_poison["nrm"] = torch.as_tensor(nrm_poison)
    b2 = prepare_net(net_poison, 0.4, rng=np.random.default_rng(999))
    sd2 = b2["set_dirs"].clone()

    diff = (sd1 - sd2).abs().max().item()
    ok = diff < 1e-5
    detail = f"set_dirs max|diff| = {diff:.2e}"
    if ok:
        detail += " — nrm_full 污染不影响 set_dirs, T76 泄漏已修复"
    else:
        detail += " — nrm_full 污染改变了 set_dirs, 泄漏仍存在!"
    return CheckResult("S23 数据构造毒丸: nrm_full 污染不影响 set_dirs (T76)",
                       "泄漏毒丸", ok, detail)


@check("S24 Route B 无观测裂隙不产出零向量 (BUG-1)", "泄漏毒丸")
def chk_s24_routeb_unobserved_poison():
    """BUG-1 毒丸: 整条裂隙无观测点时, 其隐伏点必须走空间回退而非零向量.

    旧实现: 无观测裂隙用 nrm[sel] (盲协议下全零) 算 mode -> 零向量假 mode ->
    回退子集内观测数=0 -> l1_local 只能返回零向量 (=90° 误差)。
    """
    from fractureflow.connectivity import fracture_aware_dirs

    rng = np.random.default_rng(42)
    n1 = np.array([1.0, 0.0, 0.0])
    pos_list, nrm_list, fid_list = [], [], []
    fid = 0
    for c in ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]):
        for _ in range(6):
            ctr = np.array(c) + rng.normal(scale=2.0, size=3)
            pts = ctr + rng.normal(scale=0.05, size=(5, 3))
            noisy = n1 + rng.normal(scale=0.02, size=3)
            pos_list.append(pts)
            nrm_list.append(np.repeat(noisy[None], 5, axis=0))
            fid_list.append(np.full(5, fid))
            fid += 1
    pos = np.concatenate(pos_list)
    nrm_true = np.concatenate(nrm_list)
    fids = np.concatenate(fid_list)
    occ = rng.random(60) < 0.4
    all_fids = np.unique(fids)
    no_obs = np.setdiff1d(all_fids, fids[~occ])
    if len(no_obs) == 0:
        occ[fids == all_fids[0]] = False
        no_obs = np.array([all_fids[0]])
    nrm_blind = nrm_true * occ[:, None]
    dirs, _ = fracture_aware_dirs(pos, nrm_blind, occ, fracture_id=fids)
    hid = ~occ
    norms = np.linalg.norm(dirs[hid], axis=1)
    n_zero = int((norms < 0.5).sum())
    ok = n_zero == 0
    return CheckResult("S24 Route B 无观测裂隙不产出零向量 (BUG-1)",
                       "泄漏毒丸", ok,
                       f"{int(hid.sum())} 隐伏点全部非零" if ok
                       else f"{n_zero}/{int(hid.sum())} 隐伏点零向量 (BUG-1 复发)")


@check("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸")
def chk_s25_site_index_stable():
    """BUG-3 毒丸: e3gt_hybrid_v2._site_index 必须用跨进程稳定哈希.

    Python 内建 str hash 受 PYTHONHASHSEED 随机化 —— 换进程后站点嵌入整体错位,
    SiteGroupCalib 校正头静默失效。修复要求 zlib.crc32; 本检查同时做源码扫描
    (禁止 hash( 用于站点映射) 与行为验证 (crc32 映射符合预期).
    """
    src_path = os.path.join(_SRC_DIR, "fractureflow", "e3gt_hybrid_v2.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    if "hash(str(" in src or "abs(hash(" in src:
        return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", False,
                           "e3gt_hybrid_v2.py 仍在使用内建 hash() 做站点映射!")
    try:
        import zlib
        from fractureflow.e3gt_hybrid_v2 import E3GTHybridV2
        m = E3GTHybridV2(d_model=32, k_knn=8, n_layers=1, n_global=1)
        b = {"src": ["BS34", "16B"], "pos": torch.zeros(2, 4, 3)}
        idx = m._site_index(b).tolist()
        want = [zlib.crc32(s.encode("utf-8")) % 32 for s in ["BS34", "16B"]]
        ok = idx == want and len(set(idx)) > 0
        detail = f"_site_index={['BS34:'+str(idx[0]), '16B:'+str(idx[1])]} (crc32 一致)" \
            if ok else f"_site_index={idx}, crc32 期望={want}"
    except Exception as e:
        return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", False, str(e))
    return CheckResult("S25 站点索引跨进程稳定 (BUG-3)", "泄漏毒丸", ok, detail)


@check("S26 L4 融合按条目索引解析 (BUG-5)", "管道断裂探测")
def chk_s26_fuse_idx_resolution():
    """BUG-5 毒丸: 同类型多井时, 合并必须用匹配时的那个 entry.

    构造两口 L0 井 + 一个 L1 露头: 旧实现按 source_type 取第一个同类型表,
    会把第二口井的组号套到第一口井的 assign 上。修复后 ConflictMatch 携带
    idx_a/idx_b, 合并组贡献必须包含两井各自的 source_name。
    """
    from fractureflow.l4.evidence import EvidenceEntry, EvidenceBundle
    from fractureflow.l4.fuse import fuse_bundle

    rng = np.random.default_rng(7)

    def make_entry(name, normals):
        n = np.asarray(normals, dtype=np.float64)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        jitter = rng.normal(0, 0.03, n.shape)
        n = n + jitter
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        return EvidenceEntry(source_type="L0", source_name=name,
                             data_type="normal_obs", normals=n)

    # 两口 L0 井, 各自两组正交方向; L1 一组方向与两井第一组都近 (<20°)
    well_a = make_entry("wellA", [[1, 0, 0]] * 30 + [[0, 1, 0]] * 30)
    well_b = make_entry("wellB", [[0.99, 0.1, 0]] * 30 + [[0, 0, 1]] * 30)
    l1 = EvidenceEntry(source_type="L1", source_name="outcrop",
                       data_type="normal_obs",
                       normals=np.tile([1, 0.05, 0], (40, 1)).astype(np.float64))
    bundle = EvidenceBundle(entries=[well_a, well_b, l1], site_name="s26")
    res = fuse_bundle(bundle, K=2)
    contrib = [c for g in res.unified_set_table.groups
               for c in g.source_contributions]
    names = set(contrib)
    both_wells_merged = any("wellA" in c for c in contrib) and \
        any("wellB" in c for c in contrib)
    ok = both_wells_merged
    detail = (f"贡献源={sorted(names)} — 同类型两井均被正确引用"
              if ok else
              f"贡献源={sorted(names)} — 有井被静默吞掉/张冠李戴 (BUG-5 复发)")
    return CheckResult("S26 L4 融合按条目索引解析 (BUG-5)", "管道断裂探测", ok, detail)


@check("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)", "管道断裂探测")
def chk_s27_local_frames_equivariance():
    """BUG-2 护栏: 修正版标架必须旋转等变, legacy 分支必须保持非等变.

    双向防回归: (a) 有人把 fixed 改坏 -> 前半 FAIL;
    (b) 有人误以为 legacy 也等变 / 或把 legacy 删掉导致冻结复现断裂 -> 后半 FAIL。
    """
    try:
        from fractureflow.geometry import local_frames
        rng = np.random.default_rng(7)
        base_pts = np.array([[3., 0, 0], [-3., 0, 0], [0, 2., 0], [0, -2., 0],
                             [0, 0, 1.], [0, 0, -1.],
                             [0.7, 0.5, 0.2], [-0.6, 0.4, -0.3]])
        centers = rng.normal(size=(1, 6, 3)) * 5.0
        knn = (centers[:, :, None, :] + base_pts[None, None]
               + rng.normal(size=(1, 6, 8, 3)) * 1e-3).astype(np.float32)
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        R = torch.tensor(Q, dtype=torch.float32)
        kp1 = torch.tensor(knn)
        kp2 = kp1 @ R.T

        def max_pair_angle(v1, v2):
            rv1 = torch.einsum("ef,blfg->bleg", R, v1)
            cosv = (rv1 * v2).sum(-2).abs().clamp(0, 1)
            return float(torch.acos(cosv).max()) * 180.0 / np.pi

        f1, _ = local_frames(kp1, 8, deterministic_e2=True, legacy=False)
        f2, _ = local_frames(kp2, 8, deterministic_e2=True, legacy=False)
        ang_fixed = max_pair_angle(f1, f2)
        l1_, _ = local_frames(kp1, 8, legacy=True)   # 显式 legacy (免疫全局默认切换)
        l2_, _ = local_frames(kp2, 8, legacy=True)
        ang_legacy = max_pair_angle(l1_, l2_)
        ok = ang_fixed < 5.0 and ang_legacy > 5.0
        detail = (f"fixed={ang_fixed:.2f}° (<5), legacy={ang_legacy:.2f}° (>5)"
                  if ok else
                  f"fixed={ang_fixed:.2f}° legacy={ang_legacy:.2f}° — 违反 T82 约定")
    except Exception as e:
        return CheckResult("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)",
                           "管道断裂探测", False, str(e))
    return CheckResult("S27 local_frames 等变性: fixed 等变 / legacy 非等变 (BUG-2)",
                       "管道断裂探测", ok, detail)


@check("S29 场景指标面-轴语义毒丸 (法向≠通道方向)", "管道断裂探测")
def chk_s29_scenario_plane_axis_semantics():
    """十四期 P0 毒丸: dominant_direction 是优势面**法向**, 场景判定必须用
    面-轴夹角 (90°−法向角)。历史 bug: 三个场景函数拿法向直接套阈值, 判级全反
    (水平组系+竖直井对误判"对齐良好")。已知几何必须给出正确判级。
    """
    try:
        from fractureflow.dfn import DFNRealization
        from fractureflow.percolation import (
            egs_connectivity_metric, mine_risk_sections, disposal_escape_priority)

        rng = np.random.default_rng(7)

        def single_set_dfn(normal):
            nrm = np.asarray(normal, float)
            nrm = nrm / np.linalg.norm(nrm)
            u = np.array([1.0, 0.0, 0.0])
            v = np.cross(nrm, u)
            if np.linalg.norm(v) < 1e-6:
                u = np.array([0.0, 1.0, 0.0])
                v = np.cross(nrm, u)
            v = v / np.linalg.norm(v)
            centers, normals = [], []
            for _ in range(12):
                a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
                centers.append(a * u + b * v + rng.uniform(-0.5, 0.5) * nrm)
                jit = nrm + rng.normal(0, 0.02, 3)
                normals.append(jit / np.linalg.norm(jit))
            return DFNRealization(centers=np.array(centers),
                                  normals=np.array(normals),
                                  radii=np.full(12, 5.0),
                                  sets=np.zeros(12, dtype=int))

        hz = single_set_dfn([0, 0, 1])   # 水平组系
        st = single_set_dfn([1, 0, 0])   # 陡立组系 (面含 z)
        egs_bad = egs_connectivity_metric(hz, [0, 0, 1])['assessment']
        mine_high = mine_risk_sections(hz, [1, 0, 0])['risk_level']
        disp_high = disposal_escape_priority(st)['escape_priority']
        ok = ('较差' in egs_bad) and (mine_high == '高') and (disp_high == '高')
        detail = (f"竖直井对⊥水平面→'{egs_bad}'; 巷道轴在水平连通面内→'{mine_high}'; "
                  f"垂直方向在陡立面内→'{disp_high}'"
                  if ok else
                  f"判级反向复发: egs='{egs_bad}' mine='{mine_high}' disp='{disp_high}'")
    except Exception as e:
        return CheckResult("S29 场景指标面-轴语义毒丸 (法向≠通道方向)",
                           "管道断裂探测", False, str(e))
    return CheckResult("S29 场景指标面-轴语义毒丸 (法向≠通道方向)",
                       "管道断裂探测", ok, detail)


@check("S30 geo_prior 批量一致性 [known-fail: H1]", "管道断裂探测")
def chk_s30_geo_prior_batch_consistency():
    """H1 known-fail 门禁 (架构师裁决 2026-08): 测机制是否仍如档案所述, 不许假绿。

    事实档案 (scripts/diag_s30_batch_mech.py 隔离诊断):
      - 根因: _kmeans_dirs_batched 共享 default_rng(seed) 按批内顺序消费,
        第 b 个网的 k-means 初始化依赖其批内位置 (numpy 参考路径每网 fresh seed);
      - 实测: 位置 0 网与 B=1 一致 (~0.03°); 位置 >=1 网在组系清晰合成数据上
        最大逐点角差 41.2° (Ks=(3..6) 过聚类 -> 多个局部最优 -> init 敏感);
      - 影响面: 仅训练期 gp 先验噪声 + 批量评测路径; 诚实榜 8 方法全 numpy
        路径, 主锚点 36.687/31.7/12.37/0.37 全部不受影响;
      - 处置: 计算冻结不动 (修了会断 e3gt 系 checkpoint 复现), 修复推迟至
        重训时代; 对外表述仅限 B=1。
    门禁语义:
      - 位置 0 网偏离 >=0.1° -> FAIL (根因机制变了, 停手上报);
      - 其余网偏离 <0.1°     -> PASS "H1 已消失" (有人修了 -> 必须更新文档+重评);
      - 其余网偏离 <50°      -> PASS known-fail (仍在档案包络内);
      - 否则                 -> FAIL (行为漂移超出档案)。
    """
    name = "S30 geo_prior 批量一致性 [known-fail: H1]"
    try:
        import torch
        from fractureflow.geo_prior import geo_prior_dirs_gpu

        def make_net(seed, n=40):
            r = np.random.default_rng(seed)
            pos = r.normal(size=(n, 3))
            nrm = np.vstack([np.tile([1.0, 0, 0], (n // 2, 1)),
                             np.tile([0.0, 1, 0], (n - n // 2, 1))])
            nrm = nrm + r.normal(0, 0.05, nrm.shape)
            nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
            mask = r.random(n) < 0.5
            return (pos.astype(np.float64), (nrm * mask[:, None]).astype(np.float64),
                    mask)

        nets = [make_net(s) for s in (101, 202, 303, 404)]
        outs_b1 = []
        for pos, nrm, mask in nets:
            o = geo_prior_dirs_gpu(torch.tensor(pos)[None], torch.tensor(nrm)[None],
                                   torch.tensor(mask, dtype=torch.float64)[None],
                                   fast=False)
            outs_b1.append(o[0].numpy())
        pos4 = torch.tensor(np.stack([x[0] for x in nets]))
        nrm4 = torch.tensor(np.stack([x[1] for x in nets]))
        mask4 = torch.tensor(np.stack([x[2] for x in nets]), dtype=torch.float64)
        out_b4 = geo_prior_dirs_gpu(pos4, nrm4, mask4, fast=False).numpy()

        devs = []
        for b, ((_, _, mask), o1) in enumerate(zip(nets, outs_b1)):
            hid = ~mask
            c = np.clip(np.abs((o1[hid] * out_b4[b][hid]).sum(-1)), 0, 1)
            devs.append(float(np.degrees(np.arccos(c)).max()))
        pos0_dev, others_worst = devs[0], max(devs[1:])

        if pos0_dev >= 0.1:
            ok = False
            detail = (f"机制变化: 位置0网偏离 B=1 达 {pos0_dev:.3f}° "
                      f"(应≈0) — 共享init诊断失效, 停手上报")
        elif others_worst < 0.1:
            ok = True
            detail = ("H1 已消失 (批量=B=1, worst<0.1°) — 有人修复了 geo_prior; "
                      "必须同步更新口径锁定文档并重评 prior 系模型数字")
        elif others_worst < 50.0:
            ok = True
            detail = (f"known-fail (H1, 按档案维持): 位置依赖偏差 worst="
                      f"{others_worst:.1f}° <50° 包络; 计算冻结, 修复推迟至重训时代; "
                      f"对外表述仅限 B=1")
        else:
            ok = False
            detail = f"H1 偏差超档案包络: {others_worst:.1f}° >=50° — geo_prior 行为漂移"
    except Exception as e:
        return CheckResult(name, "管道断裂探测", False, str(e))
    return CheckResult(name, "管道断裂探测", ok, detail)


@check("S31 json.dump allow_nan 守卫扫描 (缺守卫/错位)", "管道断裂探测")
def chk_s31_json_nan_guard():
    """L5 规则 v2 (TO-W15-05, 2026-10-28): 不看子串, 只看 allow_nan 的值.

    背景: decovalex_routeB.json 曾静默写入 NaN; 批量补丁曾把 kwarg 错插进外层
    print (f-string AST 位置陷阱)。v1 用 'allow_nan' in source_segment 判守卫,
    从不校验值 => allow_nan=True / 变量传参 / 内层嵌套守卫全被算作"已守卫"。
    v2: 只有字面 False 才算守卫; 接收者识别 json / import json as alias /
    属性链以 .json 结尾 / __import__('json'); 逐站豁免表只收已批准的故意项,
    缺失或过期不产生豁免 (fail-closed)。
    """
    import ast as _ast
    import datetime as _dt

    _EXEMPTIONS_PATH = os.path.join(
        _RESULTS_DIR, "obs_equiv_20261024", "w15_05_s31_criterion_v2_20261028",
        "s31_exemptions.json")

    def _expr_repr(e):
        if isinstance(e, _ast.Name):
            return e.id
        if isinstance(e, _ast.Attribute):
            return _expr_repr(e.value) + "." + e.attr
        if isinstance(e, _ast.Call):
            _f = e.func
            if (isinstance(_f, _ast.Name) and _f.id == "__import__"
                    and len(e.args) == 1 and isinstance(e.args[0], _ast.Constant)
                    and e.args[0].value == "json"):
                return "__import__('json')"
        return "<expr>"

    def _aliases(tree):
        out = set()
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Import):
                for a in n.names:
                    if a.name == "json" and a.asname:
                        out.add(a.asname)
        return out

    def _is_json_recv(recv, aliases):
        if isinstance(recv, _ast.Name):
            return recv.id == "json" or recv.id in aliases
        if isinstance(recv, _ast.Attribute):
            return recv.attr == "json"
        if isinstance(recv, _ast.Call):
            f = recv.func
            return (isinstance(f, _ast.Name) and f.id == "__import__"
                    and len(recv.args) == 1
                    and isinstance(recv.args[0], _ast.Constant)
                    and recv.args[0].value == "json")
        return False

    def _an_kw(node):
        for k in node.keywords:
            if k.arg == "allow_nan":
                return k
        return None

    def _an_class(kw):
        if kw is None:
            return "missing"
        v = kw.value
        if isinstance(v, _ast.Constant) and v.value is False:
            return "guarded"
        if isinstance(v, _ast.Constant) and v.value is True:
            return "true"
        return "non_literal"

    counts = {"total_json_dump_calls": 0, "guarded_literal_false": 0,
              "allow_nan_true": 0, "non_literal": 0, "missing": 0,
              "misplaced_candidates": 0, "exempt": 0, "exempt_expired": 0,
              "exempt_unmatched": 0, "unguarded_after_exemptions": 0,
              "scanned_py": 0, "parse_failures": 0}
    ex_declared, ex_errors, ex_used = 0, [], set()
    exemptions = {}
    ex_present = os.path.isfile(_EXEMPTIONS_PATH)
    if ex_present:
        try:
            with open(_EXEMPTIONS_PATH, "r", encoding="utf-8") as _fh:
                _doc = json.load(_fh)
            for i, e in enumerate(_doc.get("exemptions") or []):
                ex_declared += 1
                try:
                    exemptions[(e["file"], int(e["line"]), e["callee"])] = \
                        _dt.date.fromisoformat(str(e["expires_on"]))
                except Exception as exc:
                    ex_errors.append(f"#{i} {type(exc).__name__}")
        except Exception as exc:
            ex_errors.append(f"table {type(exc).__name__}: {exc}")
    _today = _dt.date.today()

    bad_g, bad_p, bad_s = [], [], []

    def audit(path):
        rel = os.path.relpath(path, _PROJECT_ROOT).replace("\\", "/")
        txt = open(path, "rb").read().decode("utf-8-sig")
        tree = _ast.parse(txt)
        aliases = _aliases(tree)
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            f = node.func
            kw = _an_kw(node)
            is_dd = isinstance(f, _ast.Attribute) and f.attr in ("dump", "dumps")
            if is_dd and _is_json_recv(f.value, aliases):
                counts["total_json_dump_calls"] += 1
                cls = _an_class(kw)
                if cls == "guarded":
                    counts["guarded_literal_false"] += 1
                    continue
                counts[{"true": "allow_nan_true", "non_literal": "non_literal",
                        "missing": "missing"}[cls]] += 1
                key = (rel, node.lineno, _expr_repr(f.value) + "." + f.attr)
                exp = exemptions.get(key)
                if exp is not None:
                    ex_used.add(key)
                    if _today <= exp:
                        counts["exempt"] += 1
                        continue
                    counts["exempt_expired"] += 1
                bad_g.append(f"{rel}:{node.lineno}")
            elif kw is not None:
                counts["misplaced_candidates"] += 1
                bad_p.append(f"{rel}:{node.lineno} ({_expr_repr(f)})")

    for d in (_SCRIPTS_DIR, _SRC_DIR):
        for dirpath, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                counts["scanned_py"] += 1
                p = os.path.join(dirpath, fn)
                try:
                    audit(p)
                except Exception as e:
                    counts["parse_failures"] += 1
                    bad_s.append(f"{p}: {type(e).__name__} {e}")

    counts["exempt_unmatched"] = len(set(exemptions) - ex_used)
    counts["unguarded_after_exemptions"] = (
        counts["missing"] + counts["allow_nan_true"] + counts["non_literal"]
        - counts["exempt"])
    measured = _dt.datetime.now().astimezone().isoformat()
    ok = (counts["unguarded_after_exemptions"] == 0
          and counts["misplaced_candidates"] == 0
          and counts["parse_failures"] == 0)
    detail = (
        "S31v2 口径=字面 allow_nan=False 才算守卫 "
        "枚举=AST 逐 Call 取关键字值 "
        "接收者=json|import json as alias|属性链以 .json 结尾|__import__('json') "
        f"面=scripts+src {counts['scanned_py']} .py 测量={measured} "
        f"豁免表={'在盘' if ex_present else '缺失(零豁免)'}"
        f"(declared={ex_declared} active={len(exemptions)} errors={len(ex_errors)}) "
        "counts_json=" + json.dumps(counts, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False))
    if not ok:
        detail += (f" 抽样 missing={bad_g[:3]} misplaced={bad_p[:3]}"
                   + (f" parse_fail={bad_s[:2]}" if bad_s else ""))
    return CheckResult("S31 json.dump allow_nan 守卫扫描 (缺守卫/错位)",
                       "管道断裂探测", ok, detail)


@check("S32 产状↔法向往返一致 (全库变换点扫描)", "管道断裂探测")
def chk_s32_dip_normal_roundtrip():
    """L5 铁律固化 (2026-08-28): dip↔法向双向变换必须往返一致.

    历史事故: 八期 T58 只修了正向 (read_forge_las.dip_dipdir_to_normal),
    同类逆向变换 (_nd / report_replay) 的 sin/cos 互换漏网 —— dip 被镜像成
    90°−dip, 锚点档/复现框架整条链建立在镜像法向上。本检查锁死现存全部
    变换点与权威口径的一致性; 新增变换点若不一致会在此 FAIL。
    """
    NAME = "S32 产状↔法向往返一致 (全库变换点扫描)"
    try:
        import importlib.util
        import math

        def _load(name):
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(_SCRIPTS_DIR, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        rng = np.random.default_rng(11)
        dips = rng.uniform(1.0, 89.0, size=24)   # 避开 0/90 退化端点
        dds = rng.uniform(0.0, 360.0, size=24)

        bad = []
        # 1) read_forge_las 权威正变换: nz == cos(dip), 单位化
        rfl = _load("read_forge_las")
        n_ref = np.asarray(rfl.dip_dipdir_to_normal(dds, dips), float)
        if not np.allclose(np.linalg.norm(n_ref, axis=-1), 1.0, atol=1e-6):
            bad.append("read_forge_las.dip_dipdir_to_normal 非单位")
        if float(np.abs(n_ref[:, 2] - np.cos(np.radians(dips))).max()) > 1e-6:
            bad.append("read_forge_las 正变换 nz != cos(dip)")
        # 2) forge_fmi_pipeline._nd 必须同口径
        ffp = _load("forge_fmi_pipeline")
        for dd, dp in zip(dds, dips):
            n1 = np.asarray(ffp._nd(dd, dp), float).ravel()
            if abs(float(n1[2]) - math.cos(math.radians(dp))) > 1e-6:
                bad.append(f"forge_fmi_pipeline._nd nz={n1[2]:.4f} != cos(dip={dp:.1f}°)")
                break
        # 3) report_replay 逆变换: normal -> 提取 dip 应闭合
        rr = _load("report_replay")
        n_in = np.asarray(rr._dip_dipdir_to_normal(dds, dips), float)
        dip_rt = np.degrees(np.arccos(np.clip(np.abs(n_in[:, 2]), 0, 1)))
        worst = float(np.abs(dip_rt - dips).max())
        if worst > 1e-4:
            bad.append(f"report_replay 往返 dip 最大偏差 {worst:.3f}°")

        ok = not bad
        detail = "全部产状↔法向变换点与权威口径一致" if ok else "; ".join(bad[:4])
    except Exception as e:
        return CheckResult(NAME, "管道断裂探测", False, str(e))
    return CheckResult(NAME, "管道断裂探测", ok, detail)


# ---------------------------------------------------------------------------
# 3. 口径扫描组
# ---------------------------------------------------------------------------

_DEPRECATE_QUOTE_FIELDS = ("text", "token", "reason", "snippet", "pattern",
                     "line_text", "match", "needle", "query")

def scan_report_quoted_body(content):
    """语境判定 (R278-T4 / 裁定 78 D-4): 门禁扫描报告件把"被扫描目标"原文写进
    hits[*].text / token / reason 等引用字段, 那是**引用**不是**断言** ——
    R274 的 gates/scan_report.json 正是引用了本检查自己的判废模式串, 从而把主树扫描打红。

    只认结构自证 (顶层 tool + hits[] 且条目带 token 与 status), **不按文件名豁免**;
    抠掉引用字段后的剩余部分仍照原判据扫, 覆盖面不缩。
    返回 None = 非扫描报告形态 ⇒ 整件按原规则判。
    """
    try:
        d = json.loads(content)
    except Exception:
        return None
    if not (isinstance(d, dict) and "tool" in d and isinstance(d.get("hits"), list)):
        return None
    hits = d["hits"]
    shaped = [h for h in hits if isinstance(h, dict) and "token" in h and "status" in h]
    if not shaped or len(shaped) * 2 < len(hits):
        return None
    stripped = []
    for h in hits:
        if isinstance(h, dict):
            stripped.append({k: v for k, v in h.items() if k not in _DEPRECATE_QUOTE_FIELDS})
        else:
            stripped.append(h)
    rest = {k: v for k, v in d.items() if k != "hits"}
    rest["hits"] = stripped
    try:
        return json.dumps(rest, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PF00-CLOSE 环境适配 (2026-11-06): 全仓扫描的 I/O / 内存策略
#   判据、模式串、排除清单、注册检面数、_QUICK_SKIP 一律未改。
#   原实现对 results/ 下每个 .json 做全文 UTF-8 读入并无条件 json.loads；
#   本工作区实测 85,017 件 / 4,777,525,005 B (最大单件 110 MiB)，解析期内存峰值
#   使 run_all_checks 中途卡死或进程异常终止，而 save_report 只在全部检查返回后
#   执行，故历史上出现 completed=false / return_code=null / 0xC0000409 且无报告。
#   现改为：分块二进制读完全部字节 (覆盖面不缩)，仅当字节中可能出现模式串或
#   JSON 的 \u 转义时才回退到原完整判定；其余件可证明等价地短路。
#   等价性：body=scan_report_quoted_body(content) 非 None 时其内容来自 json.loads
#   后再 dumps；能"凭空"引入模式串中文/度数字符的唯一途径是 \u 转义。若原文字节
#   既不含任何模式串也不含 \u，则 content 与 body 都不可能含模式串，该件不可能被
#   记为违规，短路不改变 PASS/FAIL。
# ---------------------------------------------------------------------------
_DEPRECATE_PATTERN_BYTES = tuple(p.encode("utf-8") for p in (
    "中位 42°", "中位 42.0°", "误差 43.2°", "误差 43.20°"))
_DEPRECATE_ESCAPE_MARKER = b"\\u"


_MAX_PATTERN_CHARS = max(len(p.decode("utf-8")) for p in _DEPRECATE_PATTERN_BYTES)
# 一条 Unicode 转义占 6 字节；最坏情况整条模式串由转义构成，跨块边界须留足重叠
_PRESCREEN_OVERLAP = 6 * _MAX_PATTERN_CHARS + max(len(p) for p in _DEPRECATE_PATTERN_BYTES)
_UNICODE_ESCAPE_BYTES_RE = re.compile(rb"\\u([0-9a-fA-F]{4})")


def _fold_unicode_escapes(buf):
    """把 JSON 的 Unicode 转义折叠成对应 UTF-8 字节（保守折叠，不判转义奇偶）。

    保守折叠只会让预筛"不敢短路"（多进全量判定），不会漏报；孤立代理或无法
    编码时原样保留。真正的违规判定仍在 chk_deprecated_numbers 逐字执行。
    """
    def _repl(m):
        try:
            return chr(int(m.group(1), 16)).encode("utf-8")
        except (ValueError, UnicodeEncodeError):
            return m.group(0)
    return _UNICODE_ESCAPE_BYTES_RE.sub(_repl, buf)


def _prescreen_chunk_may_hit(buf):
    """片段（折叠 Unicode 转义后）是否可能出现任一废弃模式串。"""
    if _DEPRECATE_ESCAPE_MARKER in buf:
        buf = _fold_unicode_escapes(buf)
    return any(p in buf for p in _DEPRECATE_PATTERN_BYTES)


# ---------------------------------------------------------------------------
# G2-DEPRECATED-SCAN-CRASH-CLOSE 修复 (2026-11-07)
#   实测见 results/product_first_plan_20261106/runs/G2_DEPRECATED_SCAN_CRASH_CLOSE/
#   manifest_summary.json：检面 85,698 件 / 4.54 GiB。原 prescreen 只要字节里出现
#   转义标记就要求全量判定，把 422 个与模式串无关的文件推入"整文件 UTF-8 读入 +
#   json.loads + json.dumps"路径（439 件 / 99 MiB）；逐阶段诊断实测其中仅 1 件为
#   扫描报告形态、0 件命中、最大单件 40,257,394 字符。叠加 torch 基线（实测
#   442.9 MiB RSS）与逐块 buf=tail+chunk 的 1 MiB 拷贝（约 40 万次），构成无界内存
#   与灾难性解析的结构性成因，与 0xC0000409 原生异常、无 traceback、无报告一致。
#   修复：① 转义标记粗筛改为"折叠 Unicode 转义后是否可能出现模式串"的精确等价
#   短路；② 边界只拼 overlap 小段，不再整块拷贝；③ 判据本体、四个模式、排除
#   清单、scan_report_quoted_body、results 扫描面一字未动。
#   等价性：judged = content（非报告形态）或 json.dumps(rest, ensure_ascii=False)
#   （报告形态）。rest 是 json.loads 结果的子集，dumps 输出的中文/度数字符只能来自
#   字面 UTF-8 字节或 Unicode 转义解码；四个模式串都不含 JSON 结构字符与转义字符，
#   故模式串若出现在 judged，必作为连续子串出现在把 content 的 Unicode 转义折叠后
#   的文本里。逆否：折叠后字节流无模式串 ⇒ judged 无模式串 ⇒ 该件不可能被记为
#   违规 ⇒ 短路不改变 PASS/FAIL。
# ---------------------------------------------------------------------------
def _deprecated_prescreen(path):
    """分块读完整个文件，返回 (needs_full_judgement, bytes_read)。

    仅在折叠 Unicode 转义后可能出现模式串时才要求全量判定；overlap 覆盖跨块边界
    （最坏：整条模式串由转义构成）。不做整文件 decode / json.loads，内存恒定。
    """
    tail = b""
    total = 0
    needs_full = False
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if not needs_full:
                edge = tail + chunk[:_PRESCREEN_OVERLAP]
                if _prescreen_chunk_may_hit(chunk) or _prescreen_chunk_may_hit(edge):
                    needs_full = True
            tail = chunk[-_PRESCREEN_OVERLAP:]
    return needs_full, total


@check("废弃数字扫描", "口径扫描")
def chk_deprecated_numbers():
    """扫描 results 目录, 确认废弃数字只出现在标注语境."""
    # 更精确的匹配: 只标记特定模式的废弃数字
    deprecated_patterns = {
        "中位 42°": "beishan dip 中位 (应为 47.7°)",
        "中位 42.0°": "beishan dip 中位 (应为 47.7°)",
        "误差 43.2°": "dip_only 误差 (arcsin bug)",
        "误差 43.20°": "dip_only 误差 (arcsin bug)",
    }
    # 工具自生成报告: 内含源码片段/模式定义字符串 (如 selfcheck.py 自身的
    # 废弃数字模式串), 会被自身扫描误报, 必须排除 (避免自引用死循环)。
    _TOOL_REPORT_JSONS = {
        "selfcheck_report.json",
        "danger_pattern_scan.json",
        "mutation_test_report.json",
        "repro_certify.json",
        "orphan_json_disposition.json",
    }
    found_deprecated = []
    scanned = 0
    scanned_bytes = 0
    short_circuited = 0
    full_judged = 0
    unreadable = 0
    for root, dirs, files in os.walk(_RESULTS_DIR):
        for f in files:
            if f.endswith(".json"):
                if f in _TOOL_REPORT_JSONS:
                    continue
                path = os.path.join(root, f)
                try:
                    needs_full, nbytes = _deprecated_prescreen(path)
                except Exception:
                    unreadable += 1
                    continue
                scanned += 1
                scanned_bytes += nbytes
                if not needs_full:
                    short_circuited += 1
                    continue
                full_judged += 1
                # 可能命中的件：原判据逐字保留，不做任何降级
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    body = scan_report_quoted_body(content)
                    judged = content if body is None else body
                    for pattern, reason in deprecated_patterns.items():
                        if pattern in judged:
                            if "deprecated" not in content.lower() and "废弃" not in content:
                                found_deprecated.append(f"{f}: {pattern}")
                except Exception:
                    pass
    detail = (f"scanned_json_files={scanned}, scanned_bytes={scanned_bytes}, "
              f"full_judged={full_judged}, equivalent_short_circuit={short_circuited}, "
              f"unreadable={unreadable}, prescreen_policy=unicode_escape_fold, "
              f"quick_skip_expanded=false, criteria_unchanged=true")
    if scanned == 0:
        return CheckResult("废弃数字扫描", "口径扫描", False,
                           "空检面: results/ 下无可扫描 .json, 按纪律不记 PASS", detail)
    ok = len(found_deprecated) == 0
    return CheckResult("废弃数字扫描", "口径扫描", ok,
                       "未发现未标注的废弃数字" if ok else f"发现: {found_deprecated[:5]}",
                       detail)

@check("口径一致性抽查", "口径扫描")
def chk_caliber_consistency():
    """抽查关键数字与口径表一致性."""
    l1_path = os.path.join(_RESULTS_DIR, "honest_leaderboard/l1_local__beishan_22.json")
    if not os.path.exists(l1_path):
        return CheckResult("口径一致性抽查", "口径扫描", True, "跳过 (文件不存在)")
    try:
        with open(l1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mae = float(data.get("mae_mean", data.get("mae", 0)))
        ok = 35 < mae < 39
        return CheckResult("口径一致性抽查", "口径扫描", ok,
                           f"l1_local MAE = {mae:.3f}°" if ok else f"l1_local MAE = {mae:.3f}° (异常)")
    except Exception as e:
        return CheckResult("口径一致性抽查", "口径扫描", False, str(e))


@check("occ 掩码语义反向扫描", "口径扫描")
def chk_occ_mask_semantics():
    """L5 铁律 (外部审查雷2, 2026-08-29): assign 掩码语义方向.

    约定: occ 为 True 表示"已观测"; assign>=0 表示入组, -1 表示未观测.
    故 'assign[~occ] = 0' 会把未观测点标成入组、已观测点留 -1, 语义反转 (休眠 bug).
    全库扫描 src/fractureflow/*.py, 禁止出现该反向写法.
    """
    import re
    hits = []
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "fractureflow")
    for root, _, files in os.walk(scan_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    for ln, line in enumerate(fh.readlines(), 1):
                        if re.search(r"assign\[\s*~occ\s*\]\s*=\s*0", line):
                            hits.append(f"{fn}:{ln}")
            except Exception:
                pass
    ok = len(hits) == 0
    return CheckResult("occ 掩码语义反向扫描", "口径扫描", ok,
                       "未检测到 assign[~occ] 反向写法" if ok else f"发现反向掩码: {hits}")


@check("S28 图账同源: 论文图表脚本零手抄数字", "口径扫描")
def chk_s28_figure_script_numbers():
    """T88/G8 制度化: gen_paper_figures.py 禁止出现未申报的浮点字面量。

    背景: Fig6 曾用 rng.uniform 模拟数据冒充实测、Fig2/Fig3 手抄受染数字
    (9.35 / 7.78)。根治规则: 所有展示用精度数字必须从 results/*.json 直读,
    布局常数 (figsize/alpha/线宽) 必须在脚本顶部 ALLOWED_LAYOUT_FLOATS 申报。
    本检查用 tokenize 提取源码中的浮点 NUMBER token (字符串/注释不扫),
    任何不在白名单内的浮点字面量 = 有人往图里手抄了数字 = FAIL。
    """
    import ast
    import io as _io
    import tokenize as _tokenize

    fig_path = os.path.join(_SCRIPTS_DIR, "gen_paper_figures.py")
    if not os.path.exists(fig_path):
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, "scripts/gen_paper_figures.py 不存在")
    try:
        with open(fig_path, encoding="utf-8") as f:
            src = f.read()
        if "S28-AUDIT" not in src:
            return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                               "口径扫描", False, "缺少 S28-AUDIT 标记 (白名单申报块被移除?)")
        # 从 AST 中提取 ALLOWED_LAYOUT_FLOATS 白名单 (字符串集合)
        allowed = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", None) == "ALLOWED_LAYOUT_FLOATS":
                        allowed = set(ast.literal_eval(node.value))
        if allowed is None:
            return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                               "口径扫描", False, "未找到 ALLOWED_LAYOUT_FLOATS 申报")
        # tokenize 扫描代码区浮点字面量 (STRING/COMMENT 天然排除)
        found = set()
        for tok in _tokenize.generate_tokens(_io.StringIO(src).readline):
            if tok.type == _tokenize.NUMBER and "." in tok.string:
                found.add(tok.string)
        bad = sorted(found - allowed)
        unused = sorted(allowed - found)
        ok = len(bad) == 0
        detail = f"代码区浮点字面量 {len(found)} 个全部在白名单内"
        if unused:
            detail += f"; 白名单未命中(仅提示): {unused[:5]}"
        if not ok:
            detail = f"发现未申报的手抄浮点数: {bad} —— 图表必须 JSON 直读!"
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", ok, detail)
    except SyntaxError as e:
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, f"语法解析失败: {e}")
    except Exception as e:
        return CheckResult("S28 图账同源: 论文图表脚本零手抄数字",
                           "口径扫描", False, f"异常: {e}")


# ---------------------------------------------------------------------------
# 4. 资产完整性组
# ---------------------------------------------------------------------------

_KEY_RESULTS = [
    "honest_leaderboard/l1_local__beishan_22.json",
    "set_table_eval.json",
    "p0_full_audit.json",
    "dip_only_upper_bound.json",
    "t58_b2b3_forge_rebuild.json",
    "t51_type_isolation_revalidation.json",
]

_KEY_TESTS = [
    "tests/test_geometry_conventions.py",
    "tests/test_terzaghi.py",
]


@check("关键 results 存在性", "资产完整性")
def chk_key_results_exist():
    missing = []
    for rel_path in _KEY_RESULTS:
        full_path = os.path.join(_RESULTS_DIR, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
    ok = len(missing) == 0
    return CheckResult("关键 results 存在性", "资产完整性", ok,
                       f"全部存在 ({len(_KEY_RESULTS)} 个)" if ok else f"缺失: {missing}")


@check("单测文件存在性", "资产完整性")
def chk_tests_exist():
    missing = []
    for t in _KEY_TESTS:
        if not os.path.exists(os.path.join(_PROJECT_ROOT, t)):
            missing.append(t)
    ok = len(missing) == 0
    return CheckResult("单测文件存在性", "资产完整性", ok,
                       "全部存在" if ok else f"缺失: {missing}")


# ---------------------------------------------------------------------------
# 5. 单元回归组
# ---------------------------------------------------------------------------

@check("回归: B1 type_aware 双键区分", "单元回归")
def chk_b1_regression():
    fuse_path = os.path.join(_SRC_DIR, "fractureflow/l4/fuse.py")
    if not os.path.exists(fuse_path):
        return CheckResult("回归: B1 type_aware 双键区分", "单元回归", True, "跳过")
    with open(fuse_path, "r", encoding="utf-8") as f:
        content = f.read()
    ok = "source_name" in content
    return CheckResult("回归: B1 type_aware 双键区分", "单元回归", ok,
                       "双键区分已实施" if ok else "未找到双键区分")


@check("回归: B5 空簇跳过", "单元回归")
def chk_b5_regression():
    fuse_path = os.path.join(_SRC_DIR, "fractureflow/l4/fuse.py")
    if not os.path.exists(fuse_path):
        return CheckResult("回归: B5 空簇跳过", "单元回归", True, "跳过")
    with open(fuse_path, "r", encoding="utf-8") as f:
        content = f.read()
    ok = "mask.sum() == 0" in content or "mask.sum()<1" in content
    return CheckResult("回归: B5 空簇跳过", "单元回归", ok,
                       "空簇跳过已实施" if ok else "未找到空簇跳过逻辑")


@check("回归: B7 贯通判定用 domain 边界 (功能)", "单元回归")
def chk_b7_regression():
    """功能级回归: 验证 _find_spanning_cluster 用真实 domain 边界而非数据极值。

    构造: domain=L=100, 一连通分量含 x=-49 (近 -L/2 壁) 与 x=+5 (近中心, 不近
    +L/2 壁)。数据极值=[-49,5], 但若按数据极值判定会误判"贯穿"; 按真实 domain
    判定应返回 False (未贯穿 +L/2 壁)。旧实现 (数据极值) 会错误返回 True。
    """
    try:
        import scipy.sparse as sp
        from fractureflow.percolation import _find_spanning_cluster
        centers = np.array([[-49.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        adj = sp.csr_matrix([[0, 1], [1, 0]])
        has_span, _ = _find_spanning_cluster(adj, centers, (100.0, 100.0, 100.0), axis=0)
        ok = (has_span is False)
        detail = ("domain 边界判定生效 (稀疏分量不误判贯穿)"
                  if ok else "仍按数据极值判定: 误判贯穿 (B7 复发!)")
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"导入/执行失败: {e}"
    return CheckResult("回归: B7 贯通判定用 domain 边界 (功能)", "单元回归", ok, detail)


# ---------------------------------------------------------------------------
# 6. 管道断裂探测组 (T67)
# ---------------------------------------------------------------------------


def _angular_mae_deg(pred, true):
    """项目主指标: mean acos|<pred,true>| (度). 复制自评测口径, 供锚点自检."""
    cos = np.clip(np.abs(np.sum(pred * true, axis=1, dtype=float)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _rand_unit(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _rand_rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


# 数学真实锚点: 两个独立均匀随机单位向量夹角余弦 |cos θ| ~ Uniform[0,1],
# 故 E[acos(|cos θ|)] = ∫₀¹ arccos(u) du = 1.0 rad = 57.2958° (不依赖任何数据).
# 任何管线喂全随机输入却给出显著偏离此值的结果 = 链路 bug。
_RANDOM_ANCHOR_DEG = 57.29577951308232


@check("S22 随机锚点 + 不变性 (数学真相锚)", "单元回归")
def chk_s22_random_anchor():
    """验证主指标在已知数学锚点上正确 + 满足旋转/置换不变性。

    - 全随机 pred vs 全随机 true → 误差必须 ≈ 57.2958°
    - pred==true → 0
    - 同旋转 R 作用于 pred 与 true → 误差不变 (旋转等变)
    - 打乱点序 → 误差不变 (置换不变)
    """
    ok = True
    parts = []

    # 1) 随机锚点
    pred = _rand_unit(50000, 1)
    true = _rand_unit(50000, 2)
    e_rand = _angular_mae_deg(pred, true)
    d_rand = abs(e_rand - _RANDOM_ANCHOR_DEG)
    if d_rand > 1.5:
        ok = False
        parts.append(f"随机锚点={e_rand:.3f}° (期望57.296°, 偏差{d_rand:.3f}°>")
    else:
        parts.append(f"随机锚点={e_rand:.3f}°")

    # 2) 恒等
    e_id = _angular_mae_deg(true, true)
    if e_id > 1e-6:
        ok = False
        parts.append(f"恒等误差={e_id:.2e}°")
    else:
        parts.append("恒等=0")

    # 3) 旋转等变
    R = _rand_rot(7)
    e_rot = _angular_mae_deg(pred @ R.T, true @ R.T)
    if abs(e_rot - e_rand) > 1e-6:
        ok = False
        parts.append(f"旋转后偏差={abs(e_rot - e_rand):.2e}°")
    else:
        parts.append("旋转不变")

    # 4) 置换不变
    perm = np.random.default_rng(3).permutation(50000)
    e_perm = _angular_mae_deg(pred[perm], true[perm])
    if abs(e_perm - e_rand) > 1e-6:
        ok = False
        parts.append(f"置换后偏差={abs(e_perm - e_rand):.2e}°")
    else:
        parts.append("置换不变")

    return CheckResult("S22 随机锚点 + 不变性 (数学真相锚)", "单元回归", ok,
                       "; ".join(parts))

@check("S17 冲突检测: >40° 双源触发 conflict_matches", "管道断裂探测")
def chk_s17_conflict_detection():
    """构造角距 >40° 的双源冲突场景, 验证 fuse_bundle 的 conflict_matches 非空."""
    try:
        from fractureflow.l4.fuse import hungarian_match_groups, classify_consistency
        from fractureflow.l4.evidence import EvidenceEntry, EvidenceBundle, build_bundle
        from fractureflow.l4.fuse import fuse_bundle

        # 构造两组角距 >40° 的法向簇
        rng = np.random.default_rng(42)
        # 组 1: 近水平面 (dip ~10°)
        n1 = 30
        dip1 = rng.normal(10, 3, n1).clip(1, 89)
        dd1 = rng.normal(0, 5, n1) % 360
        from fractureflow.l4.evidence import dip_dipdir_to_normal
        normals1 = dip_dipdir_to_normal(dip1, dd1)

        # 组 2: 近垂直面 dip ~70°, 倾向差 >40° (dd ~120°)
        n2 = 30
        dip2 = rng.normal(70, 5, n2).clip(40, 89)
        dd2 = rng.normal(120, 5, n2) % 360
        normals2 = dip_dipdir_to_normal(dip2, dd2)

        entry1 = EvidenceEntry(
            source_type="L0", source_name="well_A",
            data_type="normal_obs", normals=normals1)
        entry2 = EvidenceEntry(
            source_type="L1", source_name="well_B",
            data_type="normal_obs", normals=normals2)
        bundle = build_bundle([entry1], [entry2], site_name="s17_synthetic")

        result = fuse_bundle(bundle, K=2, seed=42)

        n_conflicts = len(result.conflict_matches)
        ok = n_conflicts > 0
        detail = f"conflict_matches 数量 = {n_conflicts}"
        if ok:
            # 验证冲突匹配中的角距确实 >35°
            max_angle = max(cm.angle_deg for cm in result.conflict_matches)
            detail += f", 最大角距 = {max_angle:.1f}°"
            if max_angle <= 35:
                ok = False
                detail += " (应 >35°)"
            else:
                detail += " (>35°, conflict 正确触发)"
        return CheckResult("S17 冲突检测: >40° 双源触发 conflict_matches",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S17 冲突检测: >40° 双源触发 conflict_matches",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


@check("S18 聚类管道: spherical_kmeans 输出有效组系", "管道断裂探测")
def chk_s18_clustering_pipeline():
    """构造 50 条合成法向 (2 组), 验证 spherical_kmeans 正常输出 centers 且 K>0."""
    try:
        from fractureflow.setlabel import spherical_kmeans
        from fractureflow.l4.evidence import dip_dipdir_to_normal

        rng = np.random.default_rng(99)
        # 第一组: dip ~20°, dd ~0° (北)
        n1 = 25
        dip1 = rng.normal(20, 5, n1).clip(1, 89)
        dd1 = rng.normal(0, 10, n1) % 360
        normals1 = dip_dipdir_to_normal(dip1, dd1)

        # 第二组: dip ~60°, dd ~180° (南)
        n2 = 25
        dip2 = rng.normal(60, 5, n2).clip(1, 89)
        dd2 = rng.normal(180, 10, n2) % 360
        normals2 = dip_dipdir_to_normal(dip2, dd2)

        all_normals = np.concatenate([normals1, normals2], axis=0)

        centers, assign = spherical_kmeans(all_normals, K=2, seed=42)

        ok = (centers is not None and len(centers) == 2 and
              assign is not None and len(assign) == 50 and
              len(np.unique(assign)) == 2)
        detail = f"centers.shape={centers.shape}, assign 唯一组={np.unique(assign)}, N={len(assign)}"
        if not ok:
            detail += " (应 K=2, 50 条, 2 组均非空)"
        return CheckResult("S18 聚类管道: spherical_kmeans 输出有效组系",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S18 聚类管道: spherical_kmeans 输出有效组系",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


@check("S19 SetTable 生成: set_table_from_normals K=4", "管道断裂探测")
def chk_s19_settable_generation():
    """构造 K=4 合成法向数据, 验证 set_table_from_normals 正常生成 SetTable."""
    try:
        from fractureflow.dfn import set_table_from_normals
        from fractureflow.l4.evidence import dip_dipdir_to_normal

        rng = np.random.default_rng(7)
        # 四组法向, 角距明显分离
        group_params = [
            (15, 0, 20),    # dip 15°, dd 0°
            (35, 90, 20),   # dip 35°, dd 90°
            (55, 180, 20),  # dip 55°, dd 180°
            (75, 270, 20),  # dip 75°, dd 270°
        ]
        all_normals = []
        for dip_mean, dd_mean, n in group_params:
            dips = rng.normal(dip_mean, 5, n).clip(1, 89)
            dds = rng.normal(dd_mean, 8, n) % 360
            normals = dip_dipdir_to_normal(dips, dds)
            all_normals.append(normals)
        all_normals = np.concatenate(all_normals, axis=0)

        st, set_ids = set_table_from_normals(all_normals, K=4, seed=42)

        ok = (st is not None and st.centers is not None and
              st.centers.shape[0] == 4 and len(set_ids) == 80 and
              len(np.unique(set_ids)) == 4)
        if ok:
            detail = f"SetTable K={st.centers.shape[0]}, proportions={st.proportions.round(3).tolist()}"
        else:
            detail = f"centers.shape={st.centers.shape if st.centers is not None else None}, set_ids 唯一={np.unique(set_ids)}"
        return CheckResult("S19 SetTable 生成: set_table_from_normals K=4",
                           "管道断裂探测", ok, detail)
    except Exception as e:
        return CheckResult("S19 SetTable 生成: set_table_from_normals K=4",
                           "管道断裂探测", False, f"异常: {e}", traceback.format_exc())


_IS_WIN32 = (sys.platform == "win32")


def _set_console_ctrlc_ignore(ignore: bool) -> None:
    """win32: 进程级忽略/恢复控制台 CTRL_C (NULL handler), 非 win32 no-op.

    R190 S20 韧性加固 (T1 授权的环境修复路线): R61 登记的外部 CTRL_C 注入源
    对 1075s+ 的全量 pytest 窗口命中率极高, 原三次退避重试是"每次重新起跑"
    模式, 风暴期必然耗尽额度 (39 PASS / 1 FAIL 终态即被它压住)。双侧免疫:
      ① 子进程 CREATE_NEW_PROCESS_GROUP → 组内 CTRL_C 默认禁用 (MSDN 语义),
         pytest 不再死于 0xC000013A;
      ② 父进程等待窗口忽略 CTRL_C → subprocess.run 不再被 KeyboardInterrupt
         打断, 一次起跑跑到底; finally 必恢复, 本窗口之外 Ctrl+C 语义不变。
    判据零改动: 真实 rc=1 失败与超时仍不重试、不掩盖 (R61 纪律);
    守卫契约兼容: 仍走 sc.subprocess.run 形态 (test_v_r61_guards C 组 mock 面)。
    """
    if not _IS_WIN32:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, 1 if ignore else 0)
    except Exception:
        pass  # 免疫失败则退化为原退避重试护栏, 不改变判据


def _run_pytest_subprocess(max_attempts=3) -> Tuple[bool, str]:
    """调用 pytest 跑 tests/ 下单测, 返回 (全部通过, 详情字符串).

    R61 加固 (2026-08-26): 本机存在周期性外部 CTRL_C 注入源 (用户态自动化工具
    嫌疑), 父进程等待与 pytest 子进程执行均会被随机打断 (一夜实测 7 次, 时点
    145~224s 散布; 直接运行偶有幸免; 子进程中断点 r30_frame_audit.py:281 为
    随机执行位置, 该文件本身无嫌疑)。对策: 仅对 Ctrl-C 类中断 (父侧
    KeyboardInterrupt / 子侧 STATUS_CONTROL_C_EXIT=0xC000013A) 退避重试;
    真实测试失败 (rc=1 等) 与超时**不重试**, 不掩盖。
    R190 (2026-09-08): 重试升级为双侧免疫 (见 _set_console_ctrlc_ignore),
    重试护栏原样保留作为免疫失效时的退化路径。
    """
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", _TESTS_DIR, "-v", "--tb=short", "-q"]
    # R188: 给子进程打嵌套标记。S20 自己就是"selfcheck 里再跑 pytest", 若某条单测又
    # 去实跑 selfcheck (入口对账类守卫), 成本会按层数放大并挤占下面的 1500s 预算。
    # 见到该标记的重负载守卫应就地降级 (skip), 其覆盖不丢: 门禁命令清单第 ② 步
    # (`python -m pytest tests/`) 与外层 selfcheck 本身已分别承担两侧证据。
    child_env = {**os.environ, "FRACTUREFLOW_SELFCHECK_NESTED": "1"}
    # R8-S20 (2026-08-24): 实测 225.02s → 取 2x=450;
    # R79 (2026-08-29 静默窗口): 实测全量 1075.32s (13 failed / 976 passed / 1 deselected,
    # 17m55s) → 450 与 R80 建议的 900 都必然超时, 预算提到 1500s (~1.4x 余量)。
    pytest_budget_s = 1500
    status_ctrl_c = -1073741510  # 0xC000013A STATUS_CONTROL_C_EXIT
    last_note = "pytest 未运行"
    for attempt in range(1, max_attempts + 1):
        try:
            _set_console_ctrlc_ignore(True)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=pytest_budget_s,
                    cwd=_PROJECT_ROOT, env=child_env,
                    creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | subprocess.CREATE_NO_WINDOW,
                )
            finally:
                _set_console_ctrlc_ignore(False)
            output = proc.stdout + "\n" + proc.stderr
            summary_lines = [l for l in output.splitlines()
                             if "passed" in l or "failed" in l]
            summary = (summary_lines[-1].strip() if summary_lines
                       else f"exit code={proc.returncode}")
            if proc.returncode == 0:
                return True, summary
            if proc.returncode == status_ctrl_c:
                last_note = f"子进程被环境 CTRL_C 击中 (rc={proc.returncode})"
            else:
                return False, summary  # 真实失败原样上报, 不重试
        except subprocess.TimeoutExpired:
            return False, "pytest 超时 (>%ds)" % pytest_budget_s
        except FileNotFoundError:
            return False, "pytest 未安装"
        except KeyboardInterrupt:
            last_note = "父进程等待被环境 CTRL_C 击中"
        if attempt < max_attempts:
            time.sleep(20 * attempt)  # 错开外部注入源相位
    return False, (f"{last_note}; 重试 {max_attempts} 次仍被打断 "
                   f"(R61 登记: 环境 CTRL_C 风暴, 非测试失败)")


@check("S21 数字台账: JSON 一致性 + 落单扫描", "管道断裂探测")
def chk_s21_digital_ledger():
    """验证 digital_ledger.json 中引用的 active JSON 全部存在且值一致 (容差 0.1°).
    结构守卫: entries/deprecated 的 ID 全文件唯一 + changelog 必须为正规变更单
    (date/task/changes) 格式, 防止 ledger 对象误贴进 changelog 造成的重复条目.
    同时扫描 results/ 下未被台账引用的 JSON (落单数字), 仅报告不报错.
    """
    ledger_path = os.path.join(_RESULTS_DIR, "digital_ledger.json")
    if not os.path.exists(ledger_path):
        return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                           "管道断裂探测", False,
                           "digital_ledger.json 不存在")

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
    except Exception as e:
        return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                           "管道断裂探测", False, f"JSON 解析失败: {e}")

    entries = ledger.get("entries", [])

    # 结构守卫 (2026-08-29 外部审查台账自愈): ID 唯一性 + changelog 变更单格式.
    # 背景: 曾有 NNS-051/052 原始 ledger 对象误贴进 changelog 数组 (与 entries 重复
    # 且内容有出入), 而 S21 只查 entries->source_json 值一致性不查结构, 造成
    # "扫描通过"的假安全感. 本守卫把该模式提为硬性扫描规则 (L5 铁律).
    struct_errs = []
    seen = {}
    for arr in ("entries", "deprecated"):
        for i, e in enumerate(ledger.get(arr, [])):
            eid = e.get("id", "")
            if not eid:
                struct_errs.append(f"{arr}[{i}] 缺 id 字段")
                continue
            if eid in seen:
                struct_errs.append(f"ID 重复: {eid} 同时出现于 {seen[eid]} 与 {arr}[{i}]")
            else:
                seen[eid] = f"{arr}[{i}]"
    for i, c in enumerate(ledger.get("changelog", [])):
        if not ("date" in c and "task" in c and isinstance(c.get("changes"), list)):
            struct_errs.append(f"changelog[{i}] 非正规变更单格式 (需 date/task/changes list)")

    mismatch = []
    missing = []
    unextractable = []  # P1 修复: 提取失败单独计数, 防假 PASS
    checked = 0
    tolerance = 0.1  # deg (浮点/多 seed 容差)

    for entry in entries:
        if entry.get("status") not in ("active", "active-oracle"):
            continue
        rel_path = entry.get("source_json", "")
        if not rel_path:
            continue
        full_path = os.path.join(_PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # 尝试从 JSON 中提取 mae, 支持 k_level 和 source_path 提示
        k_hint = entry.get("k_level", "12")
        source_path = entry.get("source_path", "")
        actual = _extract_mae(data, k_hint=k_hint, source_path=source_path)
        expected = entry.get("value")
        if actual is not None and expected is not None:
            # 允许容差, 也允许百分比单位 (值域 0-1 vs 0-100)
            scale = 1.0
            if entry.get("unit") == "percent":
                scale = 100.0
            diff = abs(actual * scale - expected)
            if diff > tolerance:
                mismatch.append(f"{rel_path}: expected={expected}, actual={actual}, diff={diff:.4f}")
            checked += 1
        else:
            # 提取失败 → 单独计数, 不静默计入 checked (防假 PASS)
            skip_reason = ""
            if not rel_path:
                skip_reason = "no source_json"
            elif actual is None:
                skip_reason = "extract_mae=NaN"
            elif expected is None:
                skip_reason = "expected=None (台账无 value 字段)"
            unextractable.append(f"{rel_path}: {skip_reason}")

    # 落单 JSON 扫描 (仅报告, 不影响 PASS/FAIL)
    referenced_files = set()
    for entry in entries:
        rj = entry.get("source_json", "")
        if rj:
            referenced_files.add(os.path.normpath(os.path.join(_PROJECT_ROOT, rj)))

    orphaned = []
    for root, dirs, files in os.walk(_RESULTS_DIR):
        for f in files:
            if f.endswith(".json") and f != "digital_ledger.json":
                full = os.path.normpath(os.path.join(root, f))
                if full not in referenced_files:
                    # 排除已知被 docs 引用的 (在 selfcheck 内不做文档扫描, 仅报告)
                    orphaned.append(os.path.relpath(full, _PROJECT_ROOT))

    detail = (f"checked={checked}, mismatch={len(mismatch)}, "
               f"missing={len(missing)}, unextractable={len(unextractable)}, "
               f"struct_errs={len(struct_errs)}, orphaned_files={len(orphaned)}")
    if missing:
        detail += f", MISSING: {missing[:3]}"
    if mismatch:
        detail += f", MISMATCH: {mismatch[:3]}"
    if unextractable:
        detail += f", UNEXTRACTABLE: {unextractable[:3]}"
    if struct_errs:
        detail += f", STRUCT_ERR: {struct_errs[:3]}"
    if orphaned:
        detail += f", ORPHANNED_SAMPLE: {orphaned[:5]}"

    # PASS 条件: 无 mismatch + 无 missing + 无 unextractable + 无结构错误
    ok = (len(mismatch) == 0 and len(missing) == 0 and
          len(unextractable) == 0 and len(struct_errs) == 0)
    return CheckResult("S21 数字台账: JSON 一致性 + 落单扫描",
                       "管道断裂探测", ok, detail)


def _extract_mae(data, k_hint: str = "12", source_path: str = "") -> Optional[float]:
    """从多种常见 JSON 结构中提取 mae 数值.

    优先级:
      0. source_path 指定路径 (如 "threshold_sources.information_ceiling_31_7deg.value")
      1. 顶层 mae_mean / mae / value (honest_leaderboard 格式)
      2. 嵌套 K-level modal_err_mean (set_table 格式, 按 k_hint 选取)
      3. result 单键包装 (如 beishan_kmeans_obs_k12.json 的 result.mae_mean)
      4. 深层递归搜索 modal_err_mean / mae_mean (phase0_oracle_floor_strict 等)
    """
    if not isinstance(data, dict):
        return None
    # 0. source_path 指定路径 (支持 dict key 和 list 下标, 如 "site_results.0.value")
    if source_path:
        parts = source_path.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list):
                try:
                    cur = cur[int(p)]
                except (ValueError, IndexError):
                    cur = None
                    break
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    # 1. 顶层直接键
    for key in ("mae_mean", "mae", "value"):
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
    # 2. 嵌套 K-level 结构 (set_table): 按 k_hint 选取
    hint_val = None
    last_val = None
    for k, v in data.items():
        if isinstance(v, dict):
            for key in ("modal_err_mean", "mae_mean", "mae"):
                if key in v and isinstance(v[key], (int, float)):
                    val = float(v[key])
                    last_val = val
                    if k == k_hint:
                        hint_val = val
    if hint_val is not None:
        return hint_val
    if last_val is not None:
        return last_val
    # 3. result 单键包装: 如果 data 只有一个 dict 值, 递归进入
    dict_vals = [v for v in data.values() if isinstance(v, dict)]
    if len(dict_vals) == 1:
        inner = _extract_mae(dict_vals[0], k_hint=k_hint)
        if inner is not None:
            return inner
    # 4. 深层递归搜索: 收集所有 modal_err_mean/mae_mean, 优先匹配 k_hint 路径
    best_hint = None
    best_any = None

    def _recurse(obj, path_keys):
        nonlocal best_hint, best_any
        if not isinstance(obj, dict):
            return
        for key in ("modal_err_mean", "mae_mean", "mae"):
            if key in obj and isinstance(obj[key], (int, float)):
                val = float(obj[key])
                if best_any is None:
                    best_any = val
                if k_hint in path_keys and best_hint is None:
                    best_hint = val
        for k, v in obj.items():
            if isinstance(v, dict):
                _recurse(v, path_keys + [k])

    _recurse(data, [])
    if best_hint is not None:
        return best_hint
    return best_any


@check("S20 --with-tests: pytest 整合", "管道断裂探测")
def chk_s20_pytest_integration():
    """验证 pytest 调用管道通畅, tests/ 下单测全部通过."""
    # 检查是否有 pytest 可用
    python_exe = sys.executable
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if proc.returncode != 0:
            return CheckResult("S20 --with-tests: pytest 整合",
                               "管道断裂探测", False,
                               "pytest 不可用 (exit code={})".format(proc.returncode))
    except Exception as e:
        return CheckResult("S20 --with-tests: pytest 整合",
                           "管道断裂探测", False,
                           f"pytest 版本检查失败: {e}")

    # 检查 tests/ 目录下是否有测试文件
    test_files = [f for f in os.listdir(_TESTS_DIR) if f.startswith("test_") and f.endswith(".py")]
    if not test_files:
        return CheckResult("S20 --with-tests: pytest 整合",
                           "管道断裂探测", False,
                           "tests/ 下无测试文件")

    # 实际跑 pytest (仅当 tests/ 下有文件时)
    passed, summary = _run_pytest_subprocess()
    return CheckResult("S20 --with-tests: pytest 整合",
                       "管道断裂探测", passed,
                       f"tests 文件: {test_files}; {summary}")


# ---------------------------------------------------------------------------
# 主运行器
# ---------------------------------------------------------------------------

# --quick 轻量模式跳过的检查 (按注册名)。只减**执行集**, 不改判据、不改注册数:
# 跳过的项一律记入报告 summary.skipped, 绝不伪装成 PASS/FAIL。
# 唯一成员 = S20: 全库唯一会跑完整 pytest 的检查 (预算 1500s, 实测 15-20 min),
# 也是守卫/并行复跑直跑入口时的唯一慢点。
_QUICK_SKIP = ("S20 --with-tests: pytest 整合",)


# ---------------------------------------------------------------------------
# 崩溃取证 (NATIVE-CRASH-REPRODUCIBILITY-HARDENING, 2026-11-07)
#   背景: 外部沙箱钩子 DLL (sandboxcap.dll) 在进程内触发 0xC0000409 快失败时,
#   进程无 traceback、无报告落盘, 历史故障因此无法回答"死在哪个检查"。
#   本段只做**观测**, 不参与任何判据、不改报告 schema、不改 PASS/FAIL:
#     1) 逐检查追加一行 JSONL, 且 check_start 先 flush+fsync 再执行检查体,
#        因此原生崩溃后仍能从最后一行定位首个失败点;
#     2) 首行 env 记录本进程是否被已知外部钩子 DLL 注入 (只记录, 不判定)。
#   进度文件路径: 环境变量 FF_SELFCHECK_PROGRESS 优先, 否则 <--out>.progress.jsonl。
# ---------------------------------------------------------------------------

_HOOK_MODULES = ("sandboxcap.dll", "network_hook.dll")


def _hook_module_loaded(name: str):
    """探测本进程是否加载了指定外部 DLL; 探测失败返回 None (不影响判定)."""
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetModuleHandleW(name))
    except Exception:
        return None


def resolve_progress_path(out_path: Optional[str]) -> Optional[str]:
    """进度日志路径: 环境变量优先, 否则与报告同目录的 <report>.progress.jsonl."""
    env = os.environ.get("FF_SELFCHECK_PROGRESS")
    if env:
        return os.path.abspath(env)
    if out_path:
        return os.path.abspath(out_path) + ".progress.jsonl"
    return None


def _progress_write(fh, payload: dict) -> None:
    """追加一行进度记录并立即落盘 (flush+fsync), 供原生崩溃后取证."""
    if fh is None:
        return
    try:
        record = dict(payload)
        record.setdefault("measured_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    except Exception:
        pass


def _progress_env_record() -> dict:
    """进程级环境记录 (只观测): 是否被已知外部钩子注入、环境标记、解释器信息."""
    return {
        "event": "env",
        "pid": os.getpid(),
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "platform": sys.platform,
        "sandbox_env": {k: v for k, v in os.environ.items() if k.startswith("SANDBOX_")},
        "hook_modules_loaded": {name: _hook_module_loaded(name) for name in _HOOK_MODULES},
    }


def run_all_checks(skip_slow: bool = False,
                   progress_path: Optional[str] = None) -> Dict[str, List[CheckResult]]:
    """运行所有注册检查, 按组返回结果.

    skip_slow=True (--quick): 跳过 _QUICK_SKIP 中的慢检查 (不判定, 只登记 skipped).
    progress_path: 崩溃取证 JSONL 路径; None = 不写进度 (默认关闭, 不改变任何判定).
    """
    results: Dict[str, List[CheckResult]] = {}
    progress = None
    if progress_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(progress_path)), exist_ok=True)
            progress = open(progress_path, "a", encoding="utf-8")
        except Exception:
            progress = None
    try:
        _progress_write(progress, _progress_env_record())
        for index, (group, name, fn) in enumerate(_ALL_CHECKS):
            if skip_slow and name in _QUICK_SKIP:
                _progress_write(progress, {"event": "check_skip", "index": index,
                                           "group": group, "name": name})
                continue
            _progress_write(progress, {"event": "check_start", "index": index,
                                       "group": group, "name": name})
            started = time.time()
            try:
                result = fn()
            except Exception as e:
                result = CheckResult(name, group, False, f"异常: {e}", traceback.format_exc())
            _progress_write(progress, {"event": "check_end", "index": index,
                                       "group": group, "name": name,
                                       "passed": bool(result.passed),
                                       "elapsed_s": round(time.time() - started, 3)})
            results.setdefault(group, []).append(result)
    finally:
        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass
    return results


def print_report(results: Dict[str, List[CheckResult]]) -> bool:
    """打印报告, 返回是否全部通过."""
    total_passed = 0
    total_failed = 0
    print("=" * 60)
    print("  fractureflow 一键自检报告")
    print("=" * 60)
    for group, checks in results.items():
        print(f"\n【{group}】")
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            symbol = "✓" if c.passed else "✗"
            print(f"  {symbol} [{status}] {c.name}: {c.message}")
            if not c.passed and c.detail:
                print(f"         详情: {c.detail[:200]}")
            if c.passed:
                total_passed += 1
            else:
                total_failed += 1
    print(f"\n{'=' * 60}")
    print(f"  总计: {total_passed} PASS, {total_failed} FAIL")
    print(f"{'=' * 60}")
    return total_failed == 0


def save_report(results: Dict[str, List[CheckResult]], path: str,
                summary_extra: Optional[dict] = None):
    """保存报告到 JSON.

    summary_extra: 运行模式元信息 (mode / registered_total / skipped)。
    刻意不写时间戳/主机名/耗时 —— 同一模式下两个入口的报告必须可逐字段 (乃至字节) 比对。
    """
    data = {
        "groups": {
            group: [c.to_dict() for c in checks]
            for group, checks in results.items()
        },
        "summary": {
            "total": sum(len(c) for c in results.values()),
            "passed": sum(1 for checks in results.values() for c in checks if c.passed),
            "failed": sum(1 for checks in results.values() for c in checks if not c.passed),
        },
    }
    if summary_extra:
        data["summary"].update(summary_extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder, allow_nan=False)


def main():
    """主入口 (直跑与 -m 两形态等价, R188 起).

    支持 CLI 参数:
        --with-tests: 额外调用 pytest 运行 tests/ 下单测并汇总结果 (S20 之外再跑一次).
        --quick     : 跳过 _QUICK_SKIP 中的慢检查 (默认全量含 S20 全量 pytest, 15-20 min).
        --out PATH  : 报告落盘路径, 默认 results/selfcheck_report.json.
    退出码: 0 = 全部通过, 1 = 有红账 (与是否哪个入口无关)。
    """
    parser = argparse.ArgumentParser(
        description="fractureflow 一键自检 (v2, 含管道断裂探测)")
    parser.add_argument("--with-tests", action="store_true", default=False,
                        help="额外运行 pytest 单测 (耗时增加, 默认关闭)")
    parser.add_argument("--quick", action="store_true", default=False,
                        help="轻量模式: 跳过 S20 全量 pytest 慢检查 (只减执行集, 不改判据)")
    parser.add_argument("--out", default=os.path.join(_RESULTS_DIR, "selfcheck_report.json"),
                        help="报告落盘路径 (默认 results/selfcheck_report.json; "
                             "守卫/并行在飞时建议指向临时目录, 避免覆盖现报)")
    args = parser.parse_args()

    skipped = list(_QUICK_SKIP) if args.quick else []
    if skipped:
        print("[--quick] 跳过慢检查 (登记于报告 summary.skipped): " + ", ".join(skipped))

    out_path = os.path.abspath(args.out)
    progress_path = resolve_progress_path(out_path)
    results = run_all_checks(skip_slow=args.quick, progress_path=progress_path)

    # 如果开启 --with-tests, 跑 pytest 并追加结果
    if args.with_tests:
        print("\n[--with-tests 模式] 运行 pytest 单测...")
        passed, summary = _run_pytest_subprocess()
        pytest_result = CheckResult(
            "pytest 单测汇总", "管道断裂探测", passed, summary)
        results.setdefault("管道断裂探测", []).append(pytest_result)

    all_passed = print_report(results)
    save_report(results, out_path, summary_extra={
        "mode": "quick" if args.quick else "full",
        "registered_total": len(_ALL_CHECKS),
        "skipped": skipped,
    })
    print(f"\n报告已落盘: {out_path}")
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# B3-4 几何约定 grep 门禁 (集中化根治三次犯案的结构性方案)
#   - 独立 S 编号 (S33), 与 R8 初级单在修的 S20/S31 不同区域, 无碰撞风险.
#   - 若 R8 初级单交付覆盖了本文件, 请重新追加本函数 (门禁脚本为权威, 见 scripts/).
# ---------------------------------------------------------------------------
@check("S33 几何约定 grep 门禁 (手搓产状转换拦截)", "几何约定")
def chk_s33_geometry_conventions():
    """B3-4: 扫描 src/fractureflow 下手搓产状<->法向/走向 转换 (BUG-A/BUG-B 类).

    调用 scripts/check_geometry_conventions.py; 发现违规 (手搓 atan2+sin/cos
    构造法向/走向, 或 BUG-A 轴互换) 即 FAIL。集中权威实现见 geometry.py。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    script = os.path.join(root, "scripts", "check_geometry_conventions.py")
    if not os.path.exists(script):
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          f"门禁脚本缺失: {script}")
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          "门禁脚本执行超时 (>120s)")
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        return CheckResult("S33 几何约定 grep 门禁", "几何约定", False,
                          "发现手搓产状转换违规 (见详情)", out[-1500:])
    return CheckResult("S33 几何约定 grep 门禁", "几何约定", True, "0 处违规")


# ---------------------------------------------------------------------------
# W38 产状<->法向 参数序与往返硬闸 (F-W31-2 兑现; 与 S32 并列, 不取代 S32)
#   S32 只断言 nz 与单位性 => 水平分量 sin/cos 互换 (十五期 _nd 事故类) 从它眼底
#   过去; S35 补的是"与权威三分量一致 + 参数序台账 + 逆向铁律分档 + 新增实现禁令"。
#   判据本体在 scripts/w38_geometry_gate.py (单一实现, 可独立跑)。
# ---------------------------------------------------------------------------
@check("S35 产状<->法向参数序台账与三分量往返硬闸", "几何约定")
def chk_s35_attitude_argorder_gate():
    """W38: 每个登记生成器逐分量等于 geometry 权威, 偏差 > 1e-9 即 FAIL。

    覆盖四件 (缺一即本条无牙, 毒丸见 tests/test_v_w38_.py):
      A 生成器 vs 权威 三分量一致 (抓 sin/cos 水平分量互换 = S32 盲区);
      B 参数序台账 (倾角在第几槽, 运行时实测登记, 改了槽即红);
      C 逆向无向铁律分档 (FOLDS/FLIPS_180 改档即红, 往好改也一样红 —— 改档=改在册);
      D 新增未登记转换器定义禁令 (src 递归 + scripts 顶层; 白名单只准缩不准扩)。
    """
    NAME = "S35 产状<->法向参数序台账与三分量往返硬闸"
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    script = os.path.join(root, "scripts", "w38_geometry_gate.py")
    if not os.path.exists(script):
        return CheckResult(NAME, "几何约定", False, f"门禁脚本缺失: {script}")
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return CheckResult(NAME, "几何约定", False, "门禁执行超时 (>300s)")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return CheckResult(NAME, "几何约定", False, "产状<->法向硬闸违规 (见详情)", out[-1800:])
    return CheckResult(NAME, "几何约定", True, out.strip().splitlines()[-1] if out.strip() else "PASS")


# ---------------------------------------------------------------------------
# P17b 静默半加载扫描 (2026-09-02 审查发现)
#   背景: t82_e2_drift_gate / compare_locked 等 11 处曾把 ckpt ema 按
#   模型 named_parameters 键集过滤后再以 strict=False 载入 ——
#   e3gt_hybrid.pt 缺 residual_head 4 键时该头保持随机初始化且无任何警告,
#   漂移闸门/排行榜数字被静默污染 (实测半随机模型 MAE 53~65° vs 训练 ~35°)。
#   统一权威入口 = trainer.load_ema_strict (缺键/形状失配必 raise)。
#   注: 本注释不得写出会被下方正则命中的字面 idiom (自引用陷阱, 同废弃数字扫描)。
# ---------------------------------------------------------------------------
@check("S34 静默半加载扫描 (named_parameters 过滤 + strict=False)", "管道断裂探测")
def chk_s34_silent_halfload():
    """L5 规则: 禁止按模型参数键集过滤 ckpt 字典后 strict=False 半加载的 idiom."""
    import re
    NAME = "S34 静默半加载扫描 (named_parameters 过滤 + strict=False)"
    try:
        pat = re.compile(r"in\s+dict\(\s*\w+\.named_parameters\(\)\s*\)")
        bad = []
        for d in (_SCRIPTS_DIR, _SRC_DIR):
            for dirpath, _, files in os.walk(d):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        txt = open(p, encoding="utf-8").read()
                    except Exception:
                        continue
                    for i, line in enumerate(txt.splitlines(), 1):
                        if pat.search(line):
                            bad.append(f"{os.path.relpath(p, _PROJECT_ROOT)}:{i}")
        ok = not bad
        detail = ("全库无过滤式半加载 idiom (统一走 trainer.load_ema_strict)"
                  if ok else f"残留 {len(bad)} 处: {bad[:4]}")
    except Exception as e:
        return CheckResult(NAME, "管道断裂探测", False, str(e))
    return CheckResult(NAME, "管道断裂探测", ok, detail)


# ---------------------------------------------------------------------------
# R185 方向统计 (Bingham / ACG / 可微 Terzaghi)
# ---------------------------------------------------------------------------

@check("R185 配分函数对 Dawson 闭式", "R185 方向统计")
def chk_r185_norm_dawson():
    """Bingham 配分函数 p=3 对称族有 Dawson 闭式, 数值积分必须对上."""
    try:
        import scipy.special as sp
        from fractureflow.orthostats import log_bingham_norm_np
        worst, arg = 0.0, None
        for k in (0.0, 1.0, 20.0, 100.0):
            got = float(np.exp(log_bingham_norm_np(np.array([0.0, k, k]))))
            want = 4.0 * np.pi if k == 0 else 4.0 * np.pi * sp.dawsn(np.sqrt(k)) / np.sqrt(k)
            r = abs(got - want) / want
            if r > worst:
                worst, arg = r, k
        ok = worst < 1e-10
        return CheckResult("R185 配分函数对 Dawson 闭式", "R185 方向统计", ok,
                           f"max rel err {worst:.2e} (<1e-10)" if ok
                           else f"k={arg} 相对误差 {worst:.2e} 超闸")
    except Exception as e:
        return CheckResult("R185 配分函数对 Dawson 闭式", "R185 方向统计", False, str(e))


@check("R185 批量拟合 == 单条拟合", "R185 方向统计")
def chk_r185_batch_equals_single():
    """批量 MLE 必须与逐条一致 —— 覆盖率实验与可微层都建立在它之上."""
    try:
        from fractureflow.orthostats import (
            fit_bingham, fit_bingham_batch, log_bingham_norm_np, Bingham)
        lam = np.array([0.0, 15.0, 15.0])
        ln = float(log_bingham_norm_np(lam))
        worst = 0.0
        Xs = []
        for i in range(8):
            b = Bingham(axes=_rand_rot(11 + i), lmbda=lam, log_norm=ln)
            Xs.append(b.sample(60, np.random.default_rng(1000 + i)))
        fb = fit_bingham_batch(torch.tensor(np.stack(Xs), dtype=torch.float64))
        for i in range(8):
            fs = fit_bingham(Xs[i])
            worst = max(worst, float(np.max(np.abs(fb[i].lmbda - fs.lmbda))))
        ok = worst < 1e-6
        return CheckResult("R185 批量拟合 == 单条拟合", "R185 方向统计", ok,
                           f"max |Δλ| = {worst:.2e}" if ok else f"λ 偏差 {worst:.2e}")
    except Exception as e:
        return CheckResult("R185 批量拟合 == 单条拟合", "R185 方向统计", False, str(e))


@check("R185 可微层梯度检验", "R185 方向统计")
def chk_r185_gradcheck():
    """井轴 → (λ, M, S) 的 autograd vs 有限差分 (依赖 bingham._LS_MARGIN 修复)."""
    try:
        from fractureflow.orthostats import (
            TerzaghiBinghamLayer, log_bingham_norm_np, Bingham)
        lam = np.array([0.0, 6.0, 25.0])
        b = Bingham(axes=_rand_rot(5), lmbda=lam,
                    log_norm=float(log_bingham_norm_np(lam)))
        X = torch.tensor(b.sample(150, np.random.default_rng(5)), dtype=torch.float64)
        layer = TerzaghiBinghamLayer(axis=np.array([0.0, 0.0, 1.0]), mode="soft")

        def f(ax):
            o = layer(X, create_graph=True, axis=ax)
            return o["lmbda"].sum() + 2.0 * o["axes"][0, 0] + 3.0 * o["S"][1, 1]

        ax0 = torch.tensor([0.12, -0.05, 0.99], dtype=torch.float64, requires_grad=True)
        g = torch.autograd.grad(f(ax0), ax0)[0].detach().numpy()
        h, fd = 1e-6, np.zeros(3)
        for i in range(3):
            xp, xm = ax0.detach().clone(), ax0.detach().clone()
            xp[i] += h; xm[i] -= h
            fd[i] = (f(xp).item() - f(xm).item()) / (2 * h)
        rel = float(np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1e-8)))
        ok = rel < 1e-5
        return CheckResult("R185 可微层梯度检验", "R185 方向统计", ok,
                           f"max rel err {rel:.2e}" if ok else f"梯度相对误差 {rel:.2e} 超闸")
    except Exception as e:
        return CheckResult("R185 可微层梯度检验", "R185 方向统计", False, str(e))


@check("R185 terzaghi_summary 返回 dict", "R185 方向统计")
def chk_r185_terzaghi_summary():
    """回归守卫: 返回体曾掉进 _validated_normals 的 return 之后, 函数返回 None."""
    try:
        from fractureflow.terzaghi import terzaghi_summary
        out = terzaghi_summary(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]]), well_axis=None)
        need = ("mean_weight", "max_weight", "min_weight", "n_clipped",
                "mean_angle_to_axis_deg")
        ok = isinstance(out, dict) and all(k in out for k in need)
        return CheckResult("R185 terzaghi_summary 返回 dict", "R185 方向统计", ok,
                           "返回完整 dict" if ok else f"返回 {type(out).__name__}")
    except Exception as e:
        return CheckResult("R185 terzaghi_summary 返回 dict", "R185 方向统计", False, str(e))


@check("R185 R185 交付物存在且门全过", "R185 方向统计")
def chk_r185_artifacts():
    """results/r185_bingham/ 下的 JSON 必须存在且 overall/gate 为 PASS."""
    try:
        import json
        base = os.path.join(_PROJECT_ROOT, "results", "r185_bingham")
        need = ["prereg.json", "mace_fit_tests.json", "acg_moment_report.json",
                "terzaghi_gradcheck.json", "coverage_calibration.json",
                "e3gt_compat_report.md", "verdict.md"]
        missing = [f for f in need if not os.path.exists(os.path.join(base, f))]
        if missing:
            return CheckResult("R185 R185 交付物存在且门全过", "R185 方向统计", False,
                               f"缺 {missing}")
        cov = json.load(open(os.path.join(base, "coverage_calibration.json"), encoding="utf-8"))
        t0 = json.load(open(os.path.join(base, "mace_fit_tests.json"), encoding="utf-8"))
        t2 = json.load(open(os.path.join(base, "terzaghi_gradcheck.json"), encoding="utf-8"))
        gates = {
            "t3_overall": bool(cov.get("overall_pass")),
            "t0_normalizer": bool(t0["gate"]["normalizer_gate"]),
            "t0_recovery": bool(t0["gate"]["fit_recovery_gate"]),
            "t2_gradcheck": bool(t2["gradcheck"]["gate"]),
            "t2_legacy_agree": bool(t2["legacy_agreement"]["gate"]),
        }
        bad = [k for k, v in gates.items() if not v]
        ok = not bad
        return CheckResult("R185 R185 交付物存在且门全过", "R185 方向统计", ok,
                           f"门全过: {gates}" if ok else f"未过: {bad}")
    except Exception as e:
        return CheckResult("R185 R185 交付物存在且门全过", "R185 方向统计", False, str(e))


if __name__ == "__main__":
    sys.exit(main())
