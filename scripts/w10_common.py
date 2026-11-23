# -*- coding: utf-8 -*-
"""W10 · 共享件 —— 全部估计器/原子写**委托**给 W9 的冻结实现，禁在本线重写公式。

设计（裁定 95.1 ④「优雅且完备优先」）：
  · kish_neff / ci_cluster_general / atomic_write_json / sha256_file  = 直接复用 scripts/w9_common.py
  · 主箱对数与逐网记录 = 复用 scripts/r226_anchor_305.py::collect（原计算件同码路径）
  · 本线只新增：候选源 → net dict 装配器、只读面 sha 采集、盘面普查工具
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "results", "w10_site_pool_inventory")

# ── 委托 W9 冻结件（单一实现，防口径分叉）────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "w10_w9_ref", os.path.join(ROOT, "scripts", "w9_common.py"))
W9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W9)

kish_neff = W9.kish_neff
ci_cluster_general = W9.ci_cluster_general
m_bkey = W9.m_bkey
primary_bin_arrays = W9.primary_bin_arrays
net_labels = W9.net_labels
group_arrays = W9.group_arrays
PRIMARY_BIN = W9.PRIMARY_BIN
Z95 = W9.Z95
BOOT_SEED = W9.BOOT_SEED
N_BOOT = W9.N_BOOT
atomic_write_json = W9.atomic_write_json
atomic_write_text = W9.atomic_write_text
sha256_file = W9.sha256_file
sha256_bytes = W9.sha256_bytes
load_r226 = W9.load_r226


def pin_threads():
    return W9.pin_threads()


def assert_env():
    return W9.assert_env()


# ── 只读面 sha 采集 ─────────────────────────────────────────────────
def snapshot_sha(relpaths):
    """对只读面取 sha256；缺失文件如实记 MISSING，不静默跳过（军规⑭）。"""
    out = {}
    for rp in relpaths:
        ap = rp if os.path.isabs(rp) else os.path.join(ROOT, rp.replace("\\", "/"))
        if os.path.isfile(ap):
            out[rp] = {"exists": True, "sha256": sha256_file(ap),
                       "bytes": os.path.getsize(ap)}
        else:
            out[rp] = {"exists": False, "sha256": None, "bytes": None,
                       "note": "MISSING（未静默跳过）"}
    return out


def read_json(relpath):
    ap = relpath if os.path.isabs(relpath) else os.path.join(ROOT, relpath)
    with open(ap, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


# ── 候选源 → net dict 装配 ──────────────────────────────────────────
def attitude_pair_to_normal(dip_deg, dipdir_deg):
    """产状 (dip, dip_direction) → 单位法向。

    与仓库现役口径一致（AGENTS.md 十五期 `_nd` 修复后的形式）：
        n = [sin(d)·sin(a), sin(d)·cos(a), cos(d)]
    z 轴向上、a 自北起算。S32 产状↔法向往返检查锁的就是这一约定。
    """
    import numpy as np
    d = np.deg2rad(np.asarray(dip_deg, dtype=np.float64))
    a = np.deg2rad(np.asarray(dipdir_deg, dtype=np.float64))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def make_net(pos, normals, src, wid=None, widx=None):
    """装配成 loaded_real_nets 同构 net dict（r226::collect 只读 pos/nrm）。"""
    import numpy as np
    import torch
    pos = np.asarray(pos, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    return {"pos": torch.from_numpy(pos), "nrm": torch.from_numpy(nrm),
            "len": torch.ones((len(pos), 1), dtype=torch.float32),
            "lith": torch.zeros(len(pos), dtype=torch.long),
            "s1": torch.tensor([1.0, 0.0, 0.0]), "s3": torch.tensor([0.0, 1.0, 0.0]),
            # prepare_net 对已存在但值为 None 的键会 int(None) 崩（本单实测）⇒ 显式落默认值
            "src": src, "wid": "" if wid is None else str(wid),
            "widx": 0 if widx is None else int(widx)}


def primary_pairs_by_cluster(r226_mod, nets):
    """用 r226 同码路径算逐网主箱对数，返回 (per_net_records, total_pairs, nets_with_bin)。

    只取对数与角度数组；**不在本单汇总成任何新的点估计作为锚**（红线 1）。
    """
    import numpy as np
    lp = r226_mod.load_original()
    _agg, per_net = r226_mod.collect(lp, nets)
    key = W9.m_bkey(PRIMARY_BIN[0])
    recs = []
    for rec in per_net:
        if key in rec["bins"]:
            b = rec["bins"][key]
            recs.append({"net_idx": int(rec["net_idx"]), "src": rec["src"],
                         "n_points": int(rec["n_points"]), "n_pairs": int(b["n"]),
                         "line_type": bool(rec["line_type"]),
                         "mean_ang_deg_DIAGNOSTIC_ONLY": float(np.mean(b["ang"])),
                         "sum_ang_deg": float(np.sum(b["ang"]))})
    tot = int(sum(r["n_pairs"] for r in recs))
    return recs, tot, len(recs)


def disk_usage(path="C:/"):
    import shutil
    try:
        u = shutil.disk_usage(path)
        return {"path": path, "total_GB": round(u.total / 1024 ** 3, 2),
                "used_GB": round(u.used / 1024 ** 3, 2),
                "free_GB": round(u.free / 1024 ** 3, 2),
                "used_pct": round(100.0 * u.used / u.total, 1)}
    except OSError as e:
        return {"path": path, "error": str(e)}


def now_iso():
    import datetime
    return datetime.datetime.now().astimezone().isoformat()
