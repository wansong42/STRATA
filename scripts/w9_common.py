# -*- coding: utf-8 -*-
"""W9 · 墙锚 CI 的聚类单位与有效样本量审计 —— 共享件（环境/原子写/基线装载/通用簇 bootstrap）

红线（任务书 §3）：本件**唯一旋钮 = 重抽样单位**。
主箱 0.25r、nrm_full 口径、百分位法、簇内整块重抽、n_boot=10000、seed=20260916 全部
逐字沿用 r226（`scripts/r226_anchor_305.py` 的 `collect()` / `ci_cluster()`），不重写、不改公式。

通用簇 bootstrap 与 r226 `ci_cluster` 在 unit=network 上必须**逐位相同**（W9-FID-2 硬闸），
这既是保真闸，也是"单位管道是活的"这一把守的正例。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

# 逐位确定性：在 numpy/BLAS 加载前钉线程（坑 45 / A-8 同族）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "results", "w9_ci_cluster_unit")

# ── 照抄 r226 的常量（禁改）────────────────────────────────────────────
PRIMARY_BIN = (0.0, 0.25)
N_BOOT = 10000
BOOT_SEED = 20260916
Z95 = 1.96

# T0 逐位锚点（源 = results/r226_anchor_305/anchor_stats.json，只读）
ANCHOR_STATS = os.path.join(ROOT, "results", "r226_anchor_305", "anchor_stats.json")
T0_MEAN_FULL = 30.49781036376953      # anchor_value_deg / bin_table[0].mean（全精度）
T0_MEAN_4DP = 30.4978                 # statistics.mean_deg
T0_N_PAIRS = 1652
T0_NET_LO = 27.014327144622804
T0_NET_HI = 34.2478009223938
T0_NET_SD = 1.8270276267490637
T0_DESIGN_EFFECT = 9.443
T0_N_CLUSTERS_NET = 37
# E:/ff_scripts/local_probe.py 的 lineage 指纹（代码路径同一性证据）
LINEAGE_SHA = "9be4c23f88f4b5bc152ca383f428f0ece8c4ee7dab309e760de5a0aebb90820b"

# T3 预注册阈值（铁律 4：先于任何结果件冻结）
T3_RATIO_INDEFENSIBLE = 2.0
T3_RATIO_KEEP = 1.2


def assert_env():
    """坑 43：入口断言 python/numpy/scipy 版本（唯一实现在 scripts/r206_env_guard.py）。"""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import r206_env_guard
    r206_env_guard.assert_pinned_env()


def pin_threads():
    """把 torch 钉到单线程并断言成功（逐位确定性的执行环境限定，A-8 同族）。"""
    import torch
    torch.set_num_threads(1)
    if int(torch.get_num_threads()) != 1:
        raise SystemExit("W9 要求 torch 单线程（逐位确定性），当前=%s" % torch.get_num_threads())
    return {"torch": torch.__version__, "torch_threads": int(torch.get_num_threads()),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % type(o))


def atomic_write_json(path, obj):
    """严格 UTF-8 + allow_nan=False + tempfile/os.replace 原子写 + 写后回读自证。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False,
                     default=_json_default, allow_nan=False)
    if "\x00" in txt:
        raise SystemExit("W9 红线：产物含 NUL: %s" % path)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".w9tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    # 回读自证：可解析 + 无 NUL + 字节稳定
    raw = open(path, "rb").read()
    json.loads(raw.decode("utf-8"))
    if b"\x00" in raw:
        raise SystemExit("W9 红线：回读含 NUL: %s" % path)
    return len(raw), sha256_bytes(raw)


def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "\x00" in text:
        raise SystemExit("W9 红线：产物含 NUL: %s" % path)
    fd, tmp = tempfile.mkstemp(prefix=".w9tmp_", suffix=os.path.splitext(path)[1],
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    raw = open(path, "rb").read()
    raw.decode("utf-8")          # 严格 UTF-8（异常即抛）
    return len(raw), sha256_bytes(raw)


def load_r226():
    """把 r226 当**库**加载（不触发其 main()/不写其产物），复用其代码路径。"""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    spec = importlib.util.spec_from_file_location("w9_r226_ref",
                                                  os.path.join(ROOT, "scripts", "r226_anchor_305.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # 模块体只定义常量/函数，无副作用
    return mod


def load_baseline(m):
    """走 r226 同码路径装载 test 折，返回 (r226 模块, lp, te, agg, per_net)。"""
    from fractureflow.splits import load_split
    lp = m.load_original()
    _tr, _va, te = load_split(m.REAL, 42)
    agg, per_net = m.collect(lp, te)
    return lp, te, agg, per_net


def net_labels(te):
    """逐网标识：src / wid / widx（None 如实保留，不猜测）。"""
    rows = []
    for i, net in enumerate(te):
        wid = net.get("wid", None)
        widx = net.get("widx", None)
        rows.append({
            "net_idx": i,
            "src": str(net.get("src", "?")),
            "wid": wid if isinstance(wid, str) else None,
            "widx": int(widx) if isinstance(widx, (int, np.integer)) else None,
        })
    return rows


def primary_bin_arrays(per_net, lo=PRIMARY_BIN[0]):
    """按 r226 的簇顺序（test 折序）取主箱逐网角距数组。"""
    key = m_bkey(lo)
    out = []
    for rec in per_net:
        if key in rec["bins"]:
            out.append((rec["net_idx"], rec["bins"][key]["ang"]))
    return out


def m_bkey(x):
    return "%g" % float(x)


def ci_cluster_general(clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """通用簇 bootstrap —— 与 r226 `ci_cluster` 同一算法：
    同一 default_rng(seed)、每拍 rng.integers(0,k,k)、簇内全部对整块 concatenate、
    同一 .mean()、同一 quantile(0.025/0.975)、同一 std(ddof=1)。
    唯一差别 = 簇由调用方给定（单位可换），因此 unit=net 时必须逐位复现现役 CI。
    """
    k = len(clusters)
    if k == 0:
        raise SystemExit("W9: 空簇集")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        boot[i] = np.concatenate([clusters[j] for j in idx]).mean()
    return {
        "method": "cluster_bootstrap_percentile",
        "n_clusters": k,
        "n_boot": n_boot,
        "seed": seed,
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "halfwidth": float((np.quantile(boot, 0.975) - np.quantile(boot, 0.025)) / 2.0),
        "sd_of_boot_mean": float(boot.std(ddof=1)),
    }


def group_arrays(pairs, keyfn):
    """把 [(net_idx, ang)] 按 keyfn(net_idx) 归并成簇。

    顺序是**判据的一部分**（rng.integers 抽的是下标 ⇒ 簇序变则结果变），故显式固定为
    "按簇内最小 net_idx 升序"；这对 unit=net 恰与 r226 的 test 折序一致（W9-FID-2 逐位闸的前提），
    成员 concatenate 亦按 net_idx 升序（float32 求和顺序固定）。
    """
    from collections import defaultdict
    d = defaultdict(list)
    for nidx, arr in pairs:
        d[keyfn(nidx)].append((nidx, arr))
    ordered = sorted(d.items(), key=lambda kv: min(i for i, _ in kv[1]))
    keys = [kk for kk, _ in ordered]
    clusters = []
    for _, members in ordered:
        ms = [a for _, a in sorted(members, key=lambda t: t[0])]
        clusters.append(np.concatenate(ms) if len(ms) > 1 else ms[0])
    return keys, clusters


def kish_neff(counts):
    """Kish 有效样本量（对数加权）：N_eff = (Σn)^2 / Σn^2。"""
    c = np.asarray([float(x) for x in counts], dtype=np.float64)
    if c.size == 0 or c.sum() <= 0:
        return None
    return float(c.sum() ** 2 / (c ** 2).sum())
