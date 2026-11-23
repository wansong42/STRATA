# -*- coding: utf-8 -*-
"""W38 · 产状<->法向 参数序与往返硬闸 (常驻把守, 单一实现)。

本件补的是 S32/S33 都没盖住的三面:
  A. **往返闭合 + 与权威全三分量一致** —— S32 只断言 nz 与单位性, 因此
     水平分量 sin/cos 互换 (十五期 `_nd` 事故类) 从 S32 眼底过去; 本件按
     三分量逐格比权威, 互换必红。
  B. **参数序台账** —— 每个现役正向生成器登记"倾角在第几槽"(运行时实测得出,
     非名字猜测), 现盘与台账不符即红; 新增未登记生成器即红 (构造上不可能
     再悄悄多一份手搓实现)。
  C. **逆向函数铁律分档** —— 无向铁律 (n 与 -n 同 (dip,dipdir)) 的当前实现状态
     逐件登记为 FOLDS / FLIPS_180 / ANGLE_TO_VECTOR; 任何一件**改档**即红
     (往好改也要走登记, 因为改档=改在册行为, 属重定价面)。

口径说明 (计数必带范围):
  - 正向台账面 = PROducers 表 (现役 10 件, 由 W38 普查+探针实测得出);
  - 未登记新实现扫描面 = src/**.py (递归) + scripts/*.py (仅顶层);
  - 本件不修任何数值路径 (§0 授权面), 只把守。

用法: python scripts/w38_geometry_gate.py [--enforce] [--json OUT]
退出码: 0=全绿, 1=有违规 (--enforce 时违规直接 SystemExit)。
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

TOL_COMP = 1e-9          # 任务书 M3 规定的三分量容差
N_RANDOM = 2000
SEED = 20261002

# --- 唯一权威 (任务书 M2) --------------------------------------------------
AUTH_FILE = "src/fractureflow/geometry.py"
AUTH_DEF = "dip_dipdir_to_normal"

# --- 正向生成器台账: (rel_file, func, 倾角槽,  arity, 出处) ----------------
# dip_slot 由 scripts/w38_probe.py 运行时实测 (probe_baseline.json#probes),
# 不是从名字或 docstring 猜的。slot=0 即与权威签名同序。
PRODUCERS = (
    ("src/fractureflow/geometry.py",              "dip_dipdir_to_normal",       0, 2),
    ("src/fractureflow/l4/evidence.py",           "dip_dipdir_to_normal",       0, 2),
    ("src/fractureflow/obs_forward.py",           "dip_dipdir_to_normal",       0, 2),
    ("src/fractureflow/site_model.py",            "dip_dipdir_to_normal",       0, 2),
    ("scripts/borehole_excel_entry.py",           "dip_dipdir_to_normal",       0, 2),
    ("scripts/w8_local_evidence.py",              "dip_dipdir_to_normal",       0, 2),
    ("scripts/w10_common.py",                     "attitude_pair_to_normal",    0, 2),
    ("scripts/read_forge_las.py",                 "dip_dipdir_to_normal",       1, 2),
    ("scripts/report_replay.py",                  "_dip_dipdir_to_normal",      1, 2),
    ("scripts/forge_fmi_pipeline.py",             "_nd",                        1, 2),
)

# --- 逆向函数铁律分档台账 --------------------------------------------------
# FOLDS     = n 与 -n 给同一 (dip, dipdir)  [满足无向铁律]
# FLIPS_180 = nz<0 时方位差 180°            [不满足; 现状登记, 改动须另单重定价]
# ANGLE_TO_VECTOR = 角->向量映射, 铁律不适用
#
# ⚠ 改档记录 (2026-10-02 · W43 `results/w43_product_dipole_convention/`):
#   本单按 M4 裁定 ③「只修产品链出口」把三条**产品出口**从 FLIPS_180 改判 FOLDS
#   (改为委托权威 geometry.normal_to_dip_dipdir)。改档=改在册行为, 故台账同单更新,
#   更新后本闸对**两个方向**仍有牙: 谁把出口改回自写 atan2 (无 +180) 即红。
#   未动的四条 (l4/evidence · fusion_pipeline · forge_fmi_pipeline · beishan_forge_pipeline)
#   维持 FLIPS_180 —— 它们不在 8 入口产品闭包内 (摄入侧/研究侧), 按 W42 裁定 ② 不批。
INVERSE_REGISTER = (
    ("src/fractureflow/geometry.py",        "normal_to_dip_dipdir",    "FOLDS"),
    ("src/fractureflow/obs_forward.py",     "normal_to_dip_dipdir",    "FOLDS"),
    ("src/fractureflow/site_model.py",      "normal_to_dip_dipdir",    "FOLDS"),
    ("src/fractureflow/l4/evidence.py",     "normal_to_dip_dipdir",    "FLIPS_180"),
    ("src/fractureflow/borehole_report.py", "normal_to_dip_dir",       "FOLDS"),   # W43 改档
    ("src/fractureflow/report.py",          "_dir_to_dip_azimuth",     "FOLDS"),   # W43 改档
    ("src/fractureflow/report.py",          "_dir_to_dip_azimuth_local", "FOLDS"), # W43 新增登记
    ("src/fractureflow/html_report.py",     "_dir_to_dip_azimuth",     "FLIPS_180"),
    ("scripts/fusion_pipeline.py",          "normal_to_dip_dipdir",    "FLIPS_180"),
    ("scripts/forge_fmi_pipeline.py",       "dip_dir_of",              "FOLDS"),   # W43 改档
    ("scripts/beishan_forge_pipeline.py",   "dip_dir_of",              "FLIPS_180"),
)

# --- 定义面白名单 (允许转换器定义存在的文件; 新增未登记文件即红) ------------
# 依据 = W38 普查 (census.json) 时点实测, 不是手写清单; **只准缩不准扩** ——
# 要新增转换器, 正确做法是复用 geometry 权威, 而不是往这里加一行。
DEF_ALLOW_SRC = {
    "src/fractureflow/geometry.py", "src/fractureflow/l4/evidence.py",
    "src/fractureflow/obs_forward.py", "src/fractureflow/site_model.py",
    "src/fractureflow/borehole_report.py", "src/fractureflow/report.py",
    "src/fractureflow/html_report.py",
}
DEF_ALLOW_SCRIPTS = {
    "scripts/beishan_forge_pipeline.py", "scripts/borehole_excel_entry.py",
    "scripts/check_challenge.py", "scripts/forge_fmi_pipeline.py",
    "scripts/fusion_pipeline.py", "scripts/literature_truth_compare.py",
    "scripts/make_challenge_dataset.py", "scripts/r132_censored_fit.py",
    "scripts/r149_orientation_precheck.py", "scripts/r152_catalog_adapter.py",
    "scripts/r152_focal_pipeline.py", "scripts/r161_gdr_pipeline.py",
    "scripts/read_forge.py", "scripts/read_forge_las.py", "scripts/report_replay.py",
    "scripts/v_r100_ui_evidence.py", "scripts/v_r96_fusion_reconcile.py",
    "scripts/v_r98_sensitivity.py", "scripts/w10_common.py",
    "scripts/w1_forward_sim.py", "scripts/w8_local_evidence.py",
}
DEF_ALLOW = DEF_ALLOW_SRC | DEF_ALLOW_SCRIPTS

DEF_NAME_RE = re.compile(
    r"(to_normal|to_unit|to_plane|normal_to|dipdir_to|dip_dir_to|strike_to"
    r"|orientation_to|attitude_to|attitude_pair|azimuth_.*_to|to_dipdir"
    r"|to_strike|to_azimuth|dip_dir_of|dir_to_dip|^_nd$)", re.I)

# 薄封装必须显式关键字转调 (P2 的攻击面): 登记为封装的文件里, 对权威的调用
# 一律要求 keyword-only 形态。

GRID = [(0.0, 0.0), (0.0, 90.0), (90.0, 0.0), (45.0, 90.0), (37.0, 123.0),
        (12.5, 311.0), (77.25, 5.5), (89.9, 179.99), (1.0, 359.0),
        (60.0, 270.0), (30.0, 45.0), (22.5, 135.0)]


_ABANDONED_WRAPPERS = []  # 见 load_rel: 必须持引用, 否则其 finalizer 会关掉共享 buffer


def load_rel(rel, tag):
    """src 下按包导入 (相对导入需要), scripts 下按文件路径导入。

    被检件不得拖垮检查器 (W38 实测抓到 scripts/w8_local_evidence.py:22 在 import 期
    `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`): 换流本身可恢复, 但**被弃用
    的旧 wrapper 一旦被 GC, 其 finalizer 会关掉与新流共享的 buffer** —— 在 pytest 里就
    是 "I/O operation on closed file", 整份报告消失。故先把弃用件挂进模块级列表保住引用,
    再恢复原流。该 import 期副作用本身按 R48 登记移交, 本件不代修。
    """
    if rel.startswith("src/fractureflow/"):
        modname = "fractureflow." + rel[len("src/fractureflow/"):-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[:-len(".__init__")]
        return importlib.import_module(modname)
    keep = (sys.stdout, sys.stderr)
    spec = importlib.util.spec_from_file_location("_w38g_" + tag, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        for cur, orig in ((sys.stdout, keep[0]), (sys.stderr, keep[1])):
            if cur is not orig:
                _ABANDONED_WRAPPERS.append(cur)   # 持引用 => 不触发 finalizer 关 buffer
        sys.stdout, sys.stderr = keep
    return mod


def _to_nx3(arr, n):
    """把生成器输出归一为 (n,3): 兼容 (n,3) / (3,n) / (3,) 三种形态。"""
    a = np.asarray(arr, float)
    if a.shape == (3,):
        a = a[None, :] if n == 1 else np.repeat(a[None, :], n, axis=0)
    if a.ndim != 2 or a.shape[1] != 3:
        if a.ndim == 2 and a.shape[0] == 3:
            a = a.T
        else:
            raise ValueError(f"无法归一为 (n,3): shape={np.asarray(arr).shape}")
    if a.shape[0] != n:
        raise ValueError(f"样本数不符: {a.shape[0]} != {n}")
    return a


def _call_forward_array(fn, dips, dds, slot, arity):
    args = [dds, dips] if slot == 1 else [dips, dds]
    if slot == 2:
        args = [dips, dds, dips]
    if arity == 3 and slot != 2:
        args = args + [np.ones_like(dips)]
    return _to_nx3(fn(*[np.asarray(x, float) for x in args]), len(dips))


# ---------------------------------------------------------------------------
# A · 与权威三分量一致 + 往返闭合
# ---------------------------------------------------------------------------

def check_producers():
    from fractureflow.geometry import dip_dipdir_to_normal as auth
    bad, checked = [], 0
    rng = np.random.default_rng(SEED)
    dips = np.concatenate([rng.uniform(0.0, 90.0, N_RANDOM),
                           np.array([g[0] for g in GRID], float)])
    dds = np.concatenate([rng.uniform(0.0, 360.0, N_RANDOM),
                          np.array([g[1] for g in GRID], float)])
    ref = np.asarray(auth(dips, dds), float)
    for rel, name, slot, arity in PRODUCERS:
        try:
            mod = load_rel(rel, os.path.basename(rel)[:-3])
            fn = getattr(mod, name)
            vec = _call_forward_array(fn, dips, dds, slot, arity)
            one = _call_forward_array(fn, dips[:1], dds[:1], slot, arity)
            if not np.allclose(one[0], vec[0], atol=0, rtol=0):
                bad.append({"file": rel, "name": name,
                            "why": "单点调用与批量调用结果不同 (形态/广播疑义)"})
        except Exception as e:  # noqa: BLE001
            bad.append({"file": rel, "name": name, "why": f"{type(e).__name__}: {e}"})
            continue
        checked += 1
        if vec.shape != ref.shape:
            bad.append({"file": rel, "name": name, "why": f"shape {vec.shape} != {ref.shape}"})
            continue
        dev = float(np.abs(vec - ref).max())
        if dev > TOL_COMP:
            i = int(np.abs(vec - ref).max(axis=1).argmax())
            bad.append({"file": rel, "name": name, "dip_slot": slot,
                        "max_abs_component_dev": dev, "tolerance": TOL_COMP,
                        "worst_input_deg": [float(dips[i]), float(dds[i])],
                        "got": [round(float(x), 9) for x in vec[i]],
                        "authority": [round(float(x), 9) for x in ref[i]],
                        "why": "与权威三分量不一致 (含 sin/cos 水平分量互换类)"})
    return checked, bad


def check_roundtrip():
    """产状->法向->产状->法向, 三分量偏差 > TOL_COMP 即违规。

    硬断言域 = dip in [1, 89] 度。理由写死在此, 不留给后人重新发现:
      dip = arccos|nz| 的条件数 = 1/sin(dip); 权威归一化带 +1e-12, 故 dip->0 时
      往返重构误差的解析下界 = 1e-12/sin(dip), 在 (0,1) 度内就已越过 1e-9。
      即"全域 + 固定绝对容差"数学上不可达, 那不是口径错而是浮点条件数
      (S32 同样避开 0/90 退化端点)。全域偏差另测一份作 diagnostic 随行报出,
      不进判据, 也不许被当成被掩盖的失败。
    """
    from fractureflow.geometry import dip_dipdir_to_normal as fwd, normal_to_dip_dipdir as inv
    rng = np.random.default_rng(SEED)
    grid_hard = [g for g in GRID if 1.0 <= g[0] <= 89.0]
    hard_dip = np.concatenate([rng.uniform(1.0, 89.0, N_RANDOM),
                               np.array([g[0] for g in grid_hard], float)])
    hard_dd = np.concatenate([rng.uniform(0.0, 360.0, N_RANDOM),
                              np.array([g[1] for g in grid_hard], float)])
    full_dip = np.concatenate([rng.uniform(0.0, 90.0, N_RANDOM),
                               np.array([g[0] for g in GRID], float)])
    full_dd = np.concatenate([rng.uniform(0.0, 360.0, N_RANDOM),
                              np.array([g[1] for g in GRID], float)])

    def _dev(dip, dd):
        n1 = fwd(dip, dd)
        d2, dd2 = inv(n1)
        n2 = fwd(d2, dd2)
        return n1, d2, dd2, float(np.abs(n1 - n2).max())

    n1h, d2h, dd2h, dev_hard = _dev(hard_dip, hard_dd)
    dev_full = _dev(full_dip, full_dd)[3]
    bound_hard = 1e-12 * 2.5 / math.sin(math.radians(1.0))
    d3, dd3 = inv(-n1h)
    dd_delta = np.minimum(np.abs(dd2h - dd3), 360.0 - np.abs(dd2h - dd3))
    iron = float(max(np.abs(d2h - d3).max(), dd_delta.max()))
    viol = []
    # W46 加严 (补盲区, 阈值不动): 边界带 |nz|==0 的两个**带符号零**形态必须同样过铁律。
    # 正向生成器产不出精确 0.0 (cos(90)=tiny), 所以旧探针覆盖不到这一类 ——
    # 而真实入库语料 (egs_collab 1425 行里的 334 行) 恰恰存的是 -0.0。
    bnd = np.array([[0.5, 0.8660254037844387, 0.0],
                    [0.5, 0.8660254037844387, -0.0],
                    [1.0, 0.0, -0.0], [0.0, 1.0, -0.0]])
    db_p, dd_p = inv(bnd)
    db_m, dd_m = inv(-bnd)
    bnd_dev = float(np.maximum(np.abs(db_p - db_m).max(),
                               np.minimum(np.abs(dd_p - dd_m),
                                           360.0 - np.abs(dd_p - dd_m)).max()))
    if bnd_dev > TOL_COMP:
        viol.append({"why": "边界带 (|nz|==0) 无向铁律偏差 %.3e > %g => 带符号零被漏" % (bnd_dev, TOL_COMP),
                     "max_dev": bnd_dev})
    if dev_hard > TOL_COMP:
        viol.append({"why": "硬域往返三分量偏差 %.3e > %g" % (dev_hard, TOL_COMP),
                     "max_dev": dev_hard})
    if iron > TOL_COMP:
        viol.append({"why": "权威无向铁律偏差 %.3e > %g" % (iron, TOL_COMP), "max_dev": iron})
    return {"n_samples_hard_domain": int(len(hard_dip)), "hard_domain_dip_deg": [1.0, 89.0],
            "n_samples_full_domain": int(len(full_dip)),
            "seed": SEED, "tolerance": TOL_COMP,
            "max_abs_component_dev": dev_hard,
            "max_abs_component_dev_full_domain": dev_full,
            "condition_explained_upper_bound_hard_domain": bound_hard,
            "margin_vs_tolerance": (TOL_COMP / dev_hard) if dev_hard > 0 else float("inf"),
            "iron_law_dev_deg": iron, "violations": viol,
            "boundary_band_iron_law_dev_deg": bnd_dev,
            "note": "full-domain 读数是诊断随行, 不进判据 (条件数解释见 docstring)"}


def check_inverse_register():
    from fractureflow.geometry import dip_dipdir_to_normal as fwd
    n_plus = fwd(np.array([37.0]), np.array([123.0]))
    forms = [("row13", n_plus, -n_plus), ("flat3", n_plus[0], -n_plus[0])]
    bad, checked = [], 0
    for rel, name, want in INVERSE_REGISTER:
        called = None
        try:
            mod = load_rel(rel, os.path.basename(rel)[:-3])
            fn = getattr(mod, name)
            for tag, pp, mm in forms:
                try:
                    f1 = np.ravel(np.asarray(fn(pp), float))
                    f2 = np.ravel(np.asarray(fn(mm), float))
                    called = (tag, f1, f2)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if called is None:
                raise ValueError("两种输入形态 (row13/flat3) 均不被接受")
        except Exception as e:  # noqa: BLE001
            bad.append({"file": rel, "name": name, "why": f"{type(e).__name__}: {e}"})
            continue
        tag, f1, f2 = called
        checked += 1
        if len(f1) == 3 and abs(float(np.linalg.norm(f1)) - 1.0) < 1e-6:
            got, dev = "ANGLE_TO_VECTOR", None
        else:
            k = len(f1)
            dev = float(abs(f1[0] - f2[0]))
            if k >= 2:
                dev = float(max(dev, float(np.minimum(abs(f1[1] - f2[1]),
                                                      360 - abs(f1[1] - f2[1])))))
            got = "FOLDS" if dev <= 1e-9 else ("FLIPS_180" if abs(dev - 180.0) < 1e-6 else "OTHER")
        if got != want:
            bad.append({"file": rel, "name": name, "registered": want, "measured": got,
                        "input_form": tag,
                        "g_n_vs_g_minus_n_dev_deg": dev,
                        "why": "逆向铁律分档漂移 => 改档属改在册行为, 须另单重定价"})
    return checked, bad


# ---------------------------------------------------------------------------
# B/C · 定义面与封装面 (AST)
# ---------------------------------------------------------------------------

def scan_definitions():
    """新增未登记的转换器定义 => 违规; 已登记但倾角槽变了 => 由 check_producers 抓。"""
    unregistered, total = [], 0
    roots = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py") and not fn.startswith("test_"):
                roots.append(os.path.join(dirpath, fn))
    sdir = os.path.join(ROOT, "scripts")
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".py"):
            roots.append(os.path.join(sdir, fn))
    for p in roots:
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        try:
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and DEF_NAME_RE.search(node.name):
                total += 1
                if rel not in DEF_ALLOW:
                    unregistered.append({"file": rel, "lineno": node.lineno, "name": node.name,
                                         "why": "src/递归+scripts/顶层扫描面内未登记的转换器定义"})
    return total, unregistered


AUTH_PARAM_NAMES = {"dip_deg", "dip_dir_deg"}
# (封装所在文件, 封装函数名) —— 每个登记封装的**函数体**必须是一次对权威的
# 显式关键字转调 (0 位置实参 + 关键字名恰为权威形参名)。P2 毒丸攻击的正是这一条。
WRAPPERS = (
    ("scripts/read_forge_las.py", "dip_dipdir_to_normal"),
    ("scripts/report_replay.py", "_dip_dipdir_to_normal"),
)


def check_wrapper_keyword_explicitness():
    """登记封装 => 体内必须有且仅有一次对权威的显式关键字转调。"""
    bad = []
    for rel, fname in WRAPPERS:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            bad.append({"file": rel, "fn": fname, "why": "登记封装文件不在盘"})
            continue
        tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == fname), None)
        if node is None:
            bad.append({"file": rel, "fn": fname, "why": "登记封装函数不在盘"})
            continue
        hits = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if not nm.endswith("dip_dipdir_to_normal"):
                continue
            if nm == fname:                      # 自引用不算转调
                continue
            hits.append((call, nm))
        if len(hits) != 1:
            bad.append({"file": rel, "fn": fname, "n_authority_calls": len(hits),
                        "why": "对权威转调次数 = %d (应为恰好 1)" % len(hits)})
            continue
        call, nm = hits[0]
        kw = sorted(k.arg for k in call.keywords if k.arg)
        if call.args:
            bad.append({"file": rel, "fn": fname, "lineno": call.lineno, "callee": nm,
                        "keyword_names": kw, "n_positional": len(call.args),
                        "why": "封装对权威位置直传 (应显式关键字转调) => 参数序错会静默通过"})
        elif set(kw) != AUTH_PARAM_NAMES:
            bad.append({"file": rel, "fn": fname, "lineno": call.lineno, "callee": nm,
                        "keyword_names": kw,
                        "why": "关键字名应恰为 %s" % sorted(AUTH_PARAM_NAMES)})
    return bad


def run_all():
    n_prod, prod_bad = check_producers()
    rt = check_roundtrip()
    n_inv, inv_bad = check_inverse_register()
    n_def, def_bad = scan_definitions()
    wrap_bad = check_wrapper_keyword_explicitness()
    if n_prod == 0 or n_inv == 0:
        raise SystemExit("W38 gate: 台账样本为空 => 拒绝判绿 (DR-5)")
    violations = (prod_bad + rt["violations"] + inv_bad + def_bad + wrap_bad)
    return {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tolerance_component": TOL_COMP,
        "scope": {"producers_registered": len(PRODUCERS), "producers_checked_ok": n_prod,
                  "inverses_registered": len(INVERSE_REGISTER), "inverses_checked_ok": n_inv,
                  "definition_scan_roots": "src/**.py 递归 + scripts/*.py 顶层",
                  "definitions_seen_in_scope": n_def,
                  "enumeration": "实际调用 + AST, 非文本 grep"},
        "roundtrip": rt,
        "producer_vs_authority_violations": prod_bad,
        "inverse_category_violations": inv_bad,
        "unregistered_definitions": def_bad,
        "wrapper_violations": wrap_bad,
        "n_violations": len(violations),
        "verdict": "PASS" if not violations else "FAIL",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--json", dest="out", default=None)
    ns = ap.parse_args(argv)
    rep = run_all()
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
            fh.write("\n")
    print(f"产状<->法向硬闸: {rep['verdict']} "
          f"(违规 {rep['n_violations']} / 生成器 {rep['scope']['producers_checked_ok']} "
          f"/ 逆向 {rep['scope']['inverses_checked_ok']} "
          f"/ 往返偏差 {rep['roundtrip']['max_abs_component_dev']:.3e}"
          f", 余量 {rep['roundtrip']['margin_vs_tolerance']:.2f}x)")
    for v in (rep["producer_vs_authority_violations"] + rep["inverse_category_violations"]
              + rep["unregistered_definitions"] + rep["wrapper_violations"]):
        print("  [红]", v.get("file", "-"), v.get("name", ""), v.get("why", ""))
    if ns.enforce and rep["n_violations"]:
        raise SystemExit(f"W38 gate: {rep['n_violations']} 处违规 (--enforce)")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
