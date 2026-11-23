# -*- coding: utf-8 -*-
"""W38-M1/M3 · 转换器约定的**运行时**判定 + 改前后数值同一性基线 (DR-4)。

为什么必须有这一步: 静态结构推断会把 attitude_to_sine(dip, az, R) 这种
"返回 (A,u,v) 正弦系数而非单位法向" 的函数误当 ENU 法向装配, 从而对无辜函数
提出传序指控 —— 那正是本项目反复踩的"叙述当制品"。本探针用**实际调用**回答:
  ① 它到底是不是法向生成器 (输出是否单位向量);
  ② 哪个实参槽被当成倾角 (nz 是否 == cos(该槽实参));
  ③ 无向铁律: n 与 -n 是否返回同一 (dip, dipdir);
  ④ 往返闭合: 产状->法向->产状->法向 偏差 (权威 geometry 4000 抽样);
  ⑤ 改前后同一调用的输出 sha256 (证明 M2 薄封装化零数值漂移)。

每个模块在**独立子进程**里加载 (部分 script 顶层有副作用), threads=1 钉死 (坑 45)。

用法:
  python scripts/w38_probe.py --phase baseline --out results/.../probe_baseline.json
  python scripts/w38_probe.py --phase post     --out results/.../probe_post.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SEP = "‖"  # 子进程协议分隔符

# 每槽给一个互不相同、cos 值也互不相同的角度, 使"nz=cos(第 i 槽)"可唯一归因
SLOT_ANGLES = [37.0, 123.0, 61.0, 13.0, 77.0, 51.0]
GRID = [(0.0, 0.0), (0.0, 90.0), (90.0, 0.0), (45.0, 90.0), (37.0, 123.0),
        (12.5, 311.0), (77.25, 5.5), (89.9, 179.99), (1.0, 359.0),
        (60.0, 270.0), (30.0, 45.0), (22.5, 135.0)]


def _sha(obj):
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(obj).hexdigest()[:16]
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(obj.astype(np.float64)).tobytes()).hexdigest()[:16]
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:16]


def _as_flat(v):
    arr = np.asarray(v, float)
    return [round(float(x), 12) for x in arr.ravel().tolist()], arr


# ---------------------------------------------------------------------------
# 子进程侧
# ---------------------------------------------------------------------------

def _import_target(path):
    """src/fractureflow/** 必须按**包**导入 (相对导入), 其余按文件路径独立加载。"""
    rp = os.path.relpath(path, _ROOT).replace("\\", "/")
    if rp.startswith("src/fractureflow/"):
        modname = "fractureflow." + rp[len("src/fractureflow/"):-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]
        if os.path.join(_ROOT, "src") not in sys.path:
            sys.path.insert(0, os.path.join(_ROOT, "src"))
        return importlib.import_module(modname)
    spec = importlib.util.spec_from_file_location("_w38_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_one(path, name, arity):
    mod = _import_target(path)
    fn = getattr(mod, name, None)
    if fn is None:
        return {"callable": False, "why": "attribute missing after exec"}

    out = {"callable": True, "arity_declared": arity}
    if arity >= 2:
        out.update(probe_forward(fn, arity))
    if arity == 1:
        out.update(probe_inverse(fn))
    return out


def probe_forward(fn, arity):
    """多参正向: 用可归因实参定出倾角槽, 并判输出是否单位法向。"""
    res = {}
    angles = SLOT_ANGLES[:arity]
    attempts = []
    for maker in ("scalar", "array1"):
        try:
            if maker == "scalar":
                args = list(angles)
            else:
                args = [np.array([a], float) for a in angles]
            v = fn(*args)
            flat, arr = _as_flat(v)
            attempts.append({"maker": maker, "shape": list(arr.shape), "flat": flat,
                             "norm": float(np.linalg.norm(arr))})
        except Exception as e:  # noqa: BLE001
            attempts.append({"maker": maker, "error": f"{type(e).__name__}: {e}"})
    ok = [a for a in attempts if "flat" in a]
    res["probe_forms"] = ok or attempts
    if not ok:
        res["callable"] = False
        res["why"] = "all call forms raised"
        return res
    f = ok[0]
    is_vec = (len(f["flat"]) == 3 and abs(f["norm"] - 1.0) < 1e-6)
    res["is_unit_vector"] = bool(is_vec)
    if is_vec:
        nz = abs(f["flat"][2])
        hits = [i for i, a in enumerate(angles)
                if abs(nz - abs(math.cos(math.radians(a)))) < 1e-9]
        res["dip_slot_empirical"] = (f"slot{hits[0]}" if len(hits) == 1 else
                                     ("AMBIGUOUS" if hits else "NEITHER"))
        res["dip_slot_indices"] = hits
    # 基线网格: 固定可复现调用 (改前后逐位比对用)
    grid = []
    for pair in GRID:
        try:
            args = list(pair) + list(angles[2:])
            if f["maker"] == "array1":
                args = [np.array([x], float) for x in args]
            flat, _ = _as_flat(fn(*args))
            grid.append(flat)
        except Exception as e:  # noqa: BLE001
            grid.append([f"{type(e).__name__}"])
    res["grid_vector"] = grid
    res["grid_sha"] = _sha(str(grid))
    return res


def probe_inverse(fn):
    """单参逆向: 无向铁律的正确测法 = g(n) 对比 g(-n) (不是对比原产状)。"""
    res = {}
    r, a = math.radians(37.0), math.radians(123.0)
    vec = [math.sin(r) * math.sin(a), math.sin(r) * math.cos(a), math.cos(r)]
    shapes = {"flat3": np.array(vec, float), "row13": np.array([vec], float)}
    chosen = None
    for tag, n_plus in shapes.items():
        try:
            f1, _ = _as_flat(fn(n_plus))
            f2, _ = _as_flat(fn(-n_plus))
            chosen = (tag, f1, f2)
            break
        except Exception:  # noqa: BLE001
            continue
    if chosen is None:
        return {"callable": False, "why": "no inverse input shape accepted (tried flat3/row13)"}
    tag, f1, f2 = chosen
    res["input_shape"] = tag
    res["inverse_sample"] = f1
    # 角->向量 映射 (输出是单位 3 向量) 不是逆向函数, 铁律判据不适用
    if len(f1) == 3 and abs(float(np.linalg.norm(f1)) - 1.0) < 1e-6:
        res["is_angle_to_vector_map"] = True
        res["iron_law_applicable"] = False
        res["why"] = "输出为单位 3 向量 => 角->向量映射, 非法向->产状逆变换"
        grid = []
        for dip, dd in GRID:
            try:
                inp = np.array([dd], float) if tag == "flat3" else np.array([[dd]], float)
                flat, _ = _as_flat(fn(inp))
                grid.append(flat)
            except Exception as e:  # noqa: BLE001
                grid.append([type(e).__name__])
        res["inverse_grid_vector"] = grid
        res["grid_sha"] = _sha(str(grid))
        res["callable"] = True
        return res
    res["is_angle_to_vector_map"] = False
    res["iron_law_applicable"] = True
    diffs = [abs(x - y) for x, y in zip(f1, f2)]
    if len(f1) >= 2:                                  # (dip, dipdir/strike) 形态
        diffs = [abs(f1[0] - f2[0]),
                 abs(min(abs(f1[1] - f2[1]), 360 - abs(f1[1] - f2[1])))]
    res["iron_law_g_n_vs_g_minus_n_max_dev_deg"] = float(max(diffs)) if diffs else None
    res["iron_law_holds_within_1e_minus_9"] = bool(diffs and max(diffs) <= 1e-9)
    res["iron_law_bitwise_exact"] = bool(f1 == f2)
    grid = []
    for dip, dd in GRID:
        rr, aa = math.radians(dip), math.radians(dd)
        v = [math.sin(rr) * math.sin(aa), math.sin(rr) * math.cos(aa), math.cos(rr)]
        n = np.array(v, float) if tag == "flat3" else np.array([v], float)
        try:
            flat, _ = _as_flat(fn(n))
            grid.append(flat)
        except Exception as e:  # noqa: BLE001
            grid.append([type(e).__name__])
    res["inverse_grid_vector"] = grid
    res["grid_sha"] = _sha(str(grid))
    res["callable"] = True
    return res


def child_main(argv):
    path, name, arity = argv[0], argv[1], int(argv[2])
    try:
        res = probe_one(path, name, arity)
    except Exception as e:  # noqa: BLE001
        res = {"callable": False, "why": f"import/exec {type(e).__name__}: {e}"}
    sys.stdout.write(SEP + json.dumps(res, ensure_ascii=False, allow_nan=False))
    return 0


# ---------------------------------------------------------------------------
# 主控侧
# ---------------------------------------------------------------------------

def load_targets(census_path):
    with open(census_path, "r", encoding="utf-8") as fh:
        c = json.load(fh)
    keep = ("UNARY_INVERSE", "DIP_FIRST", "DIPDIR_FIRST_OR_LATER", "INVERSE_SCALAR", "DELEGATE")
    return c, [d for d in c["definitions"] if d["inferred"]["kind"] in keep]


def run(path, name, arity, timeout=120):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
                "KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg"})
    try:
        p = subprocess.run([sys.executable, os.path.join(_HERE, "w38_probe.py"),
                            "--child", path, name, str(arity)],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=_ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return {"callable": False, "why": "TIMEOUT"}
    buf = p.stdout or ""
    if SEP in buf:
        try:
            return json.loads(buf.split(SEP, 1)[1])
        except json.JSONDecodeError as e:
            return {"callable": False, "why": f"bad child json: {e}"}
    return {"callable": False, "why": f"no marker rc={p.returncode} err={(p.stderr or '')[-200:]}"}


def authority_roundtrip(n=4000, seed=20261002, tol=1e-9):
    """M3 内核: 权威 geometry 的双向闭合 + 铁律 (偏差 > tol 即应 SystemExit)。"""
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from fractureflow.geometry import dip_dipdir_to_normal, normal_to_dip_dipdir
    rng = np.random.default_rng(seed)
    dip = rng.uniform(0.0, 90.0, n)
    dd = rng.uniform(0.0, 360.0, n)
    n1 = dip_dipdir_to_normal(dip, dd)
    d2, d2v = normal_to_dip_dipdir(n1)
    n2 = dip_dipdir_to_normal(d2, d2v)
    d3, dd3 = normal_to_dip_dipdir(-n1)
    dip_iron = float(np.abs(d2 - d3).max())
    dd_iron = float(np.minimum(np.abs(d2v - dd3), 360.0 - np.abs(d2v - dd3)).max())
    iron = bool(max(dip_iron, dd_iron) <= 1e-9)
    return {"n_samples": int(n), "threads": "1", "seed": seed, "tolerance": tol,
            "iron_law_dip_dev_deg": dip_iron, "iron_law_dipdir_dev_deg": dd_iron,
            "max_abs_component_dev": float(np.abs(n1 - n2).max()),
            "attitude_dev_deg_max": float(max(np.abs(d2 - dip).max(),
                                             np.abs(np.minimum(np.abs(d2v - dd),
                                                              360 - np.abs(d2v - dd))).max())),
            "iron_law_holds_within_1e_minus_9": iron,
            "iron_law_bitwise_exact": bool(np.array_equal(d2, d3) and np.array_equal(d2v, dd3))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="baseline", choices=["baseline", "post"])
    ap.add_argument("--census", default=os.path.join(_ROOT, "results", "w38_geometry_argorder", "census.json"))
    ap.add_argument("--out", default=None)
    ns = ap.parse_args(argv)

    sys.path.insert(0, _HERE)
    from r206_env_guard import assert_pinned_env
    assert_pinned_env("W38 probe")

    census, targets = load_targets(ns.census)
    rows, failed = [], []
    for d in targets:
        path = os.path.join(_ROOT, d["file"])
        if not os.path.isfile(path):
            failed.append({"file": d["file"], "name": d["name"], "why": "missing"})
            continue
        arity = d["arity"]
        if d["inferred"]["kind"] in ("DIP_FIRST", "DIPDIR_FIRST_OR_LATER") and arity < 2:
            arity = 2
        res = run(path, d["name"], arity)
        row = {"file": d["file"], "lineno": d["lineno"], "name": d["name"],
               "signature": d["params_positional"], "kwonly": d["params_kwonly"],
               "structural_kind": d["inferred"]["kind"],
               "structural_dip_slot": d["inferred"].get("dip_slot_index")}
        row.update(res)
        if (row.get("is_unit_vector") and row["structural_kind"] in
                ("DIP_FIRST", "DIPDIR_FIRST_OR_LATER")
                and str(row.get("dip_slot_empirical", "")).startswith("slot")):
            emp = int(row["dip_slot_empirical"][4:])
            struct = row.get("structural_dip_slot")
            row["dip_slot_numeric"] = emp
            row["structural_and_numeric_agree"] = bool(struct == emp)
            row["reversed_vs_geometry_signature"] = bool(emp != 0)
        rows.append(row)
        if not res.get("callable"):
            failed.append({"file": d["file"], "name": d["name"], "why": res.get("why", "?")})

    # DR-5: 空样本禁记 PASS
    if not rows:
        raise SystemExit("W38 probe: 探针样本为空, 拒绝出结论 (DR-5)")

    unit = [r for r in rows if r.get("is_unit_vector")]
    disagree = [r for r in rows if r.get("structural_and_numeric_agree") is False]
    out = ns.out or os.path.join(_ROOT, "results", "w38_geometry_argorder", f"probe_{ns.phase}.json")
    payload = {
        "phase": ns.phase,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": {"python": os.path.basename(sys.executable), "numpy": np.__version__},
        "scope": {"source_census": os.path.relpath(ns.census, _ROOT).replace("\\", "/"),
                  "n_definitions_total": census["counts"]["definitions"],
                  "n_targeted": len(targets), "n_probed": len(rows),
                  "n_unit_vector_producers": len(unit), "n_probe_failed": len(failed),
                  "enumeration": "子进程隔离逐文件 import + 实际调用"},
        "authority_roundtrip_geometry": authority_roundtrip(),
        "structural_vs_numeric_disagreements": [
            {"file": r["file"], "lineno": r["lineno"], "name": r["name"],
             "structural_kind": r["structural_kind"], "numeric_slot": r.get("dip_slot_numeric"),
             "is_unit_vector": r.get("is_unit_vector")} for r in disagree],
        "probes": rows,
        "probe_failures": failed,
        "baseline_sha_over_all_grids": _sha(sorted(
            f"{r['file']}:{r['name']}:{r.get('grid_sha', '-')}" for r in rows)),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        fh.write("\n")
    print(f"[w38-probe] phase={ns.phase} probed={len(rows)} unit={len(unit)} "
          f"failed={len(failed)} struct!=numeric={len(disagree)}")
    for r in disagree:
        print(f"  [DISAGREE] {r['file']}:{r['lineno']} {r['name']} struct={r['structural_kind']}"
              f" numeric_slot={r.get('dip_slot_numeric')} unit={r.get('is_unit_vector')}")
    print(f"[w38-probe] authority roundtrip dev={payload['authority_roundtrip_geometry']['max_abs_component_dev']:.3e}"
          f" iron_law_1e-9={payload['authority_roundtrip_geometry']['iron_law_holds_within_1e_minus_9']}"
          f" bitwise={payload['authority_roundtrip_geometry']['iron_law_bitwise_exact']}")
    print(f"[w38-probe] out -> {os.path.relpath(out, _ROOT).replace(chr(92), '/')}")
    return 0


if __name__ == "__main__":
    _a = sys.argv[1:]
    if _a and _a[0] == "--child":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        sys.exit(child_main(_a[1:]))
    sys.exit(main(_a))
