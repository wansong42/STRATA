# -*- coding: utf-8 -*-
"""R156 T1/T2 · r152_catalog_adapter.py — 地震目录解析与几何适配层 (standalone).

职责（R156 任务书 §2 T1收尾+T2; R152 v1.1 §2 T1/T2 判据全文继承）:
  1. 解析 GCMT NDK (jan76_dec25.ndk) -> results/r152_focal_migration/gcmt_events.csv
  2. 解析 NCSN FPFIT .mech 月度文件       -> results/r152_focal_migration/ncsn_events.csv
  3. 几何适配: 两节面 -> 2 轴向法向 -> 与编录表同构的 (dip, dip_dir) 数组
  4. 往返一致性自检: 法向 -> 产状 -> 法向 逐位 (np.array_equal, bitwise)
   5. 矩张量重建数学验证: 从 NDK 全 MT 本征分解抽取 DC 分量 (eigenvalues
      [-a,0,+a]), 与解码出的 (strike,dip,rake) 重建的 DC 矩张量对比 —
      两者均为纯 DC, 相关性应饱和到 1 (节面解码正确性决定性检验)

红线:
  - standalone: 只 import numpy/pandas-free 标准库; 不 import/修改 src/ 冻结模块
  - 解析表 = 全量解析 (质量字段如实带下); 预注册过滤规则由 pipeline 执行 (R152 T1)
  - 只写 results/r152_focal_migration/ 下两个解析表; 不碰任何既有文件

几何约定 (与主项目编录表同构, ENU, 方位角自北顺时针):
  平面 (dip_dir α, dip δ):  n = (sinδ·sinα, sinδ·cosα, cosδ)   (↑半球法向)
  strike φ: α = φ + 90;  反解: δ = arccos(n_z), α = atan2(n_x, n_y)

冻结状态 (R156 收刀关账, 2026-09-04):
  - 冻结日期: 2026-09-04
  - 确定性双跑已证: results/r152_focal_migration/determinism_report.json
  - prereg sha256: 5cec2b24351de1ea0c37557bab55e71c9cacbdf80583a80d1516d0edc0a2ed21
  - 本域已收刀, 不再投入新单
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

RAW = os.path.join("results", "r152_focal_migration", "raw")
OUT_DIR = "results/r152_focal_migration"
NDK_PATH = os.path.join(RAW, "jan76_dec25.ndk")
MECH_GLOB = os.path.join(RAW, "ncsn_mech", "mechanism", "*", "*.mech")
GCMT_CSV = os.path.join(OUT_DIR, "gcmt_events.csv")
NCSN_CSV = os.path.join(OUT_DIR, "ncsn_events.csv")

PREREG_PATH = os.path.join(OUT_DIR, "prereg.json")


# ---------------------------------------------------------------------------
# 几何 (T2 适配层核心, standalone)
# ---------------------------------------------------------------------------
def orientation_to_normal(dip_dir_deg, dip_deg):
    """(dip_dir, dip) -> unit normal, ENU, azimuth from North clockwise."""
    a = np.radians(np.asarray(dip_dir_deg, dtype=np.float64))
    d = np.radians(np.asarray(dip_deg, dtype=np.float64))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def normal_to_orientation(n):
    """unit normal -> (dip_dir, dip). 逆变换, 与正向逐位互逆."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    dip = np.degrees(np.arccos(np.clip(n[..., 2], -1.0, 1.0)))
    dip_dir = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360.0
    return dip_dir, dip


def strike_to_normal(strike_deg, dip_deg):
    """(strike, dip) -> unit normal (strike 右手法则, dip_dir = strike + 90)."""
    return orientation_to_normal((np.asarray(strike_deg) + 90.0) % 360.0, dip_deg)


def roundtrip_bitwise(dd, dp, digits_dd, digits_dp):
    """S32 式往返铁判据 (含目录原生量化网格)。

    平面产状在目录中是有限小数网格上的值 (GCMT strike/dip: 2/0 位小数;
    NCSN dip_dir/dip: 0/0 位小数)。浮点三角正逆变换各损失 ~1e-13 度,
    直接逐位比较不可能; 正确的逐位判据 = 法向 -> 产状 -> 量化回目录网格
    (与源值全等) -> 重建法向 (与原法向 np.array_equal 逐位)。
    返回 (grid_ok, bitwise_ok, max_recover_err_deg, n)。
    """
    dd = np.asarray(dd, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    n = orientation_to_normal(dd, dp)
    dd2, dp2 = normal_to_orientation(n)
    err = float(np.max(np.abs(dd2 - dd))) if dd2.size else 0.0
    q = 10.0 ** digits_dd
    dd2q = np.round(dd2 * q) / q
    q2 = 10.0 ** digits_dp
    dp2q = np.round(dp2 * q2) / q2
    grid_ok = bool(np.all(dd2q == dd) and np.all(dp2q == dp))
    n2 = orientation_to_normal(dd2q, dp2q)
    bitwise_ok = bool(np.array_equal(n, n2))
    return grid_ok, bitwise_ok, err, dd.size


# ---------------------------------------------------------------------------
# GCMT NDK 解析
# ---------------------------------------------------------------------------
def _f(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def parse_ndk(path):
    """GCMT NDK: 每事件 5 行. 返回 (events, stats)."""
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().splitlines()
    n = len(lines)
    if n % 5 != 0:
        raise SystemExit(f"[FATAL] NDK lines {n} not multiple of 5")
    events = []
    anomalies = {"l1_numeric_fields": 0, "l5_token_count": 0,
                 "plane_invalid": 0, "tensor_short": 0,
                 "degenerate_dip0": 0}
    for b in range(n // 5):
        l1, l2, l3, l4, l5 = lines[5 * b:5 * b + 5]
        t1 = l1.split()
        # L1: id date time lat lon depth [mb] [MS] region-text...
        evid, date, tim = t1[0], t1[1], t1[2]
        lat_h = _f(t1[3]); lon_h = _f(t1[4]); dep_h = _f(t1[5])
        mb = ms = None
        region_txt = []
        for tok in t1[6:]:
            v = _f(tok)
            if v is not None and "e" not in tok.lower():
                if mb is None:
                    mb = v
                elif ms is None:
                    ms = v
                else:
                    region_txt.append(tok)
            else:
                region_txt.append(tok)
        cmt_name = l2.split()[0] if l2.split() else ""
        # L3: CENTROID: t err lat err lon err dep err type stamp
        t3 = l3.split()
        if t3[0] != "CENTROID:" or len(t3) < 9:
            anomalies["l1_numeric_fields"] += 1
        c_lat = _f(t3[3]); c_lon = _f(t3[5]); c_dep = _f(t3[7])
        c_type = t3[9] if len(t3) > 9 else ""
        # L4: exponent + 6 x (val err)
        t4 = l4.split()
        if len(t4) != 13:
            anomalies["tensor_short"] += 1
        expo = int(t4[0])
        comps_rtp_err = [_f(x) for x in t4[1:]]
        comps_rtp = comps_rtp_err[0::2]
        # L5: version + 16 numbers; planes = 最后 6 个
        t5 = l5.split()
        ver = t5[0]
        nums = [_f(x) for x in t5[1:]]
        if len(nums) != 16 or any(v is None for v in nums):
            anomalies["l5_token_count"] += 1
            continue
        s1, d1, r1, s2, d2, r2 = nums[10:16]
        if not all(_in(v) for v in (s1, d1, r1, s2, d2, r2)):
            anomalies["plane_invalid"] += 1
            continue
        # 退化定向 (dip=0: 水平面无 defined dip direction, 法向不存在非平凡解)
        # 见 prereg.json amendments[0]; 排除计数, 不入法向管线
        degenerate = (d1 == 0.0) or (d2 == 0.0)
        if degenerate:
            anomalies["degenerate_dip0"] += 1
        m0 = float(_m0_dyncm(comps_rtp, expo))
        v0 = nums[0]
        assert v0 is not None  # guarded by line 147 any() check
        m0_dc = float(v0) * (10.0 ** expo)  # V10 nums[0] = scalar moment of best DC
        dc_pct = (m0_dc / m0 * 100.0) if m0 > 0 else 0.0
        events.append({
            "evid": evid, "cmt_name": cmt_name, "date": date, "time": tim,
            "hyp_lat": lat_h, "hyp_lon": lon_h, "hyp_depth": dep_h,
            "mb": mb, "MS": ms, "region_txt": " ".join(region_txt).strip(),
            "c_lat": c_lat, "c_lon": c_lon, "c_depth": c_dep, "c_type": c_type,
            "exponent": expo, "Mrr": comps_rtp[0], "Mtt": comps_rtp[1],
            "Mpp": comps_rtp[2], "Mrt": comps_rtp[3], "Mrp": comps_rtp[4],
            "Mtp": comps_rtp[5], "M0_dyncm": m0,
            "dc_pct": dc_pct, "ndk_ver": ver,
            "strike1": s1, "dip1": d1, "rake1": r1,
            "strike2": s2, "dip2": d2, "rake2": r2,
        })
    return events, {"n_lines": n, "n_blocks": n // 5, "anomalies": anomalies}


def _in(v):
    return v is not None and math.isfinite(v)


def _m0_dyncm(comps_rtp, expo):
    """标量矩 (偏量部分): M0 = sqrt(sum Mij^2 / 2), 单位 dyn*cm."""
    mrr, mtt, mpp, mrt, mrp, mtp = [x if _in(x) else 0.0 for x in comps_rtp]
    s = mrr ** 2 + mtt ** 2 + mpp ** 2 + 2 * (mrt ** 2 + mrp ** 2 + mtp ** 2)
    return math.sqrt(s / 2.0) * (10.0 ** expo)


def mt_from_dc(strike, dip, rake, m0):
    """(strike,dip,rake) -> 满矩张量 3x3 (ENU), M0 标量.
    M = M0 (n s^T + s n^T); 滑移矢量 (Aki 约定, 挂墙相对下盘, λ=+90 逆冲=上坡):
      s = cos(l)*u_strike - sin(l)*u_dip.
    注意比较时需 ENU<->rtp 轴变换 (r=z, t=-N, p=E)."""
    phi = math.radians(strike)
    d = math.radians(dip)
    lam = math.radians(rake)
    a = phi + math.pi / 2.0  # dip direction
    u_strike = np.array([math.sin(phi), math.cos(phi), 0.0])
    u_dip = np.array([math.cos(d) * math.sin(a), math.cos(d) * math.cos(a), -math.sin(d)])
    s = math.cos(lam) * u_strike - math.sin(lam) * u_dip
    n = np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])
    M = m0 * (np.outer(n, s) + np.outer(s, n))
    return M


def conjugate_plane(strike, dip, rake):
    """双力偶共轭节面: 给定 (strike,dip,rake) -> (strike2,dip2,rake2).

    数学关系 (DC): n2 = s1 (节面1 滑移矢 = 节面2 法向), s2 = n1,
    轴向规范化 (n_z>=0) 后反解产状; 返回 (strike2, dip2, rake2, degenerate).
    degenerate = dip2==0 (直立纯倾滑的共轭为水平面, 同 amendment#1 排除)。
    """
    n1 = strike_to_normal(strike, dip)
    d = math.radians(dip)
    lam = math.radians(rake)
    phi = math.radians(strike)
    a = phi + math.pi / 2.0
    u_strike = np.array([math.sin(phi), math.cos(phi), 0.0])
    u_dip = np.array([math.cos(d) * math.sin(a), math.cos(d) * math.cos(a), -math.sin(d)])
    s1 = math.cos(lam) * u_strike - math.sin(lam) * u_dip
    n2 = s1
    if n2[2] < 0.0:
        n2 = -n2
        s2 = -n1
    else:
        s2 = n1
    dd2, dp2 = normal_to_orientation(n2[None, :])
    dd2 = float(dd2[0]); dp2 = float(dp2[0])
    strike2 = (dd2 - 90.0) % 360.0
    a2 = math.radians(dd2)
    d2 = math.radians(dp2)
    u_st2 = np.array([math.sin(strike2 * math.pi / 180.0),
                      math.cos(strike2 * math.pi / 180.0), 0.0])
    u_dp2 = np.array([math.cos(d2) * math.sin(a2), math.cos(d2) * math.cos(a2),
                      -math.sin(d2)])
    lam2 = math.degrees(math.atan2(-float(s2 @ u_dp2), float(s2 @ u_st2)))
    degenerate = (dp2 <= 0.0)
    return strike2, dp2, lam2, degenerate


def mt_enu_to_rtp(M):
    """ENU 矩张量 -> GCMT (r,t,p)=(up, south, east) 分量序.
    r=z, t=-y, p=x:  Mrr=Mzz, Mtt=Myy, Mpp=Mxx, Mrt=-Mzy, Mrp=Mzx, Mtp=-Myx."""
    return np.array([M[2, 2], M[1, 1], M[0, 0], -M[2, 1], M[2, 0], -M[1, 0]])


def mt_ndk_to_enu(mrr, mtt, mpp, mrt, mrp, mtp):
    """NDK (r,t,p) -> ENU 对称张量."""
    M = np.array([
        [mpp, -mtp, mrp],   # x row: xx=pp, xy=-tp, xz=rp
        [-mtp, mtt, -mrt],  # y row
        [mrp, -mrt, mrr],   # z row
    ])
    return M


# ---------------------------------------------------------------------------
# NCSN FPFIT .mech 解析 (官方列定义: ncsn.mech.txt)
# ---------------------------------------------------------------------------
def _col(line, a, b, cast, default=None):
    """切片转数值; 空白/异常返回 default (可选字段容错, 老年代空白列不整行丢弃)."""
    tok = line[a:b]
    try:
        return cast(tok)
    except (TypeError, ValueError):
        return default


def parse_mech_file(path):
    """FPFIT mech 记号化/正则解析 (格式漂移免疫)。

    依据: 官方列宽文档 (ncsn.mech.txt) 与实际文件存在 ±1 列漂移, 固定切片
    不可靠 (实测 '*' 行 evid 混入星号导致配对崩坏)。改用:
      - 头部 8 记号: date hhmm sec latdeg latmin londeg lonmin depth (token)
      - 平面块正则 (在 line[80:] 搜索): magcode? dipdir(1-3位) dip(2位)rake(可负,可粘连) misfit nfm
      - 尾部正则: conv/multi 旗标 + evid
    核心字段不匹配 -> bad_core 整行弃; 辅助字段 (mag/rms) 尽力而为, None 不影响判据
    (预注册过滤只用 misfit/nfm/conv/dip/region, 均来自正则捕获组)。
    """
    out = []
    bad_core = 0
    head_re = re.compile(
        r"^(\d{8})\s+(\d{4})\s+([\d.]+)\s+(\d{1,3})\s+([\d.]+)\s+"
        r"(\d{1,3})\s+([\d.]+)\s+([\d.]+)")
    plane_re = re.compile(
        r"([A-Z]?)\s*(\d{1,3})\s+(\d{1,2})\s*(-?\d{1,3})\s+(\d\.\d{2})\s+(\d{1,3})")
    tail_re = re.compile(r"([*C]{0,2})\s*(\d{3,10})\s*$")
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_head = head_re.match(line)
            if not m_head:
                bad_core += 1
                continue
            m_plane = plane_re.search(line[80:])
            m_tail = tail_re.search(line.rstrip())
            if not m_plane or not m_tail:
                bad_core += 1
                continue
            try:
                rec = {
                    "date_raw": m_head.group(1), "hhmm": m_head.group(2),
                    "sec": float(m_head.group(3)),
                    "lat": float(m_head.group(4)) + float(m_head.group(5)) / 60.0,
                    "lon": -(float(m_head.group(6)) + float(m_head.group(7)) / 60.0),
                    "depth": float(m_head.group(8)),
                    "dip_dir": float(m_plane.group(2)),
                    "dip": float(m_plane.group(3)),
                    "rake": float(m_plane.group(4)),
                    "misfit": float(m_plane.group(5)),
                    "nfm": float(m_plane.group(6)),
                    "flags": m_tail.group(1),
                    "evid": m_tail.group(2),
                }
            except ValueError:
                bad_core += 1
                continue
            if not (0.0 <= rec["dip_dir"] < 360.0 and 0.0 <= rec["dip"] <= 90.0
                    and -180.0 <= rec["rake"] <= 180.0
                    and 0.0 <= rec["misfit"] <= 0.5):
                bad_core += 1
                continue
            # 头部辅助字段尽力而为 (mag/rms; 过滤不依赖)
            toks = line.split()
            rec["mag"] = _f(toks[8]) if len(toks) > 8 else None
            rec["rms"] = _f(toks[12]) if len(toks) > 12 else None
            rec["conv"] = "C" if "C" in rec["flags"] else ""
            rec["multi"] = "*" if "*" in rec["flags"] else ""
            rec["time"] = f"{rec['hhmm'][:2]}:{rec['hhmm'][2:]}:{rec['sec']:05.2f}"
            rec["date"] = f"{rec['date_raw'][:4]}/{rec['date_raw'][4:6]}/{rec['date_raw'][6:8]}"
            rec["src_file"] = os.path.basename(path)
            out.append(rec)
    return out, bad_core, 0


def pair_mech_events(recs):
    """NCSN 事件化 (amendment #2 数据模型):

    一行 = 一个完整 FPFIT 解 (优选节面 + rake, 第二节点面可导)。
    '*' 行 = FPFIT 多重解备选 (实测数值验证: 非共轭面), 只计数不入事件。
    同 evid 多条非 '*' 行: 取首条, 计数异常。
    plane2 = conjugate_plane(strike1, dip1, rake1) (数学精确)。
    """
    events = []
    stats = {"n_lines": len(recs), "n_events": 0, "n_alt_lines": 0,
             "n_multi_preferred": 0, "n_star_only": 0, "n_conjugate_degenerate": 0}
    order = []
    groups = {}
    for g in recs:
        key = g["evid"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(g)
    for key in order:
        group = groups[key]
        pref = [g for g in group if g["multi"] != "*"]
        alts = [g for g in group if g["multi"] == "*"]
        stats["n_alt_lines"] += len(alts)
        if not pref:
            stats["n_star_only"] += 1
            continue
        if len(pref) > 1:
            stats["n_multi_preferred"] += 1
        p1 = pref[0]
        s2, d2, r2, degen = conjugate_plane(
            (p1["dip_dir"] - 90.0) % 360.0, p1["dip"], p1["rake"])
        if degen:
            stats["n_conjugate_degenerate"] += 1
        h = p1
        events.append({
            "evid": h["evid"], "date": h["date"], "time": h["time"],
            "lat": h["lat"], "lon": h["lon"], "depth_km": h["depth"],
            "mag": h["mag"], "rms": h["rms"],
            "misfit1": h["misfit"], "misfit2": None,
            "nfm1": h["nfm"], "nfm2": None,
            "conv1": h["conv"], "conv2": "",
            "n_alt": len(alts),
            "strike1": (h["dip_dir"] - 90.0) % 360.0, "dip1": h["dip"],
            "rake1": h["rake"],
            "strike2": s2, "dip2": d2, "rake2": r2,
            "src_file": h["src_file"],
        })
        stats["n_events"] += 1
    return events, stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- GCMT NDK ----
    g_events, g_stats = parse_ndk(NDK_PATH)
    print(f"[ndk] blocks={g_stats['n_blocks']} events={len(g_events)} "
          f"anomalies={g_stats['anomalies']}")

    # ---- NCSN mech ----
    mech_files = sorted(glob.glob(MECH_GLOB))
    n_events = []
    bad_core_total = 0
    seen_evid = set()
    n_dup_evid = 0
    file_stats = {"n_lines": 0, "n_alt_lines": 0, "n_multi_preferred": 0,
                  "n_star_only": 0, "n_conjugate_degenerate": 0}
    for p in mech_files:
        recs, bad, auxn = parse_mech_file(p)
        evs, st = pair_mech_events(recs)
        bad_core_total += bad
        for k in ("n_lines", "n_alt_lines", "n_multi_preferred",
                  "n_star_only", "n_conjugate_degenerate"):
            file_stats[k] += st[k]
        for e in evs:
            if e["evid"] in seen_evid:
                n_dup_evid += 1
                continue
            seen_evid.add(e["evid"])
            n_events.append(e)
    n_stats = dict(file_stats)
    n_stats["n_events"] = len(n_events)
    n_stats["n_dup_evid_dropped"] = n_dup_evid
    print(f"[mech] files={len(mech_files)} lines={n_stats['n_lines']} "
          f"events={n_stats['n_events']} alt_lines={n_stats['n_alt_lines']} "
          f"multi_pref={n_stats['n_multi_preferred']} "
          f"star_only={n_stats['n_star_only']} "
          f"conj_degenerate={n_stats['n_conjugate_degenerate']} "
          f"dup_dropped={n_dup_evid} bad_core={bad_core_total}")

    # ---- 自检 1: 法向->产状->法向 逐位往返 (T2 铁判据, 含目录量化网格) ----
    # 退化平面 (dip=0) 不参与 (prereg amendments[0]); GCMT strike/dip 网格 2/0 位小数;
    # NCSN dip_dir/dip 网格 0/0 位小数
    def _planes_dd_dp(ev_list, strike_key):
        dds, dps = [], []
        for e in ev_list:
            for i in (1, 2):
                if e[f"dip{i}"] <= 0.0:
                    continue
                dds.append((e[f"strike{i}"] + 90.0) % 360.0
                           if strike_key else e[f"strike{i}"])
                dps.append(e[f"dip{i}"])
        return np.array(dds), np.array(dps)

    gcmt_dd, gcmt_dp = _planes_dd_dp(g_events, True)
    # NCSN 逐位判据只用于节面 1 (目录整数格值); 共轭面是计算值, 走数值容差判据
    ncsn_dd = np.array([(e["strike1"] + 90.0) % 360.0 for e in n_events
                        if e["dip1"] > 0.0])
    ncsn_dp = np.array([e["dip1"] for e in n_events if e["dip1"] > 0.0])
    # 共轭面 (计算值, 不在目录网格上, amendment#2): 数值往返容差判据
    ncsn_dd2 = np.array([(e["strike2"] + 90.0) % 360.0 for e in n_events
                         if e["dip2"] > 0.0])
    ncsn_dp2 = np.array([e["dip2"] for e in n_events if e["dip2"] > 0.0])
    edge_dd = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 315.0])
    edge_dp = np.array([0.0, 30.0, 90.0, 60.0, 45.0, 80.0, 15.0])
    ok_g = roundtrip_bitwise(gcmt_dd, gcmt_dp, 2, 0)
    ok_n = roundtrip_bitwise(ncsn_dd, ncsn_dp, 0, 0)
    ok_e = roundtrip_bitwise(edge_dd, edge_dp, 0, 0)
    # 共轭面: 法向 -> 产状 -> 法向, 逐分量最大偏差 (浮点三角不可逐位, 判据 1e-12)
    n2c = orientation_to_normal(ncsn_dd2, ncsn_dp2)
    dd2r, dp2r = normal_to_orientation(n2c)
    n2r = orientation_to_normal(dd2r, dp2r)
    conj_dev = float(np.max(np.abs(n2c - n2r))) if n2c.size else 0.0
    ok_n2 = conj_dev <= 1e-12
    for name, r in (("gcmt", ok_g), ("ncsn_p1", ok_n), ("edge", ok_e)):
        print(f"[roundtrip] {name}: grid={r[0]} bitwise={r[1]} "
              f"max_recover_err={r[2]:.3e} deg n={r[3]}")
    print(f"[roundtrip] ncsn_conj: numeric_dev={conj_dev:.3e} (<=1e-12: {ok_n2}) "
          f"n={ncsn_dd2.size}")
    if not (ok_g[0] and ok_g[1] and ok_n[0] and ok_n[1]
            and ok_e[0] and ok_e[1] and ok_n2):
        print("[FATAL] round-trip grid/bitwise check FAILED")
        return 3

    # ---- 自检 2: MT 重建数学验证 (节面解码决定性检验) ----
    # GCMT 满矩张量含 CLVD, corr 随 dc_pct (NDK line5 字段3, 双力偶百分比)
    # 单调上升并在纯 DC 端饱和到 1 (全表入 selfcheck JSON 作审计证据);
    # 闸门锚定纯 DC 子集 dc_pct∈[98,100.01]: median>=0.999 且 min>=0.995。
    rows = []
    for e in g_events:
        if e.get("dc_pct") is None or e["M0_dyncm"] <= 0:
            continue
        # Rebuild DC MT from decoded (strike,dip,rake) — unit scalar moment
        M_dc_rebuilt = mt_from_dc(e["strike1"], e["dip1"], e["rake1"], 1.0)
        dc_rebuilt_rtp = mt_enu_to_rtp(M_dc_rebuilt)
        # Extract DC part of observed full MT via eigenvalue decomposition
        # Full MT eigenvalues: λ1 ≥ λ2 ≥ λ3; DC part = a*(u1 u1^T - u3 u3^T), a=(λ1-λ3)/2
        M_obs = mt_ndk_to_enu(e["Mrr"], e["Mtt"], e["Mpp"],
                              e["Mrt"], e["Mrp"], e["Mtp"])
        eigvals, eigvecs = np.linalg.eigh(M_obs)
        order = np.argsort(eigvals)[::-1]  # descending
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        a_dc = (eigvals[0] - eigvals[2]) / 2.0  # DC scalar moment (eigenvalue)
        if a_dc <= 0:
            continue
        M_dc_obs = a_dc * (np.outer(eigvecs[:, 0], eigvecs[:, 0])
                           - np.outer(eigvecs[:, 2], eigvecs[:, 2]))
        dc_obs_rtp = mt_enu_to_rtp(M_dc_obs)
        no = np.linalg.norm(dc_obs_rtp)
        nd = np.linalg.norm(dc_rebuilt_rtp)
        if no <= 0 or nd <= 0:
            continue
        rows.append((e["dc_pct"], float(np.dot(dc_obs_rtp, dc_rebuilt_rtp) / (no * nd))))
    rows.sort()
    be_all = np.array([r[0] for r in rows])
    co_all = np.array([r[1] for r in rows])
    bin_table = []
    for lo, hi in [(50, 75), (75, 90), (90, 95), (95, 98), (98, 99),
                   (99, 99.5), (99.5, 100.01)]:
        m = (be_all >= lo) & (be_all < hi)
        if m.sum() == 0:
            continue
        bin_table.append({"dc_pct_bin": [lo, hi], "n": int(m.sum()),
                          "median_corr": float(np.median(co_all[m])),
                          "min_corr": float(co_all[m].min())})
    m = (be_all >= 98.0) & (be_all <= 100.01)
    cors = co_all[m]
    med = float(np.median(cors)) if cors.size else 0.0
    mn = float(cors.min()) if cors.size else 0.0
    print(f"[mt-check] n_total={len(rows)} pure_dc_subset(n={cors.size}) "
          f"median={med:.5f} min={mn:.5f}")
    for b in bin_table:
        print(f"[mt-check]   dc_pct{b['dc_pct_bin']}: n={b['n']} "
              f"median={b['median_corr']:.5f} min={b['min_corr']:.5f}")
    if not (cors.size >= 1000 and med >= 0.999 and mn >= 0.995):
        print("[FATAL] MT reconstruction validation failed -> plane decode wrong")
        return 4

    # ---- 落盘解析表 (全量, 质量字段如实带下; 过滤在 pipeline 按 prereg 执行) ----
    g_fields = ["evid", "cmt_name", "date", "time", "hyp_lat", "hyp_lon", "hyp_depth",
                "mb", "MS", "region_txt", "c_lat", "c_lon", "c_depth", "c_type",
                "exponent", "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp", "M0_dyncm",
                "dc_pct", "ndk_ver",
                "strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]
    write_csv(GCMT_CSV, g_fields, g_events)
    print(f"[save] {GCMT_CSV} rows={len(g_events)}")

    n_fields = ["evid", "date", "time", "lat", "lon", "depth_km", "mag", "rms",
                "misfit1", "misfit2", "nfm1", "nfm2", "conv1", "conv2", "n_alt",
                "strike1", "dip1", "rake1", "strike2", "dip2", "rake2", "src_file"]
    write_csv(NCSN_CSV, n_fields, n_events)
    print(f"[save] {NCSN_CSV} rows={len(n_events)}")

    # ---- 自检元数据 (stdout 汇总, 供 verdict 引用; 无时间戳, 确定性输出) ----
    summary = {
        "gcmt": {"events": len(g_events), "anomalies": g_stats["anomalies"]},
        "ncsn": {"files": len(mech_files), "stats": n_stats,
                 "bad_core": bad_core_total},
        "roundtrip": {
            "gcmt": {"grid": ok_g[0], "bitwise": ok_g[1],
                     "max_recover_err_deg": ok_g[2], "n": ok_g[3]},
            "ncsn_plane1": {"grid": ok_n[0], "bitwise": ok_n[1],
                            "max_recover_err_deg": ok_n[2], "n": ok_n[3]},
            "ncsn_conjugate": {"numeric_max_dev": conj_dev, "tol": 1e-12,
                               "pass": ok_n2, "n": int(ncsn_dd2.size)},
            "edge": {"grid": ok_e[0], "bitwise": ok_e[1],
                     "max_recover_err_deg": ok_e[2], "n": ok_e[3]},
        },
        "mt_check": {
            "method": "DC part of observed full MT extracted via eigenvalue decomposition (eigenvalues [-a,0,+a], a=(l_max-l_min)/2); compared with mt_from_dc(strike,dip,rake) rebuilt DC MT — both pure DC, correlation saturates at 1 if plane decode is correct",
            "gate": "pure-DC subset dc_pct in [98,100.01]: median>=0.999 and min>=0.995",
            "bins": bin_table,
            "pure_dc": {"n": int(cors.size), "median_corr": med, "min_corr": mn,
                        "pass": bool(cors.size >= 1000 and med >= 0.999 and mn >= 0.995)},
        },
    }
    with open(os.path.join(OUT_DIR, "adapter_selfcheck.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("[save] adapter_selfcheck.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
