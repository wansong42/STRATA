# -*- coding: utf-8 -*-
"""W26: 决策报告组装器 v1 —— 带不确定度的场景判定 + 观测建议 (研究线原型).

⚠️ 水印: 研究线原型, 禁入对外承诺文本 (裁定 58 对外零量化承诺继续有效)。

定位 (任务书 results/w26_decision_report_prototype/task_spec_v0.md):
  把在册散件 (组系表 / P32+渗流 / T46 置信校准 / L4 证据分级) 组装成
  "地质师在不确定度受控下敢做决策" 的单文件报告:
    输入井编录 (CSV/LAS/npz/pt) -> 组系表 -> P32/渗流 -> 场景判定
    -> 证据分级 -> 观测建议 (反事实可红) -> Markdown + JSON。

本模块是**组装器**, 不是新算法: 每一步都调用冻结 API, 数字一律 API 直出,
禁手抄 (R0 由 scripts/w26_r0_replay.py 独立对账)。唯一新增逻辑是:
  (a) 证据分级判定 (L4 表语义的数据档位判定, 非精度测量);
  (b) 不确定度带 = bootstrap_cone_95 × 在册 alpha (T46 校准链, 禁裸分位数);
  (c) 观测建议引擎 v0 = 解析触发谓词 + 反事实电池 (CF-DIP / CF-COV / CF-LOO)。

冻结 API 消费清单 (与 prereg.json 一致):
  装载: site_model.load_wells_from_npz / load_wells_from_dir / load_well_csv,
        l4.evidence.from_forge_fmi_pt, auto_label_borehole.read_borehole_csv
  组系表: dfn.auto_select_K (Krange=(2,8), seed=42) + dfn.set_table_from_normals
  一致性: site_model.joint_rule_partition (仅建议层)
  P32/渗流: dfn.estimate_p32_interval + percolation.percolation_curve (两阶段)
  场景: percolation.egs_connectivity_metric / mine_risk_sections /
        disposal_escape_priority (在 p32_crit 处, dfn_from_borehole [4/6] 同构)
  校准: l4.fuse.bootstrap_confidence_cone + results/l4_calibration.json 在册 alpha

诚实边界:
  - 筛查级结论: Baecher 盘 + 假设尺寸分布, 非真实连通性判定;
  - 分级是数据档位判定, 不是精度测量 (逐点误差受信息墙 ~31.7°±0.5° 约束);
  - 场景判定的工程轴 (井对连线/巷道轴) 未给时按默认 [1,0,0] 并显式声明;
  - alpha 借用规则: beishan/forge 用在册 per-site alpha, 其余场地用
    alpha_global=17.6267 并标注跨场地借用 (T46 回测面外的应用, 边界句披露)。

用法:
    python scripts/w26_decision_report.py --input data/real/beishan_wells.npz \
        --site-name beishan22 --out-dir results/w26_decision_report_prototype/beishan22
"""

import os
import sys
import time
import hashlib

import numpy as np

from .dfn import (auto_select_K, set_table_from_normals, estimate_p32_interval,
                  generate_dfn)
from .percolation import (percolation_curve, egs_connectivity_metric,
                          mine_risk_sections, disposal_escape_priority)
from .site_model import load_wells_from_npz, load_wells_from_dir, load_well_csv
from .l4.evidence import from_forge_fmi_pt, normal_to_dip_dipdir
from .l4.fuse import bootstrap_confidence_cone, estimate_kappa
from .site_model import joint_rule_partition

# ---------------------------------------------------------------------------
# 在册常量与在册文件读数 (禁手抄: 一律运行时直读在册 JSON)
# ---------------------------------------------------------------------------

WATERMARK = "研究线原型, 禁入对外承诺文本 (裁定 58: 对外零量化承诺)"

# 在册校准锚 (results/l4_calibration.json, T46): 运行时直读, 不硬编码数值
_CALIB_RELPATH = ("results", "l4_calibration.json")
# dip-only 上界 (results/dip_only_upper_bound.json): 运行时直读
_DIPONLY_RELPATH = ("results", "dip_only_upper_bound.json")

# 两阶段渗流网格与 dfn_from_borehole.py [3/6] 冻结模式一致
_COARSE_GRID = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
_COARSE_SEEDS = 5       # 阶段 A 粗定位实现数
_FINE_SEEDS = 20        # 阶段 B 细网格实现数 (H6 合规 >=20)
_BETA_DEFAULT = 3.5     # 编录表无迹长数据时的假设 β (进 assumptions)
_SEED = 42              # 全链主种子 (K 选择 / 组系表 / 场景 DFN)

# 观测建议触发阈值 (prereg.json#recommendation_engine_v0 预注册)
_TH_BAND_OVERLAP = None   # 区间重叠为解析谓词, 无自由参数
_TH_KEYWELL_CENTER_DISP = 10.0   # REC-3 组心位移阈值 (度)
_TH_COVCONE_RATIO = 1.5          # REC-5 锥带宽度比阈值
_TH_SCENARIO_EDGE = 10.0         # REC-6 阈值边缘带 (度)
_OBS_MIN_GRADE_C = 20            # L4 表 C/D 档分界 (单井观测数)


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_registered_json(*relpath):
    """直读在册 results JSON (禁手抄数字的制度化: 数值一律运行时读)。"""
    path = os.path.join(_project_root(), *relpath)
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_calibration():
    """T46 在册校准锚链 (alpha + coverage 90.6%)。"""
    d = _read_registered_json(*_CALIB_RELPATH)
    return {
        "alpha_per_site": d["kappa_scaling_alpha_per_site"],
        "alpha_global": d["kappa_scaling_alpha_global"],
        "recalibrated_coverage": d["recalibrated_coverage"],
        "K": d["K"],
        "source": "results/" + "/".join(_CALIB_RELPATH[1:]),
    }


def _load_diponly_bounds():
    d = _read_registered_json(*_DIPONLY_RELPATH)
    e = d["expected_error_on_beishan"]
    return {"mean": e["mean"], "median": e["median"], "p90": e["p90"],
            "source": "results/" + "/".join(_DIPONLY_RELPATH[1:])}


def _sha12(obj):
    """对象规范序列化后 sha12 (R0 对账指纹)。"""
    blob = repr(np.asarray(obj, dtype=float).tolist()).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 装载层 (冻结 loader, 不重写)
# ---------------------------------------------------------------------------

def load_site(source, site_name=None):
    """装载场地输入 -> 统一井列表。

    支持: .npz (beishan 井法向) / CSV 目录或单 CSV (编录表) /
          .pt (FORGE FMI, ftype 拆分条目)。
    返回 dict: {site_name, kind, wells, n_total_obs}
      wells: [{well_id, nrm, has_azimuth, n_obs, source_type, origin}]
      has_azimuth=False 表示 dip-only (npz 法向无方位缺失概念, 恒 True;
      dip-only 反事实由 CF-DIP 在内存中构造, 不由输入文件携带)。
    """
    source = os.path.abspath(source)
    if not os.path.exists(source):
        raise ValueError(f"输入不存在: {source}")
    wells = []
    if source.endswith(".npz"):
        loaded = load_wells_from_npz(source)
        for w in loaded:
            wells.append(dict(well_id=w["well_id"], nrm=np.asarray(w["nrm"], float),
                              has_azimuth=True, source_type="L0", origin="npz"))
        kind = "npz_wells"
    elif source.endswith(".pt"):
        entries = from_forge_fmi_pt(source)
        for e in entries:
            wells.append(dict(well_id=e.source_name, nrm=np.asarray(e.normals, float),
                              has_azimuth=True, source_type=e.source_type,
                              origin="forge_pt(ftype=%s)" % e.structure_type))
        kind = "forge_pt"
    elif os.path.isdir(source):
        loaded = load_wells_from_dir(source)
        for w in loaded:
            wells.append(dict(well_id=w["well_id"], nrm=np.asarray(w["nrm"], float),
                              has_azimuth=True, source_type="L0", origin="csv_dir"))
        kind = "csv_dir"
    elif source.lower().endswith(".csv"):
        # 单 CSV: 冻结 read_well_csv (T91 FIX-2 语义: 缺列友好报错)
        w = load_well_csv(source)
        wells.append(dict(well_id=w["well_id"], nrm=np.asarray(w["nrm"], float),
                          has_azimuth=True, source_type="L0", origin="csv"))
        kind = "csv"
    else:
        raise ValueError(f"不支持的输入类型: {source} (支持 .npz/.pt/.csv/CSV 目录)")

    for w in wells:
        nrm = np.asarray(w["nrm"], float)
        keep = np.linalg.norm(nrm, axis=1) > 0.5
        if not bool(keep.all()):
            nrm = nrm[keep]
        w["nrm"] = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        w["n_obs"] = int(len(nrm))
    if not wells:
        raise ValueError(f"输入无有效井: {source}")
    if site_name is None:
        site_name = os.path.splitext(os.path.basename(source))[0]
    # 证据单元标签 (forge_pt 的单元是 ftype 拆分条目, 非物理井 —— 建议/联合
    # 措辞必须跟随, 禁把条目当井误导)
    units_label = ("FMI ftype 拆分条目 (非物理井; 多井池化决策须以物理井为单位重评)"
                   if kind == "forge_pt" else "物理井")
    return {
        "site_name": site_name, "kind": kind, "wells": wells,
        "source_path": source, "units_label": units_label,
        "n_total_obs": int(sum(w["n_obs"] for w in wells)),
    }


# ---------------------------------------------------------------------------
# 证据分级 (L4 表语义的数据档位判定; prereg_amendments_1: L1 在场即 B 档)
# ---------------------------------------------------------------------------

def evidence_grade(site):
    """A/B/C/D/E 分级 + 适用边界句 (prereg#evidence_grading_rules)。"""
    wells = site["wells"]
    types = sorted({w["source_type"] for w in wells})
    has_l1 = "L1" in types
    has_l0 = "L0" in types
    n_total = site["n_total_obs"]
    sparse_wells = [w["well_id"] for w in wells if w["n_obs"] < _OBS_MIN_GRADE_C]
    diponly = [w["well_id"] for w in wells if not w["has_azimuth"]]

    if diponly:
        grade = "E"
        reasons = [f"dip-only 井在场 (缺方位角): {diponly}"]
        band = _load_diponly_bounds()
        reasons.append(
            "dip-only 期望误差 mean=%.1f° / median=%.1f° / p90=%.1f° (在册 %s) "
            "远超可用阈值, 组系表抑制, 必须补测方位角" % (
                band["mean"], band["median"], band["p90"], band["source"]))
    elif has_l1 or (has_l0 and "L2" in types and "L3" in types):
        grade = "B"
        reasons = ["FMI (L1) 证据在场 -> L4 表 §3 决策路径判 B 档 (6–10° 语义, "
                   "T58 后以 ≤12° 工程阈值内如实报告为准)"]
        if not has_l0:
            reasons.append("纯 L1 单源: L4 表 B 档的双源编录前提未满足, "
                           "分级按数据档位执行而非测量值 (prereg_amendments_1)")
    elif has_l0 and not sparse_wells and n_total >= _OBS_MIN_GRADE_C:
        grade = "C"
        reasons = [f"仅 L0 编录表, 有方位角, 总观测 {n_total} >= 20 -> C 档 "
                   f"(10–15° 承诺档语义)"]
    elif has_l0:
        grade = "D"
        reasons = [f"仅 L0 编录表, 有方位角, 但存在稀疏井 {sparse_wells} (<20) -> D 档 "
                   "(15–20° 语义, 组系表降级为参考)"]
    else:
        grade = "D"
        reasons = ["数据档位未覆盖 (无 L0/L1), 按 D 档保守处理"]
    boundary = ("该分级不适用于数据档位以外的精度评估: 分级是数据档位判定, 非精度测量; "
                "精度承诺口径 = 组系表 modal_err, 逐点误差受信息墙 ~31.7°±0.5° 约束 "
                "(在册 NNS-045 口径), 与本分级无关。")
    return {"grade": grade, "reasons": reasons, "boundary": boundary,
            "sparse_wells": sparse_wells, "diponly_wells": diponly,
            "source_types": types}


# ---------------------------------------------------------------------------
# 组系表 + P32 + 渗流 + 场景 (全部冻结 API 直出)
# ---------------------------------------------------------------------------

def _pooled_normals(site, wells=None):
    wells = site["wells"] if wells is None else wells
    nrm = np.vstack([w["nrm"] for w in wells])
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)


def site_set_table(site, K=None, seed=_SEED):
    """冻结组系表路径: auto_select_K (可固定) + set_table_from_normals。"""
    nrm = _pooled_normals(site)
    if K is None:
        K, kinfo = auto_select_K(nrm, Krange=(2, 8), seed=seed)
        kinfo = {k: (v if isinstance(v, (int, str)) else round(float(v), 6))
                 for k, v in kinfo.items()}
    else:
        kinfo = {"fixed_K": int(K), "fixed_reason": "counterfactual: K 固定 base 值"}
    st, set_ids = set_table_from_normals(nrm, K=K, seed=seed)
    return st, set_ids, int(K), kinfo, nrm


def p32_estimate(site, wells=None, domain=(50.0, 50.0, 50.0)):
    nrm = _pooled_normals(site, wells)
    net = {"nrm_full": nrm, "nrm": nrm}
    return estimate_p32_interval(net, tuple(domain))


def percolation_two_stage(st, domain, beta=_BETA_DEFAULT):
    """两阶段渗流曲线 (dfn_from_borehole.py [3/6] 冻结模式: 粗 5 seeds 定位
    + crit 邻域细网格 20 seeds)。返回 (fine_result, coarse_result)。"""
    domain = tuple(domain)
    p32_max = min(5.0, 15000 * (np.pi * (0.5 * 5.0)) / (domain[0] * domain[1] * domain[2]))
    grid = _COARSE_GRID[_COARSE_GRID <= p32_max]
    if len(grid) < 3:
        grid = np.linspace(0.05, p32_max, 6)
    coarse = percolation_curve(st, grid, beta=beta, domain=domain,
                               seeds=range(_COARSE_SEEDS), pbc=False)
    crit_coarse = coarse["p32_crit"]
    fine_grid = np.array([crit_coarse * 0.7, crit_coarse * 0.85,
                          crit_coarse, crit_coarse * 1.15, crit_coarse * 1.3])
    fine_grid = fine_grid[fine_grid > 0]
    fine_grid = np.unique(np.round(fine_grid, 3))
    fine = percolation_curve(st, fine_grid, beta=beta, domain=domain,
                             seeds=range(_FINE_SEEDS), pbc=False)
    return fine, coarse


# 场景工程轴默认 (dfn_from_borehole AXIS_DEFAULT 同源; 未给 = 未经确认的工程假设,
# 必须随 axis_provenance 明示)
_AXIS_DEFAULT = [1.0, 0.0, 0.0]


def _check_axis(v, where):
    """轴只校验不归一: 冻结场景函数内部会归一 (well_pair_axis/(norm+1e-12)),
    装配器若预归一会引入 ulp 级偏差且偏离客户给定值。零向量/非有限在此拒绝。"""
    a = np.asarray(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"{where} 需 3 个分量, got shape={a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{where} 含 NaN/Inf, 无法作为工程轴")
    if float(np.linalg.norm(a)) == 0.0:
        raise ValueError(f"{where} 是零向量, 无法确定方向")
    return a


def scenario_metrics(st, p32_ref, domain, beta=_BETA_DEFAULT, seed=_SEED,
                     well_pair_axis=None, tunnel_axis=None):
    """三场景判定 (冻结函数直出) + 轴来源声明 + 不适用边界句 (R1)。"""
    dfn = generate_dfn(st, p32=p32_ref, beta=beta, domain=tuple(domain), seed=seed)
    wp = _check_axis(well_pair_axis if well_pair_axis is not None else _AXIS_DEFAULT,
                     "well_pair_axis")
    tn = _check_axis(tunnel_axis if tunnel_axis is not None else _AXIS_DEFAULT,
                     "tunnel_axis")
    egs = egs_connectivity_metric(dfn, wp)
    mine = mine_risk_sections(dfn, tn)
    disp = disposal_escape_priority(dfn)

    def _axis_prov(name, given):
        return {"axis_used": [round(float(x), 9) for x in (wp if name == "well_pair" else tn)],
                "source": ("explicit_cli" if given else "default_horizontal_x (未经客户确认的工程假设, 非实测值)")}
    prov = {"well_pair": _axis_prov("well_pair", well_pair_axis is not None),
            "tunnel": _axis_prov("tunnel", tunnel_axis is not None)}
    out = {
        "reference_p32": float(p32_ref),
        "reference_note": "三场景在 base p32_crit 处评估 (dfn_from_borehole [4/6] 冻结模式), "
                          "DFN 为 Baecher 盘随机实现, 裂隙位置非实测",
        "axis_provenance": prov,
        "egs": _fmt_scenario(egs, "angle_to_well_pair_deg", "assessment",
                             "该判定不适用于未按本报告假设方位布井的场地: 井对连线轴%s; "
                             "且以 Baecher 盘筛查模型为前提, 非水力试验/压裂实测结论。" % (
                                 "为默认假设 [1,0,0]" if well_pair_axis is None else "为客户给定"),),
        "mine": _fmt_scenario(mine, "angle_to_tunnel_deg", "risk_level",
                              "该判定不适用于巷道轴向与假设不符的场地: 巷道轴%s; "
                              "筛查级风险标记, 非稳定性结论或支护设计依据。" % (
                                  "为默认假设 [1,0,0]" if tunnel_axis is None else "为客户给定"),),
        "disposal": _fmt_scenario(disp, "angle_to_vertical_deg", "escape_priority",
                                  "该判定不适用于废罐主轴非竖直布置的场地: 假设垂直逃逸方向 "
                                  "[0,0,1]; 筛查级优先级, 非核素运移模拟结论。"),
    }
    return out


def _fmt_scenario(raw, angle_key, verdict_key, boundary):
    """冻结场景输出 -> 报告字段 (保留冻结原值, 补边界句)。"""
    d = {"dominant_direction": [round(float(x), 9) for x in raw["dominant_direction"]],
         "connectivity_fraction": float(raw["connectivity_fraction"]),
         angle_key: (None if raw.get(angle_key) is None else float(raw[angle_key])),
         verdict_key: raw[verdict_key],
         "assumptions": raw["assumptions"],
         "boundary": boundary}
    return d


# ---------------------------------------------------------------------------
# 校准不确定度带 (R2: cone_95 × 在册 alpha, 禁裸分位数)
# ---------------------------------------------------------------------------

def _infer_calib_site(site):
    """把输入场地映射到 T46 在册校准键 (beishan/forge)。

    依据 = 输入路径与场地名 (T46 回测面就这两个场地); 命中才用 per-site alpha,
    否则回退 alpha_global 并披露跨场地借用。禁按记忆硬编码 alpha 数值。
    """
    haystack = (site["source_path"].replace("\\", "/").lower() + " " + site["site_name"].lower())
    if "beishan" in haystack:
        return "beishan"
    if "forge" in haystack:
        return "forge"
    return None


def calibrated_group_bands(nrm, set_ids, st, site_name, calib_site=None,
                           seed=_SEED, n_bootstrap=100):
    """逐组: 冻结 bootstrap 锥半角 × 在册 alpha (T46 校准链)。"""
    calib = _load_calibration()
    alpha_site = calib["alpha_per_site"].get(calib_site) if calib_site else None
    if alpha_site is not None:
        alpha = float(alpha_site)
        alpha_kind = "site (T46 留一回测在场: %s)" % calib_site
        alpha_source = calib["source"] + "#kappa_scaling_alpha_per_site." + str(calib_site)
    else:
        alpha = float(calib["alpha_global"])
        alpha_kind = "global (跨场地借用: 本场地不在 T46 回测面内, 边界句披露)"
        alpha_source = calib["source"] + "#kappa_scaling_alpha_global"
    groups = []
    for k in range(st.K):
        pts = nrm[set_ids == k]
        n = int(len(pts))
        if n >= 3:
            cone_raw = float(bootstrap_confidence_cone(
                pts, n_bootstrap=n_bootstrap, ci_level=0.95, seed=seed))
            kappa = float(estimate_kappa(pts, st.centers[k]))
        else:
            cone_raw, kappa = 180.0, 0.01
        dip, dip_dir = normal_to_dip_dipdir(st.centers[k].reshape(1, 3))
        groups.append({
            "group": int(k), "n_obs": n,
            "dip_deg": round(float(dip[0]), 3), "dip_dir_deg": round(float(dip_dir[0]), 3),
            "proportion": round(float(st.proportions[k]), 6),
            "kappa": round(kappa, 3),
            "cone95_raw_deg": round(cone_raw, 4),
            "cone_calibrated_deg": round(cone_raw * alpha, 4),
            "note": ("观测 <3, 不确定度极大 (锥半角取 180°)" if n < 3 else ""),
        })
    return {
        "alpha": alpha, "alpha_kind": alpha_kind,
        "alpha_source": alpha_source,
        "recalibrated_coverage": calib["recalibrated_coverage"],
        "calibration_applicability": (
            "T46 留一回测 (beishan L0 22 井 + FORGE L1, K=%d pooled): 锥半角×alpha 后 "
            "经验覆盖率 %.2f%%。可交换性假设仅适用于同场地内单元间, 不构成分布无关保证; "
            "回测真值 = 留出单元观测-only 组模态 (无全量泄漏)。" % (
                calib["K"], calib["recalibrated_coverage"] * 100)),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# 反事实电池 (M3: 扰动观测面 -> 判定/建议必须可变)
# ---------------------------------------------------------------------------

def _scenario_tiers(sc):
    """场景判定档位三元组 (翻转检测的比对键)。"""
    return {"egs": sc["egs"]["assessment"],
            "mine": sc["mine"]["risk_level"],
            "disposal": sc["disposal"]["escape_priority"]}


def _max_center_disp(st_a, st_b):
    """两组系表组心的最大匹配角距 (匈牙利, acos|cos| 度) —— 复用 site_model
    的配对语义但就地实现 (两组心数可能不同, 取 min(K) 最优指派)。"""
    from scipy.optimize import linear_sum_assignment
    a, b = np.asarray(st_a.centers, float), np.asarray(st_b.centers, float)
    Ka, Kb = len(a), len(b)
    if Ka == 0 or Kb == 0:
        return 180.0
    cost = np.zeros((Ka, Kb))
    for i in range(Ka):
        for j in range(Kb):
            c = abs(float(np.clip(np.dot(a[i], b[j]), -1.0, 1.0)))
            cost[i, j] = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
    row, col = linear_sum_assignment(cost[:min(Ka, Kb), :min(Ka, Kb)])
    return float(cost[row, col].max())


def cf_dip_only(site, base_grade):
    """CF-DIP: 剥离方位角 (dip-only) -> 分级必须翻 E, 组系表抑制 (解析)。"""
    wells_do = [dict(w, has_azimuth=False) for w in site["wells"]]
    site_do = dict(site, wells=wells_do)
    grade_do = evidence_grade(site_do)
    flip = (grade_do["grade"] != base_grade["grade"])
    return {
        "what": "剥离全部方位角 (dip-only 反事实)",
        "grade_base": base_grade["grade"], "grade_cf": grade_do["grade"],
        "set_table_suppressed": True,
        "recommendation_REC2_fires": True,
        "flip": bool(flip),
        "note": "解析反事实: dip-only 上界 (%s) 使组系表不可用, 分级 C->E 翻转" % (
            "results/dip_only_upper_bound.json"),
    }


def cf_coverage_40(site, base, K_base, domain, seed=_SEED):
    """CF-COV: 每井观测掩码 rng(999) obs_frac=0.4 (项目标准观测协议) 重走
    组系表 (K 固定 base) + 锥带 + p32。"""
    wells_cf = []
    for w in site["wells"]:
        n = w["n_obs"]
        occ = np.random.default_rng(999).random(n) < 0.4
        nrm_cf = w["nrm"][occ]
        wells_cf.append(dict(w, nrm=nrm_cf, n_obs=int(len(nrm_cf))))
    site_cf = dict(site, wells=wells_cf,
                   n_total_obs=int(sum(w["n_obs"] for w in wells_cf)))
    if site_cf["n_total_obs"] < 4:
        return {"what": "覆盖率降至 0.4 (rng999 协议)", "skipped": "掩码后无足够观测",
                "flip": False}
    st_cf, ids_cf, _K, _ki, nrm_cf = site_set_table(site_cf, K=K_base, seed=seed)
    bands_cf = calibrated_group_bands(nrm_cf, ids_cf, st_cf, site["site_name"],
                                      calib_site=_infer_calib_site(site), seed=seed)
    p32_cf = p32_estimate(site_cf, wells_cf, domain)
    # 锥带宽度比 (同组号对比; 组心漂移使组号语义近似, 以保守的最大比值口径计)
    ratios = []
    for g_base, g_cf in zip(base["bands"]["groups"], bands_cf["groups"]):
        if g_base["cone_calibrated_deg"] > 0 and g_cf["n_obs"] >= 3:
            ratios.append(g_cf["cone_calibrated_deg"] / g_base["cone_calibrated_deg"])
    n_groups_widened = int(sum(1 for r in ratios if r >= _TH_COVCONE_RATIO))
    n_sparse = int(sum(1 for w in wells_cf if w["n_obs"] < _OBS_MIN_GRADE_C))
    flip = bool(n_groups_widened >= 1 or n_sparse >= 1)
    return {
        "what": "每井观测掩码 rng=default_rng(999) obs_frac=0.4 (项目标准观测协议), K 固定 %d" % K_base,
        "n_obs_base": base["n_total_obs"], "n_obs_cf": site_cf["n_total_obs"],
        "cone_ratio_per_group": [round(r, 4) for r in ratios],
        "n_groups_widened_ge_1.5x": n_groups_widened,
        "n_wells_below_20_obs_cf": n_sparse,
        "p32_p50_base": base["p32"]["p32_p50"], "p32_p50_cf": p32_cf["p32_p50"],
        "REC5_fires": bool(n_groups_widened >= 1),
        "flip_criterion": "锥带宽度比 >= 1.5 的组数 >=1, 或任一井观测数 <20",
        "flip": flip,
    }


def cf_leave_one_out(site, base, K_base, domain, p32_ref, beta=_BETA_DEFAULT,
                     seed=_SEED, well_pair_axis=None, tunnel_axis=None):
    """CF-LOO: 逐井移除 -> 组系表 (K 固定) + p32 + 三场景 (base p32_crit 锚定)。"""
    rows = []
    n_wells = len(site["wells"])
    for i in range(n_wells):
        wells_wo = [w for j, w in enumerate(site["wells"]) if j != i]
        if not wells_wo:
            continue
        site_wo = dict(site, wells=wells_wo)
        st_wo, _ids, _K, _ki, nrm_wo = site_set_table(site_wo, K=K_base, seed=seed)
        p32_wo = p32_estimate(site_wo, wells_wo, domain)
        try:
            sc_wo = scenario_metrics(st_wo, p32_ref, domain, beta=beta, seed=seed,
                                     well_pair_axis=well_pair_axis,
                                     tunnel_axis=tunnel_axis)
        except Exception as e:   # 场景生成失败 (如退化组系) 如实登记
            rows.append({"removed_well": site["wells"][i]["well_id"],
                         "error": str(e)[:120], "flip": True})
            continue
        tiers = _scenario_tiers(sc_wo)
        disp = _max_center_disp(base["st"], st_wo)
        p32_out_of_band = not (base["p32"]["p32_p10"] <= p32_wo["p32_p50"]
                               <= base["p32"]["p32_p90"])
        flip = bool(tiers != base["scenario_tiers"] or disp >= _TH_KEYWELL_CENTER_DISP
                    or p32_out_of_band)
        rows.append({
            "removed_well": site["wells"][i]["well_id"],
            "n_obs_removed": site["wells"][i]["n_obs"],
            "scenario_tiers": tiers,
            "scenario_flip": bool(tiers != base["scenario_tiers"]),
            "max_center_disp_deg": round(disp, 3),
            "p32_p50_cf": p32_wo["p32_p50"],
            "p32_p50_out_of_base_band": bool(p32_out_of_band),
            "flip": flip,
        })
    n_flips = int(sum(1 for r in rows if r["flip"]))
    return {
        "what": "逐井移除 (K 固定 %d, 场景在 base p32_crit=%.4f 锚定)" % (K_base, p32_ref),
        "per_well": rows,
        "n_wells": n_wells, "n_flips": n_flips,
        "flipped_wells": [r["removed_well"] for r in rows if r["flip"]],
        "flip_criterion": "场景档位变化 OR 组心位移 >= 10° OR p32_p50 越出 base [p10,p90]",
        "flip": bool(n_flips >= 1),
    }


# ---------------------------------------------------------------------------
# 观测建议引擎 v0 (每条 = 解析触发谓词 + 动作 + 反事实可消性声明)
# ---------------------------------------------------------------------------

def _band_overlap(p32, perc):
    """REC-1 谓词: P32 估计带与渗流阈值带相交 (解析)。"""
    return not (p32["p32_p90"] < perc["p32_crit_lower"]
                or p32["p32_p10"] > perc["p32_crit_upper"])


def _scenario_edge(sc):
    """REC-6 谓词: 任一场景面-轴夹角落在其档位阈值 (30/60) ±10° 内。"""
    edges = []
    for scen, angle_key in (("egs", "angle_to_well_pair_deg"),
                            ("mine", "angle_to_tunnel_deg"),
                            ("disposal", "angle_to_vertical_deg")):
        ang = sc[scen].get(angle_key)
        if ang is None:
            continue
        for thr in (30.0, 60.0):
            if abs(ang - thr) <= _TH_SCENARIO_EDGE:
                edges.append({"scenario": scen, "angle_deg": round(ang, 2),
                              "threshold_deg": thr})
    return edges


def build_recommendations(site, base, cf):
    """逐条评估 REC-1..6 (prereg#recommendation_engine_v0); 未触发者也要落名
    与谓词读数 (证未触发, 防装饰)。"""
    recs = []
    # REC-1 P32 带与阈值带重叠
    ov = _band_overlap(base["p32"], base["perc"])
    recs.append({
        "id": "REC-1_p32_band_overlap",
        "triggered": bool(ov),
        "trigger_evidence": {
            "p32_band": [base["p32"]["p32_p10"], base["p32"]["p32_p90"]],
            "crit_band": [base["perc"]["p32_crit_lower"], base["perc"]["p32_crit_upper"]],
            "overlap": bool(ov)},
        "action": ("补尺寸数据 (露头迹长/岩心扫描) 收窄 P32 带; 带收窄前连通性判定输出"
                   "『不确定』, 不做可用/不可用二值断言") if ov else
                  "不触发: P32 带与阈值带分离 (连通性判定在当前带下稳定)",
        "counterfactual_kill": "带分离的场地该建议不存在 (谓词为假即消失)",
    })
    # REC-2 dip-only
    diponly = base["grade"]["diponly_wells"]
    recs.append({
        "id": "REC-2_dip_only",
        "triggered": bool(diponly),
        "trigger_evidence": {"diponly_wells": diponly},
        "action": "补测方位角 (dip-only 期望误差 mean=%.1f° / median=%.1f°, 在册 %s)" % (
            _load_diponly_bounds()["mean"], _load_diponly_bounds()["median"],
            _load_diponly_bounds()["source"]) if diponly else
            "不触发: 全井含方位角",
        "counterfactual_kill": "全井含方位角时该建议不存在 (CF-DIP 反事实展示其触发态)",
    })
    # REC-3 关键井 (CF-LOO 驱动)
    loo = cf["CF-LOO"]
    flipped = loo["flipped_wells"]
    units = site.get("units_label", "物理井")
    recs.append({
        "id": "REC-3_key_well",
        "triggered": bool(flipped),
        "trigger_evidence": {"flipped_wells": flipped, "n_flips": loo["n_flips"],
                             "units_label": units,
                             "flip_criterion": loo["flip_criterion"]},
        "action": ("关键证据单元 (%s) 列入复测/保护清单: %s; 下一步加孔/加密观测优先补在"
                   "关键单元代表的组系覆盖薄弱区" % (units, ", ".join(map(str, flipped))))
                  if flipped else
                  "不触发: 移除任一单元均无判定翻转 (场地组系判定对单点损失稳健)",
        "counterfactual_kill": "无翻转场地不点名任何单元 (禁恒真装饰)",
    })
    # REC-4 联合规则 (L20/L30)
    cons = base.get("joint_rule")
    if cons is not None:
        joint_wells = [v["well"] for v in cons["verdicts"]
                       if v["tier"] in ("joint", "pair_only")]
        indep_wells = [v["well"] for v in cons["verdicts"] if v["tier"] == "independent"]
        # 在册场地级对照 (运行时直读, 禁手抄): L20/L30 规则判读 ≠ 池化红利,
        # R60 评估显示部分场地全独立更优 (beishan22: 19.15 vs 规则臂 20.77)
        r60_note, r60_ref = "", None
        try:
            r60 = _read_registered_json("results", "v_r60_joint_rule_eval.json")
            site_eval = r60.get("inputs", {}).get(site["site_name"])
            if isinstance(site_eval, dict) and "rule" in site_eval:
                r60_ref = "results/v_r60_joint_rule_eval.json#inputs." + site["site_name"]
                rule_arm = site_eval["rule"]
                indep_arm = site_eval.get("independent")
                pool_arm = site_eval.get("full_pool")
                parts = ["规则臂 %.2f±%.2f°" % (rule_arm["mean_deg"], rule_arm["std_deg"])]
                if indep_arm:
                    parts.append("全独立 %.2f±%.2f°" % (indep_arm["mean_deg"], indep_arm["std_deg"]))
                if pool_arm:
                    parts.append("全池化 %.2f±%.2f°" % (pool_arm["mean_deg"], pool_arm["std_deg"]))
                if indep_arm and indep_arm["mean_deg"] < rule_arm["mean_deg"]:
                    r60_note = ("⚠ 在册场地级评估 (%s): %s —— 全独立更优, "
                                "本场地池化红利不成立, 建议保持独立编录" % (
                                    r60_ref, ", ".join(parts)))
                else:
                    r60_note = "在册场地级评估 (%s): %s —— 规则池化不劣于独立, 按规则执行" % (
                        r60_ref, ", ".join(parts))
            else:
                r60_note = ("⚠ 本场地无在册池化红利对照; 多井联合在诚实口径下可能无增益"
                            "甚至劣化 (八期 DEP-013), 池化决策须以物理井为单位重评")
        except Exception:
            r60_note = "⚠ 在册池化对照不可得, 池化决策须以物理井为单位重评"
        recs.append({
            "id": "REC-4_joint_rule",
            "triggered": bool(joint_wells),
            "trigger_evidence": {"joint_or_pair_units": joint_wells,
                                 "independent_units": indep_wells,
                                 "units_label": units,
                                 "tiers": {str(v["well"]): v["tier"] for v in cons["verdicts"]},
                                 "rule": "L20/L30 在册规则 (<20° 联合 / 20–35° 仅联最一致对)"},
            "action": ("L20/L30 判读: 单元 %s 一致性达标 (可池化候选), %s。%s" % (
                joint_wells, units, r60_note))
                      if joint_wells else
                      "不触发: 全部单元 independent (无联合红利, 与在册多井审计结论同向)",
            "counterfactual_kill": "全 independent 时降级为『无联合红利』声明",
        })
    # REC-5 覆盖率
    cov = cf["CF-COV"]
    recs.append({
        "id": "REC-5_coverage",
        "triggered": bool(cov.get("REC5_fires")),
        "trigger_evidence": {"n_groups_widened_ge_1.5x": cov.get("n_groups_widened_ge_1.5x", 0),
                             "cone_ratio_per_group": cov.get("cone_ratio_per_group", [])},
        "action": "加密观测 (缩短编录间距/提高观测比例) 优先于加新孔" if cov.get("REC5_fires")
                  else "不触发: 覆盖率减半未使校准带显著变宽",
        "counterfactual_kill": "覆盖率扰动不改变带的场地该建议不存在",
    })
    # REC-6 场景阈值边缘
    edges = _scenario_edge(base["scenarios"])
    recs.append({
        "id": "REC-6_scenario_threshold_proximity",
        "triggered": bool(edges),
        "trigger_evidence": {"edges": edges, "edge_band_deg": _TH_SCENARIO_EDGE},
        "action": ("场景判定处于档位阈值边缘 (±10°), 建议补测主方向后再定档: %s" % edges)
                  if edges else "不触发: 所有场景夹角远离档位阈值",
        "counterfactual_kill": "夹角远离阈值的场地该建议不存在",
    })
    return recs


# ---------------------------------------------------------------------------
# 主组装器
# ---------------------------------------------------------------------------

def build_decision_report(source, out_dir, site_name=None, domain=(50.0, 50.0, 50.0),
                          beta=_BETA_DEFAULT, seed=_SEED, K=None,
                          well_pair_axis=None, tunnel_axis=None,
                          run_counterfactuals=True, quiet=False):
    """端到端: 装载 -> 分级 -> 组系表 -> P32/渗流 -> 场景 -> 校准带 -> 反事实
    -> 建议 -> Markdown + JSON。全部数字冻结 API 直出。"""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    site = load_site(source, site_name=site_name)

    # 1. 证据分级
    grade = evidence_grade(site)

    # 2. 组系表 (E 级抑制)
    if grade["grade"] == "E":
        st = set_ids = None
        K_used, kinfo = 0, {"suppressed": "dip-only 场地组系表抑制"}
        bands = None
        nrm = None
    else:
        st, set_ids, K_used, kinfo, nrm = site_set_table(site, K=K, seed=seed)

    # 3. P32 + 渗流 + 场景
    p32 = p32_estimate(site, domain=domain)
    if st is not None:
        fine, coarse = percolation_two_stage(st, domain, beta=beta)
        scenarios = scenario_metrics(st, fine["p32_crit"], domain, beta=beta,
                                     seed=seed, well_pair_axis=well_pair_axis,
                                     tunnel_axis=tunnel_axis)
        bands = calibrated_group_bands(nrm, set_ids, st, site["site_name"],
                                       calib_site=_infer_calib_site(site), seed=seed)
    else:
        fine = coarse = None
        scenarios = None

    # 4. 多井一致性建议层 (>=2 井; 冻结 joint_rule_partition, 只作建议不作组系表)
    joint = None
    if st is not None and len(site["wells"]) >= 2:
        part = joint_rule_partition([w["nrm"] for w in site["wells"]],
                                    K_audit=K_used, seed=0)
        joint = {"verdicts": part["verdicts"], "groups": part["groups"],
                 "units_label": site["units_label"],
                 "rule": "L20/L30 在册决策规则 (results/multiwell_joint_audit.json 推导)",
                 "note": "建议层输出, 不构成本报告组系表的生成路径"}

    # 5. 连通性判定 (解析: p32 位置 vs 阈值带)
    if fine is not None:
        overlap = _band_overlap(p32, fine)
        if p32["p32_p90"] < fine["p32_crit_lower"]:
            conn_verdict = "估计带上缘低于阈值带下缘 -> 当前证据下连通风险低"
        elif p32["p32_p10"] > fine["p32_crit_upper"]:
            conn_verdict = "估计带下缘高于阈值带上缘 -> 当前证据下连通风险高"
        else:
            conn_verdict = "P32 带与阈值带相交 -> 连通性判定不确定 (需收窄带)"
    else:
        overlap, conn_verdict = None, "不适用 (dip-only 场地组系表/渗流抑制)"

    # 6. base 包 (反事实与建议的共同参照)
    base = {
        "site": site, "grade": grade, "st": st, "set_ids": set_ids,
        "K": K_used, "kinfo": kinfo, "p32": p32,
        "perc": fine, "perc_coarse": coarse, "scenarios": scenarios,
        "scenario_tiers": (_scenario_tiers(scenarios) if scenarios else None),
        "bands": bands, "joint_rule": joint, "conn_verdict": conn_verdict,
        "band_overlap": overlap, "n_total_obs": site["n_total_obs"],
        "domain": tuple(domain), "beta": beta, "seed": seed,
    }

    # 7. 反事实电池
    cf = {"CF-DIP": None, "CF-COV": None, "CF-LOO": None}
    if run_counterfactuals:
        cf["CF-DIP"] = cf_dip_only(site, grade)
        if st is not None:
            cf["CF-COV"] = cf_coverage_40(site, base, K_used, domain, seed=seed)
            cf["CF-LOO"] = cf_leave_one_out(site, base, K_used, domain,
                                            fine["p32_crit"], beta=beta, seed=seed,
                                            well_pair_axis=well_pair_axis,
                                            tunnel_axis=tunnel_axis)

    # 8. 建议 (在 base + 反事实之上评估)
    recs = build_recommendations(site, base, cf) if st is not None else [
        {"id": "REC-2_dip_only", "triggered": True,
         "trigger_evidence": {"diponly_wells": grade["diponly_wells"]},
         "action": "补测方位角 (dip-only 场地, 组系表/场景判定全部抑制)",
         "counterfactual_kill": "补测方位角后本建议消失, 报告恢复全链判定"}]

    # 9. R3 翻转计数
    flips = []
    for name, block in cf.items():
        if block and block.get("flip"):
            flips.append(name)
    r3 = {"flipped_blocks": flips,
          "n_recommendations_triggered": int(sum(1 for r in recs if r["triggered"])),
          "gate": ">=1 个反事实块翻转 => 可红; 全部恒真 => R3 FAIL",
          "pass": bool(flips)}

    # 10. JSON
    report = _to_json_dict(site, base, cf, recs, r3, t0)
    json_path = os.path.join(out_dir, "decision_report.json")
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    md_path = os.path.join(out_dir, "decision_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))
    if not quiet:
        print("[W26] %s: grade=%s K=%s p32_p50=%.4f crit=%.4f REC触发=%d 翻转=%s" % (
            site["site_name"], grade["grade"], K_used, p32["p32_p50"],
            (fine["p32_crit"] if fine else float("nan")),
            r3["n_recommendations_triggered"], flips or "无"))
        print("[W26] JSON -> %s" % json_path)
        print("[W26] MD   -> %s" % md_path)
    return report


def _to_json_dict(site, base, cf, recs, r3, t0):
    import json
    p32 = base["p32"]
    perc = base["perc"]
    out = {
        "watermark": WATERMARK,
        "line": "W26",
        "unit": "w26_decision_report_prototype",
        "site_name": site["site_name"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "elapsed_sec": round(time.time() - t0, 1),
        "inputs": {
            "source_path": site["source_path"], "kind": site["kind"],
            "domain_m": [float(x) for x in base["domain"]],
            "beta": base["beta"], "seed": base["seed"],
            "n_wells": len(site["wells"]), "n_total_obs": site["n_total_obs"],
            "units_label": site.get("units_label", "物理井"),
            "wells": [{"well_id": w["well_id"], "n_obs": w["n_obs"],
                       "source_type": w["source_type"], "origin": w["origin"]}
                      for w in site["wells"]],
        },
        "evidence_grade": base["grade"],
        "set_table": (None if base["st"] is None else {
            "K": base["K"],
            "K_selection": base["kinfo"],
            "centers": [[round(float(x), 9) for x in c] for c in base["st"].centers],
            "concentrations": [round(float(x), 6) for x in base["st"].concentrations],
            "proportions": [round(float(x), 6) for x in base["st"].proportions],
            "set_ids_sha12": _sha12(base["set_ids"]),
            "api": "dfn.auto_select_K(Krange=(2,8),seed=42) + dfn.set_table_from_normals(seed=42)",
        }),
        "p32_estimate": {
            "p32_p10": p32["p32_p10"], "p32_p50": p32["p32_p50"], "p32_p90": p32["p32_p90"],
            "N_obs": p32["N_obs"], "obs_frac": p32["obs_frac"],
            "assumptions": p32["assumptions"],
            "api": "dfn.estimate_p32_interval",
        },
        "percolation": (None if perc is None else {
            "p32_grid_coarse": [float(x) for x in base["perc_coarse"]["p32_grid"]],
            "p_conn_coarse": [float(x) for x in base["perc_coarse"]["p_conn"]],
            "p32_grid": [float(x) for x in perc["p32_grid"]],
            "p_conn": [float(x) for x in perc["p_conn"]],
            "p32_crit": perc["p32_crit"], "p32_crit_lower": perc["p32_crit_lower"],
            "p32_crit_upper": perc["p32_crit_upper"],
            "n_realizations_per_point": perc["n_realizations"],
            "assumptions": perc["assumptions"],
            "api": "percolation.percolation_curve (两阶段: 粗 5 seeds + 细 20 seeds)",
        }),
        "connectivity_judgment": {
            "verdict": base["conn_verdict"],
            "band_overlap": base["band_overlap"],
            "boundary": "该判定不适用于真实连通性断言: 渗流为 Baecher 盘筛查模型 "
                        "(假设 β=3.5 尺寸分布), 输出是证据位置关系, 不是水文实测。",
        },
        "scenarios": base["scenarios"],
        "uncertainty": base["bands"],
        "multiwell_consistency": base["joint_rule"],
        "counterfactuals": cf,
        "recommendations": recs,
        "R3_counterfactual_gate": r3,
        "assumptions": _assumptions_list(base),
        "honest_boundary": {
            "screening_level": True,
            "research_prototype": True,
            "no_external_quantitative_promises": "裁定 58: 对外零量化承诺; 本报告数字为研究线读数",
            "dfn_positions_random": "DFN 裂隙位置为随机实现 (Baecher 盘), 非实测定位",
            "size_distribution_assumed": "β=%.2f 为假设值, 无迹长数据" % base["beta"],
        },
        "provenance": {
            "task": "W26 决策报告原型 (results/w26_decision_report_prototype/task_spec_v0.md)",
            "frozen_apis": ["dfn.auto_select_K", "dfn.set_table_from_normals",
                            "dfn.estimate_p32_interval", "dfn.generate_dfn",
                            "percolation.percolation_curve",
                            "percolation.egs_connectivity_metric",
                            "percolation.mine_risk_sections",
                            "percolation.disposal_escape_priority",
                            "l4.fuse.bootstrap_confidence_cone",
                            "l4.fuse.estimate_kappa",
                            "site_model.joint_rule_partition",
                            "l4.evidence.from_forge_fmi_pt",
                            "site_model.load_wells_from_npz/load_wells_from_dir/load_well_csv"],
            "registered_inputs": [_CALIB_RELPATH and "results/l4_calibration.json",
                                   "results/dip_only_upper_bound.json"],
        },
    }
    return out


def _assumptions_list(base):
    """H1 纪律: assumptions 非空, 汇集全链假设。"""
    items = [
        "组系表 = 观测-only 球面 k-means (无组标签, 语义为数据属性分组), 非真值组系",
        "P32 为编录表量级估计 (半径范围 [0.5, 5.0]m 几何平均近似), 非实测体密度",
        "渗流为 Baecher 盘筛查模型, β=%.2f 为假设尺寸分布, 非真实连通性判定" % base["beta"],
        "场景判定工程轴未给时为默认假设 [1,0,0] (EGS 井对连线 / 巷道轴), 非客户实测",
        "不确定度带 = bootstrap 锥 × T46 在册 alpha (coverage %.2f%% 留一回测), "
        "非贝叶斯后验" % (_load_calibration()["recalibrated_coverage"] * 100),
        "分级为数据档位判定 (L4 表语义), 非精度测量",
    ]
    if base["perc"] is not None:
        items.append(base["perc"]["assumptions"])
    if base["bands"] is not None:
        items.append(base["bands"]["calibration_applicability"])
    return items


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------

def _render_markdown(r):
    L = []
    L.append("# 决策报告 v1 · %s" % r["site_name"])
    L.append("")
    L.append("> **%s**" % r["watermark"])
    L.append(">")
    L.append("> 生成 %s | 输入 %s (%d 井 / %d 观测) | 域 %s m | β=%.2f (假设)"
             % (r["written_at"], r["inputs"]["source_path"], r["inputs"]["n_wells"],
                r["inputs"]["n_total_obs"], "×".join(str(x) for x in r["inputs"]["domain_m"]),
                r["inputs"]["beta"]))
    L.append("")
    # 分级
    g = r["evidence_grade"]
    L.append("## 1. 证据分级 (L4 数据档位判定)")
    L.append("")
    L.append("**等级 %s**" % g["grade"])
    for why in g["reasons"]:
        L.append("- %s" % why)
    L.append("- 边界: %s" % g["boundary"])
    L.append("")
    # 组系表
    if r["set_table"] is not None:
        stt = r["set_table"]
        L.append("## 2. 组系表 (冻结 API 直出, K=%d)" % stt["K"])
        L.append("")
        L.append("| 组 | 倾角(°) | 倾向(°) | 占比 | κ | 校准锥带(±°) | n_obs |")
        L.append("|---|---|---|---|---|---|---|")
        unc = r["uncertainty"]
        for gr in unc["groups"]:
            L.append("| %d | %.1f | %.1f | %.1f%% | %.1f | **%.1f** | %d |" % (
                gr["group"], gr["dip_deg"], gr["dip_dir_deg"], gr["proportion"] * 100,
                gr["kappa"], gr["cone_calibrated_deg"], gr["n_obs"]))
        L.append("")
        L.append("- 不确定度带 = bootstrap 锥 95%% × α=%.4f (%s), 锚链: %s"
                 % (unc["alpha"], unc["alpha_kind"], unc["alpha_source"]))
        L.append("- 校准语义: %s" % unc["calibration_applicability"])
        L.append("- K 选择: %s" % stt["K_selection"])
        L.append("")
    # 渗流
    if r["percolation"] is not None:
        p = r["percolation"]
        L.append("## 3. 渗流与连通性判定 (筛查级)")
        L.append("")
        L.append("- P32 估计: p50=%.4f [p10=%.4f, p90=%.4f] (编录表量级估计)"
                 % (r["p32_estimate"]["p32_p50"], r["p32_estimate"]["p32_p10"],
                    r["p32_estimate"]["p32_p90"]))
        L.append("- 渗流阈值: p32_crit=%.4f [%.4f, %.4f] (每点 %d 实现, β=%.2f 假设)"
                 % (p["p32_crit"], p["p32_crit_lower"], p["p32_crit_upper"],
                    p["n_realizations_per_point"], r["inputs"]["beta"]))
        L.append("- **连通性判定: %s**" % r["connectivity_judgment"]["verdict"])
        L.append("- 边界: %s" % r["connectivity_judgment"]["boundary"])
        L.append("")
        # 场景
        sc = r["scenarios"]
        L.append("## 4. 场景判定 (在 p32_crit=%.4f 处评估)" % sc["reference_p32"])
        L.append("")
        rows = [("EGS 地热", "egs", "assessment", "angle_to_well_pair_deg"),
                ("矿山/隧道突水", "mine", "risk_level", "angle_to_tunnel_deg"),
                ("核废料处置逃逸", "disposal", "escape_priority", "angle_to_vertical_deg")]
        L.append("| 场景 | 判定 | 面-轴夹角(°) | 连通分数 |")
        L.append("|---|---|---|---|")
        for cn, key, vk, ak in rows:
            s = sc[key]
            ang = s.get(ak)
            L.append("| %s | **%s** | %s | %.3f |" % (
                cn, s[vk], ("—" if ang is None else "%.1f" % ang),
                s["connectivity_fraction"]))
        L.append("")
        for cn, key, _vk, _ak in rows:
            L.append("- %s: %s" % (cn, sc[key]["boundary"]))
        L.append("- 轴来源: %s" % sc["axis_provenance"])
        L.append("")
    # 反事实
    L.append("## 5. 反事实稳定性 (观测建议的牙)")
    L.append("")
    for name in ("CF-DIP", "CF-COV", "CF-LOO"):
        blk = r["counterfactuals"].get(name)
        if blk is None:
            continue
        L.append("- **%s**: %s -> 翻转=%s" % (name, blk["what"], blk.get("flip")))
        if name == "CF-LOO" and blk.get("flipped_wells"):
            L.append("  - 翻转井: %s (判据: %s)" % (blk["flipped_wells"], blk["flip_criterion"]))
        if name == "CF-COV" and "cone_ratio_per_group" in blk:
            L.append("  - 锥带宽度比 (逐组): %s" % blk["cone_ratio_per_group"])
    L.append("")
    L.append("- R3 闸: %s" % r["R3_counterfactual_gate"]["gate"])
    L.append("")
    # 建议
    L.append("## 6. 观测建议 (v0, 谓词驱动)")
    L.append("")
    L.append("| 建议 | 触发 | 动作 |")
    L.append("|---|---|---|")
    for rec in r["recommendations"]:
        L.append("| %s | %s | %s |" % (
            rec["id"], "✔ 触发" if rec["triggered"] else "✘ 未触发(谓词假, 不装饰)",
            rec["action"]))
    L.append("")
    L.append("每条建议 = 解析触发谓词 + 动作 + 反事实可消性声明; 未触发的建议如实落名, 禁恒真装饰。")
    L.append("")
    # 假设
    L.append("## 7. 假设与诚实边界")
    L.append("")
    for a in r["assumptions"]:
        L.append("- %s" % a)
    L.append("- 裂隙位置为随机实现 (Baecher 盘), 非实测定位; 本报告为研究线原型, "
             "禁入对外承诺文本。")
    L.append("")
    return "\n".join(L) + "\n"
