# -*- coding: utf-8 -*-
"""R288 观测成本顾问 v1 —— 纯检索内核（无预测层）。

回答一个问题: **给定场地相对临界强度的位置, 连通/不通筛查要判对, 最少多少观测、
几口整孔编录井; 以及在什么情况下必须拒绝硬判。**

设计约束（唯一实现依据 = results/r288_obs_advisor/prereg.json）:
  * 全部读数来自 results/v_r47_cost_curve.json 的 68 个已实测 cell —— **检索, 不重算**;
  * 全部阈值来自 prereg 解析值（其上游为 R47 预注册件文字字段）—— **零内联**;
  * **禁插值**: |ln r| 相近的两档所需井数差一倍, 难度对 |ln r| 不单调, 故连续 r 只回落实测档;
  * 零 MC / 零训练 / 零新指标 / 零随机数消耗。

对外承诺: 无。本模块产出的是成本与弃权判定, 不是对某块场地的精度保证。
"""
from __future__ import annotations

import json
import math
import os
import re

__all__ = ["LookupError_", "ObservationCostAdvisor", "load_curve"]

# ---------------------------------------------------------------------------
# 只读输入位的文件名（路径本身不是度量值）
# ---------------------------------------------------------------------------
CURVE_REL = "results/v_r47_cost_curve.json"
PREREG_REL = "results/r288_obs_advisor/prereg.json"


class LookupError_(ValueError):
    """检索失败：输入落在已实测网格之外且无保守回落, 或输入非法。"""


def _default_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_curve(root=None):
    """运行时直读 R47 成本曲线件（不做任何加工）。"""
    root = root or _default_root()
    with open(os.path.join(root, CURVE_REL), encoding="utf-8") as f:
        return json.load(f)


def load_prereg(root=None):
    root = root or _default_root()
    with open(os.path.join(root, PREREG_REL), encoding="utf-8") as f:
        return json.load(f)


def _parse_threshold(pre, key):
    """从 prereg 的 criteria 块取解析值; 缺失即硬停（阈值不许内联于代码）。"""
    crit = pre.get("criteria_read_from_upstream") or {}
    item = crit.get(key)
    if not item or item.get("value") is None:
        raise SystemExit("[obs_advisor] prereg 缺阈值 %r —— 禁止内联, 请检查 %s" % (key, PREREG_REL))
    return item["value"]


def _parse_grid_hit_tolerance(pre):
    """r 与实测档『同一格』的容差: 从裁决 D-3 文字解析, 不落字面量。"""
    for r in pre.get("rulings_and_conflicts", []):
        if r.get("id") != "D-3":
            continue
        m = re.search(r"(\d(?:\.\d+)?[eE][-+]?\d+)", r.get("ruling", ""))
        if m:
            return float(m.group(1))
    raise SystemExit("[obs_advisor] 裁决 D-3 容差解析失败 —— 禁止内联")


def _as_float(v):
    return float(v)


# ---------------------------------------------------------------------------
# 内核
# ---------------------------------------------------------------------------
class ObservationCostAdvisor(object):
    """对 R47 已算结果的检索器。

    典型用法::

        adv = ObservationCostAdvisor()
        out = adv.advise(p32_interval=(0.30, 0.36))       # 强度区间, 临界值用内置参考系
        out = adv.advise(p32=0.31, crit=0.2405, target=0.95)
    """

    # 每档交付读数的字段集（全部是 cell 里的在册位点, 不新建名字）
    ROW_KEYS = ("W", "n_wells", "n_obs_mean", "n_obs_q10", "acc_strict", "acc_conf",
                "conf_coverage", "ceiling_site", "forced_abstain_rate", "iv_acc_strict",
                "cell_id")

    def __init__(self, root=None, curve=None, prereg=None):
        self.root = root or _default_root()
        self.curve = curve if curve is not None else load_curve(self.root)
        self.pre = prereg if prereg is not None else load_prereg(self.root)

        crit = self.pre["criteria_read_from_upstream"]
        self.target_default = _as_float(_parse_threshold(self.pre, "target_acc_site_strict"))
        self.band_halfwidth_ln = _as_float(_parse_threshold(self.pre, "abstain_band_ln_halfwidth"))
        self.min_obs_forced_abstain = int(_parse_threshold(self.pre, "forced_abstain_min_obs"))
        self.grid_hit_tol = _parse_grid_hit_tolerance(self.pre)

        # 档位表：prereg 解析的 names/mults 必须与曲线件 meta.levels 逐位相符，否则硬停
        names = list(_parse_threshold_list(self.pre, "level_names"))
        mults = [float(x) for x in _parse_threshold_list(self.pre, "level_mults")]
        meta_levels = self.curve["meta"]["levels"]
        if [m["name"] for m in meta_levels] != names or [float(m["mult"]) for m in meta_levels] != mults:
            raise SystemExit("[obs_advisor] prereg 档位与曲线件 meta.levels 不符 —— 拒用")
        self.level_names = names
        self.level_mults = mults
        self.p32_target_of_level = {m["name"]: float(m["p32_target"]) for m in meta_levels}

        self.far_levels = list(_parse_threshold_list(self.pre, "far_levels"))
        self.near_levels = list(_parse_threshold_list(self.pre, "near_levels"))
        self.W_grid = [int(x) for x in _parse_threshold_list(self.pre, "grid_W_wells")]
        self.f_grid = [float(x) for x in _parse_threshold_list(self.pre, "grid_f_coverage")]

        calib = self.curve["meta"]["calibration"]
        self.crit_ref = float(calib["p32_crit_true"])
        self.anchor = dict(crit["frozen_anchor_p32_crit"]["value"])

        self._cells = self._index_cells()
        self._wells = self._axis_table("wells", "wholewell")
        self._cov = self._axis_table("coverage", None)
        self.f_star = dict(self.curve["f_star"])
        self.honest_boundary = list(self.curve["honest_boundary"])
        self._assert_grid_matches_prereg()

    def _assert_grid_matches_prereg(self):
        """曲线件里实际存在的格点必须与 prereg 网格逐位相符, 否则检索面不完整 = 硬停。"""
        got_w = sorted({r["W"] for lv in self.level_names for r in self._wells[lv]})
        got_f = sorted({r["f"] for lv in self.level_names for r in self._cov[lv]})
        if got_w != sorted(self.W_grid):
            raise SystemExit("[obs_advisor] wells 格点与 prereg W_grid 不符: %r vs %r"
                             % (got_w, sorted(self.W_grid)))
        if got_f != sorted(self.f_grid):
            raise SystemExit("[obs_advisor] coverage 格点与 prereg f_grid 不符: %r vs %r"
                             % (got_f, sorted(self.f_grid)))
        empty = [lv for lv in self.level_names if not self._wells[lv] or not self._cov[lv]]
        if empty:
            raise SystemExit("[obs_advisor] 以下档位缺格: %r" % empty)

    # ---------------- 索引（检索面的建立, 无任何数值加工） ----------------
    def _index_cells(self):
        cells = {}
        for c in self.curve["cells"]:
            cells[(c["level"], c["axis"], c["mode"], float(c["x"]))] = c
        return cells

    def _axis_table(self, axis, mode):
        """level -> [row...]，按 x 升序; row 只搬运 cell 内在册位点。"""
        table = {name: [] for name in self.level_names}
        for (level, ax, md, x), c in self._cells.items():
            if ax != axis or (mode is not None and md != mode):
                continue
            pa = c["p10_axis"]
            table[level].append({
                "W": int(x) if axis == "wells" else None,
                "f": x if axis == "coverage" else None,
                "n_wells": int(x) if axis == "wells" else None,
                "n_obs_mean": float(c["n_obs_mean"]),
                "n_obs_q10": float(c["n_obs_q10"]),
                "acc_strict": float(pa["acc_site_strict"]),
                "acc_conf": float(pa["acc_site_conf"]),
                "conf_coverage": float(pa["conf_coverage"]),
                "ceiling_site": float(c["ceiling_site"]),
                "forced_abstain_rate": float(c["forced_abstain_rate"]),
                "iv_acc_strict": float(c["interval_inventory"]["acc_site_strict"]),
                "cell_id": c["cell_id"],
                "mode": md,
                "level_mult": float(c["level_mult"]),
                "p_span": float(c["p_span"]),
                "n_ok": int(c["n_ok"]),
                "n_reps": int(c["n_reps"]),
            })
        for name in table:
            table[name].sort(key=lambda r: (r["W"] if axis == "wells" else r["f"]))
        return table

    # ---------------- r 的定义与档位归属 ----------------
    def ratio(self, p32, crit=None):
        crit = self.crit_ref if crit is None else float(crit)
        if crit <= 0 or p32 <= 0:
            raise LookupError_("P32 与临界强度必须为正 (收到 p32=%r crit=%r)" % (p32, crit))
        return float(p32) / crit

    def classify(self, r):
        """r -> 档位归属（最近实测档 + 相邻两侧 + 是否出格/命中格点）。"""
        if r <= 0:
            raise LookupError_("r 必须为正")
        ln = math.log(r)
        dists = [(abs(ln - math.log(m)), i) for i, m in enumerate(self.level_mults)]
        dists.sort()
        nearest_i = dists[0][1]
        exact = dists[0][0] <= self.grid_hit_tol
        ordered = sorted(range(len(self.level_mults)), key=lambda i: self.level_mults[i])
        if exact:
            # 对数空间重合即视为该实测档本身: 否则 r 的浮点末位 (1 ulp) 会把答案
            # 从"取该档读数"翻转成"与邻档取更苛刻者", 属噪声驱动结论。
            bracket = [nearest_i]
        else:
            below = [i for i in ordered if self.level_mults[i] <= r]
            above = [i for i in ordered if self.level_mults[i] >= r]
            bracket = []
            if below:
                bracket.append(below[-1])
            if above and above[0] != (below[-1] if below else None):
                bracket.append(above[0])
        return {
            "r": r,
            "ln_r": ln,
            "nearest_level": self.level_names[nearest_i],
            "nearest_mult": self.level_mults[nearest_i],
            "nearest_log_distance": dists[0][0],
            "exact_grid_hit": exact,
            "bracket_levels": [self.level_names[i] for i in bracket],
            "bracket_mults": [self.level_mults[i] for i in bracket],
            "beyond_measured_grid": bool(r < min(self.level_mults) or r > max(self.level_mults)),
            "beyond_side": ("below_grid" if r < min(self.level_mults)
                            else "above_grid" if r > max(self.level_mults) else None),
            "stratification_group": ("far_from_critical" if self.level_names[nearest_i] in self.far_levels
                                     else "near_critical"),
        }

    def in_abstain_band(self, r_lo, r_hi=None):
        """弃权门 1：查询区间与冻结弃权带 [|ln r| < halfwidth] 是否相交。"""
        r_hi = r_lo if r_hi is None else r_hi
        lo, hi = min(r_lo, r_hi), max(r_lo, r_hi)
        return {"band_halfwidth_ln": self.band_halfwidth_ln,
                "band_r_lo": math.exp(-self.band_halfwidth_ln),
                "band_r_hi": math.exp(self.band_halfwidth_ln),
                "query_r_lo": lo, "query_r_hi": hi,
                "intersects": (lo <= math.exp(self.band_halfwidth_ln))
                              and (hi >= math.exp(-self.band_halfwidth_ln)),
                "rule": "abstain_gate_1_band"}

    # ---------------- 单档检索答案 ----------------
    def level_answer(self, level, target=None, axis="wells"):
        """在该档已实测的格点上找**最小**满足 target 的格; 找不到即不可达标。"""
        target = self.target_default if target is None else float(target)
        rows = self._wells[level] if axis == "wells" else self._cov[level]
        mult = self.level_mults[self.level_names.index(level)]
        out = {"level": level, "level_mult": mult,
               "target": target, "axis": axis,
               "curve": rows, "ceiling_site": rows[0]["ceiling_site"] if rows else None,
               "best_acc_strict": max((r["acc_strict"] for r in rows), default=None),
               "best_cell_id": None}
        for r in rows:
            if out["best_cell_id"] is None or r["acc_strict"] == out["best_acc_strict"]:
                out["best_cell_id"] = r["cell_id"]
        reached = [r for r in rows if r["acc_strict"] >= target]
        # 弃权门 3: 推荐格的观测条数不得低到触发强制弃权
        eligible = [r for r in reached if r["n_obs_mean"] >= self.min_obs_forced_abstain]
        if not eligible:
            out.update({"reachable": False, "W_star": None, "f_star": None,
                        "row_at_star": None,
                        "why": (("ceiling_site (%g) < target (%g): 属物理不可辨识, 不是观测质量问题"
                                 % (out["ceiling_site"], target)) if not reached else
                                ("达标格无一满足最小观测条数门 (forced_abstain, N_obs < %d)"
                                 % self.min_obs_forced_abstain))})
            return out
        star = min(eligible, key=lambda r: (r["W"] if axis == "wells" else r["f"]))
        out.update({"reachable": True,
                    "W_star": star["W"] if axis == "wells" else None,
                    "f_star": star["f"] if axis == "coverage" else None,
                    "row_at_star": star,
                    "n_obs_mean_at_star": star["n_obs_mean"],
                    "n_obs_q10_at_star": star["n_obs_q10"],
                    "forced_abstain_min_obs_gate": self.min_obs_forced_abstain})
        return out

    def _hardness_key(self, ans):
        """苛刻序（max 语义）: 不可达标 > 所需格更大; 平手感时取更靠近临界的一侧（更保守）。"""
        if not ans["reachable"]:
            return (1, float("inf"), 0.0)
        return (0, ans["W_star"], -abs(math.log(ans["level_mult"])))

    # ---------------- 交付接口 ----------------
    def rows(self, level, axis="wells"):
        """某档在已实测格点上的全部在册读数（升序）; 只搬运 cell, 不加工。"""
        src = self._wells if axis == "wells" else self._cov
        if level not in src:
            raise LookupError_("未知档位 %r (在册: %r)" % (level, list(self.level_names)))
        return [dict(r) for r in src[level]]

    def cell_of(self, cell_id):
        """cell_id -> (在 cells 数组中的位置, cell)；位置是溯源指针的一部分。"""
        for i, c in enumerate(self.curve["cells"]):
            if c.get("cell_id") == cell_id:
                return i, c
        raise LookupError_("cell_id 不在册: %r" % cell_id)

    def advise(self, p32=None, p32_interval=None, crit=None, target=None):
        """主入口。p32 或 p32_interval 二选一; crit 省略时用 R47 自标定参考系。"""
        if (p32 is None) == (p32_interval is None):
            raise LookupError_("必须且只能给 p32 或 p32_interval 之一")
        target = self.target_default if target is None else float(target)
        crit_used = self.crit_ref if crit is None else float(crit)
        pair = [float(p32), float(p32)] if p32 is not None else [float(p32_interval[0]),
                                                                 float(p32_interval[1])]
        if min(pair) <= 0:
            raise LookupError_("P32 区间必须为正")
        r_lo, r_hi = sorted(self.ratio(v, crit_used) for v in pair)

        band = self.in_abstain_band(r_lo, r_hi)
        # 区间内最靠近临界的一端（最小 |ln r|）定档
        r_worst = min((r_lo, r_hi), key=lambda x: abs(math.log(x)))
        cls = self.classify(r_worst)

        result = {
            "query": {"p32_lo": min(pair), "p32_hi": max(pair),
                      "crit_used": crit_used,
                      "crit_reference_origin": ("R47 自标定 crit_true (档位定义的母体)"
                                                if crit is None else "caller-supplied"),
                      "target": target,
                      "target_is_r47_prereg_default": target == self.target_default,
                      "r_lo": r_lo, "r_hi": r_hi, "r_worst": r_worst},
            "abstain_band_check": band,
            "classification": cls,
            "frozen_anchor_p32_crit": self.anchor["p32_crit"],
            "frozen_anchor_p32_crit_lower": self.anchor["p32_crit_lower"],
            "frozen_anchor_p32_crit_upper": self.anchor["p32_crit_upper"],
            "reference_vs_anchor_abs_diff": abs(crit_used - float(self.anchor["p32_crit"])),
            "watermark": self.pre["watermark"],
        }

        if band["intersects"]:
            result["verdict"] = "REFUSE_HARD_JUDGMENT"
            at_crit = self.level_answer("at_crit", target=target)
            result["reason"] = ("场地强度区间与预注册弃权带相交: |ln(P32/crit)| < %g "
                                "(= ln %g), 按产品冻结语义输出『不确定』而非结论。"
                                % (self.band_halfwidth_ln,
                                   _as_float(_parse_threshold(self.pre, "abstain_band_ratio"))))
            result["bayesian_ceiling_of_at_crit_level"] = at_crit["ceiling_site"]
            result["honest_boundary_refs"] = ["HB-3"]
            result["recommendation"] = None
            result["next_step"] = ("本方法无法在该档位给出判级成本; 需换证据档位 "
                                   "(如稠密点云/成像段或增加独立场地证据), 而非单纯加井。")
            return result

        cand = [self.level_answer(lv, target=target) for lv in cls["bracket_levels"]] or \
               [self.level_answer(cls["nearest_level"], target=target)]
        governing = max(cand, key=self._hardness_key)
        result["bracket_answers"] = [self._slim(c) for c in cand]
        result["governing_level_answer"] = self._slim(governing)
        result["governing_rule"] = ("exact_grid_hit" if cls["exact_grid_hit"]
                                    else "harder_of_bracket")
        result["beyond_measured_grid"] = cls["beyond_measured_grid"]
        result["beyond_side"] = cls["beyond_side"]

        cov = {mode: self._best_coverage(governing["level"], target) for mode in ("point", "window")}
        result["coverage_axis"] = cov
        result["coverage_f_star_read_directly"] = {
            "well_mode": self.f_star["well_mode"], "image_mode": self.f_star["image_mode"]}

        if not governing["reachable"]:
            result["verdict"] = "NOT_ACHIEVABLE_AT_MEASURED_CELLS"
            result["reason"] = governing["why"]
            result["honest_boundary_refs"] = ["HB-3"]
            result["recommendation"] = None
            return result

        row = governing["row_at_star"]
        result["verdict"] = "ACHIEVABLE_AT_MEASURED_CELLS"
        result["honest_boundary_refs"] = ["HB-1", "HB-2", "HB-3", "HB-4"]
        result["recommendation"] = {
            "axis": "wells/wholewell (整孔全编录)",
            "n_wells_star": row["W"],
            "acc_site_strict_at_star": row["acc_strict"],
            "acc_site_conf_at_star": row["acc_conf"],
            "conf_coverage_at_star": row["conf_coverage"],
            "n_obs_mean_at_star": row["n_obs_mean"],
            "n_obs_q10_at_star": row["n_obs_q10"],
            "forced_abstain_rate_at_star": row["forced_abstain_rate"],
            "naive_catalog_arm_iv_acc_at_star": row["iv_acc_strict"],
            "cell_id": row["cell_id"],
            "ceiling_site": row["ceiling_site"],
        }
        result["marginal_cost_along_level"] = self._marginal(governing["level"], row["W"])
        result["no_accuracy_promise"] = (
            "本栏是 R47 已实测格点的检索结果 (每格 %d 次重复 × %d 场地/档), "
            "不是对具体场地的精度承诺; 场地需自带其 P32 与临界强度的可比口径。"
            % (governing["curve"][0]["n_reps"], _as_int(_parse_threshold(self.pre, "n_sites_per_level")))
        )
        return result

    def _best_coverage(self, level, target):
        """覆盖率轴：本档在预注册 f_grid 内的最好成绩与是否达标（不达标即 f* 不存在）。"""
        per_mode = {}
        for mode in ("point", "window"):
            ans = self.level_answer(level, target=target, axis="coverage")
            per_mode[mode] = {"best_acc_strict": ans["best_acc_strict"],
                              "reachable_within_prereg_f_grid": ans["reachable"],
                              "f_star": ans.get("f_star"),
                              "ceiling_site": ans["ceiling_site"],
                              "curve": [{"f": r["f"], "acc_strict": r["acc_strict"],
                                         "n_obs_mean": r["n_obs_mean"],
                                         "forced_abstain_rate": r["forced_abstain_rate"],
                                         "cell_id": r["cell_id"]} for r in ans["curve"]]}
        return per_mode

    def _marginal(self, level, w_star):
        """『少打一口井掉几个点』——同一档相邻实测格的判对率差, 纯差值派生。"""
        rows = list(self._wells[level])
        out = []
        for a, b in zip(rows, rows[1:]):
            pick = lambda r: {"cell_id": r["cell_id"], "W": r["W"],
                              "acc_strict": r["acc_strict"], "n_obs_mean": r["n_obs_mean"]}
            out.append({
                "from": pick(a), "to": pick(b),
                "delta_acc": b["acc_strict"] - a["acc_strict"],
                "delta_wells": b["W"] - a["W"],
                "delta_acc_per_well": (b["acc_strict"] - a["acc_strict"]) / float(b["W"] - a["W"]),
                "at_or_below_recommendation": (None if w_star is None else b["W"] <= w_star),
            })
        return out

    def marginal(self, level, w_star=None):
        """公开入口: 同档相邻实测格的边际成本表（见 _marginal）。"""
        return self._marginal(level, w_star)

    def _slim(self, ans):
        return {k: v for k, v in ans.items() if k != "curve"}

    def lookup_table(self, target=None):
        """交付主体的机读查表件：逐实测档 × 两轴。"""
        target = self.target_default if target is None else float(target)
        table = {}
        for level in self.level_names:
            wells = self.level_answer(level, target=target, axis="wells")
            table[level] = {
                "level_mult": wells["level_mult"],
                "p32_target_at_this_level": self.p32_target_of_level[level],
                "stratification_group": ("far_from_critical" if level in self.far_levels
                                         else "near_critical"),
                "wells_axis": {
                    "target": target,
                    "reachable": wells["reachable"],
                    "W_star": wells["W_star"],
                    "acc_strict_at_star": (wells["row_at_star"]["acc_strict"]
                                           if wells["reachable"] else None),
                    "n_obs_mean_at_star": wells.get("n_obs_mean_at_star"),
                    "n_obs_q10_at_star": wells.get("n_obs_q10_at_star"),
                    "conf_acc_at_star": (wells["row_at_star"]["acc_conf"]
                                         if wells["reachable"] else None),
                    "iv_naive_arm_acc_at_star": (wells["row_at_star"]["iv_acc_strict"]
                                                 if wells["reachable"] else None),
                    "ceiling_site": wells["ceiling_site"],
                    "best_acc_strict_on_grid": wells["best_acc_strict"],
                    "curve": [{k: r[k] for k in
                               ("W", "n_obs_mean", "n_obs_q10", "acc_strict", "acc_conf",
                                "conf_coverage", "forced_abstain_rate", "iv_acc_strict",
                                "ceiling_site", "cell_id")} for r in wells["curve"]],
                    "why_if_unreachable": wells.get("why"),
                },
                "coverage_axis": self._best_coverage(level, target),
                "marginal_cost": self._marginal(level, wells.get("W_star")),
            }
        return table


def _parse_threshold_list(pre, key):
    crit = pre.get("criteria_read_from_upstream") or {}
    item = crit.get(key)
    if not item or item.get("value") is None:
        raise SystemExit("[obs_advisor] prereg 缺字段 %r —— 禁止内联" % key)
    return item["value"]


def _as_int(v):
    return int(v)
