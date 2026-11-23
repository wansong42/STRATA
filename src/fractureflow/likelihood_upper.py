# -*- coding: utf-8 -*-
"""S2-U · 区域似然上界（闭式乘积）的精确实现。

数学命题（外部复核件 claim，见 proof_pointer）：
    sup_{F1 ⊆ S ⊆ F1 ∪ U} [Z^Y] Π_{i∈S} F_i  ≤  [Z^Y] Π_{i∈F1} F_i · Π_{i∈U} G_i
其中 G_i = F_i 把常数项（EMPTY 结局系数）置 1；所有系数非负。

实现要点：
  * 逐盘结局表由 kernel.disc_outcome_table 提供（预测侧同定义枚举）；本模块不做枚举；
  * 稀疏计数 DP：按 Y 的计数向量逐分量截断，一次相乘即得目标系数；
  * 全部概率/系数用 fractions.Fraction；判定路径零 float（float 仅作展示字段）；
  * 只证上界有效性：本函数不是 likelihood 点值，禁入对外口径。

本文件为 S2-U 新模块；不修改 observation_kernel.py 任何字节。
"""
from __future__ import annotations

import math
from fractions import Fraction

EMPTY_OUTCOME = "EMPTY"

FORMULA = ("sup_{F1 subset S subset F1 cup U} [Z^Y] prod_{i in S} F_i  <=  "
           "[Z^Y] prod_{i in F1} F_i prod_{i in U} G_i,  G_i = F_i with constant term set to 1")
PROOF = ("all coefficients are non-negative; each free disc's two choices (present: F_i; absent: 1) "
         "are dominated coefficient-wise by G_i; coefficient-wise domination of non-negative "
         "polynomials is preserved under multiplication; taking [Z^Y] is monotone")
ALLOWED_RECORD_KEYS = frozenset((
    "design_id", "well_id", "exposure_segment", "quantization", "counts",
    "detected_total", "detected", "kernel_id"))


class RegionError(ValueError):
    """区域表示非法。"""


class PublicRecordError(ValueError):
    """公开记录非法。"""


def _int_face(value, name: str, d: int):
    if not isinstance(value, (list, tuple)):
        raise RegionError("region[%r] 必须为 int 列表" % name)
    out = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, int):
            raise RegionError("region[%r] 含非 int: %r" % (name, x))
        if x < 0 or x >= d:
            raise RegionError("region[%r] 下标越界: %r (d=%d)" % (name, x, d))
        out.append(int(x))
    if len(set(out)) != len(out):
        raise RegionError("region[%r] 含重复下标" % name)
    return out


def parse_region(region, d: int) -> dict:
    """严格解析 {'F1': [...], 'U': [...]}；其余盘补为已知缺席 F0。"""
    if not isinstance(region, dict):
        raise RegionError("region 必须为 dict")
    if set(region.keys()) != {"F1", "U"}:
        raise RegionError("region 的 key 必须恰为 {'F1', 'U'}，现为 %r" % sorted(region.keys()))
    f1 = _int_face(region["F1"], "F1", d)
    u = _int_face(region["U"], "U", d)
    if set(f1) & set(u):
        raise RegionError("F1 与 U 必须不相交: 交集 %r" % sorted(set(f1) & set(u)))
    used = set(f1) | set(u)
    f0 = [i for i in range(d) if i not in used]
    return {"F1": f1, "U": u, "F0": f0}


def _parse_counts(public_record: dict) -> dict:
    if not isinstance(public_record, dict):
        raise PublicRecordError("public_record 必须为 dict")
    extra = set(public_record.keys()) - set(ALLOWED_RECORD_KEYS)
    if extra:
        raise PublicRecordError("public_record 含未声明键: %r" % sorted(extra))
    missing = {"counts", "detected_total", "detected", "kernel_id",
               "design_id", "well_id", "exposure_segment", "quantization"} - set(public_record)
    if missing:
        raise PublicRecordError("public_record 缺键: %r" % sorted(missing))
    counts = public_record["counts"]
    if not isinstance(counts, list):
        raise PublicRecordError("public_record.counts 必须为 list")
    out = {}
    last = -1
    for i, item in enumerate(counts):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PublicRecordError("counts[%d] 必须为 [cell_id, count]" % i)
        cell, cnt = item
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise PublicRecordError("counts[%d] cell_id 非法: %r" % (i, cell))
        if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt <= 0:
            raise PublicRecordError("counts[%d] count 必须为正 int: %r" % (i, cnt))
        if cell <= last:
            raise PublicRecordError("counts 必须按 cell_id 严格升序（现 %d 在 %d 后）" % (cell, last))
        last = cell
        out[cell] = cnt
    if public_record["detected_total"] != sum(out.values()):
        raise PublicRecordError("detected_total != sum(counts)")
    if bool(public_record["detected"]) != bool(out):
        raise PublicRecordError("detected 与 counts 非空性不一致")
    return out


def _factor_poly(kernel, action_spec, disc_index: int, tier: str, free: bool,
                 cell_pos: dict):
    """逐盘因子：返回 {计数向量: 系数}；free=True 时 G_i（常数项置 1）。"""
    table = kernel.disc_outcome_table(action_spec, disc_index, tier)
    if not isinstance(table, dict) or not table:
        raise PublicRecordError("disc_outcome_table[%d] 空/非 dict" % disc_index)
    n = len(cell_pos)
    zero = tuple(0 for _ in range(n))
    poly = {}
    total = Fraction(0)
    for outcome, p in table.items():
        prob = Fraction(p)
        if prob < 0:
            raise PublicRecordError("disc_outcome_table[%d] 含负系数" % disc_index)
        total += prob
        if outcome == EMPTY_OUTCOME:
            if free:
                continue
            poly[zero] = poly.get(zero, Fraction(0)) + prob
        else:
            pos = cell_pos.get(int(outcome))
            if pos is None:
                continue  # 该格在 Y 中计数上限为 0，任何出现都被截断
            vec = [0] * n
            vec[pos] = 1
            key = tuple(vec)
            poly[key] = poly.get(key, Fraction(0)) + prob
    if free:
        poly[zero] = Fraction(1)
    if total != 1:
        raise PublicRecordError("disc_outcome_table[%d] 概率和 != 1: %s" % (disc_index, total))
    return poly


def _dp_coefficient(kernel, action_spec, record_cells, region_parsed, tier):
    """对区域立方体上界做一次计数 DP；返回 (coef, stats)。"""
    cells = list(record_cells)
    caps = tuple(record_cells[c] for c in cells)
    cell_pos = {c: i for i, c in enumerate(cells)}
    zero = tuple(0 for _ in cells)
    factors = []
    for i in region_parsed["F1"]:
        factors.append(_factor_poly(kernel, action_spec, i, tier, False, cell_pos))
    for i in region_parsed["U"]:
        factors.append(_factor_poly(kernel, action_spec, i, tier, True, cell_pos))
    dp = {zero: Fraction(1)}
    mult = 0
    states_max = 1
    for fac in factors:
        ndp = {}
        for partial, p0 in dp.items():
            for monomial, c in fac.items():
                mult += 1
                if monomial:
                    vec = list(partial)
                    ok = True
                    for pos, inc in enumerate(monomial):
                        if inc == 0:
                            continue
                        pos2 = pos
                        vec[pos2] += inc
                        if vec[pos2] > caps[pos2]:
                            ok = False
                            break
                    if not ok:
                        continue
                    key = tuple(vec)
                else:
                    key = partial
                ndp[key] = ndp.get(key, Fraction(0)) + p0 * c
        dp = ndp
        states_max = max(states_max, len(dp))
        if not dp:
            return Fraction(0), {"dp_multiplications": mult, "dp_states_max": states_max,
                                 "n_factors": len(factors)}
    return dp.get(caps, Fraction(0)), {"dp_multiplications": mult,
                                       "dp_states_max": states_max,
                                       "n_factors": len(factors)}


def _lambda_zero(kernel, tier: str):
    spec = getattr(kernel, "spec", None)
    if not isinstance(spec, dict):
        return True, "fixture_spec_absent"
    try:
        tiers = spec["noise_model"]["noise_tiers"]
        t = tiers[tier]
        fd = t.get("false_detection") or {}
        num = int(fd.get("lambda_numerator", 0))
    except Exception:
        return True, "lambda_field_absent"
    return (num == 0), "lambda_numerator=%d" % num


def _bound_one(kernel, public_record, region_parsed, action_spec, tier):
    cells = _parse_counts(public_record)
    if public_record["kernel_id"] != getattr(kernel, "kernel_id", None):
        raise PublicRecordError("public_record.kernel_id 与现读 kernel_id 不符")
    d = len(kernel.candidate_geometry())
    if len(region_parsed["F1"]) + len(region_parsed["U"]) > d:
        raise RegionError("region 项数超过 d")
    # 让 kernel 自己完成记录/行动/状态校验（state 全 0 时快速返回 0，但校验完整执行）
    if not hasattr(kernel, "record_probability_exact"):
        raise PublicRecordError("kernel 缺 record_probability_exact（无法完成校验）")
    kernel.record_probability_exact(public_record, [0] * d, action_spec, tier=tier)
    coef, st = _dp_coefficient(kernel, action_spec, cells, region_parsed, tier)
    return coef, st, cells


def log_prob_upper(kernel, public_record, region, action, history=None, tier: str = "N1"):
    """声明区域上的联合似然选取最大值的精确上界（内部数学件，非点值）。

    kernel: 具备 disc_outcome_table / record_probability_exact / candidate_geometry 的核实例
    public_record: kernel 公开记录（counts/detected_total/detected/kernel_id/...）
    region: {'F1': [int...], 'U': [int...]}；其余盘为已知缺席 F0
    action: 行动规格（kernel 同 schema）
    history: 可选，[(action_spec, public_record), ...]；每个历史观测独立坐标相乘
    返回 dict：status / bound_fraction / bound_float / bound_log / direction / region /
              evidence / kernel_id / spec_fingerprint / exact_arithmetic / notes
    """
    d = len(kernel.candidate_geometry())
    parsed = parse_region(region, d)
    lam_zero, lam_note = _lambda_zero(kernel, tier)
    base = {
        "kernel_id": getattr(kernel, "kernel_id", None),
        "region": parsed,
        "direction": ">=",
        "exact_arithmetic": True,
        "spec_fingerprint": {
            "kernel_id": getattr(kernel, "kernel_id", None),
            "spec_content_sha12": getattr(kernel, "spec_content_sha12", None),
            "code_sha12": getattr(kernel, "code_sha12", None),
            "action_id": action.get("action_id") if isinstance(action, dict) else None,
            "tier": tier,
        },
    }
    if not lam_zero:
        out = dict(base)
        out.update({
            "status": "UNSUPPORTED_LAMBDA_POSITIVE",
            "bound_fraction": None, "bound_float": None, "bound_log": None,
            "evidence": {"formula": FORMULA, "proof": PROOF,
                         "proof_pointer": "results/architect_main_20261012_dispatch/"
                                          "probe_20261018_review_v2_supplements.json"
                                          "#B_region_likelihood_upper_bound",
                         "lambda_check": lam_note},
            "notes": ["λ>0 分支不承载：闭式上界只证 λ=0（本版 kernel 解析期硬拒 λ>0）。",
                      "本件只作内部上界引用，禁入对外口径；不是 likelihood 点值。"],
        })
        return out

    terms = [("MAIN", public_record, action)]
    if history:
        if not isinstance(history, (list, tuple)):
            raise PublicRecordError("history 必须为 [(action_spec, public_record), ...]")
        for entry in history:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise PublicRecordError("history 每项必须为 (action_spec, public_record)")
            terms.append(("HISTORY", entry[1], entry[0]))

    total = Fraction(1)
    total_mult = 0
    states_max = 0
    terms_ev = []
    all_cells = []
    for label, rec, act in terms:
        coef, st, cells = _bound_one(kernel, rec, parsed, act, tier)
        total *= coef
        total_mult += st["dp_multiplications"]
        states_max = max(states_max, st["dp_states_max"])
        all_cells.append({"term": label, "n_observed_cells": len(cells),
                          "observed_total": sum(cells.values()),
                          "dp_multiplications": st["dp_multiplications"],
                          "dp_states_max": st["dp_states_max"]})
        terms_ev.append(label)

    cube = 1 << len(parsed["U"])
    bound_fraction = "%d/%d" % (total.numerator, total.denominator)
    if total > 0:
        bound_log = math.log(total.numerator) - math.log(total.denominator)
    else:
        bound_log = None
    evidence = {
        "formula": FORMULA,
        "proof": PROOF,
        "proof_pointer": ("results/architect_main_20261012_dispatch/"
                          "probe_20261018_review_v2_supplements.json"
                          "#B_region_likelihood_upper_bound"),
        "superset_cube_size": cube,
        "brute_force_equivalent_dp_passes": cube,
        "dp_multiplications": total_mult,
        "dp_states_max": states_max,
        "region_terms": len(parsed["F1"]) + len(parsed["U"]),
        "free_bits": len(parsed["U"]),
        "known_absent": list(parsed["F0"]),
        "n_observed_cells": all_cells[0]["n_observed_cells"],
        "per_term": all_cells,
        "lambda_check": lam_note,
        "tier": tier,
        "notes_on_arithmetic": "all coefficients are fractions.Fraction; float fields are display-only",
    }
    out = dict(base)
    out.update({
        "status": "OK",
        "bound_fraction": bound_fraction,
        "bound_float": float(total),
        "bound_log": bound_log,
        "evidence": evidence,
        "notes": [
            "这是内部上界，不是 likelihood 点值；禁入对外口径。",
            "适用条件：kernel 逐盘结局表系数非负；λ=0（本版 kernel 解析期硬拒 λ>0）。",
            "G_i = F_i 常数项置 1；上界对区域立方体内每一个 S 同时成立。",
            "返回的 direction='>=' 指 log 空间：log sup_S P(S 作用下的 Y) <= log upper。",
        ],
    })
    return out


__all__ = ["log_prob_upper", "parse_region", "RegionError", "PublicRecordError",
           "FORMULA", "PROOF", "EMPTY_OUTCOME"]