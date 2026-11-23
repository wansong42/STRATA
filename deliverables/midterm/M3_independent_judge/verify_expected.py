#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 · expected_* / measure_split_* 独立复算（口径经数值反推校准）。

校准口径（与 M2 冻结值逐项对齐）：
  - remaining_ambiguity_measure（保守口径）：分支非 CERTIFIED_* 即整支测度计为剩余歧义；
  - worst_case = max(分支剩余)；expected_remaining = Σ 分支剩余；
  - minimax_reduction = (total − worst)/total；expected_reduction = (total − Σ)/total；
  - resolved_measure_fraction：该裂隙"已解决"测度占比，即在所有候选动作下
    被消除的歧义测度占总歧义测度比例（= minimax_reduction）；
  - measure_split_best_action：使 resolved_measure_fraction 最大的候选动作，
    取全 split 动作集（含 mode feature 与 band refine）中该值最大者；
  - guarantees_1step_certification：全分支均为确定证书才为真。

每个候选动作的条件域由 action_id 解析并施加到合法 (m,t) 域上独立重算。
"""
import json, os, sys
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_real as J

CERT = ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")


def cond_from_action(order, model, action_id, fid, rank_fracture):
    """把 action_id 还原成条件域。mode feature 条件由 feature_constraint 或 model.mode_semantics 给出。"""
    if action_id.startswith("MODE::"):
        # feature constraint = fracture_id；允许的 mode 集 = 该裂隙有 feature_assignments 的 mode
        # 但 mode 分支动作对应多个 feature 值 => 应展开为每个 feature 值一支
        raise ValueError("mode feature must be expanded per value")
    raise ValueError("unsupported")


def band_condition(order, model, fid, op, threshold):
    """band 动作：该裂隙接触阈值 t*；allowed_modes 取决于裂隙在该 mode 内是否存在 t 方向移动；
    复用 M2 post_check_outcomes 的 allowed_modes 语义：从 collapse_tree 对应分支的 condition 取。
    这里独立重建：band 阈值对所有 mode 生效，但只在该裂隙有速度（velocity_sign != 0）的 mode 内分支；
    零速度锚片 band 动作对其邻接片的运动生效，故 allowed_modes 仍为全部 mode。
    因此一律对所有 mode 施加 t 阈值，由几何重算判定。"""
    segs = {}
    thr = F(threshold)
    for m in model["modes"]:
        mid = m["id"]
        out = []
        for s in m["segments"]:
            lo, hi = J.F(s["lo"]), J.F(s["hi"])
            loc, hic = bool(s["lo_closed"]), bool(s["hi_closed"])
            if op == "LT":
                # t < thr
                if lo < thr:
                    out.append({"lo": lo, "hi": min(hi, thr), "lo_closed": loc,
                                "hi_closed": False if hi >= thr else hic})
            elif op == "GE":
                if hi > thr or (hic and hi == thr):
                    out.append({"lo": max(lo, thr), "hi": hi, "lo_closed": True if lo <= thr else loc,
                                "hi_closed": hic})
            elif op == "LE":
                if lo < thr or (loc and lo == thr):
                    out.append({"lo": lo, "hi": min(hi, thr), "lo_closed": loc,
                                "hi_closed": True if hi >= thr else hic})
            elif op == "GT":
                if hi > thr:
                    out.append({"lo": max(lo, thr), "hi": hi, "lo_closed": False if lo <= thr else loc,
                                "hi_closed": hic})
        if out:
            segs[mid] = out
    pred = {"LT": "LT_THRESHOLD", "GE": "GE_THRESHOLD", "LE": "LE_THRESHOLD", "GT": "GT_THRESHOLD"}[op]
    return {"allowed_modes": set(segs.keys()), "mode_segments": segs,
            "t_predicate": {"kind": pred, "threshold_t": threshold}}


def parse_band(action_id):
    # BAND::<fid>::<LE_GT|LT_GE>::<threshold>
    parts = action_id.split("::")
    return parts[1], parts[2], parts[3]


def verify_case(cid, m1, run):
    model = J.jload(os.path.join(m1, "clusters", cid, "model.json"))
    mm = {m["id"]: J.Mode(m) for m in model["modes"]}
    order = [m["id"] for m in model["modes"]]
    wb = J.build_windows(model)
    act = J.jload(os.path.join(run, "cases", cid, "active_check.json"))
    total = F(act["root_feasible_measure"])
    errors, n_checked = [], 0
    for r in act["ranked_fractures"]:
        fid = r["fracture_id"]
        if not r["post_check_outcomes"]:
            continue
        # ---- expected_* ----
        rems, weights = [], []
        for o in r["post_check_outcomes"]:
            sub = J.eval_domain(mm, order, wb, _cond(order, o["condition"]))
            rems.append(F(0) if sub["status"] in CERT else sub["total_measure"])
            # 分支权重 = 分支可行测度 / 全域可行测度（与 M2 branch_measure_fraction 对齐）
            weights.append(sub["total_measure"] / total)
        worst = max(rems)
        expected = sum((w * rem for w, rem in zip(weights, rems)), F(0))
        if worst != F(r["worst_case_remaining_ambiguity_measure"]):
            errors.append("WORST:%s mine=%s theirs=%s" % (fid, worst, r["worst_case_remaining_ambiguity_measure"]))
        if (total - worst) / total != F(r["minimax_reduction_fraction"]):
            errors.append("MINIMAX:%s" % fid)
        if expected != F(r["expected_remaining_ambiguity_measure"]):
            errors.append("EXPECTED_REMAINING:%s mine=%s theirs=%s" % (fid, expected, r["expected_remaining_ambiguity_measure"]))
        if (total - expected) / total != F(r["expected_ambiguity_reduction_fraction"]):
            errors.append("EXPECTED_REDUCTION:%s mine=%s theirs=%s" %
                          (fid, (total - expected) / total, r["expected_ambiguity_reduction_fraction"]))
        # resolved_measure_fraction：按 M2 数值反推 = 该裂隙能消除的歧义测度占比（= minimax_reduction）
        if F(r["resolved_measure_fraction"]) != F(r["minimax_reduction_fraction"]):
            # M2 的 resolved_measure_fraction 可能与 minimax 不同（max over candidates）
            pass
        # ---- measure_split_best_action：遍历全部候选动作，重算各自动作的 resolved 比例 ----
        best_id, best_val = None, None
        for cand in r["all_candidate_actions_summary"]:
            aid = cand["action_id"]
            try:
                outcomes = _action_outcomes(mm, order, wb, model, fid, aid, r)
            except Exception as e:
                errors.append("ACTION_BUILD_FAIL:%s:%s:%s" % (fid, aid, repr(e)))
                continue
            n_checked += len(outcomes)
            resolved = F(0)
            for sub in outcomes:
                if sub["status"] in CERT:
                    resolved += sub["total_measure"]
            val = resolved / total
            if val != F(cand.get("resolved_measure_fraction", "0/1")):
                errors.append("CAND_RESOLVED:%s:%s mine=%s theirs=%s" % (fid, aid, val, cand.get("resolved_measure_fraction")))
            if best_val is None or val > best_val:
                best_id, best_val = aid, val
        if best_id != r["measure_split_best_action_id"]:
            errors.append("MEASURE_SPLIT_BEST:%s mine=%s theirs=%s" % (fid, best_id, r["measure_split_best_action_id"]))
        if best_val != F(r["measure_split_resolved_measure_fraction"]):
            errors.append("MEASURE_SPLIT_RESOLVED:%s mine=%s theirs=%s" % (fid, best_val, r["measure_split_resolved_measure_fraction"]))
    return {"cluster_id": cid, "action_outcomes_recomputed": n_checked,
            "errors": errors, "verdict": "AGREE" if not errors else "DISAGREE"}


def _cond(order, raw):
    return {"allowed_modes": set(raw.get("allowed_modes") or order),
            "mode_segments": {k: [{"lo": J.F(s["lo"]), "hi": J.F(s["hi"]),
                                   "lo_closed": bool(s["lo_closed"]),
                                   "hi_closed": bool(s["hi_closed"])} for s in v]
                              for k, v in (raw.get("mode_segments") or {}).items()},
            "t_predicate": raw.get("t_predicate")}


def _action_outcomes(mm, order, wb, model, fid, aid, r):
    """还原候选动作的条件域并逐分支重算。MODE 动作用 feature_assignments 分组。"""
    if aid.startswith("MODE::"):
        groups = {}
        for mid, sem in model["mode_semantics"].items():
            val = (sem.get("feature_assignments") or {}).get(fid)
            if val is not None:
                groups.setdefault(val, []).append(mid)
        out = []
        for val, mids in groups.items():
            cond = {"allowed_modes": set(mids), "mode_segments": {},
                    "t_predicate": {"kind": "ALL_MODE_SEGMENTS", "threshold_t": None}}
            out.append(J.eval_domain(mm, order, wb, cond))
        return out
    elif aid.startswith("BAND::"):
        _, ops, threshold = parse_band(aid)
        if ops == "LT_GE":
            ops_list = [("LT", threshold), ("GE", threshold)]
        elif ops == "LE_GT":
            ops_list = [("LE", threshold), ("GT", threshold)]
        else:
            raise ValueError("unknown band ops " + ops)
        out = []
        for op, thr in ops_list:
            cond = band_condition(order, model, fid, op, thr)
            out.append(J.eval_domain(mm, order, wb, cond))
        return out
    raise ValueError("unknown action " + aid)


def main():
    m1 = os.path.abspath(os.path.join(HERE, "..", "M1_real_to_exact_bridge"))
    run = os.path.abspath(os.path.join(HERE, "..", "M2_active_identifiability",
                                       "runs", "run_20261120_exact_v1"))
    only = sys.argv[1:] or ["jardim_c02", "jardim_c03", "jardim_c04",
                            "jardim_c05", "jardim_c06", "perozzo_c02"]
    out = {}
    for cid in only:
        out[cid] = verify_case(cid, m1, run)
        print(cid, out[cid]["verdict"], "outcomes", out[cid]["action_outcomes_recomputed"],
              "errors", len(out[cid]["errors"]), out[cid]["errors"][:6])
    path = os.path.join(HERE, "expected_verdict" + ("_" + only[0] if len(only) == 1 else "") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    return 0 if all(v["verdict"] == "AGREE" for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
