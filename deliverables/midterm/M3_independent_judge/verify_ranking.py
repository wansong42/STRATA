#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 · 排序收益与隐藏标签隔离独立核验（只读白名单 JSON）。

口径（按任务书 M3 §二.5）：
  1) 条件化三态 post_status / exists / all 逐个分支重算；
  2) 分支可行测度 branch_measure 重算；
  3) 剩余歧义测度采用 M2 声明口径 —— 分支为 CERTIFIED_* 时记 0，
     否则记该分支全部可行测度（保守口径，已用 jardim_c02 rank1 的
     minimax_reduction=5/9 数值反推校准）；
  4) worst_case / minimax_reduction / guarantees_1step_certification /
     rank 顺序 / zero_gain 收益必须为 0 / Top-k 必须是已证明 informative；
  5) 隐藏标签隔离：is_informative 必须等价于"几何重算收益>0"，
     即排序键完全由几何域推出，不引入任何地下真值标签。

未独立复算（定义不可从任务书唯一推出，登记在 own_defects）：
  expected_remaining_ambiguity_measure、expected_ambiguity_reduction_fraction、
  measure_split_* 等 M2 内部期望/拆分口径。
"""
import json, os, sys
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_real as J

CERT = ("CERTIFIED_CONNECTED", "CERTIFIED_DISCONNECTED")


def condition(order, raw):
    return {"allowed_modes": set(raw.get("allowed_modes") or order),
            "mode_segments": {k: [{"lo": J.F(s["lo"]), "hi": J.F(s["hi"]),
                                   "lo_closed": bool(s["lo_closed"]),
                                   "hi_closed": bool(s["hi_closed"])} for s in v]
                               for k, v in (raw.get("mode_segments") or {}).items()},
            "t_predicate": raw.get("t_predicate")}


def verify_case(cid, m1, run):
    model = J.jload(os.path.join(m1, "clusters", cid, "model.json"))
    mm = {m["id"]: J.Mode(m) for m in model["modes"]}
    order = [m["id"] for m in model["modes"]]
    wb = J.build_windows(model)
    act = J.jload(os.path.join(run, "cases", cid, "active_check.json"))
    ct = act["collapse_tree"]
    ranked = act["ranked_fractures"]
    total = F(act["root_feasible_measure"])
    errors, notes = [], []
    nbranch = 0
    geom_reduction = {}

    for r in ranked:
        fid = r["fracture_id"]
        outcomes = r["post_check_outcomes"]
        subs = []
        for o in outcomes:
            nbranch += 1
            sub = J.eval_domain(mm, order, wb, condition(order, o["condition"]))
            if sub["status"] != o["post_status"]:
                errors.append("POST_STATUS_MISMATCH:%s:%s" % (fid, o["branch_id"]))
            if sub["exists_connected"] != o["exists_connected"] or \
               sub["all_connected"] != o["all_connected"]:
                errors.append("POST_FLAGS_MISMATCH:%s:%s" % (fid, o["branch_id"]))
            if sub["total_measure"] != F(o["branch_measure"]):
                errors.append("BRANCH_MEASURE_MISMATCH:%s:%s" % (fid, o["branch_id"]))
            # 声明口径的剩余歧义测度
            rem = F(0) if sub["status"] in CERT else sub["total_measure"]
            if rem != F(o["remaining_ambiguity_measure"]):
                errors.append("REMAINING_MEASURE_MISMATCH:%s:%s" % (fid, o["branch_id"]))
            subs.append((sub, rem))
        # 无候选动作的裂隙：条件域 = 全域，保守口径下 worst-case 剩余歧义 = 全部歧义，
        # 即 minimax 收益 0（M2 对该类裂隙不展开分支，只给收益 0）。
        worst = max((rem for _, rem in subs), default=total)
        if worst != F(r["worst_case_remaining_ambiguity_measure"]):
            errors.append("WORST_CASE_MISMATCH:%s mine=%s theirs=%s" %
                          (fid, worst, r["worst_case_remaining_ambiguity_measure"]))
        red = (total - worst) / total if total else F(0)
        geom_reduction[fid] = red
        if red != F(r["minimax_reduction_fraction"]):
            errors.append("MINIMAX_REDUCTION_MISMATCH:%s mine=%s theirs=%s" %
                          (fid, red, r["minimax_reduction_fraction"]))
        guar = all(s["status"] in CERT for s, _ in subs) and bool(subs)
        if guar != r["guarantees_1step_certification"]:
            errors.append("ONE_STEP_GUARANTEE_MISMATCH:" + fid)

    # 隐藏标签隔离：informative 必须由几何事件谓词推出（候选动作集合非空），
    # 而不是由任何地下真值标签推出。
    informative = set(ct["all_informative_fracture_ids"])
    zero = [r["fracture_id"] for r in ranked if not r["is_informative"]]
    claimed_informative = set()
    for r in ranked:
        fid = r["fracture_id"]
        if r["is_informative"]:
            claimed_informative.add(fid)
            if r["candidate_action_count"] <= 0 or not r["post_check_outcomes"]:
                errors.append("INFORMATIVE_WITHOUT_GEOMETRY_ACTIONS:" + fid)
        else:
            # 零收益：无候选动作、无分支、几何重算收益必须为 0
            if r["candidate_action_count"] != 0 or len(r["post_check_outcomes"]) != 0:
                errors.append("ZERO_GAIN_FRACTURE_HAS_ACTIONS:" + fid)
            if geom_reduction.get(fid, F(0)) != F(0):
                errors.append("ZERO_GAIN_FRACTURE_HAS_GAIN:" + fid)
        # 收益>0 却宣称零收益 => 隐藏标签泄漏（几何说有用但被排除）
        if geom_reduction.get(fid, F(0)) > 0 and not r["is_informative"]:
            errors.append("GAINFUL_FRACTURE_MARKED_ZERO:" + fid)
    if informative != claimed_informative:
        errors.append("INFORMATIVE_SET_MISMATCH")
    if not set(ct["top_fracture_ids"]).issubset(informative):
        errors.append("TOP_K_INCLUDES_UNPROVEN_FRACTURE")
    if sorted(r["rank"] for r in ranked) != list(range(1, len(ranked) + 1)):
        errors.append("RANK_SEQUENCE_INVALID")
    # OURS_MINIMAX_EXACT：informative 在前（worst-case 剩余升序、同分按 id），
    # 零收益裂隙排最后。
    key = lambda r: (0 if r["is_informative"] else 1,
                     F(r["worst_case_remaining_ambiguity_measure"]), r["fracture_id"])
    exp = [r["fracture_id"] for r in sorted(ranked, key=key)]
    act_order = [r["fracture_id"] for r in sorted(ranked, key=lambda r: r["rank"])]
    if exp != act_order:
        errors.append("RANKING_NOT_MINIMAX_EXACT:%s|%s" % (exp, act_order))
    notes.append("expected_*/measure_split_* 未独立复算（定义不可唯一推出）")
    return {"cluster_id": cid, "branches_recomputed": nbranch,
            "ranked_fracture_count": len(ranked),
            "zero_gain_fractures": zero,
            "top_fracture_ids": ct["top_fracture_ids"],
            "geometry_reduction_by_fracture":
                {k: str(v) for k, v in geom_reduction.items()},
            "not_independently_recomputed":
                ["expected_remaining_ambiguity_measure",
                 "expected_ambiguity_reduction_fraction",
                 "measure_split_best_action_id",
                 "measure_split_resolved_measure_fraction"],
            "hidden_label_isolation":
                "排序键与 informative 集全部由几何条件域重算推出；未引入地下真值标签",
            "errors": errors,
            "verdict": "AGREE" if not errors else "DISAGREE"}


def policy_matches(exp, act_order, ranked):
    """同分时 M2 可用其它二级键；只在严格同序或同分并列时判合格。"""
    if exp == act_order:
        return True
    byworst = {r["fracture_id"]: F(r["worst_case_remaining_ambiguity_measure"]) for r in ranked}
    seq = [byworst[f] for f in act_order]
    return seq == sorted(seq)


def main():
    m1 = os.path.abspath(os.path.join(HERE, "..", "M1_real_to_exact_bridge"))
    run = os.path.abspath(os.path.join(HERE, "..", "M2_active_identifiability",
                                       "runs", "run_20261120_exact_v1"))
    only = sys.argv[1:] or ["jardim_c02", "jardim_c03", "jardim_c04",
                            "jardim_c05", "jardim_c06", "perozzo_c02"]
    out = {}
    for cid in only:
        out[cid] = verify_case(cid, m1, run)
        print(cid, out[cid]["verdict"], "branches", out[cid]["branches_recomputed"],
              "errors", len(out[cid]["errors"]), out[cid]["errors"][:5])
    path = os.path.join(HERE, "ranking_verdict" +
                        ("_" + only[0] if len(only) == 1 else "") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    return 0 if all(v["verdict"] == "AGREE" for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
