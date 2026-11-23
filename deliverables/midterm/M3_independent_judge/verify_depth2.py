#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 · collapse_tree 深度 2（next_step）子分支独立复核 + 测度分解勾稽。

覆盖任务书 M3 §二.5 的"主动复核坍缩结论"全部层级：
  depth1（75 条，已由 judge_real 覆盖）→ next_step（depth2，74 条，本脚本覆盖）。

逐条 depth-2 分支独立重算 post_status / exists / all / collapsed_to_certified；
并对 depth1 分支测度做分解勾稽：sum(depth-2 分支可行测度) == depth1 分支测度。
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
    errors, nbranch, parents = [], 0, 0
    for fid, fr in (ct["fractures"] or {}).items():
        for d1 in fr["branches"]:
            ns = d1.get("next_step")
            if not ns or not ns.get("branches"):
                continue
            parents += 1
            sum_measure = F(0)
            for b in ns["branches"]:
                nbranch += 1
                sub = J.eval_domain(mm, order, wb, condition(order, b["condition"]))
                if sub["status"] != b["post_status"]:
                    errors.append("D2_STATUS:%s/%s/%s mine=%s theirs=%s" %
                                  (fid, d1["branch_id"], b["branch_id"], sub["status"], b["post_status"]))
                if sub["exists_connected"] != b["exists_connected"] or sub["all_connected"] != b["all_connected"]:
                    errors.append("D2_FLAGS:%s/%s/%s" % (fid, d1["branch_id"], b["branch_id"]))
                if sub["total_measure"] != F(b["branch_measure"]):
                    errors.append("D2_MEASURE:%s/%s/%s" % (fid, d1["branch_id"], b["branch_id"]))
                collapsed = sub["status"] in CERT
                if collapsed != b["collapsed_to_certified"]:
                    errors.append("D2_COLLAPSED:%s/%s/%s" % (fid, d1["branch_id"], b["branch_id"]))
                sum_measure += sub["total_measure"]
            if sum_measure != F(d1["branch_measure"]):
                errors.append("D1_PARTITION:%s/%s sum_children=%s parent=%s" %
                              (fid, d1["branch_id"], sum_measure, d1["branch_measure"]))
    return {"cluster_id": cid, "depth2_parents": parents, "depth2_branches": nbranch,
            "errors": errors, "verdict": "AGREE" if not errors else "DISAGREE"}


def main():
    m1 = os.path.abspath(os.path.join(HERE, "..", "M1_real_to_exact_bridge"))
    run = os.path.abspath(os.path.join(HERE, "..", "M2_active_identifiability",
                                       "runs", "run_20261120_exact_v1"))
    only = sys.argv[1:] or ["jardim_c02", "jardim_c03", "jardim_c04",
                            "jardim_c05", "jardim_c06", "perozzo_c02"]
    out = {}
    for cid in only:
        out[cid] = verify_case(cid, m1, run)
        print(cid, out[cid]["verdict"], "branches", out[cid]["depth2_branches"],
              "errors", len(out[cid]["errors"]), out[cid]["errors"][:5])
    path = os.path.join(HERE, "depth2_verdict" + ("_" + only[0] if len(only) == 1 else "") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    return 0 if all(v["verdict"] == "AGREE" for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
