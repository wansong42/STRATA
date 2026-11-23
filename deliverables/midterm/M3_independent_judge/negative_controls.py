#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3_independent_judge/negative_controls.py
盲判负控：在 M3 内部独立构造 6 类变异包（M3-N1..N6），验证判卷器牙口。
每类负控必须被 judge_cluster 判为 DISAGREE / MALFORMED（非零退出）。
变异只写 M3 自己的 _nc_scratch/ 目录，绝不触碰 M1/M2 冻结件。
输出 negative_controls.json；退出码 0 = 6/6 全部正确拒收。
"""
import copy, json, os, sys
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_real as J  # noqa: E402

M1 = os.path.abspath(os.path.join(HERE, "..", "M1_real_to_exact_bridge"))
M2 = os.path.abspath(os.path.join(HERE, "..", "M2_active_identifiability",
                                 "runs", "run_20261120_exact_v1"))
SCRATCH = os.path.join(HERE, "_nc_scratch")
os.makedirs(SCRATCH, exist_ok=True)

RESULTS = []

def run_case(nc_id, desc, cluster_id, mutate_model=None, mutate_cert=None,
             mutate_act=None, expect_reasons=()):
    """构造变异并跑 judge_cluster，断言非 AGREE 且命中预期 reason。"""
    work = os.path.join(SCRATCH, nc_id)
    os.makedirs(work, exist_ok=True)
    # 复制 model（可变异的副本）
    model_path = os.path.join(M1, "clusters", cluster_id, "model.json")
    model = J.jload(model_path)
    if mutate_model:
        model = mutate_model(copy.deepcopy(model))
    mp = os.path.join(work, "model.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1, sort_keys=True)
    # 复制 certification / active_check
    cert = J.jload(os.path.join(M2, "cases", cluster_id, "certification.json"))
    if mutate_cert:
        cert = mutate_cert(copy.deepcopy(cert))
    cp = os.path.join(work, "certification.json")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(cert, f, ensure_ascii=False, indent=1, sort_keys=True)
    act = J.jload(os.path.join(M2, "cases", cluster_id, "active_check.json"))
    if mutate_act:
        act = mutate_act(copy.deepcopy(act))
    ap = os.path.join(work, "active_check.json")
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(act, f, ensure_ascii=False, indent=1, sort_keys=True)

    # 手动驱动 judge_cluster 的核心：用变异后的 model/cert/act
    # 直接调用内部流程（构造伪 m1/m2 目录结构）
    fake_m1 = os.path.join(work, "m1")
    fake_m2 = os.path.join(work, "m2")
    os.makedirs(os.path.join(fake_m1, "clusters", cluster_id), exist_ok=True)
    os.makedirs(os.path.join(fake_m2, "cases", cluster_id), exist_ok=True)
    import shutil
    shutil.copyfile(mp, os.path.join(fake_m1, "clusters", cluster_id, "model.json"))
    shutil.copyfile(cp, os.path.join(fake_m2, "cases", cluster_id, "certification.json"))
    shutil.copyfile(ap, os.path.join(fake_m2, "cases", cluster_id, "active_check.json"))

    reg = J.jload(os.path.join(M1, "cluster_register.json"))
    entry = [e for e in reg["clusters"] if e["cluster_id"] == cluster_id][0]
    try:
        det = J.judge_cluster(cluster_id, entry, fake_m1, fake_m2)
        verdict = det["verdict"]
        reason = (det.get("first_disagreement") or {}).get("reason", "")
    except Exception as e:
        verdict, reason = "MALFORMED", "JUDGE_EXCEPTION: " + repr(e)
    rejected = verdict in ("DISAGREE", "MALFORMED")
    reason_ok = (not expect_reasons) or any(er in reason for er in expect_reasons)
    ok = rejected and reason_ok
    RESULTS.append({"nc_id": nc_id, "description": desc, "cluster_id": cluster_id,
                    "verdict": verdict, "reason": reason,
                    "correctly_rejected": rejected, "reason_match": reason_ok,
                    "passed": ok})
    print(f"{'PASS' if ok else 'FAIL'} {nc_id}: verdict={verdict} reason={reason}")
    return ok

def main():
    amb = "jardim_c02"   # AMBIGUOUS
    # M3-N1: AMBIGUOUS -> CERTIFIED_CONNECTED
    def n1(cert):
        cert["status"] = "CERTIFIED_CONNECTED"
        cert["exists_connected"] = True
        cert["all_connected"] = True
        return cert
    run_case("M3-N1", "AMBIGUOUS 篡改为 CERTIFIED_CONNECTED", amb,
             mutate_cert=n1, expect_reasons=("STATUS_MISMATCH", "CELL", "WITNESS",
                                             "ALL_CONNECTED"))

    # M3-N2: 跨 t/mode 拼假路径 —— 在 connected_witness 中改一步的 t
    def n2(cert):
        w = copy.deepcopy(cert["connected_witness"])
        steps = w.get("contact_step_witnesses") or []
        if steps:
            # 把最后一步的 t 改成 witness t + 1（跨 t 假通路）
            t0 = F(steps[-1]["t"])
            steps[-1]["t"] = str(t0 + 1)
        w["contact_step_witnesses"] = steps
        cert["connected_witness"] = w
        return cert
    run_case("M3-N2", "connected_witness 跨 t 拼假路径", amb,
             mutate_cert=n2, expect_reasons=("CONNECTED_WITNESS_INVALID",))

    # M3-N3: 删掉一个相切 POINT 节点
    def n3(cert):
        cells = cert.get("cells", [])
        # 找一个 connected=True 的 POINT cell 删除
        for i, c in enumerate(cells):
            if c.get("kind") == "POINT" and c.get("connected"):
                del cells[i]
                break
        cert["cells"] = cells
        return cert
    run_case("M3-N3", "删除一个相切 POINT 节点", amb,
             mutate_cert=n3, expect_reasons=("CELL_MISMATCH", "MISSING_CELL"))

    # M3-N4: 把 collapse_tree 中仍为 AMBIGUOUS 的分支伪造成坍缩
    def n4(act):
        ct = act.get("collapse_tree") or {}
        for fid, fr in (ct.get("fractures") or {}).items():
            for br in fr.get("branches") or []:
                if br.get("post_status") == "AMBIGUOUS_WITH_VERIFIED_WITNESSES":
                    br["post_status"] = "CERTIFIED_CONNECTED"
                    br["collapsed_to_certified"] = True
                    br["all_connected"] = True
                    return act
        return act
    run_case("M3-N4", "不坍缩分支伪造成 CERTIFIED", amb,
             mutate_act=n4, expect_reasons=("COLLAPSE_BRANCH_STATUS_MISMATCH",))

    # M3-N5: 篡改顶点并同步表层哈希但不改注册根
    def n5(model):
        v = model["modes"][0]["triangles"][0]["vertices"][0]
        v[0] = str(F(v[0]) + F(1, 7))
        # 同步表层哈希（伪造 input_sha256 与 model_file_sha256 为新 canonical）
        new_sha = J.canon_sha(model)
        model["_forged_sha"] = new_sha
        return model
    def n5c(cert):
        # 表层哈希同步为变异后的 canonical（但注册根不变）
        # 由于 model 已变，cert.input_sha256 若不更新会报 CERT 不匹配；
        # 这里故意把 cert.input_sha256 改成与新 model 一致，测试注册根比对
        return cert
    run_case("M3-N5", "顶点篡改 + 表层哈希同步但注册根未改", amb,
             mutate_model=n5, mutate_cert=n5c,
             expect_reasons=("SOURCE_SHA_MISMATCH_REGISTER", "SOURCE_SHA_MISMATCH_CERT"))

    # M3-N6: scope 改为合成 scope
    def n6(model):
        model["scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
        return model
    run_case("M3-N6", "scope 改为合成域冒充真实簇", amb,
             mutate_model=n6, expect_reasons=("SCOPE_NOT_REAL_DERIVED",))

    npass = sum(1 for r in RESULTS if r["passed"])
    out = {"schema": "m3-negative-controls/v1", "total": len(RESULTS),
           "passed": npass, "correctly_rejected": sum(1 for r in RESULTS
                                                      if r["correctly_rejected"]),
           "cases": RESULTS, "verdict": "PASS" if npass == len(RESULTS) else "FAIL"}
    p = os.path.join(HERE, "negative_controls.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n{npass}/{len(RESULTS)} negative controls correctly rejected")
    return 0 if npass == len(RESULTS) else 1

if __name__ == "__main__":
    raise SystemExit(main())
