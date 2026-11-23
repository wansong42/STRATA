# -*- coding: utf-8 -*-
"""S2-C · 证书链的端到端成本：把增量验证器接入 S3-A/S3-B 的闭环账（本单唯一新核心模块）。

判据权威 = `results/s2c_certificate_cost/prereg_s2c.json` 与任务单
`results/architect_main_20261012_dispatch/task_order_S2_C_certificate_chain_cost_20261018.md`
（sha12 ``a8856df2659f``）§2.1/§4。本件**只读**消费三个既有模块，字节零改：

* ``closed_loop_expanded``（S3-A：世界/策略/停表/证书发证路的语义来源）
* ``incremental_verifier``（S2-V：增量验证器与内容寻址缓存）
* ``statistical_certificate``（S2-N：证书格式与全验证器）

强制语义（任务单 §2.1 逐条）：

1. **同判**：同一轨迹/同一张证书上 full 与 incremental 逐张 accept/reject 必须相同；
   差异 0（差异出现 ⇒ 调用侧按停机规则处理；本件如实记录 ``equivalence_diff``，不掩盖）。
2. **端到端分列**：行动 / 求解选点 / 证书组装 / 验证 / 总时间，全部 ``MEASURED_ONLY``；
   双模式同轨迹单趟实测 + 分列重建（``end_to_end.full_seconds`` / ``incremental_seconds``
   为扣除另一模式验证调用墙钟的派生量，口径逐字登记；直接单模式口径见子集两趟账）。
3. **链条口径**：按 S2-V 链条口径（终末区域内容跨证书复用）实现于 :func:`chain_cost`；
   ``reuse_requires_prev=True`` 严格 prev 口径为主口径，批次累计口径为次生口径（脚本另跑）。
4. **增量面**：只重用父区域/状态集/证书字段未变的子树；缓存键带内容 sha；不得按
   「上次通过」放行 —— 由 ``incremental_verifier`` 本体承担，本件**不新增任何旁路**，
   不调 ``ReuseCache`` 内件，不改命中央的证据语义。
5. **证书**：复用 S2-N/S2-N0 证书格式与验证器；空集/混合/冲突不发证书（发证路的
   ``StatisticalUnresolved`` 原样记 status/reason）。
6. **不外推**：只对 S3-A 的声明世界/种子/预算/α 负责（``W_TH2/4/10/12/11`` × 5 种子 ×
   5 策略；本件自身不新增世界）。

口径：``research_line=S2C_certificate_chain_cost``；本层不写盘；输出只含整数/布尔/字符串
与 ``MEASURED_ONLY`` 墙钟。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import closed_loop_expanded as cle
from . import incremental_verifier as iv
from . import statistical_certificate as sc

__all__ = [
    "WATERMARK", "MODES", "CHAIN_MODES", "CostAccountingError", "ChainState",
    "run_with_cost", "certificate_rows", "chain_cost", "canonical_json", "sha12_of",
]

WATERMARK = {
    "research_line": "S2C_certificate_chain_cost",
    "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS",
    "noise": "N1",
    "dictionary": "COMPLETE_KNOWN",
    "claim_scope": "MEASURED_ONLY",
    "forbidden": ["真实贯通判定", "渐近优势", "成本优势外推", "对外承诺文本"],
}

MODES = ("full", "incremental")
#: 链条口径（与 S2-V 的三口径同名同序）
CHAIN_MODES = ("P1_CHAIN_STRICT", "P2_CHAIN_BATCH", "P3_REPLAY_STRICT")


class CostAccountingError(RuntimeError):
    """成本分列自洽性破裂（分列和超过总墙钟等）：硬抛，不静默夹紧。"""


def canonical_json(obj: Any) -> str:
    return iv.canonical_json(obj)


def sha12_of(obj: Any) -> str:
    return iv.sha12_of(obj)


class ChainState:
    """链序状态：一个共享证据缓存 + 上一张已发证书（增量面严格 prev 口径的载体）。

    * ``cache``：``incremental_verifier.ReuseCache`` 实例，跨证书跨 run 常驻（进程内）。
    * ``prev_cert``：链条上一张**已发出**证书的 body（字典）；未发证书的尝试不推进链。
    * 本件只读它的公开面（``entries`` 不动、不预填、不清洗），任何证据写入都由
      ``verify_incremental`` 本体完成。
    """

    def __init__(self, reuse_requires_prev: bool = True) -> None:
        self.cache = iv.ReuseCache()
        self.prev_cert: Optional[Mapping] = None
        self.prev_item_id: Optional[str] = None
        self.reuse_requires_prev = bool(reuse_requires_prev)
        self.n_issued_seen = 0

    def consume(self, cert_body: Mapping, item_id: str) -> None:
        """链条推进：只认已发出证书（空集/混合/冲突不发证 ⇒ 不推进）。"""
        self.prev_cert = cert_body
        self.prev_item_id = str(item_id)
        self.n_issued_seen += 1


def _new_timers() -> Dict[str, float]:
    return {"action_s": 0.0, "choice_s": 0.0, "eprocess_main_s": 0.0,
            "eprocess_cert_s": 0.0, "certify_only_s": 0.0, "public_build_s": 0.0,
            "verify_full_s": 0.0, "verify_inc_s": 0.0}


def _public_of(world, steps) -> "sc.PublicEvidence":
    """与 S3-A ``issue_stat_certificate`` 同一构造（验证请求的公开面）。"""
    return sc.PublicEvidence(model_id=world.static.table.table_id,
                             kernel=world.static.kernel, tier=world.static.tier,
                             states=world.static.states,
                             steps=[(world.static.by_id[str(a)], r) for a, r in steps],
                             labels=world.labels, cache=world._pcache)


def _verify_full(table, public, query_id: str, cert: Mapping) -> dict:
    return sc.verify_statistical_certificate(table, public, query_id, cert)


def _verify_incremental(chain: ChainState, table, public, query_id: str, cert: Mapping,
                        full_verifier) -> dict:
    return iv.verify_incremental(chain.prev_cert, cert, table, full_verifier=full_verifier,
                                 cache=chain.cache, public_evidence=public,
                                 reuse_requires_prev=chain.reuse_requires_prev)


def run_with_cost(strategy: str, model, action_pool, budget: int, alpha, rng,
                  verifier_mode: str = "full", *,
                  chain: Optional[ChainState] = None,
                  dual_measure: bool = True) -> dict:
    """跑一条含噪闭环轨迹并把证书验证接入账（S3-A ``run_policy_expanded`` 的镜像）。

    参数与语义（任务单 §2.1 的签名 + 两个显式扩参）：

    * ``strategy/model/action_pool/budget/alpha/rng``：与 S3-A 逐项同义；
      ``model`` = ``closed_loop_expanded.ExpandedWorld``。
    * ``verifier_mode``：**主模式**——该模式的 accept/reject 驱动作停表与
      「获证正确」终末判定（``full`` ⇒ 与 S3-A 完全相同的行为）。
    * ``chain``：链序状态（跨 run 传同一实例即成链）；缺省新建（单 run 链）。
    * ``dual_measure``：True = 另一模式作为影子同张再验一次（只记读数，不驱动轨迹，
      差异记 ``equivalence_diff``）；False = 只跑主模式（子集两趟直接口径用）。
    * 本函数**不改** ``closed_loop_expanded`` 任一字节：``_stop_and_why`` 以只读调用
      复用（停表语义的唯一来源），策略选择复用 ``choose_action_expanded``。
    """
    world = model
    if not isinstance(world, cle.ExpandedWorld):
        raise cle.ClosedLoopExpandedError("model 必须是 ExpandedWorld")
    if strategy not in cle.STRATEGIES:
        raise cle.ClosedLoopExpandedError("未知策略 %r" % (strategy,))
    pool = tuple(str(a) for a in action_pool)
    if set(pool) != set(world.pool_ids):
        raise cle.ClosedLoopExpandedError("%s: action_pool 与声明池不一致" % world.world_id)
    if int(budget) != world.budget_int:
        raise cle.ClosedLoopExpandedError("%s: budget 与声明不一致" % world.world_id)
    from fractions import Fraction as _Fr
    if _Fr(alpha) != world.alpha_seq:
        raise cle.ClosedLoopExpandedError("%s: alpha 与声明不一致" % world.world_id)
    if verifier_mode not in MODES:
        raise CostAccountingError("未知 verifier_mode %r" % (verifier_mode,))
    if chain is None:
        chain = ChainState()
    t_start = time.perf_counter()
    T = _new_timers()
    n_choose = n_cert = n_verify_full = n_verify_inc = 0
    n_pass_diff = 0
    equivalence_diffs: List[dict] = []

    steps = []                                    # [(action_id, record)] 公开历史
    allowed = tuple(int(s) for s in world.initial_allowed)
    used = 0
    executed: Tuple[str, ...] = ()
    read_so_far = 0
    rows: List[dict] = []
    cert_rows: List[dict] = []

    def issue(tag: str) -> dict:
        nonlocal n_cert, n_verify_full, n_verify_inc, n_pass_diff
        out = {"step_tag": str(tag), "issued": False, "status": "NOT_ISSUED", "reason": None,
               "label_claim": None, "allowed_n": len(world.static.states),
               "allowed_sha12": None, "n_cubes": 0, "verified": None,
               "verify_reasons": [], "certificate_id": None, "certificate_sha12": None,
               "certify_seconds": 0.0, "verify_seconds": 0.0, "s2n_status": None,
               "certificate": None, "verification": None, "verifications": {},
               "certify_detail": None}
        if not steps:
            out["status"] = "NOT_ATTEMPTED"
            out["reason"] = "T0_NO_PUBLIC_HISTORY_PER_S2N_PROTOCOL"
            return out
        n_cert += 1
        t0 = time.perf_counter()
        t_ep = time.perf_counter()
        try:
            ep = world.eprocess(steps)
            allowed_now = tuple(ep.state_set())
            out["allowed_n"] = len(allowed_now)
            out["allowed_sha12"] = cle.sha12_obj(list(allowed_now))
            out["s2n_status"] = str(ep.status())
        except Exception as exc:                          # noqa: BLE001
            T["eprocess_cert_s"] += time.perf_counter() - t_ep
            out["status"] = "EXCEPTION"
            out["reason"] = "EPROCESS_RAISED: %r" % (exc,)
            out["certify_seconds"] = round(time.perf_counter() - t0, 6)
            return out
        T["eprocess_cert_s"] += time.perf_counter() - t_ep
        tc = time.perf_counter()
        try:
            res = sc.certify_with_statistics(world.static.table, ep, cle.CERT_QUERY,
                                             dict(cle.CERT_BUDGET_TEMPLATE), world.alpha_seq)
        except Exception as exc:                          # noqa: BLE001
            T["certify_only_s"] += time.perf_counter() - tc
            out["status"] = "EXCEPTION"
            out["reason"] = "CERTIFY_RAISED: %r" % (exc,)
            out["certify_seconds"] = round(time.perf_counter() - t0, 6)
            return out
        T["certify_only_s"] += time.perf_counter() - tc
        out["certify_seconds"] = round(time.perf_counter() - t0, 6)
        if not isinstance(res, sc.StatisticalCertificate):
            out["status"] = str(res.status)
            out["reason"] = str(res.reason)
            detail = res.detail if isinstance(res.detail, dict) else {}
            out["certify_detail"] = cle.jsonable(detail)
            return out
        cert_body = res.to_dict()
        out.update({"issued": True, "status": "CERTIFICATE",
                    "label_claim": str(cert_body.get("label_claim")),
                    "certificate_id": cert_body.get("certificate_id"),
                    "certificate_sha12": cle.sha12_obj(cert_body),
                    "certificate": cert_body,
                    "n_cubes": int(res.s2n.get("n_terminal_regions") or 0)})
        # ---- 验证接入：公开面构造算一次，两模式共用；顺序 = full 先（主模式先）----
        tb = time.perf_counter()
        public = _public_of(world, steps)
        t_pub = time.perf_counter() - tb
        T["public_build_s"] += t_pub
        full_verifier = lambda c: _verify_full(world.static.table, public, cle.CERT_QUERY, c)  # noqa: E731
        run_modes = [verifier_mode] + ([m for m in MODES if m != verifier_mode]
                                       if dual_measure else [])
        for m in run_modes:
            tv = time.perf_counter()
            try:
                if m == "full":
                    rep = _verify_full(world.static.table, public, cle.CERT_QUERY, cert_body)
                    n_verify_full += 1
                    T["verify_full_s"] += time.perf_counter() - tv
                    out["verifications"]["full"] = {
                        "pass": bool(rep.get("pass")),
                        "reasons": list(rep.get("reasons", [])),
                        "call_seconds": round(time.perf_counter() - tv, 6),
                        "verifier_version": rep.get("verifier_version")}
                else:
                    rep = _verify_incremental(chain, world.static.table, public,
                                              cle.CERT_QUERY, cert_body, full_verifier)
                    n_verify_inc += 1
                    T["verify_inc_s"] += time.perf_counter() - tv
                    out["verifications"]["incremental"] = {
                        "pass": bool(rep.get("pass")),
                        "reasons": list(rep.get("reasons", [])),
                        "call_seconds": round(time.perf_counter() - tv, 6),
                        "reused_regions": int(rep.get("reused_regions") or 0),
                        "rechecked_regions": int(rep.get("rechecked_regions") or 0),
                        "cost": dict(rep.get("cost") or {}),
                        "cache_stats": dict(rep.get("cache_stats") or {}),
                        "reuse_policy": dict(rep.get("reuse") or {}),
                        "verifier_version": rep.get("verifier_version")}
            except Exception as exc:                      # noqa: BLE001
                if m == "full":
                    T["verify_full_s"] += time.perf_counter() - tv
                else:
                    T["verify_inc_s"] += time.perf_counter() - tv
                out["verifications"][m] = {"pass": None, "reasons": ["VERIFY_RAISED: %r" % (exc,)],
                                           "raised": "%s: %s" % (type(exc).__name__, exc)}
        primary = out["verifications"].get(verifier_mode) or {}
        out["verified"] = (bool(primary.get("pass"))
                           if primary.get("pass") is not None else False)
        out["verify_reasons"] = list(primary.get("reasons", []))
        out["verification"] = {"pass": out["verified"], "reasons": list(out["verify_reasons"])}
        # verify_seconds = 公开面构造 + 本张全部验证调用墙钟之和（单模式 ≈ S3-A 的
        # verify_seconds 语义：public + 一次验证；双模式时如实记合计，分列另见 cost）
        out["verify_seconds"] = round(
            t_pub + sum(float((out["verifications"].get(m) or {}).get("call_seconds") or 0.0)
                        for m in out["verifications"]), 6)
        if dual_measure and len(out["verifications"]) == 2:
            pf = out["verifications"]["full"].get("pass")
            pi = out["verifications"]["incremental"].get("pass")
            rf = list(out["verifications"]["full"].get("reasons") or [])
            ri = list(out["verifications"]["incremental"].get("reasons") or [])
            out["equivalence"] = {"pass_equal": bool(pf == pi),
                                  "reasons_equal": bool(rf == ri),
                                  "n_reasons_full": len(rf), "n_reasons_inc": len(ri)}
            if not out["equivalence"]["pass_equal"]:
                n_pass_diff += 1
                equivalence_diffs.append({
                    "certificate_id": out["certificate_id"], "tag": tag,
                    "pass_full": pf, "pass_inc": pi,
                    "reasons_full": rf[:4], "reasons_inc": ri[:4]})
        return out

    cert0 = issue("t0")
    certs_seq: List[dict] = [cert0]
    rows.append({"t": 0, "action_id": None, "cost": 0, "used_before": 0, "used_after": 0,
                 "record_key": None, "record_sha12": None, "record": None,
                 "allowed_before": len(allowed), "allowed_after": len(allowed),
                 "allowed_sha12": cle.sha12_obj(list(allowed)),
                 "allowed_state_ids": list(allowed) if len(allowed) <= 256 else None,
                 "certificate": _summary(cert0), "eprocess": None, "choice_detail": None,
                 "read_mask_so_far": 0,
                 "pin": "t0：S2-N 协议要求非空公开历史 ⇒ 不发证书也不尝试"})
    stop = cle._stop_and_why(world, allowed, used, executed, 0, world.max_steps, cert0)

    while not stop:
        step = len(steps)
        pub = world.public(allowed, used, executed, step, world.history_sha12(steps))
        ts = time.perf_counter()
        choice = cle.choose_action_expanded(strategy, world, pub, rng, read_so_far)
        T["choice_s"] += time.perf_counter() - ts
        n_choose += 1
        aid = str(choice["action_id"])
        if aid not in pub.legal:
            raise cle.ClosedLoopExpandedError("%s/%s: 选中非法行动 %r"
                                              % (world.world_id, strategy, aid))
        before = used
        allowed_before_n = len(allowed)
        used += world.costs[aid]
        if used > world.budget_int:
            raise cle.ClosedLoopExpandedError("%s: 超预算" % world.world_id)
        ta = time.perf_counter()
        record = world.observe(aid)
        T["action_s"] += time.perf_counter() - ta
        executed = executed + (aid,)
        read_so_far |= int(world.static.read_masks[aid])
        steps.append((aid, record))
        te = time.perf_counter()
        ep = world.eprocess(steps)
        T["eprocess_main_s"] += time.perf_counter() - te
        allowed = tuple(ep.state_set())
        cert = issue("t%d" % len(steps))
        certs_seq.append(cert)
        if cert["issued"]:
            item_id = _item_id(world.world_id, strategy, len(steps))
            cert_rows.append(_cert_row(world, strategy, len(steps), cert, chain))
            chain.consume(cert["certificate"], item_id)
        rows.append({"t": len(steps), "action_id": aid, "cost": int(world.costs[aid]),
                     "cost_domain_length": float(world.costs_dl[aid]),
                     "used_before": int(before), "used_after": int(used),
                     "record_key": cle.record_key_json(record),
                     "record_sha12": cle.sha12_obj(record),
                     "record": record,
                     "record_counts": record.get("counts"),
                     "allowed_before": int(allowed_before_n),
                     "allowed_after": len(allowed),
                     "allowed_sha12": cle.sha12_obj(list(allowed)),
                     "allowed_state_ids": list(allowed) if len(allowed) <= 256 else None,
                     "certificate": _summary(cert),
                     "eprocess": {"t": int(ep.t), "status": str(ep.status()),
                                  "n_S": len(ep.state_set()),
                                  "Gamma": list(ep.target_set())},
                     "choice_detail": cle.trim_choice_detail(choice["detail"]),
                     "read_mask_so_far": int(read_so_far)})
        stop = cle._stop_and_why(world, allowed, used, executed, len(steps),
                                 world.max_steps, cert)

    total = time.perf_counter() - t_start
    measured = sum(T.values())
    residual = total - measured
    if residual < -1e-6:
        raise CostAccountingError("分列和 %.9f 超过总墙钟 %.9f：计时器重叠，拒绝出账"
                                  % (measured, total))
    residual = max(0.0, residual)
    e2e_full = total - T["verify_inc_s"]
    e2e_inc = total - T["verify_full_s"]
    answer = None
    for c in certs_seq:
        if c["issued"] and c["verified"]:
            answer = int(c["label_claim"])
            break
    dup_used = ("XS1" in executed) and ("XS1M" in executed)
    trace = {
        "schema": "s2c.trace.v1",
        "world_id": world.world_id, "strategy": strategy,
        "truth_state": int(world.truth_state), "truth_label": int(world.labels[world.truth_state]),
        "tier": world.static.tier, "kernel_id": world.static.kernel.kernel_id,
        "budget": int(world.budget_int), "alpha_seq": str(world.alpha_seq),
        "max_steps": int(world.max_steps), "noise_seed": int(world.noise_seed),
        "verifier_mode": verifier_mode, "dual_measure": bool(dual_measure),
        "steps": rows,
        "stop": stop, "used": int(used), "n_steps": len(steps),
        "executed": list(executed), "answer": answer,
        "correct": (None if answer is None
                    else bool(answer == int(world.labels[world.truth_state]))),
        "certificate_attempts": int(n_cert),
        "certificates_issued": int(len(cert_rows)),
        "certificates_verified_primary": int(sum(1 for c in cert_rows
                                                 if c["verify"][verifier_mode]["pass"])),
        "n_pass_diff": int(n_pass_diff),
        "equivalence_diffs": equivalence_diffs,
        "history_sha12": world.history_sha12(steps),
        "cost": {
            "action_seconds": round(T["action_s"], 6),
            "choice_seconds": round(T["choice_s"], 6),
            "eprocess_main_seconds": round(T["eprocess_main_s"], 6),
            "eprocess_cert_seconds": round(T["eprocess_cert_s"], 6),
            "certify_only_seconds": round(T["certify_only_s"], 6),
            "public_build_seconds": round(T["public_build_s"], 6),
            "verify_full_seconds": round(T["verify_full_s"], 6),
            "verify_inc_seconds": round(T["verify_inc_s"], 6),
            "total_seconds": round(total, 6),
            "residual_seconds": round(residual, 6),
            "compartments": {
                "action": round(T["action_s"], 6),
                "solve_choice": round(T["choice_s"] + T["eprocess_main_s"], 6),
                "assemble": round(T["eprocess_cert_s"] + T["certify_only_s"], 6),
                "verify_full_incl_public": round(T["public_build_s"] + T["verify_full_s"], 6),
                "verify_incremental_incl_public": round(T["public_build_s"] + T["verify_inc_s"], 6),
                "total": round(total, 6),
                "residual": round(residual, 6)},
            "end_to_end": {
                "full_seconds": round(e2e_full, 6),
                "incremental_seconds": round(e2e_inc, 6),
                "reconstruction_note": ("双模式同轨迹单趟实测 + 分列重建：端到端 = 总墙钟 − "
                                        "另一模式验证调用墙钟；直接单模式口径见子集两趟账")},
            "n_choose_calls": int(n_choose), "n_certify_calls": int(n_cert),
            "n_verify_full_calls": int(n_verify_full), "n_verify_inc_calls": int(n_verify_inc),
            "basis": "MEASURED_ONLY"},
        "certificate_rows": cert_rows,
        "watermark": dict(WATERMARK),
    }
    return trace


def _summary(cert: dict) -> dict:
    """与 S3-A ``_cert_summary`` 同名同义 + 双模式验证块（超集，不改原键语义）。"""
    return {"issued": bool(cert["issued"]), "status": cert["status"], "reason": cert["reason"],
            "label_claim": cert["label_claim"], "verified": cert["verified"],
            "verify_reasons": list(cert["verify_reasons"]),
            "certificate_id": cert["certificate_id"],
            "certificate_sha12": cert["certificate_sha12"],
            "allowed_n": int(cert["allowed_n"]), "n_cubes": int(cert.get("n_cubes", 0)),
            "allowed_sha12": cert["allowed_sha12"], "s2n_status": cert.get("s2n_status"),
            "certify_seconds": round(cert["certify_seconds"], 6),
            "verify_seconds": round(cert["verify_seconds"], 6),
            "verifications": {m: {k: v for k, v in (cert["verifications"].get(m) or {}).items()}
                              for m in cert["verifications"]},
            "equivalence": cert.get("equivalence")}


def _item_id(world_id: str, strategy: str, t: int) -> str:
    return "%s|%s|t%d" % (world_id, strategy, t)


def _term_shas(cert_body: Mapping) -> Tuple[List[str], List[str]]:
    term = [r for r in cert_body.get("regions", []) if r.get("is_terminal")]
    term_shas = [iv.region_content_sha12(r, iv.S2N_FLAVOR) for r in term]
    all_shas = [iv.region_content_sha12(r, iv.S2N_FLAVOR)
                for r in cert_body.get("regions", [])]
    return term_shas, all_shas


def _cert_row(world, strategy: str, t: int, cert: dict, chain: ChainState) -> dict:
    """单张证书的链条账行（live 对象只出现在本次调用内，行本身 JSON 可落盘）。"""
    body = cert["certificate"]
    term_shas, all_shas = _term_shas(body)
    prev_contents = (iv._prev_content_shas(chain.prev_cert, iv.S2N_FLAVOR)  # noqa: SLF001
                     if chain.prev_cert is not None else None)
    term_in_prev = (sum(1 for s in term_shas if s in prev_contents)
                    if prev_contents is not None else 0)
    all_in_prev = (sum(1 for s in all_shas if s in prev_contents)
                   if prev_contents is not None else 0)
    term = [r for r in body.get("regions", []) if r.get("is_terminal")]
    row = {
        "item_id": _item_id(world.world_id, strategy, t),
        "world_id": world.world_id, "strategy": strategy, "t": int(t),
        "certificate_id": cert["certificate_id"], "cert_sha12": cert["certificate_sha12"],
        "cert_body": body,
        "label_claim": cert["label_claim"], "allowed_n": int(cert["allowed_n"]),
        "n_regions": len(body.get("regions", [])), "n_terminal_regions": len(term),
        "n_excluded_regions": sum(1 for r in term if r.get("excluded")),
        "terminal_states_total": sum(int(r.get("state_count") or 0) for r in term),
        "term_content_sha12s": term_shas, "all_region_content_sha12s": all_shas,
        "prev_item_id": chain.prev_item_id,
        "term_content_in_prev": int(term_in_prev),
        "term_content_total": len(term_shas),
        "all_content_in_prev": int(all_in_prev),
        "verify": {},
    }
    for m in cert["verifications"]:
        v = cert["verifications"][m]
        row["verify"][m] = {
            "pass": v.get("pass"), "call_seconds": v.get("call_seconds"),
            "n_reasons": len(v.get("reasons") or [])}
        if m == "incremental":
            cost = dict(v.get("cost") or {})
            row["verify"][m].update({
                "reused_regions": v.get("reused_regions"),
                "rechecked_regions": v.get("rechecked_regions"),
                "reused_states": int(cost.get("reused_states") or 0),
                "audit_states": int(cost.get("audit_states") or 0),
                "hits": int(cost.get("hits") or 0), "misses": int(cost.get("misses") or 0),
                "blocked_by_prev": int(cost.get("blocked_by_prev") or 0),
                "blocked_by_parent": int(cost.get("blocked_by_parent") or 0),
                "not_verified_regions": int(cost.get("not_verified_regions") or 0)})
    return row


def certificate_rows(trace: Mapping) -> List[dict]:
    """从 ``run_with_cost`` 的 trace 里取链条账行（已发证书，执行序）。"""
    return [dict(r) for r in (trace.get("certificate_rows") or [])]


def chain_cost(ledger, verifier_mode: str = "incremental") -> dict:
    """链条成本：逐证书终末区域内容复用率 + 验证时间 + （两模式在场时）判读与原因。

    ``ledger`` = ``{"rows": [...]}``（:func:`certificate_rows` 产出的行集，执行序）
    或行列表本身。``verifier_mode`` 选择本口径的主列（``full``/``incremental``）。
    判读规则（prereg §chain_verdict 冻结）：

    * ``reuse_state_share`` = Σreused_states / (Σreused_states + Σaudit_states)（确定性计数）；
    * ``inc_over_full`` = Σ verify.incremental.call_seconds / Σ verify.full.call_seconds；
    * 复用为 0 且存在被 prev 挡下区域 ⇒ ``CHAIN_NO_REUSE_NO_GAIN``；
    * 复用 >0 且 ``inc_over_full`` < 1 ⇒ ``CHAIN_DECLINE_WITH_REUSE``；
    * 复用 >0 且 ``inc_over_full`` ≥ 1 ⇒ ``CHAIN_NO_GAIN_DESPITE_REUSE``（原因逐条给数）。
    """
    if verifier_mode not in MODES:
        raise CostAccountingError("未知 verifier_mode %r" % (verifier_mode,))
    rows = list(ledger.get("rows") if isinstance(ledger, Mapping) else ledger)
    if not rows:
        raise CostAccountingError("空账：链条口径不得在空面上出账（SystemExit 由调用侧）")
    n_certs = len(rows)
    full_s = inc_s = 0.0
    n_full = n_inc = 0
    reused_states = audit_states = hits = misses = blocked_prev = blocked_parent = 0
    not_verified = 0
    term_total = 0
    term_in_prev = 0
    per_cert = []
    n_with_prev = 0
    n_with_reuse = 0
    n_pass_diff = 0
    for r in rows:
        v = r.get("verify") or {}
        f = v.get("full") or {}
        inc = v.get("incremental") or {}
        if f.get("call_seconds") is not None:
            full_s += float(f["call_seconds"])
            n_full += 1
        if inc.get("call_seconds") is not None:
            inc_s += float(inc["call_seconds"])
            n_inc += 1
        reused_states += int(inc.get("reused_states") or 0)
        audit_states += int(inc.get("audit_states") or 0)
        hits += int(inc.get("hits") or 0)
        misses += int(inc.get("misses") or 0)
        blocked_prev += int(inc.get("blocked_by_prev") or 0)
        blocked_parent += int(inc.get("blocked_by_parent") or 0)
        not_verified += int(inc.get("not_verified_regions") or 0)
        term_total += int(r.get("term_content_total") or 0)
        term_in_prev += int(r.get("term_content_in_prev") or 0)
        if r.get("prev_item_id") is not None:
            n_with_prev += 1
        if int(inc.get("reused_states") or 0) > 0:
            n_with_reuse += 1
        if f.get("pass") is not None and inc.get("pass") is not None \
                and bool(f["pass"]) != bool(inc["pass"]):
            n_pass_diff += 1
        per_cert.append({
            "item_id": r.get("item_id"), "world_id": r.get("world_id"),
            "strategy": r.get("strategy"), "t": r.get("t"),
            "prev_item_id": r.get("prev_item_id"),
            "term_content_total": r.get("term_content_total"),
            "term_content_in_prev": r.get("term_content_in_prev"),
            "term_reuse_vs_prev_share": (round(int(r.get("term_content_in_prev") or 0)
                                               / float(max(1, int(r.get("term_content_total")
                                                                 or 0))), 6)),
            "verify_full_seconds": f.get("call_seconds"),
            "verify_inc_seconds": inc.get("call_seconds"),
            "inc_over_full": (round(float(inc["call_seconds"]) / float(f["call_seconds"]), 6)
                              if f.get("call_seconds") and inc.get("call_seconds") else None),
            "reused_states": int(inc.get("reused_states") or 0),
            "audit_states": int(inc.get("audit_states") or 0),
            "hits": int(inc.get("hits") or 0), "misses": int(inc.get("misses") or 0),
            "blocked_by_prev": int(inc.get("blocked_by_prev") or 0),
            "blocked_by_parent": int(inc.get("blocked_by_parent") or 0),
        })
    share = (round(reused_states / float(max(1, reused_states + audit_states)), 6))
    out = {
        "schema": "s2c.chain_cost.v1",
        "verifier_mode": verifier_mode,
        "n_certs": n_certs,
        "n_certs_with_prev": n_with_prev,
        "n_certs_with_reuse": n_with_reuse,
        "terminal_regions_total": term_total,
        "terminal_regions_in_prev_total": term_in_prev,
        "term_reuse_vs_prev_share": round(term_in_prev / float(max(1, term_total)), 6),
        "verify_full_seconds_total": round(full_s, 6),
        "verify_inc_seconds_total": round(inc_s, 6),
        "reused_states": reused_states, "audit_states": audit_states,
        "reuse_state_share": share,
        "hits": hits, "misses": misses,
        "blocked_by_prev": blocked_prev, "blocked_by_parent": blocked_parent,
        "not_verified_regions": not_verified,
        "n_pass_diff": n_pass_diff,
        "per_cert": per_cert,
        "comparison": None,
        "basis": "MEASURED_ONLY",
    }
    if n_full == n_certs and n_inc == n_certs and full_s > 0:
        ratio = inc_s / full_s
        ratios = sorted(float(x["inc_over_full"]) for x in per_cert
                        if x["inc_over_full"] is not None)
        med = ratios[len(ratios) // 2] if ratios else None
        if reused_states == 0 and (blocked_prev > 0 or misses > 0):
            verdict = "CHAIN_NO_REUSE_NO_GAIN"
        elif reused_states > 0 and ratio < 1.0:
            verdict = "CHAIN_DECLINE_WITH_REUSE"
        elif reused_states > 0:
            verdict = "CHAIN_NO_GAIN_DESPITE_REUSE"
        else:
            verdict = "CHAIN_NO_GAIN"
        reasons = [
            "复用状态 %d / 重算状态 %d（复用占比 %.6f）" % (reused_states, audit_states, share),
            "验证墙钟 full %.6f s / inc %.6f s（比值 %.6f；逐张中位 %.6f）"
            % (full_s, inc_s, ratio, med if med is not None else float("nan")),
            "命中区域 %d / 未命中 %d（prev 条件挡下 %d、父稳定挡下 %d）"
            % (hits, misses, blocked_prev, blocked_parent),
            "终末区域内容在上一张证书中的复用计数 %d / %d" % (term_in_prev, term_total),
        ]
        out["comparison"] = {
            "inc_over_full": round(ratio, 6),
            "per_cert_ratio_median": med,
            "per_cert_ratio_min": (ratios[0] if ratios else None),
            "per_cert_ratio_max": (ratios[-1] if ratios else None),
            "verdict": verdict, "reasons": reasons,
        }
    return out
