# -*- coding: utf-8 -*-
"""S2-V · 增量验证器（复用面；判据与全验证器逐行对应、按内容寻址复用证据）。

本件是任务单 S2-V（sha12 ``c5ef28d09759``）§2.1 要求的**新建模块**；
``certificate_solver.py`` / ``statistical_certificate.py`` / ``identification.py``
只读（import 复用其验证侧助手，不改一字节）。

强制语义（任务单 §2.1 逐条）：

1. **同判据**：对同一证书，本模块与全验证器必须给同一 accept/reject。本件把全验证器
   （``certificate_solver.verify_certificate`` / ``statistical_certificate.
   verify_statistical_certificate``）的检查骨架**复制式镜像**到本模块，并在唯一的
   昂贵点（终末区域逐状态审计）插入缓存分支；原因串、检查顺序与 continue/break
   语义逐行对应。任何 accept/reject 差异 ⇒ 增量不可信，停（由调用侧按停机规则处理）。
2. **增量面**：只重用「父区域/状态集/证书字段未变」的子树——命中条件 = 区域内容
   sha12 相同 ∧ 表身份相同 ∧ 验证器版本相同 ∧ **当前父区域内容 sha12 与填证时相同**
   ∧（当 ``reuse_requires_prev=True`` 时）该内容在 ``previous_certificate`` 的区域内
   也出现（任务单字面「未变」= 相对上一张证书）。新增/变化区域逐节点重算。
3. **缓存键**：``sha12({"region_sha12", "table_id", "table_identity_sha12",
   "verifier_version"})``；命中即携带**重算证据**（外包计数、三路一致标志、方向标签、
   忠实性摘要、状态数摘要），绝不按「上次通过」放行——命中的每次仍重放声明比较、
   域/树/覆盖/顶层见证与见证重放（廉价部分逐次重算，昂贵部分用证据重绑定）。
4. **成本**：逐张返回 ``cost``（``basis=MEASURED_ONLY``：入口墙钟 + 审计/重用状态数 +
   命中/未命中/被策略挡下计数）。
5. **边界**：证书格式与验证器接口与 S2-N0/S2-N 一致；不新增「跳过验证」的快捷路径；
   不认识的 schema 原样转交 ``full_verifier``（更强路径，不是跳过）。
6. **状态上限**：``state_cap`` 只作**簿记政策**（§2.1 第 6 条）：命中不耗审计、不受限；
   未命中且 ``|st| > state_cap`` ⇒ 记 ``not_verified``、拒发 pass，**不得把未验证状态
   当已验证**。

口径：``research_line=S2V_incremental_verifier``；本层不写盘；输出只含整数/布尔/
字符串/整数对与耗时（``MEASURED_ONLY``）。
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import certificate_solver as cs
from . import identification as ident
from . import statistical_certificate as sc

INCREMENTAL_VERSION = "s2v.incremental/1"
KEY_SCHEMA = "s2v.region_key/v1"
EVIDENCE_SCHEMA = "s2v.region_evidence/v1"
IDENTITY_SCHEMA = "s2v.table_identity/v1"
RESULT_SCHEMA = "s2v.verify_incremental/v1"
S2_FLAVOR = "s2"
S2N_FLAVOR = "s2n"
PROBE_QID_A = "__S2V_PROBE_A__"
PROBE_QID_B = "__S2V_PROBE_B__"
PROBE_N_STATES = 16
_TZ = timezone(timedelta(hours=8))


class IncrementalVerifierError(ValueError):
    """本件内部契约破裂：一律抛出，不静默降级。"""


def _now() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 规范化 / 摘要
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组等转成稳定 JSON 形态（顺序敏感处由调用方先排序）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha12_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:12]


def _array_digest(arr) -> dict:
    a = np.asarray(arr).astype(np.int64)
    return {"n": int(a.shape[0]) if a.ndim else 0, "sum": int(a.sum()),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def _deriv_fingerprint(d: "ident.StateDerivation") -> str:
    return sha12_of({
        "nodes": sorted(str(x) for x in d.nodes),
        "edges": sorted([str(u), str(v)] for (u, v) in d.edges),
        "a_side": sorted(str(x) for x in d.a_side),
        "b_side": sorted(str(x) for x in d.b_side),
        "attach_a": bool(d.attach_a), "attach_b": bool(d.attach_b),
        "label": int(d.label)})


def _probe_states(n_states: int) -> List[int]:
    cand = {0, 1, 2, n_states - 1, n_states // 2, n_states // 3,
            (2 * n_states) // 3, n_states // 4, n_states // 5, n_states // 7,
            n_states // 8, (3 * n_states) // 8, (5 * n_states) // 8,
            (7 * n_states) // 8, n_states // 16, (15 * n_states) // 16}
    return sorted({int(x) for x in cand if 0 <= int(x) < n_states})[:PROBE_N_STATES]


def table_identity(table) -> dict:
    """表身份：覆盖派生闭包关合的一切可序列化数据；含 query_id 独立性探针。

    注意（登记性局限）：派生函数本体是闭包，不可哈希；本件用「声明数据摘要 +
    探针」绑定身份，代码版本由 ``verifier_version`` 承担。该局限按任务单 §2.1
    第 3 条登记在 prereg ``known_limits``。
    """
    tags = dict(getattr(table, "tags", {}) or {})
    labels = {}
    for k, v in (getattr(table, "labels", {}) or {}).items():
        labels[str(k)] = _array_digest(v)
    attach = {}
    for k, v in (getattr(table, "attach", {}) or {}).items():
        aa, bb = v
        attach[str(k)] = [_array_digest(aa), _array_digest(bb)]
    deriv = getattr(table, "derivation", None)
    probe = {"status": "NO_DERIVATION", "n_states": 0, "mismatches": 0,
             "qid_independent": None, "probe_qids": [PROBE_QID_A, PROBE_QID_B]}
    if deriv is not None:
        states = _probe_states(int(table.n_states))
        mism = []
        try:
            for s in states:
                fa = _deriv_fingerprint(deriv(PROBE_QID_A, int(s)))
                fb = _deriv_fingerprint(deriv(PROBE_QID_B, int(s)))
                if fa != fb:
                    mism.append(int(s))
            probe = {"status": "PROBED", "n_states": len(states),
                     "mismatches": len(mism), "mismatch_states": mism[:8],
                     "qid_independent": not mism,
                     "probe_qids": [PROBE_QID_A, PROBE_QID_B],
                     "probe_states": states}
        except Exception as e:  # noqa: BLE001  如实报探针失败，不猜
            probe = {"status": "PROBE_ERROR", "n_states": len(states),
                     "mismatches": None, "qid_independent": None,
                     "error": "%s: %s" % (type(e).__name__, e)}
    doc = {
        "schema": IDENTITY_SCHEMA,
        "table_id": str(table.table_id), "n_bits": int(table.n_bits),
        "n_states": int(table.n_states),
        "s2_structure": tags.get("s2_structure"),
        "s2_queries": tags.get("s2_queries"),
        "labels": labels,
        "baseline": [a.to_dict() for a in tuple(getattr(table, "baseline", ()) or ())],
        "actions": [a.to_dict() for a in tuple(getattr(table, "actions", ()) or ())],
        "attach": attach,
        "tags": _jsonable(tags),
        "derivation_probe": probe,
    }
    doc["identity_sha12"] = sha12_of(doc)
    return doc


def identity_sha12(table) -> str:
    return table_identity(table)["identity_sha12"]


# ---------------------------------------------------------------------------
# 区域内容 / 缓存键
# ---------------------------------------------------------------------------

def region_content(region: Mapping, flavor: str) -> dict:
    """区域内容 = 验证器**实际读取**的字段（region_id/parent_id/depth/state_ids 除外）。

    只纳入被读取字段：多纳入只会多失配（保守），少纳入会把没复验的东西放行（禁止）。
    """
    cubes = [[int(c["fixed_mask"]), int(c["fixed_vals"])]
             for c in (region.get("cubes") or [])]
    base = {"cubes": cubes, "is_terminal": bool(region.get("is_terminal")),
            "label_claim": region.get("label_claim"),
            "envelope_counts": region.get("envelope_counts"),
            "witness": region.get("witness")}
    if flavor == S2_FLAVOR:
        return base
    if flavor == S2N_FLAVOR:
        return dict(base, excluded=bool(region.get("excluded")),
                    statistical=region.get("statistical"))
    raise IncrementalVerifierError("未知 flavor %r" % (flavor,))


def region_content_sha12(region: Mapping, flavor: str) -> str:
    return sha12_of(region_content(region, flavor))


def region_key(content_sha12: str, table_ident_sha12: str, verifier_version: str) -> str:
    return sha12_of({"key_schema": KEY_SCHEMA, "region_sha12": content_sha12,
                     "table_id_note": "见表身份", "table_identity_sha12": table_ident_sha12,
                     "verifier_version": verifier_version})


def _parent_content_sha12(region: Mapping, by_id: Mapping, flavor: str) -> Optional[str]:
    p = region.get("parent_id")
    if p is None:
        return None
    pr = by_id.get(p)
    if pr is None:
        return "<PARENT_MISSING>"
    return region_content_sha12(pr, flavor)


class ReuseCache:
    """证据缓存：区域条目 + 两个纯函数记忆层（允许集 / 公开历史重放）。

    语义边界（prereg 登记）：缓存条目只由本模块的**全审计镜像**产生；命中只做
    「证据重绑定 + 廉价部分逐次重算」；剔除区的统计界依赖公开历史，**从不入缓存、
    逐张重算**。
    """

    def __init__(self) -> None:
        self.entries: Dict[str, dict] = {}
        self.allowed: Dict[Tuple[str, str, str], dict] = {}
        self.replay: Dict[Tuple[str, str], dict] = {}
        self.stats = {"region_lookups": 0, "hits": 0, "misses": 0,
                      "blocked_by_prev": 0, "blocked_by_parent": 0, "fills": 0,
                      "allowed_memo_hits": 0, "allowed_memo_misses": 0,
                      "replay_memo_hits": 0, "replay_memo_misses": 0,
                      "not_verified_regions": 0}

    def region_lookup(self, key: str) -> Optional[dict]:
        self.stats["region_lookups"] += 1
        return self.entries.get(key)

    def region_fill(self, key: str, evidence: dict) -> None:
        self.stats["fills"] += 1
        self.entries[key] = evidence


def _new_cost() -> dict:
    return {"region_lookups": 0, "hits": 0, "misses": 0, "blocked_by_prev": 0,
            "blocked_by_parent": 0, "fills": 0, "audit_states": 0,
            "reused_states": 0, "not_verified_regions": 0,
            "allowed_memo_hits": 0, "replay_memo_hits": 0}


def _memo_allowed(cache: ReuseCache, table, query_id: str, vv: str) -> Tuple[List[int], List[str], bool]:
    """``cs._v_allowed`` 的逐查询记忆层（纯函数重算的缓存，带摘要证据）。"""
    tid = identity_sha12(table)
    key = (str(table.table_id), tid, str(query_id), str(vv))
    ent = cache.allowed.get(key)
    if ent is not None:
        cache.stats["allowed_memo_hits"] += 1
        return list(ent["allowed"]), list(ent["why"]), True
    allowed, why = cs._v_allowed(table, query_id)
    cache.allowed[key] = {"allowed": list(allowed), "why": list(why),
                          "n": len(allowed), "sha12": sha12_of(sorted(allowed))}
    cache.stats["allowed_memo_misses"] += 1
    return list(allowed), list(why), False


def _prev_content_shas(previous_certificate: Optional[Mapping], flavor: str) -> Optional[set]:
    if previous_certificate is None:
        return None
    out = set()
    for r in (previous_certificate.get("regions") or []):
        out.add(region_content_sha12(r, flavor))
    return out


# ---------------------------------------------------------------------------
# s2 核：``cs.verify_certificate`` 的复制式镜像 + 缓存分支
# ---------------------------------------------------------------------------

def _assemble(res: dict, t0: float, cache: ReuseCache, extra: dict) -> dict:
    counters = dict(extra.pop("cost_counters", {}) or {})
    res["schema"] = RESULT_SCHEMA
    res["incremental_version"] = INCREMENTAL_VERSION
    res["cost"] = dict(counters, basis="MEASURED_ONLY",
                       incremental_seconds=round(time.perf_counter() - t0, 6))
    res["cache_stats"] = dict(cache.stats)
    res.update(extra)
    return res


def _inc_s2(prev: Optional[Mapping], cert: Mapping, table, query_id: str,
            cache: ReuseCache, reuse_requires_prev: bool,
            state_cap: Optional[int], vv: str) -> dict:
    t0 = time.perf_counter()
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    cost = _new_cost()
    tid = identity_sha12(table)

    if not isinstance(cert, Mapping):
        return _assemble({"pass": False, "reasons": ["证书不是映射"],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": {}, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    cert = dict(cert)
    unknown = [k for k in cert if k not in cs.CERT_KEYS]
    if unknown:
        reasons.append(f"证书含未知顶层字段 {unknown}")
    bad_keys = [k for k in cert if any(t in str(k).lower() for t in cs.FORBIDDEN_KEY_TOKENS)]
    if bad_keys:
        reasons.append(f"证书含真值/神谕字段 {bad_keys}")
    prov = cert.get("selection_provenance")
    if prov != cs.SELECTION_PROVENANCE:
        reasons.append(f"选点来源非法 {prov!r}（只许 {cs.SELECTION_PROVENANCE}）")
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != table.n_bits:
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    struct = cs.structure_of(table)
    n_bits = table.n_bits

    allowed, why, memo_hit = _memo_allowed(cache, table, query_id, vv)
    if memo_hit:
        cost["allowed_memo_hits"] = 1
    checks["allowed_recomputed"] = len(allowed)
    if why:
        reasons.extend(why)
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if declared_allowed != sorted(allowed):
        reasons.append(f"允许集不符：声明 {len(declared_allowed)} / 重算 {len(allowed)}")
    allowed_set = set(allowed)

    regions = list(cert.get("regions", []))
    if not regions:
        return _assemble({"pass": False, "reasons": reasons + ["证书无区域"],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": checks, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    ids = [r.get("region_id") for r in regions]
    if len(set(ids)) != len(ids):
        reasons.append("区域 id 重复")
    by_id = {r["region_id"]: r for r in regions}
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        reasons.append(f"根区域不唯一 ({len(roots)})")
        return _assemble({"pass": False, "reasons": reasons,
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": checks, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    root = roots[0]
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append(f"区域 {r.get('region_id')} 的父区域不存在")
            kids.setdefault(p, []).append(r)

    def states_of(r: Mapping) -> List[int]:
        # 与 cs.verify_certificate 内同一语义（逐调用现场相同：非法立方逐次记原因）
        out: set = set()
        for c in r.get("cubes", []):
            m, v = int(c["fixed_mask"]), int(c["fixed_vals"])
            if v & ~m:
                reasons.append(f"区域 {r.get('region_id')} 子立方非法")
                continue
            free = [b for b in range(n_bits) if not (m >> b) & 1]
            ss = [v]
            for b in free:
                ss = ss + [s | (1 << b) for s in ss]
            out.update(ss)
        return sorted(out)

    root_states = states_of(root)
    if root_states != sorted(allowed):
        reasons.append(f"根区域 != 允许集（{len(root_states)} vs {len(allowed)}）")

    # 树：内部节点子区域必须两两不交且并集 = 父（与原验证器同款显式状态集检查）
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = states_of(r)
        if r.get("is_terminal"):
            if ch:
                reasons.append(f"终末区域 {r.get('region_id')} 仍有子区域")
            if not st:
                reasons.append(f"终末区域 {r.get('region_id')} 为空")
            continue
        if not ch:
            reasons.append(f"内部区域 {r.get('region_id')} 无子区域")
            continue
        union: List[int] = []
        for c in ch:
            union.extend(states_of(c))
        if len(union) != len(set(union)):
            reasons.append(f"区域 {r.get('region_id')} 子区域重叠")
        if sorted(set(union)) != st:
            reasons.append(f"区域 {r.get('region_id')} 子区域并集 != 父区域")

    # 终末区域：三路外包 + 标签 + 见证 + 忠实性（含缓存分支）
    terminal = [r for r in regions if r.get("is_terminal")]
    claims: set = set()
    prev_contents = _prev_content_shas(prev, S2_FLAVOR) if reuse_requires_prev else set()
    for r in terminal:
        rid = r.get("region_id")
        st = states_of(r)
        claim = r.get("label_claim")
        if claim not in ("0", "1"):
            reasons.append(f"终末区域 {rid} 标签非法 {claim!r}")
            continue
        claims.add(claim)
        if not st:
            reasons.append(f"终末区域 {rid} 为空：空区域不得建外包、不得发证书")
            continue
        if not set(st) <= allowed_set:
            reasons.append(f"终末区域 {rid} 含允许集外状态")
        if table.derivation is None:
            reasons.append("表未提供派生：无法重算外包")
            continue
        content = region_content(r, S2_FLAVOR)
        csha = sha12_of(content)
        key = region_key(csha, tid, vv)
        entry = cache.region_lookup(key)
        parent_sha = _parent_content_sha12(r, by_id, S2_FLAVOR)
        hit = False
        if entry is None:
            cost["misses"] += 1
        elif entry.get("parent_content_sha12") != parent_sha:
            cost["misses"] += 1
            cost["blocked_by_parent"] += 1
        elif reuse_requires_prev and (prev_contents is None or csha not in prev_contents):
            cost["misses"] += 1
            cost["blocked_by_prev"] += 1
        else:
            hit = True
            cost["hits"] += 1
            cost["reused_states"] += entry["n_states"]
        if hit:
            # ---- 命中路径：廉价部分逐次重算；昂贵部分用证据重绑定 ----
            if not entry["three_routes"]["set_vs_quantifier"]:
                reasons.append(f"区域 {rid} 集合路与逐元素路不等")
            if not entry["three_routes"]["explicit_vs_set"]:
                reasons.append(f"区域 {rid} 验证器显式外包与 identification 集合路不等")
            declared = r.get("envelope_counts")
            if declared is not None and declared != entry["env_counts"]:
                reasons.append(f"区域 {rid} 声明外包计数与重算不等")
            lab_v = entry["lab_v"]
            if lab_v != entry["lab_s"]:
                reasons.append(f"区域 {rid} 验证器方向规则与 region_label 不等 "
                               f"({lab_v} vs {entry['lab_s']})")
            if lab_v == "INSTRUMENT_DEFECT_MONOTONICITY":
                reasons.append(f"区域 {rid} 外包单调性破裂")
                continue
            want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                    else ident.RegionLabel.ALL_DISCONNECTED.value)
            if lab_v != want:
                reasons.append(f"区域 {rid} 声明 {claim} 但重算 {lab_v}")
            fm = entry["faithfulness"]["mismatch"]
            if fm is not None:
                if fm["kind"] == "DERIV_NOT_FAITHFUL":
                    reasons.append(f"区域 {rid} 状态 {fm['state_id']} 派生与标签不忠实")
                else:
                    reasons.append(f"区域 {rid} 状态 {fm['state_id']} 标签 "
                                   f"{fm['label']} != 声明 {claim}")
        else:
            if state_cap is not None and len(st) > int(state_cap):
                reasons.append(f"区域 {rid} 新审计状态数 {len(st)} 超过声明上限 {state_cap}："
                               f"未验证状态不得当已验证（簿记政策，非标签判据）")
                cost["not_verified_regions"] += 1
                continue
            # ---- 未命中路径：全审计（与 cs.verify_certificate 逐行同款） ----
            derivs = [table.derivation(query_id, int(s)) for s in st]
            cost["audit_states"] += len(st)
            env_s = ident.envelope_of(derivs)
            env_q = ident.envelope_of_quantifier(derivs)
            env_v = cs._v_envelope_from_states(struct, st, n_bits)
            routes = {"set_vs_quantifier": env_s.to_dict() == env_q.to_dict(),
                      "explicit_vs_set": env_v.to_dict() == env_s.to_dict()}
            if not routes["set_vs_quantifier"]:
                reasons.append(f"区域 {rid} 集合路与逐元素路不等")
            if not routes["explicit_vs_set"]:
                reasons.append(f"区域 {rid} 验证器显式外包与 identification 集合路不等")
            env_counts = env_s.element_counts()
            declared = r.get("envelope_counts")
            if declared is not None and declared != env_counts:
                reasons.append(f"区域 {rid} 声明外包计数与重算不等")
            lab_v = cs._v_direction(env_v)
            lab_s = ident.region_label(env_s)
            if lab_v != lab_s:
                reasons.append(f"区域 {rid} 验证器方向规则与 region_label 不等 "
                               f"({lab_v} vs {lab_s})")
            evidence = {
                "schema": EVIDENCE_SCHEMA, "flavor": S2_FLAVOR,
                "content_sha12": csha, "table_id": str(table.table_id),
                "table_identity_sha12": tid, "verifier_version": vv,
                "incremental_version": INCREMENTAL_VERSION,
                "parent_content_sha12": parent_sha, "filled_at": _now(),
                "filled_by": "full_audit_mirror", "n_states": len(st),
                "states_sha12": sha12_of(sorted(st)), "env_counts": env_counts,
                "three_routes": routes, "lab_v": lab_v, "lab_s": lab_s,
                "faithfulness": {"n": len(st), "mismatch": None},
            }
            if lab_v == "INSTRUMENT_DEFECT_MONOTONICITY":
                reasons.append(f"区域 {rid} 外包单调性破裂")
                cache.region_fill(key, evidence)
                cost["fills"] += 1
                continue
            want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                    else ident.RegionLabel.ALL_DISCONNECTED.value)
            if lab_v != want:
                reasons.append(f"区域 {rid} 声明 {claim} 但重算 {lab_v}")
            for d in derivs:
                if int(d.label) != int(1 if ident.state_connected(d) else 0):
                    evidence["faithfulness"]["mismatch"] = {
                        "kind": "DERIV_NOT_FAITHFUL", "state_id": int(d.state_id),
                        "label": int(d.label)}
                    reasons.append(f"区域 {rid} 状态 {d.state_id} 派生与标签不忠实")
                    break
                if int(d.label) != int(claim):
                    evidence["faithfulness"]["mismatch"] = {
                        "kind": "LABEL_NE_CLAIM", "state_id": int(d.state_id),
                        "label": int(d.label)}
                    reasons.append(f"区域 {rid} 状态 {d.state_id} 标签 "
                                   f"{d.label} != 声明 {claim}")
                    break
            cache.region_fill(key, evidence)
            cost["fills"] += 1
        # 见证：两条路径都逐次重算（廉价；且对 query 语义最敏感的一段不复用）
        w = r.get("witness")
        if not w or "state_id" not in w:
            reasons.append(f"区域 {rid} 无见证")
        else:
            sid = int(w["state_id"])
            if sid not in set(st):
                reasons.append(f"区域 {rid} 见证 {sid} 不在区域内")
            elif int(w.get("label", -1)) != int(claim):
                reasons.append(f"区域 {rid} 见证标签与区域标签不符")
            else:
                rep = ident.reverify_witness(table.derivation(query_id, sid), int(claim),
                                             w.get("path"))
                if not rep["ok"]:
                    reasons.append(f"区域 {rid} 见证不可重放 {rep.get('reason')}")

    # 覆盖：终末区域两两不交、并集 = 允许集
    all_states: List[int] = []
    for r in terminal:
        all_states.extend(states_of(r))
    if len(all_states) != len(set(all_states)):
        reasons.append("终末区域划分不互斥")
    if sorted(set(all_states)) != sorted(allowed):
        missing = sorted(set(allowed) - set(all_states))
        extra = sorted(set(all_states) - set(allowed))
        reasons.append(f"终末区域覆盖 != 允许集 (缺 {len(missing)} / 多 {len(extra)})")
    if len(claims) > 1:
        reasons.append(f"终末区域标签不一致 {sorted(claims)}：混合查询不得发单标签证书")
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    elif int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
        reasons.append("证书顶层见证标签与 label_claim 不符")
    checks["regions"] = len(regions)
    checks["terminal_regions"] = len(terminal)
    checks["audit_states"] = cost["audit_states"]
    checks["reused_states"] = cost["reused_states"]
    checks["claims"] = sorted(claims)
    return _assemble({"pass": not reasons, "reasons": reasons,
                      "reused_regions": cost["hits"],
                      "rechecked_regions": cost["misses"],
                      "verifier_version": cs.VERIFIER_VERSION,
                      "flavor": S2_FLAVOR},
                     t0, cache, {"checks": checks, "cost_counters": cost,
                                 "table_identity_sha12": tid})


# ---------------------------------------------------------------------------
# s2n 核：``sc.verify_statistical_certificate`` 的复制式镜像 + 缓存分支
# ---------------------------------------------------------------------------

def _memo_replay(cache: ReuseCache, public, vv: str) -> Tuple[dict, List[str], bool]:
    h12 = sc.history_sha12(public)
    key = (h12, vv)
    ent = cache.replay.get(key)
    if ent is not None:
        cache.stats["replay_memo_hits"] += 1
        return ent["rep"], list(ent["reasons"]), True
    buf: List[str] = []
    rep = sc._v_replay(public, buf)
    cache.replay[key] = {"rep": rep, "reasons": list(buf),
                         "history_sha12": h12,
                         "states_sha12": sha12_of(sorted(int(s) for s in rep["states"]))}
    cache.stats["replay_memo_misses"] += 1
    return rep, list(buf), False


def _inc_s2n(prev: Optional[Mapping], cert: Mapping, table, public, query_id: str,
             cache: ReuseCache, reuse_requires_prev: bool,
             state_cap: Optional[int], vv: str) -> dict:
    t0 = time.perf_counter()
    if not isinstance(cert, Mapping):
        return _assemble({"pass": False, "reasons": ["证书不是映射"],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": {}, "cost_counters": _new_cost(),
                                     "table_identity_sha12": identity_sha12(table)})
    cert = dict(cert)
    if str(cert.get("schema")) == cs.SCHEMA_CERT:
        out = _inc_s2(prev, cert, table, query_id, cache, reuse_requires_prev,
                      state_cap, vv)
        res = dict(out)
        res["s2n"] = {"reused_s2_verifier": True,
                      "note": "s2.certificate/v1 原样转交本件 s2 增量核（接口复用）"}
        return res
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    cost = _new_cost()
    tid = identity_sha12(table)
    if str(cert.get("schema")) != sc.SCHEMA_STAT_CERT:
        return _assemble({"pass": False,
                          "reasons": ["未知证书 schema %r" % (cert.get("schema"),)],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": checks, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    unknown = [k for k in cert if k not in sc.CERT_KEYS]
    if unknown:
        reasons.append("证书含未知顶层字段 %s" % (unknown,))
    bad = [k for k in cert if any(t in str(k).lower() for t in sc.FORBIDDEN_KEY_TOKENS)]
    if bad:
        reasons.append("证书含真值/神谕字段 %s" % (bad,))
    if "certified" in cert:
        reasons.append("证书含 certified 布尔：验证器不认该字段")
    if cert.get("selection_provenance") != sc.SELECTION_PROVENANCE:
        reasons.append("选点来源非法 %r" % (cert.get("selection_provenance"),))
    if cert.get("verifier_version") != sc.VERIFIER_VERSION:
        reasons.append("证书 verifier_version 不是 %s" % sc.VERIFIER_VERSION)
    if cert.get("query_id") != query_id:
        reasons.append("证书 query_id 与验证请求不符")
    if int(cert.get("n_bits", -1)) != int(table.n_bits):
        reasons.append("证书 n_bits 与表不符")
    if cert.get("table_id") != table.table_id:
        reasons.append("证书 table_id 与表不符")
    if not isinstance(public, sc.PublicEvidence):
        raise sc.StatisticalCertificateError("验证器只接受 PublicEvidence（公开历史对象）")
    if str(cert.get("model_id")) != public.model_id:
        reasons.append("证书 model_id 与公开历史不符")
    s2n = dict(cert.get("s2n") or {})
    if str(s2n.get("history_sha12")) != sc.history_sha12(public):
        reasons.append("公开历史摘要不符：证书不是对这段观测出的")
    if str(s2n.get("tier")) != str(public.tier):
        reasons.append("噪声档不符 (%r vs %r)" % (s2n.get("tier"), public.tier))
    alpha_seq = sc._frac(s2n.get("alpha_seq", "0"))
    if not (0 < alpha_seq < 1):
        reasons.append("证书 alpha_seq 非法 %r" % (s2n.get("alpha_seq"),))
    thr = 1 / alpha_seq
    n_bits = int(table.n_bits)
    struct = cs.structure_of(table)
    rep, replay_reasons, replay_hit = _memo_replay(cache, public, vv)
    if replay_hit:
        cost["replay_memo_hits"] = 1
    reasons.extend(replay_reasons)
    Q, D, E, P, vstates = rep["Q"], rep["D"], rep["E"], rep["P"], rep["states"]
    t_hist = len(Q) - 1
    checks["history_t"] = t_hist
    if int(s2n.get("t", -1)) != t_hist:
        reasons.append("证书声明的 t 与公开历史长度不符")

    mode = str(cert.get("domain_mode"))
    if mode == "ALL_STATES":
        domain = list(range(1 << n_bits))
    elif mode == "CONSISTENT_SET":
        domain = sc._v_consistent_states(table, query_id)
    else:
        reasons.append("未知 domain_mode %r" % (mode,))
        domain = []
    declared_allowed = [int(s) for s in cert.get("allowed_state_ids", [])]
    if declared_allowed != sorted(domain):
        reasons.append("域名不符：声明 %d / 重算 %d" % (len(declared_allowed), len(domain)))
    if not domain:
        reasons.append("空域：空集不得发证书")

    regions = list(cert.get("regions", []))
    checks["regions"] = len(regions)
    if not regions:
        return _assemble({"pass": False, "reasons": reasons + ["证书无区域"],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": checks, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    ids = [r.get("region_id") for r in regions]
    if len(set(ids)) != len(ids):
        reasons.append("区域 id 重复")
    by_id = {r["region_id"]: r for r in regions}
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        return _assemble({"pass": False, "reasons": reasons + ["根区域不唯一 (%d)" % len(roots)],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": checks, "cost_counters": cost,
                                     "table_identity_sha12": tid})
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if p not in by_id:
                reasons.append("区域 %r 的父区域不存在" % (r.get("region_id"),))
            kids.setdefault(p, []).append(r)
    st_of = {r["region_id"]: sc._v_states_of(r.get("cubes", []), n_bits, reasons,
                                             r.get("region_id")) for r in regions}
    if st_of[roots[0]["region_id"]] != sorted(domain):
        reasons.append("根区域 != 域（%d vs %d）"
                       % (len(st_of[roots[0]["region_id"]]), len(domain)))
    for r in regions:
        ch = kids.get(r.get("region_id"), [])
        st = st_of[r["region_id"]]
        if r.get("is_terminal"):
            if ch:
                reasons.append("终末区域 %r 仍有子区域" % (r.get("region_id"),))
            if not st:
                reasons.append("终末区域 %r 为空" % (r.get("region_id"),))
            continue
        if not ch:
            reasons.append("内部区域 %r 无子区域" % (r.get("region_id"),))
            continue
        union: List[int] = []
        for c in ch:
            union.extend(st_of[c["region_id"]])
        if len(union) != len(set(union)):
            reasons.append("区域 %r 子区域重叠" % (r.get("region_id"),))
        if sorted(set(union)) != st:
            reasons.append("区域 %r 子区域并集 != 父区域" % (r.get("region_id"),))

    idx_of = {s: i for i, s in enumerate(vstates)}
    n_excl = 0
    n_surv = 0
    claims: set = set()
    witness_holes: List[str] = []
    prev_contents = _prev_content_shas(prev, S2N_FLAVOR) if reuse_requires_prev else set()
    for r in regions:
        if not r.get("is_terminal"):
            continue
        rid = r.get("region_id")
        st = st_of[rid]
        excluded = bool(r.get("excluded"))
        if excluded:
            # 剔除区：统计界依赖公开历史 —— 从不复用，逐张重算（与原件逐行同款）
            n_excl += 1
            stat = dict(r.get("statistical") or {})
            if not stat.get("excluded"):
                reasons.append("区域 %r 标了剔除但 statistical 段非剔除" % (rid,))
                continue
            if r.get("label_claim") is not None:
                reasons.append("剔除区域 %r 不得同时标标签" % (rid,))
                continue
            u = int(stat.get("u", 0))
            if not (1 <= u <= t_hist):
                reasons.append("区域 %r 的 u=%r 越出历史 (1..%d)" % (rid, u, t_hist))
                continue
            statistic = str(stat.get("statistic"))
            if statistic not in (sc.STAT_JOINT, sc.STAT_STEPWISE):
                reasons.append("区域 %r 的统计量 %r 非法" % (rid, statistic))
            if str(stat.get("bound_rule")) != sc.BOUND_RULE:
                reasons.append("区域 %r 的界规则 %r 非法" % (rid, stat.get("bound_rule")))
            d_vals = [D[u][idx_of[s]] for s in st if s in idx_of]
            if len(d_vals) != len(st):
                reasons.append("区域 %r 含公开历史状态集外状态" % (rid,))
                continue
            sup_d = max(d_vals)
            if statistic == sc.STAT_STEPWISE:
                step = sc.Fraction(1, 1)
                for v in range(1, u + 1):
                    step *= max(P[v - 1][idx_of[s]] for s in st)
                target = step
            else:
                target = sup_d
            q = Q[u]
            if not sc._is_inf(stat.get("sup_D_u")):
                declared = sc._frac(stat.get("sup_D_u"))
                if declared != target:
                    reasons.append("区域 %r 声明的 sup_D_u 与重算（%s）不等"
                                   % (rid, statistic))
                for s, dv in zip(st, d_vals):
                    if dv > declared:
                        reasons.append("区域 %r 的界不是上界：状态 %d 的 D_u=%s > %s"
                                       % (rid, s, dv, declared))
                        break
                if statistic == sc.STAT_STEPWISE and sup_d > declared:
                    reasons.append("区域 %r 用联合上确界去界逐步统计量：联合 sup=%s > 声明 %s"
                                   % (rid, sup_d, declared))
            if not sc._is_inf(stat.get("Q_u")) and sc._frac(stat.get("Q_u")) != q:
                reasons.append("区域 %r 声明的 Q_u 与重算不等" % (rid,))
            survivors = []
            for s in st:
                i = idx_of[s]
                emax = sc._max_e([E[v][i] for v in range(1, t_hist + 1)])
                if not (sc._is_inf(emax) or sc._frac(emax) >= thr):
                    survivors.append(s)
            if survivors:
                reasons.append("剔除区域 %r 仍含未被统计排除的状态 %s"
                               % (rid, survivors[:8]))
            if target != 0 and q / target < thr:
                reasons.append("区域 %r 的 Q_u/界 未达 1/alpha（重算口径 %s）" % (rid, statistic))
            declared_emax = dict(stat.get("per_state_emax") or {})
            for s in st:
                if str(s) not in declared_emax:
                    reasons.append("区域 %r 缺状态 %d 的 per_state_emax" % (rid, s))
                    break
            cost["audit_states"] += len(st)
            continue
        n_surv += 1
        claim = r.get("label_claim")
        if claim not in ("0", "1"):
            reasons.append("终末区域 %r 标签非法 %r（既非剔除也非 0/1）" % (rid, claim))
            continue
        claims.add(claim)
        if not st:
            reasons.append("终末区域 %r 为空" % (rid,))
            continue
        if not set(st) <= set(domain):
            reasons.append("终末区域 %r 含域外状态" % (rid,))
        if table.derivation is None:
            reasons.append("表未提供派生：无法重算外包")
            continue
        content = region_content(r, S2N_FLAVOR)
        csha = sha12_of(content)
        key = region_key(csha, tid, vv)
        entry = cache.region_lookup(key)
        parent_sha = _parent_content_sha12(r, by_id, S2N_FLAVOR)
        hit = False
        if entry is None:
            cost["misses"] += 1
        elif entry.get("parent_content_sha12") != parent_sha:
            cost["misses"] += 1
            cost["blocked_by_parent"] += 1
        elif reuse_requires_prev and (prev_contents is None or csha not in prev_contents):
            cost["misses"] += 1
            cost["blocked_by_prev"] += 1
        else:
            hit = True
            cost["hits"] += 1
            cost["reused_states"] += entry["n_states"]
        if hit:
            if not entry["three_routes"]["set_vs_quantifier"]:
                reasons.append("区域 %r 集合路与逐元素路不等" % (rid,))
            if not entry["three_routes"]["explicit_vs_set"]:
                reasons.append("区域 %r 验证器显式路与集合路不等" % (rid,))
            declared_env = r.get("envelope_counts")
            if declared_env is not None and declared_env != entry["env_counts"]:
                reasons.append("区域 %r 声明外包计数与重算不等" % (rid,))
            lab = entry["lab_v"]
            if lab == "INSTRUMENT_DEFECT_MONOTONICITY":
                reasons.append("区域 %r 外包单调性破裂" % (rid,))
                continue
            want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                    else ident.RegionLabel.ALL_DISCONNECTED.value)
            if lab != want:
                reasons.append("区域 %r 声明 %s 但重算 %s" % (rid, claim, lab))
            fm = entry["faithfulness"]["mismatch"]
            if fm is not None:
                if fm["kind"] == "DERIV_NOT_FAITHFUL":
                    reasons.append("区域 %r 状态 %d 派生与标签不忠实"
                                   % (rid, fm["state_id"]))
                else:
                    reasons.append("区域 %r 状态 %d 标签 %d != 声明 %s"
                                   % (rid, fm["state_id"], fm["label"], claim))
        else:
            if state_cap is not None and len(st) > int(state_cap):
                reasons.append("区域 %r 新审计状态数 %d 超过声明上限 %s："
                               "未验证状态不得当已验证（簿记政策，非标签判据）"
                               % (rid, len(st), state_cap))
                cost["not_verified_regions"] += 1
                continue
            derivs = [table.derivation(query_id, int(s)) for s in st]
            cost["audit_states"] += len(st)
            env_set = ident.envelope_of(derivs)
            env_q = ident.envelope_of_quantifier(derivs)
            env_exp = sc._v_explicit_env(struct, st, n_bits)
            routes = {"set_vs_quantifier": env_set.to_dict() == env_q.to_dict(),
                      "explicit_vs_set": env_exp.to_dict() == env_set.to_dict()}
            if not routes["set_vs_quantifier"]:
                reasons.append("区域 %r 集合路与逐元素路不等" % (rid,))
            if not routes["explicit_vs_set"]:
                reasons.append("区域 %r 验证器显式路与集合路不等" % (rid,))
            env_counts = env_set.element_counts()
            declared_env = r.get("envelope_counts")
            if declared_env is not None and declared_env != env_counts:
                reasons.append("区域 %r 声明外包计数与重算不等" % (rid,))
            lab = sc._v_direction(env_exp)
            mismatch = None
            if lab == "INSTRUMENT_DEFECT_MONOTONICITY":
                reasons.append("区域 %r 外包单调性破裂" % (rid,))
            else:
                want = (ident.RegionLabel.ALL_CONNECTED.value if claim == "1"
                        else ident.RegionLabel.ALL_DISCONNECTED.value)
                if lab != want:
                    reasons.append("区域 %r 声明 %s 但重算 %s" % (rid, claim, lab))
                for d in derivs:
                    if int(d.label) != int(1 if ident.state_connected(d) else 0):
                        mismatch = {"kind": "DERIV_NOT_FAITHFUL",
                                    "state_id": int(d.state_id), "label": int(d.label)}
                        reasons.append("区域 %r 状态 %d 派生与标签不忠实" % (rid, d.state_id))
                        break
                    if int(d.label) != int(claim):
                        mismatch = {"kind": "LABEL_NE_CLAIM",
                                    "state_id": int(d.state_id), "label": int(d.label)}
                        reasons.append("区域 %r 状态 %d 标签 %d != 声明 %s"
                                       % (rid, d.state_id, d.label, claim))
                        break
            evidence = {
                "schema": EVIDENCE_SCHEMA, "flavor": S2N_FLAVOR,
                "content_sha12": csha, "table_id": str(table.table_id),
                "table_identity_sha12": tid, "verifier_version": vv,
                "incremental_version": INCREMENTAL_VERSION,
                "parent_content_sha12": parent_sha, "filled_at": _now(),
                "filled_by": "full_audit_mirror", "n_states": len(st),
                "states_sha12": sha12_of(sorted(st)), "env_counts": env_counts,
                "three_routes": routes, "lab_v": lab, "lab_s": lab,
                "faithfulness": {"n": len(st), "mismatch": mismatch},
            }
            cache.region_fill(key, evidence)
            cost["fills"] += 1
            if lab == "INSTRUMENT_DEFECT_MONOTONICITY":
                continue
        w = r.get("witness")
        if not w or "state_id" not in w:
            reasons.append("区域 %r 无见证" % (rid,))
        else:
            sid = int(w["state_id"])
            if sid not in set(st):
                reasons.append("区域 %r 见证 %d 不在区域内" % (rid, sid))
            elif int(w.get("label", -1)) != int(claim):
                reasons.append("区域 %r 见证标签与区域标签不符" % (rid,))
            else:
                rep_w = ident.reverify_witness(table.derivation(query_id, sid),
                                               int(claim), w.get("path"))
                if not rep_w["ok"]:
                    reasons.append("区域 %r 见证不可重放 %r" % (rid, rep_w.get("reason")))
            if sid in idx_of:
                i = idx_of[sid]
                emax_w = sc._max_e([E[v][i] for v in range(1, t_hist + 1)])
                if not sc._is_inf(emax_w) and sc._frac(emax_w) >= thr:
                    witness_holes.append(rid)
    # 覆盖：域内每状态要么在剔除节点，要么在被声明标签的存活节点
    surv_states: List[int] = []
    for r in regions:
        if r.get("is_terminal") and not r.get("excluded"):
            surv_states.extend(st_of[r["region_id"]])
    all_terminal: List[int] = []
    for r in regions:
        if r.get("is_terminal"):
            all_terminal.extend(st_of[r["region_id"]])
    if len(all_terminal) != len(set(all_terminal)):
        reasons.append("终末区域划分不互斥")
    if sorted(set(all_terminal)) != sorted(domain):
        missing = sorted(set(domain) - set(all_terminal))
        extra = sorted(set(all_terminal) - set(domain))
        reasons.append("终末区域覆盖 != 域 (缺 %d / 多 %d)" % (len(missing), len(extra)))
    if len(claims) > 1:
        reasons.append("存活终末区域标签不一致 %s：混合查询不得发单标签证书"
                       % (sorted(claims),))
    top = cert.get("witness") or {}
    if not top or "state_id" not in top:
        reasons.append("证书顶层无见证")
    else:
        if int(top.get("label", -1)) != int(cert.get("label_claim", -1)):
            reasons.append("证书顶层见证标签与 label_claim 不符")
        sid = int(top["state_id"])
        if sid in idx_of:
            i = idx_of[sid]
            emax_t = sc._max_e([E[v][i] for v in range(1, t_hist + 1)])
            if not sc._is_inf(emax_t) and sc._frac(emax_t) >= thr:
                reasons.append("证书顶层见证状态已被统计排除")
    possible = []
    for s in domain:
        if s not in idx_of:
            continue
        i = idx_of[s]
        emax = sc._max_e([E[v][i] for v in range(1, t_hist + 1)])
        if not (sc._is_inf(emax) or sc._frac(emax) >= thr):
            possible.append(s)
    if not possible:
        reasons.append("统计可行集为空（S_t=∅）：空集不得发证书")
    if int(s2n.get("n_statistical_exclusions", -1)) != n_excl:
        reasons.append("证书声明的剔除节点数与重算不符")
    if int(s2n.get("n_possible_states", -1)) != len(possible):
        reasons.append("证书声明的可行状态数 %r 与重算 %d 不符"
                       % (s2n.get("n_possible_states"), len(possible)))
    if int(s2n.get("n_surviving_states", -1)) != len(set(surv_states)):
        reasons.append("证书声明的存活区域状态数与重算不符")
    if bool(cert.get("label_claim")) is False and cert.get("label_claim") not in ("0", "1"):
        reasons.append("证书 label_claim 非法 %r" % (cert.get("label_claim"),))
    if claims and str(cert.get("label_claim")) not in claims:
        reasons.append("证书 label_claim 与存活区域标签不符")
    checks.update({"excluded_terminals": n_excl, "surviving_terminals": n_surv,
                   "surviving_states": len(set(surv_states)),
                   "possible_states": len(possible),
                   "regions_with_excluded_witness": len(witness_holes),
                   "audit_states": cost["audit_states"],
                   "reused_states": cost["reused_states"],
                   "claims": sorted(claims), "alpha_seq": str(alpha_seq),
                   "threshold": str(thr)})
    return _assemble({"pass": not reasons, "reasons": reasons,
                      "reused_regions": cost["hits"],
                      "rechecked_regions": cost["misses"],
                      "verifier_version": sc.VERIFIER_VERSION,
                      "flavor": S2N_FLAVOR,
                      "s2n": {"reused_s2_verifier": False, "independent_replay": True,
                              "calls_production_solver": False}},
                     t0, cache, {"checks": checks, "cost_counters": cost,
                                 "table_identity_sha12": tid})


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def verify_incremental(previous_certificate, new_certificate, table, *,
                       full_verifier, cache: Optional[ReuseCache] = None,
                       public_evidence=None, verifier_version: Optional[str] = None,
                       reuse_requires_prev: bool = True,
                       state_cap: Optional[int] = None) -> dict:
    """增量重放（接口按任务单 §2.1；返回 ``pass/reasons/reused_regions/
    rechecked_regions/cost`` 及检查读数）。

    - ``previous_certificate``：diff 基准（``reuse_requires_prev=True`` 时，只有在该
      证书区域集合中出现过的区域内容才允许命中）。``None`` ⇒ 全部按新增处理。
    - ``full_verifier``：全验证器可调用体（同上下文绑定）；只在 schema 不认识时
      原样转交（更强路径）。
    - ``cache``：``ReuseCache`` 或 ``None``（自动新建）。跨证书调用请复用同一实例。
    - ``public_evidence``：s2n 统计证书所需的公开历史（``sc.PublicEvidence``）。
    - ``verifier_version``：显式覆盖缓存键里的验证器版本（默认按 flavor 取全验证器版本）。
    - ``reuse_requires_prev``：True = 任务单字面「未变 vs 上一张证书」；False = 批次
      累计口径（任何早先验证过的同内容区域可命中；作为次生口径登记）。
    - ``state_cap``：簿记政策（见模块头 §6）；``None`` = 不设限。
    """
    t0 = time.perf_counter()
    if cache is None:
        cache = ReuseCache()
    if not isinstance(new_certificate, Mapping):
        return _assemble({"pass": False, "reasons": ["证书不是映射"],
                          "reused_regions": 0, "rechecked_regions": 0},
                         t0, cache, {"checks": {}, "cost_counters": _new_cost(),
                                     "table_identity_sha12": None,
                                     "flavor": None})
    schema = str(dict(new_certificate).get("schema"))
    if schema == cs.SCHEMA_CERT:
        vv = str(verifier_version or f"{cs.VERIFIER_VERSION}+{INCREMENTAL_VERSION}")
        out = _inc_s2(previous_certificate, new_certificate, table,
                      str(new_certificate.get("query_id")), cache,
                      bool(reuse_requires_prev), state_cap, vv)
    elif schema == sc.SCHEMA_STAT_CERT:
        vv = str(verifier_version or f"{sc.VERIFIER_VERSION}+{INCREMENTAL_VERSION}")
        if public_evidence is None:
            raise IncrementalVerifierError(
                "s2.statistical_certificate/v1 需要 public_evidence（公开历史）")
        out = _inc_s2n(previous_certificate, new_certificate, table, public_evidence,
                       str(new_certificate.get("query_id")), cache,
                       bool(reuse_requires_prev), state_cap, vv)
    else:
        res = dict(full_verifier(new_certificate))
        out = _assemble({"pass": bool(res.get("pass")),
                         "reasons": list(res.get("reasons", [])),
                         "reused_regions": 0, "rechecked_regions": 0},
                        t0, cache, {"checks": res.get("checks"), "cost_counters": _new_cost(),
                                    "table_identity_sha12": None, "flavor": None,
                                    "fallback": "UNSUPPORTED_SCHEMA_VIA_FULL_VERIFIER"})
    out["reuse"] = {"policy": "STRICT_PREV" if reuse_requires_prev else "BATCH_CACHE",
                    "requires_prev": bool(reuse_requires_prev),
                    "state_cap": state_cap}
    return out


__all__ = [
    "INCREMENTAL_VERSION", "KEY_SCHEMA", "EVIDENCE_SCHEMA", "RESULT_SCHEMA",
    "S2_FLAVOR", "S2N_FLAVOR", "IncrementalVerifierError", "ReuseCache",
    "canonical_json", "sha12_of", "table_identity", "identity_sha12",
    "region_content", "region_content_sha12", "region_key", "verify_incremental",
]
