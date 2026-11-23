# -*- coding: utf-8 -*-
"""S2-S2 · 并行通道结构的闭合模块（通道分解 + 逐通道证书 + 组合判据 + 独立验证器）。

本件是任务单 S2-S2（sha12 ``5f4006a528c5``）§2.1 要求的**新建模块**；
``certificate_solver.py`` / ``finite_scale.py`` / ``identification.py`` / ``finite_scope.py``
一字未改（只 import）。判据权威 = 任务单 §4 + ``results/s2s2_parallel_channel/prereg_s2s2.json``。

设计（预注册口径，先于第一次分解冻结）
----------------------------------------
1. **通道分解**：按表的**声明结构**（``s2_structure``）与**声明读位**（``s2_queries``）把查询拆成
   并行通道。对 S2-S 的两族声明表：
   - ``SER``：每个阶段 = 一个单位通道（1 位）；``PAR``：每个阶段 = 一个通道（2 位并行）。
   - 只有**路径阶段**（stage < target_node）进入通道层；路径外阶段登记为
     ``off_path_stages``（对标签无影响，逐位声明）。
   - **仅承认单位读**（每个读固定一个位）。多位置读（组读）会让允许集不再是通道投影之积 ⇒
     分解不适用，登记 ``DECOMPOSITION_NOT_APPLICABLE``，不删、不猜。
   - 允许集 = 通道局部允许集之积 × 路径外读位（符号化相等检查：固定掩码与取值逐位对账）。
2. **通道状态**（纯由局部子立方 + 声明标签规则推出，不看任何标签）：
   - ``CHANNEL_ALL1``：局部每个状态 g=1（通道至少一位固定为 1；记 ``designated_bit``）；
   - ``CHANNEL_ALL0``：局部每个状态 g=0（通道两位全固定为 0）；
   - ``CHANNEL_MIXED``：两种都有（记局部见证对 witness_0 / witness_1）。
3. **组合判据**（声明标签 = 路径通道「非空」之合取 AND）：
   - 任一通道 ``ALL0`` ⇒ 组合标签 {0}（may 割集：该阶段无 may 边）；
   - 全通道 ``ALL1`` ⇒ 组合标签 {1}（must 路径：每阶段一条固定为 1 的边）；
   - 否则 ⇒ 标签集 {0,1}（混合）：由积结构造双见证（一 1 一 0），并给出全覆盖划分树。
   - 任一通道未决 ⇒ 组合未决（不删、不猜）。
4. **证书分层**：
   - 单标签：组合证书 = 严格 ``s2.certificate/v1``（单根区域树，由冻结 ``certificate_solver.certify``
     签发，可被冻结验证器 ``verify_certificate`` 独立重放）；
   - 混合：``s2s2.combination_proof/v1``（区域/子立方/见证词汇与 ``s2.certificate/v1`` 同构；
     冻结验证器按设计拒收混合单标签证书，故混合只走本模块验证器 + 冻结 ``certify`` 交叉核对）；
   - 通道层：``s2s2.channel_certificate/v1``（逐通道局部允许集/状态/见证，独立验证）。
5. **独立验证器**：本模块 ``verify_*`` 自带结构解析、允许集重算、通道重算、外包方向规则与
   见证重放；**不调用** ``solve_partial`` / ``certify`` / ``verify_certificate``，**不读**任何
   ``certified`` 布尔，**不读** ``table.labels``；见证重放只消费 ``table.derivation`` 的三要素
   并按自带 BFS 与自带「链式合取」两路互校。
6. **不外推**：只对声明档（d=12/16/18 的 SER/PAR × END/MID 表族与 9 个声明情景）负责。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from fractureflow import certificate_solver as cs
from fractureflow import identification as ident

SCHEMA_CHANNEL_CERT = "s2s2.channel_certificate/v1"
SCHEMA_CHANNEL_BUNDLE = "s2s2.channel_bundle/v1"
SCHEMA_COMBINATION_PROOF = "s2s2.combination_proof/v1"
SCHEMA_VERIFY = "s2s2.verify/v1"
VERIFIER_VERSION = "s2s2.verifier/1"
SELECTION_PROVENANCE = "SYMBOLIC_ENVELOPE"
WATERMARK = {"research_line": "S2S2_parallel_channel",
             "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N0",
             "dictionary": "COMPLETE_KNOWN", "claim_scope": "DECLARED_SYNTHETIC_TABLE"}
STATUS_ALL1 = "CHANNEL_ALL1"
STATUS_ALL0 = "CHANNEL_ALL0"
STATUS_MIXED = "CHANNEL_MIXED"
FORBIDDEN_KEY_TOKENS = ("truth", "oracle", "hidden", "certified")
CHANNEL_CERT_KEYS = ("schema", "channel_id", "table_id", "query_id", "stage", "bits",
                     "nodes", "local", "status", "designated_bit", "witness_0", "witness_1",
                     "costs", "budget", "watermark")
PROOF_KEYS = ("schema", "proof_id", "table_id", "query_id", "n_bits", "decomposition",
              "partition", "witnesses", "combination_rule", "frozen_crosscheck", "budget",
              "costs", "stats", "verifier_version", "selection_provenance", "watermark")


class ParallelChannelError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 纯整数子立方工具（验证器自带一份，不依赖任何冻结实现）
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def cube_count(mask: int, vals: int, n_bits: int) -> int:
    if int(vals) & ~int(mask):
        raise ParallelChannelError("子立方非法：固定值越出固定掩码")
    return 1 << (int(n_bits) - popcount(mask))


def cube_states(mask: int, vals: int, n_bits: int, cap: Optional[int] = None) -> List[int]:
    n = cube_count(mask, vals, n_bits)
    if cap is not None and n > int(cap):
        raise ParallelChannelError("子立方状态数 %d 超上限 %s" % (n, cap))
    free = [b for b in range(int(n_bits)) if not (int(mask) >> b) & 1]
    out = [int(vals)]
    for b in free:
        out = out + [s | (1 << b) for s in out]
    return sorted(out)


def cube_subset(child_mask: int, child_vals: int, parent_mask: int, parent_vals: int) -> bool:
    if (int(child_mask) & int(parent_mask)) != int(parent_mask):
        return False
    return not ((int(child_vals) ^ int(parent_vals)) & int(parent_mask))


def cubes_disjoint(a_mask: int, a_vals: int, b_mask: int, b_vals: int) -> bool:
    common = int(a_mask) & int(b_mask)
    return bool((int(a_vals) ^ int(b_vals)) & common)


def bit_forced_one(mask: int, vals: int, bit: int) -> bool:
    return bool((int(mask) >> int(bit)) & 1) and bool((int(vals) >> int(bit)) & 1)


def bit_forced_zero(mask: int, vals: int, bit: int) -> bool:
    return bool((int(mask) >> int(bit)) & 1) and not bool((int(vals) >> int(bit)) & 1)


def bit_allows_one(mask: int, vals: int, bit: int) -> bool:
    return not bit_forced_zero(mask, vals, bit)


# ---------------------------------------------------------------------------
# 声明结构 / 读位解析（验证器自带一份；不 import certificate_solver 的解析器）
# ---------------------------------------------------------------------------

def _layer(name: str) -> int:
    s = str(name)
    if len(s) < 2 or s[0] != "v" or not s[1:].isdigit():
        raise ParallelChannelError("节点名不合声明形态: %r" % name)
    return int(s[1:])


def parse_structure(table) -> dict:
    """自带结构解析：EDGE_BITS 声明链（节点 v0..vk；边只连相邻层）。"""
    raw = table.tags.get("s2_structure") if hasattr(table, "tags") else None
    if not raw:
        raise ParallelChannelError("%s: 表未登记 s2_structure" % getattr(table, "table_id", "?"))
    if isinstance(raw, str):
        raw = json.loads(raw)
    kind = str(raw.get("kind"))
    if kind != "EDGE_BITS":
        raise ParallelChannelError("结构 kind=%r 不支持通道分解（只声明 EDGE_BITS）" % kind)
    nodes = [str(x) for x in raw.get("nodes", [])]
    edges = [(int(b), str(u), str(v)) for (b, u, v) in raw.get("edges", [])]
    a_nodes = [str(x) for x in raw.get("a_nodes", [])]
    b_nodes = [str(x) for x in raw.get("b_nodes", [])]
    a_inc = [int(x) for x in raw.get("a_incident_bits", [])]
    b_inc = [int(x) for x in raw.get("b_incident_bits", [])]
    if not nodes or not edges:
        raise ParallelChannelError("结构缺节点或边")
    layers = [_layer(x) for x in nodes]
    if sorted(layers) != list(range(len(layers))):
        raise ParallelChannelError("节点层不连续: %s" % layers)
    k = len(layers) - 1
    if len(a_nodes) != 1 or len(b_nodes) != 1:
        raise ParallelChannelError("A/B 端点必须各恰一个节点")
    la, lb = _layer(a_nodes[0]), _layer(b_nodes[0])
    if la != 0:
        raise ParallelChannelError("A 端点不在层 0: %d" % la)
    if not (1 <= lb <= k):
        raise ParallelChannelError("B 端点层越界: %d" % lb)
    for (b, u, v) in edges:
        du, dv = _layer(u), _layer(v)
        if abs(du - dv) != 1:
            raise ParallelChannelError("边 (bit=%d, %s, %s) 非相邻层" % (b, u, v))
        if b < 0 or b >= int(table.n_bits):
            raise ParallelChannelError("边位号越界: %d" % b)
    bits_all = [b for (b, _u, _v) in edges]
    if len(set(bits_all)) != len(bits_all):
        raise ParallelChannelError("同一位出现在多条边")
    stage_bits: Dict[int, List[int]] = {}
    stage_edges: Dict[int, List[Tuple[int, str, str]]] = {}
    for (b, u, v) in edges:
        s = min(_layer(u), _layer(v))
        stage_bits.setdefault(s, []).append(int(b))
        stage_edges.setdefault(s, []).append((int(b), u, v))
    if sorted(stage_bits) != list(range(k)):
        raise ParallelChannelError("阶段编号不连续: %s" % sorted(stage_bits))
    for s in stage_bits:
        stage_bits[s] = sorted(stage_bits[s])
    return {"kind": kind, "nodes": nodes, "edges": edges, "a_nodes": a_nodes, "b_nodes": b_nodes,
            "a_incident_bits": sorted(a_inc), "b_incident_bits": sorted(b_inc),
            "n_layers": len(layers), "k": k, "target_node": lb,
            "stage_bits": stage_bits, "stage_edges": stage_edges}


def chain_meta(struct: dict) -> dict:
    """链式元信息：路径阶段 = stage < target_node；路径外阶段对标签无影响。

    有牙性：声明链若与「合取式标签」不等价（a 端点不在第一阶段 / b 端点未覆盖第 t-1 阶段），
    直接判结构不支持，不硬凑。
    """
    k = int(struct["k"])
    t = int(struct["target_node"])
    a_bits = set(struct["a_incident_bits"])
    stage0 = set(struct["stage_bits"][0])
    if a_bits != stage0:
        raise ParallelChannelError("A 端点关联位 %s != 第一阶段位 %s（结构非声明链）"
                                   % (sorted(a_bits), sorted(stage0)))
    b_bits = set(struct["b_incident_bits"])
    stage_t1 = set(struct["stage_bits"][t - 1])
    if not stage_t1 <= b_bits:
        raise ParallelChannelError("B 端点关联位缺第 %d 阶段（结构非声明链）" % (t - 1))
    extra = b_bits - stage_t1
    if extra and extra != set(struct["stage_bits"].get(t, [])):
        raise ParallelChannelError("B 端点关联位含非相邻阶段: %s" % sorted(extra))
    path_stages = list(range(t))
    off_stages = list(range(t, k))
    return {"n_stages": k, "target_node": t, "path_stages": path_stages,
            "off_path_stages": off_stages,
            "stage_bits": {int(s): [int(b) for b in bs] for s, bs in struct["stage_bits"].items()},
            "n_edges": len(struct["edges"])}


def parse_reads(table, query_id: str) -> Tuple[List[Tuple[int, int]], str, List[str]]:
    """自带读位解析（B13_SCENARIO 形态）；返回 (reads, kind, reasons)。"""
    reasons: List[str] = []
    q = table.tags.get("s2_queries")
    if not q:
        return [], "", ["表未登记 s2_queries"]
    meta = q if isinstance(q, Mapping) else json.loads(q)
    kind = str(meta.get("kind"))
    if kind != "B13_SCENARIO":
        return [], kind, ["查询类型 %r 非 B13_SCENARIO：通道分解未声明该形态" % kind]
    prefix = "B13:scenario:"
    if not str(query_id).startswith(prefix):
        return [], kind, ["query_id 形态不符: %r" % query_id]
    name = str(query_id)[len(prefix):]
    sc = (meta.get("scenarios") or {}).get(name)
    if sc is None:
        return [], kind, ["未知情景 %r" % name]
    reads: List[Tuple[int, int]] = []
    for item in sc.get("reads", []):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            reasons.append("读项形态不符: %r" % (item,))
            continue
        reads.append((int(item[0]), int(item[1])))
    return reads, kind, reasons


def reads_are_unit(reads: Sequence[Tuple[int, int]]) -> Tuple[bool, List[str]]:
    """单位读 = 每个读恰固定一个位；重复位或非 0/1 取值都拒。"""
    reasons: List[str] = []
    seen = set()
    for (b, v) in reads:
        if b in seen:
            reasons.append("读数重复位 %d" % b)
        seen.add(int(b))
        if int(v) not in (0, 1):
            reasons.append("读值非二值: %r" % (v,))
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# 通道对象与分解
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    channel_id: str
    stage: int
    bits: Tuple[int, ...]
    nodes: Tuple[str, str]
    fixed_mask: int
    fixed_vals: int
    local_states: Tuple[int, ...]      # 局部位型（全局位号；只可能通道位被置 1）
    local_g: Tuple[int, ...]           # 与 local_states 平行的 g 值
    status: str
    designated_bit: Optional[int]
    witness_0: Optional[int]
    witness_1: Optional[int]

    @property
    def bits_mask(self) -> int:
        m = 0
        for b in self.bits:
            m |= 1 << int(b)
        return m

    def to_dict(self) -> dict:
        return {"channel_id": self.channel_id, "stage": int(self.stage),
                "bits": [int(b) for b in self.bits], "nodes": [str(self.nodes[0]), str(self.nodes[1])],
                "fixed_mask": int(self.fixed_mask), "fixed_vals": int(self.fixed_vals),
                "local_states": [int(s) for s in self.local_states],
                "local_g": [int(g) for g in self.local_g],
                "status": self.status, "designated_bit": self.designated_bit,
                "witness_0": self.witness_0, "witness_1": self.witness_1}


def _classify_channel(bits: Sequence[int], fm: int, fv: int) -> dict:
    free = [b for b in bits if not (int(fm) >> int(b)) & 1]
    states = [int(fv)]
    for b in free:
        states = states + [s | (1 << int(b)) for s in states]
    states = sorted(states)
    mask = 0
    for b in bits:
        mask |= 1 << int(b)
    g = [1 if (s & mask) else 0 for s in states]
    if all(x == 0 for x in g):
        return {"status": STATUS_ALL0, "states": states, "g": g,
                "designated_bit": None, "witness_0": states[0], "witness_1": None}
    if all(x == 1 for x in g):
        designated = min(b for b in bits if bit_forced_one(fm, fv, b))
        return {"status": STATUS_ALL1, "states": states, "g": g,
                "designated_bit": int(designated), "witness_0": None, "witness_1": states[0]}
    w0 = min(s for s, x in zip(states, g) if x == 0)
    w1 = min(s for s, x in zip(states, g) if x == 1)
    return {"status": STATUS_MIXED, "states": states, "g": g,
            "designated_bit": None, "witness_0": int(w0), "witness_1": int(w1)}


@dataclass
class ChannelSet:
    table_id: str
    query_id: str
    n_bits: int
    meta: dict
    reads: List[Tuple[int, int]]
    unit_reads: bool
    allowed_mask: int
    allowed_vals: int
    channels: List[Channel]
    off_path_stages: List[int]
    off_path_read_bits: List[int]
    ok: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def n_path_channels(self) -> int:
        return len(self.channels)

    def status_counts(self) -> dict:
        out = {STATUS_ALL1: 0, STATUS_ALL0: 0, STATUS_MIXED: 0}
        for ch in self.channels:
            out[ch.status] += 1
        return out

    def off_path_pattern(self) -> int:
        p = 0
        for (b, v) in self.reads:
            if int(b) not in {int(x) for ch in self.channels for x in ch.bits}:
                if int(v):
                    p |= 1 << int(b)
        return p

    def to_dict(self) -> dict:
        return {"table_id": self.table_id, "query_id": self.query_id, "n_bits": int(self.n_bits),
                "n_stages": int(self.meta["n_stages"]), "target_node": int(self.meta["target_node"]),
                "reads": [[int(b), int(v)] for (b, v) in self.reads],
                "unit_reads": bool(self.unit_reads),
                "allowed_cube": {"fixed_mask": int(self.allowed_mask),
                                 "fixed_vals": int(self.allowed_vals)},
                "channels": [ch.to_dict() for ch in self.channels],
                "off_path_stages": [int(s) for s in self.off_path_stages],
                "off_path_read_bits": [int(b) for b in self.off_path_read_bits],
                "ok": bool(self.ok), "reasons": list(self.reasons)}


def decompose_query(table, query: str) -> ChannelSet:
    """完整分解对象（含允许集、通道层、路径外登记）。"""
    struct = parse_structure(table)
    meta = chain_meta(struct)
    n_bits = int(table.n_bits)
    reads, kind, reasons = parse_reads(table, query)
    unit, why = reads_are_unit(reads)
    reasons = list(reasons) + list(why)
    mask = 0
    vals = 0
    for (b, v) in reads:
        mask |= 1 << int(b)
        vals |= 1 << int(b) if int(v) else 0
    stage_of_bit: Dict[int, int] = {}
    for s, bs in meta["stage_bits"].items():
        for b in bs:
            stage_of_bit[int(b)] = int(s)
    channels: List[Channel] = []
    off_bits: List[int] = []
    for s in meta["path_stages"]:
        bits = tuple(meta["stage_bits"][s])
        fm = mask & 0
        fm = 0
        fv = 0
        for b in bits:
            if (mask >> b) & 1:
                fm |= 1 << b
                if (vals >> b) & 1:
                    fv |= 1 << b
        cls = _classify_channel(bits, fm, fv)
        edges = struct["stage_edges"][s]
        u, v = str(edges[0][1]), str(edges[0][2])
        channels.append(Channel(channel_id="CH:stage:%d" % s, stage=int(s), bits=bits,
                                nodes=(u, v), fixed_mask=int(fm), fixed_vals=int(fv),
                                local_states=tuple(cls["states"]), local_g=tuple(cls["g"]),
                                status=cls["status"], designated_bit=cls["designated_bit"],
                                witness_0=cls["witness_0"], witness_1=cls["witness_1"]))
    for (b, _v) in reads:
        if int(b) not in stage_of_bit:
            reasons.append("读位 %d 不在任何声明阶段上" % int(b))
        elif stage_of_bit[int(b)] >= meta["target_node"]:
            off_bits.append(int(b))
    ok = bool(unit and not reasons)
    if not unit:
        reasons.append("DECOMPOSITION_NOT_APPLICABLE: 非单位读（组读）不被通道分解承认")
    return ChannelSet(table_id=str(getattr(table, "table_id", "?")), query_id=str(query),
                      n_bits=n_bits, meta=meta, reads=reads, unit_reads=bool(unit),
                      allowed_mask=int(mask), allowed_vals=int(vals), channels=channels,
                      off_path_stages=list(meta["off_path_stages"]),
                      off_path_read_bits=sorted(set(off_bits)), ok=ok, reasons=reasons)


def decompose_channels(table, query) -> List[Channel]:
    """任务单 §2.1 接口：查询 → 并行通道列表（路径阶段；每通道独立允许集）。"""
    return decompose_query(table, query).channels


# ---------------------------------------------------------------------------
# 逐通道证书
# ---------------------------------------------------------------------------

@dataclass
class ChannelCertificate:
    channel_id: str
    table_id: str
    query_id: str
    stage: int
    bits: Tuple[int, ...]
    nodes: Tuple[str, str]
    fixed_mask: int
    fixed_vals: int
    local_states: Tuple[int, ...]
    local_g: Tuple[int, ...]
    status: str
    designated_bit: Optional[int]
    witness_0: Optional[int]
    witness_1: Optional[int]
    costs: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA_CHANNEL_CERT, "channel_id": self.channel_id,
                "table_id": self.table_id, "query_id": self.query_id, "stage": int(self.stage),
                "bits": [int(b) for b in self.bits],
                "nodes": [str(self.nodes[0]), str(self.nodes[1])],
                "local": {"fixed_mask": int(self.fixed_mask), "fixed_vals": int(self.fixed_vals),
                          "states": [int(s) for s in self.local_states],
                          "g": [int(g) for g in self.local_g]},
                "status": self.status, "designated_bit": self.designated_bit,
                "witness_0": self.witness_0, "witness_1": self.witness_1,
                "costs": dict(self.costs), "budget": dict(self.budget),
                "watermark": dict(WATERMARK)}


@dataclass
class Bound:
    scope: str
    entity_id: str
    reason: str
    detail: dict
    frontier: List[dict]
    stats: dict

    def to_dict(self) -> dict:
        return {"schema": "s2s2.bound/v1", "scope": self.scope, "entity_id": self.entity_id,
                "reason": self.reason, "detail": dict(self.detail),
                "frontier": [dict(x) for x in self.frontier], "stats": dict(self.stats)}


@dataclass
class Unresolved:
    scope: str
    entity_id: str
    reason: str
    detail: dict
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"schema": "s2s2.unresolved/v1", "scope": self.scope, "entity_id": self.entity_id,
                "reason": self.reason, "detail": dict(self.detail), "stats": dict(self.stats)}


def _local_recheck(ch: Channel) -> dict:
    cls = _classify_channel(ch.bits, ch.fixed_mask, ch.fixed_vals)
    return cls


def certify_channel(channel, table, budget: Optional[Mapping] = None,
                    query_id: Optional[str] = None
                    ) -> "ChannelCertificate | Bound | Unresolved":
    """逐通道证书：局部允许集（≤ 声明上限）→ 全枚举级局部判定 + 见证。"""
    t0 = time.perf_counter()
    bud = {"local_states_cap": 16, "regions_cap": 16}
    bud.update(dict(budget or {}))
    if not isinstance(channel, Channel):
        raise ParallelChannelError("certify_channel 只接受 Channel 对象")
    n_local = 1 << len(channel.bits)
    if n_local > int(bud["local_states_cap"]):
        return Bound(scope="channel", entity_id=channel.channel_id,
                     reason="CHANNEL_LOCAL_STATES_OVER_BUDGET",
                     detail={"local_states_cap": int(bud["local_states_cap"]),
                             "n_local": int(n_local)},
                     frontier=[channel.to_dict()], stats={"basis": "MEASURED_ONLY"})
    cls = _local_recheck(channel)
    if cls["status"] != channel.status or cls["witness_0"] != channel.witness_0 \
            or cls["witness_1"] != channel.witness_1 or cls["designated_bit"] != channel.designated_bit:
        return Unresolved(scope="channel", entity_id=channel.channel_id,
                          reason="CHANNEL_SELF_INCONSISTENT",
                          detail={"claimed": channel.status, "recomputed": cls["status"]},
                          stats={})
    cert = ChannelCertificate(
        channel_id=channel.channel_id, table_id=str(table.table_id),
        query_id=str(query_id or ""),
        stage=channel.stage, bits=channel.bits, nodes=channel.nodes,
        fixed_mask=channel.fixed_mask, fixed_vals=channel.fixed_vals,
        local_states=channel.local_states, local_g=channel.local_g, status=channel.status,
        designated_bit=channel.designated_bit, witness_0=channel.witness_0,
        witness_1=channel.witness_1,
        costs={"basis": "MEASURED_ONLY", "local_states": len(channel.local_states),
               "envelope_calls": 1, "regions": 1, "seconds": round(time.perf_counter() - t0, 9)},
        budget={"local_states_cap": int(bud["local_states_cap"]),
                "regions_cap": int(bud["regions_cap"])})
    return cert


# ---------------------------------------------------------------------------
# 组合：单标签证书（冻结 s2.certificate/v1）/ 混合证明（s2s2.combination_proof/v1）
# ---------------------------------------------------------------------------

@dataclass
class ChannelResults:
    decomposition: ChannelSet
    certificates: List[ChannelCertificate]
    table: object

    def by_stage(self) -> Dict[int, ChannelCertificate]:
        return {int(c.stage): c for c in self.certificates}


@dataclass
class MixedProof:
    proof_id: str
    table_id: str
    query_id: str
    n_bits: int
    decomposition: dict
    regions: List[dict]
    n_leaves: int
    leaf_claims: Dict[str, int]
    witness_1: dict
    witness_0: dict
    frozen_crosscheck: dict
    budget: dict
    costs: dict
    stats: dict

    def to_dict(self) -> dict:
        return {"schema": SCHEMA_COMBINATION_PROOF, "proof_id": self.proof_id,
                "table_id": self.table_id, "query_id": self.query_id, "n_bits": int(self.n_bits),
                "decomposition": self.decomposition,
                "partition": {"root_id": "R0", "regions": self.regions,
                              "n_regions": len(self.regions), "n_leaves": int(self.n_leaves),
                              "leaf_claims": dict(self.leaf_claims)},
                "witnesses": {"label_1": dict(self.witness_1), "label_0": dict(self.witness_0)},
                "combination_rule": "AND_OVER_CHANNELS: ANY_ALL0=>{0} / ALL_ALL1=>{1} / ELSE={0,1}",
                "frozen_crosscheck": dict(self.frozen_crosscheck),
                "budget": dict(self.budget), "costs": dict(self.costs), "stats": dict(self.stats),
                "verifier_version": VERIFIER_VERSION,
                "selection_provenance": SELECTION_PROVENANCE,
                "watermark": dict(WATERMARK)}


def predicted_label_set(statuses: Sequence[str]) -> List[str]:
    if any(s == STATUS_ALL0 for s in statuses):
        return ["0"]
    if all(s == STATUS_ALL1 for s in statuses):
        return ["1"]
    return ["0", "1"]


def _own_probe_label(struct_meta: dict, struct: dict, state: int) -> int:
    """自带「链式合取」标签：路径阶段全非空 ⇒ 1。只在验证器与见证重放里用。"""
    for s in struct_meta["path_stages"]:
        m = 0
        for b in struct_meta["stage_bits"][s]:
            m |= 1 << int(b)
        if not (int(state) & m):
            return 0
    return 1


def _build_regions_for_mixed(cset: ChannelSet, budget: dict) -> List[cs.Region]:
    """混合判据的全覆盖划分树（三叉单立方分裂；节点/立方数超预算 ⇒ 由调用方记 Bound）。"""
    mixed = [ch for ch in cset.channels if ch.status == STATUS_MIXED]
    regions: List[cs.Region] = []
    root = cs.Region("R0", None, 0, (cs.Cube(cset.allowed_mask, cset.allowed_vals),))
    regions.append(root)
    frontier: List[Tuple[str, int, int, int]] = [("R0", 0, cset.allowed_mask, cset.allowed_vals)]
    seq = 1
    for ch in mixed:
        new_frontier: List[Tuple[str, int, int, int]] = []
        for (pid, depth, cmask, cvals) in frontier:
            kids = _split_cube_for_channel(ch.bits, cmask, cvals)
            for (kmask, kvals, is_z) in kids:
                if len(regions) + 1 > int(budget["nodes_cap"]):
                    raise ParallelChannelError("BOUND:NODES_CAP:%d" % int(budget["nodes_cap"]))
                r = cs.Region("R%d" % seq, pid, depth + 1, (cs.Cube(kmask, kvals),))
                seq += 1
                regions.append(r)
                if not is_z:
                    new_frontier.append((r.region_id, depth + 1, kmask, kvals))
        frontier = new_frontier
    return regions


def _split_cube_for_channel(bits: Sequence[int], cmask: int, cvals: int
                            ) -> List[Tuple[int, int, bool]]:
    """通道分裂：A（低位=1）/ B（低位=0 高位=1）/ Z（两位=0），或固定 0 时的 B/Z。"""
    bits = [int(b) for b in bits]
    if len(bits) == 1:
        b = bits[0]
        if bit_forced_zero(cmask, cvals, b):
            raise ParallelChannelError("Z 通道不进分裂树")
        return [(cmask | (1 << b), cvals | (1 << b), False),
                (cmask | (1 << b), cvals, True)]
    b0, b1 = bits
    f0z = bit_forced_zero(cmask, cvals, b0)
    f1z = bit_forced_zero(cmask, cvals, b1)
    if f0z and f1z:
        raise ParallelChannelError("Z 通道不进分裂树")
    if f0z:
        return [(cmask | (1 << b1), cvals | (1 << b1), False),
                (cmask | (1 << b1), cvals, True)]
    if f1z:
        return [(cmask | (1 << b0), cvals | (1 << b0), False),
                (cmask | (1 << b0), cvals, True)]
    return [(cmask | (1 << b0), cvals | (1 << b0), False),
            (cmask | (1 << b0) | (1 << b1), cvals | (1 << b1), False),
            (cmask | (1 << b0) | (1 << b1), cvals, True)]


def _replay_state(table, cset: ChannelSet, state: int) -> dict:
    """见证重放：自带结构标签 + 三要素 BFS 两路互校（不用 ident.state_connected 做判据）。"""
    struct = parse_structure(table)
    d = table.derivation(cset.query_id, int(state))
    adj: Dict = {}
    for (u, v) in d.edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen = set(d.a_side & d.nodes)
    queue = list(seen)
    reached = False
    head = 0
    goals = set(d.b_side & d.nodes)
    while head < len(queue) and not reached:
        x = queue[head]
        head += 1
        if x in goals:
            reached = True
            break
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                queue.append(y)
    bfs_label = int(bool(d.attach_a) and bool(d.attach_b) and reached)
    chain_label = _own_probe_label(cset.meta, struct, int(state))
    frozen = ident.state_connected(d)
    faithful = int(d.label) == bfs_label
    return {"state_id": int(state), "bfs_label": int(bfs_label), "chain_label": int(chain_label),
            "state_connected_route": int(bool(frozen)), "label_field": int(d.label),
            "faithful_label_field": bool(faithful), "ok": bool(bfs_label == chain_label == int(frozen))}


def _witness_states(cset: ChannelSet) -> Tuple[int, int]:
    """由通道见证装配标签 1 / 标签 0 的全局见证状态。"""
    off = cset.off_path_pattern()
    w1 = off
    for ch in cset.channels:
        if ch.status == STATUS_ALL1:
            w1 |= ch.local_states[0]
        elif ch.status == STATUS_MIXED:
            w1 |= int(ch.witness_1)
        else:
            raise ParallelChannelError("ALL0 通道不应参与混合见证装配")
    anchor = next(ch for ch in cset.channels if ch.status == STATUS_MIXED)
    w0 = off
    for ch in cset.channels:
        if ch is anchor:
            w0 |= int(ch.witness_0)
        elif ch.status == STATUS_ALL1:
            w0 |= ch.local_states[0]
        else:
            w0 |= int(ch.witness_1)
    return int(w1), int(w0)


def combine_channels(channel_results, query: str, budget: Optional[Mapping] = None
                     ) -> "cs.Certificate | MixedProof | Bound | Unresolved":
    """组合判据。channel_results 须为 ChannelResults（含分解与逐通道证书）。

    返回：单标签 ⇒ 冻结 ``s2.certificate/v1`` 的 ``cs.Certificate``；
    混合 ⇒ ``MixedProof``；任一通道未决 ⇒ ``Unresolved``；超预算 ⇒ ``Bound``。
    """
    t0 = time.perf_counter()
    if isinstance(channel_results, ChannelResults):
        cset, certs, table = (channel_results.decomposition, channel_results.certificates,
                              channel_results.table)
    else:
        raise ParallelChannelError("combine_channels 需要 ChannelResults（分解 + 通道证书）")
    if str(query) != cset.query_id:
        raise ParallelChannelError("query 与分解不符: %r vs %r" % (query, cset.query_id))
    bud = {"nodes_cap": 4096, "cubes_cap": 4096}
    bud.update(dict(budget or {}))
    if not cset.ok:
        return Unresolved(scope="combination", entity_id=cset.query_id,
                          reason="DECOMPOSITION_NOT_APPLICABLE",
                          detail={"reasons": list(cset.reasons)}, stats={})
    if len(certs) != len(cset.channels):
        return Unresolved(scope="combination", entity_id=cset.query_id,
                          reason="CHANNEL_LAYER_INCOMPLETE",
                          detail={"n_certs": len(certs), "n_channels": len(cset.channels)},
                          stats={})
    for c in certs:
        if not isinstance(c, ChannelCertificate):
            return Unresolved(scope="combination", entity_id=cset.query_id,
                              reason="CHANNEL_NOT_DECIDED",
                              detail={"channel": getattr(c, "entity_id", None),
                                      "sub_reason": getattr(c, "reason", None)}, stats={})
    statuses = [c.status for c in certs]
    want = predicted_label_set(statuses)
    if want in (["0"], ["1"]):
        root = cs.Region("R0", None, 0, (cs.Cube(cset.allowed_mask, cset.allowed_vals),))
        res = cs.certify(table, cset.query_id, [root], budget=bud,
                         stats={"decompose_seconds": 0.0, "channels_seconds": 0.0,
                                "combination_rule": "ALL_ALL1" if want == ["1"] else "ANY_ALL0"},
                         certificate_id="S2S2C:%s:%s" % (cset.table_id, cset.query_id))
        if isinstance(res, cs.Certificate) and res.label_claim == want[0]:
            res.watermark = dict(WATERMARK)
            res.stats["combination_seconds"] = round(time.perf_counter() - t0, 6)
            return res
        good = res.label_claim if isinstance(res, cs.Certificate) else getattr(res, "reason", "?")
        raise ParallelChannelError("规则与冻结签发分歧: 通道判定 %s vs 冻结 %s"
                                   % (want, good))
    # 混合
    try:
        regions = _build_regions_for_mixed(cset, bud)
    except ParallelChannelError as e:
        if str(e).startswith("BOUND:"):
            return Bound(scope="combination", entity_id=cset.query_id,
                         reason="COMBO_NODES_OVER_BUDGET",
                         detail={"nodes_cap": int(bud["nodes_cap"])},
                         frontier=[{"note": "未落完成的通道分裂树；不删不猜"}],
                         stats={"basis": "MEASURED_ONLY"})
        raise
    if len(regions) > int(bud["cubes_cap"]):
        return Bound(scope="combination", entity_id=cset.query_id,
                     reason="COMBO_CUBES_OVER_BUDGET",
                     detail={"cubes_cap": int(bud["cubes_cap"]), "n_regions": len(regions)},
                     frontier=[{"note": "区域数即立方数（全单立方树）"}],
                     stats={"basis": "MEASURED_ONLY"})
    res = cs.certify(table, cset.query_id, regions, budget=bud,
                     stats={"combination_rule": "MIXED_WITNESS_PAIR"},
                     certificate_id="S2S2C:%s:%s" % (cset.table_id, cset.query_id))
    if not isinstance(res, cs.Unresolved) or res.reason != "MIXED_LABELS":
        good = res.label_claim if isinstance(res, cs.Certificate) else getattr(res, "reason", "?")
        raise ParallelChannelError("混合判定与冻结签发分歧: %s" % good)
    claims = sorted({r.label_claim for r in res.partition})
    if claims != ["0", "1"]:
        raise ParallelChannelError("混合分区标签集异常: %s" % claims)
    w1, w0 = _witness_states(cset)
    rep1 = _replay_state(table, cset, w1)
    rep0 = _replay_state(table, cset, w0)
    if not (rep1["ok"] and rep1["chain_label"] == 1):
        raise ParallelChannelError("标签 1 见证重放失败: %s" % rep1)
    if not (rep0["ok"] and rep0["chain_label"] == 0):
        raise ParallelChannelError("标签 0 见证重放失败: %s" % rep0)
    for (b, v) in cset.reads:
        if ((w1 >> int(b)) & 1) != int(v) or ((w0 >> int(b)) & 1) != int(v):
            raise ParallelChannelError("见证不满足读数 (%d,%d)" % (int(b), int(v)))
    leaves = [r for r in regions if r.is_terminal]
    leaf_claims = {"0": sum(1 for r in leaves if r.label_claim == "0"),
                   "1": sum(1 for r in leaves if r.label_claim == "1")}
    region_docs = [r.to_dict(cset.n_bits) for r in regions]
    for rd in region_docs:
        rd.pop("state_ids", None)
        rd.pop("state_ids_head", None)
    proof = MixedProof(
        proof_id="S2S2P:%s:%s" % (cset.table_id, cset.query_id),
        table_id=cset.table_id, query_id=cset.query_id, n_bits=cset.n_bits,
        decomposition={"unit_reads": bool(cset.unit_reads),
                       "reads": [[int(b), int(v)] for (b, v) in cset.reads],
                       "allowed_cube": {"fixed_mask": int(cset.allowed_mask),
                                        "fixed_vals": int(cset.allowed_vals)},
                       "allowed_states": cube_count(cset.allowed_mask, cset.allowed_vals,
                                                    cset.n_bits),
                       "channels": [c.to_dict() for c in certs],
                       "off_path_stages": [int(s) for s in cset.off_path_stages],
                       "off_path_read_bits": [int(b) for b in cset.off_path_read_bits],
                       "status_counts": cset.status_counts()},
        regions=region_docs, n_leaves=len(leaves), leaf_claims=leaf_claims,
        witness_1={"state_id": int(w1), "label": 1, "replay": rep1},
        witness_0={"state_id": int(w0), "label": 0, "replay": rep0},
        frozen_crosscheck={"route": "certificate_solver.certify(s2s2 tree)", "result": "MIXED_LABELS",
                           "claims": claims, "partition_terms": len(res.partition),
                           "s2s_reconcile_caliber": ("missing/extra/wrong/duplicate 在验证侧按声明"
                                                     "上限材质化核对（超限记 NOT_MEASURED）")},
        budget={"nodes_cap": int(bud["nodes_cap"]), "cubes_cap": int(bud["cubes_cap"]),
                "nodes_used": len(regions), "cubes_used": len(regions),
                "per_channel": {"local_states_cap": 16, "regions_cap": 16}},
        costs={"basis": "MEASURED_ONLY", "regions": len(regions),
               "combination_seconds": round(time.perf_counter() - t0, 6)},
        stats={"combination_rule": "MIXED_WITNESS_PAIR", "claims": claims})
    return proof


# ---------------------------------------------------------------------------
# 独立验证器（不调用 solve_partial / certify / verify_certificate；不读 labels）
# ---------------------------------------------------------------------------

def _own_direction(mask: int, vals: int, struct: dict, meta: dict
                   ) -> Tuple[Optional[str], dict]:
    """自带 must/may 方向规则：must 有路 ⇒ 1；may 无路 ⇒ 0；否则 None。"""
    must_edges = set()
    may_edges = set()
    stage_edge_node: Dict[int, Tuple[str, str]] = {}
    for s, es in struct["stage_edges"].items():
        for (b, u, v) in es:
            stage_edge_node[int(b)] = (u, v)
    for (b, u, v) in struct["edges"]:
        if bit_forced_one(mask, vals, b):
            must_edges.add((min(u, v), max(u, v)))
        if bit_allows_one(mask, vals, b):
            may_edges.add((min(u, v), max(u, v)))
    a_node = struct["a_nodes"][0]
    b_node = struct["b_nodes"][0]
    a_inc = struct["a_incident_bits"]
    b_inc = struct["b_incident_bits"]
    must_attach_a = any(bit_forced_one(mask, vals, b) for b in a_inc)
    may_attach_a = any(bit_allows_one(mask, vals, b) for b in a_inc)
    must_attach_b = any(bit_forced_one(mask, vals, b) for b in b_inc)
    may_attach_b = any(bit_allows_one(mask, vals, b) for b in b_inc)

    def connected(edges) -> bool:
        adj: Dict = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        seen = {a_node}
        queue = [a_node]
        head = 0
        while head < len(queue):
            x = queue[head]
            head += 1
            if x == b_node:
                return True
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        return False

    must_conn = bool(must_attach_a and must_attach_b and connected(must_edges))
    may_conn = bool(may_attach_a and may_attach_b and connected(may_edges))
    detail = {"must_connected": must_conn, "may_connected": may_conn,
              "must_attach_a": must_attach_a, "must_attach_b": must_attach_b,
              "may_attach_a": may_attach_a, "may_attach_b": may_attach_b}
    if must_conn and not may_conn:
        return "INSTRUMENT_DEFECT_MONOTONICITY", detail
    if must_conn:
        return "1", detail
    if not may_conn:
        return "0", detail
    return None, detail


def _sample_states(mask: int, vals: int, n_bits: int, cap: int) -> List[int]:
    n = cube_count(mask, vals, n_bits)
    if n <= int(cap):
        return cube_states(mask, vals, n_bits)
    out = []
    for i in range(int(cap)):
        # 均匀步进（含首尾），去重后按升序返回
        out.append(_ith_free_state(mask, vals, n_bits, (i * (n - 1)) // (int(cap) - 1)))
    return sorted(set(out))


def _ith_free_state(mask: int, vals: int, n_bits: int, i: int) -> int:
    free = [b for b in range(int(n_bits)) if not (int(mask) >> b) & 1]
    s = int(vals)
    for k, b in enumerate(free):
        if (int(i) >> k) & 1:
            s |= 1 << b
    return s


def _verify_structure_and_reads(table, doc_reads: Optional[Sequence], query_id: str
                                ) -> Tuple[Optional[dict], Optional[dict], List[str]]:
    reasons: List[str] = []
    try:
        struct = parse_structure(table)
        meta = chain_meta(struct)
    except ParallelChannelError as e:
        return None, None, ["结构解析失败: %s" % e]
    reads, kind, why = parse_reads(table, query_id)
    reasons.extend(why)
    unit, why2 = reads_are_unit(reads)
    reasons.extend(why2)
    if not unit:
        reasons.append("非单位读：通道分解不被承认")
    if doc_reads is not None:
        got = [[int(b), int(v)] for (b, v) in reads]
        if [list(x) for x in doc_reads] != got:
            reasons.append("证书声明读组与表声明不等")
    return struct, meta, reasons


def verify_channel_certificate(doc: Mapping, table, query_id: str) -> dict:
    """通道证书独立验证：局部子立方重算、状态枚举重算、状态/见证自洽。"""
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if str(doc.get("schema")) != SCHEMA_CHANNEL_CERT:
        reasons.append("通道证书 schema 不符: %r" % doc.get("schema"))
    unknown = [k for k in doc if k not in CHANNEL_CERT_KEYS]
    if unknown:
        reasons.append("通道证书含未知字段 %s" % unknown)
    bad = [k for k in doc if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad:
        reasons.append("通道证书含真值/生产布尔字段 %s" % bad)
    if str(doc.get("table_id")) != str(table.table_id):
        reasons.append("通道证书 table_id 不符")
    if str(doc.get("query_id")) != str(query_id):
        reasons.append("通道证书 query_id 不符")
    bits = [int(b) for b in doc.get("bits", [])]
    loc = doc.get("local") or {}
    if "fixed_mask" not in loc or "fixed_vals" not in loc:
        reasons.append("通道证书缺局部固定位")
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    fm, fv = int(loc["fixed_mask"]), int(loc["fixed_vals"])
    if fv & ~fm:
        reasons.append("局部固定位非法")
    for b in bits:
        if (fm >> b) & 1 == 0 and fv & (1 << b):
            reasons.append("未固定位带值: %d" % b)
    cls = _classify_channel(bits, fm, fv)
    checks["local_states"] = len(cls["states"])
    if str(doc.get("status")) != cls["status"]:
        reasons.append("通道状态与重算不符: %r vs %r" % (doc.get("status"), cls["status"]))
    want_states = [int(s) for s in cls["states"]]
    if [int(s) for s in loc.get("states", [])] != want_states:
        reasons.append("局部状态枚举不符")
    if [int(g) for g in loc.get("g", [])] != [int(g) for g in cls["g"]]:
        reasons.append("局部 g 值不符")
    if cls["status"] == STATUS_ALL1:
        d = doc.get("designated_bit")
        if d is None or not bit_forced_one(fm, fv, int(d)) or int(d) not in bits:
            reasons.append("ALL1 通道缺合法 designated_bit")
    if cls["status"] == STATUS_ALL0:
        if any(bit_allows_one(fm, fv, b) for b in bits):
            reasons.append("ALL0 通道仍有可置 1 的位（仪器分歧）")
    if cls["status"] == STATUS_MIXED:
        w0, w1 = doc.get("witness_0"), doc.get("witness_1")
        if w0 is None or w1 is None or int(w0) not in want_states or int(w1) not in want_states:
            reasons.append("MIXED 通道见证不在局部允许集")
        else:
            if int(w0) & cls_bits_mask(bits):
                reasons.append("witness_0 未取 g=0")
            if not (int(w1) & cls_bits_mask(bits)):
                reasons.append("witness_1 未取 g=1")
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION, "checks": checks,
            "channel_id": doc.get("channel_id")}


def cls_bits_mask(bits: Sequence[int]) -> int:
    m = 0
    for b in bits:
        m |= 1 << int(b)
    return m


def verify_channel_bundle(bundle: Mapping, table, query_id: str) -> dict:
    """通道层独立验证：与本地重算的分解逐项对账（含「并集 = 原允许集」的符号检查）。"""
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if str(bundle.get("schema")) != SCHEMA_CHANNEL_BUNDLE:
        reasons.append("通道束 schema 不符: %r" % bundle.get("schema"))
    try:
        fresh = decompose_query(table, query_id)
    except ParallelChannelError as e:
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": ["重算分解失败: %s" % e],
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    if not fresh.ok:
        reasons.append("本地重算分解未通过: %s" % fresh.reasons)
    d = fresh.to_dict()
    doc_dec = bundle.get("decomposition") or {}
    if doc_dec.get("allowed_cube") != d["allowed_cube"]:
        reasons.append("允许集子立方与重算不符")
    if [list(x) for x in doc_dec.get("reads", [])] != [list(x) for x in d["reads"]]:
        reasons.append("读组与重算不符")
    # 「并集 = 原允许集」：固定掩码逐位对账（每个读位恰属一个通道/路径外）
    union_mask = 0
    union_vals = 0
    for ch in d["channels"]:
        for b in ch["bits"]:
            if (int(d["allowed_cube"]["fixed_mask"]) >> int(b)) & 1:
                union_mask |= 1 << int(b)
                if (int(d["allowed_cube"]["fixed_vals"]) >> int(b)) & 1:
                    union_vals |= 1 << int(b)
    for b in d["off_path_read_bits"]:
        union_mask |= 1 << int(b)
        if (int(d["allowed_cube"]["fixed_vals"]) >> int(b)) & 1:
            union_vals |= 1 << int(b)
    checks["union_mask_equals_allowed"] = (union_mask == int(d["allowed_cube"]["fixed_mask"])
                                           and union_vals == int(d["allowed_cube"]["fixed_vals"]))
    if not checks["union_mask_equals_allowed"]:
        reasons.append("通道并集 != 原允许集（固定位对账失败）")
    doc_chs = doc_dec.get("channels") or []
    if len(doc_chs) != len(d["channels"]):
        reasons.append("通道数与重算不符: %d vs %d" % (len(doc_chs), len(d["channels"])))
    else:
        def _fm_fv(b: Mapping) -> Tuple[int, int]:
            loc = b.get("local") or {}
            if "fixed_mask" in loc:
                return int(loc["fixed_mask"]), int(loc["fixed_vals"])
            return int(b.get("fixed_mask", -1)), int(b.get("fixed_vals", -1))

        def _states_of(b: Mapping):
            loc = b.get("local") or {}
            if "states" in loc:
                return [int(x) for x in loc["states"]]
            return [int(x) for x in b.get("local_states", [])]

        for a, b in zip(d["channels"], doc_chs):
            bfm, bfv = _fm_fv(b)
            if (a["channel_id"] != b.get("channel_id")
                    or a["bits"] != [int(x) for x in b.get("bits", [])]
                    or a["status"] != b.get("status")
                    or a["fixed_mask"] != bfm or a["fixed_vals"] != bfv
                    or [int(x) for x in a["local_states"]] != _states_of(b)):
                reasons.append("通道 %s 与重算不符" % a["channel_id"])
    if [int(s) for s in doc_dec.get("off_path_stages", [])] != [int(s) for s in d["off_path_stages"]]:
        reasons.append("路径外阶段登记与重算不符")
    # 抽样：允许集任一状态在各通道上的投影必须落在通道局部允许集内（并集=允许集的样本侧）
    try:
        cap = 4096
        sample = _sample_states(int(d["allowed_cube"]["fixed_mask"]),
                                int(d["allowed_cube"]["fixed_vals"]), int(table.n_bits), cap)
        ch_by_bits = [(cls_bits_mask(ch["bits"]), set(int(x) for x in ch["local_states"]))
                      for ch in d["channels"]]
        bad = 0
        for s in sample:
            for (bm, ok_states) in ch_by_bits:
                if (int(s) & bm) not in ok_states:
                    bad += 1
                    break
        checks["sample_states"] = len(sample)
        checks["sample_projection_violations"] = bad
        if bad:
            reasons.append("样本状态通道投影越出通道允许集 (%d)" % bad)
    except ParallelChannelError as e:
        reasons.append("样本核对失败: %s" % e)
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION, "checks": checks}


def _verify_witness_own(struct: dict, meta: dict, table, query_id: str, state: int,
                        claimed: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    d = table.derivation(query_id, int(state))
    adj: Dict = {}
    for (u, v) in d.edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    goals = set(d.b_side & d.nodes)
    seen = set(d.a_side & d.nodes)
    queue = list(seen)
    head = 0
    reached = False
    while head < len(queue) and not reached:
        x = queue[head]
        head += 1
        if x in goals:
            reached = True
        else:
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
    bfs = int(bool(d.attach_a) and bool(d.attach_b) and reached)
    chain = _own_probe_label(meta, struct, int(state))
    if bfs != chain:
        reasons.append("见证 %d：结构路与链式路不等 (%d vs %d)" % (state, bfs, chain))
    if chain != int(claimed):
        reasons.append("见证 %d：重算标签 %d != 声明 %d" % (state, chain, int(claimed)))
    if int(d.label) != chain:
        reasons.append("见证 %d：三要素派生标签不忠实 (%d vs %d)" % (state, int(d.label), chain))
    return (not reasons), reasons


def verify_combination_document(doc: Mapping, table, query_id: str) -> dict:
    """组合层独立验证：单标签（s2.certificate/v1 同构根证书）与混合证明两形态。"""
    reasons: List[str] = []
    checks: Dict[str, object] = {}
    if str(doc.get("schema")) == cs.SCHEMA_CERT:
        return _verify_single_label_document(doc, table, query_id, reasons, checks)
    if str(doc.get("schema")) != SCHEMA_COMBINATION_PROOF:
        return {"schema": SCHEMA_VERIFY, "pass": False,
                "reasons": ["未知组合件 schema: %r" % doc.get("schema")],
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    return _verify_mixed_proof(doc, table, query_id, reasons, checks)


def _allowed_from_doc_or_table(table, query_id: str, reasons: List[str]) -> Tuple[Optional[dict], Optional[dict], int, int]:
    struct, meta, why = _verify_structure_and_reads(table, None, query_id)
    reasons.extend(why)
    if struct is None:
        return None, None, 0, 0
    reads, _kind, _why = parse_reads(table, query_id)
    mask = 0
    vals = 0
    for (b, v) in reads:
        mask |= 1 << int(b)
        if int(v):
            vals |= 1 << int(b)
    return struct, meta, mask, vals


def _verify_single_label_document(doc: Mapping, table, query_id: str,
                                  reasons: List[str], checks: Dict[str, object]) -> dict:
    if str(doc.get("table_id")) != str(table.table_id):
        reasons.append("证书 table_id 不符")
    if str(doc.get("query_id")) != str(query_id):
        reasons.append("证书 query_id 不符")
    bad = [k for k in doc if k not in cs.CERT_KEYS]
    if bad:
        reasons.append("证书含未知顶层字段 %s" % bad)
    forbidden = [k for k in doc if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if forbidden:
        reasons.append("证书含真值/生产布尔字段 %s" % forbidden)
    struct, meta, mask, vals = _allowed_from_doc_or_table(table, query_id, reasons)
    if struct is None:
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    n_bits = int(table.n_bits)
    declared_allowed = [int(s) for s in doc.get("allowed_state_ids", [])]
    n_allowed = cube_count(mask, vals, n_bits)
    if len(declared_allowed) != n_allowed:
        reasons.append("允许集计数不符: %d vs %d" % (len(declared_allowed), n_allowed))
    elif declared_allowed != cube_states(mask, vals, n_bits):
        reasons.append("允许集与非声明的读组重算不符")
    claims = doc.get("label_claim")
    own, det = _own_direction(mask, vals, struct, meta)
    checks["own_direction"] = own
    checks["own_direction_detail"] = det
    if own == "INSTRUMENT_DEFECT_MONOTONICITY":
        reasons.append("外包单调性破裂（仪器缺陷）")
    if own != claims:
        reasons.append("区域外包方向 %r != 声明 %r" % (own, claims))
    regions = list(doc.get("regions") or [])
    if len(regions) != 1 or regions[0].get("parent_id") is not None:
        reasons.append("单标签组合证书必须恰一根区域")
    r0 = regions[0] if regions else {}
    cubes = list(r0.get("cubes") or [])
    if len(cubes) != 1 or int(cubes[0].get("fixed_mask", -1)) != mask \
            or int(cubes[0].get("fixed_vals", -1)) != vals:
        reasons.append("根区域子立方 != 重算允许集")
    w = doc.get("witness") or {}
    sid = w.get("state_id")
    if sid is None:
        reasons.append("证书缺见证")
    else:
        want_label = 1 if claims == "1" else 0
        ok, why = _verify_witness_own(struct, meta, table, query_id, int(sid), want_label)
        if not ok:
            reasons.extend(why)
        for (b, v) in _declared_reads_pairs(table, query_id):
            if ((int(sid) >> int(b)) & 1) != int(v):
                reasons.append("见证不满足读位 (%d=%d)" % (int(b), int(v)))
                break
        checks["witness_state"] = int(sid)
    # 抽样等价：结构路 vs 链式路（声明上限内全量，超限均匀抽样）
    sample = _sample_states(mask, vals, n_bits, 4096)
    bad = 0
    for s in sample:
        d = table.derivation(query_id, int(s))
        adj: Dict = {}
        for (u, v) in d.edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        goals = set(d.b_side & d.nodes)
        seen = set(d.a_side & d.nodes)
        queue = list(seen)
        head = 0
        reached = False
        while head < len(queue) and not reached:
            x = queue[head]
            head += 1
            if x in goals:
                reached = True
            else:
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        queue.append(y)
        bfs = int(bool(d.attach_a) and bool(d.attach_b) and reached)
        if bfs != _own_probe_label(meta, struct, int(s)):
            bad += 1
            break
    checks["sample_states"] = len(sample)
    checks["sample_equivalence_violations"] = bad
    if bad:
        reasons.append("样本上结构路与链式合取不等")
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION, "checks": checks}


def _declared_reads_pairs(table, query_id: str) -> List[Tuple[int, int]]:
    reads, _k, _w = parse_reads(table, query_id)
    return reads


def _verify_mixed_proof(doc: Mapping, table, query_id: str,
                        reasons: List[str], checks: Dict[str, object]) -> dict:
    unknown = [k for k in doc if k not in PROOF_KEYS]
    if unknown:
        reasons.append("混合证明含未知顶层字段 %s" % unknown)
    bad = [k for k in doc if any(t in str(k).lower() for t in FORBIDDEN_KEY_TOKENS)]
    if bad:
        reasons.append("混合证明含真值/生产布尔字段 %s" % bad)
    if str(doc.get("table_id")) != str(table.table_id):
        reasons.append("table_id 不符")
    if str(doc.get("query_id")) != str(query_id):
        reasons.append("query_id 不符")
    dec = doc.get("decomposition") or {}
    struct, meta, mask, vals = _allowed_from_doc_or_table(table, query_id, reasons)
    if struct is None:
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    n_bits = int(table.n_bits)
    cube = dec.get("allowed_cube") or {}
    if int(cube.get("fixed_mask", -1)) != mask or int(cube.get("fixed_vals", -1)) != vals:
        reasons.append("允许集子立方与读组重算不符")
    if int(dec.get("allowed_states", -1)) != cube_count(mask, vals, n_bits):
        reasons.append("允许状态计数不符")
    # 通道层：逐通道重算状态/见证
    for ch in dec.get("channels") or []:
        vc = verify_channel_certificate(ch, table, query_id)
        if not vc["pass"]:
            reasons.append("通道 %s 验证失败: %s" % (ch.get("channel_id"), vc["reasons"][:2]))
    # 组合判据：从重算状态推期望标签集
    statuses = [str(ch.get("status")) for ch in (dec.get("channels") or [])]
    want = predicted_label_set(statuses)
    checks["predicted_label_set"] = want
    if want != ["0", "1"]:
        reasons.append("混合证明的通道状态并不推出混合（%s）" % want)
    # 划分树
    part = doc.get("partition") or {}
    regions = list(part.get("regions") or [])
    if not regions:
        reasons.append("混合证明无划分")
        return {"schema": SCHEMA_VERIFY, "pass": False, "reasons": reasons,
                "verifier_version": VERIFIER_VERSION, "checks": checks}
    by_id: Dict[str, dict] = {}
    dup = 0
    for r in regions:
        rid = str(r.get("region_id"))
        if rid in by_id:
            dup += 1
        by_id[rid] = r
    if dup:
        reasons.append("区域 id 重复 %d" % dup)
    roots = [r for r in regions if r.get("parent_id") is None]
    if len(roots) != 1:
        reasons.append("根区域不唯一 (%d)" % len(roots))
    else:
        r0 = roots[0]
        rc = list(r0.get("cubes") or [])
        if len(rc) != 1 or not cube_subset(int(rc[0].get("fixed_mask", 0)), int(rc[0].get("fixed_vals", 0)), mask, vals) \
                or not cube_subset(mask, vals, int(rc[0].get("fixed_mask", 0)), int(rc[0].get("fixed_vals", 0))):
            reasons.append("根区域 != 允许集")
        if r0.get("is_terminal"):
            reasons.append("根区域不得为终末")
    kids: Dict[str, List[dict]] = {}
    for r in regions:
        p = r.get("parent_id")
        if p is not None:
            if str(p) not in by_id:
                reasons.append("父区域 %s 不存在" % p)
            kids.setdefault(str(p), []).append(r)
    # 树：内部节点单立方；子区域互斥且计数和 = 父
    leaf_count = 0
    leaf_claims: Dict[str, int] = {"0": 0, "1": 0}
    coverage = 0
    for r in regions:
        ch = kids.get(str(r.get("region_id")), [])
        cubes = list(r.get("cubes") or [])
        if r.get("is_terminal"):
            if ch:
                reasons.append("终末区域 %s 仍有子区域" % r.get("region_id"))
            if len(cubes) != 1:
                reasons.append("终末区域 %s 非单立方" % r.get("region_id"))
                continue
            claim = r.get("label_claim")
            if claim not in ("0", "1"):
                reasons.append("终末区域 %s 标签非法 %r" % (r.get("region_id"), claim))
                continue
            cm, cv = int(cubes[0].get("fixed_mask", -1)), int(cubes[0].get("fixed_vals", -1))
            if not cube_subset(cm, cv, mask, vals):
                reasons.append("终末区域 %s 越出允许集" % r.get("region_id"))
            own, det = _own_direction(cm, cv, struct, meta)
            if own != claim:
                reasons.append("终末区域 %s 方向 %r != 声明 %r" % (r.get("region_id"), own, claim))
            w = r.get("witness") or {}
            sid = w.get("state_id")
            if sid is None:
                reasons.append("终末区域 %s 缺见证" % r.get("region_id"))
            elif not ((int(sid) & cm) == (cv & cm)):
                reasons.append("终末区域 %s 见证不在区域内" % r.get("region_id"))
            else:
                okw, whyw = _verify_witness_own(struct, meta, table, query_id, int(sid), int(claim))
                if not okw:
                    reasons.extend(whyw)
            leaf_count += 1
            leaf_claims[claim] = leaf_claims.get(claim, 0) + 1
            coverage += cube_count(cm, cv, n_bits)
        else:
            if not ch:
                reasons.append("内部区域 %s 无子区域" % r.get("region_id"))
                continue
            if len(cubes) != 1:
                reasons.append("内部区域 %s 非单立方" % r.get("region_id"))
                continue
            pm, pv = int(cubes[0].get("fixed_mask", -1)), int(cubes[0].get("fixed_vals", -1))
            total = 0
            child_cubes: List[Tuple[int, int]] = []
            for k in ch:
                kc = list(k.get("cubes") or [])
                if len(kc) != 1:
                    reasons.append("子区域 %s 非单立方" % k.get("region_id"))
                    continue
                km, kv = int(kc[0].get("fixed_mask", -1)), int(kc[0].get("fixed_vals", -1))
                if not cube_subset(km, kv, pm, pv):
                    reasons.append("子区域 %s 未继承父区域" % k.get("region_id"))
                child_cubes.append((km, kv))
                total += cube_count(km, kv, n_bits)
            for i in range(len(child_cubes)):
                for j in range(i + 1, len(child_cubes)):
                    if not cubes_disjoint(child_cubes[i][0], child_cubes[i][1],
                                          child_cubes[j][0], child_cubes[j][1]):
                        reasons.append("区域 %s 子区域重叠" % r.get("region_id"))
                        break
            if total != cube_count(pm, pv, n_bits):
                reasons.append("区域 %s 子区域计数和 != 父" % r.get("region_id"))
    checks["n_regions"] = len(regions)
    checks["n_leaves"] = leaf_count
    checks["leaf_claims"] = leaf_claims
    checks["declared_leaf_claims"] = part.get("leaf_claims")
    if dict(part.get("leaf_claims") or {}) != leaf_claims:
        reasons.append("声明叶标签计数与重算不符")
    if leaf_claims.get("0", 0) == 0 or leaf_claims.get("1", 0) == 0:
        reasons.append("划分未同时含两标签（不足证混合）")
    if coverage != cube_count(mask, vals, n_bits):
        reasons.append("叶覆盖 %d != 允许集 %d" % (coverage, cube_count(mask, vals, n_bits)))
    # 顶层双见证
    ws = doc.get("witnesses") or {}
    for key, want_label in (("label_1", 1), ("label_0", 0)):
        w = ws.get(key) or {}
        sid = w.get("state_id")
        if sid is None or int(w.get("label", -1)) != want_label:
            reasons.append("顶层见证 %s 缺或标签不符" % key)
            continue
        for (b, v) in _declared_reads_pairs(table, query_id):
            if ((int(sid) >> int(b)) & 1) != int(v):
                reasons.append("顶层见证 %s 不满足读位 (%d=%d)" % (key, b, v))
                break
        okw, whyw = _verify_witness_own(struct, meta, table, query_id, int(sid), want_label)
        if not okw:
            reasons.extend(whyw)
        checks["witness_%s_state" % key] = int(sid)
    # 预算
    bud = doc.get("budget") or {}
    if "nodes_cap" in bud and int(bud["nodes_cap"]) > 4096:
        reasons.append("声明节点预算超出 S2-S 档 (4096)")
    if int(bud.get("nodes_used", len(regions))) != len(regions):
        reasons.append("预算 nodes_used 与划分不符")
    return {"schema": SCHEMA_VERIFY, "pass": not reasons, "reasons": reasons,
            "verifier_version": VERIFIER_VERSION, "checks": checks}


__all__ = [
    "SCHEMA_CHANNEL_CERT", "SCHEMA_CHANNEL_BUNDLE", "SCHEMA_COMBINATION_PROOF",
    "SCHEMA_VERIFY", "VERIFIER_VERSION", "SELECTION_PROVENANCE", "WATERMARK",
    "STATUS_ALL1", "STATUS_ALL0", "STATUS_MIXED",
    "ParallelChannelError", "Channel", "ChannelSet", "ChannelCertificate", "ChannelResults",
    "MixedProof", "Bound", "Unresolved",
    "decompose_channels", "decompose_query", "certify_channel", "combine_channels",
    "predicted_label_set", "verify_channel_certificate", "verify_channel_bundle",
    "verify_combination_document",
    "parse_structure", "chain_meta", "parse_reads", "cube_count", "cube_states",
    "cube_subset", "cubes_disjoint", "bit_forced_one", "bit_forced_zero", "bit_allows_one",
]
