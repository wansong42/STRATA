# -*- coding: utf-8 -*-
"""S2-S · 规模可扩的声明 N0 有限表族（SER / PAR × END / MID）。

本件是任务单 S2-S（sha12 ``7cbfff081b7b``）§2.1 要求的**新建模块**；
``finite_scope.py`` / ``certificate_solver.py`` / ``identification.py`` 一字未改（只 import）。

口径（判据权威 = 任务单 §4 + ``results/s2s_scale_cost/prereg_s2s.json``）：

- 表族两条，按 d（位数）可扩：
  * ``SER``：d 条串联通道位，节点 v00..v_d；位 i = 通道 (v_i, v_{i+1})。
  * ``PAR``：d = 2k，k 个阶段，每阶段两条平行通道位（2i, 2i+1）连接同一节点对 (v_i, v_{i+1})。
- 目标函数（**声明规则的显式布尔函数**，非随机标签）：
  ``label(s) = 1`` 当且仅当「从 A 到目标的每个阶段都至少有一条通道位在场」。
  SER 下该式即「全部位为 1」；PAR 下即「每阶段至少一条在场」。两者都等价于
  声明图上的 A→目标可达性，由「阶段式」定义给出，不依赖任何数据或标签统计。
- 两种目标：``END``（末端节点）/ ``MID``（中段节点 v_{m}），m = 阶段数 // 2。
  同一张位表、同一套动作，只换 A→目标 的端点对，因此「不同端点/目标」是**同一表族内的对照**。
- 观测协议：复用 ``certificate_solver`` 的 ``B13_SCENARIO`` 查询形态（该模块只读，
  无法新增 kind）。``query_id = "B13:scenario:<name>"``；``scenario[name].reads`` 给出
  被读位与观测值。**读组全为单位读**：多元组读只释放在场计数，允许集会不再是子立方，
  求解器的声明情景协议不接受，本单不做（登记为已知限制）。
- 本层不写盘；供 ``scripts/s2s_*.py`` 调用。

纪律：
- 标签只由声明规则算出；全枚举（逐状态三要素 + 可达性）与精简实现（阶段掩码整数运算）
  是两条独立实现，必须逐状态一致，否则 ``SystemExit``。
- 不按答案筛表/筛查询：位表、动作面板、情景（读位与取值）全由 d 与 target 的**声明规则**
  生成，与解答无关。
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fractureflow import finite_scope as fs
from fractureflow import identification as ident

SCHEMA_TABLE = "s2s.scale_table/v1"
FAMILIES = ("SER", "PAR")
TARGETS = ("END", "MID")
SCENARIOS = ("S0_free", "S1_alt3", "S2_alt5", "S3_alt7",
             "S4_zero_panel8", "S5_one_panel8", "S6_path", "S7_cut", "S8_bridge0")
QUERY_PREFIX = "B13:scenario:"
WATERMARK = {"research_line": "S2S_scale_cost",
             "tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N0",
             "dictionary": "COMPLETE_KNOWN", "claim_scope": "DECLARED_SYNTHETIC_TABLE"}


class ScaleError(ValueError):
    pass


def node_name(i: int) -> str:
    return "v%02d" % int(i)


# ---------------------------------------------------------------------------
# 表族几何（全部由 d 与 target 声明生成，不看任何标签）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleSpec:
    family: str
    d: int
    target: str

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ScaleError(f"未知表族 {self.family!r}")
        if self.target not in TARGETS:
            raise ScaleError(f"未知目标 {self.target!r}")
        if int(self.d) < 8:
            raise ScaleError("d 过小（<8）：动作面板无法凑满 8 个")
        if self.family == "PAR" and int(self.d) % 2 != 0:
            raise ScaleError("PAR 要求 d 为偶数（每阶段两位）")

    @property
    def table_id(self) -> str:
        return "S2S-%s-%s-d%d" % (self.family, self.target, int(self.d))

    @property
    def n_bits(self) -> int:
        return int(self.d)

    @property
    def k(self) -> int:
        """阶段数：SER 每阶段一位，PAR 每阶段两位。"""
        return int(self.d) // 2 if self.family == "PAR" else int(self.d)

    @property
    def n_nodes(self) -> int:
        return self.k + 1

    @property
    def target_node(self) -> int:
        return self.k if self.target == "END" else self.k // 2

    def stage_bits(self, i: int) -> Tuple[int, ...]:
        if self.family == "PAR":
            return (2 * i, 2 * i + 1)
        return (i,)

    def to_dict(self) -> dict:
        return {"family": self.family, "d": int(self.d), "target": self.target,
                "table_id": self.table_id, "n_bits": self.n_bits, "k": self.k,
                "n_nodes": self.n_nodes, "target_node": self.target_node}


def edge_bits(spec: ScaleSpec) -> List[Tuple[int, str, str]]:
    """位 → (u, v) 的声明映射；端点已按 ``identification._canon`` 口径（字典序 min,max）规范化。"""
    out: List[Tuple[int, str, str]] = []
    for i in range(spec.k):
        u, v = node_name(i), node_name(i + 1)
        lo, hi = (u, v) if u <= v else (v, u)
        for b in spec.stage_bits(i):
            out.append((int(b), lo, hi))
    return out


def target_stage_masks(spec: ScaleSpec) -> List[int]:
    """A→目标 路径上的阶段掩码（每阶段「至少一位在场」即该阶段通）。"""
    masks: List[int] = []
    for i in range(spec.target_node):
        m = 0
        for b in spec.stage_bits(i):
            m |= 1 << int(b)
        masks.append(m)
    return masks


def all_stage_masks(spec: ScaleSpec) -> List[int]:
    masks: List[int] = []
    for i in range(spec.k):
        m = 0
        for b in spec.stage_bits(i):
            m |= 1 << int(b)
        masks.append(m)
    return masks


def panel_bits(d: int) -> List[int]:
    """声明动作面板（按位号升序去重；至少 8 个）。规则与标签无关。"""
    cand = [0, 1, 2, 3, d // 2 - 1, d // 2, d // 2 + 1, d - 2, d - 1]
    out = sorted({int(b) for b in cand if 0 <= b < d})
    if len(out) < 8:
        raise ScaleError("动作面板不足 8 个（d=%d）" % d)
    return out


def path_bits(spec: ScaleSpec) -> List[int]:
    """声明的「最短通道路径」位：SER = 每位；PAR = 每阶段的奇数位（与求解器位序一致）。"""
    out: List[int] = []
    for i in range(spec.target_node):
        bits = spec.stage_bits(i)
        out.append(bits[-1])
    return out


def cut_bits(spec: ScaleSpec) -> List[int]:
    """声明的割观测：第一阶段两条通道位 + 面板低位，取前 6 个不同位，全部固定为 0。"""
    s0 = list(spec.stage_bits(0))
    merged = sorted(set(s0) | set(panel_bits(spec.d)))
    return merged[:6]


def scenario_reads(spec: ScaleSpec, name: str) -> List[Tuple[int, int]]:
    """声明情景 → [(位, 观测值)]。取值全由固定规则给出，不看答案。"""
    pan = panel_bits(spec.d)
    if name == "S0_free":
        return []
    if name == "S1_alt3":
        return [(pan[i], 1 - (i % 2)) for i in range(3)]
    if name == "S2_alt5":
        return [(pan[i], 1 - (i % 2)) for i in range(5)]
    if name == "S3_alt7":
        return [(pan[i], 1 - (i % 2)) for i in range(7)]
    if name == "S4_zero_panel8":
        return [(pan[i], 0) for i in range(8)]
    if name == "S5_one_panel8":
        return [(pan[i], 1) for i in range(8)]
    if name == "S6_path":
        return [(b, 1) for b in path_bits(spec)]
    if name == "S7_cut":
        return [(b, 0) for b in cut_bits(spec)]
    if name == "S8_bridge0":
        return [(0, 0)]
    raise ScaleError("未知情景 %r" % name)


def query_id(name: str) -> str:
    return QUERY_PREFIX + str(name)


def reads_of(table: fs.FiniteTable, qid: str) -> List[Tuple[int, int]]:
    meta = table.tags["s2_queries"]
    meta = meta if isinstance(meta, dict) else None
    if not meta:
        raise ScaleError("表未登记 s2_queries")
    name = qid[len(QUERY_PREFIX):] if qid.startswith(QUERY_PREFIX) else qid
    sc = meta["scenarios"].get(name)
    if sc is None:
        raise ScaleError("未知情景 %r" % name)
    return [(int(b), int(v)) for (b, v) in sc["reads"]]


# ---------------------------------------------------------------------------
# 建表
# ---------------------------------------------------------------------------

def stage_label(masks: Sequence[int], s: int) -> int:
    """精简实现（整数位运算，无对象构造）：所有阶段至少一位在场。"""
    for m in masks:
        if not (int(s) & int(m)):
            return 0
    return 1


def label_array(spec: ScaleSpec) -> np.ndarray:
    """向量化标签（声明规则的直接实现；与逐状态两路互为独立实现）。"""
    n = 1 << spec.n_bits
    states = np.arange(n, dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    for m in target_stage_masks(spec):
        ok &= (states & np.int64(m)) != 0
    return ok.astype(np.int8)


def _derivation_factory(spec: ScaleSpec, labels: np.ndarray):
    bits = edge_bits(spec)
    nodes = frozenset(node_name(i) for i in range(spec.n_nodes))
    a_node = node_name(0)
    b_node = node_name(spec.target_node)
    a_inc = tuple(b for (b, u, v) in bits if a_node in (u, v))
    b_inc = tuple(b for (b, u, v) in bits if b_node in (u, v))

    def deriv(qid: str, s: int) -> "ident.StateDerivation":
        s = int(s)
        es = [(u, v) for (b, u, v) in bits if (s >> b) & 1]
        att_a = any((s >> b) & 1 for b in a_inc)
        att_b = any((s >> b) & 1 for b in b_inc)
        return ident.StateDerivation(
            state_id=s, nodes=nodes, edges=frozenset(es),
            a_side=frozenset({a_node}) if att_a else frozenset(),
            b_side=frozenset({b_node}) if att_b else frozenset(),
            attach_a=bool(att_a), attach_b=bool(att_b), label=int(labels[s]))

    return deriv, {"nodes": sorted(nodes), "a_node": a_node, "b_node": b_node,
                   "a_incident_bits": [int(x) for x in a_inc],
                   "b_incident_bits": [int(x) for x in b_inc]}


def structure_of_spec(spec: ScaleSpec) -> dict:
    """``certificate_solver.TableStructure`` 的 dict 形态（EDGE_BITS）。"""
    bits = edge_bits(spec)
    a_node = node_name(0)
    b_node = node_name(spec.target_node)
    return {"kind": "EDGE_BITS", "n_bits": spec.n_bits,
            "nodes": [node_name(i) for i in range(spec.n_nodes)],
            "edges": [[int(b), u, v] for (b, u, v) in bits],
            "a_nodes": [a_node], "b_nodes": [b_node],
            "a_incident_bits": [int(b) for (b, u, v) in bits if a_node in (u, v)],
            "b_incident_bits": [int(b) for (b, u, v) in bits if b_node in (u, v)],
            "certain": [], "uncertain": [], "contact_edges": [],
            "attach_a_disks": [], "attach_b_disks": []}


def scenario_meta(spec: ScaleSpec) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in SCENARIOS:
        reads = scenario_reads(spec, name)
        out[name] = {"reads": [[int(b), int(v)] for (b, v) in reads],
                     "n_reads": len(reads),
                     "observation_in_forward_image": True,
                     "note": "读组全为单位读；任意位值观测在声明表内都可实现（无 MODEL_CONFLICT）"}
    return out


def action_panel(spec: ScaleSpec) -> Tuple[fs.ReadAction, ...]:
    """声明的动作面板：每位一个单位读组，代价 1/2/3 三档轮转（声明值，不做选点）。"""
    acts = []
    for i, b in enumerate(panel_bits(spec.d)):
        acts.append(fs.ReadAction("P%02d" % i, (1 << int(b),), 1 + (i % 3)))
    return tuple(acts)


def build_table(spec: ScaleSpec) -> Tuple[fs.FiniteTable, dict]:
    labels = label_array(spec)
    deriv, dmeta = _derivation_factory(spec, labels)
    table = fs.FiniteTable(
        table_id=spec.table_id, n_bits=spec.n_bits, labels={"LABEL": labels},
        baseline=(), actions=action_panel(spec), attach={}, derivation=deriv,
        tags={"tier": "SYNTHETIC_EXCLUDED_FROM_REAL_CLAIMS", "noise": "N0",
              "dictionary": "COMPLETE_KNOWN",
              "claim_scope": "DECLARED_SYNTHETIC_TABLE",
              "model": "S2S 声明阶段式路径族（%s/%s）" % (spec.family, spec.target),
              "geometry_semantics": "DECLARED_COMPUTATIONAL",
              "s2_structure": structure_of_spec(spec),
              "s2_queries": {"kind": "B13_SCENARIO", "scenarios": scenario_meta(spec)},
              "s2s_spec": spec.to_dict()})
    meta = {"spec": spec.to_dict(), "derivation": dmeta,
            "n_c1": int(labels.sum()), "n_c0": int(labels.shape[0] - labels.sum()),
            "actions": [a.to_dict() for a in table.actions],
            "panel_bits": panel_bits(spec.d), "path_bits": path_bits(spec),
            "cut_bits": cut_bits(spec),
            "target_stage_masks": target_stage_masks(spec)}
    return table, meta


# ---------------------------------------------------------------------------
# 允许集与两路全枚举
# ---------------------------------------------------------------------------

def allowed_states(d: int, reads: Sequence[Tuple[int, int]]) -> List[int]:
    """允许集（位值观测）：与全部读数一致的位赋值集合，按升序。"""
    mask = 0
    vals = 0
    seen = set()
    for (b, v) in reads:
        b, v = int(b), int(v)
        if b in seen:
            raise ScaleError("读数重复位 %d" % b)
        seen.add(b)
        mask |= 1 << b
        if v:
            vals |= 1 << b
    n = 1 << int(d)
    return [s for s in range(n) if (s & mask) == vals]


def full_enum_reference(table: fs.FiniteTable, allowed: Sequence[int]) -> dict:
    """参考全枚举：逐状态建三要素 + ``state_connected``（验证器同路的显式口径）。"""
    t0 = time.perf_counter()
    n1 = 0
    for s in allowed:
        d = table.derivation("LABEL", int(s))
        if ident.state_connected(d):
            n1 += 1
    return {"states": len(allowed), "n_c1": int(n1), "n_c0": int(len(allowed) - n1),
            "seconds": round(time.perf_counter() - t0, 6),
            "route": "state_derivation + identification.state_connected"}


def full_enum_lean(masks: Sequence[int], allowed: Sequence[int]) -> dict:
    """精简全枚举：逐状态整数阶段掩码判定（无对象构造）。"""
    t0 = time.perf_counter()
    n1 = 0
    for s in allowed:
        if stage_label(masks, int(s)):
            n1 += 1
    return {"states": len(allowed), "n_c1": int(n1), "n_c0": int(len(allowed) - n1),
            "seconds": round(time.perf_counter() - t0, 6),
            "route": "integer_stage_masks"}


# ---------------------------------------------------------------------------
# 内存（Windows 工作集；无 psapi 则如实报 None）
# ---------------------------------------------------------------------------

def _meminfo() -> dict:
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        ok = fn(k32.GetCurrentProcess(), ctypes.byref(pmc), ctypes.c_ulong(pmc.cb))
        if not ok:
            return {"peak_rss_mb": None, "rss_mb": None,
                    "route": "psapi.GetProcessMemoryInfo",
                    "error": "GetProcessMemoryInfo 返回 0 (err=%d)" % ctypes.get_last_error()}
        return {"peak_rss_mb": round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 3),
                "rss_mb": round(pmc.WorkingSetSize / (1024.0 * 1024.0), 3),
                "route": "psapi.GetProcessMemoryInfo"}
    except Exception as e:  # 如实报取不到，不猜
        return {"peak_rss_mb": None, "rss_mb": None,
                "route": "psapi.GetProcessMemoryInfo", "error": "%s: %s" % (type(e).__name__, e)}


def peak_rss_mb() -> Optional[float]:
    return _meminfo()["peak_rss_mb"]


def rss_mb() -> Optional[float]:
    return _meminfo()["rss_mb"]


__all__ = [
    "SCHEMA_TABLE", "FAMILIES", "TARGETS", "SCENARIOS", "QUERY_PREFIX", "WATERMARK",
    "ScaleError", "ScaleSpec", "node_name", "edge_bits", "target_stage_masks",
    "all_stage_masks", "panel_bits", "path_bits", "cut_bits", "scenario_reads",
    "query_id", "reads_of", "stage_label", "label_array", "structure_of_spec",
    "scenario_meta", "action_panel", "build_table", "allowed_states",
    "full_enum_reference", "full_enum_lean", "peak_rss_mb", "rss_mb",
]
