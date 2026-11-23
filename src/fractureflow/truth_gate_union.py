# -*- coding: utf-8 -*-
"""集合档真值分级表 · 并集访问器（R206-T7b 授权令落地）。

背景
----
R204 交付 `results/r204_set_metric_truth_gate/truth_gate.json`（17 站点分级的原始件）；
R207 靶场落地后产出修正案 `results/r207_synth_truth_range/truth_gate_amendment.json`
（9 个 `TRUTH(synthetic)` 站点，提案态 `proposed_pending_architect_merge`）。

R206-T7b 已把修正案**脚本化合并**进 `truth_gate.json#rows`（17 → 26 行，
原 17 行逐字节零改动），并给 `grade_legend` 增补 `TRUTH(synthetic)`。
本模块是该面的**唯一读取入口**：

* 合并前过渡期：消费者须经本访问器读 **并集**，不得各自散读两件（"散读"会让
  "某站点到底有没有裁判" 出现两个答案，正是 R204 勘误的病灶）；
* 合并后：修正案仍在盘（提案态原件不删，零硬删除），并集与主件**恒等**，
  访问器退化为幂等的完整性校验 —— 双源不一致即报错，防未来再分片后口径漂移。

铁律
----
* CIRCULAR 站点**禁报覆盖率/精度**（只可报"一致性"），此处只做分级表读取，不做声明；
* 水印：研究线数字，禁入对外。
"""
from __future__ import annotations

import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))

TRUTH_GATE_REL = "results/r204_set_metric_truth_gate/truth_gate.json"
AMENDMENT_REL = "results/r207_synth_truth_range/truth_gate_amendment.json"

#: 允许出现在 `rows[].grade` 的分级枚举（R206-T7b 增补 TRUTH(synthetic)）
GRADES = ("TRUTH", "TRUTH(synthetic)", "TRUTH-DEGENERATE", "CIRCULAR(no truth)", "NONE")


class UnionError(RuntimeError):
    """双源不一致 / 修正案未合并 —— 调用方必须停手上报, 不得静默降级。"""


def _load(rel):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p


def canonical(obj):
    """逐字节比较用的规范化序列化（键排序 + 紧凑分隔符 + 禁 NaN）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def row_sha(row):
    return hashlib.sha256(canonical(row).encode("utf-8")).hexdigest()


def load_truth_gate_rows(strict=True):
    """读并集分级表。

    返回 dict: `rows` / `grade_legend` / `provenance` / `n_rows` / `merged` / `contradictions`。

    `merged=True` 表示修正案的行已并入主件（R206-T7b 后为常态）；
    `merged=False` 表示仍处过渡期（主件尚未合并），此时 `rows` 为两件的并集 ——
    调用方**不需要**关心是哪种，取到的集合语义相同。
    """
    gate, gp = _load(TRUTH_GATE_REL)
    amend, ap = _load(AMENDMENT_REL)

    g_rows = dict(gate.get("rows") or {})
    a_rows = dict(amend.get("rows") or {})
    legend = dict(gate.get("grade_legend") or {})
    legend.update(amend.get("grade_legend_addendum") or {})

    contradictions = []
    for k in set(g_rows) & set(a_rows):
        if canonical(g_rows[k]) != canonical(a_rows[k]):
            contradictions.append(k)
    if strict and contradictions:
        raise UnionError("并集冲突: 同名站点在两件中内容不一致 %s" % contradictions)

    merged = {**a_rows, **g_rows}          # 主件优先（合并后主件即权威）
    bad = sorted(k for k, v in merged.items() if v.get("grade") not in GRADES)
    if strict and bad:
        raise UnionError("未知分级 (grade 越出枚举 %s): %s" % (list(GRADES), bad))

    return dict(rows=merged, grade_legend=legend, n_rows=len(merged),
                merged=bool(set(a_rows) <= set(g_rows)),
                from_gate=len(g_rows), from_amendment=len(a_rows),
                contradictions=contradictions,
                provenance=[os.path.relpath(gp, ROOT).replace("\\", "/"),
                            os.path.relpath(ap, ROOT).replace("\\", "/")])


def assert_union_complete(expected_n=None):
    """完整性断言（供守卫/消费者开跑前调用）。"""
    got = load_truth_gate_rows()
    if expected_n is not None and got["n_rows"] != expected_n:
        raise UnionError("并集行数 %d != 期望 %d" % (got["n_rows"], expected_n))
    return got
