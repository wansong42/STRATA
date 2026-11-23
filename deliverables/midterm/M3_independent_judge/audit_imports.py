#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3_independent_judge/audit_imports.py
AST 导入审计 + 读取台账违规扫描。

证明：
  1. M3_independent_judge/*.py 中不存在 import / from ... import M1/M2/TC/TB/src
     求解代码，也不存在对禁读 .py 的 open()/read_text()/importlib 动态加载；
  2. read_ledger.jsonl 中没有任何 .py 读取，也没有白名单之外的 M1/M2 路径。

输出 import_audit.json，退出码 0=零违规。
"""
import ast, json, os, sys

FORBIDDEN_IMPORT_PREFIXES = (
    "M1_real_to_exact_bridge", "M2_active_identifiability",
    "TC_network", "TB_feasible", "fractureflow",
)
FORBIDDEN_PATH_MARKERS = (
    "M1_real_to_exact_bridge", "M2_active_identifiability",
    "TC_network", "TB_feasible", "src/fractureflow", "src\\fractureflow",
)
# M3 允许读取的数据文件（相对片段匹配）
ALLOWED_JSON_HINTS = (
    "cluster_register.json", "model.json", "summary.json",
    "certification.json", "active_check.json",
    "self_test_models",  # M3 自造的精确小例
    "M3_independent_judge",  # M3 自身输出
)

def iter_py_files(root):
    for name in sorted(os.listdir(root)):
        if name.endswith(".py"):
            yield os.path.join(root, name)

def scan_file(path):
    """AST 扫描 import 与字面 .py 读取调用。返回违规列表。"""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORT_PREFIXES or \
                   any(a.name.startswith(p) for p in FORBIDDEN_IMPORT_PREFIXES):
                    violations.append({"file": path, "line": node.lineno,
                                       "kind": "IMPORT_FORBIDDEN", "module": a.name})
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.startswith(p) for p in FORBIDDEN_IMPORT_PREFIXES):
                violations.append({"file": path, "line": node.lineno,
                                   "kind": "IMPORT_FROM_FORBIDDEN", "module": mod})
        elif isinstance(node, ast.Call):
            # open("...M1...xxx.py") / Path.read_text 等字面路径
            func = node.func
            fname = ""
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname in ("open", "read_text", "read_bytes", "exec_module",
                         "load_source", "load_module", "spec_from_file_location"):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value.replace("\\", "/")
                        if s.endswith(".py") and any(mk.replace("\\", "/") in s
                                                   for mk in FORBIDDEN_PATH_MARKERS):
                            violations.append({"file": path, "line": node.lineno,
                                               "kind": "READ_FORBIDDEN_PY", "path": s})
    # 动态 import 仅在真实 importlib 调用且目标为禁读路径时才算违规
    if "importlib" in src:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
               node.func.attr in ("import_module", "exec_module", "load_source",
                                  "load_module", "spec_from_file_location"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value.replace("\\", "/")
                        if any(mk.replace("\\", "/") in s for mk in FORBIDDEN_PATH_MARKERS):
                            violations.append({"file": path, "line": node.lineno,
                                               "kind": "SUSPECT_DYNAMIC_IMPORT", "path": s})
    return violations

def scan_ledger(ledger_path):
    violations = []
    total = 0
    if not os.path.exists(ledger_path):
        return total, [{"kind": "LEDGER_MISSING", "path": ledger_path}]
    with open(ledger_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            p = rec.get("path", "").replace("\\", "/")
            if p.endswith(".py"):
                violations.append({"line": ln, "kind": "LEDGER_PY_READ", "path": p})
                continue
            # 白名单校验：必须是允许的 JSON（或 M3 自身输出）
            base = os.path.basename(p)
            if base not in ("cluster_register.json", "model.json", "summary.json",
                            "certification.json", "active_check.json") and \
               "self_test_models" not in p and "M3_independent_judge" not in p:
                violations.append({"line": ln, "kind": "LEDGER_NON_WHITELIST", "path": p})
            if any(mk.replace("\\", "/") in p for mk in FORBIDDEN_PATH_MARKERS):
                if base not in ("cluster_register.json", "model.json", "summary.json",
                                "certification.json", "active_check.json"):
                    violations.append({"line": ln, "kind": "LEDGER_FORBIDDEN_DIR",
                                       "path": p})
    return total, violations

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "import_audit.json")
    files = list(iter_py_files(here))
    all_viol = []
    for f in files:
        all_viol.extend(scan_file(f))
    ledger_path = os.path.join(here, "read_ledger.jsonl")
    total, ledger_viol = scan_ledger(ledger_path)
    all_viol.extend(ledger_viol)
    report = {
        "schema": "m3-import-audit/v1",
        "scanned_py_files": [os.path.basename(f) for f in files],
        "ast_import_violations": len([v for v in all_viol
                                      if v.get("kind", "").startswith("IMPORT")]),
        "read_ledger_records": total,
        "ledger_violations": len(ledger_viol),
        "violations": len(all_viol),
        "violations_detail": all_viol,
        "forbidden_import_prefixes": list(FORBIDDEN_IMPORT_PREFIXES),
        "verdict": "PASS" if not all_viol else "FAIL",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps({"violations": len(all_viol), "ledger_records": total,
                      "verdict": report["verdict"]}, ensure_ascii=False))
    return 0 if not all_viol else 1

if __name__ == "__main__":
    raise SystemExit(main())
