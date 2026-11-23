# -*- coding: utf-8 -*-
"""结果件复现校验工具."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_dict(d) -> str:
    blob = json.dumps(d, allow_nan=False, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def sha256_metrics(obj: dict) -> str:
    """Hash of the metrics block only (excludes provenance.sha256)."""
    metrics = obj.get("metrics", {})
    return sha256_dict(metrics)

def verify_result_file(result_json_path) -> Dict[str, Any]:
    """校验结果件: 重算 metrics hash vs 存储的 provenance.sha256."""
    p = Path(result_json_path)
    if not p.exists():
        return {"path": str(p), "error": "FILE_NOT_FOUND"}
    obj = json.loads(p.read_text(encoding="utf-8"))
    prov = obj.get("provenance", {})
    expected = prov.get("sha256", obj.get("sha256", ""))
    # recompute from metrics only
    actual = sha256_metrics(obj)
    match = (expected == actual) if expected else None
    return {"path": str(p), "expected_sha256": expected, "actual_sha256": actual,
            "match": match, "has_provenance_sha": bool(expected)}

def scan_results_dir(dir_path=None) -> list:
    if dir_path is None:
        dir_path = PROJECT_ROOT / "results"
    dir_path = Path(dir_path)
    out = []
    for p in dir_path.rglob("*.json"):
        try:
            v = verify_result_file(p)
            out.append({"path": str(p.relative_to(dir_path)), "match": v.get("match"), "ok": True})
        except Exception as e:
            out.append({"path": str(p.relative_to(dir_path)), "ok": False, "error": str(e)})
    return out
