# -*- coding: utf-8 -*-
"""NNS 主台账统一追加 + 验证."""
from __future__ import annotations
import json, os, shutil, time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAIN_LEDGER = PROJECT_ROOT / "results" / "digital_ledger.json"

def load_ledger(path=None) -> dict:
    p = Path(path) if path else MAIN_LEDGER
    if not p.exists():
        return {"version": "v1.0", "entries": [], "changelog": []}
    return json.loads(p.read_text(encoding="utf-8"))

def _next_vid(ledger: dict) -> int:
    entries = ledger.get("entries", [])
    if not entries:
        return 1
    ids = [e.get("v_id", 0) for e in entries]
    return max(ids) + 1

def append_result(entry: dict, ledger_path=None, backup=True) -> dict:
    """追加一条结果到主台账 (append-only)."""
    p = Path(ledger_path) if ledger_path else MAIN_LEDGER
    ledger = load_ledger(p)

    if backup and p.exists():
        bak = p.with_suffix(f".backup_{int(time.time())}.json")
        shutil.copy2(p, bak)

    vid = _next_vid(ledger)
    rec = {
        "v_id": vid,
        "label": entry.get("label", f"V{vid}"),
        "value": entry["value"],
        "unit": entry.get("unit", "deg"),
        "source": entry.get("source", "unknown"),
        "caliber": entry.get("caliber", "research"),
        "evidence": entry.get("evidence", {}),
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes": entry.get("supersedes"),
    }
    ledger.setdefault("entries", []).append(rec)
    ledger.setdefault("changelog", []).append({
        "v_id": vid, "action": "append", "at": rec["added_at"]
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ledger, f, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=False)
    return rec

def verify_ledger(ledger_path=None) -> dict:
    """自检: version 字段 + 唯一 v_id + 计数."""
    p = Path(ledger_path) if ledger_path else MAIN_LEDGER
    ledger = load_ledger(p)
    entries = ledger.get("entries", [])
    v_ids = [e.get("v_id") for e in entries]
    return {
        "path": str(p),
        "version": ledger.get("version"),
        "n_entries": len(entries),
        "unique_v_ids": len(set(v_ids)),
        "has_dup_v_ids": len(v_ids) != len(set(v_ids)),
        "has_version_field": "version" in ledger,
    }
