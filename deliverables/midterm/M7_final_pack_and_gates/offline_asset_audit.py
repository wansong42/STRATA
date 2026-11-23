# -*- coding: utf-8 -*-
"""M7 · 离线资产审计：外部 URL 引用 + 本地引用可解析性。

口径：
  * app_owned = 除 vendor/ 之外的 HTML/CSS/JS
  * vendor/    = 第三方打包库（plotly.min.js），其中字符串常量含地图/文档 URL，
                 但本应用不调用这些代码路径；运行时是否真联网由 cold_start_smoke 的
                 Chrome 请求日志单独举证，不以字符串计数代替。
  * 外部引用判定：http(s):// 或 //host/，排除 W3C XML 命名空间（非网络请求）。
退出码：0 无违规；2 app_owned 外部引用；3 本地引用缺失；4 枚举为空。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
DEFAULT_ROOT = ROOT / "demos" / "workbench"
M6_DECK = ROOT / "results" / "midterm_sprint_20261120" / "M6_midterm_defense_deck" / "midterm_deck.html"

URL_RE = re.compile(r"https?://[^\s\"'<>()\\]+")
PROTO_REL_RE = re.compile(r"(?<![:/\"'])//([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?::\d+)?/[^\s\"'<>()\\]*")
SRC_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.I)
CSS_URL_RE = re.compile(r"""url\(\s*["']?([^"')]+)["']?\s*\)""", re.I)
NS_HOSTS = ("www.w3.org",)


ASSET_REF_RE = re.compile(r"^[A-Za-z0-9_./-]+\.(?:png|jpe?g|svg|gif|webp|ico|css|js|mjs|woff2?|ttf|otf|json)$", re.I)


def _is_local_asset_ref(v: str) -> bool:
    return bool(ASSET_REF_RE.match(v.split("?")[0].split("#")[0]))


def scan_file(p: Path, root: Path, vendor: bool) -> Dict[str, Any]:
    text = p.read_text(encoding="utf-8", errors="replace")
    rel = str(p.relative_to(root)).replace("\\", "/")
    hits = []
    for m in URL_RE.finditer(text):
        u = m.group(0)
        host = re.sub(r"^https?://", "", u).split("/")[0].lower()
        kind = "namespace" if host in NS_HOSTS else ("vendor_string" if vendor else "external")
        line = text.count("\n", 0, m.start()) + 1
        hits.append({"url": u[:200], "kind": kind, "line": line})
    for m in PROTO_REL_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        hits.append({"url": "//" + m.group(0)[2:][:180], "kind": "vendor_string" if vendor else "external",
                     "line": line})
    local_refs = []
    if p.suffix.lower() in (".html", ".htm"):
        for m in SRC_RE.finditer(text):
            v = m.group(1).strip()
            if v.startswith(("http://", "https://", "//", "data:", "blob:", "#", "mailto:", "javascript:")):
                continue
            if not _is_local_asset_ref(v):
                continue
            local_refs.append(v)
    if p.suffix.lower() == ".css":
        for m in CSS_URL_RE.finditer(text):
            v = m.group(1).strip()
            if v.startswith(("http://", "https://", "//", "data:", "blob:", "#")):
                continue
            if not _is_local_asset_ref(v):
                continue
            local_refs.append(v)
    missing = []
    for v in local_refs:
        q = v.split("?")[0].split("#")[0]
        tgt = (p.parent / q).resolve()
        if not tgt.exists():
            missing.append(v)
    return {"file": rel, "bytes": p.stat().st_size, "vendor": vendor,
            "external_hits": [h for h in hits if h["kind"] == "external"],
            "vendor_string_hits": [h for h in hits if h["kind"] == "vendor_string"],
            "namespace_hits": len([h for h in hits if h["kind"] == "namespace"]),
            "local_refs": len(local_refs), "missing_local_refs": missing}


def scan_root(root: Path, label: str) -> Dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".html", ".htm", ".js", ".css")]
    if not files:
        return {"label": label, "root": str(root), "files": 0, "error": "NO_SCANNABLE_FILES"}
    rows = []
    for p in sorted(files):
        vendor = "vendor" in p.relative_to(root).parts
        rows.append(scan_file(p, root, vendor))
    app_ext = [(r["file"], h) for r in rows if not r["vendor"] for h in r["external_hits"]]
    missing = [(r["file"], v) for r in rows for v in r["missing_local_refs"]]
    return {
        "label": label, "root": str(root), "files": len(rows),
        "app_files": sum(1 for r in rows if not r["vendor"]),
        "vendor_files": sum(1 for r in rows if r["vendor"]),
        "app_external_hits": len(app_ext),
        "app_external_details": [{"file": f, **h} for f, h in app_ext],
        "vendor_literal_hits": sum(len(r["vendor_string_hits"]) for r in rows),
        "namespace_hits": sum(r["namespace_hits"] for r in rows),
        "local_refs_checked": sum(r["local_refs"] for r in rows),
        "missing_local_refs": [{"file": f, "ref": v} for f, v in missing],
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates" / "offline_asset_audit.json"))
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"ROOT_MISSING: {root}", file=sys.stderr)
        return 4
    strata = scan_root(root, "strata_workbench")
    deck = {}
    if M6_DECK.is_file():
        deck = scan_file(M6_DECK, M6_DECK.parent, vendor=False)
        deck = {"label": "m6_offline_deck", "file": str(M6_DECK), **deck,
                "external_hits": deck["external_hits"], "app_external_hits": len(deck["external_hits"]),
                "missing_local_refs": deck["missing_local_refs"]}
    app_ext = strata.get("app_external_hits", 0) + (len(deck.get("external_hits", [])) if deck else 0)
    missing = len(strata.get("missing_local_refs", [])) + (len(deck.get("missing_local_refs", [])) if deck else 0)
    verdict = "PASS" if app_ext == 0 and missing == 0 and strata.get("files", 0) > 0 else "FAILED"
    out = {
        "schema": "m7-offline-asset-audit/v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "STRATA 前端静态资产（demos/workbench）+ M6 离线 HTML 演示包；外部 URL 字面量分级 + 本地引用可解析性",
        "caliber": {
            "app_owned_external": "除 vendor/ 外，任何 http(s):// 或 //host/ 出现即记违规（W3C 命名空间除外）",
            "vendor_literal": "vendor/plotly.min.js 内地图/文档 URL 为库字符串常量，本应用不调用；运行期以 Chrome 请求日志为准",
            "local_refs": "HTML src/href 与 CSS url() 的相对路径必须在盘",
        },
        "strata": {k: v for k, v in strata.items() if k != "rows"},
        "strata_rows": strata.get("rows", []),
        "m6_deck": deck,
        "verdict": verdict,
        "not_claimed": [
            "字符串层面的外部 URL 计数不等于实际联网行为；运行期外部请求数由 cold_start_smoke.json#badge_probe 举证",
            "未对 plotly.min.js 做逐行语义审计，只判定本应用未引用其地图/地理代码路径",
        ],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "app_external": app_ext, "missing_local": missing,
                      "vendor_literals": strata.get("vendor_literal_hits"),
                      "files": strata.get("files"), "out": str(outp)}, ensure_ascii=False, indent=1))
    if app_ext:
        return 2
    if missing:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
