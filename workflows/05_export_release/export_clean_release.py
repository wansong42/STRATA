# -*- coding: utf-8 -*-
"""Export a clean, standalone, <1 GB deliverable release bundle with zero legacy clutter.

Usage:
    # Preview included items and total size without writing files
    E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --dry-run

    # Export clean release bundle to target folder
    E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --out E:/fracture_release_20261123
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]

EXCLUDE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", "node_modules", "_work", "render",
    "pre_edit_archive", "_gate_runs", "_scratch",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".ply", ".las", ".laz", ".pt", ".pth", ".safetensors", ".onnx"}
MAX_SINGLE_FILE_MB = 80.0


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_plan(github_clean: bool = False) -> list[tuple[pathlib.Path, str]]:
    """Return list of (src_abs_path, rel_dest_in_release)."""
    items: list[tuple[pathlib.Path, str]] = []
    max_mb = 45.0 if github_clean else MAX_SINGLE_FILE_MB

    def add_file(rel_src: str, rel_dst: str | None = None):
        p = ROOT / rel_src
        if p.exists() and p.is_file():
            if p.stat().st_size <= max_mb * 1024 * 1024:
                items.append((p, rel_dst or rel_src))

    def add_tree(rel_src: str, rel_dst: str, allow_large: bool = False):
        base = ROOT / rel_src
        if not base.exists():
            return
        for dp, dn, fn in os.walk(str(base)):
            dn[:] = [d for d in dn if d not in EXCLUDE_DIR_NAMES and not d.startswith("_architect_audit_")]
            for f in fn:
                fp = pathlib.Path(dp) / f
                if fp.suffix.lower() in EXCLUDE_SUFFIXES:
                    continue
                if github_clean and ("交接文档" in f or "AGENTS" in f):
                    continue
                if not allow_large and fp.stat().st_size > max_mb * 1024 * 1024:
                    continue
                sub = fp.relative_to(base).as_posix()
                items.append((fp, f"{rel_dst}/{sub}"))

    # 1. Root docs, registry & contracts
    root_files = ["README.md", "NAVIGATION.md", "data_registry.json", ".gitignore"]
    if not github_clean:
        root_files.append("AGENTS.md")
    for rf in root_files:
        add_file(rf)
    add_tree("contracts", "contracts")

    # 2. Clean core source, workflows, exact triangle kernel, and public showcase
    add_tree("src/fractureflow", "src/fractureflow")
    add_tree("workflows", "workflows")
    add_tree("deploy/strata_showcase", "deploy/strata_showcase")
    add_tree("results/w180_technical_route_20261117/execution_next/triangle_contact",
             "src/exact_triangle_contact")

    # 3. Core gate scripts only (no legacy archive)
    prot_path = ROOT / "contracts/protected_paths.json"
    if prot_path.exists():
        prot = json.loads(prot_path.read_text(encoding="utf-8"))
        for sc in prot.get("protected_scripts", []):
            add_file(f"scripts/{sc}", f"scripts/{sc}")

    # 4. Curated midterm deliverables (PPT + 3 Word docs + M0-M7 frozen core certificates)
    mt = "results/midterm_sprint_20261120"
    add_file(f"{mt}/PPT_大创报告版_20261122/大创中期答辩_大创报告版.pptx",
             "deliverables/midterm/大创中期答辩_大创报告版.pptx")
    for doc_name in [
        "01_演讲稿与评委问答.docx",
        "02_项目通俗介绍_零基础版.docx",
        "03_技术说明书_打印备查版.docx",
        "验收报告.json",
    ]:
        add_file(f"{mt}/答辩配套文档_20261122/{doc_name}",
                 f"deliverables/midterm/配套文档/{doc_name}")
    if not github_clean:
        add_file(f"{mt}/架构师交接文档_20261122_中期收官与下半程接手.md",
                 "deliverables/midterm/架构师交接文档_中期收官.md")

    for mdir in [
        "M0_prereg_and_scope", "M1_real_to_exact_bridge", "M2_active_identifiability",
        "M3_independent_judge", "M4_benchmark_figures", "M5_strata_workbench_demo",
        "M6_midterm_defense_deck", "M7_final_pack_and_gates",
    ]:
        add_tree(f"{mt}/{mdir}", f"deliverables/midterm/{mdir}")

    # 5. Any finalized Phase-2 release folders placed under results/active_runs/release_*
    active_runs = ROOT / "results/active_runs"
    if active_runs.exists():
        for sub in sorted(active_runs.iterdir()):
            if sub.is_dir() and sub.name.startswith("release_"):
                add_tree(f"results/active_runs/{sub.name}", f"deliverables/phase2/{sub.name}")

    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Export clean standalone release bundle (<1 GB).")
    ap.add_argument("--dry-run", action="store_true", help="Print file count and total size without copying.")
    ap.add_argument("--github-clean", action="store_true",
                    help="Export GitHub-safe public bundle (excludes AGENTS.md/internal handover docs, enforces <45 MB per file).")
    ap.add_argument("--out", default="", help="Destination directory for the clean release bundle.")
    args = ap.parse_args()

    plan = collect_plan(github_clean=args.github_clean)
    total_bytes = sum(src.stat().st_size for src, _ in plan)
    print(f"Clean Release Plan: {len(plan):,} files, {total_bytes / 1024 / 1024:.2f} MB")

    if args.dry_run:
        # Top-level breakdown
        buckets: dict[str, list[int]] = {}
        for src, dst in plan:
            top = dst.split("/")[0]
            buckets.setdefault(top, [0, 0])
            buckets[top][0] += 1
            buckets[top][1] += src.stat().st_size
        for k, (cnt, b) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
            print(f"  {b / 1024 / 1024:7.2f} MB | {cnt:5,d} files | {k}/")
        return 0

    if not args.out:
        raise SystemExit("Please pass --out <target_dir> or --dry-run.")

    out_dir = pathlib.Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Target directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for src, rel_dst in plan:
        dst = out_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_entries.append({
            "path": rel_dst,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        })

    manifest = {
        "schema": "fractureflow.clean_release_manifest.v1",
        "exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_files": len(manifest_entries),
        "total_bytes": sum(e["bytes"] for e in manifest_entries),
        "entries": manifest_entries,
    }
    (out_dir / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"EXPORTED CLEAN RELEASE -> {out_dir} ({len(manifest_entries):,} files, {manifest['total_bytes']/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
