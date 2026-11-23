# -*- coding: utf-8 -*-
"""Evaluation Platform CLI.

Usage:
  python -m fractureflow.platform verify               # verify all results/*.json
  python -m fractureflow.platform ledger                # show ledger status
  python -m fractureflow.platform leaderboard           # aggregate B* baseline results
  python -m fractureflow.platform constants            # show frozen anchors & metrics
"""
import argparse
import json
import os
import sys

# Ensure src/ is in path when run as `python -m fractureflow.platform`
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJ, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fractureflow.platform import scoring, ledger, verify, leaderboard


def cmd_verify(args):
    """Verify all result JSONs under results/."""
    results = verify.scan_results_dir()
    ok = sum(1 for r in results if r["ok"])
    n_match = sum(1 for r in results if r.get("match") is True)
    n_mismatch = sum(1 for r in results if r.get("match") is False)
    n_missing = sum(1 for r in results if r.get("match") is None)
    print(f"Scanned: {len(results)} files")
    print(f"  Valid JSON: {ok}")
    print(f"  SHA match: {n_match}")
    print(f"  SHA mismatch: {n_mismatch}")
    print(f"  No SHA field: {n_missing}")
    if args.verbose:
        for r in results:
            if r.get("match") is False:
                print(f"  MISMATCH: {r['path']}")
    return 0 if n_mismatch == 0 else 1


def cmd_ledger(args):
    """Show ledger status."""
    v = ledger.verify_ledger()
    print(json.dumps(v, allow_nan=False, indent=2, ensure_ascii=False))
    return 0


def cmd_leaderboard(args):
    """Aggregate baseline results into comparison table."""
    import glob
    patterns = args.patterns or [
        "results/r110_b1/scores/*.json",
        "results/r110_b2_scores/*.json",
        "results/r110_b3_scores/*.json",
        "results/r110_b4_scores/*.json",
        "results/r110_b5_scores/*.json",
    ]
    files = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    if not files:
        print("No result files found.", file=sys.stderr)
        return 1
    agg = leaderboard.aggregate_methods(files)
    print(leaderboard.compare_table(agg))
    return 0


def cmd_constants(args):
    """Print frozen constants."""
    print("=== FROZEN ANCHORS (B-1 caliber) ===")
    for k, v in scoring.FROZEN_ANCHORS.items():
        print(f"  {k}: {v}")
    print(f"\n=== FROZEN METRICS ({len(scoring.FROZEN_METRIC_NAMES)}) ===")
    for m in scoring.FROZEN_METRIC_NAMES:
        print(f"  {m}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="FractureFlow Evaluation Platform CLI")
    sub = p.add_subparsers(dest="cmd")
    vp = sub.add_parser("verify", help="Verify result JSONs")
    vp.add_argument("-v", "--verbose", action="store_true")
    vp.set_defaults(func=cmd_verify)
    sub.add_parser("ledger", help="Show ledger status").set_defaults(func=cmd_ledger)
    lp = sub.add_parser("leaderboard", help="Aggregate baseline results")
    lp.add_argument("patterns", nargs="*", help="Glob patterns for result JSONs")
    lp.set_defaults(func=cmd_leaderboard)
    sub.add_parser("constants", help="Print frozen constants").set_defaults(func=cmd_constants)
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
