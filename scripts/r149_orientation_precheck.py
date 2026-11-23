#!/usr/bin/env python3
"""R149 T2: Orientation precheck for beachhead data.

Reads all in-disk data sources that carry orientation information,
computes dip/strike statistics, and judges whether each source has
meaningful multi-orientation structure (strike spread >= 30 deg or
number of detected sets >= 2).  Results are written as per-source
JSON files into results/r149_data_beachhead/precheck/.

Data sources covered:
  - GDR 1222 (30 CSVs, direct Strike/DipAngle columns)
  - GDR 1646 (1 CSV, direct Strike/Dip_Angle columns)
  - GDR 1750 (2 CSVs: Discrete + Stochastic, direct Strike/Dip_Angle)
  - DECOVALEX DFN (normal_vectors.dat, 3-component normals -> dip/strike)
  - RockBench LAS (pure XYZ point cloud, no orientation -> skip with verdict)
  - FORGE microseismic CSV (event catalog, no orientation -> skip with verdict)

Judgement criteria (per R155 task book):
  - PASS  if strike spread >= 30 deg  OR  n_sets >= 2
  - FAIL  if isotropic (no dominant direction) or single-orientation only
"""

import csv
import json
import math
import os
import struct
import zipfile
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join("results", "r149_data_beachhead")
EXTRACTED = os.path.join(BASE, "extracted")
PRECHECK_DIR = os.path.join(BASE, "precheck")

PYTHON = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circ_mean(angles_deg, half_circle=False):
    """Circular mean of angles in degrees.

    If half_circle=True, data is in [0, 180) (bidirectional / folded normals).
    Double angles to [0, 360), compute circular mean, halve the result.
    """
    if not angles_deg:
        return None
    if half_circle:
        angles = [a * 2 % 360 for a in angles_deg]
    else:
        angles = [a % 360 for a in angles_deg]
    s_total = sum(math.sin(math.radians(a)) for a in angles)
    c_total = sum(math.cos(math.radians(a)) for a in angles)
    mean_rad = math.atan2(s_total, c_total)
    mean_deg = math.degrees(mean_rad) % 360
    if half_circle:
        mean_deg = mean_deg / 2
    return mean_deg


def circ_std(angles_deg, half_circle=False):
    """Circular standard deviation in degrees.

    If half_circle=True, data is bidirectional [0, 180).
    Double angles, compute, the result is in [0, 90).
    """
    if not angles_deg:
        return None
    if half_circle:
        angles = [a * 2 % 360 for a in angles_deg]
    else:
        angles = [a % 360 for a in angles_deg]
    s_total = sum(math.sin(math.radians(a)) for a in angles)
    c_total = sum(math.cos(math.radians(a)) for a in angles)
    n = len(angles)
    R = math.hypot(s_total, c_total) / n
    if R < 1e-12:
        return 90.0
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


def strike_spread(angles_deg, half_circle=False):
    """Angular spread: 360 (or 180) minus the largest empty sector gap.

    For full-circle data (half_circle=False): spread = 360 - max_gap.
    For half-circle data (half_circle=True): spread = 180 - max_gap.
    Capped at 180 (max possible pairwise distance on a circle).
    """
    if not angles_deg or len(angles_deg) < 2:
        return 0.0
    period = 180 if half_circle else 360
    sorted_ang = sorted(set(a % period for a in angles_deg))
    if len(sorted_ang) == 1:
        return 0.0
    max_gap = 0.0
    for i in range(len(sorted_ang)):
        j = (i + 1) % len(sorted_ang)
        gap = (sorted_ang[j] - sorted_ang[i]) % period
        max_gap = max(max_gap, gap)
    spread = period - max_gap
    return min(spread, period / 2)


def detect_sets_strikes(strikes_deg, n_bins=36, min_peak_frac=0.15, half_circle=False):
    """Detect number of orientation sets from strike angles.
    Uses histogram peak detection: bins with count >= min_peak_frac * max_bin
    that are local maxima are counted as sets.

    If half_circle=True, histogram is over [0, 180) with n_bins/2 bins.
    """
    if not strikes_deg:
        return 0, []
    period = 180 if half_circle else 360
    actual_bins = n_bins // 2 if half_circle else n_bins
    hist = []
    for i in range(actual_bins):
        lo = i * (period / actual_bins)
        hi = (i + 1) * (period / actual_bins)
        count = sum(1 for a in strikes_deg if lo <= (a % period) < hi)
        hist.append(count)
    max_count = max(hist) if hist else 0
    if max_count == 0:
        return 0, []
    threshold = min_peak_frac * max_count
    peaks = []
    for i in range(actual_bins):
        if hist[i] < threshold:
            continue
        prev = hist[(i - 1) % actual_bins]
        nxt = hist[(i + 1) % actual_bins]
        if hist[i] >= prev and hist[i] >= nxt:
            center = (i + 0.5) * (period / actual_bins)
            peaks.append((center, hist[i]))
    # Merge peaks within 20 deg
    merged = []
    for p in peaks:
        if merged and abs(p[0] - merged[-1][0]) < 20:
            if p[1] > merged[-1][1]:
                merged[-1] = p
        else:
            merged.append(p)
    return len(merged), [p[0] for p in merged]


def normal_to_dip_strike(nx, ny, nz):
    """Convert a unit normal (nx,ny,nz) to (dip_deg, strike_deg).

    Normals are bidirectional (n and -n represent the same plane).
    We fold strike to [0, 180) to handle the +/-n ambiguity.

    dip = arccos(|nz|)  (angle between normal and vertical)
    strike = atan2(nx, -ny) mod 180  (folded to half-circle)
    """
    dip = math.degrees(math.acos(min(abs(nz), 1.0)))
    strike = math.degrees(math.atan2(nx, -ny)) % 180
    return dip, strike


def write_json(name, data):
    path = os.path.join(PRECHECK_DIR, name + ".json")
    os.makedirs(PRECHECK_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  -> {path}")
    return data


def judge(strikes, dips, source_name, half_circle=False):
    """Produce verdict dict from strike/dip lists.

    If half_circle=True, strikes are in [0, 180) (folded bidirectional normals).
    """
    n = len(strikes)
    spread = strike_spread(strikes, half_circle=half_circle) if strikes else 0.0
    n_sets, set_centers = detect_sets_strikes(strikes, half_circle=half_circle) if strikes else (0, [])
    cmean = circ_mean(strikes, half_circle=half_circle) if strikes else None
    cstd = circ_std(strikes, half_circle=half_circle) if strikes else None
    dip_median = sorted(dips)[len(dips)//2] if dips else None
    dip_min = min(dips) if dips else None
    dip_max = max(dips) if dips else None

    if n == 0:
        verdict = "NO_ORIENTATION_DATA"
    elif n_sets >= 2:
        # Multiple orientation sets detected — PASS even if circ_std is high
        # (high circ_std with detectable peaks = well-separated sets, not isotropic)
        verdict = "PASS"
    elif cstd is not None and cstd > 75:
        verdict = "FAIL_ISOTROPIC"
    elif spread < 30 and n_sets < 2:
        verdict = "FAIL_SINGLE_ORIENTATION"
    else:
        verdict = "PASS"

    period = 180 if half_circle else 360
    return {
        "source": source_name,
        "n_fractures": n,
        "half_circle": half_circle,
        "strike_circular_mean_deg": round(cmean, 2) if cmean is not None else None,
        "strike_circular_std_deg": round(cstd, 2) if cstd is not None else None,
        "strike_spread_deg": round(spread, 2),
        "n_detected_sets": n_sets,
        "set_centers_deg": [round(c, 1) for c in set_centers],
        "dip_median_deg": round(dip_median, 1) if dip_median is not None else None,
        "dip_range_deg": [round(dip_min, 1), round(dip_max, 1)] if dip_min is not None else None,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# GDR 1222: 30 CSVs with Strike, DipAngle columns
# ---------------------------------------------------------------------------
def precheck_gdr1222():
    print("[T2] GDR 1222 (30 CSVs)")
    dir_path = os.path.join(EXTRACTED, "gdr1222")
    if not os.path.isdir(dir_path):
        return write_json("gdr1222", {"source": "gdr1222", "verdict": "MISSING_DIR"})
    strikes, dips = [], []
    files_used = []
    for i in range(1, 31):
        fname = f"Filtered_800m_{i}.csv"
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        files_used.append(fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = float(row.get("Strike[deg]", row.get("Strike", "")))
                    d = float(row.get("DipAngle[deg]", row.get("DipAngle", "")))
                    strikes.append(s)
                    dips.append(d)
                except (ValueError, TypeError):
                    continue
    result = judge(strikes, dips, "gdr1222")
    result["files_read"] = files_used
    return write_json("gdr1222", result)


# ---------------------------------------------------------------------------
# GDR 1646: 1 CSV with Strike[deg], Dip_Angle[deg] columns
# ---------------------------------------------------------------------------
def precheck_gdr1646():
    print("[T2] GDR 1646 (DFN CSV)")
    fpath = os.path.join(
        EXTRACTED, "gdr1646",
        "2024 July Updated MEQ DFN", "Local Coordinates",
        "New MEQ DFN v2 July 2024.csv",
    )
    if not os.path.isfile(fpath):
        return write_json("gdr1646", {"source": "gdr1646", "verdict": "MISSING_FILE"})
    strikes, dips = [], []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row.get("Strike[deg]", row.get("Strike", "")))
                d = float(row.get("Dip_Angle[deg]", row.get("Dip_Angle", "")))
                strikes.append(s)
                dips.append(d)
            except (ValueError, TypeError):
                continue
    result = judge(strikes, dips, "gdr1646")
    result["file"] = os.path.basename(fpath)
    return write_json("gdr1646", result)


# ---------------------------------------------------------------------------
# GDR 1750: 2 CSVs (Discrete + Stochastic) with Strike[deg], Dip_Angle[deg]
# ---------------------------------------------------------------------------
def precheck_gdr1750():
    print("[T2] GDR 1750 (Discrete + Stochastic)")
    base = os.path.join(
        EXTRACTED, "gdr1750", "Global Coordinates Feet",
        "FORGE 2025 v1 DFN Fracture Files",
    )
    files = {
        "discrete": os.path.join(base, "DFN 2025 v1 Discrete Fractures global ft.csv"),
        "stochastic": os.path.join(base, "DFN 2025 v1 Stochastic Fractures global ft.csv"),
    }
    all_strikes, all_dips = [], []
    per_file = {}
    for label, fpath in files.items():
        if not os.path.isfile(fpath):
            per_file[label] = {"verdict": "MISSING_FILE"}
            continue
        strikes, dips = [], []
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = float(row.get("Strike[deg]", row.get("Strike", "")))
                    d = float(row.get("Dip_Angle[deg]", row.get("Dip_Angle", "")))
                    strikes.append(s)
                    dips.append(d)
                except (ValueError, TypeError):
                    continue
        per_file[label] = judge(strikes, dips, f"gdr1750_{label}")
        per_file[label]["file"] = os.path.basename(fpath)
        all_strikes.extend(strikes)
        all_dips.extend(dips)
    combined = judge(all_strikes, all_dips, "gdr1750_combined")
    combined["per_file"] = per_file
    return write_json("gdr1750", combined)


# ---------------------------------------------------------------------------
# DECOVALEX DFN: normal_vectors.dat inside zip
# ---------------------------------------------------------------------------
def precheck_decovalex():
    print("[T2] DECOVALEX DFN (normal_vectors.dat)")
    zip_path = os.path.join(BASE, "raw", "zen14873207_decovalex_d23_4frac_dfn.zip")
    if not os.path.isfile(zip_path):
        return write_json("decovalex_dfn", {"source": "decovalex_dfn", "verdict": "MISSING_ZIP"})

    results = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find all normal_vectors.dat files
        nv_files = [n for n in zf.namelist() if n.endswith("normal_vectors.dat") and "__MACOSX" not in n]
        for nvf in nv_files:
            data = zf.read(nvf)
            lines = [l for l in data.decode("ascii", errors="replace").strip().split("\n") if l.strip()]
            normals = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        nx, ny, nz = float(parts[0]), float(parts[1]), float(parts[2])
                        # Normalize
                        mag = math.sqrt(nx*nx + ny*ny + nz*nz)
                        if mag > 1e-12:
                            normals.append((nx/mag, ny/mag, nz/mag))
                    except ValueError:
                        continue
            strikes, dips = [], []
            for nx, ny, nz in normals:
                dip, strike = normal_to_dip_strike(nx, ny, nz)
                strikes.append(strike)
                dips.append(dip)
            label = nvf.replace("/", "_").replace("\\", "_").replace(".dat", "")
            results[label] = judge(strikes, dips, f"decovalex_{label}", half_circle=True)
            results[label]["n_normals"] = len(normals)

    # Combined verdict for the 4fracplus (main) file
    main_key = None
    for k in results:
        if "4fracplus" in k:
            main_key = k
            break
    if main_key:
        results["_main_verdict"] = results[main_key]["verdict"]
    else:
        results["_main_verdict"] = "UNKNOWN"

    return write_json("decovalex_dfn", results)


# ---------------------------------------------------------------------------
# RockBench LAS: pure XYZ point cloud, no orientation columns
# ---------------------------------------------------------------------------
def precheck_rockbench():
    print("[T2] RockBench LAS (point cloud, no orientation)")
    las_path = os.path.join(EXTRACTED, "rockbench", "realslope.las")
    if not os.path.isfile(las_path):
        return write_json("rockbench", {"source": "rockbench", "verdict": "MISSING_FILE"})
    # We already know from header parsing: LAS 1.4, PF6, 30 bytes/point,
    # 1 VLR (projection only), 0 extra dimensions -> no dip/strike.
    result = {
        "source": "rockbench",
        "file": "realslope.las",
        "las_version": "1.4",
        "point_format": 6,
        "point_record_length": 30,
        "n_points": 1236573,
        "extra_dimensions": 0,
        "orientation_columns": "none",
        "verdict": "NO_ORIENTATION_DATA",
        "note": (
            "Pure XYZ point cloud (LAS 1.4 PF6, 0 extra dimensions). "
            "Orientation extraction would require RANSAC plane segmentation "
            "(cf. label_free_dirs pipeline), not direct column read. "
            "Excluded from T2 orientation precheck per task book scope."
        ),
    }
    return write_json("rockbench", result)


# ---------------------------------------------------------------------------
# FORGE microseismic CSV: event catalog, no orientation columns
# ---------------------------------------------------------------------------
def precheck_forge_microseismic():
    print("[T2] FORGE microseismic CSV (event catalog)")
    csv_path = os.path.join(BASE, "raw", "zen12817881_microseismic_FORGE2024.csv")
    if not os.path.isfile(csv_path):
        return write_json("forge_microseismic", {"source": "forge_microseismic", "verdict": "MISSING_FILE"})
    result = {
        "source": "forge_microseismic",
        "file": os.path.basename(csv_path),
        "orientation_columns": "none",
        "columns": ["time", "ID", "lat", "lon", "depth", "..."],
        "verdict": "NO_ORIENTATION_DATA",
        "note": (
            "Microseismic event catalog (time/ID/lat/lon/depth). "
            "No fracture orientation columns. Excluded from T2."
        ),
    }
    return write_json("forge_microseismic", result)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(results):
    """Write a combined summary JSON."""
    summary = {
        "task": "R149_T2_orientation_precheck",
        "n_sources": len(results),
        "sources": {},
    }
    pass_count = 0
    fail_count = 0
    skip_count = 0
    for name, data in results.items():
        if isinstance(data, dict):
            v = data.get("verdict", data.get("_main_verdict", "UNKNOWN"))
        else:
            v = "UNKNOWN"
        summary["sources"][name] = v
        if v == "PASS":
            pass_count += 1
        elif v.startswith("FAIL"):
            fail_count += 1
        else:
            skip_count += 1
    summary["verdict_counts"] = {"PASS": pass_count, "FAIL": fail_count, "SKIP": skip_count}
    path = os.path.join(PRECHECK_DIR, "_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n[T2] Summary: PASS={pass_count} FAIL={fail_count} SKIP={skip_count}")
    print(f"  -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PRECHECK_DIR, exist_ok=True)
    print("=" * 60)
    print("R149 T2: Orientation Precheck")
    print("=" * 60)

    results = {}
    results["gdr1222"] = precheck_gdr1222()
    results["gdr1646"] = precheck_gdr1646()
    results["gdr1750"] = precheck_gdr1750()
    results["decovalex_dfn"] = precheck_decovalex()
    results["rockbench"] = precheck_rockbench()
    results["forge_microseismic"] = precheck_forge_microseismic()

    write_summary(results)
    print("\n[T2] Done.")


if __name__ == "__main__":
    main()
