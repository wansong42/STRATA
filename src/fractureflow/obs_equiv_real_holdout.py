# -*- coding: utf-8 -*-
"""REAL_CENSORING_HOLDOUT contract, validator and local-asset audit.

This module is deliberately separate from the synthetic obs_equiv pilot.  It
never turns borehole expert picks into a full 3D network, never derives
two-point connectivity labels, and never mixes EXACT_SYNTHETIC rows into a
real holdout.

The current real-data candidate is a whole-well cross-well observation
holdout from FORGE Milford: 16A(78)-32 + 16B(78)-32 observed, 58-32 held out.
Truth level is EXPERT_BOREHOLE_PICKS; the task family is held-out borehole
pick observation, not the frozen TWO_POINT_CONNECTIVITY query.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shlex
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_MANIFEST = "obs_equiv.real_censoring_holdout.v1"
SCHEMA_AUDIT = "obs_equiv.real_censoring_holdout.audit.v1"
SCHEMA_VALIDATION = "obs_equiv.real_censoring_holdout.validation.v1"
SCHEMA_METRICS = "obs_equiv.real_censoring_holdout.per_well_metrics.v1"
WATERMARK = "DATA_AUDIT / REAL_CENSORING_HOLDOUT / PICKS_ONLY"

GROUP_REAL_CENSORING_HOLDOUT = "REAL_CENSORING_HOLDOUT"
TRUTH_EXPERT_BOREHOLE_PICKS = "EXPERT_BOREHOLE_PICKS"
TRUTH_HELD_OUT_OBSERVATION = "HELD_OUT_OBSERVATION"
REAL_TRUTH_LEVELS = (TRUTH_EXPERT_BOREHOLE_PICKS, TRUTH_HELD_OUT_OBSERVATION)
FORBIDDEN_TRUTH_LEVELS = ("EXACT_SYNTHETIC",)

TASK_FAMILY = "HELD_OUT_BOREHOLE_PICK_OBSERVATION"
ALLOWED_CLAIM = "CROSS_WELL_HELD_OUT_OBSERVATION_ONLY"
UNDIRECTED_CLAIM = "UNDIRECTED_AXIAL_OBSERVATION_ONLY"
ORIENTATION_CONVENTION = "UNDIRECTED_STRIKE_DIP"
AXIS_SIGN_CONVENTION = "CANONICAL_Z_NONNEGATIVE"
SAMPLE_UNIT = "WELL"
SITE_SCOPE = "FORGE_MILFORD_SINGLE_SITE_WHOLE_WELL_HOLDOUT"
METRIC_FAMILY = "UNDIRECTED_AXIAL_ACUTE_ANGLE_DEG"
# RV2-F-001: machine-readable metric semantics.
# The nearest-observed-pick-axis metric selects the nearest observed axis *with
# the held-out target axis itself*; it is a target-dependent descriptive
# distance, not a deployable predictor and not a prediction error.
METRIC_ID_OBSERVED_MEAN = "OBSERVED_EQUAL_WELL_AXIAL_MEAN"
METRIC_ID_NEAREST_AXIS = "TARGET_DEPENDENT_NEAREST_AXIS_DISTANCE"
METRIC_ROLE_OBSERVED_PREDICTOR = "DEPLOYABLE_OBSERVED_ONLY_PREDICTOR"
METRIC_ROLE_TARGET_DEPENDENT = "TARGET_DEPENDENT_DESCRIPTIVE_DISTANCE"
PREDICTOR_SEMANTICS_NOT_DEPLOYABLE = "NOT_A_DEPLOYABLE_PREDICTOR"
FORBIDDEN_CLAIM_TOKENS = (
    "REAL_GENERALIZATION",
    "FULL_3D",
    "CONNECTIVITY_TRUTH",
    "CONNECTIVITY_ACCURACY",
    "DIRECTED_DIP_DIRECTION_ERROR",
    "3D_TRUTH",
    "EXACT_SYNTHETIC",
)
REQUIRED_FORBIDDEN_CLAIMS = (
    "不得冒充完整三维真值",
    "不得声明真实泛化",
    "不得与 EXACT_SYNTHETIC 混组",
    "不得生成连通性真值",
    "不得把无向轴写成有向 dip direction",
    "不得声称 directed dip-direction 全覆盖",
    "不得输出 connectivity accuracy / 三维真值 / 真实泛化",
)

REQUIRED_MANIFEST_FIELDS = (
    "schema",
    "group_type",
    "truth_level",
    "task_family",
    "site_id",
    "site_scope",
    "sample_unit",
    "orientation_convention",
    "metric_artifact",
    "fixture",
    "real_data",
    "observed_units",
    "held_out_units",
    "source_assets",
    "license",
    "license_status",
    "license_evidence",
    "observation_protocol",
    "evaluation_time",
    "allowed_claims",
    "forbidden_claims",
    "counts",
)
ALLOWED_MANIFEST_FIELDS = REQUIRED_MANIFEST_FIELDS + ("metric_family", "claim_scope", "validation", "notes")
REQUIRED_UNIT_FIELDS = (
    "unit_id",
    "unit_type",
    "n_observations",
    "source_path",
    "sha256",
    "depth_range_ft",
    "types",
    "fracture_filter",
    "orientation_convention",
    "n_orientation_axes",
    "directed_dip_direction_n",
    "directed_dip_direction_total",
)
UNIT_TYPES = ("well", "site", "core")
REQUIRED_PROTOCOL_FIELDS = (
    "modality",
    "coordinate_frame",
    "truth_level_rationale",
    "enumeration",
    "depth_reference",
    "no_3d_geometry_claim",
    "no_connectivity_label",
    "orientation_convention",
    "axis_sign_convention",
    "metric_family",
    "directed_dip_direction_metric",
    "connectivity_metric",
    "three_d_truth",
    "real_generalization",
)
FORBIDDEN_KEYS = frozenset({
    "query_labels",
    "reference_posterior",
    "prior_weights",
    "truth_sets",
    "true_normal",
    "true_center",
    "full_network",
    "ground_truth_network",
    "hidden_truth",
    "hidden_objects",
    "oracle_label",
    "connectivity_label",
    "two_point_connectivity",
    "full_3d_truth",
    "directed_dip_direction_error",
    "connectivity_accuracy",
    "3d_truth_accuracy",
    "real_generalization_claim",
    "scene_ids",
    "disks",
    "radius",
    "exact_synthetic_truth",
})
FORBIDDEN_KEY_PREFIXES = ("hidden_", "oracle_", "ground_truth")
FORBIDDEN_KEY_SUFFIXES = ("_seed",)
FORBIDDEN_METRIC_KEYS = frozenset({
    "directed_dip_direction_error",
    "connectivity_accuracy",
    "3d_truth_accuracy",
    "full_3d_truth",
    "real_generalization_claim",
})


class RealHoldoutError(ValueError):
    """Malformed real-holdout manifest or unreadable local asset."""


class BlockedByData(RuntimeError):
    """Real manifest cannot be built from the audited assets."""

    def __init__(self, blockers: Sequence[Mapping[str, Any]]):
        self.blockers = [dict(item) for item in blockers]
        super().__init__("BLOCKED_BY_DATA: " + "; ".join(str(item.get("code")) for item in self.blockers))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_exclusive(path: str | Path, document: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(document) + "\n")
    return target


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == -9999.0:
        return None
    return number


def _sha_ok(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _classify_pick(type_value: Any) -> tuple[bool, bool, bool]:
    text = "" if type_value is None else str(type_value)
    lowered = text.lower()
    is_fracture = "fracture" in lowered
    is_induced = "induced" in lowered
    return is_fracture, is_induced, is_fracture and not is_induced


def _summarize_picks(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Any]:
    types = Counter()
    natural_depths = []
    natural_dips = []
    natural_azimuths = []
    n_rows = 0
    n_fracture = 0
    n_induced = 0
    n_natural = 0
    n_non_fracture = 0
    n_malformed = 0
    for row in rows:
        n_rows += 1
        type_value = row.get("type")
        types[str(type_value)] += 1
        if row.get("is_fracture") is None or row.get("is_induced") is None:
            n_malformed += 1
        is_fracture, is_induced, is_natural = _classify_pick(type_value)
        if is_fracture:
            n_fracture += 1
        else:
            n_non_fracture += 1
        if is_induced:
            n_induced += 1
        if is_natural:
            n_natural += 1
            depth = _finite_number(row.get("depth"))
            if depth is not None:
                natural_depths.append(depth)
            dip = _finite_number(row.get("dip"))
            if dip is not None:
                natural_dips.append(dip)
            azimuth = _finite_number(row.get("azimuth"))
            if azimuth is not None:
                natural_azimuths.append(azimuth)
    return {
        "probe": "PARSED",
        "source": source,
        "n_rows": n_rows,
        "n_fracture": n_fracture,
        "n_natural": n_natural,
        "n_induced": n_induced,
        "n_non_fracture": n_non_fracture,
        "n_malformed": n_malformed,
        "types": dict(sorted(types.items(), key=lambda item: (-item[1], item[0]))),
        "depth_md_ft": [min(natural_depths), max(natural_depths)] if natural_depths else None,
        "dip_deg": [min(natural_dips), max(natural_dips)] if natural_dips else None,
        "azimuth_deg": [min(natural_azimuths), max(natural_azimuths)] if natural_azimuths else None,
    }


def _curve_names(lines: Sequence[str]) -> list[str]:
    names = []
    in_curves = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("~curve"):
            in_curves = True
            continue
        if in_curves and stripped.startswith("~"):
            break
        if in_curves and stripped and not stripped.startswith("#"):
            name = stripped.split(".", 1)[0].strip()
            if name:
                names.append(name)
    return names


def probe_las_picks(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    curves = _curve_names(lines)
    if not curves:
        raise RealHoldoutError("LAS curve block not found: %s" % target)
    ascii_start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("~ascii"):
            ascii_start = index + 1
            break
    if ascii_start is None:
        raise RealHoldoutError("LAS ~Ascii block not found: %s" % target)
    rows = []
    n_malformed = 0
    for line in lines[ascii_start:]:
        if not line.strip():
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            n_malformed += 1
            continue
        if len(parts) < len(curves):
            n_malformed += 1
            continue
        row = dict(zip(curves, parts))
        normalized = {str(key).strip().upper(): value for key, value in row.items()}
        type_value = normalized.get("TYPE")
        if type_value is None:
            type_value = parts[-1]
        depth = None
        for key in ("MD", "TDEP", "DEPTH", "DEPT"):
            if key in normalized:
                depth = _finite_number(normalized[key])
                break
        dip = None
        for key in ("DIP_TRU", "DIP", "DIP_ANGLE"):
            if key in normalized:
                dip = _finite_number(normalized[key])
                break
        azimuth = None
        for key in ("AZIMUTH", "DIP_DIRECTION", "DIPDIR"):
            if key in normalized:
                azimuth = _finite_number(normalized[key])
                break
        is_fracture, is_induced, _ = _classify_pick(type_value)
        rows.append({
            "type": type_value,
            "depth": depth,
            "dip": dip,
            "azimuth": azimuth,
            "is_fracture": is_fracture,
            "is_induced": is_induced,
        })
    summary = _summarize_picks(rows, str(target))
    summary["probe"] = "LAS_PICKS"
    summary["curve_names"] = curves
    summary["n_malformed"] = summary.get("n_malformed", 0) + n_malformed
    summary["bytes"] = target.stat().st_size
    summary["sha256"] = file_sha256(target)
    return summary


def _colnum(reference: str | None) -> int | None:
    match = re.match(r"([A-Z]+)", reference or "")
    if not match:
        return None
    number = 0
    for char in match.group(1):
        number = number * 26 + (ord(char) - 64)
    return number


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str | None:
    value = cell.find("m:v", _NS)
    text = None if value is None else value.text
    cell_type = cell.attrib.get("t")
    if cell_type == "s" and text is not None:
        return shared[int(text)]
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(_TAG_T))
    return text


def _xlsx_sheets_from_bytes(data: bytes) -> list[tuple[str, list[dict[int, str | None]]]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared: list[str] = []
        names = archive.namelist()
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _NS):
                shared.append("".join(node.text or "" for node in item.iter(_TAG_T)))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = [
            (sheet.attrib.get("name"), sheet.attrib.get(_REL_ID))
            for sheet in workbook.find("m:sheets", _NS)
        ]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        result = []
        for name, rel_id in sheets:
            target = relmap.get(rel_id)
            if not target:
                continue
            path = "xl/" + target.lstrip("/")
            sheet_root = ET.fromstring(archive.read(path))
            rows = []
            for row in sheet_root.findall(".//m:sheetData/m:row", _NS):
                values = {}
                for cell in row.findall("m:c", _NS):
                    column = _colnum(cell.attrib.get("r"))
                    if column is not None:
                        values[column] = _cell_value(cell, shared)
                rows.append(values)
            result.append((str(name), rows))
        return result


def _xlsx_sheets(path: str | Path) -> list[tuple[str, list[dict[int, str | None]]]]:
    return _xlsx_sheets_from_bytes(Path(path).read_bytes())


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_TAG_T = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _header_map(header: Mapping[int, str | None]) -> dict[str, int]:
    result = {}
    for column, value in header.items():
        if value is None:
            continue
        name = str(value).strip()
        if name:
            result[name] = column
    return result


def _rows_from_sheet(
    rows: Sequence[Mapping[int, str | None]],
    *,
    required: Sequence[str],
) -> tuple[list[Mapping[int, str | None]], dict[str, int]]:
    for index, row in enumerate(rows):
        mapping = _header_map(row)
        if all(name in mapping for name in required):
            return list(rows[index + 1:]), mapping
    raise RealHoldoutError("xlsx header not found; required=%r" % (list(required),))


def probe_xlsx_fmi(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    sheets = _xlsx_sheets(target)
    selected = None
    header_map = None
    data_rows = None
    for _name, rows in sheets:
        try:
            candidate_rows, candidate_map = _rows_from_sheet(rows, required=("MD", "Type"))
        except RealHoldoutError:
            continue
        selected = _name
        header_map = candidate_map
        data_rows = candidate_rows
        break
    if data_rows is None or header_map is None:
        raise RealHoldoutError("FMI xlsx sheet with MD/Type not found: %s" % target)
    type_col = header_map["Type"]
    md_col = header_map["MD"]
    dip_col = header_map.get("Dip")
    azimuth_col = header_map.get("Dip Direction") or header_map.get("Azimuth")
    rows = []
    for row in data_rows:
        type_value = row.get(type_col)
        md = _finite_number(row.get(md_col))
        if md is None or type_value is None:
            continue
        dip = _finite_number(row.get(dip_col)) if dip_col is not None else None
        azimuth = _finite_number(row.get(azimuth_col)) if azimuth_col is not None else None
        is_fracture, is_induced, _ = _classify_pick(type_value)
        rows.append({
            "type": type_value,
            "depth": md,
            "dip": dip,
            "azimuth": azimuth,
            "is_fracture": is_fracture,
            "is_induced": is_induced,
        })
    summary = _summarize_picks(rows, str(target))
    summary["probe"] = "XLSX_FMI_PICKS"
    summary["sheet"] = selected
    summary["bytes"] = target.stat().st_size
    summary["sha256"] = file_sha256(target)
    return summary


def probe_core_zip(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with zipfile.ZipFile(target) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        rows = []
        intervals = []
        for member in members:
            sheets = _xlsx_sheets_from_bytes(archive.read(member))
            matched = False
            for name, sheet_rows in sheets:
                if "as measured" not in name.lower():
                    continue
                try:
                    data_rows, header_map = _rows_from_sheet(sheet_rows, required=("MD RKB ft", "Type"))
                except RealHoldoutError:
                    continue
                matched = True
                md_col = header_map["MD RKB ft"]
                type_col = header_map["Type"]
                depths = []
                for row in data_rows:
                    md = _finite_number(row.get(md_col))
                    type_value = row.get(type_col)
                    if md is None or type_value is None:
                        continue
                    depths.append(md)
                    is_fracture, is_induced, _ = _classify_pick(type_value)
                    rows.append({
                        "type": type_value,
                        "depth": md,
                        "dip": None,
                        "azimuth": None,
                        "is_fracture": is_fracture,
                        "is_induced": is_induced,
                    })
                intervals.append({
                    "member": member,
                    "sheet": name,
                    "n_rows": len(depths),
                    "depth_md_ft": [min(depths), max(depths)] if depths else None,
                })
            if not matched:
                intervals.append({"member": member, "sheet": None, "n_rows": 0, "depth_md_ft": None})
    summary = _summarize_picks(rows, str(target))
    summary["probe"] = "CORE_ZIP_XLSX"
    summary["n_intervals"] = len(members)
    summary["intervals"] = intervals
    summary["bytes"] = target.stat().st_size
    summary["sha256"] = file_sha256(target)
    return summary


def _scan_forbidden_keys(value: Any, path: str = "$") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            if (
                lowered in FORBIDDEN_KEYS
                or any(lowered.startswith(prefix) for prefix in FORBIDDEN_KEY_PREFIXES)
                or any(lowered.endswith(suffix) for suffix in FORBIDDEN_KEY_SUFFIXES)
            ):
                hits.append({"path": "%s.%s" % (path, key)})
            hits.extend(_scan_forbidden_keys(child, "%s.%s" % (path, key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            hits.extend(_scan_forbidden_keys(child, "%s[%d]" % (path, index)))
    return hits


def _check(result: dict[str, Any], name: str, ok: bool, code: str, detail: str) -> None:
    result["checks"][name] = bool(ok)
    result["n_checks"] += 1
    if not ok:
        result["violations"].append({"code": code, "detail": detail})


def validate_real_holdout_manifest(
    manifest: Mapping[str, Any],
    *,
    require_real: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA_VALIDATION,
        "pass": False,
        "checks": {},
        "n_checks": 0,
        "violations": [],
        "truth_level": None,
        "fixture": None,
        "real_data": None,
    }
    if not isinstance(manifest, Mapping):
        result["violations"].append({"code": "MANIFEST_NOT_OBJECT", "detail": type(manifest).__name__})
        result["enumerated_checks"] = result["n_checks"]
        return result

    observed = manifest.get("observed_units")
    held_out = manifest.get("held_out_units")
    if not isinstance(observed, list) or not observed:
        raise SystemExit("EMPTY_OBSERVED_UNIT_SET: real holdout requires observed units")
    if not isinstance(held_out, list) or not held_out:
        raise SystemExit("EMPTY_HELD_OUT_UNIT_SET: real holdout requires held-out units")

    for field in REQUIRED_MANIFEST_FIELDS:
        _check(result, "field:%s" % field, field in manifest, "FIELD_MISSING", "missing required field %r" % field)
    unknown = sorted(set(manifest) - set(ALLOWED_MANIFEST_FIELDS))
    _check(result, "no_unknown_fields", not unknown, "UNKNOWN_FIELD", "unknown fields: %r" % unknown)

    _check(result, "schema", manifest.get("schema") == SCHEMA_MANIFEST, "SCHEMA_MISMATCH", "schema must be %r" % SCHEMA_MANIFEST)
    _check(
        result,
        "group_type",
        manifest.get("group_type") == GROUP_REAL_CENSORING_HOLDOUT,
        "GROUP_TYPE",
        "group_type must be %r" % GROUP_REAL_CENSORING_HOLDOUT,
    )
    truth_level = manifest.get("truth_level")
    result["truth_level"] = truth_level
    _check(
        result,
        "truth_level_real",
        truth_level in REAL_TRUTH_LEVELS,
        "TRUTH_LEVEL_FORBIDDEN",
        "truth_level must be one of %r, got %r" % (list(REAL_TRUTH_LEVELS), truth_level),
    )
    _check(
        result,
        "task_family",
        manifest.get("task_family") == TASK_FAMILY,
        "TASK_FAMILY",
        "task_family must be %r" % TASK_FAMILY,
    )
    _check(
        result,
        "site_scope",
        isinstance(manifest.get("site_scope"), str) and bool(manifest.get("site_scope")),
        "SITE_SCOPE_MISSING",
        "site_scope must be a non-empty string",
    )
    _check(
        result,
        "sample_unit",
        manifest.get("sample_unit") == SAMPLE_UNIT,
        "SAMPLE_UNIT",
        "sample_unit must be %r" % SAMPLE_UNIT,
    )
    _check(
        result,
        "orientation_convention",
        manifest.get("orientation_convention") == ORIENTATION_CONVENTION,
        "ORIENTATION_CONVENTION",
        "orientation_convention must be %r" % ORIENTATION_CONVENTION,
    )
    _check(
        result,
        "metric_artifact",
        isinstance(manifest.get("metric_artifact"), str) and bool(manifest.get("metric_artifact")),
        "METRIC_ARTIFACT_MISSING",
        "metric_artifact must be a non-empty pointer",
    )

    fixture = manifest.get("fixture")
    real_data = manifest.get("real_data")
    result["fixture"] = fixture
    result["real_data"] = real_data
    _check(result, "fixture_bool", isinstance(fixture, bool), "FIXTURE_FLAG", "fixture must be bool")
    _check(result, "real_data_bool", isinstance(real_data, bool), "FIXTURE_FLAG", "real_data must be bool")
    _check(
        result,
        "fixture_xor_real",
        isinstance(fixture, bool) and isinstance(real_data, bool) and fixture != real_data,
        "FIXTURE_FLAGS",
        "exactly one of fixture/real_data must be true",
    )
    if require_real:
        _check(result, "require_real", fixture is False, "FIXTURE_NOT_REAL", "fixture manifest cannot satisfy require_real")
    if real_data is True:
        _check(
            result,
            "real_site_scope",
            manifest.get("site_scope") == SITE_SCOPE,
            "SITE_SCOPE_MISMATCH",
            "real manifest site_scope must be %r" % SITE_SCOPE,
        )

    license_status = manifest.get("license_status")
    if real_data is True:
        _check(
            result,
            "real_license_verified",
            license_status == "PUBLIC_VERIFIED",
            "LICENSE_NOT_VERIFIED",
            "real manifest requires license_status=PUBLIC_VERIFIED, got %r" % (license_status,),
        )
    elif fixture is True:
        _check(
            result,
            "fixture_license_label",
            license_status in ("FIXTURE", "PUBLIC_VERIFIED"),
            "LICENSE_NOT_VERIFIED",
            "fixture license_status must be FIXTURE or PUBLIC_VERIFIED",
        )
    _check(
        result,
        "license_named",
        isinstance(manifest.get("license"), str) and manifest.get("license") not in ("", "UNKNOWN"),
        "LICENSE_MISSING",
        "license must be named",
    )
    _check(
        result,
        "license_evidence",
        isinstance(manifest.get("license_evidence"), str) and bool(manifest.get("license_evidence")),
        "LICENSE_MISSING",
        "license_evidence must be a non-empty pointer",
    )

    source_assets = manifest.get("source_assets")
    _check(
        result,
        "source_assets_present",
        isinstance(source_assets, list) and len(source_assets) > 0,
        "SOURCE_ASSETS_MISSING",
        "source_assets must be a non-empty list",
    )
    source_shas = set()
    if isinstance(source_assets, list):
        for index, asset in enumerate(source_assets):
            prefix = "source_assets[%d]" % index
            if not isinstance(asset, Mapping):
                _check(result, prefix, False, "SOURCE_ASSET_NOT_OBJECT", "source asset must be an object")
                continue
            for field in ("dataset_id", "source_url", "sha256", "license", "license_status", "declared_truth_level"):
                _check(result, "%s.%s" % (prefix, field), field in asset, "SOURCE_ASSET_FIELD_MISSING", "missing %r" % field)
            _check(result, "%s.sha256" % prefix, _sha_ok(asset.get("sha256")), "SOURCE_ASSET_SHA", "invalid sha256")
            _check(
                result,
                "%s.truth_level" % prefix,
                asset.get("declared_truth_level") in REAL_TRUTH_LEVELS,
                "SOURCE_ASSET_TRUTH_LEVEL_FORBIDDEN",
                "declared_truth_level must be real, got %r" % (asset.get("declared_truth_level"),),
            )
            if real_data is True:
                _check(
                    result,
                    "%s.license_public" % prefix,
                    asset.get("license_status") == "PUBLIC_VERIFIED",
                    "SOURCE_ASSET_LICENSE_NOT_PUBLIC",
                    "real source asset must be PUBLIC_VERIFIED",
                )
            source_shas.add(asset.get("sha256"))

    observed_ids = set()
    for index, unit in enumerate(observed):
        prefix = "observed_units[%d]" % index
        if not isinstance(unit, Mapping):
            _check(result, prefix, False, "UNIT_NOT_OBJECT", "unit must be an object")
            continue
        for field in REQUIRED_UNIT_FIELDS:
            _check(result, "%s.%s" % (prefix, field), field in unit, "UNIT_FIELD_MISSING", "missing %r" % field)
        unit_id = unit.get("unit_id")
        observed_ids.add(unit_id)
        _check(result, "%s.unit_id" % prefix, isinstance(unit_id, str) and bool(unit_id), "UNIT_ID", "unit_id must be non-empty")
        _check(result, "%s.unit_type" % prefix, unit.get("unit_type") in UNIT_TYPES, "UNIT_TYPE", "unit_type must be one of %r" % (UNIT_TYPES,))
        count = unit.get("n_observations")
        _check(result, "%s.n_observations" % prefix, isinstance(count, int) and not isinstance(count, bool) and count > 0, "UNIT_COUNT", "n_observations must be positive")
        _check(result, "%s.orientation_convention" % prefix, unit.get("orientation_convention") == ORIENTATION_CONVENTION, "UNIT_ORIENTATION_CONVENTION", "unit orientation_convention must be %r" % ORIENTATION_CONVENTION)
        n_axes = unit.get("n_orientation_axes")
        _check(result, "%s.n_orientation_axes" % prefix, isinstance(n_axes, int) and not isinstance(n_axes, bool) and n_axes == count, "UNIT_AXIS_COUNT", "n_orientation_axes must equal n_observations")
        directed_n = unit.get("directed_dip_direction_n")
        directed_total = unit.get("directed_dip_direction_total")
        _check(result, "%s.directed_total" % prefix, isinstance(directed_total, int) and not isinstance(directed_total, bool) and directed_total == count, "UNIT_DIRECTED_TOTAL", "directed_dip_direction_total must equal n_observations")
        _check(result, "%s.directed_n" % prefix, isinstance(directed_n, int) and not isinstance(directed_n, bool) and isinstance(count, int) and 0 <= directed_n <= count, "UNIT_DIRECTED_COUNT", "directed_dip_direction_n must be in [0,n_observations]")
        _check(result, "%s.sha256" % prefix, _sha_ok(unit.get("sha256")), "UNIT_SHA256", "unit sha256 invalid")
        _check(result, "%s.sha_linked" % prefix, unit.get("sha256") in source_shas, "UNIT_SHA_NOT_IN_SOURCES", "unit sha256 not present in source_assets")
        _check(result, "%s.source_path" % prefix, isinstance(unit.get("source_path"), str) and bool(unit.get("source_path")), "UNIT_SOURCE_MISSING", "source_path must be non-empty")
        depth_range = unit.get("depth_range_ft")
        _check(
            result,
            "%s.depth_range" % prefix,
            isinstance(depth_range, list) and len(depth_range) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in depth_range) and float(depth_range[0]) <= float(depth_range[1]),
            "UNIT_DEPTH_RANGE",
            "depth_range_ft must be [min,max]",
        )
        _check(result, "%s.types" % prefix, isinstance(unit.get("types"), list) and len(unit.get("types")) > 0, "UNIT_TYPES", "types must be a non-empty list")
        _check(result, "%s.filter" % prefix, isinstance(unit.get("fracture_filter"), str) and bool(unit.get("fracture_filter")), "UNIT_FILTER", "fracture_filter must be non-empty")

    held_out_ids = set()
    for index, unit in enumerate(held_out):
        prefix = "held_out_units[%d]" % index
        if not isinstance(unit, Mapping):
            _check(result, prefix, False, "UNIT_NOT_OBJECT", "unit must be an object")
            continue
        for field in REQUIRED_UNIT_FIELDS:
            _check(result, "%s.%s" % (prefix, field), field in unit, "UNIT_FIELD_MISSING", "missing %r" % field)
        unit_id = unit.get("unit_id")
        held_out_ids.add(unit_id)
        _check(result, "%s.unit_id" % prefix, isinstance(unit_id, str) and bool(unit_id), "UNIT_ID", "unit_id must be non-empty")
        _check(result, "%s.unit_type" % prefix, unit.get("unit_type") in UNIT_TYPES, "UNIT_TYPE", "unit_type must be one of %r" % (UNIT_TYPES,))
        count = unit.get("n_observations")
        _check(result, "%s.n_observations" % prefix, isinstance(count, int) and not isinstance(count, bool) and count > 0, "UNIT_COUNT", "n_observations must be positive")
        _check(result, "%s.orientation_convention" % prefix, unit.get("orientation_convention") == ORIENTATION_CONVENTION, "UNIT_ORIENTATION_CONVENTION", "unit orientation_convention must be %r" % ORIENTATION_CONVENTION)
        n_axes = unit.get("n_orientation_axes")
        _check(result, "%s.n_orientation_axes" % prefix, isinstance(n_axes, int) and not isinstance(n_axes, bool) and n_axes == count, "UNIT_AXIS_COUNT", "n_orientation_axes must equal n_observations")
        directed_n = unit.get("directed_dip_direction_n")
        directed_total = unit.get("directed_dip_direction_total")
        _check(result, "%s.directed_total" % prefix, isinstance(directed_total, int) and not isinstance(directed_total, bool) and directed_total == count, "UNIT_DIRECTED_TOTAL", "directed_dip_direction_total must equal n_observations")
        _check(result, "%s.directed_n" % prefix, isinstance(directed_n, int) and not isinstance(directed_n, bool) and isinstance(count, int) and 0 <= directed_n <= count, "UNIT_DIRECTED_COUNT", "directed_dip_direction_n must be in [0,n_observations]")
        _check(result, "%s.sha256" % prefix, _sha_ok(unit.get("sha256")), "UNIT_SHA256", "unit sha256 invalid")
        _check(result, "%s.sha_linked" % prefix, unit.get("sha256") in source_shas, "UNIT_SHA_NOT_IN_SOURCES", "unit sha256 not present in source_assets")
        _check(result, "%s.source_path" % prefix, isinstance(unit.get("source_path"), str) and bool(unit.get("source_path")), "UNIT_SOURCE_MISSING", "source_path must be non-empty")
        depth_range = unit.get("depth_range_ft")
        _check(
            result,
            "%s.depth_range" % prefix,
            isinstance(depth_range, list) and len(depth_range) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in depth_range) and float(depth_range[0]) <= float(depth_range[1]),
            "UNIT_DEPTH_RANGE",
            "depth_range_ft must be [min,max]",
        )
        _check(result, "%s.types" % prefix, isinstance(unit.get("types"), list) and len(unit.get("types")) > 0, "UNIT_TYPES", "types must be a non-empty list")
        _check(result, "%s.filter" % prefix, isinstance(unit.get("fracture_filter"), str) and bool(unit.get("fracture_filter")), "UNIT_FILTER", "fracture_filter must be non-empty")

    _check(
        result,
        "units_independent",
        not (observed_ids & held_out_ids),
        "UNIT_NOT_INDEPENDENT",
        "observed and held-out unit IDs overlap: %r" % sorted(observed_ids & held_out_ids),
    )

    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        observed_sum = sum(unit.get("n_observations", 0) for unit in observed if isinstance(unit, Mapping))
        held_out_sum = sum(unit.get("n_observations", 0) for unit in held_out if isinstance(unit, Mapping))
        _check(result, "counts.observed", counts.get("observed") == observed_sum, "COUNT_MISMATCH", "counts.observed != sum(observed units)")
        _check(result, "counts.held_out", counts.get("held_out") == held_out_sum, "COUNT_MISMATCH", "counts.held_out != sum(held_out units)")
        _check(result, "counts.total", counts.get("total") == observed_sum + held_out_sum, "COUNT_MISMATCH", "counts.total != observed+held_out")
        _check(result, "counts.n_observed_units", counts.get("n_observed_units") == len(observed), "COUNT_MISMATCH", "counts.n_observed_units mismatch")
        _check(result, "counts.n_held_out_units", counts.get("n_held_out_units") == len(held_out), "COUNT_MISMATCH", "counts.n_held_out_units mismatch")
        axes_sum = sum(unit.get("n_orientation_axes", 0) for unit in observed + held_out if isinstance(unit, Mapping))
        directed_sum = sum(unit.get("directed_dip_direction_n", 0) for unit in observed + held_out if isinstance(unit, Mapping))
        if "n_orientation_axes" in counts:
            _check(result, "counts.n_orientation_axes", counts.get("n_orientation_axes") == axes_sum, "COUNT_MISMATCH", "counts.n_orientation_axes mismatch")
        if "n_directed_dip_direction" in counts:
            _check(result, "counts.n_directed_dip_direction", counts.get("n_directed_dip_direction") == directed_sum, "COUNT_MISMATCH", "counts.n_directed_dip_direction mismatch")
    else:
        _check(result, "counts_object", False, "COUNT_MISMATCH", "counts must be an object")

    allowed_claims = manifest.get("allowed_claims")
    _check(result, "allowed_claims_list", isinstance(allowed_claims, list), "CLAIM_NOT_ALLOWED", "allowed_claims must be a list")
    if isinstance(allowed_claims, list):
        _check(result, "allowed_claim_present", ALLOWED_CLAIM in allowed_claims, "CLAIM_NOT_ALLOWED", "allowed_claims must include %r" % ALLOWED_CLAIM)
        forbidden_hits = [token for token in FORBIDDEN_CLAIM_TOKENS if any(token in str(claim) for claim in allowed_claims)]
        _check(result, "no_overclaim_tokens", not forbidden_hits, "CLAIM_NOT_ALLOWED", "overclaim tokens in allowed_claims: %r" % forbidden_hits)

    forbidden_claims = manifest.get("forbidden_claims")
    _check(result, "forbidden_claims_list", isinstance(forbidden_claims, list), "CLAIM_FORBIDDEN_MISSING", "forbidden_claims must be a list")
    if isinstance(forbidden_claims, list):
        blob = " ".join(str(item) for item in forbidden_claims)
        missing = [phrase for phrase in REQUIRED_FORBIDDEN_CLAIMS if phrase not in blob]
        _check(result, "required_forbidden_claims", not missing, "CLAIM_FORBIDDEN_MISSING", "missing forbidden claims: %r" % missing)

    protocol = manifest.get("observation_protocol")
    if isinstance(protocol, Mapping):
        for field in REQUIRED_PROTOCOL_FIELDS:
            _check(result, "protocol.%s" % field, field in protocol, "PROTOCOL_MISSING", "protocol missing %r" % field)
        _check(result, "protocol.no_3d_geometry_claim", protocol.get("no_3d_geometry_claim") is True, "PROTOCOL_3D_CLAIM", "no_3d_geometry_claim must be true")
        _check(result, "protocol.no_connectivity_label", protocol.get("no_connectivity_label") is True, "PROTOCOL_CONNECTIVITY_CLAIM", "no_connectivity_label must be true")
        _check(result, "protocol.rationale", isinstance(protocol.get("truth_level_rationale"), str) and bool(protocol.get("truth_level_rationale")), "PROTOCOL_MISSING", "truth_level_rationale must be non-empty")
        _check(result, "protocol.orientation_convention", protocol.get("orientation_convention") == ORIENTATION_CONVENTION, "PROTOCOL_ORIENTATION_CONVENTION", "protocol orientation_convention must be %r" % ORIENTATION_CONVENTION)
        _check(result, "protocol.axis_sign_convention", protocol.get("axis_sign_convention") == AXIS_SIGN_CONVENTION, "PROTOCOL_AXIS_SIGN", "axis_sign_convention must be %r" % AXIS_SIGN_CONVENTION)
        _check(result, "protocol.metric_family", protocol.get("metric_family") == METRIC_FAMILY, "PROTOCOL_METRIC_FAMILY", "metric_family must be %r" % METRIC_FAMILY)
        for flag in ("directed_dip_direction_metric", "connectivity_metric", "three_d_truth", "real_generalization"):
            _check(result, "protocol.%s_false" % flag, protocol.get(flag) is False, "PROTOCOL_FORBIDDEN_METRIC", "%s must be false" % flag)
    else:
        _check(result, "protocol_object", False, "PROTOCOL_MISSING", "observation_protocol must be an object")

    _check(
        result,
        "evaluation_time",
        isinstance(manifest.get("evaluation_time"), str) and bool(manifest.get("evaluation_time")),
        "PROTOCOL_MISSING",
        "evaluation_time must be non-empty",
    )
    if "metric_family" in manifest:
        _check(result, "metric_family", manifest.get("metric_family") == METRIC_FAMILY, "METRIC_FAMILY", "metric_family must be %r" % METRIC_FAMILY)
    claim_scope = manifest.get("claim_scope")
    if claim_scope is not None:
        _check(result, "claim_scope.object", isinstance(claim_scope, Mapping), "CLAIM_SCOPE", "claim_scope must be an object")
        if isinstance(claim_scope, Mapping):
            _check(result, "claim_scope.allowed", claim_scope.get("allowed_claim") == UNDIRECTED_CLAIM, "CLAIM_SCOPE", "claim_scope.allowed_claim must be %r" % UNDIRECTED_CLAIM)
            for flag in ("directed_dip_direction_metric", "connectivity_metric", "three_d_truth", "real_generalization"):
                _check(result, "claim_scope.%s_false" % flag, claim_scope.get(flag) is False, "CLAIM_SCOPE", "claim_scope.%s must be false" % flag)
            blob = canonical_json(claim_scope)
            hits = [token for token in FORBIDDEN_CLAIM_TOKENS if token in blob]
            _check(result, "claim_scope.no_overclaim", not hits, "CLAIM_SCOPE", "overclaim tokens in claim_scope: %r" % hits)

    forbidden_key_hits = _scan_forbidden_keys(manifest)
    _check(result, "no_forbidden_keys", not forbidden_key_hits, "FORBIDDEN_KEY", "forbidden keys: %r" % forbidden_key_hits[:8])

    result["pass"] = not result["violations"]
    result["enumerated_checks"] = result["n_checks"]
    return result


def _read_las_pick_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    curves = _curve_names(lines)
    if not curves:
        raise RealHoldoutError("LAS curve block not found: %s" % target)
    ascii_start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("~ascii"):
            ascii_start = index + 1
            break
    if ascii_start is None:
        raise RealHoldoutError("LAS ~Ascii block not found: %s" % target)
    rows = []
    for line in lines[ascii_start:]:
        if not line.strip():
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            continue
        if len(parts) < len(curves):
            continue
        row = dict(zip(curves, parts))
        normalized = {str(key).strip().upper(): value for key, value in row.items()}
        type_value = normalized.get("TYPE")
        if type_value is None:
            type_value = parts[-1]
        depth = None
        for key in ("MD", "TDEP", "DEPTH", "DEPT"):
            if key in normalized:
                depth = _finite_number(normalized[key])
                break
        dip = None
        for key in ("DIP_TRU", "DIP", "DIP_ANGLE"):
            if key in normalized:
                dip = _finite_number(normalized[key])
                break
        azimuth = None
        for key in ("AZIMUTH", "DIP_DIRECTION", "DIPDIR"):
            if key in normalized:
                azimuth = _finite_number(normalized[key])
                break
        is_fracture, is_induced, is_natural = _classify_pick(type_value)
        rows.append({
            "type": type_value,
            "depth": depth,
            "dip": dip,
            "dip_direction": None,
            "azimuth": azimuth,
            "is_fracture": is_fracture,
            "is_induced": is_induced,
            "is_natural": is_natural,
        })
    return rows


def _read_xlsx_pick_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    sheets = _xlsx_sheets(target)
    data_rows = None
    header_map = None
    for _name, rows in sheets:
        try:
            candidate_rows, candidate_map = _rows_from_sheet(rows, required=("MD", "Type"))
        except RealHoldoutError:
            continue
        data_rows = candidate_rows
        header_map = candidate_map
        break
    if data_rows is None or header_map is None:
        raise RealHoldoutError("FMI xlsx sheet with MD/Type not found: %s" % target)
    rows = []
    for row in data_rows:
        type_value = row.get(header_map["Type"])
        md = _finite_number(row.get(header_map["MD"]))
        if md is None or type_value is None:
            continue
        dip = _finite_number(row.get(header_map.get("Dip"))) if header_map.get("Dip") is not None else None
        dip_direction = _finite_number(row.get(header_map.get("Dip Direction"))) if header_map.get("Dip Direction") is not None else None
        azimuth = _finite_number(row.get(header_map.get("Azimuth"))) if header_map.get("Azimuth") is not None else None
        is_fracture, is_induced, is_natural = _classify_pick(type_value)
        rows.append({
            "type": type_value,
            "depth": md,
            "dip": dip,
            "dip_direction": dip_direction,
            "azimuth": azimuth,
            "is_fracture": is_fracture,
            "is_induced": is_induced,
            "is_natural": is_natural,
        })
    return rows


def _canonical_axis(axis: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = [float(value) for value in axis]
    norm = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RealHoldoutError("invalid plane axis: %r" % (list(axis),))
    x, y, z = x / norm, y / norm, z / norm
    if abs(z) <= 1e-12:
        z = 0.0
    if z < 0.0 or (z == 0.0 and (x < 0.0 or (abs(x) <= 1e-12 and y < 0.0))):
        x, y, z = -x, -y, -z
    return (x, y, z)


def plane_axis_from_dip_direction(dip_deg: float, dip_direction_deg: float) -> tuple[float, float, float]:
    dip = math.radians(float(dip_deg))
    dip_direction = math.radians(float(dip_direction_deg))
    return _canonical_axis((
        math.sin(dip) * math.sin(dip_direction),
        math.sin(dip) * math.cos(dip_direction),
        math.cos(dip),
    ))


def fixture_manifest() -> dict[str, Any]:
    source_a = "0" * 64
    source_b = "1" * 64
    observed = [{
        "unit_id": "FIXTURE_WELL_OBS_A",
        "unit_type": "well",
        "n_observations": 2,
        "source_path": "fixture://observed_a",
        "sha256": source_a,
        "depth_range_ft": [0.0, 10.0],
        "types": ["FIXTURE_FRACTURE"],
        "fracture_filter": "fixture only",
        "orientation_convention": ORIENTATION_CONVENTION,
        "n_orientation_axes": 2,
        "directed_dip_direction_n": 2,
        "directed_dip_direction_total": 2,
    }]
    held_out = [{
        "unit_id": "FIXTURE_WELL_HOLD_B",
        "unit_type": "well",
        "n_observations": 3,
        "source_path": "fixture://held_out_b",
        "sha256": source_b,
        "depth_range_ft": [10.0, 20.0],
        "types": ["FIXTURE_FRACTURE"],
        "fracture_filter": "fixture only",
        "orientation_convention": ORIENTATION_CONVENTION,
        "n_orientation_axes": 3,
        "directed_dip_direction_n": 3,
        "directed_dip_direction_total": 3,
    }]
    return {
        "schema": SCHEMA_MANIFEST,
        "group_type": GROUP_REAL_CENSORING_HOLDOUT,
        "truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
        "task_family": TASK_FAMILY,
        "site_id": "FIXTURE_SITE",
        "site_scope": "FIXTURE_SITE",
        "sample_unit": SAMPLE_UNIT,
        "orientation_convention": ORIENTATION_CONVENTION,
        "metric_artifact": "fixture://metrics",
        "metric_family": METRIC_FAMILY,
        "fixture": True,
        "real_data": False,
        "observed_units": observed,
        "held_out_units": held_out,
        "source_assets": [
            {
                "dataset_id": "FIXTURE_OBSERVED_A",
                "source_url": "fixture://observed_a",
                "sha256": source_a,
                "license": "FIXTURE_ONLY",
                "license_status": "FIXTURE",
                "declared_truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
            },
            {
                "dataset_id": "FIXTURE_HELD_OUT_B",
                "source_url": "fixture://held_out_b",
                "sha256": source_b,
                "license": "FIXTURE_ONLY",
                "license_status": "FIXTURE",
                "declared_truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
            },
        ],
        "license": "FIXTURE_ONLY",
        "license_status": "FIXTURE",
        "license_evidence": "fixture://not-a-license",
        "observation_protocol": {
            "modality": "FIXTURE_ONLY",
            "coordinate_frame": "FIXTURE",
            "truth_level_rationale": "fixture only; not real data; no 3D truth",
            "enumeration": "FIXTURE_ROWS",
            "depth_reference": "MD_FT",
            "no_3d_geometry_claim": True,
            "no_connectivity_label": True,
            "orientation_convention": ORIENTATION_CONVENTION,
            "axis_sign_convention": AXIS_SIGN_CONVENTION,
            "metric_family": METRIC_FAMILY,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "evaluation_time": "FIXTURE_TIME",
        "allowed_claims": [ALLOWED_CLAIM],
        "forbidden_claims": list(REQUIRED_FORBIDDEN_CLAIMS),
        "claim_scope": {
            "allowed_claim": UNDIRECTED_CLAIM,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "counts": {
            "observed": 2,
            "held_out": 3,
            "total": 5,
            "n_observed_units": 1,
            "n_held_out_units": 1,
            "n_orientation_axes": 5,
            "n_directed_dip_direction": 5,
        },
        "validation": {"fixture_only": True},
    }


_ASSET_SPECS = (
    {
        "dataset_id": "forge_gdr1595_16a_fmi_picks",
        "unit_id": "FORGE_16A(78)-32",
        "asset_class": "FMI_PICKS",
        "probe": "XLSX_FMI",
        "role": "observed_candidate",
    },
    {
        "dataset_id": "forge_gdr1727_16b_fmi_reinterp",
        "unit_id": "FORGE_16B(78)-32",
        "asset_class": "FMI_PICKS",
        "probe": "LAS_PICKS",
        "role": "observed_candidate",
    },
    {
        "dataset_id": "forge_gdr1076_58_32_fmi",
        "unit_id": "FORGE_58-32",
        "asset_class": "FMI_PICKS",
        "probe": "LAS_PICKS",
        "role": "held_out_candidate",
    },
    {
        "dataset_id": "forge_gdr1708_16b_core",
        "unit_id": "FORGE_16B(78)-32_CORE",
        "asset_class": "CORE",
        "probe": "CORE_ZIP",
        "role": "not_independent",
    },
    {
        "dataset_id": "forge_gdr1222_dfn",
        "unit_id": None,
        "asset_class": "DFN_MODEL",
        "probe": "REGISTRY_ONLY",
        "role": "not_observation",
    },
    {
        "dataset_id": "forge_gdr1646_meq_dfn",
        "unit_id": None,
        "asset_class": "DFN_MODEL",
        "probe": "REGISTRY_ONLY",
        "role": "not_observation",
    },
    {
        "dataset_id": "forge_gdr1750_dfn_2025",
        "unit_id": None,
        "asset_class": "DFN_MODEL",
        "probe": "REGISTRY_ONLY",
        "role": "not_observation",
    },
    {
        "dataset_id": "geocrack_patched_images",
        "unit_id": None,
        "asset_class": "2D_MASKS",
        "probe": "REGISTRY_ONLY",
        "role": "not_3d_borehole",
    },
    {
        "dataset_id": "aland_getaberget_traces_20m",
        "unit_id": None,
        "asset_class": "2D_TRACES",
        "probe": "REGISTRY_ONLY",
        "role": "license_unverified",
    },
    {
        "dataset_id": "decovalex_4frac_dfn",
        "unit_id": None,
        "asset_class": "SYNTHETIC_DFN",
        "probe": "REGISTRY_ONLY",
        "role": "synthetic_benchmark",
    },
    {
        "dataset_id": "e_disk_hrf_frac",
        "unit_id": None,
        "asset_class": "WALL_POINT_CLOUD",
        "probe": "REGISTRY_ONLY",
        "role": "license_unknown",
    },
    {
        "dataset_id": "e_disk_new_frac",
        "unit_id": None,
        "asset_class": "WALL_POINT_CLOUD",
        "probe": "REGISTRY_ONLY",
        "role": "license_unknown",
    },
)

_SUPPLEMENTARY_ASSETS = (
    {
        "dataset_id": "egs_collab_well_instrumentation",
        "local_path": "data/external/gdr_egs_collab",
        "modality": "well instrumentation and source/receiver geometry",
        "license": "CC-BY-4.0 (secondary project note; not re-verified)",
        "license_status": "UNVERIFIED",
        "label_provenance": "UNLABELED",
        "asset_class": "INSTRUMENTATION",
        "unit_id": None,
        "role": "not_fracture_observation",
        "notes": "no fracture picks; only well instrumentation",
    },
    {
        "dataset_id": "snake_river_inel1_las",
        "local_path": "data/external/snake_river_INEL1",
        "modality": "LAS well logs; fracture-id log PDF not downloaded",
        "license": "CC-BY-4.0 (secondary project note; not re-verified)",
        "license_status": "UNVERIFIED",
        "label_provenance": "UNLABELED",
        "asset_class": "WELL_LOGS",
        "unit_id": "INEL-1",
        "role": "no_fracture_picks",
        "notes": "fracture identification PDF absent",
    },
    {
        "dataset_id": "published_reports_catalogue",
        "local_path": "data/external/published_reports",
        "modality": "extracted report tables / templates",
        "license": "NO_LICENSE_FOUND",
        "license_status": "UNVERIFIED",
        "label_provenance": "EXPERT_INTERPRETED",
        "asset_class": "REPORT_TABLES",
        "unit_id": None,
        "role": "template_only",
        "notes": "manifest rows are identified templates, not verified raw borehole tables",
    },
)

_GLOBAL_BLOCKERS = (
    {
        "code": "NO_3D_CONNECTIVITY_TRUTH",
        "detail": "Borehole assets provide 1D picks; no verifiable 3D fracture network or two-point connectivity labels. Deriving labels would require inventing disk centers/radii and inter-well fractures.",
    },
    {
        "code": "PICKS_ONLY_TRUTH_LEVEL",
        "detail": "Only EXPERT_BOREHOLE_PICKS is available. Raw FMI/OTV image volumes for HELD_OUT_OBSERVATION are not on disk.",
    },
    {
        "code": "SINGLE_SITE",
        "detail": "All suitable borehole picks are FORGE Milford. Whole-well holdout is possible; cross-site holdout is not.",
    },
    {
        "code": "NO_REFERENCE_POSTERIOR",
        "detail": "No generative prior, prior weights or full-scene enumeration exists for these wells; a reference posterior cannot be computed.",
    },
    {
        "code": "TASK_FAMILY_NOT_IN_FROZEN_QUERY",
        "detail": "A picks-only cross-well observation holdout is not the frozen TWO_POINT_CONNECTIVITY query. It needs explicit architecture approval before becoming a benchmark arm.",
    },
    {
        "code": "WELL_SURVEY_MACHINE_READABILITY",
        "detail": "58-32 has only PDF directional surveys; the 16B survey .xlsx in the zip is not a valid OOXML archive. MD-to-XYZ registration is not validated and is not used here.",
    },
    {
        "code": "CORE_NOT_INDEPENDENT",
        "detail": "The 16B core covers 3 intervals in the same well as the 16B FMI log and is not an independent well/site unit.",
    },
    {
        "code": "NO_RAW_SENSOR_OBSERVATION",
        "detail": "Only interpreted LAS/XLSX picks are local; no raw FMI images or DLIS sensor volumes.",
    },
)


def _asset_truth_level(asset_class: str, label_provenance: str | None) -> str:
    if label_provenance == "EXACT":
        return "EXACT_SYNTHETIC"
    if label_provenance == "EXPERT_INTERPRETED":
        if asset_class in ("FMI_PICKS", "CORE"):
            return TRUTH_EXPERT_BOREHOLE_PICKS
        if asset_class in ("2D_MASKS", "2D_TRACES"):
            return "EXPERT_SURFACE_INTERPRETATION"
        if asset_class == "DFN_MODEL":
            return "DFN_MODEL_NOT_OBSERVATION"
    if asset_class == "SYNTHETIC_DFN":
        return "EXACT_SYNTHETIC"
    return "NOT_ALLOWED_" + str(label_provenance or "UNKNOWN")


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _audit_asset(spec: Mapping[str, Any], row: Mapping[str, Any] | None, evidence: Mapping[str, Any] | None, root: Path) -> dict[str, Any]:
    dataset_id = spec["dataset_id"]
    row = dict(row or {})
    local_path = row.get("local_path") or spec.get("local_path")
    path = _resolve_path(root, local_path)
    exists = bool(path and path.exists())
    is_file = bool(path and path.is_file())
    size = path.stat().st_size if is_file and path else None
    sha256_match = None
    actual_sha = None
    if is_file and size is not None and size <= 100 * 1000 * 1000:
        actual_sha = file_sha256(path)
        sha256_match = actual_sha == row.get("sha256")
    probe: dict[str, Any] = {"probe": "REGISTRY_ONLY", "bytes": size}
    blockers: list[dict[str, str]] = []
    if not exists:
        blockers.append({"code": "FILE_MISSING", "detail": str(path)})
    if sha256_match is False:
        blockers.append({"code": "SHA_MISMATCH", "detail": "registry sha256 differs from on-disk bytes"})
    probe_kind = spec.get("probe")
    if probe["probe"] == "REGISTRY_ONLY" and probe_kind and probe_kind != "REGISTRY_ONLY":
        try:
            if probe_kind == "LAS_PICKS":
                probe = probe_las_picks(path)
            elif probe_kind == "XLSX_FMI":
                probe = probe_xlsx_fmi(path)
            elif probe_kind == "CORE_ZIP":
                probe = probe_core_zip(path)
            if sha256_match is None and probe.get("sha256"):
                sha256_match = probe.get("sha256") == row.get("sha256")
        except (OSError, ValueError, RealHoldoutError) as exc:
            blockers.append({"code": "PROBE_FAILED", "detail": str(exc)})
            probe = {"probe": "PROBE_FAILED", "error": str(exc), "bytes": size}
    license_status = row.get("license_status", "UNKNOWN")
    license_name = row.get("license", "UNKNOWN")
    label_provenance = row.get("label_provenance", "UNKNOWN")
    truth_level = _asset_truth_level(spec["asset_class"], label_provenance)
    if license_status != "PUBLIC_VERIFIED":
        blockers.append({"code": "LICENSE_NOT_PUBLIC_VERIFIED", "detail": "%s/%s" % (license_name, license_status)})
    if truth_level not in REAL_TRUTH_LEVELS:
        blockers.append({"code": "TRUTH_LEVEL_NOT_ALLOWED", "detail": truth_level})
    if spec["asset_class"] != "FMI_PICKS":
        blockers.append({"code": "NOT_AN_FMI_WELL_PICK_ASSET", "detail": spec["asset_class"]})
    if spec["asset_class"] == "FMI_PICKS" and probe.get("n_natural", 0) <= 0:
        blockers.append({"code": "NO_NATURAL_PICKS", "detail": "probe found no natural fracture picks"})
    role = spec.get("role")
    if role == "not_independent":
        blockers.append({"code": "CORE_NOT_INDEPENDENT", "detail": "same well as 16B FMI; 3 intervals only"})
    if role == "not_observation":
        blockers.append({"code": "DFN_MODEL_NOT_OBSERVATION", "detail": "site-level DFN model, not a held-out observation"})
    if role == "not_3d_borehole":
        blockers.append({"code": "NOT_3D_BOREHOLE", "detail": "2D image patches; station IDs absent in patch release"})
    if role == "license_unverified":
        blockers.append({"code": "LICENSE_UNVERIFIED_OR_2D", "detail": "no data license found; 2D traces only"})
    if role == "synthetic_benchmark":
        blockers.append({"code": "SYNTHETIC_BENCHMARK_NOT_REAL", "detail": "dfnWorks synthetic benchmark"})
    if role == "license_unknown":
        blockers.append({"code": "LICENSE_UNKNOWN", "detail": "no upstream record; single wall point cloud"})
    if role in ("not_fracture_observation", "no_fracture_picks"):
        blockers.append({"code": "NO_FRACTURE_PICKS", "detail": "no fracture-pick observation in this asset"})
    if role == "template_only":
        blockers.append({"code": "NO_RAW_PICKS", "detail": "catalogue/template rows only; no verified raw borehole table"})
    suitable = (
        spec["asset_class"] == "FMI_PICKS"
        and license_status == "PUBLIC_VERIFIED"
        and truth_level == TRUTH_EXPERT_BOREHOLE_PICKS
        and exists
        and sha256_match is not False
        and probe.get("n_natural", 0) > 0
        and bool(spec.get("unit_id"))
        and not blockers
    )
    return {
        "dataset_id": dataset_id,
        "site_id": row.get("site_id"),
        "modality": row.get("modality", spec.get("modality")),
        "asset_class": spec["asset_class"],
        "role": role,
        "unit_id": spec.get("unit_id"),
        "independent_unit": "well" if spec["asset_class"] == "FMI_PICKS" else ("core" if spec["asset_class"] == "CORE" else None),
        "local_path": local_path,
        "exists": exists,
        "bytes": size,
        "sha256_registry": row.get("sha256"),
        "sha256_actual": actual_sha,
        "sha256_match_registry": sha256_match,
        "license": license_name,
        "license_status": license_status,
        "license_evidence_level": (evidence or {}).get("evidence_level"),
        "source_url": row.get("source_url"),
        "label_provenance": label_provenance,
        "truth_level_candidate": truth_level,
        "probe": probe,
        "suitable_for_picks_only_cross_well_holdout": bool(suitable),
        "blockers": blockers,
    }


def audit_local_assets(
    registry_path: str | Path,
    license_evidence_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    registry_path = Path(registry_path)
    license_evidence_path = Path(license_evidence_path)
    registry = read_json(registry_path)
    rows = registry.get("rows") if isinstance(registry, Mapping) else None
    if not rows:
        raise SystemExit("EMPTY_REGISTRY: no rows; refusing to audit an empty set")
    evidence_doc = read_json(license_evidence_path)
    entries = evidence_doc.get("entries") if isinstance(evidence_doc, Mapping) else []
    evidence_by_id = {}
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("dataset_id"):
                evidence_by_id[entry["dataset_id"]] = entry
    elif isinstance(entries, Mapping):
        evidence_by_id = dict(entries)
    rows_by_id = {row.get("dataset_id"): row for row in rows if isinstance(row, Mapping)}
    assets = []
    for spec in _ASSET_SPECS:
        row = rows_by_id.get(spec["dataset_id"])
        assets.append(_audit_asset(spec, row, evidence_by_id.get(spec["dataset_id"]), root))
    for spec in _SUPPLEMENTARY_ASSETS:
        row = {
            "dataset_id": spec["dataset_id"],
            "site_id": None,
            "modality": spec["modality"],
            "local_path": spec["local_path"],
            "license": spec["license"],
            "license_status": spec["license_status"],
            "label_provenance": spec["label_provenance"],
            "source_url": None,
            "sha256": None,
        }
        asset = _audit_asset(spec, row, None, root)
        asset["notes"] = spec.get("notes")
        asset["registry_status"] = "NOT_IN_G58"
        assets.append(asset)
    assets.sort(key=lambda item: str(item.get("dataset_id")))
    suitable = [asset for asset in assets if asset.get("suitable_for_picks_only_cross_well_holdout")]
    suitable_by_unit = {}
    for asset in suitable:
        suitable_by_unit[asset["unit_id"]] = asset
    held_out_unit = "FORGE_58-32" if "FORGE_58-32" in suitable_by_unit else (sorted(suitable_by_unit)[-1] if suitable_by_unit else None)
    held_out = suitable_by_unit.get(held_out_unit)
    observed = [asset for unit_id, asset in sorted(suitable_by_unit.items()) if unit_id != held_out_unit]
    n_observed = sum(int(asset["probe"].get("n_natural", 0)) for asset in observed)
    n_held_out = int(held_out["probe"].get("n_natural", 0)) if held_out else 0
    candidate = {
        "candidate_id": "forge_milford_cross_well_58_32_holdout",
        "site_id": "forge_milford",
        "task_family": TASK_FAMILY,
        "truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
        "license": "CC-BY-4.0",
        "license_status": "PUBLIC_VERIFIED",
        "n_independent_wells": len(suitable),
        "observed_units": [
            {
                "unit_id": asset["unit_id"],
                "dataset_id": asset["dataset_id"],
                "n_natural_picks": asset["probe"].get("n_natural"),
                "n_fracture_picks": asset["probe"].get("n_fracture"),
                "n_induced_picks": asset["probe"].get("n_induced"),
                "n_rows": asset["probe"].get("n_rows"),
                "depth_md_ft": asset["probe"].get("depth_md_ft"),
                "source_path": asset["local_path"],
                "sha256_match_registry": asset["sha256_match_registry"],
            }
            for asset in observed
        ],
        "held_out_unit": {
            "unit_id": held_out["unit_id"],
            "dataset_id": held_out["dataset_id"],
            "n_natural_picks": held_out["probe"].get("n_natural"),
            "n_fracture_picks": held_out["probe"].get("n_fracture"),
            "n_induced_picks": held_out["probe"].get("n_induced"),
            "n_rows": held_out["probe"].get("n_rows"),
            "depth_md_ft": held_out["probe"].get("depth_md_ft"),
            "source_path": held_out["local_path"],
            "sha256_match_registry": held_out["sha256_match_registry"],
        } if held_out else None,
        "n_observed_natural_picks": n_observed,
        "n_held_out_natural_picks": n_held_out,
        "n_total_natural_picks": n_observed + n_held_out,
        "supports_cross_well_observation_holdout": bool(held_out and observed),
        "supports_two_point_connectivity_truth": False,
        "supports_full_3d_truth": False,
        "supports_cross_site_holdout": False,
        "decision": "PENDING_USER_APPROVAL",
    }
    scope_verdicts = {
        "existing_two_point_connectivity_query": "BLOCKED_BY_DATA",
        "picks_only_cross_well_observation_holdout": "CANDIDATE_PENDING_APPROVAL" if candidate["supports_cross_well_observation_holdout"] else "BLOCKED_BY_DATA",
        "full_3d_real_truth": "NOT_SUPPORTED",
        "real_generalization": "NOT_SUPPORTED",
        "cross_site_holdout": "BLOCKED_BY_DATA",
    }
    return {
        "schema": SCHEMA_AUDIT,
        "watermark": WATERMARK,
        "generated_at_utc": utc_now(),
        "audit_status": "CANDIDATE_PICKS_ONLY_PENDING_APPROVAL" if candidate["supports_cross_well_observation_holdout"] else "BLOCKED_BY_DATA",
        "scope_verdicts": scope_verdicts,
        "real_manifest_written": False,
        "real_adapter_implemented": False,
        "source": {
            "registry_path": str(registry_path),
            "registry_sha256": file_sha256(registry_path),
            "license_evidence_path": str(license_evidence_path),
            "license_evidence_sha256": file_sha256(license_evidence_path),
        },
        "n_assets": len(assets),
        "n_public_verified": sum(1 for asset in assets if asset.get("license_status") == "PUBLIC_VERIFIED"),
        "n_independent_borehole_wells": len(suitable),
        "suitable_unit_ids": sorted(suitable_by_unit),
        "candidate": candidate,
        "blockers": [dict(item) for item in _GLOBAL_BLOCKERS] + [
            {
                "code": "AWAITING_USER_APPROVAL",
                "detail": "The picks-only task family is not approved for benchmark use; no real manifest was written.",
            }
        ],
        "assets": assets,
        "readings": {
            "scope": "ALL_G58_REGISTRY_ASSETS_PLUS_SUPPLEMENTARY_LOCAL_ASSETS",
            "enumeration": "EXHAUSTIVE_ASSET_PROBE",
            "measured_at": utc_now(),
        },
    }


_AXIS_SPECS = (
    {
        "dataset_id": "forge_gdr1595_16a_fmi_picks",
        "unit_id": "FORGE_16A(78)-32",
        "role": "observed",
        "mode": "STRIKE_DIP",
        "directed_field": "Dip Direction",
    },
    {
        "dataset_id": "forge_gdr1727_16b_fmi_reinterp",
        "unit_id": "FORGE_16B(78)-32",
        "role": "observed",
        "mode": "DIP_AZIMUTH_DIP",
        "directed_field": "Azimuth (true dip azimuth)",
    },
    {
        "dataset_id": "forge_gdr1076_58_32_fmi",
        "unit_id": "FORGE_58-32",
        "role": "held_out",
        "mode": "DIP_AZIMUTH_DIP",
        "directed_field": "Azimuth (true dip azimuth)",
    },
)


def extract_orientation_axes(
    registry_path: str | Path,
    license_evidence_path: str | Path,
    *,
    project_root: str | Path | None = None,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    audit = audit or audit_local_assets(registry_path, license_evidence_path, project_root=root)
    assets = {asset["dataset_id"]: asset for asset in audit["assets"]}
    wells = {}
    for spec in _AXIS_SPECS:
        asset = assets.get(spec["dataset_id"])
        if asset is None:
            raise BlockedByData([{"code": "ASSET_MISSING", "detail": spec["dataset_id"]}])
        path = _resolve_path(root, asset.get("local_path"))
        if path is None or not path.is_file():
            raise BlockedByData([{"code": "ASSET_MISSING", "detail": str(path)}])
        if spec["mode"] == "STRIKE_DIP":
            rows = _read_xlsx_pick_rows(path)
        else:
            rows = _read_las_pick_rows(path)
        natural = [row for row in rows if row.get("is_natural")]
        axes = []
        missing = []
        directed_n = 0
        for row in natural:
            dip = row.get("dip")
            if spec["mode"] == "STRIKE_DIP":
                strike = row.get("azimuth")
                if row.get("dip_direction") is not None:
                    directed_n += 1
                if dip is None or strike is None:
                    missing.append({"type": row.get("type"), "depth": row.get("depth")})
                    continue
                dip_direction = (float(strike) + 90.0) % 360.0
            else:
                dip_direction = row.get("azimuth")
                if row.get("azimuth") is not None:
                    directed_n += 1
                if dip is None or dip_direction is None:
                    missing.append({"type": row.get("type"), "depth": row.get("depth")})
                    continue
            axes.append(plane_axis_from_dip_direction(float(dip), float(dip_direction)))
        if missing:
            raise BlockedByData([{
                "code": "FIELD_SEMANTICS_UNIFICATION_FAILED",
                "detail": "%s has %d natural picks without the required undirected strike+dip fields; first=%r"
                % (spec["unit_id"], len(missing), missing[:3]),
            }])
        wells[spec["unit_id"]] = {
            "unit_id": spec["unit_id"],
            "dataset_id": spec["dataset_id"],
            "role": spec["role"],
            "source_path": asset.get("local_path"),
            "sha256": asset.get("sha256_registry"),
            "mode": spec["mode"],
            "source_mode": "XLSX_STRIKE_DIP" if spec["mode"] == "STRIKE_DIP" else "LAS_DIP_AZIMUTH_DIP",
            "field_semantics": (
                "Azimuth=strike; Dip=dip; Dip Direction is optional and not used for the undirected axis"
                if spec["mode"] == "STRIKE_DIP"
                else "Azimuth=true dip azimuth; Dip_TRU=dip; converted to an undirected axis (up to sign)"
            ),
            "directed_field": spec["directed_field"],
            "directed_n": directed_n,
            "directed_total": len(natural),
            "n_natural_picks": len(natural),
            "n_orientation_axes": len(axes),
            "orientation_convention": ORIENTATION_CONVENTION,
            "axis_sign_convention": AXIS_SIGN_CONVENTION,
            "axes": [[float(c) for c in axis] for axis in axes],
        }
    return {
        "schema": "obs_equiv.real_censoring_holdout.axes.v1",
        "orientation_convention": ORIENTATION_CONVENTION,
        "axis_sign_convention": AXIS_SIGN_CONVENTION,
        "wells": wells,
    }


def build_real_manifest(
    registry_path: str | Path,
    license_evidence_path: str | Path,
    *,
    project_root: str | Path | None = None,
    evaluation_time: str | None = None,
    metric_artifact: str = "results/obs_equiv_20261024/real_censoring_holdout_20261024/per_well_metrics.json",
) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    audit = audit_local_assets(registry_path, license_evidence_path, project_root=root)
    candidate = audit["candidate"]
    if not candidate.get("supports_cross_well_observation_holdout"):
        raise BlockedByData(audit["blockers"])
    assets = {asset["dataset_id"]: asset for asset in audit["assets"]}
    axes_data = extract_orientation_axes(
        registry_path, license_evidence_path, project_root=root, audit=audit
    )

    def _unit(unit: Mapping[str, Any]) -> dict[str, Any]:
        dataset_id = unit.get("dataset_id")
        asset = assets.get(dataset_id)
        if asset is None:
            raise RealHoldoutError("audited asset not found: %r" % (dataset_id,))
        probe = asset.get("probe") or {}
        type_counts = probe.get("types") or {}
        included = sorted(
            name for name in type_counts
            if _classify_pick(name)[2]
        )
        n_included = sum(int(type_counts[name]) for name in included)
        if not included:
            raise RealHoldoutError("no natural picks for %r" % (dataset_id,))
        if n_included != int(probe.get("n_natural", -1)):
            raise RealHoldoutError(
                "natural pick count mismatch for %r: types=%d probe=%r"
                % (dataset_id, n_included, probe.get("n_natural"))
            )
        depth_range = probe.get("depth_md_ft")
        if not depth_range:
            raise RealHoldoutError("no MD depth range for %r" % (dataset_id,))
        axis_well = axes_data["wells"][unit["unit_id"]]
        return {
            "unit_id": unit["unit_id"],
            "unit_type": "well",
            "n_observations": n_included,
            "source_path": asset["local_path"],
            "sha256": asset["sha256_registry"],
            "depth_range_ft": [float(depth_range[0]), float(depth_range[1])],
            "types": included,
            "fracture_filter": (
                "type contains 'fracture' and not 'induced'; "
                "excludes Bed_Boundary/Fault/MicroFault/True Vertical Feature"
            ),
            "orientation_convention": ORIENTATION_CONVENTION,
            "n_orientation_axes": int(axis_well["n_orientation_axes"]),
            "directed_dip_direction_n": int(axis_well["directed_n"]),
            "directed_dip_direction_total": int(axis_well["directed_total"]),
            "source_mode": axis_well["source_mode"],
        }

    observed_units = [_unit(unit) for unit in candidate["observed_units"]]
    held_out_units = [_unit(candidate["held_out_unit"])]
    source_assets = []
    for unit in list(candidate["observed_units"]) + [candidate["held_out_unit"]]:
        asset = assets[unit["dataset_id"]]
        source_assets.append({
            "dataset_id": asset["dataset_id"],
            "source_url": asset["source_url"],
            "sha256": asset["sha256_registry"],
            "license": asset["license"],
            "license_status": asset["license_status"],
            "declared_truth_level": asset["truth_level_candidate"],
        })
    try:
        license_display = str(Path(license_evidence_path).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        license_display = str(license_evidence_path)
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "group_type": GROUP_REAL_CENSORING_HOLDOUT,
        "truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
        "task_family": TASK_FAMILY,
        "site_id": candidate["site_id"],
        "site_scope": SITE_SCOPE,
        "sample_unit": SAMPLE_UNIT,
        "orientation_convention": ORIENTATION_CONVENTION,
        "metric_artifact": metric_artifact,
        "metric_family": METRIC_FAMILY,
        "fixture": False,
        "real_data": True,
        "observed_units": observed_units,
        "held_out_units": held_out_units,
        "source_assets": source_assets,
        "license": "CC-BY-4.0",
        "license_status": "PUBLIC_VERIFIED",
        "license_evidence": license_display,
        "observation_protocol": {
            "modality": "BOREHOLE_FMI_PICKS",
            "coordinate_frame": "WELL_DEPTH_AND_ORIENTATION",
            "truth_level_rationale": (
                "FMI picks are expert interpretations along each well; the held-out well's picks "
                "are observations, not a full 3D structure truth or a connectivity label."
            ),
            "enumeration": "EXHAUSTIVE_NATURAL_PICK_ROWS_FROM_LOCAL_LAS_XLSX",
            "depth_reference": "MD_FT",
            "no_3d_geometry_claim": True,
            "no_connectivity_label": True,
            "orientation_convention": ORIENTATION_CONVENTION,
            "axis_sign_convention": AXIS_SIGN_CONVENTION,
            "metric_family": METRIC_FAMILY,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "evaluation_time": evaluation_time or utc_now(),
        "allowed_claims": [ALLOWED_CLAIM],
        "forbidden_claims": list(REQUIRED_FORBIDDEN_CLAIMS),
        "claim_scope": {
            "allowed_claim": UNDIRECTED_CLAIM,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "counts": {
            "observed": sum(unit["n_observations"] for unit in observed_units),
            "held_out": sum(unit["n_observations"] for unit in held_out_units),
            "total": sum(unit["n_observations"] for unit in observed_units + held_out_units),
            "n_observed_units": len(observed_units),
            "n_held_out_units": len(held_out_units),
            "n_orientation_axes": sum(unit["n_orientation_axes"] for unit in observed_units + held_out_units),
            "n_directed_dip_direction": sum(unit["directed_dip_direction_n"] for unit in observed_units + held_out_units),
        },
        "validation": {
            "schema": SCHEMA_VALIDATION,
            "real_data": True,
            "fixture_only": False,
        },
        "notes": (
            "Picks-only cross-well observation holdout. 16A(78)-32 + 16B(78)-32 observed; "
            "58-32 held out as a whole well. All three wells use the same UNDIRECTED_STRIKE_DIP "
            "axis (normal up to sign). 16A has Dip Direction only for 48/560 natural picks; "
            "the axis uses strike+dip for all 560. 16B/58-32 use true dip azimuth + dip, converted "
            "to the same undirected axis. Single site (FORGE Milford). No disk geometry, no query "
            "labels, no reference posterior, no connectivity metric, no real generalization claim."
        ),
    }
    validation = validate_real_holdout_manifest(manifest, require_real=True)
    if not validation["pass"]:
        raise RealHoldoutError("real manifest validation failed: %r" % (validation["violations"][:6],))
    return {"manifest": manifest, "validation": validation, "axes": axes_data}


def _angle_stats(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    array = np.clip(array, 0.0, 90.0)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _axis_valid(axis: Any) -> bool:
    if not isinstance(axis, list) or len(axis) != 3:
        return False
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
        for value in axis
    ):
        return False
    x, y, z = [float(value) for value in axis]
    if abs(math.sqrt(x * x + y * y + z * z) - 1.0) > 1e-9:
        return False
    if z < -1e-12:
        return False
    if abs(z) <= 1e-12 and (x < -1e-12 or (abs(x) <= 1e-12 and y < -1e-12)):
        return False
    return True


def _stats_valid(stats: Any, expected_n: int) -> bool:
    if not isinstance(stats, Mapping) or stats.get("n") != expected_n:
        return False
    for key in ("mean", "median", "p90", "min", "max"):
        value = stats.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 90.0:
            return False
    return True


def _scan_metric_forbidden_keys(value: Any, path: str = "$") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_METRIC_KEYS:
                hits.append({"path": "%s.%s" % (path, key)})
            hits.extend(_scan_metric_forbidden_keys(child, "%s.%s" % (path, key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            hits.extend(_scan_metric_forbidden_keys(child, "%s[%d]" % (path, index)))
    return hits


def _check_semantic_flag(
    result: dict[str, Any],
    name: str,
    holder: Any,
    key: str,
    expected: Any,
) -> None:
    """Fail closed on a missing or contradictory machine-readable semantic flag."""
    if not isinstance(holder, Mapping):
        _check(result, name, False, "METRIC_SEMANTICS_MISSING", "metric block missing for flag %r" % key)
        return
    if key not in holder:
        _check(result, name, False, "METRIC_SEMANTICS_MISSING", "missing semantic flag %r" % key)
        return
    value = holder.get(key)
    ok = (value is expected) if isinstance(expected, bool) else (value == expected)
    _check(
        result,
        name,
        ok,
        "METRIC_SEMANTICS_CONTRADICTION",
        "semantic flag %r must be %r, got %r" % (key, expected, value),
    )


def _check_well_metric_semantics(result: dict[str, Any], well: Mapping[str, Any], prefix: str) -> None:
    mean_block = well.get("to_observed_axial_mean_acute_angle_deg")
    nearest_block = well.get("to_nearest_observed_axis_acute_angle_deg")
    _check(result, "%s.semantics.mean_object" % prefix, isinstance(mean_block, Mapping), "METRIC_SEMANTICS_MISSING", "to_observed_axial_mean block must be an object carrying semantics")
    _check(result, "%s.semantics.nearest_object" % prefix, isinstance(nearest_block, Mapping), "METRIC_SEMANTICS_MISSING", "to_nearest_observed_axis block must be an object carrying semantics")
    for key, expected in (
        ("metric_role", METRIC_ROLE_OBSERVED_PREDICTOR),
        ("metric_id", METRIC_ID_OBSERVED_MEAN),
        ("deployable_predictor", True),
        ("uses_held_out_target_for_selection", False),
    ):
        _check_semantic_flag(result, "%s.semantics.mean.%s" % (prefix, key), mean_block, key, expected)
    mean_selection = mean_block.get("selection_inputs") if isinstance(mean_block, Mapping) else None
    _check(result, "%s.semantics.mean.selection_inputs" % prefix, mean_selection == ["OBSERVED_PICK_AXES"], "METRIC_SEMANTICS_CONTRADICTION", "observed-mean selection_inputs must be ['OBSERVED_PICK_AXES']")
    for key, expected in (
        ("metric_role", METRIC_ROLE_TARGET_DEPENDENT),
        ("metric_id", METRIC_ID_NEAREST_AXIS),
        ("deployable_predictor", False),
        ("uses_held_out_target_for_selection", True),
        ("not_prediction_error", True),
        ("not_a_label_leak", True),
        ("predictor_is_legacy_alias", True),
        ("predictor_semantics", PREDICTOR_SEMANTICS_NOT_DEPLOYABLE),
    ):
        _check_semantic_flag(result, "%s.semantics.nearest.%s" % (prefix, key), nearest_block, key, expected)
    nearest_selection = nearest_block.get("selection_inputs") if isinstance(nearest_block, Mapping) else None
    _check(result, "%s.semantics.nearest.selection_inputs" % prefix, isinstance(nearest_selection, list) and set(nearest_selection) == {"HELD_OUT_TARGET_AXES", "OBSERVED_PICK_AXES"}, "METRIC_SEMANTICS_CONTRADICTION", "nearest selection_inputs must be HELD_OUT_TARGET_AXES + OBSERVED_PICK_AXES")
    legacy_predictor = nearest_block.get("predictor") if isinstance(nearest_block, Mapping) else None
    _check(result, "%s.semantics.nearest.legacy_predictor_key" % prefix, isinstance(legacy_predictor, str) and bool(legacy_predictor), "METRIC_SEMANTICS_MISSING", "legacy 'predictor' key must be kept for backward compatibility")


def _check_aggregate_metric_roles(result: dict[str, Any], aggregate: Mapping[str, Any]) -> None:
    roles = aggregate.get("metric_roles")
    _check(result, "aggregate.metric_roles_object", isinstance(roles, Mapping), "METRIC_SEMANTICS_MISSING", "aggregate.metric_roles must be an object")
    if isinstance(roles, Mapping):
        _check(result, "aggregate.metric_roles.observed_mean", roles.get(METRIC_ID_OBSERVED_MEAN) == METRIC_ROLE_OBSERVED_PREDICTOR, "METRIC_SEMANTICS_CONTRADICTION", "metric_roles must mark the observed-only mean as %r" % METRIC_ROLE_OBSERVED_PREDICTOR)
        _check(result, "aggregate.metric_roles.nearest", roles.get(METRIC_ID_NEAREST_AXIS) == METRIC_ROLE_TARGET_DEPENDENT, "METRIC_SEMANTICS_CONTRADICTION", "metric_roles must mark the nearest-axis metric as %r" % METRIC_ROLE_TARGET_DEPENDENT)
        _check(result, "aggregate.metric_roles.complete", set(roles) == {METRIC_ID_OBSERVED_MEAN, METRIC_ID_NEAREST_AXIS}, "METRIC_SEMANTICS_CONTRADICTION", "metric_roles must map exactly the two comparative metrics")
    deployable = aggregate.get("deployable_predictor_metric_ids")
    _check(result, "aggregate.deployable_predictor_metric_ids", isinstance(deployable, list) and list(deployable) == [METRIC_ID_OBSERVED_MEAN], "METRIC_SEMANTICS_CONTRADICTION", "deployable_predictor_metric_ids must be exactly [%r]" % METRIC_ID_OBSERVED_MEAN)
    _check(result, "aggregate.nearest_is_not_deployable", METRIC_ID_NEAREST_AXIS not in (deployable if isinstance(deployable, list) else [METRIC_ID_NEAREST_AXIS]), "METRIC_SEMANTICS_CONTRADICTION", "the target-dependent nearest-axis metric must never be listed as a deployable predictor")


def build_per_well_metrics(
    registry_path: str | Path,
    license_evidence_path: str | Path,
    *,
    axes_data: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    evaluation_time: str | None = None,
    metric_artifact: str = "results/obs_equiv_20261024/real_censoring_holdout_20261024/per_well_metrics.json",
) -> dict[str, Any]:
    import numpy as np

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    axes_data = axes_data or extract_orientation_axes(
        registry_path, license_evidence_path, project_root=root
    )
    wells: dict[str, Any] = {}
    observed_ids: list[str] = []
    held_out_ids: list[str] = []
    for unit_id, data in axes_data["wells"].items():
        axes = np.asarray(data["axes"], dtype=float)
        if axes.ndim != 2 or axes.shape[1] != 3 or axes.shape[0] == 0:
            raise RealHoldoutError("invalid axes for %r" % (unit_id,))
        norms = np.linalg.norm(axes, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-9):
            raise RealHoldoutError("non-unit axis for %r" % (unit_id,))
        matrix = axes.T @ axes / axes.shape[0]
        eigvals, eigvecs = np.linalg.eigh(matrix)
        mean_axis = _canonical_axis(tuple(float(value) for value in eigvecs[:, -1]))
        angles = np.degrees(np.arccos(np.clip(np.abs(axes @ np.asarray(mean_axis)), 0.0, 1.0)))
        within = _angle_stats(angles)
        used_for_axis = (
            "not_used_optional" if data["mode"] == "STRIKE_DIP" else "converted_to_undirected_axis"
        )
        well = {
            "unit_id": unit_id,
            "dataset_id": data["dataset_id"],
            "role": data["role"],
            "source_path": data["source_path"],
            "sha256": data["sha256"],
            "source_mode": data["source_mode"],
            "field_semantics": data["field_semantics"],
            "n_natural_picks": int(data["n_natural_picks"]),
            "n_orientation_axes": int(data["n_orientation_axes"]),
            "orientation_convention": ORIENTATION_CONVENTION,
            "axis_sign_convention": AXIS_SIGN_CONVENTION,
            "directed_dip_direction_coverage": {
                "field": data["directed_field"],
                "n": int(data["directed_n"]),
                "total": int(data["directed_total"]),
                "used_for_axis": used_for_axis,
            },
            "axial_mean_axis": [float(value) for value in mean_axis],
            "axial_resultant_strength": float(eigvals[-1]),
            "within_well_acute_angle_to_mean_deg": within,
            "no_connectivity_metric": True,
            "no_3d_truth": True,
        }
        wells[unit_id] = well
        if data["role"] == "observed":
            observed_ids.append(unit_id)
        elif data["role"] == "held_out":
            held_out_ids.append(unit_id)
    if not observed_ids or not held_out_ids:
        raise BlockedByData([{"code": "OBSERVED_OR_HELD_OUT_MISSING", "detail": "metrics require both roles"}])
    observed_ids.sort()
    held_out_ids.sort()
    observed_means = np.asarray([wells[unit_id]["axial_mean_axis"] for unit_id in observed_ids], dtype=float)
    raw_prediction = observed_means.sum(axis=0)
    norm = float(np.linalg.norm(raw_prediction))
    degenerate = norm < 1e-9
    if degenerate:
        predicted_axis = _canonical_axis(tuple(observed_means[0]))
    else:
        predicted_axis = _canonical_axis(tuple(raw_prediction / norm))
    observed_axes = np.vstack([
        np.asarray(axes_data["wells"][unit_id]["axes"], dtype=float) for unit_id in observed_ids
    ])
    held_out_well_id = held_out_ids[0]
    held_axes = np.asarray(axes_data["wells"][held_out_well_id]["axes"], dtype=float)
    angles_to_prediction = np.degrees(np.arccos(np.clip(
        np.abs(held_axes @ np.asarray(predicted_axis)), 0.0, 1.0
    )))
    nearest_dots = np.abs(held_axes @ observed_axes.T)
    nearest_angles = np.degrees(np.arccos(np.clip(nearest_dots.max(axis=1), 0.0, 1.0)))
    wells[held_out_well_id]["to_observed_axial_mean_acute_angle_deg"] = {
        **_angle_stats(angles_to_prediction),
        "predictor": "OBSERVED_EQUAL_WELL_AXIAL_MEAN",
        "metric_id": METRIC_ID_OBSERVED_MEAN,
        "metric_role": METRIC_ROLE_OBSERVED_PREDICTOR,
        "deployable_predictor": True,
        "uses_held_out_target_for_selection": False,
        "selection_inputs": ["OBSERVED_PICK_AXES"],
        "observed_well_ids": list(observed_ids),
        "observed_mean_axis": [float(value) for value in predicted_axis],
        "observed_mean_axis_degenerate": bool(degenerate),
        "descriptive_only": True,
        "no_connectivity_metric": True,
        "no_3d_truth": True,
    }
    wells[held_out_well_id]["to_nearest_observed_axis_acute_angle_deg"] = {
        **_angle_stats(nearest_angles),
        "predictor": "NEAREST_OBSERVED_PICK_AXIS",
        "predictor_is_legacy_alias": True,
        "predictor_semantics": PREDICTOR_SEMANTICS_NOT_DEPLOYABLE,
        "metric_id": METRIC_ID_NEAREST_AXIS,
        "metric_role": METRIC_ROLE_TARGET_DEPENDENT,
        "uses_held_out_target_for_selection": True,
        "selection_inputs": ["HELD_OUT_TARGET_AXES", "OBSERVED_PICK_AXES"],
        "deployable_predictor": False,
        "not_prediction_error": True,
        "not_a_label_leak": True,
        "observed_well_ids": list(observed_ids),
        "density_dependent": True,
        "descriptive_only": True,
        "no_connectivity_metric": True,
        "no_3d_truth": True,
    }
    well_level_mean = sum(
        well["within_well_acute_angle_to_mean_deg"]["mean"] for well in wells.values()
    ) / len(wells)
    metrics = {
        "schema": SCHEMA_METRICS,
        "truth_level": TRUTH_EXPERT_BOREHOLE_PICKS,
        "site_scope": SITE_SCOPE,
        "sample_unit": SAMPLE_UNIT,
        "orientation_convention": ORIENTATION_CONVENTION,
        "axis_sign_convention": AXIS_SIGN_CONVENTION,
        "metric_family": METRIC_FAMILY,
        "observation_protocol": {
            "modality": "BOREHOLE_FMI_PICKS",
            "coordinate_frame": "WELL_DEPTH_AND_ORIENTATION",
            "truth_level_rationale": (
                "FMI picks are expert interpretations; all wells are converted to the same "
                "undirected plane axis (normal up to sign). The held-out well's picks are "
                "observations, not full 3D structure truth."
            ),
            "enumeration": "EXHAUSTIVE_NATURAL_PICK_AXES_PER_WELL",
            "depth_reference": "MD_FT",
            "no_3d_geometry_claim": True,
            "no_connectivity_label": True,
            "orientation_convention": ORIENTATION_CONVENTION,
            "axis_sign_convention": AXIS_SIGN_CONVENTION,
            "metric_family": METRIC_FAMILY,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "evaluation_time": evaluation_time or utc_now(),
        "wells": wells,
        "aggregate": {
            "n_wells": len(wells),
            "n_observed_wells": len(observed_ids),
            "n_held_out_wells": len(held_out_ids),
            "sample_unit": SAMPLE_UNIT,
            "well_level_unweighted_mean_within_well_acute_angle_deg": float(well_level_mean),
            "held_out_to_observed_axial_mean_acute_angle_deg": dict(
                wells[held_out_well_id]["to_observed_axial_mean_acute_angle_deg"]
            ),
            "held_out_to_nearest_observed_axis_acute_angle_deg": dict(
                wells[held_out_well_id]["to_nearest_observed_axis_acute_angle_deg"]
            ),
            "metric_roles": {
                METRIC_ID_OBSERVED_MEAN: METRIC_ROLE_OBSERVED_PREDICTOR,
                METRIC_ID_NEAREST_AXIS: METRIC_ROLE_TARGET_DEPENDENT,
            },
            "deployable_predictor_metric_ids": [METRIC_ID_OBSERVED_MEAN],
            "non_predictor_metric_ids": [METRIC_ID_NEAREST_AXIS],
            "n_picks_not_independent": True,
            "descriptive_only": True,
            "inferential_unit": SAMPLE_UNIT,
            "no_confidence_interval": True,
            "no_connectivity_metric": True,
            "no_3d_truth": True,
        },
        "claim_scope": {
            "allowed_claim": UNDIRECTED_CLAIM,
            "directed_dip_direction_metric": False,
            "connectivity_metric": False,
            "three_d_truth": False,
            "real_generalization": False,
        },
        "metric_artifact": metric_artifact,
    }
    validation = validate_per_well_metrics(metrics, require_real=True)
    if not validation["pass"]:
        raise RealHoldoutError("per-well metrics validation failed: %r" % (validation["violations"][:6],))
    return {"metrics": metrics, "validation": validation}


def validate_per_well_metrics(
    metrics: Mapping[str, Any],
    *,
    require_real: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "obs_equiv.real_censoring_holdout.metrics_validation.v1",
        "pass": False,
        "checks": {},
        "n_checks": 0,
        "violations": [],
    }
    if not isinstance(metrics, Mapping):
        result["violations"].append({"code": "METRICS_NOT_OBJECT", "detail": type(metrics).__name__})
        result["enumerated_checks"] = result["n_checks"]
        return result
    wells = metrics.get("wells")
    if not isinstance(wells, Mapping) or not wells:
        raise SystemExit("EMPTY_METRIC_WELL_SET: per-well metrics require wells")
    required = (
        "schema", "truth_level", "site_scope", "sample_unit", "orientation_convention",
        "axis_sign_convention", "metric_family", "observation_protocol", "evaluation_time",
        "wells", "aggregate", "claim_scope", "metric_artifact",
    )
    for field in required:
        _check(result, "field:%s" % field, field in metrics, "FIELD_MISSING", "missing required field %r" % field)
    _check(result, "schema", metrics.get("schema") == SCHEMA_METRICS, "SCHEMA_MISMATCH", "metrics schema mismatch")
    _check(result, "truth_level", metrics.get("truth_level") in REAL_TRUTH_LEVELS, "TRUTH_LEVEL_FORBIDDEN", "truth_level must be real")
    _check(result, "site_scope", isinstance(metrics.get("site_scope"), str) and bool(metrics.get("site_scope")), "SITE_SCOPE_MISSING", "site_scope must be non-empty")
    _check(result, "sample_unit", metrics.get("sample_unit") == SAMPLE_UNIT, "SAMPLE_UNIT", "sample_unit must be %r" % SAMPLE_UNIT)
    _check(result, "orientation_convention", metrics.get("orientation_convention") == ORIENTATION_CONVENTION, "ORIENTATION_CONVENTION", "orientation_convention mismatch")
    _check(result, "axis_sign_convention", metrics.get("axis_sign_convention") == AXIS_SIGN_CONVENTION, "AXIS_SIGN", "axis_sign_convention mismatch")
    _check(result, "metric_family", metrics.get("metric_family") == METRIC_FAMILY, "METRIC_FAMILY", "metric_family mismatch")
    if require_real:
        _check(result, "real_site_scope", metrics.get("site_scope") == SITE_SCOPE, "SITE_SCOPE_MISMATCH", "real metrics site_scope mismatch")
        _check(result, "real_truth_level", metrics.get("truth_level") == TRUTH_EXPERT_BOREHOLE_PICKS, "TRUTH_LEVEL", "real metrics truth_level must be EXPERT_BOREHOLE_PICKS")
    observed_ids = []
    held_out_ids = []
    for unit_id, well in wells.items():
        prefix = "wells[%s]" % unit_id
        if not isinstance(well, Mapping):
            _check(result, prefix, False, "WELL_NOT_OBJECT", "well must be an object")
            continue
        role = well.get("role")
        _check(result, "%s.role" % prefix, role in ("observed", "held_out"), "WELL_ROLE", "role must be observed/held_out")
        if role == "observed":
            observed_ids.append(unit_id)
        elif role == "held_out":
            held_out_ids.append(unit_id)
        n = well.get("n_natural_picks")
        n_axes = well.get("n_orientation_axes")
        _check(result, "%s.n_natural_picks" % prefix, isinstance(n, int) and not isinstance(n, bool) and n > 0, "WELL_COUNT", "n_natural_picks must be positive")
        _check(result, "%s.n_orientation_axes" % prefix, isinstance(n_axes, int) and not isinstance(n_axes, bool) and n_axes == n, "WELL_AXIS_COUNT", "n_orientation_axes must equal n_natural_picks")
        _check(result, "%s.orientation_convention" % prefix, well.get("orientation_convention") == ORIENTATION_CONVENTION, "WELL_ORIENTATION_CONVENTION", "orientation_convention mismatch")
        coverage = well.get("directed_dip_direction_coverage")
        _check(result, "%s.coverage_object" % prefix, isinstance(coverage, Mapping), "WELL_COVERAGE", "directed coverage must be an object")
        if isinstance(coverage, Mapping):
            _check(result, "%s.coverage.total" % prefix, coverage.get("total") == n, "WELL_COVERAGE", "coverage.total must equal n_natural_picks")
            cov_n = coverage.get("n")
            _check(result, "%s.coverage.n" % prefix, isinstance(cov_n, int) and not isinstance(cov_n, bool) and isinstance(n, int) and 0 <= cov_n <= n, "WELL_COVERAGE", "coverage.n must be in [0,n_natural_picks]")
            _check(result, "%s.coverage.field" % prefix, isinstance(coverage.get("field"), str) and bool(coverage.get("field")), "WELL_COVERAGE", "coverage.field must be non-empty")
            _check(result, "%s.coverage.used" % prefix, coverage.get("used_for_axis") in ("not_used_optional", "converted_to_undirected_axis"), "WELL_COVERAGE", "coverage.used_for_axis invalid")
        _check(result, "%s.axis" % prefix, _axis_valid(well.get("axial_mean_axis")), "WELL_AXIS", "axial_mean_axis must be canonical unit axis")
        strength = well.get("axial_resultant_strength")
        _check(result, "%s.strength" % prefix, isinstance(strength, (int, float)) and not isinstance(strength, bool) and math.isfinite(float(strength)) and (1.0 / 3.0 - 1e-9) <= float(strength) <= 1.0 + 1e-9, "WELL_STRENGTH", "axial_resultant_strength must be in [1/3,1]")
        _check(result, "%s.within_stats" % prefix, _stats_valid(well.get("within_well_acute_angle_to_mean_deg"), n if isinstance(n, int) else -1), "WELL_STATS", "within-well stats invalid")
        _check(result, "%s.no_connectivity" % prefix, well.get("no_connectivity_metric") is True, "WELL_FLAG", "no_connectivity_metric must be true")
        _check(result, "%s.no_3d" % prefix, well.get("no_3d_truth") is True, "WELL_FLAG", "no_3d_truth must be true")
        if role == "held_out":
            _check(result, "%s.to_mean" % prefix, _stats_valid(well.get("to_observed_axial_mean_acute_angle_deg"), n if isinstance(n, int) else -1), "HELD_OUT_STATS", "held-out to-observed-mean stats invalid")
            _check(result, "%s.to_nearest" % prefix, _stats_valid(well.get("to_nearest_observed_axis_acute_angle_deg"), n if isinstance(n, int) else -1), "HELD_OUT_STATS", "held-out nearest-observed stats invalid")
            _check_well_metric_semantics(result, well, prefix)
    _check(result, "observed_present", bool(observed_ids), "ROLE_MISSING", "at least one observed well required")
    _check(result, "held_out_present", bool(held_out_ids), "ROLE_MISSING", "at least one held-out well required")
    _check(result, "roles_disjoint", not (set(observed_ids) & set(held_out_ids)), "ROLE_OVERLAP", "observed and held-out well IDs overlap")
    aggregate = metrics.get("aggregate")
    _check(result, "aggregate_object", isinstance(aggregate, Mapping), "AGGREGATE", "aggregate must be an object")
    if isinstance(aggregate, Mapping):
        _check(result, "aggregate.n_wells", aggregate.get("n_wells") == len(wells), "AGGREGATE", "n_wells mismatch")
        _check(result, "aggregate.n_observed", aggregate.get("n_observed_wells") == len(observed_ids), "AGGREGATE", "n_observed_wells mismatch")
        _check(result, "aggregate.n_held_out", aggregate.get("n_held_out_wells") == len(held_out_ids), "AGGREGATE", "n_held_out_wells mismatch")
        _check(result, "aggregate.sample_unit", aggregate.get("sample_unit") == SAMPLE_UNIT, "AGGREGATE", "sample_unit mismatch")
        _check(result, "aggregate.n_picks_not_independent", aggregate.get("n_picks_not_independent") is True, "AGGREGATE", "n_picks_not_independent must be true")
        _check(result, "aggregate.descriptive_only", aggregate.get("descriptive_only") is True, "AGGREGATE", "descriptive_only must be true")
        _check(result, "aggregate.inferential_unit", aggregate.get("inferential_unit") == SAMPLE_UNIT, "AGGREGATE", "inferential_unit mismatch")
        _check(result, "aggregate.no_ci", aggregate.get("no_confidence_interval") is True, "AGGREGATE", "no_confidence_interval must be true")
        value = aggregate.get("well_level_unweighted_mean_within_well_acute_angle_deg")
        _check(result, "aggregate.well_mean", isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 90.0, "AGGREGATE", "well-level mean invalid")
        _check_aggregate_metric_roles(result, aggregate)
    protocol = metrics.get("observation_protocol")
    _check(result, "protocol_object", isinstance(protocol, Mapping), "PROTOCOL", "observation_protocol must be an object")
    if isinstance(protocol, Mapping):
        _check(result, "protocol.no_3d", protocol.get("no_3d_geometry_claim") is True, "PROTOCOL", "no_3d_geometry_claim must be true")
        _check(result, "protocol.no_connectivity", protocol.get("no_connectivity_label") is True, "PROTOCOL", "no_connectivity_label must be true")
        _check(result, "protocol.orientation", protocol.get("orientation_convention") == ORIENTATION_CONVENTION, "PROTOCOL", "orientation_convention mismatch")
        _check(result, "protocol.axis_sign", protocol.get("axis_sign_convention") == AXIS_SIGN_CONVENTION, "PROTOCOL", "axis_sign_convention mismatch")
        _check(result, "protocol.metric_family", protocol.get("metric_family") == METRIC_FAMILY, "PROTOCOL", "metric_family mismatch")
        for flag in ("directed_dip_direction_metric", "connectivity_metric", "three_d_truth", "real_generalization"):
            _check(result, "protocol.%s_false" % flag, protocol.get(flag) is False, "PROTOCOL", "%s must be false" % flag)
    claim_scope = metrics.get("claim_scope")
    _check(result, "claim_scope_object", isinstance(claim_scope, Mapping), "CLAIM_SCOPE", "claim_scope must be an object")
    if isinstance(claim_scope, Mapping):
        _check(result, "claim_scope.allowed", claim_scope.get("allowed_claim") == UNDIRECTED_CLAIM, "CLAIM_SCOPE", "allowed_claim must be %r" % UNDIRECTED_CLAIM)
        for flag in ("directed_dip_direction_metric", "connectivity_metric", "three_d_truth", "real_generalization"):
            _check(result, "claim_scope.%s_false" % flag, claim_scope.get(flag) is False, "CLAIM_SCOPE", "%s must be false" % flag)
    forbidden_hits = _scan_metric_forbidden_keys(metrics)
    _check(result, "no_forbidden_metric_keys", not forbidden_hits, "FORBIDDEN_METRIC_KEY", "forbidden metric keys: %r" % forbidden_hits[:8])
    result["pass"] = not result["violations"]
    result["enumerated_checks"] = result["n_checks"]
    return result


__all__ = [
    "SCHEMA_MANIFEST",
    "SCHEMA_AUDIT",
    "SCHEMA_VALIDATION",
    "SCHEMA_METRICS",
    "WATERMARK",
    "GROUP_REAL_CENSORING_HOLDOUT",
    "TRUTH_EXPERT_BOREHOLE_PICKS",
    "TRUTH_HELD_OUT_OBSERVATION",
    "REAL_TRUTH_LEVELS",
    "TASK_FAMILY",
    "ALLOWED_CLAIM",
    "UNDIRECTED_CLAIM",
    "ORIENTATION_CONVENTION",
    "AXIS_SIGN_CONVENTION",
    "SAMPLE_UNIT",
    "SITE_SCOPE",
    "METRIC_FAMILY",
    "METRIC_ID_OBSERVED_MEAN",
    "METRIC_ID_NEAREST_AXIS",
    "METRIC_ROLE_OBSERVED_PREDICTOR",
    "METRIC_ROLE_TARGET_DEPENDENT",
    "PREDICTOR_SEMANTICS_NOT_DEPLOYABLE",
    "REQUIRED_FORBIDDEN_CLAIMS",
    "RealHoldoutError",
    "BlockedByData",
    "utc_now",
    "canonical_json",
    "sha256_json",
    "file_sha256",
    "read_json",
    "write_json_exclusive",
    "probe_las_picks",
    "probe_xlsx_fmi",
    "probe_core_zip",
    "plane_axis_from_dip_direction",
    "extract_orientation_axes",
    "validate_real_holdout_manifest",
    "fixture_manifest",
    "audit_local_assets",
    "build_real_manifest",
    "build_per_well_metrics",
    "validate_per_well_metrics",
]
