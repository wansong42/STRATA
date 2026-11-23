# -*- coding: utf-8 -*-
"""W152 asset registration field contract (machine readable).

The contract is the unlock artefact for W154's portable project: W154 copies or
hash-relocates assets, so it needs to know, field by field, what identifies an
asset, how it is located, how it is fingerprinted, which unit and coordinate
basis it carries, whether it may be redistributed, and what a missing,
replaced or relocated asset means.

Nothing here decides whether a project is acceptable -- it only fixes field
names, types, requiredness and the three resolution states so both sides read
the same bytes the same way.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional

CONTRACT_VERSION = "fractureflow.asset_contract.v1"
ROLES = ("RAW_MEASUREMENT", "DERIVED_DISPLAY", "AUXILIARY_CALIBRATION")
LOCATOR_MODES = ("repo_relative", "project_relative", "external_uri")
UNIT_STATUS = ("KNOWN", "UNKNOWN")
COORDINATE_BASES = ("source_coordinates", "scene_crs", "project_local", "synthetic_local")
RESOLUTION_STATES = ("MISSING", "REPLACED", "RELOCATED")
RESOLUTION_ACTIONS = {
    "MISSING": ("BLOCK_DISPLAY", "REQUEST_RELOCATION", "SHOW_LIMITED_VIEW"),
    "REPLACED": ("BLOCK_DISPLAY", "WARN_AND_BLOCK_SILENT_SUBSTITUTION"),
    "RELOCATED": ("HASH_RELOCATE_OK", "REQUEST_CONFIRMATION"),
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CST = timezone(timedelta(hours=8))


FIELDS: List[Dict[str, Any]] = [
    {
        "name": "asset_id",
        "type": "string",
        "required": True,
        "enum": None,
        "description": "资产身份。项目内唯一，跨重开稳定；不得由显示名派生。",
        "example": "uav_wall_slice__points__sha256_2e788629",
    },
    {
        "name": "role",
        "type": "string",
        "required": True,
        "enum": list(ROLES),
        "description": "资产角色。RAW_MEASUREMENT 是可计量原始件，DERIVED_DISPLAY 只能显示不能当测量。",
        "example": "DERIVED_DISPLAY",
    },
    {
        "name": "source_locator",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "来源定位：mode ∈ {repo_relative, project_relative, external_uri}；value 为相对路径或 URI；redistributable 必须在导入时显式给出，不得默认 true。",
        "example": {
            "mode": "repo_relative",
            "value": "results/w152_general_import/fixtures/uav_wall_slice/points.ply",
            "redistributable": False,
        },
    },
    {
        "name": "fingerprint",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "指纹：algorithm 固定 sha256；value 为 64 位小写十六进制；scope 声明覆盖范围（本项目要求全文件字节）。",
        "example": {
            "algorithm": "sha256",
            "value": "2e788629b7e4eebac1ed3d8edf5823127eb6c7195b3772b49c69cb495b2ceeed",
            "scope": "full_file_bytes",
        },
    },
    {
        "name": "unit",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "单位：declared 为资料声明值；status ∈ {KNOWN, UNKNOWN}；UNKNOWN 时 metric_output_allowed 必须为 false。",
        "example": {"declared": "metre", "status": "KNOWN", "metric_output_allowed": True},
    },
    {
        "name": "coordinate",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "坐标：basis ∈ %s；version 供失效判断；origin 为局部原点；crs 可为 null 但必须显式出现。" % (list(COORDINATE_BASES),),
        "example": {
            "basis": "source_coordinates",
            "version": "scene_source_coordinates_v1",
            "origin": [685.594482421875, 516.4776611328125, 613.4718017578125],
            "crs": None,
        },
    },
    {
        "name": "license",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "许可与使用范围：license_id（SPDX 或 UNVERIFIED）；usage_scope 为文字边界；redistribution_allowed 决定便携包能否自包含复制。",
        "example": {
            "license_id": "CC-BY-4.0",
            "usage_scope": "INTERNAL_DEVELOPMENT_ONLY_NO_REDISTRIBUTION",
            "redistribution_allowed": False,
            "evidence": "data/structural_reconstruction_pilot/observations/jardim_garcia_quarry_uh/observation.json",
        },
    },
    {
        "name": "derivation",
        "type": "object",
        "required": False,
        "enum": None,
        "description": "原始计量资产与显示派生资产的关系：派生件必填，derived_from 指向上级 asset_id；is_measurement 必须为 false 时不得用于计量结论。",
        "example": {
            "derived_from": ["jardim_garcia_uh__raw__sha256_2e788629"],
            "method": "deterministic_index_stride",
            "is_measurement": False,
            "display_only": True,
        },
    },
    {
        "name": "resolution",
        "type": "object",
        "required": True,
        "enum": None,
        "description": "缺失/替换/重定位三态语义：三个键都必须出现，动作取自固定表；禁止静默回退成合成数据。",
        "example": {
            "on_missing": "BLOCK_DISPLAY",
            "on_replaced": "WARN_AND_BLOCK_SILENT_SUBSTITUTION",
            "on_relocated": "HASH_RELOCATE_OK",
        },
    },
]


def _now() -> str:
    return datetime.now(CST).isoformat()


def contract_document(*, generated_at: Optional[str] = None) -> Dict[str, Any]:
    """Return the full machine-readable contract document."""
    required = [field["name"] for field in FIELDS if field["required"]]
    optional = [field["name"] for field in FIELDS if not field["required"]]
    return {
        "schema": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "generated_at": generated_at or _now(),
        "producer": "results/w152_general_import/run_acceptance.py",
        "status": "SUBMITTED_FOR_W154_FIELD_REVIEW",
        "field_count": len(FIELDS),
        "fields": FIELDS,
        "required_fields": required,
        "optional_fields": optional,
        "resolution_states": list(RESOLUTION_STATES),
        "resolution_actions": {key: list(value) for key, value in RESOLUTION_ACTIONS.items()},
        "notes": [
            "本契约只定义字段与三态语义，不判定项目可接受性。",
            "字段合格不等于 W154 解阻：W154 必须逐字段复核并自行出具结论。",
            "redistributable / redistribution_allowed 未显式给出时按 false 处理，不得默认允许。",
        ],
    }


def example_record() -> Dict[str, Any]:
    """One complete record, assembled from the field examples (self-check fixture)."""
    return {field["name"]: json.loads(json.dumps(field["example"], allow_nan=False)) for field in FIELDS}


def build_record(*, asset_id: str, role: str, locator_mode: str, locator_value: str,
                 redistributable: bool, fingerprint: str, fingerprint_scope: str,
                 unit_declared: str, unit_status: str, metric_allowed: bool,
                 coordinate_basis: str, coordinate_version: str, origin: List[float],
                 crs: Any, license_id: str, usage_scope: str, redistribution_allowed: bool,
                 license_evidence: str, resolution: Mapping[str, str],
                 derivation: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "asset_id": asset_id,
        "role": role,
        "source_locator": {
            "mode": locator_mode,
            "value": locator_value,
            "redistributable": bool(redistributable),
        },
        "fingerprint": {
            "algorithm": "sha256",
            "value": fingerprint,
            "scope": fingerprint_scope,
        },
        "unit": {
            "declared": unit_declared,
            "status": unit_status,
            "metric_output_allowed": bool(metric_allowed),
        },
        "coordinate": {
            "basis": coordinate_basis,
            "version": coordinate_version,
            "origin": list(origin),
            "crs": crs,
        },
        "license": {
            "license_id": license_id,
            "usage_scope": usage_scope,
            "redistribution_allowed": bool(redistribution_allowed),
            "evidence": license_evidence,
        },
        "resolution": dict(resolution),
    }
    if derivation is not None:
        record["derivation"] = dict(derivation)
    return record


def validate_record(record: Mapping[str, Any]) -> List[str]:
    """Return every violation; an empty list means the record satisfies the contract."""
    problems: List[str] = []
    if not isinstance(record, Mapping):
        return ["record must be one JSON object"]
    for field in FIELDS:
        if field["required"] and field["name"] not in record:
            problems.append("missing required field: %s" % field["name"])
    if str(record.get("role")) not in ROLES:
        problems.append("role must be one of %s" % (ROLES,))
    locator = record.get("source_locator")
    if not isinstance(locator, Mapping):
        problems.append("source_locator must be an object")
    else:
        if locator.get("mode") not in LOCATOR_MODES:
            problems.append("source_locator.mode must be one of %s" % (LOCATOR_MODES,))
        if not isinstance(locator.get("redistributable"), bool):
            problems.append("source_locator.redistributable must be an explicit boolean")
        if locator.get("mode") in ("repo_relative", "project_relative"):
            value = str(locator.get("value") or "")
            if not value or value.startswith(("/", "\\")) or ".." in value.replace("\\", "/").split("/"):
                problems.append("relative locator must not be absolute or contain '..'")
    fingerprint = record.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        problems.append("fingerprint must be an object")
    else:
        if fingerprint.get("algorithm") != "sha256":
            problems.append("fingerprint.algorithm must be sha256")
        if not SHA256_PATTERN.match(str(fingerprint.get("value") or "")):
            problems.append("fingerprint.value must be 64 lowercase hex characters")
        if fingerprint.get("scope") not in ("full_file_bytes", "declared_subset"):
            problems.append("fingerprint.scope must be full_file_bytes or declared_subset")
    unit = record.get("unit")
    if not isinstance(unit, Mapping):
        problems.append("unit must be an object")
    else:
        if unit.get("status") not in UNIT_STATUS:
            problems.append("unit.status must be KNOWN or UNKNOWN")
        if unit.get("status") == "UNKNOWN" and unit.get("metric_output_allowed") is not False:
            problems.append("unit.status UNKNOWN requires metric_output_allowed=false")
        if not isinstance(unit.get("metric_output_allowed"), bool):
            problems.append("unit.metric_output_allowed must be an explicit boolean")
    coordinate = record.get("coordinate")
    if not isinstance(coordinate, Mapping):
        problems.append("coordinate must be an object")
    else:
        if coordinate.get("basis") not in COORDINATE_BASES:
            problems.append("coordinate.basis must be one of %s" % (COORDINATE_BASES,))
        origin = coordinate.get("origin")
        if not (isinstance(origin, list) and len(origin) == 3):
            problems.append("coordinate.origin must be three numbers")
        if "crs" not in coordinate:
            problems.append("coordinate.crs must be present (null allowed)")
    lic = record.get("license")
    if not isinstance(lic, Mapping):
        problems.append("license must be an object")
    else:
        for key in ("license_id", "usage_scope"):
            if not str(lic.get(key) or "").strip():
                problems.append("license.%s must be non-empty" % key)
        if not isinstance(lic.get("redistribution_allowed"), bool):
            problems.append("license.redistribution_allowed must be an explicit boolean")
    resolution = record.get("resolution")
    if not isinstance(resolution, Mapping):
        problems.append("resolution must be an object")
    else:
        for state, actions in RESOLUTION_ACTIONS.items():
            key = "on_" + state.lower()
            if key not in resolution:
                problems.append("resolution.%s is required" % key)
            elif resolution.get(key) not in actions:
                problems.append("resolution.%s must be one of %s" % (key, actions))
    derivation = record.get("derivation")
    if derivation is not None:
        if not isinstance(derivation, Mapping):
            problems.append("derivation must be an object when present")
        else:
            if not isinstance(derivation.get("derived_from"), list) or not derivation.get("derived_from"):
                problems.append("derivation.derived_from must be a non-empty list")
            if not isinstance(derivation.get("is_measurement"), bool):
                problems.append("derivation.is_measurement must be an explicit boolean")
    elif record.get("role") == "DERIVED_DISPLAY":
        problems.append("role DERIVED_DISPLAY requires a derivation block")
    return problems


def selfcheck() -> Dict[str, Any]:
    """Enumerate the contract and prove the fixture record validates."""
    document = contract_document(generated_at="SELFCHECK_FIXED_TIMESTAMP")
    sample = example_record()
    violations = validate_record(sample)
    missing_examples = [field["name"] for field in FIELDS if field.get("example") is None]
    missing_requiredness = [field["name"] for field in FIELDS if "required" not in field]
    resolved_states = sorted(set(document["resolution_states"]))
    return {
        "schema": "fractureflow.asset_contract.selfcheck.v1",
        "contract_version": CONTRACT_VERSION,
        "measured_at": _now(),
        "field_count": len(FIELDS),
        "required_count": len(document["required_fields"]),
        "optional_count": len(document["optional_fields"]),
        "missing_examples": missing_examples,
        "missing_requiredness": missing_requiredness,
        "resolution_states_enumerated": resolved_states,
        "resolution_state_count": len(resolved_states),
        "sample_record_violations": violations,
        "sample_record_violation_count": len(violations),
        "ok": not violations and not missing_examples and not missing_requiredness,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="W152 asset contract emitter and self-check")
    parser.add_argument("--contract", help="write the contract document to this path")
    parser.add_argument("--selfcheck", help="write the self-check result to this path")
    arguments = parser.parse_args(argv)
    result = selfcheck()
    if arguments.contract:
        with open(arguments.contract, "w", encoding="utf-8") as handle:
            json.dump(contract_document(), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    if arguments.selfcheck:
        with open(arguments.selfcheck, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
