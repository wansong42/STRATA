# -*- coding: utf-8 -*-
"""W154 portable project package (.ffportable) format v1.

A portable project is a directory package that carries everything a reopened
project needs on another path or another machine, without copying measurement
bytes that may not be redistributed:

    <name>.ffportable/
      portable_manifest.json          # self-describing index, written last but one
      PUBLISHED                       # publication marker == manifest body sha256
      project/                        # project bundle snapshot (head + one version)
        head.json
        versions/<tag>/snapshot.json|events.jsonl|manifest.json|support/*.json
      evidence/evidence_ledger.json   # W153 sidecar copied byte for byte (optional)
      evidence/binding.json           # explicit sidecar -> project binding record
      assets/assets.json              # W152 nine-field asset records
      assets/files/<asset_id>/<name>  # byte copies, SELF_CONTAINED mode only
      coordinates/coordinate_basis.json
      licenses/licenses.json
      README.txt
      RECOVERY.md
      state/                          # mutable, never part of the file index
        relocation_audit.jsonl        # append-only hash relocation audit

Design rules enforced here rather than documented only:

* the package is fully built in a hidden staging directory on the same volume,
  verified there, and published by one os.rename; a failed or cancelled export
  never leaves a directory that could be mistaken for a formal result;
* every hash is the SHA-256 of the whole file bytes;
* JSON is parsed strictly (parse_constant rejects NaN/Infinity, then a finite
  walk rejects overflowed floats); no pickle, no exec/eval, no import of
  anything inside the package;
* package-relative paths must be plain relative POSIX paths; symlinks and
  reparse points are never followed; absolute paths stored in asset records are
  relocation hints for files outside the package and are never used to read
  package content;
* REFERENCE_ONLY (default for real point clouds and ROI) stores locator plus
  full-file SHA-256 and refuses to copy measurement bytes; SELF_CONTAINED only
  accepts assets whose licence explicitly allows redistribution and whose role
  is not a measurement, and raises SELF_CONTAINED_FORBIDDEN_ASSET instead of
  silently downgrading.

Open states are explicit and never aliased: FULL, RESTRICTED, REPLACED,
RELOCATED, NEEDS_RECOMPUTE, CORRUPT. REPLACED (hash differs) and CORRUPT are
non-zero exits; the remaining states exit 0 with a status callers must read
(evidence.current is never true unless the sidecar binding verifies).

CLI: python -m fractureflow.surface_product.portable_project <command>

    export  --project <bundle.ffsrf> --out <name.ffportable> [--mode MODE]
    open    --package <name.ffportable> [--repo-root DIR] [--source-root DIR]...
    verify  --package <name.ffportable>
    relocate --package <name.ffportable> --asset-id ID --path FILE
    export-delivery --package <name.ffportable> --out DIR

Test-only hook: W154_TEST_FAULT=after_staging|before_publish (default off)
injects a failure at the named point so the recovery matrix can prove that no
formal package appears.  Production paths never read it.
"""
from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import asset_contract
from . import evidence_ledger as eledger
from .contracts import ProjectSnapshot, file_sha256
from .project_store import ProjectStore, ProjectStoreError

# -- format ------------------------------------------------------------------

FORMAT_VERSION = "1.1.0"
READABLE_MAJOR = "1"
PORTABLE_SCHEMA = "fractureflow.surface_product.portable_project.v1"
MANIFEST_NAME = "portable_manifest.json"
PUBLICATION_MARKER = "PUBLISHED"
EXPORT_SCHEMA = "fractureflow.surface_product.portable_export.v1"
OPEN_SCHEMA = "fractureflow.surface_product.portable_open.v1"
VERIFY_SCHEMA = "fractureflow.surface_product.portable_verify.v1"
RELOCATE_SCHEMA = "fractureflow.surface_product.portable_relocate.v1"
DELIVERY_SCHEMA = "fractureflow.surface_product.portable_delivery.v1"
ERROR_SCHEMA = "fractureflow.surface_product.portable_error.v1"
ASSETS_SCHEMA = "fractureflow.surface_product.portable_assets.v1"
COORDINATES_SCHEMA = "fractureflow.surface_product.portable_coordinates.v1"
LICENSES_SCHEMA = "fractureflow.surface_product.portable_licenses.v1"
BINDING_SCHEMA = "fractureflow.surface_product.portable_evidence_binding.v1"
AUDIT_SCHEMA = "fractureflow.surface_product.portable_relocation_audit.v1"
SOURCE_INTEGRITY_SCHEMA = "fractureflow.surface_product.portable_source_integrity.v1"
GENERATOR_NAME = "fractureflow.surface_product.portable_project"
GENERATOR_VERSION = "1.1.0"

REFERENCE_ONLY = "REFERENCE_ONLY"
SELF_CONTAINED = "SELF_CONTAINED"
DISTRIBUTION_MODES = (REFERENCE_ONLY, SELF_CONTAINED)

STATUS_FULL = "FULL"
STATUS_RESTRICTED = "RESTRICTED"
STATUS_REPLACED = "REPLACED"
STATUS_RELOCATED = "RELOCATED"
STATUS_NEEDS_RECOMPUTE = "NEEDS_RECOMPUTE"
STATUS_CORRUPT = "CORRUPT"
STATUS_FULL_WITH_WARNINGS = "FULL_WITH_WARNINGS"
OPEN_STATUSES = (STATUS_FULL, STATUS_FULL_WITH_WARNINGS, STATUS_RESTRICTED,
                 STATUS_REPLACED, STATUS_RELOCATED, STATUS_NEEDS_RECOMPUTE,
                 STATUS_CORRUPT)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISSING = 3
EXIT_HASH_MISMATCH = 4
EXIT_CORRUPT = 5
EXIT_FORBIDDEN = 6
EXIT_UNSAFE_PATH = 7
EXIT_RESOURCE_LIMIT = 8
EXIT_CONFLICT = 9
EXIT_FAULT = 10

ERROR_EXIT_CODES: Dict[str, int] = {
    "USAGE_ERROR": EXIT_USAGE,
    "PACKAGE_NOT_FOUND": EXIT_MISSING,
    "PROJECT_BUNDLE_NOT_FOUND": EXIT_MISSING,
    "ASSET_MISSING": EXIT_MISSING,
    "RELOCATE_SOURCE_MISSING": EXIT_MISSING,
    "HASH_MISMATCH": EXIT_HASH_MISMATCH,
    "ASSET_HASH_MISMATCH": EXIT_HASH_MISMATCH,
    "ASSET_DRIFT_NOT_ACCEPTABLE": EXIT_HASH_MISMATCH,
    "ASSET_DRIFT_NOT_APPLICABLE": EXIT_HASH_MISMATCH,
    "DRIFT_REASON_REQUIRED": EXIT_USAGE,
    "SOURCE_INTEGRITY_INVALID": EXIT_CORRUPT,
    "RELOCATE_HASH_MISMATCH": EXIT_HASH_MISMATCH,
    "BUNDLE_HASH_MISMATCH": EXIT_HASH_MISMATCH,
    "MANIFEST_BODY_HASH_MISMATCH": EXIT_CORRUPT,
    "MANIFEST_MISSING": EXIT_CORRUPT,
    "MANIFEST_INVALID": EXIT_CORRUPT,
    "MANIFEST_TOO_LARGE": EXIT_RESOURCE_LIMIT,
    "SCHEMA_UNSUPPORTED": EXIT_CORRUPT,
    "FORMAT_VERSION_UNSUPPORTED": EXIT_CORRUPT,
    "PUBLICATION_MARKER_MISSING": EXIT_CORRUPT,
    "PUBLICATION_MARKER_MISMATCH": EXIT_CORRUPT,
    "FILE_INDEX_EMPTY": EXIT_CORRUPT,
    "FILE_INDEX_MISMATCH": EXIT_CORRUPT,
    "UNEXPECTED_PACKAGE_FILE": EXIT_CORRUPT,
    "SYMLINK_REJECTED": EXIT_UNSAFE_PATH,
    "SPECIAL_FILE_REJECTED": EXIT_UNSAFE_PATH,
    "UNSAFE_PATH": EXIT_UNSAFE_PATH,
    "PATH_ESCAPE": EXIT_UNSAFE_PATH,
    "NON_FINITE_NUMBER": EXIT_CORRUPT,
    "JSON_INVALID": EXIT_CORRUPT,
    "ENCODING_INVALID": EXIT_CORRUPT,
    "PROJECT_BUNDLE_INVALID": EXIT_CORRUPT,
    "EVIDENCE_SIDECAR_INVALID": EXIT_CORRUPT,
    "ASSET_DOCUMENT_INVALID": EXIT_CORRUPT,
    "ASSET_ROLE_UNKNOWN": EXIT_CORRUPT,
    "ASSET_POLICY_INVALID": EXIT_CORRUPT,
    "FINGERPRINT_SCOPE_UNSUPPORTED": EXIT_CORRUPT,
    "COORDINATE_BASIS_INVALID": EXIT_CORRUPT,
    "LICENSE_INVALID": EXIT_CORRUPT,
    "SCRIPT_IN_PACKAGE": EXIT_FORBIDDEN,
    "SELF_CONTAINED_FORBIDDEN_ASSET": EXIT_FORBIDDEN,
    "REFERENCE_MODE_ASSET_BYTES_PRESENT": EXIT_FORBIDDEN,
    "EMPTY_PROJECT": EXIT_CORRUPT,
    "EMPTY_ASSET_LIST": EXIT_CORRUPT,
    "EMPTY_ENUMERATION": EXIT_CORRUPT,
    "RESOURCE_LIMIT_EXCEEDED": EXIT_RESOURCE_LIMIT,
    "PACKAGE_EXISTS": EXIT_CONFLICT,
    "PACKAGE_READ_ONLY": EXIT_CONFLICT,
    "IO_WRITE_FAILED": EXIT_CORRUPT,
    "EXPORT_CONFLICT": EXIT_CONFLICT,
    "TEST_FAULT_AFTER_STAGING": EXIT_FAULT,
    "TEST_FAULT_BEFORE_PUBLISH": EXIT_FAULT,
    "PORTABLE_ERROR": EXIT_CORRUPT,
}

LIMITS: Dict[str, int] = {
    "max_manifest_bytes": 32 * 1024 * 1024,
    "max_json_bytes": 256 * 1024 * 1024,
    "max_package_files": 20000,
    "max_package_bytes": 2 * 1024 * 1024 * 1024,
    "max_packaged_asset_bytes": 256 * 1024 * 1024,
    "max_depth": 16,
}

ROLES_CONTRACT = ("RAW_MEASUREMENT", "DERIVED_DISPLAY", "AUXILIARY_CALIBRATION")
ROLES_EXTENDED = ROLES_CONTRACT + ("DERIVED_ROI_MEASUREMENT",)
MEASUREMENT_ROLES = ("RAW_MEASUREMENT", "DERIVED_ROI_MEASUREMENT")
DRIFT_ACCEPTABLE_ROLES = ("AUXILIARY_CALIBRATION",)
OBSERVED_STATES = ("MATCHED", "DRIFT_ACCEPTED", "NOT_PRESENT", "DEFERRED",
                   "HASH_CHECK_DISABLED", "NOT_MEASURED")
LOCATOR_MODES = ("repo_relative", "project_relative", "external_uri")
RELATIVE_LOCATOR_MODES = ("repo_relative", "project_relative")
COORDINATE_BASES = ("source_coordinates", "scene_crs", "project_local", "synthetic_local")
RESOLUTION_MISSING_ACTIONS = ("REQUEST_RELOCATION", "SHOW_LIMITED_VIEW")
RESOLUTION_REPLACED_ACTIONS = ("WARN_AND_BLOCK_SILENT_SUBSTITUTION",)
RESOLUTION_RELOCATED_ACTIONS = ("HASH_RELOCATE_OK",)
REQUIRED_ASSET_FIELDS = ("asset_id", "role", "source_locator", "fingerprint",
                         "unit", "coordinate", "license", "resolution")
ORIGIN_UNKNOWN_STATUSES = ("POINT_CLOUD_EXPORT_ORIGIN_UNKNOWN", "UNKNOWN_ORIGIN", "UNKNOWN")
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_PACKAGE_EXTENSIONS = frozenset({
    ".py", ".pyc", ".pyo", ".pth", ".bat", ".cmd", ".ps1", ".exe", ".dll",
    ".so", ".dylib", ".js", ".mjs", ".cjs", ".vbs", ".jar", ".com", ".scr",
})
TOP_LEVEL_ALLOWED = frozenset({
    MANIFEST_NAME, PUBLICATION_MARKER, "project", "evidence", "assets",
    "coordinates", "licenses", "README.txt", "RECOVERY.md", "state",
})
REQUIRED_LAYOUT = (
    "project/head.json",
    "assets/assets.json",
    "coordinates/coordinate_basis.json",
    "licenses/licenses.json",
    "README.txt",
    "RECOVERY.md",
)


class PortableError(RuntimeError):
    """A stable, machine-readable portable-package failure."""

    def __init__(self, message: str, code: str = "PORTABLE_ERROR",
                 *, details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        self.exit_code = int(ERROR_EXIT_CODES.get(self.code, EXIT_CORRUPT))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": ERROR_SCHEMA,
            "measured_at": _now_iso(),
            "status": "ERROR",
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": copy.deepcopy(self.details),
            "exit_code": self.exit_code,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Return a plain dict for contract objects, mappings and record-like values.

    Project snapshots carry contract value objects (contracts._ContractObject)
    that expose to_dict()/dataclass fields but are not Mapping instances.
    """
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        payload = converter()
        if isinstance(payload, Mapping):
            return dict(payload)
    names = getattr(type(value), "__dataclass_fields__", None)
    if names:
        return {name: getattr(value, name) for name in names}
    return {}


def _reject_constant(token: str) -> Any:
    raise PortableError("non-finite number token %r is not allowed" % token,
                        "NON_FINITE_NUMBER")


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PortableError("non-finite number at %s" % path, "NON_FINITE_NUMBER")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, "%s.%s" % (path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, "%s[%d]" % (path, index))
        return


def _parse_json_strict(text: str, label: str) -> Any:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except PortableError:
        raise
    except json.JSONDecodeError as exc:
        raise PortableError("JSON does not parse (%s): %s" % (label, exc), "JSON_INVALID") from exc
    _assert_finite(value, label)
    return value


def _read_json_strict(path: os.PathLike[str] | str, *,
                      max_bytes: int = LIMITS["max_json_bytes"]) -> Any:
    target = Path(path)
    if not target.is_file():
        raise PortableError("JSON document is missing: %s" % target, "MANIFEST_MISSING")
    size = target.stat().st_size
    if size > max_bytes:
        raise PortableError("JSON document exceeds %d bytes: %s" % (max_bytes, target),
                            "RESOURCE_LIMIT_EXCEEDED")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PortableError("JSON document is not UTF-8: %s" % exc, "ENCODING_INVALID") from exc
    return _parse_json_strict(text, str(target))


def _read_text(path: os.PathLike[str] | str, *,
               max_bytes: int = LIMITS["max_json_bytes"]) -> str:
    target = Path(path)
    if not target.is_file():
        raise PortableError("text document is missing: %s" % target, "MANIFEST_MISSING")
    size = target.stat().st_size
    if size > max_bytes:
        raise PortableError("text document exceeds %d bytes: %s" % (max_bytes, target),
                            "RESOURCE_LIMIT_EXCEEDED")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PortableError("text document is not UTF-8: %s" % exc,
                            "ENCODING_INVALID") from exc


def _safe_relpath(value: Any, label: str = "path") -> str:
    """Return a plain relative POSIX path, or raise UNSAFE_PATH."""
    if not isinstance(value, str):
        raise PortableError("%s must be a string, got %r" % (label, type(value).__name__),
                            "UNSAFE_PATH")
    text = value
    if not text or chr(0) in text:
        raise PortableError("%s is empty or carries NUL" % label, "UNSAFE_PATH")
    if chr(92) in text:
        raise PortableError("%s must use '/' separators: %r" % (label, text), "UNSAFE_PATH")
    if text.startswith("/") or text.startswith("~"):
        raise PortableError("%s must not be absolute: %r" % (label, text), "PATH_ESCAPE")
    if re.match(r"^[A-Za-z]:", text):
        raise PortableError("%s must not carry a drive letter: %r" % (label, text), "PATH_ESCAPE")
    if text.startswith("//"):
        raise PortableError("%s must not be a UNC path: %r" % (label, text), "PATH_ESCAPE")
    for part in text.split("/"):
        if part in ("", ".", ".."):
            raise PortableError("%s has an empty, '.', or '..' component: %r" % (label, text),
                                "PATH_ESCAPE")
        if ":" in part:
            raise PortableError("%s has a ':' (stream) component: %r" % (label, text),
                                "UNSAFE_PATH")
    return text


def _is_safe_relative(value: Any) -> bool:
    try:
        _safe_relpath(value)
        return True
    except PortableError:
        return False


def _assert_inside(root: Path, candidate: Path, label: str) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents:
        raise PortableError("%s escapes %s: %s" % (label, root_resolved, candidate_resolved),
                            "PATH_ESCAPE")
    return candidate_resolved


def _lstat_is_reparse(entry: os.DirEntry) -> bool:
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return bool(attributes & 0x400)


def _walk_files(root: Path, *, skip_dirs: Sequence[str] = ()) -> List[str]:
    """List every regular file below root as a package-relative POSIX path.

    Symlinks and reparse points are rejected, never followed; special files are
    rejected.  Sorted for deterministic manifests.
    """
    root = Path(root)
    if not root.is_dir():
        raise PortableError("directory is missing: %s" % root, "PACKAGE_NOT_FOUND")
    results: List[str] = []

    def visit(directory: Path, prefix: str, depth: int) -> None:
        if depth > LIMITS["max_depth"]:
            raise PortableError("package directory depth exceeds %d below %s"
                                % (LIMITS["max_depth"], prefix or "."),
                                "RESOURCE_LIMIT_EXCEEDED")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise PortableError("directory cannot be listed: %s" % exc, "PACKAGE_NOT_FOUND") from exc
        for entry in entries:
            relative = prefix + entry.name
            if entry.is_symlink() or _lstat_is_reparse(entry):
                raise PortableError("symlink or reparse point is not allowed: %s" % relative,
                                    "SYMLINK_REJECTED")
            if entry.is_dir(follow_symlinks=False):
                if relative in skip_dirs:
                    continue
                visit(Path(entry.path), relative + "/", depth + 1)
            elif entry.is_file(follow_symlinks=False):
                results.append(relative)
            else:
                raise PortableError("special file is not allowed: %s" % relative,
                                    "SPECIAL_FILE_REJECTED")

    visit(root, "", 0)
    return results


def _byte_size(root: Path, relatives: Iterable[str]) -> int:
    total = 0
    for relative in relatives:
        total += Path(root, relative).stat().st_size
    return total


READ_ONLY_ERRNOS = (errno.EACCES, errno.EPERM, errno.EROFS)
RESOURCE_ERRNOS = (errno.ENOSPC, errno.EDQUOT, errno.EFBIG, errno.ENAMETOOLONG)


def _write_error(exc: OSError, *, target: str, action: str) -> "PortableError":
    """Map a failed write to a stable code instead of letting a traceback escape."""
    err = getattr(exc, "errno", None)
    details = {"errno": err, "winerror": getattr(exc, "winerror", None),
               "path": target, "action": action, "reason": str(exc)}
    if isinstance(exc, PermissionError) or err in READ_ONLY_ERRNOS:
        return PortableError("target is read-only, cannot %s: %s" % (action, target),
                             "PACKAGE_READ_ONLY", details=details)
    if err in RESOURCE_ERRNOS:
        return PortableError("write failed for lack of space or name limits while %s: %s"
                             % (action, target), "RESOURCE_LIMIT_EXCEEDED", details=details)
    return PortableError("I/O write failed while %s: %s" % (action, target),
                         "IO_WRITE_FAILED", details=details)


def _mkdir_checked(path: Path, *, parents: bool = False, exist_ok: bool = False,
                   action: str = "create directory") -> None:
    try:
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as exc:
        raise _write_error(exc, target=str(path), action=action) from exc


def _rename_checked(source: Path, target: Path, *, action: str) -> None:
    try:
        os.rename(str(source), str(target))
    except OSError as exc:
        raise _write_error(exc, target=str(target), action=action) from exc


def _copy_file_verified(source: Path, destination: Path) -> Dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    _mkdir_checked(destination.parent, parents=True, exist_ok=True,
                   action="create directory for %s" % destination.name)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, destination.open("wb") as writer:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise _write_error(exc, target=str(destination), action="copy file") from exc
    return {"sha256": digest.hexdigest(), "bytes": size}


def _write_bytes(path: Path, data: bytes) -> Dict[str, Any]:
    _mkdir_checked(path.parent, parents=True, exist_ok=True,
                   action="create directory for %s" % path.name)
    digest = hashlib.sha256(data).hexdigest()
    try:
        with path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise _write_error(exc, target=str(path), action="write file") from exc
    return {"sha256": digest, "bytes": len(data)}


def _write_json(path: Path, value: Any) -> Dict[str, Any]:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + chr(10)
    return _write_bytes(path, text.encode("utf-8"))


def _fault_mode(explicit: Optional[str] = None) -> Optional[str]:
    value = explicit if explicit is not None else os.environ.get("W154_TEST_FAULT", "")
    value = str(value or "").strip()
    if not value:
        return None
    if value not in ("after_staging", "before_publish"):
        raise PortableError("unknown W154_TEST_FAULT value %r" % value, "USAGE_ERROR")
    return value


def _slug(value: str, limit: int = 40) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return (text or "item")[:limit]


# -- assets ------------------------------------------------------------------

def _asset_record(*, asset_id: str, role: str, locator_mode: str, locator_value: str,
                  redistributable: bool, fingerprint: str, unit_declared: str,
                  unit_status: str, metric_allowed: bool, coordinate_basis: str,
                  coordinate_version: str, origin: Optional[Sequence[float]],
                  origin_status: Optional[str], crs: Any, license_id: str,
                  usage_scope: str, redistribution_allowed: bool, training_allowed: bool,
                  license_evidence: str, resolution: Mapping[str, str],
                  derivation: Optional[Mapping[str, Any]] = None,
                  portable: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
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
            "scope": "full_file_bytes",
        },
        "unit": {
            "declared": unit_declared,
            "status": unit_status,
            "metric_output_allowed": bool(metric_allowed),
        },
        "coordinate": {
            "basis": coordinate_basis,
            "version": coordinate_version,
            "origin": list(origin) if origin is not None else None,
            "origin_status": origin_status,
            "crs": crs,
        },
        "license": {
            "license_id": license_id,
            "usage_scope": usage_scope,
            "redistribution_allowed": bool(redistribution_allowed),
            "training_allowed": bool(training_allowed),
            "evidence": license_evidence,
        },
        "resolution": dict(resolution),
        "portable": dict(portable or {}),
    }
    if derivation is not None:
        record["derivation"] = dict(derivation)
    return record


def default_resolution() -> Dict[str, str]:
    return {
        "on_missing": "REQUEST_RELOCATION",
        "on_replaced": "WARN_AND_BLOCK_SILENT_SUBSTITUTION",
        "on_relocated": "HASH_RELOCATE_OK",
    }


def _observation_role(observation: Any, source_file: Mapping[str, Any]) -> str:
    coverage = dict(getattr(observation, "coverage", None) or {})
    declared = str(coverage.get("asset_role") or "")
    if declared in ROLES_EXTENDED:
        return declared
    media = str(source_file.get("media_type") or "")
    path = str(source_file.get("path") or "")
    if media in ("application/octet-stream", "application/las", "application/ply") or             path.lower().endswith((".ply", ".las", ".laz", ".xyz", ".pts", ".bin")):
        return "RAW_MEASUREMENT"
    return "AUXILIARY_CALIBRATION"


def derive_assets_from_project(project: ProjectSnapshot, *,
                               license_evidence: str = "",
                               contract_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Build a reference-mode asset document from the project's own declarations.

    Only files the project itself names are registered.  Nothing is invented: a
    file whose SHA-256 the observation does not declare is skipped (and listed
    under skipped with a reason) rather than fingerprinted by guessing.
    """
    artifacts = dict(project.artifacts or {})
    session = dict(artifacts.get("workbench_session") or {})
    registry = dict(session.get("source_registry") or {})
    provenance = dict(session.get("source_provenance") or {})
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen: set = set()
    coordinate_frame = str(project.observations[0].coordinate_frame
                           if project.observations else "unknown")

    def add(record: Dict[str, Any]) -> None:
        if record["asset_id"] in seen:
            return
        seen.add(record["asset_id"])
        records.append(record)

    raw_asset_id = None
    raw_block = dict(provenance.get("raw") or {})
    if raw_block.get("path") and raw_block.get("sha256"):
        raw_asset_id = "%s__raw__sha256_%s" % (
            _slug(str(registry.get("id") or "source")), str(raw_block["sha256"])[:12])
        add(_asset_record(
            asset_id=raw_asset_id,
            role="RAW_MEASUREMENT",
            locator_mode="repo_relative",
            locator_value=str(raw_block["path"]),
            redistributable=False,
            fingerprint=str(raw_block["sha256"]),
            unit_declared="metre",
            unit_status="KNOWN",
            metric_allowed=True,
            coordinate_basis="source_coordinates",
            coordinate_version=coordinate_frame,
            origin=None,
            origin_status="POINT_CLOUD_EXPORT_ORIGIN_UNKNOWN",
            crs=None,
            license_id="UNVERIFIED",
            usage_scope=str(registry.get("license_status") or "UNKNOWN_LICENSE_SCOPE"),
            redistribution_allowed=False,
            training_allowed=False,
            license_evidence=license_evidence,
            resolution=default_resolution(),
            portable={"storage": "REFERENCE",
                      "declared_repo_relative": str(raw_block["path"]),
                      "note": "upstream raw measurement; registered by SHA-256 only, never copied"},
        ))

    for observation in project.observations:
        files = [_as_mapping(item)
                 for item in (getattr(observation, "source_files", None) or [])]
        if not files:
            skipped.append({"observation_id": observation.observation_id,
                            "reason": "NO_SOURCE_FILES_DECLARED"})
            continue
        primary = max(files, key=lambda item: int(item.get("byte_count") or 0))
        for source_file in files:
            path = str(source_file.get("path") or "")
            sha = str(source_file.get("sha256") or "")
            if not path or not SHA256_RE.match(sha):
                skipped.append({"observation_id": observation.observation_id, "path": path,
                                "reason": "UNDECLARED_OR_UNHASHED_SOURCE"})
                continue
            role = _observation_role(observation, source_file)
            if source_file is not primary and role != "AUXILIARY_CALIBRATION":
                role = "AUXILIARY_CALIBRATION"
            if path.lower().startswith("synthetic://"):
                skipped.append({"observation_id": observation.observation_id, "path": path,
                                "reason": "SYNTHETIC_URI_HAS_NO_FILE"})
                continue
            if not _is_safe_relative(path):
                skipped.append({"observation_id": observation.observation_id, "path": path,
                                "reason": "LOCATOR_NOT_RELATIVE"})
                continue
            origin = list(getattr(observation, "local_origin", None) or [])
            add(_asset_record(
                asset_id="%s__%s__sha256_%s" % (_slug(observation.observation_id),
                                                 _slug(os.path.basename(path), 20), sha[:12]),
                role=role,
                locator_mode="repo_relative",
                locator_value=path,
                redistributable=False,
                fingerprint=sha,
                unit_declared=str(getattr(observation, "unit", "UNKNOWN") or "UNKNOWN"),
                unit_status="KNOWN" if str(getattr(observation, "unit", "")).lower() in
                ("metre", "meter", "m") else "UNKNOWN",
                metric_allowed=str(getattr(observation, "unit", "")).lower() in
                ("metre", "meter", "m"),
                coordinate_basis="source_coordinates",
                coordinate_version=str(getattr(observation, "coordinate_frame", "") or "unknown"),
                origin=origin if len(origin) == 3 else None,
                origin_status=None if len(origin) == 3 else "UNKNOWN_ORIGIN",
                crs=None,
                license_id="UNVERIFIED",
                usage_scope=str(getattr(observation, "license", "") or "UNKNOWN_LICENSE_SCOPE"),
                redistribution_allowed=False,
                training_allowed=False,
                license_evidence=license_evidence,
                resolution=default_resolution(),
                derivation=(None if role == "RAW_MEASUREMENT" or
                             (not raw_asset_id and role != "DERIVED_DISPLAY") else {
                    "derived_from": [raw_asset_id] if raw_asset_id else [],
                    "method": "observation_declared_source_file",
                    "is_measurement": role in MEASUREMENT_ROLES,
                    "display_only": role == "DERIVED_DISPLAY",
                    "provenance_status": ("UPSTREAM_DECLARED" if raw_asset_id
                                          else "UPSTREAM_NOT_DECLARED_IN_PROJECT"),
                }),
                portable={"storage": "REFERENCE", "declared_repo_relative": path},
            ))

    roi = dict(provenance.get("roi") or {})
    index_path = roi.get("source_index_file")
    index_sha = roi.get("source_index_sha256")
    if index_path and index_sha and _is_safe_relative(str(index_path)):
        add(_asset_record(
            asset_id="%s__source_index__sha256_%s" % (_slug(str(registry.get("id") or "roi")),
                                                       str(index_sha)[:12]),
            role="AUXILIARY_CALIBRATION",
            locator_mode="repo_relative",
            locator_value=str(index_path),
            redistributable=False,
            fingerprint=str(index_sha),
            unit_declared="metre",
            unit_status="KNOWN",
            metric_allowed=False,
            coordinate_basis="source_coordinates",
            coordinate_version=coordinate_frame,
            origin=None,
            origin_status="UNKNOWN_ORIGIN",
            crs=None,
            license_id="UNVERIFIED",
            usage_scope=str(registry.get("license_status") or "UNKNOWN_LICENSE_SCOPE"),
            redistribution_allowed=False,
            training_allowed=False,
            license_evidence=license_evidence,
            resolution=default_resolution(),
            derivation=(None if not raw_asset_id else {
                "derived_from": [raw_asset_id],
                "method": "roi_source_ordinal_index",
                "is_measurement": False,
                "display_only": False,
            }),
            portable={"storage": "REFERENCE", "declared_repo_relative": str(index_path),
                      "note": "ROI ordinal to upstream source ordinal map; support only"},
        ))

    return {
        "schema": ASSETS_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "project_id": project.project_id,
        "contract": {
            "contract_version": asset_contract.CONTRACT_VERSION,
            "contract_sha256": contract_sha256,
            "extensions": [
                "role:DERIVED_ROI_MEASUREMENT (additive; absent from contract v1 enum)",
                "coordinate.origin:null+origin_status (contract v1 requires three numbers)",
                "license.training_allowed (absent from contract v1)",
                "portable.storage/portable.declared_repo_relative (portable layer)",
            ],
        },
        "assets": records,
        "skipped": skipped,
        "counts": _asset_counts(records),
        "not_claimed": [
            "An asset record is a reference, not an endorsement to redistribute.",
            "redistribution_allowed defaults to false when not explicitly stated.",
        ],
    }


def _asset_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_role: Dict[str, int] = {}
    by_storage: Dict[str, int] = {}
    for record in records:
        role = str(record.get("role"))
        by_role[role] = by_role.get(role, 0) + 1
        storage = str((record.get("portable") or {}).get("storage") or "REFERENCE")
        by_storage[storage] = by_storage.get(storage, 0) + 1
    return {
        "asset_count": len(records),
        "by_role": by_role,
        "by_storage": by_storage,
        "measurement_count": sum(by_role.get(role, 0) for role in MEASUREMENT_ROLES),
        "redistributable_count": sum(
            1 for record in records
            if bool((record.get("license") or {}).get("redistribution_allowed"))),
    }


def normalize_asset_document(document: Any) -> List[Dict[str, Any]]:
    """Return the validated asset records of one asset document."""
    if not isinstance(document, Mapping):
        raise PortableError("asset document must be one JSON object", "ASSET_DOCUMENT_INVALID")
    records = document.get("assets")
    if not isinstance(records, list) or not records:
        raise PortableError("asset list is empty; a package without assets is not a result",
                            "EMPTY_ASSET_LIST")
    problems: List[str] = []
    seen: set = set()
    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            problems.append("assets[%d] must be an object" % index)
            continue
        record = copy.deepcopy(dict(raw))
        for problem in validate_asset_record(record):
            problems.append("assets[%d] %s" % (index, problem))
        asset_id = str(record.get("asset_id"))
        if asset_id in seen:
            problems.append("duplicate asset_id %r" % asset_id)
        seen.add(asset_id)
        normalized.append(record)
    if problems:
        raise PortableError("asset document is invalid: %s" % "; ".join(problems[:12]),
                            "ASSET_DOCUMENT_INVALID", details={"problems": problems})
    return normalized


def validate_asset_record(record: Mapping[str, Any]) -> List[str]:
    """Validate one asset record against the W152 contract plus portable policy."""
    problems: List[str] = []
    for field in REQUIRED_ASSET_FIELDS:
        if field not in record:
            problems.append("missing required field: %s" % field)
    asset_id = record.get("asset_id")
    if not isinstance(asset_id, str) or not ASSET_ID_RE.match(asset_id):
        problems.append("asset_id must match %s" % ASSET_ID_RE.pattern)
    role = str(record.get("role") or "")
    if role not in ROLES_EXTENDED:
        problems.append("role must be one of %s" % (ROLES_EXTENDED,))
    locator = record.get("source_locator")
    if not isinstance(locator, Mapping):
        problems.append("source_locator must be an object")
    else:
        mode = str(locator.get("mode") or "")
        if mode not in LOCATOR_MODES:
            problems.append("source_locator.mode must be one of %s" % (LOCATOR_MODES,))
        value = locator.get("value")
        if not isinstance(value, str) or not value.strip():
            problems.append("source_locator.value must be a non-empty string")
        elif mode in RELATIVE_LOCATOR_MODES and not _is_safe_relative(value):
            problems.append("source_locator.value must be a safe relative path for mode %s" % mode)
        if not isinstance(locator.get("redistributable"), bool):
            problems.append("source_locator.redistributable must be an explicit boolean")
    fingerprint = record.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        problems.append("fingerprint must be an object")
    else:
        if fingerprint.get("algorithm") != "sha256":
            problems.append("fingerprint.algorithm must be sha256")
        if not SHA256_RE.match(str(fingerprint.get("value") or "")):
            problems.append("fingerprint.value must be 64 lowercase hex characters")
        if fingerprint.get("scope") != "full_file_bytes":
            problems.append("fingerprint.scope must be full_file_bytes (whole file bytes)")
    unit = record.get("unit")
    if not isinstance(unit, Mapping):
        problems.append("unit must be an object")
    else:
        if unit.get("status") not in ("KNOWN", "UNKNOWN"):
            problems.append("unit.status must be KNOWN or UNKNOWN")
        if not isinstance(unit.get("metric_output_allowed"), bool):
            problems.append("unit.metric_output_allowed must be an explicit boolean")
        if unit.get("status") == "UNKNOWN" and unit.get("metric_output_allowed") is not False:
            problems.append("unit.status UNKNOWN requires metric_output_allowed=false")
        if not str(unit.get("declared") or "").strip():
            problems.append("unit.declared must be non-empty")
    coordinate = record.get("coordinate")
    if not isinstance(coordinate, Mapping):
        problems.append("coordinate must be an object")
    else:
        if coordinate.get("basis") not in COORDINATE_BASES:
            problems.append("coordinate.basis must be one of %s" % (COORDINATE_BASES,))
        if not str(coordinate.get("version") or "").strip():
            problems.append("coordinate.version must be a non-empty string")
        origin = coordinate.get("origin")
        status = str(coordinate.get("origin_status") or "")
        if origin is None:
            if status not in ORIGIN_UNKNOWN_STATUSES:
                problems.append("coordinate.origin may be null only with an explicit "
                                "origin_status in %s" % (ORIGIN_UNKNOWN_STATUSES,))
        elif not (isinstance(origin, list) and len(origin) == 3):
            problems.append("coordinate.origin must be three numbers or null with origin_status")
        else:
            for index, item in enumerate(origin):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    problems.append("coordinate.origin[%d] must be numeric" % index)
                elif not math.isfinite(float(item)):
                    problems.append("coordinate.origin[%d] must be finite" % index)
        if "crs" not in coordinate:
            problems.append("coordinate.crs must be present (null allowed)")
    license_block = record.get("license")
    if not isinstance(license_block, Mapping):
        problems.append("license must be an object")
    else:
        if not str(license_block.get("license_id") or "").strip():
            problems.append("license.license_id must be non-empty")
        if not str(license_block.get("usage_scope") or "").strip():
            problems.append("license.usage_scope must be non-empty")
        if not isinstance(license_block.get("redistribution_allowed"), bool):
            problems.append("license.redistribution_allowed must be an explicit boolean")
        if "training_allowed" in license_block and                 not isinstance(license_block.get("training_allowed"), bool):
            problems.append("license.training_allowed must be a boolean when present")
    resolution = record.get("resolution")
    if not isinstance(resolution, Mapping):
        problems.append("resolution must be an object")
    else:
        if resolution.get("on_missing") not in RESOLUTION_MISSING_ACTIONS:
            problems.append("resolution.on_missing must be one of %s (limited view or relocate)"
                            % (RESOLUTION_MISSING_ACTIONS,))
        if resolution.get("on_replaced") not in RESOLUTION_REPLACED_ACTIONS:
            problems.append("resolution.on_replaced must be %s"
                            % (RESOLUTION_REPLACED_ACTIONS,))
        if resolution.get("on_relocated") not in RESOLUTION_RELOCATED_ACTIONS:
            problems.append("resolution.on_relocated must be %s"
                            % (RESOLUTION_RELOCATED_ACTIONS,))
    derivation = record.get("derivation")
    if derivation is not None:
        if not isinstance(derivation, Mapping):
            problems.append("derivation must be an object when present")
        else:
            derived_from = derivation.get("derived_from")
            provenance_status = str(derivation.get("provenance_status") or "")
            if not isinstance(derived_from, list):
                problems.append("derivation.derived_from must be a list")
            elif not derived_from and not (role == "DERIVED_DISPLAY" and
                                           provenance_status == "UPSTREAM_NOT_DECLARED_IN_PROJECT"):
                problems.append("derivation.derived_from must be a non-empty list unless a "
                                "DERIVED_DISPLAY record declares "
                                "provenance_status=UPSTREAM_NOT_DECLARED_IN_PROJECT")
            if not isinstance(derivation.get("is_measurement"), bool):
                problems.append("derivation.is_measurement must be an explicit boolean")
            if not str(derivation.get("method") or "").strip():
                problems.append("derivation.method must be non-empty")
            if role == "DERIVED_ROI_MEASUREMENT" and derivation.get("is_measurement") is not True:
                problems.append("role DERIVED_ROI_MEASUREMENT requires derivation.is_measurement=true")
            if role == "DERIVED_DISPLAY" and derivation.get("display_only") is not True:
                problems.append("role DERIVED_DISPLAY requires derivation.display_only=true")
    elif role in ("DERIVED_DISPLAY", "DERIVED_ROI_MEASUREMENT"):
        problems.append("role %s requires a derivation block" % role)
    portable = record.get("portable")
    if portable is not None and not isinstance(portable, Mapping):
        problems.append("portable must be an object when present")
    return problems


# -- coordinates / licences / evidence ---------------------------------------

def _source_domain(project: ProjectSnapshot) -> str:
    artifacts = dict(project.artifacts or {})
    session = dict(artifacts.get("workbench_session") or {})
    declared = str(session.get("domain") or "")
    if declared in ("internal_real", "synthetic"):
        return declared
    for observation in project.observations:
        coverage = dict(getattr(observation, "coverage", None) or {})
        domain = str(coverage.get("domain") or "")
        if domain == "real":
            return "internal_real"
        if domain == "synthetic" or bool(coverage.get("synthetic")):
            return "synthetic"
    if bool(artifacts.get("synthetic")):
        return "synthetic"
    return "unknown"


def _normalized_declared_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _snapshot_declarations(project: ProjectSnapshot) -> Dict[str, List[str]]:
    """Collect every source file path/SHA that the project snapshot itself declares.

    The assets document is a *claim*; this map is the declaration it must agree
    with.  Paths are compared textually after slash normalisation only, so a
    mismatch cannot be hidden behind a basename or a different directory.
    """
    declared: Dict[str, set] = {}

    def add(path: Any, digest: Any) -> None:
        relative = _normalized_declared_path(path)
        if not relative:
            return
        declared.setdefault(relative, set()).add(str(digest or ""))

    artifacts = dict(project.artifacts or {})
    session = dict(artifacts.get("workbench_session") or {})
    provenance = dict(session.get("source_provenance") or {})
    raw = dict(provenance.get("raw") or {})
    if raw.get("path"):
        add(raw.get("path"), raw.get("sha256"))
    roi = dict(provenance.get("roi") or {})
    if roi.get("source_index_file"):
        add(roi.get("source_index_file"), roi.get("source_index_sha256"))
    for observation in project.observations:
        for source_file in (getattr(observation, "source_files", None) or []):
            path = getattr(source_file, "path", None)
            digest = getattr(source_file, "sha256", None)
            if path is None and isinstance(source_file, Mapping):
                path = source_file.get("path") or source_file.get("uri")
                digest = source_file.get("sha256")
            add(path, digest)
    return {key: sorted(value) for key, value in declared.items()}


def _record_expected_hashes(record: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    fingerprint = dict(record.get("fingerprint") or {})
    for candidate in (fingerprint.get("value"),
                      (record.get("portable") or {}).get("expected_sha256"),
                      (record.get("portable") or {}).get("declared_fingerprint")):
        text = str(candidate or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _snapshot_asset_mismatches(records: Sequence[Mapping[str, Any]], project: ProjectSnapshot,
                               *, source_domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compare asset claims with the snapshot's own source-file declarations.

    Synthetic contract fixtures are exempt: their source files are synthetic://
    declarations, not real external measurements, and the directed tests build
    their asset documents by hand.  Every real-domain package is checked.
    """
    domain = source_domain if source_domain is not None else _source_domain(project)
    if domain == "synthetic":
        return []
    declarations = _snapshot_declarations(project)
    issues: List[Dict[str, Any]] = []
    seen: set = set()

    def add_issue(record: Mapping[str, Any], reason: str, **extra: Any) -> None:
        asset_id = str(record.get("asset_id") or "")
        key = (asset_id, reason)
        if key in seen:
            return
        seen.add(key)
        issue = {
            "asset_id": asset_id,
            "role": str(record.get("role") or ""),
            "reason": reason,
            "source_domain": domain,
        }
        issue.update(extra)
        issues.append(issue)

    for record in records:
        role = str(record.get("role") or "")
        locator = dict(record.get("source_locator") or {})
        path = _normalized_declared_path(locator.get("value"))
        expected_values = _record_expected_hashes(record)
        if len(expected_values) > 1:
            add_issue(record, "ASSET_DECLARED_HASH_CONFLICT",
                      path=path, expected_sha256=expected_values)
            continue
        if not expected_values:
            if role in MEASUREMENT_ROLES:
                add_issue(record, "ASSET_HASH_UNDECLARED", path=path)
            continue
        expected = expected_values[0]
        declared = declarations.get(path)
        if not declared:
            if role in MEASUREMENT_ROLES:
                add_issue(record, "SNAPSHOT_DECLARATION_MISSING", path=path,
                          expected_sha256=expected)
            continue
        if len(set(declared)) != 1:
            add_issue(record, "SNAPSHOT_DECLARATION_AMBIGUOUS", path=path,
                      expected_sha256=expected, declared_sha256=declared)
            continue
        if expected != declared[0]:
            add_issue(record, "SNAPSHOT_HASH_MISMATCH", path=path,
                      expected_sha256=expected, declared_sha256=declared[0])
    return issues


def _geometry_digest(project: ProjectSnapshot) -> str:
    payload = {
        "fragments": [
            [fragment.fragment_id, fragment.plane_local.to_dict(),
             [region.to_dict() for region in fragment.finite_regions_local]]
            for fragment in project.fragments
        ],
        "surfaces": [
            [surface.surface_id, list(surface.member_fragment_ids),
             [region.to_dict() for region in surface.finite_support]]
            for surface in project.surfaces
        ],
    }
    return _canonical_sha256(payload)


def build_coordinate_basis(project: ProjectSnapshot, assets: Sequence[Mapping[str, Any]],
                           *, source_domain: str) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []
    crs_known = False
    for observation in project.observations:
        rois = []
        for roi in getattr(observation, "roi", None) or []:
            block = _as_mapping(roi)
            rois.append({
                "roi_id": block.get("roi_id"),
                "unit": block.get("unit"),
                "coordinate_frame": block.get("coordinate_frame"),
                "bounds_min": list(block.get("bounds_min") or []),
                "bounds_max": list(block.get("bounds_max") or []),
            })
        frame = str(getattr(observation, "coordinate_frame", "") or "")
        if "crs" in frame.lower() or "epsg" in frame.lower():
            crs_known = True
        observations.append({
            "observation_id": observation.observation_id,
            "coordinate_frame": frame,
            "unit": observation.unit,
            "local_origin": list(getattr(observation, "local_origin", None) or []),
            "roi": rois,
            "registration": copy.deepcopy(getattr(observation, "registration", None) or {}),
        })
    artifacts = dict(project.artifacts or {})
    scene = dict(artifacts.get("workbench_scene") or {})
    scene_frame = copy.deepcopy(scene.get("frame")) if isinstance(scene.get("frame"), Mapping) else None
    session = dict(artifacts.get("workbench_session") or {})
    display_convention = ("SOURCE_MINUS_LOCAL_ORIGIN"
                          if str(session.get("domain") or "") == "internal_real"
                          else "SYNTHETIC_LOCAL")
    return {
        "schema": COORDINATES_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "project_id": project.project_id,
        "source_domain": source_domain,
        "observations": observations,
        "scene_frame": scene_frame,
        "asset_basis": [
            {
                "asset_id": record.get("asset_id"),
                "basis": (record.get("coordinate") or {}).get("basis"),
                "version": (record.get("coordinate") or {}).get("version"),
                "origin": (record.get("coordinate") or {}).get("origin"),
                "origin_status": (record.get("coordinate") or {}).get("origin_status"),
                "crs": (record.get("coordinate") or {}).get("crs"),
            }
            for record in assets
        ],
        "display_convention": display_convention,
        "georeferencing": "CRS_DECLARED" if crs_known else "UNKNOWN_XY_ORIGIN",
        "not_claimed": [
            "No CRS transformation, no georeferencing and no metric closure is claimed.",
            "A local origin is a display subtraction, not a measured survey origin.",
        ],
    }


def build_licenses(project: ProjectSnapshot, assets: Sequence[Mapping[str, Any]],
                   *, distribution_mode: str) -> Dict[str, Any]:
    artifacts = dict(project.artifacts or {})
    session = dict(artifacts.get("workbench_session") or {})
    registry = dict(session.get("source_registry") or {})
    rows = []
    for record in assets:
        block = dict(record.get("license") or {})
        rows.append({
            "asset_id": record.get("asset_id"),
            "role": record.get("role"),
            "license_id": block.get("license_id"),
            "usage_scope": block.get("usage_scope"),
            "redistribution_allowed": bool(block.get("redistribution_allowed")),
            "training_allowed": bool(block.get("training_allowed")),
            "evidence": block.get("evidence"),
        })
    redistributable = [row["asset_id"] for row in rows if row["redistribution_allowed"]]
    return {
        "schema": LICENSES_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "project_id": project.project_id,
        "distribution_mode": distribution_mode,
        "project_license": {
            "license_id": str(registry.get("license_status") or "UNVERIFIED"),
            "usage_scope": str(registry.get("license_status") or "UNKNOWN_LICENSE_SCOPE"),
            "redistribution_allowed": bool(redistributable) and len(redistributable) == len(rows),
            "training_allowed": False,
            "evidence": "project snapshot observation/licence declarations",
        },
        "assets": rows,
        "summary": {
            "asset_count": len(rows),
            "redistributable_count": len(redistributable),
            "non_redistributable_count": len(rows) - len(redistributable),
            "distribution": ("SELF_CONTAINED_REDISTRIBUTABLE_ASSETS" if distribution_mode == SELF_CONTAINED
                             else "REFERENCES_ONLY_NO_MEASUREMENT_BYTES"),
            "training_allowed": False,
        },
        "not_claimed": [
            "A licence id recorded here is a statement copied from the source, not a legal review.",
            "redistribution_allowed defaults to false; an absent statement is never read as permission.",
        ],
    }


def _evaluate_evidence(project: ProjectSnapshot, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate the W153 sidecar binding against the current project snapshot."""
    project_hash = project.compute_content_hash()
    if payload is None:
        return {"binding": "MISSING", "reasons": ["EVIDENCE_SIDECAR_ABSENT"],
                "snapshot_id": None, "record_count": 0, "status_counts": {},
                "schema": None, "project_content_hash": None}
    try:
        verdict = eledger.validate_payload(project, payload)
    except Exception as exc:  # noqa: BLE001 - a broken sidecar is a stated condition
        return {"binding": "MISMATCH", "reasons": ["RECORDS_UNREADABLE:%s" % exc],
                "snapshot_id": payload.get("snapshot_id"), "record_count": 0,
                "status_counts": {}, "schema": payload.get("schema"),
                "project_content_hash": payload.get("project_content_hash")}
    records = dict(verdict.get("records") or {})
    reasons = list(verdict.get("reasons") or [])
    binding = "VALID" if verdict.get("valid") else "MISMATCH"
    if payload.get("project_content_hash") != project_hash and "PROJECT_HASH_MISMATCH" not in reasons:
        reasons.append("PROJECT_HASH_MISMATCH")
        binding = "MISMATCH"
    return {
        "binding": binding,
        "reasons": reasons,
        "snapshot_id": payload.get("snapshot_id"),
        "expected_snapshot_id": verdict.get("expected_snapshot_id"),
        "record_count": len(records),
        "status_counts": eledger.status_counts(records) if records else {},
        "schema": payload.get("schema"),
        "project_content_hash": payload.get("project_content_hash"),
    }


def _sidecar_path_for(bundle: Path, head: Mapping[str, Any]) -> Optional[Path]:
    tag = head.get("version_dir")
    if not isinstance(tag, str) or not _is_safe_relative(tag):
        return None
    candidate = Path(bundle) / "versions" / tag / "evidence_ledger.json"
    return candidate if candidate.is_file() else None


def _manifest_body_sha256(manifest: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(manifest))
    body["manifest_body_sha256"] = None
    return _canonical_sha256(body)


def _load_bundle(bundle: Path) -> Tuple[ProjectSnapshot, Dict[str, Any]]:
    try:
        project, receipt = ProjectStore(bundle).load()
    except ProjectStoreError as exc:
        raise PortableError("project bundle cannot be loaded: %s" % exc,
                            "PROJECT_BUNDLE_INVALID",
                            details={"cause_code": getattr(exc, "code", None)}) from exc
    if not receipt.get("verified"):
        raise PortableError("project bundle did not verify", "PROJECT_BUNDLE_INVALID")
    if not project.observations:
        raise PortableError("project has no observations; an empty project is not portable",
                            "EMPTY_PROJECT")
    if not project.fragments and not project.surfaces:
        raise PortableError("project has no fragments or surfaces; nothing to reopen",
                            "EMPTY_PROJECT")
    return project, receipt


def _copy_project_bundle(bundle: Path, head: Mapping[str, Any],
                         staging_project: Path) -> Dict[str, Any]:
    """Copy head.json plus the published version directory, byte for byte."""
    tag = head.get("version_dir")
    if not isinstance(tag, str) or "/" in tag or chr(92) in tag or tag in ("", ".", ".."):
        raise PortableError("head.version_dir is not a safe directory name: %r" % (tag,),
                            "UNSAFE_PATH")
    version_dir = Path(bundle) / "versions" / tag
    if not version_dir.is_dir():
        raise PortableError("published version directory is missing: %s" % version_dir,
                            "PROJECT_BUNDLE_INVALID")
    head_source = Path(bundle) / "head.json"
    copied = _copy_file_verified(head_source, staging_project / "head.json")
    if copied["sha256"] != file_sha256(head_source):
        raise PortableError("head.json copy does not match its source", "BUNDLE_HASH_MISMATCH")
    files: Dict[str, Dict[str, Any]] = {"project/head.json": copied}
    for relative in _walk_files(version_dir):
        source = version_dir / relative
        destination = staging_project / "versions" / tag / relative
        result = _copy_file_verified(source, destination)
        if result["sha256"] != file_sha256(source):
            raise PortableError("copied file does not match its source: %s" % relative,
                                "BUNDLE_HASH_MISMATCH")
        files["project/versions/%s/%s" % (tag, relative)] = result
    snapshot_relative = "versions/%s/snapshot.json" % tag
    if snapshot_relative not in {
            relative[len("project/"):] for relative in files}:
        raise PortableError("copied bundle has no snapshot.json", "PROJECT_BUNDLE_INVALID")
    snapshot_sha = files["project/%s" % snapshot_relative]["sha256"]
    if snapshot_sha != head.get("snapshot_sha256"):
        raise PortableError("snapshot sha256 differs from head", "BUNDLE_HASH_MISMATCH")
    manifest_relative = "project/versions/%s/manifest.json" % tag
    if manifest_relative in files and files[manifest_relative]["sha256"] != head.get("manifest_sha256"):
        raise PortableError("version manifest sha256 differs from head", "BUNDLE_HASH_MISMATCH")
    return {
        "version_dir": tag,
        "files": files,
        "snapshot_sha256": snapshot_sha,
        "head_sha256": copied["sha256"],
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def _finalize_assets(document: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                     distribution_mode: str, source_domain: str,
                     asset_roots: Sequence[Path],
                     verify_asset_hashes: bool,
                     asset_hash_limit_bytes: int,
                     accept_source_drift: Sequence[str] = (),
                     drift_reason: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                                                      Dict[str, Any]]:
    """Apply distribution rules; keep declared (expected) hashes distinct from observed ones."""
    finalized: List[Dict[str, Any]] = []
    forbidden: List[Dict[str, Any]] = []
    accepted_ids = {str(item) for item in accept_source_drift}
    reason_text = str(drift_reason or "").strip()
    if accepted_ids and not reason_text:
        raise PortableError(
            "--accept-source-drift requires a non-empty --drift-reason",
            "DRIFT_REASON_REQUIRED", details={"asset_ids": sorted(accepted_ids)})
    consumed_accepts: set = set()
    accepted_drift: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    for raw in records:
        record = copy.deepcopy(dict(raw))
        portable = dict(record.get("portable") or {})
        license_block = dict(record.get("license") or {})
        locator = dict(record.get("source_locator") or {})
        role = str(record.get("role"))
        asset_id = str(record.get("asset_id"))
        allowed = license_block.get("redistribution_allowed") is True
        locator_allowed = locator.get("redistributable") is True
        if distribution_mode == SELF_CONTAINED:
            reasons = []
            if not allowed:
                reasons.append("license.redistribution_allowed is not true")
            if not locator_allowed:
                reasons.append("source_locator.redistributable is not true")
            if role in MEASUREMENT_ROLES and source_domain != "synthetic":
                reasons.append("measurement role %s in source domain %s" % (role, source_domain))
            if reasons:
                forbidden.append({"asset_id": record.get("asset_id"), "role": role,
                                  "reasons": reasons})
                continue
            portable["storage"] = "PACKAGED"
            portable["packaged_path"] = "assets/files/%s/%s" % (
                _slug(str(record.get("asset_id")), 64),
                _safe_relpath(os.path.basename(str(locator.get("value") or "asset.bin")),
                              "packaged asset name"))
        else:
            portable["storage"] = "REFERENCE"
            portable["packaged_path"] = None
        fingerprint = dict(record.get("fingerprint") or {})
        expected = str(fingerprint.get("value") or "")
        portable["expected_sha256"] = expected
        portable["observed_sha256"] = None
        portable["asset_id_basis"] = "EXPECTED_SHA256_FROM_PROJECT_SNAPSHOT"
        value = str(locator.get("value") or "")
        candidates: List[Path] = []
        if locator.get("mode") in RELATIVE_LOCATOR_MODES and _is_safe_relative(value):
            for root in asset_roots:
                candidates.append(Path(root) / value)
        elif value:
            candidates.append(Path(value))
        found = next((item for item in candidates if item.is_file()), None)
        portable["hash_verified"] = False
        state = "NOT_MEASURED"
        if found is None:
            state = "NOT_PRESENT"
            portable["hash_verification"] = "ASSET_NOT_PRESENT_AT_EXPORT"
        else:
            portable["last_seen_absolute"] = str(found.resolve())
            size = found.stat().st_size
            portable["observed_bytes"] = int(size)
            if not verify_asset_hashes:
                state = "HASH_CHECK_DISABLED"
                portable["hash_verification"] = "HASH_CHECK_DISABLED_BY_CALLER"
            elif size > int(asset_hash_limit_bytes):
                state = "DEFERRED"
                portable["hash_verification"] = "ASSET_TOO_LARGE_FOR_EXPORT_HASH_CHECK"
            else:
                actual = file_sha256(found)
                portable["observed_sha256"] = actual
                if actual == expected:
                    state = "MATCHED"
                    portable["hash_verified"] = True
                    portable["hash_verification"] = "FULL_FILE_SHA256_MATCHED"
                elif role in MEASUREMENT_ROLES:
                    raise PortableError(
                        "measurement asset source changed; drift is never acceptable: %s"
                        % asset_id, "ASSET_DRIFT_NOT_ACCEPTABLE",
                        details={"asset_id": asset_id, "role": role,
                                 "expected_sha256": expected, "observed_sha256": actual,
                                 "path": str(found),
                                 "policy": "RAW_MEASUREMENT and DERIVED_ROI_MEASUREMENT cannot "
                                           "be re-baselined"})
                elif asset_id not in accepted_ids or role not in DRIFT_ACCEPTABLE_ROLES:
                    raise PortableError(
                        "declared fingerprint differs from the file on disk: %s" % asset_id,
                        "ASSET_HASH_MISMATCH",
                        details={"asset_id": asset_id, "role": role, "path": str(found),
                                 "expected_sha256": expected, "observed_sha256": actual,
                                 "hint": "re-run with --accept-source-drift <asset_id> "
                                         "--drift-reason <reason>; only AUXILIARY_CALIBRATION "
                                         "assets may be accepted"})
                else:
                    state = "DRIFT_ACCEPTED"
                    consumed_accepts.add(asset_id)
                    portable["hash_verification"] = "SOURCE_DRIFT_ACCEPTED"
                    portable["declared_asset_id"] = asset_id
                    accepted_drift.append({
                        "asset_id": asset_id,
                        "declared_asset_id": asset_id,
                        "role": role,
                        "expected_sha256": expected,
                        "observed_sha256": actual,
                        "observed_bytes": int(size),
                        "accepted_at": _now_iso(),
                        "reason": reason_text,
                    })
        portable["observed_state"] = state
        observations.append({
            "asset_id": asset_id,
            "role": role,
            "expected_sha256": expected,
            "observed_sha256": portable.get("observed_sha256"),
            "observed_state": state,
            "hash_verified": bool(portable.get("hash_verified")),
        })
        record["portable"] = portable
        finalized.append(record)
    if forbidden:
        raise PortableError(
            "SELF_CONTAINED refuses assets that may not be redistributed: %s"
            % "; ".join("%s (%s)" % (item["asset_id"], ", ".join(item["reasons"]))
                        for item in forbidden),
            "SELF_CONTAINED_FORBIDDEN_ASSET", details={"forbidden": forbidden})
    unused = accepted_ids - consumed_accepts
    if unused:
        raise PortableError(
            "--accept-source-drift names assets that were not observed as drifted: %s"
            % sorted(unused), "ASSET_DRIFT_NOT_APPLICABLE",
            details={"asset_ids": sorted(unused),
                     "hint": "acceptance is only valid for an existing AUXILIARY_CALIBRATION "
                             "asset whose observed whole-file sha256 differs from the snapshot"})
    source_integrity = {
        "schema": SOURCE_INTEGRITY_SCHEMA,
        "clean": not accepted_drift,
        "checked_at": _now_iso(),
        "policy": {
            "declared_sha256_source": "PROJECT_SNAPSHOT_DECLARED_SHA256",
            "measurement_roles_never_accept_drift": list(MEASUREMENT_ROLES),
            "drift_acceptable_roles": list(DRIFT_ACCEPTABLE_ROLES),
            "asset_id_basis": "EXPECTED_SHA256_FROM_PROJECT_SNAPSHOT",
        },
        "accepted_drift": accepted_drift,
        "assets": observations,
    }
    result = dict(document)
    result["assets"] = finalized
    result["distribution_mode"] = distribution_mode
    result["source_domain"] = source_domain
    result["counts"] = _asset_counts(finalized)
    result["measured_at"] = _now_iso()
    result["source_integrity"] = source_integrity
    return result, finalized, source_integrity
def _readme_text(*, mode: str, project_block: Mapping[str, Any], asset_count: int,
                 source_domain: str) -> str:
    lines = [
        "W154 PORTABLE PROJECT PACKAGE (format %s)" % FORMAT_VERSION,
        "",
        "Project : %s" % project_block.get("project_id"),
        "Version : %s" % project_block.get("project_version"),
        "Hash    : %s" % project_block.get("content_hash"),
        "Mode    : %s" % mode,
        "Domain  : %s" % source_domain,
        "Assets  : %d registered (see assets/assets.json)" % asset_count,
        "",
        "WHAT THIS PACKAGE CONTAINS",
        "  project/     project bundle snapshot (head.json + one version directory)",
        "  evidence/    W153 evidence sidecar and its explicit project binding",
        "  assets/      W152 nine-field asset records (locator + full-file SHA-256)",
        "  coordinates/ coordinate basis, local origin and ROI bounds",
        "  licenses/    licence and redistribution statements, per asset",
        "  RECOVERY.md  how to reopen, verify and recover",
        "",
        "WHAT IT DOES NOT CONTAIN",
        "  Real point clouds and ROI measurement bytes are NOT copied in",
        "  REFERENCE_ONLY mode.  They stay where they are and are checked by",
        "  whole-file SHA-256.  Missing or replaced sources are reported, never",
        "  replaced by synthetic data.",
        "",
        "COMMANDS (python -m fractureflow.surface_product.portable_project ...)",
        "  verify --package <this directory>",
        "  open   --package <this directory> [--repo-root DIR] [--source-root DIR]",
        "  relocate --package <this directory> --asset-id ID --path FILE",
        "  export-delivery --package <this directory> --out DIR",
        "",
        "Open states: FULL / FULL_WITH_WARNINGS / RESTRICTED / REPLACED /",
        "RELOCATED / NEEDS_RECOMPUTE / CORRUPT.  REPLACED and CORRUPT exit",
        "non-zero.  FULL_WITH_WARNINGS means an auxiliary source drift was",
        "explicitly accepted at export: ok=true but clean=false, and it is never",
        "reported as FULL.  NEEDS_RECOMPUTE or RESTRICTED means evidence is NOT",
        "current.",
    ]
    return chr(10).join(lines) + chr(10)


def _recovery_text(*, mode: str, project_block: Mapping[str, Any],
                   evidence: Mapping[str, Any], assets: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Recovery instructions",
        "",
        "Package format %s, generator %s %s." % (FORMAT_VERSION, GENERATOR_NAME, GENERATOR_VERSION),
        "Project %s version %s." % (project_block.get("project_id"),
                                    project_block.get("project_version")),
        "",
        "## 1. Verify integrity first",
        "",
        "    python -m fractureflow.surface_product.portable_project verify --package .",
        "",
        "verify re-hashes every file listed in portable_manifest.json, reloads the",
        "project bundle through the public ProjectStore reader and re-checks the",
        "evidence binding.  A non-zero exit or status CORRUPT means do not use the",
        "package: do not edit the manifest to make it pass.",
        "",
        "## 2. Reopen",
        "",
        "    python -m fractureflow.surface_product.portable_project open --package .",
        "",
        "Open never reads the original build directory.  It reads project/,",
        "evidence/ and the coordinate metadata in this package only.",
        "",
        "## 3. Sources missing or moved",
        "",
        "Every asset is located by locator plus whole-file SHA-256.  When a source",
        "is missing, open reports RESTRICTED and REQUEST_RELOCATION: geometry stays",
        "viewable, synthetic support points are never substituted, and evidence is",
        "not current.  To point at a moved copy:",
        "",
        "    python -m fractureflow.surface_product.portable_project relocate \\",
        "        --package . --asset-id <id> --path <file>",
        "",
        "relocate only accepts a file whose whole-file SHA-256 equals the recorded",
        "fingerprint (HASH_RELOCATE_OK) and appends an audit line under state/.",
        "A file with a different hash is refused; it is never silently substituted.",
        "",
        "## 3b. Source drift (expected vs observed)",
        "",
        "portable_manifest.json records source_integrity: the snapshot-declared",
        "SHA-256 is expected, the export-time measurement is observed.  At open a",
        "file whose bytes differ from expected is REPLACED (exit 4) and silent",
        "substitution is blocked.  An AUXILIARY_CALIBRATION drift may have been",
        "accepted explicitly at export: open then reports FULL_WITH_WARNINGS with",
        "clean=false and a SOURCE_DRIFT_ACCEPTED warning.  Measurement assets",
        "(RAW_MEASUREMENT / DERIVED_ROI_MEASUREMENT) can never be accepted;",
        "recover their original bytes or the package is not clean.",
        "",
        "## 4. Evidence state",
        "",
        "evidence binding: %s" % evidence.get("binding"),
        "snapshot id      : %s" % evidence.get("snapshot_id"),
        "",
        "A package whose sidecar is absent or whose binding does not verify opens",
        "as NEEDS_RECOMPUTE: evidence is not current until it is recomputed from",
        "the current geometry.  If the sources needed for recomputation are also",
        "missing, open reports RESTRICTED instead.",
        "",
        "## 5. Distribution",
        "",
        "Mode %s.  Redistributable assets: %d of %d." % (
            mode,
            sum(1 for item in assets
                if (item.get("license") or {}).get("redistribution_allowed") is True),
            len(assets)),
        "Do not redistribute this package further unless every asset licence says",
        "redistribution_allowed is true.",
        "",
        "## 6. Known boundaries",
        "",
        "* No cloud sync, no account system, no cross-machine acceptance is claimed",
        "  by this file.  A second machine is a separate measurement.",
        "* The package carries metadata and the project snapshot; it does not make",
        "  the source measurement bytes portable in REFERENCE_ONLY mode.",
        "* Format migration is explicit: an unknown major version is refused rather",
        "  than guessed.",
    ]
    return chr(10).join(lines) + chr(10)


def export_portable(package_path: os.PathLike[str] | str,
                    bundle_path: os.PathLike[str] | str, *,
                    distribution_mode: str = REFERENCE_ONLY,
                    assets_document: Optional[Mapping[str, Any]] = None,
                    license_evidence: str = "",
                    contract_sha256: Optional[str] = None,
                    asset_roots: Sequence[os.PathLike[str] | str] = (),
                    verify_asset_hashes: bool = True,
                    asset_hash_limit_bytes: int = 512 * 1024 * 1024,
                    accept_source_drift: Sequence[str] = (),
                    drift_reason: str = "",
                    fault: Optional[str] = None) -> Dict[str, Any]:
    """Build, self-verify and atomically publish one portable package."""
    mode = str(distribution_mode or REFERENCE_ONLY)
    if mode not in DISTRIBUTION_MODES:
        raise PortableError("distribution_mode must be one of %s" % (DISTRIBUTION_MODES,),
                            "USAGE_ERROR")
    fault = _fault_mode(fault)
    package = Path(package_path)
    if package.suffix.lower() != ".ffportable":
        raise PortableError("package path must end with .ffportable: %s" % package, "USAGE_ERROR")
    if package.exists():
        raise PortableError("package target already exists; refusing to overwrite: %s" % package,
                            "PACKAGE_EXISTS")
    bundle = Path(bundle_path)
    if not (bundle / "head.json").is_file():
        raise PortableError("project bundle has no head.json: %s" % bundle,
                            "PROJECT_BUNDLE_NOT_FOUND")
    head = _read_json_strict(bundle / "head.json", max_bytes=LIMITS["max_manifest_bytes"])
    project, receipt = _load_bundle(bundle)
    tag = head.get("version_dir")
    version_dir = bundle / "versions" / str(tag)
    if not version_dir.is_dir():
        raise PortableError("published version directory is missing: %s" % version_dir,
                            "PROJECT_BUNDLE_INVALID")
    snapshot_sha = file_sha256(version_dir / "snapshot.json")
    if snapshot_sha != head.get("snapshot_sha256"):
        raise PortableError("snapshot sha256 differs from head", "BUNDLE_HASH_MISMATCH")

    source_domain = _source_domain(project)
    if assets_document is None:
        document = derive_assets_from_project(project, license_evidence=license_evidence,
                                              contract_sha256=contract_sha256)
        records = normalize_asset_document(document)
    else:
        records = normalize_asset_document(assets_document)
        document = dict(assets_document)
    mismatches = _snapshot_asset_mismatches(records, project, source_domain=source_domain)
    if mismatches:
        raise PortableError(
            "assets document disagrees with the project snapshot source declarations; "
            "refusing to export a package whose expected hashes were laundered",
            "ASSET_HASH_MISMATCH",
            details={"mismatches": mismatches,
                     "policy": "EXPECTED_SHA256_MUST_COME_FROM_PROJECT_SNAPSHOT"})

    roots = [Path(item) for item in asset_roots] or [Path.cwd()]
    assets_doc, finalized, source_integrity = _finalize_assets(
        document, records, distribution_mode=mode, source_domain=source_domain,
        asset_roots=roots, verify_asset_hashes=verify_asset_hashes,
        asset_hash_limit_bytes=asset_hash_limit_bytes,
        accept_source_drift=accept_source_drift, drift_reason=drift_reason)

    coordinates_doc = build_coordinate_basis(project, finalized, source_domain=source_domain)
    licenses_doc = build_licenses(project, finalized, distribution_mode=mode)
    sidecar_path = _sidecar_path_for(bundle, head)
    sidecar_payload: Optional[Dict[str, Any]] = None
    sidecar_bytes: Optional[bytes] = None
    if sidecar_path is not None:
        sidecar_payload = _read_json_strict(sidecar_path)
        sidecar_bytes = sidecar_path.read_bytes()
    evidence = _evaluate_evidence(project, sidecar_payload)

    project_block = {
        "project_id": project.project_id,
        "project_version": project.project_version,
        "content_hash": project.compute_content_hash(),
        "schema_version": project.schema_version,
        "bundle_schema": head.get("schema"),
        "bundle_id": bundle.name,
        "version_dir": tag,
        "snapshot_sha256": snapshot_sha,
        "head_sha256": file_sha256(bundle / "head.json"),
        "object_counts": {
            "observations": len(project.observations),
            "fragments": len(project.fragments),
            "surfaces": len(project.surfaces),
            "relations": len(project.relations),
            "protected_regions": len(project.protected_regions),
        },
        "geometry_digest": _geometry_digest(project),
    }

    package_id = "%s_v%06d_%s_%s" % (_slug(project.project_id, 48), project.project_version,
                                     project_block["content_hash"][:12], uuid.uuid4().hex[:8])
    created_at = _now_iso()
    parent = package.parent
    if not parent.is_dir():
        _mkdir_checked(parent, parents=True, exist_ok=True,
                       action="create parent directory")
    staging = parent / (".ffportable_staging_%s_%s" % (package.name, uuid.uuid4().hex[:10]))
    _mkdir_checked(staging, parents=False, exist_ok=False,
                   action="create staging directory")
    published = False
    try:
        staging_project = staging / "project"
        project_copy = _copy_project_bundle(bundle, head, staging_project)
        payload_files: Dict[str, Dict[str, Any]] = dict(project_copy["files"])

        def add_payload(relative: str, result: Mapping[str, Any]) -> None:
            payload_files[relative] = {"sha256": str(result["sha256"]),
                                       "bytes": int(result["bytes"])}

        evidence_block: Dict[str, Any] = {
            "present": False, "path": None, "sha256": None,
            "binding": evidence["binding"], "reasons": list(evidence["reasons"]),
            "snapshot_id": evidence.get("snapshot_id"),
            "record_count": int(evidence.get("record_count") or 0),
            "status_counts": dict(evidence.get("status_counts") or {}),
            "schema": evidence.get("schema"),
        }
        if sidecar_bytes is not None:
            add_payload("evidence/evidence_ledger.json",
                        _write_bytes(staging / "evidence" / "evidence_ledger.json", sidecar_bytes))
            evidence_block.update({
                "present": True,
                "path": "evidence/evidence_ledger.json",
                "sha256": payload_files["evidence/evidence_ledger.json"]["sha256"],
            })
            binding_document = {
                "schema": BINDING_SCHEMA,
                "measured_at": created_at,
                "package_id": package_id,
                "project_id": project.project_id,
                "project_version": project.project_version,
                "project_content_hash": project_block["content_hash"],
                "sidecar_path": "evidence/evidence_ledger.json",
                "sidecar_sha256": evidence_block["sha256"],
                "sidecar_schema": evidence.get("schema"),
                "snapshot_id": evidence.get("snapshot_id"),
                "expected_snapshot_id": evidence.get("expected_snapshot_id"),
                "record_count": int(evidence.get("record_count") or 0),
                "status_counts": dict(evidence.get("status_counts") or {}),
                "binding": evidence["binding"],
                "reasons": list(evidence["reasons"]),
                "current": evidence["binding"] == "VALID",
                "not_claimed": ["A recorded binding is not a promise that the evidence is current."],
            }
            add_payload("evidence/binding.json",
                        _write_json(staging / "evidence" / "binding.json", binding_document))

        packaged_bytes = 0
        for record in finalized:
            portable = dict(record.get("portable") or {})
            if portable.get("storage") != "PACKAGED":
                continue
            source_value = str((record.get("source_locator") or {}).get("value") or "")
            source = next((Path(root) / source_value for root in roots
                           if (Path(root) / source_value).is_file()), None)
            if source is None:
                raise PortableError("SELF_CONTAINED asset is not present on disk: %s"
                                    % record.get("asset_id"), "ASSET_MISSING",
                                    details={"asset_id": record.get("asset_id")})
            size = source.stat().st_size
            if size > LIMITS["max_packaged_asset_bytes"]:
                raise PortableError("asset exceeds the packaged-asset byte limit: %s"
                                    % record.get("asset_id"), "RESOURCE_LIMIT_EXCEEDED")
            destination = staging / str(portable["packaged_path"])
            result = _copy_file_verified(source, destination)
            if result["sha256"] != str((record.get("fingerprint") or {}).get("value")):
                raise PortableError("packaged asset bytes do not match the fingerprint: %s"
                                    % record.get("asset_id"), "ASSET_HASH_MISMATCH")
            portable["hash_verified"] = True
            portable["hash_verification"] = "FULL_FILE_SHA256_MATCHED"
            record["portable"] = portable
            packaged_bytes += int(result["bytes"])
            add_payload(str(portable["packaged_path"]), result)
        if mode == REFERENCE_ONLY and packaged_bytes:
            raise PortableError("REFERENCE_ONLY must not copy asset bytes",
                                "REFERENCE_MODE_ASSET_BYTES_PRESENT")

        assets_doc["assets"] = finalized
        assets_doc["counts"] = _asset_counts(finalized)
        assets_doc["measured_at"] = created_at
        assets_doc["distribution_mode"] = mode
        assets_doc["source_domain"] = source_domain
        assets_doc["source_integrity"] = source_integrity
        add_payload("assets/assets.json",
                    _write_json(staging / "assets" / "assets.json", assets_doc))
        add_payload("coordinates/coordinate_basis.json",
                    _write_json(staging / "coordinates" / "coordinate_basis.json", coordinates_doc))
        add_payload("licenses/licenses.json",
                    _write_json(staging / "licenses" / "licenses.json", licenses_doc))
        add_payload("README.txt", _write_bytes(
            staging / "README.txt",
            _readme_text(mode=mode, project_block=project_block,
                         asset_count=len(finalized), source_domain=source_domain).encode("utf-8")))
        add_payload("RECOVERY.md", _write_bytes(
            staging / "RECOVERY.md",
            _recovery_text(mode=mode, project_block=project_block,
                           evidence=evidence_block, assets=finalized).encode("utf-8")))

        if fault == "after_staging":
            raise PortableError("test fault injected after the staging build",
                                "TEST_FAULT_AFTER_STAGING")

        written = _walk_files(staging)
        if len(written) != len(payload_files):
            raise PortableError(
                "staged file count %d differs from the payload index %d"
                % (len(written), len(payload_files)), "FILE_INDEX_MISMATCH")
        missing = sorted(set(payload_files) - set(written))
        if missing:
            raise PortableError("staged payload files are missing: %s" % missing[:8],
                                "FILE_INDEX_MISMATCH")
        total_bytes = _byte_size(staging, written)
        if len(written) > LIMITS["max_package_files"]:
            raise PortableError("package has more than %d files" % LIMITS["max_package_files"],
                                "RESOURCE_LIMIT_EXCEEDED")
        if total_bytes > LIMITS["max_package_bytes"]:
            raise PortableError("package exceeds %d bytes" % LIMITS["max_package_bytes"],
                                "RESOURCE_LIMIT_EXCEEDED")
        files_index = {relative: {"sha256": file_sha256(staging / relative),
                                  "bytes": Path(staging / relative).stat().st_size}
                       for relative in written}
        manifest: Dict[str, Any] = {
            "schema": PORTABLE_SCHEMA,
            "format_version": FORMAT_VERSION,
            "generator": {
                "name": GENERATOR_NAME,
                "version": GENERATOR_VERSION,
                "runtime": "python %d.%d" % sys.version_info[:2],
            },
            "created_at": created_at,
            "package_id": package_id,
            "distribution_mode": mode,
            "source_domain": source_domain,
            "project": project_block,
            "evidence": evidence_block,
            "assets": {
                "path": "assets/assets.json",
                "sha256": payload_files["assets/assets.json"]["sha256"],
                "asset_count": len(finalized),
                "by_role": dict(_asset_counts(finalized)["by_role"]),
                "by_storage": dict(_asset_counts(finalized)["by_storage"]),
                "redistributable_count": int(_asset_counts(finalized)["redistributable_count"]),
                "packaged_bytes": packaged_bytes,
            },
            "coordinates": {
                "path": "coordinates/coordinate_basis.json",
                "sha256": payload_files["coordinates/coordinate_basis.json"]["sha256"],
                "display_convention": coordinates_doc["display_convention"],
                "georeferencing": coordinates_doc["georeferencing"],
            },
            "licenses": {
                "path": "licenses/licenses.json",
                "sha256": payload_files["licenses/licenses.json"]["sha256"],
                "redistribution_allowed": licenses_doc["project_license"]["redistribution_allowed"],
                "training_allowed": False,
            },
            "recovery": {
                "readme": "README.txt",
                "readme_sha256": payload_files["README.txt"]["sha256"],
                "recovery": "RECOVERY.md",
                "recovery_sha256": payload_files["RECOVERY.md"]["sha256"],
            },
            "source_integrity": source_integrity,
            "files": files_index,
            "file_count": len(files_index),
            "total_bytes": total_bytes,
            "limits": dict(LIMITS),
            "compatibility": {
                "min_reader_major": READABLE_MAJOR,
                "migration": "explicit only; an unknown major version is refused, never guessed",
            },
            "manifest_body_sha256": None,
        }
        manifest["manifest_body_sha256"] = _manifest_body_sha256(manifest)
        _write_json(staging / MANIFEST_NAME, manifest)
        _write_bytes(staging / PUBLICATION_MARKER,
                     (manifest["manifest_body_sha256"] + chr(10)).encode("utf-8"))

        if fault == "before_publish":
            raise PortableError("test fault injected before the atomic publish",
                                "TEST_FAULT_BEFORE_PUBLISH")

        staged = verify_package(staging)
        if staged["status"] != "OK":
            raise PortableError("staged package failed self-verification: %s"
                                % [item["id"] for item in staged["checks"] if not item["ok"]],
                                "FILE_INDEX_MISMATCH",
                                details={"failed_checks": staged["failed_checks"]})
        if package.exists():
            raise PortableError("package target appeared during export: %s" % package,
                                "PACKAGE_EXISTS")
        _rename_checked(staging, package, action="publish package")
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)

    final = verify_package(package)
    if final["status"] != "OK":
        raise PortableError("published package failed read-back verification",
                            "FILE_INDEX_MISMATCH",
                            details={"failed_checks": final["failed_checks"]})
    manifest = _read_json_strict(package / MANIFEST_NAME,
                                max_bytes=LIMITS["max_manifest_bytes"])
    marker = (package / PUBLICATION_MARKER).read_text(encoding="utf-8").strip()
    if marker != manifest["manifest_body_sha256"]:
        raise PortableError("published marker disagrees with the manifest body sha256",
                            "PUBLICATION_MARKER_MISMATCH")
    receipt_document = {
        "schema": EXPORT_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "status": "EXPORTED",
        "ok": True,
        "exit_code": EXIT_OK,
        "package": {
            "path": str(package),
            "package_id": manifest["package_id"],
            "manifest_sha256": file_sha256(package / MANIFEST_NAME),
            "manifest_body_sha256": manifest["manifest_body_sha256"],
            "file_count": int(manifest["file_count"]),
            "total_bytes": int(manifest["total_bytes"]),
        },
        "project": dict(manifest["project"]),
        "distribution_mode": mode,
        "source_domain": source_domain,
        "assets": {
            "asset_count": len(finalized),
            "by_role": dict(_asset_counts(finalized)["by_role"]),
            "by_storage": dict(_asset_counts(finalized)["by_storage"]),
            "reference_count": sum(1 for item in finalized
                                   if (item.get("portable") or {}).get("storage") == "REFERENCE"),
            "packaged_count": sum(1 for item in finalized
                                  if (item.get("portable") or {}).get("storage") == "PACKAGED"),
            "hash_verified_count": sum(1 for item in finalized
                                       if (item.get("portable") or {}).get("hash_verified") is True),
            "hash_deferred": [
                {"asset_id": item.get("asset_id"),
                 "reason": (item.get("portable") or {}).get("hash_verification")}
                for item in finalized
                if (item.get("portable") or {}).get("hash_verified") is not True
            ],
            "packaged_bytes": packaged_bytes,
        },
        "source_integrity": {
            "clean": bool(source_integrity["clean"]),
            "accepted_drift": [dict(item) for item in source_integrity["accepted_drift"]],
        },
        "evidence": dict(evidence_block),
        "copied_asset_bytes": bool(packaged_bytes),
        "raw_measurement_bytes_copied": sum(
            int((item.get("portable") or {}).get("observed_bytes") or 0)
            for item in finalized
            if (item.get("portable") or {}).get("storage") == "PACKAGED"
            and item.get("role") in MEASUREMENT_ROLES),
        "verification": {
            "status": final["status"],
            "checked_files": final["counts"]["checked_files"],
            "enumerated_checks": len(final["checks"]),
        },
        "recompute_script": "src/fractureflow/surface_product/portable_project.py",
        "not_claimed": [
            "Export is not cross-machine acceptance; a second machine is a separate measurement.",
            "REFERENCE_ONLY does not make measurement bytes portable.",
        ],
    }
    return receipt_document


# -- verification ------------------------------------------------------------

def _load_manifest(package: Path, *, require_published: bool = True) -> Dict[str, Any]:
    package = Path(package)
    if not package.is_dir():
        raise PortableError("package directory is missing: %s" % package, "PACKAGE_NOT_FOUND")
    manifest_path = package / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PortableError("portable manifest is missing: %s" % manifest_path, "MANIFEST_MISSING")
    manifest = _read_json_strict(manifest_path, max_bytes=LIMITS["max_manifest_bytes"])
    if not isinstance(manifest, Mapping):
        raise PortableError("portable manifest must be one JSON object", "MANIFEST_INVALID")
    if manifest.get("schema") != PORTABLE_SCHEMA:
        raise PortableError("package schema %r is not %r"
                            % (manifest.get("schema"), PORTABLE_SCHEMA), "SCHEMA_UNSUPPORTED")
    version = manifest.get("format_version")
    if not isinstance(version, str) or version.split(".")[0] != READABLE_MAJOR:
        raise PortableError("format_version %r is not readable by format %s"
                            % (version, FORMAT_VERSION), "FORMAT_VERSION_UNSUPPORTED")
    recorded = manifest.get("manifest_body_sha256")
    actual = _manifest_body_sha256(manifest)
    if recorded != actual:
        raise PortableError("manifest body sha256 differs from the recorded value",
                            "MANIFEST_BODY_HASH_MISMATCH",
                            details={"recorded": recorded, "actual": actual})
    if require_published:
        marker = package / PUBLICATION_MARKER
        if not marker.is_file():
            raise PortableError("package is not published (no %s marker); an interrupted "
                                "export must never be read as a formal result" % PUBLICATION_MARKER,
                                "PUBLICATION_MARKER_MISSING")
        content = marker.read_text(encoding="utf-8").strip()
        if content != actual:
            raise PortableError("publication marker does not match the manifest body sha256",
                                "PUBLICATION_MARKER_MISMATCH")
    return dict(manifest)


def _index_entries(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise PortableError("manifest enumerates no files", "FILE_INDEX_EMPTY")
    result: Dict[str, Mapping[str, Any]] = {}
    for relative, entry in files.items():
        relative = _safe_relpath(relative, "manifest.files key")
        if not isinstance(entry, Mapping):
            raise PortableError("manifest.files[%r] must be an object" % relative,
                                "MANIFEST_INVALID")
        value = str(entry.get("sha256") or "")
        if not SHA256_RE.match(value):
            raise PortableError("manifest.files[%r].sha256 is not a sha256" % relative,
                                "MANIFEST_INVALID")
        result[relative] = entry
    return result


def _check_package_index(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    package = Path(package)
    index = _index_entries(manifest)
    checked = 0
    total = 0
    for relative, entry in sorted(index.items()):
        target = package / relative
        if not target.is_file():
            raise PortableError("indexed file is missing: %s" % relative, "FILE_INDEX_MISMATCH")
        actual = file_sha256(target)
        if actual != str(entry.get("sha256")):
            raise PortableError("hash mismatch for %s" % relative, "FILE_INDEX_MISMATCH",
                                details={"relative": relative,
                                         "expected": entry.get("sha256"), "actual": actual})
        checked += 1
        total += target.stat().st_size
    written = _walk_files(package)
    allowed = set(index) | {MANIFEST_NAME, PUBLICATION_MARKER}
    extras = [item for item in written
              if item not in allowed and not item.startswith("state/")]
    if extras:
        raise PortableError("package carries files that the manifest does not index: %s"
                            % extras[:8], "UNEXPECTED_PACKAGE_FILE")
    state_files = [item for item in written if item.startswith("state/")]
    return {"checked_files": checked, "total_bytes": total, "extra_state_files": len(state_files)}


def _check_package_layout(package: Path) -> Dict[str, Any]:
    package = Path(package)
    entries = sorted(os.scandir(package), key=lambda item: item.name)
    unexpected = [entry.name for entry in entries if entry.name not in TOP_LEVEL_ALLOWED]
    if unexpected:
        raise PortableError("unexpected top-level package entries: %s" % unexpected,
                            "UNEXPECTED_PACKAGE_FILE")
    missing = [relative for relative in REQUIRED_LAYOUT if not (package / relative).is_file()]
    if missing:
        raise PortableError("required package entries are missing: %s" % missing,
                            "FILE_INDEX_MISMATCH")
    return {"top_level": [entry.name for entry in entries]}


def _check_package_scripts(package: Path) -> Dict[str, Any]:
    found = []
    for relative in _walk_files(package):
        suffix = os.path.splitext(relative)[1].lower()
        if suffix in FORBIDDEN_PACKAGE_EXTENSIONS:
            found.append(relative)
    if found:
        raise PortableError("package carries executable or script files: %s" % found[:8],
                            "SCRIPT_IN_PACKAGE")
    return {"forbidden_extensions": sorted(FORBIDDEN_PACKAGE_EXTENSIONS)}


def _check_package_json_finite(package: Path) -> Dict[str, Any]:
    parsed = 0
    for relative in _walk_files(package):
        if not relative.lower().endswith(".json"):
            continue
        target = package / relative
        if target.stat().st_size > LIMITS["max_json_bytes"]:
            raise PortableError("JSON document exceeds the byte limit: %s" % relative,
                                "RESOURCE_LIMIT_EXCEEDED")
        _read_json_strict(target)
        parsed += 1
    if parsed <= 0:
        raise PortableError("package carries no JSON document at all", "EMPTY_ENUMERATION")
    return {"json_documents": parsed}


def _check_package_project(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    project, receipt = _load_bundle(Path(package) / "project")
    block = dict(manifest.get("project") or {})
    checks = {
        "project_id": project.project_id,
        "project_version": project.project_version,
        "content_hash": project.compute_content_hash(),
    }
    for key, value in checks.items():
        if value != block.get(key):
            raise PortableError("project %s %r differs from the manifest %r"
                                % (key, value, block.get(key)), "PROJECT_BUNDLE_INVALID")
    snapshot_sha = file_sha256(Path(package) / "project" / "versions"
                               / str(block.get("version_dir")) / "snapshot.json")
    if snapshot_sha != block.get("snapshot_sha256"):
        raise PortableError("snapshot sha256 differs from the manifest", "BUNDLE_HASH_MISMATCH")
    return {"verified": True, "enumerated_files": receipt.get("enumerated_files"),
            "object_counts": dict(block.get("object_counts") or {})}


def _check_package_evidence(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    block = dict(manifest.get("evidence") or {})
    package = Path(package)
    if not block.get("present"):
        legacy = str(manifest.get("format_version") or "") == "1.0.0"
        detail = {"binding": "MISSING",
                  "reasons": ["EVIDENCE_SIDECAR_ABSENT"],
                  "legacy_absent": bool(legacy),
                  "legacy_format": "1.0.0" if legacy else None}
        if legacy:
            detail["reasons"].append("LEGACY_ABSENT")
        return detail
    path = package / "evidence" / "evidence_ledger.json"
    if not path.is_file():
        raise PortableError("manifest records an evidence sidecar that is absent",
                            "FILE_INDEX_MISMATCH")
    if file_sha256(path) != block.get("sha256"):
        raise PortableError("evidence sidecar hash differs from the manifest", "FILE_INDEX_MISMATCH")
    payload = _read_json_strict(path)
    project, _receipt = _load_bundle(package / "project")
    evaluated = _evaluate_evidence(project, payload)
    if evaluated["binding"] != block.get("binding"):
        evaluated["manifest_binding"] = block.get("binding")
    if payload.get("snapshot_id") != block.get("snapshot_id"):
        evaluated["reasons"] = list(evaluated.get("reasons") or []) + ["MANIFEST_SNAPSHOT_ID_DIFFERS"]
    return evaluated


def _check_package_assets(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    package = Path(package)
    document = _read_json_strict(package / "assets" / "assets.json")
    records = normalize_asset_document(document)
    mode = str(manifest.get("distribution_mode") or "")
    if mode not in DISTRIBUTION_MODES:
        raise PortableError("manifest distribution_mode %r is not supported" % mode,
                            "MANIFEST_INVALID")
    source_domain = str(manifest.get("source_domain") or "unknown")
    packaged_any = False
    for record in records:
        portable = dict(record.get("portable") or {})
        storage = str(portable.get("storage") or "REFERENCE")
        if mode == REFERENCE_ONLY:
            if storage != "REFERENCE":
                raise PortableError("REFERENCE_ONLY package records a non-reference asset: %s"
                                    % record.get("asset_id"), "MANIFEST_INVALID")
        else:
            if storage != "PACKAGED":
                raise PortableError("SELF_CONTAINED package records an unpackaged asset: %s"
                                    % record.get("asset_id"), "MANIFEST_INVALID")
            packaged_any = True
            relative = _safe_relpath(portable.get("packaged_path"), "packaged_path")
            target = package / relative
            if not target.is_file():
                raise PortableError("packaged asset is missing: %s" % relative, "FILE_INDEX_MISMATCH")
            if file_sha256(target) != str((record.get("fingerprint") or {}).get("value")):
                raise PortableError("packaged asset hash differs from the fingerprint: %s"
                                    % record.get("asset_id"), "ASSET_HASH_MISMATCH")
            license_block = dict(record.get("license") or {})
            if license_block.get("redistribution_allowed") is not True:
                raise PortableError("packaged asset is not redistributable: %s"
                                    % record.get("asset_id"), "SELF_CONTAINED_FORBIDDEN_ASSET")
            if record.get("role") in MEASUREMENT_ROLES and source_domain != "synthetic":
                raise PortableError("packaged measurement asset in a non-synthetic package: %s"
                                    % record.get("asset_id"), "SELF_CONTAINED_FORBIDDEN_ASSET")
        if record.get("role") in ROLES_EXTENDED and record.get("role") not in ROLES_CONTRACT:
            portable.setdefault("contract_extensions", [])
    if mode == REFERENCE_ONLY and (package / "assets" / "files").exists():
        raise PortableError("REFERENCE_ONLY package carries asset bytes",
                            "REFERENCE_MODE_ASSET_BYTES_PRESENT")
    block = dict(manifest.get("assets") or {})
    if int(block.get("asset_count") or -1) != len(records):
        raise PortableError("manifest asset count %r differs from %d records"
                            % (block.get("asset_count"), len(records)), "MANIFEST_INVALID")
    manifest_si = manifest.get("source_integrity")
    document_si = document.get("source_integrity")
    if (manifest_si is None and document_si is None
            and str(manifest.get("format_version") or "") == "1.0.0"):
        return {"asset_count": len(records), "distribution_mode": mode,
                "packaged_assets": packaged_any, "source_integrity": "LEGACY_ABSENT"}
    if not isinstance(manifest_si, Mapping) or not isinstance(document_si, Mapping):
        raise PortableError("source_integrity block is missing from the manifest or assets.json",
                            "SOURCE_INTEGRITY_INVALID")
    accepted = list(manifest_si.get("accepted_drift") or [])
    if bool(manifest_si.get("clean")) != (len(accepted) == 0):
        raise PortableError("source_integrity.clean disagrees with accepted_drift",
                            "SOURCE_INTEGRITY_INVALID")
    record_ids = {str(record.get("asset_id")) for record in records}
    observed_states = {str(record.get("asset_id")):
                       str((record.get("portable") or {}).get("observed_state") or "")
                       for record in records}
    accepted_ids = set()
    for entry in accepted:
        if not isinstance(entry, Mapping):
            raise PortableError("accepted_drift entry is not an object",
                                "SOURCE_INTEGRITY_INVALID")
        entry_id = str(entry.get("asset_id") or "")
        if not entry_id or entry_id not in record_ids:
            raise PortableError("accepted_drift names an unknown asset: %r" % entry_id,
                                "SOURCE_INTEGRITY_INVALID")
        if str(entry.get("role")) not in DRIFT_ACCEPTABLE_ROLES:
            raise PortableError("accepted_drift role may only be AUXILIARY_CALIBRATION: %s"
                                % entry_id, "SOURCE_INTEGRITY_INVALID")
        expected_hash = str(entry.get("expected_sha256") or "")
        observed_hash = str(entry.get("observed_sha256") or "")
        if not SHA256_RE.match(expected_hash) or not SHA256_RE.match(observed_hash):
            raise PortableError("accepted_drift hashes are not sha256: %s" % entry_id,
                                "SOURCE_INTEGRITY_INVALID")
        if expected_hash == observed_hash:
            raise PortableError("accepted_drift records equal hashes: %s" % entry_id,
                                "SOURCE_INTEGRITY_INVALID")
        if not str(entry.get("reason") or "").strip():
            raise PortableError("accepted_drift has no reason: %s" % entry_id,
                                "SOURCE_INTEGRITY_INVALID")
        accepted_ids.add(entry_id)
        if observed_states.get(entry_id) != "DRIFT_ACCEPTED":
            raise PortableError("accepted_drift asset %s is not recorded as DRIFT_ACCEPTED"
                                % entry_id, "SOURCE_INTEGRITY_INVALID")
    for record_id, state in observed_states.items():
        if state == "DRIFT_ACCEPTED" and record_id not in accepted_ids:
            raise PortableError("asset %s records DRIFT_ACCEPTED without an accepted_drift entry"
                                % record_id, "SOURCE_INTEGRITY_INVALID")
    for record in records:
        fingerprint = str((record.get("fingerprint") or {}).get("value") or "")
        expected_field = str((record.get("portable") or {}).get("expected_sha256") or "")
        if expected_field and expected_field != fingerprint:
            raise PortableError("asset %s portable.expected_sha256 differs from fingerprint"
                                % record.get("asset_id"), "SOURCE_INTEGRITY_INVALID")
    manifest_key = _canonical_sha256({
        "clean": bool(manifest_si.get("clean")),
        "accepted_drift": [{key: entry.get(key) for key in (
            "asset_id", "role", "expected_sha256", "observed_sha256", "reason")}
            for entry in accepted],
        "observed_states": observed_states,
    })
    document_key = _canonical_sha256({
        "clean": bool(document_si.get("clean")),
        "accepted_drift": [{key: entry.get(key) for key in (
            "asset_id", "role", "expected_sha256", "observed_sha256", "reason")}
            for entry in (document_si.get("accepted_drift") or [])],
        "observed_states": observed_states,
    })
    if manifest_key != document_key:
        raise PortableError("manifest source_integrity differs from assets.json",
                            "SOURCE_INTEGRITY_INVALID")
    return {"asset_count": len(records), "distribution_mode": mode,
            "packaged_assets": packaged_any}


def _check_package_assets_snapshot(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Assets may not claim hashes the package's own project snapshot never declared."""
    package = Path(package)
    project, _receipt = _load_bundle(package / "project")
    document = _read_json_strict(package / "assets" / "assets.json")
    records = normalize_asset_document(document)
    source_domain = _source_domain(project)
    mismatches = _snapshot_asset_mismatches(records, project, source_domain=source_domain)
    if mismatches:
        raise PortableError(
            "assets document disagrees with the package project snapshot source declarations",
            "SOURCE_INTEGRITY_INVALID",
            details={"mismatches": mismatches, "source_domain": source_domain})
    return {"asset_count": len(records), "source_domain": source_domain,
            "snapshot_declarations": len(_snapshot_declarations(project))}


def _check_package_documents(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    package = Path(package)
    coordinates = _read_json_strict(package / "coordinates" / "coordinate_basis.json")
    if coordinates.get("schema") != COORDINATES_SCHEMA:
        raise PortableError("coordinate basis schema is unexpected", "COORDINATE_BASIS_INVALID")
    observations = coordinates.get("observations")
    if not isinstance(observations, list) or not observations:
        raise PortableError("coordinate basis enumerates no observation", "EMPTY_ENUMERATION")
    for item in observations:
        if not str(item.get("coordinate_frame") or "").strip():
            raise PortableError("coordinate frame is empty for %r" % item.get("observation_id"),
                                "COORDINATE_BASIS_INVALID")
        if not str(item.get("unit") or "").strip():
            raise PortableError("unit is empty for %r" % item.get("observation_id"),
                                "COORDINATE_BASIS_INVALID")
    licenses = _read_json_strict(package / "licenses" / "licenses.json")
    if licenses.get("schema") != LICENSES_SCHEMA:
        raise PortableError("licence document schema is unexpected", "LICENSE_INVALID")
    summary = dict(licenses.get("summary") or {})
    if int(summary.get("asset_count") or -1) != int((manifest.get("assets") or {}).get("asset_count") or -2):
        raise PortableError("licence asset count differs from the manifest", "LICENSE_INVALID")
    for row in licenses.get("assets") or []:
        if not isinstance(row.get("redistribution_allowed"), bool):
            raise PortableError("licence row has no explicit redistribution_allowed",
                                "LICENSE_INVALID")
    if not _read_text(package / "README.txt").strip():
        raise PortableError("README.txt is empty", "EMPTY_ENUMERATION")
    if not _read_text(package / "RECOVERY.md").strip():
        raise PortableError("RECOVERY.md is empty", "EMPTY_ENUMERATION")
    return {"observations": len(observations), "licence_rows": len(licenses.get("assets") or [])}


def verify_package(package_path: os.PathLike[str] | str, *,
                   require_published: bool = True) -> Dict[str, Any]:
    """Verify one package and return a check-by-check document."""
    package = Path(package_path)
    document: Dict[str, Any] = {
        "schema": VERIFY_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "package": str(package),
        "require_published": bool(require_published),
        "status": "OK",
        "ok": True,
        "exit_code": EXIT_OK,
        "manifest": None,
        "checks": [],
        "failed_checks": [],
        "problems": [],
        "counts": {},
        "evidence": {},
    }

    def check(check_id: str, function) -> bool:
        try:
            detail = function()
            document["checks"].append({"id": check_id, "ok": True, "detail": detail})
            return True
        except PortableError as exc:
            item = {"id": check_id, "ok": False, "detail": exc.message, "code": exc.code}
            problem = {"code": exc.code, "message": exc.message}
            if exc.details:
                item["details"] = copy.deepcopy(exc.details)
                problem["details"] = copy.deepcopy(exc.details)
            document["checks"].append(item)
            document["failed_checks"].append(check_id)
            document["problems"].append(problem)
            return False
        except OSError as exc:
            document["checks"].append({"id": check_id, "ok": False, "detail": str(exc),
                                       "code": "OS_ERROR"})
            document["failed_checks"].append(check_id)
            document["problems"].append({"code": "OS_ERROR", "message": str(exc)})
            return False

    try:
        manifest = _load_manifest(package, require_published=require_published)
    except PortableError as exc:
        document["checks"].append({"id": "manifest", "ok": False,
                                   "detail": exc.message, "code": exc.code})
        document["failed_checks"].append("manifest")
        document["problems"].append({"code": exc.code, "message": exc.message})
        document["status"] = "CORRUPT"
        document["ok"] = False
        document["exit_code"] = EXIT_CORRUPT
        return document
    document["manifest"] = {
        "schema": manifest.get("schema"),
        "format_version": manifest.get("format_version"),
        "package_id": manifest.get("package_id"),
        "distribution_mode": manifest.get("distribution_mode"),
        "created_at": manifest.get("created_at"),
        "manifest_body_sha256": manifest.get("manifest_body_sha256"),
        "file_count": manifest.get("file_count"),
        "total_bytes": manifest.get("total_bytes"),
        "project": dict(manifest.get("project") or {}),
        "evidence": dict(manifest.get("evidence") or {}),
    }

    check("partition_manifest", lambda: {"body_sha256": manifest.get("manifest_body_sha256")})
    check("partition_layout", lambda: _check_package_layout(package))
    check("partition_index", lambda: _check_package_index(package, manifest))
    check("partition_scripts", lambda: _check_package_scripts(package))
    check("partition_json_finite", lambda: _check_package_json_finite(package))
    check("partition_project", lambda: _check_package_project(package, manifest))
    check("partition_evidence", lambda: _check_package_evidence(package, manifest))
    check("partition_assets", lambda: _check_package_assets(package, manifest))
    check("partition_assets_snapshot", lambda: _check_package_assets_snapshot(package, manifest))
    check("partition_documents", lambda: _check_package_documents(package, manifest))
    check("partition_limits", lambda: _check_package_limits(package, manifest))
    for item in document["checks"]:
        if item["id"] == "partition_evidence" and item["ok"]:
            document["evidence"] = dict(item["detail"] or {})
    index_detail = next((item["detail"] for item in document["checks"]
                         if item["id"] == "partition_index" and item["ok"]), {})
    document["counts"] = {
        "checked_files": int((index_detail or {}).get("checked_files") or 0),
        "total_bytes": int((index_detail or {}).get("total_bytes") or 0),
        "state_files": int((index_detail or {}).get("extra_state_files") or 0),
    }
    if document["failed_checks"]:
        document["ok"] = False
        failed = set(document["failed_checks"])
        checks = {item.get("id"): item for item in document["checks"]}
        snapshot_failed = "partition_assets_snapshot" in failed
        assets_ok = bool((checks.get("partition_assets") or {}).get("ok"))
        index_assets_only = False
        if "partition_index" in failed:
            detail = json.dumps((checks.get("partition_index") or {}).get("detail"),
                                ensure_ascii=False, allow_nan=False)
            index_assets_only = ("assets/assets.json" in detail
                                 or "assets\\\\assets.json" in detail)
        source_replacement = (snapshot_failed and assets_ok
                              and (failed == {"partition_assets_snapshot"}
                                   or (failed == {"partition_assets_snapshot", "partition_index"}
                                       and index_assets_only)))
        if source_replacement:
            document["status"] = STATUS_REPLACED
            document["exit_code"] = EXIT_HASH_MISMATCH
            document["mismatches"] = list(
                (checks.get("partition_assets_snapshot", {}).get("details") or {}).get("mismatches") or [])
            document["state_rule"] = "SOURCE_DECLARATION_MISMATCH"
        else:
            document["status"] = "CORRUPT"
            document["exit_code"] = EXIT_CORRUPT
    return document


def _check_package_limits(package: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    files = _walk_files(package)
    if len(files) > LIMITS["max_package_files"]:
        raise PortableError("package has more than %d files" % LIMITS["max_package_files"],
                            "RESOURCE_LIMIT_EXCEEDED")
    total = _byte_size(package, files)
    if total > LIMITS["max_package_bytes"]:
        raise PortableError("package exceeds %d bytes" % LIMITS["max_package_bytes"],
                            "RESOURCE_LIMIT_EXCEEDED")
    declared = manifest.get("file_count")
    if declared is not None:
        indexed = len(_index_entries(manifest))
        if int(declared) != indexed:
            raise PortableError("manifest file_count %r differs from %d indexed files"
                                % (declared, indexed), "MANIFEST_INVALID")
    return {"file_count": len(files), "total_bytes": total}


# -- open state machine ------------------------------------------------------

def _relocation_audit(package: Path, *, enabled: bool = True) -> Dict[str, Dict[str, Any]]:
    if not enabled:
        return {}
    path = Path(package) / "state" / "relocation_audit.jsonl"
    if not path.is_file():
        return {}
    entries: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        entry = _parse_json_strict(text, "relocation_audit.jsonl:%d" % line_number)
        if not isinstance(entry, Mapping):
            raise PortableError("relocation audit line %d is not an object" % line_number,
                                "MANIFEST_INVALID")
        asset_id = str(entry.get("asset_id") or "")
        if asset_id:
            entries[asset_id] = dict(entry)
    return entries


def _candidate_paths(record: Mapping[str, Any], *, package: Path,
                     repo_root: Optional[Path],
                     source_roots: Sequence[Path],
                     audit_entry: Optional[Mapping[str, Any]]) -> List[Tuple[str, Path]]:
    portable = dict(record.get("portable") or {})
    locator = dict(record.get("source_locator") or {})
    value = str(locator.get("value") or "")
    mode = str(locator.get("mode") or "")
    candidates: List[Tuple[str, Path]] = []
    if str(portable.get("storage") or "REFERENCE") == "PACKAGED" and portable.get("packaged_path"):
        candidates.append(("package", Path(package) / _safe_relpath(portable["packaged_path"],
                                                                    "packaged_path")))
    if value:
        if mode in RELATIVE_LOCATOR_MODES and _is_safe_relative(value):
            if repo_root is not None:
                candidates.append(("repo_root", Path(repo_root) / value))
            for root in source_roots:
                candidates.append(("source_root", Path(root) / value))
            for root in source_roots:
                candidates.append(("source_root_basename",
                                   Path(root) / os.path.basename(value)))
        elif mode == "external_uri" and value.startswith("file://"):
            candidates.append(("external_uri", Path(value[len("file://"):])))
    hint = portable.get("last_seen_absolute")
    if isinstance(hint, str) and hint:
        candidates.append(("last_seen_absolute", Path(hint)))
    if isinstance(hint, list):
        for item in hint:
            if isinstance(item, str) and item:
                candidates.append(("last_seen_absolute", Path(item)))
    if audit_entry and audit_entry.get("path"):
        candidates.append(("relocation_audit", Path(str(audit_entry["path"]))))
    return candidates


DECLARED_ORIGINS = ("repo_root", "source_root", "external_uri", "last_seen_absolute")


def _hash_candidate(candidate: Path, expected: str,
                    accepted_entry: Optional[Mapping[str, Any]],
                    role: str, hash_limit_bytes: int) -> Dict[str, Any]:
    size = int(candidate.stat().st_size)
    if hash_limit_bytes and size > int(hash_limit_bytes):
        return {"bytes": size, "hash_verified": False,
                "hash_verification": "DEFERRED_BY_HASH_LIMIT"}
    actual = file_sha256(candidate)
    if actual == expected:
        return {"bytes": size, "sha256_actual": actual, "hash_verified": True,
                "hash_verification": "FULL_FILE_SHA256_MATCHED"}
    if (accepted_entry is not None and role in DRIFT_ACCEPTABLE_ROLES
            and actual == str(accepted_entry.get("observed_sha256") or "")):
        return {"bytes": size, "sha256_actual": actual, "hash_verified": False,
                "hash_verification": "SOURCE_DRIFT_ACCEPTED", "drift_accepted": True}
    return {"bytes": size, "sha256_actual": actual, "hash_verified": True,
            "hash_verification": "FULL_FILE_SHA256_MISMATCH"}


def _resolve_assets(records: Sequence[Mapping[str, Any]], *, package: Path,
                    repo_root: Optional[Path], source_roots: Sequence[Path],
                    audit: Mapping[str, Mapping[str, Any]],
                    accepted: Optional[Mapping[str, Mapping[str, Any]]] = None,
                    hash_limit_bytes: int = 0) -> List[Dict[str, Any]]:
    """Resolve each asset, checking declared locations before any shadow copy.

    A declared path that exists with different bytes always wins over a
    hash-matching copy elsewhere: silent substitution stays blocked. Only an
    explicit relocation-audit entry may carry a package across a replaced
    declared path, and that yields RELOCATED plus a declared_mismatch record.
    """
    accepted_map = dict(accepted or {})
    items: List[Dict[str, Any]] = []
    for record in records:
        fingerprint = dict(record.get("fingerprint") or {})
        expected = str(fingerprint.get("value") or "")
        asset_id = str(record.get("asset_id"))
        role = str(record.get("role"))
        accepted_entry = accepted_map.get(asset_id)
        existing: List[Tuple[str, Path]] = []
        skipped: List[str] = []
        for origin, candidate in _candidate_paths(record, package=package,
                                                  repo_root=repo_root,
                                                  source_roots=source_roots,
                                                  audit_entry=audit.get(asset_id)):
            try:
                if candidate.is_symlink():
                    skipped.append("%s: %s" % (origin, "symlink not followed"))
                    continue
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            existing.append((origin, candidate))

        package_hit = next(((o, p) for o, p in existing if o == "package"), None)
        audit_hit = next(((o, p) for o, p in existing if o == "relocation_audit"), None)
        chosen: Optional[Dict[str, Any]] = None
        mismatch: Optional[Dict[str, Any]] = None

        def _match(origin: str, candidate: Path, outcome: Mapping[str, Any]) -> Dict[str, Any]:
            return {"origin": origin, "path": str(candidate), **dict(outcome)}

        def _is_match(outcome: Mapping[str, Any]) -> bool:
            return str(outcome.get("hash_verification")) in (
                "FULL_FILE_SHA256_MATCHED", "SOURCE_DRIFT_ACCEPTED")

        declared_match: Optional[Dict[str, Any]] = None
        declared_deferred: Optional[Dict[str, Any]] = None
        declared_mismatch: Optional[Dict[str, Any]] = None
        for origin, candidate in existing:
            if origin == "package" or origin not in DECLARED_ORIGINS:
                continue
            outcome = _hash_candidate(candidate, expected, accepted_entry, role,
                                      hash_limit_bytes)
            if _is_match(outcome):
                if declared_match is None:
                    declared_match = _match(origin, candidate, outcome)
            elif outcome.get("hash_verification") == "DEFERRED_BY_HASH_LIMIT":
                if declared_deferred is None:
                    declared_deferred = _match(origin, candidate, outcome)
            else:
                if declared_mismatch is None:
                    declared_mismatch = _match(origin, candidate, outcome)

        if package_hit is not None:
            origin, candidate = package_hit
            outcome = _hash_candidate(candidate, expected, accepted_entry, role,
                                      hash_limit_bytes)
            if _is_match(outcome) or outcome.get("hash_verification") == "DEFERRED_BY_HASH_LIMIT":
                chosen = _match(origin, candidate, outcome)
            else:
                mismatch = _match(origin, candidate, outcome)
        elif declared_mismatch is not None:
            mismatch = declared_mismatch
            if audit_hit is not None:
                origin, candidate = audit_hit
                outcome = _hash_candidate(candidate, expected, accepted_entry, role,
                                          hash_limit_bytes)
                if _is_match(outcome) or outcome.get("hash_verification") == "DEFERRED_BY_HASH_LIMIT":
                    chosen = _match(origin, candidate, outcome)
                    chosen["declared_mismatch"] = dict(declared_mismatch)
                    mismatch = None
        elif declared_match is not None:
            chosen = declared_match
        elif declared_deferred is not None:
            chosen = declared_deferred
        else:
            for origin, candidate in existing:
                if origin in ("package", "relocation_audit"):
                    continue
                outcome = _hash_candidate(candidate, expected, accepted_entry, role,
                                          hash_limit_bytes)
                if _is_match(outcome) or outcome.get("hash_verification") == "DEFERRED_BY_HASH_LIMIT":
                    chosen = _match(origin, candidate, outcome)
                    break
            if chosen is None and audit_hit is not None:
                origin, candidate = audit_hit
                outcome = _hash_candidate(candidate, expected, accepted_entry, role,
                                          hash_limit_bytes)
                if _is_match(outcome) or outcome.get("hash_verification") == "DEFERRED_BY_HASH_LIMIT":
                    chosen = _match(origin, candidate, outcome)

        if chosen is not None:
            if chosen.get("drift_accepted"):
                status = "DRIFT_ACCEPTED"
            elif chosen["origin"] == "relocation_audit":
                status = "RELOCATED"
            else:
                status = "AVAILABLE"
            item = {
                "asset_id": asset_id, "role": role, "status": status,
                "origin": chosen["origin"], "path": chosen["path"],
                "bytes": chosen["bytes"],
                "sha256_expected": expected,
                "sha256_actual": chosen.get("sha256_actual"),
                "hash_verified": bool(chosen["hash_verified"]),
                "hash_verification": chosen["hash_verification"],
                "accepted_drift": bool(chosen.get("drift_accepted")),
                "resolution": dict(record.get("resolution") or {}),
            }
            if chosen.get("declared_mismatch"):
                item["declared_mismatch"] = chosen["declared_mismatch"]
            if skipped:
                item["skipped_candidates"] = skipped
            items.append(item)
        elif mismatch is not None:
            items.append({
                "asset_id": asset_id, "role": role, "status": "REPLACED",
                "origin": mismatch["origin"], "path": mismatch["path"],
                "bytes": mismatch["bytes"], "sha256_expected": expected,
                "sha256_actual": mismatch.get("sha256_actual"), "hash_verified": True,
                "hash_verification": "FULL_FILE_SHA256_MISMATCH",
                "accepted_drift": False,
                "resolution": dict(record.get("resolution") or {}),
                "skipped_candidates": skipped,
            })
        else:
            items.append({
                "asset_id": asset_id, "role": role, "status": "MISSING",
                "origin": None, "path": None, "bytes": None,
                "sha256_expected": expected, "sha256_actual": None,
                "hash_verified": False, "hash_verification": "NOT_FOUND",
                "accepted_drift": False,
                "resolution": dict(record.get("resolution") or {}),
                "skipped_candidates": skipped,
            })
    return items


def _resolve_status(*, evidence: Mapping[str, Any], assets: Sequence[Mapping[str, Any]],
                    project: ProjectSnapshot) -> Tuple[str, Dict[str, Any]]:
    replaced = [item for item in assets if item["status"] == "REPLACED"]
    missing = [item for item in assets if item["status"] == "MISSING"]
    relocated = [item for item in assets if item["status"] == "RELOCATED"]
    accepted_drift = [item for item in assets if item["status"] == "DRIFT_ACCEPTED"]
    deferred = [item for item in assets
                if item["status"] not in ("MISSING", "DRIFT_ACCEPTED")
                and not item.get("hash_verified")]
    measurement = [item for item in assets if item.get("role") in MEASUREMENT_ROLES]
    measurement_matched = all(item["status"] in ("AVAILABLE", "RELOCATED")
                              and item.get("hash_verified") for item in measurement)
    recompute_available = bool(measurement) and measurement_matched
    binding_valid = evidence.get("binding") == "VALID"
    if replaced:
        status = STATUS_REPLACED
    elif not binding_valid:
        status = STATUS_NEEDS_RECOMPUTE if recompute_available else STATUS_RESTRICTED
    elif missing:
        status = STATUS_RESTRICTED
    elif relocated:
        status = STATUS_RELOCATED
    elif accepted_drift:
        status = STATUS_FULL_WITH_WARNINGS
    else:
        status = STATUS_FULL
    clean = status in (STATUS_FULL, STATUS_RELOCATED)
    evidence_current = (binding_valid and not replaced and not missing and not deferred
                        and measurement_matched
                        and status in (STATUS_FULL, STATUS_RELOCATED,
                                       STATUS_FULL_WITH_WARNINGS))
    facts = {
        "replaced_count": len(replaced),
        "missing_count": len(missing),
        "relocated_count": len(relocated),
        "accepted_drift_count": len(accepted_drift),
        "hash_deferred_count": len(deferred),
        "measurement_asset_count": len(measurement),
        "measurement_matched": bool(measurement_matched),
        "recompute_available": bool(recompute_available),
        "evidence_current": bool(evidence_current),
        "clean": bool(clean),
        "restricted": status not in (STATUS_FULL, STATUS_RELOCATED,
                                     STATUS_FULL_WITH_WARNINGS),
    }
    return status, facts

def _verification_allows_replaced_open(verification: Mapping[str, Any]) -> bool:
    """A coherent source-declaration mismatch is reported as REPLACED, not CORRUPT.

    Package tampering that happens to touch only assets/assets.json is still a
    source-declaration mismatch: the assets are untrustworthy and must not be
    opened as FULL.  Any other failed check keeps the package CORRUPT.
    """
    if verification.get("status") == STATUS_REPLACED:
        return True
    failed = set(verification.get("failed_checks") or [])
    if "partition_assets_snapshot" not in failed:
        return False
    if failed - {"partition_assets_snapshot", "partition_index"}:
        return False
    checks = {item.get("id"): item for item in verification.get("checks") or []}
    if not (checks.get("partition_assets") or {}).get("ok"):
        return False
    if "partition_index" in failed:
        detail = json.dumps((checks.get("partition_index") or {}).get("detail"),
                            ensure_ascii=False, allow_nan=False)
        if "assets/assets.json" not in detail and "assets\\\\assets.json" not in detail:
            return False
    return True


def open_package(package_path: os.PathLike[str] | str, *,
                 repo_root: Optional[os.PathLike[str] | str] = None,
                 source_roots: Sequence[os.PathLike[str] | str] = (),
                 hash_limit_bytes: int = 0,
                 skip_verify: bool = False,
                 use_relocation_audit: bool = True) -> Dict[str, Any]:
    """Reopen one package and report an explicit, non-aliased state."""
    package = Path(package_path)
    document: Dict[str, Any] = {
        "schema": OPEN_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "package": str(package),
        "status": None,
        "ok": False,
        "exit_code": EXIT_OK,
        "project": None,
        "evidence": None,
        "assets": None,
        "coordinates": None,
        "view": None,
        "clean": None,
        "source_integrity": None,
        "warnings": [],
        "reads_outside_package": [],
        "build_report_used": False,
        "not_claimed": [
            "Reopening is not cross-machine acceptance; a second machine is a separate measurement.",
            "No synthetic support points are ever substituted for a missing real source.",
        ],
    }
    if not skip_verify:
        verification = verify_package(package)
        document["verification"] = {
            "status": verification["status"],
            "checked_files": verification["counts"].get("checked_files"),
            "failed_checks": verification["failed_checks"],
        }
        if verification["status"] != "OK":
            if not _verification_allows_replaced_open(verification):
                document["status"] = STATUS_CORRUPT
                document["exit_code"] = EXIT_CORRUPT
                document["problems"] = verification["problems"]
                document["clean"] = False
                document["warnings"].append(
                    "package integrity verification failed; nothing was opened")
                return document
            document["warnings"].append(
                "package verification reports a source-declaration mismatch; continuing "
                "only to report REPLACED, never to open it as current")
    else:
        document["verification"] = {"status": "SKIPPED"}
    manifest = _load_manifest(package)
    project, _receipt = _load_bundle(package / "project")
    project_block = dict(manifest.get("project") or {})
    assets_document = _read_json_strict(package / "assets" / "assets.json")
    records = normalize_asset_document(assets_document)
    snapshot_mismatches = _snapshot_asset_mismatches(
        records, project, source_domain=_source_domain(project))
    if (not skip_verify and verification["status"] != "OK" and not snapshot_mismatches):
        document["status"] = STATUS_CORRUPT
        document["exit_code"] = EXIT_CORRUPT
        document["problems"] = verification["problems"]
        document["clean"] = False
        document["warnings"].append(
            "package integrity verification failed; nothing was opened")
        return document
    coordinates = _read_json_strict(package / "coordinates" / "coordinate_basis.json")
    evidence_block = dict(manifest.get("evidence") or {})
    sidecar_payload: Optional[Dict[str, Any]] = None
    if evidence_block.get("present"):
        sidecar_payload = _read_json_strict(package / "evidence" / "evidence_ledger.json")
    evidence = _evaluate_evidence(project, sidecar_payload)
    if evidence_block.get("binding") and evidence_block.get("binding") != evidence["binding"]:
        document["warnings"].append(
            "manifest recorded evidence binding %r but live evaluation gives %r"
            % (evidence_block.get("binding"), evidence["binding"]))
    audit = _relocation_audit(package, enabled=use_relocation_audit)
    manifest_source_integrity = dict(manifest.get("source_integrity") or {})
    accepted_drift = {str(entry.get("asset_id")): dict(entry)
                      for entry in manifest_source_integrity.get("accepted_drift") or []
                      if isinstance(entry, Mapping)}
    assets = _resolve_assets(records, package=package, repo_root=Path(repo_root) if repo_root else None,
                             source_roots=[Path(item) for item in source_roots],
                             audit=audit, accepted=accepted_drift,
                             hash_limit_bytes=int(hash_limit_bytes))
    status, facts = _resolve_status(evidence=evidence, assets=assets, project=project)
    if snapshot_mismatches:
        status = STATUS_REPLACED
        facts["clean"] = False
        facts["restricted"] = True
        facts["evidence_current"] = False
        facts["snapshot_mismatch_count"] = len(snapshot_mismatches)
    document["status"] = status
    document["ok"] = status not in (STATUS_REPLACED, STATUS_CORRUPT)
    document["exit_code"] = (EXIT_HASH_MISMATCH if status == STATUS_REPLACED
                             else EXIT_CORRUPT if status == STATUS_CORRUPT else EXIT_OK)
    document["clean"] = bool(facts["clean"])
    document["source_integrity"] = {
        "clean": (bool(manifest_source_integrity.get("clean", facts["clean"]))
                  and not snapshot_mismatches),
        "snapshot_clean": not snapshot_mismatches,
        "snapshot_mismatches": snapshot_mismatches,
        "policy": dict(manifest_source_integrity.get("policy") or {}),
        "accepted_drift": [dict(entry) for entry in accepted_drift.values()],
        "asset_count": len(assets),
    }
    document["package"] = str(package)
    document["manifest"] = {
        "package_id": manifest.get("package_id"),
        "distribution_mode": manifest.get("distribution_mode"),
        "source_domain": manifest.get("source_domain"),
        "created_at": manifest.get("created_at"),
        "manifest_body_sha256": manifest.get("manifest_body_sha256"),
    }
    document["project"] = {
        "project_id": project.project_id,
        "project_version": project.project_version,
        "content_hash": project.compute_content_hash(),
        "schema_version": project.schema_version,
        "object_counts": dict(project_block.get("object_counts") or {}),
        "geometry_digest": _geometry_digest(project),
        "manifest_geometry_digest": project_block.get("geometry_digest"),
        "geometry_matches_manifest": _geometry_digest(project) == project_block.get("geometry_digest"),
    }
    if not document["project"]["geometry_matches_manifest"]:
        document["warnings"].append("geometry digest differs from the manifest value")
    document["evidence"] = {
        "present": bool(evidence_block.get("present")),
        "binding": evidence["binding"],
        "reasons": list(evidence.get("reasons") or []),
        "snapshot_id": evidence.get("snapshot_id"),
        "record_count": int(evidence.get("record_count") or 0),
        "status_counts": dict(evidence.get("status_counts") or {}),
        "current": bool(facts["evidence_current"]),
        "origin": "PACKAGE_SIDECAR" if evidence_block.get("present") else "ABSENT",
        "recompute_available": bool(facts["recompute_available"]),
        "recompute_script": eledger.RECOMPUTE_SCRIPT,
    }
    document["assets"] = {
        "count": len(assets),
        "by_status": {name: sum(1 for item in assets if item["status"] == name)
                      for name in ("AVAILABLE", "RELOCATED", "DRIFT_ACCEPTED",
                                   "MISSING", "REPLACED")},
        "items": assets,
        "distribution_mode": manifest.get("distribution_mode"),
    }
    document["coordinates"] = {
        "display_convention": coordinates.get("display_convention"),
        "georeferencing": coordinates.get("georeferencing"),
        "observations": [{"observation_id": item.get("observation_id"),
                          "coordinate_frame": item.get("coordinate_frame"),
                          "unit": item.get("unit"),
                          "local_origin": item.get("local_origin")}
                         for item in coordinates.get("observations") or []],
    }
    document["view"] = {
        "geometry_viewable": True,
        "restricted": bool(facts["restricted"]),
        "synthetic_support_displayed": False,
        "measurement_conclusions_allowed": bool(facts["evidence_current"]),
    }
    document["facts"] = facts
    document["state_machine"] = {
        "states": list(OPEN_STATUSES),
        "applied_rule": _status_rule_name(status),
        "options": _status_options(status),
    }
    for item in assets:
        if item["status"] == "REPLACED":
            document["warnings"].append(
                "REPLACED asset %s: recorded sha256 %s, file on disk %s (%s); silent "
                "substitution is blocked" % (item["asset_id"], item["sha256_expected"],
                                             item["sha256_actual"], item["path"]))
        elif item["status"] == "MISSING":
            document["warnings"].append(
                "MISSING asset %s: REQUEST_RELOCATION; geometry stays viewable, evidence is "
                "not current, no synthetic support is shown" % item["asset_id"])
        elif item["status"] == "RELOCATED" and item.get("declared_mismatch"):
            drift = item["declared_mismatch"]
            document["warnings"].append(
                "RELOCATED asset %s: declared path %s held sha256 %s instead of %s; the "
                "equal-hash file was accepted through the explicit relocation audit (%s)"
                % (item["asset_id"], drift["path"], drift.get("sha256_actual"),
                   item["sha256_expected"], item["path"]))
        elif item["status"] == "DRIFT_ACCEPTED":
            document["warnings"].append(
                "SOURCE_DRIFT_ACCEPTED:%s: expected sha256 %s, file on disk %s (%s); "
                "source drift was accepted at export, clean=false"
                % (item["asset_id"], item["sha256_expected"], item["sha256_actual"],
                   item["path"]))
        elif not item.get("hash_verified"):
            document["warnings"].append(
                "asset %s was located but its hash was not re-checked (%s)"
                % (item["asset_id"], item.get("hash_verification")))
    for issue in snapshot_mismatches:
        document["warnings"].append(
            "PACKAGE_SNAPSHOT_ASSET_MISMATCH:%s: %s (path=%s, expected=%s, snapshot=%s); "
            "the package is REPLACED and must not be opened as current"
            % (issue.get("asset_id"), issue.get("reason"), issue.get("path"),
               issue.get("expected_sha256"), issue.get("declared_sha256")))
    if status == STATUS_NEEDS_RECOMPUTE:
        document["warnings"].append(
            "evidence binding is %s (%s); evidence is not current until recomputed"
            % (evidence["binding"], ", ".join(evidence.get("reasons") or []) or "no reason"))
    if status in (STATUS_RESTRICTED, STATUS_NEEDS_RECOMPUTE):
        if not evidence_block.get("present"):
            document["warnings"].append(
                "EVIDENCE_SIDECAR_ABSENT: no W153 evidence sidecar is present; evidence "
                "cannot be current for this package")
        if str(manifest.get("format_version") or "") == "1.0.0":
            document["warnings"].append(
                "LEGACY_FORMAT_1_0: the package predates evidence sidecar support; "
                "recompute or accept a restricted view")
        document["warnings"].append(
            "RESTRICTED_NO_SYNTHETIC_SUPPORT: geometry remains viewable, but no synthetic "
            "support points are displayed and evidence is not current")
    reads = sorted({item["path"] for item in assets if item.get("path")})
    document["reads_outside_package"] = reads
    document["measured_at"] = _now_iso()
    return document


def _status_rule_name(status: str) -> str:
    return {
        STATUS_FULL: "all assets hash-matched and evidence binding valid",
        STATUS_FULL_WITH_WARNINGS: "all measurement assets matched; declared auxiliary-source "
                                   "drift was accepted at export",
        STATUS_RESTRICTED: "source missing or evidence unbound with no recompute source",
        STATUS_REPLACED: "an asset hash or a package snapshot declaration differs from the recorded fingerprint",
        STATUS_RELOCATED: "an asset was found through the relocation audit with an equal hash",
        STATUS_NEEDS_RECOMPUTE: "evidence sidecar absent or binding mismatch; recompute available",
        STATUS_CORRUPT: "package integrity verification failed",
    }.get(status, "unknown")


def _status_options(status: str) -> List[str]:
    return {
        STATUS_FULL: [],
        STATUS_FULL_WITH_WARNINGS: ["SOURCE_DRIFT_ACCEPTED", "REQUEST_RELOCATION"],
        STATUS_RESTRICTED: ["REQUEST_RELOCATION", "SHOW_LIMITED_VIEW"],
        STATUS_REPLACED: ["WARN_AND_BLOCK_SILENT_SUBSTITUTION", "REQUEST_RELOCATION"],
        STATUS_RELOCATED: ["HASH_RELOCATE_OK"],
        STATUS_NEEDS_RECOMPUTE: ["RECOMPUTE", "RESTRICTED"],
        STATUS_CORRUPT: ["REJECT_PACKAGE"],
    }.get(status, [])


# -- relocation --------------------------------------------------------------

def relocate_asset(package_path: os.PathLike[str] | str, asset_id: str,
                   new_path: os.PathLike[str] | str, *, audit: bool = True) -> Dict[str, Any]:
    """Point one asset at a moved file whose whole-file SHA-256 is identical."""
    package = Path(package_path)
    verification = verify_package(package)
    if verification["status"] != "OK":
        raise PortableError("refusing to relocate inside a corrupt package",
                            "FILE_INDEX_MISMATCH",
                            details={"failed_checks": verification["failed_checks"]})
    document = _read_json_strict(package / "assets" / "assets.json")
    records = normalize_asset_document(document)
    record = next((item for item in records if str(item.get("asset_id")) == str(asset_id)), None)
    if record is None:
        raise PortableError("asset is not registered in this package: %r" % (asset_id,),
                            "ASSET_MISSING")
    source = Path(new_path)
    if source.is_symlink():
        raise PortableError("relocation source is a symlink: %s" % source, "SYMLINK_REJECTED")
    if not source.is_file():
        raise PortableError("relocation source is not a file: %s" % source,
                            "RELOCATE_SOURCE_MISSING")
    expected = str((record.get("fingerprint") or {}).get("value") or "")
    actual = file_sha256(source)
    if actual != expected:
        raise PortableError(
            "relocation refused: whole-file sha256 differs from the recorded fingerprint",
            "RELOCATE_HASH_MISMATCH",
            details={"asset_id": asset_id, "path": str(source), "expected": expected,
                     "actual": actual, "action": "WARN_AND_BLOCK_SILENT_SUBSTITUTION"})
    entry = {
        "schema": AUDIT_SCHEMA,
        "measured_at": _now_iso(),
        "asset_id": asset_id,
        "role": record.get("role"),
        "path": str(source.resolve()),
        "sha256": actual,
        "expected_sha256": expected,
        "bytes": int(source.stat().st_size),
        "action": "HASH_RELOCATE_OK",
        "package_id": _load_manifest(package).get("package_id"),
    }
    audit_relative = "state/relocation_audit.jsonl"
    if audit:
        audit_path = package / "state" / "relocation_audit.jsonl"
        try:
            _mkdir_checked(audit_path.parent, parents=True, exist_ok=True,
                           action="create package state directory")
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + chr(10))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise _write_error(exc, target=str(audit_path),
                               action="write relocation audit") from exc
    return {
        "schema": RELOCATE_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "status": "RELOCATED",
        "ok": True,
        "exit_code": EXIT_OK,
        "asset_id": asset_id,
        "path": str(source.resolve()),
        "sha256": actual,
        "hash_relocate": "HASH_RELOCATE_OK",
        "audit": {"written": bool(audit), "path": audit_relative},
        "not_claimed": ["Relocation proves byte identity, not that the file is the right "
                        "measurement for any other purpose."],
    }


# -- delivery export ---------------------------------------------------------

def export_delivery(package_path: os.PathLike[str] | str,
                    out_path: os.PathLike[str] | str, *,
                    fault: Optional[str] = None) -> Dict[str, Any]:
    """Export a delivery folder (CSV + geometry JSON) from a package alone."""
    import csv as _csv
    import io as _io

    fault = _fault_mode(fault)
    package = Path(package_path)
    verification = verify_package(package)
    if verification["status"] != "OK":
        raise PortableError("refusing to export a delivery from a corrupt package",
                            "FILE_INDEX_MISMATCH",
                            details={"failed_checks": verification["failed_checks"]})
    manifest = _load_manifest(package)
    project, _receipt = _load_bundle(package / "project")
    evidence_block = dict(manifest.get("evidence") or {})
    sidecar_payload = None
    if evidence_block.get("present"):
        sidecar_payload = _read_json_strict(package / "evidence" / "evidence_ledger.json")
    evidence = _evaluate_evidence(project, sidecar_payload)
    records: Dict[str, Dict[str, Any]] = {}
    if sidecar_payload is not None:
        try:
            records = eledger.records_from_payload(sidecar_payload)
        except Exception:  # noqa: BLE001 - an unreadable sidecar yields unavailable rows
            records = {}
    binding_valid = evidence["binding"] == "VALID"
    source_domain = str(manifest.get("source_domain") or "unknown")
    coordinates = _read_json_strict(package / "coordinates" / "coordinate_basis.json")
    out = Path(out_path)
    if out.exists():
        raise PortableError("delivery target already exists: %s" % out, "PACKAGE_EXISTS")
    parent = out.parent
    if not parent.is_dir():
        _mkdir_checked(parent, parents=True, exist_ok=True,
                       action="create parent directory")
    staging = parent / (".delivery_staging_%s_%s" % (out.name, uuid.uuid4().hex[:10]))
    _mkdir_checked(staging, parents=False, exist_ok=False,
                   action="create staging directory")
    published = False
    try:
        project_document = project.to_dict()
        surface_geometry = dict((project.artifacts or {}).get("surface_geometry") or {})
        fragments = {item.fragment_id: item for item in project.fragments}
        geometry_rows = []
        for surface in project.surfaces:
            geometry_rows.append({
                "surface_id": surface.surface_id,
                "member_fragment_ids": list(surface.member_fragment_ids),
                "finite_support": [item.to_dict() for item in surface.finite_support],
                "member_planes_local": {member: fragments[member].plane_local.to_dict()
                                        for member in surface.member_fragment_ids
                                        if member in fragments},
                "provenance": surface.provenance,
                "interpretation_status": surface.interpretation_status,
                "geometry_mode": (surface_geometry.get(surface.surface_id) or {}).get("mode"),
                "geometry_frame_local": (surface_geometry.get(surface.surface_id) or {}).get("frame_local"),
                "unit_status": ";".join(item.unit for item in project.observations),
                "source_domain": source_domain,
                "evidence_status": ("current" if binding_valid else "unavailable"),
            })
        fragment_rows = []
        for fragment in project.fragments:
            quality = dict(fragment.quality or {})
            fragment_rows.append({
                "fragment_id": fragment.fragment_id,
                "observation_id": fragment.observation_id,
                "support_ref": fragment.support_ref.to_dict(),
                "plane_local": fragment.plane_local.to_dict(),
                "finite_regions_local": [item.to_dict() for item in fragment.finite_regions_local],
                "quality": {key: quality.get(key) for key in
                            ("rmse", "max_abs_residual", "plane_tolerance", "region_areas",
                             "hole_count", "geometry_source", "fit_method", "fit_config_hash",
                             "requires_human_review", "frame_local")},
                "version": fragment.version,
            })
        relation_rows = []
        for relation in project.relations:
            record = records.get(relation.relation_id) or {}
            row = relation.to_dict()
            row["evidence_status"] = (str(record.get("status"))
                                      if binding_valid and record else "unavailable")
            row["evidence_state"] = record.get("state") if binding_valid else None
            row["evidence_snapshot_id"] = evidence.get("snapshot_id") if binding_valid else None
            relation_rows.append(row)
        files: Dict[str, Dict[str, Any]] = {}

        def write(relative: str, result: Mapping[str, Any]) -> None:
            files[relative] = {"sha256": str(result["sha256"]), "bytes": int(result["bytes"])}

        write("project.json", _write_json(staging / "project.json", project_document))
        write("geometry.json", _write_json(staging / "geometry.json", {
            "schema": "fractureflow.surface_product.delivery_geometry.v1",
            "project_id": project.project_id,
            "project_version": project.project_version,
            "content_hash": project.compute_content_hash(),
            "source_domain": source_domain,
            "unit_declarations": [item.unit for item in project.observations],
            "coordinate_frames": [item.coordinate_frame for item in project.observations],
            "display_convention": coordinates.get("display_convention"),
            "georeferencing": coordinates.get("georeferencing"),
            "surface_geometry": copy.deepcopy(surface_geometry),
            "surfaces": geometry_rows,
            "fragments": fragment_rows,
            "evidence_status": ("current" if binding_valid else "unavailable"),
            "evidence_binding": evidence["binding"],
            "measurement_policy": "FINITE_LOCAL_GEOMETRY_ONLY_NO_METRIC_OR_GEOGRAPHIC_CONCLUSION",
            "support_policy": "SUPPORT_INDICES_REFER_TO_THE_ORIGINAL_OBSERVATION_POINTS",
        }))
        write("relations.json", _write_json(staging / "relations.json", {
            "schema": "fractureflow.surface_product.delivery_relations.v1",
            "project_id": project.project_id,
            "project_version": project.project_version,
            "content_hash": project.compute_content_hash(),
            "source_domain": source_domain,
            "evidence_binding": evidence["binding"],
            "evidence_snapshot_id": evidence.get("snapshot_id"),
            "evidence_current": binding_valid,
            "unknown_relation_count": sum(item.state == "unknown" for item in project.relations),
            "relations": relation_rows,
        }))
        if sidecar_payload is not None:
            write("evidence_ledger.json",
                  _write_json(staging / "evidence_ledger.json", sidecar_payload))
        else:
            write("evidence_ledger.json", _write_json(staging / "evidence_ledger.json", {
                "schema": "fractureflow.surface_product.delivery_evidence.v1",
                "binding": evidence["binding"],
                "reasons": list(evidence.get("reasons") or []),
                "current": False,
            }))
        rows = []
        for surface in project.surfaces:
            rows.append([
                project.project_id, project.project_version, surface.surface_id,
                surface.version, json.dumps(list(surface.member_fragment_ids), allow_nan=False),
                surface.interpretation_status, surface.provenance,
                len(surface.finite_support),
                sum(len(item.holes) for item in surface.finite_support),
                ";".join(item.unit for item in project.observations),
                source_domain,
                "current" if binding_valid else "unavailable",
            ])
        buffer = _io.StringIO()
        writer = _csv.writer(buffer, lineterminator=chr(10))
        writer.writerow(["project_id", "project_version", "surface_id", "surface_version",
                         "member_fragment_ids", "interpretation_status", "provenance",
                         "finite_region_count", "hole_count", "unit_status", "source_domain",
                         "evidence_status"])
        writer.writerows(rows)
        csv_text = chr(65279) + buffer.getvalue()
        write("surfaces.csv", _write_bytes(staging / "surfaces.csv", csv_text.encode("utf-8")))
        delivery_manifest = {
            "schema": DELIVERY_SCHEMA,
            "format_version": FORMAT_VERSION,
            "measured_at": _now_iso(),
            "package_id": manifest.get("package_id"),
            "project": dict(manifest.get("project") or {}),
            "source_domain": source_domain,
            "distribution_mode": manifest.get("distribution_mode"),
            "unit_declarations": [item.unit for item in project.observations],
            "coordinate_frames": [item.coordinate_frame for item in project.observations],
            "evidence": {
                "binding": evidence["binding"],
                "snapshot_id": evidence.get("snapshot_id"),
                "current": binding_valid,
                "record_count": int(evidence.get("record_count") or 0),
                "status_counts": dict(evidence.get("status_counts") or {}),
            },
            "object_counts": {
                "surfaces": len(project.surfaces),
                "fragments": len(project.fragments),
                "relations": len(project.relations),
            },
            "files": files,
            "file_count": len(files),
            "not_claimed": [
                "A delivery export is not a quality certificate and carries no metric or "
                "geographic conclusion.",
                "evidence_status is current only when the package sidecar binding verifies.",
            ],
        }
        write("manifest.json", _write_json(staging / "manifest.json", delivery_manifest))
        if fault == "after_staging":
            raise PortableError("test fault injected after the delivery staging build",
                                "TEST_FAULT_AFTER_STAGING")
        # Read-back through a fresh parse, never from the objects in memory.
        readback: Dict[str, Any] = {"checked": 0, "csv_rows": 0}
        for relative, entry in sorted(files.items()):
            target = staging / relative
            if file_sha256(target) != entry["sha256"]:
                raise PortableError("delivery read-back hash mismatch: %s" % relative,
                                    "FILE_INDEX_MISMATCH")
            if relative.endswith(".json"):
                _read_json_strict(target)
            readback["checked"] += 1
        csv_lines = (staging / "surfaces.csv").read_text(encoding="utf-8-sig").splitlines()
        readback["csv_rows"] = max(0, len(csv_lines) - 1)
        if readback["csv_rows"] != len(project.surfaces):
            raise PortableError("delivery CSV row count %d differs from %d surfaces"
                                % (readback["csv_rows"], len(project.surfaces)),
                                "FILE_INDEX_MISMATCH")
        if readback["csv_rows"] <= 0:
            raise PortableError("delivery CSV has no data rows", "EMPTY_ENUMERATION")
        if fault == "before_publish":
            raise PortableError("test fault injected before the delivery publish",
                                "TEST_FAULT_BEFORE_PUBLISH")
        if out.exists():
            raise PortableError("delivery target appeared during export: %s" % out,
                                "PACKAGE_EXISTS")
        _rename_checked(staging, out, action="publish delivery")
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)
    return {
        "schema": DELIVERY_SCHEMA,
        "format_version": FORMAT_VERSION,
        "measured_at": _now_iso(),
        "status": "EXPORTED",
        "ok": True,
        "exit_code": EXIT_OK,
        "delivery": str(out),
        "file_count": len(files),
        "files": files,
        "readback": readback,
        "source_domain": source_domain,
        "evidence": {"binding": evidence["binding"], "current": binding_valid,
                     "snapshot_id": evidence.get("snapshot_id")},
        "not_claimed": ["A delivery export is not a quality certificate."],
    }


# -- CLI ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractureflow.surface_product.portable_project",
        description="W154 portable project package (format %s)" % FORMAT_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="build and publish a portable package")
    export.add_argument("--project", required=True, help="project bundle (.ffsrf) path")
    export.add_argument("--out", required=True, help="package path ending in .ffportable")
    export.add_argument("--mode", default=REFERENCE_ONLY, choices=list(DISTRIBUTION_MODES))
    export.add_argument("--assets-json", default=None, help="explicit W152 asset document")
    export.add_argument("--license-evidence", default="", help="licence evidence pointer")
    export.add_argument("--contract-sha256", default=None,
                        help="sha256 of the W152 asset contract used for this package")
    export.add_argument("--asset-root", action="append", default=[],
                        help="root used to locate declared assets (repeatable)")
    export.add_argument("--verify-asset-hashes", dest="verify_asset_hashes",
                        action="store_true", default=True)
    export.add_argument("--no-verify-asset-hashes", dest="verify_asset_hashes",
                        action="store_false")
    export.add_argument("--asset-hash-limit-bytes", type=int, default=512 * 1024 * 1024)
    export.add_argument("--accept-source-drift", action="append", default=[],
                        help="asset_id whose auxiliary source drift is explicitly accepted "
                             "(repeatable; requires --drift-reason)")
    export.add_argument("--drift-reason", default="",
                        help="audit reason recorded with --accept-source-drift")

    open_cmd = sub.add_parser("open", help="reopen a portable package")
    open_cmd.add_argument("--package", required=True)
    open_cmd.add_argument("--repo-root", default=None)
    open_cmd.add_argument("--source-root", action="append", default=[])
    open_cmd.add_argument("--hash-limit-bytes", type=int, default=0)
    open_cmd.add_argument("--no-verify", dest="skip_verify", action="store_true")

    verify = sub.add_parser("verify", help="verify a portable package")
    verify.add_argument("--package", required=True)
    verify.add_argument("--allow-unpublished", dest="require_published",
                        action="store_false", default=True)

    relocate = sub.add_parser("relocate", help="point one asset at a moved identical file")
    relocate.add_argument("--package", required=True)
    relocate.add_argument("--asset-id", required=True, dest="asset_id")
    relocate.add_argument("--path", required=True)
    relocate.add_argument("--no-audit", dest="audit", action="store_false", default=True)

    delivery = sub.add_parser("export-delivery", help="export CSV + geometry from a package")
    delivery.add_argument("--package", required=True)
    delivery.add_argument("--out", required=True)
    return parser


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (TypeError, ValueError, OSError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    document: Dict[str, Any]
    try:
        if args.command == "export":
            assets_document = None
            if args.assets_json:
                assets_document = _read_json_strict(Path(args.assets_json))
            document = export_portable(
                args.out, args.project, distribution_mode=args.mode,
                assets_document=assets_document,
                license_evidence=str(args.license_evidence or ""),
                contract_sha256=args.contract_sha256,
                asset_roots=[Path(item) for item in args.asset_root],
                verify_asset_hashes=bool(args.verify_asset_hashes),
                asset_hash_limit_bytes=int(args.asset_hash_limit_bytes),
                accept_source_drift=list(args.accept_source_drift),
                drift_reason=str(args.drift_reason))
            code = EXIT_OK
        elif args.command == "open":
            document = open_package(
                args.package, repo_root=args.repo_root, source_roots=args.source_root,
                hash_limit_bytes=int(args.hash_limit_bytes),
                skip_verify=bool(args.skip_verify))
            code = int(document["exit_code"])
        elif args.command == "verify":
            document = verify_package(args.package,
                                      require_published=bool(args.require_published))
            code = int(document["exit_code"])
        elif args.command == "relocate":
            document = relocate_asset(args.package, args.asset_id, args.path,
                                      audit=bool(args.audit))
            code = EXIT_OK
        elif args.command == "export-delivery":
            document = export_delivery(args.package, args.out)
            code = EXIT_OK
        else:  # pragma: no cover - argparse enforces the choices
            raise PortableError("unknown command %r" % args.command, "USAGE_ERROR")
    except PortableError as exc:
        document = exc.to_dict()
        code = int(exc.exit_code)
    sys.stdout.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT_VERSION", "PORTABLE_SCHEMA", "MANIFEST_NAME", "PUBLICATION_MARKER",
    "REFERENCE_ONLY", "SELF_CONTAINED", "DISTRIBUTION_MODES", "OPEN_STATUSES",
    "STATUS_FULL", "STATUS_FULL_WITH_WARNINGS", "STATUS_RESTRICTED", "STATUS_REPLACED",
    "STATUS_RELOCATED",
    "STATUS_NEEDS_RECOMPUTE", "STATUS_CORRUPT", "PortableError",
    "export_portable", "open_package", "verify_package", "relocate_asset",
    "export_delivery", "derive_assets_from_project", "normalize_asset_document",
    "validate_asset_record", "build_coordinate_basis", "build_licenses", "main",
]
