# -*- coding: utf-8 -*-
"""W152 generic dataset entrance: disk registry, safe path resolution, strict input.

One new legitimate dataset is admitted by editing one JSON registry file and
running one command -- never by editing Python source.  Every path that comes
from a document is treated as hostile until it is proven relative and confined:

* absolute paths, drive letters, UNC prefixes, ``~`` and ``..`` are rejected;
* resolved targets must stay inside the declared root (symlinks included);
* JSON is parsed with ``parse_constant`` rejecting ``NaN`` / ``Infinity``;
* byte ceilings fire before any large read;
* archive members are listed and rejected before extraction.

These are gates, not conventions: each raises ``DatasetError`` with a stable
machine-readable code that the negative-control suite asserts against.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .contracts import ContractError

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_SCHEMA = "fractureflow.dataset_registry.v1"
REGISTRY_SCHEMA_V2 = "fractureflow.dataset_registry.v2"
ENTRY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{1,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_REGISTRY_BYTES = 512 * 1024
MAX_OBSERVATION_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ASSET_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2048
MAX_ARCHIVE_UNCOMPRESSED = 512 * 1024 * 1024
UNKNOWN_UNIT_TOKENS = ("UNKNOWN", "UNVERIFIED", "UNSPECIFIED")
DOMAINS = ("synthetic", "internal_real")
ROLES = ("RAW_MEASUREMENT", "DERIVED_DISPLAY", "AUXILIARY_CALIBRATION")
# Additive v2 vocabulary: a full-resolution spatial subset of a raw measurement
# cloud with a source-ordinal map.  It is measurement support, not a display
# decimation, so it must not be filed as DERIVED_DISPLAY.
ROLES_V2 = ROLES + ("DERIVED_ROI_MEASUREMENT",)


class DatasetError(ContractError):
    """A dataset entrance boundary violation with a stable code."""

    def __init__(self, message: str, code: str = "DATASET_INVALID") -> None:
        super().__init__(message, code=code)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise DatasetError(message, code=code)


def is_safe_relative(value: Any) -> bool:
    """True only for a plain relative path with no escape component."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\x00" in text:
        return False
    text = text.replace("\\", "/")
    if text.startswith("/") or text.startswith("~"):
        return False
    if text.startswith("//"):
        return False
    if re.match(r"^[A-Za-z]:", text):
        return False
    return ".." not in text.split("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_strict(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Read one JSON document, refusing oversized files and non-finite numbers."""
    target = Path(path)
    _require(target.is_file(), "MISSING_ASSET", "JSON document is missing: %s" % target)
    size = target.stat().st_size
    _require(size <= max_bytes, "RESOURCE_LIMIT_EXCEEDED",
             "JSON document exceeds %d bytes: %s" % (max_bytes, target))
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError("JSON document is not UTF-8: %s" % exc, code="ENCODING_INVALID") from exc

    def _reject(token: str) -> Any:
        raise DatasetError("non-finite number token %r is not allowed" % token,
                           code="NON_FINITE_NUMBER")

    try:
        return json.loads(text, parse_constant=_reject)
    except json.JSONDecodeError as exc:
        raise DatasetError("JSON document does not parse: %s" % exc, code="JSON_INVALID") from exc


def resolve_relative(root: Path, relative: Any, *, must_exist: bool = True,
                     kind: Optional[str] = None) -> Path:
    """Resolve ``relative`` under ``root`` and prove the result never escapes."""
    _require(is_safe_relative(relative), "UNSAFE_PATH",
             "path must be relative and free of '..': %r" % (relative,))
    root_resolved = Path(root).resolve()
    target = (root_resolved / str(relative)).resolve()
    _require(target == root_resolved or root_resolved in target.parents, "PATH_ESCAPE",
             "resolved path escapes the declared root: %s" % target)
    if must_exist:
        _require(target.exists(), "MISSING_ASSET", "declared asset is absent: %s" % target)
        if kind == "file":
            _require(target.is_file(), "ASSET_KIND_MISMATCH", "declared asset is not a file: %s" % target)
        if kind == "dir":
            _require(target.is_dir(), "ASSET_KIND_MISMATCH", "declared asset is not a directory: %s" % target)
    return target


def repo_relative(path: Path) -> str:
    """Return a POSIX repo-relative path, or fail: the service only reads inside the repo."""
    target = Path(path).resolve()
    root = REPO_ROOT.resolve()
    _require(target == root or root in target.parents, "PATH_ESCAPE",
             "asset lies outside the repository: %s" % target)
    return target.relative_to(root).as_posix()


def check_asset_size(path: Path, *, max_bytes: int = DEFAULT_MAX_ASSET_BYTES) -> int:
    size = Path(path).stat().st_size
    _require(size <= max_bytes, "RESOURCE_LIMIT_EXCEEDED",
             "asset exceeds %d bytes: %s" % (max_bytes, path))
    _require(size > 0, "EMPTY_INPUT", "asset is empty: %s" % path)
    return size


def safe_archive_members(archive_path: Path, *, max_members: int = MAX_ARCHIVE_MEMBERS,
                         max_uncompressed_bytes: int = MAX_ARCHIVE_UNCOMPRESSED) -> List[str]:
    """List archive members, refusing traversal, absolute names and symlinks."""
    target = Path(archive_path)
    _require(target.is_file(), "MISSING_ASSET", "archive is missing: %s" % target)
    names: List[str] = []
    try:
        with zipfile.ZipFile(target) as bundle:
            members = bundle.infolist()
            _require(len(members) <= max_members, "RESOURCE_LIMIT_EXCEEDED",
                     "archive declares more than %d members" % max_members)
            total = 0
            for info in members:
                name = info.filename
                _require(is_safe_relative(name), "ARCHIVE_TRAVERSAL",
                         "archive member escapes the extraction root: %r" % name)
                mode = (info.external_attr >> 16) & 0o170000
                _require(mode != 0o120000, "ARCHIVE_SYMLINK",
                         "archive member is a symlink: %r" % name)
                total += int(info.file_size)
                _require(total <= max_uncompressed_bytes, "RESOURCE_LIMIT_EXCEEDED",
                         "archive uncompressed total exceeds %d bytes" % max_uncompressed_bytes)
                names.append(name)
    except zipfile.BadZipFile as exc:
        raise DatasetError("archive does not parse: %s" % exc, code="BAD_ARCHIVE") from exc
    _require(bool(names), "EMPTY_INPUT", "archive has no members: %s" % target)
    return names


def load_registry(path: Path) -> Dict[str, Any]:
    """Load and validate the dataset registry; entries stay displayable when absent."""
    document = read_json_strict(Path(path), max_bytes=MAX_REGISTRY_BYTES)
    _require(isinstance(document, Mapping), "REGISTRY_INVALID", "registry must be one JSON object")
    schema = document.get("schema")
    _require(schema in (REGISTRY_SCHEMA, REGISTRY_SCHEMA_V2), "REGISTRY_INVALID",
             "registry schema must be %s or %s" % (REGISTRY_SCHEMA, REGISTRY_SCHEMA_V2))
    roles = ROLES if schema == REGISTRY_SCHEMA else ROLES_V2
    declared_roots = document.get("allowed_roots")
    _require(isinstance(declared_roots, list) and bool(declared_roots), "REGISTRY_INVALID",
             "registry must declare at least one allowed root")
    roots: List[Path] = []
    for item in declared_roots:
        _require(is_safe_relative(item), "UNSAFE_PATH",
                 "allowed root must be repo-relative: %r" % (item,))
        roots.append(resolve_relative(REPO_ROOT, item, must_exist=False))
    entries = document.get("entries")
    _require(isinstance(entries, list), "REGISTRY_INVALID", "registry entries must be a list")

    seen: set[str] = set()
    resolved: List[Dict[str, Any]] = []
    for raw in entries:
        _require(isinstance(raw, Mapping), "REGISTRY_INVALID", "registry entry must be an object")
        entry = dict(raw)
        identifier = str(entry.get("id") or "")
        _require(bool(ENTRY_ID_PATTERN.match(identifier)), "REGISTRY_INVALID",
                 "entry id must match %s" % ENTRY_ID_PATTERN.pattern)
        _require(identifier not in seen, "REGISTRY_INVALID", "duplicate entry id: %s" % identifier)
        seen.add(identifier)
        _require(str(entry.get("label") or "").strip() != "", "REGISTRY_INVALID",
                 "entry %s has no label" % identifier)
        _require(entry.get("domain") in DOMAINS, "REGISTRY_INVALID",
                 "entry %s domain must be one of %s" % (identifier, DOMAINS))
        _require(entry.get("role") in roles, "REGISTRY_INVALID",
                 "entry %s role must be one of %s" % (identifier, roles))
        for key in ("unit_status", "license_status"):
            _require(str(entry.get(key) or "").strip() != "", "REGISTRY_INVALID",
                     "entry %s has no %s" % (identifier, key))
        build_dir = entry.get("build_dir")
        _require(is_safe_relative(build_dir), "UNSAFE_PATH",
                 "entry %s build_dir must be relative" % identifier)
        target = None
        for root in roots:
            candidate = resolve_relative(root, build_dir, must_exist=False)
            if candidate == root or root in candidate.parents:
                target = candidate
                break
        _require(target is not None, "PATH_ESCAPE",
                 "entry %s build_dir escapes every allowed root" % identifier)
        entry["_build_path"] = str(target)
        entry["_build_exists"] = target.is_dir()
        point_cloud = entry.get("point_cloud")
        if point_cloud is not None:
            _require(is_safe_relative(point_cloud), "UNSAFE_PATH",
                     "entry %s point_cloud must be repo-relative" % identifier)
            entry["_point_cloud_path"] = str(resolve_relative(REPO_ROOT, point_cloud, must_exist=False))
        resolved.append(entry)
    return {
        "schema": schema,
        "path": str(Path(path)),
        "allowed_roots": [str(item) for item in roots],
        "entries": resolved,
    }


def _finite_bounds(value: Any, label: str) -> List[float]:
    _require(isinstance(value, list) and len(value) == 3, "OBSERVATION_INCOMPLETE",
             "%s must be three numbers" % label)
    numbers: List[float] = []
    for item in value:
        _require(isinstance(item, (int, float)) and not isinstance(item, bool), "OBSERVATION_INCOMPLETE",
                 "%s must be numeric" % label)
        number = float(item)
        _require(number == number and number not in (float("inf"), float("-inf")),
                 "NON_FINITE_NUMBER", "%s carries a non-finite number" % label)
        numbers.append(number)
    return numbers


def parse_observation_dataset(dataset_dir: Path, *, allow_root: Optional[Path] = None,
                              max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES) -> Dict[str, Any]:
    """Validate one dataset directory and return its resolved assets.

    The dataset is admitted only if its point cloud is inside the repository
    (the workbench re-reads it later), matches its declared SHA256, and carries
    finite ROI bounds plus an explicit unit/licence statement.
    """
    directory = Path(dataset_dir).resolve()
    if allow_root is not None:
        root = Path(allow_root).resolve()
        _require(directory == root or root in directory.parents, "PATH_ESCAPE",
                 "dataset directory escapes the declared root: %s" % directory)
    _require(directory.is_dir(), "MISSING_ASSET", "dataset directory is absent: %s" % directory)
    observation_path = directory / "observation.json"
    document = read_json_strict(observation_path, max_bytes=MAX_OBSERVATION_BYTES)
    _require(isinstance(document, Mapping), "OBSERVATION_INCOMPLETE",
             "observation.json must be one JSON object")

    point_cloud = document.get("point_cloud")
    _require(isinstance(point_cloud, Mapping), "OBSERVATION_INCOMPLETE",
             "observation.json has no point_cloud block")
    declared_path = point_cloud.get("path")
    _require(is_safe_relative(declared_path), "UNSAFE_PATH",
             "point_cloud.path must be relative: %r" % (declared_path,))
    candidates = [directory / str(declared_path), REPO_ROOT / str(declared_path)]
    ply_path = next((item.resolve() for item in candidates if item.is_file()), None)
    _require(ply_path is not None, "MISSING_ASSET",
             "point cloud is absent under the dataset and the repo root: %s" % declared_path)
    repo_relative(ply_path)  # confines the asset to the repository
    check_asset_size(ply_path, max_bytes=max_asset_bytes)

    declared_sha = str(point_cloud.get("sha256") or "").lower()
    _require(bool(SHA256_PATTERN.match(declared_sha)), "ASSET_FINGERPRINT_INVALID",
             "point_cloud.sha256 must be 64 hex characters")
    actual_sha = sha256_file(ply_path)
    _require(actual_sha == declared_sha, "ASSET_HASH_MISMATCH",
             "point cloud SHA256 differs from observation.json")

    point_total = point_cloud.get("point_total")
    _require(isinstance(point_total, int) and not isinstance(point_total, bool) and point_total > 0,
             "OBSERVATION_INCOMPLETE", "point_cloud.point_total must be a positive integer")

    roi = document.get("roi")
    _require(isinstance(roi, Mapping), "OBSERVATION_INCOMPLETE", "observation.json has no roi block")
    bounds = roi.get("bounds")
    _require(isinstance(bounds, Mapping), "OBSERVATION_INCOMPLETE", "roi.bounds is missing")
    bounds_min = _finite_bounds(bounds.get("min"), "roi.bounds.min")
    bounds_max = _finite_bounds(bounds.get("max"), "roi.bounds.max")
    _require(all(low <= high for low, high in zip(bounds_min, bounds_max)), "OBSERVATION_INCOMPLETE",
             "roi.bounds.min exceeds roi.bounds.max")

    coordinate = document.get("coordinate") or {}
    _require(isinstance(coordinate, Mapping), "OBSERVATION_INCOMPLETE", "coordinate must be an object")
    unit_status = str(coordinate.get("unit_status") or "UNKNOWN_SOURCE_COORDINATE_UNIT")
    unit = str(coordinate.get("point_cloud_unit") or "source_coordinate_units_unverified")
    upper = unit_status.upper()
    unit_known = not any(token in upper for token in UNKNOWN_UNIT_TOKENS)

    license_doc = document.get("license") or {}
    _require(isinstance(license_doc, Mapping), "OBSERVATION_INCOMPLETE", "license must be an object")
    license_status = str(license_doc.get("status") or "UNVERIFIED")

    scene_id = str(document.get("scene_id") or directory.name)
    return {
        "dataset_dir": str(directory),
        "observation_path": str(observation_path),
        "observation_sha256": sha256_file(observation_path),
        "observation_bytes": int(observation_path.stat().st_size),
        "document": document,
        "scene_id": scene_id,
        "point_cloud_path": str(ply_path),
        "point_cloud_repo_path": repo_relative(ply_path),
        "point_cloud_sha256": actual_sha,
        "point_cloud_bytes": int(ply_path.stat().st_size),
        "point_total": int(point_total),
        "roi_bounds_min": bounds_min,
        "roi_bounds_max": bounds_max,
        "unit": unit,
        "unit_status": unit_status,
        "unit_known": bool(unit_known),
        "coordinate_frame": str(coordinate.get("coordinate_frame") or (scene_id + "_source_coordinates")),
        "license_status": license_status,
        "license_spdx": str(license_doc.get("spdx_id") or "UNVERIFIED"),
        "redistribution_allowed": bool(license_doc.get("redistribution_allowed") is True),
    }


def iter_registry_ids(registry: Mapping[str, Any]) -> Iterable[str]:
    for entry in registry.get("entries") or []:
        yield str(entry.get("id"))
