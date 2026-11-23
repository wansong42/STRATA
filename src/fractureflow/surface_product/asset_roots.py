# -*- coding: utf-8 -*-
"""W160 controlled asset root allowlist.

The 256 MiB small-asset ceiling is a gate, not a streaming capability: the
complete 1.63 GB cloud enters only through the bounded-memory ROI reader.  Any
future external asset (for example E:/fracture_data) must be named by one
controlled registry document -- never by an HTTP caller and never by a request
field.  Roots are resolved and then proven confined:

* relative references only; absolute paths, drive letters, UNC prefixes, '~',
  NUL and '..' are refused;
* the resolved target must stay inside the declared root (case-insensitive on
  Windows, via normcase + commonpath);
* every component between the root and the target is checked with lstat for a
  reparse point (junction / symlink); a reparse component is an escape until
  proven otherwise, and this module refuses it;
* a root that is itself a reparse point is refused as an external root.

Negative controls live in the W160 acceptance script; every refusal carries a
stable machine code.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWLIST_SCHEMA = "fractureflow.asset_root_allowlist.v1"
MAX_ALLOWLIST_BYTES = 256 * 1024
ROOT_KINDS = ("repo", "external")
_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,63}$"


class AssetRootError(ValueError):
    """A root-allowlist boundary violation with a stable code."""

    def __init__(self, message: str, code: str = "ROOT_POLICY_INVALID") -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AssetRootError(message, code=code)


def is_safe_relative(value: Any) -> bool:
    """True only for a plain relative path with no escape component."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\x00" in text:
        return False
    text = text.replace("\\", "/")
    if text.startswith("/") or text.startswith("~") or text.startswith("//"):
        return False
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return False
    return ".." not in text.split("/")


def is_reparse_point(path: Any) -> bool:
    """True for a symlink, junction or any other reparse point."""
    target = Path(path)
    try:
        info = os.lstat(target)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _canonical(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _within(root: Path, target: Path) -> bool:
    root_text = _norm(root)
    target_text = _norm(target)
    if target_text == root_text:
        return True
    try:
        return os.path.commonpath([root_text, target_text]) == root_text
    except ValueError:
        return False


def _reparse_components(root: Path, target: Path) -> List[str]:
    """Return relative components of target (root excluded) that are reparse points."""
    try:
        relative = target.relative_to(root)
    except ValueError:
        return ["<outside-root>"]
    found: List[str] = []
    current = root
    for part in relative.parts:
        current = current / part
        if is_reparse_point(current):
            found.append(str(current))
    return found


def load_root_allowlist(config_path: Any) -> Dict[str, Any]:
    """Load and validate one controlled root allowlist document."""
    target = Path(config_path)
    _require(target.is_file(), "ALLOWLIST_MISSING", "root allowlist is missing: %s" % target)
    size = int(target.stat().st_size)
    _require(size <= MAX_ALLOWLIST_BYTES, "RESOURCE_LIMIT_EXCEEDED",
             "root allowlist exceeds %d bytes" % MAX_ALLOWLIST_BYTES)
    try:
        document = json.loads(target.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise AssetRootError("root allowlist does not parse: %s" % exc, code="ALLOWLIST_INVALID") from exc
    _require(isinstance(document, Mapping), "ALLOWLIST_INVALID", "root allowlist must be one object")
    _require(document.get("schema") == ALLOWLIST_SCHEMA, "ALLOWLIST_INVALID",
             "root allowlist schema must be %s" % ALLOWLIST_SCHEMA)
    raw_roots = document.get("roots")
    _require(isinstance(raw_roots, list) and bool(raw_roots), "ALLOWLIST_INVALID",
             "root allowlist declares no roots")
    import re
    roots: List[Dict[str, Any]] = []
    seen: set = set()
    for raw in raw_roots:
        _require(isinstance(raw, Mapping), "ALLOWLIST_INVALID", "root entry must be an object")
        identifier = str(raw.get("id") or "")
        _require(bool(re.match(_ID_PATTERN, identifier)), "ALLOWLIST_INVALID",
                 "root id must match %s" % _ID_PATTERN)
        _require(identifier not in seen, "ALLOWLIST_INVALID", "duplicate root id: %s" % identifier)
        seen.add(identifier)
        kind = str(raw.get("kind") or "")
        _require(kind in ROOT_KINDS, "ALLOWLIST_INVALID",
                 "root %s kind must be one of %s" % (identifier, ROOT_KINDS))
        declared = raw.get("path")
        _require(isinstance(declared, str) and declared.strip() != "", "ALLOWLIST_INVALID",
                 "root %s has no path" % identifier)
        if kind == "repo":
            _require(is_safe_relative(declared), "UNSAFE_PATH",
                     "repo root %s must be repo-relative: %r" % (identifier, declared))
            candidate = REPO_ROOT / declared
        else:
            _require(not is_safe_relative(declared), "ALLOWLIST_INVALID",
                     "external root %s must be an explicit absolute path: %r" % (identifier, declared))
            candidate = Path(declared)
        _require(candidate.exists(), "ROOT_MISSING", "root %s is absent: %s" % (identifier, candidate))
        _require(candidate.is_dir(), "ROOT_KIND_MISMATCH", "root %s is not a directory: %s" % (identifier, candidate))
        _require(not is_reparse_point(candidate), "ROOT_REPARSE_POINT",
                 "root %s is a reparse point (junction/symlink); refusing it as a root" % identifier)
        canonical = _canonical(candidate)
        if kind == "repo":
            _require(_within(REPO_ROOT, canonical), "PATH_ESCAPE",
                     "repo root %s resolves outside the repository: %s" % (identifier, canonical))
        roots.append({
            "id": identifier,
            "kind": kind,
            "declared": declared,
            "root": str(canonical),
            "is_reparse_point": False,
            "notes": raw.get("notes"),
            "redistribution_allowed": bool(raw.get("redistribution_allowed") is True),
        })
    return {"schema": ALLOWLIST_SCHEMA, "path": str(target), "sha256": _sha256_file(target),
            "roots": roots}


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_by_id(policy: Mapping[str, Any], root_id: str) -> Dict[str, Any]:
    for root in policy.get("roots") or []:
        if root.get("id") == root_id:
            return dict(root)
    raise AssetRootError("root id is not in the allowlist: %r" % (root_id,), code="ROOT_NOT_ALLOWED")


def resolve_asset(policy: Mapping[str, Any], root_id: str, relative: Any, *,
                  must_exist: bool = True, kind: Optional[str] = None) -> Dict[str, Any]:
    """Resolve one asset reference under an allowlisted root and prove confinement."""
    root = root_by_id(policy, root_id)
    _require(is_safe_relative(relative), "UNSAFE_PATH",
             "asset reference must be relative and free of '..': %r" % (relative,))
    root_path = Path(root["root"])
    candidate = root_path / str(relative)
    _require(not is_reparse_point(candidate), "REPARSE_ESCAPE",
             "asset is itself a reparse point: %s" % candidate)
    reparse = _reparse_components(root_path, candidate)
    _require(not reparse, "REPARSE_ESCAPE",
             "asset path crosses reparse components: %s" % "; ".join(reparse[:4]))
    resolved = _canonical(candidate)
    _require(_within(root_path, resolved), "PATH_ESCAPE",
             "resolved asset escapes root %s: %s" % (root_id, resolved))
    if must_exist:
        _require(candidate.exists(), "MISSING_ASSET", "asset is missing: %s" % candidate)
        if kind == "file":
            _require(candidate.is_file(), "ASSET_KIND_MISMATCH", "asset is not a file: %s" % candidate)
        if kind == "dir":
            _require(candidate.is_dir(), "ASSET_KIND_MISMATCH", "asset is not a directory: %s" % candidate)
    record = {
        "root_id": root_id, "root_kind": root["kind"], "root": str(root_path),
        "relative": str(relative), "resolved": str(resolved),
        "within_root": True, "reparse_checked": True,
        "repo_relative": _repo_relative(resolved),
        "exists": bool(candidate.exists()),
    }
    if candidate.is_file():
        record["bytes"] = int(candidate.stat().st_size)
    return record


def _repo_relative(path: Path) -> Optional[str]:
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def describe_policy(policy: Mapping[str, Any]) -> Dict[str, Any]:
    return {"schema": policy.get("schema"), "path": policy.get("path"),
            "sha256": policy.get("sha256"),
            "roots": [{key: root.get(key) for key in ("id", "kind", "root", "notes")}
                      for root in policy.get("roots") or []]}
