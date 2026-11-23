# -*- coding: utf-8 -*-
"""PF03: bundle-level atomic save, read-back verification and schema gating.

The on-disk layout follows technical book section 11.4::

    <root>.ffsrf/
      head.json                  # publication pointer, written last
      versions/<tag>/
        manifest.json            # per-version hashes and denominators
        snapshot.json
        events.jsonl
        support/<observation>.json
        quality/invariants.json
        exports/manifest.json

Publishing a new version is a single atomic pointer replacement.  Every file is
written into a staging directory, fully verified there, then the staging
directory is renamed and ``head.json`` is atomically replaced as the last step.
An interruption before the pointer write can therefore never be observed as a
half-written *new* project: readers always see the previous published version.

PF03-CONTRACT: both core.v1 and core.v2 bundles are explicitly supported.  A
v1 bundle is read-only compatible with pre-addendum engine-level events; the
optional ``migrate_legacy_v1_project`` helper promotes them to public v2 events
without changing their meaning.  An unknown schema_version is still an explicit
SchemaMigrationError, never a guessed conversion.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .contracts import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    SUPPORTED_SCHEMA_VERSIONS,
    ContractError,
    ProjectSnapshot,
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    file_sha256,
    io_path,
    migrate_event_log_to_v2,
    read_json,
    validate_event_actions_for_schema,
)
from .edit_engine import PartialWriteError
from .observed_surface_graph import assert_invariants, check_invariants

BUNDLE_SCHEMA = "fractureflow.surface_product.project_bundle.v1"
HEAD_FILE = "head.json"
VERSIONS_DIR = "versions"
INTERRUPT_STAGES = frozenset({
    "partial_snapshot", "during_support_files", "before_version_publish",
    "after_version_publish",
})


class ProjectStoreError(ContractError):
    def __init__(self, message: str, code: str = "PROJECT_STORE_INVALID") -> None:
        super().__init__(message, code)


class SchemaMigrationError(ProjectStoreError):
    def __init__(self, found: str, expected: str) -> None:
        super().__init__(
            "schema_version %r is not %r; no automatic migration is attempted" % (found, expected),
            "SCHEMA_MIGRATION_UNSUPPORTED",
        )
        self.found = found
        self.expected = expected


@dataclass(frozen=True)
class SaveReceipt:
    head_path: str
    version_dir: str
    project_id: str
    project_version: int
    content_hash: str
    document_sha256: str
    file_count: int
    event_count: int
    support_hashes: Dict[str, str]
    measured_at: str
    recompute_script: str
    verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_path": self.head_path,
            "version_dir": self.version_dir,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "content_hash": self.content_hash,
            "document_sha256": self.document_sha256,
            "file_count": self.file_count,
            "event_count": self.event_count,
            "support_hashes": dict(self.support_hashes),
            "measured_at": self.measured_at,
            "recompute_script": self.recompute_script,
            "verified": self.verified,
        }


_WINDOWS_RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + ["com%d" % index for index in range(1, 10)]
    + ["lpt%d" % index for index in range(1, 10)]
)
_SUPPORT_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _support_relative_path(observation_id: str) -> str:
    """Return the bundle-relative support file name for one observation.

    The public ID grammar accepts ':' (and a few names Windows reserves), but
    Win32 cannot create support/obs:alt.json -- a contract-valid project then
    failed to save with WinError 87 inside the staging directory (measured with
    a colon-bearing observation ID).  Unsafe names are therefore mapped
    deterministically and collision-free: unsafe characters become '_' and the
    first 8 hex digits of the observation ID's sha256 are appended.  Already
    safe names keep their historical file path verbatim, so every existing
    bundle stays byte-identical.
    """
    if not _SUPPORT_NAME_UNSAFE.search(observation_id) and observation_id.lower() not in _WINDOWS_RESERVED_NAMES:
        return "support/%s.json" % observation_id
    safe = _SUPPORT_NAME_UNSAFE.sub("_", observation_id)
    if safe.lower() in _WINDOWS_RESERVED_NAMES:
        safe = "_" + safe
    return "support/%s_%s.json" % (safe, hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:8])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_bytes(path: Path, data: bytes) -> str:
    target = Path(io_path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        return file_sha256(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _support_payload(project: ProjectSnapshot) -> Dict[str, Dict[str, Any]]:
    payload = project.to_dict()
    result: Dict[str, Dict[str, Any]] = {}
    for observation in payload["observations"]:
        observation_id = observation["observation_id"]
        indices = sorted({
            index
            for fragment in payload["fragments"]
            if fragment["observation_id"] == observation_id
            for index in fragment["support_ref"]["indices"]
        })
        body = {
            "observation_id": observation_id,
            "sha256": observation["sha256"],
            "source_hashes": sorted({item["sha256"] for item in observation["source_files"]}),
            "roi_ids": sorted(roi["roi_id"] for roi in observation["roi"]),
            "support_indices": indices,
            "storage_uris": sorted({
                fragment["support_ref"]["storage_uri"]
                for fragment in payload["fragments"]
                if fragment["observation_id"] == observation_id
            }),
            "enumerated_fragments": sum(
                1 for fragment in payload["fragments"]
                if fragment["observation_id"] == observation_id
            ),
        }
        body["content_hash"] = canonical_sha256(body)
        result[observation_id] = body
    return result


def migrate_legacy_v1_project(project: ProjectSnapshot) -> ProjectSnapshot:
    """Promote a read-only v1 bundle to the v2 public edit-action contract.

    Only the schema marker and the missing public ``EditEvent`` payloads of
    legacy engine-level events are filled in.  Project version, event IDs,
    actions, targets, payloads, timestamps and business geometry are preserved
    verbatim; the content hash is recomputed because the serialized bytes change.
    This is an explicit, optional migration -- ``load`` never performs it silently.
    """
    if not isinstance(project, ProjectSnapshot):
        raise ProjectStoreError("migrate requires ProjectSnapshot", "OBJECT_REQUIRED")
    if project.schema_version != SCHEMA_VERSION:
        raise ProjectStoreError("migration source must be a core.v1 project",
                                "SCHEMA_MIGRATION_UNSUPPORTED")
    payload = copy.deepcopy(project.to_dict())
    payload.pop("content_hash", None)
    artifacts = dict(payload.get("artifacts") or {})
    legacy_events = list(artifacts.get("event_log") or [])
    payload["schema_version"] = SCHEMA_VERSION_V2
    artifacts["event_log"] = migrate_event_log_to_v2(legacy_events)
    payload["artifacts"] = artifacts
    payload["content_hash"] = canonical_sha256(payload)
    migrated = ProjectSnapshot.from_dict(payload)
    assert_invariants(migrated)
    return migrated


class ProjectStore:
    """Versioned, single-writer project bundle with strict read-back."""

    def __init__(self, root: os.PathLike[str] | str, *, recompute_script: str = "") -> None:
        self.root = Path(root)
        self.recompute_script = recompute_script or "src/fractureflow/surface_product/project_store.py"

    # -- writing --------------------------------------------------------------

    def save(
        self,
        project: ProjectSnapshot,
        *,
        interrupt: Optional[str] = None,
    ) -> SaveReceipt:
        if not isinstance(project, ProjectSnapshot):
            raise ProjectStoreError("save requires ProjectSnapshot", "OBJECT_REQUIRED")
        if interrupt is not None and interrupt not in INTERRUPT_STAGES:
            raise ProjectStoreError("unknown interrupt stage %r" % interrupt, "BAD_INTERRUPT_STAGE")
        report = check_invariants(project)
        if not report.ok:
            raise ProjectStoreError(
                "refusing to save a project with dangling references: %s"
                % [item["code"] for item in report.issues],
                "INVARIANT_VIOLATION",
            )
        document = project.to_dict()
        content_hash = document["content_hash"]
        events = list((project.artifacts or {}).get("event_log") or [])
        catalog = validate_event_actions_for_schema(events, project.schema_version)
        if not catalog["ok"]:
            raise ProjectStoreError(
                "event log violates the %s public action catalog: %s"
                % (project.schema_version, catalog["rejected"]), "EVENT_CATALOG_INVALID",
            )
        events_blob = "".join(canonical_json(event) + "\n" for event in events).encode("utf-8")
        support = _support_payload(project)
        measured_at = _now_iso()

        json_files: Dict[str, Any] = {
            "snapshot.json": document,
            "quality/invariants.json": report.to_dict(),
            "exports/manifest.json": {
                "schema": BUNDLE_SCHEMA,
                "project_version": project.project_version,
                "exports": [],
                "scope": "PF03 synthetic bundle; no external product export is claimed",
            },
        }
        for observation_id, body in support.items():
            json_files[_support_relative_path(observation_id)] = body

        tag = "v%06d_%s" % (project.project_version, content_hash[:12])
        version_dir = self.root / VERSIONS_DIR / tag
        if Path(io_path(version_dir)).exists():
            tag = "%s_%s" % (tag, uuid.uuid4().hex[:8])
            version_dir = self.root / VERSIONS_DIR / tag
        staging = Path(io_path(self.root / (".staging_" + uuid.uuid4().hex)))
        staging.mkdir(parents=True, exist_ok=False)

        # 1. Write every file into staging, with an optional injected interruption.
        if interrupt == "partial_snapshot":
            _atomic_write_bytes(staging / "snapshot.json", canonical_json(document).encode("utf-8")[:24])
            raise PartialWriteError("partial_snapshot@%s" % staging)
        # Every nested path (support/<observation>.json) gets its parent
        # directory created at this single write entry.  The shared
        # atomic_write_json also creates parents, but the guarantee belongs to
        # the staging layout, not to a helper's good behaviour: with a long
        # bundle path the Win32 file path can exceed MAX_PATH and the write then
        # fails with FileNotFoundError even though the directory exists.
        # io_path opts those calls into the extended-length path API.
        for relative, value in json_files.items():
            target = Path(io_path(staging / relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target, value)
            if relative == "snapshot.json" and interrupt == "during_support_files":
                raise PartialWriteError("during_support_files@%s" % staging)
        _atomic_write_bytes(staging / "events.jsonl", events_blob)

        file_hashes = {
            relative: file_sha256(staging / relative)
            for relative in sorted(set(json_files) | {"events.jsonl"})
        }
        support_hashes = {
            relative: file_hashes[relative]
            for relative in file_hashes if relative.startswith("support/")
        }
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": project.schema_version,
            "project_id": project.project_id,
            "project_version": project.project_version,
            "content_hash": content_hash,
            "snapshot_sha256": file_hashes["snapshot.json"],
            "events_sha256": file_hashes["events.jsonl"],
            "support_hashes": support_hashes,
            "files": file_hashes,
            "file_count": len(file_hashes),
            "event_count": len(events),
            "event_catalog": {
                "enumerated_events": catalog["enumerated_events"],
                "contract_events": catalog["contract_events"],
                "legacy_engine_events": len(catalog["legacy_engine_events"]),
                "rejected": catalog["rejected"],
                "public_action_denominator": catalog["public_action_denominator"],
            },
            "invariant_counts": report.counts,
            "measured_at": measured_at,
            "recompute_script": self.recompute_script,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        manifest_sha256 = file_sha256(staging / "manifest.json")

        # 2. Verify the complete staging bundle BEFORE publishing anything.
        self._verify_bundle(staging, manifest, expected_head=None)
        if interrupt == "before_version_publish":
            raise PartialWriteError("before_version_publish@%s" % staging)

        # 3. Publish the version directory, then the head pointer (last write).
        publish_dir = Path(io_path(version_dir))
        publish_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging), str(publish_dir))
        if interrupt == "after_version_publish":
            raise PartialWriteError("after_version_publish@%s" % version_dir)
        head = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": project.schema_version,
            "project_id": project.project_id,
            "project_version": project.project_version,
            "content_hash": content_hash,
            "version_dir": tag,
            "snapshot_sha256": manifest["snapshot_sha256"],
            "manifest_sha256": manifest_sha256,
            "measured_at": measured_at,
            "recompute_script": self.recompute_script,
        }
        atomic_write_json(self.root / HEAD_FILE, head)

        # 4. Strict read-back through the public API.
        _loaded, receipt = self.load()
        if not receipt["verified"] or receipt["content_hash"] != content_hash:
            raise ProjectStoreError("published bundle failed read-back", "READBACK_MISMATCH")
        return SaveReceipt(
            head_path=HEAD_FILE,
            version_dir=tag,
            project_id=project.project_id,
            project_version=project.project_version,
            content_hash=content_hash,
            document_sha256=manifest["snapshot_sha256"],
            file_count=int(manifest["file_count"]),
            event_count=len(events),
            support_hashes=support_hashes,
            measured_at=measured_at,
            recompute_script=self.recompute_script,
            verified=True,
        )

    # -- reading --------------------------------------------------------------

    def load(self) -> Tuple[ProjectSnapshot, Dict[str, Any]]:
        head_path = Path(io_path(self.root / HEAD_FILE))
        if not head_path.is_file():
            raise ProjectStoreError("no published head.json; an interrupted save has no version", "NO_PUBLISHED_HEAD")
        head = read_json(head_path)
        if head.get("schema") != BUNDLE_SCHEMA:
            raise SchemaMigrationError(str(head.get("schema")), BUNDLE_SCHEMA)
        tag = head.get("version_dir")
        if not isinstance(tag, str) or not tag or "/" in tag or "\\" in tag or ".." in tag:
            raise ProjectStoreError("head.version_dir is not a safe relative directory name", "UNSAFE_PATH")
        version_dir = self.root / VERSIONS_DIR / tag
        if not Path(io_path(version_dir)).is_dir():
            raise ProjectStoreError("published version directory is missing", "VERSION_MISSING")
        manifest = read_json(version_dir / "manifest.json")
        if head.get("manifest_sha256") != file_sha256(version_dir / "manifest.json"):
            raise ProjectStoreError("head.manifest_sha256 differs from the published manifest", "MANIFEST_HASH_MISMATCH")
        manifest_schema = manifest.get("schema_version")
        if manifest_schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaMigrationError(str(manifest_schema),
                                       "|".join(sorted(SUPPORTED_SCHEMA_VERSIONS)))
        # Older heads may omit the marker; the manifest remains authoritative.
        head_schema = head.get("schema_version", manifest_schema)
        if head_schema != manifest_schema:
            raise SchemaMigrationError(str(head_schema), str(manifest_schema))
        if (manifest.get("project_version") != head.get("project_version")
                or manifest.get("content_hash") != head.get("content_hash")):
            raise ProjectStoreError("head and manifest disagree on version or hash", "HEAD_MANIFEST_MISMATCH")
        # W170 保存性能：_verify_bundle 已对同一份 snapshot.json 完成 read_json + from_dict +
        # check_invariants（及逐文件哈希、事件目录、content_hash、quality 校验）。这里复用其解析
        # 结果，不再二次 read_json/from_dict；assert_invariants 保留作发布前最后一道断言。
        # 全部校验项与 verified 读回语义不削弱，仅消除一次冗余的全量反序列化。
        parsed: List[Any] = []
        receipt = self._verify_bundle(version_dir, manifest, expected_head=head, _parsed_out=parsed)
        project = parsed[0]
        assert_invariants(project)
        return project, receipt

    def verify(self) -> Dict[str, Any]:
        _project, receipt = self.load()
        return receipt

    # -- internal verification ------------------------------------------------

    def _verify_bundle(
        self,
        directory: Path,
        manifest: Mapping[str, Any],
        *,
        expected_head: Optional[Mapping[str, Any]],
        _parsed_out: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        files = manifest.get("files") or {}
        if not isinstance(files, dict) or not files:
            raise ProjectStoreError("manifest enumerates no files", "EMPTY_ENUMERATION")
        enumerated = 0
        required = {"snapshot.json", "events.jsonl", "quality/invariants.json", "exports/manifest.json"}
        if not required.issubset(set(files)):
            raise ProjectStoreError("bundle is missing required files", "FILE_MISSING")
        for relative, expected_sha in sorted(files.items()):
            if (not isinstance(relative, str) or relative.startswith("/")
                    or ".." in Path(relative).parts or "\\" in relative):
                raise ProjectStoreError("unsafe relative path in manifest", "UNSAFE_PATH")
            path = Path(io_path(directory / relative))
            if not path.is_file():
                raise ProjectStoreError("bundle file is missing: %s" % relative, "FILE_MISSING")
            actual_sha = file_sha256(path)
            if actual_sha != expected_sha:
                raise ProjectStoreError("hash mismatch for %s" % relative, "HASH_MISMATCH")
            enumerated += 1
        support_hashes = manifest.get("support_hashes") or {}
        for relative, expected_sha in sorted(support_hashes.items()):
            if files.get(relative) != expected_sha:
                raise ProjectStoreError("support hash disagrees with file list: %s" % relative, "SUPPORT_HASH_MISMATCH")
        events_path = Path(io_path(directory / "events.jsonl"))
        event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(event_lines) != int(manifest.get("event_count", -1)):
            raise ProjectStoreError("event log line count differs from manifest", "EVENT_COUNT_MISMATCH")
        event_ids = []
        parsed_events = []
        for line in event_lines:
            event = json.loads(line)
            event_ids.append(event.get("event_id"))
            parsed_events.append(event)
        if len(set(event_ids)) != len(event_ids):
            raise ProjectStoreError("duplicate event ID in events.jsonl", "DUPLICATE_EVENT_ID")

        document = read_json(directory / "snapshot.json")
        document_schema = document.get("schema_version")
        if document_schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaMigrationError(str(document_schema),
                                       "|".join(sorted(SUPPORTED_SCHEMA_VERSIONS)))
        manifest_schema = manifest.get("schema_version")
        if document_schema != manifest_schema:
            raise SchemaMigrationError(str(document_schema), str(manifest_schema))
        catalog = validate_event_actions_for_schema(parsed_events, document_schema)
        if not catalog["ok"]:
            raise ProjectStoreError(
                "published event log violates the %s public action catalog: %s"
                % (document_schema, catalog["rejected"]), "EVENT_CATALOG_INVALID",
            )
        if document.get("content_hash") != manifest.get("content_hash"):
            raise ProjectStoreError("snapshot content_hash differs from manifest", "CONTENT_HASH_MISMATCH")
        if expected_head is not None and document.get("content_hash") != expected_head.get("content_hash"):
            raise ProjectStoreError("snapshot content_hash differs from head", "HEAD_HASH_MISMATCH")
        try:
            project = ProjectSnapshot.from_dict(document)
        except ContractError as exc:
            raise ProjectStoreError("snapshot contract invalid: %s" % exc, "SNAPSHOT_INVALID") from exc
        invariants = check_invariants(project)
        if not invariants.ok:
            raise ProjectStoreError("snapshot has dangling references", "INVARIANT_VIOLATION")
        quality = read_json(directory / "quality" / "invariants.json")
        if not quality.get("ok") or quality.get("counts", {}).get("enumerated_total", 0) <= 0:
            raise ProjectStoreError("quality report is empty or failed", "QUALITY_INVALID")
        expected_document_sha = file_sha256(directory / "snapshot.json")
        # W170 保存性能：把已通过全部校验的解析结果交给调用方复用（如 load），
        # 避免对同一份 snapshot.json 二次 read_json + from_dict。校验项一字未减。
        if _parsed_out is not None:
            _parsed_out.append(project)
        return {
            "verified": True,
            "schema": BUNDLE_SCHEMA,
            "schema_version": project.schema_version,
            "project_id": project.project_id,
            "project_version": project.project_version,
            "content_hash": document["content_hash"],
            "document_sha256": expected_document_sha,
            "support_hashes": dict(support_hashes),
            "enumerated_files": enumerated,
            "event_count": len(event_lines),
            "event_catalog": {
                "enumerated_events": catalog["enumerated_events"],
                "contract_events": catalog["contract_events"],
                "legacy_engine_events": len(catalog["legacy_engine_events"]),
                "public_action_denominator": catalog["public_action_denominator"],
            },
            "invariant_counts": invariants.counts,
            "measured_at": _now_iso(),
            "recompute_script": self.recompute_script,
        }


__all__ = [
    "BUNDLE_SCHEMA", "INTERRUPT_STAGES", "ProjectStore", "ProjectStoreError",
    "SaveReceipt", "SchemaMigrationError", "migrate_legacy_v1_project",
]
