# -*- coding: utf-8 -*-
"""Versioned, public and executable contracts for the surface product.

The module deliberately stops at data boundaries: it does not infer geological
truth, intersect infinite planes, solve edits or train models.  All persisted
objects are JSON-ready, finite and free of hidden evaluation references.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping as _MappingABC
from dataclasses import MISSING, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from shapely.geometry import Polygon as ShapelyPolygon

SCHEMA_VERSION = "fractureflow.surface_product.core.v1"
FIXTURE_SCHEMA = "fractureflow.surface_product.synthetic_fixture.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

# --- PF03-CONTRACT append-only addendum -------------------------------------
# The v1 schema and its ten edit actions keep their original meaning.  DELETE,
# UNDO and REDO are introduced under an explicitly versioned v2 schema; no v1
# constant, event field or old event meaning is rewritten, and v1 bundles stay
# readable.  This is a public contract addendum, not a solver or UI claim.
SCHEMA_VERSION_V2 = "fractureflow.surface_product.core.v2"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_V2})
LATEST_SCHEMA_VERSION = SCHEMA_VERSION_V2
EDIT_ACTION_ADDENDUM_ID = "fractureflow.surface_product.edit_actions.v2"

RELATION_TYPES = {"same_observed_surface", "intersects_in_roi", "adjacent_in_roi"}
RELATION_STATES = {"yes", "no", "unknown"}
UNKNOWN_REASONS = {
    "NOT_EVALUATED", "OCCLUDED", "OUTSIDE_ROI", "INSUFFICIENT_SUPPORT",
    "CANDIDATE_MISSING", "MODEL_OOD", "CONFLICT", "AMBIGUOUS",
}
EDIT_ACTIONS = {
    "SELECT_SUPPORT", "ADD_FRAGMENT", "SPLIT_SURFACE", "MERGE_SURFACE",
    "EDIT_BOUNDARY", "SET_RELATION", "SET_UNKNOWN", "PROTECT_REGION",
    "UNPROTECT_REGION", "SET_INTERPRETATION",
}
# v1 is frozen; v2 only appends.  Every consumer (engine, machine contract,
# technical book and tests) reads these sets instead of re-declaring strings.
EDIT_ACTIONS_V1 = frozenset(EDIT_ACTIONS)
EXTENDED_EDIT_ACTIONS = frozenset({"DELETE_OBJECT", "UNDO", "REDO"})
EDIT_ACTIONS_V2 = EDIT_ACTIONS_V1 | EXTENDED_EDIT_ACTIONS

# Machine-readable, JSON-ready public addendum.  It is the single source for
# action sets, payload requirements and the undo/redo version policy.
EDIT_ACTION_ADDENDUM_V2: Dict[str, Any] = {
    "addendum_id": EDIT_ACTION_ADDENDUM_ID,
    "introduced_by": "PF03-CONTRACT",
    "base_schema_version": SCHEMA_VERSION,
    "declared_schema_version": SCHEMA_VERSION_V2,
    "policy": "APPEND_ONLY_V1_ACTIONS_AND_OLD_EVENT_MEANING_UNCHANGED",
    "action_sets": {
        SCHEMA_VERSION: sorted(EDIT_ACTIONS_V1),
        SCHEMA_VERSION_V2: sorted(EDIT_ACTIONS_V2),
    },
    "added_actions": {
        "DELETE_OBJECT": {
            "introduced_in": SCHEMA_VERSION_V2,
            "payload_required_keys": [],
            "semantics": "retire the target objects and cascade dangling relations; retired ids are recorded in lineage",
        },
        "UNDO": {
            "introduced_in": SCHEMA_VERSION_V2,
            "payload_required_keys": ["undone_event_id"],
            "version_policy": "NEW_VERSION_NO_VERSION_ROLLBACK",
            "semantics": "restore geometry, relations, provenance, protection, parameters and lineage as a new project version",
        },
        "REDO": {
            "introduced_in": SCHEMA_VERSION_V2,
            "payload_required_keys": ["redone_event_id"],
            "version_policy": "NEW_VERSION_NO_VERSION_ROLLBACK",
            "context_policy": "CHANGED_CONTEXT_IS_RECOMPUTE_REQUIRED_NEVER_SILENTLY_RECOMPUTED",
            "identity_policy": "BYTE_IDENTICAL_RECORDED_RESULT_ONLY",
        },
    },
    "legacy_read_policy": {
        "schema_version": SCHEMA_VERSION,
        "status": "READ_ONLY_COMPATIBLE",
        "rule": "pre-addendum engine-level DELETE_OBJECT/UNDO/REDO events without a contract_event remain readable; current writers never emit them",
    },
    "not_claimed": ["joint solver", "local propagation", "real UI", "real data", "product acceptance"],
}
EDIT_HARDNESS = {"hard", "soft"}
PROVENANCE = {
    "DIRECT_OBSERVATION", "DETERMINISTIC_RULE", "LEARNED_INFERENCE",
    "HUMAN_CONFIRMED", "HUMAN_INTERPRETED", "WEAK_LABEL",
}
PROJECT_MODES = {"manual", "rule_assisted", "learned_assisted"}

# These keys are forbidden at every nesting level, including otherwise unused
# payload dictionaries.  A public object cannot smuggle evaluation references.
_FORBIDDEN_EXACT = {
    "truth", "groundtruth", "hiddentruth", "hiddenreference", "testtruth",
    "testreference", "evaluationreference", "privateanswer", "answerkey",
    "labels", "reference", "references",
}
_FORBIDDEN_PARTS = ("ground_truth", "hidden_truth", "hidden_reference", "test_truth", "test_reference")


class ContractError(ValueError):
    """A public contract boundary violation."""

    def __init__(self, message: str, code: str = "CONTRACT_INVALID") -> None:
        super().__init__(message)
        self.code = code


class VersionConflictError(ContractError):
    """A mutation attempted to overwrite a newer project version."""

    def __init__(self, base_version: int, project_version: int) -> None:
        super().__init__(
            "base_version %s does not match project_version %s" % (base_version, project_version),
            code="VERSION_CONFLICT",
        )
        self.base_version = base_version
        self.project_version = project_version


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def assert_public_payload(value: Any, path: str = "$") -> None:
    """Reject hidden references, non-string keys and non-finite numbers.

    A contract object is walked field by field instead of being serialized
    first: serializing inside the validator made every nested object validate
    its whole subtree again, so one full-document check cost O(n^2) (measured
    on the internal PF02 project: 26,086,172 recursive validator calls and
    ~3.2 s per ``to_dict()``).  Field walking visits every node exactly once
    and keeps the same verdict.
    """
    if isinstance(value, _ContractObject):
        for member in fields(value):
            assert_public_payload(getattr(value, member.name), "%s.%s" % (path, member.name))
        return
    if isinstance(value, _MappingABC):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("non-string JSON key at %s" % path, "NON_STRING_KEY")
            normalised = _normalise_key(key)
            compact = normalised.replace("_", "")
            if (compact in _FORBIDDEN_EXACT or normalised in _FORBIDDEN_EXACT
                    or any(part in normalised for part in _FORBIDDEN_PARTS)):
                raise ContractError("hidden reference key %r at %s" % (key, path), "HIDDEN_REFERENCE")
            assert_public_payload(item, "%s.%s" % (path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_public_payload(item, "%s[%d]" % (path, index))
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite number at %s" % path, "NON_FINITE")
        return
    raise ContractError("non-public JSON value %r at %s" % (type(value).__name__, path), "NON_JSON_VALUE")


def _plain_tree(value: Any) -> Any:
    """Deep public copy; the caller guarantees ``value`` was validated already."""
    if isinstance(value, _ContractObject):
        return value._payload_unchecked()
    if isinstance(value, _MappingABC):
        return {key: _plain_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_tree(item) for item in value]
    return value


def _public_tree(value: Any) -> Any:
    assert_public_payload(value)
    return _plain_tree(value)


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON text; arrays retain their order.

    A ``ProjectSnapshot`` serves the text sealed at construction time (see the
    cache note in its ``__post_init__``); the snapshot is an immutable instance,
    so those bytes are the canonical bytes.  ``serialize_project_uncached``
    recomputes the document from the fields and proves the cold path agrees.
    """
    if isinstance(value, ProjectSnapshot):
        return value._canonical_json()
    return json.dumps(
        _public_tree(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    if isinstance(value, ProjectSnapshot):
        return value._canonical_sha256()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json_validated(value: Any) -> str:
    """Serialize a plain tree that already passed assert_public_payload.

    Byte-identical to canonical_json for any plain JSON tree, including the
    same refusal of NaN/Infinity (allow_nan=False).  It omits the second walk
    of _public_tree because the caller owns a tree that just came out of this
    module's validated boundary (project.to_dict() / business_payload).  It
    refuses contract objects outright: those must go through canonical_json so
    the sealed cache is used.
    """
    if isinstance(value, _ContractObject):
        raise ContractError(
            "canonical_json_validated accepts only plain validated trees; use canonical_json for contract objects",
            "VALIDATED_TREE_REQUIRED",
        )
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def canonical_sha256_validated(value: Any) -> str:
    return hashlib.sha256(canonical_json_validated(value).encode("utf-8")).hexdigest()

AUDIT_ONLY_PROJECT_KEYS = frozenset({"created_at"})


def _strip_audit_only_fields(value: Any) -> Any:
    """Drop audit-time fields from a deep project view; never from storage."""
    if isinstance(value, _MappingABC):
        return {
            key: _strip_audit_only_fields(item)
            for key, item in value.items()
            if key not in AUDIT_ONLY_PROJECT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_audit_only_fields(item) for item in value]
    return value


def project_identity_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Logical project content used by the deterministic identity digest.

    content_hash is the digest result itself.  created_at records WHEN an
    event happened; it is real audit metadata but not part of WHAT the project
    logically is (W157 / D-1).  Geometry, support indices, relations, lineage,
    parameters, event IDs/actions/actors/payloads and event order all stay in
    the digest, so two different logical projects cannot alias.
    """
    body = {key: item for key, item in payload.items() if key != "content_hash"}
    return _strip_audit_only_fields(body)


def project_identity_digest(payload: Mapping[str, Any]) -> str:
    """Same logical project -> same digest at any wall clock."""
    return hashlib.sha256(
        json.dumps(
            project_identity_payload(payload), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def serialize_project_uncached(project: "ProjectSnapshot") -> Dict[str, Any]:
    """Cold-path project serialization used to audit the sealed cache.

    Rebuilds the whole document from the dataclass fields, re-validates it and
    never reads the instance cache.  Any byte or field difference from
    ``project.to_dict()`` is a cache defect, not a formatting preference.
    """
    if not isinstance(project, ProjectSnapshot):
        raise ContractError("serialize_project_uncached requires ProjectSnapshot", "OBJECT_REQUIRED")
    base = project._payload_unchecked(exclude=("content_hash",))
    assert_public_payload(base)
    digest = project_identity_digest(base)
    if project.content_hash is None or project.content_hash == digest:
        stored_hash = digest
    else:
        legacy_digest = hashlib.sha256(json.dumps(
            base, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if project.content_hash != legacy_digest:
            raise ContractError("project content_hash does not match content",
                                "CONTENT_HASH_MISMATCH")
        stored_hash = project.content_hash
    base["content_hash"] = stored_hash
    return base


def io_path(path: os.PathLike[str] | str) -> str:
    """Return a path string that Win32 file APIs accept beyond MAX_PATH.

    On Windows ``tempfile.mkstemp`` and the plain Win32 file APIs fail with
    ``FileNotFoundError [WinError 206/3]`` once the *file* path exceeds 260
    characters, even when every parent directory exists.  A ``\\\\?\\`` prefix
    opts the call into the extended-length path API (measured: a 288-character
    staging path succeeds with the prefix and fails without it).  Short paths
    are returned unchanged so on-disk names and messages stay ordinary.
    """
    text = os.fspath(path)
    if os.name != "nt" or text.startswith("\\\\?\\") or not os.path.isabs(text):
        return text
    if len(text) < 240:
        return text
    absolute = os.path.abspath(text)
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(io_path(path)).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _identity_tree(value: Any, unordered_fields: Tuple[str, ...], parent: Optional[str] = None) -> Any:
    if isinstance(value, _MappingABC):
        return {
            key: _identity_tree(item, unordered_fields, key)
            for key, item in sorted(value.items()) if key != "content_hash"
        }
    if isinstance(value, (list, tuple)):
        items = [_identity_tree(item, unordered_fields, parent) for item in value]
        if parent in unordered_fields:
            items.sort(key=canonical_json)
        return items
    return value


def stable_id(kind: str, payload: Mapping[str, Any], *, unordered_fields: Iterable[str] = ()) -> str:
    """Derive a persistent ID, sorting only fields declared order-insensitive.

    Coordinates remain order-sensitive.  Collection fields such as member IDs
    must be named explicitly in ``unordered_fields``.
    """
    _identifier(kind, "kind")
    if not isinstance(payload, _MappingABC):
        raise ContractError("stable ID payload must be a mapping")
    normalised = _identity_tree(_public_tree(payload), tuple(unordered_fields))
    return "%s_%s" % (kind, canonical_sha256(normalised)[:32])


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("%s must be a non-empty string" % name, "TEXT_REQUIRED")


def _identifier(value: Any, name: str) -> None:
    _text(value, name)
    if not ID_RE.fullmatch(value):
        raise ContractError("%s is not a persistent public ID: %r" % (name, value), "ID_INVALID")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError("%s must be 64 lowercase hexadecimal characters" % name, "SHA256_INVALID")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError("%s must be an integer >= %d" % (name, minimum), "INTEGER_INVALID")


def _number(value: Any, name: str, minimum: Optional[float] = None,
            maximum: Optional[float] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError("%s must be finite numeric" % name, "NON_FINITE")
    if minimum is not None and value < minimum:
        raise ContractError("%s must be >= %s" % (name, minimum), "NUMBER_RANGE")
    if maximum is not None and value > maximum:
        raise ContractError("%s must be <= %s" % (name, maximum), "NUMBER_RANGE")


def _vector(value: Any, name: str, length: int) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ContractError("%s must contain %d coordinates" % (name, length), "VECTOR_INVALID")
    for index, coordinate in enumerate(value):
        _number(coordinate, "%s[%d]" % (name, index))


def _string_ids(value: Any, name: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ContractError("%s must be a %sID array" % (name, "non-empty " if nonempty else ""), "IDS_INVALID")
    for item in value:
        _identifier(item, name)
    if len(set(value)) != len(value):
        raise ContractError("%s contains duplicate IDs" % name, "DUPLICATE_ID")


def _mapping(value: Any, name: str) -> None:
    if not isinstance(value, _MappingABC):
        raise ContractError("%s must be an object" % name, "OBJECT_REQUIRED")
    assert_public_payload(value, name)


def _timestamp(value: Any, name: str) -> None:
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("%s must be ISO-8601" % name, "TIMESTAMP_INVALID") from exc
    if parsed.tzinfo is None:
        raise ContractError("%s must include a timezone" % name, "TIMESTAMP_INVALID")


def _enum(value: Any, allowed: Iterable[str], name: str) -> None:
    if value not in set(allowed):
        raise ContractError("%s has unsupported value %r" % (name, value), "ENUM_INVALID")


# --- PF03-CONTRACT: executable versioned action policy ----------------------

def edit_actions_for_schema(schema_version: str) -> frozenset:
    """Return the public action set declared by one supported schema version."""
    _text(schema_version, "schema_version")
    if schema_version == SCHEMA_VERSION:
        return EDIT_ACTIONS_V1
    if schema_version == SCHEMA_VERSION_V2:
        return EDIT_ACTIONS_V2
    raise ContractError("unsupported schema_version %r" % schema_version, "SCHEMA_VERSION")


def contract_schema_for_action(action: str) -> str:
    """Report the schema version in which one public edit action was introduced."""
    if action in EDIT_ACTIONS_V1:
        return SCHEMA_VERSION
    if action in EXTENDED_EDIT_ACTIONS:
        return SCHEMA_VERSION_V2
    raise ContractError("unknown public edit action %r" % action, "ENUM_INVALID")


def validate_extended_action_payload(action: str, payload: Mapping[str, Any]) -> None:
    """Give the three v2 actions machine-checkable payload semantics."""
    if action not in EXTENDED_EDIT_ACTIONS:
        return
    declaration = EDIT_ACTION_ADDENDUM_V2["added_actions"][action]
    for key in declaration["payload_required_keys"]:
        if key not in payload:
            raise ContractError("%s payload must carry %s" % (action, key), "EXTENDED_ACTION_PAYLOAD")
        _identifier(payload[key], "%s.%s" % (action, key))


def validate_event_actions_for_schema(
    events: Sequence[Mapping[str, Any]],
    schema_version: str,
    *,
    allow_legacy_engine_events: bool = True,
) -> Dict[str, Any]:
    """Check a serialized event log against the versioned public action catalog.

    A v1 bundle may still contain pre-addendum engine-level DELETE/UNDO/REDO
    records (no ``contract_event``): that is a read-only legacy path only.  A v2
    bundle must give every event a validated public ``EditEvent``.  An empty log
    is legal for a new project but is reported as zero enumeration, never as a
    vacuous PASS over action coverage.
    """
    allowed = edit_actions_for_schema(schema_version)
    if not isinstance(events, (list, tuple)):
        raise ContractError("event log must be an array", "EVENT_LOG_INVALID")
    rejected: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []
    contractual = 0
    for index, event in enumerate(events):
        _mapping(event, "event_log[%d]" % index)
        action = event.get("action")
        if action not in EDIT_ACTIONS_V2:
            rejected.append({"index": index, "action": action, "code": "ACTION_NOT_PUBLIC"})
            continue
        has_contract = event.get("contract_event") is not None
        if has_contract:
            try:
                restored = EditEvent.from_dict(event["contract_event"])
            except ContractError as exc:
                rejected.append({"index": index, "action": action,
                                 "code": "CONTRACT_EVENT_INVALID", "detail": exc.code})
                continue
            if restored.action != action:
                rejected.append({"index": index, "action": action, "code": "ACTION_EVENT_MISMATCH"})
                continue
            contractual += 1
        if action in EXTENDED_EDIT_ACTIONS and schema_version == SCHEMA_VERSION:
            if allow_legacy_engine_events and not has_contract and event.get("contract_action") is not True:
                legacy.append({"index": index, "action": action})
            else:
                rejected.append({"index": index, "action": action,
                                 "code": "V2_ACTION_WRITTEN_IN_V1_PROJECT"})
            continue
        if action not in allowed:
            rejected.append({"index": index, "action": action, "code": "ACTION_NOT_IN_SCHEMA"})
            continue
        if schema_version == SCHEMA_VERSION_V2 and not has_contract:
            rejected.append({"index": index, "action": action,
                             "code": "V2_EVENT_MISSING_PUBLIC_EDIT_EVENT"})
            continue
        if has_contract and event.get("contract_action") is not True:
            rejected.append({"index": index, "action": action,
                             "code": "CONTRACT_ACTION_FLAG_MISMATCH"})
    return {
        "schema_version": schema_version,
        "enumerated_events": len(events),
        "public_action_denominator": len(EDIT_ACTIONS_V2),
        "allowed_action_denominator": len(allowed),
        "contract_events": contractual,
        "legacy_engine_events": legacy,
        "rejected": rejected,
        "ok": not rejected,
        "vacuous_pass": False,
        "zero_enumeration": len(events) == 0,
    }


def schema_version_for_event_actions(events: Sequence[Mapping[str, Any]]) -> str:
    """A log containing a v2 action declares the v2 project schema."""
    for event in events:
        if event.get("action") in EXTENDED_EDIT_ACTIONS:
            return SCHEMA_VERSION_V2
    return SCHEMA_VERSION


def migrate_event_log_to_v2(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Promote legacy engine-level events to public EditEvents, meaning unchanged.

    Only ``contract_event`` / ``contract_action`` are filled in; event IDs,
    actions, targets, payloads, base versions, actors and timestamps are copied
    verbatim.  The result is re-validated against the v2 public catalog.
    """
    fields = tuple(EditEvent.__dataclass_fields__)
    migrated: List[Dict[str, Any]] = []
    for index, event in enumerate(events):
        _mapping(event, "legacy event_log[%d]" % index)
        record = dict(event)
        if record.get("contract_event") is None:
            missing = [key for key in fields if key not in record]
            if missing:
                raise ContractError("legacy event %d lacks fields: %s" % (index, missing), "MIGRATION_INVALID")
            record["contract_event"] = EditEvent.from_dict(
                {key: record[key] for key in fields}
            ).to_dict()
            record["contract_action"] = True
        migrated.append(record)
    report = validate_event_actions_for_schema(
        migrated, SCHEMA_VERSION_V2, allow_legacy_engine_events=False
    )
    if not report["ok"]:
        raise ContractError("migrated event log fails the v2 contract: %s" % report["rejected"],
                            "MIGRATION_INVALID")
    return migrated


def _kwargs(cls: type, payload: Mapping[str, Any]) -> Dict[str, Any]:
    _mapping(payload, cls.__name__)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractError("%s has unknown fields: %s" % (cls.__name__, unknown), "UNKNOWN_FIELD")
    missing = [
        field.name for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING and field.name not in payload
    ]
    if missing:
        raise ContractError("%s is missing fields: %s" % (cls.__name__, missing), "MISSING_FIELD")
    return dict(payload)


class _ContractObject:
    def _payload_unchecked(self, *, exclude: Iterable[str] = ()) -> Dict[str, Any]:
        """Build the public payload without re-validating nested subtrees."""
        excluded = set(exclude)
        return {
            member.name: _plain_tree(getattr(self, member.name))
            for member in fields(self) if member.name not in excluded
        }

    def _payload(self, *, exclude: Iterable[str] = ()) -> Dict[str, Any]:
        payload = self._payload_unchecked(exclude=exclude)
        assert_public_payload(payload)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return self._payload()


@dataclass(frozen=True)
class SourceFile(_ContractObject):
    path: str
    sha256: str
    byte_count: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _text(self.path, "source path")
        _sha256(self.sha256, "source sha256")
        _integer(self.byte_count, "byte_count")
        _text(self.media_type, "media_type")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceFile":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class ROI(_ContractObject):
    roi_id: str
    unit: str
    coordinate_frame: str
    bounds_min: List[float]
    bounds_max: List[float]

    def __post_init__(self) -> None:
        _identifier(self.roi_id, "roi_id")
        _text(self.unit, "ROI unit")
        _text(self.coordinate_frame, "ROI coordinate_frame")
        _vector(self.bounds_min, "bounds_min", 3)
        _vector(self.bounds_max, "bounds_max", 3)
        if any(low >= high for low, high in zip(self.bounds_min, self.bounds_max)):
            raise ContractError("ROI bounds must be finite and strictly increasing", "ROI_INVALID")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ROI":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class ObservationBundle(_ContractObject):
    schema_version: str
    observation_id: str
    source_files: List[SourceFile]
    sha256: str
    unit: str
    coordinate_frame: str
    local_origin: List[float]
    roi: List[ROI]
    coverage: Dict[str, Any]
    registration: Dict[str, Any]
    acquisition: Dict[str, Any]
    license: str

    def __post_init__(self) -> None:
        _text(self.schema_version, "observation schema_version")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractError("unsupported observation schema_version", "SCHEMA_VERSION")
        _identifier(self.observation_id, "observation_id")
        if not isinstance(self.source_files, (list, tuple)) or not self.source_files:
            raise ContractError("source_files must be non-empty", "SOURCE_REQUIRED")
        if not all(isinstance(item, SourceFile) for item in self.source_files):
            raise ContractError("source_files must contain SourceFile objects", "SOURCE_REQUIRED")
        paths = [item.path for item in self.source_files]
        if len(paths) != len(set(paths)):
            raise ContractError("duplicate source path", "DUPLICATE_ID")
        _sha256(self.sha256, "observation sha256")
        _text(self.unit, "unit")
        _text(self.coordinate_frame, "coordinate_frame")
        _vector(self.local_origin, "local_origin", 3)
        if not isinstance(self.roi, (list, tuple)) or not self.roi:
            raise ContractError("at least one finite ROI is required", "ROI_INVALID")
        if not all(isinstance(item, ROI) for item in self.roi):
            raise ContractError("roi must contain ROI objects", "ROI_INVALID")
        _string_ids([item.roi_id for item in self.roi], "roi IDs", nonempty=True)
        for item in self.roi:
            if item.unit != self.unit or item.coordinate_frame != self.coordinate_frame:
                raise ContractError("ROI unit/frame must match observation", "UNIT_FRAME_MISMATCH")
        for name in ("coverage", "registration", "acquisition"):
            _mapping(getattr(self, name), name)
        if not self.coverage:
            raise ContractError("coverage cannot be empty", "COVERAGE_REQUIRED")
        _text(self.license, "license")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationBundle":
        values = _kwargs(cls, payload)
        values["source_files"] = [SourceFile.from_dict(item) for item in values["source_files"]]
        values["roi"] = [ROI.from_dict(item) for item in values["roi"]]
        return cls(**values)


@dataclass(frozen=True)
class SupportReference(_ContractObject):
    observation_id: str
    source_sha256: str
    storage_uri: str
    index_kind: str
    indices: List[int]
    roi_id: str
    decimation_lineage: Dict[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "support observation_id")
        _sha256(self.source_sha256, "source_sha256")
        _text(self.storage_uri, "storage_uri")
        _text(self.index_kind, "index_kind")
        if not isinstance(self.indices, (list, tuple)) or not self.indices:
            raise ContractError("support indices must be non-empty", "SUPPORT_REQUIRED")
        for index in self.indices:
            _integer(index, "support index")
        if len(set(self.indices)) != len(self.indices):
            raise ContractError("support indices must be unique", "DUPLICATE_ID")
        _identifier(self.roi_id, "support roi_id")
        _mapping(self.decimation_lineage, "decimation_lineage")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupportReference":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class PlaneLocal(_ContractObject):
    normal: List[float]
    offset: float

    def __post_init__(self) -> None:
        _vector(self.normal, "plane normal", 3)
        norm = math.sqrt(sum(float(item) ** 2 for item in self.normal))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ContractError("plane normal must have unit length", "NORMAL_NOT_UNIT")
        _number(self.offset, "plane offset")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlaneLocal":
        return cls(**_kwargs(cls, payload))


def _ring(value: Any, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise ContractError("%s must contain at least four positions" % name, "RING_INVALID")
    for index, point in enumerate(value):
        _vector(point, "%s[%d]" % (name, index), 2)
    if list(value[0]) != list(value[-1]):
        raise ContractError("%s must be explicitly closed" % name, "RING_NOT_CLOSED")
    for index in range(len(value) - 2):
        if list(value[index]) == list(value[index + 1]):
            raise ContractError("%s contains consecutive duplicate points" % name, "RING_INVALID")


@dataclass(frozen=True)
class FiniteRegion(_ContractObject):
    exterior: List[List[float]]
    holes: List[List[List[float]]]

    def __post_init__(self) -> None:
        _ring(self.exterior, "exterior")
        if not isinstance(self.holes, (list, tuple)):
            raise ContractError("holes must be an array", "RING_INVALID")
        for index, hole in enumerate(self.holes):
            _ring(hole, "holes[%d]" % index)
        polygon = ShapelyPolygon(self.exterior, self.holes)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-12:
            raise ContractError("finite region must be a valid non-zero simple polygon", "POLYGON_INVALID")

    def to_shapely(self) -> ShapelyPolygon:
        return ShapelyPolygon(self.exterior, self.holes)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FiniteRegion":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class Fragment(_ContractObject):
    fragment_id: str
    observation_id: str
    support_ref: SupportReference
    plane_local: PlaneLocal
    finite_regions_local: List[FiniteRegion]
    residuals: List[float]
    quality: Dict[str, Any]
    candidate_generator: Dict[str, Any]
    version: int

    def __post_init__(self) -> None:
        _identifier(self.fragment_id, "fragment_id")
        _identifier(self.observation_id, "fragment observation_id")
        if not isinstance(self.support_ref, SupportReference):
            raise ContractError("support_ref must be a SupportReference", "SUPPORT_REQUIRED")
        if self.support_ref.observation_id != self.observation_id:
            raise ContractError("fragment and support observation IDs differ", "SUPPORT_MISMATCH")
        if not isinstance(self.plane_local, PlaneLocal):
            raise ContractError("plane_local must be a PlaneLocal", "PLANE_INVALID")
        if not isinstance(self.finite_regions_local, (list, tuple)) or not self.finite_regions_local:
            raise ContractError("finite_regions_local must be non-empty", "FINITE_SUPPORT_REQUIRED")
        if not all(isinstance(item, FiniteRegion) for item in self.finite_regions_local):
            raise ContractError("finite_regions_local contains an invalid region", "POLYGON_INVALID")
        if not isinstance(self.residuals, (list, tuple)):
            raise ContractError("residuals must be an array", "NUMBER_RANGE")
        for index, value in enumerate(self.residuals):
            _number(value, "residuals[%d]" % index)
        _mapping(self.quality, "quality")
        _mapping(self.candidate_generator, "candidate_generator")
        _identifier(self.candidate_generator.get("candidate_batch_id"), "candidate_batch_id")
        _sha256(self.candidate_generator.get("generator_config_hash"), "generator_config_hash")
        _integer(self.version, "version", 1)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Fragment":
        values = _kwargs(cls, payload)
        values["support_ref"] = SupportReference.from_dict(values["support_ref"])
        values["plane_local"] = PlaneLocal.from_dict(values["plane_local"])
        values["finite_regions_local"] = [FiniteRegion.from_dict(item) for item in values["finite_regions_local"]]
        return cls(**values)


@dataclass(frozen=True)
class Surface(_ContractObject):
    surface_id: str
    member_fragment_ids: List[str]
    interpretation_status: str
    finite_support: List[FiniteRegion]
    provenance: str
    version: int

    def __post_init__(self) -> None:
        _identifier(self.surface_id, "surface_id")
        _string_ids(self.member_fragment_ids, "member_fragment_ids", nonempty=True)
        _text(self.interpretation_status, "interpretation_status")
        if not isinstance(self.finite_support, (list, tuple)) or not self.finite_support:
            raise ContractError("surface finite_support must be non-empty", "FINITE_SUPPORT_REQUIRED")
        if not all(isinstance(item, FiniteRegion) for item in self.finite_support):
            raise ContractError("surface finite_support contains an invalid region", "POLYGON_INVALID")
        _enum(self.provenance, PROVENANCE, "surface provenance")
        _integer(self.version, "version", 1)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Surface":
        values = _kwargs(cls, payload)
        values["finite_support"] = [FiniteRegion.from_dict(item) for item in values["finite_support"]]
        return cls(**values)


@dataclass(frozen=True)
class Relation(_ContractObject):
    relation_id: str
    src_id: str
    dst_id: str
    relation_type: str
    state: str
    unknown_reason: Optional[str]
    roi_id: str
    evidence_refs: List[str]
    provenance: str
    version: int

    def __post_init__(self) -> None:
        for name in ("relation_id", "src_id", "dst_id", "roi_id"):
            _identifier(getattr(self, name), name)
        if self.src_id == self.dst_id:
            raise ContractError("relation endpoints must differ", "RELATION_INVALID")
        _enum(self.relation_type, RELATION_TYPES, "relation_type")
        _enum(self.state, RELATION_STATES, "relation state")
        if self.state == "unknown":
            _enum(self.unknown_reason, UNKNOWN_REASONS, "unknown_reason")
        elif self.unknown_reason is not None:
            raise ContractError("unknown_reason is only valid for unknown state", "UNKNOWN_REASON_INVALID")
        _string_ids(self.evidence_refs, "evidence_refs", nonempty=True)
        _enum(self.provenance, PROVENANCE, "relation provenance")
        _integer(self.version, "version", 1)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Relation":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class EditEvent(_ContractObject):
    event_id: str
    project_id: str
    base_version: int
    parent_event_id: Optional[str]
    actor: str
    action: str
    target_ids: List[str]
    payload: Dict[str, Any]
    tolerance: float
    hardness: str
    created_at: str

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.project_id, "project_id")
        _integer(self.base_version, "base_version")
        if self.parent_event_id is not None:
            _identifier(self.parent_event_id, "parent_event_id")
        _text(self.actor, "actor")
        _enum(self.action, EDIT_ACTIONS_V2, "edit action")
        validate_extended_action_payload(self.action, self.payload)
        _string_ids(self.target_ids, "target_ids", nonempty=True)
        _mapping(self.payload, "edit payload")
        _number(self.tolerance, "tolerance", 0.0)
        _enum(self.hardness, EDIT_HARDNESS, "edit hardness")
        _timestamp(self.created_at, "created_at")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EditEvent":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class ScoreBundle(_ContractObject):
    model_version: str
    candidate_batch_id: str
    association_scores: Dict[str, float]
    relation_scores: Dict[str, float]
    geometry_deltas: Dict[str, Any]
    raw_uncertainty: Dict[str, float]
    feature_summary: Dict[str, Any]
    inference_config_hash: str

    def __post_init__(self) -> None:
        _text(self.model_version, "model_version")
        _identifier(self.candidate_batch_id, "candidate_batch_id")
        for name in ("association_scores", "relation_scores", "raw_uncertainty"):
            scores = getattr(self, name)
            _mapping(scores, name)
            for key, value in scores.items():
                _identifier(key, "%s key" % name)
                _number(value, "%s[%s]" % (name, key), 0.0 if name == "raw_uncertainty" else None)
        _mapping(self.geometry_deltas, "geometry_deltas")
        _mapping(self.feature_summary, "feature_summary")
        _sha256(self.inference_config_hash, "inference_config_hash")

    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoreBundle":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class UpdateResult(_ContractObject):
    accepted: bool
    project_version: int
    changed_ids: List[str]
    affected_ids: List[str]
    reason_codes: List[str]
    energy_trace: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    warnings: List[str]
    content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ContractError("accepted must be boolean", "BOOLEAN_REQUIRED")
        _integer(self.project_version, "project_version")
        _string_ids(self.changed_ids, "changed_ids")
        _string_ids(self.affected_ids, "affected_ids")
        if not isinstance(self.reason_codes, (list, tuple)) or not self.reason_codes:
            raise ContractError("reason_codes must be non-empty", "REASON_REQUIRED")
        for reason in self.reason_codes:
            _text(reason, "reason_code")
        if not isinstance(self.energy_trace, (list, tuple)):
            raise ContractError("energy_trace must be an array", "ENERGY_INVALID")
        if self.accepted and not self.energy_trace:
            raise ContractError("accepted updates require an energy trace", "ENERGY_INVALID")
        score_hashes = set()
        previous_energy = None
        previous_step = -1
        for index, step in enumerate(self.energy_trace):
            _mapping(step, "energy_trace[%d]" % index)
            if not {"step", "energy", "score_bundle_hash"}.issubset(step):
                raise ContractError("energy step lacks step/energy/score_bundle_hash", "ENERGY_INVALID")
            _integer(step["step"], "energy step", 0)
            if step["step"] <= previous_step:
                raise ContractError("energy steps must be strictly increasing", "ENERGY_INVALID")
            previous_step = step["step"]
            _number(step["energy"], "energy")
            if previous_energy is not None and step["energy"] > previous_energy:
                raise ContractError("energy must be monotonically non-increasing", "ENERGY_NOT_MONOTONIC")
            previous_energy = step["energy"]
            _sha256(step["score_bundle_hash"], "score_bundle_hash")
            score_hashes.add(step["score_bundle_hash"])
        if len(score_hashes) > 1:
            raise ContractError("scores changed within one local solve", "SCORE_MUTATION")
        if not isinstance(self.conflicts, (list, tuple)):
            raise ContractError("conflicts must be an array", "CONFLICT_INVALID")
        for conflict in self.conflicts:
            _mapping(conflict, "conflict")
        if not isinstance(self.warnings, (list, tuple)) or not all(isinstance(item, str) for item in self.warnings):
            raise ContractError("warnings must be a string array", "WARNING_INVALID")
        _sha256(self.content_hash, "update content_hash")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UpdateResult":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class ReliabilityReport(_ContractObject):
    calibration_version: str
    coverage: float
    risk: float
    review_queue: List[str]
    ood_ids: List[str]
    conflict_ids: List[str]
    unknown_reason_counts: Dict[str, int]
    scope: str
    denominator: int
    measured_at: str
    recompute_script: str

    def __post_init__(self) -> None:
        _text(self.calibration_version, "calibration_version")
        _number(self.coverage, "coverage", 0.0, 1.0)
        _number(self.risk, "risk", 0.0, 1.0)
        for name in ("review_queue", "ood_ids", "conflict_ids"):
            _string_ids(getattr(self, name), name)
        _mapping(self.unknown_reason_counts, "unknown_reason_counts")
        for reason, count in self.unknown_reason_counts.items():
            _enum(reason, UNKNOWN_REASONS, "unknown reason")
            _integer(count, "unknown reason count")
        _text(self.scope, "scope")
        _integer(self.denominator, "denominator", 1)
        if sum(self.unknown_reason_counts.values()) > self.denominator:
            raise ContractError("unknown counts exceed denominator", "DENOMINATOR_INVALID")
        _timestamp(self.measured_at, "measured_at")
        _text(self.recompute_script, "recompute_script")

    def capability_acceptance_eligible(self) -> bool:
        """Coverage and a non-all-unknown denominator are mandatory."""
        return self.coverage > 0.0 and sum(self.unknown_reason_counts.values()) < self.denominator

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityReport":
        return cls(**_kwargs(cls, payload))


@dataclass(frozen=True)
class ProtectedRegion(_ContractObject):
    region_id: str
    roi_id: str
    finite_regions: List[FiniteRegion]

    def __post_init__(self) -> None:
        _identifier(self.region_id, "region_id")
        _identifier(self.roi_id, "protected roi_id")
        if not isinstance(self.finite_regions, (list, tuple)) or not self.finite_regions:
            raise ContractError("protected region requires finite geometry", "FINITE_SUPPORT_REQUIRED")
        if not all(isinstance(item, FiniteRegion) for item in self.finite_regions):
            raise ContractError("invalid protected finite region", "POLYGON_INVALID")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtectedRegion":
        values = _kwargs(cls, payload)
        values["finite_regions"] = [FiniteRegion.from_dict(item) for item in values["finite_regions"]]
        return cls(**values)


@dataclass(frozen=True)
class ProjectSnapshot(_ContractObject):
    schema_version: str
    project_id: str
    project_version: int
    observations: List[ObservationBundle]
    fragments: List[Fragment]
    surfaces: List[Surface]
    relations: List[Relation]
    event_head: Optional[str]
    parameters: Dict[str, Any]
    protected_regions: List[ProtectedRegion]
    artifacts: Dict[str, Any]
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.schema_version, "project schema_version")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractError("unsupported project schema_version", "SCHEMA_VERSION")
        _identifier(self.project_id, "project_id")
        _integer(self.project_version, "project_version")
        for name, expected in (
            ("observations", ObservationBundle), ("fragments", Fragment),
            ("surfaces", Surface), ("relations", Relation),
            ("protected_regions", ProtectedRegion),
        ):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or not all(isinstance(item, expected) for item in value):
                raise ContractError("%s must contain %s objects" % (name, expected.__name__), "OBJECT_REQUIRED")
        _string_ids([item.observation_id for item in self.observations], "observation IDs")
        _string_ids([item.fragment_id for item in self.fragments], "fragment IDs")
        _string_ids([item.surface_id for item in self.surfaces], "surface IDs")
        _string_ids([item.relation_id for item in self.relations], "relation IDs")
        _string_ids([item.region_id for item in self.protected_regions], "protected region IDs")
        geometry_ids = [item.fragment_id for item in self.fragments] + [item.surface_id for item in self.surfaces]
        _string_ids(geometry_ids, "geometry IDs")
        if self.event_head is not None:
            _identifier(self.event_head, "event_head")
        _mapping(self.parameters, "parameters")
        if "project_mode" in self.parameters:
            _enum(self.parameters["project_mode"], PROJECT_MODES, "project_mode")
        _mapping(self.artifacts, "artifacts")

        observations = {item.observation_id: item for item in self.observations}
        all_rois = {roi.roi_id for observation in self.observations for roi in observation.roi}
        for fragment in self.fragments:
            observation = observations.get(fragment.observation_id)
            if observation is None:
                raise ContractError("fragment has dangling observation reference", "DANGLING_REFERENCE")
            if fragment.support_ref.roi_id not in {roi.roi_id for roi in observation.roi}:
                raise ContractError("fragment support has dangling ROI reference", "DANGLING_REFERENCE")
            source_hashes = {observation.sha256} | {item.sha256 for item in observation.source_files}
            if fragment.support_ref.source_sha256 not in source_hashes:
                raise ContractError("support source hash is absent from observation", "SOURCE_MISMATCH")
        fragment_ids = {item.fragment_id for item in self.fragments}
        for surface in self.surfaces:
            if not set(surface.member_fragment_ids).issubset(fragment_ids):
                raise ContractError("surface has dangling fragment reference", "DANGLING_REFERENCE")
        for relation in self.relations:
            if relation.src_id not in geometry_ids or relation.dst_id not in geometry_ids:
                raise ContractError("relation has dangling endpoint", "DANGLING_REFERENCE")
            if relation.roi_id not in all_rois:
                raise ContractError("relation has dangling ROI reference", "DANGLING_REFERENCE")
        for protected in self.protected_regions:
            if protected.roi_id not in all_rois:
                raise ContractError("protected region has dangling ROI reference", "DANGLING_REFERENCE")
        # -- sealed serialization cache --------------------------------------
        # A ProjectSnapshot is immutable by construction: the dataclass is
        # frozen, every nested contract object is frozen, and every content
        # change in this code base builds a NEW snapshot (``from_dict``,
        # ``dataclasses.replace`` or ``EditEngine._assemble``).  The cache is
        # therefore keyed by the instance itself; its invalidation rule is
        # "new content => new instance => new cache".  There is no global or
        # cross-instance cache, so no stale bytes can be served to a different
        # project state.  ``to_dict`` returns a deep copy so callers cannot
        # mutate the sealed payload, and ``serialize_project_uncached``
        # recomputes the document from the fields for byte-level auditing.
        base = self._payload_unchecked(exclude=("content_hash",))
        assert_public_payload(base)
        if self.content_hash is not None:
            _sha256(self.content_hash, "project content_hash")
        # W157 / D-1: the identity digest drops audit-only created_at fields,
        # so the same logical project replays to the same content_hash at a
        # different wall clock.  Legacy artifacts hashed the raw payload with
        # timestamps included; they stay readable and their stored hash bytes
        # are preserved, never rewritten.
        digest = project_identity_digest(base)
        if self.content_hash is None or self.content_hash == digest:
            stored_hash = digest
        else:
            legacy_digest = hashlib.sha256(json.dumps(
                base, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if self.content_hash != legacy_digest:
                raise ContractError("project content_hash does not match content",
                                    "CONTENT_HASH_MISMATCH")
            stored_hash = self.content_hash
        payload = dict(base)
        payload["content_hash"] = stored_hash
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        object.__setattr__(self, "_serialized_payload", payload)
        object.__setattr__(self, "_serialized_json", text)
        object.__setattr__(self, "_serialized_digest", stored_hash)
        object.__setattr__(self, "_serialized_sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())

    # -- cached serialization (see the seal note in __post_init__) -----------

    def compute_content_hash(self) -> str:
        return self._serialized_digest

    def _canonical_json(self) -> str:
        return self._serialized_json

    def _canonical_sha256(self) -> str:
        return self._serialized_sha256

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._serialized_payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectSnapshot":
        values = _kwargs(cls, payload)
        if "content_hash" not in payload:
            raise ContractError("serialized project requires content_hash", "MISSING_FIELD")
        values["observations"] = [ObservationBundle.from_dict(item) for item in values["observations"]]
        values["fragments"] = [Fragment.from_dict(item) for item in values["fragments"]]
        values["surfaces"] = [Surface.from_dict(item) for item in values["surfaces"]]
        values["relations"] = [Relation.from_dict(item) for item in values["relations"]]
        values["protected_regions"] = [ProtectedRegion.from_dict(item) for item in values["protected_regions"]]
        return cls(**values)


def check_base_version(base_version: int, project_version: int) -> None:
    """Reject stale writes before any mutation can occur."""
    _integer(base_version, "base_version")
    _integer(project_version, "project_version")
    if base_version != project_version:
        raise VersionConflictError(base_version, project_version)


def find_hard_conflicts(events: Sequence[EditEvent]) -> List[Dict[str, Any]]:
    """Return incompatible hard edits at the same base version; never relax them."""
    conflicts: List[Dict[str, Any]] = []
    hard = [event for event in events if event.hardness == "hard"]
    for left_index, left in enumerate(hard):
        for right in hard[left_index + 1:]:
            if left.event_id == right.event_id:
                raise ContractError("duplicate edit event ID", "DUPLICATE_ID")
            same_write = (
                left.project_id == right.project_id
                and left.base_version == right.base_version
                and bool(set(left.target_ids) & set(right.target_ids))
            )
            different_intent = left.action != right.action or canonical_json(left.payload) != canonical_json(right.payload)
            if same_write and different_intent:
                conflicts.append({
                    "code": "HARD_EDIT_CONFLICT",
                    "event_ids": sorted([left.event_id, right.event_id]),
                    "target_ids": sorted(set(left.target_ids) & set(right.target_ids)),
                })
    return conflicts


def _no_duplicate_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key %r" % key, "DUPLICATE_KEY")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ContractError("non-finite JSON constant %s" % value, "NON_FINITE")


def read_json(path: os.PathLike[str] | str) -> Any:
    with Path(io_path(path)).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_no_duplicate_pairs, parse_constant=_invalid_constant)
    assert_public_payload(value)
    return value


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> str:
    """Write canonical JSON using same-directory flush, replace and readback."""
    destination = Path(io_path(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
        readback = read_json(destination)
        if canonical_json(readback) != canonical_json(value):
            raise ContractError("atomic readback differs from source", "READBACK_MISMATCH")
        return canonical_sha256(readback)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def save_project(project: ProjectSnapshot, destination: os.PathLike[str] | str) -> Dict[str, Any]:
    if not isinstance(project, ProjectSnapshot):
        raise ContractError("save_project requires ProjectSnapshot", "OBJECT_REQUIRED")
    document = project.to_dict()
    document_sha256 = atomic_write_json(destination, document)
    return {
        "project_id": project.project_id,
        "project_version": project.project_version,
        "content_hash": document["content_hash"],
        "document_sha256": document_sha256,
        "path": str(Path(destination)),
    }


def _fixture_objects(document: Mapping[str, Any]) -> List[_ContractObject]:
    constructors: Dict[str, Callable[[Mapping[str, Any]], _ContractObject]] = {
        "Fragment": Fragment.from_dict,
        "Surface": Surface.from_dict,
        "Relation": Relation.from_dict,
        "EditEvent": EditEvent.from_dict,
        "FiniteRegion": FiniteRegion.from_dict,
        "ProtectedRegion": ProtectedRegion.from_dict,
    }
    objects: List[_ContractObject] = []
    for index, item in enumerate(document["objects"]):
        _mapping(item, "objects[%d]" % index)
        if set(item) != {"type", "data"}:
            raise ContractError("fixture object must have exactly type and data", "FIXTURE_INVALID")
        constructor = constructors.get(item["type"])
        if constructor is None:
            raise ContractError("unsupported fixture object type %r" % item["type"], "FIXTURE_INVALID")
        objects.append(constructor(item["data"]))
    return objects


def validate_fixture_document(document: Mapping[str, Any]) -> List[_ContractObject]:
    """Validate one explicitly synthetic semantic counterexample."""
    _mapping(document, "fixture")
    required = {"schema", "fixture_id", "kind", "synthetic", "semantic", "objects"}
    if set(document) != required:
        raise ContractError("fixture fields differ from the public fixture schema", "FIXTURE_INVALID")
    if document["schema"] != FIXTURE_SCHEMA or document["synthetic"] is not True:
        raise ContractError("fixture must use the synthetic fixture schema", "FIXTURE_INVALID")
    _identifier(document["fixture_id"], "fixture_id")
    _text(document["kind"], "fixture kind")
    _text(document["semantic"], "fixture semantic")
    if not isinstance(document["objects"], list) or not document["objects"]:
        raise SystemExit("fixture objects are empty; PASS is forbidden")
    objects = _fixture_objects(document)
    kind = document["kind"]
    if kind == "same_direction_distinct_instances":
        fragments = [item for item in objects if isinstance(item, Fragment)]
        if len(fragments) != 2 or fragments[0].fragment_id == fragments[1].fragment_id:
            raise ContractError("same-direction fixture needs two distinct fragments", "FIXTURE_INVALID")
        if fragments[0].plane_local.normal != fragments[1].plane_local.normal:
            raise ContractError("same-direction fixture normals differ", "FIXTURE_INVALID")
        if fragments[0].plane_local.offset == fragments[1].plane_local.offset:
            raise ContractError("same-direction fixture needs distinct plane instances", "FIXTURE_INVALID")
    elif kind == "polygon_with_hole":
        if len(objects) != 1 or not isinstance(objects[0], FiniteRegion) or not objects[0].holes:
            raise ContractError("hole fixture lacks a hole", "FIXTURE_INVALID")
    elif kind == "finite_disjoint":
        surfaces = [item for item in objects if isinstance(item, Surface)]
        if len(surfaces) != 2:
            raise ContractError("finite-disjoint fixture needs two surfaces", "FIXTURE_INVALID")
        left = surfaces[0].finite_support[0].to_shapely()
        right = surfaces[1].finite_support[0].to_shapely()
        if left.intersection(right).area != 0.0:
            raise ContractError("finite-disjoint fixture supports intersect", "FIXTURE_INVALID")
    elif kind == "occluded_unknown":
        relations = [item for item in objects if isinstance(item, Relation)]
        if len(relations) != 1 or relations[0].state != "unknown" or relations[0].unknown_reason != "OCCLUDED":
            raise ContractError("occlusion fixture must remain unknown/OCCLUDED", "FIXTURE_INVALID")
    elif kind == "conflicting_hard_edits":
        events = [item for item in objects if isinstance(item, EditEvent)]
        if len(events) != 2 or not find_hard_conflicts(events):
            raise ContractError("conflict fixture does not produce a hard conflict", "FIXTURE_INVALID")
    elif kind == "protected_region":
        if len(objects) != 1 or not isinstance(objects[0], ProtectedRegion):
            raise ContractError("protection fixture needs one ProtectedRegion", "FIXTURE_INVALID")
    else:
        raise ContractError("unknown fixture kind %r" % kind, "FIXTURE_INVALID")
    return objects


def load_fixture_directory(directory: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    """Load and validate every fixture; an empty collection exits nonzero."""
    root = Path(directory)
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    if not paths:
        raise SystemExit("empty or missing fixture directory: %s" % root)
    documents = []
    identifiers = set()
    for path in paths:
        document = read_json(path)
        validate_fixture_document(document)
        if document["fixture_id"] in identifiers:
            raise ContractError("duplicate fixture ID", "DUPLICATE_ID")
        identifiers.add(document["fixture_id"])
        documents.append(document)
    return documents


def assert_negative_probe(predicate: Callable[[Any], bool], negative_probe: Any) -> None:
    """Reject a tautological predicate instead of recording a vacuous PASS."""
    if predicate(negative_probe) is not False:
        raise SystemExit("predicate accepts its falsifying probe; vacuous PASS is forbidden")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PF00 synthetic contract negative controls")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture-dir", help="validate a non-empty synthetic fixture directory")
    group.add_argument("--tautology-negative-control", action="store_true", help="must exit nonzero")
    arguments = parser.parse_args(argv)
    try:
        if arguments.tautology_negative_control:
            assert_negative_probe(lambda _probe: True, negative_probe=False)
        else:
            documents = load_fixture_directory(arguments.fixture_dir)
            print(json.dumps(
                {"fixture_count": len(documents), "status": "SYNTHETIC_CONTRACTS_VALID"},
                sort_keys=True, allow_nan=False,
            ))
        return 0
    except ContractError as exc:
        print("%s: %s" % (exc.code, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
