# -*- coding: utf-8 -*-
"""Public contracts for the observed-surface product vertical slice.

This package intentionally contains contract and serialization primitives only.
Geometry predicates, editing transactions and learned inference belong to later
work packages; synthetic fixtures are not research references.
"""

from .contracts import (
    EDIT_ACTION_ADDENDUM_ID,
    EDIT_ACTION_ADDENDUM_V2,
    EDIT_ACTIONS,
    EDIT_ACTIONS_V1,
    EDIT_ACTIONS_V2,
    EXTENDED_EDIT_ACTIONS,
    LATEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    SUPPORTED_SCHEMA_VERSIONS,
    ContractError,
    EditEvent,
    FiniteRegion,
    Fragment,
    ObservationBundle,
    PlaneLocal,
    ProjectSnapshot,
    ProtectedRegion,
    Relation,
    ReliabilityReport,
    ROI,
    ScoreBundle,
    SourceFile,
    SupportReference,
    Surface,
    UpdateResult,
    VersionConflictError,
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    check_base_version,
    contract_schema_for_action,
    edit_actions_for_schema,
    load_fixture_directory,
    migrate_event_log_to_v2,
    read_json,
    save_project,
    schema_version_for_event_actions,
    stable_id,
    validate_event_actions_for_schema,
    validate_extended_action_payload,
)

__all__ = [
    "SCHEMA_VERSION", "ContractError", "EditEvent", "FiniteRegion",
    "Fragment", "ObservationBundle", "PlaneLocal", "ProjectSnapshot",
    "ProtectedRegion", "Relation", "ReliabilityReport", "ROI",
    "ScoreBundle", "SourceFile", "SupportReference", "Surface",
    "UpdateResult", "VersionConflictError", "atomic_write_json",
    "canonical_json", "canonical_sha256", "check_base_version",
    "load_fixture_directory", "read_json", "save_project", "stable_id",
    # PF03-CONTRACT append-only exports; v1 objects above are unchanged.
    "SCHEMA_VERSION_V2", "LATEST_SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
    "EDIT_ACTION_ADDENDUM_ID", "EDIT_ACTION_ADDENDUM_V2", "EDIT_ACTIONS",
    "EDIT_ACTIONS_V1", "EDIT_ACTIONS_V2", "EXTENDED_EDIT_ACTIONS",
    "edit_actions_for_schema", "contract_schema_for_action",
    "validate_extended_action_payload", "validate_event_actions_for_schema",
    "schema_version_for_event_actions", "migrate_event_log_to_v2",
]
