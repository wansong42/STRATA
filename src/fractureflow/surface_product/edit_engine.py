# -*- coding: utf-8 -*-
"""PF03: editing transactions, dependency closure, undo/redo and stable IDs.

Design contract
---------------
* Every mutation is a two-phase transaction: ``preview`` changes nothing,
  ``commit`` re-checks the base version, the one-shot preview token and the
  input hash, then swaps the whole project in one assignment.
* Undo restores the recorded business state (geometry, relations, provenance,
  protection, parameters, lineage), not merely display state.  Redo only
  restores a byte-identical recorded result; a changed context is rejected as
  RECOMPUTE_REQUIRED instead of being silently recomputed.
* PF03 contains NO joint solver and NO learned local propagation.  The
  dependency closure enumerates the four declared dependency kinds and records
  reason paths; it does not claim a solved energy trace (that is PF04 scope).
* PF03-CONTRACT closed the recorded contract gap: DELETE_OBJECT / UNDO / REDO
  are now public v2 edit actions from ``contracts.EDIT_ACTIONS_V2``.  Every
  engine event -- including these three -- carries a validated public
  ``EditEvent`` and ``contract_action=True``.  The v1 ten-action set and the
  meaning of already-recorded events are unchanged, and v1 bundles that still
  hold pre-addendum engine-level events remain readable (read-only legacy path).
* Undo/redo each publish a NEW project version; they never roll a version number
  back.  A redo whose context changed is still rejected as RECOMPUTE_REQUIRED.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .contracts import (
    EDIT_ACTIONS_V2,
    EXTENDED_EDIT_ACTIONS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    ContractError,
    EditEvent,
    ProjectSnapshot,
    VersionConflictError,
    canonical_sha256,
    canonical_sha256_validated,
    check_base_version,
    find_hard_conflicts,
    project_identity_digest,
    schema_version_for_event_actions,
    validate_event_actions_for_schema,
)
from .observed_surface_graph import (
    LINEAGE_KEY,
    SOLVER_GROUPS_KEY,
    assert_invariants,
    business_payload,
    check_invariants,
    lineage_links,
    make_id,
    normalise_order,
    reference_index,
    retired_ids,
)
from .support_geometry import (
    MIN_SUPPORT_POINTS,
    SupportGeometryError,
    coplanarity_report,
    fit_support_geometry,
    fixture_points_for_fragment,
    frame_of as support_frame_of,
    normalise_frame as normalise_support_frame,
    region_metrics,
    transform_region,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64

# PF03-CONTRACT: these are now public v2 contract actions, not engine-only ones.
CONTRACT_ADDENDUM_ACTIONS = EXTENDED_EDIT_ACTIONS
ENGINE_ONLY_ACTIONS = CONTRACT_ADDENDUM_ACTIONS  # legacy alias; value unchanged, meaning is now public
SUPPORTED_ACTIONS = frozenset(EDIT_ACTIONS_V2)
# These actions create a brand new object; their target ID is not yet in the
# graph and must not be treated as a dangling reference.
NEW_OBJECT_ACTIONS = frozenset({"PROTECT_REGION"})
EDIT_EVENT_FIELDS = tuple(item.name for item in EditEvent.__dataclass_fields__.values())
# W151: real support-point provider.  It receives (fragment_id, observation_id,
# original_point_indices) and returns local 3D points or None.  A production
# engine is always constructed with a provider; without one the engine stays on
# the explicit legacy fixture path and labels that path in every warning.
SupportPointProvider = Callable[[str, str, Sequence[int]], Optional[Sequence[Sequence[float]]]]
FragmentFrameProvider = Callable[[str], Optional[Mapping[str, Any]]]

COLLECTION_ID_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("observations", "observation_id"),
    ("fragments", "fragment_id"),
    ("surfaces", "surface_id"),
    ("relations", "relation_id"),
    ("protected_regions", "region_id"),
)


class TransactionError(ContractError):
    def __init__(self, message: str, code: str = "TRANSACTION_REJECTED") -> None:
        super().__init__(message, code)


class PreviewTokenError(TransactionError):
    def __init__(self, reason: str) -> None:
        super().__init__("preview token rejected: %s" % reason, "PREVIEW_TOKEN_INVALID")


class PartialWriteError(TransactionError):
    def __init__(self, stage: str) -> None:
        super().__init__("injected partial-write failure at %s; state unchanged" % stage, "PARTIAL_WRITE")


class RecomputeRequiredError(TransactionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "RECOMPUTE_REQUIRED")


@dataclass(frozen=True)
class EditDraft:
    action: str
    target_ids: List[str]
    payload: Dict[str, Any]
    base_version: int
    actor: str = "pf03_synthetic_actor"
    hardness: str = "hard"
    tolerance: float = 0.0
    parent_event_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(frozen=True)
class Preview:
    token: str
    base_version: int
    input_hash: str
    event: Dict[str, Any]
    dependency: Dict[str, Any]
    diff: Dict[str, Any]
    candidate_business: Dict[str, Any]
    candidate_project: ProjectSnapshot
    events_new: List[Dict[str, Any]]
    undo_new: List[Dict[str, Any]]
    redo_new: List[Dict[str, Any]]
    created_at: str
    contract_action: bool
    candidate_business_hash: str = ""
    # W151: fitted plane/boundary/residual preview, never the raw candidate
    # project.  The UI renders this before commit; an empty object means the
    # action has no geometry consequence.
    geometry_preview: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable preview receipt; never contains the raw candidate state."""
        return {
            "token": self.token,
            "base_version": self.base_version,
            "input_hash": self.input_hash,
            "event_id": self.event["event_id"],
            "action": self.event["action"],
            "target_ids": list(self.event["target_ids"]),
            "dependency": copy.deepcopy(self.dependency),
            "diff": copy.deepcopy(self.diff),
            "created_at": self.created_at,
            "contract_action": self.contract_action,
            # Computed once by EditEngine.preview from the same candidate tree
            # as after_hash; recomputing here walked the 724 kB business tree a
            # second time per preview (measured 0.17 s of the 3.5 s compute).
            "candidate_business_hash": self.candidate_business_hash or canonical_sha256(self.candidate_business),
            "geometry_preview": copy.deepcopy(self.geometry_preview),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TransactionResult:
    accepted: bool
    project_version: int
    event_id: str
    changed_ids: List[str]
    affected_ids: List[str]
    reason_codes: List[str]
    reason_paths: List[Dict[str, Any]]
    warnings: List[str]
    diff: Dict[str, Any]
    content_hash: str
    contract_action: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "project_version": self.project_version,
            "event_id": self.event_id,
            "changed_ids": list(self.changed_ids),
            "affected_ids": list(self.affected_ids),
            "reason_codes": list(self.reason_codes),
            "reason_paths": copy.deepcopy(self.reason_paths),
            "warnings": list(self.warnings),
            "diff": copy.deepcopy(self.diff),
            "content_hash": self.content_hash,
            "contract_action": self.contract_action,
        }


@dataclass(frozen=True)
class UndoResult:
    kind: str
    project_version: int
    event_id: str
    restored_semantic_hash: str
    identity_restored: bool
    recomputed: bool
    warnings: List[str]
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "project_version": self.project_version,
            "event_id": self.event_id,
            "restored_semantic_hash": self.restored_semantic_hash,
            "identity_restored": self.identity_restored,
            "recomputed": self.recomputed,
            "warnings": list(self.warnings),
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Self-built synthetic contract project (never a research reference)
# ---------------------------------------------------------------------------

def _region(offset: float = 0.0) -> Dict[str, Any]:
    return {
        "exterior": [
            [offset, 0.0], [offset + 1.0, 0.0], [offset + 1.0, 1.0],
            [offset, 1.0], [offset, 0.0],
        ],
        "holes": [],
    }


def _support(roi_id: str, indices: List[int]) -> Dict[str, Any]:
    return {
        "observation_id": "obs_main",
        "source_sha256": HASH_A,
        "storage_uri": "synthetic://obs_main/support",
        "index_kind": "original_point_index",
        "indices": list(indices),
        "roi_id": roi_id,
        "decimation_lineage": {},
    }


def synthetic_fragment(
    fragment_id: str,
    offset: float = 0.0,
    roi_id: str = "roi_main",
    indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """A public, explicitly synthetic Fragment payload for the contract project."""
    return {
        "fragment_id": fragment_id,
        "observation_id": "obs_main",
        "support_ref": _support(roi_id, indices or [0, 1, 2]),
        "plane_local": {"normal": [0.0, 0.0, 1.0], "offset": offset},
        "finite_regions_local": [_region(offset)],
        "residuals": [0.0, 0.1, -0.1],
        "quality": {"rmse": 0.08165, "synthetic": True},
        "candidate_generator": {"candidate_batch_id": "batch_main", "generator_config_hash": HASH_C},
        "version": 1,
    }


def synthetic_surface(surface_id: str, member_ids: Sequence[str], offset: float = 0.0) -> Dict[str, Any]:
    return {
        "surface_id": surface_id,
        "member_fragment_ids": list(member_ids),
        "interpretation_status": "synthetic_contract_case",
        "finite_support": [_region(offset)],
        "provenance": "DIRECT_OBSERVATION",
        "version": 1,
    }


def synthetic_relation(
    relation_id: str,
    src_id: str,
    dst_id: str,
    roi_id: str = "roi_main",
    state: str = "unknown",
    unknown_reason: Optional[str] = "OCCLUDED",
) -> Dict[str, Any]:
    return {
        "relation_id": relation_id,
        "src_id": src_id,
        "dst_id": dst_id,
        "relation_type": "intersects_in_roi",
        "state": state,
        "unknown_reason": unknown_reason if state == "unknown" else None,
        "roi_id": roi_id,
        "evidence_refs": ["ev_view_01"],
        "provenance": "DIRECT_OBSERVATION",
        "version": 1,
    }


def build_synthetic_project() -> ProjectSnapshot:
    """Return a non-empty, self-built contract project with four dependency kinds."""
    observation = {
        # PF03-CONTRACT: newly built synthetic projects declare the v2 public
        # edit-action catalog.  Existing v1 bundles remain readable unchanged.
        "schema_version": SCHEMA_VERSION_V2,
        "observation_id": "obs_main",
        "source_files": [{"path": "synthetic://obs_main/points", "sha256": HASH_A,
                          "byte_count": 12, "media_type": "application/json"}],
        "sha256": HASH_B,
        "unit": "m",
        "coordinate_frame": "synthetic_local",
        "local_origin": [0.0, 0.0, 0.0],
        "roi": [
            {"roi_id": "roi_main", "unit": "m", "coordinate_frame": "synthetic_local",
             "bounds_min": [-10.0, -10.0, -10.0], "bounds_max": [10.0, 10.0, 10.0]},
            {"roi_id": "roi_far", "unit": "m", "coordinate_frame": "synthetic_local",
             "bounds_min": [0.0, 0.0, 0.0], "bounds_max": [10.0, 10.0, 10.0]},
        ],
        "coverage": {"point_count": 12, "synthetic": True},
        "registration": {},
        "acquisition": {"method": "explicit_synthetic_pf03_contract_project"},
        "license": "SYNTHETIC_CONTRACT_ONLY",
    }
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V2,
        "project_id": "proj_pf03",
        "project_version": 1,
        "observations": [observation],
        "fragments": [
            synthetic_fragment("frag_a", 0.0, "roi_main", [0, 1, 2]),
            synthetic_fragment("frag_b", 2.0, "roi_main", [2, 3, 4]),
            synthetic_fragment("frag_c", 4.0, "roi_main", [5, 6, 7]),
            synthetic_fragment("frag_d", 8.0, "roi_far", [8, 9, 10]),
        ],
        "surfaces": [
            synthetic_surface("surf_a", ["frag_a", "frag_b"], 0.0),
            synthetic_surface("surf_b", ["frag_c"], 4.0),
            synthetic_surface("surf_c", ["frag_d"], 8.0),
        ],
        "relations": [
            synthetic_relation("rel_ab", "surf_a", "surf_b"),
            synthetic_relation("rel_bc", "surf_b", "surf_c", "roi_far"),
            synthetic_relation("rel_ac", "surf_a", "surf_c", "roi_far", "yes", None),
        ],
        "event_head": None,
        "parameters": {"project_mode": "manual", "solver": "PF03_TRANSACTION_ONLY"},
        "protected_regions": [
            {"region_id": "protect_linked", "roi_id": "roi_main", "finite_regions": [_region(0.25)]},
            {"region_id": "protect_far", "roi_id": "roi_far", "finite_regions": [_region(8.25)]},
        ],
        "artifacts": {
            "synthetic": True,
            "scope": "PF03 self-built synthetic contract project; not a research reference",
            LINEAGE_KEY: {"links": [], "retired_ids": []},
            SOLVER_GROUPS_KEY: [
                {"group_id": "solver_linked", "object_ids": ["surf_a", "protect_linked", "rel_ac"]},
            ],
            "event_log": [],
            "undo_stack": [],
            "redo_stack": [],
        },
    }
    payload["content_hash"] = project_identity_digest(payload)
    project = ProjectSnapshot.from_dict(payload)
    assert_invariants(project)
    return project


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic_hash(business: Mapping[str, Any]) -> str:
    """Hash of business state excluding versioning metadata.

    ``project_version``, ``event_head`` and ``schema_version`` are versioning
    metadata, not geometry/relation/provenance/protection/parameter/lineage
    state.  A v1->v2 addendum marker must therefore not masquerade as a changed
    business context and trigger RECOMPUTE_REQUIRED; real semantic changes in
    every other field are still detected.
    """
    payload = copy.deepcopy(dict(business))
    payload.pop("project_version", None)
    payload.pop("event_head", None)
    payload.pop("schema_version", None)
    return canonical_sha256_validated(payload)


def _id_map(business: Mapping[str, Any], collection: str, id_field: str) -> Dict[str, str]:
    return {item[id_field]: canonical_sha256_validated(item) for item in business[collection]}


def _make_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    added: List[str] = []
    removed: List[str] = []
    changed: List[str] = []
    relations_changed: List[str] = []
    unknown_changed: List[str] = []
    geometry_changed: List[str] = []
    for collection, id_field in COLLECTION_ID_FIELDS:
        left, right = _id_map(before, collection, id_field), _id_map(after, collection, id_field)
        added.extend(sorted(set(right) - set(left)))
        removed.extend(sorted(set(left) - set(right)))
        changed.extend(sorted(key for key in set(left) & set(right) if left[key] != right[key]))
        if collection in ("fragments", "surfaces"):
            geometry_changed.extend(sorted(key for key in set(left) & set(right) if left[key] != right[key]))
        if collection == "relations":
            relations_changed.extend(sorted(key for key in set(left) & set(right) if left[key] != right[key]))
            states_before = {item["relation_id"]: item["state"] for item in before["relations"]}
            states_after = {item["relation_id"]: item["state"] for item in after["relations"]}
            unknown_changed.extend(sorted(
                key for key in set(states_before) & set(states_after)
                if states_before[key] != states_after[key]
            ))
    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
        "geometry_changed": sorted(geometry_changed),
        "relations_changed": sorted(relations_changed),
        "unknown_changed": sorted(unknown_changed),
        "before_hash": canonical_sha256_validated(before),
        "after_hash": canonical_sha256_validated(after),
        "enumerated_collections": len(COLLECTION_ID_FIELDS),
    }


def _bump(item: Dict[str, Any]) -> None:
    item["version"] = int(item["version"]) + 1


def _find(business: Dict[str, Any], object_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    for collection, id_field in COLLECTION_ID_FIELDS:
        for item in business[collection]:
            if item[id_field] == object_id:
                return collection, item
    return None, None


# ---------------------------------------------------------------------------
# Edit engine
# ---------------------------------------------------------------------------

class EditEngine:
    """Single-writer, versioned transaction engine over one ProjectSnapshot."""

    def __init__(
        self,
        project: ProjectSnapshot,
        *,
        clock: Callable[[], str] = _now_iso,
        preview_ttl_seconds: float = 900.0,
        events: Optional[Sequence[Mapping[str, Any]]] = None,
        undo_stack: Optional[Sequence[Mapping[str, Any]]] = None,
        redo_stack: Optional[Sequence[Mapping[str, Any]]] = None,
        support_points: Optional[SupportPointProvider] = None,
        fragment_frames: Optional[FragmentFrameProvider] = None,
        geometry_options: Optional[Mapping[str, Any]] = None,
        real_geometry_required: bool = False,
    ) -> None:
        assert_invariants(project)
        self.project = project
        self.events: List[Dict[str, Any]] = [dict(item) for item in (events or [])]
        self.undo_stack: List[Dict[str, Any]] = [dict(item) for item in (undo_stack or [])]
        self.redo_stack: List[Dict[str, Any]] = [dict(item) for item in (redo_stack or [])]
        self._clock = clock
        self._ttl = float(preview_ttl_seconds)
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._trace: List[Dict[str, Any]] = []
        self._support_points = support_points
        self._fragment_frames = fragment_frames
        self._geometry_options = dict(geometry_options or {})
        self._real_geometry = support_points is not None
        # T-D: a project whose provenance is a real measurement must never fall
        # back to the labelled fixture reconstruction when the original point
        # source is missing.  Without this flag the fixture branch silently
        # updated support indices only and kept the previous plane/regions.
        self._real_geometry_required = bool(real_geometry_required)

    # -- construction / reload ------------------------------------------------

    @classmethod
    def from_project(cls, project: ProjectSnapshot, **kwargs: Any) -> "EditEngine":
        """Reload the engine bookkeeping from a saved/read-back project."""
        artifacts = dict(project.artifacts or {})
        events = artifacts.get("event_log") or []
        undo = artifacts.get("undo_stack") or []
        redo = artifacts.get("redo_stack") or []
        if not isinstance(events, list) or not isinstance(undo, list) or not isinstance(redo, list):
            raise TransactionError("saved engine bookkeeping has the wrong type", "RELOAD_INVALID")
        catalog = validate_event_actions_for_schema(events, project.schema_version)
        if not catalog["ok"]:
            raise TransactionError(
                "reloaded event log violates the %s public action catalog: %s"
                % (project.schema_version, catalog["rejected"]), "RELOAD_INVALID",
            )
        engine = cls(project, events=events, undo_stack=undo, redo_stack=redo, **kwargs)
        if events and project.event_head != events[-1].get("event_id"):
            raise TransactionError("event_head does not match the reloaded event log", "RELOAD_INVALID")
        return engine

    def sorted_view(self) -> ProjectSnapshot:
        return normalise_order(self.project)

    def operation_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._trace)

    # -- assembly -------------------------------------------------------------

    def _assemble(
        self,
        business: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        undo_stack: Sequence[Mapping[str, Any]],
        redo_stack: Sequence[Mapping[str, Any]],
    ) -> ProjectSnapshot:
        payload = copy.deepcopy(dict(business))
        payload.pop("content_hash", None)
        # A log containing a v2 action declares the v2 project schema; an
        # all-v1 log keeps v1 so that existing PF03 bundles stay byte-stable.
        derived = schema_version_for_event_actions(events)
        current = business.get("schema_version", SCHEMA_VERSION)
        payload["schema_version"] = (
            SCHEMA_VERSION_V2 if SCHEMA_VERSION_V2 in (derived, current) else SCHEMA_VERSION
        )
        artifacts = dict(payload.get("artifacts") or {})
        for key in ("event_log", "undo_stack", "redo_stack"):
            artifacts.pop(key, None)
        artifacts["event_log"] = copy.deepcopy(list(events))
        artifacts["undo_stack"] = copy.deepcopy(list(undo_stack))
        artifacts["redo_stack"] = copy.deepcopy(list(redo_stack))
        payload["artifacts"] = artifacts
        digest = project_identity_digest(payload)
        payload["content_hash"] = digest
        return ProjectSnapshot.from_dict(payload)

    def _validate_candidate(self, candidate: ProjectSnapshot) -> None:
        report = check_invariants(candidate)
        if not report.ok:
            raise TransactionError(
                "candidate violates structural invariants: %s" % [item["code"] for item in report.issues],
                "INVARIANT_VIOLATION",
            )

    # -- dependency closure ---------------------------------------------------

    def _dependency_closure(
        self, project: ProjectSnapshot, target_ids: Sequence[str]
    ) -> Dict[str, Any]:
        """Enumerate the four declared dependency kinds and record reason paths."""
        payload = project.to_dict()
        index = reference_index(project)
        known = set(index["all_ids"]) | set(retired_ids(project))
        adjacency: Dict[str, List[Tuple[str, str]]] = {}

        def link(left: str, right: str, kind: str) -> None:
            if left == right or left not in known or right not in known:
                return
            adjacency.setdefault(left, []).append((right, kind))
            adjacency.setdefault(right, []).append((left, kind))

        fragments = payload["fragments"]
        for left, right in combinations(fragments, 2):
            left_support, right_support = left["support_ref"], right["support_ref"]
            shared = set(left_support["indices"]) & set(right_support["indices"])
            if shared:
                link(left["fragment_id"], right["fragment_id"], "SUPPORT_SHARED_POINTS")
            if (left_support["storage_uri"] == right_support["storage_uri"]
                    and left_support["source_sha256"] == right_support["source_sha256"]):
                link(left["fragment_id"], right["fragment_id"], "SUPPORT_STORAGE")
        for surface in payload["surfaces"]:
            surface_id = surface["surface_id"]
            members = surface["member_fragment_ids"]
            for member in members:
                link(surface_id, member, "INSTANCE_MEMBER")
            for left, right in combinations(members, 2):
                link(left, right, "INSTANCE_SIBLING")
        for entry in lineage_links(project):
            for parent in entry.get("parent_ids") or []:
                for child in entry.get("child_ids") or []:
                    link(parent, child, "LINEAGE_%s" % str(entry.get("kind")).upper())
        for relation in payload["relations"]:
            relation_id = relation["relation_id"]
            link(relation["src_id"], relation["dst_id"], "RELATION")
            link(relation_id, relation["src_id"], "RELATION_ENDPOINT")
            link(relation_id, relation["dst_id"], "RELATION_ENDPOINT")
        for group in (payload["artifacts"] or {}).get(SOLVER_GROUPS_KEY) or []:
            for left, right in combinations(group.get("object_ids") or [], 2):
                link(left, right, "SOLVER_GROUP")

        visited: Set[str] = set()
        paths: Dict[str, List[str]] = {}
        queue: List[Tuple[str, List[str]]] = [(target, [target]) for target in target_ids]
        for target, path in queue:
            if target in visited:
                continue
            visited.add(target)
            paths[target] = path
            for neighbour, kind in adjacency.get(target, []):
                if neighbour not in visited:
                    queue.append((neighbour, path + [neighbour]))
        closure = sorted(visited & known)
        kinds = sorted({kind for node in visited for _, kind in adjacency.get(node, [])})
        reason_paths = [
            {
                "object_id": node,
                "path": paths[node],
                "hop_count": len(paths[node]) - 1,
                "dependency_kinds": sorted({kind for _, kind in adjacency.get(node, []) if kind}),
            }
            for node in sorted(visited & known)
        ]
        return {
            "closure": closure,
            "reason_paths": reason_paths,
            "dependency_kinds": kinds,
            "target_ids": list(target_ids),
            "counts": {
                "enumerated_graphs": len(payload["fragments"]) + len(payload["surfaces"]),
                "enumerated_relations": len(payload["relations"]),
                "enumerated_solver_groups": len((payload["artifacts"] or {}).get(SOLVER_GROUPS_KEY) or []),
                "enumerated_lineage_links": len(lineage_links(project)),
                "closure_size": len(closure),
                "reason_paths": len(reason_paths),
            },
            "denominator": len(closure),
        }

    # -- W151: real support-point geometry -----------------------------------

    def _fit_options(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        options = dict(self._geometry_options)
        fit_body = body.get("fit")
        if isinstance(fit_body, Mapping):
            options.update(dict(fit_body))
        return options

    def _resolve_fragment_points(
        self, fragment: Dict[str, Any], indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, str]:
        """Resolve exactly the selected original indices, or refuse.

        A production engine has a provider; a provider returning ``None`` is a
        hard refusal (missing source), never a synthetic fallback.  Without a
        provider the engine is on the explicit legacy fixture path and labels
        the reconstructed geometry.
        """
        chosen = [int(item) for item in (indices if indices is not None
                                          else fragment["support_ref"]["indices"])]
        if not chosen:
            raise SupportGeometryError("support selection is empty", "INSUFFICIENT_SUPPORT")
        if self._support_points is not None:
            result = self._support_points(
                str(fragment["fragment_id"]), str(fragment["observation_id"]), chosen,
            )
            if result is None:
                raise SupportGeometryError(
                    "original support points are unavailable for fragment %s" % fragment["fragment_id"],
                    "SUPPORT_POINTS_UNAVAILABLE",
                    fragment_id=fragment["fragment_id"], indices=chosen[:20],
                )
            array = np.asarray(result, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] != len(chosen):
                raise SupportGeometryError(
                    "support provider returned %s points for %d requested indices"
                    % (getattr(array, "shape", None), len(chosen)),
                    "SUPPORT_POINTS_UNAVAILABLE",
                )
            return array, "ORIGINAL_POINT_INDEX"
        if self._real_geometry_required:
            raise SupportGeometryError(
                "this project declares a real measurement but the engine has no original "
                "point provider; the fixture reconstruction path is refused",
                "SUPPORT_POINTS_UNAVAILABLE", fragment_id=fragment["fragment_id"],
            )
        base = fixture_points_for_fragment(fragment)
        if len(base) == 0:
            raise SupportGeometryError("fixture reconstructs no support points", "SUPPORT_POINTS_UNAVAILABLE")
        picked = np.asarray([base[index % len(base)] for index in chosen], dtype=np.float64)
        return picked, "SYNTHETIC_FIXTURE_RECONSTRUCTION"

    def _resolve_selection_points(
        self, observation_id: str, indices: Sequence[int],
    ) -> Tuple[np.ndarray, str]:
        chosen = [int(item) for item in indices]
        if not chosen:
            raise SupportGeometryError("support selection is empty", "INSUFFICIENT_SUPPORT")
        if self._support_points is None:
            raise SupportGeometryError(
                "creating a face from a support selection requires an original point source; "
                "this engine has none",
                "SUPPORT_POINTS_UNAVAILABLE", observation_id=observation_id,
            )
        result = self._support_points("", observation_id, chosen)
        if result is None:
            raise SupportGeometryError(
                "original support points are unavailable for observation %s" % observation_id,
                "SUPPORT_POINTS_UNAVAILABLE", observation_id=observation_id,
            )
        array = np.asarray(result, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] != len(chosen):
            raise SupportGeometryError(
                "support provider returned %s points for %d requested indices"
                % (getattr(array, "shape", None), len(chosen)),
                "SUPPORT_POINTS_UNAVAILABLE",
            )
        return array, "ORIGINAL_POINT_INDEX"

    def _frame_for_fragment(self, fragment: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Return the frame in which this fragment's local regions are expressed.

        The second value records whether the frame was declared by the fragment
        itself or by a provider.  An inferred frame is not silently used to move
        another fragment's geometry.

        T-D: the fragment's own persisted ``quality.frame_local`` wins over the
        provider.  The provider is bound when the engine is created, so inside a
        transaction that just refitted this fragment its answer is the *previous*
        frame; the working fragment already carries the new one.  (Measured on the
        real saved project: provider frame != latest frame after a SELECT_SUPPORT
        commit.)  The provider is still the fallback for fragments that never
        stored a frame.
        """
        fid = str(fragment["fragment_id"])
        quality = dict(fragment.get("quality") or {})
        stored = quality.get("frame_local")
        if isinstance(stored, Mapping) and stored:
            return normalise_support_frame(stored), True
        if self._fragment_frames is not None:
            provided = self._fragment_frames(fid)
            if provided:
                return normalise_support_frame(provided), True
        plane = fragment["plane_local"]
        return support_frame_of(plane["normal"], float(plane["offset"])), False

    def _fragment_fit(
        self, fragment: Dict[str, Any], indices: Sequence[int], body: Mapping[str, Any],
        *, label: str,
    ) -> Tuple[np.ndarray, Any, str]:
        points, source = self._resolve_fragment_points(fragment, indices)
        fit = fit_support_geometry(
            points, options=self._fit_options(body),
            previous_normal=fragment["plane_local"]["normal"],
        )
        return points, fit, source

    def _apply_fit_to_fragment(
        self, fragment: Dict[str, Any], indices: Sequence[int], fit: Any,
        source: str, event_id: str, label: str,
    ) -> Dict[str, Any]:
        metrics = region_metrics(fit.regions)
        fragment["support_ref"]["indices"] = [int(item) for item in indices]
        fragment["plane_local"] = {"normal": list(fit.normal), "offset": float(fit.offset)}
        fragment["finite_regions_local"] = copy.deepcopy(fit.regions)
        fragment["residuals"] = [float(item) for item in fit.residuals]
        quality = dict(fragment.get("quality") or {})
        quality.update({
            "support_point_count": int(fit.support_point_count),
            "rmse": float(fit.rmse),
            "max_abs_residual": float(fit.max_abs_residual),
            "plane_tolerance": float(fit.plane_tolerance),
            "finite_region_count": int(metrics["region_count"]),
            "hole_count": int(metrics["hole_count"]),
            "total_area_local": float(metrics["total_area"]),
            "region_areas": [float(item) for item in metrics["region_areas"]],
            "fit_method": "svd_plane_plus_alpha_shape",
            "fit_config_hash": fit.fit_config["config_hash"],
            "frame_local": fit.frame(),
            "geometry_source": source,
            "requires_human_review": True,
            "refit_event_id": event_id,
            "refit_action": label,
            "synthetic": source.startswith("SYNTHETIC"),
        })
        fragment["quality"] = quality
        candidate = dict(fragment.get("candidate_generator") or {})
        candidate["support_refit"] = {
            "event_id": event_id,
            "fit_config_hash": fit.fit_config["config_hash"],
            "geometry_source": source,
            "label": label,
        }
        fragment["candidate_generator"] = candidate
        return metrics

    def _fit_preview(self, fit: Any, source: str, label: str) -> Dict[str, Any]:
        metrics = region_metrics(fit.regions)
        return {
            "label": label,
            "geometry_source": source,
            "support_point_count": int(fit.support_point_count),
            "plane_local": {"normal": list(fit.normal), "offset": float(fit.offset)},
            "frame_local": fit.frame(),
            "residuals": {
                "rmse": float(fit.rmse),
                "max_abs_residual": float(fit.max_abs_residual),
                "plane_tolerance": float(fit.plane_tolerance),
                "residual_count": int(fit.residual_count),
            },
            "regions": copy.deepcopy(fit.regions),
            "region_metrics": metrics,
            "fit_config": dict(fit.fit_config),
            "requires_human_review": True,
        }

    def _surface_member_dicts(self, working: Mapping[str, Any], members: Sequence[str]) -> List[Dict[str, Any]]:
        index = {item["fragment_id"]: item for item in working["fragments"]}
        result = []
        for member in members:
            if member not in index:
                raise TransactionError("surface references a missing fragment: %s" % member, "BAD_TARGET")
            result.append(index[member])
        return result

    def _fixture_surface_regions(
        self, working: Mapping[str, Any], members: Sequence[str], *, event_id: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Legacy fixture-only member-region union.

        Used only when no original point provider exists.  It never invents a
        synthetic plane; it concatenates the members' own stored regions and is
        labelled as a fixture path in the event warnings.
        """
        fragments = self._surface_member_dicts(working, members)
        regions: List[Dict[str, Any]] = []
        for fragment in fragments:
            regions.extend(copy.deepcopy(fragment["finite_regions_local"]))
        if not regions:
            raise SupportGeometryError("fixture surface finite support is empty", "FINITE_DOMAIN_EMPTY")
        metrics = region_metrics(regions)
        metadata = {
            "frame_local": None,
            "member_count": len(fragments),
            "region_metrics": metrics,
            "components": [{
                "fragment_id": fragment["fragment_id"],
                "frame_local": None,
                "source_region_count": len(fragment["finite_regions_local"]),
                "transformed_to_target": False,
            } for fragment in fragments],
            "policy": "SYNTHETIC_FIXTURE_MEMBER_REGIONS_WITHOUT_COMMON_FRAME",
            "event_id": event_id,
        }
        return regions, metadata

    def _surface_regions_from_members(
        self, working: Mapping[str, Any], members: Sequence[str], *, body: Mapping[str, Any],
        event_id: str, error_code: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        fragments = self._surface_member_dicts(working, members)
        if not fragments:
            raise TransactionError("surface has no member fragment", "BAD_TARGET")
        frames: List[Dict[str, Any]] = []
        declared: List[bool] = []
        for fragment in fragments:
            frame, is_declared = self._frame_for_fragment(fragment)
            frames.append(frame)
            declared.append(is_declared)
        if len(fragments) > 1:
            for position in range(1, len(fragments)):
                if not declared[position]:
                    raise SupportGeometryError(
                        "cannot place member %s in a common support plane: its local frame is not declared"
                        % fragments[position]["fragment_id"],
                        "SUPPORT_FRAME_UNAVAILABLE", fragment_id=fragments[position]["fragment_id"],
                    )
            angle_tolerance = float(body.get("coplanar_angle_deg", 1.0))
            offset_tolerance = body.get("coplanar_offset_tolerance")
            report = coplanarity_report(
                frames, angle_tolerance_deg=angle_tolerance,
                offset_tolerance=(float(offset_tolerance) if offset_tolerance is not None else None),
            )
            if not report["ok"]:
                raise SupportGeometryError(
                    "surface members are not coplanar; refusing a single-plane finite support",
                    error_code, **report,
                )
        else:
            report = coplanarity_report(frames)
        target = frames[0]
        regions: List[Dict[str, Any]] = []
        for position, fragment in enumerate(fragments):
            for region in fragment["finite_regions_local"]:
                if position == 0:
                    regions.append(copy.deepcopy(region))
                else:
                    regions.append(transform_region(region, frames[position], target))
        if not regions:
            raise SupportGeometryError("surface finite support is empty", "FINITE_DOMAIN_EMPTY")
        metrics = region_metrics(regions)
        metadata = {
            "frame_local": target,
            "coplanarity": report,
            "member_count": len(fragments),
            "region_metrics": metrics,
            "components": [
                {
                    "fragment_id": fragment["fragment_id"],
                    "frame_local": frames[position],
                    "source_region_count": len(fragment["finite_regions_local"]),
                    "transformed_to_target": position != 0,
                }
                for position, fragment in enumerate(fragments)
            ],
            "policy": "REAL_SUPPORT_FINITE_REGIONS_NO_CONVEX_FILL",
            "event_id": event_id,
        }
        return regions, metadata

    def _store_surface_geometry(
        self, working: Dict[str, Any], surface_id: str, metadata: Mapping[str, Any], mode: str,
    ) -> None:
        artifacts = working.setdefault("artifacts", {})
        store = artifacts.get("surface_geometry")
        if not isinstance(store, dict):
            store = {}
            artifacts["surface_geometry"] = store
        entry = copy.deepcopy(dict(metadata))
        entry["surface_id"] = surface_id
        entry["mode"] = mode
        entry["requires_human_review"] = True
        store[surface_id] = entry

    def _rebuild_surface_geometry(
        self, working: Dict[str, Any], surface_id: str, *, body: Mapping[str, Any],
        event_id: str, error_code: str, mode: str,
    ) -> Dict[str, Any]:
        surface = None
        for item in working["surfaces"]:
            if item["surface_id"] == surface_id:
                surface = item
                break
        if surface is None:
            raise TransactionError("surface is missing: %s" % surface_id, "BAD_TARGET")
        regions, metadata = self._surface_regions_from_members(
            working, surface["member_fragment_ids"], body=body, event_id=event_id, error_code=error_code,
        )
        surface["finite_support"] = copy.deepcopy(regions)
        _bump(surface)
        self._store_surface_geometry(working, surface_id, metadata, mode)
        return {"surface_id": surface_id, "regions": regions, "metadata": metadata}

    def _assert_hypothetical_coplanarity(
        self, working: Mapping[str, Any], fragment_id: str, new_frame: Mapping[str, Any],
        *, body: Mapping[str, Any], error_code: str,
    ) -> None:
        for surface in working["surfaces"]:
            members = list(surface["member_fragment_ids"])
            if fragment_id not in members or len(members) <= 1:
                continue
            frames: List[Dict[str, Any]] = []
            for member in members:
                if member == fragment_id:
                    frames.append(normalise_support_frame(new_frame))
                else:
                    fragment = next(item for item in working["fragments"] if item["fragment_id"] == member)
                    frame, declared = self._frame_for_fragment(fragment)
                    if not declared:
                        raise SupportGeometryError(
                            "cannot verify coplanarity for member %s: local frame is not declared" % member,
                            "SUPPORT_FRAME_UNAVAILABLE", fragment_id=member,
                        )
                    frames.append(frame)
            report = coplanarity_report(
                frames,
                angle_tolerance_deg=float(body.get("coplanar_angle_deg", 1.0)),
                offset_tolerance=(float(body["coplanar_offset_tolerance"])
                                  if body.get("coplanar_offset_tolerance") is not None else None),
            )
            if not report["ok"]:
                raise SupportGeometryError(
                    "the new support geometry would break surface %s coplanarity" % surface["surface_id"],
                    error_code, surface_id=surface["surface_id"], **report,
                )

    def _geometry_preview_for_ids(self, working: Mapping[str, Any], ids: Sequence[str],
                                  *, label: str) -> Dict[str, Any]:
        fragments = {item["fragment_id"]: item for item in working["fragments"]}
        surfaces = {item["surface_id"]: item for item in working["surfaces"]}
        objects: Dict[str, Any] = {}
        for object_id in ids:
            fragment = fragments.get(object_id)
            if fragment is not None:
                metrics = region_metrics(fragment["finite_regions_local"])
                objects[object_id] = {
                    "kind": "fragment",
                    "support_count": len(fragment["support_ref"]["indices"]),
                    "plane_local": copy.deepcopy(fragment["plane_local"]),
                    "regions": copy.deepcopy(fragment["finite_regions_local"]),
                    "region_metrics": metrics,
                    "quality": {
                        key: fragment["quality"].get(key)
                        for key in ("rmse", "max_abs_residual", "plane_tolerance", "geometry_source",
                                    "requires_human_review", "fit_method", "frame_local")
                    },
                }
                continue
            surface = surfaces.get(object_id)
            if surface is not None:
                metrics = region_metrics(surface["finite_support"])
                objects[object_id] = {
                    "kind": "surface",
                    "members": list(surface["member_fragment_ids"]),
                    "finite_support": copy.deepcopy(surface["finite_support"]),
                    "region_metrics": metrics,
                }
        return {"label": label, "objects": objects,
                "policy": "REAL_SUPPORT_GEOMETRY_ONLY_NO_SYNTHETIC_PLANE"}

    def _lineage_link(self, kind: str, parents: Sequence[str], children: Sequence[str],
                      event_id: str, note: Optional[str] = None) -> Dict[str, Any]:
        link = {"kind": kind, "parent_ids": list(parents), "child_ids": list(children),
                "event_id": event_id}
        if note:
            link["note"] = note
        return link

    # -- action application ---------------------------------------------------

    def _apply_action(
        self,
        working: Dict[str, Any],
        event: Dict[str, Any],
        index: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = event["action"]
        targets = list(event["target_ids"])
        body = dict(event["payload"])
        event_id = event["event_id"]
        changed: List[str] = []
        warnings: List[str] = []
        reason_codes: List[str] = ["ACTION_%s" % action]
        new_links: List[Dict[str, Any]] = []
        newly_retired: List[str] = []
        existing_ids = set(index["all_ids"])
        geometry_preview: Dict[str, Any] = {}

        if action == "SPLIT_SURFACE":
            collection, target_object = _find(working, targets[0])
            fragment_target = target_object if collection == "fragments" else None
            if fragment_target is None and body.get("fragment_id"):
                _, fragment_target = _find(working, str(body["fragment_id"]))
            if fragment_target is not None and collection != "surfaces":
                # ---- Fragment-level geometric split (W151) -----------------
                fragment = fragment_target
                if fragment.get("fragment_id") != str(body.get("fragment_id", fragment["fragment_id"])):
                    raise TransactionError("fragment target does not match payload fragment_id", "BAD_TARGET")
                all_indices = [int(item) for item in fragment["support_ref"]["indices"]]
                new_indices_raw = body.get("new_indices")
                if new_indices_raw is None and body.get("split_axis") is not None:
                    axis = np.asarray(body["split_axis"], dtype=np.float64)
                    norm = float(np.linalg.norm(axis))
                    if not math.isfinite(norm) or norm <= 1e-12:
                        raise TransactionError("split_axis must be a finite non-zero vector", "BAD_SPLIT")
                    axis = axis / norm
                    origin = np.asarray(body.get("split_point") or [0.0, 0.0, 0.0], dtype=np.float64)
                    if origin.shape != (3,):
                        raise TransactionError("split_point must be a 3D point", "BAD_SPLIT")
                    all_points, split_source = self._resolve_fragment_points(fragment, all_indices)
                    side = (all_points - origin) @ axis
                    new_indices = [index for index, value in zip(all_indices, side) if float(value) > 0.0]
                    keep_indices = [index for index, value in zip(all_indices, side) if float(value) <= 0.0]
                else:
                    new_indices = [int(item) for item in (new_indices_raw or [])]
                    if not new_indices or len(set(new_indices)) != len(new_indices):
                        raise TransactionError("fragment split needs unique new_indices", "BAD_SPLIT")
                    if not set(new_indices).issubset(all_indices):
                        raise TransactionError("new_indices must be a subset of the current support", "BAD_SPLIT")
                    keep_raw = body.get("keep_indices")
                    keep_indices = ([int(item) for item in keep_raw] if keep_raw is not None
                                    else [index for index in all_indices if index not in set(new_indices)])
                if not new_indices or not keep_indices:
                    raise TransactionError("fragment split must leave non-empty support on both sides", "BAD_SPLIT")
                if len(set(keep_indices)) != len(keep_indices) or not set(keep_indices).issubset(all_indices):
                    raise TransactionError("keep_indices must be a unique subset of the current support", "BAD_SPLIT")
                if set(keep_indices) & set(new_indices):
                    raise TransactionError("keep_indices and new_indices overlap", "BAD_SPLIT")
                if sorted(set(keep_indices) | set(new_indices)) != sorted(set(all_indices)):
                    raise TransactionError(
                        "fragment split must partition the current support, not drop or invent points",
                        "BAD_SPLIT",
                    )
                if len(keep_indices) < MIN_SUPPORT_POINTS or len(new_indices) < MIN_SUPPORT_POINTS:
                    raise SupportGeometryError(
                        "fragment split sides need at least %d support points" % MIN_SUPPORT_POINTS,
                        "INSUFFICIENT_SUPPORT", keep=len(keep_indices), new=len(new_indices),
                    )
                keep_points, keep_fit, keep_source = self._fragment_fit(
                    fragment, keep_indices, body, label="fragment_split_keep",
                )
                new_points, new_fit, new_source = self._fragment_fit(
                    fragment, new_indices, body, label="fragment_split_new",
                )
                # Decide, before mutating, whether each containing surface can
                # keep both children.  If their fitted planes are no longer
                # coplanar, the child gets its own Surface instead of forcing
                # two planes into one finite support.
                surface_plans: List[Dict[str, Any]] = []
                for surface in working["surfaces"]:
                    members = list(surface["member_fragment_ids"])
                    if fragment["fragment_id"] not in members:
                        continue
                    frames: List[Dict[str, Any]] = []
                    for member in members:
                        if member == fragment["fragment_id"]:
                            frames.append(normalise_support_frame(keep_fit.frame()))
                        else:
                            other = next(item for item in working["fragments"] if item["fragment_id"] == member)
                            frame, declared = self._frame_for_fragment(other)
                            if not declared:
                                raise SupportGeometryError(
                                    "cannot verify coplanarity for member %s" % member,
                                    "SUPPORT_FRAME_UNAVAILABLE", fragment_id=member,
                                )
                            frames.append(frame)
                    frames.append(normalise_support_frame(new_fit.frame()))
                    report = coplanarity_report(
                        frames, angle_tolerance_deg=float(body.get("coplanar_angle_deg", 1.0)),
                        offset_tolerance=(float(body["coplanar_offset_tolerance"])
                                          if body.get("coplanar_offset_tolerance") is not None else None),
                    )
                    surface_plans.append({"surface": surface, "join": bool(report["ok"]),
                                          "report": report})
                new_fragment_id = make_id(
                    "frag", {"parent": fragment["fragment_id"], "indices": sorted(new_indices),
                             "event": event_id},
                    existing_ids, unordered_fields=("indices",),
                )
                new_fragment = copy.deepcopy(fragment)
                new_fragment["fragment_id"] = new_fragment_id
                new_fragment["version"] = 1
                self._apply_fit_to_fragment(fragment, keep_indices, keep_fit, keep_source, event_id,
                                            "fragment_split_keep")
                _bump(fragment)
                self._apply_fit_to_fragment(new_fragment, new_indices, new_fit, new_source, event_id,
                                            "fragment_split_new")
                working["fragments"].append(new_fragment)
                changed.extend([fragment["fragment_id"], new_fragment_id])
                surface_ids: List[str] = []
                for plan in surface_plans:
                    surface = plan["surface"]
                    if plan["join"]:
                        if new_fragment_id not in surface["member_fragment_ids"]:
                            surface["member_fragment_ids"].append(new_fragment_id)
                        self._rebuild_surface_geometry(
                            working, surface["surface_id"], body=body, event_id=event_id,
                            error_code="SPLIT_BREAKS_SURFACE_COPLANARITY", mode="FRAGMENT_GEOMETRY_SPLIT",
                        )
                        surface_ids.append(surface["surface_id"])
                        changed.append(surface["surface_id"])
                    else:
                        # Children diverged geometrically: keep the parent
                        # surface for the retained fragment and create a new
                        # surface for the new child.  No copied boundary and no
                        # relation is fabricated for the new surface.
                        self._rebuild_surface_geometry(
                            working, surface["surface_id"], body=body, event_id=event_id,
                            error_code="SPLIT_BREAKS_SURFACE_COPLANARITY",
                            mode="FRAGMENT_GEOMETRY_SPLIT_KEEP",
                        )
                        child_surface_id = make_id(
                            "surf", {"parent": surface["surface_id"], "fragment": new_fragment_id,
                                     "event": event_id},
                            existing_ids | {new_fragment_id},
                        )
                        child_surface = {
                            "surface_id": child_surface_id,
                            "member_fragment_ids": [new_fragment_id],
                            "interpretation_status": surface["interpretation_status"],
                            "finite_support": copy.deepcopy(new_fit.regions),
                            "provenance": surface["provenance"],
                            "version": 1,
                        }
                        working["surfaces"].append(child_surface)
                        self._store_surface_geometry(working, child_surface_id, {
                            "frame_local": new_fit.frame(),
                            "member_count": 1,
                            "region_metrics": region_metrics(new_fit.regions),
                            "components": [{
                                "fragment_id": new_fragment_id,
                                "frame_local": new_fit.frame(),
                                "source_region_count": len(new_fit.regions),
                                "transformed_to_target": False,
                            }],
                            "policy": "REAL_SUPPORT_POINTS_NO_AUTO_CANDIDATE_REQUIRED",
                            "event_id": event_id,
                        }, "FRAGMENT_GEOMETRY_SPLIT_NEW_SURFACE")
                        surface_ids.extend([surface["surface_id"], child_surface_id])
                        changed.extend([surface["surface_id"], child_surface_id])
                        new_links.append(self._lineage_link(
                            "split", [surface["surface_id"]], [surface["surface_id"], child_surface_id],
                            event_id,
                            note="fragment children were not coplanar; child surface created instead of forcing one plane",
                        ))
                        warnings.append("FRAGMENT_SPLIT_GEOMETRY_NOT_COPLANAR_NEW_SURFACE")
                        reason_codes.append("FRAGMENT_SPLIT_NEW_SURFACE")
                new_links.append(self._lineage_link(
                    "fragment_split", [fragment["fragment_id"]],
                    [fragment["fragment_id"], new_fragment_id], event_id,
                    note="support points partitioned; both children refitted independently",
                ))
                reason_codes.extend(["LINEAGE_FRAGMENT_SPLIT", "FRAGMENT_GEOMETRY_SPLIT"])
                warnings.append("HUMAN_REVIEW_REQUIRED: both child fragments were refitted from selected support")
                geometry_preview = {
                    "kind": "fragment_split",
                    "keep": self._fit_preview(keep_fit, keep_source, "fragment_split_keep"),
                    "new": self._fit_preview(new_fit, new_source, "fragment_split_new"),
                    "objects": self._geometry_preview_for_ids(
                        working, [fragment["fragment_id"], new_fragment_id] + surface_ids,
                        label="fragment_split",
                    )["objects"],
                }
            else:
                if collection != "surfaces":
                    raise TransactionError("SPLIT_SURFACE requires a surface or fragment target", "BAD_TARGET")
                surface = target_object
                mode = str(body.get("mode") or "interpretation_only")
                if mode not in ("interpretation_only", "surface_interpretation"):
                    raise TransactionError("split mode must be surface interpretation only", "BAD_SPLIT")
                moved = list(body.get("member_ids_for_new") or [])
                if not moved or not set(moved).issubset(surface["member_fragment_ids"]):
                    raise TransactionError("split members must be a non-empty subset", "BAD_SPLIT")
                remaining = [member for member in surface["member_fragment_ids"] if member not in moved]
                if not remaining:
                    raise TransactionError("split must leave at least one member", "BAD_SPLIT")
                # Rebuild both sides from their real member finite regions before
                # mutating anything.  A non-coplanar side is an explicit refusal,
                # never a synthetic plane invented by the engine.
                if self._real_geometry:
                    remaining_regions, remaining_meta = self._surface_regions_from_members(
                        working, remaining, body=body, event_id=event_id,
                        error_code="SPLIT_REQUIRES_COPLANAR_CHILD_SUPPORT",
                    )
                    moved_regions, moved_meta = self._surface_regions_from_members(
                        working, moved, body=body, event_id=event_id,
                        error_code="SPLIT_REQUIRES_COPLANAR_CHILD_SUPPORT",
                    )
                else:
                    remaining_regions, remaining_meta = self._fixture_surface_regions(
                        working, remaining, event_id=event_id,
                    )
                    moved_regions, moved_meta = self._fixture_surface_regions(
                        working, moved, event_id=event_id,
                    )
                    warnings.append("FIXTURE_LEGACY_SPLIT_USES_MEMBER_REGIONS_NO_REFIT")
                new_surface_id = make_id(
                    "surf", {"parent": targets[0], "members": sorted(moved), "event": event_id},
                    existing_ids, unordered_fields=("members",),
                )
                surface["member_fragment_ids"] = remaining
                surface["finite_support"] = copy.deepcopy(remaining_regions)
                _bump(surface)
                self._store_surface_geometry(working, targets[0], remaining_meta, "INTERPRETATION_ONLY")
                new_surface = {
                    "surface_id": new_surface_id,
                    "member_fragment_ids": list(moved),
                    "interpretation_status": str(body.get("interpretation_status")
                                                  or surface["interpretation_status"]),
                    "finite_support": copy.deepcopy(moved_regions),
                    "provenance": surface["provenance"],
                    "version": 1,
                }
                working["surfaces"].append(new_surface)
                self._store_surface_geometry(working, new_surface_id, moved_meta, "INTERPRETATION_ONLY")
                changed.extend([targets[0], new_surface_id])
                if body.get("offset") is not None:
                    warnings.append("SYNTHETIC_OFFSET_IGNORED: split geometry comes from member support regions")
                for relation in list(working["relations"]):
                    if relation["src_id"] == targets[0] or relation["dst_id"] == targets[0]:
                        clone = copy.deepcopy(relation)
                        if clone["src_id"] == targets[0]:
                            clone["src_id"] = new_surface_id
                        if clone["dst_id"] == targets[0]:
                            clone["dst_id"] = new_surface_id
                        clone["relation_id"] = make_id(
                            "rel", {"src": clone["src_id"], "dst": clone["dst_id"],
                                    "type": clone["relation_type"], "roi": clone["roi_id"],
                                    "event": event_id},
                            existing_ids | {item["relation_id"] for item in working["relations"]},
                        )
                        clone["version"] = 1
                        working["relations"].append(clone)
                        changed.append(clone["relation_id"])
                new_links.append({
                    "kind": "split", "parent_ids": [targets[0]],
                    "child_ids": [targets[0], new_surface_id], "event_id": event_id,
                    "note": "first child retains the original persistent ID; geometry from real member regions",
                })
                reason_codes.extend(["LINEAGE_SPLIT", "INSTANCE_MEMBER", "SURFACE_INTERPRETATION_SPLIT"])
                geometry_preview = {
                    "kind": "surface_interpretation_split",
                    "surfaces": self._geometry_preview_for_ids(
                        working, [targets[0], new_surface_id], label="surface_split",
                    )["objects"],
                }

        elif action == "MERGE_SURFACE":
            surfaces = []
            for target in targets:
                collection, item = _find(working, target)
                if collection != "surfaces":
                    raise TransactionError("MERGE_SURFACE requires surface targets", "BAD_TARGET")
                surfaces.append(item)
            if len(surfaces) < 2:
                raise TransactionError("merge needs at least two surfaces", "BAD_MERGE")
            mode = str(body.get("mode") or "interpretation_only")
            if mode not in ("interpretation_only", "fuse_geometry"):
                raise TransactionError("merge mode must be interpretation_only or fuse_geometry", "BAD_MERGE")
            members: List[str] = []
            for item in surfaces:
                for member in item["member_fragment_ids"]:
                    if member not in members:
                        members.append(member)
            merged_id = make_id(
                "surf", {"parents": sorted(targets), "members": sorted(members), "event": event_id},
                existing_ids, unordered_fields=("parents", "members"),
            )
            if self._real_geometry:
                # Coplanarity is checked for both modes: one Surface payload
                # stores one finite support, so a multi-plane interpretation
                # cannot be represented without lying about its geometry.
                _, coplanar_meta = self._surface_regions_from_members(
                    working, members, body=body, event_id=event_id,
                    error_code="MERGE_REQUIRES_COPLANAR_SUPPORT",
                )
                if mode == "fuse_geometry":
                    point_batches = []
                    sources = set()
                    for member in members:
                        fragment = next(item for item in working["fragments"]
                                        if item["fragment_id"] == member)
                        points, source = self._resolve_fragment_points(fragment)
                        point_batches.append(points)
                        sources.add(source)
                    if len(set(sources)) != 1:
                        raise SupportGeometryError(
                            "merged support mixes original and fixture geometry", "SUPPORT_SOURCE_MIXED"
                        )
                    all_points = np.vstack(point_batches)
                    fit = fit_support_geometry(all_points, options=self._fit_options(body))
                    merged_regions = copy.deepcopy(fit.regions)
                    metadata = {
                        "frame_local": fit.frame(),
                        "coplanarity": coplanar_meta["coplanarity"],
                        "member_count": len(members),
                        "region_metrics": region_metrics(merged_regions),
                        "components": copy.deepcopy(coplanar_meta["components"]),
                        "fit": self._fit_preview(fit, next(iter(sources)), "merge_fuse_refit"),
                        "policy": "REAL_SUPPORT_POINTS_REFIT_NO_CONVEX_FILL",
                        "event_id": event_id,
                    }
                    geometry_preview = {
                        "kind": "merge_fuse_geometry",
                        "fit": metadata["fit"],
                    }
                    reason_codes.append("MERGE_GEOMETRY_SUPPORT_REFIT")
                else:
                    merged_regions, metadata = self._surface_regions_from_members(
                        working, members, body=body, event_id=event_id,
                        error_code="MERGE_REQUIRES_COPLANAR_SUPPORT",
                    )
                    geometry_preview = {
                        "kind": "merge_interpretation_only",
                        "surfaces": self._geometry_preview_for_ids(
                            working, [merged_id], label="merge_interpretation",
                        )["objects"],
                    }
                    reason_codes.append("MERGE_INTERPRETATION_KEEPS_PER_FRAGMENT_REGIONS")
                merged = {
                    "surface_id": merged_id,
                    "member_fragment_ids": list(members),
                    "interpretation_status": str(body.get("interpretation_status")
                                                  or surfaces[0]["interpretation_status"]),
                    "finite_support": copy.deepcopy(merged_regions),
                    "provenance": str(body.get("provenance") or surfaces[0]["provenance"]),
                    "version": max(int(item["version"]) for item in surfaces) + 1,
                }
            else:
                # Legacy fixture path only (no original point provider).  It is
                # labelled in the event warnings and is never reachable from the
                # workbench service, which always installs a provider.
                merged = copy.deepcopy(surfaces[0])
                merged["surface_id"] = merged_id
                merged["member_fragment_ids"] = members
                merged["version"] = max(int(item["version"]) for item in surfaces) + 1
                merged_regions = copy.deepcopy(surfaces[0]["finite_support"])
                metadata = {
                    "frame_local": None,
                    "member_count": len(members),
                    "region_metrics": region_metrics(merged_regions),
                    "components": [],
                    "policy": "SYNTHETIC_FIXTURE_LEGACY_FIRST_MEMBER_GEOMETRY",
                    "event_id": event_id,
                }
                warnings.append("FIXTURE_LEGACY_MERGE_GEOMETRY_FROM_FIRST_MEMBER")
                geometry_preview = {
                    "kind": "merge_fixture_legacy",
                    "warning": "FIXTURE_LEGACY_MERGE_GEOMETRY_FROM_FIRST_MEMBER",
                }
            working["surfaces"] = [item for item in working["surfaces"] if item["surface_id"] not in targets]
            working["surfaces"].append(merged)
            if self._real_geometry and mode == "interpretation_only":
                geometry_preview = {
                    "kind": "merge_interpretation_only",
                    "surfaces": self._geometry_preview_for_ids(
                        working, [merged_id], label="merge_interpretation",
                    )["objects"],
                }
            self._store_surface_geometry(
                working, merged_id, metadata,
                "FUSE_REFIT" if (self._real_geometry and mode == "fuse_geometry") else "INTERPRETATION_ONLY",
            )
            changed.append(merged_id)
            newly_retired.extend(targets)

            rewritten: Dict[str, Dict[str, Any]] = {}
            for relation in working["relations"]:
                src = merged_id if relation["src_id"] in targets else relation["src_id"]
                dst = merged_id if relation["dst_id"] in targets else relation["dst_id"]
                relation["src_id"], relation["dst_id"] = src, dst
                _bump(relation)
                key = "|".join([relation["relation_type"], src, dst, relation["roi_id"]])
                if src == dst:
                    newly_retired.append(relation["relation_id"])
                    warnings.append("SELF_LOOP_REMOVED_ON_MERGE:" + relation["relation_id"])
                    continue
                if key in rewritten:
                    newly_retired.append(relation["relation_id"])
                    warnings.append("DUPLICATE_RELATION_COLLAPSED:" + relation["relation_id"])
                    continue
                rewritten[key] = relation
                # Persistent ID is preserved: only the endpoints and version
                # change.  Re-deriving relation IDs from endpoints would look
                # like an ID shift after a merge.
                changed.append(relation["relation_id"])
            working["relations"] = [
                item for item in working["relations"]
                if item["relation_id"] not in newly_retired
            ]
            new_links.append({
                "kind": "merge", "parent_ids": sorted(targets), "child_ids": [merged_id],
                "event_id": event_id,
            })
            reason_codes.extend(["LINEAGE_MERGE", "RELATION_ENDPOINT_REWIRED"])

        elif action == "DELETE_OBJECT":
            doomed = set(targets)
            for target in targets:
                collection, item = _find(working, target)
                if collection is None:
                    raise TransactionError("delete target does not exist: %s" % target, "BAD_TARGET")
                if collection == "surfaces":
                    doomed.update(item["member_fragment_ids"])
                elif collection == "fragments":
                    for surface in working["surfaces"]:
                        if target in surface["member_fragment_ids"]:
                            surface["member_fragment_ids"] = [
                                member for member in surface["member_fragment_ids"] if member != target
                            ]
                            if not surface["member_fragment_ids"]:
                                doomed.add(surface["surface_id"])
                            else:
                                _bump(surface)
                                changed.append(surface["surface_id"])
            # Cascade: a surface emptied by a deleted fragment disappears too.
            for surface in list(working["surfaces"]):
                if surface["surface_id"] in doomed:
                    doomed.update(surface["member_fragment_ids"])
            # A fragment shared with a surviving surface must NOT be removed;
            # otherwise that surface would be left with a dangling member.
            surviving_members = {
                member
                for surface in working["surfaces"]
                if surface["surface_id"] not in doomed
                for member in surface["member_fragment_ids"]
            }
            doomed = {
                object_id for object_id in doomed
                if object_id not in set(index["fragment_ids"]) or object_id not in surviving_members
            }
            for relation in list(working["relations"]):
                if (relation["src_id"] in doomed or relation["dst_id"] in doomed
                        or relation["relation_id"] in doomed):
                    doomed.add(relation["relation_id"])
            for collection, id_field in COLLECTION_ID_FIELDS:
                working[collection] = [
                    item for item in working[collection] if item[id_field] not in doomed
                ]
            newly_retired.extend(sorted(doomed))
            changed.extend(sorted(doomed))
            new_links.append({
                "kind": "delete", "parent_ids": sorted(doomed), "child_ids": [],
                "event_id": event_id,
            })
            reason_codes.append("CASCADE_DELETE_REFERENCES")

        elif action in ("SET_RELATION", "SET_UNKNOWN"):
            for target in targets:
                collection, relation = _find(working, target)
                if collection != "relations":
                    raise TransactionError("relation target required", "BAD_TARGET")
                if action == "SET_RELATION":
                    state = body.get("state")
                    if state not in ("yes", "no", "unknown"):
                        raise TransactionError("SET_RELATION needs a valid state", "BAD_RELATION_STATE")
                    relation["state"] = state
                    relation["unknown_reason"] = body.get("unknown_reason") if state == "unknown" else None
                else:
                    relation["state"] = "unknown"
                    relation["unknown_reason"] = body.get("unknown_reason", "NOT_EVALUATED")
                relation["provenance"] = "HUMAN_CONFIRMED"
                _bump(relation)
                changed.append(target)
            reason_codes.append("RELATION_STATE_CHANGED")

        elif action == "EDIT_BOUNDARY":
            collection, surface = _find(working, targets[0])
            if collection != "surfaces":
                raise TransactionError("boundary target must be a surface", "BAD_TARGET")
            ring = body["ring"]
            surface["finite_support"][0] = {"exterior": ring, "holes": []}
            _bump(surface)
            changed.append(targets[0])
            reason_codes.append("GEOMETRY_BOUNDARY_CHANGED")

        elif action == "SET_INTERPRETATION":
            for target in targets:
                collection, surface = _find(working, target)
                if collection != "surfaces":
                    raise TransactionError("interpretation target must be a surface", "BAD_TARGET")
                surface["interpretation_status"] = body["interpretation_status"]
                _bump(surface)
                changed.append(target)

        elif action == "PROTECT_REGION":
            region_id = body["region_id"]
            if any(item["region_id"] == region_id for item in working["protected_regions"]):
                raise TransactionError("protected region already exists", "DUPLICATE_ID")
            working["protected_regions"].append({
                "region_id": region_id,
                "roi_id": body["roi_id"],
                "finite_regions": copy.deepcopy(body["finite_regions"]),
            })
            changed.append(region_id)

        elif action == "UNPROTECT_REGION":
            working["protected_regions"] = [
                item for item in working["protected_regions"] if item["region_id"] not in targets
            ]
            changed.extend(targets)
            newly_retired.extend(targets)

        elif action == "ADD_FRAGMENT":
            if "fragment" in body:
                fragment = copy.deepcopy(body["fragment"])
                if any(item["fragment_id"] == fragment["fragment_id"] for item in working["fragments"]):
                    raise TransactionError("fragment ID already exists", "DUPLICATE_ID")
                working["fragments"].append(fragment)
                collection, surface = _find(working, body.get("surface_id", ""))
                if collection != "surfaces":
                    raise TransactionError("ADD_FRAGMENT needs a target surface", "BAD_TARGET")
                surface["member_fragment_ids"].append(fragment["fragment_id"])
                _bump(surface)
                if self._real_geometry:
                    self._rebuild_surface_geometry(
                        working, surface["surface_id"], body=body, event_id=event_id,
                        error_code="ADD_FRAGMENT_REQUIRES_COPLANAR_SUPPORT", mode="MEMBER_ADDED",
                    )
                changed.extend([fragment["fragment_id"], body["surface_id"]])
                new_links.append({
                    "kind": "add", "parent_ids": [body["surface_id"]],
                    "child_ids": [fragment["fragment_id"]], "event_id": event_id,
                })
                if self._real_geometry:
                    geometry_preview = {
                        "kind": "add_fragment",
                        "objects": self._geometry_preview_for_ids(
                            working, [fragment["fragment_id"], body["surface_id"]], label="add_fragment",
                        )["objects"],
                    }
            else:
                # ---- W151: create a real face from a support selection -----
                observation_id = body.get("observation_id")
                target_collection, target_object = _find(working, targets[0])
                if observation_id is None:
                    if target_collection == "observations":
                        observation_id = targets[0]
                    elif target_collection == "surfaces" and target_object["member_fragment_ids"]:
                        member = next(item for item in working["fragments"]
                                      if item["fragment_id"] == target_object["member_fragment_ids"][0])
                        observation_id = member["observation_id"]
                    else:
                        raise TransactionError(
                            "ADD_FRAGMENT from support needs observation_id or an observation/surface target",
                            "BAD_TARGET",
                        )
                observation = next((item for item in working["observations"]
                                    if item["observation_id"] == observation_id), None)
                if observation is None:
                    raise TransactionError("observation is missing: %s" % observation_id, "BAD_TARGET")
                indices = [int(item) for item in (body.get("indices") or [])]
                if not indices or len(set(indices)) != len(indices):
                    raise TransactionError("support indices must be non-empty and unique", "BAD_SUPPORT")
                roi_id = str(body.get("roi_id") or observation["roi"][0]["roi_id"])
                if roi_id not in {item["roi_id"] for item in observation["roi"]}:
                    raise TransactionError("support ROI is not declared by the observation", "BAD_SUPPORT")
                points, source = self._resolve_selection_points(observation_id, indices)
                fit = fit_support_geometry(points, options=self._fit_options(body))
                if len(indices) < MIN_SUPPORT_POINTS:
                    raise SupportGeometryError("support selection is too small", "INSUFFICIENT_SUPPORT")
                source_files = list(observation.get("source_files") or [])
                source_sha = (source_files[0]["sha256"] if source_files else observation["sha256"])
                storage_uri = (source_files[0]["path"] if source_files else
                               "observation://%s/points" % observation_id)
                new_fragment_id = make_id(
                    "frag", {"observation": observation_id, "indices": sorted(indices), "event": event_id},
                    existing_ids, unordered_fields=("indices",),
                )
                fit_metrics = region_metrics(fit.regions)
                new_fragment = {
                    "fragment_id": new_fragment_id,
                    "observation_id": observation_id,
                    "support_ref": {
                        "observation_id": observation_id,
                        "source_sha256": source_sha,
                        "storage_uri": storage_uri,
                        "index_kind": "original_point_index",
                        "indices": indices,
                        "roi_id": roi_id,
                        "decimation_lineage": {"selected_from": "original_point_index",
                                               "auto_candidate_missed": True},
                    },
                    "plane_local": {"normal": list(fit.normal), "offset": float(fit.offset)},
                    "finite_regions_local": copy.deepcopy(fit.regions),
                    "residuals": [float(item) for item in fit.residuals],
                    "quality": {
                        "support_point_count": int(fit.support_point_count),
                        "rmse": float(fit.rmse),
                        "max_abs_residual": float(fit.max_abs_residual),
                        "plane_tolerance": float(fit.plane_tolerance),
                        "finite_region_count": int(fit_metrics["region_count"]),
                        "hole_count": int(fit_metrics["hole_count"]),
                        "total_area_local": float(fit_metrics["total_area"]),
                        "region_areas": [float(item) for item in fit_metrics["region_areas"]],
                        "fit_method": "svd_plane_plus_alpha_shape",
                        "fit_config_hash": fit.fit_config["config_hash"],
                        "frame_local": fit.frame(),
                        "geometry_source": source,
                        "requires_human_review": True,
                        "human_selected_auto_candidate_missed": True,
                        "synthetic": False,
                    },
                    "candidate_generator": {
                        "candidate_batch_id": ("human_support_selection_" + event_id)[:120],
                        "generator_config_hash": fit.fit_config["config_hash"],
                        "support_refit": {"event_id": event_id, "geometry_source": source},
                    },
                    "version": 1,
                }
                surface_id = body.get("surface_id")
                if surface_id:
                    collection, surface = _find(working, surface_id)
                    if collection != "surfaces":
                        raise TransactionError("ADD_FRAGMENT needs a target surface", "BAD_TARGET")
                    self._assert_hypothetical_coplanarity(
                        working, new_fragment_id, fit.frame(), body=body,
                        error_code="ADD_FRAGMENT_REQUIRES_COPLANAR_SUPPORT",
                    )
                    surface["member_fragment_ids"].append(new_fragment_id)
                    working["fragments"].append(new_fragment)
                    _bump(surface)
                    self._rebuild_surface_geometry(
                        working, surface_id, body=body, event_id=event_id,
                        error_code="ADD_FRAGMENT_REQUIRES_COPLANAR_SUPPORT", mode="NEW_FACE_FROM_SUPPORT",
                    )
                else:
                    surface_id = make_id(
                        "surf", {"fragment": new_fragment_id, "event": event_id},
                        existing_ids,
                    )
                    new_surface_payload = dict(body.get("new_surface") or {})
                    provenance = str(new_surface_payload.get("provenance") or "HUMAN_INTERPRETED")
                    new_surface = {
                        "surface_id": surface_id,
                        "member_fragment_ids": [new_fragment_id],
                        "interpretation_status": str(
                            new_surface_payload.get("interpretation_status") or "human_selected_face"
                        ),
                        "finite_support": copy.deepcopy(fit.regions),
                        "provenance": provenance,
                        "version": 1,
                    }
                    working["fragments"].append(new_fragment)
                    working["surfaces"].append(new_surface)
                    metadata = {
                        "frame_local": fit.frame(),
                        "member_count": 1,
                        "region_metrics": fit_metrics,
                        "components": [{
                            "fragment_id": new_fragment_id,
                            "frame_local": fit.frame(),
                            "source_region_count": len(fit.regions),
                            "transformed_to_target": False,
                        }],
                        "policy": "REAL_SUPPORT_POINTS_NO_AUTO_CANDIDATE_REQUIRED",
                        "event_id": event_id,
                    }
                    self._store_surface_geometry(working, surface_id, metadata, "NEW_FACE_FROM_SUPPORT")
                changed.extend([new_fragment_id, surface_id])
                new_links.append(self._lineage_link(
                    "create_from_support", [], [new_fragment_id, surface_id], event_id,
                    note="human-selected support points from %s; no automatic candidate was required"
                         % observation_id,
                ))
                reason_codes.extend(["SUPPORT_GEOMETRY_REFIT", "NEW_FACE_FROM_SUPPORT",
                                     "AUTO_CANDIDATE_MISSED_PATH"])
                warnings.append("HUMAN_REVIEW_REQUIRED: new face geometry was fitted from selected support points")
                geometry_preview = {
                    "kind": "create_face_from_support",
                    "fit": self._fit_preview(fit, source, "create_face_from_support"),
                    "objects": self._geometry_preview_for_ids(
                        working, [new_fragment_id, surface_id], label="create_face_from_support",
                    )["objects"],
                }

        elif action == "SELECT_SUPPORT":
            collection, fragment = _find(working, targets[0])
            if collection != "fragments":
                raise TransactionError("SELECT_SUPPORT needs a fragment target", "BAD_TARGET")
            indices = [int(item) for item in body["indices"]]
            if not indices or len(set(indices)) != len(indices):
                raise TransactionError("support indices must be non-empty and unique", "BAD_SUPPORT")
            if not self._real_geometry:
                if self._real_geometry_required:
                    # T-D: a real-measurement project with no original point
                    # provider is a hard refusal.  The previous behaviour
                    # (replace indices, keep the old plane/regions/residuals)
                    # produced a silently inconsistent fragment.
                    raise SupportGeometryError(
                        "SELECT_SUPPORT on a real measurement requires the original "
                        "point source; this engine has no verified provider",
                        "SUPPORT_POINTS_UNAVAILABLE", fragment_id=fragment["fragment_id"],
                        indices=indices[:20],
                    )
                # Explicit legacy fixture path: no original point source exists.
                # The defect (index-only update) is preserved only here and is
                # labelled so it can never be mistaken for a real refit.
                fragment["support_ref"]["indices"] = indices
                _bump(fragment)
                changed.append(targets[0])
                warnings.append("FIXTURE_LEGACY_SELECT_SUPPORT_INDEX_ONLY_NO_REFIT")
                geometry_preview = {
                    "kind": "support_reselect_fixture_legacy",
                    "warning": "FIXTURE_LEGACY_SELECT_SUPPORT_INDEX_ONLY_NO_REFIT",
                }
            else:
                points, fit, source = self._fragment_fit(fragment, indices, body, label="select_support")
                self._assert_hypothetical_coplanarity(
                    working, fragment["fragment_id"], fit.frame(), body=body,
                    error_code="SELECT_SUPPORT_BREAKS_SURFACE_COPLANARITY",
                )
                metrics = self._apply_fit_to_fragment(
                    fragment, indices, fit, source, event_id, "select_support",
                )
                _bump(fragment)
                changed.append(targets[0])
                surface_ids = []
                for surface in working["surfaces"]:
                    if fragment["fragment_id"] in surface["member_fragment_ids"]:
                        self._rebuild_surface_geometry(
                            working, surface["surface_id"], body=body, event_id=event_id,
                            error_code="SELECT_SUPPORT_BREAKS_SURFACE_COPLANARITY",
                            mode="SUPPORT_REFIT",
                        )
                        surface_ids.append(surface["surface_id"])
                        changed.append(surface["surface_id"])
                new_links.append(self._lineage_link(
                    "support_reselection", [fragment["fragment_id"]], [fragment["fragment_id"]],
                    event_id, note="support indices replaced and fragment geometry refitted",
                ))
                reason_codes.extend(["SUPPORT_GEOMETRY_REFIT", "HUMAN_REVIEW_REQUIRED"])
                warnings.append("HUMAN_REVIEW_REQUIRED: refitted from the newly selected support points")
                geometry_preview = {
                    "kind": "support_reselect",
                    "fit": self._fit_preview(fit, source, "select_support"),
                    "objects": self._geometry_preview_for_ids(
                        working, [fragment["fragment_id"]] + surface_ids, label="select_support",
                    )["objects"],
                }

        else:
            raise TransactionError("unsupported edit action %r" % action, "ACTION_UNSUPPORTED")

        return {
            "changed_ids": sorted(set(changed)),
            "reason_codes": reason_codes,
            "warnings": warnings,
            "new_links": new_links,
            "newly_retired": newly_retired,
            "geometry_preview": geometry_preview,
        }

    # -- event records --------------------------------------------------------

    def _make_event(self, draft: EditDraft, parent: Optional[str], created_at: str) -> Dict[str, Any]:
        event_ids = {item["event_id"] for item in self.events}
        # W157 / D-1: identity is derived from logical content and a
        # deterministic event ordinal, never from the wall clock.  created_at
        # stays in the returned record as real audit metadata.
        event_id = make_id("event", {
            "project_id": self.project.project_id,
            "base_version": draft.base_version,
            "sequence": len(self.events),
            "parent": parent,
            "action": draft.action,
            "targets": sorted(draft.target_ids),
            "payload": draft.payload,
            "actor": draft.actor,
            "hardness": draft.hardness,
            "tolerance": draft.tolerance,
        }, event_ids, unordered_fields=("targets",))
        return {
            "event_id": event_id,
            "project_id": self.project.project_id,
            "base_version": draft.base_version,
            "parent_event_id": parent,
            "actor": draft.actor,
            "action": draft.action,
            "target_ids": list(draft.target_ids),
            "payload": copy.deepcopy(draft.payload),
            "tolerance": float(draft.tolerance),
            "hardness": draft.hardness,
            "created_at": created_at,
        }

    def _contract_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """Every public action, including the three v2 additions, has one EditEvent."""
        if event["action"] not in EDIT_ACTIONS_V2:
            raise TransactionError("action %r is not a public contract action" % event["action"],
                                   "CONTRACT_EVENT_REQUIRED")
        payload = {key: copy.deepcopy(event[key]) for key in EDIT_EVENT_FIELDS}
        return EditEvent.from_dict(payload).to_dict()

    def _hard_conflict_check(self, event: Dict[str, Any]) -> None:
        if event["hardness"] != "hard" or event["action"] not in EDIT_ACTIONS_V2:
            return
        siblings = []
        for record in self._tokens.values():
            other = record["preview"]
            if (not record["used"] and not record.get("discarded")
                    and other.event["hardness"] == "hard"
                    and other.event["action"] in EDIT_ACTIONS_V2
                    and other.base_version == event["base_version"]):
                siblings.append(EditEvent.from_dict(
                    {key: other.event[key] for key in EDIT_EVENT_FIELDS}
                ))
        if not siblings:
            return
        candidate = EditEvent.from_dict({key: event[key] for key in EDIT_EVENT_FIELDS})
        conflicts = find_hard_conflicts([candidate] + siblings)
        if conflicts:
            raise TransactionError("incompatible hard edits at one base version", "HARD_EDIT_CONFLICT")

    def _engine_record(
        self,
        event: Dict[str, Any],
        *,
        result_version: int,
        input_hash_before: str,
        input_hash_after: str,
        applied: Dict[str, Any],
        dependency: Mapping[str, Any],
        warnings: Sequence[str],
    ) -> Dict[str, Any]:
        contract_event = self._contract_event(event)
        return {
            **copy.deepcopy(event),
            "result_version": result_version,
            "input_hash_before": input_hash_before,
            "input_hash_after": input_hash_after,
            "changed_ids": list(applied["changed_ids"]),
            "affected_ids": sorted(set(applied["changed_ids"]) | set(dependency["closure"])),
            "reason_codes": list(applied["reason_codes"]),
            "reason_paths": copy.deepcopy(dependency["reason_paths"]),
            "dependency_closure": copy.deepcopy(dependency["closure"]),
            "dependency_counts": copy.deepcopy(dependency.get("counts", {})),
            "contract_action": True,
            "contract_event": contract_event,
            "warnings": list(warnings),
        }

    # -- two-phase transaction ------------------------------------------------

    def preview(self, draft: EditDraft) -> Preview:
        before = self.project
        check_base_version(draft.base_version, before.project_version)
        if draft.action not in SUPPORTED_ACTIONS:
            raise TransactionError("unsupported edit action %r" % draft.action, "ACTION_UNSUPPORTED")
        if not draft.target_ids:
            raise TransactionError("edit target list is empty", "EMPTY_TARGETS")
        index = reference_index(before)
        known = set(index["all_ids"])
        if draft.action not in NEW_OBJECT_ACTIONS:
            unknown_targets = sorted(set(draft.target_ids) - known)
            if unknown_targets:
                raise TransactionError("unknown target IDs: %s" % unknown_targets, "DANGLING_TARGET")
        parent = draft.parent_event_id
        if parent is None:
            parent = before.event_head
        elif parent != before.event_head:
            raise TransactionError("parent_event_id does not match event_head", "PARENT_EVENT_MISMATCH")
        created_at = draft.created_at or self._clock()
        event = self._make_event(draft, parent, created_at)
        self._hard_conflict_check(event)

        input_hash = canonical_sha256_validated(business_payload(before))
        dependency = self._dependency_closure(before, draft.target_ids)
        working = business_payload(before)
        applied = self._apply_action(working, event, index)

        lineage = dict(working["artifacts"].get(LINEAGE_KEY) or {"links": [], "retired_ids": []})
        lineage["links"] = list(lineage.get("links") or []) + applied["new_links"]
        lineage["retired_ids"] = sorted(set(lineage.get("retired_ids") or []) | set(applied["newly_retired"]))
        working["artifacts"][LINEAGE_KEY] = lineage

        candidate_business = copy.deepcopy(working)
        candidate_business["project_version"] = before.project_version + 1
        candidate_business["event_head"] = event["event_id"]
        after_hash = canonical_sha256_validated(candidate_business)
        diff = _make_diff(business_payload(before), candidate_business)

        undo_entry = {
            "event_id": event["event_id"],
            "kind": "edit",
            "action": event["action"],
            "before": business_payload(before),
            "after": copy.deepcopy(candidate_business),
        }
        record = self._engine_record(
            event,
            result_version=candidate_business["project_version"],
            input_hash_before=input_hash,
            input_hash_after=after_hash,
            applied={"changed_ids": applied["changed_ids"], "reason_codes": applied["reason_codes"]},
            dependency=dependency,
            warnings=applied["warnings"],
        )
        events_new = self.events + [record]
        undo_new = self.undo_stack + [undo_entry]
        redo_new: List[Dict[str, Any]] = []
        candidate_project = self._assemble(candidate_business, events_new, undo_new, redo_new)
        self._validate_candidate(candidate_project)

        # W157 / D-1: the preview capability token is session-scoped, but its
        # derivation is deterministic too, so a replay allocates the same token
        # sequence.  make_id keeps live tokens unique without a random nonce.
        token = make_id("preview", {
            "project_id": before.project_id,
            "input_hash": input_hash,
            "base_version": draft.base_version,
            "event_id": event["event_id"],
            "sequence": len(self._tokens),
        }, set(self._tokens))
        preview = Preview(
            token=token,
            base_version=draft.base_version,
            input_hash=input_hash,
            event=event,
            dependency=dependency,
            diff=diff,
            candidate_business=candidate_business,
            candidate_project=candidate_project,
            events_new=events_new,
            undo_new=undo_new,
            redo_new=redo_new,
            created_at=created_at,
            contract_action=record["contract_action"],
            candidate_business_hash=after_hash,
            geometry_preview=copy.deepcopy(applied.get("geometry_preview") or {}),
            warnings=list(applied.get("warnings") or []),
        )
        self._tokens[token] = {"preview": preview, "used": False, "created": time.monotonic()}
        # Preview does not touch self.project, self.events or either stack.
        return preview

    def discard(self, preview_token: str) -> bool:
        """Release an unused preview token.

        A discarded token can never be committed, and it stops occupying its
        base version in the hard-edit conflict check.  Without this, a user
        cancel left the engine's stale sibling in place and the next unrelated
        hard edit at the same base version was refused with HARD_EDIT_CONFLICT
        (measured in the PF05 browser acceptance run).
        """
        entry = self._tokens.get(preview_token)
        if entry is None or entry["used"] or entry.get("discarded"):
            return False
        entry["discarded"] = True
        return True

    def commit(
        self,
        preview_token: str,
        base_version: int,
        *,
        fail_stage: Optional[str] = None,
    ) -> TransactionResult:
        # Version is re-checked first: a stale write can never overwrite a newer one.
        check_base_version(base_version, self.project.project_version)
        entry = self._tokens.get(preview_token)
        if entry is None:
            raise PreviewTokenError("FORGED_OR_UNKNOWN_TOKEN")
        if entry.get("discarded"):
            raise PreviewTokenError("CANCELLED_TOKEN")
        if entry["used"]:
            raise PreviewTokenError("REPLAYED_TOKEN")
        if time.monotonic() - entry["created"] > self._ttl:
            raise PreviewTokenError("EXPIRED_TOKEN")
        preview = entry["preview"]
        if preview.base_version != base_version:
            raise PreviewTokenError("TOKEN_VERSION_MISMATCH")
        current_hash = canonical_sha256(business_payload(self.project))
        if current_hash != preview.input_hash:
            raise PreviewTokenError("INPUT_HASH_CHANGED_SINCE_PREVIEW")
        if fail_stage in ("before_snapshot_build", "after_snapshot_build"):
            entry["used"] = True
            raise PartialWriteError(str(fail_stage))

        candidate = preview.candidate_project
        self._validate_candidate(candidate)
        if fail_stage == "before_state_assign":
            entry["used"] = True
            raise PartialWriteError("before_state_assign")
        record = preview.events_new[-1]
        self.project = candidate
        self.events = list(preview.events_new)
        self.undo_stack = list(preview.undo_new)
        self.redo_stack = list(preview.redo_new)
        entry["used"] = True
        self._trace.append({"kind": "commit", "receipt": record})
        return TransactionResult(
            accepted=True,
            project_version=candidate.project_version,
            event_id=preview.event["event_id"],
            changed_ids=list(record["changed_ids"]),
            affected_ids=list(record["affected_ids"]),
            reason_codes=list(record["reason_codes"]),
            reason_paths=copy.deepcopy(record["reason_paths"]),
            warnings=list(record["warnings"]),
            diff=copy.deepcopy(preview.diff),
            content_hash=candidate.to_dict()["content_hash"],
            contract_action=preview.contract_action,
        )

    # -- undo / redo ----------------------------------------------------------

    def undo(self) -> UndoResult:
        if not self.undo_stack:
            raise TransactionError("undo stack is empty", "UNDO_EMPTY")
        entry = self.undo_stack[-1]
        current_business = business_payload(self.project)
        restored = copy.deepcopy(entry["before"])
        new_version = int(current_business["project_version"]) + 1
        event_id = make_id("event", {
            "project_id": self.project.project_id,
            "base_version": self.project.project_version,
            "sequence": len(self.events),
            "action": "UNDO",
            "undone_event_id": entry["event_id"],
        }, {item["event_id"] for item in self.events})
        restored["project_version"] = new_version
        restored["event_head"] = event_id
        event = {
            "event_id": event_id,
            "project_id": self.project.project_id,
            "base_version": self.project.project_version,
            "parent_event_id": self.project.event_head,
            "actor": "pf03_undo",
            "action": "UNDO",
            "target_ids": [entry["event_id"]],
            "payload": {"undone_event_id": entry["event_id"]},
            "tolerance": 0.0,
            "hardness": "hard",
            "created_at": self._clock(),
        }
        record = self._engine_record(
            event,
            result_version=new_version,
            input_hash_before=canonical_sha256(current_business),
            input_hash_after=canonical_sha256(restored),
            applied={"changed_ids": [entry["event_id"]], "reason_codes": ["UNDO_RESTORE_BUSINESS_STATE"]},
            dependency={"closure": [entry["event_id"]], "reason_paths": []},
            warnings=["UNDO_RESTORES_GEOMETRY_RELATIONS_PROVENANCE_PROTECTION_AND_LINEAGE"],
        )
        events_new = self.events + [record]
        # Pop the undone entry and move it to the redo stack.  Appending an
        # "undo of undo" entry to the undo stack here would make consecutive
        # undos oscillate between two states instead of walking backwards.
        undo_new = list(self.undo_stack[:-1])
        redo_new = self.redo_stack + [{
            "event_id": entry["event_id"],
            "expected_after": copy.deepcopy(entry["after"]),
            "expected_event_count": len(self.events) + 1,
            "expected_semantic_hash": _semantic_hash(entry["before"]),
            "kind": "redo_of_" + str(entry.get("action", "edit")),
        }]
        candidate = self._assemble(restored, events_new, undo_new, redo_new)
        self._validate_candidate(candidate)
        self.project = candidate
        self.events = events_new
        self.undo_stack = undo_new
        self.redo_stack = redo_new
        self._trace.append({"kind": "undo", "event_id": event_id})
        return UndoResult(
            kind="undo",
            project_version=new_version,
            event_id=event_id,
            restored_semantic_hash=_semantic_hash(restored),
            identity_restored=True,
            recomputed=False,
            warnings=["UNDO_IS_A_NEW_VERSION; the old project version is never overwritten"],
            content_hash=candidate.to_dict()["content_hash"],
        )

    def redo(self) -> UndoResult:
        if not self.redo_stack:
            raise TransactionError("redo stack is empty", "REDO_EMPTY")
        entry = self.redo_stack[-1]
        current_business = business_payload(self.project)
        if len(self.events) != entry["expected_event_count"]:
            raise RecomputeRequiredError(
                "context changed after undo (%d events, expected %d); PF03 has no joint solver to recompute"
                % (len(self.events), entry["expected_event_count"])
            )
        if _semantic_hash(current_business) != entry["expected_semantic_hash"]:
            raise RecomputeRequiredError("semantic context changed after undo; recomputation is not implemented")
        restored = copy.deepcopy(entry["expected_after"])
        new_version = int(current_business["project_version"]) + 1
        event_id = make_id("event", {
            "project_id": self.project.project_id,
            "base_version": self.project.project_version,
            "sequence": len(self.events),
            "action": "REDO",
            "redone_event_id": entry["event_id"],
        }, {item["event_id"] for item in self.events})
        restored["project_version"] = new_version
        restored["event_head"] = event_id
        event = {
            "event_id": event_id,
            "project_id": self.project.project_id,
            "base_version": self.project.project_version,
            "parent_event_id": self.project.event_head,
            "actor": "pf03_redo",
            "action": "REDO",
            "target_ids": [entry["event_id"]],
            "payload": {"redone_event_id": entry["event_id"]},
            "tolerance": 0.0,
            "hardness": "hard",
            "created_at": self._clock(),
        }
        record = self._engine_record(
            event,
            result_version=new_version,
            input_hash_before=canonical_sha256(current_business),
            input_hash_after=canonical_sha256(restored),
            applied={"changed_ids": [entry["event_id"]], "reason_codes": ["REDO_IDENTITY_RESTORE"]},
            dependency={"closure": [entry["event_id"]], "reason_paths": []},
            warnings=["REDO_REQUIRES_IDENTICAL_CONTEXT; changed context is RECOMPUTE_REQUIRED, never silently recomputed"],
        )
        events_new = self.events + [record]
        undo_new = self.undo_stack + [{
            "event_id": event_id,
            "kind": "redo",
            "before": current_business,
            "after": copy.deepcopy(restored),
        }]
        redo_new = self.redo_stack[:-1]
        candidate = self._assemble(restored, events_new, undo_new, redo_new)
        self._validate_candidate(candidate)
        self.project = candidate
        self.events = events_new
        self.undo_stack = undo_new
        self.redo_stack = redo_new
        self._trace.append({"kind": "redo", "event_id": event_id})
        return UndoResult(
            kind="redo",
            project_version=new_version,
            event_id=event_id,
            restored_semantic_hash=_semantic_hash(restored),
            identity_restored=True,
            recomputed=False,
            warnings=["REDO restored a byte-identical recorded result; recomputed=false"],
            content_hash=candidate.to_dict()["content_hash"],
        )


__all__ = [
    "CONTRACT_ADDENDUM_ACTIONS", "EXTENDED_EDIT_ACTIONS", "EditDraft",
    "EditEngine", "ENGINE_ONLY_ACTIONS", "PartialWriteError",
    "Preview", "PreviewTokenError", "RecomputeRequiredError", "SUPPORTED_ACTIONS",
    "TransactionError", "TransactionResult", "UndoResult",
    "build_synthetic_project", "synthetic_fragment", "synthetic_relation",
    "synthetic_surface",
]
