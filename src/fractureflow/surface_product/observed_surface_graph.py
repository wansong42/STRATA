# -*- coding: utf-8 -*-
"""PF03: persistent identity, references, lineage and structural invariants.

Scope boundary (hand-off with PF02)
----------------------------------
This module owns ONLY the identity/reference/invariant part of the observed
surface graph: stable IDs, cross-object reference integrity, split/merge
lineage, order normalisation and the invariant report.  Finite geometric
predicates, candidate generation and spatial indexing belong to PF02's
``geometry_kernel.py`` and are deliberately NOT implemented here.  No infinite
plane is ever intersected and no hidden evaluation reference is consulted.

PF03 creates this file; PF02 may extend it later.  Nothing in this module
modifies an existing PF00 contract file.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .contracts import (
    ContractError,
    ProjectSnapshot,
    canonical_sha256,
    stable_id,
)

# Artifact keys owned by the edit engine.  They are excluded from the
# "business state" so that undo snapshots never contain themselves.
ENGINE_ARTIFACT_KEYS = frozenset({"event_log", "undo_stack", "redo_stack"})
LINEAGE_KEY = "lineage"
SOLVER_GROUPS_KEY = "solver_groups"
LINEAGE_KINDS = frozenset({
    "split", "merge", "delete", "add",
    # W151: real fragment-level and support-level provenance.  These are
    # additive kinds; no existing kind or recorded link changes meaning.
    "fragment_split", "support_reselection", "create_from_support",
})


@dataclasses.dataclass(frozen=True)
class InvariantReport:
    """A non-vacuous structural audit with explicit enumeration denominators."""

    ok: bool
    issues: List[Dict[str, Any]]
    counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "issues": list(self.issues), "counts": dict(self.counts)}

    @property
    def enumerated_total(self) -> int:
        return int(self.counts["enumerated_total"])


def _issue(code: str, object_kind: str, object_id: str, detail: str) -> Dict[str, Any]:
    return {"code": code, "object_kind": object_kind, "object_id": object_id, "detail": detail}


def business_payload(project: ProjectSnapshot) -> Dict[str, Any]:
    """Serializable project state without content_hash and engine bookkeeping.

    Undo/redo records store this projection.  Keeping engine metadata out of it
    prevents a snapshot from recursively containing its own undo stack.
    """
    payload = project.to_dict()
    payload.pop("content_hash", None)
    artifacts = dict(payload.get("artifacts") or {})
    for key in ENGINE_ARTIFACT_KEYS:
        artifacts.pop(key, None)
    payload["artifacts"] = artifacts
    return payload


def normalise_order(project: ProjectSnapshot) -> ProjectSnapshot:
    """Sort top-level ID-addressed collections, preserving every inner array.

    Display order is not identity.  Inner arrays (member IDs, polygon rings,
    support indices) stay untouched because coordinate/sequence order is
    semantically meaningful; only dictionary-addressed collections are sorted.
    """
    replacements: Dict[str, Any] = {
        "observations": sorted(project.observations, key=lambda item: item.observation_id),
        "fragments": sorted(project.fragments, key=lambda item: item.fragment_id),
        "surfaces": sorted(project.surfaces, key=lambda item: item.surface_id),
        "relations": sorted(project.relations, key=lambda item: item.relation_id),
        "protected_regions": sorted(project.protected_regions, key=lambda item: item.region_id),
        "content_hash": None,
    }
    return dataclasses.replace(project, **replacements)


def _sorted_payload_list(items: Iterable[Mapping[str, Any]], id_field: str) -> List[Dict[str, Any]]:
    return sorted((dict(item) for item in items), key=lambda item: str(item[id_field]))


def normalise_payload_order(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Same normalisation as :func:`normalise_order`, on a plain project dict."""
    result = dict(payload)
    result.pop("content_hash", None)
    for key, id_field in (
        ("observations", "observation_id"),
        ("fragments", "fragment_id"),
        ("surfaces", "surface_id"),
        ("relations", "relation_id"),
        ("protected_regions", "region_id"),
    ):
        result[key] = _sorted_payload_list(result[key], id_field)
    return result


def canonical_order_hash(project: ProjectSnapshot) -> str:
    """Hash that is invariant to display order of ID-addressed collections.

    Equality before and after sorting proves array order is not identity;
    inner arrays and the append-only event log remain order-sensitive.
    """
    return canonical_sha256(normalise_payload_order(project.to_dict()))


def reference_index(project: ProjectSnapshot) -> Dict[str, Any]:
    """Return an explicit, finite index of every object ID in the graph."""
    payload = project.to_dict()
    fragments = {item["fragment_id"] for item in payload["fragments"]}
    surfaces = {item["surface_id"] for item in payload["surfaces"]}
    relations = {item["relation_id"] for item in payload["relations"]}
    observations = {item["observation_id"] for item in payload["observations"]}
    rois = {roi["roi_id"] for observation in payload["observations"] for roi in observation["roi"]}
    protected = {item["region_id"] for item in payload["protected_regions"]}
    return {
        "fragment_ids": sorted(fragments),
        "surface_ids": sorted(surfaces),
        "relation_ids": sorted(relations),
        "observation_ids": sorted(observations),
        "roi_ids": sorted(rois),
        "protected_ids": sorted(protected),
        "geometry_ids": sorted(fragments | surfaces),
        "all_ids": sorted(fragments | surfaces | relations | protected | observations),
    }


def lineage_links(project: ProjectSnapshot) -> List[Dict[str, Any]]:
    value = (project.artifacts or {}).get(LINEAGE_KEY) or {}
    links = value.get("links") or []
    if not isinstance(links, list):
        raise ContractError("artifacts.lineage.links must be an array", "LINEAGE_INVALID")
    return [dict(item) for item in links]


def retired_ids(project: ProjectSnapshot) -> List[str]:
    value = (project.artifacts or {}).get(LINEAGE_KEY) or {}
    items = value.get("retired_ids") or []
    if not isinstance(items, list):
        raise ContractError("artifacts.lineage.retired_ids must be an array", "LINEAGE_INVALID")
    return list(items)


def with_lineage(
    project: ProjectSnapshot,
    *,
    links: Sequence[Mapping[str, Any]],
    retired: Sequence[str],
) -> ProjectSnapshot:
    artifacts = dict(project.artifacts or {})
    artifacts[LINEAGE_KEY] = {
        "links": [dict(item) for item in links],
        "retired_ids": sorted(set(retired)),
    }
    return dataclasses.replace(project, artifacts=artifacts, content_hash=None)


def make_id(
    kind: str,
    payload: Mapping[str, Any],
    existing_ids: Iterable[str],
    *,
    unordered_fields: Iterable[str] = (),
) -> str:
    """Content-derived ID that never reuses an existing ID or an array index.

    A collision is resolved by a deterministic salt, so the same input on the
    same graph always produces the same ID while a changed graph cannot silently
    alias two distinct objects.
    """
    taken = set(existing_ids)
    for salt in range(64):
        body = dict(payload)
        if salt:
            body["_collision_salt"] = salt
        candidate = stable_id(kind, body, unordered_fields=tuple(unordered_fields))
        if candidate not in taken:
            return candidate
    raise ContractError("cannot allocate a unique %s ID" % kind, "ID_ALLOCATION_FAILED")


def check_invariants(project: ProjectSnapshot) -> InvariantReport:
    """Enumerate every reference in the graph and report dangling/misplaced IDs.

    The report always carries a non-empty denominator.  An empty graph would be
    reported as an issue rather than silently passing.
    """
    payload = project.to_dict()
    issues: List[Dict[str, Any]] = []
    index = reference_index(project)
    fragments = {item["fragment_id"]: item for item in payload["fragments"]}
    surfaces = {item["surface_id"]: item for item in payload["surfaces"]}
    observations = {item["observation_id"]: item for item in payload["observations"]}
    rois = {roi["roi_id"] for observation in payload["observations"] for roi in observation["roi"]}
    geometry_ids = set(index["geometry_ids"])

    if len(set(index["fragment_ids"]) | set(index["surface_ids"])) != len(index["geometry_ids"]):
        issues.append(_issue("ID_COLLISION", "geometry", "*", "fragment and surface IDs overlap"))
    if not index["geometry_ids"]:
        issues.append(_issue("EMPTY_GEOMETRY", "geometry", "*", "no geometry objects enumerated"))

    for fragment_id, fragment in fragments.items():
        observation = observations.get(fragment["observation_id"])
        if observation is None:
            issues.append(_issue("DANGLING_OBSERVATION", "fragment", fragment_id, "observation missing"))
            continue
        support = fragment["support_ref"]
        observation_rois = {roi["roi_id"] for roi in observation["roi"]}
        if support["roi_id"] not in observation_rois:
            issues.append(_issue("DANGLING_SUPPORT_ROI", "fragment", fragment_id, support["roi_id"]))
        known_hashes = {observation["sha256"]} | {item["sha256"] for item in observation["source_files"]}
        if support["source_sha256"] not in known_hashes:
            issues.append(_issue("DANGLING_SUPPORT_HASH", "fragment", fragment_id, support["source_sha256"]))
        indices = support["indices"]
        if not indices or len(set(indices)) != len(indices):
            issues.append(_issue("INVALID_SUPPORT_INDICES", "fragment", fragment_id, "empty or duplicate"))
        if support["observation_id"] != fragment["observation_id"]:
            issues.append(_issue("SUPPORT_OBSERVATION_MISMATCH", "fragment", fragment_id, ""))

    referenced_fragments: Set[str] = set()
    for surface_id, surface in surfaces.items():
        members = surface["member_fragment_ids"]
        referenced_fragments.update(members)
        for member in members:
            if member not in fragments:
                issues.append(_issue("DANGLING_MEMBER", "surface", surface_id, member))
        if len(set(members)) != len(members):
            issues.append(_issue("DUPLICATE_MEMBER", "surface", surface_id, "duplicate member ID"))
    for fragment_id in fragments:
        if fragment_id not in referenced_fragments:
            issues.append(_issue("ORPHAN_FRAGMENT", "fragment", fragment_id, "not referenced by any surface"))

    for relation in payload["relations"]:
        relation_id = relation["relation_id"]
        for endpoint in (relation["src_id"], relation["dst_id"]):
            if endpoint not in geometry_ids:
                issues.append(_issue("DANGLING_RELATION_ENDPOINT", "relation", relation_id, endpoint))
        if relation["src_id"] == relation["dst_id"]:
            issues.append(_issue("RELATION_SELF_LOOP", "relation", relation_id, ""))
        if relation["roi_id"] not in rois:
            issues.append(_issue("DANGLING_RELATION_ROI", "relation", relation_id, relation["roi_id"]))
        if not relation["evidence_refs"]:
            issues.append(_issue("EMPTY_EVIDENCE", "relation", relation_id, ""))

    for protected in payload["protected_regions"]:
        if protected["roi_id"] not in rois:
            issues.append(_issue("DANGLING_PROTECTED_ROI", "protected_region", protected["region_id"], protected["roi_id"]))

    events = (payload["artifacts"] or {}).get("event_log") or []
    event_ids: Set[str] = set()
    for position, event in enumerate(events):
        event_id = event.get("event_id", "")
        if not event_id or event_id in event_ids:
            issues.append(_issue("DUPLICATE_EVENT_ID", "event", event_id or str(position), ""))
        event_ids.add(event_id)
        parent = event.get("parent_event_id")
        if parent is not None and parent not in event_ids and parent != event_id:
            issues.append(_issue("DANGLING_PARENT_EVENT", "event", event_id, str(parent)))
    if events:
        if payload["event_head"] != events[-1].get("event_id"):
            issues.append(_issue("EVENT_HEAD_MISMATCH", "project", payload["project_id"], str(payload["event_head"])))
    elif payload["event_head"] is not None:
        issues.append(_issue("EVENT_HEAD_WITHOUT_LOG", "project", payload["project_id"], str(payload["event_head"])))

    known_lineage_ids = geometry_ids | set(index["relation_ids"]) | set(index["protected_ids"]) | set(retired_ids(project))
    for position, link in enumerate(lineage_links(project)):
        if link.get("kind") not in LINEAGE_KINDS:
            issues.append(_issue("LINEAGE_KIND_INVALID", "lineage", str(position), str(link.get("kind"))))
        for field in ("parent_ids", "child_ids"):
            for object_id in link.get(field) or []:
                if object_id not in known_lineage_ids:
                    issues.append(_issue("DANGLING_LINEAGE_ID", "lineage", str(position), object_id))
        if not link.get("event_id"):
            issues.append(_issue("LINEAGE_EVENT_MISSING", "lineage", str(position), ""))

    counts = {
        "observations": len(payload["observations"]),
        "rois": len(rois),
        "fragments": len(payload["fragments"]),
        "surfaces": len(payload["surfaces"]),
        "relations": len(payload["relations"]),
        "protected_regions": len(payload["protected_regions"]),
        "events": len(events),
        "lineage_links": len(lineage_links(project)),
        "support_references": len(payload["fragments"]),
        "geometry_ids": len(index["geometry_ids"]),
    }
    counts["enumerated_total"] = sum(counts.values())
    if counts["enumerated_total"] <= 0:
        issues.append(_issue("EMPTY_INVARIANT_SURFACE", "project", payload["project_id"], "no objects enumerated"))
        counts["enumerated_total"] = 0
    return InvariantReport(ok=not issues, issues=issues, counts=counts)


def assert_invariants(project: ProjectSnapshot) -> InvariantReport:
    report = check_invariants(project)
    if not report.ok:
        raise ContractError(
            "structural invariants violated: %s" % [item["code"] for item in report.issues[:5]],
            "INVARIANT_VIOLATION",
        )
    return report


def id_mappings(project: ProjectSnapshot) -> Dict[str, Dict[str, str]]:
    """Return content-derived fingerprints used to detect ID misalignment."""
    payload = project.to_dict()
    result: Dict[str, Dict[str, str]] = {}
    for key, id_field in (
        ("fragments", "fragment_id"),
        ("surfaces", "surface_id"),
        ("relations", "relation_id"),
        ("protected_regions", "region_id"),
    ):
        result[key] = {
            item[id_field]: canonical_sha256(item)
            for item in payload[key]
        }
    return result


def fingerprint_differences(
    left: ProjectSnapshot, right: ProjectSnapshot
) -> Dict[str, List[str]]:
    """Compare ID -> content maps, separating changed from added/removed IDs."""
    a, b = id_mappings(left), id_mappings(right)
    differences: Dict[str, List[str]] = {}
    for key in sorted(set(a) | set(b)):
        left_map, right_map = a.get(key, {}), b.get(key, {})
        changed = sorted(set(left_map) & set(right_map) - {k for k in left_map if left_map[k] == right_map.get(k)})
        moved = sorted(set(left_map) ^ set(right_map))
        differences[key] = {"changed": changed, "added_or_removed": moved}
    return differences


__all__ = [
    "ENGINE_ARTIFACT_KEYS", "InvariantReport", "LINEAGE_KEY", "SOLVER_GROUPS_KEY",
    "assert_invariants", "business_payload", "canonical_order_hash", "check_invariants",
    "fingerprint_differences", "id_mappings", "lineage_links", "make_id",
    "normalise_order", "normalise_payload_order", "reference_index",
    "retired_ids", "with_lineage",
    "GRAPH_SCHEMA_VERSION", "ObservationSpec", "ObservedGraphBuild",
    "build_observed_graph", "save_and_reload",
]


# ---------------------------------------------------------------------------
# PF02 appended block: pure graph construction (PF03 section above untouched)
# ---------------------------------------------------------------------------
# This block ONLY projects a candidate batch plus a relation report into PF00
# contract objects.  It does not own identity policy, edit transactions, undo,
# redo or atomic storage semantics -- those remain PF03's write surface.
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from .contracts import (  # noqa: E402
    SCHEMA_VERSION,
    FiniteRegion,
    Fragment,
    ObservationBundle,
    PlaneLocal,
    ROI,
    Relation,
    SourceFile,
    SupportReference,
    Surface,
    canonical_json,
    read_json,
    save_project,
    stable_id,
)
from .geometry_kernel import CandidateBatch, RelationReport  # noqa: E402

GRAPH_SCHEMA_VERSION = "fractureflow.surface_product.observed_graph.v1"
PF02_RECOMPUTE_SCRIPT = "src/fractureflow/surface_product/observed_surface_graph.py"


@dataclasses.dataclass(frozen=True)
class ObservationSpec:
    """Explicit observation metadata.  Missing units/CRS are declared, never
    silently defaulted to metres or to a geographic CRS."""

    observation_id: str
    source_files: List[SourceFile]
    observation_sha256: str
    unit: str
    unit_status: str
    coordinate_frame: str
    local_origin: List[float]
    roi_id: str
    roi_bounds_min: List[float]
    roi_bounds_max: List[float]
    license: str
    coverage: Dict[str, Any]
    registration: Dict[str, Any]
    acquisition: Dict[str, Any]
    support_source_sha256: str
    support_storage_uri: str
    decimation_lineage: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_files": [item.to_dict() for item in self.source_files],
            "observation_sha256": self.observation_sha256,
            "unit": self.unit,
            "unit_status": self.unit_status,
            "coordinate_frame": self.coordinate_frame,
            "local_origin": list(self.local_origin),
            "roi_id": self.roi_id,
            "roi_bounds_min": list(self.roi_bounds_min),
            "roi_bounds_max": list(self.roi_bounds_max),
            "license": self.license,
            "coverage": dict(self.coverage),
            "registration": dict(self.registration),
            "acquisition": dict(self.acquisition),
            "support_source_sha256": self.support_source_sha256,
            "support_storage_uri": self.support_storage_uri,
            "decimation_lineage": dict(self.decimation_lineage),
        }


@dataclasses.dataclass
class ObservedGraphBuild:
    schema_version: str
    project_id: str
    project_path: str
    project_document_sha256: str
    reloaded_content_hash: str
    fragment_total: int
    surface_total: int
    relation_total: int
    unevaluated_pair_total: int
    instances_preserved: bool
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "project_path": self.project_path,
            "project_document_sha256": self.project_document_sha256,
            "reloaded_content_hash": self.reloaded_content_hash,
            "fragment_total": self.fragment_total,
            "surface_total": self.surface_total,
            "relation_total": self.relation_total,
            "unevaluated_pair_total": self.unevaluated_pair_total,
            "instances_preserved": bool(self.instances_preserved),
            "statistics": self.statistics,
            "recompute_script": PF02_RECOMPUTE_SCRIPT,
        }


def _pf02_observation_bundle(spec: ObservationSpec) -> ObservationBundle:
    return ObservationBundle(
        schema_version=SCHEMA_VERSION,
        observation_id=spec.observation_id,
        source_files=list(spec.source_files),
        sha256=spec.observation_sha256,
        unit=spec.unit,
        coordinate_frame=spec.coordinate_frame,
        local_origin=list(spec.local_origin),
        roi=[ROI(spec.roi_id, spec.unit, spec.coordinate_frame,
                 list(spec.roi_bounds_min), list(spec.roi_bounds_max))],
        coverage=dict(spec.coverage),
        registration=dict(spec.registration),
        acquisition=dict(spec.acquisition),
        license=spec.license,
    )


def build_observed_graph(*, batch: CandidateBatch, relations: RelationReport,
                         spec: ObservationSpec, project_id: str,
                         project_version: int = 1,
                         extra_parameters: Optional[Mapping[str, Any]] = None
                         ) -> ProjectSnapshot:
    """Project candidates into PF00 contract objects, one Surface per Fragment.

    No learned weights, no automatic merges, no inferred geological truth.
    Unevaluated pairs produce no Relation object (a missing edge is not a no).
    """
    observation = _pf02_observation_bundle(spec)
    fragments: List[Fragment] = []
    surfaces: List[Surface] = []
    surface_by_candidate: Dict[str, str] = {}

    for candidate in batch.fragments:
        support_ref = SupportReference(
            observation_id=spec.observation_id,
            source_sha256=spec.support_source_sha256,
            storage_uri=spec.support_storage_uri,
            index_kind="original_point_index",
            indices=list(candidate.support_indices),
            roi_id=spec.roi_id,
            decimation_lineage=dict(spec.decimation_lineage),
        )
        region_objects = [FiniteRegion.from_dict(item.to_dict())
                          for item in candidate.domain.regions]
        fragments.append(Fragment(
            fragment_id=candidate.candidate_id,
            observation_id=spec.observation_id,
            support_ref=support_ref,
            plane_local=PlaneLocal(list(candidate.normal_local), float(candidate.offset_local)),
            finite_regions_local=region_objects,
            residuals=[float(item) for item in candidate.residuals],
            quality=dict(candidate.quality),
            candidate_generator={
                "candidate_batch_id": batch.batch_id,
                "generator_config_hash": batch.config_hash,
            },
            version=1,
        ))
        surface_id = stable_id("surf", {"fragment_id": candidate.candidate_id})
        surface_by_candidate[candidate.candidate_id] = surface_id
        surfaces.append(Surface(
            surface_id=surface_id,
            member_fragment_ids=[candidate.candidate_id],
            interpretation_status="candidate_unconfirmed",
            finite_support=[FiniteRegion.from_dict(item.to_dict())
                            for item in candidate.domain.regions],
            provenance="DETERMINISTIC_RULE",
            version=1,
        ))

    relation_objects: List[Relation] = []
    for decision in relations.decisions:
        src = surface_by_candidate.get(decision.src_id)
        dst = surface_by_candidate.get(decision.dst_id)
        if src is None or dst is None or src == dst:
            continue
        relation_objects.append(Relation(
            relation_id=stable_id("rel", {"src": src, "dst": dst,
                                         "relation_type": decision.relation_type}),
            src_id=src, dst_id=dst,
            relation_type=decision.relation_type,
            state=decision.state,
            unknown_reason=decision.unknown_reason,
            roi_id=spec.roi_id,
            evidence_refs=list(decision.evidence_refs),
            provenance="DETERMINISTIC_RULE",
            version=1,
        ))

    parameters: Dict[str, Any] = {
        "project_mode": "manual",
        "candidate_batch_id": batch.batch_id,
        "generator_config_hash": batch.config_hash,
        "unit": spec.unit,
        "unit_status": spec.unit_status,
        "coordinate_frame": spec.coordinate_frame,
        "metric_or_geographic_conclusions_emitted": False,
        "learned_weights_used": False,
    }
    if extra_parameters:
        parameters.update(dict(extra_parameters))
    return ProjectSnapshot(
        schema_version=SCHEMA_VERSION,
        project_id=project_id,
        project_version=project_version,
        observations=[observation],
        fragments=fragments,
        surfaces=surfaces,
        relations=relation_objects,
        event_head=None,
        parameters=parameters,
        protected_regions=[],
        artifacts={},
    )


def save_and_reload(project: ProjectSnapshot, destination: Any,
                    *, unevaluated_pair_total: int) -> ObservedGraphBuild:
    """Atomic save plus a real readback through the PF00 contract deserialiser."""
    path = Path(destination)
    receipt = save_project(project, path)
    reloaded = ProjectSnapshot.from_dict(read_json(path))
    expected = project.to_dict()["content_hash"]
    actual = reloaded.to_dict()["content_hash"]
    if expected != actual or canonical_json(reloaded.to_dict()) != canonical_json(project.to_dict()):
        raise ValueError("reloaded project differs from the in-memory graph")
    members = [member for surface in project.surfaces for member in surface.member_fragment_ids]
    instances_preserved = (len(project.fragments) == len(project.surfaces)
                           and len(set(members)) == len(members)
                           and all(len(surface.member_fragment_ids) == 1
                                   for surface in project.surfaces))
    statistics = {
        "fragment_denominator": len(project.fragments),
        "surface_denominator": len(project.surfaces),
        "relation_denominator": len(project.relations),
        "candidate_denominator": len(project.fragments),
        "unevaluated_pair_total": int(unevaluated_pair_total),
        "missing_edge_is_no": False,
        "one_surface_per_fragment": True,
        "support_references_total": len(project.fragments),
        "readback_content_hash_matches": True,
    }
    return ObservedGraphBuild(
        schema_version=GRAPH_SCHEMA_VERSION,
        project_id=project.project_id,
        project_path=str(path),
        project_document_sha256=receipt["document_sha256"],
        reloaded_content_hash=actual,
        fragment_total=len(project.fragments),
        surface_total=len(project.surfaces),
        relation_total=len(project.relations),
        unevaluated_pair_total=int(unevaluated_pair_total),
        instances_preserved=bool(instances_preserved),
        statistics=statistics,
    )
