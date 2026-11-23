# -*- coding: utf-8 -*-
"""W153 evidence ledger: explicit version basis and finite-kernel recomputation.

The workbench session previously reused the immutable build's relation report
for the whole life of a session.  Geometry edits, undo, redo and solve
application changed the project but left that evidence in place, so a relation
could still be presented as supported by geometry that no longer existed.

This module separates two questions that the old code conflated:

* what a relation's current evidence *is* (RelationEvidence from
  geometry_kernel.evaluate_pair), and
* whether that evidence still describes the *current* project version
  (status plus an explicit geometric basis).

Status vocabulary (public, four values):

current      basis matches the project and the finite kernel was recomputed
stale        basis no longer matches, or recomputation failed
conflict     recomputation with the same basis produced different evidence
out_of_scope no usable build context / points / frame / endpoint

unknown remains a relation conclusion (the evidence state); it is not a
ledger status.  Stale/out-of-scope evidence is retained for side-by-side
inspection but is never handed to the solver as current support.
"""
from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.ops import unary_union

from .contracts import (
    ContractError,
    FiniteRegion,
    Fragment,
    ProjectSnapshot,
    Relation,
    Surface,
    canonical_json,
    canonical_sha256,
)
from .geometry_kernel import (
    CandidateParameters,
    FiniteDomain,
    FragmentCandidate,
    evaluate_pair,
)

LEDGER_SCHEMA = "fractureflow.surface_product.evidence_ledger.v1"
BASIS_SCHEMA = "fractureflow.surface_product.evidence_basis.v1"
SNAPSHOT_SCHEMA = "fractureflow.surface_product.evidence_snapshot.v1"
METHOD = "finite_domain_relation_kernel_v1"
RECOMPUTE_SCRIPT = "src/fractureflow/surface_product/evidence_ledger.py"

STATUS_CURRENT = "current"
STATUS_STALE = "stale"
STATUS_CONFLICT = "conflict"
STATUS_OUT_OF_SCOPE = "out_of_scope"
STATUS_VALUES = (STATUS_CURRENT, STATUS_STALE, STATUS_CONFLICT, STATUS_OUT_OF_SCOPE)

# A context is only claimed when the full source cloud can actually be read.
MAX_CONTEXT_POINTS = 2_000_000
MAX_RELATIONS = 5_000

EVIDENCE_KEYS = (
    "state",
    "unknown_reason",
    "infinite_plane_intersects",
    "finite_domain_intersects",
    "finite_shared_length",
    "support_coverage",
    "support_coverage_status",
    "support_sample_total",
    "support_sample_covered",
    "overlap_area",
    "reason_codes",
    "evidence_refs",
)

BASIS_KEYS = ("relation_id", "status", "state", "unknown_reason", "basis", "evidence", "error_code")


class EvidenceUnavailable(Exception):
    """The pair cannot be evaluated from the declared source context."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


class EvidenceComputationError(Exception):
    """The pair was attempted and failed; never reported as current."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_content_hash(project: ProjectSnapshot) -> str:
    return project.compute_content_hash()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_frame(frame: Mapping[str, Any]) -> Dict[str, Any]:
    normal = np.asarray(frame.get("normal"), dtype=np.float64)
    origin = np.asarray(frame.get("origin"), dtype=np.float64)
    u = np.asarray(frame.get("basis_u"), dtype=np.float64)
    v = np.asarray(frame.get("basis_v"), dtype=np.float64)
    if normal.shape != (3,) or origin.shape != (3,) or u.shape != (3,) or v.shape != (3,):
        raise ContractError("evidence frame vectors must be 3D", "EVIDENCE_FRAME_INVALID")
    if not (np.all(np.isfinite(normal)) and np.all(np.isfinite(origin))
            and np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ContractError("evidence frame is not finite", "EVIDENCE_FRAME_INVALID")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ContractError("evidence frame normal is degenerate", "EVIDENCE_FRAME_INVALID")
    normal = normal / norm
    u = u - normal * float(u @ normal)
    if float(np.linalg.norm(u)) <= 1e-12:
        raise ContractError("evidence frame basis_u is parallel to normal", "EVIDENCE_FRAME_INVALID")
    u = u / float(np.linalg.norm(u))
    v = np.cross(normal, u)
    v = v / float(np.linalg.norm(v))
    return {
        "normal": [float(item) for item in normal],
        "origin": [float(item) for item in origin],
        "basis_u": [float(item) for item in u],
        "basis_v": [float(item) for item in v],
        "offset": float(normal @ origin),
    }


def frame_for_fragment(fragment: Fragment, fallback: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    quality = dict(fragment.quality or {})
    stored = quality.get("frame_local")
    if isinstance(stored, Mapping) and stored:
        try:
            return _unit_frame(stored)
        except ContractError:
            pass
    if fallback:
        try:
            return _unit_frame(fallback)
        except ContractError:
            return None
    return None


def frame_for_surface(surface_geometry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    stored = surface_geometry.get("frame_local")
    if isinstance(stored, Mapping) and stored:
        try:
            return _unit_frame(stored)
        except ContractError:
            return None
    return None


def _fragment_identity(fragment: Fragment) -> Dict[str, Any]:
    support = fragment.support_ref
    quality = dict(fragment.quality or {})
    return {
        "kind": "fragment",
        "fragment_id": fragment.fragment_id,
        "observation_id": fragment.observation_id,
        "support": {
            "observation_id": support.observation_id,
            "source_sha256": support.source_sha256,
            "storage_uri": support.storage_uri,
            "index_kind": support.index_kind,
            "indices": [int(item) for item in support.indices],
            "roi_id": support.roi_id,
            "decimation_lineage": copy.deepcopy(support.decimation_lineage),
        },
        "plane_local": fragment.plane_local.to_dict(),
        "finite_regions_local": [item.to_dict() for item in fragment.finite_regions_local],
        "frame_local": copy.deepcopy(quality.get("frame_local")),
        "fit": {key: quality.get(key) for key in
                ("rmse", "max_abs_residual", "plane_tolerance", "fit_method",
                 "fit_config_hash", "geometry_source")},
    }


def _surface_identity(surface: Surface, fragments: Mapping[str, Fragment],
                      surface_geometry: Mapping[str, Any],
                      fragment_digests: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "kind": "surface",
        "surface_id": surface.surface_id,
        "member_fragment_ids": list(surface.member_fragment_ids),
        "member_digests": [fragment_digests.get(member) for member in surface.member_fragment_ids],
        "finite_support": [item.to_dict() for item in surface.finite_support],
        "frame_local": copy.deepcopy(surface_geometry.get("frame_local")),
        "mode": surface_geometry.get("mode"),
    }


def geometry_identities(project: ProjectSnapshot) -> Dict[str, Dict[str, Any]]:
    """Return explicit identity payloads for every fragment and surface."""
    surface_geometry = dict((project.artifacts or {}).get("surface_geometry") or {})
    fragments = {item.fragment_id: item for item in project.fragments}
    identities: Dict[str, Dict[str, Any]] = {}
    fragment_digests: Dict[str, str] = {}
    for fragment in project.fragments:
        payload = _fragment_identity(fragment)
        digest = canonical_sha256(payload)
        fragment_digests[fragment.fragment_id] = digest
        identities[fragment.fragment_id] = {"kind": "fragment", "digest": digest, "payload": payload}
    for surface in project.surfaces:
        stored = surface_geometry.get(surface.surface_id) or {}
        payload = _surface_identity(surface, fragments, stored, fragment_digests)
        identities[surface.surface_id] = {
            "kind": "surface", "digest": canonical_sha256(payload), "payload": payload,
        }
    return identities


def relation_basis(project: ProjectSnapshot, relation: Relation,
                   identities: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    cache = identities if identities is not None else geometry_identities(project)
    src = cache.get(relation.src_id) or {}
    dst = cache.get(relation.dst_id) or {}
    return {
        "schema": BASIS_SCHEMA,
        "relation_type": relation.relation_type,
        "roi_id": relation.roi_id,
        "src": {"id": relation.src_id, "kind": src.get("kind"), "digest": src.get("digest")},
        "dst": {"id": relation.dst_id, "kind": dst.get("kind"), "digest": dst.get("digest")},
    }


def relation_basis_map(project: ProjectSnapshot) -> Dict[str, Dict[str, Any]]:
    identities = geometry_identities(project)
    return {item.relation_id: relation_basis(project, item, identities)
            for item in project.relations}


def geometry_digest_map(project: ProjectSnapshot) -> Dict[str, str]:
    return {object_id: str(item.get("digest")) for object_id, item in geometry_identities(project).items()}


def diff_object_ids(before: ProjectSnapshot, after: ProjectSnapshot) -> List[str]:
    """Object IDs whose geometry/membership identity changed (added/removed/changed)."""
    left = geometry_digest_map(before)
    right = geometry_digest_map(after)
    return sorted(object_id for object_id in (set(left) | set(right))
                  if left.get(object_id) != right.get(object_id))


def _domain_from_regions(regions: Sequence[FiniteRegion]) -> FiniteDomain:
    if not regions:
        raise EvidenceUnavailable("EMPTY_FINITE_DOMAIN", "endpoint has no finite support regions")
    merged = unary_union([item.to_shapely() for item in regions])
    if merged.is_empty or float(merged.area) <= 0.0:
        raise EvidenceUnavailable("EMPTY_FINITE_DOMAIN", "endpoint finite support has zero area")
    holes = sum(len(item.holes) for item in regions)
    return FiniteDomain(
        regions=list(regions),
        convex_area=float(merged.convex_hull.area),
        concave_area=float(merged.area),
        actual_support_area=float(merged.area),
        support_grid_total=0,
        support_grid_covered=0,
        support_coverage_ratio=0.0,
        hole_count=holes,
        component_count=len(regions),
        simplify_tolerance=0.0,
        simplify_area_delta=0.0,
        simplify_hausdorff=0.0,
        simplify_topology_changed=False,
        construction="project_snapshot",
        degraded=False,
        degradation_reasons=[],
    )


def _endpoint_spec(project: ProjectSnapshot, object_id: str) -> Dict[str, Any]:
    fragments = {item.fragment_id: item for item in project.fragments}
    surfaces = {item.surface_id: item for item in project.surfaces}
    if object_id in fragments:
        fragment = fragments[object_id]
        return {
            "kind": "fragment",
            "object": fragment,
            "observation_ids": [fragment.observation_id],
            "indices": [int(item) for item in fragment.support_ref.indices],
            "regions": list(fragment.finite_regions_local),
            "roi_id": fragment.support_ref.roi_id,
        }
    surface = surfaces.get(object_id)
    if surface is None:
        raise EvidenceUnavailable("ENDPOINT_MISSING", "relation endpoint %s is absent" % object_id)
    member_indices: List[int] = []
    observation_ids: Set[str] = set()
    for member in surface.member_fragment_ids:
        fragment = fragments.get(member)
        if fragment is None:
            raise EvidenceUnavailable("MEMBER_MISSING", "surface member %s is absent" % member)
        observation_ids.add(fragment.observation_id)
        member_indices.extend(int(item) for item in fragment.support_ref.indices)
    return {
        "kind": "surface",
        "object": surface,
        "observation_ids": sorted(observation_ids),
        "indices": sorted(set(member_indices)),
        "regions": list(surface.finite_support),
        "roi_id": surface.surface_id,
    }


def _combine_points(points: Mapping[str, np.ndarray],
                    parts: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, Dict[str, int], Dict[str, List[int]]]:
    observation_ids: List[str] = []
    for part in parts:
        for observation_id in part["observation_ids"]:
            if observation_id not in observation_ids:
                observation_ids.append(observation_id)
    arrays: List[np.ndarray] = []
    offsets: Dict[str, int] = {}
    total = 0
    for observation_id in observation_ids:
        array = points.get(observation_id)
        if array is None:
            raise EvidenceUnavailable(
                "SUPPORT_POINTS_UNAVAILABLE",
                "raw support points are unavailable for observation %s" % observation_id,
            )
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise EvidenceUnavailable("SUPPORT_POINTS_INVALID", "support points are not an (N, 3) array")
        if len(array) + total > MAX_CONTEXT_POINTS:
            raise EvidenceUnavailable(
                "RESOURCE_LIMIT_EXCEEDED",
                "support points exceed the declared local evidence budget",
            )
        offsets[observation_id] = total
        arrays.append(array)
        total += len(array)
    if not arrays:
        raise EvidenceUnavailable("SUPPORT_POINTS_UNAVAILABLE", "no observation points were supplied")
    combined = arrays[0] if len(arrays) == 1 else np.vstack(arrays)
    remapped: Dict[str, List[int]] = {}
    for part in parts:
        object_id = str(part["object"].fragment_id if part["kind"] == "fragment" else part["object"].surface_id)
        if len(part["observation_ids"]) != 1:
            raise EvidenceUnavailable(
                "MULTI_OBSERVATION_ENDPOINT",
                "one endpoint spans several observations; no shared index space exists",
            )
        observation_id = part["observation_ids"][0]
        shift = offsets[observation_id]
        indices = []
        for index in part["indices"]:
            if not 0 <= int(index) < total:
                raise EvidenceUnavailable("SUPPORT_INDEX_OUT_OF_RANGE", "support index is outside the source cloud")
            indices.append(int(index) + shift)
        remapped[object_id] = indices
    return combined, offsets, remapped


def _candidate_for(part: Mapping[str, Any], frame: Mapping[str, Any],
                   support_indices: Sequence[int]) -> FragmentCandidate:
    region_objects = list(part["regions"])
    domain = _domain_from_regions(region_objects)
    normal = np.asarray(frame["normal"], dtype=np.float64)
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u = np.asarray(frame["basis_u"], dtype=np.float64)
    v = np.asarray(frame["basis_v"], dtype=np.float64)
    rings: List[np.ndarray] = []
    for region in region_objects:
        rings.append(np.asarray(region.exterior, dtype=np.float64))
        rings.extend(np.asarray(hole, dtype=np.float64) for hole in region.holes)
    world = np.vstack([origin + np.outer(ring[:, 0], u) + np.outer(ring[:, 1], v)
                       for ring in rings])
    aabb_min = np.min(world, axis=0)
    aabb_max = np.max(world, axis=0)
    object_id = str(part["object"].fragment_id if part["kind"] == "fragment" else part["object"].surface_id)
    return FragmentCandidate(
        candidate_id=object_id,
        observation_id=part["observation_ids"][0] if part["observation_ids"] else "unknown",
        roi_id=str(part["roi_id"]),
        support_indices=[int(item) for item in support_indices],
        center_local=[float(item) for item in origin],
        normal_local=[float(item) for item in normal],
        offset_local=float(normal @ origin),
        basis_u=[float(item) for item in u],
        basis_v=[float(item) for item in v],
        domain=domain,
        residuals=[],
        scale_label="evidence_ledger",
        lineage=[],
        nms_suppressed=[],
        quality={"source": "evidence_ledger_projection"},
        aabb_min=[float(item) for item in aabb_min],
        aabb_max=[float(item) for item in aabb_max],
    )


def _evidence_document(decision: Any) -> Dict[str, Any]:
    payload = decision.to_dict()
    return {key: copy.deepcopy(payload.get(key)) for key in EVIDENCE_KEYS}


def _record(relation: Relation, *, status: str, state: Optional[str], unknown_reason: Optional[str],
            basis: Mapping[str, Any], evidence: Optional[Mapping[str, Any]] = None,
            origin: str, project_hash: str, error: Optional[Mapping[str, Any]] = None,
            prior: Optional[Mapping[str, Any]] = None,
            conflict: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "relation_id": relation.relation_id,
        "relation_type": relation.relation_type,
        "src_id": relation.src_id,
        "dst_id": relation.dst_id,
        "status": status,
        "state": state,
        "unknown_reason": unknown_reason,
        "basis": copy.deepcopy(dict(basis)),
        "evidence": copy.deepcopy(dict(evidence or {})),
        "origin": origin,
        "computed_at": _now_iso(),
        "project_content_hash": project_hash,
        "error": copy.deepcopy(dict(error)) if error else None,
        "error_code": (error or {}).get("code") if error else None,
        "prior": copy.deepcopy(dict(prior)) if prior else None,
        "conflict": copy.deepcopy(dict(conflict)) if conflict else None,
    }


def record_from_decision(project: ProjectSnapshot, relation: Relation,
                         decision: Mapping[str, Any], *,
                         basis: Mapping[str, Any]) -> Dict[str, Any]:
    """Wrap an immutable build decision as a current ledger record."""
    evidence = {key: copy.deepcopy(decision.get(key)) for key in EVIDENCE_KEYS}
    state = evidence.get("state")
    if state not in ("yes", "no", "unknown"):
        return unavailable_record(
            project, relation, basis, "BUILD_EVIDENCE_INVALID",
            "immutable build decision carries an invalid relation state",
        )
    return _record(relation, status=STATUS_CURRENT, state=str(state),
                   unknown_reason=evidence.get("unknown_reason"), basis=basis,
                   evidence=evidence, origin="BUILD_REPORT",
                   project_hash=project_content_hash(project))


def unavailable_record(project: ProjectSnapshot, relation: Relation,
                       basis: Mapping[str, Any], code: str, message: str) -> Dict[str, Any]:
    return _record(relation, status=STATUS_OUT_OF_SCOPE, state=None, unknown_reason=None,
                   basis=basis, origin="NO_RECOMPUTE",
                   project_hash=project_content_hash(project),
                   error={"code": str(code), "message": str(message)})


def _prior_summary(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    return {
        "status": record.get("status"), "state": record.get("state"),
        "unknown_reason": record.get("unknown_reason"),
        "basis": copy.deepcopy(record.get("basis")),
        "evidence": copy.deepcopy(record.get("evidence")),
        "computed_at": record.get("computed_at"),
        "project_content_hash": record.get("project_content_hash"),
        "error_code": record.get("error_code"),
    }


def compute_relation(project: ProjectSnapshot, relation: Relation, *,
                     frames: Mapping[str, Mapping[str, Any]],
                     points: Mapping[str, np.ndarray],
                     spacing: float, parameters: CandidateParameters,
                     basis: Optional[Mapping[str, Any]] = None,
                     prior: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate one relation with the same finite kernel as the build report."""
    basis = dict(basis or relation_basis(project, relation))
    project_hash = project_content_hash(project)
    try:
        src_spec = _endpoint_spec(project, relation.src_id)
        dst_spec = _endpoint_spec(project, relation.dst_id)
        src_frame = frames.get(relation.src_id)
        dst_frame = frames.get(relation.dst_id)
        if src_frame is None or dst_frame is None:
            raise EvidenceUnavailable(
                "FRAME_UNAVAILABLE",
                "one endpoint has no explicit frame; geometric identity cannot be projected",
            )
        combined, _offsets, remapped = _combine_points(points, [src_spec, dst_spec])
        left = _candidate_for(src_spec, src_frame, remapped[str(relation.src_id)])
        right = _candidate_for(dst_spec, dst_frame, remapped[str(relation.dst_id)])
        decision = evaluate_pair(left, right, params=parameters,
                                 spacing=float(spacing), points_local=combined)
        evidence = _evidence_document(decision)
        state = str(evidence.get("state"))
        if state not in ("yes", "no", "unknown"):
            raise EvidenceComputationError("EVIDENCE_STATE_INVALID", "finite kernel returned an unknown state")
        record = _record(relation, status=STATUS_CURRENT, state=state,
                         unknown_reason=evidence.get("unknown_reason"),
                         basis=basis, evidence=evidence, origin="LOCAL_RECOMPUTE",
                         project_hash=project_hash, prior=_prior_summary(prior))
        if (prior and prior.get("status") == STATUS_CURRENT
                and prior.get("basis") == basis
                and prior.get("origin") == "LOCAL_RECOMPUTE"):
            if prior.get("evidence") != evidence:
                record["status"] = STATUS_CONFLICT
                record["conflict"] = {
                    "previous_state": prior.get("state"),
                    "previous_evidence": copy.deepcopy(prior.get("evidence")),
                    "candidate_state": state,
                    "candidate_evidence": evidence,
                    "code": "SAME_BASIS_DIFFERENT_EVIDENCE",
                }
        return record
    except EvidenceUnavailable as exc:
        return _record(relation, status=STATUS_OUT_OF_SCOPE, state=(prior or {}).get("state"),
                       unknown_reason=(prior or {}).get("unknown_reason"), basis=basis,
                       evidence=(prior or {}).get("evidence"), origin="NO_RECOMPUTE",
                       project_hash=project_hash, prior=_prior_summary(prior),
                       error={"code": exc.code, "message": exc.message})
    except Exception as exc:  # noqa: BLE001 - failure is recorded, never promoted to current
        return _record(relation, status=STATUS_STALE, state=(prior or {}).get("state"),
                       unknown_reason=(prior or {}).get("unknown_reason"), basis=basis,
                       evidence=(prior or {}).get("evidence"), origin="RECOMPUTE_FAILED",
                       project_hash=project_hash, prior=_prior_summary(prior),
                       error={"code": "RECOMPUTE_FAILED", "message": "%s: %s" % (type(exc).__name__, exc)})


def recompute_relations(project: ProjectSnapshot, relation_ids: Iterable[str], *,
                        frames: Mapping[str, Mapping[str, Any]],
                        points: Mapping[str, np.ndarray],
                        spacing: float, parameters: CandidateParameters,
                        basis_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
                        prior: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    relation_by_id = {item.relation_id: item for item in project.relations}
    basis_map = dict(basis_map or relation_basis_map(project))
    prior = dict(prior or {})
    records: Dict[str, Dict[str, Any]] = {}
    recomputed = 0
    failed = 0
    out_of_scope = 0
    for relation_id in sorted(set(str(item) for item in relation_ids)):
        relation = relation_by_id.get(relation_id)
        if relation is None:
            continue
        record = compute_relation(
            project, relation, frames=frames, points=points, spacing=float(spacing),
            parameters=parameters, basis=basis_map.get(relation_id), prior=prior.get(relation_id),
        )
        records[relation_id] = record
        if record["status"] in (STATUS_CURRENT, STATUS_CONFLICT):
            recomputed += 1
        elif record["status"] == STATUS_STALE:
            failed += 1
        else:
            out_of_scope += 1
    return {
        "records": records,
        "requested": len(records),
        "recomputed": recomputed,
        "failed": failed,
        "out_of_scope": out_of_scope,
    }


def recompute_all(project: ProjectSnapshot, *, frames: Mapping[str, Mapping[str, Any]],
                  points: Mapping[str, np.ndarray], spacing: float,
                  parameters: CandidateParameters,
                  prior: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    return recompute_relations(project, [item.relation_id for item in project.relations],
                               frames=frames, points=points, spacing=spacing,
                               parameters=parameters, prior=prior)


def classify(project: ProjectSnapshot,
             records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mark current records whose basis no longer matches as stale."""
    basis_map = relation_basis_map(project)
    result: Dict[str, Dict[str, Any]] = {}
    for relation_id, basis in basis_map.items():
        record = dict(records.get(relation_id) or {})
        if not record:
            relation = next(item for item in project.relations if item.relation_id == relation_id)
            result[relation_id] = _record(
                relation, status=STATUS_OUT_OF_SCOPE, state=None, unknown_reason=None,
                basis=basis, origin="MISSING_RECORD", project_hash=project_content_hash(project),
                error={"code": "MISSING_RECORD", "message": "no ledger record exists for this relation"},
            )
            continue
        if record.get("status") == STATUS_CURRENT and record.get("basis") != basis:
            record["status"] = STATUS_STALE
            record["error"] = {
                "code": "BASIS_MISMATCH",
                "message": "recorded geometric basis no longer describes the current project",
            }
            record["error_code"] = "BASIS_MISMATCH"
        result[relation_id] = record
    return result


def snapshot_id(project_hash: str, records: Mapping[str, Mapping[str, Any]]) -> str:
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "method": METHOD,
        "project_content_hash": str(project_hash),
        "records": [
            {key: copy.deepcopy(record.get(key)) for key in BASIS_KEYS}
            for _relation_id, record in sorted(records.items())
        ],
    }
    return canonical_sha256(payload)


def status_counts(records: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    counts = {status: 0 for status in STATUS_VALUES}
    for record in records.values():
        status = str(record.get("status"))
        if status in counts:
            counts[status] += 1
    return counts


def evidence_snapshot(project: ProjectSnapshot,
                      records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    project_hash = project_content_hash(project)
    counts = status_counts(records)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "method": METHOD,
        "recompute_script": RECOMPUTE_SCRIPT,
        "project_id": project.project_id,
        "project_version": project.project_version,
        "project_content_hash": project_hash,
        "relation_count": len(project.relations),
        "record_count": len(records),
        "status_counts": counts,
        "snapshot_id": snapshot_id(project_hash, records),
        "measured_at": _now_iso(),
    }


def current_evidence(records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Only current records may support a solver conclusion."""
    result: Dict[str, Dict[str, Any]] = {}
    for relation_id, record in records.items():
        if record.get("status") != STATUS_CURRENT:
            continue
        evidence = copy.deepcopy(record.get("evidence") or {})
        if not evidence:
            continue
        evidence["state"] = record.get("state")
        evidence["unknown_reason"] = record.get("unknown_reason")
        result[str(relation_id)] = evidence
    return result


def ledger_payload(project: ProjectSnapshot,
                   records: Mapping[str, Mapping[str, Any]], *,
                   context_status: str, meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    snapshot = evidence_snapshot(project, records)
    return {
        "schema": LEDGER_SCHEMA,
        "project_id": project.project_id,
        "project_version": project.project_version,
        "project_content_hash": snapshot["project_content_hash"],
        "method": METHOD,
        "context_status": str(context_status),
        "status_counts": snapshot["status_counts"],
        "snapshot_id": snapshot["snapshot_id"],
        "records": [copy.deepcopy(record) for _key, record in sorted(records.items())],
        "meta": copy.deepcopy(dict(meta or {})),
        "measured_at": _now_iso(),
        "recompute_script": RECOMPUTE_SCRIPT,
    }


def records_from_payload(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = payload.get("records") or []
    if isinstance(records, Mapping):
        return {str(key): copy.deepcopy(value) for key, value in records.items()}
    return {str(item["relation_id"]): copy.deepcopy(item) for item in records}


def validate_payload(project: ProjectSnapshot, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a sidecar/export ledger against the current project.

    Returns a structured verdict; it never raises on a mismatched ledger so a
    reopened project can state exactly why the evidence had to be rebuilt.
    """
    reasons: List[str] = []
    if payload.get("schema") != LEDGER_SCHEMA:
        reasons.append("SCHEMA_MISMATCH")
    if payload.get("project_content_hash") != project_content_hash(project):
        reasons.append("PROJECT_HASH_MISMATCH")
    try:
        records = records_from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "reasons": reasons + ["RECORDS_INVALID:%s" % exc], "records": {}}
    relation_ids = {item.relation_id for item in project.relations}
    missing = sorted(relation_ids - set(records))
    extra = sorted(set(records) - relation_ids)
    if missing:
        reasons.append("RELATIONS_MISSING:%d" % len(missing))
    if extra:
        reasons.append("RELATIONS_EXTRA:%d" % len(extra))
    expected_snapshot = snapshot_id(project_content_hash(project), records)
    if payload.get("snapshot_id") != expected_snapshot:
        reasons.append("SNAPSHOT_ID_MISMATCH")
    basis_map = relation_basis_map(project)
    mismatched: List[str] = []
    for relation_id, record in records.items():
        expected = basis_map.get(relation_id)
        if expected is None:
            continue
        if record.get("status") == STATUS_CURRENT and record.get("basis") != expected:
            mismatched.append(relation_id)
    if mismatched:
        reasons.append("RECORD_BASIS_MISMATCH:%d" % len(mismatched))
    return {
        "valid": not reasons,
        "reasons": reasons,
        "records": records,
        "missing": missing,
        "extra": extra,
        "basis_mismatched": mismatched,
        "expected_snapshot_id": expected_snapshot,
    }


def normalize_context(context: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not context:
        return None
    try:
        spacing = float(context["spacing"])
        parameters = context["parameters"]
        points = context["points"]
        frames = context["frames"]
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(spacing) or spacing <= 0:
        return None
    if not isinstance(parameters, CandidateParameters):
        return None
    if not isinstance(points, Mapping) or not isinstance(frames, Mapping):
        return None
    return {
        "spacing": spacing,
        "parameters": parameters,
        "points": points,
        "frames": frames,
        "source": str(context.get("source") or "unknown"),
        "origin": list(context.get("origin") or [0.0, 0.0, 0.0]),
    }
