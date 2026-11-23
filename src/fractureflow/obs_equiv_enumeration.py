"""Finite hypothesis-space enumeration over an independent geometry path.

The geometry primitives here are re-derived from the declared conventions and
do not import ``obs_equiv_geometry``, ``connectivity_qoi``, ``intersect_disk``
or ``topology_query``. Agreement with those paths is a reconciliation reading,
not an assumption.

Scope: exact synthetic planar-disk worlds only. No zones, surfaces or
continuous domains.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fractureflow.obs_equiv_manifest import ContractError, Disk, Scene, TwoPointQuery

TAU_SIN = 1e-8
TAU_LEN_REL = 1e-9
WEIGHT_SUM_TOL = 1e-9
GENERATION_DECLARED = "DECLARED_SCENES"
GENERATION_PRESENCE = "PRESENCE_MASKS"
CLASSIFICATION_FIXED = "FIXED_HIDDEN_PROTOTYPES"
CLASSIFICATION_VARYING = "VARYING_HIDDEN_GEOMETRY"
COVERAGE_COMPLETE = "COMPLETE_ALL_PRESENCE_MASKS"
COVERAGE_VARYING = "NOT_APPLICABLE_VARYING_HIDDEN_GEOMETRY"
COVERAGE_INCOMPLETE = "INCOMPLETE_OR_DUPLICATE"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scale(vector: Sequence[float], factor: float) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def point_disk_distance(disk: Disk, point: Sequence[float]) -> float:
    delta = _sub(point, disk.center)
    axial = _dot(delta, disk.normal)
    planar = _sub(delta, _scale(disk.normal, axial))
    radial_gap = max(_norm(planar) - disk.radius, 0.0)
    return math.sqrt(axial * axial + radial_gap * radial_gap)


def length_scale(disks: Sequence[Disk]) -> float:
    radii = sorted(disk.radius for disk in disks)
    if not radii:
        return 1.0
    middle = len(radii) // 2
    if len(radii) % 2:
        return radii[middle]
    return (radii[middle - 1] + radii[middle]) / 2.0


def disks_intersect(first: Disk, second: Disk, scale: float) -> bool:
    delta = _sub(second.center, first.center)
    cross = _cross(first.normal, second.normal)
    sine = _norm(cross)
    if sine <= TAU_SIN:
        signed_separation = _dot(delta, first.normal)
        planar = _sub(delta, _scale(first.normal, signed_separation))
        return abs(signed_separation) <= TAU_LEN_REL * scale and _norm(planar) < first.radius + second.radius
    direction = _scale(cross, 1.0 / sine)
    first_offset = _dot(delta, second.normal) / sine
    second_offset = _dot(delta, first.normal) / sine
    if abs(first_offset) >= first.radius or abs(second_offset) >= second.radius:
        return False
    projected_distance = abs(_dot(delta, direction))
    first_half_chord = math.sqrt(max(first.radius * first.radius - first_offset * first_offset, 0.0))
    second_half_chord = math.sqrt(max(second.radius * second.radius - second_offset * second_offset, 0.0))
    return projected_distance < first_half_chord + second_half_chord


def intersection_margins(first: Disk, second: Disk, scale: float) -> dict[str, Any]:
    delta = _sub(second.center, first.center)
    cross = _cross(first.normal, second.normal)
    sine = _norm(cross)
    if sine <= TAU_SIN:
        signed_separation = _dot(delta, first.normal)
        planar = _sub(delta, _scale(first.normal, signed_separation))
        branch_margin = abs(abs(signed_separation) - TAU_LEN_REL * scale)
        if abs(signed_separation) > TAU_LEN_REL * scale:
            return {"branch": branch_margin, "branch_kind": "PARALLEL_SKEW", "overlap": None}
        return {
            "branch": branch_margin,
            "branch_kind": "PARALLEL_COPLANAR",
            "overlap": abs(first.radius + second.radius - _norm(planar)),
        }
    first_offset = _dot(delta, second.normal) / sine
    second_offset = _dot(delta, first.normal) / sine
    direction = _scale(cross, 1.0 / sine)
    projected_distance = abs(_dot(delta, direction))
    overlap = min(
        abs(abs(first_offset) - first.radius),
        abs(abs(second_offset) - second.radius),
    )
    if abs(first_offset) < first.radius and abs(second_offset) < second.radius:
        first_half_chord = math.sqrt(first.radius * first.radius - first_offset * first_offset)
        second_half_chord = math.sqrt(second.radius * second.radius - second_offset * second_offset)
        overlap = min(overlap, abs(first_half_chord + second_half_chord - projected_distance))
    return {"branch": abs(sine - TAU_SIN), "branch_kind": "CROSSING", "overlap": overlap}


def decision_margins(scene: Scene, query: TwoPointQuery) -> dict[str, Any]:
    scale = length_scale(scene.disks)
    attachment_margins = [
        abs(query.attach_radius - point_disk_distance(disk, point))
        for disk in scene.disks
        for point in (query.point_a, query.point_b)
    ]
    pair_margins = [
        intersection_margins(first, second, scale)
        for index, first in enumerate(scene.disks)
        for second in scene.disks[index + 1:]
    ]
    overlap_margins = [entry["overlap"] for entry in pair_margins if entry["overlap"] is not None]
    branch_margins = [
        entry["branch"] for entry in pair_margins
        if entry["branch_kind"] in ("PARALLEL_COPLANAR", "PARALLEL_SKEW")
    ]
    physical = attachment_margins + overlap_margins
    return {
        "attachment_min": min(attachment_margins) if attachment_margins else None,
        "intersection_overlap_min": min(overlap_margins) if overlap_margins else None,
        "branch_tolerance_min": min(branch_margins) if branch_margins else None,
        "min": min(physical) if physical else None,
        "n_attachment_decisions": len(attachment_margins),
        "n_pair_decisions": len(pair_margins),
    }


def connection_witness(scene: Scene, query: TwoPointQuery) -> dict[str, Any]:
    attached_a = sorted(
        disk.object_id for disk in scene.disks
        if point_disk_distance(disk, query.point_a) <= query.attach_radius
    )
    attached_b = sorted(
        disk.object_id for disk in scene.disks
        if point_disk_distance(disk, query.point_b) <= query.attach_radius
    )
    scale = length_scale(scene.disks)
    edges = set()
    for index, first in enumerate(scene.disks):
        for second in scene.disks[index + 1:]:
            if disks_intersect(first, second, scale):
                edges.add(tuple(sorted((first.object_id, second.object_id))))
    ordered_edges = sorted(edges)
    connected = False
    reached: list[str] = []
    if attached_a and attached_b:
        adjacency: dict[str, set[str]] = {disk.object_id: set() for disk in scene.disks}
        for left, right in ordered_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        seen = set(attached_a)
        queue = sorted(attached_a)
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        reached = sorted(seen)
        connected = not seen.isdisjoint(attached_b)
    return {
        "attached_a": attached_a,
        "attached_b": attached_b,
        "edges": [list(edge) for edge in ordered_edges],
        "reached": reached,
        "connected": bool(connected),
    }


def checked_weights(weights: Sequence[float], expected_count: int) -> tuple[float, ...]:
    if isinstance(weights, (str, bytes)) or not isinstance(weights, (list, tuple)):
        raise ContractError("weights must be a sequence of numbers")
    if len(weights) != expected_count or expected_count < 1:
        raise ContractError("weights must match the hypothesis count")
    parsed = tuple(_finite(weight, "weight") for weight in weights)
    if any(weight < 0.0 for weight in parsed):
        raise ContractError("weights must be non-negative")
    total = math.fsum(parsed)
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ContractError(f"weights must sum to one, got {total!r}")
    return parsed


def normalize_weights(weights: Sequence[float], expected_count: int | None = None) -> tuple[float, ...]:
    if isinstance(weights, (str, bytes)) or not isinstance(weights, (list, tuple)):
        raise ContractError("weights must be a sequence of numbers")
    if expected_count is not None and len(weights) != expected_count:
        raise ContractError("weights must match the hypothesis count")
    if not weights:
        raise ContractError("cannot normalize an empty weight vector")
    parsed = tuple(_finite(weight, "weight") for weight in weights)
    if any(weight < 0.0 for weight in parsed):
        raise ContractError("weights must be non-negative")
    total = math.fsum(parsed)
    if total <= 0.0:
        raise ContractError("weights must have a positive total")
    return tuple(weight / total for weight in parsed)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    disks: tuple[Disk, ...]
    weight: float

    def __init__(self, hypothesis_id: str, disks: Sequence[Disk], weight: float):
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ContractError("hypothesis_id must be a non-empty string")
        parsed = tuple(disks)
        if any(not isinstance(disk, Disk) for disk in parsed):
            raise ContractError("hypothesis disks must be Disk instances")
        ids = [disk.object_id for disk in parsed]
        if len(ids) != len(set(ids)):
            raise ContractError("hypothesis disk ids must be unique")
        parsed_weight = _finite(weight, "weight")
        if parsed_weight < 0.0:
            raise ContractError("hypothesis weight must be non-negative")
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "disks", parsed)
        object.__setattr__(self, "weight", parsed_weight)

    def scene(self) -> Scene:
        return Scene(self.hypothesis_id, self.disks)

    def nonvisible_ids(self, visible_object_ids: Sequence[str]) -> frozenset[str]:
        visible = set(visible_object_ids)
        return frozenset(disk.object_id for disk in self.disks if disk.object_id not in visible)


def _id_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ContractError(f"{name} must be a sequence of strings")
    parsed = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in parsed):
        raise ContractError(f"{name} must contain non-empty strings")
    if len(parsed) != len(set(parsed)):
        raise ContractError(f"{name} must be unique")
    return parsed


@dataclass(frozen=True)
class FiniteHypothesisSpace:
    space_id: str
    visible_object_ids: tuple[str, ...]
    hidden_object_ids: tuple[str, ...]
    query: TwoPointQuery
    hypotheses: tuple[Hypothesis, ...]
    generation: str = GENERATION_DECLARED

    def __init__(
        self,
        space_id: str,
        visible_object_ids: Sequence[str],
        hidden_object_ids: Sequence[str],
        query: TwoPointQuery,
        hypotheses: Sequence[Hypothesis],
        generation: str = GENERATION_DECLARED,
    ):
        if not isinstance(space_id, str) or not space_id.strip():
            raise ContractError("space_id must be a non-empty string")
        if not isinstance(query, TwoPointQuery):
            raise ContractError("query must be a TwoPointQuery")
        if generation not in (GENERATION_DECLARED, GENERATION_PRESENCE):
            raise ContractError(f"unsupported generation: {generation!r}")
        visible = _id_tuple(visible_object_ids, "visible_object_ids")
        hidden = _id_tuple(hidden_object_ids, "hidden_object_ids")
        if set(visible) & set(hidden):
            raise ContractError("visible and hidden object ids must be disjoint")
        parsed = tuple(hypotheses)
        if not parsed:
            raise SystemExit(f"empty hypothesis set: {space_id}")
        if any(not isinstance(hypothesis, Hypothesis) for hypothesis in parsed):
            raise ContractError("hypotheses must be Hypothesis instances")
        hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in parsed]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ContractError("hypothesis ids must be unique")
        checked_weights(tuple(hypothesis.weight for hypothesis in parsed), len(parsed))
        visible_set = set(visible)
        declared = visible_set | set(hidden)
        observed_hidden: set[str] = set()
        visible_geometry: dict[str, set[tuple[Any, ...]]] = {object_id: set() for object_id in visible}
        for hypothesis in parsed:
            ids = {disk.object_id for disk in hypothesis.disks}
            missing = visible_set - ids
            if missing:
                raise ContractError(f"hypothesis {hypothesis.hypothesis_id!r} omits visible objects: {sorted(missing)}")
            unknown = ids - declared
            if unknown:
                raise ContractError(f"hypothesis {hypothesis.hypothesis_id!r} has undeclared objects: {sorted(unknown)}")
            observed_hidden |= ids - visible_set
            for disk in hypothesis.disks:
                if disk.object_id in visible_set:
                    visible_geometry[disk.object_id].add((disk.center, disk.normal, disk.radius))
        if observed_hidden != set(hidden):
            omitted = sorted(set(hidden) - observed_hidden)
            raise ContractError(f"declared hidden objects never appear in any hypothesis: {omitted}")
        varying_visible = sorted(object_id for object_id, variants in visible_geometry.items() if len(variants) != 1)
        if varying_visible:
            raise ContractError(f"visible geometry differs across hypotheses: {varying_visible}")
        object.__setattr__(self, "space_id", space_id)
        object.__setattr__(self, "visible_object_ids", visible)
        object.__setattr__(self, "hidden_object_ids", hidden)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "hypotheses", parsed)
        object.__setattr__(self, "generation", generation)
        if generation == GENERATION_PRESENCE:
            expected = 2 ** len(hidden)
            if len(parsed) != expected:
                raise ContractError(f"presence space must enumerate {expected} masks, got {len(parsed)}")
            masks = [hypothesis.nonvisible_ids(visible) for hypothesis in parsed]
            if len(set(masks)) != len(masks):
                raise ContractError("presence space contains duplicate masks")
            full = {
                frozenset(mask)
                for size in range(len(hidden) + 1)
                for mask in itertools.combinations(hidden, size)
            }
            missing = full - set(masks)
            if missing:
                raise ContractError(f"presence space misses masks: {[sorted(mask) for mask in missing]}")

    def masks(self) -> tuple[frozenset[str], ...]:
        return tuple(hypothesis.nonvisible_ids(self.visible_object_ids) for hypothesis in self.hypotheses)


@dataclass(frozen=True)
class EnumerationResult:
    space_id: str
    generation: str
    hypothesis_count: int
    hidden_object_count: int
    weights: tuple[float, ...]
    labels: tuple[bool, ...]
    posterior: float
    readings: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "generation": self.generation,
            "hypothesis_count": self.hypothesis_count,
            "hidden_object_count": self.hidden_object_count,
            "weights": list(self.weights),
            "labels": list(self.labels),
            "posterior": self.posterior,
            "readings": list(self.readings),
        }


def evaluate_space(space: FiniteHypothesisSpace) -> EnumerationResult:
    if not isinstance(space, FiniteHypothesisSpace):
        raise ContractError("space must be a FiniteHypothesisSpace")
    if not space.hypotheses:
        raise SystemExit(f"empty hypothesis set: {space.space_id}")
    readings = []
    for hypothesis in space.hypotheses:
        scene = hypothesis.scene()
        witness = connection_witness(scene, space.query)
        margins = decision_margins(scene, space.query)
        readings.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "weight": hypothesis.weight,
            "label": witness["connected"],
            "witness": witness,
            "min_decision_margin": margins["min"],
            "decision_margins": margins,
        })
    labels = tuple(bool(reading["label"]) for reading in readings)
    posterior = math.fsum(
        reading["weight"] for reading in readings if reading["label"]
    )
    return EnumerationResult(
        space_id=space.space_id,
        generation=space.generation,
        hypothesis_count=len(readings),
        hidden_object_count=len(space.hidden_object_ids),
        weights=tuple(reading["weight"] for reading in readings),
        labels=labels,
        posterior=posterior,
        readings=tuple(readings),
    )


def build_presence_space(
    space_id: str,
    visible_disks: Sequence[Disk],
    hidden_disks: Sequence[Disk],
    hidden_probabilities: Mapping[str, float] | Sequence[float],
    query: TwoPointQuery,
) -> FiniteHypothesisSpace:
    visible = tuple(visible_disks)
    hidden = tuple(hidden_disks)
    if any(not isinstance(disk, Disk) for disk in visible + hidden):
        raise ContractError("disks must be Disk instances")
    visible_ids = [disk.object_id for disk in visible]
    hidden_ids = [disk.object_id for disk in hidden]
    if len(visible_ids) != len(set(visible_ids)) or len(hidden_ids) != len(set(hidden_ids)):
        raise ContractError("disk ids must be unique")
    if set(visible_ids) & set(hidden_ids):
        raise ContractError("visible and hidden disk ids must be disjoint")
    if isinstance(hidden_probabilities, Mapping):
        if set(hidden_probabilities) != set(hidden_ids):
            raise ContractError("hidden_probabilities must cover exactly the hidden disk ids")
        probabilities = tuple(hidden_probabilities[object_id] for object_id in hidden_ids)
    else:
        probabilities = tuple(hidden_probabilities)
    if len(probabilities) != len(hidden):
        raise ContractError("hidden_probabilities must match the hidden disk count")
    parsed_probabilities = tuple(_finite(value, "hidden_probability") for value in probabilities)
    if any(not 0.0 < probability < 1.0 for probability in parsed_probabilities):
        raise ContractError("hidden probabilities must be strictly between zero and one")
    masks = list(itertools.product((False, True), repeat=len(hidden)))
    raw_weights = [
        math.prod(
            probability if present else 1.0 - probability
            for probability, present in zip(parsed_probabilities, mask)
        )
        for mask in masks
    ]
    weights = normalize_weights(raw_weights, len(masks))
    hypotheses = []
    for mask, weight in zip(masks, weights):
        disks = visible + tuple(
            disk for disk, present in zip(hidden, mask) if present
        )
        suffix = "".join("1" if present else "0" for present in mask)
        hypotheses.append(Hypothesis(f"{space_id}__mask_{suffix}", disks, weight))
    return FiniteHypothesisSpace(
        space_id,
        visible_ids,
        hidden_ids,
        query,
        hypotheses,
        GENERATION_PRESENCE,
    )


def audit_hypothesis_structure(
    scenes: Sequence[Scene],
    visible_object_ids: Sequence[str],
) -> dict[str, Any]:
    parsed_scenes = tuple(scenes)
    if not parsed_scenes:
        raise SystemExit("empty scene set")
    if any(not isinstance(scene, Scene) for scene in parsed_scenes):
        raise ContractError("scenes must be Scene instances")
    visible = _id_tuple(visible_object_ids, "visible_object_ids")
    visible_set = set(visible)
    visible_variants: dict[str, set[tuple[Any, ...]]] = {object_id: set() for object_id in visible}
    hidden_variants: dict[str, set[tuple[Any, ...]]] = {}
    hidden_present: dict[str, int] = {}
    masks: list[tuple[str, ...]] = []
    for scene in parsed_scenes:
        ids = {disk.object_id for disk in scene.disks}
        missing = visible_set - ids
        if missing:
            raise ContractError(f"scene {scene.scene_id!r} omits visible objects: {sorted(missing)}")
        hidden_here = ids - visible_set
        masks.append(tuple(sorted(hidden_here)))
        for disk in scene.disks:
            signature = (disk.center, disk.normal, disk.radius)
            if disk.object_id in visible_set:
                visible_variants[disk.object_id].add(signature)
            else:
                hidden_variants.setdefault(disk.object_id, set()).add(signature)
                hidden_present[disk.object_id] = hidden_present.get(disk.object_id, 0) + 1
    hidden_ids = sorted(hidden_variants)
    varying = any(len(variants) > 1 for variants in hidden_variants.values())
    visible_invariant = all(len(variants) == 1 for variants in visible_variants.values())
    expected_masks = {
        frozenset(mask)
        for size in range(len(hidden_ids) + 1)
        for mask in itertools.combinations(hidden_ids, size)
    }
    observed_masks = {frozenset(mask) for mask in masks}
    missing_masks = sorted((sorted(mask) for mask in expected_masks - observed_masks), key=lambda mask: (len(mask), mask))
    duplicate_masks = sorted(
        (list(mask) for mask in observed_masks if masks.count(tuple(sorted(mask))) > 1),
        key=lambda mask: (len(mask), mask),
    )
    if varying:
        coverage = COVERAGE_VARYING
    elif not missing_masks and not duplicate_masks and len(parsed_scenes) == len(expected_masks):
        coverage = COVERAGE_COMPLETE
    else:
        coverage = COVERAGE_INCOMPLETE
    return {
        "scene_count": len(parsed_scenes),
        "visible_object_ids": list(visible),
        "hidden_object_ids": hidden_ids,
        "visible_geometry_invariant": visible_invariant,
        "visible_geometry_variants": {
            object_id: len(variants) for object_id, variants in sorted(visible_variants.items())
        },
        "hidden_geometry_variants": {
            object_id: len(variants) for object_id, variants in sorted(hidden_variants.items())
        },
        "classification": CLASSIFICATION_VARYING if varying else CLASSIFICATION_FIXED,
        "mask_coverage": coverage,
        "expected_mask_count": len(expected_masks),
        "observed_mask_count": len(observed_masks),
        "missing_masks": missing_masks,
        "duplicate_masks": duplicate_masks,
        "hidden_ids_never_present": sorted(object_id for object_id in hidden_ids if not hidden_present.get(object_id)),
    }
