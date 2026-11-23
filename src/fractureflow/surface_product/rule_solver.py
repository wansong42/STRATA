# -*- coding: utf-8 -*-
"""PF04 strong rule scores and a shared, fixed-score local solver.

The scorer uses only project observations, finite geometry and optional PF02
finite-predicate evidence.  The solver reuses PF03 preview/commit transactions;
it never mutates source observations or relaxes hard edits; an unknown relation
becomes ``no`` only with available finite ``no`` evidence.  Scores are fixed.
The result is a local deterministic update, not a global-optimum or real-quality
claim.  No model weights, GPU, network or training runtime are required.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .contracts import (
    ContractError,
    EditEvent,
    ProjectSnapshot,
    ScoreBundle,
    UpdateResult,
    canonical_json,
    canonical_sha256,
    find_hard_conflicts,
)
from .edit_engine import (
    EditDraft,
    EditEngine,
    FragmentFrameProvider,
    SupportPointProvider,
    TransactionError,
)
from .observed_surface_graph import business_payload, check_invariants, reference_index

ASSOCIATION_FEATURES = (
    "normal_alignment", "plane_proximity", "support_overlap", "support_density",
    "residual_agreement", "shared_observation", "same_roi",
)
RELATION_FEATURES = (
    "finite_domain", "shared_length", "support_coverage", "coverage_available",
    "evidence_yes", "evidence_no", "stable_geometry",
)
AUTO_ACTOR = "pf04_rule_solver"
DEFAULT_TIME = "2026-11-06T00:00:00+00:00"


def _association_weights() -> Dict[str, float]:
    return {
        "normal_alignment": 1.5, "plane_proximity": 2.5, "support_overlap": 0.75,
        "support_density": 0.25, "residual_agreement": 0.25,
        "shared_observation": 0.5, "same_roi": 0.5,
    }


def _relation_weights() -> Dict[str, float]:
    return {
        "finite_domain": 1.0, "shared_length": 1.0, "support_coverage": 1.0,
        "coverage_available": 0.25, "evidence_yes": 2.5, "evidence_no": -3.0,
        "stable_geometry": 0.5,
    }


def _energy_weights() -> Dict[str, float]:
    return {"obs": 1.0, "score": 1.0, "geo": 1.0, "edit": 1.0, "protect": 1.0}


@dataclass(frozen=True)
class RuleConfig:
    """Versioned scoring and solving configuration; all fields affect its hash."""

    mode: str = "manual"
    association_weights: Dict[str, float] = field(default_factory=_association_weights)
    relation_weights: Dict[str, float] = field(default_factory=_relation_weights)
    association_bias: float = -4.0
    relation_bias: float = -1.5
    association_threshold: float = 0.75
    relation_threshold: float = 0.60
    plane_scale: float = 1.0
    residual_scale: float = 0.1
    density_scale: float = 4.0
    shared_length_scale: float = 0.1
    max_iterations: int = 4
    min_energy_improvement: float = 1e-9
    accept_ties: bool = False
    recompute_scope: str = "local"
    seed_ids: Tuple[str, ...] = ()
    created_at: str = DEFAULT_TIME
    energy_weights: Dict[str, float] = field(default_factory=_energy_weights)
    calibration: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "development_calibrated"):
            raise ContractError("unknown rule mode", "RULE_MODE")
        if self.recompute_scope not in ("local", "full"):
            raise ContractError("recompute_scope must be local or full", "RECOMPUTE_SCOPE")
        for name, expected in (
            ("association_weights", ASSOCIATION_FEATURES), ("relation_weights", RELATION_FEATURES),
            ("energy_weights", ("obs", "score", "geo", "edit", "protect")),
        ):
            value = getattr(self, name)
            if set(value) != set(expected) or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                for item in value.values()
            ):
                raise ContractError("%s has invalid feature names or values" % name, "RULE_CONFIG")
        for name in ("association_threshold", "relation_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ContractError("%s must be in [0,1]" % name, "RULE_CONFIG")
        for name in ("plane_scale", "residual_scale", "density_scale", "shared_length_scale"):
            if getattr(self, name) <= 0.0:
                raise ContractError("%s must be positive" % name, "RULE_CONFIG")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ContractError("max_iterations must be a positive integer", "RULE_CONFIG")
        if self.min_energy_improvement < 0.0:
            raise ContractError("min_energy_improvement cannot be negative", "RULE_CONFIG")
        if self.mode == "development_calibrated":
            if (self.calibration.get("partition") != "development"
                    or int(self.calibration.get("association_record_count", 0)) < 1
                    or int(self.calibration.get("relation_record_count", 0)) < 1):
                raise ContractError("calibrated rules require non-empty development-only fits", "CALIBRATION_SCOPE")
        elif self.calibration:
            raise ContractError("manual rules cannot carry a calibration claim", "CALIBRATION_SCOPE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "fractureflow.surface_product.pf04.rule_config.v1",
            "mode": self.mode,
            "association_weights": dict(self.association_weights),
            "relation_weights": dict(self.relation_weights),
            "association_bias": self.association_bias,
            "relation_bias": self.relation_bias,
            "association_threshold": self.association_threshold,
            "relation_threshold": self.relation_threshold,
            "plane_scale": self.plane_scale,
            "residual_scale": self.residual_scale,
            "density_scale": self.density_scale,
            "shared_length_scale": self.shared_length_scale,
            "max_iterations": self.max_iterations,
            "min_energy_improvement": self.min_energy_improvement,
            "accept_ties": self.accept_ties,
            "recompute_scope": self.recompute_scope,
            "seed_ids": list(self.seed_ids),
            "created_at": self.created_at,
            "energy_weights": dict(self.energy_weights),
            "calibration": copy.deepcopy(self.calibration),
        }

    def config_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ProtectionSet:
    """Explicit protected object IDs plus project ProtectedRegion objects."""

    object_ids: Tuple[str, ...] = ()
    include_project_regions: bool = True

    @classmethod
    def from_value(cls, value: Any) -> "ProtectionSet":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"object_ids", "include_project_regions"}
            if unknown:
                raise ContractError("unknown protection fields: %s" % sorted(unknown), "PROTECTION_INVALID")
            return cls(tuple(value.get("object_ids") or ()), bool(value.get("include_project_regions", True)))
        if isinstance(value, str):
            return cls((value,))
        try:
            return cls(tuple(value))
        except TypeError as exc:
            raise ContractError("protected must be an ID iterable or ProtectionSet", "PROTECTION_INVALID") from exc


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def _linear_score(features: Mapping[str, float], names: Sequence[str],
                  weights: Mapping[str, float], bias: float) -> float:
    return _sigmoid(float(bias) + sum(float(weights[name]) * float(features[name]) for name in names))


def _rms(fragment: Any) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in fragment.residuals) / max(1, len(fragment.residuals)))


def _area(fragment: Any) -> float:
    return sum(float(region.to_shapely().area) for region in fragment.finite_regions_local)


def _fit_logistic(samples: Sequence[Mapping[str, Any]], names: Sequence[str], *,
                  l2: float, max_iterations: int) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    if not samples:
        raise SystemExit("development calibration samples are empty; PASS is forbidden")
    rows, targets = [], []
    scene_ids = []
    for index, sample in enumerate(samples):
        if sample.get("partition") != "development":
            raise ContractError("calibration sample %d is not development-only" % index, "CALIBRATION_SCOPE")
        features = sample.get("features")
        if not isinstance(features, Mapping) or any(name not in features for name in names):
            raise ContractError("calibration sample lacks a complete feature vector", "CALIBRATION_FEATURES")
        target = sample.get("target")
        if isinstance(target, bool) or target not in (0, 1):
            raise ContractError("calibration target must be numeric 0 or 1", "CALIBRATION_TARGET")
        row = [float(features[name]) for name in names]
        if not all(math.isfinite(value) for value in row):
            raise ContractError("calibration features must be finite", "CALIBRATION_FEATURES")
        rows.append(row)
        targets.append(float(target))
        scene_ids.append(str(sample.get("scene_id", "development_scene_%04d" % index)))
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    regularizer[0, 0] = 0.0
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        logits = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - y) + regularizer @ beta
        variance = np.maximum(probability * (1.0 - probability), 1e-6)
        hessian = design.T @ (variance[:, None] * design) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        old_loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * l2 * np.sum(beta[1:] ** 2))
        accepted = False
        for _ in range(16):
            candidate = beta - step
            candidate_logits = np.clip(design @ candidate, -40.0, 40.0)
            new_loss = float(np.mean(np.logaddexp(0.0, candidate_logits) - y * candidate_logits)
                             + 0.5 * l2 * np.sum(candidate[1:] ** 2))
            if new_loss <= old_loss + 1e-12:
                beta = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted or float(np.linalg.norm(step)) < 1e-8:
            break
    final_logits = np.clip(design @ beta, -40.0, 40.0)
    receipt = {
        "record_count": len(samples), "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()), "scene_count": len(set(scene_ids)),
        "feature_names": list(names), "iterations": iterations,
        "loss": float(np.mean(np.logaddexp(0.0, final_logits) - y * final_logits)
                      + 0.5 * l2 * np.sum(beta[1:] ** 2)),
        "l2": float(l2), "method": "deterministic_development_only_logistic_irls",
    }
    return {name: float(beta[index + 1]) for index, name in enumerate(names)}, float(beta[0]), receipt


def fit_development_rule_config(*, association_samples: Sequence[Mapping[str, Any]],
                                relation_samples: Sequence[Mapping[str, Any]],
                                base: Optional[RuleConfig] = None, l2: float = 0.1,
                                max_iterations: int = 100) -> RuleConfig:
    """Fit both heads using only explicitly registered development samples."""
    association, association_bias, association_fit = _fit_logistic(
        association_samples, ASSOCIATION_FEATURES, l2=l2, max_iterations=max_iterations)
    relation, relation_bias, relation_fit = _fit_logistic(
        relation_samples, RELATION_FEATURES, l2=l2, max_iterations=max_iterations)
    calibration = {
        "partition": "development", "association_record_count": len(association_samples),
        "relation_record_count": len(relation_samples),
        "association_fit": association_fit, "relation_fit": relation_fit,
        "score_interpretation": "development_fitted_rule_score_not_calibrated_probability",
    }
    return dataclasses.replace(
        base or RuleConfig(), mode="development_calibrated",
        association_weights=association, relation_weights=relation,
        association_bias=association_bias, relation_bias=relation_bias,
        calibration=calibration,
    )


def pf02_evidence_by_relation(project: ProjectSnapshot, report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Adapt PF02 fragment-pair evidence to project relation IDs without reinterpreting it."""
    surface_by_fragment = {
        fragment_id: surface.surface_id
        for surface in project.surfaces for fragment_id in surface.member_fragment_ids
    }
    matches: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for decision in report.get("decisions") or []:
        src = surface_by_fragment.get(decision.get("src_id"))
        dst = surface_by_fragment.get(decision.get("dst_id"))
        if src is not None and dst is not None:
            matches.setdefault((src, dst, decision.get("relation_type")), []).append(decision)
    result: Dict[str, Dict[str, Any]] = {}
    for relation in project.relations:
        for order in ((relation.src_id, relation.dst_id), (relation.dst_id, relation.src_id)):
            key = (order[0], order[1], relation.relation_type)
            decisions = matches.get(key) or []
            if len(decisions) == 1:
                result[relation.relation_id] = dict(decisions[0])
                break
    return result


class RuleScorer:
    def __init__(self, project: ProjectSnapshot, *, config: Optional[RuleConfig] = None,
                 relation_evidence: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(project, ProjectSnapshot):
            raise ContractError("rule scoring requires ProjectSnapshot", "OBJECT_REQUIRED")
        self.project = project
        self.config = config or RuleConfig()
        self.relation_evidence = {
            str(key): value.to_dict() if hasattr(value, "to_dict") else dict(value)
            for key, value in (relation_evidence or {}).items()
        }

    def _association_rows(self) -> List[Dict[str, Any]]:
        surface_by_fragment = {
            fragment_id: surface.surface_id
            for surface in self.project.surfaces for fragment_id in surface.member_fragment_ids
        }
        rows = []
        for left, right in combinations(sorted(self.project.fragments, key=lambda item: item.fragment_id), 2):
            alignment = abs(float(np.dot(left.plane_local.normal, right.plane_local.normal)))
            plane_gap = abs(float(left.plane_local.offset - right.plane_local.offset))
            left_support = set(left.support_ref.indices)
            right_support = set(right.support_ref.indices)
            overlap = len(left_support & right_support) / max(1, min(len(left_support), len(right_support)))
            left_density = len(left_support) / max(_area(left), 1e-12)
            right_density = len(right_support) / max(_area(right), 1e-12)
            density = math.sqrt(left_density * right_density)
            features = {
                "normal_alignment": alignment,
                "plane_proximity": math.exp(-plane_gap / self.config.plane_scale),
                "support_overlap": overlap,
                "support_density": min(1.0, density / self.config.density_scale),
                "residual_agreement": math.exp(-abs(_rms(left) - _rms(right)) / self.config.residual_scale),
                "shared_observation": float(left.observation_id == right.observation_id),
                "same_roi": float(left.support_ref.roi_id == right.support_ref.roi_id),
            }
            score_id = "assoc_" + canonical_sha256([left.fragment_id, right.fragment_id])[:24]
            rows.append({
                "score_id": score_id, "src_id": left.fragment_id, "dst_id": right.fragment_id,
                "features": features,
                "same_surface": surface_by_fragment.get(left.fragment_id) == surface_by_fragment.get(right.fragment_id),
                "score": _linear_score(features, ASSOCIATION_FEATURES,
                                       self.config.association_weights, self.config.association_bias),
            })
        return rows

    def _relation_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for relation in sorted(self.project.relations, key=lambda item: item.relation_id):
            evidence = self.relation_evidence.get(relation.relation_id)
            available = evidence is not None
            evidence = evidence or {}
            coverage = evidence.get("support_coverage")
            state = evidence.get("state") if available else "NOT_EVALUATED"
            codes = [str(item) for item in evidence.get("reason_codes") or []]
            features = {
                "finite_domain": float(bool(evidence.get("finite_domain_intersects"))),
                "shared_length": min(1.0, max(0.0, float(evidence.get("finite_shared_length") or 0.0))
                                     / self.config.shared_length_scale),
                "support_coverage": float(coverage) if coverage is not None else 0.0,
                "coverage_available": float(coverage is not None),
                "evidence_yes": float(state == "yes"),
                "evidence_no": float(state == "no"),
                "stable_geometry": float("SMALL_ANGLE_INTERSECTION_LINE_UNSTABLE" not in codes),
            }
            raw_score = _linear_score(features, RELATION_FEATURES,
                                      self.config.relation_weights, self.config.relation_bias)
            score = raw_score if state in ("yes", "no") else 0.5
            rows.append({
                "score_id": relation.relation_id, "features": features,
                "evidence_available": available, "evidence_state": state,
                "unknown_reason": evidence.get("unknown_reason"),
                "score": score, "raw_score": raw_score,
            })
        return rows

    def score(self) -> ScoreBundle:
        if not self.project.fragments:
            raise SystemExit("cannot score an empty fragment collection")
        associations = self._association_rows()
        relations = self._relation_rows()
        geometry: Dict[str, Any] = {}
        for surface in self.project.surfaces:
            if len(surface.member_fragment_ids) != 1 or len(surface.finite_support) != 1:
                continue
            fragment = next(item for item in self.project.fragments if item.fragment_id == surface.member_fragment_ids[0])
            if len(fragment.finite_regions_local) != 1:
                continue
            ring = fragment.finite_regions_local[0].exterior
            if canonical_json(ring) != canonical_json(surface.finite_support[0].exterior):
                geometry[surface.surface_id] = {
                    "ring": copy.deepcopy(ring), "source_fragment_id": fragment.fragment_id,
                    "reason_code": "OBSERVED_FINITE_SUPPORT_REESTIMATE",
                }
        batch_ids = sorted({item.candidate_generator["candidate_batch_id"] for item in self.project.fragments})
        batch_id = batch_ids[0] if len(batch_ids) == 1 else "batch_" + canonical_sha256(batch_ids)[:24]
        scores = {
            **{row["score_id"]: row["score"] for row in associations},
            **{row["score_id"]: row["score"] for row in relations},
        }
        return ScoreBundle(
            model_version="pf04_%s_rules_v1" % self.config.mode,
            candidate_batch_id=batch_id,
            association_scores={row["score_id"]: row["score"] for row in associations},
            relation_scores={row["score_id"]: row["score"] for row in relations},
            geometry_deltas=geometry,
            raw_uncertainty={key: (1.0 if key in {row["score_id"] for row in relations if row["evidence_state"] not in ("yes", "no")}
                                   else 1.0 - abs(2.0 * value - 1.0)) for key, value in scores.items()},
            feature_summary={
                "rule_mode": self.config.mode, "calibration": copy.deepcopy(self.config.calibration),
                "association_rows": associations, "relation_rows": relations,
                "association_denominator": len(associations), "relation_denominator": len(relations),
                "geometry_proposal_denominator": len(geometry), "model_weights_used": False,
                "association_threshold": self.config.association_threshold,
                "relation_threshold": self.config.relation_threshold,
                "finite_evidence_policy": "PF02 evidence only; missing or unknown evidence is never no",
            },
            inference_config_hash=self.config.config_hash(),
        )


def predict_joint_scores(graph: ProjectSnapshot, events: Sequence[EditEvent],
                         context: Optional[Mapping[str, Any]] = None) -> ScoreBundle:
    """Shared inference entry point used by manual and development-calibrated rules."""
    context = context or {}
    config = context.get("rule_config")
    if config is None:
        config = RuleConfig()
    if not isinstance(config, RuleConfig):
        raise ContractError("context.rule_config must be RuleConfig", "RULE_CONFIG")
    bundle = RuleScorer(graph, config=config, relation_evidence=context.get("relation_evidence")).score()
    summary = copy.deepcopy(bundle.feature_summary)
    summary["edit_condition"] = {
        "event_count": len(events),
        "target_ids": sorted({target for event in events for target in event.target_ids}),
        "hard_event_count": sum(event.hardness == "hard" for event in events),
    }
    return dataclasses.replace(bundle, feature_summary=summary)


class _RuleTransactionEngine(EditEngine):
    """Keep PF03 transaction semantics while recording actual automatic provenance."""

    def _apply_action(self, working: Dict[str, Any], event: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
        applied = super()._apply_action(working, event, index)
        if event.get("actor") != AUTO_ACTOR:
            return applied
        if event["action"] in ("SET_RELATION", "SET_UNKNOWN"):
            for relation in working["relations"]:
                if relation["relation_id"] in event["target_ids"]:
                    relation["provenance"] = "DETERMINISTIC_RULE"
        elif event["action"] in ("EDIT_BOUNDARY", "MERGE_SURFACE"):
            changed = set(applied["changed_ids"])
            for surface in working["surfaces"]:
                if surface["surface_id"] in changed:
                    surface["provenance"] = "DETERMINISTIC_RULE"
        return applied


def _object_map(project: ProjectSnapshot) -> Dict[str, Dict[str, Any]]:
    result = {}
    for item in project.observations + project.fragments + project.surfaces + project.relations + project.protected_regions:
        values = item.to_dict()
        result[values.get("observation_id") or values.get("fragment_id") or values.get("surface_id")
               or values.get("relation_id") or values.get("region_id")] = values
    return result


def _protected_drift(before: ProjectSnapshot, after: ProjectSnapshot, protected_ids: Set[str]) -> List[str]:
    if not protected_ids:
        return []
    left, right = _object_map(before), _object_map(after)
    return sorted(object_id for object_id in protected_ids
                  if canonical_json(left.get(object_id)) != canonical_json(right.get(object_id)))


def _diff_ids(preview: Any) -> Set[str]:
    return set(preview.diff.get("changed") or []) | set(preview.diff.get("added") or []) | set(preview.diff.get("removed") or [])


def update_stop_status(result: UpdateResult) -> str:
    if not result.energy_trace:
        return "NO_ENERGY_TRACE"
    return str(result.energy_trace[-1].get("stop_status", "STOP_STATUS_MISSING"))


class RuleSolver:
    """One-shot alternating local optimizer over a fixed ScoreBundle."""

    def __init__(self, graph: ProjectSnapshot, *, config: Optional[RuleConfig] = None,
                 support_points: Optional[SupportPointProvider] = None,
                 fragment_frames: Optional[FragmentFrameProvider] = None,
                 real_geometry_required: bool = False) -> None:
        if not isinstance(graph, ProjectSnapshot):
            raise ContractError("solve_consistent_update graph must be ProjectSnapshot", "OBJECT_REQUIRED")
        self.config = config or RuleConfig()
        # Contract objects are treated as immutable; PF03 preview deep-copies the
        # business state before every mutation, so a second full parse is needless.
        self._original = graph
        self.project = graph
        # T-D: every engine this solver creates over a real project must carry the
        # same original-point provider.  Before this, the solver engine had no
        # provider, so a SELECT_SUPPORT event was applied index-only and the old
        # plane/boundary/residuals survived into the published candidate.
        self._support_points = support_points
        self._fragment_frames = fragment_frames
        self._real_geometry_required = bool(real_geometry_required)
        self.engine = self._new_engine(graph)
        self.stop_status = "NOT_STARTED"
        self._closed = False

    def _new_engine(self, project: ProjectSnapshot) -> _RuleTransactionEngine:
        return _RuleTransactionEngine.from_project(
            project, support_points=self._support_points,
            fragment_frames=self._fragment_frames,
            real_geometry_required=self._real_geometry_required,
        )

    def _energy(self, project: ProjectSnapshot, scores: ScoreBundle,
                events: Sequence[EditEvent], protection_baseline: ProjectSnapshot,
                protected_ids: Set[str]) -> Dict[str, float]:
        obs = sum(_rms(fragment) for fragment in project.fragments) / max(1, len(project.fragments))
        association_rows = scores.feature_summary.get("association_rows") or []
        surface_by_fragment = {
            fragment_id: surface.surface_id for surface in project.surfaces for fragment_id in surface.member_fragment_ids
        }
        association_energy = 0.0
        for row in association_rows:
            score = float(scores.association_scores[row["score_id"]])
            same = surface_by_fragment.get(row["src_id"]) == surface_by_fragment.get(row["dst_id"])
            association_energy += (1.0 - score) if same else score
        if association_rows:
            association_energy /= len(association_rows)
        relation_energy = 0.0
        for relation in project.relations:
            score = float(scores.relation_scores.get(relation.relation_id, 0.5))
            relation_energy += (1.0 - score) if relation.state == "yes" else score if relation.state == "no" else 0.5
        if project.relations:
            relation_energy /= len(project.relations)
        score_energy = 0.5 * (association_energy + relation_energy)
        geometry_energy = 0.0
        surfaces = {item.surface_id: item for item in project.surfaces}
        for surface_id, delta in scores.geometry_deltas.items():
            surface = surfaces.get(surface_id)
            if surface is not None and "ring" in delta:
                geometry_energy += float(canonical_json(surface.finite_support[0].exterior) != canonical_json(delta["ring"]))
        if scores.geometry_deltas:
            geometry_energy /= len(scores.geometry_deltas)
        edit_energy = self._edit_penalty(project, events)
        protect_energy = float(len(_protected_drift(protection_baseline, project, protected_ids)))
        components = {"obs": obs, "score": score_energy, "geo": geometry_energy,
                      "edit": edit_energy, "protect": protect_energy}
        components["total"] = sum(self.config.energy_weights[name] * components[name]
                                  for name in ("obs", "score", "geo", "edit", "protect"))
        return {name: float(value) for name, value in components.items()}

    @staticmethod
    def _edit_penalty(project: ProjectSnapshot, events: Sequence[EditEvent]) -> float:
        relations = {item.relation_id: item for item in project.relations}
        surfaces = {item.surface_id: item for item in project.surfaces}
        fragments = {item.fragment_id: item for item in project.fragments}
        penalty = 0.0
        for event in events:
            for target in event.target_ids:
                if event.action == "SET_RELATION" and target in relations:
                    penalty += float(relations[target].state != event.payload.get("state"))
                elif event.action == "SET_UNKNOWN" and target in relations:
                    penalty += float(relations[target].state != "unknown")
                elif event.action == "EDIT_BOUNDARY" and target in surfaces:
                    penalty += float(canonical_json(surfaces[target].finite_support[0].exterior)
                                     != canonical_json(event.payload.get("ring")))
                elif event.action == "SET_INTERPRETATION" and target in surfaces:
                    penalty += float(surfaces[target].interpretation_status != event.payload.get("interpretation_status"))
                elif event.action == "SELECT_SUPPORT" and target in fragments:
                    penalty += float(fragments[target].support_ref.indices != event.payload.get("indices"))
        return penalty

    def _receipt(self, *, accepted: bool, project: ProjectSnapshot, changed: Set[str], affected: Set[str],
                 reasons: Sequence[str], trace: List[Dict[str, Any]], conflicts: List[Dict[str, Any]],
                 warnings: Sequence[str]) -> UpdateResult:
        trace[-1]["stop_status"] = self.stop_status
        result = UpdateResult(
            accepted=accepted, project_version=project.project_version,
            changed_ids=sorted(changed), affected_ids=sorted(affected),
            reason_codes=list(dict.fromkeys(list(reasons) + ["STOP_" + self.stop_status])),
            energy_trace=trace, conflicts=conflicts,
            warnings=list(dict.fromkeys(list(warnings) + ["LOCAL_UPDATE_NOT_GLOBAL_OPTIMUM"])),
            content_hash=project.to_dict()["content_hash"],
        )
        self.project = project
        if self.engine.project is not project:
            self.engine = self._new_engine(project)
        self._closed = True
        return result

    def _conflict_receipt(self, conflicts: List[Dict[str, Any]], scores: ScoreBundle,
                          status: str, protected_ids: Set[str]) -> UpdateResult:
        self.stop_status = status
        self.project = self._original
        self.engine = self._new_engine(self._original)
        energy = self._energy(self._original, scores, (), self._original, protected_ids)
        trace = [{"step": 0, "energy": energy["total"], "energy_components": energy,
                  "score_bundle_hash": scores.content_hash(), "changed_ids": [], "reason_codes": ["FEASIBILITY_CHECK"]}]
        return self._receipt(accepted=False, project=self._original, changed=set(), affected=set(),
                             reasons=["FEASIBILITY_CHECK"], trace=trace, conflicts=conflicts, warnings=[])

    def solve(self, events: Sequence[EditEvent], scores: ScoreBundle, protected: Any = None) -> UpdateResult:
        if self._closed:
            raise ContractError("a RuleSolver is one solve round; create a new round for new scores/config", "SOLVE_ROUND_CLOSED")
        if not isinstance(scores, ScoreBundle):
            raise ContractError("scores must be ScoreBundle", "OBJECT_REQUIRED")
        events = [item if isinstance(item, EditEvent) else EditEvent.from_dict(item) for item in events]
        protection = ProtectionSet.from_value(protected)
        known = set(reference_index(self.project)["all_ids"])
        protected_ids = set(protection.object_ids)
        if protection.include_project_regions:
            protected_ids.update(item.region_id for item in self.project.protected_regions)
        unknown_protected = protected_ids - known
        if unknown_protected:
            raise ContractError("unknown protected IDs: %s" % sorted(unknown_protected), "PROTECTION_INVALID")
        if any(event.project_id != self.project.project_id for event in events):
            raise ContractError("edit event project_id mismatch", "PROJECT_MISMATCH")
        hard_conflicts = find_hard_conflicts(events)
        if hard_conflicts:
            detailed = []
            objects = _object_map(self.project)
            for conflict in hard_conflicts:
                involved = [event for event in events if event.event_id in conflict["event_ids"]]
                detailed.append({**conflict,
                    "constraint": "INCOMPATIBLE_HARD_EDITS", "object_ids": conflict["target_ids"],
                    "current_values": {target: objects.get(target) for target in conflict["target_ids"]},
                    "requested_values": [{"event_id": event.event_id, "action": event.action, "payload": event.payload}
                                         for event in involved],
                    "suggestion": "keep one hard intent or issue a later explicit version; no constraint was relaxed"})
            return self._conflict_receipt(detailed, scores, "HARD_EDIT_CONFLICT", protected_ids)
        for event in events:
            blocked = protected_ids & set(event.target_ids)
            if blocked and event.action != "UNPROTECT_REGION":
                return self._conflict_receipt([{
                    "code": "PROTECTED_OBJECT_EDIT", "event_ids": [event.event_id],
                    "object_ids": sorted(blocked), "constraint": "PROTECTED_CONTENT_IMMUTABLE",
                    "current_values": {target: _object_map(self.project).get(target) for target in sorted(blocked)},
                    "requested_values": [{"action": event.action, "payload": event.payload}],
                    "suggestion": "review protection explicitly before a separate edit round",
                }], scores, "NO_FEASIBLE_UPDATE", protected_ids)

        changed: Set[str] = set()
        affected: Set[str] = set()
        scope: Set[str] = set(self.config.seed_ids)
        reason_paths: List[Dict[str, Any]] = []
        warnings: List[str] = ["NO_MODEL_WEIGHTS_USED"]
        accepted_any = False
        for event in events:
            version = self.engine.project.project_version
            try:
                dependency = self.engine._dependency_closure(self.engine.project, event.target_ids)
                preview = self.engine.preview(EditDraft(
                    event.action, list(event.target_ids), dict(event.payload), version,
                    actor=event.actor, hardness=event.hardness, tolerance=event.tolerance,
                    created_at=event.created_at,
                ))
                drift = _protected_drift(self.engine.project, preview.candidate_project, protected_ids)
                if drift:
                    raise TransactionError("protected dependent content changed: %s" % drift, "PROTECTION_DRIFT")
                receipt = self.engine.commit(preview.token, version)
            except (TransactionError, ContractError) as exc:
                conflict = {"code": getattr(exc, "code", "TRANSACTION_REJECTED"), "event_ids": [event.event_id],
                            "object_ids": list(event.target_ids), "constraint": "PF03_TRANSACTION_OR_HARD_CONSTRAINT",
                            "current_values": None, "requested_values": [{"action": event.action, "payload": event.payload}],
                            "suggestion": "resolve the named transaction conflict in a new round", "detail": str(exc)}
                return self._conflict_receipt([conflict], scores, "NO_FEASIBLE_UPDATE", protected_ids)
            accepted_any = True
            changed.update(receipt.changed_ids)
            affected.update(receipt.affected_ids)
            scope.update(dependency["closure"])
            reason_paths.extend(dependency["reason_paths"])
            if event.action == "UNPROTECT_REGION":
                protected_ids.difference_update(event.target_ids)
                warnings.append("EXPLICIT_USER_UNPROTECT_APPLIED")

        protection_baseline = self.engine.project
        if self.config.recompute_scope == "full":
            scope = set(reference_index(self.engine.project)["all_ids"])
            warnings.append("EXPLICIT_FULL_RECOMPUTE")
        else:
            if self.config.seed_ids:
                dependency = self.engine._dependency_closure(self.engine.project, list(self.config.seed_ids))
                scope.update(dependency["closure"])
                reason_paths.extend(dependency["reason_paths"])
            warnings.append("DEPENDENCY_CLOSURE_LOCAL_RECOMPUTE")
        hard_locked = {target for event in events if event.hardness == "hard" for target in event.target_ids}
        historical = (self._original.artifacts or {}).get("event_log") or []
        for record in list(historical) + list(self.engine.events):
            if record.get("hardness") == "hard":
                hard_locked.update(record.get("target_ids") or [])

        energy = self._energy(self.engine.project, scores, events, protection_baseline, protected_ids)
        round_id = canonical_sha256({"project": self._original.to_dict()["content_hash"],
                                     "scores": scores.content_hash(), "config": self.config.to_dict()})
        trace = [{"step": 0, "energy": energy["total"], "energy_components": energy,
                  "score_bundle_hash": scores.content_hash(), "solve_round_id": round_id,
                  "changed_ids": [], "reason_codes": ["INITIAL_FEASIBLE_STATE"],
                  "recompute_scope": self.config.recompute_scope, "dependency_scope": sorted(scope),
                  "reason_paths": copy.deepcopy(reason_paths)}]
        reasons = ["FIXED_SCORE_ALTERNATING_LOCAL_SOLVE"]
        rejected: Dict[str, int] = {}
        for iteration in range(self.config.max_iterations):
            iteration_accepted = False
            for phase in ("association", "geometry", "relation"):
                proposals = self._proposals(self.engine.project, scores, phase, scope, hard_locked)
                best = None
                current_energy = trace[-1]["energy"]
                for proposal in proposals:
                    version = self.engine.project.project_version
                    try:
                        preview = self.engine.preview(EditDraft(
                            proposal["action"], proposal["target_ids"], proposal["payload"], version,
                            actor=AUTO_ACTOR, hardness="soft", created_at=self.config.created_at,
                        ))
                    except (TransactionError, ContractError) as exc:
                        rejected[getattr(exc, "code", "PROPOSAL_REJECTED")] = rejected.get(getattr(exc, "code", "PROPOSAL_REJECTED"), 0) + 1
                        continue
                    touched = _diff_ids(preview)
                    if touched & protected_ids or touched & hard_locked:
                        rejected["HARD_OR_PROTECTED_TOUCHED"] = rejected.get("HARD_OR_PROTECTED_TOUCHED", 0) + 1
                        continue
                    candidate_energy = self._energy(preview.candidate_project, scores, events,
                                                    protection_baseline, protected_ids)
                    improvement = current_energy - candidate_energy["total"]
                    allowed = improvement > self.config.min_energy_improvement or (
                        self.config.accept_ties and abs(improvement) <= self.config.min_energy_improvement
                        and canonical_json(proposal) < canonical_json(best["proposal"] if best else {"z": "z"}))
                    if not allowed:
                        rejected["ENERGY_NOT_DECREASED"] = rejected.get("ENERGY_NOT_DECREASED", 0) + 1
                        continue
                    key = (candidate_energy["total"], canonical_json(proposal))
                    if best is None or key < best["key"]:
                        best = {"key": key, "proposal": proposal, "preview": preview, "energy": candidate_energy}
                if best is None:
                    continue
                version = self.engine.project.project_version
                receipt = self.engine.commit(best["preview"].token, version)
                accepted_any = True
                iteration_accepted = True
                changed.update(receipt.changed_ids)
                affected.update(receipt.affected_ids)
                affected.update(best["preview"].dependency["closure"])
                energy = self._energy(self.engine.project, scores, events, protection_baseline, protected_ids)
                if energy["total"] > current_energy + 1e-9:
                    raise RuntimeError("internal energy monotonicity violation")
                reasons.extend([phase.upper() + "_UPDATE", best["proposal"]["reason_code"]])
                trace.append({"step": len(trace), "energy": energy["total"], "energy_components": energy,
                              "score_bundle_hash": scores.content_hash(), "solve_round_id": round_id,
                              "iteration": iteration + 1, "phase": phase,
                              "changed_ids": list(receipt.changed_ids), "affected_ids": list(receipt.affected_ids),
                              "reason_codes": list(receipt.reason_codes) + [best["proposal"]["reason_code"]],
                              "reason_paths": copy.deepcopy(best["preview"].dependency["reason_paths"]),
                              "proposal": best["proposal"], "recompute_scope": self.config.recompute_scope,
                              "dependency_scope": sorted(scope)})
            if not iteration_accepted:
                self.stop_status = "NO_IMPROVEMENT"
                break
        else:
            self.stop_status = "ITERATION_LIMIT"
        if not scope and self.config.recompute_scope == "local":
            warnings.append("EMPTY_LOCAL_DEPENDENCY_SCOPE_NO_AUTOMATIC_PROPOSALS")
        if rejected:
            warnings.append("REJECTED_PROPOSALS:" + canonical_json(rejected))
            trace[-1]["rejected_proposals"] = dict(sorted(rejected.items()))
        reasons.append("DETERMINISTIC_RECOMPUTE")
        return self._receipt(accepted=accepted_any, project=self.engine.project, changed=changed, affected=affected,
                             reasons=reasons, trace=trace, conflicts=[], warnings=warnings)

    @staticmethod
    def _proposals(project: ProjectSnapshot, scores: ScoreBundle, phase: str,
                   scope: Set[str], hard_locked: Set[str]) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        surfaces = {item.surface_id: item for item in project.surfaces}
        relations = {item.relation_id: item for item in project.relations}
        surface_by_fragment = {
            fragment_id: surface.surface_id for surface in project.surfaces for fragment_id in surface.member_fragment_ids
        }
        association_threshold = float(scores.feature_summary.get("association_threshold", 0.75))
        relation_threshold = float(scores.feature_summary.get("relation_threshold", 0.60))
        if phase == "association":
            threshold = association_threshold
            for row in scores.feature_summary.get("association_rows") or []:
                left = surface_by_fragment.get(row["src_id"])
                right = surface_by_fragment.get(row["dst_id"])
                if left is None or right is None or left == right:
                    continue
                if scores.association_scores.get(row["score_id"], 0.0) < threshold:
                    continue
                targets = sorted([left, right])
                if not set(targets).issubset(scope) or set(targets) & hard_locked:
                    continue
                if any(relation.relation_type == "same_observed_surface" and relation.state == "no"
                       and {relation.src_id, relation.dst_id} == set(targets) for relation in project.relations):
                    continue
                proposals.append({"action": "MERGE_SURFACE", "target_ids": targets, "payload": {},
                                  "reason_code": "HIGH_ASSOCIATION_SCORE", "score_id": row["score_id"]})
        elif phase == "geometry":
            for surface_id, delta in sorted(scores.geometry_deltas.items()):
                surface = surfaces.get(surface_id)
                if surface is None or "ring" not in delta or surface_id not in scope or surface_id in hard_locked:
                    continue
                if canonical_json(surface.finite_support[0].exterior) == canonical_json(delta["ring"]):
                    continue
                proposals.append({"action": "EDIT_BOUNDARY", "target_ids": [surface_id],
                                  "payload": {"ring": copy.deepcopy(delta["ring"])},
                                  "reason_code": str(delta.get("reason_code", "RULE_GEOMETRY_UPDATE"))})
        elif phase == "relation":
            rows = {row["score_id"]: row for row in scores.feature_summary.get("relation_rows") or []}
            for relation_id, row in sorted(rows.items()):
                relation = relations.get(relation_id)
                if (relation is None or relation_id not in scope or relation_id in hard_locked
                        or relation.provenance == "HUMAN_CONFIRMED" or not row.get("evidence_available")):
                    continue
                state = row.get("evidence_state")
                score = scores.relation_scores.get(relation_id, 0.5)
                if state == "yes" and score >= relation_threshold:
                    requested, reason = "yes", None
                elif state == "no" and score <= 1.0 - relation_threshold:
                    requested, reason = "no", None
                elif state == "unknown":
                    requested, reason = "unknown", row.get("unknown_reason") or "INSUFFICIENT_SUPPORT"
                else:
                    continue
                if relation.state == requested and relation.unknown_reason == reason:
                    continue
                payload = {"state": requested}
                if reason is not None:
                    payload["unknown_reason"] = reason
                proposals.append({"action": "SET_RELATION", "target_ids": [relation_id], "payload": payload,
                                  "reason_code": "PF02_FINITE_RELATION_EVIDENCE", "score_id": relation_id})
        return sorted(proposals, key=canonical_json)


def solve_consistent_update(graph: ProjectSnapshot, events: Sequence[EditEvent], scores: ScoreBundle,
                            protected: Any = None, *, config: Optional[RuleConfig] = None) -> UpdateResult:
    """Common PF04/PF07 signature; use RuleSolver directly to retrieve the project."""
    if config is not None and scores.inference_config_hash != config.config_hash():
        raise ContractError("ScoreBundle belongs to a different config; start a new scoring/solve round",
                            "SCORE_CONFIG_MISMATCH")
    return RuleSolver(graph, config=config).solve(events, scores, protected)
