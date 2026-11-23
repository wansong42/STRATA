# -*- coding: utf-8 -*-
"""PF05 local workbench application service.

The service joins the existing public contracts, edit transactions, project
store and PF04 rule solver without reimplementing their geometry or energy
rules.  Long solves run in isolated processes: timeout and cancellation really
terminate the worker and never publish its candidate project.  Only an explicit
apply of a successful job, guarded by base version and content hash, can change
the current session.

This is a development vertical slice, not a real-data product acceptance claim.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import shapely

from .contracts import (
    LATEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    ContractError,
    EditEvent,
    ProjectSnapshot,
    VersionConflictError,
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    check_base_version,
    file_sha256,
    io_path,
    read_json,
)
from .edit_engine import (
    EditDraft,
    EditEngine,
    PartialWriteError,
    PreviewTokenError,
    TransactionError,
    build_synthetic_project,
)
from .dataset_registry import DatasetError, is_safe_relative, load_registry
from . import evidence_ledger as eledger
from .geometry_kernel import LocalFrame, read_ply_points
from .project_store import ProjectStore, migrate_legacy_v1_project
from . import portable_project as portablep
from .support_geometry import normalise_frame as normalise_support_frame
from .rule_solver import (
    ProtectionSet,
    RuleConfig,
    RuleScorer,
    RuleSolver,
    pf02_evidence_by_relation,
)

ROOT = Path(__file__).resolve().parents[3]
PF02_BUILD = ROOT / "results" / "product_first_plan_20261106" / "runs" / "PF02" / "build"
RECOMPUTE_SCRIPT = "src/fractureflow/surface_product/workbench_service.py"
# W153: evidence sidecar inside each persisted version directory. ProjectStore
# verifies only the files listed in its own manifest, so this additive file
# cannot silently change the snapshot bytes; it is read back and validated.
EVIDENCE_SIDECAR = "evidence_ledger.json"
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled", "timeout"})
ACTIVE_JOB_STATES = frozenset({"queued", "running", "cancelling"})
MAX_IMPORT_BYTES = 12 * 1024 * 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_non_finite(token: str) -> Any:
    """json parse_constant hook: NaN/Infinity in a package manifest is a hard error."""
    raise ValueError("non-finite JSON constant %r" % (token,))


def _rehash(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("content_hash", None)
    result["content_hash"] = canonical_sha256(result)
    return result


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "project"
    return "%s_%s" % (cleaned[:72], hashlib.sha256(value.encode("utf-8")).hexdigest()[:10])


# W158: the provenance domain has one source of truth.  A registered real
# dataset keeps its declared domain through the read-only v1 open, the explicit
# v2 migration, save and reopen; it must never fall through to "synthetic"
# merely because its catalog ID is not the fixed internal PF02 entry.
REAL_DOMAIN = "internal_real"
SYNTHETIC_DOMAIN = "synthetic"


def _domain_from_scene(scene_context: Optional[Mapping[str, Any]]) -> Optional[str]:
    kind = str((scene_context or {}).get("source_kind") or "")
    if kind == REAL_DOMAIN:
        return REAL_DOMAIN
    if kind:
        return SYNTHETIC_DOMAIN
    return None


def _project_domain(source_id: str, scene_context: Optional[Mapping[str, Any]] = None,
                    artifacts: Optional[Mapping[str, Any]] = None) -> str:
    """Resolve one session's domain without silently downgrading real data.

    The scene block wins because it was copied from the build that actually
    supplied the support points; the persisted session block is the fallback for
    a reopened project.  Only when neither declares anything does the source ID
    decide (the fixed internal PF02 build is real, everything else synthetic).
    """
    declared = _domain_from_scene(scene_context)
    if declared is not None:
        return declared
    stored = (artifacts or {}).get("workbench_session") or {}
    stored_domain = str(stored.get("domain") or "")
    if stored_domain in (REAL_DOMAIN, SYNTHETIC_DOMAIN):
        return stored_domain
    if source_id == "internal_pf02":
        return REAL_DOMAIN
    return SYNTHETIC_DOMAIN


def _project_declares_real_measurement(project: ProjectSnapshot) -> bool:
    """True when the snapshot itself claims a real measurement provenance.

    Read from three independent declarations; any one of them is enough to put
    the project under the real-geometry policy (fail closed, never downgrade).
    """
    artifacts = dict(project.artifacts or {})
    scene = dict(artifacts.get("workbench_scene") or {})
    if str(scene.get("source_kind") or "") == REAL_DOMAIN:
        return True
    session = dict(artifacts.get("workbench_session") or {})
    if str(session.get("domain") or "") == REAL_DOMAIN:
        return True
    return any(str(dict(getattr(item, "coverage", None) or {}).get("domain") or "") == "real"
               for item in project.observations)


def _quality_status_for(domain: str) -> str:
    return "UNDEFINED" if domain == REAL_DOMAIN else "SYNTHETIC_FUNCTIONAL_PROBE_ONLY"


def _real_quality_for(domain: str) -> str:
    return "undefined" if domain == REAL_DOMAIN else "synthetic_functional_probe_only"


def _measurement_status_for(domain: str) -> str:
    return "UNDEFINED_UNVERIFIED_UNIT" if domain == REAL_DOMAIN else "SYNTHETIC_NO_METRIC"


def _json_error(exc: BaseException, fallback: str = "WORKBENCH_ERROR") -> Dict[str, Any]:
    return {
        "code": getattr(exc, "code", fallback),
        "message": str(exc),
        "type": type(exc).__name__,
    }


def _atomic_text(path: Path, text: str) -> str:
    # The export directory can legitimately exceed MAX_PATH on Windows (a
    # long workbench home + a long export id).  Every file API in this helper
    # therefore goes through io_path; the fast path returns short names
    # unchanged, so ordinary exports keep ordinary paths.
    target = Path(io_path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    temporary = Path(io_path(temporary_name))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        return file_sha256(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _basis(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if not math.isfinite(norm) or norm <= 0:
        raise ContractError("surface normal is degenerate", "VISUAL_GEOMETRY_INVALID")
    n = n / norm
    seed = np.asarray([0.0, 0.0, 1.0]) if abs(float(n[2])) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    u = np.cross(seed, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return n, u, v


def _candidate_scene_context(batch_path: Path, source_kind: str) -> Dict[str, Any]:
    document = read_json(batch_path)
    return {
        "schema": "fractureflow.surface_product.workbench_scene.v1",
        "source_kind": source_kind,
        "frame": copy.deepcopy(document["frame"]),
        "candidate_batch_path": str(batch_path),
        "candidates": {
            item["candidate_id"]: {
                "center_local": copy.deepcopy(item["center_local"]),
                "basis_u": copy.deepcopy(item["basis_u"]),
                "basis_v": copy.deepcopy(item["basis_v"]),
            }
            for item in document.get("fragments") or []
        },
        "display_policy": "LOCAL_DISPLAY_COORDINATES_ONLY_NO_MEASUREMENT_RESCALING",
    }


def _synthetic_scene_context() -> Dict[str, Any]:
    return {
        "schema": "fractureflow.surface_product.workbench_scene.v1",
        "source_kind": "synthetic_contract",
        "frame": {
            "origin": [0.0, 0.0, 0.0], "scale": 1.0,
            "unit_status": "SYNTHETIC_UNITLESS", "coordinate_frame": "synthetic_local",
        },
        "candidates": {},
        "display_policy": "SYNTHETIC_SUPPORT_RECONSTRUCTION_NOT_MEASUREMENT",
    }


def build_pf05_synthetic_project() -> ProjectSnapshot:
    """Build a non-empty v2 project with a real hole in its public finite region."""
    project = build_synthetic_project()
    payload = project.to_dict()
    hole = [[[0.20, 0.20], [0.42, 0.20], [0.42, 0.42], [0.20, 0.42], [0.20, 0.20]]]
    payload["fragments"][0]["finite_regions_local"][0]["holes"] = copy.deepcopy(hole)
    payload["surfaces"][0]["finite_support"][0]["holes"] = copy.deepcopy(hole)
    artifacts = dict(payload.get("artifacts") or {})
    artifacts["workbench_session"] = {
        "catalog_id": "synthetic_contract", "domain": "synthetic",
        "quality_status": "SYNTHETIC_FUNCTIONAL_PROBE_ONLY",
    }
    artifacts["workbench_scene"] = _synthetic_scene_context()
    payload["artifacts"] = artifacts
    return ProjectSnapshot.from_dict(_rehash(payload))


# ---------------------------------------------------------------------------
# T-D: immutable real-geometry source descriptor
#
# The 2026-11-18 probe measured the product defect: the manual transaction and
# the default rule worker agreed on the selection identity but only the manual
# one refitted plane / finite regions / residuals.  The worker had no original
# point provider, so SELECT_SUPPORT ran on the legacy fixture branch and kept
# the previous geometry.  A worker also cannot be trusted to re-derive *which*
# bytes are the measurement: the descriptor below is built (and read back) by
# the parent from the session, pinned by SHA-256, and re-verified in the child
# against both the project's own declaration and the bytes on disk.
# ---------------------------------------------------------------------------

GEOMETRY_SOURCE_SCHEMA = "fractureflow.surface_product.real_geometry_source.v1"
MEASUREMENT_ROLES_FORBIDDEN = ("DERIVED_DISPLAY", "DISPLAY_ONLY", "DISPLAY")
GEOMETRY_POLICY = "REAL_SUPPORT_POINTS_ONLY_WHEN_SOURCE_IS_INTERNAL_REAL"


def _local_frame(scene_context: Optional[Mapping[str, Any]]) -> Optional[LocalFrame]:
    frame = (scene_context or {}).get("frame")
    if not isinstance(frame, Mapping) or not frame:
        return None
    return LocalFrame(
        origin=[float(item) for item in frame["origin"]],
        scale=float(frame["scale"]),
        unit_status=str(frame["unit_status"]),
        coordinate_frame=str(frame["coordinate_frame"]),
    )


def _declared_ply_source(project: ProjectSnapshot, observation_id: str):
    observation = next((item for item in project.observations
                        if item.observation_id == observation_id), None)
    if observation is None:
        return None, None
    source = next((item for item in observation.source_files
                   if str(item.path).lower().endswith(".ply")), None)
    return observation, source


def _assert_source_is_measurement(observation: Any, relative: str) -> None:
    """A display decimation may never stand in for the measurement it came from."""
    coverage = dict(getattr(observation, "coverage", None) or {})
    role = str(coverage.get("asset_role") or "")
    if role in MEASUREMENT_ROLES_FORBIDDEN:
        raise ContractError(
            "observation %s declares asset_role=%s; a display asset cannot supply support points"
            % (observation.observation_id, role),
            "SOURCE_NOT_MEASUREMENT",
        )
    if coverage.get("is_measurement") is False or coverage.get("display_only") is True:
        raise ContractError(
            "observation %s does not declare itself as a measurement" % observation.observation_id,
            "SOURCE_NOT_MEASUREMENT",
        )


def _resolve_source_path(relative: str) -> Path:
    path = (ROOT / str(relative)).resolve()
    if ROOT.resolve() not in path.parents:
        raise ContractError("declared source leaves the repository root: %s" % relative,
                            "SOURCE_OUTSIDE_ROOT")
    if not path.is_file():
        raise ContractError("declared source file is absent: %s" % relative,
                            "SOURCE_POINTS_UNAVAILABLE")
    return path


def build_geometry_source(project: ProjectSnapshot,
                          scene_context: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pin every real PLY measurement this project declares; refuse one that fails.

    Returning ``None`` means "this project is not a real measurement", never
    "verification was skipped".  Any unreadable, substituted, non-measurement or
    outside-root source raises instead of degrading to a fixture.
    """
    if (scene_context or {}).get("source_kind") != REAL_DOMAIN:
        return None
    frame = _local_frame(scene_context)
    if frame is None:
        raise ContractError("a real session has no local frame", "GEOMETRY_FRAME_UNAVAILABLE")
    entries: Dict[str, Dict[str, Any]] = {}
    for observation in project.observations:
        observation, source = _declared_ply_source(project, observation.observation_id)
        if observation is None or source is None:
            continue
        relative = str(source.path)
        _assert_source_is_measurement(observation, relative)
        path = _resolve_source_path(relative)
        points, header = read_ply_points(path)
        declared_sha = str(source.sha256 or "")
        if declared_sha and header["sha256"] != declared_sha:
            raise ContractError(
                "same-path bytes changed for %s: declared %s, on disk %s"
                % (relative, declared_sha, header["sha256"]),
                "SOURCE_SHA_MISMATCH",
            )
        entries[observation.observation_id] = {
            "path": relative,
            "sha256": header["sha256"],
            "byte_count": int(header["byte_count"]),
            "vertex_count": int(header["vertex_count"]),
            "point_count": int(len(points)),
        }
    if not entries:
        raise ContractError("a real project declares no readable PLY measurement",
                            "SOURCE_POINTS_UNAVAILABLE")
    return {
        "schema": GEOMETRY_SOURCE_SCHEMA,
        "source_kind": REAL_DOMAIN,
        "frame": frame.to_dict(),
        "observations": entries,
        "policy": "PINNED_BY_SHA256_REVERIFIED_IN_WORKER",
    }


def assert_geometry_source_matches_project(project: ProjectSnapshot,
                                           descriptor: Mapping[str, Any]) -> None:
    """The child re-checks the pin against the project it was handed.

    This is the guard that makes "change the source" a failure rather than a
    silently different geometry: the descriptor and the project must name the
    same observation, path and bytes, and the frame must be the project's frame.
    """
    if not isinstance(descriptor, Mapping) or descriptor.get("schema") != GEOMETRY_SOURCE_SCHEMA:
        raise ContractError("geometry source descriptor is missing or unknown",
                            "SOURCE_DESCRIPTOR_INVALID")
    if str(descriptor.get("source_kind")) != REAL_DOMAIN:
        raise ContractError("geometry source descriptor is not a real measurement pin",
                            "SOURCE_DESCRIPTOR_INVALID")
    scene = dict((project.artifacts or {}).get("workbench_scene") or {})
    if scene:
        if str(scene.get("source_kind") or "") != REAL_DOMAIN:
            raise ContractError("descriptor says real but the project scene does not",
                                "SOURCE_DESCRIPTOR_MISMATCH")
        if canonical_json(scene.get("frame")) != canonical_json(descriptor.get("frame")):
            raise ContractError("worker frame differs from the project frame",
                                "GEOMETRY_FRAME_MISMATCH")
    pinned = descriptor.get("observations") or {}
    for observation in project.observations:
        observation, source = _declared_ply_source(project, observation.observation_id)
        if observation is None or source is None:
            continue
        entry = pinned.get(observation.observation_id)
        if not isinstance(entry, Mapping):
            raise ContractError(
                "observation %s declares a PLY measurement that the job did not pin"
                % observation.observation_id,
                "SOURCE_DESCRIPTOR_MISMATCH",
            )
        if str(entry.get("path")) != str(source.path):
            raise ContractError(
                "observation %s is pinned to %s but the project declares %s"
                % (observation.observation_id, entry.get("path"), source.path),
                "SOURCE_DESCRIPTOR_MISMATCH",
            )
        if str(entry.get("sha256")) != str(source.sha256):
            raise ContractError(
                "observation %s pin and project declaration disagree on the SHA-256"
                % observation.observation_id,
                "SOURCE_DESCRIPTOR_MISMATCH",
            )


def verify_geometry_source_bytes(descriptor: Mapping[str, Any]) -> int:
    """Read every pinned measurement and verify its SHA-256 before solving.

    A source-integrity failure is not an "infeasible edit": it must fail the job
    outright, not come back as a rejected candidate that still looks solved.
    Returns the number of files verified (the enumeration count).
    """
    entries = descriptor.get("observations") or {}
    verified = 0
    for observation_id, entry in sorted(entries.items()):
        path = _resolve_source_path(str(entry.get("path")))
        _points, header = read_ply_points(path)
        if header["sha256"] != str(entry.get("sha256")):
            raise ContractError(
                "same-path bytes differ from the pinned SHA-256 for %s (%s)"
                % (observation_id, entry.get("path")),
                "SOURCE_SHA_MISMATCH",
            )
        if int(header["vertex_count"]) != int(entry.get("vertex_count")):
            raise ContractError(
                "pinned vertex count does not match the file for %s" % observation_id,
                "SOURCE_DESCRIPTOR_MISMATCH",
            )
        verified += 1
    if verified == 0:
        raise ContractError("geometry source descriptor pins no measurement",
                            "SOURCE_DESCRIPTOR_INVALID")
    return verified


def verify_event_index_bounds(project: ProjectSnapshot, events: Sequence[Any],
                              descriptor: Mapping[str, Any]) -> int:
    """A support index outside the pinned cloud is an integrity failure.

    The engine would report it as an infeasible edit (still unpublished); the
    worker turns it into a failed job because the request itself is impossible
    against the measurement that was pinned.
    """
    entries = descriptor.get("observations") or {}
    fragments = {item.fragment_id: item for item in project.fragments}
    checked = 0
    for event in events:
        action = getattr(event, "action", None)
        if action != "SELECT_SUPPORT":
            continue
        payload = dict(getattr(event, "payload", None) or {})
        indices = [int(item) for item in (payload.get("indices") or [])]
        for target in getattr(event, "target_ids", ()) or ():
            fragment = fragments.get(str(target))
            if fragment is None:
                continue
            entry = entries.get(fragment.observation_id)
            if not isinstance(entry, Mapping):
                raise ContractError(
                    "observation %s has no pinned measurement" % fragment.observation_id,
                    "SOURCE_POINTS_UNAVAILABLE",
                )
            vertex_count = int(entry.get("vertex_count"))
            if any(item < 0 or item >= vertex_count for item in indices):
                raise ContractError(
                    "support index outside the pinned measurement for %s: %d vertices"
                    % (fragment.observation_id, vertex_count),
                    "SUPPORT_INDEX_OUT_OF_RANGE",
                )
            checked += 1
    return checked


def providers_from_geometry_source(project: ProjectSnapshot,
                                   descriptor: Mapping[str, Any]):
    """Rebuild (support_points, fragment_frames) inside the worker.

    Both providers verify the bytes at first use and cache by (path, sha256) —
    never by file name alone, so a same-path substitution cannot be served from
    a previous read.
    """
    frame = _local_frame({"frame": descriptor.get("frame")})
    if frame is None:
        raise ContractError("geometry source descriptor has no valid frame",
                            "GEOMETRY_FRAME_UNAVAILABLE")
    scene = dict((project.artifacts or {}).get("workbench_scene") or {})
    entries = dict(descriptor.get("observations") or {})
    cache: Dict[Tuple[str, str], np.ndarray] = {}

    def _points_for(entry: Mapping[str, Any]) -> np.ndarray:
        key = (str(entry.get("path")), str(entry.get("sha256")))
        if key in cache:
            return cache[key]
        path = _resolve_source_path(key[0])
        points, header = read_ply_points(path)
        if header["sha256"] != key[1]:
            raise ContractError(
                "same-path bytes differ from the pinned SHA-256 for %s" % key[0],
                "SOURCE_SHA_MISMATCH",
            )
        if int(header["vertex_count"]) != int(entry.get("vertex_count")):
            raise ContractError("pinned vertex count does not match the file for %s" % key[0],
                                "SOURCE_DESCRIPTOR_MISMATCH")
        cache[key] = np.asarray(points, dtype=np.float64)
        return cache[key]

    def support_points(fragment_id: str, observation_id: str,
                       indices: Sequence[int]) -> Optional[Sequence[Sequence[float]]]:
        entry = entries.get(str(observation_id))
        if not isinstance(entry, Mapping):
            return None
        points = _points_for(entry)
        chosen = np.asarray([int(item) for item in indices], dtype=np.int64)
        if chosen.size and (int(chosen.min()) < 0 or int(chosen.max()) >= len(points)):
            raise ContractError(
                "support index out of range for observation %s" % observation_id,
                "SUPPORT_INDEX_OUT_OF_RANGE",
            )
        return frame.to_local(points[chosen]).tolist()

    def fragment_frames(fragment_id: str) -> Optional[Mapping[str, Any]]:
        fragment = next((item for item in project.fragments
                         if item.fragment_id == fragment_id), None)
        if fragment is None:
            return None
        return _frame_for_fragment_data(project, scene, fragment)

    return support_points, fragment_frames


def _frame_for_fragment_data(project: ProjectSnapshot, scene_context: Mapping[str, Any],
                             fragment: Any) -> Optional[Dict[str, Any]]:
    quality = dict(fragment.quality or {})
    stored = quality.get("frame_local")
    if isinstance(stored, Mapping) and stored:
        try:
            return normalise_support_frame(stored)
        except ContractError:
            pass
    candidate = (scene_context.get("candidates") or {}).get(fragment.fragment_id)
    if candidate:
        plane = fragment.plane_local
        frame = {
            "normal": list(plane.normal),
            "origin": list(candidate.get("center_local") or
                           (np.asarray(plane.normal, dtype=np.float64) * plane.offset)),
            "basis_u": list(candidate.get("basis_u") or []),
            "basis_v": list(candidate.get("basis_v") or []),
        }
        try:
            return normalise_support_frame(frame)
        except ContractError:
            return None
    return None


@dataclass
class Session:
    project: ProjectSnapshot
    engine: EditEngine
    source_id: str
    read_only: bool
    scene_context: Dict[str, Any]
    relation_evidence: Dict[str, Any] = field(default_factory=dict)
    # W153: explicit evidence ledger (records + meta) that travels with this
    # session.  relation_evidence is now only the current subset handed to the
    # solver; it is never the whole cache.
    evidence_ledger: Dict[str, Any] = field(default_factory=dict)
    evidence_meta: Dict[str, Any] = field(default_factory=dict)
    bundle_path: Optional[Path] = None
    saved_hash: Optional[str] = None
    cancelled_tokens: set = field(default_factory=set)
    # Exactly one live preview may exist per session.  The client keeps a mirror
    # of this record; the server is the authority on whether a token is still
    # committable, so a page reload can never resurrect or lose it silently.
    live_preview: Optional[Dict[str, Any]] = None
    last_save: Optional[Dict[str, Any]] = None
    preview_releases: List[Dict[str, Any]] = field(default_factory=list)
    # Tokens whose commit was rejected for a reason that leaves the preview
    # itself alive (today: stale base_version).  A later explicit cancel uses
    # this note as the release reason so the negative control is auditable.
    commit_rejections: Dict[str, str] = field(default_factory=dict)


def _public_job(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Job record without the worker's full candidate project.

    The HTTP surface only ever needs status/progress/phase/error/fallback plus a
    small summary of the candidate; the candidate document itself is used
    server-side by ``apply_job`` and never travels over ``/api/state``.
    """
    public = {key: copy.deepcopy(value) for key, value in record.items() if key != "result"}
    result = record.get("result") or {}
    update = (result or {}).get("update") or {}
    public["result_summary"] = ({
        "accepted": update.get("accepted"),
        "step_count": len(update.get("energy_trace") or []),
        "content_hash": update.get("content_hash"),
        "project_version": (result.get("project") or {}).get("project_version"),
    } if result else None)
    return public


WORKER_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0, %r);"
    "from fractureflow.surface_product.workbench_service import _solve_process_entry;"
    "_solve_process_entry(sys.argv[1])"
) % str(ROOT / "src")


class _WorkerProcess:
    """Handle for the no-console solve worker.

    ``multiprocessing`` on Windows spawns by inheriting the parent's console and
    exposes no ``creationflags``.  From an interactive desktop session that
    means the child either attaches to an existing console or (from a console
    parent) can surface one.  The 2026-11-18 audit could not assert either way
    from the code alone, so the launch was changed to ``subprocess`` with
    ``CREATE_NO_WINDOW``: the guarantee is a property of the launch call itself
    and is measured in ``window_risk_audit.json``.
    """

    def __init__(self, popen: "subprocess.Popen") -> None:
        self._popen = popen

    @property
    def pid(self) -> Optional[int]:
        return self._popen.pid

    @property
    def exitcode(self) -> Optional[int]:
        return self._popen.returncode

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def terminate(self) -> None:
        try:
            self._popen.terminate()
        except OSError:
            pass

    def kill(self) -> None:
        try:
            self._popen.kill()
        except OSError:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def _spawn_solve_worker(directory: Path) -> _WorkerProcess:
    """Launch the solve worker with no console window, by construction."""
    creationflags = int(subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    environment = dict(os.environ)
    source_root = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root + (os.pathsep + existing if existing else "")
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with (directory / "worker_stdout.log").open("wb") as out, \
            (directory / "worker_stderr.log").open("wb") as err:
        popen = subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-c", WORKER_BOOTSTRAP,
             str(directory / "input.json")],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, cwd=str(ROOT),
            env=environment, creationflags=creationflags, shell=False, close_fds=True,
        )
    return _WorkerProcess(popen)


class JobManager:
    """Isolated process jobs with an independent wall-clock watchdog."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._runtime: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        *,
        project: ProjectSnapshot,
        config: RuleConfig,
        events: Sequence[Mapping[str, Any]],
        relation_evidence: Mapping[str, Any],
        evidence_snapshot_id: Optional[str] = None,
        evidence_total: int = 0,
        evidence_excluded: int = 0,
        protected: Any,
        timeout_seconds: float,
        solver_mode: str,
        fault: str = "none",
        resource_limit_points: int = 2_000_000,
        resource_limit_fragments: int = 5_000,
        geometry_source: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not 0.05 <= float(timeout_seconds) <= 3600.0:
            raise ContractError("timeout_seconds must be in [0.05, 3600]", "TIMEOUT_INVALID")
        if resource_limit_points < 1 or resource_limit_fragments < 1:
            raise ContractError("resource limits must be positive", "RESOURCE_LIMIT_INVALID")
        if fault not in ("none", "hang", "error"):
            raise ContractError("unknown acceptance fault", "FAULT_INVALID")
        job_id = "job_" + uuid.uuid4().hex[:20]
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        point_count = sum(int(item.coverage.get("point_count") or 0) for item in project.observations)
        record = {
            "schema": "fractureflow.surface_product.workbench_job.v1",
            "job_id": job_id,
            "kind": "rule_solve" if solver_mode == "rule" else "learned_solve",
            "status": "queued",
            "progress": 0.0,
            "phase": "RESOURCE_CHECK",
            "project_id": project.project_id,
            "base_version": project.project_version,
            "base_content_hash": project.to_dict()["content_hash"],
            "submitted_at": _now_iso(),
            "deadline_monotonic": None,
            "timeout_seconds": float(timeout_seconds),
            "solver_mode": solver_mode,
            "fault_fixture": fault,
            "resource_limits": {
                "point_count": point_count, "point_limit": int(resource_limit_points),
                "fragment_count": len(project.fragments), "fragment_limit": int(resource_limit_fragments),
            },
            "applied": False,
            "evidence": {
                "snapshot_id": evidence_snapshot_id,
                "relation_total": int(evidence_total),
                "current_supplied": len(relation_evidence),
                "excluded_from_solver": int(evidence_excluded),
            },
            "fallback": {
                "available": True, "path": "MANUAL_EDIT_OR_RULE_SOLVE",
                "project_version": project.project_version,
                "content_hash": project.to_dict()["content_hash"],
            },
            "measured_at": _now_iso(),
            "recompute_script": RECOMPUTE_SCRIPT,
        }
        if point_count > resource_limit_points or len(project.fragments) > resource_limit_fragments:
            record.update({
                "status": "failed", "phase": "RESOURCE_CHECK_REJECTED",
                "error": {"code": "RESOURCE_LIMIT_EXCEEDED", "message": "project exceeds declared local resource budget"},
                "ended_at": _now_iso(),
            })
            atomic_write_json(directory / "job.json", record)
            with self._lock:
                self._jobs[job_id] = record
            return copy.deepcopy(record)

        payload = {
            "project": project.to_dict(),
            "config": config.to_dict(),
            "events": copy.deepcopy(list(events)),
            "relation_evidence": copy.deepcopy(dict(relation_evidence)),
            "evidence_snapshot_id": evidence_snapshot_id,
            "evidence_total": int(evidence_total),
            "evidence_excluded": int(evidence_excluded),
            "protected": ({"object_ids": list(protected.object_ids), "include_project_regions": protected.include_project_regions}
                          if isinstance(protected, ProtectionSet) else copy.deepcopy(protected)),
            "solver_mode": solver_mode,
            "fault": fault,
            "result_path": str(directory / "worker_result.json"),
            "progress_path": str(directory / "worker_progress.json"),
            # T-D: the worker can only refit from the original measurement if it
            # is told, immutably, which bytes are that measurement.  Absent means
            # "this project is not a real measurement".
            "geometry_source": copy.deepcopy(dict(geometry_source)) if geometry_source else None,
        }
        # The payload is persisted before spawn so the worker never depends on
        # interactive stdin, a shell, a console window or inherited UI state.
        atomic_write_json(directory / "input.json", payload)
        process = _spawn_solve_worker(directory)
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = record
            self._runtime[job_id] = {"process": process, "cancel": cancel_event, "directory": directory}
        record["pid"] = process.pid
        record["worker_launch"] = {
            "kind": "subprocess_no_window",
            "executable": sys.executable,
            "creationflags": int(subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0,
            "console_window_created": False,
        }
        record["status"] = "running"
        record["started_at"] = _now_iso()
        record["deadline_monotonic"] = time.monotonic() + float(timeout_seconds)
        atomic_write_json(directory / "job.json", record)
        watcher = threading.Thread(target=self._watch, args=(job_id,), name="pf05-" + job_id, daemon=True)
        self._runtime[job_id]["watcher"] = watcher
        watcher.start()
        return copy.deepcopy(record)

    def _watch(self, job_id: str) -> None:
        runtime = self._runtime[job_id]
        process: _WorkerProcess = runtime["process"]
        cancel_event: threading.Event = runtime["cancel"]
        directory: Path = runtime["directory"]
        result_path = directory / "worker_result.json"
        terminal_status: Optional[str] = None
        error: Optional[Dict[str, Any]] = None
        while True:
            record = self._jobs[job_id]
            if cancel_event.is_set():
                terminal_status = "cancelled"
                error = {"code": "CANCELLED_BY_USER", "message": "worker termination requested by user"}
                break
            if time.monotonic() >= float(record["deadline_monotonic"]):
                # The worker's ONLY output is an atomic result document. When it
                # is already on disk at the budget boundary, the job finished
                # and must be reaped by that result rather than killed as a
                # timeout. On Windows a spawned interpreter can linger during
                # teardown after publishing a complete candidate; the previous
                # code then emitted a false WALL_CLOCK_TIMEOUT and discarded a
                # valid candidate (measured on the PF05 internal-PF02 90s solve,
                # 2026-11-07). The wall clock still fires when no result exists.
                if result_path.is_file():
                    try:
                        finished = read_json(result_path)
                        terminal_status = str(finished.get("status") or "failed")
                        if terminal_status == "succeeded":
                            error = None
                        else:
                            error = finished.get("error") or {"code": "WORKER_FAILED", "message": "worker reported failure"}
                    except Exception as exc:
                        terminal_status = "failed"
                        error = _json_error(exc, "WORKER_RESULT_INVALID")
                    break
                terminal_status = "timeout"
                error = {"code": "WALL_CLOCK_TIMEOUT", "message": "solve exceeded its wall-clock budget"}
                break
            if not process.is_alive():
                if result_path.is_file():
                    try:
                        result = read_json(result_path)
                        terminal_status = str(result.get("status") or "failed")
                        if terminal_status == "succeeded":
                            error = None
                        else:
                            error = result.get("error") or {"code": "WORKER_FAILED", "message": "worker reported failure"}
                    except Exception as exc:
                        terminal_status = "failed"
                        error = _json_error(exc, "WORKER_RESULT_INVALID")
                else:
                    terminal_status = "failed"
                    error = {"code": "WORKER_EXIT_WITHOUT_RESULT", "message": "worker exited without a result", "returncode": process.exitcode}
                break
            progress_path = directory / "worker_progress.json"
            if progress_path.is_file():
                try:
                    progress = read_json(progress_path)
                    with self._lock:
                        self._jobs[job_id].update({
                            "progress": float(progress.get("progress") or 0.0),
                            "phase": str(progress.get("phase") or "RUNNING"),
                        })
                except (ValueError, OSError):
                    pass
            time.sleep(0.04)

        if terminal_status in ("cancelled", "timeout") and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        elif terminal_status in ("succeeded", "failed") and process.is_alive():
            # A complete atomic result already ended the job; give the spawned
            # interpreter a short chance to exit on its own, then reap it so a
            # daemon worker is not left behind. The candidate on disk is final.
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        process.join(timeout=0.2)
        result = read_json(result_path) if result_path.is_file() and terminal_status == "succeeded" else None
        with self._lock:
            record = self._jobs[job_id]
            record.update({
                "status": terminal_status or "failed",
                "progress": 1.0 if terminal_status == "succeeded" else float(record.get("progress") or 0.0),
                "phase": "COMPLETED" if terminal_status == "succeeded" else "FALLBACK_READY",
                "ended_at": _now_iso(),
                "worker_returncode": process.exitcode,
                "error": error,
                "result": result,
            })
            record.pop("deadline_monotonic", None)
            atomic_write_json(directory / "job.json", record)

    def get(self, job_id: str, *, public: bool = False) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ContractError("unknown job ID", "JOB_NOT_FOUND")
            record = copy.deepcopy(self._jobs[job_id])
        return _public_job(record) if public else record

    def list(self, *, public: bool = False) -> List[Dict[str, Any]]:
        """Return jobs in submission order.

        The previous implementation sorted by ``job_id``, which is a random
        uuid: "the last element" was therefore not the newest job and a UI or
        acceptance script reading ``jobs[-1]`` picked an arbitrary submission.
        Ordering is now explicit on ``submitted_at`` (microsecond ISO-8601) with
        the job id only as a deterministic tie-break for equal timestamps.

        ``public=True`` drops the worker's full candidate project.  A succeeded
        internal-PF02 job carries a ~2.3 MB document, and shipping it inside
        every ``/api/state`` poll made the browser parse megabytes per refresh
        (measured: 13-28 s of frozen UI between a click and its request).
        """
        with self._lock:
            rows = [copy.deepcopy(self._jobs[key]) for key in self._jobs]
        rows.sort(key=lambda row: (str(row.get("submitted_at") or ""), str(row.get("job_id") or "")))
        return [_public_job(row) for row in rows] if public else rows

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self.get(job_id)
            if record["status"] in TERMINAL_JOB_STATES:
                return record
            self._jobs[job_id]["status"] = "cancelling"
            self._jobs[job_id]["phase"] = "CANCEL_REQUESTED"
            self._runtime[job_id]["cancel"].set()
            return self.get(job_id)

    def mark_applied(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record["applied"]:
                raise ContractError("job result was already applied", "JOB_REPLAY")
            if record["status"] != "succeeded":
                raise ContractError("only a succeeded job can be applied", "JOB_NOT_SUCCESSFUL")
            record["applied"] = True
            record["applied_at"] = _now_iso()
            atomic_write_json(self._runtime[job_id]["directory"] / "job.json", record)

    def shutdown(self) -> None:
        for record in self.list():
            if record["status"] in ACTIVE_JOB_STATES:
                self.cancel(record["job_id"])
        for runtime in list(self._runtime.values()):
            watcher = runtime.get("watcher")
            if watcher is not None:
                watcher.join(timeout=2.0)


def _write_progress(path: str, progress: float, phase: str) -> bool:
    """Best-effort progress telemetry; the atomic result document is authoritative.

    On Windows the worker's os.replace can lose a sharing race with the parent
    watcher reading the previous sample (measured WinError 5 / PermissionError on
    worker_progress.json).  A lost progress sample must never fail a solve, so it
    is retried briefly and then skipped; result publishing runs on its own path.
    """
    payload = {"progress": float(progress), "phase": phase, "measured_at": _now_iso()}
    for attempt in range(5):
        try:
            atomic_write_json(Path(path), payload)
            return True
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    return False


def _solve_process_entry(input_path: str) -> None:
    """Spawn-safe worker entry; the only output is an atomic result document."""
    directory = Path(input_path).parent
    result_path = directory / "worker_result.json"
    try:
        payload = read_json(input_path)
        progress_path = payload["progress_path"]
        _write_progress(progress_path, 0.05, "INPUT_RESTORE")
        fault = payload["fault"]
        if fault == "hang":
            _write_progress(progress_path, 0.20, "ACCEPTANCE_HANG_FIXTURE")
            while True:
                time.sleep(0.10)
        if fault == "error":
            raise RuntimeError("PF05 acceptance-injected solver failure; project candidate was not published")
        if payload["solver_mode"] != "rule":
            raise ContractError("no learned model is registered for PF05; use manual edit or deterministic rules", "MODEL_UNAVAILABLE")
        project = ProjectSnapshot.from_dict(payload["project"])
        config_values = dict(payload["config"])
        config_values.pop("schema", None)
        config_values["seed_ids"] = tuple(config_values.get("seed_ids") or ())
        config = RuleConfig(**config_values)
        events = [EditEvent.from_dict(item) for item in payload.get("events") or []]
        # T-D: rebind the original measurement.  The descriptor is verified
        # against this project's own declaration (same observation/path/SHA and
        # same frame) before a single point is read; a mismatch fails the job
        # instead of publishing geometry derived from different bytes.
        descriptor = payload.get("geometry_source")
        real_required = descriptor is not None or _project_declares_real_measurement(project)
        support_points = None
        fragment_frames = None
        if descriptor is not None:
            assert_geometry_source_matches_project(project, descriptor)
            verify_geometry_source_bytes(descriptor)
            verify_event_index_bounds(project, events, descriptor)
            support_points, fragment_frames = providers_from_geometry_source(project, descriptor)
        elif real_required:
            raise ContractError(
                "this project declares a real measurement but the job carries no "
                "pinned source descriptor; refusing to solve on stale geometry",
                "SOURCE_POINTS_UNAVAILABLE",
            )
        _write_progress(progress_path, 0.20, "RULE_SCORING")
        scores = RuleScorer(project, config=config, relation_evidence=payload.get("relation_evidence") or {}).score()
        _write_progress(progress_path, 0.50, "RULE_SOLVE")
        solver = RuleSolver(project, config=config, support_points=support_points,
                            fragment_frames=fragment_frames,
                            real_geometry_required=real_required)
        update = solver.solve(events, scores, payload.get("protected"))
        _write_progress(progress_path, 0.90, "CANDIDATE_READY_NOT_PUBLISHED")
        result = {
            "status": "succeeded",
            "project": solver.project.to_dict(),
            "update": update.to_dict(),
            "score_receipt": {
                "content_hash": scores.content_hash(),
                "model_version": scores.model_version,
                "association_denominator": scores.feature_summary.get("association_denominator"),
                "relation_denominator": scores.feature_summary.get("relation_denominator"),
                "model_weights_used": False,
            },
            "measured_at": _now_iso(),
        }
        atomic_write_json(result_path, result)
        _write_progress(progress_path, 1.0, "COMPLETED")
    except Exception as exc:
        atomic_write_json(result_path, {
            "status": "failed", "error": _json_error(exc, "SOLVER_FAILED"),
            "fallback": {"available": True, "path": "MANUAL_EDIT_OR_RULE_SOLVE"},
            "measured_at": _now_iso(),
        })


class WorkbenchService:
    def __init__(self, workspace: os.PathLike[str] | str, *,
                 dataset_registry: Optional[os.PathLike[str] | str] = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.projects_root = self.workspace / "projects"
        self.exports_root = self.workspace / "exports"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.exports_root.mkdir(parents=True, exist_ok=True)
        self.jobs = JobManager(self.workspace / ".jobs")
        self._session: Optional[Session] = None
        self._lock = threading.RLock()
        self._point_cache: Dict[str, np.ndarray] = {}
        self._evidence_source_cache: Dict[str, Dict[str, Any]] = {}
        self._evidence_points_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._registry_path = Path(dataset_registry).resolve() if dataset_registry else None
        self._registry_cache: Optional[Dict[str, Any]] = None
        self._registry_error: Optional[Dict[str, Any]] = None
        # W163: last portable package opened through this service (read-only view).
        self._portable: Optional[Dict[str, Any]] = None

    # -- W152: disk-registered datasets ------------------------------------

    def _dataset_registry(self) -> Optional[Dict[str, Any]]:
        """Load the dataset registry once; a bad registry is reported, never raised."""
        if self._registry_path is None:
            return None
        if self._registry_cache is None and self._registry_error is None:
            if not self._registry_path.is_file():
                self._registry_error = {"code": "MISSING_ASSET",
                                        "message": "dataset registry is absent",
                                        "path": str(self._registry_path)}
            else:
                try:
                    self._registry_cache = load_registry(self._registry_path)
                except DatasetError as exc:
                    self._registry_error = {"code": exc.code, "message": str(exc),
                                            "path": str(self._registry_path)}
        return self._registry_cache

    def _registry_entry(self, catalog_id: str) -> Optional[Dict[str, Any]]:
        registry = self._dataset_registry() or {}
        for entry in registry.get("entries") or []:
            if entry.get("id") == catalog_id:
                return entry
        return None

    def _build_dir_for_source(self, source_id: str,
                              artifacts: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        """Locate the immutable build that declared this source, read-only."""
        if source_id == "internal_pf02":
            return PF02_BUILD / "internal-real__pontrelli_wall_internal"
        entry = self._registry_entry(source_id)
        if entry is not None and entry.get("_build_exists"):
            return Path(entry["_build_path"])
        stored = ((artifacts or {}).get("workbench_session") or {}).get("source_registry") or {}
        relative = stored.get("build_dir")
        if is_safe_relative(relative):
            candidate = (ROOT / str(relative)).resolve()
            if ROOT.resolve() in candidate.parents and candidate.is_dir():
                return candidate
        return None

    def _scene_for_source(self, source_id: str,
                          artifacts: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        directory = self._build_dir_for_source(source_id, artifacts)
        if directory is None:
            return None
        batch_path = directory / "candidate_batch.json"
        if not batch_path.is_file():
            return None
        entry = self._registry_entry(source_id) or {}
        domain = str(entry.get("domain") or
                     ((artifacts or {}).get("workbench_session") or {}).get("domain") or REAL_DOMAIN)
        return _candidate_scene_context(
            batch_path, "internal_real" if domain == REAL_DOMAIN else "synthetic_contract")

    def _relation_evidence_for(self, project: ProjectSnapshot, source_id: str,
                               artifacts: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Rebuild relation evidence from the immutable build; never from a guess."""
        directory = self._build_dir_for_source(source_id, artifacts)
        if directory is None:
            return {}
        report_path = directory / "relation_report.json"
        if not report_path.is_file():
            return {}
        return pf02_evidence_by_relation(project, read_json(report_path))

    # -- W153: evidence ledger ------------------------------------------------
    # relation_evidence used to be a session-lifetime copy of the immutable
    # build report.  These helpers give every relation an explicit geometric
    # basis, recompute affected pairs with the same finite kernel, and keep the
    # ledger's status separate from the relation's yes/no/unknown conclusion.

    def _evidence_source(self, project: ProjectSnapshot, source_id: str,
                         scene_context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve spacing, parameters and raw support points from the build.

        A missing PLY, an unreadable batch or a cloud over the declared budget
        returns None.  The caller then records out_of_scope evidence instead of
        inventing support points.
        """
        if scene_context.get("source_kind") != REAL_DOMAIN:
            return None
        directory = self._build_dir_for_source(source_id, project.artifacts)
        if directory is None:
            return None
        batch_path = directory / "candidate_batch.json"
        cache_key = str(batch_path)
        cached = self._evidence_source_cache.get(cache_key)
        if cached is None:
            if not batch_path.is_file():
                return None
            try:
                batch = read_json(batch_path)
                spacing = float((batch.get("audit") or {}).get("median_point_spacing"))
                raw_parameters = dict(batch.get("parameters") or {})
                allowed = {field.name for field in dataclasses.fields(eledger.CandidateParameters)}
                parameters = eledger.CandidateParameters(
                    **{key: value for key, value in raw_parameters.items() if key in allowed})
                origin = [float(item) for item in
                          ((batch.get("frame") or {}).get("origin") or [0.0, 0.0, 0.0])]
            except Exception:
                return None
            if not math.isfinite(spacing) or spacing <= 0.0:
                return None
            cached = {"spacing": spacing, "parameters": parameters, "origin": origin}
            self._evidence_source_cache[cache_key] = cached
        points: Dict[str, np.ndarray] = {}
        total = 0
        origin = np.asarray(cached["origin"], dtype=np.float64)
        for observation in project.observations:
            point_key = (cache_key, observation.observation_id)
            array = self._evidence_points_cache.get(point_key)
            if array is None:
                raw = self._raw_points_for(project, scene_context, observation.observation_id)
                if raw is None:
                    return None
                array = np.asarray(raw, dtype=np.float64) - origin
                if total + len(array) > eledger.MAX_CONTEXT_POINTS:
                    return None
                self._evidence_points_cache[point_key] = array
            total += len(array)
            points[observation.observation_id] = array
        if not points:
            return None
        return {
            "spacing": float(cached["spacing"]),
            "parameters": cached["parameters"],
            "origin": list(cached["origin"]),
            "points": points,
            "source": "candidate_batch:" + str(batch_path),
        }

    def _evidence_frames(self, project: ProjectSnapshot,
                         scene_context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Explicit frame per fragment and surface; absent frames stay absent."""
        surface_geometry = dict((project.artifacts or {}).get("surface_geometry") or {})
        candidates = scene_context.get("candidates") or {}
        frames: Dict[str, Dict[str, Any]] = {}
        for fragment in project.fragments:
            fallback = None
            candidate = candidates.get(fragment.fragment_id)
            if candidate:
                plane = fragment.plane_local
                fallback = {
                    "normal": list(plane.normal),
                    "origin": list(candidate.get("center_local") or
                                   (np.asarray(plane.normal, dtype=np.float64) * plane.offset)),
                    "basis_u": list(candidate.get("basis_u") or []),
                    "basis_v": list(candidate.get("basis_v") or []),
                }
            frame = eledger.frame_for_fragment(fragment, fallback)
            if frame is not None:
                frames[fragment.fragment_id] = frame
        for surface in project.surfaces:
            stored = surface_geometry.get(surface.surface_id) or {}
            frame = eledger.frame_for_surface(stored)
            if frame is None and len(surface.member_fragment_ids) == 1:
                frame = frames.get(surface.member_fragment_ids[0])
            if frame is not None:
                frames[surface.surface_id] = frame
        return frames

    def _evidence_context(self, project: ProjectSnapshot, source_id: str,
                          scene_context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        source = self._evidence_source(project, source_id, scene_context)
        if source is None:
            return None
        frames = self._evidence_frames(project, scene_context)
        if not frames:
            return None
        context = dict(source)
        context["frames"] = frames
        return eledger.normalize_context(context)

    def _initial_evidence_records(self, project: ProjectSnapshot,
                                  relation_evidence: Mapping[str, Any],
                                  context: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        basis_map = eledger.relation_basis_map(project)
        records: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for relation in project.relations:
            basis = basis_map.get(relation.relation_id) or {}
            decision = (relation_evidence or {}).get(relation.relation_id)
            if decision:
                records[relation.relation_id] = eledger.record_from_decision(
                    project, relation, decision, basis=basis)
            elif context is not None:
                missing.append(relation.relation_id)
            else:
                records[relation.relation_id] = eledger.unavailable_record(
                    project, relation, basis, "NO_EVIDENCE_CONTEXT",
                    "no build context exists for this session; evidence is out of scope",
                )
        if missing and context is not None:
            result = eledger.recompute_relations(
                project, missing, frames=context["frames"], points=context["points"],
                spacing=context["spacing"], parameters=context["parameters"],
                basis_map=basis_map, prior=records,
            )
            records.update(result["records"])
        return records

    def _install_evidence(self, session: Session,
                          payload: Optional[Mapping[str, Any]] = None) -> None:
        project = session.project
        decisions = session.relation_evidence or {}
        # The immutable build already carries a decision for every relation.
        # Reading the full PLY merely to confirm that is deferred until the
        # first invalidation or explicit full recompute; a missing decision
        # forces the context now because it cannot be fabricated.
        missing_decisions = [item.relation_id for item in project.relations
                             if item.relation_id not in decisions]
        context: Optional[Dict[str, Any]] = None
        if missing_decisions and not payload:
            context = self._evidence_context(project, session.source_id, session.scene_context)
        records: Optional[Dict[str, Dict[str, Any]]] = None
        sidecar: Dict[str, Any] = {"valid": None, "reasons": []}
        origin = "BUILD_REPORT_OR_RECOMPUTE"
        if payload:
            verdict = eledger.validate_payload(project, payload)
            sidecar = {
                "valid": verdict.get("valid"), "reasons": list(verdict.get("reasons") or []),
                "missing": len(verdict.get("missing") or []),
                "extra": len(verdict.get("extra") or []),
                "basis_mismatched": len(verdict.get("basis_mismatched") or []),
            }
            try:
                records = eledger.records_from_payload(payload)
                records = eledger.classify(project, records)
                origin = "SIDECAR"
            except Exception as exc:  # noqa: BLE001 - a bad sidecar rebuilds
                sidecar = {"valid": False, "reasons": ["RECORDS_UNREADABLE:%s" % exc]}
                records = None
        if records is None:
            records = self._initial_evidence_records(project, session.relation_evidence, context)
        snapshot = eledger.evidence_snapshot(project, records)
        if context is not None:
            context_status = "AVAILABLE"
        elif session.scene_context.get("source_kind") == REAL_DOMAIN:
            context_status = "DEFERRED_UNTIL_RECOMPUTE"
        else:
            context_status = "UNAVAILABLE"
        session.evidence_ledger = {"records": records}
        session.evidence_meta = {
            "context_status": context_status,
            "origin": origin,
            "sidecar": sidecar,
            "snapshot_id": snapshot["snapshot_id"],
            "project_content_hash": snapshot["project_content_hash"],
            "status_counts": snapshot["status_counts"],
            "last_refresh": None,
        }
        session.relation_evidence = eledger.current_evidence(records)

    def _evidence_public(self, session: Session) -> Dict[str, Any]:
        records = dict((session.evidence_ledger or {}).get("records") or {})
        snapshot = eledger.evidence_snapshot(session.project, records)
        counts = dict(snapshot["status_counts"])
        meta = dict(session.evidence_meta or {})
        return {
            "schema": snapshot["schema"],
            "method": snapshot["method"],
            "snapshot_id": snapshot["snapshot_id"],
            "project_id": snapshot["project_id"],
            "project_version": snapshot["project_version"],
            "project_content_hash": snapshot["project_content_hash"],
            "relation_count": snapshot["relation_count"],
            "record_count": snapshot["record_count"],
            "status_counts": counts,
            "current_count": counts.get(eledger.STATUS_CURRENT, 0),
            "stale_count": counts.get(eledger.STATUS_STALE, 0),
            "conflict_count": counts.get(eledger.STATUS_CONFLICT, 0),
            "out_of_scope_count": counts.get(eledger.STATUS_OUT_OF_SCOPE, 0),
            "excluded_from_solver": max(0, snapshot["relation_count"] - counts.get(eledger.STATUS_CURRENT, 0)),
            "context_status": meta.get("context_status"),
            "origin": meta.get("origin"),
            "sidecar": meta.get("sidecar"),
            "last_refresh": copy.deepcopy(meta.get("last_refresh")),
            "measured_at": snapshot["measured_at"],
            "recompute_script": eledger.RECOMPUTE_SCRIPT,
        }

    def _refresh_evidence(self, session: Session, changed_ids: Sequence[str], *,
                          reason: str, full: bool = False) -> Dict[str, Any]:
        """Invalidate and recompute evidence affected by one project change.

        The ledger is replaced only after the project hash is re-read.  A
        version change observed during recomputation marks the affected records
        stale (INPUT_CHANGED); it never publishes an old-geometry result as
        current.
        """
        project = session.project
        records = dict((session.evidence_ledger or {}).get("records") or {})
        records = eledger.classify(project, records)
        basis_map = eledger.relation_basis_map(project)
        changed = sorted({str(item) for item in changed_ids if item})
        closure: set = set()
        if changed:
            try:
                dependency = session.engine._dependency_closure(project, changed)
                closure = set((dependency or {}).get("closure") or [])
            except Exception:  # noqa: BLE001 - closure refinement is best-effort
                closure = set()
        affected: List[str] = []
        for relation in project.relations:
            record = records.get(relation.relation_id) or {}
            if full or relation.relation_id in changed:
                affected.append(relation.relation_id)
            elif relation.src_id in closure or relation.dst_id in closure:
                affected.append(relation.relation_id)
            elif relation.src_id in changed or relation.dst_id in changed:
                affected.append(relation.relation_id)
            elif record.get("status") != eledger.STATUS_CURRENT:
                affected.append(relation.relation_id)
            elif record.get("basis") != basis_map.get(relation.relation_id):
                affected.append(relation.relation_id)
        before_hash = project.compute_content_hash()
        context = self._evidence_context(project, session.source_id, session.scene_context)
        recomputed = 0
        failed = 0
        out_of_scope = 0
        if affected and context is not None:
            result = eledger.recompute_relations(
                project, affected, frames=context["frames"], points=context["points"],
                spacing=context["spacing"], parameters=context["parameters"],
                basis_map=basis_map, prior=records,
            )
            records.update(result["records"])
            recomputed = int(result["recomputed"])
            failed = int(result["failed"])
            out_of_scope = int(result["out_of_scope"])
        after_hash = project.compute_content_hash()
        input_changed = before_hash != after_hash
        if input_changed:
            for relation_id in affected:
                record = records.get(relation_id)
                if record is None:
                    continue
                record["status"] = eledger.STATUS_STALE
                record["error"] = {
                    "code": "INPUT_CHANGED",
                    "message": "project version changed while evidence was being recomputed",
                }
                record["error_code"] = "INPUT_CHANGED"
        snapshot = eledger.evidence_snapshot(project, records)
        session.evidence_ledger = {"records": records}
        meta = dict(session.evidence_meta or {})
        refresh = {
            "reason": str(reason),
            "changed_ids": changed[:64],
            "changed_count": len(changed),
            "closure_size": len(closure),
            "affected": len(affected),
            "recomputed": recomputed,
            "failed": failed,
            "out_of_scope": out_of_scope,
            "input_changed": bool(input_changed),
            "input_hash_before": before_hash,
            "snapshot_id": snapshot["snapshot_id"],
            "project_content_hash": snapshot["project_content_hash"],
            "measured_at": _now_iso(),
        }
        meta.update({
            "context_status": "AVAILABLE" if context is not None else "UNAVAILABLE",
            "snapshot_id": snapshot["snapshot_id"],
            "project_content_hash": snapshot["project_content_hash"],
            "status_counts": snapshot["status_counts"],
            "last_refresh": refresh,
        })
        session.evidence_meta = meta
        session.relation_evidence = eledger.current_evidence(records)
        return refresh

    def evidence_recompute_all(self) -> Dict[str, Any]:
        """Recompute every relation with the same kernel as the local path."""
        with self._lock:
            session = self._require_session()
            context = self._evidence_context(session.project, session.source_id, session.scene_context)
            if context is None:
                raise ContractError("no source context is available for evidence recomputation",
                                    "EVIDENCE_CONTEXT_UNAVAILABLE")
            records = dict((session.evidence_ledger or {}).get("records") or {})
            result = eledger.recompute_all(
                session.project, frames=context["frames"], points=context["points"],
                spacing=context["spacing"], parameters=context["parameters"], prior=records,
            )
            records.update(result["records"])
            snapshot = eledger.evidence_snapshot(session.project, records)
            session.evidence_ledger = {"records": records}
            meta = dict(session.evidence_meta or {})
            meta.update({
                "context_status": "AVAILABLE",
                "snapshot_id": snapshot["snapshot_id"],
                "project_content_hash": snapshot["project_content_hash"],
                "status_counts": snapshot["status_counts"],
                "last_refresh": {
                    "reason": "FULL_RECOMPUTE", "affected": len(records),
                    "recomputed": result["recomputed"], "failed": result["failed"],
                    "out_of_scope": result["out_of_scope"], "input_changed": False,
                    "snapshot_id": snapshot["snapshot_id"],
                    "project_content_hash": snapshot["project_content_hash"],
                    "measured_at": _now_iso(),
                },
            })
            session.evidence_meta = meta
            session.relation_evidence = eledger.current_evidence(records)
            return {"evidence": self._evidence_public(session), "recomputed": result["recomputed"],
                    "failed": result["failed"], "out_of_scope": result["out_of_scope"]}

    def dataset_registry_status(self) -> Dict[str, Any]:
        registry = self._dataset_registry()
        if self._registry_path is None:
            return {"status": "NOT_CONFIGURED", "path": None, "entry_count": 0}
        if self._registry_error is not None:
            return {"status": "REJECTED", "path": str(self._registry_path), "entry_count": 0,
                    "error": dict(self._registry_error)}
        return {"status": "OK", "path": str(self._registry_path),
                "entry_count": len((registry or {}).get("entries") or []),
                "allowed_roots": list((registry or {}).get("allowed_roots") or [])}

    def shutdown(self) -> None:
        self.jobs.shutdown()

    def _require_session(self) -> Session:
        if self._session is None:
            raise ContractError("no project is open", "NO_PROJECT")
        return self._session

    # -- W151: real support-point and frame providers ------------------------

    def _raw_points_for(self, project: ProjectSnapshot, scene_context: Mapping[str, Any],
                        observation_id: str) -> Optional[np.ndarray]:
        if scene_context.get("source_kind") != "internal_real":
            return None
        observation = next((item for item in project.observations
                            if item.observation_id == observation_id), None)
        if observation is None:
            return None
        source = next((item for item in observation.source_files
                       if item.path.lower().endswith(".ply")), None)
        if source is None:
            return None
        path = (ROOT / source.path).resolve()
        if not path.is_file() or ROOT.resolve() not in path.parents:
            return None
        key = str(path)
        if key not in self._point_cache:
            points, _audit = read_ply_points(path)
            self._point_cache[key] = np.asarray(points, dtype=np.float64)
        return self._point_cache[key]

    def _frame_for_fragment(self, project: ProjectSnapshot, scene_context: Mapping[str, Any],
                            fragment: Any) -> Optional[Dict[str, Any]]:
        return _frame_for_fragment_data(project, scene_context, fragment)

    def _geometry_providers(self, project: ProjectSnapshot, scene_context: Mapping[str, Any]):
        """Return (support_points, fragment_frames) for a real source or (None, None).

        T-D: coordinates go through ``LocalFrame.to_local`` so a non-unit scale is
        honoured (the previous subtraction ignored ``scale`` and was only correct
        at scale 1).  The closures resolve the *current* session project instead
        of a captured fragment map, so a provider cannot answer with a frame from
        before the edit it is being asked about.

        Synthetic fixture projects deliberately receive no provider: the engine
        then stays on its labelled legacy fixture path instead of pretending
        reconstructed support points are original measurements.
        """
        if scene_context.get("source_kind") != REAL_DOMAIN:
            return None, None
        frame = _local_frame(scene_context)
        origin = np.asarray((scene_context.get("frame") or {}).get("origin") or [0.0, 0.0, 0.0],
                            dtype=np.float64)

        def _current_project() -> ProjectSnapshot:
            session = self._session
            return session.project if session is not None else project

        def support_points(fragment_id: str, observation_id: str,
                           indices: Sequence[int]) -> Optional[Sequence[Sequence[float]]]:
            current = _current_project()
            points = self._raw_points_for(current, scene_context, observation_id)
            if points is None:
                return None
            chosen = np.asarray([int(item) for item in indices], dtype=np.int64)
            if len(chosen) and (int(chosen.min()) < 0 or int(chosen.max()) >= len(points)):
                raise ContractError(
                    "support index out of range for observation %s" % observation_id,
                    "SUPPORT_INDEX_OUT_OF_RANGE",
                )
            selected = points[chosen]
            local = frame.to_local(selected) if frame is not None else selected - origin
            return local.tolist()

        def fragment_frames(fragment_id: str) -> Optional[Mapping[str, Any]]:
            current = _current_project()
            fragment = next((item for item in current.fragments
                             if item.fragment_id == fragment_id), None)
            if fragment is None:
                return None
            return self._frame_for_fragment(current, scene_context, fragment)

        return support_points, fragment_frames

    def _make_session(self, project: ProjectSnapshot, *, source_id: str, read_only: bool,
                      scene_context: Mapping[str, Any], relation_evidence: Optional[Mapping[str, Any]] = None,
                      bundle_path: Optional[Path] = None, saved_hash: Optional[str] = None,
                      evidence_ledger: Optional[Mapping[str, Any]] = None) -> Session:
        support_points, fragment_frames = self._geometry_providers(project, scene_context)
        engine = EditEngine.from_project(
            project, support_points=support_points, fragment_frames=fragment_frames,
            real_geometry_required=scene_context.get("source_kind") == REAL_DOMAIN,
        )
        session = Session(
            project=project, engine=engine, source_id=source_id, read_only=read_only,
            scene_context=copy.deepcopy(dict(scene_context)),
            relation_evidence=copy.deepcopy(dict(relation_evidence or {})),
            bundle_path=bundle_path, saved_hash=saved_hash,
        )
        self._install_evidence(session, evidence_ledger)
        return session

    def catalog(self) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = [
            {"id": "synthetic_contract", "label": "新建合成项目", "domain": "synthetic", "editable_on_open": True,
             "scope": "v2公共动作、有限面孔洞、关系、求解、撤销/重做"},
            {"id": "internal_pf02", "label": "载入内部 PF02 项目", "domain": "internal_real", "editable_on_open": False,
             "scope": "只读v1兼容；显式迁移副本后编辑；单位/许可未核实，不外发"},
        ]
        registry = self._dataset_registry() or {}
        for entry in registry.get("entries") or []:
            entries.append({
                "id": entry["id"],
                "label": entry["label"],
                "domain": entry["domain"],
                "editable_on_open": False,
                "role": entry["role"],
                "scope": "登记资料（unit=%s；license=%s；%s）" % (
                    entry["unit_status"], entry["license_status"],
                    "可打开" if entry["_build_exists"] else "构建产物缺失"),
                "build_present": bool(entry["_build_exists"]),
            })
        return {
            "schema": "fractureflow.surface_product.workbench_catalog.v1",
            "development_only": True,
            "release_statuses": {
                "ENGINEERING_BASELINE_DELIVERED": False,
                "INTELLIGENT_PROTOTYPE_VALIDATED": False,
                "HARDENED_PRODUCT_RELEASE": False,
            },
            "entries": entries,
            "dataset_registry": self.dataset_registry_status(),
            "actions": ["SELECT_SUPPORT", "ADD_FRAGMENT", "SPLIT_SURFACE", "MERGE_SURFACE", "EDIT_BOUNDARY",
                        "SET_RELATION", "SET_UNKNOWN", "PROTECT_REGION", "UNPROTECT_REGION", "SET_INTERPRETATION",
                        "DELETE_OBJECT", "UNDO", "REDO"],
            "recompute_script": RECOMPUTE_SCRIPT,
        }

    def _bundle_for(self, project_id: str) -> Path:
        return self.projects_root / (_safe_name(project_id) + ".ffsrf")

    def _adopt(
        self, project: ProjectSnapshot, *, source_id: str, read_only: bool,
        scene_context: Mapping[str, Any], relation_evidence: Optional[Mapping[str, Any]] = None,
        bundle_path: Optional[Path] = None, saved_hash: Optional[str] = None,
        evidence_ledger: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self._session = self._make_session(
                project, source_id=source_id, read_only=read_only,
                scene_context=scene_context, relation_evidence=relation_evidence,
                bundle_path=bundle_path, saved_hash=saved_hash,
                evidence_ledger=evidence_ledger,
            )
            return self.state()

    def _open_build_dir(self, directory: Path, source_id: str, *,
                        label: str, domain: str = REAL_DOMAIN) -> Dict[str, Any]:
        """Open one PF02 build directory. Shared by the fixed entry and registry entries."""
        project_path = directory / "project.json"
        batch_path = directory / "candidate_batch.json"
        relation_path = directory / "relation_report.json"
        if not project_path.is_file() or not batch_path.is_file() or not relation_path.is_file():
            raise ContractError("%s enumeration is incomplete" % label, "EMPTY_ENUMERATION")
        project = ProjectSnapshot.from_dict(read_json(project_path))
        evidence = pf02_evidence_by_relation(project, read_json(relation_path))
        if not evidence:
            raise ContractError("%s relation evidence adaptation is empty" % label, "EMPTY_ENUMERATION")
        return self._adopt(
            project, source_id=source_id, read_only=project.schema_version == SCHEMA_VERSION,
            scene_context=_candidate_scene_context(
                batch_path, "internal_real" if domain == REAL_DOMAIN else "synthetic_contract"),
            relation_evidence=evidence,
        )

    def open_catalog(self, catalog_id: str) -> Dict[str, Any]:
        if catalog_id == "synthetic_contract":
            project = build_pf05_synthetic_project()
            return self._adopt(project, source_id=catalog_id, read_only=False,
                               scene_context=_synthetic_scene_context())
        if catalog_id == "internal_pf02":
            return self._open_build_dir(
                PF02_BUILD / "internal-real__pontrelli_wall_internal",
                catalog_id, label="internal PF02")
        entry = self._registry_entry(catalog_id)
        if entry is None:
            raise ContractError("unknown catalog entry", "CATALOG_NOT_FOUND")
        if not entry["_build_exists"]:
            raise ContractError("registered dataset build directory is absent", "MISSING_ASSET")
        return self._open_build_dir(Path(entry["_build_path"]), catalog_id,
                                    label="dataset " + catalog_id, domain=str(entry["domain"]))

    def import_project(self, document: Any, *, declared_bytes: int = 0) -> Dict[str, Any]:
        if declared_bytes > MAX_IMPORT_BYTES:
            raise ContractError("project import exceeds the 12 MiB boundary", "RESOURCE_LIMIT_EXCEEDED")
        if not isinstance(document, Mapping):
            raise ContractError("project import must be one JSON object", "PROJECT_JSON_INVALID")
        # Reject non-finite numbers and hidden evaluation keys before any session change.
        canonical_json(document)
        project = ProjectSnapshot.from_dict(document)
        artifacts = dict(project.artifacts or {})
        scene = artifacts.get("workbench_scene") or _synthetic_scene_context()
        return self._adopt(project, source_id="import_json", read_only=project.schema_version == SCHEMA_VERSION,
                           scene_context=scene)

    def migrate_current(self) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if not session.read_only and session.project.schema_version == LATEST_SCHEMA_VERSION:
                return self.state()
            migrated = migrate_legacy_v1_project(session.project)
            artifacts = dict(migrated.artifacts or {})
            stored_session = dict(artifacts.get("workbench_session") or {})
            source_domain = _project_domain(session.source_id, session.scene_context, artifacts)
            session_block = {
                "catalog_id": session.source_id,
                "domain": source_domain,
                "quality_status": _quality_status_for(source_domain),
                "source_schema_version": session.project.schema_version,
                "explicit_migration": True,
            }
            entry = self._registry_entry(session.source_id)
            if entry is not None:
                # Keep the registered source's own declaration with the copy:
                # role, unit/licence statements and the repo-relative build the
                # edited copy was derived from (the original stays read-only).
                session_block["source_registry"] = {
                    "id": entry.get("id"), "label": entry.get("label"),
                    "domain": entry.get("domain"), "role": entry.get("role"),
                    "unit_status": entry.get("unit_status"),
                    "license_status": entry.get("license_status"),
                    "build_dir": entry.get("build_dir"),
                    "point_cloud": entry.get("point_cloud"),
                }
                provenance = entry.get("provenance")
                if provenance:
                    # W160: carry the registered asset/ROI/source-index chain into
                    # the saved copy so a reopened project can still trace every
                    # support point back to the original source ordinal.
                    session_block["source_provenance"] = copy.deepcopy(provenance)
            stored_session.update(session_block)
            artifacts["workbench_session"] = stored_session
            artifacts["workbench_scene"] = copy.deepcopy(session.scene_context)
            migrated_payload = migrated.to_dict()
            migrated_payload["artifacts"] = artifacts
            migrated = ProjectSnapshot.from_dict(_rehash(migrated_payload))
            bundle = self._bundle_for(migrated.project_id)
            evidence = session.relation_evidence or self._relation_evidence_for(
                migrated, session.source_id, artifacts)
            return self._adopt(migrated, source_id=session.source_id, read_only=False,
                               scene_context=session.scene_context, relation_evidence=evidence,
                               bundle_path=bundle, saved_hash=None)

    def close_project(self) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            for job in self.jobs.list():
                if job["project_id"] == session.project.project_id and job["status"] in ACTIVE_JOB_STATES:
                    raise ContractError("cancel or await the active job before closing", "JOB_ACTIVE")
            self._session = None
            return {"closed": True, "library": self.library()}

    def library(self) -> List[Dict[str, Any]]:
        result = []
        if not self.projects_root.is_dir():
            return result
        for head_path in sorted(self.projects_root.glob("*.ffsrf/head.json")):
            try:
                head = read_json(head_path)
                result.append({
                    "bundle_id": head_path.parent.name,
                    "project_id": head.get("project_id"),
                    "project_version": head.get("project_version"),
                    "content_hash": head.get("content_hash"),
                    "schema_version": head.get("schema_version"),
                    "measured_at": head.get("measured_at"),
                })
            except Exception:
                result.append({"bundle_id": head_path.parent.name, "corrupt": True})
        return result

    def open_saved(self, bundle_id: str) -> Dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle_id) or ".." in bundle_id:
            raise ContractError("unsafe bundle ID", "UNSAFE_PATH")
        path = self.projects_root / bundle_id
        if not path.is_dir() or path.parent != self.projects_root:
            raise ContractError("saved project is missing", "PROJECT_NOT_FOUND")
        project, receipt = ProjectStore(path, recompute_script=RECOMPUTE_SCRIPT).load()
        if not receipt.get("verified"):
            raise ContractError("saved bundle failed independent verification", "BUNDLE_VERIFY_FAILED")
        artifacts = dict(project.artifacts or {})
        source_id = str((artifacts.get("workbench_session") or {}).get("catalog_id") or "saved_project")
        scene = artifacts.get("workbench_scene")
        if scene is None:
            # Older bundles may not carry the scene block.  Reconstruct it from
            # the immutable build rather than silently labelling real data as
            # synthetic; a declared real source without any scene is a hard error.
            scene = self._scene_for_source(source_id, artifacts)
        domain = _project_domain(source_id, scene, artifacts)
        if scene is None:
            if domain == REAL_DOMAIN:
                raise ContractError(
                    "saved project declares a real source but carries no scene context",
                    "SCENE_CONTEXT_MISSING",
                )
            scene = _synthetic_scene_context()
        evidence = self._relation_evidence_for(project, source_id, artifacts)
        ledger_payload = None
        version_tag = ""
        head_path = path / "head.json"
        if head_path.is_file():
            try:
                version_tag = str(read_json(head_path).get("version_dir") or "")
            except Exception:  # noqa: BLE001 - ProjectStore already verified the head
                version_tag = ""
        if version_tag:
            sidecar_path = path / "versions" / version_tag / EVIDENCE_SIDECAR
            if sidecar_path.is_file():
                try:
                    ledger_payload = read_json(sidecar_path)
                except Exception:  # noqa: BLE001 - an unreadable sidecar rebuilds
                    ledger_payload = None
        return self._adopt(project, source_id=source_id, read_only=project.schema_version == SCHEMA_VERSION,
                           scene_context=scene, relation_evidence=evidence, bundle_path=path,
                           saved_hash=project.to_dict()["content_hash"] if receipt.get("verified") else None,
                           evidence_ledger=ledger_payload)

    def save_current(self, *, interrupt: Optional[str] = None) -> Dict[str, Any]:
        """Persist the live project and return a receipt, never an inference.

        ``saved=True`` is only emitted together with a non-empty, verified
        receipt whose content hash equals the session project hash.  A caller
        must never read "the state says version N, so the save worked" -- the
        old flow could not distinguish a saved head from a dirty session.
        """
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only; explicitly migrate a copy first", "LEGACY_READ_ONLY")
            bundle = session.bundle_path or self._bundle_for(session.project.project_id)
            before_head = bundle / "head.json"
            before = read_json(before_head) if before_head.is_file() else None
            current_hash = session.project.compute_content_hash()
            try:
                receipt = ProjectStore(bundle, recompute_script=RECOMPUTE_SCRIPT).save(session.project, interrupt=interrupt)
            except Exception as exc:
                after = read_json(before_head) if before_head.is_file() else None
                failed = {
                    "saved": False, "dirty": True, "persisted": False,
                    "error": _json_error(exc, "SAVE_FAILED"),
                    "rollback": {
                        "previous_head_preserved": before == after,
                        "previous_version": (before or {}).get("project_version"),
                        "current_session_version": session.project.project_version,
                        "current_session_hash_unchanged": True,
                    },
                    "measured_at": _now_iso(),
                }
                session.last_save = {"saved": False, "error": failed["error"], "measured_at": failed["measured_at"]}
                return failed
            body = receipt.to_dict()
            if not body.get("content_hash") or body.get("content_hash") != current_hash:
                raise ContractError("save receipt does not describe the current project",
                                    "SAVE_RECEIPT_MISMATCH")
            if not body.get("verified"):
                raise ContractError("save readback did not verify", "SAVE_READBACK_FAILED")
            session.bundle_path = bundle
            session.saved_hash = body["content_hash"]
            readback = ProjectStore(bundle).verify()
            if not readback.get("verified"):
                raise ContractError("post-save readback did not verify", "SAVE_READBACK_FAILED")
            receipt_body = {
                "saved": True, "dirty": False, "persisted": True,
                "receipt": body, "readback": readback,
                "bundle_id": bundle.name,
                "project_version": session.project.project_version,
                "content_hash": body["content_hash"],
                "measured_at": _now_iso(),
            }
            evidence_public = self._evidence_public(session)
            ledger_document = eledger.ledger_payload(
                session.project, dict((session.evidence_ledger or {}).get("records") or {}),
                context_status=str(evidence_public.get("context_status") or "UNKNOWN"),
                meta={"last_refresh": evidence_public.get("last_refresh")},
            )
            version_dir = bundle / "versions" / str(body.get("version_dir") or "")
            if not version_dir.is_dir():
                raise ContractError("saved version directory is absent", "SAVE_READBACK_FAILED")
            sidecar_path = version_dir / EVIDENCE_SIDECAR
            atomic_write_json(sidecar_path, ledger_document)
            sidecar_sha = file_sha256(sidecar_path)
            if read_json(sidecar_path).get("snapshot_id") != ledger_document["snapshot_id"]:
                raise ContractError("evidence sidecar readback mismatch", "SAVE_READBACK_FAILED")
            receipt_body["evidence"] = {
                "snapshot_id": ledger_document["snapshot_id"],
                "project_content_hash": ledger_document["project_content_hash"],
                "record_count": len(ledger_document["records"]),
                "status_counts": copy.deepcopy(ledger_document["status_counts"]),
                "path": str(sidecar_path.relative_to(bundle.parent)).replace("\\", "/"),
                "sha256": sidecar_sha,
                "verified": True,
            }
            session.last_save = {
                "saved": True, "project_version": receipt_body["project_version"],
                "content_hash": receipt_body["content_hash"], "bundle_id": bundle.name,
                "file_count": body.get("file_count"), "verified": True,
                "evidence_snapshot_id": ledger_document["snapshot_id"],
                "evidence_sha256": sidecar_sha,
                "measured_at": receipt_body["measured_at"],
            }
            return receipt_body

    def state(self) -> Dict[str, Any]:
        with self._lock:
            if self._session is None:
                return {"project_open": False, "library": self.library(),
                        "jobs": self.jobs.list(public=True),
                        "portable": self._portable_public()}
            session = self._require_session()
            project = session.project
            domain = _project_domain(session.source_id, session.scene_context, project.artifacts)
            document = project.to_dict()
            observations = []
            for item in project.observations:
                observations.append({
                    "observation_id": item.observation_id, "unit": item.unit,
                    "coordinate_frame": item.coordinate_frame, "coverage": copy.deepcopy(item.coverage),
                    "license": item.license, "registration": copy.deepcopy(item.registration),
                })
            objects = []
            surface_geometry = (project.artifacts or {}).get("surface_geometry") or {}
            for item in project.fragments:
                quality = dict(item.quality or {})
                objects.append({
                    "id": item.fragment_id, "kind": "fragment", "label": item.fragment_id,
                    "observation_id": item.observation_id, "support_count": len(item.support_ref.indices),
                    "support_indices": list(item.support_ref.indices[:96]),
                    "plane_local": item.plane_local.to_dict(),
                    "geometry": {
                        "region_count": len(item.finite_regions_local),
                        "hole_count": sum(len(region.holes) for region in item.finite_regions_local),
                        "total_area": float(sum(region.to_shapely().area for region in item.finite_regions_local)),
                        "requires_human_review": bool(quality.get("requires_human_review", False)),
                        "geometry_source": quality.get("geometry_source"),
                        "rmse": quality.get("rmse"),
                    },
                    "version": item.version,
                })
            unknown_count = 0
            for item in project.surfaces:
                stored_geometry = surface_geometry.get(item.surface_id) or {}
                objects.append({
                    "id": item.surface_id, "kind": "surface", "label": item.surface_id,
                    "members": list(item.member_fragment_ids), "status": item.interpretation_status,
                    "provenance": item.provenance,
                    "geometry": {
                        "region_count": len(item.finite_support),
                        "hole_count": sum(len(region.holes) for region in item.finite_support),
                        "total_area": float(sum(region.to_shapely().area for region in item.finite_support)),
                        "mode": stored_geometry.get("mode"),
                        "requires_human_review": bool(stored_geometry.get("requires_human_review", False)),
                    },
                    "version": item.version,
                })
            for item in project.relations:
                unknown_count += int(item.state == "unknown")
                objects.append({"id": item.relation_id, "kind": "relation", "label": item.relation_id,
                                "src_id": item.src_id, "dst_id": item.dst_id, "state": item.state,
                                "unknown_reason": item.unknown_reason, "relation_type": item.relation_type,
                                "version": item.version})
            for item in project.protected_regions:
                objects.append({"id": item.region_id, "kind": "protected", "label": item.region_id,
                                "roi_id": item.roi_id})
            evidence_public = self._evidence_public(session)
            return {
                "project_open": True,
                "project_id": project.project_id,
                "project_version": project.project_version,
                "content_hash": document["content_hash"],
                "schema_version": project.schema_version,
                "read_only_legacy": session.read_only,
                "source_id": session.source_id,
                "dirty": session.saved_hash != document["content_hash"],
                "saved_hash": session.saved_hash,
                "saved_state": {
                    # "saved" is a statement about the persisted head, never a
                    # guess from a version number.  A save receipt is the only
                    # thing that sets it.
                    "saved": session.saved_hash is not None and session.saved_hash == document["content_hash"],
                    "dirty": session.saved_hash != document["content_hash"],
                    "saved_hash": session.saved_hash,
                    "last_save": copy.deepcopy(session.last_save),
                },
                "live_preview": self._live_preview_public(session),
                "preview_releases": copy.deepcopy(session.preview_releases[-8:]),
                "bundle_id": session.bundle_path.name if session.bundle_path else None,
                "counts": {key: len(getattr(project, key)) for key in
                           ("observations", "fragments", "surfaces", "relations", "protected_regions")},
                "unknown_relation_count": unknown_count,
                "real_quality": _real_quality_for(domain),
                "support_selectable": session.scene_context.get("source_kind") == "internal_real",
                "source_domain": domain,
                "relation_evidence_count": len(session.relation_evidence),
                "evidence": evidence_public,
                "evidence_snapshot_id": evidence_public["snapshot_id"],
                "evidence_status_counts": copy.deepcopy(evidence_public["status_counts"]),
                "evidence_excluded_from_solver": evidence_public["excluded_from_solver"],
                "geometry_policy": "REAL_SUPPORT_POINTS_ONLY_WHEN_SOURCE_IS_INTERNAL_REAL",
                "metric_or_geographic_conclusions_emitted": False,
                "observations": observations,
                "objects": objects,
                "can_undo": bool(session.engine.undo_stack) and not session.read_only,
                "can_redo": bool(session.engine.redo_stack) and not session.read_only,
                "events": copy.deepcopy(session.engine.events[-40:]),
                "parameters": copy.deepcopy(project.parameters),
                "library": self.library(),
                "jobs": self.jobs.list(public=True),
                "portable": self._portable_public(),
            }

    def object_detail(self, object_id: str) -> Dict[str, Any]:
        session = self._require_session()
        for collection, id_field in (("fragments", "fragment_id"), ("surfaces", "surface_id"),
                                     ("relations", "relation_id"), ("protected_regions", "region_id"),
                                     ("observations", "observation_id")):
            for item in getattr(session.project, collection):
                if getattr(item, id_field) == object_id:
                    return {"kind": collection, "object": item.to_dict()}
        raise ContractError("object ID is not present", "OBJECT_NOT_FOUND")

    def evidence(self, object_id: str) -> Dict[str, Any]:
        session = self._require_session()
        project = session.project
        record = ((session.evidence_ledger or {}).get("records") or {}).get(object_id)
        if record is not None:
            basis_now = eledger.relation_basis_map(project).get(object_id)
            return {
                "object_id": object_id, "kind": "relation",
                "relation_evidence": copy.deepcopy(record),
                "status": record.get("status"),
                "basis_current": record.get("basis") == basis_now,
                "current_supports_solver": record.get("status") == eledger.STATUS_CURRENT,
                "note": "stale/conflict/out-of-scope evidence is kept for comparison; only current evidence supports a conclusion",
            }
        domain = _project_domain(session.source_id, session.scene_context, project.artifacts)
        fragments = {item.fragment_id: item for item in project.fragments}
        surfaces = {item.surface_id: item for item in project.surfaces}
        fragment = fragments.get(object_id)
        related_surface = surfaces.get(object_id)
        if fragment is None and related_surface is not None and related_surface.member_fragment_ids:
            fragment = fragments.get(related_surface.member_fragment_ids[0])
        if fragment is not None:
            observation = next(item for item in project.observations if item.observation_id == fragment.observation_id)
            support = fragment.support_ref
            points = self._support_points(session, fragment, limit=600)
            return {
                "object_id": object_id, "fragment_id": fragment.fragment_id,
                "observation_id": observation.observation_id,
                "source_files": [item.to_dict() for item in observation.source_files],
                "unit": observation.unit, "coordinate_frame": observation.coordinate_frame,
                "license": observation.license, "registration": copy.deepcopy(observation.registration),
                "support": support.to_dict(),
                "support_indices_preview": list(support.indices[:80]),
                "support_point_preview": points,
                "point_preview_policy": "ORIGINAL_POINT_INDEX" if session.scene_context.get("source_kind") == "internal_real"
                                        else "SYNTHETIC_SUPPORT_RECONSTRUCTION_NOT_MEASUREMENT",
                "real_quality": (_real_quality_for(domain) if domain == REAL_DOMAIN
                                 else "not_applicable_synthetic"),
            }
        detail = self.object_detail(object_id)
        return {"object_id": object_id, "object": detail["object"],
                "evidence_refs": detail["object"].get("evidence_refs") or [],
                "note": "关系与保护对象通过其端点/ROI回到支持证据；无伪造点坐标。"}

    # -- preview lifecycle -----------------------------------------------------
    # Only one preview token may be live per session.  Every exit path --
    # commit, cancel, expiry, supersede, version conflict, any rejected commit --
    # releases the token so it can never keep occupying its base version and
    # turn the next legitimate hard edit into a HARD_EDIT_CONFLICT.

    def _preview_ttl(self, session: Session) -> float:
        return float(getattr(session.engine, "_ttl", 900.0))

    def _release_live_preview(self, session: Session, reason: str) -> Optional[Dict[str, Any]]:
        entry = session.live_preview
        if not entry:
            return None
        token = str(entry.get("token") or "")
        session.live_preview = None
        released = False
        if token:
            released = bool(session.engine.discard(token))
            session.cancelled_tokens.add(token)
        receipt = {
            "token": token, "released": released, "reason": reason,
            "base_version": entry.get("base_version"),
            "measured_at": _now_iso(),
        }
        session.preview_releases.append(receipt)
        del session.preview_releases[:-20]
        return receipt

    def _expire_live_preview(self, session: Session) -> Optional[Dict[str, Any]]:
        entry = session.live_preview
        if not entry:
            return None
        age = time.monotonic() - float(entry.get("created_monotonic") or time.monotonic())
        if age <= self._preview_ttl(session):
            return None
        return self._release_live_preview(session, "EXPIRED_TOKEN")

    def _live_preview_public(self, session: Session) -> Optional[Dict[str, Any]]:
        entry = session.live_preview
        if not entry:
            return None
        age = time.monotonic() - float(entry.get("created_monotonic") or time.monotonic())
        return {
            "token": entry.get("token"),
            "base_version": entry.get("base_version"),
            "action": entry.get("action"),
            "target_ids": list(entry.get("target_ids") or []),
            "diff": copy.deepcopy(entry.get("diff") or {}),
            "geometry_preview": copy.deepcopy(entry.get("geometry_preview") or {}),
            "warnings": list(entry.get("warnings") or []),
            "created_at": entry.get("created_at"),
            "age_seconds": round(age, 3),
            "expires_in_seconds": round(max(0.0, self._preview_ttl(session) - age), 3),
        }

    def preview(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only; explicitly migrate a copy first", "LEGACY_READ_ONLY")
            expired = self._expire_live_preview(session)
            draft = EditDraft(
                action=str(body.get("action")), target_ids=list(body.get("target_ids") or []),
                payload=copy.deepcopy(body.get("payload") or {}), base_version=int(body.get("base_version")),
                actor="local_browser_user", hardness=str(body.get("hardness") or "hard"),
                tolerance=float(body.get("tolerance") or 0.0), created_at=_now_iso(),
            )
            fingerprint = canonical_json({
                "action": draft.action, "target_ids": sorted(str(item) for item in draft.target_ids),
                "payload": draft.payload, "base_version": draft.base_version, "hardness": draft.hardness,
            })
            live = session.live_preview
            if live and live.get("fingerprint") == fingerprint:
                # The same draft arrived again: a client retry, a double click or
                # a replay of the same button press.  Returning the live token is
                # the only safe answer -- minting a second preview would release
                # the first one and the in-flight response would carry a dead
                # token (measured: a >60 s preview made the browser retry and the
                # first response's token was already superseded).
                return {
                    "preview": copy.deepcopy(live.get("public") or {}),
                    "project_unchanged": True, "idempotent": True,
                    "expired_token": (expired or {}).get("token"),
                    "superseded_token": None,
                    "live_preview": self._live_preview_public(session),
                }
            superseded = self._release_live_preview(session, "SUPERSEDED_BY_NEW_PREVIEW")
            started = time.monotonic()
            # compute_content_hash() serves the sealed digest of this immutable
            # snapshot; to_dict() deep-copied the 724 kB document twice per
            # preview merely to read one string (measured 0.08 s).
            before = session.project.compute_content_hash()
            preview = session.engine.preview(draft)
            elapsed = round(time.monotonic() - started, 3)
            assert session.project.compute_content_hash() == before
            public = preview.to_dict()
            session.live_preview = {
                "token": public["token"],
                "base_version": public["base_version"],
                "action": public["action"],
                "target_ids": list(public["target_ids"]),
                "diff": copy.deepcopy(public["diff"]),
                "geometry_preview": copy.deepcopy(public.get("geometry_preview") or {}),
                "warnings": list(public.get("warnings") or []),
                "created_at": public["created_at"],
                "created_monotonic": time.monotonic(),
                "fingerprint": fingerprint,
                "public": public,
                "compute_seconds": elapsed,
            }
            return {
                "preview": public, "project_unchanged": True, "idempotent": False,
                "compute_seconds": elapsed,
                "expired_token": (expired or {}).get("token"),
                "superseded_token": (superseded or {}).get("token"),
                "live_preview": self._live_preview_public(session),
            }

    def commit(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only", "LEGACY_READ_ONLY")
            token = str(body.get("token") or "")
            if token in session.cancelled_tokens:
                raise PreviewTokenError("CANCELLED_TOKEN")
            try:
                receipt = session.engine.commit(token, int(body.get("base_version")))
            except PartialWriteError:
                # The engine marks a partial write as used, so the slot must be
                # released here or it stays an inert hard sibling forever.
                self._release_live_preview(session, "COMMIT_PARTIAL_WRITE")
                raise
            except PreviewTokenError as exc:
                # Only poison the live slot when this attempt actually addressed
                # the live token.  A forged token or a replay after a successful
                # commit must not destroy a valid preview the user is still
                # looking at (the previous blanket release turned every rejected
                # attempt into CANCELLED_TOKEN for the legitimate live token).
                live = session.live_preview
                if live and live.get("token") == token:
                    self._release_live_preview(session, "TOKEN_REJECTED_" + str(exc.code))
                raise
            except VersionConflictError:
                # A stale base_version proves only that this *attempt* cannot
                # land.  The preview itself is still valid at its own base
                # version (the workbench test commits it afterwards at that
                # base), so the slot and the token must survive.  The rejection
                # is remembered; if the front end then cancels the dangling
                # token, the release receipt names this reason.
                session.commit_rejections[token] = "COMMIT_REJECTED_STALE_VERSION"
                raise
            session.project = session.engine.project
            session.live_preview = None
            changed = list(getattr(receipt, "changed_ids", []) or []) + list(getattr(receipt, "affected_ids", []) or [])
            self._refresh_evidence(session, changed, reason="COMMIT")
            return {"transaction": receipt.to_dict(), "persisted": False, "dirty": True,
                    "state": self.state()}

    def cancel_preview(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            token = str(body.get("token") or "")
            if not token:
                raise ContractError("preview token is required", "PREVIEW_TOKEN_INVALID")
            before = session.project.compute_content_hash()
            # Releasing the engine slot is what makes the cancel real: the
            # token becomes uncommittable and stops blocking the next hard edit.
            released = session.engine.discard(token)
            session.cancelled_tokens.add(token)
            entry = session.live_preview if session.live_preview and session.live_preview.get("token") == token else None
            if entry:
                session.live_preview = None
            if released:
                session.preview_releases.append({
                    "token": token, "released": True,
                    "reason": session.commit_rejections.pop(token, "CANCELLED_BY_USER"),
                    "base_version": (entry or {}).get("base_version"),
                    "measured_at": _now_iso(),
                })
                del session.preview_releases[:-20]
            else:
                session.commit_rejections.pop(token, None)
            return {"cancelled": True, "token_released": released,
                    "project_unchanged": before == session.project.compute_content_hash()}

    def undo(self, base_version: int) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only", "LEGACY_READ_ONLY")
            check_base_version(base_version, session.project.project_version)
            before = session.project
            receipt = session.engine.undo()
            session.project = session.engine.project
            self._refresh_evidence(session, eledger.diff_object_ids(before, session.project), reason="UNDO")
            return {"transaction": receipt.to_dict(), "state": self.state()}

    def redo(self, base_version: int) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only", "LEGACY_READ_ONLY")
            check_base_version(base_version, session.project.project_version)
            before = session.project
            receipt = session.engine.redo()
            session.project = session.engine.project
            self._refresh_evidence(session, eledger.diff_object_ids(before, session.project), reason="REDO")
            return {"transaction": receipt.to_dict(), "state": self.state()}

    def start_solve(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if session.read_only:
                raise ContractError("legacy v1 is read-only; migrate before solving", "LEGACY_READ_ONLY")
            check_base_version(int(body.get("base_version")), session.project.project_version)
            config_body = dict(body.get("config") or {})
            internal = session.source_id == "internal_pf02"
            config = RuleConfig(
                mode="manual",
                max_iterations=int(config_body.get("max_iterations", 1 if internal else 2)),
                recompute_scope=str(config_body.get("recompute_scope", "full" if internal else "local")),
                association_threshold=float(config_body.get("association_threshold", 1.0 if internal else 0.75)),
                relation_threshold=float(config_body.get("relation_threshold", 1.0 if internal else 0.60)),
                seed_ids=tuple(config_body.get("seed_ids") or ()),
            )
            # W153: no stale result may enter a solve.  Refresh first, then hand
            # the solver only records whose basis is current for this version.
            refresh = self._refresh_evidence(session, [], reason="PRE_SOLVE")
            current = eledger.current_evidence(session.evidence_ledger.get("records") or {})
            public = self._evidence_public(session)
            # T-D: pin the original measurement before the worker starts.  A real
            # project whose source is missing, substituted, or a display-only
            # asset raises here (fail closed) instead of letting the worker fall
            # back to the stored, stale geometry.
            geometry_source = build_geometry_source(session.project, session.scene_context)
            return self.jobs.submit(
                project=session.project, config=config, events=list(body.get("events") or []),
                relation_evidence=current,
                evidence_snapshot_id=public["snapshot_id"],
                evidence_total=int(public["relation_count"]),
                evidence_excluded=int(public["excluded_from_solver"]),
                protected=ProtectionSet.from_value(body.get("protected")),
                timeout_seconds=float(body.get("timeout_seconds", 30.0)),
                solver_mode=str(body.get("solver_mode") or "rule"),
                fault=str(body.get("fault") or "none"),
                resource_limit_points=int(body.get("resource_limit_points", 2_000_000)),
                resource_limit_fragments=int(body.get("resource_limit_fragments", 5_000)),
                geometry_source=geometry_source,
            )

    def apply_job(self, job_id: str, base_version: int) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            check_base_version(base_version, session.project.project_version)
            job = self.jobs.get(job_id)
            if job["project_id"] != session.project.project_id or job["base_version"] != base_version:
                raise ContractError("job belongs to another project or base version", "JOB_CONTEXT_MISMATCH")
            if job["base_content_hash"] != session.project.compute_content_hash():
                raise VersionConflictError(job["base_version"], session.project.project_version)
            if job["status"] != "succeeded":
                raise ContractError("failed/cancelled/timeout jobs have no publishable candidate", "JOB_NOT_SUCCESSFUL")
            result = job.get("result") or {}
            project = ProjectSnapshot.from_dict(result["project"])
            if project.project_id != session.project.project_id:
                raise ContractError("worker returned another project", "JOB_CONTEXT_MISMATCH")
            if project.to_dict()["content_hash"] != result["update"]["content_hash"]:
                raise ContractError("worker result hash mismatch", "JOB_RESULT_INVALID")
            self.jobs.mark_applied(job_id)
            old_project = session.project
            old_records = list((session.evidence_ledger.get("records") or {}).values())
            self._session = self._make_session(
                project, source_id=session.source_id, read_only=False,
                scene_context=session.scene_context, relation_evidence=session.relation_evidence,
                bundle_path=session.bundle_path, saved_hash=session.saved_hash,
                evidence_ledger={"records": old_records},
            )
            self._refresh_evidence(self._session, eledger.diff_object_ids(old_project, project),
                                   reason="SOLVE_APPLY")
            return {"applied": True, "update": result["update"], "state": self.state()}

    def _points_for_observation(self, session: Session, observation_id: str) -> Optional[np.ndarray]:
        return self._raw_points_for(session.project, session.scene_context, observation_id)

    def _support_points(self, session: Session, fragment: Any, *, limit: int) -> List[List[float]]:
        indices = list(fragment.support_ref.indices)
        if not indices:
            return []
        chosen = np.asarray(indices, dtype=np.int64)
        if len(chosen) > limit:
            chosen = chosen[np.linspace(0, len(chosen) - 1, limit).astype(int)]
        points = self._points_for_observation(session, fragment.observation_id)
        origin = np.asarray(session.scene_context.get("frame", {}).get("origin") or [0, 0, 0], dtype=np.float64)
        if points is not None:
            valid = chosen[(chosen >= 0) & (chosen < len(points))]
            return (points[valid] - origin).tolist()
        # Synthetic fallback is explicitly marked as reconstructed evidence, not raw measurement.
        frame = self._frame_for_fragment(session.project, session.scene_context, fragment)
        if frame is not None:
            n = np.asarray(frame["normal"], dtype=np.float64)
            u = np.asarray(frame["basis_u"], dtype=np.float64)
            v = np.asarray(frame["basis_v"], dtype=np.float64)
            center = np.asarray(frame["origin"], dtype=np.float64)
        else:
            candidate = session.scene_context.get("candidates", {}).get(fragment.fragment_id, {})
            n, u, v = _basis(fragment.plane_local.normal)
            center = np.asarray(candidate.get("center_local") or (n * fragment.plane_local.offset),
                                dtype=np.float64)
        ring = fragment.finite_regions_local[0].exterior
        result = []
        for index in chosen:
            position = float(index % max(1, len(ring) - 1))
            left = int(math.floor(position))
            right = min(left + 1, len(ring) - 1)
            fraction = position - left
            uv = (1 - fraction) * np.asarray(ring[left]) + fraction * np.asarray(ring[right])
            xyz = center + uv[0] * u + uv[1] * v
            result.append(xyz.tolist())
        return result

    def scene(self, *, max_raw_points: int = 6500) -> Dict[str, Any]:
        session = self._require_session()
        project = session.project
        evidence_public = self._evidence_public(session)
        context = session.scene_context
        candidates = context.get("candidates") or {}
        fragment_by_id = {item.fragment_id: item for item in project.fragments}
        traces: List[Dict[str, Any]] = []
        raw_total = 0
        for observation in project.observations:
            points = self._points_for_observation(session, observation.observation_id)
            if points is None:
                # Deterministic finite-support reconstruction for the self-built synthetic scene.
                support_points = []
                support_meta = []
                for fragment in project.fragments:
                    if fragment.observation_id == observation.observation_id:
                        reconstructed = self._support_points(session, fragment, limit=80)
                        support_points.extend(reconstructed)
                        chosen = list(fragment.support_ref.indices)
                        if len(chosen) > 80:
                            chosen = [chosen[int(round(pos))] for pos in np.linspace(0, len(chosen) - 1, 80)]
                        support_meta.extend([[fragment.fragment_id, int(index)]
                                             for index in chosen[:len(reconstructed)]])
                display = np.asarray(support_points, dtype=np.float64) if support_points else np.empty((0, 3))
                provenance = "SYNTHETIC_SUPPORT_RECONSTRUCTION_NOT_MEASUREMENT"
                customdata = support_meta
                selectable = False
            else:
                origin = np.asarray(context.get("frame", {}).get("origin") or [0, 0, 0], dtype=np.float64)
                indices = np.arange(len(points))
                if len(points) > max_raw_points:
                    indices = np.linspace(0, len(points) - 1, max_raw_points).astype(int)
                display = points[indices] - origin
                provenance = "ORIGINAL_POINT_CLOUD_DECIMATED_FOR_DISPLAY"
                customdata = [int(item) for item in indices]
                selectable = True
            raw_total += len(display)
            if len(display):
                traces.append({
                    "type": "scatter3d", "mode": "markers", "name": "原始点 / " + observation.observation_id,
                    "x": display[:, 0].tolist(), "y": display[:, 1].tolist(), "z": display[:, 2].tolist(),
                    "customdata": customdata,
                    "marker": {"size": 1.8, "color": "#7e8b91", "opacity": 0.48},
                    "meta": {"layer": "points", "object_id": observation.observation_id,
                             "observation_id": observation.observation_id,
                             "provenance": provenance,
                             "selectable_support": bool(selectable)},
                    "hovertemplate": "%{meta.object_id}<br>显示坐标 %{x:.3f}, %{y:.3f}, %{z:.3f}<br>原始索引 %{customdata}<extra></extra>",
                })

        surface_centers: Dict[str, np.ndarray] = {}
        surface_geometry = (project.artifacts or {}).get("surface_geometry") or {}
        for surface in project.surfaces:
            if not surface.member_fragment_ids:
                continue
            fragment = fragment_by_id.get(surface.member_fragment_ids[0])
            if fragment is None:
                continue
            stored_geometry = surface_geometry.get(surface.surface_id) or {}
            stored_frame = stored_geometry.get("frame_local")
            frame = None
            if isinstance(stored_frame, Mapping) and stored_frame:
                try:
                    frame = normalise_support_frame(stored_frame)
                except ContractError:
                    frame = None
            if frame is None:
                frame = self._frame_for_fragment(project, context, fragment)
            if frame is not None:
                n = np.asarray(frame["normal"], dtype=np.float64)
                u = np.asarray(frame["basis_u"], dtype=np.float64)
                v = np.asarray(frame["basis_v"], dtype=np.float64)
                center = np.asarray(frame["origin"], dtype=np.float64)
            else:
                candidate = candidates.get(fragment.fragment_id, {})
                n, fallback_u, fallback_v = _basis(fragment.plane_local.normal)
                u = np.asarray(candidate.get("basis_u") or fallback_u, dtype=np.float64)
                v = np.asarray(candidate.get("basis_v") or fallback_v, dtype=np.float64)
                center = np.asarray(candidate.get("center_local") or (n * fragment.plane_local.offset), dtype=np.float64)
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            ii: List[int] = []
            jj: List[int] = []
            kk: List[int] = []
            centroid_parts = []
            for region in surface.finite_support:
                polygon = region.to_shapely()
                if polygon.is_empty:
                    continue
                triangles = shapely.constrained_delaunay_triangles(polygon)
                for triangle in triangles.geoms:
                    coords = list(triangle.exterior.coords)[:3]
                    base = len(xs)
                    for uv in coords:
                        xyz = center + float(uv[0]) * u + float(uv[1]) * v
                        xs.append(float(xyz[0])); ys.append(float(xyz[1])); zs.append(float(xyz[2]))
                        centroid_parts.append(xyz)
                    ii.append(base); jj.append(base + 1); kk.append(base + 2)
                # Holes are separate visible evidence lines; they are never filled by measurement logic.
                for ring in [polygon.exterior] + list(polygon.interiors):
                    line_xyz = [center + float(uv[0]) * u + float(uv[1]) * v for uv in list(ring.coords)]
                    traces.append({
                        "type": "scatter3d", "mode": "lines", "name": "有限边界 / " + surface.surface_id,
                        "x": [p[0] for p in line_xyz], "y": [p[1] for p in line_xyz], "z": [p[2] for p in line_xyz],
                        "line": {"color": "#d8b46a" if ring == polygon.exterior else "#ef7d68", "width": 3},
                        "meta": {"layer": "surfaces", "object_id": surface.surface_id, "hole": ring != polygon.exterior},
                        "hoverinfo": "name", "showlegend": False,
                    })
            if xs:
                color = "#4fb3b0" if surface.interpretation_status != "human_reviewed" else "#d8b46a"
                traces.append({
                    "type": "mesh3d", "name": surface.surface_id,
                    "x": xs, "y": ys, "z": zs, "i": ii, "j": jj, "k": kk,
                    "color": color, "opacity": 0.62, "flatshading": True,
                    "meta": {"layer": "surfaces", "object_id": surface.surface_id,
                             "geometry_mode": stored_geometry.get("mode") or "UNRECORDED",
                             "requires_human_review": bool(stored_geometry.get("requires_human_review", False))},
                    "hovertemplate": surface.surface_id + "<extra></extra>",
                })
                surface_centers[surface.surface_id] = np.mean(np.asarray(centroid_parts), axis=0)

        relation_groups: Dict[str, Dict[str, List[Any]]] = {
            "yes": {"x": [], "y": [], "z": [], "ids": []},
            "no": {"x": [], "y": [], "z": [], "ids": []},
            "unknown": {"x": [], "y": [], "z": [], "ids": []},
        }
        for relation in project.relations:
            if relation.src_id not in surface_centers or relation.dst_id not in surface_centers:
                continue
            left, right = surface_centers[relation.src_id], surface_centers[relation.dst_id]
            state = relation.state if relation.state in relation_groups else "unknown"
            group = relation_groups[state]
            # None separators keep every relation individually addressable in one
            # trace.  256 per-relation traces made the internal PF02 scene take
            # ~30 s to paint in headless Chrome and froze real clicks; three
            # batched traces carry the same geometry.
            group["x"].extend([float(left[0]), float(right[0]), None])
            group["y"].extend([float(left[1]), float(right[1]), None])
            group["z"].extend([float(left[2]), float(right[2]), None])
            group["ids"].append(relation.relation_id)
        for state, group in relation_groups.items():
            if not group["ids"]:
                continue
            color = "#73c991" if state == "yes" else "#ef7d68" if state == "no" else "#9a8fb5"
            traces.append({
                "type": "scatter3d", "mode": "lines", "name": "关系 / " + state,
                "x": group["x"], "y": group["y"], "z": group["z"],
                "line": {"color": color, "width": 2, "dash": "dash" if state == "unknown" else "solid"},
                "meta": {"layer": "relations", "object_id": "relation_group:" + state,
                         "state": state, "relation_ids": group["ids"], "batched": True},
                "hovertemplate": "关系组 " + state + " · " + str(len(group["ids"])) + " 条<extra></extra>",
                "showlegend": False,
            })

        for protected in project.protected_regions:
            for region in protected.finite_regions:
                polygon = region.to_shapely()
                # Protection is in ROI-local 2D; absent a registered 3D plane it
                # is drawn on z=0 with an explicit hover limitation.
                coords = list(polygon.exterior.coords)
                traces.append({
                    "type": "scatter3d", "mode": "lines", "name": "保护 / " + protected.region_id,
                    "x": [p[0] for p in coords], "y": [p[1] for p in coords], "z": [0.0] * len(coords),
                    "line": {"color": "#f0a65a", "width": 5},
                    "meta": {"layer": "protected", "object_id": protected.region_id,
                             "display_limit": "ROI_LOCAL_Z0_NOT_REGISTERED_3D"},
                    "hovertemplate": protected.region_id + "<br>ROI局部投影，非三维配准<extra></extra>",
                })
        return {
            "traces": traces,
            "project_version": project.project_version,
            "content_hash": project.to_dict()["content_hash"],
            "trace_count": len(traces),
            "raw_display_point_count": raw_total,
            "surface_count": len(project.surfaces),
            "relation_count": len(project.relations),
            "unknown_relation_count": sum(item.state == "unknown" for item in project.relations),
            "hole_count": sum(len(region.holes) for item in project.surfaces for region in item.finite_support),
            "support_selectable": context.get("source_kind") == "internal_real",
            "surface_geometry_modes": {
                surface.surface_id: (surface_geometry.get(surface.surface_id) or {}).get("mode")
                for surface in project.surfaces
            },
            "display_coordinates": "SOURCE_MINUS_LOCAL_ORIGIN" if context.get("source_kind") == "internal_real" else "SYNTHETIC_LOCAL",
            "unit": project.observations[0].unit if project.observations else "UNKNOWN",
            "evidence_snapshot_id": evidence_public["snapshot_id"],
            "evidence_status_counts": copy.deepcopy(evidence_public["status_counts"]),
            "evidence_excluded_from_solver": evidence_public["excluded_from_solver"],
            "not_a_measurement_rescaling": True,
        }

    def export_current(self) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            project = session.project
            domain = _project_domain(session.source_id, session.scene_context, project.artifacts)
            document = project.to_dict()
            export_id = "%s_v%06d_%s_%s" % (_safe_name(project.project_id), project.project_version,
                                            document["content_hash"][:12], uuid.uuid4().hex[:8])
            directory = self.exports_root / export_id
            directory.mkdir(parents=True, exist_ok=False)
            files = ["project.json", "relations.json", "surfaces.csv", "geometry.json", "evidence_ledger.json"]
            evidence_public = self._evidence_public(session)
            evidence_records = dict((session.evidence_ledger or {}).get("records") or {})
            evidence_document = eledger.ledger_payload(
                project, evidence_records,
                context_status=str(evidence_public.get("context_status") or "UNKNOWN"),
                meta={"last_refresh": evidence_public.get("last_refresh")},
            )
            evidence_rows = []
            for item in project.relations:
                row = item.to_dict()
                record = evidence_records.get(item.relation_id) or {}
                row["evidence_status"] = record.get("status")
                row["evidence_state"] = record.get("state")
                evidence_rows.append(row)
            atomic_write_json(directory / "project.json", document)
            atomic_write_json(directory / "relations.json", {
                "project_id": project.project_id, "project_version": project.project_version,
                "content_hash": document["content_hash"],
                "evidence_snapshot_id": evidence_public["snapshot_id"],
                "evidence_status_counts": copy.deepcopy(evidence_public["status_counts"]),
                "relations": evidence_rows,
                "unknown_relation_count": sum(item.state == "unknown" for item in project.relations),
            })
            geometry = []
            fragments = {item.fragment_id: item for item in project.fragments}
            surface_geometry = (project.artifacts or {}).get("surface_geometry") or {}
            for surface in project.surfaces:
                geometry.append({
                    "surface_id": surface.surface_id, "member_fragment_ids": list(surface.member_fragment_ids),
                    "finite_support": [item.to_dict() for item in surface.finite_support],
                    "member_planes_local": {member: fragments[member].plane_local.to_dict()
                                             for member in surface.member_fragment_ids if member in fragments},
                    "provenance": surface.provenance, "interpretation_status": surface.interpretation_status,
                    "geometry_mode": (surface_geometry.get(surface.surface_id) or {}).get("mode"),
                    "geometry_frame_local": (surface_geometry.get(surface.surface_id) or {}).get("frame_local"),
                })
            fragment_geometry = []
            for fragment in project.fragments:
                quality = dict(fragment.quality or {})
                fragment_geometry.append({
                    "fragment_id": fragment.fragment_id,
                    "observation_id": fragment.observation_id,
                    "support_ref": fragment.support_ref.to_dict(),
                    "plane_local": fragment.plane_local.to_dict(),
                    "finite_regions_local": [item.to_dict() for item in fragment.finite_regions_local],
                    "quality": {
                        key: quality.get(key) for key in
                        ("rmse", "max_abs_residual", "plane_tolerance", "region_areas", "hole_count",
                         "geometry_source", "fit_method", "fit_config_hash", "requires_human_review",
                         "frame_local", "refit_event_id", "refit_action")
                    },
                    "version": fragment.version,
                })
            atomic_write_json(directory / "geometry.json", {
                "project_id": project.project_id, "project_version": project.project_version,
                "content_hash": document["content_hash"], "surfaces": geometry,
                "fragments": fragment_geometry,
                "surface_geometry": copy.deepcopy(surface_geometry),
                "unit_statuses": [item.unit for item in project.observations],
                "coordinate_frames": [item.coordinate_frame for item in project.observations],
                "measurement_policy": "FINITE_LOCAL_GEOMETRY_ONLY_NO_METRIC_OR_GEOGRAPHIC_CONCLUSION",
                "support_policy": "EXPORTED_SUPPORT_INDICES_REFER_TO_ORIGINAL_OBSERVATION_POINTS",
            })
            rows = []
            for surface in project.surfaces:
                rows.append([
                    project.project_id, project.project_version, surface.surface_id, surface.version,
                    canonical_json(surface.member_fragment_ids), surface.interpretation_status, surface.provenance,
                    len(surface.finite_support), sum(len(item.holes) for item in surface.finite_support),
                    ";".join(item.unit for item in project.observations),
                    _measurement_status_for(domain),
                ])
            csv_text = "\ufeff" + "project_id,project_version,surface_id,surface_version,member_fragment_ids,interpretation_status,provenance,finite_region_count,hole_count,unit_status,measurement_status\n"
            import io
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerows(rows)
            csv_text += buffer.getvalue()
            _atomic_text(directory / "surfaces.csv", csv_text)
            atomic_write_json(directory / "evidence_ledger.json", evidence_document)
            manifest = {
                "schema": "fractureflow.surface_product.workbench_export.v1",
                "export_id": export_id, "project_id": project.project_id,
                "project_version": project.project_version,
                "schema_version": project.schema_version,
                "content_hash": document["content_hash"],
                "object_counts": {key: len(getattr(project, key)) for key in
                                  ("observations", "fragments", "surfaces", "relations", "protected_regions")},
                "unknown_relation_count": sum(item.state == "unknown" for item in project.relations),
                "unit_statuses": [item.unit for item in project.observations],
                "coordinate_frames": [item.coordinate_frame for item in project.observations],
                "license_statuses": [item.license for item in project.observations],
                "source_domain": domain,
                "real_quality": _real_quality_for(domain),
                "evidence": {
                    "schema": evidence_document["schema"],
                    "method": evidence_document["method"],
                    "snapshot_id": evidence_document["snapshot_id"],
                    "project_content_hash": evidence_document["project_content_hash"],
                    "status_counts": copy.deepcopy(evidence_document["status_counts"]),
                    "record_count": len(evidence_document["records"]),
                    "context_status": evidence_document["context_status"],
                    "file": "evidence_ledger.json",
                },
                "metric_or_geographic_conclusions_emitted": False,
                "files": {name: file_sha256(directory / name) for name in files},
                "measured_at": _now_iso(), "recompute_script": RECOMPUTE_SCRIPT,
            }
            atomic_write_json(directory / "manifest.json", manifest)
            command = [sys.executable, "-m", "fractureflow.surface_product.workbench_service", "verify-export", str(directory)]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            child_env = dict(os.environ)
            child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                       timeout=30, check=False, shell=False, creationflags=flags, env=child_env)
            if completed.returncode != 0:
                raise ContractError("new-process export readback failed: " + (completed.stderr or completed.stdout),
                                    "EXPORT_READBACK_FAILED")
            readback = json.loads(completed.stdout)
            if not readback.get("verified"):
                raise ContractError("new-process export readback did not verify", "EXPORT_READBACK_FAILED")
            if int(readback.get("enumeration_count") or 0) <= 0:
                raise ContractError("new-process export readback enumerated zero checks", "EMPTY_ENUMERATION")
            atomic_write_json(directory / "readback.json", readback)
            manifest["readback"] = {"path": "readback.json", "sha256": file_sha256(directory / "readback.json"),
                                    "verified": bool(readback.get("verified")), "process_returncode": completed.returncode}
            manifest["evidence"]["sha256"] = file_sha256(directory / "evidence_ledger.json")
            atomic_write_json(directory / "manifest.json", manifest)
            return {
                "exported": True, "export_id": export_id,
                # W170 W169C-04：导出目录在盘路径原文（绝对路径），供界面回执/清单展示与接收方定位。
                "export_dir": str(directory),
                "project_version": project.project_version,
                "content_hash": document["content_hash"],
                "schema_version": project.schema_version,
                "enumeration_count": int(readback.get("enumeration_count") or 0),
                "manifest": manifest, "readback": readback,
                "evidence": {key: copy.deepcopy(manifest["evidence"][key]) for key in
                             ("snapshot_id", "project_content_hash", "status_counts", "record_count",
                              "context_status", "file", "sha256")},
                "files": [{"name": name, "download_url": "/download/%s/%s" % (export_id, name)}
                          for name in files + ["manifest.json", "readback.json"]],
            }

    # -- W163: W154 portable package, wired into the workbench ---------------
    #
    # The portable package stays a *read-only view*: opening one never replaces
    # the editing session and never reconstructs synthetic support points for a
    # missing real source.  Every number shown here comes from
    # portable_project.open_package / relocate_asset on the package on disk.

    # 0 = hash every referenced file on open.  W154's own reopen semantics
    # require the measurement assets to be hash-matched before evidence can be
    # called current; deferring the 1.63 GB raw cloud would make every real
    # package open as "not current" for a bookkeeping reason, not a real one.
    PORTABLE_HASH_LIMIT_BYTES = 0
    PORTABLE_EXPORT_HASH_LIMIT_BYTES = 512 * 1024 * 1024

    def _portable_roots(self) -> List[Path]:
        return [path.resolve() for path in (
            self.workspace / "portable",
            ROOT / "results" / "w154_portable_project" / "delivery" / "examples",
            ROOT / "results" / "w163_delivery_integration" / "fixtures",
        )]

    def portable_roots(self) -> List[str]:
        return [str(path) for path in self._portable_roots()]

    def _portable_resolve(self, value: Any) -> Path:
        raw = str(value or "").strip().strip('"')
        if not raw:
            raise ContractError("portable package path is required", "PORTABLE_PATH_REQUIRED")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        for root in self._portable_roots():
            if candidate != root and root not in candidate.parents:
                continue
            if candidate.suffix.lower() != ".ffportable":
                raise ContractError("not a .ffportable package: %s" % candidate,
                                    "PORTABLE_NOT_A_PACKAGE")
            if not candidate.is_dir():
                raise ContractError("portable package directory is missing: %s" % candidate,
                                    "PORTABLE_NOT_FOUND")
            return candidate
        raise ContractError("portable package is outside the allowed roots: %s" % candidate,
                            "PORTABLE_PATH_NOT_ALLOWED")

    def _portable_entry(self, package: Path) -> Dict[str, Any]:
        try:
            # Not contracts.read_json: the portable manifest legitimately carries
            # keys such as assets.by_storage["REFERENCE"], which the project-JSON
            # hidden-reference filter rejects.  Parse strictly, then read.
            manifest = json.loads((package / portablep.MANIFEST_NAME).read_text(encoding="utf-8"),
                                  parse_constant=_reject_non_finite)
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest is not an object")
            project = dict(manifest.get("project") or {})
            return {
                "path": str(package), "name": package.name, "readable": True,
                "package_id": manifest.get("package_id"),
                "format_version": manifest.get("format_version"),
                "distribution_mode": manifest.get("distribution_mode"),
                "source_domain": manifest.get("source_domain"),
                "created_at": manifest.get("created_at"),
                "project": {key: project.get(key) for key in
                            ("project_id", "project_version", "content_hash", "geometry_digest")},
                "object_counts": dict(project.get("object_counts") or {}),
                "evidence": dict(manifest.get("evidence") or {}),
            }
        except Exception as exc:  # noqa: BLE001 - a bad package is listed, never hidden
            return {"path": str(package), "name": package.name, "readable": False,
                    "error": {"code": type(exc).__name__, "message": str(exc)}}

    def portable_candidates(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for root in self._portable_roots():
            if not root.is_dir():
                continue
            for package in sorted(root.glob("*.ffportable")):
                rows.append(self._portable_entry(package))
        return rows

    def portable_w154_verdict(self) -> Dict[str, Any]:
        path = ROOT / "results" / "w154_portable_project" / "portable_verdict.json"
        if not path.is_file():
            return {"present": False, "path": str(path)}
        try:
            document = read_json(path)
        except Exception as exc:  # noqa: BLE001 - a broken verdict is reported as such
            return {"present": True, "path": str(path), "readable": False,
                    "error": {"code": type(exc).__name__, "message": str(exc)}}
        totals = dict(((document.get("matrix") or {}).get("totals") or {}))
        return {
            "present": True, "readable": True, "path": str(path),
            "verdict": document.get("verdict"),
            "second_machine_status": document.get("second_machine_status"),
            "format_version": document.get("format_version"),
            "generated_at": document.get("generated_at"),
            "matrix": {key: totals.get(key) for key in
                       ("cases", "checks", "checks_passed", "failed", "not_run", "passed")},
        }

    def _portable_public(self) -> Optional[Dict[str, Any]]:
        document = self._portable
        if not document:
            return None
        project = dict(document.get("project") or {})
        evidence = dict(document.get("evidence") or {})
        assets = dict(document.get("assets") or {})
        return {
            "path": document.get("package"),
            "package_id": (document.get("manifest") or {}).get("package_id"),
            "status": document.get("status"),
            "ok": bool(document.get("ok")),
            "clean": document.get("clean"),
            "project": {key: project.get(key) for key in
                        ("project_id", "project_version", "content_hash",
                         "geometry_digest", "geometry_matches_manifest")},
            "evidence": {key: evidence.get(key) for key in
                         ("present", "binding", "current", "snapshot_id", "record_count")},
            "assets": {"count": assets.get("count"),
                       "by_status": dict(assets.get("by_status") or {}),
                       "distribution_mode": assets.get("distribution_mode")},
            "warning_count": len(document.get("warnings") or []),
            "options": list((document.get("state_machine") or {}).get("options") or []),
            "measured_at": document.get("measured_at"),
        }

    def portable_status(self) -> Dict[str, Any]:
        return {
            "schema": "fractureflow.surface_product.workbench_portable_status.v1",
            "measured_at": _now_iso(),
            "w154": self.portable_w154_verdict(),
            "roots": self.portable_roots(),
            "candidates": self.portable_candidates(),
            "current": copy.deepcopy(self._portable),
            "current_summary": self._portable_public(),
        }

    def _portable_open_path(self, package: Path) -> Dict[str, Any]:
        # Receiver semantics: resolve only against the locations recorded in the
        # package (declared locator + last seen absolute), never silently against
        # this development repository.  Searching the dev tree would turn a
        # receiver-side MISSING into a fake AVAILABLE on the author's machine.
        try:
            document = portablep.open_package(
                package, repo_root=None, source_roots=[],
                hash_limit_bytes=self.PORTABLE_HASH_LIMIT_BYTES)
        except portablep.PortableError as exc:
            raise ContractError(exc.message, exc.code) from exc
        self._portable = document
        return document

    def portable_open(self, package_path: Any) -> Dict[str, Any]:
        # Verification hashes the referenced measurement files and can take a
        # while; it never touches the editing session, so the service lock is
        # not held while it runs.
        package = self._portable_resolve(package_path)
        return self._portable_open_path(package)

    def portable_export(self, *, distribution_mode: str = portablep.REFERENCE_ONLY) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            project = session.project
            current_hash = project.compute_content_hash()
            if session.bundle_path is None or session.saved_hash is None:
                raise ContractError("save the project first; a portable package is built from "
                                    "the persisted bundle", "PORTABLE_SAVE_REQUIRED")
            if session.saved_hash != current_hash:
                raise ContractError("the session has unsaved changes; save before exporting",
                                    "PORTABLE_DIRTY")
            out_root = self.workspace / "portable"
            out_root.mkdir(parents=True, exist_ok=True)
            # Bounded name: the W154 staging directory embeds it and the copied
            # bundle adds ~90 more characters, so a long name breaks MAX_PATH
            # mid-copy.  Identity lives in portable_manifest.json, not the name.
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", project.project_id)[:18] or "project"
            name = "%s_%s_%s.ffportable" % (slug, current_hash[:8], uuid.uuid4().hex[:4])
            package = out_root / name
            try:
                receipt = portablep.export_portable(
                    package, Path(session.bundle_path), distribution_mode=distribution_mode,
                    asset_roots=[ROOT],
                    asset_hash_limit_bytes=self.PORTABLE_EXPORT_HASH_LIMIT_BYTES)
            except portablep.PortableError as exc:
                raise ContractError(exc.message, exc.code) from exc
            opened = self._portable_open_path(package)
            receipt_package = dict(receipt.get("package") or {})
            # The W154 receipt carries assets.by_storage with keys "REFERENCE" /
            # "PACKAGED".  contracts.canonical_json rejects an exact "reference"
            # key anywhere in a response (the project-JSON hidden-reference
            # guard), so the transport copy keeps only the explicit counters.
            raw_assets = dict(receipt.get("assets") or {})
            assets_out = {key: raw_assets.get(key) for key in
                          ("asset_count", "reference_count", "packaged_count",
                           "hash_verified_count", "packaged_bytes")}
            assets_out["by_role"] = dict(raw_assets.get("by_role") or {})
            assets_out["hash_deferred"] = list(raw_assets.get("hash_deferred") or [])
            return {
                "exported": True,
                "package": str(package),
                "package_id": receipt_package.get("package_id"),
                "format_version": receipt.get("format_version"),
                "distribution_mode": receipt.get("distribution_mode"),
                "source_domain": receipt.get("source_domain"),
                "project": dict(receipt.get("project") or {}),
                "assets": assets_out,
                "evidence": {key: (receipt.get("evidence") or {}).get(key) for key in
                             ("binding", "snapshot_id", "record_count", "status_counts")},
                "source_integrity": dict(receipt.get("source_integrity") or {}),
                "verification": dict(receipt.get("verification") or {}),
                "copied_asset_bytes": bool(receipt.get("copied_asset_bytes")),
                "open": opened,
            }

    def portable_relocate(self, package_path: Any, asset_id: Any, new_path: Any) -> Dict[str, Any]:
        with self._lock:
            package = self._portable_resolve(package_path)
            asset = str(asset_id or "").strip()
            if not asset:
                raise ContractError("asset_id is required", "PORTABLE_ASSET_REQUIRED")
            raw = str(new_path or "").strip().strip('"')
            if not raw:
                raise ContractError("new_path is required", "PORTABLE_PATH_REQUIRED")
            source = Path(raw)
            if not source.is_absolute():
                source = ROOT / source
            try:
                relocated = portablep.relocate_asset(package, asset, source)
            except portablep.PortableError as exc:
                raise ContractError(exc.message, exc.code) from exc
            opened = self._portable_open_path(package)
            return {"relocated": relocated, "open": opened}

    def portable_close(self) -> Dict[str, Any]:
        with self._lock:
            self._portable = None
            return {"closed": True, "portable": None}

    def download_path(self, export_id: str, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", export_id) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ContractError("unsafe export download path", "UNSAFE_PATH")
        directory = (self.exports_root / export_id).resolve()
        path = (directory / name).resolve()
        if directory.parent != self.exports_root.resolve() or path.parent != directory or not Path(io_path(path)).is_file():
            raise ContractError("export file is not available", "EXPORT_NOT_FOUND")
        # Hand the HTTP layer a path that stays readable past MAX_PATH.
        return Path(io_path(path))


def verify_export(directory: os.PathLike[str] | str) -> Dict[str, Any]:
    """Independent new-process export reader used by the CLI below."""
    root = Path(directory)
    manifest = read_json(root / "manifest.json")
    checks: List[Dict[str, Any]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "ok": actual == expected})

    for name, expected_sha in manifest["files"].items():
        check("file_sha256:" + name, file_sha256(root / name), expected_sha)
    project_document = read_json(root / "project.json")
    project = ProjectSnapshot.from_dict(project_document)
    relations_document = read_json(root / "relations.json")
    geometry_document = read_json(root / "geometry.json")
    evidence_document = read_json(root / "evidence_ledger.json")
    evidence_verdict = eledger.validate_payload(project, evidence_document)
    with Path(io_path(root / "surfaces.csv")).open("r", encoding="utf-8-sig", newline="") as handle:
        surface_rows = list(csv.DictReader(handle))
    check("project_version", project.project_version, manifest["project_version"])
    check("content_hash", project.to_dict()["content_hash"], manifest["content_hash"])
    check("schema_version", project.schema_version, manifest.get("schema_version"))
    check("readback_is_current_v2", project.schema_version == LATEST_SCHEMA_VERSION, True)
    check("real_quality_not_measured",
          manifest.get("real_quality") in ("undefined", "synthetic_functional_probe_only"), True)
    check("surface_csv_count", len(surface_rows), len(project.surfaces))
    check("geometry_surface_count", len(geometry_document["surfaces"]), len(project.surfaces))
    check("relations_count", len(relations_document["relations"]), len(project.relations))
    for collection in ("observations", "fragments", "surfaces", "relations", "protected_regions"):
        check("manifest_object_count:" + collection, manifest["object_counts"][collection], len(getattr(project, collection)))
    check("unknown_relation_count", relations_document["unknown_relation_count"], manifest["unknown_relation_count"])
    check("evidence_valid", evidence_verdict["valid"], True)
    check("evidence_reasons", evidence_verdict["reasons"], [])
    check("evidence_snapshot_matches_manifest",
          evidence_document["snapshot_id"], manifest["evidence"]["snapshot_id"])
    check("evidence_project_content_hash",
          evidence_document["project_content_hash"], manifest["content_hash"])
    check("evidence_record_count", len(evidence_document["records"]), len(project.relations))
    check("evidence_no_current_basis_mismatch", len(evidence_verdict["basis_mismatched"]), 0)
    check("unit_statuses", geometry_document["unit_statuses"], manifest["unit_statuses"])
    check("coordinate_frames", geometry_document["coordinate_frames"], manifest["coordinate_frames"])
    check("metric_or_geographic_conclusions_emitted", manifest["metric_or_geographic_conclusions_emitted"], False)
    verified = bool(checks) and all(item["ok"] for item in checks)
    return {
        "schema": "fractureflow.surface_product.workbench_export_readback.v1",
        "verified": verified, "checks": checks, "enumeration_count": len(checks),
        "reader_process_id": os.getpid(), "measured_at": _now_iso(),
        "recompute_script": RECOMPUTE_SCRIPT,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-export")
    verify_parser.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "verify-export":
        result = verify_export(args.directory)
        print(canonical_json(result))
        return 0 if result["verified"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
