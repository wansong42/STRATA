# -*- coding: utf-8 -*-
"""P3 structure-model harness skeleton (G70 readiness only).

Boundary
--------
This module is a *readiness* artefact: no scientific training happens here.
It wraps a tiny deterministic set encoder used only for 1-2 step smokes on
synthetic tensors.  Model input is restricted to the G66 public payload
(``fractureflow.public_payload.v1``).  Model output may contain only

    candidate_geometry / object_type / instance_grouping /
    relation_candidates / uncertainty

and may never emit ``CERTIFIED_*``; every query must still go through the P2
``HypothesisSet -> Gamma_Q -> certificate verifier`` route.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from fractureflow.public_payload import FORBIDDEN_KEYS as _G66_FORBIDDEN_KEYS

PUBLIC_SCHEMA_ID = "fractureflow.public_payload.v1"
PUBLIC_TOP_LEVEL = frozenset({
    "schema", "operator", "operator_version", "n_observed",
    "noise_spec", "measurement", "public_order", "observations",
})
PUBLIC_ENTRY_KEYS = frozenset({"position", "normal", "record"})
PUBLIC_NOISE_KEYS = frozenset({"roughness_deg", "noise_seed", "noise_model"})
FORBIDDEN_KEYS = frozenset(_G66_FORBIDDEN_KEYS)

ALLOWED_OUTPUT_FIELDS = (
    "candidate_geometry",
    "object_type",
    "instance_grouping",
    "relation_candidates",
    "uncertainty",
)
FORBIDDEN_OUTPUT_PREFIX = "CERTIFIED_"
FEATURE_DIM = 8


class P3ContractError(ValueError):
    """Public input / model output contract violation."""


class P3EmptyInputError(SystemExit):
    """Empty observation payload: fail loudly, never predict on nothing."""


class P3OutputBoundaryError(P3ContractError):
    """Model output tried to cross the certificate boundary."""


class P3BudgetError(RuntimeError):
    """Training resource guard rejected the run."""

    def __init__(self, code: str, detail: dict):
        super().__init__(code)
        self.code = code
        self.detail = detail


def _walk(obj, path="payload"):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, "%s[%d]" % (path, i))


def _finite(x, where):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise P3ContractError("non-numeric value at %s: %r" % (where, x))
    if not math.isfinite(v):
        raise P3ContractError("non-finite value at %s: %r" % (where, x))
    return v


def _unit3(x, where):
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise P3ContractError("%s must be a 3-vector" % where)
    v = [_finite(c, "%s[%d]" % (where, i)) for i, c in enumerate(x)]
    n = math.sqrt(sum(c * c for c in v))
    if abs(n - 1.0) > 1e-6:
        raise P3ContractError("non-unit normal at %s: |n|=%.12g" % (where, n))
    return v


def validate_model_input(payload) -> dict:
    """Accept exactly a G66 public payload; reject hidden fields anywhere."""
    if not isinstance(payload, dict):
        raise P3ContractError("payload must be a dict, got %s" % type(payload).__name__)
    for path, obj in _walk(payload):
        if isinstance(obj, dict):
            for key in obj:
                if str(key) in FORBIDDEN_KEYS:
                    raise P3ContractError("forbidden key %r at %s" % (key, path))
    keys = set(payload)
    if keys != PUBLIC_TOP_LEVEL:
        unknown = sorted(keys - PUBLIC_TOP_LEVEL)
        missing = sorted(PUBLIC_TOP_LEVEL - keys)
        raise P3ContractError("top-level mismatch: unknown=%r missing=%r" % (unknown, missing))
    if payload["schema"] != PUBLIC_SCHEMA_ID:
        raise P3ContractError("schema %r != %r" % (payload["schema"], PUBLIC_SCHEMA_ID))
    if not isinstance(payload["noise_spec"], dict) or set(payload["noise_spec"]) != PUBLIC_NOISE_KEYS:
        raise P3ContractError("noise_spec keys != %r" % sorted(PUBLIC_NOISE_KEYS))
    _finite(payload["noise_spec"]["roughness_deg"], "noise_spec.roughness_deg")
    if not isinstance(payload["measurement"], dict):
        raise P3ContractError("measurement must be a dict")
    if not isinstance(payload["public_order"], str):
        raise P3ContractError("public_order must be a string")
    obs = payload["observations"]
    if not isinstance(obs, list) or not obs:
        raise P3EmptyInputError("G70: empty public payload, refusing to predict")
    if int(payload["n_observed"]) != len(obs):
        raise P3ContractError("n_observed %r != len(observations) %d"
                             % (payload["n_observed"], len(obs)))
    for i, entry in enumerate(obs):
        if not isinstance(entry, dict) or set(entry) != PUBLIC_ENTRY_KEYS:
            raise P3ContractError("observation[%d] keys != %r" % (i, sorted(PUBLIC_ENTRY_KEYS)))
        if not isinstance(entry["position"], (list, tuple)) or len(entry["position"]) != 3:
            raise P3ContractError("observation[%d].position must be a 3-vector" % i)
        for j, c in enumerate(entry["position"]):
            _finite(c, "observations[%d].position[%d]" % (i, j))
        _unit3(entry["normal"], "observations[%d].normal" % i)
        if not isinstance(entry["record"], dict) or not entry["record"]:
            raise P3ContractError("observation[%d].record must be a non-empty dict" % i)
        for k, v in entry["record"].items():
            if str(k) in FORBIDDEN_KEYS:
                raise P3ContractError("forbidden record key %r at observation[%d]" % (k, i))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                _finite(v, "observations[%d].record.%s" % (i, k))
    return payload


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _scan_strings(obj, path="output"):
    for p, v in _walk(obj, path):
        if isinstance(v, str) and FORBIDDEN_OUTPUT_PREFIX in v.upper():
            raise P3OutputBoundaryError(
                "model output carries certificate token at %s: %r" % (p, v))


def validate_model_output(out) -> dict:
    """Only candidate structure fields; never a certificate."""
    if not isinstance(out, dict):
        raise P3ContractError("output must be a dict")
    _scan_strings(out)
    unknown = sorted(set(out) - set(ALLOWED_OUTPUT_FIELDS))
    if unknown:
        raise P3OutputBoundaryError("output fields outside contract: %r" % unknown)
    if "candidate_geometry" not in out or "uncertainty" not in out:
        raise P3ContractError("output must carry candidate_geometry and uncertainty")
    return out


def output_sha256(out: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(validate_model_output(out))).hexdigest()


def set_deterministic(seed: int = 0) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    supported = hasattr(torch, "use_deterministic_algorithms")
    if supported:
        torch.use_deterministic_algorithms(True)
    return {
        "seed": int(seed),
        "use_deterministic_algorithms_available": bool(supported),
        "use_deterministic_algorithms_enabled": bool(
            supported and torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def tensor_state_sha(tensors) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        if t is None:
            h.update(("%s:NONE\n" % name).encode("utf-8"))
            continue
        arr = t.detach().to("cpu").contiguous().numpy()
        h.update(("%s|%s|%r\n" % (name, arr.dtype.str, arr.shape)).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class ResourceBudget:
    """Training-run counter: 4 GPU-h/run, 3 variants, 12 GPU-h total."""

    def __init__(self, max_run_gpu_h=4.0, max_variants=3, max_total_gpu_h=12.0):
        self.max_run_gpu_h = float(max_run_gpu_h)
        self.max_variants = int(max_variants)
        self.max_total_gpu_h = float(max_total_gpu_h)
        self.runs = []

    def state(self) -> dict:
        variants = sorted({r["variant_id"] for r in self.runs})
        total = round(sum(r["gpu_hours"] for r in self.runs), 6)
        return {"n_runs": len(self.runs), "variants": variants,
                "n_variants": len(variants), "total_gpu_hours": total,
                "limits": {"per_run_gpu_h": self.max_run_gpu_h,
                           "max_variants": self.max_variants,
                           "max_total_gpu_h": self.max_total_gpu_h}}

    def register_run(self, variant_id: str, gpu_hours: float) -> dict:
        try:
            hours = float(gpu_hours)
        except (TypeError, ValueError):
            raise P3BudgetError("GPU_HOURS_NOT_NUMERIC", {"value": repr(gpu_hours)})
        if not math.isfinite(hours) or hours < 0.0:
            raise P3BudgetError("GPU_HOURS_INVALID", {"value": repr(gpu_hours)})
        if hours > self.max_run_gpu_h:
            raise P3BudgetError("PER_RUN_LIMIT_EXCEEDED",
                                {"requested": hours, "limit": self.max_run_gpu_h})
        variants = {r["variant_id"] for r in self.runs}
        if variant_id not in variants and len(variants) >= self.max_variants:
            raise P3BudgetError("VARIANT_LIMIT_EXCEEDED",
                                {"variant_id": variant_id, "limit": self.max_variants})
        total = sum(r["gpu_hours"] for r in self.runs)
        if total + hours > self.max_total_gpu_h + 1e-12:
            raise P3BudgetError("TOTAL_BUDGET_EXCEEDED",
                                {"requested": hours, "used": total,
                                 "limit": self.max_total_gpu_h})
        self.runs.append({"variant_id": str(variant_id), "gpu_hours": hours})
        return self.state()


def _record_scalars(record: dict):
    vals = [float(v) for v in record.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals) / 10.0


def payload_features(payload: dict):
    validate_model_input(payload)
    rows = []
    for entry in payload["observations"]:
        rmean, rcount = _record_scalars(entry["record"])
        rows.append(list(map(float, entry["position"]))
                    + list(map(float, entry["normal"])) + [rmean, rcount])
    return rows


def payloads_to_tensor(payloads):
    rows = [payload_features(p) for p in payloads]
    n = {len(r) for r in rows}
    if len(n) != 1:
        raise P3ContractError("batch payloads must share n_observed: %r" % sorted(n))
    feats = torch.tensor(rows, dtype=torch.float32)
    mask = torch.ones(feats.shape[:2], dtype=torch.float32)
    return feats, mask


class TinyStructureEncoder(nn.Module):
    """Deterministic smoke encoder; not a scientific model."""

    def __init__(self, hidden=16, n_type=3, n_group=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_geometry = nn.Linear(hidden, 6)
        self.head_type = nn.Linear(hidden, n_type)
        self.head_group = nn.Linear(hidden, n_group)
        self.head_uncertainty = nn.Linear(hidden, 1)
        self.head_relation = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                           nn.Linear(hidden, 1))

    def forward(self, feats, mask):
        h = self.encoder(feats)
        b, n, _ = h.shape
        geometry = self.head_geometry(h)
        type_logits = self.head_type(h)
        group_logits = self.head_group(h)
        uncertainty = self.head_uncertainty(h).squeeze(-1)
        left = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
        right = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        relation = self.head_relation(torch.cat([left, right], dim=-1)).squeeze(-1)
        return {"geometry": geometry, "type_logits": type_logits,
                "group_logits": group_logits, "uncertainty": uncertainty,
                "relation": relation, "mask": mask}


class P3Harness:
    """fit / predict / checkpoint skeleton bound to the public payload."""

    def __init__(self, device="cuda", seed=0, hidden=16):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.determinism = set_deterministic(seed)
        self.model = TinyStructureEncoder(hidden=hidden).to(self.device)

    def fit(self, payloads, steps=1, lr=1e-3):
        feats, mask = payloads_to_tensor(payloads)
        feats, mask = feats.to(self.device), mask.to(self.device)
        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        losses = []
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            out = self.model(feats, mask)
            target = feats[:, :, :6]
            shape = out["geometry"].pow(2).mean()
            loss = (out["geometry"] - target).pow(2).mean()
            loss = loss + 1e-3 * (shape + out["type_logits"].pow(2).mean()
                                  + out["group_logits"].pow(2).mean()
                                  + out["relation"].pow(2).mean()
                                  + out["uncertainty"].pow(2).mean())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return {"steps": int(steps), "losses": losses}

    def predict(self, payload: dict) -> dict:
        feats, mask = payloads_to_tensor([payload])
        feats, mask = feats.to(self.device), mask.to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model(feats, mask)
            n = feats.shape[1]
            geometry = out["geometry"][0].detach().cpu().tolist()
            type_probs = torch.softmax(out["type_logits"][0], dim=-1).detach().cpu().tolist()
            groups = torch.argmax(out["group_logits"][0], dim=-1).detach().cpu().tolist()
            uncertainty = out["uncertainty"][0].detach().cpu().tolist()
            relation = torch.sigmoid(out["relation"][0]).detach().cpu().tolist()
        if was_training:
            self.model.train()
        result = {
            "candidate_geometry": [
                {"index": i,
                 "centroid": [round(float(v), 6) for v in geometry[i][:3]],
                 "normal": [round(float(v), 6) for v in geometry[i][3:]]}
                for i in range(n)],
            "object_type": [
                {"index": i, "probs": [round(float(p), 6) for p in type_probs[i]]}
                for i in range(n)],
            "instance_grouping": [{"index": i, "group": int(groups[i])} for i in range(n)],
            "relation_candidates": [
                {"i": i, "j": j, "score": round(float(relation[i][j]), 6)}
                for i in range(n) for j in range(i + 1, n)],
            "uncertainty": [{"index": i, "value": round(float(uncertainty[i]), 6)}
                            for i in range(n)],
        }
        return validate_model_output(result)

    def state_shas(self) -> dict:
        params = {n: p for n, p in self.model.named_parameters()}
        grads = {n: p.grad for n, p in self.model.named_parameters()}
        return {"params_sha256": tensor_state_sha(params),
                "grads_sha256": tensor_state_sha(grads)}

    def save_checkpoint(self, path) -> dict:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "schema": "g70.p3_checkpoint.v1",
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "feature_dim": FEATURE_DIM,
            "determinism": self.determinism,
            "readiness_only": True,
            "scientific_training": False,
        }
        torch.save(payload, path)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return {"path": str(path), "sha256": digest,
                "bytes": os.path.getsize(path)}

    def load_checkpoint(self, path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("schema") != "g70.p3_checkpoint.v1":
            raise P3ContractError("unexpected checkpoint schema: %r" % payload.get("schema"))
        self.model.load_state_dict(payload["state_dict"])
        return {"schema": payload["schema"], "readiness_only": payload.get("readiness_only"),
                "scientific_training": payload.get("scientific_training")}
