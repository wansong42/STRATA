# -*- coding: utf-8 -*-
"""G77 candidate generator (DATA_DRIVEN_CANDIDATE_ONLY).

This module is a *candidate front-end* for the P2 chain.  It proposes a finite
disc inventory (HypothesisSet universe) from G66 public payloads only and never
issues a certificate.  The base encoder is imported unmodified from
``fractureflow.p3_model`` and every emitted world answer is passed through
``p3_model.validate_model_output``.

Boundary
--------
* input  : public payloads (G66) only; no truth keys, no hidden ids
* output : candidate_geometry / object_type / instance_grouping /
           relation_candidates / uncertainty only
* queries: every candidate world is executed by ``p2_integration.run_case``
           (Gamma_Q + verifier); this module cannot issue certificates
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

from fractureflow import p3_model as P3

CANDIDATE_SCHEMA = "g77.world_candidate.v1"
CHECKPOINT_SCHEMA = "g77.p3_candidate_checkpoint.v1"
FORBIDDEN_CANDIDATE_KEYS = frozenset({
    "truth", "truth_label", "truth_id", "truth_disc", "truth_normal",
    "truth_centre", "truth_family", "truth_group", "truth_unc",
    "disc", "disc_id", "disc_index", "frac_id", "frac_ids",
    "status", "certificate",
})
RADIUS_FLOOR = 1e-4
CERTIFICATE_TOKEN = "CERTIFIED_"


class CandidateBoundaryError(P3.P3ContractError):
    """Candidate layer tried to cross the certificate boundary."""


class SplitLeakError(RuntimeError):
    """World appears in more than one split."""


class CandidateDataError(SystemExit):
    """Empty denominator / empty training set: fail loudly, never PASS."""


def _unit(v):
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not (math.isfinite(n) and n > 0.0):
        raise CandidateBoundaryError("degenerate normal")
    return a / n


def _basis(n):
    ref = np.asarray([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 = _unit(e1)
    e2 = _unit(np.cross(n, e1))
    return e1, e2


def _kasa(q):
    k = q.shape[0]
    A = np.column_stack([2.0 * q[:, 0], 2.0 * q[:, 1], np.ones(k)])
    b = (q * q).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if not (math.isfinite(r2) and r2 > 0.0):
        return None, None
    return np.asarray([cx, cy]), float(math.sqrt(r2))


def _plane_and_center(hits, normals, radius_prior):
    """Plane through every hit by construction; centre/radius from hits."""
    P = np.asarray(hits, dtype=np.float64)
    k = P.shape[0]
    Nmean = _unit(np.sum([_unit(n) for n in normals], axis=0))
    if k >= 3:
        c0 = P.mean(axis=0)
        U, S, Vt = np.linalg.svd(P - c0)
        n = _unit(Vt[-1])
        plane_resid = float(abs((P - c0) @ n).max())
        if float(S[-1]) > 1e-6 * max(float(S[0]), 1e-12):
            n = _unit(Vt[-1])
    elif k == 2:
        u = _unit(P[1] - P[0])
        m = Nmean - float(Nmean @ u) * u
        if float(np.linalg.norm(m)) < 1e-8:
            m = np.cross(u, np.asarray([0.0, 0.0, 1.0]))
            if float(np.linalg.norm(m)) < 1e-8:
                m = np.cross(u, np.asarray([1.0, 0.0, 0.0]))
        n = _unit(m)
        c0 = P[0].copy()
        plane_resid = float(abs((P - c0) @ n).max())
    else:
        n = Nmean.copy()
        c0 = P[0].copy()
        plane_resid = 0.0
    if float(n @ Nmean) < 0.0:
        n = -n
    e1, e2 = _basis(n)
    q = np.column_stack([(P - c0) @ e1, (P - c0) @ e2])
    if k >= 3:
        centre2d, r_fit = _kasa(q)
        if centre2d is None or not math.isfinite(r_fit) or r_fit > 5.0:
            centre2d = q.mean(axis=0)
            r_fit = 0.0
    elif k == 2:
        centre2d = 0.5 * (q[0] + q[1])
        r_fit = 0.5 * float(np.linalg.norm(q[0] - q[1]))
    else:
        centre2d = q[0].copy()
        r_fit = 0.0
    centre = c0 + centre2d[0] * e1 + centre2d[1] * e2
    maxdist = float(np.max(np.linalg.norm(P - centre, axis=1)))
    radius = max(maxdist, float(r_fit))
    if radius <= 0.0:
        radius = float(radius_prior)
    return {"centre": centre, "normal": n, "e1": e1, "e2": e2,
            "radius": radius, "maxdist": maxdist, "plane_resid": plane_resid}


class CandidateModel(nn.Module):
    """TinyStructureEncoder base + cross-well linker + candidate refiner."""

    def __init__(self, hidden=64, link_hidden=96, n_type=2, n_group=4):
        super().__init__()
        self.base = P3.TinyStructureEncoder(hidden=hidden, n_type=n_type, n_group=n_group)
        self.link_in = 2 * (P3.FEATURE_DIM + 7 + hidden) + 6
        self.linker = nn.Sequential(
            nn.Linear(self.link_in, link_hidden), nn.ReLU(),
            nn.Linear(link_hidden, link_hidden), nn.ReLU(),
            nn.Linear(link_hidden, 1),
        )

    def encode(self, feats8, extra7):
        x = torch.cat([feats8, extra7], dim=-1)
        h = self.base.encoder(feats8)
        return x, h

    def heads(self, feats8, h):
        return {
            "geometry": self.base.head_geometry(h),
            "type_logits": self.base.head_type(h),
            "group_logits": self.base.head_group(h),
            "uncertainty": self.base.head_uncertainty(h).squeeze(-1),
        }

    @staticmethod
    def pair_geom(feats8, well_ids, pi, pj):
        pos = feats8[:, :3]
        nor = feats8[:, 3:6]
        dp = pos[pi] - pos[pj]
        dist = torch.linalg.norm(dp, dim=-1)
        pdi = (dp * nor[pi]).sum(dim=-1)
        pdj = (dp * nor[pj]).sum(dim=-1)
        cos = (nor[pi] * nor[pj]).sum(dim=-1)
        same = (well_ids[pi] == well_ids[pj]).to(feats8.dtype)
        return torch.stack([dist, pdi, pdj, cos, 1.0 - cos, same], dim=-1)

    def relation_logits(self, h, pi, pj):
        left = h[pi]; right = h[pj]
        return self.base.head_relation(torch.cat([left, right], dim=-1)).squeeze(-1)

    def link_logits(self, feats8, extra7, well_ids, pi, pj):
        x, h = self.encode(feats8, extra7)
        fi = torch.cat([x, h], dim=-1)
        pg = self.pair_geom(feats8, well_ids, pi, pj)
        inp = torch.cat([fi[pi], fi[pj], pg], dim=-1)
        return self.linker(inp).squeeze(-1)


class CandidateRefiner(nn.Module):
    """Candidate-level radius margin + in-plane centre offset."""

    def __init__(self, n_in=11, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        out = self.net(x)
        margin = torch.nn.functional.softplus(out[:, 0])
        offset = torch.tanh(out[:, 1:3])
        return margin, offset


def world_observations(world):
    """Flatten a world's public payloads into observation records.

    Every field here is public (payload entries + public well direction).
    Truth labels are kept in separate keys and never enter model features.
    """
    obs = []
    for w in world["wells"]:
        payload = w["payload"]
        feats = P3.payload_features(payload)
        d = payload["measurement"]["d"]
        labels = w.get("obs_truth") or [{}] * len(payload["observations"])
        if len(labels) != len(payload["observations"]):
            raise CandidateBoundaryError("obs_truth length mismatch in %r" % w.get("index"))
        for k, (row, entry) in enumerate(zip(feats, payload["observations"])):
            rec = entry["record"]
            extra = [float(d[0]), float(d[1]), float(d[2]),
                     float(rec["depth"]), float(rec["dip"]),
                     float(rec["dipdir"]), float(rec["terzaghi_angle"])]
            truth = labels[k]
            disc_idx = truth.get("disc")
            disc = ((world.get("discs") or [None] * (int(disc_idx) + 1))[int(disc_idx)]
                    if disc_idx is not None else None)
            obs.append({
                "well_index": int(w["index"]),
                "position": [float(x) for x in entry["position"]],
                "normal": [float(x) for x in entry["normal"]],
                "feat8": [float(x) for x in row],
                "extra7": extra,
                "truth_disc": disc_idx,
                "truth_family": truth.get("family"),
                "truth_group": truth.get("group"),
                "truth_unc": truth.get("unc"),
                "truth_centre": (None if disc is None else [float(x) for x in disc["center"]]),
                "truth_normal": (None if disc is None else [float(x) for x in disc["normal"]]),
            })
    return obs


def world_tensors(world):
    obs = world_observations(world)
    n = len(obs)
    feats8 = torch.tensor([o["feat8"] for o in obs], dtype=torch.float32)
    extra7 = torch.tensor([o["extra7"] for o in obs], dtype=torch.float32)
    well_ids = torch.tensor([o["well_index"] for o in obs], dtype=torch.long)
    disc = torch.tensor([int(o["truth_disc"]) for o in obs], dtype=torch.long)
    family = torch.tensor([int(o["truth_family"]) for o in obs], dtype=torch.long)
    group = torch.tensor([int(o["truth_group"]) for o in obs], dtype=torch.long)
    unc = torch.tensor([float(o["truth_unc"]) for o in obs], dtype=torch.float32)
    pi, pj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pi.append(i); pj.append(j)
    return {"obs": obs, "n": n, "feats8": feats8, "extra7": extra7,
            "well_ids": well_ids, "disc": disc, "family": family,
            "group": group, "unc": unc,
            "pi": torch.tensor(pi, dtype=torch.long),
            "pj": torch.tensor(pj, dtype=torch.long)}


def world_link_matrix(model, tensors, device="cpu"):
    model.eval()
    n = tensors["n"]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    feats8 = tensors["feats8"].to(device)
    extra7 = tensors["extra7"].to(device)
    well_ids = tensors["well_ids"].to(device)
    pi = tensors["pi"].to(device); pj = tensors["pj"].to(device)
    with torch.no_grad():
        logits = model.link_logits(feats8, extra7, well_ids, pi, pj)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    out = np.zeros((n, n), dtype=np.float64)
    for k, (i, j) in enumerate(zip(pi.tolist(), pj.tolist())):
        out[i, j] = out[j, i] = float(probs[k])
    return out


def world_heads(model, tensors, device="cpu"):
    """Per-observation model outputs (no truth fields touched)."""
    model.eval()
    feats8 = tensors["feats8"].to(device)
    h = model.base.encoder(feats8)
    heads = model.heads(feats8, h)
    with torch.no_grad():
        centre = heads["geometry"][:, :3].detach().cpu().numpy()
        normal = heads["geometry"][:, 3:6].detach().cpu().numpy()
        nrm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.clip(nrm, 1e-9, None)
        type_probs = torch.softmax(heads["type_logits"], dim=-1).detach().cpu().numpy()
        group = torch.argmax(heads["group_logits"], dim=-1).detach().cpu().numpy()
        unc = heads["uncertainty"].detach().cpu().numpy()
    return {"centre": centre, "normal": normal, "type_probs": type_probs,
            "group": group, "unc": unc}


def refiner_features(tensors, idx, scores, geo, heads, box_scale):
    P = np.asarray([tensors["obs"][i]["position"] for i in idx], dtype=np.float64)
    d = np.linalg.norm(P - geo["centre"], axis=1)
    spread = float(d.std() / max(box_scale, 1e-12))
    pair_scores = [scores[i, j] for i in idx for j in idx if i < j]
    feats = [
        min(len(idx), 8) / 8.0,
        float(geo["maxdist"] / max(box_scale, 1e-12)),
        float(d.mean() / max(box_scale, 1e-12)),
        spread,
        float(geo["radius"] / max(box_scale, 1e-12)),
        float(np.mean(pair_scores)) if pair_scores else 0.0,
        float(np.min(pair_scores)) if pair_scores else 0.0,
        float(np.mean([0.0 if heads is None else float(heads["unc"][i]) for i in idx])),
        1.0 / max(len(idx), 1),
        float(geo["plane_resid"] / max(box_scale, 1e-12)),
        float(d.max() / max(box_scale, 1e-12)),
    ]
    return feats


def assemble_candidates(tensors, scores, tau_link, radius_prior, refiner=None,
                        box_scale=1.0, heads=None):
    """Union-find over cross-well learned links -> candidate discs."""
    n = tensors["n"]
    wells = tensors["well_ids"].tolist()
    obs = tensors["obs"]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if wells[i] != wells[j] and scores[i, j] > tau_link:
                union(i, j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    keys = sorted(comps, key=lambda r: min(comps[r]))
    cands = []
    for cid, root in enumerate(keys):
        idx = comps[root]
        hits = [obs[i]["position"] for i in idx]
        normals = [obs[i]["normal"] for i in idx]
        geo = _plane_and_center(hits, normals, radius_prior)
        centre = geo["centre"].copy()
        if heads is not None and len(idx) == 1:
            prior = np.asarray(heads["centre"][idx[0]], dtype=np.float64) * float(box_scale)
            prior = prior - float((prior - geo["centre"]) @ geo["normal"]) * geo["normal"]
            centre = prior
        radius = max(float(geo["radius"]), float(np.max(np.linalg.norm(
            np.asarray(hits, dtype=np.float64) - centre, axis=1))) + RADIUS_FLOOR)
        feats = refiner_features(tensors, idx, scores, geo, heads, box_scale)
        pred_margin = 0.0
        if refiner is not None:
            with torch.no_grad():
                rdev = next(refiner.parameters()).device
                xt = torch.tensor([feats], dtype=torch.float32, device=rdev)
                margin, offset = refiner(xt)
            pred_margin = float(margin[0]) * box_scale
            off = offset[0].detach().cpu().numpy() * 0.35 * box_scale
            centre = centre + off[0] * geo["e1"] + off[1] * geo["e2"]
            Pc = np.asarray(hits, dtype=np.float64)
            maxdist_new = float(np.max(np.linalg.norm(Pc - centre, axis=1)))
            radius = max(float(geo["maxdist"]) + pred_margin + RADIUS_FLOOR,
                         maxdist_new + RADIUS_FLOOR)
        radius = min(max(radius, RADIUS_FLOOR), 5.0 * max(box_scale, 1e-12))
        unc = float(np.mean([0.0 if heads is None else float(heads["unc"][i]) for i in idx]))
        tprobs = None if heads is None else np.mean([heads["type_probs"][i] for i in idx], axis=0)
        cands.append({
            "index": cid,
            "candidate_id": "C%03d" % cid,
            "centre": [float(x) for x in centre],
            "normal": [float(x) for x in geo["normal"]],
            "radius": float(radius),
            "n_hits": len(idx),
            "obs_indices": list(idx),
            "geom_centre": [float(x) for x in geo["centre"]],
            "geom_radius": float(geo["radius"]),
            "pred_margin": float(pred_margin),
            "features": [float(x) for x in feats],
            "geo_maxdist": float(geo["maxdist"]),
            "uncertainty": unc,
            "type_probs": ([float(x) for x in tprobs] if tprobs is not None else [0.5, 0.5]),
            "e1": [float(x) for x in geo["e1"]],
            "e2": [float(x) for x in geo["e2"]],
        })
    return cands


def candidate_output(cands, tensors, scores, heads=None):
    """Contract-shaped world answer; validated before it leaves the module."""
    out = {
        "candidate_geometry": [
            {"index": c["index"], "candidate_id": c["candidate_id"],
             "centroid": c["centre"], "normal": c["normal"],
             "radius": c["radius"], "n_hits": c["n_hits"]}
            for c in cands
        ],
        "object_type": [],
        "instance_grouping": [],
        "relation_candidates": [],
        "uncertainty": [],
    }
    mom = {}
    for c in cands:
        for i in c["obs_indices"]:
            mom[i] = c["index"]
    for i, o in enumerate(tensors["obs"]):
        probs = [0.5, 0.5] if heads is None else [float(x) for x in heads["type_probs"][i]]
        unc = 0.0 if heads is None else float(heads["unc"][i])
        out["object_type"].append({"index": i, "probs": probs})
        out["instance_grouping"].append({"index": i, "group": int(mom.get(i, -1))})
        out["uncertainty"].append({"index": i, "value": unc})
    n = tensors["n"]
    for i in range(n):
        for j in range(i + 1, n):
            s = float(scores[i, j])
            if s >= 0.05:
                out["relation_candidates"].append({"i": i, "j": j, "score": round(s, 6)})
    return out


def validate_candidate_output(out):
    P3.validate_model_output(out)

    def walk(obj, path="out"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) in FORBIDDEN_CANDIDATE_KEYS:
                    raise CandidateBoundaryError(
                        "candidate output carries forbidden key %r at %s" % (k, path))
                if CERTIFICATE_TOKEN in str(k).upper():
                    raise CandidateBoundaryError(
                        "candidate output carries certificate token in key %r at %s" % (k, path))
                walk(v, "%s.%s" % (path, k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, "%s[%d]" % (path, i))
        elif isinstance(obj, str) and CERTIFICATE_TOKEN in obj.upper():
            raise CandidateBoundaryError("certificate token in candidate output")

    walk(out)
    return out


def issue_certificate_directly(*_args, **_kwargs):
    raise CandidateBoundaryError(
        "a candidate generator cannot issue certificates; route through Gamma_Q")


def assert_split_disjoint(worlds_by_split):
    seen = {}
    for split, worlds in worlds_by_split.items():
        if not worlds:
            raise CandidateDataError("empty split %s; refusing to proceed" % split)
        for w in worlds:
            wid = str(w["world_id"])
            sd = int(w["seed"])
            if wid in seen:
                raise SplitLeakError("world %s appears in %s and %s"
                                     % (wid, seen[wid], split))
            seen[wid] = split
    seeds = {}
    for split, worlds in worlds_by_split.items():
        for w in worlds:
            sd = int(w["seed"])
            if sd in seeds and seeds[sd] != split:
                raise SplitLeakError("seed %d reused across %s/%s"
                                     % (sd, seeds[sd], split))
            seeds[sd] = split
    return {"worlds": len(seen), "seeds": len(seeds), "leaks": 0}


def assert_training_worlds(worlds):
    if not worlds:
        raise CandidateDataError("empty training set; refuse to fit")
    for w in worlds:
        if str(w.get("domain", "")) != "SYNTHETIC":
            raise CandidateBoundaryError("non-synthetic training world %r" % w.get("world_id"))
    return True


def save_checkpoint(path, model, refiner, meta):
    obj = {"schema": CHECKPOINT_SCHEMA,
           "model": model.state_dict(),
           "refiner": refiner.state_dict() if refiner is not None else None,
           "meta": meta}
    torch.save(obj, str(path))
    with open(path, "rb") as f:
        return {"path": str(path), "bytes": len(f.read()),
                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def load_checkpoint(path, device="cpu"):
    obj = torch.load(str(path), map_location=device, weights_only=True)
    if obj.get("schema") != CHECKPOINT_SCHEMA:
        raise CandidateBoundaryError("unexpected checkpoint schema %r" % obj.get("schema"))
    return obj
