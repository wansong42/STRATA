# -*- coding: utf-8 -*-
"""dataset_registry.py -- machine-readable dataset registry with license boundaries.

Unit G58 (DATA_AUDIT / ZERO_RESEARCH_READING). This module only manipulates
metadata about data assets: identity, license status, split grouping and the
P0 gate question.  It never reads research payloads and produces no research
numeric reading.

Public API
----------
DatasetRegistry.load(path) -> DatasetRegistry
    .rows, .by_site(), .split_groups()
    .open(dataset_id, split=None, purpose="internal_probe") -> RegistryEntry(dict)
    .validate(...) -> list[Violation]
    .p0_gate() -> dict

JSON IO is UTF-8, allow_nan=False, atomic (same-directory temp file + os.replace).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

SCHEMA = "g58.dataset_registry.v1"
SCAN_EVIDENCE_SCHEMA = "g58.scan_evidence.v1"
LICENSE_EVIDENCE_SCHEMA = "g58.license_evidence.v1"
P0_GATE_SCHEMA = "g58.p0_gate.v1"

#: The task order calls the row schema "19 fields" but enumerates 21 names.
#: We implement the full enumerated set (superset of 19) and record the
#: count mismatch as a defect.  Any missing name is a hard violation.
REQUIRED_FIELDS = (
    "dataset_id",
    "source_url",
    "local_path",
    "accessed_at",
    "sha256",
    "license",
    "license_status",
    "redistribution",
    "modality",
    "site_id",
    "lithology",
    "coordinate_frame",
    "resolution",
    "object_types",
    "labels",
    "label_provenance",
    "alignment_status",
    "split_group",
    "allowed_claims",
    "forbidden_claims",
    "raw_or_derived",
)

LICENSE_STATUS_VALUES = ("PUBLIC_VERIFIED", "APPROVED_INTERNAL", "UNVERIFIED", "UNKNOWN")
LABEL_PROVENANCE_VALUES = ("EXACT", "EXPERT_INTERPRETED", "DERIVED_GEOMETRIC", "WEAK", "UNLABELED")
RAW_OR_DERIVED_VALUES = ("raw", "derived")

VERIFIED_LICENSE_STATUSES = ("PUBLIC_VERIFIED", "APPROVED_INTERNAL")
UNVERIFIED_LICENSE_STATUSES = ("UNVERIFIED", "UNKNOWN")
PUBLIC_REPRODUCIBLE_TOKEN = "public_reproducible"

#: Internal / unverified assets must carry these two prohibitions verbatim.
INTERNAL_RESTRICTED_FORBIDDEN = ("不得作为公开基准", "不得外推真实域")

#: Synthetic generation domain is not a field site; it never counts toward P0.
VIRTUAL_SITE_IDS = ("SYNTHETIC_VIRTUAL",)

#: Identity guards frozen for this unit (see own_defects for rationale).
FORGE_INTERPRETED_DATASET_IDS = (
    "forge_gdr1595_16a_fmi_picks",
    "forge_gdr1708_16b_core",
    "forge_gdr1727_16b_fmi_reinterp",
    "forge_gdr1076_58_32_fmi",
)
E_DISK_DERIVED_DATASET_IDS = (
    "e_disk_new_frac",
    "e_disk_new_subsample",
    "e_disk_routeb_result",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Fields allowed to be explicit null (no external record / not retrieved):
#: internal project artefacts have no source_url and unverified assets no access date.
NULLABLE_FIELDS = ("source_url", "accessed_at")

#: Byte thresholds for the declared sampling rule (E2).
SAMPLING_BYTE_THRESHOLD = 200 * 1000 * 1000
MIN_SAMPLE_POINTS = 3


class LicenseBoundaryError(RuntimeError):
    """Raised when a dataset is opened outside what its license status allows."""


class SplitGroupError(ValueError):
    """Raised when a requested split does not match the registered split group."""


class Violation(object):
    """One registry defect.  Structural, not prose."""

    __slots__ = ("code", "dataset_id", "detail")

    def __init__(self, code, dataset_id, detail):
        self.code = str(code)
        self.dataset_id = dataset_id
        self.detail = str(detail)

    def to_dict(self):
        return {"code": self.code, "dataset_id": self.dataset_id, "detail": self.detail}

    def __repr__(self):
        return "Violation(%r, %r, %r)" % (self.code, self.dataset_id, self.detail)

    def __eq__(self, other):
        if not isinstance(other, Violation):
            return NotImplemented
        return (self.code, self.dataset_id, self.detail) == (other.code, other.dataset_id, other.detail)


def canonical_json(obj):
    """Deterministic JSON text used for content digests."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def file_sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def dump_json(path, obj):
    """UTF-8, allow_nan=False, atomic write (same-directory tmp + os.replace)."""
    text = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_format_ok(value):
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def _is_bool(value):
    return isinstance(value, bool)


def _nonempty_list(value):
    return isinstance(value, list) and len(value) > 0


class RegistryEntry(dict):
    """A single registry row (mapping) with attribute access for convenience."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _normalize_row(row):
    if not isinstance(row, dict):
        raise TypeError("registry row must be an object, got %r" % (type(row),))
    return RegistryEntry(row)


class DatasetRegistry(object):
    """Read-only registry view plus the license boundary and P0 gate."""

    def __init__(self, rows, meta=None, scan_evidence=None, license_evidence=None):
        rows = [_normalize_row(r) for r in (rows or [])]
        self.rows = rows
        self.meta = dict(meta or {})
        self.scan_evidence = scan_evidence
        self.license_evidence = license_evidence

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path, scan_evidence=None, license_evidence=None):
        obj = load_json(path)
        if isinstance(obj, list):
            rows, meta = obj, {}
        elif isinstance(obj, dict):
            rows = obj.get("rows", [])
            meta = {k: v for k, v in obj.items() if k != "rows"}
        else:
            raise TypeError("registry file must contain a list or an object with a rows list")
        if scan_evidence is None:
            sibling = os.path.join(os.path.dirname(os.path.abspath(path)), "scan_evidence.json")
            if os.path.exists(sibling):
                scan_evidence = load_json(sibling)
        if license_evidence is None:
            sibling = os.path.join(os.path.dirname(os.path.abspath(path)), "license_evidence.json")
            if os.path.exists(sibling):
                license_evidence = load_json(sibling)
        return cls(rows, meta, scan_evidence, license_evidence)

    # ------------------------------------------------------------- accessors
    def get(self, dataset_id):
        for row in self.rows:
            if row.get("dataset_id") == dataset_id:
                return row
        raise KeyError("dataset_id not registered: %r" % (dataset_id,))

    def by_site(self):
        out = {}
        for row in self.rows:
            out.setdefault(row.get("site_id"), []).append(row)
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    def split_groups(self):
        """Mapping split_group -> sorted site_ids that claim it."""
        out = {}
        for row in self.rows:
            group = row.get("split_group")
            site = row.get("site_id")
            out.setdefault(group, set()).add(site)
        return {g: sorted(s, key=str) for g, s in sorted(out.items(), key=lambda kv: str(kv[0]))}

    def content_payload(self):
        """Registry content that must be byte-stable across runs (E7).

        generated_at_utc and any other build-time stamp live outside this payload.
        """
        return {"schema": self.meta.get("schema", SCHEMA), "unit": self.meta.get("unit"), "rows": [dict(r) for r in self.rows]}

    def content_sha256(self):
        return canonical_sha256(self.content_payload())

    # ------------------------------------------------------- license boundary
    def open(self, dataset_id, split=None, purpose="internal_probe"):
        """Return the registry entry after a license boundary check.

        Rule (task order 4.2): a row whose license_status is UNVERIFIED/UNKNOWN and
        which is not redistributable may only be opened for purpose="internal_probe";
        every other purpose raises LicenseBoundaryError.
        """
        row = self.get(dataset_id)
        if split is not None and split != row.get("split_group"):
            raise SplitGroupError(
                "dataset %r belongs to split_group %r, not %r" % (dataset_id, row.get("split_group"), split)
            )
        status = row.get("license_status")
        if status in UNVERIFIED_LICENSE_STATUSES and row.get("redistribution") is not True:
            if purpose != "internal_probe":
                raise LicenseBoundaryError(
                    "dataset %r has license_status=%s and redistribution=%r; only purpose=internal_probe is allowed, got %r"
                    % (dataset_id, status, row.get("redistribution"), purpose)
                )
        return row

    # --------------------------------------------------------------- validate
    def validate(self, scan_evidence=None, license_evidence=None):
        evidence = self.scan_evidence if scan_evidence is None else scan_evidence
        license_ev = self.license_evidence if license_evidence is None else license_evidence
        return validate_registry_rows(self.rows, scan_evidence=evidence, license_evidence=license_ev)

    # ---------------------------------------------------------------- p0 gate
    def p0_gate(self):
        return evaluate_p0_gate(self.rows)


def validate_registry_rows(rows, scan_evidence=None, license_evidence=None):
    """Pure validator over a row list; returns list[Violation] (empty == clean).

    Raises SystemExit when the registry is empty: an empty sample set must never
    be recorded as a PASS (AGENTS.md "empty set must exit").
    """
    rows = list(rows or [])
    if not rows:
        raise SystemExit("EMPTY_REGISTRY: zero rows; refusing to evaluate (a PASS on an empty set is forbidden)")

    violations = []
    seen_ids = {}

    for idx, row in enumerate(rows):
        rid = row.get("dataset_id") if isinstance(row, dict) else None
        tag = rid if rid else "#%d" % idx
        if not isinstance(row, dict):
            violations.append(Violation("ROW_NOT_OBJECT", tag, "row is %r" % (type(row),)))
            continue

        # E1: every required field present and not None (an empty list is present)
        for field in REQUIRED_FIELDS:
            if field not in row:
                violations.append(Violation("FIELD_MISSING", tag, "required field %r missing" % (field,)))
                continue
            if row[field] is None and field not in NULLABLE_FIELDS:
                violations.append(Violation("FIELD_MISSING", tag, "required field %r missing/null" % (field,)))
                continue
            value = row[field]
            if isinstance(value, str) and not value.strip() and field not in ("source_url", "accessed_at"):
                violations.append(Violation("FIELD_EMPTY", tag, "required field %r is an empty string" % (field,)))

        if rid:
            seen_ids[rid] = seen_ids.get(rid, 0) + 1

        # enums / types
        if row.get("license_status") not in LICENSE_STATUS_VALUES:
            violations.append(Violation("ENUM_LICENSE_STATUS", tag, "license_status=%r not in %r" % (row.get("license_status"), LICENSE_STATUS_VALUES)))
        if row.get("label_provenance") not in LABEL_PROVENANCE_VALUES:
            violations.append(Violation("ENUM_LABEL_PROVENANCE", tag, "label_provenance=%r not in %r" % (row.get("label_provenance"), LABEL_PROVENANCE_VALUES)))
        if row.get("raw_or_derived") not in RAW_OR_DERIVED_VALUES:
            violations.append(Violation("ENUM_RAW_OR_DERIVED", tag, "raw_or_derived=%r not in %r" % (row.get("raw_or_derived"), RAW_OR_DERIVED_VALUES)))
        if not _is_bool(row.get("redistribution")):
            violations.append(Violation("REDISTRIBUTION_NOT_BOOL", tag, "redistribution=%r must be a bool" % (row.get("redistribution"),)))

        # E1-type structural fields
        if not isinstance(row.get("labels"), list):
            violations.append(Violation("LABELS_NOT_LIST", tag, "labels must be a list, got %r" % (type(row.get("labels")),)))
        if not isinstance(row.get("object_types"), list) or not row.get("object_types"):
            violations.append(Violation("OBJECT_TYPES_EMPTY", tag, "object_types must be a non-empty list"))
        for field in ("allowed_claims", "forbidden_claims"):
            if not isinstance(row.get(field), list):
                violations.append(Violation("CLAIMS_NOT_LIST", tag, "%s must be a list" % (field,)))

        # E2: sha256 shape (mismatch against evidence is handled below)
        if not sha256_format_ok(row.get("sha256")):
            violations.append(Violation("SHA_FORMAT", tag, "sha256=%r is not 64 lowercase hex chars" % (row.get("sha256"),)))

        if not isinstance(row.get("split_group"), str) or not row.get("split_group").strip():
            violations.append(Violation("SPLIT_GROUP_EMPTY", tag, "split_group must be a non-empty string"))

        # E4: license honesty
        status = row.get("license_status")
        if status in UNVERIFIED_LICENSE_STATUSES:
            if row.get("redistribution") is True:
                violations.append(Violation("LICENSE_REDISTRIBUTION_UNKNOWN", tag, "license_status=%s may not declare redistribution=True" % (status,)))
            claims_blob = canonical_json(list(row.get("allowed_claims") or []))
            if PUBLIC_REPRODUCIBLE_TOKEN in claims_blob:
                violations.append(Violation("LICENSE_PUBLIC_CLAIM_UNKNOWN", tag, "license_status=%s may not claim %s" % (status, PUBLIC_REPRODUCIBLE_TOKEN)))
        if status == "PUBLIC_VERIFIED":
            if not row.get("source_url") or row.get("license") in (None, "", "UNKNOWN"):
                violations.append(Violation("LICENSE_EVIDENCE_MISSING", tag, "PUBLIC_VERIFIED requires a source_url and a named license"))
            if license_evidence is not None and _license_evidence_entry(license_evidence, rid) is None:
                violations.append(Violation("LICENSE_EVIDENCE_MISSING", tag, "PUBLIC_VERIFIED row has no license_evidence entry"))

        # E5: forbidden claims must be non-empty and bite for restricted assets
        forbidden = row.get("forbidden_claims") or []
        if not _nonempty_list(forbidden):
            violations.append(Violation("FORBIDDEN_CLAIMS_EMPTY", tag, "forbidden_claims must be non-empty"))
        else:
            blob = canonical_json(forbidden)
            restricted = (status in UNVERIFIED_LICENSE_STATUSES) or (row.get("redistribution") is not True)
            if restricted:
                for phrase in INTERNAL_RESTRICTED_FORBIDDEN:
                    if phrase not in blob:
                        violations.append(Violation("FORBIDDEN_CLAIMS_WEAK", tag, "restricted asset missing mandatory prohibition %r" % (phrase,)))

        # identity guards (raw/derived vs label provenance)
        prov = row.get("label_provenance")
        kod = row.get("raw_or_derived")
        if kod == "derived" and prov == "EXACT":
            violations.append(Violation("IDENTITY_PROVENANCE_CONFLICT", tag, "derived data may not carry EXACT labels"))
        if prov == "EXACT" and kod != "raw":
            violations.append(Violation("IDENTITY_PROVENANCE_CONFLICT", tag, "EXACT labels require raw_or_derived='raw'"))
        if rid in FORGE_INTERPRETED_DATASET_IDS:
            if kod != "derived" or prov != "EXPERT_INTERPRETED":
                violations.append(Violation("IDENTITY_FORGE_PICKS", tag, "FORGE interpreted picks must be raw_or_derived='derived' + label_provenance='EXPERT_INTERPRETED'"))
        if rid in E_DISK_DERIVED_DATASET_IDS and kod != "derived":
            violations.append(Violation("IDENTITY_DERIVED_ASSET_RAW", tag, "E-disk derived asset must be raw_or_derived='derived'"))

    # duplicate ids
    for rid, count in sorted(seen_ids.items()):
        if count > 1:
            violations.append(Violation("DUPLICATE_DATASET_ID", rid, "dataset_id appears %d times" % (count,)))

    # E3 split grouping
    violations.extend(_check_split_grouping(rows))

    # E2 sha / sampling check against scan evidence
    if scan_evidence is not None:
        violations.extend(_check_scan_evidence(rows, scan_evidence))

    return violations


def _license_evidence_entry(license_evidence, dataset_id):
    if not isinstance(license_evidence, dict):
        return None
    entries = license_evidence.get("entries")
    if isinstance(entries, dict):
        return entries.get(dataset_id)
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("dataset_id") == dataset_id:
                return entry
    return None


def _check_split_grouping(rows):
    """E3: within-site single split group; split groups never span sites."""
    violations = []
    by_site = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        site = row.get("site_id")
        by_site.setdefault(site, {}).setdefault(row.get("split_group"), []).append(row.get("dataset_id"))

    for site, groups in sorted(by_site.items(), key=lambda kv: str(kv[0])):
        if len(groups) > 1:
            violations.append(
                Violation(
                    "SPLIT_WITHIN_SITE_CONFLICT",
                    str(site),
                    "site has %d split groups: %r (raw/derived of one site must share one split_group)"
                    % (len(groups), {str(g): ids for g, ids in sorted(groups.items(), key=lambda kv: str(kv[0]))}),
                )
            )

    group_sites = {}
    for site, groups in by_site.items():
        for group in groups:
            group_sites.setdefault(group, set()).add(site)
    for group, sites in sorted(group_sites.items(), key=lambda kv: str(kv[0])):
        if len(sites) > 1:
            violations.append(
                Violation("SPLIT_SHARED_BETWEEN_SITES", str(group), "split_group claimed by %d sites: %r" % (len(sites), sorted(sites, key=str)))
            )
    return violations


def _check_scan_evidence(rows, scan_evidence):
    """E2: registered sha256 must equal the recomputed on-disk sha256."""
    violations = []
    files = {}
    for item in (scan_evidence.get("files") or []):
        if isinstance(item, dict) and item.get("path"):
            files[item["path"]] = item
    row_map = {}
    for item in (scan_evidence.get("rows") or []):
        if isinstance(item, dict) and item.get("dataset_id"):
            row_map[item["dataset_id"]] = item

    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = row.get("dataset_id")
        local_path = row.get("local_path")
        evidence = row_map.get(rid)
        if evidence is None:
            violations.append(Violation("SCAN_EVIDENCE_MISSING", rid, "no scan_evidence row for dataset_id"))
            continue
        if evidence.get("exists") is False:
            continue
        rec_sha = evidence.get("sha256")
        if row.get("sha256") != rec_sha:
            violations.append(Violation("SHA_MISMATCH", rid, "registry sha256 %r != recomputed %r" % (row.get("sha256"), rec_sha)))
        file_entry = files.get(local_path)
        if file_entry is not None and file_entry.get("exists"):
            if file_entry.get("sha256") != rec_sha:
                violations.append(Violation("SCAN_INTERNAL_MISMATCH", rid, "scan file sha256 disagrees with scan row sha256"))
            if file_entry.get("sampled") is True:
                points = file_entry.get("sample_points") or []
                if len(points) < MIN_SAMPLE_POINTS:
                    violations.append(
                        Violation("SAMPLING_INSUFFICIENT", rid, "declared sampling has %d sample points (< %d)" % (len(points), MIN_SAMPLE_POINTS))
                    )
    return violations


def evaluate_p0_gate(rows):
    """Mechanical P0 evaluation (task order 4.3).

    Eligible site group (strict reading):
      * every row in the site has license_status in {PUBLIC_VERIFIED, APPROVED_INTERNAL}
      * every row carries an explicit boolean redistribution
      * the site has exactly one split_group
      * at least one structure object type in the site carries labels (labels non-empty)

    Synthetic generation domain (VIRTUAL_SITE_IDS) is not a field site and is excluded.

    The gate verdict is PASS only if >= 2 eligible site groups AND >= 2 of them are
    backed by PUBLIC_VERIFIED licenses (conservative overlay: internal assets must not
    be used to support a public-reproducibility gate).
    """
    rows = list(rows or [])
    by_site = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        by_site.setdefault(row.get("site_id"), []).append(row)

    sites = []
    for site, srows in sorted(by_site.items(), key=lambda kv: str(kv[0])):
        statuses = sorted({str(r.get("license_status")) for r in srows})
        groups = sorted({str(r.get("split_group")) for r in srows})
        license_all_verified = all(r.get("license_status") in VERIFIED_LICENSE_STATUSES for r in srows)
        license_any_verified = any(r.get("license_status") in VERIFIED_LICENSE_STATUSES for r in srows)
        redistribution_explicit = all(isinstance(r.get("redistribution"), bool) for r in srows)
        labeled = sum(1 for r in srows if r.get("labels"))
        excluded = site in VIRTUAL_SITE_IDS
        eligible = bool(
            (not excluded)
            and len(groups) == 1
            and license_all_verified
            and redistribution_explicit
            and labeled > 0
        )
        lenient = bool((not excluded) and len(groups) == 1 and license_any_verified and redistribution_explicit and labeled > 0)
        if excluded:
            basis = "virtual_excluded"
        elif any(r.get("license_status") == "PUBLIC_VERIFIED" for r in srows):
            basis = "public_verified"
        else:
            basis = "internal_approved"
        sites.append(
            {
                "site_id": site,
                "split_group": groups[0] if len(groups) == 1 else None,
                "n_rows": len(srows),
                "license_statuses": statuses,
                "license_all_verified": license_all_verified,
                "license_any_verified": license_any_verified,
                "redistribution_explicit": redistribution_explicit,
                "n_labeled_rows": labeled,
                "virtual": excluded,
                "eligible": eligible,
                "eligible_lenient": lenient,
                "basis": basis,
            }
        )

    eligible_sites = [s["site_id"] for s in sites if s["eligible"]]
    public_eligible = [s["site_id"] for s in sites if s["eligible"] and s["basis"] == "public_verified"]
    lenient_sites = [s["site_id"] for s in sites if s["eligible_lenient"]]

    pass_condition = len(eligible_sites) >= 2 and len(public_eligible) >= 2
    verdict = "P0_GATE_PASS" if pass_condition else "P0_GATE_INSUFFICIENT"
    reason = (
        "eligible_site_groups=%d (public_verified=%d, internal_approved=%d); "
        "split_groups_non_overlapping=%s; virtual_sites_excluded=%r; "
        "PASS requires eligible>=2 and public_verified>=2 -> %s"
        % (
            len(eligible_sites),
            len(public_eligible),
            len(eligible_sites) - len(public_eligible),
            str(_groups_non_overlapping(rows)).lower(),
            list(VIRTUAL_SITE_IDS),
            "PASS" if pass_condition else "INSUFFICIENT",
        )
    )

    return {
        "schema": P0_GATE_SCHEMA,
        "unit": "G58",
        "question": "是否至少有 2 类公开或明确获准的数据，能形成独立场地分割（split_group 互不重叠）？",
        "verdict": verdict,
        "eligible_sites": eligible_sites,
        "n_eligible_sites": len(eligible_sites),
        "public_verified_sites": public_eligible,
        "n_public_verified_sites": len(public_eligible),
        "eligible_sites_lenient_any_verified_row": lenient_sites,
        "criteria": {
            "license_status_in": list(VERIFIED_LICENSE_STATUSES),
            "license_reading": "strict_all_rows_v1",
            "redistribution": "explicit_bool",
            "split_group": "exactly_one_per_site",
            "labels": "at_least_one_labeled_row_in_site",
            "virtual_site_ids_excluded": list(VIRTUAL_SITE_IDS),
            "verdict_rule": "PASS iff n_eligible>=2 and n_public_verified>=2 else INSUFFICIENT",
        },
        "sites": sites,
        "reason": reason,
    }


def _groups_non_overlapping(rows):
    group_sites = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_sites.setdefault(row.get("split_group"), set()).add(row.get("site_id"))
    return all(len(sites) == 1 for sites in group_sites.values())
