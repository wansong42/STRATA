# -*- coding: utf-8 -*-
"""obs_equiv_data_registry.py -- M0 machine-readable data/licence/truth registry.

Unit: M0 (observation-equivalence data registry audit, 2026-10-24).
Watermark: M0_DATA_AUDIT / ZERO_ADAPTER / ZERO_REAL_HOLDOUT

This module only registers metadata about data assets.  It does not download
files, does not implement adapters, does not run REAL_CENSORING_HOLDOUT and
does not produce research readings.

Two hard rules
--------------
1. A row whose licence or truth level is not verified carries ``UNKNOWN``.
   Directory accessibility is never upgraded to "file verified", "labels
   verified" or "licence verified".
2. ``public_or_private`` means **public-shareability under the recorded
   licence**, not mere network accessibility.  A public GitHub repository with
   no licence (for example the Aland trace repo) is registered as ``PRIVATE``
   with ``license_status=NO_LICENSE_FOUND``.

Truth levels follow DEEP_RESEARCH_complex_rock_discontinuity_counterfactual_
20261024.md section 5.1:
EXACT_SYNTHETIC / LAB_VOLUME_REFERENCE / EXPERT_SURFACE_INTERPRETATION /
EXPERT_BOREHOLE_PICKS / HELD_OUT_OBSERVATION, plus UNKNOWN for this audit.
No row is silently promoted: a derived label (set_ids, fracture_id produced by
a project adapter) is never registered as truth.

The catalogue is intentionally explicit.  ``build_registry`` probes local
paths (existence, size, sha256 for files below a limit, top-level counts for
directories) and fills ``verified_at``/``verified_scope`` accordingly.  A
probe proves only what it names; it never proves payload labels or a licence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "obs_equiv.data_registry.v1"
SCHEMA_SUMMARY = "obs_equiv.data_registry.summary.v1"
SCHEMA_VERDICT = "obs_equiv.data_registry.verdict.v1"
SCHEMA_COVERAGE = "obs_equiv.data_registry.coverage.v1"
WATERMARK = "M0_DATA_AUDIT / ZERO_ADAPTER / ZERO_REAL_HOLDOUT"

UNKNOWN = "UNKNOWN"

TRUTH_LEVELS = (
    "EXACT_SYNTHETIC",
    "LAB_VOLUME_REFERENCE",
    "EXPERT_SURFACE_INTERPRETATION",
    "EXPERT_BOREHOLE_PICKS",
    "HELD_OUT_OBSERVATION",
    UNKNOWN,
)

TRUTH_STATUSES = (
    "PROJECT_RECORDED",
    "DOCUMENTED_IN_TECH_BOOK",
    "INHERITED_FROM_SOURCE",
    "PARTIAL",
    "VOIDED",
    UNKNOWN,
)

LICENSE_STATUSES = (
    "VERIFIED_PRIMARY",
    "VERIFIED_INHERITED",
    "PROJECT_RECORDED",
    "PROJECT_INTERNAL",
    "RESTRICTED_NONCOMMERCIAL",
    "RESTRICTED_INTERNAL",
    "NO_LICENSE_FOUND",
    UNKNOWN,
)

#: Statuses that must be represented by license_id == UNKNOWN.  A row may not
#: wash an unknown/restricted licence into a named permissive licence.
LICENSE_UNKNOWN_STATUSES = frozenset({"UNKNOWN", "NO_LICENSE_FOUND"})
LICENSE_NONPERMISSIVE_STATUSES = frozenset({"RESTRICTED_NONCOMMERCIAL", "RESTRICTED_INTERNAL"})
LICENSE_NO_NAME_STATUSES = LICENSE_UNKNOWN_STATUSES | LICENSE_NONPERMISSIVE_STATUSES

PUBLIC_PRIVATE = ("PUBLIC", "PRIVATE", UNKNOWN)

INDEPENDENT_UNITS = ("整井", "整站点", "整岩芯", "无")

MODALITIES = (
    "synthetic_network",
    "synthetic_image_mask",
    "network_tensor",
    "dfn_table",
    "dfn_geometry",
    "borehole_fmi",
    "borehole_log",
    "borehole_image",
    "core_analysis",
    "core_ct",
    "point_cloud",
    "outcrop_trace_map",
    "outcrop_image_patch",
    "image_mask",
    "orthomosaic",
    "microseismic_catalog",
    "well_metadata",
    "literature_table",
    "diagnostic_archive",
    "unknown",
)

DEDUPE_PREFIXES = (
    "DERIVED_FROM",
    "DERIVED_COPY_OF",
    "SUBSET_OF",
    "DUPLICATE_SOURCE",
)
DEDUPE_PLAIN = (
    "UNIQUE",
    "UNKNOWN",
    "DUPLICATE_COPIES_PRESENT",
    "PARTIAL_DOWNLOAD",
)

CLAIM_ALLOWED_PREFIX = "ALLOWED:"
CLAIM_FORBIDDEN_PREFIX = "FORBIDDEN:"
CLAIM_FORBIDDEN_WORDS = ("全部可用", "all available", "all assets verified")
ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

REQUIRED_FIELDS = (
    "asset_id",
    "path",
    "path_extra",
    "source",
    "modality",
    "license_id",
    "license_status",
    "truth_level",
    "truth_status",
    "site_id",
    "independent_unit",
    "independent_unit_count",
    "public_or_private",
    "dedupe_status",
    "claim_boundary",
    "verified_at",
    "verified_scope",
    "evidence_refs",
    "notes",
)

NULLABLE_FIELDS = ("path",)

# ---------------------------------------------------------------------------
# Evidence references (project-relative paths unless marked as a section)
# ---------------------------------------------------------------------------
EV_DATA_MAP = "DATA_MAP.md"
EV_EFRACTURE = "data/external/EFRACTURE_DATA.md"
EV_BOOK = "DEEP_RESEARCH_complex_rock_discontinuity_counterfactual_20261024.md#5.1-5.2-5.10"
EV_COMPLIANCE = "docs/开源合规_第三方许可与数据出处清单_20260829.md"
EV_G58_REG = "results/g58_dataset_registry_20261021/dataset_registry.json"
EV_G58_LIC = "results/g58_dataset_registry_20261021/license_evidence.json"
EV_G58_SCAN = "results/g58_dataset_registry_20261021/scan_evidence.json"
EV_R149_REPORT = "results/r149_data_beachhead/verification_report.md"
EV_R149_M4 = "results/r149_data_beachhead/M4_三档小结.md"
EV_R159_INV = "results/r159_g1_unlock/inventory.json"
EV_R159_API = "results/r159_g1_unlock/zen18078781_api_record.json"
EV_R176_LIC = "results/r176_geocrack_ingest/license_and_mirror_check.md"
EV_R181_INV = "results/r181_geocrack_body/inventory_full.json"
EV_ALAND_README = "data/external/aland_islands_repo/README.rst"
EV_TOXIC = "data/external/utah_forge_fmi/TOXICITY_NOTE_FORGE_58_32_routeA_mirror_normal_20261004.json"
EV_DEEPCRACK_README = "data/external/efracture_data/downloads/04_Crack_Datasets/DeepCrack-master/README.md"
EV_ARCHIVE_AGENTS = "archive/AGENTS_full_20261016.md"

# ---------------------------------------------------------------------------
# Claim boundaries.  Every row must carry at least one FORBIDDEN line.
# ---------------------------------------------------------------------------
CB_SYNTHETIC = [
    "ALLOWED: 合成声明域内的全结构、查询与参考后验评价",
    "ALLOWED: 机制留出与基线判别力测试",
    "FORBIDDEN: 真实泛化声明",
    "FORBIDDEN: 真实完整三维查询准确率",
]
CB_FMI = [
    "ALLOWED: 井壁深度-产状-类型监督与跨井留出的观测一致性评价",
    "ALLOWED: 作为 EXPERT_BOREHOLE_PICKS 层级的解释拾取使用",
    "FORBIDDEN: 真实完整三维后验或内部贯通真值",
    "FORBIDDEN: 把解释叠加图作为无泄漏输入",
]
CB_SURFACE = [
    "ALLOWED: 二维/表面迹线、表面几何与遮挡压力测试",
    "ALLOWED: 观测算子与跨站点划分测试",
    "FORBIDDEN: 升级为三维内部网络真值",
    "FORBIDDEN: 真实泛化声明",
]
CB_SURFACE_DERIVED = CB_SURFACE + [
    "FORBIDDEN: 把 adapter 派生的 fracture_id/set_ids 当作独立结构真值",
]
CB_DFN = [
    "ALLOWED: 产状表/组系交叉对照与受控 DFN 几何基准",
    "ALLOWED: 方向统计与标注组中心对照",
    "FORBIDDEN: 真实完整三维查询准确率",
    "FORBIDDEN: 天然真实泛化声明",
]
CB_RAW = [
    "ALLOWED: 内部盘点、格式/坐标/覆盖范围检查",
    "FORBIDDEN: 作为真值或公开主分母",
    "FORBIDDEN: 真实泛化声明",
]
CB_DERIVED_LABEL = [
    "ALLOWED: 派生标签的方法对照与内部诊断",
    "FORBIDDEN: 把派生 set_ids/fracture_id 当作结构真值",
    "FORBIDDEN: 让泄漏性标签进入主评价或对外口径",
]
CB_INTERNAL = [
    "ALLOWED: 内部盘点与谱系记录",
    "FORBIDDEN: 公开分发或作为公开主分母",
    "FORBIDDEN: 真实泛化声明",
]
CB_CANDIDATE = [
    "ALLOWED: 候选清单、来源与许可追踪",
    "FORBIDDEN: 声称文件/标签/许可已核实",
    "FORBIDDEN: 作为主分母或真实泛化证据",
]
CB_TOXIC = [
    "ALLOWED: 谱系留档与因果对照",
    "FORBIDDEN: 组系统计推断（mode_dip/mode_dip_dir/组数）",
    "FORBIDDEN: 任何对外精度或增益承诺",
    "FORBIDDEN: 当作发表法向的等价替代",
]
CB_PARTIAL = [
    "ALLOWED: 已到盘子集的 2D 图像/标签盘点",
    "ALLOWED: 观测算子与分割辅助任务的方法对照",
    "FORBIDDEN: 把部分下载写成完整数据集",
    "FORBIDDEN: 三维内部结构真值或真实泛化声明",
]
CB_RESTRICTED = [
    "ALLOWED: 非商业研究/教学用途（按上游 README）",
    "FORBIDDEN: 再分发或商业使用",
    "FORBIDDEN: 作为公开主分母",
]


def _asset(
    asset_id: str,
    *,
    path: str | None = None,
    path_extra: Sequence[str] = (),
    source: str,
    modality: str,
    license_id: str = UNKNOWN,
    license_status: str = UNKNOWN,
    truth_level: str = UNKNOWN,
    truth_status: str = UNKNOWN,
    site_id: str = UNKNOWN,
    independent_unit: str = "无",
    independent_unit_count: Any = UNKNOWN,
    public_or_private: str = UNKNOWN,
    dedupe_status: str = "UNIQUE",
    claim_boundary: Sequence[str],
    verified_at: str = UNKNOWN,
    verified_scope: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    notes: str = "",
) -> dict:
    """Build one catalogue row.  Every field stays explicit in the output."""
    if not verified_scope:
        verified_scope = ("not_verified",)
    return {
        "asset_id": asset_id,
        "path": path,
        "path_extra": list(path_extra),
        "source": source,
        "modality": modality,
        "license_id": license_id,
        "license_status": license_status,
        "truth_level": truth_level,
        "truth_status": truth_status,
        "site_id": site_id,
        "independent_unit": independent_unit,
        "independent_unit_count": independent_unit_count,
        "public_or_private": public_or_private,
        "dedupe_status": dedupe_status,
        "claim_boundary": list(claim_boundary),
        "verified_at": verified_at,
        "verified_scope": list(verified_scope),
        "evidence_refs": list(evidence_refs),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Catalogue.  Rows are metadata-only; paths are probed by build_registry().
# ---------------------------------------------------------------------------
CATALOGUE: tuple[dict, ...] = (
    # -- synthetic / project-generated ------------------------------------
    _asset(
        "synth_10000",
        path="data/synth/synth_10000.pt",
        source="project-generated: fractureflow synthetic generator (synth_engine)",
        modality="synthetic_network",
        license_id="PROJECT_INTERNAL",
        license_status="PROJECT_INTERNAL",
        truth_level="EXACT_SYNTHETIC",
        truth_status="PROJECT_RECORDED",
        site_id="SYNTHETIC_VIRTUAL",
        independent_unit="无",
        independent_unit_count=10000,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SYNTHETIC,
        verified_scope=("truth_project_record", "license_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_COMPLIANCE),
        notes=(
            "DATA_MAP records 10,000 synthetic networks with exact set_ids/set_dirs/obs_mask; "
            "not re-opened in M0. R145 records no third-party rights. Synthetic declared domain only."
        ),
    ),
    _asset(
        "synth_val",
        path="data/synth/synth_val.pt",
        source="project-generated: fractureflow synthetic generator (synth_engine)",
        modality="synthetic_network",
        license_id="PROJECT_INTERNAL",
        license_status="PROJECT_INTERNAL",
        truth_level="EXACT_SYNTHETIC",
        truth_status="PROJECT_RECORDED",
        site_id="SYNTHETIC_VIRTUAL",
        independent_unit="无",
        independent_unit_count=2000,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SYNTHETIC,
        verified_scope=("truth_project_record", "license_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_COMPLIANCE),
        notes="DATA_MAP records 2,000 synthetic validation networks; not re-opened in M0.",
    ),
    _asset(
        "vision_synthetic",
        path="data/vision/synthetic",
        path_extra=(
            "data/vision/train_meta.json",
            "data/vision/val_meta.json",
            "data/vision/test_meta.json",
        ),
        source="project-generated synthetic image/mask pairs",
        modality="synthetic_image_mask",
        license_id="PROJECT_INTERNAL",
        license_status="PROJECT_INTERNAL",
        truth_level="EXACT_SYNTHETIC",
        truth_status="PROJECT_RECORDED",
        site_id="SYNTHETIC_VIRTUAL",
        independent_unit="无",
        independent_unit_count=1500,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SYNTHETIC,
        verified_scope=("truth_project_record", "license_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_COMPLIANCE),
        notes=(
            "1500 image/mask pairs (1200 train / 150 val / 150 test) per meta JSON; "
            "2D synthetic segmentation only. Site names in meta are texture labels, not independent field sites."
        ),
    ),
    _asset(
        "vision_deepcrack",
        path="data/external/efracture_data/downloads/04_Crack_Datasets/DeepCrack-master/dataset",
        path_extra=("data/vision/deepcrack",),
        source="https://github.com/yhlleo/DeepCrack",
        modality="image_mask",
        license_id=UNKNOWN,
        license_status="RESTRICTED_NONCOMMERCIAL",
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="DEEPCRACK",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RESTRICTED,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_DEEPCRACK_README),
        notes=(
            "300 train + 237 test meta entries; 1,076 files. README restricts use to non-commercial "
            "research/education and claims copyright on part of the images/masks; no SPDX licence file. "
            "M0 records licence/truth as UNKNOWN, not as a permissive licence."
        ),
    ),
    _asset(
        "r60_wells_csv",
        path="data/real/r60_wells_csv",
        source="project-generated known-solution test fixture (tests/test_v_r60_joint_rule.py::_three_wells)",
        modality="synthetic_network",
        license_id="PROJECT_INTERNAL",
        license_status="PROJECT_INTERNAL",
        truth_level="EXACT_SYNTHETIC",
        truth_status="PROJECT_RECORDED",
        site_id="SYNTHETIC_VIRTUAL",
        independent_unit="无",
        independent_unit_count=3,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SYNTHETIC,
        verified_scope=("truth_project_record", "license_project_record"),
        evidence_refs=(EV_COMPLIANCE,),
        notes="Three CSV wells; R145 records project-generated synthetic known solution, no third-party rights.",
    ),
    # -- project real corpora / derived nets ------------------------------
    _asset(
        "real_nets_loaded",
        path="data/real/loaded_real_nets.pt",
        source="project-compiled multi-site corpus; per-site upstream sources not individually verified",
        modality="network_tensor",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="MULTI_SITE",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_INTERNAL,
        verified_scope=("license_not_verified", "truth_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "391 networks / 8 sites recorded in DATA_MAP (beishan 22, coso 60, egs_collab 24, fallon 44, "
            "fmi_rev 31, forge 60, forge_16b 90, mining 60). Unique well count not verified; the file is a "
            "composite, not one independent unit. R145 forbids distribution because it contains Beishan data "
            "and mixed, not individually verified, site licences."
        ),
    ),
    _asset(
        "real_nets_loaded_setid",
        path="data/real/loaded_real_nets_setid.pt",
        source="derived from real_nets_loaded by offline spherical k-means set labelling",
        modality="network_tensor",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="MULTI_SITE",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:real_nets_loaded",
        claim_boundary=CB_DERIVED_LABEL,
        verified_scope=("license_not_verified", "truth_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "set_ids 0-6 are derived geometric labels. DATA_MAP records that full-normal clustering lets "
            "hidden normals influence labels (leakage-prone); they are not structural truth."
        ),
    ),
    _asset(
        "real_nets_loaded_setid_opt",
        path="data/real/loaded_real_nets_setid_opt.pt",
        source="derived from real_nets_loaded by vMF-BIC set labelling",
        modality="network_tensor",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="MULTI_SITE",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:real_nets_loaded",
        claim_boundary=CB_DERIVED_LABEL,
        verified_scope=("license_not_verified", "truth_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "Optimised vMF-BIC set_ids (src/fractureflow/setlabel_opt.py) are derived labels, not truth. "
            "R145 forbids distribution because the parent corpus contains Beishan data."
        ),
    ),
    _asset(
        "beishan_wells",
        path="data/real/beishan_wells.npz",
        source="Beishan pre-selection area; project-compiled; source publication not verified",
        modality="network_tensor",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="BEISHAN",
        independent_unit="整井",
        independent_unit_count=22,
        public_or_private="PRIVATE",
        dedupe_status="UNKNOWN",
        claim_boundary=CB_INTERNAL,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_DATA_MAP, EV_COMPLIANCE, EV_ARCHIVE_AGENTS),
        notes=(
            "DATA_MAP records 22 wells x 40 normals = 880 rows. Historical audit records report sliding-window "
            "overlap and degenerate structure (not an independent per-fracture catalogue). High-level radioactive "
            "waste pre-selection sensitivity; R145 forbids distribution."
        ),
    ),
    _asset(
        "forge16a_net",
        path="data/real/forge16A_net.pt",
        source="derived from GDR submission 1595 (16A FMI picks) into the project net protocol",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="INHERITED_FROM_SOURCE",
        site_id="FORGE_MILFORD_16A",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:forge_16a_fmi_log",
        claim_boundary=CB_FMI,
        verified_scope=("license_inherited_from_primary", "truth_inherited_from_source"),
        evidence_refs=(EV_G58_LIC, EV_COMPLIANCE),
        notes=(
            "Derived net from public CC-BY-4.0 FMI picks; attribution required. R145 says distributable with "
            "attribution, while G58 records PROJECT_INTERNAL/redistribution=false; the conflict is recorded, "
            "not resolved in M0."
        ),
    ),
    _asset(
        "forge_fmi_2wells",
        path="data/external/utah_forge_fmi/forge_fmi_2wells.pt",
        path_extra=("data/external/utah_forge_fmi/forge_fmi_2wells_summary.json",),
        source="derived from GDR 1727 (16B) + GDR 1076 (58-32) FMI picks",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="INHERITED_FROM_SOURCE",
        site_id="FORGE_MILFORD_16B_58_32",
        independent_unit="整井",
        independent_unit_count=2,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:forge_16b_fmi_las+forge_58_32_fmi_las",
        claim_boundary=CB_FMI,
        verified_scope=("license_inherited_from_primary", "truth_inherited_from_source"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG),
        notes=(
            "Summary JSON records 2,087 picks (16B) + 2,241 picks (58-32), including induced/natural type counts. "
            "Two wells are two units, not one; attribution required. G58/R145 redistribution conflict recorded."
        ),
    ),
    _asset(
        "forge2024_single",
        path="data/real/forge2024_single.pt",
        source="derived from GDR submission 1222 (FORGE DFN 2024); exact .pt mapping not re-verified",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:gdr1222_dfn",
        claim_boundary=CB_DFN,
        verified_scope=("license_inherited_from_primary", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "Single-well derived network; 94 MB. The GDR source is public CC-BY-4.0, but the derived network is "
            "not a complete 3D truth. R145 suggests link-only due to size; G58 marks internal. Mapping to GDR 1222 "
            "is by project naming and is not re-verified in M0."
        ),
    ),
    _asset(
        "forge2024_multi",
        path="data/real/forge2024_multi.pt",
        source="derived from GDR submission 1222 (FORGE DFN 2024); exact .pt mapping not re-verified",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:gdr1222_dfn",
        claim_boundary=CB_DFN,
        verified_scope=("license_inherited_from_primary", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes="Multi-well derived network; source licence CC-BY-4.0; derived truth tier remains UNKNOWN.",
    ),
    _asset(
        "forge2024meq_single",
        path="data/real/forge2024meq_single.pt",
        source="derived from GDR submission 1646 (MEQ DFN v2); exact .pt mapping not re-verified",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:gdr1646_meq_dfn",
        claim_boundary=CB_DFN,
        verified_scope=("license_inherited_from_primary", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes="Single-well MEQ-derived network; source licence CC-BY-4.0; derived truth tier UNKNOWN.",
    ),
    _asset(
        "forge2024meq_multi",
        path="data/real/forge2024meq_multi.pt",
        source="derived from GDR submission 1646 (MEQ DFN v2); exact .pt mapping not re-verified",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:gdr1646_meq_dfn",
        claim_boundary=CB_DFN,
        verified_scope=("license_inherited_from_primary", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes="Multi-well MEQ-derived network; source licence CC-BY-4.0; derived truth tier UNKNOWN.",
    ),
    _asset(
        "pontrelli_single",
        path="data/real/pontrelli_single.pt",
        source="derived from the Pontrelli quarry digital outcrop (Francioni et al. 2021)",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="PONTRELLI_QUARRY",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:pontrelli_raw_point_clouds",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_BOOK),
        notes=(
            "Digital-outcrop derived net. R145 hard decision: link-only, no redistribution. The data-attachment "
            "licence and the interpretation label provenance are not verified in M0."
        ),
    ),
    _asset(
        "pontrelli_multi",
        path="data/real/pontrelli_multi.pt",
        source="derived from the Pontrelli quarry digital outcrop (Francioni et al. 2021)",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="PONTRELLI_QUARRY",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:pontrelli_raw_point_clouds",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_BOOK),
        notes="Multi-panel Pontrelli derived net; link-only per R145; surface geometry only.",
    ),
    _asset(
        "pontrelli_raw_point_clouds",
        path="data/external/efracture_data/Pontrelli_wall.zip",
        path_extra=(
            "data/external/efracture_data/Wall_point_cloud.zip",
            "data/external/efracture_data/Pontrelli",
        ),
        source="Pontrelli quarry digital outcrop (Francioni et al. 2021, DOI 10.5194/se-12-2055-2021)",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="PONTRELLI_QUARRY",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_BOOK),
        notes=(
            "Raw/point-cloud holdings for the Pontrelli quarry. The paper is public, but the data attachment "
            "licence is unverified (R145 section 6 U6) and R145 hard decision is link-only."
        ),
    ),
    _asset(
        "aland_20m",
        path="data/real/aland_20m.pt",
        source="derived from Aland trace GeoJSON via project adapter",
        modality="network_tensor",
        license_id=UNKNOWN,
        license_status="NO_LICENSE_FOUND",
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="INHERITED_FROM_SOURCE",
        site_id="ALAND_GETA",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:aland_traces_repo",
        claim_boundary=CB_SURFACE_DERIVED,
        verified_scope=("license_primary_negative", "truth_inherited_from_source"),
        evidence_refs=(EV_G58_LIC, EV_ALAND_README, EV_COMPLIANCE),
        notes=(
            "Derived from 8 Aland trace windows (one site). The repository has no licence (API field null, no "
            "LICENSE file); README only states ownership by the author and GTK KYT KARIKKO. Derived labels "
            "(fracture_id/set_ids) are project-generated. 2D surface traces only."
        ),
    ),
    _asset(
        "aland_traces_repo",
        path="data/external/aland_islands_repo/data/trace_data/traces",
        path_extra=("data/external/aland_islands_repo/data/azimuth_sets.json",),
        source="https://github.com/nialov/multi-scale-fracture-networks-aland-islands-2022",
        modality="outcrop_trace_map",
        license_id=UNKNOWN,
        license_status="NO_LICENSE_FOUND",
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="PROJECT_RECORDED",
        site_id="ALAND_GETA",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_primary_negative", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_ALAND_README, EV_COMPLIANCE),
        notes=(
            "13 GeoJSON trace files across 20m / 200000 / infinity scales plus azimuth_sets.json. The repository "
            "has no licence; R145 forbids redistribution. Eight 20m windows are windows of one site, not eight "
            "independent sites."
        ),
    ),
    _asset(
        "aland_orthomosaic",
        path="data/external/efracture_data/downloads/03_Aland_Islands_Ortho/geta_orthomosaics.zip",
        path_extra=("data/external/efracture_data/downloads/03_Aland_Islands_Ortho/extracted",),
        source="https://zenodo.org/record/4719627",
        modality="orthomosaic",
        license_id="CC-BY-4.0",
        license_status="PROJECT_RECORDED",
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="ALAND_GETA",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_project_record", "truth_not_verified"),
        evidence_refs=(EV_G58_LIC, EV_COMPLIANCE),
        notes=(
            "2.2 GB orthomosaic zip; G58 records the Zenodo 4719627 record as CC-BY-4.0. Raw imagery, no fracture "
            "truth labels; it is the source for trace interpretation, not a truth asset."
        ),
    ),
    # -- public / staged DFN, FMI, core, traces ---------------------------
    _asset(
        "decovalex_4frac",
        path="data/external/decovalex_dfn/4-frac_DFN_data.zip",
        path_extra=(
            "data/external/decovalex_dfn/dfn_extra",
            "results/r149_data_beachhead/raw/zen14873207_decovalex_d23_4frac_dfn.zip",
        ),
        source="https://zenodo.org/records/14873207 (DOI 10.5281/zenodo.14873207)",
        modality="dfn_geometry",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXACT_SYNTHETIC",
        truth_status="DOCUMENTED_IN_TECH_BOOK",
        site_id="DECOVALEX_4FRAC",
        independent_unit="无",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_DFN,
        verified_scope=("license_primary_api", "truth_documented_in_tech_book"),
        evidence_refs=(EV_G58_LIC, EV_R149_REPORT, EV_BOOK),
        notes=(
            "dfnWorks 4-frac deterministic benchmark: normal_vectors / translations / radii / polygons. The "
            "geometry is exact and controlled, but it is not a natural-rock site. R149 precheck: 4 + 1,089 "
            "normals; 2 and 4 orientation groups. Copies exist in data/external and results/; byte equality "
            "against the Zenodo original is not re-checked in M0."
        ),
    ),
    _asset(
        "forge_16a_fmi_log",
        path="data/external/FORGE_16A-32_FMI_fracture_log.xlsx",
        source="https://gdr.openei.org/submissions/1595 (DOI 10.15121/2339498)",
        modality="borehole_fmi",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD_16A",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="UNIQUE",
        claim_boundary=CB_FMI,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "63,611 B FMI fracture log (MD/TVD/azimuth/dip direction/dip/type), GDR 1595. G58 records the "
            "GDR primary page licence CC-BY-4.0. The label payload was not re-opened in M0."
        ),
    ),
    _asset(
        "forge_16b_core_zip",
        path="data/external/FORGE_16B-32_core_fracture_analysis.zip",
        path_extra=("data/external/_forge_extract",),
        source="https://gdr.openei.org/submissions/1708 (DOI 10.15121/2516749)",
        modality="core_analysis",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD_16B",
        independent_unit="整岩芯",
        independent_unit_count=3,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_FMI,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "GDR 1708 core fracture analysis zip; extracted Stim #1-#3 core PDF/XLSX in _forge_extract. Three "
            "stimulation core intervals are not three independent wells. GDR primary page licence CC-BY-4.0."
        ),
    ),
    _asset(
        "forge_16b_core_extract",
        path="data/external/_forge_extract",
        source="extracted from FORGE_16B-32_core_fracture_analysis.zip (GDR 1708)",
        modality="core_analysis",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="INHERITED_FROM_SOURCE",
        site_id="FORGE_MILFORD_16B",
        independent_unit="整岩芯",
        independent_unit_count=3,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_COPY_OF:forge_16b_core_zip",
        claim_boundary=CB_FMI,
        verified_scope=("license_inherited_from_primary", "truth_inherited_from_source"),
        evidence_refs=(EV_G58_LIC, EV_COMPLIANCE),
        notes="Extracted core PDFs/DOCX/XLSX from the GDR 1708 zip; attribution required.",
    ),
    _asset(
        "forge_16b_fmi_las",
        path="data/external/utah_forge_fmi/FORGE_16B_FMI_Reinterpretation.las",
        source="https://gdr.openei.org/submissions/1727",
        modality="borehole_fmi",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD_16B",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="UNIQUE",
        claim_boundary=CB_FMI,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes=(
            "LAS 3.0 reinterpretation of Thrubit FMI, well 16B(78)-32; GDR 1727 primary page licence CC-BY-4.0. "
            "Natural/induced pick types are expert-interpreted, not a complete 3D truth."
        ),
    ),
    _asset(
        "forge_58_32_fmi_las",
        path="data/external/utah_forge_fmi/FORGE_58_32_FMI_Run1_2226_7550ft.las",
        source="https://gdr.openei.org/submissions/1076",
        modality="borehole_fmi",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD_58_32",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="UNIQUE",
        claim_boundary=CB_FMI,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE),
        notes="FMI LAS, well 58-32 (MU-ESW1), 2,226-7,550 ft; GDR 1076 primary page licence CC-BY-4.0.",
    ),
    _asset(
        "forge_fmi_derived_outputs",
        path="data/external/utah_forge_fmi/forge_fmi_2wells.pt",
        path_extra=(
            "data/external/utah_forge_fmi/FORGE_16B_FMI_Reinterpretation_routeA.pt",
            "data/external/utah_forge_fmi/FORGE_16B_FMI_Reinterpretation_routeA_group_table.csv",
            "data/external/utah_forge_fmi/forge_fmi_survey.json",
            "data/external/utah_forge_fmi/forge_fmi_survey.png",
        ),
        source="derived from GDR 1727 + GDR 1076 FMI picks",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level="EXPERT_BOREHOLE_PICKS",
        truth_status="INHERITED_FROM_SOURCE",
        site_id="FORGE_MILFORD_16B_58_32",
        independent_unit="整井",
        independent_unit_count=2,
        public_or_private="PUBLIC",
        dedupe_status="DERIVED_FROM:forge_16b_fmi_las+forge_58_32_fmi_las",
        claim_boundary=CB_FMI,
        verified_scope=("license_inherited_from_primary", "truth_inherited_from_source"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_TOXIC),
        notes=(
            "Derived 16B/58-32 FMI products. The 16B routeA outputs are registered here; the 58-32 routeA mirror "
            "is separated into forge_58_32_routea_mirror because W74 marked it toxic. Attribution required."
        ),
    ),
    _asset(
        "forge_58_32_routea_mirror",
        path="data/external/utah_forge_fmi/FORGE_58_32_FMI_Run1_2226_7550ft_routeA.pt",
        path_extra=("data/external/utah_forge_fmi/FORGE_58_32_FMI_Run1_2226_7550ft_routeA_group_table.csv",),
        source="derived from GDR 1076 (58-32 FMI) by the project routeA pipeline",
        modality="network_tensor",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_INHERITED",
        truth_level=UNKNOWN,
        truth_status="VOIDED",
        site_id="FORGE_MILFORD_58_32",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:forge_58_32_fmi_las",
        claim_boundary=CB_TOXIC,
        verified_scope=("license_inherited_from_primary", "truth_voided_by_w74"),
        evidence_refs=(EV_TOXIC, EV_G58_LIC),
        notes=(
            "W74 toxicity note: normals are a sin/cos complement mirror (nz=sin(dip), upper-hemisphere), not the "
            "published normal; group statistics are void. Keep only for lineage/audit; do not use as truth, "
            "precision evidence or a published-normal substitute."
        ),
    ),
    _asset(
        "snake_river_inel1",
        path="data/external/snake_river_INEL1",
        source="GDR/USGS public well data; individual DOI/licence page not verified",
        modality="borehole_log",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="SNAKE_RIVER_INEL1",
        independent_unit="整井",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_EFRACTURE, EV_COMPLIANCE),
        notes=(
            "24 LAS logs + lithology + rock strength. The 'INEL-1 Fracture ID Log.pdf' is not on disk; no fracture "
            "picks are registered. EFRACTURE_DATA/R145 record GDR CC-BY-4.0 as a project claim, but the individual "
            "DOI/licence page is not verified; M0 keeps licence UNKNOWN."
        ),
    ),
    _asset(
        "egs_collab_instrumentation",
        path="data/external/gdr_egs_collab",
        source="GDR EGS Collab well instrumentation; individual DOI/licence page not verified",
        modality="well_metadata",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="EGS_COLLAB",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_EFRACTURE, EV_COMPLIANCE),
        notes=(
            "Well instrumentation depths, source-receiver mapping, geode geometry, accelerometer orientations. "
            "Not a fracture-orientation dataset; no fracture truth. Licence page not verified in M0."
        ),
    ),
    _asset(
        "forge_wellheads",
        path="data/external/forge_wellheads",
        path_extra=("data/external/wellheads_forge.json",),
        source="GDR/USGS FORGE well surveys; individual DOI/licence page not verified",
        modality="well_metadata",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="FORGE_MILFORD",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_EFRACTURE, EV_COMPLIANCE),
        notes="Directional surveys / well locations for MD-to-xyz work; no fracture truth. Licence not verified.",
    ),
    _asset(
        "published_reports",
        path="data/external/published_reports",
        source="public geotechnical reports (CNKI/ministry portals); individual URLs/rights not enumerated",
        modality="literature_table",
        license_id=UNKNOWN,
        license_status="RESTRICTED_INTERNAL",
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="PUBLISHED_REPORTS_MULTIPLE",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RESTRICTED,
        verified_scope=("license_restricted_internal", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_DATA_MAP),
        notes=(
            "Compiled report excerpts and extracted CSV tables. The README forbids redistributing original PDFs; "
            "third-party report copyright. Individual boreholes/tables and truth levels are not verified in M0."
        ),
    ),
    _asset(
        "literature_group_table",
        path="data/external/literature_sets.csv",
        source="project-compiled orientation set table from published papers/project tables",
        modality="literature_table",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="LITERATURE_MULTIPLE",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_EFRACTURE, EV_COMPLIANCE),
        notes=(
            "Set-centre table citing literature and project routeA group tables. It is a secondary compilation; "
            "the underlying reports and individual licences are not verified in M0."
        ),
    ),
    # -- E: point clouds and special assets -------------------------------
    _asset(
        "hrf_frac",
        path="data/external/efracture_data/hrf_frac.npz",
        source="OpenTopography UT19_Bunds wall point cloud; project-labelled derivative",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="PROJECT_RECORDED",
        site_id="HRF_WALL",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:hrf_raw_point_clouds",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_not_verified", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE, EV_DATA_MAP),
        notes=(
            "4.48M-point labelled wall point cloud (xyz/nrm/fid + fracture normals), 136 fractures + noise. "
            "G58 records expert-interpreted fracture_id; M0 did not re-open the payload. Underlying OpenTopography "
            "UT19_Bunds licence is unverified; R145 forbids redistribution of derived local files."
        ),
    ),
    _asset(
        "new_frac",
        path="data/external/efracture_data/new_frac.npz",
        source="second wall scan; project-labelled derivative; site attribution ambiguous",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="PROJECT_RECORDED",
        site_id=UNKNOWN,
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:hrf_raw_point_clouds",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_not_verified", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG, EV_COMPLIANCE, EV_DATA_MAP),
        notes=(
            "2.45M-point labelled wall point cloud, 568 fractures + noise. G58 records expert-interpreted "
            "fracture_id, but project records note historical site-attribution ambiguity (Pontrelli misuse). "
            "Site_id is therefore UNKNOWN in M0."
        ),
    ),
    _asset(
        "new_subsample",
        path="data/external/efracture_data/new_subsample.npz",
        source="point subsample of the second wall scan",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id=UNKNOWN,
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:new_frac",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG),
        notes="Unlabelled point subsample (g/r/class_cnt); not a fracture-truth asset.",
    ),
    _asset(
        "routeb_result",
        path="data/external/efracture_data/routeB_result.npz",
        source="project diagnostic result archive for the labelled wall point clouds",
        modality="diagnostic_archive",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="HRF_WALL",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PRIVATE",
        dedupe_status="DERIVED_FROM:hrf_frac+new_frac",
        claim_boundary=CB_INTERNAL,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_G58_LIC, EV_G58_REG),
        notes=(
            "Diagnostic archive (predicted/true normals, per-point errors, fracture ids). It is an old result "
            "carrier, not a truth dataset; oracle/diagnostic readings must not enter external claims."
        ),
    ),
    _asset(
        "hrf_raw_point_clouds",
        path="data/external/efracture_data/points.laz",
        path_extra=("data/external/efracture_data/points_new_758M.laz",),
        source="OpenTopography UT19_Bunds raw wall point clouds; project acquisition copy",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="HRF_WALL",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_COMPLIANCE, EV_DATA_MAP),
        notes=(
            "Raw LAZ point clouds (481 MB + 759 MB). OpenTopography UT19_Bunds licence is unverified (R145 U2); "
            "no fracture labels. These are the source of hrf_frac/new_frac."
        ),
    ),
    _asset(
        "geocrack_body",
        path="data/external/efracture_data/geocrack_official/patched_images",
        path_extra=(
            "data/external/efracture_data/geocrack/patched_images.zip",
            "data/external/efracture_data/geocrack_r126",
            "data/external/efracture_data/geocrack_test",
        ),
        source="https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/E4OXHQ",
        modality="outcrop_image_patch",
        license_id="CC0-1.0",
        license_status="VERIFIED_PRIMARY",
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="PROJECT_RECORDED",
        site_id="GEOCRACK_11_SITES",
        independent_unit="整站点",
        independent_unit_count=11,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_primary_api", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_R176_LIC, EV_R181_INV),
        notes=(
            "24,986 PNG = 12,493 original/mask pairs at 224x224 (R181 inventory); 49 source stems, 11 sites in "
            "the paper. Local patches are not site-georeferenced, so site-level partitioning is not available "
            "from the files alone. CC0-1.0 is API-verified; the paper text is CC BY-NC-ND and must not be copied. "
            "The R126 subset has one known mispaired stem (0V5A0037)."
        ),
    ),
    _asset(
        "loviisa_traces",
        path="data/external/efracture_data/loviisa_traces.gpkg",
        path_extra=(
            "data/external/efracture_data/loviisa_traces_exported",
            "data/external/efracture_data/loviisa_traces.zip",
            "data/external/efracture_data/loviisa.zip",
        ),
        source="GTK / Loviisa trace export (EPSG:3067); GTK open data",
        modality="outcrop_trace_map",
        license_id="CC-BY-4.0",
        license_status="PROJECT_RECORDED",
        truth_level="EXPERT_SURFACE_INTERPRETATION",
        truth_status="PROJECT_RECORDED",
        site_id="LOVIISA",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_SURFACE,
        verified_scope=("license_project_record", "truth_project_record"),
        evidence_refs=(EV_DATA_MAP, EV_COMPLIANCE),
        notes=(
            "6,760 traces / 8 subsites recorded in DATA_MAP W43. GTK is recorded as CC-BY-4.0; the original GTK "
            "page is not re-checked in M0. Surface traces only; 8 subsites are not 8 independent sites."
        ),
    ),
    # -- GDR / Zenodo / RockBench staged public data ----------------------
    _asset(
        "gdr1222_dfn",
        path="results/r149_data_beachhead/raw/gdr1222_Filtered_800m_all_csv.zip",
        path_extra=("results/r149_data_beachhead/extracted/gdr1222",),
        source="https://gdr.openei.org/submissions/1222 (DOI 10.15121/1776595)",
        modality="dfn_table",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_DFN,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_R149_REPORT, EV_R149_M4),
        notes=(
            "30 CSV inside the zip; R149 precheck 146,238 fracture rows, 3 orientation sets. GDR 1222 is a DFN "
            "model built from microseismic data, not a complete 3D truth and not a borehole-pick tier. Licence "
            "CC-BY-4.0 verified on the GDR primary page. Other staging copies exist under forge_dfns/."
        ),
    ),
    _asset(
        "gdr1646_meq_dfn",
        path="results/r149_data_beachhead/raw/gdr1646_2024_07_July_Updated_MEQ_DFN.zip",
        path_extra=("results/r149_data_beachhead/extracted/gdr1646",),
        source="https://gdr.openei.org/submissions/1646 (DOI 10.15121/2440870)",
        modality="dfn_table",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_DFN,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_R149_REPORT, EV_R149_M4),
        notes=(
            "MEQ DFN v2 July 2024; R149 precheck 44 fracture rows, 3 orientation sets. DFN model, not a complete "
            "3D truth. Licence CC-BY-4.0 verified on the GDR primary page."
        ),
    ),
    _asset(
        "gdr1750_dfn_2025",
        path="results/r149_data_beachhead/raw/gdr1750_DFN_Update_2025_v1.zip",
        path_extra=("results/r149_data_beachhead/extracted/gdr1750",),
        source="https://gdr.openei.org/submissions/1750",
        modality="dfn_table",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level=UNKNOWN,
        truth_status="PROJECT_RECORDED",
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="DUPLICATE_COPIES_PRESENT",
        claim_boundary=CB_DFN,
        verified_scope=("license_primary_page", "truth_project_record"),
        evidence_refs=(EV_G58_LIC, EV_R149_REPORT, EV_R149_M4, EV_DATA_MAP),
        notes=(
            "Discrete (NF/HF 133) + stochastic (Set 1-4, 19,544) DFN; R149 precheck 19,677 rows. DATA_MAP R161 "
            "records Set 3/4 near-isotropic (Set 3 dispersion ~62.6 deg), so the set labels are not uniformly "
            "directional truth. Licence CC-BY-4.0 verified on the GDR primary page."
        ),
    ),
    _asset(
        "forge2024_microseismic",
        path="results/r149_data_beachhead/raw/zen12817881_microseismic_FORGE2024.csv",
        source="https://zenodo.org/records/12817881",
        modality="microseismic_catalog",
        license_id="CC-BY-SA-4.0",
        license_status="PROJECT_RECORDED",
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="FORGE_MILFORD",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PUBLIC",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_project_record", "truth_not_verified"),
        evidence_refs=(EV_R149_REPORT, EV_R149_M4),
        notes=(
            "2,901-event microseismic catalog (R149 M4). R149 reports a DataCite API check of CC BY-SA 4.0. "
            "No fracture orientation fields; not a fracture or connectivity truth asset."
        ),
    ),
    _asset(
        "zenodo_18078781",
        path="results/r159_g1_unlock/label.zip",
        path_extra=(
            "results/r159_g1_unlock/extracted/label",
            "results/r159_g1_unlock/rockmass.zip",
        ),
        source="https://zenodo.org/records/18078781 (DOI 10.5281/zenodo.18078781)",
        modality="outcrop_image_patch",
        license_id="CC-BY-4.0",
        license_status="VERIFIED_PRIMARY",
        truth_level=UNKNOWN,
        truth_status="PARTIAL",
        site_id="ZENODO_18078781_MULTIPLE",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        public_or_private="PUBLIC",
        dedupe_status="PARTIAL_DOWNLOAD",
        claim_boundary=CB_PARTIAL,
        verified_scope=("license_primary_api", "truth_partial_download"),
        evidence_refs=(EV_R159_API, EV_R159_INV, EV_R149_REPORT),
        notes=(
            "Zenodo API records CC-BY-4.0. Local label.zip contains 3,803 files / 3,687 PNG labels (968 Benchmark "
            "+ 200 Box + 2,519 FracMan), 12.4% of the 29,752 claimed; the real_world_rock_slope subset is absent. "
            "rockmass.zip (2.9 GB) is present but not audited. Patches are not independent sites."
        ),
    ),
    _asset(
        "rockbench_realslope",
        path="results/r149_data_beachhead/extracted/rockbench/realslope.las",
        source="https://github.com/lcdfljm/rock-discontinuity-point-cloud",
        modality="point_cloud",
        license_id=UNKNOWN,
        license_status=UNKNOWN,
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="ROCKBENCH_OURAY",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        dedupe_status="UNIQUE",
        claim_boundary=CB_RAW,
        verified_scope=("license_not_verified", "truth_not_verified"),
        evidence_refs=(EV_R149_REPORT, EV_R149_M4),
        notes=(
            "37 MB LAS, 1,236,573 points, LAS 1.4 (R149). R149 M4 records MIT, but no primary licence file/page "
            "is on disk; M0 keeps licence UNKNOWN. No fracture_id or orientation columns; raw slope point cloud."
        ),
    ),
    # -- public candidates not downloaded ---------------------------------
    _asset(
        "candidate_fatdom_m11",
        source="https://doi.org/10.5281/zenodo.18313900 (FATDOM-DB model 11)",
        modality="point_cloud",
        site_id="FATDOM_M11",
        independent_unit="整站点",
        independent_unit_count=1,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_R149_M4, EV_BOOK),
        notes=(
            "Candidate only; 146 GB, not downloaded. DataCite description claims structural measurements, but the "
            "full annotation vectors are unverified. DataCite rights field was empty; expected CC-BY-4.0 is not "
            "verified. Licence and truth level UNKNOWN."
        ),
    ),
    _asset(
        "candidate_konusdalen_west",
        source="Norstore / Svalbox-DOM_2020-0039 (Konusdalen West, Svalbard); URL not resolved",
        modality="point_cloud",
        site_id="KONUSDALEN_WEST",
        independent_unit="整站点",
        independent_unit_count=1,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_R149_REPORT, EV_R149_M4, EV_BOOK),
        notes=(
            "Candidate only; no online download resolved in R149 (two searches). DOM/orthophoto plus fault/fracture/"
            "bedding interpretation claimed; file resolution, licence and label truth are UNKNOWN."
        ),
    ),
    _asset(
        "candidate_v3geo",
        source="https://v3geo.com (Virtual 3D Geoscience); per-model licences",
        modality="point_cloud",
        site_id="V3GEO_MULTIPLE",
        independent_unit="整站点",
        independent_unit_count=UNKNOWN,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_DATA_MAP, EV_R149_M4, EV_BOOK),
        notes=(
            "Candidate discovery entry. JS-rendered SPA blocked automated access in R149; per-model licences and "
            "labels are not verified. M0 does not claim any V3Geo file or licence."
        ),
    ),
    _asset(
        "candidate_national_borehole_db",
        source="https://ndcp.cgsi.cn/ (China Geological Survey national borehole database)",
        modality="borehole_log",
        site_id="CGS_NATIONAL_DB",
        independent_unit="整井",
        independent_unit_count=UNKNOWN,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_EFRACTURE, EV_R149_M4, EV_BOOK),
        notes=(
            "Candidate only; registration/authorisation required, DNS/network not reachable in R149. No file, "
            "label or licence verified. Not usable as a public main denominator in M0."
        ),
    ),
    _asset(
        "candidate_plos_one_wellwall",
        source="https://doi.org/10.1371/journal.pone.0299471 (PLOS ONE well-wall image set)",
        modality="borehole_image",
        license_id="CC-BY-4.0",
        license_status="PROJECT_RECORDED",
        truth_level=UNKNOWN,
        truth_status=UNKNOWN,
        site_id="PLOS_ONE_20_WELLS",
        independent_unit="整井",
        independent_unit_count=20,
        public_or_private="PUBLIC",
        claim_boundary=CB_CANDIDATE,
        verified_scope=("license_project_record", "truth_not_verified"),
        evidence_refs=(EV_DATA_MAP, EV_R149_M4),
        notes=(
            "Candidate only; DATA_MAP/R149 M4 record 20 wells / 1,013 well-wall images under CC-BY-4.0. The "
            "supporting-information files and annotation truth are not on disk or verified in M0. 20 wells are "
            "potential units; this is not a downloaded dataset."
        ),
    ),
    _asset(
        "candidate_aspo_hrl_dfn",
        source="SKB IPR-01-71 (Aspo HRL DFN report, 1996); non-public licence",
        modality="dfn_table",
        site_id="ASPO_HRL",
        independent_unit="整站点",
        independent_unit_count=1,
        public_or_private="PRIVATE",
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_DATA_MAP, EV_R149_M4),
        notes=(
            "Candidate/report only; R149 records 3,513 fractures in an appendix but the PDF is not downloaded. "
            "SKB internal/non-public licence; no file, label or redistribution right verified in M0."
        ),
    ),
    _asset(
        "candidate_carrara_mxrct",
        source="technical book section 5.2 citation [22] (Carrara mu-XRCT); citation not resolved",
        modality="core_ct",
        truth_level="LAB_VOLUME_REFERENCE",
        truth_status="DOCUMENTED_IN_TECH_BOOK",
        site_id="CARRARA_MXRCT",
        independent_unit="整岩芯",
        independent_unit_count=1,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only", "truth_documented_in_tech_book"),
        evidence_refs=(EV_BOOK,),
        notes=(
            "Candidate only; single sample/single lithology CT volume. The book describes a CT reference and warns "
            "that the reference segmentation is not absolute truth. No file or licence is on disk; licence UNKNOWN."
        ),
    ),
    _asset(
        "candidate_dpm",
        source="technical book section 5.2 citation [17] (DPM candidate-discovery entry)",
        modality="point_cloud",
        site_id="DPM",
        independent_unit="无",
        independent_unit_count=UNKNOWN,
        claim_boundary=CB_CANDIDATE,
        verified_scope=("source_record_only",),
        evidence_refs=(EV_BOOK,),
        notes=(
            "Candidate-discovery entry only. The book explicitly warns that directory accessibility is not the same "
            "as per-dataset licence, files or labels being verified. No download in M0."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _violation(code: str, asset_id: str, detail: str) -> dict:
    return {"code": code, "asset_id": asset_id, "detail": detail}


def _is_unknown(value: Any) -> bool:
    return value == UNKNOWN


def _dedupe_refs(dedupe_status: str) -> list[str]:
    if not isinstance(dedupe_status, str) or ":" not in dedupe_status:
        return []
    _, raw = dedupe_status.split(":", 1)
    return [item for item in raw.split("+") if item]


def validate_registry(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Validate rows structurally; returns a list of violations (empty == clean).

    Raises SystemExit when the registry is empty: an empty sample set must not
    be recorded as a clean audit (project rule: no PASS on empty input).
    """
    rows = list(rows or [])
    if not rows:
        raise SystemExit("EMPTY_REGISTRY: zero rows; refusing to record a clean audit")

    violations: list[dict] = []
    seen_ids: dict[str, int] = {}
    all_ids = [str(row.get("asset_id")) for row in rows if isinstance(row, Mapping)]

    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            violations.append(_violation("ROW_NOT_OBJECT", "#%d" % idx, "row is %r" % (type(row),)))
            continue
        rid = str(row.get("asset_id") or "#%d" % idx)
        for field in REQUIRED_FIELDS:
            if field not in row:
                violations.append(_violation("FIELD_MISSING", rid, "required field %r missing" % field))
                continue
            if row[field] is None and field not in NULLABLE_FIELDS:
                violations.append(_violation("FIELD_MISSING", rid, "required field %r is null" % field))
        if not isinstance(row.get("asset_id"), str) or not ASSET_ID_RE.match(row["asset_id"]):
            violations.append(_violation("ASSET_ID_FORMAT", rid, "asset_id must match %s" % ASSET_ID_RE.pattern))
        seen_ids[rid] = seen_ids.get(rid, 0) + 1

        path = row.get("path")
        if path is not None and (not isinstance(path, str) or not path.strip()):
            violations.append(_violation("PATH_INVALID", rid, "path must be a non-empty string or null"))
        if not isinstance(row.get("path_extra"), list) or not all(
            isinstance(p, str) and p.strip() for p in row.get("path_extra") or []
        ):
            violations.append(_violation("PATH_EXTRA_INVALID", rid, "path_extra must be a list of non-empty strings"))
        if not isinstance(row.get("source"), str) or not row["source"].strip():
            violations.append(_violation("SOURCE_EMPTY", rid, "source must be a non-empty string"))
        if row.get("modality") not in MODALITIES:
            violations.append(_violation("MODALITY_INVALID", rid, "modality=%r not in %r" % (row.get("modality"), MODALITIES)))
        if row.get("license_status") not in LICENSE_STATUSES:
            violations.append(_violation("LICENSE_STATUS_INVALID", rid, "license_status=%r" % (row.get("license_status"),)))
        if row.get("truth_level") not in TRUTH_LEVELS:
            violations.append(_violation("TRUTH_LEVEL_INVALID", rid, "truth_level=%r" % (row.get("truth_level"),)))
        if row.get("truth_status") not in TRUTH_STATUSES:
            violations.append(_violation("TRUTH_STATUS_INVALID", rid, "truth_status=%r" % (row.get("truth_status"),)))
        if row.get("public_or_private") not in PUBLIC_PRIVATE:
            violations.append(_violation("PUBLIC_PRIVATE_INVALID", rid, "public_or_private=%r" % (row.get("public_or_private"),)))
        if row.get("independent_unit") not in INDEPENDENT_UNITS:
            violations.append(_violation("INDEPENDENT_UNIT_INVALID", rid, "independent_unit=%r" % (row.get("independent_unit"),)))
        count = row.get("independent_unit_count")
        if count != UNKNOWN and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            violations.append(_violation("UNIT_COUNT_INVALID", rid, "independent_unit_count=%r" % (count,)))
        if not isinstance(row.get("site_id"), str) or not row["site_id"].strip():
            violations.append(_violation("SITE_ID_EMPTY", rid, "site_id must be a non-empty string"))
        if not isinstance(row.get("notes"), str):
            violations.append(_violation("NOTES_NOT_STRING", rid, "notes must be a string"))
        if not isinstance(row.get("evidence_refs"), list) or not all(
            isinstance(x, str) and x.strip() for x in row.get("evidence_refs") or []
        ):
            violations.append(_violation("EVIDENCE_REFS_INVALID", rid, "evidence_refs must be a list of non-empty strings"))
        if not isinstance(row.get("verified_scope"), list) or not row.get("verified_scope"):
            violations.append(_violation("VERIFIED_SCOPE_EMPTY", rid, "verified_scope must be a non-empty list"))
        if not isinstance(row.get("verified_at"), str) or not row["verified_at"].strip():
            violations.append(_violation("VERIFIED_AT_INVALID", rid, "verified_at must be a non-empty string"))

        # Licence honesty: unknown/restricted statuses must not carry a named licence.
        status = row.get("license_status")
        license_id = row.get("license_id")
        if status in LICENSE_NO_NAME_STATUSES and license_id != UNKNOWN:
            violations.append(
                _violation("LICENSE_WASH", rid, "license_status=%s requires license_id=UNKNOWN, got %r" % (status, license_id))
            )
        if status in {"VERIFIED_PRIMARY", "VERIFIED_INHERITED", "PROJECT_RECORDED"} and license_id in (None, "", UNKNOWN):
            violations.append(
                _violation("LICENSE_EVIDENCE_MISSING", rid, "license_status=%s requires a named license" % status)
            )
        if status == "PROJECT_INTERNAL" and license_id != "PROJECT_INTERNAL":
            violations.append(_violation("LICENSE_INTERNAL_INVALID", rid, "PROJECT_INTERNAL status requires license_id=PROJECT_INTERNAL"))

        # Truth honesty: UNKNOWN/voided/partial statuses must not carry a known tier.
        truth_status = row.get("truth_status")
        truth_level = row.get("truth_level")
        if truth_status in {UNKNOWN, "VOIDED", "PARTIAL"} and truth_level != UNKNOWN:
            violations.append(
                _violation("TRUTH_WASH", rid, "truth_status=%s requires truth_level=UNKNOWN, got %r" % (truth_status, truth_level))
            )
        if truth_level != UNKNOWN and truth_status in {UNKNOWN, "VOIDED", "PARTIAL"}:
            violations.append(_violation("TRUTH_STATUS_CONFLICT", rid, "known truth_level with status %s" % truth_status))

        # Claim boundaries: non-empty, at least one FORBIDDEN line, no "all available" claim.
        boundary = row.get("claim_boundary")
        if not isinstance(boundary, list) or not boundary or not all(isinstance(x, str) and x.strip() for x in boundary):
            violations.append(_violation("CLAIM_BOUNDARY_EMPTY", rid, "claim_boundary must be a non-empty string list"))
        else:
            if not any(x.startswith(CLAIM_FORBIDDEN_PREFIX) for x in boundary):
                violations.append(_violation("CLAIM_FORBIDDEN_MISSING", rid, "claim_boundary must contain a FORBIDDEN line"))
            blob = " ".join(boundary).lower()
            for word in CLAIM_FORBIDDEN_WORDS:
                if word.lower() in blob:
                    violations.append(_violation("CLAIM_ALL_AVAILABLE", rid, "claim_boundary contains forbidden phrase %r" % word))

        # Dedupe declarations.
        dedupe = row.get("dedupe_status")
        if not isinstance(dedupe, str) or not dedupe.strip():
            violations.append(_violation("DEDUPE_EMPTY", rid, "dedupe_status must be non-empty"))
        else:
            prefix = dedupe.split(":", 1)[0]
            if prefix not in DEDUPE_PLAIN and prefix not in DEDUPE_PREFIXES:
                violations.append(_violation("DEDUPE_INVALID", rid, "unknown dedupe_status %r" % dedupe))
            if prefix in DEDUPE_PREFIXES:
                refs = _dedupe_refs(dedupe)
                if not refs:
                    violations.append(_violation("DEDUPE_REF_MISSING", rid, "dedupe_status %r lacks references" % dedupe))
                for ref in refs:
                    if ref not in all_ids:
                        violations.append(_violation("DEDUPE_REF_UNKNOWN", rid, "dedupe reference %r does not exist" % ref))

    for rid, n in sorted(seen_ids.items()):
        if n > 1:
            violations.append(_violation("DUPLICATE_ASSET_ID", rid, "asset_id appears %d times" % n))

    return violations


# ---------------------------------------------------------------------------
# Probes and registry build
# ---------------------------------------------------------------------------
def file_sha256(path: str | os.PathLike, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _now_local() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def probe_path(root: str | os.PathLike, rel_path: str | None, hash_limit_bytes: int = 64 * 1024 * 1024) -> dict:
    """Probe one path without reading payloads.  Large files are not hashed."""
    if not rel_path:
        return {
            "path": None,
            "exists_on_disk": False,
            "path_kind": "missing",
            "bytes": UNKNOWN,
            "sha256": UNKNOWN,
            "n_top_level_files": UNKNOWN,
            "n_top_level_dirs": UNKNOWN,
            "hash_skipped_reason": "no_path",
        }
    base = Path(root)
    target = Path(rel_path)
    if not target.is_absolute():
        target = base / target
    exists = target.exists()
    if not exists:
        return {
            "path": str(rel_path),
            "exists_on_disk": False,
            "path_kind": "missing",
            "bytes": UNKNOWN,
            "sha256": UNKNOWN,
            "n_top_level_files": UNKNOWN,
            "n_top_level_dirs": UNKNOWN,
            "hash_skipped_reason": "missing",
        }
    if target.is_file():
        size = target.stat().st_size
        digest = UNKNOWN
        skipped = None
        if size <= hash_limit_bytes:
            try:
                digest = file_sha256(target)
            except OSError:
                skipped = "hash_error"
        else:
            skipped = "size_exceeds_limit"
        return {
            "path": str(rel_path),
            "exists_on_disk": True,
            "path_kind": "file",
            "bytes": size,
            "sha256": digest,
            "n_top_level_files": 1,
            "n_top_level_dirs": 0,
            "hash_skipped_reason": skipped,
        }
    if target.is_dir():
        n_files = n_dirs = UNKNOWN
        try:
            entries = list(os.scandir(target))
            n_files = sum(1 for e in entries if e.is_file(follow_symlinks=False))
            n_dirs = sum(1 for e in entries if e.is_dir(follow_symlinks=False))
        except OSError:
            pass
        return {
            "path": str(rel_path),
            "exists_on_disk": True,
            "path_kind": "directory",
            "bytes": UNKNOWN,
            "sha256": UNKNOWN,
            "n_top_level_files": n_files,
            "n_top_level_dirs": n_dirs,
            "hash_skipped_reason": "directory",
        }
    return {
        "path": str(rel_path),
        "exists_on_disk": True,
        "path_kind": "other",
        "bytes": UNKNOWN,
        "sha256": UNKNOWN,
        "n_top_level_files": UNKNOWN,
        "n_top_level_dirs": UNKNOWN,
        "hash_skipped_reason": "other",
    }


def build_registry(
    root: str | os.PathLike,
    *,
    probe: bool = True,
    hash_limit_bytes: int = 64 * 1024 * 1024,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Return catalogue rows, optionally with local probe results."""
    source_rows = CATALOGUE if rows is None else rows
    built: list[dict] = []
    timestamp = _now_local() if probe else UNKNOWN
    for raw in source_rows:
        row = copy.deepcopy(dict(raw))
        if probe and row.get("path"):
            primary = probe_path(root, row["path"], hash_limit_bytes)
            extras = [probe_path(root, p, hash_limit_bytes) for p in row.get("path_extra") or []]
            row.update(
                {
                    "exists_on_disk": primary["exists_on_disk"],
                    "path_kind": primary["path_kind"],
                    "bytes": primary["bytes"],
                    "sha256": primary["sha256"],
                    "n_top_level_files": primary["n_top_level_files"],
                    "n_top_level_dirs": primary["n_top_level_dirs"],
                    "hash_skipped_reason": primary["hash_skipped_reason"],
                    "extra_probes": extras,
                }
            )
            scope = list(row.get("verified_scope") or [])
            if primary["exists_on_disk"]:
                scope.append("file_exists")
                if primary["sha256"] != UNKNOWN:
                    scope.append("sha256")
                elif primary["path_kind"] == "file":
                    scope.append("size_only")
                if primary["path_kind"] == "directory":
                    scope.append("top_level_entry_count")
                row["verified_at"] = timestamp
            else:
                scope.append("path_missing")
                row["verified_at"] = UNKNOWN
            row["verified_scope"] = sorted(set(scope))
        else:
            row.setdefault("exists_on_disk", UNKNOWN)
            row.setdefault("path_kind", UNKNOWN)
            row.setdefault("bytes", UNKNOWN)
            row.setdefault("sha256", UNKNOWN)
            row.setdefault("n_top_level_files", UNKNOWN)
            row.setdefault("n_top_level_dirs", UNKNOWN)
            row.setdefault("hash_skipped_reason", "probe_disabled")
            row.setdefault("extra_probes", [])
        built.append(row)
    return built


# ---------------------------------------------------------------------------
# Summary, coverage, M0 readiness
# ---------------------------------------------------------------------------
def _counts(values: Iterable[Any], universe: Sequence[str] | None = None) -> dict:
    out: dict[str, int] = {str(key): 0 for key in (universe or ())}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _dedupe_class(value: str) -> str:
    return value.split(":", 1)[0] if isinstance(value, str) else str(value)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict:
    rows = list(rows)
    unknown_license = sorted(
        r["asset_id"] for r in rows if r.get("license_id") == UNKNOWN
    )
    unknown_truth = sorted(r["asset_id"] for r in rows if r.get("truth_level") == UNKNOWN)
    unknown_either = sorted(set(unknown_license) | set(unknown_truth))
    restricted = sorted(
        r["asset_id"]
        for r in rows
        if r.get("license_status") in LICENSE_NONPERMISSIVE_STATUSES
    )
    weak = sorted(
        r["asset_id"]
        for r in rows
        if r.get("license_status") == "PROJECT_RECORDED"
        or r.get("truth_status") in {"PROJECT_RECORDED", "DOCUMENTED_IN_TECH_BOOK", "PARTIAL"}
    )
    missing = sorted(
        r["asset_id"]
        for r in rows
        if r.get("path") and r.get("exists_on_disk") is False
    )
    candidates = sorted(
        r["asset_id"]
        for r in rows
        if r.get("public_or_private") == "PUBLIC"
        and r.get("license_id") != UNKNOWN
        and r.get("truth_level") != UNKNOWN
        and r.get("independent_unit") != "无"
        and r.get("truth_status") not in {"VOIDED", "PARTIAL"}
        and r.get("dedupe_status") in {"UNIQUE", "DUPLICATE_COPIES_PRESENT"}
    )
    public_rows = [r for r in rows if r.get("public_or_private") == "PUBLIC"]
    public_unknown_license = sorted(r["asset_id"] for r in public_rows if r.get("license_id") == UNKNOWN)
    public_unknown_truth = sorted(r["asset_id"] for r in public_rows if r.get("truth_level") == UNKNOWN)

    blocking = sorted(set(public_unknown_license))
    m0_reason = (
        "No public main denominator is declared in this task. "
        "Public assets with UNKNOWN licence=%d; public assets with UNKNOWN truth=%d; "
        "eligible public primary rows with known licence+truth+unit=%d (derived copies excluded). "
        "The M0 condition cannot be recorded as PASS unless a denominator is declared and every "
        "dataset in it has a non-UNKNOWN licence."
        % (len(public_unknown_license), len(public_unknown_truth), len(candidates))
    )
    m0_verdict = "INSUFFICIENT" if blocking else "NOT_DECLARED"

    return {
        "schema": SCHEMA_SUMMARY,
        "watermark": WATERMARK,
        "n_assets": len(rows),
        "n_local_on_disk": sum(1 for r in rows if r.get("exists_on_disk") is True),
        "n_missing_local_paths": len(missing),
        "n_candidates_not_downloaded": sum(1 for r in rows if not r.get("path")),
        "n_with_any_unknown": len(unknown_either),
        "not_all_usable": True,
        "counts": {
            "truth_level": _counts((r.get("truth_level") for r in rows), TRUTH_LEVELS),
            "truth_status": _counts((r.get("truth_status") for r in rows), TRUTH_STATUSES),
            "license_id": _counts((r.get("license_id") for r in rows)),
            "license_status": _counts((r.get("license_status") for r in rows), LICENSE_STATUSES),
            "independent_unit": _counts((r.get("independent_unit") for r in rows), INDEPENDENT_UNITS),
            "public_or_private": _counts((r.get("public_or_private") for r in rows), PUBLIC_PRIVATE),
            "modality": _counts((r.get("modality") for r in rows), MODALITIES),
            "dedupe_class": _counts(_dedupe_class(r.get("dedupe_status")) for r in rows),
        },
        "unknown_license_assets": unknown_license,
        "unknown_truth_assets": unknown_truth,
        "unknown_either_assets": unknown_either,
        "restricted_assets": restricted,
        "weak_evidence_assets": weak,
        "missing_local_paths": missing,
        "public_assets": sorted(r["asset_id"] for r in public_rows),
        "public_assets_with_unknown_license": public_unknown_license,
        "public_assets_with_unknown_truth": public_unknown_truth,
        "candidate_public_main_denominator": candidates,
        "m0_gate": {
            "condition": (
                "Every intended dataset has a machine-readable row; the public main denominator "
                "has no UNKNOWN licence; insufficient data only narrows the claim, never writes PASS."
            ),
            "main_denominator_declared": False,
            "pass": False,
            "verdict": m0_verdict,
            "blocking_unknown_licenses": blocking,
            "reason": m0_reason,
        },
        "truth_level_scope_note": (
            "UNKNOWN is used for raw sensor/DFN-model/diagnostic assets whose truth tier is not one of the "
            "five section 5.1 tiers. HELD_OUT_OBSERVATION has zero rows here; the parallel REAL_CENSORING_HOLDOUT "
            "task owns that asset class and this audit does not run it."
        ),
        "REAL_GENERALIZATION_CLAIM": "NOT_SUPPORTED",
        "FULL_3D_REAL_TRUTH": "NOT_SUPPORTED",
        "REAL_CENSORING_HOLDOUT_RUN": False,
        "ADAPTERS_IMPLEMENTED": False,
    }


#: Top-level entries under data/external that are not data assets for M0.
DEFAULT_EXCLUSIONS = {
    "data/external/CNKI搜索到的相关论文.md": "文献题录，不是数据资产",
    "data/external/EFRACTURE_DATA.md": "内部数据操作手册，不是数据资产",
    "data/external/cnki_papers": "知网论文全文/摘录，版权受限，非数据资产",
    "data/external/cnki_screenshot.png": "检索截图，非数据资产",
    "data/external/cnki_search_result.png": "检索截图，非数据资产",
    "data/external/aland_traces": "空占位目录；实际迹线在 aland_islands_repo 内",
}


def _covered(rel: str, registered: set[str]) -> bool:
    rel = rel.replace("\\", "/").rstrip("/")
    for item in registered:
        item = item.replace("\\", "/").rstrip("/")
        if rel == item or rel.startswith(item + "/") or item.startswith(rel + "/"):
            return True
    return False


def check_coverage(
    root: str | os.PathLike,
    rows: Sequence[Mapping[str, Any]],
    *,
    roots: Sequence[str] = ("data/real", "data/synth", "data/external"),
    exclusions: Mapping[str, str] | None = None,
) -> dict:
    """Scan top-level entries of the M0 scope and report registered/excluded gaps."""
    base = Path(root)
    excluded = dict(DEFAULT_EXCLUSIONS if exclusions is None else exclusions)
    registered: set[str] = set()
    for row in rows:
        for key in ("path", "path_extra"):
            values = row.get(key) if key == "path_extra" else [row.get(key)]
            for value in values or []:
                if value:
                    registered.add(str(value))

    covered: list[str] = []
    excluded_hits: list[dict] = []
    unregistered: list[str] = []
    n_entries = 0
    for rel_root in roots:
        directory = base / rel_root
        if not directory.exists():
            continue
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            rel = (Path(rel_root) / entry.name).as_posix()
            n_entries += 1
            if rel in excluded:
                excluded_hits.append({"path": rel, "reason": excluded[rel]})
            elif _covered(rel, registered):
                covered.append(rel)
            else:
                unregistered.append(rel)
    return {
        "schema": SCHEMA_COVERAGE,
        "scope_roots": list(roots),
        "scan_depth": 1,
        "n_top_level_entries": n_entries,
        "n_covered": len(covered),
        "n_excluded": len(excluded_hits),
        "n_unregistered": len(unregistered),
        "unregistered": unregistered,
        "excluded": excluded_hits,
        "notes": (
            "Scan depth=1 over data/real, data/synth and data/external. The E: junction "
            "(data/external/efracture_data) is one scope entry covered by its registered sub-assets; "
            "scripts, virtual environments, caches and scratch directories inside it are not data assets "
            "and are not enumerated. Known staging/code/report entries not registered as M0 assets: "
            "forge_dfns/, r193_staging/, pySimFrac/, pySimFrac.zip, pySimFrac_extract/, "
            "downloads/02_USGS_Fracture_Traces/, downloads/05_Synthetic_DFN/. "
            "Registered/derived copies are recorded in dedupe_status."
        ),
    }


def audit_status(rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any] | None = None) -> str:
    rows = list(rows)
    if not rows:
        return "BLOCKED_BY_DATA"
    if validate_registry(rows):
        return "REGISTRY_AUDIT_PARTIAL"
    if any(r.get("path") and r.get("exists_on_disk") is False for r in rows):
        return "REGISTRY_AUDIT_PARTIAL"
    if coverage is not None and coverage.get("n_unregistered", 0) > 0:
        return "REGISTRY_AUDIT_PARTIAL"
    return "REGISTRY_AUDIT_COMPLETE"


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------
def canonical_json(document: Any) -> str:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(document: Any) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def write_json(path: str | os.PathLike, document: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")
    os.replace(tmp, target)
    return str(target)


def write_jsonl(path: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
    os.replace(tmp, target)
    return str(target)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry(path: str | os.PathLike) -> list[dict]:
    document = read_json(path)
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        rows = document.get("rows", [])
    else:
        raise TypeError("registry file must contain a list or an object with a rows list")
    return [dict(row) for row in rows]


def content_payload(rows: Sequence[Mapping[str, Any]]) -> dict:
    return {"schema": SCHEMA, "rows": [dict(r) for r in rows]}


def content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Snapshot hash (includes probe timestamps and on-disk facts)."""
    return canonical_sha256(content_payload(rows))


def catalogue_payload() -> dict:
    """Static catalogue payload; no probes, no timestamps, stable across runs."""
    return {"schema": SCHEMA, "rows": [copy.deepcopy(dict(r)) for r in CATALOGUE]}


def catalogue_sha256() -> str:
    """Stable hash of the static catalogue (what was declared, not when probed)."""
    return canonical_sha256(catalogue_payload())


__all__ = [
    "SCHEMA",
    "SCHEMA_SUMMARY",
    "SCHEMA_VERDICT",
    "SCHEMA_COVERAGE",
    "WATERMARK",
    "UNKNOWN",
    "TRUTH_LEVELS",
    "TRUTH_STATUSES",
    "LICENSE_STATUSES",
    "LICENSE_UNKNOWN_STATUSES",
    "LICENSE_NONPERMISSIVE_STATUSES",
    "PUBLIC_PRIVATE",
    "INDEPENDENT_UNITS",
    "MODALITIES",
    "REQUIRED_FIELDS",
    "CATALOGUE",
    "DEFAULT_EXCLUSIONS",
    "build_registry",
    "validate_registry",
    "summarize",
    "check_coverage",
    "audit_status",
    "probe_path",
    "file_sha256",
    "write_json",
    "write_jsonl",
    "read_json",
    "load_registry",
    "canonical_json",
    "canonical_sha256",
    "content_payload",
    "content_sha256",
    "catalogue_payload",
    "catalogue_sha256",
]
