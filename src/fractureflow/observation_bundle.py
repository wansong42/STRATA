# -*- coding: utf-8 -*-
"""G60 · PublicObservationBundle —— E0 公开观测隔离视图（零研究读数）。

任务书: results/architect_main_20261012_dispatch/p18_20261021/
        task_order_G60_observation_isolation_20261021.md
水印  : CONTRACT_INFRA / E0_LEAK_TEST

本模块只做一件事: 把 obs_forward.Observation 转成**只含已观测数值**的公开视图,
并用四道守卫挡住 E0 的四条旁路（字段 / 对象引用 / 序列化 / 数值）:

  * validate_no_hidden_fields(): 递归扫描属性名 / dict 键 / ndarray 载荷 / 来源标记,
    命中 FORBIDDEN_KEYS 或 *_seed / truth_* / hidden_* 等来源模式即抛 HiddenFieldError;
  * to_json() / from_json(): 递归键扫描 + 隐藏键抛错 + 往返等价;
  * truth_unreachability_audit(hidden): 与隐藏真值逐值比对, 返回 value_leaks /
    seed_recoverable / alias_refs;
  * identity_guard(): public_id 必须由公开数值派生（哈希序位）, 不得由 frac_id /
    DFN 下标直接生成。

纪律: 纯标准库 + numpy; 全部 JSON allow_nan=False（NaN 一律编码为 JSON null）;
本模块不 import obs_forward, 不引用隐藏真值——隐藏侧只作为 truth_unreachability_audit
的显式入参, 生产路径（from_observation / to_json / validate）不触碰它。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

WATERMARK = "CONTRACT_INFRA / E0_LEAK_TEST"
SCHEMA_JSON = "g60.public_observation_bundle.v1"

# 任务书 §4.2 硬编码禁键清单（逐字照抄，禁止增删本集合本身）。
FORBIDDEN_KEYS = frozenset({
    "truth_sets", "truth_id", "true_normal", "hidden_truth", "hidden_objects",
    "full_network", "oracle_label", "ground_truth_network", "dfn_seed",
    "world_seed", "true_sets", "frac_ids",
})

# 身份键: 任务书 §7.5 要求"记录条目保留 frac_id ⇒ 拦截红", 故与 FORBIDDEN_KEYS
# 分开登记（不污染 §4.2 的硬编码清单），扫描时同样抛 HiddenFieldError。
IDENTITY_KEYS = frozenset({"frac_id", "frac_index", "dfn_index", "disc_id"})

_SOURCE_PREFIXES = ("truth_", "hidden_", "oracle_", "ground_truth", "full_network")
_SOURCE_SUFFIXES = ("_seed",)

PUBLIC_FIELD_NAMES = (
    "modality", "raw_reference", "coordinate_frame", "resolution",
    "values", "geometry", "visibility_mask", "missing_reason",
    "noise_spec", "action_spec", "timestamp", "provenance_public",
)

# 任务书 §4.1: modality ∈ 六类, 由 operator 映射（映射表在 prereg 登记）。
class Modality(str, Enum):
    BOREHOLE_FMI = "borehole_fmi"
    CORE_PHOTO = "core_photo"
    CT = "ct"
    POINT_CLOUD = "point_cloud"
    MAP = "map"
    SYNTHETIC = "synthetic"


MODALITY_BY_OPERATOR = {
    "virtual_borehole": Modality.BOREHOLE_FMI,
    "virtual_borehole_multi": Modality.BOREHOLE_FMI,
    "virtual_fmi": Modality.BOREHOLE_FMI,
    "virtual_outcrop_scan": Modality.POINT_CLOUD,
    "synthetic_forward": Modality.SYNTHETIC,
}


class MissingReason(str, Enum):
    # 任务书 §4.1 / 技术书 §4.3 层 A 的四类缺测 + 已观测。
    NOT_OBSERVED = "NOT_OBSERVED"
    OBSERVED_NEGATIVE = "OBSERVED_NEGATIVE"
    OCCLUDED = "OCCLUDED"
    DEVICE_FAILED = "DEVICE_FAILED"
    NONE = "NONE"


MISSING_REASON_CODE = {
    MissingReason.NONE: 0,
    MissingReason.NOT_OBSERVED: 1,
    MissingReason.OBSERVED_NEGATIVE: 2,
    MissingReason.OCCLUDED: 3,
    MissingReason.DEVICE_FAILED: 4,
}
CODE_TO_MISSING_REASON = {v: k for k, v in MISSING_REASON_CODE.items()}
MISSING_REASON_CODE_TO_NAME = {v: k.value for k, v in MISSING_REASON_CODE.items()}

_VALUES_KEYS = frozenset({"normals", "positions", "depth", "public_ids"})
_GEOMETRY_KEYS = frozenset({"array_length", "n_observed", "sensor"})
_PROVENANCE_ALLOWED = ("operator", "params_public", "version", "timestamp")
_NOISE_KEYS = frozenset({"model", "sigma_deg"})
_PARAM_WHITELIST = (
    "x0", "d", "well_center", "well_axis", "well_radius",
    "scan_z", "bbox_xy", "density", "roughness_deg",
)
_ALLOWED_ATTRS = PUBLIC_FIELD_NAMES + ("_public_cache",)
_PROVENANCE_PUBLIC_KEYS = frozenset(_PROVENANCE_ALLOWED)


class ObservationBundleError(Exception):
    """公开观测包的通用错误（结构非法 / 缺测掩码冲突 / JSON 结构非法等）。"""


class HiddenFieldError(ObservationBundleError):
    """公开视图中出现隐藏真值键 / 来源标记。"""

    def __init__(self, message: str, violations: Optional[list] = None):
        super().__init__(message)
        self.violations = list(violations or [])


class IdentityGuardError(HiddenFieldError):
    """public_id 由 frac_id / DFN 下标直接生成（身份旁路）。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _norm_key(key: Any) -> str:
    return str(key).strip().lower()


def key_violation(key: Any) -> Optional[str]:
    """返回 'FORBIDDEN' / 'IDENTITY' / None。来源模式按任务书 §4.1 剥除规则。"""
    k = _norm_key(key)
    if k in FORBIDDEN_KEYS:
        return "FORBIDDEN"
    if k in IDENTITY_KEYS:
        return "IDENTITY"
    for suf in _SOURCE_SUFFIXES:
        if k.endswith(suf):
            return "FORBIDDEN"
    for pre in _SOURCE_PREFIXES:
        if k.startswith(pre):
            return "FORBIDDEN"
    return None


def _jsonable(obj: Any) -> Any:
    """转成可 allow_nan=False 序列化的纯 JSON 值; 非有限浮点直接拒绝。"""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            raise ObservationBundleError("非有限浮点 %r 不能进入公开 JSON" % (obj,))
        return v
    if isinstance(obj, np.ndarray):
        raise ObservationBundleError("ndarray 不应进入 _jsonable; 用 _array_to_json")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    raise ObservationBundleError("公开视图不支持的类型: %s" % type(obj).__name__)


def _fs(x: float) -> Optional[float]:
    v = float(x)
    return None if math.isnan(v) else v


def _array_to_json(arr: np.ndarray) -> list:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return [_fs(x) for x in a]
    return [[_fs(x) for x in row] for row in a]


def _array_from_json(rows: Any) -> np.ndarray:
    try:
        a = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ObservationBundleError("公开数组反序列化失败: %s" % exc)
    if a.ndim not in (1, 2):
        raise ObservationBundleError("公开数组维度非法: %r" % (a.shape,))
    return a


def _unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def scan_document_keys(doc: Any, *, path: str = "$", _depth: int = 0) -> list:
    """递归扫描 JSON 文档的键名; 返回 violation 列表（不抛错）。"""
    out: list = []
    if _depth > 16:
        return out
    if isinstance(doc, dict):
        for k, v in doc.items():
            kn = _norm_key(k)
            kind = key_violation(k)
            if kind:
                out.append({"path": "%s.%s" % (path, kn), "key": str(k), "kind": kind})
            out.extend(scan_document_keys(v, path="%s.%s" % (path, kn), _depth=_depth + 1))
    elif isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            out.extend(scan_document_keys(v, path="%s[%d]" % (path, i), _depth=_depth + 1))
    return out


def _scan_object(self_obj: Any, *, path: str = "$", _depth: int = 0) -> list:
    """递归扫描对象/属性/字典/ndarray 载荷的键名与来源标记。"""
    out: list = []
    if _depth > 16:
        return out
    if self_obj is None or isinstance(self_obj, (bool, int, float, str, bytes)):
        return out
    if isinstance(self_obj, Enum) or isinstance(self_obj, np.generic):
        return out
    if isinstance(self_obj, np.ndarray):
        if self_obj.dtype.kind == "O":
            for i, item in enumerate(self_obj.reshape(-1)):
                out.extend(_scan_object(item, path="%s[%d]" % (path, i), _depth=_depth + 1))
        return out
    if isinstance(self_obj, dict):
        for k, v in self_obj.items():
            kn = _norm_key(k)
            kind = key_violation(k)
            if kind:
                out.append({"path": "%s.%s" % (path, kn), "key": str(k), "kind": kind})
            out.extend(_scan_object(v, path="%s.%s" % (path, kn), _depth=_depth + 1))
        return out
    if isinstance(self_obj, (list, tuple)):
        for i, v in enumerate(self_obj):
            out.extend(_scan_object(v, path="%s[%d]" % (path, i), _depth=_depth + 1))
        return out
    if hasattr(self_obj, "__dict__"):
        for k, v in vars(self_obj).items():
            kn = _norm_key(k)
            kind = key_violation(k)
            if kind:
                out.append({"path": "%s.%s" % (path, kn), "key": str(k), "kind": kind})
            out.extend(_scan_object(v, path="%s.%s" % (path, kn), _depth=_depth + 1))
        return out
    return out


def _entry_public_key(normal: np.ndarray, position: np.ndarray, depth: float) -> str:
    """位序键: 只由公开数值决定（法向 / 位置 / 深度）, 与 DFN 下标无关。"""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(normal, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(position, dtype=np.float64).tobytes())
    h.update(struct.pack("<d", float(depth)))
    return h.hexdigest()


def expected_public_ids(normals: np.ndarray, positions: np.ndarray, depth: np.ndarray,
                        observed_index: np.ndarray) -> List[str]:
    """严格由公开数值派生不透明 public_id（p000…）, 位序按 sha256 字典序。"""
    observed_index = np.asarray(observed_index, dtype=int)
    keys = []
    for gi in observed_index:
        keys.append(_entry_public_key(normals[gi], positions[gi], depth[gi]))
    order = sorted(range(len(keys)), key=lambda k: (keys[k], k))
    ids = [None] * len(keys)
    for rank, pos in enumerate(order):
        ids[pos] = "p%03d" % rank
    return ids


# ---------------------------------------------------------------------------
# 公开观测包
# ---------------------------------------------------------------------------

class PublicObservationBundle:
    """E0 公开观测视图（12 个公开字段, 任务书 §4.1 缺一即 FAIL）。"""

    FORBIDDEN_KEYS = FORBIDDEN_KEYS
    IDENTITY_KEYS = IDENTITY_KEYS
    PUBLIC_FIELD_NAMES = PUBLIC_FIELD_NAMES
    MODALITY_BY_OPERATOR = MODALITY_BY_OPERATOR
    WATERMARK = WATERMARK

    def __init__(self, modality, raw_reference, coordinate_frame, resolution,
                 values, geometry, visibility_mask, missing_reason,
                 noise_spec, action_spec, timestamp, provenance_public):
        self.modality = modality
        self.raw_reference = raw_reference
        self.coordinate_frame = coordinate_frame
        self.resolution = resolution
        self.values = values
        self.geometry = geometry
        self.visibility_mask = np.asarray(visibility_mask, dtype=bool).reshape(-1).copy()
        self.missing_reason = np.asarray(missing_reason, dtype=np.int8).reshape(-1).copy()
        self.noise_spec = noise_spec
        self.action_spec = action_spec
        self.timestamp = timestamp
        self.provenance_public = provenance_public
        self._normalise_fields()
        self._public_cache = {
            "n_entries": int(self.visibility_mask.shape[0]),
            "n_observed": int(self.visibility_mask.sum()),
            "public_id_order_sha256": hashlib.sha256(
                ",".join(self.values["public_ids"]).encode("utf-8")).hexdigest(),
        }
        self.validate_no_hidden_fields(strict=True)

    # -- 构造 ------------------------------------------------------------
    @classmethod
    def from_observation(cls, obs, *, modality=None, exposed_mask=None,
                         occluded_mask=None, device_failed_mask=None,
                         timestamp="", raw_reference="", coordinate_frame="",
                         resolution=None) -> "PublicObservationBundle":
        required = ("operator", "obs_mask", "obs_normals", "obs_positions",
                    "frac_ids", "truth_sets", "roughness_deg", "provenance", "records")
        missing = [n for n in required if not hasattr(obs, n)]
        if missing:
            raise ObservationBundleError("Observation 缺字段: %r" % (missing,))

        obs_mask = np.asarray(obs.obs_mask, dtype=bool).reshape(-1)
        M = int(obs_mask.shape[0])
        if M == 0:
            # 任务书 §7.1: 空观测（M=0）必须显式失败, 不得静默产出空包。
            raise ObservationBundleError("EMPTY_OBSERVATION: M=0（空观测禁止构造公开视图）")

        normals = np.asarray(obs.obs_normals, dtype=np.float64)
        positions = np.asarray(obs.obs_positions, dtype=np.float64)
        if normals.shape != (M, 3) or positions.shape != (M, 3):
            raise ObservationBundleError(
                "观测数组形状非法: normals=%r positions=%r M=%d" %
                (normals.shape, positions.shape, M))
        frac_ids = np.asarray(obs.frac_ids, dtype=int).reshape(-1)
        if not np.array_equal(frac_ids, np.flatnonzero(obs_mask)):
            raise ObservationBundleError("obs.frac_ids 与 obs_mask 不一致（上游语义被破坏）")
        truth_sets = np.asarray(obs.truth_sets).reshape(-1)
        if truth_sets.shape[0] != M:
            raise ObservationBundleError("obs.truth_sets 长度 %d != M=%d" % (truth_sets.shape[0], M))

        def _mask(m, name):
            if m is None:
                return np.zeros(M, dtype=bool)
            a = np.asarray(m, dtype=bool).reshape(-1)
            if a.shape != (M,):
                raise ObservationBundleError("%s 形状 %r != (M,)" % (name, a.shape))
            return a

        occluded = _mask(occluded_mask, "occluded_mask")
        device_failed = _mask(device_failed_mask, "device_failed_mask")
        exposed = _mask(exposed_mask, "exposed_mask")
        named = (("occluded_mask", occluded), ("device_failed_mask", device_failed),
                 ("exposed_mask", exposed))
        for name, arr in named:
            if np.any(arr & obs_mask):
                raise ObservationBundleError("%s 与已观测条目重叠（缺测语义冲突）" % name)
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                if np.any(named[i][1] & named[j][1]):
                    raise ObservationBundleError(
                        "%s 与 %s 重叠（缺测语义冲突）" % (named[i][0], named[j][0]))

        # values: 只保留已观测数值; 未观测位一律 NaN。
        normals_pub = np.full((M, 3), np.nan, dtype=np.float64)
        positions_pub = np.full((M, 3), np.nan, dtype=np.float64)
        depth_pub = np.full(M, np.nan, dtype=np.float64)
        normals_pub[obs_mask] = normals[obs_mask].copy()
        positions_pub[obs_mask] = positions[obs_mask].copy()

        operator = str(obs.operator)
        provenance = copy.deepcopy(dict(obs.provenance or {}))
        records = copy.deepcopy(list(obs.records or []))
        depth_key = {"virtual_borehole": "depth", "virtual_fmi": "depth_center"}.get(operator)
        if depth_key:
            depth_map = {}
            for rec in records:
                if isinstance(rec, dict) and "frac_id" in rec and depth_key in rec:
                    try:
                        depth_map[int(rec["frac_id"])] = float(rec[depth_key])
                    except (TypeError, ValueError):
                        continue
            for gi in np.flatnonzero(obs_mask):
                if int(gi) in depth_map:
                    depth_pub[gi] = depth_map[int(gi)]

        observed_index = np.flatnonzero(obs_mask)
        public_ids = expected_public_ids(normals_pub, positions_pub, depth_pub, observed_index)

        reasons = np.full(M, MISSING_REASON_CODE[MissingReason.NOT_OBSERVED], dtype=np.int8)
        reasons[exposed] = MISSING_REASON_CODE[MissingReason.OBSERVED_NEGATIVE]
        reasons[occluded] = MISSING_REASON_CODE[MissingReason.OCCLUDED]
        reasons[device_failed] = MISSING_REASON_CODE[MissingReason.DEVICE_FAILED]
        reasons[obs_mask] = MISSING_REASON_CODE[MissingReason.NONE]

        params_public = {}
        sources = [provenance]
        if isinstance(provenance.get("params"), dict):
            sources.append(provenance["params"])
        for key in _PARAM_WHITELIST:
            for src in sources:
                if key in src:
                    try:
                        params_public[key] = _jsonable(src[key])
                    except ObservationBundleError:
                        params_public.pop(key, None)
                    break
        version = provenance.get("version")
        if not isinstance(version, str) or key_violation(version):
            version = "UNKNOWN"
        provenance_public = {
            "operator": operator,
            "params_public": params_public,
            "version": version,
            "timestamp": str(timestamp),
        }

        roughness = float(obs.roughness_deg) if obs.roughness_deg is not None else 0.0
        if not math.isfinite(roughness) or roughness < 0.0:
            raise ObservationBundleError("roughness_deg 非法: %r" % (roughness,))
        noise_spec = {"model": "isotropic_angular_normal", "sigma_deg": roughness}

        if modality is None:
            mod = MODALITY_BY_OPERATOR.get(operator, Modality.SYNTHETIC)
        elif isinstance(modality, Modality):
            mod = modality
        else:
            try:
                mod = Modality(str(modality))
            except ValueError:
                raise ObservationBundleError("未知 modality: %r" % (modality,))

        geometry = {
            "array_length": M,
            "n_observed": int(obs_mask.sum()),
            "sensor": copy.deepcopy(params_public),
        }
        values = {
            "normals": normals_pub,
            "positions": positions_pub,
            "depth": depth_pub,
            "public_ids": list(public_ids),
        }

        bundle = cls(
            modality=mod,
            raw_reference=str(raw_reference),
            coordinate_frame=str(coordinate_frame),
            resolution=copy.deepcopy(resolution),
            values=values,
            geometry=geometry,
            visibility_mask=obs_mask.copy(),
            missing_reason=reasons,
            noise_spec=noise_spec,
            action_spec={},
            timestamp=str(timestamp),
            provenance_public=provenance_public,
        )
        # 内部身份守卫: 只证明 public_id 由公开数值派生（不读 frac_ids 生成）。
        bundle.identity_guard()
        return bundle

    # -- 规范化 ----------------------------------------------------------
    def _normalise_fields(self) -> None:
        if not isinstance(self.modality, Modality):
            self.modality = Modality(str(self.modality))
        self.raw_reference = "" if self.raw_reference is None else str(self.raw_reference)
        self.coordinate_frame = ("" if self.coordinate_frame is None
                                 else str(self.coordinate_frame))
        self.timestamp = "" if self.timestamp is None else str(self.timestamp)
        if not isinstance(self.values, dict):
            raise ObservationBundleError("values 必须是 dict")
        normals = np.asarray(self.values.get("normals"), dtype=np.float64)
        positions = np.asarray(self.values.get("positions"), dtype=np.float64)
        depth = np.asarray(self.values.get("depth"), dtype=np.float64).reshape(-1)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ObservationBundleError("values.normals 形状非法: %r" % (normals.shape,))
        if positions.shape != normals.shape:
            raise ObservationBundleError("values.positions 形状非法: %r" % (positions.shape,))
        M = int(normals.shape[0])
        if depth.shape != (M,):
            raise ObservationBundleError("values.depth 形状非法: %r" % (depth.shape,))
        if self.visibility_mask.shape != (M,):
            raise ObservationBundleError("visibility_mask 长度 != %d" % M)
        if self.missing_reason.shape != (M,):
            raise ObservationBundleError("missing_reason 长度 != %d" % M)
        ids = list(self.values.get("public_ids", []))
        self.values = {
            "normals": np.array(normals, dtype=np.float64, copy=True),
            "positions": np.array(positions, dtype=np.float64, copy=True),
            "depth": np.array(depth, dtype=np.float64, copy=True),
            "public_ids": [str(x) for x in ids],
        }
        self.geometry = copy.deepcopy(dict(self.geometry or {}))
        self.noise_spec = copy.deepcopy(dict(self.noise_spec or {}))
        self.action_spec = copy.deepcopy(dict(self.action_spec or {}))
        self.provenance_public = copy.deepcopy(dict(self.provenance_public or {}))

    # -- 守卫 1: 隐藏字段 / 来源标记 --------------------------------------
    def validate_no_hidden_fields(self, strict: bool = True) -> None:
        violations = _scan_object(self, path="$")
        if violations:
            forbidden = [v for v in violations if v["kind"] == "FORBIDDEN"]
            if forbidden:
                raise HiddenFieldError(
                    "公开视图出现隐藏真值键: %r" % (forbidden[:8],), violations)
            raise IdentityGuardError(
                "公开视图出现身份键（frac_id/DFN 下标）: %r" % (violations[:8],), violations)
        if strict:
            self._validate_structure()

    def _validate_structure(self) -> None:
        attrs = set(vars(self))
        extra = attrs - set(_ALLOWED_ATTRS)
        if extra:
            raise ObservationBundleError("公开视图含未声明属性: %r" % (sorted(extra),))
        absent = set(PUBLIC_FIELD_NAMES) - attrs
        if absent:
            raise ObservationBundleError("公开视图缺字段: %r" % (sorted(absent),))

        M = int(self.visibility_mask.shape[0])
        normals = self.values["normals"]
        positions = self.values["positions"]
        depth = self.values["depth"]
        mask = self.visibility_mask
        reasons = self.missing_reason
        if M == 0:
            raise ObservationBundleError("公开视图 M=0（空观测禁止）")
        if not mask.dtype == np.bool_:
            raise ObservationBundleError("visibility_mask 非 bool")
        for gi in np.flatnonzero(mask):
            if not np.all(np.isfinite(normals[gi])) or not np.all(np.isfinite(positions[gi])):
                raise ObservationBundleError("已观测条目 %d 数值非有限" % gi)
        for gi in np.flatnonzero(~mask):
            if np.any(np.isfinite(normals[gi])) or np.any(np.isfinite(positions[gi])):
                raise ObservationBundleError("未观测条目 %d 保留了真值（必须 NaN）" % gi)
            if math.isfinite(float(depth[gi])):
                raise ObservationBundleError("未观测条目 %d 的 depth 非 NaN" % gi)
            if int(reasons[gi]) == MISSING_REASON_CODE[MissingReason.NONE]:
                raise ObservationBundleError("未观测条目 %d 缺测理由为 NONE" % gi)
            if int(reasons[gi]) not in CODE_TO_MISSING_REASON:
                raise ObservationBundleError("缺测理由编码非法: %r" % int(reasons[gi]))
        for gi in np.flatnonzero(mask):
            if int(reasons[gi]) != MISSING_REASON_CODE[MissingReason.NONE]:
                raise ObservationBundleError("已观测条目 %d 缺测理由非 NONE" % gi)
        ids = list(self.values["public_ids"])
        if len(ids) != int(mask.sum()):
            raise ObservationBundleError(
                "public_ids 数 %d != 已观测数 %d" % (len(ids), int(mask.sum())))
        for k, pid in enumerate(ids):
            if not isinstance(pid, str) or len(pid) != 4 or pid[0] != "p" or not pid[1:].isdigit():
                raise IdentityGuardError("public_id 格式非法: %r （应为 p000…）" % (pid,))
        if len(set(ids)) != len(ids):
            raise IdentityGuardError("public_id 重复")
        expected = expected_public_ids(normals, positions, depth, np.flatnonzero(mask))
        if ids != expected:
            raise IdentityGuardError("public_id 不是由公开数值派生（疑似 frac_id 派生）")
        if not set(self.provenance_public) <= _PROVENANCE_PUBLIC_KEYS:
            raise ObservationBundleError(
                "provenance_public 键越界: %r" % (sorted(set(self.provenance_public) - _PROVENANCE_PUBLIC_KEYS),))
        if not isinstance(self.provenance_public.get("params_public", {}), dict):
            raise ObservationBundleError("provenance_public.params_public 非 dict")
        if set(self.values) != _VALUES_KEYS:
            raise ObservationBundleError("values 键越界: %r" % (sorted(self.values),))
        if not set(self.geometry) <= _GEOMETRY_KEYS:
            raise ObservationBundleError("geometry 键越界: %r" % (sorted(self.geometry),))
        if not isinstance(self.geometry.get("sensor", {}), dict):
            raise ObservationBundleError("geometry.sensor 非 dict")
        if set(self.noise_spec) != _NOISE_KEYS:
            raise ObservationBundleError("noise_spec 键越界: %r" % (sorted(self.noise_spec),))
        sigma = self.noise_spec.get("sigma_deg")
        if not isinstance(sigma, float) or not math.isfinite(sigma) or sigma < 0.0:
            raise ObservationBundleError("noise_spec.sigma_deg 非法: %r" % (sigma,))
        if not isinstance(self.action_spec, dict):
            raise ObservationBundleError("action_spec 非 dict")
        _jsonable(self.resolution)

    # -- 守卫 2: 身份 ----------------------------------------------------
    def identity_guard(self, frac_ids_observed: Optional[Sequence[int]] = None) -> dict:
        """public_id 只能由公开数值派生; 给出隐藏 frac_ids 时额外做不可反推检查。"""
        mask = self.visibility_mask
        normals = self.values["normals"]
        positions = self.values["positions"]
        depth = self.values["depth"]
        ids = list(self.values["public_ids"])
        expected = expected_public_ids(normals, positions, depth, np.flatnonzero(mask))
        checks = {
            "format_ok": all(isinstance(x, str) and len(x) == 4
                             and x[0] == "p" and x[1:].isdigit() for x in ids),
            "unique_ok": len(set(ids)) == len(ids),
            "public_derivation_ok": ids == expected,
        }
        if frac_ids_observed is not None:
            frac_ids = np.asarray(frac_ids_observed, dtype=int).reshape(-1)
            if frac_ids.shape[0] != len(ids):
                raise IdentityGuardError("frac_ids 长度与 public_ids 不等")
            # 任务书 §7.12: 用 frac_id 生成 public_id 的签名 = 逐位等于 p%03d % frac_id;
            # n_obs<=1 时两种构造天然同形, 记 INDISTINGUISHABLE 而非 leak。
            frac_derived = ["p%03d" % int(x) for x in frac_ids]
            same = (ids == frac_derived)
            checks["n_obs"] = int(len(ids))
            checks["frac_derived_signature"] = bool(same)
            checks["indistinguishable"] = bool(len(ids) <= 1)
            checks["not_frac_id_derived"] = (not same) or (len(ids) <= 1)
            if same and len(ids) >= 2:
                raise IdentityGuardError(
                    "public_id 逐位等于 p%%03d %% frac_id（身份旁路）: %r" % (ids[:5],))
        ok = checks["format_ok"] and checks["unique_ok"] and checks["public_derivation_ok"]
        if not ok:
            raise IdentityGuardError("public_id 身份守卫失败: %r" % (checks,))
        return {"ok": True, "checks": checks}

    # -- 守卫 3: 序列化 --------------------------------------------------
    def to_json(self) -> dict:
        self.validate_no_hidden_fields(strict=True)
        doc = {
            "schema": SCHEMA_JSON,
            "modality": self.modality.value,
            "raw_reference": self.raw_reference,
            "coordinate_frame": self.coordinate_frame,
            "resolution": _jsonable(self.resolution),
            "values": {
                "normals": _array_to_json(self.values["normals"]),
                "positions": _array_to_json(self.values["positions"]),
                "depth": _array_to_json(self.values["depth"]),
                "public_ids": list(self.values["public_ids"]),
            },
            "geometry": _jsonable(self.geometry),
            "visibility_mask": [bool(x) for x in self.visibility_mask],
            "missing_reason": [MISSING_REASON_CODE_TO_NAME[int(c)]
                               for c in self.missing_reason],
            "noise_spec": _jsonable(self.noise_spec),
            "action_spec": _jsonable(self.action_spec),
            "timestamp": self.timestamp,
            "provenance_public": _jsonable(self.provenance_public),
        }
        violations = scan_document_keys(doc)
        if violations:
            raise HiddenFieldError("to_json 文档含隐藏键: %r" % (violations[:8],), violations)
        return doc

    @classmethod
    def from_json(cls, doc: dict) -> "PublicObservationBundle":
        if not isinstance(doc, dict):
            raise ObservationBundleError("from_json 需要 dict, 收到 %s" % type(doc).__name__)
        violations = scan_document_keys(doc)
        if violations:
            forbidden = [v for v in violations if v["kind"] == "FORBIDDEN"]
            if forbidden:
                raise HiddenFieldError(
                    "from_json 拒绝隐藏键（禁止静默丢弃）: %r" % (forbidden[:8],), violations)
            raise IdentityGuardError(
                "from_json 拒绝身份键（禁止静默丢弃）: %r" % (violations[:8],), violations)
        if doc.get("schema") != SCHEMA_JSON:
            raise ObservationBundleError("JSON schema 非法: %r" % (doc.get("schema"),))
        allowed_top = {"schema"} | set(PUBLIC_FIELD_NAMES)
        unknown = set(doc) - allowed_top
        if unknown:
            raise ObservationBundleError("from_json 顶层未知键（禁止静默丢弃）: %r" % (sorted(unknown),))
        absent = set(PUBLIC_FIELD_NAMES) - set(doc)
        if absent:
            raise ObservationBundleError("from_json 缺字段: %r" % (sorted(absent),))
        values_doc = doc["values"]
        if not isinstance(values_doc, dict):
            raise ObservationBundleError("values 非 dict")
        unknown_v = set(values_doc) - _VALUES_KEYS
        if unknown_v:
            raise ObservationBundleError("values 未知键: %r" % (sorted(unknown_v),))
        reasons = []
        for name in doc["missing_reason"]:
            try:
                reasons.append(MISSING_REASON_CODE[MissingReason(str(name))])
            except ValueError:
                raise ObservationBundleError("missing_reason 非法值: %r" % (name,))
        mask = np.asarray(doc["visibility_mask"], dtype=bool).reshape(-1)
        try:
            mod = Modality(str(doc["modality"]))
        except ValueError:
            raise ObservationBundleError("modality 非法: %r" % (doc["modality"],))
        bundle = cls(
            modality=mod,
            raw_reference=doc["raw_reference"],
            coordinate_frame=doc["coordinate_frame"],
            resolution=doc["resolution"],
            values={
                "normals": _array_from_json(values_doc["normals"]),
                "positions": _array_from_json(values_doc["positions"]),
                "depth": _array_from_json(values_doc["depth"]),
                "public_ids": list(values_doc["public_ids"]),
            },
            geometry=doc["geometry"],
            visibility_mask=mask,
            missing_reason=np.asarray(reasons, dtype=np.int8),
            noise_spec=doc["noise_spec"],
            action_spec=doc["action_spec"],
            timestamp=doc["timestamp"],
            provenance_public=doc["provenance_public"],
        )
        return bundle

    # -- 守卫 4: 数值不可达 ----------------------------------------------
    def truth_unreachability_audit(self, hidden) -> dict:
        """数值旁路审计: 公开数组 vs 隐藏真值（truth_sets / 未观测真法向 / seed 派生量）。"""
        h = hidden
        if not isinstance(h, dict):
            h = {
                "obs_normals": np.asarray(h.obs_normals, dtype=np.float64),
                "obs_positions": np.asarray(h.obs_positions, dtype=np.float64),
                "truth_sets": np.asarray(h.truth_sets).reshape(-1),
                "frac_ids": np.asarray(h.frac_ids, dtype=int).reshape(-1),
                "obs_mask": np.asarray(h.obs_mask, dtype=bool).reshape(-1),
                "provenance": dict(h.provenance or {}),
            }
        mask = self.visibility_mask
        unobs = ~mask
        normals = self.values["normals"]
        positions = self.values["positions"]
        depth = self.values["depth"]
        truth_sets = np.asarray(h["truth_sets"]).reshape(-1)
        hidden_normals = np.asarray(h["obs_normals"], dtype=np.float64)
        hidden_positions = np.asarray(h["obs_positions"], dtype=np.float64)
        leaks: list = []
        # 逐值旁路口径: "public 数组整体等于 truth_sets" 才算泄漏（NaN 使未观测位不可复现）。
        # 单元素恰好等于某个 set 标签（如 z=0.0 与 set 0）不作泄漏, 否则成为恒真谓词。
        eq_truth_sets = 0
        ts_arrays_equal = []
        if truth_sets.shape[0] == mask.shape[0]:
            for name, arr in (("values.normals", normals),
                              ("values.positions", positions),
                              ("values.depth", depth)):
                if arr.ndim == 1:
                    same = bool(arr.shape == truth_sets.shape and np.array_equal(arr, truth_sets))
                else:
                    same = bool(np.array_equal(arr, np.repeat(truth_sets[:, None], 3, axis=1)))
                if same:
                    eq_truth_sets += 1
                    ts_arrays_equal.append(name)
            if eq_truth_sets:
                leaks.append({"kind": "equals_truth_sets", "arrays": ts_arrays_equal})
        finite_unobs_normals = int(np.sum(np.all(np.isfinite(normals[unobs]), axis=1)))
        finite_unobs_positions = int(np.sum(np.all(np.isfinite(positions[unobs]), axis=1)))
        if finite_unobs_normals:
            leaks.append({"kind": "finite_unobserved_normals",
                          "n_entries": finite_unobs_normals})
        if finite_unobs_positions:
            leaks.append({"kind": "finite_unobserved_positions",
                          "n_entries": finite_unobs_positions})
        if finite_unobs_normals:
            eq_true_normal = int(np.sum(np.all(
                normals[unobs] == hidden_normals[unobs], axis=1)))
            if eq_true_normal:
                leaks.append({"kind": "equals_unobserved_true_normal",
                              "n_entries": eq_true_normal})
        if finite_unobs_positions:
            eq_true_pos = int(np.sum(np.all(
                positions[unobs] == hidden_positions[unobs], axis=1)))
            if eq_true_pos:
                leaks.append({"kind": "equals_unobserved_true_position",
                              "n_entries": eq_true_pos})

        def _json_scan_numbers(obj, out, path="$"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    out.append((str(k), v))
                    _json_scan_numbers(v, out, "%s.%s" % (path, k))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _json_scan_numbers(v, out, "%s[%d]" % (path, i))
            return out

        try:
            doc = self.to_json()
        except ObservationBundleError:
            # 脏包（控制面毒丸）可能连 to_json 都过不了; 数值旁路仍必须可审计。
            doc = {}
        seed_hits = [k for k, _ in _json_scan_numbers(doc, []) if "seed" in str(k).lower()]
        hidden_seed = None
        prov = h["provenance"]
        for key in ("dfn_seed", "world_seed"):
            if key in prov and isinstance(prov[key], (int, np.integer)):
                hidden_seed = int(prov[key])
                break
        numeric_attempts = []
        if hidden_seed is not None:
            for key, val in _json_scan_numbers(doc, []):
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    numeric_attempts.append(
                        {"key": key, "value": float(val),
                         "equals_seed": (int(val) == hidden_seed)})
                elif isinstance(val, str):
                    numeric_attempts.append(
                        {"key": key, "value": val,
                         "equals_seed": (val == str(hidden_seed))})
        num_hits = [a for a in numeric_attempts if a["equals_seed"]]
        seed_recoverable = bool(seed_hits or num_hits)
        seed_attempts = {
            "scan_seed_key_pattern": {"hits": seed_hits,
                                      "keys_scanned": len(_json_scan_numbers(doc, []))},
            "scalar_equals_seed": {"n_scalars": len(numeric_attempts),
                                   "hits": num_hits[:10], "n_hits": len(num_hits)},
            "hidden_seed_present_in_hidden_side": hidden_seed is not None,
            "public_id_numeric_check": [pid for pid in self.values["public_ids"]
                                        if hidden_seed is not None
                                        and pid[1:].lstrip("0") == str(hidden_seed)][:3],
        }
        if hidden_seed is not None and seed_attempts["public_id_numeric_check"]:
            seed_recoverable = True

        alias_refs = []
        hid_arrays = {"obs_normals": hidden_normals, "obs_positions": hidden_positions,
                      "truth_sets": truth_sets}
        for key in ("frac_ids", "obs_mask"):
            if key in h:
                hid_arrays[key] = np.asarray(h[key])
        pub_arrays = {"values.normals": normals, "values.positions": positions,
                      "values.depth": depth, "visibility_mask": mask,
                      "missing_reason": self.missing_reason}
        for pname, parr in pub_arrays.items():
            for hname, harr in hid_arrays.items():
                try:
                    if np.shares_memory(parr, harr):
                        alias_refs.append({"public": pname, "hidden": hname,
                                           "kind": "shares_memory"})
                except (TypeError, ValueError):
                    continue
        if self.provenance_public is prov:
            alias_refs.append({"public": "provenance_public", "hidden": "provenance",
                               "kind": "is"})
        return {
            "value_leaks": leaks,
            "n_value_leaks": len(leaks),
            "seed_recoverable": bool(seed_recoverable),
            "seed_attempts": seed_attempts,
            "alias_refs": alias_refs,
            "n_alias_refs": len(alias_refs),
            "checked": {
                "n_entries": int(mask.shape[0]),
                "n_observed": int(mask.sum()),
                "n_unobserved": int(unobs.sum()),
                "truth_sets_equal_entries": eq_truth_sets,
                "finite_unobserved_normals": finite_unobs_normals,
                "finite_unobserved_positions": finite_unobs_positions,
            },
            "world_regenerable": False,
            "world_regeneration_note": (
                "公开视图不含离散半径/中心/未观测真法向, 未观测位一律 NaN, "
                "seed 不可恢复; 无任何可重生成 DFN 的自由度。"),
        }

    # -- 等价 ------------------------------------------------------------
    def state_projection(self) -> dict:
        """用于往返等价比较的规范化投影（不写盘, 不含隐藏键）。"""
        return {
            "modality": self.modality.value,
            "raw_reference": self.raw_reference,
            "coordinate_frame": self.coordinate_frame,
            "resolution": _jsonable(self.resolution),
            "normals": _array_to_json(self.values["normals"]),
            "positions": _array_to_json(self.values["positions"]),
            "depth": _array_to_json(self.values["depth"]),
            "public_ids": list(self.values["public_ids"]),
            "geometry": _jsonable(self.geometry),
            "visibility_mask": [bool(x) for x in self.visibility_mask],
            "missing_reason": [MISSING_REASON_CODE_TO_NAME[int(c)]
                               for c in self.missing_reason],
            "noise_spec": _jsonable(self.noise_spec),
            "action_spec": _jsonable(self.action_spec),
            "timestamp": self.timestamp,
            "provenance_public": _jsonable(self.provenance_public),
        }

    def equivalent(self, other: "PublicObservationBundle") -> bool:
        if not isinstance(other, PublicObservationBundle):
            return False
        return self.state_projection() == other.state_projection()


# ---------------------------------------------------------------------------
# 便捷函数（witness CLI / 控制面复用）
# ---------------------------------------------------------------------------

def scan_for_hidden(obj) -> list:
    """对任意对象（bundle / dict / 数组）做隐藏键扫描; 返回 violation 列表。"""
    return _scan_object(obj, path="$")


def json_document_has_hidden_keys(doc: dict) -> list:
    return scan_document_keys(doc)


__all__ = [
    "WATERMARK", "SCHEMA_JSON", "FORBIDDEN_KEYS", "IDENTITY_KEYS",
    "PUBLIC_FIELD_NAMES", "MODALITY_BY_OPERATOR", "Modality", "MissingReason",
    "MISSING_REASON_CODE", "CODE_TO_MISSING_REASON",
    "ObservationBundleError", "HiddenFieldError", "IdentityGuardError",
    "PublicObservationBundle", "expected_public_ids", "key_violation",
    "scan_document_keys", "scan_for_hidden", "json_document_has_hidden_keys",
]
