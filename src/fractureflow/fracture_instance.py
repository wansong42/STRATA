# -*- coding: utf-8 -*-
"""FractureInstance 契约 v1（R183-T0）—— 合成真值与未来真实估计共用的唯一实例格式。

外部方案 V1.0 第四章定版字段（R183 任务书 §1 照抄，不发明）：
  id          迹线段级全局唯一
  plane       (n, d)：单位法向 + 平面偏移，n·x = d（米制场景系）
  bingham     (M, Z) 占位（R185 回填）
  extent      3D 范围 + 迹线段集合（段级质量标记）
  quality     内点率 / 残差 / 质量门状态
  provenance  来源模块 / 方法版本 / 时间戳
  schema_ver  契约版本

纪律：
  - 序列化一律 allow_nan=False（项目 NaN 守卫双规）；
  - 合成真值（synthetic_engine）与真实估计（未来 R182/CVAT、SAM 管线）共用本格式与同一校验器，禁双标；
  - 角度单位度、长度单位米、像素坐标 (col,row) 左上原点。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

SCHEMA_VER = "fracture_instance_v1"

SEGMENT_MARKS = ("clean", "broken", "eroded", "occluded", "merged", "grazing_excluded")
GATE_STATUSES = ("pass", "warn", "fail", "unrated")


class ContractError(ValueError):
    """契约校验失败。"""


def _fin(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ContractError(f"non-finite {name}: {x!r}")
    return v


@dataclass
class Plane:
    """n·x = d；n 单位法向（无向约定，符号不定），d 单位米。"""

    n: list
    d: float

    def to_dict(self) -> dict:
        return {"n": [float(x) for x in self.n], "d": _fin(self.d, "plane.d")}

    @staticmethod
    def from_dict(obj: dict) -> "Plane":
        n = [float(x) for x in obj["n"]]
        if len(n) != 3:
            raise ContractError("plane.n must have 3 components")
        return Plane(n=n, d=float(obj["d"]))


@dataclass
class SegmentQuality:
    inlier_rate: float
    residual_m: float
    gate: str = "unrated"

    def to_dict(self) -> dict:
        return {
            "inlier_rate": _fin(self.inlier_rate, "seg.inlier_rate"),
            "residual_m": _fin(self.residual_m, "seg.residual_m"),
            "gate": str(self.gate),
        }

    @staticmethod
    def from_dict(obj: dict) -> "SegmentQuality":
        return SegmentQuality(
            inlier_rate=float(obj["inlier_rate"]),
            residual_m=float(obj["residual_m"]),
            gate=str(obj.get("gate", "unrated")),
        )


@dataclass
class TraceSegment:
    """一段迹线。pts_m 为 3D 表面点（米制场景系）；pts_px 为投影像素坐标（可选）。"""

    seg_id: str
    pts_m: Optional[list] = None  # (M,3) GT 必填；OBS（像素系记录）可省
    pts_px: Optional[list] = None  # (M,2)
    width_px: float = 1.0
    mark: str = "clean"
    quality: SegmentQuality = field(default_factory=lambda: SegmentQuality(1.0, 0.0))

    def to_dict(self) -> dict:
        obj = {
            "seg_id": str(self.seg_id),
            "width_px": _fin(self.width_px, "seg.width_px"),
            "mark": str(self.mark),
            "quality": self.quality.to_dict(),
        }
        if self.pts_m is not None:
            obj["pts_m"] = [[float(c) for c in p] for p in self.pts_m]
        if self.pts_px is not None:
            obj["pts_px"] = [[float(c) for c in p] for p in self.pts_px]
        return obj

    @staticmethod
    def from_dict(obj: dict) -> "TraceSegment":
        if obj["mark"] not in SEGMENT_MARKS:
            raise ContractError(f"unknown segment mark: {obj['mark']}")
        return TraceSegment(
            seg_id=str(obj["seg_id"]),
            pts_m=[[float(c) for c in p] for p in obj["pts_m"]] if obj.get("pts_m") else None,
            pts_px=[[float(c) for c in p] for p in obj["pts_px"]] if obj.get("pts_px") else None,
            width_px=float(obj.get("width_px", 1.0)),
            mark=str(obj["mark"]),
            quality=SegmentQuality.from_dict(obj.get("quality", {})),
        )


@dataclass
class Extent:
    """3D 范围 + 迹线段集合。"""

    bbox_min: list
    bbox_max: list
    segments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bbox_min": [float(x) for x in self.bbox_min],
            "bbox_max": [float(x) for x in self.bbox_max],
            "segments": [s.to_dict() for s in self.segments],
        }

    @staticmethod
    def from_dict(obj: dict) -> "Extent":
        return Extent(
            bbox_min=[float(x) for x in obj["bbox_min"]],
            bbox_max=[float(x) for x in obj["bbox_max"]],
            segments=[TraceSegment.from_dict(s) for s in obj.get("segments", [])],
        )


@dataclass
class BinghamPlaceholder:
    """R185 回填位：M (3,3)、Z (3,)；v0 恒占位。"""

    M: Optional[list] = None
    Z: Optional[list] = None
    status: str = "placeholder_r185_backfill"

    def to_dict(self) -> dict:
        return {"M": self.M, "Z": self.Z, "status": self.status}

    @staticmethod
    def from_dict(obj: dict) -> "BinghamPlaceholder":
        return BinghamPlaceholder(
            M=obj.get("M"), Z=obj.get("Z"), status=str(obj.get("status", "placeholder_r185_backfill"))
        )


@dataclass
class Provenance:
    source: str
    method: str
    method_version: str
    created_at: str
    scene_id: Optional[str] = None
    seed: Optional[int] = None
    params_hash: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "method": str(self.method),
            "method_version": str(self.method_version),
            "created_at": str(self.created_at),
            "scene_id": self.scene_id,
            "seed": self.seed,
            "params_hash": self.params_hash,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(obj: dict) -> "Provenance":
        return Provenance(
            source=str(obj["source"]),
            method=str(obj["method"]),
            method_version=str(obj["method_version"]),
            created_at=str(obj["created_at"]),
            scene_id=obj.get("scene_id"),
            seed=int(obj["seed"]) if obj.get("seed") is not None else None,
            params_hash=obj.get("params_hash"),
            notes=obj.get("notes"),
        )


@dataclass
class FractureInstance:
    """单条裂缝实例（GT 粒度 = 裂缝；OBS 粒度 = 迹线段，见 fracture_instance_v1.yaml）。"""

    id: str
    plane: Plane
    extent: Extent
    quality: SegmentQuality
    provenance: Provenance
    bingham: BinghamPlaceholder = field(default_factory=BinghamPlaceholder)
    schema_ver: str = SCHEMA_VER

    def to_dict(self) -> dict:
        return {
            "schema_ver": self.schema_ver,
            "id": str(self.id),
            "plane": self.plane.to_dict(),
            "bingham": self.bingham.to_dict(),
            "extent": self.extent.to_dict(),
            "quality": self.quality.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @staticmethod
    def from_dict(obj: dict) -> "FractureInstance":
        if obj.get("schema_ver") != SCHEMA_VER:
            raise ContractError(f"schema_ver mismatch: {obj.get('schema_ver')!r}")
        q = obj["quality"]
        return FractureInstance(
            id=str(obj["id"]),
            plane=Plane.from_dict(obj["plane"]),
            bingham=BinghamPlaceholder.from_dict(obj.get("bingham", {})),
            extent=Extent.from_dict(obj["extent"]),
            quality=SegmentQuality.from_dict(q),
            provenance=Provenance.from_dict(obj["provenance"]),
        )


@dataclass
class FractureInstanceSet:
    """契约集合（一场景 / 一数据包）+ 场景元数据。"""

    scene_id: str
    m_per_px: float
    frame: dict
    instances: list = field(default_factory=list)
    schema_ver: str = SCHEMA_VER

    def to_dict(self) -> dict:
        return {
            "schema_ver": self.schema_ver,
            "scene_id": str(self.scene_id),
            "m_per_px": _fin(self.m_per_px, "m_per_px"),
            "frame": dict(self.frame),
            "instances": [i.to_dict() for i in self.instances],
        }

    @staticmethod
    def from_dict(obj: dict) -> "FractureInstanceSet":
        if obj.get("schema_ver") != SCHEMA_VER:
            raise ContractError(f"schema_ver mismatch: {obj.get('schema_ver')!r}")
        return FractureInstanceSet(
            scene_id=str(obj["scene_id"]),
            m_per_px=float(obj["m_per_px"]),
            frame=dict(obj.get("frame", {})),
            instances=[FractureInstance.from_dict(i) for i in obj.get("instances", [])],
        )


def validate_set(s: FractureInstanceSet) -> dict:
    """完整校验；返回统计。违例抛 ContractError。"""
    if s.schema_ver != SCHEMA_VER:
        raise ContractError("schema_ver mismatch")
    if not np.isfinite(s.m_per_px) or s.m_per_px <= 0:
        raise ContractError("m_per_px must be positive finite")
    ids, seg_ids = set(), set()
    n_seg = 0
    for inst in s.instances:
        if inst.id in ids:
            raise ContractError(f"duplicate instance id: {inst.id}")
        ids.add(inst.id)
        n = np.asarray(inst.plane.n, dtype=np.float64)
        if not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1.0) > 1e-6:
            raise ContractError(f"plane.n not unit for {inst.id}")
        if not np.isfinite(inst.plane.d):
            raise ContractError(f"plane.d non-finite for {inst.id}")
        if inst.quality.gate not in GATE_STATUSES:
            raise ContractError(f"bad gate status for {inst.id}")
        pts_all = []
        for seg in inst.extent.segments:
            if seg.seg_id in seg_ids:
                raise ContractError(f"duplicate seg_id: {seg.seg_id}")
            seg_ids.add(seg.seg_id)
            has_m = seg.pts_m is not None and len(seg.pts_m) > 0
            has_px = seg.pts_px is not None and len(seg.pts_px) > 0
            if not (has_m or has_px):
                raise ContractError(f"segment {seg.seg_id}: neither pts_m nor pts_px")
            if has_m:
                arr = np.asarray(seg.pts_m, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
                    raise ContractError(f"bad pts_m shape for {seg.seg_id}")
                if not np.isfinite(arr).all():
                    raise ContractError(f"non-finite pts_m for {seg.seg_id}")
                if seg.pts_px is not None:
                    pp = np.asarray(seg.pts_px, dtype=np.float64)
                    if pp.ndim != 2 or pp.shape[0] != arr.shape[0] or pp.shape[1] != 2:
                        raise ContractError(f"bad pts_px shape for {seg.seg_id}")
                pts_all.append(arr)
            else:
                pp = np.asarray(seg.pts_px, dtype=np.float64)
                if pp.ndim != 2 or pp.shape[0] < 2 or pp.shape[1] != 2:
                    raise ContractError(f"bad pts_px shape for {seg.seg_id}")
                if not np.isfinite(pp).all():
                    raise ContractError(f"non-finite pts_px for {seg.seg_id}")
                pts_all.append(pp)
            if seg.mark not in SEGMENT_MARKS:
                raise ContractError(f"bad mark for {seg.seg_id}")
            n_seg += 1
        if pts_all:
            cat = np.vstack(pts_all)
            lo, hi = inst.extent.bbox_min, inst.extent.bbox_max
            if cat.shape[1] == 3:
                if not (np.all(cat.min(0) >= np.asarray(lo) - 1e-6) and np.all(cat.max(0) <= np.asarray(hi) + 1e-6)):
                    raise ContractError(f"pts outside extent bbox for {inst.id}")
            else:  # 像素系记录：bbox 前两维 = (col,row) 包围盒
                lo2, hi2 = np.asarray(lo[:2]), np.asarray(hi[:2])
                if not (np.all(cat.min(0) >= lo2 - 1e-6) and np.all(cat.max(0) <= hi2 + 1e-6)):
                    raise ContractError(f"pts outside pixel bbox for {inst.id}")
    return {"n_instances": len(s.instances), "n_segments": n_seg}


def save_json(obj: Any, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_set(s: FractureInstanceSet, path) -> None:
    validate_set(s)
    save_json(s.to_dict(), path)


def load_set(path) -> FractureInstanceSet:
    return FractureInstanceSet.from_dict(load_json(path))


def params_hash(obj: Any) -> str:
    """规范 JSON 的 sha256（参数指纹，provenance 用）。"""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
