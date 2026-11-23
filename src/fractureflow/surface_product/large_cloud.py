# -*- coding: utf-8 -*-
"""W160 bounded-memory access to a complete PLY point cloud.

Boundary (not a convention): the 1.63 GB / 60,343,162-point source is never
loaded as one array.  The reader parses the PLY header, then walks fixed-size
vertex blocks with seek + read(n * record_bytes).  The only accumulated product
is a capped ROI; file identity is recomputed in fixed byte chunks.

This module is deliberately separate from geometry_kernel.read_ply_points
(which materialises a whole file under the 256 MiB small-asset ceiling).  A
streaming ROI path must never be described as a raised in-memory asset cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BLOCK_POINTS = 1_048_576
DEFAULT_SHA_CHUNK_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
SOURCE_INDEX_DTYPE = np.dtype("<u4")
_UINT32_MAX = 2 ** 32 - 1

PLY_SCALAR = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "int64": "<i8", "uint64": "<u8",
    "float32": "<f4", "float64": "<f8",
}
_DTYPE_PLY = {"float32": "float", "float64": "double"}


class LargeCloudError(ValueError):
    """A bounded-memory access boundary violation with a stable code."""

    def __init__(self, message: str, code: str = "LARGE_CLOUD_INVALID") -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise LargeCloudError(message, code=code)


def _rss_bytes() -> Optional[int]:
    try:
        import psutil
    except Exception:  # noqa: BLE001 - optional measurement dependency
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class PlyHeader:
    path: str
    format: str
    vertex_count: int
    properties: Tuple[Tuple[str, str], ...]
    record_bytes: int
    body_offset: int
    file_bytes: int
    dtype: np.dtype

    @property
    def property_names(self) -> List[str]:
        return [name for name, _kind in self.properties]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "vertex_count": int(self.vertex_count),
            "properties": [{"name": name, "type": kind} for name, kind in self.properties],
            "record_bytes": int(self.record_bytes),
            "body_offset": int(self.body_offset),
            "file_bytes": int(self.file_bytes),
            "expected_end_offset": int(self.body_offset + self.vertex_count * self.record_bytes),
        }


def read_ply_header(source_path: Any) -> PlyHeader:
    """Parse and validate one binary little-endian PLY header.

    Refuses: missing file, bad magic, non little-endian formats, list-valued
    vertex properties, unsupported scalars, missing xyz, empty vertex element,
    oversized headers and a body length that disagrees with the file size.
    """
    target = Path(source_path)
    _require(target.is_file(), "MISSING_ASSET", "point cloud file is missing: %s" % target)
    file_bytes = int(target.stat().st_size)
    _require(file_bytes > 0, "EMPTY_ASSET", "point cloud file is empty: %s" % target)
    with target.open("rb") as handle:
        first = handle.readline()
        _require(first.strip() == b"ply", "BAD_PLY_HEADER", "not a PLY file: %s" % target)
        header_bytes = len(first)
        fmt = ""
        vertex_count: Optional[int] = None
        properties: List[Tuple[str, str]] = []
        current: Optional[str] = None
        while True:
            line = handle.readline()
            header_bytes += len(line)
            _require(bool(line), "BAD_PLY_HEADER", "PLY header has no end_header")
            _require(header_bytes <= MAX_HEADER_BYTES, "BAD_PLY_HEADER",
                     "PLY header exceeds %d bytes" % MAX_HEADER_BYTES)
            text = line.decode("ascii", "replace").strip()
            if not text:
                continue
            parts = text.split()
            if parts[0] == "end_header":
                break
            if parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "format":
                _require(len(parts) >= 2, "BAD_PLY_HEADER", "malformed format line: %r" % text)
                fmt = parts[1]
            elif parts[0] == "element":
                _require(len(parts) >= 3, "BAD_PLY_HEADER", "malformed element line: %r" % text)
                current = parts[1]
                if current == "vertex":
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as exc:
                        raise LargeCloudError("vertex count is not an integer: %r" % parts[2],
                                              code="BAD_PLY_HEADER") from exc
            elif parts[0] == "property":
                if current != "vertex":
                    continue
                _require(len(parts) == 3, "PLY_PROPERTY_LIST_UNSUPPORTED",
                         "vertex list properties are not supported: %r" % text)
                _require(parts[1] in PLY_SCALAR, "UNSUPPORTED_PLY_SCALAR",
                         "unsupported PLY scalar type %r" % parts[1])
                properties.append((parts[2], parts[1]))
        body_offset = int(handle.tell())
    _require(fmt == "binary_little_endian", "UNSUPPORTED_PLY_FORMAT",
             "only binary_little_endian PLY is supported, got %r" % fmt)
    _require(vertex_count is not None, "BAD_PLY_HEADER", "PLY declares no vertex element")
    _require(int(vertex_count) > 0, "EMPTY_ASSET", "PLY vertex element is empty")
    names = [name for name, _kind in properties]
    _require({"x", "y", "z"}.issubset(set(names)), "PLY_MISSING_XYZ", "PLY vertex lacks x/y/z")
    seen: Dict[str, int] = {}
    fields: List[Tuple[str, str]] = []
    for name, kind in properties:
        count = seen.get(name, 0)
        seen[name] = count + 1
        key = name if count == 0 else "%s_%d" % (name, count)
        fields.append((key, PLY_SCALAR[kind]))
    dtype = np.dtype(fields)
    expected_end = body_offset + int(vertex_count) * int(dtype.itemsize)
    _require(file_bytes == expected_end, "TRUNCATED_ASSET",
             "file size %d disagrees with header body length %d (delta %+d)"
             % (file_bytes, expected_end, file_bytes - expected_end))
    return PlyHeader(path=str(target), format=fmt, vertex_count=int(vertex_count),
                     properties=tuple(properties), record_bytes=int(dtype.itemsize),
                     body_offset=body_offset, file_bytes=file_bytes, dtype=dtype)


def sha256_stream(source_path: Any, *, chunk_bytes: int = DEFAULT_SHA_CHUNK_BYTES,
                  cancel_after_chunks: Optional[int] = None) -> Dict[str, Any]:
    """Recompute a file SHA256 in fixed chunks; never read the file whole."""
    target = Path(source_path)
    _require(target.is_file(), "MISSING_ASSET", "file is missing: %s" % target)
    _require(int(chunk_bytes) > 0, "BAD_ARGUMENT", "chunk_bytes must be positive")
    digest = hashlib.sha256()
    chunks = 0
    read_bytes = 0
    started = time.perf_counter()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
            read_bytes += len(chunk)
            chunks += 1
            if cancel_after_chunks is not None and chunks >= int(cancel_after_chunks):
                raise LargeCloudError("hash pass cancelled after %d chunks" % chunks, code="CANCELLED")
    return {"sha256": digest.hexdigest(), "bytes": int(read_bytes), "chunks": chunks,
            "chunk_bytes": int(chunk_bytes), "seconds": round(time.perf_counter() - started, 3)}


def iter_vertex_blocks(source_path: Any, header: PlyHeader, *,
                       block_points: int = DEFAULT_BLOCK_POINTS) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_ordinal, records) for fixed-size vertex blocks."""
    _require(int(block_points) > 0, "BAD_ARGUMENT", "block_points must be positive")
    target = Path(source_path)
    with target.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        _require(int(handle.tell()) == header.file_bytes, "ASSET_CHANGED_WHILE_OPEN",
                 "file size changed after the header was parsed: %s" % target)
        handle.seek(header.body_offset)
        start = 0
        remaining = header.vertex_count
        while remaining > 0:
            count = min(int(block_points), remaining)
            expected = count * header.record_bytes
            payload = handle.read(expected)
            _require(len(payload) == expected, "TRUNCATED_ASSET",
                     "block %d reads %d of %d bytes" % (start, len(payload), expected))
            records = np.frombuffer(payload, dtype=header.dtype, count=count)
            yield start, records
            start += count
            remaining -= count


def scan_bounds(source_path: Any, header: PlyHeader, *,
                block_points: int = DEFAULT_BLOCK_POINTS) -> Dict[str, Any]:
    """One streaming pass: finite bounds, finite/non-finite counts, block count."""
    started = time.perf_counter()
    baseline = _rss_bytes()
    peak = baseline
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    blocks = 0
    finite = 0
    non_finite = 0
    for _start, records in iter_vertex_blocks(source_path, header, block_points=block_points):
        xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
        ok = np.isfinite(xyz).all(axis=1)
        finite += int(np.count_nonzero(ok))
        non_finite += int(len(ok) - np.count_nonzero(ok))
        if np.any(ok):
            values = xyz[ok]
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
        blocks += 1
        current = _rss_bytes()
        if current is not None and (peak is None or current > peak):
            peak = current
    _require(finite > 0, "EMPTY_ASSET", "point cloud has no finite points")
    return {
        "bounds_min": [float(v) for v in minimum],
        "bounds_max": [float(v) for v in maximum],
        "finite_points": int(finite),
        "non_finite_points": int(non_finite),
        "blocks": int(blocks),
        "block_points": int(block_points),
        "peak_rss_bytes": peak,
        "baseline_rss_bytes": baseline,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _validate_bounds(bounds_min: Sequence[float], bounds_max: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(list(bounds_min), dtype=np.float64)
    high = np.asarray(list(bounds_max), dtype=np.float64)
    _require(low.shape == (3,) and high.shape == (3,), "ROI_BOUNDS_INVALID",
             "ROI bounds must be three numbers each")
    _require(bool(np.all(np.isfinite(low))) and bool(np.all(np.isfinite(high))), "ROI_BOUNDS_INVALID",
             "ROI bounds carry non-finite numbers")
    _require(bool(np.all(low <= high)), "ROI_BOUNDS_INVALID", "ROI bounds min exceeds max")
    return low, high


def extract_roi(source_path: Any, header: PlyHeader, *, bounds_min: Sequence[float],
                bounds_max: Sequence[float], block_points: int = DEFAULT_BLOCK_POINTS,
                max_points: int = 3_000_000,
                cancel_after_blocks: Optional[int] = None) -> Dict[str, Any]:
    """Stream the complete file once and accumulate the capped ROI.

    Selection is an inclusive AABB over x/y/z with no stride and no decimation.
    Accumulated order is ascending source ordinal, so ROI local index i is
    stable as long as the ROI rule, block size and source bytes are unchanged.
    """
    low, high = _validate_bounds(bounds_min, bounds_max)
    _require(int(max_points) > 0, "BAD_ARGUMENT", "max_points must be positive")
    _require(header.vertex_count <= _UINT32_MAX, "SOURCE_INDEX_UNSUPPORTED",
             "vertex count %d does not fit a uint32 source index" % header.vertex_count)
    started = time.perf_counter()
    baseline = _rss_bytes()
    peak = baseline
    points_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    hits = 0
    scanned = 0
    blocks = 0
    non_finite = 0
    coordinate_dtype: Optional[np.dtype] = None
    per_block: List[Dict[str, Any]] = []
    for start, records in iter_vertex_blocks(source_path, header, block_points=block_points):
        if cancel_after_blocks is not None and blocks >= int(cancel_after_blocks):
            raise LargeCloudError("ROI extraction cancelled after %d blocks" % blocks, code="CANCELLED")
        xyz = np.column_stack([records["x"], records["y"], records["z"]])
        if coordinate_dtype is None:
            coordinate_dtype = xyz.dtype
        elif xyz.dtype != coordinate_dtype:
            coordinate_dtype = np.result_type(coordinate_dtype, xyz.dtype)
        finite = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
        non_finite += int(len(finite) - np.count_nonzero(finite))
        mask = (
            finite
            & (xyz[:, 0] >= low[0]) & (xyz[:, 0] <= high[0])
            & (xyz[:, 1] >= low[1]) & (xyz[:, 1] <= high[1])
            & (xyz[:, 2] >= low[2]) & (xyz[:, 2] <= high[2])
        )
        block_hits = int(np.count_nonzero(mask))
        if block_hits:
            points_parts.append(np.asarray(xyz[mask], dtype=coordinate_dtype))
            ordinals = np.arange(start, start + len(records), dtype=np.uint64)[mask]
            _require(int(ordinals.max()) <= _UINT32_MAX, "SOURCE_INDEX_UNSUPPORTED",
                     "source ordinal exceeds uint32 range")
            index_parts.append(ordinals.astype(SOURCE_INDEX_DTYPE))
            hits += block_hits
            _require(hits <= int(max_points), "ROI_TOO_LARGE",
                     "ROI hit count %d exceeds the preregistered cap %d" % (hits, int(max_points)))
        scanned += len(records)
        blocks += 1
        per_block.append({"block": blocks - 1, "start": int(start), "count": int(len(records)),
                          "hits": block_hits})
        current = _rss_bytes()
        if current is not None and (peak is None or current > peak):
            peak = current
    _require(scanned == header.vertex_count, "TRUNCATED_ASSET",
             "scanned %d of %d vertices" % (scanned, header.vertex_count))
    _require(hits > 0, "EMPTY_ROI", "ROI selected no points")
    _require(coordinate_dtype is not None, "EMPTY_ASSET", "no vertex data was read")
    points = np.concatenate(points_parts, axis=0).astype(coordinate_dtype, copy=False)
    source_index = np.concatenate(index_parts, axis=0).astype(SOURCE_INDEX_DTYPE, copy=False)
    _require(len(points) == len(source_index) == hits, "SOURCE_INDEX_INVALID",
             "ROI points and source index disagree")
    _require(bool(np.all(np.diff(source_index.astype(np.int64)) > 0)), "SOURCE_INDEX_INVALID",
             "source indices are not strictly increasing")
    del points_parts, index_parts
    return {
        "points": points,
        "source_index": source_index,
        "coordinate_dtype": str(np.dtype(coordinate_dtype)),
        "hit_count": int(hits),
        "scanned_points": int(scanned),
        "blocks": int(blocks),
        "block_points": int(block_points),
        "non_finite_excluded": int(non_finite),
        "bounds_min": [float(v) for v in low],
        "bounds_max": [float(v) for v in high],
        "source_index_min": int(source_index[0]),
        "source_index_max": int(source_index[-1]),
        "peak_rss_bytes": peak,
        "baseline_rss_bytes": baseline,
        "rss_method": "psutil RSS sampled after each block" if baseline is not None
                      else "unavailable (psutil absent)",
        "seconds": round(time.perf_counter() - started, 3),
        "per_block": per_block,
    }


def write_xyz_ply(target_path: Any, points: np.ndarray, *, dtype: Optional[np.dtype] = None) -> Dict[str, Any]:
    """Write an xyz-only binary PLY atomically (temp file + replace)."""
    values = np.asarray(points)
    _require(values.ndim == 2 and values.shape[1] == 3, "BAD_ARGUMENT",
             "points must be an N x 3 array")
    chosen = np.dtype(dtype or values.dtype)
    _require(str(chosen) in ("float32", "float64"), "BAD_ARGUMENT",
             "only float32/float64 xyz PLY output is supported, got %s" % chosen)
    token = _DTYPE_PLY[chosen.name]
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header_text = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                   "property %s x\nproperty %s y\nproperty %s z\nend_header\n"
                   % (len(values), token, token, token))
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(header_text.encode("ascii"))
            np.ascontiguousarray(values, dtype=chosen).tofile(handle)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target), "points": int(len(values)), "dtype": str(chosen),
            "bytes": int(target.stat().st_size)}


def write_source_index(target_path: Any, source_index: np.ndarray) -> Dict[str, Any]:
    """Write the ROI local -> source ordinal map atomically as little-endian uint32."""
    values = np.ascontiguousarray(np.asarray(source_index), dtype=SOURCE_INDEX_DTYPE)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        values.tofile(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return {"path": str(target), "count": int(len(values)), "dtype": "<u4",
            "bytes": int(target.stat().st_size), "sha256": digest}


def read_source_index(source_path: Any, *, count: int) -> np.ndarray:
    target = Path(source_path)
    _require(target.is_file(), "MISSING_ASSET", "source index file is missing: %s" % target)
    expected = int(count) * SOURCE_INDEX_DTYPE.itemsize
    _require(int(target.stat().st_size) == expected, "SOURCE_INDEX_INVALID",
             "source index size %d != %d" % (target.stat().st_size, expected))
    values = np.fromfile(target, dtype=SOURCE_INDEX_DTYPE, count=int(count))
    _require(len(values) == int(count), "SOURCE_INDEX_INVALID", "source index is short")
    return values


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_rows_by_ordinal(source_path: Any, header: PlyHeader, ordinals: Sequence[int],
                         *, fields: Sequence[str] = ("x", "y", "z")) -> np.ndarray:
    """Read a few vertex records by ordinal with one seek each."""
    chosen = np.asarray(list(ordinals), dtype=np.int64)
    _require(bool(np.all(chosen >= 0)) and bool(np.all(chosen < header.vertex_count)),
             "SOURCE_INDEX_OUT_OF_RANGE", "ordinal outside the declared vertex range")
    columns: List[List[float]] = [[] for _ in fields]
    with Path(source_path).open("rb") as handle:
        for ordinal in chosen:
            handle.seek(header.body_offset + int(ordinal) * header.record_bytes)
            payload = handle.read(header.record_bytes)
            _require(len(payload) == header.record_bytes, "TRUNCATED_ASSET",
                     "short read at ordinal %d" % ordinal)
            record = np.frombuffer(payload, dtype=header.dtype, count=1)[0]
            for column, name in zip(columns, fields):
                column.append(float(record[name]))
    return np.asarray(columns, dtype=np.float64).T
