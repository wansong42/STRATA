from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence

from fractureflow.obs_equiv_manifest import Disk, Scene, TwoPointQuery, normalized_weights

TAU_SIN = 1e-8
TAU_LEN_REL = 1e-9


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


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return (value[0] * factor, value[1] * factor, value[2] * factor)


def point_disk_distance(disk: Disk, point: Sequence[float]) -> float:
    delta = _sub(point, disk.center)
    axial = _dot(delta, disk.normal)
    planar = _sub(delta, _scale(disk.normal, axial))
    radial_gap = max(_norm(planar) - disk.radius, 0.0)
    return math.sqrt(axial * axial + radial_gap * radial_gap)


def disks_intersect(first: Disk, second: Disk, length_scale: float) -> bool:
    delta = _sub(second.center, first.center)
    cross = _cross(first.normal, second.normal)
    sine = _norm(cross)
    if sine <= TAU_SIN:
        signed_separation = _dot(delta, first.normal)
        coplanar = abs(signed_separation) <= TAU_LEN_REL * length_scale
        planar = _sub(delta, _scale(first.normal, signed_separation))
        return coplanar and _norm(planar) < first.radius + second.radius
    direction = _scale(cross, 1.0 / sine)
    first_offset = _dot(delta, second.normal) / sine
    second_offset = _dot(delta, first.normal) / sine
    projected_center_distance = abs(_dot(delta, direction))
    if abs(first_offset) >= first.radius or abs(second_offset) >= second.radius:
        return False
    first_half_chord = math.sqrt(max(first.radius * first.radius - first_offset * first_offset, 0.0))
    second_half_chord = math.sqrt(max(second.radius * second.radius - second_offset * second_offset, 0.0))
    return projected_center_distance < first_half_chord + second_half_chord


def scene_length_scale(scene: Scene) -> float:
    radii = sorted(disk.radius for disk in scene.disks)
    if not radii:
        return 1.0
    middle = len(radii) // 2
    if len(radii) % 2:
        return radii[middle]
    return (radii[middle - 1] + radii[middle]) / 2.0


def intersection_edges(scene: Scene) -> frozenset[tuple[str, str]]:
    scale = scene_length_scale(scene)
    edges = set()
    for index, first in enumerate(scene.disks):
        for second in scene.disks[index + 1:]:
            if disks_intersect(first, second, scale):
                edges.add((first.object_id, second.object_id))
    return frozenset(edges)


def _attached(scene: Scene, point: Sequence[float], radius: float) -> frozenset[str]:
    return frozenset(disk.object_id for disk in scene.disks if point_disk_distance(disk, point) <= radius)


def two_point_connected(scene: Scene, query: TwoPointQuery) -> bool:
    sources = _attached(scene, query.point_a, query.attach_radius)
    targets = _attached(scene, query.point_b, query.attach_radius)
    if not sources or not targets:
        return False
    adjacency = {disk.object_id: set() for disk in scene.disks}
    for first, second in intersection_edges(scene):
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached = set(sources)
    queue = deque(sorted(sources))
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return not reached.isdisjoint(targets)


def query_labels(scenes: Iterable[Scene], query: TwoPointQuery) -> tuple[bool, ...]:
    return tuple(two_point_connected(scene, query) for scene in scenes)


def reference_posterior(scenes: Sequence[Scene], weights: Sequence[float], query: TwoPointQuery) -> tuple[float, tuple[bool, ...]]:
    parsed_weights = normalized_weights(weights, len(scenes))
    labels = query_labels(scenes, query)
    probability = sum(weight for weight, label in zip(parsed_weights, labels) if label)
    return probability, labels
