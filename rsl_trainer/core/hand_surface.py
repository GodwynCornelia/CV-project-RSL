"""A volumetric skin around observed joints, without inventing their hidden depth."""

from dataclasses import dataclass

import numpy as np


def _sphere():
    latitudes, longitudes = 5, 8
    vertices = []
    for row in range(latitudes + 1):
        theta = np.pi * row / latitudes
        for column in range(longitudes):
            phi = 2 * np.pi * column / longitudes
            vertices.append([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
    faces = []
    for row in range(latitudes):
        for column in range(longitudes):
            a = row * longitudes + column
            b = row * longitudes + (column + 1) % longitudes
            c, d = a + longitudes, b + longitudes
            if row:
                faces.append([a, b, c])
            if row < latitudes - 1:
                faces.append([b, d, c])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


UNIT_SPHERE, SPHERE_FACES = _sphere()
FINGER_BONES = ((0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20))


@dataclass
class HandSurface:
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    observed_joints: np.ndarray
    has_source_depth: bool


def build_hand_surface(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape not in ((21, 2), (21, 3)) or not np.isfinite(points).all():
        raise ValueError("Ожидались 21 точка кисти с координатами X, Y или X, Y, Z")
    measured = points.shape[1] == 3
    joints = points.copy() if measured else np.column_stack([points, np.zeros(21, dtype=np.float32)])
    palm_axis = joints[[5, 9, 13, 17]].mean(axis=0) - joints[0]
    palm_length = float(np.linalg.norm(palm_axis))
    if palm_length < 1e-5:
        raise ValueError("Недостаточно геометрии для построения кисти")
    vertices, normals, faces = [], [], []

    def ellipsoid(center, axis, length, width, depth):
        along = axis / max(float(np.linalg.norm(axis)), 1e-6)
        reference = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(along, reference)) > 0.95:
            reference = np.array([1, 0, 0], dtype=np.float32)
        across = np.cross(along, reference)
        across /= max(float(np.linalg.norm(across)), 1e-6)
        outward = np.cross(across, along)
        basis = np.column_stack([across, along, outward])
        sizes = np.array([width, length, depth], dtype=np.float32)
        transformed = (UNIT_SPHERE * sizes) @ basis.T + center
        transformed_normals = (UNIT_SPHERE / sizes) @ basis.T
        transformed_normals /= np.maximum(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-6)
        offset = sum(len(part) for part in vertices)
        vertices.append(transformed)
        normals.append(transformed_normals)
        faces.append(SPHERE_FACES + offset)

    def capsule(start, end, radius):
        axis = end - start
        distance = float(np.linalg.norm(axis))
        along = axis / distance
        reference = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(along, reference)) > 0.95:
            reference = np.array([1, 0, 0], dtype=np.float32)
        across = np.cross(along, reference)
        across /= max(float(np.linalg.norm(across)), 1e-6)
        outward = np.cross(across, along)
        basis = np.column_stack([across, along, outward])
        # Duplicate the equator at either end to form a cylinder with rounded caps.
        theta = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2,
                          np.pi / 2, np.pi * 2 / 3, np.pi * 5 / 6, np.pi])
        phi = np.arange(8) * np.pi / 4
        unit = np.stack([np.sin(theta)[:, None] * np.cos(phi),
                         np.broadcast_to(np.cos(theta)[:, None], (8, 8)),
                         np.sin(theta)[:, None] * np.sin(phi)], axis=-1).reshape(-1, 3)
        local = unit * radius
        local[:, 1] += np.repeat(np.array([1] * 4 + [-1] * 4), 8) * distance / 2
        offset = sum(len(part) for part in vertices)
        triangles = []
        for row in range(7):
            for column in range(8):
                a, b = row * 8 + column, row * 8 + (column + 1) % 8
                c, d = a + 8, b + 8
                if row:
                    triangles.append([a, b, c])
                if row < 6:
                    triangles.append([b, d, c])
        vertices.append(local @ basis.T + (start + end) / 2)
        normals.append(unit @ basis.T)
        faces.append(np.asarray(triangles, dtype=np.int32) + offset)

    across = np.array([palm_axis[1], -palm_axis[0], 0], dtype=np.float32)
    across /= max(float(np.linalg.norm(across)), 1e-6)
    projected = joints[[5, 9, 13, 17]] @ across
    palm_width = max(float(np.ptp(projected)) * 0.58, palm_length * 0.16)
    center = joints[0] + palm_axis * 0.56
    ellipsoid(center, palm_axis, palm_length * 0.63, palm_width, palm_length * (0.19 if measured else 0.07))
    for a, b in FINGER_BONES:
        axis = joints[b] - joints[a]
        length = float(np.linalg.norm(axis))
        if length < palm_length * 0.01:
            continue
        # Constant anatomical thickness comes from palm length, not a foreshortened width.
        radius = palm_length * (0.105 if a < 5 else 0.085)
        if a >= 17:
            radius *= 0.85
        capsule(joints[a], joints[b], radius)
    return HandSurface(np.concatenate(vertices), np.concatenate(normals), np.concatenate(faces), joints, measured)


def surface_view(surface, yaw=0.0, pitch=0.0):
    """Rotate the shell; 2D sources keep coplanar joint centers by construction."""
    yaw, pitch = np.radians([yaw, pitch])
    rotate_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rotate_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rotation = rotate_x @ rotate_y
    center = surface.observed_joints[[0, 5, 9, 13, 17]].mean(axis=0)
    return (surface.vertices - center) @ rotation.T + center, surface.normals @ rotation.T
