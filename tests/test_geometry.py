"""
test_geometry.py - Tests for computational geometry, parametric curves, and manifolds.
"""

import numpy as np
import pytest

from src.geometry import (
    normalize,
    rodrigues_rotation,
    rotation_matrix_from_vectors,
    coordinate_frame_matrix,
    BezierCurve3D,
    petal_width_profile,
    petal_cross_section_curvature,
    compute_mesh_normals
)


def test_normalize():
    v = np.array([3.0, 4.0, 0.0])
    u = normalize(v)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.allclose(u, [0.6, 0.8, 0.0])

    # Zero vector safety
    v_zero = np.array([0.0, 0.0, 0.0])
    u_zero = normalize(v_zero)
    assert not np.any(np.isnan(u_zero))


def test_rodrigues_rotation():
    v = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    # 90 degrees around Z axis: [1, 0, 0] -> [0, 1, 0]
    v_rot = rodrigues_rotation(v, axis, np.pi / 2.0)
    assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-6)

    # 180 degrees: [1, 0, 0] -> [-1, 0, 0]
    v_rot180 = rodrigues_rotation(v, axis, np.pi)
    assert np.allclose(v_rot180, [-1.0, 0.0, 0.0], atol=1e-6)


def test_coordinate_frame_matrix():
    facing = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    mat = coordinate_frame_matrix(facing, up, roll=0.0)

    # Must be orthogonal matrix: R @ R.T = I
    assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-6)
    # Determinant must be +1 (special orthogonal SO(3))
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-6)
    # Z axis must align with facing
    assert np.allclose(mat[:, 2], facing, atol=1e-6)


def test_bezier_curve_3d():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([1.0, 1.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    curve = BezierCurve3D(p0, p1, p2, p3)

    # Endpoint interpolation
    pts = curve.evaluate(np.array([0.0, 1.0]))
    assert np.allclose(pts[0], p0)
    assert np.allclose(pts[1], p3)

    # Midpoint
    mid = curve.evaluate(np.array([0.5]))[0]
    assert np.all(np.isfinite(mid))

    # Bishop frames
    T, N, B = curve.compute_frames(np.linspace(0.0, 1.0, 10))
    for i in range(10):
        assert np.isclose(np.linalg.norm(T[i]), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(N[i]), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(B[i]), 1.0, atol=1e-5)
        assert np.isclose(np.dot(T[i], N[i]), 0.0, atol=1e-5)
        assert np.isclose(np.dot(T[i], B[i]), 0.0, atol=1e-5)


def test_petal_width_profile():
    s = np.linspace(0.0, 1.0, 100)
    w = petal_width_profile(s, scale=0.45, alpha=1.20, beta=0.88)

    # Dirichlet boundary conditions
    assert np.isclose(w[0], 0.0, atol=1e-5)
    assert np.isclose(w[-1], 0.0, atol=1e-5)

    # Non-negativity
    assert np.all(w >= 0.0)

    # Peak width occurs in interior midsection (0.4 < s < 0.7)
    peak_idx = np.argmax(w)
    s_peak = s[peak_idx]
    assert 0.40 <= s_peak <= 0.70


def test_petal_cross_section_curvature():
    S, T = np.meshgrid(np.linspace(0.0, 1.0, 20), np.linspace(-1.0, 1.0, 11), indexing='ij')
    c = petal_cross_section_curvature(S, T)
    assert np.all(np.isfinite(c))
    assert c.shape == S.shape


def test_compute_mesh_normals():
    # Simple quad with two triangles in XY plane facing +Z
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    normals = compute_mesh_normals(vertices, faces)
    assert normals.shape == (4, 3)
    for n in normals:
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-5)
        assert np.isclose(n[2], 1.0, atol=1e-5)
