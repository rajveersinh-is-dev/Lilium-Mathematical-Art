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


def test_raffia_tie_wraps_the_stem_bundle():
    """
    The artisan raffia tie must grip the real stem cluster: no cord vertex may
    lie inside a stem tube, and the wrap must reach at least as far as the stems
    it binds. Guards against the tie being authored against a nominal ellipse
    that is narrower than the actual bundle.
    """
    from src.foliage import BouquetScene

    scene = BouquetScene(seed=2026)
    geom = scene.generate_bouquet_geometry()
    verts = geom["vertices"]
    mat_ids = geom["mat_ids"]

    tie = verts[mat_ids == 9]
    stems = verts[mat_ids == 6]
    assert len(tie) > 500 and len(stems) > 500

    sample = tie[::6]
    worst = np.inf
    for stem in scene.stems:
        u = np.linspace(0.0, 1.0, 160)
        centre = stem.bezier.evaluate(u)
        radius = stem.r_base * (1.0 - u) + stem.r_tip * u
        d = np.linalg.norm(sample[:, None, :] - centre[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        clearance = d[np.arange(len(sample)), nearest] - radius[nearest]
        worst = min(worst, float(clearance.min()))
    assert worst > 0.0, f"raffia tie penetrates a stem by {-worst:.5f}"

    # the wrap must enclose the bundle at the tie height
    band = (stems[:, 1] > -0.66) & (stems[:, 1] < -0.58)
    tie_band = (tie[:, 1] > -0.66) & (tie[:, 1] < -0.58)
    cx = 0.5 * (stems[band, 0].min() + stems[band, 0].max())
    cz = 0.5 * (stems[band, 2].min() + stems[band, 2].max())
    stem_r = np.hypot(stems[band, 0] - cx, stems[band, 2] - cz).max()
    tie_r = np.hypot(tie[tie_band, 0] - cx, tie[tie_band, 2] - cz).max()
    assert tie_r >= stem_r, f"tie ({tie_r:.3f}) does not enclose the bundle ({stem_r:.3f})"
