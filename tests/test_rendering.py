"""
test_rendering.py - Tests for Z-buffer rasterizer, camera projection, and PBR shading.
"""

import numpy as np
import pytest

from src.foliage import BouquetScene
from src.renderer import LilyRenderer


def test_camera_projection():
    renderer = LilyRenderer(width=640, height=360, fov_deg=36.0)
    # Point along optical axis in front of camera
    test_pt = renderer.camera_pos + np.array([0.0, 0.0, -2.0])
    screen_coords, valid = renderer.project_to_screen(test_pt.reshape(1, 3))
    assert valid[0]
    # Projected X, Y should be near screen center
    assert 200.0 < screen_coords[0, 0] < 440.0
    assert 100.0 < screen_coords[0, 1] < 260.0


def test_material_ids_present():
    scene = BouquetScene(seed=2026)
    geom = scene.generate_bouquet_geometry()
    mat_ids = geom["mat_ids"]

    # Verify all 7 botanical material classes are present in the mesh
    unique_ids = set(np.unique(mat_ids))
    expected_ids = {1, 2, 3, 4, 5, 6, 7}
    for m in expected_ids:
        assert m in unique_ids, f"Material ID {m} missing from scene geometry"


def test_rendered_image_properties():
    scene = BouquetScene(seed=2026)
    geom = scene.generate_bouquet_geometry()

    width, height = 480, 300
    renderer = LilyRenderer(width=width, height=height)
    img = renderer.render(geom)

    # Dimensional checks
    assert img.shape == (height, width, 3)
    assert img.dtype == np.uint8

    # Numerical validity
    assert not np.any(np.isnan(img))
    assert np.all(img >= 0)
    assert np.all(img <= 255)

    # Ensure scene is not blank / solid color
    assert np.std(img) > 10.0
