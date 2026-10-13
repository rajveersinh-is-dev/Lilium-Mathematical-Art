"""
test_determinism.py - Verifies bitwise deterministic reproducibility of Project Lilium.
"""

import numpy as np
import pytest

from src.foliage import BouquetScene
from src.renderer import LilyRenderer


def test_geometry_determinism():
    """Confirms that identical seeds generate identical 3D meshes."""
    scene1 = BouquetScene(seed=2026)
    geom1 = scene1.generate_bouquet_geometry()

    scene2 = BouquetScene(seed=2026)
    geom2 = scene2.generate_bouquet_geometry()

    assert np.array_equal(geom1["vertices"], geom2["vertices"])
    assert np.array_equal(geom1["faces"], geom2["faces"])
    assert np.array_equal(geom1["normals"], geom2["normals"])
    assert np.array_equal(geom1["uvs"], geom2["uvs"])
    assert np.array_equal(geom1["mat_ids"], geom2["mat_ids"])


def test_rendering_determinism():
    """Confirms that identical scenes render exact identical pixel values."""
    scene = BouquetScene(seed=2026)
    geom = scene.generate_bouquet_geometry()

    # Small test viewport for fast exact check
    renderer1 = LilyRenderer(width=320, height=200)
    img1 = renderer1.render(geom)

    renderer2 = LilyRenderer(width=320, height=200)
    img2 = renderer2.render(geom)

    assert np.array_equal(img1, img2)
