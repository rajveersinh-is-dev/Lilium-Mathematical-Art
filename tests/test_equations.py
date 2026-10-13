"""
test_equations.py - Tests for mathematical specifications, Typst typesetting, and LaTeX sources.
"""

from pathlib import Path
import pytest
from PIL import Image

from src.math_spec import (
    get_mathematical_model_data,
    generate_typst_panel_markup,
    generate_latex_source
)
from src.composition import render_equation_panel


def test_mathematical_model_data():
    data = get_mathematical_model_data()
    assert "domain" in data
    assert "petal_profile" in data
    assert "perianth" in data
    assert "lighting" in data

    assert data["perianth"]["inner_petals"] == 3
    assert data["perianth"]["outer_sepals"] == 3
    assert data["perianth"]["stamens"] == 6
    assert data["domain"]["width"] == 2400
    assert data["domain"]["height"] == 1440


def test_typst_markup_structure():
    markup = generate_typst_panel_markup()
    assert "ANALYTICAL FOUNDATIONS OF THE LILIUM SYNTHESIS ENGINE" in markup
    # Check all 10 analytical sections exist
    for i in range(1, 11):
        assert f"== {i}." in markup


def test_latex_source_structure():
    tex = generate_latex_source()
    assert r"\documentclass" in tex
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex
    assert r"\section{Coordinate Domain and Camera Projection}" in tex
    assert r"\section{Parametric Tepal Manifold $\mathbf{S}_{j,k}$}" in tex
    assert r"\section{Trimerous Perianth Configuration}" in tex


def test_typst_compilation():
    test_out = "outputs/test_panel.png"
    img = render_equation_panel(
        output_path=test_out,
        typ_source_path="equations/equation_panel.typ",
        target_width=1200,
        target_height=440
    )
    assert isinstance(img, Image.Image)
    assert img.size == (1200, 440)
    assert Path(test_out).exists()
    # Clean up test output
    Path(test_out).unlink(missing_ok=True)
