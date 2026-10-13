"""
materials.py - Physically-motivated botanical materials and BRDF evaluation.

Provides:
- Material definitions for white lily petals, outer sepals, filaments, pollen-dusted anthers,
  pistil, stems, and lanceolate leaves.
- High-fidelity procedural surface evaluation:
  - Longitudinal vein fields and papillae nectar ridges.
  - Subsurface scattering / thin-sheet translucency for white petals.
  - Waxy cuticle specular microfacet sheen.
"""

from typing import Tuple, Dict, Any
import numpy as np


class MaterialProperties:
    """Optical parameters for a biological surface."""
    def __init__(
        self,
        base_color: np.ndarray,
        roughness: float = 0.4,
        specular_k: float = 0.2,
        specular_exp: float = 32.0,
        subsurface_k: float = 0.25,
        subsurface_color: np.ndarray = None
    ):
        self.base_color = np.asarray(base_color, dtype=np.float32)
        self.roughness = roughness
        self.specular_k = specular_k
        self.specular_exp = specular_exp
        self.subsurface_k = subsurface_k
        self.subsurface_color = (
            np.asarray(subsurface_color, dtype=np.float32)
            if subsurface_color is not None
            else self.base_color.copy()
        )


def evaluate_petal_surface_color(
    uv: np.ndarray,
    is_inner: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluates procedural texture for a lily petal at coordinate (s, t):
    - s in [0, 1]: normalized distance from receptacle to apex.
    - t in [-1, 1]: normalized transverse coordinate across petal half-width.
    
    Returns:
        albedo: RGB color (3,)
        translucency: float in [0, 1]
        roughness: float
    """
    s = np.clip(uv[0], 0.0, 1.0)
    t = np.clip(uv[1], -1.0, 1.0)
    abs_t = abs(t)
    
    # 1. Base luminous ivory white with subtle cream warmth
    ivory_white = np.array([0.985, 0.980, 0.965], dtype=np.float32)
    
    # 2. Receptacle / Throat Celadon Green Gradient
    # Near the base (s < 0.40), petals emerge as translucent chartreuse/celadon green
    celadon_green = np.array([0.52, 0.84, 0.28], dtype=np.float32)
    throat_factor = np.exp(-7.5 * (s ** 1.4))
    
    # 3. Midrib Translucent Groove along t = 0
    # Central groove has pale lime-ivory translucency extending further up the petal
    midrib_green = np.array([0.62, 0.84, 0.38], dtype=np.float32)
    midrib_factor = (1.0 - 0.45 * s) * np.exp(-26.0 * (t ** 2))
    
    # 4. Longitudinal Vein Intensity Field: V(s, t) = sum_k a_k exp(-(t - τ_k)^2 / 2σ^2)
    vein_modulation = 1.0
    for k in range(1, 5):
        tau = 0.22 * k * (1.0 - 0.15 * s)
        vein_modulation += 0.04 * np.exp(-((abs_t - tau) ** 2) / 0.002)
    
    # 5. Carmine Papillae (Nectar ridges / delicate pink-red specks near throat)
    # Characteristic of Lilium auratum / oriental hybrids: raised crimson-pink dots near s in [0.12, 0.38]
    papillae_mask = 0.0
    if 0.12 < s < 0.38 and abs_t < 0.48:
        dot = np.sin(42.0 * s) * np.cos(34.0 * t)
        if dot > 0.65:
            papillae_mask = (dot - 0.65) / 0.35 * (0.38 - s) / 0.26 * (1.0 - abs_t / 0.48)
            
    carmine_dot = np.array([0.74, 0.18, 0.28], dtype=np.float32)
    
    # Blend components
    color = ivory_white * (1.0 - throat_factor) + celadon_green * throat_factor
    color = color * (1.0 - 0.45 * midrib_factor) + midrib_green * (0.45 * midrib_factor)
    color = color * (0.97 + 0.03 * vein_modulation)
    color = color * (1.0 - papillae_mask) + carmine_dot * papillae_mask
    
    # Translucency: petal edges (t -> ±1) and throat are thinner and transmit more light
    translucency = 0.24 + 0.38 * (abs_t ** 2) + 0.32 * throat_factor
    roughness = 0.35 - 0.08 * throat_factor
    
    return np.clip(color, 0.0, 1.0), float(translucency), float(roughness)


def evaluate_leaf_surface_color(
    uv: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluates procedural texture for lanceolate lily leaves at coordinate (s, t).
    """
    s = np.clip(uv[0], 0.0, 1.0)
    t = np.clip(uv[1], -1.0, 1.0)
    abs_t = abs(t)
    
    # Deep botanical waxy green
    deep_green = np.array([0.13, 0.30, 0.11], dtype=np.float32)
    # Lighter midrib vein running down the center
    midrib_light = np.array([0.28, 0.48, 0.20], dtype=np.float32)
    midrib_factor = np.exp(-36.0 * (t ** 2))
    
    # Longitudinal leaf striping
    stripe = 0.04 * np.sin(28.0 * t)
    
    color = deep_green * (1.0 - midrib_factor) + midrib_light * midrib_factor
    color = color + stripe
    
    return np.clip(color, 0.0, 1.0), 0.12, 0.28


def get_material_by_id(mat_id: int, uv: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Returns (albedo, roughness, specular_k, subsurface_k) for a given surface point.
    Material IDs:
    1: Inner Lily Petal
    2: Outer Lily Sepal
    3: Stamen Filament
    4: Pollen-dusted Anther
    5: Pistil & Stigma
    6: Stem
    7: Leaf
    """
    if mat_id == 1:
        albedo, trans, rough = evaluate_petal_surface_color(uv, is_inner=True)
        return albedo, rough, 0.22, trans
    elif mat_id == 2:
        albedo, trans, rough = evaluate_petal_surface_color(uv, is_inner=False)
        return albedo, rough, 0.24, trans * 0.9
    elif mat_id == 3:
        # Filament: pale succulent lime-white
        return np.array([0.82, 0.92, 0.74], dtype=np.float32), 0.35, 0.20, 0.30
    elif mat_id == 4:
        # Anther: rich golden-orange pollen dust
        # Oriental lilies have striking ochre-amber-orange pollen
        pollen_base = np.array([0.84, 0.45, 0.06], dtype=np.float32)
        return pollen_base, 0.65, 0.08, 0.05
    elif mat_id == 5:
        # Pistil style & stigma: fresh lime-pistachio green
        return np.array([0.70, 0.84, 0.54], dtype=np.float32), 0.32, 0.22, 0.25
    elif mat_id == 6:
        # Stem: firm botanical green
        return np.array([0.22, 0.44, 0.16], dtype=np.float32), 0.36, 0.18, 0.15
    elif mat_id == 7:
        albedo, trans, rough = evaluate_leaf_surface_color(uv)
        return albedo, rough, 0.32, trans
    else:
        # Fallback default
        return np.array([0.9, 0.9, 0.9], dtype=np.float32), 0.4, 0.2, 0.2
