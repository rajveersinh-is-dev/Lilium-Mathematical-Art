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
        vein_modulation += 0.03 * np.exp(-((abs_t - tau) ** 2) / 0.005)
    
    # 5. Carmine Papillae (Nectar ridges / delicate pink-red specks near throat)
    # Characteristic of Lilium auratum / oriental hybrids: raised crimson-pink dots near s in [0.12, 0.38]
    # Scaled down intensity for a more natural look (less artificial noise).
    papillae_mask = 0.0
    if 0.12 < s < 0.38 and abs_t < 0.48:
        dot = np.sin(55.0 * s) * np.cos(45.0 * t)
        if dot > 0.85:
            papillae_mask = (dot - 0.85) / 0.15 * (0.38 - s) / 0.26 * (1.0 - abs_t / 0.48)
            papillae_mask *= 0.4  # dampen the effect so it doesn't look like noisy render artifacts
            
    carmine_dot = np.array([0.70, 0.22, 0.32], dtype=np.float32)
    
    # Blend components
    color = ivory_white * (1.0 - throat_factor) + celadon_green * throat_factor
    color = color * (1.0 - 0.35 * midrib_factor) + midrib_green * (0.35 * midrib_factor)
    color = color * (0.97 + 0.03 * vein_modulation)
    color = color * (1.0 - papillae_mask) + carmine_dot * papillae_mask
    
    # Translucency: petal edges (t -> ±1) and throat are thinner and transmit more light
    # Slightly higher translucency for softer realism.
    translucency = 0.30 + 0.40 * (abs_t ** 2) + 0.28 * throat_factor
    roughness = 0.42 - 0.10 * throat_factor  # Sligthly rougher overall so it's less plastic-like
    
    return np.clip(color, 0.0, 1.0), float(translucency), float(roughness)


def evaluate_leaf_surface_color(
    uv: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluates procedural texture for lanceolate lily leaves at coordinate (s, t).
    """
    s = float(np.clip(uv[0], 0.0, 1.0))
    t = float(np.clip(uv[1], -1.0, 1.0))
    abs_t = abs(t)
    
    # Deep botanical waxy forest green (lightened to avoid artificial black-green)
    deep_green = np.array([0.05, 0.16, 0.05], dtype=np.float32)
    # Lighter spring-green midrib vein running down the center
    midrib_light = np.array([0.10, 0.26, 0.08], dtype=np.float32)
    midrib_factor = float(np.exp(-32.0 * (t ** 2)))
    
    # Secondary Monocot Parallel Veins
    v1 = np.exp(-180.0 * ((abs_t - 0.20) ** 2))
    v2 = np.exp(-180.0 * ((abs_t - 0.40) ** 2))
    v3 = np.exp(-180.0 * ((abs_t - 0.60) ** 2))
    v4 = np.exp(-180.0 * ((abs_t - 0.78) ** 2))
    v5 = np.exp(-180.0 * ((abs_t - 0.92) ** 2))
    sec_veins = float((v1 + v2 + v3 + v4 + v5) * 0.006)  # Reduced intensity for realism
    
    # Parallel cellular striations
    striation = float(0.003 * np.sin(160.0 * t) * (1.0 - 0.3 * s))
    
    color = deep_green * (1.0 - midrib_factor) + midrib_light * midrib_factor
    color[1] += sec_veins + striation
    color[0] += 0.35 * sec_veins
    
    # Translucent margins
    if abs_t > 0.78:
        edge_w = (abs_t - 0.78) / 0.22
        color[1] += 0.045 * edge_w
        color[0] += 0.018 * edge_w
        trans = 0.14 + 0.26 * edge_w
    else:
        trans = 0.10 + 0.06 * (s ** 2)
        
    roughness = 0.18 * (1.0 - midrib_factor) + 0.28 * midrib_factor
    return np.clip(color, 0.0, 1.0), float(trans), float(roughness)


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
        return np.array([0.78, 0.88, 0.75], dtype=np.float32), 0.45, 0.15, 0.30
    elif mat_id == 4:
        # Anther: rich golden-orange pollen dust
        # Oriental lilies have striking ochre-amber-orange pollen
        # Muted ochre/rust for realism, rather than bright neon orange
        pollen_base = np.array([0.62, 0.30, 0.12], dtype=np.float32)
        return pollen_base, 0.75, 0.05, 0.05
    elif mat_id == 5:
        # Pistil style & stigma: fresh lime-pistachio green
        return np.array([0.70, 0.84, 0.54], dtype=np.float32), 0.32, 0.22, 0.25
    elif mat_id == 6:
        # Stem: firm botanical green
        return np.array([0.18, 0.38, 0.15], dtype=np.float32), 0.45, 0.18, 0.25
    elif mat_id == 7:
        albedo, trans, rough = evaluate_leaf_surface_color(uv)
        # Reduced artificial specular highlight, returning higher roughness
        return albedo, rough, 0.15, trans
    elif mat_id == 8:
        # Lily Bud (Jade/celadon base transitioning to creamy ivory tip)
        return np.array([0.55, 0.78, 0.52], dtype=np.float32), 0.22, 0.44, 0.35
    elif mat_id == 9:
        # Artisan Florist Twine / Binding Threads (Natural spun linen/raffia fiber)
        return np.array([0.76, 0.67, 0.50], dtype=np.float32), 0.68, 0.14, 0.06
    else:
        # Fallback default
        return np.array([0.9, 0.9, 0.9], dtype=np.float32), 0.4, 0.2, 0.2
