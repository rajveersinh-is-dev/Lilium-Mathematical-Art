"""
lighting.py - Multi-light studio illumination, microfacet BRDF, and background field.

Provides:
- Directional key, fill, and rim lights.
- Diffuse, microfacet specular, and forward-scattering translucency shading.
- Background chiaroscuro radiance field for photographic botanical depth.
"""

from typing import List, Tuple
import numpy as np

from src.geometry import normalize


class DirectionalLight:
    """Directional light source with color and intensity."""
    def __init__(self, direction: np.ndarray, color: np.ndarray, intensity: float):
        self.direction = normalize(np.asarray(direction, dtype=np.float32).reshape(3))
        self.color = np.asarray(color, dtype=np.float32).reshape(3)
        self.intensity = float(intensity)


class StudioLighting:
    """Three-point studio lighting with botanical chiaroscuro environment."""
    def __init__(self):
        # 1. Key light: sculpted directional illumination from upper-left (skims across petal curves)
        self.key = DirectionalLight(
            direction=normalize(np.array([-0.52, 0.72, 0.46])),
            color=np.array([1.00, 0.975, 0.940]),
            intensity=1.45
        )
        # 2. Fill light: soft cool fill from lower-right (fill:key ~ 0.43 for open shadows)
        self.fill = DirectionalLight(
            direction=normalize(np.array([0.60, 0.22, 0.76])),
            color=np.array([0.88, 0.92, 1.00]),
            intensity=0.62
        )
        # 3. Rim / Back light: gentle edge backlight (kept low so edges do not blow out)
        self.rim = DirectionalLight(
            direction=normalize(np.array([0.10, 0.80, -0.58])),
            color=np.array([0.95, 0.96, 0.92]),
            intensity=0.30
        )
        # Ambient color: refined chiaroscuro fill
        self.ambient_radiance = np.array([0.075, 0.082, 0.075], dtype=np.float32)

    def shade_surface(
        self,
        normal: np.ndarray,
        view_dir: np.ndarray,
        albedo: np.ndarray,
        roughness: float,
        specular_k: float,
        subsurface_k: float
    ) -> np.ndarray:
        """
        Calculates PBR surface radiance:
        I = I_ambient + sum_lights (I_diffuse + I_specular + I_subsurface)
        """
        n = normalize(normal)
        v = normalize(view_dir)
        
        # Ambient component
        color = albedo * self.ambient_radiance
        
        lights = [self.key, self.fill, self.rim]
        spec_exp = max(4.0, (1.0 - roughness) * 64.0)
        
        for light in lights:
            l_dir = light.direction
            l_color = light.color * light.intensity
            
            # Diffuse (Lambert)
            n_dot_l = float(np.dot(n, l_dir))
            diffuse_factor = max(0.0, n_dot_l)
            
            # Specular (Blinn-Phong)
            h = normalize(l_dir + v)
            n_dot_h = max(0.0, float(np.dot(n, h)))
            spec_factor = (n_dot_h ** spec_exp) if diffuse_factor > 0.0 else 0.0
            
            # Subsurface Translucency / Forward scattering
            # When light is behind the thin petal (n_dot_l < 0), light scatters forward towards viewer
            trans_factor = max(0.0, -n_dot_l) * subsurface_k
            
            # Combine light contribution
            diff_term = albedo * l_color * diffuse_factor
            spec_term = l_color * (specular_k * spec_factor)
            trans_term = albedo * l_color * (trans_factor * 0.45)
            
            color += diff_term + spec_term + trans_term
            
        return color


def evaluate_background_field(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Evaluates deep botanical chiaroscuro background tone:
    A museum-quality dark charcoal-forest green with a soft, warm botanical light pool
    illuminating the space behind the white-lily bouquet.
    - u, v in [0, 1]
    """
    # Radial distance from floral focal point (upper center of bouquet)
    dx = u - 0.50
    dy = v - 0.38
    r2 = dx * dx * 1.25 + dy * dy
    
    # Soft Gaussian light pool behind bouquet
    light_pool = np.exp(-3.2 * r2)[:, :, None]
    
    # Colors: museum-quality velvety dark charcoal-sage with soft warm botanical luminance
    center_color = np.array([0.040, 0.048, 0.040], dtype=np.float32).reshape(1, 1, 3)
    edge_color = np.array([0.009, 0.011, 0.009], dtype=np.float32).reshape(1, 1, 3)
    
    # Soft vertical atmospheric gradation
    vert_falloff = (0.92 + 0.16 * (1.0 - v))[:, :, None]
    
    bg_rgb = (center_color * light_pool + edge_color * (1.0 - light_pool)) * vert_falloff
    return np.clip(bg_rgb, 0.0, 1.0)

