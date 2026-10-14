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
        # 1. Key light: warm directional illumination from upper-left in front
        self.key = DirectionalLight(
            direction=np.array([-0.42, 0.65, 0.63]),
            color=np.array([1.0, 0.99, 0.96]),
            intensity=1.25
        )
        # 2. Fill light: cool diffused light from upper-right
        self.fill = DirectionalLight(
            direction=np.array([0.55, 0.38, 0.74]),
            color=np.array([0.88, 0.94, 1.0]),
            intensity=0.55
        )
        # 3. Rim / Back light: edge backlight from behind-above for petal translucency
        self.rim = DirectionalLight(
            direction=np.array([0.20, 0.72, -0.66]),
            color=np.array([0.96, 0.98, 0.92]),
            intensity=0.45
        )
        # Ambient color: soft botanical fill
        self.ambient_radiance = np.array([0.15, 0.17, 0.15], dtype=np.float32)

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
    dy = v - 0.40
    r2 = dx * dx + dy * dy
    
    # Soft Gaussian light pool behind bouquet
    light_pool = np.exp(-3.5 * r2)[:, :, None]
    
    # Colors: warm deep olive-charcoal center, velvety near-black forest green edges
    center_color = np.array([0.038, 0.055, 0.036], dtype=np.float32).reshape(1, 1, 3)
    edge_color = np.array([0.009, 0.015, 0.011], dtype=np.float32).reshape(1, 1, 3)
    
    # Soft vertical atmospheric gradation
    vert_falloff = (0.92 + 0.16 * (1.0 - v))[:, :, None]
    
    bg_rgb = (center_color * light_pool + edge_color * (1.0 - light_pool)) * vert_falloff
    return np.clip(bg_rgb, 0.0, 1.0)
