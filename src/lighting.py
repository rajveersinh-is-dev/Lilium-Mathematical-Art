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
        # 1. Key light: soft, diffused studio softbox illumination from upper-left
        self.key = DirectionalLight(
            direction=np.array([-0.65, 0.55, 0.52]),
            color=np.array([1.0, 0.98, 0.94]),
            intensity=1.15  # Reduced intensity for softer lighting, avoiding overexposure
        )
        # 2. Fill light: soft cool ambient fill from lower-right
        self.fill = DirectionalLight(
            direction=np.array([0.55, 0.35, 0.76]),
            color=np.array([0.90, 0.95, 1.0]),
            intensity=0.75  # Increased fill to retain detail in shaded petals
        )
        # 3. Rim / Back light: edge backlight from behind-above for petal and leaf translucency
        self.rim = DirectionalLight(
            direction=np.array([0.15, 0.85, -0.50]),
            color=np.array([0.96, 0.98, 0.92]),
            intensity=0.45  # Considerably reduced rim light to prevent harsh, high-contrast streaks
        )
        # Ambient color: heightened soft botanical ambient fill
        self.ambient_radiance = np.array([0.12, 0.14, 0.12], dtype=np.float32)

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
    Evaluates soft botanical studio background tone:
    A restrained, realistic dark-green/gray botanical background providing contrast
    without appearing as a flat, artificial black void.
    - u, v in [0, 1]
    """
    # Radial distance from floral focal point (upper center of bouquet)
    dx = u - 0.50
    dy = v - 0.38
    r2 = dx * dx * 1.25 + dy * dy
    
    # Soft Gaussian light pool behind bouquet
    light_pool = np.exp(-2.5 * r2)[:, :, None]
    
    # Colors: restrained dark-green studio backdrop, significantly lighter than black void
    center_color = np.array([0.065, 0.082, 0.070], dtype=np.float32).reshape(1, 1, 3)
    edge_color = np.array([0.025, 0.035, 0.030], dtype=np.float32).reshape(1, 1, 3)
    
    # Soft vertical atmospheric gradation
    vert_falloff = (0.85 + 0.25 * (1.0 - v))[:, :, None]
    
    bg_rgb = (center_color * light_pool + edge_color * (1.0 - light_pool)) * vert_falloff
    return np.clip(bg_rgb, 0.0, 1.0)

