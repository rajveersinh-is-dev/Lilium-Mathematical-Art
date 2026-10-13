"""
renderer.py - High-resolution parallel Z-buffer rasterizer and PBR shading pipeline.

Features:
- Analytical perspective camera with macro botanical lens parameters.
- Double-sided triangle rasterizer ensuring reflexed petals and foliage sheets render with correct normals.
- High-performance Numba-compiled PBR shading pipeline evaluating:
  - Ivory white petal albedo with celadon green throat gradient.
  - Midrib translucent groove and branching longitudinal veins.
  - Micro-papillae nectar dots near floral throat.
  - Golden-orange pollen-dusted anther texture.
  - Multi-light illumination (key, fill, rim) with forward subsurface transmission.
  - Depth ambient occlusion and filmic tone mapping.
"""

from typing import Dict, Any, Tuple
import numpy as np
import numba

from src.geometry import normalize
from src.lighting import StudioLighting, evaluate_background_field


@numba.njit(fastmath=True)
def rasterize_triangles_twosided(
    screen_v: np.ndarray,      # (N_tri, 3, 3) -> (tri, vertex, [x, y, z])
    normals_v: np.ndarray,     # (N_tri, 3, 3)
    uvs_v: np.ndarray,         # (N_tri, 3, 2)
    mat_ids_v: np.ndarray,     # (N_tri,)
    width: int,
    height: int,
    z_buffer: np.ndarray,      # (H, W)
    out_normals: np.ndarray,   # (H, W, 3)
    out_uvs: np.ndarray,       # (H, W, 2)
    out_mat_ids: np.ndarray    # (H, W)
):
    """
    Z-buffer rasterizer supporting two-sided thin botanical surfaces.
    Fills z_buffer, out_normals, out_uvs, and out_mat_ids.
    """
    n_tri = screen_v.shape[0]
    
    for i in range(n_tri):
        v0x = screen_v[i, 0, 0]
        v0y = screen_v[i, 0, 1]
        v0z = screen_v[i, 0, 2]
        
        v1x = screen_v[i, 1, 0]
        v1y = screen_v[i, 1, 1]
        v1z = screen_v[i, 1, 2]
        
        v2x = screen_v[i, 2, 0]
        v2y = screen_v[i, 2, 1]
        v2z = screen_v[i, 2, 2]
        
        # Triangle area in screen space
        area = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
        if abs(area) < 1e-4:
            continue  # Degenerate sliver
            
        flip_normal = False
        n0x = normals_v[i, 0, 0]
        n0y = normals_v[i, 0, 1]
        n0z = normals_v[i, 0, 2]
        n1x = normals_v[i, 1, 0]
        n1y = normals_v[i, 1, 1]
        n1z = normals_v[i, 1, 2]
        n2x = normals_v[i, 2, 0]
        n2y = normals_v[i, 2, 1]
        n2z = normals_v[i, 2, 2]
        
        uv0u = uvs_v[i, 0, 0]
        uv0v = uvs_v[i, 0, 1]
        uv1u = uvs_v[i, 1, 0]
        uv1v = uvs_v[i, 1, 1]
        uv2u = uvs_v[i, 2, 0]
        uv2v = uvs_v[i, 2, 1]
        
        # If triangle is viewed from the back, swap vertices to orient frontally
        if area < 0.0:
            v1x, v2x = v2x, v1x
            v1y, v2y = v2y, v1y
            v1z, v2z = v2z, v1z
            n1x, n2x = n2x, n1x
            n1y, n2y = n2y, n1y
            n1z, n2z = n2z, n1z
            uv1u, uv2u = uv2u, uv1u
            uv1v, uv2v = uv2v, uv1v
            area = -area
            flip_normal = True
            
        inv_area = 1.0 / area
        
        # Bounding box
        min_x = max(0, int(min(v0x, min(v1x, v2x))))
        max_x = min(width - 1, int(max(v0x, max(v1x, v2x))) + 1)
        min_y = max(0, int(min(v0y, min(v1y, v2y))))
        max_y = min(height - 1, int(max(v0y, max(v1y, v2y))) + 1)
        
        mat_id = mat_ids_v[i]
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                p_x = float(px) + 0.5
                p_y = float(py) + 0.5
                
                # Barycentric coordinates
                w0 = (v2x - v1x) * (p_y - v1y) - (v2y - v1y) * (p_x - v1x)
                w1 = (v0x - v2x) * (p_y - v2y) - (v0y - v2y) * (p_x - v2x)
                w2 = (v1x - v0x) * (p_y - v0y) - (v1y - v0y) * (p_x - v0x)
                
                if w0 >= -1e-5 and w1 >= -1e-5 and w2 >= -1e-5:
                    b0 = w0 * inv_area
                    b1 = w1 * inv_area
                    b2 = w2 * inv_area
                    
                    z = b0 * v0z + b1 * v1z + b2 * v2z
                    
                    # Depth test (smaller z is closer to camera)
                    if z < z_buffer[py, px]:
                        z_buffer[py, px] = z
                        out_mat_ids[py, px] = mat_id
                        
                        # Interpolate normal
                        nx = b0 * n0x + b1 * n1x + b2 * n2x
                        ny = b0 * n0y + b1 * n1y + b2 * n2y
                        nz = b0 * n0z + b1 * n1z + b2 * n2z
                        if flip_normal:
                            nx = -nx
                            ny = -ny
                            nz = -nz
                            
                        norm_len = (nx * nx + ny * ny + nz * nz) ** 0.5
                        if norm_len > 1e-6:
                            out_normals[py, px, 0] = nx / norm_len
                            out_normals[py, px, 1] = ny / norm_len
                            out_normals[py, px, 2] = nz / norm_len
                            
                        # Interpolate UV
                        out_uvs[py, px, 0] = b0 * uv0u + b1 * uv1u + b2 * uv2u
                        out_uvs[py, px, 1] = b0 * uv0v + b1 * uv1v + b2 * uv2v


@numba.njit(fastmath=True)
def shade_scene_numba(
    height: int,
    width: int,
    out_mat_ids: np.ndarray,
    out_normals: np.ndarray,
    out_uvs: np.ndarray,
    z_buffer: np.ndarray,
    rendered_rgb: np.ndarray,
    key_dir: np.ndarray,
    key_col: np.ndarray,
    key_intensity: float,
    fill_dir: np.ndarray,
    fill_col: np.ndarray,
    fill_intensity: float,
    rim_dir: np.ndarray,
    rim_col: np.ndarray,
    rim_intensity: float,
    ambient_col: np.ndarray
):
    """Numba-accelerated PBR shading and optical synthesis across all pixels."""
    l1_c0 = key_col[0] * key_intensity
    l1_c1 = key_col[1] * key_intensity
    l1_c2 = key_col[2] * key_intensity
    
    l2_c0 = fill_col[0] * fill_intensity
    l2_c1 = fill_col[1] * fill_intensity
    l2_c2 = fill_col[2] * fill_intensity
    
    l3_c0 = rim_col[0] * rim_intensity
    l3_c1 = rim_col[1] * rim_intensity
    l3_c2 = rim_col[2] * rim_intensity
    
    # View direction towards camera (+Z)
    vx = 0.0
    vy = 0.0
    vz = 1.0
    
    # Half vectors for specular reflection
    h1_x = key_dir[0] + vx
    h1_y = key_dir[1] + vy
    h1_z = key_dir[2] + vz
    len_h1 = max(1e-6, (h1_x**2 + h1_y**2 + h1_z**2)**0.5)
    h1_x /= len_h1; h1_y /= len_h1; h1_z /= len_h1
    
    h2_x = fill_dir[0] + vx
    h2_y = fill_dir[1] + vy
    h2_z = fill_dir[2] + vz
    len_h2 = max(1e-6, (h2_x**2 + h2_y**2 + h2_z**2)**0.5)
    h2_x /= len_h2; h2_y /= len_h2; h2_z /= len_h2
    
    h3_x = rim_dir[0] + vx
    h3_y = rim_dir[1] + vy
    h3_z = rim_dir[2] + vz
    len_h3 = max(1e-6, (h3_x**2 + h3_y**2 + h3_z**2)**0.5)
    h3_x /= len_h3; h3_y /= len_h3; h3_z /= len_h3
    
    for py in range(height):
        for px in range(width):
            m_id = out_mat_ids[py, px]
            if m_id == 0:
                continue
                
            u_val = out_uvs[py, px, 0]
            v_val = out_uvs[py, px, 1]
            s = min(1.0, max(0.0, u_val))
            t = min(1.0, max(-1.0, v_val))
            abs_t = abs(t)
            
            nx = out_normals[py, px, 0]
            ny = out_normals[py, px, 1]
            nz = out_normals[py, px, 2]
            
            # Default material properties
            r_alb = 0.985
            g_alb = 0.975
            b_alb = 0.955
            rough = 0.38
            spec_k = 0.22
            trans_k = 0.25
            
            if m_id == 1 or m_id == 2:  # Inner Lily Petal or Outer Sepal
                # Base pristine ivory white with subtle cream warmth
                r_alb = 0.985
                g_alb = 0.980
                b_alb = 0.965
                
                # Vivid celadon / chartreuse throat star gradient
                throat_factor = np.exp(-7.5 * (s ** 1.4))
                r_alb = r_alb * (1.0 - throat_factor) + 0.52 * throat_factor
                g_alb = g_alb * (1.0 - throat_factor) + 0.84 * throat_factor
                b_alb = b_alb * (1.0 - throat_factor) + 0.28 * throat_factor
                
                # Midrib translucent groove running longitudinally
                midrib = (1.0 - 0.45 * s) * np.exp(-26.0 * (t * t))
                r_alb = r_alb * (1.0 - 0.45 * midrib) + 0.62 * (0.45 * midrib)
                g_alb = g_alb * (1.0 - 0.45 * midrib) + 0.84 * (0.45 * midrib)
                b_alb = b_alb * (1.0 - 0.45 * midrib) + 0.38 * (0.45 * midrib)
                
                # Longitudinal vein modulation
                vein_mod = 1.0
                for k in range(1, 5):
                    tau = 0.22 * float(k) * (1.0 - 0.15 * s)
                    vein_mod += 0.04 * np.exp(-((abs_t - tau) ** 2) / 0.002)
                r_alb *= (0.97 + 0.03 * vein_mod)
                g_alb *= (0.97 + 0.03 * vein_mod)
                b_alb *= (0.97 + 0.03 * vein_mod)
                
                # Carmine papillae (raised nectar specks near throat)
                if 0.12 < s < 0.38 and abs_t < 0.48:
                    dot = np.sin(42.0 * s) * np.cos(34.0 * t)
                    if dot > 0.65:
                        p_mask = (dot - 0.65) / 0.35 * (0.38 - s) / 0.26 * (1.0 - abs_t / 0.48)
                        r_alb = r_alb * (1.0 - p_mask) + 0.74 * p_mask
                        g_alb = g_alb * (1.0 - p_mask) + 0.18 * p_mask
                        b_alb = b_alb * (1.0 - p_mask) + 0.28 * p_mask
                        
                trans_k = 0.24 + 0.38 * (abs_t ** 2) + 0.32 * throat_factor
                rough = 0.35 - 0.08 * throat_factor
                spec_k = 0.24 if m_id == 1 else 0.26
                
            elif m_id == 3:  # Stamen Filament
                r_alb = 0.82
                g_alb = 0.92
                b_alb = 0.74
                rough = 0.35
                spec_k = 0.20
                trans_k = 0.30
                
            elif m_id == 4:  # Golden-Orange Pollen Anther
                r_alb = 0.84
                g_alb = 0.45
                b_alb = 0.06
                rough = 0.65
                spec_k = 0.08
                trans_k = 0.05
                
            elif m_id == 5:  # Pistil / Stigma
                r_alb = 0.70
                g_alb = 0.84
                b_alb = 0.54
                rough = 0.32
                spec_k = 0.22
                trans_k = 0.25
                
            elif m_id == 6:  # Stem
                r_alb = 0.22
                g_alb = 0.44
                b_alb = 0.16
                rough = 0.36
                spec_k = 0.18
                trans_k = 0.15
                
            elif m_id == 7:  # Leaf
                r_alb = 0.13
                g_alb = 0.30
                b_alb = 0.11
                # Midrib line
                mid_l = np.exp(-36.0 * (t * t))
                r_alb = r_alb * (1.0 - mid_l) + 0.28 * mid_l
                g_alb = g_alb * (1.0 - mid_l) + 0.48 * mid_l
                b_alb = b_alb * (1.0 - mid_l) + 0.20 * mid_l
                # Longitudinal stripe
                strp = 0.04 * np.sin(28.0 * t)
                r_alb = max(0.0, min(1.0, r_alb + strp))
                g_alb = max(0.0, min(1.0, g_alb + strp))
                b_alb = max(0.0, min(1.0, b_alb + strp))
                rough = 0.28
                spec_k = 0.32
                trans_k = 0.12
                
            # Depth ambient occlusion
            depth_val = z_buffer[py, px]
            ao = max(0.60, min(1.0, 1.18 - 0.10 * (depth_val - 2.5)))
            
            # Perianth cavity AO: deep throat of lily flower receives natural occlusion
            if m_id == 1 or m_id == 2:
                cavity_ao = 0.58 + 0.42 * (s ** 0.85)
                ao *= cavity_ao
                if m_id == 2:  # Outer sepals receive subtle shadow from forward inner petals
                    ao *= 0.92
            
            # Ambient light
            c_r = r_alb * ambient_col[0] * ao
            c_g = g_alb * ambient_col[1] * ao
            c_b = b_alb * ambient_col[2] * ao
            
            spec_exp = max(6.0, (1.0 - rough) * 80.0)
            
            # Light 1: Key
            n_dot_l1 = nx * key_dir[0] + ny * key_dir[1] + nz * key_dir[2]
            d1 = max(0.0, n_dot_l1)
            t1 = max(0.0, -n_dot_l1) * trans_k * 0.48
            n_dot_h1 = max(0.0, nx * h1_x + ny * h1_y + nz * h1_z)
            s1 = (n_dot_h1 ** spec_exp) * spec_k if d1 > 0.0 else 0.0
            c_r += l1_c0 * (r_alb * (d1 + t1) + s1)
            c_g += l1_c1 * (g_alb * (d1 + t1) + s1)
            c_b += l1_c2 * (b_alb * (d1 + t1) + s1)
            
            # Light 2: Fill
            n_dot_l2 = nx * fill_dir[0] + ny * fill_dir[1] + nz * fill_dir[2]
            d2 = max(0.0, n_dot_l2)
            t2 = max(0.0, -n_dot_l2) * trans_k * 0.45
            n_dot_h2 = max(0.0, nx * h2_x + ny * h2_y + nz * h2_z)
            s2 = (n_dot_h2 ** spec_exp) * spec_k if d2 > 0.0 else 0.0
            c_r += l2_c0 * (r_alb * (d2 + t2) + s2)
            c_g += l2_c1 * (g_alb * (d2 + t2) + s2)
            c_b += l2_c2 * (b_alb * (d2 + t2) + s2)
            
            # Light 3: Rim / Backlight (produces edge luminescence & thin petal translucency)
            n_dot_l3 = nx * rim_dir[0] + ny * rim_dir[1] + nz * rim_dir[2]
            d3 = max(0.0, n_dot_l3)
            t3 = max(0.0, -n_dot_l3) * trans_k * 0.55
            n_dot_h3 = max(0.0, nx * h3_x + ny * h3_y + nz * h3_z)
            s3 = (n_dot_h3 ** spec_exp) * spec_k if d3 > 0.0 else 0.0
            
            # Fresnel grazing-angle rim glow along thin petal margins
            fresnel_rim = (1.0 - min(1.0, abs(nz))) ** 3.0
            rim_edge_glow = fresnel_rim * trans_k * 0.35
            
            c_r += l3_c0 * (r_alb * (d3 + t3 + rim_edge_glow) + s3)
            c_g += l3_c1 * (g_alb * (d3 + t3 + rim_edge_glow) + s3)
            c_b += l3_c2 * (b_alb * (d3 + t3 + rim_edge_glow) + s3)
            
            rendered_rgb[py, px, 0] = c_r
            rendered_rgb[py, px, 1] = c_g
            rendered_rgb[py, px, 2] = c_b


class LilyRenderer:
    """
    High-fidelity botanical image synthesizer.
    Renders 3D geometry into RGB image with PBR lighting and filmic tone curve.
    """
    def __init__(
        self,
        width: int = 2400,
        height: int = 1440,
        fov_deg: float = 38.0,
        camera_pos: np.ndarray = None,
        camera_target: np.ndarray = None
    ):
        self.width = width
        self.height = height
        self.fov = np.radians(fov_deg)
        self.camera_pos = (
            np.asarray(camera_pos, dtype=np.float64).reshape(3)
            if camera_pos is not None
            else np.array([0.0, 0.20, 4.35], dtype=np.float64)
        )
        self.camera_target = (
            np.asarray(camera_target, dtype=np.float64).reshape(3)
            if camera_target is not None
            else np.array([0.0, 0.12, 0.0], dtype=np.float64)
        )
        self.lighting = StudioLighting()

    def project_to_screen(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D world vertices to 2D screen coordinates and camera space depth."""
        cam_dir = normalize(self.camera_target - self.camera_pos)
        cam_up_ref = np.array([0.0, 1.0, 0.0])
        cam_right = normalize(np.cross(cam_dir, cam_up_ref))
        cam_up = normalize(np.cross(cam_right, cam_dir))
        
        v_rel = vertices - self.camera_pos[None, :]
        v_x = np.sum(v_rel * cam_right[None, :], axis=-1)
        v_y = np.sum(v_rel * cam_up[None, :], axis=-1)
        v_z = np.sum(v_rel * cam_dir[None, :], axis=-1)
        
        aspect = self.width / self.height
        focal = 1.0 / np.tan(self.fov * 0.5)
        
        z_safe = np.maximum(v_z, 0.01)
        ndc_x = (v_x / z_safe) * (focal / aspect)
        ndc_y = (v_y / z_safe) * focal
        
        screen_x = (ndc_x * 0.5 + 0.5) * self.width
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height
        
        screen_coords = np.stack([screen_x, screen_y, v_z], axis=-1)
        valid = v_z > 0.1
        return screen_coords, valid

    def render(self, scene_geom: Dict[str, Any]) -> np.ndarray:
        """
        Executes complete rendering pipeline:
        1. Projects vertices to screen coordinates.
        2. Two-sided triangle rasterization with Z-buffering.
        3. Numba-accelerated optical PBR shading with translucency.
        4. Tone mapping and gamma correction to 8-bit RGB array.
        """
        verts = scene_geom["vertices"]
        faces = scene_geom["faces"]
        normals = scene_geom["normals"]
        uvs = scene_geom["uvs"]
        mat_ids = scene_geom["mat_ids"]
        
        screen_coords, _ = self.project_to_screen(verts)
        
        tri_screen = screen_coords[faces]   # (F, 3, 3)
        tri_normals = normals[faces]        # (F, 3, 3)
        tri_uvs = uvs[faces]                # (F, 3, 2)
        tri_mat_ids = mat_ids[faces[:, 0]]  # (F,)
        
        z_buffer = np.full((self.height, self.width), 1e9, dtype=np.float32)
        out_normals = np.zeros((self.height, self.width, 3), dtype=np.float32)
        out_uvs = np.zeros((self.height, self.width, 2), dtype=np.float32)
        out_mat_ids = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Execute two-sided rasterization
        rasterize_triangles_twosided(
            tri_screen.astype(np.float32),
            tri_normals.astype(np.float32),
            tri_uvs.astype(np.float32),
            tri_mat_ids.astype(np.int32),
            self.width,
            self.height,
            z_buffer,
            out_normals,
            out_uvs,
            out_mat_ids
        )
        
        # Precompute background field
        u_coords = (np.arange(self.width) + 0.5) / self.width
        v_coords = (np.arange(self.height) + 0.5) / self.height
        U, V = np.meshgrid(u_coords, v_coords)
        bg_image = evaluate_background_field(U, V)
        
        rendered_rgb = bg_image.copy()
        
        # Execute compiled PBR shading
        shade_scene_numba(
            self.height,
            self.width,
            out_mat_ids,
            out_normals,
            out_uvs,
            z_buffer,
            rendered_rgb,
            self.lighting.key.direction.astype(np.float32),
            self.lighting.key.color.astype(np.float32),
            self.lighting.key.intensity,
            self.lighting.fill.direction.astype(np.float32),
            self.lighting.fill.color.astype(np.float32),
            self.lighting.fill.intensity,
            self.lighting.rim.direction.astype(np.float32),
            self.lighting.rim.color.astype(np.float32),
            self.lighting.rim.intensity,
            self.lighting.ambient_radiance.astype(np.float32)
        )
        
        # Filmic Tone Mapping (Reinhard-ACES hybrid) with calibrated exposure & Gamma 2.2
        exposed_rgb = rendered_rgb * 1.35
        l_white = 1.65
        tone_mapped = (exposed_rgb * (1.0 + exposed_rgb / (l_white * l_white))) / (1.0 + exposed_rgb)
        gamma_corrected = np.power(np.clip(tone_mapped, 0.0, 1.0), 1.0 / 2.2)
        
        return (gamma_corrected * 255.0).astype(np.uint8)
