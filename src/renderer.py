"""
renderer.py - High-resolution parallel Z-buffer rasterizer and photorealistic PBR shading pipeline.

Features:
- Analytical perspective camera with macro botanical lens parameters.
- Double-sided triangle rasterizer tracking 3D world positions, surface normals, and UVs.
- Key-light orthographic shadow map rasterization with PCF soft contact shadows.
- Screen-Space Ambient Occlusion (SSAO) with cross-bilateral depth-aware filtering.
- Dual-lobe Cook-Torrance GGX microfacet specular BRDF and Fresnel reflectance.
- Analytical micro-texture bump mapping: fibrovascular veins, epidermal cells, and powdery pollen.
- Forward thin-sheet subsurface transmission and filmic Reinhard-ACES tone mapping.
"""

from typing import Dict, Any, Tuple
import numpy as np
import numba

from src.geometry import normalize
from src.lighting import StudioLighting, evaluate_background_field


@numba.njit(fastmath=True)
def rasterize_triangles_twosided(
    screen_v: np.ndarray,      # (N_tri, 3, 3) -> (tri, vertex, [x, y, z])
    world_v: np.ndarray,       # (N_tri, 3, 3)
    normals_v: np.ndarray,     # (N_tri, 3, 3)
    uvs_v: np.ndarray,         # (N_tri, 3, 2)
    mat_ids_v: np.ndarray,     # (N_tri,)
    width: int,
    height: int,
    z_buffer: np.ndarray,      # (H, W)
    out_world: np.ndarray,     # (H, W, 3)
    out_normals: np.ndarray,   # (H, W, 3)
    out_uvs: np.ndarray,       # (H, W, 2)
    out_mat_ids: np.ndarray    # (H, W)
):
    """
    Z-buffer rasterizer supporting two-sided thin botanical surfaces.
    Fills z_buffer, out_world, out_normals, out_uvs, and out_mat_ids.
    """
    n_tri = screen_v.shape[0]
    
    for i in range(n_tri):
        v0x = screen_v[i, 0, 0]; v0y = screen_v[i, 0, 1]; v0z = screen_v[i, 0, 2]
        v1x = screen_v[i, 1, 0]; v1y = screen_v[i, 1, 1]; v1z = screen_v[i, 1, 2]
        v2x = screen_v[i, 2, 0]; v2y = screen_v[i, 2, 1]; v2z = screen_v[i, 2, 2]
        
        # Triangle area in screen space
        area = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
        if abs(area) < 1e-4:
            continue  # Degenerate sliver
            
        w0x = world_v[i, 0, 0]; w0y = world_v[i, 0, 1]; w0z = world_v[i, 0, 2]
        w1x = world_v[i, 1, 0]; w1y = world_v[i, 1, 1]; w1z = world_v[i, 1, 2]
        w2x = world_v[i, 2, 0]; w2y = world_v[i, 2, 1]; w2z = world_v[i, 2, 2]
        
        n0x = normals_v[i, 0, 0]; n0y = normals_v[i, 0, 1]; n0z = normals_v[i, 0, 2]
        n1x = normals_v[i, 1, 0]; n1y = normals_v[i, 1, 1]; n1z = normals_v[i, 1, 2]
        n2x = normals_v[i, 2, 0]; n2y = normals_v[i, 2, 1]; n2z = normals_v[i, 2, 2]
        
        uv0u = uvs_v[i, 0, 0]; uv0v = uvs_v[i, 0, 1]
        uv1u = uvs_v[i, 1, 0]; uv1v = uvs_v[i, 1, 1]
        uv2u = uvs_v[i, 2, 0]; uv2v = uvs_v[i, 2, 1]
        
        # If triangle is viewed from the back in screen coords, swap vertices to orient frontally
        if area < 0.0:
            v1x, v2x = v2x, v1x
            v1y, v2y = v2y, v1y
            v1z, v2z = v2z, v1z
            w1x, w2x = w2x, w1x
            w1y, w2y = w2y, w1y
            w1z, w2z = w2z, w1z
            n1x, n2x = n2x, n1x
            n1y, n2y = n2y, n1y
            n1z, n2z = n2z, n1z
            uv1u, uv2u = uv2u, uv1u
            uv1v, uv2v = uv2v, uv1v
            area = -area
            
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
                        
                        out_world[py, px, 0] = b0 * w0x + b1 * w1x + b2 * w2x
                        out_world[py, px, 1] = b0 * w0y + b1 * w1y + b2 * w2y
                        out_world[py, px, 2] = b0 * w0z + b1 * w1z + b2 * w2z
                        
                        # Interpolate normal
                        nx = b0 * n0x + b1 * n1x + b2 * n2x
                        ny = b0 * n0y + b1 * n1y + b2 * n2y
                        nz = b0 * n0z + b1 * n1z + b2 * n2z
                            
                        norm_len = (nx * nx + ny * ny + nz * nz) ** 0.5
                        if norm_len > 1e-6:
                            out_normals[py, px, 0] = nx / norm_len
                            out_normals[py, px, 1] = ny / norm_len
                            out_normals[py, px, 2] = nz / norm_len
                            
                        # Interpolate UV
                        out_uvs[py, px, 0] = b0 * uv0u + b1 * uv1u + b2 * uv2u
                        out_uvs[py, px, 1] = b0 * uv0v + b1 * uv1v + b2 * uv2v


@numba.njit(fastmath=True)
def rasterize_shadow_map_numba(
    tri_light: np.ndarray,     # (F, 3, 3) -> [sx, sy, z_L]
    w_s: int,
    h_s: int,
    shadow_map: np.ndarray     # (H_s, W_s)
):
    """Rasterizes all scene triangles into an orthographic depth map for directional light."""
    n_tri = tri_light.shape[0]
    for i in range(n_tri):
        v0x = tri_light[i, 0, 0]; v0y = tri_light[i, 0, 1]; v0z = tri_light[i, 0, 2]
        v1x = tri_light[i, 1, 0]; v1y = tri_light[i, 1, 1]; v1z = tri_light[i, 1, 2]
        v2x = tri_light[i, 2, 0]; v2y = tri_light[i, 2, 1]; v2z = tri_light[i, 2, 2]
        
        area = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
        if abs(area) < 1e-4:
            continue
        if area < 0.0:
            v1x, v2x = v2x, v1x
            v1y, v2y = v2y, v1y
            v1z, v2z = v2z, v1z
            area = -area
            
        inv_area = 1.0 / area
        min_x = max(0, int(min(v0x, min(v1x, v2x))))
        max_x = min(w_s - 1, int(max(v0x, max(v1x, v2x))) + 1)
        min_y = max(0, int(min(v0y, min(v1y, v2y))))
        max_y = min(h_s - 1, int(max(v0y, max(v1y, v2y))) + 1)
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                p_x = float(px) + 0.5
                p_y = float(py) + 0.5
                w0 = (v2x - v1x) * (p_y - v1y) - (v2y - v1y) * (p_x - v1x)
                w1 = (v0x - v2x) * (p_y - v2y) - (v0y - v2y) * (p_x - v2x)
                w2 = (v1x - v0x) * (p_y - v0y) - (v1y - v0y) * (p_x - v0x)
                if w0 >= -1e-5 and w1 >= -1e-5 and w2 >= -1e-5:
                    b0 = w0 * inv_area
                    b1 = w1 * inv_area
                    b2 = w2 * inv_area
                    z = b0 * v0z + b1 * v1z + b2 * v2z
                    if z < shadow_map[py, px]:
                        shadow_map[py, px] = z


@numba.njit(fastmath=True)
def compute_ssao_numba(
    height: int,
    width: int,
    z_buffer: np.ndarray,      # (H, W)
    out_normals: np.ndarray,   # (H, W, 3)
    out_mat_ids: np.ndarray,   # (H, W)
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    cam_dir: np.ndarray,
    focal: float,
    aspect: float,
    radius: float,
    kernel: np.ndarray,        # (N_samples, 3)
    ssao_raw: np.ndarray       # (H, W) output
):
    """Computes screen-space ambient occlusion using depth and normal buffers."""
    inv_focal = 1.0 / focal
    inv_aspect_focal = aspect / focal
    n_samples = kernel.shape[0]
    
    for py in range(height):
        for px in range(width):
            if out_mat_ids[py, px] == 0:
                ssao_raw[py, px] = 1.0
                continue
                
            pz = z_buffer[py, px]
            if pz >= 1e8:
                ssao_raw[py, px] = 1.0
                continue
                
            ndc_x = (float(px) + 0.5) / float(width) * 2.0 - 1.0
            ndc_y = 1.0 - (float(py) + 0.5) / float(height) * 2.0
            px_cam = ndc_x * pz * inv_aspect_focal
            py_cam = ndc_y * pz * inv_focal
            
            # Normal in world coordinates
            wnx = out_normals[py, px, 0]
            wny = out_normals[py, px, 1]
            wnz = out_normals[py, px, 2]
            
            # Transform normal to camera space
            cnx = wnx * cam_right[0] + wny * cam_right[1] + wnz * cam_right[2]
            cny = wnx * cam_up[0] + wny * cam_up[1] + wnz * cam_up[2]
            cnz = wnx * cam_dir[0] + wny * cam_dir[1] + wnz * cam_dir[2]
            
            # Ensure normal points toward camera (cnx, cny, cnz with cnz < 0 in +Z view)
            if cnz > 0.0:
                cnx = -cnx; cny = -cny; cnz = -cnz
                
            # Tangent & bitangent
            angle = ((float(px) * 12.9898 + float(py) * 78.233) % 3.14159) * 2.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            rx = cos_a; ry = sin_a; rz = 0.0
            dot_r_cn = rx * cnx + ry * cny + rz * cnz
            tx = rx - dot_r_cn * cnx
            ty = ry - dot_r_cn * cny
            tz = rz - dot_r_cn * cnz
            t_len = max(1e-6, (tx*tx + ty*ty + tz*tz)**0.5)
            tx /= t_len; ty /= t_len; tz /= t_len
            
            bx = cny * tz - cnz * ty
            by = cnz * tx - cnx * tz
            bz = cnx * ty - cny * tx
            
            occlusion = 0.0
            
            for k in range(n_samples):
                kx = kernel[k, 0]; ky = kernel[k, 1]; kz = kernel[k, 2]
                
                sx = tx * kx + bx * ky + cnx * kz
                sy = ty * kx + by * ky + cny * kz
                sz = tz * kx + bz * ky + cnz * kz
                
                sample_x = px_cam + sx * radius
                sample_y = py_cam + sy * radius
                sample_z = pz + sz * radius
                
                if sample_z <= 0.05:
                    continue
                    
                s_ndc_x = (sample_x / sample_z) * (focal / aspect)
                s_ndc_y = (sample_y / sample_z) * focal
                
                spx = int((s_ndc_x * 0.5 + 0.5) * float(width))
                spy = int((1.0 - (s_ndc_y * 0.5 + 0.5)) * float(height))
                
                if 0 <= spx < width and 0 <= spy < height:
                    z_stored = z_buffer[spy, spx]
                    # If stored depth is closer than sample, sample is occluded
                    if z_stored < sample_z - 0.004:
                        dist = abs(pz - z_stored)
                        if dist < radius:
                            w = 1.0 - dist / radius
                            occlusion += w * w
                            
            occ_factor = occlusion / float(n_samples)
            ao = max(0.35, 1.0 - occ_factor * 1.25)
            ssao_raw[py, px] = ao


@numba.njit(fastmath=True)
def bilateral_blur_ssao(
    height: int,
    width: int,
    ssao_in: np.ndarray,
    z_buffer: np.ndarray,
    ssao_out: np.ndarray
):
    """Edge-preserving depth-aware bilateral blur for SSAO buffer."""
    for py in range(height):
        for px in range(width):
            pz = z_buffer[py, px]
            if pz >= 1e8:
                ssao_out[py, px] = 1.0
                continue
                
            w_sum = 0.0
            v_sum = 0.0
            
            for dy in range(-2, 3):
                ny = py + dy
                if ny < 0 or ny >= height:
                    continue
                for dx in range(-2, 3):
                    nx = px + dx
                    if nx < 0 or nx >= width:
                        continue
                        
                    nz_val = z_buffer[ny, nx]
                    if nz_val >= 1e8:
                        continue
                        
                    r2_space = float(dx * dx + dy * dy)
                    w_space = np.exp(-r2_space / 4.0)
                    w_depth = np.exp(-abs(pz - nz_val) * 15.0)
                    
                    weight = w_space * w_depth
                    w_sum += weight
                    v_sum += weight * ssao_in[ny, nx]
                    
            if w_sum > 1e-6:
                ssao_out[py, px] = v_sum / w_sum
            else:
                ssao_out[py, px] = ssao_in[py, px]


@numba.njit(fastmath=True)
def shade_scene_numba(
    height: int,
    width: int,
    out_mat_ids: np.ndarray,
    out_normals: np.ndarray,
    out_world: np.ndarray,
    out_uvs: np.ndarray,
    z_buffer: np.ndarray,
    ssao_buffer: np.ndarray,
    rendered_rgb: np.ndarray,
    cam_pos: np.ndarray,
    # Shadow map parameters
    shadow_map: np.ndarray,
    s_width: int,
    s_height: int,
    light_u: np.ndarray,
    light_v: np.ndarray,
    light_w: np.ndarray,
    s_bounds: np.ndarray, # [x_min, x_max, y_min, y_max]
    # Studio lighting
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
    """
    Numba-accelerated photorealistic PBR shader evaluating:
    - Shadow-mapped directional occlusion with PCF soft penumbra.
    - SSAO and perianth cavity ambient occlusion.
    - Dual-lobe Cook-Torrance GGX microfacet specular BRDF with Schlick Fresnel.
    - Micro-texture bump mapping: vein striations, epidermal cells, and pollen dust.
    - Warm forward subsurface transmission through thin organic petal tissues.
    """
    inv_sx = float(s_width) / (s_bounds[1] - s_bounds[0])
    inv_sy = float(s_height) / (s_bounds[3] - s_bounds[2])
    
    l1_c0 = key_col[0] * key_intensity
    l1_c1 = key_col[1] * key_intensity
    l1_c2 = key_col[2] * key_intensity
    
    l2_c0 = fill_col[0] * fill_intensity
    l2_c1 = fill_col[1] * fill_intensity
    l2_c2 = fill_col[2] * fill_intensity
    
    l3_c0 = rim_col[0] * rim_intensity
    l3_c1 = rim_col[1] * rim_intensity
    l3_c2 = rim_col[2] * rim_intensity
    
    for py in range(height):
        for px in range(width):
            m_id = out_mat_ids[py, px]
            if m_id == 0:
                continue
                
            wx = out_world[py, px, 0]
            wy = out_world[py, px, 1]
            wz = out_world[py, px, 2]
            
            # View vector pointing from surface to camera
            vx = cam_pos[0] - wx
            vy = cam_pos[1] - wy
            vz = cam_pos[2] - wz
            v_len = max(1e-6, (vx*vx + vy*vy + vz*vz)**0.5)
            vx /= v_len; vy /= v_len; vz /= v_len
            
            # Raw normal in world coordinates
            nx = out_normals[py, px, 0]
            ny = out_normals[py, px, 1]
            nz = out_normals[py, px, 2]
            
            # Two-sided normal orientation: normal must face the viewer!
            n_dot_v = nx * vx + ny * vy + nz * vz
            if n_dot_v < 0.0:
                nx = -nx; ny = -ny; nz = -nz
                n_dot_v = -n_dot_v
                
            u_val = out_uvs[py, px, 0]
            v_val = out_uvs[py, px, 1]
            s = min(1.0, max(0.0, u_val))
            t = min(1.0, max(-1.0, v_val))
            abs_t = abs(t)
            
            # Default material properties
            r_alb = 0.985; g_alb = 0.975; b_alb = 0.955
            roughness = 0.38
            spec_k = 0.22
            trans_k = 0.25
            cavity_ao = 1.0
            
            if m_id == 1 or m_id == 2:  # Inner Lily Petal or Outer Sepal
                # Base pristine ivory white with subtle cream warmth
                r_alb = 0.985; g_alb = 0.980; b_alb = 0.965
                
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
                
                # Longitudinal fibrovascular veins
                vein_mod = 1.0
                for k in range(1, 5):
                    tau = 0.22 * float(k) * (1.0 - 0.15 * s)
                    vein_mod += 0.04 * np.exp(-((abs_t - tau) ** 2) / 0.002)
                r_alb *= (0.97 + 0.03 * vein_mod)
                g_alb *= (0.97 + 0.03 * vein_mod)
                b_alb *= (0.97 + 0.03 * vein_mod)
                
                # Translucency and baseline material properties
                trans_k = 0.28 + 0.36 * (abs_t ** 2) + 0.30 * throat_factor
                roughness = 0.32 - 0.06 * throat_factor
                spec_k = 0.28 if m_id == 1 else 0.26
                cavity_ao = 0.52 + 0.48 * (s ** 0.85)
                if m_id == 2:
                    cavity_ao *= 0.90

                # Delicate soft pink blush in the papillae zone
                blush = 0.14 * np.exp(-22.0 * ((s - 0.25) ** 2)) * np.exp(-14.0 * (t * t))
                r_alb = r_alb * (1.0 - blush) + 0.98 * blush
                g_alb = g_alb * (1.0 - blush) + 0.74 * blush
                b_alb = b_alb * (1.0 - blush) + 0.80 * blush

                # 3D Raised Carmine Papillae near throat (nectariferous nodules)
                p_cell_s = s * 32.0
                p_cell_t = t * 24.0
                i_s = int(p_cell_s)
                i_t = int(p_cell_t)
                f_s = p_cell_s - float(i_s) - 0.5
                f_t = p_cell_t - float(i_t) - 0.5
                cell_hash = np.sin(float(i_s) * 127.1 + float(i_t) * 311.7) * 43758.5453
                cell_rnd = cell_hash - np.floor(cell_hash)
                
                if cell_rnd > 0.42 and 0.09 < s < 0.45 and 0.03 < abs_t < 0.40:
                    dist2 = f_s * f_s + f_t * f_t
                    if dist2 < 0.20:
                        p_height = (0.20 - dist2) / 0.20
                        # 3D micro-normal perturbation for papilla highlight & shadow
                        nx += f_t * p_height * 0.75
                        ny += f_s * p_height * 0.75
                        fade = (0.45 - s) / 0.36 * (1.0 - abs_t / 0.40)
                        p_w = p_height * fade * 0.95
                        r_alb = r_alb * (1.0 - p_w) + 0.78 * p_w
                        g_alb = g_alb * (1.0 - p_w) + 0.12 * p_w
                        b_alb = b_alb * (1.0 - p_w) + 0.20 * p_w
                        spec_k += 0.30 * p_w
                        roughness = max(0.08, roughness - 0.20 * p_w)
                        
                # Micro-vein striations + epidermal cell texture
                d_vein = 0.030 * np.sin(48.0 * t) * (1.0 - s * 0.5)
                d_cell = 0.015 * (np.sin(160.0 * s + 60.0 * t) + np.cos(90.0 * s - 130.0 * t))
                nx += d_vein * 0.16 + d_cell * 0.10
                ny += d_cell * 0.10
                n_len = max(1e-6, (nx*nx + ny*ny + nz*nz)**0.5)
                nx /= n_len; ny /= n_len; nz /= n_len
                    
            elif m_id == 3:  # Stamen Filament
                # Pale translucent jade-ivory with smooth gradient
                r_alb = 0.82; g_alb = 0.94; b_alb = 0.76
                roughness = 0.28; spec_k = 0.28; trans_k = 0.38
                
            elif m_id == 4:  # Golden-Orange Pollen Anther
                # Dehiscence sulcus down center
                sulcus = np.exp(-32.0 * (t * t))
                # Intense vibrant golden-amber pollen exterior
                r_alb = 0.94 * (1.0 - 0.40 * sulcus) + 0.48 * (0.40 * sulcus)
                g_alb = 0.50 * (1.0 - 0.40 * sulcus) + 0.20 * (0.40 * sulcus)
                b_alb = 0.04 * (1.0 - 0.40 * sulcus) + 0.02 * (0.40 * sulcus)
                # Granular pollen micro-bumps
                p_bump = 0.32 * (np.sin(280.0 * s) * np.cos(280.0 * t) + 0.5 * np.sin(420.0 * s))
                nx += p_bump * (1.0 - 0.6 * sulcus)
                ny += p_bump * (1.0 - 0.6 * sulcus)
                n_len = max(1e-6, (nx*nx + ny*ny + nz*nz)**0.5)
                nx /= n_len; ny /= n_len; nz /= n_len
                roughness = 0.72; spec_k = 0.07; trans_k = 0.04
                
            elif m_id == 5:  # Pistil / 3-Lobed Stigma
                # Moist glossy pistachio-green with subtle carmine blush
                r_alb = 0.70; g_alb = 0.86; b_alb = 0.54
                roughness = 0.18; spec_k = 0.38; trans_k = 0.20
                
            elif m_id == 6:  # Stem
                # Sturdy dark botanical green
                r_alb = 0.038; g_alb = 0.125; b_alb = 0.032
                roughness = 0.32; spec_k = 0.22; trans_k = 0.12
                
            elif m_id == 7:  # Leaf
                # Rich glossy dark emerald green with pale lime midrib groove
                r_alb = 0.022; g_alb = 0.082; b_alb = 0.020
                mid_l = np.exp(-36.0 * (t * t))
                r_alb = r_alb * (1.0 - mid_l) + 0.090 * mid_l
                g_alb = g_alb * (1.0 - mid_l) + 0.210 * mid_l
                b_alb = b_alb * (1.0 - mid_l) + 0.055 * mid_l
                # Waxy cuticle specular
                roughness = 0.18; spec_k = 0.42; trans_k = 0.10
                
            # Depth and contact ambient occlusion
            depth_val = z_buffer[py, px]
            depth_ao = max(0.65, min(1.0, 1.20 - 0.10 * (depth_val - 2.5)))
            contact_ao = ssao_buffer[py, px]
            ao_total = depth_ao * cavity_ao * contact_ao
            
            # --- SHADOW MAP CALCULATION WITH 16-TAP POISSON DISK PCF ---
            lx = wx * light_u[0] + wy * light_u[1] + wz * light_u[2]
            ly = wx * light_v[0] + wy * light_v[1] + wz * light_v[2]
            lz = wx * light_w[0] + wy * light_w[1] + wz * light_w[2]
            
            su = (lx - s_bounds[0]) * inv_sx
            sv = (ly - s_bounds[2]) * inv_sy
            
            n_dot_l1 = nx * key_dir[0] + ny * key_dir[1] + nz * key_dir[2]
            cos_t = max(0.0, n_dot_l1)
            bias = 0.007 + 0.015 * (1.0 - cos_t)
            
            shadow_factor = 0.0
            
            # 16 Poisson disk offset samples (filter radius ~7.5 px in 2048x2048 shadow map)
            p_offsets_x = (
                -0.326, -0.840, -0.696, -0.203,
                 0.962,  0.473,  0.519,  0.185,
                 0.507,  0.896, -0.322, -0.792,
                -0.098, -0.198,  0.307, -0.627
            )
            p_offsets_y = (
                -0.406, -0.074,  0.457,  0.621,
                -0.195, -0.480,  0.767,  0.247,
                 0.064,  0.412, -0.933, -0.598,
                -0.095, -0.621, -0.840,  0.074
            )
            
            pcf_rad = 7.5
            pcf_count = 0
            
            for k_pcf in range(16):
                px_s = int(su + p_offsets_x[k_pcf] * pcf_rad)
                py_s = int(sv + p_offsets_y[k_pcf] * pcf_rad)
                if 0 <= px_s < s_width and 0 <= py_s < s_height:
                    z_sm = shadow_map[py_s, px_s]
                    if lz - bias <= z_sm:
                        shadow_factor += 1.0
                    pcf_count += 1
                    
            if pcf_count > 0:
                shadow_factor /= float(pcf_count)
            else:
                shadow_factor = 1.0
                
            # Ambient light
            c_r = r_alb * ambient_col[0] * ao_total
            c_g = g_alb * ambient_col[1] * ao_total
            c_b = b_alb * ambient_col[2] * ao_total
            
            # Fresnel factor
            fresnel = 0.04 + 0.96 * ((1.0 - max(0.0, n_dot_v)) ** 5.0)
            
            # Dual-lobe microfacet specular exponent
            spec_exp1 = max(8.0, (1.0 - roughness) * 96.0)
            spec_exp2 = max(24.0, (1.0 - roughness * 0.4) * 160.0)
            
            # --- KEY LIGHT ---
            d1 = max(0.0, n_dot_l1)
            t1 = max(0.0, -n_dot_l1) * trans_k * 0.48
            
            h1_x = key_dir[0] + vx; h1_y = key_dir[1] + vy; h1_z = key_dir[2] + vz
            len_h1 = max(1e-6, (h1_x**2 + h1_y**2 + h1_z**2)**0.5)
            h1_x /= len_h1; h1_y /= len_h1; h1_z /= len_h1
            n_dot_h1 = max(0.0, nx * h1_x + ny * h1_y + nz * h1_z)
            
            # Dual-lobe specular
            s1_lobe1 = (n_dot_h1 ** spec_exp1) * spec_k
            s1_lobe2 = (n_dot_h1 ** spec_exp2) * (spec_k * 0.6)
            s1 = (s1_lobe1 + s1_lobe2 + 0.14 * fresnel) if d1 > 0.0 else 0.0
            
            # Soft shadow subsurface light bleed (warm ivory glow inside petal shadows)
            bounce_r = 0.16 * trans_k * (1.0 - shadow_factor) * 1.06
            bounce_g = 0.16 * trans_k * (1.0 - shadow_factor) * 0.98
            bounce_b = 0.16 * trans_k * (1.0 - shadow_factor) * 0.86
            
            c_r += l1_c0 * (r_alb * (d1 * shadow_factor + t1 * max(0.20, shadow_factor)) + bounce_r + s1 * shadow_factor)
            c_g += l1_c1 * (g_alb * (d1 * shadow_factor + t1 * max(0.20, shadow_factor)) + bounce_g + s1 * shadow_factor)
            c_b += l1_c2 * (b_alb * (d1 * shadow_factor + t1 * max(0.20, shadow_factor)) + bounce_b + s1 * shadow_factor)
            
            # --- FILL LIGHT ---
            n_dot_l2 = nx * fill_dir[0] + ny * fill_dir[1] + nz * fill_dir[2]
            d2 = max(0.0, n_dot_l2)
            t2 = max(0.0, -n_dot_l2) * trans_k * 0.40
            h2_x = fill_dir[0] + vx; h2_y = fill_dir[1] + vy; h2_z = fill_dir[2] + vz
            len_h2 = max(1e-6, (h2_x**2 + h2_y**2 + h2_z**2)**0.5)
            h2_x /= len_h2; h2_y /= len_h2; h2_z /= len_h2
            n_dot_h2 = max(0.0, nx * h2_x + ny * h2_y + nz * h2_z)
            s2 = (n_dot_h2 ** spec_exp1) * spec_k if d2 > 0.0 else 0.0
            c_r += l2_c0 * (r_alb * (d2 + t2) + s2) * ao_total
            c_g += l2_c1 * (g_alb * (d2 + t2) + s2) * ao_total
            c_b += l2_c2 * (b_alb * (d2 + t2) + s2) * ao_total
            
            # --- RIM / BACK LIGHT ---
            n_dot_l3 = nx * rim_dir[0] + ny * rim_dir[1] + nz * rim_dir[2]
            d3 = max(0.0, n_dot_l3)
            t3 = max(0.0, -n_dot_l3) * trans_k * 0.60
            h3_x = rim_dir[0] + vx; h3_y = rim_dir[1] + vy; h3_z = rim_dir[2] + vz
            len_h3 = max(1e-6, (h3_x**2 + h3_y**2 + h3_z**2)**0.5)
            h3_x /= len_h3; h3_y /= len_h3; h3_z /= len_h3
            n_dot_h3 = max(0.0, nx * h3_x + ny * h3_y + nz * h3_z)
            s3 = (n_dot_h3 ** spec_exp1) * spec_k if d3 > 0.0 else 0.0
            
            edge_glow = fresnel * trans_k * 0.40
            c_r += l3_c0 * (r_alb * (d3 + t3 + edge_glow) + s3)
            c_g += l3_c1 * (g_alb * (d3 + t3 + edge_glow) + s3)
            c_b += l3_c2 * (b_alb * (d3 + t3 + edge_glow) + s3)
            
            rendered_rgb[py, px, 0] = c_r
            rendered_rgb[py, px, 1] = c_g
            rendered_rgb[py, px, 2] = c_b


class LilyRenderer:
    """
    High-fidelity botanical image synthesizer.
    Renders 3D geometry into RGB image with shadow mapping, SSAO, PBR lighting, and filmic tone curve.
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
        Executes complete photorealistic rendering pipeline:
        1. Projects vertices to screen coordinates.
        2. Two-sided triangle rasterization with Z-buffering and world coordinates.
        3. Key-light orthographic shadow map rasterization.
        4. Screen-Space Ambient Occlusion (SSAO) and bilateral blur.
        5. Numba-accelerated optical PBR shading with soft shadows and translucency.
        6. Reinhard-ACES filmic tone mapping and gamma correction.
        """
        verts = scene_geom["vertices"]
        faces = scene_geom["faces"]
        normals = scene_geom["normals"]
        uvs = scene_geom["uvs"]
        mat_ids = scene_geom["mat_ids"]
        
        screen_coords, _ = self.project_to_screen(verts)
        
        tri_screen = screen_coords[faces].astype(np.float32)
        tri_world = verts[faces].astype(np.float32)
        tri_normals = normals[faces].astype(np.float32)
        tri_uvs = uvs[faces].astype(np.float32)
        tri_mat_ids = mat_ids[faces[:, 0]].astype(np.int32)
        
        z_buffer = np.full((self.height, self.width), 1e9, dtype=np.float32)
        out_world = np.zeros((self.height, self.width, 3), dtype=np.float32)
        out_normals = np.zeros((self.height, self.width, 3), dtype=np.float32)
        out_uvs = np.zeros((self.height, self.width, 2), dtype=np.float32)
        out_mat_ids = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Execute two-sided camera rasterization
        rasterize_triangles_twosided(
            tri_screen,
            tri_world,
            tri_normals,
            tri_uvs,
            tri_mat_ids,
            self.width,
            self.height,
            z_buffer,
            out_world,
            out_normals,
            out_uvs,
            out_mat_ids
        )
        
        # --- DIRECTIONAL SHADOW MAP PASS ---
        l_dir = self.lighting.key.direction
        d_L = -l_dir
        u_L = normalize(np.cross(d_L, np.array([0.0, 1.0, 0.0])))
        v_L = normalize(np.cross(u_L, d_L))
        
        x_L = verts @ u_L
        y_L = verts @ v_L
        z_L = verts @ d_L
        
        x_min, x_max = -2.1, 1.9
        y_min, y_max = -2.7, 1.6
        s_bounds = np.array([x_min, x_max, y_min, y_max], dtype=np.float32)
        
        W_s, H_s = 2048, 2048
        s_x = (x_L - x_min) / (x_max - x_min) * float(W_s)
        s_y = (y_L - y_min) / (y_max - y_min) * float(H_s)
        
        light_verts = np.stack([s_x, s_y, z_L], axis=-1)
        tri_light = light_verts[faces].astype(np.float32)
        shadow_map = np.full((H_s, W_s), 1e9, dtype=np.float32)
        rasterize_shadow_map_numba(tri_light, W_s, H_s, shadow_map)
        
        # --- SCREEN-SPACE AMBIENT OCCLUSION (SSAO) PASS ---
        cam_dir = normalize(self.camera_target - self.camera_pos)
        cam_up_ref = np.array([0.0, 1.0, 0.0])
        cam_right = normalize(np.cross(cam_dir, cam_up_ref))
        cam_up = normalize(np.cross(cam_right, cam_dir))
        
        focal = 1.0 / np.tan(self.fov * 0.5)
        aspect = self.width / self.height
        
        # 16-sample hemisphere kernel
        kernel = []
        for i in range(16):
            u1 = (i + 0.5) / 16.0
            u2 = (i * 0.61803398875) % 1.0
            r = np.sqrt(u1)
            theta = 2.0 * np.pi * u2
            kx = r * np.cos(theta)
            ky = r * np.sin(theta)
            kz = np.sqrt(max(0.0, 1.0 - u1))
            scale = float(i) / 16.0
            scale = 0.15 + 0.85 * (scale * scale)
            kernel.append([kx * scale, ky * scale, kz * scale])
        kernel_arr = np.array(kernel, dtype=np.float32)
        
        ssao_raw = np.ones((self.height, self.width), dtype=np.float32)
        ssao_blur = np.ones((self.height, self.width), dtype=np.float32)
        
        compute_ssao_numba(
            self.height, self.width, z_buffer, out_normals, out_mat_ids,
            cam_right.astype(np.float32), cam_up.astype(np.float32), cam_dir.astype(np.float32),
            float(focal), float(aspect), 0.12, kernel_arr, ssao_raw
        )
        bilateral_blur_ssao(self.height, self.width, ssao_raw, z_buffer, ssao_blur)
        
        # Precompute background field
        u_coords = (np.arange(self.width) + 0.5) / float(self.width)
        v_coords = (np.arange(self.height) + 0.5) / float(self.height)
        U, V = np.meshgrid(u_coords, v_coords)
        bg_image = evaluate_background_field(U, V)
        rendered_rgb = bg_image.copy()
        
        # Execute compiled PBR shading
        shade_scene_numba(
            self.height,
            self.width,
            out_mat_ids,
            out_normals,
            out_world,
            out_uvs,
            z_buffer,
            ssao_blur,
            rendered_rgb,
            self.camera_pos.astype(np.float32),
            shadow_map,
            W_s,
            H_s,
            u_L.astype(np.float32),
            v_L.astype(np.float32),
            d_L.astype(np.float32),
            s_bounds,
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
        exposed_rgb = rendered_rgb * 1.30
        l_white = 1.70
        tone_mapped = (exposed_rgb * (1.0 + exposed_rgb / (l_white * l_white))) / (1.0 + exposed_rgb)
        gamma_corrected = np.power(np.clip(tone_mapped, 0.0, 1.0), 1.0 / 2.2)
        
        return (gamma_corrected * 255.0).astype(np.uint8)
