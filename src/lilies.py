"""
lilies.py - Parametric botanical model of Lilium (White Oriental Lily) perianth and organs.

Features:
- Trimerous 6-tepal perianth (3 inner petals + 3 outer sepals) with organic variation.
- Parametric backward reflexing, trumpet throat depth, and ruffled undulating margins.
- Reproductive organs: 6 slender filaments with elongated versatile golden-orange anthers,
  plus 1 central pistil style with 3-lobed stigma.
- Developmental stages: fully open flowers, 3/4 views, profile trumpet flowers, and swelling buds.
- Generation of high-density triangle meshes with analytical UV coordinates, normals,
  and biological material IDs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from src.geometry import (
    normalize,
    rodrigues_rotation,
    rotation_matrix_from_vectors,
    coordinate_frame_matrix,
    petal_width_profile,
    petal_cross_section_curvature,
    compute_mesh_normals
)


class LilyTepal:
    """
    Parametric botanical model of a single Lilium tepal (petal or sepal).
    
    Botanical Morphology:
    - Basal Perianth Tube: For s <= 0.28, the 6 tepals interlock to form a deep chalice/calyx funnel.
    - Voluptuous Obovate Blade: Expands into broad, curved blade with rounded shoulders (peak s ≈ 0.54).
    - Recurved Reflex: Tips arch backward gracefully (sepals reflex strongly by ~90°, petals gently).
    - Romantic Wavy Margins: Multi-harmonic undulations along tepal margins.
    - True 3D Cross-Section: Concave throat cup, convex dorsal keel, ventral nectary furrow, zero zero-normals.
    """
    def __init__(
        self,
        index: int,
        is_inner: bool,
        base_angle: float,
        length: float,
        max_width: float,
        throat_depth: float,
        reflex_amount: float,
        curl_z: float,
        is_bud: bool = False,
        opening: float = 1.0,
        angle_jitter: float = 0.0,
        width_harmonic: float = 0.35,
        ruffle_amplitude: float = 0.012,
        ruffle_freq: float = 1.35,
        phase: float = 0.0
    ):
        self.index = index
        self.is_inner = is_inner
        self.base_angle = base_angle + angle_jitter
        self.length = length
        self.max_width = max_width
        self.throat_depth = throat_depth
        self.reflex_amount = reflex_amount
        self.curl_z = curl_z
        self.is_bud = is_bud
        self.opening = opening
        self.width_harmonic = width_harmonic
        self.ruffle_amplitude = ruffle_amplitude
        self.ruffle_freq = ruffle_freq
        self.phase = phase

    def generate_mesh(
        self,
        n_s: int = 56,
        n_t: int = 27
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D vertices, faces, normals, and UV coordinates in local flower coordinates.
        Returns:
            vertices: shape (V, 3)
            faces: shape (F, 3)
            normals: shape (V, 3)
            uvs: shape (V, 2) where u in [0, 1] (s), v in [-1, 1] (t)
        """
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        dir_x = np.cos(self.base_angle)
        dir_y = np.sin(self.base_angle)
        
        whorl_z = 0.024 * self.length if self.is_inner else -0.024 * self.length
        
        if self.is_bud:
            # Closed/swelling bud: tepals form a fusiform tapered spindle with organic spine curve
            curve_s = 0.040 * self.length * (s_vals ** 1.8)
            r_s = self.length * (0.045 * s_vals + 0.155 * self.opening * np.sin(np.pi * s_vals))
            z_s = self.length * (0.96 * s_vals) + whorl_z
            
            # Width profile clasping adjacent tepals: circumf / 6
            w_profile = (np.pi / 3.0) * r_s * 1.06
            W = w_profile[:, None] * T
            
            # Spine tangent
            ds = 0.001
            s_plus = np.clip(s_vals + ds, 0.0, 1.0)
            r_plus = self.length * (0.045 * s_plus + 0.155 * self.opening * np.sin(np.pi * s_plus))
            z_plus = self.length * (0.96 * s_plus) + whorl_z
            
            tan_x = (r_plus - r_s) * dir_x / ds
            tan_y = (r_plus - r_s) * dir_y / ds
            tan_z = (z_plus - z_s) / ds
            spine_tan = normalize(np.stack([tan_x, tan_y, tan_z], axis=-1))
            
            b0 = np.array([-np.sin(self.base_angle), np.cos(self.base_angle), 0.0])
            proj = np.sum(spine_tan * b0[None, :], axis=-1, keepdims=True)
            binormals = normalize(b0[None, :] - proj * spine_tan, axis=-1)
            normals_spine = normalize(np.cross(spine_tan, binormals), axis=-1)
            
            # Convex outer shell for bud
            c_elev = 0.032 * self.length * (1.0 - T ** 2)
            
            spine_x = r_s * dir_x + curve_s * np.cos(self.base_angle)
            spine_y = r_s * dir_y + curve_s * np.sin(self.base_angle)
            
        else:
            # TRUE BOTANICAL LILIUM PERIANTH:
            # 1. Radial expansion r(s):
            # Basal perianth tube: narrow and fused for s <= 0.22, flaring generously in trumpet bell
            r_tube = (0.028 + 0.10 * s_vals) * self.length
            flare_s = np.maximum(0.0, (s_vals - 0.22) / 0.78)
            r_flare = 0.72 * self.length * self.opening * (flare_s ** 1.25)
            r_s = r_tube + r_flare
            
            # Subtle organic lateral petal swirl
            twist_amp = (0.06 if self.is_inner else -0.05)
            angle_s = self.base_angle + twist_amp * (s_vals ** 1.5)
            dir_x = np.cos(angle_s)
            dir_y = np.sin(angle_s)
            
            # 2. Longitudinal Z trajectory:
            # Deep throat funnel receding into the green calyx tube
            z_throat = -0.05 * self.length * (np.maximum(0.0, 1.0 - s_vals / 0.22) ** 1.5)
            z_cup = z_throat + 0.44 * self.length * np.sin(0.5 * np.pi * (s_vals ** 0.88))
            
            # Backward recurvature (reflexing):
            # Inner petals reflex gracefully (~45°); outer sepals reflex dramatically (~75°) in a smooth sweep
            if self.is_inner:
                refl_start = 0.50
                refl_amount = 0.65 * self.reflex_amount
                refl_power = 1.65
            else:
                refl_start = 0.40
                refl_amount = 1.15 * self.reflex_amount
                refl_power = 1.55
                
            z_refl = refl_amount * (np.maximum(0.0, (s_vals - refl_start) / (1.0 - refl_start)) ** refl_power)
            z_s = z_cup - z_refl + self.curl_z * (s_vals ** 2) + whorl_z
            
            # Spine tangent frame
            ds = 0.001
            s_plus = np.clip(s_vals + ds, 0.0, 1.0)
            r_tube_p = (0.028 + 0.10 * s_plus) * self.length
            flare_s_p = np.maximum(0.0, (s_plus - 0.22) / 0.78)
            r_plus = r_tube_p + 0.72 * self.length * self.opening * (flare_s_p ** 1.25)
            angle_s_p = self.base_angle + twist_amp * (s_plus ** 1.5)
            dir_x_p = np.cos(angle_s_p)
            dir_y_p = np.sin(angle_s_p)
            
            z_throat_p = -0.05 * self.length * (np.maximum(0.0, 1.0 - s_plus / 0.22) ** 1.5)
            z_cup_p = z_throat_p + 0.44 * self.length * np.sin(0.5 * np.pi * (s_plus ** 0.88))
            z_refl_p = refl_amount * (np.maximum(0.0, (s_plus - refl_start) / (1.0 - refl_start)) ** refl_power)
            z_plus = z_cup_p - z_refl_p + self.curl_z * (s_plus ** 2) + whorl_z
            
            tan_x = (r_plus * dir_x_p - r_s * dir_x) / ds
            tan_y = (r_plus * dir_y_p - r_s * dir_y) / ds
            tan_z = (z_plus - z_s) / ds
            spine_tan = normalize(np.stack([tan_x, tan_y, tan_z], axis=-1))
            
            b0 = np.stack([-np.sin(angle_s), np.cos(angle_s), np.zeros_like(angle_s)], axis=-1)
            proj = np.sum(spine_tan * b0, axis=-1, keepdims=True)
            binormals = normalize(b0 - proj * spine_tan, axis=-1)
            normals_spine = normalize(np.cross(spine_tan, binormals), axis=-1)
            
            # 3. Botanical Stellate Perianth Width Profile:
            # Elegant obovate/lanceolate blade with slender basal unguis and sharp acute apex
            if self.is_inner:
                w_profile = petal_width_profile(
                    s_vals,
                    scale=self.max_width,
                    alpha=1.20,
                    beta=0.90,
                    gamma1=self.width_harmonic,
                    gamma2=-0.10,
                    phi=self.phase
                )
            else:
                w_profile = petal_width_profile(
                    s_vals,
                    scale=self.max_width,
                    alpha=1.15,
                    beta=1.05,
                    gamma1=self.width_harmonic,
                    gamma2=-0.10,
                    phi=self.phase
                )
            
            # Gentle margin wave modulation along edges
            ruffle_mod = 1.0 + 0.035 * (s_vals ** 1.3) * np.sin(2.0 * np.pi * self.ruffle_freq * s_vals + self.phase)
            w_profile = w_profile * ruffle_mod
            W = w_profile[:, None] * T
            
            # 4. Botanical 3D Cross-Section Architecture:
            # - Basal concave cup: inside throat (s < 0.35)
            throat_cup = -0.024 * self.length * np.exp(-12.0 * (s_vals ** 1.8))[:, None] * (1.0 - T ** 2)
            # - Concave inner midrib camber: depression along center (T=0) creating an authentic boat-shaped petal
            midrib_camber = -0.020 * self.length * (np.sin(np.pi * (s_vals ** 0.85)) ** 1.2)[:, None] * (1.0 - T ** 2)
            # - Ventral nectary furrow
            furrow_mask = (s_vals > 0.05) & (s_vals < 0.44)
            furrow_s = np.where(furrow_mask, np.sin(np.pi * (s_vals - 0.05) / 0.39), 0.0)
            ventral_furrow = -0.014 * self.length * furrow_s[:, None] * np.exp(-32.0 * (T ** 2))
            # - Vertical margin waves
            edge_ruffle = (
                self.ruffle_amplitude
                * self.length
                * (S ** 1.3)
                * (T ** 2)
                * np.sin(2.0 * np.pi * self.ruffle_freq * S + self.phase)
            )
            c_elev = throat_cup + midrib_camber + ventral_furrow + edge_ruffle
            
            spine_x = r_s * dir_x
            spine_y = r_s * dir_y
        
        # 3D surface vertices
        vx = spine_x[:, None] + W * binormals[:, 0, None] + c_elev * normals_spine[:, 0, None]
        vy = spine_y[:, None] + W * binormals[:, 1, None] + c_elev * normals_spine[:, 1, None]
        vz = z_s[:, None] + W * binormals[:, 2, None] + c_elev * normals_spine[:, 2, None]
        
        vertices = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
        uvs = np.stack([S, T], axis=-1).reshape(-1, 2)
        
        # Faces
        faces = []
        for i in range(n_s - 1):
            for j in range(n_t - 1):
                idx0 = i * n_t + j
                idx1 = i * n_t + (j + 1)
                idx2 = (i + 1) * n_t + j
                idx3 = (i + 1) * n_t + (j + 1)
                faces.append([idx0, idx1, idx2])
                faces.append([idx1, idx3, idx2])
                
        faces = np.array(faces, dtype=np.int32)
        normals = compute_mesh_normals(vertices, faces)
        return vertices, faces, normals, uvs


class LilyFlower:
    """
    Complete Lilium blossom model including 6 tepals, 6 stamens with anthers,
    and 1 central pistil with stigma.
    """
    def __init__(
        self,
        flower_id: int,
        center: np.ndarray,
        facing_direction: np.ndarray,
        up_direction: np.ndarray,
        scale: float = 1.0,
        opening: float = 1.0,
        is_bud: bool = False,
        roll: float = 0.0,
        seed: int = 42
    ):
        self.flower_id = flower_id
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.facing = normalize(np.asarray(facing_direction, dtype=np.float64).reshape(3))
        self.up = normalize(np.asarray(up_direction, dtype=np.float64).reshape(3))
        self.scale = scale
        self.opening = np.clip(opening, 0.15, 1.0)
        self.is_bud = is_bud
        self.roll = roll
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Orthonormal 3D frame aligning [0, 0, 1] with facing and rotated by roll
        self.rot_matrix = coordinate_frame_matrix(self.facing, self.up, roll=self.roll)
        
        self.tepals: List[LilyTepal] = []
        self._build_tepals()

    def _build_tepals(self):
        """Construct exactly 6 tepals arranged in two trimerous whorls."""
        if self.is_bud:
            base_length = 0.70 * self.scale
            for i in range(6):
                angle = i * (np.pi / 3.0)
                tepal = LilyTepal(
                    index=i,
                    is_inner=(i % 2 == 0),
                    base_angle=angle,
                    length=base_length,
                    max_width=0.10 * base_length,
                    throat_depth=0.0,
                    reflex_amount=0.0,
                    curl_z=0.0,
                    is_bud=True,
                    opening=self.opening,
                    ruffle_amplitude=0.0
                )
                self.tepals.append(tepal)
            return

        base_length = 0.72 * self.scale
        inner_width = 0.215 * base_length * (0.65 + 0.35 * self.opening)
        outer_width = 0.150 * base_length * (0.65 + 0.35 * self.opening)
        throat_depth = 0.38 * base_length
        reflex = 0.40 * base_length * (self.opening ** 1.3)
        
        # 3 Inner Tepals (Petals) at 0°, 120°, 240°
        for i in range(3):
            angle = i * (2.0 * np.pi / 3.0)
            angle_jitter = self.rng.uniform(-0.03, 0.03)
            width_var = self.rng.uniform(0.97, 1.03)
            refl_var = self.rng.uniform(0.95, 1.05)
            ruffle_amp = 0.016 * self.rng.uniform(0.90, 1.10)
            
            tepal = LilyTepal(
                index=i,
                is_inner=True,
                base_angle=angle,
                length=base_length * self.rng.uniform(0.98, 1.02),
                max_width=inner_width * width_var,
                throat_depth=throat_depth,
                reflex_amount=reflex * refl_var,
                curl_z=-0.03 * self.scale,
                is_bud=False,
                opening=self.opening,
                angle_jitter=angle_jitter,
                width_harmonic=self.rng.uniform(0.30, 0.38),
                ruffle_amplitude=ruffle_amp,
                ruffle_freq=self.rng.uniform(3.2, 4.2),
                phase=self.rng.uniform(0.0, 2.0 * np.pi)
            )
            self.tepals.append(tepal)
            
        # 3 Outer Tepals (Sepals) at 60°, 180°, 300°
        for i in range(3):
            angle = np.pi / 3.0 + i * (2.0 * np.pi / 3.0)
            angle_jitter = self.rng.uniform(-0.03, 0.03)
            width_var = self.rng.uniform(0.96, 1.04)
            refl_var = self.rng.uniform(0.95, 1.05)
            ruffle_amp = 0.010 * self.rng.uniform(0.90, 1.10)
            
            tepal = LilyTepal(
                index=3 + i,
                is_inner=False,
                base_angle=angle,
                length=base_length * 1.04 * self.rng.uniform(0.98, 1.02),
                max_width=outer_width * width_var,
                throat_depth=throat_depth,
                reflex_amount=reflex * refl_var * 1.10,
                curl_z=-0.04 * self.scale,
                is_bud=False,
                opening=self.opening,
                angle_jitter=angle_jitter,
                width_harmonic=self.rng.uniform(0.26, 0.34),
                ruffle_amplitude=ruffle_amp,
                ruffle_freq=self.rng.uniform(3.2, 4.2),
                phase=self.rng.uniform(0.0, 2.0 * np.pi)
            )
            self.tepals.append(tepal)

    def generate_flower_geometry(self) -> Dict[str, Any]:
        """
        Generates full 3D geometry for this flower:
        - tepals mesh (vertices, faces, normals, uvs, material_ids)
        - stamens and anthers meshes
        - pistil and stigma meshes
        """
        all_verts = []
        all_faces = []
        all_normals = []
        all_uvs = []
        all_mat_ids = []
        
        vert_offset = 0
        
        # 1. Generate Tepals
        for tepal in self.tepals:
            t_verts, t_faces, t_normals, t_uvs = tepal.generate_mesh(n_s=56, n_t=27)
            world_verts = self.center + (self.rot_matrix @ t_verts.T).T
            world_normals = (self.rot_matrix @ t_normals.T).T
            
            all_verts.append(world_verts)
            all_faces.append(t_faces + vert_offset)
            all_normals.append(world_normals)
            all_uvs.append(t_uvs)
            
            if self.is_bud:
                mat_id = 8  # Bud jade/ivory
            else:
                mat_id = 1 if tepal.is_inner else 2
            all_mat_ids.append(np.full(len(t_verts), mat_id, dtype=np.int32))
            vert_offset += len(t_verts)
            
        # 2. Generate Reproductive Organs (only if opening > 0.4 and not bud)
        if self.opening > 0.4 and not self.is_bud:
            filament_len = 0.44 * self.scale * (self.opening ** 1.35)
            anther_len = 0.120 * self.scale
            anther_radius = 0.0085 * self.scale
            
            for k in range(6):
                ang = k * (np.pi / 3.0) + np.pi / 6.0 + self.rng.uniform(-0.10, 0.10)
                
                # Filament tube mesh (cylinder with 8 radial slices, 10 height steps)
                # True botanical trumpet stamen: slender filament arching gracefully from throat
                u_steps = np.linspace(0.0, 1.0, 10)
                fil_verts = []
                for u in u_steps:
                    r_u = 0.20 * self.scale * (self.opening ** 1.35) * (np.sin(0.5 * np.pi * u) ** 1.15)
                    z_u = (0.04 + 0.38 * (self.opening ** 1.25) * (u ** 0.88)) * self.scale
                    sag = -0.010 * self.scale * (u ** 1.8)
                    spine_local = np.array([r_u * np.cos(ang), r_u * np.sin(ang) + sag, z_u])
                    spine_world = self.center + self.rot_matrix @ spine_local
                    radius = (0.0044 - 0.0014 * u) * self.scale
                    for phi_idx in range(8):
                        phi = phi_idx * (2.0 * np.pi / 8.0)
                        radial_vec = np.array([np.cos(phi), np.sin(phi), 0.0])
                        fil_verts.append(spine_world + self.rot_matrix @ (radial_vec * radius))
                        
                fil_verts = np.array(fil_verts, dtype=np.float64)
                r_tip = 0.20 * self.scale * (self.opening ** 1.35)
                z_tip = (0.04 + 0.38 * (self.opening ** 1.25)) * self.scale
                sag_tip = -0.010 * self.scale
                fil_tip_local = np.array([r_tip * np.cos(ang), r_tip * np.sin(ang) + sag_tip, z_tip])
                fil_tip_world = self.center + self.rot_matrix @ fil_tip_local
                
                fil_faces = []
                for i in range(9):
                    for j in range(8):
                        j_next = (j + 1) % 8
                        p0 = i * 8 + j
                        p1 = i * 8 + j_next
                        p2 = (i + 1) * 8 + j
                        p3 = (i + 1) * 8 + j_next
                        fil_faces.append([p0, p1, p2])
                        fil_faces.append([p1, p3, p2])
                fil_faces = np.array(fil_faces, dtype=np.int32)
                fil_normals = compute_mesh_normals(fil_verts, fil_faces)
                fil_uvs = np.zeros((len(fil_verts), 2), dtype=np.float64)
                
                all_verts.append(fil_verts)
                all_faces.append(fil_faces + vert_offset)
                all_normals.append(fil_normals)
                all_uvs.append(fil_uvs)
                all_mat_ids.append(np.full(len(fil_verts), 3, dtype=np.int32))  # 3 = filament (pale celery green)
                vert_offset += len(fil_verts)
                
                # Versatile Anther at filament tip: organic 3D poise poised perpendicular to filament
                yaw = self.rng.uniform(-0.55, 0.55)
                pitch = self.rng.uniform(-0.55, 0.55)
                roll = self.rng.uniform(-0.35, 0.35)
                anther_dir = normalize(np.array([
                    -np.sin(ang) + yaw,
                    np.cos(ang) + pitch,
                    0.25 * np.cos(ang) + roll
                ]))
                anther_ortho1 = normalize(np.cross(anther_dir, np.array([0.0, 0.0, 1.0])))
                anther_ortho2 = normalize(np.cross(anther_dir, anther_ortho1))
                
                n_lat = 14
                n_lon = 16
                anther_verts = []
                anther_uvs_list = []
                
                for lat_idx in range(n_lat):
                    lat_frac = lat_idx / (n_lat - 1)
                    lat = -np.pi / 2.0 + lat_frac * np.pi
                    # Subtle natural crescent curvature of anther
                    crescent_curve = -0.0028 * self.scale * (1.0 - np.cos(lat))
                    
                    for lon_idx in range(n_lon):
                        lon_frac = lon_idx / n_lon
                        lon = lon_idx * (2.0 * np.pi / n_lon)
                        theca_mod = 1.0 + 0.35 * np.cos(2.0 * lon)
                        rad_curr = anther_radius * theca_mod * np.cos(lat)
                        
                        x_e = (anther_len * 0.5) * np.sin(lat)
                        y_e = rad_curr * np.cos(lon) + crescent_curve
                        z_e = rad_curr * np.sin(lon)
                        
                        disp_local = x_e * anther_dir + y_e * anther_ortho1 + z_e * anther_ortho2
                        disp_world = self.rot_matrix @ disp_local
                        anther_verts.append(fil_tip_world + disp_world)
                        anther_uvs_list.append([lat_frac, lon_frac * 2.0 - 1.0])
                        
                anther_verts = np.array(anther_verts, dtype=np.float64)
                anther_uvs = np.array(anther_uvs_list, dtype=np.float64)
                
                anther_faces = []
                for i in range(n_lat - 1):
                    for j in range(n_lon):
                        j_next = (j + 1) % n_lon
                        p0 = i * n_lon + j
                        p1 = i * n_lon + j_next
                        p2 = (i + 1) * n_lon + j
                        p3 = (i + 1) * n_lon + j_next
                        anther_faces.append([p0, p1, p2])
                        anther_faces.append([p1, p3, p2])
                anther_faces = np.array(anther_faces, dtype=np.int32)
                anther_normals = compute_mesh_normals(anther_verts, anther_faces)
                
                all_verts.append(anther_verts)
                all_faces.append(anther_faces + vert_offset)
                all_normals.append(anther_normals)
                all_uvs.append(anther_uvs)
                all_mat_ids.append(np.full(len(anther_verts), 4, dtype=np.int32))  # 4 = anther (velvety cinnamon pollen)
                vert_offset += len(anther_verts)
                
            # Central Pistil & 3-Lobed Stigma (Extends elegantly beyond stamens)
            pistil_len = (0.06 + 0.44 * (self.opening ** 1.25)) * self.scale
            stigma_world = self.center + self.rot_matrix @ np.array([0.008 * self.scale, 0.006 * self.scale, pistil_len])
            pistil_base = self.center + self.rot_matrix @ np.array([0.0, 0.0, 0.05 * self.scale])
            
            # Style tube
            pistil_verts = []
            pistil_uvs_list = []
            n_style_steps = 12
            for u in np.linspace(0.0, 1.0, n_style_steps):
                center_pt = (1.0 - u) * pistil_base + u * stigma_world
                r_pis = (0.0065 - 0.0018 * u) * self.scale
                for phi_idx in range(8):
                    phi = phi_idx * (2.0 * np.pi / 8.0)
                    rad_v = np.array([np.cos(phi), np.sin(phi), 0.0])
                    pistil_verts.append(center_pt + self.rot_matrix @ (rad_v * r_pis))
                    pistil_uvs_list.append([u, phi_idx / 8.0])
                    
            pistil_faces = []
            for i in range(n_style_steps - 1):
                for j in range(8):
                    j_next = (j + 1) % 8
                    p0 = i * 8 + j
                    p1 = i * 8 + j_next
                    p2 = (i + 1) * 8 + j
                    p3 = (i + 1) * 8 + j_next
                    pistil_faces.append([p0, p1, p2])
                    pistil_faces.append([p1, p3, p2])
            
            # 3-Lobed Swollen Stigma Pad at style tip
            n_stigma_radial = 18
            stigma_rad_base = 0.016 * self.scale
            stigma_tip_center = stigma_world + self.rot_matrix @ np.array([0.0, 0.0, 0.008 * self.scale])
            
            # Center vertex of stigma
            stigma_center_idx = len(pistil_verts)
            pistil_verts.append(stigma_tip_center)
            pistil_uvs_list.append([1.0, 0.0])
            
            for j in range(n_stigma_radial):
                theta = j * (2.0 * np.pi / n_stigma_radial)
                # 3-lobed clover morphology
                r_lobe = stigma_rad_base * (1.0 + 0.38 * np.cos(3.0 * theta))
                lobe_disp = np.array([r_lobe * np.cos(theta), r_lobe * np.sin(theta), -0.003 * self.scale])
                pistil_verts.append(stigma_world + self.rot_matrix @ lobe_disp)
                pistil_uvs_list.append([1.0, j / n_stigma_radial])
                
            # Stigma cap triangles
            for j in range(n_stigma_radial):
                j_next = (j + 1) % n_stigma_radial
                v_curr = stigma_center_idx + 1 + j
                v_next = stigma_center_idx + 1 + j_next
                pistil_faces.append([stigma_center_idx, v_curr, v_next])
                
            pistil_verts = np.array(pistil_verts, dtype=np.float64)
            pistil_faces = np.array(pistil_faces, dtype=np.int32)
            pistil_normals = compute_mesh_normals(pistil_verts, pistil_faces)
            pistil_uvs = np.array(pistil_uvs_list, dtype=np.float64)
            
            all_verts.append(pistil_verts)
            all_faces.append(pistil_faces + vert_offset)
            all_normals.append(pistil_normals)
            all_uvs.append(pistil_uvs)
            all_mat_ids.append(np.full(len(pistil_verts), 5, dtype=np.int32))  # 5 = pistil (pistachio/lime green)
            vert_offset += len(pistil_verts)

        return {
            "vertices": np.vstack(all_verts),
            "faces": np.vstack(all_faces),
            "normals": np.vstack(all_normals),
            "uvs": np.vstack(all_uvs),
            "mat_ids": np.concatenate(all_mat_ids)
        }
