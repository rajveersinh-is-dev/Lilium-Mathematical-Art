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
    """Parametric representation of a single lily tepal (petal or sepal)."""
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
        ruffle_amplitude: float = 0.018,
        ruffle_freq: float = 4.0,
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
        
        # Grid of s and t
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        dir_x = np.cos(self.base_angle)
        dir_y = np.sin(self.base_angle)
        
        whorl_z = 0.02 * self.length if self.is_inner else -0.02 * self.length
        
        if self.is_bud:
            # Closed or swelling bud: tepals form a fusiform tapered spindle
            r_s = self.length * (0.04 * s_vals + 0.16 * self.opening * np.sin(np.pi * s_vals))
            z_s = self.length * (0.95 * s_vals) + whorl_z
            
            # Width profile clasping adjacent tepals: circumf / 6
            w_profile = (np.pi / 3.0) * r_s * 1.08
            W = w_profile[:, None] * T
            
            # Spine tangent
            ds = 0.001
            s_plus = np.clip(s_vals + ds, 0.0, 1.0)
            r_plus = self.length * (0.04 * s_plus + 0.16 * self.opening * np.sin(np.pi * s_plus))
            z_plus = self.length * (0.95 * s_plus) + whorl_z
            
            tan_x = (r_plus - r_s) * dir_x / ds
            tan_y = (r_plus - r_s) * dir_y / ds
            tan_z = (z_plus - z_s) / ds
            spine_tan = normalize(np.stack([tan_x, tan_y, tan_z], axis=-1))
            
            b0 = np.array([-np.sin(self.base_angle), np.cos(self.base_angle), 0.0])
            proj = np.sum(spine_tan * b0[None, :], axis=-1, keepdims=True)
            binormals = normalize(b0[None, :] - proj * spine_tan, axis=-1)
            normals_spine = normalize(np.cross(spine_tan, binormals), axis=-1)
            
            # Curved convex shell for bud
            c_elev = 0.03 * self.length * (1.0 - T ** 2)
            
        else:
            # Open or partially open lily perianth
            # Radial distance: narrow throat tube expanding into broad flared star
            r_s = self.length * (0.06 * s_vals + 0.78 * self.opening * (s_vals ** 1.45))
            
            # Z axis: chalice tube projecting forward, then tips reflexing backward
            z_tube = 0.32 * self.length * s_vals
            z_refl = self.reflex_amount * (np.maximum(0.0, s_vals - 0.38) ** 1.85)
            z_s = z_tube - z_refl + self.curl_z * (s_vals ** 2) + whorl_z
            
            # Spine tangent via finite differences
            ds = 0.001
            s_plus = np.clip(s_vals + ds, 0.0, 1.0)
            r_plus = self.length * (0.06 * s_plus + 0.78 * self.opening * (s_plus ** 1.45))
            z_tube_p = 0.32 * self.length * s_plus
            z_refl_p = self.reflex_amount * (np.maximum(0.0, s_plus - 0.38) ** 1.85)
            z_plus = z_tube_p - z_refl_p + self.curl_z * (s_plus ** 2) + whorl_z
            
            tan_x = (r_plus - r_s) * dir_x / ds
            tan_y = (r_plus - r_s) * dir_y / ds
            tan_z = (z_plus - z_s) / ds
            spine_tan = normalize(np.stack([tan_x, tan_y, tan_z], axis=-1))
            
            b0 = np.array([-np.sin(self.base_angle), np.cos(self.base_angle), 0.0])
            proj = np.sum(spine_tan * b0[None, :], axis=-1, keepdims=True)
            binormals = normalize(b0[None, :] - proj * spine_tan, axis=-1)
            normals_spine = normalize(np.cross(spine_tan, binormals), axis=-1)
            
            # Width profile: organic lily unguis expanding into broad blade
            w_profile = petal_width_profile(
                s_vals,
                scale=self.max_width,
                alpha=1.20,
                beta=0.88,
                gamma1=self.width_harmonic,
                gamma2=-0.12,
                phi=self.phase
            )
            W = w_profile[:, None] * T
            
            # Elevation: groove, arch, and edge ruffles
            c_elev = petal_cross_section_curvature(
                S,
                T,
                groove_depth=0.030 * self.length,
                arch_height=0.038 * self.length,
                margin_undulation=self.ruffle_amplitude * self.length,
                undulation_freq=self.ruffle_freq,
                margin_phase=self.phase
            )
            
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
            base_length = 0.62 * self.scale
            inner_width = 0.07 * base_length
            outer_width = 0.06 * base_length
            throat_depth = 0.0
            reflex = 0.0
        else:
            base_length = 0.74 * self.scale
            # Authentic botanical aspect ratio with natural overlap
            inner_width = 0.22 * base_length * self.opening
            outer_width = 0.16 * base_length * self.opening
            throat_depth = 0.32 * base_length
            reflex = 0.44 * base_length * (self.opening ** 1.5)
        
        # 3 Inner Tepals (Petals) at 0°, 120°, 240°
        for i in range(3):
            angle = i * (2.0 * np.pi / 3.0)
            angle_jitter = self.rng.uniform(-0.04, 0.04)
            width_var = self.rng.uniform(0.96, 1.04)
            refl_var = self.rng.uniform(0.94, 1.06)
            ruffle_amp = 0.026 * self.rng.uniform(0.85, 1.15)
            
            tepal = LilyTepal(
                index=i,
                is_inner=True,
                base_angle=angle,
                length=base_length * self.rng.uniform(0.98, 1.02),
                max_width=inner_width * width_var,
                throat_depth=throat_depth,
                reflex_amount=reflex * refl_var,
                curl_z=-0.05 * self.scale,
                is_bud=self.is_bud,
                opening=self.opening,
                angle_jitter=angle_jitter,
                width_harmonic=self.rng.uniform(0.32, 0.42),
                ruffle_amplitude=ruffle_amp,
                ruffle_freq=self.rng.uniform(3.5, 4.5),
                phase=self.rng.uniform(0.0, 2.0 * np.pi)
            )
            self.tepals.append(tepal)
            
        # 3 Outer Tepals (Sepals) at 60°, 180°, 300°
        for i in range(3):
            angle = np.pi / 3.0 + i * (2.0 * np.pi / 3.0)
            angle_jitter = self.rng.uniform(-0.04, 0.04)
            width_var = self.rng.uniform(0.95, 1.05)
            refl_var = self.rng.uniform(0.94, 1.06)
            ruffle_amp = 0.020 * self.rng.uniform(0.85, 1.15)
            
            tepal = LilyTepal(
                index=3 + i,
                is_inner=False,
                base_angle=angle,
                length=base_length * 1.03 * self.rng.uniform(0.98, 1.02),
                max_width=outer_width * width_var,
                throat_depth=throat_depth,
                reflex_amount=reflex * refl_var * 1.04,
                curl_z=-0.04 * self.scale,
                is_bud=self.is_bud,
                opening=self.opening,
                angle_jitter=angle_jitter,
                width_harmonic=self.rng.uniform(0.30, 0.40),
                ruffle_amplitude=ruffle_amp,
                ruffle_freq=self.rng.uniform(3.5, 4.5),
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
            # Transform from local flower coordinates to world space
            # World = Center + RotMatrix @ Local
            world_verts = self.center + (self.rot_matrix @ t_verts.T).T
            world_normals = (self.rot_matrix @ t_normals.T).T
            
            all_verts.append(world_verts)
            all_faces.append(t_faces + vert_offset)
            all_normals.append(world_normals)
            all_uvs.append(t_uvs)
            
            mat_id = 1 if tepal.is_inner else 2  # 1 = inner petal, 2 = outer sepal
            all_mat_ids.append(np.full(len(t_verts), mat_id, dtype=np.int32))
            vert_offset += len(t_verts)
            
        # 2. Generate Reproductive Organs (only if opening > 0.3)
        if self.opening > 0.3 and not self.is_bud:
            # 6 Stamens (Filaments + Golden Anthers)
            filament_len = 0.40 * self.scale * self.opening
            anther_len = 0.096 * self.scale
            anther_radius = 0.016 * self.scale
            
            for k in range(6):
                # Angle spaced between tepals
                ang = k * (np.pi / 3.0) + np.pi / 6.0 + self.rng.uniform(-0.08, 0.08)
                # Filament curves outward from flower center
                rad_disp = 0.17 * self.scale * self.opening
                fil_tip_local = np.array([
                    rad_disp * np.cos(ang),
                    rad_disp * np.sin(ang),
                    filament_len * self.rng.uniform(0.95, 1.05)
                ])
                fil_tip_world = self.center + self.rot_matrix @ fil_tip_local
                fil_base_world = self.center + self.rot_matrix @ np.array([0.0, 0.0, -0.05 * self.scale])
                
                # Filament tube mesh (cylinder with 8 radial slices, 10 height steps)
                u_steps = np.linspace(0.0, 1.0, 10)
                fil_verts = []
                for u in u_steps:
                    # Curved spine
                    curv = 0.06 * np.sin(np.pi * u)
                    spine_pt = (1.0 - u) * fil_base_world + u * fil_tip_world + self.rot_matrix @ np.array([curv * np.cos(ang), curv * np.sin(ang), 0.0])
                    # Circular slice
                    radius = (0.005 - 0.0015 * u) * self.scale
                    for phi_idx in range(8):
                        phi = phi_idx * (2.0 * np.pi / 8.0)
                        radial_vec = np.array([np.cos(phi), np.sin(phi), 0.0])
                        fil_verts.append(spine_pt + self.rot_matrix @ (radial_vec * radius))
                        
                fil_verts = np.array(fil_verts, dtype=np.float64)
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
                all_mat_ids.append(np.full(len(fil_verts), 3, dtype=np.int32))  # 3 = filament (pale green)
                vert_offset += len(fil_verts)
                
                # Anther at filament tip: versatile 2-lobed thecae with dehiscence sulcus
                anther_dir = normalize(np.array([-np.sin(ang), np.cos(ang), self.rng.uniform(-0.25, 0.25)]))
                anther_ortho1 = normalize(np.cross(anther_dir, np.array([0.0, 0.0, 1.0])))
                anther_ortho2 = normalize(np.cross(anther_dir, anther_ortho1))
                
                n_lat = 14
                n_lon = 16
                anther_verts = []
                anther_uvs_list = []
                
                for lat_idx in range(n_lat):
                    lat_frac = lat_idx / (n_lat - 1)
                    lat = -np.pi / 2.0 + lat_frac * np.pi
                    # Bananoid gentle curvature along length
                    curve_offset = 0.008 * self.scale * np.cos(lat)
                    
                    for lon_idx in range(n_lon):
                        lon_frac = lon_idx / n_lon
                        lon = lon_idx * (2.0 * np.pi / n_lon)
                        
                        # Two-lobed pollen thecae profile: indentation at lon = 0 and lon = pi
                        theca_mod = 1.0 + 0.32 * np.cos(2.0 * lon)
                        rad_curr = anther_radius * theca_mod * np.cos(lat)
                        
                        x_e = (anther_len * 0.5) * np.sin(lat)
                        y_e = rad_curr * np.cos(lon) + curve_offset
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
                all_mat_ids.append(np.full(len(anther_verts), 4, dtype=np.int32))  # 4 = anther (golden pollen)
                vert_offset += len(anther_verts)
                
            # Central Pistil & 3-Lobed Stigma
            pistil_len = 0.44 * self.scale * self.opening
            stigma_world = self.center + self.rot_matrix @ np.array([0.02 * self.scale, 0.01 * self.scale, pistil_len])
            pistil_base = self.center + self.rot_matrix @ np.array([0.0, 0.0, -0.06 * self.scale])
            
            # Style tube
            pistil_verts = []
            pistil_uvs_list = []
            n_style_steps = 12
            for u in np.linspace(0.0, 1.0, n_style_steps):
                center_pt = (1.0 - u) * pistil_base + u * stigma_world
                r_pis = (0.0075 - 0.0018 * u) * self.scale
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
            stigma_rad_base = 0.018 * self.scale
            stigma_tip_center = stigma_world + self.rot_matrix @ np.array([0.0, 0.0, 0.012 * self.scale])
            
            # Center vertex of stigma
            stigma_center_idx = len(pistil_verts)
            pistil_verts.append(stigma_tip_center)
            pistil_uvs_list.append([1.0, 0.0])
            
            for j in range(n_stigma_radial):
                theta = j * (2.0 * np.pi / n_stigma_radial)
                # 3-lobed clover morphology
                r_lobe = stigma_rad_base * (1.0 + 0.42 * np.cos(3.0 * theta))
                lobe_disp = np.array([r_lobe * np.cos(theta), r_lobe * np.sin(theta), -0.004 * self.scale])
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
