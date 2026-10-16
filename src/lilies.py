"""
lilies.py - Parametric botanical model of Lilium (White Oriental Lily) perianth and organs.

Features:
- Trimerous 6-tepal perianth (3 inner petals + 3 outer sepals) with continuous fluted base (whorl_z=0 at s<=0.14).
- Parametric backward reflexing, trumpet throat depth, and ruffled undulating margins.
- Reproductive organs: 6 slender clustered filaments with watertight dome caps and versatile chocolate-russet anthers,
  plus 1 central pistil style with 3-lobed papillate stigma.
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
    Parametric surface representation for a single Lilium tepal:
    P(s, t) = c(s) + t * w(s) * b(s) + h(s, t) * n(s)
    Features:
    - Continuous perianth tube: whorl_z = 0 at base (s <= 0.14)
    - Convex beta-harmonic width profile w(s)
    - Distinct recurvature (retroflexion) for outer sepals
    - Smooth midrib keel, ventral nectary furrow, and margin flutter
    """
    def __init__(
        self,
        index: int,
        is_inner: bool,
        base_angle: float,
        length: float = 0.72,
        opening: float = 1.0,
        is_bud: bool = False,
        width_scale: float = 1.0,
        ruffle_amplitude: float = 0.012,
        ruffle_freq: float = 2.5,
        phase: float = 0.0
    ):
        self.index = index
        self.is_inner = is_inner
        self.base_angle = base_angle
        self.length = length
        self.opening = np.clip(opening, 0.15, 1.0)
        self.is_bud = is_bud
        self.width_scale = width_scale
        self.ruffle_amplitude = ruffle_amplitude
        self.ruffle_freq = ruffle_freq
        self.phase = phase

    def generate_mesh(
        self,
        n_s: int = 96,
        n_t: int = 40
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')

        if self.is_bud:
            # Elegant closed bud spindle
            r_s = 0.022 * self.length + 0.095 * self.length * np.sin(np.pi * (s_vals ** 0.85))
            dir_x = np.cos(self.base_angle)
            dir_y = np.sin(self.base_angle)
            z_s = 0.95 * self.length * s_vals

            w_prof = (np.pi / 3.0) * r_s * 1.06
            W = w_prof[:, None] * T

            b0 = np.array([-np.sin(self.base_angle), np.cos(self.base_angle), 0.0])
            n0 = np.array([np.cos(self.base_angle), np.sin(self.base_angle), 0.0])

            c_elev = 0.014 * self.length * np.sin(np.pi * s_vals)[:, None] * (1.0 - T ** 2)

            vx = (r_s * dir_x)[:, None] + W * b0[0] + c_elev * n0[0]
            vy = (r_s * dir_y)[:, None] + W * b0[1] + c_elev * n0[1]
            vz = z_s[:, None] + W * b0[2] + c_elev * n0[2]

            vertices = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
            uvs = np.stack([S, T], axis=-1).reshape(-1, 2)
            faces = []
            for i in range(n_s - 1):
                for j in range(n_t - 1):
                    p0 = i * n_t + j
                    p1 = i * n_t + (j + 1)
                    p2 = (i + 1) * n_t + j
                    p3 = (i + 1) * n_t + (j + 1)
                    faces.append([p0, p1, p2])
                    faces.append([p1, p3, p2])
            faces = np.array(faces, dtype=np.int32)
            normals = compute_mesh_normals(vertices, faces)
            return vertices, faces, normals, uvs

        # Open bloom:
        # Perianth tube radius: expands smoothly from base (s=0)
        r_tube = (0.024 + 0.076 * s_vals) * self.length
        flare_s = np.maximum(0.0, (s_vals - 0.18) / 0.82)
        r_flare = 0.66 * self.length * self.opening * (flare_s ** 1.28)
        r_s = r_tube + r_flare

        # Subtle botanical petal torsion
        twist_amp = (0.034 if self.is_inner else -0.028)
        angle_s = self.base_angle + twist_amp * (s_vals ** 1.6)
        dir_x = np.cos(angle_s)
        dir_y = np.sin(angle_s)

        # Forward trumpet projection
        z_tube = 0.45 * self.length * np.sin(0.5 * np.pi * (s_vals ** 0.85))

        # Whorl separation ONLY develops after perianth tube (s > 0.14)
        # At s <= 0.14, whorl_z is EXACTLY 0, ensuring seamless continuous perianth tube!
        whorl_factor = np.clip((s_vals - 0.14) / 0.26, 0.0, 1.0)
        if self.is_inner:
            refl_start = 0.52
            refl_amt = 0.28 * self.length * (self.opening ** 1.3)
            refl_pow = 1.82
            whorl_z = 0.014 * self.length * whorl_factor
        else:
            refl_start = 0.42
            refl_amt = 0.48 * self.length * (self.opening ** 1.3)
            refl_pow = 1.62
            whorl_z = -0.014 * self.length * whorl_factor

        # Elegant recurvature curling backwards
        z_refl = refl_amt * (np.maximum(0.0, (s_vals - refl_start) / (1.0 - refl_start)) ** refl_pow)
        z_s = z_tube - z_refl + whorl_z

        # Local Frenet-Serret frame along spine
        ds = 0.001
        s_p = np.clip(s_vals + ds, 0.0, 1.0)
        r_tube_p = (0.024 + 0.076 * s_p) * self.length
        flare_s_p = np.maximum(0.0, (s_p - 0.18) / 0.82)
        r_p = r_tube_p + 0.66 * self.length * self.opening * (flare_s_p ** 1.28)
        angle_p = self.base_angle + twist_amp * (s_p ** 1.6)
        dir_xp = np.cos(angle_p)
        dir_yp = np.sin(angle_p)

        z_tube_p = 0.45 * self.length * np.sin(0.5 * np.pi * (s_p ** 0.85))
        z_refl_p = refl_amt * (np.maximum(0.0, (s_p - refl_start) / (1.0 - refl_start)) ** refl_pow)
        whorl_factor_p = np.clip((s_p - 0.14) / 0.26, 0.0, 1.0)
        whorl_z_p = (0.014 if self.is_inner else -0.014) * self.length * whorl_factor_p
        z_sp = z_tube_p - z_refl_p + whorl_z_p

        tan_x = (r_p * dir_xp - r_s * dir_x) / ds
        tan_y = (r_p * dir_yp - r_s * dir_y) / ds
        tan_z = (z_sp - z_s) / ds
        spine_tan = normalize(np.stack([tan_x, tan_y, tan_z], axis=-1))

        b0 = np.stack([-np.sin(angle_s), np.cos(angle_s), np.zeros_like(angle_s)], axis=-1)
        proj = np.sum(spine_tan * b0, axis=-1, keepdims=True)
        binormals = normalize(b0 - proj * spine_tan, axis=-1)
        normals_spine = normalize(np.cross(spine_tan, binormals), axis=-1)

        # Convex Beta-Harmonic Width Profile w(s)
        if self.is_inner:
            # Broad, voluptuous inner petal (peak at s ≈ 0.52)
            max_w = 0.224 * self.length * self.width_scale * (0.70 + 0.30 * self.opening)
            s_norm = np.clip(s_vals, 0.0, 1.0)
            raw_w = (s_norm ** 0.95) * ((1.0 - s_norm) ** 0.64)
            w_peak = (0.52 ** 0.95) * (0.48 ** 0.64)
            w_shape = raw_w / w_peak
        else:
            # Slender, elegantly keeled outer sepal (peak at s ≈ 0.48)
            max_w = 0.158 * self.length * self.width_scale * (0.70 + 0.30 * self.opening)
            s_norm = np.clip(s_vals, 0.0, 1.0)
            raw_w = (s_norm ** 0.90) * ((1.0 - s_norm) ** 0.68)
            w_peak = (0.48 ** 0.90) * (0.52 ** 0.68)
            w_shape = raw_w / w_peak

        w_prof = max_w * w_shape
        # Ensure airtight tube at base: each tepal covers pi/3 radians
        w_tube = (np.pi / 6.0) * r_s * (np.maximum(0.0, 1.0 - s_vals / 0.20) ** 1.25)
        w_prof = np.maximum(w_prof, w_tube)
        W = w_prof[:, None] * T

        # Transverse Cross-Section Camber h(s, t)
        throat_cup = -0.024 * self.length * np.exp(-16.0 * (s_vals ** 1.8))[:, None] * (1.0 - T ** 2)

        if self.is_inner:
            midrib_camber = (
                -0.026 * self.length * (np.sin(np.pi * (s_vals ** 0.82)))[:, None] * (1.0 - T ** 2)
                + 0.006 * self.length * (np.sin(np.pi * s_vals))[:, None] * np.exp(-18.0 * (T ** 2))
            )
        else:
            midrib_camber = (
                -0.018 * self.length * (np.sin(np.pi * (s_vals ** 0.82)))[:, None] * (1.0 - (T ** 2) ** 1.2)
                + 0.008 * self.length * (np.sin(np.pi * s_vals))[:, None] * np.exp(-14.0 * (T ** 2))
            )

        # Nectary furrow groove
        furrow_mask = (s_vals > 0.02) & (s_vals < 0.34)
        furrow_s = np.where(furrow_mask, np.sin(np.pi * (s_vals - 0.02) / 0.32), 0.0)
        nectar_groove = -0.012 * self.length * furrow_s[:, None] * np.exp(-32.0 * (T ** 2))

        # Margin ruffle fluttering
        ruffle_mask = (S ** 1.1) * (T ** 2)
        edge_ruffle = (
            self.ruffle_amplitude
            * self.length
            * ruffle_mask
            * (np.sin(2.0 * np.pi * self.ruffle_freq * S + self.phase) + 0.35 * np.sin(4.0 * np.pi * self.ruffle_freq * S))
        )

        apex_taper = np.clip((1.0 - S) / 0.04, 0.0, 1.0) ** 0.70
        c_elev = (throat_cup + midrib_camber + nectar_groove + edge_ruffle) * apex_taper

        spine_x = r_s * dir_x
        spine_y = r_s * dir_y

        vx = spine_x[:, None] + W * binormals[:, 0, None] + c_elev * normals_spine[:, 0, None]
        vy = spine_y[:, None] + W * binormals[:, 1, None] + c_elev * normals_spine[:, 1, None]
        vz = z_s[:, None] + W * binormals[:, 2, None] + c_elev * normals_spine[:, 2, None]

        vertices = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
        uvs = np.stack([S, T], axis=-1).reshape(-1, 2)

        faces = []
        for i in range(n_s - 1):
            for j in range(n_t - 1):
                p0 = i * n_t + j
                p1 = i * n_t + (j + 1)
                p2 = (i + 1) * n_t + j
                p3 = (i + 1) * n_t + (j + 1)
                faces.append([p0, p1, p2])
                faces.append([p1, p3, p2])

        faces = np.array(faces, dtype=np.int32)
        normals = compute_mesh_normals(vertices, faces)
        return vertices, faces, normals, uvs


class LilyFlower:
    """
    Complete Lilium blossom model including 6 tepals, 6 stamens with anthers,
    receptacle calyx cup, fluted ovary column, and 1 central pistil with 3-lobed stigma.
    """
    def __init__(
        self,
        flower_id: int,
        center: np.ndarray,
        facing_direction: np.ndarray = None,
        up_direction: np.ndarray = None,
        scale: float = 1.0,
        opening: float = 1.0,
        is_bud: bool = False,
        roll: float = 0.0,
        seed: int = 42,
        facing: np.ndarray = None,
        up: np.ndarray = None
    ):
        self.flower_id = flower_id
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        f_dir = facing if facing is not None else facing_direction
        u_dir = up if up is not None else up_direction
        self.facing = normalize(np.asarray(f_dir, dtype=np.float64).reshape(3))
        self.up = normalize(np.asarray(u_dir, dtype=np.float64).reshape(3))
        self.scale = scale
        self.opening = np.clip(opening, 0.15, 1.0)
        self.is_bud = is_bud
        self.roll = roll
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Orthonormal basis for flower coordinate system
        fwd = self.facing
        side = normalize(np.cross(self.up, fwd))
        up_ortho = normalize(np.cross(fwd, side))

        if abs(self.roll) > 1e-4:
            cos_r = np.cos(self.roll)
            sin_r = np.sin(self.roll)
            side_r = cos_r * side + sin_r * up_ortho
            up_r = -sin_r * side + cos_r * up_ortho
            side = side_r
            up_ortho = up_r

        self.rot_matrix = np.column_stack([side, up_ortho, fwd])
        self.tepals: List[LilyTepal] = []
        self._build_tepals()

    def _build_tepals(self):
        """Construct exactly 6 tepals arranged in two trimerous whorls."""
        base_length = (0.64 if self.is_bud else 0.74) * self.scale
        if self.is_bud:
            for i in range(6):
                ang = i * (np.pi / 3.0)
                tepal = LilyTepal(
                    index=i,
                    is_inner=(i % 2 == 0),
                    base_angle=ang,
                    length=base_length,
                    opening=self.opening,
                    is_bud=True
                )
                self.tepals.append(tepal)
            return

        # Inner petals: 0, 120, 240 deg
        for i in range(3):
            ang = i * (2.0 * np.pi / 3.0) + self.rng.uniform(-0.02, 0.02)
            tepal = LilyTepal(
                index=i,
                is_inner=True,
                base_angle=ang,
                length=base_length * self.rng.uniform(0.98, 1.02),
                opening=self.opening,
                width_scale=self.rng.uniform(0.97, 1.03),
                ruffle_amplitude=0.013 * self.rng.uniform(0.9, 1.1),
                ruffle_freq=self.rng.uniform(2.2, 3.0),
                phase=self.rng.uniform(0.0, 2.0 * np.pi)
            )
            self.tepals.append(tepal)

        # Outer sepals: 60, 180, 300 deg
        for i in range(3):
            ang = np.pi / 3.0 + i * (2.0 * np.pi / 3.0) + self.rng.uniform(-0.02, 0.02)
            tepal = LilyTepal(
                index=3 + i,
                is_inner=False,
                base_angle=ang,
                length=base_length * 1.03 * self.rng.uniform(0.98, 1.02),
                opening=self.opening,
                width_scale=self.rng.uniform(0.96, 1.04),
                ruffle_amplitude=0.009 * self.rng.uniform(0.9, 1.1),
                ruffle_freq=self.rng.uniform(2.2, 3.0),
                phase=self.rng.uniform(0.0, 2.0 * np.pi)
            )
            self.tepals.append(tepal)

    def generate_flower_geometry(self) -> Dict[str, Any]:
        """
        Generates full 3D geometry for this flower:
        - 6 tepals mesh
        - Floral receptacle cup and 3-carpellate ovary column
        - 6 stamens with dome-capped filaments and versatile chocolate-russet anthers
        - Central style and 3-lobed papillate stigma
        """
        all_verts = []
        all_faces = []
        all_normals = []
        all_uvs = []
        all_mat_ids = []
        vert_offset = 0

        # 1. Six Tepals
        if self.is_bud:
            for tepal in self.tepals:
                v, f, n, u = tepal.generate_mesh(n_s=70, n_t=28)
                v_world = self.center + (self.rot_matrix @ v.T).T
                n_world = (self.rot_matrix @ n.T).T
                all_verts.append(v_world)
                all_faces.append(f + vert_offset)
                all_normals.append(n_world)
                all_uvs.append(u)
                all_mat_ids.append(np.full(len(v), 8, dtype=np.int32))  # Bud jade/ivory
                vert_offset += len(v)
        else:
            for tepal in self.tepals:
                v, f, n, u = tepal.generate_mesh(n_s=96, n_t=40)
                v_world = self.center + (self.rot_matrix @ v.T).T
                n_world = (self.rot_matrix @ n.T).T
                all_verts.append(v_world)
                all_faces.append(f + vert_offset)
                all_normals.append(n_world)
                all_uvs.append(u)
                m_id = 1 if tepal.is_inner else 2
                all_mat_ids.append(np.full(len(v), m_id, dtype=np.int32))
                vert_offset += len(v)

        # 2. Floral Receptacle Cup & Central Ovary
        if not self.is_bud:
            n_rad = 20
            n_rings = 5
            z_floor = -0.025 * self.scale
            z_rim = -0.008 * self.scale
            r_rim = 0.024 * self.scale

            rec_verts = []
            rec_uvs = []
            rec_verts.append(np.array([0.0, 0.0, z_floor]))
            rec_uvs.append([0.0, 0.0])

            for r_idx in range(1, n_rings + 1):
                frac = r_idx / n_rings
                r_curr = (frac ** 1.3) * r_rim
                z_curr = z_floor + frac * (z_rim - z_floor)
                for j in range(n_rad):
                    th = j * (2.0 * np.pi / n_rad)
                    rec_verts.append(np.array([r_curr * np.cos(th), r_curr * np.sin(th), z_curr]))
                    rec_uvs.append([frac, j / n_rad])

            rec_faces = []
            for j in range(n_rad):
                jn = (j + 1) % n_rad
                rec_faces.append([0, 1 + j, 1 + jn])

            for r_idx in range(1, n_rings):
                r_start = 1 + (r_idx - 1) * n_rad
                r_next = 1 + r_idx * n_rad
                for j in range(n_rad):
                    jn = (j + 1) % n_rad
                    rec_faces.append([r_start + j, r_start + jn, r_next + jn])
                    rec_faces.append([r_start + j, r_next + jn, r_next + j])

            # Ovary column (3-carpellate fluted cylinder)
            n_ov_slices = 18
            n_ov_steps = 10
            z_ov_start = -0.025 * self.scale
            z_ov_end = 0.070 * self.scale
            r_ov_base = 0.010 * self.scale

            ov_start = len(rec_verts)
            for i in range(n_ov_steps):
                u = i / (n_ov_steps - 1)
                z_ov = z_ov_start + u * (z_ov_end - z_ov_start)
                r_ov = r_ov_base * (1.0 - 0.20 * (u ** 1.5))
                for j in range(n_ov_slices):
                    th = j * (2.0 * np.pi / n_ov_slices)
                    r_c = r_ov * (1.0 + 0.10 * np.cos(3.0 * th))
                    rec_verts.append(np.array([r_c * np.cos(th), r_c * np.sin(th), z_ov]))
                    rec_uvs.append([u, j / n_ov_slices])

            for i in range(n_ov_steps - 1):
                for j in range(n_ov_slices):
                    jn = (j + 1) % n_ov_slices
                    p0 = ov_start + i * n_ov_slices + j
                    p1 = ov_start + i * n_ov_slices + jn
                    p2 = ov_start + (i + 1) * n_ov_slices + j
                    p3 = ov_start + (i + 1) * n_ov_slices + jn
                    rec_faces.append([p0, p1, p3])
                    rec_faces.append([p0, p3, p2])

            rec_verts_local = np.array(rec_verts, dtype=np.float64)
            rec_verts_world = self.center + (self.rot_matrix @ rec_verts_local.T).T
            rec_faces = np.array(rec_faces, dtype=np.int32)
            rec_normals = compute_mesh_normals(rec_verts_world, rec_faces)
            rec_uvs = np.array(rec_uvs, dtype=np.float64)

            all_verts.append(rec_verts_world)
            all_faces.append(rec_faces + vert_offset)
            all_normals.append(rec_normals)
            all_uvs.append(rec_uvs)
            all_mat_ids.append(np.full(len(rec_verts_world), 5, dtype=np.int32))  # Celery lime ovary
            vert_offset += len(rec_verts_world)

        # 3. Reproductive Organs (Only for open blossoms)
        if not self.is_bud and self.opening > 0.4:
            for k in range(6):
                is_inner_stamen = (k % 2 == 0)
                ang = k * (np.pi / 3.0) + np.pi / 6.0 + self.rng.uniform(-0.03, 0.03)

                anther_len = (0.082 if is_inner_stamen else 0.076) * self.scale * self.rng.uniform(0.97, 1.03)
                anther_radius = 0.0070 * self.scale

                r_base = 0.016 * self.scale
                z_base = -0.012 * self.scale

                # Natural botanical stamen cluster: gathered gracefully around style
                r_tip = (0.108 if is_inner_stamen else 0.095) * self.scale * self.opening * self.rng.uniform(0.97, 1.03)
                z_tip = (0.365 if is_inner_stamen else 0.340) * self.scale * (self.opening ** 1.1) * self.rng.uniform(0.97, 1.03)
                sag_tip = -0.010 * self.scale

                u_steps = np.linspace(0.0, 1.0, 14)
                fil_verts = []
                fil_uvs = []

                for u_idx, u in enumerate(u_steps):
                    r_u = r_base + (r_tip - r_base) * (u ** 1.35)
                    z_u = z_base + (z_tip - z_base) * (u ** 0.88) + sag_tip * (u ** 2.2)
                    pt_local = np.array([r_u * np.cos(ang), r_u * np.sin(ang), z_u])

                    r_fil = (0.0024 - 0.0008 * u) * self.scale
                    for p in range(8):
                        phi = p * (2.0 * np.pi / 8.0)
                        disp = r_fil * np.array([
                            -np.sin(ang) * np.sin(phi),
                            np.cos(ang) * np.sin(phi),
                            np.cos(phi)
                        ])
                        fil_verts.append(self.center + self.rot_matrix @ (pt_local + disp))
                        fil_uvs.append([u, p / 8.0])

                fil_tip_world = self.center + self.rot_matrix @ np.array([r_tip * np.cos(ang), r_tip * np.sin(ang), z_tip + sag_tip])

                # Watertight rounded dome cap at filament tip
                tip_cap_idx = len(fil_verts)
                fil_verts.append(fil_tip_world + self.rot_matrix @ np.array([0.0, 0.0, 0.002 * self.scale]))
                fil_uvs.append([1.0, 0.5])

                fil_verts = np.array(fil_verts, dtype=np.float64)
                fil_uvs = np.array(fil_uvs, dtype=np.float64)

                fil_faces = []
                for i in range(len(u_steps) - 1):
                    for j in range(8):
                        jn = (j + 1) % 8
                        p0 = i * 8 + j
                        p1 = i * 8 + jn
                        p2 = (i + 1) * 8 + j
                        p3 = (i + 1) * 8 + jn
                        fil_faces.append([p0, p1, p2])
                        fil_faces.append([p1, p3, p2])

                last_ring_start = (len(u_steps) - 1) * 8
                for j in range(8):
                    jn = (j + 1) % 8
                    fil_faces.append([last_ring_start + j, last_ring_start + jn, tip_cap_idx])

                fil_faces = np.array(fil_faces, dtype=np.int32)
                fil_normals = compute_mesh_normals(fil_verts, fil_faces)

                all_verts.append(fil_verts)
                all_faces.append(fil_faces + vert_offset)
                all_normals.append(fil_normals)
                all_uvs.append(fil_uvs)
                all_mat_ids.append(np.full(len(fil_verts), 3, dtype=np.int32))  # Filament
                vert_offset += len(fil_verts)

                # Versatile Anther: Attached mid-dorsum, velvety chocolate-russet pollen
                yaw = self.rng.uniform(-0.40, 0.40)
                pitch = self.rng.uniform(-0.45, 0.30)
                roll = self.rng.uniform(-0.30, 0.30)
                anther_dir = normalize(np.array([
                    -np.sin(ang) + yaw,
                    np.cos(ang) + pitch,
                    0.18 * np.cos(ang) + roll
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
                    crescent_curve = -0.0032 * self.scale * (1.0 - np.cos(lat))

                    for lon_idx in range(n_lon):
                        lon_frac = lon_idx / n_lon
                        lon = lon_idx * (2.0 * np.pi / n_lon)
                        theca_mod = 1.0 + 0.35 * np.cos(2.0 * lon)
                        rad_curr = anther_radius * theca_mod * np.cos(lat)

                        x_e = (anther_len * 0.5) * np.sin(lat)
                        y_e = rad_curr * np.cos(lon) + crescent_curve
                        z_e = rad_curr * np.sin(lon)

                        disp_local = x_e * anther_dir + y_e * anther_ortho1 + z_e * anther_ortho2
                        anther_verts.append(fil_tip_world + self.rot_matrix @ disp_local)
                        anther_uvs_list.append([lat_frac, lon_frac])

                anther_verts = np.array(anther_verts, dtype=np.float64)
                anther_uvs = np.array(anther_uvs_list, dtype=np.float64)
                anther_faces = []
                for i in range(n_lat - 1):
                    for j in range(n_lon):
                        jn = (j + 1) % n_lon
                        p0 = i * n_lon + j
                        p1 = i * n_lon + jn
                        p2 = (i + 1) * n_lon + j
                        p3 = (i + 1) * n_lon + jn
                        anther_faces.append([p0, p1, p2])
                        anther_faces.append([p1, p3, p2])

                anther_faces = np.array(anther_faces, dtype=np.int32)
                anther_normals = compute_mesh_normals(anther_verts, anther_faces)

                all_verts.append(anther_verts)
                all_faces.append(anther_faces + vert_offset)
                all_normals.append(anther_normals)
                all_uvs.append(anther_uvs)
                all_mat_ids.append(np.full(len(anther_verts), 4, dtype=np.int32))  # Pollen
                vert_offset += len(anther_verts)

            # Central Style and Stigma
            pistil_len = 0.42 * self.scale * self.opening
            style_arch = 0.020 * self.scale
            style_droop = -0.015 * self.scale
            stigma_world = self.center + self.rot_matrix @ np.array([style_arch, style_droop, pistil_len])

            pistil_verts = []
            pistil_uvs_list = []
            n_style_steps = 14
            for u in np.linspace(0.0, 1.0, n_style_steps):
                arch_u = style_arch * (u ** 1.35)
                droop_u = style_droop * (u ** 1.55)
                z_p = 0.070 * self.scale * (1.0 - u) + pistil_len * u
                pt_local = np.array([arch_u, droop_u, z_p])
                r_style = (0.0034 - 0.0008 * u) * self.scale
                for p in range(8):
                    phi = p * (2.0 * np.pi / 8.0)
                    disp = r_style * np.array([np.cos(phi), np.sin(phi), 0.0])
                    pistil_verts.append(self.center + self.rot_matrix @ (pt_local + disp))
                    pistil_uvs_list.append([u, p / 8.0])

            pistil_faces = []
            for i in range(n_style_steps - 1):
                for j in range(8):
                    jn = (j + 1) % 8
                    p0 = i * 8 + j
                    p1 = i * 8 + jn
                    p2 = (i + 1) * 8 + j
                    p3 = (i + 1) * 8 + jn
                    pistil_faces.append([p0, p1, p2])
                    pistil_faces.append([p1, p3, p2])

            # Stigma Cap: 3-lobed papillate crown
            stigma_base_idx = len(pistil_verts)
            n_st_rings = 4
            n_st_rad = 18
            r_stigma_max = 0.0125 * self.scale

            pistil_verts.append(stigma_world + self.rot_matrix @ np.array([0.0, 0.0, 0.007 * self.scale]))
            pistil_uvs_list.append([1.0, 0.0])

            for ring in range(1, n_st_rings + 1):
                frac = ring / n_st_rings
                z_st = 0.007 * self.scale * (1.0 - frac ** 1.5)
                for j in range(n_st_rad):
                    th = j * (2.0 * np.pi / n_st_rad)
                    r_lobe = (frac * r_stigma_max) * (1.0 + 0.32 * np.cos(3.0 * th))
                    disp = np.array([r_lobe * np.cos(th), r_lobe * np.sin(th), z_st])
                    pistil_verts.append(stigma_world + self.rot_matrix @ disp)
                    pistil_uvs_list.append([1.0, j / n_st_rad])

            for j in range(n_st_rad):
                jn = (j + 1) % n_st_rad
                pistil_faces.append([stigma_base_idx, stigma_base_idx + 1 + j, stigma_base_idx + 1 + jn])

            for ring in range(1, n_st_rings):
                r_start = stigma_base_idx + 1 + (ring - 1) * n_st_rad
                r_next = stigma_base_idx + 1 + ring * n_st_rad
                for j in range(n_st_rad):
                    jn = (j + 1) % n_st_rad
                    pistil_faces.append([r_start + j, r_start + jn, r_next + jn])
                    pistil_faces.append([r_start + j, r_next + jn, r_next + j])

            pistil_verts = np.array(pistil_verts, dtype=np.float64)
            pistil_faces = np.array(pistil_faces, dtype=np.int32)
            pistil_normals = compute_mesh_normals(pistil_verts, pistil_faces)
            pistil_uvs = np.array(pistil_uvs_list, dtype=np.float64)

            all_verts.append(pistil_verts)
            all_faces.append(pistil_faces + vert_offset)
            all_normals.append(pistil_normals)
            all_uvs.append(pistil_uvs)
            all_mat_ids.append(np.full(len(pistil_verts), 5, dtype=np.int32))  # Pistil
            vert_offset += len(pistil_verts)

        return {
            "vertices": np.vstack(all_verts),
            "faces": np.vstack(all_faces),
            "normals": np.vstack(all_normals),
            "uvs": np.vstack(all_uvs),
            "mat_ids": np.concatenate(all_mat_ids)
        }

    def generate_geometry(self) -> Dict[str, Any]:
        """Convenience alias for generate_flower_geometry()."""
        return self.generate_flower_geometry()
