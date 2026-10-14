"""
foliage.py - Stems, lanceolate leaves, and bouquet spatial composition.

Features:
- Parametric cubic Bézier stems connecting flower receptacles to bouquet base.
- Lanceolate lily leaves with longitudinal midrib groove and gravitational droop.
- Cohesive bouquet arrangement with 8 blooming lilies, stems, and lush framing foliage.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from src.geometry import (
    normalize,
    rodrigues_rotation,
    rotation_matrix_from_vectors,
    BezierCurve3D,
    compute_mesh_normals
)


class LilyLeaf:
    """
    Parametric lanceolate lily leaf with organic botanical sigmoid curvature,
    gravitational sag, axial twist, sunken midrib canal, transverse camber,
    and delicate margin undulations.
    """
    def __init__(
        self,
        attach_point: np.ndarray,
        direction: np.ndarray,
        length: float,
        max_width: float,
        droop: float = 0.32,
        curl: float = 0.05,
        twist: float = 0.0,
        arch: float = 0.16,
        wave: float = 0.022,
        seed: int = 100
    ):
        self.attach = np.asarray(attach_point, dtype=np.float64).reshape(3)
        self.dir = normalize(np.asarray(direction, dtype=np.float64).reshape(3))
        self.length = length
        self.max_width = max_width
        self.droop = droop
        self.curl = curl
        self.twist = twist
        self.arch = arch
        self.wave = wave
        self.seed = seed

    def generate_mesh(
        self,
        n_s: int = 42,
        n_t: int = 17
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D vertices, faces, normals, and UVs for the realistic leaf.
        """
        rng = np.random.RandomState(self.seed)
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        # World Up vector is +Y
        world_up = np.array([0.0, 1.0, 0.0])
        # Binormal across the leaf blade (horizontal transverse vector)
        binormal = np.cross(self.dir, world_up)
        b_norm = np.linalg.norm(binormal)
        if b_norm < 1e-4:
            binormal = np.cross(self.dir, np.array([1.0, 0.0, 0.0]))
            b_norm = np.linalg.norm(binormal)
        binormal = binormal / max(1e-6, b_norm)
        
        # Surface normal (perpendicular to blade facing generally upward)
        normal = normalize(np.cross(binormal, self.dir))
        
        # 1. Botanical Sigmoid Midrib Spine Trajectory:
        # Arch rises in the first 35% of length, then gravity droop takes over smoothly
        arch_profile = self.arch * np.sin(np.pi * (s_vals ** 0.80))
        droop_profile = self.droop * (s_vals ** 2.0) + 0.25 * self.droop * (np.maximum(0.0, s_vals - 0.60) ** 2.4)
        curl_profile = self.curl * (s_vals ** 1.8 - 0.35 * (s_vals ** 2.4))
        
        spine_pts = (
            self.attach[None, :]
            + (s_vals[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_profile[:, None]) * normal[None, :]
            - (self.length * droop_profile[:, None]) * world_up[None, :]
            + (self.length * curl_profile[:, None]) * binormal[None, :]
        )  # (n_s, 3)
        
        # 2. Lanceolate Width Profile:
        # Peak width at s ≈ 0.36, tapering smoothly to acute apex at s = 1.0
        s_clamped = np.clip(s_vals, 0.0, 1.0)
        alpha, beta = 0.65, 1.15
        s_peak = alpha / (alpha + beta)
        norm_factor = 1.0 / ((s_peak ** alpha) * ((1.0 - s_peak) ** beta))
        w_profile = self.max_width * norm_factor * (s_clamped ** alpha) * ((1.0 - s_clamped) ** beta)
        
        # Transverse width offset
        W = w_profile[:, None] * T  # (n_s, n_t)
        
        # 3. Axial Helical Torsion (Twisting leaf blade):
        # Leaf twists smoothly around its midrib as it arches outwards
        twist_ang = self.twist * (s_vals ** 1.5) + 0.06 * np.sin(np.pi * s_vals)
        cos_tw = np.cos(twist_ang)[:, None]
        sin_tw = np.sin(twist_ang)[:, None]
        binormal_tw = cos_tw * binormal[None, :] + sin_tw * normal[None, :]
        normal_tw = -sin_tw * binormal[None, :] + cos_tw * normal[None, :]
        
        # 4. Botanical 3D Cross-Section Architecture:
        # - Sunken central midrib canal (ventral sulcus) along t=0
        # - Transverse convex camber of the leaf lamina
        # - Delicate sinusoidal edge ruffles (margin wave)
        phi_wave = rng.uniform(0.0, 2.0 * np.pi)
        midrib_groove = -0.045 * self.max_width * np.exp(-24.0 * (T ** 2)) * (1.0 - 0.30 * S)
        transverse_camber = 0.08 * self.max_width * (1.0 - 0.45 * S) * (np.abs(T) - (T ** 2))
        margin_waves = self.wave * self.length * (S ** 1.3) * (np.abs(T) ** 1.8) * np.sin(10.0 * S + phi_wave)
        
        cross_elevation = midrib_groove + transverse_camber + margin_waves
        
        # Combine midrib spine, transverse width, and cross-section elevation
        vx = spine_pts[:, 0, None] + W * binormal_tw[:, 0, None] + cross_elevation * normal_tw[:, 0, None]
        vy = spine_pts[:, 1, None] + W * binormal_tw[:, 1, None] + cross_elevation * normal_tw[:, 1, None]
        vz = spine_pts[:, 2, None] + W * binormal_tw[:, 2, None] + cross_elevation * normal_tw[:, 2, None]
        
        vertices = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
        uvs = np.stack([S, T], axis=-1).reshape(-1, 2)
        
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


class LilyStem:
    """3D cubic Bézier stem with tubular radial geometry."""
    def __init__(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        radius_base: float = 0.032,
        radius_tip: float = 0.024
    ):
        self.bezier = BezierCurve3D(p0, p1, p2, p3)
        self.r_base = radius_base
        self.r_tip = radius_tip

    def generate_mesh(
        self,
        n_steps: int = 30,
        n_radial: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_vals = np.linspace(0.0, 1.0, n_steps)
        curve_pts = self.bezier.evaluate(u_vals)
        T, N, B = self.bezier.compute_frames(u_vals)
        
        radii = self.r_base * (1.0 - u_vals) + self.r_tip * u_vals
        
        verts = []
        uvs = []
        for i in range(n_steps):
            pt = curve_pts[i]
            r = radii[i]
            n_vec = N[i]
            b_vec = B[i]
            for j in range(n_radial):
                theta = j * (2.0 * np.pi / n_radial)
                radial_disp = r * (np.cos(theta) * n_vec + np.sin(theta) * b_vec)
                verts.append(pt + radial_disp)
                uvs.append([u_vals[i], theta / (2.0 * np.pi)])
                
        verts = np.array(verts, dtype=np.float64)
        uvs = np.array(uvs, dtype=np.float64)
        
        faces = []
        for i in range(n_steps - 1):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                p0_idx = i * n_radial + j
                p1_idx = i * n_radial + j_next
                p2_idx = (i + 1) * n_radial + j
                p3_idx = (i + 1) * n_radial + j_next
                faces.append([p0_idx, p1_idx, p2_idx])
                faces.append([p1_idx, p3_idx, p2_idx])
                
        faces = np.array(faces, dtype=np.int32)
        normals = compute_mesh_normals(verts, faces)
        return verts, faces, normals, uvs


class BouquetScene:
    """
    Assembles a complete, natural botanical bouquet containing:
    - 8 blooming lilies with varied 3D poses, rotations, and scales.
    - Curved stems flowing into a central bouquet base.
    - 32 lanceolate lily leaves radiating outward to frame the composition.
    """
    def __init__(self, seed: int = 2026):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.flowers: List[Dict[str, Any]] = []
        self.stems: List[LilyStem] = []
        self.leaves: List[LilyLeaf] = []
        self._build_scene()

    def _build_scene(self):
        """Constructs the full 3D bouquet configuration."""
        bouquet_base = np.array([0.0, -2.2, -0.4])
        
        # 8 Blooming Lilies with realistic botanical 3D placement
        # Positions arranged in a rich, multi-layered dome bouquet layout
        flower_configs = [
            # 1. Central Hero Blossom (Full Frontal 3/4 Face, magnificent opening)
            {
                "id": 1,
                "center": np.array([0.0, 0.08, 0.40]),
                "facing": np.array([0.04, 0.15, 0.98]),
                "up": np.array([0.0, 1.0, 0.0]),
                "scale": 1.15,
                "opening": 1.0,
                "is_bud": False,
                "roll": 0.22
            },
            # 2. Upper Left Blossom (Facing slightly up and left)
            {
                "id": 2,
                "center": np.array([-0.62, 0.52, 0.30]),
                "facing": np.array([-0.20, 0.22, 0.95]),
                "up": np.array([0.2, 0.95, 0.0]),
                "scale": 1.02,
                "opening": 0.96,
                "is_bud": False,
                "roll": -0.48
            },
            # 3. Upper Right Blossom (Facing slightly up and right)
            {
                "id": 3,
                "center": np.array([0.65, 0.48, 0.32]),
                "facing": np.array([0.22, 0.20, 0.95]),
                "up": np.array([-0.18, 0.96, 0.0]),
                "scale": 1.00,
                "opening": 0.96,
                "is_bud": False,
                "roll": 0.55
            },
            # 4. Lower Left Blossom (Resting forward-left)
            {
                "id": 4,
                "center": np.array([-0.72, -0.38, 0.24]),
                "facing": np.array([-0.35, 0.14, 0.92]),
                "up": np.array([0.15, 0.98, 0.0]),
                "scale": 0.98,
                "opening": 0.92,
                "is_bud": False,
                "roll": 0.38
            },
            # 5. Lower Right Blossom (Resting forward-right)
            {
                "id": 5,
                "center": np.array([0.72, -0.35, 0.26]),
                "facing": np.array([0.32, 0.12, 0.93]),
                "up": np.array([-0.12, 0.98, 0.0]),
                "scale": 1.00,
                "opening": 0.94,
                "is_bud": False,
                "roll": -0.32
            },
            # 6. Deep Top Center Blossom (Receded in depth, core bloom)
            {
                "id": 6,
                "center": np.array([0.02, 0.88, 0.10]),
                "facing": np.array([0.02, 0.28, 0.95]),
                "up": np.array([0.0, 1.0, 0.0]),
                "scale": 0.92,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.72
            },
            # 7. Side Trumpet Blossom Left (Profile view showing throat and perianth tube)
            {
                "id": 7,
                "center": np.array([-1.15, 0.12, 0.05]),
                "facing": np.array([-0.80, 0.18, 0.57]),
                "up": np.array([0.0, 0.95, 0.3]),
                "scale": 0.95,
                "opening": 0.85,
                "is_bud": False,
                "roll": -0.15
            },
            # 8. Three-Quarter Blossom Right (Tilted upward-right)
            {
                "id": 8,
                "center": np.array([1.12, 0.15, 0.08]),
                "facing": np.array([0.75, 0.20, 0.63]),
                "up": np.array([0.0, 0.95, 0.3]),
                "scale": 0.94,
                "opening": 0.85,
                "is_bud": False,
                "roll": 0.28
            }
        ]
        
        self.flowers = flower_configs
        
        # Build Stems for each flower
        for fc in flower_configs:
            f_center = fc["center"]
            f_facing = fc["facing"]
            # Stem attaches smoothly to receptacle base
            if fc["is_bud"]:
                p3 = f_center - 0.04 * f_facing
            else:
                p3 = f_center - 0.12 * fc["scale"] * f_facing
                
            p0 = bouquet_base + np.array([self.rng.uniform(-0.15, 0.15), 0.0, self.rng.uniform(-0.1, 0.1)])
            
            # Intermediate Bézier control points
            mid_z = 0.5 * (p0[2] + p3[2])
            mid_y = 0.5 * (p0[1] + p3[1])
            mid_x = 0.5 * (p0[0] + p3[0])
            
            p1 = np.array([p0[0] * 0.7 + mid_x * 0.3, mid_y - 0.3, mid_z - 0.15])
            p2 = np.array([p3[0] * 0.8 + mid_x * 0.2, mid_y + 0.1, mid_z + 0.1])
            
            stem = LilyStem(p0, p1, p2, p3, radius_base=0.038, radius_tip=0.026)
            self.stems.append(stem)
            
            # Attach leaves along each stem
            n_stem_leaves = self.rng.randint(4, 6)
            u_leaf_vals = np.linspace(0.18, 0.85, n_stem_leaves)
            for u_l in u_leaf_vals:
                leaf_attach = stem.bezier.evaluate(np.array([u_l]))[0]
                T_stem = stem.bezier.tangent(np.array([u_l]))[0]
                
                # Leaf angles outwards around stem
                phi_leaf = self.rng.uniform(0, 2.0 * np.pi)
                ortho_stem = np.array([np.cos(phi_leaf), 0.15 + 0.10 * np.sin(phi_leaf), np.sin(phi_leaf)])
                leaf_dir = normalize(ortho_stem + 0.30 * T_stem)
                
                leaf = LilyLeaf(
                    attach_point=leaf_attach,
                    direction=leaf_dir,
                    length=self.rng.uniform(0.95, 1.45),
                    max_width=self.rng.uniform(0.18, 0.25),
                    droop=self.rng.uniform(0.28, 0.48),
                    curl=self.rng.uniform(-0.10, 0.10),
                    twist=self.rng.uniform(-0.35, 0.35),
                    arch=self.rng.uniform(0.12, 0.22),
                    wave=self.rng.uniform(0.015, 0.030),
                    seed=int(self.rng.randint(1000, 99999))
                )
                self.leaves.append(leaf)
                
        # Multi-tiered botanical collar leaves framing the bouquet perimeter
        # Tier 1: Outer perimeter broad sweeping leaves (14 leaves)
        for k in range(14):
            ang_collar = k * (2.0 * np.pi / 14.0) + self.rng.uniform(-0.12, 0.12)
            attach_collar = bouquet_base + np.array([
                0.32 * np.cos(ang_collar),
                0.35 + self.rng.uniform(-0.10, 0.10),
                0.32 * np.sin(ang_collar)
            ])
            dir_collar = normalize(np.array([
                np.cos(ang_collar),
                -0.18 + 0.25 * np.sin(ang_collar),
                0.35 + 0.25 * np.cos(ang_collar)
            ]))
            collar_leaf = LilyLeaf(
                attach_point=attach_collar,
                direction=dir_collar,
                length=self.rng.uniform(1.65, 2.25),
                max_width=self.rng.uniform(0.26, 0.36),
                droop=self.rng.uniform(0.40, 0.65),
                curl=self.rng.uniform(-0.12, 0.12),
                twist=self.rng.uniform(-0.45, 0.45),
                arch=self.rng.uniform(0.18, 0.28),
                wave=self.rng.uniform(0.020, 0.038),
                seed=int(self.rng.randint(10000, 99999))
            )
            self.leaves.append(collar_leaf)

        # Tier 2: Mid-canopy outward leaves cradling the blooms (10 leaves)
        for k in range(10):
            ang_mid = k * (2.0 * np.pi / 10.0) + self.rng.uniform(-0.15, 0.15)
            attach_mid = bouquet_base + np.array([
                0.22 * np.cos(ang_mid),
                0.75 + self.rng.uniform(-0.10, 0.10),
                0.22 * np.sin(ang_mid)
            ])
            dir_mid = normalize(np.array([
                np.cos(ang_mid),
                0.10 + 0.25 * np.sin(ang_mid),
                0.40 + 0.20 * np.cos(ang_mid)
            ]))
            mid_leaf = LilyLeaf(
                attach_point=attach_mid,
                direction=dir_mid,
                length=self.rng.uniform(1.25, 1.70),
                max_width=self.rng.uniform(0.22, 0.30),
                droop=self.rng.uniform(0.30, 0.45),
                curl=self.rng.uniform(-0.10, 0.10),
                twist=self.rng.uniform(-0.35, 0.35),
                arch=self.rng.uniform(0.15, 0.24),
                wave=self.rng.uniform(0.018, 0.032),
                seed=int(self.rng.randint(50000, 99999))
            )
            self.leaves.append(mid_leaf)

    def generate_bouquet_geometry(self) -> Dict[str, Any]:
        """
        Synthesizes full geometry for the entire bouquet:
        - All 8 blooming lilies
        - All 8 stems
        - All 35+ lanceolate leaves
        """
        from src.lilies import LilyFlower
        
        all_verts = []
        all_faces = []
        all_normals = []
        all_uvs = []
        all_mat_ids = []
        
        vert_offset = 0
        
        # 1. Generate Flowers
        for fc in self.flowers:
            flower = LilyFlower(
                flower_id=fc["id"],
                center=fc["center"],
                facing_direction=fc["facing"],
                up_direction=fc["up"],
                scale=fc["scale"],
                opening=fc["opening"],
                is_bud=fc["is_bud"],
                roll=fc.get("roll", 0.0),
                seed=self.seed + fc["id"] * 101
            )
            geom = flower.generate_flower_geometry()
            
            all_verts.append(geom["vertices"])
            all_faces.append(geom["faces"] + vert_offset)
            all_normals.append(geom["normals"])
            all_uvs.append(geom["uvs"])
            all_mat_ids.append(geom["mat_ids"])
            vert_offset += len(geom["vertices"])
            
        # 2. Generate Stems
        for stem in self.stems:
            s_verts, s_faces, s_normals, s_uvs = stem.generate_mesh(n_steps=24, n_radial=8)
            all_verts.append(s_verts)
            all_faces.append(s_faces + vert_offset)
            all_normals.append(s_normals)
            all_uvs.append(s_uvs)
            all_mat_ids.append(np.full(len(s_verts), 6, dtype=np.int32))  # 6 = stem green
            vert_offset += len(s_verts)
            
        # 3. Generate Leaves
        for leaf in self.leaves:
            l_verts, l_faces, l_normals, l_uvs = leaf.generate_mesh(n_s=44, n_t=17)
            all_verts.append(l_verts)
            all_faces.append(l_faces + vert_offset)
            all_normals.append(l_normals)
            all_uvs.append(l_uvs)
            all_mat_ids.append(np.full(len(l_verts), 7, dtype=np.int32))  # 7 = leaf green
            vert_offset += len(l_verts)
            
        return {
            "vertices": np.vstack(all_verts),
            "faces": np.vstack(all_faces),
            "normals": np.vstack(all_normals),
            "uvs": np.vstack(all_uvs),
            "mat_ids": np.concatenate(all_mat_ids)
        }
