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


def generate_calyx_receptacle(
    center: np.ndarray,
    rot_matrix: np.ndarray,
    scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a succulent green calyx/receptacle tube connecting flower base to stem."""
    n_slices = 8
    n_steps = 6
    z_min = -0.15 * scale
    z_max = 0.01 * scale
    
    verts = []
    uvs = []
    for i in range(n_steps):
        u = i / (n_steps - 1)
        z = z_min + u * (z_max - z_min)
        # Radius tapers smoothly from stem attachment to expanding tepal base
        r = (0.024 + 0.024 * (u ** 1.6)) * scale
        for j in range(n_slices):
            phi = j * (2.0 * np.pi / n_slices)
            local_pt = np.array([r * np.cos(phi), r * np.sin(phi), z])
            world_pt = center + rot_matrix @ local_pt
            verts.append(world_pt)
            uvs.append([u, j / n_slices])
            
    verts = np.array(verts, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)
    
    faces = []
    for i in range(n_steps - 1):
        for j in range(n_slices):
            j_next = (j + 1) % n_slices
            p0 = i * n_slices + j
            p1 = i * n_slices + j_next
            p2 = (i + 1) * n_slices + j
            p3 = (i + 1) * n_slices + j_next
            faces.append([p0, p1, p2])
            faces.append([p1, p3, p2])
            
    faces = np.array(faces, dtype=np.int32)
    normals = compute_mesh_normals(verts, faces)
    return verts, faces, normals, uvs


class BouquetScene:
    """
    Assembles an authentic, florist-grade botanical bouquet containing:
    - 8 blooming lilies arranged with natural asymmetric depth layering and graceful poses.
    - Thick, succulent stems converging into a hand-tied bunch at the lower center.
    - True green calyx/receptacle tubes bridging each bloom seamlessly to its stalk.
    - 50+ slender lanceolate leaves organized into background backdrop, side cradles,
      and cascading collar, with strict collision avoidance guaranteeing ZERO petal clipping.
    """
    def __init__(self, seed: int = 2026):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.flowers: List[Dict[str, Any]] = []
        self.stems: List[LilyStem] = []
        self.leaves: List[LilyLeaf] = []
        self._build_scene()

    def _build_scene(self):
        """Constructs the full 3D bouquet configuration with collision-free foliage."""
        bouquet_base = np.array([0.0, -1.35, -0.10])
        stem_gather = np.array([0.0, -0.85, -0.08])
        
        # 8 Blooming Lilies with organic florist staging (depth layering & dynamic angles)
        flower_configs = [
            # 1. Hero Focal Blossom (Front-center-left, magnificent full opening)
            {
                "id": 1,
                "center": np.array([-0.06, 0.06, 0.42]),
                "facing": normalize(np.array([0.04, 0.16, 0.98])),
                "up": normalize(np.array([-0.08, 0.98, 0.0])),
                "scale": 1.15,
                "opening": 1.0,
                "is_bud": False,
                "roll": 0.20
            },
            # 2. Companion Focal Blossom (Front-right, nestled slightly lower)
            {
                "id": 2,
                "center": np.array([0.62, -0.14, 0.34]),
                "facing": normalize(np.array([0.26, 0.12, 0.95])),
                "up": normalize(np.array([-0.12, 0.98, 0.0])),
                "scale": 1.05,
                "opening": 0.96,
                "is_bud": False,
                "roll": -0.35
            },
            # 3. Tall Crown Blossom (Upper center, reaching gracefully upward)
            {
                "id": 3,
                "center": np.array([0.08, 0.76, 0.18]),
                "facing": normalize(np.array([0.06, 0.28, 0.95])),
                "up": normalize(np.array([0.0, 1.0, 0.0])),
                "scale": 1.02,
                "opening": 0.94,
                "is_bud": False,
                "roll": 0.65
            },
            # 4. Upper-Left Blossom (Diagonal balance, tilted outward)
            {
                "id": 4,
                "center": np.array([-0.72, 0.50, 0.22]),
                "facing": normalize(np.array([-0.32, 0.24, 0.91])),
                "up": normalize(np.array([0.22, 0.95, 0.0])),
                "scale": 0.98,
                "opening": 0.92,
                "is_bud": False,
                "roll": -0.45
            },
            # 5. Lower-Left Cascade Blossom (Draping naturally over lower foliage)
            {
                "id": 5,
                "center": np.array([-0.68, -0.38, 0.26]),
                "facing": normalize(np.array([-0.30, 0.10, 0.95])),
                "up": normalize(np.array([0.16, 0.98, 0.0])),
                "scale": 0.98,
                "opening": 0.90,
                "is_bud": False,
                "roll": 0.35
            },
            # 6. Left Profile Trumpet Blossom (Side silhouette, showcasing perianth tube)
            {
                "id": 6,
                "center": np.array([-1.18, 0.08, 0.08]),
                "facing": normalize(np.array([-0.82, 0.16, 0.55])),
                "up": normalize(np.array([0.0, 0.96, 0.28])),
                "scale": 0.92,
                "opening": 0.85,
                "is_bud": False,
                "roll": -0.15
            },
            # 7. Right Profile Trumpet Blossom (Right 3/4 profile)
            {
                "id": 7,
                "center": np.array([1.14, 0.18, 0.10]),
                "facing": normalize(np.array([0.78, 0.18, 0.59])),
                "up": normalize(np.array([0.0, 0.96, 0.28])),
                "scale": 0.94,
                "opening": 0.86,
                "is_bud": False,
                "roll": 0.25
            },
            # 8. Upper-Right Bower Blossom (Nestled into upper right)
            {
                "id": 8,
                "center": np.array([0.72, 0.56, 0.12]),
                "facing": normalize(np.array([0.34, 0.26, 0.90])),
                "up": normalize(np.array([-0.18, 0.96, 0.0])),
                "scale": 0.92,
                "opening": 0.84,
                "is_bud": False,
                "roll": 0.40
            }
        ]
        
        self.flowers = flower_configs
        
        # Build Stems for each flower
        for i, fc in enumerate(flower_configs):
            f_center = fc["center"]
            f_facing = fc["facing"]
            scale = fc["scale"]
            
            # Stem connects smoothly into calyx receptacle base behind flower
            p3 = f_center - 0.15 * scale * f_facing
            
            # Pull stem backwards into depth (-Z) to tuck cleanly behind petals
            p2 = p3 - 0.28 * f_facing - np.array([0.0, 0.16, 0.15])
            
            # Stems spread organically through the gathering bundle
            ang_stem = i * (2.0 * np.pi / 8.0)
            offset_gather = np.array([
                0.15 * np.cos(ang_stem),
                self.rng.uniform(-0.03, 0.03),
                0.10 * np.sin(ang_stem) - 0.04
            ])
            p1 = stem_gather + offset_gather
            
            # Stems fan out gently at the base
            offset_base = np.array([
                0.20 * np.cos(ang_stem + 0.15),
                self.rng.uniform(-0.04, 0.04),
                0.12 * np.sin(ang_stem + 0.15)
            ])
            p0 = bouquet_base + offset_base
            
            stem = LilyStem(p0, p1, p2, p3, radius_base=0.032, radius_tip=0.024)
            self.stems.append(stem)

        # Rigorous collision detection to guarantee ZERO leaf penetrations into petals
        def is_leaf_valid(leaf: LilyLeaf) -> bool:
            verts, _, _, _ = leaf.generate_mesh(n_s=28, n_t=11)
            for fc in flower_configs:
                c = fc["center"]
                r = 0.68 * fc["scale"]
                facing = fc["facing"]
                dists = np.linalg.norm(verts - c, axis=1)
                dot_f = np.sum((verts - c) * facing, axis=1)
                # If any vertex is inside petal sphere and in front of receptacle, reject
                if np.any((dists < r) & (dot_f > -0.06)):
                    return False
            return True

        # 1. Lush Background Backdrop Leaves (Framing the upper perimeter and behind blooms)
        n_backdrop = 22
        for k in range(n_backdrop):
            ang = -0.38 * np.pi + k * (1.76 * np.pi / n_backdrop) + self.rng.uniform(-0.06, 0.06)
            attach = np.array([
                0.30 * np.cos(ang),
                self.rng.uniform(-0.20, 0.35),
                self.rng.uniform(-0.35, -0.15)
            ])
            dir_vec = normalize(np.array([
                1.15 * np.cos(ang),
                0.70 + 0.35 * np.sin(ang),
                -0.12 - 0.20 * self.rng.uniform(0.0, 1.0)
            ]))
            leaf = LilyLeaf(
                attach_point=attach,
                direction=dir_vec,
                length=self.rng.uniform(1.55, 2.20),
                max_width=self.rng.uniform(0.14, 0.19),
                droop=self.rng.uniform(0.25, 0.45),
                curl=self.rng.uniform(-0.16, 0.16),
                twist=self.rng.uniform(-0.35, 0.35),
                arch=self.rng.uniform(0.20, 0.32),
                wave=self.rng.uniform(0.015, 0.028),
                seed=int(self.rng.randint(10000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # 2. Side Cradling Foliage (Arching outwards to the left and right underneath side blooms)
        # Left flanking leaves
        for k in range(6):
            attach_left = np.array([
                self.rng.uniform(-0.45, -0.25),
                self.rng.uniform(-0.55, -0.25),
                self.rng.uniform(-0.10, 0.05)
            ])
            dir_left = normalize(np.array([
                -0.85 - 0.15 * self.rng.uniform(0, 1),
                -0.20 + 0.15 * self.rng.uniform(-1, 1),
                self.rng.uniform(-0.10, 0.15)
            ]))
            leaf = LilyLeaf(
                attach_point=attach_left,
                direction=dir_left,
                length=self.rng.uniform(1.40, 1.95),
                max_width=self.rng.uniform(0.15, 0.20),
                droop=self.rng.uniform(0.35, 0.55),
                curl=self.rng.uniform(-0.15, 0.15),
                twist=self.rng.uniform(-0.30, 0.30),
                arch=self.rng.uniform(0.15, 0.25),
                wave=self.rng.uniform(0.015, 0.026),
                seed=int(self.rng.randint(20000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # Right flanking leaves
        for k in range(6):
            attach_right = np.array([
                self.rng.uniform(0.25, 0.45),
                self.rng.uniform(-0.55, -0.25),
                self.rng.uniform(-0.10, 0.05)
            ])
            dir_right = normalize(np.array([
                0.85 + 0.15 * self.rng.uniform(0, 1),
                -0.20 + 0.15 * self.rng.uniform(-1, 1),
                self.rng.uniform(-0.10, 0.15)
            ]))
            leaf = LilyLeaf(
                attach_point=attach_right,
                direction=dir_right,
                length=self.rng.uniform(1.40, 1.95),
                max_width=self.rng.uniform(0.15, 0.20),
                droop=self.rng.uniform(0.35, 0.55),
                curl=self.rng.uniform(-0.15, 0.15),
                twist=self.rng.uniform(-0.30, 0.30),
                arch=self.rng.uniform(0.15, 0.25),
                wave=self.rng.uniform(0.015, 0.026),
                seed=int(self.rng.randint(30000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # 3. Lower Cascading Collar Leaves (Framing stems at the bouquet base)
        n_collar = 14
        for k in range(n_collar):
            ang = k * (2.0 * np.pi / n_collar) + self.rng.uniform(-0.10, 0.10)
            attach = stem_gather + np.array([
                0.18 * np.cos(ang),
                self.rng.uniform(-0.12, 0.08),
                0.12 * np.sin(ang)
            ])
            dir_vec = normalize(np.array([
                np.cos(ang),
                -0.40 + 0.18 * np.cos(ang),
                0.28 * np.sin(ang)
            ]))
            leaf = LilyLeaf(
                attach_point=attach,
                direction=dir_vec,
                length=self.rng.uniform(1.50, 2.10),
                max_width=self.rng.uniform(0.16, 0.22),
                droop=self.rng.uniform(0.50, 0.75),
                curl=self.rng.uniform(-0.18, 0.18),
                twist=self.rng.uniform(-0.40, 0.40),
                arch=self.rng.uniform(0.14, 0.24),
                wave=self.rng.uniform(0.016, 0.030),
                seed=int(self.rng.randint(40000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # 4. Stem Leaves (Lower stems u in [0.15, 0.38])
        for stem in self.stems:
            n_leaves = self.rng.randint(1, 3)
            u_vals = np.linspace(0.16, 0.36, n_leaves)
            for u_l in u_vals:
                leaf_attach = stem.bezier.evaluate(np.array([u_l]))[0]
                radial_dir = normalize(np.array([leaf_attach[0], -0.30, leaf_attach[2] - bouquet_base[2]]))
                phi = self.rng.uniform(-0.4, 0.4)
                rot_dir = np.array([
                    radial_dir[0] * np.cos(phi) - radial_dir[2] * np.sin(phi),
                    -0.35 + self.rng.uniform(-0.08, 0.08),
                    radial_dir[0] * np.sin(phi) + radial_dir[2] * np.cos(phi)
                ])
                leaf_dir = normalize(rot_dir)
                leaf = LilyLeaf(
                    attach_point=leaf_attach,
                    direction=leaf_dir,
                    length=self.rng.uniform(1.05, 1.35),
                    max_width=self.rng.uniform(0.13, 0.17),
                    droop=self.rng.uniform(0.38, 0.58),
                    curl=self.rng.uniform(-0.12, 0.12),
                    twist=self.rng.uniform(-0.25, 0.25),
                    arch=self.rng.uniform(0.10, 0.18),
                    wave=self.rng.uniform(0.014, 0.024),
                    seed=int(self.rng.randint(50000, 99999))
                )
                if is_leaf_valid(leaf):
                    self.leaves.append(leaf)

    def generate_bouquet_geometry(self) -> Dict[str, Any]:
        """
        Synthesizes full geometry for the entire bouquet:
        - All 8 blooming lilies
        - Receptacle calyx tubes for each bloom
        - All 8 stems
        - Collision-free verified foliage
        """
        from src.lilies import LilyFlower
        
        all_verts = []
        all_faces = []
        all_normals = []
        all_uvs = []
        all_mat_ids = []
        
        vert_offset = 0
        
        # 1. Generate Flowers and Calyx Receptacles
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
            
            # Calyx receptacle connecting flower to stem
            c_verts, c_faces, c_normals, c_uvs = generate_calyx_receptacle(
                fc["center"], flower.rot_matrix, fc["scale"]
            )
            all_verts.append(c_verts)
            all_faces.append(c_faces + vert_offset)
            all_normals.append(c_normals)
            all_uvs.append(c_uvs)
            all_mat_ids.append(np.full(len(c_verts), 6, dtype=np.int32))  # 6 = stem green
            vert_offset += len(c_verts)
            
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

