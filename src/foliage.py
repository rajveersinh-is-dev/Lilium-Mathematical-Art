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
    Parametric lanceolate Lilium leaf with realistic botanical arch,
    gravitational droop, longitudinal midrib groove, transverse camber,
    and delicate margin ripples.
    """
    def __init__(
        self,
        attach_point: np.ndarray,
        direction: np.ndarray,
        length: float,
        max_width: float,
        droop: float = 0.42,
        curl: float = 0.04,
        twist: float = 0.0,
        arch: float = 0.18,
        wave: float = 0.012,
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
        """Generates 3D vertices, faces, normals, and UVs for the realistic leaf."""
        rng = np.random.RandomState(self.seed)
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        world_up = np.array([0.0, 1.0, 0.0])
        
        # 1. Botanical spine trajectory:
        # Leaves emerge with initial direction, arch upward gently, then gravity pulls them down
        # in a smooth, continuous, weeping curve
        arch_profile = self.arch * np.sin(np.pi * (s_vals ** 0.75))
        droop_profile = self.droop * (s_vals ** 2.1)
        curl_profile = self.curl * (s_vals ** 1.8)
        
        # Upward anatomical normal for the leaf blade
        leaf_up = world_up - np.dot(world_up, self.dir) * self.dir
        if np.linalg.norm(leaf_up) < 1e-4:
            leaf_up = np.array([0.0, 0.0, 1.0])
        leaf_up = normalize(leaf_up)
        
        # Initial binormal across blade
        leaf_b0 = normalize(np.cross(self.dir, leaf_up))
        
        spine_pts = (
            self.attach[None, :]
            + (s_vals[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_profile[:, None]) * leaf_up[None, :]
            - (self.length * droop_profile[:, None]) * world_up[None, :]
            + (self.length * curl_profile[:, None]) * leaf_b0[None, :]
        )
        
        # Compute dynamic moving frame along the curved spine
        ds = 0.001
        s_plus = np.clip(s_vals + ds, 0.0, 1.0)
        arch_p = self.arch * np.sin(np.pi * (s_plus ** 0.75))
        droop_p = self.droop * (s_plus ** 2.1)
        curl_p = self.curl * (s_plus ** 1.8)
        spine_pts_plus = (
            self.attach[None, :]
            + (s_plus[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_p[:, None]) * leaf_up[None, :]
            - (self.length * droop_p[:, None]) * world_up[None, :]
            + (self.length * curl_p[:, None]) * leaf_b0[None, :]
        )
        
        spine_tangents = normalize(spine_pts_plus - spine_pts, axis=-1)
        
        # Parallel transport / moving frame along spine starting from leaf anatomical frame
        t0 = spine_tangents[0]
        n0 = leaf_up - np.dot(leaf_up, t0) * t0
        if np.linalg.norm(n0) < 1e-4:
            alt_up = np.array([0.0, 0.0, 1.0])
            n0 = alt_up - np.dot(alt_up, t0) * t0
        n0 = normalize(n0)
        b0 = normalize(np.cross(t0, n0))
        
        curr_n = n0
        curr_b = b0
        
        binormals = []
        normals = []
        
        for i in range(n_s):
            if i > 0:
                t_prev = spine_tangents[i - 1]
                t_curr = spine_tangents[i]
                v = np.cross(t_prev, t_curr)
                cos_theta = np.dot(t_prev, t_curr)
                if np.linalg.norm(v) > 1e-6:
                    rot_axis = normalize(v)
                    rot_ang = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                    curr_n = rodrigues_rotation(curr_n, rot_axis, rot_ang)
                curr_b = normalize(np.cross(t_curr, curr_n))
                curr_n = normalize(np.cross(curr_b, t_curr))
            
            # Apply smooth axial twist
            ang_tw = self.twist * (s_vals[i] ** 1.3)
            cos_tw = np.cos(ang_tw)
            sin_tw = np.sin(ang_tw)
            b_tw = cos_tw * curr_b + sin_tw * curr_n
            n_tw = -sin_tw * curr_b + cos_tw * curr_n
            
            binormals.append(b_tw)
            normals.append(n_tw)
            
        binormals = np.array(binormals)
        normals = np.array(normals)
        
        # 2. Slender Lanceolate Width Profile:
        # True lily leaf: peak at s ≈ 0.38, tapering gracefully to acute tip
        w_profile = (s_vals ** 0.85) * ((1.0 - s_vals) ** 1.15)
        w_peak = (0.38 ** 0.85) * (0.62 ** 1.15)
        w_profile = self.max_width * (w_profile / w_peak)
        W = w_profile[:, None] * T
        
        # 3. Botanical 3D Cross-Section:
        # Sunken central midrib canal, transverse convex camber, delicate margin ripples
        phi_wave = rng.uniform(0.0, 2.0 * np.pi)
        midrib_groove = -0.008 * self.max_width * np.exp(-18.0 * (T ** 2))
        transverse_camber = 0.024 * self.max_width * (1.0 - T ** 2)
        margin_wave = self.wave * self.length * (S ** 1.2) * (T ** 2) * np.sin(6.0 * S + phi_wave)
        c_elev = midrib_groove + transverse_camber + margin_wave
        
        vx = spine_pts[:, 0, None] + W * binormals[:, 0, None] + c_elev * normals[:, 0, None]
        vy = spine_pts[:, 1, None] + W * binormals[:, 1, None] + c_elev * normals[:, 1, None]
        vz = spine_pts[:, 2, None] + W * binormals[:, 2, None] + c_elev * normals[:, 2, None]
        
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
        normals_mesh = compute_mesh_normals(vertices, faces)
        return vertices, faces, normals_mesh, uvs


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
        self.curve = self.bezier
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


def generate_raffia_tie(
    center: np.ndarray = np.array([0.0, -0.62, -0.04]),
    radius: float = 0.088,
    height: float = 0.08
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates an artisan coiled natural raffia/twine tie binding the bouquet stems."""
    n_coils = 5
    n_theta = 36
    n_profile = 8
    cord_r = 0.007
    
    verts = []
    uvs = []
    total_steps = n_coils * n_theta
    for i in range(total_steps):
        t_frac = i / float(total_steps - 1)
        theta = t_frac * n_coils * 2.0 * np.pi
        y = center[1] - 0.5 * height + t_frac * height
        r = radius + 0.004 * np.sin(4.0 * theta)
        c_x = center[0] + r * np.cos(theta)
        c_z = center[2] + r * np.sin(theta)
        
        t_vec = normalize(np.array([-np.sin(theta), height / (n_coils * 2.0 * np.pi * radius), np.cos(theta)]))
        n_vec = np.array([np.cos(theta), 0.0, np.sin(theta)])
        b_vec = normalize(np.cross(t_vec, n_vec))
        
        for p in range(n_profile):
            phi = p * (2.0 * np.pi / n_profile)
            disp = cord_r * (np.cos(phi) * n_vec + np.sin(phi) * b_vec)
            verts.append(np.array([c_x + disp[0], y + disp[1], c_z + disp[2]]))
            uvs.append([t_frac, p / float(n_profile)])
            
    verts = np.array(verts, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)
    faces = []
    for i in range(total_steps - 1):
        for p in range(n_profile):
            p_next = (p + 1) % n_profile
            p0 = i * n_profile + p
            p1 = i * n_profile + p_next
            p2 = (i + 1) * n_profile + p
            p3 = (i + 1) * n_profile + p_next
            faces.append([p0, p1, p2])
            faces.append([p1, p3, p2])
            
    faces = np.array(faces, dtype=np.int32)
    normals = compute_mesh_normals(verts, faces)
    return verts, faces, normals, uvs


class BouquetScene:
    """
    Assembles an authentic, florist-grade botanical bouquet containing:
    - 6 blooming lilies and 2 tapered buds with master 3D depth layering and organic dynamic poses.
    - Thick, succulent stems converging into an artisan hand-tied bunch with coiled natural raffia tie.
    - True green calyx/receptacle tubes bridging each bloom seamlessly to its stalk.
    - Lush, broad lanceolate lily foliage organized into supportive collar leaves, side flanks,
      and an upper background fan, with verified collision-free geometry.
    """
    def __init__(self, seed: int = 2026):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.flowers: List[Dict[str, Any]] = []
        self.stems: List[LilyStem] = []
        self.leaves: List[LilyLeaf] = []
        self._build_scene()

    def _build_scene(self):
        """Constructs the full 3D florist bouquet with collision-free foliage."""
        stem_tie = np.array([0.0, -0.62, -0.04])
        stem_base = np.array([0.0, -1.05, -0.06])
        
        # 8 Flowers arranged with authentic florist hierarchy and natural 3D depth:
        flower_configs = [
            # 1. Hero Focal Blossom: Majestic central anchor, broad and radiant
            {
                "id": 1,
                "center": np.array([-0.06, 0.04, 0.44]),
                "facing": normalize(np.array([0.04, 0.12, 0.99])),
                "up": normalize(np.array([-0.08, 0.98, -0.05])),
                "scale": 1.05,
                "opening": 1.00,
                "is_bud": False,
                "roll": 0.15
            },
            # 2. Right Mid-Tier Trumpet: Nestled right of focal bloom, flaring outward
            {
                "id": 2,
                "center": np.array([0.58, -0.06, 0.28]),
                "facing": normalize(np.array([0.38, 0.12, 0.92])),
                "up": normalize(np.array([-0.12, 0.98, -0.15])),
                "scale": 0.96,
                "opening": 0.96,
                "is_bud": False,
                "roll": -0.28
            },
            # 3. Left Mid-Tier Trumpet: Nestled left of focal bloom, flaring outward
            {
                "id": 3,
                "center": np.array([-0.62, 0.10, 0.26]),
                "facing": normalize(np.array([-0.42, 0.15, 0.90])),
                "up": normalize(np.array([0.15, 0.97, -0.18])),
                "scale": 0.96,
                "opening": 0.96,
                "is_bud": False,
                "roll": -0.35
            },
            # 4. Crown Apex Blossom: The tall spire of the bouquet reaching toward the light
            {
                "id": 4,
                "center": np.array([0.04, 0.70, 0.14]),
                "facing": normalize(np.array([0.06, 0.36, 0.93])),
                "up": normalize(np.array([-0.05, 0.88, -0.47])),
                "scale": 0.92,
                "opening": 0.90,
                "is_bud": False,
                "roll": 0.25
            },
            # 5. Lower Cascade Left Blossom: Cradled below hero bloom, nodding downward
            {
                "id": 5,
                "center": np.array([-0.48, -0.38, 0.28]),
                "facing": normalize(np.array([-0.26, -0.34, 0.90])),
                "up": normalize(np.array([0.14, 0.82, 0.55])),
                "scale": 0.90,
                "opening": 0.88,
                "is_bud": False,
                "roll": -0.22
            },
            # 6. Lower Cascade Right Blossom: Cradled below hero bloom, nodding downward-right
            {
                "id": 6,
                "center": np.array([0.44, -0.42, 0.26]),
                "facing": normalize(np.array([0.24, -0.36, 0.90])),
                "up": normalize(np.array([-0.12, 0.82, 0.55])),
                "scale": 0.90,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.22
            },
            # 7. Left Profile Trumpet Blossom: Side silhouette showcasing elegant perianth tube
            {
                "id": 7,
                "center": np.array([-0.96, 0.20, 0.10]),
                "facing": normalize(np.array([-0.74, 0.20, 0.64])),
                "up": normalize(np.array([0.18, 0.92, -0.34])),
                "scale": 0.88,
                "opening": 0.82,
                "is_bud": False,
                "roll": 0.18
            },
            # 8. Upper Right Bower Blossom: Youthful trumpet framing upper right flank
            {
                "id": 8,
                "center": np.array([0.78, 0.42, 0.12]),
                "facing": normalize(np.array([0.52, 0.32, 0.79])),
                "up": normalize(np.array([-0.20, 0.88, -0.42])),
                "scale": 0.86,
                "opening": 0.80,
                "is_bud": False,
                "roll": -0.26
            }
        ]
        
        self.flowers = flower_configs
        
        # Build Stems for each flower
        for i, fc in enumerate(flower_configs):
            f_center = fc["center"]
            f_facing = fc["facing"]
            scale = fc["scale"]
            
            p3 = f_center - 0.18 * scale * f_facing
            p2 = p3 - 0.26 * f_facing - np.array([0.0, 0.16, 0.12])
            
            ang_spiral = i * (2.0 * np.pi / 8.0)
            p1 = stem_tie + np.array([
                0.080 * np.cos(ang_spiral),
                self.rng.uniform(-0.02, 0.02),
                0.050 * np.sin(ang_spiral)
            ])
            
            ang_cut = ang_spiral + 0.65 * np.pi
            p0 = stem_base + np.array([
                0.14 * np.cos(ang_cut),
                self.rng.uniform(-0.03, 0.03),
                0.08 * np.sin(ang_cut)
            ])
            
            stem = LilyStem(p0, p1, p2, p3, radius_base=0.034, radius_tip=0.022)
            self.stems.append(stem)

        # Collision detection: ensures leaves cradle blooms without piercing inner petal bell or occluding focal bloom
        def is_leaf_valid(leaf: LilyLeaf) -> bool:
            verts, _, _, _ = leaf.generate_mesh(n_s=16, n_t=5)
            
            # Guard the hero focal blossom (Bloom 1): no leaf may enter its forward visual cone
            hero_c = flower_configs[0]["center"]
            dx = verts[:, 0] - hero_c[0]
            dy = verts[:, 1] - hero_c[1]
            dist_hero_xy = np.sqrt(dx * dx + dy * dy)
            if np.any((verts[:, 2] > (hero_c[2] - 0.08)) & (dist_hero_xy < 0.62 * flower_configs[0]["scale"])):
                return False

            for fc in flower_configs:
                if fc.get("is_bud", False):
                    continue
                c = fc["center"]
                facing = fc["facing"]
                scale = fc["scale"]
                v_rel = verts - c[None, :]
                d_par = np.sum(v_rel * facing[None, :], axis=1)
                v_perp = v_rel - d_par[:, None] * facing[None, :]
                d_perp = np.linalg.norm(v_perp, axis=1)
                
                # Reject if leaf penetrates inside flower trumpet bell (forward hemisphere)
                inside_bell = (d_par > -0.04 * scale) & (d_par < 0.32 * scale) & (d_perp < 0.36 * scale)
                if np.sum(inside_bell) >= 2:
                    return False
            return True

        # 1. Broad Lanceolate Framing Foliage (30 verified botanical leaves):
        # A. Upper Crown Fan (6 leaves reaching up to frame the top apex bloom)
        for k in range(6):
            ang = -0.55 * np.pi + k * (1.10 * np.pi / 5.0) + self.rng.uniform(-0.06, 0.06)
            attach = np.array([
                0.20 * np.cos(ang),
                self.rng.uniform(0.30, 0.48),
                self.rng.uniform(-0.14, -0.06)
            ])
            dir_vec = normalize(np.array([
                0.65 * np.cos(ang),
                self.rng.uniform(0.75, 0.92),
                self.rng.uniform(-0.10, 0.08)
            ]))
            leaf = LilyLeaf(
                attach_point=attach,
                direction=dir_vec,
                length=self.rng.uniform(1.20, 1.45),
                max_width=self.rng.uniform(0.14, 0.18),
                droop=self.rng.uniform(0.18, 0.28),
                curl=self.rng.uniform(-0.03, 0.03),
                twist=self.rng.uniform(-0.10, 0.10),
                arch=self.rng.uniform(0.18, 0.24),
                wave=0.008,
                seed=int(self.rng.randint(10000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # B. Left & Right Flank Wings (12 leaves framing the sides with lush balance)
        for k in range(3):
            # Left Upper Wing (arching upward-left)
            attach_lu = np.array([
                -self.rng.uniform(0.20, 0.40),
                self.rng.uniform(0.35, 0.55),
                self.rng.uniform(-0.16, -0.06)
            ])
            dir_lu = normalize(np.array([
                -self.rng.uniform(0.85, 1.15),
                self.rng.uniform(0.30, 0.60),
                self.rng.uniform(-0.10, 0.05)
            ]))
            leaf_lu = LilyLeaf(
                attach_point=attach_lu,
                direction=dir_lu,
                length=self.rng.uniform(1.35, 1.65),
                max_width=self.rng.uniform(0.15, 0.19),
                droop=self.rng.uniform(0.25, 0.35),
                curl=-0.02,
                twist=self.rng.uniform(-0.12, 0.12),
                arch=self.rng.uniform(0.22, 0.28),
                wave=0.008,
                seed=int(self.rng.randint(20000, 29999))
            )
            if is_leaf_valid(leaf_lu):
                self.leaves.append(leaf_lu)

            # Left Lower Wing (arching downward-left)
            attach_ll = np.array([
                -self.rng.uniform(0.25, 0.45),
                self.rng.uniform(-0.25, -0.05),
                self.rng.uniform(-0.16, -0.06)
            ])
            dir_ll = normalize(np.array([
                -self.rng.uniform(0.95, 1.25),
                self.rng.uniform(-0.05, 0.20),
                self.rng.uniform(-0.10, 0.05)
            ]))
            leaf_ll = LilyLeaf(
                attach_point=attach_ll,
                direction=dir_ll,
                length=self.rng.uniform(1.35, 1.65),
                max_width=self.rng.uniform(0.15, 0.19),
                droop=self.rng.uniform(0.28, 0.38),
                curl=-0.02,
                twist=self.rng.uniform(-0.12, 0.12),
                arch=self.rng.uniform(0.22, 0.28),
                wave=0.008,
                seed=int(self.rng.randint(25000, 34999))
            )
            if is_leaf_valid(leaf_ll):
                self.leaves.append(leaf_ll)

            # Right Upper Wing (arching upward-right)
            attach_ru = np.array([
                self.rng.uniform(0.20, 0.40),
                self.rng.uniform(0.35, 0.55),
                self.rng.uniform(-0.16, -0.06)
            ])
            dir_ru = normalize(np.array([
                self.rng.uniform(0.85, 1.15),
                self.rng.uniform(0.30, 0.60),
                self.rng.uniform(-0.10, 0.05)
            ]))
            leaf_ru = LilyLeaf(
                attach_point=attach_ru,
                direction=dir_ru,
                length=self.rng.uniform(1.35, 1.65),
                max_width=self.rng.uniform(0.15, 0.19),
                droop=self.rng.uniform(0.25, 0.35),
                curl=0.02,
                twist=self.rng.uniform(-0.12, 0.12),
                arch=self.rng.uniform(0.22, 0.28),
                wave=0.008,
                seed=int(self.rng.randint(30000, 39999))
            )
            if is_leaf_valid(leaf_ru):
                self.leaves.append(leaf_ru)

            # Right Lower Wing (arching downward-right)
            attach_rl = np.array([
                self.rng.uniform(0.25, 0.45),
                self.rng.uniform(-0.25, -0.05),
                self.rng.uniform(-0.16, -0.06)
            ])
            dir_rl = normalize(np.array([
                self.rng.uniform(0.95, 1.25),
                self.rng.uniform(-0.05, 0.20),
                self.rng.uniform(-0.10, 0.05)
            ]))
            leaf_rl = LilyLeaf(
                attach_point=attach_rl,
                direction=dir_rl,
                length=self.rng.uniform(1.35, 1.65),
                max_width=self.rng.uniform(0.15, 0.19),
                droop=self.rng.uniform(0.28, 0.38),
                curl=0.02,
                twist=self.rng.uniform(-0.12, 0.12),
                arch=self.rng.uniform(0.22, 0.28),
                wave=0.008,
                seed=int(self.rng.randint(35000, 44999))
            )
            if is_leaf_valid(leaf_rl):
                self.leaves.append(leaf_rl)

        # C. Lower Supportive Collar (10 leaves arching laterally to cradle the bouquet base)
        for k in range(10):
            ang = -0.92 * np.pi + k * (1.84 * np.pi / 9.0) + self.rng.uniform(-0.05, 0.05)
            attach = np.array([
                0.14 * np.cos(ang),
                self.rng.uniform(-0.45, -0.30),
                0.08 * np.sin(ang) - 0.08
            ])
            dir_c = normalize(np.array([
                1.05 * np.cos(ang),
                self.rng.uniform(0.12, 0.28),
                0.35 * np.sin(ang) - 0.12
            ]))
            leaf = LilyLeaf(
                attach_point=attach,
                direction=dir_c,
                length=self.rng.uniform(1.30, 1.65),
                max_width=self.rng.uniform(0.15, 0.19),
                droop=self.rng.uniform(0.28, 0.38),
                curl=self.rng.uniform(-0.04, 0.04),
                twist=self.rng.uniform(-0.15, 0.15),
                arch=self.rng.uniform(0.22, 0.28),
                wave=0.008,
                seed=int(self.rng.randint(40000, 99999))
            )
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)

        # D. Inter-Floral Cauline Leaves (8 leaves peeking through flower gaps in depth)
        for k in range(8):
            stem_idx = k
            u_attach = self.rng.uniform(0.40, 0.70)
            stem_pt = self.stems[stem_idx].curve.evaluate(np.array([u_attach]))[0]
            
            ang_leaf = k * 137.5 * (np.pi / 180.0) + self.rng.uniform(-0.10, 0.10)
            dir_vec = normalize(np.array([
                0.85 * np.cos(ang_leaf),
                self.rng.uniform(0.25, 0.55),
                0.35 * np.sin(ang_leaf) - 0.16
            ]))
            leaf = LilyLeaf(
                attach_point=stem_pt,
                direction=dir_vec,
                length=self.rng.uniform(0.85, 1.10),
                max_width=self.rng.uniform(0.13, 0.16),
                droop=self.rng.uniform(0.24, 0.34),
                curl=self.rng.uniform(-0.03, 0.03),
                twist=self.rng.uniform(-0.10, 0.10),
                arch=self.rng.uniform(0.20, 0.26),
                wave=0.007,
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

        # 4. Generate Artisan Raffia Hand-Tie Binding
        r_verts, r_faces, r_normals, r_uvs = generate_raffia_tie()
        all_verts.append(r_verts)
        all_faces.append(r_faces + vert_offset)
        all_normals.append(r_normals)
        all_uvs.append(r_uvs)
        all_mat_ids.append(np.full(len(r_verts), 6, dtype=np.int32))  # 6 = organic cord/stem
        vert_offset += len(r_verts)
            
        return {
            "vertices": np.vstack(all_verts),
            "faces": np.vstack(all_faces),
            "normals": np.vstack(all_normals),
            "uvs": np.vstack(all_uvs),
            "mat_ids": np.concatenate(all_mat_ids)
        }

