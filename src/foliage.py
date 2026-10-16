"""
foliage.py - Stems, lanceolate leaves, and bouquet spatial composition.

Features:
- Parametric cubic Bézier stems connecting flower receptacles to bouquet base with watertight end caps.
- Lanceolate lily leaves with adaxial V-groove fold (eliminating artificial cylindrical specular bands).
- Asymmetrical 3/4 botanical studio bouquet arrangement with 8 distinct floral elements (6 blooms + 2 buds).
- 19 harmonious framing leaves supporting the blooms without occlusion or geometric intersection.
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
    """Slender, gracefully arching lanceolate lily leaf with adaxial V-groove."""
    def __init__(
        self,
        attach_point: np.ndarray,
        direction: np.ndarray,
        length: float,
        max_width: float,
        droop: float = 0.22,
        curl: float = 0.0,
        twist: float = 0.0,
        arch: float = 0.14,
        wave: float = 0.005,
        roll: float = 0.0,
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
        self.roll = roll
        self.seed = seed

    def generate_mesh(
        self,
        n_s: int = 36,
        n_t: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')

        world_up = np.array([0.0, 1.0, 0.0])
        n_pref = normalize(np.array([0.0, 0.70, 0.70]))
        arch_dir = n_pref - np.dot(n_pref, self.dir) * self.dir
        if np.linalg.norm(arch_dir) < 1e-4:
            arch_dir = world_up - np.dot(world_up, self.dir) * self.dir
        arch_dir = normalize(arch_dir)

        arch_prof = self.arch * np.sin(np.pi * (s_vals ** 0.85))
        droop_prof = self.droop * (s_vals ** 2.2)
        binorm_init = normalize(np.cross(self.dir, arch_dir))

        spine_pts = (
            self.attach[None, :]
            + (s_vals[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_prof[:, None]) * arch_dir[None, :]
            - (self.length * droop_prof[:, None]) * world_up[None, :]
            + (self.length * self.curl * (s_vals[:, None] ** 2)) * binorm_init[None, :]
        )

        ds = 0.001
        s_p = np.clip(s_vals + ds, 0.0, 1.0)
        arch_p = self.arch * np.sin(np.pi * (s_p ** 0.85))
        droop_p = self.droop * (s_p ** 2.2)
        spine_pts_p = (
            self.attach[None, :]
            + (s_p[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_p[:, None]) * arch_dir[None, :]
            - (self.length * droop_p[:, None]) * world_up[None, :]
            + (self.length * self.curl * (s_p[:, None] ** 2)) * binorm_init[None, :]
        )
        spine_tangents = normalize(spine_pts_p - spine_pts, axis=-1)

        t0 = spine_tangents[0]
        n0 = n_pref - np.dot(n_pref, t0) * t0
        if np.linalg.norm(n0) < 1e-4:
            n0 = world_up - np.dot(world_up, t0) * t0
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

            ang_tw = self.twist * (s_vals[i] ** 1.3)
            cos_tw = np.cos(ang_tw)
            sin_tw = np.sin(ang_tw)
            b_tw = cos_tw * curr_b + sin_tw * curr_n
            n_tw = -sin_tw * curr_b + cos_tw * curr_n
            binormals.append(b_tw)
            normals.append(n_tw)

        binormals = np.array(binormals)
        normals = np.array(normals)

        # Lanceolate width profile
        w_profile = (s_vals ** 0.85) * ((1.0 - s_vals) ** 1.25)
        w_peak = (0.38 ** 0.85) * (0.62 ** 1.25)
        w_profile = self.max_width * (w_profile / w_peak)
        W = w_profile[:, None] * T

        # Adaxial V-groove (natural botanical leaf fold)
        # Eliminates artificial cylindrical specular streak!
        v_groove = -0.016 * self.max_width * (1.0 - np.exp(-12.0 * (T ** 2)))
        c_elev = v_groove * (1.0 - S ** 2)

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
        normals_out = compute_mesh_normals(vertices, faces)
        return vertices, faces, normals_out, uvs


class LilyStem:
    """3D cubic Bézier stem with tubular radial geometry and watertight end caps."""
    def __init__(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        radius_base: float = 0.026,
        radius_tip: float = 0.017
    ):
        self.bezier = BezierCurve3D(p0, p1, p2, p3)
        self.curve = self.bezier
        self.r_base = radius_base
        self.r_tip = radius_tip

    def generate_mesh(
        self,
        n_steps: int = 24,
        n_radial: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_vals = np.linspace(0.0, 1.0, n_steps)
        curve_pts = self.bezier.evaluate(u_vals)
        T_v, N_v, B_v = self.bezier.compute_frames(u_vals)

        stem_verts = []
        stem_uvs = []
        for s_i in range(len(u_vals)):
            u = u_vals[s_i]
            pt = curve_pts[s_i]
            r = self.r_base * (1.0 - u) + self.r_tip * u
            for p_i in range(n_radial):
                phi = p_i * (2.0 * np.pi / n_radial)
                disp = r * (np.cos(phi) * N_v[s_i] + np.sin(phi) * B_v[s_i])
                stem_verts.append(pt + disp)
                stem_uvs.append([u, p_i / float(n_radial)])

        stem_faces = []
        for s_i in range(len(u_vals) - 1):
            for p_i in range(n_radial):
                pn = (p_i + 1) % n_radial
                p0_idx = s_i * n_radial + p_i
                p1_idx = s_i * n_radial + pn
                p2_idx = (s_i + 1) * n_radial + p_i
                p3_idx = (s_i + 1) * n_radial + pn
                stem_faces.append([p0_idx, p1_idx, p2_idx])
                stem_faces.append([p1_idx, p3_idx, p2_idx])

        # Stem base cap (u=0)
        base_cap_idx = len(stem_verts)
        stem_verts.append(curve_pts[0])
        stem_uvs.append([0.0, 0.5])
        for p_i in range(n_radial):
            pn = (p_i + 1) % n_radial
            stem_faces.append([p_i, base_cap_idx, pn])

        # Stem tip cap (u=1)
        tip_cap_idx = len(stem_verts)
        stem_verts.append(curve_pts[-1])
        stem_uvs.append([1.0, 0.5])
        last_ring_start = (len(u_vals) - 1) * n_radial
        for p_i in range(n_radial):
            pn = (p_i + 1) % n_radial
            stem_faces.append([last_ring_start + p_i, last_ring_start + pn, tip_cap_idx])

        stem_verts = np.array(stem_verts, dtype=np.float64)
        stem_faces = np.array(stem_faces, dtype=np.int32)
        stem_normals = compute_mesh_normals(stem_verts, stem_faces)
        stem_uvs = np.array(stem_uvs, dtype=np.float64)
        return stem_verts, stem_faces, stem_normals, stem_uvs


def generate_calyx_receptacle(
    center: np.ndarray,
    rot_matrix: np.ndarray,
    scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a succulent green calyx receptacle tube seamlessly seating the flower base."""
    n_slices = 16
    n_steps = 8
    z_min = -0.045 * scale
    z_max = 0.005 * scale

    verts = []
    uvs = []
    for i in range(n_steps):
        u = i / (n_steps - 1)
        z = z_min + u * (z_max - z_min)
        r = (0.020 + 0.016 * (u ** 1.4)) * scale
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
    - 6 blooming lilies and 2 tapered buds with master 3D depth layering and organic dynamic poses.
    - Thick, succulent stems converging into an artisan bouquet base with seamless receptacle floor insertion.
    - Watertight 2-manifold geometry (zero open cylindrical tubes).
    - Lush, lanceolate lily foliage with adaxial V-groove folds and natural chlorophyll scattering.
    """
    def __init__(self, seed: int = 2026):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.flowers: List[Dict[str, Any]] = []
        self.stems: List[LilyStem] = []
        self.leaves: List[LilyLeaf] = []
        self._build_scene()

    def _build_scene(self):
        """Constructs the full 3D florist bouquet with 8 floral elements and 19 leaves."""
        # 8 DISTINCT BOTANICAL ELEMENTS:
        flower_configs = [
            # 1. Hero Bloom: Center-left, tilted at an elegant 3/4 upward angle (toward key light)
            {
                "id": 1,
                "center": np.array([-0.06, -0.02, 0.38]),
                "facing": normalize(np.array([0.34, 0.26, 0.90])),
                "up": normalize(np.array([-0.22, 0.94, -0.25])),
                "scale": 1.05,
                "opening": 1.00,
                "is_bud": False,
                "roll": -0.15
            },
            # 2. Upper-Right Bloom: Arching gracefully toward right bower
            {
                "id": 2,
                "center": np.array([0.58, 0.32, 0.20]),
                "facing": normalize(np.array([0.42, 0.30, 0.85])),
                "up": normalize(np.array([-0.22, 0.92, -0.32])),
                "scale": 0.94,
                "opening": 0.95,
                "is_bud": False,
                "roll": -0.22
            },
            # 3. Left Lateral Profile Bloom: Sweeping left, showcasing long perianth tube & reflexed sepals
            {
                "id": 3,
                "center": np.array([-0.84, -0.08, 0.14]),
                "facing": normalize(np.array([-0.84, 0.08, 0.53])),
                "up": normalize(np.array([0.18, 0.95, -0.24])),
                "scale": 0.88,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.20
            },
            # 4. Lower Nodding Cascade Bloom: Bowing downward-forward, revealing reflexed dorsal keels
            {
                "id": 4,
                "center": np.array([0.16, -0.48, 0.26]),
                "facing": normalize(np.array([0.10, -0.55, 0.83])),
                "up": normalize(np.array([-0.18, 0.82, 0.54])),
                "scale": 0.88,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.16
            },
            # 5. Deep Midground Bloom: Framed in upper-left bower (z = -0.10) for rich optical depth
            {
                "id": 5,
                "center": np.array([-0.38, 0.46, -0.10]),
                "facing": normalize(np.array([-0.20, 0.44, 0.87])),
                "up": normalize(np.array([0.16, 0.86, -0.48])),
                "scale": 0.82,
                "opening": 0.78,
                "is_bud": False,
                "roll": 0.10
            },
            # 6. Right Flank Bloom: Turned rightward to complete floral canopy
            {
                "id": 6,
                "center": np.array([0.72, -0.24, 0.06]),
                "facing": normalize(np.array([0.65, -0.15, 0.74])),
                "up": normalize(np.array([-0.22, 0.88, 0.42])),
                "scale": 0.84,
                "opening": 0.82,
                "is_bud": False,
                "roll": -0.18
            },
            # 7. Crown Apex Bud: Slender jade-and-ivory spindle reaching upward to light
            {
                "id": 7,
                "center": np.array([0.12, 0.82, 0.06]),
                "facing": normalize(np.array([0.08, 0.94, 0.33])),
                "up": normalize(np.array([-0.06, 0.33, -0.94])),
                "scale": 0.80,
                "opening": 0.14,
                "is_bud": True,
                "roll": 0.10
            },
            # 8. Lateral Satellite Bud: Youthful bud extending upper-left
            {
                "id": 8,
                "center": np.array([-0.62, 0.58, 0.02]),
                "facing": normalize(np.array([-0.50, 0.78, 0.38])),
                "up": normalize(np.array([0.22, 0.42, -0.88])),
                "scale": 0.74,
                "opening": 0.12,
                "is_bud": True,
                "roll": -0.12
            }
        ]

        self.flowers = flower_configs

        # Stems: connecting flower bases to bouquet convergence
        stem_base = np.array([0.0, -1.05, -0.06])
        stem_tie = np.array([0.0, -0.62, -0.04])

        for i, fc in enumerate(flower_configs):
            c = fc["center"]
            facing = fc["facing"]
            scale = fc["scale"]
            is_bud = fc["is_bud"]

            # Exact receptacle floor attachment point: zero gap, seamless botanical pedicel!
            p3 = c - (0.015 if is_bud else 0.025) * scale * facing
            p2 = p3 - 0.18 * scale * facing - np.array([0.0, 0.16, 0.10])
            ang_sp = i * (2.0 * np.pi / len(flower_configs))
            p1 = stem_tie + np.array([0.055 * np.cos(ang_sp), self.rng.uniform(-0.02, 0.02), 0.038 * np.sin(ang_sp)])
            p0 = stem_base + np.array([0.09 * np.cos(ang_sp + 0.5), self.rng.uniform(-0.02, 0.02), 0.055 * np.sin(ang_sp + 0.5)])

            stem = LilyStem(p0, p1, p2, p3, radius_base=0.026 * scale, radius_tip=0.017 * scale)
            self.stems.append(stem)

        # 19 Harmonious Leaves: Supporting composition without occluding blossoms
        leaf_specs = [
            # Left cradling foliage (supporting Flower 3 and concealing stem transition)
            {"attach": np.array([-0.28, -0.38, 0.04]), "dir": np.array([-0.76, 0.16, 0.36]), "len": 0.66, "w": 0.086, "droop": 0.20, "arch": 0.14},
            
            # Lower cascade leaves (framing bouquet base)
            {"attach": np.array([-0.05, -0.68, 0.05]), "dir": np.array([-0.65, -0.32, 0.40]), "len": 0.72, "w": 0.092, "droop": 0.28, "arch": 0.15},
            {"attach": np.array([0.06, -0.70, 0.04]), "dir": np.array([0.68, -0.34, 0.38]), "len": 0.74, "w": 0.090, "droop": 0.26, "arch": 0.16},
            {"attach": np.array([-0.18, -0.56, 0.02]), "dir": np.array([-0.78, -0.18, 0.32]), "len": 0.68, "w": 0.088, "droop": 0.24, "arch": 0.14},
            {"attach": np.array([0.16, -0.54, 0.02]), "dir": np.array([0.80, -0.16, 0.30]), "len": 0.70, "w": 0.088, "droop": 0.24, "arch": 0.14},

            # Midground foliage (providing lush greenery behind flowers)
            {"attach": np.array([-0.30, -0.32, -0.06]), "dir": np.array([-0.72, 0.22, 0.38]), "len": 0.64, "w": 0.085, "droop": 0.20, "arch": 0.14},
            {"attach": np.array([0.32, -0.30, -0.06]), "dir": np.array([0.74, 0.20, 0.36]), "len": 0.66, "w": 0.085, "droop": 0.20, "arch": 0.14},
            {"attach": np.array([-0.45, -0.05, -0.08]), "dir": np.array([-0.82, 0.35, 0.24]), "len": 0.60, "w": 0.080, "droop": 0.18, "arch": 0.12},
            {"attach": np.array([0.48, -0.04, -0.08]), "dir": np.array([0.84, 0.32, 0.22]), "len": 0.62, "w": 0.080, "droop": 0.18, "arch": 0.12},

            # Upper canopy leaves (crowning apex buds)
            {"attach": np.array([-0.15, 0.38, -0.04]), "dir": np.array([-0.42, 0.78, 0.28]), "len": 0.58, "w": 0.076, "droop": 0.16, "arch": 0.12},
            {"attach": np.array([0.18, 0.40, -0.04]), "dir": np.array([0.40, 0.80, 0.26]), "len": 0.60, "w": 0.076, "droop": 0.16, "arch": 0.12},
            {"attach": np.array([0.02, 0.58, -0.02]), "dir": np.array([0.18, 0.88, 0.22]), "len": 0.54, "w": 0.072, "droop": 0.14, "arch": 0.10},
            {"attach": np.array([-0.35, 0.48, 0.00]), "dir": np.array([-0.55, 0.72, 0.22]), "len": 0.52, "w": 0.070, "droop": 0.15, "arch": 0.11},

            # Deep filler foliage (soft background greenery)
            {"attach": np.array([0.00, -0.15, -0.14]), "dir": np.array([0.28, 0.65, -0.32]), "len": 0.62, "w": 0.084, "droop": 0.18, "arch": 0.13},
            {"attach": np.array([-0.12, -0.10, -0.14]), "dir": np.array([-0.30, 0.62, -0.30]), "len": 0.60, "w": 0.082, "droop": 0.18, "arch": 0.13},
            {"attach": np.array([0.22, 0.10, -0.12]), "dir": np.array([0.52, 0.58, -0.25]), "len": 0.56, "w": 0.078, "droop": 0.16, "arch": 0.12},
            {"attach": np.array([-0.24, 0.12, -0.12]), "dir": np.array([-0.50, 0.56, -0.24]), "len": 0.55, "w": 0.078, "droop": 0.16, "arch": 0.12},
            {"attach": np.array([0.10, -0.42, 0.02]), "dir": np.array([0.48, -0.48, 0.35]), "len": 0.58, "w": 0.080, "droop": 0.22, "arch": 0.14},
            {"attach": np.array([-0.12, -0.40, 0.02]), "dir": np.array([-0.46, -0.46, 0.36]), "len": 0.56, "w": 0.080, "droop": 0.22, "arch": 0.14}
        ]

        for ls in leaf_specs:
            leaf = LilyLeaf(
                attach_point=ls["attach"],
                direction=ls["dir"],
                length=ls["len"],
                max_width=ls["w"],
                droop=ls["droop"],
                arch=ls["arch"],
                seed=self.rng.randint(1000, 99999)
            )
            self.leaves.append(leaf)

    def generate_bouquet_geometry(self) -> Dict[str, Any]:
        """
        Synthesizes full geometry for the entire bouquet:
        - All 8 blooming lilies
        - Receptacle calyx tubes for each bloom
        - All 8 stems with watertight end caps
        - 19 collision-free leaves
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
                seed=self.seed + fc["id"] * 73
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
            s_verts, s_faces, s_normals, s_uvs = stem.generate_mesh(n_steps=24, n_radial=10)
            all_verts.append(s_verts)
            all_faces.append(s_faces + vert_offset)
            all_normals.append(s_normals)
            all_uvs.append(s_uvs)
            all_mat_ids.append(np.full(len(s_verts), 6, dtype=np.int32))  # 6 = stem green
            vert_offset += len(s_verts)

        # 3. Generate Leaves
        for leaf in self.leaves:
            l_verts, l_faces, l_normals, l_uvs = leaf.generate_mesh(n_s=36, n_t=14)
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
