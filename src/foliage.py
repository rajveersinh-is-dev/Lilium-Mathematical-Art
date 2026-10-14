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
    """Parametric lanceolate lily leaf with parabolic cross-section, gravity droop, and axial twist."""
    def __init__(
        self,
        attach_point: np.ndarray,
        direction: np.ndarray,
        length: float,
        max_width: float,
        droop: float = 0.25,
        curl: float = 0.05,
        twist: float = 0.0,
        seed: int = 100
    ):
        self.attach = np.asarray(attach_point, dtype=np.float64).reshape(3)
        self.dir = normalize(np.asarray(direction, dtype=np.float64).reshape(3))
        self.length = length
        self.max_width = max_width
        self.droop = droop
        self.curl = curl
        self.twist = twist
        self.seed = seed

    def generate_mesh(
        self,
        n_s: int = 28,
        n_t: int = 13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D vertices, faces, normals, and UVs for the leaf.
        """
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        # Up vector for leaf
        up_ref = np.array([0.0, 0.0, 1.0])
        binormal = normalize(np.cross(self.dir, up_ref))
        if np.linalg.norm(binormal) < 1e-4:
            binormal = np.array([1.0, 0.0, 0.0])
        normal = normalize(np.cross(binormal, self.dir))
        
        # Midrib spine with gravity droop
        # p(s) = attach + s * L * dir - droop * s^2 * normal + curl * s^3 * binormal
        spine_pts = (
            self.attach[None, :]
            + (s_vals[:, None] * self.length) * self.dir[None, :]
            - (self.droop * (s_vals[:, None] ** 2) * self.length) * normal[None, :]
            + (self.curl * (s_vals[:, None] ** 3) * self.length) * binormal[None, :]
        )  # (n_s, 3)
        
        # Lanceolate width profile: w(s) = W_max * s^0.55 * (1 - s)^1.15
        s_clamped = np.clip(s_vals, 0.0, 1.0)
        s_peak = 0.55 / (0.55 + 1.15)
        norm_factor = 1.0 / ((s_peak ** 0.55) * ((1.0 - s_peak) ** 1.15))
        w_profile = self.max_width * norm_factor * (s_clamped ** 0.55) * ((1.0 - s_clamped) ** 1.15)
        
        W = w_profile[:, None] * T  # (n_s, n_t)
        
        # Axial twist as leaf arches outward
        twist_ang = self.twist * (s_vals ** 1.8)
        cos_tw = np.cos(twist_ang)[:, None]
        sin_tw = np.sin(twist_ang)[:, None]
        binormal_tw = cos_tw * binormal[None, :] + sin_tw * normal[None, :]
        normal_tw = -sin_tw * binormal[None, :] + cos_tw * normal[None, :]
        
        # V-shaped cross section: fold elevation along midrib
        # Center t=0 is lowest, edges are slightly raised
        v_fold = 0.035 * self.max_width * (T ** 2)
        
        vx = spine_pts[:, 0, None] + W * binormal_tw[:, 0, None] + v_fold * normal_tw[:, 0, None]
        vy = spine_pts[:, 1, None] + W * binormal_tw[:, 1, None] + v_fold * normal_tw[:, 1, None]
        vz = spine_pts[:, 2, None] + W * binormal_tw[:, 2, None] + v_fold * normal_tw[:, 2, None]
        
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
                ortho_stem = np.array([np.cos(phi_leaf), np.sin(phi_leaf), 0.20])
                leaf_dir = normalize(ortho_stem + 0.35 * T_stem)
                
                leaf = LilyLeaf(
                    attach_point=leaf_attach,
                    direction=leaf_dir,
                    length=self.rng.uniform(0.85, 1.25),
                    max_width=self.rng.uniform(0.16, 0.22),
                    droop=self.rng.uniform(0.22, 0.36),
                    curl=self.rng.uniform(-0.06, 0.06),
                    seed=int(self.rng.randint(1000, 99999))
                )
                self.leaves.append(leaf)
                
        # 16 Prominent Framing Collar Leaves radiating around the bouquet perimeter
        for k in range(16):
            ang_collar = k * (2.0 * np.pi / 16.0) + self.rng.uniform(-0.10, 0.10)
            attach_collar = bouquet_base + np.array([
                0.28 * np.cos(ang_collar),
                0.40 + self.rng.uniform(-0.1, 0.1),
                0.28 * np.sin(ang_collar)
            ])
            dir_collar = normalize(np.array([
                np.cos(ang_collar),
                -0.12 + 0.30 * np.sin(ang_collar),
                0.32 + 0.25 * np.cos(ang_collar)
            ]))
            collar_leaf = LilyLeaf(
                attach_point=attach_collar,
                direction=dir_collar,
                length=self.rng.uniform(1.25, 1.75),
                max_width=self.rng.uniform(0.22, 0.30),
                droop=self.rng.uniform(0.28, 0.42),
                curl=self.rng.uniform(-0.08, 0.08),
                seed=int(self.rng.randint(10000, 99999))
            )
            self.leaves.append(collar_leaf)

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
            l_verts, l_faces, l_normals, l_uvs = leaf.generate_mesh(n_s=36, n_t=15)
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
