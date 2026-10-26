"""
foliage.py - Stems, lanceolate leaves, and bouquet spatial composition.

Features:
- Parametric cubic Bézier stems connecting flower receptacles to bouquet base.
- Lanceolate lily leaves with longitudinal midrib groove and gravitational droop.
- Cohesive bouquet arrangement with 7 blooming lilies, stems, and lush framing foliage.
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
    delicate margin ripples, and view-conscious blade orientation.
    """
    def __init__(
        self,
        attach_point: np.ndarray,
        direction: np.ndarray,
        length: float,
        max_width: float,
        droop: float = 0.18,
        curl: float = 0.0,
        twist: float = 0.0,
        arch: float = 0.12,
        wave: float = 0.006,
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
        n_s: int = 44,
        n_t: int = 17
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates 3D vertices, faces, normals, and UVs for the realistic leaf."""
        rng = np.random.RandomState(self.seed)
        s_vals = np.linspace(0.0, 1.0, n_s)
        t_vals = np.linspace(-1.0, 1.0, n_t)
        S, T = np.meshgrid(s_vals, t_vals, indexing='ij')
        
        world_up = np.array([0.0, 1.0, 0.0])
        # Preferred blade normal: balanced between upward (+Y) and camera (+Z)
        n_pref = normalize(np.array([0.0, 0.60, 0.80]))
        
        # Arching direction perpendicular to spine direction
        arch_dir = n_pref - np.dot(n_pref, self.dir) * self.dir
        if np.linalg.norm(arch_dir) < 1e-4:
            arch_dir = world_up - np.dot(world_up, self.dir) * self.dir
        arch_dir = normalize(arch_dir)
        
        arch_profile = self.arch * np.sin(np.pi * (s_vals ** 0.80))
        droop_profile = self.droop * (s_vals ** 2.2)
        curl_profile = self.curl * (s_vals ** 1.8)
        
        binorm_init = normalize(np.cross(self.dir, arch_dir))
        
        spine_pts = (
            self.attach[None, :]
            + (s_vals[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_profile[:, None]) * arch_dir[None, :]
            - (self.length * droop_profile[:, None]) * world_up[None, :]
            + (self.length * curl_profile[:, None]) * binorm_init[None, :]
        )
        
        # Tangents along curved spine
        ds = 0.001
        s_plus = np.clip(s_vals + ds, 0.0, 1.0)
        arch_p = self.arch * np.sin(np.pi * (s_plus ** 0.80))
        droop_p = self.droop * (s_plus ** 2.2)
        curl_p = self.curl * (s_plus ** 1.8)
        spine_pts_plus = (
            self.attach[None, :]
            + (s_plus[:, None] * self.length) * self.dir[None, :]
            + (self.length * arch_p[:, None]) * arch_dir[None, :]
            - (self.length * droop_p[:, None]) * world_up[None, :]
            + (self.length * curl_p[:, None]) * binorm_init[None, :]
        )
        spine_tangents = normalize(spine_pts_plus - spine_pts, axis=-1)
        
        # Initial frame at s=0
        t0 = spine_tangents[0]
        n0 = n_pref - np.dot(n_pref, t0) * t0
        if np.linalg.norm(n0) < 1e-4:
            n0 = world_up - np.dot(world_up, t0) * t0
        n0 = normalize(n0)
        b0 = normalize(np.cross(t0, n0))
        
        # Apply initial roll
        cos_r = np.cos(self.roll)
        sin_r = np.sin(self.roll)
        n0_rot = cos_r * n0 + sin_r * b0
        b0_rot = -sin_r * n0 + cos_r * b0
        
        curr_n = n0_rot
        curr_b = b0_rot
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
        
        # Authentic lanceolate width profile:
        # Peak at s ≈ 0.35, slender taper to acute tip at s = 1.0
        w_profile = (s_vals ** 0.80) * ((1.0 - s_vals) ** 1.30)
        w_peak = (0.35 ** 0.80) * (0.65 ** 1.30)
        w_profile = self.max_width * (w_profile / w_peak)
        W = w_profile[:, None] * T
        
        # Anatomical cross section:
        # Longitudinal midrib vein groove and sculpted transverse camber
        phi_wave = rng.uniform(0.0, 2.0 * np.pi)
        midrib_groove = -0.016 * self.max_width * np.exp(-16.0 * (T ** 2))
        transverse_camber = 0.032 * self.max_width * (1.0 - T ** 2)
        margin_wave = self.wave * self.length * (S ** 1.2) * (T ** 2) * np.sin(8.0 * S + phi_wave)
        tip_taper = np.clip((1.0 - S) / 0.08, 0.0, 1.0) ** 0.75
        c_elev = (midrib_groove + transverse_camber + margin_wave) * tip_taper
        
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
    n_slices = 16
    n_steps = 10
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


def get_bundle_boundary(theta: float, y: float) -> np.ndarray:
    """
    Evaluates the smooth elliptical boundary of the stem bundle
    at height y and polar angle theta around the Y-axis.
    """
    dy = y - (-0.62)
    c_x = -0.0055
    c_z = -0.0866 - 0.080 * dy
    # Snug fit: tight grip around stem cluster
    r_x = 0.188 + 0.320 * dy
    r_z = 0.082 + 0.090 * dy
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    denom = np.sqrt((r_z * cos_t) ** 2 + (r_x * sin_t) ** 2)
    r = (r_x * r_z) / max(1e-5, denom)
    
    # Subtle organic waviness of natural stem contours
    r_wave = r + 0.0020 * np.sin(4.0 * theta + 0.4) + 0.0012 * np.cos(6.0 * theta)
    return np.array([c_x + r_wave * cos_t, y, c_z + r_wave * sin_t])


def _sample_stems(stems, n_per_stem: int = 400):
    """Dense samples of every stem centreline: (x, y, z, tube radius)."""
    xs, ys, zs, rs = [], [], [], []
    for st in stems:
        u = np.linspace(0.0, 1.0, n_per_stem)
        C = st.bezier.evaluate(u)
        r = st.r_base * (1.0 - u) + st.r_tip * u
        xs.append(C[:, 0]); ys.append(C[:, 1]); zs.append(C[:, 2]); rs.append(r)
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(rs))


def measure_bundle_envelope(stems, y_lo: float, y_hi: float):
    """
    Support function of the stem cluster's cross-section.

    A cord under tension wraps the *convex hull* of the bundle, so the usable
    boundary is  R(theta, y) = max_k [ c_k(theta, y) + r_k ],  the support
    function of the union of the stem discs. Every stem point p satisfies
    p . n <= R(theta, y), so a cord laid on R + margin can never enter a stem.
    Returns (cx, cz, samples) with (cx, cz) the centroid of the cluster, which
    lies inside the hull and is used as the polar origin.
    """
    X, Y, Z, Rr = _sample_stems(stems)
    m = (Y >= y_lo) & (Y <= y_hi)
    if m.sum() < 8:
        raise ValueError("not enough stem samples across the tie band")
    cx = float(X[m].mean())
    cz = float(Z[m].mean())
    return cx, cz, (X, Y, Z, Rr)


def envelope_radius(samples, cx: float, cz: float, theta: float,
                    y: float, y_slack: float = 0.0015) -> float:
    """
    Radius of the bundle's convex hull at (theta, y), about (cx, cz).

    The height window is widened by each sample's own tube radius, because a
    tilted stem of radius r centred at y' still occupies height y whenever
    |y - y'| <= r. Ignoring this lets a cord slip into the flank of a leaning
    stalk; the window makes the hull conservative, so the wrap stays clear.
    """
    X, Y, Z, Rr = samples
    m = np.abs(Y - y) <= (Rr + y_slack)
    if not m.any():
        return 0.0
    proj = (X[m] - cx) * np.cos(theta) + (Z[m] - cz) * np.sin(theta)
    return float((proj + Rr[m]).max())


def envelope_radius_many(samples, cx: float, cz: float, theta: np.ndarray,
                         y: np.ndarray, y_slack: float = 0.0015) -> np.ndarray:
    """Vectorised envelope_radius over arrays of (theta, y)."""
    X, Y, Z, Rr = samples
    theta = np.atleast_1d(theta)
    y = np.atleast_1d(y)
    out = np.zeros(len(theta))
    for i in range(len(theta)):
        m = np.abs(Y - y[i]) <= (Rr + y_slack)
        if not m.any():
            continue
        out[i] = float(((X[m] - cx) * np.cos(theta[i])
                        + (Z[m] - cz) * np.sin(theta[i]) + Rr[m]).max())
    return out


def _tube_along_polyline(
    pts: np.ndarray,
    radius,
    n_prof: int,
    uv_u_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweeps a circular tube along a polyline using a stable reference frame."""
    n = len(pts)
    ref = np.array([0.0, 0.0, 1.0])
    verts = []
    uvs = []
    for i in range(n):
        nxt = pts[min(i + 1, n - 1)]
        prv = pts[max(i - 1, 0)]
        tan = normalize(nxt - prv)
        if abs(float(np.dot(tan, ref))) > 0.98:
            ref = np.array([0.0, 1.0, 0.0])
        n_vec = normalize(np.cross(tan, ref))
        b_vec = normalize(np.cross(tan, n_vec))
        r = radius(i) if callable(radius) else radius
        for p in range(n_prof):
            phi = p * (2.0 * np.pi / n_prof)
            verts.append(pts[i] + r * (np.cos(phi) * n_vec + np.sin(phi) * b_vec))
            uvs.append([i / float(max(1, n - 1)) * uv_u_scale + 0.1, p / float(n_prof)])
    verts = np.array(verts, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)
    faces = []
    for i in range(n - 1):
        for p in range(n_prof):
            p_next = (p + 1) % n_prof
            p0 = i * n_prof + p
            p1 = i * n_prof + p_next
            p2 = (i + 1) * n_prof + p
            p3 = (i + 1) * n_prof + p_next
            faces.append([p0, p1, p2])
            faces.append([p1, p3, p2])
    return verts, uvs, np.array(faces, dtype=np.int32)


def generate_raffia_tie(stems=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates high-fidelity artisan florist binding threads, laid on the *measured*
    convex hull of the stem cluster so the cord grips the real bundle instead of a
    nominal ellipse:

    1. Dense helical thread coils, pitched at exactly one cord diameter so that
       successive turns touch without self-overlapping
    2. Criss-cross diagonal cinch strands resting on top of the coil layer
    3. Front artisan knot, two bow loops and two drooping thread tails, all placed
       outside the hull so that nothing intersects the stalks
    """
    all_verts = []
    all_faces = []
    all_uvs = []
    v_offset = 0

    y_min, y_max = -0.655, -0.585
    cord_r = 0.0026

    if stems is not None:
        cx, cz, samples = measure_bundle_envelope(stems, y_min - 0.012, y_max + 0.012)

        def hull_r(theta: float, y: float) -> float:
            return envelope_radius(samples, cx, cz, theta, y)

        def boundary(theta: float, y: float) -> np.ndarray:
            R = hull_r(theta, y)
            return np.array([cx + R * np.cos(theta), y, cz + R * np.sin(theta)])
    else:  # legacy fallback: nominal ellipse
        cx, cz = -0.0055, -0.0866

        def hull_r(theta: float, y: float) -> float:
            p = get_bundle_boundary(theta, y)
            return float(np.hypot(p[0] - cx, p[2] - cz))

        def boundary(theta: float, y: float) -> np.ndarray:
            return get_bundle_boundary(theta, y)

    # Pitch the helix at no less than one cord diameter, so successive turns
    # touch the way real twine does instead of interpenetrating.
    coil_span = (y_max - y_min) - 0.006
    n_coils = max(1, int(np.floor(coil_span / (2.0 * cord_r))))
    pitch = coil_span / n_coils
    assert pitch >= 2.0 * cord_r, "coil pitch must not be tighter than one cord diameter"
    steps_per_coil = 42
    total_steps = n_coils * steps_per_coil
    n_tube_prof = 14

    coil_centres = []
    for i in range(total_steps):
        t_frac = i / float(total_steps - 1)
        theta = t_frac * n_coils * 2.0 * np.pi
        y = y_min + 0.003 + t_frac * (y_max - y_min - 0.006)
        n_dir = np.array([np.cos(theta), 0.0, np.sin(theta)])
        coil_centres.append(boundary(theta, y) + (cord_r * 1.02) * n_dir)
    coil_centres = np.array(coil_centres)

    coil_verts = []
    coil_uvs = []
    for i in range(total_steps):
        centre = coil_centres[i]
        nxt = coil_centres[min(i + 1, total_steps - 1)]
        prv = coil_centres[max(i - 1, 0)]
        tangent = normalize(nxt - prv)
        theta = (i / float(total_steps - 1)) * n_coils * 2.0 * np.pi
        n_raw = np.array([np.cos(theta), 0.0, np.sin(theta)])
        binormal = normalize(np.cross(tangent, n_raw))
        normal = normalize(np.cross(binormal, tangent))
        for p in range(n_tube_prof):
            phi = p * (2.0 * np.pi / n_tube_prof)
            coil_verts.append(centre + cord_r * (np.cos(phi) * normal + np.sin(phi) * binormal))
            coil_uvs.append([(i / float(total_steps - 1)) * n_coils + 0.1, p / float(n_tube_prof)])

    coil_verts = np.array(coil_verts, dtype=np.float64)
    coil_uvs = np.array(coil_uvs, dtype=np.float64)
    coil_faces = []
    for i in range(total_steps - 1):
        for p in range(n_tube_prof):
            p_next = (p + 1) % n_tube_prof
            p0 = i * n_tube_prof + p
            p1 = i * n_tube_prof + p_next
            p2 = (i + 1) * n_tube_prof + p
            p3 = (i + 1) * n_tube_prof + p_next
            coil_faces.append([p0, p1, p2])
            coil_faces.append([p1, p3, p2])

    all_verts.append(coil_verts)
    all_faces.append(np.array(coil_faces, dtype=np.int32) + v_offset)
    all_uvs.append(coil_uvs)
    v_offset += len(coil_verts)

    # --- 2. criss-cross cinch strands, resting on top of the coil layer ---
    diag_r = 0.90 * cord_r
    diag_off = 2.02 * cord_r + diag_r
    diag_steps = 48
    for sign in [1.0, -1.0]:
        diag_pts = []
        for i in range(diag_steps):
            u = i / float(diag_steps - 1)
            y = y_min + 0.005 + u * (y_max - y_min - 0.010)
            theta = 0.50 * np.pi + sign * (u - 0.5) * 1.30 * np.pi
            n_dir = np.array([np.cos(theta), 0.0, np.sin(theta)])
            diag_pts.append(boundary(theta, y) + diag_off * n_dir)
        d_verts, d_uvs, d_faces = _tube_along_polyline(np.array(diag_pts), diag_r, 10, 3.5)
        all_verts.append(d_verts)
        all_faces.append(d_faces + v_offset)
        all_uvs.append(d_uvs)
        v_offset += len(d_verts)

    # --- 3. artisan knot, seated on the front of the hull ---
    theta_front = 0.50 * np.pi
    y_knot = -0.620
    knot_half_z = 0.011
    knot_center = np.array([
        cx,
        y_knot,
        cz + hull_r(theta_front, y_knot) + diag_off + knot_half_z,
    ])

    n_k_lat = 12
    n_k_lon = 14
    knot_verts = []
    knot_uvs = []
    for la in range(n_k_lat):
        v_lat = la / float(n_k_lat - 1)
        lat = -0.5 * np.pi + v_lat * np.pi
        for lo in range(n_k_lon):
            lon = lo * (2.0 * np.pi / n_k_lon)
            k_disp = np.array([
                0.014 * np.cos(lat) * np.cos(lon),
                0.011 * np.sin(lat),
                knot_half_z * np.cos(lat) * np.sin(lon),
            ])
            knot_verts.append(knot_center + k_disp)
            knot_uvs.append([v_lat + 0.1, lo / float(n_k_lon)])

    knot_verts = np.array(knot_verts, dtype=np.float64)
    knot_uvs = np.array(knot_uvs, dtype=np.float64)
    knot_faces = []
    for la in range(n_k_lat - 1):
        for lo in range(n_k_lon):
            lo_next = (lo + 1) % n_k_lon
            p0 = la * n_k_lon + lo
            p1 = la * n_k_lon + lo_next
            p2 = (la + 1) * n_k_lon + lo
            p3 = (la + 1) * n_k_lon + lo_next
            knot_faces.append([p0, p1, p2])
            knot_faces.append([p1, p3, p2])

    all_verts.append(knot_verts)
    all_faces.append(np.array(knot_faces, dtype=np.int32) + v_offset)
    all_uvs.append(knot_uvs)
    v_offset += len(knot_verts)

    # --- 4. two bow loops, swept forward of the knot so they clear the stalks ---
    loop_r = 0.0038
    for loop_sign in [-1.0, 1.0]:
        n_l_steps = 28
        angles = np.linspace(0.0, 2.0 * np.pi, n_l_steps)
        loop_pts = []
        for a in angles:
            lx = knot_center[0] + loop_sign * (0.026 * (np.sin(a) ** 2) * (1.0 + 0.35 * np.cos(a)))
            ly = knot_center[1] + 0.018 * np.sin(a)
            lz = knot_center[2] + 0.010 * (1.0 - np.cos(a))
            loop_pts.append(np.array([lx, ly, lz]))
        l_verts, l_uvs, l_faces = _tube_along_polyline(np.array(loop_pts), loop_r, 10, 1.0)
        all_verts.append(l_verts)
        all_faces.append(l_faces + v_offset)
        all_uvs.append(l_uvs)
        v_offset += len(l_verts)

    # --- 5. two drooping tails, held in front of the stalks at every height ---
    tail_cfg = [
        {"dx0": -0.006, "dx3": -0.030, "y_end": -0.792, "r": 0.0042, "n_steps": 32},
        {"dx0": 0.006, "dx3": 0.036, "y_end": -0.762, "r": 0.0039, "n_steps": 28},
    ]

    def front_z(y: float) -> float:
        if stems is not None:
            return cz + hull_r(theta_front, y)
        return knot_center[2]

    for tc in tail_cfg:
        ys = np.linspace(-0.628, tc["y_end"], tc["n_steps"])
        pts = []
        for k, y in enumerate(ys):
            u = k / float(tc["n_steps"] - 1)
            x = knot_center[0] + tc["dx0"] + (tc["dx3"] - tc["dx0"]) * (u ** 1.4)
            z = front_z(y) + tc["r"] + 0.0035
            pts.append(np.array([x, y, z]))
        pts = np.array(pts)
        t_verts, t_uvs, t_faces = _tube_along_polyline(
            pts,
            lambda i, r0=tc["r"], n=tc["n_steps"]: r0 * (1.0 - 0.20 * ((i / float(n - 1)) ** 2)),
            10, 6.0)
        all_verts.append(t_verts)
        all_faces.append(t_faces + v_offset)
        all_uvs.append(t_uvs)
        v_offset += len(t_verts)

    final_verts = np.vstack(all_verts)
    final_faces = np.vstack(all_faces)
    final_uvs = np.vstack(all_uvs)
    final_normals = compute_mesh_normals(final_verts, final_faces)
    return final_verts, final_faces, final_normals, final_uvs


class BouquetScene:
    """
    Assembles an authentic, florist-grade botanical bouquet containing:
    - 7 blooming lilies with master 3D depth layering and organic dynamic poses.
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
        
        # 9 Botanical Elements with authentic florist hierarchy, varied viewing angles,
        # natural developmental stages (open trumpets, side profiles, deep midground, and fresh developing buds)
        flower_configs = [
            # 1. Hero Focal Blossom: Majestic central anchor, broad and radiant
            {
                "id": 1,
                "center": np.array([0.00, 0.00, 0.55]),
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
                "center": np.array([0.80, -0.10, 0.38]),
                "facing": normalize(np.array([0.38, 0.10, 0.92])),
                "up": normalize(np.array([-0.12, 0.98, -0.15])),
                "scale": 0.96,
                "opening": 0.96,
                "is_bud": False,
                "roll": -0.28
            },
            # 3. Left Profile Trumpet Blossom: Dramatic 3D side profile showcasing long perianth tube & reflexed sepals
            {
                "id": 3,
                "center": np.array([-0.95, 0.10, 0.15]),
                "facing": normalize(np.array([-0.86, 0.14, 0.48])),
                "up": normalize(np.array([0.18, 0.94, -0.28])),
                "scale": 0.90,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.18
            },
            # 4. Upper Right Bower Blossom: Youthful trumpet framing upper right flank
            {
                "id": 4,
                "center": np.array([0.70, 0.45, 0.10]),
                "facing": normalize(np.array([0.48, 0.34, 0.80])),
                "up": normalize(np.array([-0.20, 0.88, -0.42])),
                "scale": 0.88,
                "opening": 0.84,
                "is_bud": False,
                "roll": -0.26
            },
            # 5. Lower Cascade Left Blossom: Cradled below hero bloom, nodding downward
            {
                "id": 5,
                "center": np.array([-0.55, -0.42, 0.45]),
                "facing": normalize(np.array([-0.22, -0.38, 0.89])),
                "up": normalize(np.array([0.14, 0.82, 0.55])),
                "scale": 0.90,
                "opening": 0.88,
                "is_bud": False,
                "roll": -0.22
            },
            # 6. Lower Cascade Right Blossom: Cradled below hero bloom, nodding downward-right
            {
                "id": 6,
                "center": np.array([0.55, -0.48, 0.40]),
                "facing": normalize(np.array([0.24, -0.38, 0.89])),
                "up": normalize(np.array([-0.12, 0.82, 0.55])),
                "scale": 0.90,
                "opening": 0.88,
                "is_bud": False,
                "roll": 0.22
            },
            # 7. Deep Midground Blossom: Nestled deeper within the bouquet, creating true 3D spatial depth
            {
                "id": 7,
                "center": np.array([-0.30, 0.42, -0.05]),
                "facing": normalize(np.array([-0.18, 0.42, 0.88])),
                "up": normalize(np.array([0.12, 0.88, -0.45])),
                "scale": 0.84,
                "opening": 0.74,
                "is_bud": False,
                "roll": 0.10
            }
        ]
        
        self.flowers = flower_configs
        
        # Build Stems for each flower and bud
        n_fl = len(flower_configs)
        for i, fc in enumerate(flower_configs):
            f_center = fc["center"]
            f_facing = fc["facing"]
            scale = fc["scale"]
            is_bud = fc.get("is_bud", False)
            
            p3_offset = (0.10 if is_bud else 0.18) * scale
            p3 = f_center - p3_offset * f_facing
            p2 = p3 - 0.26 * f_facing - np.array([0.0, 0.16, 0.12])
            
            ang_spiral = i * (2.0 * np.pi / n_fl)
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

        # Swept volume of the artisan raffia tie, so no leaf may be threaded
        # through the binding. Margin covers the knot, loops and tails.
        try:
            tie_cx, tie_cz, tie_samples = measure_bundle_envelope(self.stems, -0.80, -0.575)
        except Exception:
            tie_cx = tie_cz = tie_samples = None
        TIE_Y_LO, TIE_Y_HI, TIE_MARGIN = -0.800, -0.575, 0.020

        # Stem centreline samples, for keeping blades out of the stalks.
        # Kept per stem with a bounding box so each candidate leaf is only
        # tested against the stalks it could actually reach.
        stem_boxes = []
        for _st in self.stems:
            _u = np.linspace(0.0, 1.0, 120)
            _C = _st.bezier.evaluate(_u)
            _r = _st.r_base * (1.0 - _u) + _st.r_tip * _u
            _pad = float(_r.max())
            stem_boxes.append((_C.min(axis=0) - _pad, _C.max(axis=0) + _pad, _C, _r))
        stem_pts = stem_boxes if stem_boxes else None
        STEM_BLADE_CLEAR = 0.0015   # blade must stand off the stalk surface
        STEM_BASE_EMBED = 0.0070    # the petiole may meet the stalk it grows from

        # Collision detection: guarantees leaves cradle blooms without piercing perianths or clipping into each other
        accepted_sample_pts: List[np.ndarray] = []

        def _leaf_clears_flowers(attach: np.ndarray, direction: np.ndarray, length: float, f_configs: List[Dict[str, Any]]) -> Tuple[bool, Any]:
            leaf_tip = attach + length * direction
            leaf_mid = (attach + leaf_tip) * 0.5
            leaf_sphere_r = length * 0.55
            for fc in f_configs:
                fc_center = np.asarray(fc["center"])
                petal_sphere_r = 0.82 * fc["scale"]   # ~max petal radius
                if np.linalg.norm(leaf_mid - fc_center) < leaf_sphere_r + petal_sphere_r:
                    return False, fc_center
            return True, None

        def _resolve_leaf_collision(attach: np.ndarray, dir_vec: np.ndarray, length: float) -> Tuple[np.ndarray, float]:
            clears, off_center = _leaf_clears_flowers(attach, dir_vec, length, flower_configs)
            if clears:
                return dir_vec, length
            curr_dir = dir_vec.copy()
            for _ in range(5):
                away = normalize(attach - off_center)
                if np.linalg.norm(away) < 1e-4:
                    away = np.array([0.0, 1.0, 0.0])
                rot_axis = normalize(np.cross(curr_dir, away))
                if np.linalg.norm(rot_axis) < 1e-4:
                    rot_axis = np.array([0.0, 0.0, 1.0])
                curr_dir = normalize(rodrigues_rotation(curr_dir, rot_axis, np.radians(20.0)))
                clears_rot, off_center = _leaf_clears_flowers(attach, curr_dir, length, flower_configs)
                if clears_rot:
                    return curr_dir, length
            return curr_dir, length * 0.75

        def is_leaf_valid(leaf: LilyLeaf) -> bool:
            verts, _, _, uvs = leaf.generate_mesh(n_s=30, n_t=11)
            s_coords = uvs[:, 0]
            
            # 1. Guard Hero focal blossom (Bloom 1): no leaf may enter its forward visual cone
            hero_c = flower_configs[0]["center"]
            dx = verts[:, 0] - hero_c[0]
            dy = verts[:, 1] - hero_c[1]
            dist_hero_xy = np.sqrt(dx * dx + dy * dy)
            if np.any((verts[:, 2] > (hero_c[2] - 0.08)) & (dist_hero_xy < 0.62 * flower_configs[0]["scale"])):
                return False

            # 2. Strict perianth exclusion volume for all flowers
            for fc in flower_configs:
                c = fc["center"]
                facing = fc["facing"]
                scale = fc["scale"]
                v_rel = verts - c[None, :]
                d_par = np.sum(v_rel * facing[None, :], axis=1)
                v_perp = v_rel - d_par[:, None] * facing[None, :]
                d_perp = np.linalg.norm(v_perp, axis=1)
                
                # Trumpet perianth or bud spindle conical envelope
                is_b = fc.get("is_bud", False)
                if is_b:
                    in_longitudinal = (d_par >= -0.06 * scale) & (d_par <= 0.70 * scale)
                    r_envelope = scale * 0.18 + 0.03
                else:
                    in_longitudinal = (d_par >= -0.10 * scale) & (d_par <= 0.46 * scale)
                    r_envelope = scale * np.clip(0.18 + 1.10 * np.maximum(0.0, d_par / scale + 0.10), 0.18, 0.65) + 0.04
                if np.any(in_longitudinal & (d_perp <= r_envelope)):
                    return False

            # 3. Keep clear of the artisan raffia tie's swept volume
            if tie_samples is not None:
                in_band = (verts[:, 1] > TIE_Y_LO) & (verts[:, 1] < TIE_Y_HI)
                if np.any(in_band):
                    # re-sample finely: the coarse test mesh can bridge across
                    # the tie between two of its own vertices
                    fine, _, _, _ = leaf.generate_mesh(n_s=44, n_t=16)
                    fband = (fine[:, 1] > TIE_Y_LO) & (fine[:, 1] < TIE_Y_HI)
                    P = fine[fband]
                    if len(P):
                        th = np.arctan2(P[:, 2] - tie_cz, P[:, 0] - tie_cx)
                        R = envelope_radius_many(tie_samples, tie_cx, tie_cz, th, P[:, 1])
                        if np.any(np.hypot(P[:, 0] - tie_cx, P[:, 2] - tie_cz) < R + TIE_MARGIN):
                            return False

            # 4. Blades must not pass through a stem. The basal petiole is
            #    exempt from the stand-off: that is where the leaf is attached.
            if stem_pts is not None:
                lo_b, hi_b = verts.min(axis=0), verts.max(axis=0)
                near = [b for b in stem_pts
                        if not (np.any(hi_b < b[0]) or np.any(lo_b > b[1]))]
                if not near:
                    SP = SR = None
                else:
                    SP = np.vstack([b[2] for b in near])
                    SR = np.concatenate([b[3] for b in near])
                s_here = uvs[:, 0]
                for is_blade in ((True, False) if SP is not None else ()):
                    sel = (s_here > 0.18) if is_blade else (s_here <= 0.18)
                    if not np.any(sel):
                        continue
                    P = verts[sel]
                    d = np.linalg.norm(P[:, None, :] - SP[None, :, :], axis=2)
                    nearest = d.argmin(axis=1)
                    clearance = d[np.arange(len(P)), nearest] - SR[nearest]
                    limit = STEM_BLADE_CLEAR if is_blade else -STEM_BASE_EMBED
                    if float(clearance.min()) < limit:
                        return False

            # 5. Leaf-to-Leaf Collision Detection (blades must not intersect)
            blade_mask = s_coords > 0.18
            candidate_blade = verts[blade_mask]
            
            if len(accepted_sample_pts) > 0 and len(candidate_blade) > 0:
                cand_pts = candidate_blade[::2]
                all_accepted = np.vstack(accepted_sample_pts)
                diff = cand_pts[:, None, :] - all_accepted[None, :, :]
                dist_sq = np.sum(diff * diff, axis=-1)
                min_dist = np.sqrt(np.min(dist_sq))
                if min_dist < 0.055:
                    return False
                    
            return True

        def try_add_leaf(leaf: LilyLeaf) -> bool:
            if is_leaf_valid(leaf):
                self.leaves.append(leaf)
                verts, _, _, uvs = leaf.generate_mesh(n_s=30, n_t=11)
                blade_pts = verts[uvs[:, 0] > 0.18][::2]
                accepted_sample_pts.append(blade_pts)
                return True
            return False

        # 1. Broad Lanceolate Framing Foliage (Verified Botanical Lilium Foliage):
        # A. Upper Crown Fan (Framing the crown apex bloom from behind, Z in [-0.25, -0.12])
        fan_angles = np.linspace(-0.62 * np.pi, 0.62 * np.pi, 12)
        for ang in fan_angles:
            if abs(ang) < 0.16:
                continue
            for attempt in range(50):
                a_jitter = ang + self.rng.uniform(-0.05, 0.05)
                attach = np.array([
                    0.22 * np.cos(a_jitter),
                    self.rng.uniform(0.36, 0.54),
                    self.rng.uniform(-0.25, -0.12)
                ])
                dir_vec = normalize(np.array([
                    0.85 * np.cos(a_jitter),
                    self.rng.uniform(0.65, 0.88),
                    self.rng.uniform(-0.16, -0.04)
                ]))
                length = self.rng.uniform(0.88, 1.20)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=attach,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.13, 0.18),
                    droop=self.rng.uniform(0.24, 0.36),
                    curl=self.rng.uniform(-0.06, 0.06),
                    twist=self.rng.uniform(-0.35, 0.35),
                    arch=self.rng.uniform(0.14, 0.22),
                    wave=self.rng.uniform(0.006, 0.012),
                    roll=self.rng.uniform(-0.40, 0.40),
                    seed=int(self.rng.randint(10000, 99999))
                )
                if try_add_leaf(leaf):
                    break

        # B. Left Lateral Flank (Radiating out to the left, framing the bouquet silhouette)
        left_pitches = [0.35, 0.18, 0.0, -0.18, -0.32]
        for pitch in left_pitches:
            for attempt in range(50):
                attach = np.array([
                    -self.rng.uniform(0.24, 0.40),
                    self.rng.uniform(-0.10, 0.30),
                    self.rng.uniform(-0.24, -0.10)
                ])
                dir_vec = normalize(np.array([
                    -self.rng.uniform(0.90, 1.15),
                    pitch + self.rng.uniform(-0.05, 0.05),
                    self.rng.uniform(-0.16, -0.04)
                ]))
                length = self.rng.uniform(1.10, 1.30)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=attach,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.14, 0.17),
                    droop=self.rng.uniform(0.16, 0.22),
                    curl=-0.02,
                    twist=self.rng.uniform(-0.20, 0.20),
                    arch=self.rng.uniform(0.10, 0.15),
                    wave=0.005,
                    roll=self.rng.uniform(-0.30, 0.15),
                    seed=int(self.rng.randint(20000, 39999))
                )
                if try_add_leaf(leaf):
                    break

        # C. Right Lateral Flank (Radiating out to the right, framing the bouquet silhouette)
        right_pitches = [0.35, 0.18, 0.0, -0.18, -0.32]
        for pitch in right_pitches:
            for attempt in range(50):
                attach = np.array([
                    self.rng.uniform(0.24, 0.40),
                    self.rng.uniform(-0.10, 0.30),
                    self.rng.uniform(-0.24, -0.10)
                ])
                dir_vec = normalize(np.array([
                    self.rng.uniform(0.90, 1.15),
                    pitch + self.rng.uniform(-0.05, 0.05),
                    self.rng.uniform(-0.16, -0.04)
                ]))
                length = self.rng.uniform(1.10, 1.30)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=attach,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.14, 0.17),
                    droop=self.rng.uniform(0.16, 0.22),
                    curl=0.02,
                    twist=self.rng.uniform(-0.20, 0.20),
                    arch=self.rng.uniform(0.10, 0.15),
                    wave=0.005,
                    roll=self.rng.uniform(-0.15, 0.30),
                    seed=int(self.rng.randint(40000, 59999))
                )
                if try_add_leaf(leaf):
                    break

        # D. Lower Supportive Flank Collar (Left & Right only, cradling lower flowers, leaving stems and raffia tie visible)
        for k in range(3):
            for attempt in range(50):
                attach = np.array([
                    -self.rng.uniform(0.24, 0.38),
                    self.rng.uniform(-0.46, -0.34),
                    self.rng.uniform(-0.16, -0.05)
                ])
                dir_vec = normalize(np.array([
                    -self.rng.uniform(0.85, 1.10),
                    -self.rng.uniform(0.20, 0.40),
                    self.rng.uniform(-0.12, 0.0)
                ]))
                length = self.rng.uniform(0.95, 1.15)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=attach,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.14, 0.17),
                    droop=self.rng.uniform(0.16, 0.22),
                    curl=-0.02,
                    twist=self.rng.uniform(-0.15, 0.15),
                    arch=self.rng.uniform(0.10, 0.14),
                    wave=0.005,
                    roll=-0.20,
                    seed=int(self.rng.randint(60000, 69999))
                )
                if try_add_leaf(leaf):
                    break

        for k in range(3):
            for attempt in range(50):
                attach = np.array([
                    self.rng.uniform(0.24, 0.38),
                    self.rng.uniform(-0.46, -0.34),
                    self.rng.uniform(-0.16, -0.05)
                ])
                dir_vec = normalize(np.array([
                    self.rng.uniform(0.85, 1.10),
                    -self.rng.uniform(0.20, 0.40),
                    self.rng.uniform(-0.12, 0.0)
                ]))
                length = self.rng.uniform(0.95, 1.15)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=attach,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.14, 0.17),
                    droop=self.rng.uniform(0.16, 0.22),
                    curl=0.02,
                    twist=self.rng.uniform(-0.15, 0.15),
                    arch=self.rng.uniform(0.10, 0.14),
                    wave=0.005,
                    roll=0.20,
                    seed=int(self.rng.randint(70000, 79999))
                )
                if try_add_leaf(leaf):
                    break

        # E. Cauline Accents (Behind lower flowers, peeking through gaps)
        for k in range(6):
            stem_idx = k
            for attempt in range(50):
                u_attach = self.rng.uniform(0.38, 0.58)
                stem_pt = self.stems[stem_idx].curve.evaluate(np.array([u_attach]))[0]
                stem_pt[2] = min(stem_pt[2], -0.10)
                ang_leaf = k * 137.5 * (np.pi / 180.0) + self.rng.uniform(-0.12, 0.12)
                dir_vec = normalize(np.array([
                    0.85 * np.cos(ang_leaf),
                    self.rng.uniform(0.15, 0.35),
                    -self.rng.uniform(0.08, 0.18)
                ]))
                length = self.rng.uniform(0.65, 0.85)
                dir_vec, length = _resolve_leaf_collision(attach, dir_vec, length)
                leaf = LilyLeaf(
                    attach_point=stem_pt,
                    direction=dir_vec,
                    length=length,
                    max_width=self.rng.uniform(0.13, 0.16),
                    droop=self.rng.uniform(0.12, 0.18),
                    curl=0.0,
                    twist=self.rng.uniform(-0.12, 0.12),
                    arch=self.rng.uniform(0.08, 0.12),
                    wave=0.005,
                    roll=self.rng.uniform(-0.20, 0.20),
                    seed=int(self.rng.randint(80000, 99999))
                )
                if try_add_leaf(leaf):
                    break

    def generate_bouquet_geometry(self) -> Dict[str, Any]:
        """
        Synthesizes full geometry for the entire bouquet:
        - All 7 blooming lilies
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
            s_verts, s_faces, s_normals, s_uvs = stem.generate_mesh(n_steps=32, n_radial=12)
            all_verts.append(s_verts)
            all_faces.append(s_faces + vert_offset)
            all_normals.append(s_normals)
            all_uvs.append(s_uvs)
            all_mat_ids.append(np.full(len(s_verts), 6, dtype=np.int32))  # 6 = stem green
            vert_offset += len(s_verts)
            
        # 3. Generate Leaves
        for leaf in self.leaves:
            l_verts, l_faces, l_normals, l_uvs = leaf.generate_mesh(n_s=64, n_t=26)
            all_verts.append(l_verts)
            all_faces.append(l_faces + vert_offset)
            all_normals.append(l_normals)
            all_uvs.append(l_uvs)
            all_mat_ids.append(np.full(len(l_verts), 7, dtype=np.int32))  # 7 = leaf green
            vert_offset += len(l_verts)

        # 4. Generate Artisan Raffia Hand-Tie Binding
        r_verts, r_faces, r_normals, r_uvs = generate_raffia_tie(self.stems)
        all_verts.append(r_verts)
        all_faces.append(r_faces + vert_offset)
        all_normals.append(r_normals)
        all_uvs.append(r_uvs)
        all_mat_ids.append(np.full(len(r_verts), 9, dtype=np.int32))  # 9 = artisan binding threads
        vert_offset += len(r_verts)
            
        return {
            "vertices": np.vstack(all_verts),
            "faces": np.vstack(all_faces),
            "normals": np.vstack(all_normals),
            "uvs": np.vstack(all_uvs),
            "mat_ids": np.concatenate(all_mat_ids)
        }

