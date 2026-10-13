"""
geometry.py - Fundamental computational geometry and parametric spline mathematics.

Provides:
- 3D vector and matrix operations (Rodrigues rotation, frames).
- Cubic Bézier 3D curves and Bishop parallel transport frames.
- Parametric lily petal width, centerline curvature, and cross-sectional profile.
- Parametric surface grid evaluation, analytical tangents, and surface normals.
"""

from typing import Tuple, Optional
import numpy as np


def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors along a specified axis with numerical epsilon protection."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


def rodrigues_rotation(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate 3D vectors around an arbitrary unit axis by an angle (in radians)
    using Rodrigues' rotation formula:
        v_rot = v * cos(θ) + (axis × v) * sin(θ) + axis * (axis · v) * (1 - cos(θ))
    """
    axis_u = normalize(axis.reshape(3))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Dot product along last dimension
    dot = np.sum(vectors * axis_u, axis=-1, keepdims=True)
    cross = np.cross(axis_u, vectors)
    
    return vectors * cos_a + cross * sin_a + axis_u * dot * (1.0 - cos_a)


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Compute the 3x3 rotation matrix that aligns vec_from with vec_to."""
    a = normalize(vec_from.reshape(3))
    b = normalize(vec_to.reshape(3))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    
    if c < -0.999999:
        # 180 degree rotation: pick an orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, ortho))
        return rodrigues_rotation(np.eye(3), axis, np.pi)
    
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
        
    kmat = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    return np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s * s))


def coordinate_frame_matrix(facing: np.ndarray, up: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """
    Constructs a 3x3 orthonormal rotation matrix mapping local canonical coordinates
    to a world-space frame oriented along 'facing', aligned with 'up', and rotated by 'roll' (radians).
    """
    z_axis = normalize(np.asarray(facing, dtype=np.float64).reshape(3))
    up_ref = normalize(np.asarray(up, dtype=np.float64).reshape(3))
    
    x_axis = np.cross(up_ref, z_axis)
    if np.linalg.norm(x_axis) < 1e-5:
        up_alt = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up_alt, z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    if abs(roll) > 1e-6:
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)
        x_rot = x_axis * cos_r + y_axis * sin_r
        y_rot = -x_axis * sin_r + y_axis * cos_r
        x_axis = normalize(x_rot)
        y_axis = normalize(y_rot)
        
    return np.stack([x_axis, y_axis, z_axis], axis=1)


class BezierCurve3D:
    """
    Cubic Bézier curve in R^3 defined by 4 control points P_0, P_1, P_2, P_3:
        B(u) = (1-u)^3 P_0 + 3(1-u)^2 u P_1 + 3(1-u) u^2 P_2 + u^3 P_3,  u in [0, 1]
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
        self.p0 = np.asarray(p0, dtype=np.float64).reshape(3)
        self.p1 = np.asarray(p1, dtype=np.float64).reshape(3)
        self.p2 = np.asarray(p2, dtype=np.float64).reshape(3)
        self.p3 = np.asarray(p3, dtype=np.float64).reshape(3)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Evaluate points along the curve for parameters u in [0, 1]. Returns shape (N, 3)."""
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        w0 = (1.0 - u) ** 3
        w1 = 3.0 * (1.0 - u) ** 2 * u
        w2 = 3.0 * (1.0 - u) * (u ** 2)
        w3 = u ** 3
        return w0 * self.p0 + w1 * self.p1 + w2 * self.p2 + w3 * self.p3

    def tangent(self, u: np.ndarray) -> np.ndarray:
        """First derivative B'(u) representing the tangent vector. Returns shape (N, 3)."""
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        d0 = 3.0 * (1.0 - u) ** 2 * (self.p1 - self.p0)
        d1 = 6.0 * (1.0 - u) * u * (self.p2 - self.p1)
        d2 = 3.0 * (u ** 2) * (self.p3 - self.p2)
        tangents = d0 + d1 + d2
        return normalize(tangents, axis=-1)

    def compute_frames(self, u: np.ndarray, initial_up: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute an orthonormal moving frame (T, N, B) along the curve using
        Bishop's parallel transport to eliminate spurious Frenet-Serret torsion.
        """
        T = self.tangent(u)
        n_pts = len(T)
        N = np.zeros_like(T)
        B = np.zeros_like(T)
        
        if initial_up is None:
            initial_up = np.array([0.0, 0.0, 1.0])
        else:
            initial_up = normalize(np.asarray(initial_up, dtype=np.float64).reshape(3))
            
        # Initial normal
        t0 = T[0]
        n0 = initial_up - np.dot(initial_up, t0) * t0
        if np.linalg.norm(n0) < 1e-6:
            alt_up = np.array([0.0, 1.0, 0.0])
            n0 = alt_up - np.dot(alt_up, t0) * t0
        n0 = normalize(n0)
        N[0] = n0
        B[0] = normalize(np.cross(t0, n0))
        
        # Parallel transport for remaining points
        for i in range(1, n_pts):
            t_prev = T[i - 1]
            t_curr = T[i]
            v = np.cross(t_prev, t_curr)
            cos_theta = float(np.dot(t_prev, t_curr))
            if np.linalg.norm(v) > 1e-10 and cos_theta < 0.999999:
                rot_axis = normalize(v)
                rot_ang = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                N[i] = rodrigues_rotation(N[i - 1], rot_axis, rot_ang)
            else:
                N[i] = N[i - 1]
            B[i] = normalize(np.cross(t_curr, N[i]))
            
        return T, N, B


def petal_width_profile(
    s: np.ndarray,
    scale: float = 1.0,
    alpha: float = 1.25,
    beta: float = 0.92,
    gamma1: float = 0.35,
    gamma2: float = -0.12,
    phi: float = 0.25
) -> np.ndarray:
    """
    Parametric width function w(s) for a lily petal/tepal:
        w(s) = C * s^α * (1 - s)^β * [1 + γ_1 sin(π s) + γ_2 sin(2π s + φ)]
    
    Properties:
    - w(0) = 0: slender unguis attachment at receptacle base.
    - w(1) = 0: tapered elegant pointed reflexed tip.
    - Maximum width occurs in the midsection (s ≈ 0.52 - 0.62).
    - Guaranteed non-negative across s in [0, 1].
    """
    s_clamped = np.clip(s, 0.0, 1.0)
    base_term = (s_clamped ** alpha) * ((1.0 - s_clamped) ** beta)
    harmonic_term = 1.0 + gamma1 * np.sin(np.pi * s_clamped) + gamma2 * np.sin(2.0 * np.pi * s_clamped + phi)
    harmonic_term = np.maximum(0.0, harmonic_term)
    
    s_peak = alpha / (alpha + beta)
    peak_val = (s_peak ** alpha) * ((1.0 - s_peak) ** beta) * (1.0 + gamma1 * np.sin(np.pi * s_peak))
    norm_c = scale / max(peak_val, 1e-6)
    return norm_c * base_term * harmonic_term


def petal_cross_section_curvature(
    s: np.ndarray,
    t: np.ndarray,
    groove_depth: float = 0.035,
    arch_height: float = 0.045,
    margin_undulation: float = 0.022,
    undulation_freq: float = 3.5,
    margin_phase: float = 0.0
) -> np.ndarray:
    """
    Evaluates the vertical surface elevation c(s, t) across the petal transverse cross-section.
    Provides natural cup curvature and ruffled margins without artificial creases.
    """
    t_clamped = np.clip(t, -1.0, 1.0)
    t2 = t_clamped ** 2
    # Smooth localized throat nectar furrow (near base s < 0.4)
    groove = -groove_depth * np.exp(-10.0 * (s ** 1.6)) * np.exp(-28.0 * t2)
    # Gentle transverse cup across the petal blade
    arch = -arch_height * (0.2 + 0.8 * s) * (t2 - 0.4 * (t2 ** 2))
    # Delicate margin wave
    ruffle = margin_undulation * (s ** 1.4) * t2 * np.sin(2.0 * np.pi * undulation_freq * s + margin_phase)
    return groove + arch + ruffle


def compute_mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute smooth area-weighted vertex normals for a triangular mesh.
    - vertices: shape (V, 3)
    - faces: shape (F, 3)
    Returns: normals shape (V, 3)
    """
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Face normals with magnitude proportional to face area
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate onto vertices
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
        
    return normalize(normals, axis=-1)
