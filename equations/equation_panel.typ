
#set page(
  width: 2400pt,
  height: 880pt,
  margin: (x: 48pt, top: 32pt, bottom: 32pt),
  fill: rgb("#ffffff")
)

#set text(
  font: "New Computer Modern",
  size: 15.5pt,
  fill: rgb("#111111")
)

#set par(justify: true, leading: 0.68em)

#align(center)[
  #text(size: 21pt, weight: "bold", tracking: 0.05em)[ANALYTICAL FOUNDATIONS OF THE LILIUM SYNTHESIS ENGINE]
  #v(4pt)
  #text(size: 12.5pt, fill: rgb("#444444"), style: "italic")[
    Parametric Perianth Manifolds, Trimerous Whorls, Translucent Subsurface BRDF, and Discrete Projective Projection
  ]
]

#v(12pt)

#columns(3, gutter: 36pt)[
  == 1. Coordinate Domain & Manifold Mapping
  Let the continuous botanical render space occupy the normalized coordinate chart:
  $ Omega = [0, 1]^2, quad u = frac(x + 1/2, W), quad v = frac(y + 1/2, H) $
  The continuous camera ray through normalized coordinates $(u, v)$ with field of view $theta_"fov" = 36degree$ is defined by:
  $ bold(r)(u, v; xi) = bold(c)_0 + lambda frac(bold(d)_0 + (2u - 1)tan(theta_"fov"/2) bold(u)_"r" + (1 - 2v)tan(theta_"fov"/2) bold(u)_"up", || bold(d)_0 + (2u - 1)tan(theta_"fov"/2) bold(u)_"r" + (1 - 2v)tan(theta_"fov"/2) bold(u)_"up" ||) $

  == 2. Parametric Tepal Manifold $bold(S)_(j,k)$
  Each of the 6 tepals of blossom $j in {1, ..., 10}$ is an embedded 2-manifold in $RR^3$ parametrized over $(s, t) in [0, 1] times [-1, 1]$:
  $ bold(S)_(j,k)(s, t) = bold(p)_(j,k)(s) + t w_k(s) bold(B)_(j,k)(s) + c_k(s, t) bold(N)_(j,k)(s) $
  where the 3D perianth spine trajectory $bold(p)_(j,k)(s)$ evaluates:
  $ bold(p)_(j,k)(s) = bold(c)_j + bold(R)_j mat(
    r_k(s) cos(psi_(j,k));
    r_k(s) sin(psi_(j,k));
    0.32 L s - h_"refl" max(0, s - 0.38)^(1.85) + z_"curl" s^2
  ) $
  with radial extension $r_k(s) = L_k (0.06 s + 0.78 s^(1.45))$.

  == 3. Canonical Width Function $w(s)$
  The transverse width distribution satisfies Dirichlet boundary conditions $w(0) = w(1) = 0$ via asymmetric beta-harmonic modulation:
  $ w(s) = C_w s^alpha (1 - s)^beta lr([ 1 + gamma_1 sin(pi s) + gamma_2 sin(2pi s + phi) ]) $
  where $alpha = 1.20, beta = 0.88, gamma_1 = 0.35, gamma_2 = -0.12$. The maximum perianth expansion occurs at:
  $ s^* = frac(alpha, alpha + beta) + epsilon_gamma approx 0.577 $

  #colbreak()

  == 4. Trimerous Perianth Configuration
  The 6 tepals form two alternating concentric trimerous whorls:
  $ psi_(j,k) = cases(
    psi_(j,0) + frac(2pi k, 3) + delta_(j,k) & quad "Inner Whorl (Petals), " k in {0, 1, 2}, \
    psi_(j,0) + frac(pi, 3) + frac(2pi(k-3), 3) + delta_(j,k) & quad "Outer Whorl (Sepals), " k in {3, 4, 5}
  ) $
  Inner tepals possess wider latitude $W_"inner" = 0.44 L$, while outer sepals form an exterior protective sheath with $W_"outer" = 0.32 L$.

  == 5. Transverse Curvature & Undulation
  The elevation function $c(s, t)$ models the nectar furrow, lateral arching, and crenulated margin ruffles:
  $ c(s, t) = -delta_"groove" e^(-26 t^2) (1 - 0.45s) + delta_"arch" s (1 - t^2) + A_"ruf" s^(1.4) t^2 sin(2pi omega s + phi) $

  == 6. Longitudinal Vein & Papillae Fields
  The multi-scale vein intensity field $V(s, t)$ combines primary midrib furrow and $K=4$ branching Gaussian pathways:
  $ V(s, t) = 1 + sum_(k=1)^K a_k exp lr( -frac((|t| - tau_k(s))^2, 2sigma_k^2) ) $
  with vein trajectories $tau_k(s) = 0.22 k (1 - 0.15s)$. Papillae nectar glands near throat evaluate:
  $ Pi(s, t) = bb(1)_(s in [0.12, 0.38]) dot sin^+(42s) cos^+(34t) dot (0.38 - s) $

  == 7. Reproductive Organs (Androecium & Gynoecium)
  The 6 versatile anthers are prolate spheroids centered at filament apices:
  $ cal(E)_k = { bold(x) in RR^3 : ((bold(x) - bold(q)_k) dot bold(e)_(1,k))^2 / a^2 + ((bold(x) - bold(q)_k) dot bold(e)_(2,k))^2 / b^2 + ((bold(x) - bold(q)_k) dot bold(e)_(3,k))^2 / c^2 <= 1 } $
  with pollen radiance $bold(C)_"pollen" = [0.88, 0.48, 0.06]^T$ and micro-granular scattering.

  #colbreak()

  == 8. Foliage & Bézier Stem Geometry
  Stem axes are $C^2$ cubic Bézier spatial trajectories:
  $ bold(B)(u) = (1-u)^3 bold(P)_0 + 3(1-u)^2 u bold(P)_1 + 3(1-u) u^2 bold(P)_2 + u^3 bold(P)_3 $
  Leaves follow lanceolate profiles with gravitational deflection:
  $ bold(p)_"leaf"(s) = bold(p)_"att" + s L_"leaf" bold(d) - frac(g, 2) s^2 bold(u)_z + w_"leaf"(s) t bold(B)_"leaf" $

  == 9. PBR Reflectance & Subsurface Transmission
  Surface radiance $I(bold(x))$ balances diffuse reflection, waxy cuticle sheen, and forward subsurface transmission:
  $ I(bold(x)) = k_a I_a bold(c)_"base" + sum_(ell=1)^3 I_ell lr([
    bold(c)_"alb" max(0, bold(n) dot bold(omega)_ell) +
    k_"spec" (bold(n) dot bold(h)_ell)^(alpha_"spec") +
    k_"sub" bold(c)_"trans" max(0, -bold(n) dot bold(omega)_ell)
  ]) + bold(I)_"rim" $
  where half-vector $bold(h)_ell = frac(bold(omega)_ell + bold(omega)_v, ||bold(omega)_ell + bold(omega)_v||)$.

  == 10. Depth Projection & Filmic Tone Operator
  The discrete pixel color $bold(F)_theta(u, v)$ emerges from Z-buffer depth minimization and ACES-Reinhard filmic tone mapping:
  $ z^*(u, v) = min_(m) { z : bold(r)(u, v) inter cal(M)_m != nothing }, quad bold(n)^*(u, v) = bold(n)(z^*) $
  $ bold(F)_theta(u, v) = cal(T) lr([ I(bold(x)^*, bold(n)^*, bold(u)_v) ]) = lr( frac(1.35 I (1 + 1.35 I / L_"white"^2), 1 + 1.35 I) )^(1 / 2.2) $
  yielding reproducible high-fidelity botanical art across all $Omega$.
]
