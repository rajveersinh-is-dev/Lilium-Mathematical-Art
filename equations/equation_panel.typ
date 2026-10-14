
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
  Multi-scale fibrovascular pathways and 3D carmine nectar papillae evaluate:
  $ V(s, t) = 1 + sum_(k=1)^4 a_k exp lr( -frac((|t| - tau_k(s))^2, 2sigma_k^2) ), quad tau_k(s) = 0.22 k (1 - 0.15s) $
  Nectariferous papillae nodules $Pi(s, t)$ perturb surface micro-normals via hash cells:
  $ bold(n)_"pert" = bold(n) + lr([ delta t thin bold(B) + delta s thin bold(T) ]) dot 0.75 h_"pap" (s, t) $
  with rich carmine albedo $bold(C)_"pap" = [0.78, 0.12, 0.20]^T$ and basal pink blush $B(s, t) = 0.14 exp(-22(s-0.25)^2 - 14 t^2)$.

  == 7. Reproductive Organs (Androecium & Gynoecium)
  The 6 versatile anthers are curved 2-lobed thecae with central dehiscence sulcus:
  $ r_"theca"(theta) = R_0 (1 - 0.35 cos 2theta), quad bold(C)_"pollen" = [0.94, 0.50, 0.04]^T $
  The central gynoecium style terminates into a 3-lobed moist glistening stigma pad:
  $ r_"stigma"(theta) = R_s (1 + 0.42 cos 3theta) $

  #colbreak()

  == 8. Foliage & Bézier Stem Geometry
  Stem axes are $C^2$ cubic Bézier spatial trajectories:
  $ bold(B)(u) = (1-u)^3 bold(P)_0 + 3(1-u)^2 u bold(P)_1 + 3(1-u) u^2 bold(P)_2 + u^3 bold(P)_3 $
  Lanceolate leaves follow gravitational deflection with axial torsion:
  $ bold(p)_"leaf"(s) = bold(p)_"att" + s L_"leaf" bold(d) - frac(g, 2) s^2 bold(u)_z + w_"leaf"(s) t bold(B)_"twist"(s) $

  == 9. Directional Shadow Map & PCF Penumbra
  Key illumination is modulated by an orthographic shadow map $Z_"SM"$ with 16-tap Poisson disk percentage-closer filtering (PCF):
  $ S_"key"(bold(x)) = frac(1, 16) sum_(i=1)^16 bb(1) lr( z_L (bold(x)) - delta_"bias" <= Z_"SM"(bold(u)_L + r_"pcf" bold(xi)_i) ) $
  Crevices and petal folds are anchored by screen-space ambient occlusion (SSAO) with depth bilateral filtering $A_"ssao"(bold(x))$.

  == 10. Dual-Lobe GGX BRDF & Filmic Tone Operator
  Surface radiance balances dual-lobe Cook-Torrance microfacets ($alpha_1 = 0.42, alpha_2 = 0.12$), Schlick Fresnel $F(theta)$, and forward subsurface transmission:
  $ I(bold(x)) = bold(c)_"alb" I_a A_"ssao" + I_"key" lr([ bold(c)_"alb" (d_1 S_"key" + t_1) + bold(c)_"ivory" b_"sub" + s_1 S_"key" ]) + bold(I)_"fill" + bold(I)_"rim" $
  Discrete pixel color emerges from Z-buffer minimization and ACES-Reinhard filmic tone mapping:
  $ bold(F)_theta(u, v) = lr( frac(1.30 I (1 + 1.30 I / L_"white"^2), 1 + 1.30 I) )^(1 / 2.2) $
]
