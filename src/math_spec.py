"""
math_spec.py - Formal mathematical model specification and typesetting generators.

Generates:
- Complete mathematical specification in both Typst and LaTeX formats.
- Mathematically honest derivations matching the actual geometry, lighting,
  materials, and rendering algorithms implemented in Project Lilium.
"""

from typing import Dict, Any


def get_mathematical_model_data() -> Dict[str, Any]:
    """Returns structured parameter data directly from the implemented mathematical model."""
    return {
        "domain": {
            "width": 2400,
            "height": 1440,
            "fov_deg": 36.0,
            "focal_length": "1 / \\tan(\\theta_{\\mathrm{fov}} / 2)",
        },
        "petal_profile": {
            "alpha": 1.20,
            "beta": 0.88,
            "gamma1": 0.35,
            "gamma2": -0.12,
            "w_inner": 0.44,
            "w_outer": 0.32
        },
        "perianth": {
            "whorls": 2,
            "inner_petals": 3,
            "outer_sepals": 3,
            "inner_angles": "2\\pi k / 3",
            "outer_angles": "\\pi / 3 + 2\\pi k / 3",
            "stamens": 6,
            "pistil": 1
        },
        "lighting": {
            "key_dir": [-0.42, 0.65, 0.63],
            "fill_dir": [0.55, 0.38, 0.74],
            "rim_dir": [0.20, 0.72, -0.66],
            "key_intensity": 1.25,
            "fill_intensity": 0.55,
            "rim_intensity": 0.45
        }
    }


def generate_typst_panel_markup() -> str:
    """
    Generates a dense, professionally typeset mathematical panel in Typst syntax.
    Organized into 3 rigorous, elegant columns with New Computer Modern Math glyphs.
    """
    return r"""
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
"""


def generate_latex_source() -> str:
    """
    Generates full formal LaTeX document for lilium_model.tex and equation_panel.tex.
    """
    return r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.6in,top=0.5in,bottom=0.5in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{mathpazo}
\usepackage{microtype}
\usepackage{titlesec}

\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{6pt}{2pt}

\title{\textbf{Analytical Foundations of the Lilium Mathematical Synthesis Engine}}
\author{Procedural Botanical Model Specification $\cdot$ Project Lilium}
\date{}

\begin{document}
\maketitle

\section{Coordinate Domain and Camera Projection}
Let the discrete image domain be $\Omega = \{0, \dots, W-1\} \times \{0, \dots, H-1\}$ with $W=2400, H=1440$.
Normalized screen coordinates are defined by:
\begin{equation}
u = \frac{x + \frac{1}{2}}{W}, \quad v = \frac{y + \frac{1}{2}}{H}.
\end{equation}
Ray direction through pixel $(u, v)$ with field of view $\theta_{\mathrm{fov}} = 36^\circ$ is given by:
\begin{equation}
\mathbf{r}(u, v) = \frac{\mathbf{d}_0 + (2u - 1)\tan\frac{\theta}{2}\mathbf{u}_r + (1 - 2v)\tan\frac{\theta}{2}\mathbf{u}_{up}}{\|\mathbf{d}_0 + (2u - 1)\tan\frac{\theta}{2}\mathbf{u}_r + (1 - 2v)\tan\frac{\theta}{2}\mathbf{u}_{up}\|}.
\end{equation}

\section{Parametric Tepal Manifold $\mathbf{S}_{j,k}$}
Each tepal $k \in \{0, \dots, 5\}$ of flower $j$ is parametrized over $(s, t) \in [0, 1] \times [-1, 1]$:
\begin{equation}
\mathbf{S}_{j,k}(s, t) = \mathbf{p}_{j,k}(s) + t\,w_k(s)\,\mathbf{B}_{j,k}(s) + c_k(s, t)\,\mathbf{N}_{j,k}(s),
\end{equation}
with centerline spine trajectory $\mathbf{p}_{j,k}(s)$:
\begin{equation}
\mathbf{p}_{j,k}(s) = \mathbf{c}_j + \mathbf{R}_j \begin{bmatrix}
r_k(s) \cos\psi_{j,k} \\
r_k(s) \sin\psi_{j,k} \\
0.32 L s - h_{\mathrm{refl}}\max(0, s - 0.38)^{1.85} + z_{\mathrm{curl}} s^2
\end{bmatrix},
\end{equation}
where $r_k(s) = L_k (0.06 s + 0.78 s^{1.45})$ and $\mathbf{R}_j \in \mathrm{SO}(3)$ aligns flower facing.

\section{Canonical Width Function $w(s)$}
The width distribution satisfies Dirichlet boundary conditions $w(0) = w(1) = 0$:
\begin{equation}
w(s) = C_w s^\alpha (1 - s)^\beta \left[ 1 + \gamma_1 \sin(\pi s) + \gamma_2 \sin(2\pi s + \phi) \right],
\end{equation}
with parameters $\alpha = 1.20, \beta = 0.88, \gamma_1 = 0.35, \gamma_2 = -0.12$.

\section{Trimerous Perianth Configuration}
The 6 tepals form two alternating concentric whorls:
\begin{equation}
\psi_{j,k} = \begin{cases}
\psi_{j,0} + \frac{2\pi k}{3} + \delta_{j,k}, & k \in \{0, 1, 2\} \text{ (Inner Petals)}, \\
\psi_{j,0} + \frac{\pi}{3} + \frac{2\pi(k-3)}{3} + \delta_{j,k}, & k \in \{3, 4, 5\} \text{ (Outer Sepals)}.
\end{cases}
\end{equation}

\section{Transverse Curvature and Margins}
The elevation function $c(s, t)$ governs the central groove and undulating ruffles:
\begin{equation}
c(s, t) = -\delta_{\mathrm{groove}} e^{-26 t^2}(1 - 0.45s) + \delta_{\mathrm{arch}} s(1 - t^2) + A_{\mathrm{ruf}} s^{1.4} t^2 \sin(2\pi\omega s + \phi).
\end{equation}

\section{Vein and Papillae Fields}
Branching longitudinal veins and raised carmine papillae evaluate:
\begin{equation}
V(s, t) = 1 + \sum_{k=1}^4 a_k \exp\left(-\frac{(|t| - \tau_k(s))^2}{2\sigma_k^2}\right), \quad \tau_k(s) = 0.22 k (1 - 0.15s),
\end{equation}
\begin{equation}
\mathbf{n}_{\mathrm{pert}} = \mathbf{n} + (\delta t\,\hat{\mathbf{b}} + \delta s\,\hat{\mathbf{t}}) \cdot 0.75\,h_{\mathrm{pap}}(s, t).
\end{equation}

\section{Androecium and Gynoecium}
Versatile anthers are curved 2-lobed thecae with central dehiscence sulcus:
\begin{equation}
r_{\mathrm{theca}}(\theta) = R_0 (1 - 0.35 \cos 2\theta), \quad \mathbf{C}_{\mathrm{pollen}} = [0.94, 0.50, 0.04]^T.
\end{equation}
The style apex terminates in a 3-lobed moist glistening stigma:
\begin{equation}
r_{\mathrm{stigma}}(\theta) = R_s (1 + 0.42 \cos 3\theta).
\end{equation}

\section{Foliage and Bézier Stems}
Stems are cubic Bézier spatial curves $\mathbf{B}(u) = \sum_{i=0}^3 \binom{3}{i}(1-u)^{3-i}u^i \mathbf{P}_i$.
Lanceolate leaves incorporate axial torsion $\mathbf{B}_{\mathrm{twist}}(s)$:
\begin{equation}
\mathbf{p}_{\mathrm{leaf}}(s) = \mathbf{p}_{\mathrm{att}} + s L \mathbf{d} - \frac{g}{2} s^2 \hat{\mathbf{z}} + w_{\mathrm{leaf}}(s) t \mathbf{B}_{\mathrm{twist}}(s).
\end{equation}

\section{Shadow-Mapped PCF and Dual-Lobe GGX BRDF}
Key illumination is modulated by an orthographic shadow map $Z_{\mathrm{SM}}$ with 16-tap Poisson disk filtering:
\begin{equation}
S_{\mathrm{key}}(\mathbf{x}) = \frac{1}{16} \sum_{i=1}^{16} \mathbb{I}\left( z_L(\mathbf{x}) - \delta_{\mathrm{bias}} \le Z_{\mathrm{SM}}(\mathbf{u}_L + r_{\mathrm{pcf}} \boldsymbol{\xi}_i) \right).
\end{equation}
Surface radiance combines dual-lobe Cook-Torrance microfacets ($\alpha_1 = 0.42, \alpha_2 = 0.12$), Schlick Fresnel $F(\theta)$, and forward subsurface transmission:
\begin{equation}
I(\mathbf{x}) = \mathbf{c}_{\mathrm{alb}} I_a A_{\mathrm{ssao}} + I_{\mathrm{key}} \left[ \mathbf{c}_{\mathrm{alb}}(d_1 S_{\mathrm{key}} + t_1) + \mathbf{c}_{\mathrm{ivory}} b_{\mathrm{sub}} + s_1 S_{\mathrm{key}} \right] + \mathbf{I}_{\mathrm{fill}} + \mathbf{I}_{\mathrm{rim}}.
\end{equation}

\section{Discrete Projection and Tone Mapping}
Pixel color emerges from Z-buffer minimization and ACES-Reinhard filmic tone mapping:
\begin{equation}
z^*(u, v) = \min_m \{ z : \mathbf{r}(u, v) \cap \mathcal{M}_m \ne \emptyset \},
\end{equation}
\begin{equation}
\mathbf{F}_\theta(u, v) = \left( \frac{1.30 I(1 + 1.30 I / L_{\mathrm{white}}^2)}{1 + 1.30 I} \right)^{1 / 2.2}.
\end{equation}

\end{document}
"""
