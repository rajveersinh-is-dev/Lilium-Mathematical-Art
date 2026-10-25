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
            "fov_deg": 34.0,
            "focal_length": "1 / \\tan(\\theta_{\\mathrm{fov}} / 2)",
        },
        "petal_profile": {
            "alpha": 1.20,
            "beta": 0.90,
            "gamma1": 0.35,
            "gamma2": -0.10,
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
            "key_dir": [-0.52, 0.72, 0.46],
            "fill_dir": [0.60, 0.22, 0.76],
            "rim_dir": [0.10, 0.80, -0.58],
            "key_intensity": 1.60,
            "fill_intensity": 0.32,
            "rim_intensity": 0.55
        }
    }


def generate_typst_panel_markup() -> str:
    """
    Generates a dense, mathematical art panel in Typst syntax
    matching the presentation style of Hamid Naderi Yeganeh's mathematical art.
    Authored by Rajveersinh Vishal Pardeshi.
    """
    return r"""
#set page(
  width: 240mm,
  height: 96mm,
  margin: (x: 8mm, top: 3.8mm, bottom: 3.4mm),
  fill: rgb(255, 255, 255)
)

#set text(font: ("Liberation Serif", "Times New Roman", "DejaVu Serif"), size: 6.2pt, fill: rgb(0, 0, 0))
#set par(leading: 0.22em, justify: false)
#show math.equation.where(block: true): set block(spacing: 0.26em)

#align(center)[
  #text(size: 13.0pt, weight: "bold")[White Lilies by Rajveersinh Vishal Pardeshi]
  
  #v(1.0pt)
  #text(size: 7.8pt)[The above picture is a converted version of a 2400 $times$ 1440 image. For $m = 1, 2, 3, dots, 2400$ and $n = 1, 2, 3, dots, 1440$, the color of the pixel of the row $n$ and the column $m$ is]
]

#v(1.0pt)

$ "rgb" lr(( F lr(( H_0 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )), thin F lr(( H_1 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )), thin F lr(( H_2 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )) )), quad "where" thin F(x) = lr(floor.l 255 lr(( frac(1.15 x (1 + 1.15 x / L_"white"^2), 1 + 1.15 x) ))^(1 / 2.2) + xi_"CMOS" (m, n) floor.r), quad L_"white" = 1.80, "and" $

$ H_v (x, y) = sum_(k=1)^(7) lr(( product_(r=1)^(k-1) (1 - A_r (x, y)) )) A_k (x, y) L_(v, k) (x, y) + lr(( product_(r=1)^(7) (1 - A_r (x, y)) )) B_v (x, y), quad A_k (x, y) = bb(1) [ z_k (x, y) = min_(r) z_r (x, y) ], quad v in {0, 1, 2}, $

$ L_(v, k) (x, y) = bold(c)_("alb", v) (k) I_a A_"ssao" (x, y) + I_"key" lr([ bold(c)_("alb", v) (k) (d_1 S_"key" + t_1) + bold(c)_("ivory", v) b_"sub" + s_1 S_"key" ]) + I_("fill", v) + I_("rim", v), $

$ bold(S)_(k, j) (s, t) = bold(c)_k + bold(R)_k lr([ bold(p)_(k, j) (s) + t thin w_k (s) bold(B)_(k, j) (s) + c(s, t) bold(N)_(k, j) (s) ]), quad s in [0, 1], quad t in [-1, 1], quad j in {0, dots, 5}, $

$ bold(p)_(k, j) (s) = lr(( r_k (s) cos(psi_(k, j) + delta_"whorl"), thin r_k (s) sin(psi_(k, j) + delta_"whorl"), thin z_k (s) )), quad r_k (s) = (0.028 + 0.10 s) L + 0.72 L o lr([ frac(s - 0.22, 0.78) ])_+^(1.25), $

$ z_k (s) = -0.05 L lr([ 1 - frac(s, 0.22) ])_+^(1.5) + 0.44 L sin(frac(pi, 2) s^(0.88)) - a_sigma lr([ frac(s - s_sigma, 1 - s_sigma) ])_+^(p_sigma) + z_"curl" s^2 + z_"whorl", quad (a_sigma, s_sigma, p_sigma) = (0.65 rho, 0.50, 1.65) " inner", thin (1.15 rho, 0.40, 1.55) " outer", $

$ w_k (s) = C_w s^(alpha_k) (1 - s)^(beta_k) lr([ 1 + gamma_1 sin(pi s) + gamma_2 sin(2pi s + phi) ]), quad (alpha_k, beta_k) = (1.20, 0.90) " inner", thin (1.15, 1.05) " outer", quad gamma_1 = 0.35, thin gamma_2 = -0.10, $

$ c(s, t) = chi(s) lr([ -0.032 L e^(-10 s^(1.8)) (1 - t^2) - kappa_k L sin(pi s^(0.85))^(1.1) (1 - |t|^(2 q_k)) - 0.016 L sin(frac(pi (s - 0.04), 0.42)) bb(1)_(0.04 < s < 0.46) e^(-24 t^2) + rho_k L s^(1.2) t^2 (sin(2pi f s + phi) + 0.35 sin(4pi f s)) ]), $

$ chi(s) = min(1, max(0, frac(1 - s, 0.08)))^(0.75), quad (kappa_k, q_k, rho_k) = (0.046, 1, 0.024) " inner", thin (0.034, 1.2, 0.018) " outer", $

$ V(s, t) = 1 + 0.04 sum_(k=1)^(4) exp lr(( -frac((|t| - 0.22 k (1 - 0.15 s))^2, 0.002) )), quad Pi(s, t) = bb(1)_(0.12 < s < 0.38) frac(lr([ sin(42 s) cos(34 t) - 0.65 ])_+, 0.35) dot frac(0.38 - s, 0.26) lr([ 1 - frac(|t|, 0.48) ])_+, $

$ r_"theca" (theta) = R_0 (1 + 0.45 cos 2theta), quad r_"stigma" (theta) = R_s (1 + 0.44 cos 3theta), quad bold(p)_"leaf" (s) = bold(p)_"att" + s L bold(d) - frac(g, 2) s^2 bold(hat(z)) + w_"leaf" (s) t bold(B)_"twist" (s), $

$ S_"key" (bold(x)) = frac(1, 16) sum_(i=1)^(16) bb(1) lr([ z_L (bold(x)) - delta_"bias" <= Z_"SM" (bold(u)_L + r_"pcf" bold(xi)_i) ]), quad A_"ssao" (bold(x)) = max lr(( 0, 1 - frac(1, 16) sum_(i=1)^(16) bb(1) lr([ z_s > z_"sample" ]) frac(max(0, bold(n) dot bold(v)_i), 1 + d_i^2) )), $

$ C(z) = min lr(( r_max, k_"CoC" frac(|z - 4.15|, z) )), quad bold(I)_"DoF" (bold(u)) = frac(1, sum_(i=1)^(32) w_i) sum_(i=1)^(32) w_i bold(I) lr(( bold(u) + C(z) bold(phi)_i )), quad bold(phi)_i = sqrt(frac(i + 0.5, 32)) mat(cos(2.40 i); sin(2.40 i)), quad delta_"CA" = kappa_"disp" r^2. $
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

\title{\textbf{White Lilies by Rajveersinh Vishal Pardeshi}}
\author{Rajveersinh Vishal Pardeshi}
\date{}

\begin{document}
\maketitle

\section{Coordinate Domain and Camera Projection}
Let the discrete image domain be $\Omega = \{0, \dots, W-1\} \times \{0, \dots, H-1\}$ with $W=2400, H=1440$.
Normalized screen coordinates are defined by:
\begin{equation}
u = \frac{x + \frac{1}{2}}{W}, \quad v = \frac{y + \frac{1}{2}}{H}.
\end{equation}
Ray direction through pixel $(u, v)$ with field of view $\theta_{\mathrm{fov}} = 34^\circ$ is given by:
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
z_k(s)
\end{bmatrix},
\end{equation}
where
\begin{equation}
r_k(s) = (0.028 + 0.10 s) L + 0.72 L o \left[\frac{s - 0.22}{0.78}\right]_+^{1.25},
\end{equation}
\begin{equation}
\begin{aligned}
z_k(s) = &-0.05 L \left[1 - \frac{s}{0.22}\right]_+^{1.5} \\
&+ 0.44 L \sin\!\left(\frac{\pi}{2} s^{0.88}\right) \\
&- a_\sigma \left[\frac{s - s_\sigma}{1 - s_\sigma}\right]_+^{p_\sigma} + z_{\mathrm{curl}} s^2 + z_{\mathrm{whorl}},
\end{aligned}
\end{equation}
with $(a_\sigma, s_\sigma, p_\sigma) = (0.65\rho, 0.50, 1.65)$ for inner petals and $(1.15\rho, 0.40, 1.55)$ for outer
sepals, $o$ the flower opening, and $\mathbf{R}_j \in \mathrm{SO}(3)$ aligning flower facing.

\section{Canonical Width Function $w_k(s)$}
The width distribution satisfies Dirichlet boundary conditions $w(0) = w(1) = 0$:
\begin{equation}
w_k(s) = C_w s^{\alpha_k} (1 - s)^{\beta_k} \left[ 1 + \gamma_1 \sin(\pi s) + \gamma_2 \sin(2\pi s + \phi) \right],
\end{equation}
with parameters $\alpha_k = 1.20, \beta_k = 0.90, \gamma_1 = 0.35, \gamma_2 = -0.10$ for inner petals, $\alpha_k = 1.15, \beta_k = 1.05$ for outer sepals, and a per-flower phase $\phi$.

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
\begin{aligned}
c(s, t) = \chi(s)\Big[ &-0.032 L\, e^{-10 s^{1.8}}(1 - t^2) \\
&- \kappa_k L \sin(\pi s^{0.85})^{1.1}\left(1 - |t|^{2 q_k}\right) \\
&- 0.016 L \sin\!\left(\tfrac{\pi(s - 0.04)}{0.42}\right)\mathbb{I}_{s \in (0.04, 0.46)}\, e^{-24 t^2} \\
&+ \rho_k L s^{1.2} t^2\left(\sin(2\pi f s + \phi) + 0.35\sin(4\pi f s)\right)\Big],
\end{aligned}
\end{equation}
with $\chi(s) = \min\!\left(1, \max\!\left(0, \frac{1 - s}{0.08}\right)\right)^{0.75}$ and
$(\kappa_k, q_k, \rho_k) = (0.046, 1, 0.024)$ for inner petals, $(0.034, 1.2, 0.018)$ for outer sepals.

\section{Vein and Papillae Fields}
Branching longitudinal veins with ridge centres $\tau_k(s) = 0.22 k (1 - 0.15s)$ and raised carmine papillae evaluate:
\begin{equation}
V(s, t) = 1 + 0.04 \sum_{k=1}^4 \exp\left(-\frac{(|t| - \tau_k(s))^2}{0.002}\right),
\end{equation}
\begin{equation}
\Pi(s, t) = \mathbb{I}_{(0.12, 0.38)}(s)\,\tfrac{\left[\sin(42 s)\cos(34 t) - 0.65\right]_+}{0.35}\cdot\tfrac{0.38 - s}{0.26}\left[1 - \tfrac{|t|}{0.48}\right]_+.
\end{equation}

\section{Androecium and Gynoecium}
Versatile anthers are curved 2-lobed thecae with central dehiscence sulcus:
\begin{equation}
r_{\mathrm{theca}}(\theta) = R_0 (1 + 0.45 \cos 2\theta), \quad \mathbf{C}_{\mathrm{pollen}} = [0.43, 0.17, 0.03]^T.
\end{equation}
The style apex terminates in a 3-lobed moist glistening stigma:
\begin{equation}
r_{\mathrm{stigma}}(\theta) = R_s (1 + 0.44 \cos 3\theta).
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

\section{Macro Optics, Thin-Lens Bokeh and Photographic Response}
Depth of field simulates a physical macro thin-lens aperture focused at $z_{\mathrm{focus}} = 4.15$ with Circle of Confusion $C(z)$ and 32-tap golden-angle bokeh sampling:
\begin{equation}
C(z) = \min\left( r_{\max}, k_{\mathrm{CoC}} \frac{|z - z_{\mathrm{focus}}|}{z} \right),
\end{equation}
\begin{equation}
\mathbf{I}_{\mathrm{DoF}}(\mathbf{u}) = \frac{\sum_{i=1}^{32} w_i \mathbf{I}(\mathbf{u} + C(z) \boldsymbol{\phi}_i)}{\sum_{i=1}^{32} w_i}, \quad \boldsymbol{\phi}_i = \sqrt{\frac{i + 0.5}{32}}\begin{pmatrix}\cos(2.40 i)\\ \sin(2.40 i)\end{pmatrix}.
\end{equation}
Silhouettes are resolved via subpixel FXAA filtering. Final sensor response integrates $\cos^4$ vignetting, ACES filmic shoulder rolloff with white point $L_{\mathrm{white}} = 1.80$, and CMOS sensor micro-grain:
\begin{equation}
\mathbf{F}_\theta(u, v) = \left( \frac{1.15 I(1 + 1.15 I / L_{\mathrm{white}}^2)}{1 + 1.15 I} \right)^{1 / 2.2} + \boldsymbol{\xi}_{\mathrm{CMOS}},
\end{equation}

\end{document}
"""
