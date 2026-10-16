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
  margin: (x: 8mm, top: 4.5mm, bottom: 4.0mm),
  fill: rgb(255, 255, 255)
)

#set text(font: ("Liberation Serif", "Times New Roman", "DejaVu Serif"), size: 6.8pt, fill: rgb(0, 0, 0))
#set par(leading: 0.40em, justify: false)
#show math.equation.where(block: true): set block(spacing: 0.48em)

#align(center)[
  #text(size: 13.0pt, weight: "bold")[White Lilies by Rajveersinh Vishal Pardeshi]
  
  #v(1.0pt)
  #text(size: 7.8pt)[The above picture is a converted version of a 2400 $times$ 1440 image. For $m = 1, 2, 3, dots, 2400$ and $n = 1, 2, 3, dots, 1440$, the color of the pixel of the row $n$ and the column $m$ is]
]

#v(1.0pt)

$ "rgb" lr(( F lr(( H_0 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )), thin F lr(( H_1 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )), thin F lr(( H_2 lr(( frac(m - 1200, 720), frac(721 - n, 720) )) )) )), quad "where" thin F(x) = lr(floor.l 255 lr(( frac(1.25 x (1 + 1.25 x / L_w^2), 1 + 1.25 x) ))^(1 / 2.2) + xi_"CMOS" (m, n) floor.r), "and" $

$ H_v (x, y) = sum_(k=1)^(8) lr(( product_(r=1)^(k-1) (1 - A_r (x, y)) )) lr([ U_k (x, y) L_(v, k) (x, y) + (1 - U_k (x, y)) B_v (x, y) ]), quad v in {0, 1, 2}, $

$ L_(v, k) (x, y) = bold(c)_("alb", v) (k) I_a A_"ssao" (x, y) + I_"key" lr([ bold(c)_("alb", v) (k) (d_1 S_"key" + t_1) + bold(c)_("ivory", v) b_"sub" + s_1 S_"key" ]) + I_("fill", v) + I_("rim", v), $

$ bold(S)_(k, j) (s, t) = bold(c)_k + bold(R)_k lr([ bold(p)_(k, j) (s) + t thin w(s) bold(B)_(k, j) (s) + c(s, t) bold(N)_(k, j) (s) ]), quad s in [0, 1], quad t in [-1, 1], quad j in {0, dots, 5}, $

$ bold(p)_(k, j) (s) = lr(( r_k (s) cos(psi_(k, j) + delta_"whorl"), thin r_k (s) sin(psi_(k, j) + delta_"whorl"), thin 0.32 L s - h_"refl" max(0, s - 0.38)^(1.85) + z_"curl" s^2 )), $

$ w(s) = C_w s^(1.20) (1 - s)^(0.88) lr([ 1 + 0.35 sin(pi s) - 0.12 sin(2pi s) ]), quad c(s, t) = -delta_"groove" e^(-26 t^2) (1 - 0.45 s) + delta_"arch" s (1 - t^2) + A_"ruf" s^(1.4) t^2 sin(2pi omega s), $

$ V(s, t) = 1 + sum_(i=1)^4 a_i exp lr(( -frac((|t| - 0.19 i (1 - 0.12 s))^2, 2 sigma_i^2) )), quad Pi(s, t) = sum_(p=1)^(N_p) h_p exp lr(( -k_p lr([ (s - s_p)^2 + (t - t_p)^2 ]) )), $

$ r_"theca" (theta) = R_0 (1 - 0.35 cos 2theta) e^(-32 t^2), quad r_"stigma" (theta) = R_s (1 + 0.42 cos 3theta), quad bold(p)_"leaf" (s) = bold(p)_"att" + s L bold(d) - frac(g, 2) s^2 bold(hat(z)) + w_"leaf" (s) t bold(B)_"twist" (s), $

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
r_{\mathrm{theca}}(\theta) = R_0 (1 - 0.35 \cos 2\theta), \quad \mathbf{C}_{\mathrm{pollen}} = [0.43, 0.17, 0.03]^T.
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

\section{Macro Optics, Thin-Lens Bokeh and Photographic Response}
Depth of field simulates a physical macro thin-lens aperture focused at $z_{\mathrm{focus}} = 4.15$ with Circle of Confusion $C(z)$ and 28-tap Fermat spiral bokeh sampling:
\begin{equation}
C(z) = \min\left( r_{\max}, k_{\mathrm{CoC}} \frac{|z - z_{\mathrm{focus}}|}{z} \right), \quad \mathbf{I}_{\mathrm{DoF}}(\mathbf{u}) = \frac{\sum_{i=1}^{28} w_i \mathbf{I}(\mathbf{u} + C(z) \boldsymbol{\phi}_i)}{\sum_{i=1}^{28} w_i}.
\end{equation}
Silhouettes are resolved via subpixel FXAA filtering. Final sensor response integrates $\cos^4$ vignetting, ACES filmic shoulder rolloff, and CMOS sensor micro-grain:
\begin{equation}
\mathbf{F}_\theta(u, v) = \left( \frac{1.25 I(1 + 1.25 I / L_{\mathrm{white}}^2)}{1 + 1.25 I} \right)^{1 / 2.2} + \boldsymbol{\xi}_{\mathrm{CMOS}}.
\end{equation}

\end{document}
"""
