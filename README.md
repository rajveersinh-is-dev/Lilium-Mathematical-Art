# Project Lilium — Mathematical Reconstruction of a High-Fidelity White-Lily Bouquet

An original mathematical artwork and procedural synthesis engine that reconstructs a photorealistic bouquet of white oriental lilies (*Lilium*) paired with a dense, academically rigorous mathematical panel detailing the analytical manifolds, phyllotaxis, and optical reflectance models that govern the image synthesis.

---

## Visual Overview & Composition Hierarchy

Inspired by the classical mathematical-art presentation aesthetic (such as the legendary strawberry-equation composition), **Project Lilium** structures its presentation into an exact square composition ($2400 \times 2400$ pixels):

1. **Upper Botanical Domain (Upper 60% — $2400 \times 1440$ px)**: A high-resolution, procedural 3D render of a lush bouquet of 10 white lilies and swelling buds nestled among lanceolate foliage against a deep chiaroscuro forest-charcoal field.
2. **Academic Caption Band ($2400 \times 80$ px)**: A dividing title banner:
   * **White Lilies — A Mathematical Reconstruction**
   * *Parametric Perianth Manifolds · Trimerous Whorls · PBR Subsurface Translucency · 2400 × 1440 Procedural Synthesis*
3. **Analytical Mathematical Panel (Lower 37% — $2400 \times 880$ px)**: A dense, 3-column white mathematical panel typeset using Typst and Computer Modern math glyphs, detailing the 10 interconnected equations and algorithms that generate the artwork from first principles.

---

## Deliverables Summary

| Artifact | Dimensions | File Path | Description |
| :--- | :--- | :--- | :--- |
| **Master Artwork** | $2400 \times 2400$ px | [`outputs/lilium_mathematical_art.png`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/outputs/lilium_mathematical_art.png) | Unified square composition: bouquet + caption + mathematical panel |
| **Botanical Render** | $2400 \times 1440$ px | [`outputs/lily_bouquet.png`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/outputs/lily_bouquet.png) | Standalone photorealistic 3D botanical render |
| **Mathematical Panel** | $2400 \times 880$ px | [`outputs/lily_equations.png`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/outputs/lily_equations.png) | Standalone 3-column typeset analytical panel |
| **Typst Source** | — | [`equations/equation_panel.typ`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/equations/equation_panel.typ) | Native Typst source at 300 PPI |
| **LaTeX Specification** | — | [`equations/lilium_model.tex`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/equations/lilium_model.tex) | Complete formal LaTeX article specification |
| **Scene Configuration** | — | [`configs/final_scene.json`](file:///c:/Users/davea/OneDrive/Desktop/Lilium-Mathematical-Art/configs/final_scene.json) | Deterministic scene parameterization |

---

## Analytical Mathematical Foundations

Every visual attribute in the bouquet is computed from first principles without precomputed lookup tables or static photographic textures.

### 1. Continuous Coordinate Domain & Camera Projection
Let $\Omega = \{0, \dots, W-1\} \times \{0, \dots, H-1\}$ with $W=2400, H=1440$. The continuous ray through normalized screen coordinates $(u, v) = \left(\frac{x+0.5}{W}, \frac{y+0.5}{H}\right)$ with field of view $\theta_{\mathrm{fov}} = 36^\circ$ evaluates:
$$\mathbf{r}(u, v) = \frac{\mathbf{d}_0 + (2u - 1)\tan\frac{\theta}{2}\mathbf{u}_r + (1 - 2v)\tan\frac{\theta}{2}\mathbf{u}_{up}}{\|\mathbf{d}_0 + (2u - 1)\tan\frac{\theta}{2}\mathbf{u}_r + (1 - 2v)\tan\frac{\theta}{2}\mathbf{u}_{up}\|}$$

### 2. Parametric Tepal Manifold $\mathbf{S}_{j,k}$
Each tepal $k \in \{0, \dots, 5\}$ of flower $j \in \{1, \dots, 10\}$ is an embedded 2-manifold in $\mathbb{R}^3$ parametrized over $(s, t) \in [0, 1] \times [-1, 1]$:
$$\mathbf{S}_{j,k}(s, t) = \mathbf{p}_{j,k}(s) + t\,w_k(s)\,\mathbf{B}_{j,k}(s) + c_k(s, t)\,\mathbf{N}_{j,k}(s)$$
where the 3D perianth spine trajectory evaluates:
$$\mathbf{p}_{j,k}(s) = \mathbf{c}_j + \mathbf{R}_j \begin{bmatrix} r_k(s) \cos\psi_{j,k} \\ r_k(s) \sin\psi_{j,k} \\ 0.32 L s - h_{\mathrm{refl}}\max(0, s - 0.38)^{1.85} + z_{\mathrm{curl}} s^2 \end{bmatrix}$$
with radial extension $r_k(s) = L_k (0.06 s + 0.78 s^{1.45})$ and rotation matrix $\mathbf{R}_j \in \mathrm{SO}(3)$ aligning the flower's 3D pose and organic roll angle.

### 3. Canonical Width Function $w(s)$
The transverse width profile satisfies Dirichlet boundary conditions $w(0) = w(1) = 0$ via an asymmetric beta-harmonic profile:
$$w(s) = C_w s^\alpha (1 - s)^\beta \left[ 1 + \gamma_1 \sin(\pi s) + \gamma_2 \sin(2\pi s + \phi) \right]$$
with $\alpha = 1.20, \beta = 0.88, \gamma_1 = 0.35, \gamma_2 = -0.12$. Peak expansion occurs in the midsection ($s^* \approx 0.577$), modeling the slender floral unguis base expanding into an organic elliptical blade.

### 4. Trimerous Perianth Configuration
The 6 tepals are partitioned into two alternating concentric whorls:
$$\psi_{j,k} = \begin{cases} \psi_{j,0} + \frac{2\pi k}{3} + \delta_{j,k}, & k \in \{0, 1, 2\} \text{ (Inner Petals)} \\ \psi_{j,0} + \frac{\pi}{3} + \frac{2\pi(k-3)}{3} + \delta_{j,k}, & k \in \{3, 4, 5\} \text{ (Outer Sepals)} \end{cases}$$
Inner petals have half-width $w_{\mathrm{inner}} = 0.22 L$, while outer sepals form an exterior protective sheath with $w_{\mathrm{outer}} = 0.16 L$. Outer sepals sit slightly behind inner petals ($\Delta z = -0.04 L$), creating natural layered occlusion.

### 5. Transverse Curvature & Margin Ruffles
Elevation $c(s, t)$ governs the central groove, lateral arching, and crenulated edge ruffles:
$$c(s, t) = -\delta_{\mathrm{groove}} e^{-26 t^2}(1 - 0.45s) + \delta_{\mathrm{arch}} s(1 - t^2) + A_{\mathrm{ruf}} s^{1.4} t^2 \sin(2\pi\omega s + \phi)$$

### 6. Longitudinal Vein & Papillae Fields
Parallel longitudinal veins and raised carmine nectar glands near the throat evaluate:
$$V(s, t) = 1 + \sum_{k=1}^4 a_k \exp\left(-\frac{(|t| - \tau_k(s))^2}{2\sigma_k^2}\right), \quad \tau_k(s) = 0.22 k (1 - 0.15s)$$
$$\Pi(s, t) = \mathbb{I}_{[0.12, 0.38]}(s) \sin^+(42s) \cos^+(34t) (0.38 - s)$$

### 7. Androecium and Gynoecium
Each flower contains 6 versatile anthers modeled as prolate ellipsoids:
$$\mathcal{E}_k = \left\{ \mathbf{x} \in \mathbb{R}^3 : \frac{((\mathbf{x} - \mathbf{q}_k)\cdot\mathbf{e}_{1,k})^2}{a^2} + \frac{((\mathbf{x} - \mathbf{q}_k)\cdot\mathbf{e}_{2,k})^2}{b^2} + \frac{((\mathbf{x} - \mathbf{q}_k)\cdot\mathbf{e}_{3,k})^2}{c^2} \le 1 \right\}$$
coated in pollen grains with golden-amber radiance $\mathbf{C}_{\mathrm{pollen}} = [0.88, 0.48, 0.06]^T$, plus a central pistil style with a 3-lobed pale green stigma.

### 8. Foliage & Bézier Stem Geometry
Stems are modeled as $C^2$ cubic Bézier spatial curves with Bishop parallel transport frames:
$$\mathbf{B}(u) = (1-u)^3 \mathbf{P}_0 + 3(1-u)^2 u \mathbf{P}_1 + 3(1-u) u^2 \mathbf{P}_2 + u^3 \mathbf{P}_3$$
Lanceolate leaves follow midrib curves drooping under gravitational deflection:
$$\mathbf{p}_{\mathrm{leaf}}(s) = \mathbf{p}_{\mathrm{att}} + s L \mathbf{d} - \frac{g}{2} s^2 \hat{\mathbf{z}} + w_{\mathrm{leaf}}(s) t \mathbf{B}_{\mathrm{leaf}}$$

### 9. Physically Motivated Optical Shading
Surface radiance balances diffuse reflection, waxy cuticle sheen, thin-sheet forward translucency, and perianth cavity ambient occlusion:
$$I(\mathbf{x}) = k_a I_a \mathbf{c}_{\mathrm{base}} \mathrm{AO}(\mathbf{x}) + \sum_{\ell=1}^3 I_\ell \left[ \mathbf{c}_{\mathrm{alb}} \max(0, \mathbf{n}\cdot\boldsymbol{\omega}_\ell) + k_{\mathrm{spec}}(\mathbf{n}\cdot\mathbf{h}_\ell)^\alpha + k_{\mathrm{sub}}\mathbf{c}_{\mathrm{trans}}\max(0, -\mathbf{n}\cdot\boldsymbol{\omega}_\ell) \right] + \mathbf{I}_{\mathrm{rim}}$$

### 10. Depth Projection & Filmic Tone Operator
Discrete pixel color emerges from Z-buffer depth minimization and ACES-Reinhard filmic tone mapping:
$$z^*(u, v) = \min_m \{ z : \mathbf{r}(u, v) \cap \mathcal{M}_m \ne \emptyset \}$$
$$\mathbf{F}_\theta(u, v) = \left( \frac{1.35 I(1 + 1.35 I / L_{\mathrm{white}}^2)}{1 + 1.35 I} \right)^{1 / 2.2}$$

---

## Repository Structure

```text
Lilium-Mathematical-Art/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── final_scene.json
├── src/
│   ├── geometry.py        # Rodrigues rotation, Bézier curves, Bishop frames, cross-sections
│   ├── lilies.py          # Trimerous perianth, tepal meshes, stamens, anthers, stigma
│   ├── foliage.py         # Bouquet spatial layout, stems, lanceolate leaves
│   ├── materials.py       # Optical BRDF, throat star, vein fields, carmine papillae
│   ├── lighting.py        # 3-point studio lighting, chiaroscuro background field
│   ├── renderer.py        # High-performance parallel Numba Z-buffer rasterizer & shader
│   ├── math_spec.py       # Formal Typst and LaTeX mathematical specifications
│   └── composition.py     # Master canvas assembly and high-resolution compositor
├── equations/
│   ├── lilium_model.tex   # Formal LaTeX specification document
│   ├── equation_panel.tex # Standalone LaTeX panel source
│   └── equation_panel.typ # Typst 300 PPI typesetting source
├── scripts/
│   ├── render_bouquet.py   # Standalone bouquet synthesis script
│   ├── render_equations.py # Standalone equation panel compilation script
│   └── compose_final.py    # Master end-to-end composite generation pipeline
├── tests/
│   ├── test_geometry.py    # Tests for Bézier curves, Dirichlet bounds, and normals
│   ├── test_determinism.py # Verifies bitwise deterministic reproducibility
│   ├── test_rendering.py   # Verifies projection, material IDs, and pixel validity
│   └── test_equations.py   # Verifies equation structure and Typst compilation
└── outputs/
    ├── lily_bouquet.png          # Standalone bouquet render (2400 x 1440)
    ├── lily_equations.png        # Standalone equation panel (2400 x 880)
    └── lilium_mathematical_art.png # Master mathematical artwork (2400 x 2400)
```

---

## Setup and Quick Start

### Installation

Requires Python 3.10+ (tested on Python 3.14):

```bash
git clone https://github.com/davea/Lilium-Mathematical-Art.git
cd Lilium-Mathematical-Art
pip install -r requirements.txt
```

### Reproducing the Complete Artwork

A single command executes the complete synthesis pipeline:

```bash
python scripts/compose_final.py
```

Expected execution time: **~5.8 seconds** on standard multicore CPU (parallelized via Numba JIT).

To generate individual components independently:

```bash
# Render standalone bouquet (outputs/lily_bouquet.png)
python scripts/render_bouquet.py

# Render standalone equation panel (outputs/lily_equations.png)
python scripts/render_equations.py
```

### Running Automated Test Suite

```bash
python -m pytest -v
```

All 16 tests verify:
- Non-negative Dirichlet boundary conditions for all petal width functions.
- Orthonormal Bishop and $\mathrm{SO}(3)$ coordinate frames.
- Bitwise identical geometry and rendering between independent runs with seed 2026.
- Bounded, finite pixel values in $[0, 255]$ with correct channel dimensions.
- Typst and LaTeX mathematical specifications consistency.

---

## Computational Performance & Hardware Requirements

- **Processor**: Standard x86-64 CPU (12 threads utilized via Numba multi-threading).
- **RAM**: $< 1.2$ GB peak working memory.
- **GPU**: Not required (pure analytical CPU rasterization).
- **Compilation**: Numba JIT compilation takes ~1.5s on initial warmup, followed by 3.3s rendering time for the complete 116,864-triangle scene at $2400 \times 1440$ resolution.

---

## License

This project is open-source under the MIT License.
