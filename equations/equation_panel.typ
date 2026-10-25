
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
