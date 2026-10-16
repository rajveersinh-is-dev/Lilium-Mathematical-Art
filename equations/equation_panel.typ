
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

$ H_v (x, y) = sum_(k=1)^(7) lr(( product_(r=1)^(k-1) (1 - A_r (x, y)) )) lr([ U_k (x, y) L_(v, k) (x, y) + (1 - U_k (x, y)) B_v (x, y) ]), quad v in {0, 1, 2}, $

$ L_(v, k) (x, y) = bold(c)_("alb", v) (k) I_a A_"ssao" (x, y) + I_"key" lr([ bold(c)_("alb", v) (k) (d_1 S_"key" + t_1) + bold(c)_("ivory", v) b_"sub" + s_1 S_"key" ]) + I_("fill", v) + I_("rim", v), $

$ bold(S)_(k, j) (s, t) = bold(c)_k + bold(R)_k lr([ bold(p)_(k, j) (s) + t thin w(s) bold(B)_(k, j) (s) + c(s, t) bold(N)_(k, j) (s) ]), quad s in [0, 1], quad t in [-1, 1], quad j in {0, dots, 5}, $

$ bold(p)_(k, j) (s) = lr(( r_k (s) cos(psi_(k, j) + delta_"whorl"), thin r_k (s) sin(psi_(k, j) + delta_"whorl"), thin 0.32 L s - h_"refl" max(0, s - 0.38)^(1.85) + z_"curl" s^2 )), $

$ w(s) = C_w s^(1.20) (1 - s)^(0.88) lr([ 1 + 0.35 sin(pi s) - 0.12 sin(2pi s) ]), quad c(s, t) = -delta_"groove" e^(-26 t^2) (1 - 0.45 s) + delta_"arch" s (1 - t^2) + A_"ruf" s^(1.4) t^2 sin(2pi omega s), $

$ V(s, t) = 1 + sum_(i=1)^4 a_i exp lr(( -frac((|t| - 0.19 i (1 - 0.12 s))^2, 2 sigma_i^2) )), quad Pi(s, t) = sum_(p=1)^(N_p) h_p exp lr(( -k_p lr([ (s - s_p)^2 + (t - t_p)^2 ]) )), $

$ r_"theca" (theta) = R_0 (1 - 0.35 cos 2theta) e^(-32 t^2), quad r_"stigma" (theta) = R_s (1 + 0.42 cos 3theta), quad bold(p)_"leaf" (s) = bold(p)_"att" + s L bold(d) - frac(g, 2) s^2 bold(hat(z)) + w_"leaf" (s) t bold(B)_"twist" (s), $

$ S_"key" (bold(x)) = frac(1, 16) sum_(i=1)^(16) bb(1) lr([ z_L (bold(x)) - delta_"bias" <= Z_"SM" (bold(u)_L + r_"pcf" bold(xi)_i) ]), quad A_"ssao" (bold(x)) = max lr(( 0, 1 - frac(1, 16) sum_(i=1)^(16) bb(1) lr([ z_s > z_"sample" ]) frac(max(0, bold(n) dot bold(v)_i), 1 + d_i^2) )), $

$ C(z) = min lr(( r_max, k_"CoC" frac(|z - 4.15|, z) )), quad bold(I)_"DoF" (bold(u)) = frac(1, sum_(i=1)^(32) w_i) sum_(i=1)^(32) w_i bold(I) lr(( bold(u) + C(z) bold(phi)_i )), quad bold(phi)_i = sqrt(frac(i + 0.5, 32)) mat(cos(2.40 i); sin(2.40 i)), quad delta_"CA" = kappa_"disp" r^2. $
