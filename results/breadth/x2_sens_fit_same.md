### sensitivity at sigma0 = 0.05 (S = median over draws of median-over-inputs delta_T / sigma0); platform mode: same
| model | family | platform | fp32 | sigma_1/2 | S(0.02) | S(0.05) | S(0.1) | S x sigma_1/2 | source |
|---|---|---|---|---|---|---|---|---|---|
| deq_transformer | DEQ | babel | 23.136 | 0.34 | 0.55 | 0.55 | 0.55 | 0.19 | X6 Anderson cap 60; half = ppl doubles |
| dt_maze13 | DT | babel | 1.000 | 0.37 | 0.87 | 0.92 | 0.94 | 0.34 | X2 grid |
| dt_maze33 | DT | babel | 1.000 | 0.26 | 0.88 | 0.93 | 0.96 | 0.25 | X2 grid |
| hrm_maze | HRM | babel | 0.748 | 1.71 | 0.23 | 0.23 | 0.23 | 0.40 | X2 width sweep, n=500 |
| hrm_sudoku | HRM | babel | 0.477 | 1.05 | 0.36 | 0.35 | 0.35 | 0.37 | X2 grid |
| maze_attn | TRM | babel | 0.875 | 1.71 | 0.32 | 0.32 | 0.32 | 0.55 | X2 grid |
| mdeq_small | DEQ | babel | 0.738 | 0.23 | 3.12 | 3.06 | 3.03 | 0.70 | X6 Broyden k=100 |
| mdeq_xl | DEQ | babel | 0.755 | 0.37 | 2.45 | 2.41 | 2.37 | 0.89 | X6 Broyden k=100 |
| sudoku_attn | TRM | babel | 0.727 | 0.80 | 0.68 | 0.69 | 0.68 | 0.55 | X2 grid |
| sudoku_mlp | TRM | babel | 0.715 | 0.14 | 1.60 | 1.74 | 2.23 | 0.24 | X2 grid |
| trm_sudoku_attn_w256_s0 (low fp32) | TRM | p4d | 0.104 | 0.37 | 1.04 | 1.03 | 1.06 | 0.38 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s1 (low fp32) | TRM | p4d | 0.172 | 0.23 | 1.31 | 1.29 | 1.30 | 0.30 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s2 (low fp32) | TRM | p4d | 0.282 | 0.28 | 1.21 | 1.21 | 1.25 | 0.34 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s0 | TRM | p4d | 0.648 | > 0.7 | 0.75 | 0.73 | 0.75 | -- | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s1 | TRM | p4d | 0.629 | 0.65 | 0.82 | 0.80 | 0.82 | 0.52 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s2 | TRM | p4d | 0.534 | 0.60 | 0.80 | 0.85 | 0.83 | 0.51 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s0 | TRM | p4d | 0.599 | 0.14 | 1.29 | 1.46 | 1.33 | 0.21 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s1 | TRM | p4d | 0.623 | 0.13 | 1.95 | 2.05 | 2.30 | 0.27 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s2 | TRM | p4d | 0.640 | 0.13 | 1.32 | 1.30 | 1.44 | 0.17 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s0 | TRM | p4d | 0.836 | 0.16 | 1.40 | 1.50 | 1.62 | 0.25 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s1 | TRM | p4d | 0.825 | 0.16 | 1.84 | 1.96 | 2.25 | 0.32 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s2 | TRM | p4d | 0.842 | 0.13 | 1.63 | 1.60 | 1.70 | 0.20 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w768_s0 | TRM | p4d | 0.753 | 0.15 | 2.05 | 2.47 | 3.63 | 0.37 | variants_noise (Q8), n=1000 |

all fitted models (n=19): delta* = 0.344, R^2 (log, slope -1) = 0.72; free fit log sigma_1/2 = -1.07 + -1.03 log S

held out DEQ: fit on 16 models, delta* 0.322; median |log ratio| 0.78 (x2.18), worst x2.78
   deq_transformer: predicted 0.580 vs measured 0.34 (ratio 1.69)
   mdeq_small: predicted 0.105 vs measured 0.23 (ratio 0.46)
   mdeq_xl: predicted 0.133 vs measured 0.37 (ratio 0.36)

held out DT: fit on 17 models, delta* 0.351; median |log ratio| 0.20 (x1.22), worst x1.43
   dt_maze13: predicted 0.383 vs measured 0.37 (ratio 1.04)
   dt_maze33: predicted 0.377 vs measured 0.26 (ratio 1.43)

held out HRM: fit on 17 models, delta* 0.339; median |log ratio| 0.13 (x1.13), worst x1.18
   hrm_maze: predicted 1.455 vs measured 1.71 (ratio 0.85)
   hrm_sudoku: predicted 0.959 vs measured 1.05 (ratio 0.91)

held out TRM: fit on 7 models, delta* 0.393; median |log ratio| 0.35 (x1.42), worst x2.37
   maze_attn: predicted 1.226 vs measured 1.71 (ratio 0.72)
   sudoku_attn: predicted 0.570 vs measured 0.80 (ratio 0.72)
   sudoku_mlp: predicted 0.226 vs measured 0.14 (ratio 1.64)
   trm_sudoku_attn_w256_s0: predicted 0.382 vs measured 0.37 (ratio 1.04) [low fp32, not scored]
   trm_sudoku_attn_w256_s1: predicted 0.304 vs measured 0.23 (ratio 1.31) [low fp32, not scored]
   trm_sudoku_attn_w256_s2: predicted 0.324 vs measured 0.28 (ratio 1.15) [low fp32, not scored]
   trm_sudoku_attn_w512_s0: predicted 0.536 vs measured > 0.7 (ratio --)
   trm_sudoku_attn_w512_s1: predicted 0.490 vs measured 0.65 (ratio 0.75)
   trm_sudoku_attn_w512_s2: predicted 0.463 vs measured 0.60 (ratio 0.77)
   trm_sudoku_mlp_w256_s0: predicted 0.269 vs measured 0.14 (ratio 1.89)
   trm_sudoku_mlp_w256_s1: predicted 0.192 vs measured 0.13 (ratio 1.44)
   trm_sudoku_mlp_w256_s2: predicted 0.304 vs measured 0.13 (ratio 2.37)
   trm_sudoku_mlp_w512_s0: predicted 0.263 vs measured 0.16 (ratio 1.60)
   trm_sudoku_mlp_w512_s1: predicted 0.201 vs measured 0.16 (ratio 1.22)
   trm_sudoku_mlp_w512_s2: predicted 0.245 vs measured 0.13 (ratio 1.95)
   trm_sudoku_mlp_w768_s0: predicted 0.159 vs measured 0.15 (ratio 1.06)

held out TRM+HRM -> DT, DEQ: fit on 14 models, delta* 0.327; median |log ratio| 0.54 (x1.72), worst x2.74
   deq_transformer: predicted 0.589 vs measured 0.34 (ratio 1.72)
   dt_maze13: predicted 0.357 vs measured 0.37 (ratio 0.97)
   dt_maze33: predicted 0.350 vs measured 0.26 (ratio 1.33)
   mdeq_small: predicted 0.107 vs measured 0.23 (ratio 0.47)
   mdeq_xl: predicted 0.136 vs measured 0.37 (ratio 0.37)

leave-one-model-out: median x1.52, worst x2.74 (mdeq_xl)

quantizer collapse predicted by delta_T > delta* = 0.327 (fit on TRM+HRM noise widths):
   TRM: 91 cells (23 collapse), accuracy 0.90, AUROC(delta) 0.98
   HRM: 26 cells (3 collapse), accuracy 0.96, AUROC(delta) 1.00
   DT: 8 cells (0 collapse), accuracy 1.00, AUROC(delta) --
   DEQ: 18 cells (3 collapse), accuracy 0.56, AUROC(delta) 0.82
   ALL: 143 cells (29 collapse), accuracy 0.87, AUROC(delta) 0.97

S(0.05) on both platforms (same model, same protocol):
   hrm_maze: Babel 0.23 vs p4d 0.23
   hrm_sudoku: Babel 0.35 vs p4d 0.35
   maze_attn: Babel 0.32 vs p4d 0.32
   sudoku_attn: Babel 0.69 vs p4d 0.69
   sudoku_mlp: Babel 1.74 vs p4d 1.74
   trm_sudoku_attn_w256_s0: Babel 1.03 vs p4d 1.03
   trm_sudoku_attn_w256_s1: Babel 1.29 vs p4d 1.29
   trm_sudoku_attn_w256_s2: Babel 1.21 vs p4d 1.21
   trm_sudoku_attn_w512_s0: Babel 0.73 vs p4d 0.73
   trm_sudoku_attn_w512_s1: Babel 0.80 vs p4d 0.80
   trm_sudoku_attn_w512_s2: Babel 0.83 vs p4d 0.85
   trm_sudoku_mlp_w256_s0: Babel 1.47 vs p4d 1.46
   trm_sudoku_mlp_w256_s1: Babel 2.05 vs p4d 2.05
   trm_sudoku_mlp_w256_s2: Babel 1.29 vs p4d 1.30
   trm_sudoku_mlp_w512_s0: Babel 1.50 vs p4d 1.50
   trm_sudoku_mlp_w512_s1: Babel 1.97 vs p4d 1.96
   trm_sudoku_mlp_w512_s2: Babel 1.61 vs p4d 1.60
   trm_sudoku_mlp_w768_s0: Babel 2.48 vs p4d 2.47
wrote results/breadth/x2_sens_fit_same.json
