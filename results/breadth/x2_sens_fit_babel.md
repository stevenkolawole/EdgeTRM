### sensitivity at sigma0 = 0.05 (S = median over draws of median-over-inputs delta_T / sigma0); platform mode: babel
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

all fitted models (n=10): delta* = 0.400, R^2 (log, slope -1) = 0.67; free fit log sigma_1/2 = -0.88 + -0.83 log S

held out DEQ: fit on 7 models, delta* 0.366; median |log ratio| 0.66 (x1.93), worst x2.44
   deq_transformer: predicted 0.660 vs measured 0.34 (ratio 1.93)
   mdeq_small: predicted 0.119 vs measured 0.23 (ratio 0.52)
   mdeq_xl: predicted 0.152 vs measured 0.37 (ratio 0.41)

held out DT: fit on 8 models, delta* 0.435; median |log ratio| 0.41 (x1.51), worst x1.77
   dt_maze13: predicted 0.474 vs measured 0.37 (ratio 1.29)
   dt_maze33: predicted 0.466 vs measured 0.26 (ratio 1.77)

held out HRM: fit on 8 models, delta* 0.404; median |log ratio| 0.05 (x1.05), worst x1.09
   hrm_maze: predicted 1.731 vs measured 1.71 (ratio 1.01)
   hrm_sudoku: predicted 1.141 vs measured 1.05 (ratio 1.09)

held out TRM: fit on 7 models, delta* 0.393; median |log ratio| 0.33 (x1.40), worst x1.64
   maze_attn: predicted 1.226 vs measured 1.71 (ratio 0.72)
   sudoku_attn: predicted 0.570 vs measured 0.80 (ratio 0.72)
   sudoku_mlp: predicted 0.226 vs measured 0.14 (ratio 1.64)

held out TRM+HRM -> DT, DEQ: fit on 5 models, delta* 0.403; median |log ratio| 0.55 (x1.74), worst x2.22
   deq_transformer: predicted 0.727 vs measured 0.34 (ratio 2.12)
   dt_maze13: predicted 0.440 vs measured 0.37 (ratio 1.20)
   dt_maze33: predicted 0.432 vs measured 0.26 (ratio 1.64)
   mdeq_small: predicted 0.132 vs measured 0.23 (ratio 0.58)
   mdeq_xl: predicted 0.167 vs measured 0.37 (ratio 0.45)

leave-one-model-out: median x1.57, worst x2.45 (mdeq_xl)

quantizer collapse predicted by delta_T > delta* = 0.403 (fit on TRM+HRM noise widths):
   DT: 8 cells (0 collapse), accuracy 1.00, AUROC(delta) --
   DEQ: 18 cells (3 collapse), accuracy 0.72, AUROC(delta) 0.82
   ALL: 26 cells (3 collapse), accuracy 0.81, AUROC(delta) 0.88

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

skipped: [('trm_sudoku_attn_w256_s0', 'sigma_1/2 on p4d'), ('trm_sudoku_attn_w256_s1', 'sigma_1/2 on p4d'), ('trm_sudoku_attn_w256_s2', 'sigma_1/2 on p4d'), ('trm_sudoku_attn_w512_s0', 'sigma_1/2 on p4d'), ('trm_sudoku_attn_w512_s1', 'sigma_1/2 on p4d'), ('trm_sudoku_attn_w512_s2', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w256_s0', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w256_s1', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w256_s2', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w512_s0', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w512_s1', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w512_s2', 'sigma_1/2 on p4d'), ('trm_sudoku_mlp_w768_s0', 'sigma_1/2 on p4d')]
wrote results/breadth/x2_sens_fit_babel.json
