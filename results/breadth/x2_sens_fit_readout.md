### sensitivity at sigma0 = 0.05 (S = median over draws of median-over-inputs delta_T / sigma0)
| model | family | fp32 | sigma_1/2 | S(0.02) | S(0.05) | S(0.1) | S x sigma_1/2 | source |
|---|---|---|---|---|---|---|---|---|
| deq_transformer | DEQ | 23.136 | 0.34 | 0.55 | 0.55 | 0.55 | 0.19 | X6 Anderson cap 60; half = ppl doubles |
| dt_maze13 | DT | 1.000 | 0.37 | 0.87 | 0.92 | 0.94 | 0.34 | X2 grid |
| dt_maze33 | DT | 1.000 | 0.26 | 0.88 | 0.93 | 0.96 | 0.25 | X2 grid |
| hrm_maze | HRM | 0.748 | 1.71 | 0.23 | 0.23 | 0.23 | 0.40 | X2 width sweep, n=500 |
| hrm_sudoku | HRM | 0.477 | 1.05 | 0.36 | 0.35 | 0.35 | 0.37 | X2 grid |
| maze_attn | TRM | 0.875 | 1.71 | 0.32 | 0.32 | 0.32 | 0.55 | X2 grid |
| mdeq_small | DEQ | 0.738 | 0.23 | 3.12 | 3.06 | 3.03 | 0.70 | X6 Broyden k=100 |
| mdeq_xl | DEQ | 0.755 | 0.37 | 2.45 | 2.41 | 2.37 | 0.89 | X6 Broyden k=100 |
| sudoku_attn | TRM | 0.727 | 0.80 | 0.68 | 0.69 | 0.68 | 0.55 | X2 grid |
| sudoku_mlp | TRM | 0.715 | 0.14 | 1.60 | 1.74 | 2.23 | 0.24 | X2 grid |
| trm_sudoku_attn_w256_s0 (low fp32) | TRM | 0.104 | 0.37 | 1.04 | 1.03 | 1.06 | 0.38 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s1 (low fp32) | TRM | 0.172 | 0.23 | 1.31 | 1.29 | 1.30 | 0.30 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s2 (low fp32) | TRM | 0.282 | 0.28 | 1.21 | 1.21 | 1.25 | 0.34 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s0 | TRM | 0.648 | 0.74 | 0.75 | 0.73 | 0.75 | 0.54 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s1 | TRM | 0.629 | 0.65 | 0.82 | 0.80 | 0.82 | 0.52 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s2 | TRM | 0.534 | 0.60 | 0.82 | 0.83 | 0.83 | 0.50 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s0 | TRM | 0.599 | 0.14 | 1.30 | 1.47 | 1.33 | 0.21 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s1 | TRM | 0.623 | 0.13 | 1.96 | 2.05 | 2.32 | 0.27 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s2 | TRM | 0.640 | 0.13 | 1.32 | 1.29 | 1.44 | 0.17 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s0 | TRM | 0.836 | 0.16 | 1.40 | 1.50 | 1.62 | 0.25 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s1 | TRM | 0.825 | 0.16 | 1.84 | 1.97 | 2.26 | 0.32 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s2 | TRM | 0.842 | 0.13 | 1.64 | 1.61 | 1.71 | 0.20 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w768_s0 | TRM | 0.753 | 0.15 | 2.05 | 2.48 | 3.64 | 0.37 | variants_noise (Q8), n=1000 |

all fitted models (n=20): delta* = 0.352, R^2 (log, slope -1) = 0.72; free fit log sigma_1/2 = -1.04 + -1.04 log S

held out DEQ: fit on 17 models, delta* 0.331; median |log ratio| 0.75 (x2.11), worst x2.70
   deq_transformer: predicted 0.597 vs measured 0.34 (ratio 1.74)
   mdeq_small: predicted 0.108 vs measured 0.23 (ratio 0.47)
   mdeq_xl: predicted 0.137 vs measured 0.37 (ratio 0.37)

held out DT: fit on 18 models, delta* 0.360; median |log ratio| 0.22 (x1.25), worst x1.47
   dt_maze13: predicted 0.392 vs measured 0.37 (ratio 1.07)
   dt_maze33: predicted 0.385 vs measured 0.26 (ratio 1.47)

held out HRM: fit on 18 models, delta* 0.348; median |log ratio| 0.10 (x1.11), worst x1.15
   hrm_maze: predicted 1.492 vs measured 1.71 (ratio 0.87)
   hrm_sudoku: predicted 0.983 vs measured 1.05 (ratio 0.94)

held out TRM: fit on 7 models, delta* 0.393; median |log ratio| 0.33 (x1.40), worst x2.38
   maze_attn: predicted 1.226 vs measured 1.71 (ratio 0.72)
   sudoku_attn: predicted 0.570 vs measured 0.80 (ratio 0.72)
   sudoku_mlp: predicted 0.226 vs measured 0.14 (ratio 1.64)
   trm_sudoku_attn_w256_s0: predicted 0.381 vs measured 0.37 (ratio 1.04) [low fp32, not scored]
   trm_sudoku_attn_w256_s1: predicted 0.304 vs measured 0.23 (ratio 1.31) [low fp32, not scored]
   trm_sudoku_attn_w256_s2: predicted 0.324 vs measured 0.28 (ratio 1.15) [low fp32, not scored]
   trm_sudoku_attn_w512_s0: predicted 0.535 vs measured 0.74 (ratio 0.73)
   trm_sudoku_attn_w512_s1: predicted 0.490 vs measured 0.65 (ratio 0.75)
   trm_sudoku_attn_w512_s2: predicted 0.476 vs measured 0.60 (ratio 0.79)
   trm_sudoku_mlp_w256_s0: predicted 0.267 vs measured 0.14 (ratio 1.88)
   trm_sudoku_mlp_w256_s1: predicted 0.192 vs measured 0.13 (ratio 1.44)
   trm_sudoku_mlp_w256_s2: predicted 0.304 vs measured 0.13 (ratio 2.38)
   trm_sudoku_mlp_w512_s0: predicted 0.263 vs measured 0.16 (ratio 1.60)
   trm_sudoku_mlp_w512_s1: predicted 0.200 vs measured 0.16 (ratio 1.22)
   trm_sudoku_mlp_w512_s2: predicted 0.245 vs measured 0.13 (ratio 1.94)
   trm_sudoku_mlp_w768_s0: predicted 0.159 vs measured 0.15 (ratio 1.05)

held out TRM+HRM -> DT, DEQ: fit on 15 models, delta* 0.338; median |log ratio| 0.58 (x1.78), worst x2.65
   deq_transformer: predicted 0.609 vs measured 0.34 (ratio 1.78)
   dt_maze13: predicted 0.368 vs measured 0.37 (ratio 1.00)
   dt_maze33: predicted 0.362 vs measured 0.26 (ratio 1.38)
   mdeq_small: predicted 0.110 vs measured 0.23 (ratio 0.48)
   mdeq_xl: predicted 0.140 vs measured 0.37 (ratio 0.38)

leave-one-model-out: median x1.51, worst x2.67 (mdeq_xl)

quantizer collapse predicted by delta_T > delta* = 0.338 (fit on TRM+HRM noise widths):
   TRM: 91 cells (23 collapse), accuracy 0.92, AUROC(delta) 0.98
   HRM: 26 cells (3 collapse), accuracy 0.96, AUROC(delta) 1.00
   DT: 8 cells (0 collapse), accuracy 1.00, AUROC(delta) --
   DEQ: 18 cells (3 collapse), accuracy 0.56, AUROC(delta) 0.82
   LM: 90 cells (27 collapse), accuracy 0.92, AUROC(delta) 0.98
   ALL: 233 cells (56 collapse), accuracy 0.90, AUROC(delta) 0.97
wrote /home/stevenkolawole/workspace/emnlp-26/EdgeTRM/results/breadth/x2_sens_fit.json
