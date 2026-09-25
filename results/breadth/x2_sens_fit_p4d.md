### sensitivity at sigma0 = 0.05 (S = median over draws of median-over-inputs delta_T / sigma0); platform mode: p4d
| model | family | platform | fp32 | sigma_1/2 | S(0.02) | S(0.05) | S(0.1) | S x sigma_1/2 | source |
|---|---|---|---|---|---|---|---|---|---|
| trm_sudoku_attn_w256_s0 (low fp32) | TRM | p4d | 0.104 | 0.37 | 1.04 | 1.03 | 1.06 | 0.38 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s1 (low fp32) | TRM | p4d | 0.172 | 0.23 | 1.31 | 1.29 | 1.30 | 0.30 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w256_s2 (low fp32) | TRM | p4d | 0.282 | 0.28 | 1.21 | 1.21 | 1.25 | 0.34 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s0 | TRM | p4d | 0.648 | 0.74 | 0.75 | 0.73 | 0.75 | 0.54 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s1 | TRM | p4d | 0.629 | 0.65 | 0.82 | 0.80 | 0.82 | 0.52 | variants_noise (Q8), n=1000 |
| trm_sudoku_attn_w512_s2 | TRM | p4d | 0.534 | 0.60 | 0.80 | 0.85 | 0.83 | 0.51 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s0 | TRM | p4d | 0.599 | 0.14 | 1.29 | 1.46 | 1.33 | 0.21 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s1 | TRM | p4d | 0.623 | 0.13 | 1.95 | 2.05 | 2.30 | 0.27 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w256_s2 | TRM | p4d | 0.640 | 0.13 | 1.32 | 1.30 | 1.44 | 0.17 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s0 | TRM | p4d | 0.836 | 0.16 | 1.40 | 1.50 | 1.62 | 0.25 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s1 | TRM | p4d | 0.825 | 0.16 | 1.84 | 1.96 | 2.25 | 0.32 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w512_s2 | TRM | p4d | 0.842 | 0.13 | 1.63 | 1.60 | 1.70 | 0.20 | variants_noise (Q8), n=1000 |
| trm_sudoku_mlp_w768_s0 | TRM | p4d | 0.753 | 0.15 | 2.05 | 2.47 | 3.63 | 0.37 | variants_noise (Q8), n=1000 |

all fitted models (n=10): delta* = 0.309, R^2 (log, slope -1) = 0.67; free fit log sigma_1/2 = -1.00 + -1.55 log S

leave-one-model-out: median x1.58, worst x2.00 (trm_sudoku_mlp_w256_s2)

quantizer collapse predicted by delta_T > delta* = 0.309 (fit on TRM+HRM noise widths):
   TRM: 91 cells (23 collapse), accuracy 0.89, AUROC(delta) 0.98
   HRM: 26 cells (3 collapse), accuracy 0.96, AUROC(delta) 1.00
   ALL: 117 cells (26 collapse), accuracy 0.91, AUROC(delta) 0.98

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

skipped: [('deq_transformer', 'sigma_1/2 on babel'), ('dt_maze13', 'sigma_1/2 on babel'), ('dt_maze33', 'sigma_1/2 on babel'), ('hrm_maze', 'sigma_1/2 on babel'), ('hrm_sudoku', 'sigma_1/2 on babel'), ('maze_attn', 'sigma_1/2 on babel'), ('mdeq_small', 'sigma_1/2 on babel'), ('mdeq_xl', 'sigma_1/2 on babel'), ('sudoku_attn', 'sigma_1/2 on babel'), ('sudoku_mlp', 'sigma_1/2 on babel')]
wrote results/breadth/x2_sens_fit_p4d.json
