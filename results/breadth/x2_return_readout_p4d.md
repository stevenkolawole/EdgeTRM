# X2 return test: read-out (`python x2_return_analyze.py --platform p4d`; rules in `x2_return_scoring.md`, fixed before any record was read)

Records: 18 files, 148 cells; platform p4d A100-40GB only.

## P1: collapse at median rho ~ 1, one threshold, rho beats phi

Cells (weight quant, activation quant, weight noise; fp32 acc >= 10%): 91, collapsed (retained < 0.5): 28.

| test | value | rule |
|---|---|---|
| (a) balanced accuracy of 'collapse iff median rho > 1' | 0.607 (plain 0.758) | >= 0.80 |
| (b) AUROC rho vs AUROC -phi | 0.604 vs 0.966 | rho > phi |
| (c) leave-one-family-out accuracy rho vs phi | 0.780 vs 0.912 | rho >= phi |
| (d) Spearman(retained, t=1 return rate) | 0.220 (Pearson 0.323); secondary, fp32-correct examples: 0.538 | >= 0.7 |
| best pooled rho threshold (post hoc, for reference) | 0.289 (bal. acc 0.679) | - |

**Verdict: FALSIFIED** (criteria a-d: [False, False, False, False]).

Collapsed cells' median rho (fp32-correct examples): 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.25, 0.33, 0.33, 0.33, 0.67, 1.33, 2.00, 2.00, 2.00, 2.00, 2.00

| family | cells | collapsed | rule bal.acc | AUROC rho | AUROC -phi | Spearman | LOFO acc rho / phi (thr rho / phi) |
|---|---|---|---|---|---|---|---|
| DT | 16 | 3 | 0.67 | 0.78 | 1.00 | 0.61 | 0.94 / 0.94 (0.29 / 0.77) |
| HRM | 22 | 7 | 0.71 | 0.81 | 0.99 | 0.47 | 0.86 / 0.95 (0.29 / 0.77) |
| MDEQ | 9 | 3 | 0.50 | 0.50 | 1.00 | 0.64 | 0.67 / 0.89 (0.29 / 0.82) |
| TRM-MLP | 17 | 6 | 0.50 | 0.58 | 0.98 | 0.69 | 0.71 / 0.88 (0.29 / 0.82) |
| TRM-attn | 27 | 9 | 0.61 | 0.46 | 0.95 | 0.33 | 0.70 / 0.89 (0.20 / 0.84) |

| compression type | cells | collapsed | rule bal.acc | AUROC rho |
|---|---|---|---|---|
| weight-quant | 46 | 9 | 0.61 | 0.66 |
| act-quant | 9 | 4 | 0.75 | 0.82 |
| weight-noise | 36 | 15 | 0.57 | 0.48 |

Per-example agreement (fp32-correct examples, all P1 cells pooled per family): t=1 return vs compressed correct

| family | examples | return rate | agree | kappa | P(correct given return) | P(correct given no return) |
|---|---|---|---|---|---|---|
| DT | 2048 | 0.940 | 0.846 | 0.379 | 0.836 | 0.000 |
| HRM | 2028 | 0.781 | 0.806 | 0.569 | 0.751 | 0.000 |
| MDEQ | 6534 | 0.999 | 0.566 | 0.001 | 0.566 | 0.375 |
| TRM-MLP | 3111 | 0.966 | 0.665 | 0.113 | 0.654 | 0.009 |
| TRM-attn | 3354 | 0.869 | 0.810 | 0.483 | 0.781 | 0.000 |

Fresh-e cells (activation noise redrawn every call):

| model | cell | fp32 | retained | return rate | median rho | phi |
|---|---|---|---|---|---|---|
| sudoku_attn | anoise0.3 | 0.742 | 1.000 | 0.770 | 0.250 | 0.910 |
| sudoku_attn | anoise0.5 | 0.742 | 0.668 | 0.738 | 0.333 | 0.751 |
| sudoku_attn | anoise0.7 | 0.742 | 0.016 | 0.555 | 0.667 | 0.443 |
| hrm_sudoku | anoise0.3 | 0.477 | 0.893 | 0.441 | 0.125 | 0.929 |
| hrm_sudoku | anoise0.5 | 0.477 | 0.287 | 0.375 | 0.500 | 0.728 |
| maze_attn | anoise1.0 | 0.875 | 0.310 | 0.896 | 0.062 | 0.919 |
| sudoku_mlp | anoise0.1 | 0.715 | 0.989 | 0.719 | 0.062 | 0.926 |
| sudoku_mlp | anoise0.2 | 0.715 | 0.918 | 0.719 | 0.062 | 0.874 |
| sudoku_mlp | anoise0.3 | 0.715 | 0.546 | 0.699 | 0.062 | 0.554 |

All cells:

| family | model | cell | type | n | fp32 | retained | return t=1 (all / fp32-correct) | median rho (fp32-correct / all) | frac t_c>=16 | phi |
|---|---|---|---|---|---|---|---|---|---|---|
| DT | dt_maze13 | noise0.1 | weight-noise | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.992 |
| DT | dt_maze13 | noise0.2 | weight-noise | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.978 |
| DT | dt_maze13 | noise0.3 | weight-noise | 128 | 1.000 | 0.562 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.922 |
| DT | dt_maze13 | noise0.5 | weight-noise | 128 | 1.000 | 0.008 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.639 |
| DT | dt_maze13 | w3a | weight-quant | 128 | 1.000 | 0.992 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.980 |
| DT | dt_maze13 | w4c | weight-quant | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.992 |
| DT | dt_maze13 | w4t | weight-quant | 128 | 1.000 | 0.977 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 0.989 |
| DT | dt_maze13 | w8c | weight-quant | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| DT | dt_maze33 | noise0.1 | weight-noise | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 0.68 | 0.992 |
| DT | dt_maze33 | noise0.2 | weight-noise | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.125 / 0.125 | 0.13 | 0.978 |
| DT | dt_maze33 | noise0.3 | weight-noise | 128 | 1.000 | 0.227 | 0.977 / 0.977 | 0.333 / 0.333 | 0.05 | 0.915 |
| DT | dt_maze33 | noise0.5 | weight-noise | 128 | 1.000 | 0.000 | 0.078 / 0.078 | 2.000 / 2.000 | 0.02 | 0.598 |
| DT | dt_maze33 | w3a | weight-quant | 128 | 1.000 | 0.969 | 1.000 / 1.000 | 0.125 / 0.125 | 0.14 | 0.980 |
| DT | dt_maze33 | w4c | weight-quant | 128 | 1.000 | 0.992 | 1.000 / 1.000 | 0.083 / 0.083 | 0.47 | 0.992 |
| DT | dt_maze33 | w4t | weight-quant | 128 | 1.000 | 0.859 | 0.992 / 0.992 | 0.167 / 0.167 | 0.06 | 0.986 |
| DT | dt_maze33 | w8c | weight-quant | 128 | 1.000 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 0.98 | 1.000 |
| HRM | hrm_maze | noise0.1 | weight-noise | 48 | 0.833 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 0.97 | 1.000 |
| HRM | hrm_maze | noise1.5 | weight-noise | 48 | 0.833 | 0.650 | 0.792 / 0.875 | 0.083 / 0.083 | 0.42 | 0.931 |
| HRM | hrm_maze | noise2.0 | weight-noise | 48 | 0.833 | 0.150 | 0.792 / 0.825 | 0.062 / 0.062 | 0.62 | 0.822 |
| HRM | hrm_maze | noise3.0 | weight-noise | 48 | 0.833 | 0.000 | 0.854 / 0.875 | 0.062 / 0.062 | 0.70 | 0.465 |
| HRM | hrm_maze | w2a | weight-quant | 48 | 0.833 | 0.950 | 0.833 / 0.925 | 0.062 / 0.062 | 0.93 | 0.959 |
| HRM | hrm_maze | w3t | weight-quant | 48 | 0.833 | 0.875 | 0.875 / 0.925 | 0.062 / 0.062 | 0.90 | 0.919 |
| HRM | hrm_maze | w4t | weight-quant | 48 | 0.833 | 1.000 | 0.917 / 0.975 | 0.062 / 0.062 | 0.93 | 0.995 |
| HRM | hrm_maze | w8c | weight-quant | 48 | 0.833 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| HRM | hrm_sudoku | anoise0.3 | act-noise(fresh) | 256 | 0.477 | 0.893 | 0.441 / 0.902 | 0.125 / 4.000 | 0.00 | 0.929 |
| HRM | hrm_sudoku | anoise0.5 | act-noise(fresh) | 256 | 0.477 | 0.287 | 0.375 / 0.770 | 0.500 / inf | 0.00 | 0.728 |
| HRM | hrm_sudoku | +a4 | act-quant | 256 | 0.477 | 0.000 | 0.180 / 0.361 | 1.333 / inf | 0.00 | 0.389 |
| HRM | hrm_sudoku | +a8 | act-quant | 256 | 0.477 | 1.008 | 0.465 / 0.934 | 0.062 / 3.000 | 0.89 | 0.969 |
| HRM | hrm_sudoku | noise0.1 | weight-noise | 256 | 0.477 | 1.033 | 0.484 / 0.951 | 0.062 / 1.333 | 0.93 | 0.978 |
| HRM | hrm_sudoku | noise0.5 | weight-noise | 256 | 0.477 | 0.926 | 0.441 / 0.885 | 0.083 / inf | 0.37 | 0.918 |
| HRM | hrm_sudoku | noise0.7 | weight-noise | 256 | 0.477 | 0.779 | 0.414 / 0.844 | 0.125 / inf | 0.12 | 0.873 |
| HRM | hrm_sudoku | noise1.0 | weight-noise | 256 | 0.477 | 0.566 | 0.426 / 0.869 | 0.250 / inf | 0.00 | 0.787 |
| HRM | hrm_sudoku | noise1.5 | weight-noise | 256 | 0.477 | 0.000 | 0.117 / 0.246 | 2.000 / inf | 0.00 | 0.496 |
| HRM | hrm_sudoku | w2a | weight-quant | 256 | 0.477 | 0.270 | 0.422 / 0.861 | 0.250 / inf | 0.00 | 0.642 |
| HRM | hrm_sudoku | w2g32 | weight-quant | 256 | 0.477 | 0.295 | 0.410 / 0.828 | 0.333 / inf | 0.05 | 0.652 |
| HRM | hrm_sudoku | w3c | weight-quant | 256 | 0.477 | 0.934 | 0.449 / 0.902 | 0.062 / 4.000 | 0.70 | 0.933 |
| HRM | hrm_sudoku | w3t | weight-quant | 256 | 0.477 | 0.000 | 0.000 / 0.000 | 2.000 / inf | 0.00 | 0.297 |
| HRM | hrm_sudoku | w4c | weight-quant | 256 | 0.477 | 1.008 | 0.469 / 0.943 | 0.062 / 3.000 | 0.91 | 0.970 |
| HRM | hrm_sudoku | w4t | weight-quant | 256 | 0.477 | 1.016 | 0.457 / 0.934 | 0.062 / inf | 0.80 | 0.953 |
| HRM | hrm_sudoku | w8c | weight-quant | 256 | 0.477 | 1.033 | 0.508 / 1.000 | 0.062 / 1.000 | 0.98 | 0.994 |
| MDEQ | mdeq_small | noise0.1 | weight-noise | 1000 | 0.726 | 0.886 | 0.997 / 0.999 | 0.062 / 0.062 | 1.00 | 0.919 |
| MDEQ | mdeq_small | noise0.2 | weight-noise | 1000 | 0.726 | 0.649 | 0.997 / 0.999 | 0.062 / 0.062 | 0.99 | 0.778 |
| MDEQ | mdeq_small | noise0.3 | weight-noise | 1000 | 0.726 | 0.218 | 0.996 / 0.997 | 0.062 / 0.062 | 1.00 | 0.620 |
| MDEQ | mdeq_small | noise0.5 | weight-noise | 1000 | 0.726 | 0.006 | 0.996 / 0.999 | 0.062 / 0.062 | 1.00 | 0.342 |
| MDEQ | mdeq_small | w4a | weight-quant | 1000 | 0.726 | 0.910 | 0.998 / 0.999 | 0.062 / 0.062 | 1.00 | 0.902 |
| MDEQ | mdeq_small | w4c | weight-quant | 1000 | 0.726 | 0.771 | 0.994 / 0.999 | 0.062 / 0.062 | 1.00 | 0.852 |
| MDEQ | mdeq_small | w4g32 | weight-quant | 1000 | 0.726 | 0.906 | 0.997 / 1.000 | 0.062 / 0.062 | 1.00 | 0.884 |
| MDEQ | mdeq_small | w4t | weight-quant | 1000 | 0.726 | 0.003 | 0.997 / 1.000 | 0.062 / 0.062 | 1.00 | 0.541 |
| MDEQ | mdeq_small | w8c | weight-quant | 1000 | 0.726 | 1.006 | 0.999 / 0.999 | 0.062 / 0.062 | 1.00 | 0.999 |
| TRM-MLP | sudoku_mlp | anoise0.1 | act-noise(fresh) | 256 | 0.715 | 0.989 | 0.719 / 0.989 | 0.062 / 0.062 | 0.96 | 0.926 |
| TRM-MLP | sudoku_mlp | anoise0.2 | act-noise(fresh) | 256 | 0.715 | 0.918 | 0.719 / 0.989 | 0.062 / 0.062 | 0.92 | 0.874 |
| TRM-MLP | sudoku_mlp | anoise0.3 | act-noise(fresh) | 256 | 0.715 | 0.546 | 0.699 / 0.962 | 0.062 / 0.062 | 0.82 | 0.554 |
| TRM-MLP | sudoku_mlp | +a4 | act-quant | 256 | 0.715 | 0.000 | 0.594 / 0.825 | 0.667 / 1.000 | 0.30 | 0.151 |
| TRM-MLP | sudoku_mlp | +a8 | act-quant | 256 | 0.715 | 1.044 | 0.727 / 1.000 | 0.062 / 0.062 | 0.96 | 0.890 |
| TRM-MLP | sudoku_mlp | w4c+a8 | act-quant | 256 | 0.715 | 1.011 | 0.707 / 0.973 | 0.062 / 0.062 | 0.95 | 0.856 |
| TRM-MLP | sudoku_mlp | noise0.05 | weight-noise | 256 | 0.715 | 1.005 | 0.719 / 0.984 | 0.062 / 0.062 | 0.96 | 0.921 |
| TRM-MLP | sudoku_mlp | noise0.1 | weight-noise | 256 | 0.715 | 0.962 | 0.707 / 0.978 | 0.062 / 0.062 | 0.93 | 0.882 |
| TRM-MLP | sudoku_mlp | noise0.15 | weight-noise | 256 | 0.715 | 0.388 | 0.691 / 0.951 | 0.062 / 0.062 | 0.83 | 0.533 |
| TRM-MLP | sudoku_mlp | noise0.2 | weight-noise | 256 | 0.715 | 0.131 | 0.707 / 0.973 | 0.062 / 0.062 | 0.80 | 0.473 |
| TRM-MLP | sudoku_mlp | noise0.3 | weight-noise | 256 | 0.715 | 0.027 | 0.703 / 0.962 | 0.062 / 0.062 | 0.78 | 0.326 |
| TRM-MLP | sudoku_mlp | w3a | weight-quant | 256 | 0.715 | 0.760 | 0.695 / 0.956 | 0.062 / 0.062 | 0.86 | 0.603 |
| TRM-MLP | sudoku_mlp | w3c | weight-quant | 256 | 0.715 | 0.355 | 0.699 / 0.967 | 0.062 / 0.062 | 0.84 | 0.303 |
| TRM-MLP | sudoku_mlp | w3g32 | weight-quant | 256 | 0.715 | 0.770 | 0.703 / 0.962 | 0.062 / 0.062 | 0.84 | 0.529 |
| TRM-MLP | sudoku_mlp | w4a | weight-quant | 256 | 0.715 | 1.016 | 0.703 / 0.978 | 0.062 / 0.062 | 0.96 | 0.907 |
| TRM-MLP | sudoku_mlp | w4c | weight-quant | 256 | 0.715 | 0.995 | 0.711 / 0.978 | 0.062 / 0.062 | 0.95 | 0.905 |
| TRM-MLP | sudoku_mlp | w4g32 | weight-quant | 256 | 0.715 | 1.005 | 0.719 / 0.989 | 0.062 / 0.062 | 0.96 | 0.919 |
| TRM-MLP | sudoku_mlp | w4t | weight-quant | 256 | 0.715 | 0.142 | 0.707 / 0.967 | 0.062 / 0.062 | 0.78 | 0.286 |
| TRM-MLP | sudoku_mlp | w5t | weight-quant | 256 | 0.715 | 0.874 | 0.715 / 0.978 | 0.062 / 0.062 | 0.92 | 0.848 |
| TRM-MLP | sudoku_mlp | w8c | weight-quant | 256 | 0.715 | 1.022 | 0.727 / 1.000 | 0.062 / 0.062 | 0.98 | 0.972 |
| TRM-attn | maze_attn | anoise1.0 | act-noise(fresh) | 48 | 0.875 | 0.310 | 0.896 / 0.929 | 0.062 / 0.062 | 0.88 | 0.919 |
| TRM-attn | maze_attn | +a4 | act-quant | 48 | 0.875 | 0.000 | 0.875 / 0.881 | 0.062 / 0.062 | 0.81 | 0.284 |
| TRM-attn | maze_attn | noise0.1 | weight-noise | 48 | 0.875 | 1.000 | 0.979 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| TRM-attn | maze_attn | noise1.0 | weight-noise | 48 | 0.875 | 1.024 | 0.979 / 1.000 | 0.062 / 0.062 | 0.98 | 0.949 |
| TRM-attn | maze_attn | noise1.5 | weight-noise | 48 | 0.875 | 0.310 | 0.938 / 0.976 | 0.062 / 0.062 | 0.95 | 0.884 |
| TRM-attn | maze_attn | noise2.0 | weight-noise | 48 | 0.875 | 0.000 | 0.979 / 1.000 | 0.062 / 0.062 | 0.93 | 0.749 |
| TRM-attn | maze_attn | noise3.0 | weight-noise | 48 | 0.875 | 0.000 | 0.917 / 0.929 | 0.062 / 0.062 | 0.79 | -0.010 |
| TRM-attn | maze_attn | w2a | weight-quant | 48 | 0.875 | 0.881 | 0.917 / 0.976 | 0.062 / 0.062 | 0.93 | 0.875 |
| TRM-attn | maze_attn | w2g32 | weight-quant | 48 | 0.875 | 0.405 | 0.938 / 0.976 | 0.062 / 0.062 | 0.88 | 0.766 |
| TRM-attn | maze_attn | w3c | weight-quant | 48 | 0.875 | 1.048 | 0.979 / 1.000 | 0.062 / 0.062 | 1.00 | 0.991 |
| TRM-attn | maze_attn | w3t | weight-quant | 48 | 0.875 | 0.000 | 0.896 / 0.929 | 0.062 / 0.062 | 0.79 | 0.403 |
| TRM-attn | maze_attn | w4t | weight-quant | 48 | 0.875 | 1.024 | 0.958 / 1.000 | 0.062 / 0.062 | 1.00 | 0.994 |
| TRM-attn | maze_attn | w8c | weight-quant | 48 | 0.875 | 1.000 | 0.979 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| TRM-attn | sudoku_attn | anoise0.3 | act-noise(fresh) | 256 | 0.742 | 1.000 | 0.770 / 0.979 | 0.250 / 0.250 | 0.00 | 0.910 |
| TRM-attn | sudoku_attn | anoise0.5 | act-noise(fresh) | 256 | 0.742 | 0.668 | 0.738 / 0.937 | 0.333 / 0.333 | 0.00 | 0.751 |
| TRM-attn | sudoku_attn | anoise0.7 | act-noise(fresh) | 256 | 0.742 | 0.016 | 0.555 / 0.716 | 0.667 / 1.000 | 0.00 | 0.443 |
| TRM-attn | sudoku_attn | +a4 | act-quant | 256 | 0.742 | 0.000 | 0.074 / 0.095 | 2.000 / 2.000 | 0.00 | 0.041 |
| TRM-attn | sudoku_attn | +a8 | act-quant | 256 | 0.742 | 1.005 | 0.770 / 0.984 | 0.167 / 0.167 | 0.02 | 0.947 |
| TRM-attn | sudoku_attn | w4c+a8 | act-quant | 256 | 0.742 | 1.005 | 0.766 / 0.974 | 0.167 / 0.167 | 0.02 | 0.925 |
| TRM-attn | sudoku_attn | noise0.1 | weight-noise | 256 | 0.742 | 0.989 | 0.770 / 0.984 | 0.167 / 0.167 | 0.11 | 0.951 |
| TRM-attn | sudoku_attn | noise0.3 | weight-noise | 256 | 0.742 | 0.911 | 0.738 / 0.947 | 0.167 / 0.167 | 0.01 | 0.898 |
| TRM-attn | sudoku_attn | noise0.5 | weight-noise | 256 | 0.742 | 0.895 | 0.766 / 0.979 | 0.167 / 0.250 | 0.01 | 0.851 |
| TRM-attn | sudoku_attn | noise0.7 | weight-noise | 256 | 0.742 | 0.711 | 0.746 / 0.963 | 0.167 / 0.250 | 0.02 | 0.719 |
| TRM-attn | sudoku_attn | noise1.0 | weight-noise | 256 | 0.742 | 0.068 | 0.527 / 0.679 | 0.333 / 1.000 | 0.01 | 0.408 |
| TRM-attn | sudoku_attn | w3a | weight-quant | 256 | 0.742 | 0.900 | 0.742 / 0.968 | 0.250 / 0.250 | 0.01 | 0.867 |
| TRM-attn | sudoku_attn | w3c | weight-quant | 256 | 0.742 | 0.847 | 0.742 / 0.958 | 0.167 / 0.250 | 0.03 | 0.824 |
| TRM-attn | sudoku_attn | w3g32 | weight-quant | 256 | 0.742 | 0.953 | 0.758 / 0.979 | 0.167 / 0.167 | 0.04 | 0.901 |
| TRM-attn | sudoku_attn | w3t | weight-quant | 256 | 0.742 | 0.000 | 0.250 / 0.326 | 2.000 / 2.000 | 0.00 | 0.017 |
| TRM-attn | sudoku_attn | w4c | weight-quant | 256 | 0.742 | 0.963 | 0.762 / 0.974 | 0.167 / 0.167 | 0.02 | 0.922 |
| TRM-attn | sudoku_attn | w4t | weight-quant | 256 | 0.742 | 0.942 | 0.762 / 0.974 | 0.250 / 0.250 | 0.01 | 0.875 |
| TRM-attn | sudoku_attn | w8c | weight-quant | 256 | 0.742 | 1.000 | 0.773 / 0.984 | 0.062 / 0.125 | 0.54 | 0.978 |

## P3: sampling gain peaks at intermediate return rates; the head picks draws with smaller rho

**Verdict: HEAD-PICK PART FALSIFIED** (head picks larger t_c in every model: False; gain peaks at intermediate return rate: None).

| model | draws | n | picked t_c > / < median-over-draws | sign-test p | mean t_c picked / all | acc RTN / best-Q | agreement with M2 per-example correctness |
|---|---|---|---|---|---|---|---|
| hrm_sudoku w3c | 16 | 256 | 43 / 15 | 0.0003 | 4.20 / 4.02 | 0.445 / 0.727 | 0.841 |
| sudoku_attn w3c | 16 | 256 | 73 / 57 | 0.1881 | 4.15 / 4.01 | 0.617 / 0.934 | 0.768 |
| sudoku_mlp w4c | 16 | 256 | 16 / 4 | 0.0118 | 11.53 / 10.89 | 0.707 / 0.938 | 0.855 |

| task | spec | RTN acc (M2 / mine) | best-Q@16 gain (M2) | RTN return rate (all / fp32-correct) | median rho | retained |
|---|---|---|---|---|---|---|

