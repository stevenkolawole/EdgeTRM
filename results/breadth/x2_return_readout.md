# X2 return test: read-out (`python x2_return_analyze.py --platform babel`; rules in `x2_return_scoring.md`, fixed before any record was read)

Records: 42 files, 243 cells; platform Babel L40S only.

## P1: collapse at median rho ~ 1, one threshold, rho beats phi

Cells (weight quant, activation quant, weight noise; fp32 acc >= 10%): 179, collapsed (retained < 0.5): 54.

| test | value | rule |
|---|---|---|
| (a) balanced accuracy of 'collapse iff median rho > 1' | 0.630 (plain 0.777) | >= 0.80 |
| (b) AUROC rho vs AUROC -phi | 0.721 vs 0.978 | rho > phi |
| (c) leave-one-family-out accuracy rho vs phi | 0.743 vs 0.939 | rho >= phi |
| (d) Spearman(retained, t=1 return rate) | 0.320 (Pearson 0.424); secondary, fp32-correct examples: 0.595 | >= 0.7 |
| best pooled rho threshold (post hoc, for reference) | 0.289 (bal. acc 0.709) | - |

**Verdict: FALSIFIED** (criteria a-d: [False, False, False, False]).

Collapsed cells' median rho (fp32-correct examples): 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12, 0.12, 0.17, 0.17, 0.25, 0.25, 0.33, 0.33, 0.33, 0.33, 0.33, 0.67, 0.67, 1.00, 1.00, 1.17, 1.17, 1.17, 1.33, 1.33, 1.33, 1.33, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00

| family | cells | collapsed | rule bal.acc | AUROC rho | AUROC -phi | Spearman | LOFO acc rho / phi (thr rho / phi) |
|---|---|---|---|---|---|---|---|
| DT | 16 | 3 | 0.67 | 0.78 | 1.00 | 0.61 | 0.75 / 0.94 (0.10 / 0.69) |
| HRM | 22 | 7 | 0.71 | 0.81 | 0.99 | 0.46 | 0.86 / 0.95 (0.29 / 0.69) |
| MDEQ | 13 | 3 | 0.50 | 0.50 | 1.00 | 0.34 | 0.77 / 1.00 (0.29 / 0.69) |
| QAT | 28 | 10 | 0.60 | 1.00 | 1.00 | 0.68 | 0.86 / 0.93 (0.29 / 0.77) |
| TRM-MLP | 31 | 13 | 0.65 | 0.73 | 1.00 | 0.60 | 0.74 / 0.94 (0.20 / 0.77) |
| TRM-attn | 33 | 10 | 0.65 | 0.54 | 0.96 | 0.06 | 0.45 / 0.91 (0.10 / 0.69) |
| seeds | 36 | 8 | 0.56 | 0.69 | 1.00 | 0.30 | 0.83 / 0.94 (0.29 / 0.77) |

| compression type | cells | collapsed | rule bal.acc | AUROC rho |
|---|---|---|---|---|
| weight-quant | 93 | 23 | 0.70 | 0.76 |
| act-quant | 9 | 4 | 0.75 | 0.82 |
| weight-noise | 77 | 27 | 0.56 | 0.66 |

Per-example agreement (fp32-correct examples, all P1 cells pooled per family): t=1 return vs compressed correct

| family | examples | return rate | agree | kappa | P(correct given return) | P(correct given no return) |
|---|---|---|---|---|---|---|
| DT | 2048 | 0.940 | 0.846 | 0.379 | 0.836 | 0.000 |
| HRM | 2028 | 0.782 | 0.806 | 0.569 | 0.751 | 0.000 |
| MDEQ | 9508 | 0.997 | 0.666 | 0.010 | 0.665 | 0.138 |
| QAT | 4711 | 0.814 | 0.719 | 0.413 | 0.654 | 0.000 |
| TRM-MLP | 5933 | 0.874 | 0.666 | 0.289 | 0.618 | 0.001 |
| TRM-attn | 4286 | 0.878 | 0.824 | 0.470 | 0.810 | 0.082 |
| seeds | 7358 | 0.929 | 0.780 | 0.309 | 0.764 | 0.017 |

Fresh-e cells (activation noise redrawn every call):

| model | cell | fp32 | retained | return rate | median rho | phi |
|---|---|---|---|---|---|---|
| hrm_sudoku | anoise0.3 | 0.477 | 0.885 | 0.453 | 0.125 | 0.931 |
| hrm_sudoku | anoise0.5 | 0.477 | 0.197 | 0.367 | 0.500 | 0.709 |
| maze_attn | anoise1.0 | 0.875 | 0.310 | 0.958 | 0.062 | 0.916 |
| sudoku_attn | anoise0.3 | 0.727 | 1.000 | 0.758 | 0.250 | 0.913 |
| sudoku_attn | anoise0.5 | 0.727 | 0.656 | 0.727 | 0.333 | 0.748 |
| sudoku_attn | anoise0.7 | 0.727 | 0.016 | 0.605 | 0.500 | 0.450 |
| sudoku_mlp | anoise0.1 | 0.715 | 0.995 | 0.715 | 0.062 | 0.924 |
| sudoku_mlp | anoise0.2 | 0.715 | 0.907 | 0.719 | 0.062 | 0.855 |
| sudoku_mlp | anoise0.3 | 0.715 | 0.541 | 0.695 | 0.062 | 0.550 |

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
| HRM | hrm_sudoku | anoise0.3 | act-noise(fresh) | 256 | 0.477 | 0.885 | 0.453 / 0.893 | 0.125 / 4.000 | 0.00 | 0.931 |
| HRM | hrm_sudoku | anoise0.5 | act-noise(fresh) | 256 | 0.477 | 0.197 | 0.367 / 0.738 | 0.500 / 4.000 | 0.00 | 0.709 |
| HRM | hrm_sudoku | +a4 | act-quant | 256 | 0.477 | 0.000 | 0.176 / 0.344 | 1.333 / 4.000 | 0.00 | 0.389 |
| HRM | hrm_sudoku | +a8 | act-quant | 256 | 0.477 | 1.057 | 0.477 / 0.926 | 0.062 / 1.667 | 0.89 | 0.969 |
| HRM | hrm_sudoku | noise0.1 | weight-noise | 256 | 0.477 | 1.033 | 0.500 / 0.967 | 0.062 / 1.333 | 0.93 | 0.978 |
| HRM | hrm_sudoku | noise0.5 | weight-noise | 256 | 0.477 | 0.926 | 0.457 / 0.902 | 0.083 / 2.667 | 0.36 | 0.918 |
| HRM | hrm_sudoku | noise0.7 | weight-noise | 256 | 0.477 | 0.779 | 0.414 / 0.820 | 0.125 / 4.000 | 0.12 | 0.873 |
| HRM | hrm_sudoku | noise1.0 | weight-noise | 256 | 0.477 | 0.566 | 0.434 / 0.877 | 0.250 / inf | 0.00 | 0.787 |
| HRM | hrm_sudoku | noise1.5 | weight-noise | 256 | 0.477 | 0.000 | 0.121 / 0.246 | 2.000 / inf | 0.00 | 0.496 |
| HRM | hrm_sudoku | w2a | weight-quant | 256 | 0.477 | 0.270 | 0.422 / 0.861 | 0.250 / inf | 0.00 | 0.642 |
| HRM | hrm_sudoku | w2g32 | weight-quant | 256 | 0.477 | 0.295 | 0.410 / 0.828 | 0.333 / inf | 0.05 | 0.652 |
| HRM | hrm_sudoku | w3c | weight-quant | 256 | 0.477 | 0.934 | 0.461 / 0.910 | 0.062 / 3.000 | 0.70 | 0.933 |
| HRM | hrm_sudoku | w3t | weight-quant | 256 | 0.477 | 0.000 | 0.000 / 0.000 | 2.000 / inf | 0.00 | 0.297 |
| HRM | hrm_sudoku | w4c | weight-quant | 256 | 0.477 | 1.008 | 0.477 / 0.943 | 0.062 / 1.333 | 0.91 | 0.969 |
| HRM | hrm_sudoku | w4t | weight-quant | 256 | 0.477 | 1.016 | 0.473 / 0.951 | 0.062 / 3.000 | 0.80 | 0.953 |
| HRM | hrm_sudoku | w8c | weight-quant | 256 | 0.477 | 1.033 | 0.504 / 0.992 | 0.062 / 1.167 | 0.98 | 0.994 |
| MDEQ | mdeq_small | noise0.1 | weight-noise | 1000 | 0.726 | 0.898 | 0.998 / 0.999 | 0.062 / 0.062 | 1.00 | 0.919 |
| MDEQ | mdeq_small | noise0.2 | weight-noise | 1000 | 0.726 | 0.639 | 0.995 / 0.997 | 0.062 / 0.062 | 1.00 | 0.778 |
| MDEQ | mdeq_small | noise0.3 | weight-noise | 1000 | 0.726 | 0.196 | 0.996 / 0.999 | 0.062 / 0.062 | 1.00 | 0.620 |
| MDEQ | mdeq_small | w4a | weight-quant | 1000 | 0.726 | 0.910 | 0.998 / 0.999 | 0.062 / 0.062 | 1.00 | 0.902 |
| MDEQ | mdeq_small | w4c | weight-quant | 1000 | 0.726 | 0.771 | 0.994 / 0.999 | 0.062 / 0.062 | 1.00 | 0.852 |
| MDEQ | mdeq_small | w4g32 | weight-quant | 1000 | 0.726 | 0.906 | 0.997 / 1.000 | 0.062 / 0.062 | 1.00 | 0.884 |
| MDEQ | mdeq_small | w4t | weight-quant | 1000 | 0.726 | 0.003 | 0.997 / 1.000 | 0.062 / 0.062 | 1.00 | 0.541 |
| MDEQ | mdeq_small | w8c | weight-quant | 1000 | 0.726 | 1.006 | 0.999 / 0.999 | 0.062 / 0.062 | 1.00 | 0.999 |
| MDEQ | mdeq_xl | noise0.3 | weight-noise | 1000 | 0.740 | 0.768 | 0.986 / 0.992 | 0.062 / 0.062 | 0.98 | 0.769 |
| MDEQ | mdeq_xl | w4c | weight-quant | 1000 | 0.740 | 0.931 | 0.987 / 0.995 | 0.062 / 0.062 | 0.98 | 0.870 |
| MDEQ | mdeq_xl | w4g32 | weight-quant | 1000 | 0.740 | 0.986 | 0.991 / 0.995 | 0.062 / 0.062 | 0.98 | 0.939 |
| MDEQ | mdeq_xl | w4t | weight-quant | 1000 | 0.740 | 0.001 | 0.973 / 0.991 | 0.062 / 0.062 | 0.98 | 0.349 |
| MDEQ | mdeq_xl | w8c | weight-quant | 1000 | 0.740 | 0.999 | 0.996 / 0.999 | 0.062 / 0.062 | 0.99 | 0.999 |
| QAT | ft_fp32_s0 | noise0.1 | weight-noise | 256 | 0.664 | 0.847 | 0.633 / 0.912 | 0.062 / 0.062 | 0.85 | 0.892 |
| QAT | ft_fp32_s0 | noise0.2 | weight-noise | 256 | 0.664 | 0.112 | 0.602 / 0.865 | 0.167 / 0.333 | 0.34 | 0.577 |
| QAT | ft_fp32_s0 | noise0.3 | weight-noise | 256 | 0.664 | 0.012 | 0.359 / 0.535 | 1.000 / 1.333 | 0.01 | 0.452 |
| QAT | ft_fp32_s0 | noise0.5 | weight-noise | 256 | 0.664 | 0.000 | 0.004 / 0.006 | 1.333 / 2.000 | 0.00 | 0.234 |
| QAT | ft_fp32_s0 | w3g32 | weight-quant | 256 | 0.664 | 0.500 | 0.559 / 0.812 | 0.083 / 0.667 | 0.48 | 0.701 |
| QAT | ft_fp32_s0 | w4c | weight-quant | 256 | 0.664 | 0.888 | 0.637 / 0.935 | 0.062 / 0.062 | 0.86 | 0.896 |
| QAT | ft_fp32_s0 | w4t | weight-quant | 256 | 0.664 | 0.000 | 0.000 / 0.000 | 2.000 / 2.000 | 0.00 | 0.272 |
| QAT | qat_lsq_s0 | noise0.1 | weight-noise | 256 | 0.578 | 1.014 | 0.574 / 0.932 | 0.062 / 0.062 | 0.91 | 0.948 |
| QAT | qat_lsq_s0 | noise0.2 | weight-noise | 256 | 0.578 | 0.676 | 0.578 / 0.932 | 0.062 / 0.333 | 0.67 | 0.820 |
| QAT | qat_lsq_s0 | noise0.3 | weight-noise | 256 | 0.578 | 0.365 | 0.570 / 0.939 | 0.125 / 0.333 | 0.16 | 0.672 |
| QAT | qat_lsq_s0 | noise0.5 | weight-noise | 256 | 0.578 | 0.007 | 0.320 / 0.520 | 1.000 / 2.000 | 0.00 | 0.428 |
| QAT | qat_lsq_s0 | w3g32 | weight-quant | 256 | 0.578 | 0.520 | 0.574 / 0.912 | 0.083 / 0.333 | 0.28 | 0.737 |
| QAT | qat_lsq_s0 | w4c | weight-quant | 256 | 0.578 | 1.020 | 0.598 / 0.953 | 0.062 / 0.062 | 0.93 | 0.951 |
| QAT | qat_lsq_s0 | w4t | weight-quant | 256 | 0.578 | 1.135 | 0.605 / 0.973 | 0.062 / 0.062 | 0.95 | 0.933 |
| QAT | qat_lsq_s1 | noise0.1 | weight-noise | 256 | 0.699 | 0.939 | 0.711 / 0.978 | 0.062 / 0.062 | 0.90 | 0.942 |
| QAT | qat_lsq_s1 | noise0.2 | weight-noise | 256 | 0.699 | 0.659 | 0.660 / 0.916 | 0.083 / 0.125 | 0.39 | 0.842 |
| QAT | qat_lsq_s1 | noise0.3 | weight-noise | 256 | 0.699 | 0.274 | 0.641 / 0.894 | 0.125 / 0.250 | 0.31 | 0.682 |
| QAT | qat_lsq_s1 | noise0.5 | weight-noise | 256 | 0.699 | 0.006 | 0.508 / 0.693 | 0.333 / 1.333 | 0.10 | 0.493 |
| QAT | qat_lsq_s1 | w3g32 | weight-quant | 256 | 0.699 | 0.849 | 0.672 / 0.944 | 0.083 / 0.083 | 0.46 | 0.898 |
| QAT | qat_lsq_s1 | w4c | weight-quant | 256 | 0.699 | 0.978 | 0.668 / 0.927 | 0.062 / 0.062 | 0.89 | 0.946 |
| QAT | qat_lsq_s1 | w4t | weight-quant | 256 | 0.699 | 0.950 | 0.676 / 0.950 | 0.062 / 0.062 | 0.73 | 0.932 |
| QAT | qat_lsq_s2 | noise0.1 | weight-noise | 256 | 0.688 | 0.869 | 0.656 / 0.932 | 0.062 / 0.062 | 0.91 | 0.938 |
| QAT | qat_lsq_s2 | noise0.2 | weight-noise | 256 | 0.688 | 0.676 | 0.645 / 0.903 | 0.062 / 0.083 | 0.52 | 0.844 |
| QAT | qat_lsq_s2 | noise0.3 | weight-noise | 256 | 0.688 | 0.284 | 0.625 / 0.881 | 0.167 / 0.333 | 0.22 | 0.689 |
| QAT | qat_lsq_s2 | noise0.5 | weight-noise | 256 | 0.688 | 0.023 | 0.531 / 0.767 | 0.333 / 1.000 | 0.10 | 0.509 |
| QAT | qat_lsq_s2 | w3g32 | weight-quant | 256 | 0.688 | 0.778 | 0.645 / 0.920 | 0.062 / 0.083 | 0.56 | 0.875 |
| QAT | qat_lsq_s2 | w4c | weight-quant | 256 | 0.688 | 0.886 | 0.648 / 0.926 | 0.062 / 0.062 | 0.88 | 0.926 |
| QAT | qat_lsq_s2 | w4t | weight-quant | 256 | 0.688 | 0.909 | 0.660 / 0.943 | 0.062 / 0.062 | 0.85 | 0.922 |
| TRM-MLP | sudoku_mlp | anoise0.1 | act-noise(fresh) | 256 | 0.715 | 0.995 | 0.715 / 0.973 | 0.062 / 0.062 | 0.96 | 0.924 |
| TRM-MLP | sudoku_mlp | anoise0.2 | act-noise(fresh) | 256 | 0.715 | 0.907 | 0.719 / 0.978 | 0.062 / 0.062 | 0.91 | 0.855 |
| TRM-MLP | sudoku_mlp | anoise0.3 | act-noise(fresh) | 256 | 0.715 | 0.541 | 0.695 / 0.956 | 0.062 / 0.062 | 0.87 | 0.550 |
| TRM-MLP | sudoku_mlp | +a4 | act-quant | 256 | 0.715 | 0.000 | 0.660 / 0.907 | 0.667 / 1.000 | 0.28 | 0.151 |
| TRM-MLP | sudoku_mlp | +a8 | act-quant | 256 | 0.715 | 1.038 | 0.727 / 0.989 | 0.062 / 0.062 | 0.94 | 0.888 |
| TRM-MLP | sudoku_mlp | w4c+a8 | act-quant | 256 | 0.715 | 1.011 | 0.719 / 0.984 | 0.062 / 0.062 | 0.95 | 0.861 |
| TRM-MLP | sudoku_mlp | noise0.05 | weight-noise | 256 | 0.715 | 1.016 | 0.738 / 1.000 | 0.062 / 0.062 | 0.97 | 0.924 |
| TRM-MLP | sudoku_mlp | noise0.1 | weight-noise | 256 | 0.715 | 0.962 | 0.719 / 0.978 | 0.062 / 0.062 | 0.92 | 0.879 |
| TRM-MLP | sudoku_mlp | noise0.15 | weight-noise | 256 | 0.715 | 0.377 | 0.719 / 0.984 | 0.062 / 0.062 | 0.81 | 0.530 |
| TRM-MLP | sudoku_mlp | noise0.2 | weight-noise | 256 | 0.715 | 0.131 | 0.703 / 0.956 | 0.062 / 0.062 | 0.79 | 0.473 |
| TRM-MLP | sudoku_mlp | noise0.3 | weight-noise | 256 | 0.715 | 0.027 | 0.703 / 0.956 | 0.062 / 0.062 | 0.78 | 0.326 |
| TRM-MLP | sudoku_mlp | w3a | weight-quant | 256 | 0.715 | 0.754 | 0.707 / 0.956 | 0.062 / 0.062 | 0.86 | 0.602 |
| TRM-MLP | sudoku_mlp | w3c | weight-quant | 256 | 0.715 | 0.355 | 0.703 / 0.956 | 0.062 / 0.062 | 0.83 | 0.303 |
| TRM-MLP | sudoku_mlp | w3g32 | weight-quant | 256 | 0.715 | 0.770 | 0.715 / 0.978 | 0.062 / 0.062 | 0.84 | 0.527 |
| TRM-MLP | sudoku_mlp | w4a | weight-quant | 256 | 0.715 | 1.000 | 0.727 / 0.989 | 0.062 / 0.062 | 0.97 | 0.900 |
| TRM-MLP | sudoku_mlp | w4c | weight-quant | 256 | 0.715 | 0.989 | 0.723 / 0.989 | 0.062 / 0.062 | 0.95 | 0.903 |
| TRM-MLP | sudoku_mlp | w4g32 | weight-quant | 256 | 0.715 | 1.005 | 0.730 / 0.989 | 0.062 / 0.062 | 0.95 | 0.914 |
| TRM-MLP | sudoku_mlp | w4t | weight-quant | 256 | 0.715 | 0.142 | 0.707 / 0.951 | 0.062 / 0.062 | 0.77 | 0.285 |
| TRM-MLP | sudoku_mlp | w5t | weight-quant | 256 | 0.715 | 0.880 | 0.715 / 0.973 | 0.062 / 0.062 | 0.91 | 0.854 |
| TRM-MLP | sudoku_mlp | w8c | weight-quant | 256 | 0.715 | 1.027 | 0.723 / 0.989 | 0.062 / 0.062 | 0.97 | 0.969 |
| TRM-MLP | trm_sudoku_mlp_w256_s0 | noise0.1 | weight-noise | 256 | 0.652 | 0.814 | 0.645 / 0.940 | 0.062 / 0.062 | 0.78 | 0.866 |
| TRM-MLP | trm_sudoku_mlp_w256_s0 | w4t | weight-quant | 256 | 0.652 | 0.000 | 0.000 / 0.000 | 2.000 / 2.000 | 0.00 | 0.266 |
| TRM-MLP | trm_sudoku_mlp_w256_s1 | noise0.1 | weight-noise | 256 | 0.668 | 0.871 | 0.676 / 0.977 | 0.062 / 0.083 | 0.63 | 0.908 |
| TRM-MLP | trm_sudoku_mlp_w256_s1 | w4t | weight-quant | 256 | 0.668 | 0.000 | 0.121 / 0.164 | 1.333 / 1.333 | 0.00 | 0.158 |
| TRM-MLP | trm_sudoku_mlp_w256_s2 | noise0.1 | weight-noise | 256 | 0.711 | 0.868 | 0.715 / 0.962 | 0.062 / 0.062 | 0.88 | 0.909 |
| TRM-MLP | trm_sudoku_mlp_w256_s2 | w4t | weight-quant | 256 | 0.711 | 0.000 | 0.375 / 0.505 | 1.167 / 1.333 | 0.02 | 0.215 |
| TRM-MLP | trm_sudoku_mlp_w512_s0 | noise0.1 | weight-noise | 256 | 0.883 | 0.973 | 0.926 / 0.991 | 0.062 / 0.062 | 0.94 | 0.906 |
| TRM-MLP | trm_sudoku_mlp_w512_s0 | w4t | weight-quant | 256 | 0.883 | 0.049 | 0.742 / 0.792 | 0.125 / 0.250 | 0.45 | 0.436 |
| TRM-MLP | trm_sudoku_mlp_w512_s1 | noise0.1 | weight-noise | 256 | 0.895 | 0.956 | 0.906 / 0.978 | 0.062 / 0.062 | 0.94 | 0.903 |
| TRM-MLP | trm_sudoku_mlp_w512_s1 | w4t | weight-quant | 256 | 0.895 | 0.035 | 0.832 / 0.900 | 0.062 / 0.062 | 0.78 | 0.179 |
| TRM-MLP | trm_sudoku_mlp_w512_s2 | noise0.1 | weight-noise | 256 | 0.891 | 0.969 | 0.906 / 0.974 | 0.062 / 0.062 | 0.92 | 0.934 |
| TRM-MLP | trm_sudoku_mlp_w512_s2 | w4t | weight-quant | 256 | 0.891 | 0.022 | 0.434 / 0.469 | 1.333 / 1.333 | 0.17 | 0.264 |
| TRM-MLP | trm_sudoku_mlp_w768_s0 | noise0.1 | weight-noise | 256 | 0.812 | 0.918 | 0.797 / 0.962 | 0.062 / 0.062 | 0.89 | 0.877 |
| TRM-MLP | trm_sudoku_mlp_w768_s0 | w4t | weight-quant | 256 | 0.812 | 0.091 | 0.734 / 0.880 | 0.062 / 0.062 | 0.74 | 0.395 |
| TRM-attn | maze_attn | anoise1.0 | act-noise(fresh) | 48 | 0.875 | 0.310 | 0.958 / 0.976 | 0.062 / 0.062 | 0.88 | 0.916 |
| TRM-attn | maze_attn | +a4 | act-quant | 48 | 0.875 | 0.000 | 0.917 / 0.929 | 0.062 / 0.062 | 0.79 | 0.282 |
| TRM-attn | maze_attn | noise0.1 | weight-noise | 48 | 0.875 | 1.000 | 0.979 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| TRM-attn | maze_attn | noise1.0 | weight-noise | 48 | 0.875 | 1.024 | 0.979 / 1.000 | 0.062 / 0.062 | 0.98 | 0.948 |
| TRM-attn | maze_attn | noise1.5 | weight-noise | 48 | 0.875 | 0.310 | 0.938 / 0.976 | 0.062 / 0.062 | 0.95 | 0.884 |
| TRM-attn | maze_attn | noise2.0 | weight-noise | 48 | 0.875 | 0.000 | 0.979 / 1.000 | 0.062 / 0.062 | 0.93 | 0.749 |
| TRM-attn | maze_attn | noise3.0 | weight-noise | 48 | 0.875 | 0.000 | 0.917 / 0.929 | 0.062 / 0.062 | 0.79 | -0.010 |
| TRM-attn | maze_attn | w2a | weight-quant | 48 | 0.875 | 0.881 | 0.917 / 0.976 | 0.062 / 0.062 | 0.93 | 0.875 |
| TRM-attn | maze_attn | w2g32 | weight-quant | 48 | 0.875 | 0.405 | 0.938 / 0.976 | 0.062 / 0.062 | 0.88 | 0.766 |
| TRM-attn | maze_attn | w3c | weight-quant | 48 | 0.875 | 1.048 | 0.979 / 1.000 | 0.062 / 0.062 | 1.00 | 0.991 |
| TRM-attn | maze_attn | w3t | weight-quant | 48 | 0.875 | 0.000 | 0.896 / 0.929 | 0.062 / 0.062 | 0.79 | 0.403 |
| TRM-attn | maze_attn | w4t | weight-quant | 48 | 0.875 | 1.024 | 0.958 / 1.000 | 0.062 / 0.062 | 1.00 | 0.994 |
| TRM-attn | maze_attn | w8c | weight-quant | 48 | 0.875 | 1.000 | 1.000 / 1.000 | 0.062 / 0.062 | 1.00 | 1.000 |
| TRM-attn | sudoku_attn | anoise0.3 | act-noise(fresh) | 256 | 0.727 | 1.000 | 0.758 / 0.984 | 0.250 / 0.250 | 0.01 | 0.913 |
| TRM-attn | sudoku_attn | anoise0.5 | act-noise(fresh) | 256 | 0.727 | 0.656 | 0.727 / 0.935 | 0.333 / 0.333 | 0.00 | 0.748 |
| TRM-attn | sudoku_attn | anoise0.7 | act-noise(fresh) | 256 | 0.727 | 0.016 | 0.605 / 0.780 | 0.500 / 0.667 | 0.00 | 0.450 |
| TRM-attn | sudoku_attn | +a4 | act-quant | 256 | 0.727 | 0.000 | 0.094 / 0.113 | 2.000 / 2.000 | 0.00 | 0.042 |
| TRM-attn | sudoku_attn | +a8 | act-quant | 256 | 0.727 | 1.043 | 0.758 / 0.984 | 0.167 / 0.167 | 0.03 | 0.949 |
| TRM-attn | sudoku_attn | w4c+a8 | act-quant | 256 | 0.727 | 1.059 | 0.762 / 0.989 | 0.167 / 0.167 | 0.01 | 0.942 |
| TRM-attn | sudoku_attn | noise0.1 | weight-noise | 256 | 0.727 | 1.027 | 0.758 / 0.978 | 0.167 / 0.167 | 0.12 | 0.961 |
| TRM-attn | sudoku_attn | noise0.3 | weight-noise | 256 | 0.727 | 0.957 | 0.750 / 0.978 | 0.167 / 0.167 | 0.02 | 0.914 |
| TRM-attn | sudoku_attn | noise0.5 | weight-noise | 256 | 0.727 | 0.925 | 0.742 / 0.968 | 0.167 / 0.250 | 0.01 | 0.858 |
| TRM-attn | sudoku_attn | noise0.7 | weight-noise | 256 | 0.727 | 0.747 | 0.750 / 0.978 | 0.167 / 0.250 | 0.03 | 0.723 |
| TRM-attn | sudoku_attn | noise1.0 | weight-noise | 256 | 0.727 | 0.070 | 0.523 / 0.683 | 0.333 / 1.167 | 0.01 | 0.409 |
| TRM-attn | sudoku_attn | w3a | weight-quant | 256 | 0.727 | 0.935 | 0.742 / 0.968 | 0.250 / 0.250 | 0.00 | 0.879 |
| TRM-attn | sudoku_attn | w3c | weight-quant | 256 | 0.727 | 0.876 | 0.758 / 0.978 | 0.167 / 0.250 | 0.03 | 0.835 |
| TRM-attn | sudoku_attn | w3g32 | weight-quant | 256 | 0.727 | 1.005 | 0.766 / 0.989 | 0.167 / 0.250 | 0.04 | 0.914 |
| TRM-attn | sudoku_attn | w3t | weight-quant | 256 | 0.727 | 0.000 | 0.246 / 0.323 | 2.000 / 2.000 | 0.00 | 0.017 |
| TRM-attn | sudoku_attn | w4c | weight-quant | 256 | 0.727 | 0.984 | 0.766 / 0.995 | 0.167 / 0.167 | 0.02 | 0.933 |
| TRM-attn | sudoku_attn | w4t | weight-quant | 256 | 0.727 | 0.989 | 0.766 / 0.973 | 0.250 / 0.250 | 0.01 | 0.881 |
| TRM-attn | sudoku_attn | w8c | weight-quant | 256 | 0.727 | 1.027 | 0.770 / 0.995 | 0.062 / 0.125 | 0.55 | 0.988 |
| TRM-attn | trm_sudoku_attn_w512_s0 | noise0.1 | weight-noise | 256 | 0.699 | 1.056 | 0.730 / 0.966 | 0.062 / 0.062 | 0.92 | 0.968 |
| TRM-attn | trm_sudoku_attn_w512_s0 | w4t | weight-quant | 256 | 0.699 | 1.028 | 0.727 / 0.955 | 0.062 / 0.083 | 0.59 | 0.909 |
| TRM-attn | trm_sudoku_attn_w512_s1 | noise0.1 | weight-noise | 256 | 0.684 | 0.971 | 0.676 / 0.943 | 0.062 / 0.104 | 0.67 | 0.886 |
| TRM-attn | trm_sudoku_attn_w512_s1 | w4t | weight-quant | 256 | 0.684 | 0.971 | 0.684 / 0.954 | 0.062 / 0.125 | 0.70 | 0.825 |
| TRM-attn | trm_sudoku_attn_w512_s2 | noise0.1 | weight-noise | 256 | 0.555 | 1.120 | 0.488 / 0.704 | 1.000 / inf | 0.02 | 0.838 |
| TRM-attn | trm_sudoku_attn_w512_s2 | w4t | weight-quant | 256 | 0.555 | 0.366 | 0.449 / 0.676 | 1.167 / inf | 0.00 | 0.643 |
| seeds | x5_attn512_s3 | noise0.1 | weight-noise | 256 | 0.770 | 0.970 | 0.777 / 0.954 | 0.062 / 0.062 | 0.91 | 0.961 |
| seeds | x5_attn512_s3 | noise0.5 | weight-noise | 256 | 0.770 | 0.838 | 0.746 / 0.934 | 0.083 / 0.125 | 0.39 | 0.863 |
| seeds | x5_attn512_s3 | noise0.7 | weight-noise | 256 | 0.770 | 0.492 | 0.734 / 0.919 | 0.062 / 0.125 | 0.57 | 0.683 |
| seeds | x5_attn512_s3 | w3c | weight-quant | 256 | 0.770 | 0.848 | 0.762 / 0.954 | 0.062 / 0.062 | 0.71 | 0.882 |
| seeds | x5_attn512_s3 | w3g32 | weight-quant | 256 | 0.770 | 0.949 | 0.762 / 0.949 | 0.062 / 0.062 | 0.89 | 0.947 |
| seeds | x5_attn512_s3 | w4c | weight-quant | 256 | 0.770 | 0.985 | 0.773 / 0.954 | 0.062 / 0.062 | 0.91 | 0.968 |
| seeds | x5_attn512_s3 | w4t | weight-quant | 256 | 0.770 | 0.350 | 0.719 / 0.878 | 0.125 / 0.167 | 0.44 | 0.674 |
| seeds | x5_attn512_s4 | noise0.1 | weight-noise | 256 | 0.754 | 1.021 | 0.746 / 0.953 | 0.062 / 0.062 | 0.83 | 0.965 |
| seeds | x5_attn512_s4 | noise0.5 | weight-noise | 256 | 0.754 | 0.710 | 0.715 / 0.917 | 0.250 / 0.333 | 0.10 | 0.785 |
| seeds | x5_attn512_s4 | noise0.7 | weight-noise | 256 | 0.754 | 0.321 | 0.656 / 0.829 | 0.250 / 0.333 | 0.28 | 0.520 |
| seeds | x5_attn512_s4 | w3c | weight-quant | 256 | 0.754 | 0.959 | 0.742 / 0.948 | 0.167 / 0.250 | 0.16 | 0.892 |
| seeds | x5_attn512_s4 | w3g32 | weight-quant | 256 | 0.754 | 0.969 | 0.742 / 0.943 | 0.125 / 0.125 | 0.28 | 0.923 |
| seeds | x5_attn512_s4 | w4c | weight-quant | 256 | 0.754 | 0.979 | 0.762 / 0.953 | 0.062 / 0.062 | 0.79 | 0.951 |
| seeds | x5_attn512_s4 | w4t | weight-quant | 256 | 0.754 | 0.839 | 0.742 / 0.959 | 0.062 / 0.125 | 0.51 | 0.845 |
| seeds | x5_attn512_s5 | noise0.1 | weight-noise | 256 | 0.699 | 0.972 | 0.715 / 0.955 | 0.062 / 0.062 | 0.91 | 0.965 |
| seeds | x5_attn512_s5 | noise0.5 | weight-noise | 256 | 0.699 | 0.866 | 0.691 / 0.933 | 0.062 / 0.125 | 0.56 | 0.888 |
| seeds | x5_attn512_s5 | noise0.7 | weight-noise | 256 | 0.699 | 0.531 | 0.543 / 0.737 | 0.250 / 0.500 | 0.10 | 0.693 |
| seeds | x5_attn512_s5 | w3c | weight-quant | 256 | 0.699 | 0.905 | 0.699 / 0.939 | 0.062 / 0.062 | 0.70 | 0.895 |
| seeds | x5_attn512_s5 | w3g32 | weight-quant | 256 | 0.699 | 0.939 | 0.711 / 0.966 | 0.062 / 0.083 | 0.70 | 0.948 |
| seeds | x5_attn512_s5 | w4c | weight-quant | 256 | 0.699 | 0.983 | 0.699 / 0.939 | 0.062 / 0.083 | 0.68 | 0.962 |
| seeds | x5_attn512_s5 | w4t | weight-quant | 256 | 0.699 | 0.872 | 0.699 / 0.944 | 0.062 / 0.083 | 0.69 | 0.906 |
| seeds | x5_mlp512_s3 | noise0.1 | weight-noise | 256 | 0.883 | 0.934 | 0.898 / 0.982 | 0.062 / 0.062 | 0.93 | 0.947 |
| seeds | x5_mlp512_s3 | noise0.15 | weight-noise | 256 | 0.883 | 0.593 | 0.875 / 0.973 | 0.062 / 0.062 | 0.65 | 0.716 |
| seeds | x5_mlp512_s3 | w3g32 | weight-quant | 256 | 0.883 | 0.739 | 0.863 / 0.956 | 0.062 / 0.062 | 0.86 | 0.816 |
| seeds | x5_mlp512_s3 | w4c | weight-quant | 256 | 0.883 | 0.965 | 0.887 / 0.982 | 0.062 / 0.062 | 0.95 | 0.949 |
| seeds | x5_mlp512_s3 | w4t | weight-quant | 256 | 0.883 | 0.053 | 0.469 / 0.513 | 1.167 / 1.333 | 0.39 | 0.579 |
| seeds | x5_mlp512_s4 | noise0.1 | weight-noise | 256 | 0.863 | 0.977 | 0.895 / 0.995 | 0.062 / 0.062 | 0.98 | 0.946 |
| seeds | x5_mlp512_s4 | noise0.15 | weight-noise | 256 | 0.863 | 0.290 | 0.848 / 0.959 | 0.062 / 0.062 | 0.66 | 0.675 |
| seeds | x5_mlp512_s4 | w3g32 | weight-quant | 256 | 0.863 | 0.900 | 0.887 / 0.982 | 0.062 / 0.062 | 0.92 | 0.893 |
| seeds | x5_mlp512_s4 | w4c | weight-quant | 256 | 0.863 | 1.027 | 0.910 / 1.000 | 0.062 / 0.062 | 0.98 | 0.952 |
| seeds | x5_mlp512_s4 | w4t | weight-quant | 256 | 0.863 | 0.095 | 0.797 / 0.891 | 0.062 / 0.125 | 0.51 | 0.550 |
| seeds | x5_mlp512_s5 | noise0.1 | weight-noise | 256 | 0.891 | 0.934 | 0.891 / 0.978 | 0.062 / 0.062 | 0.91 | 0.938 |
| seeds | x5_mlp512_s5 | noise0.15 | weight-noise | 256 | 0.891 | 0.434 | 0.883 / 0.969 | 0.062 / 0.083 | 0.55 | 0.678 |
| seeds | x5_mlp512_s5 | w3g32 | weight-quant | 256 | 0.891 | 0.899 | 0.895 / 0.982 | 0.062 / 0.062 | 0.93 | 0.892 |
| seeds | x5_mlp512_s5 | w4c | weight-quant | 256 | 0.891 | 0.978 | 0.895 / 0.982 | 0.062 / 0.062 | 0.96 | 0.960 |
| seeds | x5_mlp512_s5 | w4t | weight-quant | 256 | 0.891 | 0.140 | 0.738 / 0.829 | 0.667 / 0.667 | 0.23 | 0.514 |

## P2: polish gain ~ return rate x gap on returned examples

Matched cells (compressed per-example correctness agrees with M1's >= 98%): 20; within max(2 pts, 25%): 1.000 -> **SUPPORTED**

| family | model | cell | M1 record | c16 match | fp32-cmp gap | return t=1 | median rho | predicted gain | observed gain (c16.P8 - c16) | within rule |
|---|---|---|---|---|---|---|---|---|---|---|
| HRM | hrm_maze | w3t | p_hrm_maze_E2a.json | 1.000 | 0.104 | 0.875 | 0.062 | 0.062 | 0.062 | yes |
| HRM | hrm_maze | w2a | p_hrm_maze_E2a.json | 1.000 | 0.042 | 0.833 | 0.062 | 0.021 | 0.021 | yes |
| HRM | hrm_maze | noise1.5 | p_hrm_maze_E2b.json | 1.000 | 0.292 | 0.792 | 0.083 | 0.188 | 0.188 | yes |
| HRM | hrm_sudoku | w4t | p_hrm_sudoku_E1.json | 1.000 | -0.008 | 0.473 | 0.062 | 0.020 | 0.023 | yes |
| HRM | hrm_sudoku | w3t | p_hrm_sudoku_E1.json | 1.000 | 0.477 | 0.000 | 2.000 | 0.000 | 0.000 | yes |
| HRM | hrm_sudoku | w2a | p_hrm_sudoku_E1.json | 1.000 | 0.348 | 0.422 | 0.250 | 0.293 | 0.277 | yes |
| HRM | hrm_sudoku | noise0.7 | p_hrm_sudoku_E1.json | 1.000 | 0.105 | 0.414 | 0.125 | 0.066 | 0.082 | yes |
| HRM | hrm_sudoku | noise1.0 | p_hrm_sudoku_E1.json | 1.000 | 0.207 | 0.434 | 0.250 | 0.176 | 0.168 | yes |
| HRM | hrm_sudoku | noise1.5 | p_hrm_sudoku_E1.json | 1.000 | 0.477 | 0.121 | 2.000 | 0.117 | 0.137 | yes |
| TRM-attn | maze_attn | w3t | p_maze_attn_E3a.json | 1.000 | 0.875 | 0.896 | 0.062 | 0.812 | 0.833 | yes |
| TRM-attn | maze_attn | w2g32 | p_maze_attn_E3a.json | 1.000 | 0.521 | 0.938 | 0.062 | 0.500 | 0.521 | yes |
| TRM-attn | maze_attn | w2a | p_maze_attn_E3a.json | 1.000 | 0.104 | 0.917 | 0.062 | 0.125 | 0.125 | yes |
| TRM-attn | maze_attn | noise1.5 | p_maze_attn_E3b.json | 1.000 | 0.604 | 0.938 | 0.062 | 0.583 | 0.646 | yes |
| TRM-attn | sudoku_attn | w4t | p_sudoku_attn_B1.json | 0.965 | 0.008 | 0.766 | 0.250 | 0.027 | 0.078 | no (unmatched) |
| TRM-attn | sudoku_attn | w3g32 | p_sudoku_attn_B1.json | 0.934 | -0.004 | 0.766 | 0.167 | 0.023 | 0.047 | no (unmatched) |
| TRM-attn | sudoku_attn | w3c | p_sudoku_attn_B1.json | 0.914 | 0.090 | 0.758 | 0.167 | 0.113 | 0.113 | yes (unmatched) |
| TRM-attn | sudoku_attn | w3a | p_sudoku_attn_B1.json | 0.949 | 0.047 | 0.742 | 0.250 | 0.062 | 0.074 | yes (unmatched) |
| TRM-attn | sudoku_attn | w3t | p_sudoku_attn_B1.json | 1.000 | 0.727 | 0.246 | 2.000 | 0.234 | 0.203 | yes |
| TRM-attn | sudoku_attn | noise0.7 | p_sudoku_attn_B3.json | 0.965 | 0.184 | 0.750 | 0.167 | 0.227 | 0.230 | yes (unmatched) |
| TRM-attn | sudoku_attn | noise1.0 | p_sudoku_attn_B3.json | 1.000 | 0.676 | 0.523 | 0.333 | 0.445 | 0.430 | yes |
| TRM-attn | sudoku_attn | w4c | p_sudoku_attn_B4.json | 0.961 | 0.012 | 0.766 | 0.167 | 0.020 | 0.035 | yes (unmatched) |
| TRM-MLP | sudoku_mlp | noise0.15 | p_sudoku_mlp_A3.json | 0.984 | 0.445 | 0.719 | 0.062 | 0.445 | 0.426 | yes |
| TRM-MLP | sudoku_mlp | noise0.2 | p_sudoku_mlp_A3.json | 1.000 | 0.621 | 0.703 | 0.062 | 0.590 | 0.629 | yes |
| TRM-MLP | sudoku_mlp | noise0.3 | p_sudoku_mlp_A3.json | 1.000 | 0.695 | 0.703 | 0.062 | 0.664 | 0.656 | yes |
| TRM-MLP | sudoku_mlp | w4c | p_sudoku_mlp_A4.json | 0.984 | 0.008 | 0.723 | 0.062 | 0.023 | 0.027 | yes |
| TRM-MLP | sudoku_mlp | w4a | p_sudoku_mlp_A4.json | 0.984 | 0.000 | 0.727 | 0.062 | 0.027 | 0.047 | yes |
| TRM-MLP | sudoku_mlp | w4g32 | p_sudoku_mlp_A4.json | 0.965 | -0.004 | 0.730 | 0.062 | 0.016 | 0.035 | yes (unmatched) |
| MDEQ | mdeq_small | w4c | deq_small_dev.json:26+26 | 0.511 | 0.166 | 0.994 | 0.062 | 0.168 | 0.154 | yes (unmatched) |
| MDEQ | mdeq_small | w4g32 | deq_small_dev.json:26+26 | 0.569 | 0.068 | 0.997 | 0.062 | 0.068 | 0.069 | yes (unmatched) |
| MDEQ | mdeq_small | w4a | deq_small_dev.json:26+26 | 0.565 | 0.065 | 0.998 | 0.062 | 0.065 | 0.065 | yes (unmatched) |
| MDEQ | mdeq_xl | w4g32 | deq_xl_dense_dev.json:26+16 | 0.640 | 0.010 | 0.991 | 0.062 | 0.008 | 0.006 | yes (unmatched) |
| MDEQ | mdeq_xl | w4c | deq_xl_dense_dev.json:26+16 | 0.587 | 0.051 | 0.987 | 0.062 | 0.047 | 0.041 | yes (unmatched) |

## P3: sampling gain peaks at intermediate return rates; the head picks draws with smaller rho

**Verdict: FALSIFIED** (head picks larger t_c in every model: False; gain peaks at intermediate return rate: False).

| model | draws | n | picked t_c > / < median-over-draws | sign-test p | mean t_c picked / all | acc RTN / best-Q | agreement with M2 per-example correctness |
|---|---|---|---|---|---|---|---|
| hrm_sudoku w3c | 16 | 256 | 57 / 18 | 0.0000 | 4.80 / 4.20 | 0.445 / 0.754 | 0.839 |
| sudoku_attn w3c | 16 | 256 | 64 / 67 | 0.8614 | 3.70 / 4.00 | 0.609 / 0.918 | 0.789 |
| sudoku_mlp w4c | 16 | 256 | 15 / 1 | 0.0005 | 11.70 / 11.02 | 0.703 / 0.922 | 0.860 |

| task | spec | RTN acc (M2 / mine) | best-Q@16 gain (M2) | RTN return rate (all / fp32-correct) | median rho | retained |
|---|---|---|---|---|---|---|
| hrm_sudoku | w2g32 | 0.141 / 0.141 | 0.441 | 0.410 / 0.828 | 0.333 | 0.295 |
| hrm_sudoku | w3c | 0.445 / 0.445 | 0.309 | 0.461 / 0.910 | 0.062 | 0.934 |
| hrm_sudoku | w4t | 0.484 / 0.484 | 0.262 | 0.473 / 0.951 | 0.062 | 1.016 |
| hrm_sudoku | w4c | 0.484 / 0.480 | 0.184 | 0.477 / 0.943 | 0.062 | 1.008 |
| sudoku_mlp | w3a | 0.539 / 0.539 | 0.254 | 0.707 / 0.956 | 0.062 | 0.754 |
| sudoku_mlp | w3g32 | 0.547 / 0.551 | 0.293 | 0.715 / 0.978 | 0.062 | 0.770 |
| sudoku_mlp | w4c | 0.703 / 0.707 | 0.223 | 0.723 / 0.989 | 0.062 | 0.989 |
| sudoku_mlp | w4g32 | 0.715 / 0.719 | 0.168 | 0.730 / 0.989 | 0.062 | 1.005 |
| sudoku_attn | w3a | 0.691 / 0.680 | 0.234 | 0.742 / 0.968 | 0.250 | 0.935 |
| sudoku_attn | w3c | 0.617 / 0.637 | 0.320 | 0.758 / 0.978 | 0.167 | 0.876 |
| sudoku_attn | w3g32 | 0.719 / 0.730 | 0.188 | 0.766 / 0.989 | 0.167 | 1.005 |
| sudoku_attn | w4c | 0.742 / 0.715 | 0.148 | 0.766 / 0.995 | 0.167 | 0.984 |
| sudoku_attn | w4t | 0.711 / 0.719 | 0.227 | 0.766 / 0.973 | 0.250 | 0.989 |
| maze_attn | w4t | 0.896 / 0.896 | 0.000 | 0.958 / 1.000 | 0.062 | 1.024 |
| maze_attn | w3c | 0.917 / 0.917 | 0.000 | 0.979 / 1.000 | 0.062 | 1.048 |

## P4: sigma_1/2 from the radius along the compression direction removes the mixer bias

Fit set 15 models (fp32 >= 30%), new seeds 6. Radius prediction sigma_pred = c x 0.1 x median t_c (fp32-correct examples, noise0.1). Pooled c = 0.272; R^2 (log) -0.84.

| law | bias pred/meas MLP / attn / HRM | new seeds median / worst |
|---|---|---|
| S (from x2_smix.json) | 1.37 / 0.64 / 0.89 | x1.39 / x1.61 |
| S_mix (from x2_smix.json) | 1.19 / 0.76 / 0.98 | x1.16 / x1.29 |
| radius, 1 parameter | 3.10 / 0.25 / 0.34 | x2.02 / x2.96 |
| radius, parameter-free (c = 1) | 11.39 / 0.93 / 1.23 | - |

**Verdict: FALSIFIED** (rule: 1-parameter bias within [0.9, 1.1] for MLP and attention, new-seed median <= x1.16).

| model | mixer | held out | sigma_1/2 measured | median t_c at noise0.1 | frac censored (t_c = 16) | radius pred (c=1) | S | S_mix |
|---|---|---|---|---|---|---|---|---|
| sudoku_mlp | MLP | False | 0.141 | 16.00 | 0.92 | 1.600 | 1.74 | 1.53 |
| sudoku_attn | attn | False | 0.774 | 6.00 | 0.12 | 0.600 | 0.69 | 0.44 |
| hrm_sudoku | HRM | False | 0.983 | 16.00 | 0.93 | 1.600 | 0.35 | 0.25 |
| maze_attn | attn | False | 1.683 | 16.00 | 1.00 | 1.600 | 0.32 | 0.18 |
| hrm_maze | HRM | False | 1.716 | 16.00 | 0.97 | 1.600 | 0.23 | 0.16 |
| trm_sudoku_attn_w512_s0 | attn | False | 0.737 | 16.00 | 0.92 | 1.600 | 0.73 | 0.48 |
| trm_sudoku_attn_w512_s1 | attn | False | 0.642 | 16.00 | 0.67 | 1.600 | 0.80 | 0.53 |
| trm_sudoku_attn_w512_s2 | attn | False | 0.576 | 1.00 | 0.02 | 0.100 | 0.83 | 0.59 |
| trm_sudoku_mlp_w256_s0 | MLP | False | 0.130 | 16.00 | 0.78 | 1.600 | 1.47 | 1.05 |
| trm_sudoku_mlp_w256_s1 | MLP | False | 0.133 | 16.00 | 0.63 | 1.600 | 2.05 | 1.86 |
| trm_sudoku_mlp_w256_s2 | MLP | False | 0.130 | 16.00 | 0.88 | 1.600 | 1.29 | 1.16 |
| trm_sudoku_mlp_w512_s0 | MLP | False | 0.163 | 16.00 | 0.94 | 1.600 | 1.50 | 1.35 |
| trm_sudoku_mlp_w512_s1 | MLP | False | 0.159 | 16.00 | 0.94 | 1.600 | 1.97 | 1.78 |
| trm_sudoku_mlp_w512_s2 | MLP | False | 0.126 | 16.00 | 0.92 | 1.600 | 1.61 | 1.42 |
| trm_sudoku_mlp_w768_s0 | MLP | False | 0.147 | 16.00 | 0.89 | 1.600 | 2.48 | 2.43 |
| x5_attn512_s3 | attn | True | 0.709 | 16.00 | 0.91 | 1.600 | 0.75 | 0.40 |
| x5_attn512_s4 | attn | True | 0.578 | 16.00 | 0.83 | 1.600 | 0.72 | 0.34 |
| x5_attn512_s5 | attn | True | 0.723 | 16.00 | 0.91 | 1.600 | 0.67 | 0.42 |
| x5_mlp512_s3 | MLP | True | 0.176 | 16.00 | 0.93 | 1.600 | 1.50 | 1.32 |
| x5_mlp512_s4 | MLP | True | 0.158 | 16.00 | 0.98 | 1.600 | 1.42 | 1.32 |
| x5_mlp512_s5 | MLP | True | 0.147 | 16.00 | 0.91 | 1.600 | 1.71 | 1.52 |

