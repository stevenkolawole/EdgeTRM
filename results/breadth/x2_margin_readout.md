# X2 contraction loss (A) vs readout margin (B): read-out (Babel; `python x2_margin_analyze.py`)

Rules: `x2_margin_predictions.md` (02:47 UTC, before any margin record). Cells 202 (weight/act quant, weight noise; fp32 >= 10%), collapsed 60.

| test | value | rule |
|---|---|---|
| A: collapsed cells with late contraction ratio r >= 1 | 0.117 | >= 0.80 |
| A: held cells with r < 1 | 0.908 | >= 0.80 |
| B: collapsed cells with r < 1 | 0.883 | >= 0.80 |
| B: zero-parameter width sigma_0 x median t_r(sigma_0), new seeds: bias MLP / attn | 8.25 / 9.60 | both in [0.9, 1.1] |
| B: new seeds median / worst ratio (zero-parameter) | x8.94 / x11.07 | < S x1.39 and S_mix x1.16 |

**A: PARTLY. B: PARTLY** (clause 1 True, clause 2 False).

Pooled AUROC for collapse: r 0.690, compressed residual 0.852, -t_r 0.990.

Fit set (15): zero-parameter bias {"HRM": 4.93, "MLP": 10.37, "attn": 2.32}; 1-parameter scale 0.175, bias {"HRM": 0.86, "MLP": 1.82, "attn": 0.41}; new seeds with that scale x1.57, bias {"MLP": 1.45, "attn": 1.68}. S fit bias {"MLP": 1.37, "attn": 0.64, "HRM": 0.89}, S_mix {"MLP": 1.19, "attn": 0.76, "HRM": 0.98}.

| model | mixer | held out | sigma_1/2 | median t_r at sigma 0.05 (censored frac) | pred 0.05 x t_r | disp at 0.05 | r at 0.05 |
|---|---|---|---|---|---|---|---|
| sudoku_mlp | MLP | False | 0.141 | 48.00 (0.00) | 2.400 | 0.2759 | 0.982 |
| sudoku_attn | attn | False | 0.774 | 48.00 (0.03) | 2.400 | 0.1676 | 0.940 |
| hrm_sudoku | HRM | False | 0.983 | 128.00 (0.13) | 6.400 | 0.0212 | 0.779 |
| maze_attn | attn | False | 1.683 | 128.00 (0.40) | 6.400 | 0.0171 | 0.749 |
| hrm_maze | HRM | False | 1.716 | inf (0.72) | 6.400 | 0.0121 | 0.663 |
| trm_sudoku_attn_w512_s0 | attn | False | 0.737 | 128.00 (0.03) | 6.400 | 0.0447 | 0.790 |
| trm_sudoku_attn_w512_s1 | attn | False | 0.642 | 24.00 (0.00) | 1.200 | 0.3596 | 0.873 |
| trm_sudoku_attn_w512_s2 | attn | False | 0.576 | 4.00 (0.00) | 0.200 | 0.5359 | 0.944 |
| trm_sudoku_mlp_w256_s0 | MLP | False | 0.130 | 24.00 (0.00) | 1.200 | 0.1540 | 0.884 |
| trm_sudoku_mlp_w256_s1 | MLP | False | 0.133 | 24.00 (0.00) | 1.200 | 0.1154 | 0.768 |
| trm_sudoku_mlp_w256_s2 | MLP | False | 0.130 | 48.00 (0.00) | 2.400 | 0.0813 | 0.561 |
| trm_sudoku_mlp_w512_s0 | MLP | False | 0.163 | 32.00 (0.00) | 1.600 | 0.2142 | 0.904 |
| trm_sudoku_mlp_w512_s1 | MLP | False | 0.159 | 16.00 (0.00) | 0.800 | 0.1837 | 0.933 |
| trm_sudoku_mlp_w512_s2 | MLP | False | 0.126 | 24.00 (0.00) | 1.200 | 0.1377 | 0.987 |
| trm_sudoku_mlp_w768_s0 | MLP | False | 0.147 | 32.00 (0.00) | 1.600 | 0.2350 | 0.935 |
| x5_attn512_s3 | attn | True | 0.709 | inf (0.73) | 6.400 | 0.0412 | 0.663 |
| x5_attn512_s4 | attn | True | 0.578 | 128.00 (0.45) | 6.400 | 0.0379 | 0.851 |
| x5_attn512_s5 | attn | True | 0.723 | inf (0.87) | 6.400 | 0.0344 | 0.516 |
| x5_mlp512_s3 | MLP | True | 0.176 | 32.00 (0.00) | 1.600 | 0.2164 | 0.886 |
| x5_mlp512_s4 | MLP | True | 0.158 | 24.00 (0.00) | 1.200 | 0.1915 | 0.914 |
| x5_mlp512_s5 | MLP | True | 0.147 | 24.00 (0.00) | 1.200 | 0.0961 | 0.929 |

| model | family | fp32 | fp32 late ratio L_0 | fp32 residual |
|---|---|---|---|---|
| dt_maze13 | DT | 1.000 | 1.000 | 1.25e-01 |
| dt_maze33 | DT | 1.000 | 1.000 | 1.07e-01 |
| ft_fp32_s0 | QAT | 0.664 | 0.969 | 3.53e-02 |
| hrm_maze | HRM | 0.833 | 0.624 | 0.00e+00 |
| hrm_sudoku | HRM | 0.477 | 0.773 | 4.70e-03 |
| maze_attn | TRM-attn | 0.875 | 0.669 | 1.80e-05 |
| qat_lsq_s0 | QAT | 0.578 | 1.000 | 6.28e-02 |
| qat_lsq_s1 | QAT | 0.699 | 0.637 | 5.10e-05 |
| qat_lsq_s2 | QAT | 0.688 | 0.676 | 7.40e-05 |
| qat_naive_s0 | QAT | 0.000 | 1.000 | 3.70e-01 |
| sudoku_attn | TRM-attn | 0.727 | 0.926 | 4.90e-02 |
| sudoku_mlp | TRM-MLP | 0.715 | 0.981 | 4.48e-02 |
| trm_sudoku_attn_w512_s0 | TRM-attn | 0.699 | 0.764 | 1.72e-03 |
| trm_sudoku_attn_w512_s1 | TRM-attn | 0.684 | 0.855 | 4.71e-02 |
| trm_sudoku_attn_w512_s2 | TRM-attn | 0.555 | 0.995 | 3.42e-01 |
| trm_sudoku_mlp_w256_s0 | TRM-MLP | 0.652 | 0.951 | 1.27e-02 |
| trm_sudoku_mlp_w256_s1 | TRM-MLP | 0.668 | 0.760 | 4.32e-04 |
| trm_sudoku_mlp_w256_s2 | TRM-MLP | 0.711 | 0.596 | 1.19e-04 |
| trm_sudoku_mlp_w512_s0 | TRM-MLP | 0.883 | 0.862 | 7.26e-03 |
| trm_sudoku_mlp_w512_s1 | TRM-MLP | 0.895 | 0.959 | 1.57e-02 |
| trm_sudoku_mlp_w512_s2 | TRM-MLP | 0.891 | 0.984 | 3.23e-02 |
| trm_sudoku_mlp_w768_s0 | TRM-MLP | 0.812 | 0.851 | 7.12e-03 |
| x5_attn512_s3 | seeds | 0.770 | 0.629 | 2.70e-04 |
| x5_attn512_s4 | seeds | 0.754 | 0.692 | 1.00e-06 |
| x5_attn512_s5 | seeds | 0.699 | 0.510 | 2.00e-06 |
| x5_mlp512_s3 | seeds | 0.883 | 0.893 | 3.89e-03 |
| x5_mlp512_s4 | seeds | 0.863 | 0.958 | 4.01e-02 |
| x5_mlp512_s5 | seeds | 0.891 | 0.933 | 2.40e-03 |

| family | model | cell | retained | r | residual | t_r | disp |
|---|---|---|---|---|---|---|---|
| DT | dt_maze13 | noise0.5 | 0.008 | 0.992 | 0.0744 | 0.30 | 0.8813 |
| DT | dt_maze13 | noise0.3 | 0.562 | 1.000 | 0.3039 | 1.50 | 0.3819 |
| DT | dt_maze13 | w4t | 0.977 | 1.000 | 0.0825 | 4.00 | 0.1497 |
| DT | dt_maze13 | w3a | 0.992 | 1.000 | 0.0719 | 4.00 | 0.2008 |
| DT | dt_maze13 | noise0.05 | 1.000 | 1.000 | 0.1167 | 12.00 | 0.0580 |
| DT | dt_maze13 | w8c | 1.000 | 1.000 | 0.1269 | 128.00 | 0.0060 |
| DT | dt_maze13 | w4c | 1.000 | 1.000 | 0.1183 | 8.00 | 0.1173 |
| DT | dt_maze13 | noise0.1 | 1.000 | 1.000 | 0.1475 | 6.00 | 0.1152 |
| DT | dt_maze13 | noise0.2 | 1.000 | 0.999 | 0.0945 | 4.00 | 0.2058 |
| DT | dt_maze33 | noise0.5 | 0.000 | 0.999 | 0.0778 | 0.30 | 0.9461 |
| DT | dt_maze33 | noise0.3 | 0.227 | 1.000 | 0.3054 | 1.00 | 0.3859 |
| DT | dt_maze33 | w4t | 0.859 | 1.000 | 0.0580 | 3.00 | 0.1500 |
| DT | dt_maze33 | w3a | 0.969 | 1.000 | 0.0695 | 3.00 | 0.1996 |
| DT | dt_maze33 | w4c | 0.992 | 1.000 | 0.1061 | 6.00 | 0.1233 |
| DT | dt_maze33 | noise0.05 | 1.000 | 1.000 | 0.1040 | 6.00 | 0.0736 |
| DT | dt_maze33 | w8c | 1.000 | 1.000 | 0.1105 | 48.00 | 0.0087 |
| DT | dt_maze33 | noise0.1 | 1.000 | 1.000 | 0.1232 | 3.00 | 0.1250 |
| DT | dt_maze33 | noise0.2 | 1.000 | 1.000 | 0.0756 | 3.00 | 0.2065 |
| HRM | hrm_maze | noise3.0 | 0.000 | 0.773 | 0.1335 | 0.70 | 1.1594 |
| HRM | hrm_maze | noise2.0 | 0.150 | 0.942 | 0.0228 | 1.00 | 0.6331 |
| HRM | hrm_maze | noise1.5 | 0.650 | 0.544 | 0.0000 | 3.00 | 0.3799 |
| HRM | hrm_maze | w3t | 0.875 | 0.572 | 0.0000 | 4.00 | 0.3704 |
| HRM | hrm_maze | w2a | 0.950 | 0.703 | 0.0000 | 6.00 | 0.2557 |
| HRM | hrm_maze | noise0.05 | 1.000 | 0.663 | 0.0000 | inf | 0.0121 |
| HRM | hrm_maze | w8c | 1.000 | 0.609 | 0.0000 | inf | 0.0021 |
| HRM | hrm_maze | w4t | 1.000 | 0.628 | 0.0000 | 16.00 | 0.0763 |
| HRM | hrm_maze | noise0.1 | 1.000 | 0.701 | 0.0000 | inf | 0.0230 |
| HRM | hrm_sudoku | w3t | 0.000 | 0.929 | 0.0901 | 0.70 | 1.1505 |
| HRM | hrm_sudoku | noise1.5 | 0.000 | 0.917 | 0.1021 | 0.70 | 0.9860 |
| HRM | hrm_sudoku | +a4 | 0.000 | 1.000 | 0.2820 | 0.70 | 1.1769 |
| HRM | hrm_sudoku | w2a | 0.270 | 0.904 | 0.0426 | 0.70 | 0.8928 |
| HRM | hrm_sudoku | w2g32 | 0.295 | 0.934 | 0.1230 | 0.70 | 0.8854 |
| HRM | hrm_sudoku | noise1.0 | 0.566 | 0.900 | 0.0226 | 2.50 | 0.5226 |
| HRM | hrm_sudoku | noise0.7 | 0.779 | 0.799 | 0.0069 | 6.00 | 0.3056 |
| HRM | hrm_sudoku | noise0.5 | 0.926 | 0.803 | 0.0063 | 12.00 | 0.2149 |
| HRM | hrm_sudoku | w3c | 0.934 | 0.790 | 0.0051 | 12.00 | 0.1823 |
| HRM | hrm_sudoku | w4c | 1.008 | 0.777 | 0.0064 | 32.00 | 0.0734 |
| HRM | hrm_sudoku | w4t | 1.016 | 0.781 | 0.0061 | 12.00 | 0.1628 |
| HRM | hrm_sudoku | noise0.05 | 1.033 | 0.779 | 0.0059 | 128.00 | 0.0212 |
| HRM | hrm_sudoku | w8c | 1.033 | 0.775 | 0.0047 | inf | 0.0041 |
| HRM | hrm_sudoku | noise0.1 | 1.033 | 0.779 | 0.0049 | 64.00 | 0.0424 |
| HRM | hrm_sudoku | +a8 | 1.057 | 0.961 | 0.0187 | 32.00 | 0.0555 |
| QAT | ft_fp32_s0 | w4t | 0.000 | 0.982 | 0.1396 | 0.70 | 1.1688 |
| QAT | ft_fp32_s0 | noise0.5 | 0.000 | 0.901 | 0.0774 | 0.70 | 1.1828 |
| QAT | ft_fp32_s0 | noise0.3 | 0.012 | 0.877 | 0.0233 | 0.70 | 1.0811 |
| QAT | ft_fp32_s0 | noise0.2 | 0.112 | 0.933 | 0.0400 | 0.70 | 1.0315 |
| QAT | ft_fp32_s0 | w3g32 | 0.500 | 0.939 | 0.0364 | 0.70 | 0.9157 |
| QAT | ft_fp32_s0 | noise0.1 | 0.847 | 0.994 | 0.0502 | 12.00 | 0.2138 |
| QAT | ft_fp32_s0 | w4c | 0.888 | 0.999 | 0.0507 | 24.00 | 0.2198 |
| QAT | ft_fp32_s0 | noise0.05 | 0.924 | 0.928 | 0.0073 | 24.00 | 0.1500 |
| QAT | qat_lsq_s0 | noise0.5 | 0.007 | 0.899 | 0.0381 | 0.70 | 1.0579 |
| QAT | qat_lsq_s0 | noise0.3 | 0.365 | 0.929 | 0.0211 | 0.70 | 0.9960 |
| QAT | qat_lsq_s0 | w3g32 | 0.520 | 0.928 | 0.0121 | 2.25 | 0.5922 |
| QAT | qat_lsq_s0 | noise0.2 | 0.676 | 1.000 | 0.0594 | 4.00 | 0.2294 |
| QAT | qat_lsq_s0 | noise0.1 | 1.014 | 1.001 | 0.0731 | 8.00 | 0.1273 |
| QAT | qat_lsq_s0 | noise0.05 | 1.020 | 1.000 | 0.0655 | 12.00 | 0.0840 |
| QAT | qat_lsq_s0 | w4c | 1.020 | 1.000 | 0.0616 | 8.00 | 0.1448 |
| QAT | qat_lsq_s0 | w4t | 1.135 | 0.969 | 0.0167 | 12.00 | 0.2429 |
| QAT | qat_lsq_s1 | noise0.5 | 0.006 | 0.991 | 0.1619 | 0.70 | 1.0311 |
| QAT | qat_lsq_s1 | noise0.3 | 0.274 | 0.914 | 0.0504 | 0.70 | 0.9387 |
| QAT | qat_lsq_s1 | noise0.2 | 0.659 | 0.770 | 0.0008 | 6.00 | 0.2826 |
| QAT | qat_lsq_s1 | w3g32 | 0.849 | 0.625 | 0.0000 | 12.00 | 0.2375 |
| QAT | qat_lsq_s1 | noise0.1 | 0.939 | 0.682 | 0.0001 | 16.00 | 0.1373 |
| QAT | qat_lsq_s1 | w4t | 0.950 | 0.993 | 0.0640 | 16.00 | 0.2045 |
| QAT | qat_lsq_s1 | w4c | 0.978 | 0.857 | 0.0043 | 24.00 | 0.1467 |
| QAT | qat_lsq_s1 | noise0.05 | 0.989 | 0.626 | 0.0000 | 48.00 | 0.0645 |
| QAT | qat_lsq_s2 | noise0.5 | 0.023 | 0.991 | 0.0840 | 0.70 | 1.0122 |
| QAT | qat_lsq_s2 | noise0.3 | 0.284 | 0.978 | 0.0507 | 0.70 | 0.9236 |
| QAT | qat_lsq_s2 | noise0.2 | 0.676 | 0.823 | 0.0013 | 8.00 | 0.2373 |
| QAT | qat_lsq_s2 | w3g32 | 0.778 | 0.698 | 0.0001 | 8.00 | 0.2264 |
| QAT | qat_lsq_s2 | noise0.1 | 0.869 | 0.782 | 0.0005 | 24.00 | 0.1209 |
| QAT | qat_lsq_s2 | w4c | 0.886 | 0.657 | 0.0000 | 24.00 | 0.1298 |
| QAT | qat_lsq_s2 | w4t | 0.909 | 0.984 | 0.0023 | 16.00 | 0.1913 |
| QAT | qat_lsq_s2 | noise0.05 | 0.977 | 0.723 | 0.0002 | 48.00 | 0.0544 |
| TRM-MLP | sudoku_mlp | +a4 | 0.000 | 1.001 | 0.5086 | 1.00 | 1.2425 |
| TRM-MLP | sudoku_mlp | noise0.3 | 0.027 | 1.000 | 0.2053 | 0.70 | 1.1599 |
| TRM-MLP | sudoku_mlp | noise0.2 | 0.131 | 0.997 | 0.2694 | 0.70 | 1.0445 |
| TRM-MLP | sudoku_mlp | w4t | 0.142 | 0.999 | 0.2277 | 1.00 | 1.1753 |
| TRM-MLP | sudoku_mlp | w3c | 0.355 | 0.998 | 0.2864 | 0.70 | 1.1514 |
| TRM-MLP | sudoku_mlp | noise0.15 | 0.377 | 1.001 | 0.3449 | 1.00 | 0.9940 |
| TRM-MLP | sudoku_mlp | w3a | 0.754 | 1.002 | 0.1362 | 3.00 | 0.7859 |
| TRM-MLP | sudoku_mlp | w3g32 | 0.770 | 1.002 | 0.1588 | 1.50 | 0.9457 |
| TRM-MLP | sudoku_mlp | w5t | 0.880 | 0.989 | 0.0344 | 16.00 | 0.3591 |
| TRM-MLP | sudoku_mlp | noise0.1 | 0.962 | 0.983 | 0.0316 | 16.00 | 0.3520 |
| TRM-MLP | sudoku_mlp | w4c | 0.989 | 0.970 | 0.0288 | 24.00 | 0.3188 |
| TRM-MLP | sudoku_mlp | w4a | 1.000 | 0.958 | 0.0467 | 24.00 | 0.3186 |
| TRM-MLP | sudoku_mlp | w4g32 | 1.005 | 0.982 | 0.0413 | 48.00 | 0.2832 |
| TRM-MLP | sudoku_mlp | w4c+a8 | 1.011 | 0.996 | 0.0936 | 16.00 | 0.3887 |
| TRM-MLP | sudoku_mlp | noise0.05 | 1.016 | 0.982 | 0.0493 | 48.00 | 0.2759 |
| TRM-MLP | sudoku_mlp | noise0.05 | 1.016 | 0.982 | 0.0493 | 48.00 | 0.2759 |
| TRM-MLP | sudoku_mlp | w8c | 1.027 | 0.981 | 0.0452 | inf | 0.1444 |
| TRM-MLP | sudoku_mlp | +a8 | 1.038 | 1.000 | 0.1004 | 24.00 | 0.3379 |
| TRM-MLP | trm_sudoku_mlp_w256_s0 | w4t | 0.000 | 0.711 | 0.0094 | 0.50 | 1.1713 |
| TRM-MLP | trm_sudoku_mlp_w256_s0 | noise0.1 | 0.814 | 0.962 | 0.0204 | 12.00 | 0.2047 |
| TRM-MLP | trm_sudoku_mlp_w256_s0 | noise0.05 | 0.916 | 0.884 | 0.0064 | 24.00 | 0.1540 |
| TRM-MLP | trm_sudoku_mlp_w256_s1 | w4t | 0.000 | 0.985 | 0.3553 | 1.00 | 1.2527 |
| TRM-MLP | trm_sudoku_mlp_w256_s1 | noise0.1 | 0.871 | 0.883 | 0.0055 | 12.00 | 0.2549 |
| TRM-MLP | trm_sudoku_mlp_w256_s1 | noise0.05 | 1.018 | 0.768 | 0.0005 | 24.00 | 0.1154 |
| TRM-MLP | trm_sudoku_mlp_w256_s2 | w4t | 0.000 | 0.979 | 0.1626 | 1.00 | 1.2234 |
| TRM-MLP | trm_sudoku_mlp_w256_s2 | noise0.1 | 0.868 | 0.738 | 0.0003 | 12.00 | 0.1787 |
| TRM-MLP | trm_sudoku_mlp_w256_s2 | noise0.05 | 0.984 | 0.561 | 0.0000 | 48.00 | 0.0813 |
| TRM-MLP | trm_sudoku_mlp_w512_s0 | w4t | 0.049 | 1.001 | 0.1396 | 0.70 | 0.9800 |
| TRM-MLP | trm_sudoku_mlp_w512_s0 | noise0.1 | 0.973 | 0.879 | 0.0094 | 16.00 | 0.2975 |
| TRM-MLP | trm_sudoku_mlp_w512_s0 | noise0.05 | 1.040 | 0.904 | 0.0083 | 32.00 | 0.2142 |
| TRM-MLP | trm_sudoku_mlp_w512_s1 | w4t | 0.035 | 0.988 | 0.2297 | 0.70 | 1.2229 |
| TRM-MLP | trm_sudoku_mlp_w512_s1 | noise0.1 | 0.956 | 0.926 | 0.0144 | 6.00 | 0.3061 |
| TRM-MLP | trm_sudoku_mlp_w512_s1 | noise0.05 | 1.013 | 0.933 | 0.0158 | 16.00 | 0.1837 |
| TRM-MLP | trm_sudoku_mlp_w512_s2 | w4t | 0.022 | 0.997 | 0.2639 | 0.70 | 1.1828 |
| TRM-MLP | trm_sudoku_mlp_w512_s2 | noise0.1 | 0.969 | 0.989 | 0.0354 | 12.00 | 0.2247 |
| TRM-MLP | trm_sudoku_mlp_w512_s2 | noise0.05 | 0.987 | 0.987 | 0.0371 | 24.00 | 0.1377 |
| TRM-MLP | trm_sudoku_mlp_w768_s0 | w4t | 0.091 | 0.977 | 0.0606 | 0.70 | 1.1013 |
| TRM-MLP | trm_sudoku_mlp_w768_s0 | noise0.1 | 0.918 | 0.978 | 0.0476 | 8.00 | 0.4267 |
| TRM-MLP | trm_sudoku_mlp_w768_s0 | noise0.05 | 0.981 | 0.935 | 0.0086 | 32.00 | 0.2350 |
| TRM-attn | maze_attn | w3t | 0.000 | 0.981 | 0.0655 | 0.70 | 1.0978 |
| TRM-attn | maze_attn | noise2.0 | 0.000 | 0.704 | 0.0003 | 1.00 | 0.7330 |
| TRM-attn | maze_attn | noise3.0 | 0.000 | 0.847 | 0.4169 | 1.00 | 1.3676 |
| TRM-attn | maze_attn | +a4 | 0.000 | 0.993 | 0.4871 | 0.70 | 1.1539 |
| TRM-attn | maze_attn | noise1.5 | 0.310 | 0.777 | 0.0006 | 4.00 | 0.5041 |
| TRM-attn | maze_attn | w2g32 | 0.405 | 0.839 | 0.0108 | 0.70 | 0.6937 |
| TRM-attn | maze_attn | w2a | 0.881 | 0.736 | 0.0003 | 2.00 | 0.5418 |
| TRM-attn | maze_attn | noise0.05 | 1.000 | 0.749 | 0.0000 | 128.00 | 0.0171 |
| TRM-attn | maze_attn | w8c | 1.000 | 0.697 | 0.0000 | inf | 0.0030 |
| TRM-attn | maze_attn | noise0.1 | 1.000 | 0.617 | 0.0000 | 64.00 | 0.0321 |
| TRM-attn | maze_attn | w4t | 1.024 | 0.664 | 0.0000 | 14.00 | 0.1279 |
| TRM-attn | maze_attn | noise1.0 | 1.024 | 0.595 | 0.0000 | 6.00 | 0.3433 |
| TRM-attn | maze_attn | w3c | 1.048 | 0.759 | 0.0002 | 12.00 | 0.1585 |
| TRM-attn | sudoku_attn | w3t | 0.000 | 0.980 | 0.4472 | 0.70 | 1.4071 |
| TRM-attn | sudoku_attn | +a4 | 0.000 | 1.015 | 0.5843 | 0.70 | 1.3661 |
| TRM-attn | sudoku_attn | noise1.0 | 0.070 | 0.950 | 0.1549 | 0.70 | 1.1248 |
| TRM-attn | sudoku_attn | noise0.7 | 0.747 | 0.963 | 0.0901 | 6.00 | 0.6120 |
| TRM-attn | sudoku_attn | w3c | 0.876 | 0.944 | 0.0655 | 12.00 | 0.4473 |
| TRM-attn | sudoku_attn | noise0.5 | 0.925 | 0.925 | 0.0416 | 12.00 | 0.4419 |
| TRM-attn | sudoku_attn | w3a | 0.935 | 0.910 | 0.0443 | 16.00 | 0.3762 |
| TRM-attn | sudoku_attn | noise0.3 | 0.957 | 0.762 | 0.0074 | 24.00 | 0.3006 |
| TRM-attn | sudoku_attn | w4c | 0.984 | 0.980 | 0.0675 | 32.00 | 0.2719 |
| TRM-attn | sudoku_attn | w4t | 0.989 | 0.875 | 0.0321 | 16.00 | 0.4269 |
| TRM-attn | sudoku_attn | w3g32 | 1.005 | 0.914 | 0.0538 | 16.00 | 0.3277 |
| TRM-attn | sudoku_attn | w8c | 1.027 | 0.920 | 0.0529 | 128.00 | 0.0590 |
| TRM-attn | sudoku_attn | noise0.1 | 1.027 | 0.955 | 0.0595 | 48.00 | 0.1845 |
| TRM-attn | sudoku_attn | noise0.05 | 1.043 | 0.940 | 0.0489 | 48.00 | 0.1676 |
| TRM-attn | sudoku_attn | +a8 | 1.043 | 0.979 | 0.0595 | 24.00 | 0.2432 |
| TRM-attn | sudoku_attn | w4c+a8 | 1.059 | 0.986 | 0.0711 | 24.00 | 0.2911 |
| TRM-attn | trm_sudoku_attn_w512_s0 | w4t | 1.028 | 0.750 | 0.0014 | 8.00 | 0.3647 |
| TRM-attn | trm_sudoku_attn_w512_s0 | noise0.05 | 1.034 | 0.790 | 0.0021 | 128.00 | 0.0447 |
| TRM-attn | trm_sudoku_attn_w512_s0 | noise0.1 | 1.056 | 0.816 | 0.0019 | 48.00 | 0.0888 |
| TRM-attn | trm_sudoku_attn_w512_s1 | noise0.05 | 0.966 | 0.873 | 0.0508 | 24.00 | 0.3596 |
| TRM-attn | trm_sudoku_attn_w512_s1 | noise0.1 | 0.971 | 0.828 | 0.0545 | 24.00 | 0.3843 |
| TRM-attn | trm_sudoku_attn_w512_s1 | w4t | 0.971 | 0.827 | 0.0529 | 12.00 | 0.5271 |
| TRM-attn | trm_sudoku_attn_w512_s2 | w4t | 0.366 | 0.925 | 0.3026 | 0.70 | 0.9511 |
| TRM-attn | trm_sudoku_attn_w512_s2 | noise0.05 | 1.063 | 0.944 | 0.3465 | 4.00 | 0.5359 |
| TRM-attn | trm_sudoku_attn_w512_s2 | noise0.1 | 1.120 | 0.975 | 0.3456 | 4.00 | 0.5718 |
| seeds | x5_attn512_s3 | noise1.0 | 0.122 | 0.957 | 0.2412 | 0.70 | 1.0847 |
| seeds | x5_attn512_s3 | w4t | 0.350 | 0.915 | 0.2495 | 0.70 | 0.9821 |
| seeds | x5_attn512_s3 | noise0.7 | 0.492 | 0.889 | 0.0628 | 0.70 | 0.9841 |
| seeds | x5_attn512_s3 | noise0.5 | 0.838 | 0.650 | 0.0003 | 12.00 | 0.3872 |
| seeds | x5_attn512_s3 | w3c | 0.848 | 0.650 | 0.0002 | 12.00 | 0.3537 |
| seeds | x5_attn512_s3 | noise0.05 | 0.949 | 0.663 | 0.0003 | inf | 0.0412 |
| seeds | x5_attn512_s3 | w3g32 | 0.949 | 0.621 | 0.0002 | 32.00 | 0.1874 |
| seeds | x5_attn512_s3 | noise0.1 | 0.970 | 0.616 | 0.0002 | 96.00 | 0.0809 |
| seeds | x5_attn512_s3 | w4c | 0.985 | 0.691 | 0.0004 | 32.00 | 0.1446 |
| seeds | x5_attn512_s4 | noise1.0 | 0.000 | 0.974 | 0.1665 | 0.70 | 1.3614 |
| seeds | x5_attn512_s4 | noise0.7 | 0.321 | 0.965 | 0.3163 | 0.70 | 1.0896 |
| seeds | x5_attn512_s4 | noise0.5 | 0.710 | 0.795 | 0.0060 | 6.00 | 0.4545 |
| seeds | x5_attn512_s4 | w4t | 0.839 | 0.854 | 0.0609 | 6.00 | 0.3935 |
| seeds | x5_attn512_s4 | w3c | 0.959 | 0.735 | 0.0062 | 12.00 | 0.3879 |
| seeds | x5_attn512_s4 | w3g32 | 0.969 | 0.643 | 0.0000 | 16.00 | 0.2003 |
| seeds | x5_attn512_s4 | w4c | 0.979 | 0.813 | 0.0000 | 32.00 | 0.1332 |
| seeds | x5_attn512_s4 | noise0.05 | 0.995 | 0.851 | 0.0000 | 128.00 | 0.0379 |
| seeds | x5_attn512_s4 | noise0.1 | 1.021 | 0.794 | 0.0000 | 96.00 | 0.0739 |
| seeds | x5_attn512_s5 | noise1.0 | 0.000 | 0.936 | 0.1208 | 0.70 | 1.2049 |
| seeds | x5_attn512_s5 | noise0.7 | 0.531 | 0.847 | 0.1099 | 0.70 | 0.8692 |
| seeds | x5_attn512_s5 | noise0.5 | 0.866 | 0.558 | 0.0000 | 12.00 | 0.3596 |
| seeds | x5_attn512_s5 | w4t | 0.872 | 0.611 | 0.0000 | 8.00 | 0.3188 |
| seeds | x5_attn512_s5 | w3c | 0.905 | 0.508 | 0.0000 | 8.00 | 0.3873 |
| seeds | x5_attn512_s5 | w3g32 | 0.939 | 0.563 | 0.0000 | 24.00 | 0.1734 |
| seeds | x5_attn512_s5 | noise0.1 | 0.972 | 0.536 | 0.0000 | 96.00 | 0.0705 |
| seeds | x5_attn512_s5 | noise0.05 | 0.983 | 0.516 | 0.0000 | inf | 0.0344 |
| seeds | x5_attn512_s5 | w4c | 0.983 | 0.525 | 0.0000 | 48.00 | 0.1264 |
| seeds | x5_mlp512_s3 | noise0.3 | 0.049 | 0.917 | 0.0482 | 0.70 | 1.0811 |
| seeds | x5_mlp512_s3 | w4t | 0.053 | 0.923 | 0.0225 | 0.50 | 0.9105 |
| seeds | x5_mlp512_s3 | noise0.2 | 0.367 | 0.982 | 0.0546 | 1.00 | 0.9113 |
| seeds | x5_mlp512_s3 | noise0.15 | 0.593 | 0.980 | 0.0401 | 1.50 | 0.8718 |
| seeds | x5_mlp512_s3 | w3g32 | 0.739 | 0.965 | 0.0239 | 12.00 | 0.3908 |
| seeds | x5_mlp512_s3 | noise0.1 | 0.934 | 0.848 | 0.0039 | 12.00 | 0.2270 |
| seeds | x5_mlp512_s3 | w4c | 0.965 | 0.965 | 0.0400 | 24.00 | 0.2379 |
| seeds | x5_mlp512_s3 | noise0.05 | 0.996 | 0.886 | 0.0054 | 32.00 | 0.2164 |
| seeds | x5_mlp512_s4 | noise0.3 | 0.009 | 0.994 | 0.1444 | 0.70 | 1.0472 |
| seeds | x5_mlp512_s4 | noise0.2 | 0.059 | 0.984 | 0.1424 | 0.70 | 0.9783 |
| seeds | x5_mlp512_s4 | w4t | 0.095 | 1.000 | 0.1069 | 0.70 | 0.9524 |
| seeds | x5_mlp512_s4 | noise0.15 | 0.290 | 0.969 | 0.1648 | 1.00 | 0.9069 |
| seeds | x5_mlp512_s4 | w3g32 | 0.900 | 0.986 | 0.0278 | 12.00 | 0.3073 |
| seeds | x5_mlp512_s4 | noise0.1 | 0.977 | 0.980 | 0.0261 | 12.00 | 0.2530 |
| seeds | x5_mlp512_s4 | w4c | 1.027 | 0.979 | 0.0236 | 24.00 | 0.2734 |
| seeds | x5_mlp512_s4 | noise0.05 | 1.032 | 0.914 | 0.0330 | 24.00 | 0.1915 |
| seeds | x5_mlp512_s5 | noise0.3 | 0.004 | 0.996 | 0.1099 | 0.70 | 1.0768 |
| seeds | x5_mlp512_s5 | noise0.2 | 0.057 | 1.008 | 0.1037 | 0.70 | 0.9857 |
| seeds | x5_mlp512_s5 | w4t | 0.140 | 1.002 | 0.1623 | 0.70 | 0.9923 |
| seeds | x5_mlp512_s5 | noise0.15 | 0.434 | 1.000 | 0.1045 | 1.00 | 0.8821 |
| seeds | x5_mlp512_s5 | w3g32 | 0.899 | 0.902 | 0.0125 | 6.00 | 0.3320 |
| seeds | x5_mlp512_s5 | noise0.1 | 0.934 | 0.875 | 0.0020 | 6.00 | 0.2196 |
| seeds | x5_mlp512_s5 | w4c | 0.978 | 0.851 | 0.0019 | 16.00 | 0.1960 |
| seeds | x5_mlp512_s5 | noise0.05 | 1.004 | 0.929 | 0.0021 | 24.00 | 0.0961 |
