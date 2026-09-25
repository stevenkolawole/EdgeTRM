# frozen selection on dev (0-999), evaluation on held-out (1000-1999): sudoku_attn

dev files: p_sudoku_attn_B1.json, p_sudoku_attn_B2.json, p_sudoku_attn_B3.json, p_sudoku_attn_B4.json, p_sudoku_attn_B5.json, p_sudoku_attn_B6.json, p_sudoku_attn_T128_B9.json, p_sudoku_attn_T256_n8c3_R.json, p_sudoku_attn_T256_w3c_R.json, p_sudoku_attn_T256_w3g32_R.json, p_sudoku_attn_T256_w4c_R.json, p_sudoku_attn_T32_R.json, p_sudoku_attn_T64_B10.json, p_sudoku_attn_T64_B7.json, p_sudoku_attn_T64_B8.json, p_sudoku_attn_ref.json
held-out files: p_sudoku_attn_T128_heldout_R.json, p_sudoku_attn_T128b_heldout_R.json, p_sudoku_attn_T16_heldout_R.json, p_sudoku_attn_T256_heldout_R.json, p_sudoku_attn_T256n_heldout_R.json, p_sudoku_attn_T32_heldout_R.json, p_sudoku_attn_T64_heldout_R.json, p_sudoku_attn_T64b_heldout_R.json

configurations compared on dev before the pick: 535 hybrid schedules for this checkpoint (263 with INT8 polish and a quantized cheap map); across every M1 dev record: 2801 hybrid schedules, 5828 schedules of any kind.

| T | frozen pick (dev) | dev acc | comparator (dev pick) | dev acc | dev delta, p | dev p x m_ckpt | dev p x m_all | held-out hybrid | held-out comparator | held-out delta (discordant) | held-out p | fp32 dev max <= 2T | fp32 held-out max <= 2T | held-out vs fp32 at its best depth <= 2T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | w3g32 12 + w8c 4 | 67.8 | n8a3 24 | 68.5 | -0.7, p=0.61 (64/71) | 1 | 1 | not run | not run | -- | -- | 69.2 | 81.9 | fp32@31: 81.9, -1.9 (37/56), p=0.061 |
| 32 | n8c3 16 + w8c 16 | 73.1 | n8c4 43 | 71.8 | +1.3, p=0.22 (54/41) | 1 | 1 | not run | not run | -- | -- | 72.4 | 83.5 | -- |
| 64 | w3c 48 + w8t 16 | 76.3 | n8c3 90 | 74.2 | +2.1, p=0.085 (78/57) | 1 | 1 | 87.9 | n8c3 90: 84.8 | +3.1 (79/48) | 0.0075 | 74.2 | 85.1 | fp32@119: 85.1, +2.8 (56/28), p=0.003 |
| 128 | w3c 96 + w8t 32 | 79.5 | n8c3 180 | 76.7 | +2.8, p=0.017 (78/50) | 1 | 1 | 90.8 | n8c3 180: 87.8 | +3.0 (64/34) | 0.0032 | 75.9 | 86.7 | fp32@254: 86.7, +4.1 (62/21), p=7.5e-06 |
| 256 | w3c 224 + w8c 32 | 82.5 | w3c 256 | 77.7 | +4.8, p=3.5e-11 (53/5) | 1.9e-08 | 2e-07 | 91.9 | w3c 256: 87.0 | +4.9 (51/2) | 3.2e-13 | 77.1 | 88.1 | fp32@512: 88.1, +3.8 (59/21), p=2.5e-05 |

schedules compared on dev, by task and checkpoint: dt_maze13: 710; dt_maze33: 640; hrm_maze: 120; hrm_sudoku: 621; maze_attn: 138; sudoku_attn: 1817; sudoku_attn trm_sudoku_attn_w512_s0: 75; sudoku_attn trm_sudoku_attn_w512_s1: 75; sudoku_attn trm_sudoku_attn_w512_s2: 75; sudoku_mlp: 1557
