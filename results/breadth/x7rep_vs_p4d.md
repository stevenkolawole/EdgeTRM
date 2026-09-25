| model | quantizer | axis | p4d change [95% CI] (verdict) | Babel change [95% CI] (verdict) | Babel range | same sign | same verdict |
|---|---|---|---|---|---|---|---|
| hrm_sudoku | w4c | nsup | +0.6 [-1.1, +2.2] (flat) | -0.3 [-2.0, +1.3] (flat) | 1->32 | NO | yes |
| hrm_sudoku | w4c | H | +0.7 [-1.1, +2.5] (flat) | -0.2 [-2.0, +1.6] (flat) | 1->4 | NO | yes |
| hrm_sudoku | w4g32 | nsup | +0.7 [-0.9, +2.3] (flat) | +0.5 [-1.1, +2.0] (flat) | 1->32 | yes | yes |
| hrm_sudoku | w4g32 | H | +0.3 [-1.4, +1.8] (flat) | +0.1 [-1.5, +1.7] (flat) | 1->4 | yes | yes |
| hrm_sudoku | w4g32+a4g32 | nsup | +2.5 [+0.7, +4.3] (narrows) | +1.6 [-0.3, +3.5] (flat) | 1->32 | yes | no |
| hrm_sudoku | w4g32+a4g32 | H | +2.5 [+0.6, +4.4] (narrows) | +0.8 [-1.0, +2.8] (flat) | 1->4 | yes | no |
| hrm_sudoku | w4mx+a4mx | nsup | +0.9 [-1.1, +2.7] (flat) | +0.8 [-1.2, +2.6] (flat) | 1->32 | yes | yes |
| hrm_sudoku | w4mx+a4mx | H | +1.5 [-0.5, +3.5] (flat) | +1.9 [-0.1, +4.1] (flat) | 1->4 | yes | yes |
| hrm_sudoku | w4t | nsup | -0.6 [-2.5, +1.3] (flat) | -1.4 [-3.3, +0.5] (flat) | 1->32 | yes | yes |
| hrm_sudoku | w4t | H | +1.7 [-0.3, +3.7] (flat) | +0.9 [-1.1, +2.9] (flat) | 1->4 | yes | yes |
| hrm_sudoku | w8t+a8ts@0 | nsup | +2.5 [+0.7, +4.2] (narrows) | +1.9 [+0.1, +3.8] (narrows) | 1->32 | yes | yes |
| hrm_sudoku | w8t+a8ts@0 | H | +2.0 [+0.1, +3.8] (narrows) | +1.2 [-0.7, +3.1] (flat) | 1->4 | yes | no |
| maze_attn | w4c | nsup | -0.8 [-2.0, +0.4] (flat) | -0.8 [-2.0, +0.4] (flat) | 1->32 | yes | yes |
| maze_attn | w4c | H | +0.0 [+0.0, +0.0] (flat) | +0.0 [+0.0, +0.0] (flat) | 1->3 | yes | yes |
| maze_attn | w4g32 | nsup | +0.0 [-1.2, +1.2] (flat) | +0.0 [-1.2, +1.2] (flat) | 1->32 | yes | yes |
| maze_attn | w4g32 | H | -0.2 [-0.6, +0.0] (flat) | +0.0 [+0.0, +0.0] (flat) | 1->3 | NO | yes |
| maze_attn | w4g32+a4g32 | nsup | +0.0 [-1.8, +2.0] (flat) | +0.2 [-1.8, +2.2] (flat) | 1->32 | yes | yes |
| maze_attn | w4g32+a4g32 | H | +0.0 [+0.0, +0.0] (flat) | +0.2 [+0.0, +0.6] (flat) | 1->3 | yes | yes |
| maze_attn | w4mx+a4mx | nsup | +2.2 [-0.2, +4.6] (flat) | +1.4 [-1.2, +3.8] (flat) | 1->32 | yes | yes |
| maze_attn | w4mx+a4mx | H | -0.2 [-0.6, +0.0] (flat) | +0.0 [+0.0, +0.0] (flat) | 1->3 | NO | yes |
| maze_attn | w4t | nsup | +1.4 [-1.0, +4.0] (flat) | +1.2 [-1.2, +3.8] (flat) | 1->32 | yes | yes |
| maze_attn | w4t | H | +0.0 [-0.6, +0.6] (flat) | -0.2 [-0.6, +0.0] (flat) | 1->3 | NO | yes |
| maze_attn | w8t+a8ts@0 | nsup | (no p4d record) | +0.8 [-1.0, +2.6] (flat) | 1->32 | - | - |
| maze_attn | w8t+a8ts@0 | H | (no p4d record) | +0.4 [+0.0, +1.0] (flat) | 1->3 | - | - |
| sudoku_attn | w4c | nsup | +0.7 [-1.0, +2.4] (flat) | +1.0 [-0.6, +2.6] (flat) | 1->32 | yes | yes |
| sudoku_attn | w4c | H | +0.3 [-2.0, +2.6] (flat) | +0.6 [-1.8, +2.9] (flat) | 1->6 | yes | yes |
| sudoku_attn | w4g32 | nsup | +1.0 [-0.7, +2.6] (flat) | +1.7 [+0.1, +3.2] (narrows) | 1->32 | yes | no |
| sudoku_attn | w4g32 | H | +0.4 [-2.0, +2.7] (flat) | +1.6 [-0.6, +3.8] (flat) | 1->6 | yes | yes |
| sudoku_attn | w4g32+a4g32 | nsup | +1.9 [+0.0, +3.6] (flat) | +2.0 [+0.1, +3.9] (narrows) | 1->32 | yes | no |
| sudoku_attn | w4g32+a4g32 | H | +0.5 [-2.0, +3.0] (flat) | +1.9 [-0.5, +4.3] (flat) | 1->6 | yes | yes |
| sudoku_attn | w4mx+a4mx | nsup | +1.4 [-0.6, +3.3] (flat) | +2.9 [+0.9, +5.0] (narrows) | 1->32 | yes | no |
| sudoku_attn | w4mx+a4mx | H | +3.5 [+1.0, +6.1] (narrows) | +3.8 [+1.2, +6.2] (narrows) | 1->6 | yes | yes |
| sudoku_attn | w4t | nsup | -2.8 [-5.0, -0.5] (widens) | -1.5 [-3.8, +0.7] (flat) | 1->32 | yes | no |
| sudoku_attn | w4t | H | +1.7 [-1.1, +4.3] (flat) | +3.1 [+0.4, +5.9] (narrows) | 1->6 | yes | no |
| sudoku_attn | w8t+a8ts@0 | nsup | +0.7 [-1.2, +2.5] (flat) | +2.2 [+0.6, +3.8] (narrows) | 1->32 | yes | no |
| sudoku_attn | w8t+a8ts@0 | H | +0.4 [-2.0, +2.8] (flat) | +2.4 [-0.1, +4.9] (flat) | 1->6 | yes | yes |
| sudoku_mlp | w4c | nsup | +1.1 [-1.9, +4.1] (flat) | +1.8 [-1.1, +4.5] (flat) | 1->32 | yes | yes |
| sudoku_mlp | w4c | H | -1.2 [-3.3, +1.1] (flat) | -0.5 [-2.5, +1.6] (flat) | 1->6 | yes | yes |
| sudoku_mlp | w4g32 | nsup | -0.2 [-2.4, +2.1] (flat) | +0.9 [-1.2, +3.0] (flat) | 1->32 | NO | yes |
| sudoku_mlp | w4g32 | H | -1.2 [-3.3, +0.9] (flat) | -0.1 [-2.0, +1.9] (flat) | 1->6 | yes | yes |
| sudoku_mlp | w4g32+a4g32 | nsup | +5.0 [+2.2, +7.9] (narrows) | +5.0 [+2.3, +7.8] (narrows) | 1->32 | yes | yes |
| sudoku_mlp | w4g32+a4g32 | H | -0.3 [-2.5, +1.9] (flat) | +0.1 [-1.9, +2.2] (flat) | 1->6 | NO | yes |
| sudoku_mlp | w4mx+a4mx | nsup | +7.5 [+4.1, +10.9] (narrows) | +8.0 [+4.8, +11.2] (narrows) | 1->32 | yes | yes |
| sudoku_mlp | w4mx+a4mx | H | +0.1 [-2.3, +2.3] (flat) | +0.2 [-2.1, +2.6] (flat) | 1->6 | yes | yes |
| sudoku_mlp | w8t+a8ts@0 | nsup | +13.8 [+10.8, +16.9] (narrows) | +13.3 [+10.2, +16.2] (narrows) | 1->32 | yes | yes |
| sudoku_mlp | w8t+a8ts@0 | H | +2.5 [+0.2, +5.0] (narrows) | +2.8 [+0.4, +5.3] (narrows) | 1->6 | yes | yes |

paired series: 44; same sign of change (non-strict, zero counted as positive): 37 -- strict sign: 35; same verdict: 35; verdict differs: hrm_sudoku/w4g32+a4g32/nsup: p4d narrows vs Babel flat; hrm_sudoku/w4g32+a4g32/H: p4d narrows vs Babel flat; hrm_sudoku/w8t+a8ts@0/H: p4d narrows vs Babel flat; sudoku_attn/w4g32/nsup: p4d flat vs Babel narrows; sudoku_attn/w4g32+a4g32/nsup: p4d flat vs Babel narrows; sudoku_attn/w4mx+a4mx/nsup: p4d flat vs Babel narrows; sudoku_attn/w4t/nsup: p4d widens vs Babel flat; sudoku_attn/w4t/H: p4d flat vs Babel narrows; sudoku_attn/w8t+a8ts@0/nsup: p4d flat vs Babel narrows
