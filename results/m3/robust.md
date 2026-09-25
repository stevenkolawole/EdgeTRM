
### hrm_sudoku w2c
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| omse_ar_layer | 45.1 / 44.3 / 45.5 | 45.0 [43.4, 46.5] |
| dT_gptq | 38.3 / 37.2 / 41.4 | 39.0 [33.6, 44.4] |
| blk_gptq | 38.0 / 37.5 / 38.4 | 38.0 [36.8, 39.1] |
| omse_gptq | 29.9 / 29.7 / 31.5 | 30.4 [27.9, 32.8] |
| mse_gptq | 0.0 / 0.0 / 0.0 | 0.0 [0.0, 0.0] |
- step (dT_gptq) minus best layer-wise omse_ar_layer: -6.0 points [-10.1, -1.9] over 3 paired sets
- step (dT_gptq) minus blk_gptq: +1.0 points [-3.4, +5.4] over 3 paired sets

### hrm_sudoku w2g32
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| omse_ar_layer | 47.8 / 48.5 / 46.9 | 47.7 [45.7, 49.7] |
| dT_gptq | 44.6 / 43.6 / 44.8 | 44.3 [42.7, 45.9] |
| blk_gptq | 44.1 / 41.8 / 44.2 | 43.4 [40.0, 46.7] |
| omse_gptq | 39.4 / 40.6 / 39.7 | 39.9 [38.3, 41.5] |
| mse_gptq | 0.2 / 0.2 / 0.2 | 0.2 [0.2, 0.2] |
- step (dT_gptq) minus best layer-wise omse_ar_layer: -3.4 points [-6.9, +0.1] over 3 paired sets
- step (dT_gptq) minus blk_gptq: +1.0 points [-0.8, +2.8] over 3 paired sets

### hrm_sudoku w3t
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| blk_gptq | 49.3 / 48.2 | 48.8 [41.8, 55.7] |
| omse_ar_layer | 48.7 | 48.7 [nan, nan] |
| dT_gptq | 48.7 / 48.6 | 48.6 [48.0, 49.3] |
| omse_gptq | 48.9 / 48.0 | 48.4 [42.7, 54.2] |
| mse_gptq | 26.6 / 27.4 | 27.0 [21.9, 32.1] |
- step (dT_gptq) minus best layer-wise omse_ar_layer: +0.0 points [+nan, +nan] over 1 paired sets
- step (dT_gptq) minus blk_gptq: -0.1 points [-6.5, +6.3] over 2 paired sets

### mdeq_small w4t
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| dT_gptq | 66.7 / 66.0 / 66.3 | 66.3 [65.4, 67.2] |
| mse_gptq | 66.0 / 65.7 / 65.6 | 65.8 [65.3, 66.3] |
| omse_gptq | 61.5 / 65.4 / 66.6 | 64.5 [57.8, 71.2] |
- step (dT_gptq) minus best layer-wise mse_gptq: +0.6 points [-0.1, +1.2] over 3 paired sets

### sudoku_attn w3t
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| omse_ar_layer | 67.0 / 66.5 / 68.7 | 67.4 [64.5, 70.3] |
| omse_gptq | 67.4 / 66.4 / 66.2 | 66.7 [65.1, 68.3] |
| dTj_gptq | 66.3 / 66.0 / 65.8 | 66.0 [65.4, 66.7] |
| dT_gptq | 63.7 / 65.9 / 65.6 | 65.1 [62.1, 68.0] |
| blk_gptq | 63.7 / 63.8 / 63.6 | 63.7 [63.5, 63.9] |
| mse_gptq | 33.2 / 34.8 / 36.5 | 34.8 [30.7, 38.9] |
- step (dT_gptq) minus best layer-wise omse_ar_layer: -2.3 points [-6.1, +1.4] over 3 paired sets
- step (dT_gptq) minus blk_gptq: +1.4 points [-1.6, +4.3] over 3 paired sets

### sudoku_mlp w3c
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| omse_ar_layer | 70.6 / 70.4 / 70.9 | 70.6 [70.0, 71.3] |
| omse_gptq | 68.5 / 68.2 / 69.2 | 68.6 [67.4, 69.9] |
| dTj_gptq | 68.2 / 66.7 / 67.9 | 67.6 [65.6, 69.6] |
| dT_gptq | 65.9 / 67.8 / 69.1 | 67.6 [63.6, 71.6] |
| blk_gptq | 67.5 / 66.1 / 68.5 | 67.4 [64.4, 70.4] |
| mse_gptq | 64.9 / 63.0 / 64.5 | 64.1 [61.6, 66.6] |
- step (dT_gptq) minus best layer-wise omse_ar_layer: -3.0 points [-6.8, +0.7] over 3 paired sets
- step (dT_gptq) minus blk_gptq: +0.2 points [-3.9, +4.4] over 3 paired sets

### sudoku_mlp w4t
| method | calibration sets | mean % [95% CI over sets] |
|---|---|---|
| dTj_gptq | 71.7 / 70.7 | 71.2 [64.8, 77.6] |
| dT_gptq | 71.6 / 71.7 / 69.8 | 71.0 [68.4, 73.7] |
| blk_gptq | 70.1 / 69.6 / 71.6 | 70.4 [67.8, 73.0] |
| mse_gptq | 67.6 / 68.4 / 67.6 | 67.9 [66.7, 69.0] |
| omse_gptq | 64.5 / 66.3 / 64.3 | 65.0 [62.3, 67.8] |
| omse_ar_layer | 67.3 / 58.0 / 63.3 | 62.9 [51.3, 74.4] |
- step (dT_gptq) minus best layer-wise mse_gptq: +3.2 points [+0.9, +5.4] over 3 paired sets
- step (dT_gptq) minus blk_gptq: +0.6 points [-4.6, +5.8] over 3 paired sets
