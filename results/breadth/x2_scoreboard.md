| model | S (X2) | predicted sigma_1/2 (time) | measured sigma_1/2 (source) | ratio | within x1.46 / x1.92 |
|---|---|---|---|---|---|
| x5_attn512_s3 | 0.752 | 0.537 (2026-09-25T12:10:49Z) | 0.659 (fp32 0.686; results/q7/babel/var_attn512_s3/eval_step_65100.json) | 0.81 | yes / yes |
| ft_fp32_s0 | 1.569 | 0.257 (2026-09-25T12:12:18Z) | 0.135 (fp32 0.614; X2 width sweep ft_fp32_s0) | 1.91 | no / yes |
| qat_lsq_s0 | 0.924 | 0.437 (2026-09-25T12:23:49Z) | 0.228 (fp32 0.561; X2 width sweep qat_lsq_s0) | 1.92 | no / yes |
| x5_attn512_s5 | 0.669 | 0.603 (2026-09-25T12:28:57Z) | 0.700 (fp32 0.643; results/q7/babel/var_attn512_s5/eval_step_65100.json) | 0.86 | yes / yes |
| x5_attn512_s4 | 0.717 | 0.563 (2026-09-25T12:38:55Z) | 0.580 (fp32 0.711; results/q7/babel/var_attn512_s4/eval_step_65100.json) | 0.97 | yes / yes |
| qat_lsq_s1 | 1.213 | 0.333 (2026-09-25T15:41:33Z) | 0.234 (fp32 0.629; X2 width sweep qat_lsq_s1) | 1.42 | yes / yes |
| qat_naive_s0 | 1.088 | 0.371 (2026-09-25T15:42:35Z) | > 0.5 (X2 width sweep qat_naive_s0) | | |
| qat_lsq_s2 | 0.951 | 0.424 (2026-09-25T15:42:35Z) | 0.226 (fp32 0.618; X2 width sweep qat_lsq_s2) | 1.88 | no / yes |
| x5_mlp512_s3 | 1.502 | 0.268 (2026-09-25T17:29:25Z) | 0.173 (fp32 0.821; results/q7/babel/var_mlp512_s3/eval_step_65100.json) | 1.55 | no / yes |
| x5_mlp512_s5 | 1.710 | 0.236 (2026-09-25T17:29:27Z) | 0.138 (fp32 0.824; results/q7/babel/var_mlp512_s5/eval_step_65100.json) | 1.71 | no / yes |
| x5_mlp512_s4 | 1.422 | 0.284 (2026-09-25T17:39:32Z) | 0.159 (fp32 0.813; results/q7/babel/var_mlp512_s4/eval_step_65100.json) | 1.78 | no / yes |

scored 10: median x1.63, worst x1.92; band median x1.46, worst x1.92
