
## m1o26: dev n=500, held-out n=500

| cell | traffic | memory | dev | held-out |
|---|---|---|---|---|
| ouro26_s2_w8c_core | 16.00 | 8.0 | 49.4 | 50.2 |
| ouro26_s4_w4c_core | 16.00 | 4.0 | 77.6 | 77.6 |
| ouro26_s4_w4g128_core | 16.50 | 4.1 | 69.0 | 65.4 |
| ouro26_s4_w4g128~P3@w8c_core | 20.38 | 12.1 | 71.8 | 72.2 |
| ouro26_s5_w4g128_core | 20.62 | 4.1 | 66.8 | 65.2 |
| ouro26_s3_w8c_core | 24.00 | 8.0 | 63.0 | 60.8 |
| ouro26_s4_w4c~P23@w8c_core | 24.00 | 12.0 | 80.6 | 79.6 |
| ouro26_s4_w4g128~P03@w8c_core | 24.25 | 12.1 | 66.4 | 66.6 |
| ouro26_s4_w4g128~P23@w8c_core | 24.25 | 12.1 | 80.6 | 77.2 |
| ouro26_s6_w4g128_core | 24.75 | 4.1 | 64.0 | 62.0 |
| ouro26_s4_w8c_core | 32.00 | 8.0 | 84.4 | 82.0 |
| ouro26_s4_bf16_core | 64.00 | 16.0 | 83.6 | 83.2 |

pick (dev) ouro26_s4_w4c~P23@w8c_core 80.6 (traffic 24.00, memory 12.0); comparator ouro26_s4_w4c_core 77.6: dev +3.0 (64/49, p=0.19); held-out 79.6 vs 77.6: +2.0 (65/55, p=0.41)

pick (dev) ouro26_s4_w4g128~P23@w8c_core 80.6 (traffic 24.25, memory 12.1); comparator ouro26_s4_w4c_core 77.6: dev +3.0 (64/49, p=0.19); held-out 77.2 vs 77.6: -0.4 (67/69, p=0.93)

## m1o14: dev n=500, held-out n=500

| cell | traffic | memory | dev | held-out |
|---|---|---|---|---|
| ouro14_s2_w8c_core | 16.00 | 8.0 | 77.8 | 76.2 |
| ouro14_s4_w4c_core | 16.00 | 4.0 | 8.0 | 6.4 |
| ouro14_s4_w4g128_core | 16.50 | 4.1 | 69.4 | 69.6 |
| ouro14_s4_w4g128~P3@w8c_core | 20.38 | 12.1 | 76.6 | 72.6 |
| ouro14_s5_w4g128_core | 20.62 | 4.1 | 70.4 | 66.6 |
| ouro14_s3_w8c_core | 24.00 | 8.0 | 77.0 | 78.8 |
| ouro14_s4_w4c~P23@w8c_core | 24.00 | 12.0 | 66.6 | 66.4 |
| ouro14_s4_w4g128~P03@w8c_core | 24.25 | 12.1 | 74.8 | 73.2 |
| ouro14_s4_w4g128~P23@w8c_core | 24.25 | 12.1 | 77.2 | 76.4 |
| ouro14_s6_w4g128_core | 24.75 | 4.1 | 56.8 | 55.4 |
| ouro14_s4_w8c_core | 32.00 | 8.0 | 75.6 | 78.8 |
| ouro14_s4_bf16_core | 64.00 | 16.0 | 75.0 | 77.8 |

pick (dev) ouro14_s4_w4g128~P23@w8c_core 77.2 (traffic 24.25, memory 12.1); comparator ouro14_s2_w8c_core 77.8: dev -0.6 (51/54, p=0.85); held-out 76.4 vs 76.2: +0.2 (58/57, p=1)
