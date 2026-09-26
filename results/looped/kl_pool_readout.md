## KL(bf16 || compressed) vs latent fidelity as a pooled collapse threshold (saved tf2 records)

platform per model: huginn babel, olmo_rec32 p4d, ouro14 p4d, ouro26 p4d

| model | loops | quant | n | retained | collapse | KL | fidelity | ΔNLL % |
|---|---|---|---|---|---|---|---|---|
| huginn | 32 | w4c | 200 | 82% |  | 0.083 | 0.930 | -4% |
| huginn | 32 | w3a | 200 | 9% | yes | 0.334 | 0.782 | +17% |
| huginn | 32 | w4t | 200 | 5% | yes | 0.567 | 0.696 | +46% |
| olmo_rec32 | 8 | w8c | 100 | 116% |  | 0.000 | 1.000 | -0% |
| olmo_rec32 | 16 | w8c | 100 | 104% |  | 0.000 | 1.000 | -0% |
| olmo_rec32 | 4 | w8c | 100 | 98% |  | 0.001 | 1.000 | -0% |
| olmo_rec32 | 32 | w8c | 100 | 105% |  | 0.001 | 1.000 | -0% |
| olmo_rec32 | 8 | w4g32 | 100 | 91% |  | 0.017 | 0.984 | +1% |
| olmo_rec32 | 32 | w4g32 | 100 | 100% |  | 0.017 | 0.984 | +1% |
| olmo_rec32 | 16 | w4g32 | 100 | 91% |  | 0.017 | 0.984 | +1% |
| olmo_rec32 | 4 | w4g32 | 100 | 93% |  | 0.020 | 0.983 | +1% |
| olmo_rec32 | 32 | w4a | 500 | 80% |  | 0.053 | 0.961 | +6% |
| olmo_rec32 | 4 | w4a | 100 | 68% |  | 0.053 | 0.958 | +6% |
| olmo_rec32 | 8 | w4c | 100 | 84% |  | 0.087 | 0.947 | +8% |
| olmo_rec32 | 4 | w4c | 100 | 70% |  | 0.087 | 0.942 | +7% |
| olmo_rec32 | 16 | w4c | 100 | 80% |  | 0.088 | 0.947 | +8% |
| olmo_rec32 | 32 | w4c | 500 | 81% |  | 0.090 | 0.946 | +9% |
| olmo_rec32 | 32 | w3g32 | 200 | 0% | yes | 0.214 | 0.902 | +27% |
| olmo_rec32 | 32 | w3g128 | 200 | 41% | yes | 0.259 | 0.863 | +34% |
| olmo_rec32 | 32 | w3a | 500 | 5% | yes | 0.805 | 0.790 | +107% |
| olmo_rec32 | 8 | w3a | 100 | 12% | yes | 0.865 | 0.785 | +111% |
| olmo_rec32 | 4 | w3a | 100 | 5% | yes | 0.919 | 0.760 | +113% |
| olmo_rec32 | 32 | w3c | 200 | 0% | yes | 3.514 | 0.542 | +453% |
| olmo_rec32 | 8 | w4t | 100 | 0% | yes | 9.518 | 0.109 | +1227% |
| olmo_rec32 | 4 | w4t | 100 | 0% | yes | 9.585 | 0.116 | +1214% |
| olmo_rec32 | 32 | w4t | 100 | 0% | yes | 9.610 | 0.113 | +1230% |
| olmo_rec32 | 16 | w4t | 100 | 0% | yes | 10.222 | 0.119 | +1316% |
| ouro14 | 4 | w8c | 500 | 99% |  | 0.001 | 0.997 | -0% |
| ouro14 | 4 | w5c | 200 | 92% |  | 0.036 | 0.910 | -3% |
| ouro14 | 4 | w4g32 | 500 | 102% |  | 0.039 | 0.888 | +1% |
| ouro14 | 4 | w4g128 | 200 | 85% |  | 0.077 | 0.856 | -2% |
| ouro14 | 4 | w4a | 200 | 67% |  | 0.123 | 0.789 | +5% |
| ouro14 | 4 | w4c | 500 | 11% | yes | 0.287 | 0.737 | +18% |
| ouro26 | 4 | w4c | 500 | 93% |  | 0.093 | 0.854 | +4% |
| ouro26 | 2 | w4c | 200 | 76% |  | 0.108 | 0.808 | +7% |
| ouro26 | 8 | w4c | 200 | 73% |  | 0.174 | 0.827 | +7% |
| ouro26 | 4 | w4t | 200 | 0% | yes | 12.070 | 0.010 | +1471% |

- KL: pooled AUROC 1.000; collapsed min 0.214 vs held max 0.174; leave-one-model-out accuracy 0.946 (huginn 1.00, olmo_rec32 0.96, ouro14 1.00, ouro26 0.75)
- 1-φ: pooled AUROC 0.963; collapsed min 0.098 vs held max 0.211; leave-one-model-out accuracy 0.838 (huginn 1.00, olmo_rec32 0.88, ouro14 0.83, ouro26 0.50)
- ΔNLL %: pooled AUROC 1.000; collapsed min 0.167 vs held max 0.091; leave-one-model-out accuracy 1.000 (huginn 1.00, olmo_rec32 1.00, ouro14 1.00, ouro26 1.00)

37 cells (14 collapsed). Hidden states of compressed models were not saved (only bf16 prompt latents in ref files), so φ without the top-k massive dimensions / standardized φ cannot be recomputed without a rerun.
