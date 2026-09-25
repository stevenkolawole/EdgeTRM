# Energy per solved instance (E_step x n / accuracy of the same graph)

Accuracy is taken from the graph whose cost is counted (device-faithful): host simulation of the AI Hub compiled QDQ graph at H=1 (or the on-device chain when it exists), never the harness quantizer. Regenerated Sep 24 after the harness-accuracy version credited Sudoku-MLP w8a8/w4a8 with 61% while the compiled graphs solve 0%.

| energy source | task | format | bound | n | exact | accuracy runtime | accuracy source | mJ/answer | mJ/solved |
|---|---|---|---|---|---|---|---|---|---|
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_on_chip | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 19.90 | 1989.84 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_dram | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 42.20 | 4219.70 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_on_chip | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 39.80 | 1989.84 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_dram | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 84.39 | 4219.70 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_on_chip | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 79.59 | 221.09 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_dram | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 168.79 | 468.86 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_on_chip | 8 | 58.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 159.19 | 274.46 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_dram | 8 | 58.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 337.58 | 582.03 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 318.37 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp32 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 675.15 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_on_chip | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 6.53 | 653.12 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_dram | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 17.68 | 1768.06 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_on_chip | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 13.06 | 653.12 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_dram | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 35.36 | 1768.06 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_on_chip | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 26.12 | 72.57 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_dram | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 70.72 | 196.45 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_on_chip | 8 | 58.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 52.25 | 90.09 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_dram | 8 | 58.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 141.44 | 243.87 |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 104.50 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | fp16 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 282.89 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_on_chip | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 1.04 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_dram | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 6.62 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_on_chip | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 2.09 | 104.39 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_dram | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 13.24 | 661.86 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_on_chip | 4 | 35.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 4.18 | 11.93 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_dram | 4 | 35.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 26.47 | 75.64 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_on_chip | 8 | 61.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 8.35 | 13.69 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_dram | 8 | 61.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 52.95 | 86.80 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 16.70 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a8 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 105.90 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 1.02 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 3.81 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 2.04 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 7.62 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_on_chip | 4 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.09 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_dram | 4 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 15.24 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_on_chip | 8 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 8.18 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_dram | 8 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 30.47 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_on_chip | 16 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 16.35 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a8 | weights_dram | 16 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 60.95 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_on_chip | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 1.06 | 106.13 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_dram | 1 | 1.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 6.64 | 663.60 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_on_chip | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 2.12 | 106.13 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_dram | 2 | 2.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 13.27 | 663.60 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_on_chip | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 4.25 | 11.79 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_dram | 4 | 36.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 26.54 | 73.73 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_on_chip | 8 | 56.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 8.49 | 15.16 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_dram | 8 | 56.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 53.09 | 94.80 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 16.98 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | w8a16 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 106.18 | - |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_on_chip | 1 | 1.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 1.04 | 103.94 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_dram | 1 | 1.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 3.83 | 382.67 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_on_chip | 2 | 2.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 2.08 | 103.94 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_dram | 2 | 2.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 7.65 | 382.67 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_on_chip | 4 | 35.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.16 | 11.88 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_dram | 4 | 35.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 15.31 | 43.73 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_on_chip | 8 | 55.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 8.31 | 15.12 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_dram | 8 | 55.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 30.61 | 55.66 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_on_chip | 16 | 69.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 16.63 | 24.10 |
| estimate (Horowitz 45 nm) | sudoku_mlp | w4a16 | weights_dram | 16 | 69.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 61.23 | 88.74 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 22.22 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 52.52 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 44.44 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 105.03 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_on_chip | 4 | 26.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 88.88 | 341.84 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_dram | 4 | 26.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 210.06 | 807.93 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_on_chip | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 177.75 | 348.54 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_dram | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 420.12 | 823.77 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_on_chip | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 355.51 | 546.94 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp32 | weights_dram | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 840.25 | 1292.69 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 7.30 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 22.45 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 14.60 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 44.89 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_on_chip | 4 | 26.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 29.20 | 112.29 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_dram | 4 | 26.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 89.79 | 345.34 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_on_chip | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 58.39 | 114.49 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_dram | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 179.58 | 352.11 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_on_chip | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 116.78 | 179.67 |
| estimate (Horowitz 45 nm) | sudoku_attn | fp16 | weights_dram | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 359.15 | 552.54 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_on_chip | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 1.17 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_dram | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 8.75 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_on_chip | 2 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 2.35 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_dram | 2 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 17.49 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_on_chip | 4 | 19.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 4.69 | 24.69 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_dram | 4 | 19.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 34.99 | 184.14 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_on_chip | 8 | 50.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 9.38 | 18.76 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_dram | 8 | 50.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=100) | 69.97 | 139.95 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 18.76 | - |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a8 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 139.95 | - |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 1.14 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.93 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 2.29 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 9.86 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_on_chip | 4 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.57 | 19.87 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_dram | 4 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 19.72 | 85.74 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_on_chip | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 9.14 | 17.93 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_dram | 8 | 51.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 39.44 | 77.33 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_on_chip | 16 | 62.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 18.28 | 29.49 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a8 | weights_dram | 16 | 62.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 78.88 | 127.22 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 1.19 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 8.76 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 2.38 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 17.53 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_on_chip | 4 | 27.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.76 | 17.63 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_dram | 4 | 27.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 35.06 | 129.84 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_on_chip | 8 | 53.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 9.52 | 17.96 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_dram | 8 | 53.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 70.11 | 132.29 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_on_chip | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 19.04 | 29.29 |
| estimate (Horowitz 45 nm) | sudoku_attn | w8a16 | weights_dram | 16 | 65.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 140.22 | 215.73 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 1.16 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.95 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 2.32 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 9.89 | inf (0 solved) |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_on_chip | 4 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 4.64 | 20.18 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_dram | 4 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 19.79 | 86.04 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_on_chip | 8 | 50.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 9.28 | 18.56 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_dram | 8 | 50.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 39.58 | 79.15 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_on_chip | 16 | 63.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 18.56 | 29.46 |
| estimate (Horowitz 45 nm) | sudoku_attn | w4a16 | weights_dram | 16 | 63.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-99 (n=100) | 79.15 | 125.64 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 183.77 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 205.41 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_on_chip | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 367.54 | 3062.82 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_dram | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 410.82 | 3423.49 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_on_chip | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 735.08 | 854.74 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_dram | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 821.64 | 955.39 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_on_chip | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 1470.15 | 1670.63 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_dram | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 1643.27 | 1867.36 |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 2940.31 | - |
| estimate (Horowitz 45 nm) | maze | fp32 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 3286.55 | - |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 60.04 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 70.86 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_on_chip | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 120.07 | 1000.60 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_dram | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 141.71 | 1180.93 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_on_chip | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 240.14 | 279.24 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_dram | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 283.42 | 329.56 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_on_chip | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 480.29 | 545.78 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_dram | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 566.85 | 644.15 |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 960.58 | - |
| estimate (Horowitz 45 nm) | maze | fp16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 1133.70 | - |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_on_chip | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 9.32 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_dram | 1 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 14.73 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_on_chip | 2 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 18.63 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_dram | 2 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 29.45 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_on_chip | 4 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 37.27 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_dram | 4 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 58.91 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_on_chip | 8 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 74.53 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_dram | 8 | 0.0 | QNN HTP on Samsung Galaxy S24 | ON DEVICE Samsung_Galaxy_S24 (n=50) | 117.81 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_on_chip | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 149.06 | - |
| estimate (Horowitz 45 nm) | maze | w8a8 | weights_dram | 16 | - | - | on-device chain ran fewer steps; host runtime not substituted | 235.62 | - |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 9.30 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 12.00 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_on_chip | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 18.59 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_dram | 2 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 24.00 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_on_chip | 4 | 72.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 37.18 | 51.64 |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_dram | 4 | 72.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 48.00 | 66.67 |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_on_chip | 8 | 84.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 74.36 | 88.52 |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_dram | 8 | 84.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 96.00 | 114.29 |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 148.72 | - |
| estimate (Horowitz 45 nm) | maze | w4a8 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 192.00 | - |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 9.43 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 14.84 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_on_chip | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 18.87 | 157.23 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_dram | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 29.69 | 247.39 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_on_chip | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 37.73 | 43.88 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_dram | 4 | 86.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 59.37 | 69.04 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_on_chip | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 75.47 | 85.76 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_dram | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 118.75 | 134.94 |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 150.94 | - |
| estimate (Horowitz 45 nm) | maze | w8a16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 237.50 | - |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 9.41 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 12.12 | inf (0 solved) |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_on_chip | 2 | 10.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 18.82 | 188.25 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_dram | 2 | 10.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 24.23 | 242.35 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_on_chip | 4 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 37.65 | 42.78 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_dram | 4 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 48.47 | 55.08 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_on_chip | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 75.30 | 85.57 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_dram | 8 | 88.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, test 0-49 (n=50) | 96.94 | 110.16 |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 150.60 | - |
| estimate (Horowitz 45 nm) | maze | w4a16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 193.88 | - |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 183.78 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 205.42 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_on_chip | 2 | 13.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 367.56 | 2827.42 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_dram | 2 | 13.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 410.84 | 3160.34 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_on_chip | 4 | 21.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 735.13 | 3500.61 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_dram | 4 | 21.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 821.69 | 3912.80 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_on_chip | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 1470.26 | 6392.43 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_dram | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 1643.38 | 7145.12 |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 2940.52 | - |
| estimate (Horowitz 45 nm) | arc | fp32 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 3286.76 | - |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 60.04 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 70.86 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_on_chip | 2 | 13.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 120.08 | 923.70 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_dram | 2 | 13.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 141.72 | 1090.16 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_on_chip | 4 | 21.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 240.16 | 1143.62 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_dram | 4 | 21.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 283.44 | 1349.72 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_on_chip | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 480.32 | 2088.35 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_dram | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 566.88 | 2464.70 |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 960.64 | - |
| estimate (Horowitz 45 nm) | arc | fp16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 1133.76 | - |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 9.32 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 14.73 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_on_chip | 2 | 2.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 18.63 | 931.70 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_dram | 2 | 2.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 29.45 | 1472.70 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_on_chip | 4 | 7.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 37.27 | 532.40 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_dram | 4 | 7.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 58.91 | 841.54 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_on_chip | 8 | 14.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 74.54 | 532.40 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_dram | 8 | 14.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 117.82 | 841.54 |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 149.07 | - |
| estimate (Horowitz 45 nm) | arc | w8a8 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 235.63 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_on_chip | 1 | - | - | no device-faithful accuracy for this graph yet | 9.30 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_dram | 1 | - | - | no device-faithful accuracy for this graph yet | 12.00 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_on_chip | 2 | - | - | no device-faithful accuracy for this graph yet | 18.59 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_dram | 2 | - | - | no device-faithful accuracy for this graph yet | 24.00 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_on_chip | 4 | - | - | no device-faithful accuracy for this graph yet | 37.18 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_dram | 4 | - | - | no device-faithful accuracy for this graph yet | 48.00 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_on_chip | 8 | - | - | no device-faithful accuracy for this graph yet | 74.37 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_dram | 8 | - | - | no device-faithful accuracy for this graph yet | 96.01 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 148.73 | - |
| estimate (Horowitz 45 nm) | arc | w4a8 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 192.01 | - |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_on_chip | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 9.43 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_dram | 1 | 0.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 14.84 | inf (0 solved) |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_on_chip | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 18.87 | 157.24 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_dram | 2 | 12.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 29.69 | 247.40 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_on_chip | 4 | 22.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 37.74 | 171.53 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_dram | 4 | 22.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 59.38 | 269.90 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_on_chip | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 75.47 | 328.15 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_dram | 8 | 23.0 | ONNX Runtime CPU, fused integer QDQ kernels (host sim) | host sim of compiled QDQ graph, 100 canonical rows, token exact (n=100) | 118.75 | 516.32 |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 150.95 | - |
| estimate (Horowitz 45 nm) | arc | w8a16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 237.51 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_on_chip | 1 | - | - | no device-faithful accuracy for this graph yet | 9.41 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_dram | 1 | - | - | no device-faithful accuracy for this graph yet | 12.12 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_on_chip | 2 | - | - | no device-faithful accuracy for this graph yet | 18.83 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_dram | 2 | - | - | no device-faithful accuracy for this graph yet | 24.24 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_on_chip | 4 | - | - | no device-faithful accuracy for this graph yet | 37.65 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_dram | 4 | - | - | no device-faithful accuracy for this graph yet | 48.47 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_on_chip | 8 | - | - | no device-faithful accuracy for this graph yet | 75.30 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_dram | 8 | - | - | no device-faithful accuracy for this graph yet | 96.94 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_on_chip | 16 | - | - | no device-faithful accuracy for this graph yet | 150.61 | - |
| estimate (Horowitz 45 nm) | arc | w4a16 | weights_dram | 16 | - | - | no device-faithful accuracy for this graph yet | 193.89 | - |
