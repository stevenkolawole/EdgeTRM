# EdgeTRM inner step on Arm Ethos-U at H=1 (and H=3, same toolchain)

Produced by `mcu_h1.py` / `mcu_h1_fvp.sh` / `mcu_h1_report.py` on p4d (CPU only). Vela 5.2.0; graphs quantized with the AI Edge Quantizer 0.9.0 on real carries. Vela numbers are compiler estimates of NPU cycles; operators left on the CPU are not in them. One recursion step = one supervision step (H x (L+1) applications of the shared block); ms per answer = ms per step x n.

## Headline: INT8 step on Ethos-U55-256 and Ethos-U85-256, H=1 vs H=3

Vela estimates (U55-256: Ethos_U55_High_End_Embedded, Shared_Sram, 500 MHz; U85-256: Ethos_U85_SYS_Flash_High, Shared_Sram, 500 MHz) and, where run, the Corstone-300 FVP NPU cycle count for the same compiled U55 graph.

| task | H | NPU | SRAM KiB | flash KiB | Vela cycles | ms/step | ms/answer n=8 | ms/answer n=16 | FVP NPU cycles | FVP ms/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sudoku_mlp | 1 | U55-256 | 1,220.9 | 5,117.4 | 51,997,258 | 103.99 | 832.0 | 1,663.9 | 55,383,106 | 110.77 |
| sudoku_mlp | 3 | U55-256 | 1,269.4 | 6,430.2 | 155,829,422 | 311.66 | 2,493.3 | 4,986.5 | 166,281,106 | 332.56 |
| sudoku_mlp | 1 | U85-256 | 965.0 | 5,244.0 | 40,699,379 | 81.40 | 651.2 | 1,302.4 | - | - |
| sudoku_mlp | 3 | U85-256 | 1,013.5 | 6,559.8 | 121,934,281 | 243.87 | 1,950.9 | 3,901.9 | - | - |
| sudoku_attn | 1 | U55-256 | 879.1 | 7,250.3 | 162,539,662 | 325.08 | 2,600.6 | 5,201.3 | 287,243,106 (unvalidated: NPU-model errors) | 574.49 |
| sudoku_attn | 3 | U55-256 | 927.6 | 8,839.9 | 487,456,634 | 974.91 | 7,799.3 | 15,598.6 | - | - |
| sudoku_attn | 1 | U85-256 | 838.1 | 7,672.2 | 55,113,823 | 110.23 | 881.8 | 1,763.6 | - | - |
| sudoku_attn | 3 | U85-256 | 886.6 | 9,261.0 | 165,270,487 | 330.54 | 2,644.3 | 5,288.7 | - | - |
| maze | 1 | U55-256 | 55,017.2 | 269,102.3 | 5,896,928,406 | 11,793.86 | 94,350.9 | 188,701.7 | - | - |
| maze | 3 | U55-256 | 55,475.2 | 794,647.7 | 17,689,217,794 | 35,378.44 | 283,027.5 | 566,055.0 | - | - |
| maze | 1 | U85-256 | 54,651.1 | 7,620.3 | 380,836,786 | 761.67 | 6,093.4 | 12,186.8 | - | - |
| maze | 3 | U85-256 | 55,109.1 | 8,755.0 | 1,141,709,938 | 2,283.42 | 18,267.4 | 36,534.7 | - | - |

## INT8 step, H=1 vs H=3, every Vela configuration

| task | accelerator | system config | memory | H | SRAM KiB | flash KiB | DRAM KiB | NPU cycles | ms/step | ms/answer n=8 | ms/answer n=16 | CPU ops |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sudoku_mlp | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 1,169.5 | 0.0 | 0.0 | 208,616,406 | 1,043.08 | 8,344.7 | 16,689.3 | 2 |
| sudoku_mlp | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 1,218.0 | 0.0 | 0.0 | 625,299,710 | 3,126.50 | 25,012.0 | 50,024.0 | 2 |
| sudoku_mlp | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 1,220.9 | 5,123.3 | 0.0 | 121,840,919 | 609.20 | 4,873.6 | 9,747.3 | 2 |
| sudoku_mlp | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 1,269.4 | 6,436.1 | 0.0 | 365,132,591 | 1,825.66 | 14,605.3 | 29,210.6 | 2 |
| sudoku_mlp | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 1,220.9 | 5,118.0 | 0.0 | 60,914,269 | 121.83 | 974.6 | 1,949.3 | 2 |
| sudoku_mlp | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 1,269.4 | 6,430.9 | 0.0 | 182,542,885 | 365.09 | 2,920.7 | 5,841.4 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 1,220.9 | 5,117.4 | 0.0 | 51,997,258 | 103.99 | 832.0 | 1,663.9 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 1,269.4 | 6,430.2 | 0.0 | 155,829,422 | 311.66 | 2,493.3 | 4,986.5 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 1,220.9 | 5,117.8 | 0.0 | 49,436,496 | 98.87 | 791.0 | 1,582.0 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 1,269.4 | 6,430.6 | 0.0 | 148,149,532 | 296.30 | 2,370.4 | 4,740.8 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 1,746.1 | 0.0 | 5,417.2 | 43,836,098 | 87.67 | 701.4 | 1,402.8 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 1,746.1 | 0.0 | 6,730.1 | 131,344,122 | 262.69 | 2,101.5 | 4,203.0 | 2 |
| sudoku_mlp | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 1,122.0 | 0.0 | 5,408.5 | 22,369,180 | 22.37 | 179.0 | 357.9 | 2 |
| sudoku_mlp | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 1,122.0 | 0.0 | 6,721.3 | 66,939,914 | 66.94 | 535.5 | 1,071.0 | 2 |
| sudoku_mlp | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 965.0 | 5,238.4 | 0.0 | 51,693,581 | 258.47 | 2,067.7 | 4,135.5 | 2 |
| sudoku_mlp | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 1,013.5 | 6,554.3 | 0.0 | 154,789,845 | 773.95 | 6,191.6 | 12,383.2 | 2 |
| sudoku_mlp | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 965.0 | 5,244.0 | 0.0 | 40,699,379 | 81.40 | 651.2 | 1,302.4 | 2 |
| sudoku_mlp | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 1,013.5 | 6,559.8 | 0.0 | 121,934,281 | 243.87 | 1,950.9 | 3,901.9 | 2 |
| sudoku_mlp | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 869.3 | 0.0 | 5,534.1 | 15,851,597 | 31.70 | 253.6 | 507.3 | 2 |
| sudoku_mlp | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 869.3 | 0.0 | 6,850.0 | 47,337,060 | 94.67 | 757.4 | 1,514.8 | 2 |
| sudoku_mlp | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 869.3 | 0.0 | 5,536.9 | 7,525,341 | 7.53 | 60.2 | 120.4 | 2 |
| sudoku_mlp | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 869.3 | 0.0 | 6,852.8 | 22,431,411 | 22.43 | 179.5 | 358.9 | 2 |
| sudoku_mlp | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 869.3 | 0.0 | 5,539.2 | 5,897,036 | 5.90 | 47.2 | 94.4 | 2 |
| sudoku_mlp | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 869.3 | 0.0 | 6,855.1 | 17,586,951 | 17.59 | 140.7 | 281.4 | 2 |
| sudoku_attn | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 879.1 | 0.0 | 0.0 | 659,217,718 | 3,296.09 | 26,368.7 | 52,737.4 | 2 |
| sudoku_attn | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 927.6 | 0.0 | 0.0 | 1,977,103,646 | 9,885.52 | 79,084.1 | 158,168.3 | 2 |
| sudoku_attn | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 879.1 | 7,253.2 | 0.0 | 413,918,383 | 2,069.59 | 16,556.7 | 33,113.5 | 2 |
| sudoku_attn | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 927.6 | 8,842.9 | 0.0 | 1,241,364,983 | 6,206.82 | 49,654.6 | 99,309.2 | 2 |
| sudoku_attn | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 879.1 | 7,253.4 | 0.0 | 212,126,169 | 424.25 | 3,394.0 | 6,788.0 | 2 |
| sudoku_attn | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 927.6 | 8,843.0 | 0.0 | 636,178,585 | 1,272.36 | 10,178.9 | 20,357.7 | 2 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 879.1 | 7,250.3 | 0.0 | 162,539,662 | 325.08 | 2,600.6 | 5,201.3 | 2 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 927.6 | 8,839.9 | 0.0 | 487,456,634 | 974.91 | 7,799.3 | 15,598.6 | 2 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 879.1 | 7,250.3 | 0.0 | 160,662,331 | 321.32 | 2,570.6 | 5,141.2 | 2 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 927.6 | 8,839.9 | 0.0 | 481,827,037 | 963.65 | 7,709.2 | 15,418.5 | 2 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 1,677.6 | 0.0 | 7,640.7 | 150,255,978 | 300.51 | 2,404.1 | 4,808.2 | 2 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 1,677.6 | 0.0 | 9,278.8 | 450,603,762 | 901.21 | 7,209.7 | 14,419.3 | 2 |
| sudoku_attn | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 782.1 | 0.0 | 7,646.7 | 147,038,004 | 147.04 | 1,176.3 | 2,352.6 | 2 |
| sudoku_attn | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 782.1 | 0.0 | 9,284.8 | 440,946,386 | 440.95 | 3,527.6 | 7,055.1 | 2 |
| sudoku_attn | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 838.1 | 7,668.7 | 0.0 | 65,511,327 | 327.56 | 2,620.5 | 5,240.9 | 2 |
| sudoku_attn | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 886.6 | 9,257.5 | 0.0 | 196,407,299 | 982.04 | 7,856.3 | 15,712.6 | 2 |
| sudoku_attn | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 838.1 | 7,672.2 | 0.0 | 55,113,823 | 110.23 | 881.8 | 1,763.6 | 2 |
| sudoku_attn | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 886.6 | 9,261.0 | 0.0 | 165,270,487 | 330.54 | 2,644.3 | 5,288.7 | 2 |
| sudoku_attn | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 741.1 | 0.0 | 7,972.4 | 19,313,276 | 38.63 | 309.0 | 618.0 | 2 |
| sudoku_attn | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 741.1 | 0.0 | 10,275.4 | 57,765,769 | 115.53 | 924.3 | 1,848.5 | 2 |
| sudoku_attn | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 741.1 | 0.0 | 7,253.3 | 9,768,722 | 9.77 | 78.1 | 156.3 | 2 |
| sudoku_attn | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 741.1 | 0.0 | 8,842.1 | 29,191,796 | 29.19 | 233.5 | 467.1 | 2 |
| sudoku_attn | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 741.1 | 0.0 | 7,254.5 | 6,379,564 | 6.38 | 51.0 | 102.1 | 2 |
| sudoku_attn | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 741.1 | 0.0 | 8,843.2 | 19,064,383 | 19.06 | 152.5 | 305.0 | 2 |
| maze | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 55,017.2 | 0.0 | 0.0 | 29,606,393,429 | 148,031.97 | 1,184,255.7 | 2,368,511.5 | 2 |
| maze | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 55,475.2 | 0.0 | 0.0 | 88,814,219,613 | 444,071.10 | 3,552,568.8 | 7,105,137.6 | 2 |
| maze | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 55,017.2 | 269,102.2 | 0.0 | 16,341,676,406 | 81,708.38 | 653,667.1 | 1,307,334.1 | 2 |
| maze | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 55,475.2 | 794,647.5 | 0.0 | 49,021,562,758 | 245,107.81 | 1,960,862.5 | 3,921,725.0 | 2 |
| maze | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 55,017.2 | 269,102.2 | 0.0 | 8,448,755,880 | 16,897.51 | 135,180.1 | 270,360.2 | 2 |
| maze | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 55,475.2 | 794,647.5 | 0.0 | 25,344,461,936 | 50,688.92 | 405,511.4 | 811,022.8 | 2 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 55,017.2 | 269,102.3 | 0.0 | 5,896,928,406 | 11,793.86 | 94,350.9 | 188,701.7 | 2 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 55,475.2 | 794,647.7 | 0.0 | 17,689,217,794 | 35,378.44 | 283,027.5 | 566,055.0 | 2 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| maze | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 55,017.2 | 269,102.3 | 0.0 | 5,820,546,555 | 11,641.09 | 93,128.7 | 186,257.5 | 2 |
| maze | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 55,475.2 | 794,647.7 | 0.0 | 17,460,253,783 | 34,920.51 | 279,364.1 | 558,728.1 | 2 |
| maze | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 54,101.2 | 0.0 | 296,267.3 | 6,148,044,539 | 12,296.09 | 98,368.7 | 196,737.4 | 2 |
| maze | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 54,101.2 | 0.0 | 822,270.6 | 18,442,654,087 | 36,885.31 | 295,082.5 | 590,164.9 | 2 |
| maze | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 54,101.2 | 0.0 | 296,267.9 | 6,869,478,853 | 6,869.48 | 54,955.8 | 109,911.7 | 2 |
| maze | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 54,101.2 | 0.0 | 822,271.2 | 20,605,392,209 | 20,605.39 | 164,843.1 | 329,686.3 | 2 |
| maze | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 54,651.1 | 7,620.2 | 0.0 | 747,120,889 | 3,735.60 | 29,884.8 | 59,769.7 | 2 |
| maze | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 55,109.1 | 8,755.0 | 0.0 | 2,239,850,901 | 11,199.25 | 89,594.0 | 179,188.1 | 2 |
| maze | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 54,651.1 | 7,620.3 | 0.0 | 380,836,786 | 761.67 | 6,093.4 | 12,186.8 | 2 |
| maze | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 55,109.1 | 8,755.0 | 0.0 | 1,141,709,938 | 2,283.42 | 18,267.4 | 36,534.7 | 2 |
| maze | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 53,735.1 | 0.0 | 10,786.8 | 201,208,718 | 402.42 | 3,219.3 | 6,438.7 | 2 |
| maze | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 53,735.1 | 0.0 | 11,921.5 | 601,863,682 | 1,203.73 | 9,629.8 | 19,259.6 | 2 |
| maze | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 53,735.1 | 0.0 | 10,786.9 | 152,920,894 | 152.92 | 1,223.4 | 2,446.7 | 2 |
| maze | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 53,735.1 | 0.0 | 11,921.7 | 457,648,315 | 457.65 | 3,661.2 | 7,322.4 | 2 |
| maze | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 53,735.1 | 0.0 | 10,787.0 | 82,396,664 | 82.40 | 659.2 | 1,318.3 | 2 |
| maze | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 53,735.1 | 0.0 | 11,921.8 | 246,462,150 | 246.46 | 1,971.7 | 3,943.4 | 2 |

## W8A16 step, H=1 vs H=3, every Vela configuration

| task | accelerator | system config | memory | H | SRAM KiB | flash KiB | DRAM KiB | NPU cycles | ms/step | ms/answer n=8 | ms/answer n=16 | CPU ops |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sudoku_mlp | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 2,339.0 | 0.0 | 0.0 | 361,263,197 | 1,806.32 | 14,450.5 | 28,901.1 | 2 |
| sudoku_mlp | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 2,436.0 | 0.0 | 0.0 | 1,083,194,353 | 5,415.97 | 43,327.8 | 86,655.5 | 2 |
| sudoku_mlp | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 2,390.4 | 5,166.6 | 0.0 | 189,888,566 | 949.44 | 7,595.5 | 15,191.1 | 2 |
| sudoku_mlp | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 2,487.4 | 6,565.3 | 0.0 | 569,039,430 | 2,845.20 | 22,761.6 | 45,523.2 | 2 |
| sudoku_mlp | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 2,390.4 | 5,166.6 | 0.0 | 96,176,408 | 192.35 | 1,538.8 | 3,077.6 | 2 |
| sudoku_mlp | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 2,487.4 | 6,565.3 | 0.0 | 288,211,248 | 576.42 | 4,611.4 | 9,222.8 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 2,390.4 | 5,152.4 | 0.0 | 88,520,760 | 177.04 | 1,416.3 | 2,832.7 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 2,487.4 | 6,551.1 | 0.0 | 265,269,320 | 530.54 | 4,244.3 | 8,488.6 | 2 |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_mlp | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 2,390.4 | 5,151.9 | 0.0 | 79,629,219 | 159.26 | 1,274.1 | 2,548.1 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 2,487.4 | 6,550.6 | 0.0 | 238,608,307 | 477.22 | 3,817.7 | 7,635.5 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 2,196.4 | 0.0 | 5,608.1 | 79,632,707 | 159.27 | 1,274.1 | 2,548.2 | 2 |
| sudoku_mlp | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 2,196.4 | 0.0 | 7,006.8 | 238,579,157 | 477.16 | 3,817.3 | 7,634.5 | 2 |
| sudoku_mlp | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 2,194.5 | 0.0 | 5,597.2 | 44,611,630 | 44.61 | 356.9 | 713.8 | 2 |
| sudoku_mlp | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 2,194.5 | 0.0 | 6,995.9 | 133,510,338 | 133.51 | 1,068.1 | 2,136.2 | 2 |
| sudoku_mlp | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 1,878.5 | 5,298.9 | 0.0 | 92,666,624 | 463.33 | 3,706.7 | 7,413.3 | 2 |
| sudoku_mlp | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 1,975.5 | 6,701.5 | 0.0 | 277,490,132 | 1,387.45 | 11,099.6 | 22,199.2 | 2 |
| sudoku_mlp | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 1,878.5 | 5,285.1 | 0.0 | 55,453,635 | 110.91 | 887.3 | 1,774.5 | 2 |
| sudoku_mlp | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 1,975.5 | 6,687.7 | 0.0 | 166,082,989 | 332.17 | 2,657.3 | 5,314.7 | 2 |
| sudoku_mlp | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 1,685.8 | 0.0 | 5,720.8 | 23,308,506 | 46.62 | 372.9 | 745.9 | 2 |
| sudoku_mlp | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 1,685.8 | 0.0 | 7,123.4 | 69,513,817 | 139.03 | 1,112.2 | 2,224.4 | 2 |
| sudoku_mlp | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 1,685.8 | 0.0 | 5,733.3 | 14,868,113 | 14.87 | 118.9 | 237.9 | 2 |
| sudoku_mlp | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 1,685.8 | 0.0 | 7,136.0 | 44,326,392 | 44.33 | 354.6 | 709.2 | 2 |
| sudoku_mlp | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 1,685.8 | 0.0 | 5,738.8 | 9,083,567 | 9.08 | 72.7 | 145.3 | 2 |
| sudoku_mlp | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 1,685.8 | 0.0 | 7,141.4 | 27,040,665 | 27.04 | 216.3 | 432.7 | 2 |
| sudoku_attn | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 1,164.0 | 0.0 | 0.0 | 404,251,541 | 2,021.26 | 16,170.1 | 32,340.1 | 30 |
| sudoku_attn | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 1,261.0 | 0.0 | 0.0 | 1,212,159,385 | 6,060.80 | 48,486.4 | 96,972.8 | 86 |
| sudoku_attn | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 1,376.4 | 7,306.3 | 0.0 | 262,951,262 | 1,314.76 | 10,518.1 | 21,036.1 | 30 |
| sudoku_attn | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 1,473.4 | 8,958.7 | 0.0 | 788,227,518 | 3,941.14 | 31,529.1 | 63,058.2 | 86 |
| sudoku_attn | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 1,376.4 | 7,306.3 | 0.0 | 140,200,458 | 280.40 | 2,243.2 | 4,486.4 | 30 |
| sudoku_attn | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 1,473.4 | 8,958.7 | 0.0 | 420,283,398 | 840.57 | 6,724.5 | 13,449.1 | 86 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 1,252.7 | 7,302.5 | 0.0 | 133,499,288 | 267.00 | 2,136.0 | 4,272.0 | 30 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 1,349.7 | 8,954.9 | 0.0 | 400,204,904 | 800.41 | 6,403.3 | 12,806.6 | 86 |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_attn | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 1,252.7 | 7,302.5 | 0.0 | 128,443,299 | 256.89 | 2,055.1 | 4,110.2 | 30 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 1,349.7 | 8,954.9 | 0.0 | 385,050,547 | 770.10 | 6,160.8 | 12,321.6 | 86 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 2,147.1 | 0.0 | 8,030.6 | 105,804,525 | 211.61 | 1,692.9 | 3,385.7 | 30 |
| sudoku_attn | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 2,147.1 | 0.0 | 9,780.0 | 317,034,969 | 634.07 | 5,072.6 | 10,145.1 | 86 |
| sudoku_attn | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 1,213.0 | 0.0 | 8,031.1 | 73,677,144 | 73.68 | 589.4 | 1,178.8 | 30 |
| sudoku_attn | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 1,213.0 | 0.0 | 9,780.6 | 220,649,490 | 220.65 | 1,765.2 | 3,530.4 | 86 |
| sudoku_attn | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 1,448.6 | 7,729.9 | 0.0 | 119,804,946 | 599.02 | 4,792.2 | 9,584.4 | 2 |
| sudoku_attn | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 1,545.6 | 9,382.3 | 0.0 | 359,121,302 | 1,795.61 | 14,364.9 | 28,729.7 | 2 |
| sudoku_attn | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 1,338.7 | 7,726.7 | 0.0 | 73,480,707 | 146.96 | 1,175.7 | 2,351.4 | 2 |
| sudoku_attn | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 1,435.7 | 9,379.1 | 0.0 | 220,275,363 | 440.55 | 3,524.4 | 7,048.8 | 2 |
| sudoku_attn | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 1,225.7 | 0.0 | 8,165.2 | 31,737,346 | 63.47 | 507.8 | 1,015.6 | 2 |
| sudoku_attn | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 1,225.7 | 0.0 | 10,531.6 | 94,860,021 | 189.72 | 1,517.8 | 3,035.5 | 2 |
| sudoku_attn | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 1,319.9 | 0.0 | 8,169.0 | 20,412,559 | 20.41 | 163.3 | 326.6 | 2 |
| sudoku_attn | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 1,319.9 | 0.0 | 10,535.7 | 61,021,966 | 61.02 | 488.2 | 976.4 | 2 |
| sudoku_attn | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 970.0 | 0.0 | 7,450.5 | 11,272,362 | 11.27 | 90.2 | 180.4 | 2 |
| sudoku_attn | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 970.0 | 0.0 | 9,103.0 | 33,665,044 | 33.67 | 269.3 | 538.6 | 2 |
| maze | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 1 | 43,552.9 | 0.0 | 0.0 | 3,915,498,980 | 19,577.49 | 156,620.0 | 313,239.9 | 22 |
| maze | ethos-u55-32 | Ethos_U55_Deep_Embedded | Sram_Only | 3 | 44,468.9 | 0.0 | 0.0 | 11,741,295,164 | 58,706.48 | 469,651.8 | 939,303.6 | 62 |
| maze | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 1 | 43,552.9 | 269,250.8 | 0.0 | 3,283,163,946 | 16,415.82 | 131,326.6 | 262,653.1 | 22 |
| maze | ethos-u55-64 | Ethos_U55_Deep_Embedded | Shared_Sram | 3 | 44,468.9 | 794,841.2 | 0.0 | 9,843,996,938 | 49,219.98 | 393,759.9 | 787,519.8 | 62 |
| maze | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 43,552.9 | 269,250.8 | 0.0 | 2,047,643,511 | 4,095.29 | 32,762.3 | 65,524.6 | 22 |
| maze | ethos-u55-128 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 44,468.9 | 794,841.2 | 0.0 | 6,140,051,807 | 12,280.10 | 98,240.8 | 196,481.7 | 62 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 1 | 43,552.9 | 269,250.9 | 0.0 | 1,470,686,837 | 2,941.37 | 23,531.0 | 47,062.0 | 22 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Shared_Sram | 3 | 44,468.9 | 794,841.2 | 0.0 | 4,409,419,985 | 8,818.84 | 70,550.7 | 141,101.4 | 62 |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 1 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| maze | ethos-u55-256 | Ethos_U55_High_End_Embedded | Dedicated_Sram | 3 | fails: Invalid configuration of arena_mem_area=OffChipFlash (must be Sram or Dram) | | | | | | | |
| maze | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 1 | 43,552.9 | 269,250.9 | 0.0 | 1,350,281,007 | 2,700.56 | 21,604.5 | 43,209.0 | 22 |
| maze | ethos-u65-256 | Ethos_U65_Embedded | Shared_Sram | 3 | 44,468.9 | 794,841.2 | 0.0 | 4,048,443,515 | 8,096.89 | 64,775.1 | 129,550.2 | 62 |
| maze | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 1 | 39,888.9 | 0.0 | 287,857.1 | 1,246,063,218 | 2,492.13 | 19,937.0 | 39,874.0 | 22 |
| maze | ethos-u65-256 | Ethos_U65_Mid_End | Dedicated_Sram | 3 | 39,888.9 | 0.0 | 814,363.4 | 3,734,878,314 | 7,469.76 | 59,758.1 | 119,516.1 | 62 |
| maze | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 1 | 39,888.9 | 0.0 | 287,857.7 | 1,472,953,235 | 1,472.95 | 11,783.6 | 23,567.3 | 22 |
| maze | ethos-u65-512 | Ethos_U65_High_End | Dedicated_Sram | 3 | 39,888.9 | 0.0 | 814,364.0 | 4,413,828,559 | 4,413.83 | 35,310.6 | 70,621.3 | 62 |
| maze | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 1 | 30,282.9 | 7,771.6 | 0.0 | 1,645,589,901 | 8,227.95 | 65,823.6 | 131,647.2 | 2 |
| maze | ethos-u85-128 | Ethos_U85_SYS_Flash_Low | Shared_Sram | 3 | 31,198.9 | 8,951.9 | 0.0 | 4,934,257,173 | 24,671.29 | 197,370.3 | 394,740.6 | 2 |
| maze | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 1 | 30,282.9 | 7,771.6 | 0.0 | 828,857,150 | 1,657.71 | 13,261.7 | 26,523.4 | 2 |
| maze | ethos-u85-256 | Ethos_U85_SYS_Flash_High | Shared_Sram | 3 | 31,198.9 | 8,951.9 | 0.0 | 2,485,248,670 | 4,970.50 | 39,764.0 | 79,528.0 | 2 |
| maze | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 1 | 28,450.9 | 0.0 | 12,304.1 | 559,776,473 | 1,119.55 | 8,956.4 | 17,912.8 | 2 |
| maze | ethos-u85-512 | Ethos_U85_SYS_DRAM_Low | Dedicated_Sram | 3 | 28,450.9 | 0.0 | 14,208.0 | 1,675,909,758 | 3,351.82 | 26,814.6 | 53,629.1 | 2 |
| maze | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 1 | 28,450.9 | 0.0 | 11,580.6 | 436,836,245 | 436.84 | 3,494.7 | 6,989.4 | 2 |
| maze | ethos-u85-1024 | Ethos_U85_SYS_DRAM_Mid | Dedicated_Sram | 3 | 28,450.9 | 0.0 | 13,484.6 | 1,308,482,414 | 1,308.48 | 10,467.9 | 20,935.7 | 2 |
| maze | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 1 | 28,450.9 | 0.0 | 12,304.3 | 224,712,732 | 224.71 | 1,797.7 | 3,595.4 | 2 |
| maze | ethos-u85-2048 | Ethos_U85_SYS_DRAM_High | Dedicated_Sram | 3 | 28,450.9 | 0.0 | 13,484.6 | 672,799,392 | 672.80 | 5,382.4 | 10,764.8 | 2 |

## Corstone-300 FVP (INT8)

Same Vela-compiled graph the kit embeds; NPU cycles from the Ethos-U PMU; ms at the Vela system-config clock (U55 High-End Embedded 500 MHz, U65 High-End 1 GHz). U65 runs use the kit's U65 default: Dedicated_Sram with a 384 KiB NPU cache, weights and arena in DRAM.

| task | H | NPU | NPU ACTIVE cycles | NPU TOTAL cycles | ms/step @ clock | Vela est. cycles | FVP/Vela | AXI0 rd beats | AXI0 wr beats | AXI1 rd beats | arena B | wall s | NPU-model critical msgs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sudoku_mlp | 1 | u55_256 | 55,382,093 | 55,383,106 | 110.77 @ 500 MHz | 51,997,258 | 1.065 | 23,078,357 | 9,930,244 | 4,323,892 | 1,250,980 | 553 | 0 |
| sudoku_mlp | 3 | u55_256 | 166,279,908 | 166,281,106 | 332.56 @ 500 MHz | 155,829,422 | 1.067 | 69,115,449 | 29,763,636 | 13,048,216 | 1,300,644 | 2,254 | 0 |
| sudoku_mlp | 1 | u65_256 | 108,812,417 | 108,814,148 | 108.81 @ 1,000 MHz | 105,351,493 | 1.033 | 7,211,062 | 3,185,912 | 6,816,948 | 1,148,740 | 691 | 0 |
| sudoku_attn | 1 | u55_256 | 287,242,483 | 287,243,106 | 574.49 @ 500 MHz | 162,539,662 | 1.767 | 120,194,341 | 84,462,226 | 6,472,802 | 900,980 | 2,321 | 2,254 |
| sudoku_attn | 1 | u65_256 | 1,000,719,065 | 1,000,720,148 | 1,000.72 @ 1,000 MHz | 256,195,789 | 3.906 | 12,085,652 | 5,019,702 | 62,661,375 | 625,524 | 4,755 | 2,254 |

H=3 reference (Sep 10, `~/corstone/fvp_sudoku_mlp_npu2.log`, unit-normal calibration): Sudoku-MLP INT8 on U55-256, NPU TOTAL 166,256,106 cycles.
Sudoku-MLP on U55-256, same toolchain and real-carry calibration: H=3 / H=1 FVP NPU cycles = 166,281,106 / 55,383,106 = 3.0024; the H=3 re-run differs from the Sep 10 H=3 run by +25,000 cycles (+0.015%).

## Acceptance matrix (quantizer / TFLite interpreter / Vela / TFLite Micro)

Interpreter: held-out real carries (4 samples), max |logit diff| vs torch fp32 and argmax agreement. Vela: U55-256 High-End Embedded, Shared_Sram. TFLM: arena from the recording allocator; one invoke compared with torch.

| task | H | variant | quantizer | interpreter (max abs / argmax) | Vela | TFLM arena B | TFLM invoke vs torch |
|---|---|---|---|---|---|---:|---|
| sudoku_mlp | 1 | fp32 | - | 0.0001 / 1.000 | - | - | - |
| sudoku_mlp | 1 | int8 | ok | 48.0 / 0.957 | ok, 51,997,258 cyc, 2 CPU ops | 6,811,872 | argmax 0.000, 1 distinct output values |
| sudoku_mlp | 1 | w4a16_c | ok | 19.4 / 0.994 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| sudoku_mlp | 1 | w4a8_allops | ok | 21.5 / 0.981 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_mlp | 1 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_mlp | 1 | w4a8_b128_forced | REFUSED: ValueError: Failed to get quantization parameters for tensor: arith.constant7. Error: Quantized dimension 97 i | - | - | - | - |
| sudoku_mlp | 1 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_mlp | 1 | w4a8_b32_forced | REFUSED: ValueError: Failed to get quantization parameters for tensor: arith.constant7. Error: Quantized dimension 97 i | - | - | - | - |
| sudoku_mlp | 1 | w4a8_c | ok | 54.3 / 0.954 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_mlp | 1 | w4a8_t | ok | 90.6 / 0.910 | FAILS (SIGFPE) | 7,107,200 | argmax 0.000, 1 distinct output values |
| sudoku_mlp | 1 | w8a16 | ok | 17.5 / 0.981 | ok, 88,520,760 cyc, 2 CPU ops | 8,009,440 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 1 | fp32 | - | 0.0001 / 1.000 | - | - | - |
| sudoku_attn | 1 | int8 | ok | 44.3 / 0.985 | ok, 162,539,662 cyc, 2 CPU ops | 9,719,424 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 1 | w4a16_c | ok | 44.3 / 0.978 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| sudoku_attn | 1 | w4a8_allops | ok | 44.4 / 0.981 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_attn | 1 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_attn | 1 | w4a8_b128_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 10,364,080 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 1 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_attn | 1 | w4a8_b32_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 10,364,080 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 1 | w4a8_c | ok | 44.3 / 0.981 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_attn | 1 | w4a8_t | ok | 46.4 / 0.926 | FAILS (SIGFPE) | 10,363,872 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 1 | w8a16 | ok | 44.3 / 0.991 | ok, 133,499,288 cyc, 30 CPU ops | 11,896,512 | argmax 0.000, 1 distinct output values |
| maze | 1 | fp32 | - | 0.0003 / 1.000 | - | - | - |
| maze | 1 | int8 | ok | 20.3 / 1.000 | ok, 5,896,928,406 cyc, 2 CPU ops | 32,153,488 | invoke FAILS: rc=-6;  |
| maze | 1 | w4a16_c | ok | 14.3 / 1.000 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| maze | 1 | w4a8_allops | ok | 14.6 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| maze | 1 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| maze | 1 | w4a8_b128_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 31,703,392 | invoke FAILS: rc=-6;  |
| maze | 1 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| maze | 1 | w4a8_b32_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 31,703,392 | invoke FAILS: rc=-6;  |
| maze | 1 | w4a8_c | ok | 19.6 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| maze | 1 | w4a8_t | ok | 20.3 / 1.000 | FAILS (SIGFPE) | 31,703,184 | invoke FAILS: rc=-6;  |
| maze | 1 | w8a16 | ok | 14.3 / 1.000 | ok, 1,470,686,837 cyc, 22 CPU ops | 56,972,240 | argmax 0.000, 1 distinct output values |
| sudoku_mlp | 3 | fp32 | - | 0.0001 / 1.000 | - | - | - |
| sudoku_mlp | 3 | int8 | ok | 89.5 / 0.935 | ok, 155,829,422 cyc, 2 CPU ops | 8,089,536 | argmax 0.000, 1 distinct output values |
| sudoku_mlp | 3 | w4a16_c | ok | 11.1 / 1.000 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| sudoku_mlp | 3 | w4a8_allops | ok | 12.9 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_mlp | 3 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_mlp | 3 | w4a8_b128_forced | REFUSED: ValueError: Failed to get quantization parameters for tensor: arith.constant7. Error: Quantized dimension 97 i | - | - | - | - |
| sudoku_mlp | 3 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_mlp | 3 | w4a8_b32_forced | REFUSED: ValueError: Failed to get quantization parameters for tensor: arith.constant7. Error: Quantized dimension 97 i | - | - | - | - |
| sudoku_mlp | 3 | w4a8_c | ok | 92.0 / 0.935 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_mlp | 3 | w4a8_t | ok | 115.7 / 0.806 | FAILS (SIGFPE) | 7,331,168 | argmax 0.000, 1 distinct output values |
| sudoku_mlp | 3 | w8a16 | ok | 13.6 / 1.000 | ok, 265,269,320 cyc, 2 CPU ops | 9,386,432 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 3 | fp32 | - | 0.0002 / 1.000 | - | - | - |
| sudoku_attn | 3 | int8 | ok | 12.1 / 1.000 | ok, 487,456,634 cyc, 2 CPU ops | 14,093,552 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 3 | w4a16_c | ok | 12.1 / 1.000 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| sudoku_attn | 3 | w4a8_allops | ok | 12.1 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_attn | 3 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_attn | 3 | w4a8_b128_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 13,477,536 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 3 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| sudoku_attn | 3 | w4a8_b32_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 13,477,536 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 3 | w4a8_c | ok | 14.5 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| sudoku_attn | 3 | w4a8_t | ok | 20.2 / 1.000 | FAILS (SIGFPE) | 13,477,328 | argmax 0.000, 1 distinct output values |
| sudoku_attn | 3 | w8a16 | ok | 12.1 / 1.000 | ok, 400,204,904 cyc, 86 CPU ops | 19,209,392 | argmax 0.000, 1 distinct output values |
| maze | 3 | fp32 | - | 0.0002 / 1.000 | - | - | - |
| maze | 3 | int8 | ok | 80.0 / 1.000 | ok, 17,689,217,794 cyc, 2 CPU ops | 52,918,112 | invoke FAILS: rc=-6;  |
| maze | 3 | w4a16_c | ok | 5.2 / 1.000 | FAILS (SIGFPE) | - | FAILS: Input type: INT16 with filter type: INT4 not supported. |
| maze | 3 | w4a8_allops | ok | 5.3 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| maze | 3 | w4a8_b128 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| maze | 3 | w4a8_b128_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 51,567,536 | invoke FAILS: rc=-6;  |
| maze | 3 | w4a8_b32 | REFUSED: ValueError: Unsupported op for ComputePrecision.INTEGER: TFLOperationName.FULLY_CONNECTED. Error: Quantization | - | - | - | - |
| maze | 3 | w4a8_b32_forced | ok | FAILS: RuntimeError: tflite/kernels/fully_connected.cc:519 filter->quantization.type != kTfLiteAffineQuanti | FAILS (SIGFPE) | 51,567,536 | invoke FAILS: rc=-6;  |
| maze | 3 | w4a8_c | ok | 78.9 / 1.000 | FAILS (SIGFPE) | - | FAILS: tensorflow/lite/micro/kernels/fully_connected_common.cc:99 (input->type == kTfLiteInt8 && filter->ty |
| maze | 3 | w4a8_t | ok | 76.8 / 1.000 | FAILS (SIGFPE) | 51,567,328 | invoke FAILS: rc=-6;  |
| maze | 3 | w8a16 | ok | 2.8 / 1.000 | ok, 4,409,419,985 cyc, 62 CPU ops | 97,476,128 | argmax 0.000, 1 distinct output values |

## INT4 weights on Ethos-U through TOSA 1.0 (real weights: one sudoku_attn block's SwiGLU, gate_up 3072x512 and down 512x1536, 97 tokens, from the H=1 int8 / w4a8_c graphs)

| weights | accelerator | original weight B | NPU-encoded weight B | NPU cycles |
|---|---|---:|---:|---:|
| INT8 | ethos-u55-256 | 2,359,296 | 2,186,528 | 2,258,826 |
| INT8 | ethos-u65-256 | 2,359,296 | 2,186,096 | 4,597,516 |
| INT8 | ethos-u85-256 | 2,359,296 | 2,183,760 | 2,287,132 |
| INT4 | ethos-u55-256 | 2,359,296 | 913,328 | 1,825,764 |
| INT4 | ethos-u65-256 | 2,359,296 | 913,424 | 4,258,551 |
| INT4 | ethos-u85-256 | 2,359,296 | 913,904 | 1,045,962 |

## INT4 weights on Ethos-U through TOSA 1.0 (random weights: 97 tokens, 512->1536->512)

| weights | accelerator | original weight B | NPU-encoded weight B | NPU cycles |
|---|---|---:|---:|---:|
| INT8 | ethos-u55-256 | 1,572,864 | 1,573,312 | 1,615,194 |
| INT8 | ethos-u65-256 | 1,572,864 | 1,573,296 | 1,935,839 |
| INT8 | ethos-u85-256 | 1,572,864 | 1,574,464 | 1,629,127 |
| INT4 | ethos-u55-256 | 1,572,864 | 786,960 | 1,227,363 |
| INT4 | ethos-u65-256 | 1,572,864 | 786,912 | 1,830,840 |
| INT4 | ethos-u85-256 | 1,572,864 | 788,032 | 842,695 |

## Notes

- Maze on U55/U65: Vela's per-layer CSV for maze H=1 INT8 (U55-256, Shared_Sram) puts 82.0% of NPU cycles in the 20 attention BatchMatMuls (916 tokens). The 263 MiB off-chip flash is the compiled graph's 'read_only' buffer (275,560,752 B); the Softmax-lowering Add ops are the ones reading it (2.68e8 flash accesses), consistent with one [8,916,916] 32-bit constant (26.8 MB) per softmax x 10 softmax ops at H=1 (30 at H=3: 776 MiB). On U85-256 (SYS_Flash_High) the same H=1 graph needs 7,620 KiB (7.4 MiB) of flash.
- W8A16 cycles exclude CPU fallbacks: on U55/U65 the attention BATCH_MATMULs with INT16 inputs stay on the CPU (Vela: 'IF Matmul or (FC with dynamic weights): IFM precision must be Int8'; 28 ops for sudoku_attn H=1, 20 for maze H=1), so W8A16 attention rows are not comparable with INT8 rows.
- Vela version sensitivity: the eval kit's Vela 5.1.0 estimates 136.8 M / 332.5 M cycles for the same H=1 Sudoku-MLP / Sudoku-Attn INT8 graphs on U55-256 (5.2.0: 52.0 M / 162.5 M; H=3: 410.3 M / 997.1 M vs 155.8 M / 487.5 M). The FVP runs the 5.2.0-compiled stream (mcu_h1_raw/vela510_crosscheck.json).
- FVP attention caveat: Vela lowers each U55 BATCH_MATMUL into ELEMENTWISE MUL (int8 x int8 -> int32) plus POOL REDUCE_SUM with an int32 IFM and int8 OFM. The FVP's Ethos-U55/U65 model (rev 136b7d75, Apr 2023) prints '[critical] Op=4: Invalid data type, ifm=5 ofm=1' once per such command (2,254 per H=1 Sudoku-Attn step on U55, the exact count in the register stream). The run completes, but its attention cycle count is not a validated measurement. The MLP-mixing graphs contain no matmul and run clean.
- TFLite Micro numerics: every graph TFLM invokes returns a single repeated output value (for sudoku_mlp H=1 INT8, every logit at the output zero point). Traced on sudoku_mlp H=1 INT8 (mcu_ref_kernel_diff.py): the builtin reference kernels, which TFLM shares, get the RMS-norm INT8 REDUCE SUM wrong (exact integer result -127/-128, optimized kernel -127/-128, reference kernel and TFLM -1); earlier ops differ by at most 14 LSB. TFLM arenas are valid; a CPU-only TFLM deployment of these graphs is not numerically valid without a fix to that kernel. Maze invokes abort (SIGABRT, no message).
- INT4 on Ethos-U: no TFLite route (Vela docs: 'Weight tensors must be 8-bit precision' for FULLY_CONNECTED; Vela 5.2.0 crashes with SIGFPE in regor.compile on every graph with an INT4 FULLY_CONNECTED, down to a single Linear; 5.1.0 crashes the same way on the Sep 10 W4A8 graph). The documented route is TOSA (EXT-INT4; CONV2D (i8,i4,i32)), e.g. ExecuTorch's Arm backend; the TOSA tables above show Vela 5.2 accepting INT4 weights. Evidence: mcu_h1_raw/int4/int4_notes.json.
- Ethos_U55_High_End_Embedded + Dedicated_Sram is not a valid Vela configuration (arena_mem_area=OffChipFlash); it fails at every H exactly as in the Sep 10 sweep and is kept for completeness.

