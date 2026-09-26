## Return test on looped LMs (Babel L40S, GSM8K n=100, batch 16)

| model | k | cell | bf16 | compressed | hybrid | retained | collapse | φ | cell ρ (median 1/t_c) | return (tf, t=1) | return (answer) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn | 16 | w8c | 42 | 36 | 41 | 86% |  | 1.000 | 0.25 | 100% | 80% |
| huginn | 16 | w4g32 | 42 | 35 | 39 | 83% |  | 0.978 | 0.25 | 97% | 67% |
| huginn | 16 | w4c | 42 | 33 | 37 | 79% |  | 0.929 | 0.25 | 91% | 55% |
| huginn | 16 | w3a | 42 | 3 | 40 | 7% | yes | 0.782 | 0.25 | 85% | 47% |
| huginn | 16 | w4t | 42 | 2 | 38 | 5% | yes | 0.696 | 0.25 | 89% | 54% |
| olmo_rec32 | 16 | w8c | 41 | 43 | 43 | 105% |  | 1.000 | 0.25 | 100% | 86% |
| olmo_rec32 | 16 | w4g32 | 41 | 42 | 42 | 102% |  | 0.984 | 0.25 | 100% | 87% |
| olmo_rec32 | 16 | w4a | 41 | 34 | 44 | 83% |  | 0.961 | 0.25 | 100% | 89% |
| olmo_rec32 | 16 | w4c | 41 | 35 | 46 | 85% |  | 0.946 | 0.25 | 100% | 83% |
| olmo_rec32 | 16 | w3a | 41 | 2 | 45 | 5% | yes | 0.790 | 0.25 | 100% | 85% |
| olmo_rec32 | 16 | w4t | 41 | 0 | 43 | 0% | yes | 0.114 | 0.25 | 100% | 85% |
| ouro14 | 2 | w8c | 73 | 77 | 79 | 105% |  | 0.997 | 0.25 | 100% | 86% |
| ouro14 | 2 | w4g32 | 73 | 75 | 74 | 103% |  | 0.886 | 0.72 | 73% | 73% |
| ouro14 | 2 | w4g128 | 73 | 69 | 75 | 95% |  | 0.854 | 0.97 | 52% | 68% |
| ouro14 | 2 | w4a | 73 | 55 | 72 | 75% |  | 0.786 | 1.02 | 45% | 62% |
| ouro14 | 2 | w4c | 73 | 7 | 66 | 10% | yes | 0.735 | 1.52 | 22% | 57% |
| ouro14 | 2 | w3a | 73 | 0 | 0 | 0% | yes | 0.181 | 2.29 | 0% | 0% |
| ouro14 | 2 | w4t | 73 | 0 | 0 | 0% | yes | 0.005 | 4.26 | 0% | 0% |
| ouro26 | 2 | w8c | 81 | 81 | 80 | 100% |  | 0.999 | 0.25 | 100% | 87% |
| ouro26 | 2 | w4g32 | 81 | 66 | 73 | 81% |  | 0.923 | 0.46 | 95% | 74% |
| ouro26 | 2 | w4a | 81 | 34 | 39 | 42% | yes | 0.866 | 0.64 | 72% | 36% |
| ouro26 | 2 | w4c | 81 | 78 | 74 | 96% |  | 0.852 | 0.79 | 67% | 73% |
| ouro26 | 2 | w3a | 81 | 0 | 50 | 0% | yes | 0.348 | 1.87 | 2% | 43% |
| ouro26 | 2 | w4t | 81 | 0 | 0 | 0% | yes | 0.011 | 2.71 | 0% | 0% |

### P1: one pooled threshold?

- rho: pooled AUROC 0.700; leave-one-model-out accuracy of one cut 0.708 (huginn 0.60, olmo_rec32 0.67, ouro14 0.71, ouro26 0.83)
- 1-φ: pooled AUROC 0.971; leave-one-model-out accuracy of one cut 0.833 (huginn 1.00, olmo_rec32 0.83, ouro14 0.71, ouro26 0.83)

**P1 FAILS** (criterion: AUROC(ρ) ≥ 0.95, LOMO(ρ) ≥ 0.95, LOMO(ρ) > LOMO(φ)).

### P2: per-example return predicts per-example correctness

cells with mixed compressed correctness among bf16-solved problems: 18 (huginn/w3a, huginn/w4c, huginn/w4g32, huginn/w8c, olmo_rec32/w3a, olmo_rec32/w4a, olmo_rec32/w4c, olmo_rec32/w4g32, olmo_rec32/w8c, ouro14/w4a, ouro14/w4c, ouro14/w4g128, ouro14/w4g32, ouro14/w8c, ouro26/w4a, ouro26/w4c, ouro26/w4g32, ouro26/w8c)
- answer-level return -> compressed correct: AUROC 0.616 (n = 680 correct, 382 wrong)
- t_c (distance to the edge in units of the compression error) -> compressed correct: AUROC 0.573

**P2 FAILS** (criterion: pooled AUROC ≥ 0.8).

### Scope check: do held cells return?

- huginn w4g32: return (tf, t=1) 97%, answer-level 67%, median t_c 4.00
- huginn w8c: return (tf, t=1) 100%, answer-level 80%, median t_c 4.00
- olmo_rec32 w4g32: return (tf, t=1) 100%, answer-level 87%, median t_c 4.00
- olmo_rec32 w8c: return (tf, t=1) 100%, answer-level 86%, median t_c 4.00
- ouro14 w4g32: return (tf, t=1) 73%, answer-level 73%, median t_c 1.38
- ouro14 w8c: return (tf, t=1) 100%, answer-level 86%, median t_c 4.00
- ouro26 w4g32: return (tf, t=1) 95%, answer-level 74%, median t_c 2.16
- ouro26 w8c: return (tf, t=1) 100%, answer-level 87%, median t_c 4.00
