### Return test: scoring rules, fixed before any record was read (Sep 26 01:31 UTC)
Jobs started 01:28 UTC; the only numbers seen so far are two DT maze13 log lines (w8c, w4c: return 1.000, no loss),
which cannot move any rule below. All scoring runs from the written JSON records (`x2_return_analyze.py`).
- **Unit** = (model, cell). Retained = acc_q / acc_f on the same n examples. Collapse = retained < 0.5. Cells whose fp32
  accuracy on the subset is < 10% are excluded (retained is noise there).
- **Cell rho** (primary) = median of per-example rho = 1/t_c over the fp32-correct examples; median over all examples
  reported as secondary. Cell phi = mean phi over the same examples (lower phi = more collapse).
- **P1 supported** only if all hold on the pooled weight/activation-quant and weight-noise cells: (a) the fixed rule
  "collapse iff median rho > 1" has balanced accuracy >= 0.80; (b) AUROC(rho) > AUROC(-phi); (c) leave-one-family-out
  (threshold fitted on the other families, applied to the held-out one) accuracy for rho >= that for phi; (d) Spearman
  (retained, t=1 return rate) >= 0.7. Partly = some hold; falsified = (a) fails. Fresh-e cells (anoise) are scored in a
  separate row, as are per-family results. Per-example agreement: among fp32-correct examples, fraction where
  ret1 == corr_q, and Cohen's kappa, per family.
- **P2**: observed gain = acc(M1 schedule c16.P8, polish fp32) - acc(c16) on M1's first 256 puzzles (offset 0; same
  examples as mine); predicted = mean(ret1 x (corr_f - corr_q)). Matched only where M1's c16 per-example correctness
  equals my corr_q on >= 98% of examples (same compressed model). Supported if |pred - obs| <= max(2 points, 25% of obs)
  on >= 80% of matched cells. Caveat fixed now: M1 polishes 8 fp32 steps, the return test runs 16.
- **P3**: SR draws k = 0..15 (M2's seed-0 draws) for sudoku_mlp w4c, sudoku_attn w3c, hrm_sudoku w3c, n = 256.
  Head pick = argmax over draws of the halting logit (qhalt) per example. Supported if the picked draw's t_c exceeds
  the median-over-draws t_c on more examples than the reverse (sign test p < 0.05) in each model, AND across M2's SR
  cells the best-Q@16 gain (from `results/m2/cells/*sr0.npz`, first 256) is highest at intermediate RTN return rates
  (0.2-0.8) and lower at both ends.
- **P4**: parameter-free radius prediction sigma_1/2 ~= 0.1 x median t_c at noise0.1 over fp32-correct examples (grid cap
  16 -> censored at 1.6), plus the 1-parameter version (one pooled scale). Supported if the 1-parameter fit's pred/meas
  bias by mixer is within [0.9, 1.1] for both MLP and attention (S: 1.37/0.64; S_mix: 1.19/0.76) on the same 15-model
  fit set, and held-out new-seed ratios are no worse than S_mix's (median x1.16).
