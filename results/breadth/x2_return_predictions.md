## Displaced-attractor mechanism: the RETURN TEST (coordinator, Steven's direction; predictions written before any analysis)
**Written:** 2026-09-26 01:20 UTC, before any return-test number exists; copy committed to EdgeTRM as `results/breadth/x2_return_predictions.md`.
**Mechanism under test.** F = exact step map, F_q = F + e. A fixed e (weights, deterministic weight/activation
quantizers) moves the fixed point to z*_q; the gap is permanent. A fresh e (activation noise redrawn each step)
fluctuates around z* and catches up. Collapse happens exactly when z*_q leaves the exact map's basin of z*.
**Test (label-free).** Per example: z* = fp32 end state, z*_q = the compressed model's own end state (same depth). Run
the fp32 map for k steps from z(t) = z* + t (z*_q - z*), t in a grid; "returns" = the decoded answer equals the fp32
answer from z* after the same k steps (latent distance within eps recorded as a second criterion). Return indicator =
the t = 1 test; t_c = the largest grid t up to which every grid point returns; rho = 1 / t_c.
**Predictions (as given by the coordinator, adopted unchanged):**
- P1 (headline): cell accuracy tracks the per-example return rate, and cells collapse (retained < 50%) at median rho ~ 1
  across ALL families (TRM-MLP, TRM-attn, HRM, DT, MDEQ, DEQ-T, trained seeds, LSQ/naive-QAT), mixers and compression
  types (weight quant, activation quant, weight noise) with ONE threshold; a single pooled rho threshold beats phi
  (whose collapse point is model-specific, 0.58-0.90) on AUROC and on leave-one-family-out.
- P2: M1's polish gain ~ (return rate) x (the fp32-vs-compressed gap on returned examples); null when the displacement
  is tiny (MDEQ-XL); fails outside the basin.
- P3: M2's sampling gain is largest at intermediate return rates, and the halting head picks draws with smaller rho.
- P4: predicting sigma_1/2 from the radius along the compression direction removes the width law's mixer bias
  (which S and S_mix leave: MLP over-, attention under-predicted).
**Disclosure before running.** Existing X2 records already contain a special case of the t = 1 test: the contraction
grid's cross-seeding field (fp32 map run 16 steps from the perturbed end state; `xs_ref_correct`), read out on Sep 24:
it recovered the fp32 answer for 95-99% of fp32-solved Sudoku puzzles through the TRM edge, and for the MLP mixer even
at sigma 0.3 where accuracy is 1%. That observation, if it holds under this protocol, predicts P1 FALSIFIED for the TRM
Sudoku models at k = 16. The protocol below is run as specified regardless, and scored as specified.
