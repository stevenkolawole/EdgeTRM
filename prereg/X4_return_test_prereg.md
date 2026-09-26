## Return test on looped LMs: PRE-REGISTRATION (written 2026-09-26 01:21 UTC, before any return-test number exists)
Question (coordinator / X2 unified mechanism): does the label-free return test give ONE collapse threshold across
looped LMs where latent fidelity φ needs a per-model cut?
Design (Babel L40S only, transformers 4.55.4 overlay, GSM8K test[:100], greedy 512 tokens, batch 16):
- Models and cells: Ouro-1.4B (4 loops) {w8c, w4g32, w4g128, w4a, w4c, w3a, w4t}; Ouro-2.6B (4) {w8c, w4g32, w4a,
  w4c, w3a, w4t}; Huginn (32) {w8c, w4g32, w4c, w3a, w4t}; OLMo-rec (32) {w8c, w4g32, w4c, w4a, w3a, w4t}. Quantizer on
  the looped block only (as everywhere in X4). bf16 = the "exact" map.
- Hand-off loop k = L/2 (Ouro k=2 of 4; Huginn and OLMo k=16 of 32): the state entering loop k.
- (a) Per example: hybrid model = compressed loops 0..k-1, bf16 loops k..L-1, in teacher-forced mode and in greedy
  generation. Return (answer level) := hybrid greedy answer == bf16 greedy answer. Return (teacher forced) := top-1
  agreement with bf16 on the gold-solution positions >= 0.95. Also 1 - cosine of the final latent to bf16's.
- (b) Interpolation: z = z_bf16 + t (z_q - z_bf16) at the hand-off, bf16 loops k..L-1, t in {0, .25, .5, .75, 1, 1.5, 2, 3,
  4}; t_c = first t at which teacher-forced agreement < 0.95 (linear interpolation between grid points; t_c = 4 if never);
  rho = 1/t_c per example; cell rho = median over examples.
- Collapse label = compressed greedy accuracy < 50% of bf16 greedy (same job, same 100 problems).
- P1: rho separates collapsed from held cells at ONE pooled threshold: pooled AUROC and leave-one-model-out
  accuracy of a single rho cut (fit on 3 models, score the 4th), compared with the same statistics for φ (prompt
  final-latent cosine, computed in the same job). P1 holds if pooled AUROC(rho) >= 0.95 and LOMO accuracy(rho) >= 0.95
  and LOMO accuracy(rho) > LOMO accuracy(φ).
- P2: per-example return predicts per-example correctness: among problems bf16 solves, AUROC of (answer-level
  return) and of (1/t_c) for predicting that the compressed model is correct, pooled over cells whose collapse
  label is mixed (neither 0% nor 100% correct). P2 holds if pooled AUROC >= 0.8.
- Scope: if held cells do not return either (return rate at t=1 low for w8c/w4g32), the looped LMs are not in a
  basin regime at k = L/2 and the result is a scope boundary, stated as such.
Code: EdgeTRM/ret_test.py (to be committed with the jobs). No change to these definitions after results exist.
