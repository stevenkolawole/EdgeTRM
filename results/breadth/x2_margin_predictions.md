## Contraction loss vs readout margin; P3 bug check (coordinator, Steven's questions; written before any new number)
**Written:** 2026-09-26 02:47 UTC. Inspired by the P1 result, so these are scored ONLY on records produced
after this entry (`retm_*` margin jobs, the `retk_*` k-sweep, the p4d replicate), never on the P1 records.
**(1) P3 bug check (diagnostic, existing SR records):** on each SR model, count per example how often the argmax over
draws of the halting logit is tied (top-2 within 1e-3) or saturated (all 16 draws above the logit where sigmoid
rounds to 1 in fp16, 7.6); then re-score the head pick against the picked draw's correctness and phi vs the median
draw (sign tests). No prediction is scored here; it is a check for a bug.
**(2) Two mechanisms for "collapse inside the exact basin".** Measured per cell by `x2_margin.py` (label-free):
late contraction ratio r = median over the last 3 steps of ||z_{t+1} - z_t|| / ||z_t - z_{t-1}|| on the compressed
trajectory, the compressed residual ||F_q(z_T) - z_T|| / ||z_T||, the same two for the fp32 trajectory (L_0), and the
readout margin t_r = the first t on a grid (0.1 ... 128) along z* + t (z*_q - z*) at which the decoded answer (k = 0,
no further steps) differs from the fp32 answer at z*. Cell-level values are medians over fp32-correct examples.
- (A) CONTRACTION LOSS predicts: cell r >= 1 in >= 80% of collapsed cells and r < 1 in >= 80% of held cells.
- (B) READOUT MARGIN predicts: r < 1 in >= 80% of collapsed cells as well, and the zero-parameter width
  sigma_1/2 ~= sigma_0 x median t_r(sigma_0) with sigma_0 = 0.05 (weight noise) removes the mixer bias: on the 6
  held-out new seeds, pred/meas bias within [0.9, 1.1] for MLP and attention and median |log ratio| smaller than
  S's (x1.39) and S_mix's (x1.16). Also reported: the fit-set bias and a 1-parameter version.
Scored as: A supported if both its clauses hold; B supported if both its clauses hold; otherwise partly / falsified
per clause. If neither wins, reported as such.
