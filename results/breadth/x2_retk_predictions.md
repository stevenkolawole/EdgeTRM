## Recovery depth and precision of the returning map (coordinator, after P1 was falsified; predictions before any number)
**Written:** 2026-09-26 02:38 UTC, before the k-sweep / w8c jobs exist (script `x2_retk.py`, results `retk_*.json`).
Mechanism as now read: compression damages the MAP; the state stays recoverable (in the exact map's basin).
**Test.** Per cell and example: z*_q = the compressed model's own end state (as in the return test, t = 1). Run the
returning map k steps from z*_q, k in {1, 2, 4, 8, 16} (MDEQ: Broyden budget k), with returning map = fp32, w8c, and
w4c. Recovered = decoded answer equals the fp32 answer from z* after the same k fp32 steps; answer correctness is
recorded as well. Babel and p4d each run the grid on their own; never pooled.
**Predictions (coordinator's, adopted unchanged):**
- K1: on collapsed cells (retained < 0.5), recovery with the fp32 map at k <= 4 in >= 80% of fp32-correct examples
  (scored per collapsed cell at k = 4; supported if the median collapsed cell and >= 80% of collapsed cells meet it,
  and reported at k = 1, 2 as the "few exact steps" method).
- K2: w8c recovery within 2 points of fp32 recovery (per cell, same k; supported if >= 90% of cells at k = 4 and 16 meet it).
**POST HOC (labelled; not a prediction):** what separates collapsed from held cells if not rho. Per example at z*:
the compressed map's one-step residual r_q = ||F_q(z*) - F(z*)|| / ||z*|| (the fixed-e size), ||F_q(z*) - z*|| / ||z*||,
the exact map's gain along that direction g = ||F(z* + u) - F(z*)|| / ||u|| with u = F_q(z*) - F(z*), and the
linear-response fixed-point shift r_q / (1 - g). Compared with collapse (AUROC, pooled and per family) and with S x sigma
from the width law.
