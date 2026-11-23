# Papers Steven sent on energy and cost accounting (read Nov 23)

Read status per paper; grades follow ~/workspace/guides/LITERATURE.md. Numbers are the papers' own (measured there).

| Paper | Venue, grade | Read | What it establishes | What it gives us |
|---|---|---|---|---|
| MLPerf Power (Tschand, ..., Kanter, Reddi) | IEEE Micro Top Picks 2026; anchor-level venue | full | One methodology across 9 orders of magnitude: hold *what* and *when* constant, vary *how*. Energy is nonlinear in accelerators (512 GPUs: -93% time, 3.9x energy); DLRM 2.3x ResNet energy at 1.06x MACs; MoE loses to dense at inference | "The energy cost of a useful answer can no longer be read off a single forward pass"; "mobile inference is next to be standardized". Methodology framing for our microcontroller-to-H200 span |
| Joules-to-GDP (Zeinali, Saad-Falcon, Narayan, ..., Hennessy, Re, Mirhoseini) | arXiv 2610.30858, Stanford; support | intro | Net value per joule on 220 GDPval tasks; the most capable model can give less value per joule (Gemma 4 31B: 56% of GLM 5.2's value at 23% of its energy); 8 H100s use 3.2x energy for 22% less time | Unit of account moves to useful output per joule |
| Energy cost of quantized LM agents (Abkin et al.) | arXiv 2610.32417, Texas A&M; support | intro | 4-bit saves energy per token but costs up to 3.5x more per completed task (more turns) | Per-step savings can reverse per answer; our looped version is starker (0 correct answers) |
| Intelligence per Joule as a post-training objective (Wang, Davion) | Stanford CS224R class report; caution | full (short) | Harness choice changes IPJ up to 54x (median 3x); tokens are a lossy proxy for energy | Motivation only |
| Solve the Loop: Attractor Models (USC) | arXiv 2605.12466; support | intro | 770M looped model beats a 1.3B transformer on 2x tokens; 27M model 91.4% Sudoku-Extreme; adaptive iterations by convergence; "costly to optimize and deploy" | Looped models keep scaling; equilibrium internalization may shorten loops (read in full before relying) |
| LFM3 (Liquid AI blog, Oct 27 2026) | industry blog; caution as evidence | full | 1.7B on-device agent models from hardware-in-the-loop search over 14 device/backend pairs incl. Hexagon NPU; Qualcomm, AMD, Ambarella quoted | Industry moving to on-device agents; "tokens are free" ignores battery |
| FastTTS (Chen, ..., Luk, Fan) | ASPLOS 2026 (ACM reference in PDF); anchor | intro, setup, results, ablations | Verifier-guided tree search on one RTX 4090 (24 GB), 3070 Ti, 4070 Ti: 2.2x goodput, 38-68% lower latency than vLLM; edge = consumer GPU; problem framed as the memory gap | Edge test-time compute is a recognized ASPLOS problem, but for sampled token trees on GPUs: KV memory, stragglers, goodput. No energy, no per-input stopping, no correctness, no phones |
| Request and token energy of LLM inference (Vellaisamy, Lam, Blanton, Shen; CMU ECE) | IISWC 2026; support (refereed, characterization venue) | intro, model, results, reasoning section | E_request(N) = E_fixed + N x E_step, a local affine fit (large residual at N=10); very long outputs enter a second regime where step energy grows with the KV cache (Llama-3.2-3B: 337 to 1,600 mJ/token from N=128 to 8192). Utility per joule: the accuracy-maximizing reasoning cap never maximizes utility/J; H200 lowers energy 17-28% but leaves the utility-optimal cap unchanged | Closest statement of our cost law, for token loops. Contrast: a latent loop's state does not grow, so per-step energy stays constant (our law has no intercept: median 0.5% AGX, 1.1% L40S) |
| The Price of Progress (Gundlach, Lynch, Mertens, Thompson; MIT FutureTech) | NeurIPS 2025 workshop; support | full | Price per benchmark level falls 5-10x per year (about 3x algorithmic); cost of running a benchmark stays flat or rises because reasoning uses more tokens | Per-token prices mislead; measure per task. Trend-as-headline style |

## Derived for our paper (inferred from our law, not yet stated in any record)

With no intercept, J per correct answer at n loops on device d is n x e_d / acc(n). The energy-optimal n is the argmax of
acc(n)/n, independent of e_d, so it is the same on every device; the IISWC paper observes the same invariance for
H100 vs H200. Under a latency deadline the reachable n is floor(D / t_d), which does depend on the device
(deadline_accuracy.json). Device speed changes how deep a model can think in time; device efficiency does not change the
depth at which thinking is cheapest per answer.
