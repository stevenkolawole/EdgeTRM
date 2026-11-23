Read all six files in full (01 to 06). Method and limits:
- I read the files, not the PDFs, so every claim below is second-hand from the earlier review. Under LITERATURE.md rule 1 none of it is a basis for a claim until checked against the papers.
- The files carry no T1-T4 tiers. They give read depth (FULL / SECTIONS / abstract only), venue, a "novelty risk" label, and a citable-or-not flag. The flag is [POST-JUNE] or [POST-CUTOFF], or "(C)" for citable and "(R)" for rebuttal-only.
- Grades below are my mapping: anchor = refereed A/A*, support = preprint or workshop, caution = weak markers. I did not re-verify venues.

**01 Quantization of recursive models (Sep 24)**
1. Ingolfsson et al., 2607.16237 (ETH, preprint 25 Jun, read FULL). Per-tensor INT4 drives TRM Sudoku from 84.1 to 0.0. Per-block MXInt4 fixes it. They already use "contraction vs drift" and "basin". Highest novelty risk. Support.
2. LoopQ, 2605.16343 (preprint, FULL). Loop-aware PTQ for Ouro, LoopFormer and Parcae. Its evaluation is likelihood and perplexity only. Support.
3. Our own preprint 2606.26488 (abstract and intro only). Its "errors compound across cycles" framing is superseded. Support.
4. Li, Leong, Chaffey monDEQ, 2603.10562 (IEEE Control Systems Letters, FULL). A convergence certificate whose displacement bound is independent of iteration count. Anchor-ish.
5. Lotfi et al., 2606.00206 (Meta, preprint, FULL). Quantized reasoners "cannot stop thinking". Support.
6. Hyperloop, 2604.21254 (preprint, abstract and PTQ section). INT4 GPTQ on looped LMs, perplexity only. Support.
7. Erbas et al., 2609.04490 (post-June, FULL). Storing the recurrent state at 4 bits freezes it. Caution until checked.
8. Zhao et al., ALBERT quantization, 2010.07109 (NLPCC 2020). Support.

**02 Basins and theory**
1. monDEQ 2603.10562 (FULL). Same role as above.
2. Ingolfsson (FULL). They never measure the contraction. Their Sudoku (MLP) vs Maze (attention) contrast confounds task with mixer.
3. Krishna et al., 2609.26487 (post-cutoff, FULL). TRM Jacobian norm is above 1 everywhere; a perturbation decays to 0.10x in attention and stays at 1.91x in MLP.
4. Viakhirev et al., 2608.18222 (AAAI-27 submission, post-cutoff, FULL). Settling, marginal and drifting regimes. Depth is safe if the remaining path length is below the decoder margin.
5. Bai et al., Jacobian regularization, ICML 2021. Anchor.
6. Anil et al., path independence, NeurIPS 2022. Anchor.
7. Lai et al., 2609.04963 (post-cutoff, FULL). Fractal basins.
8. Singh, 2607.11116 (post-cutoff, FULL). Audit showing an "equilibrium" can be the initialization.

**03 Recursive reasoners**
1. TRM, 2510.04871 (FULL). The halting head is a BCE correctness classifier, and all 16 steps run at test time.
2. HRM, 2506.21734 (FULL).
3. PTRM, 2605.19943 (FULL). The Q head is a verifier on Sudoku and weak on Maze-Hard.
4. Corbett et al., 2605.25230 (ICML workshop, FULL). Same split, plus a label-free Q-spread diagnostic.
5. Ren and Liu, 2601.10679. HRM keeps a found answer only about 75% of the time.
6. Popescu et al., 2607.20519 (post-June, FULL). Post-hoc confidence read-outs match or beat learned halting gates.
7. Wang and Reid, 2609.01924 (post-June). Huginn recomputes its state from the input every ~2 recurrences.
8. Ouro 2510.25741, Huginn 2502.05171, STARS 2605.26733 (ICML 2026). Ouro degrades past T=4.
9. SE-RRM 2603.02193 and Schwarzschild/Bansal (NeurIPS 2021/2022). Bansal's Table 7 shows activation perturbations are absorbed.

**04 Compression and test-time compute**
1. Wang et al., 2510.22228 (COLM 2026, FULL). Layer pruning lowers the test-time-scaling ceiling.
2. Liu et al., 2504.04823 (COLM 2025, FULL). Quantized models gain less from longer reasoning.
3. Kim et al., 2510.10964 (ICLR 2026, FULL). Memory saved by 4-bit weights cannot buy back the accuracy as longer generation on small models.
4. Hao et al., 2509.23324 (EuroSys 2026). Per-channel INT4 on the vendor NPU stack kills test-time scaling.
5. Kubwimana and Huang, EdgeReasoning 2511.01866 (IISWC 2025). The latency-budget framing for edge LLMs.
6. Carson and Higham 2018 (SIAM J. Sci. Comput.). Iterative refinement: solver precision sets the rate, residual precision sets the limit.
7. Parcae 2604.12946. Test-time curves fit L(T) = L∞ + Z·e^(−zT).
8. MORES 2607.08116 (post-cutoff). The nearest edge paper; it budgets recurrent steps of a Huginn-style model.

**05 Label-free evaluation**
1. Dutta et al., 2407.09141 (NeurIPS 2024). Flips and KL.
2. Nikolić et al., 2606.19558 (preprint, FULL). Output KL filters damaged builds but does not rank near-baseline ones.
3. Kennedy and Kennedy, 2607.28196 (4 pages, unaffiliated, caution). A data-free fidelity guard passes low-rank builds that misbehave.
4. Schaeffer et al., Mirage, 2304.15004 (NeurIPS 2023). Exact match is roughly p^L.
5. Anil et al., 2211.09961. Path independence as a label-free sanity check.
6. Chen et al., 2609.26708 (post-June, FULL). Teacher-forced training hides generation collapse.
7. Karvonen et al., DiFR 2511.20621. Activation comparison to a reference detects 4-bit builds at AUC above 0.999 from 2 tokens.
8. Kübler et al., 2602.10144 (ICLR 2026). McNemar paired test on per-sample correctness.

**06 Edge hardware**
1. Qualcomm QAIRT/AIMET/AI Hub documentation (fetched). LPBQ needs Hexagon v69 or newer.
2. Arm Vela 5.2.0 / TOSA documentation. No block scales and no CPU fallback.
3. TFLite Micro source. INT4 fully-connected weights are per-tensor only.
4. MLPerf Tiny (NeurIPS 2021 D&B, 2106.07597) and MLPerf Power (2410.12032). The energy protocol.
5. Fanariotis et al., 2509.17533. Alif E7 with Ethos-U55, GPIO-triggered energy windows.
6. MELT, MobiCom 2024 (2403.12844). Monsoon battery-bypass energy on phones and Jetson.
7. MobileLLM, ICML 2024 (2402.14905). Block-wise weight sharing on iPhone: 16.0 ms vs 15.6 ms unshared vs 29.0 ms for a non-shared 60-layer model.
8. SnapFusion, NeurIPS 2023. Latency is linear in diffusion steps.

**2. Framing rules and decisions by Steven recorded in these files**
Almost none are quotes from Steven. Most are the review agent's recommendations or constraints it read from `EdgeTRM/CLAUDE.md` and memory.
- The citation rule, from 01: "Under the BIBLE placement rule ('no post-June citations') these are for awareness and for preempting reviewers, not for citing." File 04: "listed so we know they exist, not as citations to add without Steven's say-so."
- The 5-week MLSys deadline "Oct 30" in 06 has passed. The June cutoff may no longer apply to a MobiSys reframe. That is Steven's call.
- Self-overlap rule (01, 03, 06): state that the preprint's "compounds" claim is withdrawn, and check dual-submission and anonymity.
- Vocabulary to avoid (03 §4): "basin", "phase", "fidelity" and "compressed" collide with published usage.
- "No 'first on-device' claim for test-time scaling in general."
- Evidence discipline: "Never quote rolling-log partials"; the 0.5–1.1% error figure belongs to our own law.

**3. Ideas that could make the story bigger (quotes)**
- 06 §0.9 and §6: "halting as early exit on device is a measurable energy-per-solved-instance lever that nobody has reported." Also: "TRM spends all 16 steps at test time, the halting head is a near-perfect wrongness/rightness signal on Sudoku inside the basin (Q3), so energy per solved instance can drop without any accuracy change."
- 06 §0.5: "No edge NPU runs a data-dependent loop inside the accelerator graph in production... which is also what makes 'latency is linear in iterations' true by construction and halting a host decision."
- 04 K1: "weight quantization acts on the asymptote of test-time scaling, not on its rate." Practical form (E6): a wall-clock deadline frontier, where "under tight deadlines a faster quantizer with a lower cap can win."
- 04 K4: "memory is fixed by the weights and state, and latency is linear in iterations. The deployer's budget is therefore iterations per deadline."
- 02 §5 E: "Give the recipe a pre-deployment check in the monDEQ spirit. On about 256 unlabeled inputs, measure δ_T, ρ and D_lin. Ship if D_lin sits inside the decoder margin... needs no labels." Also: "Depth-safety (settle, then stop) gives a halting rule that stays valid under in-basin compression."
- 05 G2: φ is a release check, "end-to-end agreement of the iterated state... a threshold calibrated against per-cell accuracy... a per-input defer signal". Motivated by "Broken on Arrival" (1.6% of official Ollama GGUF artifacts silently broken) and DiffChaser (TFLite and CoreML conversions introduce disagreements).
- 04 §0.5: in token space "quantization does tax" (more tokens). Latent iteration instead shows "a lowered fixed point, not longer computation." The tax-vs-cap contrast ties the edge cost question to the correctness question.
- 03 mechanism: Huginn re-grounds its state from the input every ~2 recurrences, so "a latent perturbation is washed out, while an emitted token... is where quantization error can compound."
- 06 §0.1 and §0.3: per-SoC integer-only flag (`htp-supports-fp16` absent on 888, 778G, QCS6490, 7 Gen 4 and Q-6690). Grouped INT4 exists only as LPBQ on Hexagon v69+, so the recipe differs by device.

**4. What contradicts or narrows the new review's conclusions**
(Checked against the six new reports; evidence status for each is the earlier files' own reading, mostly abstract level.)
- "Nobody measured energy or on-device cost of looped models." Narrow it; do not drop it.
  - 06 §3.5 says the same for TRM/HRM/DT specifically: 27 and 17 arXiv hits, none with device numbers.
  - Earlier-review items that measured cost of weight-shared or iterated models on devices:
    - MobileLLM-LS (block sharing on iPhone, latency, 16.0 vs 15.6 vs 29.0 ms).
    - SnapFusion and MobileDiffusion (latency per step on phones).
    - Okubo et al., 2401.02721, weight-reused Neural ODE transformer on a ZCU104 FPGA ("9.85× faster", abstract only).
    - Edge-RecViT (CVPR 2026, catalog description only).
    - MORES 2607.08116 (steps budgeted, wireless simulation).
    - TRM on ARC, 2512.11847: 2.4 GB VRAM and 31.3 samples/s on an H100.
  - MELT, MobiCom 2024, measured energy on phones, but for standard LLMs.
  - Only three of these (Okubo, Edge-RecViT, MobileLLM-LS) are directly about on-device cost of weight-shared models. The new reports cite MORES but none of the other five.
- "No pass/fail build check from early loop state exists." Mostly consistent, with three narrowings.
  - DiFR (2511.20621): an activation comparison to a trusted reference detects 4-bit builds at AUC above 0.999 from 2 tokens. It is a label-free build-level detector, though not from loop state and not scored against accuracy. Not in the new reports.
  - The TFLite Quantization Debugger and AI Hub's "compare on-device output with PyTorch output" step are label-free float-vs-quantized checks with no threshold tied to accuracy (the new reports do cite the TFLite debugger).
  - 05 notes the monDEQ certificate (2603.10562) as a "single computable check", and 02 proposes adapting it. Both are proposals, not results.
  - No earlier file reports an AUROC for a build-validity detector from an early internal state. 05 §G2 asks for exactly the threshold "calibrated against per-cell accuracy", so the earlier review agrees this is open.
- "Energy per correct answer on device is unreported." Consistent. The earlier files add the metric definition ("mJ per solved instance", with idle subtraction, from MLPerf Tiny and Fanariotis) and no counterexample. Two cautions:
  - The new review itself flags Wu et al., 2004.11293 ("correctly processed events per energy unit, 3.6x"), which is close to the metric.
  - 04 cites Han et al. 2602.13595, which found that lower precision can raise net energy in sequential reasoning. The new reports lack it.
- "Stopping vs best fixed depth of the same weights is untested." Narrow it. The earlier files contain several comparisons, none on-device and none using the model's correctness head:
  - Popescu 2607.20519: margin read-out reaches 99% at 1.55 loops vs 3.50 for fixed depth on MANO, and Ouro exits give 1.24 to 1.58x latency.
  - Logan 2607.14427 (post-June, SECTIONS): a training-free exit reaches depth-8 quality at 4.94 loops.
  - Dillon 2601.19085 (TRM): halt-first selection is 97.2% against 91.5%, but across an ensemble.
  - Ge et al. 2510.00355: "ACT does not enhance inference performance; running the model for the maximum number of reasoning steps yields the best results." The ARC Prize blog says ACT is "within a few percentage points" of fixed 16 loops. Both argue against halting as an accuracy lever.
  - Fan et al. 2409.15647: max-confidence stopping works for converging tasks and is worse for non-converging ones.
  - The unrecorded corner is energy and device cost at equal accuracy. The earlier files add that stopping is unreliable on mazes, where the residual floors at cycles (Knutson).
- Further narrowings.
  - The earlier review (04 §0.9) calls "first to measure compression against latent test-time iteration" "not defensible" (Ingolfsson EqR depth sweeps, LoopQ App. C.2, our preprint).
  - 06 says the "time, not memory" argument is not defensible for vanilla Ouro: its KV cache grows with loops (MELT, LLA).
  - The halting-head-as-verifier claim is prior art in fp32 (PTRM, Corbett). The new looped_models report found the same overlap with our own Tilted Bowl paper.

**Papers in the earlier files that are not in the new reports** (grep of the new folder by name or id, so a paper can be missing even where its topic appears)
- Not found in the new reports (zero matches):
  - Halting and stopping: PTRM 2605.19943, Corbett 2605.25230, Popescu 2607.20519, Dillon 2601.19085, Kirin 2607.18553, AdRNN 2311.06964, Ge et al. 2510.00355, Fan et al. 2409.15647.
  - Theory: Krishna 2609.26487 (id missing; the name appears twice), Viakhirev 2608.18222, Wang and Reid 2609.01924, Lai 2609.04963, Kim 2608.26556, Zhang 2607.20594, Bai 2021, Scheffer 2009.
  - Dynamics and looped LMs: Schwarzschild 2021, Knutson 2410.03020, SE-RRM 2603.02193, DART 2609.05988, Ren and Liu 2601.10679, LASER 2604.17224, LLA 2607.15456, RD-VLA 2602.07845, Parcae 2604.12946 (name only).
  - Compression and test-time scaling: Wang 2510.22228, Kim 2510.10964, Han 2602.13595, Chen 2609.26708, Lotfi 2606.00206, Erbas 2609.04490, Precision Highway 1812.09818, Hooker 1911.05248, Kinetics 2506.05333.
  - Fidelity and build checks: Nikolić 2606.19558, Kennedy 2607.28196, Schaeffer 2304.15004, "Broken on Arrival" 2609.05881, DiFR, Hendrycks, Geifman, CALM.
  - Edge hardware: LPBQ and `htp-supports-fp16`, MLPerf Tiny and MLPerf Power, Fanariotis 2509.17533, MobileLLM-LS, SnapFusion, Okubo 2401.02721, Edge-RecViT, Tiny Recursive Control 2512.16824, Perfetto/ODPM, INA3221, Joulescope/PPK2. MELT appears, but as a different paper (the new reports use the 2605.07721 "Memory-Efficient Looped Transformer" sense).
- Present in the new reports (thin): Ingolfsson (1 hit), Carson and Higham, monDEQ, LoopQ, Dutta, nn-Meter, TFLite debugger, ML-EXray, MORES, Q-DEQ, EdgeReasoning, FastTTS, Alif, Hyperloop, 2608.09444.

Files read, all under /home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_lit/: 01_quant_recursive.md, 02_basins_theory.md, 03_recursive_reasoners.md, 04_compression_testtime.md, 05_labelfree_eval.md, 06_edge_hardware.md. New reports are in /home/stevenkolawole/workspace/emnlp-26/EdgeTRM/.claude/lit/reports/. I read the first ~15 lines of each new report and grepped the rest; I did not read them in full. A concatenated copy is at /tmp/claude-1000/-home-stevenkolawole-workspace-emnlp-26/3c59b961-66b1-4d0d-abd5-a644e29d970e/scratchpad/new.txt. Nothing was edited.