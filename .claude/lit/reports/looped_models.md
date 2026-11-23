DEEP-READ REPORT, CLUSTER A (looped / adaptive-depth). Nothing under ~/workspace was edited. PDFs and text are in .../scratchpad/papers/<id>/.

HEADLINE (read first)
1. Steven is an author of two of the closest papers. 2606.26488 ("What Survives When You Compress a Recursive Reasoner for the Edge?", ML Collective + CMU) and 2609.39277 ("A Tilted Bowl Is Not a Slippery Slope", marked "Under review", the ICLR 2027 submission) both list him. Treat them as own prior work and cite them as such, not as third-party competitors. The Tilted Bowl paper already contains our "controller that stops when the halting head fires, then finishes with 8-bit loops". Its Table 3 reports +up to 15 points over fixed depth at under a third of the weight traffic on Sudoku-Extreme and Maze-Hard. So the halting-plus-repair idea is not novel to the MobiSys paper. What is new there is the measured energy and the phone/Jetson/MCU stack.
2. Nobody else measured energy or on-device cost of looped models. Everything I found is server-GPU (H100, H800, MI250X) or analytic (FLOPs, MB). Three unrelated groups have now published quantization or numerical-robustness work on looped models, so our "compounding errors" line is no longer unclaimed (see Group 3).
3. Every halting or exit comparison in the cluster counts loops or steps. None measures joules. That is the gap our "energy per correct answer" metric fills.

Evidence status: "read" means the PDF text was extracted and the method, experiments and relevant appendix sections were read. For the long papers (Huginn, Ouro, MoR, PLT, PonderNet, Bansal) I read the abstract, intro, method, the exit/efficiency sections and discussion by targeted extraction, not every appendix page. Say "sections" in the ledger for those, not "full". Venue claims come from the paper header unless a source is named.

GROUP 1: the assigned papers

1. TRM, arXiv 2510.04871. Jolicoeur-Martineau, Samsung SAIL Montreal, sole author. Preprint; I did not verify a venue (arXiv v1 Oct 2025). Credibility: refereed record in GANs and diffusion (ICLR/ICML). Authors' code is cited. Slop flags: none. Grade: support. Read in full.
- What it does: 7M-param, 2-layer net, recurses a latent z and answer y, up to N_sup=16 supervision steps. Sudoku-Extreme 87%, Maze-Hard 85%, ARC-AGI-1 45%, ARC-AGI-2 8%.
- Stopping: the halting head is trained with a binary cross-entropy "is y correct" loss. "ACT ... is only used during training, while the full N_sup = 16 supervision steps are done at test time" (this sentence is in HRM-background section 2.5, describing HRM; TRM's own test-time use is not separately stated, so check TRM's code before asserting TRM never halts at inference). TRM's ACT is a training-throughput device, not an inference-energy device.
- Efficiency claims: parameter efficiency only ("less than 0.01% of the parameters"). Memory is discussed as training memory (full backprop through recursion gives OOM at large n). No latency, energy or edge claim.
- Diff: looped; server (L40S for training); accuracy only; compute varies per input only in training; no error detection; no precision change; trained.
- Take: the 16-step test protocol is our "fixed depth" baseline arm. Cite it for the fact that the halting head exists but is unused at test time, so our inference-time use is a new use.

2. HRM, arXiv 2506.21734 (v4, 30 Sep 2026). Wang et al., Sapient Intelligence + Tsinghua. Preprint; venue not verified. Code released (github.com/sapientinc/HRM). Slop flags: none known; ARC Prize's independent analysis (cited in TRM) found deep supervision, not hierarchy, drove gains. Grade: support. Read the main text and ACT section.
- This is the closest prior art to "halting head vs best fixed depth". Its Fig. 5 compares ACT with "a fixed computational step count equivalent to ACT's M_max" on Sudoku-Extreme-Full. Quote: "ACT effectively adapts its computational resources based on task complexity, achieving significant computational savings with minimal impact on performance." The unit is mean compute steps, not energy, and the fixed arm is set at M_max, not swept to find the best fixed M.
- Memory: "maintains a constant memory footprint (O(1) compared to BPTT's O(T))", which is about training backprop.
- Take: cite as the origin of the halting head. Our contribution over it: sweep fixed depth to find the best fixed, report energy, report it per correct answer, and report on-device cases where it gains nothing (HRM and Maze on the Nano).

3. Huginn, arXiv 2502.05171. Geiping et al., Maryland, ELLIS Tübingen, LLNL; Goldstein has a long refereed record. Code and weights released. Venue: I did not verify one (arXiv v2 Feb 2025), so grade support. Read intro, method, the section on adaptive compute and KV sharing, and discussion.
- Quotes: "Latent reasoning models require less memory for training and inference than chain-of-thought reasoning models" (comparison is to CoT). In the broader-impact text: "The architecture we propose is naturally smaller than models scaled by parameter scaling, and this may have broader benefits for the local deployment of these models with commodity chips." That is an unmeasured aspiration. It is the one sentence in the cluster that explicitly anticipates our story.
- Stopping: training-free exit "KL-divergence between two successive steps ... falls below 5x10^-4", zero-shot, per token. Reported MT-Bench 5.63 standard vs 5.56 with early exits. This is a residual/stable-state rule like ours. Section 6.2 also gives zero-shot KV-cache sharing.
- Diff: looped LLM; server (MI250X for training); accuracy; per-token compute, decided by KL; no error detection; no precision change; trained.
- Take: the KL rule is a baseline for our stable1/residual rules. Cite its sentence on local deployment as an acknowledged-but-untested claim.

4. Ouro / LoopLM, arXiv 2510.25741 (v6, 11 Oct 2026). ByteDance Seed, UC Santa Cruz, Mila (Bengio), others. Preprint; venue not verified. Open weights. Grade: support. Read the exit-gate sections, the early-exit and KV-sharing sections.
- Quotes: "1.4B and 2.6B parameter LoopLMs match 4B and 8B standard transformers ... yielding 2-3x parameter-efficiency gains that are critical for deployment in resource constrained environments." They also say "deploying models with hundreds of billions of parameters requires extensive infrastructure". The related-work section calls Megrez2 "a viable path forward for edge LLM deployment with limited memory." No device was measured.
- Stopping: learned exit gate, entropy-regularised objective with a uniform prior, Q-exit rule "terminate at the first step where the cumulative probability crosses q". Fig. 5 on MMLU compares static exit (steps 1-4), a hidden-state-difference threshold, and the gate, by average exit round. The trained gate wins at every budget; the hidden-state-difference rule is "within 1%-2%". Static accuracy goes 40% (1 round), 60% (2), 67.35% (4). This is a gate-vs-fixed comparison in loop units, with a training-free rule nearly matching the gate. That supports our finding that stable1/residual/confidence rules are competitive.
- Caveat for us: Ouro's own KV note says each step needs its own KV cache, "4x memory overhead for our 4-step model", which weakens "trade time for memory" for LLMs. For TRM-type models there is no KV cache.
- Diff: looped LLM; server; accuracy vs average exit round; per-token; no error detection; no precision change; trained.

5. Mixture-of-Recursions, arXiv 2507.10524. Bae et al., KAIST AI, Mila, Google DeepMind. NeurIPS 2025 main track (verified on papers.neurips.cc via web search). Code released. Grade: anchor. Read the abstract, routing, caching, experiments and discussion by extraction.
- Quote: "MoR reuses a shared stack of layers across recursion steps to achieve parameter efficiency, while lightweight routers enable adaptive token-level thinking" and "improving memory access efficiency by selectively caching only their key-value pairs". Scale 135M to 1.7B. Throughput numbers are on GPU. No device, no energy.
- Stopping: router trained end-to-end, expert-choice or token-choice. Not input-level halting.
- Take: cite as the anchor for "looping plus adaptive depth is accepted at A* level". It also introduced continuous depth-wise batching, which the 2608.09444 paper later implemented.

6. "Depth-adaptive Inference of Looped LMs via Continuous Depth Batching", arXiv 2608.09444 (v2, 25 Sep 2026). Exists and is real. Schwethelm, Rueckert, Kaissis, TUM, Imperial, HPI. Preprint ("Preprint." header); venue not verified. Authors' refereed record is in medical ML and privacy, not looped LMs. No code link found in the extracted text. Grade: support/caution pending code. Read in full for method and results; appendices only by grep.
- Result: Ouro 1.4B and Huginn 3.5B on one H100 80GB. Refill gives 1.32-1.53x throughput on Ouro (96-99% of the estimated max speedup); Huginn no-refill 1.30-1.37x (83-96%). Exit thresholds are replayed from recorded decisions.
- Most useful sentences: "fewer FLOPs do not always reduce wall-clock time, especially in the memory bound regime"; "Looped language models ... have received little attention from the inference perspective". It also says "expensive prefill limits the end-to-end speedup".
- Where it supports us: it is an independent server-side demonstration that FLOP-based savings estimates miss wall-clock. That parallels our finding that FLOP estimates miss energy by 2.2-8x. It does not measure energy or run on edge.
- Related-work list, useful for citations: Frey 2026, Fu 2026, Li 2026, Song 2026, Kohli 2026, Lee 2026 (exit rules), Jeddi 2026 LoopFormer (ICLR 2026), Prairie 2026 Parcae (COLM 2026). I did not read these.

7. Parallel Loop Transformer, arXiv 2510.24824. ByteDance Seed. Preprint; venue not verified. Grade: support.
- Quote that undercuts our motivation: "under equal-latency budgets--typical in practical inference--the vanilla loop transformer offers no inherent advantage and can be worse due to longer decode paths, higher memory-bandwidth pressure, and larger KV caches." Also: "inference latency and memory requirements increase with each added loop". Its fix runs loops in parallel across tokens, so the extra latency is negligible because decoding is memory-bound. Measured on one H800 with vLLM, FP8.
- For us: looping trades time for memory only if you accept the latency. Our energy per answer = loops x per-step energy makes that cost visible and measurable on-device.

8. PonderNet, arXiv 2107.05407. Banino, Balaguer, Blundell, DeepMind. 8th ICML Workshop on AutoML 2021 (workshop, per header). Grade: support. Read the method and experiments by extraction.
- Stopping: learned per-step halting probability, geometric prior, an expected-compute regulariser; at inference it samples the halting step. Compute steps counted by actual forward passes. No hardware.
- Take: the origin of halting-head training (also feeds Ouro's exit gate and HRM's ACT). Cite for lineage.

9. Bansal et al., "End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking", arXiv 2202.05826 (NeurIPS 2022 per the task brief; I did not verify on proceedings, so mark unverified). Maryland, Chicago, NYU; Goldstein, Goldblum. Grade: anchor if NeurIPS main track is verified, else support. Read the method and the stopping and overthinking sections.
- Relevant: they used "a maximum confidence exit rule" for the baseline DT models and note it "marginally mitigates overthinking". Their architecture removes the need: "allows for pre-defining a large iteration number at which to terminate the recurrence rather than having to carefully choose a stopping iteration". This is the one paper that says more loops do not hurt. It matters for our finding that fixed depth burns energy needlessly while accuracy has saturated.

GROUP 2: other papers found (edge, memory, stopping)

10. MELT (Qualcomm AI Research), arXiv 2605.07721. Vendrell et al. Preprint. Read abstract, method and conclusion. Fine-tunes Ouro for constant KV memory. "~x4.0 memory compared to Ouro at same performance." No device or energy. Support (Qualcomm AI Research, known record).

11. Hyperloop Transformers, arXiv 2604.21254, MIT (Yoon Kim). Preprint. Quote: "many applications of interest such as edge and on-device deployment are further constrained by the model's memory footprint". About 50% fewer parameters than depth-matched models, "performance persists through post-training weight quantization". Memory framing only; no device measured. Support.

GROUP 3: quantization and numerical robustness (what the brief most wanted to know)
- 2606.26488, Jim, Kolawole et al. (own prior work). First systematic compression study of TRM/HRM: "quantization noise compounds across H recursive cycles"; INT8 6.51 MB, calibrated INT4 3.26 MB; analytic weights-only memory, "no MCU toolchain runs these graphs today", so no on-device measurement. Introduces carry-trajectory fidelity (cosine to the full-precision path) as a label-free predictor. This is the prior form of our fidelity statistic, and the "compression cliff" in the brief.
- 2607.16237, "Quantizing Recursive Reasoning Models", Ingolfsson, Benini et al., ETH Zürich IIS (a hardware group with a strong refereed record). Preprint, 25 Jun 2026. Read abstract, intro, results. Per-tensor INT4 collapses Sudoku exact accuracy from 84.1% to 0.0%; blockwise MXInt4 restores it; the deepest model (EqR equilibrium model) is the most sensitive. States these models "are reported almost entirely in full precision" and hardware is "motivation only" (mentions Arm Ethos-U). This overlaps the compounding explanation, and the "cliff transfers to DEQs" claim is partly covered by their EqR result. It also notes halt-head selection did not rescue INT4 (stays 0.0%), a useful data point for our repair story.
- 2609.30820, Nux Li, Meta. Huginn INT4 collapse localised to the loop-entry adapter ("feedback exposure"); one-step GPTQ worse than RTN on 5 of 9 checkpoints. Single author, but Meta affiliation and 9 checkpoints; grade support.
- LoopQ, arXiv 2605.16343 (NTU + MediaTek) and LoopQuant, arXiv 2610.16368 (Peking Univ + Houmo AI, code released): loop-aware PTQ for Ouro-class LLMs; W4A4. LoopQuant reports 65.3% less memory and 1.7x decoding speedup over BF16 on Ouro-2.6B (server GPU). I read the abstracts only. Support, not read in full.
- 2609.39277, Tilted Bowl (own, under review): "the collapse is usually blamed on rounding error that accumulates from loop to loop ... it holds only for loops that never settle." Claims a label-free prediction and the stop-then-8-bit-finish controller. This partly contradicts the brief's explanation "errors compound across loops", so the MobiSys paper's wording must agree with it (compounding for non-settling loops; a shifted fixed point for settling ones). Check with Steven before the explanation is written either way.
- Not read, search snippets only (abstract-only, cite=no): 2607.10128 (Energy-guided Recursive Model, stops on energy stability), 2611.08977 (Energy-Based Reasoners, halts on gradient norm), 2610.16427 (Resample the Answer), 2610.20136 (VLM energy on Jetson Orin Nano, not looped). These could contain stopping rules relevant to ours; worth a read.

ANSWERS TO THE QUESTIONS
Q1. Efficiency, memory, edge and adaptive-compute claims, quoted:
- Huginn: "may have broader benefits for the local deployment of these models with commodity chips."
- Ouro: "2-3x parameter-efficiency gains that are critical for deployment in resource constrained environments."
- MoR: "improving memory access efficiency by selectively caching only their key-value pairs."
- HRM: "significant computational savings with minimal impact on performance."
- PLT, against us: "the vanilla loop transformer offers no inherent advantage and can be worse."
- CDB: "fewer FLOPs do not always reduce wall-clock time."
All are parameter or memory arguments. None is tested on a device.
Q2. Energy or on-device measurement: none found for any looped model. Quantization: yes, six papers (Group 3), all accuracy-focused; the only hardware contact is the MCU-budget argument in our own 2606.26488 and a motivation paragraph in the ETH paper. My search was web search plus snippets, so absence is "not found", not proof.
Q3. Stopping compared to ours:
- Learned halting head: PonderNet, HRM/TRM ACT (Q-learning or BCE), Ouro's gate with Q-exit threshold.
- Training-free: Huginn's KL rule, Ouro's hidden-state-difference threshold, Bansal's max-confidence rule.
- Loop-count comparisons against fixed depth: HRM Fig. 5 and Ouro Fig. 5. Ouro's static arm is swept over 1-4 steps, which matches the "best fixed depth" idea.
- Ours differs by (a) reporting energy per correct answer on boards, (b) reporting the cases with no gain (HRM and Maze on the Nano), and (c) running the halting head at inference on TRM, where the original uses it only in training.
Q4. "Why now": Ouro (7.7T tokens, 1.4B/2.6B matching 4B/8B), MoR (NeurIPS 2025), Huginn, PLT and Ouro from ByteDance Seed (industry investment), CDB (first end-to-end serving system), and six quantization papers in 2026 from Qualcomm, MediaTek, Meta, ETH, Peking/Houmo, MIT. The edge and silicon companies are now working on looped models, and the questions we answer are open in their own papers.

RANKED LISTS
Three closest to our claims:
1. 2609.39277 Tilted Bowl (own): halting head stop + 8-bit finishing, label-free prediction.
2. 2606.26488 (own): compounding, label-free fidelity, edge framing.
3. 2607.16237 ETH: quantization collapse and drift on TRM/HRM/EqR.
Three that most strengthen motivation:
1. CDB 2608.09444: FLOP savings do not equal wall-clock, with a ready-made citation.
2. Ouro and Huginn deployment claims (untested, so our measurement is new).
3. Hyperloop and MELT: edge memory framing from MIT and Qualcomm, no device measurement.
Claims of ours left unclaimed by this cluster:
- Energy per answer, and the cost law (loops x per-step energy).
- 35 phones / NPU measurements, Jetson, Pi, Ethos.
- Silent faults in vendor toolchains (nobody tested toolchains).
- Label-free 2-loop build check (AUROC 1.000).
- Energy per correct answer under halting.
Claims undercut or narrowed:
- Halting head used with compounding repair: already in the Tilted Bowl submission, so it is not a new idea in the MobiSys paper.
- "Errors compound across loops" as the explanation: contested by Tilted Bowl for settling loops; the ETH paper's drift argument is similar.
- Compression cliff and label-free fidelity: overlap 2606.26488 and the ICLR submission; keep to what is new.
- "Looping trades time for memory": PLT argues it loses at equal latency, and Ouro notes KV memory grows with loops. Narrow to small non-attention recurrent models where there is no KV cache.
- Novelty of quantizing looped models: now occupied by six papers, so claim only the on-device and toolchain angle.

Ledger rows are due for every arXiv id above. Not graded: venues for HRM, TRM, Huginn, Ouro, PLT and Bansal need proceedings or OpenReview checks before any is labelled anchor.