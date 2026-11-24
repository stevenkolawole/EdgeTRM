# Second wave: halting heads and the value of a recurrence (2026-11-23)

READ STATUS. Papers 1 and 2: PDFs fetched, pdftotext, main text read in full by section; appendices read where they bear on cost (paper 1 App. C.1 timing, App. G concurrent work; paper 2 App. D inference cost, discussion). Paper 3 "Act Before It's Too Late": NOT FOUND (six searches across arXiv API, arXiv search page, web). Not read; nothing claimed about it. Please send an arXiv id or author. Two neighbours I did read in part (abstract, setup, grep for energy/hardware/baseline): 2608.09444 and 2611.07211. All text treated as data; nothing executed. No edits outside this file.

## 1. Do Halting Heads Really Save Compute in Looped Language Models?
- Wu, Heo, Tan; Apple. arXiv 2610.32246 (v1, 2026-10-23, cs.CL). Preprint; no venue found. No code link seen in the PDF; the arXiv page has a "Code, Data, Media" tab I did not open. Read: full body plus App. C, G.
- Credibility: Apple authors, pretraining runs to 1.3B at four scales, three seeds, noise floor measured (0.0045 BPB across GPU types), limitations stated, numbers consistent between abstract, Table 1 and text. Prior refereed record of the authors in this subarea: none found. No slop flags (no inflated claims, references exist and match). Grade: support (preprint, strong internal controls).
- What it does: trains PonderNet/ACT-style halting heads into looped LMs (140M-1.3B, FineWeb-Edu and Common Corpus) and counts the iterations tokens actually run. Mixture-loss heads predict E[N]~3.7 of 8 iterations but run 7.6-7.9 under the 90% exit rule. Per-step loss works only with a strong KL prior. Batched decoding (32 sequences) pays for the deepest token: 4.4 iterations vs 2.8 per token at 370M; timed 1.6x vs 2.3x speedup over 8 iterations.
- Diff:
  - model class: looped autoregressive LLM, trained from scratch.
  - hardware: one exclusive H200, bf16, ratios only, medians of nine; training on H100/H200/B200.
  - measured: bits-per-byte (BPB) and iterations (expected, realized, batched); wall-clock speedup ratios in App. C.1. No energy, no power, no joules anywhere (grep: zero hits).
  - per-input compute: yes, per token; decided by a trained halting head with a percentile exit rule.
  - detects incorrect outputs: no.
  - precision over time: no.
  - training: trained head (the paper is about the loss).
- Baselines: the "sampled-depth baseline" (no head, trained with random depth, run at a fixed number of iterations equal to the realized or batched count) and a "specialist" (a model trained at one depth, chosen to match the compared model's realized steps). This is a fixed-depth comparator at matched cost, not "best fixed depth chosen with test labels at matched accuracy". The axis is quality (BPB) at equal iterations, not cost at equal accuracy.
- Overlap with (a): none. "Table 1: What the head predicts is not what a batched pass pays." It argues iteration counts mis-state cost, in GPU time under batching, which supports the premise that a unit of account must be measured, but it never reaches energy or edge hardware.
- Overlap with (b): partial and on the opposite side of the evidence. Its sentence "the reported average can therefore overstate the saving" and "a model without a head is almost as good at that cost" is the closest claim to ours. It challenges whether halting beats fixed depth, in language modelling at batch 32 on a server GPU. Our setting is batch 1 per input on a device, where there is no deepest-token wait, a different regime the paper itself names: "Decoding one sequence at a time has no deepest token to wait for."
- Where our claim stands: narrower in model class (TRM/HRM/DEQ, per-input not per-token, no KV cache), different in unit (energy per correct answer, measured on Nano and AGX), different in baseline (best fixed depth at equal accuracy). Their result does not contradict ours but is a warning we must answer.
- Where we could fall short: (i) they show an average can overstate savings, so our ~2x claim must be a measured J/answer under the deployed batch size, not iterations; (ii) they show that "a head predicts less" is not "an exit rule spends less", so we should report realized loops under the stated rule and threshold; (iii) their depth-curve test (a head can gain at most the drop from realized steps to cap) predicts small gains when the accuracy-vs-depth curve is flat; on HRM and Maze on Nano we already see nothing gained, which is that test's prediction. Cite it as explaining our null cells.
- Settling experiment: report for each model the realized loops under the rule (not expected), the energy at batch 1 and at the batch size the app would use, and the best-fixed-depth curve slope; state where flat curves predict no gain.
- Take: framing for (b) (report realized, state threshold and batch size), the three-number reporting convention, and a citation for why batch-1 on-device is the regime where halting pays. Motivation sentence: "per-token iteration averages are commonly reported as savings" while "estimated from the iterations its rule skips rather than measured as running time" (they quote Tan et al. 2023): ours is measured.

## 2. How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models
- Schwethelm (TUM), Rueckert (TUM, Imperial, MCML), Kaissis (HPI Potsdam). arXiv 2604.21106 (v4 2026-10-28, cs.LG). Preprint; no venue found. Code: github.com/kschwethelm/looped-lm-scaling. Read: full body, App. D.
- Credibility: established group (Rueckert, Kaissis have refereed ML records; looped-LM subarea record not verified, adjacent yes), code released, versions revised, internal numbers consistent (phi 0.46 baseline, 0.38 truncated BPTT, 0.65 hyperconnections). Slop flags: none. Grade: support.
- What it does: 116 pretraining runs, r in {1,2,4,8}, effective depth fixed, fits L = E + A(N_once + r^phi N_rec)^-alpha + B D^-beta, finds phi = 0.46: a recurrence is worth about half a unique block. At r=4 a 410M looped model matches a 580M non-looped model at the training cost of a 1B one.
- Diff:
  - model class: looped LM, pretraining scaling.
  - hardware: none for inference; FLOPs and memory computed analytically (Table 6, 80 GB accelerator capacity).
  - measured: validation loss vs parameters, tokens and r. No energy, latency or halting evaluation (the halting/adaptive passage in the appendix is explicitly "future work").
  - per-input compute: no. Fixed r.
  - detects incorrect outputs: no. Precision: no. Training: trained.
- Overlap with (a): none on cost. It uses FLOPs as the cost axis ("per-token forward FLOPs F(r) stay flat"; "6.4x the FLOPs per generated token"). Our measured finding that FLOP estimates miss device energy by 2.2-8x is a direct caution on that accounting, not a conflict.
- Overlap with (b): none. Quote that decides it: "A scaling law that jointly accounts for recurrence value, gate training, variable inference compute, and serving efficiency is an important direction for future work." It names the gap we fill from the deployment side.
- Where our claim stands: different (training value of a loop vs the inference cost and stopping of a loop).
- Take: motivation. "eight loops are worth about 2.6 unique blocks" style diminishing return (value of loop k falls) is the training-side reason acc(n) saturates in our cost law J = n e_d / acc(n). Also the iso-depth vs iso-parameter inference table (6.6x KV memory at r=8) supports "loops trade memory for time" on devices. Cite as support for the premise; version-dependent numbers, cite v4.

## 3. "Act Before It's Too Late"
Not found. Searches: arXiv API title queries, arXiv search page (title and all-fields), six web queries including halting, deadline, early exit, looped, recurrent. No such title exists in the index I can reach. Could be a different title, a very recent paper, or an OpenReview-only submission. No grade, no verdict. Nothing to add to the ledger until located.

## Closer works found while following citations (partly read; grade support, read=sections)
- 2608.09444 Schwethelm et al., Continuous Depth Batching (single 80 GB H100; Ouro 1.4B, Huginn 3.5B). Fixes the batching wait that paper 1 identifies; speed only, no energy ("realizes up to 99% of the estimated maximum speedup"). Same server regime.
- 2611.07211 LoopDuet (Duke, one A100-80GB, HotpotQA, throughput and EM; "19% more throughput than the fastest tested uniform setting meeting the same accuracy criterion"). This one does compare against a best uniform (fixed-depth) setting at equal accuracy, in throughput on a server GPU, with no energy. Closest in baseline logic to (b); different in resource (throughput, not J), hardware, and class (LLM serving, loop budgets per phase, not per-input halting). Read it in full before the paper cites "best fixed depth at equal accuracy" as new.
- 2610.18174 Park and Rusch, Looped Reasoning Finishes Earlier Than You Think (Sudoku-Extreme halting, median halt step 27 vs 122; not read beyond abstract). Same task family as TRM; steps only; check before claiming halting on Sudoku is untested.

## Ranked (this cluster)
Closest to our claims: (1) 2611.07211 LoopDuet (best uniform depth at equal accuracy, throughput); (2) 2610.32246 halting-heads (halting vs fixed at matched cost, batching); (3) 2608.09444 CDB.
Strongest motivation: (1) 2610.32246 (average iterations mis-state cost; supports measuring real cost); (2) 2604.21106 (a loop is worth about half a block; diminishing return in loops, KV-memory cost of loops); (3) 2608.09444 (variable depth breaks uniform batching).
Unclaimed by this cluster: energy or power for any looped model; edge hardware; J per correct answer; cost law with no fixed term; FLOP-vs-measured gap for loops; per-input halting on non-language looped models; a baseline of best fixed depth chosen with labels.
Undercut or must be qualified: the generality of "halting saves ~2x" (paper 1 shows it can vanish in batched LLM serving; our claim is batch 1 on device and must say so); any use of iteration counts or FLOPs as the cost of a loop (papers 1 and 2 both do; our cost law is the correction).

## Wording of (b)
Say "per input, batch 1, measured joules" and "realized loops under the stated rule and threshold", not "average loops". Say "best fixed depth chosen with test labels" as the baseline (stronger than their matched-iteration baseline). Do not write "halting saves compute" without the batch-1 qualifier. Write that the saving shrinks when the depth-accuracy curve is flat (our HRM and Maze Nano nulls), citing paper 1's bound. Avoid "first" for best-fixed-depth comparison until LoopDuet is read in full.

## Ledger rows to add
2610.32246 support/preprint/full; 2604.21106 support/preprint/full; 2608.09444 support/sections; 2611.07211 support/sections (read in full before citing); 2610.18174 caution-unread/abstract/cite=no.

## Addendum 2026-11-23 (coordinator follow-up)

### LoopDuet, arXiv 2611.07211 (Du, Cheng, Li, Chen; Duke; preprint; read through setup, design, main results, headline claims; appendix not read in full). Grade: support.
- Hardware: one A100-SXM4-80GB, vLLM 0.14.1, bf16. Closed-loop concurrency C=64 (HotpotQA); open-loop Poisson traces at 60 and 10/60 requests/s. Models Huginn-0125 and recurrent OLMo-2. Metrics: exact match, F1, throughput (req/s), TTFT, TPOT. Energy and power: none (zero grep hits).
- Baseline: "uniform (S,S)" reduced depths, the "fastest tested uniform setting meeting the same accuracy criterion" (Huginn (20,20); OLMo uniform 4). Criterion is full-depth quality on the same test set, so the depth is picked offline with labels, which is the same selection logic as our best fixed depth.
- What varies: depth per serving phase (prefill P, decode S), not per input. Controller adapts P to queue age; "the runtime has no correctness feedback." No halting head, no per-input stopping.
- Quote: "Including reduced uniform depths tests whether separate phase budgets improve the quality-throughput choices available to deployment."
- Verdict: LoopDuet sets depth per serving phase and compares that static schedule with a static uniform depth, in throughput, on a server GPU, with no per-input halting and no energy. Our claim (per-input halting with the model's own head vs the best fixed depth chosen with test labels, in joules on Nano and AGX) stays distinct. Do not write "first to compare against best fixed depth"; write "we measure energy against".

### "Act Before It's Too Late: Power-Efficient LLM Inference on Mobile Device" (Chu et al., MobiSys 2026, DOI 10.1145/3745756.3809208)
Full text NOT reachable (doi.org and ACM pdf URLs return HTML; no arXiv). Read status: abstract and metadata only (via search); venue verified on the MobiSys 2026 accepted-papers and ACM proceedings listing. Reported: TurboInfer, millisecond-scale GPU frequency scaling driven by GPU stalls; stalls are >74% of token-generation latency; up to 42.6% power saving (static) and 48.3% (dynamic) on two phones. From the abstract it is a DVFS paper on autoregressive LLMs; no sign of per-iteration cost, early stopping or deadlines. Unverified beyond the abstract; cite=no until read. Grade pending (A-level venue, so likely anchor once read).

## LoopDuet (2611.07211), read by the lead from the PDF: how it differs from our halting result

Read: the main text in full (abstract to conclusion, pp. 1-14), and in the appendix the quality-criterion and
state-distance pages. Sections 4.4-4.6 and the appendix tables were read only by keyword.

Credibility: preprint (Nov 4, 2026); Duke, with Hai Li and Yiran Chen (an established systems group). It includes an AI-usage
statement: Codex and Claude Code implemented and ran the experiments and drafted the text. Grade: support.

What it does: it picks two FIXED depths per deployment, P for prompt (prefill) tokens and S for decoding tokens. It
runs them in vLLM on one A100-80GB at bf16 with 64 requests in flight, on HotpotQA with Huginn and recurrent OLMo. A
queue-age controller lowers P under load. "Offline quality measurements establish this operating range; the runtime
has no correctness feedback."

How we differ:
1. **The unit of decision.** LoopDuet sets depth per serving phase, the same for every request, and changes it with
   queue age. No input gets a depth chosen from its own state. Our halting stops each input on the model's own
   signal.
2. **The question asked of the baseline.** LoopDuet compares one static schedule (split) against another static
   schedule (uniform). Neither side adapts to the input. We compare an input-adaptive rule against the best static
   depth. The shared ingredient is a uniform-depth baseline matched on accuracy, and that ingredient is a standard
   control, not LoopDuet's contribution.
3. **What "equal accuracy" means.** LoopDuet uses "the fastest tested uniform setting meeting the same accuracy
   criterion". The criterion is "an accuracy loss of at most one percentage point with a non-significant two-sided
   exact McNemar comparison" against (32, 32), applied to a grid of 11 Huginn and 6 OLMo configurations. Ours is the
   best fixed depth of the same weights, chosen with test labels (an oracle), compared at equal accuracy.
4. **Unit of cost.** LoopDuet measures throughput (requests/s) and TTFT on a batched server GPU, and prices savings
   with a layer-application count that "omits adapter, vocabulary-head, memory-traffic, and scheduling costs". We
   measure joules per correct answer at batch 1 on the Jetson Nano and AGX Xavier. Caveat for our FLOP claim: on its
   batched A100 the layer count predicts 1.94x for Huginn (32, 16) and it measures 1.87x throughput, so the proxy
   tracks there. Our 2.2-8x FLOP miss is about energy on edge devices at batch 1 and must be worded that way.
5. **Where the saving comes from.** LoopDuet's saving comes from long prompts ("prompt positions account for over
   99% of this work") and from gathering continuing tokens into compact batches. Our models have no prompt or decode
   phase, and their state has a fixed size, so its mechanism does not apply.
6. **Its state-distance rule, relevant to our check (c).** LoopDuet screens candidate prompt depths with a
   hidden-state distance against the same model at full depth. The threshold is the pooled 90th percentile, 0.24, and
   it transfers across held-out checkpoints. That rule picks a budget for a correctly computed model. Our check
   compares a compiled device build against an fp32 host to detect a toolchain fault. Cite it in the check's related
   work: a label-free state threshold that transfers is shared ground, and our use and our reference differ.

Our wording states what we measure, with no "first" claim needed. The claim is
per-input halting with the model's own head, against the best fixed depth of the same weights chosen with test
labels, at equal accuracy, in measured joules per correct answer at batch 1 on edge devices. The related-work clause:
"LoopDuet compares phase-level depth schedules against the fastest uniform depth within an accuracy tolerance, by
throughput on a batched A100; neither side adapts depth per input."

## Act Before It's Too Late (full read)

**1. Identity.** Chu, Fan, Deng, Yu, Xie (ByteDance), Liu, Ma, Zheng (BUPT), "Act Before It's Too Late: Power-Efficient LLM Inference on Mobile Device" (TurboInfer), MobiSys 2026, doi 10.1145/3745756.3809208, 14 pp. Read in full from the PDF (all sections, appendix A.1, references skimmed). Credibility: MobiSys main-track proceedings (DOI on the paper); BUPT group has prior MobiSys/SenSys work (nnPerf, SenSys 2023; RAMS, TMC 2026), Ma and Zheng are established mobile-systems authors. No code release stated (built on open mlc-llm v0.19.0, 4,200 lines C++ plug-in; no repo link found in the text). Slop flags: none (one dangling reference marker "[84?]", one typo "Llamma", Llama-3.2-3B appears in section 4.3.4 though not in the model list). Grade: **anchor** (A-level systems venue, verified by the DOI/copyright block; confirm on the program page before citing as MobiSys).

**2. What it does.** Phone-GPU DVFS for autoregressive LLM decoding. Observation: kernels are memory bound and the host-controlled launch pipeline plus shared DRAM stalls the GPU; stalls are "over 74% of the token generation latency" (abstract; 74.26% on the fastest phone, Xiaomi 14 Pro, p.4; decoding interval up to 91.71% of total time on low-end phones). Default governor (utilisation over ~1 s) keeps frequency high. TurboInfer reads Adreno PMU ALU-instruction counters, defines frequency sensitivity S=I/F, and runs a tube-based MPC (window N=7, QP solver 1.3 ms cut to 0.14 ms, kernel-level cooling device bypassing the 80 ms thermal batching) to drop frequency during stalls and raise it at kernel start. Headline: power saving up to 42.6% (avg 27.0%) vs default and up to 28% (avg 17.4%) vs zTT in the static case; up to 40.2%/48.3% (avg 29.6%/18.2%) with background apps; power efficiency 1.1x-2.11x; decode speed within ~0.17 tokens/s (static) to 1.1 tokens/s of default.

**3. The diff.**
- Model class: autoregressive LLM, one pass per token. No looped, early-exit, DEQ or diffusion model.
- Hardware: measured on device. OnePlus 8 and Samsung Galaxy Note10 (Adreno) for the evaluation; three more phones (OnePlus 9RT, Ace Pro, Xiaomi 14 Pro) only for the stall characterisation. Monsoon power monitor. No Jetson, no NPU, no desktop GPU in the evaluation.
- Models: Qwen2.5-1.5B, Gemma2-2B, RedPajama-3B (characterisation adds Qwen2.5-0.5B, Mistral-7B, Llama2-7B). Engine MLC-LLM, fixed prompt "Print twenty words".
- Measured: average power (W), tokens/s (prefill and decode), "power efficiency" (reported as tokens/W, text says "watts per generated tokens"), frequency traces, controller overhead (<2.5 us cumulative frequency-switch delay; 5 us per PMU read; prediction accuracy 93.1% at window 5 vs LSTM 46-54.4%). No joules per token or per answer is tabulated; energy is inferred as power with speed held roughly equal.
- Compute per input: fixed. Frequency varies within a token (kernel vs stall), set by a PMU-based workload predictor, not by input difficulty.
- Detects wrong outputs: no. Accuracy: not measured at all. The claim "without compromising accuracy" rests on the argument that DVFS leaves the computation unchanged (introduction); there is no accuracy table.
- Precision changes: no. Training-free: yes for the LLM (the controller is analytical MPC; offline parameter fit per device).
- Baselines: Android msm-adreno-tz default governor; zTT (RL DVFS, MobiSys 2021) re-targeted to LLM speed feedback with TurboInfer's own perception module bolted on; GearDVFS excluded (closed source); Quickstep discussed, not run.

**4. Full comparison (rule 5).**
- Energy per answer vs per token: neither. Its unit is power and tokens/W for a 20-word generation. There is no notion of an answer, correctness, or a variable number of iterations per input.
- Deadlines: none. The only time constraint is "minimum impact on inference latency" (token rate within ~1 token/s); no deadline-constrained accuracy. Related work mentions server SLO-aware DVFS (throttLL'eM, HPCA 2025; Kakolyris CAL 2024) as other work, not as this paper's method. Our claim (b) is untouched.
- Frequency effect on per-step energy: this is the one real overlap with claim (a), power mode as a lever. It shows frequency sets the efficiency per kernel: "the GPU's energy-efficiency sweet spot shifts from 350 MHz to 425 MHz as free memory drops from 45% to 15%" (Fig. 10b, p.5); peak IPC drops up to 79.6% when memory bound; most kernels are memory bound (roofline, Fig. 5c), so "a lower computing frequency may not introduce extra computing delays" (p.4). It does not give a per-step energy model that predicts total energy; its only model is a 3-state (frequency, sensitivity, temperature) control model for the next 7 ms, and power is predicted from the temperature derivative (Eq. 12-13, linear, App. A.1), not validated as an energy predictor. No FLOP-vs-measured comparison.
- Where we stand: **different** (not identical, not narrower). It is a within-token, kernel-granularity actuator on phones for a fixed-cost autoregressive step. We are an across-iteration cost law plus per-input stopping, with accuracy as the metric and board/precision/power-mode as the cost coordinates. Deciding quote for the gap: "can we minimize the power consumption of on-device LLM inference without compromising the model's accuracy and latency?" (p.1): the question holds the computation fixed; ours changes how many passes each input gets.
- What it holds of ours: (i) power consumption of on-device LLMs is a first-order mobile bottleneck ("up to 6.15x" a social app; standby 15 h to 2.5 h); (ii) power state is a runtime lever distinct from compression; (iii) the default governor is mismatched to the workload. Strengthens our power-mode lever argument.
- Where it falls short of ours: no answer-level metric, no accuracy, no input-adaptive compute, no cost law, no deadline, no cross-device study of accuracy, nothing on Jetson.
- Where we could fall short against it: (1) it shows per-step energy is not constant within a step: it varies with frequency at ms granularity and with free memory (sweet spot moves 350 to 425 MHz, 45% to 15% free memory). Our "constant per-iteration energy for a given board, precision and power mode" holds on our fixed-mode, batch-1, idle-board measurements; a reviewer can cite this paper to say background load shifts per-step energy. We must state the condition (no co-running load, fixed power mode, governor not re-tuned per kernel) and, ideally, report the spread of per-iteration energy under a background load on one board. (2) Its fine-grained DVFS could lower per-iteration energy further and stack with halting; that is complementary, not a threat.
- Settling experiment (optional, cheap): per-iteration energy on the AGX or Nano under a co-running GPU job; if it moves less than a few percent the constant-per-iteration claim survives their memory-contention point; if not, add the condition to the claim.

**5. What to take.**
- Citable numbers (all MobiSys 2026): GPU stalls up to 74.26% of token latency on the fastest phone, up to 91.71% of autoregressive time is decoding-interval stall on low-end phones (p.4); default governor keeps frequency high and takes 0.3-0.8 s to drop after inference ends (Fig. 7); LLMs draw 3.11x-6.15x the power of everyday apps (p.3); power 4.86M-parameter LSTM to 3B LLM grows 36-fold (p.3); TurboInfer power saving up to 42.6%/48.3%, average 27.0% vs default, 17.4% vs zTT, efficiency 1.1x-2.11x (p.2). Page numbers are from the PDF page order (p.1 = abstract page); confirm against the final proceedings pagination.
- Framing: cite as support that "the stack's default power management is mismatched to LLM inference" and as system-level power (not algorithmic) savings that "complement" model-side methods; one clause in related work: "TurboInfer scales GPU frequency within each decoding step on phones to cut power at fixed output; it holds the amount of computation per input fixed and reports no accuracy or energy per answer."
- Narrow nothing. Add one condition sentence to claim (a) (fixed power mode, no co-running load).
- Motivation for the deployment-stack-mismatch story: a second MobiSys 2026 paper showing a stack assumption (utilisation-based governor) failing silently on LLMs, same pattern as our toolchain faults, though power not correctness.
- Supports "device matters": stall ratio rises with device compute capacity (Fig. 5), so faster phones waste a larger share of a step, relevant to our cross-device per-iteration time claim (b) only as a reminder that per-step time is not FLOP-proportional.

**Verdict on claims.** (a) cost law: unclaimed by this paper; one risk noted above. (a) power mode as lever: supported in spirit, not claimed. Stacking with halting: unclaimed. (b) deadlines: unclaimed. Accuracy: untouched by the paper.
