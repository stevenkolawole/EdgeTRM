CLUSTER F REPORT: quantization and reduced precision of iterative models

Read status is stated per paper. Several are "sections", not "full". My token budget did not allow a full read of all 13 papers. Verify venues on proceedings or OpenReview pages. I had no way to open those pages. Where I give a venue, it comes from the paper's own footer or from my memory, and I mark which.

Files are under .../scratchpad/papers/ (the scratchpad folder of this session). Q-Diffusion, PTQ4DM, PTQD and TDQ are in 2302.04304, 2211.15736, 2305.10657 and 2306.02316. The rest are in F_<id> folders (F_2605.16343, F_2609.30820, F_2610.16368, F_2603.14062, F_2603.10562, F_2610.05265, F_2609.24042, F_2508.12094, F_CH18). Each folder has p.pdf, p.txt and a compressed c.txt.

HEADLINE
1. Per-loop precision allocation for a looped model, with early loops low and late loops high, already exists. LoopQuant (arXiv 2610.16368, 12 Oct 2026, preprint) finds A4,A4,A4,A8,A8,A8 best on Huginn-0125 with T=6. That is 6 weeks before your deadline, so it is concurrent. It is not a repair of a broken build. Your "finish at higher precision" claim must be narrowed to "repair of a deployed, silently faulty vendor build", and its mechanism must be separated from the design-time allocation.
2. The diffusion literature points in opposite directions. PTQD (NeurIPS 2023) gives late steps more bits. TMPDiff (2026) finds the gains from upcasting concentrate in early steps. "Late loops need precision" is therefore model-specific. Your Maze and Sudoku repair results are the evidence, not an assumption borrowed from diffusion.
3. Nobody I found has done two of your things:
   - using a short-horizon state, with no labels, as a build-validity detector (AUROC 1.000);
   - running loop count as the controlled variable to show collapse on a real deployed stack.

Q1: Has anyone shown iteration-amplified collapse, where accuracy holds and then goes to 0?
- Partly. No one has varied loop count or step count as the controlled variable and shown hold-then-collapse.
- Collapse at a bit-width threshold exists in several places:
  - TDQ Table 2 (CIFAR-10, Min-Max): FID 8.73 at W8A8, 34.61 at W8A7, 332.15 at W8A6.
  - Q-Diffusion Table 1 (CIFAR-10, linear quantization): FID 4.22 at FP, 4.71 at W8A32, 141.47 at W4A32.
  - PTQD Table 3 (LSUN-Churches): Q-Diffusion MP FID 218.59 against 7.80 at W4A8.
  - MonDEQ quantization (2603.10562): "MNIST experiments confirm a phase transition at the predicted threshold: three- and four-bit post-training quantization diverge, while five-bit and above converge." The threshold is ‖ΔW‖₂ < m, a computable margin. This is the closest analogue to your "fidelity statistic predicts where".
  - Feedback Exposure (2609.30820): per-channel INT4 takes Huginn-3.5B on GSM8K from 34% bf16 to 0%.
  - LoopQ Table 1: static W4A4 on Ouro-1.4B gives LAMBADA accuracy 0.00 to 0.03, against 0.61 in BF16.
- Compounding with iteration count is documented only as a rising error curve, not as a sweep to zero accuracy:
  - Q-Diffusion Fig. 3: MSE grows over steps.
  - PTQD Fig. 4: SNR_Q falls toward 1 for W4A4 as t goes to 0.
  - TMPDiff: an additive error model.
- Feedback Exposure contradicts naive compounding in one case. For R-Llama, "the measured trajectory remains bounded without returning to the bf16 state." They call it a displaced fixed point, with displacement within 1% of its final value after 4 updates.
- Your "errors compound across loops" claim needs a qualifier. Error can saturate at a wrong fixed point, and the collapse is then about the location of the fixed point, not unbounded growth.
- 8-bit PTQ4DM on DDPM with 4000 steps shows no collapse (FID 7.10 against 7.14 FP). So many iterations alone do not collapse. The precision relative to margin is what matters.

Q2: Has anyone run early iterations at low precision and final ones at high precision as a repair for a broken deployed build?
- No repair. Yes for design-time allocation, in three places:
  - LoopQuant ILPA, on Huginn.
  - PTQD step-aware mixed precision (below).
  - TMPDiff.
- Carson–Higham iterative refinement is the classical analogue, but its refinement is a different mechanism (below).
- Nobody frames it as a repair for a toolchain fault found after deployment. Nobody demonstrates it on TensorRT or an NPU, or on a vendor INT8 build that outputs zero or a constant.

Q3: Has anyone used the early-iteration state as a diagnostic?
- Not as a label-free build-validity test. Related uses:
  - Feedback Exposure measures top-1 agreement with bf16 per recurrence step (Fig. 1). That is a reference-based per-step diagnostic and is not used to classify builds.
  - Loopy finds that "quantization configuration candidate rankings can change with recurrent depth." I read this only from the abstract and intro (status: sections).
  - LoopQ Fig. 1 plots relative L2 error across loops.
  - PTQD uses SNR_Q per step against SNR_F, and it needs an FP reference.
- Nothing reports an AUROC for broken versus working builds at 2 loops. This claim is unclaimed.

Q4: What is new in yours, at the minute level?
- Subject. They study PTQ algorithms in simulation (fake quant) on weights and activations. You study failures of real vendor toolchains on real hardware (TensorRT, QDQ, Qualcomm NPU, TFLite Micro). Every paper here uses fake quant or a custom kernel. Only Feedback Exposure checks a real kernel (Marlin), and only for one comparison.
- Target. Their quantizer is calibrated, and the method is the thing under test. Yours is a black-box faulty build with the weights fixed, and the repair is a runtime setting.
- Signal. Their signal needs a calibration set or a high-precision reference model on the same host. Yours is the state after 2 loops compared with a host reference, and it separates builds. Your AUROC drops to 0.67-0.73 across runtimes, which you should state as a limit.
- Repair. Yours changes the precision of the last loops at runtime, on an existing deployment, with no recalibration.
- Law. Your cost law and stopping results sit outside this cluster.
- Hold the line on these points: you test a different object (the build), a different signal (early-loop state), and a different intervention (runtime precision on the last loops).

Q5: What can we cite to strengthen the compounding mechanism?
- Strongest: LoopQ Proposition 2, εT ≤ Σ(Πγ)ε_quant, a Lipschitz recursion bound on loop-wise error. It is a clean bound you can borrow for the "compounding" claim.
- Feedback Exposure for the causal control. It scales the carried-over state error by λ and shows final agreement rises from 10.4% to 69.2% when prior error is removed. That is direct evidence that carried-over error causes the collapse.
- MonDEQ quantization for a DEQ margin certificate.
- Q-Diffusion Section 3.1 and PTQD Section 1 for the diffusion version (quantization errors accumulate across time steps).
- TCEC (2508.12094) derives a closed-form cumulative error for diffusion. I did not read it past the abstract and intro.

PER-PAPER ENTRIES

1. Q-Diffusion (Li, Liu, Lian, Yang, Dong, Kang, Zhang, Keutzer; UC Berkeley, Nanjing, UIUC, Peking). arXiv 2302.04304. ICCV 2023 per my memory, not verified here (the PDF I read is v3, labelled "Preprint. Under review"). Read: main text and parts of the appendix.
- Credibility: Keutzer's group has a strong quantization record. Code is public. No slop flags. Proposed grade: anchor, pending venue verification.
- What it does: it identifies two problems in diffusion PTQ, error accumulation across steps and activation distributions that change with time step. It fixes them with time-step-aware calibration sampling and shortcut-split quantization. DDIM CIFAR-10 W4A8 gives FID 4.93, against 188.11 for linear quantization. Models: DDIM, LDM on LSUN, Stable Diffusion.
- Diff: diffusion; simulated; FID and IS; compute is fixed (the same steps for every input); no detection; precision is fixed across steps (calibration varies, bits do not); training-free.
- Comparison. It holds: "the iterative inference of the noise estimation network leads to an accumulation of quantization error." It also holds Fig. 3, MSE growing across steps and "dramatically" at 4-bit. It falls short because it never varies step count, never reports collapse as a function of iterations, and has no repair by precision.
- Take: cite for "errors accumulate across denoising steps." It is the oldest and most cited statement of the mechanism.

2. PTQ4DM (Shang, Yuan, Xie, Wu, Yan; IIT, Houmo AI, Tencent, Cisco). arXiv 2211.15736. CVPR 2023 per my memory, not verified. Read: main text through the conclusion.
- Credibility: code is released. Proposed grade: anchor, pending venue verification.
- What it does: 8-bit PTQ with a calibration set drawn from a skew-normal distribution of time steps (NDTC). CIFAR-10 and ImageNet-64 FID is near FP (DDPM 4000 steps: 7.10 against 7.14).
- Diff: diffusion; simulated; FID; fixed compute; no detection; fixed precision; training-free.
- Comparison. It holds the observation that output distributions change with time step, but it contains no analysis of error accumulation at all (the word does not appear). It falls short of your claim entirely on mechanism.
- Take: the 4000-step 8-bit result shows that long iteration is not itself a cause of collapse. That supports your claim that collapse depends on margin, not step count.

3. PTQD (He, Liu, Liu, Wu, Zhou, Zhuang; Zhejiang University, ZIP Lab at Monash). arXiv 2305.10657. NeurIPS 2023 per the paper's own footer, not verified on the proceedings. Read: main text and appendix tables C.1.
- Credibility: Bohan Zhuang has refereed quantization papers. Code is public. Proposed grade: anchor, pending venue verification.
- What it does: it splits quantization noise into a correlated part (rescaled by 1+k) and an uncorrelated part (bias correction and variance schedule calibration). It adds step-aware mixed precision, W4A4 for the earliest steps and W4A8 for the rest (ImageNet 250 steps: A4 for steps 249 to 202, A8 for 201 to 0). Weights stay at 4-bit.
- Diff: diffusion; simulated plus a CUTLASS latency check on an RTX 3090 (Table 4: 5.46 s FP, 2.45 s MP); FID and BOPs; fixed compute; no detection; precision does change across steps; calibration-based.
- Comparison. It holds: "the quantization noise accumulates as the iterative sampling process progresses, leading to a significant drop in the signal-to-noise ratio (SNR)". It also holds the direction of your repair, low bits early and high bits late, chosen by "SNR_Q > SNR_F". It falls short because only activation bits change (weights fixed), the rule needs an FP reference at design time, and it is not a repair of a faulty build.
- Take: this is the closest diffusion precedent for the direction of your repair. Cite it for "precision can change over iterations." Your difference is where the allocation comes from (their SNR criterion; your fault diagnosis) and what it fixes.

4. TDQ (So, Lee, Ahn, Kim, Park; POSTECH, SqueezeBits). arXiv 2306.02316. NeurIPS 2023 per the paper's own footer, not verified. Read: main text and results, not the appendix.
- Credibility: code is public. Proposed grade: anchor, pending venue verification.
- What it does: a small module maps the time step to the activation quantization interval, so the step size changes over time at no inference cost. Min-Max W8A6 FID is 332; TDQ is near FP.
- Diff: diffusion; simulated; FID; fixed compute; no detection; the interval changes over time, bit-width does not; QAT or PTQ.
- Comparison. It does not hold your precision-over-iterations claim, since the bit-width never changes. It does not discuss accumulation.
- Take: it is cheap evidence that "precision over iterations" in the diffusion literature mostly means scale, not bits. Write the contrast as bits and runtime setting against scale and calibration.

5. TMPDiff (Lewandowski, Kurz et al.; Neuchâtel, TU Dortmund, TU Delft, Turin, LAMARR; Lydia Chen and Jian-Jia Chen are senior). arXiv 2603.14062, 14 Mar 2026. Preprint, no venue found. Read: abstract, intro, Sections 3 and 4 in part.
- Credibility: established groups, but I did not find a code link in what I read. Proposed grade: support.
- What it does: it assigns a bit-width per denoising step by an additive error model and an adaptive bisection search. At matched speedup it gives 10-20% better perceptual quality than uniform precision. On FLUX.1-dev: 90% SSIM at 2.5x over 16-bit.
- Diff: diffusion; simulated plus latency model; SSIM, LPIPS, PSNR; fixed compute; no detection; precision changes per step; calibration-based.
- Comparison. It holds: "We hypothesize that quantization errors accumulate additively across timesteps, which we then validate experimentally." It holds "assigns different numeric precision to different denoising timesteps." It also reports that the timesteps of high upcasting gain "concentrate in early denoising steps", which contradicts PTQD's late-step rule.
- Take: it supports a nuance you should write down. Which iterations need precision depends on the model. Report your own sensitivity as measured per loop.

6. LoopQ (Fang, Chen, Cheng, Chen; National Taiwan University, MediaTek). arXiv 2605.16343 v2, 12 Oct 2026. Preprint. Read: sections 1-5 and parts of the appendix.
- Credibility: Ming-Syan Chen is a long-time refereed author (data mining and systems), but I have not checked a quantization record. Proposed grade: support.
- What it does: loop-aware PTQ for Ouro, LoopFormer and Parcae: loop-dependent activation scales, selective loop-wise transforms, a cross-loop adapter and trajectory calibration. Ouro-1.4B W4A4: five-task mean accuracy retained at 92.23% of BF16, while static baselines drop to chance.
- Diff: looped; simulated PTQ with a throughput measurement; accuracy and perplexity; compute is per-input only via Ouro's halting head in an appendix (24.1% fewer iterations at q=0.25, accuracy down 1.32 points); no detection; bits do not change across loops (scales do); calibration-based.
- Comparison. It holds: "recursive error accumulation: errors introduced in earlier loops propagate through subsequent computations and can be amplified," with Proposition 2. It falls short because there is no loop-count sweep to collapse and no precision schedule.
- Take: cite the proposition for the mechanism and the Ouro result as a baseline for how bad static W4A4 is on looped models.

7. Feedback Exposure (Nux Li, Meta, single author). arXiv 2609.30820, 25 Sep 2026. Preprint. Read: abstract, intro, Section 3 and the limitations.
- Credibility: single author, no record found by me, no code link in what I read. That meets a rule-4 flag ("single author with no refereed record in the area") until checked. Its numbers are internally consistent and it has many controls. Proposed grade: caution, to be upgraded after a closer read.
- What it does: INT4 on Huginn's non-residual loop-entry adapter alone gives 10.5% top-1 agreement with bf16, while the 16 core projections give 87.9%. GSM8K goes from 34% to 0%. Setting the carried-over state error to zero raises agreement from 10.4% to 69.2%. One-step GPTQ is worse than RTN on 5 of 9 checkpoints, and accumulating the Hessian across recurrence steps fixes it.
- Diff: looped (Huginn, R-Llama, Ouro, Parcae, COCONUT) plus IIR filters and Mamba; simulated (fake quant; one Marlin check); agreement with bf16 and accuracy; fixed compute; no detection; fixed precision; training-free.
- Comparison. It holds "collapse" in the sense of 34% to 0%, and it holds the causal evidence that carried-over error is the cause. It contradicts unlimited compounding (displaced fixed point). It falls short of your claims on device faults and repair.
- Take: its controlled injection is the evidence you should replicate on a real broken build (feed bf16 state back into the faulty runtime).

8. LoopQuant (Zhao, Xu, Hu, Liu, Sun, Yang; Peking University, Houmo AI). arXiv 2610.16368, 12 Oct 2026. Preprint. Read: abstract, intro, Section 4.3, the ablation, parts of the appendix.
- Credibility: Houmo AI authors overlap with PTQ4DM and have refereed quantization work to my memory (unverified). Code link given. Proposed grade: support.
- What it does: Interaction-Aware Loop-wise Precision Allocation picks A4 or A8 per loop under a fixed average bit budget, using per-loop sensitivity plus cross-loop interactions. On Huginn-0125 (T=6, average 6 bits): ILPA [4,4,4,8,8,8] gives Wiki2 PPL 12.47 and 49.09% accuracy. Uniform [6,6,6,6,6,6] gives 13.82 and 46.92%. Random gives 15.21 and 44.83%. Local sensitivity [4,8,4,8,8,4] gives 13.05 and 48.01%.
- Diff: looped; simulated plus a decode speedup on Ouro-2.6B (1.7x over BF16); perplexity and accuracy; fixed compute; no detection; precision does change across loops; calibration-based.
- Comparison. It holds the claim that you assign low precision to early loops and high precision to late loops, on the same model class you use. It falls short because this is a design-time optimiser on a PTQ pipeline. It does not address a silently faulty vendor build, it has no label-free diagnostic and it has no hardware.
- Where we could fall short: a reviewer will ask why your schedule is not just ILPA. Our answer is the fault, the runtime setting and the hardware. We need an A/B showing that last-loop precision repairs a build where uniform recalibration does not.
- Take: it supports your direction on Huginn and sets the baseline to beat. Cite it as concurrent.

9. Loopy (Li, Zhang, Cheng, Li, Zhang; HKUST, Duke Kunshan). arXiv 2610.05265 v2, 17 Oct 2026. Preprint. Read: abstract and intro only (status: sections).
- Credibility: code link given. Proposed grade: support.
- Take: "candidate rankings can change with recurrent depth," so the best quantization configuration depends on loop count. That is a reason to fix loop count and precision together at deployment.

10. Quantization Robustness of MonDEQs (Li, Leong, Chaffey; University of Sydney). arXiv 2603.10562 v2 (math.OC), 13 Jun 2026. Preprint, IEEE format, venue unknown. Read: abstract, intro, Theorems, parts of Section V.
- Credibility: Philip Leong has a long record in arithmetic and hardware; Chaffey in control. Proposed grade: support.
- What it does: a certificate ‖ΔW‖₂ < m for existence, uniqueness, linear convergence and a bounded equilibrium displacement of a quantized MonDEQ. MNIST experiment: 3- and 4-bit post-training quantization diverge and 5-bit converges.
- Diff: DEQ; simulated; convergence and displacement; fixed compute; no detection; fixed precision.
- Take: the only quantitative DEQ result I found. Cite it for the cliff in DEQs. It does not cover your ImageNet MDEQ results.

11. Q-DEQ (Yang et al.). arXiv 2609.24042, 21 Sep 2026. Abstract only. Applies W8A8 fake quantization after a fixed-point solve. Proposed grade: caution; not relevant to collapse.

12. TCEC, Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models (Liu, Zeng et al.; ByteDance). arXiv 2508.12094 v2, Oct 2025. Preprint. Read: abstract and intro only (status: abstract/sections). It derives a closed-form cumulative error and a timestep-aware compensation. Proposed grade: support. Needs a full read before citing for the mechanism.

13. Carson and Higham, Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions. SIAM J. Sci. Comput. 40(2), A817-A847, 2018 (verified on the PDF header). Read: abstract, intro and Algorithm 1.1; I did not read the proofs.
- Credibility: Higham is the field's lead author on this subject. Proposed grade: anchor in its own field (SIAM SISC), outside the list of ML A*/A venues.
- What it does: a factorization in half precision, a working precision, and residuals in a higher precision. For κ∞(A) ≤ 10⁴ it reaches single-precision accuracy with the O(n³) part entirely in half precision. With GMRES the limit is 10⁸.
- Diff: not ML; numerical analysis; error bounds; fixed iteration structure; convergence conditions; training-free.
- Where it differs from yours, minutely. Iterative refinement computes the residual in HIGH precision and solves the correction in LOW precision, each iteration. The error contracts every iteration if κ is within range. Yours runs low-precision iterations first and high-precision iterations last, on a state that has already drifted. Their high-precision residual makes the iteration self-correcting. Yours has no such residual unless the last loops themselves act as a corrector (a contraction, which is the case for TRM and DEQs). Nothing in Carson–Higham says what happens when the low-precision iterations have a fault (a constant output); their analysis assumes bounded relative error in the solver.
- Take: cite as the classical analogue of "low precision where error is damped, high precision where accuracy is set". Its convergence-conditional-on-condition-number statement parallels your cliff, which depends on margin or contraction.

RANKINGS

Closest to your claims:
1. LoopQuant: per-loop precision on a looped model, early low and late high.
2. PTQD: step-aware mixed precision with the same direction, plus accumulation language.
3. Feedback Exposure: collapse to 0% of a looped model with a controlled causal test of error feedback.

Strongest motivation:
1. LoopQ: the bound plus W4A4 collapse on Ouro.
2. Feedback Exposure: 34% to 0% on Huginn plus a bounded displaced fixed point.
3. MonDEQ quantization: a cliff predicted by a margin.

Claims of yours this cluster leaves unclaimed:
- A label-free build-validity check from the state after 2 loops.
- Real vendor-toolchain faults on real NPU, TensorRT and microcontroller hardware.
- A runtime repair of a deployed broken build by raising precision on the last loops.
- Loop count as the controlled variable across a precision sweep on TRM, HRM and DeepThinking models.
- Energy per correct answer.

Claims of yours this cluster undercuts or requires narrowing:
- "Loop precision as a runtime setting" is no longer new. LoopQuant's ILPA already assigns it per loop.
- "Compounding" needs the qualifier that error can saturate at a displaced fixed point (Feedback Exposure).
- "Finish at higher precision" is not universally right. TMPDiff found early diffusion steps matter most, so say the direction is measured per model.
- Numbers from fake quant do not carry over to a vendor build. Everything in this cluster is simulated, so your hardware result is a real gap, but a reviewer will ask for the simulated A/B.

Open items: venue verification for items 1-4 and TMPDiff, a full read of TCEC, and a closer read of Feedback Exposure, since it is a single-author preprint.