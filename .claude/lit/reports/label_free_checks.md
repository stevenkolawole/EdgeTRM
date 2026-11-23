# Deep read, cluster G: label-free accuracy estimation and monitoring of deployed models

Downloads are under .../scratchpad/papers/<id>/ (2206.13089, 2106.13799, 2201.04234, 2407.09141, 2305.07772, 2111.11053, 2111.04779). Nothing was run and nothing edited.

Read-status caveat. I read the text of each paper with pdftotext. I read these parts closely: Dutta (sections 1-6), Baek (abstract, intro, related work, results and method summary), Jiang and Garg (abstract, intro, setup, headline claims and failure statements), Nazar (sections 1-3 plus the detection and evaluation sections) and ML-EXray (sections 1-3). I did NOT read the appendices or the theory proofs of Jiang and Garg line by line. The proofs are not needed for the claims below. Treat the "read" column as "main text, appendix skimmed".

## Short answers

1. **Is a deployed model vs its own reference, without labels, standard?** Partly.
   - Comparing a deployed or compressed model with its own reference is standard in the systems and compression literature. Examples are ML-EXray's per-layer rMSE between edge and reference pipelines, and KL divergence and "flips" for quantized LLMs.
   - Using that comparison as a label-free detector of a broken build, scored as AUROC, is not what any paper here does.
   - The ML-theory papers (Baek, Jiang, Garg) estimate accuracy under distribution shift. Their agreement is between different models, and their target is accuracy.
   - Evidence status: measured, from the papers read.
2. **Precedent for an early-iteration signal, before the answer exists?** None in this cluster.
   - Every detector here reads the final output (Nazar's softmax confidence, ATC, agreement, flips, KL).
   - ML-EXray reads intermediate tensors, but it runs the whole network on both sides and uses them to locate a fault after an accuracy drop is seen.
   - Early-exit and halting literature belongs to other clusters. I did not read it.
3. **Strongest honest claim.** For looped models, the state after 2 of N loops, compared with a host reference run, detects a silent build fault before the full run. It needs no labels, and it works on phones and TensorRT (AUROC 1.000). It degrades across runtimes (0.67-0.73), and we say so. See the end of this report for what the claim must not say.

## Per paper

### 1. Dutta et al., "Accuracy is Not All You Need" (flips)

- **Authors.** Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, Ramachandran Ramjee. Microsoft Research India.
- **Venue.** NeurIPS 2024, main-track poster. The search result shows the neurips.cc poster page and the proceedings.neurips.cc link. I read the arXiv v1 PDF, which is marked "under review", not the camera-ready.
- **Read.** Main text in full, appendix tables skimmed.
- **Credibility.**
  - Known lab.
  - The same group has refereed systems work on LLM serving (Kwatra and Ramjee, e.g. Sarathi-Serve, OSDI 2024). I know this from memory; I did not verify it.
  - Code not checked.
  - No slop flags.
- **Grade: anchor.**

**What it does.**
- It compares 16-bit models with quantized, pruned and layer-dropped LLMs (Llama2 and Yi chat, 6B-70B).
- Accuracy stays within 0-2%, but "flips" reach up to 13.6%. Flips are the share of answers that go correct to incorrect or the reverse. GSM8K shows 12-30% flips for BnB W8A8/W4A4.
- KL divergence against the baseline correlates with flips (Spearman 0.981 on MMLU).
- Flips on MMLU correlate with the MT-Bench drop for the larger models.
- Mechanism: a low "top margin" (probability gap between the best and second-best option) makes an answer likelier to change and likelier to be wrong. Incorrect answers change 2x or more as often as correct ones (MMLU 5-shot, e.g. Llama2-70B BnB 4-bit: 5.0% of correct vs 13.2% of incorrect).

**Diff.**
- Model class: autoregressive LLM.
- Hardware: server GPU, TensorRT only for SmoothQuant.
- Measured: output deviation from the baseline, plus accuracy.
- Compute per input: fixed.
- Detects wrong outputs: flips need labels. KL needs no labels, only the baseline. "AllFlips" (the paper's term in the appendix) adds wrong-to-wrong changes and needs no labels.
- Precision over time: fixed.
- Training-free.

**Full comparison.**
- *What it holds of our claim.* "Compressed models are better judged by distance metrics with respect to baseline." Its premise is that a compressed model should be judged against its own reference, and that accuracy hides divergence.
- *Where it falls short of what we do.*
  - It does not turn the distance into a detector. It proposes no threshold and no pass/fail verdict on a build.
  - It never separates broken from working builds. Its worst case is a 4-bit drift, not a silent fault such as 0% accuracy.
  - It uses the final-answer distribution only.
  - There is no on-device, runtime or toolchain axis.
  - There are no looped models.
- *Where we could fall short against it.* Our check is a state-distance on 2 loops. A reviewer will ask why that beats "just compare final outputs" (KL or agreement). We must run that baseline.
- *Settling experiment.* On the same 65 phone and runtime configurations, compute final-output agreement, KL on the final distribution, and flips (needs labels, so report it as an upper-bound reference), alongside the 2-loop state distance. Report AUROC for each. Also report each metric's cost: the number of loops run before the verdict.

**Take.**
- Cite as the premise that reference-comparison is the right lens.
- Its mechanism result is directly usable. The top-margin finding (answers with a small margin change first) predicts which problems the 2-loop check should flag first.
- Use it as framing: "accuracy is not enough" is a quotable motivation for judging a build by deviation, not accuracy.

### 2. Baek, Jiang, Raghunathan, Kolter, "Agreement-on-the-Line" (NeurIPS 2022)

- **Affiliations.** CMU; Kolter also at Bosch Center for AI. The NeurIPS 2022 header is in the paper. I did not open the proceedings page.
- **Read.** Abstract, intro, related work, results and method, partial appendix.
- **Credibility.** Strong. The senior author has refereed work in this exact subarea (Jiang et al., ICLR 2022). No slop flags.
- **Grade: anchor** (the proceedings page still to be confirmed).

**What it does.**
- When ID vs OOD accuracy is linear across a pool of models ("accuracy-on-the-line"), ID vs OOD agreement between model pairs is also linear, with matching slope and bias.
- That lets them predict OOD accuracy from unlabeled OOD data. The method is ALine-S/D.
- Mean absolute error is at most 2% where the line holds.
- It beats ATC, Average Confidence (AC) and DOC-Feat. On Camelyon17-WILDS, where the line does not hold: ALine-D 5.47, ALine-S 8.31, ATC 11.93, AC 13.30, DOC 13.57 (Table 2).
- It uses CIFAR-10.x, ImageNetV2, WILDS, CIFAR-C, etc.

**Diff.**
- Model class: one-pass classifiers.
- Hardware: server.
- Measured: accuracy under distribution shift.
- Compute per input: fixed.
- Detects wrong outputs: estimates average accuracy, label-free, using a pool of other models. It does not flag individual inputs or builds.
- Precision over time: not applicable.
- Trained models, estimator is training-free.

**Full comparison.**
- *What it holds.* "Agreement ... can be estimated with just unlabeled data" and "we can predict the OOD accuracy of classifiers". It owns label-free accuracy estimation via agreement.
- *Where it falls short of what we do.*
  - It needs a pool of different networks and a fitted line. It compares different models, not one model with its own reference.
  - Its failure mode is a shift it can only partly handle (the line does not hold).
  - It says nothing about numerical faults, toolchains or hardware.
- *Where we could fall short.* A reviewer may say "agreement estimates accuracy, so use it". Our reference is the same model on the host, so the agreement there is not a calibrated accuracy predictor. That is exactly Dutta's point: agreement between a model and its own reference can be high while accuracy differs, and the reverse.
- *Settling experiment.* Treat the reference run and the device run as the pair. Compute final-output agreement and check whether it predicts the accuracy drop, per configuration (65 phone configurations, TensorRT, runtime pairs).

**Take.** Cite for "agreement requires no labels". It shows the agreement-vs-accuracy link depends on the pool. Our setting is a single-pair, two-point comparison against a known-correct reference, which is a different object.

### 3. Jiang, Nagarajan, Baek, Kolter, "Assessing Generalization of SGD via Disagreement" (ICLR 2022)

- **Venue.** ICLR 2022, "Published as a conference paper" in the header. CMU and Google Research (Nagarajan).
- **Read.** Abstract, intro, setup, key claims, failure statements. Proofs and appendix not read in full.
- **Credibility.** Strong (Kolter, Nagarajan). No slop flags.
- **Grade: anchor.**

**What it does.**
- Two runs of the same architecture, same data, different SGD seeds. Their disagreement rate on unlabeled test data approximately equals the test error. This holds on SVHN and CIFAR with ResNet18.
- Theory: it follows if the ensemble is well-calibrated.
- Under distribution shift (PACS) it holds in some settings, "albeit not on all kinds" (Cartoon to Sketch: 0.27 vs 0.24 pretrained, 0.61 vs 0.51 not pretrained).

**Diff.** One-pass classifier, server, estimates test error, per-input compute fixed, label-free, no precision axis, training-free estimator.

**Full comparison.**
- *What it holds.* "unsupervised accuracy estimation is valuable for real-time evaluation of models when test labels are costly or unavailable".
- *Where it falls short of what we do.* The two models are independent trainings, not a deployed copy and its reference. A deployed model and its host reference share weights, so their disagreement measures numerical divergence, not generalization error.
- *Where we could fall short.* None on the main claim. It does warn us that disagreement equals error only under calibration, which we cannot assume for faulty builds.

**Take.** Use it to separate two things in the paper's text:
- disagreement between independently trained models (an error estimator), and
- disagreement between a model and its own reference (a fidelity measure).
This is the sentence that stops a reviewer folding our check into GDE.

### 4. Garg et al., "Leveraging Unlabeled Data to Predict OOD Performance" (ATC, ICLR 2022)

- **Venue.** ICLR 2022 header. CMU and Google Research. Code at github.com/saurabhgarg1996/ATC_code.
- **Read.** Abstract, intro, method, headline claims. Theory section and appendix not read in full.
- **Credibility.** Strong (Lipton, Balakrishnan, Sedghi, Neyshabur). No slop flags.
- **Grade: anchor.**

**What it does.**
- It learns a confidence threshold on labeled source validation data so that the fraction above the threshold matches validation accuracy. Predicted target accuracy is the fraction of unlabeled target examples above it. Score: negative entropy or max-confidence.
- It claims 2-4x lower estimation error than prior methods across WILDS, ImageNet, BREEDS, CIFAR and MNIST.
- It proves that, in general, identifying the accuracy is as hard as identifying the optimal predictor, so any method rests on unstated assumptions about the shift.

**Diff.** One-pass classifiers, server, accuracy estimate, label-free on the target but needs labeled source data and calibration, per-input compute fixed, no precision axis, training-free.

**Full comparison.**
- *What it holds.* "predicting the target domain accuracy using only labeled source data and unlabeled target data". It owns confidence-based, label-free accuracy prediction.
- *Where it falls short of what we do.* It reads the final confidence, so the model must finish. It cannot say a build is broken before the run completes. It also assumes the confidence distribution shifts smoothly. A constant-output fault (our TFLite Micro REDUCE_SUM bug) can produce high confidence on a wrong class, so a confidence rule may miss it. This is a hypothesis, not a result.
- *Where we could fall short.* On faults that damage the final answer but leave the 2-loop state similar, confidence will win. Our 0.67-0.73 across runtimes is where that risk lives.
- *Settling experiment.* Run ATC and max-confidence on the same builds. Calibrate the threshold on the host reference run, then score each device build by the fraction above it. Report AUROC for broken vs working, and report halting-head and loop-wise confidence as extra scores.

**Take.** This is the required confidence baseline. The impossibility result is a useful frame: no label-free estimator works without an assumption, and ours is "the reference run is correct and the fault is numerical".

### 5. Hao et al., "Nazar" (ASPLOS 2025)

- **Venue.** Proceedings of the ACM ASPLOS, March 2025 (doi 10.1145/3669940.3707246), per the authors' page at cs.columbia.edu. I read the arXiv v2 (2023). Columbia (Junfeng Yang, Asaf Cidon). Code: the authors' page lists none; the paper says it will open-source.
- **Read.** Sections 1-3, detection and evaluation parts. The root-cause analysis and adaptation parts only skimmed.
- **Credibility.** Strong. No slop flags.
- **Grade: anchor.** This is the only refereed, recent, on-device monitoring paper I found. It is ASPLOS, not MobiSys or MobiCom. I found no MobiSys or MobiCom paper on label-free accuracy monitoring. The search did not surface one, and I did not search the ACM proceedings directly. That gap is unverified, not confirmed.

**What it does.**
- It is an end-to-end system for monitoring and adapting models on phones, without user feedback.
- On-device detection is a threshold on the maximum softmax probability (default 0.9). It compared a wide range of detectors, and chose this one for being label-free, cheap and needing no surrogate model.
- The cloud does root-cause analysis across devices, then adapts by cause.
- Evaluated on Cityscapes and an ImageNet-derived animal app. It reports 15% average accuracy gain, up to 22% on all data, and 19% average and up to 50% on drifted data.

**Diff.** One-pass models. Fleet of devices. Measures data-drift detection (F1) and accuracy. No per-input compute change. Detects drifted inputs without labels or a reference. No precision change. Training-free detector.

**Full comparison.**
- *What it holds.* "the first end-to-end system for continuously monitoring ... models on mobile devices without requiring feedback from users."
- *Where it falls short of what we do.*
  - It detects input drift, not build faults.
  - It lists quantization and device discrepancies as a motivating problem ("accuracy degradation ... hard to anticipate in advance") and does not detect them.
  - It has no reference run.
  - It reads the final output.
- *Where we could fall short.* It is the working precedent for a cheap on-device, label-free detector. A reviewer will ask us to match its cost. Our check needs a host reference run, but it is done once per build, not per input.
- *Settling experiment.* Run the MSP threshold on every faulty phone configuration, and report its detection of the 15 faulty phone configurations. Also report the on-device cost of storing and comparing the 2-loop state.

**Take.** It gives the monitoring framing and cites our motivating problem as open. Cite the sentence on quantization degradation as a named gap.

### 6. Qiu et al., "ML-EXray: Visibility into ML Deployment on the Edge" (MLSys 2022)

- **Venue.** MLSys 2022, from memory. I read arXiv v1 (Nov 2021); I did not open the proceedings page. Stanford and Google (Pete Warden). Code was promised as open-source.
- **Read.** Sections 1-3.4 and the evaluation setup.
- **Credibility.** Strong, known lab, TensorFlow Lite team involved.
- **Grade: anchor** once the MLSys 2022 page is confirmed. Until then, support.

**What it does.**
- It logs per-layer tensors on the edge device and runs a reference pipeline on the same data.
- It first checks accuracy, and if there is a drop, it computes per-layer normalized rMSE between edge and reference to find the faulty op.
- It adds custom assertions for preprocessing bugs (channel order, normalization range, orientation).
- It reports that edge accuracy "can abruptly drop over 10% without obvious reasons", quantized ops that fail on specific chips, and fixes worth up to 30% accuracy.

**Diff.** One-pass CNNs on phones and edge devices (TFLite). Measures accuracy, per-layer error, latency. Compute per input fixed. Detects faults by comparison with a reference pipeline. The initial accuracy step uses labeled data (top-1). No precision change. Training-free.

**Full comparison.**
- *What it holds.* "compares performance differences and per-layer output discrepancies" against a reference pipeline, to catch "quantization issues". This is the closest prior art to comparing intermediate state of an edge build against a reference.
- *Where it falls short of what we do.*
  - The accuracy gate uses labels.
  - Per-layer rMSE is for diagnosis after a drop, not a label-free go/no-go verdict.
  - It runs the full model on both sides.
  - There are no loops, and no 2-loop-early check.
  - No AUROC over many builds.
- *Where we could fall short.* A reviewer may say "this is ML-EXray with a state at loop 2 instead of a layer". Our difference is that, for looped models, the state at loop 2 is a prefix of the computation that carries a prediction of the final failure, and the fault compounds across loops.
- *Settling experiment.* Implement the ML-EXray statistic (rMSE normalized by output scale) on our loop-2 state and on the final output. Compare AUROC with our statistic across the same 65 phone configurations and the cross-runtime pairs. Do the same for the layer-wise version on a one-loop slice.

**Take.** This is the paper to delineate against most carefully. It is the baseline, and it is also support for the claim that edge builds fail silently. Use it for the motivation numbers (over 10% drop, up to 30% fixed).

### 7. Gong et al., DAPPER (IMWUT 2023)

- **Venue.** Proc. ACM IMWUT 7(2), June 2023 (doi 10.1145/3596256). KAIST, Nokia Bell Labs, Tsinghua AIR. Ubicomp track. I read the abstract and intro only.
- **Grade: support** (refereed at a mobile-computing journal, not an A*/A ML venue). Read status: abstract and intro only, so cite=no until read further.
- It estimates performance after domain adaptation, label-free, from the mutual information between model inputs and outputs. It claims 39.8% better estimation accuracy than the best of six baselines and up to 396x less on-device computation. It targets sensing models after personalization, not builds.
- Take: a mobile-venue precedent for label-free performance estimation with on-device cost reported. Cite as a family member only.

## Items found but not relied on

- **FRAP (arXiv 2606.06335, 2026).** The search snippet says agreement-based scores can increase even when predictions are consistently wrong, and it proposes a foundation-model reference. Unread, preprint, cite=no. If true, it supports our warning that agreement alone can mislead. Read before use.
- **"Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs" (arXiv 2609.07664).** Title seen in search only. Unread, cite=no. It is close to Dutta and may be closest to our fidelity framing. Read first.
- **AETTA (arXiv 2404.01351), label-free accuracy estimation for test-time adaptation, using prediction disagreement across dropout samples (per the search summary).** The PDF text extraction came back empty, so unread. cite=no.
- **Confidence-based estimators (arXiv 2407.08649), and a 2025 benchmark of nuclear norm vs confidence (arXiv 2510.02956).** Snippets only. I found no refereed 2024-2026 main-track paper on label-free estimation. The recent ones I saw are arXiv preprints, and the "one recent refereed paper" requirement is met by Nazar (ASPLOS 2025) only on the monitoring side. This is still open for the estimation side.

## Strongest honest claim and the baselines we must run

**Claim.** The state after 2 loops, compared with a host reference run, flags a broken build without labels and before the full run. AUROC is 1.000 on phones and TensorRT. Across runtimes it drops to 0.67-0.73. Halting-sign agreement is weak (0.43-0.64) and should not be sold.

What it must not say:
- that label-free reference-comparison is new (ML-EXray and Dutta do the comparison);
- that it estimates accuracy (Baek, Jiang, Garg own that);
- that it is the first on-device label-free monitor (Nazar).

What it can say that these do not:
- a verdict on a build, from a prefix of the computation, before the answer exists;
- an explanation tied to the loop structure (errors compound across loops, so the early state already carries the fault).
That second point is our own measured finding. None of these papers tests it.

**Baselines to run, on the same builds.** Each should report AUROC for broken vs working, plus the number of loops run before the verdict.
1. Final-output agreement with the host reference (the label-free analogue of flips; Dutta's AllFlips; Baek's agreement).
2. KL divergence of the final output distribution against the host reference (Dutta).
3. Flips with labels, as an upper-bound reference only (Dutta). Report it as needing labels.
4. ATC and max-softmax-probability threshold, calibrated on the host reference (Garg, Nazar).
5. ML-EXray normalized rMSE, on the final output and on the 2-loop state, plus a layer-wise version (Qiu).
6. Our 2-loop state distance at loops 1, 2, 3 and N, to show where the signal appears and whether a prefix really is enough.

## Rankings

**Closest to our claims:** (1) ML-EXray, (2) Dutta et al. (flips), (3) Nazar.

**Strongest motivation:** (1) ML-EXray (edge accuracy drops over 10% silently, quantized ops failing on particular chips), (2) Dutta (accuracy hides divergence: flips up to 13.6% at equal accuracy), (3) Nazar (ops teams have no ground truth after deployment, and quantization degradation is hard to anticipate).

**Claims of ours this cluster leaves unclaimed:**
- A pass/fail detector of build faults scored by AUROC across 65 phone configurations and TensorRT builds.
- Detection from a prefix of the computation, before the final answer.
- The link to loop count (compounding across loops).
- Looped models, DEQs and diffusion samplers as the target class: none of the papers touches them.
- Repair by raising precision on the last loops.

**Claims of ours this cluster undercuts or narrows:**
- "Comparing against a reference without labels is new": no. Dutta (KL, AllFlips) and ML-EXray (per-layer rMSE) do this.
- "Label-free monitoring of on-device models is new": no. Nazar and DAPPER do it for drift and personalization.
- "A label-free signal predicts accuracy": not ours to claim. Garg shows it needs assumptions, and Baek's agreement needs a pool of models.
- The weak cross-runtime AUROC (0.67-0.73) is where final-output baselines may win. Until the six baselines above are run, we cannot say our check beats them.