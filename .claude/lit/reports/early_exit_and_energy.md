DEEPREAD REPORT, clusters D (early exit / adaptive compute on device) and E (energy and cost models; do FLOPs predict latency or energy). Date 2026-11-23.

READ STATUS, stated up front. All PDFs were fetched and text-extracted under .../scratchpad/papers/<id>/. I read abstracts, method, setup and every energy, baseline, power-mode and thermal passage of each paper, by section and grep. I did not read every paragraph of every paper, so the status is "sections" throughout. The ledger needs `read=sections`, not `full`, unless you want a second pass. Not obtained: HarvNet (MobiSys 2023; the ACM link returned HTML, and the NSF link was a different paper). Paper bodies were treated as data.

SHORT ANSWERS

1. Energy per correct answer on device: not found in anything I read. The nearest items are:
 - Battery Price: J/token plus an accuracy-energy Pareto. Accuracy is measured on the server only, so no per-answer join exists.
 - EE-HI (2407.11061): energy per inference at an accuracy constraint.
 - A search snippet only, unread: Wu et al. (arXiv 2004.11293, energy-harvesting multi-exit) report "correctly processed events per energy unit, 3.6x". That is close to our metric. Read it before claiming novelty on the metric.
2. Early exit against the best fixed depth at equal accuracy: no paper uses the best fixed depth of the same network as the baseline.
 - BranchyNet and SPINN compare with the final exit or with offloading systems.
 - The closest is FLEET (MobiCom 2023), whose "Ensemble" baseline is a series of smaller fixed models at different compute levels, compared at about equal accuracy. It reports latency, and energy is partly datasheet-derived.
3. FLOPs mispredict:
 - nn-Meter (MobiSys 2021): a FLOPs predictor lands within ±10% of measured latency on 22.1% of models, against 89.2% for nn-Meter, averaged over three devices.
 - Yang et al. (CVPR 2017): energy is modeled, not measured. Pruned AlexNet keeps 15% of its MACs but 27% of its energy.
 - Both are across-architecture claims. Neither tests a repeated identical step, which is where our law sits.
4. Power modes and DVFS: there is no numeric thermal-drift result anywhere in this cluster. Details are under the question list below.

PER PAPER

1. nn-Meter: Towards Accurate Latency Prediction of DNN Inference on Diverse Edge Devices
- Authors: Zhang, Han, Wei, Zheng, Cao, Yang, Liu. Microsoft Research, Tsinghua AIR.
- Venue: MobiSys 2021 (DOI 10.1145/3458864.3467882; best paper per the MS Research page; the footer of the PDF carries the ACM citation). Source: air.tsinghua.edu.cn copy of the camera-ready. Code: github.com/microsoft/nn-Meter.
- Read status: sections (method skimmed, evaluation and discussion read).
- Flags: none. Proposed grade: anchor.
- What it does: it detects kernel-fusion rules by black-box test cases, then predicts latency as a sum over kernels. Targets are Cortex-A76 CPU, Adreno GPU and Myriad X VPU, on a 26,000-model CNN dataset. Accuracy within ±10% is 99.0 / 99.1 / 83.4% (CPU / GPU / VPU), 89.2% on average. FLOPs gives 22.1%, FLOPs+MAC 17.1%, and the graph-learning predictor BRP-NAS 8.5%. Latency is measured over 50 runs with the CPU pinned at 2.42 GHz.
- Diff vs us:
 - Model class: one-pass CNN.
 - Hardware: measured on device.
 - Measured: latency only.
 - Compute per input: fixed.
 - Error detection: none.
 - Precision over time: no.
 - Training: trained regressors.
- Comparison:
 - Held of our claim: "the FLOPs method ... is simple but not a direct metric of latency", plus the kernel-fusion explanation of why.
 - Short of us: no energy, no variable depth, and thermal modeling is waved off ("difficult to conduct thermal or heat modeling, since heat dissipation depends on the external environment"). It says energy prediction "should be straightforward" by retraining on energy data, and does not do it.
 - Where we could fall short: the existing answer to "FLOPs mispredict" is a per-kernel predictor, 89% within ±10%. Our law must be argued as simpler or equally accurate for looped models. At 0.5% / 1.1% median error it is far tighter than 89% within ±10%, but the error definitions differ.
 - Settling experiment: run an nn-Meter-style kernel predictor on our boards for one loop step and compare it with the measured per-step constant.
- Take: cite as "FLOPs mispredict latency, and the fix is per-kernel, one-pass". Our step re-executes the same kernels, so a single measured constant replaces the predictor. It also fixed the CPU clock to avoid DVFS noise, which matches our need to report power-mode state.

2. SPINN: Synergistic Progressive Inference of Neural Networks over Device and Cloud
- Authors: Laskaridis, Venieris, Almeida, Leontiadis, Lane. Samsung AI Center Cambridge, Cambridge.
- Venue: MobiCom 2020 (stated on the preprint header; I did not open the program page). arXiv 2008.06402.
- Read status: sections.
- Flags: none. Proposed grade: anchor. Note it is outside the 2023-26 window and the code status is unknown.
- What it does: it jointly picks a split point and a confidence threshold (early exits trained as BranchyNet-style heads) under latency, throughput and accuracy-tolerance constraints. The testbed is a Jetson AGX Xavier client in three profiles (30 W, 10 W, underclocked 10 W) and a server, with ResNet-50/56, VGG16, MobileNetV2 and Inception v3 on CIFAR-100 and ImageNet.
- Numbers: up to 2x throughput, 6.8x lower server cost, +20.7% accuracy under a latency SLA. Energy is measured on 1000 inferences with Jetson probes plus a Monsoon meter on the radio. The spread across configurations is 5x, on par with on-device (0.76-1.12x), and 4.2x better than full offload.
- Diff vs us:
 - Model class: early-exit CNN.
 - Hardware: measured, with a modeled network.
 - Measured: latency, throughput, energy per sample (mJ).
 - Compute per input: varies, decided by softmax confidence.
 - Error detection: none.
 - Precision over time: no (8-bit transfer compression only).
 - Training: trained exits.
- Comparison:
 - Held of our claim: per-input early stopping on a Jetson at several power profiles, with energy reported per sample.
 - Its own admission, which helps us: "the scheduler does not explicitly optimise for energy ... of the client."
 - Short of us: baselines are Neurosurgeon, Edgent, device-only and server-only, never the best fixed exit, and energy per sample is not per correct answer. Power profiles change client speed, not an energy-per-step analysis.
 - Where we could fall short: it is on a Jetson AGX with early exit and a confidence threshold. Our stopping on AGX must be shown to differ in kind (weight-tied loops, the model's own halting head) and not only in baseline choice.
 - Settling experiment: add "best fixed depth at equal accuracy" as an arm on their kind of network. We do this for loops.
- Take: the 10 W / 30 W / underclocked-10 W profile design is a precedent for our power-mode arm, and a source for the claim that nobody reports energy per correct answer.

3. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks
- Authors: Teerapittayanon, McDanel, Kung. Harvard.
- Venue: ICPR 2016 (not verified on a proceedings page; the arXiv copy 1709.01686 is dated Sep 2017). Read: sections.
- Flags: none. Proposed grade: support (venue not verified as A*/A).
- What it does: it adds branch exits to LeNet, AlexNet and ResNet-110 and exits on softmax entropy below a threshold. Speedups at the "knee" threshold are 5.4x/4.7x (CPU/GPU) on LeNet, 1.5x/2.4x on AlexNet, 1.9x/1.9x on ResNet. Hardware is a 3.0 GHz CPU and a Titan X.
- Diff vs us:
 - Model class: early-exit CNN.
 - Hardware: server.
 - Measured: runtime. Energy is named in the abstract and never measured.
 - Compute per input: varies, decided by entropy.
 - Error detection: none.
 - Precision over time: no.
 - Training: trained jointly.
- Comparison:
 - Held of our claim: per-input variable depth saves compute.
 - Short of us: energy is asserted, not measured. The baseline is the single original network at its final exit. The paper itself notes accuracy is slightly below baseline for ResNet at the chosen knee.
- Take: cite for the idea. State plainly that the baseline is the final exit, the same gap as SPINN.

4. Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges (survey slot)
- Authors: Matsubara, Levorato, Restuccia. UC Irvine, Northeastern.
- Venue: ACM Computing Surveys 2022 (DOI 10.1145/3527155; not opened on the ACM page). arXiv 2103.04505. Read: sections (early-exit chapters and research challenges).
- Flags: none. Proposed grade: anchor.
- What it does: it surveys split computing and early exit. Its research challenges say users must hand-pick a confidence threshold per exit, and it suggests learning where to stop without thresholds. It also says exiting has been studied mostly on image classification.
- Comparison:
 - It names no energy-per-correct-answer metric and no best-fixed-depth comparison.
 - It supports "thresholds are hand-set and the decision rule is open", which fits our halting-head and residual rules.
 - It is split-computing heavy, so it is thin on energy measurement.
- Take: cite as the survey of record for early exit on edge. I picked this one; the other candidate was Laskaridis et al. 2106.05022 (EMDL workshop; not refereed at this level).

5. MELTing Point: Mobile Evaluation of Language Transformers
- Authors: Laskaridis, Katevas, Minto, Haddadi. Brave Software, Imperial College.
- Venue: MobiCom 2024, pp. 890-907 (confirmed via a researchr listing and the arXiv comment, not the program page). Code released as a device-farm tool. arXiv 2403.12844, read from v2.
- Read status: sections.
- Flags: none. Proposed grade: anchor.
- What it does: a device farm (Galaxy S23, Pixel 6a, iPhone 14 Pro, Jetson AGX Orin and Nano) with a Monsoon meter at 5 kHz and a thermal camera. It benchmarks LLMs with MLC, llama.cpp and LLMFarm, and reports throughput, energy per token, thermal state and accuracy under quantization.
- Numbers:
 - 0.21 / 0.20 / 0.16 mWh/token (S23 and iPhone 14 Pro on MLC-Chat, iPhone 14 Pro on LLMFarm).
 - 542 / 490 / 591 prompts per battery (40 input and 135 output tokens).
 - iPhone sustained draw 13.8 W (instantaneous over 18 W) against under 8.5 W sustained on the S23.
 - 47.9 C surface temperature.
 - Higher throughput variance is attributed to variable context and generation length "as well as potential DVFS".
 - On Jetson, "efficiency is moving the same direction as the device's TDP". The paper's explanation is that the lower power mode scales the memory clock down and the GPU stalls on memory. I did not extract the figure's numbers.
- Diff vs us:
 - Model class: autoregressive LLM.
 - Hardware: measured, with direct power at the battery terminals.
 - Measured: energy, thermal state, throughput, accuracy.
 - Compute per input: output length only.
 - Error detection: none.
 - Precision over time: no.
 - Training: n/a.
- Comparison:
 - Held of our claim: measured power-mode effects on Jetson, thermal behavior on phones, and DVFS named as a cause of variance.
 - Short of us: DVFS and throttling are discussed, never quantified as drift of a per-step constant. The unit is per token, not per correct answer, and there is no adaptive stopping.
 - Where we could fall short: its power-mode finding (higher mode, better energy per token) must be compatible with ours. Our 26-32% per-step spread across modes needs its direction stated beside this.
 - Settling experiment: per-token energy against power mode on the Nano and AGX with our own harness, next to our per-step number.
- Take: methodology and a citable motivation for the thermal and power-mode caveats. It was written by SPINN's first author, so it links the two.

6. The Battery Price of Edge AI: A study of the Environmental Impact of LLM Inference on Mobile Devices (verified to exist: arXiv 2609.11940)
- Authors: Guegain (Greenspector), Coignion (Univ. Bordeaux, LaBRI). arXiv cs.PF.
- Venue: preprint. Read status: sections.
- Flags:
 - The id prefix 2609 (September 2026) does not match the stamp "10 Jul 2026". I could not resolve this.
 - The replication package is listed as "Anonymous" on Zenodo.
 - Small n: 18 trials per configuration per device, two phones.
 - Accuracy is measured on the server only, on 200 samples per benchmark.
 - Android energy comes from the battery-discharge API, which is coarse.
 - Proposed grade: caution.
- What it does: 18 LLM configurations on a Pixel 8, an iPhone 14 and an A100 server with llama.cpp. On-device energy per token is 3x the batched-server figure. Quantization shows a non-monotonic energy curve with a sweet spot at 4 bits. 8 of 18 configurations lie on the accuracy-energy Pareto front. 88-90% of local impact is device embodied carbon.
- Diff vs us:
 - Model class: autoregressive LLM.
 - Hardware: measured on phones.
 - Measured: J/token and server accuracy.
 - Compute per input: output length only.
 - Error detection: none.
 - Training: n/a.
- Comparison, important for our law: "per-token energy is consistent across the three answer lengths" (100, 500, 1000 words). That is the "steps x per-step constant" law for autoregressive decoding, stated in a sentence. It is a measured, if lightly controlled, precedent.
- Take: it narrows what we may call new. It strengthens motivation by showing the accuracy-energy joint is already being asked for, though only across configurations and not per answer. Use as support after the date anomaly is settled.

7. When NPUs Are Not Always Faster: A Stage-Level Analysis of Mobile LLM Inference (verified: arXiv 2605.27435)
- Authors: Li, Qi, Chen. Leiden University (LIACS), NWO-funded. Qinyu Chen's prior refereed hardware record is not verified.
- Venue: preprint (cs.AR, 22 May 2026), 5 pages. Read status: full text, short.
- Flags: one phone (Snapdragon 8 Gen 3), llama.cpp, four Q4_0 models. The energy experiment covers one model (Llama-3.2-3B) and is a single paragraph. No code link seen. "First" claims. Proposed grade: caution.
- What it does: on a Snapdragon 8 Gen 3 (Hexagon v75 NPU), CPU is up to 1.6x faster in prefill. NPU decode gains are only 1.05-1.2x. Offloading raises device battery drain by 22 / 32 / 51% at 321 / 641 / 1281 prompt tokens.
- Comparison: it backs "NPU offload is not a free win for energy" and the dispatch tax (8-22x call/op time on lightweight operators). That matches the shape of our toolchain-fault story, though ours is silent correctness failures, not slowdowns. Nothing on adaptive compute.
- Take: citable as one data point after a closer read of its energy method. Do not lean on it.

8. Yang, Chen, Sze: Designing Energy-Efficient CNNs using Energy-Aware Pruning
- Venue: CVPR 2017, pp. 5687-5695 (reference [160] in the survey above). arXiv 1611.05128. MIT; Sze has a long refereed record on energy for DNN accelerators.
- Read status: sections (energy-estimation method and results).
- Flags: none. Proposed grade: anchor.
- What it does: it builds an energy model from MAC counts plus memory-hierarchy data movement, using numbers extrapolated from hardware measurements. It prunes by that energy.
- Numbers: AlexNet energy falls 3.7x and GoogLeNet 1.6x at under 1% top-5 loss. Energy ratio against MAC ratio: AlexNet 27% vs 15%, GoogLeNet 62% vs 29%, SqueezeNet 76% vs 43%. SqueezeNet, with 50x fewer weights than AlexNet, uses 33% more energy.
- Caveat: energy is estimated, not measured on a device.
- Comparison: FLOPs mispredict energy by about 1.7-2.1x on these pruning ratios. That is inside our 2.2-8x. The new thing in ours is a measured, board-specific miss rather than an estimator's.
- Take: cite for "MACs are not energy".

9. FLEET: Re-thinking computation offload for efficient inference on IoT devices with duty-cycled radios (MobiCom 2023)
- Authors: Huang, Guan, Ganesan. UMass Amherst. Venue from the PDF filename and search; not opened on the program page. Read status: sections.
- Flags: none. Proposed grade: anchor.
- What it does: early-exit offload over BLE with prioritized communication, dynamic pooling and feature fusion. Platforms are Cortex-M33, GAP8 and a Raspberry Pi Cortex-A77. It reports 1.7-6.1x latency speedup and 1.6-4.4x energy savings over JPEG offload, and 1.2-2.5x energy savings over the best baseline. Baselines compared at about equal accuracy are BranchyNet, Shallow-Deep Networks and Ensemble (a series of smaller models), plus two partitioning baselines.
- Energy method: measured on the Pi; computed from datasheet power on the M33 and GAP8.
- Comparison:
 - This is the paper closest to our "compare at equal accuracy" step. The Ensemble baseline is a fixed-depth-style comparator. Per inference, not per correct answer.
 - It does not use a weight-tied loop, so "best fixed loop count" is untested.
 - Its energy on two of three platforms is power x time, our law's structure.
- Take: cite as the nearest equal-accuracy precedent, and borrow its protocol.

10. Fast-Inf: Ultra-Fast Embedded Intelligence on the Batteryless Edge (SenSys 2024)
- Authors: Custode, Farina, Yildiz, Kilic, Yildirim, Iacca. University of Trento, Ege University. DOI 10.1145/3666025.3699335 from the PDF filename; not opened on the program page. Read status: sections.
- Flags: none. Proposed grade: anchor.
- What it does: binary-tree networks with a "leaf truncation" that skips leaves at runtime when stored energy is low, on an MSP430FR5994. It reports up to 700x speedup and lower energy than a conventional DNN, with energy measured via EnergyTrace and controlled power failures.
- Comparison: adaptive depth, measured energy, no per-correct-answer metric. Truncation is driven by energy state, not the input. It states that early-exit branches add memory and compute overhead, as a motivation for its design. Weaker overlap than FLEET.

11. Leads not read
- EE-HI (arXiv 2407.11061): preprint from IMDEA/KTH authors with released code. It measures energy per inference (Voltech PM1000+ analyzer, average power x time) on five devices and defines EPI against an accuracy QoS constraint. This is the nearest on-device "energy at an accuracy constraint" I read. Preprint, partially read. Proposed grade: support.
- AdaScale (arXiv 2412.00724): multi-branch early exit, energy modeled from a power x time formula plus empirical coefficients, not measured. Authors: Northwestern Polytechnical. Preprint. Proposed grade: caution.
- Unread leads: HarvNet (MobiSys 2023, DOI 10.1145/3581791.3596845), Wu et al. (2004.11293), UniExit (2611.09159), 2609.11939, 2305.14094.

QUESTION BY QUESTION

Q1. Energy per correct answer on device? Not in anything I read. SPINN and FLEET report energy per sample or inference. MELT and Battery Price report per token. EE-HI reports energy per inference under an accuracy floor. The unread Wu et al. snippet uses correctly-processed events per energy unit on energy-harvesting hardware; check it before the claim goes in the paper.

Q2. Early exit against the best fixed depth? None of the read papers makes this comparison for a single network. BranchyNet and SPINN compare against the final exit (SPINN also against offloading systems). FLEET compares against smaller separate models at equal accuracy, which is the closest. The best fixed loop count of a weight-tied network is not in this literature. The literature does not use it because ordinary exit networks have no clean fixed-depth variant of the same weights.

Q3. FLOPs and energy. Prior work:
 - nn-Meter: FLOPs within ±10% of latency on 22.1% of models.
 - Yang: MAC ratio understates energy ratio by 1.7-2.1x.
 - NPU paper (caution): energy rises 22-51% when more work is offloaded to the faster-looking unit.
Whether the "steps x per-step constant" law adds anything: its added content is not "FLOPs fail", which is old. It is (a) the constant is board, precision and power-mode specific and is one measurement, (b) it predicts across loop counts at 0.5% / 1.1% median error, and (c) it ties to per-input stopping. Battery Price already states the same linearity for autoregressive tokens, so present the law as a measured instance for looped models, not a new principle. The error figures use different definitions from nn-Meter's ±10% accuracy.

Q4. Power modes and DVFS.
 - SPINN: three Jetson profiles used to vary device speed; no energy-per-mode analysis.
 - MELT: Jetson energy per token improves with a higher power mode (memory-clock scaling in low modes). Phone throughput variance attributed to DVFS. Surface temperature 47.9 C. No numeric drift.
 - nn-Meter: pins the CPU clock; says thermal modeling is difficult.
 - Our 26-32% per step across power modes and 14.5% drift over 30 minutes at 10 W have no numeric counterpart here. To compare directions with MELT, we need our per-mode per-step values on AGX and Nano side by side.

RANKED LISTS

Closest to our claims:
 1. FLEET (equal-accuracy early-exit comparison, power x time energy).
 2. SPINN (early exit on a Jetson at three power profiles, energy per sample).
 3. EE-HI (energy per inference at an accuracy constraint, on devices; preprint).

Strongest motivation:
 1. nn-Meter (FLOPs 22.1% vs 89.2%; Zhang et al. do not do energy or thermals).
 2. MELT (energy, DVFS and thermal variance on real devices; explicitly leaves the quantification open).
 3. Yang et al. (MACs do not give energy; 3.7x from energy-aware pruning). The NPU paper (22-51% more drain) is a fourth, but caution-graded.

Claims of ours this cluster leaves unclaimed:
 - Energy per correct answer, measured on device, for per-input stopping.
 - Early stopping against the best fixed loop count at equal accuracy on one set of weights.
 - A measured per-step constant for looped models, with a median error of 0.5% / 1.1%.
 - A quantified thermal drift (14.5% over 30 minutes at 10 W) and a per-mode per-step spread (26-32%).
 - A FLOP-based miss of 2.2-8x for looped models specifically.

Claims of ours this cluster narrows or undercuts:
 - "Energy scales linearly with steps" as a general finding: Battery Price states it for LLM tokens, so the novelty is for looped models and per-input stopping.
 - "FLOPs mispredict energy": old (nn-Meter, Yang); claim only the measured magnitude for loops.
 - "Early exit saves energy": SPINN and BranchyNet precede us. Ours is the comparison and the metric, not the idea.
 - A low power mode helping stopping (3.2x): MELT suggests lower modes can cost more energy per unit of work on Jetson. Show absolute energy beside the ratio.

Ledger rows to add: nnmeter (MobiSys 2021, anchor), 2008.06402 (MobiCom 2020, anchor), 1709.01686 (ICPR 2016 unverified, support), 2103.04505 (CSUR 2022, anchor), 2403.12844 (MobiCom 2024, anchor), 2609.11940 (preprint, caution, date anomaly), 2605.27435 (preprint, caution), 1611.05128 (CVPR 2017, anchor), fleet_mobicom23 (anchor), fastinf_sensys24 (anchor), 2407.11061 (preprint, support), 2412.00724 (preprint, caution); all `read=sections`.

Files: /tmp/claude-1000/-home-stevenkolawole-workspace-emnlp-26/3c59b961-66b1-4d0d-abd5-a644e29d970e/scratchpad/papers/ (one folder per paper; r.txt is the extracted text).