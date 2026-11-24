Both papers read in full from the arXiv PDFs (pdftotext, main text, appendices, references). Texts are in the scratchpad: .../papers/2511.20621/q.txt and .../papers/2606.04594/q.txt. I did not fetch proceedings or OpenReview pages and did not fetch any cited paper, so the venue and credibility entries below are what the PDFs themselves state.

## 1. DiFR (2511.20621)

**Metadata.** "DiFR: Inference Verification Despite Nondeterminism." Karvonen*, Reuter*, Rinberg, Marks, Garriga-Alonso, Warr. Affiliations: MATS, Harvard, FAR AI, Anthropic (Warr). The PDF is marked "Preprint", v1, 25 Nov 2025. Code is released (github.com/adamkarvonen/difr, vLLM integration).
- Credibility: the authors are an alignment/safety group. Two of them (Rinberg, Karvonen) wrote the concurrent steganography/exfiltration verification paper they cite. I found no refereed record in systems or quantization in the PDF.
- Slop flags: none. The numbers agree across text, tables and figures. The setup is real (3 models, 9 configs each, about 1M tokens per config). The limitations section is honest.
- Proposed grade: **support** (preprint, code released, careful method).
- Read status: full.

**What it does.** A provider serves a model; a verifier replays the output on a trusted implementation and scores the divergence.
- Token-DiFR gives the verifier the sampling seed and scores the Gumbel-max logit gap between the provider's token and the verifier's token.
- Activation-DiFR sends a random-orthogonal-projection fingerprint of the activations, k=2..32 floats per token, and compares it with the verifier's own by L2 distance.
- Models: Llama-3.1-8B, Qwen3-8B, Qwen3-30B-A3B on H200/A100 GPUs. Data: 2,000 UltraChat prompts.
- Headline numbers, quoted: "detecting 4-bit quantization with AUC > 0.999 within 300 output tokens" (Token-DiFR). "Activation-DiFR detects 4-bit quantization with AUC > 0.999 using just 2 output tokens", with 25-75% less communication than TOPLOC.
- Caveat on the "2 tokens" claim: the metric is AUC at 1% FPR. Large deviations need about 4 tokens. FP8 KV-cache quantization needs 1-5 tokens on Llama and Qwen3-8B, and 16-20 on the noisy Qwen3-30B-A3B mismatched setup.

**Diff against ours, one line each.**
- Model class: autoregressive LLM, one forward pass per token. Not looped.
- Hardware: server GPUs only (H200, A100). No edge devices, no NPUs.
- Measured: the divergence score and detection AUC. Not task accuracy, latency or energy.
- Per-input compute: none varies.
- Detects incorrect outputs: yes, against a trusted reference, with no labels. A threshold and classifier are fit on honest-versus-misconfigured scores (50/50 token-level split).
- Precision over time: no.
- Training-free: the detector is training-free apart from the fitted logistic classifier or threshold.

**Full comparison (rule 5).**

*What of our check it holds.* The idea of comparing the target's internal state to a trusted reference and getting a label-free PASS/FAIL signal for a quantized build:
- "Activation-DiFR... verifies inference by comparing internal model state from the provider against those computed by a trusted verifier."
- A trusted reference with a tolerated-noise band: "Pooling scores... defines the range of benign numerical variation allowed under H0."

*Our questions.*
- **Reference.** A trusted implementation of the same model in a fixed spec (bf16, vLLM, the advertised config). The reference is the trusted side by definition. The word "trusted" appears throughout, and no section examines a wrong reference. A related effect appears only as benign hardware noise: for Qwen3-30B-A3B they note "not all honest implementations behaved equally" (the HuggingFace path gave benign Token-DiFR gaps about as large as FP8 KV quantization). They then remove it from the null and treat it as a policy choice ("up to the end user"). That is the closest they come to an implementation difference, and it is one the end user can discount. Nothing like our "host quantized simulation was itself the broken side: 0% vs 58%" case appears. They never compare against a simulated version of the target's quantized graph. Their reference is the full-precision-style spec. Our fp32 host reference is consistent with their design, and our 0.756 for the host-quantized-simulation reference is a failure mode they do not discuss.
- **Link to task accuracy.** None. All detection is about divergence from the reference. They quote Kimi K2 vendors at "61-72% similarity" from a third party (Moonshot's K2 vendor verifier) and Anthropic's postmortem as motivation. They report no accuracy on any benchmark.
- **Predict device accuracy.** No.
- **Looped or compounding errors.** Not addressed. Only the 4-bit signal grows with batch size, for the usual reason.

*Where it falls short of us.*
- Single pass: no loop count.
- No edge or NPU toolchain, so no faults like ours, which are positional, constant-output or QDQ-related.
- No accuracy tie-in.
- No accuracy prediction.
- Their AUC is at 1% FPR over token batches. Ours is PASS/FAIL AUROC over 80 builds.
- They compare activations of the same tokens, not a state refined over iterations.
- No per-loop analysis (our 0.972 after loop 1 vs 1.000 after loop 2).

*Where we could fall short against them.*
- Their detector has a calibrated null from pooled honest hardware variation (GPU type, TP, engine). Ours uses a separate fp32 host reference on different hardware. If we cannot show our PASS builds sit inside a honest-noise band across hardware, a reviewer can ask for it.
- Their fingerprints are tiny (under 1 byte per token to 30 bytes). Ours moves a full state.
- They test adversarial manipulation (the temperature attack that defeats cross-entropy). We have no such test.
- Their abstract says "detect infrastructure bugs before they affect users at scale", meaning cloud LLM serving. Comparing a computation with a reference run is a generic idea that predates DiFR; it is not theirs to hold, and we do not frame our check as theirs.

*Settling experiments.*
- Run their Activation-DiFR (random orthogonal projection, k=8/32, L2 distance, logistic regression) on our after-loop-2 states for the same 80 builds, with the same fp32 reference. Report AUROC, and AUC at 1% FPR.
- Run it on the same states after loop 1 and 3, and on a one-pass model with the same quantization, to show the loop-compounding effect they lack.
- Run their "pooled honest baseline" approach: gather benign cross-hardware fp32 variation and show our PASS builds fall inside it.

**What to take.**
- The baseline: Activation-DiFR with k=2/8/32 and the tolerated-noise-band calibration.
- The cross-entropy attack as a framing of why a reference comparison beats an output-statistics check.
- The "inference infrastructure bugs go undetected for weeks" motivation (Anthropic postmortem; Kimi K2 vendor spread), both cited in the intro, which we can cite as motivation for silent faults.
- Setting: theirs is LLM serving on servers. Ours is screening compiled edge builds of looped models before deployment, where the check must pass quantized builds that keep their accuracy and fail broken ones.

## 2. Ekka (2606.04594)

**Metadata.** "EKKA: Automated Diagnosis of Silent Errors in LLM Inference." Gu, Zhang, Zhu, Fu, Wu, Wang, Kasikci. University of Washington and AWS. The PDF's front matter reads "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026" (arXiv v2, 3 Nov 2026). Code at github.com/uw-syfi/Ekka.
- Credibility: Kasikci is a known systems author. The same group's TTrace (Jiang, Zhu, Zhang, ... Li) is cited as the precision-aware error-ratio source. I did not verify the ICML program page. I am relying on the PDF's own front matter.
- Slop flags: none. The benchmark is small (17 bugs, 9 for ablations) and honestly described. The excluded cases are listed with reasons.
- Proposed grade: **anchor if the ICML 2026 listing is confirmed on the proceedings page; otherwise support.** Rule 2 requires verification, so list it as support until Steven or the next agent checks.
- Read status: full.

**What it does.** For a buggy serving engine (vLLM, SGLang) it takes a reference (HuggingFace) and runs an LLM agent (Claude Sonnet 4.5) through component mapping, activation alignment and error analysis.
- Its error ratio is ||X_T − Y_R|| / ||X_R − Y_R||, where X_T is the target (low precision), X_R the reference in low precision, and Y_R the reference in fp32.
- It takes change-point analysis on the error ratio to name the first module where the divergence persists.
- Headline numbers: 80% pass@1 and 88% pass@5 on 17 real silent errors, 24-34% over OpenCode and Mini-SWE-Agent, about $30 per case, and 4 new bugs confirmed by developers.
- Threshold 1.5: non-buggy pairs have p99 of 0.93-1.41, buggy components range from 3.13 to 1093.75.

**Diff against ours.**
- Model class: autoregressive LLMs, plus one diffusion pipeline (Qwen Image on SGLang) and embedding or reranker models. Not looped.
- Hardware: server GPUs.
- Measured: diagnosis accuracy (which component). Accuracy regression is their motivation (43.8% of 70 studied bugs), not an output of the detector.
- Per-input compute: none.
- Detects incorrect outputs: it localizes after a bug is known (a reproduction script is an input). It does not decide PASS or FAIL for an unknown build. The authors say so: "Recent work targets silent errors but mainly for detection instead of diagnosis", and they cite TTrace and a USENIX OSDI '25 paper for detection.
- Precision over time: no.
- Training-free: yes, but agent-based with an LLM.

**Full comparison.**

*What it holds.* Aligning intermediate states against a reference, with the fp32 reference as the unit of error: "normalizes by the reference framework's precision error", "naively calculating the L2-norm or absolute error could be suboptimal". That is the same idea as our fp32 host reference. Also "Error accumulates across layers" in the discussion of numerical precision.

*Our questions.*
- **Reference.** HuggingFace in fp32 (the normaliser) and HuggingFace in low precision. Text: "a semantically correct reference implementation (e.g., HuggingFace Transformers) often exists and can serve as an oracle." They state a limit: Ekka is "most effective for silent errors that induce reproducible activation divergence and have a reference implementation to compare with."
- **When the reference is wrong.** Not handled. The knowledge base is built only from "correct implementation of the models", verified by "no gap" on downstream accuracy between target and reference (this is their one link between accuracy and the reference). If the HF reference were the broken side, nothing in the pipeline would notice. That is the gap our 0% vs 58% case fills.
- **Tie to task accuracy.** Only as motivation and as the filter for what counts as a correct implementation. The detector itself is not tied to accuracy.
- **Predict device accuracy.** No.
- **Per-step / per-module localization.** They localize the first module where the error persists. We do not localize; our state is one number after loop 2. Their change-point idea could apply across loops for a looped model.

*Where it falls short of us.*
- It is not label-free build screening: a user must already have noticed an accuracy regression and reproduced it.
- No looped models.
- No NPU or phone toolchains.
- No PASS/FAIL AUROC.
- It does not use the sustained-compounding structure as a detector; it uses it only to find the first component.

*Where we could fall short.*
- Their error ratio normalizes by the reference's own low-precision error, so benign precision noise is calibrated per component. Our statistic against fp32 has no such per-build normalisation. A reviewer could say our threshold absorbs the noise of the device's precision.
- They localize; we only classify. We could add per-loop localisation ("fails by loop 1").

*Settling experiments.*
- Compute their error ratio (target vs fp32, normalised by the reference's bf16/fp16-vs-fp32 error) on our states after each loop, and compare AUROC with ours.
- Run their threshold 1.5 as a rule on our 80 builds.

**What to take.** The error-ratio metric as a baseline and a normalisation. The 1.5 threshold. The bug taxonomy: 19.4% of silent errors are numerical-precision only, which matches the part of our faults that are quantization. The motivating Gemma 3 case (Hellaswag down nearly 30%, "months misdiagnosing"). The finding that comparing against a reference is what developers do in about 50% of cases (Figure 3): motivation for automating a check.

## Answers to the five questions

1. **Which part of our check each holds.**
   - DiFR verifies a cloud provider: "verifies inference by comparing internal model state from the provider against those computed by a trusted verifier." It flags any departure from the trusted model, so a correct quantized build counts as a deviation; it does not separate accurate quantized builds from broken ones.
   - Ekka debugs a serving engine once a bug is known, localizing layers: "normalizes by the reference framework's precision error."
   - Neither holds loops, edge devices, or the compounding-after-2-loops signal.
2. **Reference and when it is wrong.** DiFR: a trusted bf16 implementation, calibrated against honest hardware noise. Ekka: HuggingFace plus its fp32 run. Neither addresses a wrong reference. DiFR only touches a benign implementation gap (HuggingFace on Qwen3-30B-A3B) and handles it as a policy choice, and Ekka assumes the reference is correct (it builds its knowledge base only from verified-correct models). Our 0% vs 58% case is unclaimed.
3. **Tie the detector to task accuracy.** No for both. DiFR has no accuracy numbers. Ekka uses benchmark accuracy only to define bugs and to validate a correct implementation.
4. **Predict accuracy on the device.** No for both. Our host-curve-plus-one-latency prediction is unclaimed.
5. **Baselines we must run.** (a) DiFR Activation-DiFR (random projection, k=2/8/32, L2, logistic regression) on our after-loop states. (b) Ekka's error ratio and 1.5 threshold, with the fp32/low-precision normaliser. (c) A cross-hardware honest-noise band for DiFR's pooled null. Also a one-pass control, so the loop-compounding claim is not confounded.

## Ranked lists (my two papers only)

- **Closest to our claims:** 1. DiFR (Activation-DiFR). 2. Ekka. Nothing else in this cluster.
- **Most strengthen motivation:** 1. Ekka (accuracy regressions are 43.8% of silent errors; Gemma 3 Hellaswag down nearly 30%; ~50% of developers diff against a reference by hand; 19.4% purely numerical). 2. DiFR (Anthropic postmortem, Kimi K2 vendors at 61-72% similarity; 4-bit is detectable). I found no third.
- **Claims left unclaimed by this cluster:**
  - Detection on looped models, and compounding across loops as the reason it works.
  - Edge and NPU toolchain faults (QDQ, positional, INT8 REDUCE_SUM).
  - The 0.972 → 1.000 loop-1 → loop-2 result.
  - The wrong-reference failure (host quantized simulation 0.756, the 0% vs 58% case).
  - Predicting device accuracy from a host curve plus one latency.
  - Tying the detector to task accuracy.
  - Repair by running the last loops at higher precision.
- **Positioning (corrected Nov 24, Steven):** DiFR detects that a provider's model differs from the trusted one (from 2 tokens, AUC > 0.999), including a provider that quantized it; Ekka localizes known serving bugs. Neither runs on edge builds or looped models, and neither separates accurate quantized builds from broken ones. Describe our check on its own terms; a DiFR-style distance is our labelled baseline, not their result in our setting.

## Caveats
- DiFR's AUC is at 1% FPR over token batches, and ours is PASS/FAIL AUROC over builds, so the numbers are not comparable until we run their method on our data.
- Ekka's ICML 2026 listing is taken from the PDF front matter. Check the proceedings page before grading it anchor.
- Neither paper was compared against our code or data. All "falls short" statements rest on their text.