Looped and iterative models: sourced sweep, 2026-11-23. No edits, no ssh.

**Read status.** I fetched every URL below, but only abstract/listing pages or press pages. I did not read any PDF in full. Under LITERATURE.md, every paper is therefore "abstract" read and should stay cite=no in the ledger until read in full. Grades are provisional. Affiliations are not shown on arXiv abs pages, so credibility markers are partial. No slop flags found, except where noted.

**Inkling.** It does not iterate.
- Source: Thinking Machines, https://thinkingmachines.ai/news/introducing-inkling/, July 15, 2026, so released four months ago, not days. I read this page directly.
- The page describes a 975B/41B-active MoE (mixture of experts) with 66 distinct decoder layers. It has sliding-window and global attention at 5:1, short convolutions and a relative positional embedding.
- The only "iteration" in the post is a demo of 40 review rounds on a snake game, and RL rollouts. Neither is architectural.
- The HF blog (https://huggingface.co/blog/thinkingmachines-inkling) describes Inkling 2 Small (594B/17B, 48 layers) as a single forward pass with a learned weighted mix over block summaries. It is not looped.
- A search result listed "Inkling 2 Mini" as a looped MoE (middle 12 layers run 3 times). I found no primary source for it. I found only a third-party broken test fixture (tbmod/Inkling-mini-bf16). Treat it as unverified and do not cite it.

**Rows.** Format: name | org | date | venue as verified | how it iterates | whole/component | scale | URL | read | grade.

Looped and recurrent:
1. Huginn | Maryland, ELLIS/Tübingen, LLNL (Goldstein, Geiping et al.; affiliations from memory, not on the page) | 2025-02 | NeurIPS 2025 Spotlight, main conference (neurips.cc poster page) | recurrent block iterated in latent space, depth set at test time | whole | 3.5B | https://arxiv.org/abs/2502.05171 ; https://neurips.cc/virtual/2025/poster/117966 | abstract | anchor (venue verified; read still needed).
2. Ouro / LoopLM | ByteDance Seed and collaborators, with Bengio (affiliations from memory, not on the page) | 2025-10 (v6 2026-10-11) | arXiv lists no venue, so preprint | pretrained looped LM with entropy-regularized learned depth, 7.7T tokens | whole | 1.4B, 2.6B | https://arxiv.org/abs/2510.25741 | abstract | support. Claim that it matches models up to 12B is the authors' own and unchecked.
3. Mixture-of-Recursions | KAIST, Google DeepMind and Mila (Bae et al.; affiliations from memory) | 2025-07 | NeurIPS 2025 main track poster (papers.neurips.cc) | shared stack reused, router assigns per-token recursion depth | whole | 135M to 1.7B | https://arxiv.org/abs/2507.10524 ; https://neurips.cc/virtual/2025/poster/118085 | abstract | anchor.
4. Relaxed Recursive Transformers | Google (Bae, Fisch, Schuster et al.) | 2024-10 | ICLR 2025 per the arXiv Comments field. Proceedings page not checked. | pretrained model converted to one looped block plus layer-wise LoRA | whole (conversion) | Gemma 1B recursive vs 2B | https://arxiv.org/abs/2410.20672 | abstract | anchor only if the proceedings page confirms it; support until then.
5. Parcae | UC San Diego and Together AI (Prairie, Fu et al.) | 2026-04 | arXiv 2604.12946. The ICLR 2026 page is the workshop track "Latent & Implicit Thinking", not main. | stable looped LM via spectral-norm-constrained injection | whole | 140M to 770M (770M roughly matches a 1.3B Transformer, per abstract) | https://arxiv.org/abs/2604.12946 ; https://iclr.cc/virtual/2026/10016640 | abstract and search snippet | support.
6. LoopFormer | Toronto/Vector (Jeddi, Ciccone, Taati) | 2026-02 | ICLR 2026 per arXiv Comments; track not verified | looped transformer conditioned on step size, elastic depth | whole | not stated | https://arxiv.org/abs/2602.11451 | abstract | support until the track is verified.
7. Hyperloop Transformers | Yoon Kim's group (Zeitoun, Torroba-Hennigen, Kim; MIT, per memory) | 2026-04 | preprint, no venue listed | only the middle block is looped, with hyper-connections | whole | about 50% fewer parameters than depth-matched baselines; sizes not given | https://arxiv.org/abs/2604.21254 | abstract | support.
8. Attractor Models | Fein-Ashley and Rashidinejad (affiliations not listed) | 2026-05 | preprint | backbone proposes, attractor module refines to a fixed point, implicit gradients | whole | 770M LM, 27M reasoner | https://arxiv.org/abs/2605.12466 | abstract | caution. Two authors, no listed affiliation, no venue or code seen. Check the claimed 46.6% perplexity gain.

Fixed-point and tiny recursive reasoners:
9. Equilibrium Reasoners (EqR) | Huang, Geng, Kolter (CMU, per memory) | 2026 | ICML 2026 poster, verified on icml.cc page; no arXiv link on the page | fixed-point iteration, up to 40,000 unrolled layers | whole | not stated | https://icml.cc/virtual/2026/poster/61915 | abstract | anchor (Kolter's DEQ record is real; venue verified).
10. TRM | Jolicoeur-Martineau (Samsung SAIL Montreal, per memory) | 2025-10 | preprint, no venue on arXiv | a single 2-layer net applied recursively | whole | about 7M | https://arxiv.org/abs/2510.04871 | abstract | support. Single author, but an established record and a widely reproduced result.
11. HRM | Sapient Intelligence (Wang et al., per memory) | 2025-06 (v4 2026-09-30) | preprint, no venue on arXiv | two coupled recurrent modules at different timescales | whole | 27M | https://arxiv.org/abs/2506.21734 | abstract | support.

Diffusion LMs (iterative denoising):
12. LLaDA | Renmin University and Ant Group (Nie et al., per memory) | 2025-02 | preprint; no venue on arXiv | masked-diffusion denoising | whole | 8B | https://arxiv.org/abs/2502.09992 | abstract | support.
13. Dream 7B | HKU and Huawei Noah's Ark (per memory) | 2025-08 | preprint | iterative denoising of whole sequences | whole | 7B | https://arxiv.org/abs/2508.15487 | abstract | support.
14. Mercury | Inception Labs (company) | 2025-06 | tech report, arXiv | parallel multi-token diffusion | whole | not stated | https://arxiv.org/abs/2506.17298 | abstract | support. **Industry: commercial API and playground.**
15. Gemini Diffusion | Google DeepMind (company) | page undated | blog/demo page | refines blocks of noise to text | whole | not disclosed | https://deepmind.google/models/gemini-diffusion/ | page read | caution as a citation: no paper and no numbers beyond speed. **Industry: experimental demo only.**

Production and open-weight looped models, 2026:
16. GLM-5.4 | Z.ai | 2026-11-23 (today) | company announcement only, no model card | the same 28 layers run twice, with gates and a second-pass KV reuse | whole | about 750B per press, which Z.ai had not confirmed | https://z.ai/blog/glm-5.4 (fetch returned empty) ; https://thenextweb.com/news/zai-glm-5-4-loop-transformer-coding-model ; https://www.datacamp.com/blog/glm-5-4 | press only | caution. **Industry: live to Coding Plan subscribers. Weights, API and a model card are not out yet.**
17. Nanbeige4.2-3B | Nanbeige Lab (Boss Zhipin) | 2026-07-24 | tech report on arXiv | the 22-layer stack run twice (44 effective layers), 28T tokens | whole | 3B non-embedding, about 4B total | https://arxiv.org/abs/2607.22083 ; https://huggingface.co/Nanbeige/Nanbeige4.2-3B | abstract | support. **Industry: Apache-2.0 open weights.**
18. SenseNova-Loop-9B | SenseNova (company) | 2026-11-17 | model card, no tech report | layers 4 to 27 run twice, with two-stream mHC (manifold-constrained hyper-connections) | whole | 9B stored, 14.6B inference-equivalent | https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF | card read | caution until a report appears. **Industry: Apache-2.0 open weights.**

Not in the list: I did not verify LLaDA 2.x or Mercury 2, and I did not search for production deployments of TRM or HRM.

**(a) Strongest citations for "looped or iterative computation is increasingly used in model design"**
1. Huginn (NeurIPS 2025 Spotlight) and Mixture-of-Recursions (NeurIPS 2025 main). These are refereed at A* level and establish the pretrained looped LM and adaptive-depth lines.
2. Equilibrium Reasoners (ICML 2026). It is refereed and shows the fixed-point line is still active in 2026.
3. Nanbeige4.2-3B plus SenseNova-Loop-9B. Two independent open-weight releases in July and November 2026 that reuse layers. They show adoption beyond papers.
4. GLM-5.4 (Z.ai, 2026-11-23). A frontier-scale company flagship, but announcement-only, so cite it as a company statement, not as evidence.
5. Ouro, as a scale anchor once a venue is confirmed. Today it is a preprint.

**(b) Adoption evidence**
- Open-weight: Nanbeige4.2-3B and SenseNova-Loop-9B.
- Commercial: Mercury (diffusion API), GLM-5.4 (looped, subscribers only).
- Experimental: Gemini Diffusion.
- Inkling is not an adopter.
- All three looped releases are 2026. Press descriptions of GLM-5.4 came from secondary coverage; the primary page returned nothing, so verify against Z.ai's own post.

**(c) Contradicting or limiting evidence**
- "There Is No Free Loop" (Sukthanker and Cameron, 2026-10-30, preprint; https://arxiv.org/abs/2610.42030). Abstract: only a looped 0.8B model beat both baselines; at 4B and 9B, looping a trained Qwen3/3.5 model did not beat the same model trained without the loop.
- Inkling, the headline 2026 open-weight model, is a plain feed-forward stack. Most frontier open models are not looped.
- Nanbeige notes slower decoding from looping, and Halloran (https://arxiv.org/abs/2608.13987) documents deployment bugs and memory overhead for looped models on Apple Silicon.
- A halting-heads study (https://arxiv.org/abs/2610.32246) finds that heads predict fewer iterations than they actually run. Only the candidates table's abstract summary was read.
- Use "increasingly common" for the 2026 releases, not "dominant".
## Correction: Inkling 2 Mini is looped (checked against the primary source)

The sweep's "Inkling does not iterate" holds only for Inkling and Inkling 2 Small. Inkling 2 Mini, released by
Thinking Machines on Nov 20, 2026 (https://thinkingmachines.ai/news/inkling-2-mini, read in full), is a looped
transformer: "running the middle layers of the model three times for every token". It has 24 layers in four blocks of
six; the two middle blocks run three times, a fixed count ("only supports a fixed number of looping passes"). It has
26.4B total parameters with 3.4B active in the text and 5.4B in the table (the page disagrees with itself); about 20 GB
in total; "small enough to run on your laptop"; open weights (https://huggingface.co/thinkingmachines/Inkling-2-Mini,
model card https://thinkingmachines.ai/model-card/inkling-2-mini/). It also links a variable-loop research model
(https://thinkingmachines.ai/model-card/tml-d24-varloop-research/). The page says a non-looped model needed "140% more
inference compute" to match the looped loss. Goodfire's interpretability study of it
(https://www.goodfire.com/research/interpreting-inkling-2-mini/, Nov 20, 2026) confirms layers 7-18 loop T = 3 times.
Grade: support (company release with open weights and a model card; no tech report).
