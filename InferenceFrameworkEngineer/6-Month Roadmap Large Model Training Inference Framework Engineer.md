# 6-Month Roadmap: Large Model Training/Inference Framework Engineer

Oct 29, 2026 · @Ray

## How to read this roadmap

The plan is 24 weeks at about 25 hours a week (roughly 600 hours), sequenced so every responsibility in the job description is covered by a shipped, measurable artifact by week 24. Each phase ends with a gate: a deliverable with a number attached (throughput, MFU, latency, loss parity), because that is what an interviewer for this role will probe.

**Sequencing logic.** The JD lists five responsibilities: distributed training, RL systems, multimodal training, inference KV-cache/load balancing, and supporting new models. RL systems sit on top of both a training engine and an inference engine (rollouts), so training and inference come first and RL comes late. Kernels come between them because both FP8 training and fast inference depend on them.

**Entry level assumed.** Comfortable Python, basic linear algebra, can train a small model in PyTorch. No CUDA or distributed experience. An engineer who already writes GPU kernels can compress Phase 1 and Phase 3 to about half and spend the time on the capstone.

**Hardware.** Weeks 1 to 4 run on one consumer or cloud GPU. From week 5, rent 2 to 8 GPU nodes by the hour for distributed experiments; the cost estimate is in the last section.

**The JD's own hints.** It names asynchronous RL, Agent RL, OPD (on-policy distillation), KV Cache disk caching, TileLang and Ascend C, and RDMA. These are treated as first-class topics, not extras, because they signal the stack the team actually runs.

&#91;embedded content: 24-week roadmap · 6 phases, 6 gates\]

RL systems are the JD's second responsibility but sit fifth here, because an RL loop needs both a training engine (Phase 2) and a rollout engine (Phase 4) to exist first.

## Skills matrix: JD line items to exit criteria

Every row below is a line from the JD. "Expert" means you can defend a design trade-off with your own measurements, not that you have read about it. Targets marked with numbers are the gates I set for this plan; published reference points are cited in each phase.

| JD item | Beginner (can use it) | Intermediate (can modify it) | Expert by week 24 (can build and defend it) | Phase |
| --- | --- | --- | --- | --- |
| PyTorch, Python, C++ | Train a model, read stack traces | Write custom autograd functions and C++ extensions | Explain dispatcher, caching allocator and torch.compile graph breaks from profiles | 1 |
| Model parallelism (DP/TP/PP/CP/EP) | Launch FSDP with torchrun | Hand-write TP and PP for a GPT block | Pick a 4D layout for a given model and cluster and predict MFU within 10% | 2 |
| Long context | Know why attention is quadratic | Run ring/context parallel in Megatron-Core | Train at 32K+ tokens and explain the memory and comms budget | 2, 3 |
| MoE | Know top-k routing | Run expert parallel with all-to-all | Diagnose load imbalance and comms overlap in a trace | 2 |
| Low-precision training and inference | Use BF16 autocast | FP8 training with Transformer Engine | FP8 loss within 0.25% relative of BF16 on your run; FP8/INT4 serving | 2, 3, 4 |
| Operators (CUDA, Triton, TileLang, Ascend C) | Run a Triton tutorial | Write fused kernels in Triton | Attention or GEMM kernel within 1.2x of the vendor library, plus a TileLang port | 3 |
| Inference: KV cache disk caching, load balancing | Serve with vLLM or SGLang | Configure prefix caching and CPU/disk KV offload | Build a cache-aware router and a tiered KV store and show TTFT gains on a trace | 4 |
| RDMA, CPU/IO optimisation | Know what NCCL and InfiniBand do | Benchmark nccl-tests and read bus bandwidth | Tune data loading and KV transfer bottlenecks from a CPU/IO profile | 2, 4 |
| RL systems: async RL, Agent RL, OPD | Run GRPO in verl | Modify rollout and reward workers | Ship an async RL loop with staleness control and an OPD recipe, with throughput numbers | 5 |
| Multimodal training systems | Fine-tune a VLM | Pack mixed image and text sequences | Accelerate VLM data reading and memory by a measured margin | 6 |
| Working with Code Agents | Use one for boilerplate | Use one for kernel and test scaffolding | Run an agent loop that proposes, tests and benchmarks changes | All |

## Phase 1 (weeks 1-4): GPU and PyTorch foundations

Goal: read a GPU profile and say whether a step is compute-, memory- or overhead-bound. Every later phase depends on this skill, so it gets four full weeks.

| Week | Focus | Do this |
| --- | --- | --- |
| 1 | GPU architecture and the roofline | SMs, warps, HBM vs SRAM, tensor cores. Compute arithmetic intensity for GEMM, softmax and LayerNorm by hand. |
| 2 | CUDA basics | Vector add, reduction, tiled SGEMM in CUDA C++. Profile with Nsight Compute; fix uncoalesced access and bank conflicts. |
| 3 | PyTorch internals | Write a custom autograd function and a C++/CUDA extension. Use torch.profiler and memory snapshots to find the allocator's fragmentation. |
| 4 | A GPT you own end to end | Train GPT-2 124M from your own code on one GPU. Add BF16, activation checkpointing, torch.compile; measure each change. |

**Core material**

- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), chapters on memory, tiling and reductions.
- The [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo as a reading target: Karpathy's llm.c GPT-2 baseline took 45 minutes on 8xH100 to reach 3.28 validation loss, and the community cut that to about 3 minutes by January 2025 ([summary](https://www.greaterwrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)). Each record's diff is a worked lesson in what actually moves throughput.

**Gate (end of week 4)**

- [ ] Tiled CUDA SGEMM at 4096 x 4096 reaching at least 70% of cuBLAS on the same GPU.
- [ ] Your GPT-2 124M trainer, with a written table of tokens/sec and MFU for each optimisation you added.
- [ ] One-page roofline analysis of your training step from a real profile.

## Phase 2 (weeks 5-9): Distributed training, long context, MoE, FP8

Goal: hand-build each parallelism once, then run the production version in Megatron-Core and explain its numbers. This covers JD responsibility 1 (model parallelism, long context, MoE, low-precision training).

| Week | Focus | Do this |
| --- | --- | --- |
| 5 | Collectives and data parallel | Run nccl-tests; learn all-reduce, all-gather, reduce-scatter, all-to-all costs. Write DDP by hand, then use FSDP2 and ZeRO stages 1-3. |
| 6 | Tensor and pipeline parallel | Hand-write Megatron-style column/row parallel MLP and attention. Implement 1F1B pipeline schedule; measure the bubble. |
| 7 | Long context | Sequence parallel, context parallel (ring attention), Ulysses. Train at 32K tokens; budget activation memory per GPU. |
| 8 | MoE and expert parallel | Top-k router, aux-loss vs aux-loss-free balancing, all-to-all dispatch. Profile comms/compute overlap in Megatron-Core. |
| 9 | FP8 training | Transformer Engine FP8 with delayed scaling, then block-wise scaling. Run FP8 vs BF16 on the same 1B model and compare loss curves. |

**Reference points to calibrate against**

- Megatron-LM's PTD-P (tensor + pipeline + data parallel) trained a 1T-parameter model at 502 petaFLOP/s on 3,072 GPUs, 52% of theoretical peak per GPU; its interleaved schedule added 10%+ throughput ([Narayanan et al., SC21](https://arxiv.org/abs/2104.04473v5)). Read sections 3-5 before week 6.
- DeepSeek-V3's report is the best public account of FP8 + MoE + pipeline engineering at scale: its DualPipe schedule overlaps compute and communication, and it avoided tensor parallelism entirely ([report](https://arxiv.org/pdf/2412.19437)). Its FP8 run stayed below 0.25% relative loss error versus BF16, with FP32 accumulation promoted every 128 elements ([primer](https://aman.ai/primers/ai/deepseekV3)). Use that 0.25% figure as your own FP8 gate.
- Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) is built from 4,000+ scaling experiments on up to 512 GPUs and walks DP, TP, PP, CP and ZeRO with code. Treat it as the course text for this phase.

**Gate (end of week 9)**

- [ ] A from-scratch 2D (TP x DP) GPT that matches single-GPU loss to 1e-3 on a fixed seed.
- [ ] A 1-3B dense model in Megatron-Core on one 8-GPU node at 40%+ MFU, with a written explanation of where the other 60% goes.
- [ ] FP8 vs BF16 loss gap under 0.25% relative over a few thousand steps.
- [ ] A small MoE run with expert parallel and a trace showing all-to-all overlap.

## Phase 3 (weeks 10-13): Kernels in Triton, TileLang and CUDA

Goal: write the fused kernels that FP8 training and fast inference depend on. This covers JD bonus point 3 (CUDA / Triton / TileLang / Ascend C operator development).

| Week | Focus | Do this |
| --- | --- | --- |
| 10 | Triton | Fused softmax, LayerNorm/RMSNorm forward and backward, matmul with autotune. Benchmark against PyTorch eager and torch.compile. |
| 11 | FlashAttention | Read the paper, then write FlashAttention forward and backward in Triton. Add causal mask and variable-length (packed) sequences. |
| 12 | TileLang | Port your GEMM and attention to TileLang; add an FP8 block-scaled GEMM and an MLA decode kernel from its examples. |
| 13 | CUDA/CUTLASS depth | Warp-level MMA, async copies, shared-memory swizzling. Read one Hopper-specific kernel (FlashAttention-3 or DeepGEMM) line by line. |

**Why these targets**

- FlashAttention is the canonical IO-aware kernel: tiling between HBM and on-chip SRAM gave 3x speedup on GPT-2 at 1K sequence length and 2.4x on long-range arena ([NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)). Writing it yourself is the fastest way to internalise the memory hierarchy.
- [TileLang](https://iclr.cc/virtual/2026/oral/10010187) (ICLR 2026 oral) exposes explicit tile-level control of memory placement and scheduling, and expresses fused attention kernels in under 80 lines of Python. Its [PyPI page](https://pypi.org/project/tilelang/) reports FlashMLA on AMD MI300X at parity with hand-written assembly kernels, and the project ships [TileLang Puzzles](https://pypi.org/project/tilelang/) as a graded on-ramp.
- Ascend C needs Huawei Ascend hardware. If you cannot get access, note it as a gap; the tiling concepts transfer directly from TileLang and CUDA.

**Gate (end of week 13)**

- [ ] Triton FlashAttention forward and backward within 1.2x of the official FlashAttention-2 at 4K sequence length, numerically checked against PyTorch.
- [ ] TileLang FP8 block-scaled GEMM, benchmarked against cuBLAS FP8.
- [ ] Fused RMSNorm + residual kernel dropped into your Phase 2 model, with end-to-end step-time change measured.

## Phase 4 (weeks 14-17): Inference systems, KV cache tiering and load balancing

Goal: build a small serving engine, then extend a production one with disk KV caching and a cache-aware router. This is JD responsibility 4 almost word for word.

| Week | Focus | Do this |
| --- | --- | --- |
| 14 | Engine internals | Write a mini engine: continuous batching, paged KV cache, chunked prefill. Then trace a request through vLLM's scheduler and block manager. |
| 15 | Prefix reuse | Compare vLLM prefix caching with SGLang RadixAttention on a multi-turn chat trace and a RAG trace; record TTFT and throughput. |
| 16 | KV cache to CPU and disk | Run LMCache with vLLM for CPU then local-SSD offload. Build your own tiered store (GPU, DRAM, NVMe) with LRU and async prefetch. |
| 17 | PD disaggregation and routing | Split prefill and decode across two instances, transfer KV via Mooncake Transfer Engine or NIXL. Write a router that scores instances on cached-prefix length and queue depth. |

**Reference systems and their published numbers**

- vLLM's PagedAttention raised throughput 2-4x over FasterTransformer and Orca at equal latency by paging the KV cache like OS virtual memory ([Kwon et al.](https://arxiv.org/abs/2309.06180)).
- SGLang's RadixAttention reuses KV cache across calls via a radix tree; the paper reports up to 6.4x higher throughput ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract.html)).
- [Mooncake](https://arxiv.org/abs/2407.00079v3), Kimi's serving platform, separates prefill and decode clusters and pools CPU, DRAM and SSD into a disaggregated KV cache. Under real workloads it let Kimi handle 75% more requests ([GitHub](https://github.com/kvcache-ai/Mooncake)). It is the closest public design to "KV Cache disk caching and load balancing" in the JD.
- Ray's LMCache guide measured about 10-15% overhead when offloading gets no cache hits; the gain comes only when prefixes repeat ([Ray docs](https://docs.ray.io/en/latest/serve/llm/user-guides/kv-cache-offloading.html)). Your benchmark must include a no-reuse trace to show you understand this trade-off.

**Gate (end of week 17)**

- [ ] Mini engine with paged KV and continuous batching serving a 1-8B model, throughput within 2x of vLLM on the same GPU.
- [ ] Tiered KV cache with a measured TTFT reduction on a multi-turn trace and the overhead on a no-reuse trace, both reported.
- [ ] Cache-aware router vs round-robin on 2+ instances: P50/P99 TTFT and throughput.

## Phase 5 (weeks 18-21): RL training systems: async RL, Agent RL, OPD

Goal: own the full RL loop as a systems problem: a training engine, a rollout engine, weight sync between them, and the scheduling that keeps both busy. This is JD responsibility 2, and it reuses everything from Phases 2 and 4.

| Week | Focus | Do this |
| --- | --- | --- |
| 18 | Synchronous RL baseline | Run GRPO on a 1.5-7B model in verl with FSDP2 + vLLM or SGLang rollouts. Profile the timeline: generation vs training vs resharding. |
| 19 | Async RL | Read AReaL and slime's fully-async example. Decouple rollout and training, add interruptible rollouts and a staleness bound; compare against the sync baseline. |
| 20 | Agent RL | Multi-turn tool-calling rollouts with a sandbox (code execution) and verifier rewards. Handle long-tail trajectories and tokenisation across turns. |
| 21 | On-policy distillation | Implement OPD: sample from the student, score every token with a teacher's log-probs, minimise per-token reverse KL. Compare cost and accuracy against GRPO on the same task. |

**Frameworks to study, and why each**

- [verl](https://verl.readthedocs.io/en/latest/index.html) is the open-source version of the HybridFlow paper (EuroSys 2025). It supports FSDP, FSDP2 and Megatron-LM for training and vLLM, SGLang or HF Transformers for rollout ([repo](https://github.com/dot-agi/verl)). Its 3D-HybridEngine resharding between training and generation is the part to read closely.
- [slime](https://github.com/THUDM/slime) pairs Megatron training with SGLang rollout through a data buffer, and is the RL framework behind the GLM-4.5 through GLM-5.2 models. It is the cleanest codebase for Megatron + SGLang RL at scale.
- [AReaL](https://neurips.cc/virtual/2025/loc/san-diego/poster/117538) (NeurIPS 2025) fully decouples generation from training, controls data staleness, and uses a staleness-aware PPO variant; it reports up to 2.77x speedup over synchronous systems on the same GPUs with matched or better accuracy.
- On-policy distillation was laid out by Thinking Machines in October 2025 ([blog](https://thinkingmachines.ai/blog/on-policy-distillation/)). They report reaching 70% on AIME'24 with 1,800 GPU hours versus 17,920 for RL, using Qwen3-32B as teacher for Qwen3-8B ([Maginative](https://www.maginative.com/article/thinking-machines-claims-30x-cost-cut-for-training-ai-models/)). Runnable recipes, including multi-teacher and multi-turn variants, are in the [Tinker cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/).

**Gate (end of week 21)**

- [ ] Sync vs async GRPO on the same GPUs: wall-clock to a fixed reward, GPU utilisation, and the staleness setting used.
- [ ] A multi-turn agent RL run with a sandboxed tool and a verifier reward.
- [ ] An OPD run on a small student/teacher pair with GPU-hours and accuracy reported next to the GRPO run.

## Phase 6 (weeks 22-24): Multimodal training systems and the capstone

Goal: apply the stack to a vision-language model, where the bottlenecks move to data reading and uneven sequence lengths, then ship one capstone that ties the six phases together. This covers JD responsibility 3 (model structure, memory optimisation, data-reading acceleration) and responsibility 5 (rapidly landing new models).

| Week | Focus | Do this |
| --- | --- | --- |
| 22 | VLM model structure and memory | Build a ViT encoder + projector + LLM in Megatron-Core or FSDP2. Decide where the encoder sits in the pipeline; recompute vs offload for the vision tower. |
| 23 | Data-reading acceleration | Sharded streaming datasets, image decode on CPU workers vs GPU (nvJPEG/DALI), sequence packing of mixed image and text. Measure GPU idle time from data stalls before and after. |
| 24 | Capstone and write-up | Finish the capstone below; publish code, benchmarks and a technical blog post. |

**Capstone (pick one, weeks 22-24 in parallel)**

1. **VLM RL end to end.** GRPO or OPD on a small VLM in verl or slime, with packed multimodal rollouts and image-token KV reuse in SGLang. Report throughput per GPU and the bottleneck breakdown.
2. **New-model bring-up.** Pick a just-released open model and land it in Megatron-Core (training) and SGLang or vLLM (inference) with numerical parity tests. This mirrors JD responsibility 5 directly.
3. **Upstream contribution.** A merged PR to verl, slime, SGLang, vLLM, Megatron-LM or TileLang that improves a measured number. This also earns JD bonus point 1.

**Gate (end of week 24)**

- [ ] VLM training step with data-stall time cut by a measured margin, with before/after profiles.
- [ ] Capstone repo with a README that states the problem, the baseline, the change and the numbers.
- [ ] One public technical post explaining a trade-off you measured yourself.

## Bonus points plan and portfolio

Five of the JD's six bonus points can be earned inside this plan; only school grades cannot. The table shows where each one is produced.

| JD bonus point | How this plan earns it | Where |
| --- | --- | --- |
| 1. Deep open-source contributions | Start with docs and test fixes in week 10; aim for one merged performance PR by week 24 | Phases 3-6, capstone 3 |
| 2. HPC or algorithm competition results | Student cluster competitions (ASC, ISC, SC) are for students. A working engineer can substitute a public, verifiable benchmark result, such as a submitted modded-nanogpt record attempt | Phases 1-2 |
| 3. CUDA / Triton / TileLang / Ascend C | FlashAttention in Triton, FP8 GEMM in TileLang, CUDA SGEMM | Phases 1, 3 |
| 4. RDMA, CPU/IO optimisation | nccl-tests on multi-node, KV transfer over RDMA via Mooncake/NIXL, data-loader profiling | Phases 2, 4, 6 |
| 5. Top 10% grades | Not addressable by a roadmap | n/a |
| 6. Reproduced LLM, RL, multimodal papers | FlashAttention, PagedAttention, AReaL-style async RL, OPD, a VLM recipe | Phases 3-6 |

**Portfolio checklist for the interview**

- [ ] One repo per phase, each with a benchmark table and a profile screenshot.
- [ ] A single page listing every gate number you hit, with links to the runs.
- [ ] Two technical write-ups: one training-side (parallelism or FP8), one serving or RL-side.
- [ ] At least one merged upstream PR.
- [ ] A short demo of working with a code agent on a kernel or test harness, since the JD asks for it explicitly.

## Weekly cadence, compute budget and risks

**Weekly cadence (25 hours).** About 6 hours reading papers and code, 14 hours building, 3 hours benchmarking and writing up numbers, 2 hours on a weekly review against the phase gate. If a gate slips more than one week, cut scope inside the phase rather than pushing later phases, because Phase 5 depends on both Phase 2 and Phase 4.

**Compute budget.** Expect roughly USD 3,900-5,200 of rented GPU time for the whole plan. The estimate assumes USD 3-4 per H100-hour at specialist clouds, so USD 24-32 per 8-GPU node-hour; hours are my estimates of what each gate needs.

| Phase | Hardware | Estimated hours | Estimated cost (USD) |
| --- | --- | --- | --- |
| 1 Foundations | 1 GPU (own or rented) | 60 GPU-h | 180-240, or 0 on own GPU |
| 2 Distributed | 8x H100 node; 2 nodes for multi-node tests | 40 node-h + 6 h on 2 nodes | 1,250-1,660 |
| 3 Kernels | 1x H100 (Hopper needed for FP8 and FA3) | 60 GPU-h | 180-240 |
| 4 Inference | 2-8 GPUs | 15 node-h | 360-480 |
| 5 RL systems | 8x H100 node | 50 node-h | 1,200-1,600 |
| 6 Multimodal and capstone | 8x H100 node | 30 node-h | 720-960 |

Price basis: on-demand H100 rents for about USD 2-12 per GPU-hour in 2026 depending on provider ([Akash, Aug 2026](https://akash.network/the-bid/h100-rental-price-2026-cost-per-hour/)); Lambda's 8-GPU H100 SXM lists at USD 3.99 per GPU-hour ([IntuitionLabs](https://intuitionlabs.ai/pdfs/h100-rental-prices-cloud-comparison-4bf9097a9abaa5b5.pdf)), and Karpathy's nanochat cites an 8xH100 box at USD 24/hour ([nanochat](https://github.com/diffley/nanochat)). Hyperscalers cost far more: an 8xH100 node is USD 98.32/hour on Azure.

**Risks and mitigations**

- **GPU capacity and price.** One-year H100 contract prices rose almost 40%, from USD 1.70 to 2.35 per GPU-hour, between October 2025 and March 2026 ([IntuitionLabs, citing SemiAnalysis](https://intuitionlabs.ai/pdfs/cloud-gpu-rental-prices-comparison.pdf)). Book node time for Phases 2 and 5 in advance; do all debugging on one GPU first.
- **Hopper-only code.** Speedrun-grade code such as modded-nanogpt assumes Hopper kernels and does not run as-is on consumer GPUs ([write-up](https://deepseek.csdn.net/6a32003a662f9a54cb80311f.html)). Use consumer GPUs for Phases 1 and early 3 only.
- **Framework churn.** verl moved to the verl-project organisation in January 2026 and moved its recipes to a separate repository ([repo](https://github.com/dot-agi/verl)). Pin versions per phase and record them in each README.
- **Breadth over depth.** The JD rewards depth. If time runs short, drop capstone options 1 and 2 and keep the upstream PR.

## Sources

- Job description: "Large Model Training/Inference Framework Engineer" (provided).
- [Megatron-LM PTD-P, SC21](https://arxiv.org/abs/2104.04473v5)
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) and [primer](https://aman.ai/primers/ai/deepseekV3)
- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and [speedrun summary](https://www.greaterwrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)
- [FlashAttention, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- [TileLang, ICLR 2026](https://iclr.cc/virtual/2026/oral/10010187) and [PyPI](https://pypi.org/project/tilelang/)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract.html)
- [Mooncake paper](https://arxiv.org/abs/2407.00079v3) and [repo](https://github.com/kvcache-ai/Mooncake)
- [Ray Serve LMCache KV offloading guide](https://docs.ray.io/en/latest/serve/llm/user-guides/kv-cache-offloading.html)
- [verl docs](https://verl.readthedocs.io/en/latest/index.html)
- [slime](https://github.com/THUDM/slime)
- [AReaL, NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/117538)
- [Thinking Machines, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), [Maginative summary](https://www.maginative.com/article/thinking-machines-claims-30x-cost-cut-for-training-ai-models/), [Tinker cookbook recipes](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- H100 pricing: [Akash](https://akash.network/the-bid/h100-rental-price-2026-cost-per-hour/), [IntuitionLabs H100](https://intuitionlabs.ai/pdfs/h100-rental-prices-cloud-comparison-4bf9097a9abaa5b5.pdf), [IntuitionLabs cloud GPU](https://intuitionlabs.ai/pdfs/cloud-gpu-rental-prices-comparison.pdf)
