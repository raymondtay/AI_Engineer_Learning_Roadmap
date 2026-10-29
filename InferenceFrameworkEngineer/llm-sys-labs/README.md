# llm-sys-labs: Large Model Training/Inference Framework Engineer, in 24 labs

Semi-worked codebases that follow the [6-month roadmap](https://claude.ai/artifact/3TRTku594Hb97cSMdDfdCH) week by week, using the toolchain it prescribes (PyTorch, CUDA C++, Triton, TileLang, torch.distributed/FSDP2, Megatron-Core, Transformer Engine, vLLM/SGLang concepts, verl). **Every lab runs on CPU**; the same code runs on CUDA GPUs, where the GPU-only gates (speed vs cuBLAS / FlashAttention-2 / vLLM, Megatron MFU, TE FP8) also apply.

## How it works

Each lab directory holds:

| File | Role |
| --- | --- |
| `solution.py` (and `solution_*.cpp/.cu`) | reference implementation; the parts you write are fenced `# >>> SOLUTION: hint` ... `# <<< SOLUTION` |
| `exercise.py` | generated from the solution: fences replaced by the hint and `raise NotImplementedError`; everything else (scaffolding, harness, benchmarks) kept |
| `test_lab.py` | correctness and behaviour tests, run against your exercise by default |

97 TODO blocks across 24 weeks. Workflow:

```bash
pip install -r requirements-cpu.txt          # Linux or the Dockerfile; see "Platforms"
LAB=exercise pytest labs/phase2_distributed/w06_tp_pp   # red until you fill the TODOs
LAB=solution pytest labs                     # the reference: all green on CPU
python tools/gates.py                        # one-page table of every gate number -> GATES.md
python tools/make_exercises.py               # regenerate exercises (overwrites them: commit first!)
```

Peek at `solution.py` only after your tests pass or you are truly stuck.

## Map to the roadmap

| Phase | Weeks | Labs | JD responsibility |
| --- | --- | --- | --- |
| [1 Foundations](labs/phase1_foundations/README.md) | 1-4 | roofline, CUDA/C++ kernels, PyTorch internals, GPT trainer | prerequisite for all |
| [2 Distributed](labs/phase2_distributed/README.md) | 5-9 | DDP/ZeRO/FSDP2, TP/PP, ring + Ulysses, MoE/EP, FP8 | 1: parallelism, long context, MoE, low precision |
| [3 Kernels](labs/phase3_kernels/README.md) | 10-13 | Triton basics, FlashAttention fwd/bwd, TileLang, MMA/banks/pipelining | bonus 3: operator development |
| [4 Inference](labs/phase4_inference/README.md) | 14-17 | paged engine, prefix cache, tiered KV, PD + router | 4: KV cache disk caching, load balancing |
| [5 RL](labs/phase5_rl/README.md) | 18-21 | sync GRPO, async RL, agent RL, OPD | 2: async RL, Agent RL, OPD |
| [6 Multimodal](labs/phase6_multimodal/README.md) | 22-24 | tiny VLM, data loading + packing, capstone | 3 and 5: multimodal, new-model bring-up |

## What was verified, and how

Built and tested on a 2-core x86 CPU VM, Python 3.13, PyTorch 2.15.0, Triton 3.9.0, TileLang 0.1.15, no GPU.

| Claim | Evidence |
| --- | --- |
| All reference solutions pass on CPU | `LAB=solution pytest labs`: see `TEST_RESULTS.txt` |
| Exercises are well-formed | all parse; in exercise mode tests fail at the TODOs, not at import |
| Measured numbers in the phase READMEs | `GATES_solution.md`, produced by `LAB=solution python tools/gates.py` on that VM |
| CUDA sources compile | `solution_kernels.cu` and `mma_check.cu` compiled with nvcc 13.2 for sm_90 against PyTorch 2.15 headers |
| **Not executed anywhere** | every GPU-only path: CUDA kernels at runtime, `@pytest.mark.gpu` tests, `gpu_te_fp8.py`, `megatron_gate.sh`, `verl_grpo.sh`, TileLang `*_gpu` programs. Config keys in the shell scripts are checked against the projects' docs where those docs showed them; the rest are marked `VERIFY` |

Several results came out differently from what the literature or my first design predicted. They are reported as measured, with the explanation found, rather than tuned away: BF16 slower than fp32 on CPU (Week 4); FP8 trailing fp32 by ~6% at toy scale (Week 9); prefix caching costing latency on no-reuse traffic (Weeks 15-16); async RL's speedup within noise on 2 cores (Week 19); on-policy distillation roughly matching, not beating, GRPO on this task (Week 21); data-loader hand-off costs (Week 23).

## Platforms

- **Linux x86-64**: everything CPU-side works with `requirements-cpu.txt`.
- **macOS**: PyTorch runs on CPU, but Triton has no macOS wheels and TileLang's C backend was only verified on Linux. Use the `Dockerfile` (CPU-only container) for Phases 3 and the gates. MPS is deliberately not used: the labs target CUDA semantics.
- **CUDA GPU (sm_80+, Hopper for FP8)**: install `requirements-cpu.txt`, then each GPU framework per its own docs (Transformer Engine, flash-attn, Megatron-LM, vLLM, SGLang, verl). Set nothing else: `common/device.py` picks CUDA and NCCL automatically.

## Layout

```
common/        device + backend selection, launcher (gloo/NCCL), FLOP/MFU accounting,
               shared tiny GPT (byte-level, RoPE, past-KV forward), RL task
labs/phaseN_*/wNN_*/   solution, generated exercise, tests (+ GPU scripts where relevant)
tools/         make_exercises.py, gates.py
```

## Sources for the reference numbers used in the labs

H100 datasheet peaks in `common/flops.py`; Megatron-LM PTD-P (SC21) for parallelism; Korthikanti et al. 2022 for activation memory; DeepSeek-V3 report for FP8 block scaling and aux-loss-free balancing; FlashAttention (NeurIPS 2022); vLLM PagedAttention (SOSP 2023); SGLang RadixAttention (NeurIPS 2024); Mooncake; AReaL (NeurIPS 2025); Thinking Machines "On-Policy Distillation" (Oct 2025); PTX ISA `mma.m16n8k16` fragment tables. Full links are in the roadmap document.
