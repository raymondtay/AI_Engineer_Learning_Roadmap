# Week 24: capstone and write-up

Roadmap gate (end of week 24):

- [ ] VLM training step with data-stall time cut by a measured margin, with before/after profiles (Weeks 22-23).
- [ ] Capstone repo with a README that states the problem, the baseline, the change and the numbers.
- [ ] One public technical post explaining a trade-off you measured yourself.

Pick **one** capstone and run it in parallel with Weeks 22-23.

## Option 1: VLM RL end to end

GRPO or on-policy distillation on a small VLM, with packed multimodal rollouts.

- CPU prototype: combine `w22_tiny_vlm` (model), `w23_mm_dataloading` (packing) and `w18_grpo_sync` / `w21_opd` (objective). Reward: exact-match caption.
- GPU: verl or slime with a Qwen2.5-VL-class model; SGLang reuses image-token KV across a group's samples because they share the image.
- Numbers to report: samples/s per GPU, generation vs training vs weight-sync seconds per step, reward curve.

## Option 2: new-model bring-up

Land a just-released open model in a training framework and an inference engine with numerical parity tests.

- Parity ladder: (1) logits match the reference implementation on 10 fixed prompts to < 1e-3 in fp32; (2) a 50-step loss curve matches within noise; (3) greedy decode matches token for token for 64 tokens.
- CPU prototype of the method: `w14_mini_engine` already tests "engine output == reference greedy", reuse that harness.

## Option 3: upstream contribution

A merged PR to verl, slime, SGLang, vLLM, Megatron-LM or TileLang that improves a measured number.

- Start in week 10 with docs/tests PRs to learn each project's review process.
- Every PR description should contain: the benchmark command, before/after numbers, hardware.

## The one-page gate table

`python tools/gates.py` runs a quick version of every lab's measurement against your code and writes `GATES.md`. Run it at the end of each phase, commit the file, and link the runs. `LAB=solution python tools/gates.py` gives reference numbers for the same machine.

## Write-up template

```markdown
# <What I changed and the number it moved>

**Problem.** One paragraph: what was slow or wrong, with the measurement that showed it.

**Baseline.** Hardware, versions, command, and the number (with variance over N runs).

**Change.** What you did and why it should help (the mechanism, not just the diff).

**Result.** Before/after table. Profile screenshots (nsys / torch.profiler) that show where the time went.

**What did not work.** Ideas you tried that did not help, with numbers.

**Limits.** Where the result stops holding (scale, hardware, workload).
```

The labs record several trade-offs that are good write-up material because the numbers are not what people expect: BF16 autocast making CPU training 2.4x slower (Week 4), a prefix cache adding latency on a no-reuse trace (Weeks 15-16), cache-aware routing cutting P50 but leaving P99 flat once decode dominates service time (Week 17, calibrated costs), a list-of-tensors collate costing 5x in data-loader hand-off (Week 23).
