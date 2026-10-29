# Phase 6 (weeks 22-24): multimodal training systems and the capstone

| Week | Lab | You implement | Measured on CPU |
| --- | --- | --- | --- |
| 22 | `w22_tiny_vlm` | patch embedding via `unfold`, LLaVA-style input building with text-only loss mask, ViT recompute | 95% exact-match captions at 300 steps; saved activations 8.9 MB -> 6.2 MB (ViT recompute) or 5.9 MB (frozen ViT) |
| 23 | `w23_mm_dataloading` | rank x worker shard assignment, stall measurement, worker budget, first-fit-decreasing packing, block-diagonal mask + restarted positions | packed logits == unpacked; token efficiency 66% padded -> 98% packed; stall 25% -> 5% with 2 workers at an 80 ms step |
| 24 | `w24_capstone` | pick a capstone; run `tools/gates.py`; write it up | - |

**Data-loader findings from building Week 23** (all measured, all in the code comments):
1. A float32 224x224 batch is ~19 MB per worker hand-off; returning uint8 cuts it 4x.
2. Collating 32 caption tensors as a Python list cost ~20 ms per step in shared-memory hand-offs; one padded tensor cut it to ~4 ms.
3. `workers_needed()` predicted that a 20 ms step needs 3 decode workers, which a 2-core machine cannot supply, and the stall indeed stayed at 27%.

**GPU.** Repeat Week 23 with GPU decode (`torchvision.io.decode_jpeg(device="cuda")` or NVIDIA DALI) and real JPEGs; the measurement functions do not change.
