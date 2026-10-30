# Phase 5 — Multimodal data (Weeks 18–21)

The JD's multimodal direction: image-text pairs, images, video, and the
storage formats training jobs read. Data comes from `common/shapes.py`
(coloured shapes with captions, synthetic videos with known cuts and fades), so
every filter can be scored against ground truth.

Reference numbers: 2-core CPU VM, `python <lab>/solution.py`.

## Week 18 — CLIP-score filtering (`w18_image_text_filter`)

A tiny CLIP (3-layer CNN image tower, mean-of-embeddings text tower, symmetric
InfoNCE) trained on 4,000 clean pairs in ~30 s on CPU. Zero-shot 16-way
accuracy on held-out images: **0.81** (chance 0.06).

Noisy pool: 3,000 pairs, 25% wrong captions, 15% junk alt-text (60% good):

| filter | kept | precision | recall of good |
| --- | --- | --- | --- |
| none | 3000 | 0.61 | 1.00 |
| caption heuristic (has a colour and a shape word) | 2562 | 0.71 | 1.00 |
| CLIP top 30% | 900 | 1.00 | 0.50 |
| CLIP top 50% | 1500 | 0.99 | 0.82 |
| CLIP top 60% | 1800 | 0.99 | 0.98 |

The heuristic removes junk alt-text but cannot see *wrong* captions; only a
model that looks at the image can. The keep-fraction is a precision/recall
dial — DataComp's point that the threshold is a design choice to ablate.

## Week 19 — Perceptual hashing (`w19_image_dedup`)

aHash, dHash, pHash (DCT) on 96x96 synthetic "photos"; median Hamming distance
(of 64 bits) after each edit, versus the 1st percentile between *different* images:

| hash | distinct p1 | resize 50% | JPEG q=25 | brightness +25% | crop 8% | watermark | h-flip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aHash | 13 | 0 | 0 | 2 | 3 | 9 | 30 |
| dHash | 15 | 1 | 1 | 2 | 5 | 4 | 37 |
| pHash | 22 | 0 | 2 | 2 | 8 | 14 | 32 |

Dedup of 2,607 images with 607 planted near-duplicates:

| hash | radius | planted dups removed | distinct images wrongly merged |
| --- | --- | --- | --- |
| dHash | 3 | 69% | 5 |
| dHash | 7 | 92% | 108 |
| pHash | 3 | 61% | 0 |
| pHash | 7 | 74% | 0 |

No hash catches flips; crops and watermarks are hard for pHash. pHash has the
widest gap between "edited" and "different", so it never merged two distinct
images here; dHash catches more edits at the cost of false merges at r=7.
Multi-index hashing (4 x 16-bit bands, r=3) verified 2.4 candidates per query
instead of 2,607.

## Week 20 — Shot boundaries (`w20_video_shots`)

Thresholds swept on videos 0–9, scored on held-out videos 10–19 (10 shots and
3 ten-frame cross-fades each), +-2 frames:

| detector | precision | recall |
| --- | --- | --- |
| mean pixel difference, single threshold | 0.72 | 0.63 |
| colour-histogram difference, single threshold | 0.84 | 0.86 |
| colour histogram, twin-comparison | **0.92** | **0.97** |

Twin-comparison finds fades by accumulating sub-threshold differences.
Clips are cut between boundaries (trimming transition frames), filtered by
motion (a freeze frame scores 0.00 vs ~8 for a moving shot), and get a
representative keyframe for captioning.

## Week 21 — Training layouts, resume, mixtures (`w21_training_layouts`)

5,000 samples, local disk, warm page cache, two runs:

| layout | files | write s | read s |
| --- | --- | --- | --- |
| loose files | 10,000 | 0.44 / 0.48 | 0.31 / 0.21 |
| tar shards (WebDataset) | 5 | 0.57 / 0.53 | 0.46 / 0.60 |
| Parquet | 1 | 0.34 / 0.05 | 0.05 / 0.03 |

These sub-second timings are noisy (Parquet's write varied 7x between runs);
the stable result is the ordering of reads: Parquet fastest, tar slowest.

Honest reading: on a local SSD with a warm cache, Python's `tarfile` is
*slower* than loose files. The case for shards is object storage, where every
file is a request — 10,000 GETs vs 5 — which this lab does not measure.

`ShuffledStream` (per-epoch shard permutation + shuffle buffer) checkpoints
its full state; the test stops after 333 samples, restores in a fresh object,
and checks the next 367 equal an uninterrupted run exactly.

Mixture over 400,000 draws (web 1M docs, code 100k, books 8k; weights .6/.25/.15):

| source | epochs, naive weights | epochs, capped at 4 | share, capped |
| --- | --- | --- | --- |
| web | 0.24 | 0.26 | 0.65 |
| code | 1.00 | 1.08 | 0.27 |
| books | **7.49** | **4.00** | 0.08 |

## Caveats

* Toy scale and toy semantics: 16 concepts and 32x32 images. The *ranking*
  of methods is the result; the absolute numbers will not transfer.
* Week 18/24 run on CUDA if present; this build only executed them on CPU.
* JPEG and resizing are Pillow's; real pipelines see many encoders.

## Sources

* Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021.
* Schuhmann et al., "LAION-5B", NeurIPS 2022 Datasets & Benchmarks — CLIP ViT-B/32 similarity thresholds 0.28 (English) / 0.26.
* Gadre et al., "DataComp: In search of the next generation of multimodal datasets", NeurIPS 2023.
* Webster et al., "On the De-duplication of LAION-2B", arXiv:2303.12733 — "roughly 700 million images, or about 30%, of LAION-2B's images are likely duplicated".
* Norouzi, Punjani, Fleet, "Fast Search in Hamming Space with Multi-Index Hashing", CVPR 2012.
* Zhang, Kankanhalli, Smoliar, "Automatic partitioning of full-motion video", Multimedia Systems 1993 (twin-comparison).
* Chen et al., "Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers", CVPR 2024.
* WebDataset — https://github.com/webdataset/webdataset
* Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS 2023 — "up to 4 epochs of repeated data yields negligible changes to loss".
