# Phase 6 — Production and the capstone (Weeks 22–24)

Running the pipeline every day, noticing when it breaks, and proving it helps.

## Week 22 — Incremental pipeline (`w22_incremental_pipeline`)

Quality filter → exact dedup (sha1 of normalised text) → near dedup (MinHash
LSH, 14 bands x 8 rows) against *everything already kept*, batch by batch.
State = manifest + exact index + LSH index, committed by write-temp-then-rename
after the batch's output file. Five daily batches of 400 docs:

| batch | in | quality | exact dup | near dup | kept |
| --- | --- | --- | --- | --- | --- |
| crawl-0 | 400 | 103 | 31 | 30 | 236 |
| crawl-1 | 400 | 97 | 27 | 25 | 251 |
| crawl-2 | 400 | 89 | 22 | 30 | 259 |
| crawl-3 | 400 | 90 | 23 | 25 | 262 |
| crawl-4 | 400 | 114 | 24 | 20 | 242 |

* incremental output == one full recompute over all batches: **True** (1,250 docs)
* documents processed over 5 days: 2,000 incremental vs 6,000 recomputing daily
* re-running a committed batch: skipped; a crash between output and commit
  leaves state unchanged and the retry reproduces the reference exactly (tested)

## Week 23 — Observability (`w23_observability`)

Per-batch profile (log length, share of lines ending in punctuation,
non-alphabetic share, language mix); alert if PSI > 0.25, or KS p < 1e-3 *and*
the KS gap > 0.1. Reference: 3,000 clean docs; 10 batches of 1,500 per row:

| batch | detected | example alert |
| --- | --- | --- |
| clean | 0/10 false alarms | |
| broken extractor (boilerplate 8% → 35%) | 10/10 | log_words PSI=0.43; non_alpha PSI=0.32 |
| truncating fetcher (40% cut at 1 KB) | 10/10 | log_words PSI=0.30 |
| language-ID swap (half of German → "nl") | 10/10 | lang PSI=0.73 |
| spam 10% → 13% | **0/10** | |

The last row is deliberate: a 3-point shift in spam share is real but
invisible to distribution-level monitors at this batch size. Catching it needs
a metric that measures spam directly (e.g. the share rejected by the Week 6
filters), which is the exercise's stretch goal. The effect-size floor exists
because with thousands of docs a KS p-value flags shifts nobody cares about.

## Week 24 — Capstone: data ablation (`w24_ablation_capstone`)

The FineWeb/DCLM method at toy scale: the same character-level transformer
(2 layers, d=96, ~0.25M parameters), the same budget (300 steps x 64 x 64 =
1.23M characters), trained on four pipeline variants of one dirty pool (1,500
docs: 55% good, spam, boilerplate, code, lorem; 12% exact and 12% near dups),
evaluated on held-out clean documents. Two seeds each:

| variant | docs | val bits/char (mean) | spread (max−min) |
| --- | --- | --- | --- |
| raw | 1500 | 1.699 | 0.007 |
| + heuristic filter | 963 | **1.663** | 0.0004 |
| + filter + dedup (your W22 pipeline) | 735 | 1.668 | 0.004 |
| + filter + dedup + n-gram quality classifier (top 70%) | 514 | 1.665 | 0.013 |

What the evidence supports: the heuristic filter lowers held-out loss by
~0.036 bits/char, about 5x the seed spread. Dedup and the classifier make **no
difference beyond noise** here. That matches what you would expect at this
scale: with 1.2M training characters drawn as random windows, duplicates only
re-weight a pool the model sees about once, while the published dedup gains
concern memorisation and multi-epoch training at much larger scale. The
capstone write-up should say exactly this rather than claim every stage helped.

For comparison, the real ablations: FineWeb trained 1.82B-parameter models on
~28B tokens per variant; DCLM's smallest competition scale is 412M parameters
on 8.2B tokens.

## Caveats

* Week 24 runs ~4 minutes on 2 cores (tests use a 120-step version, ~30 s).
* Two seeds is the minimum for a spread; use more before believing a
  difference smaller than ~3x the spread.
* Week 22's state is a pickle in one directory — fine for a lab, not for a
  multi-writer service (use the Week 16 table or a database).

## Sources

* Penedo et al., "The FineWeb Datasets", NeurIPS 2024; FineWeb blog post — ablation models "1.82B parameters", "~28B tokens".
* Li et al., "DataComp-LM", 2024 — Table 1, 400M-1x scale: 412M parameters, 8.2B tokens.
* Lee et al., "Deduplicating Training Data Makes Language Models Better", ACL 2022.
* Siddiqi, *Credit Risk Scorecards*, Wiley 2006 — PSI rule-of-thumb bands (industry convention, not a statistical test).
* Common Crawl — https://commoncrawl.org (monthly-ish crawl releases).
