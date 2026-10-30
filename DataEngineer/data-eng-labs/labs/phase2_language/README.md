# Phase 2 — Language data processing (Weeks 5–9)

The FineWeb/DCLM pipeline, stage by stage: extraction, language ID, heuristic
filters, fuzzy/exact/semantic dedup, model-based quality, decontamination.
Data: `common/webgen.py` (labelled synthetic web documents).

Reference numbers: 2-core CPU VM, `GATES_solution.md`.

## Week 5 — Extraction and language ID (`w05_extract_langid`)

| extractor | word precision | word recall |
| --- | --- | --- |
| ours (DOM walk, link-density rule) | 1.00 | 1.00 |
| naive, WET-like (all text) | 0.91 | 1.00 |

Naive-Bayes language ID over char 1–3-grams:

| words of input | accuracy | mean confidence |
| --- | --- | --- |
| 1 | 0.925 | 0.970 |
| 2 | 1.00 | 0.999 |
| 30 | 1.00 | 1.00 |

On one-word inputs the model is *over-confident* (97% confidence, 92.5%
accuracy) — why FineWeb's 0.65 fastText threshold must not be trusted on
short texts. Our synthetic extraction task is easy; Trafilatura on real HTML
is the realistic comparison (roadmap reading).

## Week 6 — Heuristic filters + PII (`w06_heuristic_filters`)

Gopher quality + repetition, C4, and FineWeb's three custom rules with
datatrove 0.10.1 defaults (read from its installed source).

| kind | in | kept |
| --- | --- | --- |
| good | 1766 | 1765 |
| boilerplate | 230 | 0 |
| spam | 262 | 0 |
| code | 182 | 0 |
| lorem | 137 | 0 |

Top rules by documents removed: gopher_word_count 316, c4_curly_bracket 182,
gopher_stop_words 177, c4_lorem_ipsum 137. Note: C4's curly-bracket rule removes
*all* code — fine for prose corpora, wrong if you want code.
**Discrepancy found:** the FineWeb paper describes the duplicated-line-character
rule at 0.1, while datatrove 0.10.1's default is 0.01; the lab uses the code default and says so.

## Week 7 — MinHash LSH (`w07_minhash`)

S-curve for 14 bands x 8 rows: P(candidate) = 0.565 / 0.772 / 0.924 / 0.988 at
Jaccard 0.70 / 0.75 / 0.80 / 0.85 — matching FineWeb's published 56 / 77 / 92 / 98.8%.

| bands x rows | precision | recall | LSH s | exact all-pairs s |
| --- | --- | --- | --- | --- |
| 14 x 8 | 0.84 | 0.97 | 0.11 | 2.79 |
| 20 x 5 | 0.63 | 1.00 | 0.09 | 2.79 |
| 9 x 13 | 0.96 | 0.81 | 0.10 | 2.79 |

**Bug worth knowing:** drawing hash multipliers below 2^32 (instead of over
the full prime range) biased the Jaccard estimate (0.76 estimated vs 0.64
true in the test that caught it).

## Week 8 — Exact, substring and semantic dedup (`w08_exact_semantic_dedup`)

| method | exact copies removed (of 12) | reordered copies removed (of 12) | originals removed |
| --- | --- | --- | --- |
| exact hash | 12 | 0 | 0 |
| semantic (embeddings + k-means, SemDeDup) | 12 | 12 | 0 |

Suffix-array exact-substring dedup removes a shared newsletter blurb from every
document except the first (kept in doc 0: True; in others: 0).

## Week 9 — Quality classifier and decontamination (`w09_quality_decontam`)

Hashed uni+bigram linear classifier (fastText-style), deliberately overlapping
classes and 10% label noise: AUC **0.92**; top-10% precision 0.61, recall 0.61.

n-gram decontamination, recall of contaminated eval items (precision 1.0 everywhere):

| n | 0% edits | 5% edits | 15% edits |
| --- | --- | --- | --- |
| 8 | 1.00 | 1.00 | 0.80 |
| 13 | 1.00 | 0.75 | 0.25 |
| 20 | 1.00 | 0.40 | 0.10 |

The common 13-gram rule finds verbatim leaks and misses lightly edited ones.

## Caveats

Synthetic documents are far more separable than real web text; FineWeb's own
ablations are the evidence for which filters help on real data. Week 24 is
where you test them on a model, at toy scale.

## Sources

* Penedo et al., "The FineWeb Datasets", NeurIPS 2024 D&B; datatrove 0.10.1 source (default thresholds).
* Rae et al., "Scaling Language Models: Methods, Analysis & Insights from Training Gopher", 2021 (MassiveText rules).
* Raffel et al., "Exploring the Limits of Transfer Learning…" (C4), JMLR 2020.
* Broder, "On the resemblance and containment of documents", 1997 (MinHash).
* Lee et al., "Deduplicating Training Data Makes Language Models Better", ACL 2022.
* Abbas et al., "SemDeDup", 2023.
* Li et al., "DataComp-LM (DCLM)", 2024; Joulin et al., "Bag of Tricks for Efficient Text Classification" (fastText), 2017.
* Brown et al., "Language Models are Few-Shot Learners", 2020 (13-gram decontamination).
