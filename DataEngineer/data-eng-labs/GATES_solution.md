# Gates (solution)

2026-10-29 22:09 | Linux-6.18.44-fc-v185-x86_64-with-glibc2.39 | Python 3.13.18 | 2 cores | CPU only

## phase1_core / w01_columnar  (5.8s)

query: count(lang = 'de') over 20000 docs
| layout | file MB | rows | MB scanned | ms |
| --- | --- | --- | --- | --- |
| jsonl | 26.72 | 978 | 26.72 | 200 |
| parquet_snappy | 12.33 | 978 | 0.005598 | 6.562 |
| parquet_zstd | 9.384 | 978 | 0.004267 | 5.312 |
| parquet_sorted | 8.639 | 978 | 9e-05 | 1.486 |

## phase1_core / w02_query_engines  (15.6s)

| engine | Q1 ms | Q2 ms | Q3 ms |
| --- | --- | --- | --- |
| pandas | 26.85 | 19.54 | 32.51 |
| duckdb | 13.65 | 8.202 | 9.826 |
| pyarrow | 10.33 | 13.52 |  |

DuckDB plan for WHERE lang='de':
╭─ Ungrouped Aggregate ────╮
│ Aggregates: count_star() │
│ ~1 row                   │
╰─────────────┬────────────╯
╭─ Read Parquet ───────────╮
│ Function: READ_PARQUET   │
│ Filters: lang = 'de'     │
│ ~10,000 rows             │
╰──────────────────────────╯


## phase1_core / w03_tokens  (1.8s)

| set | tokens/byte |
| --- | --- |
| english prose | 0.2259 |
| german prose | 0.3947 |
| french prose | 0.4209 |
| code | 0.8785 |
| spam | 0.582 |

## phase1_core / w04_parallel  (re-run alone)

cores: 2
| workers | seconds | speedup |
| --- | --- | --- |
| 1 | 0.7669 | 1 |
| 2 | 0.4397 | 1.744 |
| 4 | 0.4481 | 1.712 |


_Note: the first full gates run measured 0.94x here because a test suite was running at the same time on this 2-core VM; this is the re-run on an idle machine. Scaling numbers are only meaningful on an otherwise idle box._

## phase2_language / w05_extract_langid  (0.3s)

| extractor | word precision | word recall |
| --- | --- | --- |
| ours | 1 | 1 |
| naive (WET-like) | 0.9076 | 1 |
| words | accuracy | mean confidence |
| --- | --- | --- |
| 1 | 0.925 | 0.9697 |
| 2 | 1 | 0.9987 |
| 30 | 1 | 1 |


## phase2_language / w06_heuristic_filters  (1.9s)

| rule | docs removed | % tokens removed |
| --- | --- | --- |
| gopher_word_count | 316 | 1.776 |
| c4_curly_bracket | 182 | 2.236 |
| gopher_stop_words | 177 | 2.144 |
| c4_lorem_ipsum | 137 | 1.102 |

| kind | in | kept |
| --- | --- | --- |
| boilerplate | 230 | 0 |
| good | 1766 | 1765 |
| spam | 262 | 0 |
| code | 182 | 0 |
| lorem | 137 | 0 |

## phase2_language / w07_minhash  (3.3s)

S-curve b=14 r=8 at s=0.7: 0.565
S-curve b=14 r=8 at s=0.75: 0.772
S-curve b=14 r=8 at s=0.8: 0.924
S-curve b=14 r=8 at s=0.85: 0.988
| bands x rows | P(cand|s=thr) | precision | recall | LSH s | exact s |
| --- | --- | --- | --- | --- | --- |
| 14x8 | 0.7716 | 0.838 | 0.974 | 0.107 | 2.787 |
| 20x5 | 0.9956 | 0.6311 | 1 | 0.09434 | 2.787 |
| 9x13 | 0.1946 | 0.9612 | 0.8052 | 0.1023 | 2.787 |


## phase2_language / w08_exact_semantic_dedup  (1.6s)

120 originals, 12 exact copies, 12 reordered copies
| method | removed exact copies | removed reordered | removed originals |
| --- | --- | --- | --- |
| exact hash | 12 | 0 | 0 |
| semantic (embeddings) | 12 | 12 | 0 |
| suffix array (newsletter blurb) | - | - | blurb kept in doc 0: True; in others: 0 |


## phase2_language / w09_quality_decontam  (2.7s)

quality classifier AUC 0.923; top-10% precision 0.61 recall 0.61
| n-gram | edit rate | precision | recall |
| --- | --- | --- | --- |
| 8 | 0 | 1 | 1 |
| 8 | 0.05 | 1 | 1 |
| 8 | 0.15 | 1 | 0.8 |
| 13 | 0 | 1 | 1 |
| 13 | 0.05 | 1 | 0.75 |
| 13 | 0.15 | 1 | 0.25 |
| 20 | 0 | 1 | 1 |
| 20 | 0.05 | 1 | 0.4 |
| 20 | 0.15 | 1 | 0.1 |


## phase3_acquisition / w10_link_control  (1.0s)

| metric | value |
| --- | --- |
| raw urls | 3960 |
| canonical | 1320 |
| shrink | 0.6667 |
| true pages lost | 0 |
| bloom bits/url | 9.586 |
| bloom fp | 0.0116 |
| mirrors | {'precision': 1.0, 'recall': 1.0, 'f1': 1.0} |

## phase3_acquisition / w11_frontier  (0.0s)

| path | allowed for DataBot | allowed for any other bot |
| --- | --- | --- |
| / | True | False |
| /private/x | False | False |
| /private/public | True | False |
| /private/public/more | False | False |
| /a/b.pdf | False | False |
| /docs/a.pdf | True | False |
| /docs/a.pdf?x=1 | True | False |

crawl-delay 2.0; fetch order: t=0 fast.example.com/0, t=0 slow.example.com/2, t=1 fast.example.com/1, t=2 fast.example.com/2, t=2 slow.example.com/1, t=4 slow.example.com/0

## phase3_acquisition / w12_fetcher  (2.4s)

reachable 123, attempted 131, missing 0, pages/s 52, DNS lookups 11 for 11 hosts, {'requests': 205, 'retries': 58, 'failures': 1}


## phase3_acquisition / w13_scheduling  (0.5s)

| policy | fetches | kept tokens | kept/fetch |
| --- | --- | --- | --- |
| bfs | 300 | 45417 | 151.4 |
| opic | 300 | 44761 | 149.2 |
| yield_ucb | 300 | 61618 | 205.4 |

300 pages, 30 refreshes/day:
| policy | mean freshness |
| --- | --- |
| proportional | 0.06401 |
| uniform | 0.3889 |
| optimal | 0.4236 |

## phase4_infrastructure / w14_lsm  (1.4s)

| policy | write amp | read amp | space amp | tables |
| --- | --- | --- | --- | --- |
| tiered | 4.332 | 0.533 | 1.126 | 6 |
| leveled | 19.05 | 0.503 | 1 | 1 |

## phase4_infrastructure / w15_log_queue  (6.3s)

2000 sends, 5% lost acks, 5% crash chance per batch commit:
| mode | idempotent producer | log records | restarts | lost | duplicates |
| --- | --- | --- | --- | --- | --- |
| at_most_once | True | 2000 | 19 | 133 | 0 |
| at_least_once | True | 2000 | 19 | 0 | 133 |
| exactly_once | False | 2137 | 20 | 0 | 137 |
| exactly_once | True | 2000 | 19 | 0 | 0 |

## phase4_infrastructure / w16_lake_table  (0.2s)

| what | value |
| --- | --- |
| pruned scan doc_id in [4200,4300] | 1/10 files read |
| 16 concurrent appends | 16 rows landed |
| after compaction | 1 live file(s), 26 older snapshots still readable |

## phase4_infrastructure / w17_dag_engine  (4.1s)

| run | ran | cached | s |
| --- | --- | --- | --- |
| cold | 5 | 0 | 1.375 |
| warm (nothing changed) | 0 | 5 | 0.006026 |
| filter param changed | 3 | 2 | 1.94 |

MinHash k=64, 5-gram shingles:
| impl | ms/doc |
| --- | --- |
| python | 2.953 |
| rust (ctypes) | 0.02843 |
| speed-up | 103.9 |

## phase5_multimodal / w18_image_text_filter  (16.1s)

device=cpu; zero-shot 16-way accuracy on held-out images: 0.814
noisy pool: 3000 pairs, 25% wrong captions, 15% junk alt-text (60% good)
| filter | kept | precision | recall of good |
| --- | --- | --- | --- |
| none | 3000 | 0.6057 | 1 |
| caption heuristic | 2562 | 0.7092 | 1 |
| CLIP top 30% | 900 | 1 | 0.4953 |
| CLIP top 50% | 1500 | 0.9947 | 0.8211 |
| CLIP top 60% | 1800 | 0.9872 | 0.978 |

## phase5_multimodal / w19_image_dedup  (5.1s)

Median Hamming distance (64 bits) after each edit, and the 1st percentile between DIFFERENT images:
| hash | distinct p1 dist | resize 50% | jpeg q=25 | brightness +25% | crop 8% | watermark | h-flip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aHash | 13 | 0 | 0 | 2 | 3 | 9 | 30 |
| dHash | 15 | 1 | 1 | 2 | 5 | 4 | 37 |
| pHash | 22 | 0 | 2 | 2 | 8 | 14 | 32 |

Dedup of 2607 images (607 planted near-dups: resize/jpeg/brightness/watermark):
| hash | radius | kept | planted dups removed | distinct images lost |
| --- | --- | --- | --- | --- |
| dHash | 3 | 2184 | 0.6886 | 5 |
| dHash | 7 | 1942 | 0.9176 | 108 |
| pHash | 3 | 2239 | 0.6063 | 0 |
| pHash | 7 | 2157 | 0.7414 | 0 |
multi-index (4x16-bit bands, r=3): 2.4 candidates verified per query vs 2607 brute force

## phase5_multimodal / w20_video_shots  (2.1s)

Thresholds swept on videos 0-9; scored on held-out videos 10-19 (10 shots, 3 ten-frame cross-fades each), +-2 frames:
| detector | precision | recall |
| --- | --- | --- |
| pixel hard | 0.7195 | 0.6333 |
| hist hard | 0.8396 | 0.8556 |
| hist twin | 0.9196 | 0.9667 |

video 0: 255 frames -> 10 clips; motion of first clip 8.03 vs freeze-frame 0.00; keyframe of clip 0 = frame 3

## phase5_multimodal / w21_training_layouts  (10.1s)

5000 samples on local disk with a warm page cache ('files' = objects = GET requests on an object store; not measured here):
| layout | files | write s | read s |
| --- | --- | --- | --- |
| loose files | 10000 | 0.4809 | 0.2134 |
| tar shards (1000/shard) | 5 | 0.5268 | 0.6039 |
| parquet | 1 | 0.04549 | 0.03109 |

Mixture over 400000 draws:
| source | weight | epochs (naive) | epochs (cap 4) | share (cap 4) |
| --- | --- | --- | --- | --- |
| web | 0.6 | 0.2404 | 0.2601 | 0.6501 |
| code | 0.25 | 0.997 | 1.08 | 0.2699 |
| books | 0.15 | 7.486 | 3.998 | 0.07997 |

## phase6_production / w22_incremental_pipeline  (4.5s)

| batch | in | quality | exact dup | near dup | kept |
| --- | --- | --- | --- | --- | --- |
| crawl-0 | 400 | 103 | 31 | 30 | 236 |
| crawl-1 | 400 | 97 | 27 | 25 | 251 |
| crawl-2 | 400 | 89 | 22 | 30 | 259 |
| crawl-3 | 400 | 90 | 23 | 25 | 262 |
| crawl-4 | 400 | 114 | 24 | 20 | 242 |

incremental == full recompute: True (1250 docs)
docs processed over 5 days: incremental 2000 (1.03s) vs daily full recompute 6000 (3.23s)
re-running committed batch crawl-2: {'batch': 'crawl-2', 'skipped': True}

## phase6_production / w23_observability  (21.0s)

Reference: 3000 clean docs. Each row: 10 batches of 1500 docs. Alert = PSI > 0.25, or KS p < 1e-3 with gap > 0.1.
| batch | detected | example |
| --- | --- | --- |
| clean (false alarms) | 0 |  |
| broken extractor | 1 | log_words: PSI=0.43; punct_lines: PSI=0.26; non_alpha: PSI=0.32 |
| truncating fetcher | 1 | log_words: PSI=0.30; punct_lines: PSI=0.26 |
| language-ID swap | 1 | lang: PSI=0.73 |
| spam +3 pts (subtle) | 0 |  |

## phase6_production / w24_ablation_capstone  (244.2s)

device=cpu; char transformer 2x96, 300 steps x 64 x 64 chars = 1.23M training chars per run, seeds 0 and 1
held-out: 60 clean reference docs
| variant | docs | chars | val bpc mean | spread (max-min) |
| --- | --- | --- | --- | --- |
| raw | 1500 | 1576004 | 1.699 | 0.007492 |
| filtered | 963 | 1431096 | 1.663 | 0.0004121 |
| filtered+dedup | 735 | 1119830 | 1.668 | 0.004412 |
| filtered+dedup+classifier | 514 | 785699 | 1.665 | 0.01336 |
(244s)
