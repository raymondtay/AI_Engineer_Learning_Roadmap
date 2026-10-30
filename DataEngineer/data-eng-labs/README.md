# data-eng-labs — 24 weeks to Pre-training Data Engineer

Semi-worked codebases that follow the roadmap
[Pre-training Data Engineer: 6-Month Roadmap](https://claude.ai/artifact/Y1KeyBxXKq97DvpaqURb6s)
week by week. Every lab runs on a laptop CPU; Week 18 and Week 24 use CUDA when present.

Each week has three files:

| file | what it is |
| --- | --- |
| `solution.py` | the reference implementation; the parts you write are fenced `# >>> SOLUTION: hint` … `# <<< SOLUTION` |
| `exercise.py` | generated from it: each fence becomes `raise NotImplementedError("TODO(...): hint")`. **This is the file you edit.** |
| `test_lab.py` | tests that pass on the solution and fail on an unfinished exercise |

```bash
pip install -r requirements-cpu.txt
LAB=exercise pytest labs/phase2_language/w07_minhash   # your work
LAB=solution pytest -q                                  # the reference, everything
python labs/phase2_language/w07_minhash/solution.py     # a lab's gate measurement
python tools/gates.py                                   # all gates, your code -> GATES.md
```

`LAB` defaults to `exercise`. Regenerate exercises after editing a solution: `python tools/make_exercises.py`
(`--check` fails if any are stale). **Copy your `exercise.py` somewhere safe before regenerating — it overwrites them.**

## The 24 weeks

| phase | week | lab | you build | the gate measures |
| --- | --- | --- | --- | --- |
| 1 Core | 1 | `w01_columnar` | JSONL → Parquet, row groups, stats pruning, projection | bytes read and time per query by layout |
| | 2 | `w02_query_engines` | the same pipeline in pandas, DuckDB, PyArrow and Spark | plan inspection, engine agreement, speed |
| | 3 | `w03_tokens` | byte-level BPE, tokens/byte by language, stable hash shuffle, external sort | tokens per byte; sort under a memory cap |
| | 4 | `w04_parallel` | multiprocessing map-reduce over shards | speed-up vs workers (Amdahl) |
| 2 Language | 5 | `w05_extract_langid` | HTML text extraction, naive-Bayes language ID | extraction precision, langid accuracy and calibration |
| | 6 | `w06_heuristic_filters` | Gopher / C4 / FineWeb filters, PII redaction | good kept vs bad kept, per-rule removals |
| | 7 | `w07_minhash` | MinHash + LSH banding, the S-curve | P/R vs exact Jaccard, speed |
| | 8 | `w08_exact_semantic_dedup` | suffix-array exact-substring dedup, SemDeDup | spans removed, semantic dups found |
| | 9 | `w09_quality_decontam` | fastText-style quality classifier, n-gram decontamination | AUC, top-k precision, recall by n and edit rate |
| 3 Acquisition | 10 | `w10_link_control` | URL normalisation, Bloom filter, SimHash mirror detection | URLs collapsed, FP rate, bits/URL |
| | 11 | `w11_frontier` | RFC 9309 robots.txt, polite per-host frontier | robots conformance, politeness |
| | 12 | `w12_fetcher` | asyncio fetcher: DNS cache, retries, checkpoint/resume | pages/s, retries, nothing lost on restart |
| | 13 | `w13_scheduling` | BFS vs OPIC vs yield-UCB, Cho & Garcia-Molina freshness | kept tokens per fetch; mean freshness |
| 4 Infrastructure | 14 | `w14_lsm` | LSM tree: WAL, SSTables, Bloom, tiered vs leveled compaction | write / read / space amplification |
| | 15 | `w15_log_queue` | partitioned log, idempotent producer, transactional consumer | lost vs duplicated under injected crashes |
| | 16 | `w16_lake_table` | Iceberg-shaped table: snapshots, manifests, CAS commit | pruning, concurrent commits, time travel |
| | 17 | `w17_dag_engine` | DAG engine with content-addressed cache; **Rust** MinHash via C ABI | cache hits; Rust vs Python speed |
| 5 Multimodal | 18 | `w18_image_text_filter` | tiny CLIP, CLIP-score filtering (**PyTorch**) | precision/recall of kept pairs vs heuristic |
| | 19 | `w19_image_dedup` | aHash / dHash / pHash, multi-index Hamming search | robustness per edit, dedup recall vs false merges |
| | 20 | `w20_video_shots` | shot boundaries (twin-comparison), clips, motion, keyframes | P/R on cuts and fades |
| | 21 | `w21_training_layouts` | WebDataset tar shards, resumable shuffle, capped mixtures | read time by layout; exact resume; epochs per source |
| 6 Production | 22 | `w22_incremental_pipeline` | incremental, idempotent, crash-safe cleaning | incremental == full recompute; work saved |
| | 23 | `w23_observability` | profiles, PSI and KS drift, alerts | detection vs false alarms on injected failures |
| | 24 | `w24_ablation_capstone` | same tiny LM, same budget, four pipeline variants | held-out bits/char with seed spread |

Each phase folder has a README with the reference numbers and the sources behind every claim.

## Verification (this build)

Measured on a 2-core x86 cloud VM, CPU only, Python 3.13, PyTorch 2.15, PyArrow 26, DuckDB 2.0, Spark 4.2.1 (OpenJDK 21), Rust 1.97.
Full output: `TEST_RESULTS.txt` and `GATES_solution.md`.

| check | result |
| --- | --- |
| `LAB=solution pytest` | **107 passed, 0 skipped** in 86 s (Spark and Rust tests ran) |
| `LAB=exercise pytest` | 92 failed + 9 errors, 6 passed — every lab fails until its TODOs are done; the 6 passes test scaffolding you are given |
| exercise TODO blocks | 82 across 24 labs; `make_exercises.py --check` clean |
| `LAB=solution python tools/gates.py` | all 24 gates produced numbers (`GATES_solution.md`, ~8 min) |

## Honest caveats

* **The data is synthetic.** `common/webgen.py`, `common/simweb.py` and `common/shapes.py` generate web pages,
  a crawlable web, images and videos *with ground truth*. That is what makes precision and recall measurable,
  but it is far cleaner and more regular than Common Crawl. Thresholds tuned here are a starting point, not a
  production setting; every phase README says which numbers to re-measure on real data.
* **Scale is laptop scale.** The algorithms are the ones used at scale (MinHash LSH, suffix arrays, LSM trees,
  Iceberg-style commits, WebDataset shards), but a 2-core box cannot show the distributed failure modes. Week 2
  runs Spark locally; the roadmap's capstone suggests running the Week 22 pipeline on a real cluster.
* **Week 24 is small.** A 0.25M-parameter character model on 1.2M characters. It reproduces the *method* of
  FineWeb/DCLM ablations, and here only the heuristic filter beat the seed spread; dedup and the classifier did
  not. That is a real result at this scale, not a bug — the README discusses why.
* **GPU paths** (Week 18, Week 24) are written device-agnostic but were only executed on CPU in this build.
* **Spark and Rust are optional.** Tests needing them are marked `spark` / `rust` and skip when Java or cargo
  is missing.
