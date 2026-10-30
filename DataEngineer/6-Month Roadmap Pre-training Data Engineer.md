# 6-Month Roadmap: Pre-training Data Engineer

Oct 29, 2026 · @Ray

## How to read this roadmap

The plan is 24 weeks at about 25 hours a week (roughly 600 hours) and covers all four recruitment directions in the JD. Everyone does a shared four-week core, then one phase per direction, then a production phase that ties them together. Each phase ends with a gate: a shipped artifact with a number attached (precision/recall of a dedup, pages per second, tokens kept per TB, stall time).

**Why this order.** Language data processing comes first after the core because it is the biggest part of the JD (MinHash and vector dedup are named explicitly) and every other direction feeds it: acquisition produces its input, infrastructure runs it, multimodal reuses its dedup and quality ideas. Infrastructure comes after acquisition so that you build a KV store and a queue knowing what the crawler needs from them.

**One track deep, three broad.** Six months is enough to be expert in one direction and solid in the other three. Pick your deep track by week 4 and spend the Phase 6 capstone on it. The gates for the other phases are set at intermediate level.

**Entry level assumed.** Comfortable Python and SQL, basic Linux. No prior crawling, dedup or distributed-systems experience. The JD's language and infrastructure directions also ask for Rust or C++; the infrastructure phase is where you write it.

**Hardware.** A laptop for weeks 1-13. From week 14, a few cloud CPU nodes and an object store bucket; GPU only for the multimodal embedding and quality-classifier work and the data ablations, a few dozen GPU-hours in total. The budget is in the last section.

**What the JD signals.** It names cleaning-feedback-driven crawl scheduling, RAG freshness, mirror-site detection, MinHash and vector dedup, TB-token pipelines, Pretrain/SFT/RL storage layouts, KV stores, message queues, data lakes and PB-scale processing. Each one is a named topic below, not an extra.

## At a glance

&#91;embedded content: 24-week roadmap · 6 phases, 6 gates\]

Phases 2-5 each cover one JD direction; Phase 6 runs them together as a daily pipeline and measures their effect on models.

## Skills matrix: JD line items to exit criteria

Each row is a line from the JD. Expert means you can defend a design choice with your own measurement. Numeric targets are gates set for this plan; published reference points are cited in the phases.

| Direction | JD item | Beginner (uses it) | Intermediate (modifies it) | Expert by week 24 (builds and defends it) | Phase |
| --- | --- | --- | --- | --- | --- |
| All | Python + pandas / Spark / DuckDB | Query Parquet with DuckDB | Write partitioned, predicate-pushdown-friendly datasets | Explain a slow job from its plan and I/O profile, and fix it | 1 |
| Language | Corpus cleaning framework; scheduling, recovery, monitoring | Run datatrove locally | Add a custom filter and stats block | Own a resumable, idempotent daily pipeline with quality regression | 2, 6 |
| Language | MinHash dedup | Run a MinHash pipeline | Choose n-gram size, bands and rows from the S-curve | Measure precision/recall against exact Jaccard and justify the threshold | 2 |
| Language | Vector dedup | Embed and cluster | SemDeDup-style per-cluster dedup | Show the effect on a downstream model, not only the removal rate | 2, 6 |
| Language | Quality, stats, regression | Apply heuristic filters | Train a quality classifier | Run a data ablation that ranks filters by model quality per token kept | 2, 6 |
| Acquisition | Network-wide selection and scheduling | Crawl a site politely | Priority frontier with per-host queues and quotas | Scheduling driven by cleaning feedback (yield of kept tokens per site) | 3 |
| Acquisition | RAG freshness | Recrawl on a timer | Change-rate-based recrawl | Measured staleness vs fetch budget trade-off | 3 |
| Acquisition | Link scale control | Normalise URLs | Detect mirrors and redundant parameters | Shrink a link library by a measured % with no loss of unique content | 3 |
| Acquisition | Download, DNS, connectivity, fault tolerance | asyncio fetcher | Retries, backoff, DNS cache, checkpointed frontier | Pages/s and failure-recovery numbers under injected faults | 3 |
| Infrastructure | KV database, message queue | Use RocksDB / Kafka | Write an LSM tree and an append-only log | Explain write/read/space amplification and delivery semantics from your own code | 4 |
| Infrastructure | Data lake, object storage | Write Parquet to S3 | Snapshot table format with atomic commits | Time travel, compaction and concurrent-writer conflict handling | 4 |
| Infrastructure | PB-scale processing framework; CPU / memory / network tuning | Run a Spark/Ray job | Build a DAG executor with shuffle and spill | Throughput tuned with profiles; a Rust or C++ hot path | 4 |
| Multimodal | Structure, storage, lifecycle, compliance | Store image-text pairs | Sharded, resumable training layouts | Layouts for Pretrain / SFT / RL with measured loader throughput | 5 |
| Multimodal | Dedup, quality screening, labeling | CLIP-score filter | Perceptual-hash dedup, video segmentation | Tie a filter to a training result | 5 |
| Multimodal | Analysis and visualisation | Plot dataset stats | Data cards and drift dashboards | Link data features to model metrics | 6 |

## Phase 1 (weeks 1-4): Data engineering core

Goal: move and measure data at speed, and know what a token costs. Every later phase is a pipeline over columnar files, so this phase is about formats, query engines and parallelism.

| Week | Focus | Do this |
| --- | --- | --- |
| 1 | Columnar formats | Parquet row groups, encodings and compression; Arrow in memory. Measure scan time and bytes read for row vs columnar, with and without predicate pushdown. |
| 2 | Query engines | The same aggregations in pandas, DuckDB and Spark (local). Read the query plans; find where each spills or serialises. |
| 3 | Tokens as the unit of account | Train a small BPE tokenizer; measure tokens per byte by language and by document type. Every later gate reports "tokens kept". |
| 4 | Parallel processing | Map, hash-partitioned shuffle, external sort, all with multiprocessing; then the same job in Ray or Spark. Measure scaling with cores. |

**Reference point.** Common Crawl's August 2026 crawl (CC-MAIN-2026-34) holds 2.14 billion pages, 360 TiB uncompressed, from 40.2 million hosts, published as WARC, WAT and WET files ([Common Crawl](https://commoncrawl.org/blog/august-2026-crawl-archive-now-available)). One monthly snapshot is the scale every design choice below must survive.

**Gate (end of week 4)**

- [ ] A table of scan time and bytes read for 4 storage layouts of the same 10 GB dataset, with the fastest one explained.
- [ ] Tokens-per-byte table for at least 3 languages with your tokenizer.
- [ ] A shuffle-based job that scales to all cores, with a speedup curve and the point where it stops scaling.

## Phase 2 (weeks 5-9): Language data processing

Goal: turn raw web pages into training text, and prove each step helps a model. This is the language direction of the JD: cleaning framework, MinHash and vector dedup, statistics and quality regression.

| Week | Focus | Do this |
| --- | --- | --- |
| 5 | Extraction and language ID | HTML to main text (Trafilatura vs your own boilerplate removal); language ID; read WET vs re-extracting from WARC. |
| 6 | Heuristic filters and PII | Implement the MassiveText (Gopher) and C4 rules and FineWeb's three custom rules; measure tokens removed by each; email/phone/IP redaction. |
| 7 | Fuzzy dedup | MinHash LSH from scratch: shingles, signatures, bands, the S-curve; then the datatrove pipeline. Precision/recall against exact Jaccard on a labelled sample. |
| 8 | Exact and semantic dedup | URL and exact-document hashing, exact-substring dedup with a suffix array, SemDeDup-style embedding clusters. |
| 9 | Model-based quality and decontamination | Train a fastText-style quality classifier and an educational-value regressor; n-gram overlap decontamination against eval sets. |

**Reference points**

- [FineWeb](https://arxiv.org/pdf/2406.17557) is 15T tokens from 96 Common Crawl snapshots. Its MinHash uses word 5-grams and 112 hashes as 14 buckets of 8, targeting 75% similarity. Deduplicating each snapshot separately matched RefinedWeb, while global dedup removed up to 90% of the oldest snapshots and scored worse in 350B-token ablations. Its custom line rules removed about 22% of tokens; C4's terminal-punctuation rule was dropped because it removed about 30%.
- FineWeb-Edu scored 460,000 pages with Llama-3-70B-Instruct, trained a linear regressor on embeddings, kept scores of 3 or more (F1 82%) and produced 1.3T tokens; applying it took 6,000 H100-hours (same paper).
- [DCLM](https://arxiv.org/abs/2406.11794v1) built a 240T-token pool and trained a 7B model on 2.6T tokens to 64% MMLU, 6.6 points above MAP-Neo with 40% less compute; its paper keeps the top 10% of documents by a fastText classifier trained on OpenHermes 2.5 and ELI5 positives ([PDF](https://export.arxiv.org/pdf/2406.11794)).
- [Lee et al.](https://arxiv.org/abs/2107.06499v1) found one 61-word sentence repeated over 60,000 times in C4; deduplicated models emit memorised text ten times less often.
- [SemDeDup](https://arxiv.org/abs/2303.09540v1) removed 50% of LAION with minimal performance loss, roughly halving training time.
- [datatrove](https://github.com/huggingface/datatrove) is the reference open framework: local, Slurm and Ray executors, resumable tasks, per-task stats, and MinHash, sentence and exact-substring dedup blocks.

**Gate (end of week 9)**

- [ ] A pipeline from WARC to clean JSONL/Parquet with a per-step table of documents and tokens removed.
- [ ] MinHash dedup with measured precision and recall at 3 thresholds, and the S-curve that predicts them.
- [ ] A quality classifier whose top-k selection beats random selection in a small-model ablation (Phase 6 runs the ablation harness; build it here).

## Phase 3 (weeks 10-13): Data acquisition pipeline

Goal: a crawler that spends its fetch budget where the cleaning pipeline says the good tokens are. This covers the acquisition direction: selection and scheduling from cleaning feedback, RAG freshness, the acquisition loop, and link scale control.

| Week | Focus | Do this |
| --- | --- | --- |
| 10 | Link scale control | URL normalisation and tracking-parameter stripping; Bloom-filter URL dedup; mirror and near-duplicate site detection with SimHash. Measure link-library shrinkage and unique content lost. |
| 11 | Frontier and politeness | Per-host queues, crawl delay, robots.txt parsing and caching, priorities and per-site quotas. |
| 12 | The fetch loop | asyncio fetcher with DNS cache, retries with backoff, timeouts, and a checkpointed frontier; inject failures and measure recovery. |
| 13 | Selection from feedback, and freshness | Score sites by the yield of kept tokens from Phase 2; link-graph priority (OPIC/PageRank); change-rate-based recrawl for RAG seeds. |

**Rules to implement exactly.** RFC 9309 says a crawler should not use a cached robots.txt for more than 24 hours unless it is unreachable, must parse at least 500 KiB, may access anything on a 4xx, must assume complete disallow on a 5xx, and should follow at least five redirects ([RFC 9309](https://www.rfc-editor.org/rfc/rfc9309)). These become unit tests in the lab.

**Why feedback-driven scheduling.** FineWeb's filters removed about 22% of tokens with one rule set and DCLM kept only the top 10% by classifier score (Phase 2). The yield of usable tokens varies a lot between sites, so a crawler that ranks hosts by measured yield spends the same budget on far more usable text. The gate asks you to measure that, not assume it.

**Gate (end of week 13)**

- [ ] A link library shrunk by a measured % with the unique-content loss reported.
- [ ] A polite crawler that passes RFC 9309 tests and recovers from injected DNS, timeout and 5xx failures, with pages/s.
- [ ] Yield-ranked scheduling vs breadth-first on the same budget: kept tokens per fetch, and staleness vs budget for a freshness set.

## Phase 4 (weeks 14-17): Data infrastructure

Goal: build each core component once at small scale so you can operate and tune the production versions. This covers the infrastructure direction: KV store, message queue, data lake, PB-scale processing framework, CPU/memory/network tuning, and the JD's Rust/C++ requirement.

| Week | Focus | Do this |
| --- | --- | --- |
| 14 | KV store | An LSM tree: write-ahead log, memtable, sorted SSTables, Bloom filters, compaction. Measure write, read and space amplification; then read RocksDB's tuning guide with your numbers in hand. |
| 15 | Message queue | A segmented append-only log with consumer offsets; at-least-once delivery; an idempotent producer (producer ID + sequence number dedup); offsets committed with output. |
| 16 | Data lake table | A snapshot table over Parquet: manifests, an atomically swapped current-snapshot pointer, optimistic concurrency with retry, time travel, small-file compaction. Compare with Apache Iceberg's design. |
| 17 | Processing framework and tuning | A DAG executor with hash-partitioned shuffle, spill to disk, retries and backpressure; port its hottest operator to Rust or C++ and profile CPU, memory and I/O. |

**Reference designs.** Kafka's idempotent producer gives each producer an ID and each message a sequence number so the broker discards resent duplicates, and exactly-once processing writes the consumer's offset in the same transaction as its output ([Confluent docs](https://docs.confluent.io/kafka/design/delivery-semantics.md)). The [Iceberg table spec](https://iceberg.apache.org/spec) is the reference for snapshot commits and time travel; read it after building your own.

**Gate (end of week 17)**

- [ ] LSM store with a write/read/space amplification table for two compaction policies.
- [ ] Queue that delivers exactly-once end to end under injected producer retries and consumer crashes, verified by a checker.
- [ ] Table format with two concurrent writers, one conflict detected and retried, and time travel to any snapshot.
- [ ] DAG job whose throughput you improved by a measured factor, with before/after profiles and the native-code operator.

## Phase 5 (weeks 18-21): Multimodal data

Goal: apply the text pipeline's ideas (filter, dedup, measure) to images and video, and store the result so training reads it fast. This covers the multimodal direction: structure and storage, cleaning and labelling efficiency, Pretrain/SFT/RL layouts.

| Week | Focus | Do this |
| --- | --- | --- |
| 18 | Image-text filtering | Embedding-similarity filtering of pairs (CLIP-score), caption heuristics, NSFW/watermark classifier hooks. Measure precision of the filter on a labelled sample. |
| 19 | Image dedup | Perceptual hashing (pHash via DCT) and embedding-based near-duplicates; robustness to crops, resizes and re-encodes. |
| 20 | Video | Shot-boundary detection, clip splitting, frame sampling; keep clips semantically consistent. |
| 21 | Training-ready layouts | Sharded tar (WebDataset-style) vs Parquet vs random-access formats; deterministic, resumable shuffling; mixture sampling; different layouts for Pretrain, SFT and RL. Measure loader throughput. |

**Reference points**

- [LAION-5B](https://arxiv.org/pdf/2210.08402) kept 5.85 billion pairs by removing pairs below 0.28 CLIP ViT-B/32 cosine similarity (0.26 for other languages), about 90% of the original 50 billion candidates.
- [DataComp](https://arxiv.org/abs/2304.14108v2) built a 12.8B-pair pool; a CLIP ViT-L/14 trained on DataComp-1B reached 79.2% zero-shot ImageNet, 3.7 points above OpenAI's CLIP ViT-L/14 at the same compute.
- [Panda-70M](https://arxiv.org/abs/2402.19479) split 3.8M high-resolution videos into semantically consistent clips and captioned them with several cross-modal teachers, yielding about 70M video-text pairs.

**Gate (end of week 21)**

- [ ] Image-text filter with precision and recall on a labelled sample, at 3 thresholds.
- [ ] Near-duplicate detector with recall under 4 kinds of edits.
- [ ] Shot detector with boundary F1 on a labelled set.
- [ ] Loader throughput table for 3 layouts, plus exact resume after a mid-epoch kill.

## Phase 6 (weeks 22-24): Production pipeline, observability, data-to-model ablations

Goal: run the pieces as a daily production line, and close the loop from data decisions to model quality. This is where the JD's "daily incremental processing, task scheduling, data statistics, quality regression, online problem checking" lives, and where its bonus points about model behaviour are earned.

| Week | Focus | Do this |
| --- | --- | --- |
| 22 | Incremental, idempotent pipeline | Stage manifests, exactly-once task completion, resumable reruns, backfills; a daily incremental run over new crawl data with dedup against everything already kept. |
| 23 | Observability and quality regression | Per-stage metrics, distribution drift tests between days, sampled human review queues, data cards; an alert that would have caught a broken extractor. |
| 24 | Data ablations and capstone | Train small models on mixtures that differ in one decision (filter, threshold, dedup scope); rank decisions by model quality per token. Finish the capstone. |

**Why ablations.** Every number in Phase 2 that mattered (FineWeb dropping a rule that removed 30% of tokens, per-dump vs global dedup, DCLM's 10% threshold) was decided by training small models, not by inspecting text. The capstone should end the same way.

**Capstone (pick one, tied to your deep track)**

1. **Language:** an end-to-end WARC to training-shard pipeline with MinHash plus semantic dedup and a quality classifier, and an ablation table.
2. **Acquisition:** a yield-driven crawler that beats breadth-first on kept tokens per fetch, with freshness for a RAG seed set.
3. **Infrastructure:** a small processing framework running the Phase 2 pipeline, with a Rust/C++ operator and profile-driven tuning.
4. **Multimodal:** an image-text pipeline (filter, dedup, sharded storage) with a measured effect on a small contrastive model.

**Gate (end of week 24)**

- [ ] Two consecutive "daily" incremental runs, the second processing only new data, with a reconciliation report.
- [ ] A drift alert that fires on a deliberately broken stage and stays quiet on a normal day.
- [ ] An ablation table ranking at least 3 data decisions by model quality per token, and a public write-up.

## Bonus points plan and portfolio

Most of the JD's bonus points can be earned inside this plan; competition results and published systems papers need time outside it.

| Direction | JD bonus point | How this plan earns it | Where |
| --- | --- | --- | --- |
| Acquisition | Math or informatics competition results | Not addressable in 6 months; algorithm practice on hashing, graphs and scheduling supports Phase 3 | outside plan |
| Acquisition | Large-scale crawling or pipeline experience | The Phase 3 crawler and its fault-injection results | 3 |
| Language | Curious about model behaviour; finds failure modes | Phase 6 ablations: which data decision changed which model behaviour | 6 |
| Language | Cross-team communication | Write-ups of each gate aimed at a training-team reader | all |
| Language | TB-token cleaning pipelines | Run the Phase 2 pipeline on at least one full Common Crawl WET segment set | 2, 6 |
| Multimodal | Distributed processing performance work | Phase 4 framework tuning applied to the Phase 5 pipeline | 4, 5 |
| Multimodal | High-concurrency, async programming | The Phase 3 asyncio fetcher | 3 |
| Multimodal | Megatron, vLLM familiarity | Use vLLM for caption or quality-label generation in Phase 5 | 5 |
| Infrastructure | Storage, KV, data lake, processing framework design | Phase 4 components with amplification and conflict tests | 4 |
| Infrastructure | Papers at systems conferences | Outside plan; a strong Phase 4 write-up is the starting point | outside plan |

**Portfolio checklist for the interview**

- [ ] One repo per phase with a benchmark table and the profile or plot behind each number.
- [ ] A single page listing every gate number with links to the runs.
- [ ] One ablation table linking data decisions to model quality.
- [ ] Two write-ups: one on dedup or filtering, one on your deep track.
- [ ] At least one upstream contribution (datatrove, NeMo Curator, Data-Juicer, DuckDB, Lance, or a crawler library).

## Weekly cadence, budget and risks

**Weekly cadence (25 hours).** About 6 hours reading papers and code, 14 hours building, 3 hours measuring and writing up numbers, 2 hours reviewing against the gate. If a gate slips more than a week, cut scope inside the phase; Phase 6 depends on Phases 2 and 3.

**Budget.** Expect roughly USD 520-560 for the whole plan at on-demand prices, much less than the training/inference roadmap because most work is CPU and storage. Hours are my estimates of what the gates need.

| Item | Assumption | Estimated cost (USD) |
| --- | --- | --- |
| CPU compute, Phases 2-6 | 300 hours of a 16-vCPU node at $0.714/hr on-demand ($0.23 spot) | 214 (70 on spot) |
| Object storage | 2 TB of raw and processed data for 4 months at $0.023/GB-month | 184 |
| GPU, embeddings / classifier / ablations | 40 H100-hours at $3-4 | 120-160 |
| Data egress | Keep processing in the bucket's region; first 100 GB/month free | \~0 |

Price basis: c7i.4xlarge (16 vCPU, 32 GiB) in us-east-1 lists at $0.714/hr on-demand and about $0.23/hr spot ([Holori](https://calculator.holori.com/aws/ec2/c7i.4xlarge/us-east-1)); S3 Standard is $0.023/GB-month for the first 50 TB, egress $0.09/GB after the first 100 GB ([Upstash](https://upstash.com/blog/aws-s3-pricing-explained-2026)); H100 rates as in the companion training/inference roadmap.

**Risks and mitigations**

- **Crawling responsibly.** Crawl only sites that allow it, follow RFC 9309, and do most Phase 3 work against a local simulated web or Common Crawl's index rather than live sites.
- **Licensing and PII.** Treat downloaded data as untrusted and possibly personal; keep raw data private and redact before sharing samples.
- **Spot interruptions.** Spot nodes are cheap but the listed interruption rate is above 20%; every job must be resumable (that is also a Phase 6 gate).
- **Measuring the wrong thing.** Removal rates are not quality. Every filter decision needs a model-side number from the Phase 6 harness.

## Sources

- Job description: "Pre-training Data Engineers" (provided).
- [FineWeb paper](https://arxiv.org/pdf/2406.17557)
- [DCLM abstract](https://arxiv.org/abs/2406.11794v1) and [paper PDF](https://export.arxiv.org/pdf/2406.11794)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499v1)
- [SemDeDup](https://arxiv.org/abs/2303.09540v1)
- [datatrove README](https://raw.githubusercontent.com/huggingface/datatrove/main/README.md)
- [Common Crawl August 2026 crawl](https://commoncrawl.org/blog/august-2026-crawl-archive-now-available)
- [RFC 9309 Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309)
- [Kafka delivery semantics (Confluent)](https://docs.confluent.io/kafka/design/delivery-semantics.md)
- [Apache Iceberg table spec](https://iceberg.apache.org/spec)
- [LAION-5B](https://arxiv.org/pdf/2210.08402)
- [DataComp](https://arxiv.org/abs/2304.14108v2)
- [Panda-70M](https://arxiv.org/abs/2402.19479)
- [c7i.4xlarge pricing (Holori)](https://calculator.holori.com/aws/ec2/c7i.4xlarge/us-east-1)
- [S3 pricing (Upstash)](https://upstash.com/blog/aws-s3-pricing-explained-2026)
