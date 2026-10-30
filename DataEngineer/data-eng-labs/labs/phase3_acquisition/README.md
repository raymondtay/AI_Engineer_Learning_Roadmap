# Phase 3 — Data acquisition (Weeks 10–13)

A crawler, built against `common/simweb.py`: a generated web with known page
quality per host, mirror sites, messy URL variants, robots.txt, crawl delays,
failing servers and Poisson change rates. Week 12 serves it over real HTTP on
127.0.0.1 with injected 5xx errors, hangs and dropped connections.

Reference numbers: 2-core CPU VM, `GATES_solution.md`.

## Week 10 — Link control (`w10_link_control`)

| metric | value |
| --- | --- |
| raw URLs → canonical | 3,960 → 1,320 (−67%) |
| true pages lost by normalisation | 0 |
| Bloom filter | 9.59 bits/URL at 1.16% measured FP (target 1%) |
| mirror hosts by SimHash | precision 1.0, recall 1.0 |

## Week 11 — robots.txt and frontier (`w11_frontier`)

RFC 9309 group selection, longest-match precedence (allow wins ties), `*` and
`$`, 500 KiB parse limit, 4xx = allow all, 5xx = disallow all, 24 h cache.
Gate trace: with `Crawl-delay: 2`, the slow host is fetched at t = 0, 2, 4
while the default-delay host runs at t = 0, 1, 2 — never two requests to one
host inside its delay. `/docs/a.pdf?x=1` is *allowed* because `$` anchors at
the end of path+query (a classic misreading).

## Week 12 — Fetcher (`w12_fetcher`)

asyncio + bounded concurrency + per-host politeness + DNS cache + retries
with backoff + checkpoint/resume, against a server that misbehaves on purpose:

| metric | value |
| --- | --- |
| reachable pages missing at the end | **0** (of 123) |
| requests / retries / permanent failures | 205 / 58 / 1 |
| DNS lookups | 11 for 11 hosts (cache works) |
| throughput | ~52 pages/s (politeness-bound, not CPU-bound) |

Bug that the tests caught while building it: a failed URL was marked done
after its retries ran out, so restarts never revisited it.

## Week 13 — Scheduling (`w13_scheduling`)

Same 300-fetch budget:

| policy | kept tokens / fetch |
| --- | --- |
| BFS | 151 |
| OPIC (link importance) | 149 |
| yield-UCB (cleaning feedback per host) | **205** (+36%) |

Link importance does not predict quality here; feedback from the cleaning
pipeline does. Recrawl for freshness (300 pages, 30 refreshes/day):

| allocation | mean freshness |
| --- | --- |
| proportional to change rate | 0.064 |
| uniform | 0.389 |
| optimal (water-filling) | 0.424 |

Cho & Garcia-Molina's counter-intuitive result reproduced: chasing fast-changing
pages wastes budget; the optimum gives up on pages that change too fast to catch.

## Caveats

The simulated web is small and its quality is per-host by construction, which
flatters host-level bandits. Real yield varies within hosts and drifts over time.
Politeness numbers are from a local server; the network is not.

## Sources

* RFC 9309, "Robots Exclusion Protocol", IETF 2022.
* RFC 3986, "URI Generic Syntax" (normalisation, dot-segment removal).
* Bloom 1970; Charikar, "Similarity estimation techniques from rounding algorithms", STOC 2002 (SimHash); Manku et al., "Detecting near-duplicates for web crawling", WWW 2007.
* Abiteboul, Preda, Cobena, "Adaptive On-Line Page Importance Computation", WWW 2003.
* Cho & Garcia-Molina, "Effective page refresh policies for web crawlers", ACM TODS 2003.
* Auer, Cesa-Bianchi, Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", 2002 (UCB1).
