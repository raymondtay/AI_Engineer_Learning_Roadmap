"""Week 13: selection from cleaning feedback, and freshness for RAG.

Roadmap: "Score sites by the yield of kept tokens from Phase 2; link-graph
priority (OPIC/PageRank); change-rate-based recrawl for RAG seeds."
Gate: yield-ranked vs breadth-first on the same budget (kept tokens per fetch),
and staleness vs budget for a freshness set.

Part A - which page next? Three policies on the same fetch budget:
  bfs        first discovered, first fetched
  opic       On-line Page Importance Computation (Abiteboul et al. 2003): each
             fetched page splits its "cash" among its out-links; fetch the URL
             with most accumulated cash. Importance, not quality.
  yield_ucb  the JD's "data selection based on data cleaning feedback signals":
             every fetched page reports how many tokens survived cleaning;
             pick the host with the best upper confidence bound on yield
             (mean + c * sqrt(ln t / n)), so new hosts still get explored.

Part B - how often to recrawl? N pages change as Poisson processes with rates
lambda_i; you can refresh R pages per day. Fetching page i every I_i days keeps
it fresh a fraction F_i = (1 - exp(-lambda_i I_i)) / (lambda_i I_i) of the time.
Cho & Garcia-Molina (2003) showed that refreshing every page equally often
beats refreshing in proportion to change rate, and that the optimum skips the
pages that change too fast to ever catch. You will reproduce all three.
"""
from __future__ import annotations

import collections
import heapq
import math
import pathlib
import random
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.simweb import SimWeb


# ------------------------------------------------------------------ A. crawl policies

def crawl(web: SimWeb, policy: str, budget: int, c: float = 2.0, seed: int = 0) -> dict:
    seeds = web.seeds(len(web.hosts))[:8]
    seen = set(seeds)
    fetched, kept = [], 0
    if policy == "bfs":
        q = collections.deque(seeds)
        while q and len(fetched) < budget:
            u = q.popleft()
            fetched.append(u)
            kept += web.kept_tokens(u)
            for l in web.pages[u].links:
                if l not in seen and l in web.pages:
                    seen.add(l)
                    q.append(l)
    elif policy == "opic":
        cash = collections.defaultdict(float, {s: 1.0 for s in seeds})
        frontier = set(seeds)
        while frontier and len(fetched) < budget:
            # >>> SOLUTION: fetch the frontier URL with most cash; give its cash equally to its out-links (adding new ones to the frontier); zero its cash
            u = max(frontier, key=lambda x: (cash[x], x))
            frontier.remove(u)
            fetched.append(u)
            kept += web.kept_tokens(u)
            links = [l for l in web.pages[u].links if l in web.pages]
            share = cash[u] / max(1, len(links))
            for l in links:
                cash[l] += share
                if l not in seen:
                    seen.add(l)
                    frontier.add(l)
            cash[u] = 0.0
            # <<< SOLUTION
    elif policy == "yield_ucb":
        queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        for s in seeds:
            queues[urlsplit(s).hostname].append(s)
        n = collections.Counter()
        total = collections.Counter()
        t = 0
        while any(queues.values()) and len(fetched) < budget:
            t += 1
            # >>> SOLUTION: among hosts with queued URLs pick max UCB (unvisited hosts first: infinite score); fetch its next URL; update n[host], total[host] with kept tokens; enqueue new links per host
            def ucb(h):
                if n[h] == 0:
                    return float("inf")
                return total[h] / n[h] + c * 100 * math.sqrt(math.log(t) / n[h])
            h = max((h for h, q in queues.items() if q), key=lambda h: (ucb(h), h))
            u = queues[h].popleft()
            fetched.append(u)
            k = web.kept_tokens(u)
            kept += k
            n[h] += 1
            total[h] += k
            for l in web.pages[u].links:
                if l not in seen and l in web.pages:
                    seen.add(l)
                    queues[urlsplit(l).hostname].append(l)
            # <<< SOLUTION
    else:
        raise ValueError(policy)
    return {"policy": policy, "fetches": len(fetched), "kept tokens": kept, "kept/fetch": kept / max(1, len(fetched))}


# ------------------------------------------------------------------ B. freshness

def freshness(lam: float, interval: float) -> float:
    """Expected fraction of time a page is fresh when refreshed every `interval` days."""
    # >>> SOLUTION: (1 - exp(-lam*I)) / (lam*I); an interval of infinity (never refreshed) gives 0
    if interval == math.inf:
        return 0.0
    x = lam * interval
    return 1.0 if x == 0 else (1 - math.exp(-x)) / x
    # <<< SOLUTION


def allocate(rates: list[float], budget: float, policy: str) -> list[float]:
    """Refresh frequencies f_i (per day) with sum f_i = budget."""
    n = len(rates)
    if policy == "uniform":
        return [budget / n] * n
    if policy == "proportional":
        s = sum(rates)
        return [budget * r / s for r in rates]
    if policy == "optimal":
        # >>> SOLUTION: greedy water-filling (F is concave in f): split budget into small steps, each time give a step to the page whose freshness rises most
        steps = 2000
        df = budget / steps
        f = [0.0] * n
        gain = lambda i: freshness(rates[i], 1 / (f[i] + df)) - (freshness(rates[i], 1 / f[i]) if f[i] > 0 else 0.0)
        heap = [(-gain(i), i) for i in range(n)]
        heapq.heapify(heap)
        for _ in range(steps):
            _, i = heapq.heappop(heap)
            f[i] += df
            heapq.heappush(heap, (-gain(i), i))
        return f
        # <<< SOLUTION
    raise ValueError(policy)


def mean_freshness(rates: list[float], freqs: list[float]) -> float:
    return sum(freshness(r, 1 / f if f > 0 else math.inf) for r, f in zip(rates, freqs)) / len(rates)


def simulate_freshness(lam: float, interval: float, days: float = 2000, seed: int = 0) -> float:
    """Monte Carlo check of the closed form: fraction of time the copy matches the page."""
    rnd = random.Random(seed)
    t, fresh_time = 0.0, 0.0
    while t < days:
        first_change = rnd.expovariate(lam)
        fresh_time += min(first_change, interval)
        t += interval
    return fresh_time / (days if t >= days else t)


def report() -> str:
    web = SimWeb(n_hosts=40, pages_per_host=30, n_mirrors=0, seed=0)
    rows = [crawl(web, p, budget=300) for p in ("bfs", "opic", "yield_ucb")]
    rates = [pg.change_rate for pg in web.pages.values()][:300]
    fr = [{"policy": p, "mean freshness": mean_freshness(rates, allocate(rates, 30.0, p))}
          for p in ("proportional", "uniform", "optimal")]
    return (table(rows, ["policy", "fetches", "kept tokens", "kept/fetch"]) + "\n\n" +
            "300 pages, 30 refreshes/day:\n" + table(fr, ["policy", "mean freshness"]))


if __name__ == "__main__":
    print(report())
