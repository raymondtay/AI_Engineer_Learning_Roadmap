# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 11: robots.txt (RFC 9309) and a polite, prioritised frontier.

Roadmap: "Per-host queues, crawl delay, robots.txt parsing and caching,
priorities and per-site quotas."

RFC 9309 rules implemented and tested here:
  * groups start with one or more "user-agent:" lines; match the most specific
    product token (case-insensitive), else "*"
  * the most specific (longest) matching allow/disallow rule wins; on a tie,
    allow wins; no matching rule means allowed
  * "*" matches any sequence of characters, "$" anchors the end
  * crawlers SHOULD NOT use a cached robots.txt for more than 24 hours unless it
    is unreachable; the parsing limit MUST be at least 500 KiB
  * 4xx ("unavailable") -> may access anything; 5xx / network error
    ("unreachable") -> MUST assume complete disallow
("Crawl-delay" is not in RFC 9309, but many sites use it; we honour it.)

Frontier: one FIFO-with-priority queue per host plus a heap of hosts keyed by
the earliest time each may be fetched again. `pop(now)` returns the best URL
from a host that is ready, never two from the same host closer than its delay.
"""
from __future__ import annotations

import heapq
import pathlib
import re
import sys
from dataclasses import dataclass, field
from urllib.parse import urlsplit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

PARSE_LIMIT = 500 * 1024
CACHE_TTL = 24 * 3600


@dataclass
class Rules:
    rules: list[tuple[bool, str]] = field(default_factory=list)  # (allow?, pattern)
    crawl_delay: float | None = None
    allow_all: bool = False
    disallow_all: bool = False


def parse_robots(text: str, agent: str) -> Rules:
    """Rules of the group that applies to `agent` (product token, e.g. 'DataBot')."""
    text = text.encode()[:PARSE_LIMIT].decode(errors="ignore")
    groups: list[tuple[list[str], Rules]] = []
    cur_agents: list[str] = []
    cur: Rules | None = None
    last_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, val = (x.strip() for x in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if not last_was_agent:
                cur_agents, cur = [], Rules()
                groups.append((cur_agents, cur))
            cur_agents.append(val.lower())
            last_was_agent = True
            continue
        last_was_agent = False
        if cur is None:
            continue
        if key in ("allow", "disallow") and val:
            cur.rules.append((key == "allow", val))
        elif key == "crawl-delay":
            try:
                cur.crawl_delay = float(val)
            except ValueError:
                pass
    # TODO: pick the group whose agents contain agent.lower(); else the '*' group; else empty Rules (allow all)
    raise NotImplementedError("TODO(w11_frontier): pick the group whose agents contain agent.lower(); else the '*' group; else empty Rules (allow all)")


def _pattern_regex(pat: str) -> re.Pattern:
    anchored = pat.endswith("$")
    body = re.escape(pat[:-1] if anchored else pat).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


def is_allowed(rules: Rules, path: str) -> bool:
    if rules.disallow_all:
        return False
    if rules.allow_all:
        return True
    best: tuple[int, bool] | None = None
    # TODO: for each rule whose pattern regex matches path from the start, score = len(pattern); keep the longest; on equal length allow beats disallow
    raise NotImplementedError("TODO(w11_frontier): for each rule whose pattern regex matches path from the start, score = len(pattern); keep the longest; on equal length allow beats disallow")


class RobotsCache:
    """robots.txt per host with RFC 9309 status handling and a 24-hour TTL."""

    def __init__(self, fetcher, agent: str = "DataBot"):
        self.fetcher = fetcher          # fetcher(host) -> (status, text); status 0 = network error
        self.agent = agent
        self.cache: dict[str, tuple[float, Rules]] = {}
        self.fetches = 0

    def rules(self, host: str, now: float) -> Rules:
        hit = self.cache.get(host)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
        self.fetches += 1
        status, text = self.fetcher(host)
        # TODO: 2xx -> parse; 4xx -> allow_all; 5xx or 0 -> disallow_all, but if a stale cached copy exists keep using it (unreachable exception to the 24h rule)
        raise NotImplementedError("TODO(w11_frontier): 2xx -> parse; 4xx -> allow_all; 5xx or 0 -> disallow_all, but if a stale cached copy exists keep using it (unreachable exception to the 24h rule)")
        self.cache[host] = (now, r)
        return r


class Frontier:
    def __init__(self, default_delay: float = 1.0, quota: int | None = None):
        self.q: dict[str, list[tuple[float, int, str]]] = {}     # host -> heap of (-priority, seq, url)
        self.ready: list[tuple[float, str]] = []                  # heap of (next_allowed_time, host)
        self.next_ok: dict[str, float] = {}
        self.delay: dict[str, float] = {}
        self.taken: dict[str, int] = {}
        self.default_delay, self.quota = default_delay, quota
        self.seq = 0

    def set_delay(self, host: str, d: float):
        self.delay[host] = d

    def push(self, url: str, priority: float = 0.0):
        host = urlsplit(url).hostname
        if self.quota is not None and self.taken.get(host, 0) >= self.quota:
            return
        # TODO: push (-priority, seq, url) on the host heap; if the host has no queue yet, schedule it in `ready` at its next_ok time (default 0)
        raise NotImplementedError("TODO(w11_frontier): push (-priority, seq, url) on the host heap; if the host has no queue yet, schedule it in `ready` at its next_ok time (default 0)")

    def pop(self, now: float) -> str | None:
        """Best URL from a host allowed to be fetched at `now`, or None."""
        # TODO: if the earliest ready time > now return None; pop host, pop its best URL, set next_ok = now + delay, count quota; re-push host if its queue is non-empty, else delete the queue
        raise NotImplementedError("TODO(w11_frontier): if the earliest ready time > now return None; pop host, pop its best URL, set next_ok = now + delay, count quota; re-push host if its queue is non-empty, else delete the queue")

    def next_time(self) -> float | None:
        return self.ready[0][0] if self.ready else None

    def __len__(self):
        return sum(len(v) for v in self.q.values())


def report() -> str:
    """Gate: RFC 9309 precedence on a tricky robots.txt, and a politeness trace."""
    from common.bench import table

    robots = ("User-agent: *\nDisallow: /\n\nUser-agent: DataBot\nUser-agent: OtherBot\n"
              "Disallow: /private\nAllow: /private/public$\nDisallow: /*.pdf$\nAllow: /docs/*.pdf$\nCrawl-delay: 2\n")
    r = parse_robots(robots, "DataBot")   # the product token, not the full UA string
    paths = ["/", "/private/x", "/private/public", "/private/public/more", "/a/b.pdf", "/docs/a.pdf", "/docs/a.pdf?x=1"]
    rows = [{"path": p, "allowed for DataBot": is_allowed(r, p),
             "allowed for any other bot": is_allowed(parse_robots(robots, "SomeBot"), p)} for p in paths]
    f = Frontier(default_delay=1.0)
    f.set_delay("slow.example.com", r.crawl_delay or 1.0)
    for i in range(3):
        f.push(f"https://slow.example.com/{i}", priority=i)
        f.push(f"https://fast.example.com/{i}")
    trace, t = [], 0.0
    while len(f):
        u = f.pop(t)
        if u is None:
            t = f.next_time()
            continue
        trace.append(f"t={t:.0f} {u.split('//')[1]}")
    return (table(rows, ["path", "allowed for DataBot", "allowed for any other bot"]) +
            f"\n\ncrawl-delay {r.crawl_delay}; fetch order: " + ", ".join(trace))


if __name__ == "__main__":
    print(report())
