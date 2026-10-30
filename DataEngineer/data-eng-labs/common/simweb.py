"""A simulated web for the acquisition labs (Weeks 10-13).

Real crawling is slow, rude to practise on, and has no ground truth. This web
is generated with known properties:

  * hosts with different *yield*: the share of pages that survive cleaning
  * mirror hosts that copy another host's pages
  * URL variants: tracking parameters, session ids, fragments, case, default ports
  * robots.txt per host (some disallow paths, some have crawl delays)
  * failure behaviour per host (5xx rate, slow responses)
  * a change rate per page (Poisson), for freshness experiments

`SimWeb.fetch(url)` serves it in-process; Week 12 also serves it over HTTP.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

from common.webgen import good_doc, spam_doc, boilerplate_doc

TRACKING = ["utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid", "sessionid"]


@dataclass
class Page:
    url: str
    text: str
    good: bool
    links: list[str] = field(default_factory=list)
    change_rate: float = 0.1  # expected changes per day


@dataclass
class Host:
    name: str
    yield_: float           # fraction of good pages
    crawl_delay: float = 1.0
    disallow: list[str] = field(default_factory=list)
    fail_rate: float = 0.0
    mirror_of: str | None = None


class SimWeb:
    def __init__(self, n_hosts: int = 40, pages_per_host: int = 30, n_mirrors: int = 4, seed: int = 0):
        rnd = random.Random(seed)
        self.rnd = rnd
        self.hosts: dict[str, Host] = {}
        self.pages: dict[str, Page] = {}
        names = [f"site{i:02d}.example.com" for i in range(n_hosts)]
        for i, h in enumerate(names):
            y = rnd.choice([0.05, 0.2, 0.5, 0.8, 0.95])
            self.hosts[h] = Host(h, y, crawl_delay=rnd.choice([0.5, 1.0, 2.0]),
                                 disallow=["/private/"] if rnd.random() < 0.3 else [],
                                 fail_rate=rnd.choice([0.0, 0.0, 0.1, 0.3]))
            for p in range(pages_per_host):
                path = f"/private/p{p}" if (self.hosts[h].disallow and p % 7 == 0) else f"/p{p}"
                good = rnd.random() < y
                text = good_doc(rnd) if good else rnd.choice([spam_doc, boilerplate_doc])(rnd)
                self.pages[f"https://{h}{path}"] = Page(f"https://{h}{path}", text, good,
                                                       change_rate=rnd.choice([0.01, 0.1, 1.0, 5.0]))
        # mirrors: copy every page of a source host under a new name
        for m in range(n_mirrors):
            src = names[m]
            mh = f"mirror{m}.example.net"
            self.hosts[mh] = Host(mh, self.hosts[src].yield_, mirror_of=src)
            for url, pg in list(self.pages.items()):
                if urlsplit(url).hostname == src:
                    nu = url.replace(src, mh)
                    self.pages[nu] = Page(nu, pg.text, pg.good, change_rate=pg.change_rate)
        # links: mostly within host, some across hosts
        urls = list(self.pages)
        by_host: dict[str, list[str]] = {}
        for u in urls:
            by_host.setdefault(urlsplit(u).hostname, []).append(u)
        for u, pg in self.pages.items():
            same = by_host[urlsplit(u).hostname]
            pg.links = rnd.sample(same, min(4, len(same))) + rnd.sample(urls, 2)

    # ---------------------------------------------------------------- views
    def seeds(self, k: int = 5) -> list[str]:
        """First allowed, existing page of the first k hosts."""
        out = []
        for h in sorted(self.hosts)[:k]:
            out.append(next(u for u in self.pages if urlsplit(u).hostname == h and "/private/" not in u))
        return out

    def robots_txt(self, host: str) -> str:
        h = self.hosts[host]
        lines = ["User-agent: *"] + [f"Disallow: {d}" for d in h.disallow] + [f"Crawl-delay: {h.crawl_delay}"]
        return "\n".join(lines) + "\n"

    def variant(self, url: str, rnd: random.Random | None = None) -> str:
        """A messy but equivalent spelling of url."""
        rnd = rnd or self.rnd
        sp = urlsplit(url)
        host = sp.hostname
        host = host.upper() if rnd.random() < 0.3 else host
        port = ":443" if rnd.random() < 0.3 else ""
        q = {k: str(rnd.randint(1, 9999)) for k in rnd.sample(TRACKING, rnd.randint(0, 3))}
        frag = f"#section{rnd.randint(1, 5)}" if rnd.random() < 0.3 else ""
        path = sp.path.replace("/p", "/./p") if rnd.random() < 0.2 else sp.path
        return f"https://{host}{port}{path}{'?' + urlencode(q) if q else ''}{frag}"

    def fetch(self, url: str) -> tuple[int, str, list[str]]:
        """(status, text, links) for a canonical URL; 404 if unknown."""
        pg = self.pages.get(url)
        if pg is None:
            return 404, "", []
        h = self.hosts[urlsplit(url).hostname]
        if self.rnd.random() < h.fail_rate:
            return 503, "", []
        return 200, pg.text, list(pg.links)

    def kept_tokens(self, url: str) -> int:
        """Ground-truth cleaning feedback: tokens a Phase 2 pipeline would keep."""
        pg = self.pages.get(url)
        return len(pg.text.split()) if pg and pg.good else 0
