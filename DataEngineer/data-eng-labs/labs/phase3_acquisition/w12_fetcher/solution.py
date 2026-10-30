"""Week 12: the fetch loop - asyncio, DNS cache, retries, checkpoint and resume.

Roadmap: "asyncio fetcher with DNS cache, retries with backoff, timeouts, and a
checkpointed frontier; inject failures and measure recovery."

A real HTTP server runs on 127.0.0.1 and serves the simulated web. Every host
name resolves (through a slow fake DNS) to that server; the Host header picks
the site. The server misbehaves on purpose: per-host 5xx rates, requests that
hang past the client timeout, and connections dropped mid-request.

The fetcher must still collect every reachable, robots-allowed page:
  * bounded concurrency (semaphore) and per-host politeness
  * DNS results cached with a TTL (lookups are latency on every new host)
  * retries with exponential backoff and jitter on 5xx, timeouts and resets
  * a checkpoint written atomically (temp file + os.replace) so a crash loses
    at most the in-flight work, and a resume never refetches finished URLs
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import random
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.simweb import SimWeb


# ------------------------------------------------------------------ server (given)

class FaultyServer:
    def __init__(self, web: SimWeb, p_hang: float = 0.05, p_drop: float = 0.05, seed: int = 0):
        self.web, self.p_hang, self.p_drop = web, p_hang, p_drop
        self.rnd = random.Random(seed)
        self.requests = 0

    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except Exception:
            writer.close()
            return
        self.requests += 1
        lines = head.decode().split("\r\n")
        path = lines[0].split(" ")[1]
        host = next(l.split(":", 1)[1].strip() for l in lines[1:] if l.lower().startswith("host:"))
        r = self.rnd.random()
        if r < self.p_drop:
            writer.close()                              # connection reset
            return
        if r < self.p_drop + self.p_hang:
            await asyncio.sleep(1.0)                    # longer than the client timeout
            writer.close()
            return
        if path == "/robots.txt":
            status, body = 200, self.web.robots_txt(host)
        else:
            status, text, links = self.web.fetch(f"https://{host}{path}")
            body = json.dumps({"text": text, "links": links}) if status == 200 else ""
        data = body.encode()
        writer.write(f"HTTP/1.1 {status} X\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
        try:
            await writer.drain()
        finally:
            writer.close()

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self


class FakeDNS:
    """Every name resolves to 127.0.0.1 after a delay; counts lookups."""

    def __init__(self, latency: float = 0.02):
        self.latency, self.lookups = latency, 0

    async def resolve(self, host: str) -> str:
        self.lookups += 1
        await asyncio.sleep(self.latency)
        return "127.0.0.1"


# ------------------------------------------------------------------ fetcher

class Fetcher:
    def __init__(self, port: int, dns: FakeDNS, concurrency: int = 32, timeout: float = 0.3,
                 max_retries: int = 4, host_interval: float = 0.0, dns_ttl: float = 300.0, seed: int = 0):
        self.port, self.dns = port, dns
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout, self.max_retries, self.host_interval, self.dns_ttl = timeout, max_retries, host_interval, dns_ttl
        self.dns_cache: dict[str, tuple[float, str]] = {}
        self.dns_inflight: dict[str, asyncio.Future] = {}
        self.host_next: dict[str, float] = {}
        self.host_lock: dict[str, asyncio.Lock] = {}
        self.rnd = random.Random(seed)
        self.stats = {"requests": 0, "retries": 0, "failures": 0}

    async def resolve(self, host: str) -> str:
        # >>> SOLUTION: return cached IP if younger than dns_ttl; coalesce concurrent lookups of the same host through one future; store the result
        now = time.monotonic()
        hit = self.dns_cache.get(host)
        if hit and now - hit[0] < self.dns_ttl:
            return hit[1]
        if host in self.dns_inflight:
            return await self.dns_inflight[host]
        fut = asyncio.get_running_loop().create_future()
        self.dns_inflight[host] = fut
        try:
            ip = await self.dns.resolve(host)
            self.dns_cache[host] = (time.monotonic(), ip)
            fut.set_result(ip)
            return ip
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            del self.dns_inflight[host]
        # <<< SOLUTION

    async def _polite(self, host: str):
        lock = self.host_lock.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self.host_next.get(host, 0) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.host_next[host] = time.monotonic() + self.host_interval

    async def get_once(self, url: str) -> tuple[int, bytes]:
        sp = urlsplit(url)
        ip = await self.resolve(sp.hostname)
        await self._polite(sp.hostname)
        self.stats["requests"] += 1

        async def go():
            reader, writer = await asyncio.open_connection(ip, self.port)
            writer.write(f"GET {sp.path or '/'} HTTP/1.1\r\nHost: {sp.hostname}\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
            raw = await reader.read()
            writer.close()
            if not raw:
                raise ConnectionResetError("empty response")
            head, _, body = raw.partition(b"\r\n\r\n")
            return int(head.split(b" ")[1]), body

        return await asyncio.wait_for(go(), self.timeout)

    async def get(self, url: str) -> tuple[int, bytes]:
        """GET with retries: 5xx, timeouts and connection errors are retried with backoff."""
        async with self.sem:
            # >>> SOLUTION: up to max_retries+1 attempts; on 5xx / TimeoutError / OSError sleep 0.02 * 2**attempt * (1 + jitter) and retry; return (status, body) on success or non-retryable status; (0, b'') when exhausted
            for attempt in range(self.max_retries + 1):
                try:
                    status, body = await self.get_once(url)
                    if status < 500:
                        return status, body
                except (asyncio.TimeoutError, OSError, ConnectionError, IndexError, ValueError):
                    pass
                if attempt < self.max_retries:
                    self.stats["retries"] += 1
                    await asyncio.sleep(0.02 * 2**attempt * (1 + self.rnd.random()))
            self.stats["failures"] += 1
            return 0, b""
            # <<< SOLUTION


# ------------------------------------------------------------------ crawl with checkpoints

def save_checkpoint(path: str, state: dict):
    # >>> SOLUTION: write JSON to path + '.tmp', flush + fsync, then os.replace onto path (atomic on POSIX)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # <<< SOLUTION


async def crawl(fetcher: Fetcher, seeds: list[str], ckpt: str, stop_after: int | None = None,
                every: int = 20) -> dict:
    """Crawl until the frontier is empty (or stop_after pages, simulating a crash)."""
    if os.path.exists(ckpt):
        st = json.load(open(ckpt))
        done, pending, seen, failed, ok = st["done"], st["pending"], set(st["seen"]), st.get("failed", {}), st["ok"]
    else:
        done, pending, seen, failed, ok = [], list(seeds), set(seeds), {}, []
    robots: dict[str, list[str]] = {}
    fetched_now: list[str] = []

    async def disallowed(url):
        h, path = urlsplit(url).hostname, urlsplit(url).path
        if h not in robots:
            s, body = await fetcher.get(f"https://{h}/robots.txt")
            text = body.decode() if s == 200 else ""
            robots[h] = [l.split(":", 1)[1].strip() for l in text.splitlines() if l.lower().startswith("disallow:")]
            if s == 0 or s >= 500:
                robots[h] = ["/"]                               # unreachable: complete disallow
        return any(d and path.startswith(d) for d in robots[h])

    async def one(url):
        if await disallowed(url):
            return url, "disallowed", None
        s, body = await fetcher.get(url)
        if s == 200:
            return url, "ok", json.loads(body)
        return url, ("failed" if s == 0 or s >= 500 else "gone"), None

    since_ckpt = 0
    while pending:
        if stop_after is not None and len(fetched_now) >= stop_after:
            break
        batch, pending = pending[:64], pending[64:]
        for url, kind, page in await asyncio.gather(*(one(u) for u in batch)):
            fetched_now.append(url)
            if kind == "failed" and failed.get(url, 0) < 2:   # exhausted retries: one more later pass
                failed[url] = failed.get(url, 0) + 1
                pending.append(url)
                continue
            done.append(url)
            if kind == "ok":
                ok.append(url)
            if page:
                for l in page["links"]:
                    if l not in seen:
                        seen.add(l)
                        pending.append(l)
        since_ckpt += len(batch)
        if since_ckpt >= every:
            save_checkpoint(ckpt, {"done": done, "pending": pending, "seen": sorted(seen), "failed": failed, "ok": ok})
            since_ckpt = 0
    save_checkpoint(ckpt, {"done": done, "pending": pending, "seen": sorted(seen), "failed": failed, "ok": ok})
    return {"done": done, "ok": ok, "fetched_now": fetched_now, "pending": pending}


def reachable(web: SimWeb, seeds: list[str]) -> set[str]:
    """Ground truth: pages reachable from seeds through links, excluding robots-disallowed paths."""
    out, stack = set(), list(seeds)
    while stack:
        u = stack.pop()
        if u in out or u not in web.pages:
            continue
        h = web.hosts[urlsplit(u).hostname]
        if any(urlsplit(u).path.startswith(d) for d in h.disallow):
            continue
        out.add(u)
        stack.extend(web.pages[u].links)
    return out


async def run(seed: int = 0, crash_after: int | None = None, ckpt_dir: str | None = None, **kw) -> dict:
    import tempfile

    web = SimWeb(n_hosts=10, pages_per_host=12, n_mirrors=1, seed=seed)
    for h in web.hosts.values():
        h.fail_rate = min(h.fail_rate, 0.3)
    server = await FaultyServer(web, seed=seed).start()
    dns = FakeDNS()
    ckpt = os.path.join(ckpt_dir or tempfile.mkdtemp(), "crawl.json")
    seeds = web.seeds(3)
    t0 = time.perf_counter()
    async with server.server:
        f = Fetcher(server.port, dns, **kw)
        first = await crawl(f, seeds, ckpt, stop_after=crash_after)
        second = await crawl(Fetcher(server.port, dns, **kw), seeds, ckpt) if crash_after else None
    elapsed = time.perf_counter() - t0
    final = (second or first)["done"]
    ok_pages = (second or first)["ok"]
    return {"truth": reachable(web, seeds), "done": final, "ok": ok_pages, "first": first, "second": second, "dns": dns.lookups,
            "hosts": len({urlsplit(u).hostname for u in final}), "stats": f.stats, "seconds": elapsed,
            "pages/s": len(ok_pages) / elapsed}


def _demo():
    r = asyncio.run(run())
    got = set(r["done"])
    print(f"reachable {len(r['truth'])}, attempted {len(got)}, missing {len(r['truth'] - got)}, "
          f"pages/s {r['pages/s']:.0f}, DNS lookups {r['dns']} for {r['hosts']} hosts, {r['stats']}")


def report() -> str:
    """The gate measurement as text (used by tools/gates.py)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _demo()
    return buf.getvalue()


if __name__ == "__main__":
    print(report())
