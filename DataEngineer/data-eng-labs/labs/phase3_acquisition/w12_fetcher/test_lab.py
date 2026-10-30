import asyncio
import collections
import tempfile
from urllib.parse import urlsplit


def test_collects_everything_despite_faults(lab):
    r = asyncio.run(lab.run(seed=1))
    got = set(r["ok"])
    assert r["truth"] <= got, f"missing {len(r['truth'] - got)}"
    assert not any("/private/" in urlsplit(u).path for u in got)      # robots respected: never fetched
    assert r["stats"]["retries"] > 0                                    # faults really happened
    assert r["dns"] == r["hosts"]                                       # one lookup per host


def test_resume_after_crash_without_refetch(lab):
    d = tempfile.mkdtemp()
    r = asyncio.run(lab.run(seed=2, crash_after=60, ckpt_dir=d))
    first, second = r["first"]["fetched_now"], r["second"]["fetched_now"]
    assert len(first) >= 60 and r["second"]["pending"] == []
    done_first = set(r["first"]["done"])
    assert done_first.isdisjoint(second)                                # nothing finished is refetched
    assert r["truth"] <= set(r["ok"])
    assert max(collections.Counter(r["ok"]).values()) == 1


def test_dns_coalescing(lab):
    async def go():
        dns = lab.FakeDNS(latency=0.05)
        f = lab.Fetcher(1, dns)
        await asyncio.gather(*(f.resolve("a.example.com") for _ in range(20)))
        return dns.lookups
    assert asyncio.run(go()) == 1
