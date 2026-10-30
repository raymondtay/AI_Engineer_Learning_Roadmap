ROBOTS = """
User-agent: DataBot
User-agent: OtherBot
Disallow: /private/
Allow: /private/public-page
Disallow: /*.pdf$
Crawl-delay: 2

User-agent: *
Disallow: /
"""


def test_group_selection(lab):
    r = lab.parse_robots(ROBOTS, "databot")
    assert r.crawl_delay == 2.0
    assert lab.is_allowed(lab.parse_robots(ROBOTS, "SomeoneElse"), "/anything") is False


def test_longest_match_and_wildcards(lab):
    r = lab.parse_robots(ROBOTS, "DataBot")
    assert lab.is_allowed(r, "/private/x") is False
    assert lab.is_allowed(r, "/private/public-page") is True      # longer allow wins
    assert lab.is_allowed(r, "/docs/a.pdf") is False               # * and $ anchor
    assert lab.is_allowed(r, "/docs/a.pdf?x=1") is True            # $ means end of path
    assert lab.is_allowed(r, "/open") is True                      # no rule -> allowed


def test_tie_goes_to_allow(lab):
    r = lab.parse_robots("User-agent: *\nDisallow: /a\nAllow: /a\n", "x")
    assert lab.is_allowed(r, "/a") is True


def test_status_handling_and_ttl(lab):
    responses = {"ok.com": (200, "User-agent: *\nDisallow: /x\n"), "gone.com": (404, ""), "down.com": (503, "")}
    cache = lab.RobotsCache(lambda h: responses[h])
    assert lab.is_allowed(cache.rules("gone.com", 0), "/x") is True      # 4xx: allow all
    assert lab.is_allowed(cache.rules("down.com", 0), "/y") is False     # 5xx: disallow all
    cache.rules("ok.com", 0)
    cache.rules("ok.com", 23 * 3600)
    assert cache.fetches == 3                                            # cached within 24 h
    responses["ok.com"] = (503, "")
    r = cache.rules("ok.com", 25 * 3600)                                 # stale, refetch fails
    assert lab.is_allowed(r, "/y") is True and cache.fetches == 4        # keep stale copy when unreachable


def test_parse_limit(lab):
    big = "User-agent: *\n" + "# filler\n" * 70000 + "Disallow: /late\n"
    assert len(big.encode()) > lab.PARSE_LIMIT
    assert lab.is_allowed(lab.parse_robots(big, "x"), "/late") is True  # beyond the limit: not parsed


def test_frontier_politeness_and_priority(lab):
    f = lab.Frontier(default_delay=1.0)
    f.set_delay("a.com", 5.0)
    for i in range(3):
        f.push(f"https://a.com/{i}", priority=i)
    f.push("https://b.com/0")
    got = []
    t = 0.0
    while len(f):
        u = f.pop(t)
        if u is None:
            t = f.next_time()
            continue
        got.append((t, u))
    a_times = [t for t, u in got if "a.com" in u]
    assert all(b - a >= 5.0 for a, b in zip(a_times, a_times[1:]))
    assert [u for _, u in got if "a.com" in u] == ["https://a.com/2", "https://a.com/1", "https://a.com/0"]


def test_quota(lab):
    f = lab.Frontier(default_delay=0.0, quota=2)
    for i in range(5):
        f.push(f"https://a.com/{i}")
    out = [f.pop(i) for i in range(5)]
    assert sum(u is not None for u in out) == 2
