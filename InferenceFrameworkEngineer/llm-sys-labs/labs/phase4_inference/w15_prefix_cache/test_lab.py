def test_block_keys_are_prefix_sensitive(lab):
    c = lab.BlockHashCache(4, 100)
    a = c.keys([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = c.keys([0, 2, 3, 4, 5, 6, 7, 8])
    assert len(a) == 2 and len(b) == 2
    assert a[1] != b[1]  # same second block, different history -> different key


def test_block_match_full_blocks_only(lab):
    c = lab.BlockHashCache(4, 100)
    c.insert([1, 2, 3, 4, 5, 6, 7])
    assert c.match([1, 2, 3, 4, 5, 6, 7, 8]) == 4  # second block was never full


def test_radix_split_and_match(lab):
    r = lab.RadixCache(1000)
    r.insert([1, 2, 3, 4, 5])
    r.insert([1, 2, 3, 9, 9])
    assert r.match([1, 2, 3, 4, 5, 6]) == 5
    assert r.match([1, 2, 3, 9]) == 4
    assert r.match([1, 2, 7]) == 2
    assert r.size == 7  # 1,2,3 shared + 4,5 + 9,9


def test_radix_lru_eviction(lab):
    r = lab.RadixCache(8)
    r.insert([1, 2, 3, 4])
    r.insert([5, 6, 7, 8])
    r.match([1, 2, 3, 4])         # touch the first
    r.insert([9, 9, 9, 9])        # over capacity: evict the least recently used leaf
    assert r.match([5, 6, 7, 8]) == 0 and r.match([1, 2, 3, 4]) == 4


def test_trace_comparison(lab):
    rows = {r["trace"]: r for r in lab.compare()}
    assert rows["multi-turn"]["radix saved"] > 0.6
    assert rows["rag"]["radix saved"] > 0.5
    assert rows["no-reuse"]["radix saved"] < 0.05
    # Token granularity can never save less than block granularity.
    for r in rows.values():
        assert r["radix saved"] >= r["block-hash saved"] - 1e-9


def test_prefix_cache_cuts_ttft(lab):
    tr = lab.multi_turn_trace(n_sessions=3, turns=4)
    assert lab.ttft_experiment(tr, True) < 0.8 * lab.ttft_experiment(tr, False)
