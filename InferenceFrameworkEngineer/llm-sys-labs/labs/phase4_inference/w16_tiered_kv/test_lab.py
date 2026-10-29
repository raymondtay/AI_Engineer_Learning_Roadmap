import random

import torch


def _blk(i):
    return torch.full((2, 2, 3, 4, 5), float(i))


def test_lru_demotion_through_tiers(lab):
    s = lab.TieredKVStore(2, 2, 10)
    for i in range(6):
        s.put(i, _blk(i))
    assert [s.where(i) for i in range(6)] == ["disk", "disk", "host", "host", "device", "device"]
    assert s.stats["demotions"] == 6
    s.close()


def test_disk_roundtrip_and_promotion(lab):
    s = lab.TieredKVStore(1, 1, 10)
    for i in range(4):
        s.put(i, _blk(i))
    assert s.where(0) == "disk"
    got = s.get(0)
    assert torch.equal(got.cpu(), _blk(0)) and s.where(0) == "device"
    assert s.stats["hit_disk"] == 1 and s.stats["promotions"] == 1
    assert s.get(99) is None and s.stats["miss"] == 1
    s.close()


def test_capacity_drops_past_last_tier(lab):
    s = lab.TieredKVStore(1, 1, 1)
    for i in range(5):
        s.put(i, _blk(i))
    assert sum(len(t) for t in s.tiers.values()) == 3
    assert len(list(s.dir.glob("*.pt"))) == 1  # evicted disk files are deleted
    s.close()


def test_prefetch_lifts_disk_to_host(lab):
    s = lab.TieredKVStore(1, 2, 10)
    for i in range(5):
        s.put(i, _blk(i))
    disk_keys = [k for k in range(5) if s.where(k) == "disk"]
    assert disk_keys == [0, 1]
    s.prefetch(disk_keys)
    s.pool.shutdown(wait=True)
    assert all(s.where(k) == "host" for k in disk_keys)
    s.close()


def test_kv_service_reuse_is_exact(lab):
    torch.manual_seed(0)
    model = lab.GPT(lab.GPTConfig(seq_len=512, n_layer=2, n_head=4, d_model=64)).eval()
    store = lab.TieredKVStore(2, 2, 100)  # tiny upper tiers: hits come from disk too
    svc, ref = lab.KVService(model, store, 8), lab.KVService(model, None, 8)
    rnd = random.Random(0)
    history = [rnd.randrange(256) for _ in range(40)]
    for turn in range(4):
        history += [rnd.randrange(256) for _ in range(20)]
        got, hit = svc.prefill(history)
        want, _ = ref.prefill(history)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
        if turn:
            assert hit > 0
    assert store.stats["hit_disk"] + store.stats["hit_host"] > 0
    store.close()
