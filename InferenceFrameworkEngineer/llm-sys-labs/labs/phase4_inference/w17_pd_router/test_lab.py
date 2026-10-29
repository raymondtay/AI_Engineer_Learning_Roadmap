from common import dist as ldist


def test_pd_matches_monolithic(lab):
    out = ldist.run(lab.pd_worker, 2)
    model = lab.make_model()
    for got, prompt in zip(out[1], lab.requests()):
        assert got == lab.greedy_monolithic(model, prompt, 8)
    cfg = model.cfg
    expect = sum(len(p) for p in lab.requests()) * lab.kv_bytes_per_token(cfg.n_layer, cfg.n_head, cfg.head_dim, 4)
    assert out[0] == expect


def test_kv_bytes(lab):
    # Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128, bf16 -> 128 KiB per token
    assert lab.kv_bytes_per_token(32, 8, 128) == 131072


def test_cache_aware_beats_round_robin(lab):
    rows = {r["policy"]: r for r in lab.compare(n_inst=4)}
    rr, ca = rows["round_robin"], rows["cache_aware"]
    # Round-robin still reuses the shared system prompt on every instance;
    # cache-aware routing also keeps each session's history on one instance.
    assert ca["prefix reuse"] > rr["prefix reuse"] + 0.1
    assert ca["P50 TTFT ms"] < rr["P50 TTFT ms"]
    assert ca["req/s"] >= rr["req/s"] * 0.99
