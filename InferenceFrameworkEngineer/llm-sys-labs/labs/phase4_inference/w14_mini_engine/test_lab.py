import torch


def _prompts(n, lo=5, hi=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist() for _ in range(n)]


def test_slots(lab):
    cache = lab.PagedKVCache(lab.GPTConfig(), 4, 8, torch.device("cpu"))
    s = cache.slots([3, 1], torch.tensor([0, 7, 8, 15]))
    assert s.tolist() == [24, 31, 8, 15]


def test_batched_matches_reference(lab):
    model = lab.make_model()
    prompts = _prompts(6)
    eng = lab.Engine(model, num_blocks=64, block_size=8, token_budget=32)  # small budget forces chunked prefill
    ids = [eng.add_request(p, 12) for p in prompts]
    out = eng.run()
    for i, p in zip(ids, prompts):
        assert out[i] == lab.greedy_reference(model, p, 12), f"request {i}"


def test_preemption_keeps_outputs_correct(lab):
    model = lab.make_model()
    prompts = _prompts(5, 20, 30, seed=3)
    eng = lab.Engine(model, num_blocks=14, block_size=8, token_budget=64)  # too few blocks for all at once
    ids = [eng.add_request(p, 16) for p in prompts]
    out = eng.run()
    assert eng.stats["preemptions"] > 0
    for i, p in zip(ids, prompts):
        assert out[i] == lab.greedy_reference(model, p, 16)
    assert eng.alloc.num_free == 14  # every block returned


def test_engine_faster_than_naive(lab):
    r = lab.benchmark(n_requests=8, prompt_len=32, gen=16)
    assert r["speedup"] > 1.5, r


import pytest


@pytest.mark.parametrize("blocks,budget,seed", [(8, 16, 0), (10, 64, 1), (12, 8, 2), (20, 33, 4), (7, 100, 5)])
def test_scheduler_stress(lab, blocks, budget, seed):
    """Tight pools and odd budgets: must finish, stay correct, and free every block."""
    model = lab.make_model()
    prompts = _prompts(6, 3, 30, seed=seed)
    eng = lab.Engine(model, num_blocks=blocks, block_size=8, token_budget=budget)
    ids = [eng.add_request(p, 10) for p in prompts]
    out = eng.run()
    for i, p in zip(ids, prompts):
        assert out[i] == lab.greedy_reference(model, p, 10)
    assert eng.alloc.num_free == blocks
