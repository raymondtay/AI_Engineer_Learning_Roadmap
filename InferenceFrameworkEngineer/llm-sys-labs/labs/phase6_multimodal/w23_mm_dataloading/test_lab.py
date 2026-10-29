import tempfile

import torch
from torch.utils.data import DataLoader

from common.tinygpt import GPT, GPTConfig


def test_every_sample_exactly_once(lab):
    d = tempfile.mkdtemp()
    shards = lab.write_shards(120, 12, d, img=8)
    seen = []
    for rank in range(2):  # two DDP ranks, three loader workers each
        ds = lab.ShardStream(shards, rank=rank, world=2)
        for b in DataLoader(ds, batch_size=8, num_workers=3, collate_fn=lab.collate):
            seen += b["id"].tolist()
    assert sorted(seen) == list(range(120))


def test_workers_needed(lab):
    assert lab.workers_needed(0.04, 0.02, 8) == (2, True)
    assert lab.workers_needed(0.10, 0.02, 4) == (5, False)


def test_workers_hide_decode_when_cores_allow(lab):
    """Step long enough that two single-threaded workers can keep up on 2+ cores."""
    d = tempfile.mkdtemp()
    shards = lab.write_shards(512, 8, d)
    dec = lab.decode_cost_per_batch(shards)
    step = max(0.06, 1.5 * dec)  # one worker's supply rate comfortably beats consumption x 2 workers
    s0 = lab.measure_stall(shards, 0, step_compute_s=step)
    s2 = lab.measure_stall(shards, 2, step_compute_s=step)
    assert s0["steps"] == s2["steps"]
    assert s2["data wait s"] < 0.5 * s0["data wait s"], (s0, s2)


def test_ffd_respects_capacity(lab):
    lengths = [50, 40, 30, 30, 20, 10, 90, 60]
    rows = lab.pack_ffd(lengths, 100)
    assert sorted(i for r in rows for i in r) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in r) <= 100 for r in rows)
    assert len(rows) == 4  # optimal here: 330 tokens need at least 4 rows of 100


def test_packing_beats_padding(lab):
    g = torch.Generator().manual_seed(0)
    lengths = [int(torch.randint(24, 80, (1,), generator=g)) for _ in range(500)]
    assert lab.packing_efficiency(lengths, 256) > lab.padding_efficiency(lengths, 32) + 0.1


def test_packed_logits_equal_unpacked(lab):
    torch.manual_seed(0)
    model = GPT(GPTConfig(seq_len=64, n_layer=2, n_head=4, d_model=32)).eval()
    seqs = [torch.randint(0, 256, (n,)) for n in (7, 13, 5, 20, 9)]
    groups = lab.pack_ffd([len(s) for s in seqs], 32)
    tok, pos, mask, seg = lab.build_packed(seqs, groups, 32)
    with torch.no_grad():
        packed, _ = model(tok, pos=pos, attn_mask=mask)
        for r in range(tok.shape[0]):
            for i in set(seg[r].tolist()) - {-1}:
                where = seg[r] == i
                alone, _ = model(seqs[i][None])
                torch.testing.assert_close(packed[r, where], alone[0], rtol=1e-4, atol=1e-4)
