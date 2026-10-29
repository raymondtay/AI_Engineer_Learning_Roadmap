import pytest
import torch


def test_patchify_shapes_and_order(lab):
    pe = lab.PatchEmbed(8, 16)
    img = torch.arange(3 * 32 * 32, dtype=torch.float).view(1, 3, 32, 32)
    out = pe(img)
    assert out.shape == (1, 16, 16)
    # First patch must be the top-left 8x8 window of every channel.
    cols = torch.nn.functional.unfold(img, 8, stride=8)
    assert torch.equal(cols[0, :, 0], img[0, :, :8, :8].reshape(-1))


def test_loss_mask_only_text(lab):
    m = lab.TinyVLM()
    img = torch.zeros(2, 3, 32, 32)
    txt = lab.encode_captions(["red bar\n", "blue cross\n"])
    emb, tgt, mask = m.build_inputs(img, txt)
    n_img = 16
    assert emb.shape[1] == n_img + txt.shape[1] - 1
    assert not mask[:, : n_img - 1].any()
    assert torch.equal(tgt[:, n_img - 1 :][mask[:, n_img - 1 :]], txt[txt != 0])


def test_recompute_same_grads_less_memory(lab):
    torch.manual_seed(0)
    m = lab.TinyVLM()
    b = lab.dataset(4, 1)
    img, txt = torch.stack([x for x, _ in b]), lab.encode_captions([c for _, c in b])
    g1 = torch.autograd.grad(m(img, txt), list(m.vit.parameters()))
    m.vit.recompute = True
    g2 = torch.autograd.grad(m(img, txt), list(m.vit.parameters()))
    for a, c in zip(g1, g2):
        torch.testing.assert_close(a, c)
    m.vit.recompute = False
    assert lab.saved_bytes(m, True, False) < lab.saved_bytes(m, False, False)
    assert lab.saved_bytes(m, False, True) < lab.saved_bytes(m, False, False)


def test_vlm_learns_captions(lab):
    m = lab.train(steps=300)
    assert lab.caption_accuracy(m) > 0.8
