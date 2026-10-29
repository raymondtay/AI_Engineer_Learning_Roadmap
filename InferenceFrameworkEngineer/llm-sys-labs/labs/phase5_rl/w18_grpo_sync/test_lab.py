import torch


def test_group_advantages(lab):
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = lab.group_advantages(r)
    assert torch.allclose(a[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(a[1], torch.zeros(4))  # all-correct group carries no signal


def test_loss_is_zero_gradient_signal_on_policy(lab):
    # On-policy (ratio = 1) with zero advantage and ref == policy -> loss 0.
    lp = torch.randn(3, 4)
    m = torch.ones(3, 4)
    assert torch.isclose(lab.grpo_loss(lp, lp, lp, torch.zeros(3), m), torch.tensor(0.0))


def test_clipping(lab):
    lp_old = torch.zeros(1, 1)
    lp_new = torch.log(torch.tensor([[2.0]])).requires_grad_()  # ratio 2 > 1 + clip
    loss = lab.grpo_loss(lp_new, lp_old, lp_new.detach(), torch.tensor([1.0]), torch.ones(1, 1), beta=0.0)
    loss.backward()
    assert torch.isclose(loss, torch.tensor(-1.2)) and lp_new.grad.abs().item() == 0.0  # clipped: no gradient


def test_grpo_improves_accuracy(lab):
    _, s = lab.train(steps=30)
    assert s["acc_end"] > s["acc_start"] + 0.1, s
    assert s["gen_s"] > 0 and s["train_s"] > 0
