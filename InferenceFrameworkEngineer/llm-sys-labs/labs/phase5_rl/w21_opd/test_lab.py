import copy

import pytest
import torch


@pytest.fixture(scope="module")
def teacher(request):
    from conftest import load_lab

    return load_lab(str(request.fspath)).make_teacher()


def test_full_kl_zero_when_student_is_teacher(lab):
    m = lab.new_policy()
    seqs = lab.sample(m, ["12+34="], 4, generator=torch.Generator().manual_seed(0))
    mask = lab.completion_mask(seqs)
    assert lab.opd_loss_full(m, copy.deepcopy(m), seqs, mask).abs().item() < 1e-6


def test_sampled_advantage_sign(lab):
    """Tokens the teacher likes more than the student get positive advantage."""
    m = lab.new_policy()
    seqs = lab.sample(m, ["12+34="], 2, generator=torch.Generator().manual_seed(0))
    mask = lab.completion_mask(seqs)
    loss_self = lab.opd_loss_sampled(m, copy.deepcopy(m), seqs, mask)
    assert abs(loss_self.item()) < 1e-6  # zero advantage when teacher == student


def test_opd_improves_student(lab, teacher):
    r = lab.run("opd_full", steps=30, teacher=teacher)
    assert r["acc_end"] > r["acc_start"] + 0.1, r
    assert r["teacher_tokens"] > 0
