import pytest
import torch


def test_fragment_layouts_are_bijections(lab):
    a = {lab.a_frag(l, i) for l in range(32) for i in range(8)}
    b = {lab.b_frag(l, i) for l in range(32) for i in range(4)}
    c = {lab.c_frag(l, i) for l in range(32) for i in range(4)}
    assert len(a) == 256 and len(b) == 128 and len(c) == 128


def test_fragment_layouts_match_ptx_table(lab):
    # Spot checks from the PTX ISA m16n8k16 diagrams.
    assert [lab.a_frag(0, i) for i in range(8)] == [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)]
    assert [lab.b_frag(5, i) for i in range(4)] == [(2, 1), (3, 1), (10, 1), (11, 1)]
    assert [lab.c_frag(31, i) for i in range(4)] == [(7, 6), (7, 7), (15, 6), (15, 7)]


def test_simulated_mma(lab):
    A, B, C = torch.randn(16, 16), torch.randn(16, 8), torch.randn(16, 8)
    torch.testing.assert_close(lab.simulate_mma(A, B, C), A @ B + C, rtol=1e-5, atol=1e-5)


def test_bank_conflicts_and_swizzle(lab):
    assert lab.bank_conflict_degree(list(range(32))) == 1        # row read
    assert lab.bank_conflict_degree([0] * 32) == 1               # broadcast
    assert lab.bank_conflict_degree(lab.column_read(32)) == 32   # naive column
    assert lab.bank_conflict_degree(lab.column_read(33)) == 1    # padding
    assert lab.bank_conflict_degree(lab.column_read(32, swizzle=True)) == 1
    # Swizzling must keep each row a permutation of its columns.
    for r in range(32):
        assert sorted(lab.swizzled_col(r, c) for c in range(32)) == list(range(32))


def test_pipeline_model(lab):
    assert lab.pipeline_time(10, 1.0, 1.0, 1) == 20.0
    assert lab.pipeline_time(10, 1.0, 1.0, 2) == 11.0
    assert lab.pipeline_time(10, 2.0, 1.0, 3) == 2.0 + 9 * 2.0 + 1.0  # load-bound


@pytest.mark.gpu
def test_hardware_mma(lab):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("mma.sync m16n8k16 needs sm_80+")
    assert lab.gpu_mma_check() < 1e-2
