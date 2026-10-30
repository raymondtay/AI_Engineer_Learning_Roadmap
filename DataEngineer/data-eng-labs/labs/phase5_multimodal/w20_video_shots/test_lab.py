import numpy as np

from common.shapes import video


def test_hist_distance_bounds(lab):
    f, _ = video(3, seed=1, fades=0)
    d = lab.diffs(f)
    assert d[0] == 0 and (d >= 0).all() and (d <= 1 + 1e-9).all()
    same = np.stack([f[0], f[0]])
    assert lab.diffs(same)[1] == 0


def test_score(lab):
    assert lab.score([10, 50, 90], [11, 48, 70]) == {"precision": 2 / 3, "recall": 2 / 3}
    assert lab.score([10, 11], [10])["precision"] == 0.5   # one-to-one matching


def test_twin_finds_fades_hard_threshold_does_not(lab):
    P_t = R_t = R_h = 0
    for s in range(10, 16):
        f, truth = video(10, seed=s, fades=3)
        d = lab.diffs(f)
        tw = lab.score(lab.twin_comparison(d, 0.5, 0.2), truth)
        hd = lab.score(lab.detect_hard(d, 0.8), truth)
        P_t += tw["precision"]; R_t += tw["recall"]; R_h += hd["recall"]
    assert R_t / 6 > 0.85 and P_t / 6 > 0.8
    assert R_h / 6 < R_t / 6 - 0.15


def test_clips_and_motion(lab):
    f, truth = video(6, seed=2, fades=0)
    cl = lab.clips(len(f), truth, min_len=8, trim=1)
    assert all(b - a >= 8 for a, b in cl)
    assert all(not (a <= c < b) or c == a for a, b in cl for c in truth)
    still = np.repeat(f[:1], 10, axis=0)
    assert lab.motion(still, (0, 10)) == 0 < lab.motion(f, cl[0])
    k = lab.keyframe(f, cl[0])
    assert cl[0][0] <= k < cl[0][1]
