import random


def test_features_normalised(lab):
    f = lab.features("a b a b")
    assert abs(sum(v * v for v in f.values()) - 1) < 1e-9


def test_classifier_ranks_quality(lab):
    r = lab.experiment()
    # Overlapping classes + 10% label noise: measured AUC ~0.92, top-10% precision ~0.6 (base rate 0.1)
    assert r["auc"] > 0.85
    assert r["top10%"]["precision"] > 0.4


def test_auc_bounds(lab):
    assert lab.auc([1, 2], [0, 0]) == 1.0
    assert lab.auc([0], [1]) == 0.0


def test_decontamination(lab):
    exact = lab.decontam_experiment(n=13, edit_rate=0.0)
    assert exact["recall"] == 1.0 and exact["precision"] == 1.0
    # Longer n-grams miss edited contamination; shorter ones catch more of it.
    assert lab.decontam_experiment(n=8, edit_rate=0.15)["recall"] > lab.decontam_experiment(n=13, edit_rate=0.15)["recall"]


def test_exact_overlap_detected(lab):
    q = " ".join(str(i) for i in range(20))
    assert lab.contaminated(["intro " + q + " outro", "unrelated text " * 10], [q]) == [0]
