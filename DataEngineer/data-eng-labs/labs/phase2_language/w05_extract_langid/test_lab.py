import random

from common.webgen import good_doc, to_html


def test_extraction_beats_naive(lab):
    rows = {r["extractor"]: r for r in lab.extraction_report(n=30)}
    ours, naive = rows["ours"], rows["naive (WET-like)"]
    assert ours["word precision"] > 0.97 and ours["word recall"] > 0.95
    assert naive["word precision"] < ours["word precision"]


def test_chrome_is_removed(lab):
    rnd = random.Random(3)
    html = to_html(rnd, good_doc(rnd), title="T")
    text = lab.extract_main_text(html)
    for chrome in ("Privacy Policy", "All rights reserved", "var t"):
        assert chrome not in text


def test_langid(lab):
    model = lab.train_langid()
    rnd = random.Random(9)
    for l in ("en", "de", "fr", "es"):
        pred, p = model.predict(good_doc(rnd, l))
        assert pred == l and p > 0.99
    rows = {r["words"]: r for r in lab.langid_accuracy_by_length(model)}
    assert rows[30]["accuracy"] > 0.97
    # Short texts: less accurate, and the model is overconfident about them
    # (mean confidence above actual accuracy) - a threshold alone will not save you.
    assert rows[1]["accuracy"] < rows[30]["accuracy"]
    assert rows[1]["mean confidence"] > rows[1]["accuracy"]
