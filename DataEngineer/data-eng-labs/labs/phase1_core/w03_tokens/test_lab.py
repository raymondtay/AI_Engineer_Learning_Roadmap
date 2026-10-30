import random

from common.webgen import good_doc


def test_roundtrip_any_text(lab):
    rnd = random.Random(0)
    merges = lab.train([good_doc(rnd) for _ in range(20)], 400)
    for s in ["Hello, world!  ", "Grüße aus München", "数据工程", "x = f(1) + 2\n\tend", ""]:
        assert lab.decode(lab.encode(s, merges), merges) == s


def test_vocab_and_compression(lab):
    rnd = random.Random(1)
    texts = [good_doc(rnd) for _ in range(30)]
    merges = lab.train(texts, 500)
    assert len(merges) == 244
    raw = sum(len(t.encode()) for t in texts)
    assert sum(len(lab.encode(t, merges)) for t in texts) < raw / 2.5


def test_first_merge_is_most_frequent_pair(lab):
    merges = lab.train(["aaab aaab aaab cd"], 257)
    assert merges == [(ord("a"), ord("a"))]


def test_english_cheaper_than_other_languages(lab):
    rows = {r["set"]: r["tokens/byte"] for r in lab.report(vocab=600)}
    assert rows["english prose"] < rows["german prose"] and rows["english prose"] < rows["french prose"]
