import random

from common.webgen import good_doc


def test_exact_dedup_normalises(lab):
    keep = lab.exact_dedup(["Hello  World", "hello world", "other", "Other\n"])
    assert keep == [0, 2]


def test_suffix_array_sorted(lab):
    data = b"banana\x00bandana"
    sa = lab.suffix_array(data)
    assert [data[i:] for i in sa] == sorted(data[i:] for i in range(len(data)))


def test_shared_span_removed_from_later_docs(lab):
    d = lab.make_dataset(30)
    docs = d["with_blurb"][:12]
    spans = lab.repeated_spans(docs, min_len=100)
    cleaned = [lab.remove_spans(t, s) for t, s in zip(docs, spans)]
    assert lab.NEWSLETTER in cleaned[0]
    for c, original in zip(cleaned[1:], d["src"][1:12]):
        assert lab.NEWSLETTER[:40] not in c
        assert original[:200] in c  # unique content survives


def test_semantic_catches_reordered_copies(lab):
    d = lab.make_dataset(120)
    texts = d["src"] + d["exact"] + d["semantic"]
    n, ne = len(d["src"]), len(d["exact"])
    removed = set(range(len(texts))) - set(lab.semdedup(lab.embed(texts)))
    reordered = set(range(n + ne, len(texts)))
    assert len(removed & reordered) >= 0.9 * len(reordered)
    assert len(removed & set(range(n))) <= 2  # almost no originals lost
    # exact hashing cannot see them at all
    assert reordered.isdisjoint(set(range(len(texts))) - set(lab.exact_dedup(texts)))
