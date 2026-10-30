import collections
import random

from common.webgen import corpus, good_doc


def test_good_english_passes(lab):
    rnd = random.Random(0)
    docs = [good_doc(rnd) for _ in range(100)]
    fails = [lab.gopher_quality(d) or lab.gopher_repetition(d) or lab.fineweb(d) for d in docs]
    assert sum(f is None for f in fails) >= 97, collections.Counter(fails)


def test_specific_rules(lab):
    assert lab.gopher_quality("too short") == "gopher_word_count"
    no_stop = " ".join(["model data system"] * 30) + "."
    assert lab.gopher_quality(no_stop) == "gopher_stop_words"
    rep = "\n".join(["the cat sat with the dog and that was all."] * 10)
    assert lab.gopher_repetition(rep) == "gopher_dup_line_frac"
    assert lab.c4("Lorem ipsum dolor sit amet")[0] == "c4_lorem_ipsum"
    assert lab.c4("x = { a: 1 }")[0] == "c4_curly_bracket"
    reason, cleaned = lab.c4("Real content here.\nThis site uses cookies to improve.\nMore content.")
    assert reason is None and "cookies" not in cleaned
    nav = "\n".join(["Home", "About us", "Contact", "Login"] * 3)
    assert lab.fineweb(nav) == "fineweb_line_punct"


def test_pii(lab):
    text, n = lab.redact_pii("Mail jane.doe@example.com, call +1-415-555-0199, server 192.168.10.4.")
    assert n == 3 and "<EMAIL>" in text and "<PHONE>" in text and "<IP>" in text
    assert "example.com" not in text


def test_pipeline_separates_kinds(lab):
    docs = [d for d in corpus(1500, seed=2) if d["lang"] == "en"]
    kept, stats = lab.run(docs)
    kept_ids = {d["id"] for d in kept}
    by = collections.defaultdict(lambda: [0, 0])
    for d in docs:
        by[d["kind"]][0] += 1
        by[d["kind"]][1] += d["id"] in kept_ids
    assert by["good"][1] / by["good"][0] > 0.95
    for bad in ("spam", "boilerplate", "code", "lorem"):
        assert by[bad][1] / by[bad][0] < 0.05, (bad, by[bad])
    assert all("@" not in d["text"] for d in kept)
