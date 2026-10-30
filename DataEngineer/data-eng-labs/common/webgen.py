"""Synthetic web corpus with known ground truth.

Every lab needs data whose answer is known: which documents are near-duplicates
of which, which are spam, which language each is in, where the PII and the
eval contamination are. Real crawls do not come with labels, so the labs
generate pages with controllable properties and keep the labels alongside.

Nothing here is downloaded or copied: word lists are generic vocabulary and all
text is assembled from templates at random.
"""
from __future__ import annotations

import random
import string

WORDS = {
    "en": ("the of and to in is that for it as was with be by on not he this are or his from at which but have an they "
           "you were her she there been one all we their has would when who will more no if out so said what up its about "
           "into than them can only other new some could time these two may then do first any my now such like our over man "
           "me even most made after also did many before must through back years where much your way well down should because "
           "each just those people how too little state good very make world still own see men work long get here between both "
           "life being under never day same another know while last might us great old year off come since against go came right "
           "used take three").split(),
    "de": ("der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass "
           "sie nach wird bei einer um am sind noch wie einem über einen so zum war haben nur oder aber vor zur bis mehr durch "
           "man sein wurde sei hatte kann gegen vom können schon wenn habe seine ihre dann unter wir soll ich eines jahr zwei "
           "jahren diese dieser wieder keine seiner worden").split(),
    "fr": ("de la le et les des en un du une que est pour qui dans par plus pas au sur ne se le ce il sont son mais avec "
           "comme ou leur nous elle ont aux été cette tout aussi ces sa ses deux être fait même bien sans entre dont temps "
           "peut ans très après autres encore faire avant avoir lui depuis tous notre premier pays donc alors moins").split(),
    "es": ("de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta "
           "entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros "
           "ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho").split(),
}
TOPIC = ("model data system network energy market climate protein language theory method result study river city "
         "history music engine design policy science health school bridge forest ocean galaxy circuit algorithm").split()
SPAM = "buy cheap best free click discount viagra casino winner offer deal online now limited bonus".split()
NAV = ["Home", "About", "Contact", "Privacy Policy", "Terms of Use", "Login", "Sign up", "Cookie settings", "Sitemap"]


def sentence(rnd: random.Random, lang: str = "en", n: tuple[int, int] = (8, 18)) -> str:
    words = WORDS[lang]
    k = rnd.randint(*n)
    toks = [rnd.choice(words) for _ in range(k)]
    if lang == "en":  # sprinkle topical content words so documents differ in meaning
        for _ in range(max(1, k // 5)):
            toks[rnd.randrange(k)] = rnd.choice(TOPIC)
    s = " ".join(toks)
    return s[0].upper() + s[1:] + rnd.choice([".", ".", ".", "?", "!"])


def paragraph(rnd, lang="en", n=(3, 7)) -> str:
    return " ".join(sentence(rnd, lang) for _ in range(rnd.randint(*n)))


def good_doc(rnd, lang="en", n_par=(3, 6)) -> str:
    return "\n".join(paragraph(rnd, lang) for _ in range(rnd.randint(*n_par)))


def spam_doc(rnd) -> str:
    lines = []
    for _ in range(rnd.randint(8, 20)):
        lines.append(" ".join(rnd.choice(SPAM) for _ in range(rnd.randint(2, 6))))
    return "\n".join(lines)


def boilerplate_doc(rnd) -> str:
    return "\n".join(rnd.choice(NAV) for _ in range(rnd.randint(10, 25)))


def code_doc(rnd) -> str:
    body = [f"function f{i}(x) {{ return x * {rnd.randint(2, 9)}; }}" for i in range(rnd.randint(5, 12))]
    return "\n".join(body)


def lorem_doc(rnd) -> str:
    return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * rnd.randint(3, 8)


def perturb(rnd, text: str, rate: float) -> str:
    """Near-duplicate: replace, delete or insert ~rate of the words."""
    words = text.split(" ")
    out = []
    for w in words:
        r = rnd.random()
        if r < rate / 3:
            continue
        if r < 2 * rate / 3:
            out.append(rnd.choice(WORDS["en"]))
            continue
        out.append(w)
        if r < rate:
            out.append(rnd.choice(WORDS["en"]))
    return " ".join(out)


def add_pii(rnd, text: str) -> str:
    user = "".join(rnd.choice(string.ascii_lowercase) for _ in range(6))
    email = f"{user}@example{rnd.randint(1, 99)}.com"
    phone = f"+1-{rnd.randint(200, 999)}-{rnd.randint(200, 999)}-{rnd.randint(1000, 9999)}"
    return text + f"\nContact {email} or call {phone}."


def to_html(rnd, text: str, title: str = "Page") -> str:
    """Wrap main text in typical page chrome: nav, sidebar links, footer, a script."""
    nav = "".join(f'<li><a href="/{n.lower().replace(" ", "-")}">{n}</a></li>' for n in rnd.sample(NAV, 6))
    side = "".join(f'<a href="/tag/{t}">{t}</a> ' for t in rnd.sample(TOPIC, 8))
    paras = "".join(f"<p>{p}</p>" for p in text.split("\n"))
    return (f"<html><head><title>{title}</title><script>var t = {{a: 1}};</script></head><body>"
            f"<nav><ul>{nav}</ul></nav><div class='sidebar'>{side}</div>"
            f"<article><h1>{title}</h1>{paras}</article>"
            f"<footer>Copyright {rnd.randint(2000, 2026)} Example Corp. All rights reserved. "
            f"<a href='/privacy'>Privacy</a></footer></body></html>")


KINDS = ("good", "spam", "boilerplate", "code", "lorem")


def corpus(n: int = 500, seed: int = 0, mix: dict | None = None, exact_dup: float = 0.05, near_dup: float = 0.1,
           near_rate: float = 0.05, langs: dict | None = None, pii: float = 0.05) -> list[dict]:
    """Documents with labels: kind, lang, dup_of (exact), near_of (near-duplicate source)."""
    rnd = random.Random(seed)
    mix = mix or {"good": 0.7, "spam": 0.1, "boilerplate": 0.08, "code": 0.07, "lorem": 0.05}
    langs = langs or {"en": 0.8, "de": 0.07, "fr": 0.07, "es": 0.06}
    docs: list[dict] = []
    for i in range(n):
        r = rnd.random()
        if docs and r < exact_dup:
            src = rnd.choice(docs)
            docs.append({**src, "id": i, "url": f"https://mirror{rnd.randint(1, 9)}.example.org/{i}",
                         "dup_of": src["id"] if src.get("dup_of") is None else src["dup_of"], "near_of": None})
            continue
        if docs and r < exact_dup + near_dup:
            src = rnd.choice([d for d in docs if d["kind"] == "good"] or docs)
            docs.append({**src, "id": i, "url": f"https://site{rnd.randint(1, 50)}.example.com/p/{i}",
                         "text": perturb(rnd, src["text"], near_rate), "dup_of": None, "near_of": src["id"]})
            continue
        kind = rnd.choices(list(mix), weights=list(mix.values()))[0]
        lang = rnd.choices(list(langs), weights=list(langs.values()))[0] if kind == "good" else "en"
        text = {"good": lambda: good_doc(rnd, lang), "spam": lambda: spam_doc(rnd),
                "boilerplate": lambda: boilerplate_doc(rnd), "code": lambda: code_doc(rnd),
                "lorem": lambda: lorem_doc(rnd)}[kind]()
        has_pii = kind == "good" and rnd.random() < pii
        if has_pii:
            text = add_pii(rnd, text)
        docs.append({"id": i, "url": f"https://site{rnd.randint(1, 50)}.example.com/p/{i}", "lang": lang,
                     "kind": kind, "text": text, "pii": has_pii, "dup_of": None, "near_of": None})
    return docs
