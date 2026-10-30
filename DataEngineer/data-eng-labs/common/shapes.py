"""Synthetic image-text and video data for the multimodal labs (Weeks 18-21).

Real LAION/DataComp samples are large and licence-encumbered; these are tiny
and have ground truth. Each image is a coloured shape on a noisy background,
captioned like web alt-text, and a known fraction of captions are wrong.
"""
from __future__ import annotations

import random

import numpy as np

COLORS = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (50, 80, 220), "yellow": (230, 210, 40)}
SHAPES = ["circle", "square", "triangle", "cross"]
FADE = 10   # frames in a cross-fade
JUNK = ["IMG_2041.jpg", "click here", "image", "photo 3 of 12", "untitled", "buy now 50% off"]


def draw(shape: str, color: str, size: int = 32, rnd: random.Random | None = None) -> np.ndarray:
    """uint8 HxWx3."""
    rnd = rnd or random.Random(0)
    nrng = np.random.default_rng(rnd.randrange(2**32))
    img = nrng.integers(0, 60, (size, size, 3)).astype(np.uint8) + np.uint8(rnd.randrange(0, 60))
    c = np.array(COLORS[color], dtype=np.uint8)
    r = rnd.randint(size // 5, size // 3)
    cy, cx = rnd.randint(r, size - r - 1), rnd.randint(r, size - r - 1)
    yy, xx = np.mgrid[:size, :size]
    if shape == "circle":
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    elif shape == "square":
        m = (abs(yy - cy) <= r) & (abs(xx - cx) <= r)
    elif shape == "triangle":
        m = (yy >= cy - r) & (yy <= cy + r) & (abs(xx - cx) <= (yy - (cy - r)) / 2)
    else:
        w = max(1, r // 3)
        m = ((abs(yy - cy) <= w) & (abs(xx - cx) <= r)) | ((abs(xx - cx) <= w) & (abs(yy - cy) <= r))
    img[m] = c
    return img


def caption(shape, color, rnd):
    t = rnd.choice(["a {c} {s}", "{c} {s} on a dark background", "photo of a {c} {s}", "the {s} is {c}", "{s}, {c}"])
    return t.format(c=color, s=shape)


def pairs(n: int, seed: int = 0, wrong: float = 0.0, junk: float = 0.0, size: int = 32):
    """Returns images uint8[n,H,W,3], captions, labels dicts (shape, color, ok)."""
    rnd = random.Random(seed)
    imgs, caps, labels = [], [], []
    for _ in range(n):
        s, c = rnd.choice(SHAPES), rnd.choice(list(COLORS))
        imgs.append(draw(s, c, size, rnd))
        u = rnd.random()
        if u < wrong:   # caption describes a different image
            s2, c2 = s, c
            while (s2, c2) == (s, c):
                s2, c2 = rnd.choice(SHAPES), rnd.choice(list(COLORS))
            caps.append(caption(s2, c2, rnd)); ok = False
        elif u < wrong + junk:
            caps.append(rnd.choice(JUNK)); ok = False
        else:
            caps.append(caption(s, c, rnd)); ok = True
        labels.append({"shape": s, "color": c, "ok": ok})
    return np.stack(imgs), caps, labels


def video(n_shots: int = 8, seed: int = 0, size: int = 32, fades: int = 2):
    """Frames uint8[T,H,W,3] made of shots (a moving shape over a static background)
    joined by hard cuts, plus `fades` cross-fades. Returns frames and true cut frame indices
    (index of the first frame of each new shot; a fade's cut is its middle frame)."""
    rnd = random.Random(seed)
    frames, cuts = [], []
    fade_at = set(rnd.sample(range(1, n_shots), min(fades, n_shots - 1)))
    for k in range(n_shots):
        s, c = rnd.choice(SHAPES), rnd.choice(list(COLORS))
        # textured background (gradient between two colours): histograms spread over
        # many bins, as in real footage, so a fade moves mass gradually
        c0 = np.array([rnd.randrange(256) for _ in range(3)], np.float32)
        c1 = np.array([rnd.randrange(256) for _ in range(3)], np.float32)
        ramp = np.linspace(0, 1, size, dtype=np.float32)[None, :, None]
        bg = np.broadcast_to(c0 + (c1 - c0) * ramp, (size, size, 3)).astype(np.uint8)
        L = rnd.randint(12, 30)
        shot = []
        x0 = rnd.randint(0, size // 3)
        for t in range(L):
            f = bg.copy().astype(np.int16)
            shape = draw(s, c, size, random.Random(k))
            m = shape.sum(-1) > 3 * 120
            shift = (x0 + t) % (size // 2)
            m = np.roll(m, shift, axis=1)
            f[m] = np.array(COLORS[c])
            f += np.random.default_rng(1000 * k + t).integers(-6, 7, f.shape)
            shot.append(np.clip(f, 0, 255).astype(np.uint8))
        if k > 0:
            if k in fade_at:
                prev = frames[-1].astype(np.float32)
                nxt = shot[0].astype(np.float32)
                for j in range(1, FADE + 1):
                    a = j / (FADE + 1)
                    frames.append(((1 - a) * prev + a * nxt).astype(np.uint8))
                cuts.append(len(frames) - (FADE + 1) // 2)
            else:
                cuts.append(len(frames))
        frames.extend(shot)
    return np.stack(frames), cuts
