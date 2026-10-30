# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 20: video into training clips - shot boundaries, fades, clip filters.

Roadmap: "Shot-boundary detection (hard cuts and gradual transitions), clip
splitting, static/low-motion filtering, keyframes for captioning."

Video datasets are built from clips, not whole videos: Panda-70M (Chen et al.
2024) splits long videos at shot boundaries and then stitches/filters them into
semantically coherent clips before captioning. A clip that straddles a cut
teaches the model that scenes teleport.

  hard cut   one frame-to-frame difference spikes. Colour-histogram distance is
             robust to motion inside a shot (PySceneDetect's ContentDetector uses
             HSV differences in the same spirit).
  gradual    a fade spreads the change over several frames, each under the
             cut threshold. Twin-comparison (Zhang, Kankanhalli & Smoliar 1993):
             a lower threshold opens a candidate, and the ACCUMULATED difference
             since the candidate started must pass the high threshold.
Metrics: precision/recall with a +-2 frame tolerance, as in TRECVID SBD evals.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.shapes import video


def hist(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalised joint RGB histogram (bins^3)."""
    q = (frame.astype(np.int32) * bins) // 256
    idx = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    h = np.bincount(idx.ravel(), minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def diffs(frames: np.ndarray, kind: str = "hist") -> np.ndarray:
    """d[t] = distance between frame t-1 and t (d[0] = 0)."""
    # TODO: 'hist': half the L1 distance between histograms (in [0,1]); 'pixel': mean absolute pixel difference / 255
    raise NotImplementedError("TODO(w20_video_shots): 'hist': half the L1 distance between histograms (in [0,1]); 'pixel': mean absolute pixel difference / 255")


def detect_hard(d: np.ndarray, thresh: float, min_gap: int = 5) -> list[int]:
    cuts = []
    for t in np.flatnonzero(d > thresh):
        if not cuts or t - cuts[-1] >= min_gap:
            cuts.append(int(t))
    return cuts


def twin_comparison(d: np.ndarray, high: float, low: float, min_gap: int = 5) -> list[int]:
    """Twin-comparison: frames with d > low form runs. A short run (<= 2 frames) whose
    peak exceeds `high` is a hard cut (at the peak). A longer run whose ACCUMULATED
    difference exceeds `high` is a gradual transition, reported at the run's middle.
    Motion spikes inside a shot are short and low, so they pass neither test."""
    cuts = []
    t = 1
    # TODO: walk runs of d > low; classify each run as above; respect min_gap from the previous cut
    raise NotImplementedError("TODO(w20_video_shots): walk runs of d > low; classify each run as above; respect min_gap from the previous cut")


def score(pred: list[int], truth: list[int], tol: int = 2) -> dict:
    # TODO: greedy one-to-one matching within +-tol; precision = matched/len(pred), recall = matched/len(truth)
    raise NotImplementedError("TODO(w20_video_shots): greedy one-to-one matching within +-tol; precision = matched/len(pred), recall = matched/len(truth)")


def clips(n_frames: int, cuts: list[int], min_len: int = 8, trim: int = 1) -> list[tuple[int, int]]:
    """Shots -> clips [start, end): drop `trim` frames each side of a boundary
    (transition residue) and clips shorter than min_len."""
    bounds = [0] + list(cuts) + [n_frames]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        a2, b2 = a + (trim if a > 0 else 0), b - (trim if b < n_frames else 0)
        if b2 - a2 >= min_len:
            out.append((a2, b2))
    return out


def motion(frames: np.ndarray, clip: tuple[int, int]) -> float:
    """Mean frame-to-frame pixel change inside a clip (low = static slide / freeze frame)."""
    a, b = clip
    return float(np.abs(np.diff(frames[a:b].astype(np.int16), axis=0)).mean())


def keyframe(frames: np.ndarray, clip: tuple[int, int]) -> int:
    """The frame closest to the clip's mean histogram (a cheap 'representative' frame for captioning)."""
    a, b = clip
    H = np.stack([hist(f) for f in frames[a:b]])
    return a + int(np.abs(H - H.mean(0)).sum(1).argmin())


def evaluate(seeds=range(10, 20), high=0.5, low=0.2) -> list[dict]:
    rows = []
    for name in ("pixel hard", "hist hard", "hist twin"):
        P = R = 0.0
        for s in seeds:
            frames, truth = video(n_shots=10, seed=s, fades=3)
            if name == "pixel hard":
                pred = detect_hard(diffs(frames, "pixel"), 0.1)
            elif name == "hist hard":
                pred = detect_hard(diffs(frames), high)
            else:
                pred = twin_comparison(diffs(frames), high, low)
            sc = score(pred, truth)
            P += sc["precision"]; R += sc["recall"]
        rows.append({"detector": name, "precision": P / len(seeds), "recall": R / len(seeds)})
    return rows


def report() -> str:
    frames, truth = video(n_shots=10, seed=0, fades=3)
    pred = twin_comparison(diffs(frames), 0.5, 0.2)
    cl = clips(len(frames), pred)
    still = np.repeat(frames[:1], 20, axis=0)
    return ("Thresholds swept on videos 0-9; scored on held-out videos 10-19 (10 shots, 3 ten-frame cross-fades each), +-2 frames:\n" +
            table(evaluate(), ["detector", "precision", "recall"]) +
            f"\n\nvideo 0: {len(frames)} frames -> {len(cl)} clips; motion of first clip {motion(frames, cl[0]):.2f} "
            f"vs freeze-frame {motion(still, (0, 20)):.2f}; keyframe of clip 0 = frame {keyframe(frames, cl[0])}")


if __name__ == "__main__":
    print(report())
