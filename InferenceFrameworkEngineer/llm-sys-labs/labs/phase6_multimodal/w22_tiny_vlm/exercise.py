# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 22: VLM model structure and memory.

Roadmap: "Build a ViT encoder + projector + LLM in Megatron-Core or FSDP2.
Decide where the encoder sits in the pipeline; recompute vs offload for the
vision tower."

The LLaVA-style recipe: image -> patches -> ViT -> MLP projector -> token
embeddings spliced in front of the text, then an ordinary causal LM. Loss on
text tokens only.

Synthetic task so it trains on CPU in seconds: 32x32 images of one coloured
shape; caption "red square\n" etc. The memory part measures what the vision
tower costs in saved activations, and what recompute and freezing buy back.

Pipeline-placement question (answer in your write-up): the ViT runs once per
image before the LLM's first layer. In pipeline parallel, putting it on stage 0
unbalances that stage by the ViT's FLOPs; options are a separate encoder stage,
giving stage 0 fewer LLM layers, or running the encoder data-parallel outside
the pipeline and sending embeddings in.
"""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.tinygpt import GPT, GPTConfig, Block

COLORS = {"red": (1.0, 0.1, 0.1), "green": (0.1, 1.0, 0.1), "blue": (0.1, 0.1, 1.0)}
SHAPES = ("square", "cross", "bar")


def render(color: str, shape: str, rnd: random.Random, size: int = 32) -> torch.Tensor:
    img = torch.zeros(3, size, size)
    c = torch.tensor(COLORS[color])[:, None, None]
    x0, y0 = rnd.randint(4, 14), rnd.randint(4, 14)
    if shape == "square":
        img[:, y0 : y0 + 12, x0 : x0 + 12] = c
    elif shape == "cross":
        img[:, y0 + 5 : y0 + 7, x0 : x0 + 14] = c
        img[:, y0 : y0 + 14, x0 + 6 : x0 + 8] = c
    else:
        img[:, y0 : y0 + 3, x0 : x0 + 16] = c
    return img + 0.05 * torch.randn_like(img)


def dataset(n: int, seed: int = 0):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        col, shp = rnd.choice(list(COLORS)), rnd.choice(SHAPES)
        out.append((render(col, shp, rnd), f"{col} {shp}\n"))
    return out


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(3 * patch * patch, dim)

    def forward(self, img):
        """img (B, 3, H, W) -> (B, N_patches, dim)."""
        # TODO: F.unfold with kernel=stride=patch gives (B, 3*p*p, N); transpose to (B, N, 3*p*p); linear
        raise NotImplementedError("TODO(w22_tiny_vlm): F.unfold with kernel=stride=patch gives (B, 3*p*p, N); transpose to (B, N, 3*p*p); linear")


class ViT(nn.Module):
    def __init__(self, patch=8, dim=64, depth=2, heads=4, img=32):
        super().__init__()
        self.embed = PatchEmbed(patch, dim)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        cfg = GPTConfig(n_layer=depth, n_head=heads, d_model=dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(depth))
        self.recompute = False

    def forward(self, img):
        x = self.embed(img) + self.pos
        full = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)[None].expand(x.shape[0], -1, -1)
        pos = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)  # RoPE off for patches: learned pos instead
        for b in self.blocks:
            # Bidirectional attention over patches: pass an all-True mask.
            x = checkpoint(b, x, pos, full, use_reentrant=False) if (self.recompute and self.training) else b(x, pos, full)
        return x


class TinyVLM(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.vit = ViT(dim=d_model)
        self.projector = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.lm = GPT(GPTConfig(seq_len=64, n_layer=2, n_head=4, d_model=d_model))

    def build_inputs(self, img, text_ids):
        """Return (embeds, targets, loss_mask) for [image tokens][text tokens]."""
        # TODO: img_emb = projector(vit(img)); txt_emb = lm.wte(text_ids); concat; targets = next tokens with image positions masked out of the loss
        raise NotImplementedError("TODO(w22_tiny_vlm): img_emb = projector(vit(img)); txt_emb = lm.wte(text_ids); concat; targets = next tokens with image positions masked out of the loss")

    def forward(self, img, text_ids):
        emb, tgt, mask = self.build_inputs(img, text_ids)
        _, loss = self.lm(embeds=emb, targets=tgt, loss_mask=mask)
        return loss

    @torch.no_grad()
    def caption(self, img, max_new=14):
        x = self.projector(self.vit(img))
        out = []
        for _ in range(max_new):
            logits, _ = self.lm(embeds=x)
            t = logits[:, -1].argmax(-1)
            out.append(t)
            x = torch.cat([x, self.lm.wte(t)[:, None]], 1)
        return torch.stack(out, 1)


def encode_captions(caps):
    L = max(len(c) for c in caps)
    return torch.tensor([list(c.encode()) + [0] * (L - len(c)) for c in caps])


def train(steps=300, batch=32, seed=0, recompute=False, freeze_vit=False):
    torch.manual_seed(seed)
    m = TinyVLM()
    m.vit.recompute = recompute
    if freeze_vit:
        for p in m.vit.parameters():
            p.requires_grad_(False)
    data = dataset(600, seed)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    rnd = random.Random(seed)
    for _ in range(steps):
        b = rnd.sample(data, batch)
        loss = m(torch.stack([x for x, _ in b]), encode_captions([c for _, c in b]))
        loss.backward()
        opt.step()
        opt.zero_grad()
    return m


def caption_accuracy(m, n=60, seed=99):
    test = dataset(n, seed)
    m.eval()
    out = m.caption(torch.stack([x for x, _ in test]))
    good = 0
    for row, (_, cap) in zip(out, test):
        text = bytes(int(t) for t in row).decode(errors="replace").split("\n")[0]
        good += text == cap.strip()
    m.train()
    return good / n


def saved_bytes(m, recompute: bool, freeze_vit: bool) -> int:
    """Activation bytes autograd keeps for one batch (Week 3's technique)."""
    m.vit.recompute = recompute
    for p in m.vit.parameters():
        p.requires_grad_(not freeze_vit)
    params = {p.untyped_storage().data_ptr() for p in m.parameters()}
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in params:
            seen[st.data_ptr()] = st.nbytes()
        return t

    b = dataset(16, 3)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = m(torch.stack([x for x, _ in b]), encode_captions([c for _, c in b]))
    del loss
    for p in m.vit.parameters():
        p.requires_grad_(True)
    m.vit.recompute = False
    return sum(seen.values())


if __name__ == "__main__":
    m = train()
    print(f"caption exact-match accuracy: {caption_accuracy(m):.0%}")
    for rc, fz in [(False, False), (True, False), (False, True)]:
        print(f"saved activations  recompute_vit={rc!s:5} freeze_vit={fz!s:5}: {saved_bytes(m, rc, fz)/1e6:.2f} MB")
