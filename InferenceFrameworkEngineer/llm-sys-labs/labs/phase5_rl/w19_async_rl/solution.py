"""Week 19: asynchronous RL - decouple rollout from training.

Roadmap: "Read AReaL and slime's fully-async example. Decouple rollout and
training, add interruptible rollouts and a staleness bound; compare against
the sync baseline."
Gate: sync vs async GRPO on the same hardware - wall-clock to a fixed reward,
utilisation, and the staleness setting used.

Architecture (AReaL's, on threads instead of GPU pools):
  WeightStore     - latest policy weights + a version number
  rollout thread  - loads the newest weights *between groups* (interruptible
                    at group granularity), generates, records the behaviour
                    policy's log-probs and the version it used, pushes to a
                    bounded queue (the queue bound is back-pressure)
  trainer         - pops batches, DROPS any whose version is older than
                    `max_staleness` updates, trains with an importance ratio
                    against the behaviour policy, publishes new weights

Staleness-aware objective: the PPO ratio is taken against the policy that
*generated* the data (logp_behav), clipped as usual. With max_staleness=0
this reduces to synchronous on-policy GRPO.

CPU note: PyTorch ops release the GIL, so the two threads really do overlap on
a multi-core CPU. The gain is bounded by cores; on GPUs the two halves run on
separate device pools and the gain is the fraction of time each side idled.
"""
from __future__ import annotations

import copy
import pathlib
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from common.rl_task import accuracy, completion_mask, make_problem, new_policy, reward, sample, sft, token_logprobs


def group_advantages(r, eps=1e-4):
    return (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)


@dataclass
class Batch:
    seqs: torch.Tensor
    adv: torch.Tensor
    mask: torch.Tensor
    logp_behav: torch.Tensor
    version: int
    mean_reward: float


class WeightStore:
    def __init__(self, model):
        self.lock = threading.Lock()
        self.state = copy.deepcopy(model.state_dict())
        self.version = 0

    def publish(self, model):
        # >>> SOLUTION: under the lock, copy the model's state_dict and bump the version
        with self.lock:
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.version += 1
        # <<< SOLUTION

    def pull(self, model, have: int) -> int:
        """Load newer weights into `model` if available; return the version now held."""
        # >>> SOLUTION: if self.version > have, load_state_dict under the lock and return the new version; else return have
        with self.lock:
            if self.version > have:
                model.load_state_dict(self.state)
                return self.version
        return have
        # <<< SOLUTION


def rollout_loop(store: WeightStore, q: queue.Queue, stop: threading.Event, P: int, G: int, seed: int, stats: dict):
    """Thread entry point: any exception is recorded so the trainer fails fast
    instead of waiting on an empty queue."""
    try:
        _rollout(store, q, stop, P, G, seed, stats)
    except BaseException as e:  # noqa: BLE001
        stats["worker_error"] = e
        stop.set()


def _rollout(store: WeightStore, q: queue.Queue, stop: threading.Event, P: int, G: int, seed: int, stats: dict):
    model = new_policy()
    version = store.pull(model, -1)
    rnd = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    while not stop.is_set():
        version = store.pull(model, version)  # interruptible: switch weights between groups
        prompts = [make_problem(rnd)[0] for _ in range(P)]
        t0 = time.perf_counter()
        seqs = sample(model, prompts, G, temperature=1.0, generator=gen)
        rew = torch.tensor([[reward(p, seqs[i * G + g, 6:].tolist()) for g in range(G)] for i, p in enumerate(prompts)])
        with torch.no_grad():
            lpb = token_logprobs(model, seqs)
        stats["gen_s"] += time.perf_counter() - t0
        b = Batch(seqs, group_advantages(rew).flatten(), completion_mask(seqs), lpb, version, rew.mean().item())
        while not stop.is_set():
            try:
                q.put(b, timeout=0.05)
                break
            except queue.Full:
                continue


def async_loss(logp, logp_behav, logp_ref, adv, mask, clip=0.2, beta=0.04):
    # >>> SOLUTION: ratio vs behaviour policy, PPO clip, k3 KL to ref, token mean over mask
    ratio = torch.exp(logp - logp_behav)
    a = adv[:, None]
    surr = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    d = logp_ref - logp
    per = surr + beta * (torch.exp(d) - d - 1)
    return (per * mask).sum() / mask.sum().clamp(min=1)
    # <<< SOLUTION


def train_async(updates: int = 30, P: int = 16, G: int = 8, max_staleness: int = 2, sft_steps: int = 300,
                lr: float = 5e-4, seed: int = 0, queue_size: int = 2) -> dict:
    policy = sft(new_policy(seed), sft_steps, seed=seed)
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    store = WeightStore(policy)
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    stats = {"gen_s": 0.0, "train_s": 0.0, "dropped": 0, "staleness": [], "acc_start": accuracy(policy)}
    worker = threading.Thread(target=rollout_loop, args=(store, q, stop, P, G, seed, stats), daemon=True)
    t_start = time.perf_counter()
    worker.start()
    done = 0
    try:
        while done < updates:
            try:
                b = q.get(timeout=0.5)
            except queue.Empty:
                if "worker_error" in stats:
                    raise RuntimeError(f"rollout thread failed: {stats['worker_error']!r}") from stats["worker_error"]
                if not worker.is_alive():
                    raise RuntimeError("rollout thread exited")
                continue
            # >>> SOLUTION: staleness = store.version - b.version; drop (count) if > max_staleness; else compute loss vs ref, step, publish
            stale = store.version - b.version
            if stale > max_staleness:
                stats["dropped"] += 1
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                lref = token_logprobs(ref, b.seqs)
            loss = async_loss(token_logprobs(policy, b.seqs), b.logp_behav, lref, b.adv, b.mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            store.publish(policy)
            stats["train_s"] += time.perf_counter() - t0
            stats["staleness"].append(stale)
            done += 1
            # <<< SOLUTION
    finally:
        stop.set()
        worker.join(timeout=30)
    stats["wall_s"] = time.perf_counter() - t_start
    stats["acc_end"] = accuracy(policy)
    return stats


def train_sync(updates: int = 30, P: int = 16, G: int = 8, sft_steps: int = 300, lr: float = 5e-4, seed: int = 0) -> dict:
    """Same pipeline with max_staleness=0 and no overlap: generate, then train."""
    policy = sft(new_policy(seed), sft_steps, seed=seed)
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rnd = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    stats = {"gen_s": 0.0, "train_s": 0.0, "acc_start": accuracy(policy)}
    t_start = time.perf_counter()
    for _ in range(updates):
        prompts = [make_problem(rnd)[0] for _ in range(P)]
        t0 = time.perf_counter()
        seqs = sample(policy, prompts, G, temperature=1.0, generator=gen)
        rew = torch.tensor([[reward(p, seqs[i * G + g, 6:].tolist()) for g in range(G)] for i, p in enumerate(prompts)])
        with torch.no_grad():
            lpb, lref = token_logprobs(policy, seqs), token_logprobs(ref, seqs)
        t1 = time.perf_counter()
        loss = async_loss(token_logprobs(policy, seqs), lpb, lref, group_advantages(rew).flatten(), completion_mask(seqs))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        stats["gen_s"] += t1 - t0
        stats["train_s"] += time.perf_counter() - t1
    stats["wall_s"] = time.perf_counter() - t_start
    stats["acc_end"] = accuracy(policy)
    return stats


if __name__ == "__main__":
    s = train_sync()
    a = train_async()
    print(f"sync : wall {s['wall_s']:.1f}s  acc {s['acc_start']:.1%} -> {s['acc_end']:.1%}")
    print(f"async: wall {a['wall_s']:.1f}s  acc {a['acc_start']:.1%} -> {a['acc_end']:.1%}  "
          f"staleness used {sorted(set(a['staleness']))}  dropped {a['dropped']}")
    print(f"async speedup {s['wall_s'] / a['wall_s']:.2f}x  (bounded here by {torch.get_num_threads()} intra-op threads / CPU cores)")
