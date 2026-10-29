"""Collect one number per roadmap gate into GATES.md - the portfolio's single page.

Runs a quick version of each lab's measurement against your exercise code
(default) or the reference solutions (LAB=solution). Labs whose TODOs are not
done yet show as "not implemented". On a GPU machine the GPU-only gates run too.

Usage:  python tools/gates.py              # your work
        LAB=solution python tools/gates.py # reference numbers for this machine
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import pathlib
import platform
import sys
import tempfile
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common  # noqa: E402,F401

import torch  # noqa: E402

WHICH = os.environ.get("LAB", "exercise")


def load(rel: str):
    path = ROOT / "labs" / rel / f"{WHICH}.py"
    name = f"gate_{path.parent.name}_{WHICH}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def g01():
    m = load("phase1_foundations/w01_roofline")
    bw, fl = m.measure_bandwidth(64 * 2**20), m.measure_matmul_flops(512)
    return f"bandwidth {bw/1e9:.1f} GB/s, matmul {fl/1e12:.3f} TFLOP/s, ridge {fl/bw:.1f} FLOP/B"


def g02():
    m = load("phase1_foundations/w02_kernels")
    r = m.bench_sgemm(512 if not torch.cuda.is_available() else 4096)
    return f"SGEMM {r['% of library']:.0f}% of library at n={r['n']} (GPU gate: >= 70% of cuBLAS at 4096)"


def g04():
    m = load("phase1_foundations/w04_gpt_trainer")
    rows = m.ablation(m.Config(steps=60))
    return "; ".join(f"{r['variant']}: {r['tokens/s']:.0f} tok/s, MFU {r['MFU %']:.1f}%" for r in rows)


def g05():
    from common import dist as ldist

    m = load("phase2_distributed/w05_ddp_zero")
    ref = m.reference_train("adam")
    params, state = ldist.run(m.zero_train, 2)[0]
    err = max((a - b).abs().max().item() for a, b in zip(ref, params))
    return f"ZeRO-2 vs Adam max |diff| {err:.1e}; Adam state per rank {state} B"


def g06():
    from common import dist as ldist

    m = load("phase2_distributed/w06_tp_pp")
    o, g = ldist.run(m.tp_check, 2)[0]
    return f"TP=2 out diff {o:.1e}, dx diff {g:.1e}; 1F1B bubble p=4 m=16: {m.bubble_fraction(4, 16):.1%}"


def g07():
    from common import dist as ldist

    m = load("phase2_distributed/w07_context_parallel")
    ring = max(ldist.run(m.cp_check, 4, "ring").values())
    uly = max(ldist.run(m.cp_check, 4, "ulysses").values())
    return f"CP=4 ring diff {ring:.1e}, Ulysses diff {uly:.1e}"


def g08():
    from common import dist as ldist

    m = load("phase2_distributed/w08_moe_ep")
    ep = max(ldist.run(m.ep_check, 4).values())
    b, a = m.balance_demo()
    return f"EP=4 diff {ep:.1e}; load max/mean {b:.2f} -> {a:.2f} (aux-free)"


def g09():
    m = load("phase2_distributed/w09_fp8")
    r = m.compare_training(150)
    return f"FP8 vs fp32 mean-loss gap {r['rel_gap']:.1%} (CPU emulation; GPU gate < 0.25% with TE)"


def g11():
    m = load("phase3_kernels/w11_flash_attention")
    if torch.cuda.is_available():
        return m.benchmark_gpu()
    q, k, v = (torch.randn(2, 2, 64, 32) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return f"Triton FA correctness diff {(m.attention(q, k, v) - ref).abs().max().item():.1e} (speed gate needs GPU)"


def g12():
    m = load("phase3_kernels/w12_tilelang")
    C, ref = m.run_blockscaled(torch.randn(64, 128), torch.randn(128, 64))
    return f"TileLang block-scaled FP8 GEMM diff {(C - ref).abs().max().item():.1e}"


def g14():
    m = load("phase4_inference/w14_mini_engine")
    r = m.benchmark(8, 32, 16)
    return f"engine {r['engine tok/s']:.0f} tok/s, {r['speedup']:.1f}x naive (GPU gate: within 2x of vLLM)"


def g15():
    m = load("phase4_inference/w15_prefix_cache")
    rows = {r["trace"]: r for r in m.compare()}
    tr = m.multi_turn_trace(n_sessions=3, turns=4)
    a, b = m.ttft_experiment(tr, False), m.ttft_experiment(tr, True)
    return (f"prefill saved: multi-turn {rows['multi-turn']['radix saved']:.0%}, rag {rows['rag']['radix saved']:.0%}; "
            f"TTFT {a*1e3:.1f} -> {b*1e3:.1f} ms")


def g16():
    m = load("phase4_inference/w16_tiered_kv")
    sys.path.insert(0, str(ROOT / "labs/phase4_inference/w15_prefix_cache"))
    w15 = load("phase4_inference/w15_prefix_cache")
    mt = m.ttft_report(w15.multi_turn_trace(n_sessions=4, turns=3), caps=(8, 32, 4096))
    nr = m.ttft_report(w15.no_reuse_trace(n=8))
    return (f"multi-turn TTFT {mt['no cache']['median TTFT ms']:.1f} -> {mt['tiered']['median TTFT ms']:.1f} ms; "
            f"no-reuse overhead {nr['tiered']['median TTFT ms'] / nr['no cache']['median TTFT ms'] - 1:+.0%}")


def g17():
    m = load("phase4_inference/w17_pd_router")
    rows = {r["policy"]: r for r in m.compare()}
    rr, ca = rows["round_robin"], rows["cache_aware"]
    return (f"P50 TTFT rr {rr['P50 TTFT ms']:.1f} vs cache-aware {ca['P50 TTFT ms']:.1f} ms; "
            f"P99 {rr['P99 TTFT ms']:.0f} vs {ca['P99 TTFT ms']:.0f} ms; reuse {rr['prefix reuse']:.0%} vs {ca['prefix reuse']:.0%}")


def g18():
    m = load("phase5_rl/w18_grpo_sync")
    _, s = m.train(steps=30)
    tot = s["gen_s"] + s["train_s"]
    return f"GRPO acc {s['acc_start']:.0%} -> {s['acc_end']:.0%}; generation {s['gen_s']/tot:.0%} of time"


def g19():
    m = load("phase5_rl/w19_async_rl")
    s, a = m.train_sync(), m.train_async()
    return (f"async {s['wall_s']/a['wall_s']:.2f}x vs sync; acc sync {s['acc_end']:.0%} / async {a['acc_end']:.0%}; "
            f"max staleness {max(a['staleness'])}")


def g20():
    m = load("phase5_rl/w20_agent_rl")
    s = m.train(steps=15)
    return f"tool use {s['tool_rate'][0]:.0%} -> {s['tool_rate'][-1]:.0%}; reward {s['reward'][0]:.2f} -> {s['reward'][-1]:.2f}"


def g21():
    m = load("phase5_rl/w21_opd")
    t = m.make_teacher()
    rows = [m.run(x, 30, teacher=t) for x in ("grpo", "opd_full")]
    return "; ".join(f"{r['method']} {r['acc_start']:.0%} -> {r['acc_end']:.0%}" for r in rows)


def g22():
    m = load("phase6_multimodal/w22_tiny_vlm")
    v = m.train(steps=300)
    return (f"caption acc {m.caption_accuracy(v):.0%}; saved act. {m.saved_bytes(v, False, False)/1e6:.1f} MB -> "
            f"{m.saved_bytes(v, True, False)/1e6:.1f} MB (ViT recompute)")


def g23():
    m = load("phase6_multimodal/w23_mm_dataloading")
    sh = m.write_shards(1024, 8, tempfile.mkdtemp())
    a, b = m.measure_stall(sh, 0, step_compute_s=0.08), m.measure_stall(sh, 2, step_compute_s=0.08)
    g = torch.Generator().manual_seed(0)
    L = [int(torch.randint(24, 80, (1,), generator=g)) for _ in range(2000)]
    return (f"stall {a['stall %']:.0f}% -> {b['stall %']:.0f}% with 2 workers; token efficiency "
            f"{m.padding_efficiency(L, 32):.0%} padded -> {m.packing_efficiency(L, 256):.0%} packed")


GATES = [("W01 roofline", g01), ("W02 CUDA/C++ kernels", g02), ("W04 GPT trainer", g04), ("W05 DDP/ZeRO", g05),
         ("W06 TP/PP", g06), ("W07 context parallel", g07), ("W08 MoE/EP", g08), ("W09 FP8", g09),
         ("W11 FlashAttention", g11), ("W12 TileLang", g12), ("W14 engine", g14), ("W15 prefix cache", g15),
         ("W16 tiered KV", g16), ("W17 PD + router", g17), ("W18 GRPO", g18), ("W19 async RL", g19),
         ("W20 agent RL", g20), ("W21 OPD", g21), ("W22 VLM", g22), ("W23 data loading", g23)]


def main():
    only = set(sys.argv[1:])
    dev = torch.cuda.get_device_name() if torch.cuda.is_available() else f"CPU x{os.cpu_count()}"
    lines = [f"# Gate results ({WHICH})", "",
             f"{datetime.date.today()} · {dev} · torch {torch.__version__} · {platform.platform()}", "",
             "| Gate | Result |", "| --- | --- |"]
    for name, fn in GATES:
        if only and not any(o in name for o in only):
            continue
        try:
            res = fn()
        except NotImplementedError as e:
            res = f"not implemented ({str(e)[:60]})"
        except Exception as e:  # report and move on
            res = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''}"
            if os.environ.get("GATES_DEBUG"):
                traceback.print_exc()
        print(f"{name:24s} {res}", flush=True)
        lines.append(f"| {name} | {res} |")
    out = ROOT / f"GATES{'' if WHICH == 'exercise' else '_' + WHICH}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
