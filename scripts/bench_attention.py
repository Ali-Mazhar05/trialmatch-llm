"""Task 4 benchmark (rubric E3): reference vs tiled online-softmax vs PyTorch SDPA.

Numerical part: several shapes, causal and non-causal, decode-style
(Tq < Tk, bottom-right mask), fp32 and fp16; max / mean absolute error against
the fp64 reference. Performance part: median time and peak extra memory.

    python scripts/bench_attention.py
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trialmatch.attention import reference_attention, sdpa_attention, tiled_attention  # noqa: E402
from trialmatch.bench_utils import env_info, save  # noqa: E402


def timed(fn, dev, repeats):
    fn()  # warm-up
    ts, mems = [], []
    for _ in range(repeats):
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t = time.perf_counter(); fn()
        if dev == "cuda":
            torch.cuda.synchronize(); mems.append((torch.cuda.max_memory_allocated() - base) / 2**20)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), (statistics.median(mems) if mems else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "bench" / "attention.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---------------- numerical validation
    num = []
    shapes = [(1, 1, 16, 16, 8), (2, 4, 100, 100, 64), (1, 12, 257, 257, 128), (1, 12, 1, 300, 128),
              (1, 12, 7, 300, 128), (2, 8, 512, 512, 64)]
    dtypes = [torch.float32] + ([torch.float16] if dev == "cuda" else [])
    for (B, H, tq, tk, d) in shapes:
        for causal in [True, False]:
            for dt in dtypes:
                q = torch.randn(B, H, tq, d, device=dev, dtype=dt)
                k = torch.randn(B, H, tk, d, device=dev, dtype=dt)
                v = torch.randn(B, H, tk, d, device=dev, dtype=dt)
                gold = reference_attention(q.double(), k.double(), v.double(), causal=causal).double()
                for name, fn in [("reference", reference_attention),
                                 ("tiled_b32", lambda *a, **kw: tiled_attention(*a, block_q=32, block_k=32, **kw)),
                                 ("tiled_b128", lambda *a, **kw: tiled_attention(*a, block_q=128, block_k=128, **kw)),
                                 ("sdpa", sdpa_attention)]:
                    err = (fn(q, k, v, causal=causal).double() - gold).abs()
                    num.append({"shape": [B, H, tq, tk, d], "causal": causal, "dtype": str(dt), "impl": name,
                                "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())})
    worst = {}
    for r in num:
        key = (r["impl"], r["dtype"])
        worst[key] = max(worst.get(key, 0), r["max_abs_err"])
    for (impl, dt), e in sorted(worst.items()):
        print(f"{impl:11s} {dt:14s} worst max|err| vs fp64 = {e:.2e}")

    # ---------------- performance (causal prefill-style, Qwen2.5-1.5B head geometry)
    perf = []
    dt = torch.float16 if dev == "cuda" else torch.float32
    seqs = [128, 512, 1024, 2048, 4096] if dev == "cuda" else [128, 512, 1024]
    for T in seqs:
        q = torch.randn(1, 12, T, 128, device=dev, dtype=dt)
        k = torch.randn(1, 12, T, 128, device=dev, dtype=dt)
        v = torch.randn(1, 12, T, 128, device=dev, dtype=dt)
        for name, fn in [("reference", lambda: reference_attention(q, k, v)),
                         ("tiled_b64", lambda: tiled_attention(q, k, v, block_q=64, block_k=64)),
                         ("tiled_b128", lambda: tiled_attention(q, k, v, block_q=128, block_k=128)),
                         ("sdpa", lambda: sdpa_attention(q, k, v))]:
            try:
                t, m = timed(fn, dev, args.repeats)
            except torch.OutOfMemoryError:
                t, m = None, "OOM"
                torch.cuda.empty_cache()
            perf.append({"seq_len": T, "impl": name, "dtype": str(dt), "median_ms": t * 1e3 if t else None,
                         "peak_extra_mem_mb": m})
            print(f"T={T:5d} {name:11s} {t*1e3 if t else float('nan'):9.2f} ms  mem={m}")
    backend = None
    if dev == "cuda":
        backend = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                   "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                   "math": torch.backends.cuda.math_sdp_enabled()}
    save({"env": env_info(), "sdpa_backends_enabled": backend, "numerical": num, "performance": perf}, args.out)


if __name__ == "__main__":
    main()
