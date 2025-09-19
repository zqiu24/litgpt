import time
import statistics as stats
from typing import List, Tuple

import torch
import argparse

# Import your existing and fused implementations
try:
    from litgpt.oft import cayley_batch, cayley_batch_optimized
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "litgpt"))
    from litgpt.oft import cayley_batch, cayley_batch_optimized


def _make_skew(B: int, D: int, device, dtype) -> torch.Tensor:
    A = torch.randn(B, D, D, device=device, dtype=dtype)
    return 0.5 * (A - A.transpose(-2, -1))


def _bench_once(Q: torch.Tensor, D: int, num_terms: int, impl: str, sync: bool) -> Tuple[float, float]:
    # Pick forward
    if impl == "baseline":
        def forward():
            return cayley_batch(Q, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
    elif impl == "fused":
        def forward():
            return cayley_batch_optimized(Q, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
    else:
        raise ValueError("impl must be 'baseline' or 'fused'")

    # Forward timing
    if sync and Q.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    R = forward()
    if sync and Q.is_cuda:
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0

    # Backward timing
    if Q.grad is not None:
        Q.grad = None
    loss = R.sum()
    if sync and Q.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    if sync and Q.is_cuda:
        torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t1) * 1000.0

    return fwd_ms, bwd_ms


def bench_cayley_neumann(
    batch_sizes: List[int],
    dims: List[int],
    num_terms: int = 5,
    repeats: int = 100,
    warmup: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float32",
    verify: bool = True,
    sync: bool = True,
) -> None:
    dev = torch.device(device)
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"Device={dev}, dtype={dt}, repeats={repeats}, warmup={warmup}, terms={num_terms}")
    print(f"{'D':>4} {'B':>4} {'impl':>9} {'fwd(ms)':>10} {'bwd(ms)':>10} {'total(ms)':>10}")

    for D in dims:
        for B in batch_sizes:
            # One base input; clone per impl to keep graphs separate
            Q0 = _make_skew(B, D, dev, dt)

            if verify:
                with torch.no_grad():
                    R_base = cayley_batch(Q0, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
                    R_fused = cayley_batch_optimized(Q0, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
                    max_diff = (R_base - R_fused).abs().max().item()
                # Grad check (separate graphs)
                Qb = Q0.detach().clone().requires_grad_(True)
                Rb = cayley_batch(Qb, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
                Rb.sum().backward()
                Gb = Qb.grad.detach().clone()

                Qf = Q0.detach().clone().requires_grad_(True)
                Rf = cayley_batch_optimized(Qf, D, use_cayley_neumann=True, num_neumann_terms=num_terms)
                Rf.sum().backward()
                Gf = Qf.grad.detach().clone()

                g_diff = (Gb - Gf).abs().max().item()
                print(f"{D:>4} {B:>4} {'verify':>9} max|ΔR|={max_diff:.3e} max|ΔG|={g_diff:.3e}")

            # Warmup
            for _ in range(warmup):
                for impl in ("baseline", "fused"):
                    Q = Q0.detach().clone().requires_grad_(True)
                    _f, _b = _bench_once(Q, D, num_terms, impl, sync)

            # Timed runs
            results = {}
            for impl in ("baseline", "fused"):
                fwds, bwds = [], []
                for _ in range(repeats):
                    Q = Q0.detach().clone().requires_grad_(True)
                    f, b = _bench_once(Q, D, num_terms, impl, sync)
                    fwds.append(f); bwds.append(b)
                results[impl] = (stats.mean(fwds), stats.mean(bwds))

            # for impl in ("baseline", "fused"):
            #     print(f"{D:>4} {B:>4} {impl:>9} fwd={results[impl][0]:>7.3f} bwd={results[impl][1]:>7.3f}")

            # Ratios fused vs baseline (p/b style)
            fwd_ratio = results["baseline"][0] / results["fused"][0]
            bwd_ratio = results["baseline"][1] / results["fused"][1]
            total_ratio = (results["baseline"][0] + results["baseline"][1]) / (results["fused"][0] + results["fused"][1])
            print(f"{D:>4} {B:>4} {'ratio b/f':>9} fwd={fwd_ratio:>7.3f} bwd={bwd_ratio:>7.3f} total={total_ratio:>7.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--batch-sizes", type=str, default="8,32,128,256,512",
                   help="Comma-separated list, e.g. 1,8,32,128")
    p.add_argument("--block-sizes", type=str, default="128,256,512",
                   help="Comma-separated list, e.g. 64,128,256")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--no-sync", action="store_true", help="Do not cuda.synchronize around timers")
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    block_sizes = [int(x) for x in args.block_sizes.split(",") if x]


    bench_cayley_neumann(
        batch_sizes=[8, 32, 128, 256],
        dims=[128, 256, 512],
        num_terms=5,
        repeats=100,
        warmup=50,
        device="cuda",
        # dtype="float32",
        dtype="bfloat16",
        verify=False,
        sync=True,
    )


if __name__ == "__main__":
    main()