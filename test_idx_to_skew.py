import time
import argparse
import statistics as stats
from typing import Tuple, List

import torch

# Try import from your repo layout; fallback by adding repo root
try:
    from litgpt.oft import (
        pytorch_skew_symmetric,
        SkewSymmetricFunction,
        SkewSymmetricFunction_optimized,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "litgpt"))
    from litgpt.oft import (
        pytorch_skew_symmetric,
        SkewSymmetricFunction,
        SkewSymmetricFunction_optimized,
    )


def get_indices(block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.triu_indices(block_size, block_size, offset=1, device=device)
    return idx[0], idx[1]  # rows, cols


def bench_once(vec: torch.Tensor, block_size: int, rows: torch.Tensor, cols: torch.Tensor,
               impl: str, device: torch.device, sync: bool,
               idx_ul: torch.Tensor = None) -> Tuple[float, float]:
    if impl == "pytorch":
        def forward():
            return pytorch_skew_symmetric(vec, block_size, rows, cols)
    elif impl == "custom":
        def forward():
            return SkewSymmetricFunction.apply(vec, block_size, rows, cols)
    elif impl == "optimized":
        def forward():
            return SkewSymmetricFunction_optimized.apply(vec, block_size, rows, cols, idx_ul)
    else:
        raise ValueError("impl must be 'pytorch', 'custom', or 'optimized'")

    # Forward timing
    if sync and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = forward()
    if sync and device.type == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0

    # Backward timing
    if vec.grad is not None:
        vec.grad = None
    loss = out.sum()
    if sync and device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    if sync and device.type == "cuda":
        torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t1) * 1000.0

    return fwd_ms, bwd_ms


def run_bench(batch_sizes: List[int], block_sizes: List[int], repeats: int, warmup: int,
              device: str, dtype: str, verify: bool, sync: bool):
    dev = torch.device(device)
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    print(f"Device={dev}, dtype={dt}, repeats={repeats}, warmup={warmup}")
    print(f"{'B':>4} {'bs':>4} {'K':>8} {'impl':>10} {'fwd(ms)':>10} {'bwd(ms)':>10}")

    for B in block_sizes:
        rows, cols = get_indices(B, dev)
        # Precompute and reuse linear indices
        idx_u = rows * B + cols
        idx_l = cols * B + rows
        idx_ul = torch.cat([idx_u, idx_l], dim=0)
        K = rows.numel()

        for bs in batch_sizes:
            vec = torch.randn(bs, K, device=dev, dtype=dt, requires_grad=True)

            if verify:
                with torch.no_grad():
                    ref = pytorch_skew_symmetric(vec, B, rows, cols)
                    cus = SkewSymmetricFunction.apply(vec, B, rows, cols)
                    opt = SkewSymmetricFunction_optimized.apply(vec, B, rows, cols, idx_ul)
                    max_diff_c = (ref - cus).abs().max().item()
                    max_diff_o = (ref - opt).abs().max().item()
                print(f"{B:>4} {bs:>4} {K:>8} {'verify':>10} max|Δ| c={max_diff_c:.3e} o={max_diff_o:.3e}")

            # Warmup
            for _ in range(warmup):
                for impl in ("pytorch", "custom", "optimized"):
                    fwd_ms, bwd_ms = bench_once(vec, B, rows, cols, impl, dev, sync, idx_ul)
                    if vec.grad is not None:
                        vec.grad = None

            # Timed runs
            results = {}
            for impl in ("pytorch", "custom", "optimized"):
                fwds, bwds = [], []
                for _ in range(repeats):
                    fwd_ms, bwd_ms = bench_once(vec, B, rows, cols, impl, dev, sync, idx_ul)
                    fwds.append(fwd_ms); bwds.append(bwd_ms)
                    if vec.grad is not None:
                        vec.grad = None
                results[impl] = (stats.mean(fwds), stats.mean(bwds))

            # Print per-impl times
            # for impl in ("pytorch", "custom", "optimized"):
            #     print(f"{B:>4} {bs:>4} {K:>8} {impl:>10} fwd={results[impl][0]:>7.3f} bwd={results[impl][1]:>7.3f}")

            # Ratios vs pytorch
            # fwd_co = results["custom"][0] / results["optimized"][0]
            # bwd_co = results["custom"][1] / results["optimized"][1]
            # print(f"{B:>4} {bs:>4} {K:>8} {'ratio':>10} c/o fwd={fwd_co:>7.3f} bwd={bwd_co:>7.3f}")
            fwd_po = results["pytorch"][0] / results["optimized"][0]
            bwd_po = results["pytorch"][1] / results["optimized"][1]
            print(f"{B:>4} {bs:>4} {K:>8} {'ratio':>10} p/o fwd={fwd_po:>7.3f} bwd={bwd_po:>7.3f}")


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

    run_bench(
        batch_sizes=batch_sizes,
        block_sizes=block_sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        device=args.device,
        dtype=args.dtype,
        verify=args.verify,
        sync=not args.no_sync,
    )


if __name__ == "__main__":
    main()