import time
import torch
# from litgpt.oft import OptimizedCustomLinear
from litgpt.oft import torch_bmm, frozen_linear
import argparse

def custom_forward(x, Rin, W, b, Rout, bsz):
    # x: [B,S,rin*bsz]
    # Rin: [rin,bsz,bsz]
    # W: [Dout, Din]=(rout*bsz, rin*bsz)
    # b: [Dout] or None
    # Rout: [rout,bsz,bsz]
    B, S, Din = x.shape
    rin = Rin.size(0)
    rout = Rout.size(0)
    assert Din == rin * bsz and W.shape == (rout * bsz, rin * bsz)
    N = B * S

    # 1) xR: per-block right multiply via batched bmm (rin is the batch dim)
    # xb = x.contiguous().view(N, rin, bsz)                  # [N, rin, b]
    xb = x.view(N, rin, bsz)
    # xb_r = xb.transpose(0, 1).contiguous()                 # [rin, N, b]
    xb_r = xb.transpose(0, 1)
    xR_r = torch.bmm(xb_r, Rin)                            # [rin, N, b] @ [rin, b, b] -> [rin, N, b]
    # xR = xR_r.transpose(0, 1).contiguous().view(B, S, rin, bsz)  # [B,S,rin,b]
    xR = xR_r.transpose(0, 1).view(B, S, rin, bsz)

    # 2) yb = (xR_flat @ W^T), single large GEMM
    xR_flat = xR.contiguous().view(N, rin * bsz)           # [N, Din]
    # xR_flat = xR.view(N, rin * bsz)
    yb_flat = xR_flat @ W.t()                               # [N, Dout]
    # yb_flat = frozen_linear(xR_flat, W, b)
    if b is not None:
        yb_flat = yb_flat + b                               # broadcast

    # 3) out = apply Rout per block via strided batched bmm
    yb = yb_flat.view(N, rout, bsz)                         # [N, rout, b]
    # yb_t = yb.transpose(0, 1).contiguous()                  # [rout, N, b]
    yb_t = yb.transpose(0, 1)
    out_t = torch.bmm(yb_t, Rout)                           # [rout, N, b]
    y = out_t.transpose(0, 1).contiguous().view(B, S, rout * bsz)
    return y


class OptimizedCustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Rin, W, b, Rout, bsz: int) -> torch.Tensor:
        # x: [B,S,rin*bsz]
        # Rin: [rin,bsz,bsz]
        # W: [Dout, Din]=(rout*bsz, rin*bsz)
        # b: [Dout] or None
        # Rout: [rout,bsz,bsz]
        B, S, Din = x.shape
        rin = Rin.size(0)
        rout = Rout.size(0)
        assert Din == rin * bsz and W.shape == (rout * bsz, rin * bsz)
        N = B * S

        # 1) xR: per-block right multiply via batched bmm (rin is the batch dim)
        # xb = x.contiguous().view(N, rin, bsz)                  # [N, rin, b]
        xb = x.view(N, rin, bsz)
        # xb_r = xb.transpose(0, 1).contiguous()                 # [rin, N, b]
        xb_r = xb.transpose(0, 1)
        xR_r = torch.bmm(xb_r, Rin)                            # [rin, N, b] @ [rin, b, b] -> [rin, N, b]
        # xR = xR_r.transpose(0, 1).contiguous().view(B, S, rin, bsz)  # [B,S,rin,b]
        xR = xR_r.transpose(0, 1).view(B, S, rin, bsz)

        # 2) yb = (xR_flat @ W^T), single large GEMM
        xR_flat = xR.contiguous().view(N, rin * bsz)           # [N, Din]
        # xR_flat = xR.view(N, rin * bsz)
        yb_flat = xR_flat @ W.t()                               # [N, Dout]
        if b is not None:
            yb_flat = yb_flat + b                               # broadcast

        # 3) out = apply Rout per block via strided batched bmm
        yb = yb_flat.view(N, rout, bsz)                         # [N, rout, b]
        # yb_t = yb.transpose(0, 1).contiguous()                  # [rout, N, b]
        yb_t = yb.transpose(0, 1)
        out_t = torch.bmm(yb_t, Rout)                           # [rout, N, b]
        y = out_t.transpose(0, 1).contiguous().view(B, S, rout * bsz)
        # y = out_t.transpose(0, 1).view(B, S, rout * bsz)

        ctx.save_for_backward(x, Rin, Rout)
        ctx.W = W
        ctx.has_bias = b is not None
        ctx.bsz, ctx.rin, ctx.rout = int(bsz), int(rin), int(rout)
        return y
    
    '''
    @staticmethod
    def forward(ctx, x, Rin, W, b, Rout, bsz: int) -> torch.Tensor:
        # x: [B,S,rin*bsz]
        # Rin: [rin,bsz,bsz]
        # W: [Dout, Din]=(rout*bsz, rin*bsz)
        # b: [Dout] or None
        # Rout: [rout,bsz,bsz]
        B, S, Din = x.shape
        rin = Rin.size(0)
        rout = Rout.size(0)
        assert Din == rin * bsz and W.shape == (rout * bsz, rin * bsz)

        xb = x.contiguous().view(B, S, rin, bsz)                         # [B,S,rin,b]
        xR = torch.einsum('bsik,ikc->bsic', xb, Rin)                     # [B,S,rin,b]

        # Use W^T for block view: Wt = [Din, Dout] -> [rin,b,rout,b]
        Wtb = W.t().contiguous().view(rin, bsz, rout, bsz)               # [rin,b,rout,b]
        yb  = torch.einsum('bsik,ikjc->bsjc', xR, Wtb)                   # [B,S,rout,b]

        if b is not None:
            yb = yb + b.view(1, 1, rout, bsz)

        outb = torch.einsum('bsjc,jcd->bsjd', yb, Rout)                  # [B,S,rout,b]
        y = outb.contiguous().view(B, S, rout * bsz)

        ctx.save_for_backward(x, Rin, Rout)
        ctx.W = W
        ctx.has_bias = b is not None
        ctx.bsz, ctx.rin, ctx.rout = int(bsz), int(rin), int(rout)
        return y

    '''
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, Rin, Rout = ctx.saved_tensors
        W = ctx.W                                           # [Dout, Din]
        bsz, rin, rout = ctx.bsz, ctx.rin, ctx.rout
        B, S, _ = grad_out.shape
        N = B * S

        # Views
        xb = x.contiguous().view(B, S, rin, bsz)            # [B,S,rin,b]

        # 1) xR = xb @ Rin (per-block)
        xR = torch.einsum('bsik,ikc->bsic', xb, Rin)        # [B,S,rin,b]

        # 2) yb = (xR_flat @ W^T), single large GEMM
        xR_flat = xR.contiguous().view(N, rin * bsz)        # [N, Din]
        # xR_flat = xR.view(N, rin * bsz)
        yb_flat = xR_flat @ W.t()                           # [N, Dout]
        # yb = yb_flat.view(B, S, rout, bsz)                  # [B,S,rout,b]

        # 3) grad_Rout via batched outer products aggregated over N
        go = grad_out.contiguous().view(N, rout, bsz)       # [N,rout,b]
        yb_r = yb_flat.view(N, rout, bsz).transpose(0, 1)   # [rout,N,b]
        go_r = go.transpose(0, 1).contiguous()              # [rout,N,b]
        grad_Rout = torch.bmm(yb_r.transpose(1, 2), go_r)   # [rout,b,b]

        # 4) grad_yb = go @ Rout^T (per block), batched bmm
        RoutT = Rout.transpose(-2, -1).contiguous()         # [rout,b,b]
        grad_yb_r = torch.bmm(go_r, RoutT)                  # [rout,N,b]
        grad_yb = grad_yb_r.transpose(0, 1).contiguous().view(B, S, rout, bsz)  # [B,S,rout,b]

        # 5) grad_xR = (grad_yb_flat @ W), single large GEMM
        grad_yb_flat = grad_yb.contiguous().view(N, rout * bsz)  # [N, Dout]
        grad_xR_flat = grad_yb_flat @ W                               # [N, Din]
        grad_xR = grad_xR_flat.view(B, S, rin, bsz)                   # [B,S,rin,b]

        # 6) grad_Rin via batched accumulation over N
        xb_r = xb.contiguous().view(N, rin, bsz).transpose(0, 1)      # [rin,N,b]
        gx_r = grad_xR.contiguous().view(N, rin, bsz).transpose(0, 1) # [rin,N,b]
        grad_Rin = torch.bmm(xb_r.transpose(1, 2), gx_r)              # [rin,b,b]

        # 7) grad_x = (grad_xR_i @ Rin_i^T) per i, batched bmm
        Rint = Rin.transpose(-2, -1).contiguous()                     # [rin,b,b]
        grad_xb_r = torch.bmm(gx_r, Rint)                             # [rin,N,b]
        grad_x = grad_xb_r.transpose(0, 1).contiguous().view(B, S, rin * bsz)

        return grad_x, grad_Rin, None, None, grad_Rout, None
    '''
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, Rin, Rout = ctx.saved_tensors
        W = ctx.W                                           # [Dout, Din]
        bsz, rin, rout = ctx.bsz, ctx.rin, ctx.rout
        B, S, _ = grad_out.shape

        xb  = x.contiguous().view(B, S, rin, bsz)           # [B,S,rin,b]
        xR  = torch.einsum('bsik,ikc->bsic', xb, Rin)       # [B,S,rin,b]
        Wb = W.contiguous().view(rout, bsz, rin, bsz)       # [rout,b,rin,b]
        Wtb = W.t().contiguous().view(rin, bsz, rout, bsz)  # [rin,b,rout,b]
        yb  = torch.einsum('bsik,ikjc->bsjc', xR, Wtb)      # [B,S,rout,b]

        go_b = grad_out.contiguous().view(B, S, rout, bsz)  # [B,S,rout,b]

        # dL/dRout and grad wrt yb
        # [B,S,rout,b] @ [B,S,rout,b] -> [rout,b,b]
        grad_Rout = torch.einsum('bsjc,bsjd->jcd', yb, go_b)                 # [rout,b,b]
        grad_yb   = torch.einsum('bsjd,jdc->bsjc', go_b, Rout.transpose(-2,-1))  # [B,S,rout,b]

        # dL/dRin and dL/dx
        # [B,S,rout,b] @ [rout,b,rin,b] -> [B,S,rin,b]
        grad_xR = torch.einsum('bsjc,jcdf->bsdf', grad_yb, Wb)             # [B,S,rin,b]
        grad_Rin = torch.einsum('bsik,bsic->ikc', xb, grad_xR)              # [rin,b,b]

        # [B,S,rout,b] @ [rout,b,rin,b] -> [B,S,rin,b]
        grad_ = torch.einsum('bsjc,jcdf->bsdf', grad_yb, Wb)             # [B,S,rin,b]
        # [B,S,rin,b] @ [rin,b,b] -> [B,S,rin,b]
        grad_xb = torch.einsum('bsdf,dfc->bsdc', grad_, Rin.transpose(-2, -1))             # [B,S,rin,b]
        grad_x = grad_xb.contiguous().view(B, S, rin * bsz)               # [B,S,Din]

        # Return grads for (x, Rin, W, b, Rout, bsz)
        return grad_x, grad_Rin, None, None, grad_Rout, None
    '''

def benchmark_custom_vs_native(
    B=1, S=2048, Din=2048, Dout=4096, bsz=256, iters=10, warmup=5, device=None, dtype=None, seed=0
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    torch.manual_seed(seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    rin = Din // bsz
    rout = Dout // bsz

    # Parameters (W is frozen)
    x0    = torch.randn(B, S, Din, device=device, dtype=dtype)
    Rin0  = torch.randn(rin, bsz, bsz, device=device, dtype=dtype) * 0.01
    W     = torch.randn(Dout, Din, device=device, dtype=dtype) * 0.02
    Rout0 = torch.randn(rout, bsz, bsz, device=device, dtype=dtype) * 0.01

    def make_leaves():
        x    = x0.detach().clone().requires_grad_(True)
        Rin  = Rin0.detach().clone().requires_grad_(True)
        Rout = Rout0.detach().clone().requires_grad_(True)
        return x, Rin, Rout

    def native_step():
        x, Rin, Rout = make_leaves()
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        x_rot = torch_bmm(x, Rin, bsz)
        y = frozen_linear(x_rot, W, None)
        y = torch_bmm(y, Rout, bsz)
        if device == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        y.sum().backward()
        if device == "cuda": torch.cuda.synchronize()
        t2 = time.time()
        return (t1 - t0), (t2 - t1), y.detach(), x.grad.detach(), Rin.grad.detach(), Rout.grad.detach()

    def custom_step():
        x, Rin, Rout = make_leaves()
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        y = custom_forward(x, Rin, W.detach(), None, Rout, Rout.size(-1))
        # y = OptimizedCustomLinear.apply(x, Rin, W.detach(), None, Rout, Rout.size(-1))
        if device == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        y.sum().backward()
        if device == "cuda": torch.cuda.synchronize()
        t2 = time.time()
        return (t1 - t0), (t2 - t1), y.detach(), x.grad.detach(), Rin.grad.detach(), Rout.grad.detach()

    # Warmup
    for _ in range(warmup):
        _ = native_step()
        _ = custom_step()

    # Correctness (single run)
    _, _, y0, gx0, gRin0, gRout0 = native_step()
    _, _, y1, gx1, gRin1, gRout1 = custom_step()

    def max_diff(a, b):
        return (a - b).abs().max().item()

    print("Correctness (max abs diff):")
    print(f"- output     : {max_diff(y0, y1):.3e}")
    print(f"- grad_x     : {max_diff(gx0, gx1):.3e}")
    print(f"- grad_R_in  : {max_diff(gRin0, gRin1):.3e}")
    print(f"- grad_R_out : {max_diff(gRout0, gRout1):.3e}")

    # Timing
    fwd_n, bwd_n, fwd_c, bwd_c = [], [], [], []
    for _ in range(iters):
        fn, bn, *_ = native_step()
        fc, bc, *_ = custom_step()
        fwd_n.append(fn); bwd_n.append(bn)
        fwd_c.append(fc); bwd_c.append(bc)

    print(f"\nTimings over {iters} iters (B={B}, S={S}, Din={Din}, Dout={Dout}, dtype={dtype}, device={device}):")
    print(f"- native forward avg: {sum(fwd_n)/iters*1000:.2f} ms, backward avg: {sum(bwd_n)/iters*1000:.2f} ms")
    print(f"- custom forward avg: {sum(fwd_c)/iters*1000:.2f} ms, backward avg: {sum(bwd_c)/iters*1000:.2f} ms")

    # Optional: separate forward/backward peak memory (CUDA)
    if device == "cuda":
        torch.cuda.synchronize()

        def native_forward_only():
            x, Rin, Rout = make_leaves()
            x_rot = torch_bmm(x, Rin, bsz)
            y = frozen_linear(x_rot, W, None)
            y = torch_bmm(y, Rout, bsz)
            return y

        def custom_forward_only():
            x, Rin, Rout = make_leaves()
            # return OptimizedCustomLinear.apply(x, Rin, W.detach(), None, Rout, Rout.size(-1))
            return custom_forward(x, Rin, W.detach(), None, Rout, Rout.size(-1))

        def measure_forward_peak(forward_fn):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            y = forward_fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return (peak - base), y

        def measure_backward_peak(y):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            y.sum().backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return (peak - base)

        # Forward peaks
        fwd_native_delta, y_native = measure_forward_peak(native_forward_only)
        bwd_native_delta = measure_backward_peak(y_native)

        fwd_custom_delta, y_custom = measure_forward_peak(custom_forward_only)
        bwd_custom_delta = measure_backward_peak(y_custom)

        print(f"- forward peak delta (MB) native/custom: {fwd_native_delta/1e6:.1f} / {fwd_custom_delta/1e6:.1f}")
        print(f"- backward peak delta (MB) native/custom: {bwd_native_delta/1e6:.1f} / {bwd_custom_delta/1e6:.1f}")


def test_saved_tensors_memory():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, S, Din, Dout, bsz = 1, 4096, 2048, 4096, 256
    rin, rout = Din // bsz, Dout // bsz
    
    x = torch.randn(B, S, Din, device=device, dtype=torch.bfloat16, requires_grad=True)
    Rin = torch.randn(rin, bsz, bsz, device=device, dtype=torch.bfloat16, requires_grad=True)
    W = torch.randn(Dout, Din, device=device, dtype=torch.bfloat16)
    Rout = torch.randn(rout, bsz, bsz, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Baseline: 3-step chain with automatic grad tracking
        base_before = torch.cuda.memory_allocated()
        x_rot = torch_bmm(x.clone().requires_grad_(True), Rin, bsz)
        y_temp = frozen_linear(x_rot, W, None) 
        y_baseline = torch_bmm(y_temp, Rout, bsz)
        base_after = torch.cuda.memory_allocated()
        base_saved = base_after - base_before
        
        # Custom: only saves x, Rin, Rout
        torch.cuda.empty_cache()
        custom_before = torch.cuda.memory_allocated()
        # y_custom = OptimizedCustomLinear.apply(x.clone().requires_grad_(True), Rin, W, None, Rout, bsz)
        y_custom = custom_forward(x.clone().requires_grad_(True), Rin, W, None, Rout, bsz)
        custom_after = torch.cuda.memory_allocated()
        custom_saved = custom_after - custom_before
        
        print(f"Memory for saved tensors:")
        print(f"- Baseline (saves intermediates): {base_saved/1e6:.1f} MB")
        print(f"- Custom (saves x,Rin,Rout only): {custom_saved/1e6:.1f} MB")
        print(f"- Savings: {(base_saved - custom_saved)/1e6:.1f} MB ({100*(base_saved-custom_saved)/base_saved:.1f}%)")
        
        # Expected savings calculation
        x_size = x.numel() * x.element_size()
        intermediate_size = x_rot.numel() * x_rot.element_size() + y_temp.numel() * y_temp.element_size()
        Rin_Rout_size = (Rin.numel() + Rout.numel()) * Rin.element_size()
        expected_savings = intermediate_size - Rin_Rout_size
        print(f"- Expected savings: {expected_savings/1e6:.1f} MB")


def test_backward_memory():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA required for accurate memory measurement")
        return
        
    B, S, Din, Dout, bsz = 1, 4096, 2048, 4096, 256
    rin, rout = Din // bsz, Dout // bsz
    
    def make_inputs():
        x = torch.randn(B, S, Din, device=device, dtype=torch.bfloat16, requires_grad=True)
        Rin = torch.randn(rin, bsz, bsz, device=device, dtype=torch.bfloat16, requires_grad=True)
        W = torch.randn(Dout, Din, device=device, dtype=torch.bfloat16)
        Rout = torch.randn(rout, bsz, bsz, device=device, dtype=torch.bfloat16, requires_grad=True)
        return x, Rin, W, Rout

    # Baseline backward peak
    torch.cuda.empty_cache()
    x, Rin, W, Rout = make_inputs()
    
    # Forward pass
    x_rot = torch_bmm(x, Rin, bsz)
    y_temp = frozen_linear(x_rot, W, None)
    y_baseline = torch_bmm(y_temp, Rout, bsz)
    
    # Measure backward peak
    torch.cuda.reset_peak_memory_stats()
    base_before = torch.cuda.memory_allocated()
    y_baseline.sum().backward()
    torch.cuda.synchronize()
    base_peak = torch.cuda.max_memory_allocated()
    base_backward_peak = base_peak - base_before
    
    # Custom backward peak  
    torch.cuda.empty_cache()
    x, Rin, W, Rout = make_inputs()
    
    # Forward pass
    # y_custom = OptimizedCustomLinear.apply(x, Rin, W, None, Rout, bsz)
    y_custom = custom_forward(x, Rin, W, None, Rout, bsz)
    
    # Measure backward peak
    torch.cuda.reset_peak_memory_stats()
    custom_before = torch.cuda.memory_allocated()
    y_custom.sum().backward()
    torch.cuda.synchronize()
    custom_peak = torch.cuda.max_memory_allocated()
    custom_backward_peak = custom_peak - custom_before
    
    print(f"Backward memory peaks:")
    print(f"- Baseline: {base_backward_peak/1e6:.1f} MB")
    print(f"- Custom: {custom_backward_peak/1e6:.1f} MB")
    print(f"- Savings: {(base_backward_peak - custom_backward_peak)/1e6:.1f} MB ({100*(base_backward_peak-custom_backward_peak)/base_backward_peak:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--S", type=int, default=4096)
    p.add_argument("--Din", type=int, default=2048)
    p.add_argument("--Dout", type=int, default=4096)
    p.add_argument("--bsz", type=int, default=256)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="bf16", choices=[None, "fp32", "fp16", "bf16"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "fp32" or args.dtype is None:
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = None

    benchmark_custom_vs_native(
        B=args.B, S=args.S, Din=args.Din, Dout=args.Dout, bsz=args.bsz,
        iters=args.iters, warmup=args.warmup, device=device, dtype=dtype
    )
    # test_saved_tensors_memory()
    # test_backward_memory()


if __name__ == "__main__":
    main()