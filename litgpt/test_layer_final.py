import argparse, torch

# ---------- Baseline: ((x @ Rin) @ W.t()) @ Rout ----------
def forward_vanilla(x, Rin, W, Rout):
    y = x @ Rin
    y = y @ W.t()
    y = y @ Rout
    return y

# ---------- Checkpointed tail: xR -> (xR @ W.t()) @ Rout ----------
class PostRotateTailCheckpointFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xR, W, Rout):
        y1 = xR @ W.t()          # [B,S,Dout], not saved
        y = y1 @ Rout
        ctx.save_for_backward(xR, Rout)  # keep small [B,S,Din] and [Dout,Dout]
        ctx.W = W
        return y

    @staticmethod
    def backward(ctx, g):
        xR, Rout = ctx.saved_tensors
        W = ctx.W
        grad_xR = (g @ Rout.t()) @ W if ctx.needs_input_grad[0] else None
        grad_Rout = None
        if ctx.needs_input_grad[2]:
            y1 = xR @ W.t()  # recompute [B,S,Dout]
            BS = y1.shape[0] * y1.shape[1]
            grad_Rout = (y1.reshape(BS, -1).t() @ g.reshape(BS, -1))
        return grad_xR, None, grad_Rout

def forward_optimized(x, Rin, W, Rout):
    xR = x @ Rin                 # keeps [B,S,Din] alive; grads flow to x, Rin
    y = PostRotateTailCheckpointFn.apply(xR, W, Rout)
    return y

# ---------- Setup, correctness, and whole F+B benchmark ----------
def setup(device, dtype, B, S, Din, Dout, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, Din, device=device, dtype=dtype, requires_grad=True)
    Rin = torch.randn(Din, Din, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(Dout, Din, device=device, dtype=dtype)  # frozen
    Rout = torch.randn(Dout, Dout, device=device, dtype=dtype, requires_grad=True)
    return x, Rin, W, Rout

def check_correctness(device, dtype, B=4, S=8, Din=16, Dout=32, tol=1e-3):
    x, Rin, W, Rout = setup(device, dtype, B, S, Din, Dout, seed=123); W.requires_grad_(False)
    y_ref = forward_vanilla(x.clone(), Rin.clone(), W, Rout.clone())
    y_opt = forward_optimized(x.clone(), Rin.clone(), W, Rout.clone())
    assert (y_ref - y_opt).abs().max().item() < tol, "forward mismatch"

    for t in (x, Rin, Rout):
        if t.grad is not None: t.grad.zero_()
    y_ref.sum().backward()
    gx_r, gRin_r, gRout_r = x.grad.clone(), Rin.grad.clone(), Rout.grad.clone()

    for t in (x, Rin, Rout):
        t.grad = None
    y_opt.sum().backward()
    gx_o, gRin_o, gRout_o = x.grad.clone(), Rin.grad.clone(), Rout.grad.clone()

    assert (gx_r - gx_o).abs().max().item() < tol, "grad_x mismatch"
    assert (gRin_r - gRin_o).abs().max().item() < tol, "grad_Rin mismatch"
    assert (gRout_r - gRout_o).abs().max().item() < tol, "grad_Rout mismatch"

def measure(step_fn, inputs, iters=10, warmup=3):
    device = inputs[0].device
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        for t in inputs:
            if t.requires_grad and t.grad is not None:
                t.grad.zero_()
        y = step_fn(*inputs)
        (y.sum()).backward()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated(device) if is_cuda else 0

    fwd_ms = bwd_ms = 0.0
    for _ in range(iters):
        for t in inputs:
            if t.requires_grad and t.grad is not None:
                t.grad.zero_()
        if is_cuda:
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
            e0.record()
        y = step_fn(*inputs)
        loss = y.sum()
        if is_cuda:
            e1.record()
        loss.backward()
        if is_cuda:
            e2.record()
            torch.cuda.synchronize()
            fwd_ms += e0.elapsed_time(e1)
            bwd_ms += e1.elapsed_time(e2)

    peak = (torch.cuda.max_memory_allocated(device) - start_mem) if is_cuda else 0
    return fwd_ms/iters, bwd_ms/iters, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--Din", type=int, default=4096)
    ap.add_argument("--Dout", type=int, default=11008)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip_check", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not args.skip_check:
        tol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        check_correctness(device, dtype, tol=tol)

    print(f"Device={device}, dtype={dtype}, B={args.B}, S={args.S}, Din={args.Din}, Dout={args.Dout}")
    print("Measuring... (fwd_ms, bwd_ms, peak_MB over fwd+bk)")

    x, Rin, W, Rout = setup(device, dtype, args.B, args.S, args.Din, args.Dout, seed=0); W.requires_grad_(False)
    f, b, m = measure(forward_vanilla, (x, Rin, W, Rout), args.iters, args.warmup)
    print(f"vanilla     : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

    x, Rin, W, Rout = setup(device, dtype, args.B, args.S, args.Din, args.Dout, seed=1); W.requires_grad_(False)
    f, b, m = measure(forward_optimized, (x, Rin, W, Rout), args.iters, args.warmup)
    print(f"optimized   : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()