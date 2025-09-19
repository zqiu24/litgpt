import argparse, torch, math

# y = x @ Rin @ (W.t() @ Rout)
# y = x @ Rin @ W.t()

# mm1: x @ Rin
# mm2: a @ W.t()
# mm3: b @ Rout

# mm3 backward: grad_Rout = b = x @ Rin @ W.t() [B, S, Dout]
# mm2 backward: 

# y = x @ Rin @ W.t()

# xr @ W.t() @ Rout
# xr @ W.t() 
# xr [B, S, Din]

# y = x @ W.t() @ Rout

def setup(device, dtype, N, Din, Dout, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, Din, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(Dout, Din, device=device, dtype=dtype)  # frozen
    Rin = torch.randn(Din, Din, device=device, dtype=dtype, requires_grad=True)
    Rout = torch.randn(Dout, Dout, device=device, dtype=dtype, requires_grad=True)
    return x, W, Rin, Rout

# ---------- Baselines (vanilla autograd) ----------
def pre_vanilla(x, Rin, W):          # y = x @ Rin @ W.t()
    return (x @ Rin) @ W.t()

def post_vanilla(x, Rout, W):        # y = x @ W.t() @ Rout
    return (x @ W.t()) @ Rout

# ---------- TRUE checkpointed versions (recompute, don't save intermediates) ----------
class PreRotateLinearCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Rin, W):
        y = (x @ Rin) @ W.t()
        # Save only the inputs, not the intermediate (x @ Rin)
        ctx.save_for_backward(x, Rin)
        ctx.W = W
        return y

    @staticmethod
    def backward(ctx, g):
        x, Rin = ctx.saved_tensors
        W = ctx.W
        # Recompute the intermediate instead of saving it
        # x_rot = x @ Rin  # recompute this
        
        grad_x = grad_Rin = None
        if ctx.needs_input_grad[0]:
            grad_x = (g @ W) @ Rin.t()
        if ctx.needs_input_grad[1]:
            grad_Rin = x.t() @ (g @ W)
        return grad_x, grad_Rin, None

def pre_checkpoint(x, Rin, W):
    return PreRotateLinearCheckpoint.apply(x, Rin, W)

class PostRotateLinearCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Rout, W):
        y = (x @ W.t()) @ Rout
        # Save only the inputs, not the intermediate (x @ W.t())
        ctx.save_for_backward(x, Rout)
        ctx.W = W
        return y

    @staticmethod
    def backward(ctx, g):
        x, Rout = ctx.saved_tensors
        W = ctx.W
        # Recompute the intermediate instead of saving it
        x_proj = x @ W.t()  # recompute this
        
        grad_x = grad_Rout = None
        if ctx.needs_input_grad[0]:
            grad_x = (g @ Rout.t()) @ W
        if ctx.needs_input_grad[1]:
            grad_Rout = x_proj.t() @ g
        return grad_x, grad_Rout, None

def post_checkpoint(x, Rout, W):
    return PostRotateLinearCheckpoint.apply(x, Rout, W)

# ---------- Correctness check ----------
def check_correctness(device, dtype, N=32, Din=64, Dout=128, tol=1e-3):
    """Check that checkpointed versions produce same results as vanilla."""
    print(f"Checking correctness (tol={tol})...")
    
    # Pre-rotation test
    x, W, Rin, Rout = setup(device, dtype, N, Din, Dout, seed=42)
    W.requires_grad_(False)
    
    # Forward pass
    y1 = pre_vanilla(x.clone(), Rin.clone(), W)
    y2 = pre_checkpoint(x.clone(), Rin.clone(), W)
    
    fwd_err = (y1 - y2).abs().max().item()
    print(f"  Pre-rotate forward error: {fwd_err:.2e}")
    assert fwd_err < tol, f"Forward mismatch: {fwd_err}"
    
    # Backward pass
    loss1 = y1.sum()
    loss2 = y2.sum()
    
    # Clear any existing gradients
    if x.grad is not None: x.grad.zero_()
    if Rin.grad is not None: Rin.grad.zero_()
    
    loss1.backward()
    grad_x1 = x.grad.clone()
    grad_Rin1 = Rin.grad.clone()
    
    x.grad.zero_()
    Rin.grad.zero_()
    
    loss2.backward()
    grad_x2 = x.grad.clone()
    grad_Rin2 = Rin.grad.clone()
    
    bwd_x_err = (grad_x1 - grad_x2).abs().max().item()
    bwd_Rin_err = (grad_Rin1 - grad_Rin2).abs().max().item()
    print(f"  Pre-rotate grad_x error: {bwd_x_err:.2e}")
    print(f"  Pre-rotate grad_Rin error: {bwd_Rin_err:.2e}")
    assert bwd_x_err < tol, f"grad_x mismatch: {bwd_x_err}"
    assert bwd_Rin_err < tol, f"grad_Rin mismatch: {bwd_Rin_err}"
    
    # Post-rotation test
    x, W, Rin, Rout = setup(device, dtype, N, Din, Dout, seed=43)
    W.requires_grad_(False)
    
    # Forward pass
    y1 = post_vanilla(x.clone(), Rout.clone(), W)
    y2 = post_checkpoint(x.clone(), Rout.clone(), W)
    
    fwd_err = (y1 - y2).abs().max().item()
    print(f"  Post-rotate forward error: {fwd_err:.2e}")
    assert fwd_err < tol, f"Forward mismatch: {fwd_err}"
    
    # Backward pass
    loss1 = y1.sum()
    loss2 = y2.sum()
    
    # Clear any existing gradients
    if x.grad is not None: x.grad.zero_()
    if Rout.grad is not None: Rout.grad.zero_()
    
    loss1.backward()
    grad_x1 = x.grad.clone()
    grad_Rout1 = Rout.grad.clone()
    
    x.grad.zero_()
    Rout.grad.zero_()
    
    loss2.backward()
    grad_x2 = x.grad.clone()
    grad_Rout2 = Rout.grad.clone()
    
    bwd_x_err = (grad_x1 - grad_x2).abs().max().item()
    bwd_Rout_err = (grad_Rout1 - grad_Rout2).abs().max().item()
    print(f"  Post-rotate grad_x error: {bwd_x_err:.2e}")
    print(f"  Post-rotate grad_Rout error: {bwd_Rout_err:.2e}")
    assert bwd_x_err < tol, f"grad_x mismatch: {bwd_x_err}"
    assert bwd_Rout_err < tol, f"grad_Rout mismatch: {bwd_Rout_err}"
    
    print("  ✓ All correctness checks passed!")

# ---------- Measurement ----------
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
        loss = y.sum()
        loss.backward()
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
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--N", type=int, default=4096)      # batch*seq
    p.add_argument("--Din", type=int, default=4096)
    p.add_argument("--Dout", type=int, default=11008)  # typical up-proj size
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-check", action="store_true", help="Skip correctness check")
    args = p.parse_args()

    dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)[args.dtype]
    device = torch.device(args.device)

    # Correctness check
    if not args.skip_check:
        tol = 1e-3 if dtype == torch.float16 else 1e-5
        check_correctness(device, dtype, tol=tol)
        print()

    x, W, Rin, Rout = setup(device, dtype, args.N, args.Din, args.Dout)
    W.requires_grad_(False)

    print(f"Device={device}, dtype={dtype}, N={args.N}, Din={args.Din}, Dout={args.Dout}")
    print("Measuring... (fwd_ms, bwd_ms, peak_MB)")

    f, b, m = measure(pre_vanilla, (x, Rin, W), args.iters)
    print(f"pre_vanilla      : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

    f, b, m = measure(pre_checkpoint, (x, Rin, W), args.iters)
    print(f"pre_checkpoint   : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

    # Re-sample to avoid caching effects
    x, W, Rin, Rout = setup(device, dtype, args.N, args.Din, args.Dout, seed=1)
    W.requires_grad_(False)

    f, b, m = measure(post_vanilla, (x, Rout, W), args.iters)
    print(f"post_vanilla     : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

    f, b, m = measure(post_checkpoint, (x, Rout, W), args.iters)
    print(f"post_checkpoint  : fwd={f:.2f}, bwd={b:.2f}, peak={m/1024/1024:.1f} MB")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()