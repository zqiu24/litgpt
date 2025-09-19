import torch
torch.set_float32_matmul_precision("high")

def block_diag_lr_matmul(A_blocks: torch.Tensor, W: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    """
    Compute (block_diag(A_blocks) @ W @ block_diag(B_blocks)) without materializing block-diagonal matrices.

    Args:
      A_blocks: (r_m, b, b) block-diagonal factors for the left (M = r_m * b)
      W:        (M, N) matrix to multiply, where M = r_m * b, N = r_n * b
      B_blocks: (r_n, b, b) block-diagonal factors for the right (N = r_n * b)

    Returns:
      Tensor of shape (M, N)
    """
    if A_blocks.ndim != 3 or B_blocks.ndim != 3:
        raise ValueError("A_blocks and B_blocks must be 3D: (r, b, b)")
    r_m, b1, b2 = A_blocks.shape
    r_n, b3, b4 = B_blocks.shape
    if not (b1 == b2 == b3 == b4):
        raise ValueError("All block sizes must match and be square b x b.")
    b = b1
    M = r_m * b
    N = r_n * b
    if W.shape != (M, N):
        raise ValueError(f"W must have shape {(M, N)}, got {tuple(W.shape)}")

    # Ensure device/dtype compatibility (keeps things simple and safe)
    if A_blocks.device != W.device or A_blocks.dtype != W.dtype:
        A_blocks = A_blocks.to(device=W.device, dtype=W.dtype)
    if B_blocks.device != W.device or B_blocks.dtype != W.dtype:
        B_blocks = B_blocks.to(device=W.device, dtype=W.dtype)

    # Reshape W into blocks and apply batched matmuls:
    # W_ = (r_m, r_n, b, b), where W_[i, j] is the (i, j) b x b block of W
    W_blocks = W.view(r_m, b, r_n, b).transpose(1, 2)  # (r_m, r_n, b, b)

    # Left multiply each block-row by corresponding A_blocks[i]
    # Shapes: (r_m, 1, b, b) @ (r_m, r_n, b, b) -> (r_m, r_n, b, b)
    left = torch.matmul(A_blocks.unsqueeze(1), W_blocks)

    # Right multiply each block-col by corresponding B_blocks[j]
    # Shapes: (r_m, r_n, b, b) @ (1, r_n, b, b) -> (r_m, r_n, b, b)
    out_blocks = torch.matmul(left, B_blocks.unsqueeze(0))

    # Fold back to (M, N)
    out = out_blocks.permute(0, 2, 1, 3).contiguous().view(M, N)
    return out


def _verify_correctness(device="cpu", dtype=torch.float32, r_m=23, r_n=12, b=32, atol=1e-4, rtol=1e-4):
    torch.manual_seed(0)
    M, N = r_m * b, r_n * b
    W = torch.randn(M, N, device=device, dtype=dtype)
    A_blocks = torch.randn(r_m, b, b, device=device, dtype=dtype)
    B_blocks = torch.randn(r_n, b, b, device=device, dtype=dtype)

    # Fast version (no big sparse matrices)
    fast = block_diag_lr_matmul(A_blocks, W, B_blocks)

    # Reference using explicit block-diagonals (memory-inefficient, but correct)
    A_full = torch.block_diag(*A_blocks.unbind(0))
    B_full = torch.block_diag(*B_blocks.unbind(0))

    ref = A_full @ W @ B_full

    ok = torch.allclose(fast, ref, atol=atol, rtol=rtol)
    return ok, fast, ref


if __name__ == "__main__":
    ok_cpu, _, _ = _verify_correctness(device="cpu")
    print("CPU correctness:", ok_cpu)
    if torch.cuda.is_available():
        ok_gpu, _, _ = _verify_correctness(device="cuda")
        print("CUDA correctness:", ok_gpu)