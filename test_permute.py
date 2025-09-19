import torch
from test_bs_matmul import block_diag_lr_matmul

def perm_matrix(perm, device=None, dtype=None):
    n = perm.numel()
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye.index_select(0, perm)

def verify_index_select_reversibility(M, N, seed=42):
    """
    Verify that each of the 4 index_select operations can be reversed:
    1. W.index_select(0, perm_out) ↔ W.index_select(0, perm_out_inv)
    2. W.index_select(0, perm_out_inv) ↔ W.index_select(0, perm_out)  
    3. W.index_select(1, perm_in_inv) ↔ W.index_select(1, perm_in)
    4. W.index_select(1, perm_in) ↔ W.index_select(1, perm_in_inv)
    
    Args:
        M: number of rows in W_orig
        N: number of columns in W_orig  
        seed: random seed for reproducibility
    """
    torch.manual_seed(seed)
    
    # Create random matrix W_orig of shape (M, N)
    W_orig = torch.randn(M, N, dtype=torch.float32)
    
    # Create random permutations
    perm_out = torch.randperm(M)
    perm_in = torch.randperm(N)
    
    # Compute inverse permutations
    perm_out_inv = torch.argsort(perm_out)
    perm_in_inv = torch.argsort(perm_in)
    
    print(f"Testing reversibility with W_orig shape: {W_orig.shape}")
    print(f"perm_out: {perm_out}")
    print(f"perm_out_inv: {perm_out_inv}")
    print(f"perm_in: {perm_in}")
    print(f"perm_in_inv: {perm_in_inv}")
    print("-" * 70)
    
    # Test 1: W.index_select(0, perm_out) → W.index_select(0, perm_out_inv) → W
    W_forward_1 = W_orig.index_select(0, perm_out)
    W_reverse_1 = W_forward_1.index_select(0, perm_out_inv)
    test1_pass = torch.allclose(W_orig, W_reverse_1, atol=1e-6, rtol=1e-6)
    max_diff1 = (W_orig - W_reverse_1).abs().max().item()
    
    print(f"Test 1: W → W.index_select(0, perm_out) → W.index_select(0, perm_out_inv) → W")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff1:.2e}")
    
    # Test 2: W.index_select(0, perm_out_inv) → W.index_select(0, perm_out) → W
    W_forward_2 = W_orig.index_select(0, perm_out_inv)
    W_reverse_2 = W_forward_2.index_select(0, perm_out)
    test2_pass = torch.allclose(W_orig, W_reverse_2, atol=1e-6, rtol=1e-6)
    max_diff2 = (W_orig - W_reverse_2).abs().max().item()
    
    print(f"Test 2: W → W.index_select(0, perm_out_inv) → W.index_select(0, perm_out) → W")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff2:.2e}")
    
    # Test 3: W.index_select(1, perm_in_inv) → W.index_select(1, perm_in) → W
    W_forward_3 = W_orig.index_select(1, perm_in_inv)
    W_reverse_3 = W_forward_3.index_select(1, perm_in)
    test3_pass = torch.allclose(W_orig, W_reverse_3, atol=1e-6, rtol=1e-6)
    max_diff3 = (W_orig - W_reverse_3).abs().max().item()
    
    print(f"Test 3: W → W.index_select(1, perm_in_inv) → W.index_select(1, perm_in) → W")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff3:.2e}")
    
    # Test 4: W.index_select(1, perm_in) → W.index_select(1, perm_in_inv) → W
    W_forward_4 = W_orig.index_select(1, perm_in)
    W_reverse_4 = W_forward_4.index_select(1, perm_in_inv)
    test4_pass = torch.allclose(W_orig, W_reverse_4, atol=1e-6, rtol=1e-6)
    max_diff4 = (W_orig - W_reverse_4).abs().max().item()
    
    print(f"Test 4: W → W.index_select(1, perm_in) → W.index_select(1, perm_in_inv) → W")
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff4:.2e}")
    
    print("-" * 70)
    
    # Bonus Test 5: Combined operations (like your W_orig_merged2 example)
    # Forward: W → index_select(0, perm_out_inv) → index_select(1, perm_in_inv)
    W_combined_forward = W_orig.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
    # Reverse: W_combined → index_select(1, perm_in) → index_select(0, perm_out)
    W_combined_reverse = W_combined_forward.index_select(1, perm_in).index_select(0, perm_out)
    test5_pass = torch.allclose(W_orig, W_combined_reverse, atol=1e-6, rtol=1e-6)
    max_diff5 = (W_orig - W_combined_reverse).abs().max().item()
    
    print(f"Test 5: Combined operations (your W_orig_merged2 example)")
    print(f"  Forward:  W → index_select(0, perm_out_inv) → index_select(1, perm_in_inv)")
    print(f"  Reverse:  W_combined → index_select(1, perm_in) → index_select(0, perm_out)")
    print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff5:.2e}")
    
    print("-" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
    print(f"Overall result: {'ALL REVERSIBILITY TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    
    return all_pass

def verify_permutation_identity_property(M, N, seed=42):
    """
    Additional verification: perm and perm_inv are truly inverses of each other
    """
    torch.manual_seed(seed)
    
    perm_out = torch.randperm(M)
    perm_in = torch.randperm(N)
    perm_out_inv = torch.argsort(perm_out)
    perm_in_inv = torch.argsort(perm_in)
    
    print(f"Verifying permutation identity properties:")
    
    # Test that applying perm then perm_inv gives identity
    M_identity_1 = perm_out[perm_out_inv]
    M_identity_2 = perm_out_inv[perm_out]
    N_identity_1 = perm_in[perm_in_inv]
    N_identity_2 = perm_in_inv[perm_in]
    
    M_expected = torch.arange(M)
    N_expected = torch.arange(N)
    
    print(f"  perm_out[perm_out_inv] == range(M): {torch.equal(M_identity_1, M_expected)}")
    print(f"  perm_out_inv[perm_out] == range(M): {torch.equal(M_identity_2, M_expected)}")
    print(f"  perm_in[perm_in_inv] == range(N): {torch.equal(N_identity_1, N_expected)}")
    print(f"  perm_in_inv[perm_in] == range(N): {torch.equal(N_identity_2, N_expected)}")
    print()


def verify_permutation_relationships(M, N, seed=42):
    """
    Verify the 4 relationships between permutation matrix operations and index_select:
    1. P_out @ W_orig ≡ W_orig.index_select(0, perm_out)
    2. P_out.t() @ W_orig ≡ W_orig.index_select(0, perm_out_inv)  
    3. W_orig @ P_in ≡ W_orig.index_select(1, perm_in_inv)
    4. W_orig @ P_in.t() ≡ W_orig.index_select(1, perm_in)
    
    Args:
        M: number of rows in W_orig
        N: number of columns in W_orig  
        seed: random seed for reproducibility
    """
    torch.manual_seed(seed)
    
    # Create random matrix W_orig of shape (M, N)
    W_orig = torch.randn(M, N, dtype=torch.float32)
    
    # Create random permutations
    perm_out = torch.randperm(M)
    perm_in = torch.randperm(N)
    
    # Compute inverse permutations
    perm_out_inv = torch.argsort(perm_out)
    perm_in_inv = torch.argsort(perm_in)
    
    # Create permutation matrices
    P_out = perm_matrix(perm_out, device=W_orig.device, dtype=W_orig.dtype)
    P_in = perm_matrix(perm_in, device=W_orig.device, dtype=W_orig.dtype)
    
    print(f"Testing with W_orig shape: {W_orig.shape}")
    print(f"perm_out: {perm_out}")
    print(f"perm_out_inv: {perm_out_inv}")
    print(f"perm_in: {perm_in}")
    print(f"perm_in_inv: {perm_in_inv}")
    print("-" * 60)
    
    # Test 1: P_out @ W_orig ≡ W_orig.index_select(0, perm_out)
    result1_matmul = P_out @ W_orig
    result1_index = W_orig.index_select(0, perm_out)
    test1_pass = torch.allclose(result1_matmul, result1_index, atol=1e-6, rtol=1e-6)
    max_diff1 = (result1_matmul - result1_index).abs().max().item()
    
    print(f"Test 1: P_out @ W_orig ≡ W_orig.index_select(0, perm_out)")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff1:.2e}")
    
    # Test 2: P_out.t() @ W_orig ≡ W_orig.index_select(0, perm_out_inv)
    result2_matmul = P_out.t() @ W_orig
    result2_index = W_orig.index_select(0, perm_out_inv)
    test2_pass = torch.allclose(result2_matmul, result2_index, atol=1e-6, rtol=1e-6)
    max_diff2 = (result2_matmul - result2_index).abs().max().item()
    
    print(f"Test 2: P_out.t() @ W_orig ≡ W_orig.index_select(0, perm_out_inv)")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff2:.2e}")
    
    # Test 3: W_orig @ P_in ≡ W_orig.index_select(1, perm_in_inv)
    result3_matmul = W_orig @ P_in
    result3_index = W_orig.index_select(1, perm_in_inv)
    test3_pass = torch.allclose(result3_matmul, result3_index, atol=1e-6, rtol=1e-6)
    max_diff3 = (result3_matmul - result3_index).abs().max().item()
    
    print(f"Test 3: W_orig @ P_in ≡ W_orig.index_select(1, perm_in_inv)")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff3:.2e}")
    
    # Test 4: W_orig @ P_in.t() ≡ W_orig.index_select(1, perm_in)
    result4_matmul = W_orig @ P_in.t()
    result4_index = W_orig.index_select(1, perm_in)
    test4_pass = torch.allclose(result4_matmul, result4_index, atol=1e-6, rtol=1e-6)
    max_diff4 = (result4_matmul - result4_index).abs().max().item()
    
    print(f"Test 4: W_orig @ P_in.t() ≡ W_orig.index_select(1, perm_in)")
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")
    print(f"  Max diff: {max_diff4:.2e}")
    
    print("-" * 60)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"Overall result: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    
    return all_pass

def main():
    torch.manual_seed(0)

    # dims
    B = 3
    r_in = 2
    r_out = 3
    b = 4
    N = r_in * b
    M = r_out * b
    R_in_batch = torch.randn(r_in, b, b).float()
    R_out_batch = torch.randn(r_out, b, b).float()

    # random tensors
    x = torch.randn(B, N).float()
    # R_in = torch.randn(N, N).float()
    # R_out = torch.randn(M, M).float()
    R_in = torch.block_diag(*R_in_batch)
    R_out = torch.block_diag(*R_out_batch)
    W_orig = torch.randn(M, N).float()  # so W_orig.t(): (N, M), N: in_features, M: out_features

    # permutations and inverses
    perm_in = torch.randperm(N)
    perm_in_inv = torch.argsort(perm_in)
    perm_out = torch.randperm(M)
    perm_out_inv = torch.argsort(perm_out)

    # ---------- Method 0: explicit permutation matrices ----------
    P_in = perm_matrix(perm_in, device=x.device, dtype=x.dtype)
    P_out = perm_matrix(perm_out, device=x.device, dtype=x.dtype)

    y_mat = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()

    # ---------- Method 1: direct indexing of last dim step-by-step ----------
    y = x
    y = y.index_select(-1, perm_in_inv)          # x @ P_in
    y = y @ R_in
    y = y.index_select(-1, perm_in)      # @ P_in.t()
    y = y @ W_orig.t()
    y = y.index_select(-1, perm_out_inv)          # @ P_out
    y = y @ R_out
    y = y.index_select(-1, perm_out)      # @ P_out.t()

    # verify
    print("[method 1] allclose:", torch.allclose(y, y_mat, atol=1e-6, rtol=1e-6))
    print("[method 1] max abs diff:", (y - y_mat).abs().max().item())

    # ---------- Method 2: direct indexing of last dim step-by-step + merged W_orig ----------
    y2 = x
    y2 = y2.index_select(-1, perm_in_inv)
    y2 = y2 @ R_in
    # W_merged.t() = P_in.t() @ W_orig.t() @ P_out
    # W_merged = P_out.t() @ W_orig @ P_in
    W_orig_merged = P_out.t() @ W_orig @ P_in
    y2 = y2 @ W_orig_merged.t()
    
    y2 = y2 @ R_out
    y2 = y2.index_select(-1, perm_out)

    print("[method 2] allclose:", torch.allclose(y2, y_mat, atol=1e-6, rtol=1e-6))
    print("[method 2] max abs diff:", (y2 - y_mat).abs().max().item())

    # ---------- Method 3: direct indexing of last dim step-by-step + merged W_orig with indexing ----------
    y3 = x
    y3 = y3.index_select(-1, perm_in_inv)
    y3 = y3 @ R_in
    # W_merged = P_out.t() @ W_orig @ P_in
    W_orig_merged1 = W_orig.index_select(0, perm_out_inv)
    W_orig_merged2 = W_orig_merged1.index_select(-1, perm_in_inv)
    y3 = y3 @ W_orig_merged2.t()
    # y2 = y2.index_select(-1, perm_out_inv)
    y3 = y3 @ R_out
    y3 = y3.index_select(-1, perm_out)

    # recover the original weights
    # W_merged = P_out.t() @ W_orig @ P_in
    # W_orig = P_out @ W_merged @ P_in.t()
    W_recovered2 = W_orig_merged2.index_select(-1, perm_in)
    W_recovered = W_recovered2.index_select(0, perm_out)

    print("[method 3] allclose:", torch.allclose(y3, y_mat, atol=1e-6, rtol=1e-6))
    print("[method 3] max abs diff:", (y3 - y_mat).abs().max().item())
    print("[method 3] allclose recovered:", torch.allclose(W_recovered, W_orig, atol=1e-6, rtol=1e-6))
    print("[method 3] max abs diff recovered:", (W_recovered - W_orig).abs().max().item())

    # ---------- Method 4: one-step forward + complete merge with indexing ----------
    y4 = x
    # y_mat = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
    # W_merged.t() = P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
    # W_merged.t() = R_in_merged @ W_orig.t() @ R_out_merged
    # R_in_merged = P_in @ R_in @ P_in.t()
    # R_out_merged = P_out @ R_out @ P_out.t()
    R_in_1 = R_in.index_select(-1, perm_in)
    R_in_2 = R_in_1.index_select(0, perm_in)
    R_out_1 = R_out.index_select(-1, perm_out)
    R_out_2 = R_out_1.index_select(0, perm_out)
    W_merged = (R_in_2 @ W_orig.t() @ R_out_2).t()
    y4 = y4 @ W_merged.t()

    print("[method 4] allclose:", torch.allclose(y4, y_mat, atol=1e-6, rtol=1e-6))
    print("[method 4] max abs diff:", (y4 - y_mat).abs().max().item())

    # ---------- Method 5: one-step forward + complete merge with indexing + bs_matmul ----------
    y5 = x
    tmp = W_orig
    tmp = tmp.t()
    tmp = tmp.index_select(0, perm_in_inv)
    tmp = tmp.index_select(-1, perm_out_inv)
    tmp = block_diag_lr_matmul(R_in_batch, tmp, R_out_batch)
    tmp = tmp.index_select(0, perm_in)
    tmp = tmp.index_select(-1, perm_out)
    tmp = tmp.t()
    y5 = y5 @ tmp.t()

    print("[method 5] allclose:", torch.allclose(y5, y_mat, atol=1e-6, rtol=1e-6))
    print("[method 5] max abs diff:", (y5 - y_mat).abs().max().item())

if __name__ == "__main__":
    M = 3
    N = 4
    verify_permutation_relationships(M, N)
    verify_index_select_reversibility(M, N)

    main()