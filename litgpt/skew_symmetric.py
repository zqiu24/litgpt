import torch
import triton
import triton.language as tl
from torch.autograd import Function
import time

@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_backward_kernel_optimized(
    grad_mat_ptr,
    grad_vec_ptr,
    N, # This is the matrix dimension D
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    stride_vec_batch,
    stride_vec_element,
    BLOCK_SIZE: tl.constexpr,
):
    # 3D program grid, similar to the forward pass
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1) # Block row ID
    pid_n = tl.program_id(2) # Block col ID

    # --- Optimization 1: Symmetrized Work Assignment ---
    # We only need to compute each grad_vec element once.
    # We can assign this work to the upper-triangle blocks.
    # Programs for lower-triangle blocks do nothing.
    if pid_m > pid_n:
        return

    # Create BLOCK_SIZE x BLOCK_SIZE tile of global indices (i, j)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i = offs_m[:, None]
    j = offs_n[None, :]

    # Pointers to the current batch in grad_mat and grad_vec
    grad_mat_batch_ptr = grad_mat_ptr + pid_batch * stride_mat_batch
    grad_vec_batch_ptr = grad_vec_ptr + pid_batch * stride_vec_batch

    # --- Optimization 2: Select only upper-triangle elements ---
    # We only care about pairs (i, j) where i < j.
    # This mask also handles boundary conditions where N is not a multiple of BLOCK_SIZE.
    upper_mask = (i < j) & (i < N) & (j < N)
    
    # --- Optimization 3: Structured Reads ---
    # Load grad_mat[i, j] for the active elements. This read is highly structured
    # and will be largely coalesced.
    grad_upper_ptr = grad_mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    grad_upper = tl.load(grad_upper_ptr, mask=upper_mask, other=0.0)

    # Load the corresponding grad_mat[j, i] partners using the same mask but
    # transposed pointers.
    grad_lower_ptr = grad_mat_batch_ptr + j * stride_mat_row + i * stride_mat_col
    grad_lower = tl.load(grad_lower_ptr, mask=upper_mask, other=0.0)

    # Compute the gradient as per the chain rule
    grad_val = grad_upper - grad_lower

    # --- Optimization 4: Efficient Forward Indexing ---
    # Replace the expensive sqrt() with the simple forward formula to find k.
    # Note: The formula is slightly different from the forward pass because of how
    # we count elements. `i * N - i*(i+1)//2 + j - i - 1` is a common variant.
    k = i * (2 * N - i - 1) // 2 + (j - i - 1)
    
    # Pointer to the destination in the gradient vector
    grad_vec_ptrs = grad_vec_batch_ptr + k * stride_vec_element
    
    # Store the result. This will be a "scatter" operation, which GPUs handle well.
    tl.store(grad_vec_ptrs, grad_val, mask=upper_mask)

@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_forward_kernel_optimized(
    vec_ptr,
    mat_ptr,
    N,
    stride_vec_batch,
    stride_vec_element,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    BLOCK_SIZE: tl.constexpr,
): 
    # 3D program IDs: batch, row block, column block
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    if pid_m > pid_n:
        return

    # Offset calculations for matrix blocks
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid matrix indices
    mask_m = offs_m < N
    mask_n = offs_n < N
    full_mask = mask_m[:, None] & mask_n[None, :]

    # Create 2D indices [BLOCK_SIZE, BLOCK_SIZE]
    i = offs_m[:, None]  # [BLOCK_SIZE, 1]
    j = offs_n[None, :]  # [1, BLOCK_SIZE]
    
    # Upper triangle processing
    upper_mask = (i < j) & full_mask
    
    # Vector index calculation for upper triangle
    upper_idx = i * (2 * N - i - 1) // 2 + (j - i - 1)

    # Batch-aware pointer arithmetic
    vec_batch_ptr = vec_ptr + pid_batch * stride_vec_batch
    vec_ptrs = vec_batch_ptr + upper_idx * stride_vec_element
    
    # Load upper triangle values
    upper_vals = tl.load(vec_ptrs, mask=upper_mask, other=0.0)
    
    # Matrix pointer calculations for current batch
    mat_batch_ptr = mat_ptr + pid_batch * stride_mat_batch
    mat_ptrs_upper = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    mat_ptrs_lower = mat_batch_ptr + j * stride_mat_row + i * stride_mat_col
    
    # Store upper values and their negatives (skew-symmetric)
    tl.store(mat_ptrs_upper, upper_vals, mask=upper_mask)
    tl.store(mat_ptrs_lower, -upper_vals, mask=upper_mask)
    
    # Zero out diagonal elements
    if pid_m == pid_n:
        diag_mask = (i == j) & full_mask
        diag_ptrs = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
        tl.store(diag_ptrs, tl.zeros((BLOCK_SIZE, BLOCK_SIZE), 
                                dtype=vec_ptr.dtype.element_ty), 
                mask=diag_mask)


class SkewSymmetric(Function):
    @staticmethod
    def forward(ctx, vec, N):
        # Calculate matrix size from vector length
        vec_size = vec.shape[1]
        batch_size = vec.shape[0]
        mat = torch.empty((batch_size, N, N), 
                            device=vec.device, dtype=vec.dtype)

        # Configure kernel launch parameters
        grid = lambda meta: (
            batch_size,
            triton.cdiv(N, meta['BLOCK_SIZE']),
            triton.cdiv(N, meta['BLOCK_SIZE'])
        )

        skew_symmetric_forward_kernel_optimized[grid](
            vec_ptr=vec,
            mat_ptr=mat,
            N=N,
            stride_vec_batch=vec.stride(0),
            stride_vec_element=vec.stride(1),
            stride_mat_batch=mat.stride(0),
            stride_mat_row=mat.stride(1),
            stride_mat_col=mat.stride(2),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(vec)
        ctx.N = N
        return mat

        '''
    @staticmethod
    def backward(ctx, grad_output):
        vec, = ctx.saved_tensors
        N = ctx.N
        batch_size, F = vec.shape
        grad_vec = torch.zeros_like(vec)
        
        # Configure kernel launch parameters
        F = N * (N - 1) // 2
        total_global_elements = batch_size * F
        grid = lambda meta: (triton.cdiv(total_global_elements, meta['BLOCK_SIZE']), )
        # grid = lambda meta: (batch_size, triton.cdiv(F, meta['BLOCK_SIZE']))
        
        skew_symmetric_backward_kernel_working[grid](
            grad_output,
            grad_vec,
            batch_size,
            N,
            stride_mat_batch=grad_output.stride(0),
            stride_mat_row=grad_output.stride(1),
            stride_mat_col=grad_output.stride(2),
            stride_vec_batch=grad_vec.stride(0),
            stride_vec_element=grad_vec.stride(1),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_vec, None
    '''

    @staticmethod
    def backward(ctx, grad_mat):
        B, D, _ = grad_mat.shape
        
        N_vec = D * (D - 1) // 2
        grad_vec = torch.empty((B, N_vec), device=grad_mat.device, dtype=grad_mat.dtype)

        grid = lambda meta: (
            B,
            triton.cdiv(D, meta['BLOCK_SIZE']),
            triton.cdiv(D, meta['BLOCK_SIZE']),
        )

        skew_symmetric_backward_kernel_optimized[grid](
            grad_mat,
            grad_vec,
            D,
            grad_mat.stride(0), grad_mat.stride(1), grad_mat.stride(2),
            grad_vec.stride(0), grad_vec.stride(1),
            # BLOCK_SIZE=16 # Or 32, requires tuning
        )
        return grad_vec, None