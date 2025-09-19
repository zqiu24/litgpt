# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Derived from https://github.com/microsoft/OFT
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With OFT (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch._dynamo import allow_in_graph
from typing_extensions import Self

import litgpt
from litgpt.config import Config as BaseConfig
from litgpt.model import GPT as BaseModel
from litgpt.model import Block as BaseBlock
from litgpt.model import CausalSelfAttention as BaseCausalSelfAttention
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights


# @torch.compile(mode="reduce-overhead")
def pytorch_skew_symmetric(vec, block_size, rows, cols):
    batch_size = vec.shape[0]
    matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

    matrix[:, rows, cols] = vec
    matrix = matrix - matrix.transpose(-2, -1)
    return matrix

def pytorch_skew_symmetric1(vec, block_size, rows, cols):
    return 0.5 * (vec - vec.transpose(-2, -1))

# @torch.compile(dynamic=True, options={"triton.cudagraphs": False})
def torch_bmm(x, R, block_size):
    Bdims = x.shape[:-1]
    xr = x.view(*Bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R)
    x_rot = xr.contiguous().view(*Bdims, -1)
    return x_rot

# @torch.compile(dynamic=True, options={"triton.cudagraphs": False})
def cayley_batch(
        Q_skew: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5
    ) -> torch.Tensor:
    """
    Perform the Cayley parametrization on a batch of skew-symmetric matrices.

    Args:
        data: A batch of skew-symmetric matrices of shape (b, r, c).
    """

    # print(Q_skew.shape)

    b = Q_skew.shape[0]
    previous_dtype = Q_skew.dtype

    # Q_skew = SkewSymmetric.apply(Q, block_size)
    # Q_skew = self._pytorch_skew_symmetric(Q, block_size, self.rows, self.cols)

    if use_cayley_neumann:
        R = torch.eye(block_size, device=Q_skew.device, dtype=Q_skew.dtype).repeat(b, 1, 1)
        if num_neumann_terms > 1:
            # R.add_(Q_skew, alpha=2.0)
            R = R + 2.0 * Q_skew
            if num_neumann_terms > 2:
                Q_squared = torch.bmm(Q_skew, Q_skew)
                # R.add_(Q_squared, alpha=2.0)
                R = R + 2.0 * Q_squared

                Q_power = Q_squared
                for i in range(3, num_neumann_terms):
                    Q_power = torch.bmm(Q_power, Q_skew)
                    # R.add_(Q_power, alpha=2.0)
                    R = R + 2.0 * Q_power
    else:
        id_mat = (
            torch.eye(Q_skew.shape[-1], device=Q_skew.device)
            .unsqueeze(0)
            .expand(b, Q_skew.shape[-1], Q_skew.shape[-1])
        )
        R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

    return R.to(previous_dtype)

def get_weight_poet(Rl, Rr, block_size, rows, cols, use_cayley_neumann=True, num_cayley_neumann_terms=5):
    r_left = Rl.size(0)
    r_right = Rr.size(0)

    vec_cat = torch.cat([Rl, Rr], dim=0).contiguous()
    # print('vec_cat', vec_cat.device)
    # print('rows', rows.device, 'cols', cols.device)
    # print('rows', rows.device, 'cols', cols.device, 'vec_cat', vec_cat.device)
    # print('rows', rows.numel(), 'vec_cat', vec_cat.shape[1])
    # print('cols', cols.numel(), 'vec_cat', vec_cat.shape[1])
    # Q_skew_cat = pytorch_skew_symmetric(vec_cat, block_size, rows, cols)
    Q_skew_cat = SkewSymmetricFunction.apply(vec_cat, block_size, rows, cols)

    # Single batched Cayley
    R_cat = cayley_batch(
        Q_skew_cat, block_size, use_cayley_neumann, num_cayley_neumann_terms
    )
    R_left, R_right = R_cat.split([r_left, r_right], dim=0)

    return R_left, R_right


class OptimizedEinsumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, R):
        """
        Forward pass uses the fast einsum. Crucially, it saves
        contiguous copies of the tensors for the backward pass.
        
        Shapes:
        x: [B, S, N, D]
        R: [N, D, D]
        """
        # Save clean copies for a fast backward pass
        # This is the key to the optimization!
        ctx.save_for_backward(x, R)

        # Perform the highly optimized forward pass
        output = torch.einsum("bsnk,nkd->bsnd", x, R)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with controlled dtype: compute in fp32 if inputs are fp16/bf16,
        then cast gradients back to inputs' dtypes.
        """
        x, R = ctx.saved_tensors
        go = grad_output.contiguous()

        compute_dtype = torch.float32 if go.dtype in (torch.float16, torch.bfloat16) else go.dtype
        x_c = x.to(compute_dtype)
        R_c = R.to(compute_dtype)
        go_c = go.to(compute_dtype)

        # grad_R: [N, D, D] = sum_{b,s} x^T @ grad_output
        grad_R = torch.einsum("bsnk,bsnd->nkd", x_c, go_c).to(R.dtype)
        # grad_x: [B, S, N, D] = grad_output @ R^T
        grad_x = torch.einsum("bsnd,ndk->bsnk", go_c, R_c.transpose(-2, -1)).to(x.dtype)
        return grad_x, grad_R


class SkewSymmetricFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vec: torch.Tensor, block_size: int, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        # Save indices for backward
        ctx.save_for_backward(rows, cols)
        ctx.block_size = block_size

        batch_size = vec.shape[0]
        M = vec.new_zeros((batch_size, block_size, block_size))
        # Place vec on (rows, cols) and -vec on the transpose locations
        M[:, rows, cols] = vec
        # M[:, cols, rows] = -vec
        return M - M.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        rows, cols = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # dL/dvec = dL/dM[rows, cols] - dL/dM[cols, rows]
        grad_vec = grad_output[:, rows, cols] - grad_output[:, cols, rows]
        return grad_vec, None, None, None


class FrozenLinearFn(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, W, bias=None):
		# x: [..., in], W: [out, in], bias: [out] or None
		# Compute in W.dtype; return y in W.dtype (mirrors F.linear on casted input)
		#x_dtype = x.dtype
		# xw = x.to(W.dtype, copy=False)
		y = x @ W.t()
		if bias is not None:
			y = y + bias
		# Save only what we need for dX
		ctx.save_for_backward(W)
		return y

	@staticmethod
	def backward(ctx, grad_out):
		W, = ctx.saved_tensors
		grad_x = None
		if ctx.needs_input_grad[0]:
			go = grad_out.to(W.dtype, copy=False)
			grad_x = go @ W  # [..., out] @ [out, in] -> [..., in]
			grad_x = grad_x.to(grad_out.dtype, copy=False)
		return grad_x, None, None

class PermutationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x: torch.Tensor, perm: torch.Tensor, inv_perm: torch.Tensor):
		ctx.save_for_backward(inv_perm)
		return x[..., perm]

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		(inv_perm,) = ctx.saved_tensors
		grad_input = grad_output[..., inv_perm]
		return grad_input, None, None

# @allow_in_graph
def permute_apply(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)

# @allow_in_graph
def frozen_linear(x, W, b):
    return FrozenLinearFn.apply(x, W, b)


class ExplicitMatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R_left, R_right, W):
        # Compute compact intermediates
        Z_left = R_left @ W
        Z_right = W @ R_right
        Y = Z_left @ R_right  # == R_left @ W @ R_right
        # Save only compact tensors
        ctx.save_for_backward(Z_left, Z_right)
        return Y

    @staticmethod
    def backward(ctx, grad_out):
        Z_left, Z_right = ctx.saved_tensors
        grad_R_left = grad_R_right = grad_W = None
        if ctx.needs_input_grad[0]:
            grad_R_left = grad_out @ Z_right.transpose(-2, -1)
        if ctx.needs_input_grad[1]:
            grad_R_right = Z_left.transpose(-2, -1) @ grad_out
        return grad_R_left, grad_R_right, grad_W


class BlockDiagFromBatchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: [B, H, H] -> Y: [B*H, B*H] with B dense HxH blocks on the diagonal
        B, H, _ = x.shape
        ctx.B = B
        ctx.H = H

        BH = B * H
        y = x.new_zeros(BH, BH)
        y4 = y.view(B, H, B, H)
        idx = torch.arange(B, device=x.device)
        # Place each block on the diagonal: y4[b, :, b, :] = x[b]
        y4[idx, :, idx, :] = x
        return y

    @staticmethod
    def backward(ctx, grad_out):
        B, H = ctx.B, ctx.H
        g4 = grad_out.view(B, H, B, H)
        idx = torch.arange(B, device=grad_out.device)
        # Extract diagonal HxH blocks as grad for each x[b]
        grad_x = g4[idx, :, idx, :]
        return grad_x

'''
class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        out_features,
        coft=False,
        eps=6e-5,
        block_share=False,
        kernel_size=(0, 0),
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(r, n_elements, device=device, dtype=dtype))
        # self.weight = nn.Parameter(torch.zeros(r, block_size, block_size, device=device, dtype=dtype))
        self.coft = coft
        self.eps = eps
        self.block_share = block_share
        # Conv2d specific parameters
        self.kernel_size = kernel_size
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer('rows', rows.to(device) if device else rows)
        self.register_buffer('cols', cols.to(device) if device else cols)
        self.poet = True
        self.update_permutation()
        if self.poet:
            n_elements_left = block_size * (block_size - 1) // 2
            r_left = out_features // block_size
            self.weight_left = nn.Parameter(torch.zeros(r_left, n_elements_left, device=device, dtype=dtype))
            # self.weight_left = nn.Parameter(torch.zeros(r_left, block_size, block_size, device=device, dtype=dtype))


    def _pytorch_skew_symmetric(self, vec, block_size, rows, cols):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        matrix[:, rows, cols] = vec
        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _pytorch_skew_symmetric1(self, vec, block_size, rows, cols):
        return 0.5 * (vec - vec.transpose(-2, -1))

    def _pytorch_skew_symmetric_inv(self, matrix, block_size):
        batch_size = matrix.shape[0]

        # Extract the upper triangular elements
        vec = matrix[:, self.rows, self.cols]
        return vec

    def _cayley_batch(
        self, Q_skew: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5
    ) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """

        b = Q_skew.shape[0]
        previous_dtype = Q_skew.dtype

        # Q_skew = SkewSymmetric.apply(Q, block_size)
        # Q_skew = self._pytorch_skew_symmetric(Q, block_size, self.rows, self.cols)

        if use_cayley_neumann:
            R = torch.eye(block_size, device=Q_skew.device, dtype=Q_skew.dtype).repeat(b, 1, 1)
            if num_neumann_terms > 1:
                R.add_(Q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    Q_squared = torch.bmm(Q_skew, Q_skew)
                    R.add_(Q_squared, alpha=2.0)

                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
        else:
            id_mat = (
                torch.eye(Q_skew.shape[-1], device=Q_skew.device)
                .unsqueeze(0)
                .expand(b, Q_skew.shape[-1], Q_skew.shape[-1])
            )
            R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, Q, eps=1e-5):
        oft_R = self._pytorch_skew_symmetric(Q, self.block_size)
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_R.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft_R.size(1), oft_R.size(1)), device=oft_R.device, dtype=oft_R.dtype)
            .unsqueeze(0)
            .expand_as(oft_R)
        )
        diff = oft_R - I
        norm_diff = torch.norm(oft_R - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_R, I + eps * (diff / norm_diff))

        return self._pytorch_skew_symmetric_inv(out, self.block_size)

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_R: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_R.shape[0] == 1:
            # block share
            blocks = [oft_R[0, ...] for i in range(rank)]
        else:
            blocks = [oft_R[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def _unfold(self, x):
        """
        Unfold with stride=1, padding=0 to preserve spatial dimensions. Only use kernel_size from base layer to define
        patch size.
        """
        batch_size, in_channels, in_height, in_width = x.shape

        if isinstance(self.kernel_size, int):
            kernel_height, kernel_width = self.kernel_size, self.kernel_size
        else:
            kernel_height, kernel_width = self.kernel_size

        stride_h = stride_w = 1
        pad_h = pad_w = 0

        # output dimensions
        out_height = (in_height + 2 * pad_h - kernel_height) // stride_h + 1
        out_width = (in_width + 2 * pad_w - kernel_width) // stride_w + 1

        # Reshape input from [B, C, H, W] to [B, C, H_out, W_out, K_H, K_W]
        x_unfolded = x.unfold(2, kernel_height, stride_h).unfold(3, kernel_width, stride_w)
        x_unfolded = x_unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_unfolded = x_unfolded.view(batch_size * out_height * out_width, -1)

        return x_unfolded

    def _fold(self, x_unfolded, orig_shape):
        """
        Fold back to preserve spatial dimensions.
        """
        batch_size, in_channels, in_height, in_width = orig_shape

        if isinstance(self.kernel_size, int):
            kernel_height, kernel_width = self.kernel_size, self.kernel_size
        else:
            kernel_height, kernel_width = self.kernel_size

        # With stride=1, padding=0:
        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1

        # Reshape: [B*H_out*W_out, C*K_H*K_W] -> [B, H_out, W_out, C, K_H, K_W]
        x_reshaped = x_unfolded.view(batch_size, out_height, out_width, in_channels, kernel_height, kernel_width)

        # Permute to: [B, C, H_out, W_out, K_H, K_W]
        x_reshaped = x_reshaped.permute(0, 3, 1, 2, 4, 5).contiguous()

        # Use F.fold to reconstruct 4D tensor
        x_folded = F.fold(
            x_reshaped.view(batch_size, in_channels * kernel_height * kernel_width, out_height * out_width),
            output_size=(in_height, in_width),
            kernel_size=(kernel_height, kernel_width),
            stride=(1, 1),
        )

        return x_folded

    def update_permutation(self):
        """Update the permutation of the indices."""
        with torch.no_grad():
            # Update permutation indices
            self.register_buffer('permutation_indices',torch.randperm(self.in_features, device=self.weight.device))
            self.register_buffer('inv_permutation_indices', torch.argsort(self.permutation_indices))
            if self.poet:
                self.register_buffer('permutation_indices_left', torch.randperm(self.out_features, device=self.weight.device))
                self.register_buffer('inv_permutation_indices_left', torch.argsort(self.permutation_indices_left))
            # self.permutation_layer = PermutationLayer(self.permutation_indices, self.inv_permutation_indices)
            # self.reverse_permutation_layer = PermutationLayer(self.inv_permutation_indices, self.permutation_indices)
            # Update permutation matrices
            # self.register_buffer('P', torch.eye(self.in_features, device=self.weight.device, dtype=self.weight.dtype)[self.permutation_indices])
            # self.register_buffer('P_T', torch.eye(self.in_features, device=self.weight.device, dtype=self.weight.dtype)[self.inv_permutation_indices])

    def forward(self, x):
        # This module doesn't need to implement the orthogonal transform
        # It's primarily a container for the parameter
        # The actual transformation logic stays in your OFTLayer

        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape

        if self.coft:
            with torch.no_grad():
                self.weight.copy_(self._project_batch(self.weight, eps=self.eps))

        Q_skew = self._pytorch_skew_symmetric(self.weight, self.block_size, self.rows, self.cols)
        orth_rotate = self._cayley_batch(
            Q_skew, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )

        # Unfold the input for Conv2d layer
        if len(orig_shape) == 4:
            x = self._unfold(x)

        folded_shape = x.shape
        rank = self.in_features // self.block_size if self.block_share else self.r
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)

        if self.block_share:
            orth_rotate = orth_rotate.repeat(rank, 1, 1)
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
        else:
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)

        x_rotated = x_rotated_reshaped.reshape(*folded_shape)

        if len(orig_shape) == 4:
            x_rotated = self._fold(x_rotated, orig_shape)

        return x_rotated.to(required_dtype)

    def get_weight(self):
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight = self.weight

        if self.coft:
            with torch.no_grad():
                weight = self._project_batch(weight, eps=self.eps)
                self.weight.copy_(weight)

        orth_rotate = self._cayley_batch(
            weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )

        rank = self.r if not self.block_share else self.in_features // self.block_size
        return self._block_diagonal(orth_rotate, rank)

    def get_weight_poet(self):
        r_left = self.weight_left.size(0)
        r_right = self.weight.size(0)

        vec_cat = torch.cat([self.weight_left, self.weight], dim=0).contiguous()
        Q_skew_cat = self._pytorch_skew_symmetric(vec_cat, self.block_size, self.rows, self.cols)
        # Q_skew_cat = SkewSymmetricFunction.apply(vec_cat, self.block_size, self.rows, self.cols)

        # Single batched Cayley
        R_cat = self._cayley_batch(
            Q_skew_cat, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )
        R_left, R_right = R_cat.split([r_left, r_right], dim=0)

        return R_left, R_right

    def get_weight_poet_tmp(self):
        Q_skew_right = self._pytorch_skew_symmetric(self.weight, self.block_size, self.rows, self.cols)
        R_right = self._cayley_batch(Q_skew_right, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms)
        return R_right
    
    '''



class OFTLayer(nn.Module):
    def __init__(self, oft_block_size: int, oft_dropout: float):
        """Store OFT specific attributes in a class.

        Args:
            oft_block_size: rank of the weight update matrices. To make sense of using OFT the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            oft_dropout: dropout that is applied on the input in the OFT branch (before multiplying by matrix A)
        """
        super().__init__()
        assert oft_block_size >= 0
        self.oft_block_size = oft_block_size
        # Optional dropout
        if oft_dropout > 0.0:
            self.oft_dropout = nn.Dropout(p=oft_dropout)
        else:
            self.oft_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        self.gradient_accumulation_steps = 1
        self.update_reset_R_gap = 0


class OFTLinear(OFTLayer):
    # OFT implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for OFT
        oft_block_size: int = 0,
        oft_dropout: float = 0.0,
        **kwargs: Any,
    ):
        """OFT wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. OFT R matrix as `self.oft_R`
        Only OFT's R matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            oft_block_size: rank of the weight update matrices. To make sense of using OFT the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            oft_dropout: dropout that is applied on the input in the OFT branch (before multiplying by matrix A)
        """
        super().__init__(oft_block_size=oft_block_size, oft_dropout=oft_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        # Actual trainable parameters
        if oft_block_size > 0:
            r_in = in_features // oft_block_size
            r_out = out_features // oft_block_size
            n_elements = oft_block_size * (oft_block_size - 1) // 2
            # self.oft_L = nn.Parameter(torch.empty(r_out, n_elements)) #, device=self.linear.weight.device, dtype=self.linear.weight.dtype))
            # self.oft_R = nn.Parameter(torch.empty(r_in, n_elements)) # , device=self.linear.weight.device, dtype=self.linear.weight.dtype))
            self.oft_L = nn.Parameter(torch.empty(r_out, n_elements))
            self.oft_R = nn.Parameter(torch.empty(r_in, n_elements))
            rows, cols = torch.triu_indices(oft_block_size, oft_block_size, 1)
            self.register_buffer('rows', rows)
            self.register_buffer('cols', cols)
            self.register_buffer('perm_in', torch.randperm(in_features))
            self.register_buffer('inv_perm_in', torch.argsort(self.perm_in))
            self.register_buffer('perm_out', torch.randperm(out_features))
            self.register_buffer('inv_perm_out', torch.argsort(self.perm_out))
            self.oft_block_size = oft_block_size
            self.in_features = in_features
            self.out_features = out_features
            self.reset_parameters()
            self.update_permutation()

        # opts = {"triton.cudagraphs": False}
        # self._compiled_train = torch.compile(self._forward_core, options=opts)
        # self._compiled_eval = torch.compile(self._forward_core, options=opts)
        self._compiled_train = self._forward_core
        self._compiled_eval = self._forward_core

    def update_permutation(self):
        """Update the permutation of the indices."""
        with torch.no_grad():
            device = self.linear.weight.device
            perm_in = torch.randperm(self.in_features, device=device)
            self.perm_in.copy_(perm_in)
            self.inv_perm_in.copy_(torch.argsort(perm_in))
            perm_out = torch.randperm(self.out_features, device=device)
            self.perm_out.copy_(perm_out)
            self.inv_perm_out.copy_(torch.argsort(perm_out))

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "oft_R"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.oft_R)
            nn.init.zeros_(self.oft_L)

    def get_oft_LR(self) -> torch.Tensor:
        """Return the orthogonal oft_R matrices from the skew-symmetric upper triangular vector."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        return self.oft_R.get_weight()

    def merge(self) -> None:
        """Merges the OFT weights into the full-rank weights (W = W + delta_W)."""
        if self.oft_block_size > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            oft_data = self.get_oft_R()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(weight_data, requires_grad=False, **weight.__dict__)
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(device=self.linear.weight.data.device)
            self.merged = True

    @torch.compile(dynamic=True, options={"triton.cudagraphs": False})
    def _forward_core(
            self, 
            x: torch.Tensor, 
            Rl: torch.Tensor,
            Rr: torch.Tensor,
            block_size: int,
            rows: torch.Tensor,
            cols: torch.Tensor,
            perm_in: torch.Tensor, 
            inv_perm_in: torch.Tensor,
            perm_out: torch.Tensor,
            inv_perm_out: torch.Tensor,
            base_weight: torch.Tensor,
            base_bias: torch.Tensor,
        ) -> torch.Tensor:

        R_left, R_right = get_weight_poet(Rl, Rr, block_size, rows, cols)  

        # x = x.index_select(-1, perm_in).contiguous()
        # x = PermutationFunction.apply(x, perm_in, inv_perm_in)
        x = permute_apply(x, perm_in, inv_perm_in)
        x_rot = torch_bmm(x, R_right, block_size)
        # y = F.linear(x_rot, self.linear.weight, self.linear.bias)
        # y = FrozenLinearFn.apply(x_rot, self.linear.weight, self.linear.bias)
        y = frozen_linear(x_rot, base_weight, base_bias)
        y_rot = torch_bmm(y, R_left, block_size)
        # y = y_rot.index_select(-1, inv_perm_out)
        # y = PermutationFunction.apply(y_rot, inv_perm_out, perm_out)
        y = permute_apply(y_rot, inv_perm_out, perm_out)
        return y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.oft_block_size == 0 or self.merged:
            return self.linear(x)

        if torch.is_grad_enabled():
            y = self._compiled_train(
                x, 
                self.oft_L, 
                self.oft_R, 
                self.oft_block_size,
                self.rows,
                self.cols,
                self.perm_in, 
                self.inv_perm_in, 
                self.perm_out, 
                self.inv_perm_out,
                self.linear.weight, self.linear.bias
            )
            return y # .clone()
        else:
            with torch.no_grad():
                return self._compiled_eval(
                    x, 
                    self.oft_L, 
                    self.oft_R, 
                    self.oft_block_size,
                    self.rows,
                    self.cols,
                    self.perm_in, 
                    self.inv_perm_in, 
                    self.perm_out, 
                    self.inv_perm_out,
                    self.linear.weight, self.linear.bias
                )




class OFTQKVLinear(OFTLinear):
    # OFT implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for OFT
        head_size: int,
        n_head: int,
        n_query_groups: int,
        oft_block_size: int = 0,
        oft_dropout: float = 0.0,
        enable_oft: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs: Any,
    ):
        """OFT wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. OFT R matrix as `self.oft_R`
        Only OFT's R matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            head_size: size of a single attention head
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `litgpt/config.py`)
            oft_block_size: rank of the weight update matrices. To make sense of using OFT the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            oft_dropout: dropout that is applied on the input in the OFT branch (before multiplying by matrix A)
            enable_oft: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply OFT we can set it as False. For example if we want to apply OFT only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(OFTLinear, self).__init__(oft_block_size=oft_block_size, oft_dropout=oft_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.head_size = head_size
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_oft, bool):
            enable_oft = [enable_oft] * 3
        assert len(enable_oft) == 3
        self.enable_oft = enable_oft

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ oft_block_size: 32
        # ⚬ enable_oft: [True, False, True]
        if oft_block_size > 0 and any(enable_oft):
            r_in = in_features // oft_block_size
            r_out = out_features // oft_block_size
            n_elements = oft_block_size * (oft_block_size - 1) // 2
            # self.oft_L = nn.Parameter(torch.empty(r_out, n_elements)) #, device=self.linear.weight.device, dtype=self.linear.weight.dtype))
            # self.oft_R = nn.Parameter(torch.empty(r_in, n_elements)) # , device=self.linear.weight.device, dtype=self.linear.weight.dtype))
            self.oft_L = nn.Parameter(torch.empty(r_out, n_elements))
            self.oft_R = nn.Parameter(torch.empty(r_in, n_elements))
            rows, cols = torch.triu_indices(oft_block_size, oft_block_size, 1)
            self.register_buffer('rows', rows)
            self.register_buffer('cols', cols)
            self.register_buffer('perm_in', torch.randperm(in_features))
            self.register_buffer('inv_perm_in', torch.argsort(self.perm_in))
            self.register_buffer('perm_out', torch.randperm(out_features))
            self.register_buffer('inv_perm_out', torch.argsort(self.perm_out))
            self.oft_block_size = oft_block_size
            self.in_features = in_features
            self.out_features = out_features
            self.update_permutation()

            # self.oft_A = nn.Parameter(torch.empty((oft_block_size * sum(enable_oft), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_oft
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
                # might not be equal to `head_size * n_head`, thus we use it directly here
                head_size * n_head * enable_q,
                head_size * n_query_groups * enable_k,
                head_size * n_query_groups * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            # self.lora_B = nn.Parameter(torch.empty(sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora

            self.reset_parameters()

            # opts = {"triton.cudagraphs": False}
            # self._compiled_train = torch.compile(self._forward_core, options=opts)
            # self._compiled_eval = torch.compile(self._forward_core, options=opts)
            self._compiled_train = self._forward_core
            self._compiled_eval = self._forward_core

    @property
    def oft_ind(self) -> torch.Tensor:
        """Lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used."""
        # Indices are needed to properly pad weight updates with zeros.
        if not hasattr(self, "_oft_ind"):
            enable_q, enable_k, enable_v = self.enable_oft
            kv_embd_size = self.linear.in_features // (self.n_head // self.n_query_groups)
            oft_ind = []
            if enable_q:
                oft_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                oft_ind.extend(range(self.linear.in_features, self.linear.in_features + kv_embd_size))
            if enable_v:
                oft_ind.extend(range(self.linear.in_features + kv_embd_size, self.linear.out_features))
            self.register_buffer(
                "_oft_ind", torch.tensor(oft_ind, device=self.linear.weight.device), persistent=False
            )

        return self._oft_ind

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad the last dimension of weight updates with zeros.

        If, based on `self.enable_oft`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_oft):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_oft: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_oft only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.oft_ind comes in handy)

        result = x.new_zeros(*x.shape[:-1], self.linear.out_features)  # (64, 64, 384)
        if result.device.type == "mps":
            result[..., self.oft_ind] = x
            return result
        else:
            return result.index_copy_(dim=-1, index=self.oft_ind, source=x)  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `litgpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be split in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            return F.conv1d(input, weight, groups=sum(self.enable_oft))  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_oft)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_oft), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)],
            dim=1,  # (B, C_output', T)
        )  # (B, C_output, T)

    def get_oft_R(self) -> torch.Tensor:
        """Return the orthogonal oft_R matrices from the skew-symmetric upper triangular vector."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        return self.oft_R.get_weight()

    def merge(self) -> None:
        """Merges the OFT weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and any(self.enable_oft) and not self.merged:
            super().merge()
    
    @torch.compile(dynamic=True, options={"triton.cudagraphs": False})
    def _forward_core(
            self, 
            x: torch.Tensor, 
            Rl: torch.Tensor,
            Rr: torch.Tensor,
            block_size: int,
            rows: torch.Tensor,
            cols: torch.Tensor,
            perm_in: torch.Tensor, 
            inv_perm_in: torch.Tensor,
            perm_out: torch.Tensor,
            inv_perm_out: torch.Tensor,
            base_weight: torch.Tensor,
            base_bias: torch.Tensor,
        ) -> torch.Tensor:

        R_left, R_right = get_weight_poet(Rl, Rr, block_size, rows, cols) 

        # x = x.index_select(-1, perm_in).contiguous()
        x = PermutationFunction.apply(x, perm_in, inv_perm_in)
        # x = permute_apply(x, perm_in, inv_perm_in)
        x_rot = torch_bmm(x, R_right, block_size)
        # y = F.linear(x_rot, self.linear.weight, self.linear.bias)
        # y = FrozenLinearFn.apply(x_rot, self.linear.weight, self.linear.bias)
        y = frozen_linear(x_rot, base_weight, base_bias)
        y_rot = torch_bmm(y, R_left, block_size)
        # y = y_rot.index_select(-1, inv_perm_out)
        y = PermutationFunction.apply(y_rot, inv_perm_out, perm_out)
        # y = permute_apply(y_rot, inv_perm_out, perm_out)
        return y_rot
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.oft_block_size == 0 or self.merged or not any(self.enable_oft):
            return self.linear(x)
        
        if torch.is_grad_enabled():
            y = self._compiled_train(
                x, 
                self.oft_L, 
                self.oft_R, 
                self.oft_block_size,
                self.rows,
                self.cols,
                self.perm_in, 
                self.inv_perm_in, 
                self.perm_out, 
                self.inv_perm_out,
                self.linear.weight, self.linear.bias
            )
            return y # .clone()
        else:
            with torch.no_grad():
                return self._compiled_eval(
                    x, 
                    self.oft_L, 
                    self.oft_R, 
                    self.oft_block_size,
                    self.rows,
                    self.cols,
                    self.perm_in, 
                    self.inv_perm_in, 
                    self.perm_out, 
                    self.inv_perm_out,
                    self.linear.weight, self.linear.bias
                )


def mark_only_oft_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except OFT's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with OFT layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"oft_only"``: only bias weight for OFT layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "oft_only", "all"]
    """
    # freeze all layers except OFT's
    for n, p in model.named_parameters():
        if "oft_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "oft_only":
        for m in model.modules():
            if isinstance(m, OFTLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def oft_filter(key: str, value: Any) -> bool:
    return "oft_" in key


@dataclass
class Config(BaseConfig):
    """
    Args:
        oft_block_size: rank of the weight update matrices. To make sense of using OFT the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        oft_dropout: dropout that is applied on the input in the OFT branch (before multiplying by matrix A)
        oft_*: whether to apply OFT to the specified weights or not
    """

    oft_block_size: int = 0
    oft_alpha: int = 1
    oft_dropout: float = 0.0
    oft_query: bool = False
    oft_key: bool = False
    oft_value: bool = False
    oft_projection: bool = False
    oft_mlp: bool = False
    oft_head: bool = False

    @property
    def mlp_class(self) -> Type:
        return getattr(litgpt.oft, self.mlp_class_name)


class GPT(BaseModel):
    # Copy & paste from :class:`model.GPT`. Note that :class:`Block` is new here.
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = create_oft_linear(
            config,
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            use_r=config.oft_head,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.max_seq_length = self.config.block_size

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, OFTLinear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight", "lm_head.bias": "lm_head.linear.bias"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config, block_idx)
        self.attn = CausalSelfAttention(config, block_idx)
        self.mlp = config.mlp_class(config)


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config, block_idx)
        # key, query, value projections for all heads, but in a batch
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.qkv = OFTQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            oft_block_size=config.oft_block_size,
            oft_dropout=config.oft_dropout,
            enable_oft=(config.oft_query, config.oft_key, config.oft_value),
            bias=config.bias or config.attn_bias,
            # for MQA/GQA support
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.proj = create_oft_linear(
            config,
            config.head_size * config.n_head,
            config.n_embd,
            use_r=config.oft_projection,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base and/or legacy checkpoints."""
        mapping = {
            "qkv.weight": "qkv.linear.weight",
            "qkv.bias": "qkv.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.linear.{attr}"
            current_key = f"{prefix}qkv.linear.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def create_oft_linear(
    config: Config,
    in_size: int,
    out_size: int,
    bias: Optional[Union[float, bool]] = None,
    use_r: Optional[bool] = None,
) -> OFTLinear:
    if bias is None:
        bias = config.bias
    if use_r is None:
        use_r = config.oft_mlp
    return OFTLinear(
        in_size,
        out_size,
        bias=bias,
        oft_block_size=(config.oft_block_size if use_r else 0),
        oft_dropout=config.oft_dropout,
    )


class GptNeoxMLP(litgpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = create_oft_linear(config, config.n_embd, config.intermediate_size)
        self.proj = create_oft_linear(config, config.intermediate_size, config.n_embd)
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(litgpt.model.LLaMAMLP):
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        nn.Module.__init__(self)
        self.intermediate_size = intermediate_size or config.intermediate_size
        # self.fc_1 = create_oft_linear(config, config.n_embd, self.intermediate_size)
        # self.fc_1 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        # self.fc_2 = create_oft_linear(config, config.n_embd, self.intermediate_size)
        # self.fc_2 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        # self.fc = nn.Linear(config.n_embd, self.intermediate_size * 2, bias=config.bias)
        self.fc = create_oft_linear(config, config.n_embd, self.intermediate_size * 2)
        self.proj = create_oft_linear(config, self.intermediate_size, config.n_embd)
        # self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(litgpt.model.LLaMAMoE):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.gate = create_oft_linear(config, config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"gate.weight": "gate.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_oft_weights(model: GPT) -> None:
    """Merge OFT weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, OFTLinear):
            module.merge()
